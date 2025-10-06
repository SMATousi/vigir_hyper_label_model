import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.sparse import coo_matrix

from bag_attention_model import BagAttentionLayer


def sparse_mean(index, value, expand=True):
    """Sparse mean pooling operation (gradient-safe version)
    
    Args:
        index: The indices of the elements, the first dimension is batch size 
        value: The values of the elements, the first dimension is batch size
        expand (bool, optional): if true expand the output to have the same size as index

    Returns:
        mean values
    """
    output_batch = []
    ind_max = int(index.max() + 1)
    for i_batch in range(value.shape[0]):
        # Use non-in-place operations to avoid gradient issues
        output_base = torch.zeros((ind_max, value.shape[2])).float().to(value.device)
        output = output_base.index_add(0, index[i_batch], value[i_batch])
        
        norm_base = torch.zeros(ind_max).to(value.device).float()
        norm = norm_base.index_add(0, index[i_batch], torch.ones_like(index[i_batch]).float()) + 1e-9
        
        output = output / norm[:, None].float()
        if expand:
            output = torch.index_select(output, 0, index[i_batch])
        output_batch.append(output)
    return torch.stack(output_batch)


class SparseGNNLayer(nn.Module):
    """
    Sparse GNN layer similar to model.py but enhanced for bag attention + GNN hybrid.
    Uses sparse operations to avoid memory explosion with large datasets.
    
    Implements message passing with:
    1. Row pooling (intra-SCC communication)
    2. Column pooling (intra-LF communication) 
    3. Global pooling (global context)
    4. Self-connection
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrices following model.py pattern
        self.linear_row = nn.Linear(in_features, in_features)    # W1: Row pooling (SCC-level)
        self.linear_col = nn.Linear(in_features, in_features)    # W2: Column pooling (LF-level)
        self.linear_global = nn.Linear(in_features, in_features) # W3: Global context
        self.linear_self = nn.Linear(in_features, in_features)   # W4: Self-connection
        
        # Final aggregation (combines all 4 message types)
        self.linear_combine = nn.Linear(in_features * 4, out_features)
        self.activation = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, index: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse GNN forward pass similar to model.py but with bag attention enhancements.
        
        Args:
            index: (B, S, 2) - [example_id, lf_id] indices
            value: (B, S, D) - token embeddings
            
        Returns:
            index: (B, S, 2) - unchanged indices
            updated_embeddings: (B, S, D) - updated token embeddings
        """
        # Sparse pooling operations (no dense adjacency matrices!)
        row_pooled = self.linear_row(sparse_mean(index[:, :, 0], value))      # Pool by example (SCC)
        col_pooled = self.linear_col(sparse_mean(index[:, :, 1], value))      # Pool by LF
        global_pooled = self.linear_global(torch.mean(value, dim=1)).unsqueeze(1).expand_as(value)  # Global context
        self_connection = self.linear_self(value)  # Self-connection
        
        # Combine all message types
        combined = torch.cat([
            self_connection,
            row_pooled, 
            col_pooled,
            global_pooled
        ], dim=2)  # (B, S, 4*D)
        
        # Final transformation
        output = self.activation(self.linear_combine(combined))  # (B, S, out_features)
        output = self.layer_norm(output)
        
        return index, output


class BagAttentionGNN(nn.Module):
    """
    Hybrid model combining bag attention mechanism with sparse GNN message passing.
    
    Architecture:
    1. Input embedding of LF outputs
    2. Apply bag attention within each SCC
    3. Apply sparse GNN layers for inter-SCC communication
    4. Final classification on example-level embeddings
    
    Memory efficient: Uses sparse operations, no dense adjacency matrices.
    """
    
    def __init__(
        self, 
        max_lf_id: int = 0, 
        embedding_dim: int = 16, 
        num_gnn_layers: int = 2,
        use_lf_reliability: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Input embedding for LF outputs
        self.input_embed = nn.Linear(1, embedding_dim)
        
        # Bag attention layer (reuse existing implementation)
        self.bag_attention = BagAttentionLayer(
            in_features=embedding_dim,
            out_features=embedding_dim,
            lf_vocab_size=max_lf_id + 1,
            use_lf_reliability=use_lf_reliability
        )
        
        # Sparse GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            SparseGNNLayer(embedding_dim, embedding_dim)
            for _ in range(num_gnn_layers)
        ])
        
        # Final classification head
        self.classify = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, index: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Sparse forward pass combining bag attention and GNN message passing.
        
        Args:
            index: (B, S, 2) - [example_id, lf_id]
            value: (B, S) - LF outputs
            
        Returns:
            preds_padded: (B, E_max) - predictions per example
            token_embeddings: (B, S, D) - final token embeddings
            aux: auxiliary information
        """
        B, S = value.shape
        
        # 1. Embed LF outputs
        lf_embeddings = self.input_embed(value.float().unsqueeze(-1))  # (B, S, D)
        
        # 2. Apply bag attention to get refined LF embeddings
        _, refined_lf_embeddings, bag_aux = self.bag_attention(index, lf_embeddings)  # (B, S, D)
        
        # 3. Apply sparse GNN layers for message passing
        current_embeddings = refined_lf_embeddings
        for gnn_layer in self.gnn_layers:
            _, current_embeddings = gnn_layer(index, current_embeddings)  # (B, S, D)
        
        # 4. Aggregate token embeddings by example to get example-level embeddings
        example_embeddings = sparse_mean(index[:, :, 0], current_embeddings, expand=False)  # (B, E_max, D)
        
        # 5. Apply classification to each example
        example_logits = self.classify(example_embeddings).squeeze(-1)  # (B, E_max)
        
        # 6. Create example mask (all examples from sparse_mean are valid)
        E_max = example_embeddings.shape[1]
        example_mask = torch.ones(B, E_max, dtype=torch.bool, device=example_embeddings.device)
        
        # Handle cases where some batches might have fewer examples
        for b in range(B):
            unique_examples = torch.unique(index[b, :, 0])
            actual_examples = len(unique_examples)
            if actual_examples < E_max:
                example_mask[b, actual_examples:] = False
                example_logits[b, actual_examples:] = 0.0
        
        aux = {
            "example_mask": example_mask,
            "example_embeddings": example_embeddings,
            "bag_aux": bag_aux
        }
        
        return example_logits, current_embeddings, aux


class StackedBagAttentionGNN(nn.Module):
    """
    Stacked version of BagAttentionGNN for deeper architectures.
    Uses sparse operations throughout for memory efficiency.
    """
    
    def __init__(
        self, 
        max_lf_id: int = 0, 
        embedding_dim: int = 16, 
        num_layers: int = 2,
        num_gnn_layers: int = 2,
        use_lf_reliability: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embed = nn.Linear(1, embedding_dim)
        
        # Bag attention layer
        self.bag_attention = BagAttentionLayer(
            in_features=embedding_dim,
            out_features=embedding_dim,
            lf_vocab_size=max_lf_id + 1,
            use_lf_reliability=use_lf_reliability
        )
        
        # Multiple sparse GNN layers
        self.gnn_layers = nn.ModuleList([
            SparseGNNLayer(embedding_dim, embedding_dim)
            for _ in range(num_layers * num_gnn_layers)  # Total GNN layers across all stacks
        ])
        
        # Final classification
        self.classify = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, 1),
            nn.Sigmoid()
        )

        print("############## Using StackedBagAttentionGNN ##############")
    
    def forward(self, index: torch.Tensor, value: torch.Tensor):
        """
        Forward pass through stacked sparse GNN layers.
        """
        B, S = value.shape
        
        # 1. Embed LF outputs
        lf_embeddings = self.input_embed(value.float().unsqueeze(-1))  # (B, S, D)
        
        # 2. Apply bag attention
        _, refined_embeddings, bag_aux = self.bag_attention(index, lf_embeddings)
        
        # 3. Apply all GNN layers sequentially
        current_embeddings = refined_embeddings
        for gnn_layer in self.gnn_layers:
            _, current_embeddings = gnn_layer(index, current_embeddings)
        
        # 4. Final example-level aggregation and classification
        example_embeddings = sparse_mean(index[:, :, 0], current_embeddings, expand=False)
        example_logits = self.classify(example_embeddings).squeeze(-1)
        
        # 5. Create example mask and apply it to logits (avoid in-place operations)
        E_max = example_embeddings.shape[1]
        example_mask = torch.ones(B, E_max, dtype=torch.bool, device=example_embeddings.device)
        
        # Handle cases where some batches might have fewer examples
        for b in range(B):
            unique_examples = torch.unique(index[b, :, 0])
            actual_examples = len(unique_examples)
            if actual_examples < E_max:
                example_mask[b, actual_examples:] = False
        
        # Apply mask to logits without in-place operation
        example_logits = example_logits * example_mask.float()
        
        aux = {
            "example_mask": example_mask,
            "example_embeddings": example_embeddings,
            "bag_aux": bag_aux
        }
        
        return example_logits, current_embeddings, aux


# Wrapper for inference (similar to LELATransformerBagWrapper)
class BagAttentionGNNWrapper:
    """
    Wrapper for trained BagAttentionGNN model for inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        max_lf_id: int,
        embedding_dim: int = 16,
        num_gnn_layers: int = 2,
        use_lf_reliability: bool = True,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        self.net = StackedBagAttentionGNN(
            max_lf_id=max_lf_id,
            embedding_dim=embedding_dim,
            num_layers=2,
            num_gnn_layers=num_gnn_layers,
            use_lf_reliability=use_lf_reliability
        )
        
        # Load checkpoint
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            self.net.load_state_dict(checkpoint['model_state_dict'])
        
        self.net.to(self.device)
        self.net.eval()
    
    def predict(self, X):
        """Return argmax class indices."""
        return np.argmax(self.predict_prob(X), axis=1)
    
    def predict_prob(self, X):
        """
        Predict probabilities for input matrix X.
        Same interface as LELATransformerBagWrapper.
        """
        X = np.asarray(X)
        valid = X != -1
        if not np.any(valid):
            return np.ones((X.shape[0], 1))
        
        n_class = int(np.max(X[valid]) + 1)
        
        if n_class == 2:
            X_binary = np.full_like(X, -1)
            X_binary[X == 1] = 1
            X_binary[X == -1] = 0
            pred = self._predict_binary(X_binary)
            preds = np.hstack([1 - pred[:, np.newaxis], pred[:, np.newaxis]])
        else:
            per_class = []
            for label in range(n_class):
                X_binary = np.full_like(X, -1)
                X_binary[X == label] = 1
                X_binary[X == -1] = 0
                pred = self._predict_binary(X_binary)
                per_class.append(pred[:, np.newaxis])
            preds = np.hstack(per_class)
        
        # Normalization
        tie_score = 1 / preds.shape[1] - 1e-9
        below = preds < tie_score
        if np.any(below):
            preds[below] = (preds[below] - tie_score) * tie_score / (tie_score - np.min(preds[below])) + tie_score
        preds = np.clip(preds, 0, 1)
        preds = preds / np.sum(preds, axis=1, keepdims=True)
        return preds
    
    def _predict_binary(self, X):
        """Binary prediction using sparse matrix conversion."""
        self.net.eval()
        X = np.asarray(X)
        X_sparse = coo_matrix(np.squeeze(X))
        
        # Build index and value tensors
        index = np.stack([X_sparse.row, X_sparse.col], axis=1)
        value = X_sparse.data
        
        index_t = torch.from_numpy(index).to(self.device).unsqueeze(0)
        value_t = torch.from_numpy(value).float().to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            preds_padded, _, aux = self.net(index_t, value_t)
        
        mask = aux["example_mask"][0]
        pred_vec = preds_padded[0, mask].detach().cpu().numpy()
        
        return pred_vec
