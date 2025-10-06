import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.sparse import coo_matrix

from bag_attention_model import BagAttentionLayer


class MessagePassingLayer(nn.Module):
    """
    Graph Neural Network message passing layer with two types of edges:
    1. Solid yellow edges: intra-type connections (LF-to-LF, SCC-to-SCC)
    2. Dashed blue edges: inter-type connections (LF-to-SCC, SCC-to-LF)
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Weight matrices for different edge types
        self.W1 = nn.Linear(embedding_dim, embedding_dim)  # Intra-type (yellow edges)
        self.W2 = nn.Linear(embedding_dim, embedding_dim)  # Inter-type LF->SCC (blue edges)
        self.W3 = nn.Linear(embedding_dim, embedding_dim)  # Inter-type SCC->LF (blue edges)
        self.W4 = nn.Linear(embedding_dim, embedding_dim)  # Self-connection
        
        # Multi-head attention for global context
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Aggregation function (linear layer with ReLU)
        self.aggregation = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(
        self, 
        node_embeddings: torch.Tensor,  # (B, N, D) - all nodes (LF tokens + SCC nodes)
        node_types: torch.Tensor,       # (B, N) - 0 for LF tokens, 1 for SCC nodes
        adjacency_intra: torch.Tensor,  # (B, N, N) - intra-type adjacency (yellow edges)
        adjacency_inter: torch.Tensor,  # (B, N, N) - inter-type adjacency (blue edges)
        node_mask: torch.Tensor         # (B, N) - mask for valid nodes
    ) -> torch.Tensor:
        """
        Perform message passing with two types of edges.
        
        Args:
            node_embeddings: Node embeddings (B, N, D)
            node_types: Node types - 0 for LF tokens, 1 for SCC nodes (B, N)
            adjacency_intra: Intra-type adjacency matrix (B, N, N)
            adjacency_inter: Inter-type adjacency matrix (B, N, N)
            node_mask: Valid node mask (B, N)
            
        Returns:
            Updated node embeddings (B, N, D)
        """
        B, N, D = node_embeddings.shape
        
        # 1. Intra-type message passing (yellow edges)
        intra_messages = torch.bmm(adjacency_intra, self.W1(node_embeddings))  # (B, N, D)
        
        # 2. Inter-type message passing (blue edges)
        inter_messages = torch.bmm(adjacency_inter, self.W2(node_embeddings))  # (B, N, D)
        
        # 3. Self-connection
        self_messages = self.W4(node_embeddings)  # (B, N, D)
        
        # 4. Global context via attention (following the paper's mention of global context)
        # Reshape for attention: (B*N, 1, D) -> (B*N, 1, D)
        flat_embeddings = node_embeddings.view(B * N, 1, D)
        flat_mask = node_mask.view(B * N)
        
        # Apply attention only to valid nodes
        valid_indices = torch.where(flat_mask)[0]
        if len(valid_indices) > 0:
            valid_embeddings = flat_embeddings[valid_indices]  # (num_valid, 1, D)
            global_context, _ = self.global_attention(
                valid_embeddings, valid_embeddings, valid_embeddings
            )  # (num_valid, 1, D)
            
            # Scatter back to original positions
            global_messages = torch.zeros_like(flat_embeddings)
            global_messages[valid_indices] = global_context
            global_messages = global_messages.view(B, N, D)
        else:
            global_messages = torch.zeros_like(node_embeddings)
        
        # 5. Aggregate all messages
        # Average pooling with normalization
        degree_intra = adjacency_intra.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N, 1)
        degree_inter = adjacency_inter.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N, 1)
        
        normalized_intra = intra_messages / degree_intra
        normalized_inter = inter_messages / degree_inter
        
        # Combine all message types
        combined_messages = (
            normalized_intra + 
            normalized_inter + 
            self_messages + 
            global_messages / N  # Normalize global context
        )
        
        # Apply aggregation function
        updated_embeddings = self.aggregation(combined_messages)
        
        # Apply mask to zero out invalid nodes
        updated_embeddings = updated_embeddings * node_mask.unsqueeze(-1)
        
        return updated_embeddings


class BagAttentionGNN(nn.Module):
    """
    Hybrid model combining bag attention mechanism with GNN message passing.
    
    Architecture:
    1. Input embedding of LF outputs
    2. Create virtual SCC nodes
    3. Apply bag attention within each SCC
    4. Perform GNN message passing between SCCs and LF tokens
    5. Final classification
    """
    
    def __init__(
        self, 
        max_lf_id: int = 0, 
        embedding_dim: int = 16, 
        num_gnn_layers: int = 2,
        num_attention_heads: int = 4,
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
        
        # SCC node embedding (learnable embeddings for virtual SCC nodes)
        self.scc_embed = nn.Parameter(torch.randn(embedding_dim) * 0.02)
        
        # GNN message passing layers
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(embedding_dim, num_attention_heads)
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
        
    def _create_adjacency_matrices(
        self, 
        index: torch.Tensor, 
        node_types: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create adjacency matrices for intra-type and inter-type connections.
        
        Args:
            index: (B, S, 2) - LF token indices
            node_types: (B, N) - node types (0=LF, 1=SCC)
            num_nodes: Total number of nodes per batch
            
        Returns:
            adjacency_intra: (B, N, N) - intra-type connections
            adjacency_inter: (B, N, N) - inter-type connections
        """
        B, S = index.shape[:2]
        device = index.device
        
        adjacency_intra = torch.zeros(B, num_nodes, num_nodes, device=device)
        adjacency_inter = torch.zeros(B, num_nodes, num_nodes, device=device)
        
        for b in range(B):
            example_ids = index[b, :, 0]  # (S,)
            
            # Create intra-type connections (LF tokens within same SCC)
            for scc_id in torch.unique(example_ids):
                # Find LF tokens belonging to this SCC
                lf_mask = (example_ids == scc_id)
                lf_indices = torch.where(lf_mask)[0]
                
                # Connect all LF tokens within this SCC (yellow edges)
                for i in lf_indices:
                    for j in lf_indices:
                        if i != j:  # No self-loops for intra-type
                            adjacency_intra[b, i, j] = 1.0
                
                # Find corresponding SCC node index
                scc_node_idx = S + scc_id.item()  # SCC nodes start after LF tokens
                
                # Create inter-type connections (blue edges)
                # LF tokens <-> SCC node
                for lf_idx in lf_indices:
                    adjacency_inter[b, lf_idx, scc_node_idx] = 1.0  # LF -> SCC
                    adjacency_inter[b, scc_node_idx, lf_idx] = 1.0  # SCC -> LF
        
        return adjacency_intra, adjacency_inter
    
    def forward(self, index: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass combining bag attention and GNN message passing.
        
        Args:
            index: (B, S, 2) - [example_id, lf_id]
            value: (B, S) - LF outputs
            
        Returns:
            preds_padded: (B, E_max) - predictions per example
            node_embeddings: (B, N, D) - final node embeddings
            aux: auxiliary information
        """
        B, S = value.shape
        device = value.device
        
        # 1. Embed LF outputs
        lf_embeddings = self.input_embed(value.float().unsqueeze(-1))  # (B, S, D)
        
        # 2. Apply bag attention to get refined LF embeddings
        _, refined_lf_embeddings, bag_aux = self.bag_attention(index, lf_embeddings)  # (B, S, D)
        
        # 3. Create virtual SCC nodes
        example_ids = index[:, :, 0]  # (B, S)
        max_scc_per_batch = []
        
        for b in range(B):
            unique_sccs = torch.unique(example_ids[b])
            max_scc_per_batch.append(len(unique_sccs))
        
        max_sccs = max(max_scc_per_batch)
        
        # Create SCC node embeddings
        scc_embeddings = self.scc_embed.unsqueeze(0).unsqueeze(0).expand(B, max_sccs, -1)  # (B, max_sccs, D)
        
        # 4. Combine LF and SCC nodes
        total_nodes = S + max_sccs
        node_embeddings = torch.cat([refined_lf_embeddings, scc_embeddings], dim=1)  # (B, S+max_sccs, D)
        
        # Create node type indicators (0=LF, 1=SCC)
        lf_types = torch.zeros(B, S, device=device)
        scc_types = torch.ones(B, max_sccs, device=device)
        node_types = torch.cat([lf_types, scc_types], dim=1)  # (B, S+max_sccs)
        
        # Create node mask
        node_mask = torch.ones(B, S, device=device)  # All LF tokens are valid
        scc_mask = torch.zeros(B, max_sccs, device=device)
        for b, num_sccs in enumerate(max_scc_per_batch):
            scc_mask[b, :num_sccs] = 1.0
        node_mask = torch.cat([node_mask, scc_mask], dim=1)  # (B, S+max_sccs)
        
        # 5. Create adjacency matrices
        adjacency_intra, adjacency_inter = self._create_adjacency_matrices(
            index, node_types, total_nodes
        )
        
        # 6. Apply GNN message passing layers
        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(
                node_embeddings, node_types, adjacency_intra, adjacency_inter, node_mask
            )
        
        # 7. Extract SCC embeddings and classify
        scc_embeddings_final = node_embeddings[:, S:, :]  # (B, max_sccs, D)
        
        # Apply classification to each SCC
        scc_logits = []
        for b in range(B):
            num_sccs = max_scc_per_batch[b]
            if num_sccs > 0:
                batch_scc_embeds = scc_embeddings_final[b, :num_sccs, :]  # (num_sccs, D)
                batch_logits = self.classify(batch_scc_embeds).squeeze(-1)  # (num_sccs,)
                scc_logits.append(batch_logits)
            else:
                scc_logits.append(torch.empty(0, device=device))
        
        # 8. Pad predictions
        E_max = max(len(logits) for logits in scc_logits) if scc_logits else 0
        preds_padded = torch.zeros(B, E_max, device=device)
        example_mask = torch.zeros(B, E_max, dtype=torch.bool, device=device)
        
        for b, logits in enumerate(scc_logits):
            L = len(logits)
            if L > 0:
                preds_padded[b, :L] = logits
                example_mask[b, :L] = True
        
        aux = {
            "example_mask": example_mask,
            "node_embeddings": node_embeddings,
            "adjacency_intra": adjacency_intra,
            "adjacency_inter": adjacency_inter,
            "bag_aux": bag_aux
        }
        
        return preds_padded, node_embeddings, aux


class StackedBagAttentionGNN(nn.Module):
    """
    Stacked version of BagAttentionGNN for deeper architectures.
    """
    
    def __init__(
        self, 
        max_lf_id: int = 0, 
        embedding_dim: int = 16, 
        num_layers: int = 2,
        num_gnn_layers: int = 2,
        num_attention_heads: int = 4,
        use_lf_reliability: bool = False
    ):
        super().__init__()
        self.input_embed = nn.Linear(1, embedding_dim)
        
        self.layers = nn.ModuleList([
            BagAttentionGNN(
                max_lf_id=max_lf_id,
                embedding_dim=embedding_dim,
                num_gnn_layers=num_gnn_layers,
                num_attention_heads=num_attention_heads,
                use_lf_reliability=use_lf_reliability
            )
            for _ in range(num_layers)
        ])
        
        self.final_classify = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, 1),
            nn.Sigmoid()
        )

        print("############## Using StackedBagAttentionGNN ##############")
    
    def forward(self, index: torch.Tensor, value: torch.Tensor):
        """
        Forward pass through stacked BagAttentionGNN layers.
        """
        current_value = value
        
        for layer in self.layers:
            preds, node_embeds, aux = layer(index, current_value)
            # Use node embeddings as input for next layer (only LF tokens)
            S = current_value.shape[1]
            current_value = node_embeds[:, :S, :].mean(dim=-1)  # (B, S)
        
        return preds, node_embeds, aux


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
        num_attention_heads: int = 4,
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
            num_attention_heads=num_attention_heads,
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
