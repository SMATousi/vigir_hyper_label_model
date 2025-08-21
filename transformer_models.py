import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
import torch.optim as optim
import math

from loss import BCEMask, BCEMaskWeighted

def sparse_mean(index, value, expand=True):
    """obtain the mean of values that share the same index (e.g. same row or column in label matrix X) 

    Args:
        index: The indices of the elements, the first dimension is batch size 
        value : The values of the elements, the first dimension is batch size
        expand (bool, optional): if true expand the output to have the same size as index

    Returns:
        mean values
    """
    output_batch = []
    ind_max = int(index.max() + 1)
    for i_batch in range(value.shape[0]):
        output = torch.zeros((ind_max, value.shape[2])).float().to(value.device).index_add_(0,
                                                                                    index[i_batch],
                                                                                    value[i_batch])
        norm = torch.zeros(ind_max).to(value.device).float().index_add_(
            0, index[i_batch], torch.ones_like(index[i_batch]).float()) + 1e-9
        output = output / norm[:, None].float()
        if expand:
            output = torch.index_select(output, 0, index[i_batch])
        output_batch.append(output)
    return torch.stack(output_batch)

class GraphTransformerLayer(nn.Module):
    """One Graph Transformer layer using self-attention"""
    def __init__(self, in_features, out_features, n_heads=4):
        super(GraphTransformerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        assert self.head_dim * n_heads == self.out_features, "out_features must be divisible by n_heads"

        self.q_linear = nn.Linear(in_features, out_features)
        self.k_linear = nn.Linear(in_features, out_features)
        self.v_linear = nn.Linear(in_features, out_features)

        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.LeakyReLU(),
            nn.Linear(out_features * 2, out_features)
        )

        self.layer_norm1 = nn.LayerNorm(out_features)
        self.layer_norm2 = nn.LayerNorm(out_features)

    def forward(self, index, value):
        # value shape: (batch, num_elements, in_features)
        
        # Self-attention part
        q = self.q_linear(value).view(value.shape[0], value.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(value).view(value.shape[0], value.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(value.shape[0], value.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, v).transpose(1, 2).contiguous().view(value.shape[0], value.shape[1], self.out_features)

        # Add & Norm
        out1 = self.layer_norm1(value + context) # Using value as residual, assuming in_features==out_features

        # Feed-forward part
        ffn_output = self.ffn(out1)

        # Add & Norm
        out2 = self.layer_norm2(out1 + ffn_output)

        return index, out2

class SequentialMultiArg(nn.Sequential):
    """helper class to stack multiple GraphTransformerLayer"""
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class LELATransformer(nn.Module):
    """The model architecture of LELA using Graph Transformers"""
    def __init__(self):
        super(LELATransformer, self).__init__()
        embed_dim = 16 # Reduced from 32
        n_heads = 2 # Reduced from 4
        self.input_embed = nn.Linear(1, embed_dim)
        self.matrix_net = SequentialMultiArg(
            GraphTransformerLayer(embed_dim, embed_dim, n_heads=n_heads),
        ) # Reduced from 2 layers to 1
        col_embed_mixed_size = embed_dim
        self.classify = nn.Sequential(
            nn.Linear(col_embed_mixed_size, col_embed_mixed_size),
            nn.LeakyReLU(),
            nn.Linear(col_embed_mixed_size, 1),
            nn.Sigmoid()
        ) # Simplified classifier

    def forward(self, index, value):
        embedded_value = self.input_embed(value.float().unsqueeze(2))
        _, elementwise_embed_sparse = self.matrix_net(index, embedded_value)
        example_embed = sparse_mean(
            index[:, :, 0], elementwise_embed_sparse, expand=False)
        pred = self.classify(example_embed).squeeze(-1)
        return pred, elementwise_embed_sparse

class LELATransformerWrapper:
    """Wrapper for the trained LELA Transformer model"""
    def __init__(self, checkpoint_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.net = LELATransformer()
        if checkpoint_path is not None:
            self.checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device(self.device)
            )
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)

    def predict_prob(self, X):
        preds = []
        n_class = int(np.max(X[X!=-1]) + 1)
        if n_class == 2:
            X_binary = np.zeros_like(X) - 1
            X_binary[X == 1] = 1
            X_binary[X == -1] = 0
            pred = self._predict_binary(X_binary)
            preds = np.hstack([1 - pred[:, np.newaxis], pred[:, np.newaxis]])
        else:
            for label in range(n_class):
                X_binary = np.zeros_like(X) - 1
                X_binary[X == label] = 1
                X_binary[X == -1] = 0
                pred = self._predict_binary(X_binary)
                preds.append(pred[:, np.newaxis])
            preds = np.hstack(preds)

        tie_score = 1 / preds.shape[1] - 1e-9
        preds[preds < tie_score] = (preds[preds < tie_score] - tie_score) * tie_score / (tie_score - np.min(preds[preds < tie_score])) + tie_score
        preds[preds > 1] = 1
        preds[preds < 0] = 0
        preds = preds / np.sum(preds, axis=1, keepdims=True)
        return preds

    def _predict_binary(self, X):
        self.net.eval()
        X_sparse = coo_matrix(np.squeeze(X))
        index = np.array([X_sparse.row, X_sparse.col]).T
        value = X_sparse.data
        index = torch.from_numpy(index).to(self.device).unsqueeze(0)
        value = torch.from_numpy(value).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            pred, _ = self.net(index, value)
        return pred.squeeze(0).cpu().numpy()


class LELASemisupervisedHelper:
    """helper class to perform semisupervised label aggregation"""
    def __init__(self, checkpoint_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.net = LELATransformer()
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=0.0001,
            amsgrad=True,
        )
        if checkpoint_path is not None:
            self.checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device(self.device)
            )

    def initialize_net(self):
        """initialize model as pretrained LELA"""
        if hasattr(self, 'checkpoint'):
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.net.to(self.device)
        self.optimizer.zero_grad()

    def fit_predict(self, X, y_partial, y_indices, weights=None):
        """finetune LELA model by miminizing the loss on the provided labels"""
        self.initialize_net()
        self.net.train()
        X_sparse = coo_matrix(np.squeeze(X))
        index = np.array([X_sparse.row, X_sparse.col]).T
        value = X_sparse.data
        index = torch.from_numpy(index).to(self.device).unsqueeze(0)
        value = torch.from_numpy(value).float().to(self.device).unsqueeze(0)
        y_complete = np.zeros(X.shape[0])
        y_complete[y_indices] = y_partial
        y_complete = torch.from_numpy(y_complete).float().to(self.device).unsqueeze(0)
        if weights:
            self.criterion = BCEMaskWeighted(weights)
        else:
            self.criterion = BCEMask()
        mask = np.zeros(X.shape[0])
        mask[y_indices] = 1
        mask = mask.astype(bool)
        mask = torch.from_numpy(mask).unsqueeze(0)
        for i in range(int(np.sqrt(len(y_partial)))):
            pred, _ = self.net(index, value)
            loss = self.criterion(pred.unsqueeze(0), y_complete, mask)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.net.eval()
        pred, _ = self.net(index, value)
        return pred
