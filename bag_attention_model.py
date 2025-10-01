import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
import torch.optim as optim
import math
import torch.nn.functional as F

from loss import BCEMask, BCEMaskWeighted

from scipy.sparse import coo_matrix

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BagAttentionLayer(nn.Module):
    """
    O(NM) bag-token cross-attention:
      - One query per SCC (sample), no LF↔LF attention.
      - Optional per-LF scalar reliability bias on attention logits.
    Shapes:
      index: (B, S, 2) with [row_id, lf_id]
      value: (B, S, Cin)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lf_vocab_size: int,
        use_lf_reliability: bool = True,
    ):
        super().__init__()
        print("############## Using BagAttentionLayer ##############")
        self.in_features = in_features
        self.out_features = out_features
        self.use_lf_reliability = use_lf_reliability

        # projections
        self.input_projection = (
            nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        )
        self.k_linear = nn.Linear(out_features, out_features)
        self.v_linear = nn.Linear(out_features, out_features)
        self.q_bag = nn.Parameter(torch.randn(out_features) * 0.02)  # shared bag query

        # per-LF reliability (logit space). Applied as log-bias in attention logits.
        # Reliability r_j = sigmoid(reliability_logit[j]) in [0,1].
        # self.reliability_logit = nn.Parameter(torch.zeros(lf_vocab_size))

        # post-aggregation refinement (token-wise, FiLM-ish)
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.layer_norm = nn.LayerNorm(out_features)

        # back-projection for API compatibility
        self.output_projection = (
            nn.Linear(out_features, in_features) if in_features != out_features else nn.Identity()
        )

    @torch.no_grad()
    def set_use_lf_reliability(self, flag: bool):
        self.use_lf_reliability = bool(flag)

    def forward(self, index: torch.Tensor, value: torch.Tensor):
        """
        Returns:
          index (unchanged),
          output embeddings shaped like `value` (B,S,Cin),
          plus an auxiliary dict with per-sample bag embeddings (optional use)
        """
        B, S, Cin = value.shape
        C = self.out_features
        x = self.input_projection(value)                # (B,S,C)
        K = self.k_linear(x)                            # (B,S,C)
        V = self.v_linear(x)                            # (B,S,C)

        row_ids = index[:, :, 0]                        # (B,S) sample/SCC id
        lf_ids  = index[:, :, 1].clamp_min(0)           # (B,S) LF id (non-neg)

        out_tokens = torch.empty_like(x)                # (B,S,C)
        bag_embeds = []                                 # collect per-bag embeddings for debug/usage

        # Process each batch independently (S usually large; B typically small)
        for b in range(B):
            gids = row_ids[b]                           # (S,)
            # stable sort to make each SCC contiguous
            # perm = torch.argsort(gids, stable=True)
            perm = torch.argsort(gids)
            invperm = torch.empty_like(perm); invperm[perm] = torch.arange(S, device=perm.device)

            Kb = K[b, perm, :]                          # (S,C)
            Vb = V[b, perm, :]
            lf_b = lf_ids[b, perm]                      # (S,)

            gids_sorted = gids[perm]
            # segment boundaries
            starts = torch.where(
                torch.cat([torch.tensor([True], device=gids.device),
                           gids_sorted[1:] != gids_sorted[:-1]])
            )[0]
            ends = torch.cat([starts[1:], torch.tensor([S], device=gids.device)])
            lengths = (ends - starts)                   # (num_segs,)
            num_segs = lengths.numel()
            maxL = int(lengths.max().item())

            # pack segments to (num_segs, maxL, C) with padding
            K_pack = Kb.new_zeros(num_segs, maxL, C)
            V_pack = Vb.new_zeros(num_segs, maxL, C)
            lf_pack = lf_b.new_zeros(num_segs, maxL)

            for s_idx, (st, en) in enumerate(zip(starts.tolist(), ends.tolist())):
                seg = slice(st, en)
                L = en - st
                K_pack[s_idx, :L, :] = Kb[seg, :]
                V_pack[s_idx, :L, :] = Vb[seg, :]
                lf_pack[s_idx, :L] = lf_b[seg]

            # attention logits for each segment from a single bag query
            q = self.q_bag.view(1, 1, C)                       # (1,1,C) broadcast
            logits = (q * K_pack).sum(dim=-1) / math.sqrt(C)   # (num_segs, maxL)

            if self.use_lf_reliability:
                # add log(r_j) bias per token
                r = torch.sigmoid(self.reliability_logit[lf_pack.clamp(min=0)])  # (num_segs, maxL)
                logits = logits + torch.log(r + 1e-8)

            # mask out padded positions
            valid = torch.arange(maxL, device=lengths.device)[None, :] < lengths[:, None]  # (num_segs,maxL)
            logits = logits.masked_fill(~valid, float("-inf"))

            attn = torch.softmax(logits, dim=-1)               # (num_segs, maxL)
            attn = attn.unsqueeze(-1)                          # (num_segs, maxL, 1)

            # bag embedding per segment: weighted sum of V
            bag = (attn * V_pack).sum(dim=1)                   # (num_segs, C)
            bag_embeds.append(bag)

            # broadcast bag back to tokens (so pooling == identity)
            # out_tokens_perm[t] = LN(x_perm[t] + FFN(bag_of_segment))
            out_perm = torch.empty_like(Kb)                    # (S,C)
            for s_idx, (st, en) in enumerate(zip(starts.tolist(), ends.tolist())):
                L = en - st
                bag_s = bag[s_idx].unsqueeze(0).expand(L, -1)  # (L,C)
                upd = self.ffn(bag_s)
                out_perm[st:en, :] = self.layer_norm(x[b, perm[st:en], :] + upd)

            out_tokens[b, :, :] = out_perm[invperm, :]

        out = self.output_projection(out_tokens)               # (B,S,Cin)
        aux = {"bag_embeds": torch.cat(bag_embeds, dim=0) if bag_embeds else None}
        return index, out, aux


class LELATransformerBag(nn.Module):
    """
    LELA-style model using BagAttentionLayer (O(NM)).
    Compatible with your (index, value) interface and sparse_mean.
    """
    def __init__(self, max_lf_id: int = 0, embedding_dim: int = 16, use_lf_reliability: bool = False):
        super().__init__()
        self.input_embed = nn.Linear(1, embedding_dim)
        self.bag_layer = BagAttentionLayer(
            in_features=embedding_dim,
            out_features=embedding_dim,
            lf_vocab_size=max_lf_id + 1,           # assume LF ids in [0, max_lf_id]
            use_lf_reliability=use_lf_reliability,
        )
        self.classify = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, 1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def set_use_lf_reliability(self, flag: bool):
        self.bag_layer.set_use_lf_reliability(flag)

    def forward(self, index: torch.Tensor, value: torch.Tensor):
        """
        Args:
            index: (B, S, 2)  with index[...,0] = sample/SCC id, index[...,1] = LF id
            value: (B, S)     scalar LF outputs (e.g., {-1,0,1} or scores)

        Returns:
            preds_padded: (B, E_max)  per-example (SCC) probabilities in [0,1], padded on the right
            token_embeds: (B, S, D)   token-level embeddings after bag attention broadcast + FFN
            aux: {
                "example_mask": (B, E_max) bool  True for valid example positions, False for padding
            }
        """
        B, S = value.shape
        # 1) Embed scalar LF outputs to D dims
        embedded_value = self.input_embed(value.float().unsqueeze(-1))   # (B, S, D)

        # 2) Run bag attention (O(NM)): returns token-wise embeddings (broadcast bag + refinement)
        #    We ignore the returned index echo; keep token_embeds and (optional) aux from the layer.
        _, token_embeds, _ = self.bag_layer(index, embedded_value)       # (B, S, D)

        # 3) For each batch item, aggregate tokens by sample/SCC id -> one vector per example
        example_ids = index[:, :, 0]                                     # (B, S)
        per_b_example_vecs = []  # list of tensors with shape (E_b, D)

        for b in range(B):
            gids = example_ids[b]                                        # (S,)
            # stable sort so each SCC's tokens become contiguous
            # perm = torch.argsort(gids, stable=True)
            perm = torch.argsort(gids)
            tb   = token_embeds[b, perm, :]                              # (S, D)
            gids_sorted = gids[perm]

            # segment boundaries for each distinct SCC id
            starts = torch.where(
                torch.cat([torch.tensor([True], device=gids.device),
                        gids_sorted[1:] != gids_sorted[:-1]])
            )[0]
            ends = torch.cat([starts[1:], torch.tensor([S], device=gids.device)])

            # mean-pool tokens within each SCC to get the bag embedding
            ex_vecs = []
            for st, en in zip(starts.tolist(), ends.tolist()):
                ex_vecs.append(tb[st:en, :].mean(dim=0, keepdim=True))   # (1, D)
            per_b_example_vecs.append(torch.cat(ex_vecs, dim=0))         # (E_b, D)

        # 4) Classify each example vector -> logits/probs (E_b,)
        per_b_logits = [self.classify(v).squeeze(-1) for v in per_b_example_vecs]  # list[(E_b,)]

        # 5) Pad to (B, E_max) and build example_mask
        E_max = max(v.numel() for v in per_b_logits) if per_b_logits else 0
        preds_padded = token_embeds.new_zeros((B, E_max))                # (B, E_max), same device/dtype as embeddings
        example_mask = torch.zeros((B, E_max), dtype=torch.bool, device=token_embeds.device)

        for b, vec in enumerate(per_b_logits):
            L = vec.numel()
            preds_padded[b, :L] = vec
            example_mask[b, :L] = True

        aux = {"example_mask": example_mask}
        return preds_padded, token_embeds, aux


class StackedLELATransformerBag(nn.Module):
    def __init__(self, max_lf_id = 0, embedding_dim=16, num_layers=4, use_lf_reliability=False):
        super().__init__()
        self.input_embed = nn.Linear(1, embedding_dim)
        self.layers = nn.ModuleList([
            BagAttentionLayer(
                in_features=embedding_dim,
                out_features=embedding_dim,
                lf_vocab_size=max_lf_id + 1,
                use_lf_reliability=use_lf_reliability
            )
            for _ in range(num_layers)
        ])
        self.classify = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, index, value):
        x = self.input_embed(value.float().unsqueeze(-1))  # (B,S,D)
        # pass through each bag-attention block
        for layer in self.layers:
            index, x, _ = layer(index, x)

        # Vectorized pooling of tokens → example embeddings
        B, S, D = x.shape
        example_ids = index[:, :, 0]  # (B, S)
        
        # Create a flattened view for efficient processing
        # Flatten batch and sequence dimensions (use reshape to handle non-contiguous tensors)
        x_flat = x.reshape(B * S, D)  # (B*S, D)
        example_ids_flat = example_ids.reshape(B * S)  # (B*S,)
        
        # Create batch offsets to make example IDs unique across batches
        batch_offsets = torch.arange(B, device=x.device).unsqueeze(1) * (example_ids.max() + 1)  # (B, 1)
        example_ids_global = example_ids + batch_offsets  # (B, S) - unique IDs across batches
        example_ids_global_flat = example_ids_global.reshape(B * S)  # (B*S,)
        
        # Use scatter_add for efficient pooling
        unique_ids, inverse_indices = torch.unique(example_ids_global_flat, return_inverse=True)
        num_examples = len(unique_ids)
        
        # Sum embeddings for each example
        example_sums = torch.zeros(num_examples, D, device=x.device, dtype=x.dtype)
        example_sums.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, D), x_flat)
        
        # Count tokens per example for averaging
        example_counts = torch.zeros(num_examples, device=x.device, dtype=torch.float)
        example_counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))
        
        # Average the embeddings
        example_embeddings = example_sums / example_counts.unsqueeze(1)  # (num_examples, D)
        
        # Get logits for all examples at once
        all_logits = self.classify(example_embeddings).squeeze(-1)  # (num_examples,)
        
        # Map back to batch structure
        # Find which batch each example belongs to
        example_to_batch = (unique_ids // (example_ids.max() + 1)).long()
        
        # Count examples per batch
        examples_per_batch = torch.zeros(B, device=x.device, dtype=torch.long)
        examples_per_batch.scatter_add_(0, example_to_batch, torch.ones_like(example_to_batch))
        
        E_max = examples_per_batch.max().item()
        
        # Create output tensors
        preds_padded = torch.zeros(B, E_max, device=x.device, dtype=all_logits.dtype)
        example_mask = torch.zeros(B, E_max, dtype=torch.bool, device=x.device)
        
        # Fill output tensors efficiently using completely vectorized operations
        if num_examples > 0:
            # Create indices for scatter operation
            batch_indices = example_to_batch  # (num_examples,)
            
            # Sort examples by batch for efficient processing
            sort_indices = torch.argsort(batch_indices)
            sorted_batch_indices = batch_indices[sort_indices]
            sorted_logits = all_logits[sort_indices]
            
            # Create within-batch position indices using vectorized cumsum approach
            # Mark batch boundaries
            is_new_batch = torch.cat([
                torch.tensor([True], device=x.device),
                sorted_batch_indices[1:] != sorted_batch_indices[:-1]
            ])
            
            # Create cumulative indices that reset at each batch
            cumsum_all = torch.arange(len(sorted_batch_indices), device=x.device)
            reset_values = torch.zeros_like(cumsum_all)
            reset_values[is_new_batch] = cumsum_all[is_new_batch]
            
            # Use cumulative maximum to propagate reset values
            reset_values = torch.cummax(reset_values, dim=0)[0]
            within_batch_indices = cumsum_all - reset_values
            
            # Use advanced indexing to fill the tensors
            preds_padded[sorted_batch_indices, within_batch_indices] = sorted_logits
            example_mask[sorted_batch_indices, within_batch_indices] = True

        return preds_padded, x, {"example_mask": example_mask}


class LELATransformerBagWrapper:
    """Wrapper for a trained Bag-Attention LELA model (O(NM)).

    Args:
        checkpoint_path (str or None): path to a checkpoint with 'model_state_dict'
        max_lf_id (int): maximum LF id (i.e., max column index) in your data
        use_lf_reliability (bool): whether to enable per-LF reliability gating
        embedding_dim (int): token embedding dimension inside the bag model
        device (str or None): e.g., 'cuda:0' or 'cpu' (auto if None)
    """
    def __init__(
        self,
        checkpoint_path,
        max_lf_id,
        use_lf_reliability=True,
        embedding_dim=16,
        device=None,
    ):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Build the bag-attention model
        self.net = StackedLELATransformerBag(
            num_layers=2,
            max_lf_id=max_lf_id,
            embedding_dim=embedding_dim,
            use_lf_reliability=use_lf_reliability,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            # If you ever switch architectures, you may need strict=False
            self.net.load_state_dict(checkpoint['model_state_dict'])

        self.net.to(self.device)
        self.net.eval()

    # Toggle per-LF reliability at inference time
    @torch.no_grad()
    def set_use_lf_reliability(self, flag: bool):
        self.net.bag_layer.set_use_lf_reliability(flag)

    def predict(self, X):
        """Return argmax class indices."""
        return np.argmax(self.predict_prob(X), axis=1)

    def predict_prob(self, X):
        """
        Multiclass API identical to your original wrapper.
        Expects X with values in {-1, 0, 1, 2, ...} where -1 = unlabeled/abstain.
        For binary, we build a 0/1 matrix and call _predict_binary once.
        For multiclass, we one-vs-rest over labels [0..C-1].
        """
        X = np.asarray(X)
        preds = []
        # n_class = max label + 1 over entries != -1
        valid = X != -1
        if not np.any(valid):
            # No signals at all: return uniform
            return np.ones((X.shape[0], 1)) if X.ndim == 1 else np.ones((X.shape[0], 1))

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

        # Tie handling & normalization (same as your original)
        tie_score = 1 / preds.shape[1] - 1e-9
        below = preds < tie_score
        if np.any(below):
            preds[below] = (preds[below] - tie_score) * tie_score / (tie_score - np.min(preds[below])) + tie_score
        preds = np.clip(preds, 0, 1)
        preds = preds / np.sum(preds, axis=1, keepdims=True)
        return preds

    def _predict_binary(self, X):
        """
        Binary probability for class 1 per example (row).
        Builds sparse COO (row=example id, col=LF id, data=value),
        runs the bag model for a single batch (B=1),
        and slices valid example logits via example_mask.
        """
        self.net.eval()
        X = np.asarray(X)
        X_sparse = coo_matrix(np.squeeze(X))

        # Build (1, S, 2) index and (1, S) value
        # index[..., 0] = row/example id; index[..., 1] = LF id (column)
        index = np.stack([X_sparse.row, X_sparse.col], axis=1)            # (S, 2)
        value = X_sparse.data                                             # (S,)

        index_t = torch.from_numpy(index).to(self.device).unsqueeze(0)    # (1, S, 2)
        value_t = torch.from_numpy(value).float().to(self.device).unsqueeze(0)  # (1, S)

        with torch.no_grad():
            preds_padded, _, aux = self.net(index_t, value_t)             # (1, E_max), mask (1, E_max)

        # Keep only valid example positions
        mask = aux["example_mask"][0]                                     # (E_max,)
        pred_vec = preds_padded[0, mask].detach().cpu().numpy()           # (num_examples,)

        return pred_vec