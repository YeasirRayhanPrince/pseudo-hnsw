"""
GPT Model for HNSW Trajectory Learning.

Based on the Decision Transformer for Atari by Andrej Karpathy,
adapted for HNSW search trajectory behavior cloning.

The model learns to predict the next node in an HNSW search trajectory
given:
- Query vector (128-dim SIFT feature)
- Current node ID in the search path
- k conditioning (number of nearest neighbors to find, -1 for single target)
- r conditioning (rank of the target nearest neighbor, 1-100)

Key design: Full trajectories without segmentation
- Each input is a complete trajectory with padding
- Mask is used to compute loss only on valid positions
- The model learns to navigate to the r-th nearest neighbor
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation."""
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """
    GPT configuration class.

    Attributes:
        vocab_size: Number of unique node IDs (action space size)
        block_size: Maximum sequence length * 2 (for state-action interleaving)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        embd_pdrop: Embedding dropout probability
        resid_pdrop: Residual dropout probability
        attn_pdrop: Attention dropout probability
        max_timestep: Maximum timestep (trajectory length)
        query_dim: Dimension of query vectors (128 for SIFT)
        max_k: Maximum k value for conditioning
        max_r: Maximum r (rank) value for conditioning
    """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.

    Implements masked self-attention where each position can only
    attend to positions before it (causal/autoregressive).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # Dropout layers
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
            .view(1, 1, config.block_size + 1, config.block_size + 1)
        )
        self.n_head = config.n_head

    def forward(self, x, padding_mask=None):
        """
        Forward pass with optional padding mask.

        Args:
            x: Input tensor (batch, seq_len, n_embd)
            padding_mask: Optional mask (batch, seq_len) where 1=valid, 0=padding
        """
        B, T, C = x.size()

        # Calculate Q, K, V for all heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Apply padding mask if provided (mask out attention to padded positions)
        if padding_mask is not None:
            # padding_mask: (B, T) -> (B, 1, 1, T) for broadcasting
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(padding_mask_expanded == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    Transformer block with pre-norm architecture.

    Consists of:
    1. LayerNorm + Self-Attention + Residual
    2. LayerNorm + MLP + Residual
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.ln1(x), padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT model for HNSW trajectory prediction.

    Architecture:
    - Query encoder: MLP to project 128-dim query vectors to n_embd
    - Node embeddings: Embedding layer for node IDs (~1M nodes)
    - k/r embeddings: Embedding layers for conditioning values
    - Positional embeddings: Local position within sequence
    - Transformer blocks
    - Output head: Linear projection to vocab_size (predict next node)

    Token sequence format (interleaved state-action):
    [state_0, action_0, state_1, action_1, ..., state_T-1]

    Where state_emb = query_emb + node_emb + k_emb + r_emb

    The model predicts the next action (node) at each state position.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Query vector encoder (128-dim -> n_embd)
        self.query_encoder = nn.Sequential(
            nn.Linear(config.query_dim, config.n_embd),
            nn.Tanh()
        )

        # Node ID embedding (~1M nodes)
        self.node_embeddings = nn.Embedding(config.vocab_size, config.n_embd)

        # k and r conditioning embeddings
        # k: -1 to max_k (add 2 for offset: -1 maps to 0, 0 maps to 1, etc.)
        self.k_emb = nn.Embedding(config.max_k + 2, config.n_embd)
        self.r_emb = nn.Embedding(config.max_r + 1, config.n_embd)

        # Action embeddings (for input, same as node embeddings but separate)
        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Tanh()
        )
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        # Positional embeddings (local position within sequence)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))

        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("Number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        Configure AdamW optimizer with weight decay.

        Separates parameters into:
        - decay: Linear weights
        - no_decay: Biases, LayerNorm, Embeddings, positional embeddings
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Positional embeddings don't decay
        no_decay.add('pos_emb')

        # Validate all parameters are accounted for
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay!"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters {param_dict.keys() - union_params} not categorized!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, query_vectors, node_ids, k_values, r_values, actions=None, targets=None, mask=None):
        """
        Forward pass through the model.

        Args:
            query_vectors: (batch, seq_len, 128) query vectors
            node_ids: (batch, seq_len) current node IDs in trajectory
            k_values: (batch,) k conditioning values (-1 to max_k)
            r_values: (batch,) r conditioning values (1 to max_r)
            actions: (batch, seq_len) previous actions (node IDs), or None
            targets: (batch, seq_len) target actions for loss computation, or None
                     Use -100 for padding positions (ignored in loss)
            mask: (batch, seq_len) attention mask (1=valid, 0=padding), or None

        Returns:
            logits: (batch, seq_len, vocab_size) predicted logits for next node
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size = query_vectors.shape[0]
        seq_len = query_vectors.shape[1]

        # Encode query vectors: (batch, seq_len, n_embd)
        query_emb = self.query_encoder(query_vectors.type(torch.float32))

        # Embed current node IDs: (batch, seq_len, n_embd)
        node_emb = self.node_embeddings(node_ids.type(torch.long))

        # Combine query and node embeddings for state representation
        state_emb = query_emb + node_emb

        # Embed k and r conditioning: (batch, n_embd)
        # k values are shifted by 1 so -1 maps to index 0
        k_emb = self.k_emb((k_values + 1).type(torch.long))  # (batch, n_embd)
        r_emb = self.r_emb(r_values.type(torch.long))  # (batch, n_embd)

        # Expand k/r to match sequence length and add to state
        k_cond = k_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_embd)
        r_cond = r_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, n_embd)
        state_emb = state_emb + k_cond + r_cond

        # Build token sequence
        if actions is not None:
            # Training mode: interleave state and action embeddings
            # [state_0, action_0, state_1, action_1, ..., state_{T-1}]
            # We predict action_t after seeing state_t

            # Clamp actions for embedding (replace -100 padding with 0)
            # The -100 values are only used in targets for loss computation
            actions_for_emb = actions.clamp(min=0)
            action_emb = self.action_embeddings(actions_for_emb.type(torch.long))

            # Create interleaved sequence: [s0, a0, s1, a1, ..., s_{T-1}]
            # Total length = 2 * seq_len - 1 (no action after last state during training)
            token_len = seq_len * 2 - 1
            token_embeddings = torch.zeros(
                (batch_size, token_len, self.config.n_embd),
                dtype=torch.float32, device=state_emb.device
            )
            token_embeddings[:, 0::2, :] = state_emb  # States at even positions
            token_embeddings[:, 1::2, :] = action_emb[:, :-1, :]  # Actions at odd positions (except last)

            # Create padding mask for interleaved sequence if mask provided
            if mask is not None:
                # Expand mask for interleaved format
                interleaved_mask = torch.zeros(
                    (batch_size, token_len),
                    dtype=torch.float32, device=mask.device
                )
                interleaved_mask[:, 0::2] = mask  # State positions
                # Action positions are valid if the state before them is valid
                interleaved_mask[:, 1::2] = mask[:, :-1]
                padding_mask = interleaved_mask
            else:
                padding_mask = None
        else:
            # Inference mode: just states (for autoregressive generation)
            token_embeddings = state_emb
            padding_mask = mask

        # Add positional embeddings
        position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        # Apply dropout and transformer
        x = self.drop(token_embeddings + position_embeddings)

        # Pass through transformer blocks with padding mask
        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        # Extract logits at state positions (we predict action after seeing state)
        if actions is not None:
            logits = logits[:, 0::2, :]  # Keep predictions from state positions (even indices)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Use ignore_index=-100 to skip padding positions
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss
