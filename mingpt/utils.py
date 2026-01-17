"""
Utility functions for HNSW Decision Transformer.

Based on the minGPT utils from Andrej Karpathy.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    """
    Zero out logits not in top-k.

    Args:
        logits: (batch, vocab_size) tensor
        k: number of top elements to keep

    Returns:
        logits with non-top-k elements set to -inf
    """
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample_hnsw(model, query_vectors, node_ids, k_values, r_values, timesteps,
                temperature=1.0, sample=False, top_k=None):
    """
    Sample next node prediction from the HNSW model.

    Args:
        model: GPT model
        query_vectors: (batch, seq_len, 128) query vectors
        node_ids: (batch, seq_len) current node IDs
        k_values: (batch, 1) k conditioning values
        r_values: (batch, 1) r conditioning values
        timesteps: (batch, 1) timestep values
        temperature: Sampling temperature
        sample: If True, sample from distribution; else take argmax
        top_k: If set, only sample from top-k logits

    Returns:
        next_node_ids: (batch, 1) predicted next node IDs
    """
    model.eval()

    # Get block size from model
    block_size = model.get_block_size()
    max_seq_len = block_size // 3  # Account for k/r, state, action tokens

    # Crop context if needed
    if query_vectors.size(1) > max_seq_len:
        query_vectors = query_vectors[:, -max_seq_len:]
        node_ids = node_ids[:, -max_seq_len:]

    # Forward pass
    logits, _ = model(
        query_vectors=query_vectors,
        node_ids=node_ids,
        k_values=k_values,
        r_values=r_values,
        timesteps=timesteps,
        targets=None
    )

    # Get logits for last position and scale by temperature
    logits = logits[:, -1, :] / temperature

    # Optionally crop to top-k
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    # Apply softmax
    probs = F.softmax(logits, dim=-1)

    # Sample or take argmax
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)

    return ix


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
