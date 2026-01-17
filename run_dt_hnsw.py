"""
HNSW Decision Transformer Training Script

Trains a Decision Transformer to clone HNSW search trajectories.
The model learns to predict the next node in the search path given:
- Query vector (128-dim SIFT features)
- Current node ID
- k conditioning (number of neighbors to find)
- r conditioning (rank of target neighbor)

Key Design: Full trajectories (no segmentation)
- Each sample is one complete trajectory
- Variable-length sequences handled with padding and attention masks
- Max trajectory length is ~36 steps, context_length=40 covers all

Usage:
  python run_dt_hnsw.py --data_dir /path/to/dataset

  # With custom parameters
  python run_dt_hnsw.py --data_dir /path/to/dataset \
    --epochs 10 --batch_size 128 --context_length 40
"""

import logging
import argparse
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from mingpt.utils import set_seed
from mingpt.model_hnsw import GPT, GPTConfig
from mingpt.trainer_hnsw import Trainer, TrainerConfig
from create_hnsw_dataset import create_dataset, get_trajectory_statistics


class HNSWTrajectoryDataset(Dataset):
  """
  PyTorch Dataset for HNSW trajectories.

  Key design: Each sample is ONE FULL trajectory (no segmentation).
  This preserves the relationship between (k, r) conditioning and the
  final target (the r-th nearest neighbor).

  Each sample contains:
  - states: Query vectors (same query repeated for each step)
  - node_ids: Current node IDs in the trajectory
  - actions: Next node IDs (targets for prediction)
  - k_value: k conditioning (-1 for single target mode)
  - r_value: Rank conditioning (1-100)
  - seq_len: Actual length of the trajectory (before padding)
  - mask: Attention mask (1 for valid positions, 0 for padding)
  """

  def __init__(self, query_id_to_vector, trajectories, max_seq_len, vocab_size):
    """
    Args:
      query_id_to_vector: Dict mapping query_id to 128-dim vector
      trajectories: List of trajectory dicts with keys: query_id, path, rank, distance
      max_seq_len: Maximum sequence length (for padding)
      vocab_size: Number of unique node IDs
    """
    self.query_id_to_vector = query_id_to_vector
    self.max_seq_len = max_seq_len
    self.vocab_size = vocab_size

    # Filter out trajectories with missing query vectors
    self.trajectories = []
    skipped = 0
    for traj in trajectories:
      if traj['query_id'] in query_id_to_vector:
        self.trajectories.append(traj)
      else:
        skipped += 1

    if skipped > 0:
      print(f"Warning: Skipped {skipped} trajectories with missing query vectors")

    print(f"Dataset initialized with {len(self.trajectories)} trajectories")

  def __len__(self):
    return len(self.trajectories)

  def __getitem__(self, idx):
    """
    Get a full trajectory.

    Returns:
      states: (max_seq_len, 128) query vectors (padded)
      node_ids: (max_seq_len,) current node IDs (padded with 0)
      actions: (max_seq_len,) next node IDs / targets (padded with -100 for ignore)
      k_value: scalar, k conditioning value
      r_value: scalar, r conditioning value
      seq_len: scalar, actual sequence length
      mask: (max_seq_len,) attention mask (1=valid, 0=padding)
    """
    traj = self.trajectories[idx]
    query_vec = self.query_id_to_vector[traj['query_id']]
    path = traj['path']  # List of (node_id, layer) tuples
    rank = traj['rank']

    # Extract node IDs from path
    path_node_ids = [node_id for node_id, layer in path]

    # Number of (state, action) pairs = len(path) - 1
    # At step t: state is current node, action is next node
    seq_len = min(len(path) - 1, self.max_seq_len)

    # Initialize padded arrays
    states = np.zeros((self.max_seq_len, 128), dtype=np.float32)
    node_ids = np.zeros(self.max_seq_len, dtype=np.int64)
    actions = np.full(self.max_seq_len, -100, dtype=np.int64)  # -100 = ignore index
    mask = np.zeros(self.max_seq_len, dtype=np.float32)

    # Fill in actual values
    for t in range(seq_len):
      states[t] = query_vec  # Same query for all steps
      node_ids[t] = path_node_ids[t]  # Current node
      actions[t] = path_node_ids[t + 1]  # Next node (target)
      mask[t] = 1.0

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    node_ids = torch.tensor(node_ids, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    k_value = torch.tensor(-1, dtype=torch.long)  # Single target mode
    r_value = torch.tensor(rank, dtype=torch.long)
    seq_len_tensor = torch.tensor(seq_len, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.float32)

    return states, node_ids, actions, k_value, r_value, seq_len_tensor, mask


def collate_fn(batch):
  """
  Custom collate function for variable-length trajectories.

  All trajectories are already padded to max_seq_len in __getitem__,
  so we just need to stack them.
  """
  states, node_ids, actions, k_values, r_values, seq_lens, masks = zip(*batch)

  return (
    torch.stack(states),      # (batch, max_seq_len, 128)
    torch.stack(node_ids),    # (batch, max_seq_len)
    torch.stack(actions),     # (batch, max_seq_len)
    torch.stack(k_values),    # (batch,)
    torch.stack(r_values),    # (batch,)
    torch.stack(seq_lens),    # (batch,)
    torch.stack(masks)        # (batch, max_seq_len)
  )


def main(args):
  # Set random seed
  set_seed(args.seed)

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
  )
  logger = logging.getLogger(__name__)

  # Data paths
  trajectory_path = os.path.join(args.data_dir, 'sift1M_result_path_v2.csv')
  vector_path = os.path.join(args.data_dir, 'sift_query_raw_vectors.csv')

  # Load data
  logger.info("Loading dataset...")
  query_id_to_vector, trajectories = create_dataset(
    trajectory_path,
    vector_path,
    args.max_trajectories
  )

  # Get statistics
  stats = get_trajectory_statistics(trajectories)
  logger.info(f"Dataset statistics:")
  logger.info(f"  Trajectories: {stats['num_trajectories']}")
  logger.info(f"  Unique queries: {stats['num_unique_queries']}")
  logger.info(f"  Unique nodes: {stats['num_unique_nodes']}")
  logger.info(f"  Max node ID: {stats['max_node_id']}")
  logger.info(f"  Path length: min={stats['path_length_min']}, max={stats['path_length_max']}, "
        f"mean={stats['path_length_mean']:.2f}")
  logger.info(f"  Rank range: {stats['rank_min']} to {stats['rank_max']}")
  logger.info(f"  Layer distribution: {stats['layer_distribution']}")

  # Vocab size = max node ID + 1
  vocab_size = stats['max_node_id'] + 1
  logger.info(f"Vocabulary size: {vocab_size}")

  # Max sequence length should cover all trajectories
  # Path length max is the number of nodes, seq_len = path_len - 1 (state-action pairs)
  max_seq_len = args.context_length
  if stats['path_length_max'] > max_seq_len:
    logger.warning(f"Some trajectories ({stats['path_length_max']} steps) exceed "
            f"context_length ({max_seq_len}). Consider increasing context_length.")

  # Create dataset
  train_dataset = HNSWTrajectoryDataset(
    query_id_to_vector,
    trajectories,
    max_seq_len,
    vocab_size
  )
  logger.info(f"Training dataset size: {len(train_dataset)}")

  # Model configuration
  # block_size = max_seq_len * 2 for interleaved [state, action, state, action, ...]
  block_size = max_seq_len * 2
  mconf = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    node_embd=args.node_embd,  # Separate (smaller) embedding for node IDs
    query_dim=128,  # SIFT feature dimension
    max_timestep=max_seq_len,
    max_k=100,  # Maximum k for conditioning (-1 to 100)
    max_r=100,  # Maximum rank (1-100)
  )

  logger.info("Creating model...")
  model = GPT(mconf)

  # Log model info
  n_params = sum(p.numel() for p in model.parameters())
  logger.info(f"Model parameters: {n_params:,}")
  logger.info(f"Node embedding: {vocab_size} x {args.node_embd} = {vocab_size * args.node_embd * 4 / 1e6:.1f} MB (shared for states & actions)")
  logger.info(f"State projection: {128 + 3*args.node_embd} -> {args.n_embd}")

  # Trainer configuration
  tconf = TrainerConfig(
    max_epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_decay=args.lr_decay,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * max_seq_len,
    num_workers=args.num_workers,
    seed=args.seed,
    ckpt_dir=args.ckpt_dir,
  )

  # Create trainer and train
  logger.info("Starting training...")
  trainer = Trainer(model, train_dataset, None, tconf, collate_fn=collate_fn)
  trainer.train()

  logger.info("Training complete!")


if __name__ == '__main__':
  # Argument parser (only runs when executed directly)
  parser = argparse.ArgumentParser(description='HNSW Decision Transformer Training')
  parser.add_argument('--seed', type=int, default=123, help='Random seed')
  parser.add_argument('--context_length', type=int, default=40, help='Max trajectory length (>= max path length)')
  parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
  parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
  parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
  parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
  parser.add_argument('--n_embd', type=int, default=128, help='Transformer embedding dimension')
  parser.add_argument('--node_embd', type=int, default=32, help='Node ID embedding dimension (smaller than n_embd)')
  parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
  parser.add_argument('--data_dir', type=str, required=True, help='Directory containing dataset files')
  parser.add_argument('--max_trajectories', type=int, default=None, help='Maximum trajectories to load (None for all)')
  parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Checkpoint directory')
  parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
  parser.add_argument('--lr_decay', action='store_true', help='Use learning rate decay')
  args = parser.parse_args()

  main(args)
