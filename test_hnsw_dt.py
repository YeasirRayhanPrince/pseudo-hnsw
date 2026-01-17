"""
Test script for HNSW Decision Transformer.

Verifies:
1. Data loading works correctly
2. Dataset class returns correct shapes (full trajectories)
3. Model forward pass works with masks
4. Loss computation works (ignores padding)
5. Each sample is ONE full trajectory (no segmentation)
"""

import sys
import torch
import numpy as np

# Add project to path
sys.path.insert(0, '/scratch/gilbreth/yrayhan/pseudo-hnsw')

from create_hnsw_dataset import create_dataset, get_trajectory_statistics
from mingpt.model_hnsw import GPT, GPTConfig
from run_dt_hnsw import HNSWTrajectoryDataset, collate_fn


def test_data_loading():
    """Test data loading with a small subset."""
    print("=" * 60)
    print("Testing data loading...")
    print("=" * 60)

    data_dir = '/scratch/gilbreth/yrayhan/pseudo-hnsw/dataset/SIFT1M/01_07_26_M_16_efCons_200'
    trajectory_path = f'{data_dir}/sift1M_result_path_v2.csv'
    vector_path = f'{data_dir}/sift_query_raw_vectors.csv'

    # Load small subset
    query_id_to_vector, trajectories = create_dataset(
        trajectory_path, vector_path, max_trajectories=100
    )

    # Get statistics
    stats = get_trajectory_statistics(trajectories)

    print(f"\nNumber of query vectors: {len(query_id_to_vector)}")
    print(f"Number of trajectories: {len(trajectories)}")
    print(f"Path length range: {stats['path_length_min']} - {stats['path_length_max']}")
    print(f"Rank range: {stats['rank_min']} - {stats['rank_max']}")

    # Check first trajectory
    traj = trajectories[0]
    print(f"\nFirst trajectory:")
    print(f"  Query ID: {traj['query_id']}")
    print(f"  Path length: {len(traj['path'])}")
    print(f"  First 3 steps: {traj['path'][:3]}")
    print(f"  Last 3 steps: {traj['path'][-3:]}")
    print(f"  Rank: {traj['rank']}")
    print(f"  Distance: {traj['distance']}")

    # Verify query vector exists for this trajectory
    if traj['query_id'] in query_id_to_vector:
        print(f"  Query vector shape: {query_id_to_vector[traj['query_id']].shape}")
    else:
        print(f"  WARNING: Query ID {traj['query_id']} not found in vector dict!")

    return query_id_to_vector, trajectories, stats


def test_dataset_class(query_id_to_vector, trajectories, stats):
    """Test PyTorch dataset class (full trajectories, no segmentation)."""
    print("\n" + "=" * 60)
    print("Testing dataset class (full trajectories)...")
    print("=" * 60)

    # Parameters
    max_seq_len = 40  # Should cover all trajectories (max is ~36)
    vocab_size = stats['max_node_id'] + 1

    # Create dataset
    dataset = HNSWTrajectoryDataset(
        query_id_to_vector,
        trajectories,
        max_seq_len,
        vocab_size
    )

    print(f"Dataset size: {len(dataset)} (should equal number of trajectories)")
    print(f"Max seq len: {max_seq_len}")
    print(f"Vocab size: {vocab_size}")

    # Verify dataset size equals number of trajectories (no segmentation!)
    assert len(dataset) == len(trajectories), \
        f"Dataset size {len(dataset)} != num trajectories {len(trajectories)}"
    print("PASS: Dataset size equals number of trajectories (no segmentation)")

    # Get a sample
    states, node_ids, actions, k_value, r_value, seq_len, mask = dataset[0]

    print(f"\nSample shapes:")
    print(f"  states: {states.shape} (expected: ({max_seq_len}, 128))")
    print(f"  node_ids: {node_ids.shape} (expected: ({max_seq_len},))")
    print(f"  actions: {actions.shape} (expected: ({max_seq_len},))")
    print(f"  k_value: {k_value} (expected: -1)")
    print(f"  r_value: {r_value} (should be rank from trajectory)")
    print(f"  seq_len: {seq_len} (actual trajectory length)")
    print(f"  mask: {mask.shape}, sum={mask.sum().item()} (should equal seq_len)")

    # Verify mask sum equals seq_len
    assert mask.sum().item() == seq_len.item(), \
        f"Mask sum {mask.sum().item()} != seq_len {seq_len.item()}"
    print("PASS: Mask sum equals sequence length")

    # Verify padding uses -100 for ignore_index
    if seq_len.item() < max_seq_len:
        first_padding = actions[seq_len.item()].item()
        assert first_padding == -100, f"Expected -100 for padding, got {first_padding}"
        print("PASS: Padding positions use -100 (ignore_index)")

    # Verify the trajectory matches the original
    traj = trajectories[0]
    original_path_len = len(traj['path']) - 1  # seq_len = path_len - 1
    assert seq_len.item() == original_path_len, \
        f"seq_len {seq_len.item()} != original path len - 1 ({original_path_len})"
    print(f"PASS: Sequence length matches original trajectory")

    # Verify final action is the last node in the path (the r-th NN target!)
    last_valid_action = actions[seq_len.item() - 1].item()
    expected_final_node = traj['path'][-1][0]  # Last node ID in path
    assert last_valid_action == expected_final_node, \
        f"Final action {last_valid_action} != expected final node {expected_final_node}"
    print(f"PASS: Final action is the r-th nearest neighbor target ({expected_final_node})")

    return dataset, vocab_size, max_seq_len


def test_collate_fn(dataset):
    """Test the custom collate function."""
    print("\n" + "=" * 60)
    print("Testing collate function...")
    print("=" * 60)

    # Get a batch of samples
    batch_size = 4
    batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]

    # Collate
    states, node_ids, actions, k_values, r_values, seq_lens, masks = collate_fn(batch)

    print(f"Batch shapes:")
    print(f"  states: {states.shape}")
    print(f"  node_ids: {node_ids.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  k_values: {k_values.shape}")
    print(f"  r_values: {r_values.shape}")
    print(f"  seq_lens: {seq_lens.shape}, values: {seq_lens.tolist()}")
    print(f"  masks: {masks.shape}")

    return states, node_ids, actions, k_values, r_values, seq_lens, masks


def test_model_forward(vocab_size, max_seq_len):
    """Test model forward pass with masks."""
    print("\n" + "=" * 60)
    print("Testing model forward pass...")
    print("=" * 60)

    # Small model for testing
    block_size = max_seq_len * 2  # For interleaved state-action
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        query_dim=128,
        max_timestep=max_seq_len,
        max_k=100,
        max_r=100,
    )

    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create dummy batch with padding
    batch_size = 4
    seq_len = max_seq_len

    query_vectors = torch.randn(batch_size, seq_len, 128)
    node_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))

    # Create actions with padding (-100)
    actions = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    masks = torch.zeros(batch_size, seq_len)

    # Fill in valid portions with different lengths
    actual_lens = [15, 20, 25, 30]
    for b, actual_len in enumerate(actual_lens):
        actions[b, :actual_len] = torch.randint(0, min(vocab_size, 1000), (actual_len,))
        masks[b, :actual_len] = 1.0

    k_values = torch.ones(batch_size, dtype=torch.long) * -1  # Single target mode
    r_values = torch.randint(1, 101, (batch_size,))

    print(f"\nInput shapes:")
    print(f"  query_vectors: {query_vectors.shape}")
    print(f"  node_ids: {node_ids.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  k_values: {k_values.shape}")
    print(f"  r_values: {r_values.shape}")
    print(f"  masks: {masks.shape}")
    print(f"  actual_lens: {actual_lens}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, loss = model(
            query_vectors=query_vectors,
            node_ids=node_ids,
            k_values=k_values,
            r_values=r_values,
            actions=actions,
            targets=actions,
            mask=masks
        )

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")

    # Verify loss is computed (should be finite, not NaN)
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is infinite!"
    print("PASS: Loss is finite and valid")

    return model


def test_inference_mode(model, vocab_size, max_seq_len):
    """Test inference mode (no actions provided)."""
    print("\n" + "=" * 60)
    print("Testing inference mode...")
    print("=" * 60)

    batch_size = 2
    seq_len = 1  # Start with just one state

    query_vectors = torch.randn(batch_size, seq_len, 128)
    node_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))
    k_values = torch.ones(batch_size, dtype=torch.long) * -1
    r_values = torch.randint(1, 101, (batch_size,))

    model.eval()
    with torch.no_grad():
        logits, loss = model(
            query_vectors=query_vectors,
            node_ids=node_ids,
            k_values=k_values,
            r_values=r_values,
            actions=None,
            targets=None,
            mask=None
        )

    print(f"Inference output shape: {logits.shape}")
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    preds = probs.argmax(dim=-1)
    print(f"Predicted next nodes: {preds}")
    print("PASS: Inference mode works")


def test_full_pipeline(query_id_to_vector, trajectories, stats):
    """Test full training pipeline with actual data."""
    print("\n" + "=" * 60)
    print("Testing full pipeline with actual data...")
    print("=" * 60)

    max_seq_len = 40
    vocab_size = stats['max_node_id'] + 1
    block_size = max_seq_len * 2

    # Create dataset
    dataset = HNSWTrajectoryDataset(
        query_id_to_vector,
        trajectories,
        max_seq_len,
        vocab_size
    )

    # Create model
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        query_dim=128,
        max_timestep=max_seq_len,
        max_k=100,
        max_r=100,
    )
    model = GPT(config)

    # Get a batch using collate_fn
    batch_size = min(8, len(dataset))
    batch = [dataset[i] for i in range(batch_size)]
    states, node_ids, actions, k_values, r_values, seq_lens, masks = collate_fn(batch)

    print(f"Batch shapes from actual data:")
    print(f"  states: {states.shape}")
    print(f"  seq_lens: {seq_lens.tolist()}")
    print(f"  r_values: {r_values.tolist()}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, loss = model(
            query_vectors=states,
            node_ids=node_ids,
            k_values=k_values,
            r_values=r_values,
            actions=actions,
            targets=actions,
            mask=masks
        )

    print(f"\nForward pass results:")
    print(f"  logits shape: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")

    # Check predictions at valid positions
    for b in range(min(3, batch_size)):
        valid_len = seq_lens[b].item()
        valid_logits = logits[b, :valid_len, :]
        valid_targets = actions[b, :valid_len]

        probs = torch.softmax(valid_logits, dim=-1)
        preds = probs.argmax(dim=-1)

        # Final prediction (most important - this is the r-th NN target)
        final_pred = preds[-1].item()
        final_target = valid_targets[-1].item()
        final_r = r_values[b].item()

        print(f"\n  Sample {b}: r={final_r}, seq_len={valid_len}")
        print(f"    Final target (r-th NN): {final_target}")
        print(f"    Final prediction: {final_pred}")
        print(f"    Match: {final_pred == final_target}")

    print("\nPASS: Full pipeline works with actual data")


def main():
    print("HNSW Decision Transformer Test Suite")
    print("=" * 60)
    print("Key design: FULL TRAJECTORIES (no segmentation)")
    print("=" * 60)

    # Test data loading
    query_id_to_vector, trajectories, stats = test_data_loading()

    # Test dataset class (full trajectories)
    dataset, vocab_size, max_seq_len = test_dataset_class(query_id_to_vector, trajectories, stats)

    # Test collate function
    test_collate_fn(dataset)

    # Use larger vocab for model testing
    test_vocab_size = max(vocab_size, 1000000)  # At least 1M for SIFT

    # Test model forward pass
    model = test_model_forward(test_vocab_size, max_seq_len)

    # Test inference mode
    test_inference_mode(model, test_vocab_size, max_seq_len)

    # Test full pipeline with actual data
    test_full_pipeline(query_id_to_vector, trajectories, stats)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
