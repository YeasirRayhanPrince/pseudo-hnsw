"""
HNSW Trajectory Dataset Loader

Loads HNSW search trajectories and query vectors for training a Decision Transformer.
"""

import csv
import numpy as np
from collections import defaultdict


def load_query_vectors(vector_path):
	"""
	Load query vectors from CSV file.

	Args:
		vector_path: Path to sift_query_raw_vectors.csv or sift_base_raw_vectors.csv

	Returns:
		query_id_to_vector: dict mapping query_id (int) to vector (np.array of shape (128,))
	"""
	print(f"Loading query vectors from {vector_path}...")
	query_id_to_vector = {}
	with open(vector_path, 'r') as f:
		reader = csv.reader(f)
		header = next(reader)  # Skip header
		for row in reader:
			query_id = int(row[0])  # First column is vector_id/query_id
			vec = np.array([float(x) for x in row[1:129]], dtype=np.float32)
			query_id_to_vector[query_id] = vec
	print(f"Loaded {len(query_id_to_vector)} vectors")
	return query_id_to_vector


def parse_trajectory(row):
	"""
	Parse a single trajectory row.

	Format: query_id, node_id, layer, node_id, layer, ..., rank, distance,

	Args:
		row: List of strings from CSV row

	Returns:
		dict with keys: query_id, path (list of (node_id, layer)), rank, distance
	"""
	# Remove empty strings (from trailing comma)
	row = [x for x in row if x.strip()]

	query_id = int(row[0])

	# Last two values are rank and distance
	rank = int(row[-2])
	distance = float(row[-1])

	# Middle values are pairs of (node_id, layer)
	path = []
	for i in range(1, len(row) - 2, 2):
		node_id = int(row[i])
		layer = int(row[i + 1])
		path.append((node_id, layer))

	return {
		'query_id': query_id,
		'path': path,
		'rank': rank,
		'distance': distance
	}


def load_trajectories(trajectory_path, max_trajectories=None):
	"""
	Load HNSW trajectories from CSV file.

	Args:
		trajectory_path: Path to sift1M_result_path_v2.csv
		max_trajectories: Maximum number of trajectories to load (None for all)

	Returns:
		List of trajectory dicts, each with keys:
			- query_id: int
			- path: list of (node_id, layer) tuples
			- rank: int (1-100)
			- distance: float
	"""
	print(f"Loading trajectories from {trajectory_path}...")
	trajectories = []
	with open(trajectory_path, 'r') as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if max_trajectories is not None and i >= max_trajectories:
				break
			try:
				traj = parse_trajectory(row)
				trajectories.append(traj)
			except (ValueError, IndexError) as e:
				print(f"Warning: Failed to parse row {i}: {e}")
				continue

			if (i + 1) % 100000 == 0:
				print(f"  Loaded {i + 1} trajectories...")

	print(f"Loaded {len(trajectories)} trajectories")
	return trajectories


def get_trajectory_statistics(trajectories):
	"""
	Compute statistics about the trajectories.

	Args:
		trajectories: List of trajectory dicts

	Returns:
		dict with statistics
	"""
	path_lengths = [len(t['path']) for t in trajectories]
	ranks = [t['rank'] for t in trajectories]
	query_ids = [t['query_id'] for t in trajectories]

	# Get unique node IDs
	all_node_ids = set()
	for t in trajectories:
		for node_id, layer in t['path']:
			all_node_ids.add(node_id)

	# Layer distribution
	layer_counts = defaultdict(int)
	for t in trajectories:
		for node_id, layer in t['path']:
			layer_counts[layer] += 1

	stats = {
		'num_trajectories': len(trajectories),
		'num_unique_queries': len(set(query_ids)),
		'num_unique_nodes': len(all_node_ids),
		'max_node_id': max(all_node_ids) if all_node_ids else 0,
		'path_length_min': min(path_lengths),
		'path_length_max': max(path_lengths),
		'path_length_mean': np.mean(path_lengths),
		'path_length_std': np.std(path_lengths),
		'rank_min': min(ranks),
		'rank_max': max(ranks),
		'layer_distribution': dict(sorted(layer_counts.items()))
	}
	return stats


def create_dataset(trajectory_path, vector_path, max_trajectories=None):
	"""
	Create the HNSW dataset for training.

	Args:
		trajectory_path: Path to trajectory CSV
		vector_path: Path to vector CSV
		max_trajectories: Maximum trajectories to load (None for all)

	Returns:
		query_id_to_vector: dict mapping query_id to vector (128-dim np.array)
		trajectories: List of trajectory dicts
	"""
	query_id_to_vector = load_query_vectors(vector_path)
	trajectories = load_trajectories(trajectory_path, max_trajectories)

	# Verify all trajectory query_ids have corresponding vectors
	missing_query_ids = set()
	for traj in trajectories:
		if traj['query_id'] not in query_id_to_vector:
			missing_query_ids.add(traj['query_id'])

	if missing_query_ids:
		print(f"Warning: {len(missing_query_ids)} trajectories have query_ids not in vector file")
		print(f"  Missing query_ids (first 10): {list(missing_query_ids)[:10]}")

	return query_id_to_vector, trajectories


def prepare_training_data(query_id_to_vector, trajectories, context_length=30):
	"""
	Prepare data in a format suitable for the Decision Transformer.

	For each trajectory, we create training samples where:
	- State: query vector + current node ID
	- Action: next node ID
	- Conditioning: k=-1 (single target), r=rank

	Args:
		query_id_to_vector: dict mapping query_id to vector (128-dim np.array)
		trajectories: List of trajectory dicts
		context_length: Maximum context length for transformer

	Returns:
		data: dict with training data
	"""
	# Extract node IDs (actions) from paths
	# For HNSW, the path shows the sequence of nodes visited
	# Action at step t is the node_id at step t+1

	all_states = []  # Query vectors
	all_node_ids = []  # Current node IDs
	all_actions = []  # Next node IDs (targets)
	all_k_values = []  # k conditioning (-1 for single target)
	all_r_values = []  # rank conditioning
	all_timesteps = []  # Position in trajectory
	done_idxs = []  # End of trajectory indices

	total_steps = 0
	skipped_trajectories = 0

	for traj in trajectories:
		query_id = traj['query_id']

		# Skip if query_id not in vector dict
		if query_id not in query_id_to_vector:
			skipped_trajectories += 1
			continue

		query_vec = query_id_to_vector[query_id]
		path = traj['path']
		rank = traj['rank']

		# For each step in the path (except the last), predict the next node
		for t in range(len(path) - 1):
			current_node_id, current_layer = path[t]
			next_node_id, next_layer = path[t + 1]

			all_states.append(query_vec)
			all_node_ids.append(current_node_id)
			all_actions.append(next_node_id)
			all_k_values.append(-1)  # Single target mode
			all_r_values.append(rank)
			all_timesteps.append(t)
			total_steps += 1

		done_idxs.append(total_steps)

	if skipped_trajectories > 0:
		print(f"Warning: Skipped {skipped_trajectories} trajectories with missing query vectors")

	data = {
		'states': np.array(all_states, dtype=np.float32),
		'node_ids': np.array(all_node_ids, dtype=np.int64),
		'actions': np.array(all_actions, dtype=np.int64),
		'k_values': np.array(all_k_values, dtype=np.int64),
		'r_values': np.array(all_r_values, dtype=np.int64),
		'timesteps': np.array(all_timesteps, dtype=np.int64),
		'done_idxs': np.array(done_idxs, dtype=np.int64)
	}

	print(f"\nPrepared training data:")
	print(f"  Total steps: {total_steps}")
	print(f"  States shape: {data['states'].shape}")
	print(f"  Actions shape: {data['actions'].shape}")

	return data


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--trajectory_path', type=str, required=True)
	parser.add_argument('--vector_path', type=str, required=True)
	parser.add_argument('--max_trajectories', type=int, default=None)
	args = parser.parse_args()

	query_id_to_vector, trajectories = create_dataset(
		args.trajectory_path,
		args.vector_path,
		args.max_trajectories
	)

	# Prepare training data
	data = prepare_training_data(query_id_to_vector, trajectories)
