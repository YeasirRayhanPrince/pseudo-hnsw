"""
Trainer for HNSW Decision Transformer.

Based on the Decision Transformer trainer for Atari,
adapted for HNSW trajectory learning.

Key features:
- Handles variable-length sequences with padding masks
- Custom collate function support
- Loss computed only on valid (non-padded) positions
"""

import math
import logging
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class TrainerConfig:
    """
    Trainer configuration class.

    Attributes:
        max_epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        betas: Adam optimizer betas
        grad_norm_clip: Gradient clipping threshold
        weight_decay: L2 regularization weight
        lr_decay: Whether to use learning rate decay
        warmup_tokens: Number of tokens for linear warmup
        final_tokens: Number of tokens at which LR reaches 10% of initial
        ckpt_dir: Directory to save checkpoints
        num_workers: DataLoader workers
        seed: Random seed
    """
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_dir = None
    num_workers = 0
    seed = 123

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    """
    Trainer class for HNSW Decision Transformer.

    Handles:
    - Training loop with gradient accumulation
    - Learning rate scheduling (warmup + cosine decay)
    - Checkpointing
    - Logging
    - Variable-length sequences with padding masks
    """

    def __init__(self, model, train_dataset, test_dataset, config, collate_fn=None):
        """
        Initialize trainer.

        Args:
            model: The GPT model
            train_dataset: Training dataset
            test_dataset: Test dataset (optional)
            config: TrainerConfig
            collate_fn: Custom collate function for DataLoader
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.collate_fn = collate_fn

        # Set up device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Create checkpoint directory
        if self.config.ckpt_dir is not None:
            os.makedirs(self.config.ckpt_dir, exist_ok=True)

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        if self.config.ckpt_dir is None:
            return

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'loss': loss
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

    def train(self):
        """Main training loop."""
        model, config = self.model, self.config
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        self.tokens = 0  # Counter for LR decay

        for epoch in range(config.max_epochs):
            train_loss = self._run_epoch('train', epoch, optimizer)

            # Run evaluation if test dataset provided
            if self.test_dataset is not None:
                test_loss = self._run_epoch('test', epoch)
                logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.5f}, test_loss={test_loss:.5f}")
            else:
                logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.5f}")

            # Save checkpoint
            self.save_checkpoint(epoch + 1, train_loss)

        return train_loss

    def _run_epoch(self, split, epoch_num, optimizer=None):
        """Run single epoch of training or evaluation."""
        is_train = split == 'train'
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset

        loader = DataLoader(
            data,
            shuffle=is_train,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{split} epoch {epoch_num + 1}")

        for it, batch in pbar:
            # Unpack batch (new format with masks)
            states, node_ids, actions, k_values, r_values, seq_lens, masks = batch

            # Move to device
            states = states.to(self.device)
            node_ids = node_ids.to(self.device)
            actions = actions.to(self.device)
            k_values = k_values.to(self.device)
            r_values = r_values.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            with torch.set_grad_enabled(is_train):
                logits, loss = self.model(
                    query_vectors=states,
                    node_ids=node_ids,
                    k_values=k_values,
                    r_values=r_values,
                    actions=actions,
                    targets=actions,  # Target is the next action (with -100 for padding)
                    mask=masks
                )
                loss = loss.mean()  # Handle DataParallel
                losses.append(loss.item())

            if is_train:
                # Backprop
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                optimizer.step()

                # Learning rate decay
                if self.config.lr_decay:
                    # Count valid tokens (non-padded)
                    self.tokens += masks.sum().item()
                    if self.tokens < self.config.warmup_tokens:
                        lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
                    else:
                        progress = float(self.tokens - self.config.warmup_tokens) / \
                                   float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate

                # Update progress bar
                pbar.set_description(
                    f"epoch {epoch_num + 1} iter {it}: loss {loss.item():.5f}, lr {lr:e}"
                )

        avg_loss = float(np.mean(losses))
        return avg_loss

    def evaluate_trajectory_accuracy(self, num_samples=1000):
        """
        Evaluate trajectory prediction accuracy.

        Samples trajectories from test set and computes:
        - Top-1 accuracy: Fraction of correctly predicted next nodes
        - Top-5 accuracy: Fraction where correct node is in top-5 predictions
        - Final node accuracy: Accuracy of predicting the final target (r-th NN)
        """
        if self.test_dataset is None:
            logger.warning("No test dataset provided for evaluation")
            return {}

        self.model.eval()

        loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )

        top1_correct = 0
        top5_correct = 0
        final_top1_correct = 0
        final_top5_correct = 0
        total = 0
        final_total = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                if total >= num_samples:
                    break

                states, node_ids, actions, k_values, r_values, seq_lens, masks = batch

                states = states.to(self.device)
                node_ids = node_ids.to(self.device)
                actions = actions.to(self.device)
                k_values = k_values.to(self.device)
                r_values = r_values.to(self.device)
                masks = masks.to(self.device)

                logits, _ = self.model(
                    query_vectors=states,
                    node_ids=node_ids,
                    k_values=k_values,
                    r_values=r_values,
                    actions=actions,
                    targets=None,
                    mask=masks
                )

                # Get predictions and targets for valid positions only
                batch_size = states.shape[0]
                for b in range(batch_size):
                    seq_len = seq_lens[b].item()
                    if seq_len == 0:
                        continue

                    # Valid positions
                    valid_logits = logits[b, :seq_len, :]  # (seq_len, vocab_size)
                    valid_targets = actions[b, :seq_len]  # (seq_len,)

                    # Top-1 accuracy for all positions
                    probs = torch.softmax(valid_logits, dim=-1)
                    top1_preds = probs.argmax(dim=-1)
                    top1_correct += (top1_preds == valid_targets).sum().item()

                    # Top-5 accuracy for all positions
                    top5_preds = probs.topk(5, dim=-1).indices
                    top5_correct += (top5_preds == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()

                    total += seq_len

                    # Final node accuracy (most important!)
                    final_logits = valid_logits[-1]  # Last valid position
                    final_target = valid_targets[-1]

                    final_probs = torch.softmax(final_logits, dim=-1)
                    final_top1_pred = final_probs.argmax()
                    final_top1_correct += (final_top1_pred == final_target).item()

                    final_top5_pred = final_probs.topk(5).indices
                    final_top5_correct += (final_target in final_top5_pred).item()

                    final_total += 1

        metrics = {
            'top1_accuracy': top1_correct / total if total > 0 else 0,
            'top5_accuracy': top5_correct / total if total > 0 else 0,
            'final_top1_accuracy': final_top1_correct / final_total if final_total > 0 else 0,
            'final_top5_accuracy': final_top5_correct / final_total if final_total > 0 else 0,
            'num_tokens': total,
            'num_trajectories': final_total
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
