import argparse
import datetime
import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TransformerConfig
from models.transformer.transformer import Transformer
from utils.tokenizer import Tokenizer


class TinyShakespeareDataset(Dataset):
    def __init__(self, raw_str_data: str, tokenizer: Tokenizer, seq_len: int):
        # Form a list of tokenized strings
        self.tokenized_data = tokenizer.encode(raw_str_data)
        # Create overlapping sequences of tokens for training
        self.inputs = []
        self.labels = []

        # For each possible sequence start position
        for i in range(0, len(self.tokenized_data) - seq_len):
            # Input is sequence_length tokens
            self.inputs.append(self.tokenized_data[i : i + seq_len])
            # Label is sequence shifted by 1 (predict next token)
            self.labels.append(self.tokenized_data[i + 1 : i + seq_len + 1])

        assert len(self.inputs) == len(self.labels)

        # Print the size of the dataset
        print(f"Dataset size: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def cosine_lr_schedule(step: int, max_steps: int) -> float:
    # Warmup phase
    if step < args.warmup_steps:
        return (float(step) / float(max(1, args.warmup_steps))) * args.lr

    # Cosine decay phase
    progress = float(step - args.warmup_steps) / float(
        max(1, max_steps - args.warmup_steps)
    )
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def validate(
    model: Union[Transformer],
    tokenizer: Tokenizer,
    val_dataloader: DataLoader,
):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validating")
        for X, y in progress_bar:
            X, y = X.to(args.device), y.to(args.device)
            _, loss = model(X, y)
            val_loss += loss

    # Calculate average loss
    avg_val_loss = val_loss / len(val_dataloader)
    # Convert average loss to perplexity
    avg_ppl = torch.exp(avg_val_loss).item()
    return avg_val_loss, avg_ppl


def plot_graphs(train_losses, val_losses, steps):
    plt.figure(figsize=(12, 5))

    # Plot losses and perplexity side by side
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label="Train Loss", alpha=0.8)
    val_steps = [args.eval_every * (i + 1) for i in range(len(val_losses))]
    plt.plot(val_steps, val_losses, label="Validation Loss", alpha=0.8)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot perplexity
    plt.subplot(1, 2, 2)
    train_ppl = [torch.exp(torch.tensor(loss)).item() for loss in train_losses]
    val_ppl = [torch.exp(torch.tensor(loss)).item() for loss in val_losses]
    plt.plot(steps, train_ppl, label="Train Perplexity", alpha=0.8)
    plt.plot(val_steps, val_ppl, label="Validation Perplexity", alpha=0.8)
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.title("Training and Validation Perplexity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plots
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"training_curves_{args.model_type}_{date}.png")
    plt.close()


def train():
    # Read data file
    raw_str_data = open("./data.txt", "r").read()

    # Create tokenizer and dataset
    tokenizer = Tokenizer(raw_str_data)
    dataset = TinyShakespeareDataset(raw_str_data, tokenizer, args.seq_len)

    # Calculate split indices
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_ratio)

    # Sequential split
    train_data = torch.utils.data.Subset(dataset, range(train_size))
    val_data = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    train_dataloader = DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=True)

    max_steps = (
        len(train_dataloader) // args.gradient_accumulation_steps
    ) * args.epochs

    # Initialize model
    if args.model_type == "transformer":
        config = TransformerConfig(
            seq_len=args.seq_len,
            vocab_size=tokenizer.vocab_size,
            device=args.device,
            num_layers=4,
            hidden_size=128,
            ffn_size=128 * 4 * 2 // 3,
            num_heads=8,
            base=10000.0,
        )
        model = Transformer(config=config).to(args.device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    # Calculate and print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Model Type: {args.model_type}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024 * 1024):.2f}\n")

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/{args.model_type}_{date}")

    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: cosine_lr_schedule(step, max_steps)
    )

    # Add max gradient norm for clipping
    max_grad_norm = 1.0
    global_step = 0

    # Add these lists to store metrics for plotting
    train_losses = []
    val_losses = []
    steps = []

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")

        for step, (X, y) in enumerate(progress_bar):
            # Forward pass
            X, y = X.to(args.device), y.to(args.device)
            logits, loss = model(X, y)

            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            global_step += 1

            # Only update weights and log metrics after accumulating gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Add gradient clipping before optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                opt.step()
                scheduler.step()
                opt.zero_grad()

                # Store metrics for plotting (use unscaled loss)
                train_loss = loss.item() * args.gradient_accumulation_steps
                train_losses.append(train_loss)
                steps.append(global_step)

                writer.add_scalar("Loss/train_step", train_loss, global_step)
                writer.add_scalar(
                    "Learning_rate", scheduler.get_last_lr()[0], global_step
                )

                # Calculate perplexity from loss
                train_ppl = torch.exp(torch.tensor(train_loss)).item()
                writer.add_scalar("Perplexity/train_step", train_ppl, global_step)

                # Run validation every args.eval_every steps
                if global_step % args.eval_every == 0:
                    model.eval()
                    avg_val_loss, avg_val_ppl = validate(
                        model, tokenizer, val_dataloader
                    )
                    val_losses.append(avg_val_loss.item())
                    writer.add_scalar("Loss/validation_step", avg_val_loss, global_step)
                    writer.add_scalar(
                        "Perplexity/validation_step", avg_val_ppl, global_step
                    )
                    model.train()  # Switch back to training mode

    # Plot and save the training curves
    plot_graphs(train_losses, val_losses, steps)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train different language model architectures"
    )

    # Data Arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of the training data",
    )

    # Model Arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="transformer",
        choices=["transformer"],
        help="Model type (currently only transformer is supported)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=32, help="Max sequence length of the model"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training"
    )

    # Training Arguments
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument(
        "--min-lr", type=float, default=0.001, help="Minimum learning rate"
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--train-batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--eval-every", type=int, default=100, help="Validation every n steps"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=8192, help="Validation batch size"
    )
    return parser.parse_args()


def main():
    train()


if __name__ == "__main__":
    torch.manual_seed(10)
    args = parse_args()
    main()
