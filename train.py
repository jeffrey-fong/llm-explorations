import argparse
import collections
import datetime
import json
import math
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.transformer.transformer import ClassicalTransformer, DecoderOnlyTransformer
from utils.tokenizer import Tokenizer


class GutenbergPoetryDataset(Dataset):
    def __init__(self, seq_len: int, tokenizer: Tokenizer):
        # Form a list of tokenized strings, one for each group id.
        grouped_data = collections.defaultdict(list)
        with open("gutenberg_poetry_corpus.ndjson", "r") as file:
            for line in file:
                item = json.loads(line.strip())
                grouped_data[item["gid"]].append(item["s"])
        self.data = [
            tokenizer.encode(" ".join(sentences)) for sentences in grouped_data.values()
        ]
        self.labels = [data[1:] + [tokenizer.pad_token_id] for data in self.data]

        # Split data into sequences of length seq_len
        for i, (data, labels) in enumerate(zip(self.data, self.labels)):
            self.data[i] = [
                self.data[i][j : min(j + seq_len, len(data))]
                for j in range(0, len(data), seq_len)
            ]
            self.labels[i] = [
                self.labels[i][j : min(j + seq_len, len(labels))]
                for j in range(0, len(labels), seq_len)
            ]
            # Pad the last sequence if necessary
            if len(self.data[i][-1]) < seq_len:
                self.data[i][-1] = self.data[i][-1] + [tokenizer.pad_token_id] * (
                    seq_len - len(self.data[i][-1])
                )
            if len(self.labels[i][-1]) < seq_len:
                self.labels[i][-1] = self.labels[i][-1] + [tokenizer.pad_token_id] * (
                    seq_len - len(self.labels[i][-1])
                )

        # Flatten the nested lists
        self.data = [seq for doc in self.data for seq in doc]
        self.labels = [seq for doc in self.labels for seq in doc]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def cosine_lr_schedule(step: int, max_steps: int) -> float:
    min_lr = min(args.lr / 5, 1e-4)

    # Warmup phase
    if step < args.warmup_steps:
        return (float(step) / float(max(1, args.warmup_steps))) * args.lr

    # Cosine decay phase
    progress = float(step - args.warmup_steps) / float(
        max(1, max_steps - args.warmup_steps)
    )
    return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def validate(
    model: Union[ClassicalTransformer, DecoderOnlyTransformer],
    tokenizer: Tokenizer,
    val_dataloader: DataLoader,
):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(args.device), y.to(args.device)
            _, loss = model(X, y)
            val_loss += loss

    avg_val_loss = val_loss / len(val_dataloader)
    return avg_val_loss


def train(
    tokenizer: Tokenizer, model: Union[ClassicalTransformer, DecoderOnlyTransformer]
):
    dataset = GutenbergPoetryDataset(args.seq_len, tokenizer)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [args.train_ratio, 1 - args.train_ratio]
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=True)

    max_steps = (
        len(train_dataloader) // args.gradient_accumulation_steps
    ) * args.epochs

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/{args.model_type}_{date}")

    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: cosine_lr_schedule(step, max_steps)
    )

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")
        train_loss = 0.0
        for step, (X, y) in enumerate(progress_bar):
            # Forward pass
            X, y = X.to(args.device), y.to(args.device)
            logits, loss = model(X, y)

            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            # Only update weights and log metrics after accumulating gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad()

                curr_step = (step + 1) // args.gradient_accumulation_steps

                # Log training metrics (use unscaled loss for logging)
                train_loss += loss.item() * args.gradient_accumulation_steps
                writer.add_scalar(
                    "Loss/train_step",
                    loss.item() * args.gradient_accumulation_steps,
                    curr_step,
                )
                writer.add_scalar(
                    "Learning_rate", scheduler.get_last_lr()[0], curr_step
                )

                # Run validation every args.eval_every steps
                if curr_step % args.eval_every == 0:
                    avg_val_loss = validate(model, tokenizer, val_dataloader)
                    writer.add_scalar("Loss/validation_step", avg_val_loss, curr_step)
                    model.train()  # Switch back to training mode

        # Log average training loss for epoch
        num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps
        avg_train_loss = train_loss / num_training_steps
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        # Validation

        writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)
        print(f"Validation loss: {avg_val_loss}")

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
        choices=["transformer", "decoder-only"],
        help="Model type (currently only transformer is supported)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Max sequence length of the model"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training"
    )

    # Training Arguments
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--eval-every", type=int, default=1000, help="Validation every n steps"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=32, help="Validation batch size"
    )
    return parser.parse_args()


def main():
    tokenizer = Tokenizer()
    if args.model_type == "transformer":
        model = ClassicalTransformer(
            seq_len=args.seq_len,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            device=args.device,
        ).to(args.device)
    elif args.model_type == "decoder-only":
        model = DecoderOnlyTransformer(
            seq_len=args.seq_len,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            device=args.device,
        ).to(args.device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    train(tokenizer, model)


if __name__ == "__main__":
    torch.manual_seed(10)
    args = parse_args()
    main()
