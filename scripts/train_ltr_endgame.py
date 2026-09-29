#!/usr/bin/env python3
"""train_ltr_endgame.py - Ultra-Lightweight 4-Pattern Endgame Move Ordering Trainer.

Implements ticket EVAL-009 (Egaroucid Paradigm):
Trains an ultra-compact 4-pattern model on exact-solved WTHOR games (data/ltr_exact_solved_15yr.pt)
using Soft-Target Knowledge Distillation + Listwise LTR.

Pattern Architecture (91,854 states = 183.7 KB):
1. Corner 3x3: 9 squares, 3^9 = 19,683 states (4 corners)
2. Edge + 2X: 10 squares, 3^10 = 59,049 states (4 edges)
3. 2x4 Rectangle: 8 squares, 3^8 = 6,561 states (8 octants)
4. Diagonal 8: 8 squares, 3^8 = 6,561 states (2 diagonals)

Exports weights directly to src/end_patterns_data.h for 100% L2 cache residency.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent scripts dir to import coeffs_tool mirror maps
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from coeffs_tool import init_mirror_maps, _MIRROR_MAPS

CORNER33_COUNT = 19683
EDGE2X_COUNT = 59049
RECT24_COUNT = 6561
DIAG8_COUNT = 6561
TOTAL_WEIGHTS = CORNER33_COUNT + EDGE2X_COUNT + RECT24_COUNT + DIAG8_COUNT  # 91854


class EndgameLTRDataset(Dataset):
    """Pre-processed tensor dataset for fast batched training."""

    def __init__(self, samples: List[dict], label_smoothing: float = 0.05):
        self.items = []
        for s in samples:
            child_feats = s["child_features"]
            k = len(child_feats)
            if k < 2:
                continue

            best_idx = s["best_idx"]
            if best_idx < 0 or best_idx >= k:
                continue

            # Extract 18 pattern indices per candidate move
            # c33: 34, 35, 36, 37
            # e2x: 0, 1, 2, 3
            # r24: derived from 38..45
            # d8: 16, 17
            move_indices = []
            for f in child_feats:
                c33 = [f[34], f[35], f[36], f[37]]
                e2x = [f[0], f[1], f[2], f[3]]
                r24 = [((f[i] // 243) % 81) * 81 + (f[i] % 81) for i in range(38, 46)]
                d8 = [f[16], f[17]]
                move_indices.append(c33 + e2x + r24 + d8)  # 18 indices

            # Label smoothed target distribution centered on best_idx
            targets = torch.full((k,), label_smoothing / (k - 1), dtype=torch.float32)
            targets[best_idx] = 1.0 - label_smoothing

            self.items.append((
                torch.tensor(move_indices, dtype=torch.long),  # (k, 18)
                targets,                                       # (k,)
                best_idx,
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    """Pad moves across batch items for vectorized computation."""
    max_k = max(item[0].shape[0] for item in batch)
    batch_size = len(batch)

    padded_indices = torch.zeros((batch_size, max_k, 18), dtype=torch.long)
    padded_targets = torch.zeros((batch_size, max_k), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    best_indices = torch.zeros(batch_size, dtype=torch.long)

    for i, (indices, targets, best_idx) in enumerate(batch):
        k = indices.shape[0]
        padded_indices[i, :k] = indices
        padded_targets[i, :k] = targets
        mask[i, :k] = True
        best_indices[i] = best_idx

    return padded_indices, padded_targets, mask, best_indices


class EndgamePatternModel(nn.Module):
    """Ultra-lightweight 4-pattern move ordering model."""

    def __init__(self):
        super().__init__()
        self.c33_emb = nn.Embedding(CORNER33_COUNT, 1)
        self.e2x_emb = nn.Embedding(EDGE2X_COUNT, 1)
        self.r24_emb = nn.Embedding(RECT24_COUNT, 1)
        self.d8_emb = nn.Embedding(DIAG8_COUNT, 1)

        # Initialize weights to zero
        nn.init.zeros_(self.c33_emb.weight)
        nn.init.zeros_(self.e2x_emb.weight)
        nn.init.zeros_(self.r24_emb.weight)
        nn.init.zeros_(self.d8_emb.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (batch_size, max_k, 18)
        c33: indices[:, :, 0:4]
        e2x: indices[:, :, 4:8]
        r24: indices[:, :, 8:16]
        d8:  indices[:, :, 16:18]
        """
        c33_scores = self.c33_emb(indices[:, :, 0:4]).squeeze(-1).sum(dim=-1)
        e2x_scores = self.e2x_emb(indices[:, :, 4:8]).squeeze(-1).sum(dim=-1)
        r24_scores = self.r24_emb(indices[:, :, 8:16]).squeeze(-1).sum(dim=-1)
        d8_scores = self.d8_emb(indices[:, :, 16:18]).squeeze(-1).sum(dim=-1)
        return c33_scores + e2x_scores + r24_scores + d8_scores


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, temp: float = 1.0) -> Tuple[float, float, float]:
    model.eval()
    correct_top1 = 0
    correct_top2 = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for padded_indices, padded_targets, mask, best_indices in loader:
            padded_indices = padded_indices.to(device)
            padded_targets = padded_targets.to(device)
            mask = mask.to(device)
            best_indices = best_indices.to(device)

            scores = model(padded_indices)  # (B, max_k)
            scores = scores.masked_fill(~mask, -1e9)

            log_probs = nn.functional.log_softmax(scores / temp, dim=-1)
            loss = -(padded_targets * log_probs).sum(dim=-1).mean()
            total_loss += loss.item() * len(best_indices)

            # Top-1 accuracy
            pred_top1 = torch.argmax(scores, dim=-1)
            correct_top1 += (pred_top1 == best_indices).sum().item()

            # Top-2 accuracy
            top2_preds = torch.topk(scores, k=min(2, scores.shape[1]), dim=-1).indices
            best_expanded = best_indices.unsqueeze(-1)
            correct_top2 += (top2_preds == best_expanded).any(dim=-1).sum().item()

            total += len(best_indices)

    return total_loss / max(1, total), (correct_top1 / max(1, total)) * 100.0, (correct_top2 / max(1, total)) * 100.0


def export_c_header(model: EndgamePatternModel, output_path: Path, weight_scale: float = 128.0):
    init_mirror_maps()
    m33 = _MIRROR_MAPS["mirror33"]
    m8x2 = _MIRROR_MAPS["mirror8x2"]
    m8 = _MIRROR_MAPS["mirror8"]

    c33_weights = model.c33_emb.weight.detach().cpu().squeeze(-1).numpy()
    e2x_weights = model.e2x_emb.weight.detach().cpu().squeeze(-1).numpy()
    r24_weights = model.r24_emb.weight.detach().cpu().squeeze(-1).numpy()
    d8_weights = model.d8_emb.weight.detach().cpu().squeeze(-1).numpy()

    # Symmetrize tables
    full_c33 = [int(round(float(c33_weights[m33[i]]) * weight_scale)) for i in range(CORNER33_COUNT)]
    full_e2x = [int(round(float(e2x_weights[m8x2[i]]) * weight_scale)) for i in range(EDGE2X_COUNT)]
    full_r24 = [int(round(float(r24_weights[i]) * weight_scale)) for i in range(RECT24_COUNT)]
    full_d8 = [int(round(float(d8_weights[m8[i]]) * weight_scale)) for i in range(DIAG8_COUNT)]

    # Clamp to int16 range
    def clamp_i16(v):
        return max(-32767, min(32767, v))

    full_weights = [clamp_i16(v) for v in (full_c33 + full_e2x + full_r24 + full_d8)]
    assert len(full_weights) == TOTAL_WEIGHTS, f"Expected {TOTAL_WEIGHTS} weights, got {len(full_weights)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("/*\n")
        f.write("  File:       end_patterns_data.h\n")
        f.write("  Generated:  scripts/train_ltr_endgame.py\n")
        f.write("  Model:      Ultra-Lightweight 4-Pattern Move Ordering Model (EVAL-009)\n")
        f.write("  Geometry:   Corner 3x3 (19683), Edge+2X (59049), 2x4 Rect (6561), Diag 8 (6561)\n")
        f.write(f"  Total:      {TOTAL_WEIGHTS} entries (183.7 KB), 100% L2 cache resident\n")
        f.write(f"  Scale:      1 disc = {weight_scale} fixed-point units\n")
        f.write("*/\n\n")
        f.write("#ifndef END_PATTERNS_DATA_H\n")
        f.write("#define END_PATTERNS_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"static const int16_t end_pattern_weights[{TOTAL_WEIGHTS}] = {{\n")

        # Format 12 numbers per line
        chunk_size = 12
        for i in range(0, len(full_weights), chunk_size):
            chunk = full_weights[i:i + chunk_size]
            line = ", ".join(f"{w:6d}" for w in chunk)
            if i + chunk_size < len(full_weights):
                f.write(f"  {line},\n")
            else:
                f.write(f"  {line}\n")

        f.write("};\n\n")
        f.write("#endif /* END_PATTERNS_DATA_H */\n")

    print(f"Exported {TOTAL_WEIGHTS} weights to {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB text)")


def main():
    parser = argparse.ArgumentParser(description="Train 4-pattern endgame move ordering model via LTR distillation.")
    parser.add_argument("--data", type=Path, default=Path("data/ltr_exact_solved_15yr.pt"), help="Path to exact solved LTR dataset")
    parser.add_argument("--stages", type=str, default="7,8,9,10", help="Stages to include (default: 7,8,9,10 covering discs 40-58, 6-24 empties)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing (default: 0.05)")
    parser.add_argument("--temp", type=float, default=1.0, help="Student temperature (default: 1.0)")
    parser.add_argument("--weight-scale", type=float, default=128.0, help="Fixed-point scaling factor (default: 128.0)")
    parser.add_argument("--val-split", type=float, default=0.10, help="Validation fraction (default: 0.10)")
    parser.add_argument("--out-header", type=Path, default=Path("src/end_patterns_data.h"), help="Output C header file")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    t0 = time.time()
    raw_data = torch.load(args.data, weights_only=False)
    print(f"Loaded {len(raw_data)} total samples in {time.time() - t0:.2f}s")

    target_stages = set(int(s.strip()) for s in args.stages.split(",") if s.strip())
    filtered_samples = [s for s in raw_data if s["stage"] in target_stages]
    print(f"Filtered {len(filtered_samples)} samples in stages {sorted(target_stages)}")

    # Train / Val split
    split_idx = int((1.0 - args.val_split) * len(filtered_samples))
    train_samples = filtered_samples[:split_idx]
    val_samples = filtered_samples[split_idx:]
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    print("Building tensor datasets...")
    t0 = time.time()
    train_dataset = EndgameLTRDataset(train_samples, label_smoothing=args.label_smoothing)
    val_dataset = EndgameLTRDataset(val_samples, label_smoothing=args.label_smoothing)
    print(f"Dataset tensors built in {time.time() - t0:.2f}s")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = EndgamePatternModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-3)

    # Initial evaluation
    val_loss, val_top1, val_top2 = evaluate_model(model, val_loader, device, temp=args.temp)
    print(f"Epoch  0/{args.epochs} | Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-2: {val_top2:.2f}% (Untrained baseline)")

    best_val_top1 = 0.0
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        ep_start = time.time()

        for padded_indices, padded_targets, mask, best_indices in train_loader:
            padded_indices = padded_indices.to(device)
            padded_targets = padded_targets.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            scores = model(padded_indices)
            scores = scores.masked_fill(~mask, -1e9)

            log_probs = nn.functional.log_softmax(scores / args.temp, dim=-1)
            loss = -(padded_targets * log_probs).sum(dim=-1).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(best_indices)
            total_samples += len(best_indices)

        scheduler.step()
        train_loss = total_loss / max(1, total_samples)

        # Validation
        val_loss, val_top1, val_top2 = evaluate_model(model, val_loader, device, temp=args.temp)
        elapsed = time.time() - ep_start
        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-2: {val_top2:.2f}% | Time: {elapsed:.1f}s")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1

    print(f"\nTraining complete in {time.time() - start_train_time:.1f}s. Best Val Top-1 Acc: {best_val_top1:.2f}%")

    # Export weights to C header
    export_c_header(model, args.out_header, weight_scale=args.weight_scale)


if __name__ == "__main__":
    main()
