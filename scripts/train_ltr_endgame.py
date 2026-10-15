#!/usr/bin/env python3
"""train_ltr_endgame.py - Calibrated 3-Pattern Endgame Move Ordering Trainer.

Trains an ultra-compact 3-pattern model (Corner 3x3, Edge+2X, 2x4 Rectangle; 85,293 weights = 170.6 KB)
on exact-solved WTHOR games (data/ltr_exact_solved_15yr.pt) using Soft-Target Knowledge Distillation + Listwise LTR.
Applies exact spatial mirror tying (m33, m8x2, mirror8) and reference header standard deviation normalization.
Exports weights directly to src/end_patterns_data.h as `const int16_t end_pattern_weights[85293]`.
"""

import argparse
import os
import re
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
TOTAL_WEIGHTS = CORNER33_COUNT + EDGE2X_COUNT + RECT24_COUNT  # 85293


class EndgameLTRDataset(Dataset):
    """Pre-processed tensor dataset for fast batched training."""

    def __init__(self, samples: List[dict], scores: Optional[List[float]] = None, has_scores: Optional[List[bool]] = None, label_smoothing: float = 0.05):
        self.items = []
        for i, s in enumerate(samples):
            if "child_features_16" in s:
                move_indices = s["child_features_16"]
            elif "child_features_18" in s:
                move_indices = [m[:16] for m in s["child_features_18"]]
            else:
                child_feats = s.get("child_features", [])
                if len(child_feats) < 2:
                    continue
                move_indices = []
                for f in child_feats:
                    c33 = [f[34], f[35], f[36], f[37]]
                    e2x = [f[0], f[1], f[2], f[3]]
                    r24 = [((f[idx] // 243) % 81) * 81 + (f[idx] % 81) for idx in range(38, 46)]
                    move_indices.append(c33 + e2x + r24)  # 16 indices

            k = len(move_indices)
            if k < 2:
                continue

            best_idx = s["best_idx"]
            if best_idx < 0 or best_idx >= k:
                continue

            # Label smoothed target distribution centered on best_idx
            targets = torch.full((k,), label_smoothing / (k - 1), dtype=torch.float32)
            targets[best_idx] = 1.0 - label_smoothing

            score_val = float(scores[i]) if (scores is not None and i < len(scores)) else 0.0
            has_sc = bool(has_scores[i]) if (has_scores is not None and i < len(has_scores)) else (scores is not None and scores[i] != 0.0)

            self.items.append((
                torch.tensor(move_indices, dtype=torch.long),  # (k, 16)
                targets,                                       # (k,)
                best_idx,
                score_val,
                has_sc,
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    """Pad moves across batch items for vectorized computation."""
    max_k = max(item[0].shape[0] for item in batch)
    batch_size = len(batch)

    padded_indices = torch.zeros((batch_size, max_k, 16), dtype=torch.long)
    padded_targets = torch.zeros((batch_size, max_k), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    best_indices = torch.zeros(batch_size, dtype=torch.long)
    target_scores = torch.zeros(batch_size, dtype=torch.float32)
    has_scores = torch.zeros(batch_size, dtype=torch.bool)

    for i, (indices, targets, best_idx, score_val, has_sc) in enumerate(batch):
        k = indices.shape[0]
        padded_indices[i, :k] = indices
        padded_targets[i, :k] = targets
        mask[i, :k] = True
        best_indices[i] = best_idx
        target_scores[i] = score_val
        has_scores[i] = has_sc

    return padded_indices, padded_targets, mask, best_indices, target_scores, has_scores


class EndgamePatternModel(nn.Module):
    """Calibrated 3-pattern move ordering model with exact spatial mirror tying."""

    def __init__(self):
        super().__init__()
        self.c33_emb = nn.Embedding(CORNER33_COUNT, 1)
        self.e2x_emb = nn.Embedding(EDGE2X_COUNT, 1)
        self.r24_emb = nn.Embedding(RECT24_COUNT, 1)

        # Initialize weights to zero
        nn.init.zeros_(self.c33_emb.weight)
        nn.init.zeros_(self.e2x_emb.weight)
        nn.init.zeros_(self.r24_emb.weight)

        init_mirror_maps()
        self.register_buffer("m33", torch.tensor(_MIRROR_MAPS["mirror33"], dtype=torch.long))
        self.register_buffer("m8x2", torch.tensor(_MIRROR_MAPS["mirror8x2"], dtype=torch.long))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (batch_size, max_k, 16)"""
        c_idx = self.m33[indices[:, :, 0:4]]
        e_idx = self.m8x2[indices[:, :, 4:8]]
        r_idx = indices[:, :, 8:16]

        c33_scores = self.c33_emb(c_idx).squeeze(-1).sum(dim=-1)
        e2x_scores = self.e2x_emb(e_idx).squeeze(-1).sum(dim=-1)
        r24_scores = self.r24_emb(r_idx).squeeze(-1).sum(dim=-1)
        return c33_scores + e2x_scores + r24_scores


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, temp: float = 1.0, mse_weight: float = 0.02) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    correct_top1 = 0
    correct_top2 = 0
    total = 0
    scored_total = 0
    total_ltr_loss = 0.0
    total_mse_loss = 0.0
    total_abs_err = 0.0

    with torch.no_grad():
        for padded_indices, padded_targets, mask, best_indices, target_scores, has_scores in loader:
            padded_indices = padded_indices.to(device)
            padded_targets = padded_targets.to(device)
            mask = mask.to(device)
            best_indices = best_indices.to(device)
            target_scores = target_scores.to(device)
            has_scores = has_scores.to(device)

            scores = model(padded_indices)  # (B, max_k) in discs
            best_scores = scores.gather(1, best_indices.unsqueeze(-1)).squeeze(-1)

            # MSE and MAE in discs (only on samples with ground-truth score)
            if has_scores.any():
                sc_best = best_scores[has_scores]
                sc_target = target_scores[has_scores]
                mse = nn.functional.mse_loss(sc_best, sc_target)
                mae = (sc_best - sc_target).abs().mean()
                total_mse_loss += mse.item() * len(sc_best)
                total_abs_err += mae.item() * len(sc_best)
                scored_total += len(sc_best)

            # LTR cross entropy
            masked_scores = scores.masked_fill(~mask, -1e9)
            log_probs = nn.functional.log_softmax(masked_scores / temp, dim=-1)
            ltr_loss = -(padded_targets * log_probs).sum(dim=-1).mean()
            total_ltr_loss += ltr_loss.item() * len(best_indices)

            # Top-1 accuracy
            pred_top1 = torch.argmax(masked_scores, dim=-1)
            correct_top1 += (pred_top1 == best_indices).sum().item()

            # Top-2 accuracy
            top2_preds = torch.topk(masked_scores, k=min(2, masked_scores.shape[1]), dim=-1).indices
            best_expanded = best_indices.unsqueeze(-1)
            correct_top2 += (top2_preds == best_expanded).any(dim=-1).sum().item()

            total += len(best_indices)

    avg_ltr = total_ltr_loss / max(1, total)
    avg_mse = total_mse_loss / max(1, scored_total) if scored_total > 0 else 0.0
    avg_mae = total_abs_err / max(1, scored_total) if scored_total > 0 else 0.0
    top1_pct = (correct_top1 / max(1, total)) * 100.0
    top2_pct = (correct_top2 / max(1, total)) * 100.0
    total_loss = avg_ltr + mse_weight * avg_mse

    return total_loss, avg_ltr, avg_mse, avg_mae, top1_pct, top2_pct


def load_header_weights(path: Path) -> torch.Tensor:
    with open(path, "r") as f:
        text = f.read()
    match = re.search(r"end_pattern_weights\[\d+\]\s*=\s*\{([^}]+)\};", text)
    if not match:
        raise ValueError(f"Could not find end_pattern_weights array in {path}")
    raw_nums = [int(x) for x in re.findall(r"-?\d+", match.group(1))]
    assert len(raw_nums) == TOTAL_WEIGHTS, f"Expected {TOTAL_WEIGHTS} weights, found {len(raw_nums)}"
    return torch.tensor(raw_nums, dtype=torch.float32)


def export_c_header(
    model: EndgamePatternModel,
    output_path: Path,
    ref_header: Path,
    val_loader: DataLoader,
    weight_scale: float = 128.0,
    c33_overlap_scale: float = 15.0 / 16.0,
    e2x_overlap_scale: float = 1.0,
):
    m33 = model.m33.cpu()
    m8x2 = model.m8x2.cpu()
    w_c33 = model.c33_emb.weight.detach().cpu().squeeze(-1)[m33] * weight_scale
    w_e2x = model.e2x_emb.weight.detach().cpu().squeeze(-1)[m8x2] * weight_scale
    w_r24 = model.r24_emb.weight.detach().cpu().squeeze(-1) * weight_scale

    # Reference standard deviation normalization if ref_header exists
    if ref_header and ref_header.exists():
        print(f"Normalizing group stds against reference header {ref_header}...")
        w_ref = load_header_weights(ref_header)
        ref_c33 = w_ref[0:19683]
        ref_e2x = w_ref[19683:19683+59049]
        ref_r24 = w_ref[19683+59049:TOTAL_WEIGHTS]

        all_c33, all_e2x, all_r24 = [], [], []
        all_ref_c33, all_ref_e2x, all_ref_r24 = [], [], []
        with torch.no_grad():
            for padded_indices, _, mask, _, _, _ in val_loader:
                valid_moves = padded_indices[mask].cpu()
                c_idx = valid_moves[:, 0:4]
                e_idx = valid_moves[:, 4:8]
                r_idx = valid_moves[:, 8:16]

                all_c33.append(w_c33[c_idx].sum(dim=-1))
                all_e2x.append(w_e2x[e_idx].sum(dim=-1))
                all_r24.append(w_r24[r_idx].sum(dim=-1))

                all_ref_c33.append(ref_c33[c_idx].sum(dim=-1))
                all_ref_e2x.append(ref_e2x[e_idx].sum(dim=-1))
                all_ref_r24.append(ref_r24[r_idx].sum(dim=-1))

        std_c33 = torch.cat(all_c33).std().item()
        std_e2x = torch.cat(all_e2x).std().item()
        std_r24 = torch.cat(all_r24).std().item()

        ref_std_c33 = torch.cat(all_ref_c33).std().item()
        ref_std_e2x = torch.cat(all_ref_e2x).std().item()
        ref_std_r24 = torch.cat(all_ref_r24).std().item()

        if std_c33 > 1e-5:
            w_c33 *= (ref_std_c33 / std_c33)
        if std_e2x > 1e-5:
            w_e2x *= (ref_std_e2x / std_e2x)
        if std_r24 > 1e-5:
            w_r24 *= (ref_std_r24 / std_r24)
        print(f"Scaled group stds: c33 {std_c33:.2f}->{ref_std_c33:.2f}, e2x {std_e2x:.2f}->{ref_std_e2x:.2f}, r24 {std_r24:.2f}->{ref_std_r24:.2f}")

    if c33_overlap_scale != 1.0:
        w_c33 *= c33_overlap_scale
        print(f"Applied c33 spatial overlap scale: {c33_overlap_scale:.4f}")
    if e2x_overlap_scale != 1.0:
        w_e2x *= e2x_overlap_scale
        print(f"Applied e2x spatial overlap scale: {e2x_overlap_scale:.4f}")

    full_c33 = [int(round(float(v))) for v in w_c33.tolist()]
    full_e2x = [int(round(float(v))) for v in w_e2x.tolist()]
    full_r24 = [int(round(float(v))) for v in w_r24.tolist()]

    def clamp_i16(v):
        return max(-32767, min(32767, v))

    full_weights = [clamp_i16(v) for v in (full_c33 + full_e2x + full_r24)]
    assert len(full_weights) == TOTAL_WEIGHTS, f"Expected {TOTAL_WEIGHTS} weights, got {len(full_weights)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("/*\n")
        f.write("  File:       end_patterns_data.h\n")
        f.write("  Generated:  scripts/train_ltr_endgame.py\n")
        f.write("  Model:      Calibrated 3-Pattern Move Ordering Evaluator\n")
        f.write("  Geometry:   Corner 3x3 (19683), Edge+2X (59049), 2x4 Rect (6561)\n")
        f.write(f"  Total:      {TOTAL_WEIGHTS} entries (170.6 KB), 100% L2 cache resident\n")
        f.write(f"  Scale:      1 disc = {weight_scale} fixed-point units\n")
        f.write("*/\n\n")
        f.write("#ifndef END_PATTERNS_DATA_H\n")
        f.write("#define END_PATTERNS_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const int16_t end_pattern_weights[{TOTAL_WEIGHTS}] = {{\n")

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


def load_c_header(model: EndgamePatternModel, header_path: Path, weight_scale: float = 128.0):
    w_all = load_header_weights(header_path)
    w_c33 = w_all[0:19683] / weight_scale
    w_e2x = w_all[19683:19683+59049] / weight_scale
    w_r24 = w_all[19683+59049:TOTAL_WEIGHTS] / weight_scale

    with torch.no_grad():
        model.c33_emb.weight.copy_(w_c33.unsqueeze(-1))
        model.e2x_emb.weight.copy_(w_e2x.unsqueeze(-1))
        model.r24_emb.weight.copy_(w_r24.unsqueeze(-1))
    print(f"Loaded {TOTAL_WEIGHTS} initial weights from {header_path} successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train 3-pattern endgame move ordering model via hybrid Texel MSE + LTR distillation.")
    parser.add_argument("--data", type=Path, default=Path("data/ltr_exact_solved_15yr.pt"), help="Path to exact solved LTR dataset")
    parser.add_argument("--scores", type=Path, default=Path("data/ltr_exact_solved_15yr_scores.pt"), help="Path to exact solved scores tensor")
    parser.add_argument("--stages", type=str, default="7,8,9,10", help="Stages to include (default: 7,8,9,10 covering discs 40-58, 6-24 empties)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing (default: 0.05)")
    parser.add_argument("--temp", type=float, default=1.0, help="Student temperature (default: 1.0)")
    parser.add_argument("--mse-weight", type=float, default=0.02, help="Weight for Texel MSE disc difference loss (default: 0.02)")
    parser.add_argument("--weight-scale", type=float, default=128.0, help="Fixed-point scaling factor (default: 128.0)")
    parser.add_argument("--val-split", type=float, default=0.10, help="Validation fraction (default: 0.10)")
    parser.add_argument("--init-header", type=Path, default=None, help="Initial weights C header to fine-tune from")
    parser.add_argument("--cutoff-data", type=Path, default=None, help="Path to search cutoff dataset to mix in")
    parser.add_argument("--cutoff-ratio", type=float, default=0.5, help="Fraction of training samples from cutoff dataset")
    parser.add_argument("--ref-header", type=Path, default=Path("src/end_patterns_data.h"), help="Reference C header for group std normalization")
    parser.add_argument("--c33-overlap-scale", type=float, default=15.0 / 16.0, help="Spatial overlap calibration factor for c33 (default: 15/16 = 0.9375)")
    parser.add_argument("--e2x-overlap-scale", type=float, default=1.0, help="Spatial overlap calibration factor for e2x (default: 1.0)")
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

    scores_data = None
    if args.scores and args.scores.exists():
        loaded_scores = torch.load(args.scores, weights_only=False)
        if len(loaded_scores) == len(raw_data):
            print(f"Loading scores from {args.scores}...")
            scores_data = loaded_scores
        else:
            print(f"Scores length {len(loaded_scores)} != data length {len(raw_data)}, running pure LTR mode (mse_weight=0.0).")
            args.mse_weight = 0.0
    else:
        args.mse_weight = 0.0

    target_stages = set(int(s.strip()) for s in args.stages.split(",") if s.strip())
    filtered_samples = []
    filtered_scores = []
    for i, s in enumerate(raw_data):
        stage_val = s.get("stage", 7)
        if not target_stages or stage_val in target_stages:
            filtered_samples.append(s)
            if scores_data is not None and i < len(scores_data):
                filtered_scores.append(scores_data[i].item() if hasattr(scores_data[i], "item") else float(scores_data[i]))
            else:
                filtered_scores.append(0.0)

    print(f"Filtered {len(filtered_samples)} samples in stages {sorted(target_stages)}")
    has_scores_list = [scores_data is not None] * len(filtered_samples)

    if args.cutoff_data and args.cutoff_data.exists():
        print(f"Loading search cutoff dataset from {args.cutoff_data}...")
        cutoff_raw = torch.load(args.cutoff_data, weights_only=False)
        cutoff_filtered = []
        for s in cutoff_raw:
            stage_val = s.get("stage", 7)
            if not target_stages or stage_val in target_stages:
                cutoff_filtered.append(s)
        print(f"Loaded {len(cutoff_filtered)} cutoff samples matching stages {sorted(target_stages)}")

        import random
        random.seed(42)
        num_cutoffs = min(len(cutoff_filtered), int(len(filtered_samples) * args.cutoff_ratio / max(0.01, 1.0 - args.cutoff_ratio)))
        selected_cutoffs = random.sample(cutoff_filtered, num_cutoffs) if len(cutoff_filtered) > num_cutoffs else cutoff_filtered

        print(f"Mixing {len(filtered_samples)} WTHOR samples with {len(selected_cutoffs)} search cutoff samples...")
        filtered_samples = filtered_samples + selected_cutoffs
        filtered_scores = filtered_scores + [0.0] * len(selected_cutoffs)
        has_scores_list = has_scores_list + [False] * len(selected_cutoffs)

        combined = list(zip(filtered_samples, filtered_scores, has_scores_list))
        random.shuffle(combined)
        f_s, f_sc, f_h = zip(*combined)
        filtered_samples = list(f_s)
        filtered_scores = list(f_sc)
        has_scores_list = list(f_h)

    # Train / Val split
    split_idx = int((1.0 - args.val_split) * len(filtered_samples))
    train_samples = filtered_samples[:split_idx]
    train_scores = filtered_scores[:split_idx]
    train_has_scores = has_scores_list[:split_idx]
    val_samples = filtered_samples[split_idx:]
    val_scores = filtered_scores[split_idx:]
    val_has_scores = has_scores_list[split_idx:]
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    print("Building tensor datasets...")
    t0 = time.time()
    train_dataset = EndgameLTRDataset(train_samples, scores=train_scores, has_scores=train_has_scores, label_smoothing=args.label_smoothing)
    val_dataset = EndgameLTRDataset(val_samples, scores=val_scores, has_scores=val_has_scores, label_smoothing=args.label_smoothing)
    print(f"Dataset tensors built in {time.time() - t0:.2f}s")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = EndgamePatternModel().to(device)
    if args.init_header and args.init_header.exists():
        load_c_header(model, args.init_header, weight_scale=args.weight_scale)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-3)

    # Initial evaluation
    val_loss, val_ltr, val_mse, val_mae, val_top1, val_top2 = evaluate_model(model, val_loader, device, temp=args.temp, mse_weight=args.mse_weight)
    print(f"Epoch  0/{args.epochs} | Val Loss: {val_loss:.4f} (LTR: {val_ltr:.4f}, MSE: {val_mse:.2f}, MAE: {val_mae:.2f}d) | Val Top-1: {val_top1:.2f}% | Val Top-2: {val_top2:.2f}% (Untrained baseline)")

    best_val_top1 = 0.0
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        ep_start = time.time()

        for padded_indices, padded_targets, mask, best_indices, target_scores, has_scores in train_loader:
            padded_indices = padded_indices.to(device)
            padded_targets = padded_targets.to(device)
            mask = mask.to(device)
            best_indices = best_indices.to(device)
            target_scores = target_scores.to(device)
            has_scores = has_scores.to(device)

            optimizer.zero_grad()
            scores = model(padded_indices)
            best_scores = scores.gather(1, best_indices.unsqueeze(-1)).squeeze(-1)

            if has_scores.any() and args.mse_weight > 0.0:
                sc_best = best_scores[has_scores]
                sc_target = target_scores[has_scores]
                mse_loss = nn.functional.mse_loss(sc_best, sc_target)
            else:
                mse_loss = torch.tensor(0.0, device=device)

            masked_scores = scores.masked_fill(~mask, -1e9)
            log_probs = nn.functional.log_softmax(masked_scores / args.temp, dim=-1)
            ltr_loss = -(padded_targets * log_probs).sum(dim=-1).mean()

            loss = ltr_loss + args.mse_weight * mse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(best_indices)
            total_samples += len(best_indices)

        scheduler.step()
        train_loss = total_loss / max(1, total_samples)

        val_loss, val_ltr, val_mse, val_mae, val_top1, val_top2 = evaluate_model(model, val_loader, device, temp=args.temp, mse_weight=args.mse_weight)
        elapsed = time.time() - ep_start
        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (LTR: {val_ltr:.4f}, MSE: {val_mse:.2f}, MAE: {val_mae:.2f}d) | Val Top-1: {val_top1:.2f}% | Val Top-2: {val_top2:.2f}% | Time: {elapsed:.1f}s")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1

    print(f"\nTraining complete in {time.time() - start_train_time:.1f}s. Best Val Top-1 Acc: {best_val_top1:.2f}%")

    # Export weights to C header
    export_c_header(
        model,
        args.out_header,
        ref_header=args.ref_header,
        val_loader=val_loader,
        weight_scale=args.weight_scale,
        c33_overlap_scale=args.c33_overlap_scale,
        e2x_overlap_scale=args.e2x_overlap_scale,
    )


if __name__ == "__main__":
    main()
