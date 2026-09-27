#!/usr/bin/env python3
"""
Scientific Learning-to-Rank / Top-1 Accuracy Optimization for Bitboard Static Move Ordering.

Formulation:
  For each endgame position P with legal candidate moves {m_1, ..., m_k} and
  minimax-optimal best move m*, optimize feature weights w to maximize Top-1 Accuracy:
    Top-1 Acc = P( argmax_m S(P, m) == m* )

Score function with empties interaction:
  S(P, m) = sum_j (a_j + b_j * E) * x_j(P, m)
where E is remaining empty squares.

Loss: Group-wise Softmax Cross-Entropy (ListNet / Plackett-Luce) with Elastic Net regularization.
"""

import sys
import os
import argparse
import numpy as np
from scipy.optimize import minimize

FEATURE_NAMES = [
    "corner",
    "open_x",
    "open_c",
    "safe_x",
    "parity",
    "opp_mob",
    "opp_corner_moves",
    "frontier",
    "flips",
    "edge_stable"
]

def load_dataset(csv_path):
    print(f"Loading dataset from {csv_path}...")
    positions = {}
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        col_idx = {name: i for i, name in enumerate(header)}
        
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            pos_id = parts[col_idx["pos_id"]]
            empties = int(parts[col_idx["empties"]])
            sq = int(parts[col_idx["sq"]])
            label = int(parts[col_idx["label"]])
            
            feats = [float(parts[col_idx[name]]) for name in FEATURE_NAMES]
            
            if pos_id not in positions:
                positions[pos_id] = {
                    "empties": empties,
                    "sqs": [],
                    "labels": [],
                    "feats": []
                }
            positions[pos_id]["sqs"].append(sq)
            positions[pos_id]["labels"].append(label)
            positions[pos_id]["feats"].append(feats)

    # Filter positions: must have at least 2 moves and exactly one positive label
    valid_positions = []
    for pos_id, data in positions.items():
        labels = np.array(data["labels"])
        if len(labels) >= 2 and np.sum(labels == 1) == 1:
            data["feats"] = np.array(data["feats"], dtype=np.float64)
            data["best_idx"] = int(np.where(labels == 1)[0][0])
            valid_positions.append(data)

    print(f"Loaded {len(valid_positions)} valid search decision positions ({sum(len(p['sqs']) for p in valid_positions)} total candidate moves).")
    return valid_positions

def evaluate_top1(positions, score_fn):
    correct = 0
    total = len(positions)
    for p in positions:
        scores = score_fn(p)
        pred_idx = np.argmax(scores)
        if pred_idx == p["best_idx"]:
            correct += 1
    return correct / total if total > 0 else 0.0

def baseline_score_fn(p):
    # Current engine baseline:
    # move_score = 64 * parity - 128 * (opp_mob + opp_corner_moves)
    # in our feature vector:
    # index 4: parity
    # index 5: opp_mob
    # index 6: opp_corner_moves
    feats = p["feats"]
    return 64.0 * feats[:, 4] - 128.0 * (feats[:, 5] + feats[:, 6])

def fit_ranker(positions, use_interaction=True, l1_reg=0.01, l2_reg=0.001):
    num_base = len(FEATURE_NAMES)
    dim = num_base * 2 if use_interaction else num_base

    # Pre-extract matrices for fast vectorized loss computation
    pos_data = []
    for p in positions:
        E = float(p["empties"])
        base_x = p["feats"] # (K, num_base)
        if use_interaction:
            # Normalize E around 11 (midpoint of 8-14) to improve conditioning
            E_norm = (E - 11.0) / 3.0
            inter_x = base_x * E_norm
            x = np.hstack([base_x, inter_x])
        else:
            x = base_x
        pos_data.append((x, p["best_idx"]))

    def loss_and_grad(w):
        total_loss = 0.0
        grad = np.zeros_like(w)
        N = len(pos_data)

        for x, best_idx in pos_data:
            scores = x @ w
            # Stable softmax
            max_s = np.max(scores)
            exp_s = np.exp(scores - max_s)
            sum_exp = np.sum(exp_s)
            probs = exp_s / sum_exp

            # Negative log likelihood of best move
            total_loss -= np.log(probs[best_idx] + 1e-12)

            # Gradient: E_{p}[x] - x_{best}
            # d/dw (-log p_{best}) = sum_k p_k x_k - x_{best}
            grad += (probs @ x) - x[best_idx]

        loss = total_loss / N
        grad = grad / N

        # Regularization (smooth L1 / Huber or L2)
        # Using pseudo-Huber for smooth L1: sqrt(w^2 + eps^2)
        eps = 1e-4
        l1_penalty = l1_reg * np.sum(np.sqrt(w**2 + eps**2))
        l1_grad = l1_reg * w / np.sqrt(w**2 + eps**2)

        l2_penalty = 0.5 * l2_reg * np.sum(w**2)
        l2_grad = l2_reg * w

        return loss + l1_penalty + l2_penalty, grad + l1_grad + l2_grad

    # Initial weights: initialize close to baseline
    w0 = np.zeros(dim)
    w0[4] = 0.5   # parity
    w0[5] = -1.0  # opp_mob
    w0[6] = -1.0  # opp_corner_moves

    print(f"Optimizing {dim} weights via L-BFGS-B (ListNet loss with L1={l1_reg}, L2={l2_reg})...")
    res = minimize(loss_and_grad, w0, method="L-BFGS-B", jac=True,
                   options={"maxiter": 200, "disp": True})
    w_opt = res.x
    return w_opt, dim

def main():
    parser = argparse.ArgumentParser(description="Learning-to-Rank for Okapi Static Move Ordering")
    parser.add_argument("csv_file", help="Path to training CSV file")
    parser.add_argument("--l1", type=float, default=0.005, help="L1 regularization parameter")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization parameter")
    parser.add_argument("--no-interaction", action="store_true", help="Disable empties interaction terms")
    args = parser.parse_args()

    positions = load_dataset(args.csv_file)
    if len(positions) == 0:
        print("Error: No valid positions found.")
        sys.exit(1)

    # Train/Validation split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(positions))
    split = int(0.8 * len(positions))
    train_pos = [positions[i] for i in indices[:split]]
    val_pos = [positions[i] for i in indices[split:]]

    print(f"\nTrain set: {len(train_pos)} positions | Val set: {len(val_pos)} positions")

    # Evaluate Baseline Top-1
    base_train_acc = evaluate_top1(train_pos, baseline_score_fn)
    base_val_acc = evaluate_top1(val_pos, baseline_score_fn)
    print(f"\n=== Baseline Static Ordering (parity=64, opp_mob=-128) ===")
    print(f"  Train Top-1 Accuracy: {base_train_acc * 100:.2f}%")
    print(f"  Val   Top-1 Accuracy: {base_val_acc * 100:.2f}%")

    # Fit Ranker
    use_interaction = not args.no_interaction
    w_opt, dim = fit_ranker(train_pos, use_interaction=use_interaction, l1_reg=args.l1, l2_reg=args.l2)

    # Evaluation function for optimized model
    def opt_score_fn(p):
        E = float(p["empties"])
        base_x = p["feats"]
        if use_interaction:
            E_norm = (E - 11.0) / 3.0
            inter_x = base_x * E_norm
            x = np.hstack([base_x, inter_x])
        else:
            x = base_x
        return x @ w_opt

    opt_train_acc = evaluate_top1(train_pos, opt_score_fn)
    opt_val_acc = evaluate_top1(val_pos, opt_score_fn)

    print(f"\n=== Optimized Learning-to-Rank Model ===")
    print(f"  Train Top-1 Accuracy: {opt_train_acc * 100:.2f}%  (Δ: {(opt_train_acc - base_train_acc)*100:+.2f}%)")
    print(f"  Val   Top-1 Accuracy: {opt_val_acc * 100:.2f}%  (Δ: {(opt_val_acc - base_val_acc)*100:+.2f}%)")

    # Feature Importance Breakdown
    num_base = len(FEATURE_NAMES)
    print("\n=== Learned Feature Weights ===")
    print(f"{'Feature':<20} {'Base Weight (a_j)':<18} {'Empties Sens. (b_j)':<20} {'Net @ E=10':<12} {'Net @ E=14':<12}")
    print("-" * 84)

    # Calibration scale: calibrate so that opp_mob is roughly -128
    mob_idx = 5
    mob_base = abs(w_opt[mob_idx])
    scale = 128.0 / mob_base if mob_base > 1e-4 else 128.0

    table_c_code = []
    for i, name in enumerate(FEATURE_NAMES):
        a = w_opt[i]
        b = w_opt[num_base + i] if use_interaction else 0.0
        
        # At E=10: E_norm = (10-11)/3 = -0.333
        # At E=14: E_norm = (14-11)/3 = +1.000
        net_10 = scale * (a + b * (-0.333))
        net_14 = scale * (a + b * (1.000))
        
        print(f"{name:<20} {scale * a:<18.1f} {scale * b:<20.1f} {net_10:<12.1f} {net_14:<12.1f}")

    # Generate C table for depths 8 to 14
    print("\n=== Generated C Lookup Table (Depths 8 to 14) ===")
    print("static const int static_order_weights[15][10] = {")
    for E in range(15):
        if E < 8:
            print(f"  /* E={E:2d} */ {{ 0 }},")
        else:
            E_norm = (E - 11.0) / 3.0 if use_interaction else 0.0
            row_weights = []
            for i in range(num_base):
                a = w_opt[i]
                b = w_opt[num_base + i] if use_interaction else 0.0
                val = int(round(scale * (a + b * E_norm)))
                row_weights.append(val)
            weights_str = ", ".join(f"{v:6d}" for v in row_weights)
            print(f"  /* E={E:2d} */ {{ {weights_str} }},")
    print("};")

if __name__ == "__main__":
    main()
