#!/usr/bin/env python3
"""
scripts/train_move_ordering_nn.py - Neural Network Exploration for Bitboard Move Ordering.

Investigates feature interactions, non-linear representations, and depth sensitivity
for endgame static move ordering (empties E in [8, 16]).
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add scripts directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from distill_move_ordering_tree import (
    parse_wthor_games, OthelloBoard, popcount, generate_all_c,
    INNER_MASK, MASK_64, CORNER_MASK, CORNERS, X_MAP, C_MAP,
    BLACK, WHITE, EMPTY, get_quadrant
)

FEATURE_NAMES = [
    "raw_opp_mob",        # 0: direct legal replies of opponent
    "opp_corner_moves",   # 1: opponent moves accessing corners
    "pot_mob_opp",        # 2: frontier empty squares of opponent (touching new_opp_bits)
    "pot_mob_my",         # 3: frontier empty squares of player (touching bb_flips)
    "is_open_x",          # 4: 1 if X-square and corner empty
    "is_open_c_poisoned", # 5: 1 if C-square, corner empty, and opp_corner_moves > 0
    "wipeout",            # 6: 1 if raw_opp_mob == 0
    "parity",             # 7: 1 if quadrant has odd empties
    "is_corner",          # 8: 1 if move is on corner
    "flips",              # 9: number of discs flipped
    "empties",            # 10: remaining empty squares E (8-16)
]

def bitboard_neighbors(b):
    not_file_a = 0xFEFEFEFEFEFEFEFE
    not_file_h = 0x7F7F7F7F7F7F7F7F
    h_east = (((b << 1) & MASK_64) | (b >> 7) | ((b << 9) & MASK_64)) & not_file_a
    h_west = ((b >> 1) | ((b << 7) & MASK_64) | (b >> 9)) & not_file_h
    v_span = ((b << 8) & MASK_64) | (b >> 8)
    return (h_east | h_west | v_span) & MASK_64

def bitboard_frontier(discs, empty):
    return popcount(bitboard_neighbors(discs) & empty)

def extract_nn_dataset(wthor_dir="data/wthor", min_empties=8, max_empties=16, max_games=None):
    wthor_files = sorted(Path(wthor_dir).glob("WTH_20*.wtb"))
    print(f"Scanning {len(wthor_files)} WTHOR files in {wthor_dir}...")
    positions = []
    total_games = 0
    exact_games = 0
    t0 = time.time()

    for wf in wthor_files:
        for real_score, theo_score, moves in parse_wthor_games(wf):
            total_games += 1
            if max_games and total_games >= max_games:
                break
            if real_score != theo_score:
                continue
            exact_games += 1

            board = OthelloBoard()
            for played_r, played_c in moves:
                if board.side_to_move == EMPTY:
                    break
                empties = 64 - board.disc_count
                if min_empties <= empties <= max_empties:
                    legals = board.legal_moves(board.side_to_move)
                    if len(legals) >= 2 and (played_r, played_c) in legals:
                        # Extract bitboards
                        my_bits = 0
                        opp_bits = 0
                        quad_empties = [0, 0, 0, 0]
                        for r in range(8):
                            for c in range(8):
                                sq_bit = 1 << (r * 8 + c)
                                if board.board[r][c] == board.side_to_move:
                                    my_bits |= sq_bit
                                elif board.board[r][c] != EMPTY:
                                    opp_bits |= sq_bit
                                else:
                                    q = get_quadrant(r, c)
                                    quad_empties[q] += 1

                        cand_features = []
                        best_idx = legals.index((played_r, played_c))

                        for m_r, m_c in legals:
                            flips_list = board.get_flips(m_r, m_c, board.side_to_move)
                            bb_flips = sum(1 << (fr * 8 + fc) for fr, fc in flips_list)
                            # Add played square to bb_flips
                            bb_flips |= (1 << (m_r * 8 + m_c))
                            new_opp_bits = opp_bits & ~bb_flips
                            empty_bb = ~(bb_flips | new_opp_bits) & MASK_64

                            # Opponent direct legal replies
                            opp_moves = generate_all_c(new_opp_bits, bb_flips)
                            raw_opp_mob = popcount(opp_moves)
                            opp_corner_moves = popcount(opp_moves & CORNER_MASK)

                            # Potential mobility
                            pot_mob_opp = bitboard_frontier(new_opp_bits, empty_bb)
                            pot_mob_my = bitboard_frontier(bb_flips, empty_bb)

                            # Board shapes
                            is_corner = 1 if (m_r, m_c) in CORNERS else 0
                            
                            is_open_x = 0
                            if (m_r, m_c) in X_MAP:
                                cr, cc = X_MAP[(m_r, m_c)]
                                if board.board[cr][cc] == EMPTY:
                                    is_open_x = 1

                            is_open_c_poisoned = 0
                            if (m_r, m_c) in C_MAP and opp_corner_moves > 0:
                                cr, cc = C_MAP[(m_r, m_c)]
                                if board.board[cr][cc] == EMPTY:
                                    is_open_c_poisoned = 1

                            wipeout = 1 if raw_opp_mob == 0 else 0
                            q = get_quadrant(m_r, m_c)
                            parity = 1 if (quad_empties[q] % 2 == 1) else 0

                            feats = [
                                float(raw_opp_mob),
                                float(opp_corner_moves),
                                float(pot_mob_opp),
                                float(pot_mob_my),
                                float(is_open_x),
                                float(is_open_c_poisoned),
                                float(wipeout),
                                float(parity),
                                float(is_corner),
                                float(len(flips_list)),
                                float(empties),
                            ]
                            cand_features.append(feats)

                        positions.append({
                            "empties": empties,
                            "moves": legals,
                            "best_idx": best_idx,
                            "features": np.array(cand_features, dtype=np.float32)
                        })

                if not board.make_move(played_r, played_c):
                    break

    dt = time.time() - t0
    print(f"Extracted {len(positions)} decision positions ({sum(len(p['moves']) for p in positions)} moves) from {exact_games} exact games in {dt:.2f}s.")
    return positions


class MoveOrderingNN(nn.Module):
    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: [K, in_features] -> [K, 1] -> [K]
        return self.net(x).squeeze(-1)


def evaluate_model_top1(positions, model_fn):
    correct = 0
    total = len(positions)
    by_depth = {e: [0, 0] for e in range(8, 17)}
    for p in positions:
        scores = model_fn(p)
        pred_idx = int(np.argmax(scores))
        e = p["empties"]
        if e in by_depth:
            by_depth[e][1] += 1
            if pred_idx == p["best_idx"]:
                by_depth[e][0] += 1
        if pred_idx == p["best_idx"]:
            correct += 1
    total_acc = correct / total if total > 0 else 0.0
    depth_acc = {e: (by_depth[e][0] / by_depth[e][1]) if by_depth[e][1] > 0 else 0.0 for e in by_depth if by_depth[e][1] > 0}
    return total_acc, depth_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wthor-dir", default="data/wthor")
    parser.add_argument("--max-games", type=int, default=15000)
    parser.add_argument("--save-cache", default="data/nn_move_ordering_cache.npz")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()

    if os.path.exists(args.save_cache):
        print(f"Loading cached positions from {args.save_cache}...")
        cache = np.load(args.save_cache, allow_pickle=True)
        positions = list(cache["positions"])
        print(f"Loaded {len(positions)} positions from cache.")
    else:
        positions = extract_nn_dataset(args.wthor_dir, min_empties=8, max_empties=16, max_games=args.max_games)
        if args.save_cache:
            print(f"Saving {len(positions)} positions to {args.save_cache}...")
            np.savez_compressed(args.save_cache, positions=positions)

    # Train / Val split (80/20)
    np.random.seed(42)
    torch.manual_seed(42)
    perm = np.random.permutation(len(positions))
    split = int(0.8 * len(positions))
    train_pos = [positions[i] for i in perm[:split]]
    val_pos = [positions[i] for i in perm[split:]]
    print(f"Train positions: {len(train_pos)} | Val positions: {len(val_pos)}")

    # 1. Baseline Evaluation: Okapi Master heuristic
    # move_score = -128 * (raw_opp_mob + opp_corner_moves) + 64 * parity - 128 * is_open_x - 256 * is_open_c_poisoned + 512 * wipeout
    def baseline_score(p):
        f = p["features"]
        # f: [raw_opp_mob, opp_corner_moves, pot_mob_opp, pot_mob_my, is_open_x, is_open_c_poisoned, wipeout, parity, is_corner, flips, empties]
        mob = -128.0 * (f[:, 0] + f[:, 1])
        parity = 64.0 * f[:, 7]
        open_x = -128.0 * f[:, 4]
        open_c = -256.0 * f[:, 5]
        wipeout = 512.0 * f[:, 6]
        return mob + parity + open_x + open_c + wipeout

    base_acc, base_depth = evaluate_model_top1(val_pos, baseline_score)
    print(f"\n[Baseline Okapi Master]: Overall Val Top-1: {base_acc*100:.2f}%")
    for e in range(8, 17):
        if e in base_depth:
            print(f"  E = {e:2d}: {base_depth[e]*100:.2f}%")

    # 2. Train PyTorch NN
    in_dim = len(FEATURE_NAMES)
    model = MoveOrderingNN(in_dim, hidden_dim=32)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining MoveOrderingNN (MLP 11 -> 32 -> 16 -> 1)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        # Shuffle train positions
        np.random.shuffle(train_pos)
        
        # Mini-batch accumulation
        optimizer.zero_grad()
        batch_loss = 0.0
        step_count = 0

        for idx, p in enumerate(train_pos):
            x = torch.tensor(p["features"], dtype=torch.float32)
            target = torch.tensor(p["best_idx"], dtype=torch.long)
            logits = model(x).unsqueeze(0)  # [1, K]
            loss = loss_fn(logits, target.unsqueeze(0))
            batch_loss = batch_loss + loss

            if (idx + 1) % args.batch_size == 0 or (idx + 1) == len(train_pos):
                batch_loss = batch_loss / (args.batch_size if (idx + 1) % args.batch_size == 0 else len(train_pos) % args.batch_size)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += batch_loss.item()
                batch_loss = 0.0
                step_count += 1
        model.eval()
        def nn_score(p):
            with torch.no_grad():
                x = torch.tensor(p["features"], dtype=torch.float32)
                return model(x).detach().cpu().numpy()
        val_acc, val_depth = evaluate_model_top1(val_pos, nn_score)
        print(f"Epoch {epoch:2d}/{args.epochs:2d} - Loss: {total_loss/step_count:.4f} - Val Top-1: {val_acc*100:.2f}%")


    # 3. Final In-depth Depth Breakdown & Feature Importance Analysis
    print("\n=== Model Comparison: Top-1 Accuracy by Remaining Empties (E) ===")
    print(f"{'Depth (E)':<10} {'Positions':<10} {'Baseline':<12} {'Neural Net':<12} {'Δ Top-1':<10}")
    print("-" * 56)
    val_counts = {e: sum(1 for p in val_pos if p["empties"] == e) for e in range(8, 17)}
    for e in range(8, 17):
        if e in val_depth and e in base_depth:
            d_acc = val_depth[e] * 100
            b_acc = base_depth[e] * 100
            diff = d_acc - b_acc
            print(f"E = {e:<6d} {val_counts[e]:<10d} {b_acc:6.2f}%      {d_acc:6.2f}%      {diff:+6.2f}%")
    print(f"{'OVERALL':<10} {len(val_pos):<10d} {base_acc*100:6.2f}%      {val_acc*100:6.2f}%      {(val_acc - base_acc)*100:+6.2f}%")

    # 4. Feature Sensitivity & Permutation Importance Analysis
    print("\n=== Neural Network Feature Importance (Permutation Sensitivity) ===")
    print("Measuring degradation in Top-1 Accuracy when each feature is shuffled across candidate moves:")
    feat_importance = {}
    for f_idx, f_name in enumerate(FEATURE_NAMES):
        corrupted_correct = 0
        for p in val_pos:
            feats = p["features"].copy()
            # Permute feature f_idx across candidate moves of this position
            np.random.shuffle(feats[:, f_idx])
            x = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                scores = model(x).detach().cpu().numpy()
            if np.argmax(scores) == p["best_idx"]:
                corrupted_correct += 1
        corrupted_acc = corrupted_correct / len(val_pos)
        importance_drop = (val_acc - corrupted_acc) * 100
        feat_importance[f_name] = importance_drop
        print(f"  {f_name:<20}: drop = {importance_drop:+.2f}%")

    # 5. Shallow (8-12) vs Deep (14-16) Feature Importance Comparison
    print("\n=== Feature Sensitivity: Shallow (E=8..12) vs Deep (E=14..16) ===")
    for subset_name, subset_e in [("Shallow (E=8..12)", range(8, 13)), ("Deep (E=14..16)", range(14, 17))]:
        sub_pos = [p for p in val_pos if p["empties"] in subset_e]
        sub_base_acc, _ = evaluate_model_top1(sub_pos, nn_score)
        print(f"\n[{subset_name} - N={len(sub_pos)}]: Baseline NN Top-1 = {sub_base_acc*100:.2f}%")
        for f_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            f_name = FEATURE_NAMES[f_idx]
            corrupted_correct = 0
            for p in sub_pos:
                feats = p["features"].copy()
                np.random.shuffle(feats[:, f_idx])
                x = torch.tensor(feats, dtype=torch.float32)
                with torch.no_grad():
                    scores = model(x).detach().cpu().numpy()
                if np.argmax(scores) == p["best_idx"]:
                    corrupted_correct += 1
            drop = (sub_base_acc - (corrupted_correct / len(sub_pos))) * 100
            print(f"  {f_name:<20}: drop = {drop:+.2f}%")

if __name__ == "__main__":
    main()
