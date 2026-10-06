#!/usr/bin/env python3
"""
scripts/distill_move_ordering_tree.py - Distilled Move Ordering Decision Tree and Rules.

Formulates knowledge distillation from 49k exact endgame positions:
1. Trains shallow interpretable Decision Trees (CART depths 2, 3, 4, 5)
2. Formulates compact integer scoring rules in C (< 20ns execution)
3. Outputs comprehensive Top-1 accuracy comparisons by depth (E=8..16)
4. Generates C code for src/end.c
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text, _tree

# Add scripts directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

FEATURE_NAMES = [
    "raw_opp_mob",        # 0: direct legal replies of opponent
    "opp_corner_moves",   # 1: opponent moves accessing corners
    "pot_mob_opp",        # 2: frontier empty squares of opponent
    "pot_mob_my",         # 3: frontier empty squares of player (potential opponent replies)
    "is_open_x",          # 4: 1 if X-square and corner empty
    "is_open_c_poisoned", # 5: 1 if C-square, corner empty, and opp_corner_moves > 0
    "wipeout",            # 6: 1 if raw_opp_mob == 0
    "parity",             # 7: 1 if quadrant has odd empties
    "is_corner",          # 8: 1 if move is on corner
    "flips",              # 9: number of discs flipped
    "empties",            # 10: remaining empty squares E (8-16)
]

def evaluate_top1(positions, score_fn):
    correct = 0
    total = len(positions)
    by_depth = {e: [0, 0] for e in range(8, 17)}
    for p in positions:
        scores = score_fn(p)
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

def evaluate_baseline(p):
    # Okapi Master Baseline:
    # 64 * parity - 128 * (raw_opp_mob + opp_corner_moves) - 128 * is_open_x - 256 * is_open_c_poisoned + 512 * wipeout
    f = p["features"]
    return (
        64.0 * f[:, 7]
        - 128.0 * (f[:, 0] + f[:, 1])
        - 128.0 * f[:, 4]
        - 256.0 * f[:, 5]
        + 512.0 * f[:, 6]
    )

def evaluate_distilled_rules_v1(p):
    """
    Distilled Rules V1: Add is_corner bonus and pot_mob_my penalty.
    """
    f = p["features"]
    raw_mob = f[:, 0]
    opp_corner = f[:, 1]
    pot_mob_my = f[:, 3]
    is_open_x = f[:, 4]
    is_open_c = f[:, 5]
    wipeout = f[:, 6]
    parity = f[:, 7]
    is_corner = f[:, 8]
    empties = f[:, 10]

    score = np.zeros(len(f), dtype=np.float32)
    score += 512.0 * wipeout
    score += 256.0 * is_corner
    score += 64.0 * parity
    score -= 128.0 * raw_mob
    score -= 256.0 * opp_corner
    score -= 128.0 * is_open_x
    score -= 256.0 * is_open_c
    score -= 16.0 * pot_mob_my
    return score

def evaluate_distilled_rules_v2(p):
    """
    Distilled Rules V2: Optimized integer weights tuned for deep positions.
    Scaled potential mobility and corner control.
    """
    f = p["features"]
    raw_mob = f[:, 0]
    opp_corner = f[:, 1]
    pot_mob_my = f[:, 3]
    is_open_x = f[:, 4]
    is_open_c = f[:, 5]
    wipeout = f[:, 6]
    parity = f[:, 7]
    is_corner = f[:, 8]

    # Integer scoring in units of 1:
    # Corner captures are decisive (+512)
    # Opponent wipeout is decisive (+512)
    # Severe penalty for giving corner (-256)
    # Severe penalty for open X (-256)
    # Opponent direct replies (-128 per reply)
    # Opponent potential replies (-16 per potential frontier square)
    # Parity (+64)
    return (
        512.0 * wipeout
        + 512.0 * is_corner
        - 256.0 * opp_corner
        - 256.0 * is_open_x
        - 256.0 * is_open_c
        - 128.0 * raw_mob
        - 16.0 * pot_mob_my
        + 64.0 * parity
    )

def generate_c_tree_code(tree, feature_names):
    t = tree.tree_
    lines = []
    lines.append("/* Distilled Move Ordering Decision Tree (Depth %d) */" % tree.max_depth)
    lines.append("static INLINE int")
    lines.append("distilled_tree_score( int raw_opp_mob, int opp_corner_moves, int pot_mob_my,")
    lines.append("                     int is_open_x, int is_open_c_poisoned, int wipeout,")
    lines.append("                     int parity, int is_corner, int empties ) {")

    def recurse(node, depth):
        indent = "  " * depth
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[t.feature[node]]
            threshold = t.threshold[node]
            lines.append(f"{indent}if ( {name} <= {threshold:.1f}f ) {{")
            recurse(t.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(t.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            val = int(round(t.value[node][0][0] * 1000.0))
            lines.append(f"{indent}return {val};")

    recurse(0, 1)
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/nn_move_ordering_cache.npz")
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        print(f"Error: cache file {args.cache} does not exist. Run scripts/train_move_ordering_nn.py first.")
        sys.exit(1)

    print(f"Loading {args.cache}...")
    cache = np.load(args.cache, allow_pickle=True)
    positions = list(cache["positions"])

    # Train / Val split (80/20) matching NN
    np.random.seed(42)
    perm = np.random.permutation(len(positions))
    split = int(0.8 * len(positions))
    train_pos = [positions[i] for i in perm[:split]]
    val_pos = [positions[i] for i in perm[split:]]

    print(f"Loaded {len(positions)} positions | Train: {len(train_pos)} | Val: {len(val_pos)}")

    # 1. Baseline Evaluation
    val_base_acc, val_base_depth = evaluate_top1(val_pos, evaluate_baseline)
    print(f"\n[Baseline Okapi Master]: Val Top-1 = {val_base_acc*100:.2f}%")

    # 2. Evaluate Distilled Rules V1 and V2
    val_v1_acc, val_v1_depth = evaluate_top1(val_pos, evaluate_distilled_rules_v1)
    print(f"[Distilled Rules V1]  : Val Top-1 = {val_v1_acc*100:.2f}% (Δ: {(val_v1_acc - val_base_acc)*100:+.2f}%)")

    val_v2_acc, val_v2_depth = evaluate_top1(val_pos, evaluate_distilled_rules_v2)
    print(f"[Distilled Rules V2]  : Val Top-1 = {val_v2_acc*100:.2f}% (Δ: {(val_v2_acc - val_base_acc)*100:+.2f}%)")

    # 3. Train CART Decision Trees
    X_train = []
    y_train = []
    for p in train_pos:
        K = len(p["moves"])
        b_idx = p["best_idx"]
        for idx in range(K):
            X_train.append(p["features"][idx])
            y_train.append(1.0 if idx == b_idx else -1.0 / (K - 1))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    best_tree = None
    best_tree_acc = 0.0
    best_depth = 0

    print("\n--- Decision Tree (CART) Distillation Sweep ---")
    for depth in [2, 3, 4, 5]:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=100, random_state=42)
        tree.fit(X_train, y_train)

        def tree_score(p, m=tree):
            return m.predict(p["features"])

        t_acc, _ = evaluate_top1(train_pos, tree_score)
        v_acc, v_depth = evaluate_top1(val_pos, tree_score)
        print(f"  Tree Depth {depth}: Train Top-1 = {t_acc*100:.2f}%, Val Top-1 = {v_acc*100:.2f}% (Δ vs Base: {(v_acc - val_base_acc)*100:+.2f}%)")
        if v_acc > best_tree_acc:
            best_tree_acc = v_acc
            best_tree = tree
            best_depth = depth

    print(f"\nOptimal Distilled Decision Tree (Depth {best_depth}): Val Top-1 = {best_tree_acc*100:.2f}%")
    print("\n=== Distilled Tree Structure ===")
    print(export_text(best_tree, feature_names=FEATURE_NAMES))

    def best_tree_score(p):
        return best_tree.predict(p["features"])
    _, tree_val_depth = evaluate_top1(val_pos, best_tree_score)

    # 4. In-depth Depth Breakdown Table
    print("\n=== Depth Breakdown (Top-1 Accuracy by Remaining Empties E) ===")
    print(f"{'Depth (E)':<10} {'Positions':<10} {'Baseline':<12} {'Distilled V2':<14} {'Distilled Tree':<16}")
    print("-" * 64)
    val_by_e = {e: sum(1 for p in val_pos if p["empties"] == e) for e in range(8, 17)}
    for e in range(8, 17):
        print(f"E = {e:<6d} {val_by_e[e]:<10d} {val_base_depth[e]*100:6.2f}%      {val_v2_depth[e]*100:6.2f}%        {tree_val_depth[e]*100:6.2f}%")
    print(f"{'OVERALL':<10} {len(val_pos):<10d} {val_base_acc*100:6.2f}%      {val_v2_acc*100:6.2f}%        {best_tree_acc*100:6.2f}%")

    # 5. Generate C Code
    print("\n=== Generated C Code for Distilled Decision Tree ===")
    print(generate_c_tree_code(best_tree, FEATURE_NAMES))

if __name__ == "__main__":
    main()
