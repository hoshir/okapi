#!/usr/bin/env python3
"""
scripts/distill_move_ordering_tree.py - Distilled Move Ordering vs Classical Hand-Crafted Move Ordering.

Formulates knowledge distillation for endgame static move ordering:
Extracts exact-solved endgame decision points (8 <= empties <= 14) from WTHOR championship games
where games were played with zero theoretical error (real_score == theo_score).

Evaluates:
  1. Classical Hand-Crafted Hierarchical Move Ordering (ORDR-002: parity + X-square penalty - opp_mob)
  2. Shallow Interpretable Decision Trees (CART depths 2, 3, 4, 5)
  3. Empirical Distilled Lexicographical Priority Rule Lists
Generates C code for the best distilled model and outputs comprehensive Top-1 accuracy comparisons.
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np

# Bitboard helpers
INNER_MASK = 0x7E7E7E7E7E7E7E7E
MASK_64 = 0xFFFFFFFFFFFFFFFF
CORNER_MASK = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

# Coordinates
CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
X_MAP = {
    (1, 1): (0, 0),
    (1, 6): (0, 7),
    (6, 1): (7, 0),
    (6, 6): (7, 7),
}
C_MAP = {
    (0, 1): (0, 0), (1, 0): (0, 0),
    (0, 6): (0, 7), (1, 7): (0, 7),
    (6, 0): (7, 0), (7, 1): (7, 0),
    (6, 7): (7, 7), (7, 6): (7, 7),
}

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

BLACK = 1
WHITE = 2
EMPTY = 0

def get_quadrant(r, c):
    return (0 if r < 4 else 2) + (0 if c < 4 else 1)

def popcount(x):
    return bin(x).count("1")

def generate_all_c(my_bits, opp_bits):
    opp_inner = opp_bits & INNER_MASK
    moves = 0
    for o, shift in [(opp_inner, 1), (opp_bits, 8), (opp_inner, 7), (opp_inner, 9)]:
        # Right shifts
        flip = (my_bits >> shift) & o
        flip |= (flip >> shift) & o
        adj = o & (o >> shift)
        flip |= (flip >> (2 * shift)) & adj
        flip |= (flip >> (2 * shift)) & adj
        moves |= flip >> shift
        
        # Left shifts
        flip = ((my_bits << shift) & MASK_64) & o
        flip |= ((flip << shift) & MASK_64) & o
        adj = o & ((o << shift) & MASK_64)
        flip |= ((flip << (2 * shift)) & MASK_64) & adj
        flip |= ((flip << (2 * shift)) & MASK_64) & adj
        moves |= (flip << shift) & MASK_64
        
    moves &= ~(my_bits | opp_bits) & MASK_64
    return moves

class OthelloBoard:
    def __init__(self):
        self.board = [[EMPTY] * 8 for _ in range(8)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.side_to_move = BLACK
        self.disc_count = 4

    def get_flips(self, r: int, c: int, color: int):
        if self.board[r][c] != EMPTY:
            return []
        opp = WHITE if color == BLACK else BLACK
        all_flips = []
        for dr, dc in DIRECTIONS:
            flips = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == opp:
                flips.append((nr, nc))
                nr += dr
                nc += dc
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == color and flips:
                all_flips.extend(flips)
        return all_flips

    def legal_moves(self, color: int):
        moves = []
        for r in range(8):
            for c in range(8):
                if self.get_flips(r, c, color):
                    moves.append((r, c))
        return moves

    def make_move(self, r: int, c: int) -> bool:
        flips = self.get_flips(r, c, self.side_to_move)
        if not flips:
            return False
        self.board[r][c] = self.side_to_move
        for fr, fc in flips:
            self.board[fr][fc] = self.side_to_move
        self.disc_count += 1
        opp = WHITE if self.side_to_move == BLACK else BLACK

        if self.legal_moves(opp):
            self.side_to_move = opp
        elif not self.legal_moves(self.side_to_move):
            self.side_to_move = EMPTY
        return True

def parse_wthor_games(wthor_path):
    games = []
    with open(wthor_path, "rb") as f:
        header = f.read(16)
        if len(header) < 16:
            return games
        n_games = header[4] | (header[5] << 8) | (header[6] << 16) | (header[7] << 24)
        record_size = 68
        for _ in range(n_games):
            rec = f.read(record_size)
            if len(rec) < record_size:
                break
            real_score = rec[6]
            theo_score = rec[7]
            moves = []
            for b in rec[8:]:
                if b == 0:
                    break
                col = (b % 10) - 1
                row = (b // 10) - 1
                moves.append((row, col))
            games.append((real_score, theo_score, moves))
    return games

FEATURE_NAMES = [
    "is_corner",       # 0: 1 if corner, 0 otherwise
    "is_open_x",       # 1: 1 if X-square and adjacent corner empty, 0 otherwise
    "is_open_c",       # 2: 1 if C-square and adjacent corner empty, 0 otherwise
    "is_safe_x",       # 3: 1 if X-square and adjacent corner occupied, 0 otherwise
    "parity",          # 4: 1 if quadrant has odd empties, 0 otherwise
    "raw_opp_mob",     # 5: Opponent legal replies flipping from bb_flips
    "opp_corner_moves",# 6: Opponent corner replies
    "weighted_mob",    # 7: 128 * (raw_opp_mob + opp_corner_moves)
    "flips",           # 8: Number of discs flipped
    "empties"          # 9: Remaining empty squares (8-14)
]

def extract_dataset(wthor_dir="data/wthor", min_empties=8, max_empties=14, max_games=None):
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
                            new_opp_bits = opp_bits & ~bb_flips

                            # Generate opponent moves flipping from bb_flips
                            opp_moves = generate_all_c(new_opp_bits, bb_flips)
                            raw_opp_mob = popcount(opp_moves)
                            opp_corner_moves = popcount(opp_moves & CORNER_MASK)
                            weighted_mob = 128 * (raw_opp_mob + opp_corner_moves)

                            # Static features
                            is_corner = 1 if (m_r, m_c) in CORNERS else 0
                            
                            is_open_x = 0
                            is_safe_x = 0
                            if (m_r, m_c) in X_MAP:
                                cr, cc = X_MAP[(m_r, m_c)]
                                if board.board[cr][cc] == EMPTY:
                                    is_open_x = 1
                                else:
                                    is_safe_x = 1

                            is_open_c = 0
                            if (m_r, m_c) in C_MAP:
                                cr, cc = C_MAP[(m_r, m_c)]
                                if board.board[cr][cc] == EMPTY:
                                    is_open_c = 1

                            q = get_quadrant(m_r, m_c)
                            parity = 1 if (quad_empties[q] % 2 == 1) else 0

                            feats = [
                                is_corner,
                                is_open_x,
                                is_open_c,
                                is_safe_x,
                                parity,
                                raw_opp_mob,
                                opp_corner_moves,
                                weighted_mob,
                                len(flips_list),
                                empties
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
    print(f"Extracted {len(positions)} decision positions ({sum(len(p['moves']) for p in positions)} moves) from {exact_games} exact games ({total_games} total) in {dt:.2f}s.")
    return positions

def evaluate_top1(positions, score_fn):
    correct = 0
    total = len(positions)
    by_depth = {e: [0, 0] for e in range(8, 15)}
    for p in positions:
        scores = score_fn(p)
        pred_idx = int(np.argmax(scores))
        e = p["empties"]
        by_depth[e][1] += 1
        if pred_idx == p["best_idx"]:
            correct += 1
            by_depth[e][0] += 1
    total_acc = correct / total if total > 0 else 0.0
    depth_acc = {e: (by_depth[e][0] / by_depth[e][1]) if by_depth[e][1] > 0 else 0.0 for e in by_depth}
    return total_acc, depth_acc

def evaluate_baseline(p):
    # Classical baseline: 64 * parity - weighted_mob
    f = p["features"]
    return 64.0 * f[:, 4] - f[:, 7]

def evaluate_ordr002(p):
    # Hand-crafted ORDR-002: 64 * parity - 128 * is_open_x - weighted_mob
    f = p["features"]
    return 64.0 * f[:, 4] - 128.0 * f[:, 1] - f[:, 7]

def evaluate_distilled_rules(p):
    """
    Distilled Lexicographical Priority Rule Set:
    Learned from empirical decision tree boundaries:
    1. If opponent has 0 mobility (wipeout/pass), unconditionally highest priority.
    2. Severe penalty (-256) if opponent gains immediate corner.
    3. Open X-square penalty (-128) only if opponent does not pass.
    4. Mobility minimization + Parity bonus (+64).
    """
    f = p["features"]
    score = np.zeros(len(f), dtype=np.float32)
    # Feature indices:
    # 0: is_corner, 1: is_open_x, 4: parity, 5: raw_opp_mob, 6: opp_corner_moves, 7: weighted_mob
    for i in range(len(f)):
        raw_mob = f[i, 5]
        opp_corner = f[i, 6]
        parity = f[i, 4]
        is_open_x = f[i, 1]
        
        # Priority tiers
        if raw_mob == 0:
            # Wipeout / opponent pass move: unconditional top static priority
            s = 10000.0
        else:
            s = 0.0
            # Opponent corner access is catastrophic
            if opp_corner > 0:
                s -= 256.0 * opp_corner
            # Open X-square penalty
            if is_open_x > 0:
                s -= 128.0
            # Opponent mobility minimization
            s -= 128.0 * raw_mob
            # Parity bonus
            if parity > 0:
                s += 64.0
        score[i] = s
    return score

def main():
    parser = argparse.ArgumentParser(description="Distilled Move Ordering vs Classical Hand-Crafted Rules")
    parser.add_argument("--wthor-dir", default="data/wthor", help="Directory with WTHOR .wtb files")
    parser.add_argument("--save-data", default=None, help="Cache extracted dataset to .npz")
    parser.add_argument("--load-data", default=None, help="Load extracted dataset from .npz")
    args = parser.parse_args()

    # 1. Load or extract positions
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading cached dataset from {args.load_data}...")
        data = np.load(args.load_data, allow_pickle=True)
        positions = list(data["positions"])
    else:
        positions = extract_dataset(args.wthor_dir)
        if args.save_data:
            print(f"Saving dataset to {args.save_data}...")
            np.savez_compressed(args.save_data, positions=positions)

    # 2. Train / Validation Split (80 / 20)
    np.random.seed(42)
    perm = np.random.permutation(len(positions))
    split = int(0.8 * len(positions))
    train_pos = [positions[i] for i in perm[:split]]
    val_pos = [positions[i] for i in perm[split:]]

    print(f"\nDataset: {len(positions)} positions | Train: {len(train_pos)} | Val: {len(val_pos)}")

    # 3. Evaluate Baselines
    train_base, _ = evaluate_top1(train_pos, evaluate_baseline)
    val_base, val_base_depth = evaluate_top1(val_pos, evaluate_baseline)
    print(f"\n[Baseline Engine]:")
    print(f"  Train Top-1: {train_base * 100:.2f}% | Val Top-1: {val_base * 100:.2f}%")

    train_002, _ = evaluate_top1(train_pos, evaluate_ordr002)
    val_002, val_002_depth = evaluate_top1(val_pos, evaluate_ordr002)
    print(f"\n[ORDR-002 Hand-Crafted]:")
    print(f"  Train Top-1: {train_002 * 100:.2f}% | Val Top-1: {val_002 * 100:.2f}% (Δ: {(val_002 - val_base)*100:+.2f}%)")

    # 4. Train Distilled Decision Trees via scikit-learn
    try:
        from sklearn.tree import DecisionTreeRegressor, export_text
    except ImportError:
        print("Error: scikit-learn is required. Run with `uv run --with scikit-learn`.")
        sys.exit(1)

    # Formulate Training Data for CART:
    # Pairwise margin / ListNet ranking surrogate:
    # For each candidate move, target = +1.0 if best_idx else -1.0 / (K - 1)
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
    print(f"\nTraining set matrix: {X_train.shape[0]} candidate moves, {X_train.shape[1]} features.")

    best_tree = None
    best_tree_acc = 0.0
    best_depth = 0

    print("\n--- Shallow Decision Tree (CART) Sweep ---")
    for depth in [2, 3, 4, 5]:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=42)
        tree.fit(X_train, y_train)

        def tree_score_fn(p, model=tree):
            return model.predict(p["features"])

        t_acc, _ = evaluate_top1(train_pos, tree_score_fn)
        v_acc, v_depth = evaluate_top1(val_pos, tree_score_fn)
        print(f"  Tree Depth {depth}: Train Top-1 = {t_acc*100:.2f}%, Val Top-1 = {v_acc*100:.2f}% (Δ vs ORDR-002: {(v_acc - val_002)*100:+.2f}%)")
        if v_acc > best_tree_acc:
            best_tree_acc = v_acc
            best_tree = tree
            best_depth = depth

    print(f"\nOptimal Distilled Decision Tree (Depth {best_depth}): Val Top-1 = {best_tree_acc*100:.2f}%")
    print("\n=== Distilled Tree Structure ===")
    print(export_text(best_tree, feature_names=FEATURE_NAMES))

    # 5. Evaluate Distilled Lexicographical Rule List
    train_rules, _ = evaluate_top1(train_pos, evaluate_distilled_rules)
    val_rules, val_rules_depth = evaluate_top1(val_pos, evaluate_distilled_rules)
    print(f"\n[Distilled Priority Rules]:")
    print(f"  Train Top-1: {train_rules * 100:.2f}% | Val Top-1: {val_rules * 100:.2f}% (Δ vs ORDR-002: {(val_rules - val_002)*100:+.2f}%)")

    # 6. Depth Breakdown Table
    print("\n=== Depth Breakdown (Top-1 Accuracy by Remaining Empties E) ===")
    print(f"{'Depth (E)':<10} {'Positions':<10} {'Baseline':<12} {'ORDR-002':<12} {'Distilled Tree':<16} {'Distilled Rules':<16}")
    print("-" * 76)
    def best_tree_fn(p):
        return best_tree.predict(p["features"])
    _, tree_val_depth = evaluate_top1(val_pos, best_tree_fn)

    val_by_e = {e: sum(1 for p in val_pos if p["empties"] == e) for e in range(8, 15)}
    for e in range(8, 15):
        print(f"E = {e:<6d} {val_by_e[e]:<10d} {val_base_depth[e]*100:6.2f}%      {val_002_depth[e]*100:6.2f}%      {tree_val_depth[e]*100:6.2f}%           {val_rules_depth[e]*100:6.2f}%")

    print(f"{'OVERALL':<10} {len(val_pos):<10d} {val_base*100:6.2f}%      {val_002*100:6.2f}%      {best_tree_acc*100:6.2f}%           {val_rules*100:6.2f}%")

    # 7. Generate C Code for Distilled Decision Tree & Rules
    print("\n=== Generated C Code for Distilled Move Ordering ===")
    c_code = generate_c_decision_tree(best_tree, FEATURE_NAMES)
    print(c_code)

def generate_c_decision_tree(tree, feature_names):
    from sklearn.tree import _tree
    t = tree.tree_
    lines = []
    lines.append("/* Distilled Move Ordering Decision Tree (Depth %d) */" % tree.max_depth)
    lines.append("static INLINE int")
    lines.append("distilled_tree_score( int is_corner, int is_open_x, int is_open_c,")
    lines.append("                     int parity, int raw_opp_mob, int opp_corner_moves,")
    lines.append("                     int weighted_mob, int flips, int empties ) {")

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

if __name__ == "__main__":
    main()
