#!/usr/bin/env python3
"""analyze_aspiration_error.py - Empirical error profiling for endgame aspiration windows.

Analyzes the error distribution between the 4-pattern static evaluation (and aspiration center)
and ground-truth exact solve scores across empty square counts E in [14, 32].
"""

import argparse
import ctypes
import glob
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.parse_wthor import parse_wthor_file
from scripts.generate_eval_data import OthelloBoard, BLACK, WHITE, EMPTY
from scripts.eval_candidate import FFO_POSITIONS


def load_patterns_lib():
    lib_path = REPO_ROOT / "build" / "libend_patterns.dylib"
    if not lib_path.exists():
        os.system(f"clang -shared -o {lib_path} {REPO_ROOT}/build/obj/end_patterns.o {REPO_ROOT}/build/obj/end_aspiration.o")
    lib = ctypes.CDLL(str(lib_path))
    lib.end_pattern_evaluate.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lib.end_pattern_evaluate.restype = ctypes.c_int
    lib.end_aspiration_center.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.end_aspiration_center.restype = ctypes.c_int
    return lib


def get_bitboards(board):
    black_bits = 0
    white_bits = 0
    for r in range(8):
        for c in range(8):
            sq = r * 8 + c
            val = board[r][c]
            if val == BLACK:
                black_bits |= (1 << sq)
            elif val == WHITE:
                white_bits |= (1 << sq)
    return black_bits, white_bits


def parse_board_str_to_bb(board_str):
    board_part, side = board_str.split()
    bb_x = 0
    bb_o = 0
    for i, ch in enumerate(board_part):
        if ch == "X":
            bb_x |= (1 << i)
        elif ch == "O":
            bb_o |= (1 << i)
    if side == "X":
        return bb_x, bb_o, side
    else:
        return bb_o, bb_x, side


def main():
    parser = argparse.ArgumentParser(description="Profile aspiration window center errors.")
    parser.add_argument("--wthor-dir", type=Path, default=REPO_ROOT / "data" / "wthor")
    parser.add_argument("--max-games", type=int, default=5000, help="Max exact WTHOR games to analyze")
    parser.add_argument("--max-score-diff", type=int, default=40, help="Filter out blowout games (|score| > diff)")
    args = parser.parse_args()

    lib = load_patterns_lib()

    # Store errors per empty square count
    # empties -> list of (true_score - center)
    errors = defaultdict(list)
    raw_errors = defaultdict(list)
    true_scores = defaultdict(list)
    centers = defaultdict(list)

    target_empties = list(range(14, 25))  # 14 to 24 empties from WTHOR
    wtb_files = sorted(args.wthor_dir.glob("WTH_*.wtb"), reverse=True)

    print(f"Scanning WTHOR games (max {args.max_games} exact games)...")
    games_parsed = 0
    for wtb_path in wtb_files:
        if games_parsed >= args.max_games:
            break
        for game in parse_wthor_file(wtb_path):
            if game.real_score != game.theo_score:
                continue
            if abs(game.theo_score - 32) > (args.max_score_diff // 2):
                continue

            b = OthelloBoard()
            theo_diff = 2 * game.theo_score - 64

            for move_idx, (r, c) in enumerate(game.moves):
                if b.side_to_move == EMPTY:
                    break
                empties = 64 - b.disc_count
                if empties in target_empties:
                    true_score = theo_diff if b.side_to_move == BLACK else -theo_diff
                    bb_black, bb_white = get_bitboards(b.board)
                    my_bits, opp_bits = (bb_black, bb_white) if b.side_to_move == BLACK else (bb_white, bb_black)

                    est = lib.end_pattern_evaluate(my_bits, opp_bits)
                    center = lib.end_aspiration_center(est, 60)

                    err = true_score - center
                    raw_err = true_score - (est / 128.0)

                    errors[empties].append(err)
                    raw_errors[empties].append(raw_err)
                    true_scores[empties].append(true_score)
                    centers[empties].append(center)

                b.make_move(r, c)

            games_parsed += 1
            if games_parsed >= args.max_games:
                break

    print(f"Analyzed {games_parsed} exact WTHOR games across {len(target_empties)} empty square counts.")

    # Also analyze FFO suite
    print("Analyzing FFO benchmark suite (20 positions, 20-34 empties)...")
    ffo_errors = defaultdict(list)
    for name, board_str, b_sc, w_sc, moves in FFO_POSITIONS:
        my_bits, opp_bits, side = parse_board_str_to_bb(board_str)
        board_part, _ = board_str.split()
        empties = board_part.count("-")
        true_diff = (b_sc - w_sc) if side == "X" else (w_sc - b_sc)
        est = lib.end_pattern_evaluate(my_bits, opp_bits)
        center = lib.end_aspiration_center(est, 60)
        ffo_errors[empties].append(true_diff - center)

    print("\n" + "=" * 105)
    print(f"{'Empties (E)':>12} | {'Samples':>7} | {'Mean (mu)':>10} | {'Std (sigma)':>11} | {'MAE':>7} | {'80% Tile':>9} | {'95% Tile':>9} | {'Hit [-1,1]':>10} | {'Hit [-3,3]':>10}")
    print("=" * 105)

    all_empties = sorted(set(list(errors.keys()) + list(ffo_errors.keys())))
    sigma_by_e = {}
    q80_by_e = {}

    for emp in all_empties:
        combined = errors[emp] if emp in errors else ffo_errors[emp]
        arr = np.array(combined)
        abs_arr = np.abs(arr)
        n = len(arr)
        mu = np.mean(arr)
        sigma = np.std(arr)
        mae = np.mean(abs_arr)
        q80 = np.percentile(abs_arr, 80)
        q95 = np.percentile(abs_arr, 95)
        h1 = np.mean(abs_arr <= 1) * 100
        h3 = np.mean(abs_arr <= 3) * 100

        sigma_by_e[emp] = sigma
        q80_by_e[emp] = q80

        tag = " (FFO)" if emp not in errors else ""
        print(f"{emp:>10} em{tag} | {n:>7} | {mu:>+10.2f} | {sigma:>11.2f} | {mae:>7.2f} | {q80:>9.1f} | {q95:>9.1f} | {h1:>9.1f}% | {h3:>9.1f}%")

    print("-" * 105)

    # Generate Proposed Adaptive Step Table
    print("\n=== Proposed C Adaptive Step Table for src/end.c ===")
    print("/* Adaptive aspiration jump step size per empty square count: step = adaptive_step_table[empties] */")
    print("static const int adaptive_aspiration_step[33] = {")
    step_vals = []
    for e in range(33):
        if e < 16:
            step = 2
        elif e in q80_by_e:
            # Step size set to roughly half of 80th percentile error, even integer >= 2
            val = int(math.ceil(q80_by_e[e] / 4.0) * 2)
            step = max(2, min(val, 12))
        else:
            # Interpolate or extrapolate
            step = max(2, min(2 + (e - 16) // 2 * 2, 12))
        step_vals.append(str(step))
    for i in range(0, 33, 8):
        chunk = step_vals[i:i+8]
        print(f"  {', '.join(chunk)}, /* empties {i}..{min(i+7, 32)} */")
    print("};")


if __name__ == "__main__":
    main()
