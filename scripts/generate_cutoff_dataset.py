#!/usr/bin/env python3
"""generate_cutoff_dataset.py - Search-Aware Fail-High Cutoff Dataset Generator (EVAL-013).

Collects and parses raw binary search cutoff logs produced by Zebra's endgame solver
(when ZEBRA_CUTOFF_LOG is enabled), extracts 18-pattern feature tensors for all candidate
moves, and compiles them into a PyTorch dataset for Learning-to-Rank distillation.
"""

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Bitboard helpers & directions
INNER_MASK = 0x7E7E7E7E7E7E7E7E
MASK_64 = 0xFFFFFFFFFFFFFFFF

# Pattern geometry from src/end_patterns.c
CORNER33_SQS = [
    [18, 17, 16, 10, 9, 8, 2, 1, 0],
    [42, 41, 40, 50, 49, 48, 58, 57, 56],
    [21, 22, 23, 13, 14, 15, 5, 6, 7],
    [45, 46, 47, 53, 54, 55, 61, 62, 63]
]

EDGE2X_SQS = [
    [49, 9, 56, 48, 40, 32, 24, 16, 8, 0],
    [54, 14, 63, 55, 47, 39, 31, 23, 15, 7],
    [14, 9, 7, 6, 5, 4, 3, 2, 1, 0],
    [54, 49, 63, 62, 61, 60, 59, 58, 57, 56]
]

RECT24_SQS = [
    [11, 10, 9, 8, 3, 2, 1, 0],
    [51, 50, 49, 48, 59, 58, 57, 56],
    [12, 13, 14, 15, 4, 5, 6, 7],
    [52, 53, 54, 55, 60, 61, 62, 63],
    [25, 17, 9, 1, 24, 16, 8, 0],
    [30, 22, 14, 6, 31, 23, 15, 7],
    [33, 41, 49, 57, 32, 40, 48, 56],
    [38, 46, 54, 62, 39, 47, 55, 63]
]

DIAG8_SQS = [
    [63, 54, 45, 36, 27, 18, 9, 0],
    [56, 49, 42, 35, 28, 21, 14, 7]
]

SHIFTS = [
    (1, INNER_MASK, lambda x: (x << 1) & MASK_64, lambda x: x >> 1),
    (-1, INNER_MASK, lambda x: x >> 1, lambda x: (x << 1) & MASK_64),
    (8, MASK_64, lambda x: (x << 8) & MASK_64, lambda x: x >> 8),
    (-8, MASK_64, lambda x: x >> 8, lambda x: (x << 8) & MASK_64),
    (9, INNER_MASK, lambda x: (x << 9) & MASK_64, lambda x: x >> 9),
    (7, INNER_MASK, lambda x: (x << 7) & MASK_64, lambda x: x >> 7),
    (-7, INNER_MASK, lambda x: x >> 7, lambda x: (x << 7) & MASK_64),
    (-9, INNER_MASK, lambda x: x >> 9, lambda x: (x << 9) & MASK_64),
]


def generate_all_moves(my_bits: int, opp_bits: int) -> List[int]:
    moves_bb = 0
    empty = ~(my_bits | opp_bits) & MASK_64
    for _, mask, step, _ in SHIFTS:
        eff_mask = mask if mask != MASK_64 else opp_bits
        flip = step(my_bits) & eff_mask
        flip |= step(flip) & eff_mask
        flip |= step(flip) & eff_mask
        flip |= step(flip) & eff_mask
        flip |= step(flip) & eff_mask
        flip |= step(flip) & eff_mask
        moves_bb |= step(flip) & empty

    legal_moves = []
    for sq in range(64):
        if (moves_bb >> sq) & 1:
            legal_moves.append(sq)
    return legal_moves


def make_move_bitboard(my_bits: int, opp_bits: int, sq: int) -> Tuple[int, int]:
    sq_bit = 1 << sq
    flips = 0
    for _, mask, step, _ in SHIFTS:
        eff_mask = mask if mask != MASK_64 else opp_bits
        curr = step(sq_bit) & eff_mask
        ray = 0
        while curr != 0:
            ray |= curr
            curr = step(curr) & eff_mask
        if ray != 0:
            anchor = step(ray) & my_bits
            if anchor != 0:
                flips |= ray

    new_my = my_bits | flips | sq_bit
    new_opp = opp_bits & ~flips
    return new_my, new_opp


def extract_18_indices(my_bits: int, opp_bits: int) -> List[int]:
    """Extracts 18 pattern indices from bitboards matching src/end_patterns.c."""
    indices = []
    # 4 Corner 3x3
    for sqs in CORNER33_SQS:
        idx = 0
        for s in sqs:
            mb = (my_bits >> s) & 1
            ob = (opp_bits >> s) & 1
            idx = idx * 3 + (1 - ob + mb)
        indices.append(idx)

    # 4 Edge + 2X
    for sqs in EDGE2X_SQS:
        idx = 0
        for s in sqs:
            mb = (my_bits >> s) & 1
            ob = (opp_bits >> s) & 1
            idx = idx * 3 + (1 - ob + mb)
        indices.append(idx)

    # 8 2x4 Rectangles
    for sqs in RECT24_SQS:
        idx = 0
        for s in sqs:
            mb = (my_bits >> s) & 1
            ob = (opp_bits >> s) & 1
            idx = idx * 3 + (1 - ob + mb)
        indices.append(idx)

    # 2 Diagonals
    for sqs in DIAG8_SQS:
        idx = 0
        for s in sqs:
            mb = (my_bits >> s) & 1
            ob = (opp_bits >> s) & 1
            idx = idx * 3 + (1 - ob + mb)
        indices.append(idx)

    return indices


def run_cutoff_collection(repo_root: Path, log_path: Path, positions_file: Optional[Path] = None, max_games: int = 50):
    """Executes scrzebra on endgame positions with ZEBRA_CUTOFF_LOG active."""
    bin_dir = repo_root / "build" / "bin"
    scrzebra = bin_dir / "scrzebra"

    if not scrzebra.exists():
        subprocess.run(["make", "-s", "scrzebra"], cwd=repo_root, check=True)

    if log_path.exists():
        log_path.unlink()

    env = os.environ.copy()
    env["ZEBRA_CUTOFF_LOG"] = str(log_path.resolve())

    # Build position list: start with FFO suite
    positions = []
    if positions_file and positions_file.exists():
        with open(positions_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    positions.append(line)
    else:
        # 1. 20 FFO positions from tests/ffotest.scr
        ffo_scr = repo_root / "tests" / "ffotest.scr"
        if ffo_scr.exists():
            with open(ffo_scr, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("%") and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2 and len(parts[0]) == 64:
                            positions.append(f"{parts[0]} {parts[1]}")

        # 2. Extract diverse positions from WTHOR databases across stages 7, 8, 9, 10 (empties 12 to 20)
        wthor_dir = repo_root / "data" / "wthor"
        wthor_files = sorted(wthor_dir.glob("WTH_202*.wtb"), reverse=True)
        if wthor_files:
            try:
                sys.path.insert(0, str(repo_root / "scripts"))
                from parse_wthor import parse_wthor_file
                from generate_eval_data import OthelloBoard, BLACK, WHITE, EMPTY

                for wfile in wthor_files[:2]:
                    for game in parse_wthor_file(wfile):
                        b = OthelloBoard()
                        for r, c in game.moves:
                            if b.side_to_move == EMPTY:
                                break
                            emp = 64 - b.disc_count
                            if emp in (14, 16, 18, 20):
                                chars = []
                                for row in range(8):
                                    for col in range(8):
                                        sq = b.board[row][col]
                                        chars.append('X' if sq == BLACK else ('O' if sq == WHITE else '-'))
                                side = 'X' if b.side_to_move == BLACK else 'O'
                                pos_str = f"{''.join(chars)} {side}"
                                if pos_str not in positions:
                                    positions.append(pos_str)
                            if not b.make_move(r, c):
                                break
                        if len(positions) >= max_games:
                            break
                    if len(positions) >= max_games:
                        break
            except Exception as e:
                print(f"Note: WTHOR extraction skipped ({e})")

    print(f"Collecting fail-high cutoffs from {len(positions)} positions...")
    t0 = time.time()
    for i, pos in enumerate(positions[:max_games]):
        cmd = [
            str(scrzebra),
            "-n", "1",
            "-h", "20",
            "-e", "0",
            "-line", "1",
            "-script", "/dev/stdin",
            "/dev/null"
        ]
        try:
            subprocess.run(
                cmd,
                cwd=bin_dir,
                input=pos + "\n",
                text=True,
                capture_output=True,
                env=env,
                timeout=30.0
            )
        except subprocess.TimeoutExpired:
            pass
        sz = log_path.stat().st_size if log_path.exists() else 0
        print(f"  [{i+1}/{min(len(positions), max_games)}] Processed: {pos[:30]}... | Log size: {sz / 1024:.1f} KB ({sz // 20} cutoffs)")

    print(f"Collection complete in {time.time() - t0:.1f}s. Total logged cutoffs: {log_path.stat().st_size // 20}")


def parse_and_convert_log(log_path: Path, out_path: Path, max_samples: Optional[int] = None) -> int:
    """Parses binary cutoffs log and compiles PyTorch dataset."""
    if not log_path.exists():
        raise FileNotFoundError(f"Cutoffs log {log_path} not found.")

    file_size = log_path.stat().st_size
    num_records = file_size // 20
    print(f"Parsing {num_records} binary cutoff records from {log_path} ({file_size / 1024 / 1024:.2f} MB)...")

    # Cache per unique position: (my_bits, opp_bits) -> (legals, cand_indices_18)
    pos_cache: Dict[Tuple[int, int], Tuple[List[int], List[List[int]]]] = {}
    samples = []

    t0 = time.time()
    with open(log_path, "rb") as f:
        count = 0
        while True:
            chunk = f.read(20)
            if len(chunk) < 20:
                break
            my_bits, opp_bits, cutoff_move, empties, move_idx, total_moves = struct.unpack("<QQBBBB", chunk)

            board_key = (my_bits, opp_bits)
            if board_key in pos_cache:
                legals, cand_indices = pos_cache[board_key]
            else:
                legals = generate_all_moves(my_bits, opp_bits)
                if len(legals) < 2:
                    pos_cache[board_key] = ([], [])
                    continue
                cand_indices = []
                for m in legals:
                    child_my, child_opp = make_move_bitboard(my_bits, opp_bits, m)
                    indices = extract_18_indices(child_my, child_opp)
                    cand_indices.append(indices)
                pos_cache[board_key] = (legals, cand_indices)

            if cutoff_move not in legals:
                continue

            best_idx = legals.index(cutoff_move)
            # Map empties to stage (Stage 7: 19-24, Stage 8: 15-18, Stage 9: 11-14, Stage 10: 8-10)
            if empties >= 19:
                stage = 7
            elif empties >= 15:
                stage = 8
            elif empties >= 11:
                stage = 9
            else:
                stage = 10

            samples.append({
                "stage": stage,
                "empties": empties,
                "best_idx": best_idx,
                "child_features_18": cand_indices,
                "move_index": move_idx,
                "total_moves": len(legals),
            })

            count += 1
            if max_samples and count >= max_samples:
                break

    print(f"Processed {count} valid cutoffs into {len(samples)} training samples (unique positions: {len(pos_cache)}) in {time.time() - t0:.2f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out_path)
    print(f"Saved PyTorch cutoff dataset to {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.2f} MB)")
    return len(samples)


def main():
    parser = argparse.ArgumentParser(description="Generate and convert search-aware fail-high cutoff logs (EVAL-013).")
    parser.add_argument("--log", type=Path, default=Path("data/cutoffs.bin"), help="Path to raw binary cutoffs file")
    parser.add_argument("--out", type=Path, default=Path("data/ltr_cutoff_dataset.pt"), help="Output PyTorch dataset path")
    parser.add_argument("--collect", action="store_true", help="Run scrzebra to collect cutoffs")
    parser.add_argument("--positions", type=Path, default=None, help="Custom positions file to solve")
    parser.add_argument("--max-games", type=int, default=50, help="Max positions to solve during collection")
    parser.add_argument("--max-samples", type=int, default=None, help="Max cutoff samples to parse")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.collect or not args.log.exists():
        run_cutoff_collection(repo_root, args.log, positions_file=args.positions, max_games=args.max_games)

    parse_and_convert_log(args.log, args.out, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
