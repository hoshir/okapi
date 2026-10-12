import numpy as np
import ctypes
import sys
from pathlib import Path

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_aspiration_error import parse_board_str_to_bb
from scripts.eval_candidate import FFO_POSITIONS
from scripts.parse_wthor import parse_wthor_file
from scripts.generate_eval_data import OthelloBoard, BLACK, WHITE, EMPTY

lib = ctypes.CDLL(str(REPO_ROOT / "build" / "libhelper_eval.dylib"))
lib.get_features.argtypes = [
    ctypes.c_uint64, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
]

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

print("Loading 4,000 exact WTHOR games (2018-2025)...")
wtb_files = sorted((REPO_ROOT / "data" / "wthor").glob("WTH_*.wtb"), reverse=True)
data = []
games = 0
for wtb_path in wtb_files:
    if games >= 4000:
        break
    for game in parse_wthor_file(wtb_path):
        if game.real_score != game.theo_score:
            continue
        b = OthelloBoard()
        theo_diff = 2 * game.theo_score - 64
        for move_idx, (r, c) in enumerate(game.moves):
            if b.side_to_move == EMPTY:
                break
            empties = 64 - b.disc_count
            if 16 <= empties <= 28:
                true_score = theo_diff if b.side_to_move == BLACK else -theo_diff
                bb_b, bb_w = get_bitboards(b.board)
                my_bits, opp_bits = (bb_b, bb_w) if b.side_to_move == BLACK else (bb_w, bb_b)
                est = ctypes.c_int()
                mob = ctypes.c_int()
                disc = ctypes.c_int()
                lib.get_features(my_bits, opp_bits, ctypes.byref(est), ctypes.byref(mob), ctypes.byref(disc))
                data.append((est.value / 128.0, empties, mob.value, disc.value, true_score))
            b.make_move(r, c)
        games += 1
        if games >= 4000:
            break

print(f"Loaded {len(data)} exact-solved positions across empties [16..28].")

ffo_data = []
for name, board_str, b_sc, w_sc, moves in FFO_POSITIONS:
    my_bits, opp_bits, side = parse_board_str_to_bb(board_str)
    board_part, _ = board_str.split()
    empties = board_part.count("-")
    true_diff = (b_sc - w_sc) if side == "X" else (w_sc - b_sc)
    est = ctypes.c_int()
    mob = ctypes.c_int()
    disc = ctypes.c_int()
    lib.get_features(my_bits, opp_bits, ctypes.byref(est), ctypes.byref(mob), ctypes.byref(disc))
    ffo_data.append({
        "name": name,
        "emp": empties,
        "true": true_diff,
        "est_d": est.value / 128.0,
        "mob": mob.value,
        "disc": disc.value,
    })

def to_even(x):
    return int(np.round(x / 2.0) * 2.0)

# Evaluate Baseline
base_wthor_errs = [abs(to_even(x) - t) for x, e, m, d, t in data]
base_ffo_errs = [abs(to_even(d["est_d"]) - d["true"]) for d in ffo_data]
print(f"Baseline MAE: WTHOR = {np.mean(base_wthor_errs):.2f}, FFO = {np.mean(base_ffo_errs):.2f}")
print(f"Baseline Hit [-2, 2]: WTHOR = {np.mean([e <= 2 for e in base_wthor_errs])*100:.1f}%, FFO = {np.mean([e <= 2 for e in base_ffo_errs])*100:.1f}%")

# Grid search for the optimal e-Sigmoid parameters
print("\nOptimizing e-Sigmoid parameters on WTHOR...")
best_score = 999.0
best_cfg = None

for x0 in [3.2, 3.5, 3.8, 4.0]:
    for k in [0.8, 1.0, 1.2]:
        for s_max in [3.0, 3.5, 4.0, 4.5]:
            for mob_coef in [0.0, 1.0, 1.5]:
                errs = []
                for x, e, m, disc, t in data:
                    ax = abs(x)
                    w = np.clip((28.0 - e) / 12.0, 0.0, 1.0)
                    sigma = 1.0 / (1.0 + np.exp(-k * (ax - x0)))
                    scale = 1.0 + s_max * w * sigma
                    base = x * scale
                    if mob_coef > 0:
                        if x > 1.0 and m >= 6:
                            base += (m - 5) * mob_coef * w
                        elif x < -1.0 and m <= -6:
                            base += (m + 5) * mob_coef * w
                    errs.append(abs(to_even(base) - t))
                mae = np.mean(errs)
                if mae < best_score:
                    best_score = mae
                    best_cfg = (x0, k, s_max, mob_coef)

x0, k, s_max, mob_coef = best_cfg
print(f"Best Config: x0={x0:.1f}, k={k:.1f}, s_max={s_max:.1f}, mob_coef={mob_coef:.1f} | WTHOR MAE = {best_score:.2f}")

# Evaluate Final Proposal on WTHOR and FFO
def final_predict(x, e, m, d):
    ax = abs(x)
    w = np.clip((28.0 - e) / 12.0, 0.0, 1.0)
    sigma = 1.0 / (1.0 + np.exp(-k * (ax - x0)))
    scale = 1.0 + s_max * w * sigma
    base = x * scale
    if mob_coef > 0:
        if x > 1.0 and m >= 6:
            base += (m - 5) * mob_coef * w
        elif x < -1.0 and m <= -6:
            base += (m + 5) * mob_coef * w
    return to_even(base)

final_wthor_errs = [abs(final_predict(x, e, m, d) - t) for x, e, m, d, t in data]
final_ffo_errs = [abs(final_predict(d["est_d"], d["emp"], d["mob"], d["disc"]) - d["true"]) for d in ffo_data]

print(f"\nFinal Proposal MAE: WTHOR = {np.mean(final_wthor_errs):.2f} (-{np.mean(base_wthor_errs)-np.mean(final_wthor_errs):.2f}), FFO = {np.mean(final_ffo_errs):.2f} (-{np.mean(base_ffo_errs)-np.mean(final_ffo_errs):.2f})")
print(f"Final Proposal Hit [-2, 2]: WTHOR = {np.mean([e <= 2 for e in final_wthor_errs])*100:.1f}%, FFO = {np.mean([e <= 2 for e in final_ffo_errs])*100:.1f}%")

print("\n=== Final Proposal Predictions on Complete FFO Suite ===")
print(f"{'Position':<8} | {'Emp':>3} | {'True':>5} | {'Base Center':>11} | {'Final Center':>12} | {'Base Err':>8} | {'Final Err':>9}")
print("-" * 75)
for d in ffo_data:
    b = to_even(d["est_d"])
    f = final_predict(d["est_d"], d["emp"], d["mob"], d["disc"])
    print(f"{d['name']:<8} | {d['emp']:3d} | {d['true']:>+5d} | {b:>+11d} | {f:>+12d} | {b - d['true']:>+8d} | {f - d['true']:>+9d}")
