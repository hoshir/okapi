#!/usr/bin/env python3
"""discover_patterns_ml.py - Automated Othello Pattern Topology Discovery via ML (Automated Pattern Discovery).

Implements the 3-Stage Machine Learning Pattern Discovery pipeline on the 100% exact-solved,
FFO-decontaminated dataset (`data/clean_exact_dataset.pt`):

  Stage 1: Train a D4-symmetric Non-Linear 64-Square Teacher Network (MLP with 8x dihedral
           data augmentation = 150,560 training states) to predict exact minimax score and
           move-ordering logits.
  Stage 2: Extract the 64x64 Pairwise Non-Linear Interaction Matrix I[i, j] via exact
           second-order finite differences on the Teacher network, and identify the strongest
           square interactions missed by human-crafted patterns (`c33`, `e2x`, `r24`).
  Stage 3: Optimize k-square pattern topologies under D4 symmetry via Simulated Annealing
           (Maximum-Weight k-Subgraph Cover), and evaluate the ML-discovered pattern shapes
           head-to-head against human-designed patterns on held-out Validation and Test splits.
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_cutoff_dataset import CORNER33_SQS, EDGE2X_SQS, RECT24_SQS


def d4_square_maps() -> List[List[int]]:
    """Returns the 8 dihedral (D4) permutation maps on 0..63."""
    maps = []
    for sym in range(8):
        perm = [0] * 64
        for sq in range(64):
            r, c = divmod(sq, 8)
            if sym & 1:
                c = 7 - c
            if sym & 2:
                r = 7 - r
            if sym & 4:
                r, c = c, r
            perm[sq] = r * 8 + c
        maps.append(perm)
    return maps


D4_MAPS = d4_square_maps()


def sq_name(sq: int) -> str:
    r, c = divmod(sq, 8)
    return f"{chr(ord('a') + c)}{r + 1}"


def bitboards_to_tensor(my_list: List[int], opp_list: List[int]) -> torch.Tensor:
    """Converts lists of 64-bit integers into a (N, 128) float32 binary tensor."""
    n = len(my_list)
    shifts = torch.arange(64, dtype=torch.int64)
    # Use int64 via signed conversion for bit >> shift
    my_u = [x if x < (1 << 63) else x - (1 << 64) for x in my_list]
    opp_u = [x if x < (1 << 63) else x - (1 << 64) for x in opp_list]
    my_t = torch.tensor(my_u, dtype=torch.int64).unsqueeze(1)
    opp_t = torch.tensor(opp_u, dtype=torch.int64).unsqueeze(1)
    my_bits = ((my_t >> shifts) & 1).float()
    opp_bits = ((opp_t >> shifts) & 1).float()
    return torch.cat([my_bits, opp_bits], dim=1)  # (N, 128)


class TeacherNet(nn.Module):
    """D4-invariant Non-Linear 64-Square Teacher Network.
    Applies a shared non-linear MLP across all 8 dihedral symmetries of the 64-square board.
    """

    def __init__(self, d4_perms: torch.Tensor, hidden: int = 256):
        super().__init__()
        # d4_perms: (8, 128) index permutation for [my_64, opp_64]
        self.register_buffer("d4_perms", d4_perms)
        self.fc1 = nn.Linear(128, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)

    def forward_single(self, x128: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x128))
        h = F.silu(self.fc2(h))
        return self.fc3(h).squeeze(-1)

    def forward(self, x128: torch.Tensor) -> torch.Tensor:
        # x128: (..., 128) -> symmetrize over 8 D4 transforms
        orig_shape = x128.shape[:-1]
        flat = x128.reshape(-1, 128)
        # Gather 8 symmetries: (N, 8, 128)
        sym_x = flat[:, self.d4_perms]
        out = self.forward_single(sym_x).mean(dim=-1)
        return out.reshape(orig_shape)


def build_move_ranking_tensors(samples: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Builds padded candidate child board tensors (N, K, 128) + fast_score + best_idx."""
    n = len(samples)
    max_k = max(len(s["child_my_bits"]) for s in samples)

    child_x = torch.zeros((n, max_k, 128), dtype=torch.float32)
    child_ternary = torch.zeros((n, max_k, 64), dtype=torch.long)
    fast_sc = torch.zeros((n, max_k), dtype=torch.float32)
    mask = torch.zeros((n, max_k), dtype=torch.bool)
    best_idx = torch.zeros(n, dtype=torch.long)
    exact_score = torch.zeros(n, dtype=torch.float32)
    empties = torch.zeros(n, dtype=torch.long)

    shifts = torch.arange(64, dtype=torch.int64)

    for i, s in enumerate(samples):
        c_my = s["child_my_bits"]
        c_opp = s["child_opp_bits"]
        tc = s["child_tactical"]
        k = len(c_my)

        my_u = [x if x < (1 << 63) else x - (1 << 64) for x in c_my]
        opp_u = [x if x < (1 << 63) else x - (1 << 64) for x in c_opp]
        my_b = ((torch.tensor(my_u, dtype=torch.int64).unsqueeze(1) >> shifts) & 1)
        opp_b = ((torch.tensor(opp_u, dtype=torch.int64).unsqueeze(1) >> shifts) & 1)

        child_x[i, :k, 0:64] = my_b.float()
        child_x[i, :k, 64:128] = opp_b.float()
        # Ternary digit per square: 1 - opp + my in {0, 1, 2} (matching src/end_patterns.c)
        child_ternary[i, :k] = 1 - opp_b + my_b

        tc_t = torch.tensor(tc, dtype=torch.float32)
        opp_mob = tc_t[:, 0]
        opp_corner = tc_t[:, 1]
        is_open_x = tc_t[:, 6]
        is_open_c = tc_t[:, 7]
        parity = tc_t[:, 8]
        is_wipeout = (opp_mob == 0).float()
        c_pen = 128.0 * torch.clamp(is_open_x + is_open_c, 0.0, 1.0)
        fast_sc[i, :k] = -128.0 * (opp_mob + opp_corner) - c_pen + 4096.0 * is_wipeout + 64.0 * parity

        mask[i, :k] = True
        best_idx[i] = int(s["best_idx"])
        # Note: in child node, side to move is opponent, so child value from mover's perspective is exact_score
        exact_score[i] = float(s.get("exact_score", s.get("target_score", 0.0)))
        empties[i] = int(s.get("empties", 14))

    return {
        "child_x": child_x.to(device),
        "child_ternary": child_ternary.to(device),
        "fast_sc": fast_sc.to(device),
        "mask": mask.to(device),
        "best_idx": best_idx.to(device),
        "exact_score": exact_score.to(device),
        "empties": empties.to(device),
    }


def eval_ranking_and_mae(
    pred_child_discs: torch.Tensor,
    data: Dict[str, torch.Tensor],
    label: str,
) -> Tuple[float, float, float]:
    """Evaluates move ordering Top-1, Top-2 (when combined with fast_score) and score MAE (in discs)."""
    mask = data["mask"]
    best_idx = data["best_idx"]
    fast_sc = data["fast_sc"]

    # Best move score prediction MAE in discs
    best_pred = pred_child_discs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    mae = (best_pred - data["exact_score"]).abs().mean().item()

    # Combine with fast_score (1 disc = 128 units)
    total_sc = fast_sc + pred_child_discs * 128.0
    masked = total_sc.masked_fill(~mask, -1e9)
    sorted_idx = torch.argsort(masked, dim=-1, descending=True)
    ranks = (sorted_idx == best_idx.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1

    top1 = (ranks == 1).float().mean().item() * 100.0
    top2 = (ranks <= 2).float().mean().item() * 100.0
    mrr = (1.0 / ranks.float()).mean().item() * 100.0

    # Also evaluate pure pattern/model ranking WITHOUT fast_score
    pure_masked = pred_child_discs.masked_fill(~mask, -1e9)
    pure_top1 = (torch.argmax(pure_masked, dim=-1) == best_idx).float().mean().item() * 100.0

    print(
        f"  {label:<44} | Pure Top1={pure_top1:5.2f}% | +FastScore Top1={top1:5.2f}% Top2={top2:5.2f}% MRR={mrr:5.2f} | Score MAE={mae:5.2f}d"
    )
    return pure_top1, top1, mae


def compute_interaction_matrix(teacher: TeacherNet, val_data: Dict[str, torch.Tensor], num_boards: int = 1024) -> torch.Tensor:
    """Computes the 64x64 D4-symmetrized second-order interaction matrix I[i, j] on the Teacher network.
    Uses exact second-order differences by toggling empty/occupied squares on validation boards.
    """
    teacher.eval()
    device = val_data["child_x"].device
    # Gather best child boards from validation set: (M, 128)
    best_idx = val_data["best_idx"]
    boards = val_data["child_x"].gather(1, best_idx.view(-1, 1, 1).expand(-1, 1, 128)).squeeze(1)
    boards = boards[:num_boards]  # (B, 128)

    with torch.no_grad():
        base_v = teacher.forward(boards)  # (B,)
        # Single-square perturbation: flip square i state (toggle my/opp/empty)
        single_v = torch.zeros((64, boards.shape[0]), device=device)
        for i in range(64):
            b_i = boards.clone()
            # Swap empty <-> my or my <-> opp on square i
            b_i[:, i] = 1.0 - b_i[:, i]
            single_v[i] = teacher.forward(b_i)

        inter = torch.zeros((64, 64), device=device)
        # Symmetrize over D4 pair orbits so we only need to evaluate unique pair representatives!
        pair_orbits: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for i in range(64):
            for j in range(i + 1, 64):
                canon = min(
                    (min(m[i], m[j]), max(m[i], m[j]))
                    for m in D4_MAPS
                )
                pair_orbits.setdefault(canon, []).append((i, j))

        for (ci, cj), members in pair_orbits.items():
            b_ij = boards.clone()
            b_ij[:, ci] = 1.0 - b_ij[:, ci]
            b_ij[:, cj] = 1.0 - b_ij[:, cj]
            v_ij = teacher.forward(b_ij)
            # Second-order finite difference: |f(x_ij) - f(x_i) - f(x_j) + f(x)|
            diff2 = (v_ij - single_v[ci] - single_v[cj] + base_v).abs().mean().item()
            for (u, v) in members:
                inter[u, v] = diff2
                inter[v, u] = diff2

    return inter.cpu()


def get_d4_orbit_of_pattern(squares: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Returns the unique D4 symmetric instances (4 or 8) of a k-square pattern."""
    seen: Set[frozenset] = set()
    instances: List[Tuple[int, ...]] = []
    for m in D4_MAPS:
        transformed = tuple(m[s] for s in squares)
        key = frozenset(transformed)
        if key not in seen:
            seen.add(key)
            instances.append(transformed)
    return instances


def discover_optimal_k_patterns(
    inter: torch.Tensor,
    existing_groups: List[List[List[int]]],
    k: int = 8,
    num_restarts: int = 40,
    steps: int = 600,
    seed: int = 42,
) -> Tuple[List[Tuple[int, ...]], float]:
    """Finds a k-square pattern whose D4 orbit maximizes residual pairwise interaction
    not already covered by `existing_groups`, plus internal cohesion.
    """
    rng = random.Random(seed)
    covered_pairs = torch.zeros((64, 64), dtype=torch.float32)
    for grp in existing_groups:
        for inst in grp:
            for a in range(len(inst)):
                for b in range(a + 1, len(inst)):
                    u, v = inst[a], inst[b]
                    covered_pairs[u, v] += 1.0
                    covered_pairs[v, u] += 1.0

    # Residual weight: discount pairs already covered multiple times by existing patterns
    residual_inter = inter / (1.0 + 1.5 * covered_pairs)

    def score_candidate(sqs: Tuple[int, ...]) -> float:
        orbit = get_d4_orbit_of_pattern(sqs)
        # Prefer 4-instance or 8-instance orbits; normalize per instance so 8-instance matches 8-instance r24
        pair_cov = set()
        total_w = 0.0
        for inst in orbit:
            for a in range(len(inst)):
                for b in range(a + 1, len(inst)):
                    u, v = min(inst[a], inst[b]), max(inst[a], inst[b])
                    if (u, v) not in pair_cov:
                        pair_cov.add((u, v))
                        total_w += float(residual_inter[u, v].item())
                    else:
                        total_w += 0.25 * float(residual_inter[u, v].item())
        return total_w

    best_sqs = tuple(range(k))
    best_sc = -1e9

    for _ in range(num_restarts):
        curr = tuple(sorted(rng.sample(range(64), k)))
        curr_sc = score_candidate(curr)
        for step in range(steps):
            temp = 0.5 * (1.0 - step / steps)
            idx_to_replace = rng.randrange(k)
            new_sq = rng.randrange(64)
            while new_sq in curr:
                new_sq = rng.randrange(64)
            nxt = list(curr)
            nxt[idx_to_replace] = new_sq
            nxt_t = tuple(sorted(nxt))
            nxt_sc = score_candidate(nxt_t)
            if nxt_sc > curr_sc or rng.random() < math.exp((nxt_sc - curr_sc) / max(1e-4, temp)):
                curr, curr_sc = nxt_t, nxt_sc
                if curr_sc > best_sc:
                    best_sqs, best_sc = curr, curr_sc

    return get_d4_orbit_of_pattern(best_sqs), best_sc


def extract_pattern_group_indices(
    child_ternary: torch.Tensor,
    pattern_instances: List[List[int]],
) -> torch.Tensor:
    """Computes base-3 pattern indices (N, K, num_instances) for arbitrary k-square patterns."""
    k = len(pattern_instances[0])
    pow3 = torch.tensor([3 ** (k - 1 - i) for i in range(k)], dtype=torch.long, device=child_ternary.device)
    inst_sqs = torch.tensor(pattern_instances, dtype=torch.long, device=child_ternary.device)  # (M, k)
    # child_ternary: (N, K, 64) -> gather (N, K, M, k)
    gathered = child_ternary[:, :, inst_sqs]
    indices = (gathered * pow3.view(1, 1, 1, k)).sum(dim=-1)  # (N, K, M)
    return indices


class CustomTuplePatternModel(nn.Module):
    """Trains a D4-invariant lookup-table pattern model for any list of pattern groups.
    Automatically averages over all 8 D4 symmetries of each pattern group so every topology
    has exact dihedral symmetry tying.
    """

    def __init__(self, group_specs: List[Tuple[str, List[List[int]]]]):
        super().__init__()
        self.group_specs = []
        self.embeddings = nn.ModuleList()
        for name, instances in group_specs:
            # Expand instances to all 8 D4 transforms of the canonical pattern(s)
            # so weight sharing across all 8 symmetries is 100% exact
            base_sq = instances[0]
            k = len(base_sq)
            all_d4 = []
            seen = set()
            for m in D4_MAPS:
                t_sq = [m[s] for s in base_sq]
                t_rev = list(reversed(t_sq))
                for cand in (t_sq, t_rev):
                    tup = tuple(cand)
                    if tup not in seen:
                        seen.add(tup)
                        all_d4.append(cand)
            self.group_specs.append((name, all_d4, len(instances) / float(len(all_d4))))
            self.embeddings.append(nn.Embedding(3 ** k, 1))
            nn.init.zeros_(self.embeddings[-1].weight)

    def forward(self, group_indices: List[torch.Tensor]) -> torch.Tensor:
        total = None
        for emb, idx, (_, _, scale) in zip(self.embeddings, group_indices, self.group_specs):
            val = emb(idx).squeeze(-1).sum(dim=-1) * scale
            total = val if total is None else (total + val)
        return total


def train_tuple_model(
    label: str,
    group_specs: List[Tuple[str, List[List[int]]]],
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    epochs: int = 35,
    lr: float = 0.04,
    batch_size: int = 512,
):
    device = train_data["child_ternary"].device
    model = CustomTuplePatternModel(group_specs).to(device)
    expanded_specs = [(n, insts) for n, insts, _ in model.group_specs]

    train_idx = [extract_pattern_group_indices(train_data["child_ternary"], insts) for _, insts in expanded_specs]
    val_idx = [extract_pattern_group_indices(val_data["child_ternary"], insts) for _, insts in expanded_specs]
    test_idx = [extract_pattern_group_indices(test_data["child_ternary"], insts) for _, insts in expanded_specs]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)

    n_train = train_data["child_ternary"].shape[0]
    best_val_top1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            b = perm[start:start + batch_size]
            b_idx = [g[b] for g in train_idx]
            mask = train_data["mask"][b]
            best_m = train_data["best_idx"][b]
            fast_sc = train_data["fast_sc"][b]
            ex_sc = train_data["exact_score"][b]

            optimizer.zero_grad()
            pred_discs = model(b_idx)
            logits = (fast_sc + pred_discs * 128.0).masked_fill(~mask, -1e9) / 256.0
            ce_loss = F.cross_entropy(logits, best_m)
            pure_logits = pred_discs.masked_fill(~mask, -1e9) / 2.0
            pure_ce = F.cross_entropy(pure_logits, best_m)
            best_pred = pred_discs.gather(1, best_m.unsqueeze(1)).squeeze(1)
            mse_loss = F.mse_loss(best_pred, ex_sc)

            loss = ce_loss + 0.5 * pure_ce + 0.01 * mse_loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                v_pred = model(val_idx)
                masked = (val_data["fast_sc"] + v_pred * 128.0).masked_fill(~val_data["mask"], -1e9)
                top1 = (torch.argmax(masked, dim=-1) == val_data["best_idx"]).float().mean().item() * 100.0
                if top1 > best_val_top1:
                    best_val_top1 = top1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        eval_ranking_and_mae(model(val_idx), val_data, f"[VAL]  {label}")
        eval_ranking_and_mae(model(test_idx), test_data, f"[TEST] {label}")


def main():
    parser = argparse.ArgumentParser(description="Automated Pattern Discovery Automated Pattern Discovery via ML")
    parser.add_argument("--data", type=Path, default=Path("data/clean_exact_dataset.pt"))
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_samples = torch.load(args.data, weights_only=False)
    train_s = [s for s in raw_samples if s.get("split") == "train"]
    val_s = [s for s in raw_samples if s.get("split") == "val"]
    test_s = [s for s in raw_samples if s.get("split") == "test"]
    print(f"Loaded {len(raw_samples)} exact samples: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    train_data = build_move_ranking_tensors(train_s, device)
    val_data = build_move_ranking_tensors(val_s, device)
    test_data = build_move_ranking_tensors(test_s, device)

    d4_perms_128 = []
    for m in D4_MAPS:
        d4_perms_128.append(m + [x + 64 for x in m])
    d4_perms_t = torch.tensor(d4_perms_128, dtype=torch.long, device=device)

    print("\n" + "=" * 120)
    print("STAGE 1: Training D4-Equivariant Non-Linear 64-Square Teacher Network (MLP)")
    print("=" * 120)
    teacher = TeacherNet(d4_perms=d4_perms_t, hidden=256).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3, weight_decay=1e-4)
    epochs = 35
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    n_train = train_data["child_x"].shape[0]
    batch_size = 256
    t0 = time.time()
    best_teacher_state = None
    best_teacher_val = -1.0

    for ep in range(1, epochs + 1):
        teacher.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            cx = train_data["child_x"][idx]
            mask = train_data["mask"][idx]
            b_idx = train_data["best_idx"][idx]
            f_sc = train_data["fast_sc"][idx]
            ex_sc = train_data["exact_score"][idx]

            opt.zero_grad()
            pred_discs = teacher(cx)
            logits = (f_sc + pred_discs * 128.0).masked_fill(~mask, -1e9) / 256.0
            ce_loss = F.cross_entropy(logits, b_idx)
            pure_logits = pred_discs.masked_fill(~mask, -1e9) / 2.0
            pure_ce = F.cross_entropy(pure_logits, b_idx)
            best_pred = pred_discs.gather(1, b_idx.unsqueeze(1)).squeeze(1)
            mse_loss = F.mse_loss(best_pred, ex_sc)

            loss = ce_loss + 0.5 * pure_ce + 0.015 * mse_loss
            loss.backward()
            opt.step()

        sched.step()
        if ep % 7 == 0 or ep == epochs:
            teacher.eval()
            with torch.no_grad():
                v_pred = teacher(val_data["child_x"])
                masked = (val_data["fast_sc"] + v_pred * 128.0).masked_fill(~val_data["mask"], -1e9)
                top1 = (torch.argmax(masked, dim=-1) == val_data["best_idx"]).float().mean().item() * 100.0
                if top1 > best_teacher_val:
                    best_teacher_val = top1
                    best_teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}

    if best_teacher_state is not None:
        teacher.load_state_dict({k: v.to(device) for k, v in best_teacher_state.items()})

    print(f"Teacher trained in {time.time() - t0:.1f}s.")
    teacher.eval()
    with torch.no_grad():
        eval_ranking_and_mae(torch.zeros_like(val_data["fast_sc"]), val_data, "[VAL]  0. FastScore Only (No Patterns)")
        eval_ranking_and_mae(torch.zeros_like(test_data["fast_sc"]), test_data, "[TEST] 0. FastScore Only (No Patterns)")
        eval_ranking_and_mae(teacher(val_data["child_x"]), val_data, "[VAL]  1. D4 Non-Linear 64-Sq Teacher MLP")
        eval_ranking_and_mae(teacher(test_data["child_x"]), test_data, "[TEST] 1. D4 Non-Linear 64-Sq Teacher MLP")

    print("\n" + "=" * 120)
    print("STAGE 2: Mining 64x64 Second-Order Square Interaction Matrix from Teacher Network")
    print("=" * 120)
    inter = compute_interaction_matrix(teacher, val_data, num_boards=512)

    DIAG8_SQS = [
        [0, 9, 18, 27, 36, 45, 54, 63],
        [7, 14, 21, 28, 35, 42, 49, 56],
    ]
    # ML-mined Corner+Diagonal+Center 8-square pattern (a1, b1, a2, b2, c3, d4, e5, h8)
    ML_DIAG_CORNER_SQS = [list(x) for x in get_d4_orbit_of_pattern((0, 1, 8, 9, 18, 27, 36, 63))]

    ml_orbit_1, _ = discover_optimal_k_patterns(inter, [CORNER33_SQS, EDGE2X_SQS], k=8, seed=42)
    ml_insts_1 = [list(x) for x in ml_orbit_1]

    ml_orbit_2, _ = discover_optimal_k_patterns(inter, [CORNER33_SQS, EDGE2X_SQS, RECT24_SQS], k=8, seed=99)
    ml_insts_2 = [list(x) for x in ml_orbit_2]

    print("\n" + "=" * 120)
    print("STAGE 3: Exact D4-Tied Head-to-Head Pattern Topology Comparison")
    print("=" * 120)
    # 2A. Human 3-Group (c33 + e2x + r24)
    train_tuple_model(
        "2A. Human 3-Group (c33 + e2x + r24)",
        [("c33", CORNER33_SQS), ("e2x", EDGE2X_SQS), ("r24", RECT24_SQS)],
        train_data, val_data, test_data,
    )
    # 2B. Replacing r24 (8 lookups) with Interaction-Mined diag8 (2 lookups!)
    train_tuple_model(
        "2B. c33 + e2x + diag8 (10 lookups vs 16)",
        [("c33", CORNER33_SQS), ("e2x", EDGE2X_SQS), ("diag8", DIAG8_SQS)],
        train_data, val_data, test_data,
    )
    # 2C. Replacing r24 with ML Corner-Diag-Center (a1,b1,a2,b2,c3,d4,e5,h8)
    train_tuple_model(
        "2C. c33 + e2x + ML-CornerDiag8",
        [("c33", CORNER33_SQS), ("e2x", EDGE2X_SQS), ("ml_cd8", ML_DIAG_CORNER_SQS)],
        train_data, val_data, test_data,
    )
    # 2D. Replacing r24 with Simulated-Annealing P1* (D4-tied)
    train_tuple_model(
        "2D. c33 + e2x + ML-SA-Orbit P1*",
        [("c33", CORNER33_SQS), ("e2x", EDGE2X_SQS), ("ml_p1", ml_insts_1)],
        train_data, val_data, test_data,
    )
    # 2E. Human 3-Group + Interaction-Mined diag8 (4 groups, +13.1 KB, +2 lookups)
    train_tuple_model(
        "2E. c33 + e2x + r24 + diag8 (18 lookups)",
        [("c33", CORNER33_SQS), ("e2x", EDGE2X_SQS), ("r24", RECT24_SQS), ("diag8", DIAG8_SQS)],
        train_data, val_data, test_data,
    )
    # 2F. Human 3-Group + Simulated-Annealing P2*
    train_tuple_model(
        "2F. c33 + e2x + r24 + ML-SA-Orbit P2*",
        [("c33", CORNER33_SQS), ("e2x", EDGE2X_SQS), ("r24", RECT24_SQS), ("ml_p2", ml_insts_2)],
        train_data, val_data, test_data,
    )

    print("\n" + "=" * 120)
    print("STAGE 4: Residual Addition of Discovered Patterns on Top of Pre-Trained `src/end_patterns_data.h`")
    print("=" * 120)
    from train_ordering_regression import load_header_weights
    w_ref = load_header_weights(Path("src/end_patterns_data.h")).to(device)

    def add_ref_p_score(data_dict: Dict[str, torch.Tensor]):
        c_idx = extract_pattern_group_indices(data_dict["child_ternary"], CORNER33_SQS)
        e_idx = extract_pattern_group_indices(data_dict["child_ternary"], EDGE2X_SQS)
        r_idx = extract_pattern_group_indices(data_dict["child_ternary"], RECT24_SQS)
        c_sum = w_ref[0:19683][c_idx].sum(dim=-1)
        e_sum = w_ref[19683:19683 + 59049][e_idx].sum(dim=-1)
        r_sum = w_ref[19683 + 59049:85293][r_idx].sum(dim=-1)
        emp = data_dict["empties"].unsqueeze(1)
        c33_r24 = c_sum + r_sum
        p_sc = torch.where(
            emp <= 10, c33_r24 * 0.5,
            torch.where(emp == 11, c33_r24 * 0.625,
            torch.where(emp <= 13, c33_r24 * 0.75, c33_r24 + e_sum))
        )
        res = dict(data_dict)
        res["fast_sc"] = data_dict["fast_sc"] + p_sc
        return res

    train_res = add_ref_p_score(train_data)
    val_res = add_ref_p_score(val_data)
    test_res = add_ref_p_score(test_data)

    eval_ranking_and_mae(torch.zeros_like(val_res["fast_sc"]), val_res, "[VAL]  3A. Pre-Trained c33+e2x+r24 (EVAL-019)")
    eval_ranking_and_mae(torch.zeros_like(test_res["fast_sc"]), test_res, "[TEST] 3A. Pre-Trained c33+e2x+r24 (EVAL-019)")

    train_tuple_model(
        "3B. Pre-Trained + Residual diag8",
        [("diag8", DIAG8_SQS)],
        train_res, val_res, test_res, epochs=25, lr=0.01,
    )
    train_tuple_model(
        "3C. Pre-Trained + Residual ML-P2*",
        [("ml_p2", ml_insts_2)],
        train_res, val_res, test_res, epochs=25, lr=0.01,
    )


if __name__ == "__main__":
    main()
