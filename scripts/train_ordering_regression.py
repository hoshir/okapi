#!/usr/bin/env python3
"""train_ordering_regression.py - Joint Calibration of Endgame `fast_score` Linear Weights
and 16-Pattern LTR Table (`src/end_patterns_data.h`) on Decontaminated Exact + Cutoff Data (EVAL-019).

Strictly uses ONLY zero-runtime-cost features available in `src/end.c`:
  - `opp_mob` (locked at -128.0 scale anchor)
  - `opp_corner_mob`
  - `is_corner`
  - `is_open_x`
  - `is_open_c`
  - `quad_parity`
  - `flips`
  - `is_wipeout` (`opp_mob == 0`)
across 3 depth bands:
  - Band 0: `empties <= 10` (`9..10` empties)
  - Band 1: `empties 11..13` (`<= FAST_PATTERN_DEPTH`)
  - Band 2: `empties >= 14` (`> FAST_PATTERN_DEPTH`)
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from coeffs_tool import _MIRROR_MAPS, init_mirror_maps

CORNER33_COUNT = 19683
EDGE2X_COUNT = 59049
RECT24_COUNT = 6561
TOTAL_WEIGHTS = CORNER33_COUNT + EDGE2X_COUNT + RECT24_COUNT  # 85,293


def load_header_weights(path: Path) -> torch.Tensor:
    with open(path, "r") as f:
        text = f.read()
    match = re.search(r"end_pattern_weights\[\d+\]\s*=\s*\{([^}]+)\};", text)
    if not match:
        raise ValueError(f"Could not find end_pattern_weights array in {path}")
    raw_nums = [int(x) for x in re.findall(r"-?\d+", match.group(1))]
    assert len(raw_nums) == TOTAL_WEIGHTS, f"Expected {TOTAL_WEIGHTS} weights, found {len(raw_nums)}"
    return torch.tensor(raw_nums, dtype=torch.float32)


def sample_signature(s: dict, m33: List[int], m8x2: List[int]) -> Tuple[Tuple[int, ...], ...]:
    f16 = s["child_features_16"] if "child_features_16" in s else [m[:16] for m in s["child_features_18"]]
    sigs = []
    for m in f16:
        c = tuple(sorted(m33[x] for x in m[0:4]))
        e = tuple(sorted(m8x2[x] for x in m[4:8]))
        sigs.append(c + e)
    return tuple(sorted(sigs))


def build_split_tensors(samples: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Packs variable-length candidate move lists into padded batch tensors."""
    n = len(samples)
    max_k = max(
        len(s["child_features_16"] if "child_features_16" in s else s["child_features_18"])
        for s in samples
    )

    pat_idx = torch.zeros((n, max_k, 16), dtype=torch.long)
    # Zero-cost runtime tactical features (8):
    # [0] opp_mob, [1] opp_corner_mob, [2] is_corner, [3] is_open_x,
    # [4] is_open_c, [5] quad_parity, [6] flips, [7] is_wipeout
    tact = torch.zeros((n, max_k, 8), dtype=torch.float32)
    has_tact = torch.zeros(n, dtype=torch.bool)
    mask = torch.zeros((n, max_k), dtype=torch.bool)
    best_idx = torch.zeros(n, dtype=torch.long)
    empties = torch.zeros(n, dtype=torch.long)
    target_score = torch.zeros(n, dtype=torch.float32)
    has_score = torch.zeros(n, dtype=torch.bool)

    for i, s in enumerate(samples):
        f16 = s["child_features_16"] if "child_features_16" in s else [m[:16] for m in s["child_features_18"]]
        k = len(f16)
        pat_idx[i, :k] = torch.tensor(f16, dtype=torch.long)
        if "child_tactical" in s:
            tc_t = torch.tensor(s["child_tactical"], dtype=torch.float32)
            # Raw child_tactical layout in generate_clean_dataset.py:
            # [0] opp_mob, [1] opp_corner_moves, [2] eff_mob, [3] pot_mob_opp, [4] pot_mob_my,
            # [5] is_corner, [6] is_open_x, [7] is_open_c, [8] parity, [9] flips
            tact[i, :k, 0] = tc_t[:, 0]
            tact[i, :k, 1] = tc_t[:, 1]
            tact[i, :k, 2] = tc_t[:, 5]
            tact[i, :k, 3] = tc_t[:, 6]
            tact[i, :k, 4] = tc_t[:, 7]
            tact[i, :k, 5] = tc_t[:, 8]
            tact[i, :k, 6] = tc_t[:, 9]
            tact[i, :k, 7] = (tc_t[:, 0] == 0).float()
            has_tact[i] = True
        mask[i, :k] = True
        best_idx[i] = int(s["best_idx"])
        empties[i] = int(s.get("empties", 64 - s.get("disc_count", 48)))
        if "target_score" in s:
            target_score[i] = float(s["target_score"])
            has_score[i] = True

    return {
        "pat_idx": pat_idx.to(device),
        "tact": tact.to(device),
        "has_tact": has_tact.to(device),
        "mask": mask.to(device),
        "best_idx": best_idx.to(device),
        "empties": empties.to(device),
        "target_score": target_score.to(device),
        "has_score": has_score.to(device),
    }


def apply_lazy_margin(
    fast_score: torch.Tensor,
    p_score: torch.Tensor,
    data: Dict[str, torch.Tensor],
    lazy_margin: float = 768.0,
) -> torch.Tensor:
    """Simulates the exact two-stage Branch & Bound Pattern Evaluation in `src/end.c:1258-1289`."""
    mask = data["mask"]
    is_corner = data["tact"][:, :, 2] > 0.5
    masked_fast = fast_score.masked_fill(~mask, -1e9)
    top_fast, _ = masked_fast.max(dim=-1, keepdim=True)
    qualifies = ((top_fast - fast_score) <= lazy_margin) | is_corner
    return torch.where(qualifies, fast_score + p_score, fast_score - 10000.0)


def compute_current_engine_scores(
    data: Dict[str, torch.Tensor],
    w_all: torch.Tensor,
    fast_pattern_depth: int = 13,
    lazy_margin: float = 768.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the exact move ordering scores used by `src/end.c` right now (both full and Lazy768)."""
    pat_idx = data["pat_idx"]
    tact = data["tact"]
    empties = data["empties"]

    w_c33 = w_all[0:CORNER33_COUNT]
    w_e2x = w_all[CORNER33_COUNT:CORNER33_COUNT + EDGE2X_COUNT]
    w_r24 = w_all[CORNER33_COUNT + EDGE2X_COUNT:TOTAL_WEIGHTS]

    c33_sum = w_c33[pat_idx[:, :, 0:4]].sum(dim=-1)
    e2x_sum = w_e2x[pat_idx[:, :, 4:8]].sum(dim=-1)
    r24_sum = w_r24[pat_idx[:, :, 8:16]].sum(dim=-1)

    opp_mob = tact[:, :, 0]
    opp_corner = tact[:, :, 1]
    is_open_x = tact[:, :, 3]
    is_open_c = tact[:, :, 4]
    parity = tact[:, :, 5]
    is_wipeout = tact[:, :, 7]

    c_sq_pen = 128.0 * torch.clamp(is_open_x + is_open_c, 0.0, 1.0)
    fast_score = (
        -128.0 * (opp_mob + opp_corner)
        - c_sq_pen
        + 4096.0 * is_wipeout
        + 64.0 * parity
    )

    shallow_mask = (empties <= fast_pattern_depth).unsqueeze(-1)
    p_score = torch.where(
        shallow_mask,
        (c33_sum + r24_sum) * 0.5,
        c33_sum + e2x_sum + r24_sum,
    )
    full_scores = fast_score + p_score
    lazy_scores = apply_lazy_margin(fast_score, p_score, data, lazy_margin=lazy_margin)
    return full_scores, lazy_scores


def evaluate_scores(
    scores: torch.Tensor,
    data: Dict[str, torch.Tensor],
    label: str = "",
) -> Dict[str, float]:
    mask = data["mask"]
    best_idx = data["best_idx"]
    empties = data["empties"]

    masked_scores = scores.masked_fill(~mask, -1e9)
    sorted_indices = torch.argsort(masked_scores, dim=-1, descending=True)
    ranks = (sorted_indices == best_idx.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1

    def metrics_for_subset(sub_mask: torch.Tensor) -> Tuple[float, float, float, int]:
        cnt = int(sub_mask.sum().item())
        if cnt == 0:
            return 0.0, 0.0, 0.0, 0
        r = ranks[sub_mask].float()
        top1 = (r == 1).float().mean().item() * 100.0
        top2 = (r <= 2).float().mean().item() * 100.0
        mrr = (1.0 / r).mean().item() * 100.0
        return top1, top2, mrr, cnt

    all_m = torch.ones_like(empties, dtype=torch.bool)
    b_9_10 = (empties >= 9) & (empties <= 10)
    b_11_13 = (empties >= 11) & (empties <= 13)
    b_14_20 = empties >= 14

    t1_all, t2_all, mrr_all, n_all = metrics_for_subset(all_m)
    t1_9, _, _, _ = metrics_for_subset(b_9_10)
    t1_11, _, _, _ = metrics_for_subset(b_11_13)
    t1_14, _, _, _ = metrics_for_subset(b_14_20)

    if label:
        print(
            f"  {label:<42} | All({n_all}): Top1={t1_all:5.2f}% Top2={t2_all:5.2f}% MRR={mrr_all:5.2f} "
            f"| 9-10: {t1_9:5.2f}% | 11-13: {t1_11:5.2f}% | 14-20: {t1_14:5.2f}%"
        )

    return {
        "top1_all": t1_all,
        "top2_all": t2_all,
        "mrr_all": mrr_all,
        "top1_9_10": t1_9,
        "top1_11_13": t1_11,
        "top1_14_20": t1_14,
    }


class JointOrderingModel(nn.Module):
    """3-band zero-cost tactical weights + mirror-tied 16-pattern weights."""

    def __init__(
        self,
        w_init: torch.Tensor,
        m33: torch.Tensor,
        m8x2: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("m33", m33)
        self.register_buffer("m8x2", m8x2)

        self.c33_w = nn.Parameter(w_init[0:CORNER33_COUNT].clone())
        self.e2x_w = nn.Parameter(w_init[CORNER33_COUNT:CORNER33_COUNT + EDGE2X_COUNT].clone())
        self.r24_w = nn.Parameter(w_init[CORNER33_COUNT + EDGE2X_COUNT:TOTAL_WEIGHTS].clone())

        self.register_buffer("c33_init", self.c33_w.detach().clone())
        self.register_buffer("e2x_init", self.e2x_w.detach().clone())
        self.register_buffer("r24_init", self.r24_w.detach().clone())

        # 8 zero-cost tactical features:
        # [0] opp_mob (locked -128), [1] opp_corner_mob, [2] is_corner, [3] is_open_x,
        # [4] is_open_c, [5] quad_parity, [6] flips, [7] is_wipeout
        init_tact = torch.tensor(
            [-128.0, -128.0, 0.0, -128.0, -128.0, 64.0, 0.0, 4096.0],
            dtype=torch.float32,
        )
        # 3 depth bands: [0] 9..10, [1] 11..13, [2] 14..20
        self.tact_w = nn.Parameter(torch.stack([init_tact.clone(), init_tact.clone(), init_tact.clone()], dim=0))

        # Pattern scale for shallow bands (band 0: 9..10, band 1: 11..13, band 2: 14..20)
        # Current engine uses 0.5 for <=13 and 1.0 for >=14
        self.shallow_pat_scale = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.deep_e2x_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def get_band_idx(self, empties: torch.Tensor) -> torch.Tensor:
        return torch.where(
            empties <= 10,
            torch.zeros_like(empties),
            torch.where(empties <= 13, torch.ones_like(empties), torch.full_like(empties, 2)),
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pat_idx = data["pat_idx"]
        tact = data["tact"]
        has_tact = data["has_tact"]
        empties = data["empties"]

        c_idx = self.m33[pat_idx[:, :, 0:4]]
        e_idx = self.m8x2[pat_idx[:, :, 4:8]]
        r_idx = pat_idx[:, :, 8:16]

        c33_sum = self.c33_w[c_idx].sum(dim=-1)
        e2x_sum = self.e2x_w[e_idx].sum(dim=-1)
        r24_sum = self.r24_w[r_idx].sum(dim=-1)

        band = self.get_band_idx(empties)  # (N,) in {0, 1, 2}
        tw = self.tact_w[band]             # (N, 8)
        tw = torch.cat([torch.full((tw.shape[0], 1), -128.0, device=tw.device), tw[:, 1:]], dim=1)

        fast_score = (tact * tw.unsqueeze(1)).sum(dim=-1)
        fast_score = torch.where(has_tact.unsqueeze(1), fast_score, torch.zeros_like(fast_score))

        s0 = self.shallow_pat_scale[0]
        s1 = self.shallow_pat_scale[1]
        p_Band0 = s0 * (c33_sum + r24_sum)
        p_Band1 = s1 * (c33_sum + r24_sum)
        p_Band2 = c33_sum + self.deep_e2x_scale * e2x_sum + r24_sum

        b_exp = band.unsqueeze(1)
        p_score = torch.where(b_exp == 0, p_Band0, torch.where(b_exp == 1, p_Band1, p_Band2))
        total_score = fast_score + p_score
        raw_pat = c33_sum + self.deep_e2x_scale * e2x_sum + r24_sum
        return total_score, raw_pat, fast_score, p_score

    def prior_l2_loss(self) -> torch.Tensor:
        return (
            F.mse_loss(self.c33_w, self.c33_init)
            + F.mse_loss(self.e2x_w, self.e2x_init)
            + F.mse_loss(self.r24_w, self.r24_init)
        )


def train_model(
    name: str,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    w_init: torch.Tensor,
    m33: torch.Tensor,
    m8x2: torch.Tensor,
    train_patterns: bool,
    train_tactical: bool,
    prior_reg: float = 0.05,
    lr_tact: float = 2.0,
    lr_pat: float = 0.4,
    epochs: int = 60,
    batch_size: int = 512,
) -> JointOrderingModel:
    device = train_data["pat_idx"].device
    model = JointOrderingModel(w_init=w_init, m33=m33, m8x2=m8x2).to(device)

    param_groups = []
    if train_tactical:
        param_groups.append(
            {
                "params": [model.tact_w, model.shallow_pat_scale, model.deep_e2x_scale],
                "lr": lr_tact,
                "weight_decay": 1e-4,
            }
        )
    else:
        model.tact_w.requires_grad_(False)
        model.shallow_pat_scale.requires_grad_(False)
        model.deep_e2x_scale.requires_grad_(False)

    if train_patterns:
        param_groups.append({"params": [model.c33_w, model.e2x_w, model.r24_w], "lr": lr_pat, "weight_decay": 0.0})
    else:
        model.c33_w.requires_grad_(False)
        model.e2x_w.requires_grad_(False)
        model.r24_w.requires_grad_(False)

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.05)

    n_train = train_data["pat_idx"].shape[0]
    best_val_sel = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch = {k: v[idx] for k, v in train_data.items()}

            optimizer.zero_grad()
            scores, raw_pat, _, _ = model(batch)
            masked_scores = scores.masked_fill(~batch["mask"], -1e9)

            logits = masked_scores / 256.0
            ce_loss = F.cross_entropy(logits, batch["best_idx"])

            best_sc = scores.gather(1, batch["best_idx"].unsqueeze(1))
            margin_loss = F.relu(128.0 - (best_sc - scores)).masked_fill(~batch["mask"], 0.0).sum(dim=-1).mean() / 256.0

            loss = ce_loss + 0.25 * margin_loss
            if batch["has_score"].any():
                hs = batch["has_score"]
                best_pat_discs = raw_pat[hs].gather(1, batch["best_idx"][hs].unsqueeze(1)).squeeze(1) / 128.0
                mse_loss = F.mse_loss(best_pat_discs, batch["target_score"][hs])
                loss = loss + 0.005 * mse_loss

            if train_patterns and prior_reg > 0.0:
                loss = loss + prior_reg * model.prior_l2_loss()

            loss.backward()
            optimizer.step()

        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_full, _, val_fast, val_p = model(val_data)
                val_lazy = apply_lazy_margin(val_fast, val_p, val_data)
                m = evaluate_scores(val_lazy, val_data, label="")
                sel = m["top1_all"] + 0.5 * m["top2_all"] + 0.5 * m["mrr_all"]
                if sel > best_val_sel:
                    best_val_sel = sel
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        val_full, _, val_fast, val_p = model(val_data)
        test_full, _, test_fast, test_p = model(test_data)
        evaluate_scores(val_full, val_data, label=f"[VAL-Full]     {name}")
        evaluate_scores(apply_lazy_margin(val_fast, val_p, val_data), val_data, label=f"[VAL-Lazy768]  {name}")
        evaluate_scores(test_full, test_data, label=f"[TEST-Full]    {name}")
        evaluate_scores(apply_lazy_margin(test_fast, test_p, test_data), test_data, label=f"[TEST-Lazy768] {name}")

    return model


def print_learned_tactical(model: JointOrderingModel, title: str):
    feat_names = [
        "opp_mob",
        "opp_corner_mob",
        "is_corner",
        "is_open_x",
        "is_open_c",
        "quad_parity",
        "flips",
        "is_wipeout",
    ]
    tw = model.tact_w.detach().cpu()
    tw[:, 0] = -128.0
    sps = model.shallow_pat_scale.detach().cpu()
    de2x = model.deep_e2x_scale.detach().cpu().item()
    print(f"\n--- {title}: Zero-Cost Runtime Parameters ---")
    print(f"  Pattern Scale: Band 9-10 = {sps[0]:.3f} (was 0.500), Band 11-13 = {sps[1]:.3f} (was 0.500), Deep e2x = {de2x:.3f} (was 1.000)")
    print(f"  {'Feature':<16} | {'Current C':>10} | {'Band 9-10':>12} | {'Band 11-13':>12} | {'Band 14-20':>12}")
    cur = [-128, -128, 0, -128, -128, 64, 0, 4096]
    for i, fn in enumerate(feat_names):
        print(f"  {fn:<16} | {cur[i]:10d} | {tw[0, i].item():12.1f} | {tw[1, i].item():12.1f} | {tw[2, i].item():12.1f}")


def export_model_header(model: JointOrderingModel, w_ref: torch.Tensor, val_data: Dict[str, torch.Tensor], out_path: Path):
    m33 = model.m33.cpu()
    m8x2 = model.m8x2.cpu()
    w_c33 = model.c33_w.detach().cpu()[m33]
    w_e2x = model.e2x_w.detach().cpu()[m8x2]
    w_r24 = model.r24_w.detach().cpu()

    ref_cpu = w_ref.detach().cpu()
    ref_c33 = ref_cpu[0:CORNER33_COUNT]
    ref_e2x = ref_cpu[CORNER33_COUNT:CORNER33_COUNT + EDGE2X_COUNT]
    ref_r24 = ref_cpu[CORNER33_COUNT + EDGE2X_COUNT:TOTAL_WEIGHTS]

    # Normalize each pattern group's standard deviation on validation moves to match w_ref exactly
    valid_moves = val_data["pat_idx"][val_data["mask"]].detach().cpu()
    c_idx = valid_moves[:, 0:4]
    e_idx = valid_moves[:, 4:8]
    r_idx = valid_moves[:, 8:16]

    for w_new, w_old, idx in [(w_c33, ref_c33, c_idx), (w_e2x, ref_e2x, e_idx), (w_r24, ref_r24, r_idx)]:
        std_new = w_new[idx].sum(dim=-1).std().item()
        std_old = w_old[idx].sum(dim=-1).std().item()
        if std_new > 1e-5:
            w_new.mul_(std_old / std_new)

    def clamp_i16(v: float) -> int:
        return max(-32767, min(32767, int(round(v))))

    full_weights = [clamp_i16(v) for v in (w_c33.tolist() + w_e2x.tolist() + w_r24.tolist())]
    assert len(full_weights) == TOTAL_WEIGHTS

    with open(out_path, "w") as f:
        f.write("/*\n")
        f.write("  File:       end_patterns_data.h\n")
        f.write("  Generated:  scripts/train_ordering_regression.py (EVAL-019)\n")
        f.write("  Model:      Jointly Calibrated 3-Pattern + Tactical Move Ordering Evaluator\n")
        f.write("  Geometry:   Corner 3x3 (19683), Edge+2X (59049), 2x4 Rect (6561)\n")
        f.write(f"  Total:      {TOTAL_WEIGHTS} entries (170.6 KB), 100% L2 cache resident\n")
        f.write("  Scale:      1 disc = 128.0 fixed-point units\n")
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
    print(f"\nExported group-std-normalized pattern header to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EVAL-019 Joint Ordering & LTR Calibration")
    parser.add_argument("--data", type=Path, default=Path("data/clean_exact_dataset.pt"))
    parser.add_argument("--cutoff-data", type=Path, default=Path("data/ltr_cutoff_dataset.pt"))
    parser.add_argument("--ref-header", type=Path, default=Path("src/end_patterns_data.h"))
    parser.add_argument("--export-header", type=Path, default=Path("/tmp/end_patterns_residual.h"))
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    init_mirror_maps()
    m33_list = _MIRROR_MAPS["mirror33"]
    m8x2_list = _MIRROR_MAPS["mirror8x2"]
    m33 = torch.tensor(m33_list, dtype=torch.long, device=device)
    m8x2 = torch.tensor(m8x2_list, dtype=torch.long, device=device)

    raw_samples = torch.load(args.data, weights_only=False)
    train_s = [s for s in raw_samples if s.get("split") == "train"]
    val_s = [s for s in raw_samples if s.get("split") == "val"]
    test_s = [s for s in raw_samples if s.get("split") == "test"]
    print(f"Loaded {len(raw_samples)} exact samples: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    train_data = build_split_tensors(train_s, device)
    val_data = build_split_tensors(val_s, device)
    test_data = build_split_tensors(test_s, device)

    w_ref = load_header_weights(args.ref_header).to(device)

    print("\n" + "=" * 120)
    print("0. Baseline Evaluation (Current `src/end.c` formula + Current `src/end_patterns_data.h`)")
    print("=" * 120)
    with torch.no_grad():
        val_full, val_lazy = compute_current_engine_scores(val_data, w_ref)
        test_full, test_lazy = compute_current_engine_scores(test_data, w_ref)
        evaluate_scores(val_lazy, val_data, label="[VAL-Lazy768]  Baseline0 (Current Engine)")
        evaluate_scores(test_lazy, test_data, label="[TEST-Lazy768] Baseline0 (Current Engine)")

    print("\n" + "=" * 120)
    print("1. Arm 1: Residual-Locked Clean LTR Fine-Tune (Exact `src/end.c` fast_score + Clean Data)")
    print("=" * 120)
    arm_res = train_model(
        name="Arm1 (Residual-Locked Clean LTR)",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        w_init=w_ref,
        m33=m33,
        m8x2=m8x2,
        train_patterns=True,
        train_tactical=False,
        prior_reg=0.25,
        lr_pat=0.15,
        epochs=25,
    )
    if args.export_header:
        export_model_header(arm_res, w_ref, val_data, args.export_header)


if __name__ == "__main__":
    main()
