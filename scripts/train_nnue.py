#!/usr/bin/env python3
"""
Othello NNUE Training & Export: Pattern-Free Othello NNUE Training, Quantization & C Header Export
============================================================================
Trains the E-Unified Pattern-Free Othello NNUE on the audited 100% engine-solved
dataset (`data/clean_exact_dataset.pt`, train=18,820, val=2,303, test=2,313):
  - 192-bit raw board input `(my_bits[0..63], opp_bits[0..63], empty_bits[0..63])`
    where `empty_bits` is folded into `(nnue_w_my, nnue_w_opp, nnue_bias)` at
    export time with zero inference overhead.
  - Incremental `int16_t` accumulator (`64` lanes = `8` D4 channels x `8` symmetries)
    with non-saturating soft L1 normalization.
  - All-Pairs Local & Global D4-equivariant SCReLU / Bilinear activations (`196` features).
  - 16 Corner 3x3 & Quadrant Empty Parity features (`pf[0..15]`) + 8 Zero-Cost
    Bitboard Tactical features (`tf[0..7]`: `opp_mob`, `opp_corner`, `is_corner`,
    `is_open_x_or_c`, `quad_parity`, `flipped`, `is_wipeout`, `is_single_reply`)
    + 8 Tactical SCReLU/Bilinear interactions (`ts[0..7]`) + 64 Spatial x Tactical/Parity
    cross-interactions (`cross_r[0..31]`, `cross_a[0..31]`) = 292 features total (NNUE-64).
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from discover_patterns_ml import (
    D4_MAPS,
    eval_ranking_and_mae,
)

C33_MASKS = [
    0x0000000000070707,  # a1..c3
    0x0000000000E0E0E0,  # f1..h3
    0x0707070000000000,  # a6..c8
    0xE0E0E00000000000,  # f6..h8
]
QUAD_MASKS = [
    0x000000000F0F0F0F,
    0x00000000F0F0F0F0,
    0x0F0F0F0F00000000,
    0xF0F0F0F000000000,
]
CORNER_BITS = [0, 7, 56, 63]
X_BITS = [9, 14, 49, 54]


def mask_to_64vec(m: int) -> torch.Tensor:
    return torch.tensor([(m >> sq) & 1 for sq in range(64)], dtype=torch.float32)


C33_MAT = torch.stack([mask_to_64vec(m) for m in C33_MASKS], dim=1)    # (64, 4)
QUAD_MAT = torch.stack([mask_to_64vec(m) for m in QUAD_MASKS], dim=1)  # (64, 4)


def upper_tri_indices(c: int) -> Tuple[List[int], List[int]]:
    idx1, idx2 = [], []
    for i in range(c):
        for j in range(i, c):
            idx1.append(i)
            idx2.append(j)
    return idx1, idx2


def build_e_unified_tensors(samples: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Builds padded candidate tensors including exact int16-scaled Corner Parity (16) and Tactical (8) features."""
    n = len(samples)
    max_k = max(len(s["child_my_bits"]) for s in samples)

    child_x = torch.zeros((n, max_k, 128), dtype=torch.float32)
    fast_sc = torch.zeros((n, max_k), dtype=torch.float32)
    pf_q = torch.zeros((n, max_k, 16), dtype=torch.int32)
    tf_q = torch.zeros((n, max_k, 8), dtype=torch.int32)
    tact_raw = torch.zeros((n, max_k, 6), dtype=torch.int32)
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
        my_b = ((torch.tensor(my_u, dtype=torch.int64).unsqueeze(1) >> shifts) & 1).float()
        opp_b = ((torch.tensor(opp_u, dtype=torch.int64).unsqueeze(1) >> shifts) & 1).float()
        emp_b = 1.0 - my_b - opp_b

        child_x[i, :k, 0:64] = my_b
        child_x[i, :k, 64:128] = opp_b

        c33_cnt = torch.round(emp_b @ C33_MAT).to(torch.int32)    # (k, 4)
        c33_odd = c33_cnt & 1                                     # (k, 4)
        quad_cnt = torch.round(emp_b @ QUAD_MAT).to(torch.int32)  # (k, 4)
        quad_odd = quad_cnt & 1                                   # (k, 4)

        c_emp = torch.round(emp_b[:, CORNER_BITS]).to(torch.int32)
        my_x = torch.round(my_b[:, X_BITS]).to(torch.int32)
        opp_x = torch.round(opp_b[:, X_BITS]).to(torch.int32)
        my_corn = torch.round(my_b[:, CORNER_BITS]).to(torch.int32)
        opp_corn = torch.round(opp_b[:, CORNER_BITS]).to(torch.int32)
        disc_diff = torch.round(my_b.sum(dim=-1) - opp_b.sum(dim=-1)).to(torch.int32)

        pf_q[i, :k, 0] = c33_odd.sum(dim=-1) << 5
        pf_q[i, :k, 1] = (c_emp & c33_odd).sum(dim=-1) << 5
        pf_q[i, :k, 2] = (c_emp & (1 - c33_odd)).sum(dim=-1) << 5
        pf_q[i, :k, 3] = (c33_cnt == 1).to(torch.int32).sum(dim=-1) << 5
        pf_q[i, :k, 4] = (c33_cnt == 2).to(torch.int32).sum(dim=-1) << 5
        pf_q[i, :k, 5] = (c33_cnt == 3).to(torch.int32).sum(dim=-1) << 5
        pf_q[i, :k, 6] = quad_odd.sum(dim=-1) << 5
        pf_q[i, :k, 7] = (quad_cnt == 1).to(torch.int32).sum(dim=-1) << 5
        pf_q[i, :k, 8] = c_emp.sum(dim=-1) << 5
        pf_q[i, :k, 9] = (c_emp & my_x).sum(dim=-1) << 5
        pf_q[i, :k, 10] = (c_emp & opp_x).sum(dim=-1) << 5
        pf_q[i, :k, 11] = my_corn.sum(dim=-1) << 5
        pf_q[i, :k, 12] = opp_corn.sum(dim=-1) << 5
        pf_q[i, :k, 13] = disc_diff << 2
        pf_q[i, :k, 14] = (c_emp & c33_odd & my_x).sum(dim=-1) << 5
        pf_q[i, :k, 15] = (c_emp & c33_odd & opp_x).sum(dim=-1) << 5

        tc_t = torch.tensor(tc, dtype=torch.float32)
        opp_mob = torch.round(tc_t[:, 0]).to(torch.int32)
        opp_corner = torch.round(tc_t[:, 1]).to(torch.int32)
        is_corner = torch.round(tc_t[:, 5]).to(torch.int32)
        is_open_x = torch.round(tc_t[:, 6]).to(torch.int32)
        is_open_c = torch.round(tc_t[:, 7]).to(torch.int32)
        parity = torch.round(tc_t[:, 8]).to(torch.int32)
        flips = torch.round(tc_t[:, 9]).to(torch.int32)

        is_open_xc = torch.clamp(is_open_x + is_open_c, 0, 1)
        is_wipeout = (opp_mob == 0).to(torch.int32)
        is_single = (opp_mob == 1).to(torch.int32)

        fast_sc[i, :k] = (
            -128.0 * (opp_mob + opp_corner).float()
            - 128.0 * is_open_xc.float()
            + 4096.0 * is_wipeout.float()
            + 64.0 * parity.float()
        )

        tact_raw[i, :k, 0] = opp_mob
        tact_raw[i, :k, 1] = opp_corner
        tact_raw[i, :k, 2] = is_corner
        tact_raw[i, :k, 3] = is_open_xc
        tact_raw[i, :k, 4] = parity
        tact_raw[i, :k, 5] = flips

        tf_q[i, :k, 0] = torch.clamp(opp_mob, 0, 16) << 3
        tf_q[i, :k, 1] = torch.clamp(opp_corner, 0, 4) << 6
        tf_q[i, :k, 2] = is_corner << 7
        tf_q[i, :k, 3] = is_open_xc << 7
        tf_q[i, :k, 4] = parity << 7
        tf_q[i, :k, 5] = torch.clamp(flips, 0, 16) << 3
        tf_q[i, :k, 6] = is_wipeout << 7
        tf_q[i, :k, 7] = is_single << 7

        mask[i, :k] = True
        best_idx[i] = int(s["best_idx"])
        exact_score[i] = float(s.get("exact_score", s.get("target_score", 0.0)))
        empties[i] = int(s.get("empties", 14))

    return {
        "child_x": child_x.to(device),
        "fast_sc": fast_sc.to(device),
        "pf_q": pf_q.to(device),
        "tf_q": tf_q.to(device),
        "tact_raw": tact_raw.to(device),
        "mask": mask.to(device),
        "best_idx": best_idx.to(device),
        "exact_score": exact_score.to(device),
        "empties": empties.to(device),
    }


class D4OthelloNNUE(nn.Module):
    """
    E-Unified Pattern-Free D4-Equivariant Othello NNUE:
      - 192-bit input `(my_64, opp_64, empty_64)` folded into `(w_my, w_opp, bias)`
      - Incremental `int16_t` accumulator (`H = 8 * C` lanes)
      - All-Pairs Local + Global D4 SCReLU / Bilinear features (`2*C + 5*P`)
      - 16 Corner 3x3 & Quadrant Empty Parity features (`pf[0..15]`)
      - 8 Zero-Cost Bitboard Tactical features (`tf[0..7]`)
      - 8 Tactical SCReLU / Bilinear interaction terms (`ts[0..7]`)
      - `8*C` Cross-Interactions (`cross_r`, `cross_a`) between spatial channels and
        `[opp_mob, quad_parity, open_corner_odd_c33, open_corner_x_diff]`.
    """

    def __init__(self, num_channels: int = 8, num_stage_buckets: int = 5):
        super().__init__()
        self.C = num_channels
        self.H = 8 * num_channels
        self.num_stage_buckets = num_stage_buckets

        self.w_my_canon = nn.Parameter(torch.randn(self.C, 64) * 0.15)
        self.w_opp_canon = nn.Parameter(torch.randn(self.C, 64) * 0.15)
        self.w_emp_canon = nn.Parameter(torch.randn(self.C, 64) * 0.15)
        self.bias_canon = nn.Parameter(torch.zeros(self.C))

        self.register_buffer("d4_perms", torch.tensor(D4_MAPS, dtype=torch.long))
        i1, i2 = upper_tri_indices(self.C)
        self.register_buffer("pair_i", torch.tensor(i1, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(i2, dtype=torch.long))

        self.num_pairs = len(i1)
        self.base_feat_dim = 2 * self.C + 5 * self.num_pairs
        self.extra_feat_dim = 16 + 8 + 8 + 8 * self.C
        self.feat_dim = self.base_feat_dim + self.extra_feat_dim

        self.order_weight = nn.Parameter(
            torch.randn(num_stage_buckets, self.feat_dim) * 0.02
        )
        self.order_bias = nn.Parameter(torch.zeros(num_stage_buckets))

        self.value_weight = nn.Parameter(
            torch.randn(num_stage_buckets, self.feat_dim) * 0.02
        )
        self.value_bias = nn.Parameter(torch.zeros(num_stage_buckets))

    def get_expanded_transformer(self, qat: bool = False):
        eff_my = self.w_my_canon - self.w_emp_canon
        eff_opp = self.w_opp_canon - self.w_emp_canon
        eff_bias = self.bias_canon + self.w_emp_canon.sum(dim=-1)

        l1_norm = (
            0.30
            * (
                eff_my.abs().sum(dim=-1, keepdim=True)
                + eff_opp.abs().sum(dim=-1, keepdim=True)
            )
            + 1.0
        )
        w_my_c = eff_my / l1_norm
        w_opp_c = eff_opp / l1_norm
        bias_c = torch.clamp(eff_bias / l1_norm.squeeze(-1), -0.95, 0.95)

        w_my = w_my_c[:, self.d4_perms].permute(2, 0, 1).contiguous()
        w_opp = w_opp_c[:, self.d4_perms].permute(2, 0, 1).contiguous()
        bias = bias_c.unsqueeze(1).expand(self.C, 8).contiguous()

        if qat:
            Q1 = 128.0
            w_my = w_my + (torch.round(w_my * Q1) / Q1 - w_my).detach()
            w_opp = w_opp + (torch.round(w_opp * Q1) / Q1 - w_opp).detach()
            bias = bias + (torch.round(bias * Q1) / Q1 - bias).detach()

        return w_my, w_opp, bias

    def forward_features(
        self,
        child_x: torch.Tensor,
        pf_q: torch.Tensor,
        tf_q: torch.Tensor,
        qat: bool = False,
    ) -> torch.Tensor:
        Q1 = 128.0
        w_my, w_opp, bias = self.get_expanded_transformer(qat=qat)
        orig_shape = child_x.shape[:-1]
        B = int(np.prod(orig_shape))
        flat = child_x.reshape(B, 128)
        m_flat = flat[:, 0:64]
        o_flat = flat[:, 64:128]

        acc = (
            torch.matmul(m_flat, w_my.reshape(64, self.H))
            + torch.matmul(o_flat, w_opp.reshape(64, self.H))
            + bias.reshape(1, self.H)
        ).reshape(B, self.C, 8)

        vp = torch.clamp(acc, 0.0, 1.0)
        vn = torch.clamp(-acc, 0.0, 1.0)
        r = vp - vn
        a = vp + vn

        def q_mul(x, y):
            prod = x * y
            if qat:
                prod = prod + (torch.floor(prod * Q1) / Q1 - prod).detach()
            return prod

        r_i = r[:, self.pair_i, :]
        r_j = r[:, self.pair_j, :]
        a_i = a[:, self.pair_i, :]
        a_j = a[:, self.pair_j, :]

        s_vp = vp.sum(dim=-1)
        s_vn = vn.sum(dim=-1)
        s_r = s_vp - s_vn
        s_a = s_vp + s_vn

        s_rr = q_mul(r_i, r_j).sum(dim=-1)
        s_aa = q_mul(a_i, a_j).sum(dim=-1)
        s_ra = q_mul(r_i, a_j).sum(dim=-1)

        g_rr = (s_r[:, self.pair_i] * s_r[:, self.pair_j]) * (1.0 / 8.0)
        g_aa = (s_a[:, self.pair_i] * s_a[:, self.pair_j]) * (1.0 / 8.0)
        if qat:
            g_rr = g_rr + (torch.floor(g_rr * Q1) / Q1 - g_rr).detach()
            g_aa = g_aa + (torch.floor(g_aa * Q1) / Q1 - g_aa).detach()

        pf_i = pf_q.reshape(B, 16)
        tf_i = tf_q.reshape(B, 8)
        pf_f = pf_i.float() / Q1
        tf_f = tf_i.float() / Q1

        ts_i = torch.stack(
            [
                (tf_i[:, 0] * tf_i[:, 0]) >> 7,
                (tf_i[:, 0] * tf_i[:, 4]) >> 7,
                (tf_i[:, 0] * pf_i[:, 1]) >> 7,
                (tf_i[:, 4] * pf_i[:, 1]) >> 7,
                (pf_i[:, 9] * pf_i[:, 1]) >> 7,
                (pf_i[:, 10] * pf_i[:, 1]) >> 7,
                (tf_i[:, 5] * tf_i[:, 5]) >> 7,
                (pf_i[:, 13] * pf_i[:, 13]) >> 7,
            ],
            dim=-1,
        )
        ts_f = ts_i.float() / Q1

        keys_f = torch.stack(
            [
                tf_f[:, 0],
                tf_f[:, 4],
                pf_f[:, 1],
                pf_f[:, 9] - pf_f[:, 10],
            ],
            dim=-1,
        )  # (B, 4)

        cross_r = (s_r.unsqueeze(-1) * keys_f.unsqueeze(1)).reshape(B, self.C * 4) * 0.25
        cross_a = (s_a.unsqueeze(-1) * keys_f.unsqueeze(1)).reshape(B, self.C * 4) * 0.25
        if qat:
            cross_r = cross_r + (torch.floor(cross_r * Q1) / Q1 - cross_r).detach()
            cross_a = cross_a + (torch.floor(cross_a * Q1) / Q1 - cross_a).detach()

        feats = torch.cat(
            [s_vp, s_vn, s_rr, s_aa, s_ra, g_rr, g_aa, pf_f, tf_f, ts_f, cross_r, cross_a],
            dim=-1,
        )
        return feats.reshape(*orig_shape, self.feat_dim)

    def stage_bucket_index(self, empties: torch.Tensor) -> torch.Tensor:
        b = torch.zeros_like(empties, dtype=torch.long)
        b = torch.where(empties >= 11, torch.ones_like(b) * 1, b)
        b = torch.where(empties >= 14, torch.ones_like(b) * 2, b)
        b = torch.where(empties >= 18, torch.ones_like(b) * 3, b)
        b = torch.where(empties >= 22, torch.ones_like(b) * 4, b)
        return b

    def forward(
        self, data: Dict[str, torch.Tensor], qat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(
            data["child_x"], data["pf_q"], data["tf_q"], qat=qat
        )
        buckets = self.stage_bucket_index(data["empties"])

        ow, ob = self.order_weight, self.order_bias
        vw, vb = self.value_weight, self.value_bias
        if qat:
            Q2 = 64.0
            ow = ow + (torch.round(ow * Q2) / Q2 - ow).detach()
            ob = ob + (torch.round(ob * Q2) / Q2 - ob).detach()
            vw = vw + (torch.round(vw * Q2) / Q2 - vw).detach()
            vb = vb + (torch.round(vb * Q2) / Q2 - vb).detach()

        w_o = ow[buckets].unsqueeze(1)
        b_o = ob[buckets].unsqueeze(1)
        ord_logits = (feats * w_o).sum(dim=-1) + b_o

        w_v = vw[buckets].unsqueeze(1)
        b_v = vb[buckets].unsqueeze(1)
        val_preds = (feats * w_v).sum(dim=-1) + b_v
        return ord_logits, val_preds


def eval_dual_heads(
    ord_discs: torch.Tensor,
    val_discs: torch.Tensor,
    data: Dict[str, torch.Tensor],
    label: str,
) -> Tuple[float, float, float]:
    mask = data["mask"]
    best_idx = data["best_idx"]
    fast_sc = data["fast_sc"]

    best_val = val_discs.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    mae = (best_val - data["exact_score"]).abs().mean().item()

    total_sc = fast_sc + ord_discs * 128.0
    masked = total_sc.masked_fill(~mask, -1e9)
    sorted_idx = torch.argsort(masked, dim=-1, descending=True)
    ranks = (sorted_idx == best_idx.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1

    top1 = (ranks == 1).float().mean().item() * 100.0
    top2 = (ranks <= 2).float().mean().item() * 100.0
    mrr = (1.0 / ranks.float()).mean().item() * 100.0

    pure_masked = ord_discs.masked_fill(~mask, -1e9)
    pure_top1 = (
        torch.argmax(pure_masked, dim=-1) == best_idx
    ).float().mean().item() * 100.0

    print(
        f"  {label:<44} | Pure Top1={pure_top1:5.2f}% | +FastScore Top1={top1:5.2f}% Top2={top2:5.2f}% MRR={mrr:5.2f} | Score MAE={mae:5.2f}d"
    )
    return pure_top1, top1, mae


def simulate_int16_exact_c(
    model: D4OthelloNNUE,
    data: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Bit-exact integer simulation of the C/NEON evaluation pipeline (`int16_t` Q1=128
    accumulator + `int16_t` Q2=64 stage-bucketed readout weights).
    """
    Q1 = 128
    Q2 = 64
    model.eval()
    with torch.no_grad():
        w_my, w_opp, bias = model.get_expanded_transformer(qat=True)
        w_my_q = torch.round(w_my * Q1).to(torch.int32).cpu().numpy()
        w_opp_q = torch.round(w_opp * Q1).to(torch.int32).cpu().numpy()
        bias_q = torch.round(bias * Q1).to(torch.int32).cpu().numpy()
        ow_q = torch.round(model.order_weight * Q2).to(torch.int32).cpu().numpy()
        ob_q = torch.round(model.order_bias * Q2 * Q1).to(torch.int32).cpu().numpy()
        vw_q = torch.round(model.value_weight * Q2).to(torch.int32).cpu().numpy()
        vb_q = torch.round(model.value_bias * Q2 * Q1).to(torch.int32).cpu().numpy()

    C = model.C
    H = model.H
    cx_np = data["child_x"].cpu().numpy()
    pf_np = data["pf_q"].cpu().numpy().astype(np.int32)
    tf_np = data["tf_q"].cpu().numpy().astype(np.int32)
    tact_raw_np = data["tact_raw"].cpu().numpy().astype(np.int32)
    empties_np = data["empties"].cpu().numpy()
    B, M, _ = cx_np.shape
    m_np = cx_np[:, :, 0:64].reshape(B * M, 64).astype(np.int32)
    o_np = cx_np[:, :, 64:128].reshape(B * M, 64).astype(np.int32)

    acc = (
        m_np @ w_my_q.reshape(64, H)
        + o_np @ w_opp_q.reshape(64, H)
        + bias_q.reshape(1, H)
    ).reshape(B * M, C, 8)
    assert np.all(acc >= -32768) and np.all(acc <= 32767), "int16 overflow!"

    vp = np.clip(acc, 0, Q1)
    vn = np.clip(-acc, 0, Q1)
    r = vp - vn
    a = vp + vn

    i1, i2 = upper_tri_indices(C)
    r_i = r[:, i1, :]
    r_j = r[:, i2, :]
    a_i = a[:, i1, :]
    a_j = a[:, i2, :]

    s_vp = vp.sum(axis=-1)
    s_vn = vn.sum(axis=-1)
    s_r = s_vp - s_vn
    s_a = s_vp + s_vn

    s_rr = ((r_i * r_j) >> 7).sum(axis=-1)
    s_aa = ((a_i * a_j) >> 7).sum(axis=-1)
    s_ra = ((r_i * a_j) >> 7).sum(axis=-1)

    g_rr = (s_r[:, i1] * s_r[:, i2]) >> 10
    g_aa = (s_a[:, i1] * s_a[:, i2]) >> 10

    pf_flat = pf_np.reshape(B * M, 16)
    tf_flat = tf_np.reshape(B * M, 8)

    ts_flat = np.stack(
        [
            (tf_flat[:, 0] * tf_flat[:, 0]) >> 7,
            (tf_flat[:, 0] * tf_flat[:, 4]) >> 7,
            (tf_flat[:, 0] * pf_flat[:, 1]) >> 7,
            (tf_flat[:, 4] * pf_flat[:, 1]) >> 7,
            (pf_flat[:, 9] * pf_flat[:, 1]) >> 7,
            (pf_flat[:, 10] * pf_flat[:, 1]) >> 7,
            (tf_flat[:, 5] * tf_flat[:, 5]) >> 7,
            (pf_flat[:, 13] * pf_flat[:, 13]) >> 7,
        ],
        axis=-1,
    )

    keys_flat = np.stack(
        [
            tf_flat[:, 0],
            tf_flat[:, 4],
            pf_flat[:, 1],
            pf_flat[:, 9] - pf_flat[:, 10],
        ],
        axis=-1,
    )  # (B*M, 4)

    cross_r = (s_r[:, :, None] * keys_flat[:, None, :]) >> 9
    cross_a = (s_a[:, :, None] * keys_flat[:, None, :]) >> 9
    cross_r = cross_r.reshape(B * M, C * 4)
    cross_a = cross_a.reshape(B * M, C * 4)

    feats_q = np.concatenate(
        [s_vp, s_vn, s_rr, s_aa, s_ra, g_rr, g_aa, pf_flat, tf_flat, ts_flat, cross_r, cross_a],
        axis=-1,
    ).reshape(B, M, model.feat_dim)
    assert np.all(feats_q >= -32768) and np.all(feats_q <= 32767), "feats_q int16 overflow!"

    b_idx = np.zeros(B, dtype=np.int64)
    b_idx = np.where(empties_np >= 11, 1, b_idx)
    b_idx = np.where(empties_np >= 14, 2, b_idx)
    b_idx = np.where(empties_np >= 18, 3, b_idx)
    b_idx = np.where(empties_np >= 22, 4, b_idx)

    raw_ord = (feats_q * ow_q[b_idx][:, None, :]).sum(axis=-1) + ob_q[b_idx][:, None]
    raw_val = (feats_q * vw_q[b_idx][:, None, :]).sum(axis=-1) + vb_q[b_idx][:, None]

    ref_cases = []
    for t_idx in range(min(4, B)):
        m_bits_arr = m_np.reshape(B, M, 64)[t_idx, 0]
        o_bits_arr = o_np.reshape(B, M, 64)[t_idx, 0]
        my_u64 = sum(int(m_bits_arr[sq]) << sq for sq in range(64))
        opp_u64 = sum(int(o_bits_arr[sq]) << sq for sq in range(64))
        tr = tact_raw_np[t_idx, 0]
        ref_cases.append(
            (
                my_u64,
                opp_u64,
                int(empties_np[t_idx]),
                int(tr[0]),
                int(tr[1]),
                int(tr[2]),
                int(tr[3]),
                int(tr[4]),
                int(tr[5]),
                int(raw_ord[t_idx, 0]),
                int(raw_val[t_idx, 0]),
            )
        )

    ord_q = raw_ord.astype(np.float32) / float(Q1 * Q2)
    val_q = raw_val.astype(np.float32) / float(Q1 * Q2)
    dev = data["child_x"].device
    return (
        torch.tensor(ord_q, dtype=torch.float32, device=dev),
        torch.tensor(val_q, dtype=torch.float32, device=dev),
        (w_my_q, w_opp_q, bias_q, ow_q, ob_q, vw_q, vb_q, ref_cases),
    )


def export_nnue_header(path: str, quant_params: tuple, num_channels: int):
    w_my_q, w_opp_q, bias_q, ow_q, ob_q, vw_q, vb_q, ref_cases = quant_params
    w_flip_q = w_my_q - w_opp_q
    H = num_channels * 8
    F_dim = ow_q.shape[1]
    num_buckets = ow_q.shape[0]

    lines = [
        "/* Auto-generated by scripts/train_nnue.py (Othello NNUE Training & Export E-Unified) - DO NOT EDIT */",
        "#ifndef NNUE_DATA_H",
        "#define NNUE_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NNUE_CHANNELS {num_channels}",
        f"#define NNUE_ACC_DIM {H}",
        f"#define NNUE_FEAT_DIM {F_dim}",
        f"#define NNUE_STAGE_BUCKETS {num_buckets}",
        "",
        "/* Initial accumulator bias (NNUE_ACC_DIM int16_t lanes, 16-byte aligned) */",
        f"static const int16_t nnue_bias[NNUE_ACC_DIM] __attribute__((aligned(16))) = {{",
    ]
    b_flat = bias_q.reshape(-1)
    lines.append("  " + ", ".join(str(int(x)) for x in b_flat))
    lines.append("};\n")

    def fmt_table_2d(name, arr2d, comment):
        out = [
            f"/* {comment} */",
            f"static const int16_t {name}[64][NNUE_ACC_DIM] __attribute__((aligned(16))) = {{",
        ]
        for sq in range(64):
            row_str = ", ".join(str(int(x)) for x in arr2d[sq].reshape(-1))
            out.append(f"  {{ {row_str} }},")
        out.append("};\n")
        return "\n".join(out)

    lines.append(
        fmt_table_2d(
            "nnue_w_my",
            w_my_q,
            "Square weight for child_my_bits (parent mover discs after move, 192-bit empty folded)",
        )
    )
    lines.append(
        fmt_table_2d(
            "nnue_w_opp",
            w_opp_q,
            "Square weight for new_opp_bits (parent opponent discs after move, 192-bit empty folded)",
        )
    )
    lines.append(
        fmt_table_2d(
            "nnue_w_flip",
            w_flip_q,
            "Incremental delta w_my[f] - w_opp[f] when square f is flipped from p_opp to child_my_bits",
        )
    )

    lines.append(
        "/* Stage-bucketed move-ordering readout weights [NNUE_STAGE_BUCKETS][NNUE_FEAT_DIM] */"
    )
    lines.append(
        "static const int16_t nnue_order_weight[NNUE_STAGE_BUCKETS][NNUE_FEAT_DIM] __attribute__((aligned(16))) = {"
    )
    for b in range(num_buckets):
        row_str = ", ".join(str(int(x)) for x in ow_q[b].reshape(-1))
        lines.append(f"  {{ {row_str} }},")
    lines.append("};\n")

    lines.append("/* Stage-bucketed move-ordering bias (scaled by Q1*Q2 = 8192) */")
    lines.append(
        "static const int32_t nnue_order_bias[NNUE_STAGE_BUCKETS] = {"
        + ", ".join(str(int(x)) for x in ob_q.reshape(-1))
        + "};\n"
    )

    lines.append(
        "/* Stage-bucketed exact-value readout weights [NNUE_STAGE_BUCKETS][NNUE_FEAT_DIM] */"
    )
    lines.append(
        "static const int16_t nnue_value_weight[NNUE_STAGE_BUCKETS][NNUE_FEAT_DIM] __attribute__((aligned(16))) = {"
    )
    for b in range(num_buckets):
        row_str = ", ".join(str(int(x)) for x in vw_q[b].reshape(-1))
        lines.append(f"  {{ {row_str} }},")
    lines.append("};\n")

    lines.append("/* Stage-bucketed exact-value bias (scaled by Q1*Q2 = 8192) */")
    lines.append(
        "static const int32_t nnue_value_bias[NNUE_STAGE_BUCKETS] = {"
        + ", ".join(str(int(x)) for x in vb_q.reshape(-1))
        + "};\n"
    )

    lines.append(
        "/* Bit-exact reference test cases exported from Python QAT simulator */"
    )
    lines.append("typedef struct {")
    lines.append("  uint64_t n_my;")
    lines.append("  uint64_t n_opp;")
    lines.append("  int empties;")
    lines.append("  int opp_mob;")
    lines.append("  int opp_corner_moves;")
    lines.append("  int is_corner;")
    lines.append("  int is_open_x_or_c;")
    lines.append("  int quad_parity;")
    lines.append("  int flipped;")
    lines.append("  int32_t expected_raw_order;")
    lines.append("  int32_t expected_raw_value;")
    lines.append("} NnueRefCase;\n")
    lines.append(f"#define NNUE_REF_CASE_COUNT {len(ref_cases)}")
    lines.append("static const NnueRefCase nnue_ref_cases[NNUE_REF_CASE_COUNT] = {")
    for my_u64, opp_u64, emp, om, oc, ic, oxc, qp, fl, r_ord, r_val in ref_cases:
        lines.append(
            f"  {{ 0x{my_u64:016x}ULL, 0x{opp_u64:016x}ULL, {emp}, {om}, {oc}, {ic}, {oxc}, {qp}, {fl}, {r_ord}, {r_val} }},"
        )
    lines.append("};\n")
    lines.append("#endif /* NNUE_DATA_H */\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(
        f"Exported quantized NNUE tables to {path} ({os.path.getsize(path)/1024:.1f} KB)"
    )


def train_nnue_variant(
    num_channels: int,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 50,
):
    model = D4OthelloNNUE(num_channels=num_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    n_train = train_data["child_x"].shape[0]
    batch_size = 256
    t0 = time.time()
    best_val_top1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        use_qat = ep >= (epochs - 15)
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            sub = {k: v[idx] for k, v in train_data.items()}

            opt.zero_grad()
            ord_discs, val_discs = model(sub, qat=use_qat)
            logits = (sub["fast_sc"] + ord_discs * 128.0).masked_fill(~sub["mask"], -1e9) / 256.0
            ce_loss = F.cross_entropy(logits, sub["best_idx"])

            best_val_pred = val_discs.gather(1, sub["best_idx"].unsqueeze(1)).squeeze(1)
            val_loss = F.smooth_l1_loss(best_val_pred, sub["exact_score"], beta=3.0)

            loss = ce_loss + 0.50 * val_loss
            loss.backward()
            opt.step()

        sched.step()
        if use_qat and (ep % 3 == 0 or ep == epochs):
            model.eval()
            with torch.no_grad():
                v_ord, _ = model(val_data, qat=True)
                masked = (val_data["fast_sc"] + v_ord * 128.0).masked_fill(
                    ~val_data["mask"], -1e9
                )
                top1 = (
                    torch.argmax(masked, dim=-1) == val_data["best_idx"]
                ).float().mean().item() * 100.0
                if top1 > best_val_top1:
                    best_val_top1 = top1
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    elapsed = time.time() - t0
    model.eval()
    print(
        f"\n--- E-Unified NNUE-{8*num_channels} ({num_channels} D4 channels = {8*num_channels} int16 lanes, FeatDim={model.feat_dim}, trained in {elapsed:.1f}s) ---"
    )
    with torch.no_grad():
        val_ord_fp, val_val_fp = model(val_data, qat=True)
        test_ord_fp, test_val_fp = model(test_data, qat=True)
        eval_dual_heads(
            val_ord_fp, val_val_fp, val_data, f"[VAL]  E-Unified NNUE-{8*num_channels} (PyTorch QAT)"
        )
        eval_dual_heads(
            test_ord_fp,
            test_val_fp,
            test_data,
            f"[TEST] E-Unified NNUE-{8*num_channels} (PyTorch QAT)",
        )

    val_ord_q, val_val_q, quant_params = simulate_int16_exact_c(model, val_data)
    test_ord_q, test_val_q, _ = simulate_int16_exact_c(model, test_data)
    max_diff = (
        ((test_val_q - test_val_fp).abs() * test_data["mask"].float()).max().item()
    )
    print(
        f"  Max int16 C/NEON Quantization Discrepancy vs PyTorch QAT: {max_diff:.4f} discs"
    )
    eval_dual_heads(
        val_ord_q,
        val_val_q,
        val_data,
        f"[VAL]  E-Unified NNUE-{8*num_channels} (Exact int16 C/NEON)",
    )
    eval_dual_heads(
        test_ord_q,
        test_val_q,
        test_data,
        f"[TEST] E-Unified NNUE-{8*num_channels} (Exact int16 C/NEON)",
    )
    return model, quant_params


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_samples = torch.load("data/clean_exact_dataset.pt", weights_only=False)
    train_s = [s for s in raw_samples if s.get("split") == "train"]
    val_s = [s for s in raw_samples if s.get("split") == "val"]
    test_s = [s for s in raw_samples if s.get("split") == "test"]
    print(
        f"Loaded {len(raw_samples)} exact samples: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}"
    )

    train_data = build_e_unified_tensors(train_s, device)
    val_data = build_e_unified_tensors(val_s, device)
    test_data = build_e_unified_tensors(test_s, device)

    print("\n" + "=" * 120)
    print("BASELINES ON HELD-OUT TEST SPLIT")
    print("=" * 120)
    eval_ranking_and_mae(
        torch.zeros_like(val_data["fast_sc"]),
        val_data,
        "[VAL]  0. FastScore Only (No Patterns)",
    )
    eval_ranking_and_mae(
        torch.zeros_like(test_data["fast_sc"]),
        test_data,
        "[TEST] 0. FastScore Only (No Patterns)",
    )

    print("\n" + "=" * 120)
    print("TRAINING E-UNIFIED OTHELLO NNUE (192-BIT EMPTY FOLDED + CORNER PARITY + BITBOARD TACTICAL)")
    print("=" * 120)

    # 1. E-Unified NNUE-32 (4 D4 channels = 32 int16 lanes = 4 NEON registers)
    train_nnue_variant(4, train_data, val_data, test_data, device, epochs=50)

    # 2. E-Unified NNUE-64 (8 D4 channels = 64 int16 lanes = 8 NEON registers)
    _, q64 = train_nnue_variant(8, train_data, val_data, test_data, device, epochs=50)

    # 3. E-Unified NNUE-128 (16 D4 channels = 128 int16 lanes = 16 NEON registers)
    train_nnue_variant(16, train_data, val_data, test_data, device, epochs=50)

    export_nnue_header("src/nnue_data.h", q64, num_channels=8)


if __name__ == "__main__":
    main()
