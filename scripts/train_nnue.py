#!/usr/bin/env python3
"""
Othello NNUE Training & Export: Pattern-Free Othello NNUE Training, Quantization & C Header Export
============================================================================
Trains the Fast-Readout E-Unified Pattern-Free Othello NNUE on the audited 100%
engine-solved dataset (`data/clean_exact_dataset.pt`, train=18,820, val=2,303, test=2,313):
  - 192-bit raw board input `(my_bits[0..63], opp_bits[0..63], empty_bits[0..63])`
    where `empty_bits` is folded into `(nnue_w_my, nnue_w_opp, nnue_bias)` at
    export time with zero inference overhead.
  - Incremental `int16_t` accumulator (`64` lanes = `7` D4 spatial channels x `8` symmetries
    in lanes `0..55` + `1` accumulator-embedded Corner 3x3 / Quadrant Empty Count / Open-Corner
    channel in lanes `56..63` where `nnue_w_flip[sq][56..63] == 0` for all 64 squares).
  - Vectorized `hsum8_s16` SCReLU / cyclic neighbor readout (`f0..f4` = 40 features) +
    post-`hsum8` register interactions (`f5..f8` = 32 features) + `hsum4_halves_s16`
    corner/quadrant parity (`f9` = 8 features) + bitboard tactical & bilinear interactions
    (`f10..f11` = 16 features) = 96 features (`12` ARM64 NEON `int16x8_t` vectors).
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


def mask_to_64vec(m: int) -> torch.Tensor:
    return torch.tensor([(m >> sq) & 1 for sq in range(64)], dtype=torch.float32)


C33_MAT = torch.stack([mask_to_64vec(m) for m in C33_MASKS], dim=1)    # (64, 4)
QUAD_MAT = torch.stack([mask_to_64vec(m) for m in QUAD_MASKS], dim=1)  # (64, 4)


def build_parity_channel_weights() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the exact integer weights for accumulator lanes 56..63 (`a7`):
      - Lanes 0..3 (`56..59`): `c33_empty_cnt[c] + (c_emp[c] << 5)` for corner c = 0..3
      - Lanes 4..7 (`60..63`): `quad_empty_cnt[c]` for quadrant c = 0..3
    Because `w_my_par == w_opp_par` for every square on the board,
    `w_flip_par = w_my_par - w_opp_par == 0` identically for all 64 squares!
    """
    bias_par = np.array([9 + 32, 9 + 32, 9 + 32, 9 + 32, 16, 16, 16, 16], dtype=np.int32)
    w_par = np.zeros((64, 8), dtype=np.int32)
    for sq in range(64):
        for c in range(4):
            if (C33_MASKS[c] >> sq) & 1:
                w_par[sq, c] -= 1
            if sq == CORNER_BITS[c]:
                w_par[sq, c] -= 32
            if (QUAD_MASKS[c] >> sq) & 1:
                w_par[sq, 4 + c] -= 1
    return w_par, w_par.copy(), bias_par


def build_e_unified_tensors(samples: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
    """Builds padded candidate tensors including exact int16-scaled Corner Parity and Tactical features."""
    n = len(samples)
    max_k = max(len(s["child_my_bits"]) for s in samples)

    child_x = torch.zeros((n, max_k, 128), dtype=torch.float32)
    fast_sc = torch.zeros((n, max_k), dtype=torch.float32)
    hsum_par_q = torch.zeros((n, max_k, 5), dtype=torch.int32)
    f9_par_q = torch.zeros((n, max_k, 8), dtype=torch.int32)
    f10_tact_q = torch.zeros((n, max_k, 8), dtype=torch.int32)
    f11_extra_q = torch.zeros((n, max_k, 8), dtype=torch.int32)
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
        my_corn = torch.round(my_b[:, CORNER_BITS]).to(torch.int32).sum(dim=-1)
        opp_corn = torch.round(opp_b[:, CORNER_BITS]).to(torch.int32).sum(dim=-1)
        disc_diff = torch.round(my_b.sum(dim=-1) - opp_b.sum(dim=-1)).to(torch.int32)

        open_odd = (c_emp & c33_odd).sum(dim=-1)
        open_even = (c_emp & (1 - c33_odd)).sum(dim=-1)
        sum_c33_odd = c33_odd.sum(dim=-1)
        sum_quad_odd = quad_odd.sum(dim=-1)
        sum_c33_eq1 = (c33_cnt == 1).to(torch.int32).sum(dim=-1)
        sum_quad_eq1 = (quad_cnt == 1).to(torch.int32).sum(dim=-1)
        sum_c33_eq2 = (c33_cnt == 2).to(torch.int32).sum(dim=-1)
        sum_quad_eq2 = (quad_cnt == 2).to(torch.int32).sum(dim=-1)
        sum_c_emp = c_emp.sum(dim=-1)

        # Lane 7 of f0..f4 (hsum8_s16 8th vector sums)
        hsum_par_q[i, :k, 0] = open_odd << 6
        hsum_par_q[i, :k, 1] = open_even << 6
        hsum_par_q[i, :k, 2] = (sum_c33_odd + sum_quad_odd) << 5
        hsum_par_q[i, :k, 3] = (sum_c33_eq1 + sum_quad_eq1) << 5
        hsum_par_q[i, :k, 4] = (sum_c33_eq2 + sum_quad_eq2) << 5

        # Vector f9: hsum4_halves_s16(odd_vec, eq1_vec, eq2_vec, cemp_vec) << 5
        f9_par_q[i, :k, 0] = sum_c33_odd << 5
        f9_par_q[i, :k, 1] = sum_quad_odd << 5
        f9_par_q[i, :k, 2] = sum_c33_eq1 << 5
        f9_par_q[i, :k, 3] = sum_quad_eq1 << 5
        f9_par_q[i, :k, 4] = sum_c33_eq2 << 5
        f9_par_q[i, :k, 5] = sum_quad_eq2 << 5
        f9_par_q[i, :k, 6] = sum_c_emp << 5
        f9_par_q[i, :k, 7] = 0

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

        tf0 = torch.clamp(opp_mob, 0, 16) << 3
        tf1 = torch.clamp(opp_corner, 0, 4) << 6
        tf2 = is_corner << 7
        tf3 = is_open_xc << 7
        tf4 = parity << 7
        tf5 = torch.clamp(flips, 0, 16) << 3
        tf6 = is_wipeout << 7
        tf7 = (tf0 * tf0) >> 7

        f10_tact_q[i, :k, 0] = tf0
        f10_tact_q[i, :k, 1] = tf1
        f10_tact_q[i, :k, 2] = tf2
        f10_tact_q[i, :k, 3] = tf3
        f10_tact_q[i, :k, 4] = tf4
        f10_tact_q[i, :k, 5] = tf5
        f10_tact_q[i, :k, 6] = tf6
        f10_tact_q[i, :k, 7] = tf7

        # Put (tf0 * tf4) >> 7 into f9_par_q[:, 7] (where cemp_vec[4..7] was 0)
        f9_par_q[i, :k, 7] = (tf0 * tf4) >> 7

        mask[i, :k] = True
        best_idx[i] = int(s["best_idx"])
        exact_score[i] = float(s.get("exact_score", s.get("target_score", 0.0)))
        empties[i] = int(s.get("empties", 14))

    return {
        "child_x": child_x.to(device),
        "fast_sc": fast_sc.to(device),
        "hsum_par_q": hsum_par_q.to(device),
        "f9_par_q": f9_par_q.to(device),
        "f10_tact_q": f10_tact_q.to(device),
        "tact_raw": tact_raw.to(device),
        "mask": mask.to(device),
        "best_idx": best_idx.to(device),
        "exact_score": exact_score.to(device),
        "empties": empties.to(device),
    }


class FastD4OthelloNNUE(nn.Module):
    """
    Fast-Readout Pattern-Free D4-Equivariant Othello NNUE (NNUE-002):
      - `C = 7` D4 spatial channels (`56` int16 lanes `acc->v[0..55]`)
      - `1` accumulator-embedded parity/open-corner channel (`8` int16 lanes `acc->v[56..63]`, `w_flip == 0`)
      - `11` ARM64 NEON `int16x8_t` feature vectors (`NNUE_FEAT_DIM = 88`), zero popcountll!
    """

    def __init__(self, num_spatial_channels: int = 7, num_stage_buckets: int = 4):
        super().__init__()
        self.C = num_spatial_channels
        self.H_spatial = 8 * num_spatial_channels
        self.H_total = 64
        self.num_stage_buckets = num_stage_buckets
        self.feat_dim = 88

        self.w_my_canon = nn.Parameter(torch.randn(self.C, 64) * 0.15)
        self.w_opp_canon = nn.Parameter(torch.randn(self.C, 64) * 0.15)
        self.w_emp_canon = nn.Parameter(torch.randn(self.C, 64) * 0.15)
        self.bias_canon = nn.Parameter(torch.zeros(self.C))

        self.register_buffer("d4_perms", torch.tensor(D4_MAPS, dtype=torch.long))
        c1 = [(c + 1) % self.C for c in range(self.C)]
        self.register_buffer("c1_idx", torch.tensor(c1, dtype=torch.long))
        rot1 = [(k + 1) % 8 for k in range(8)]
        self.register_buffer("rot1_idx", torch.tensor(rot1, dtype=torch.long))

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
        data: Dict[str, torch.Tensor],
        qat: bool = False,
    ) -> torch.Tensor:
        Q1 = 128.0
        child_x = data["child_x"]
        w_my, w_opp, bias = self.get_expanded_transformer(qat=qat)
        orig_shape = child_x.shape[:-1]
        B = int(np.prod(orig_shape))
        flat = child_x.reshape(B, 128)
        m_flat = flat[:, 0:64]
        o_flat = flat[:, 64:128]

        acc = (
            torch.matmul(m_flat, w_my.reshape(64, self.H_spatial))
            + torch.matmul(o_flat, w_opp.reshape(64, self.H_spatial))
            + bias.reshape(1, self.H_spatial)
        ).reshape(B, self.C, 8)

        vp = torch.clamp(acc, 0.0, 1.0)
        vn = torch.clamp(-acc, 0.0, 1.0)
        r = vp - vn

        def q_mul(x, y):
            prod = x * y
            if qat:
                prod = prod + (torch.floor(prod * Q1) / Q1 - prod).detach()
            return prod

        s_vp = vp.sum(dim=-1)
        s_vn = vn.sum(dim=-1)
        sp = q_mul(vp, vp).sum(dim=-1)
        sn = q_mul(vn, vn).sum(dim=-1)
        cr = q_mul(r, r[:, self.c1_idx, :]).sum(dim=-1)

        hp_f = data["hsum_par_q"].reshape(B, 5).float() / Q1
        f0 = torch.cat([s_vp, hp_f[:, 0:1]], dim=-1)
        f1 = torch.cat([s_vn, hp_f[:, 1:2]], dim=-1)
        f2 = torch.cat([sp, hp_f[:, 2:3]], dim=-1)
        f3 = torch.cat([sn, hp_f[:, 3:4]], dim=-1)
        f4 = torch.cat([cr, hp_f[:, 4:5]], dim=-1)

        sr_raw = f0 - f1
        sa_raw = f0 + f1
        if qat:
            sr_q = torch.round(sr_raw * Q1)
            sa_q = torch.round(sa_raw * Q1)
            sr_s = torch.floor(sr_q / 8.0) / Q1
            sa_s = torch.floor(sa_q / 8.0) / Q1
            sr_s = sr_raw * 0.125 + (sr_s - sr_raw * 0.125).detach()
            sa_s = sa_raw * 0.125 + (sa_s - sa_raw * 0.125).detach()
        else:
            sr_s = sr_raw * 0.125
            sa_s = sa_raw * 0.125

        def q_post(prod, shift_div: float):
            out = prod * (128.0 / shift_div)
            if qat:
                out = out + (torch.floor(out * Q1) / Q1 - out).detach()
            return out

        f5 = q_post(sr_s * sr_s, 16.0)
        f6 = q_post(sa_s * sa_s, 16.0)
        f7 = q_post(sr_s * sr_s[:, self.rot1_idx], 16.0)

        f9 = data["f9_par_q"].reshape(B, 8).float() / Q1
        f10 = data["f10_tact_q"].reshape(B, 8).float() / Q1

        f8 = q_post(sr_s * f10[:, 0:1], 64.0)

        feats = torch.cat([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dim=-1)
        return feats.reshape(*orig_shape, self.feat_dim)

    def stage_bucket_index(self, empties: torch.Tensor) -> torch.Tensor:
        b = torch.zeros_like(empties, dtype=torch.long)
        b = torch.where(empties >= 11, torch.ones_like(b) * 1, b)
        b = torch.where(empties >= 14, torch.ones_like(b) * 2, b)
        b = torch.where(empties >= 17, torch.ones_like(b) * 3, b)
        return b

    def forward(
        self, data: Dict[str, torch.Tensor], qat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(data, qat=qat)
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
    model: FastD4OthelloNNUE,
    data: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Bit-exact integer simulation of the C/NEON evaluation pipeline (`int16_t` Q1=128
    64-lane accumulator with `a7` embedded parity + `int16_t` Q2=64 88-feature readout).
    """
    Q1 = 128
    Q2 = 64
    model.eval()
    with torch.no_grad():
        w_my, w_opp, bias = model.get_expanded_transformer(qat=True)
        w_my_sp = torch.round(w_my * Q1).to(torch.int32).cpu().numpy().reshape(64, 56)
        w_opp_sp = torch.round(w_opp * Q1).to(torch.int32).cpu().numpy().reshape(64, 56)
        bias_sp = torch.round(bias * Q1).to(torch.int32).cpu().numpy().reshape(56)
        ow_q = torch.round(model.order_weight * Q2).to(torch.int32).cpu().numpy()
        ob_q = torch.round(model.order_bias * Q2 * Q1).to(torch.int32).cpu().numpy()
        vw_q = torch.round(model.value_weight * Q2).to(torch.int32).cpu().numpy()
        vb_q = torch.round(model.value_bias * Q2 * Q1).to(torch.int32).cpu().numpy()

    w_my_par, w_opp_par, bias_par = build_parity_channel_weights()
    w_my_q = np.concatenate([w_my_sp, w_my_par], axis=1)    # (64, 64)
    w_opp_q = np.concatenate([w_opp_sp, w_opp_par], axis=1)  # (64, 64)
    bias_q = np.concatenate([bias_sp, bias_par], axis=0)     # (64,)

    cx_np = data["child_x"].cpu().numpy()
    f10_np = data["f10_tact_q"].cpu().numpy().astype(np.int32)
    tact_raw_np = data["tact_raw"].cpu().numpy().astype(np.int32)
    empties_np = data["empties"].cpu().numpy()
    B, M, _ = cx_np.shape
    N = B * M
    m_np = cx_np[:, :, 0:64].reshape(N, 64).astype(np.int32)
    o_np = cx_np[:, :, 64:128].reshape(N, 64).astype(np.int32)

    acc64 = m_np @ w_my_q + o_np @ w_opp_q + bias_q[None, :]
    assert np.all(acc64 >= -32768) and np.all(acc64 <= 32767), "int16 overflow!"

    acc_sp = acc64[:, :56].reshape(N, 7, 8)
    a7 = acc64[:, 56:64]  # (N, 8)

    vp = np.clip(acc_sp, 0, Q1)
    vn = np.clip(-acc_sp, 0, Q1)
    r = vp - vn
    c1 = [(c + 1) % 7 for c in range(7)]

    s_vp = vp.sum(axis=-1)
    s_vn = vn.sum(axis=-1)
    sp = ((vp * vp) >> 7).sum(axis=-1)
    sn = ((vn * vn) >> 7).sum(axis=-1)
    cr = ((r * r[:, c1, :]) >> 7).sum(axis=-1)

    # Extract parity vectors directly from accumulator lane a7!
    cnt_vec = a7 & 31
    odd_vec = a7 & 1
    cemp_vec = a7 >> 5
    open_odd_vec = cemp_vec & odd_vec
    open_even_vec = cemp_vec & (1 - odd_vec)
    eq1_vec = (cnt_vec == 1).astype(np.int32)
    eq2_vec = (cnt_vec == 2).astype(np.int32)

    f0 = np.concatenate([s_vp, (open_odd_vec.sum(axis=-1, keepdims=True) << 6)], axis=-1)
    f1 = np.concatenate([s_vn, (open_even_vec.sum(axis=-1, keepdims=True) << 6)], axis=-1)
    f2 = np.concatenate([sp, (odd_vec.sum(axis=-1, keepdims=True) << 5)], axis=-1)
    f3 = np.concatenate([sn, (eq1_vec.sum(axis=-1, keepdims=True) << 5)], axis=-1)
    f4 = np.concatenate([cr, (eq2_vec.sum(axis=-1, keepdims=True) << 5)], axis=-1)

    sr_s = (f0 - f1) >> 3
    sa_s = (f0 + f1) >> 3
    rot1 = [(k + 1) % 8 for k in range(8)]
    sr_rot1 = sr_s[:, rot1]

    f5 = (sr_s * sr_s) >> 4
    f6 = (sa_s * sa_s) >> 4
    f7 = (sr_s * sr_rot1) >> 4

    f10 = f10_np.reshape(N, 8)
    f8 = (sr_s * f10[:, 0:1]) >> 6

    f9 = np.stack(
        [
            odd_vec[:, :4].sum(axis=-1) << 5,
            odd_vec[:, 4:].sum(axis=-1) << 5,
            eq1_vec[:, :4].sum(axis=-1) << 5,
            eq1_vec[:, 4:].sum(axis=-1) << 5,
            eq2_vec[:, :4].sum(axis=-1) << 5,
            eq2_vec[:, 4:].sum(axis=-1) << 5,
            cemp_vec[:, :4].sum(axis=-1) << 5,
            (f10[:, 0] * f10[:, 4]) >> 7,
        ],
        axis=-1,
    )

    feats_q = np.concatenate(
        [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], axis=-1
    ).reshape(B, M, 88)
    assert np.all(feats_q >= -32768) and np.all(feats_q <= 32767), "feats_q int16 overflow!"

    b_idx = np.zeros(B, dtype=np.int64)
    b_idx = np.where(empties_np >= 11, 1, b_idx)
    b_idx = np.where(empties_np >= 14, 2, b_idx)
    b_idx = np.where(empties_np >= 17, 3, b_idx)

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


def export_nnue_header(path: str, quant_params: tuple, num_spatial_channels: int = 7):
    w_my_q, w_opp_q, bias_q, ow_q, ob_q, vw_q, vb_q, ref_cases = quant_params
    w_flip_q = w_my_q - w_opp_q
    assert np.all(w_flip_q[:, 56:64] == 0), "Expected w_flip == 0 on parity lanes 56..63!"
    H = 64
    F_dim = ow_q.shape[1]
    num_buckets = ow_q.shape[0]

    lines = [
        "/* Auto-generated by scripts/train_nnue.py (Othello NNUE-002 Fast-Readout) - DO NOT EDIT */",
        "#ifndef NNUE_DATA_H",
        "#define NNUE_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NNUE_SPATIAL_CHANNELS {num_spatial_channels}",
        f"#define NNUE_CHANNELS 8",
        f"#define NNUE_ACC_DIM {H}",
        f"#define NNUE_FEAT_DIM {F_dim}",
        f"#define NNUE_STAGE_BUCKETS {num_buckets}",
        "",
        "/* Initial accumulator bias (NNUE_ACC_DIM int16_t lanes, 16-byte aligned) */",
        "static const int16_t nnue_bias[NNUE_ACC_DIM] __attribute__((aligned(16))) = {",
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
            "Square weight for child_my_bits (lanes 0..55: 7 D4 channels, lanes 56..63: corner/quad parity)",
        )
    )
    lines.append(
        fmt_table_2d(
            "nnue_w_opp",
            w_opp_q,
            "Square weight for new_opp_bits (lanes 0..55: 7 D4 channels, lanes 56..63: corner/quad parity)",
        )
    )
    lines.append(
        fmt_table_2d(
            "nnue_w_flip",
            w_flip_q,
            "Incremental delta w_my[f] - w_opp[f] (lanes 56..63 are identically 0 for all 64 squares)",
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


def train_fast_nnue(
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 55,
):
    model = FastD4OthelloNNUE(num_spatial_channels=7).to(device)
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
            with torch.no_grad():
                model.order_weight[:, 80].clamp_(max=0.0)
                model.order_weight[:, 81].clamp_(max=0.0)
                model.order_weight[:, 87].clamp_(max=0.0)

        sched.step()
        if use_qat and (ep % 2 == 0 or ep == epochs):
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

    # Fine-tune value_weight and value_bias on frozen QAT features (leaves order_weight & Top-1 untouched)
    for name, p in model.named_parameters():
        p.requires_grad = "value_" in name
    v_opt = torch.optim.AdamW([model.value_weight, model.value_bias], lr=5e-3, weight_decay=5e-4)
    with torch.no_grad():
        tr_feats = model.forward_features(train_data, qat=True)
        N = tr_feats.shape[0]
        best_tr_feats = tr_feats[torch.arange(N, device=device), train_data["best_idx"]]
        tr_buckets = model.stage_bucket_index(train_data["empties"])
        tr_target = train_data["exact_score"]

    for _ in range(300):
        v_opt.zero_grad()
        Q2 = 64.0
        vw = model.value_weight + (torch.round(model.value_weight * Q2) / Q2 - model.value_weight).detach()
        vb = model.value_bias + (torch.round(model.value_bias * Q2) / Q2 - model.value_bias).detach()
        preds = (best_tr_feats * vw[tr_buckets]).sum(dim=-1) + vb[tr_buckets]
        loss = F.l1_loss(preds, tr_target)
        loss.backward()
        v_opt.step()

    elapsed = time.time() - t0
    model.eval()
    print(
        f"\n--- Fast-Readout NNUE-64 (7 D4 spatial channels + a7 parity channel = 64 int16 lanes, FeatDim={model.feat_dim}, trained in {elapsed:.1f}s) ---"
    )
    with torch.no_grad():
        val_ord_fp, val_val_fp = model(val_data, qat=True)
        test_ord_fp, test_val_fp = model(test_data, qat=True)
        eval_dual_heads(
            val_ord_fp, val_val_fp, val_data, "[VAL]  Fast-Readout NNUE-64 (PyTorch QAT)"
        )
        eval_dual_heads(
            test_ord_fp,
            test_val_fp,
            test_data,
            "[TEST] Fast-Readout NNUE-64 (PyTorch QAT)",
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
        "[VAL]  Fast-Readout NNUE-64 (Exact int16 C/NEON)",
    )
    eval_dual_heads(
        test_ord_q,
        test_val_q,
        test_data,
        "[TEST] Fast-Readout NNUE-64 (Exact int16 C/NEON)",
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
    print("TRAINING FAST-READOUT E-UNIFIED OTHELLO NNUE (NNUE-002)")
    print("=" * 120)

    _, q64 = train_fast_nnue(train_data, val_data, test_data, device, epochs=55)
    export_nnue_header("src/nnue_data.h", q64, num_spatial_channels=7)


if __name__ == "__main__":
    main()
