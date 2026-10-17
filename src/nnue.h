/*
   File:          nnue.h
   Description:   Fast-Readout Pattern-Free Othello NNUE with ARM64 NEON Incremental Accumulator,
                  Accumulator-Embedded Corner/Quadrant Parity, and Bitboard Tactical Features (NNUE-002).
*/

#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>

#define NNUE_MAX_ACC_DIM 64

typedef struct {
  int16_t v[NNUE_MAX_ACC_DIM] __attribute__((aligned(16)));
} NnueAcc;

typedef struct {
  int opp_mob;
  int opp_corner_moves;
  int is_corner;
  int is_open_x_or_c;
  int quad_parity;
  int flipped;
} NnueTactical;

/* Initialize accumulator directly from child bitboards (n_my, n_opp) */
void nnue_init_acc_scalar(NnueAcc *acc, uint64_t n_my, uint64_t n_opp);
void nnue_init_acc(NnueAcc *acc, uint64_t n_my, uint64_t n_opp);

/*
   Initialize parent base accumulator once per node from parent bitboards (p_my, p_opp),
   so each candidate move (sq, flips) only adds 1 placed square + flipped squares.
*/
void nnue_init_parent_base_acc(NnueAcc *base_acc, uint64_t p_my, uint64_t p_opp);

/*
   Incrementally compute child accumulator from parent base_acc for a move at `sq`
   flipping `flips`:
     child_acc = base_acc + nnue_w_my[sq] + sum_{f in flips} nnue_w_flip[f]
   Note: lanes 56..63 hold accumulator-embedded corner/quadrant empty counts & open-corner
   flags where `nnue_w_flip[f][56..63] == 0` for all 64 squares.
*/
void nnue_update_child_acc(NnueAcc *child_acc, const NnueAcc *base_acc, int sq, uint64_t flips);

/*
   Evaluate both readout heads (raw integer sums scaled by Q1*Q2 = 8192 = 64 * 128).
   Dividing by 64 (`>> 6`) converts directly to 128 units/disc (`fast_score` / `MIDGAME_SCORE` scale).
*/
void nnue_evaluate_acc_scalar(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties,
  int32_t *out_raw_order,
  int32_t *out_raw_value
);

void nnue_evaluate_acc(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties,
  int32_t *out_raw_order,
  int32_t *out_raw_value
);

/* Fast single-head evaluations returning 128 units = 1 disc */
int nnue_evaluate_order_acc(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties
);

int nnue_evaluate_value_acc(
  const NnueAcc *acc,
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties
);

/*
   Fused full-board NEON accumulator init + value-head readout in registers
   without storing/reloading NnueAcc to stack memory (~35 ns/call on Apple Silicon).
*/
int nnue_evaluate_value_board(
  uint64_t n_my,
  uint64_t n_opp,
  const NnueTactical *tact,
  int empties
);

/* Bit-exact verification against Python QAT reference cases */
int nnue_verify_reference_cases(void);

#endif /* NNUE_H */
