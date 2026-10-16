/*
   File:           tlstate.h

   Contents:       The search state each thread keeps to itself.

   Mach-O resolves the address of a _Thread_local through a call to
   _tlv_get_addr, and the compiler cannot keep the result across a
   call, so a function touching ten thread-local variables pays ten
   calls -- which the profile put at the top of midgame search time,
   ahead of any of the engine's own code.  The tls-model settings that
   would turn those into a fixed offset (initial-exec, local-exec) are
   accepted and then ignored on this platform; they only bite on ELF.

   So the hot state lives in one object instead: one call resolves it
   and every member is a constant offset away.  The names are kept as
   macros so that the rest of the engine reads exactly as before.

   The functions that take a board to work on call their parameter
   IN_BOARD, because BOARD is one of the names below.
*/



#ifndef TLSTATE_H
#define TLSTATE_H



#include "bitboard.h"
#include "constant.h"
#include "counter.h"



/* The basic board type. One index for each position;
   a1=11, h1=18, a8=81, h8=88. */
typedef int Board[128];


typedef struct {
  /* Counted once per node, placed at offset 0 so every solver accesses
     hot per-node state within a single L1 cache line using immediate offsets. */
  CounterType ts_nodes;
  CounterType ts_evaluations;

  /* The 64-bit hash key of the current position. */
  unsigned int ts_hash1, ts_hash2;

  /* The endgame's and midgame's hot scalar state. */
  unsigned int ts_region_parity;
  int ts_end_best_move, ts_end_best_root_move;
  int ts_best_mid_move, ts_best_mid_root_move;
  int ts_allow_midgame_hash_probe;
  int ts_allow_midgame_hash_update;
  int ts_midgame_full_width_depth;
  int ts_counter_phase;
  int ts_disks_played;
  int ts_split_nesting;
  BitBoard ts_empty_order_mask;
  BitBoard ts_bb_flips;
  double ts_last_panic_check;

  /* The same position as bitboards, indexed by colour. */
  BitBoard ts_board_bits[3];

  /* The evaluation pattern indices, in black perspective. */
  unsigned short ts_eval_pattern_index[48] __attribute__((aligned(16)));

  /* Hot endgame & midgame move count per stage (offset < 512B). */
  int ts_move_count[MAX_SEARCH_DEPTH];
  int ts_pv_depth[MAX_SEARCH_DEPTH];
  int ts_piece_count[3][MAX_SEARCH_DEPTH];
  unsigned int ts_hash_stored1[MAX_SEARCH_DEPTH];
  unsigned int ts_hash_stored2[MAX_SEARCH_DEPTH];
  BitBoard ts_flip_mask[65];

  /* Cached flip counts and bitboards per endgame search level. */
  unsigned char ts_cached_flipped[MAX_SEARCH_DEPTH + 1][100];
  BitBoard ts_cached_flips[MAX_SEARCH_DEPTH + 1][100];

  /* Saved pattern indices on the way down for zero-cost unmake in midgame. */
  unsigned short ts_eval_pattern_stored[MAX_SEARCH_DEPTH][48] __attribute__((aligned(16)));

  /* Per-thread move ordering, evaluation, and PV state. */
  int ts_move_list[MAX_SEARCH_DEPTH][64];
  int ts_sorted_move_order[64][64];
  Board ts_evals[61];
  int ts_feas_index_list[64][64];
  int ts_history_score[2][100];
  int ts_pv[MAX_SEARCH_DEPTH][MAX_SEARCH_DEPTH];

  /* The position, one square per entry. */
  Board ts_board;
} ThreadState;

extern _Thread_local ThreadState tls;


#define board               (tls.ts_board)
#define board_bits          (tls.ts_board_bits)
#define flip_mask           (tls.ts_flip_mask)
#define piece_count         (tls.ts_piece_count)
#define hash_stored1        (tls.ts_hash_stored1)
#define hash_stored2        (tls.ts_hash_stored2)
#define eval_pattern_index  (tls.ts_eval_pattern_index)
#define hash1               (tls.ts_hash1)
#define hash2               (tls.ts_hash2)
#define disks_played        (tls.ts_disks_played)
#define bb_flips            (tls.ts_bb_flips)
#define empty_order_mask    (tls.ts_empty_order_mask)
#define region_parity       (tls.ts_region_parity)
#define end_best_move       (tls.ts_end_best_move)
#define end_best_root_move  (tls.ts_end_best_root_move)
#define nodes               (tls.ts_nodes)
#define evaluations         (tls.ts_evaluations)
#define last_panic_check    (tls.ts_last_panic_check)
#define sorted_move_order   (tls.ts_sorted_move_order)
#define history_score       (tls.ts_history_score)



#endif  /* TLSTATE_H */
