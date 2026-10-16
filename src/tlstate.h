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


/* A doubly linked entry in the list of empty squares the endgame
   walks; defined here because the list itself lives below. */

typedef struct {
  int pred;
  int succ;
} MoveLink;


typedef struct {
  /* The position, one square per entry. */
  Board ts_board;

  /* The same position as bitboards, indexed by colour. */
  BitBoard ts_board_bits[3];

  /* The discs turned by the move made at each stage. */
  BitBoard ts_flip_mask[65];

  int ts_piece_count[3][MAX_SEARCH_DEPTH];

  /* The hash codes at each node on the way down. */
  unsigned int ts_hash_stored1[MAX_SEARCH_DEPTH];
  unsigned int ts_hash_stored2[MAX_SEARCH_DEPTH];

  /* The evaluation pattern indices, in black perspective. */
  unsigned short ts_eval_pattern_index[48];

  /* The 64-bit hash key of the current position. */
  unsigned int ts_hash1, ts_hash2;

  /* The number of discs played from the initial position.  Must match
     the current state of the BOARD variable. */
  int ts_disks_played;

  /* The endgame's own state.  Its solvers touch five or six of these
     per node, which is why they are here rather than standing alone.

     BB_FLIPS is where the flip test leaves the mover's discs with the
     turned ones added; END_MOVE_LIST is the list of empty squares;
     REGION_PARITY has a bit per quadrant with an odd number of them.
     END_BEST_MOVE is the endgame's own, spelled apart from the
     midgame local of the same name. */
  BitBoard ts_bb_flips;
  MoveLink ts_end_move_list[100];
  unsigned int ts_region_parity;
  int ts_end_best_move, ts_end_best_root_move;

  /* Cached flip bitboards and endgame interior state per search level. */
  BitBoard ts_cached_flips[MAX_SEARCH_DEPTH + 1][100];
  int ts_cached_flipped[MAX_SEARCH_DEPTH + 1][100];
  int ts_split_nesting;
  double ts_last_panic_check;

  /* Counted once per node, so it is on the hottest path there is. */
  CounterType ts_nodes;
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
#define end_move_list       (tls.ts_end_move_list)
#define region_parity       (tls.ts_region_parity)
#define end_best_move       (tls.ts_end_best_move)
#define end_best_root_move  (tls.ts_end_best_root_move)
#define nodes               (tls.ts_nodes)
#define last_panic_check    (tls.ts_last_panic_check)



#endif  /* TLSTATE_H */
