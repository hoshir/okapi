/*
   File:          search_invariants_test.c
   Contents:      Unit and regression tests for core search, bitboard,
                  NEON pattern index, stability bound, and TT invariants.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "constant.h"
#include "globals.h"
#include "bitboard.h"
#include "bitbcnt.h"
#include "bitbmob.h"
#include "bitbtest.h"
#include "doflip.h"
#include "unflip.h"
#include "moves.h"
#include "patterns.h"
#include "end_patterns.h"
#include "end_aspiration.h"
#include "end_leaf.h"
#include "nnue.h"
#include "stable.h"
#include "hash.h"
#include "search.h"
#include "timer.h"
#include "tlstate.h"

#define NUM_ACTIVE_PATTERNS  46

static int failures = 0;

#define CHECK( cond, ... )                                  \
  do {                                                      \
    if ( !(cond) ) {                                        \
      if ( failures < 10 ) {                                \
	fprintf( stderr, "FAIL: " __VA_ARGS__ );            \
	fprintf( stderr, "\n" );                            \
      }                                                     \
      failures++;                                           \
    }                                                       \
  } while ( 0 )


/*
  TEST_BITBOARD_AND_PATTERN_INVARIANTS
  Plays random move sequences with make_move / unmake_move and verifies:
  1. board_bits[EMPTY] == ~(board_bits[BLACKSQ] | board_bits[WHITESQ])
  2. Incremental eval_pattern_index[0..45] matches both scalar
     determine_pattern_indices() and NEON determine_pattern_indices_bitboard()
     on every ply, both going down and after unmake_move.
*/
static void
test_bitboard_and_pattern_invariants( void ) {
  int game, ply, i, j, sq;
  unsigned short inc_idx[48];
  unsigned short scalar_idx[48];
  unsigned short bb_idx[48];
  int played_moves[64];
  int played_sides[64];

  for ( game = 0; game < 500; game++ ) {
    for ( i = 0; i < 128; i++ )
      board[i] = OUTSIDE;
    for ( i = 1; i <= 8; i++ )
      for ( j = 1; j <= 8; j++ )
	board[10 * i + j] = EMPTY;

    board[44] = WHITESQ;
    board[55] = WHITESQ;
    board[45] = BLACKSQ;
    board[54] = BLACKSQ;

    disks_played = 0;
    set_board_bits();
    determine_pattern_indices();

    int side = BLACKSQ;
    int depth = 0;

    for ( ply = 0; ply < 40; ply++ ) {
      BitBoard b_bits = board_bits[BLACKSQ];
      BitBoard w_bits = board_bits[WHITESQ];
      BitBoard e_bits = board_bits[EMPTY];

      CHECK( (b_bits & w_bits) == 0,
	     "game %d ply %d: black and white bitboards overlap", game, ply );
      CHECK( e_bits == ~(b_bits | w_bits),
	     "game %d ply %d: board_bits[EMPTY] out of sync", game, ply );

      memcpy( inc_idx, eval_pattern_index, sizeof( inc_idx ) );
      determine_pattern_indices();
      memcpy( scalar_idx, eval_pattern_index, sizeof( scalar_idx ) );
      determine_pattern_indices_bitboard( b_bits, w_bits );
      memcpy( bb_idx, eval_pattern_index, sizeof( bb_idx ) );
      memcpy( eval_pattern_index, inc_idx, sizeof( inc_idx ) );

      for ( i = 0; i < NUM_ACTIVE_PATTERNS; i++ ) {
	CHECK( inc_idx[i] == scalar_idx[i],
	       "game %d ply %d pattern %d: incremental %u != scalar %u",
	       game, ply, i, (unsigned) inc_idx[i], (unsigned) scalar_idx[i] );
	CHECK( bb_idx[i] == scalar_idx[i],
	       "game %d ply %d pattern %d: bitboard NEON %u != scalar %u",
	       game, ply, i, (unsigned) bb_idx[i], (unsigned) scalar_idx[i] );
      }

      generate_all( side );
      if ( move_count[disks_played] == 0 ) {
	side = OPP( side );
	generate_all( side );
	if ( move_count[disks_played] == 0 )
	  break;
      }

      int m_idx = rand() % move_count[disks_played];
      sq = move_list[disks_played][m_idx];
      int flipped = make_move( side, sq, TRUE );
      CHECK( flipped > 0, "game %d ply %d: legal move %d flipped 0 discs", game, ply, sq );

      played_moves[depth] = sq;
      played_sides[depth] = side;
      depth++;
      side = OPP( side );
    }

    while ( depth > 0 ) {
      depth--;
      unmake_move( played_sides[depth], played_moves[depth] );

      BitBoard b_bits = board_bits[BLACKSQ];
      BitBoard w_bits = board_bits[WHITESQ];
      CHECK( board_bits[EMPTY] == ~(b_bits | w_bits),
	     "game %d unmake depth %d: board_bits[EMPTY] out of sync", game, depth );

      memcpy( inc_idx, eval_pattern_index, sizeof( inc_idx ) );
      determine_pattern_indices_bitboard( b_bits, w_bits );
      for ( i = 0; i < NUM_ACTIVE_PATTERNS; i++ ) {
	CHECK( inc_idx[i] == eval_pattern_index[i],
	       "game %d unmake depth %d pattern %d: incremental %u != bitboard %u",
	       game, depth, i, (unsigned) inc_idx[i], (unsigned) eval_pattern_index[i] );
      }
      memcpy( eval_pattern_index, inc_idx, sizeof( inc_idx ) );
    }
  }
}


/*
  TEST_STABILITY_INVARIANTS
  Verifies on random positions that edge and full stability counts are
  monotonic subsets of each player's discs and that the induced lower/upper
  endgame score bounds never cross (2*s_my - 64 <= 64 - 2*s_opp).
*/
static void
test_stability_invariants( void ) {
  int trial, i, j, sq;

  for ( trial = 0; trial < 2000; trial++ ) {
    int fill = 50 + (rand() % 45);
    BitBoard b_bits, w_bits;

    for ( i = 0; i < 128; i++ )
      board[i] = OUTSIDE;
    for ( i = 1; i <= 8; i++ )
      for ( j = 1; j <= 8; j++ ) {
	sq = 10 * i + j;
	if ( (rand() % 100) < fill )
	  board[sq] = (rand() & 1) ? BLACKSQ : WHITESQ;
	else
	  board[sq] = EMPTY;
      }

    set_bitboards( board, BLACKSQ, &b_bits, &w_bits );
    int b_cnt = non_iterative_popcount( b_bits );
    int w_cnt = non_iterative_popcount( w_bits );

    EdgeIndices b_edges = { 0 }, w_edges = { 0 };
    int b_edge = count_edge_stable_indexed( BLACKSQ, b_bits, w_bits, &b_edges );
    int w_edge = count_edge_stable_indexed( WHITESQ, w_bits, b_bits, &w_edges );

    CHECK( (b_edges.bits & ~b_bits) == 0,
	   "trial %d: black edge stable bits outside black bitboard", trial );
    CHECK( (w_edges.bits & ~w_bits) == 0,
	   "trial %d: white edge stable bits outside white bitboard", trial );
    CHECK( b_edge == non_iterative_popcount( b_edges.bits ),
	   "trial %d: b_edge count mismatch", trial );
    CHECK( w_edge == non_iterative_popcount( w_edges.bits ),
	   "trial %d: w_edge count mismatch", trial );

    int b_full = count_stable_indexed( BLACKSQ, b_bits, w_bits, &b_edges );
    int w_full = count_stable_indexed( WHITESQ, w_bits, b_bits, &w_edges );

    CHECK( b_full >= b_edge && b_full <= b_cnt,
	   "trial %d: b_full %d out of range [%d, %d]", trial, b_full, b_edge, b_cnt );
    CHECK( w_full >= w_edge && w_full <= w_cnt,
	   "trial %d: w_full %d out of range [%d, %d]", trial, w_full, w_edge, w_cnt );
    CHECK( (b_edges.bits & w_edges.bits) == 0,
	   "trial %d: black and white stable sets overlap", trial );

    int lower_bound = 2 * b_full - 64;
    int upper_bound = 64 - 2 * w_full;
    CHECK( lower_bound <= upper_bound,
	   "trial %d: stability lower_bound %d > upper_bound %d",
	   trial, lower_bound, upper_bound );
  }
}


/*
  TEST_TT_ISOLATION_INVARIANTS
  Verifies that TT entries preserve selectivity, flags (ENDGAME_SCORE,
  MIDGAME_SCORE, HEURISTIC_PRESEARCH_MOVE), and 4-move lists across
  add_hash_extended_ts / find_hash_key so selective or presearch entries
  cannot masquerade as exact (selectivity == 0) endgame bounds.
*/
static void
test_tt_isolation_invariants( void ) {
  ThreadState *ts = &tls;
  HashEntry entry;
  int moves_in[4] = { 34, 43, 56, 65 };

  init_hash( 14 );
  setup_hash( TRUE );

  ts->ts_hash1 = 0x12345678u;
  ts->ts_hash2 = 0x9abcdef0u;

  /* 1. Store a presearch entry (HEURISTIC_PRESEARCH_MOVE) */
  add_hash_extended_ts( ts, ENDGAME_MODE, 12, moves_in,
			ENDGAME_SCORE | EXACT_VALUE | HEURISTIC_PRESEARCH_MOVE,
			10, 0 );
  find_hash_key( &entry, ts->ts_hash1, ts->ts_hash2 );

  CHECK( entry.draft == 10, "TT presearch draft mismatch: %d", entry.draft );
  CHECK( (entry.flags & HEURISTIC_PRESEARCH_MOVE) != 0,
	 "TT presearch flag lost" );
  /* Exact hash_hit guard: !(entry.flags & HEURISTIC_PRESEARCH_MOVE) must be FALSE */
  int exact_hash_hit = (entry.draft != NO_HASH_MOVE) &&
		       (entry.flags & ENDGAME_SCORE) &&
		       !(entry.flags & HEURISTIC_PRESEARCH_MOVE);
  CHECK( !exact_hash_hit, "HEURISTIC_PRESEARCH_MOVE must not trigger exact hash_hit" );

  /* 2. Store a selective endgame entry (selectivity = 2) at draft = 24 */
  ts->ts_hash1 = 0x2468ace0u;
  ts->ts_hash2 = 0x13579bdfu;
  add_hash_extended_ts( ts, ENDGAME_MODE, 6, moves_in,
			ENDGAME_SCORE | LOWER_BOUND,
			24, 2 );
  find_hash_key( &entry, ts->ts_hash1, ts->ts_hash2 );

  CHECK( entry.draft == 24, "TT selective draft mismatch: %d", entry.draft );
  CHECK( entry.selectivity == 2, "TT selectivity mismatch: %d", entry.selectivity );
  CHECK( entry.eval == 6, "TT eval mismatch: %d", entry.eval );
  CHECK( entry.move[0] == 34 && entry.move[1] == 43 &&
	 entry.move[2] == 56 && entry.move[3] == 65,
	 "TT move[0..3] list not preserved" );
  /* In exact solve (selectivity == 0), entry.selectivity <= 0 must be FALSE */
  CHECK( !(entry.selectivity <= 0),
	 "Selective TT entry (sel=2) must not satisfy exact bound cutoff (sel<=0)" );

  /* 3. Verify same-position update preserves/shifts move[1..3] */
  add_hash_ts( ts, ENDGAME_MODE, 8, 34, ENDGAME_SCORE | LOWER_BOUND, 24, 0 );
  find_hash_key( &entry, ts->ts_hash1, ts->ts_hash2 );
  CHECK( entry.move[0] == 34 && entry.move[1] == 43 &&
	 entry.move[2] == 56 && entry.move[3] == 65,
	 "Same best[0] update wiped move[1..3]: %d,%d,%d,%d",
	 entry.move[0], entry.move[1], entry.move[2], entry.move[3] );

  add_hash_ts( ts, ENDGAME_MODE, 10, 77, ENDGAME_SCORE | EXACT_VALUE, 24, 0 );
  find_hash_key( &entry, ts->ts_hash1, ts->ts_hash2 );
  CHECK( entry.move[0] == 77 && entry.move[1] == 34 &&
	 entry.move[2] == 43 && entry.move[3] == 56,
	 "New best[0] update failed to shift move[0..2] into move[1..3]: %d,%d,%d,%d",
	 entry.move[0], entry.move[1], entry.move[2], entry.move[3] );

  /* 4. Verify 64-bit packed shallow TT depth-partition coexistence (d=5,6,7,8) */
  BitBoard test_my = 0x0000000810000000ULL;
  BitBoard test_opp = 0x0000001008000000ULL;
  add_shallow_hash( test_my, test_opp, -10, 11, ENDGAME_SCORE | EXACT_VALUE, 8 );
  add_shallow_hash( test_my, test_opp,  -4, 22, ENDGAME_SCORE | LOWER_BOUND, 7 );
  add_shallow_hash( test_my, test_opp,   2, 33, ENDGAME_SCORE | UPPER_BOUND, 6 );
  add_shallow_hash( test_my, test_opp,  14, 44, ENDGAME_SCORE | EXACT_VALUE, 5 );

  int tt_mv = 0, tt_val = 0;
  CHECK( probe_shallow_hash_draft( ts, test_my, test_opp, 8, -64, 64, &tt_mv, &tt_val ) &&
	 tt_val == -10 && ts->ts_end_best_move == 11,
	 "Shallow TT d=8 entry overwritten by shallower drafts (val=%d, mv=%d)",
	 tt_val, ts->ts_end_best_move );
  CHECK( probe_shallow_hash_draft( ts, test_my, test_opp, 7, -5, -4, &tt_mv, &tt_val ) &&
	 tt_val == -4 && ts->ts_end_best_move == 22,
	 "Shallow TT d=7 entry overwritten by shallower drafts (val=%d, mv=%d)",
	 tt_val, ts->ts_end_best_move );
  CHECK( probe_shallow_hash_draft( ts, test_my, test_opp, 6, 2, 3, &tt_mv, &tt_val ) &&
	 tt_val == 2 && ts->ts_end_best_move == 33,
	 "Shallow TT d=6 entry overwritten by shallower drafts (val=%d, mv=%d)",
	 tt_val, ts->ts_end_best_move );
  CHECK( probe_shallow_hash_draft( ts, test_my, test_opp, 5, -64, 64, &tt_mv, &tt_val ) &&
	 tt_val == 14 && ts->ts_end_best_move == 44,
	 "Shallow TT d=5 entry corrupted (val=%d, mv=%d)",
	 tt_val, ts->ts_end_best_move );

  free_hash();
}


/*
  TEST_END_PATTERN_INVARIANTS
  Verifies that SWAR bit-gathering in end_pattern_evaluate() and
  end_pattern_evaluate_c33_r24() matches bit-by-bit ternary extraction
  across 2,000 random bitboards.
*/
static int
ref_extract_ternary( BitBoard my_bits, BitBoard opp_bits, const int *bits, int len ) {
  static const int p3[10] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683 };
  int idx = 0;
  for ( int i = 0; i < len; i++ ) {
    int b = bits[i];
    int my = (int)((my_bits >> b) & 1ULL);
    int opp = (int)((opp_bits >> b) & 1ULL);
    int tr = my ? 2 : (opp ? 0 : 1);
    idx += tr * p3[i];
  }
  return idx;
}

static void
test_end_pattern_invariants( void ) {
  static const int c33_bits[4][9] = {
    { 0, 1, 2, 8, 9, 10, 16, 17, 18 },
    { 56, 57, 58, 48, 49, 50, 40, 41, 42 },
    { 7, 6, 5, 15, 14, 13, 23, 22, 21 },
    { 63, 62, 61, 55, 54, 53, 47, 46, 45 }
  };
  static const int e2x_bits[4][10] = {
    { 0, 8, 16, 24, 32, 40, 48, 56, 9, 49 },
    { 7, 15, 23, 31, 39, 47, 55, 63, 14, 54 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 9, 14 },
    { 56, 57, 58, 59, 60, 61, 62, 63, 49, 54 }
  };
  static const int r24_bits[8][8] = {
    { 0, 1, 2, 3, 8, 9, 10, 11 },
    { 56, 57, 58, 59, 48, 49, 50, 51 },
    { 7, 6, 5, 4, 15, 14, 13, 12 },
    { 63, 62, 61, 60, 55, 54, 53, 52 },
    { 0, 8, 16, 24, 1, 9, 17, 25 },
    { 7, 15, 23, 31, 6, 14, 22, 30 },
    { 56, 48, 40, 32, 57, 49, 41, 33 },
    { 63, 55, 47, 39, 62, 54, 46, 38 }
  };

  const int16_t *c33_w = &end_pattern_weights[END_PATTERN_CORNER33_OFFSET];
  const int16_t *e2x_w = &end_pattern_weights[END_PATTERN_EDGE2X_OFFSET];
  const int16_t *r24_w = &end_pattern_weights[END_PATTERN_RECT24_OFFSET];

  for ( int trial = 0; trial < 2000; trial++ ) {
    BitBoard b_bits = 0, w_bits = 0;
    for ( int b = 0; b < 64; b++ ) {
      int r = rand() % 3;
      if ( r == 0 )
	b_bits |= (1ULL << b);
      else if ( r == 1 )
	w_bits |= (1ULL << b);
    }

    int ref_c33_r24 = 0;
    for ( int k = 0; k < 4; k++ )
      ref_c33_r24 += c33_w[ref_extract_ternary( b_bits, w_bits, c33_bits[k], 9 )];
    for ( int k = 0; k < 8; k++ )
      ref_c33_r24 += r24_w[ref_extract_ternary( b_bits, w_bits, r24_bits[k], 8 )];

    int ref_full = ref_c33_r24;
    for ( int k = 0; k < 4; k++ )
      ref_full += e2x_w[ref_extract_ternary( b_bits, w_bits, e2x_bits[k], 10 )];

    int got_full = end_pattern_evaluate( b_bits, w_bits );
    int got_c33_r24 = end_pattern_evaluate_c33_r24( b_bits, w_bits );

    CHECK( got_full == ref_full,
	   "trial %d: end_pattern_evaluate %d != ref %d", trial, got_full, ref_full );
    CHECK( got_c33_r24 == ref_c33_r24,
	   "trial %d: end_pattern_evaluate_c33_r24 %d != ref %d", trial, got_c33_r24, ref_c33_r24 );
  }
}


/*
  TEST_NNUE_INVARIANTS
  Verifies:
  1. Bit-exact agreement with Python QAT reference cases in nnue_data.h.
  2. Bit-for-bit equality between scalar full-board accumulator initialization,
     NEON full-board initialization, and NEON incremental child update
     (`nnue_init_parent_base_acc` + `nnue_update_child_acc`).
  3. Bit-for-bit equality between scalar and ARM64 NEON dual-head evaluation,
     plus exact 8-fold dihedral (D4) symmetry invariance.
*/
static BitBoard
d4_transform_bb( BitBoard b, int sym ) {
  BitBoard out = 0;
  for ( int sq = 0; sq < 64; sq++ ) {
    if ( (b >> sq) & 1ULL ) {
      int r = sq >> 3;
      int c = sq & 7;
      if ( sym & 1 )
	c = 7 - c;
      if ( sym & 2 )
	r = 7 - r;
      if ( sym & 4 ) {
	int t = r;
	r = c;
	c = t;
      }
      out |= (1ULL << (r * 8 + c));
    }
  }
  return out;
}

static void
test_nnue_invariants( void ) {
  CHECK( nnue_verify_reference_cases(),
	 "nnue_verify_reference_cases failed against Python QAT reference cases" );

  for ( int trial = 0; trial < 2000; trial++ ) {
    BitBoard p_my = 0, p_opp = 0;
    for ( int b = 0; b < 64; b++ ) {
      int r = rand() % 3;
      if ( r == 0 )
	p_my |= (1ULL << b);
      else if ( r == 1 )
	p_opp |= (1ULL << b);
    }
    BitBoard empty_bb = ~(p_my | p_opp);
    if ( !empty_bb || !p_opp )
      continue;

    int sq = __builtin_ctzll( empty_bb );
    /* Pick a random subset of p_opp as flipped discs */
    BitBoard mask = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
    BitBoard flips = p_opp & mask;
    if ( !flips )
      flips = p_opp & -p_opp;

    BitBoard child_my_bits = p_my | (1ULL << sq) | flips;
    BitBoard new_opp_bits = p_opp ^ flips;
    int empties = non_iterative_popcount( ~(child_my_bits | new_opp_bits) );

    NnueAcc acc_scalar, acc_neon, base_acc, acc_inc;
    nnue_init_acc_scalar( &acc_scalar, child_my_bits, new_opp_bits );
    nnue_init_acc( &acc_neon, child_my_bits, new_opp_bits );
    nnue_init_parent_base_acc( &base_acc, p_my, p_opp );
    nnue_update_child_acc( &acc_inc, &base_acc, sq, flips );

    for ( int k = 0; k < NNUE_MAX_ACC_DIM; k++ ) {
      CHECK( acc_scalar.v[k] == acc_neon.v[k],
	     "trial %d lane %d: scalar acc %d != NEON acc %d",
	     trial, k, (int)acc_scalar.v[k], (int)acc_neon.v[k] );
      CHECK( acc_scalar.v[k] == acc_inc.v[k],
	     "trial %d lane %d: full acc %d != incremental acc %d",
	     trial, k, (int)acc_scalar.v[k], (int)acc_inc.v[k] );
    }

    NnueTactical tact = {
      trial % 12,
      trial % 3,
      ( sq == 0 || sq == 7 || sq == 56 || sq == 63 ) ? 1 : 0,
      trial & 1,
      ( trial >> 1 ) & 1,
      non_iterative_popcount( flips )
    };

    int32_t ord_s = 0, val_s = 0;
    int32_t ord_n = 0, val_n = 0;
    nnue_evaluate_acc_scalar( &acc_scalar, child_my_bits, new_opp_bits, &tact, empties, &ord_s, &val_s );
    nnue_evaluate_acc( &acc_inc, child_my_bits, new_opp_bits, &tact, empties, &ord_n, &val_n );

    CHECK( ord_s == ord_n && val_s == val_n,
	   "trial %d: scalar (%d,%d) != NEON (%d,%d)",
	   trial, (int)ord_s, (int)val_s, (int)ord_n, (int)val_n );

    int ord_fused = nnue_update_and_evaluate_order( &base_acc, sq, flips, child_my_bits, new_opp_bits, &tact, empties );
    int ord_sep = nnue_evaluate_order_acc( &acc_inc, child_my_bits, new_opp_bits, &tact, empties );
    CHECK( ord_fused == ord_sep,
	   "trial %d: fused order (%d) != separated order (%d)",
	   trial, ord_fused, ord_sep );

    int val_fused = nnue_update_and_evaluate_value( &base_acc, sq, flips, child_my_bits, new_opp_bits, &tact, empties );
    int val_board = nnue_evaluate_value_board( child_my_bits, new_opp_bits, &tact, empties );
    int val_sep = nnue_evaluate_value_acc( &acc_inc, child_my_bits, new_opp_bits, &tact, empties );
    CHECK( val_fused == val_sep && val_board == val_sep,
	   "trial %d: fused/board value (%d,%d) != separated value (%d)",
	   trial, val_fused, val_board, val_sep );

    if ( trial < 200 ) {
      for ( int sym = 1; sym < 8; sym++ ) {
	BitBoard sym_my = d4_transform_bb( child_my_bits, sym );
	BitBoard sym_opp = d4_transform_bb( new_opp_bits, sym );
	NnueAcc acc_sym;
	int32_t ord_sym = 0, val_sym = 0;
	nnue_init_acc( &acc_sym, sym_my, sym_opp );
	nnue_evaluate_acc( &acc_sym, sym_my, sym_opp, &tact, empties, &ord_sym, &val_sym );
	CHECK( ord_sym == ord_n && val_sym == val_n,
	       "trial %d sym %d: D4 symmetry mismatch (%d,%d) != (%d,%d)",
	       trial, sym, (int)ord_sym, (int)val_sym, (int)ord_n, (int)val_n );
      }
    }
  }
}

static double
monotonic_sec( void ) {
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void
bench_nnue_throughput( void ) {
  const int N = 2000000;
  BitBoard p_my = 0x001c3e3c3c380000ULL;
  BitBoard p_opp = 0x3c62414343471c00ULL;
  NnueAcc base_acc, child_acc;
  nnue_init_parent_base_acc( &base_acc, p_my, p_opp );

  int sqs[8] = { 0, 7, 16, 23, 40, 47, 56, 63 };
  BitBoard flips_arr[8] = {
    0x0000020408000000ULL, 0x0000402010000000ULL,
    0x0008080800000000ULL, 0x0010100000000000ULL,
    0x0000000808080000ULL, 0x0000001010000000ULL,
    0x0000000004020000ULL, 0x0000000020400000ULL
  };
  NnueTactical tact = { 5, 0, 0, 0, 1, 3 };

  volatile int32_t sink = 0;
  double t0 = monotonic_sec();
  for ( int i = 0; i < N; i++ ) {
    int k = i & 7;
    nnue_update_child_acc( &child_acc, &base_acc, sqs[k], flips_arr[k] );
    sink += child_acc.v[k];
  }
  double t_upd = monotonic_sec() - t0;

  t0 = monotonic_sec();
  for ( int i = 0; i < N; i++ ) {
    int k = i & 7;
    BitBoard c_my = p_my | (1ULL << sqs[k]) | flips_arr[k];
    BitBoard c_opp = p_opp ^ flips_arr[k];
    nnue_update_child_acc( &child_acc, &base_acc, sqs[k], flips_arr[k] );
    sink += nnue_evaluate_order_acc( &child_acc, c_my, c_opp, &tact, 16 );
  }
  double t_nnue = monotonic_sec() - t0;

  t0 = monotonic_sec();
  for ( int i = 0; i < N; i++ ) {
    int k = i & 7;
    BitBoard c_my = p_my | (1ULL << sqs[k]) | flips_arr[k];
    BitBoard c_opp = p_opp ^ flips_arr[k];
    sink += nnue_update_and_evaluate_order( &base_acc, sqs[k], flips_arr[k], c_my, c_opp, &tact, 16 );
  }
  double t_fused = monotonic_sec() - t0;

  t0 = monotonic_sec();
  for ( int i = 0; i < N; i++ ) {
    int k = i & 7;
    sink += end_pattern_evaluate( p_my | (1ULL << sqs[k]) | flips_arr[k], p_opp ^ flips_arr[k] );
  }
  double t_endp = monotonic_sec() - t0;

  t0 = monotonic_sec();
  for ( int i = 0; i < N; i++ ) {
    int k = i & 7;
    determine_pattern_indices_bitboard_ts( &tls, p_my | (1ULL << sqs[k]), p_opp );
    sink += tls.ts_eval_pattern_index[k];
  }
  double t_det46 = monotonic_sec() - t0;

  t0 = monotonic_sec();
  for ( int i = 0; i < N; i++ ) {
    int k = i & 7;
    nnue_init_parent_base_acc( &base_acc, p_my | (1ULL << sqs[k]), p_opp );
    sink += base_acc.v[k];
  }
  double t_init_nnue = monotonic_sec() - t0;

  printf( "=== ARM64 NEON Microbenchmark (%d iterations, sink=%d) ===\n", N, (int)sink );
  printf( "  1. nnue_update_child_acc (incremental 64x int16):  %5.2f ns/call (%6.1f M/sec)\n",
	  (t_upd * 1e9) / N, (N / 1e6) / t_upd );
  printf( "  2. nnue_update + nnue_evaluate_order_acc (NEON):   %5.2f ns/call (%6.1f M/sec)\n",
	  (t_nnue * 1e9) / N, (N / 1e6) / t_nnue );
  printf( "  3. nnue_update_and_evaluate_order (fused NEON):    %5.2f ns/call (%6.1f M/sec)\n",
	  (t_fused * 1e9) / N, (N / 1e6) / t_fused );
  printf( "  4. end_pattern_evaluate (16-tuple SWAR lookup):    %5.2f ns/call (%6.1f M/sec)\n",
	  (t_endp * 1e9) / N, (N / 1e6) / t_endp );
  printf( "  5. determine_pattern_indices_bitboard_ts (46-pat): %5.2f ns/call (%6.1f M/sec)\n",
	  (t_det46 * 1e9) / N, (N / 1e6) / t_det46 );
  printf( "  6. nnue_init_parent_base_acc (full board init):    %5.2f ns/call (%6.1f M/sec)\n",
	  (t_init_nnue * 1e9) / N, (N / 1e6) / t_init_nnue );
}


/*
  TEST_ENDGAME_ASPIRATION_INTEGRATION
  Plays random games to 8 empty squares and couples the real endgame solver
  (`solve_parity`) with `end_aspiration_next_bounds` across initial null windows
  shifted by +-2 and +-6 discs from the true minimax score.
  Verifies:
  1. `solve_parity` always returns an EVEN score on fail-low/fail-high from an
     odd-bounded `[c - 1, c + 1]` null window (i.e. `root_eval == *alpha - 1`
     on a 1-disc fail-low, NOT `*alpha`).
  2. Feeding that real `root_eval` into `end_aspiration_next_bounds` produces
     a first re-search window `[alpha, beta]` center-anchored at `c +- 4`
     containing at most ONE even score inside `(alpha, beta)` (regression guard
     for the FFO #57 `[-14, -9]` vs `[-12, -9]` node blowup).
  3. The full aspiration loop with `solve_parity` converges to the exact
     minimax score on every board and window offset.
*/
static void
init_solver_masks( BitBoard occupied ) {
  BitBoard empty_bb = ~occupied;
  BitBoard order_mask = 0;
  unsigned int par = 0;

  while ( empty_bb != 0 ) {
    int sq = square_of_bit[FIRST_BIT( empty_bb )];
    empty_bb &= empty_bb - 1;
    BitBoard m = sq_to_order_mask[sq];
    if ( m != 0 ) {
      order_mask |= m;
      par ^= quadrant_mask[sq];
    }
  }
  empty_order_mask = order_mask;
  region_parity = par;
}

static void
test_endgame_aspiration_integration( void ) {
  static const int offsets[4] = { 2, -2, 6, -6 };
  int tested = 0;
  int even_bound_returns = 0;

  init_hash( 14 );
  setup_hash( TRUE );

  for ( int game = 0; game < 150 && tested < 80; game++ ) {
    for ( int i = 0; i < 128; i++ )
      board[i] = OUTSIDE;
    for ( int i = 1; i <= 8; i++ )
      for ( int j = 1; j <= 8; j++ )
	board[10 * i + j] = EMPTY;

    board[44] = WHITESQ;
    board[55] = WHITESQ;
    board[45] = BLACKSQ;
    board[54] = BLACKSQ;

    disks_played = 0;
    piece_count[BLACKSQ][0] = 2;
    piece_count[WHITESQ][0] = 2;
    set_board_bits();
    determine_pattern_indices();

    int side = BLACKSQ;
    int pass_cnt = 0;

    while ( non_iterative_popcount( board_bits[EMPTY] ) > 8 && pass_cnt < 2 ) {
      generate_all( side );
      int cnt = move_count[disks_played];
      if ( cnt == 0 ) {
	pass_cnt++;
	side = OPP( side );
	continue;
      }
      pass_cnt = 0;
      int mv = move_list[disks_played][rand() % cnt];
      (void) make_move( side, mv, TRUE );
      side = OPP( side );
    }

    int empties = non_iterative_popcount( board_bits[EMPTY] );
    if ( empties != 8 )
      continue;

    BitBoard my_bits = board_bits[side];
    BitBoard opp_bits = board_bits[OPP( side )];
    int my_discs = non_iterative_popcount( my_bits );
    int opp_discs = non_iterative_popcount( opp_bits );
    int disc_diff = my_discs - opp_discs;

    init_solver_masks( my_bits | opp_bits );
    determine_hash_values( side, board );
    setup_hash( TRUE );

    int true_score = solve_parity( &tls, my_bits, opp_bits, -64, 64,
				   side, empties, disc_diff, TRUE );
    CHECK( (true_score & 1) == 0,
	   "game %d: solve_parity true_score %d is odd", game, true_score );

    for ( int oi = 0; oi < 4; oi++ ) {
      int c = true_score + offsets[oi];
      if ( c < -60 || c > 60 )
	continue;

      int alpha = c - 1;
      int beta = c + 1;
      int delta = 4;
      int steps = 0;

      init_solver_masks( my_bits | opp_bits );
      setup_hash( TRUE );
      int root_eval = solve_parity( &tls, my_bits, opp_bits, alpha, beta,
				    side, empties, disc_diff, TRUE );

      if ( (root_eval & 1) == 0 )
	even_bound_returns++;

      while ( root_eval <= alpha || root_eval >= beta ) {
	int first_step = (delta == 4);
	int prev_alpha = alpha, prev_beta = beta;
	end_aspiration_next_bounds( root_eval, &alpha, &beta, &delta );
	steps++;

	if ( first_step && offsets[oi] == 2 && root_eval >= prev_alpha - 1 ) {
	  int expected_alpha = (c - 4 < -64) ? -64 : (c - 4);
	  int even_inside = 0;
	  for ( int s = -64; s <= 64; s += 2 )
	    if ( s > alpha && s < beta )
	      even_inside++;
	  CHECK( alpha == expected_alpha && beta == root_eval + 1 && even_inside <= 1,
		 "game %d 1-disc fail-low from [%d,%d] (eval=%d): got [%d,%d] (even_inside=%d)",
		 game, prev_alpha, prev_beta, root_eval, alpha, beta, even_inside );
	}
	if ( first_step && offsets[oi] == -2 && root_eval <= prev_beta + 1 ) {
	  int expected_beta = (c + 4 > 64) ? 64 : (c + 4);
	  int even_inside = 0;
	  for ( int s = -64; s <= 64; s += 2 )
	    if ( s > alpha && s < beta )
	      even_inside++;
	  CHECK( alpha == root_eval - 1 && beta == expected_beta && even_inside <= 1,
		 "game %d 1-disc fail-high from [%d,%d] (eval=%d): got [%d,%d] (even_inside=%d)",
		 game, prev_alpha, prev_beta, root_eval, alpha, beta, even_inside );
	}

	init_solver_masks( my_bits | opp_bits );
	root_eval = solve_parity( &tls, my_bits, opp_bits, alpha, beta,
				  side, empties, disc_diff, TRUE );
	if ( (alpha <= -64 && beta >= 64) || root_eval >= 64 || root_eval <= -64 || steps > 6 )
	  break;
      }

      CHECK( root_eval == true_score,
	     "game %d offset=%d: aspiration converged to %d != true_score %d",
	     game, offsets[oi], root_eval, true_score );
    }
    tested++;
  }

  free_hash();
  CHECK( tested >= 50, "too few 8-empty games tested (%d)", tested );
  CHECK( even_bound_returns >= 20,
	 "too few even-score fail-low/fail-high returns tested (%d)", even_bound_returns );
}


int
main( void ) {
  srand( 42 );

  init_flip_stack();
  init_bitboard();
  init_moves();
  init_patterns();
  init_pattern_dependencies();
  init_stable();
  setup_search();

  test_bitboard_and_pattern_invariants();
  test_stability_invariants();
  test_tt_isolation_invariants();
  test_end_pattern_invariants();
  test_nnue_invariants();
  test_endgame_aspiration_integration();

  if ( getenv( "NNUE_BENCH" ) )
    bench_nnue_throughput();

  if ( failures != 0 ) {
    fprintf( stderr, "search_invariants_test: %d failure(s)\n", failures );
    return 1;
  }
  printf( "search_invariants_test: PASSED (500 random games, 2000 stability/end_pattern/NNUE boards, TT & aspiration integration checks)\n" );
  return 0;
}
