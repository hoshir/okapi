/*
   File:          search_invariants_test.c
   Contents:      Unit and regression tests for core search, bitboard,
                  NEON pattern index, stability bound, and TT invariants.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#include "stable.h"
#include "hash.h"
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


int
main( void ) {
  srand( 42 );

  init_flip_stack();
  init_bitboard();
  init_moves();
  init_patterns();
  init_pattern_dependencies();
  init_stable();

  test_bitboard_and_pattern_invariants();
  test_stability_invariants();
  test_tt_isolation_invariants();
  test_end_pattern_invariants();

  if ( failures != 0 ) {
    fprintf( stderr, "search_invariants_test: %d failure(s)\n", failures );
    return 1;
  }
  printf( "search_invariants_test: PASSED (500 random games, 2000 stability/end_pattern boards, TT isolation checks)\n" );
  return 0;
}
