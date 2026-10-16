/*
   File:          end_leaf.c

   Created:       2026

   Contents:      Leaf solvers (1 to 8 empty squares) for the endgame search.
*/

#include "porting.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bitbcnt.h"
#include "bitboard.h"
#include "bitbtest.h"
#include "constant.h"
#include "counter.h"
#include "end.h"
#include "end_leaf.h"
#include "hash.h"
#include "macros.h"
#include "search.h"
#include "stable.h"
#include "tlstate.h"

/*
  Compact 32-bit state word for leaf solvers (fits in argument register w7):
    bits 0..3 : quadrant parity bitmask (0x0..0xF)
    bits 4..5 : side to move color (BLACKSQ = 0, WHITESQ = 2; XOR 0x20 flips side)
    bit  8    : pass_legal flag (1 = pass allowed, 0 = second consecutive pass)
*/
#define LEAF_STATE( par, color, pass_legal ) \
  (((unsigned int)(par) & 0xFU) | ((unsigned int)(color) << 4) | ((unsigned int)((pass_legal) != 0) << 8))
#define LEAF_PAR( state )        ((state) & 0xFU)
#define LEAF_COLOR( state )      ((int)(((state) >> 4) & 3U))
#define LEAF_PASS_LEGAL( state ) ((int)(((state) >> 8) & 1U))
#define LEAF_NEXT_STATE( state, sq ) \
  (((state) ^ 0x20U ^ quadrant_mask[sq]) | 0x100U)
#define LEAF_PASS_STATE( state ) \
  (((state) ^ 0x20U) & ~0x100U)



static ALWAYS_INLINE uint64_t
leaf_remove_empty_byte( uint64_t empties, unsigned int sh ) {
  uint64_t low_mask = (1ULL << sh) - 1ULL;
  return (empties & low_mask) | ((empties >> 8) & ~low_mask);
}



#if USE_SHALLOW_TT
static ALWAYS_INLINE int
leaf_probe_shallow_tt( ThreadState *ts,
		       int draft,
		       int alpha,
		       int beta,
		       int *tt_move,
		       int *tt_val ) {
  return probe_shallow_hash_draft( ts, draft, alpha, beta, tt_move, tt_val );
}

static ALWAYS_INLINE void
leaf_promote_tt_move_packed( uint64_t *cand_sq_ptr,
			     uint64_t *cand_sh_ptr,
			     int max_bits,
			     int tt_move ) {
  if ( tt_move == 0 )
    return;
  uint64_t cand_sq = *cand_sq_ptr;
  if ( (int)(cand_sq & 0xFF) == tt_move )
    return;
  int sh;
  for ( sh = 8; sh < max_bits; sh += 8 ) {
    if ( (int)((cand_sq >> sh) & 0xFF) == tt_move ) {
      uint64_t cand_sh = *cand_sh_ptr;
      uint64_t low_mask = (1ULL << sh) - 1ULL;
      uint64_t clear_mask = low_mask | (0xFFULL << sh);
      uint64_t sq_byte = (cand_sq >> sh) & 0xFFULL;
      uint64_t sh_byte = (cand_sh >> sh) & 0xFFULL;
      *cand_sq_ptr = sq_byte | ((cand_sq & low_mask) << 8) | (cand_sq & ~clear_mask);
      *cand_sh_ptr = sh_byte | ((cand_sh & low_mask) << 8) | (cand_sh & ~clear_mask);
      return;
    }
  }
}
#endif



#if USE_STABILITY
static ALWAYS_INLINE int
leaf_stability_cutoff( int oppcol,
		       BitBoard my_bits,
		       BitBoard opp_bits,
		       int alpha,
		       int *beta_ptr ) {
  int beta = *beta_ptr;
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 && (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    if ( alpha >= 0 && (64 - 2 * opp_cnt <= alpha || 64 - 2 * opp_cnt < beta) ) {
      EdgeIndices edges;
      int s_edge = count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      int stability_bound = 64 - 2 * s_edge;
      if ( stability_bound <= alpha )
	return TRUE;
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
	int max_stable = s_edge + non_iterative_popcount( opp_bits & CENTRAL_MASK );
	if ( 64 - 2 * max_stable <= alpha || 64 - 2 * max_stable < beta ) {
	  stability_bound = 64 - 2 * count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
	}
	if ( stability_bound < beta )
	  *beta_ptr = stability_bound + 1;
	if ( stability_bound <= alpha )
	  return TRUE;
      }
    }
  }
  return FALSE;
}
#endif



static ALWAYS_INLINE int
eval_two_empty_move( ThreadState *ts,
		     int sq_other,
		     BitBoard opp_bits,
		     BitBoard new_my_bits,
		     int flipped,
		     int beta,
		     int disc_diff ) {
  int ev = disc_diff + 2 * flipped;
  int opp_flipped;

  ts->ts_nodes.lo++;

  opp_flipped = CountFlips_bitboard( sq_other, opp_bits & ~new_my_bits );
  if ( opp_flipped != 0 )
    ev -= 2 * opp_flipped;
  else if ( ev >= 0 ) {
    ev += 2;
    if ( ev < beta )
      ev += 2 * CountFlips_bitboard( sq_other, new_my_bits );
  }
  else if ( ev < beta ) {
    opp_flipped = CountFlips_bitboard( sq_other, new_my_bits );
    if ( opp_flipped != 0 )
      ev += 2 * (opp_flipped + 1);
  }
  return ev;
}

static int
solve_two_empty( ThreadState *ts,
		 BitBoard my_bits,
		 BitBoard opp_bits,
		 int sq1,
		 int sq2,
		 int alpha,
		 int beta,
		 int disc_diff,
		 int pass_legal ) {
  int score = -INFINITE_EVAL;
  int flipped;
  int ev;
  BitBoard new_my_bits;

  ts->ts_nodes.lo++;

  flipped = TestFlips_wrapper_to( sq1, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    score = eval_two_empty_move( ts, sq2, opp_bits, new_my_bits, flipped, beta, disc_diff );
    if ( score > alpha ) {
      if ( score >= beta )
	return score;
      alpha = score;
    }
  }

  flipped = TestFlips_wrapper_to( sq2, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    ev = eval_two_empty_move( ts, sq1, opp_bits, new_my_bits, flipped, beta, disc_diff );
    if ( ev >= score )
      return ev;
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !pass_legal ) {
      if ( disc_diff > 0 )
	return disc_diff + 2;
      if ( disc_diff < 0 )
	return disc_diff - 2;
      return 0;
    }
    return -solve_two_empty( ts, opp_bits, my_bits, sq1, sq2, -beta,
			     -alpha, -disc_diff, FALSE );
  }

  return score;
}



static int
solve_three_empty( ThreadState *ts,
		   BitBoard my_bits,
		   BitBoard opp_bits,
		   uint64_t empties,
		   int alpha,
		   int beta,
		   int disc_diff,
		   unsigned int state ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
  int flipped;
  int new_disc_diff;
  int ev;

  ts->ts_nodes.lo++;

  int sq1 = (int)(empties & 0xFF);
  int sq2 = (int)((empties >> 8) & 0xFF);
  int sq3 = (int)((empties >> 16) & 0xFF);
  unsigned int par = LEAF_PAR( state );

  if ( !(quadrant_mask[sq1] & par) ) {
    if ( quadrant_mask[sq2] & par ) {
      int tmp = sq1; sq1 = sq2; sq2 = tmp;
    } else {
      int tmp = sq3; sq3 = sq2; sq2 = sq1; sq1 = tmp;
    }
  }

  flipped = TestFlips_wrapper_to( sq1, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    score = -solve_two_empty( ts, new_opp_bits, new_my_bits, sq2, sq3,
			      -beta, -alpha, new_disc_diff, TRUE );
    if ( score >= beta )
      return score;
    else if ( score > alpha )
      alpha = score;
  }

  flipped = TestFlips_wrapper_to( sq2, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_two_empty( ts, new_opp_bits, new_my_bits, sq1, sq3,
			   -beta, -alpha, new_disc_diff, TRUE );
    if ( ev >= beta )
      return ev;
    else if ( ev > score ) {
      score = ev;
      if ( score > alpha )
	alpha = score;
    }
  }

  flipped = TestFlips_wrapper_to( sq3, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_two_empty( ts, new_opp_bits, new_my_bits, sq1, sq2,
			   -beta, -alpha, new_disc_diff, TRUE );
    if ( ev >= score )
      return ev;
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !LEAF_PASS_LEGAL( state ) ) {
      if ( disc_diff > 0 )
	return disc_diff + 3;
      if ( disc_diff < 0 )
	return disc_diff - 3;
      return 0;
    }
    else {
      uint64_t ordered = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16);
      return -solve_three_empty( ts, opp_bits, my_bits, ordered,
				 -beta, -alpha, -disc_diff, LEAF_PASS_STATE( state ) );
    }
  }

  return score;
}



static int
solve_four_empty( ThreadState *ts,
		  BitBoard my_bits,
		  BitBoard opp_bits,
		  uint64_t empties,
		  int alpha,
		  int beta,
		  int disc_diff,
		  unsigned int state ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
  int flipped;
  int new_disc_diff;
  int ev;

  ts->ts_nodes.lo++;

  int sq1 = (int)(empties & 0xFF);
  int sq2 = (int)((empties >> 8) & 0xFF);
  int sq3 = (int)((empties >> 16) & 0xFF);
  int sq4 = (int)((empties >> 24) & 0xFF);
  unsigned int par = LEAF_PAR( state );

  if ( par != 0 && par != 0xFU ) {
    uint64_t odd_sq = 0, even_sq = 0;
    int n_odd = 0, n_even = 0;
    if ( quadrant_mask[sq1] & par ) { odd_sq |= (uint64_t)sq1 << n_odd; n_odd += 8; } else { even_sq |= (uint64_t)sq1 << n_even; n_even += 8; }
    if ( quadrant_mask[sq2] & par ) { odd_sq |= (uint64_t)sq2 << n_odd; n_odd += 8; } else { even_sq |= (uint64_t)sq2 << n_even; n_even += 8; }
    if ( quadrant_mask[sq3] & par ) { odd_sq |= (uint64_t)sq3 << n_odd; n_odd += 8; } else { even_sq |= (uint64_t)sq3 << n_even; n_even += 8; }
    if ( quadrant_mask[sq4] & par ) { odd_sq |= (uint64_t)sq4 << n_odd; n_odd += 8; } else { even_sq |= (uint64_t)sq4 << n_even; n_even += 8; }
    if ( n_odd == 16 ) {
      empties = odd_sq | (even_sq << 16);
      sq1 = (int)(empties & 0xFF);
      sq2 = (int)((empties >> 8) & 0xFF);
      sq3 = (int)((empties >> 16) & 0xFF);
      sq4 = (int)((empties >> 24) & 0xFF);
    }
  }

  flipped = TestFlips_wrapper_to( sq1, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    score = -solve_three_empty( ts, new_opp_bits, new_my_bits, empties >> 8,
				-beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, sq1 ) );
    if ( score >= beta )
      return score;
    else if ( score > alpha )
      alpha = score;
  }

  flipped = TestFlips_wrapper_to( sq2, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_three_empty( ts, new_opp_bits, new_my_bits,
			     (empties & 0xFFULL) | ((empties >> 8) & ~0xFFULL),
			     -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, sq2 ) );
    if ( ev >= beta )
      return ev;
    else if ( ev > score ) {
      score = ev;
      if ( score > alpha )
	alpha = score;
    }
  }

  flipped = TestFlips_wrapper_to( sq3, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_three_empty( ts, new_opp_bits, new_my_bits,
			     (empties & 0xFFFFULL) | ((empties >> 8) & ~0xFFFFULL),
			     -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, sq3 ) );
    if ( ev >= beta )
      return ev;
    else if ( ev > score ) {
      score = ev;
      if ( score > alpha )
	alpha = score;
    }
  }

  flipped = TestFlips_wrapper_to( sq4, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_three_empty( ts, new_opp_bits, new_my_bits, empties,
			     -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, sq4 ) );
    if ( ev >= score )
      return ev;
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !LEAF_PASS_LEGAL( state ) ) {
      if ( disc_diff > 0 )
	return disc_diff + 4;
      if ( disc_diff < 0 )
	return disc_diff - 4;
      return 0;
    }
    else
      return -solve_four_empty( ts, opp_bits, my_bits, empties,
				-beta, -alpha, -disc_diff, LEAF_PASS_STATE( state ) );
  }

  return score;
}



static int
solve_five_empty( ThreadState *ts,
		  BitBoard my_bits,
		  BitBoard opp_bits,
		  uint64_t empties,
		  int alpha,
		  int beta,
		  int disc_diff,
		  unsigned int state ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
  int in_alpha = alpha;
#endif
  int color = LEAF_COLOR( state );
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  ts->ts_nodes.lo++;

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
  int tt_val;
  if ( leaf_probe_shallow_tt( ts, 5, alpha, beta, NULL, &tt_val ) )
    return tt_val;
#endif

#if USE_STABILITY
  if ( leaf_stability_cutoff( oppcol, my_bits, opp_bits, alpha, &beta ) )
    return alpha;
#endif

  /* At 5 empties (odd), par is mathematically guaranteed non-zero. */
  unsigned int par = LEAF_PAR( state );
  uint64_t odd_sq = 0, even_sq = 0;
  uint64_t odd_sh = 0, even_sh = 0;
  int n_odd = 0, n_even = 0;
  int sh, k;

  for ( sh = 0; sh < 40; sh += 8 ) {
    uint64_t sq = (empties >> sh) & 0xFFULL;
    if ( quadrant_mask[sq] & par ) {
      odd_sq |= sq << n_odd;
      odd_sh |= (uint64_t)sh << n_odd;
      n_odd += 8;
    } else {
      even_sq |= sq << n_even;
      even_sh |= (uint64_t)sh << n_even;
      n_even += 8;
    }
  }
  uint64_t cand_sq = odd_sq | (even_sq << n_odd);
  uint64_t cand_sh = odd_sh | (even_sh << n_odd);

  for ( k = 0; k < 5; k++, cand_sq >>= 8, cand_sh >>= 8 ) {
    int m = (int)(cand_sq & 0xFF);
    flipped = TestFlips_wrapper_to( m, my_bits, opp_bits, &new_my_bits );
    if ( flipped != 0 ) {
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
      uint64_t next_empties = leaf_remove_empty_byte( empties, (unsigned int)(cand_sh & 0xFF) );
      ev = -solve_four_empty( ts, new_opp_bits, new_my_bits, next_empties,
			      -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, m ) );
      if ( ev >= beta ) {
	ts->ts_end_best_move = m;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 5, ts );
#endif
	return ev;
      }
      else if ( ev > score ) {
	score = ev;
	if ( score > alpha )
	  alpha = score;
	best_sq = m;
      }
    }
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !LEAF_PASS_LEGAL( state ) ) {
      if ( disc_diff > 0 ) return disc_diff + 5;
      if ( disc_diff < 0 ) return disc_diff - 5;
      return 0;
    }
    else {
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      ev = -solve_five_empty( ts, opp_bits, my_bits, empties,
			      -beta, -alpha, -disc_diff, LEAF_PASS_STATE( state ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      return ev;
    }
  }

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 5, ts );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 5, ts );
#endif
  ts->ts_end_best_move = best_sq;
  return score;
}



static int
solve_six_empty( ThreadState *ts,
		 BitBoard my_bits,
		 BitBoard opp_bits,
		 uint64_t empties,
		 int alpha,
		 int beta,
		 int disc_diff,
		 unsigned int state ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
  int in_alpha = alpha;
#endif
  int color = LEAF_COLOR( state );
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  ts->ts_nodes.lo++;

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
  int tt_val;
  if ( leaf_probe_shallow_tt( ts, 6, alpha, beta, NULL, &tt_val ) )
    return tt_val;
#endif

#if USE_STABILITY
  if ( leaf_stability_cutoff( oppcol, my_bits, opp_bits, alpha, &beta ) )
    return alpha;
#endif

  unsigned int par = LEAF_PAR( state );
  uint64_t cand_sq, cand_sh;
  int sh, k;

  if ( par != 0 ) {
    uint64_t odd_sq = 0, even_sq = 0;
    uint64_t odd_sh = 0, even_sh = 0;
    int n_odd = 0, n_even = 0;
    for ( sh = 0; sh < 48; sh += 8 ) {
      uint64_t sq = (empties >> sh) & 0xFFULL;
      if ( quadrant_mask[sq] & par ) {
	odd_sq |= sq << n_odd;
	odd_sh |= (uint64_t)sh << n_odd;
	n_odd += 8;
      } else {
	even_sq |= sq << n_even;
	even_sh |= (uint64_t)sh << n_even;
	n_even += 8;
      }
    }
    cand_sq = odd_sq | (even_sq << n_odd);
    cand_sh = odd_sh | (even_sh << n_odd);
  }
  else {
    cand_sq = empties;
    cand_sh = 0x282018100800ULL;
  }

  for ( k = 0; k < 6; k++, cand_sq >>= 8, cand_sh >>= 8 ) {
    int m = (int)(cand_sq & 0xFF);
    flipped = TestFlips_wrapper_to( m, my_bits, opp_bits, &new_my_bits );
    if ( flipped != 0 ) {
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
      unsigned int diff1, diff2;
      end_hash_diff( new_my_bits, my_bits, color, m, &diff1, &diff2 );
      ts->ts_hash1 ^= diff1;
      ts->ts_hash2 ^= diff2;
#endif
      uint64_t next_empties = leaf_remove_empty_byte( empties, (unsigned int)(cand_sh & 0xFF) );
      ev = -solve_five_empty( ts, new_opp_bits, new_my_bits, next_empties,
			      -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, m ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
      ts->ts_hash1 ^= diff1;
      ts->ts_hash2 ^= diff2;
#endif
      if ( ev >= beta ) {
	ts->ts_end_best_move = m;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 6, ts );
#endif
	return ev;
      }
      else if ( ev > score ) {
	score = ev;
	if ( score > alpha )
	  alpha = score;
	best_sq = m;
      }
    }
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !LEAF_PASS_LEGAL( state ) ) {
      if ( disc_diff > 0 )
	return disc_diff + 6;
      if ( disc_diff < 0 )
	return disc_diff - 6;
      return 0;
    }
    else {
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      ev = -solve_six_empty( ts, opp_bits, my_bits, empties,
			     -beta, -alpha, -disc_diff, LEAF_PASS_STATE( state ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      return ev;
    }
  }

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 6, ts );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 6, ts );
#endif

  ts->ts_end_best_move = best_sq;
  return score;
}



static int
solve_seven_empty( ThreadState *ts,
		   BitBoard my_bits,
		   BitBoard opp_bits,
		   uint64_t empties,
		   int alpha,
		   int beta,
		   int disc_diff,
		   unsigned int state ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
  int in_alpha = alpha;
#endif
  int color = LEAF_COLOR( state );
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  ts->ts_nodes.lo++;

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
  int tt_move = 0, tt_val;
  if ( leaf_probe_shallow_tt( ts, 7, alpha, beta, &tt_move, &tt_val ) )
    return tt_val;
#endif

#if USE_STABILITY
  if ( leaf_stability_cutoff( oppcol, my_bits, opp_bits, alpha, &beta ) )
    return alpha;
#endif

  /* At 7 empties (odd), par is mathematically guaranteed non-zero. */
  unsigned int par = LEAF_PAR( state );
  uint64_t odd_sq = 0, even_sq = 0;
  uint64_t odd_sh = 0, even_sh = 0;
  int n_odd = 0, n_even = 0;
  int sh, k;

  for ( sh = 0; sh < 56; sh += 8 ) {
    uint64_t sq = (empties >> sh) & 0xFFULL;
    if ( quadrant_mask[sq] & par ) {
      odd_sq |= sq << n_odd;
      odd_sh |= (uint64_t)sh << n_odd;
      n_odd += 8;
    } else {
      even_sq |= sq << n_even;
      even_sh |= (uint64_t)sh << n_even;
      n_even += 8;
    }
  }
  uint64_t cand_sq = odd_sq | (even_sq << n_odd);
  uint64_t cand_sh = odd_sh | (even_sh << n_odd);

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
  leaf_promote_tt_move_packed( &cand_sq, &cand_sh, 56, tt_move );
#endif

  for ( k = 0; k < 7; k++, cand_sq >>= 8, cand_sh >>= 8 ) {
    int m = (int)(cand_sq & 0xFF);
    flipped = TestFlips_wrapper_to( m, my_bits, opp_bits, &new_my_bits );
    if ( flipped != 0 ) {
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
      unsigned int diff1, diff2;
      end_hash_diff( new_my_bits, my_bits, color, m, &diff1, &diff2 );
      ts->ts_hash1 ^= diff1;
      ts->ts_hash2 ^= diff2;
#endif
      uint64_t next_empties = leaf_remove_empty_byte( empties, (unsigned int)(cand_sh & 0xFF) );
      ev = -solve_six_empty( ts, new_opp_bits, new_my_bits, next_empties,
			     -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, m ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
      ts->ts_hash1 ^= diff1;
      ts->ts_hash2 ^= diff2;
#endif
      if ( ev >= beta ) {
	ts->ts_end_best_move = m;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 7, ts );
#endif
	return ev;
      }
      else if ( ev > score ) {
	score = ev;
	if ( score > alpha )
	  alpha = score;
	best_sq = m;
      }
    }
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !LEAF_PASS_LEGAL( state ) ) {
      if ( disc_diff > 0 )
	return disc_diff + 7;
      if ( disc_diff < 0 )
	return disc_diff - 7;
      return 0;
    }
    else {
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      ev = -solve_seven_empty( ts, opp_bits, my_bits, empties,
			       -beta, -alpha, -disc_diff, LEAF_PASS_STATE( state ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      return ev;
    }
  }

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 7, ts );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 7, ts );
#endif

  ts->ts_end_best_move = best_sq;
  return score;
}



static int
solve_eight_empty( ThreadState *ts,
		   BitBoard my_bits,
		   BitBoard opp_bits,
		   uint64_t empties,
		   int alpha,
		   int beta,
		   int disc_diff,
		   unsigned int state ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
  int in_alpha = alpha;
#endif
  int color = LEAF_COLOR( state );
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  ts->ts_nodes.lo++;

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
  int tt_move = 0, tt_val;
  if ( leaf_probe_shallow_tt( ts, 8, alpha, beta, &tt_move, &tt_val ) )
    return tt_val;
#endif

#if USE_STABILITY
  if ( leaf_stability_cutoff( oppcol, my_bits, opp_bits, alpha, &beta ) )
    return alpha;
#endif

  /*
    All empty corners are already at the lowest byte positions of `empties`
    (from prepare_to_solve). Placing (quadrant_mask[sq] & par) || corner
    into odd_sq orders: corners -> odd non-corners -> even non-corners.
  */
  unsigned int par = LEAF_PAR( state );
  uint64_t odd_sq = 0, even_sq = 0;
  uint64_t odd_sh = 0, even_sh = 0;
  int n_odd = 0, n_even = 0;
  int sh, k;

  for ( sh = 0; sh < 64; sh += 8 ) {
    uint64_t sq = (empties >> sh) & 0xFFULL;
    if ( (quadrant_mask[sq] & par) || (square_mask[sq] & CORNER_MASK) ) {
      odd_sq |= sq << n_odd;
      odd_sh |= (uint64_t)sh << n_odd;
      n_odd += 8;
    } else {
      even_sq |= sq << n_even;
      even_sh |= (uint64_t)sh << n_even;
      n_even += 8;
    }
  }
  uint64_t cand_sq = odd_sq;
  uint64_t cand_sh = odd_sh;
  if ( n_odd < 64 ) {
    cand_sq |= even_sq << n_odd;
    cand_sh |= even_sh << n_odd;
  }

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
  leaf_promote_tt_move_packed( &cand_sq, &cand_sh, 64, tt_move );
#endif

  for ( k = 0; k < 8; k++, cand_sq >>= 8, cand_sh >>= 8 ) {
    int m = (int)(cand_sq & 0xFF);
    flipped = TestFlips_wrapper_to( m, my_bits, opp_bits, &new_my_bits );
    if ( flipped != 0 ) {
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
      unsigned int diff1, diff2;
      end_hash_diff( new_my_bits, my_bits, color, m, &diff1, &diff2 );
      ts->ts_hash1 ^= diff1;
      ts->ts_hash2 ^= diff2;
#endif
      uint64_t next_empties = leaf_remove_empty_byte( empties, (unsigned int)(cand_sh & 0xFF) );
      ev = -solve_seven_empty( ts, new_opp_bits, new_my_bits, next_empties,
			       -beta, -alpha, new_disc_diff, LEAF_NEXT_STATE( state, m ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
      ts->ts_hash1 ^= diff1;
      ts->ts_hash2 ^= diff2;
#endif
      if ( ev >= beta ) {
	ts->ts_end_best_move = m;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 8, ts );
#endif
	return ev;
      }
      else if ( ev > score ) {
	score = ev;
	if ( score > alpha )
	  alpha = score;
	best_sq = m;
      }
    }
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !LEAF_PASS_LEGAL( state ) ) {
      if ( disc_diff > 0 )
	return disc_diff + 8;
      if ( disc_diff < 0 )
	return disc_diff - 8;
      return 0;
    }
    else {
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      ev = -solve_eight_empty( ts, opp_bits, my_bits, empties,
			       -beta, -alpha, -disc_diff, LEAF_PASS_STATE( state ) );
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      return ev;
    }
  }

#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 8, ts );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 8, ts );
#endif

  ts->ts_end_best_move = best_sq;
  return score;
}



/*
  SOLVE_PARITY
  Dispatcher to register-packed leaf solvers for depths 1 through 8.
*/

int
solve_parity( ThreadState *ts,
	      BitBoard my_bits,
	      BitBoard opp_bits,
	      int alpha,
	      int beta,
	      int color,
	      int empties,
	      int disc_diff,
	      int pass_legal ) {
  unsigned int state = LEAF_STATE( ts->ts_region_parity, color, pass_legal );

  if ( empties == 8 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    int sq6 = ts->ts_end_move_list[sq5].succ;
    int sq7 = ts->ts_end_move_list[sq6].succ;
    int sq8 = ts->ts_end_move_list[sq7].succ;
    uint64_t packed = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16) |
                      ((uint64_t)sq4 << 24) | ((uint64_t)sq5 << 32) | ((uint64_t)sq6 << 40) |
                      ((uint64_t)sq7 << 48) | ((uint64_t)sq8 << 56);
    return solve_eight_empty( ts, my_bits, opp_bits, packed, alpha, beta, disc_diff, state );
  }

  if ( empties == 7 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    int sq6 = ts->ts_end_move_list[sq5].succ;
    int sq7 = ts->ts_end_move_list[sq6].succ;
    uint64_t packed = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16) |
                      ((uint64_t)sq4 << 24) | ((uint64_t)sq5 << 32) | ((uint64_t)sq6 << 40) |
                      ((uint64_t)sq7 << 48);
    return solve_seven_empty( ts, my_bits, opp_bits, packed, alpha, beta, disc_diff, state );
  }

  if ( empties == 6 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    int sq6 = ts->ts_end_move_list[sq5].succ;
    uint64_t packed = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16) |
                      ((uint64_t)sq4 << 24) | ((uint64_t)sq5 << 32) | ((uint64_t)sq6 << 40);
    return solve_six_empty( ts, my_bits, opp_bits, packed, alpha, beta, disc_diff, state );
  }

  if ( empties == 5 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    uint64_t packed = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16) |
                      ((uint64_t)sq4 << 24) | ((uint64_t)sq5 << 32);
    return solve_five_empty( ts, my_bits, opp_bits, packed, alpha, beta, disc_diff, state );
  }

  if ( empties == 4 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    uint64_t packed = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16) |
                      ((uint64_t)sq4 << 24);
    return solve_four_empty( ts, my_bits, opp_bits, packed, alpha, beta, disc_diff, state );
  }

  if ( empties == 3 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    uint64_t packed = (uint64_t)sq1 | ((uint64_t)sq2 << 8) | ((uint64_t)sq3 << 16);
    return solve_three_empty( ts, my_bits, opp_bits, packed, alpha, beta, disc_diff, state );
  }

  if ( empties == 2 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    return solve_two_empty( ts, my_bits, opp_bits, sq1, sq2,
			    alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 1 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int flipped = CountFlips_bitboard( sq1, my_bits );
    if ( flipped != 0 ) {
      ts->ts_end_best_move = sq1;
      return disc_diff + 2 * flipped + 1;
    }
    if ( !pass_legal ) {
      if ( disc_diff > 0 ) return disc_diff + 1;
      if ( disc_diff < 0 ) return disc_diff - 1;
      return 0;
    }
    flipped = CountFlips_bitboard( sq1, opp_bits );
    if ( flipped != 0 ) {
      ts->ts_end_best_move = sq1;
      return disc_diff - 2 * flipped - 1;
    }
    if ( disc_diff > 0 ) return disc_diff + 1;
    if ( disc_diff < 0 ) return disc_diff - 1;
    return 0;
  }

  return disc_diff;
}
