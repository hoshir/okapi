/*
   File:          end_leaf.c

   Created:       2026

   Contents:      Leaf solvers (4 to 8 empty squares) for the endgame search.
*/

#include "porting.h"

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



int
solve_four_empty( ThreadState *ts,
		  BitBoard my_bits,
		  BitBoard opp_bits,
		  int sq1,
		  int sq2,
		  int sq3,
		  int sq4,
		  int alpha,
		  int beta,
		  int disc_diff,
		  int pass_legal ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
  int flipped;
  int new_disc_diff;
  int ev;

  ts->ts_nodes.lo++;

  unsigned int par = quadrant_mask[sq1] ^ quadrant_mask[sq2] ^ quadrant_mask[sq3] ^ quadrant_mask[sq4];
  if ( par != 0 ) {
    int m_odd[4], m_even[4];
    int n_odd = 0, n_even = 0;
    if ( quadrant_mask[sq1] & par ) m_odd[n_odd++] = sq1; else m_even[n_even++] = sq1;
    if ( quadrant_mask[sq2] & par ) m_odd[n_odd++] = sq2; else m_even[n_even++] = sq2;
    if ( quadrant_mask[sq3] & par ) m_odd[n_odd++] = sq3; else m_even[n_even++] = sq3;
    if ( quadrant_mask[sq4] & par ) m_odd[n_odd++] = sq4; else m_even[n_even++] = sq4;
    if ( n_odd == 2 ) {
      sq1 = m_odd[0]; sq2 = m_odd[1]; sq3 = m_even[0]; sq4 = m_even[1];
    }
  }

  flipped = TestFlips_wrapper_to( sq1, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    score = -solve_three_empty( ts, new_opp_bits, new_my_bits, sq2, sq3, sq4,
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
    ev = -solve_three_empty( ts, new_opp_bits, new_my_bits, sq1, sq3, sq4,
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
    ev = -solve_three_empty( ts, new_opp_bits, new_my_bits, sq1, sq2, sq4,
			     -beta, -alpha, new_disc_diff, TRUE );
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
    ev = -solve_three_empty( ts, new_opp_bits, new_my_bits, sq1, sq2, sq3,
			     -beta, -alpha, new_disc_diff, TRUE );
    if ( ev >= score )
      return ev;
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !pass_legal ) {  /* Four empty squares */
      if ( disc_diff > 0 )
	return disc_diff + 4;
      if ( disc_diff < 0 )
	return disc_diff - 4;
      return 0;
    }
    else
      return -solve_four_empty( ts, opp_bits, my_bits, sq1, sq2, sq3, sq4,
				-beta, -alpha, -disc_diff, FALSE );
  }

  return score;
}



#if USE_SHALLOW_TT
static ALWAYS_INLINE int
leaf_probe_shallow_tt( ThreadState *ts,
		       int draft,
		       int alpha,
		       int beta,
		       int *tt_move,
		       int *tt_val ) {
  HashEntry entry;
  find_shallow_hash( &entry, ts );
  if ( (entry.draft == draft) && (entry.flags & ENDGAME_SCORE) ) {
    if ( (entry.flags & EXACT_VALUE) ||
	 ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	 ((entry.flags & UPPER_BOUND) && entry.eval <= alpha) ) {
      ts->ts_end_best_move = entry.move[0];
      *tt_val = entry.eval;
      return TRUE;
    }
    if ( (tt_move != NULL) && (entry.flags & (LOWER_BOUND | EXACT_VALUE)) )
      *tt_move = entry.move[0];
  }
  return FALSE;
}

static ALWAYS_INLINE void
leaf_promote_tt_move( int *cand, int n, int tt_move ) {
  if ( tt_move != 0 ) {
    int i, k;
    for ( i = 1; i < n; i++ ) {
      if ( (cand[i] & 0xFF) == tt_move ) {
	int tmp = cand[i];
	for ( k = i; k > 0; k-- )
	  cand[k] = cand[k - 1];
	cand[0] = tmp;
	break;
      }
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



int
solve_five_empty( ThreadState *ts,
		  BitBoard my_bits,
		  BitBoard opp_bits,
		  int sq1,
		  int sq2,
		  int sq3,
		  int sq4,
		  int sq5,
		  int alpha,
		  int beta,
		  int color,
		  int disc_diff,
		  int pass_legal ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
  int in_alpha = alpha;
#endif
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
  unsigned int par = quadrant_mask[sq1] ^ quadrant_mask[sq2] ^
                     quadrant_mask[sq3] ^ quadrant_mask[sq4] ^
                     quadrant_mask[sq5];
  int cand[5], m_even[5];
  int n_odd = 0, n_even = 0;
  if ( quadrant_mask[sq1] & par ) cand[n_odd++] = (0 << 8) | sq1; else m_even[n_even++] = (0 << 8) | sq1;
  if ( quadrant_mask[sq2] & par ) cand[n_odd++] = (1 << 8) | sq2; else m_even[n_even++] = (1 << 8) | sq2;
  if ( quadrant_mask[sq3] & par ) cand[n_odd++] = (2 << 8) | sq3; else m_even[n_even++] = (2 << 8) | sq3;
  if ( quadrant_mask[sq4] & par ) cand[n_odd++] = (3 << 8) | sq4; else m_even[n_even++] = (3 << 8) | sq4;
  if ( quadrant_mask[sq5] & par ) cand[n_odd++] = (4 << 8) | sq5; else m_even[n_even++] = (4 << 8) | sq5;
  int i, k;
  for ( i = 0; i < n_even; i++ )
    cand[n_odd + i] = m_even[i];

  for ( k = 0; k < 5; k++ ) {
    int m = cand[k] & 0xFF;
    int idx = cand[k] >> 8;
    flipped = TestFlips_wrapper_to( m, my_bits, opp_bits, &new_my_bits );
    if ( flipped != 0 ) {
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
      ev = -solve_four_empty( ts, new_opp_bits, new_my_bits,
                              (idx <= 0) ? sq2 : sq1,
                              (idx <= 1) ? sq3 : sq2,
                              (idx <= 2) ? sq4 : sq3,
                              (idx <= 3) ? sq5 : sq4,
                              -beta, -alpha, new_disc_diff, TRUE );
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
    if ( !pass_legal ) {
      if ( disc_diff > 0 ) return disc_diff + 5;
      if ( disc_diff < 0 ) return disc_diff - 5;
      return 0;
    }
    else {
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 5)
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
#endif
      ev = -solve_five_empty( ts, opp_bits, my_bits, sq1, sq2, sq3, sq4, sq5,
                              -beta, -alpha, oppcol, -disc_diff, FALSE );
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



int
solve_six_empty( ThreadState *ts,
		 BitBoard my_bits,
		 BitBoard opp_bits,
		 int sq1,
		 int sq2,
		 int sq3,
		 int sq4,
		 int sq5,
		 int sq6,
		 int alpha,
		 int beta,
		 int color,
		 int disc_diff,
		 int pass_legal ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 6)
  int in_alpha = alpha;
#endif
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

  int cand[6];
  int k;

  unsigned int par = quadrant_mask[sq1] ^ quadrant_mask[sq2] ^
                     quadrant_mask[sq3] ^ quadrant_mask[sq4] ^
                     quadrant_mask[sq5] ^ quadrant_mask[sq6];
  if ( par != 0 ) {
    int m_even[6];
    int n_odd = 0, n_even = 0, i;
    if ( quadrant_mask[sq1] & par ) cand[n_odd++] = (0 << 8) | sq1; else m_even[n_even++] = (0 << 8) | sq1;
    if ( quadrant_mask[sq2] & par ) cand[n_odd++] = (1 << 8) | sq2; else m_even[n_even++] = (1 << 8) | sq2;
    if ( quadrant_mask[sq3] & par ) cand[n_odd++] = (2 << 8) | sq3; else m_even[n_even++] = (2 << 8) | sq3;
    if ( quadrant_mask[sq4] & par ) cand[n_odd++] = (3 << 8) | sq4; else m_even[n_even++] = (3 << 8) | sq4;
    if ( quadrant_mask[sq5] & par ) cand[n_odd++] = (4 << 8) | sq5; else m_even[n_even++] = (4 << 8) | sq5;
    if ( quadrant_mask[sq6] & par ) cand[n_odd++] = (5 << 8) | sq6; else m_even[n_even++] = (5 << 8) | sq6;
    for ( i = 0; i < n_even; i++ )
      cand[n_odd + i] = m_even[i];
  }
  else {
    cand[0] = (0 << 8) | sq1; cand[1] = (1 << 8) | sq2; cand[2] = (2 << 8) | sq3;
    cand[3] = (3 << 8) | sq4; cand[4] = (4 << 8) | sq5; cand[5] = (5 << 8) | sq6;
  }

  for ( k = 0; k < 6; k++ ) {
    int m = cand[k] & 0xFF;
    int idx = cand[k] >> 8;
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
      ev = -solve_five_empty( ts, new_opp_bits, new_my_bits,
			     (idx <= 0) ? sq2 : sq1,
			     (idx <= 1) ? sq3 : sq2,
			     (idx <= 2) ? sq4 : sq3,
			     (idx <= 3) ? sq5 : sq4,
			     (idx <= 4) ? sq6 : sq5,
			     -beta, -alpha, oppcol, new_disc_diff, TRUE );
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
    if ( !pass_legal ) {  /* Six empty squares */
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
      ev = -solve_six_empty( ts, opp_bits, my_bits, sq1, sq2, sq3, sq4, sq5, sq6,
			     -beta, -alpha, oppcol, -disc_diff, FALSE );
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



int
solve_seven_empty( ThreadState *ts,
		   BitBoard my_bits,
		   BitBoard opp_bits,
		   int sq1,
		   int sq2,
		   int sq3,
		   int sq4,
		   int sq5,
		   int sq6,
		   int sq7,
		   int alpha,
		   int beta,
		   int color,
		   int disc_diff,
		   int pass_legal ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
  int in_alpha = alpha;
#endif
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
  unsigned int par = quadrant_mask[sq1] ^ quadrant_mask[sq2] ^
                     quadrant_mask[sq3] ^ quadrant_mask[sq4] ^
                     quadrant_mask[sq5] ^ quadrant_mask[sq6] ^
                     quadrant_mask[sq7];
  int cand[7], m_even[7];
  int n_odd = 0, n_even = 0, i, k;
  if ( quadrant_mask[sq1] & par ) cand[n_odd++] = (0 << 8) | sq1; else m_even[n_even++] = (0 << 8) | sq1;
  if ( quadrant_mask[sq2] & par ) cand[n_odd++] = (1 << 8) | sq2; else m_even[n_even++] = (1 << 8) | sq2;
  if ( quadrant_mask[sq3] & par ) cand[n_odd++] = (2 << 8) | sq3; else m_even[n_even++] = (2 << 8) | sq3;
  if ( quadrant_mask[sq4] & par ) cand[n_odd++] = (3 << 8) | sq4; else m_even[n_even++] = (3 << 8) | sq4;
  if ( quadrant_mask[sq5] & par ) cand[n_odd++] = (4 << 8) | sq5; else m_even[n_even++] = (4 << 8) | sq5;
  if ( quadrant_mask[sq6] & par ) cand[n_odd++] = (5 << 8) | sq6; else m_even[n_even++] = (5 << 8) | sq6;
  if ( quadrant_mask[sq7] & par ) cand[n_odd++] = (6 << 8) | sq7; else m_even[n_even++] = (6 << 8) | sq7;
  for ( i = 0; i < n_even; i++ )
    cand[n_odd + i] = m_even[i];
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 7)
  leaf_promote_tt_move( cand, 7, tt_move );
#endif

  for ( k = 0; k < 7; k++ ) {
    int m = cand[k] & 0xFF;
    int idx = cand[k] >> 8;
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
      ev = -solve_six_empty( ts, new_opp_bits, new_my_bits,
			     (idx <= 0) ? sq2 : sq1,
			     (idx <= 1) ? sq3 : sq2,
			     (idx <= 2) ? sq4 : sq3,
			     (idx <= 3) ? sq5 : sq4,
			     (idx <= 4) ? sq6 : sq5,
			     (idx <= 5) ? sq7 : sq6,
			     -beta, -alpha, oppcol, new_disc_diff, TRUE );
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
    if ( !pass_legal ) {  /* Seven empty squares */
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
      ev = -solve_seven_empty( ts, opp_bits, my_bits,
			       sq1, sq2, sq3, sq4, sq5, sq6, sq7,
			       -beta, -alpha, oppcol, -disc_diff, FALSE );
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



int
solve_eight_empty( ThreadState *ts,
		   BitBoard my_bits,
		   BitBoard opp_bits,
		   int sq1,
		   int sq2,
		   int sq3,
		   int sq4,
		   int sq5,
		   int sq6,
		   int sq7,
		   int sq8,
		   int alpha,
		   int beta,
		   int color,
		   int disc_diff,
		   int pass_legal ) {
  BitBoard new_opp_bits;
  BitBoard new_my_bits;
  int score = -INFINITE_EVAL;
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
  int in_alpha = alpha;
#endif
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

  int orig_sq[8];
  int cand[8];
  int i, k;

  orig_sq[0] = sq1; orig_sq[1] = sq2; orig_sq[2] = sq3; orig_sq[3] = sq4;
  orig_sq[4] = sq5; orig_sq[5] = sq6; orig_sq[6] = sq7; orig_sq[7] = sq8;

  unsigned int par = quadrant_mask[sq1] ^ quadrant_mask[sq2] ^
                     quadrant_mask[sq3] ^ quadrant_mask[sq4] ^
                     quadrant_mask[sq5] ^ quadrant_mask[sq6] ^
                     quadrant_mask[sq7] ^ quadrant_mask[sq8];

  int corners[8], odds[8], evens[8];
  int n_corner = 0, n_odd = 0, n_even = 0;
  for ( i = 0; i < 8; i++ ) {
    int sq = orig_sq[i];
    int packed = (i << 8) | sq;
    if ( square_mask[sq] & CORNER_MASK )
      corners[n_corner++] = packed;
    else if ( quadrant_mask[sq] & par )
      odds[n_odd++] = packed;
    else
      evens[n_even++] = packed;
  }
  int idx_out = 0;
  for ( i = 0; i < n_corner; i++ ) cand[idx_out++] = corners[i];
  for ( i = 0; i < n_odd; i++ ) cand[idx_out++] = odds[i];
  for ( i = 0; i < n_even; i++ ) cand[idx_out++] = evens[i];
#if USE_SHALLOW_TT && (SHALLOW_TT_MIN_DEPTH <= 8)
  leaf_promote_tt_move( cand, 8, tt_move );
#endif

  for ( k = 0; k < 8; k++ ) {
    int m = cand[k] & 0xFF;
    int idx = cand[k] >> 8;
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
      ev = -solve_seven_empty( ts, new_opp_bits, new_my_bits,
			       (idx <= 0) ? sq2 : sq1,
			       (idx <= 1) ? sq3 : sq2,
			       (idx <= 2) ? sq4 : sq3,
			       (idx <= 3) ? sq5 : sq4,
			       (idx <= 4) ? sq6 : sq5,
			       (idx <= 5) ? sq7 : sq6,
			       (idx <= 6) ? sq8 : sq7,
			       -beta, -alpha, oppcol, new_disc_diff, TRUE );
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
    if ( !pass_legal ) {  /* Eight empty squares */
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
      ev = -solve_eight_empty( ts, opp_bits, my_bits,
			       sq1, sq2, sq3, sq4, sq5, sq6, sq7, sq8,
			       -beta, -alpha, oppcol, -disc_diff, FALSE );
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
