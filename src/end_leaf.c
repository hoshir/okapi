/*
   File:          end_leaf.c

   Created:       2026

   Contents:      Leaf solvers (4 to 7 empty squares) for the endgame search.
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
solve_four_empty( BitBoard my_bits,
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
  int score = -INFINITE_EVAL;
  int flipped;
  int new_disc_diff;
  int ev;

  INCREMENT_COUNTER( nodes );

  if ( region_parity != 0 ) {
    int m[4];
    int head = 0, tail = 3;
    if ( quadrant_mask[sq1] & region_parity ) m[head++] = sq1; else m[tail--] = sq1;
    if ( quadrant_mask[sq2] & region_parity ) m[head++] = sq2; else m[tail--] = sq2;
    if ( quadrant_mask[sq3] & region_parity ) m[head++] = sq3; else m[tail--] = sq3;
    if ( quadrant_mask[sq4] & region_parity ) m[head++] = sq4; else m[tail--] = sq4;
    sq1 = m[0]; sq2 = m[1]; sq3 = m[2]; sq4 = m[3];
  }

  flipped = TestFlips_wrapper( sq1, my_bits, opp_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    score = -solve_three_empty( new_opp_bits, bb_flips, sq2, sq3, sq4,
				-beta, -alpha, new_disc_diff, TRUE );
    if ( score >= beta )
      return score;
    else if ( score > alpha )
      alpha = score;
  }

  flipped = TestFlips_wrapper( sq2, my_bits, opp_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_three_empty( new_opp_bits, bb_flips, sq1, sq3, sq4,
			     -beta, -alpha, new_disc_diff, TRUE );
    if ( ev >= beta )
      return ev;
    else if ( ev > score ) {
      score = ev;
      if ( score > alpha )
	alpha = score;
    }
  }

  flipped = TestFlips_wrapper( sq3, my_bits, opp_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_three_empty( new_opp_bits, bb_flips, sq1, sq2, sq4,
			     -beta, -alpha, new_disc_diff, TRUE );
    if ( ev >= beta )
      return ev;
    else if ( ev > score ) {
      score = ev;
      if ( score > alpha )
	alpha = score;
    }
  }

  flipped = TestFlips_wrapper( sq4, my_bits, opp_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    ev = -solve_three_empty( new_opp_bits, bb_flips, sq1, sq2, sq3,
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
      return -solve_four_empty( opp_bits, my_bits, sq1, sq2, sq3, sq4,
				-beta, -alpha, -disc_diff, FALSE );
  }

  return score;
}



int
solve_five_empty( BitBoard my_bits,
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
  int score = -INFINITE_EVAL;
  int in_alpha = alpha;
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  INCREMENT_COUNTER( nodes );

#if USE_SHALLOW_TT
  {
    HashEntry entry;
    find_shallow_hash( &entry );
    if ( (entry.draft == 5) && (entry.flags & ENDGAME_SCORE) ) {
      if ( (entry.flags & EXACT_VALUE) ||
	   ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	   ((entry.flags & UPPER_BOUND) && entry.eval <= alpha) ) {
	end_best_move = entry.move[0];
	return entry.eval;
      }
    }
  }
#endif

#if USE_STABILITY
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 && (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    if ( alpha >= 0 && (64 - 2 * opp_cnt <= alpha || 64 - 2 * opp_cnt < beta) ) {
      int stability_bound;
      EdgeIndices edges;
      stability_bound = 64 - 2 * count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      if ( stability_bound <= alpha )
        return alpha;
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
        stability_bound = 64 - 2 * count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        if ( stability_bound < beta )
          beta = stability_bound + 1;
        if ( stability_bound <= alpha )
          return alpha;
      }
    }
  }
#endif

  int orig_sq[5];
  int cand[5];
  int k;

  orig_sq[0] = sq1; orig_sq[1] = sq2; orig_sq[2] = sq3;
  orig_sq[3] = sq4; orig_sq[4] = sq5;

  if ( region_parity != 0 ) {
    int m_even[5];
    int n_odd = 0, n_even = 0;
    int i;
    for ( i = 0; i < 5; i++ ) {
      if ( quadrant_mask[orig_sq[i]] & region_parity )
        cand[n_odd++] = orig_sq[i];
      else
        m_even[n_even++] = orig_sq[i];
    }
    for ( i = 0; i < n_even; i++ )
      cand[n_odd + i] = m_even[i];
  }
  else {
    for ( k = 0; k < 5; k++ )
      cand[k] = orig_sq[k];
  }

  for ( k = 0; k < 5; k++ ) {
    int m = cand[k];
    flipped = TestFlips_wrapper( m, my_bits, opp_bits );
    if ( flipped != 0 ) {
      int rem[4];
      int r = 0, j;
      for ( j = 0; j < 5; j++ ) {
        if ( orig_sq[j] != m )
          rem[r++] = orig_sq[j];
      }
      FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
      region_parity ^= quadrant_mask[m];
      ev = -solve_four_empty( new_opp_bits, bb_flips,
                              rem[0], rem[1], rem[2], rem[3],
                              -beta, -alpha, new_disc_diff, TRUE );
      region_parity ^= quadrant_mask[m];
      if ( ev >= beta ) {
        end_best_move = m;
#if USE_SHALLOW_TT
        add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 5 );
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
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      ev = -solve_five_empty( opp_bits, my_bits, orig_sq[0], orig_sq[1], orig_sq[2], orig_sq[3], orig_sq[4],
                              -beta, -alpha, oppcol, -disc_diff, FALSE );
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      return ev;
    }
  }

#if USE_SHALLOW_TT
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 5 );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 5 );
#endif
  end_best_move = best_sq;
  return score;
}



int
solve_six_empty( BitBoard my_bits,
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
  int score = -INFINITE_EVAL;
  int in_alpha = alpha;
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  INCREMENT_COUNTER( nodes );

#if USE_SHALLOW_TT
  {
    HashEntry entry;
    find_shallow_hash( &entry );
    if ( (entry.draft == 6) && (entry.flags & ENDGAME_SCORE) ) {
      if ( (entry.flags & EXACT_VALUE) ||
	   ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	   ((entry.flags & UPPER_BOUND) && entry.eval <= alpha) ) {
	end_best_move = entry.move[0];
	return entry.eval;
      }
    }
  }
#endif

#if USE_STABILITY
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 && (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    if ( alpha >= 0 && (64 - 2 * opp_cnt <= alpha || 64 - 2 * opp_cnt < beta) ) {
      int stability_bound;
      EdgeIndices edges;
      stability_bound = 64 - 2 * count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      if ( stability_bound <= alpha )
        return alpha;
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
        stability_bound = 64 - 2 * count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        if ( stability_bound < beta )
          beta = stability_bound + 1;
        if ( stability_bound <= alpha )
          return alpha;
      }
    }
  }
#endif

  int orig_sq[6];
  int cand[6];
  int k;

  orig_sq[0] = sq1; orig_sq[1] = sq2; orig_sq[2] = sq3;
  orig_sq[3] = sq4; orig_sq[4] = sq5; orig_sq[5] = sq6;

  if ( region_parity != 0 ) {
    int m_even[6];
    int n_odd = 0, n_even = 0;
    int i;
    for ( i = 0; i < 6; i++ ) {
      if ( quadrant_mask[orig_sq[i]] & region_parity )
	cand[n_odd++] = orig_sq[i];
      else
	m_even[n_even++] = orig_sq[i];
    }
    for ( i = 0; i < n_even; i++ )
      cand[n_odd + i] = m_even[i];
  }
  else {
    for ( k = 0; k < 6; k++ )
      cand[k] = orig_sq[k];
  }

  for ( k = 0; k < 6; k++ ) {
    int m = cand[k];
    flipped = TestFlips_wrapper( m, my_bits, opp_bits );
    if ( flipped != 0 ) {
      int rem[5];
      int r = 0, j;
      for ( j = 0; j < 6; j++ ) {
	if ( orig_sq[j] != m )
	  rem[r++] = orig_sq[j];
      }
      FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
      region_parity ^= quadrant_mask[m];
#if USE_SHALLOW_TT
      unsigned int diff1, diff2;
      end_hash_diff( bb_flips, my_bits, color, m, &diff1, &diff2 );
      hash1 ^= diff1;
      hash2 ^= diff2;
#endif
      ev = -solve_five_empty( new_opp_bits, bb_flips,
			     rem[0], rem[1], rem[2], rem[3], rem[4],
			     -beta, -alpha, oppcol, new_disc_diff, TRUE );
#if USE_SHALLOW_TT
      hash1 ^= diff1;
      hash2 ^= diff2;
#endif
      region_parity ^= quadrant_mask[m];
      if ( ev >= beta ) {
	end_best_move = m;
#if USE_SHALLOW_TT
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 6 );
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
    else
      return -solve_six_empty( opp_bits, my_bits, sq1, sq2, sq3, sq4, sq5, sq6,
			       -beta, -alpha, oppcol, -disc_diff, FALSE );
  }

#if USE_SHALLOW_TT
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 6 );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 6 );
#endif

  return score;
}



int
solve_seven_empty( BitBoard my_bits,
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
  int score = -INFINITE_EVAL;
  int in_alpha = alpha;
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  INCREMENT_COUNTER( nodes );

#if USE_SHALLOW_TT
  {
    HashEntry entry;
    find_shallow_hash( &entry );
    if ( (entry.draft == 7) && (entry.flags & ENDGAME_SCORE) ) {
      if ( (entry.flags & EXACT_VALUE) ||
	   ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	   ((entry.flags & UPPER_BOUND) && entry.eval <= alpha) ) {
	end_best_move = entry.move[0];
	return entry.eval;
      }
    }
  }
#endif

#if USE_STABILITY
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 && (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    if ( alpha >= 0 && (64 - 2 * opp_cnt <= alpha || 64 - 2 * opp_cnt < beta) ) {
      int stability_bound;
      EdgeIndices edges;
      stability_bound = 64 - 2 * count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      if ( stability_bound <= alpha )
        return alpha;
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
        stability_bound = 64 - 2 * count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        if ( stability_bound < beta )
          beta = stability_bound + 1;
        if ( stability_bound <= alpha )
          return alpha;
      }
    }
  }
#endif

  int orig_sq[7];
  int cand[7];
  int k;

  orig_sq[0] = sq1; orig_sq[1] = sq2; orig_sq[2] = sq3; orig_sq[3] = sq4;
  orig_sq[4] = sq5; orig_sq[5] = sq6; orig_sq[6] = sq7;

  if ( region_parity != 0 ) {
    int m_even[7];
    int n_odd = 0, n_even = 0;
    int i;
    for ( i = 0; i < 7; i++ ) {
      if ( quadrant_mask[orig_sq[i]] & region_parity )
	cand[n_odd++] = orig_sq[i];
      else
	m_even[n_even++] = orig_sq[i];
    }
    for ( i = 0; i < n_even; i++ )
      cand[n_odd + i] = m_even[i];
  }
  else {
    for ( k = 0; k < 7; k++ )
      cand[k] = orig_sq[k];
  }

  for ( k = 0; k < 7; k++ ) {
    int m = cand[k];
    flipped = TestFlips_wrapper( m, my_bits, opp_bits );
    if ( flipped != 0 ) {
      int rem[6];
      int r = 0, j;
      for ( j = 0; j < 7; j++ ) {
	if ( orig_sq[j] != m )
	  rem[r++] = orig_sq[j];
      }
      FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
      region_parity ^= quadrant_mask[m];
#if USE_SHALLOW_TT
      unsigned int diff1, diff2;
      end_hash_diff( bb_flips, my_bits, color, m, &diff1, &diff2 );
      hash1 ^= diff1;
      hash2 ^= diff2;
#endif
      ev = -solve_six_empty( new_opp_bits, bb_flips,
			     rem[0], rem[1], rem[2], rem[3], rem[4], rem[5],
			     -beta, -alpha, oppcol, new_disc_diff, TRUE );
#if USE_SHALLOW_TT
      hash1 ^= diff1;
      hash2 ^= diff2;
#endif
      region_parity ^= quadrant_mask[m];
      if ( ev >= beta ) {
	end_best_move = m;
#if USE_SHALLOW_TT
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 7 );
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
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      ev = -solve_seven_empty( opp_bits, my_bits,
			       orig_sq[0], orig_sq[1], orig_sq[2], orig_sq[3], orig_sq[4], orig_sq[5], orig_sq[6],
			       -beta, -alpha, oppcol, -disc_diff, FALSE );
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      return ev;
    }
  }

#if USE_SHALLOW_TT
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 7 );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 7 );
#endif

  return score;
}



int
solve_eight_empty( BitBoard my_bits,
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
  int score = -INFINITE_EVAL;
  int in_alpha = alpha;
  int oppcol = OPP( color );
  int flipped;
  int new_disc_diff;
  int ev;
  int best_sq = 0;

  INCREMENT_COUNTER( nodes );

#if USE_SHALLOW_TT
  {
    HashEntry entry;
    find_shallow_hash( &entry );
    if ( (entry.draft == 8) && (entry.flags & ENDGAME_SCORE) ) {
      if ( (entry.flags & EXACT_VALUE) ||
	   ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	   ((entry.flags & UPPER_BOUND) && entry.eval <= alpha) ) {
	end_best_move = entry.move[0];
	return entry.eval;
      }
    }
  }
#endif

#if USE_STABILITY
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 && (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    if ( alpha >= 0 && (64 - 2 * opp_cnt <= alpha || 64 - 2 * opp_cnt < beta) ) {
      int stability_bound;
      EdgeIndices edges;
      stability_bound = 64 - 2 * count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      if ( stability_bound <= alpha )
        return alpha;
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
        stability_bound = 64 - 2 * count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        if ( stability_bound < beta )
          beta = stability_bound + 1;
        if ( stability_bound <= alpha )
          return alpha;
      }
    }
  }
#endif

  int orig_sq[8];
  int cand[8];
  int i, k;

  orig_sq[0] = sq1; orig_sq[1] = sq2; orig_sq[2] = sq3; orig_sq[3] = sq4;
  orig_sq[4] = sq5; orig_sq[5] = sq6; orig_sq[6] = sq7; orig_sq[7] = sq8;

  int corners[8], odds[8], evens[8];
  int n_corner = 0, n_odd = 0, n_even = 0;
  for ( i = 0; i < 8; i++ ) {
    int sq = orig_sq[i];
    if ( (1ULL << sq) & CORNER_MASK )
      corners[n_corner++] = sq;
    else if ( quadrant_mask[sq] & region_parity )
      odds[n_odd++] = sq;
    else
      evens[n_even++] = sq;
  }
  int idx = 0;
  for ( i = 0; i < n_corner; i++ ) cand[idx++] = corners[i];
  for ( i = 0; i < n_odd; i++ ) cand[idx++] = odds[i];
  for ( i = 0; i < n_even; i++ ) cand[idx++] = evens[i];

  for ( k = 0; k < 8; k++ ) {
    int m = cand[k];
    flipped = TestFlips_wrapper( m, my_bits, opp_bits );
    if ( flipped != 0 ) {
      int rem[7];
      int r = 0, j;
      for ( j = 0; j < 8; j++ ) {
	if ( orig_sq[j] != m )
	  rem[r++] = orig_sq[j];
      }
      FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
      new_disc_diff = -disc_diff - 2 * flipped - 1;
      region_parity ^= quadrant_mask[m];
#if USE_SHALLOW_TT
      unsigned int diff1, diff2;
      end_hash_diff( bb_flips, my_bits, color, m, &diff1, &diff2 );
      hash1 ^= diff1;
      hash2 ^= diff2;
#endif
      ev = -solve_seven_empty( new_opp_bits, bb_flips,
			       rem[0], rem[1], rem[2], rem[3], rem[4], rem[5], rem[6],
			       -beta, -alpha, oppcol, new_disc_diff, TRUE );
#if USE_SHALLOW_TT
      hash1 ^= diff1;
      hash2 ^= diff2;
#endif
      region_parity ^= quadrant_mask[m];
      if ( ev >= beta ) {
	end_best_move = m;
#if USE_SHALLOW_TT
	add_shallow_hash( ev, m, ENDGAME_SCORE | LOWER_BOUND, 8 );
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
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      ev = -solve_eight_empty( opp_bits, my_bits,
			       orig_sq[0], orig_sq[1], orig_sq[2], orig_sq[3],
			       orig_sq[4], orig_sq[5], orig_sq[6], orig_sq[7],
			       -beta, -alpha, oppcol, -disc_diff, FALSE );
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      return ev;
    }
  }

#if USE_SHALLOW_TT
  if ( score <= in_alpha )
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | UPPER_BOUND, 8 );
  else
    add_shallow_hash( score, best_sq, ENDGAME_SCORE | EXACT_VALUE, 8 );
#endif

  return score;
}

