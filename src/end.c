/*
   File:          end.c

   Created:       1994
   
   Authors:       Gunnar Andersson (gunnar@radagast.se)

   Contents:      The fast endgame solver.
*/



#include "porting.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "autoplay.h"
#include "bitbcnt.h"
#include "bitbmob.h"
#include "bitboard.h"
#include "bitbtest.h"
#include "cntflip.h"
#include "counter.h"
#include "display.h"
#include "doflip.h"
#include "end.h"
#include "end_aspiration.h"
#include "end_leaf.h"
#include "end_patterns.h"
#include "epcstat.h"
#include "eval.h"
#include "getcoeff.h"
#include "globals.h"
#include "hash.h"
#include "macros.h"
#include "midgame.h"
#include "moves.h"
#include "osfbook.h"
#include "search.h"
#include "patterns.h"
#include "probcut.h"
#include "search.h"
#include "stable.h"
#include "threads.h"
#include "texts.h"
#include "timer.h"
#include "unflip.h"



#define USE_MPC                      1
#define MAX_SELECTIVITY              9
#define DISABLE_SELECTIVITY          18

#define PV_EXPANSION                 16

#ifdef _WIN32_WCE
#define EVENT_CHECK_INTERVAL         25000.0
#else
#define EVENT_CHECK_INTERVAL         250000.0
#endif

#define LOW_LEVEL_DEPTH              8
#define HASH_DEPTH                   (LOW_LEVEL_DEPTH + 1)

/* The disc difference when special wipeout move ordering is tried.
   This means more aggressive use of fastest first. */
#define WIPEOUT_THRESHOLD            60
#define REGION_PARITY_BONUS          64



/* Use stability pruning? */
#ifndef USE_STABILITY
#define USE_STABILITY                TRUE
#endif

/* Use shallow transposition table for low-depth endgame nodes? */
#ifndef USE_SHALLOW_TT
#define USE_SHALLOW_TT               TRUE
#define SHALLOW_TT_MIN_DEPTH         5
#endif




/* Pseudo-probabilities corresponding to the percentiles.
   These are taken from the normal distribution; to the percentile
   x corresponds the probability Pr(-x <= Y <= x) where Y is a N(0,1)
   variable. */

static const double confidence[MAX_SELECTIVITY + 1] =
{ 1.000, 0.99, 0.98, 0.954, 0.911, 0.838, 0.729, 0.576, 0.383, 0.197 };

/* Percentiles used in the endgame MPC */
static const double end_percentile[MAX_SELECTIVITY + 1] =
{ 100.0, 4.0, 3.0, 2.0, 1.7, 1.4, 1.1, 0.8, 0.5, 0.25 };

/* Number of discs that the side to move at the root has to win with. */
static int komi_shift;



/*
  BB_VALID_MOVE
  Determine if a (hash) move is legal in the position given by the
  bitboards.  Replaces the array-based valid_move.
  Note: clobbers bb_flips via TestFlips_wrapper.
*/

static int
bb_valid_move( int move, BitBoard my_bits, BitBoard opp_bits ) {
  BitBoard dummy;
  int row = move / 10;
  int col = move % 10;

  if ( (row < 1) || (row > 8) || (col < 1) || (col > 8) )
    return FALSE;

  if ( (my_bits | opp_bits) & square_mask[move] )
    return FALSE;

  return TestFlips_bitboard_to( move, my_bits, opp_bits, &dummy ) > 0;
}



/*
  PREPARE_TO_SOLVE
  Create the list of empty squares.
*/

static void
prepare_to_solve( BitBoard occupied ) {
  /* fixed square ordering: */
  /* jcw's order, which is the best of 4 tried (according to Warren Smith) */
  static const unsigned char worst2best[64] = {
    /*B2*/      22 , 27 , 72 , 77 ,
    /*B1*/      12 , 17 , 21 , 28 , 71 , 78 , 82,  87 ,
    /*C2*/      23 , 26 , 32 , 37 , 62 , 67 , 73 , 76 ,
    /*D2*/      24 , 25 , 42 , 47 , 52 , 57 , 74 , 75 ,
    /*D3*/      34 , 35 , 43 , 46 , 53 , 56 , 64 , 65 ,
    /*C1*/      13 , 16 , 31 , 38 , 61 , 68 , 83 , 86 ,
    /*D1*/      14 , 15 , 41 , 48 , 51 , 58 , 84 , 85 ,
    /*C3*/      33 , 36 , 63 , 66 ,
    /*A1*/      11 , 18 , 81 , 88 , 
    /*D4*/      44 , 45 , 54 , 45
  };
  int i;
  int last_sq;

  region_parity = 0;

  last_sq = END_MOVE_LIST_HEAD;
  for ( i = 59; i >=0; i-- ) {
    int sq = worst2best[i];
    if ( !(occupied & square_mask[sq]) ) {
      end_move_list[last_sq].succ = sq;
      end_move_list[sq].pred = last_sq;
      region_parity ^= quadrant_mask[sq];
      last_sq = sq;
    }
  }
  end_move_list[last_sq].succ = END_MOVE_LIST_TAIL;
}








/*
  SOLVE_TWO_EMPTY
  SOLVE_THREE_EMPTY
  SOLVE_FOUR_EMPTY
  SOLVE_PARITY
  SOLVE_PARITY_HASH
  SOLVE_PARITY_HASH_HIGH
  These are the core routines of the low level endgame code.
  They all perform the same task: Return the score for the side to move.
  Structural differences:
  * SOLVE_TWO_EMPTY may only be called for *exactly* two empty
  * SOLVE_THREE_EMPTY may only be called for *exactly* three empty
  * SOLVE_PARITY delegates to leaf solvers (end_leaf.c) for <= 7 empties
  * SOLVE_PARITY_HASH_HIGH uses stability, TT, fastest-first ordering
    and PVS for 8 to 12 empties
*/

static int
solve_two_empty( BitBoard my_bits,
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

  INCREMENT_COUNTER( nodes );

  /* Overall strategy: Lazy evaluation whenever possible, i.e., don't
     update bitboards until they are used. Also look at alpha and beta
     in order to perform strength reduction: Feasibility testing is
     faster than counting number of flips. */

  /* Try the first of the two empty squares... */

  flipped = TestFlips_wrapper( sq1, my_bits, opp_bits );
  if ( flipped != 0 ) {  /* SQ1 feasible for me */
    INCREMENT_COUNTER( nodes );

    ev = disc_diff + 2 * flipped;

    flipped = CountFlips_bitboard( sq2, opp_bits & ~bb_flips );
    if ( flipped != 0 )
      ev -= 2 * flipped;
    else {  /* He passes, check if SQ2 is feasible for me */
      if ( ev >= 0 ) {  /* I'm ahead, so EV will increase by at least 2 */
	ev += 2;
	if ( ev < beta )  /* Only bother if not certain fail-high */
	  ev += 2 * CountFlips_bitboard( sq2, bb_flips );
      }
      else {
	if ( ev < beta ) {  /* Only bother if not fail-high already */
	  flipped = CountFlips_bitboard( sq2, bb_flips );
	  if ( flipped != 0 )  /* SQ2 feasible for me, game over */
	    ev += 2 * (flipped + 1);
	  /* ELSE: SQ2 will end up empty, game over */
	}
      }
    }

    /* Being legal, the first move is the best so far */
    score = ev;
    if ( score > alpha ) {
      if ( score >= beta )
	return score;
      alpha = score;
    }
  }

  /* ...and then the second */

  flipped = TestFlips_wrapper( sq2, my_bits, opp_bits );
  if ( flipped != 0 ) {  /* SQ2 feasible for me */
    INCREMENT_COUNTER( nodes );

    ev = disc_diff + 2 * flipped;

    flipped = CountFlips_bitboard( sq1, opp_bits & ~bb_flips );
    if ( flipped != 0 )  /* SQ1 feasible for him, game over */
      ev -= 2 * flipped;
    else {  /* He passes, check if SQ1 is feasible for me */
      if ( ev >= 0 ) {  /* I'm ahead, so EV will increase by at least 2 */
	ev += 2;
	if ( ev < beta )  /* Only bother if not certain fail-high */
	  ev += 2 * CountFlips_bitboard( sq1, bb_flips );
      }
      else {
	if ( ev < beta ) {  /* Only bother if not fail-high already */
	  flipped = CountFlips_bitboard( sq1, bb_flips );
	  if ( flipped != 0 )  /* SQ1 feasible for me, game over */
	    ev += 2 * (flipped + 1);
	  /* ELSE: SQ1 will end up empty, game over */
	}
      }
    }

    /* If the second move is better than the first (if that move was legal),
       its score is the score of the position */
    if ( ev >= score )
      return ev;
  }

  /* If both SQ1 and SQ2 are illegal I have to pass,
     otherwise return the best score. */

  if ( score == -INFINITE_EVAL ) {
    if ( !pass_legal ) {  /* Two empty squares */
      if ( disc_diff > 0 )
	return disc_diff + 2;
      if ( disc_diff < 0 )
	return disc_diff - 2;
      return 0;
    }
    else
      return -solve_two_empty( opp_bits, my_bits, sq1, sq2, -beta,
			       -alpha, -disc_diff, FALSE );
  }
  else
    return score;
}


int
solve_three_empty( BitBoard my_bits,
		   BitBoard opp_bits,
		   int sq1,
		   int sq2,
		   int sq3,
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

  unsigned int q1 = quadrant_mask[sq1];
  unsigned int q2 = quadrant_mask[sq2];
  unsigned int q3 = quadrant_mask[sq3];
  unsigned int par = q1 ^ q2 ^ q3;
  if ( !(q1 & par) ) {
    if ( q2 & par ) {
      int tmp = sq1; sq1 = sq2; sq2 = tmp;
    } else {
      int tmp = sq1; sq1 = sq3; sq3 = tmp;
    }
  }

  flipped = TestFlips_wrapper( sq1, my_bits, opp_bits );
  if ( flipped != 0 ) {
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
    new_disc_diff = -disc_diff - 2 * flipped - 1;
    score = -solve_two_empty( new_opp_bits, bb_flips, sq2, sq3,
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
    ev = -solve_two_empty( new_opp_bits, bb_flips, sq1, sq3,
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
    ev = -solve_two_empty( new_opp_bits, bb_flips, sq1, sq2,
			   -beta, -alpha, new_disc_diff, TRUE );
    if ( ev >= score )
      return ev;
  }

  if ( score == -INFINITE_EVAL ) {
    if ( !pass_legal ) {  /* Three empty squares */
      if ( disc_diff > 0 )
	return disc_diff + 3;
      if ( disc_diff < 0 )
	return disc_diff - 3;
      return 0;  /* Can't reach this code, only keep it for symmetry */
    }
    else
      return -solve_three_empty( opp_bits, my_bits, sq1, sq2, sq3,
				 -beta, -alpha, -disc_diff, FALSE );
  }

  return score;
}



/* Forward declaration for end_search_pvs */
static int
end_search_pvs( BitBoard my_bits,
		BitBoard opp_bits,
		int alpha,
		int beta,
		int side_to_move,
		int empties,
		int disc_diff,
		int pass_legal,
		int level,
		int selectivity,
		int *selective_cutoff );

/* Forward declaration for end_search_nws */
static int
end_search_nws( BitBoard my_bits,
		BitBoard opp_bits,
		int alpha,
		int side_to_move,
		int empties,
		int disc_diff,
		int pass_legal,
		int level,
		int selectivity,
		int *selective_cutoff );

/*
  SOLVE_PARITY
  Dispatcher to optimized leaf solvers for depths 1 through 7.
*/

static int
solve_parity( BitBoard my_bits,
	      BitBoard opp_bits,
	      int alpha,
	      int beta, 
	      int color,
	      int empties,
	      int disc_diff,
	      int pass_legal,
	      int level ) {
  if ( empties == 8 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    int sq3 = end_move_list[sq2].succ;
    int sq4 = end_move_list[sq3].succ;
    int sq5 = end_move_list[sq4].succ;
    int sq6 = end_move_list[sq5].succ;
    int sq7 = end_move_list[sq6].succ;
    int sq8 = end_move_list[sq7].succ;
    return solve_eight_empty( my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5, sq6, sq7, sq8,
			      alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 7 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    int sq3 = end_move_list[sq2].succ;
    int sq4 = end_move_list[sq3].succ;
    int sq5 = end_move_list[sq4].succ;
    int sq6 = end_move_list[sq5].succ;
    int sq7 = end_move_list[sq6].succ;
    return solve_seven_empty( my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5, sq6, sq7,
			      alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 6 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    int sq3 = end_move_list[sq2].succ;
    int sq4 = end_move_list[sq3].succ;
    int sq5 = end_move_list[sq4].succ;
    int sq6 = end_move_list[sq5].succ;
    return solve_six_empty( my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5, sq6,
			    alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 5 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    int sq3 = end_move_list[sq2].succ;
    int sq4 = end_move_list[sq3].succ;
    int sq5 = end_move_list[sq4].succ;
    return solve_five_empty( my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5,
			     alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 4 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    int sq3 = end_move_list[sq2].succ;
    int sq4 = end_move_list[sq3].succ;
    return solve_four_empty( my_bits, opp_bits, sq1, sq2, sq3, sq4,
			     alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 3 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    int sq3 = end_move_list[sq2].succ;
    return solve_three_empty( my_bits, opp_bits, sq1, sq2, sq3,
			      alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 2 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = end_move_list[sq1].succ;
    return solve_two_empty( my_bits, opp_bits, sq1, sq2,
			    alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 1 ) {
    int sq1 = end_move_list[END_MOVE_LIST_HEAD].succ;
    int flipped = TestFlips_wrapper( sq1, my_bits, opp_bits );
    if ( flipped != 0 )
      return disc_diff + 2 * flipped + 1;
    if ( !pass_legal ) {
      if ( disc_diff > 0 ) return disc_diff + 1;
      if ( disc_diff < 0 ) return disc_diff - 1;
      return 0;
    }
    flipped = TestFlips_wrapper( sq1, opp_bits, my_bits );
    if ( flipped != 0 )
      return disc_diff - 2 * flipped - 1;
    if ( disc_diff > 0 ) return disc_diff + 1;
    if ( disc_diff < 0 ) return disc_diff - 1;
    return 0;
  }

  if ( empties <= 0 ) {
    if ( disc_diff > 0 ) return disc_diff;
    if ( disc_diff < 0 ) return disc_diff;
    return 0;
  }

  /* Fallback for empties > LOW_LEVEL_DEPTH */
  int selective_cutoff = FALSE;
  return end_search_pvs( my_bits, opp_bits, alpha, beta, color,
			 empties, disc_diff, pass_legal, level,
			 0, &selective_cutoff );
}



/*
  END_MAKE_MOVE
  Make SQ: update incremental hash keys, region parity, and unlink from move list.
*/

INLINE static void
end_make_move( int sq,
	       BitBoard new_my_bits,
	       BitBoard my_bits,
	       int color,
	       unsigned int *diff1,
	       unsigned int *diff2,
	       int *pred,
	       int *succ ) {
  end_hash_diff( new_my_bits, my_bits, color, sq, diff1, diff2 );
  hash1 ^= *diff1;
  hash2 ^= *diff2;
  prefetch_hash_endgame_key( hash2 );
  region_parity ^= quadrant_mask[sq];
  *pred = end_move_list[sq].pred;
  *succ = end_move_list[sq].succ;
  end_move_list[*pred].succ = *succ;
  end_move_list[*succ].pred = *pred;
}



/*
  END_UNMAKE_MOVE
  Unmake SQ: restore incremental hash keys, region parity, and relink into move list.
*/

INLINE static void
end_unmake_move( int sq,
		 unsigned int diff1,
		 unsigned int diff2,
		 int pred,
		 int succ ) {
  region_parity ^= quadrant_mask[sq];
  hash1 ^= diff1;
  hash2 ^= diff2;
  end_move_list[pred].succ = sq;
  end_move_list[succ].pred = sq;
}



/*
  END_PROBE_2PLY_ETC
  Selective 2-Ply Enhanced Transposition Cutoffs (ETC) at cut nodes.
  When opponent mobility after candidate move SQ is low (1..MAX_2PLY_ETC_MOB),
  probes transposition table for opponent replies in O(1) bitboard time.

  Returns:
    ETC_2PLY_CUTOFF  (2): All opponent replies prove >= beta (fail-high cutoff).
    ETC_2PLY_REFUTED (1): At least one opponent reply proves <= alpha (candidate fails low).
    ETC_2PLY_NONE    (0): Inconclusive.
*/

#define ETC_2PLY_NONE     0
#define ETC_2PLY_REFUTED  1
#define ETC_2PLY_CUTOFF   2

#ifndef MIN_2PLY_ETC_DEPTH
#define MIN_2PLY_ETC_DEPTH 11
#endif

#ifndef MAX_2PLY_ETC_MOB
#define MAX_2PLY_ETC_MOB   3
#endif

#define FORCED_NONE     0
#define FORCED_REFUTED  1
#define FORCED_CUTOFF   2

/*
  PROBE_FORCED_REPLY_CUTOFF
  Topology-driven forced reply follow-through pruning (PRUN-007).
  When candidate move leaves opponent with exactly 1 legal reply (B = 1),
  the sub-path is non-branching (forced reply / 一本道).
  Without artificial depth thresholds, rolls out along the forced corridor
  using pure 64-bit bitboard state in CPU registers:
  - Advances moves along the corridor (including passes).
  - Probes Transposition Table at each intermediate state for instant beta-cutoff or fail-low refutation.
  - Zero mutation of end_move_list or global state (100% thread-safe and non-invasive).
*/

INLINE static int
probe_forced_reply_cutoff( int cand_sq,
			   BitBoard child_my_bits,
			   BitBoard child_opp_bits,
			   unsigned int diff1_my,
			   unsigned int diff2_my,
			   int side_to_move,
			   int empties,
			   int alpha,
			   int beta,
			   int selectivity,
			   int *cutoff_score ) {
  (void) cand_sq;
  BitBoard opp_moves = bitboard_moves( child_opp_bits, child_my_bits );
  int opp_mob = non_iterative_popcount( opp_moves );

  if ( opp_mob != 1 )
    return FORCED_NONE;

  int cur_side = OPP( side_to_move );
  BitBoard cur_my_bits = child_opp_bits;
  BitBoard cur_opp_bits = child_my_bits;
  unsigned int cum_diff1 = diff1_my;
  unsigned int cum_diff2 = diff2_my;
  int cur_empties = empties - 1;
  int k = 1;
  int pass_legal = TRUE;

  while ( k < 16 && cur_empties > 0 ) {
    BitBoard cur_moves = bitboard_moves( cur_my_bits, cur_opp_bits );
    int cur_mob = non_iterative_popcount( cur_moves );

    if ( cur_mob == 1 ) {
      int sq = square_of_bit[FIRST_BIT( cur_moves )];
      BitBoard new_my_bits;
      int flipped = TestFlips_bitboard_to( sq, cur_my_bits, cur_opp_bits, &new_my_bits );
      if ( flipped == 0 )
	break;

      unsigned int d1, d2;
      end_hash_diff( new_my_bits, cur_my_bits, cur_side, sq, &d1, &d2 );
      cum_diff1 ^= d1;
      cum_diff2 ^= d2;

      BitBoard next_my = cur_opp_bits & ~new_my_bits;
      BitBoard next_opp = new_my_bits;
      cur_my_bits = next_my;
      cur_opp_bits = next_opp;
      cur_side = OPP( cur_side );
      cur_empties--;
      k++;
      pass_legal = TRUE;

      /* Edge stability check (PRUN-006): immediate geometric cutoff or refutation */
      if ( square_mask[sq] & BORDER_MASK ) {
	EdgeIndices edges;
	int s_edge = count_edge_stable_indexed( OPP( cur_side ), cur_opp_bits, cur_my_bits, &edges );
	int bound = 2 * s_edge - 64;
	if ( OPP( cur_side) == side_to_move ) {
	  /* side_to_move score is >= bound */
	  if ( bound >= beta ) {
	    *cutoff_score = bound;
	    return FORCED_CUTOFF;
	  }
	  if ( edges.bits != 0 && (cur_opp_bits & CENTRAL_MASK) != 0 ) {
	    int cnt = non_iterative_popcount( cur_opp_bits );
	    if ( 2 * cnt - 64 >= beta ) {
	      int s_full = count_stable_indexed( OPP( cur_side ), cur_opp_bits, cur_my_bits, &edges );
	      bound = 2 * s_full - 64;
	      if ( bound >= beta ) {
		*cutoff_score = bound;
		return FORCED_CUTOFF;
	      }
	    }
	  }
	}
	else {
	  /* Opponent score is >= bound => side_to_move score is <= -bound */
	  if ( -bound <= alpha ) {
	    return FORCED_REFUTED;
	  }
	  if ( edges.bits != 0 && (cur_opp_bits & CENTRAL_MASK) != 0 ) {
	    int cnt = non_iterative_popcount( cur_opp_bits );
	    if ( -(2 * cnt - 64) <= alpha ) {
	      int s_full = count_stable_indexed( OPP( cur_side ), cur_opp_bits, cur_my_bits, &edges );
	      bound = 2 * s_full - 64;
	      if ( -bound <= alpha ) {
		return FORCED_REFUTED;
	      }
	    }
	  }
	}
      }

      /* Probe Transposition Table at intermediate position P_k */
      hash1 ^= cum_diff1;
      hash2 ^= cum_diff2;
      prefetch_hash_endgame_key( hash2 );
      HashEntry g_entry;
      find_hash( &g_entry, ENDGAME_MODE );
      hash1 ^= cum_diff1;
      hash2 ^= cum_diff2;

      if ( (g_entry.flags & ENDGAME_SCORE) &&
	   (g_entry.draft >= cur_empties) &&
	   (g_entry.selectivity <= selectivity) ) {
	if ( cur_side == side_to_move ) {
	  /* Even ply: my turn (side_to_move) */
	  if ( (g_entry.flags & (LOWER_BOUND | EXACT_VALUE)) && (g_entry.eval >= beta) ) {
	    *cutoff_score = g_entry.eval;
	    return FORCED_CUTOFF;
	  }
	  if ( (g_entry.flags & (UPPER_BOUND | EXACT_VALUE)) && (g_entry.eval <= alpha) ) {
	    return FORCED_REFUTED;
	  }
	}
	else {
	  /* Odd ply: opponent turn (OPP(side_to_move)). Score is -eval */
	  if ( (g_entry.flags & (UPPER_BOUND | EXACT_VALUE)) && (-g_entry.eval >= beta) ) {
	    *cutoff_score = -g_entry.eval;
	    return FORCED_CUTOFF;
	  }
	  if ( (g_entry.flags & (LOWER_BOUND | EXACT_VALUE)) && (-g_entry.eval <= alpha) ) {
	    return FORCED_REFUTED;
	  }
	}
      }
    }
    else if ( cur_mob == 0 ) {
      if ( !pass_legal ) {
	/* Double pass: game over */
	int s0_discs = non_iterative_popcount( (cur_side == side_to_move) ? cur_my_bits : cur_opp_bits );
	int s1_discs = non_iterative_popcount( (cur_side == side_to_move) ? cur_opp_bits : cur_my_bits );
	int term_diff = s0_discs - s1_discs;
	int term_score = (term_diff > 0) ? (term_diff + cur_empties)
		       : (term_diff < 0) ? (term_diff - cur_empties) : 0;
	if ( term_score >= beta ) {
	  *cutoff_score = term_score;
	  return FORCED_CUTOFF;
	}
	if ( term_score <= alpha ) {
	  return FORCED_REFUTED;
	}
	break;
      }

      /* Single pass: switch side and continue */
      cum_diff1 ^= hash_flip_color1;
      cum_diff2 ^= hash_flip_color2;
      BitBoard temp = cur_my_bits;
      cur_my_bits = cur_opp_bits;
      cur_opp_bits = temp;
      cur_side = OPP( cur_side );
      k++;
      pass_legal = FALSE;
    }
    else {
      /* Branching point reached (cur_mob >= 2): corridor ends */
      break;
    }
  }

  return FORCED_NONE;
}

INLINE static int
end_probe_2ply_etc( int sq,
		    BitBoard child_my_bits,
		    BitBoard child_opp_bits,
		    unsigned int diff1_my,
		    unsigned int diff2_my,
		    int side_to_move,
		    int empties,
		    int alpha,
		    int beta,
		    int selectivity,
		    int *cutoff_score ) {
  BitBoard opp_moves = bitboard_moves( child_opp_bits, child_my_bits );
  int opp_mob = non_iterative_popcount( opp_moves );

  if ( opp_mob == 1 ) {
    int res = probe_forced_reply_cutoff( sq, child_my_bits, child_opp_bits,
					 diff1_my, diff2_my, side_to_move,
					 empties, alpha, beta, selectivity,
					 cutoff_score );
    if ( res == FORCED_CUTOFF )
      return ETC_2PLY_CUTOFF;
    if ( res == FORCED_REFUTED )
      return ETC_2PLY_REFUTED;
    return ETC_2PLY_NONE;
  }

  if ( opp_mob == 0 ) {
    BitBoard my_replies = bitboard_moves( child_my_bits, child_opp_bits );
    if ( my_replies == 0 ) {
      /* Terminal double pass: game over */
      int my_cnt = non_iterative_popcount( child_my_bits );
      int opp_cnt = non_iterative_popcount( child_opp_bits );
      int term_diff = my_cnt - opp_cnt;
      int rem_empties = empties - 1;
      int term_score = (term_diff > 0) ? (term_diff + rem_empties)
		     : (term_diff < 0) ? (term_diff - rem_empties) : 0;
      if ( term_score >= beta ) {
	*cutoff_score = term_score;
	return ETC_2PLY_CUTOFF;
      }
      if ( term_score <= alpha ) {
	return ETC_2PLY_REFUTED;
      }
      return ETC_2PLY_NONE;
    }
    else {
      /* Single pass: opponent passes, probe TT at post-pass state (mover to play) */
      unsigned int pass_d1 = diff1_my ^ hash_flip_color1;
      unsigned int pass_d2 = diff2_my ^ hash_flip_color2;
      hash1 ^= pass_d1;
      hash2 ^= pass_d2;
      prefetch_hash_endgame_key( hash2 );
      HashEntry p_entry;
      find_hash( &p_entry, ENDGAME_MODE );
      hash1 ^= pass_d1;
      hash2 ^= pass_d2;

      if ( (p_entry.flags & ENDGAME_SCORE) &&
	   (p_entry.draft >= empties - 1) &&
	   (p_entry.selectivity <= selectivity) ) {
	if ( (p_entry.flags & (LOWER_BOUND | EXACT_VALUE)) && (p_entry.eval >= beta) ) {
	  *cutoff_score = p_entry.eval;
	  return ETC_2PLY_CUTOFF;
	}
	if ( (p_entry.flags & (UPPER_BOUND | EXACT_VALUE)) && (p_entry.eval <= alpha) ) {
	  return ETC_2PLY_REFUTED;
	}
      }
      return ETC_2PLY_NONE;
    }
  }

  if ( empties < MIN_2PLY_ETC_DEPTH )
    return ETC_2PLY_NONE;

  if ( opp_mob < 1 || opp_mob > MAX_2PLY_ETC_MOB )
    return ETC_2PLY_NONE;

  int oppcol = OPP( side_to_move );
  BitBoard moves_bb = opp_moves;
  int all_ge_beta = TRUE;
  int min_g_eval = INFINITE_EVAL;

  while ( moves_bb != 0 ) {
    int bit = FIRST_BIT( moves_bb );
    moves_bb &= moves_bb - 1;
    int opp_sq = square_of_bit[bit];

    BitBoard opp_flips_new_bits;
    int flipped = TestFlips_bitboard_to( opp_sq, child_opp_bits, child_my_bits, &opp_flips_new_bits );
    if ( flipped == 0 ) {
      all_ge_beta = FALSE;
      continue;
    }

    unsigned int diff1_opp, diff2_opp;
    end_hash_diff( opp_flips_new_bits, child_opp_bits, oppcol, opp_sq, &diff1_opp, &diff2_opp );

    unsigned int g_diff1 = diff1_my ^ diff1_opp;
    unsigned int g_diff2 = diff2_my ^ diff2_opp;

    hash1 ^= g_diff1;
    hash2 ^= g_diff2;
    prefetch_hash_endgame_key( hash2 );
    HashEntry g_entry;
    find_hash( &g_entry, ENDGAME_MODE );
    hash1 ^= g_diff1;
    hash2 ^= g_diff2;

    int valid_entry = (g_entry.flags & ENDGAME_SCORE) &&
		      (g_entry.draft >= empties - 2) &&
		      (g_entry.selectivity <= selectivity);

    if ( valid_entry ) {
      /* Case 1: Refutation - Opponent reply holds my score <= alpha */
      if ( (g_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
	   (g_entry.eval <= alpha) ) {
	return ETC_2PLY_REFUTED;
      }

      /* Check if this reply guarantees score >= beta */
      if ( (g_entry.flags & (LOWER_BOUND | EXACT_VALUE)) &&
	   (g_entry.eval >= beta) ) {
	if ( g_entry.eval < min_g_eval )
	  min_g_eval = g_entry.eval;
      }
      else {
	all_ge_beta = FALSE;
      }
    }
    else {
      all_ge_beta = FALSE;
    }
  }

  /* Case 2: Immediate Fail-High Cutoff - All replies guarantee >= beta */
  if ( all_ge_beta && min_g_eval != INFINITE_EVAL ) {
    *cutoff_score = min_g_eval;
    return ETC_2PLY_CUTOFF;
  }

  return ETC_2PLY_NONE;
}



/*
  SYNC_BOARD_FROM_BITBOARDS
  Synchronize 100-cell array board, board_bits, piece_count, and
  pattern indices from bitboards for midgame heuristic pre-search.
*/

static void
sync_board_from_bitboards( BitBoard my_bits, BitBoard opp_bits, int side_to_move, int empties ) {
  BitBoard b_bits = (side_to_move == BLACKSQ ? my_bits : opp_bits);
  BitBoard w_bits = (side_to_move == BLACKSQ ? opp_bits : my_bits);
  int i, j;

  board_bits[BLACKSQ] = b_bits;
  board_bits[WHITESQ] = w_bits;
  board_bits[EMPTY] = ~(b_bits | w_bits);

  for ( i = 1; i <= 8; i++ ) {
    for ( j = 1; j <= 8; j++ ) {
      int pos = 10 * i + j;
      BitBoard mask = square_mask[pos];
      if ( b_bits & mask )
	board[pos] = BLACKSQ;
      else if ( w_bits & mask )
	board[pos] = WHITESQ;
      else
	board[pos] = EMPTY;
    }
  }

  disks_played = 60 - empties;
  int b_count = non_iterative_popcount( b_bits );
  int w_count = non_iterative_popcount( w_bits );
  piece_count[BLACKSQ][disks_played] = b_count;
  piece_count[WHITESQ][disks_played] = w_count;

  determine_pattern_indices();
}







/*
  UPDATE_BEST_LIST
*/

static void
update_best_list( int *best_list, int move, int best_list_index,
		  int *best_list_length ) {
  int i;

  if ( best_list_index < *best_list_length )
    for ( i = best_list_index; i >= 1; i-- )
      best_list[i] = best_list[i - 1];
  else {
    for ( i = 3; i >= 1; i-- )
      best_list[i] = best_list[i - 1];
    if ( *best_list_length < 4 )
      (*best_list_length)++;
  }
  best_list[0] = move;
}
/*
  PARALLEL ROOT SIBLINGS

  Young-brothers-wait: the first root move is searched on its own, and
  only once it has produced a bound are the remaining moves handed out
  to the pool.  Each of them gets a null window, which is what the
  sequential search would have given them anyway, and the vast majority
  fail low -- those the sequential loop can then skip outright.  A move
  that fails high is left alone here and re-searched in order by the
  sequential loop, so the score and the principal variation are still
  produced by exactly the same code as before.
*/

/* Remaining depth at or above which a node is worth splitting. */
#define PARALLEL_SPLIT_DEPTH         11

/* Dynamic nesting limits: allows helper threads on deep subtrees to split */
#ifndef MAX_SPLIT_NESTING
#define MAX_SPLIT_NESTING             4
#endif
#define SPLIT_NESTING_MARGIN          2

/* How many split levels deep this thread currently is */
static _Thread_local int split_nesting;
static _Thread_local BitBoard cached_flips[MAX_SEARCH_DEPTH + 1][100];
static _Thread_local int cached_flipped[MAX_SEARCH_DEPTH + 1][100];

/* Fast check on per-node search path: zero when no splits are live anywhere */
#define SPLIT_ABANDONED()  ((active_splits != 0) && split_abandoned())

/*
  SPLIT_ABANDONED
  Walks up the SplitPoint parent chain to check if any ancestor split
  point encountered a beta cutoff.
*/

static INLINE int
split_abandoned( void ) {
  const SplitPoint *sp;

  for ( sp = current_split_point; sp != NULL; sp = sp->parent ) {
    if ( atomic_load_explicit( &sp->cutoff_occurred, memory_order_relaxed ) )
      return TRUE;
  }

  return FALSE;
}


/*
  END_SEARCH_SIBLING
  Worker callback: executes search for a single stolen sibling move locklessly.
*/

static void
end_search_sibling( SplitPoint *sp, int idx ) {
  int move = sp->moves[idx];
  BitBoard my_bits = sp->my_bits;
  BitBoard opp_bits = sp->opp_bits;
  BitBoard new_my_bits, new_opp_bits;
  int child_selective_cutoff = FALSE;
  int score;

  if ( split_abandoned() || is_panic_abort() || force_return )
    return;

  int cur_alpha = atomic_load_explicit( &sp->alpha, memory_order_acquire );
  if ( cur_alpha >= sp->beta ) {
    atomic_store_explicit( &sp->cutoff_occurred, true, memory_order_release );
    return;
  }
  sp->searched_alpha[idx] = cur_alpha;

  // Restore thread-local search state for this sibling
  hash1 = sp->sp_hash1;
  hash2 = sp->sp_hash2;
  region_parity = sp->sp_region_parity;
  memcpy( end_move_list, sp->sp_end_move_list, sizeof( end_move_list ) );

  int flipped = TestFlips_wrapper( move, my_bits, opp_bits );
  if ( flipped == 0 )
    return;

  new_my_bits = bb_flips;
  FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

  if ( sp->level + 1 <= MAX_SEARCH_DEPTH ) {
    tls.stable_discs[BLACKSQ][sp->level + 1] = sp->saved_stable[BLACKSQ];
    tls.stable_discs[WHITESQ][sp->level + 1] = sp->saved_stable[WHITESQ];
  }

  unsigned int diff1, diff2;
  int pred, succ;
  end_make_move( move, new_my_bits, my_bits, sp->side_to_move, &diff1, &diff2, &pred, &succ );

  int child_disc_diff = -sp->disc_diff - 2 * flipped - 1;

  split_nesting++;
  disks_played = 60 - sp->empties;
  score = -end_search_nws( new_opp_bits, new_my_bits,
                           -(cur_alpha + 1),
                           OPP( sp->side_to_move ),
                           sp->empties - 1,
                           child_disc_diff,
                           TRUE,
                           sp->level + 1,
                           sp->selectivity,
                           &child_selective_cutoff );
  split_nesting--;

  end_unmake_move( move, diff1, diff2, pred, succ );

  if ( !split_abandoned() && !is_panic_abort() && !force_return && abs( score ) < 20000 ) {
    sp->score[idx] = score;
    sp->cutoff[idx] = child_selective_cutoff;
    sp->valid[idx] = TRUE;
    if ( score >= sp->beta ) {
      atomic_store_explicit( &sp->cutoff_occurred, true, memory_order_release );
    } else if ( score > cur_alpha ) {
      int old_a = cur_alpha;
      while ( score > old_a &&
              !atomic_compare_exchange_weak_explicit( &sp->alpha, &old_a, score,
                                                      memory_order_release, memory_order_relaxed ) ) {}
    }
  }
}


/*
  DISPATCH_SIBLINGS
  Pure bitboard lock-free sibling dispatch.
  Retrieves SplitPoint from per-thread static buffer, publishes to thread slot,
  and executes lock-free work-stealing across available cores.
*/

static int
dispatch_siblings( BitBoard my_bits, BitBoard opp_bits,
		   int side_to_move, int level, int empties,
		   int disc_diff, int alpha, int beta,
		   int selectivity, int searched_move,
		   const int *best_list, int best_list_length,
		   int pre_search_done,
		   int *proven, int *proven_score, int *proven_cutoff,
		   int *cutoff_move, int *cutoff_score, int *cutoff_selective ) {
  SplitPoint *sp = ybwc_get_split_point( thread_id, split_nesting );
  int count = 0;
  int used[100];
  int i, sq;

  for ( i = 0; i < 100; i++ )
    used[i] = FALSE;
  used[searched_move] = TRUE;

  /* 1. First priority: remaining moves in best_list (hash moves) */
  for ( i = 0; i < best_list_length; i++ ) {
    sq = best_list[i];
    if ( !used[sq] && !((my_bits | opp_bits) & square_mask[sq]) &&
	 (TestFlips_wrapper( sq, my_bits, opp_bits ) > 0) ) {
      used[sq] = TRUE;
      if ( count < MAX_ROOT_MOVES )
	sp->moves[count++] = sq;
    }
  }

  /* 2. Second priority: remaining legal moves */
  if ( pre_search_done ) {
    int rem_moves[MAX_ROOT_MOVES];
    int rem_count = 0;
    for ( i = 0; i < move_count[disks_played]; i++ ) {
      sq = move_list[disks_played][i];
      if ( !used[sq] ) {
	rem_moves[rem_count++] = sq;
	used[sq] = TRUE;
      }
    }
    for ( i = 0; i < rem_count; i++ ) {
      int best_idx = i;
      int best_ev = evals[disks_played][rem_moves[i]];
      int j;
      for ( j = i + 1; j < rem_count; j++ ) {
	if ( evals[disks_played][rem_moves[j]] > best_ev ) {
	  best_idx = j;
	  best_ev = evals[disks_played][rem_moves[j]];
	}
      }
      if ( best_idx != i ) {
	int tmp = rem_moves[i];
	rem_moves[i] = rem_moves[best_idx];
	rem_moves[best_idx] = tmp;
      }
      if ( count < MAX_ROOT_MOVES )
	sp->moves[count++] = rem_moves[i];
    }
  } else {
    for ( i = 0; i < MOVE_ORDER_SIZE; i++ ) {
      sq = sorted_move_order[disks_played][i];
      if ( !used[sq] && !((my_bits | opp_bits) & square_mask[sq]) &&
	   (TestFlips_wrapper( sq, my_bits, opp_bits ) > 0) ) {
	used[sq] = TRUE;
	if ( count < MAX_ROOT_MOVES )
	  sp->moves[count++] = sq;
      }
    }
  }

  if ( count == 0 )
    return FALSE;

  sp->move_count = count;
  for ( i = 0; i < count; i++ ) {
    sp->score[i] = 0;
    sp->cutoff[i] = FALSE;
    sp->valid[i] = FALSE;
    sp->searched_alpha[i] = alpha;
  }

  sp->my_bits = my_bits;
  sp->opp_bits = opp_bits;
  sp->side_to_move = side_to_move;
  sp->empties = empties;
  sp->disc_diff = disc_diff;
  sp->level = level;
  sp->selectivity = selectivity;
  atomic_init( &sp->alpha, alpha );
  sp->beta = beta;
  sp->sp_hash1 = hash1;
  sp->sp_hash2 = hash2;
  sp->sp_region_parity = region_parity;
  memcpy( sp->sp_end_move_list, end_move_list, sizeof( end_move_list ) );

  if ( level <= MAX_SEARCH_DEPTH ) {
    sp->saved_stable[BLACKSQ] = tls.stable_discs[BLACKSQ][level];
    sp->saved_stable[WHITESQ] = tls.stable_discs[WHITESQ][level];
  } else {
    sp->saved_stable[BLACKSQ] = 0;
    sp->saved_stable[WHITESQ] = 0;
  }

  sp->search_fn = end_search_sibling;
  sp->parent = current_split_point;

  (void) __sync_fetch_and_add( &active_splits, 1 );
  ybwc_split( sp );
  (void) __sync_fetch_and_sub( &active_splits, 1 );

  int final_alpha = atomic_load_explicit( &sp->alpha, memory_order_acquire );
  int found_cutoff = FALSE;
  for ( i = 0; i < count; i++ ) {
    if ( sp->valid[i] ) {
      if ( sp->score[i] >= beta ) {
        *cutoff_move = sp->moves[i];
        *cutoff_score = sp->score[i];
        *cutoff_selective = sp->cutoff[i];
        found_cutoff = TRUE;
        break;
      }
      if ( (sp->score[i] <= sp->searched_alpha[i]) && (sp->score[i] <= final_alpha) ) {
        proven[sp->moves[i]] = TRUE;
        proven_score[sp->moves[i]] = sp->score[i];
        proven_cutoff[sp->moves[i]] = sp->cutoff[i];
        if ( sp->cutoff[i] )
          *cutoff_selective = TRUE;
      }
    }
  }
  return found_cutoff;
}

/*
  END_ORDER_MOVES_PRESEARCH
  Move ordering helper for deep endgame search.
  Performs 1-ply/2-ply ETC screening followed by pure bitboard static
  move ordering to order candidate moves.
*/

static int
end_order_moves_presearch( int level,
			   int empties,
			   int side_to_move,
			   BitBoard my_bits,
			   BitBoard opp_bits,
			   int beta,
			   int curr_alpha,
			   int selectivity,
			   int use_hash,
			   const int *best_list,
			   int best_list_length,
			   int can_split,
			   const int *proven,
			   const int *proven_score,
			   int *etc_tried_ptr,
			   int *etc_cutoff_score ) {
  int i, j;
  int move;
  int etc_demoted[100];
  BitBoard new_opp_bits;

  for ( i = 0; i < 100; i++ )
    etc_demoted[i] = FALSE;

  typedef struct {
    int sq;
    int flipped;
    BitBoard flips;
    BitBoard child_my_bits;
  } CandidateMove;

  CandidateMove candidates[64];
  int candidate_count = 0;

  BitBoard legal_moves = generate_all_c( my_bits, opp_bits );
  if ( *etc_tried_ptr != 0 )
    legal_moves &= ~square_mask[*etc_tried_ptr];
  for ( j = 0; j < best_list_length; j++ ) {
    legal_moves &= ~square_mask[best_list[j]];
  }

  for ( int shallow_index = 0; shallow_index < MOVE_ORDER_SIZE; shallow_index++ ) {
    if ( legal_moves == 0 )
      break;
    move = sorted_move_order[disks_played][shallow_index];
    if ( legal_moves & square_mask[move] ) {
      legal_moves &= ~square_mask[move];
      int flipped = TestFlips_wrapper( move, my_bits, opp_bits );
      if ( flipped > 0 ) {
	candidates[candidate_count].sq = move;
	candidates[candidate_count].flipped = flipped;
	candidates[candidate_count].flips = bb_flips;
	candidates[candidate_count].child_my_bits = my_bits | square_mask[move] | bb_flips;
	if ( level <= MAX_SEARCH_DEPTH ) {
	  cached_flipped[level][move] = flipped;
	  cached_flips[level][move] = bb_flips;
	}
	candidate_count++;
      }
    }
  }

  /* Pass 1: Lightweight 1-ply ETC scan for candidate moves before static ordering */
  if ( use_hash ) {
    for ( i = 0; i < candidate_count; i++ ) {
      move = candidates[i].sq;
      BitBoard child_my_bits = candidates[i].child_my_bits;

      if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) )
	continue;

      unsigned int diff1, diff2;
      end_hash_diff( child_my_bits, my_bits, side_to_move, move, &diff1, &diff2 );
      hash1 ^= diff1;
      hash2 ^= diff2;
      prefetch_hash_endgame_key( hash2 );
      HashEntry etc_entry;
      find_hash( &etc_entry, ENDGAME_MODE );
      hash1 ^= diff1;
      hash2 ^= diff2;

      int etc1_demoted = FALSE;
      if ( (etc_entry.flags & ENDGAME_SCORE) &&
	   (etc_entry.draft >= empties - 1) &&
	   (etc_entry.selectivity <= selectivity) ) {
	if ( (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
	     (etc_entry.eval <= -beta) ) {
	  *etc_tried_ptr = move;
	  *etc_cutoff_score = -etc_entry.eval;
	  return TRUE;
	}
	else if ( (etc_entry.flags & (LOWER_BOUND | EXACT_VALUE)) &&
		  (etc_entry.eval >= -curr_alpha) ) {
	  etc_demoted[move] = TRUE;
	  etc1_demoted = TRUE;
	}
      }

      if ( !etc1_demoted ) {
	int cutoff_score;
	int etc2_res = end_probe_2ply_etc( move, child_my_bits, opp_bits & ~child_my_bits,
					   diff1, diff2, side_to_move, empties,
					   curr_alpha, beta, selectivity, &cutoff_score );
	if ( etc2_res == ETC_2PLY_CUTOFF ) {
	  *etc_tried_ptr = move;
	  *etc_cutoff_score = cutoff_score;
	  return TRUE;
	}
	else if ( etc2_res == ETC_2PLY_REFUTED ) {
	  etc_demoted[move] = TRUE;
	}
      }
    }
  }

  /* Pass 2: Pure bitboard static move ordering across all candidate moves */
  int refuted_moves[64];
  int refuted_count = 0;

  BitBoard stable_both = 0;
  if ( level <= MAX_SEARCH_DEPTH )
    stable_both = tls.stable_discs[BLACKSQ][level] | tls.stable_discs[WHITESQ][level];
  BitBoard dead = find_dead_squares( ~(my_bits | opp_bits), stable_both );
  unsigned int effective_parity = region_parity ^ dead_quadrant_parity( dead );

  for ( i = 0; i < candidate_count; i++ ) {
    move = candidates[i].sq;
    if ( move == *etc_tried_ptr )
      continue;

    if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) ) {
      evals[disks_played][move] = -INFINITE_EVAL;
      move_list[disks_played][move_count[disks_played]++] = move;
      continue;
    }
    if ( etc_demoted[move] ) {
      if ( beta == curr_alpha + 1 ) {
	refuted_moves[refuted_count++] = move;
	continue;
      }
      evals[disks_played][move] = -INFINITE_EVAL;
      move_list[disks_played][move_count[disks_played]++] = move;
      continue;
    }

    BitBoard child_my_bits = candidates[i].child_my_bits;
    FULL_ANDNOT( new_opp_bits, opp_bits, candidates[i].flips );

    EdgeIndices edges;
    int my_edge_stable = count_edge_stable_indexed( side_to_move, child_my_bits, new_opp_bits, &edges );

    int lower_bound = 2 * my_edge_stable - 64;
    if ( lower_bound >= beta ) {
      *etc_tried_ptr = move;
      *etc_cutoff_score = lower_bound;
      return TRUE;
    }
    if ( edges.bits != 0 && (child_my_bits & CENTRAL_MASK) != 0 ) {
      int my_cnt = non_iterative_popcount( child_my_bits );
      if ( 2 * my_cnt - 64 >= beta ) {
	int s_full = count_stable_indexed( side_to_move, child_my_bits, new_opp_bits, &edges );
	lower_bound = 2 * s_full - 64;
	if ( lower_bound >= beta ) {
	  *etc_tried_ptr = move;
	  *etc_cutoff_score = lower_bound;
	  return TRUE;
	}
      }
    }

    BitBoard opp_moves = generate_all_c( new_opp_bits, child_my_bits );
    int raw_opp_mob = non_iterative_popcount( opp_moves );

    if ( raw_opp_mob == 0 ) {
      BitBoard my_replies = generate_all_c( child_my_bits, new_opp_bits );
      if ( my_replies == 0 ) {
	int my_cnt = non_iterative_popcount( child_my_bits );
	int opp_cnt = non_iterative_popcount( new_opp_bits );
	int term_diff = my_cnt - opp_cnt;
	int rem_empties = empties - 1;
	int term_score = (term_diff > 0) ? (term_diff + rem_empties)
		       : (term_diff < 0) ? (term_diff - rem_empties) : 0;
	if ( term_score >= beta ) {
	  *etc_tried_ptr = move;
	  *etc_cutoff_score = term_score;
	  return TRUE;
	}
	else if ( term_score <= curr_alpha ) {
	  if ( beta == curr_alpha + 1 ) {
	    refuted_moves[refuted_count++] = move;
	    continue;
	  }
	  evals[disks_played][move] = -INFINITE_EVAL;
	  move_list[disks_played][move_count[disks_played]++] = move;
	  continue;
	}
      }
    }

    int opp_corner_moves = non_iterative_popcount( opp_moves & 0x8100000000000081ull );

    /* ORDR-008: Penalize open C-square move if adjacent corner is empty and opponent gets immediate reply */
    int c_square_penalty = 0;
    if ( empties >= 15 && (square_mask[move] & 0x4281000000008142ull) != 0 ) {
      int bit = bit_position[move];
      int corner_bit = (bit == 1 || bit == 8) ? 0 :
                       (bit == 6 || bit == 15) ? 7 :
                       (bit == 48 || bit == 57) ? 56 : 63;
      if ( ((child_my_bits | new_opp_bits) & (1ull << corner_bit)) == 0 &&
           (opp_moves & (1ull << corner_bit)) != 0 ) {
        c_square_penalty = 128;
      }
    }

    int quadrant_parity = (quadrant_mask[move] & effective_parity) != 0;
    int move_score = end_pattern_evaluate( child_my_bits, new_opp_bits ) -
		     128 * (raw_opp_mob + opp_corner_moves) -
		     c_square_penalty +
		     (raw_opp_mob == 0 ? 4096 : 0) +
		     (quadrant_parity ? REGION_PARITY_BONUS : 0);

    evals[disks_played][move] = move_score;
    move_list[disks_played][move_count[disks_played]++] = move;
  }

  if ( beta == curr_alpha + 1 && move_count[disks_played] == 0 && best_list_length == 0 && refuted_count > 0 ) {
    int fb_move = refuted_moves[0];
    evals[disks_played][fb_move] = -INFINITE_EVAL;
    move_list[disks_played][move_count[disks_played]++] = fb_move;
  }

  return FALSE;
}


/*
  END_PRESEARCH_AB
  Pure-bitboard alpha-beta pre-search helper using 4-pattern endgame LTR model (ARCH-006).
  Runs iterative deepening before exact endgame solve to seed Move 0 throughout the top 10 plies.
*/

static int presearch_nodes = 0;
static int presearch_budget = 0;
static int presearch_aborted = FALSE;

static int
end_presearch_ab( BitBoard my_bits, BitBoard opp_bits,
		  int side_to_move, int depth, int empties,
		  int alpha, int beta, int level, int *best_move ) {
  if ( presearch_aborted || is_panic_abort() || force_return )
    return alpha;

  INCREMENT_COUNTER( nodes );
  if ( ++presearch_nodes > presearch_budget ) {
    presearch_aborted = TRUE;
    return alpha;
  }
  if ( best_move != NULL )
    *best_move = 0;

  if ( depth <= 0 ) {
    int val = end_pattern_evaluate( my_bits, opp_bits );
    if ( (region_parity != 0) && (empties & 1) )
      val += REGION_PARITY_BONUS;
    return val;
  }

  /* Move Generation & Pass Handling */
  BitBoard moves = generate_all_c( my_bits, opp_bits );
  if ( moves == 0 ) {
    BitBoard opp_moves = generate_all_c( opp_bits, my_bits );
    if ( opp_moves == 0 ) {
      int disc_diff = non_iterative_popcount( my_bits ) - non_iterative_popcount( opp_bits );
      int final_score = (disc_diff > 0) ? (disc_diff + empties) :
			((disc_diff < 0) ? (disc_diff - empties) : 0);
      return final_score * 128;
    }
    hash1 ^= hash_flip_color1;
    hash2 ^= hash_flip_color2;
    int val = -end_presearch_ab( opp_bits, my_bits, OPP( side_to_move ),
				 depth, empties, -beta, -alpha, level + 1, NULL );
    hash1 ^= hash_flip_color1;
    hash2 ^= hash_flip_color2;
    return val;
  }

  /* Transposition Table Probing */
  HashEntry entry;
  find_hash( &entry, ENDGAME_MODE );
  int hash_move = 0;
  if ( entry.draft != NO_HASH_MOVE && bb_valid_move( entry.move[0], my_bits, opp_bits ) ) {
    hash_move = entry.move[0];
  }

  if ( level > 0 && entry.draft != NO_HASH_MOVE && entry.draft >= depth ) {
    int tt_val;
    if ( (entry.flags & MIDGAME_SCORE) && !(entry.flags & HEURISTIC_PRESEARCH_MOVE) ) {
      tt_val = entry.eval;
    } else {
      tt_val = entry.eval * 128;
    }
    if ( entry.flags & EXACT_VALUE ) {
      if ( best_move != NULL ) *best_move = hash_move;
      return tt_val;
    }
    if ( (entry.flags & LOWER_BOUND) && tt_val >= beta ) {
      if ( best_move != NULL ) *best_move = hash_move;
      return tt_val;
    }
    if ( (entry.flags & UPPER_BOUND) && tt_val <= alpha ) {
      if ( best_move != NULL ) *best_move = hash_move;
      return tt_val;
    }
  }

  /* Move Ordering: Move 0 = hash_move, remaining sorted by quadrant parity & corner/X */
  int moves_arr[64];
  int scores_arr[64];
  int n_moves = 0;

  if ( hash_move != 0 ) {
    moves_arr[n_moves] = hash_move;
    scores_arr[n_moves] = 100000;
    n_moves++;
  }

  BitBoard rem_moves = moves;
  while ( rem_moves != 0 ) {
    int bit = FIRST_BIT( rem_moves );
    rem_moves &= rem_moves - 1;
    int sq = square_of_bit[bit];
    if ( sq == hash_move )
      continue;

    BitBoard new_my_bits;
    TestFlips_bitboard_to( sq, my_bits, opp_bits, &new_my_bits );
    BitBoard flipped = new_my_bits & ~my_bits & ~square_mask[sq];
    BitBoard new_my = new_my_bits;
    BitBoard new_opp = opp_bits ^ flipped;
    BitBoard opp_replies = generate_all_c( new_opp, new_my );
    int opp_mob = non_iterative_popcount( opp_replies );
    int opp_corners = non_iterative_popcount( opp_replies & 0x8100000000000081ull );

    int score = - 128 * opp_mob - 256 * opp_corners;
    if ( opp_mob == 0 ) score += 1024;
    if ( quadrant_mask[sq] & region_parity ) score += 128;
    BitBoard bb = square_mask[sq];
    if ( bb & CORNER_MASK ) score += 512;
    else if ( bb & 0x0042000000004200ull ) score -= 256;

    moves_arr[n_moves] = sq;
    scores_arr[n_moves] = score;
    n_moves++;
  }

  for ( int i = 1; i < n_moves; i++ ) {
    int m = moves_arr[i];
    int s = scores_arr[i];
    int j = i - 1;
    while ( j >= 0 && scores_arr[j] < s ) {
      moves_arr[j + 1] = moves_arr[j];
      scores_arr[j + 1] = scores_arr[j];
      j--;
    }
    moves_arr[j + 1] = m;
    scores_arr[j + 1] = s;
  }

  /* Search Loop */
  int orig_alpha = alpha;
  int best_val = -INFINITE_EVAL;
  int best_sq = moves_arr[0];

  for ( int i = 0; i < n_moves; i++ ) {
    int sq = moves_arr[i];
    BitBoard new_my_bits;
    TestFlips_bitboard_to( sq, my_bits, opp_bits, &new_my_bits );

    unsigned int diff1, diff2;
    end_hash_diff( new_my_bits, my_bits, side_to_move, sq, &diff1, &diff2 );
    hash1 ^= diff1;
    hash2 ^= diff2;
    region_parity ^= quadrant_mask[sq];

    int child_best = 0;
    int val;
    if ( i == 0 ) {
      val = -end_presearch_ab( opp_bits & ~new_my_bits,
			       new_my_bits,
			       OPP( side_to_move ),
			       depth - 1,
			       empties - 1,
			       -beta,
			       -alpha,
			       level + 1,
			       &child_best );
    } else {
      val = -end_presearch_ab( opp_bits & ~new_my_bits,
			       new_my_bits,
			       OPP( side_to_move ),
			       depth - 1,
			       empties - 1,
			       -alpha - 1,
			       -alpha,
			       level + 1,
			       &child_best );
      if ( val > alpha && val < beta ) {
	val = -end_presearch_ab( opp_bits & ~new_my_bits,
				 new_my_bits,
				 OPP( side_to_move ),
				 depth - 1,
				 empties - 1,
				 -beta,
				 -alpha,
				 level + 1,
				 &child_best );
      }
    }

    region_parity ^= quadrant_mask[sq];
    hash1 ^= diff1;
    hash2 ^= diff2;

    if ( presearch_aborted || is_panic_abort() || force_return )
      break;

    if ( val > best_val ) {
      best_val = val;
      best_sq = sq;
      if ( val > alpha ) {
	alpha = val;
	if ( alpha >= beta )
	  break;
      }
    }
  }

  if ( presearch_aborted || is_panic_abort() || force_return )
    return alpha;

  /* TT Storage */
  int flags;
  int score_to_store;
  if ( best_val >= beta ) {
    flags = LOWER_BOUND;
    score_to_store = (best_val >= 0) ? (best_val / 128) : ((best_val - 127) / 128);
  }
  else if ( best_val > orig_alpha ) {
    flags = EXACT_VALUE;
    score_to_store = (best_val >= 0) ? ((best_val + 64) / 128) : ((best_val - 64) / 128);
  }
  else {
    flags = UPPER_BOUND;
    score_to_store = (best_val >= 0) ? ((best_val + 127) / 128) : (best_val / 128);
  }

  int best_list[4];
  best_list[0] = best_sq;
  best_list[1] = 0;
  best_list[2] = 0;
  best_list[3] = 0;

  if ( !presearch_aborted ) {
    add_hash_extended( ENDGAME_MODE, score_to_store, best_list,
		       flags | MIDGAME_SCORE | HEURISTIC_PRESEARCH_MOVE, depth, 0 );
  }

  if ( best_move != NULL )
    *best_move = best_sq;

  return best_val;
}


/*
  END_SEARCH_NWS
  Dedicated Null-Window Search (NWS) core.
  Optimized for Cut and All nodes (beta == alpha + 1).
  Bypasses PV tracking, re-searches, and UI sweep updates.
*/

static int
end_search_nws( BitBoard my_bits,
		BitBoard opp_bits,
		int alpha,
		int side_to_move,
		int empties,
		int disc_diff,
		int pass_legal,
		int level,
		int selectivity,
		int *selective_cutoff ) {
  int beta = alpha + 1;
  int oppcol = OPP( side_to_move );
  int use_hash;
  int hash_hit = FALSE;
  HashEntry entry;

  *selective_cutoff = FALSE;

  if ( SPLIT_ABANDONED() )
    return SEARCH_ABORT;

  /* 1. Terminal leaf dispatch (no PV tracking needed in NWS) */
  if ( empties <= LOW_LEVEL_DEPTH ) {
    return solve_parity( my_bits, opp_bits, alpha, beta, side_to_move,
			 empties, disc_diff, pass_legal, level );
  }

  INCREMENT_COUNTER( nodes );

  /* 2. Symmetric stability bounds check */
#if USE_STABILITY
  if ( level <= MAX_SEARCH_DEPTH ) {
    if ( tls.stable_discs[oppcol][level] != 0 ) {
      int s = non_iterative_popcount( tls.stable_discs[oppcol][level] );
      int upper_bound = 64 - 2 * s;
      if ( upper_bound <= alpha ) {
        return alpha;
      }
    }
    if ( tls.stable_discs[side_to_move][level] != 0 ) {
      int s = non_iterative_popcount( tls.stable_discs[side_to_move][level] );
      int lower_bound = 2 * s - 64;
      if ( lower_bound >= beta ) {
        return lower_bound;
      }
    }
  }

  if ( (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    int min_upper = 64 - 2 * opp_cnt;
    if ( min_upper <= alpha ) {
      EdgeIndices edges;
      int s_edge = count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      if ( level <= MAX_SEARCH_DEPTH )
        tls.stable_discs[oppcol][level] |= edges.bits;
      int upper_bound = 64 - 2 * s_edge;
      if ( upper_bound <= alpha ) {
        return alpha;
      }
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
        int s_full = count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        if ( level <= MAX_SEARCH_DEPTH )
          tls.stable_discs[oppcol][level] |= (oppcol == BLACKSQ ? last_black_stable : last_white_stable);
        upper_bound = 64 - 2 * s_full;
        if ( upper_bound <= alpha ) {
          return alpha;
        }
      }
    }
  }

  if ( (my_bits & BORDER_MASK) != 0 ) {
    int my_cnt = non_iterative_popcount( my_bits );
    int max_lower = 2 * my_cnt - 64;
    if ( max_lower >= beta ) {
      EdgeIndices edges;
      int s_edge = count_edge_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
      if ( level <= MAX_SEARCH_DEPTH )
        tls.stable_discs[side_to_move][level] |= edges.bits;
      int lower_bound = 2 * s_edge - 64;
      if ( lower_bound >= beta ) {
        return lower_bound;
      }
      if ( edges.bits != 0 && (my_bits & CENTRAL_MASK) != 0 ) {
        int s_full = count_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
        if ( level <= MAX_SEARCH_DEPTH )
          tls.stable_discs[side_to_move][level] |= (side_to_move == BLACKSQ ? last_black_stable : last_white_stable);
        lower_bound = 2 * s_full - 64;
        if ( lower_bound >= beta ) {
          return lower_bound;
        }
      }
    }
  }
#endif

  /* 3. Transposition table probing */
  use_hash = USE_HASH_TABLE;

  if ( use_hash ) {
    find_hash( &entry, ENDGAME_MODE );
    if ( (entry.draft == empties) &&
	 (entry.selectivity <= selectivity) &&
	 bb_valid_move( entry.move[0], my_bits, opp_bits ) &&
	 (entry.flags & ENDGAME_SCORE) &&
	 ((entry.flags & EXACT_VALUE) ||
	  ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	  ((entry.flags & UPPER_BOUND) && entry.eval <= alpha)) ) {
      if ( entry.selectivity > 0 )
	*selective_cutoff = TRUE;
      return entry.eval;
    }

    hash_hit = (entry.draft != NO_HASH_MOVE) &&
	       bb_valid_move( entry.move[0], my_bits, opp_bits ) &&
	       ((entry.flags & ENDGAME_SCORE) || (entry.flags & HEURISTIC_PRESEARCH_MOVE));
  }

  /* 4. Setup and MPC */
  int i;
  int move;
  int move_index;
  int first;
  int pre_search_done, etc_tried;
  int best_list_index, best_list_length;
  int best_list[8];
  int proven[100], proven_score[100], proven_cutoff[100];
  int siblings_dispatched = FALSE;
  int can_split;
  int best;
  int curr_val;
  int saved_disks_played = disks_played;

  disks_played = 60 - empties;

  if ( level <= MAX_SEARCH_DEPTH ) {
    memset( cached_flipped[level], 0, sizeof(cached_flipped[level]) );
  }

  if ( USE_MPC && (level > 2) && (selectivity > 0) ) {
    int cut;
    sync_board_from_bitboards( my_bits, opp_bits, side_to_move, empties );
    for ( cut = 0; cut < use_end_cut[disks_played]; cut++ ) {
      int shallow_remains = end_mpc_depth[disks_played][cut];
      int mpc_bias = ceil( end_mean[disks_played][shallow_remains] * 128.0 );
      int mpc_window = ceil( end_sigma[disks_played][shallow_remains] *
			     end_percentile[selectivity] * 128.0 );
      int beta_bound = 128 * beta + mpc_bias + mpc_window;
      int alpha_bound = 128 * alpha + mpc_bias - mpc_window;
      int shallow_val =
	tree_search( level, level + shallow_remains, side_to_move,
		     alpha_bound, beta_bound, use_hash, FALSE, pass_legal );
      if ( shallow_val >= beta_bound ) {
	if ( use_hash )
	  add_hash( ENDGAME_MODE, alpha, 0,
		    ENDGAME_SCORE | LOWER_BOUND, empties, selectivity );
	*selective_cutoff = TRUE;
	disks_played = saved_disks_played;
	return beta;
      }
      if ( shallow_val <= alpha_bound ) {
	if ( use_hash )
	  add_hash( ENDGAME_MODE, beta, 0,
		    ENDGAME_SCORE | UPPER_BOUND, empties, selectivity );
	*selective_cutoff = TRUE;
	disks_played = saved_disks_played;
	return alpha;
      }
    }
  }

  first = TRUE;
  can_split = FALSE;
  best = -INFINITE_EVAL;
  pre_search_done = FALSE;
  etc_tried = 0;

  /* Initialize move list and check hash table moves */
  move_count[disks_played] = 0;
  best_list_length = 0;
  for ( i = 0; i < 8; i++ )
    best_list[i] = 0;

  if ( hash_hit ) {
    best_list[0] = entry.move[0];
    best_list_length = 1;
  }

  /* 5. NWS Move loop */
  for ( move_index = 0, best_list_index = 0; TRUE;
	move_index++, best_list_index++ ) {
    int child_selective_cutoff;
    BitBoard new_my_bits;
    BitBoard new_opp_bits;

    if ( best_list_index < best_list_length ) {
      move = best_list[best_list_index];
      move_count[disks_played]++;
    }
    else {
      if ( !pre_search_done ) {
	int etc_cutoff_score = 0;
	if ( end_order_moves_presearch( level, empties, side_to_move,
					my_bits, opp_bits, beta, alpha,
					selectivity, use_hash,
					best_list, best_list_length,
					can_split, proven, proven_score,
					&etc_tried,
					&etc_cutoff_score ) ) {
	  best_list[0] = etc_tried;
	  if ( selectivity > 0 )
	    *selective_cutoff = TRUE;
	  if ( use_hash )
	    add_hash_extended( ENDGAME_MODE, etc_cutoff_score, best_list,
			       ENDGAME_SCORE | LOWER_BOUND, empties,
			       *selective_cutoff ? selectivity : 0 );
	  disks_played = saved_disks_played;
	  return etc_cutoff_score;
	}
	pre_search_done = TRUE;
      }

      if ( move_index == move_count[disks_played] )
	break;
      move = select_move( move_index, move_count[disks_played] );
    }

    double node_val = counter_value( &nodes );
    if ( node_val - last_panic_check >= EVENT_CHECK_INTERVAL ) {
      last_panic_check = node_val;
      check_panic_abort();
      if ( echo )
	display_buffers();
      handle_event( TRUE, FALSE, TRUE );
      if ( is_panic_abort() || force_return ) {
	disks_played = saved_disks_played;
	return SEARCH_ABORT;
      }
    }

    unsigned int diff1, diff2;
    int pred, succ;
    int flipped;
    if ( level <= MAX_SEARCH_DEPTH && cached_flipped[level][move] > 0 ) {
      flipped = cached_flipped[level][move];
      bb_flips = cached_flips[level][move];
    }
    else {
      flipped = TestFlips_wrapper( move, my_bits, opp_bits );
    }
    new_my_bits = bb_flips;
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

    end_make_move( move, new_my_bits, my_bits, side_to_move, &diff1, &diff2, &pred, &succ );

    if ( level + 1 <= MAX_SEARCH_DEPTH ) {
      tls.stable_discs[BLACKSQ][level + 1] = tls.stable_discs[BLACKSQ][level];
      tls.stable_discs[WHITESQ][level + 1] = tls.stable_discs[WHITESQ][level];
    }

    int new_disc_diff = -disc_diff - 2 * flipped - 1;

    if ( can_split && proven[move] && (proven_score[move] <= alpha) ) {
      curr_val = proven_score[move];
      child_selective_cutoff = proven_cutoff[move];
    }
    else {
      curr_val = -end_search_nws( new_opp_bits, new_my_bits,
				  -beta, OPP( side_to_move ),
				  empties - 1, new_disc_diff, TRUE, level + 1,
				  selectivity, &child_selective_cutoff );
    }

    end_unmake_move( move, diff1, diff2, pred, succ );

    if ( abs( curr_val ) >= 20000 || is_panic_abort() || force_return || SPLIT_ABANDONED() ) {
      disks_played = saved_disks_played;
      return SEARCH_ABORT;
    }

    /* Cutoff check in NWS: any score >= beta causes an immediate cutoff */
    if ( curr_val >= beta ) {
      *selective_cutoff = child_selective_cutoff;
      update_best_list( best_list, move, best_list_index, &best_list_length );
      if ( use_hash )
	add_hash_extended( ENDGAME_MODE, curr_val, best_list,
			   ENDGAME_SCORE | LOWER_BOUND, empties,
			   *selective_cutoff ? selectivity : 0 );
      disks_played = saved_disks_played;
      return curr_val;
    }

    if ( curr_val > best )
      best = curr_val;
    if ( child_selective_cutoff )
      *selective_cutoff = TRUE;

    if ( (best_list_index >= best_list_length) && (best_list_length < 4) )
      best_list[best_list_length++] = move;

    if ( !siblings_dispatched && first &&
	 (empties >= PARALLEL_SPLIT_DEPTH + SPLIT_NESTING_MARGIN * split_nesting) &&
	 (split_nesting <= MAX_SPLIT_NESTING) && (threads_count() > 1) &&
	 (threads_idle_count() > 0) &&
	 !is_panic_abort() && !force_return ) {
      can_split = TRUE;
      for ( i = 0; i < 100; i++ )
	proven[i] = FALSE;
      siblings_dispatched = TRUE;
      int cut_move = 0, cut_score = 0, cut_selective = 0;
      if ( dispatch_siblings( my_bits, opp_bits, side_to_move, level,
			      empties, disc_diff, alpha, beta, selectivity, move,
			      best_list, best_list_length, pre_search_done,
			      proven, proven_score, proven_cutoff,
			      &cut_move, &cut_score, &cut_selective ) ) {
	*selective_cutoff = cut_selective;
	update_best_list( best_list, cut_move, best_list_index, &best_list_length );
	if ( use_hash )
	  add_hash_extended( ENDGAME_MODE, cut_score, best_list,
			     ENDGAME_SCORE | LOWER_BOUND, empties,
			     *selective_cutoff ? selectivity : 0 );
	disks_played = saved_disks_played;
	return cut_score;
      }
      else {
	if ( cut_selective )
	  *selective_cutoff = TRUE;
      }
    }

    first = FALSE;
  }

  /* 6. All-node fail-low or pass/terminal */
  if ( !first ) {
    if ( use_hash ) {
      add_hash_extended( ENDGAME_MODE, best, best_list,
			 ENDGAME_SCORE | UPPER_BOUND, empties,
			 *selective_cutoff ? selectivity : 0 );
    }
    disks_played = saved_disks_played;
    return best;
  }
  else if ( pass_legal ) {
    if ( use_hash ) {
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
    }
    curr_val = -end_search_nws( opp_bits, my_bits,
				-beta, OPP( side_to_move ),
				empties, -disc_diff, FALSE, level,
				selectivity, selective_cutoff );
    if ( use_hash ) {
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
    }
    disks_played = saved_disks_played;
    return curr_val;
  }
  else {
    disks_played = saved_disks_played;
    if ( disc_diff > 0 )
      return disc_diff + empties;
    else if ( disc_diff < 0 )
      return disc_diff - empties;
    else
      return 0;
  }
}


/*
  END_SEARCH_PVS
  Single recursive PVS endgame search core.
  Unifies deep endgame search (>= 13 empties) and middle endgame search (8-12 empties).
*/

static int
end_search_pvs( BitBoard my_bits,
		BitBoard opp_bits,
		int alpha,
		int beta,
		int side_to_move,
		int empties,
		int disc_diff,
		int pass_legal,
		int level,
		int selectivity,
		int *selective_cutoff ) {
  static char buffer[16];
  HashEntry entry;
  int oppcol = OPP( side_to_move );
  int use_hash;
  int hash_hit = FALSE;

  *selective_cutoff = FALSE;

  if ( beta == alpha + 1 && level > 0 ) {
    return end_search_nws( my_bits, opp_bits, alpha, side_to_move,
			   empties, disc_diff, pass_legal, level,
			   selectivity, selective_cutoff );
  }

  if ( SPLIT_ABANDONED() )
    return SEARCH_ABORT;

  /* 1. Terminal leaf dispatch */
  if ( empties <= LOW_LEVEL_DEPTH ) {
    int res = solve_parity( my_bits, opp_bits, alpha, beta, side_to_move,
			    empties, disc_diff, pass_legal, level );
    pv_depth[level] = level + 1;
    pv[level][level] = end_best_move;
    if ( level == 0 )
      end_best_root_move = end_best_move;
    return res;
  }

  INCREMENT_COUNTER( nodes );

  /* 2. Symmetric stability bounds check */
#if USE_STABILITY
  if ( level <= MAX_SEARCH_DEPTH ) {
    if ( tls.stable_discs[oppcol][level] != 0 ) {
      int s = non_iterative_popcount( tls.stable_discs[oppcol][level] );
      int upper_bound = 64 - 2 * s;
      if ( upper_bound <= alpha ) {
        pv_depth[level] = level;
        return alpha;
      }
      if ( upper_bound < beta )
        beta = upper_bound + 1;
    }
    if ( tls.stable_discs[side_to_move][level] != 0 ) {
      int s = non_iterative_popcount( tls.stable_discs[side_to_move][level] );
      int lower_bound = 2 * s - 64;
      if ( lower_bound >= beta ) {
        pv_depth[level] = level;
        return lower_bound;
      }
      if ( lower_bound > alpha )
        alpha = lower_bound;
    }
    if ( alpha >= beta ) {
      pv_depth[level] = level;
      return alpha;
    }
  }

  if ( (opp_bits & BORDER_MASK) != 0 ) {
    int opp_cnt = non_iterative_popcount( opp_bits );
    int min_upper = 64 - 2 * opp_cnt;
    if ( min_upper <= alpha || min_upper < beta ) {
      EdgeIndices edges;
      int s_edge = count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
      if ( level <= MAX_SEARCH_DEPTH )
        tls.stable_discs[oppcol][level] |= edges.bits;
      int upper_bound = 64 - 2 * s_edge;
      if ( upper_bound <= alpha ) {
        pv_depth[level] = level;
        return alpha;
      }
      if ( upper_bound < beta )
        beta = upper_bound + 1;
      if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
        int s_full = count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        if ( level <= MAX_SEARCH_DEPTH )
          tls.stable_discs[oppcol][level] |= (oppcol == BLACKSQ ? last_black_stable : last_white_stable);
        upper_bound = 64 - 2 * s_full;
        if ( upper_bound <= alpha ) {
          pv_depth[level] = level;
          return alpha;
        }
        if ( upper_bound < beta )
          beta = upper_bound + 1;
      }
      if ( alpha >= beta ) {
        pv_depth[level] = level;
        return alpha;
      }
    }
  }

  if ( (my_bits & BORDER_MASK) != 0 ) {
    int my_cnt = non_iterative_popcount( my_bits );
    int max_lower = 2 * my_cnt - 64;
    if ( max_lower >= beta || max_lower > alpha ) {
      EdgeIndices edges;
      int s_edge = count_edge_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
      if ( level <= MAX_SEARCH_DEPTH )
        tls.stable_discs[side_to_move][level] |= edges.bits;
      int lower_bound = 2 * s_edge - 64;
      if ( lower_bound >= beta ) {
        pv_depth[level] = level;
        return lower_bound;
      }
      if ( lower_bound > alpha )
        alpha = lower_bound;
      if ( edges.bits != 0 && (my_bits & CENTRAL_MASK) != 0 ) {
        int s_full = count_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
        if ( level <= MAX_SEARCH_DEPTH )
          tls.stable_discs[side_to_move][level] |= (side_to_move == BLACKSQ ? last_black_stable : last_white_stable);
        lower_bound = 2 * s_full - 64;
        if ( lower_bound >= beta ) {
          pv_depth[level] = level;
          return lower_bound;
        }
        if ( lower_bound > alpha )
          alpha = lower_bound;
      }
      if ( alpha >= beta ) {
        pv_depth[level] = level;
        return alpha;
      }
    }
  }
#endif

  /* 3. Root UI reporting */
  if ( level == 0 ) {
    sprintf( buffer, "[%d,%d]:", alpha, beta );
    clear_sweep();
  }

  /* 4. Transposition table probing */
  use_hash = USE_HASH_TABLE;

  if ( use_hash ) {
    find_hash( &entry, ENDGAME_MODE );
    if ( (entry.draft == empties) &&
	 (entry.selectivity <= selectivity) &&
	 bb_valid_move( entry.move[0], my_bits, opp_bits ) &&
	 (entry.flags & ENDGAME_SCORE) &&
	 ((entry.flags & EXACT_VALUE) ||
	  ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	  ((entry.flags & UPPER_BOUND) && entry.eval <= alpha)) ) {
      end_best_move = entry.move[0];
      pv[level][level] = entry.move[0];
      pv_depth[level] = level + 1;
      if ( (level == 0) && !get_ponder_move() ) {
	send_sweep( "%c%c", TO_SQUARE( entry.move[0] ) );
	if ( (entry.flags & ENDGAME_SCORE) && (entry.flags & EXACT_VALUE) )
	  send_sweep( "=%d", entry.eval );
	else if ( (entry.flags & ENDGAME_SCORE) && (entry.flags & LOWER_BOUND) )
	  send_sweep( ">%d", entry.eval - 1 );
	else
	  send_sweep( "<%d", entry.eval + 1 );
#ifdef TEXT_BASED
	fflush( stdout );
#endif
      }
      if ( entry.selectivity > 0 )
	*selective_cutoff = TRUE;
      return entry.eval;
    }

    hash_hit = (entry.draft != NO_HASH_MOVE) &&
	       bb_valid_move( entry.move[0], my_bits, opp_bits ) &&
	       ((entry.flags & ENDGAME_SCORE) || (entry.flags & HEURISTIC_PRESEARCH_MOVE));
  }

  /* 5. Recursive PVS search loop */
  double node_val;
  int i;
  int move;
  int move_index;
  int update_pv, first;
  int curr_alpha;
  int pre_search_done, etc_tried;
  int best_list_index, best_list_length;
  int best_list[8];
  int proven[100], proven_score[100], proven_cutoff[100];
  int siblings_dispatched = FALSE;
  int can_split;
  int best;
  int curr_val;
  int saved_disks_played = disks_played;

  disks_played = 60 - empties;

  if ( level <= MAX_SEARCH_DEPTH ) {
    memset( cached_flipped[level], 0, sizeof(cached_flipped[level]) );
  }

  /* Use endgame multi-prob-cut to selectively prune the tree */
  if ( USE_MPC && (level > 2) && (selectivity > 0) ) {
    int cut;
    sync_board_from_bitboards( my_bits, opp_bits, side_to_move, empties );
    for ( cut = 0; cut < use_end_cut[disks_played]; cut++ ) {
	int shallow_remains = end_mpc_depth[disks_played][cut];
	int mpc_bias = ceil( end_mean[disks_played][shallow_remains] * 128.0 );
	int mpc_window = ceil( end_sigma[disks_played][shallow_remains] *
			       end_percentile[selectivity] * 128.0 );
	int beta_bound = 128 * beta + mpc_bias + mpc_window;
	int alpha_bound = 128 * alpha + mpc_bias - mpc_window;
	int shallow_val =
	  tree_search( level, level + shallow_remains, side_to_move,
		       alpha_bound, beta_bound, use_hash, FALSE, pass_legal );
	if ( shallow_val >= beta_bound ) {
	  if ( use_hash )
	    add_hash( ENDGAME_MODE, alpha, pv[level][level],
		      ENDGAME_SCORE | LOWER_BOUND, empties, selectivity );
	  *selective_cutoff = TRUE;
	  disks_played = saved_disks_played;
	  return beta;
	}
	if ( shallow_val <= alpha_bound ) {
	  if ( use_hash )
	    add_hash( ENDGAME_MODE, beta, pv[level][level],
		      ENDGAME_SCORE | UPPER_BOUND, empties, selectivity );
	  *selective_cutoff = TRUE;
	  disks_played = saved_disks_played;
	  return alpha;
	}
    }
  }

  first = TRUE;
  can_split = FALSE;
  best = -INFINITE_EVAL;
  pre_search_done = FALSE;
  etc_tried = 0;
  curr_alpha = alpha;

  /* Initialize move list and check hash table moves */
  move_count[disks_played] = 0;
  best_list_length = 0;
  for ( i = 0; i < 8; i++ )
    best_list[i] = 0;
  if ( hash_hit ) {
    best_list[0] = entry.move[0];
    best_list_length = 1;
  }

  if ( level == 0 && end_best_root_move != 0 && bb_valid_move( end_best_root_move, my_bits, opp_bits ) ) {
    int pos = -1;
    for ( i = 0; i < best_list_length; i++ ) {
      if ( best_list[i] == end_best_root_move ) { pos = i; break; }
    }
    if ( pos > 0 ) {
      for ( i = pos; i > 0; i-- ) {
        best_list[i] = best_list[i - 1];
      }
      best_list[0] = end_best_root_move;
    }
    else if ( pos < 0 ) {
      /* Prepend to best_list[0] */
      for ( i = best_list_length; i > 0; i-- ) {
        best_list[i] = best_list[i - 1];
      }
      best_list[0] = end_best_root_move;
      best_list_length++;
    }
  }

  for ( move_index = 0, best_list_index = 0; TRUE;
	  move_index++, best_list_index++ ) {
    int child_selective_cutoff;
    BitBoard new_my_bits;
    BitBoard new_opp_bits;

    if ( best_list_index < best_list_length ) {
	move = best_list[best_list_index];
	move_count[disks_played]++;
    }
    else {
	if ( !pre_search_done ) {
	  int etc_cutoff_score = 0;
	  if ( end_order_moves_presearch( level, empties, side_to_move,
					  my_bits, opp_bits, beta, curr_alpha,
					  selectivity, use_hash,
					  best_list, best_list_length,
					  can_split, proven, proven_score,
					  &etc_tried,
					  &etc_cutoff_score ) ) {
	    best_list[0] = etc_tried;
	    if ( selectivity > 0 )
	      *selective_cutoff = TRUE;
	    if ( use_hash )
	      add_hash_extended( ENDGAME_MODE, etc_cutoff_score, best_list,
				 ENDGAME_SCORE | LOWER_BOUND, empties,
				 *selective_cutoff ? selectivity : 0 );
	    pv_depth[level] = level + 1;
	    pv[level][level] = etc_tried;
	    if ( level == 0 )
	      end_best_root_move = etc_tried;
	    disks_played = saved_disks_played;
	    return etc_cutoff_score;
	  }
	  pre_search_done = TRUE;
	}

	if ( move_index == move_count[disks_played] )
	  break;
	move = select_move( move_index, move_count[disks_played] );
    }

    node_val = counter_value( &nodes );
    if ( node_val - last_panic_check >= EVENT_CHECK_INTERVAL ) {
	last_panic_check = node_val;
	check_panic_abort();
	if ( echo )
	  display_buffers();
	handle_event( TRUE, FALSE, TRUE );
	if ( is_panic_abort() || force_return ) {
	  disks_played = saved_disks_played;
	  return SEARCH_ABORT;
	}
    }

    if ( (level == 0) && !get_ponder_move() ) {
	if ( first )
	  send_sweep( "%-10s ", buffer );
	send_sweep( "%c%c", TO_SQUARE( move ) );
    }

    unsigned int diff1, diff2;
    int pred, succ;
    int flipped;
    if ( level <= MAX_SEARCH_DEPTH && cached_flipped[level][move] > 0 ) {
      flipped = cached_flipped[level][move];
      bb_flips = cached_flips[level][move];
    }
    else {
      flipped = TestFlips_wrapper( move, my_bits, opp_bits );
    }
    new_my_bits = bb_flips;
    FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

    end_make_move( move, new_my_bits, my_bits, side_to_move, &diff1, &diff2, &pred, &succ );

    if ( level + 1 <= MAX_SEARCH_DEPTH ) {
	tls.stable_discs[BLACKSQ][level + 1] = tls.stable_discs[BLACKSQ][level];
	tls.stable_discs[WHITESQ][level + 1] = tls.stable_discs[WHITESQ][level];
    }

    int new_disc_diff = -disc_diff - 2 * flipped - 1;

    update_pv = FALSE;
    if ( first ) {
	best = curr_val =
	  -end_search_pvs( new_opp_bits, new_my_bits,
			   -beta, -curr_alpha, OPP( side_to_move ),
			   empties - 1, new_disc_diff, TRUE, level + 1,
			   selectivity, &child_selective_cutoff );
	update_pv = TRUE;
	if ( level == 0 )
	  end_best_root_move = move;
	curr_alpha = MAX( best, curr_alpha );
    }
    else {
	curr_alpha = MAX( best, curr_alpha );
	if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) ) {
	  curr_val = proven_score[move];
	  child_selective_cutoff = proven_cutoff[move];
	}
	else
	  curr_val =
	    -end_search_nws( new_opp_bits, new_my_bits,
			     -(curr_alpha + 1), OPP( side_to_move ),
			     empties - 1, new_disc_diff, TRUE, level + 1,
			     selectivity, &child_selective_cutoff );

	if ( (curr_val > curr_alpha) && (curr_val < beta) ) {
	  if ( selectivity > 0 )
	    curr_val =
	      -end_search_pvs( new_opp_bits, new_my_bits,
			       -beta, INFINITE_EVAL, OPP( side_to_move ),
			       empties - 1, new_disc_diff, TRUE, level + 1,
			       selectivity, &child_selective_cutoff );
	  else
	    curr_val =
	      -end_search_pvs( new_opp_bits, new_my_bits,
			       -beta, -curr_val, OPP( side_to_move ),
			       empties - 1, new_disc_diff, TRUE, level + 1,
			       selectivity, &child_selective_cutoff );
	  if ( curr_val > best ) {
	    best = curr_val;
	    update_pv = TRUE;
	    if ( (level == 0) && !is_panic_abort() && !force_return )
	      end_best_root_move = move;
	  }
	}
	else if ( curr_val > best ) {
	  best = curr_val;
	  update_pv = TRUE;
	  if ( (level == 0) && !is_panic_abort() && !force_return )
	    end_best_root_move = move;
	}
    }

    if ( best >= beta )
	*selective_cutoff = child_selective_cutoff;
    else if ( child_selective_cutoff )
	*selective_cutoff = TRUE;

    end_unmake_move( move, diff1, diff2, pred, succ );

    if ( abs( curr_val ) >= 20000 || is_panic_abort() || force_return || SPLIT_ABANDONED() ) {
	disks_played = saved_disks_played;
	return SEARCH_ABORT;
    }

    if ( (level == 0) && !get_ponder_move() ) {
	if ( update_pv ) {
	  if ( curr_val <= alpha )
	    send_sweep( "<%d", curr_val + 1 );
	  else {
	    if ( curr_val >= beta )
	      send_sweep( ">%d", curr_val - 1 );
	    else {
	      send_sweep( "=%d", curr_val );
	    }
	  }
	}
	send_sweep( " " );
	if ( update_pv && (move_index > 0) && echo )
	  display_sweep( stdout );
    }

    if ( update_pv ) {
	update_best_list( best_list, move, best_list_index, &best_list_length );
	pv[level][level] = move;
	if ( pv_depth[level + 1] > level + 1 )
	  pv_depth[level] = pv_depth[level + 1];
	else
	  pv_depth[level] = level + 1;
	for ( i = level + 1; i < pv_depth[level]; i++ )
	  pv[level][i] = pv[level + 1][i];
    }
    if ( best >= beta ) {
	if ( use_hash )
	  add_hash_extended( ENDGAME_MODE, best, best_list,
			     ENDGAME_SCORE | LOWER_BOUND, empties,
			     *selective_cutoff ? selectivity : 0 );
	disks_played = saved_disks_played;
	return best;
    }

    if ( (best_list_index >= best_list_length) && !update_pv &&
	   (best_list_length < 4) )
	best_list[best_list_length++] = move;

    if ( !siblings_dispatched && first &&
	 (empties >= PARALLEL_SPLIT_DEPTH + SPLIT_NESTING_MARGIN * split_nesting) &&
	 (split_nesting <= MAX_SPLIT_NESTING) && (threads_count() > 1) &&
	 (threads_idle_count() > 0) &&
	 !is_panic_abort() && !force_return ) {
      can_split = TRUE;
      for ( i = 0; i < 100; i++ )
	proven[i] = FALSE;
      siblings_dispatched = TRUE;
      int cut_move = 0, cut_score = 0, cut_selective = 0;
      if ( dispatch_siblings( my_bits, opp_bits, side_to_move, level,
			      empties, disc_diff, curr_alpha, beta, selectivity, move,
			      best_list, best_list_length, pre_search_done,
			      proven, proven_score, proven_cutoff,
			      &cut_move, &cut_score, &cut_selective ) ) {
	best = cut_score;
	*selective_cutoff = cut_selective;
	best_list[0] = cut_move;
	pv[level][level] = cut_move;
	pv_depth[level] = level + 1;
	if ( level == 0 )
	  end_best_root_move = cut_move;
	if ( use_hash )
	  add_hash_extended( ENDGAME_MODE, best, best_list,
			     ENDGAME_SCORE | LOWER_BOUND, empties,
			     *selective_cutoff ? selectivity : 0 );
	disks_played = saved_disks_played;
	return best;
      }
      else {
	if ( cut_selective )
	  *selective_cutoff = TRUE;
      }
    }

    first = FALSE;
  }

  if ( !first ) {
    if ( use_hash ) {
	int flags = ENDGAME_SCORE;
	if ( best > alpha )
	  flags |= EXACT_VALUE;
	else
	  flags |= UPPER_BOUND;
	add_hash_extended( ENDGAME_MODE, best, best_list, flags, empties,
			   *selective_cutoff ? selectivity : 0 );
    }
    disks_played = saved_disks_played;
    return best;
  }
  else if ( pass_legal ) {
    if ( use_hash ) {
	hash1 ^= hash_flip_color1;
	hash2 ^= hash_flip_color2;
    }
    curr_val = -end_search_pvs( opp_bits, my_bits,
				  -beta, -alpha, OPP( side_to_move ),
				  empties, -disc_diff, FALSE, level,
				  selectivity, selective_cutoff );
    if ( use_hash ) {
	hash1 ^= hash_flip_color1;
	hash2 ^= hash_flip_color2;
    }
    disks_played = saved_disks_played;
    return curr_val;
  }
  else {
    pv_depth[level] = level;
    disks_played = saved_disks_played;
    if ( disc_diff > 0 )
	return disc_diff + empties;
    else if ( disc_diff < 0 )
	return disc_diff - empties;
    else
	return 0;
  }
}



/*
  END_TREE_WRAPPER
  Wrapper onto END_SEARCH_PVS which applies the knowledge that
  the range of valid scores is [-64,+64].  Komi, if any, is accounted for.
*/

static int
end_tree_wrapper( int level,
		  int max_depth,
		  int side_to_move,
		  int alpha,
		  int beta,
		  int selectivity,
		  int void_legal ) {
  (void) max_depth;
  int selective_cutoff;
  BitBoard my_bits, opp_bits;
  Board saved_root_board;
  memcpy( saved_root_board, board, sizeof( Board ) );

  set_bitboards( board, side_to_move, &my_bits, &opp_bits );

  prepare_to_solve( my_bits | opp_bits );

  if ( level == 0 ) {
    EdgeIndices eb, ew;
    BitBoard b_bits = (side_to_move == BLACKSQ ? my_bits : opp_bits);
    BitBoard w_bits = (side_to_move == BLACKSQ ? opp_bits : my_bits);

    (void) count_edge_stable_indexed( BLACKSQ, b_bits, w_bits, &eb );
    (void) count_stable_indexed( BLACKSQ, b_bits, w_bits, &eb );
    tls.stable_discs[BLACKSQ][0] = last_black_stable;

    (void) count_edge_stable_indexed( WHITESQ, w_bits, b_bits, &ew );
    (void) count_stable_indexed( WHITESQ, w_bits, b_bits, &ew );
    tls.stable_discs[WHITESQ][0] = last_white_stable;
  }
  else if ( level <= MAX_SEARCH_DEPTH ) {
    tls.stable_discs[BLACKSQ][level] = 0;
    tls.stable_discs[WHITESQ][level] = 0;
  }

  int my_discs = non_iterative_popcount( my_bits );
  int opp_discs = non_iterative_popcount( opp_bits );
  int empties = 64 - my_discs - opp_discs;
  int disc_diff = my_discs - opp_discs;

  int result = end_search_pvs( my_bits, opp_bits,
			       MAX( alpha - komi_shift, -64 ),
			       MIN( beta - komi_shift, 64 ),
			       side_to_move,
			       empties,
			       disc_diff,
			       void_legal,
			       level,
			       selectivity,
			       &selective_cutoff );
  memcpy( board, saved_root_board, sizeof( Board ) );
  return result + komi_shift;
}



/*
   FULL_EXPAND_PV
   Pad the PV with optimal moves in the low-level phase.
*/

static void
full_expand_pv( int side_to_move,
		int selectivity ) {
  int i;
  int pass_count;
  int new_pv_depth;
  int new_pv[61];
  int new_side_to_move[61];

  new_pv_depth = 0;
  pass_count = 0;
  while ( pass_count < 2 ) {
    int move;

    generate_all( side_to_move );
    if ( move_count[disks_played] > 0 ) {
      int empties = 64 - disc_count( BLACKSQ ) - disc_count( WHITESQ );

      (void) end_tree_wrapper( new_pv_depth, empties, side_to_move,
			       -64, 64, selectivity, TRUE );
      move = pv[new_pv_depth][new_pv_depth];
      new_pv[new_pv_depth] = move;
      new_side_to_move[new_pv_depth] = side_to_move;
      (void) make_move( side_to_move, move, TRUE );
      new_pv_depth++;
    }
    else {
      hash1 ^= hash_flip_color1;
      hash2 ^= hash_flip_color2;
      pass_count++;
    }
    side_to_move = OPP( side_to_move );
  }
  for ( i = new_pv_depth - 1; i >= 0; i-- )
    unmake_move( new_side_to_move[i], new_pv[i] );
  for ( i = 0; i < new_pv_depth; i++ )
    pv[0][i] = new_pv[i];
  pv_depth[0] = new_pv_depth;
}



/*
  SEND_SOLVE_STATUS
  Displays endgame results - partial or full.
*/

static void
send_solve_status( int empties,
		   int side_to_move,
		   EvaluationType *eval_info ) {
  char *eval_str;
  double node_val;

  set_current_eval( *eval_info );
  clear_status();
  send_status( "-->  %2d  ", empties );
  eval_str = produce_eval_text( *eval_info, TRUE );
  send_status( "%-10s  ", eval_str );
  free( eval_str );
  node_val = counter_value( &nodes );
  send_status_nodes( node_val );
  if ( get_ponder_move() )
    send_status( "{%c%c} ", TO_SQUARE( get_ponder_move() ) );
  send_status_pv( pv[0], empties );
  send_status_time( get_elapsed_time() );
  if ( get_elapsed_time() > 0.0001 )
    send_status( "%6.0f %s  ", node_val / (get_elapsed_time() + 0.0001),
		 NPS_ABBREV);
}



/*
  END_GAME
  Provides an interface to the fast endgame solver.
*/

int
end_game( int side_to_move,
	  int wld,
	  int force_echo,
	  int allow_book,
	  int komi,
	  EvaluationType *eval_info ) {
  double current_confidence;
  enum { WIN, LOSS, DRAW, UNKNOWN } solve_status;
  int book_move;
  int empties;
  int selectivity;
  int alpha, beta;
  int any_search_result, exact_score_failed;
  int incomplete_search;
  int long_selective_search;
  int old_depth, old_eval;
  int last_window_center;
  int old_pv[MAX_SEARCH_DEPTH];
  EvaluationType book_eval_info;

  smp_clear_stop();
  /* increment_hash_generation(); */

  empties = 64 - disc_count( BLACKSQ ) - disc_count( WHITESQ );

  /* In komi games, the WLD window is adjusted. */

  if ( side_to_move == BLACKSQ )
    komi_shift = komi;
  else
    komi_shift = -komi;

  /* Check if the position is solved (WLD or exact) in the book. */

  book_move = PASS;
  if ( allow_book ) {
    /* Is the exact score known? */

    fill_move_alternatives( side_to_move, FULL_SOLVED );
    book_move = get_book_move( side_to_move, FALSE, eval_info );
    if ( book_move != PASS ) {
      root_eval = eval_info->score / 128;
      hash_expand_pv( side_to_move, ENDGAME_MODE, EXACT_VALUE, 0 );
      send_solve_status( empties, side_to_move, eval_info );
      return book_move;
    }

    /* Is the WLD status known? */

    fill_move_alternatives( side_to_move, WLD_SOLVED );
    if ( komi_shift == 0 ) {
      book_move = get_book_move( side_to_move, FALSE, eval_info );
      if ( book_move != PASS ) {
	if ( wld ) {
	  root_eval = eval_info->score / 128;
	  hash_expand_pv( side_to_move, ENDGAME_MODE,
			  EXACT_VALUE | UPPER_BOUND | LOWER_BOUND, 0  );
	  send_solve_status( empties, side_to_move, eval_info );
	  return book_move;
	}
	else
	  book_eval_info = *eval_info;
      }
    }

    fill_endgame_hash( HASH_DEPTH, 0 );
  }

  last_panic_check = 0.0;
  solve_status = UNKNOWN;
  old_eval = 0;

  BitBoard root_my_bits, root_opp_bits;
  set_bitboards( board, side_to_move, &root_my_bits, &root_opp_bits );
  if ( bb_valid_move( pv[0][0], root_my_bits, root_opp_bits ) )
    end_best_root_move = pv[0][0];
  else
    end_best_root_move = 0;

  /* Prepare for the shallow searches using the midgame eval */

  piece_count[BLACKSQ][disks_played] = disc_count( BLACKSQ );
  piece_count[WHITESQ][disks_played] = disc_count( WHITESQ );

  if ( empties > 32 )
    set_panic_threshold( 0.20 );
  else if ( empties < 22 )
    set_panic_threshold( 0.50 );
  else
    set_panic_threshold( 0.50 - (empties - 22) * 0.03 );

  reset_buffer_display();

  /* Make sure the pre-searches don't mess up the hash table */

  toggle_midgame_hash_usage( TRUE, FALSE );

  incomplete_search = FALSE;
  any_search_result = FALSE;

  /* Start off by selective endgame search */

  last_window_center = 0;

  if ( !wld && (empties >= 16) ) {
    set_bitboards( board, side_to_move, &root_my_bits, &root_opp_bits );
    int est = end_pattern_evaluate( root_my_bits, root_opp_bits );
    last_window_center = end_aspiration_center( est, 60 );
  }

  if ( empties > DISABLE_SELECTIVITY ) {
    if ( wld ) {
      for ( selectivity = MAX_SELECTIVITY; (selectivity > 0) &&
	      !is_panic_abort() && !force_return; selectivity-- ) {
	unsigned int flags;
	EvalResult res;

	alpha = -1;
	beta = +1;
	root_eval = end_tree_wrapper( 0, empties, side_to_move,
				      alpha, beta, selectivity, TRUE );

	adjust_counter( &nodes );

	if ( is_panic_abort() || force_return )
	  break;

	any_search_result = TRUE;
	old_eval = root_eval;
	store_pv( old_pv, &old_depth );
	current_confidence = confidence[selectivity];

	flags = EXACT_VALUE;
	if ( root_eval == 0 )
	  res = DRAWN_POSITION;
	else {
	  flags |= (UPPER_BOUND | LOWER_BOUND);
	  if ( root_eval > 0 )
	    res = WON_POSITION;
	  else
	    res = LOST_POSITION;
	}
	*eval_info =
	  create_eval_info( SELECTIVE_EVAL, res, root_eval * 128,
			    current_confidence, empties, FALSE );
	if ( full_output_mode ) {
	  hash_expand_pv( side_to_move, ENDGAME_MODE, flags, selectivity );
	  send_solve_status( empties, side_to_move, eval_info );
	}
      }
    }
    else {
      static const int sel_schedule[] = { 6, 3, 2, 1 };
      int pass_idx;
      for ( pass_idx = 0; pass_idx < 4 &&
	      !is_panic_abort() && !force_return; pass_idx++ ) {
	selectivity = sel_schedule[pass_idx];
	alpha = last_window_center - 1;
	beta = last_window_center + 1;

	root_eval = end_tree_wrapper( 0, empties, side_to_move,
				      alpha, beta, selectivity, TRUE );

	if ( root_eval <= alpha ) {
	  int steps = 0;
	  do {
	    last_window_center -= 2;
	    alpha = last_window_center - 1;
	    beta = last_window_center + 1;
	    if ( is_panic_abort() || force_return || ++steps >= 32 )
	      break;
	    root_eval = end_tree_wrapper( 0, empties, side_to_move,
					  alpha, beta, selectivity, TRUE );
	  } while ( root_eval <= alpha );
	  root_eval = last_window_center;
	}
	else if ( root_eval >= beta ) {
	  int steps = 0;
	  do {
	    last_window_center += 2;
	    alpha = last_window_center - 1;
	    beta = last_window_center + 1;
	    if ( is_panic_abort() || force_return || ++steps >= 32 )
	      break;
	    root_eval = end_tree_wrapper( 0, empties, side_to_move,
					  alpha, beta, selectivity, TRUE );
	  } while ( root_eval >= beta );
	  root_eval = last_window_center;
	}

	adjust_counter( &nodes );

	if ( is_panic_abort() || force_return )
	  break;

	last_window_center = root_eval;

	if ( !is_panic_abort() && !force_return ) {
	  any_search_result = TRUE;
	  old_eval = root_eval;
	  store_pv( old_pv, &old_depth );
	  current_confidence = confidence[selectivity];

	  *eval_info =
	    create_eval_info( SELECTIVE_EVAL, UNSOLVED_POSITION,
			      root_eval * 128, current_confidence,
			      empties, FALSE );
	  if ( full_output_mode ) {
	    hash_expand_pv( side_to_move, ENDGAME_MODE, EXACT_VALUE, selectivity );
	    send_solve_status( empties, side_to_move, eval_info );
	  }
	}
      }
    }
  }
  else
    selectivity = 0;

  /* Check if the selective search took more than 40% of the allocated
       time. If this is the case, there is no point attempting WLD. */

  long_selective_search = check_threshold( 0.35 );

  /* Make sure the panic abort flag is set properly; it must match
     the status of long_selective_search. This is not automatic as
     it is not guaranteed that any selective search was performed. */

  check_panic_abort();

  if ( force_return || (wld && (is_panic_abort() || long_selective_search)) ) {

    /* Don't try non-selective solve. */

    if ( any_search_result ) {
      if ( echo && (is_panic_abort() || force_return) ) {
#ifdef TEXT_BASED
	printf( "%s %.1f %c %s\n", SEMI_PANIC_ABORT_TEXT, get_elapsed_time(),
		SECOND_ABBREV, SEL_SEARCH_TEXT );
#endif
	if ( full_output_mode ) {
	  unsigned int flags;

	  flags = EXACT_VALUE;
	  if ( solve_status != DRAW )
	    flags |= (UPPER_BOUND | LOWER_BOUND);
	  hash_expand_pv( side_to_move, ENDGAME_MODE, flags, selectivity );
	  send_solve_status( empties, side_to_move, eval_info );
	}
      }
      pv[0][0] = end_best_root_move;
      pv_depth[0] = 1;
      root_eval = old_eval;
      clear_panic_abort();
    }
    else {
#ifdef TEXT_BASED
      if ( echo )
	printf( "%s %.1f %c %s\n", PANIC_ABORT_TEXT, get_elapsed_time(),
		SECOND_ABBREV, SEL_SEARCH_TEXT );
#endif
      root_eval = SEARCH_ABORT;                   
    }

    if ( echo || force_echo )
      display_status( stdout, FALSE );

    if ( (book_move != PASS) &&
	 ((book_eval_info.res == WON_POSITION) ||
	  (book_eval_info.res == DRAWN_POSITION)) ) {
      /* If there is a known win (or mismarked draw) available,
	   always play it upon timeout. */
      *eval_info = book_eval_info;
      root_eval = eval_info->score / 128;
      return book_move;
    }
    else
      return pv[0][0];
  }

  /* Start non-selective solve */
  clear_panic_abort();

  if ( !wld && (empties >= 16) ) {
    set_bitboards( board, side_to_move, &root_my_bits, &root_opp_bits );
    prepare_to_solve( root_my_bits | root_opp_bits );
    determine_hash_values( side_to_move, board );

    int max_presearch = MIN( 14, empties - 8 );
    int pre_best = 0;
    int last_eval = 0;
    presearch_nodes = 0;
    presearch_budget = 100000;
    presearch_aborted = FALSE;

    for ( int d = 4; d <= max_presearch; d += 2 ) {
      int cur_best = 0;
      int val;
      if ( d == 4 ) {
	val = end_presearch_ab( root_my_bits, root_opp_bits,
				side_to_move, d, empties,
				-INFINITE_EVAL, INFINITE_EVAL,
				0, &cur_best );
      } else {
	int alpha = last_eval - 256;
	int beta = last_eval + 256;
	val = end_presearch_ab( root_my_bits, root_opp_bits,
				side_to_move, d, empties,
				alpha, beta,
				0, &cur_best );
	if ( !presearch_aborted && !is_panic_abort() && !force_return ) {
	  int delta = 256;
	  while ( (val <= alpha || val >= beta) && !presearch_aborted && !is_panic_abort() && !force_return ) {
	    if ( val <= alpha )
	      alpha = MAX( -INFINITE_EVAL, alpha - delta );
	    if ( val >= beta )
	      beta = MIN( INFINITE_EVAL, beta + delta );
	    delta = 2 * delta + 256;
	    val = end_presearch_ab( root_my_bits, root_opp_bits,
				    side_to_move, d, empties,
				    alpha, beta,
				    0, &cur_best );
	  }
	}
      }

      if ( presearch_aborted || is_panic_abort() || force_return )
	break;

      last_eval = val;
      if ( cur_best != 0 && bb_valid_move( cur_best, root_my_bits, root_opp_bits ) ) {
	pre_best = cur_best;
      }
    }

    if ( !any_search_result && pre_best != 0 ) {
      end_best_root_move = pre_best;
      pv[0][0] = pre_best;
      last_window_center = end_aspiration_center( last_eval, 62 );
    }
    determine_hash_values( side_to_move, board );
    prepare_to_solve( root_my_bits | root_opp_bits );
    adjust_counter( &nodes );
  }



  if ( wld ) {
    alpha = -1;
    beta = +1;
  }
  else {
    alpha = last_window_center - 1;
    beta = last_window_center + 1;
  }

  root_eval = end_tree_wrapper( 0, empties, side_to_move,
				alpha, beta, 0, TRUE );

  adjust_counter( &nodes );

  if ( !is_panic_abort() && !force_return ) {
    if ( !wld ) {
      int delta = 4;
      while ( (root_eval <= alpha || root_eval >= beta) && !is_panic_abort() && !force_return ) {
        if ( root_eval >= 64 || root_eval <= -64 )
          break;

        if ( delta >= 64 ) {
          alpha = -64;
          beta = 64;
          root_eval = end_tree_wrapper( 0, empties, side_to_move,
                                        alpha, beta, 0, TRUE );
          adjust_counter( &nodes );
          break;
        }

        if ( root_eval <= alpha ) {
          /* Fail low: widen lower bound exponentially */
          alpha = MAX( -64, last_window_center - delta );
          beta = MIN( 64, last_window_center + 1 );
        }
        else if ( root_eval >= beta ) {
          /* Fail high: widen upper bound exponentially */
          alpha = MAX( -64, last_window_center - 1 );
          beta = MIN( 64, last_window_center + delta );
        }
        root_eval = end_tree_wrapper( 0, empties, side_to_move,
                                      alpha, beta, 0, TRUE );
        adjust_counter( &nodes );
        if ( (alpha <= -64 && beta >= 64) || root_eval >= 64 || root_eval <= -64 )
          break;
        delta *= 2;
        if ( delta > 64 ) delta = 64;
      }
    }
    if ( !is_panic_abort() && !force_return ) {
      EvalResult res;
      if ( root_eval < 0 )
	res = LOST_POSITION;
      else if ( root_eval == 0 )
	res = DRAWN_POSITION;
      else
	res = WON_POSITION;
      if ( wld ) {
	unsigned int flags;

	if ( root_eval == 0 )
	  flags = EXACT_VALUE;
	else
	  flags = UPPER_BOUND | LOWER_BOUND;
	*eval_info =
	  create_eval_info( WLD_EVAL, res, root_eval * 128, 0.0, empties, FALSE );
	if ( full_output_mode ) {
	  hash_expand_pv( side_to_move, ENDGAME_MODE, flags, 0 );
	  send_solve_status( empties, side_to_move, eval_info );
	}
      }
      else {
	*eval_info =
	  create_eval_info( EXACT_EVAL, res, root_eval * 128, 0.0, 
			    empties, FALSE );
	if ( full_output_mode ) {
	  hash_expand_pv( side_to_move, ENDGAME_MODE, EXACT_VALUE, 0 );
	  send_solve_status( empties, side_to_move, eval_info );
	}
      }
    }
  }

  adjust_counter( &nodes );

  /* Check for abort. */

  if ( is_panic_abort() || force_return ) {
    if ( any_search_result ) {
      if ( echo ) {
#ifdef TEXT_BASED
	printf( "%s %.1f %c %s\n", SEMI_PANIC_ABORT_TEXT,
		get_elapsed_time(), SECOND_ABBREV, WLD_SEARCH_TEXT );
#endif
	if ( full_output_mode ) {
	  unsigned int flags;

	  flags = EXACT_VALUE;
	  if ( root_eval != 0 )
	    flags |= (UPPER_BOUND | LOWER_BOUND);
	  hash_expand_pv( side_to_move, ENDGAME_MODE, flags, 0 );
	  send_solve_status( empties, side_to_move, eval_info );
	}
	if ( echo || force_echo )
	  display_status( stdout, FALSE );
      }
      restore_pv( old_pv, old_depth );
      root_eval = old_eval;
      clear_panic_abort();
    }
    else {
#ifdef TEXT_BASED
      if ( echo )
	printf( "%s %.1f %c %s\n", PANIC_ABORT_TEXT,
		get_elapsed_time(), SECOND_ABBREV, WLD_SEARCH_TEXT );
#endif
      root_eval = SEARCH_ABORT;
    }

    return pv[0][0];
  }

  /* Update solve info. */

  store_pv( old_pv, &old_depth );
  old_eval = root_eval;

  if ( !is_panic_abort() && !force_return && (empties > earliest_wld_solve) )
    earliest_wld_solve = empties;


  /* Check for aborted search. */

  exact_score_failed = FALSE;
  if ( incomplete_search ) {
    if ( echo ) {
#ifdef TEXT_BASED
      printf( "%s %.1f %c %s\n", SEMI_PANIC_ABORT_TEXT,
	      get_elapsed_time(), SECOND_ABBREV, EXACT_SEARCH_TEXT );
#endif
      if ( full_output_mode ) {
	hash_expand_pv( side_to_move, ENDGAME_MODE, EXACT_VALUE, 0 );
	send_solve_status( empties, side_to_move, eval_info );
      }
      if ( echo || force_echo )
	display_status( stdout, FALSE );
    }
    pv[0][0] = end_best_root_move;
    pv_depth[0] = 1;
    root_eval = old_eval;
    exact_score_failed = TRUE;
    clear_panic_abort();
  }
      
  if ( abs( root_eval ) % 2 == 1 ) {
    if ( root_eval > 0 )
      root_eval++;
    else
      root_eval--;
  }

  if ( !exact_score_failed && !wld && (empties > earliest_full_solve) )
    earliest_full_solve = empties;

  if ( !wld && !exact_score_failed ) {
    eval_info->type = EXACT_EVAL;
    eval_info->score = root_eval * 128;
  }

  if ( !wld && !exact_score_failed ) {
    hash_expand_pv( side_to_move, ENDGAME_MODE, EXACT_VALUE, 0 );
    send_solve_status( empties, side_to_move, eval_info );
  }

  if ( echo || force_echo )
    display_status( stdout, FALSE );

  /* For shallow endgames, we can afford to compute the entire PV
     move by move. */

  if ( !wld && !incomplete_search && !force_return &&
       (empties <= PV_EXPANSION) )
    full_expand_pv( side_to_move, 0 );

  return pv[0][0];
}


