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

#define REGION_PARITY_BONUS          64
#define FAST_PATTERN_DEPTH           13
#define LAZY_PATTERN_MARGIN          768
#define ORDER_TS_MIN_DEPTH           15
#define ORDER_TS_TOPK                2





/* Pseudo-probabilities corresponding to the percentiles.
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
*/

static ALWAYS_INLINE int
bb_valid_move( int move, BitBoard my_bits, BitBoard opp_bits ) {
  BitBoard dummy;

  if ( (unsigned int) move >= 100 || (((~(my_bits | opp_bits)) & square_mask[move]) == 0) )
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

int
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

  /* Overall strategy: Lazy evaluation whenever possible, i.e., don't
     update bitboards until they are used. Also look at alpha and beta
     in order to perform strength reduction: Feasibility testing is
     faster than counting number of flips. */

  /* Try the first of the two empty squares... */

  flipped = TestFlips_wrapper_to( sq1, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {  /* SQ1 feasible for me */
    ts->ts_nodes.lo++;

    ev = disc_diff + 2 * flipped;

    flipped = CountFlips_bitboard( sq2, opp_bits & ~new_my_bits );
    if ( flipped != 0 )
      ev -= 2 * flipped;
    else {  /* He passes, check if SQ2 is feasible for me */
      if ( ev >= 0 ) {  /* I'm ahead, so EV will increase by at least 2 */
	ev += 2;
	if ( ev < beta )  /* Only bother if not certain fail-high */
	  ev += 2 * CountFlips_bitboard( sq2, new_my_bits );
      }
      else {
	if ( ev < beta ) {  /* Only bother if not fail-high already */
	  flipped = CountFlips_bitboard( sq2, new_my_bits );
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

  flipped = TestFlips_wrapper_to( sq2, my_bits, opp_bits, &new_my_bits );
  if ( flipped != 0 ) {  /* SQ2 feasible for me */
    ts->ts_nodes.lo++;

    ev = disc_diff + 2 * flipped;

    flipped = CountFlips_bitboard( sq1, opp_bits & ~new_my_bits );
    if ( flipped != 0 )  /* SQ1 feasible for him, game over */
      ev -= 2 * flipped;
    else {  /* He passes, check if SQ1 is feasible for me */
      if ( ev >= 0 ) {  /* I'm ahead, so EV will increase by at least 2 */
	ev += 2;
	if ( ev < beta )  /* Only bother if not certain fail-high */
	  ev += 2 * CountFlips_bitboard( sq1, new_my_bits );
      }
      else {
	if ( ev < beta ) {  /* Only bother if not fail-high already */
	  flipped = CountFlips_bitboard( sq1, new_my_bits );
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
      return -solve_two_empty( ts, opp_bits, my_bits, sq1, sq2, -beta,
			       -alpha, -disc_diff, FALSE );
  }
  else
    return score;
}


int
solve_three_empty( ThreadState *ts,
		   BitBoard my_bits,
		   BitBoard opp_bits,
		   int sq1,
		   int sq2,
		   int sq3,
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

  unsigned int q1 = quadrant_mask[sq1];
  unsigned int q2 = quadrant_mask[sq2];
  unsigned int q3 = quadrant_mask[sq3];
  unsigned int par = q1 ^ q2 ^ q3;
  if ( !(q1 & par) ) {
    if ( q2 & par ) {
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
    if ( !pass_legal ) {  /* Three empty squares */
      if ( disc_diff > 0 )
	return disc_diff + 3;
      if ( disc_diff < 0 )
	return disc_diff - 3;
      return 0;  /* Can't reach this code, only keep it for symmetry */
    }
    else
      return -solve_three_empty( ts, opp_bits, my_bits, sq1, sq2, sq3,
				 -beta, -alpha, -disc_diff, FALSE );
  }

  return score;
}



/* Forward declaration for end_search_pvs */
static int
end_search_pvs( ThreadState *ts,
		BitBoard my_bits,
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
end_search_nws( ThreadState *ts,
		BitBoard my_bits,
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
solve_parity( ThreadState *ts,
	      BitBoard my_bits,
	      BitBoard opp_bits,
	      int alpha,
	      int beta, 
	      int color,
	      int empties,
	      int disc_diff,
	      int pass_legal,
	      int level ) {
  if ( empties == 8 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    int sq6 = ts->ts_end_move_list[sq5].succ;
    int sq7 = ts->ts_end_move_list[sq6].succ;
    int sq8 = ts->ts_end_move_list[sq7].succ;
    return solve_eight_empty( ts, my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5, sq6, sq7, sq8,
			      alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 7 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    int sq6 = ts->ts_end_move_list[sq5].succ;
    int sq7 = ts->ts_end_move_list[sq6].succ;
    return solve_seven_empty( ts, my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5, sq6, sq7,
			      alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 6 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    int sq6 = ts->ts_end_move_list[sq5].succ;
    return solve_six_empty( ts, my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5, sq6,
			    alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 5 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    int sq5 = ts->ts_end_move_list[sq4].succ;
    return solve_five_empty( ts, my_bits, opp_bits, sq1, sq2, sq3, sq4, sq5,
			     alpha, beta, color, disc_diff, pass_legal );
  }

  if ( empties == 4 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    int sq4 = ts->ts_end_move_list[sq3].succ;
    return solve_four_empty( ts, my_bits, opp_bits, sq1, sq2, sq3, sq4,
			     alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 3 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    int sq3 = ts->ts_end_move_list[sq2].succ;
    return solve_three_empty( ts, my_bits, opp_bits, sq1, sq2, sq3,
			      alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 2 ) {
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int sq2 = ts->ts_end_move_list[sq1].succ;
    return solve_two_empty( ts, my_bits, opp_bits, sq1, sq2,
			    alpha, beta, disc_diff, pass_legal );
  }

  if ( empties == 1 ) {
    BitBoard dummy;
    int sq1 = ts->ts_end_move_list[END_MOVE_LIST_HEAD].succ;
    int flipped = TestFlips_wrapper_to( sq1, my_bits, opp_bits, &dummy );
    if ( flipped != 0 )
      return disc_diff + 2 * flipped + 1;
    if ( !pass_legal ) {
      if ( disc_diff > 0 ) return disc_diff + 1;
      if ( disc_diff < 0 ) return disc_diff - 1;
      return 0;
    }
    flipped = TestFlips_wrapper_to( sq1, opp_bits, my_bits, &dummy );
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
  return end_search_pvs( ts, my_bits, opp_bits, alpha, beta, color,
			 empties, disc_diff, pass_legal, level,
			 0, &selective_cutoff );
}



/*
  END_MAKE_MOVE
  Make SQ: update incremental hash keys, region parity, and unlink from move list.
*/

static ALWAYS_INLINE void
end_make_move( ThreadState *ts,
	       int sq,
	       BitBoard new_my_bits,
	       BitBoard my_bits,
	       int color,
	       unsigned int *diff1,
	       unsigned int *diff2,
	       int *pred,
	       int *succ ) {
  end_hash_diff( new_my_bits, my_bits, color, sq, diff1, diff2 );
  ts->ts_hash1 ^= *diff1;
  ts->ts_hash2 ^= *diff2;
  prefetch_hash_endgame_key( ts->ts_hash2 );
  ts->ts_region_parity ^= quadrant_mask[sq];
  *pred = ts->ts_end_move_list[sq].pred;
  *succ = ts->ts_end_move_list[sq].succ;
  ts->ts_end_move_list[*pred].succ = *succ;
  ts->ts_end_move_list[*succ].pred = *pred;
}



/*
  END_UNMAKE_MOVE
  Unmake SQ: restore incremental hash keys, region parity, and relink into move list.
*/

static ALWAYS_INLINE void
end_unmake_move( ThreadState *ts,
		 int sq,
		 unsigned int diff1,
		 unsigned int diff2,
		 int pred,
		 int succ ) {
  ts->ts_region_parity ^= quadrant_mask[sq];
  ts->ts_hash1 ^= diff1;
  ts->ts_hash2 ^= diff2;
  ts->ts_end_move_list[pred].succ = sq;
  ts->ts_end_move_list[succ].pred = sq;
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

#ifndef MIN_ETC_DEPTH
#define MIN_ETC_DEPTH      10
#endif

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

static ALWAYS_INLINE int
probe_forced_reply_cutoff( const ThreadState *ts,
			   int cand_sq,
			   BitBoard child_my_bits,
			   BitBoard child_opp_bits,
			   BitBoard opp_moves,
			   int opp_mob,
			   unsigned int diff1_my,
			   unsigned int diff2_my,
			   int side_to_move,
			   int empties,
			   int alpha,
			   int beta,
			   int selectivity,
			   int *cutoff_score ) {
  (void) cand_sq;

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
  BitBoard cur_moves = opp_moves;
  int cur_mob = 1;

  while ( k < 16 && cur_empties > 0 ) {
    if ( k > 1 ) {
      cur_moves = bitboard_moves( cur_my_bits, cur_opp_bits );
      cur_mob = non_iterative_popcount( cur_moves );
    }

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
	if ( OPP( cur_side ) == side_to_move ) {
	  /* side_to_move score is >= bound */
	  if ( bound >= beta ) {
	    *cutoff_score = bound;
	    return FORCED_CUTOFF;
	  }
	  if ( edges.bits != 0 && (cur_opp_bits & CENTRAL_MASK) != 0 ) {
	    int max_stable = s_edge + non_iterative_popcount( cur_opp_bits & CENTRAL_MASK );
	    if ( 2 * max_stable - 64 >= beta ) {
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
	    int max_stable = s_edge + non_iterative_popcount( cur_opp_bits & CENTRAL_MASK );
	    if ( -(2 * max_stable - 64) <= alpha ) {
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
      unsigned int p_h1 = ts->ts_hash1 ^ cum_diff1;
      unsigned int p_h2 = ts->ts_hash2 ^ cum_diff2;
      prefetch_hash_endgame_key( p_h2 );
      HashEntry g_entry;
      find_hash_key( &g_entry, p_h1, p_h2 );

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

static ALWAYS_INLINE int
end_probe_2ply_etc( const ThreadState *ts,
		    int sq,
		    BitBoard child_my_bits,
		    BitBoard child_opp_bits,
		    BitBoard opp_moves,
		    int opp_mob,
		    unsigned int diff1_my,
		    unsigned int diff2_my,
		    int side_to_move,
		    int empties,
		    int alpha,
		    int beta,
		    int selectivity,
		    int child_best_reply,
		    int *cutoff_score ) {
  if ( opp_mob == 1 ) {
    int res = probe_forced_reply_cutoff( ts, sq, child_my_bits, child_opp_bits,
					 opp_moves, opp_mob,
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
      unsigned int p_h1 = ts->ts_hash1 ^ diff1_my ^ hash_flip_color1;
      unsigned int p_h2 = ts->ts_hash2 ^ diff2_my ^ hash_flip_color2;
      prefetch_hash_endgame_key( p_h2 );
      HashEntry p_entry;
      find_hash_key( &p_entry, p_h1, p_h2 );

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

  int oppcol = OPP( side_to_move );
  int child_best_ge_beta = FALSE;
  int child_best_eval = INFINITE_EVAL;

  /* Priority check: probe opponent's known TT best reply if available */
  if ( child_best_reply != 0 ) {
    BitBoard opp_flips_new_bits;
    int flipped = TestFlips_bitboard_to( child_best_reply, child_opp_bits, child_my_bits, &opp_flips_new_bits );
    if ( flipped > 0 ) {
      unsigned int diff1_opp, diff2_opp;
      end_hash_diff( opp_flips_new_bits, child_opp_bits, oppcol, child_best_reply, &diff1_opp, &diff2_opp );
      unsigned int g_h1 = ts->ts_hash1 ^ diff1_my ^ diff1_opp;
      unsigned int g_h2 = ts->ts_hash2 ^ diff2_my ^ diff2_opp;
      prefetch_hash_endgame_key( g_h2 );
      HashEntry g_entry;
      find_hash_key( &g_entry, g_h1, g_h2 );

      if ( (g_entry.flags & ENDGAME_SCORE) &&
	   (g_entry.draft >= empties - 2) &&
	   (g_entry.selectivity <= selectivity) ) {
	if ( (g_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
	     (g_entry.eval <= alpha) ) {
	  return ETC_2PLY_REFUTED;
	}
	if ( (g_entry.flags & (LOWER_BOUND | EXACT_VALUE)) &&
	     (g_entry.eval >= beta) ) {
	  child_best_ge_beta = TRUE;
	  child_best_eval = g_entry.eval;
	}
      }
    }
  }

  if ( opp_mob < 1 || opp_mob > MAX_2PLY_ETC_MOB )
    return ETC_2PLY_NONE;

  BitBoard moves_bb = opp_moves;
  int all_ge_beta = TRUE;
  int min_g_eval = INFINITE_EVAL;

  while ( moves_bb != 0 ) {
    int bit = FIRST_BIT( moves_bb );
    moves_bb &= moves_bb - 1;
    int opp_sq = square_of_bit[bit];

    if ( opp_sq == child_best_reply ) {
      if ( child_best_ge_beta ) {
	if ( child_best_eval < min_g_eval )
	  min_g_eval = child_best_eval;
      }
      else {
	all_ge_beta = FALSE;
      }
      continue;
    }

    BitBoard opp_flips_new_bits;
    int flipped = TestFlips_bitboard_to( opp_sq, child_opp_bits, child_my_bits, &opp_flips_new_bits );
    if ( flipped == 0 ) {
      all_ge_beta = FALSE;
      continue;
    }

    unsigned int diff1_opp, diff2_opp;
    end_hash_diff( opp_flips_new_bits, child_opp_bits, oppcol, opp_sq, &diff1_opp, &diff2_opp );

    unsigned int g_h1 = ts->ts_hash1 ^ diff1_my ^ diff1_opp;
    unsigned int g_h2 = ts->ts_hash2 ^ diff2_my ^ diff2_opp;
    prefetch_hash_endgame_key( g_h2 );
    HashEntry g_entry;
    find_hash_key( &g_entry, g_h1, g_h2 );

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

static void __attribute__((unused))
sync_board_from_bitboards( BitBoard my_bits, BitBoard opp_bits, int side_to_move, int empties ) {
  BitBoard b_bits = (side_to_move == BLACKSQ ? my_bits : opp_bits);
  BitBoard w_bits = (side_to_move == BLACKSQ ? opp_bits : my_bits);

  board_bits[BLACKSQ] = b_bits;
  board_bits[WHITESQ] = w_bits;
  board_bits[EMPTY] = ~(b_bits | w_bits);

  disks_played = 60 - empties;
  piece_count[BLACKSQ][disks_played] = non_iterative_popcount( b_bits );
  piece_count[WHITESQ][disks_played] = non_iterative_popcount( w_bits );

  determine_pattern_indices_bitboard( b_bits, w_bits );
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
#define split_nesting  (tls.ts_split_nesting)
#define cached_flips   (tls.ts_cached_flips)
#define cached_flipped (tls.ts_cached_flipped)

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

  ThreadState *ts = &tls;
  // Restore thread-local search state for this sibling
  ts->ts_hash1 = sp->sp_hash1;
  ts->ts_hash2 = sp->sp_hash2;
  ts->ts_region_parity = sp->sp_region_parity;
  memcpy( ts->ts_end_move_list, sp->sp_end_move_list, sizeof( ts->ts_end_move_list ) );

  int flipped = TestFlips_wrapper_to( move, my_bits, opp_bits, &new_my_bits );
  if ( flipped == 0 )
    return;

  FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );

  unsigned int diff1, diff2;
  int pred, succ;
  end_make_move( ts, move, new_my_bits, my_bits, sp->side_to_move, &diff1, &diff2, &pred, &succ );

  int child_disc_diff = -sp->disc_diff - 2 * flipped - 1;

  ts->ts_split_nesting++;
  ts->ts_disks_played = 60 - sp->empties;
  score = -end_search_nws( ts, new_opp_bits, new_my_bits,
                           -(cur_alpha + 1),
                           OPP( sp->side_to_move ),
                           sp->empties - 1,
                           child_disc_diff,
                           TRUE,
                           sp->level + 1,
                           sp->selectivity,
                           &child_selective_cutoff );
  ts->ts_split_nesting--;

  end_unmake_move( ts, move, diff1, diff2, pred, succ );

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
dispatch_siblings( ThreadState *ts,
		   BitBoard my_bits, BitBoard opp_bits,
		   int side_to_move, int level, int empties,
		   int disc_diff, int alpha, int beta,
		   int selectivity, int searched_move,
		   const int *best_list, int best_list_length,
		   int *proven, int *proven_score, int *proven_cutoff,
		   int *cutoff_move, int *cutoff_score, int *cutoff_selective ) {
  SplitPoint *sp = ybwc_get_split_point( thread_id, ts->ts_split_nesting );
  int count = 0;
  int used[100];
  int i, sq;
  int stage = ts->ts_disks_played;

  for ( i = 0; i < 100; i++ )
    used[i] = FALSE;
  used[searched_move] = TRUE;

  /* 1. First priority: any remaining move in best_list (e.g. TT move at root) */
  for ( i = 0; i < best_list_length; i++ ) {
    sq = best_list[i];
    if ( (unsigned int) sq < 100 && !used[sq] ) {
      used[sq] = TRUE;
      if ( count < MAX_ROOT_MOVES )
	sp->moves[count++] = sq;
    }
  }

  /* 2. Second priority: remaining legal moves ordered by end_order_moves_presearch */
  {
    int rem_moves[MAX_ROOT_MOVES];
    int rem_count = 0;
    int mcnt = move_count[stage];
    const int *mlist = move_list[stage];
    const int *ev = evals[stage];
    for ( i = best_list_length; i < mcnt; i++ ) {
      sq = mlist[i];
      if ( !used[sq] ) {
	rem_moves[rem_count++] = sq;
	used[sq] = TRUE;
      }
    }
    for ( i = 0; i < rem_count; i++ ) {
      int best_idx = i;
      int best_ev = ev[rem_moves[i]];
      int j;
      for ( j = i + 1; j < rem_count; j++ ) {
	if ( ev[rem_moves[j]] > best_ev ) {
	  best_idx = j;
	  best_ev = ev[rem_moves[j]];
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
  sp->sp_hash1 = ts->ts_hash1;
  sp->sp_hash2 = ts->ts_hash2;
  sp->sp_region_parity = ts->ts_region_parity;
  memcpy( sp->sp_end_move_list, ts->ts_end_move_list, sizeof( ts->ts_end_move_list ) );

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
end_order_moves_presearch( ThreadState *ts,
			   int level,
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
  int stage = ts->ts_disks_played;
  int *ev_stage = evals[stage];
  int *mlist_stage = move_list[stage];
  int mcnt = move_count[stage];

  typedef struct {
    int sq;
    short flipped;
    short opp_mob;
    short demoted;
    int fast_score;
    BitBoard child_my_bits;
    BitBoard new_opp_bits;
    BitBoard opp_moves;
  } CandidateMove;

  CandidateMove candidates[36];
  int candidate_count = 0;

  BitBoard legal_moves = generate_all_c( my_bits, opp_bits );
  if ( *etc_tried_ptr != 0 )
    legal_moves &= ~square_mask[*etc_tried_ptr];
  for ( j = 0; j < best_list_length; j++ ) {
    legal_moves &= ~square_mask[best_list[j]];
  }

  const int *smo = sorted_move_order[stage];
  for ( int shallow_index = 0; shallow_index < MOVE_ORDER_SIZE; shallow_index++ ) {
    if ( legal_moves == 0 )
      break;
    move = smo[shallow_index];
    if ( legal_moves & square_mask[move] ) {
      legal_moves &= ~square_mask[move];
      if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) ) {
	ev_stage[move] = -INFINITE_EVAL;
	mlist_stage[mcnt++] = move;
	continue;
      }
      BitBoard child_my_bits;
      int flipped = TestFlips_wrapper_to( move, my_bits, opp_bits, &child_my_bits );
      if ( flipped > 0 ) {
	BitBoard new_opp_bits;
	FULL_ANDNOT( new_opp_bits, opp_bits, child_my_bits );
	candidates[candidate_count].sq = move;
	candidates[candidate_count].flipped = (short) flipped;
	candidates[candidate_count].opp_mob = -1;
	candidates[candidate_count].demoted = FALSE;
	candidates[candidate_count].child_my_bits = child_my_bits;
	candidates[candidate_count].new_opp_bits = new_opp_bits;
	if ( level <= MAX_SEARCH_DEPTH ) {
	  ts->ts_cached_flipped[level][move] = flipped;
	  ts->ts_cached_flips[level][move] = child_my_bits;
	}
	candidate_count++;
      }
    }
  }

  /* Phase 0: Immediate stability cutoff scan across all candidate moves (pure bitboard table, ~3ns per move) */
  {
    int my_base_cnt = non_iterative_popcount( my_bits );
    EdgeIndices base_edges = { 0 };
    int base_edge_stable = -1;
    for ( i = 0; i < candidate_count; i++ ) {
      int my_cnt = my_base_cnt + 1 + candidates[i].flipped;
      if ( 2 * my_cnt - 64 < beta )
        continue;
      BitBoard child_my_bits = candidates[i].child_my_bits;
      BitBoard new_opp_bits = candidates[i].new_opp_bits;
      if ( ((child_my_bits | new_opp_bits) & CORNER_MASK) == 0 ||
           (child_my_bits & BORDER_MASK) == 0 )
        continue;
      if ( beta > -8 && (child_my_bits & CENTRAL_MASK) == 0 )
        continue;
      EdgeIndices edges;
      int my_edge_stable;
      if ( (square_mask[candidates[i].sq] & BORDER_MASK) == 0 ) {
        if ( base_edge_stable < 0 )
          base_edge_stable = count_edge_stable_indexed( side_to_move, child_my_bits, new_opp_bits, &base_edges );
        edges = base_edges;
        my_edge_stable = base_edge_stable;
      } else {
        my_edge_stable = count_edge_stable_indexed( side_to_move, child_my_bits, new_opp_bits, &edges );
      }
      int lower_bound = 2 * my_edge_stable - 64;
      if ( lower_bound >= beta ) {
        *etc_tried_ptr = candidates[i].sq;
        *etc_cutoff_score = lower_bound;
        return TRUE;
      }
      if ( edges.bits != 0 && (child_my_bits & CENTRAL_MASK) != 0 ) {
        int max_stable = my_edge_stable + non_iterative_popcount( child_my_bits & CENTRAL_MASK );
        if ( 2 * max_stable - 64 >= beta ) {
          int s_full = count_stable_indexed( side_to_move, child_my_bits, new_opp_bits, &edges );
          lower_bound = 2 * s_full - 64;
          if ( lower_bound >= beta ) {
            *etc_tried_ptr = candidates[i].sq;
            *etc_cutoff_score = lower_bound;
            return TRUE;
          }
        }
      }
    }
  }

  /* Pass 1: Lightweight 1-ply ETC scan for candidate moves before static ordering (PRUN-020) */
  if ( use_hash && (empties >= MIN_ETC_DEPTH) ) {
    unsigned int base_h1 = ts->ts_hash1;
    unsigned int base_h2 = ts->ts_hash2;
    for ( i = 0; i < candidate_count; i++ ) {
      move = candidates[i].sq;
      BitBoard child_my_bits = candidates[i].child_my_bits;

      if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) )
	continue;

      unsigned int diff1, diff2;
      end_hash_diff( child_my_bits, my_bits, side_to_move, move, &diff1, &diff2 );
      unsigned int c_h1 = base_h1 ^ diff1;
      unsigned int c_h2 = base_h2 ^ diff2;
      prefetch_hash_endgame_key( c_h2 );
      HashEntry etc_entry;
      find_hash_key( &etc_entry, c_h1, c_h2 );

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
	  candidates[i].demoted = TRUE;
	  etc1_demoted = TRUE;
	}
      }

      if ( !etc1_demoted && selectivity > 0 ) {
	BitBoard new_opp_bits = candidates[i].new_opp_bits;
	BitBoard opp_moves = generate_all_c( new_opp_bits, child_my_bits );
	int opp_mob = non_iterative_popcount( opp_moves );
	candidates[i].opp_moves = opp_moves;
	candidates[i].opp_mob = (short) opp_mob;

	int cutoff_score;
	int child_best_reply = 0;
	if ( (etc_entry.draft != NO_HASH_MOVE) && (etc_entry.move[0] != 0) &&
	     ((unsigned int)etc_entry.move[0] < 100) &&
	     ((opp_moves & square_mask[etc_entry.move[0]]) != 0) ) {
	  child_best_reply = etc_entry.move[0];
	}
	int etc2_res = end_probe_2ply_etc( ts, move, child_my_bits, new_opp_bits,
					   opp_moves, opp_mob,
					   diff1, diff2, side_to_move, empties,
					   curr_alpha, beta, selectivity,
					   child_best_reply, &cutoff_score );
	if ( etc2_res == ETC_2PLY_CUTOFF ) {
	  *etc_tried_ptr = move;
	  *etc_cutoff_score = cutoff_score;
	  return TRUE;
	}
	else if ( etc2_res == ETC_2PLY_REFUTED ) {
	  candidates[i].demoted = TRUE;
	}
      }
    }
  }

  /* Pass 2: Pure bitboard static move ordering with Lazy Pattern Evaluation (ORDR-012) */
  int refuted_moves[36];
  int refuted_count = 0;
  unsigned int effective_parity = ts->ts_region_parity;

  CandidateMove *surv = candidates;
  int surv_count = 0;

  for ( i = 0; i < candidate_count; i++ ) {
    move = candidates[i].sq;
    if ( move == *etc_tried_ptr )
      continue;

    if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) ) {
      ev_stage[move] = -INFINITE_EVAL;
      mlist_stage[mcnt++] = move;
      continue;
    }
    if ( candidates[i].demoted ) {
      if ( beta == curr_alpha + 1 ) {
	refuted_moves[refuted_count++] = move;
	continue;
      }
      ev_stage[move] = -INFINITE_EVAL;
      mlist_stage[mcnt++] = move;
      continue;
    }

    BitBoard child_my_bits = candidates[i].child_my_bits;
    BitBoard new_opp_bits = candidates[i].new_opp_bits;
    BitBoard opp_moves;
    int raw_opp_mob = candidates[i].opp_mob;
    if ( raw_opp_mob < 0 ) {
      opp_moves = generate_all_c( new_opp_bits, child_my_bits );
      raw_opp_mob = non_iterative_popcount( opp_moves );
    } else {
      opp_moves = candidates[i].opp_moves;
    }

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
	  move_count[stage] = mcnt;
	  return TRUE;
	}
	else if ( term_score <= curr_alpha ) {
	  if ( beta == curr_alpha + 1 ) {
	    refuted_moves[refuted_count++] = move;
	    continue;
	  }
	  ev_stage[move] = -INFINITE_EVAL;
	  mlist_stage[mcnt++] = move;
	  continue;
	}
      }
      else if ( use_hash && (empties >= MIN_ETC_DEPTH) && (selectivity == 0) ) {
	unsigned int diff1, diff2;
	end_hash_diff( child_my_bits, my_bits, side_to_move, move, &diff1, &diff2 );
	HashEntry p_entry;
	find_hash_key( &p_entry, ts->ts_hash1 ^ diff1 ^ hash_flip_color1,
		       ts->ts_hash2 ^ diff2 ^ hash_flip_color2 );
	if ( (p_entry.flags & ENDGAME_SCORE) &&
	     (p_entry.draft >= empties - 1) &&
	     (p_entry.selectivity == 0) ) {
	  if ( (p_entry.flags & (EXACT_VALUE | LOWER_BOUND)) && p_entry.eval >= beta ) {
	    *etc_tried_ptr = move;
	    *etc_cutoff_score = p_entry.eval;
	    move_count[stage] = mcnt;
	    return TRUE;
	  }
	  if ( (p_entry.flags & (EXACT_VALUE | UPPER_BOUND)) && p_entry.eval <= curr_alpha ) {
	    if ( beta == curr_alpha + 1 ) {
	      refuted_moves[refuted_count++] = move;
	      continue;
	    }
	    ev_stage[move] = -INFINITE_EVAL;
	    mlist_stage[mcnt++] = move;
	    continue;
	  }
	}
      }
    }

    int opp_corner_moves = non_iterative_popcount( opp_moves & 0x8100000000000081ull );

    int c_square_penalty = 0;
    if ( (square_mask[move] & 0x42C300000000C342ull) != 0 ) {
      int bit = bit_position[move];
      int corner_bit = ((bit & 7) < 4 ? 0 : 7) + (bit < 32 ? 0 : 56);
      if ( ((child_my_bits | new_opp_bits) & (1ull << corner_bit)) == 0 ) {
        if ( (square_mask[move] & 0x0042000000004200ull) != 0 ||
             (opp_moves & (1ull << corner_bit)) != 0 ) {
          c_square_penalty = 128;
        }
      }
    }

    int quadrant_parity = (quadrant_mask[move] & effective_parity) != 0;
    int fast_score = - 128 * (raw_opp_mob + opp_corner_moves) -
		     c_square_penalty +
		     (raw_opp_mob == 0 ? 4096 : 0) +
		     (quadrant_parity ? REGION_PARITY_BONUS : 0);

    surv[surv_count].sq = move;
    surv[surv_count].fast_score = fast_score;
    surv[surv_count].child_my_bits = child_my_bits;
    surv[surv_count].new_opp_bits = new_opp_bits;
    surv[surv_count].opp_moves = opp_moves;
    surv_count++;
  }

  if ( surv_count == 1 ) {
    /* Fast-path for single legal move: no pattern evaluation needed */
    int m = surv[0].sq;
    ev_stage[m] = surv[0].fast_score;
    mlist_stage[mcnt++] = m;
  }
  else if ( surv_count > 1 ) {
    /* Sort surviving moves by fast_score descending (insertion sort for <= 32 elements) */
    for ( i = 1; i < surv_count; i++ ) {
      CandidateMove tmp = surv[i];
      j = i - 1;
      while ( j >= 0 && surv[j].fast_score < tmp.fast_score ) {
        surv[j + 1] = surv[j];
        j--;
      }
      surv[j + 1] = tmp;
    }

    /* Branch & Bound Pattern Evaluation:
       Evaluate candidate 0, then only evaluate subsequent candidates within LAZY_PATTERN_MARGIN
       (or corner moves, whose high pattern value can overcome temporary opponent mobility) */
    int top_fast = surv[0].fast_score;
    for ( i = 0; i < surv_count; i++ ) {
      int m = surv[i].sq;
      if ( i == 0 || (top_fast - surv[i].fast_score <= LAZY_PATTERN_MARGIN) ||
	   (square_mask[m] & CORNER_MASK) != 0 ) {
        int p_score;
        if ( empties <= FAST_PATTERN_DEPTH ) {
          p_score = end_pattern_evaluate_c33_r24( surv[i].child_my_bits, surv[i].new_opp_bits ) / 2;
        } else {
          p_score = end_pattern_evaluate( surv[i].child_my_bits, surv[i].new_opp_bits );
        }
        ev_stage[m] = p_score + surv[i].fast_score;
      }
      else {
        /* Pruned candidate: assign fast_score with offset to rank strictly behind evaluated candidates */
        ev_stage[m] = surv[i].fast_score - 10000;
      }
      mlist_stage[mcnt++] = m;
    }

    /* Selective shallow tree_search deepening on top-K candidates (SRCH-005)
       Evaluates shallow lookahead for the top surviving candidates at deep endgame
       nodes (empties >= ORDER_TS_MIN_DEPTH), providing tactical lookahead to avoid
       combinatorial tree explosion during both selective passes and exact solve. */
    if ( empties >= ORDER_TS_MIN_DEPTH ) {
      /* Sort surv by evals descending so top candidates can be deepened */
      for ( i = 1; i < surv_count; i++ ) {
        CandidateMove tmp = surv[i];
        int m_tmp = tmp.sq;
        int ev_tmp = ev_stage[m_tmp];
        int k = i - 1;
        while ( k >= 0 && ev_stage[surv[k].sq] < ev_tmp ) {
          surv[k + 1] = surv[k];
          k--;
        }
        surv[k + 1] = tmp;
      }

      int d = (empties < 18) ? 2 : (empties < 24) ? 3 : (empties < 28) ? 4 : 6;
      if ( best_list_length == 0 &&
           ((empties >= 20 && empties <= 22) ||
            (empties == 23 && level >= 2)) )
        d += 1;
      int upper = (-curr_alpha + 8) * 128;
      int ff_thr = 128 * curr_alpha + (int) ceil( (end_mean[stage][d] + 0.45 * end_sigma[stage][d]) * 128.0 );
      int k_limit = (ORDER_TS_TOPK < surv_count) ? ORDER_TS_TOPK : surv_count;
      int any_promoted = 0;
      int v0 = 0;
      BitBoard b_parent = (side_to_move == BLACKSQ) ? my_bits : opp_bits;
      BitBoard w_parent = (side_to_move == BLACKSQ) ? opp_bits : my_bits;
      int parent_b_cnt = non_iterative_popcount( b_parent );
      int parent_w_cnt = non_iterative_popcount( w_parent );
      unsigned short saved_parent_pi[48] __attribute__((aligned(16)));
      determine_pattern_indices_bitboard( b_parent, w_parent );
      memcpy( saved_parent_pi, eval_pattern_index, sizeof( saved_parent_pi ) );
      for ( i = 0; i < k_limit; i++ ) {
        int m = surv[i].sq;
        unsigned int saved_parity = ts->ts_region_parity;
        ts->ts_region_parity ^= quadrant_mask[m];
        unsigned int diff1, diff2;
        end_hash_diff( surv[i].child_my_bits, my_bits, side_to_move, m, &diff1, &diff2 );
        ts->ts_hash1 ^= diff1;
        ts->ts_hash2 ^= diff2;
        {
          int flipped_cnt = (int) surv[i].flipped;
          BitBoard b_child = (side_to_move == BLACKSQ) ? surv[i].child_my_bits : surv[i].new_opp_bits;
          BitBoard w_child = (side_to_move == BLACKSQ) ? surv[i].new_opp_bits : surv[i].child_my_bits;
          board_bits[BLACKSQ] = b_child;
          board_bits[WHITESQ] = w_child;
          board_bits[EMPTY] = ~(b_child | w_child);
          disks_played = 60 - (empties - 1);
          piece_count[BLACKSQ][disks_played] = (side_to_move == BLACKSQ) ? (parent_b_cnt + flipped_cnt + 1) : (parent_b_cnt - flipped_cnt);
          piece_count[WHITESQ][disks_played] = (side_to_move == WHITESQ) ? (parent_w_cnt + flipped_cnt + 1) : (parent_w_cnt - flipped_cnt);
          memcpy( eval_pattern_index, saved_parent_pi, sizeof( saved_parent_pi ) );
          update_pattern_indices( side_to_move, m, (surv[i].child_my_bits ^ my_bits) & ~square_mask[m], 1 );
        }
        int v = -tree_search( level + 1, level + d, OPP( side_to_move ),
                              -INFINITE_EVAL, upper, TRUE, TRUE, TRUE );
        ts->ts_hash1 ^= diff1;
        ts->ts_hash2 ^= diff2;
        ts->ts_disks_played = 60 - empties;
        ts->ts_region_parity = saved_parity;
        if ( i == 0 )
          v0 = v;

        if ( v > ff_thr ) {
          int mob = non_iterative_popcount( surv[i].opp_moves );
          ev_stage[m] = 1000000 + v - 460 * mob;
          any_promoted = 1;
        }
        else if ( i > 0 && !any_promoted && (empties < 20 || curr_alpha >= 18) && v > v0 + 256 ) {
          ev_stage[m] = ev_stage[surv[0].sq] + (v - v0);
        }
        if ( i == 1 && !any_promoted && surv_count >= 3 && d <= 4 )
          k_limit = 3;
      }
    }
  }

  if ( beta == curr_alpha + 1 && mcnt == 0 && best_list_length == 0 && refuted_count > 0 ) {
    int fb_move = refuted_moves[0];
    ev_stage[fb_move] = -INFINITE_EVAL;
    mlist_stage[mcnt++] = fb_move;
  }

  move_count[stage] = mcnt;
  return FALSE;
}


/*
  END_SEARCH_NWS
  Dedicated Null-Window Search (NWS) core.
  Optimized for Cut and All nodes (beta == alpha + 1).
  Bypasses PV tracking, re-searches, and UI sweep updates.
*/

static int
end_search_nws( ThreadState *ts,
		BitBoard my_bits,
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

  int opp_stable_checked = (pass_legal == 2);
  if ( pass_legal )
    pass_legal = TRUE;

  *selective_cutoff = FALSE;

  if ( SPLIT_ABANDONED() )
    return SEARCH_ABORT;

  /* 1. Terminal leaf dispatch (no PV tracking needed in NWS) */
  if ( empties <= LOW_LEVEL_DEPTH ) {
    return solve_parity( ts, my_bits, opp_bits, alpha, beta, side_to_move,
			 empties, disc_diff, pass_legal, level );
  }

  ts->ts_nodes.lo++;

  /* 2. Transposition table probing */
  use_hash = USE_HASH_TABLE;

  if ( use_hash ) {
    find_hash_key( &entry, ts->ts_hash1, ts->ts_hash2 );
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
	       (entry.flags & ENDGAME_SCORE) &&
	       !(entry.flags & HEURISTIC_PRESEARCH_MOVE);
  }

  /* 3. Symmetric stability bounds check */
#if USE_STABILITY
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 ) {
    if ( !opp_stable_checked && (opp_bits & BORDER_MASK) != 0 &&
         (alpha >= 8 || (opp_bits & CENTRAL_MASK) != 0) ) {
      int opp_cnt = non_iterative_popcount( opp_bits );
      int min_upper = 64 - 2 * opp_cnt;
      if ( min_upper <= alpha ) {
        EdgeIndices edges;
        int s_edge = count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        int upper_bound = 64 - 2 * s_edge;
        if ( upper_bound <= alpha ) {
          return alpha;
        }
        if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
          int max_stable = s_edge + non_iterative_popcount( opp_bits & CENTRAL_MASK );
          if ( 64 - 2 * max_stable <= alpha ) {
            int s_full = count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
            upper_bound = 64 - 2 * s_full;
            if ( upper_bound <= alpha ) {
              return alpha;
            }
          }
        }
      }
    }

    if ( (my_bits & BORDER_MASK) != 0 &&
         (beta <= -8 || (my_bits & CENTRAL_MASK) != 0) ) {
      int my_cnt = non_iterative_popcount( my_bits );
      int max_lower = 2 * my_cnt - 64;
      if ( max_lower >= beta ) {
        EdgeIndices edges;
        int s_edge = count_edge_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
        int lower_bound = 2 * s_edge - 64;
        if ( lower_bound >= beta ) {
          return lower_bound;
        }
        if ( edges.bits != 0 && (my_bits & CENTRAL_MASK) != 0 ) {
          int max_stable = s_edge + non_iterative_popcount( my_bits & CENTRAL_MASK );
          if ( 2 * max_stable - 64 >= beta ) {
            int s_full = count_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
            lower_bound = 2 * s_full - 64;
            if ( lower_bound >= beta ) {
              return lower_bound;
            }
          }
        }
      }
    }
  }
#endif

  /* 4. Setup and MPC */
  int i;
  int move;
  int move_index;
  int first;
  int pre_search_done, etc_tried;
  int hash_move, best_move, best_list_length;
  int proven[100], proven_score[100], proven_cutoff[100];
  int siblings_dispatched = FALSE;
  int can_split;
  int best;
  int curr_val;
  int saved_disks_played = ts->ts_disks_played;
  int stage = 60 - empties;

  ts->ts_disks_played = stage;

  if ( USE_MPC && (level > 2) && (selectivity > 0) && (use_end_cut[stage] > 0) ) {
    int cut;
    sync_board_from_bitboards( my_bits, opp_bits, side_to_move, empties );
    for ( cut = 0; cut < use_end_cut[stage]; cut++ ) {
      int shallow_remains = end_mpc_depth[stage][cut];
      int mpc_bias = ceil( end_mean[stage][shallow_remains] * 128.0 );
      int mpc_window = ceil( end_sigma[stage][shallow_remains] *
			     end_percentile[selectivity] * 128.0 );
      int beta_bound = 128 * beta + mpc_bias + mpc_window;
      int alpha_bound = 128 * alpha + mpc_bias - mpc_window;
      int shallow_val =
	tree_search( level, level + shallow_remains, side_to_move,
		     alpha_bound, beta_bound, use_hash, FALSE, pass_legal );
      if ( shallow_val >= beta_bound ) {
	if ( use_hash )
	  add_hash_ts( ts, ENDGAME_MODE, beta, 0,
		       ENDGAME_SCORE | LOWER_BOUND, empties, selectivity );
	*selective_cutoff = TRUE;
	ts->ts_disks_played = saved_disks_played;
	return beta;
      }
      if ( shallow_val <= alpha_bound ) {
	if ( use_hash )
	  add_hash_ts( ts, ENDGAME_MODE, alpha, 0,
		       ENDGAME_SCORE | UPPER_BOUND, empties, selectivity );
	*selective_cutoff = TRUE;
	ts->ts_disks_played = saved_disks_played;
	return alpha;
      }
    }
  }

  first = TRUE;
  can_split = FALSE;
  best = -INFINITE_EVAL;
  pre_search_done = FALSE;
  etc_tried = 0;

  /* Initialize move list and check hash table move */
  hash_move = hash_hit ? entry.move[0] : 0;
  best_move = hash_move;
  best_list_length = hash_hit ? 1 : 0;
  move_count[stage] = best_list_length;

  /* 5. NWS Move loop */
  for ( move_index = 0; TRUE; move_index++ ) {
    int child_selective_cutoff;
    BitBoard new_my_bits;
    BitBoard new_opp_bits;

    if ( move_index < best_list_length ) {
      move = hash_move;
    }
    else {
      if ( !pre_search_done ) {
	int etc_cutoff_score = 0;
	if ( end_order_moves_presearch( ts, level, empties, side_to_move,
					my_bits, opp_bits, beta, alpha,
					selectivity, use_hash,
					&hash_move, best_list_length,
					can_split, proven, proven_score,
					&etc_tried,
					&etc_cutoff_score ) ) {
	  if ( selectivity > 0 )
	    *selective_cutoff = TRUE;
	  if ( use_hash )
	    add_hash_ts( ts, ENDGAME_MODE, etc_cutoff_score, etc_tried,
			 ENDGAME_SCORE | LOWER_BOUND, empties,
			 *selective_cutoff ? selectivity : 0 );
	  ts->ts_disks_played = saved_disks_played;
	  return etc_cutoff_score;
	}
	pre_search_done = TRUE;
      }

      if ( move_index == move_count[stage] )
	break;
      move = select_move( move_index, move_count[stage] );
    }

    double node_val = counter_value( &ts->ts_nodes );
    if ( node_val - ts->ts_last_panic_check >= EVENT_CHECK_INTERVAL ) {
      ts->ts_last_panic_check = node_val;
      check_panic_abort();
      if ( echo )
	display_buffers();
      handle_event( TRUE, FALSE, TRUE );
      if ( is_panic_abort() || force_return ) {
	ts->ts_disks_played = saved_disks_played;
	return SEARCH_ABORT;
      }
    }

    if ( can_split && proven[move] && (proven_score[move] <= alpha) ) {
      curr_val = proven_score[move];
      child_selective_cutoff = proven_cutoff[move];
    }
    else {
      unsigned int diff1, diff2;
      int pred, succ;
      int flipped;
      if ( pre_search_done && level <= MAX_SEARCH_DEPTH ) {
        flipped = ts->ts_cached_flipped[level][move];
        new_my_bits = ts->ts_cached_flips[level][move];
      }
      else {
        flipped = TestFlips_wrapper_to( move, my_bits, opp_bits, &new_my_bits );
      }
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );

      end_make_move( ts, move, new_my_bits, my_bits, side_to_move, &diff1, &diff2, &pred, &succ );

      int new_disc_diff = -disc_diff - 2 * flipped - 1;

      curr_val = -end_search_nws( ts, new_opp_bits, new_my_bits,
				  -beta, OPP( side_to_move ),
				  empties - 1, new_disc_diff,
				  (pre_search_done && level <= MAX_SEARCH_DEPTH) ? 2 : TRUE,
				  level + 1,
				  selectivity, &child_selective_cutoff );

      end_unmake_move( ts, move, diff1, diff2, pred, succ );
    }

    if ( abs( curr_val ) >= 20000 || is_panic_abort() || force_return || SPLIT_ABANDONED() ) {
      ts->ts_disks_played = saved_disks_played;
      return SEARCH_ABORT;
    }

    /* Cutoff check in NWS: any score >= beta causes an immediate cutoff */
    if ( curr_val >= beta ) {
      *selective_cutoff = child_selective_cutoff;
      if ( use_hash )
	add_hash_ts( ts, ENDGAME_MODE, curr_val, move,
		     ENDGAME_SCORE | LOWER_BOUND, empties,
		     *selective_cutoff ? selectivity : 0 );
      ts->ts_disks_played = saved_disks_played;
      return curr_val;
    }

    if ( curr_val > best )
      best = curr_val;
    if ( child_selective_cutoff )
      *selective_cutoff = TRUE;

    if ( first && best_move == 0 )
      best_move = move;

    if ( !siblings_dispatched && first &&
	 (empties >= PARALLEL_SPLIT_DEPTH + SPLIT_NESTING_MARGIN * ts->ts_split_nesting) &&
	 (ts->ts_split_nesting <= MAX_SPLIT_NESTING) && (threads_count() > 1) &&
	 (threads_idle_count() > 0) &&
	 !is_panic_abort() && !force_return ) {
      if ( !pre_search_done ) {
	int etc_cutoff_score = 0;
	if ( end_order_moves_presearch( ts, level, empties, side_to_move,
					my_bits, opp_bits, beta, alpha,
					selectivity, use_hash,
					&hash_move, best_list_length,
					FALSE, proven, proven_score,
					&etc_tried,
					&etc_cutoff_score ) ) {
	  if ( selectivity > 0 )
	    *selective_cutoff = TRUE;
	  if ( use_hash )
	    add_hash_ts( ts, ENDGAME_MODE, etc_cutoff_score, etc_tried,
			 ENDGAME_SCORE | LOWER_BOUND, empties,
			 *selective_cutoff ? selectivity : 0 );
	  ts->ts_disks_played = saved_disks_played;
	  return etc_cutoff_score;
	}
	pre_search_done = TRUE;
      }
      can_split = TRUE;
      for ( i = 0; i < 100; i++ )
	proven[i] = FALSE;
      siblings_dispatched = TRUE;
      int cut_move = 0, cut_score = 0, cut_selective = 0;
      if ( dispatch_siblings( ts, my_bits, opp_bits, side_to_move, level,
			      empties, disc_diff, alpha, beta, selectivity, move,
			      &hash_move, best_list_length,
			      proven, proven_score, proven_cutoff,
			      &cut_move, &cut_score, &cut_selective ) ) {
	*selective_cutoff = cut_selective;
	if ( use_hash )
	  add_hash_ts( ts, ENDGAME_MODE, cut_score, cut_move,
		       ENDGAME_SCORE | LOWER_BOUND, empties,
		       *selective_cutoff ? selectivity : 0 );
	ts->ts_disks_played = saved_disks_played;
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
      add_hash_ts( ts, ENDGAME_MODE, best, best_move,
		   ENDGAME_SCORE | UPPER_BOUND, empties,
		   *selective_cutoff ? selectivity : 0 );
    }
    ts->ts_disks_played = saved_disks_played;
    return best;
  }
  else if ( pass_legal ) {
    if ( use_hash ) {
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
    }
    curr_val = -end_search_nws( ts, opp_bits, my_bits,
				-beta, OPP( side_to_move ),
				empties, -disc_diff, FALSE, level,
				selectivity, selective_cutoff );
    if ( use_hash ) {
      ts->ts_hash1 ^= hash_flip_color1;
      ts->ts_hash2 ^= hash_flip_color2;
    }
    ts->ts_disks_played = saved_disks_played;
    return curr_val;
  }
  else {
    ts->ts_disks_played = saved_disks_played;
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
end_search_pvs( ThreadState *ts,
		BitBoard my_bits,
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
  char buffer[16];
  HashEntry entry;
  int oppcol = OPP( side_to_move );
  int use_hash;
  int hash_hit = FALSE;

  *selective_cutoff = FALSE;

  if ( beta == alpha + 1 && level > 0 ) {
    return end_search_nws( ts, my_bits, opp_bits, alpha, side_to_move,
			   empties, disc_diff, pass_legal, level,
			   selectivity, selective_cutoff );
  }

  if ( SPLIT_ABANDONED() )
    return SEARCH_ABORT;

  /* 1. Terminal leaf dispatch */
  if ( empties <= LOW_LEVEL_DEPTH ) {
    int res = solve_parity( ts, my_bits, opp_bits, alpha, beta, side_to_move,
			    empties, disc_diff, pass_legal, level );
    pv_depth[level] = level + 1;
    pv[level][level] = ts->ts_end_best_move;
    if ( level == 0 )
      ts->ts_end_best_root_move = ts->ts_end_best_move;
    return res;
  }

  ts->ts_nodes.lo++;

  /* 2. Symmetric stability bounds check */
#if USE_STABILITY
  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 ) {
    if ( (opp_bits & BORDER_MASK) != 0 ) {
      int opp_cnt = non_iterative_popcount( opp_bits );
      int min_upper = 64 - 2 * opp_cnt;
      if ( min_upper <= alpha || min_upper < beta ) {
        EdgeIndices edges;
        int s_edge = count_edge_stable_indexed( oppcol, opp_bits, my_bits, &edges );
        int upper_bound = 64 - 2 * s_edge;
        if ( upper_bound <= alpha ) {
          pv_depth[level] = level;
          return alpha;
        }
        if ( upper_bound < beta )
          beta = upper_bound + 1;
        if ( edges.bits != 0 && (opp_bits & CENTRAL_MASK) != 0 ) {
          int s_full = count_stable_indexed( oppcol, opp_bits, my_bits, &edges );
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
        int lower_bound = 2 * s_edge - 64;
        if ( lower_bound >= beta ) {
          pv_depth[level] = level;
          return lower_bound;
        }
        if ( lower_bound > alpha )
          alpha = lower_bound;
        if ( edges.bits != 0 && (my_bits & CENTRAL_MASK) != 0 ) {
          int s_full = count_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
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
    find_hash_key( &entry, ts->ts_hash1, ts->ts_hash2 );
    if ( (entry.draft == empties) &&
	 (entry.selectivity <= selectivity) &&
	 bb_valid_move( entry.move[0], my_bits, opp_bits ) &&
	 (entry.flags & ENDGAME_SCORE) &&
	 ((entry.flags & EXACT_VALUE) ||
	  ((entry.flags & LOWER_BOUND) && entry.eval >= beta) ||
	  ((entry.flags & UPPER_BOUND) && entry.eval <= alpha)) ) {
      ts->ts_end_best_move = entry.move[0];
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
	       (entry.flags & ENDGAME_SCORE) &&
	       !(entry.flags & HEURISTIC_PRESEARCH_MOVE);
  }

  /* 5. Recursive PVS search loop */
  double node_val;
  int i;
  int move;
  int move_index;
  int update_pv, first;
  int curr_alpha;
  int pre_search_done, etc_tried;
  int best_list_length, best_move;
  int best_list[2];
  int proven[100], proven_score[100], proven_cutoff[100];
  int siblings_dispatched = FALSE;
  int can_split;
  int best;
  int curr_val;
  int saved_disks_played = ts->ts_disks_played;
  int stage = 60 - empties;

  ts->ts_disks_played = stage;

  /* Use endgame multi-prob-cut to selectively prune the tree */
  if ( USE_MPC && (level > 2) && (selectivity > 0) && (use_end_cut[stage] > 0) ) {
    int cut;
    sync_board_from_bitboards( my_bits, opp_bits, side_to_move, empties );
    for ( cut = 0; cut < use_end_cut[stage]; cut++ ) {
	int shallow_remains = end_mpc_depth[stage][cut];
	int mpc_bias = ceil( end_mean[stage][shallow_remains] * 128.0 );
	int mpc_window = ceil( end_sigma[stage][shallow_remains] *
			       end_percentile[selectivity] * 128.0 );
	int beta_bound = 128 * beta + mpc_bias + mpc_window;
	int alpha_bound = 128 * alpha + mpc_bias - mpc_window;
	int shallow_val =
	  tree_search( level, level + shallow_remains, side_to_move,
		       alpha_bound, beta_bound, use_hash, FALSE, pass_legal );
	if ( shallow_val >= beta_bound ) {
	  if ( use_hash )
	    add_hash_ts( ts, ENDGAME_MODE, beta, 0,
			 ENDGAME_SCORE | LOWER_BOUND, empties, selectivity );
	  *selective_cutoff = TRUE;
	  ts->ts_disks_played = saved_disks_played;
	  return beta;
	}
	if ( shallow_val <= alpha_bound ) {
	  if ( use_hash )
	    add_hash_ts( ts, ENDGAME_MODE, alpha, 0,
			 ENDGAME_SCORE | UPPER_BOUND, empties, selectivity );
	  *selective_cutoff = TRUE;
	  ts->ts_disks_played = saved_disks_played;
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

  /* Initialize move list and check root/hash priority moves */
  best_list_length = 0;
  if ( level == 0 && ts->ts_end_best_root_move != 0 &&
       bb_valid_move( ts->ts_end_best_root_move, my_bits, opp_bits ) ) {
    best_list[best_list_length++] = ts->ts_end_best_root_move;
  }
  if ( hash_hit && (best_list_length == 0 || entry.move[0] != best_list[0]) ) {
    best_list[best_list_length++] = entry.move[0];
  }
  best_move = (best_list_length > 0) ? best_list[0] : 0;
  move_count[stage] = best_list_length;

  for ( move_index = 0; TRUE; move_index++ ) {
    int child_selective_cutoff;
    BitBoard new_my_bits;
    BitBoard new_opp_bits;

    if ( move_index < best_list_length ) {
	move = best_list[move_index];
    }
    else {
	if ( !pre_search_done ) {
	  int etc_cutoff_score = 0;
	  if ( end_order_moves_presearch( ts, level, empties, side_to_move,
					  my_bits, opp_bits, beta, curr_alpha,
					  selectivity, use_hash,
					  best_list, best_list_length,
					  can_split, proven, proven_score,
					  &etc_tried,
					  &etc_cutoff_score ) ) {
	    if ( selectivity > 0 )
	      *selective_cutoff = TRUE;
	    if ( use_hash )
	      add_hash_ts( ts, ENDGAME_MODE, etc_cutoff_score, etc_tried,
			   ENDGAME_SCORE | LOWER_BOUND, empties,
			   *selective_cutoff ? selectivity : 0 );
	    pv_depth[level] = level + 1;
	    pv[level][level] = etc_tried;
	    if ( level == 0 )
	      ts->ts_end_best_root_move = etc_tried;
	    ts->ts_disks_played = saved_disks_played;
	    return etc_cutoff_score;
	  }
	  pre_search_done = TRUE;
	}

	if ( move_index == move_count[stage] )
	  break;
	move = select_move( move_index, move_count[stage] );
    }

    node_val = counter_value( &ts->ts_nodes );
    if ( node_val - ts->ts_last_panic_check >= EVENT_CHECK_INTERVAL ) {
	ts->ts_last_panic_check = node_val;
	check_panic_abort();
	if ( echo )
	  display_buffers();
	handle_event( TRUE, FALSE, TRUE );
	if ( is_panic_abort() || force_return ) {
	  ts->ts_disks_played = saved_disks_played;
	  return SEARCH_ABORT;
	}
    }

    if ( (level == 0) && !get_ponder_move() ) {
	if ( first )
	  send_sweep( "%-10s ", buffer );
	send_sweep( "%c%c", TO_SQUARE( move ) );
    }

    update_pv = FALSE;
    if ( !first && can_split && proven[move] && (proven_score[move] <= MAX( best, curr_alpha )) ) {
      curr_alpha = MAX( best, curr_alpha );
      curr_val = proven_score[move];
      if ( curr_val > best )
	best = curr_val;
      child_selective_cutoff = proven_cutoff[move];
    }
    else {
      unsigned int diff1, diff2;
      int pred, succ;
      int flipped;
      if ( pre_search_done && level <= MAX_SEARCH_DEPTH ) {
        flipped = ts->ts_cached_flipped[level][move];
        new_my_bits = ts->ts_cached_flips[level][move];
      }
      else {
        flipped = TestFlips_wrapper_to( move, my_bits, opp_bits, &new_my_bits );
      }
      FULL_ANDNOT( new_opp_bits, opp_bits, new_my_bits );

      end_make_move( ts, move, new_my_bits, my_bits, side_to_move, &diff1, &diff2, &pred, &succ );

      int new_disc_diff = -disc_diff - 2 * flipped - 1;

      if ( first ) {
	best = curr_val =
	  -end_search_pvs( ts, new_opp_bits, new_my_bits,
			   -beta, -curr_alpha, OPP( side_to_move ),
			   empties - 1, new_disc_diff, TRUE, level + 1,
			   selectivity, &child_selective_cutoff );
	update_pv = TRUE;
	if ( level == 0 )
	  ts->ts_end_best_root_move = move;
	curr_alpha = MAX( best, curr_alpha );
      }
      else {
	curr_alpha = MAX( best, curr_alpha );
	curr_val =
	  -end_search_nws( ts, new_opp_bits, new_my_bits,
			   -(curr_alpha + 1), OPP( side_to_move ),
			   empties - 1, new_disc_diff,
			   (pre_search_done && level <= MAX_SEARCH_DEPTH) ? 2 : TRUE,
			   level + 1,
			   selectivity, &child_selective_cutoff );

	if ( (curr_val > curr_alpha) && (curr_val < beta) ) {
	  if ( selectivity > 0 )
	    curr_val =
	      -end_search_pvs( ts, new_opp_bits, new_my_bits,
			       -beta, INFINITE_EVAL, OPP( side_to_move ),
			       empties - 1, new_disc_diff, TRUE, level + 1,
			       selectivity, &child_selective_cutoff );
	  else
	    curr_val =
	      -end_search_pvs( ts, new_opp_bits, new_my_bits,
			       -beta, -curr_val, OPP( side_to_move ),
			       empties - 1, new_disc_diff, TRUE, level + 1,
			       selectivity, &child_selective_cutoff );
	  if ( curr_val > best ) {
	    best = curr_val;
	    update_pv = TRUE;
	    if ( (level == 0) && !is_panic_abort() && !force_return )
	      ts->ts_end_best_root_move = move;
	  }
	}
	else if ( curr_val > best ) {
	  best = curr_val;
	  update_pv = TRUE;
	  if ( (level == 0) && !is_panic_abort() && !force_return )
	    ts->ts_end_best_root_move = move;
	}
      }

      end_unmake_move( ts, move, diff1, diff2, pred, succ );
    }

    if ( best >= beta )
	*selective_cutoff = child_selective_cutoff;
    else if ( child_selective_cutoff )
	*selective_cutoff = TRUE;

    if ( abs( curr_val ) >= 20000 || is_panic_abort() || force_return || SPLIT_ABANDONED() ) {
	ts->ts_disks_played = saved_disks_played;
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
	best_move = move;
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
	  add_hash_ts( ts, ENDGAME_MODE, best, best_move,
		       ENDGAME_SCORE | LOWER_BOUND, empties,
		       *selective_cutoff ? selectivity : 0 );
	ts->ts_disks_played = saved_disks_played;
	return best;
    }

    if ( !siblings_dispatched && first &&
	 (empties >= PARALLEL_SPLIT_DEPTH + SPLIT_NESTING_MARGIN * ts->ts_split_nesting) &&
	 (ts->ts_split_nesting <= MAX_SPLIT_NESTING) && (threads_count() > 1) &&
	 (threads_idle_count() > 0) &&
	 !is_panic_abort() && !force_return ) {
      if ( !pre_search_done ) {
	int etc_cutoff_score = 0;
	if ( end_order_moves_presearch( ts, level, empties, side_to_move,
					my_bits, opp_bits, beta, curr_alpha,
					selectivity, use_hash,
					best_list, best_list_length,
					FALSE, proven, proven_score,
					&etc_tried,
					&etc_cutoff_score ) ) {
	  if ( selectivity > 0 )
	    *selective_cutoff = TRUE;
	  if ( use_hash )
	    add_hash_ts( ts, ENDGAME_MODE, etc_cutoff_score, etc_tried,
			 ENDGAME_SCORE | LOWER_BOUND, empties,
			 *selective_cutoff ? selectivity : 0 );
	  pv_depth[level] = level + 1;
	  pv[level][level] = etc_tried;
	  if ( level == 0 )
	    ts->ts_end_best_root_move = etc_tried;
	  ts->ts_disks_played = saved_disks_played;
	  return etc_cutoff_score;
	}
	pre_search_done = TRUE;
      }
      can_split = TRUE;
      for ( i = 0; i < 100; i++ )
	proven[i] = FALSE;
      siblings_dispatched = TRUE;
      int cut_move = 0, cut_score = 0, cut_selective = 0;
      if ( dispatch_siblings( ts, my_bits, opp_bits, side_to_move, level,
			      empties, disc_diff, curr_alpha, beta, selectivity, move,
			      best_list, best_list_length,
			      proven, proven_score, proven_cutoff,
			      &cut_move, &cut_score, &cut_selective ) ) {
	best = cut_score;
	*selective_cutoff = cut_selective;
	best_move = cut_move;
	pv[level][level] = cut_move;
	pv_depth[level] = level + 1;
	if ( level == 0 )
	  ts->ts_end_best_root_move = cut_move;
	if ( use_hash )
	  add_hash_ts( ts, ENDGAME_MODE, best, cut_move,
		       ENDGAME_SCORE | LOWER_BOUND, empties,
		       *selective_cutoff ? selectivity : 0 );
	ts->ts_disks_played = saved_disks_played;
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
	add_hash_ts( ts, ENDGAME_MODE, best, best_move, flags, empties,
		     *selective_cutoff ? selectivity : 0 );
    }
    ts->ts_disks_played = saved_disks_played;
    return best;
  }
  else if ( pass_legal ) {
    if ( use_hash ) {
	ts->ts_hash1 ^= hash_flip_color1;
	ts->ts_hash2 ^= hash_flip_color2;
    }
    curr_val = -end_search_pvs( ts, opp_bits, my_bits,
				  -beta, -alpha, OPP( side_to_move ),
				  empties, -disc_diff, FALSE, level,
				  selectivity, selective_cutoff );
    if ( use_hash ) {
	ts->ts_hash1 ^= hash_flip_color1;
	ts->ts_hash2 ^= hash_flip_color2;
    }
    ts->ts_disks_played = saved_disks_played;
    return curr_val;
  }
  else {
    pv_depth[level] = level;
    ts->ts_disks_played = saved_disks_played;
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
  ThreadState *ts = &tls;

  set_bitboards( ts->ts_board, side_to_move, &my_bits, &opp_bits );

  prepare_to_solve( my_bits | opp_bits );

  int my_discs = non_iterative_popcount( my_bits );
  int opp_discs = non_iterative_popcount( opp_bits );
  int empties = 64 - my_discs - opp_discs;
  int disc_diff = my_discs - opp_discs;

  int result = end_search_pvs( ts, my_bits, opp_bits,
			       MAX( alpha - komi_shift, -64 ),
			       MIN( beta - komi_shift, 64 ),
			       side_to_move,
			       empties,
			       disc_diff,
			       void_legal,
			       level,
			       selectivity,
			       &selective_cutoff );
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

  /* Allow midgame pre-searches to read/write TT (ENDGAME_SCORE entries
     are strictly protected against MIDGAME_SCORE overwrites in hash.c) */

  toggle_midgame_hash_usage( TRUE, TRUE );

  incomplete_search = FALSE;
  any_search_result = FALSE;

  /* Start off by selective endgame search */

  last_window_center = 0;

  if ( !wld && (empties >= 16) ) {
    int stage = 60 - empties;
    int saved_pv00 = pv[0][0];
    int ts_val = 0, ts_val_6 = 0;
    set_midgame_full_width_depth( 4 );
    for ( int id_d = 4; id_d <= 8; id_d += 2 ) {
      sync_board_from_bitboards( root_my_bits, root_opp_bits, side_to_move, empties );
      ts_val = tree_search( 0, id_d, side_to_move, -INFINITE_EVAL, INFINITE_EVAL,
                            TRUE, (id_d >= 8), TRUE );
      if ( id_d == 6 )
        ts_val_6 = ts_val;
    }
    set_midgame_full_width_depth( 0 );
    if ( end_best_root_move == 0 && bb_valid_move( pv[0][0], root_my_bits, root_opp_bits ) )
      end_best_root_move = pv[0][0];
    pv[0][0] = saved_pv00;
    int rts_adj_6 = ts_val_6 - (int) llround( end_mean[stage][6] * 128.0 );
    int rts_adj_8 = ts_val - (int) llround( end_mean[stage][8] * 128.0 );
    int rts_adj_val = (abs( rts_adj_6 ) >= 14 * 128) ? rts_adj_6 : rts_adj_8;
    last_window_center = end_aspiration_center( rts_adj_val, 60 );
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
      static const int sel_schedule[] = { 6, 3, 2 };
      int num_passes = (int)(sizeof( sel_schedule ) / sizeof( sel_schedule[0] ));
      int pass_idx;
      for ( pass_idx = 0; pass_idx < num_passes &&
	      !is_panic_abort() && !force_return; pass_idx++ ) {
	selectivity = sel_schedule[pass_idx];
	alpha = last_window_center - 1;
	beta = last_window_center + 1;

	adjust_counter( &nodes );
	double pass_start_nodes = counter_value( &nodes );
	double pass_start_time = get_real_timer();
	int center_before = last_window_center;
	int total_steps = 0;

	root_eval = end_tree_wrapper( 0, empties, side_to_move,
				      alpha, beta, selectivity, TRUE );

	if ( root_eval <= alpha || root_eval >= beta ) {
	  int step_dir = (root_eval <= alpha) ? -2 : 2;
	  int steps = 0;
	  do {
	    last_window_center =
	      end_aspiration_step_center( last_window_center, root_eval, step_dir );
	    alpha = last_window_center - 1;
	    beta = last_window_center + 1;
	    if ( is_panic_abort() || force_return || ++steps >= 32 )
	      break;
	    root_eval = end_tree_wrapper( 0, empties, side_to_move,
					  alpha, beta, selectivity, TRUE );
	  } while ( (step_dir < 0) ? (root_eval <= alpha) : (root_eval >= beta) );
	  root_eval = last_window_center;
	  total_steps = (step_dir < 0) ? -steps : steps;
	}

	adjust_counter( &nodes );
	double pass_nodes = counter_value( &nodes ) - pass_start_nodes;
	double pass_time = get_real_timer() - pass_start_time;

	if ( getenv( "PASS_DIAG" ) != NULL ) {
	  fprintf( stderr, "PASS_DIAG: empties=%d pass=%d sel=%d center_before=%d center_after=%d eval=%d steps=%d nodes=%.0f time=%.3fs\n",
		   empties, pass_idx, selectivity, center_before, last_window_center, root_eval, total_steps, pass_nodes, pass_time );
	}

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





  adjust_counter( &nodes );
  double exact_start_nodes = counter_value( &nodes );
  double exact_start_time = get_real_timer();
  int exact_initial_center = last_window_center;
  int exact_researches = 0;

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
        exact_researches++;
        if ( root_eval >= 64 || root_eval <= -64 )
          break;

        end_aspiration_next_bounds( root_eval, &alpha, &beta, &delta );
        root_eval = end_tree_wrapper( 0, empties, side_to_move,
                                      alpha, beta, 0, TRUE );
        adjust_counter( &nodes );
        if ( (alpha <= -64 && beta >= 64) || root_eval >= 64 || root_eval <= -64 )
          break;
      }
    }

    adjust_counter( &nodes );
    double exact_nodes = counter_value( &nodes ) - exact_start_nodes;
    double exact_time = get_real_timer() - exact_start_time;

    if ( getenv( "PASS_DIAG" ) != NULL ) {
      fprintf( stderr, "PASS_DIAG: empties=%d pass=exact sel=0 initial_center=%d final_eval=%d researches=%d nodes=%.0f time=%.3fs\n",
	       empties, exact_initial_center, root_eval, exact_researches, exact_nodes, exact_time );
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


