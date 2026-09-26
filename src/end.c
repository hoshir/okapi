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
#include "end_leaf.h"
#include "epcstat.h"
#include "eval.h"
#include "getcoeff.h"
#include "globals.h"
#include "hash.h"
#include "macros.h"
#include "midgame.h"
#include "moves.h"
#include "osfbook.h"
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

#define PRE_SEARCH_DEEP_THRESHOLD    18

#define SELECTIVE_PRE_DEPTH_THRESHOLD 2
#define SELECTIVE_PRE_DEPTH_TOP_K 2

#ifdef _WIN32_WCE
#define EVENT_CHECK_INTERVAL         25000.0
#else
#define EVENT_CHECK_INTERVAL         250000.0
#endif

#define LOW_LEVEL_DEPTH              7
#define FASTEST_FIRST_DEPTH          12
#define HASH_DEPTH                   (LOW_LEVEL_DEPTH + 1)

#define VERY_HIGH_EVAL               1000000

#define GOOD_TRANSPOSITION_EVAL      10000000

#define FRONTIER_MOB_FACTOR          48

/* The disc difference when special wipeout move ordering is tried.
   This means more aggressive use of fastest first. */
#define WIPEOUT_THRESHOLD            60

/* Use stability pruning? */
#ifndef USE_STABILITY
#define USE_STABILITY                TRUE
#endif

/* Use shallow transposition table for low-depth endgame nodes? */
#ifndef USE_SHALLOW_TT
#define USE_SHALLOW_TT               TRUE
#define SHALLOW_TT_MIN_DEPTH         5
#endif



typedef enum {
  NOTHING,
  SELECTIVE_SCORE,
  WLD_SCORE,
  EXACT_SCORE
} SearchStatus;






/* The parities of the regions are in the region_parity bit vector. */


/* Pseudo-probabilities corresponding to the percentiles.
   These are taken from the normal distribution; to the percentile
   x corresponds the probability Pr(-x <= Y <= x) where Y is a N(0,1)
   variable. */

static const double confidence[MAX_SELECTIVITY + 1] =
{ 1.000, 0.99, 0.98, 0.954, 0.911, 0.838, 0.729, 0.576, 0.383, 0.197 };

/* Percentiles used in the endgame MPC */
static const double end_percentile[MAX_SELECTIVITY + 1] =
{ 100.0, 4.0, 3.0, 2.0, 1.7, 1.4, 1.1, 0.8, 0.5, 0.25 };

#if USE_STABILITY
#define  HIGH_STABILITY_THRESHOLD     0
#endif



static int true_found, true_val;

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
  END_STORE_TT
  Store search result into the transposition table in endgame mode.
*/

INLINE static void
end_store_tt( int score,
	      int best_move,
	      int in_alpha,
	      int beta,
	      int empties ) {
  end_best_move = best_move;
  if ( score >= beta )
    add_hash( ENDGAME_MODE, score, end_best_move,
	      ENDGAME_SCORE | LOWER_BOUND, empties, 0 );
  else if ( score > in_alpha )
    add_hash( ENDGAME_MODE, score, end_best_move,
	      ENDGAME_SCORE | EXACT_VALUE, empties, 0 );
  else
    add_hash( ENDGAME_MODE, score, end_best_move,
	      ENDGAME_SCORE | UPPER_BOUND, empties, 0 );
}



/*
  END_HANDLE_PASS
  Handle pass move in endgame search: game-over or flip colors and continue.
*/

INLINE static int
end_handle_pass( BitBoard my_bits,
		 BitBoard opp_bits,
		 int alpha,
		 int beta,
		 int oppcol,
		 int empties,
		 int disc_diff,
		 int pass_legal,
		 int level ) {
  int score;
  int selective_cutoff = FALSE;

  if ( !pass_legal ) {  /* Last move also pass, game over */
    pv_depth[level] = level;
    if ( disc_diff > 0 )
      return disc_diff + empties;
    if ( disc_diff < 0 )
      return disc_diff - empties;
    return 0;
  }

  /* Opponent gets the chance to play */
  hash1 ^= hash_flip_color1;
  hash2 ^= hash_flip_color2;
  prefetch_hash_endgame_key( hash2 );
  if ( level + 1 <= MAX_SEARCH_DEPTH ) {
    tls.stable_discs[BLACKSQ][level + 1] = tls.stable_discs[BLACKSQ][level];
    tls.stable_discs[WHITESQ][level + 1] = tls.stable_discs[WHITESQ][level];
  }
  score = -end_search_pvs( opp_bits, my_bits, -beta, -alpha,
			   oppcol, empties, -disc_diff, FALSE, level + 1,
			   0, &selective_cutoff );
  hash1 ^= hash_flip_color1;
  hash2 ^= hash_flip_color2;
  return score;
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
		  int *best_list_length, int verbose ) {
  int i;

  verbose = FALSE;

  if ( verbose ) {
    printf( "move=%2d  index=%d  length=%d      ", move, best_list_index,
	    *best_list_length );
    printf( "Before:  " );
    for ( i = 0; i < 4; i++ )
      printf( "%2d ", best_list[i] );
  }

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

  if ( verbose ) {
    printf( "      After:  " );
    for ( i = 0; i < 4; i++ )
      printf( "%2d ", best_list[i] );
    puts( "" );
  }
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

#define MAX_ROOT_MOVES               64

/* Remaining depth at or above which a node is worth splitting.
   Depth 11 distributes work efficiently across threads (SIMP-002). */
#define PARALLEL_SPLIT_DEPTH         11

/* How far the splits may nest, and how much more of the tree a node has
   to have left before it may start a batch at each level of nesting.
   Splitting is speculative -- every sibling is searched, including the
   ones a cutoff would have spared -- so a batch inside a batch costs
   real work, and without the taper the top plies each pay for one. */
#define MAX_SPLIT_NESTING             1
#define SPLIT_NESTING_MARGIN          6

typedef struct SiblingBatchTag {
  SearchState root;
  BitBoard my_bits;
  BitBoard opp_bits;
  BitBoard saved_stable[3];
  struct SiblingBatchTag *parent;   /* the batch this one was started from */
  int level;
  int empties;
  int disc_diff;
  int side_to_move;
  int alpha;                        /* null window is (alpha, alpha + 1) */
  int beta;                         /* the split node's own beta */
  int selectivity;
  volatile int abandon;             /* the node fails high; stop searching */
  int move[MAX_ROOT_MOVES];
  int score[MAX_ROOT_MOVES];
  int cutoff[MAX_ROOT_MOVES];
  int valid[MAX_ROOT_MOVES];
} SiblingBatch;


/* How many jobs deep this thread is, so that a node reached from inside
   a job can tell how far the splitting has already nested, and the
   innermost batch whose job it is running, so that it can tell whether
   what it is searching is still wanted. */
static _Thread_local int split_nesting;
static _Thread_local SiblingBatch *current_batch;

/* Batches in flight anywhere.  Reading a thread-local costs a call on
   this platform, and the test below sits on the per-node path, so ask
   this plain global first: it is zero for the whole of a search that
   never split, which is every single-threaded one. */
static volatile int active_splits;

#define SPLIT_ABANDONED()  ((active_splits != 0) && split_abandoned())


/*
  SPLIT_ABANDONED
  TRUE once the node a job belongs to -- or any node further out that
  this thread is nested inside -- has been proved to fail high.  Every
  sibling still being searched for such a node is work the sequential
  search would never have done: it stops at the first move that reaches
  beta, and the batch has now found one.

  Callers on the per-node path go through the macro of the same name;
  the two below are already inside a job, where the global cannot be
  zero.
*/

static int
split_abandoned( void ) {
  const SiblingBatch *b;

  for ( b = current_batch; b != NULL; b = b->parent )
    if ( b->abandon )
      return TRUE;

  return FALSE;
}


static void
search_sibling( int index, void *context ) {
  SiblingBatch *batch = (SiblingBatch *) context;
  BitBoard my_bits, opp_bits, new_my_bits, new_opp_bits;
  int move = batch->move[index];
  int child_selective_cutoff = FALSE;
  int score, bailed;
  SiblingBatch *saved_batch = current_batch;

  split_nesting++;
  current_batch = batch;
  search_state_load( &batch->root );
  my_bits = batch->my_bits;
  opp_bits = batch->opp_bits;

  if ( split_abandoned() ) {
    current_batch = saved_batch;
    split_nesting--;
    return;
  }
  int flipped = TestFlips_wrapper( move, my_bits, opp_bits );
  if ( flipped == 0 ) {
    current_batch = saved_batch;
    split_nesting--;
    return;
  }
  new_my_bits = bb_flips;
  FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

  if ( batch->level + 1 <= MAX_SEARCH_DEPTH ) {
    tls.stable_discs[BLACKSQ][batch->level + 1] = batch->saved_stable[BLACKSQ];
    tls.stable_discs[WHITESQ][batch->level + 1] = batch->saved_stable[WHITESQ];
  }

  unsigned int diff1, diff2;
  int pred, succ;
  end_make_move( move, new_my_bits, my_bits, batch->side_to_move, &diff1, &diff2, &pred, &succ );

  int child_disc_diff = -batch->disc_diff - 2 * flipped - 1;

  score = -end_search_pvs( new_opp_bits, new_my_bits,
			   -(batch->alpha + 1), -batch->alpha,
			   OPP( batch->side_to_move ),
			   batch->empties - 1,
			   child_disc_diff,
			   TRUE,
			   batch->level + 1,
			   batch->selectivity,
			   &child_selective_cutoff );

  end_unmake_move( move, diff1, diff2, pred, succ );

  /* Ask before raising the flag ourselves: a job that was cut short
     part way through has no score worth keeping, while the one that
     ran to the end and found the cutoff does. */
  bailed = split_abandoned();
  current_batch = saved_batch;
  split_nesting--;

  if ( !bailed && !is_panic_abort() && !force_return ) {
    batch->score[index] = score;
    batch->cutoff[index] = child_selective_cutoff;
    batch->valid[index] = TRUE;
    if ( score >= batch->beta )
      batch->abandon = TRUE;
  }
}


/*
  DISPATCH_SIBLINGS
  Search every legal move of this node except SEARCHED_MOVE in parallel
  with a null window around ALPHA.  Fills PROVEN[sq] with a score for
  each move that was proved not to beat ALPHA, which the sequential
  loop can then take instead of searching the move itself.

  The batch stops early once one of the moves reaches BETA, since the
  node then fails high and the sequential loop would never have looked
  at the rest.  Without that the batch searches every sibling to the
  end, which is what made splitting inside a split cost more than the
  threads it kept busy were worth.
*/

static void
dispatch_siblings( BitBoard my_bits, BitBoard opp_bits,
		   int side_to_move, int level, int empties,
		   int disc_diff, int alpha, int beta,
		   int selectivity, int searched_move,
		   const int *best_list, int best_list_length,
		   int pre_search_done,
		   int *proven, int *proven_score, int *proven_cutoff ) {
  /* Splits nest, so several batches can be live on one thread at once
     and the batch cannot be a single static.  It carries a whole
     SearchState, which is too much to want on the search's own stack
     several times over, and a split is rare enough that the allocation
     does not show up. */
  SiblingBatch *batch;
  int move[MAX_ROOT_MOVES];
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
	move[count++] = sq;
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
	move[count++] = rem_moves[i];
    }
  } else {
    for ( i = 0; i < MOVE_ORDER_SIZE; i++ ) {
      sq = sorted_move_order[disks_played][i];
      if ( !used[sq] && !((my_bits | opp_bits) & square_mask[sq]) &&
	   (TestFlips_wrapper( sq, my_bits, opp_bits ) > 0) ) {
	used[sq] = TRUE;
	if ( count < MAX_ROOT_MOVES )
	  move[count++] = sq;
      }
    }
  }

  if ( count == 0 )
    return;

  batch = (SiblingBatch *) malloc( sizeof( SiblingBatch ) );
  if ( batch == NULL )
    return;

  for ( i = 0; i < count; i++ ) {
    batch->move[i] = move[i];
    batch->score[i] = 0;
    batch->cutoff[i] = FALSE;
    batch->valid[i] = FALSE;
  }

  batch->parent = current_batch;
  batch->abandon = FALSE;
  batch->level = level;
  if ( level <= MAX_SEARCH_DEPTH ) {
    batch->saved_stable[BLACKSQ] = tls.stable_discs[BLACKSQ][level];
    batch->saved_stable[WHITESQ] = tls.stable_discs[WHITESQ][level];
  } else {
    batch->saved_stable[BLACKSQ] = 0;
    batch->saved_stable[WHITESQ] = 0;
  }
  batch->empties = empties;
  batch->disc_diff = disc_diff;
  batch->side_to_move = side_to_move;
  batch->alpha = alpha;
  batch->beta = beta;
  batch->selectivity = selectivity;
  batch->my_bits = my_bits;
  batch->opp_bits = opp_bits;
  sync_board_from_bitboards( my_bits, opp_bits, side_to_move, empties );
  search_state_save( &batch->root );

  (void) __sync_fetch_and_add( &active_splits, 1 );
  threads_run( search_sibling, batch, count );
  (void) __sync_fetch_and_sub( &active_splits, 1 );

  /* The calling thread takes part in the batch -- and in any other
     batch that had work while it waited -- so put its own state back
     the way the sequential search left it. */
  search_state_load( &batch->root );

  for ( i = 0; i < count; i++ )
    if ( batch->valid[i] && (batch->score[i] <= alpha) ) {
      proven[batch->move[i]] = TRUE;
      proven_score[batch->move[i]] = batch->score[i];
      proven_cutoff[batch->move[i]] = batch->cutoff[i];
    }

  free( batch );
}

/*
  END_ORDER_MOVES_PRESEARCH
  Heuristic pre-search move ordering helper for deep endgame search.
  Performs 1-ply ETC screening followed by shallow tree_search evaluations
  to order candidate moves.
*/

static int
end_order_moves_presearch( int level,
			   int pre_depth,
			   int empties,
			   int side_to_move,
			   BitBoard my_bits,
			   BitBoard opp_bits,
			   int alpha,
			   int beta,
			   int curr_alpha,
			   int selectivity,
			   int use_hash,
			   const int *best_list,
			   int best_list_length,
			   int can_split,
			   const int *proven,
			   const int *proven_score,
			   const HashEntry *mid_entry,
			   int *etc_tried_ptr,
			   int *etc_cutoff_score ) {
  int shallow_index;
  int etc_move = 0;
  int i, j;
  int move;
  int curr_val;
  int mobility;
  int threshold;
  int cand_moves_arr[64], cand_scores_arr[64];
  int cand_count;
  int top_k, k_idx;
  int shallow_score;
  int etc_demoted[100];
  BitBoard new_opp_bits;

  for ( i = 0; i < 100; i++ )
    etc_demoted[i] = FALSE;

  sync_board_from_bitboards( my_bits, opp_bits, side_to_move, empties );

  /* Pass 1: Lightweight 1-ply ETC scan for candidate moves before shallow tree_search */
  if ( use_hash ) {
    for ( shallow_index = 0; shallow_index < MOVE_ORDER_SIZE;
	  shallow_index++ ) {
      int already_checked;

      move = sorted_move_order[disks_played][shallow_index];
      if ( move == *etc_tried_ptr )
	continue;
      already_checked = FALSE;
      for ( j = 0; j < best_list_length; j++ )
	if ( move == best_list[j] )
	  already_checked = TRUE;

      if ( !already_checked && !((my_bits | opp_bits) & square_mask[move]) &&
	   (TestFlips_wrapper( move, my_bits, opp_bits ) > 0) ) {
	if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) )
	  continue;

	BitBoard child_my_bits;
	int flipped = TestFlips_bitboard_to( move, my_bits, opp_bits, &child_my_bits );
	if ( flipped != 0 ) {
	  unsigned int diff1, diff2;
	  end_hash_diff( child_my_bits, my_bits, side_to_move, move, &diff1, &diff2 );
	  hash1 ^= diff1;
	  hash2 ^= diff2;
	  prefetch_hash_endgame_key( hash2 );
	  HashEntry etc_entry;
	  find_hash( &etc_entry, ENDGAME_MODE );
	  hash1 ^= diff1;
	  hash2 ^= diff2;

	  if ( (etc_entry.flags & ENDGAME_SCORE) &&
	       (etc_entry.draft >= empties - 1) &&
	       (etc_entry.selectivity <= selectivity) ) {
	    if ( (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
		 (etc_entry.eval <= -beta) ) {
	      etc_move = move;
	      *etc_tried_ptr = etc_move;
	      *etc_cutoff_score = -etc_entry.eval;
	      return TRUE;
	    }
	    else if ( (etc_entry.flags & (LOWER_BOUND | EXACT_VALUE)) &&
		      (etc_entry.eval >= -curr_alpha) ) {
	      etc_demoted[move] = TRUE;
	    }
	  }
	}
      }
    }
  }

  if ( etc_move != 0 ) {
    *etc_tried_ptr = etc_move;
    evals[disks_played][etc_move] = GOOD_TRANSPOSITION_EVAL;
    move_list[disks_played][move_count[disks_played]] = etc_move;
    move_count[disks_played]++;
  }
  else {
    if ( (level == 0) || (pre_depth < SELECTIVE_PRE_DEPTH_THRESHOLD) ) {
      threshold =
	MIN( WIPEOUT_THRESHOLD * 128,
	     128 * alpha + fast_first_threshold[disks_played][pre_depth] );

      for ( shallow_index = 0; shallow_index < MOVE_ORDER_SIZE;
	    shallow_index++ ) {
	int already_checked;

	move = sorted_move_order[disks_played][shallow_index];
	if ( move == *etc_tried_ptr )
	  continue;
	already_checked = FALSE;
	for ( j = 0; j < best_list_length; j++ )
	  if ( move == best_list[j] )
	    already_checked = TRUE;

	if ( !already_checked && !((my_bits | opp_bits) & square_mask[move]) &&
	     (TestFlips_wrapper( move, my_bits, opp_bits ) > 0) ) {
	  if ( (can_split && proven[move] && (proven_score[move] <= curr_alpha)) || etc_demoted[move] ) {
	    evals[disks_played][move] = -INFINITE_EVAL;
	    move_list[disks_played][move_count[disks_played]] = move;
	    move_count[disks_played]++;
	    continue;
	  }
	  FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

	  (void) make_move( side_to_move, move, TRUE );
	  curr_val = 0;

	  /* Enhanced Transposition Cutoff: It's a good idea to
	     transpose back into a position in the hash table. */

	  if ( use_hash ) {
	    HashEntry etc_entry;

	    prefetch_hash_endgame_key( hash2 );
	    find_hash( &etc_entry, ENDGAME_MODE );
	    if ( (etc_entry.flags & ENDGAME_SCORE) &&
		 (etc_entry.draft == empties - 1) ) {
	      curr_val += 384;
	      if ( etc_entry.selectivity <= selectivity ) {
		if ( (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
		     (etc_entry.eval <= -beta) )
		  curr_val = GOOD_TRANSPOSITION_EVAL;
		if ( (etc_entry.flags & LOWER_BOUND) &&
		     (etc_entry.eval >= -alpha) )
		  curr_val -= 640;
	      }
	    }
	  }

	  /* Determine the midgame score. If it is worse than
	     alpha-8, a fail-high is likely so precision in that
	     range is not worth the extra nodes required. */

	  if ( curr_val != GOOD_TRANSPOSITION_EVAL )
	    curr_val -=
	      tree_search( level + 1, level + pre_depth,
			   OPP( side_to_move ), -INFINITE_EVAL,
			   (-alpha + 8) * 128, TRUE, TRUE, TRUE );

	  /* Make the moves which are highly likely to result in
	     fail-high in decreasing order of mobility for the
	     opponent. */

	  if ( (curr_val > threshold) || (move == mid_entry->move[0]) ) {
	    if ( curr_val > WIPEOUT_THRESHOLD * 128 )
	      curr_val += 2 * VERY_HIGH_EVAL;
	    else
	      curr_val += VERY_HIGH_EVAL;
	    if ( curr_val < GOOD_TRANSPOSITION_EVAL ) {
	      mobility = bitboard_mobility( new_opp_bits, bb_flips );
	      if ( curr_val > 2 * VERY_HIGH_EVAL )
		curr_val -= 2 * ff_mob_factor[disks_played - 1] * mobility;
	      else
		curr_val -= ff_mob_factor[disks_played - 1] * mobility;
	    }
	  }

	  unmake_move( side_to_move, move );
	  evals[disks_played][move] = curr_val;
	  move_list[disks_played][move_count[disks_played]] = move;
	  move_count[disks_played]++;

	  /* If this move achieves an ETC beta-cutoff, no need to evaluate further candidate moves */
	  if ( curr_val == GOOD_TRANSPOSITION_EVAL )
	    break;
	}
      }
    }
    else {
      cand_count = 0;

      for ( shallow_index = 0; shallow_index < MOVE_ORDER_SIZE;
	    shallow_index++ ) {
	int already_checked;

	move = sorted_move_order[disks_played][shallow_index];
	if ( move == *etc_tried_ptr )
	  continue;
	already_checked = FALSE;
	for ( j = 0; j < best_list_length; j++ )
	  if ( move == best_list[j] )
	    already_checked = TRUE;

	if ( !already_checked && !((my_bits | opp_bits) & square_mask[move]) &&
	     (TestFlips_wrapper( move, my_bits, opp_bits ) > 0) ) {
	  if ( (can_split && proven[move] && (proven_score[move] <= curr_alpha)) || etc_demoted[move] ) {
	    cand_moves_arr[cand_count] = move;
	    cand_scores_arr[cand_count] = -INFINITE_EVAL;
	    cand_count++;
	    continue;
	  }
	  FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

	  (void) make_move( side_to_move, move, TRUE );
	  curr_val = 0;

	  if ( use_hash ) {
	    HashEntry etc_entry;

	    prefetch_hash_endgame_key( hash2 );
	    find_hash( &etc_entry, ENDGAME_MODE );
	    if ( (etc_entry.flags & ENDGAME_SCORE) &&
		 (etc_entry.draft == empties - 1) &&
		 (etc_entry.selectivity <= selectivity) &&
		 (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
		 (etc_entry.eval <= -beta) ) {
	      curr_val = GOOD_TRANSPOSITION_EVAL;
	    }
	  }

	  if ( curr_val == GOOD_TRANSPOSITION_EVAL ) {
	    unmake_move( side_to_move, move );
	    cand_moves_arr[cand_count] = move;
	    cand_scores_arr[cand_count] = GOOD_TRANSPOSITION_EVAL;
	    cand_count++;
	    break;
	  }

	  /* Stage 1 (Fast Screening): 1-ply lookahead */
	  curr_val -=
	    tree_search( level + 1, level + 1,
			 OPP( side_to_move ), -INFINITE_EVAL,
			 (-alpha + 8) * 128, TRUE, TRUE, TRUE );
	  mobility = bitboard_mobility( new_opp_bits, bb_flips );
#if FRONTIER_MOB_FACTOR > 0
	  {
	    BitBoard empty_bits = ~(bb_flips | new_opp_bits);
	    int pot_mobility = bitboard_frontier( bb_flips, empty_bits );
	    shallow_score = curr_val - ff_mob_factor[disks_played - 1] * mobility
				     - FRONTIER_MOB_FACTOR * pot_mobility;
	  }
#else
	  shallow_score = curr_val - ff_mob_factor[disks_played - 1] * mobility;
#endif

	  unmake_move( side_to_move, move );

	  cand_moves_arr[cand_count] = move;
	  cand_scores_arr[cand_count] = shallow_score;
	  cand_count++;
	}
      }

      /* Sort cand_moves_arr descending by cand_scores_arr */
      for ( i = 1; i < cand_count; i++ ) {
	int key_move = cand_moves_arr[i];
	int key_score = cand_scores_arr[i];
	int k = i - 1;
	while ( k >= 0 && cand_scores_arr[k] < key_score ) {
	  cand_moves_arr[k + 1] = cand_moves_arr[k];
	  cand_scores_arr[k + 1] = cand_scores_arr[k];
	  k--;
	}
	cand_moves_arr[k + 1] = key_move;
	cand_scores_arr[k + 1] = key_score;
      }

      /* Stage 2 (Selective Deepening) */
      top_k = MIN( cand_count, SELECTIVE_PRE_DEPTH_TOP_K );

      threshold =
	MIN( WIPEOUT_THRESHOLD * 128,
	     128 * alpha + fast_first_threshold[disks_played][pre_depth] );

      for ( k_idx = 0; k_idx < cand_count; k_idx++ ) {
	move = cand_moves_arr[k_idx];
	if ( k_idx < top_k ) {
	  if ( cand_scores_arr[k_idx] == GOOD_TRANSPOSITION_EVAL ) {
	    evals[disks_played][move] = GOOD_TRANSPOSITION_EVAL;
	    move_list[disks_played][move_count[disks_played]] = move;
	    move_count[disks_played]++;
	    break;
	  }
	  if ( cand_scores_arr[k_idx] == -INFINITE_EVAL ) {
	    evals[disks_played][move] = -INFINITE_EVAL;
	    move_list[disks_played][move_count[disks_played]] = move;
	    move_count[disks_played]++;
	    continue;
	  }

	  (void) TestFlips_wrapper( move, my_bits, opp_bits );
	  FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
	  (void) make_move( side_to_move, move, TRUE );
	  curr_val = 0;

	  if ( use_hash ) {
	    HashEntry etc_entry;

	    prefetch_hash_endgame_key( hash2 );
	    find_hash( &etc_entry, ENDGAME_MODE );
	    if ( (etc_entry.flags & ENDGAME_SCORE) &&
		 (etc_entry.draft == empties - 1) ) {
	      curr_val += 384;
	      if ( etc_entry.selectivity <= selectivity ) {
		if ( (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
		     (etc_entry.eval <= -beta) )
		  curr_val = GOOD_TRANSPOSITION_EVAL;
		if ( (etc_entry.flags & LOWER_BOUND) &&
		     (etc_entry.eval >= -alpha) )
		  curr_val -= 640;
	      }
	    }
	  }

	  if ( curr_val != GOOD_TRANSPOSITION_EVAL )
	    curr_val -=
	      tree_search( level + 1, level + pre_depth,
			   OPP( side_to_move ), -INFINITE_EVAL,
			   (-alpha + 8) * 128, TRUE, TRUE, TRUE );

	  if ( (curr_val > threshold) || (move == mid_entry->move[0]) ) {
	    if ( curr_val > WIPEOUT_THRESHOLD * 128 )
	      curr_val += 2 * VERY_HIGH_EVAL;
	    else
	      curr_val += VERY_HIGH_EVAL;
	    if ( curr_val < GOOD_TRANSPOSITION_EVAL ) {
	      mobility = bitboard_mobility( new_opp_bits, bb_flips );
	      if ( curr_val > 2 * VERY_HIGH_EVAL )
		curr_val -= 2 * ff_mob_factor[disks_played - 1] * mobility;
	      else
		curr_val -= ff_mob_factor[disks_played - 1] * mobility;
	    }
	  }

	  unmake_move( side_to_move, move );
	  evals[disks_played][move] = curr_val;
	  move_list[disks_played][move_count[disks_played]] = move;
	  move_count[disks_played]++;

	  if ( curr_val == GOOD_TRANSPOSITION_EVAL )
	    break;
	}
	else {
	  evals[disks_played][move] = cand_scores_arr[k_idx];
	  move_list[disks_played][move_count[disks_played]] = move;
	  move_count[disks_played]++;
	}
      }
    }
  }
  return FALSE;
}

/* Move bonuses without and with parity for the squares.
   These are only used when sorting moves in the 8-12 empties
   range and were automatically tuned by OPTIMIZE. */
static const unsigned char move_bonus[2][128] = {  /* 2 * 100 used */
  {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,  24,   1,   0,  25,  25,   0,   1,  24,   0,
      0,   1,   0,   0,   0,   0,   0,   0,   1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,  25,   0,   0,   0,   0,   0,   0,  25,   0,
      0,  25,   0,   0,   0,   0,   0,   0,  25,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   1,   0,   0,   0,   0,   0,   0,   1,   0,
      0,  24,   1,   0,  25,  25,   0,   1,  24,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
  {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0, 128,  86, 122, 125, 125, 122,  86, 128,   0,
      0,  86, 117, 128, 128, 128, 128, 117,  86,   0,
      0, 122, 128, 128, 128, 128, 128, 128, 122,   0,
      0, 125, 128, 128, 128, 128, 128, 128, 125,   0,
      0, 125, 128, 128, 128, 128, 128, 128, 125,   0,
      0, 122, 128, 128, 128, 128, 128, 128, 122,   0,
      0,  86, 117, 128, 128, 128, 128, 117,  86,   0,
      0, 128,  86, 122, 125, 125, 122,  86, 128,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }
};

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
  HashEntry entry, mid_entry;
  int oppcol = OPP( side_to_move );
  int in_alpha = alpha;
  int use_hash;
  int hash_hit = FALSE;
  int hash_move = -1;

  *selective_cutoff = FALSE;

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
        return (empties <= FASTEST_FIRST_DEPTH) ? lower_bound : beta;
      }
      if ( lower_bound > alpha )
        alpha = lower_bound;
    }
    if ( alpha >= beta ) {
      pv_depth[level] = level;
      return alpha;
    }
  }

  if ( ((my_bits | opp_bits) & CORNER_MASK) != 0 ) {
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
        if ( edges.bits != 0 ) {
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
          return (empties <= FASTEST_FIRST_DEPTH) ? lower_bound : beta;
        }
        if ( lower_bound > alpha )
          alpha = lower_bound;
        if ( edges.bits != 0 ) {
          int s_full = count_stable_indexed( side_to_move, my_bits, opp_bits, &edges );
          if ( level <= MAX_SEARCH_DEPTH )
            tls.stable_discs[side_to_move][level] |= (side_to_move == BLACKSQ ? last_black_stable : last_white_stable);
          lower_bound = 2 * s_full - 64;
          if ( lower_bound >= beta ) {
            pv_depth[level] = level;
            return (empties <= FASTEST_FIRST_DEPTH) ? lower_bound : beta;
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
  mid_entry.draft = NO_HASH_MOVE;
  mid_entry.flags = 0;

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

    if ( empties <= FASTEST_FIRST_DEPTH ) {
      if ( (entry.draft == empties) &&
	   (entry.selectivity == 0) &&
	   (entry.flags & ENDGAME_SCORE) &&
	   bb_valid_move( entry.move[0], my_bits, opp_bits ) ) {
	hash_move = entry.move[0];
      }
    }
    else {
      hash_hit = (entry.draft != NO_HASH_MOVE) &&
		 ((entry.flags & (EXACT_VALUE | LOWER_BOUND)) ||
		  (entry.draft >= empties));

      find_hash( &mid_entry, MIDGAME_MODE );
      if ( (mid_entry.draft != NO_HASH_MOVE) &&
	   (mid_entry.flags & MIDGAME_SCORE) ) {
	if ( (level <= 4) || (mid_entry.flags & (EXACT_VALUE | LOWER_BOUND)) ) {
	  if ( (level == 0) && !hash_hit &&
	       (mid_entry.eval < WIPEOUT_THRESHOLD * 128) ) {
	    entry = mid_entry;
	    hash_hit = TRUE;
	  }
	}
      }
    }
  }

  /* 5. Move Ordering & Loop by Depth Band */
  if ( empties <= FASTEST_FIRST_DEPTH ) {
    /* ---------------------------------------------------------------
       SHALLOW ENDGAME (8-12 empties):
       Fast bitboard static ordering + doubly-linked move list
       --------------------------------------------------------------- */
    BitBoard new_opp_bits;
    BitBoard nws_my_bits;
    BitBoard best_new_my_bits = 0, best_new_opp_bits = 0;
    int i;
    int score;
    int flipped, best_flipped = 0;
    int new_disc_diff;
    int ev;
    int moves = 0;
    int parity;
    int best_value = -INFINITE_EVAL, best_index = 0;
    int pred, succ;
    int sq, old_sq, best_sq = 0;
    int move_order[64];
    int goodness[64];
    unsigned int diff1, diff2;

    for ( old_sq = END_MOVE_LIST_HEAD, sq = end_move_list[old_sq].succ;
	  sq != END_MOVE_LIST_TAIL;
	  old_sq = sq, sq = end_move_list[sq].succ ) {
      flipped = TestFlips_wrapper( sq, my_bits, opp_bits );
      if ( flipped != 0 ) {
	INCREMENT_COUNTER( nodes );

	FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );
	end_move_list[old_sq].succ = end_move_list[sq].succ;

	if ( quadrant_mask[sq] & region_parity )
	  parity = 1;
	else
	  parity = 0;
	goodness[moves] = move_bonus[parity][sq];
	if ( sq == hash_move )
	  goodness[moves] += 128;

	goodness[moves] -= weighted_mobility( new_opp_bits, bb_flips );

	if ( goodness[moves] > best_value ) {
	  best_value = goodness[moves];
	  best_index = moves;
	  best_new_my_bits = bb_flips;
	  best_new_opp_bits = new_opp_bits;
	  best_flipped = flipped;
	}

	end_move_list[old_sq].succ = sq;

	if ( use_hash ) {
	  unsigned int diff1, diff2;
	  end_hash_diff( bb_flips, my_bits, side_to_move, sq, &diff1, &diff2 );
	  hash1 ^= diff1;
	  hash2 ^= diff2;
	  prefetch_hash_endgame_key( hash2 );
	  HashEntry etc_entry;
	  find_hash( &etc_entry, ENDGAME_MODE );
	  hash1 ^= diff1;
	  hash2 ^= diff2;

	  if ( (etc_entry.flags & ENDGAME_SCORE) &&
	       (etc_entry.draft >= empties - 1) &&
	       (etc_entry.selectivity <= selectivity) ) {
	    if ( (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
		 (etc_entry.eval <= -beta) ) {
	      int score = -etc_entry.eval;
	      end_store_tt( score, sq, in_alpha, beta, empties );
	      if ( level == 0 )
		end_best_root_move = sq;
	      return score;
	    }
	    else if ( (etc_entry.flags & (LOWER_BOUND | EXACT_VALUE)) &&
		      (etc_entry.eval >= -alpha) ) {
	      goodness[moves] -= 10000;
	    }
	  }
	}

	move_order[moves] = sq;
	moves++;
      }
    }

    if ( moves == 0 )
      return end_handle_pass( my_bits, opp_bits, alpha, beta, oppcol,
			      empties, disc_diff, pass_legal, level );

    /* Primary move: full window [-beta, -alpha] */
    sq = move_order[best_index];
    end_make_move( sq, best_new_my_bits, my_bits, side_to_move, &diff1, &diff2, &pred, &succ );

    new_disc_diff = -disc_diff - 2 * best_flipped - 1;
    if ( level + 1 <= MAX_SEARCH_DEPTH ) {
      tls.stable_discs[BLACKSQ][level + 1] = tls.stable_discs[BLACKSQ][level];
      tls.stable_discs[WHITESQ][level + 1] = tls.stable_discs[WHITESQ][level];
    }

    score = -end_search_pvs( best_new_opp_bits, best_new_my_bits,
			     -beta, -alpha, oppcol, empties - 1,
			     new_disc_diff, TRUE, level + 1,
			     selectivity, selective_cutoff );

    end_unmake_move( sq, diff1, diff2, pred, succ );

    best_sq = sq;
    if ( score > alpha ) {
      if ( score >= beta ) {
	end_store_tt( score, best_sq, in_alpha, beta, empties );
	pv_depth[level] = level + 1;
	pv[level][level] = best_sq;
	if ( level == 0 ) {
	  end_best_root_move = best_sq;
	  if ( !get_ponder_move() ) {
	    send_sweep( "%-10s ", buffer );
	    send_sweep( "%c%c", TO_SQUARE( best_sq ) );
	    send_sweep( ">%d", score - 1 );
	  }
	}
	return score;
      }
      alpha = score;
    }

    /* Sibling moves loop (PVS) */
    move_order[best_index] = move_order[0];
    goodness[best_index] = goodness[0];

    for ( i = 1; i < moves; i++ ) {
      int j;

      best_value = goodness[i];
      best_index = i;
      for ( j = i + 1; j < moves; j++ )
	if ( goodness[j] > best_value ) {
	  best_value = goodness[j];
	  best_index = j;
	}
      sq = move_order[best_index];
      move_order[best_index] = move_order[i];
      goodness[best_index] = goodness[i];

      flipped = TestFlips_wrapper( sq, my_bits, opp_bits );
      FULL_ANDNOT( new_opp_bits, opp_bits, bb_flips );

      end_make_move( sq, bb_flips, my_bits, side_to_move, &diff1, &diff2, &pred, &succ );

      new_disc_diff = -disc_diff - 2 * flipped - 1;

      if ( level + 1 <= MAX_SEARCH_DEPTH ) {
	tls.stable_discs[BLACKSQ][level + 1] = tls.stable_discs[BLACKSQ][level];
	tls.stable_discs[WHITESQ][level + 1] = tls.stable_discs[WHITESQ][level];
      }

      /* Null-window search [-(alpha+1), -alpha] */
      nws_my_bits = bb_flips;
      ev = -end_search_pvs( new_opp_bits, nws_my_bits, -(alpha + 1), -alpha,
			    oppcol, empties - 1, new_disc_diff, TRUE, level + 1,
			    selectivity, selective_cutoff );

      /* Re-search on unexpected fail-high within (alpha, beta) */
      if ( ev > alpha && ev < beta ) {
	if ( level + 1 <= MAX_SEARCH_DEPTH ) {
	  tls.stable_discs[BLACKSQ][level + 1] = tls.stable_discs[BLACKSQ][level];
	  tls.stable_discs[WHITESQ][level + 1] = tls.stable_discs[WHITESQ][level];
	}
	ev = -end_search_pvs( new_opp_bits, nws_my_bits, -beta, -ev,
			      oppcol, empties - 1, new_disc_diff, TRUE, level + 1,
			      selectivity, selective_cutoff );
      }

      end_unmake_move( sq, diff1, diff2, pred, succ );

      if ( ev > score ) {
	score = ev;
	if ( ev > alpha ) {
	  if ( ev >= beta ) {
	    end_store_tt( score, sq, in_alpha, beta, empties );
	    pv_depth[level] = level + 1;
	    pv[level][level] = sq;
	    if ( level == 0 ) {
	      end_best_root_move = sq;
	      if ( !get_ponder_move() ) {
		send_sweep( "%-10s ", buffer );
		send_sweep( "%c%c", TO_SQUARE( sq ) );
		send_sweep( ">%d", score - 1 );
	      }
	    }
	    return score;
	  }
	  alpha = ev;
	}
	best_sq = sq;
      }
    }

    end_store_tt( score, best_sq, in_alpha, beta, empties );
    pv_depth[level] = level + 1;
    pv[level][level] = best_sq;
    if ( level == 0 ) {
      end_best_root_move = best_sq;
      if ( !get_ponder_move() ) {
	send_sweep( "%-10s ", buffer );
	send_sweep( "%c%c", TO_SQUARE( best_sq ) );
	if ( score <= in_alpha )
	  send_sweep( "<%d", score + 1 );
	else if ( score >= beta )
	  send_sweep( ">%d", score - 1 );
	else
	  send_sweep( "=%d", score );
      }
    }
    return score;
  }
  else {
    /* ---------------------------------------------------------------
       DEEP ENDGAME (>= 13 empties):
       Heuristic pre-search ordering + SMP parallel sibling dispatch
       --------------------------------------------------------------- */
    double node_val;
    int i;
    int move;
    int move_index;
    int pre_depth;
    int update_pv, first;
    int curr_alpha;
    int pre_search_done, etc_tried;
    int best_list_index, best_list_length;
    int best_list[4];
    int proven[100], proven_score[100], proven_cutoff[100];
    int siblings_dispatched = FALSE;
    int can_split;
    int best;
    int curr_val;
    int saved_disks_played = disks_played;

    disks_played = 60 - empties;

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

    /* Shallow pre-search depth: unified 2-tier threshold (SIMP-003) */
    pre_depth = (empties >= PRE_SEARCH_DEEP_THRESHOLD ? 4 : 2);

    first = TRUE;
    can_split = (empties >= PARALLEL_SPLIT_DEPTH +
		 SPLIT_NESTING_MARGIN * split_nesting) &&
      (split_nesting <= MAX_SPLIT_NESTING) && (threads_count() > 1) &&
      (threads_idle_count() > 0);
    if ( can_split )
      for ( i = 0; i < 100; i++ )
	proven[i] = FALSE;
    best = -INFINITE_EVAL;
    pre_search_done = FALSE;
    etc_tried = 0;
    curr_alpha = alpha;

    /* Initialize move list and check hash table moves */
    move_count[disks_played] = 0;
    best_list_length = 0;
    for ( i = 0; i < 4; i++ )
      best_list[i] = 0;
    if ( hash_hit )
      for ( i = 0; i < 4; i++ ) {
	int cand_sq = entry.move[i];
	if ( bb_valid_move( cand_sq, my_bits, opp_bits ) ) {
	  best_list[best_list_length++] = cand_sq;

	  if ( use_hash ) {
	    BitBoard child_my_bits;
	    int flipped = TestFlips_bitboard_to( cand_sq, my_bits, opp_bits, &child_my_bits );
	    if ( flipped != 0 ) {
	      unsigned int diff1, diff2;
	      end_hash_diff( child_my_bits, my_bits, side_to_move, cand_sq, &diff1, &diff2 );
	      hash1 ^= diff1;
	      hash2 ^= diff2;
	      prefetch_hash_endgame_key( hash2 );
	      HashEntry etc_entry;
	      find_hash( &etc_entry, ENDGAME_MODE );
	      hash1 ^= diff1;
	      hash2 ^= diff2;

	      if ( (etc_entry.flags & ENDGAME_SCORE) &&
		   (etc_entry.draft >= empties - 1) &&
		   (etc_entry.selectivity <= selectivity) &&
		   (etc_entry.flags & (UPPER_BOUND | EXACT_VALUE)) &&
		   (etc_entry.eval <= -beta) ) {
		int score = -etc_entry.eval;
		best_list[0] = cand_sq;
		if ( use_hash )
		  add_hash_extended( ENDGAME_MODE, score, best_list,
				     ENDGAME_SCORE | LOWER_BOUND, empties,
				     *selective_cutoff ? selectivity : 0 );
		if ( level == 0 )
		  end_best_root_move = cand_sq;
		disks_played = saved_disks_played;
		return score;
	      }
	    }
	  }
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
	  if ( end_order_moves_presearch( level, pre_depth, empties, side_to_move,
					  my_bits, opp_bits, alpha, beta, curr_alpha,
					  selectivity, use_hash,
					  best_list, best_list_length,
					  can_split, proven, proven_score,
					  &mid_entry, &etc_tried,
					  &etc_cutoff_score ) ) {
	    best_list[0] = etc_tried;
	    if ( use_hash )
	      add_hash_extended( ENDGAME_MODE, etc_cutoff_score, best_list,
				 ENDGAME_SCORE | LOWER_BOUND, empties,
				 *selective_cutoff ? selectivity : 0 );
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
      int flipped = TestFlips_wrapper( move, my_bits, opp_bits );
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
      }
      else {
	curr_alpha = MAX( best, curr_alpha );
	if ( can_split && proven[move] && (proven_score[move] <= curr_alpha) ) {
	  curr_val = proven_score[move];
	  child_selective_cutoff = proven_cutoff[move];
	}
	else
	  curr_val =
	    -end_search_pvs( new_opp_bits, new_my_bits,
			     -(curr_alpha + 1), -curr_alpha, OPP( side_to_move ),
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

      if ( is_panic_abort() || force_return || SPLIT_ABANDONED() ) {
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
	      true_found = TRUE;
	      true_val = curr_val;
	    }
	  }
	}
	send_sweep( " " );
	if ( update_pv && (move_index > 0) && echo )
	  display_sweep( stdout );
      }

      if ( update_pv ) {
	update_best_list( best_list, move, best_list_index, &best_list_length,
			  level == 0 );
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

      if ( can_split && first && !siblings_dispatched &&
	   (threads_idle_count() > 0) &&
	   !is_panic_abort() && !force_return ) {
	siblings_dispatched = TRUE;
	dispatch_siblings( my_bits, opp_bits, side_to_move, level,
			   empties, disc_diff, best, beta, selectivity, move,
			   best_list, best_list_length, pre_search_done,
			   proven, proven_score, proven_cutoff );
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
  increment_hash_generation();

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

  if ( empties > DISABLE_SELECTIVITY ) {
    if ( wld )
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
    else
      for ( selectivity = MAX_SELECTIVITY; (selectivity > 0) &&
	      !is_panic_abort() && !force_return; selectivity-- ) {
	alpha = last_window_center - 1;
	beta = last_window_center + 1;

	root_eval = end_tree_wrapper( 0, empties, side_to_move,
				      alpha, beta, selectivity, TRUE );

	if ( root_eval <= alpha ) {
	  do {
	    last_window_center -= 2;
	    alpha = last_window_center - 1;
	    beta = last_window_center + 1;
	    if ( is_panic_abort() || force_return )
	      break;
	    root_eval = end_tree_wrapper( 0, empties, side_to_move,
					  alpha, beta, selectivity, TRUE );
	  } while ( root_eval <= alpha );
	  root_eval = last_window_center;
	}
	else if ( root_eval >= beta ) {
	  do {
	    last_window_center += 2;
	    alpha = last_window_center - 1;
	    beta = last_window_center + 1;
	    if ( is_panic_abort() || force_return )
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
  else
    selectivity = 0;

  /* Check if the selective search took more than 40% of the allocated
       time. If this is the case, there is no point attempting WLD. */

  long_selective_search = check_threshold( 0.35 );

  /* Make sure the panic abort flag is set properly; it must match
     the status of long_selective_search. This is not automatic as
     it is not guaranteed that any selective search was performed. */

  check_panic_abort();

  if ( is_panic_abort() || force_return || long_selective_search ) {

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
      if ( root_eval <= alpha ) {
	int ceiling_value = last_window_center - 2;
	while ( 1 ) {
	  alpha = ceiling_value - 1;
	  beta = ceiling_value;
	  root_eval = end_tree_wrapper( 0, empties, side_to_move,
					alpha, beta, 0, TRUE );
	  if ( is_panic_abort() || force_return )
	    break;
	  if ( root_eval > alpha )
	    break;
	  else
	    ceiling_value -= 2;
	}
      }
      else if ( root_eval >= beta ) {
	int floor_value = last_window_center + 2;
	while ( 1 ) {
	  alpha = floor_value - 1;
	  beta = floor_value + 1;
	  root_eval = end_tree_wrapper( 0, empties, side_to_move,
					alpha, beta, 0, TRUE );
	  if ( is_panic_abort() || force_return )
	    break;
	  assert( root_eval > alpha );
	  if ( root_eval < beta )
	    break;
	  else
	    floor_value += 2;
	}
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


