/*
   File:          end_leaf.h

   Created:       2026

   Contents:      Leaf solvers (4 to 7 empty squares) for the endgame search.
*/

#ifndef END_LEAF_H
#define END_LEAF_H

#include "bitboard.h"
#include "bitbtest.h"
#include "constant.h"
#include "counter.h"
#include "end.h"
#include "hash.h"
#include "macros.h"
#include "search.h"
#include "stable.h"
#include "tlstate.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USE_STABILITY
#define USE_STABILITY                TRUE
#endif

#ifndef USE_SHALLOW_TT
#define USE_SHALLOW_TT               TRUE
#define SHALLOW_TT_MIN_DEPTH         5
#endif

extern BitBoard neighborhood_mask[100];

/*
  TESTFLIPS_WRAPPER
  Checks if SQ is a valid move by
  (1) verifying that there exists a neighboring opponent disc,
  (2) verifying that the move flips some disc.
*/

INLINE static int
TestFlips_wrapper( int sq,
		   BitBoard my_bits,
		   BitBoard opp_bits ) {
  int flipped;

  if ( (neighborhood_mask[sq] & opp_bits) != 0 )
    flipped = TestFlips_bitboard( sq, my_bits, opp_bits );
  else
    flipped = 0;

  return flipped;
}

/*
  END_HASH_DIFF
  Compute the incremental hash-key update for COLOR playing SQ,
  from the difference between the old and new bitboards of the mover.
  Replaces the array-based DoFlips_hash.
*/

static INLINE void
end_hash_diff( const BitBoard new_my_bits, const BitBoard my_bits,
	       int color, int sq,
	       unsigned int *diff1, unsigned int *diff2 ) {
  BitBoard fl;
  int bit, index;
  unsigned int d1 = hash_put_value1[color][sq];
  unsigned int d2 = hash_put_value2[color][sq];

  fl = (new_my_bits ^ my_bits) & ~square_mask[sq];
  while ( fl != 0 ) {
    bit = FIRST_BIT( fl );
    fl &= fl - 1;
    index = 10 * (bit / 8 + 1) + (bit % 8) + 1;
    d1 ^= hash_flip1[index];
    d2 ^= hash_flip2[index];
  }

  *diff1 = d1;
  *diff2 = d2;
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
		   int pass_legal );

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
		  int pass_legal );

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
		  int pass_legal );

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
		 int pass_legal );

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
		   int pass_legal );

#ifdef __cplusplus
}
#endif

#endif  /* END_LEAF_H */
