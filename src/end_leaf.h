/*
   File:          end_leaf.h

   Created:       2026

   Contents:      Leaf solvers (1 to 8 empty squares) for the endgame search.
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
#endif
#ifndef SHALLOW_TT_MIN_DEPTH
#define SHALLOW_TT_MIN_DEPTH         5
#endif

/*
  TESTFLIPS_WRAPPER
  Checks if SQ is a valid move by
  (1) verifying that there exists a neighboring opponent disc,
  (2) verifying that the move flips some disc.
*/

static ALWAYS_INLINE int
TestFlips_wrapper_to( int sq,
		      BitBoard my_bits,
		      BitBoard opp_bits,
		      BitBoard *new_my_bits ) {
  if ( (neighborhood_mask[sq] & opp_bits) != 0 )
    return TestFlips_bitboard_to( sq, my_bits, opp_bits, new_my_bits );
  else
    return 0;
}

INLINE static int
TestFlips_wrapper( int sq,
		   BitBoard my_bits,
		   BitBoard opp_bits ) {
  return TestFlips_wrapper_to( sq, my_bits, opp_bits, &bb_flips );
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
  BitBoard fl = (new_my_bits ^ my_bits) & ~square_mask[sq];
  unsigned long long d = hash_put_packed[color][sq];

  while ( fl != 0 ) {
    int bit = FIRST_BIT( fl );
    fl &= fl - 1;
    d ^= hash_flip_bit[bit];
  }

  *diff1 = (unsigned int) d;
  *diff2 = (unsigned int) (d >> 32);
}

int
solve_parity( ThreadState *ts,
	      BitBoard my_bits,
	      BitBoard opp_bits,
	      int alpha,
	      int beta,
	      int color,
	      int empties,
	      int disc_diff,
	      int pass_legal );

#ifdef __cplusplus
}
#endif

#endif  /* END_LEAF_H */
