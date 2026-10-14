/*
   File:          bitbcnt.h

   Created:       November 22, 1999
   
   Authors:       Gunnar Andersson (gunnar@radagast.se)

   Contents:
*/



#ifndef BITBCNT_H
#define BITBCNT_H



#include "bitboard.h"
#include "macros.h"

static ALWAYS_INLINE BitBoard
highest_guarded_bit_cnt( BitBoard b ) {
#if defined( __GNUC__ )
  return 0x8000000000000000ull >> __builtin_clzll( b | 1 );
#else
  BitBoard h = b | 1;
  h |= h >> 1;
  h |= h >> 2;
  h |= h >> 4;
  h |= h >> 8;
  h |= h >> 16;
  h |= h >> 32;
  return h - (h >> 1);
#endif
}

/* SQ is a board coordinate, 11..88.  Every square but SQ is assumed
   occupied: whatever is not in MY_BITS belongs to the opponent. */
static ALWAYS_INLINE int
CountFlips_bitboard( int sq, BitBoard my_bits ) {
  if ( (neighborhood_mask[sq] & ~my_bits) == 0 )
    return 0;

  const FlipRays *r = &flip_rays[sq];
  BitBoard flips = 0;
  BitBoard ray, t, f;
  int i;

  for ( i = 0; i < 4; i++ ) {
    ray = r->dn[i];
    t = my_bits & ray;
    f = ((t & -t) - 1) & ray;
    f &= -(BitBoard) (t != 0);
    flips |= f;
  }

  for ( i = 0; i < 4; i++ ) {
    ray = r->up[i];
    t = my_bits & ray;
    f = ray & ~((highest_guarded_bit_cnt( t ) << 1) - 1);
    f &= -(BitBoard) (t != 0);
    flips |= f;
  }

  return non_iterative_popcount( flips );
}

#endif  /* BITBCNT_H */
