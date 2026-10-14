/*
   File:          bitbtest.h

   Created:       November 22, 1999
   
   Authors:       Gunnar Andersson (gunnar@radagast.se)

   Contents:
*/



#ifndef BITBTEST_H
#define BITBTEST_H



#include "bitboard.h"
#include "macros.h"
#include "tlstate.h"




/* SQ is a board coordinate, 11..88.  The mover's discs with the
   turned ones added land in BB_FLIPS, which the endgame reads back
   well after the call; callers that only want the flips of a move
   they are making must use the _TO form and keep out of that
   variable. */
/* The highest set bit of (b | 1) — bit 0 stands in for "none" and is
   voided by the t != 0 guard at the use site. */

static ALWAYS_INLINE BitBoard
highest_guarded_bit( BitBoard b ) {
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

static ALWAYS_INLINE int
TestFlips_bitboard_to( int sq, BitBoard my_bits, BitBoard opp_bits,
		       BitBoard *new_my_bits ) {
  if ( (neighborhood_mask[sq] & opp_bits) == 0 )
    return 0;
  const FlipRays *r = &flip_rays[sq];
  BitBoard flips = 0;
  BitBoard ray, t, f;
  int i;

  for ( i = 0; i < 4; i++ ) {
    /* Towards higher bits: the flip candidates sit below the nearest
       own disc (t & -t) and above the played square. */
    ray = r->dn[i];
    t = my_bits & ray;
    f = ((t & -t) - 1) & ray;
    f &= -(BitBoard) (t != 0);
    f &= -(BitBoard) ((f & ~opp_bits) == 0);
    flips |= f;
  }

  for ( i = 0; i < 4; i++ ) {
    /* Towards lower bits: above the nearest own disc instead. */
    ray = r->up[i];
    t = my_bits & ray;
    f = ray & ~((highest_guarded_bit( t ) << 1) - 1);
    f &= -(BitBoard) (t != 0);
    f &= -(BitBoard) ((f & ~opp_bits) == 0);
    flips |= f;
  }

  *new_my_bits = my_bits | flips | square_mask[sq];

  return non_iterative_popcount( flips );
}

int
TestFlips_bitboard( int sq, BitBoard my_bits, BitBoard opp_bits );



#endif  /* BITBTEST_H */
