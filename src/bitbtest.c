/*
   File:          bitbtest.c

   Authors:       Gunnar Andersson (gunnar@radagast.se)
	          Toshihiko Okuhara

   Contents:      Count flips and returns new_my_bits in bb_flips.

   The 32-bit implementation carried one hand-unrolled function per
   square, most of whose bulk dealt with flip lines crossing the
   boundary between the two halves of the board.  With the board in a
   single 64-bit word there is no boundary, and one branch-free routine
   working from precomputed ray masks covers every square: along each
   ray the discs between the played square and the nearest own disc are
   flipped if and only if they are all the opponent's.

   This piece of software is released under the GPL.
   See the file COPYING for more information.
*/

#include <stdlib.h>

#include "macros.h"
#include "bitboard.h"
#include "bitbtest.h"




/*
  TestFlips_bitboard_to is inlined in bitbtest.h.
*/


/*
  The endgame keeps the result in BB_FLIPS and reads it back some way
  after the call -- across a shallow midgame search, in the
  fastest-first ordering -- so anything that flips discs on the side
  must leave that variable alone and use TestFlips_bitboard_to.
*/

int
TestFlips_bitboard( int sq, BitBoard my_bits, BitBoard opp_bits ) {
  return TestFlips_bitboard_to( sq, my_bits, opp_bits, &bb_flips );
}
