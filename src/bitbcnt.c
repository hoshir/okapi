/*
   File:          bitbcnt.c

   Authors:       Gunnar Andersson (gunnar@radagast.se)
	          Toshihiko Okuhara

   Contents:      Count the number of discs flipped by a move when
                  every square except the one being played is occupied,
                  so everything that is not the mover's is the
                  opponent's.  The endgame leaf code uses this for the
                  last empty square.

   Works from the same precomputed ray masks as TestFlips_bitboard in
   bitbtest.c; with the board full, the all-opponent test on the run of
   discs before the nearest own disc always holds, so it is dropped.

   This piece of software is released under the GPL.
   See the file COPYING for more information.
*/

#include "macros.h"
#include "bitboard.h"
#include "bitbcnt.h"

/* CountFlips_bitboard is inlined via bitbcnt.h */
void bitbcnt_anchor( void ) {}

