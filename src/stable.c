/*
   File:          stable.c

   Created:       March 20, 1999

   Authors:       Gunnar Andersson (gunnar@radagast.se)
                  David John Summers
                  Toshihiko Okuhara

   Contents:      Code which conservatively estimates the number of
                  stable (unflippable) discs using the concept
		  "Zardoz stability" along with edge tables.

   This piece of software is released under the GPL.
   See the file COPYING for more information.
*/



#include "porting.h"

#include <stdio.h>



#include "bitboard.h"
#include "bitbtest.h"
#include "constant.h"
#include "end.h"
#include "macros.h"
#include "patterns.h"
#include "safemem.h"
#include "stable.h"



/* This constant is used in the DynP stuff for edge stability
   and simply denotes "value not known". */
#define  UNDETERMINED             -1

/* When this flag is set, the DynP tables are calculated and
   output and then the program is terminated. */
#define  DEBUG                    0



/* Global variables */

/* All discs determined as stable last time COUNT_STABLE was called
   for the two colors */
_Thread_local BitBoard last_black_stable, last_white_stable;



/* Local variables */

/* Direct 64 KB lookup table mapping player & opponent 8-bit edge patterns to player's stable discs */
static uint8_t edge_stable_table[256 * 256];



/*
  GET_FULL_LINES
  Computes the bitboard of squares whose line along DIR is completely filled
  with occupied discs from border to border using Kogge-Stone parallel prefix.
*/

static INLINE BitBoard
get_full_lines( BitBoard line, int dir ) {
  BitBoard full_l = line, full_r = line;
  BitBoard edge_l = line & BORDER_MASK, edge_r = line & BORDER_MASK;
  int d = dir;

  full_l &= edge_l | (full_l >> d); full_r &= edge_r | (full_r << d);
  edge_l |= edge_l >> d;            edge_r |= edge_r << d;
  d <<= 1;

  full_l &= edge_l | (full_l >> d); full_r &= edge_r | (full_r << d);
  edge_l |= edge_l >> d;            edge_r |= edge_r << d;
  d <<= 1;

  full_l &= edge_l | (full_l >> d); full_r &= edge_r | (full_r << d);

  return full_r & full_l;
}



/*
  EDGE_ZARDOZ_STABLE
  Determines the bit mask for the full set of transitive stable discs.
  Expands from stable edge discs and filled line intersections into
  the central board using bit-parallel propagation across all 4 axes.
*/

INLINE static void
edge_zardoz_stable( BitBoard *ss,
		    BitBoard dd,
		    BitBoard od ) {
  const BitBoard central_mask = dd & CENTRAL_MASK;
  if ( central_mask == 0 )
    return;

  const BitBoard disc = dd | od;
  const BitBoard full_h = get_full_lines( disc, 1 );
  const BitBoard full_v = get_full_lines( disc, 8 );
  const BitBoard full_d7 = get_full_lines( disc, 7 );
  const BitBoard full_d9 = get_full_lines( disc, 9 );

  BitBoard new_stable = *ss | (full_h & full_v & full_d7 & full_d9 & central_mask);

  BitBoard stable = 0;
  while ( (new_stable & ~stable) != 0 ) {
    stable |= new_stable;
    BitBoard stable_h = (stable >> 1) | (stable << 1) | full_h;
    BitBoard stable_v = (stable >> 8) | (stable << 8) | full_v;
    BitBoard stable_d7 = (stable >> 7) | (stable << 7) | full_d7;
    BitBoard stable_d9 = (stable >> 9) | (stable << 9) | full_d9;
    new_stable = stable_h & stable_v & stable_d7 & stable_d9 & central_mask;
  }

  *ss = stable;
}



INLINE static BitBoard
unpack_fileA( unsigned int t ) {
  BitBoard b = (((t & 0x0Fu) * 0x00204081u) & 0x01010101u);
  b |= ((BitBoard)(((t >> 4) * 0x00204081u) & 0x01010101u)) << 32;
  return b;
}

INLINE static BitBoard
unpack_fileH( unsigned int t ) {
  BitBoard b = (((t & 0x0Fu) * 0x10204080u) & 0x80808080u);
  b |= ((BitBoard)(((t >> 4) * 0x10204080u) & 0x80808080u)) << 32;
  return b;
}



/*
  COUNT_EDGE_STABLE_INDEXED
  Returns the number of stable edge discs for COLOR and writes the 64-bit
  stable edge bitboard into edges->bits.
*/

int
count_edge_stable_indexed( int color,
			   BitBoard col_bits,
			   BitBoard opp_bits,
			   EdgeIndices *edges ) {
  (void) color;
  unsigned int p_r1 = (unsigned int)(col_bits & 0xFF);
  unsigned int o_r1 = (unsigned int)(opp_bits & 0xFF);
  unsigned int p_r8 = (unsigned int)((col_bits >> 56) & 0xFF);
  unsigned int o_r8 = (unsigned int)((opp_bits >> 56) & 0xFF);

  unsigned int p_fA = (unsigned int)(((col_bits & 0x0101010101010101ull) * 0x0102040810204080ull) >> 56);
  unsigned int o_fA = (unsigned int)(((opp_bits & 0x0101010101010101ull) * 0x0102040810204080ull) >> 56);

  unsigned int p_fH = (unsigned int)((((col_bits >> 7) & 0x0101010101010101ull) * 0x0102040810204080ull) >> 56);
  unsigned int o_fH = (unsigned int)((((opp_bits >> 7) & 0x0101010101010101ull) * 0x0102040810204080ull) >> 56);

  BitBoard st = (BitBoard) edge_stable_table[p_r1 | (o_r1 << 8)]
              | ((BitBoard) edge_stable_table[p_r8 | (o_r8 << 8)] << 56)
              | unpack_fileA( edge_stable_table[p_fA | (o_fA << 8)] )
              | unpack_fileH( edge_stable_table[p_fH | (o_fH << 8)] );

  edges->bits = st;
  return non_iterative_popcount( st );
}






/*
  COUNT_STABLE_INDEXED
  Returns the number of stable discs for COLOR given the calculated EDGES.
*/

int
count_stable_indexed( int color,
		      BitBoard col_bits,
		      BitBoard opp_bits,
		      const EdgeIndices *edges ) {
  BitBoard col_stable = edges->bits;

  /* Expand the stable edge discs into a full set of stable discs */
  if ( (col_bits & CENTRAL_MASK) != 0 )
    edge_zardoz_stable( &col_stable, col_bits, opp_bits );

  if ( color == BLACKSQ )
    last_black_stable = col_stable;
  else
    last_white_stable = col_stable;

  return non_iterative_popcount( col_stable );
}






/*
  RECURSIVE_FIND_STABLE
  Returns a bit mask describing the set of stable discs in the
  edge PATTERN. Used only during engine initialization to populate
  the direct 64 KB edge_stable_table.
*/

static int
recursive_find_stable( int pattern, short *temp_edge_stable ) {
  int i, j;
  int new_pattern;
  int stable;
  int temp;
  int row[8], stored_row[8];

  if ( temp_edge_stable[pattern] != UNDETERMINED )
    return temp_edge_stable[pattern];

  temp = pattern;
  for ( i = 0; i < 8; i++, temp /= 3 )
    row[i] = temp % 3;

  /* All positions stable unless proved otherwise. */

  stable = 255;

  /* Play out the 8 different moves and AND together the stability masks. */

  for ( j = 0; j < 8; j++ )
    stored_row[j] = row[j];

  for ( i = 0; i < 8; i++ ) {

    /* Make sure we work with the original configuration */

    for ( j = 0; j < 8; j++ )
      row[j] = stored_row[j];

    if ( row[i] == EMPTY ) {  /* Empty ==> playable! */

      /* Mark the empty square as unstable and store position */

      stable &= ~(1 << i);

      /* Play out a black move */

      row[i] = BLACKSQ;
      if ( i >= 2 ) {
	j = i - 1;
	while ( (j >= 1) && (row[j] == WHITESQ) )
	  j--;
	if ( row[j] == BLACKSQ )
	  for ( j++; j < i; j++ ) {
	    row[j] = BLACKSQ;
	    stable &= ~(1 << j);
	  }
      }
      if ( i <= 5 ) {
	j = i + 1;
	while ( (j <= 6) && (row[j] == WHITESQ) )
	  j++;
	if ( row[j] == BLACKSQ )
	  for ( j--; j > i; j-- ) {
	    row[j] = BLACKSQ;
	    stable &= ~(1 << j);
	  }
      }
      new_pattern = 0;
      for ( j = 0; j < 8; j++ )
	new_pattern += pow3[j] * row[j];
      stable &= recursive_find_stable( new_pattern, temp_edge_stable );

      /* Restore position */

      for ( j = 0; j < 8; j++ )
	row[j] = stored_row[j];

      /* Play out a white move */

      row[i] = WHITESQ;
      if ( i >= 2 ) {
	j = i - 1;
	while ( (j >= 1) && (row[j] == BLACKSQ) )
	  j--;
	if ( row[j] == WHITESQ )
	  for ( j++; j < i; j++ ) {
	    row[j] = WHITESQ;
	    stable &= ~(1 << j);
	  }
      }
      if ( i <= 5 ) {
	j = i + 1;
	while ( (j <= 6) && (row[j] == BLACKSQ) )
	  j++;
	if ( row[j] == WHITESQ )
	  for ( j--; j > i; j-- ) {
	    row[j] = WHITESQ;
	    stable &= ~(1 << j);
	  }
      }
      new_pattern = 0;
      for ( j = 0; j < 8; j++ )
	new_pattern += pow3[j] * row[j];
      stable &= recursive_find_stable( new_pattern, temp_edge_stable );
    }
  }

  /* Store and return */

  temp_edge_stable[pattern] = stable;

  return stable;
}



/*
  INIT_STABLE
  Builds the direct 64 KB lookup table mapping player & opponent
  8-bit edge patterns to player's stable discs.
  Done once at engine launch; temporary base-3 DP tables are discarded.
*/

void
init_stable( void ) {
  short *temp_edge_stable;
  short base_conv[256];
  int i, j, p, o;

  for ( i = 0; i < 256; i++ ) {
    base_conv[i] = 0;
    for ( j = 0; j < 8; j++ )
      if ( i & (1 << j) )
	base_conv[i] += pow3[j];
  }

  temp_edge_stable = (short *) safe_malloc( 6561 * sizeof( short ) );

  for ( i = 0; i < 6561; i++ )
    temp_edge_stable[i] = UNDETERMINED;
  for ( i = 0; i < 6561; i++ )
    if ( temp_edge_stable[i] == UNDETERMINED )
      (void) recursive_find_stable( i, temp_edge_stable );

  for ( p = 0; p < 256; p++ ) {
    for ( o = 0; o < 256; o++ ) {
      int idx = p | (o << 8);
      if ( (p & o) != 0 ) {
	edge_stable_table[idx] = 0;
      } else {
	int pattern = 3280 - base_conv[p] + base_conv[o];
	edge_stable_table[idx] = (uint8_t)(temp_edge_stable[pattern] & p);
      }
    }
  }

  free( temp_edge_stable );
}
