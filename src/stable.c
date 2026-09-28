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

#if defined( __ARM_NEON )
#include <arm_neon.h>
#elif defined( __AVX2__ )
#include <immintrin.h>
#endif

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
  FILLED_DIAGONALS
  Computes full lines along the two diagonal directions:
  NE-SW (step 7) into *daf_out and NW-SE (step 9) into *dbf_out.
*/

#if defined( __ARM_NEON )
INLINE static void
filled_diagonals( BitBoard occupied, BitBoard *daf_out, BitBoard *dbf_out ) {
  const uint64x2_t edge = vdupq_n_u64( occupied & BORDER_MASK );
  const int64x2_t shift_r = vcombine_s64( vcreate_s64( -7 ), vcreate_s64( -9 ) );
  const int64x2_t shift_l = vcombine_s64( vcreate_s64(  7 ), vcreate_s64(  9 ) );
  uint64x2_t full = vdupq_n_u64( occupied );

  uint64x2_t nb;
  #define NEON_ROUND() do { \
    nb = vandq_u64( vshlq_u64( full, shift_r ), vshlq_u64( full, shift_l ) ); \
    full = vandq_u64( full, vorrq_u64( nb, edge ) ); \
  } while (0)

  NEON_ROUND();
  NEON_ROUND();
  NEON_ROUND();
  NEON_ROUND();
  NEON_ROUND();
  #undef NEON_ROUND

  nb = vandq_u64( vshlq_u64( full, shift_r ), vshlq_u64( full, shift_l ) );
  uint64x2_t res = vorrq_u64( vdupq_n_u64( BORDER_MASK ), nb );
  *daf_out = vgetq_lane_u64( res, 0 );
  *dbf_out = vgetq_lane_u64( res, 1 );
}
#elif defined( __AVX2__ )
INLINE static void
filled_diagonals( BitBoard occupied, BitBoard *daf_out, BitBoard *dbf_out ) {
  const __m128i edge = _mm_set_epi64x( occupied & BORDER_MASK, occupied & BORDER_MASK );
  const __m128i shift_r_counts = _mm_set_epi64x( 9, 7 );
  const __m128i shift_l_counts = _mm_set_epi64x( 9, 7 );
  __m128i full = _mm_set_epi64x( occupied, occupied );

  #define AVX2_ROUND() do { \
    __m128i sr = _mm_srlv_epi64( full, shift_r_counts ); \
    __m128i sl = _mm_sllv_epi64( full, shift_l_counts ); \
    __m128i nb = _mm_and_si128( sr, sl ); \
    full = _mm_and_si128( full, _mm_or_si128( nb, edge ) ); \
  } while (0)

  AVX2_ROUND();
  AVX2_ROUND();
  AVX2_ROUND();
  AVX2_ROUND();
  AVX2_ROUND();
  #undef AVX2_ROUND

  __m128i sr = _mm_srlv_epi64( full, shift_r_counts );
  __m128i sl = _mm_sllv_epi64( full, shift_l_counts );
  __m128i res = _mm_or_si128( _mm_set_epi64x( BORDER_MASK, BORDER_MASK ), _mm_and_si128( sr, sl ) );

  *daf_out = (BitBoard)_mm_cvtsi128_si64( res );
  *dbf_out = (BitBoard)_mm_extract_epi64( res, 1 );
}
#else
INLINE static BitBoard
filled_lines( BitBoard occupied, int dir ) {
  const BitBoard edge = occupied & BORDER_MASK;
  BitBoard full;

  full  = occupied & (((occupied >> dir) & (occupied << dir)) | edge);
  full &= (((full >> dir) & (full << dir)) | edge);
  full &= (((full >> dir) & (full << dir)) | edge);
  full &= (((full >> dir) & (full << dir)) | edge);
  full &= (((full >> dir) & (full << dir)) | edge);

  return (full >> dir) & (full << dir);
}

INLINE static void
filled_diagonals( BitBoard occupied, BitBoard *daf_out, BitBoard *dbf_out ) {
  *daf_out = BORDER_MASK | filled_lines( occupied, 7 );
  *dbf_out = BORDER_MASK | filled_lines( occupied, 9 );
}
#endif


/*
  EDGE_ZARDOZ_STABLE
  Determines the bit mask for (a subset of) the stable discs in a position.
  Zardoz' algorithm + edge tables is used.
*/

INLINE static void
edge_zardoz_stable( BitBoard *ss,
		    BitBoard dd,
		    BitBoard od ) {
  const BitBoard central_mask = dd & CENTRAL_MASK;
  if ( central_mask == 0 )
    return;

  BitBoard ost, fb, lrf, udf, daf, dbf;
  BitBoard expand_ss;
  BitBoard t;

  fb = dd | od;

  /* A filled row protects its squares from horizontal flips; the a-
     and h-files never flip horizontally. */

  t = fb;
  t &= t >> 4;
  t &= t >> 2;
  t &= t >> 1;
  lrf = ((t & 0x0101010101010101ull) * 255) | 0x8181818181818181ull;

  /* Filled columns, by folding the rotations: afterwards a bit is set
     iff its whole column is.  Rows 1 and 8 never flip vertically. */

  t = fb;
  t &= (t >> 32) | (t << 32);
  t &= (t >> 16) | (t << 48);
  t &= (t >> 8) | (t << 56);
  udf = t | 0xFF000000000000FFull;

  /* Filled diagonals.  The border squares need no diagonal
     protection, which also covers the short diagonals. */

  filled_diagonals( fb, &daf, &dbf );

  *ss |= lrf & udf & daf & dbf & dd;

  if ( *ss == 0 )
    return;

  do {
    ost = *ss;

    BitBoard d1 = lrf | (ost << 1) | (ost >> 1);
    BitBoard d8 = udf | (ost << 8) | (ost >> 8);
    BitBoard d7 = daf | (ost << 7) | (ost >> 7);
    BitBoard d9 = dbf | (ost << 9) | (ost >> 9);

    expand_ss = d1 & d8 & d7 & d9;
    *ss = ost | (expand_ss & central_mask);
  } while ( ost != *ss );	/* changing */
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
  COUNT_EDGE_STABLE
  Computes the number of stable edge discs for COLOR.
*/

int
count_edge_stable( int color,
		   BitBoard col_bits,
		   BitBoard opp_bits ) {
  EdgeIndices e;
  return count_edge_stable_indexed( color, col_bits, opp_bits, &e );
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
  if ( (col_bits & CENTRAL_MASK) != 0 && col_stable != 0 )
    edge_zardoz_stable( &col_stable, col_bits, opp_bits );

  if ( color == BLACKSQ )
    last_black_stable = col_stable;
  else
    last_white_stable = col_stable;

  if ( col_stable != 0 )
    return non_iterative_popcount( col_stable );
  else
    return 0;
}



/*
  COUNT_STABLE
  Legacy entry point computing edge discs and expanding full stability.
*/

int
count_stable( int color,
	      BitBoard col_bits,
	      BitBoard opp_bits ) {
  EdgeIndices e;
  (void) count_edge_stable_indexed( color, col_bits, opp_bits, &e );
  return count_stable_indexed( color, col_bits, opp_bits, &e );
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
