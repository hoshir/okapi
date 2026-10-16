/*
   File:          aspirationtest.c
   Contents:      Unit test for end_aspiration_center(), the root null-window
                  center of the endgame search.

   Exact endgame scores are always even, so a window (c-1, c+1) centered
   on an odd c contains no reachable score. SRCH-031b found that forcing
   the center's parity to match the number of empties (odd for odd
   empties) made every odd-empties position pay a wide exact re-search,
   up to 71% of its nodes. This test pins the invariant down for the
   whole range of static evaluations.
*/

#include <stdio.h>
#include <stdlib.h>
#include "end_aspiration.h"

#define UNITS_PER_DISC  128

static int failures = 0;

#define CHECK( cond, ... )                                  \
  do {                                                      \
    if ( !(cond) ) {                                        \
      if ( failures < 10 ) {                                \
	fprintf( stderr, "FAIL: " __VA_ARGS__ );            \
	fprintf( stderr, "\n" );                            \
      }                                                     \
      failures++;                                           \
    }                                                       \
  } while ( 0 )

int
main( void ) {
  int limits[] = { 60, 62 };
  unsigned li;

  for ( li = 0; li < sizeof( limits ) / sizeof( limits[0] ); li++ ) {
    int limit = limits[li];
    int prev = -1000;
    int eval;

    /* Well beyond the +-64 disc range, to exercise the clamps. */
    for ( eval = -80 * UNITS_PER_DISC; eval <= 80 * UNITS_PER_DISC; eval++ ) {
      int c = end_aspiration_center( eval, limit );
      int mirror = end_aspiration_center( -eval, limit );

      CHECK( (c % 2) == 0,
	     "center %d for eval %d (limit %d) is odd", c, eval, limit );
      CHECK( c >= -limit && c <= limit,
	     "center %d for eval %d outside +-%d", c, eval, limit );
      CHECK( c >= prev,
	     "center not monotone at eval %d (%d after %d)", eval, c, prev );
      CHECK( c == -mirror,
	     "center not odd-symmetric at eval %d (%d vs %d)", eval, c, mirror );

      /* Within one disc of the evaluation (nearest even score), unless
	 clamped. */
      if ( abs( eval ) <= limit * UNITS_PER_DISC )
	CHECK( abs( c * UNITS_PER_DISC - eval ) <= UNITS_PER_DISC,
	       "center %d too far from eval %d", c, eval );
      prev = c;
    }
  }

  /* Spot checks of the rounding behaviour. */
  CHECK( end_aspiration_center( 0, 60 ) == 0, "zero eval" );
  CHECK( end_aspiration_center( 1 * UNITS_PER_DISC, 60 ) == 0 ||
	 end_aspiration_center( 1 * UNITS_PER_DISC, 60 ) == 2,
	 "odd-disc eval must land on an adjacent even score" );
  CHECK( end_aspiration_center( 3 * UNITS_PER_DISC, 60 ) == 2 ||
	 end_aspiration_center( 3 * UNITS_PER_DISC, 60 ) == 4,
	 "3-disc eval must land on 2 or 4" );
  CHECK( end_aspiration_center( 500 * UNITS_PER_DISC, 60 ) == 60,
	 "upper clamp" );
  CHECK( end_aspiration_center( -500 * UNITS_PER_DISC, 62 ) == -62,
	 "lower clamp" );

  /* Verify exact re-search opposite-bound tightening (SRCH-012):
     After initial null window [9, 11] centered at 10 fails low twice
     (first returning 9, then returning 5), beta must tighten to 6 (not reset to 11). */
  {
    int alpha = 9, beta = 11, delta = 4;
    end_aspiration_next_bounds( 9, &alpha, &beta, &delta );
    CHECK( alpha == 6 && beta == 10 && delta == 8,
	   "1st fail-low bounds: got [%d, %d] delta=%d", alpha, beta, delta );
    end_aspiration_next_bounds( 6, &alpha, &beta, &delta );
    CHECK( alpha == -2 && beta == 7 && delta == 16,
	   "2nd fail-low bounds: got [%d, %d] delta=%d", alpha, beta, delta );
  }
  {
    int alpha = 9, beta = 11, delta = 4;
    end_aspiration_next_bounds( 11, &alpha, &beta, &delta );
    CHECK( alpha == 10 && beta == 14 && delta == 8,
	   "1st fail-high bounds: got [%d, %d] delta=%d", alpha, beta, delta );
    end_aspiration_next_bounds( 14, &alpha, &beta, &delta );
    CHECK( alpha == 13 && beta == 22 && delta == 16,
	   "2nd fail-high bounds: got [%d, %d] delta=%d", alpha, beta, delta );
  }

  /* Verify selective step-center snapping on large bound returns: */
  CHECK( end_aspiration_step_center( 10, 9, -2 ) == 8,
	 "normal -2 step from 10 on eval=9" );
  CHECK( end_aspiration_step_center( 10, 2, -2 ) == 2,
	 "snap downward from 10 to even bound 2" );
  CHECK( end_aspiration_step_center( 10, 3, -2 ) == 2,
	 "snap downward from 10 on odd bound 3 to even 2" );
  CHECK( end_aspiration_step_center( 10, 11, 2 ) == 12,
	 "normal +2 step from 10 on eval=11" );
  CHECK( end_aspiration_step_center( 10, 18, 2 ) == 18,
	 "snap upward from 10 to even bound 18" );
  CHECK( end_aspiration_step_center( 10, 17, 2 ) == 18,
	 "snap upward from 10 on odd bound 17 to even 18" );

  if ( failures != 0 ) {
    fprintf( stderr, "aspirationtest: %d failure(s)\n", failures );
    return 1;
  }
  printf( "aspirationtest: all checks passed\n" );
  return 0;
}
