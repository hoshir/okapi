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

  if ( failures != 0 ) {
    fprintf( stderr, "aspirationtest: %d failure(s)\n", failures );
    return 1;
  }
  printf( "aspirationtest: all checks passed\n" );
  return 0;
}
