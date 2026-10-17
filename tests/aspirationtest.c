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
  /* Even-score 1-disc fail-low/fail-high from [c - 1, c + 1] must still
     anchor the first step (d == 4) at center c (e.g. FFO #57: c = -8,
     window [-9, -7], root_eval = -10 -> [-12, -9], NOT [-14, -9]). */
  {
    int alpha = -9, beta = -7, delta = 4;
    end_aspiration_next_bounds( -10, &alpha, &beta, &delta );
    CHECK( alpha == -12 && beta == -9 && delta == 8,
	   "even 1st fail-low bounds from [-9,-7]: got [%d, %d] delta=%d",
	   alpha, beta, delta );
  }
  {
    int alpha = 9, beta = 11, delta = 4;
    end_aspiration_next_bounds( 12, &alpha, &beta, &delta );
    CHECK( alpha == 11 && beta == 14 && delta == 8,
	   "even 1st fail-high bounds from [9,11]: got [%d, %d] delta=%d",
	   alpha, beta, delta );
  }
  /* Multi-disc stability bound jump on 1st step (root_eval < *alpha - 1)
     must anchor at root_eval rather than center. */
  {
    int alpha = -9, beta = -7, delta = 4;
    end_aspiration_next_bounds( -14, &alpha, &beta, &delta );
    CHECK( alpha == -18 && beta == -13 && delta == 8,
	   "deep stability fail-low bounds from [-9,-7]: got [%d, %d] delta=%d",
	   alpha, beta, delta );
  }

  /* Exhaustive property check across all even centers c in [-60, 60]:
     1. Single-even-score first re-search invariant (the FFO #57 invariant):
        On a 1-disc fail-low from [c - 1, c + 1] (returning even c - 2 or odd c - 1),
        new_alpha must be MAX(-64, c - 4) so that c - 4 <= new_alpha (cutoff boundary)
        and at most ONE even score (c - 2) lies strictly inside (new_alpha, new_beta).
        Likewise on a 1-disc fail-high (returning even c + 2 or odd c + 1),
        new_beta must be MIN(64, c + 4) so that c + 4 >= new_beta.
     2. Full multi-step aspiration loop convergence across all even true_eval in [-64, 64]
        when the solver returns even fail-low/fail-high bounds. */
  {
    int c, true_eval;
    for ( c = -60; c <= 60; c += 2 ) {
      /* 1-disc fail-low returning even c - 2 */
      {
	int alpha = c - 1, beta = c + 1, delta = 4;
	int expected_alpha = (c - 4 < -64) ? -64 : (c - 4);
	int even_inside = 0, s;
	end_aspiration_next_bounds( c - 2, &alpha, &beta, &delta );
	CHECK( alpha == expected_alpha && beta == c - 1,
	       "c=%d even 1-disc fail-low: got [%d, %d], expected [%d, %d]",
	       c, alpha, beta, expected_alpha, c - 1 );
	for ( s = -64; s <= 64; s += 2 ) {
	  if ( s > alpha && s < beta )
	    even_inside++;
	}
	CHECK( even_inside <= 1,
	       "c=%d even 1-disc fail-low window (%d, %d) contains %d even scores (must be <= 1)",
	       c, alpha, beta, even_inside );
      }
      /* 1-disc fail-high returning even c + 2 */
      {
	int alpha = c - 1, beta = c + 1, delta = 4;
	int expected_beta = (c + 4 > 64) ? 64 : (c + 4);
	int even_inside = 0, s;
	end_aspiration_next_bounds( c + 2, &alpha, &beta, &delta );
	CHECK( alpha == c + 1 && beta == expected_beta,
	       "c=%d even 1-disc fail-high: got [%d, %d], expected [%d, %d]",
	       c, alpha, beta, c + 1, expected_beta );
	for ( s = -64; s <= 64; s += 2 ) {
	  if ( s > alpha && s < beta )
	    even_inside++;
	}
	CHECK( even_inside <= 1,
	       "c=%d even 1-disc fail-high window (%d, %d) contains %d even scores (must be <= 1)",
	       c, alpha, beta, even_inside );
      }
      /* Full aspiration convergence for every even true_eval in [-64, 64] */
      for ( true_eval = -64; true_eval <= 64; true_eval += 2 ) {
	int alpha = c - 1, beta = c + 1, delta = 4;
	int steps = 0;
	while ( 1 ) {
	  /* Alpha-beta with even endgame scores returns:
	     - true_eval if strictly inside (alpha, beta)
	     - largest even score <= alpha on fail-low
	     - smallest even score >= beta on fail-high */
	  int root_eval;
	  if ( true_eval <= alpha )
	    root_eval = (alpha >= 0) ? (alpha & ~1) : -((-alpha + 1) & ~1);
	  else if ( true_eval >= beta )
	    root_eval = (beta >= 0) ? ((beta + 1) & ~1) : -((-beta) & ~1);
	  else
	    root_eval = true_eval;

	  if ( root_eval > alpha && root_eval < beta ) {
	    CHECK( root_eval == true_eval,
		   "c=%d true=%d converged to wrong eval %d", c, true_eval, root_eval );
	    break;
	  }
	  if ( (alpha <= -64 && beta >= 64) || root_eval >= 64 || root_eval <= -64 ) {
	    CHECK( root_eval == true_eval,
		   "c=%d true=%d boundary exit with %d", c, true_eval, root_eval );
	    break;
	  }
	  int prev_alpha = alpha, prev_beta = beta;
	  end_aspiration_next_bounds( root_eval, &alpha, &beta, &delta );
	  steps++;
	  CHECK( alpha < beta && alpha >= -64 && beta <= 64,
		 "c=%d true=%d invalid bounds [%d, %d] at step %d",
		 c, true_eval, alpha, beta, steps );
	  if ( root_eval <= prev_alpha )
	    CHECK( beta <= prev_beta,
		   "c=%d true=%d beta widened on fail-low (%d -> %d)",
		   c, true_eval, prev_beta, beta );
	  if ( root_eval >= prev_beta )
	    CHECK( alpha >= prev_alpha,
		   "c=%d true=%d alpha widened on fail-high (%d -> %d)",
		   c, true_eval, prev_alpha, alpha );
	  CHECK( steps <= 6,
		 "c=%d true=%d failed to converge in 6 steps", c, true_eval );
	  if ( steps > 6 )
	    break;
	}
      }
    }
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
