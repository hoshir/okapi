/*
   File:          end_aspiration.c

   Contents:      Root aspiration window centering for the endgame search.
                  Kept free of search dependencies so that it can be unit
                  tested on its own (see tests/aspirationtest.c).
*/

#include "end_aspiration.h"



/*
  END_ASPIRATION_CENTER
  Converts a static evaluation (1 disc = 128 units) into the center of
  the root null window (center - 1, center + 1).

  Exact endgame scores are always even: the empty squares are awarded
  to the winner, so black + white = 64 and black - white is even. A
  window centered on an odd value therefore holds no reachable score;
  every probe degrades to a one-sided bound test and the selective
  passes hand the exact solve a center that is off by one, which forces
  a wide, expensive re-search. The center must be even.
  LIMIT must be even as well.
*/

int
end_aspiration_center( int eval, int limit ) {
  int center = (eval >= 0) ? 2 * ((eval + 128) / 256) :
			     -2 * ((-eval + 128) / 256);

  if ( center < -limit )
    center = -limit;
  if ( center > limit )
    center = limit;
  return center;
}

void
end_aspiration_next_bounds( int root_eval, int *alpha, int *beta, int *delta ) {
  int d = *delta;

  if ( root_eval <= *alpha ) {
    /* Fail low: score is proven <= root_eval, so tighten beta to root_eval + 1
       and widen alpha downward by delta (anchored at center on the first step
       from the [c - 1, c + 1] null window; because exact endgame scores are
       even, a 1-disc fail-low from odd *alpha = c - 1 returns c - 2 = *alpha - 1). */
    int new_beta = root_eval + 1;
    int anchor = (d == 4 && root_eval >= *alpha - 1) ? (*alpha + 1) : root_eval;
    int new_alpha = (d >= 64) ? -64 : (anchor - d);
    if ( new_alpha < -64 ) new_alpha = -64;
    if ( new_beta > 64 ) new_beta = 64;
    if ( new_alpha >= new_beta ) new_alpha = -64;
    *alpha = new_alpha;
    *beta = new_beta;
  }
  else if ( root_eval >= *beta ) {
    /* Fail high: score is proven >= root_eval, so tighten alpha to root_eval - 1
       and widen beta upward by delta (anchored at center on the first step
       from the [c - 1, c + 1] null window; a 1-disc fail-high from odd
       *beta = c + 1 returns c + 2 = *beta + 1). */
    int new_alpha = root_eval - 1;
    int anchor = (d == 4 && root_eval <= *beta + 1) ? (*beta - 1) : root_eval;
    int new_beta = (d >= 64) ? 64 : (anchor + d);
    if ( new_alpha < -64 ) new_alpha = -64;
    if ( new_beta > 64 ) new_beta = 64;
    if ( new_beta <= new_alpha ) new_beta = 64;
    *alpha = new_alpha;
    *beta = new_beta;
  }
  else if ( d >= 64 ) {
    *alpha = -64;
    *beta = 64;
  }

  d *= 2;
  if ( d > 64 ) d = 64;
  *delta = d;
}

int
end_aspiration_step_center( int center, int root_eval, int step_dir ) {
  int next_center = center + step_dir;
  if ( step_dir < 0 && root_eval < center - 1 ) {
    int snapped = (root_eval >= 0) ? (root_eval & ~1) : -((-root_eval + 1) & ~1);
    if ( snapped < next_center )
      next_center = snapped;
  }
  else if ( step_dir > 0 && root_eval > center + 1 ) {
    int snapped = (root_eval >= 0) ? ((root_eval + 1) & ~1) : -((-root_eval) & ~1);
    if ( snapped > next_center )
      next_center = snapped;
  }
  if ( next_center < -64 ) next_center = -64;
  if ( next_center > 64 ) next_center = 64;
  return next_center;
}
