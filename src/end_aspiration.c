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
