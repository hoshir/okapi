/*
   File:          end_init.c

   Created:       2026

   Contents:      Endgame initialization, statistical tables, and configuration.
*/

#include "bitboard.h"
#include "constant.h"
#include "end_init.h"
#include "moves.h"

BitBoard neighborhood_mask[100];

const unsigned int quadrant_mask[100] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 1, 2, 2, 2, 2, 0,
  0, 1, 1, 1, 1, 2, 2, 2, 2, 0,
  0, 1, 1, 1, 1, 2, 2, 2, 2, 0,
  0, 1, 1, 1, 1, 2, 2, 2, 2, 0,
  0, 4, 4, 4, 4, 8, 8, 8, 8, 0,
  0, 4, 4, 4, 4, 8, 8, 8, 8, 0,
  0, 4, 4, 4, 4, 8, 8, 8, 8, 0,
  0, 4, 4, 4, 4, 8, 8, 8, 8, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

int earliest_wld_solve = 0;
int earliest_full_solve = 0;
int full_output_mode = TRUE;

/*
   SETUP_END
   Prepares the endgame solver for a new game.
   This means clearing a few status fields and computing tables.
*/

void
setup_end( void ) {
  int i, j;
  static const int dir_shift[8] = {1, -1, 7, -7, 8, -8, 9, -9};

  earliest_wld_solve = 0;
  earliest_full_solve = 0;
  full_output_mode = TRUE;

  /* Calculate the neighborhood masks */

  for ( i = 1; i <= 8; i++ )
    for ( j = 1; j <= 8; j++ ) {
      /* Create the neighborhood mask for the square POS */

      int pos = 10 * i + j;
      int shift = 8 * (i - 1) + (j - 1);
      unsigned int k;

      neighborhood_mask[pos] = 0;

      for ( k = 0; k < 8; k++ )
	if ( dir_mask[pos] & (1 << k) ) {
	  unsigned int neighbor = shift + dir_shift[k];
	  neighborhood_mask[pos] |= 1ull << neighbor;
	}
    }
}

/*
  GET_EARLIEST_WLD_SOLVE
  GET_EARLIEST_FULL_SOLVE
  Return the highest #empty when WLD and full solve respectively
  were completed (not initiated).
*/

int
get_earliest_wld_solve( void ) {
  return earliest_wld_solve;
}

int
get_earliest_full_solve( void ) {
  return earliest_full_solve;
}

/*
  SET_OUTPUT_MODE
  Toggles output of intermediate search status on/off.
*/

void
set_output_mode( int full ) {
  full_output_mode = full;
}
