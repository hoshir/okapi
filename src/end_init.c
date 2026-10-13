/*
   File:          end_init.c

   Created:       2026

   Contents:      Endgame initialization, statistical tables, and configuration.
*/

#include "bitboard.h"
#include "constant.h"
#include "end_init.h"


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
  earliest_wld_solve = 0;
  earliest_full_solve = 0;
  full_output_mode = TRUE;
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
