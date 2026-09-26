/*
   File:          end_init.c

   Created:       2026

   Contents:      Endgame initialization, statistical tables, and configuration.
*/

#include <math.h>

#include "bitboard.h"
#include "constant.h"
#include "end_init.h"
#include "epcstat.h"
#include "moves.h"

#define FAST_FIRST_FACTOR            0.45
#define MOB_FACTOR                   460

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

int fast_first_threshold[61][64];
int ff_mob_factor[61];

int earliest_wld_solve = 0;
int earliest_full_solve = 0;
int full_output_mode = TRUE;

static double fast_first_mean[61][64];
static double fast_first_sigma[61][64];

/*
   SETUP_END
   Prepares the endgame solver for a new game.
   This means clearing a few status fields and computing tables.
*/

void
setup_end( void ) {
  double last_mean, last_sigma;
  double ff_threshold[61];
  double prelim_threshold[61][64];
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

  /* Set the fastest-first mobility encouragements and thresholds */

  for ( i = 0; i <= 60; i++ )
    ff_mob_factor[i] = MOB_FACTOR;
  for ( i = 0; i <= 60; i++ )
    ff_threshold[i] = FAST_FIRST_FACTOR;

  /* Calculate the alpha thresholds for using fastest-first for
     each #empty and shallow search depth. */

  for ( j = 0; j <= MAX_END_CORR_DEPTH; j++ ) {
    last_sigma = 100.0;  /* Infinity in disc difference */
    last_mean = 0.0;
    for ( i = 60; i >= 0; i-- ) {
      if ( end_stats_available[i][j] ) {
	last_mean = end_mean[i][j];
	last_sigma = ff_threshold[i] * end_sigma[i][j];
      }
      fast_first_mean[i][j] = last_mean;
      fast_first_sigma[i][j] = last_sigma;
      prelim_threshold[i][j] = last_mean + last_sigma;
    }
  }
  for ( j = MAX_END_CORR_DEPTH + 1; j < 64; j++ )
    for ( i = 0; i <= 60; i++ )
      prelim_threshold[i][j] = prelim_threshold[i][MAX_END_CORR_DEPTH];
  for ( i = 0; i <= 60; i++ )
    for ( j = 0; j < 64; j++ )
      fast_first_threshold[i][j] =
	(int) ceil( prelim_threshold[i][j] * 128.0 );
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
