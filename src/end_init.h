/*
   File:          end_init.h

   Created:       2026

   Contents:      Endgame initialization, statistical tables, and configuration.
*/

#ifndef END_INIT_H
#define END_INIT_H

#include "bitboard.h"
#include "constant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Neighborhood bitboard mask for each square (1-88) */
extern BitBoard neighborhood_mask[100];

/* Quadrant parity mask for each square */
extern const unsigned int quadrant_mask[100];

/* Fastest-first thresholds and mobility factors */
extern int fast_first_threshold[61][64];
extern int ff_mob_factor[61];

/* Endgame solve depths and output mode */
extern int earliest_wld_solve;
extern int earliest_full_solve;
extern int full_output_mode;

void setup_end( void );
void set_output_mode( int full );
int get_earliest_wld_solve( void );
int get_earliest_full_solve( void );

#ifdef __cplusplus
}
#endif

#endif  /* END_INIT_H */
