/*
   File:       end_patterns.h
   Contents:   Ultra-lightweight 4-pattern move ordering model (EVAL-009).
               Egaroucid-paradigm move ordering for deep endgame search.
*/

#ifndef END_PATTERNS_H
#define END_PATTERNS_H

#include "bitboard.h"
#include <stdint.h>

#define END_PATTERN_CORNER33_COUNT 19683
#define END_PATTERN_EDGE2X_COUNT   59049
#define END_PATTERN_RECT24_COUNT   6561
#define END_PATTERN_DIAG8_COUNT    6561
#define END_PATTERN_TOTAL_WEIGHTS  91854

#define END_PATTERN_CORNER33_OFFSET 0
#define END_PATTERN_EDGE2X_OFFSET   19683
#define END_PATTERN_RECT24_OFFSET   (19683 + 59049)        /* 78732 */
#define END_PATTERN_DIAG8_OFFSET    (19683 + 59049 + 6561) /* 85293 */

/*
  END_PATTERN_EVALUATE
  Evaluates candidate move position using the 4-pattern distilled LTR model.
  my_bits:  bitboard of player who made the move (attacker)
  opp_bits: bitboard of opponent (defender / child's side to move)
  Returns score in fixed-point 1/128 disc units. Higher is better for my_bits.
*/
int end_pattern_evaluate( BitBoard my_bits, BitBoard opp_bits );

#endif /* END_PATTERNS_H */
