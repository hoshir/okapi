/*
   File:       end_patterns.h
   Contents:   Ultra-lightweight 3-group (16-pattern) move ordering model (EVAL-009..EVAL-013).
               Egaroucid-paradigm move ordering for deep endgame search.
*/

#ifndef END_PATTERNS_H
#define END_PATTERNS_H

#include "bitboard.h"
#include <stdint.h>

/* Pattern Group Offsets into end_pattern_weights[] */
#define END_PATTERN_CORNER33_OFFSET  0      /* 3^9  = 19683 entries (39.4 KB) */
#define END_PATTERN_EDGE2X_OFFSET    19683  /* 3^10 = 59049 entries (118.1 KB) */
#define END_PATTERN_RECT24_OFFSET    78732  /* 3^8  = 6561 entries  (13.1 KB) */
#define END_PATTERN_TOTAL_WEIGHTS    85293  /* Total: 85,293 int16_t = 170.6 KB */

extern const int16_t end_pattern_weights[END_PATTERN_TOTAL_WEIGHTS];

int
end_pattern_evaluate_e2x( BitBoard my_bits, BitBoard opp_bits );

int
end_pattern_evaluate( BitBoard my_bits, BitBoard opp_bits );

int
end_pattern_evaluate_c33_r24( BitBoard my_bits, BitBoard opp_bits );

#endif /* END_PATTERNS_H */
