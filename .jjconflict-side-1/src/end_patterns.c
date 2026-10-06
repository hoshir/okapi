/*
   File:       end_patterns.c
   Contents:   Ultra-lightweight 4-pattern move ordering model (EVAL-009).
               Pure 64-bit bitboard index extraction and 100% L2 cache resident evaluation.
*/

#include "end_patterns.h"
#include "end_patterns_data.h"

/*
   Square indices on the 64-bit bitboard (bit = 8 * row + col):
   Row 0: 0..7 (A1..H1)
   Row 1: 8..15 (A2..H2)
   ...
   Row 7: 56..63 (A8..H8)
*/

/* 4 Corner 3x3 (9 squares each, corner at index 8) */
static const int corner33_sqs[4][9] = {
  {18, 17, 16, 10, 9, 8, 2, 1, 0},
  {42, 41, 40, 50, 49, 48, 58, 57, 56},
  {21, 22, 23, 13, 14, 15, 5, 6, 7},
  {45, 46, 47, 53, 54, 55, 61, 62, 63}
};

/* 4 Edge + 2X (10 squares each: 2 X-squares followed by 8 edge squares) */
static const int edge2x_sqs[4][10] = {
  {49, 9, 56, 48, 40, 32, 24, 16, 8, 0},
  {54, 14, 63, 55, 47, 39, 31, 23, 15, 7},
  {14, 9, 7, 6, 5, 4, 3, 2, 1, 0},
  {54, 49, 63, 62, 61, 60, 59, 58, 57, 56}
};

/* 2 Diagonals (8 squares each) */
static const int diag8_sqs[2][8] = {
  {63, 54, 45, 36, 27, 18, 9, 0},
  {56, 49, 42, 35, 28, 21, 14, 7}
};

/* 8 2x4 Rectangles (8 squares each) */
static const int rect24_sqs[8][8] = {
  {11, 10, 9, 8, 3, 2, 1, 0},
  {51, 50, 49, 48, 59, 58, 57, 56},
  {12, 13, 14, 15, 4, 5, 6, 7},
  {52, 53, 54, 55, 60, 61, 62, 63},
  {25, 17, 9, 1, 24, 16, 8, 0},
  {30, 22, 14, 6, 31, 23, 15, 7},
  {33, 41, 49, 57, 32, 40, 48, 56},
  {38, 46, 54, 62, 39, 47, 55, 63}
};

/*
  Bitboard base-3 index extraction helpers.
  Square trit values:
    opp_bit == 1 (defender / child to move) -> trit 0
    neither set  (empty)                    -> trit 1
    my_bit == 1  (attacker / made move)     -> trit 2
  Formula: trit = 1 - opp_bit + my_bit
*/

static INLINE int
extract_9( const int *sqs, BitBoard my_bits, BitBoard opp_bits ) {
  int idx = 0;
  int i;
  for ( i = 0; i < 9; i++ ) {
    int s = sqs[i];
    int my_bit = (int)((my_bits >> s) & 1);
    int opp_bit = (int)((opp_bits >> s) & 1);
    idx = idx * 3 + (1 - opp_bit + my_bit);
  }
  return idx;
}

static INLINE int
extract_10( const int *sqs, BitBoard my_bits, BitBoard opp_bits ) {
  int idx = 0;
  int i;
  for ( i = 0; i < 10; i++ ) {
    int s = sqs[i];
    int my_bit = (int)((my_bits >> s) & 1);
    int opp_bit = (int)((opp_bits >> s) & 1);
    idx = idx * 3 + (1 - opp_bit + my_bit);
  }
  return idx;
}

static INLINE int
extract_8( const int *sqs, BitBoard my_bits, BitBoard opp_bits ) {
  int idx = 0;
  int i;
  for ( i = 0; i < 8; i++ ) {
    int s = sqs[i];
    int my_bit = (int)((my_bits >> s) & 1);
    int opp_bit = (int)((opp_bits >> s) & 1);
    idx = idx * 3 + (1 - opp_bit + my_bit);
  }
  return idx;
}

/*
  END_PATTERN_EVALUATE
  Sum of 18 pattern lookups (4 Corner 3x3, 4 Edge+2X, 8 2x4 Rect, 2 Diag 8).
  All weights reside 100% within L2 CPU cache (~184 KB).
*/
int
end_pattern_evaluate( BitBoard my_bits, BitBoard opp_bits ) {
  int score = 0;
  const int16_t *c33_w = &end_pattern_weights[END_PATTERN_CORNER33_OFFSET];
  const int16_t *e2x_w = &end_pattern_weights[END_PATTERN_EDGE2X_OFFSET];
  const int16_t *r24_w = &end_pattern_weights[END_PATTERN_RECT24_OFFSET];
  const int16_t *d8_w  = &end_pattern_weights[END_PATTERN_DIAG8_OFFSET];

  /* 4 Corner 3x3 */
  score += c33_w[extract_9( corner33_sqs[0], my_bits, opp_bits )];
  score += c33_w[extract_9( corner33_sqs[1], my_bits, opp_bits )];
  score += c33_w[extract_9( corner33_sqs[2], my_bits, opp_bits )];
  score += c33_w[extract_9( corner33_sqs[3], my_bits, opp_bits )];

  /* 4 Edge + 2X */
  score += e2x_w[extract_10( edge2x_sqs[0], my_bits, opp_bits )];
  score += e2x_w[extract_10( edge2x_sqs[1], my_bits, opp_bits )];
  score += e2x_w[extract_10( edge2x_sqs[2], my_bits, opp_bits )];
  score += e2x_w[extract_10( edge2x_sqs[3], my_bits, opp_bits )];

  /* 8 2x4 Rectangles */
  score += r24_w[extract_8( rect24_sqs[0], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[1], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[2], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[3], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[4], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[5], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[6], my_bits, opp_bits )];
  score += r24_w[extract_8( rect24_sqs[7], my_bits, opp_bits )];

  /* 2 Diagonals */
  score += d8_w[extract_8( diag8_sqs[0], my_bits, opp_bits )];
  score += d8_w[extract_8( diag8_sqs[1], my_bits, opp_bits )];

  return score;
}
