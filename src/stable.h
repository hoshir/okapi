/*
   File:          stable.h

   Created:       March 20, 1999

   Authors:       Gunnar Andersson (gunnar@radagast.se)

   Contents:      Interface to the code which conservatively estimates
                  the number of stable (unflippable) discs.
*/



#ifndef STABLE_H
#define STABLE_H



#include "bitboard.h"



#ifdef __cplusplus
extern "C" {
#endif

#define CORNER_MASK   0x8100000000000081ull
#define BORDER_MASK   0xFF818181818181FFull
#define CENTRAL_MASK  0x007E7E7E7E7E7E00ull



extern _Thread_local BitBoard last_black_stable, last_white_stable;



typedef struct {
  BitBoard bits;
} EdgeIndices;

int
count_edge_stable_indexed( int color, BitBoard col_bits, BitBoard opp_bits, EdgeIndices *edges );

int
count_stable_indexed( int color, BitBoard col_bits, BitBoard opp_bits, const EdgeIndices *edges );


void
init_stable( void );

/*
  FIND_DEAD_SQUARES
  Returns a bitboard of empty squares that can never be legally played by either player
  for the remainder of the game.
  An empty square whose 8-neighborhood consists entirely of stable discs
  (or off-board borders) can never be flanked by either player.
*/
INLINE static BitBoard
find_dead_squares( BitBoard empties, BitBoard stable_both ) {
  if ( stable_both == 0 )
    return 0;

  const BitBoard not_file_a = 0xfefefefefefefefeull;
  const BitBoard not_file_h = 0x7f7f7f7f7f7f7f7full;
  BitBoard u = ~stable_both;

  BitBoard h_east = ((u << 1) | (u >> 7) | (u << 9)) & not_file_a;
  BitBoard h_west = ((u >> 1) | (u << 7) | (u >> 9)) & not_file_h;
  BitBoard v_span = (u << 8) | (u >> 8);

  return empties & ~(h_east | h_west | v_span);
}

/*
  DEAD_QUADRANT_PARITY
  Computes the 4-bit quadrant parity mask of all dead squares.
  Quadrant 1 (Top-Left):  0x0F0F0F0F (bit 0)
  Quadrant 2 (Top-Right): 0xF0F0F0F0 (bit 1)
  Quadrant 4 (Bottom-Left): 0x0F0F0F0F00000000 (bit 2)
  Quadrant 8 (Bottom-Right): 0xF0F0F0F000000000 (bit 3)
*/
INLINE static unsigned int
dead_quadrant_parity( BitBoard dead ) {
  if ( dead == 0 )
    return 0;
  unsigned int p = 0;
  if ( non_iterative_popcount( dead & 0x0F0F0F0Full ) & 1 ) p |= 1;
  if ( non_iterative_popcount( dead & 0xF0F0F0F0ull ) & 1 ) p |= 2;
  if ( non_iterative_popcount( dead & 0x0F0F0F0F00000000ull ) & 1 ) p |= 4;
  if ( non_iterative_popcount( dead & 0xF0F0F0F000000000ull ) & 1 ) p |= 8;
  return p;
}



#ifdef __cplusplus
}
#endif



#endif  /* STABLE_H */
