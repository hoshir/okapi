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

/*
  COUNT_EDGE_STABLE
  Returns the number of stable edge discs for COLOR.
*/

int
count_edge_stable( int color, BitBoard col_bits, BitBoard opp_bits );

int
count_edge_stable_indexed( int color, BitBoard col_bits, BitBoard opp_bits, EdgeIndices *edges );


/*
  COUNT_STABLE
  Returns the number of stable discs for COLOR.
  Note: COUNT_EDGE_STABLE must have been called immediately
        before this function is called *or you lose big*.
*/

int
count_stable( int color, BitBoard col_bits, BitBoard opp_bits );

int
count_stable_indexed( int color, BitBoard col_bits, BitBoard opp_bits, const EdgeIndices *edges );


void
init_stable( void );

void
finalize_stable( void );



#ifdef __cplusplus
}
#endif



#endif  /* STABLE_H */
