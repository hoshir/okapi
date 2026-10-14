/*
   File:          counter.h

   Created:       March 29, 1999

   Author:        Gunnar Andersson (gunnar@radagast.se)

   Contents:      The interface to the counter code.
*/



#ifndef COUNTER_H
#define COUNTER_H



#include "macros.h"



#define INCREMENT_COUNTER( counter )              counter.lo++
#define INCREMENT_COUNTER_BY( counter, term )     counter.lo += (term)



typedef struct {
  unsigned int hi;
  unsigned int lo;
} CounterType;



#define DECIMAL_BASIS            100000000

static ALWAYS_INLINE void
reset_counter( CounterType *counter ) {
  counter->lo = 0;
  counter->hi = 0;
}

static ALWAYS_INLINE void
adjust_counter( CounterType *counter ) {
  while ( counter->lo >= DECIMAL_BASIS ) {
    counter->lo -= DECIMAL_BASIS;
    counter->hi++;
  }
}

static ALWAYS_INLINE double
counter_value( CounterType *counter ) {
  adjust_counter( counter );
  return ((double) DECIMAL_BASIS) * counter->hi + counter->lo;
}

static ALWAYS_INLINE void
add_counter( CounterType *sum, CounterType *term ) {
  sum->lo += term->lo;
  sum->hi += term->hi;
  adjust_counter( sum );
}



#endif  /* COUNTER_H */
