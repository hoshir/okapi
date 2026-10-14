/*
   File:          hash.h

   Created:       June 29, 1997

   Author:        Gunnar Andersson (gunnar@radagast.se)
		  Toshihiko Okuhara

   Contents:      Routines manipulating the hash table
*/



#ifndef HASH_H
#define HASH_H



#include "constant.h"
#include "macros.h"
#include "tlstate.h"



#ifdef __cplusplus
extern "C" {
#endif



#define LOWER_BOUND               1
#define UPPER_BOUND               2
#define EXACT_VALUE               4
#define MIDGAME_SCORE             8
#define ENDGAME_SCORE             16
#define SELECTIVE                 32
#define HEURISTIC_PRESEARCH_MOVE  64


#define ENDGAME_MODE              TRUE
#define MIDGAME_MODE              FALSE

#define NO_HASH_MOVE              0

#define DRAFT_VALUE_MASK          0x3F
#define DRAFT_GEN_MASK            0xC0
#define DRAFT_GEN_SHIFT           6

extern unsigned char hash_generation;

void
increment_hash_generation( void );



/* The structure returned when a hash probe resulted in a hit.
   DRAFT is the depth of the subtree beneath the node and FLAGS
   contains a flag mask (see the flag bits above). */
typedef struct {
   unsigned int key1, key2;
   int eval;
   int move[4];
   short draft;
   short selectivity;
   short flags;
} HashEntry;


typedef struct {
  unsigned int key2;
  int eval;
  unsigned int moves;
  unsigned int key1_selectivity_flags_draft;
} CompactHashEntry;


/* The number of entries in the hash table. Always a power of 2. */
extern int hash_size;
extern int hash_mask;
extern unsigned int hash_trans1;
extern unsigned int hash_trans2;
extern CompactHashEntry *hash_table;

extern int shallow_hash_mask;
extern CompactHashEntry *shallow_hash_table;


static INLINE void
prefetch_hash_endgame_key( unsigned int h2 ) {
  if ( hash_table != NULL ) {
    unsigned int code1 = h2 ^ hash_trans2;
    __builtin_prefetch( &hash_table[code1 & (hash_mask & ~3)], 0, 3 );
  }
}

static INLINE void
prefetch_hash_midgame_key( unsigned int h2 ) {
  if ( hash_table != NULL ) {
    unsigned int code1 = h2 ^ hash_trans2;
    __builtin_prefetch( &hash_table[code1 & (hash_mask & ~3)], 0, 3 );
  }
}

static INLINE void
prefetch_shallow_hash_key( unsigned int h2 ) {
  if ( shallow_hash_table != NULL ) {
    unsigned int code1 = h2 ^ hash_trans2;
    __builtin_prefetch( &shallow_hash_table[code1 & shallow_hash_mask], 0, 3 );
  }
}


/* The 64-bit hash masks for a piece of a certain color in a
   certain position. */
extern unsigned int hash_value1[3][128];
extern unsigned int hash_value2[3][128];

/* 64-bit hash masks used when a disc is played on the board;
   the relation
     hash_put_value?[][] == hash_value?[][] ^ hash_flip_color?
   is guaranteed to hold. */

extern unsigned int hash_put_value1[3][128];
extern unsigned int hash_put_value2[3][128];

/* XORs of hash_value* - used for disk flipping. */
extern unsigned int hash_flip1[128];
extern unsigned int hash_flip2[128];
extern unsigned long long hash_flip_bit[64];
extern unsigned long long hash_put_packed[3][128];

/* 64-bit hash mask for the two different sides to move. */
extern unsigned int hash_color1[3];
extern unsigned int hash_color2[3];

/* The XOR of the hash_color*, used for disk flipping. */
extern unsigned int hash_flip_color1;
extern unsigned int hash_flip_color2;




void
init_hash( int in_hash_bits );

void
resize_hash( int new_hash_bits );

void
setup_hash( int clear );

void
clear_hash_drafts( void );

void
free_hash( void );

void
determine_hash_values( int side_to_move,
		       const int *in_board );

void
set_hash_transformation( unsigned int trans1,
			 unsigned int trans2 );

void
add_hash( int reverse_mode,
	  int score,
	  int best,
	  int flags,
	  int draft,
	  int selectivity );

void
add_hash_ts( const ThreadState *ts,
	     int reverse_mode,
	     int score,
	     int best,
	     int flags,
	     int draft,
	     int selectivity );

void
add_hash_extended( int reverse_mode,
		   int score,
		   int *best,
		   int flags,
		   int draft,
		   int selectivity );

void
add_hash_extended_ts( const ThreadState *ts,
		      int reverse_mode,
		      int score,
		      int *best,
		      int flags,
		      int draft,
		      int selectivity );

void REGPARM(2)
find_hash( HashEntry *entry, int reverse_mode );

void
find_hash_key( HashEntry *entry, unsigned int h1, unsigned int h2 );

#define DEFAULT_SHALLOW_HASH_BITS    19
#define KEY1_MASK                    0xFF000000u

void
init_shallow_hash( int in_bits );

void
free_shallow_hash( void );

void
clear_shallow_hash( void );

static ALWAYS_INLINE void
find_shallow_hash( HashEntry *entry, const ThreadState *ts ) {
  int index;
  unsigned int code1, code2;
  unsigned int k2_raw, k1_p, mv;
  int ev;

  if ( shallow_hash_table == NULL ) {
    entry->draft = NO_HASH_MOVE;
    entry->flags = UPPER_BOUND;
    entry->eval = 12345678;
    entry->move[0] = 0;
    return;
  }

  code1 = ts->ts_hash2 ^ hash_trans2;
  code2 = ts->ts_hash1 ^ hash_trans1;

  index = code1 & shallow_hash_mask;

  k2_raw = __atomic_load_n( &shallow_hash_table[index].key2, __ATOMIC_ACQUIRE );
  ev = __atomic_load_n( &shallow_hash_table[index].eval, __ATOMIC_RELAXED );
  mv = __atomic_load_n( &shallow_hash_table[index].moves, __ATOMIC_RELAXED );
  k1_p = __atomic_load_n( &shallow_hash_table[index].key1_selectivity_flags_draft, __ATOMIC_RELAXED );

  if ( ((k2_raw ^ (unsigned int) ev ^ mv ^ k1_p) == code2) &&
       (((k1_p ^ code1) & KEY1_MASK) == 0) ) {
    entry->key2 = code2;
    entry->eval = ev;
    entry->move[0] = mv & 255;
    entry->move[1] = (mv >> 8) & 255;
    entry->move[2] = (mv >> 16) & 255;
    entry->move[3] = (mv >> 24) & 255;
    entry->key1 = k1_p & KEY1_MASK;
    entry->selectivity = (k1_p & 0x00ffffff) >> 16;
    entry->flags = (k1_p & 0x0000ffff) >> 8;
    entry->draft = (k1_p & DRAFT_VALUE_MASK);
    return;
  }

  entry->draft = NO_HASH_MOVE;
  entry->flags = UPPER_BOUND;
  entry->eval = 12345678;
  entry->move[0] = 0;
  entry->move[1] = 0;
  entry->move[2] = 0;
  entry->move[3] = 0;
}

static ALWAYS_INLINE void
add_shallow_hash( int score,
		  int best,
		  int flags,
		  int draft,
		  const ThreadState *ts ) {
  unsigned int index;
  unsigned int code1, code2;

  if ( shallow_hash_table == NULL )
    return;

  code1 = ts->ts_hash2 ^ hash_trans2;
  code2 = ts->ts_hash1 ^ hash_trans1;
  index = code1 & shallow_hash_mask;

  unsigned int clamped_draft = (unsigned int)(draft < 0 ? 0 : (draft > DRAFT_VALUE_MASK ? DRAFT_VALUE_MASK : draft));
  unsigned int draft_gen = clamped_draft | (((unsigned int) __atomic_load_n( &hash_generation, __ATOMIC_RELAXED ) & 0x03) << DRAFT_GEN_SHIFT);
  unsigned int k1_packed = (code1 & KEY1_MASK) + ((unsigned int)(short)flags << 8) + draft_gen;
  unsigned int moves = (unsigned int)best;
  unsigned int xor_key2 = code2 ^ (unsigned int) score ^ moves ^ k1_packed;

  __atomic_store_n( &shallow_hash_table[index].eval, score, __ATOMIC_RELAXED );
  __atomic_store_n( &shallow_hash_table[index].moves, moves, __ATOMIC_RELAXED );
  __atomic_store_n( &shallow_hash_table[index].key1_selectivity_flags_draft, k1_packed, __ATOMIC_RELAXED );
  __atomic_store_n( &shallow_hash_table[index].key2, xor_key2, __ATOMIC_RELEASE );
}

void dump_tt_distribution( const char *label );
void reset_tt_stats( void );
void print_tt_stats( const char *label );



#ifdef __cplusplus
}
#endif



#endif  /* HASH_H */
