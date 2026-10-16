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
extern unsigned long long *shallow_hash_table;


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

#define DEFAULT_SHALLOW_HASH_BITS    20
#define KEY1_MASK                    0xFF000000u

void
init_shallow_hash( int in_bits );

void
free_shallow_hash( void );

void
clear_shallow_hash( void );

/*
  64-bit Packed Shallow Hash Entry Layout (8 bytes/entry, partitioned by draft & 3):
  Bits [63:32]: code2 (32 bits)
  Bits [31:24]: code1 & 0xFF000000u (8 bits)
  Bits [23:16]: score as signed char (8 bits, range -64..+64)
  Bits [15: 8]: best move (8 bits, range 0..88)
  Bits [ 7: 4]: flags & 0x0F (4 bits: EXACT_VALUE=1, LOWER_BOUND=2, UPPER_BOUND=4)
  Bits [ 3: 0]: draft & 0x0F (4 bits: 5..8, 0=empty)
*/
#define SHALLOW_TAG_DRAFT_MASK 0xFFFFFFFFFF00000FULL

static ALWAYS_INLINE int
probe_shallow_hash_draft( const ThreadState *ts,
                          int draft,
                          int alpha,
                          int beta,
                          int *tt_move,
                          int *tt_val ) {
  if ( shallow_hash_table == NULL )
    return FALSE;

  unsigned int code1 = ts->ts_hash2 ^ hash_trans2;
  unsigned int code2 = ts->ts_hash1 ^ hash_trans1;
  unsigned int index = (code1 & (unsigned int)shallow_hash_mask) | (unsigned int)(draft & 3);
  unsigned long long expected = ((unsigned long long) code2 << 32) |
                                (unsigned long long)(code1 & KEY1_MASK) |
                                (unsigned long long)(draft & 0x0F);
  unsigned long long raw = __atomic_load_n( &shallow_hash_table[index], __ATOMIC_RELAXED );

  if ( (raw & SHALLOW_TAG_DRAFT_MASK) != expected )
    return FALSE;

  int ev = (int)(signed char)(raw >> 16);
  int mv = (int)((raw >> 8) & 0xFF);
  int flags = (int)((raw >> 4) & 0x0F);

  if ( (flags & EXACT_VALUE) ||
       ((flags & LOWER_BOUND) && ev >= beta) ||
       ((flags & UPPER_BOUND) && ev <= alpha) ) {
    ((ThreadState *)ts)->ts_end_best_move = mv;
    *tt_val = ev;
    return TRUE;
  }
  if ( (tt_move != NULL) && (flags & (LOWER_BOUND | EXACT_VALUE)) )
    *tt_move = mv;
  return FALSE;
}

static ALWAYS_INLINE void
add_shallow_hash( int score,
		  int best,
		  int flags,
		  int draft,
		  const ThreadState *ts ) {
  if ( shallow_hash_table == NULL )
    return;

  unsigned int code1 = ts->ts_hash2 ^ hash_trans2;
  unsigned int code2 = ts->ts_hash1 ^ hash_trans1;
  unsigned int index = (code1 & (unsigned int)shallow_hash_mask) | (unsigned int)(draft & 3);
  unsigned long long packed = ((unsigned long long) code2 << 32) |
                              (unsigned long long)(code1 & KEY1_MASK) |
                              ((unsigned long long)(unsigned char)(signed char)score << 16) |
                              ((unsigned long long)(best & 0xFF) << 8) |
                              ((unsigned long long)(flags & 0x0F) << 4) |
                              (unsigned long long)(draft & 0x0F);

  __atomic_store_n( &shallow_hash_table[index], packed, __ATOMIC_RELAXED );
}



#ifdef __cplusplus
}
#endif



#endif  /* HASH_H */
