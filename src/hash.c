/*
   File:          hash.c

   Created:       June 29, 1997

   Author:        Gunnar Andersson (gunnar@radagast.se)
		  Toshihiko Okuhara

   Contents:      Routines manipulating the hash table

   This piece of software is released under the GPL.
   See the file COPYING for more information.
*/



#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "hash.h"
#include "macros.h"
#include "myrandom.h"
#include "safemem.h"
#include "search.h"



/* Substitute an old position with draft n if the new position has
   draft >= n-4 */
#define REPLACEMENT_OFFSET          4

/* An 8-bit mask to isolate the "draft" part of the hash table info. */
#define DRAFT_MASK                  255

#define KEY1_MASK                   0xFF000000u

#define BUCKET_SIZE                 4
#define BUCKET_MASK( mask )         ((mask) & ~3)



/* Global variables */

unsigned char hash_generation = 0;

void
increment_hash_generation( void ) {
  hash_generation = (hash_generation + 1) & 0x03;
}

int hash_size;
int hash_mask;
unsigned int hash_trans1 = 0;
unsigned int hash_trans2 = 0;
CompactHashEntry *hash_table;

int shallow_hash_mask = 0;
CompactHashEntry *shallow_hash_table = NULL;

unsigned int hash_value1[3][128];
unsigned int hash_value2[3][128];
unsigned int hash_put_value1[3][128];
unsigned int hash_put_value2[3][128];
unsigned int hash_flip1[128];
unsigned int hash_flip2[128];
unsigned int hash_color1[3];
unsigned int hash_color2[3];
unsigned int hash_flip_color1;
unsigned int hash_flip_color2;
unsigned int hash_diff1[MAX_SEARCH_DEPTH];
unsigned int hash_diff2[MAX_SEARCH_DEPTH];



/* Local variables */

static int hash_bits;
static int rehash_count;

static int shallow_hash_bits = 0;
static int shallow_hash_size = 0;

/* ========================================================================= */
/* Transposition Table Diagnostics (TT_DIAG)                                 */
/* ========================================================================= */

typedef struct {
  unsigned long long probes;
  unsigned long long probe_hits;
  unsigned long long probe_misses;
  unsigned long long probe_miss_bucket_full;
  unsigned long long probe_miss_empty_slots;
  unsigned long long stores_attempted;
  unsigned long long stores_updated;
  unsigned long long stores_empty_slot;
  unsigned long long stores_evicted;
  unsigned long long stores_rejected;
  unsigned long long evicted_draft_counts[64];
  unsigned long long evicted_by_draft_counts[64];
} TTDiagnostics;

static TTDiagnostics tt_stats;
static int tt_diag_inited = 0;
static int tt_diag_active = 0;

static INLINE void init_tt_diag_if_needed( void ) {
  if ( !tt_diag_inited ) {
    tt_diag_active = (getenv( "TT_DIAG" ) != NULL);
    tt_diag_inited = 1;
  }
}



/*
   INIT_SHALLOW_HASH
   Allocate memory for the shallow hash table.
*/

void
init_shallow_hash( int in_bits ) {
  if ( shallow_hash_table != NULL )
    free_shallow_hash();
  shallow_hash_bits = in_bits;
  shallow_hash_size = 1 << shallow_hash_bits;
  shallow_hash_mask = shallow_hash_size - 1;
  shallow_hash_table =
    (CompactHashEntry *) safe_calloc( shallow_hash_size, sizeof( CompactHashEntry ) );
}


/*
   FREE_SHALLOW_HASH
*/

void
free_shallow_hash( void ) {
  if ( shallow_hash_table != NULL ) {
    free( shallow_hash_table );
    shallow_hash_table = NULL;
    shallow_hash_size = 0;
    shallow_hash_mask = 0;
  }
}


/*
   CLEAR_SHALLOW_HASH
*/

void
clear_shallow_hash( void ) {
  if ( shallow_hash_table != NULL ) {
    int i;
    for ( i = 0; i < shallow_hash_size; i++ )
      shallow_hash_table[i].key1_selectivity_flags_draft &= ~DRAFT_MASK;
  }
}



/*
   INIT_HASH
   Allocate memory for the hash table.
*/

void
init_hash( int in_hash_bits ) {
  size_t bytes;

  hash_bits = in_hash_bits;
  hash_size = 1 << hash_bits;
  hash_mask = hash_size - 1;
  bytes = (size_t) hash_size * sizeof( CompactHashEntry );
  hash_table =
    (CompactHashEntry *) safe_memalign( 64, bytes );
  memset( hash_table, 0, bytes );
  rehash_count = 0;
  init_shallow_hash( DEFAULT_SHALLOW_HASH_BITS );
}


/*
  RESIZE_HASH
  Changes the size of the hash table.
*/

void
resize_hash( int new_hash_bits ) {
  free( hash_table );
  init_hash( new_hash_bits );
  setup_hash( TRUE );
}



/*
  POPCOUNT
*/

static unsigned int
popcount( unsigned int b ) {
#if defined( __GNUC__ )
  return __builtin_popcount( b );
#else
  unsigned int n;

  for ( n = 0; b != 0; n++, b &= (b - 1) )
    ;

  return n;
#endif
}



/*
  GET_CLOSENESS
  Returns the closeness between the 64-bit integers (a0,a1) and (b0,b1).
  A closeness of 0 means that 32 bits differ.
*/


static unsigned int
get_closeness( unsigned int a0, unsigned int a1,
	   unsigned int b0, unsigned int b1 ) {
  return abs( (int) ( popcount( a0 ^ b0 ) + popcount( a1 ^ b1 ) ) - 32 );
}



/*
   SETUP_HASH
   Determine randomized hash masks.
*/   

void
setup_hash( int clear ) {
  int i, j;
  int pos;
  int rand_index;
  const unsigned int max_pair_closeness = 10;
  const unsigned int max_zero_closeness = 9;
  unsigned int closeness;
  unsigned int random_pair[130][2];

  if ( clear ) {
    memset( hash_table, 0, (size_t) hash_size * sizeof( CompactHashEntry ) );
    hash_generation = 0;
  }

  rand_index = 0;
  while ( rand_index < 130 ) {
  TRY_AGAIN:
    random_pair[rand_index][0] = (my_random() << 3) + (my_random() >> 2);
    random_pair[rand_index][1] = (my_random() << 3) + (my_random() >> 2);

    closeness =
      get_closeness( random_pair[rand_index][0], random_pair[rand_index][1],
		     0, 0 );
    if ( closeness > max_zero_closeness )
      goto TRY_AGAIN;
    for ( i = 0; i < rand_index; i++ ) {
      closeness =
	get_closeness( random_pair[rand_index][0], random_pair[rand_index][1],
		       random_pair[i][0], random_pair[i][1] );
      if ( closeness > max_pair_closeness )
	goto TRY_AGAIN;
      closeness =
	get_closeness( random_pair[rand_index][0], random_pair[rand_index][1],
		       random_pair[i][1], random_pair[i][0] );
      if ( closeness > max_pair_closeness )
	goto TRY_AGAIN;
    }
    rand_index++;
  }

  rand_index = 0;

  for ( i = 0; i < 128; i++ ) {
    hash_value1[BLACKSQ][i] = 0;
    hash_value2[BLACKSQ][i] = 0;
    hash_value1[WHITESQ][i] = 0;
    hash_value2[WHITESQ][i] = 0;
  }
  for ( i = 1; i <= 8; i++ )
    for ( j = 1; j <= 8; j++ ) {
      pos = 10 * i + j;
      hash_value1[BLACKSQ][pos] = random_pair[rand_index][0];
      hash_value2[BLACKSQ][pos] = random_pair[rand_index][1];
      rand_index++;
      hash_value1[WHITESQ][pos] = random_pair[rand_index][0];
      hash_value2[WHITESQ][pos] = random_pair[rand_index][1];
      rand_index++;
    }
  for ( i = 0; i < 128; i++ ) {
    hash_flip1[i] = hash_value1[BLACKSQ][i] ^ hash_value1[WHITESQ][i];
    hash_flip2[i] = hash_value2[BLACKSQ][i] ^ hash_value2[WHITESQ][i];
  }

  hash_color1[BLACKSQ] = random_pair[rand_index][0];
  hash_color2[BLACKSQ] = random_pair[rand_index][1];
  rand_index++;
  hash_color1[WHITESQ] = random_pair[rand_index][0];
  hash_color2[WHITESQ] = random_pair[rand_index][1];
  rand_index++;

  hash_flip_color1 = hash_color1[BLACKSQ] ^ hash_color1[WHITESQ];
  hash_flip_color2 = hash_color2[BLACKSQ] ^ hash_color2[WHITESQ];

  for ( j = 0; j < 128; j++ ) {
    hash_put_value1[BLACKSQ][j] = hash_value1[BLACKSQ][j] ^ hash_flip_color1;
    hash_put_value2[BLACKSQ][j] = hash_value2[BLACKSQ][j] ^ hash_flip_color2;
    hash_put_value1[WHITESQ][j] = hash_value1[WHITESQ][j] ^ hash_flip_color1;
    hash_put_value2[WHITESQ][j] = hash_value2[WHITESQ][j] ^ hash_flip_color2;
  }
}


/*
  CLEAR_HASH_DRAFTS
  Resets the draft information for all entries in the hash table.
*/

void
clear_hash_drafts( void ) {
  int i;

  for ( i = 0; i < hash_size; i++ )  /* Set the draft to 0 */
    hash_table[i].key1_selectivity_flags_draft &= ~DRAFT_MASK;
  clear_shallow_hash();
}


/*
   FREE_HASH
   Remove the hash table.
*/   

void
free_hash( void ) {
  free( hash_table );
  free_shallow_hash();
}


/*
   DETERMINE_HASH_VALUES
   Calculates the hash codes for the given board position.
*/

void
determine_hash_values( int side_to_move,
		       const int *in_board ) {
  int i, j;

  hash1 = 0;
  hash2 = 0;
  for ( i = 1; i <= 8; i++ )
    for ( j = 1; j <= 8; j++ ) {
      int pos = 10 * i + j;
      switch ( in_board[pos] ) {
      case BLACKSQ:
        hash1 ^= hash_value1[BLACKSQ][pos];
        hash2 ^= hash_value2[BLACKSQ][pos];
        break;

      case WHITESQ:
        hash1 ^= hash_value1[WHITESQ][pos];
        hash2 ^= hash_value2[WHITESQ][pos];
        break;

      default:
        break;
      }
    }
  hash1 ^= hash_color1[side_to_move];
  hash2 ^= hash_color2[side_to_move];
}


/*
   Concurrent / Lockless Safety in Transposition Table:
   Integrity is guaranteed by the XOR checksum:
     key2_stored = key2 ^ eval ^ moves ^ key1_selectivity_flags_draft
   If a reader observes an entry mid-write (where some fields are from an old
   entry and others from a new entry), the equation:
     (raw_key2 ^ eval ^ moves ^ key1_packed) == code2
   fails with probability 1 - 2^-32, rejecting torn reads without lock overhead.

   Note on memory ordering: The atomic builtins (__ATOMIC_ACQUIRE / __ATOMIC_RELEASE)
   ensure clean compiler optimization barriers and avoid C data races. However,
   because a reader may interleave with a writer mid-entry, the fundamental
   basis for torn-read safety across multiple fields is the XOR checksum itself,
   not happens-before synchronization.
*/



static INLINE void
wide_to_compact( const HashEntry *entry, CompactHashEntry *compact_entry ) {
  unsigned int clamped_draft = (unsigned int)(entry->draft < 0 ? 0 : (entry->draft > DRAFT_VALUE_MASK ? DRAFT_VALUE_MASK : entry->draft));
  unsigned int draft_gen = clamped_draft | (((unsigned int) __atomic_load_n( &hash_generation, __ATOMIC_RELAXED ) & 0x03) << DRAFT_GEN_SHIFT);
  unsigned int k1_packed = (entry->key1 & KEY1_MASK) + (entry->selectivity << 16) +
    (entry->flags << 8) + draft_gen;
  unsigned int moves = entry->move[0] + (entry->move[1] << 8) +
    (entry->move[2] << 16) + (entry->move[3] << 24);
  int eval = entry->eval;
  unsigned int xor_key2 = entry->key2 ^ (unsigned int) eval ^ moves ^ k1_packed;

  __atomic_store_n( &compact_entry->eval, eval, __ATOMIC_RELAXED );
  __atomic_store_n( &compact_entry->moves, moves, __ATOMIC_RELAXED );
  __atomic_store_n( &compact_entry->key1_selectivity_flags_draft, k1_packed, __ATOMIC_RELAXED );
  __atomic_store_n( &compact_entry->key2, xor_key2, __ATOMIC_RELEASE );
}


static INLINE void
compact_to_wide( HashEntry *entry, unsigned int code2, int ev, unsigned int mv, unsigned int k1_p ) {
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
}


/*
  SET_HASH_TRANSFORMATION
  Specify the hash code transformation masks. Changing these masks
  is the poor man's way to achieve the effect of clearing the hash
  table.
*/

void
set_hash_transformation( unsigned int trans1, unsigned int trans2 ) {
  hash_trans1 = trans1;
  hash_trans2 = trans2;
}


#define SHALLOW_NO_EVICT_DEFAULT_DRAFT  9

/*
   ADD_HASH
   Add information to the hash table using 4-way set-associativity.
   The shallowest search in the 4-entry bucket is replaced.
*/

void
add_hash( int reverse_mode,
	  int score,
	  int best,
	  int flags,
	  int draft,
	  int selectivity ) {
  int i;
  int hit = FALSE;
  int old_draft = 0;
  int old_is_old = FALSE;
  int old_flags = 0;
  int min_priority = 999999;
  int change_encouragment;
  unsigned int base, target_slot, victim_slot;
  unsigned int code1, code2;
  unsigned char curr_gen;
  HashEntry entry;

  assert( abs( score ) != SEARCH_ABORT );

  (void) reverse_mode;
  code1 = hash2 ^ hash_trans2;
  code2 = hash1 ^ hash_trans1;

  base = BUCKET_MASK( code1 & hash_mask );
  victim_slot = base;
  curr_gen = __atomic_load_n( &hash_generation, __ATOMIC_RELAXED ) & 0x03;

  int victim_is_empty = 1;
  int victim_draft = 0;
  int victim_is_old = 0;
  int victim_is_endgame = 0;

  for ( i = 0; i < BUCKET_SIZE; i++ ) {
    unsigned int idx = base + i;
    unsigned int k2 = __atomic_load_n( &hash_table[idx].key2, __ATOMIC_RELAXED );
    int ev = __atomic_load_n( &hash_table[idx].eval, __ATOMIC_RELAXED );
    unsigned int mv = __atomic_load_n( &hash_table[idx].moves, __ATOMIC_RELAXED );
    unsigned int k1_p = __atomic_load_n( &hash_table[idx].key1_selectivity_flags_draft, __ATOMIC_RELAXED );

    if ( ((k2 ^ (unsigned int) ev ^ mv ^ k1_p) == code2) && (((k1_p ^ code1) & KEY1_MASK) == 0) ) {
      hit = TRUE;
      target_slot = idx;
      old_draft = k1_p & DRAFT_VALUE_MASK;
      old_is_old = (((k1_p & DRAFT_GEN_MASK) >> DRAFT_GEN_SHIFT) != curr_gen);
      old_flags = (k1_p >> 8) & 0xFF;
      break;
    }

    int is_empty = ((k1_p & 0x0000FF00) == 0);
    int entry_draft = k1_p & DRAFT_VALUE_MASK;
    int entry_gen = (k1_p & DRAFT_GEN_MASK) >> DRAFT_GEN_SHIFT;
    int age = (curr_gen - entry_gen) & 0x03;
    int entry_flags = (k1_p >> 8) & 0xFF;
    int priority = is_empty ? -999999 : (entry_draft - (age << 5));
    if ( (flags & ENDGAME_SCORE) && !(entry_flags & ENDGAME_SCORE) ) {
      priority -= 512;
    }
    if ( !(flags & ENDGAME_SCORE) && (entry_flags & ENDGAME_SCORE) ) {
      priority += 512;
    }

    if ( priority < min_priority ) {
      min_priority = priority;
      victim_slot = idx;
      victim_is_empty = is_empty;
      victim_draft = entry_draft;
      victim_is_old = (age != 0);
      victim_is_endgame = (entry_flags & ENDGAME_SCORE) != 0;
    }
  }

  if ( flags & EXACT_VALUE )
    change_encouragment = 2;
  else
    change_encouragment = 0;

  static int min_tt_store_draft = -1;
  if ( min_tt_store_draft == -1 ) {
    char *env_draft = getenv( "MIN_TT_STORE_DRAFT" );
    min_tt_store_draft = env_draft ? atoi( env_draft ) : 0;
  }
  if ( draft < min_tt_store_draft )
    return;

  if ( hit ) {
    int old_is_endgame = (old_flags & ENDGAME_SCORE) != 0;
    int new_is_endgame = (flags & ENDGAME_SCORE) != 0;
    if ( old_is_endgame && !new_is_endgame )
      return;
    if ( !old_is_old && (!new_is_endgame || old_is_endgame) &&
         (old_draft > draft + change_encouragment + 2) )
      return;
  }
  else {
    static int rep_offset = -1;
    if ( rep_offset == -1 ) {
      char *env_rep = getenv( "REPLACEMENT_OFFSET" );
      rep_offset = env_rep ? atoi( env_rep ) : 4;
    }
    static int shallow_no_evict_draft = -1;
    if ( shallow_no_evict_draft == -1 ) {
      char *env_sne = getenv( "TT_SHALLOW_NO_EVICT" );
      shallow_no_evict_draft = env_sne ? atoi( env_sne ) : SHALLOW_NO_EVICT_DEFAULT_DRAFT;
    }

    target_slot = victim_slot;

    if ( !victim_is_empty && !victim_is_old && victim_is_endgame &&
         (victim_draft > draft) && (draft <= shallow_no_evict_draft) )
      return;

    if ( min_priority > draft + change_encouragment + rep_offset )
      return;
  }

  entry.key1 = code1;
  entry.key2 = code2;
  entry.eval = score;
  entry.move[0] = best;
  entry.move[1] = 0;
  entry.move[2] = 0;
  entry.move[3] = 0;
  entry.flags = (short) flags;
  entry.draft = (short) draft;
  entry.selectivity = selectivity;
  wide_to_compact( &entry, &hash_table[target_slot] );
}


/*
   ADD_HASH_EXTENDED
   Add information to the hash table using 4-way set-associativity.
   The shallowest search in the 4-entry bucket is replaced.
*/

void
add_hash_extended( int reverse_mode, int score, int *best, int flags,
		   int draft, int selectivity ) {
  int i;
  int hit = FALSE;
  int old_draft = 0;
  int old_is_old = FALSE;
  int old_flags = 0;
  int min_priority = 999999;
  int change_encouragment;
  unsigned int base, target_slot, victim_slot;
  unsigned int code1, code2;
  unsigned char curr_gen;
  HashEntry entry;

  (void) reverse_mode;
  code1 = hash2 ^ hash_trans2;
  code2 = hash1 ^ hash_trans1;

  base = BUCKET_MASK( code1 & hash_mask );
  victim_slot = base;
  curr_gen = __atomic_load_n( &hash_generation, __ATOMIC_RELAXED ) & 0x03;

  int victim_is_empty = 1;
  int victim_draft = 0;
  int victim_is_old = 0;
  int victim_is_endgame = 0;

  for ( i = 0; i < BUCKET_SIZE; i++ ) {
    unsigned int idx = base + i;
    unsigned int k2 = __atomic_load_n( &hash_table[idx].key2, __ATOMIC_RELAXED );
    int ev = __atomic_load_n( &hash_table[idx].eval, __ATOMIC_RELAXED );
    unsigned int mv = __atomic_load_n( &hash_table[idx].moves, __ATOMIC_RELAXED );
    unsigned int k1_p = __atomic_load_n( &hash_table[idx].key1_selectivity_flags_draft, __ATOMIC_RELAXED );

    if ( ((k2 ^ (unsigned int) ev ^ mv ^ k1_p) == code2) && (((k1_p ^ code1) & KEY1_MASK) == 0) ) {
      hit = TRUE;
      target_slot = idx;
      old_draft = k1_p & DRAFT_VALUE_MASK;
      old_is_old = (((k1_p & DRAFT_GEN_MASK) >> DRAFT_GEN_SHIFT) != curr_gen);
      old_flags = (k1_p >> 8) & 0xFF;
      break;
    }

    int is_empty = ((k1_p & 0x0000FF00) == 0);
    int entry_draft = k1_p & DRAFT_VALUE_MASK;
    int entry_gen = (k1_p & DRAFT_GEN_MASK) >> DRAFT_GEN_SHIFT;
    int age = (curr_gen - entry_gen) & 0x03;
    int entry_flags = (k1_p >> 8) & 0xFF;
    int priority = is_empty ? -999999 : (entry_draft - (age << 5));
    if ( (flags & ENDGAME_SCORE) && !(entry_flags & ENDGAME_SCORE) ) {
      priority -= 512;
    }
    if ( !(flags & ENDGAME_SCORE) && (entry_flags & ENDGAME_SCORE) ) {
      priority += 512;
    }

    if ( priority < min_priority ) {
      min_priority = priority;
      victim_slot = idx;
      victim_is_empty = is_empty;
      victim_draft = entry_draft;
      victim_is_old = (age != 0);
      victim_is_endgame = (entry_flags & ENDGAME_SCORE) != 0;
    }
  }

  if ( flags & EXACT_VALUE )
    change_encouragment = 2;
  else
    change_encouragment = 0;

  init_tt_diag_if_needed();
  if ( tt_diag_active ) {
    tt_stats.stores_attempted++;
  }

  static int min_tt_store_draft = -1;
  if ( min_tt_store_draft == -1 ) {
    char *env_draft = getenv( "MIN_TT_STORE_DRAFT" );
    min_tt_store_draft = env_draft ? atoi( env_draft ) : 0;
  }
  if ( draft < min_tt_store_draft ) {
    if ( tt_diag_active ) tt_stats.stores_rejected++;
    return;
  }

  if ( hit ) {
    int old_is_endgame = (old_flags & ENDGAME_SCORE) != 0;
    int new_is_endgame = (flags & ENDGAME_SCORE) != 0;
    if ( old_is_endgame && !new_is_endgame ) {
      if ( tt_diag_active ) tt_stats.stores_rejected++;
      return;
    }
    if ( !old_is_old && (!new_is_endgame || old_is_endgame) &&
         (old_draft > draft + change_encouragment + 2) ) {
      if ( tt_diag_active ) tt_stats.stores_rejected++;
      return;
    }
    if ( tt_diag_active ) tt_stats.stores_updated++;
  }
  else {
    static int rep_offset = -1;
    if ( rep_offset == -1 ) {
      char *env_rep = getenv( "REPLACEMENT_OFFSET" );
      rep_offset = env_rep ? atoi( env_rep ) : 4;
    }
    static int shallow_no_evict_draft = -1;
    if ( shallow_no_evict_draft == -1 ) {
      char *env_sne = getenv( "TT_SHALLOW_NO_EVICT" );
      shallow_no_evict_draft = env_sne ? atoi( env_sne ) : SHALLOW_NO_EVICT_DEFAULT_DRAFT;
    }

    target_slot = victim_slot;

    if ( !victim_is_empty && !victim_is_old && victim_is_endgame &&
         (victim_draft > draft) && (draft <= shallow_no_evict_draft) ) {
      if ( tt_diag_active ) tt_stats.stores_rejected++;
      return;
    }

    if ( min_priority > draft + change_encouragment + rep_offset ) {
      if ( tt_diag_active ) tt_stats.stores_rejected++;
      return;
    }
    if ( tt_diag_active ) {
      if ( victim_is_empty ) {
        tt_stats.stores_empty_slot++;
      } else {
        tt_stats.stores_evicted++;
        if ( victim_draft < 64 ) tt_stats.evicted_draft_counts[victim_draft]++;
        if ( draft < 64 ) tt_stats.evicted_by_draft_counts[draft]++;
      }
    }
  }

  entry.key1 = code1;
  entry.key2 = code2;
  entry.eval = score;
  for ( i = 0; i < 4; i++ )
    entry.move[i] = best[i];
  entry.flags = (short) flags;
  entry.draft = (short) draft;
  entry.selectivity = selectivity;
  wide_to_compact( &entry, &hash_table[target_slot] );
}


/*
   FIND_HASH
   Search the hash table for the current position using 4-way set-associativity.
   All 4 bucket slots in the 64-byte cache line are probed.
*/   

void REGPARM(2)
find_hash( HashEntry *entry, int reverse_mode ) {
  int i;
  unsigned int base;
  unsigned int code1, code2;

  (void) reverse_mode;
  code1 = hash2 ^ hash_trans2;
  code2 = hash1 ^ hash_trans1;

  base = BUCKET_MASK( code1 & hash_mask );

  init_tt_diag_if_needed();
  if ( tt_diag_active ) {
    tt_stats.probes++;
  }

  for ( i = 0; i < BUCKET_SIZE; i++ ) {
    unsigned int idx = base + i;
    unsigned int k2_raw = __atomic_load_n( &hash_table[idx].key2, __ATOMIC_ACQUIRE );
    int ev = __atomic_load_n( &hash_table[idx].eval, __ATOMIC_RELAXED );
    unsigned int mv = __atomic_load_n( &hash_table[idx].moves, __ATOMIC_RELAXED );
    unsigned int k1_p = __atomic_load_n( &hash_table[idx].key1_selectivity_flags_draft, __ATOMIC_RELAXED );

    if ( (k2_raw ^ (unsigned int) ev ^ mv ^ k1_p) == code2 ) {
      if ( ((k1_p ^ code1) & KEY1_MASK) == 0 ) {
        if ( tt_diag_active ) tt_stats.probe_hits++;
        compact_to_wide( entry, code2, ev, mv, k1_p );
        return;
      }
    }
  }

  if ( tt_diag_active ) {
    tt_stats.probe_misses++;
    int empty_slots = 0;
    for ( i = 0; i < BUCKET_SIZE; i++ ) {
      unsigned int idx = base + i;
      unsigned int k1_p = __atomic_load_n( &hash_table[idx].key1_selectivity_flags_draft, __ATOMIC_RELAXED );
      if ( ((k1_p & 0x0000FF00) == 0) ) empty_slots++;
    }
    if ( empty_slots == 0 ) tt_stats.probe_miss_bucket_full++;
    else tt_stats.probe_miss_empty_slots++;
  }

  entry->draft = NO_HASH_MOVE;
  entry->flags = UPPER_BOUND;
  entry->eval = INFINITE_EVAL;
  entry->move[0] = 44;
  entry->move[1] = 0;
  entry->move[2] = 0;
  entry->move[3] = 0;
}


/*
   FIND_SHALLOW_HASH
   Direct-mapped probe into the shallow hash table for low-depth endgame nodes.
*/

void REGPARM(1)
find_shallow_hash( HashEntry *entry, const ThreadState *ts ) {
  int index;
  unsigned int code1, code2;
  unsigned int k2_raw, k1_p, mv;
  int ev;

  if ( shallow_hash_table == NULL ) {
    entry->draft = NO_HASH_MOVE;
    entry->flags = UPPER_BOUND;
    entry->eval = INFINITE_EVAL;
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

  if ( (k2_raw ^ (unsigned int) ev ^ mv ^ k1_p) == code2 ) {
    if ( ((k1_p ^ code1) & KEY1_MASK) == 0 ) {
      compact_to_wide( entry, code2, ev, mv, k1_p );
      return;
    }
  }

  entry->draft = NO_HASH_MOVE;
  entry->flags = UPPER_BOUND;
  entry->eval = INFINITE_EVAL;
  entry->move[0] = 0;
  entry->move[1] = 0;
  entry->move[2] = 0;
  entry->move[3] = 0;
}


/*
   ADD_SHALLOW_HASH
   Direct-mapped store into the shallow hash table.
*/

void
add_shallow_hash( int score,
		  int best,
		  int flags,
		  int draft,
		  const ThreadState *ts ) {
  unsigned int index;
  unsigned int code1, code2;
  HashEntry entry;

  if ( shallow_hash_table == NULL )
    return;

  code1 = ts->ts_hash2 ^ hash_trans2;
  code2 = ts->ts_hash1 ^ hash_trans1;

  index = code1 & shallow_hash_mask;

  entry.key1 = code1;
  entry.key2 = code2;
  entry.eval = score;
  entry.move[0] = best;
  entry.move[1] = 0;
  entry.move[2] = 0;
  entry.move[3] = 0;
  entry.flags = (short) flags;
  entry.draft = (short) draft;
  entry.selectivity = 0;
  wide_to_compact( &entry, &shallow_hash_table[index] );
}


/* ========================================================================= */
/* TT Diagnostic Dumpers & Reporters                                         */
/* ========================================================================= */

void reset_tt_stats( void ) {
  memset( &tt_stats, 0, sizeof( tt_stats ) );
}

void print_tt_stats( const char *label ) {
  init_tt_diag_if_needed();
  if ( !tt_diag_active ) return;
  fprintf( stderr, "TT_DIAG [%s] PROBES: total=%llu hits=%llu (%.2f%%) misses=%llu (full_buckets=%llu, empty_slots=%llu)\n",
           label, tt_stats.probes, tt_stats.probe_hits,
           tt_stats.probes ? (100.0 * tt_stats.probe_hits / tt_stats.probes) : 0.0,
           tt_stats.probe_misses, tt_stats.probe_miss_bucket_full, tt_stats.probe_miss_empty_slots );
  fprintf( stderr, "TT_DIAG [%s] STORES: attempted=%llu updated=%llu empty_slots=%llu evicted=%llu rejected=%llu\n",
           label, tt_stats.stores_attempted, tt_stats.stores_updated,
           tt_stats.stores_empty_slot, tt_stats.stores_evicted, tt_stats.stores_rejected );
  fprintf( stderr, "TT_DIAG [%s] EVICTIONS BY DRAFT EVICTED: ", label );
  for ( int d = 9; d <= 28; d++ ) {
    if ( tt_stats.evicted_draft_counts[d] > 0 ) {
      fprintf( stderr, "d%d=%llu ", d, tt_stats.evicted_draft_counts[d] );
    }
  }
  fprintf( stderr, "\n" );
  fprintf( stderr, "TT_DIAG [%s] EVICTIONS BY DRAFT WRITING: ", label );
  for ( int d = 9; d <= 28; d++ ) {
    if ( tt_stats.evicted_by_draft_counts[d] > 0 ) {
      fprintf( stderr, "d%d=%llu ", d, tt_stats.evicted_by_draft_counts[d] );
    }
  }
  fprintf( stderr, "\n" );
}

void dump_tt_distribution( const char *label ) {
  init_tt_diag_if_needed();
  if ( !tt_diag_active || hash_table == NULL ) return;

  unsigned long long occupied = 0;
  unsigned long long draft_hist[64] = {0};
  unsigned long long sel_hist[16] = {0};
  unsigned long long endgame_scores = 0, midgame_scores = 0;
  unsigned long long exact_vals = 0, lower_bounds = 0, upper_bounds = 0;
  unsigned char curr_gen = __atomic_load_n( &hash_generation, __ATOMIC_RELAXED ) & 0x03;
  unsigned long long current_gen_count = 0, old_gen_count = 0;

  for ( int i = 0; i < hash_size; i++ ) {
    unsigned int k1_p = __atomic_load_n( &hash_table[i].key1_selectivity_flags_draft, __ATOMIC_RELAXED );
    int flags = (k1_p >> 8) & 0xFF;
    if ( flags == 0 ) continue;

    occupied++;
    int draft = k1_p & DRAFT_VALUE_MASK;
    if ( draft < 64 ) draft_hist[draft]++;
    int sel = (k1_p & 0x00ffffff) >> 16;
    if ( sel < 16 ) sel_hist[sel]++;
    if ( flags & ENDGAME_SCORE ) endgame_scores++;
    if ( flags & MIDGAME_SCORE ) midgame_scores++;
    if ( flags & EXACT_VALUE ) exact_vals++;
    if ( flags & LOWER_BOUND ) lower_bounds++;
    if ( flags & UPPER_BOUND ) upper_bounds++;
    int entry_gen = (k1_p & DRAFT_GEN_MASK) >> DRAFT_GEN_SHIFT;
    if ( entry_gen == curr_gen ) current_gen_count++; else old_gen_count++;
  }

  fprintf( stderr, "TT_DIAG [%s] OCCUPANCY: %llu / %d (%.2f%%) [current_gen=%llu, old_gen=%llu]\n",
           label, occupied, hash_size, 100.0 * occupied / hash_size, current_gen_count, old_gen_count );
  fprintf( stderr, "TT_DIAG [%s] SELECTIVITY: ", label );
  for ( int s = 0; s <= 9; s++ ) {
    if ( sel_hist[s] > 0 ) fprintf( stderr, "sel%d=%llu ", s, sel_hist[s] );
  }
  fprintf( stderr, "\n" );
  fprintf( stderr, "TT_DIAG [%s] DRAFTS: ", label );
  for ( int d = 9; d <= 28; d++ ) {
    if ( draft_hist[d] > 0 ) fprintf( stderr, "d%d=%llu ", d, draft_hist[d] );
  }
  fprintf( stderr, "\n" );
  fprintf( stderr, "TT_DIAG [%s] FLAGS: exact=%llu lower=%llu upper=%llu (endgame=%llu, midgame=%llu)\n",
           label, exact_vals, lower_bounds, upper_bounds, endgame_scores, midgame_scores );
}

