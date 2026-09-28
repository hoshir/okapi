/*
   File:          threads.c

   Created:       August 9, 2026

   Contents:      A small fork-join worker pool for the parallel search.

                  The pool is deliberately minimal: the caller hands in
                  a job function and a job count, and every thread --
                  the caller included -- claims jobs off a shared
                  counter until they run out.  That keeps the work
                  balanced without a queue, and it means a search can
                  fall back to running everything on the calling thread
                  simply by asking for one thread.

                  Several batches can be in flight at once.  A thread
                  running a job may start a batch of its own, and a
                  thread with nothing left to do in its own batch helps
                  with whichever other batch still has work; without
                  that, the threads that finished early would sit idle
                  until the longest job of the batch came back, which is
                  where most of the parallel search's efficiency went.
*/



#include <pthread.h>
#include <stdlib.h>

#include "constant.h"
#include "search.h"
#include "threads.h"



#define MAX_THREADS               64

/* Batches in flight at once.  One per thread would be enough for the
   nesting the search actually does; the rest is slack. */
#define MAX_BATCHES               (2 * MAX_THREADS)



typedef struct {
  void (*job)( int index, void *context );
  void *context;
  int job_count;
  int next_job;        /* first job not yet claimed */
  int outstanding;     /* jobs neither finished nor abandoned */
  int active;
  int depth;           /* nesting level of the thread that started it */
  unsigned int seq;    /* creation order, to break ties between equals */
} Batch;


typedef struct {
  pthread_mutex_t lock;
  pthread_cond_t change;   /* a batch appeared, or a job finished */
  int shutting_down;
} PoolState;



/* Local variables */

static PoolState pool;
static Batch batch[MAX_BATCHES];
static unsigned int batch_seq;
static pthread_t worker[MAX_THREADS];
static int thread_count = 1;
static int worker_count;   /* threads in worker[], i.e. thread_count - 1 */
static int pool_created;
static _Thread_local int is_worker;

/* Split points currently published by each thread */
static _Atomic(SplitPoint *) thread_split[MAX_THREADS];

/* Split points allocated statically per thread and split nesting */
static SplitPoint thread_splits[MAX_THREADS][MAX_SPLIT_NESTING + 1] __attribute__((aligned(64)));

SplitPoint *
ybwc_get_split_point( int my_id, int nesting ) {
  if ( my_id < 0 || my_id >= MAX_THREADS )
    my_id = 0;
  if ( nesting < 0 )
    nesting = 0;
  if ( nesting > MAX_SPLIT_NESTING )
    nesting = MAX_SPLIT_NESTING;
  return &thread_splits[my_id][nesting];
}

/* Mutex and cond for waiting master on each thread */
static pthread_mutex_t thread_wait_mutex[MAX_THREADS];
static pthread_cond_t thread_wait_cond[MAX_THREADS];

/* Threads parked with no work to do. */
static _Atomic int idle_count;

_Thread_local int thread_id;
_Thread_local SplitPoint *current_split_point;
volatile int active_splits;

/* How deep in the batch nesting this thread currently is: zero outside
   any job, and one more than the depth of the batch whose job it is
   running.  A batch inherits the level of the thread that started it,
   and helpers prefer the deepest work they can find, so that an inner
   batch is cleared before the outer one it is holding up. */
static _Thread_local int nesting;

/* Where the workers leave the nodes they searched.  Each thread counts
   into its own NODES, so a worker's share used to be dropped on the
   floor when its batch ended and the reported totals were the calling
   thread's alone -- which made the node count at one thread and the
   node count at eight incomparable, and hid how much extra work the
   parallel search was doing.  Written under the pool lock. */
static CounterType pooled_nodes;



/*
  YBWC_TRY_STEAL_AND_SEARCH
  Scans active split points published by other threads, selects the deepest
  available split point, and steals a candidate move locklessly.
*/

static int
ybwc_try_steal_and_search( int my_id ) {
  int i;
  int num_t = thread_count;
  SplitPoint *best_sp = NULL;
  unsigned int target_seq = 0;

  for ( i = 0; i < num_t; i++ ) {
    if ( i == my_id )
      continue;
    SplitPoint *sp = atomic_load_explicit( &thread_split[i], memory_order_acquire );
    if ( sp == NULL )
      continue;
    if ( atomic_load_explicit( &sp->cutoff_occurred, memory_order_relaxed ) )
      continue;
    if ( atomic_load_explicit( &sp->next_move_idx, memory_order_relaxed ) >= sp->move_count )
      continue;
    if ( best_sp == NULL || sp->level > best_sp->level ) {
      best_sp = sp;
      target_seq = atomic_load_explicit( &sp->sp_seq, memory_order_relaxed );
    }
  }

  if ( best_sp != NULL ) {
    int master_id = best_sp->master_thread_id;

    // 1. Safely acquire reference count using CAS loop
    int old_workers = atomic_load_explicit( &best_sp->active_workers, memory_order_acquire );
    while ( old_workers > 0 ) {
      if ( atomic_compare_exchange_weak_explicit( &best_sp->active_workers, &old_workers, old_workers + 1,
                                                  memory_order_acquire, memory_order_relaxed ) ) {
        break;
      }
    }
    if ( old_workers <= 0 ) {
      return FALSE;
    }

    // 2. Double-check that best_sp is still published, same generation, and not in cutoff
    if ( atomic_load_explicit( &thread_split[master_id], memory_order_seq_cst ) != best_sp ||
         atomic_load_explicit( &best_sp->sp_seq, memory_order_relaxed ) != target_seq ||
         atomic_load_explicit( &best_sp->cutoff_occurred, memory_order_relaxed ) ) {
      int remaining = atomic_fetch_sub_explicit( &best_sp->active_workers, 1, memory_order_release ) - 1;
      if ( remaining == 0 ) {
        pthread_mutex_lock( &thread_wait_mutex[master_id] );
        pthread_cond_signal( &thread_wait_cond[master_id] );
        pthread_mutex_unlock( &thread_wait_mutex[master_id] );
      }
      return FALSE;
    }

    // 3. Claim move locklessly
    int idx = atomic_fetch_add_explicit( &best_sp->next_move_idx, 1, memory_order_relaxed );
    if ( idx < best_sp->move_count && !atomic_load_explicit( &best_sp->cutoff_occurred, memory_order_relaxed ) ) {
      SplitPoint *saved_sp = current_split_point;
      current_split_point = best_sp;

      best_sp->search_fn( best_sp, idx );

      current_split_point = saved_sp;

      // Transfer nodes to split point
      adjust_counter( &nodes );
      uint64_t my_n = (uint64_t) nodes.hi * 100000000ULL + nodes.lo;
      if ( my_n > 0 ) {
        atomic_fetch_add_explicit( &best_sp->pooled_nodes, my_n, memory_order_relaxed );
        reset_counter( &nodes );
      }
    }

    // 4. Release reference count
    int remaining = atomic_fetch_sub_explicit( &best_sp->active_workers, 1, memory_order_release ) - 1;
    if ( remaining == 0 ) {
      pthread_mutex_lock( &thread_wait_mutex[master_id] );
      pthread_cond_signal( &thread_wait_cond[master_id] );
      pthread_mutex_unlock( &thread_wait_mutex[master_id] );
    }

    return (idx < best_sp->move_count);
  }

  return FALSE;
}



/*
  YBWC_SPLIT
  Publishes a pre-allocated SplitPoint, wakes up idle threads, searches
  moves locklessly, and waits until all active helpers complete.
*/

void
ybwc_split( SplitPoint *sp ) {
  int my_id = thread_id;
  sp->master_thread_id = my_id;
  atomic_init( &sp->next_move_idx, 0 );
  atomic_init( &sp->active_workers, 1 );
  atomic_init( &sp->cutoff_occurred, false );
  atomic_init( &sp->pooled_nodes, 0 );
  atomic_fetch_add_explicit( &sp->sp_seq, 1, memory_order_relaxed );

  // 1. Publish split point for helpers to steal from
  atomic_store_explicit( &thread_split[my_id], sp, memory_order_seq_cst );

  // 2. Wake up idle worker threads
  pthread_cond_broadcast( &pool.change );

  // 3. Master thread steals and searches moves from its own split point
  SplitPoint *saved_sp = current_split_point;
  current_split_point = sp;

  while ( !atomic_load_explicit( &sp->cutoff_occurred, memory_order_relaxed ) ) {
    int idx = atomic_fetch_add_explicit( &sp->next_move_idx, 1, memory_order_relaxed );
    if ( idx >= sp->move_count )
      break;
    sp->search_fn( sp, idx );
  }

  // 4. Unpublish this split point so no new helpers can discover it
  atomic_store_explicit( &thread_split[my_id], NULL, memory_order_seq_cst );

  // 5. Release master's own reference and wait for active helpers to finish
  int remaining = atomic_fetch_sub_explicit( &sp->active_workers, 1, memory_order_release ) - 1;
  if ( remaining > 0 ) {
    pthread_mutex_lock( &thread_wait_mutex[my_id] );
    while ( atomic_load_explicit( &sp->active_workers, memory_order_acquire ) > 0 ) {
      pthread_cond_wait( &thread_wait_cond[my_id], &thread_wait_mutex[my_id] );
    }
    pthread_mutex_unlock( &thread_wait_mutex[my_id] );
  }

  // 6. Restore parent split point if nested
  if ( sp->parent != NULL ) {
    atomic_store_explicit( &thread_split[my_id], sp->parent, memory_order_release );
  }
  current_split_point = saved_sp;

  // 7. Fold nodes into master's counter
  uint64_t w_nodes = atomic_load_explicit( &sp->pooled_nodes, memory_order_relaxed );
  if ( w_nodes > 0 ) {
    CounterType c;
    c.hi = (unsigned int)(w_nodes / 100000000ULL);
    c.lo = (unsigned int)(w_nodes % 100000000ULL);
    add_counter( &nodes, &c );
    adjust_counter( &nodes );
  }
}



/*
  CLAIM_ONE
  Run a single job off the deepest batch that still has one, or off ONLY
  if that is not NULL.  Called with the lock held; releases it while the
  job runs.  Returns FALSE when there was nothing to claim, in which
  case the lock was never dropped and the caller may park on
  POOL.CHANGE without racing.

  ONLY is how a thread suspended in the middle of its own search takes
  part without wrecking it.  The search keeps a good deal of state
  indexed by ply -- move lists, hash keys, flip masks -- which a job
  starting from a different node overwrites from its own ply downwards.
  A job of the batch this thread is waiting on starts at exactly the
  node the thread is suspended at, so it writes only below the frames
  that are live; any other job may not.  A parked worker is inside no
  search at all and so has nothing to protect: those are the threads
  that carry a nested batch.
*/

static int
claim_one( Batch *only ) {
  Batch *pick = NULL;
  int i, index, saved_nesting;

  if ( only != NULL ) {
    if ( only->next_job >= only->job_count )
      return FALSE;
    pick = only;
  }
  else {
    for ( i = 0; i < MAX_BATCHES; i++ ) {
      Batch *b = &batch[i];
      if ( !b->active || (b->next_job >= b->job_count) )
	continue;
      if ( (pick == NULL) || (b->depth > pick->depth) ||
	   ((b->depth == pick->depth) && (b->seq > pick->seq)) )
	pick = b;
    }
    if ( pick == NULL )
      return FALSE;
  }

  index = pick->next_job++;
  saved_nesting = nesting;
  nesting = pick->depth + 1;
  pthread_mutex_unlock( &pool.lock );
  pick->job( index, pick->context );
  pthread_mutex_lock( &pool.lock );
  nesting = saved_nesting;

  /* A worker owns no total of its own, so hand the nodes over here
     rather than at the end of the batch: with batches overlapping there
     is no single point where the worker is between them. */
  if ( is_worker ) {
    add_counter( &pooled_nodes, &nodes );
    reset_counter( &nodes );
  }

  if ( --pick->outstanding == 0 )
    pthread_cond_broadcast( &pool.change );

  return TRUE;
}


/*
  WORKER_MAIN
  Take jobs from YBWC split points locklessly, or from Batch if available,
  and park when none do.
*/

static void *
worker_main( void *arg ) {
  int my_id = (int)(intptr_t) arg;
  thread_id = my_id;
  is_worker = TRUE;
  init_search_thread();

  while ( !pool.shutting_down ) {
    // 1. Check for YBWC split work
    if ( ybwc_try_steal_and_search( my_id ) )
      continue;

    // 2. Check for midgame Batch work (threads_run)
    pthread_mutex_lock( &pool.lock );
    if ( claim_one( NULL ) ) {
      pthread_mutex_unlock( &pool.lock );
      continue;
    }

    // 3. Park when no work exists
    if ( !pool.shutting_down ) {
      atomic_fetch_add_explicit( &idle_count, 1, memory_order_relaxed );
      pthread_cond_wait( &pool.change, &pool.lock );
      atomic_fetch_sub_explicit( &idle_count, 1, memory_order_relaxed );
    }
    pthread_mutex_unlock( &pool.lock );
  }

  return NULL;
}



void
threads_init( int count ) {
  int i;

  thread_id = 0;

  if ( count < 1 )
    count = 1;
  if ( count > MAX_THREADS )
    count = MAX_THREADS;
  if ( pool_created && (count == thread_count) )
    return;

  threads_shutdown();

  thread_count = count;
  worker_count = count - 1;
  if ( worker_count == 0 ) {
    pool_created = TRUE;
    return;
  }

  pthread_mutex_init( &pool.lock, NULL );
  pthread_cond_init( &pool.change, NULL );
  pool.shutting_down = FALSE;
  atomic_init( &idle_count, 0 );
  batch_seq = 0;
  for ( i = 0; i < MAX_BATCHES; i++ )
    batch[i].active = FALSE;

  for ( i = 0; i < MAX_THREADS; i++ ) {
    int j;
    atomic_init( &thread_split[i], NULL );
    pthread_mutex_init( &thread_wait_mutex[i], NULL );
    pthread_cond_init( &thread_wait_cond[i], NULL );
    for ( j = 0; j <= MAX_SPLIT_NESTING; j++ ) {
      thread_splits[i][j].master_thread_id = i;
      atomic_init( &thread_splits[i][j].active_workers, 0 );
      atomic_init( &thread_splits[i][j].sp_seq, 0 );
      atomic_init( &thread_splits[i][j].cutoff_occurred, false );
      atomic_init( &thread_splits[i][j].next_move_idx, 0 );
      atomic_init( &thread_splits[i][j].pooled_nodes, 0 );
    }
  }

  pthread_attr_t attr;
  pthread_attr_init( &attr );
  pthread_attr_setstacksize( &attr, 8 * 1024 * 1024 );

  for ( i = 0; i < worker_count; i++ )
    if ( pthread_create( &worker[i], &attr, worker_main, (void *)(intptr_t)(i + 1) ) != 0 ) {
      /* Carry on with however many threads we did get */
      worker_count = i;
      thread_count = i + 1;
      break;
    }
  pthread_attr_destroy( &attr );
  pool_created = TRUE;
}


void
threads_shutdown( void ) {
  int i;

  if ( !pool_created )
    return;
  if ( worker_count > 0 ) {
    pthread_mutex_lock( &pool.lock );
    pool.shutting_down = TRUE;
    pthread_cond_broadcast( &pool.change );
    pthread_mutex_unlock( &pool.lock );

    for ( i = 0; i < worker_count; i++ )
      pthread_join( worker[i], NULL );

    pthread_cond_destroy( &pool.change );
    pthread_mutex_destroy( &pool.lock );

    for ( i = 0; i < MAX_THREADS; i++ ) {
      pthread_cond_destroy( &thread_wait_cond[i] );
      pthread_mutex_destroy( &thread_wait_mutex[i] );
    }
  }
  worker_count = 0;
  thread_count = 1;
  pool_created = FALSE;
}


int
threads_count( void ) {
  return thread_count;
}


int
threads_idle_count( void ) {
  return atomic_load_explicit( &idle_count, memory_order_relaxed );
}


void
threads_run( void (*job)( int index, void *context ), void *context,
	     int job_count ) {
  Batch *mine = NULL;
  int i;

  if ( job_count <= 0 )
    return;

  if ( worker_count == 0 ) {  /* Single-threaded: just run them here */
    for ( i = 0; i < job_count; i++ )
      job( i, context );
    return;
  }

  pthread_mutex_lock( &pool.lock );

  for ( i = 0; i < MAX_BATCHES; i++ )
    if ( !batch[i].active ) {
      mine = &batch[i];
      break;
    }
  if ( mine == NULL ) {  /* Out of slots: run the jobs here and now */
    pthread_mutex_unlock( &pool.lock );
    for ( i = 0; i < job_count; i++ )
      job( i, context );
    return;
  }

  mine->job = job;
  mine->context = context;
  mine->job_count = job_count;
  mine->next_job = 0;
  mine->outstanding = job_count;
  mine->depth = nesting;
  mine->seq = batch_seq++;
  mine->active = TRUE;
  pthread_cond_broadcast( &pool.change );

  /* The caller helps with its own batch rather than idling -- see
     claim_one for why it may not help with anyone else's -- which also
     means a batch always finishes even if no other thread ever looks at
     it, so nesting cannot deadlock.  It is up to the caller to put its
     own search state back afterwards, since a job overwrites the state
     it was using. */
  while ( mine->outstanding > 0 )
    if ( !claim_one( mine ) ) {
      atomic_fetch_add_explicit( &idle_count, 1, memory_order_relaxed );
      pthread_cond_wait( &pool.change, &pool.lock );
      atomic_fetch_sub_explicit( &idle_count, 1, memory_order_relaxed );
    }

  mine->active = FALSE;

  /* Fold what the workers did into this thread's count, so that every
     place that already reports NODES reports the whole batch. */
  add_counter( &nodes, &pooled_nodes );
  reset_counter( &pooled_nodes );

  pthread_mutex_unlock( &pool.lock );
}
