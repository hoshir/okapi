/*
   File:          end_aspiration.h

   Contents:      Root aspiration window centering for the endgame search.
*/

#ifndef END_ASPIRATION_H
#define END_ASPIRATION_H

#ifdef __cplusplus
extern "C" {
#endif

/* Even center of the root null window for a static evaluation given in
   1/128 disc units, clamped to +-LIMIT (LIMIT must be even). */
int end_aspiration_center( int eval, int limit );

/* Computes the next [alpha, beta] window for an exact re-search after
   ROOT_EVAL failed low (<= ALPHA) or failed high (>= BETA). */
void end_aspiration_next_bounds( int root_eval, int *alpha, int *beta, int *delta );

/* Advances the selective null-window center by one step in STEP_DIR (-2 or +2),
   snapping directly to ROOT_EVAL when a stability/TT bound returned a score
   further outside the null window. */
int end_aspiration_step_center( int center, int root_eval, int step_dir );

#ifdef __cplusplus
}
#endif

#endif  /* END_ASPIRATION_H */
