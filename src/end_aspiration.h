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

#ifdef __cplusplus
}
#endif

#endif  /* END_ASPIRATION_H */
