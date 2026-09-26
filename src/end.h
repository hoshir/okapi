/*
   File:          end.h

   Created:       June 25, 1997

   Author:        Gunnar Andersson (gunnar@radagast.se)

   Contents:      The interface to the endgame solver.
*/



#ifndef END_H
#define END_H



#include "end_init.h"
#include "search.h"
#include "tlstate.h"



#define END_MOVE_LIST_HEAD        0
#define END_MOVE_LIST_TAIL        99



int
end_game( int side_to_move,
	  int wld,
	  int force_echo,
	  int allow_book,
	  int komi,
	  EvaluationType *eval_info );



#endif  /* END_H */
