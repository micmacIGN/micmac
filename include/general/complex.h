#ifndef _ELISE_INCLUDE_GENERAL_COMPLEX_H_
#define _ELISE_INCLUDE_GENERAL_COMPLEX_H_

#include "StdAfx.h"

/*===================================*/
/*         polar-def                 */
/*===================================*/

class Polar_Def_Opun : public Simple_OP_UN<REAL>
{

public :
	Polar_Def_Opun(REAL teta_def) : _teta_def (teta_def) {}

	static Fonc_Num polar(Fonc_Num f,REAL teta0);

private  :
	virtual void calc_buf
		(
		REAL **,
		REAL**,
		INT,
		const Arg_Comp_Simple_OP_UN  &
		);
	REAL _teta_def;

};

#endif //_ELISE_INCLUDE_GENERAL_COMPLEX_H_