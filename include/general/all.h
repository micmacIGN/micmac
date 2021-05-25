/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/





#ifndef _ELISE_GENERAL_ALL_H
#define _ELISE_GENERAL_ALL_H
extern bool BugDG;
/*
#ifdef _WIN32
	#define ELISE_INSERT_CODE_GEN 0
#else
	*/
#define ELISE_INSERT_CODE_GEN 1
/*
#endif
*/

#include "general/sys_dep.h"


template <class Type> class ElFilo;
template <class Type> class ElFifo;

#if (USE_NOYAU)

	#include <strstream>


/*** VERSION SANS LE NOYAU (mais perd les messages d'erreur...) ****/

	#include <cstdio>
	#include <memory>
	#include <malloc.h>
	#include <iostream>

	// Noyeau Mini (pour concerver les Sortie Message dans la fenetre du noyau : ncout())
	#if (USE_NOYAU_MINI)
		#include "noyau_base/handler_sortiemessage.h"
		#include "noyau_base/nostream.h"
	#else
		// Pas de sortie message dans la fenetre du noyau
		#define ncout() cerr
	#endif

	// #include <stdlib.h>
	#include <cstdlib>
	#include <cstdio>
//	#include <iostream.h>
	#include <iostream>
	#include <cmath>
//	#include <ostream.h>
    #include <ostream>
	#include <map>
	#include <set>
	#include <vector>
	#include <string>
//	#include <fstream.h>
	#include <fstream>
// #include <iomanip.h>
//#include <iomanip>
	#include <limits>

	#include <sys/stat.h>
 
#else
#include <cmath>
#include <cstdlib>
#include <cstdio>
#if (ELISE_unix)
#include <limits>
#else
#include <limits>
#endif
#include <string>
#include <cfloat>
#include <sys/types.h>
#include <sys/stat.h>

#include <new>

#if (GPP3etPlus)
#if (MACHINE_BLERIOT)
#include <ostream>
#include <strstream>
#else
#include <sstream>  // THOM
#include <ostream> //  THOM
#endif
#else
#include <strstream>
#include <ostream>
#endif

#include <iostream>
#include <fstream>

using namespace std;
#endif


#if (USE_NOYAU)
#else
	inline std::ostream & ncout () {return std::cout;}
#endif


#include <map>
#include <set>
#include <vector>
#include <deque>
#include <list>
#include <string>

#if (Compiler_Gpp2_7_2 || MACHINE_BLERIOT)
template <class Type> Type & AT(std::vector<Type> & V,const INT & K)
{
   assert((K>=0)&&(K<INT(V.size())));
   return V[K];
}
template <class Type> const Type & AT(const std::vector<Type> & V,const INT & K)
{
   assert((K>=0)&&(K<INT(V.size())));
   return V[K];
}
#else
template <class Type> Type & AT(std::vector<Type> & V,const INT & K)
{
    return V.at(K);
}
template <class Type> const Type & AT(const std::vector<Type> & V,const INT & K) 
{
    return V.at(K);
}
#endif

   class cOrientationConique;

  
using namespace std;
#include "general/opt_debug.h"
#include "general/allocation.h"
#include "general/util.h"
#include "general/ptxd.h"
#include "general/tabulation.h"
#include "general/smart_pointeur.h"
#include "general/garb_coll_pub.h"
#include "general/abstract_types.h"

#include "general/bitm.h"
#include "general/users_op_buf.h"
#include "general/colour.h"
#include "general/graphics.h"
#include "general/window.h"
#include "general/file_im.h"
#include "general/tiff_file_im.h"
#include "general/operator.h"
#include "general/plot1d.h"
#include "general/morpho.h"
#include "general/orilib.h"

#include "general/vecto.h"
#include "general/geom_vecteur.h"
#include "general/photogram.h"
#include "general/mullgesuhlig.h"
#include "general/optim.h"
#include "general/error.h"
#include "general/arg_main.h"
#include "general/cMMSpecArg.h"
#include "general/compr_im.h"
#include "general/correl.h"

#include "general/ijpeg.h"
#include "general/cube_flux.h"
#include "general/phgr_formel.h"
#include "general/phgr_dist_unif.h"
#include "general/exemple_phgr_formel.h"
#include "general/exemple_basculement.h"
#include "general/simul_phgr.h"
#include "general/phgr_orel.h"
#include "api/vecto.h"
#include "api/el_regex.h"


#include "general/phgr_san.h"

#include "general/hassan_arrangt.h"
#include "general/etal.h"

class cElOstream
{
    public :
       cElOstream(std::ostream  *);
       std::ostream * mOStr;
};
 
template <class Type> 
cElOstream & operator << (cElOstream & anOs, const Type & aVal)
{
   if (anOs.mOStr) *(anOs.mOStr) << aVal;
   return anOs;
}


void ElisePenseBete();


#endif /* ! _ELISE_GENERAL_ALL_H */

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
