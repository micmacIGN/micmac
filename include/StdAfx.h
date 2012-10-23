#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <list>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <set>
#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
	#include <unistd.h>
#endif

extern bool BugDG;

#define ELISE_INSERT_CODE_GEN 1

#ifdef _WIN32
	#define USE_NOYAU 1
	#define ELISE_unix 0
	#define ELISE_windows 1
        #define ELISE_MacOs 0
        #define ELISE_Cygwin 0
#elif __APPLE__
	#define USE_NOYAU 0
	#define ELISE_unix 0
	#define ELISE_MacOs 1
	#define ELISE_windows 0
        #define ELISE_Cygwin 0
#elif __CYGWIN__
       #define USE_NOYAU 0
       #define ELISE_unix 0
       #define ELISE_MacOs 0
       #define ELISE_windows 0
       #define ELISE_Cygwin 1
#else
       #define USE_NOYAU 0
       #define ELISE_unix 1
       #define ELISE_MacOs 0
       #define ELISE_windows 0
       #define ELISE_Cygwin 0
#endif

#include "general/MM_InstalDir.h"
#include "general/sys_dep.h"
#include "general/opt_debug.h"
#include "general/util.h"
#include "general/ptxd.h"

#include "api/el_regex.h"

#include "general/allocation.h"
#include "general/garb_coll_pub.h"
#include "general/abstract_types.h"
#include "general/bitm.h"
#include "general/tabulation.h"

#include "private/garb_coll_tpl.h"
#include "private/flux_pts.h"
#include "private/flux_pts_tpl.h"
#include "private/bitm_def.h"
#include "private/bitm_tpl.h"

#include "im_tpl/flux.h"

#include "private/bitm_bits.h"

#include "ext_stl/fixed.h"

#include "im_tpl/image.h"

#include "XML_GEN/all.h"

#include "api/vecto.h"

// TODO : these global functions and classes should be placed somewhere else
// see all.cpp for definitions

extern std::string NoInit;
extern Pt2dr	   aNoPt;
Im2DGen AllocImGen(Pt2di aSz,const std::string & aName);

// ---------

#if(ELISE_unix)
	#include <cstring>
#endif

#ifdef MATLAB_MEX_FILE
	#include "mex.h"
	#include "matrix.h"
#endif


#include "general/hassan_arrangt.h"
#include "general/optim.h"
#include "general/abstract_types.h"
#include "graphes/graphe.h"
#include "general/phgr_formel.h"

#include "api/vecto.h"
#include "api/cox_roy.h"
#include "api/el_regex.h"

#include "graphes/algo_pcc.h"

#include "im_tpl/correl_imget.h"
#include "im_tpl/cPtOfCorrel.h"
#include "im_tpl/max_loc.h"
#include "im_tpl/correl_imget_ptr.h"
#include "im_tpl/algo_filter_exp.h"
#include "im_tpl/oper_assoc_exter.h"
#include "im_tpl/fonc_operator.h"
#include "im_tpl/algo_dist32.h"
#include "im_tpl/ex_oper_assoc_exter.h"
#include "im_tpl/elise_ex_oper_assoc_exter.h"
#include "ext_stl/Nappes.h"
#include "im_tpl/ProgDyn2D.h"
#include "im_tpl/algo_cc.h"

#include "ext_stl/fifo.h"
#include "ext_stl/intheap.h"
#include "ext_stl/pack_list.h"
#include "ext_stl/numeric.h"
#include "ext_stl/cluster.h"
#include "ext_stl/tab2D_dyn.h"
#include "ext_stl/appli_tab.h"
#include "ext_stl/elslist.h"

#include "graphes/graphe.h"
#include "graphes/graphe_implem.h"
#include "graphes/algo_planarite.h"

#include "algo_geom/qdt.h"
#include "algo_geom/qdt_implem.h"
#include "algo_geom/rvois.h"
#include "algo_geom/integer_delaunay_mediatrice.h"
#include "algo_geom/delaunay_mediatrice.h"
#include "algo_geom/qdt_insertobj.h"
#include "algo_geom/Shewchuk.h"

#include "im_special/hough.h"

#include <sys/types.h>

#include "../src/AMD/amd.h"
#include "../src/AMD/amd_internal.h"
#include "../src/geom2d/gpc.h"
#include "../src/ori_phot/all_phot.h"
#include "../src/HassanArrangt/cElHJa_all.h"
#include "../src/hough/hough_include.h"

#include "../src/uti_image/MpDcraw/MpDcraw.h"

#include "../src/uti_phgrm/ReducHom/ReducHom.h"
#include "../src/uti_phgrm/Apero/cParamApero.h"
#include "../src/uti_phgrm/Apero/Apero.h"
#include "../src/uti_phgrm/MICMAC/MICMAC.h"
#include "../src/uti_phgrm/Porto/Porto.h"
#include "../src/uti_phgrm/MICMAC/cOrientationGrille.h"
