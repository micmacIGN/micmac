#pragma once

#include "general/all.h"
#include "private/all.h"

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
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <set>


#if (ELISE_windows)
	#ifdef INT
		#undef INT
	#endif
	#include "Windows.h"
	#include "winbase.h"
	#include "direct.h"
#endif


#include "XML_GEN/all.h"
#include "XML_GEN/SuperposImage.h"
#include "XML_GEN/ParamChantierPhotogram.h"

#include "api/vecto.h"

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

#include "algo_geom/qdt.h"
#include "algo_geom/qdt_implem.h"
#include "algo_geom/rvois.h"
#include "algo_geom/delaunay_mediatrice.h"
#include "algo_geom/qdt_insertobj.h"
#include "algo_geom/Shewchuk.h"

#include "api/vecto.h"
#include "api/cox_roy.h"
#include "api/el_regex.h"

#include "ext_stl/fifo.h"
#include "ext_stl/fixed.h"
#include "ext_stl/intheap.h"
#include "ext_stl/pack_list.h"
#include "ext_stl/numeric.h"

#include "graphes/graphe.h"
#include "graphes/graphe_implem.h"
#include "graphes/algo_planarite.h"
#include "graphes/algo_pcc.h"

#include "im_tpl/correl_imget.h"
#include "im_tpl/cPtOfCorrel.h"
#include "im_tpl/image.h"
#include "im_tpl/max_loc.h"

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


