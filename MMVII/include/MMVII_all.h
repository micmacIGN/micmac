#ifndef _MMVII_ALL_H_
#define _MMVII_ALL_H_
/** \file MMVII_all.h
    \brief Contains all header of MMVII

  Try to put together files having something in common, not always easy ...
*/


// Header standar c++
#include "memory.h"
#include <memory>
#include <iostream>
#include <iomanip> 
#include <fstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <list>
#include <map>
#include <ctime>
#include <chrono>
// #include <algorithm> => put it in file requiring it as it seem to slow down compilation
#include <optional>
#include <cmath>


//========== LIB EXTEN==============


//===========================================
#include "MMVII_AllClassDeclare.h"
#include "MMVII_Error.h"
#include "MMVII_enums.h"
// Header MMVII
// #include "TypeNum.h"
#include "MMVII_Sys.h"
#include "MMVII_memory.h"
#include "MMVII_util_tpl.h"
#include "MMVII_nums.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_util.h"


// Must be here : after general class, used in case WITH_MMVII, before class using cCalculator 
#include "SymbDer/SymbDer_Common.h"

// Les class cPtxd, cPt1d, cPt2d
#include "MMVII_Ptxd.h"
#include "MMVII_Triangles.h"

// Les classe lies aux conversion vers des chaines, fichier ...
#include "MMVII_Stringifier.h"
#include "MMVII_Bench.h"
#include "cMMVII_Appli.h"


//  Classes for images manipulation

#include "MMVII_Images.h"
#include "MMVII_Image2D.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_NonLinear2DFiltering.h"
#include "MMVII_Matrix.h"
#include "MMVII_SysSurR.h"
#include "MMVII_ImageInfoExtract.h"

#include "MMVII_AimeTieP.h"
// #include "MMVII_Tpl_Images.h"

#include "MMVII_Mappings.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"

#include "MMVII_Sensor.h"

#include "MMVII_DeclareAllCmd.h"

#include "MMVII_Interpolators.h"
// communication MMVII/MMv1

#include "MMVII_MMV1Compat.h"
//  #include "MMVII_Derivatives.h" => not include by default now, requires <Eigen/Dense> => time consuming

#include "MMVII_PhgrDist.h"


#endif //  _MMVII_ALL_H_
