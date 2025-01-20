#ifndef  _MMVII_NonLinear2DFiltering_H_
#define  _MMVII_NonLinear2DFiltering_H_

#include "MMVII_Image2D.h"

namespace MMVII
{

/** \file  MMVII_NonLinear2DFiltering.h
    \brief Declaration of non linear filters and (for now ?) some basic extraction
*/


/* *********************************************** */
/*                                                 */
/*        Image Filtering                          */
/*                                                 */
/* *********************************************** */

/// Curvature in the level-line direction ; +or- corner detector
template<class Type> cIm2D<Type> CourbTgt(cIm2D<Type> aImIn);
/// Idem but apply to itself
template<class Type> void SelfCourbTgt(cIm2D<Type> aImIn);

/// Majoritar label in a given neighbouhoord
template <class Type> void SelfLabMaj(cIm2D<Type> aImIn,const cBox2di &);


/// Basic laplacien
template  <class Type> cIm2D<Type> Lapl(cIm2D<Type> aImIn); // Well linear ...

/// return the label majoritar in a neighbourhood
template<class Type> cIm2D<Type> LabMaj(cIm2D<Type>  aImIn,const cBox2di &);



///  Extincion function = dist of image (V!=0) to its complementar (V==0)
//  No longer implemented, used MMV1
//  void MakeImageDist(cIm2D<tU_INT1> aImIn,const std::string & aNameChamfer="32");




};





#endif  //   _MMVII_NonLinear2DFiltering_H_
