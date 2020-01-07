#ifndef  _MMVII_NonLinear2DFiltering_H_
#define  _MMVII_NonLinear2DFiltering_H_
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
cIm2D<tREAL4> Lapl(cIm2D<tREAL4> aImIn); // Well linear ...




};





#endif  //   _MMVII_NonLinear2DFiltering_H_
