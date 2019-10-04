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

cIm2D<tREAL4> CourbTgt(cIm2D<tREAL4> aImIn);
void SelfCourbTgt(cIm2D<tREAL4> aImIn);

template <class Type> void SelfLabMaj(cIm2D<Type> aImIn,const cBox2di &);


cIm2D<tREAL4> Lapl(cIm2D<tREAL4> aImIn); // Well linear ...




};





#endif  //   _MMVII_NonLinear2DFiltering_H_
