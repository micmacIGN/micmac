#ifndef  _MMVII_Images_H_
#define  _MMVII_Images_H_
namespace MMVII
{


/** \file MMVII_Images.h
    \brief 


*/

template <class Type> class cDataIm1D
{
    public :
    // private :
        cDataIm1D<Type>(const cPt1di & aP0,const cPt1di & aP1);
        cDataIm1D<Type>(const cPt1di & aP0,const cPt1di & aP1,const Type & aVal);

        ~cDataIm1D<Type>();


        cPt1di  mP0;
        cPt1di  mP1;
        Type *  Data;
};


};

#endif  //  _MMVII_Images_H_
