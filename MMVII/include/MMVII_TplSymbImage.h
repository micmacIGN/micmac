#ifndef  _MMVII_TplSymbIm_
#define  _MMVII_TplSymbIm_

#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_MACRO.h"


/** \file MMVII_TplSymbImage.h
    \brief Contains helpers for image as formula

*/
using namespace  NS_SymbolicDerivative;



namespace MMVII
{

template <class TypeUk,class TypeObs>   
         TypeUk  FormalBilinIm2D_Formula
                 (
                      const std::vector<TypeObs> & aVObs,
                      int aKObs0,
                      const TypeUk  &  FX,
                      const TypeUk  & FY
                 )
{
    TypeUk aX0   (aVObs.at(aKObs0));
    TypeUk aY0 (aVObs.at(aKObs0+1));
    TypeUk aCst1 = CreateCste(1.0,aX0);  // create a symbolic formula for constant 1

    TypeUk aWX1 = FX -aX0;
    TypeUk aWX0 = aCst1 - aWX1;
    TypeUk aWY1 = FY -aY0;
    TypeUk aWY0 = aCst1 - aWY1;


    return 
            aWX0 * aWY0 * aVObs.at(aKObs0+2)
          + aWX1 * aWY0 * aVObs.at(aKObs0+3)
          + aWX0 * aWY1 * aVObs.at(aKObs0+4)
          + aWX1 * aWY1 * aVObs.at(aKObs0+5) ;

}

template <class Type,class TypeIm>
   void FormalBilinIm2D_SetObs(std::vector<Type> & aVObs,size_t aK0,cPt2dr aPt,cIm2D<TypeIm> aIm)
{
     const cDataIm2D<TypeIm> & aDIm = aIm.DIm();
     cPt2di aP0 = Pt_round_down(aPt);

     SetOrPush(aVObs, aK0  ,  aPt.x()                            );
     SetOrPush(aVObs, aK0+1,  aPt.y()                            );
     SetOrPush(aVObs, aK0+2,  (Type) aDIm.GetV(aP0)              );
     SetOrPush(aVObs, aK0+3,  (Type) aDIm.GetV(aP0+cPt2di(1,0))  );
     SetOrPush(aVObs, aK0+4,  (Type) aDIm.GetV(aP0+cPt2di(0,1))  );
     SetOrPush(aVObs, aK0+5,  (Type) aDIm.GetV(aP0+cPt2di(1,1))  );
}

constexpr size_t FormalBilinIm2D_NbObs=6;

std::vector<std::string> FormalBilinIm2D_NameObs(const std::string & aPrefix);


};

#endif //  _MMVII_TplSymbIm_

