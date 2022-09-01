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

	/*
template <class Type> class cFormulaBilinIm2D
{
       public :
          // cGenerateFormulaOnIm2D(const std::string & aPrefix);
          static std::vector<std::string> NameObs(const std::string & aPrefix);
          static void InitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<tREAL4> aIm);
          static void InitObs(std::vector<Type> & aVObs,int aK0,cPt2dr aPt,cIm2D<tU_INT1> aIm);
};
*/

template <class TypeUk,class TypeObs>   
         TypeUk  FormulaBilinVal
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


};

#endif //  _MMVII_TplSymbIm_

