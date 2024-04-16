#ifndef  _MMVII_Tpl_ImGradFilter_H_
#define  _MMVII_Tpl_ImGradFilter_H_

#include "MMVII_Images.h"

namespace MMVII
{
/** \file   MMVII_TplGradImFilter.h

*/

template<class TypeIm,class TypeGrad>
    void ComputeSobel
         (
              cDataIm2D<TypeGrad> & aDGX,
              cDataIm2D<TypeGrad> & aDGY,
              const cDataIm2D<TypeIm>& aDIm
         )
{
     aDGX.AssertSameArea(aDGY);
     aDGX.AssertSameArea(aDIm);
     aDGX.InitNull();
     aDGY.InitNull();

     int aSzX = aDGX.Sz().x();
     int aSzY = aDGX.Sz().y();
     for (int aKY=1 ; aKY<aSzY-1 ; aKY++)
     {
         const TypeIm * aLinePrec = aDIm.ExtractRawData2D()[aKY-1] + 1;
         const TypeIm * aLineCur  = aDIm.ExtractRawData2D()[aKY]   + 1;
         const TypeIm * aLineNext = aDIm.ExtractRawData2D()[aKY+1] + 1;
         TypeGrad * aLineGX = aDGX.ExtractRawData2D()[aKY]   + 1;
         TypeGrad * aLineGY = aDGY.ExtractRawData2D()[aKY]   + 1;

          for (int aKX=1 ; aKX<aSzX-1 ; aKX++)
          {
              *aLineGX =   aLinePrec[ 1] + 2*aLineCur[1]  + aLineNext[1]
                          -aLinePrec[-1] - 2*aLineCur[-1] - aLineNext[-1] ;

              *aLineGY =   aLineNext[-1] + 2*aLineNext[0] + aLineNext[1]
                          -aLinePrec[-1] - 2*aLinePrec[0] - aLinePrec[1] ;
              aLinePrec++;
              aLineCur++;
              aLineNext++;
              aLineGX++;
              aLineGY++;
          }
     }
}

template<class TypeIm,class TypeGrad>
    void TruncadeComputeSobel
         (
              cDataIm2D<TypeGrad> & aDGX,
              cDataIm2D<TypeGrad> & aDGY,
              const cDataIm2D<TypeIm>& aDIm,
	      int  aDiv,
	      int  aMaxVal
         )
{
     aDGX.AssertSameArea(aDGY);
     aDGX.AssertSameArea(aDIm);
     aDGX.InitNull();
     aDGY.InitNull();

     int aSzX = aDGX.Sz().x();
     int aSzY = aDGX.Sz().y();
     for (int aKY=1 ; aKY<aSzY-1 ; aKY++)
     {
         const TypeIm * aLinePrec = aDIm.ExtractRawData2D()[aKY-1] + 1;
         const TypeIm * aLineCur  = aDIm.ExtractRawData2D()[aKY]   + 1;
         const TypeIm * aLineNext = aDIm.ExtractRawData2D()[aKY+1] + 1;
         TypeGrad * aLineGX = aDGX.ExtractRawData2D()[aKY]   + 1;
         TypeGrad * aLineGY = aDGY.ExtractRawData2D()[aKY]   + 1;

          for (int aKX=1 ; aKX<aSzX-1 ; aKX++)
          {
              int aGX =  (  aLinePrec[ 1] + 2*aLineCur[1]  + aLineNext[1]
                          -aLinePrec[-1] - 2*aLineCur[-1] - aLineNext[-1]) / aDiv ;

              int aGY =  ( aLineNext[-1] + 2*aLineNext[0] + aLineNext[1]
                          -aLinePrec[-1] - 2*aLinePrec[0] - aLinePrec[1]) / aDiv ;

              aGX = std::max(-aMaxVal,std::min(aMaxVal,aGX));
              aGY = std::max(-aMaxVal,std::min(aMaxVal,aGY));

	      *aLineGX= aGX;
	      *aLineGY= aGY;

              aLinePrec++;
              aLineCur++;
              aLineNext++;
              aLineGX++;
              aLineGY++;
          }
     }
}


class cTabulateGrad
{
      public :
           typedef cIm2D<tREAL4>      tImTab;
           typedef cDataIm2D<tREAL4>  tDataImTab;

           cTabulateGrad(int aVMax);

	   inline tREAL8 GetRho(const cPt2di & aP) const {return mDataRho->GetV(aP);}

/*
           template<class TypeGrad,TypeNorm>
               void ComputeNorm
                    (
                         cDataIm2D<TypeNorm> & aDIm,
                         const cDataIm2D<TypeGrad> & aDGX,
                         const cDataIm2D<TypeGrad> & aDGY
                    )
            {
                 aDGX.AssertSameArea(aDGY);
                 aDGX.AssertSameArea(aDIm);
                 
                 const TypeGrad * aDataGX = aDGX.RawDataLin();
                 const TypeGrad * aDataGY = aDGY.RawDataLin();
                 TypeNorm *       aDataNorm = aDIm.RawDataLin();
            }
*/
      private :
           int     mVMax;
           cPt2di  mP0;
           cPt2di  mP1;

	   tImTab      mTabRho;
	   tDataImTab* mDataRho;
	   tImTab      mTabTeta;
	   tDataImTab* mDataTeta;
};



};

#endif  //  _MMVII_Tpl_ImGradFilter_H_
