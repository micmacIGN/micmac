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



class cTabulateGrad
{
      public :
           typedef cIm2D<tREAL4>      tImTab;
           typedef cDataIm2D<tREAL4>  tDataImTab;

           cTabulateGrad(int aVMax);
           void TabulateNeighMaxLocGrad(int aNbDir,tREAL8 aRho);

	   inline tREAL8 GetRho(const cPt2di & aP) const {return mDataRho->GetV(aP);}

           template<class TypeGrad,class TypeNorm>
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
                 tREAL4 ** aDataTabRho = mDataRho->ExtractRawData2D();
               
                 for ( int aNb= aDGY.NbElem(); aNb>0 ; aNb--)
                 {
                     *aDataNorm = aDataTabRho[(int)*aDataGY][(int)*aDataGX]; 
                     aDataGX++;
                     aDataGY++;
                     aDataNorm++;
                 }
            }

template<class TypeIm,class TypeGrad>
    void ComputeSobel
         (
              cDataIm2D<TypeGrad> & aDGX,
              cDataIm2D<TypeGrad> & aDGY,
              const cDataIm2D<TypeIm>& aDIm,
	      int  aDiv
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

              aGX = std::max(mVMin,std::min(mVMax,aGX));
              aGY = std::max(mVMin,std::min(mVMax,aGY));

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

      private :
           int  VMax() const; ///< Accessor
           void TabulateTabAng(int aNbDir);
           int  Teta2Index(tREAL8 Teta) const;
           tREAL8  Index2Teta(int aInd) const;



           typedef tU_INT1  tInAng;
           int     mVMin;
           int     mVMax;
           int     mNbDirTabAng;
           cPt2di  mP0;
           cPt2di  mP1;

	   tImTab      mTabRho;
	   tDataImTab* mDataRho;
	   tImTab      mTabTeta;
	   tDataImTab* mDataTeta;

           cIm2D<tInAng>       mImIndAng;
           cDataIm2D<tInAng>*  mDataIIA;

           std::vector<std::vector<cPt2di> > mTabNeighMaxLocGrad;
};



};

#endif  //  _MMVII_Tpl_ImGradFilter_H_
