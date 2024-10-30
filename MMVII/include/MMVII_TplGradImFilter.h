#ifndef  _MMVII_Tpl_ImGradFilter_H_
#define  _MMVII_Tpl_ImGradFilter_H_

#include "MMVII_Images.h"

namespace MMVII
{
/** \file   MMVII_TplGradImFilter.h

    This file contains several implementation of "optitmized" gradient filters, 
    generally they are optimized and doi hort-cuts on safe access to images.

*/

/** A  basic an directe imlplementation of sobel gradient, if overflow : too bad ...  dont handle borders 
 * rigourously */

template<class TypeIm,class TypeGrad>
    void ComputeSobel
         (
              cDataIm2D<TypeGrad> & aDGX,
              cDataIm2D<TypeGrad> & aDGY,
              const cDataIm2D<TypeIm>& aDIm
         )
{
     // chexk domain are equals
     aDGX.AssertSameArea(aDGY);  
     aDGX.AssertSameArea(aDIm);
     // too avoid rubish on border, dont supress except by replacing ameliortion on borders
     aDGX.InitNull();
     aDGY.InitNull();

     int aSzX = aDGX.Sz().x();
     int aSzY = aDGX.Sz().y();
     // parse all line except first and last
     for (int aKY=1 ; aKY<aSzY-1 ; aKY++)
     {
         // initialize pointers on the 3 input lines,  begin at x=1
         const TypeIm * aLinePrec = aDIm.ExtractRawData2D()[aKY-1] + 1;
         const TypeIm * aLineCur  = aDIm.ExtractRawData2D()[aKY]   + 1;
         const TypeIm * aLineNext = aDIm.ExtractRawData2D()[aKY+1] + 1;

	 // initialize 2 output pointers, begin at x=1
         TypeGrad * aLineGX = aDGX.ExtractRawData2D()[aKY]   + 1;
         TypeGrad * aLineGY = aDGY.ExtractRawData2D()[aKY]   + 1;

	 // parse all columns except first and last
         for (int aKX=1 ; aKX<aSzX-1 ; aKX++)
         {
              // standar sobel's formula
	            // {F(x+1)-F(x-1)} * t[1 2 1]
              *aLineGX =   aLinePrec[ 1] + 2*aLineCur[1]  + aLineNext[1]      
                          -aLinePrec[-1] - 2*aLineCur[-1] - aLineNext[-1] ;

	            // {F(y+1)-F(y-1)} * [1 2 1]
              *aLineGY =   aLineNext[-1] + 2*aLineNext[0] + aLineNext[1]
                          -aLinePrec[-1] - 2*aLinePrec[0] - aLinePrec[1] ;
	      //  update input & output pointers
              aLinePrec++;
              aLineCur++;
              aLineNext++;
              aLineGX++;
              aLineGY++;
         }
     }
}


/**   This class is usefull for computing accelerated processing on gradient using tabulated values. Typically :
 *
 *      - if gradient is integer
 *      - if it has bounded value
 *
 *      - all value that time consuming like "rho/teta" can be pre-computed in tables
 *
 *      - for more sophisticated values like oriented neighbourhood, the tabulation is made in two times :
 *          - a tabularion that for (Gx,Gy) gives an index of theta
 *          - tabulation that gives the neighborhood for a given index of theta
 *
 *     To waranty a bounded value to gradient, the policy is 4 now pretty basic :
 *
 *         - indicate a bounding value at creation (dimension the size of tabulations)
 *         - indicate a divisor at computation
 *
 *     If necessary occurs, may evolve in less basic as non-linear compression for gradient. But it it will cost
 *     some times, it will be an option ...
 */

class cTabulateGrad
{
      public :
           typedef cIm2D<tREAL4>      tImTab;
           typedef cDataIm2D<tREAL4>  tDataImTab;

	   /// constructor, take max val to  compute tabulations
           cTabulateGrad(int aVMax);

	   /// compute tabulation of directionnal neighbooruod,  go up to R1, and memorize index <R0
           void TabulateNeighMaxLocGrad(int aNbDir,tREAL8 aRho0,tREAL8 aRho1);
	   /// for a given grad Gx,Gy extract directionnal neighboor + index < R1
           std::pair<const std::vector<cPt2di>*,int>  TabNeighMaxLocGrad(const cPt2di & aPGrad) const;

	   // not used for now ... extract inline tREAL8 GetRho(const cPt2di & aP) const {return mDataRho->GetV(aP);}


	   /** for a given pair of gradient image Gx,Gy compute the image of norm of gradient using tabulation,
	    *  !! Make no test on values, pb if out of bound,  btw will work if gradient was computed with one method of the class
	    */
           template<class TypeGrad,class TypeNorm>
               void ComputeNorm
                    (
                         cDataIm2D<TypeNorm> & aDIm,
                         const cDataIm2D<TypeGrad> & aDGX,
                         const cDataIm2D<TypeGrad> & aDGY
                    )
            {
		    // check dommain
                 aDGX.AssertSameArea(aDGY);
                 aDGX.AssertSameArea(aDIm);
                    // as it doesnt use any spatial relation, faster to process the linear data 
		    // extract the lineat buffer
                 const TypeGrad * aDataGX = aDGX.RawDataLin();
                 const TypeGrad * aDataGY = aDGY.RawDataLin();
                 TypeNorm *       aDataNorm = aDIm.RawDataLin();

		     // extract the tabulation as raw-data
                 tREAL4 ** aDataTabRho = mDataRho->ExtractRawData2D();
               
                 for ( int aNb= aDGY.NbElem(); aNb>0 ; aNb--)
                 {
                     // compute norm
                     *aDataNorm = aDataTabRho[(int)*aDataGY][(int)*aDataGX]; 
		     // increment input & output buffers
                     aDataGX++;
                     aDataGY++;
                     aDataNorm++;
                 }
            }

	   /// Version of sobel adapted for further use with tabulations
            template<class TypeIm,class TypeGrad>
                void ComputeSobel
                     (
                          cDataIm2D<TypeGrad> & aDGX,    // output X-gradient image
                          cDataIm2D<TypeGrad> & aDGY,    // output Y-gradient image
                          const cDataIm2D<TypeIm>& aDIm, // input image
	                  int  aDiv                      // divisor, used to limit the truncating effect
                     )
            {
                 mCenterGrad = cPt2dr(0.0,0.0);  // sobel is centered scheme
                 mGradDyn =  8.0 / aDiv;  // 8=4*2 because :  sobel dyn 4=1+2+1, distance bewteeb pix = 2

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
			      // standard formula with division
                          int aGX =  (  aLinePrec[ 1] + 2*aLineCur[1]  + aLineNext[1]
                                      -aLinePrec[-1] - 2*aLineCur[-1] - aLineNext[-1]) / aDiv ;

                          int aGY =  ( aLineNext[-1] + 2*aLineNext[0] + aLineNext[1]
                                      -aLinePrec[-1] - 2*aLinePrec[0] - aLinePrec[1]) / aDiv ;

                          // aGX = std::max(mVMin,std::min(mVMax,aGX));
                          // aGY = std::max(mVMin,std::min(mVMax,aGY));

			  //  memorize truncated values
	                  *aLineGX= std::max(mVMin,std::min(mVMax,aGX));
	                  *aLineGY= std::max(mVMin,std::min(mVMax,aGY));

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

           int     mVMin;  ///< lower bound of grad
           int     mVMax;  ///< upper bound of grad
           int     mNbDirTabAng; ///< number, if any, of tabulated neighboorhoud

           cPt2di  mP0;  ///< origin (negative of tabulations)
           cPt2di  mP1;  ///< upper bound of tabulation
	   cPt2dr  mCenterGrad; ///< memorize the phase diff image/grad (!= 0 for robert/mpd)
	   tREAL8  mGradDyn;    ///< memorize answer to a constant slope of 1.0, integrat divisor

	   tImTab      mTabRho;   ///< tabulation of rho as function of Gx,Gy
	   tDataImTab* mDataRho;   ///< raw data of mTabRho
	   tImTab      mTabTeta;   ///< tabulation of teta as function of Gx,Gy
	   tDataImTab* mDataTeta;  ///< raw data of mTabTeta

           cIm2D<tInAng>       mImIndAng;  ///< angular index of teta if any
           cDataIm2D<tInAng>*  mDataIIA;   ///< raw data of mImIndAng

           std::vector<std::vector<cPt2di> > mTabNeighMaxLocGrad;  ///< directional neighborhood for teta index
           std::vector<int >                 mTabIndMLGRho0;  ///< index in mTabNeighMaxLocGrad for value < Rho0
};



};

#endif  //  _MMVII_Tpl_ImGradFilter_H_
