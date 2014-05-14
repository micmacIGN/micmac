/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/


#include "StdAfx.h"
#include "bench.h"


/**********************************************/
/**********************************************/
/**********************************************/


class cBenchCorrel
{
   public :
     void DoNothing() {};

     Im2D_REAL8 mIm1;
     Im2D_REAL8 mDup1;
     Im2D_REAL8 mPds1;

     Im2D_REAL8 mIm2;
     Im2D_REAL8 mDup2;
     Im2D_REAL8 mPds2;




     void VerifCorrelCNC
          (
               Pt2di aDec,
               bool Pondered,
               Im2D_REAL8 aCPad,
               REAL anEps,
               Im2D_REAL8  aCNC,
               REAL aRatioSurf
          )
     {
	      REAL aS_CorFFT = ImCorrFromNrFFT(aCPad,aDec);

	      REAL aS,aS1,aS2,aS11,aS12,aS22;

              Symb_FNum aP (    Pondered                                ?
                                trans(mPds1.in(0),aDec)*mPds2.in(0)     :
                                trans(mIm1.inside(),aDec)
                           );
              Symb_FNum aF1 (trans(mIm1.in(0),aDec));
              Symb_FNum aF2 (mIm2.in(0));

              ELISE_COPY 
              ( 
                 mIm1.all_pts(), 
                 Virgule
                 (
                   1,aF1,aF2,
                   aF1*aF1,aF1*aF2,aF2*aF2
                 )*aP,
                 Virgule
                 (
                       Virgule(sigma(aS)  ,sigma(aS1) ,sigma(aS2)),
                       Virgule(sigma(aS11),sigma(aS12),sigma(aS22))
                 )
              );
              if (! Pondered)
	          BENCH_ASSERT(ElAbs(aS12-aS_CorFFT )<epsilon);

              aS = ElMax(aS,anEps);
              aS1 /= aS;
              aS2 /= aS;
              aS11 = aS11/aS - aS1 * aS1 ;
              aS12 = aS12/aS - aS1 * aS2 ;
              aS22 = aS22/aS - aS2 * aS2 ;

              REAL aCor = aS12 / sqrt(ElMax(anEps,aS11*aS22));	
              if (aS<aRatioSurf)
              {
                   aCor = -1 + (aCor+1) * (aS/aRatioSurf);
              }


	      REAL aNCCorFFT = ImCorrFromNrFFT(aCNC,aDec);


              BENCH_ASSERT(ElAbs(aCor-aNCCorFFT)<epsilon);
      }

     cBenchCorrel (Pt2di aSz) :
         mIm1(aSz.x,aSz.y),
         mDup1(aSz.x,aSz.y),
         mPds1(aSz.x,aSz.y),
         mIm2(aSz.x,aSz.y),
         mDup2(aSz.x,aSz.y),
         mPds2(aSz.x,aSz.y)
     {


           ELISE_COPY(mIm1.all_pts(),frandr(),mIm1.out()|mDup1.out());
           ELISE_COPY(mIm2.all_pts(),frandr(),mIm2.out()|mDup2.out());


           ELISE_COPY(mIm1.all_pts(),frandr(),mPds1.out());
           ELISE_COPY(mIm1.all_pts(),frandr(),mPds2.out());



           ElFFTCorrelCirc(mDup1,mDup2);
           Im2D_REAL8 aCPad = ElFFTCorrelPadded(mIm1, mIm2);


           REAL anEps = (1+10*NRrandom3()) * 1e-2;

           REAL aRatioSurf = aSz.x * aSz.y * (1+NRrandom3()) / 6.0;

           Im2D_REAL8 aCNC = ElFFTCorrelNCPadded(mIm1, mIm2,anEps,aRatioSurf);
           Im2D_REAL8 aPdsCNC = ElFFTPonderedCorrelNCPadded
                          (
                                mIm1.in(),
                                mIm2.in(),
                                aSz,
                                mPds1.in(),
                                mPds2.in(),
                                anEps,
                                aRatioSurf
                          );

           for (INT x =-1 ; x<= aSz.x ; x++)
           {
               for (INT y =-1 ; y<= aSz.y ; y++)
               {
                    REAL aSElise;
                    ELISE_COPY
                    (
	                mIm1.all_pts(),
		          mIm1.in()[Virgule(mod(FX+x,aSz.x),mod(FY+y,aSz.y))] 
		        * mIm2.in(),
		        sigma(aSElise)
                    );

	            REAL aSFFT = mDup1.data()[mod(y,aSz.y)][mod(x,aSz.x)];

	            BENCH_ASSERT(ElAbs(aSElise-aSFFT)<epsilon);

                    Pt2di aDec(x,y);
                    VerifCorrelCNC(aDec,false,aCPad, anEps,aCNC,aRatioSurf);
                    VerifCorrelCNC(aDec,true,aCPad, anEps,aPdsCNC,aRatioSurf);

               }
           }
     }
};



void bench_qcorrel(Pt2di aSz)
{
    cBenchCorrel aBC(aSz);
    aBC.DoNothing();
}

void  bench_fft_correl()
{
   for (INT k=0 ; k<10 ; k++)
   {
        bench_qcorrel(Pt2di(4,4));
        bench_qcorrel(Pt2di(8,4));
        bench_qcorrel(Pt2di(4,8));
        bench_qcorrel(Pt2di(8,16));
        bench_qcorrel(Pt2di(16,8));
   }
}





