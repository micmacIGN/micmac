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
#include <algorithm>




class cBenchCorrSub
{
     public :
           cBenchCorrSub(Pt2di aSz,REAL anAmpl,REAL EchR,REAL TrR);
	   void MakeACorrel();
     private :
	   class cIm
	   {
	        public :
                   Im2D_U_INT1 mIm;

		   cIm (Pt2di aSz) :
		       mIm(aSz.x,aSz.y)
		   {
		   }
	   };

           class cStat
	   {
                public :
                   void Add(REAL aV){mVals.push_back(aV);};
		   REAL Med();
	           std::vector<REAL> mVals;
	   };


	   REAL mAmpl;
	   Pt2di mSz;
	   Im2D_REAL4 mDistX;
	   Im2D_REAL4 mDistY;

           cIm mIm1;
           cIm mIm2;
           Video_Win mW1;
           Video_Win mW2;


	   REAL Rand(REAL aRab,INT aSz)
	   {
	     	return 2* aRab
			+ (aSz-4*aRab) * NRrandom3();
	   }
	   
};

REAL cBenchCorrSub::cStat::Med()
{
    std::sort(mVals.begin(),mVals.end());
    return mVals[ mVals.size()/2];
}


void cBenchCorrSub::MakeACorrel()
{

     REAL aSzVgn = 10.0;
     REAL aStep  = 0.5;


     OptCorrSubPix_Diff<U_INT1> aOCSPD
     (
           mIm1.mIm,mIm2.mIm,
	   aSzVgn,aStep,
	   Pt3dr(-10,-10,-10)
     );

      
     OptimTranslationCorrelation<U_INT1> aOpComb
     (
          0.001, 1.0, 1,
          mIm1.mIm,mIm2.mIm,
          aSzVgn,aStep
     );

      
     TIm2D<REAL4,REAL8> aTDX(mDistX);
     TIm2D<REAL4,REAL8> aTDY(mDistY);

     INT NbStep = 30;
     std::vector<cStat> aStatRef(NbStep);
     std::vector<cStat> aStatComb(NbStep);
     cStat aStatCombRef;

    ElTimer aChrono;
    REAL aTimeComb =0;
    REAL aTimeDiff =0;

     for (INT aKP=0 ; aKP<100 ; aKP++)
     {
         Pt2dr aP1(Rand(aSzVgn,mSz.x),Rand(aSzVgn,mSz.y));

	 aOpComb.SetP0Im1(aP1);
	 aOpComb.optim(Pt2dr(0,0));
	 Pt2dr aPComb = aP1+aOpComb.param();

	 Pt2dr aPComb1Rec(aTDX.getr(aPComb),aTDY.getr(aPComb));
	 aStatCombRef.Add(euclid(aP1,aPComb1Rec));
	 Pt2dr aP2Opt =  aP1;

         if (aKP % 10 ==0)
         {
           aChrono.reinit();
           for (INT aCp =0 ; aCp < 10 ; aCp++)
               aOpComb.optim(Pt2dr(0,0));
           aTimeComb += aChrono.uval();

           aChrono.reinit();
           for (INT aCp =0 ; aCp < 10 ; aCp++)
               aOCSPD.Optim(aP1,aP1);
           aTimeDiff += aChrono.uval();
         }

	 for (INT aKOpt =0 ; aKOpt<NbStep ; aKOpt++)
	 {
	     Pt3dr aPOpt = aOCSPD.Optim(aP1,aP2Opt);

	     aP2Opt = Pt2dr(aPOpt.x,aPOpt.y);

	     Pt2dr aP1Rec(aTDX.getr(aP2Opt),aTDY.getr(aP2Opt));
	     REAL aDistRef =  euclid (aP1,aP1Rec);
	     REAL aDistComb =  euclid  (aP2Opt,aPComb);
	     // cout << aDistRef  << " " << aDistComb  << " "  << "\n";
	     aStatRef[aKOpt].Add(aDistRef);
	     aStatComb[aKOpt].Add(aDistComb);
         }
     }
/*
     for (INT aKOpt =0 ; aKOpt<NbStep ; aKOpt++)
     {
          cout << "Step " << aKOpt 
	       << " DIST MED " << aStatRef[aKOpt].Med()
               << "  / Comb " <<  aStatComb[aKOpt].Med() 
	       << "\n";
     }
     cout << "DMED Comb " << aStatCombRef.Med() << "\n";
     cout << "Time "
          << " Diff " << aTimeDiff
          << " Comb " << aTimeComb
          << "\n";
*/
     BENCH_ASSERT(aStatComb[NbStep-1].Med() < 1e-1);
}

     

cBenchCorrSub::cBenchCorrSub(Pt2di aSz,REAL anAmpl,REAL EchR,REAL TrR) :
    mAmpl  (anAmpl),
    mSz    (aSz),
    mDistX(aSz.x,aSz.y),
    mDistY(aSz.x,aSz.y),
    mIm1(aSz),
    mIm2(aSz),
    mW1 (Video_Win::WStd(aSz,1.0)),
    mW2 (Video_Win::WStd(aSz,1.0))
{
     static const INT NbV = 6;
     INT  SzV[NbV] =  {1,2,4,8,16,32};
     REAL PdsV[NbV] = {1,2,3,4,5,6};

     ELISE_COPY
     (
         mIm1.mIm.all_pts(),
         10.0+ unif_noise_4(PdsV,SzV,NbV) * 235.0,
            mIm1.mIm.out()
        |   mW1.ogray()
     );



     ELISE_COPY
     (
         mIm2.mIm.all_pts(),
         Virgule
	 (
	     FX + mAmpl * sin(FY/50.0),
	     FY + mAmpl * sin(FX/50.0)
	 ),
         Virgule(mDistX.out(),mDistY.out())
	    
     );

     Fonc_Num EchDistRad = (1+cos(6+FX/60.0)) * (1+cos(FY/77.0)) / 4.0;
     Fonc_Num TrDistRad =  64.0 * ((1+cos(6+FX/72.0)) * (1+cos(FY/55.0)));
     Fonc_Num DistGeom = Virgule(mDistX.in(),mDistY.in());

     REAL CsteR =1-EchR-TrR;

     ELISE_COPY
     (
         mIm2.mIm.all_pts(),
         mIm1.mIm.in(0)[DistGeom]*(CsteR+EchR*EchDistRad) + TrR*TrDistRad,
            mIm2.mIm.out()
        |   mW2.ogray()
     );	


}

void BenchCorrSub()
{
     cBenchCorrSub aBCS(Pt2di(300,300),1.0,0.2,0.2);
     for (INT kCor=0; kCor< 3 ; kCor++)
	 aBCS.MakeACorrel();
}


