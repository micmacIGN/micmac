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


class cTestDistFraser
{
	public :
            cTestDistFraser
            (
	         REAL anAmpl,
		 REAL aR3,
		 REAL aR5,
		 REAL aR7,
		 Video_Win * aPW
	    )  :
	        mAmpl (anAmpl),
	        mDRADINIT (anAmpl,Pt2dr(NRrandC(),NRrandC()) * (anAmpl/20.0)),
	        mDist     (mDRADINIT),
		mCam (anAmpl,Pt2dr(0,0),mDist,true),
                mDCam (mCam),
		pW   (aPW)
	    {
		    if (pW)
		    {
                       mCW = pW->sz()/2;
                       mFact = ElMin(mCW.x,mCW.y) / mAmpl;
		    }
		    mDist.DRad().PushCoeff(aR3/pow(anAmpl,2));
		    mDist.DRad().PushCoeff(aR5/pow(anAmpl,4));
		    mDist.DRad().PushCoeff(aR7/pow(anAmpl,6));
	    }

            Pt2dr ToW(Pt2dr aP);
            Pt2dr FromW(Pt2dr aP);
	    cDistModStdPhpgr  & Dist() {return mDist;}

            void TestInv(REAL Step,REAL Exag);

	    void SetP1( REAL aP1)
	    {
		    mDist.P1() = aP1 /pow(mAmpl,1);
	    }
	    void SetP2( REAL aP2)
	    {
		    mDist.P2() = aP2 /pow(mAmpl,1);
	    }

	    void Setb1( REAL ab1)
	    {
                 mDist.b1() = ab1 ;
	    }
	    void Setb2( REAL ab2)
	    {
                 mDist.b2() = ab2 ;
	    }

	 private :
            REAL             mAmpl;
	    ElDistRadiale_PolynImpair mDRADINIT;
	    cDistModStdPhpgr mDist;
	    cCamStenopeModStdPhpgr mCam;
	    cDistStdFromCam  mDCam;
	    Video_Win *      pW;
	    Pt2dr            mCW;
	    REAL             mFact;
};

Pt2dr cTestDistFraser::ToW(Pt2dr aP)
{
   return mCW + aP*mFact;
}

Pt2dr cTestDistFraser::FromW(Pt2dr aP)
{
   return (aP-mCW)/mFact;
}

void cTestDistFraser::TestInv(REAL aStep,REAL Exag)
{
    mCam.Dist() =  mDist;
    static double DMax = 0;
    if (pW)
       pW->clear();

     REAL aV0 = round_up(1/aStep) * aStep;
     for (REAL anX = -aV0 ;  anX<= aV0  ; anX+= aStep)
         for (REAL anY = -aV0 ;  anY<= aV0  ; anY+= aStep)
	 {
             Pt2dr aP = Pt2dr(anX,anY)*mAmpl;
             Pt2dr aQ = mDist.Direct(aP);
	     Pt2dr aQV = aP + (aQ-aP)*Exag;

	     Pt2dr aQI = mDist.Inverse(aQ);

	     Pt2dr aq = mDCam.Direct(aP);
	     Pt2dr ap = mDCam.Inverse(aq);

	     REAL dpq = euclid(ap,aP);


	     REAL aD =  euclid(aP,aQI) ;
	     ElSetMax(DMax,aD);
	     ElSetMax(DMax,dpq);
	     REAL eps = 1e-7;
	     if (DMax>=eps)
	     {
	         cout << DMax << " : " <<  aP << aQ << "\n";
	         BENCH_ASSERT(DMax<eps);
	     }

	     if (pW)
	     {
                pW->draw_circle_abs(ToW(aP),2.0,pW->pdisc()(P8COL::red));
                pW->draw_seg(ToW(aP),ToW(aQV),pW->pdisc()(P8COL::green));
	     }
	 }
}


// Test sur une distortion "concrete", on verifie l'inversion
/*
void TestDistFraserNonForm
     (
         REAL aFact,
	 REAL Coeff
	 bool WithDcentr,
	 bool WithPlan
     )
{
}
*/





void bench_Fraser()
{
     Video_Win * pW  = Video_Win::PtrWStd(Pt2di(500,500));

     for (INT aK=0 ; aK< 2000 ; aK++)
     {
        REAL anAmpl = pow(10000.0,NRrandC());
	cout << "AMPL = " << anAmpl << "\n";
        //REAL anAmpl = 10.0;
        // cTestDistFraser aTDF(anAmpl,0.1,-0.05,0.02,pW);
        cTestDistFraser aTDF
		        (
			     anAmpl,
			     0.03*NRrandC(),
			     0.02*NRrandC(),
			     0.01*NRrandC(),
			     pW
			 );

        aTDF.SetP1(0.02*NRrandC());
        aTDF.SetP2(0.02*NRrandC());
        aTDF.Setb1(0.05*NRrandom3());
        aTDF.Setb2(0.05*NRrandC());
        aTDF.TestInv(0.10,2.0);
     }

}



