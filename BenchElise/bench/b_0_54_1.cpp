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

template <class Type> Fonc_Num GenFoncDeriv(Type aF1,Type aF2)
{
    return  FX + aF1 * aF2 + aF1* FX + cos(aF2) + sin(aF1*aF2*FX) ;
}

class bench_cste_der
{
public :


   bench_cste_der () :
       P1(1),
       P2(3)
   {
   
     for (INT aK = 0 ; aK< 100 ; aK++)
     {
         REAL aV1 = NRrandom3();
         REAL aV2 = NRrandom3();

         
         cVarSpec aC1(aV1);
         cVarSpec aC2(aV2);

         Fonc_Num F1 = GenFoncDeriv(aC1,aC2);
         Fonc_Num F2 = GenFoncDeriv(FY,FZ);

         REAL aV0 = NRrandom3();

         P1(0) = aV0;

         P2(0) = aV0;
         P2(1) = aV1;
         P2(2) = aV2;


         Verif(F1,F2);
         Verif(F1.deriv(0),F2.deriv(0));
         Verif((FX+aC1*aC1).deriv(aC1),(FX+FY*FY).deriv(1));

         Verif(F1.deriv(aC1),F2.deriv(1));
         Verif(F1.deriv(aC2).deriv(aC1),F2.deriv(2).deriv(1));
    }
  }

    PtsKD P1;
    PtsKD P2;

     void Verif(Fonc_Num aF1,Fonc_Num aF2)
     {

         BENCH_ASSERT(ElAbs(aF1.ValFonc(P1)-aF2.ValFonc(P2))<epsilon);
     }
};




void VisuD357()
{
    ElDistRadiale_Pol357 aDist
    (
              1e5,
               Pt2dr(1963.02,2085.47),
               -2.97049e-9,
               2.071425e-16,
               -6.00075e-24
    );

    Video_Win  aW=  Video_Win::WStd(Pt2di(4000,4000),0.20);


    for (int X = 0 ; X <4000 ; X+= 100)
        for (int Y = 0 ; Y <4000 ; Y+= 100)
	{
		Pt2dr aP0(X,Y);
		Pt2dr aP1(X,Y+100);
		aW.draw_seg
                (
		    aDist.Direct(aP0),
		    aDist.Direct(aP1),
		    aW.pdisc()(P8COL::red)
                );
        }

}

/*
static Pt3dr P3Rand()
{
     return Pt3dr
            (
               NRrandC()*2,
               NRrandC()*2,
               -1*(1+NRrandom3())
            );
}

static Pt2dr P2Rand() {return Pt2dr::FromPolar(0.1+NRrandom3(),NRrandC() * 10);}


static REAL RandTeta()
{
   return NRrandC() * 3e-2;
}
*/




class BenchAutoCalibr
{
      public :

	      ~BenchAutoCalibr();
	      BenchAutoCalibr(REAL DistEpipole);
	      void AddDr(REAL NoiseAngul);

	      void Solve();

               void SetDistortion
                    (
                          Pt2dr aCentre,
                          REAL  aCoef3,
                          REAL  aCoef5,
                          REAL  aCoef7
                    );
      private :

	      void SolveUndist();
	      void SolveDist();

               Pt2dr            mEpipole;
	       cElFaisceauDr2D  mFaisceau;
               Pt2dr                   mCentrDist;
               ElDistRadiale_Pol357  * mDist;
               std::vector<REAL>   mVCoeff;
               std::vector<REAL>   mVCoeffInit;
             
};

REAL ValDef (const std::vector<REAL> & aVec,INT anInd)
{
   return (anInd < INT(aVec.size())) ? aVec[anInd] : 0.0;
}

void BenchAutoCalibr::SetDistortion
     (
           Pt2dr aCentre,
           REAL  aCoef3,
           REAL  aCoef5,
           REAL  aCoef7
     )
{
     delete mDist;
     mDist = new ElDistRadiale_Pol357(1e5,aCentre,aCoef3,aCoef5,aCoef7);
     mCentrDist = aCentre;

     mVCoeff.push_back(0);
     mVCoeffInit.push_back(aCoef3);

     if ((aCoef5!=0.0) && (aCoef7!=0))
     {
         mVCoeff.push_back(0);
         mVCoeffInit.push_back(aCoef5);
     }

     if (aCoef7!=0)
     {
         mVCoeff.push_back(0);
         mVCoeffInit.push_back(aCoef7);
     }

}






BenchAutoCalibr::~BenchAutoCalibr()
{
    delete mDist;
}

BenchAutoCalibr::BenchAutoCalibr(REAL DistEpipole) :
   mEpipole (Pt2dr::FromPolar(DistEpipole,20*NRrandC())),
   mDist    (0)
{
}


void BenchAutoCalibr::AddDr(REAL NoiseAngul)
{
	Pt2dr aP(NRrandC(),NRrandC());
        REAL teta = angle(mEpipole-aP);
	teta += NRrandC() * NoiseAngul;

        Pt2dr aDir = Pt2dr::FromPolar(1,teta);

       if (mDist)
       {
           REAL eps = 1e-5;
           Pt2dr aP0 = aP;
           Pt2dr aP1 = aP + aDir * eps;

           Pt2dr aQ0 = mDist->Inverse(aP0);
           Pt2dr aQ1 = mDist->Inverse(aP1);


           // Pt2dr aR0 = mDist->Direct(aQ0);
           // Pt2dr aR1 = mDist->Direct(aQ1);
           // cout << euclid (aR0,aP0)* 1e10 << " " << euclid (aR1,aP1) << "\n";

           aP = aQ0;
           aDir = vunit(aQ1-aQ0);
       }
       mFaisceau.AddFaisceau(aP,aDir,1.0);
}

void BenchAutoCalibr::SolveDist()
{
        Pt2dr P0(-1,-1);
        Pt2dr P1(1,1);
	REAL teta =  mFaisceau.TetaDirectionInf();
	REAL phi = 0;
        bool CentreLibre = true;

       Pt2dr aCdist = mCentrDist;
       if (CentreLibre) 
         aCdist = Pt2dr(0,0);

       REAL D1 = 10;
       for (INT aK=0 ; aK <5 ; aK++)
       {
           mFaisceau.CalibrDistRadiale
           (
                aCdist,
                CentreLibre,
                teta,
                phi,
                mVCoeff
           );

           
          ElDistRadiale_Pol357 aDCalc
          (
               1e5,
               aCdist,
               ValDef(mVCoeff,0),
               ValDef(mVCoeff,1),
               ValDef(mVCoeff,2)
          );


	  D1 =  mDist->D1(aDCalc,P0,P1,10);
       }

       cout << "D1 = " << D1 << "\n";
       BENCH_ASSERT(D1 < BIG_epsilon);
}

void BenchAutoCalibr::SolveUndist()
{
	REAL teta =  mFaisceau.TetaDirectionInf();
	REAL phi = 0;

	Pt2dr aP = Pt2dr::FromPolar(1,teta);

	mFaisceau.PtsConvergenceItere(teta,phi,10,1e-10,true);

	aP = Pt2dr::FromPolar(1,teta);

	Pt2dr epi = Pt2dr::FromPolar(1/tan(phi),teta);

        BENCH_ASSERT(euclid(epi,mEpipole) < epsilon);
}

void  BenchAutoCalibr::Solve()
{
    if (mDist)
       SolveDist();
    else
       SolveUndist();
}


void bench_auto_calibration()
{
     bench_cste_der();


     for (INT aK=0 ; aK< 5 ; aK++)
     {
         // On envoie l'epipole tres loins a l'infini
         BenchAutoCalibr aBAC(50 + 3 *NRrandC());

         aBAC.SetDistortion
         (
               Pt2dr(NRrandC(),NRrandC()) * 1e-2,
               NRrandC() * 1e-2,
               NRrandC() * 1e-2,
               NRrandC() * 1e-2
         );

	 for (INT aKP = 0 ; aKP < 100 ; aKP++)
	     aBAC.AddDr(0.0);

	 aBAC.Solve();
     }



     for (INT aK=0 ; aK< 20 ; aK++)
     {
         BenchAutoCalibr aBAC(10 + 3 *NRrandC());

	 for (INT aKP = 0 ; aKP < 100 ; aKP++)
		 aBAC.AddDr(0.05);

	 for (INT aKP = 0 ; aKP < 100 ; aKP++)
		 aBAC.AddDr(0.01);

	 for (INT aKP = 0 ; aKP < 100 ; aKP++)
		 aBAC.AddDr(0.0);

	 aBAC.Solve();

     }
}




