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




void BenchcSysQuadCreuse
     (
         INT aNbVar,
         INT aNbEq,
         cNameSpaceEqF::eTypeSysResol aType
      )
{
    bool isFixe = NRrandom3() < 0.5;
    //cFormQuadCreuse aSCr(aNbVar,isFixe);
    cFormQuadCreuse aSCr( aNbVar, cElMatCreuseGen::StdNewOne( aNbVar, aNbVar, isFixe ) ); // __NEW
    //aSCr.SetEpsB(1e-10);
    cGenSysSurResol & aS1 = aSCr;

    L2SysSurResol   aSPl(aNbVar);
    cGenSysSurResol & aS2 = aSPl;

    bool SysFix = (aType == cNameSpaceEqF::eSysCreuxFixe);
    cSetEqFormelles aSet2(aType); // ,1e-10);
    cEqFormelleLineaire * anEq2_1 = aSet2.NewEqLin(1,aNbVar);
    cEqFormelleLineaire * anEq2_3 = aSet2.NewEqLin(3,aNbVar);


    Im1D_REAL8   aIm2(aNbVar,0.0);
    for (INT aK=0; aK<aNbVar; aK++)
        aSet2.Alloc().NewInc(aIm2.data()+aK);
    aSet2.SetClosed();
    //if (aType !=  cNameSpaceEqF::eSysPlein)
    //   aSet2.FQC()->SetEpsB(1e-10);


    Im1D_REAL8 aF(aNbVar,0.0);

    for (INT aNbTest=0 ; aNbTest<3 ; aNbTest++)
    {
        for (INT aX =0 ; aX<aNbVar; aX++)
        {
	    std::vector<INT> V;
	    V.push_back(aX);

            REAL aPds = 0.5;
            double C1 = 1;
            aF.data()[aX] = C1;
            REAL aCste =  NRrandom3();
	    if (isFixe)
	    {
		if (aX==0)
	           aSCr.SetOffsets(V);
                //aS1.GSSR_AddNewEquation_Indexe(V,aPds,&C1,aCste);
                aS1.GSSR_AddNewEquation_Indexe(V,aPds,&C1,aCste);
	    }
	    else 
                aS1.GSSR_AddNewEquation(aPds,aF.data(),aCste);
            aS2.GSSR_AddNewEquation(aPds,aF.data(),aCste);

	     if ((NRrandom3() < 0.5) && (! SysFix))
                anEq2_1->AddEqNonIndexee(aCste,&C1,aPds,V);
	     else
	     {
                if (SysFix)
		   aSet2.FQC()->SetOffsets(V);
                anEq2_1->AddEqIndexee(aCste,&C1,aPds,V);
	     }

            aF.data()[aX] = 0;
        }

        for (INT aK =0 ; aK<aNbEq; aK++)
	{
	     std::vector<INT> aVInd;
	     REAL   aV3[3];

             static INT I1=0;
             static INT I2=0;
             static INT I3=0;
	     bool TransI =     (NRrandom3() <0.5)
		           &&  (aK != 0)
		           &&  (I1 < aNbVar-1)
		           &&  (I2 < aNbVar-1)
		           &&  (I3 < aNbVar-1); 
	     if (TransI)
	     {
		     I1++;
		     I2++;
		     I3++;
	     }
	     else
	     {
                I1 = ElMin(aNbVar-1,INT(NRrandom3()*aNbVar));
                I2 = ElMin(aNbVar-1,INT(NRrandom3()*aNbVar));
                I3 = ElMin(aNbVar-1,INT(NRrandom3()*aNbVar));
	     }
             REAL aCste = NRrandom3();


	     aV3[0] = NRrandom3();
	     aV3[1] = NRrandom3();
	     aV3[2] = NRrandom3();

	     aF.data()[I1] += aV3[0];
             aF.data()[I2] += aV3[1];
             aF.data()[I3] += aV3[2];
	     aVInd.push_back(I1);
	     aVInd.push_back(I2);
	     aVInd.push_back(I3);

	     if ((NRrandom3() < 0.5) && (! SysFix))
                anEq2_3->AddEqNonIndexee(aCste,aV3,1,aVInd);
	     else
	     {
		if (SysFix)
		{
                   if (! TransI)
		      aSet2.FQC()->SetOffsets(aVInd);
		}
                anEq2_3->AddEqIndexee(aCste,aV3,1,aVInd);
	     }

	     if ((NRrandom3()<0.5) || isFixe)
	     {
	         if (isFixe) 
		 {
                     if (! TransI)
                        aSCr.SetOffsets(aVInd);
		 }
                 aS1.GSSR_AddNewEquation_Indexe(aVInd,1,aV3,aCste);
	     }
	     else
                 aS1.GSSR_AddNewEquation(1,aF.data(),aCste);

	     if (NRrandom3()<0.5)
                 aS2.GSSR_AddNewEquation_Indexe(aVInd,1,aV3,aCste);
	     else
                 aS2.GSSR_AddNewEquation(1,aF.data(),aCste);


             aF.data()[I1] = 0;
             aF.data()[I2] = 0;
             aF.data()[I3] = 0;
	}
        


	bool OK;
        Im1D_REAL8 Sol1 = aS1.GSSR_Solve(&OK);
        Im1D_REAL8 Sol2 = aS2.GSSR_Solve(&OK);
	REAL aDif;
	ELISE_COPY(Sol1.all_pts(),Abs(Sol1.in()-Sol2.in()),VMax(aDif));
	cout << "a Dif " << aDif << "\n";
	BENCH_ASSERT(aDif<1e-4);

	aSet2.SolveResetUpdate();
	ELISE_COPY(Sol1.all_pts(),Abs(aIm2.in()-Sol2.in()),VMax(aDif));
	cout << "a Dif " << aDif << "\n";
	BENCH_ASSERT(aDif<1e-4);



        aS1.GSSR_Reset();
        aS2.GSSR_Reset();
    }
}

void BenchcSysQuadCreuse()
{
   for (INT aK = 0; aK< 200 ; aK++)
   {
       cout << "BenchcSysQuadCreuse " << aK << "\n";
       // bool Pair = ((aK%2)==0);
       cNameSpaceEqF::eTypeSysResol aType = cNameSpaceEqF::eSysPlein;
       if ((aK%3) ==1) aType = cNameSpaceEqF::eSysCreuxMap;
       if ((aK%3) ==0) aType = cNameSpaceEqF::eSysCreuxFixe;
       BenchcSysQuadCreuse(5,1,aType);
       BenchcSysQuadCreuse(10,10,aType);
       BenchcSysQuadCreuse(20,20,aType);
       BenchcSysQuadCreuse(20,30,aType);
       BenchcSysQuadCreuse(50,50,aType);
       BenchcSysQuadCreuse(50,50,aType);
       BenchcSysQuadCreuse(50,50,aType);
   }
}



REAL DiffRel(REAL v1,REAL v2,REAL eps)
{
     return ElAbs(v1-v2) /(ElAbs(v1)+ElAbs(v2)+eps);
}

class  cBenchLeastSquare : public FoncNVarDer<REAL>
{
      public :
        cBenchLeastSquare(INT ,INT,bool );
        void  TestFoncNVar();
      private :

        REAL ValFNV(const REAL *);
        void  GradFNV(double *, const double *);

    
        INT              mNbVar;
        INT              mNbEq;
        SystLinSurResolu mSys;
	Im1D_REAL8       mSol;
	REAL             mResidu;
	Im1D_REAL8       mSolEps;

        Im1D_REAL8       mTmpVF;
        REAL *           mDataTmpVF;

};

void  cBenchLeastSquare::GradFNV(double * aGrad, const double * aP)
{
    for (INT kV=0; kV<mNbVar ; kV++)
    {
        aGrad[kV] = 0;
    }

    for (INT kEq=0; kEq<mNbEq ; kEq++)
    {
        REAL aPCom =  2*mSys.Residu(aP,kEq) * mSys.Pds(kEq);
        for (INT kV=0; kV<mNbVar ; kV++)
        {
            aGrad[kV] += aPCom *  mSys.CoefLin(kV,kEq);
        }
    }
}

void cBenchLeastSquare::TestFoncNVar()
{
    cFormQuadCreuse aFQuad(mNbVar,false);
    aFQuad.SetEpsB(1e-10);
    cOptimSommeFormelle anOSF(mNbVar);
    for (INT kEq=0; kEq<mNbEq ; kEq++)
    {
	Fonc_Num f = Fonc_Num(0);
        for (INT kV=0; kV<mNbVar ; kV++)
        {
	    f = f+mSys.CoefLin(kV,kEq)*kth_coord(kV);
        }
	f = f-mSys.CoefCste(kEq);
	f = Square(f) *  mSys.Pds(kEq);
	anOSF.Add(f,NRrandom3()<0.5);
	aFQuad.AddDiff(f);
    }
    
    // Verif de gradient + fonc sur aFQuad
    for (INT aNb=0 ; aNb<10 ; aNb++)
    {
       Im1D_REAL8 aPt(mNbVar);
       ELISE_COPY(aPt.all_pts(),frandr(),aPt.out());

       REAL aVSom = anOSF.ValFNV(aPt.data());
       REAL aVQ = aFQuad.ValFNV(aPt.data());
       REAL aVSys = mSys.L2SomResiduPond(aPt); 

       REAL aDif =   DiffRel(aVQ,aVSys,epsilon);
       BENCH_ASSERT(aDif<epsilon);


       aDif =   DiffRel(aVSom,aVSys,epsilon);
       BENCH_ASSERT(aDif<epsilon);

       Im1D_REAL8 aGradQ(mNbVar,0.0);
       Im1D_REAL8 aGradSys(mNbVar,0.0);
       Im1D_REAL8 aGradSom(mNbVar,0.0);

        aFQuad.GradFNV(aGradQ.data(),aPt.data());
        GradFNV(aGradSys.data(),aPt.data());
        anOSF.GradFNV(aGradSom.data(),aPt.data());

        for (INT kV=0; kV<mNbVar ; kV++)
	{
	    REAL aGQ = aGradQ.data()[kV];
	    REAL aGSys = aGradSys.data()[kV];
	    REAL aGSom = aGradSom.data()[kV];

            aDif =   DiffRel(aGQ,aGSys,epsilon);
            BENCH_ASSERT(aDif<epsilon);

            aDif =   DiffRel(aGSom,aGSys,epsilon);
            BENCH_ASSERT(aDif<epsilon);
	}
    }

    // Verif de la formule du gradient
	{
    for (INT aNb=0 ; aNb<10 ; aNb++)
    {
       Im1D_REAL8 aPt(mNbVar,0.0);
       Im1D_REAL8 aGrad(mNbVar,0.0);
       GradFNV(aGrad.data(),aPt.data());

       Im1D_REAL8 aDep(mNbVar,0.0);
       ELISE_COPY(aDep.all_pts(),(frandr()-0.5) * 0.0001,aDep.out());

       REAL aScal;
       ELISE_COPY(aDep.all_pts(),aDep.in()*aGrad.in(),sigma(aScal));

       REAL f1 = mSys.L2SomResiduPond(aDep);
       ELISE_COPY(aDep.all_pts(),-aDep.in(),aDep.out());
       REAL f2 =  mSys.L2SomResiduPond(aDep); 

       REAL aDif = DiffRel(f1,f2,epsilon);
       BENCH_ASSERT(aDif<BIG_epsilon);

    }
	}
    Im1D_REAL8 aPt(mNbVar,0.0);
    powel(aPt.data(),1e-8,200);

    REAL aRes1 = mSys.L2SomResiduPond(mSol);
    REAL aRes2 = mSys.L2SomResiduPond(aPt);
    REAL aDif = DiffRel(aRes1,aRes2,epsilon);
    BENCH_ASSERT(aDif<epsilon);


    ELISE_COPY(aPt.all_pts(),frandr(),aPt.out());
    GradConj(aPt.data(),1e-8,200);
    REAL aRes3 = mSys.L2SomResiduPond(aPt);
    aDif =   DiffRel(aRes1,aRes3,epsilon);
    BENCH_ASSERT(aDif<epsilon);

    ELISE_COPY(aPt.all_pts(),frandr(),aPt.out());




    anOSF. GradConjMin(aPt.data(),1e-8,200);
    REAL aRes4 = mSys.L2SomResiduPond(aPt); 
    aDif =   DiffRel(aRes1,aRes4,epsilon);
    BENCH_ASSERT(aDif<epsilon);
}


REAL cBenchLeastSquare::ValFNV(const REAL * aPt)
{
    for (INT k=0 ; k<mNbVar ; k++)
        mDataTmpVF[k] = aPt[k];

    return  mSys.L2SomResiduPond(mTmpVF);
}

cBenchLeastSquare::cBenchLeastSquare
(
    INT aNbVar,
    INT aNbEq,
    bool SomForm
)  :
   FoncNVarDer<REAL> (aNbVar),
   mNbVar (aNbVar),
   mNbEq  (aNbEq),
   mSys (aNbVar,aNbEq),
   mSol (1),
   mSolEps (aNbVar),  
   mTmpVF  (aNbVar),
   mDataTmpVF (mTmpVF.data())
{
   Im1D_REAL8 aFLin(aNbVar);
   REAL8* aDLin = aFLin.data();

   for (INT iEq = 0 ; iEq < aNbEq ; iEq++)
   {
       for (INT iVar=0 ; iVar<aNbVar ; iVar++)
       {
	    if (SomForm)
	    {
		if (iEq<2*aNbVar)
	           aDLin[iVar] = (iVar==(iEq%mNbVar));
		else
	           aDLin[iVar] = NRrandC() * (NRrandC()>0);
	    }
	    else
	       aDLin[iVar] = NRrandC();
       }
       mSys.PushEquation
       (
           aFLin,
	   NRrandC() * 1e3,
	   0.1 + NRrandom3()
       );
   }

   bool Ok;
   mSol = mSys.L2Solve(&Ok);
   mResidu = mSys.L2SomResiduPond(mSol);
   BENCH_ASSERT(Ok);

   for (INT k=0 ; k< 200 ; k++)
   {
	ELISE_COPY
	(
	    mSolEps.all_pts(),
	    mSol.in() + (frandr()-0.5),
	    mSolEps.out()
	);
	REAL ResEps =  mSys.L2SomResiduPond(mSolEps);
	 // cout << (ResEps-mResidu) << " " << mResidu << "\n";
	BENCH_ASSERT(ResEps>mResidu);
   }
   // getchar();
}

void bench_triviale_opt_sous_contrainte()
{
   // Miminise x2+y2, sous la contrainte x+y=2
     L2SysSurResol aSys(2);
     double C[2] = {1,1};
     aSys.GSSR_AddContrainte(C,3);

     double Fx[2] = {1,0};
     aSys.GSSR_AddNewEquation(1.0,Fx,0);
     double Fy[2] = {0,1};
     aSys.GSSR_AddNewEquation(1.0,Fy,0);

     Im1D_REAL8 aSol = aSys.GSSR_Solve(0);
     BENCH_ASSERT(ElAbs(aSol.data()[0] -1.5)<epsilon);
     BENCH_ASSERT(ElAbs(aSol.data()[1] -1.5)<epsilon);

}

void bench_opt_contrainte()
{ 
     INT NbVar = round_ni(1 + 10 * NRrandom3());
     L2SysSurResol aSys1(NbVar);
     L2SysSurResol aSys2(NbVar);

     INT NbContr = round_ni(NbVar*NRrandom3());
     INT NbEq =  round_ni(NbVar + 2 + 10 * NRrandom3());

     for (INT fois=0 ; fois < 3 ; fois++)
     {
         for (INT k=0 ;k<NbContr +NbEq ; k++)
         {
             Im1D_REAL8 C(NbVar);
	     ELISE_COPY(C.all_pts(), frandr(),C.out());
             REAL aVal = NRrandom3();
	     if (k<NbContr)
             {
	         aSys1.GSSR_AddContrainte(C.data(),aVal);
	         aSys2.GSSR_AddNewEquation(1e9,C.data(),aVal);
             }
	     else
             {
	         aSys1.GSSR_AddNewEquation(1,C.data(),aVal);
	         aSys2.GSSR_AddNewEquation(1,C.data(),aVal);
             }
         }

	 Im1D_REAL8 Sol1 = aSys1.GSSR_Solve(0);
	 Im1D_REAL8 Sol2 = aSys2.GSSR_Solve(0);

	 REAL Dif;
	 ELISE_COPY(Sol1.all_pts(),Abs(Sol1.in()-Sol2.in()),VMax(Dif));
         BENCH_ASSERT(Dif<GIGANTESQUE_epsilon);
         aSys1.GSSR_Reset();
         aSys2.GSSR_Reset();
     }
    
}

void bench_least_square()
{
   BenchcSysQuadCreuse();
   bench_triviale_opt_sous_contrainte();
   for (INT k=0 ; k<100 ; k++)
   {
       bench_opt_contrainte();
   }

   bool Ok;
   SystLinSurResolu mSys(1,1);
   Im1D_REAL8  aFlin(1,"2.0");
   mSys.PushEquation(aFlin,3.0,1.0);
   mSys.L2Solve(&Ok); 
   BENCH_ASSERT(Ok);


    for (INT k=0 ; k< 200 ; k++)
    {
       bool SomForm = (k&1 ==0);
       INT aNbVar = 2 + (INT)(10 * NRrandom3());
       INT aNbEq = 2+aNbVar * (1 + (INT)(10 * NRrandom3()));
       if (SomForm)
	  aNbEq += 10;
       cBenchLeastSquare aBLS(aNbVar,aNbEq,SomForm);
       aBLS.TestFoncNVar();
       cout << k << "\n";
    }
}




