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


// Point aleatoire dans l'intervalle [-1,1] en xy [-1,-2] en z,
// on peut plaser de maniere robuste des camera en (0,0)

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


static ElPackHomologue  ListeCplH
                            (
                               INT        aNbPts,
                               ElCamera & aCam1,
                               ElCamera & aCam2
                            )
{
    ElPackHomologue aRes;
    for (INT aKPts=0 ;  aKPts<aNbPts ; aKPts++)
    {
         Pt3dr  aP3 = P3Rand();
         Pt2dr  aQ1 = aCam1.R3toF2(aP3);
         Pt2dr  aQ2 = aCam2.R3toF2(aP3);
         //aRes.add (ElCplePtsHomologues(aQ1,aQ2));
         aRes.Cple_Add (ElCplePtsHomologues(aQ1,aQ2)); // __NEW
    }

/*
    for (INT x=0 ; x<10; x++)
       for (INT y=0 ; y<10; y++)
       {
       }
*/

    return aRes;
}

class BenchProjCavaliere : public ElProj32
{
     public :
          Pt2dr Proj(Pt3dr aP) const 
	  {
             return Pt2dr(aP.x,aP.y) -mPz*aP.z;
	  }
          Pt3dr DirRayon(Pt2dr aP) const
	  {
	     return Pt3dr(mPz.x,mPz.y,1);
	  }

	  void Rayon(Pt2dr aP,Pt3dr &p0,Pt3dr & p1) const 
	  {
		  p0 = Pt3dr(aP.x,aP.y,0);
		  p1 = Pt3dr(aP.x+mPz.x,aP.y+mPz.y,1);
	  }
          void  Diff(ElMatrix<REAL> &,Pt3dr) const 
	  {
		  BENCH_ASSERT(false);
	  }
	  BenchProjCavaliere(Pt2dr aPz) : mPz (aPz) {}
     private :	     
	   Pt2dr mPz;
};



static Polynome2dReal    Rand_Polynome2dReal
                         (
                             REAL CoeffX,
                             REAL CoeffY,
                             INT aDegre,
                             REAL ampl
                         )
{
   Polynome2dReal aRes (aDegre,1.0);

   for (INT kMon=0 ; kMon<aRes.NbMonome() ; kMon++)
   {
        Monome2dReal aMon = aRes.KthMonome(kMon);
        if   (aMon.DegreTot() ==0)
             aRes.SetCoeff(kMon,0);
        else if (aMon.DegreTot() >= 2)
             aRes.SetCoeff(kMon,(NRrandC()*ampl)/aRes.NbMonome());
        else if   (aMon.DegreX() ==1)
             aRes.SetCoeff(kMon,CoeffX);
        else
             aRes.SetCoeff(kMon,CoeffY);

   }
   return aRes;
}




class BenchCamCav : public ElCamera
{
    public :
       BenchCamCav(Pt2dr aPZ) :
           //ElCamera(),
           ElCamera(false/*isC2M*/,eProjectionStenope), // __NEW
           mProj (aPZ)
        {
        }
    private :
        BenchProjCavaliere mProj;

	ElDistortion22_Gen   &      Dist()       {return ElDistortion22_Triviale::TheOne;}
	const ElDistortion22_Gen  & Dist() const {return ElDistortion22_Triviale::TheOne;}

        ElProj32 &        Proj()       {return mProj;}
        const ElProj32 &  Proj() const {return mProj;}
	
	Pt3dr  ImEtProf2Terrain(const Pt2dr&, double)       const { return Pt3dr(0,0,0); }                 // __NEW
	Pt3dr  NoDistImEtProf2Terrain(const Pt2dr&, double) const { return Pt3dr(0,0,0); }                 // __NEW
        double SzDiffFinie()                                const { return 0; }                            // __NEW
        double ResolutionSol()                              const { return 0; }                            // __NEW
        double ResolutionSol(const Pt3d<double>&)           const { return 0; }                            // __NEW
        void   InstanceModifParam(NS_ParamChantierPhotogram::cCalibrationInternConique&) const { return; } // __NEW
};

class BenchCamCavDist : public ElCamera
{
    public :
       BenchCamCavDist(Pt2dr aPZ,Polynome2dReal aDx,Polynome2dReal aDy) :
           //ElCamera(),
           ElCamera(false/*isC2M*/,eProjectionStenope), // __NEW
           mProj (aPZ),
           mDist (aDx,aDy)
        {
        }
    private :
        BenchProjCavaliere mProj;
        ElDistortionPolynomiale  mDist;

	ElDistortion22_Gen   &      Dist()       {return mDist;}
	const ElDistortion22_Gen  & Dist() const {return mDist;}
        ElProj32 &        Proj()       {return mProj;}
        const ElProj32 &  Proj() const {return mProj;}
	
	Pt3dr  ImEtProf2Terrain(const Pt2dr&, double)       const { return Pt3dr(0,0,0); }                 // __NEW
	Pt3dr  NoDistImEtProf2Terrain(const Pt2dr&, double) const { return Pt3dr(0,0,0); }                 // __NEW
        double SzDiffFinie()                                const { return 0; }                            // __NEW
        double ResolutionSol()                              const { return 0; }                            // __NEW
        double ResolutionSol(const Pt3d<double>&)           const { return 0; }                            // __NEW
        void   InstanceModifParam(NS_ParamChantierPhotogram::cCalibrationInternConique&) const { return; } // __NEW
};

class BenchDistSimil : public  ElDistortion22_Gen
{
      public :
	      Pt2dr Direct(Pt2dr aP) const  {return mSim(aP);}
	      bool OwnInverse(Pt2dr & aP) const 
	      {
		      aP = mSimInv(aP);
		      return true;
	      }
	      BenchDistSimil(Pt2dr mP0,Pt2dr mSc) :
		    mSim(mP0,mSc),
	            mSimInv(mSim.inv())
	      {
                    
	      }
      private:

	      ElSimilitude mSim;
	      ElSimilitude mSimInv;
        void Diff (ElMatrix<double> &, Pt2d<double>) const
        {
		 BENCH_ASSERT(false);
        }
};

class BenchCamCavSim : public ElCamera  
{
     public :
       BenchCamCavSim(Pt2dr aPZ,Pt2dr mP0,Pt2dr mSc) :
           //ElCamera(),
           ElCamera(false/*isC2M*/,eProjectionStenope), // __NEW
           mProj (aPZ),
	   mDist (mP0,mSc)
        {
        }
    private :
        BenchProjCavaliere mProj;
	BenchDistSimil     mDist;

	ElDistortion22_Gen   &      Dist()       {return mDist;}
	const ElDistortion22_Gen  & Dist() const {return mDist;}
        ElProj32 &        Proj()       {return mProj;}
        const ElProj32 &  Proj() const {return mProj;}
	
	Pt3dr  ImEtProf2Terrain(const Pt2dr&, double)       const { return Pt3dr(0,0,0); }                 // __NEW
	Pt3dr  NoDistImEtProf2Terrain(const Pt2dr&, double) const { return Pt3dr(0,0,0); }                 // __NEW
        double SzDiffFinie()                                const { return 0; }                            // __NEW
        double ResolutionSol()                              const { return 0; }                            // __NEW
        double ResolutionSol(const Pt3d<double>&)           const { return 0; }                            // __NEW
        void   InstanceModifParam(NS_ParamChantierPhotogram::cCalibrationInternConique&) const { return; } // __NEW
};

typedef EpipolaireCoordinate aTypEpil;

void BenchWriteReadCpleEp(CpleEpipolaireCoord & aCpl1)
{
    {
	ELISE_fp aFile(ELISE_BFI_DATA_DIR "Epip", ELISE_fp::WRITE);
	aFile.write(aCpl1);
	aFile.close();
    }

     ELISE_fp aFile(ELISE_BFI_DATA_DIR "Epip", ELISE_fp::READ);
     CpleEpipolaireCoord   & aCpl2 = *aFile.read((CpleEpipolaireCoord *)0);
     aFile.close();

     REAL aChs = 0.2 + 3* NRrandom3();
     CpleEpipolaireCoord * aCplChs = aCpl1.MapingChScale(aChs);

     for (INT aK =0 ; aK< 100 ; aK++)
     {
          Pt2dr aP = P2Rand();
	  REAL aParx = NRrandC();

	  Pt2dr aQ1 = aCpl1.Hom12(aP,aParx);
	  Pt2dr aQ2 = aCpl2.Hom12(aP,aParx);
	  REAL d12 = euclid(aQ1,aQ2);

	  BENCH_ASSERT(d12<epsilon);

	  aQ1 = aCpl1.Hom21(aP,aParx);
	  aQ2 = aCpl2.Hom21(aP,aParx);
	  d12 = euclid(aQ1,aQ2);
	  BENCH_ASSERT(d12<epsilon);

	  Pt2dr aPCh1 = aCplChs->Hom12(aP*aChs,aParx*aChs);
	  Pt2dr aPCh2 = aCpl1.Hom12(aP,aParx) * aChs;

	  d12 = euclid(aPCh1,aPCh2) ;
	  BENCH_ASSERT(d12<BIG_epsilon);

     }
     delete & aCpl2;
     delete  aCplChs;
}


static Box2dr  BoxRand()
{

    return Box2dr
           (
               Pt2dr(1,1) + Pt2dr(NRrandom3(),NRrandom3()),
               Pt2dr(-1,-1) - Pt2dr(NRrandom3(),NRrandom3())
           );
}

void BenchEpipole0
     (
           ElCamera & aCam1,
           ElCamera & aCam2,
           bool       aCamAreAffine,
           INT        aDegre,
           INT        aNbPts,
           bool       aEpiIsOnX,
           REAL       anEcartMax,
	   const char *     mes
     )
{

// cout <<  "ECART " << mes << " " << anEcartMax << "\n";

   ElPackHomologue aLC = ListeCplH(aNbPts,aCam1,aCam2);


   CpleEpipolaireCoord & aCple = *CpleEpipolaireCoord::PolynomialFromHomologue
                               (
                                     aLC,
                                     aDegre,
                                     aEpiIsOnX ? Pt2dr(1,0) :  Pt2dr(1,NRrandC()/2) ,
                                     aEpiIsOnX ? Pt2dr(1,0) :  Pt2dr(1,NRrandC()/2) 
                               );


    if (!aCamAreAffine)
        aCple.AdjustTr2Boxes(BoxRand(),BoxRand());
   

   BenchWriteReadCpleEp(aCple);

    aTypEpil & anEPI1 = aCple.EPI1();
    aTypEpil & anEPI2 = aCple.EPI2();



    if (aCamAreAffine)
    {
       Pt2dr aP02 = anEPI2.P0() ;
       Pt2dr aH01 = aCple.Hom12(anEPI1.P0(),0) ;

       Pt2dr aP01 = anEPI1.P0() ;
       Pt2dr aH02 = aCple.Hom21(anEPI2.P0(),0) ;

       BENCH_ASSERT(euclid(aP02,aH01)<epsilon);
       BENCH_ASSERT(euclid(aP01,aH02)<epsilon);
    }


    // On verifie qu'un point 3D se projette en deux
    // point correspondant a la meme ligne epipolaire
    for (INT k= 0 ; k< 10 ; k++)
    {
        Pt3dr  aP3 = P3Rand();

        Pt2dr  aQ1 = aCam1.R3toF2(aP3);
        Pt2dr  aQ2 = aCam2.R3toF2(aP3);

        Pt2dr anEpi1 = anEPI1.Direct(aQ1);
        Pt2dr anEpi2 = anEPI2.Direct(aQ2);

        REAL  aDifEpi = ElAbs(anEpi1.y - anEpi2.y);
        BENCH_ASSERT(aDifEpi<anEcartMax);
    }


    // On verifie que les fonction Direct et Inverse,
    // sont bien reciproques l'une de l'autre
	{
    for (INT k= 0 ; k< 10 ; k++)
    {
        Pt2dr aP = Pt2dr(NRrandC(),NRrandC()) / 3.0;
        aTypEpil & E = (k&1) ? anEPI1 : anEPI2;

        Pt2dr aQ = E.Inverse(E.Direct(aP));
        Pt2dr aR = E.Direct(E.Inverse(aP));

        BENCH_ASSERT(euclid(aP,aQ) < anEcartMax);
        BENCH_ASSERT(euclid(aP,aR) < anEcartMax);

    }
	}



    // On verifie que pour un point d'une image,
    // un homologue qq sur l'autre image, les rayons 
    // s'intersectent 
	{
    for (INT k= 0 ; k< 10 ; k++)
    {
        Pt2dr aP1 = Pt2dr(NRrandC(),NRrandC()) / 3.0;
        REAL aParX = NRrandC() / 3.0;


        ElCamera & C1 = (k&1) ? aCam1 : aCam2;
        ElCamera & C2 = (k&1) ? aCam2 : aCam1;
        aTypEpil & E = (k&1) ? anEPI1 : anEPI2;

        Pt2dr aH2 = (k&1) ? aCple.Hom12(aP1,aParX) : aCple.Hom21(aP1,aParX);

        // La Propriete doit etre invariante par translation
        // sur une epipolaire

        aP1 = E.TransOnLineEpip(aP1,NRrandC()/6.0);



        ElSeg3D aS1 = C1.F2toRayonR3(aP1);
        ElSeg3D aS2 = C2.F2toRayonR3(aH2);


        Pt3dr aI = aS1.PseudoInter(aS2);

        REAL aD1 = aS1.DistDoite(aI);
        REAL aD2 = aS2.DistDoite(aI);

        BENCH_ASSERT(aD1 < anEcartMax);
        BENCH_ASSERT(aD2 < anEcartMax);
    }
	}


   delete & aCple;
}


void BenchCalcEpipoleAutom()
{
    Pt2dr aDir1 = P2Rand();
    Pt2dr aDir2 = P2Rand();
    Pt2dr aTr1 = P2Rand();
    Pt2dr aTr2 = P2Rand();
    Pt2dr aSc1 = P2Rand();
    Pt2dr aSc2 = P2Rand();


    BenchCamCavSim aCam1(aDir1,aTr1,aSc1);
    BenchCamCavSim aCam2(aDir2,aTr2,aSc2);

    ElPackHomologue  aPack;

    for (INT aK =0 ; aK < 200 ; aK++)
    {
        Pt3dr aP3(NRrandC() * 10,NRrandC() * 10,NRrandC() * 10);
        Pt2dr aQ1 = aCam1.R3toF2(aP3);
        Pt2dr aQ2 = aCam2.R3toF2(aP3);

   
        //aPack.add(ElCplePtsHomologues(aQ1,aQ2,0.1+NRrandom3()));
        aPack.Cple_Add(ElCplePtsHomologues(aQ1,aQ2,0.1+NRrandom3())); // __NEW
    }

     Pt2dr anEpi1,anEpi2;
     aPack.DirEpipolaire(anEpi1,anEpi2,50,6,1);


    for (INT aK = 0 ; aK <5; aK++)
    {
        Pt3dr aP3(NRrandC() * 10,NRrandC() * 10,NRrandC() * 10);
        Pt2dr aQ1 = aCam1.R3toF2(aP3);
        Pt2dr aQ2 = aCam2.R3toF2(aP3);

        ElSeg3D  aR1 = aCam1.F2toRayonR3(aQ1);
        ElSeg3D  aR2 = aCam2.F2toRayonR3(aQ2);

	Pt2dr A1 =  aCam1.R3toF2(aR2.P0());
	Pt2dr B1 =  aCam1.R3toF2(aR2.P1());



	Pt2dr A2 =  aCam2.R3toF2(aR1.P0());
	Pt2dr B2 =  aCam2.R3toF2(aR1.P1());

	Pt2dr Ok1 =  (A1-B1) / anEpi1 ;
	Pt2dr Ok2 =  (A2-B2) / anEpi2 ;
	BENCH_ASSERT(ElAbs(Ok1.y)<epsilon);
	BENCH_ASSERT(ElAbs(Ok2.y)<epsilon);
    }



}



void BenchEpipole0()
{
    //  Cameras cavaliere avec Dist en X et Y

    {
          Pt2dr aCDist = Pt2dr(NRrandC(),NRrandC()) / 10.0;
          ElDistRadiale_Pol357 aDistRad
                               (
                                   1e5,
                                   aCDist,
                                   NRrandC() * 1e-2,
                                   NRrandC() * 1e-3,
                                   NRrandC() * 1e-3
                               );

           REAL aFocale = 1.0 + NRrandC() * 1e-2;

          
          //cCamStenopeDistRadPol aCam1 (aFocale,Pt2dr(0,0),aDistRad);
          //cCamStenopeDistRadPol aCam2 (aFocale,Pt2dr(0,0),aDistRad);
	  cCamStenopeDistRadPol aCam1( false/*isDistC2M*/, aFocale, Pt2dr(0,0), aDistRad, vector<double>() ); // __NEW
	  cCamStenopeDistRadPol aCam2( false/*isDistC2M*/, aFocale, Pt2dr(0,0), aDistRad, vector<double>() ); // __NEW
 
          aCam1.SetOrientation
          (
                 ElRotation3D
                 (
                     Pt3dr(0.5+NRrandC()*1e-2,NRrandC()*1e-1,5.0),
                     RandTeta(),RandTeta(),RandTeta()
                 )
          );
          aCam2.SetOrientation
          (
                 ElRotation3D
                 (
                     Pt3dr(-0.5+NRrandC()*1e-2,NRrandC()*1e-1,5.0),
                     RandTeta(),RandTeta(),RandTeta()
                 )
          );
          // aCam1.SetOrientation(ElRotation3D(Pt3dr(1,0,10),0.0,0.0,0.0));
          // aCam2.SetOrientation(ElRotation3D(Pt3dr(-1,0,10),0.0,0.0,0.0));

          BenchEpipole0(aCam1,aCam2,false,4,1000,true,1e-2,"AAA");
    }

    // Camera sans distortion, cavaliere pure
    {
        BenchCamCav aCam1(Pt2dr(0,0));
        BenchCamCav aCam2(Pt2dr(1,0));

        BenchEpipole0(aCam1,aCam2,true,2,1000,true,epsilon,"BBB");
        BenchEpipole0(aCam1,aCam2,true,2,1000,false,epsilon,"CCC");
    }

    //  Cameras cavaliere avec Dist en Y
    {
         INT aDeg = 2;

         Polynome2dReal anIdx  =  Rand_Polynome2dReal(1.0,0.0,1,100);
         Polynome2dReal aDistY1 = Rand_Polynome2dReal(0.0,1.0,aDeg,1e-2);
         Polynome2dReal aDistY2 = Rand_Polynome2dReal(0.0,1.0,aDeg,1e-2);

         BenchCamCavDist aCam1(Pt2dr(0,0),anIdx,aDistY1);
         BenchCamCavDist aCam2(Pt2dr(1,1),anIdx,aDistY2);

         BenchEpipole0(aCam1,aCam2,false,4,1000,true,1e-4,"DDD");
    }

    //  Cameras cavaliere avec Dist en X et Y
    {
         INT aDeg = 2;


         Polynome2dReal aDistX1 = Rand_Polynome2dReal(1.0,0.0,aDeg,1e-2);
         Polynome2dReal aDistX2 = Rand_Polynome2dReal(1.0,0.0,aDeg,1e-2);

         Polynome2dReal aDistY1 = Rand_Polynome2dReal(0.0,1.0,aDeg,1e-2);
         Polynome2dReal aDistY2 = Rand_Polynome2dReal(0.0,1.0,aDeg,1e-2);

         BenchCamCavDist aCam1(Pt2dr(0,0),aDistX1,aDistY1);
         BenchCamCavDist aCam2(Pt2dr(1,1),aDistX2,aDistY2);

         BenchEpipole0(aCam1,aCam2,false,4,1000,true,1e-4,"EEE");
    }

}


void bench_epipole()
{
     for (INT aK=0 ; aK< 2 ; aK++)
     {
         BenchEpipole0();
     }

     for (INT aK=0 ; aK< 2 ; aK++)
         BenchCalcEpipoleAutom();
}




