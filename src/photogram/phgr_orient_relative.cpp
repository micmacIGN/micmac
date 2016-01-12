/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


#include "StdAfx.h"

/*
*/

// #include "../GC_photogram/cGCAOR_Fixe_F0PP0_BUnite_ID.cpp"

#define DEBUG_FONC 1


cFonctrPond::cFonctrPond(cElCompiledFonc * aFctr,REAL aPds) :
   mPds  (aPds),
   mFctr (aFctr)
{
}

/************************************************************/
/*                                                          */
/*               cIncEnsembleCamera                         */
/*                                                          */
/************************************************************/


cIncEnsembleCamera::cIncEnsembleCamera(bool DerSec) :
    mL2Opt   (true),
    mDerSec  (DerSec),
    mAlloc   (),
    mSysL2   (0),
    mSysL1   (0),
    mSys     (0),
    mIPIs    (),
    mIPEs    (),
    mImState0 (1),
    mImDir    (1),
    mMatrL2   (1,1),
    mMatrtB    (1,1),
    mMatrValP (1,1),
    mMatrVecP (1,1),
    mtBVecP   (1,1)
{
}

void cIncEnsembleCamera::VerifFiged()
{
    ELISE_ASSERT(mSys==0,"Illegal modif in cIncEnsembleCamera");
}

void cIncEnsembleCamera::SetL2Opt(bool L2Opt)
{
   mL2Opt = L2Opt;
}


void cIncEnsembleCamera::SetOpt()
{
    if (mL2Opt)
    {
       if (mSysL2==0)
          mSysL2 =  new L2SysSurResol(mAlloc.CurInc());
       mSys = mSysL2;
    }
    else
    {
       if (mSysL1==0)
          mSysL1 =  new SystLinSurResolu(mAlloc.CurInc(),1000);
       mSys = mSysL1;
    }
    BENCH_ASSERT(mSys->NbVar() == mAlloc.CurInc());
}

cIncParamIntrinseque  *  cIncEnsembleCamera::NewParamIntrinseque
                         (
                                      REAL     aFocale,
                                      bool     isFocaleFree,
                                      Pt2dr    aPP,
                                      bool     isPPFree,
                                      ElDistRadiale_PolynImpair * aDR
                         )
{
   VerifFiged();

   cIncParamIntrinseque * aRes 
        =  cIncParamIntrinseque::NewOneNoDist(mAlloc,aFocale,isFocaleFree,aPP,isPPFree,this,aDR);
   mIPIs.push_back(aRes);

  std::vector<cElCompiledFonc *> & aV = aRes->FoncteursAuxiliaires();
  for (INT aK=0; aK<INT(aV.size()) ; aK++)
      mLFoncteurs.push_back(aV[aK]);

   return aRes;
}

cIncParamExtrinseque * cIncEnsembleCamera::AddIPE(cIncParamExtrinseque * aParam)
{
    mIPEs.push_back(aParam);
    return aParam;
}

cIncParamExtrinseque *   cIncEnsembleCamera::NewParamExtrinsequeRigide(ElRotation3D aRInit)
{
   VerifFiged();
   return AddIPE (cIncParamExtrinseque::IPEFixe(mAlloc,aRInit,this));
}
cIncParamExtrinseque *   cIncEnsembleCamera::NewParamExtrinsequeLibre(ElRotation3D aRInit)
{
   VerifFiged();
   return AddIPE (cIncParamExtrinseque::IPELibre(mAlloc,aRInit,this));
}
cIncParamExtrinseque *   cIncEnsembleCamera::NewParamExtrinsequeBaseUnite
                         (Pt3dr aCRot,ElRotation3D aRInit)
{
   VerifFiged();
   return AddIPE (cIncParamExtrinseque::IPEBaseUnite(mAlloc,aCRot,aRInit,this));
}




cIncParamCamera * cIncEnsembleCamera::NewParamCamera
                  (
                           cIncParamIntrinseque &  anIPI,
                           cIncParamExtrinseque &  anIPE
                  )
{
   VerifFiged();
   cIncParamCamera * aRes = new cIncParamCamera(anIPI,anIPE,this);
   mCams.push_back(aRes);
   return aRes;
}

cIncCpleCamera * cIncEnsembleCamera::NewCpleCamera
                 (
                        cIncParamCamera & aCam1,
                        cIncParamCamera & aCam2
                 )
{
   // VerifFiged();
   cIncCpleCamera * aRes = new cIncCpleCamera(aCam1,aCam2,this);
   mCples.push_back(aRes);
   mLFoncteurs.push_back(aRes->Fonc());
   return aRes;
}


REAL cIncEnsembleCamera::AddEqCoPlan
     (
           cIncCpleCamera & aCple,
           const Pt2dr & aP1,
           const Pt2dr & aP2,
           REAL aPds
     )
{
    ELISE_ASSERT(aCple.Ensemble()==this,"cIncEnsembleCamera::AddEqCoPlan");
    SetOpt();
    if (mDerSec)
    {
       ELISE_ASSERT(mL2Opt,"Need L2 Sys with Der Sec in cIncEnsembleCamera");
       return aCple.Dev2AddEqCoPlan(aP1,aP2,aPds,*mSysL2);
    }
    else
       return aCple.DevL1AddEqCoPlan(aP1,aP2,aPds,*mSys);
}


void cIncEnsembleCamera::StdAddEq(cElCompiledFonc * pFctr,REAL aP)
{
   ELISE_ASSERT(false,"cIncEnsembleCamera::StdAddEq obsolete");
}

void cIncEnsembleCamera::AddEqRappelCentreDR
     (
          cIncParamIntrinseque & aParamI,
          REAL aPds
     )
{
    ELISE_ASSERT(aParamI.Ensemble()==this,"cIncEnsembleCamera::AddEqRappelCentreDR");
    SetOpt();
    aParamI.CurSetRappelCrd();
    std::vector<cElCompiledFonc *> & aF = aParamI.FoncteurRappelCentreDist();

    for (INT aK=0 ; aK<INT(aF.size()) ; aK++)
        StdAddEq( aF[aK],aPds);
}
void cIncEnsembleCamera::ResetEquation()
{
   if (mSys)
      mSys->GSSR_Reset(true);
}

void cIncEnsembleCamera::ItereLineaire()
{
     SetOpt();
     Im1D_REAL8  anIm =  mSys->GSSR_Solve((bool *)0);

      for (INT aK=0 ; aK < NbVal(); aK++)
           anIm.data()[aK] += mAlloc.GetVar(aK);

      SetPtCur(anIm.data());

/*
         mAlloc.SetVar(mAlloc.GetVar(aK)+anIm.data()[aK],aK);

     // Met a jour les foncteur pour une nouvelle iteration
     for (tContFcteur::iterator itF=mLFoncteurs.begin(); itF!=mLFoncteurs.end() ; itF++)
     {
         (*itF)->SetCoordCur(mAlloc.ValsVar());
     }
*/
}

template <class tCont> void DeleteListPtr(tCont & aCont)
{
   for (typename tCont::iterator anIt=aCont.begin() ; anIt!=aCont.end() ; anIt++)
   {
       delete (*anIt);
   }
}

cIncEnsembleCamera::~cIncEnsembleCamera()
{
   delete mSysL2;
   delete mSysL1;

   DeleteListPtr(mIPIs);
   DeleteListPtr(mIPEs);
   DeleteListPtr(mCams);
   DeleteListPtr(mCples);
}



/************************************************************/
/*                                                          */
/*               cIncParamCamera                            */
/*                                                          */
/************************************************************/


cIncParamCamera::cIncParamCamera
(
    cIncParamIntrinseque &  anIPI,
    cIncParamExtrinseque &  anIPE,
    cIncEnsembleCamera  *   apEns
)  :
   mIPI  (anIPI),
   mIPE  (anIPE),
   mpEns (apEns)
{
   ELISE_ASSERT
   (
           (&(mIPI.Alloc()) == & (mIPE.Alloc())) 
        && (mIPI.Ensemble() == mIPE.Ensemble()) 
        && (mpEns == mIPI.Ensemble()),
        "Different Allocator or Ensemble in cIncParamCamera::cIncParamCamera"
   );
}

cIncParamIntrinseque & cIncParamCamera::ParamIntr()
{
   return mIPI;
}
cIncParamExtrinseque & cIncParamCamera::ParamExtr()
{
   return mIPE;
}

AllocateurDInconnues & cIncParamCamera::Alloc()
{
   return mIPI.Alloc();
}

cIncEnsembleCamera * cIncParamCamera::Ensemble()
{
   return mpEns;
}


bool cIncParamCamera::SameIntrinseque(const cIncParamCamera & aCam) const
{
    return mIPI == aCam.mIPI;
}

cIncParamExtrinseque::tPosition cIncParamCamera::TPos () const
{
   return mIPE.TPos();
}

Pt3d<Fonc_Num>  cIncParamCamera::DirRayon(Pt2d<Fonc_Num> aPCam,INT aNumParamI,INT aNumParamE)
{
    return mIPE.Omega(aNumParamE) * mIPI.DirRayon(aPCam,aNumParamI);
}




Pt3d<Fonc_Num> cIncParamCamera::VecteurBase(INT aNum1,cIncParamCamera & aCam,INT aNum2)
{
    return aCam.mIPE.Tr(aNum2) - mIPE.Tr(aNum1);
}


const cIncIntervale & cIncParamCamera::III() const
{
   return mIPI.IntervInc();
}

const cIncIntervale & cIncParamCamera::IIE() const
{
   return mIPE.IntervInc();
}

cIncIntervale & cIncParamCamera::III() 
{
   return mIPI.IntervInc();
}

cIncIntervale & cIncParamCamera::IIE() 
{
   return mIPE.IntervInc();
}

std::string cIncParamCamera::NameType (bool SameIntr)
{
   return  mIPE.NameType()
          + std::string("_")
          + (SameIntr ? "ID" : mIPI.NameType());
}

void cIncParamCamera::InitFoncteur(cElCompiledFonc & aFoncteur,INT aNumI,INT aNumE)
{
   mIPI.InitFoncteur(aFoncteur,aNumI);
   mIPE.InitFoncteur(aFoncteur,aNumE);
}


/************************************************************/
/*                                                          */
/*               cIncSetLiaison                             */
/*                                                          */
/************************************************************/

cIncSetLiaison::cIncSetLiaison(cIncCpleCamera * aCple) :
   mCple (aCple)
{
}

void cIncSetLiaison::AddCple(Pt2dr aP1,Pt2dr aP2,REAL aPds)
{
    ElCplePtsHomologues aCplH  (aP1,aP2,aPds);
    mSetCplH.Cple_Add(aCplH);
}

cIncCpleCamera * cIncSetLiaison::Cple()
{
   return mCple;
}

ElPackHomologue & cIncSetLiaison::Pack() {return mSetCplH;}



/************************************************************/
/*                                                          */
/*               cIncCpleCamera                             */
/*                                                          */
/************************************************************/

cIncCpleCamera::~cIncCpleCamera()
{
   if ((! ElBugHomeMPD) || (! mWithDynFCT))
      delete mFoncteur;
}

extern cSetEqFormelles*  PtrNOSET();

cIncCpleCamera::cIncCpleCamera
(
    cIncParamCamera &     aCam1,
    cIncParamCamera &     aCam2,
    cIncEnsembleCamera  * apEns
) :
     mOrdInit   (aCam1.TPos() <= aCam2.TPos()),
     mCam1      (mOrdInit ? aCam1 : aCam2),
     mCam2      (mOrdInit ? aCam2 : aCam1),
     mpEns      (apEns),

     mMemberX1  ("XL1"),  
     mMemberX2  ("XL2"),  
     mMemberY1  ("YL1"),  
     mMemberY2  ("YL2"),  

     mParamX1  (mOrdInit ? mMemberX1 : mMemberX2),
     mParamX2  (mOrdInit ? mMemberX2 : mMemberX1),
     mParamY1  (mOrdInit ? mMemberY1 : mMemberY2),
     mParamY2  (mOrdInit ? mMemberY2 : mMemberY1),


     mSameIntr  (mCam1.SameIntrinseque(mCam2) ),
     mNumIntr2  ( mSameIntr  ? 0 : 1),
     mNumExtr2  (1),
     mP1        (cVarSpec(0,mMemberX1),cVarSpec(0,mMemberY1)),
     mP2        (cVarSpec(0,mMemberX2),cVarSpec(0,mMemberY2)),
     mDRay1     (mCam1.DirRayon(mP1,0,0)),
     mDRay2     (mCam2.DirRayon(mP2,mNumIntr2,mNumExtr2)),
     mEqCoplan  (scal(mCam1.VecteurBase(0,mCam2,mNumExtr2),mDRay1^mDRay2)),
     mLInterv   (),
     mFoncteur  (0)
{

    ELISE_ASSERT
    (
          (&(mCam1.Alloc()) == &(mCam2.Alloc())) 
       && (mCam1.Ensemble() == mCam2.Ensemble())
       && (mCam1.Ensemble()==apEns),
       "Different Alloc or Ensemble in cIncCpleCamera::cIncCpleCamera"
    );

    mCam1.IIE().SetName("Extr1");
    mLInterv.AddInterv( mCam1.IIE());

    mCam1.III().SetName("Intr1"); 
    mLInterv.AddInterv( mCam1.III());

    mCam2.IIE().SetName("Extr2");
    mLInterv.AddInterv( mCam2.IIE());

    if (! mSameIntr)
    {
       mCam2.III().SetName("Intr2"); 
       mLInterv.AddInterv( mCam2.III());
    }

    mFoncteur = cElCompiledFonc::AllocFromName(NameType());
    cout << "Foncteur Comp = " << (void *) mFoncteur <<  " Name=[" << NameType() << "]\n";
    mWithDynFCT = (mFoncteur == 0);
    if (mWithDynFCT)
    {
        mFoncteur =  cElCompiledFonc::DynamicAlloc(mLInterv,mEqCoplan);
    }


    mFoncteur->SetMappingCur(mLInterv,PtrNOSET());
    mAdrX1 = mFoncteur->RequireAdrVarLocFromString(mParamX1);
    mAdrY1 = mFoncteur->RequireAdrVarLocFromString(mParamY1);
    mAdrX2 = mFoncteur->RequireAdrVarLocFromString(mParamX2);
    mAdrY2 = mFoncteur->RequireAdrVarLocFromString(mParamY2);

    mCam1.InitFoncteur(*mFoncteur,0,0);
    mCam2.InitFoncteur(*mFoncteur,mNumIntr2,mNumExtr2);


    mDebugFonc = 0;
    if (DEBUG_FONC)
    {
       mDebugFonc  = cElCompiledFonc::DynamicAlloc(mLInterv,mEqCoplan);
       mDebugFonc->SetMappingCur(mLInterv,PtrNOSET());
       mDebugAdrX1 = mDebugFonc->RequireAdrVarLocFromString(mParamX1);
       mDebugAdrY1 = mDebugFonc->RequireAdrVarLocFromString(mParamY1);
       mDebugAdrX2 = mDebugFonc->RequireAdrVarLocFromString(mParamX2);
       mDebugAdrY2 = mDebugFonc->RequireAdrVarLocFromString(mParamY2);
       
       mCam1.InitFoncteur(*mDebugFonc,0,0);
       mCam2.InitFoncteur(*mDebugFonc,mNumIntr2,mNumExtr2);
    }
    
    InitCoord();
}



cElCompiledFonc * cIncCpleCamera::Fonc()
{
   return mFoncteur;
}

void cIncCpleCamera::InitCoord()
{
    mFoncteur->SetCoordCur(Alloc().ValsVar());
    if (mDebugFonc)
       mDebugFonc->SetCoordCur(Alloc().ValsVar());
}

AllocateurDInconnues & cIncCpleCamera::Alloc()
{
   return mCam1.Alloc();
}

cIncEnsembleCamera * cIncCpleCamera::Ensemble()
{
   return mpEns;
}

double cIncCpleCamera::DevL1AddEqCoPlan
       (
            const Pt2dr & aP1,
            const Pt2dr & aP2,
            REAL aPds,
            cGenSysSurResol & aSys
       )
{
   ELISE_ASSERT(false,"cIncCpleCamera::DevL1AddEqCoPlan obsolete");
   return 0;
}

double cIncCpleCamera::ValEqCoPlan(const Pt2dr & aP1,const Pt2dr & aP2)
{
      SetP1P2(aP1,aP2);
      mFoncteur->ComputeValAndSetIVC();
      return mFoncteur->Val(0);
}

 
double cIncCpleCamera::Dev2AddEqCoPlan
       (
            const Pt2dr & aP1,
            const Pt2dr & aP2,
            REAL aPds,
            L2SysSurResol & aSys
       )
{
    ELISE_ASSERT(false,"cIncCpleCamera::Dev2AddEqCoPlan");
    return 0;
/*
      SetP1P2(aP1,aP2);

      mFoncteur->SetValDerHess();
      mFoncteur->AddDevLimOrd2ToSysSurRes(aSys,aPds);
      return mFoncteur->Val(0);
*/
}



void cIncCpleCamera::SetP1P2(const Pt2dr & aP1,const Pt2dr & aP2)
{
   *mAdrX1 = aP1.x;
   *mAdrY1 = aP1.y;
   *mAdrX2 = aP2.x;
   *mAdrY2 = aP2.y;
   if (mDebugFonc)
   {
       *mDebugAdrX1 = aP1.x;
       *mDebugAdrY1 = aP1.y;
       *mDebugAdrX2 = aP2.x;
       *mDebugAdrY2 = aP2.y;
   }
}

std::string cIncCpleCamera::NameType ()
{
   return 
          std::string("cGCAOR_")  // class Generated Code, Amelioration Orientation Relative
        + mCam1.NameType(false)
        + std::string("_")
        + mCam2.NameType(mSameIntr);
}

void cIncCpleCamera::GenerateCode(const std::string  & aDir,const char * Name)
{
    ELISE_ASSERT(mOrdInit,"Bad Order for cIncCpleCamera::GenerateCode");
    cElCompileFN::DoEverything
    (
           aDir,
           ((Name==0) ? NameType() : std::string(Name)),
           mEqCoplan,
           mLInterv
    );
}


static ElDistRadiale_PolynImpair * DistBitOfDegre(INT aDegre)
{
  if (aDegre <=0)
     return 0;
   ElDistRadiale_PolynImpair * aRes = new  ElDistRadiale_PolynImpair(1.0,Pt2dr(0,0));
   for (INT aD =0 ; aD< aDegre ; aD++)
	   aRes->PushCoeff(0);
   return aRes;
}

void cIncCpleCamera::GenerateAllCodeGen
     (
                const std::string & aDir,
                cIncParamExtrinseque::tPosition aPos1,
                bool  aFocFree1,
                bool  aPPFree1,
		INT   aDegreDR1,
                cIncParamExtrinseque::tPosition aPos2,
                bool aSameIntr,
                bool  aFocFree2,
                bool  aPPFree2,
		INT   aDegreDR2,
                const char * aName
     )
{
    AllocateurDInconnues anAlloc;
    cIncEnsembleCamera * aPens = 0;
    ElDistRadiale_PolynImpair *  pDR1 = DistBitOfDegre(aDegreDR1);
    ElDistRadiale_PolynImpair *  pDR2 = DistBitOfDegre(aDegreDR2);
    cIncParamIntrinseque anIntr1(anAlloc,1.0,aFocFree1,Pt2dr(0,0),aPPFree1,aPens,pDR1);
    cIncParamIntrinseque anIntr2(anAlloc,1.0,aFocFree2,Pt2dr(0,0),aPPFree2,aPens,pDR2);

    ElRotation3D aRot(Pt3dr(1,0,0),0,0,0);

    cIncParamExtrinseque * anOR1  = cIncParamExtrinseque::Alloc(aPos1,anAlloc,aRot);
    cIncParamExtrinseque * anOR2  = cIncParamExtrinseque::Alloc(aPos2,anAlloc,aRot);


     cIncParamCamera  aCam1(anIntr1,*anOR1);
     cIncParamCamera  aCam2(aSameIntr ? anIntr1 : anIntr2 ,*anOR2);

     cIncCpleCamera  aCple(aCam1,aCam2);

     aCple.GenerateCode(aDir,aName);

     delete anOR1;
     delete anOR2;
}

void cIncCpleCamera::GenerateAllCodeSameIntr
     (
                           const std::string & aDir,
                           cIncParamExtrinseque::tPosition aPos1,
                           bool  aFocFree1,
                           bool  aPPFree1,
			   INT   aDegreDR1,
                           cIncParamExtrinseque::tPosition aPos2,
                           const char * aName
     )
{
      GenerateAllCodeGen
      (
            aDir,aPos1,aFocFree1,aPPFree1,aDegreDR1,aPos2,
            true,false,false,0,aName
      );
}


void cIncCpleCamera::GenerateAllCode()
{
     /*
        GenerateAllCodeSameIntr
        (
            "src/photogram/",
            cIncParamExtrinseque::ePosFixe,
            true,
            true,
            cIncParamExtrinseque::ePosBaseUnite
        );
     */
     GenerateAllCodeSameIntr
     (
           "src/photogram/",
           cIncParamExtrinseque::ePosFixe,
           false,
           false,
	   0,
           cIncParamExtrinseque::ePosBaseUnite,
           "cAmeliorMEPRel"
     );
/*
     for (INT FF =1 ; FF >=0 ; FF--)
         for (INT PPF =1 ; PPF >=0 ; PPF--)
             for (INT  P1=0 ; P1< 3 ; P1++)
                 for (INT  P2=P1 ; P2< 3 ; P2++)
                 {
                     for (INT aDegre = 0 ; aDegre < 5 ; aDegre++)
                     {
                        // if ((aDegre==0) || (aDegre>=3))
                            GenerateAllCodeSameIntr
                            (
                               "src/GC_photogram/",
                               cIncParamExtrinseque::tPosition(P1),
                               FF,
                               PPF,
			       aDegre,
                               cIncParamExtrinseque::tPosition(P2)
                            );
		     }
                 }
*/
}

void TestPH()
{
    cIncCpleCamera::GenerateAllCode();
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
