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


/*
   Sur le long terme, modifier :

     - la possiblite d'avoir des fonctions multiple (Dim N)
     - associer a un fctr, un cSetEqFormelles
*/

#include "StdAfx.h"
#include <iterator>


extern bool DebugCamBil;


/************************************************************/
/*                                                          */
/*                    cNameSpaceEqF                         */
/*                                                          */
/************************************************************/

std::string cNameSpaceEqF::TheNameEcCoplan = "cEqCoplan";
std::string cNameSpaceEqF::TheNameResiduIm1 = "cEqResiduIm1";
std::string cNameSpaceEqF::TheNameResiduIm2 = "cEqResiduIm2";

std::string & cNameSpaceEqF::NameOfTyRes(eModeResidu aMode)
{
     switch(aMode)
     {
         case eResiduCoplan : return TheNameEcCoplan;
         case eResiduIm1 : return TheNameResiduIm1;
         case eResiduIm2 : return TheNameResiduIm2;
     }
     ELISE_ASSERT(false,"Bad cNameSpaceEqF::NameOfTyRes");
     return TheNameEcCoplan;
}



/************************************************************/
/*                                                          */
/*                    cObjFormel2Destroy                    */
/*                                                          */
/************************************************************/

cObjFormel2Destroy::~cObjFormel2Destroy() 
{
}

void cObjFormel2Destroy::Update_0F2D()
{
}

/************************************************************/
/*                                                          */
/*              cDicoHomogSEF, cDicoCameraSEF               */
/*                                                          */
/************************************************************/

class cDicoHomogSEF : public map<std::string,cHomogFormelle *>
{
};

class cDicoCameraSEF : public map<std::string,cCameraFormelle *>
{
};

class cDicoRotSEF : public map<std::string,cRotationFormelle *>
{
};

class cDicoEqCorrGrid : public map<std::string,cEqCorrelGrid *>
{
};

/************************************************************/
/*                                                          */
/*                    cElemEqFormelle                       */
/*                                                          */
/************************************************************/

cElemEqFormelle::cElemEqFormelle(cSetEqFormelles & aSet,bool isTmp) :
     mSet         (aSet),
     mNumInc0     (aSet.Alloc().CurInc()),
     mIncInterv   (isTmp,"toto",aSet),
     mClosed      (false)
{
}

void cSetEqFormelles::VerifC2MForPIF(bool isDC2M,CamStenope * aCam)
{
   ELISE_ASSERT
   (
       isDC2M== aCam->DistIsC2M(),
       "Incoherence  C2M/M2C dans creation PIF"
   );
}

cElemEqFormelle::~cElemEqFormelle()
{
}

void cSetEqFormelles::SetPhaseEquation()
{

   mSys->SetPhaseEquation(&(mMOI.Alloc2Solve())); 
   for (int aKB=0 ; aKB<int(mBlocsIncAlloc.size()) ; aKB++)
   {
       bool AllFiged = true;
       for (int anI =mBlocsIncAlloc[aKB]->I0Alloc() ; anI<mBlocsIncAlloc[aKB]->I1Alloc() ; anI++)
       {
            double aV;
            if (! mSys->IsCstrUniv(anI,aV))
            {
               AllFiged = false;
            }

       }
       mBlocsIncAlloc[aKB]->SetFiged(AllFiged);
       // std::cout << "FIG-BL[" << aKB << "]=" << AllFiged << "\n";
   }

}

cSetEqFormelles * cElemEqFormelle::Set() 
{
    return &mSet;
}



const  cIncIntervale & cElemEqFormelle::IncInterv() const
{
    return mIncInterv;
}
cIncIntervale & cElemEqFormelle::IncInterv()
{
    return mIncInterv;
}
void cElemEqFormelle::CloseEEF(bool asIntervBlock)
{
   AssertUnClosed();
   mIncInterv.Close();
   mNumIncN = mSet.Alloc().CurInc();


   for (INT kInc=mNumInc0 ; kInc<mNumIncN ; kInc++)
   {
      mFoncRap.push_back(cElCompiledFonc::FoncSetVar(&mSet,kInc));
      mAllFonct.push_back(mFoncRap.back());
      mAdrFR.push_back( mFoncRap.back()->FoncSetVarAdr());
      mValsInit.push_back(mSet.Alloc().GetVar(kInc));

   }
   mSet.AddElem(*this);
   mClosed = true;

   if (asIntervBlock)
      mSet.AddABlocAlloc(&mIncInterv);

   if (DebugCamBil) // (false && asIntervBlock)
   {
      std::cout << asIntervBlock << " CLOSE-EEF " << mNumInc0 << " " << mNumIncN <<  " " << mIncInterv.Id() << "\n";
      getchar();
   }

   Virtual_CloseEEF();
}

void  cElemEqFormelle::Virtual_CloseEEF()
{
}

INT cElemEqFormelle::NbInc()
{
   return mNumIncN-mNumInc0;
}

void cElemEqFormelle::AddFoncteurEEF(cElCompiledFonc * aFctr)
{
    AssertUnClosed();
    if (aFctr != 0)
        mAllFonct.push_back(aFctr);
}

void cElemEqFormelle::ReinitOnCur()
{
     mSet.Reinit(mNumInc0,mNumIncN);

     mValsInit.clear();
     for (INT kInc=mNumInc0 ; kInc<mNumIncN ; kInc++)
         mValsInit.push_back(mSet.Alloc().GetVar(kInc));
}



cElemEqFormelle::tContFcteur  cElemEqFormelle::FoncRapp(INT i0,INT i1,const double * vals)
{
   AssertClosed();
   mSet.AssertClosed();
   tContFcteur aRes;
   for (INT i=i0; i<i1 ; i++)
   {
      *mAdrFR[i]  = vals[i];
      mFoncRap[i]->SetCoordCur(mSet.Alloc().ValsVar());
      aRes.push_back( mFoncRap[i]);
   }

   return aRes;
}

void cElemEqFormelle::SetValInitOnValCur()
{
     for (INT aK = mNumInc0; aK<mNumIncN ; aK++)
     {
         mValsInit[aK-mNumInc0] = mSet.Alloc().GetVar(aK);
     }
}




void cElemEqFormelle::SetValAndInit(REAL aVal,INT anIndGlob)
{
   mSet.Alloc().SetVar(aVal,anIndGlob);
   mValsInit[anIndGlob-mNumInc0] = aVal;
}


cElemEqFormelle::tContFcteur  cElemEqFormelle::FoncRappInit(INT i0,INT i1)
{
   return FoncRapp(i0,i1,&(mValsInit[0]));
}

void cElemEqFormelle::AddFoncRappInit
     (
         cMultiContEQF & aPush,
         INT i0,
         INT i1,
         double aTol,
         std::vector<double>* aVals
     )
{
    if (aVals)
    {
       ELISE_ASSERT(int(aVals->size())==(i1-i0),"AddFoncRappInit size vals incoherent ");
    }
    tContFcteur aPlus =  aVals ? FoncRapp(i0,i1,&((*aVals)[0])-i0)  :  FoncRappInit(i0,i1);
    for (INT aK=0; aK<INT(aPlus.size()) ; aK++)
         aPush.AddAcontrainte(aPlus[aK],aTol);
}

//#### CEST LA
REAL cElemEqFormelle::AddRappViscosite
     (
          const std::string  & aContexte,
          bool OnCur,
          int aK,
          double   aTol,
          bool     AdEq
     )
{
    if ((aK<0) || (aK>= NbInc()))
    {
        std::cout << "In Contexte " << aContexte << "\n";
        std::cout << "K= " << aK << " INTERV= [0," <<  NbInc()<< "]\n";
        ELISE_ASSERT(false,"Bad indexe for cElemEqFormelle::AddRappViscosite");
    }
    double aVCur = OnCur ? mSet.Alloc().GetVar(aK+mNumInc0) : mValsInit[aK];
    tContFcteur aPlus =  FoncRapp(aK,aK+1,&aVCur-aK);

    double aPds = 1/ElSquare(aTol);
    double  aRes =  AdEq                                        ?
                    mSet.AddEqFonctToSys(aPlus[0],aPds,false) :
                    mSet.ResiduSigne(aPlus[0])                ;
    return aPds* ElSquare(aRes);
}


cElemEqFormelle::tContFcteur & cElemEqFormelle::AllFonct()
{
   return mAllFonct;
}


void cElemEqFormelle::AssertClosed()
{
   ELISE_ASSERT(mClosed,"Ilegal Op on UnClosed cSetEqFormelles");
}
void cElemEqFormelle::AssertUnClosed()
{
   ELISE_ASSERT(! mClosed,"Ilegal Op on Closed cSetEqFormelles");
}

void cElemEqFormelle::AssertSameSet(const cElemEqFormelle & anEl2)
{
   ELISE_ASSERT(&mSet == &(anEl2.mSet),"Diff Set in cElemEqFormelle::AssertSameSet");
}

/************************************************************/
/*                                                          */
/*                    cSetEqFormelles                       */
/*                                                          */
/************************************************************/

cSetEqFormelles::cSetEqFormelles(eTypeSysResol aType,int aNbEq,bool CanUseCstr) :
   mSys          (0),
   mFQC          (0),
   mClosed       (false),
   mDicHom       (*(new cDicoHomogSEF)),
   mDicCam       (*(new cDicoCameraSEF)),
   mDicRot       (*(new cDicoRotSEF)),
   mDicEqCorGrid (*(new cDicoEqCorrGrid)),
   mTypeSys      (aType),
   mEstimNbEq    (aNbEq),
   mPt3dIncTmp   (0),
   mTmpBegun     (false),
   mIndIncTmp    (-1),
   mSolQuad      (1),
   mCurSol       (1),
   mCanUseCstr   (CanUseCstr)
{
}

double cSetEqFormelles::ResiduEqLineaire
       (
            double                       aPds,
            const std::vector<int>  &    aVInd,
            const std::vector<double>  & aVCoeff,
            double                       aB
       )
{
   ELISE_ASSERT(aVInd.size()==aVCoeff.size(),"cSetEqFormelles::AddEqLineaire");
   
   if (aPds <=0) return 0.0;
   double aV= -aB;
   for (int aNum=0 ; aNum<int(aVInd.size()); aNum++)
   {
       aV += aVCoeff[aNum] *  mAlloc.GetVar(aVInd[aNum]);
   }

   return aPds * ElSquare(aV);
}


double cSetEqFormelles::AddEqEqualVar(double aPds,int aK1,int aK2,bool  AddEq)
{
   std::vector<int> aVInd;
   std::vector<double> aVCoeff;
   aVInd.push_back(aK1);
   aVInd.push_back(aK2);
   aVCoeff.push_back(1);
   aVCoeff.push_back(-1);

   return AddEq                                    ?
             AddEqLineaire(aPds,aVInd,aVCoeff,0.0) :
          ResiduEqLineaire(aPds,aVInd,aVCoeff,0.0) ;
}


double cSetEqFormelles::AddEqLineaire
       (
            double                       aPds,
            const std::vector<int>  &    aVInd,
            const std::vector<double>  & aVCoeff,
            double                       aB
       )
{
   if (aPds <=0) return 0.0;
   ELISE_ASSERT(aVInd.size()==aVCoeff.size(),"cSetEqFormelles::AddEqLineaire");
   double aRes = ResiduEqLineaire(aPds,aVInd,aVCoeff,aB);

   std::vector<double>  aCoefFiltre;
   std::vector<int>     aIndFiltre;
   std::vector<cSsBloc> aVSsBl;

   for (int aNum=0 ; aNum<int(aVInd.size()); aNum++)
   {
      int anInd = aVInd[aNum];
      double  aCoeff = aVCoeff[aNum];
      double aValForced;
      aB -= aCoeff *  mAlloc.GetVar(anInd) ;

      // std::cout << "Ahjyyyyy " << aDC[anInd] << " " << mCurSol.tx() << " " << mAlloc.GetVar(anInd) << "\n";
      
      if (mSys->IsCstrUniv(anInd,aValForced) )
      {
// std::cout << "XXXXXXXXyuutre " << aValForced << " " << aDC[anInd] << "\n"; getchar();
           aB -= aCoeff * aValForced;
      }
      else
      {
          // if EnPtsCur = true, donc ::
           aIndFiltre.push_back(anInd);
           aCoefFiltre.push_back(aCoeff);
           aVSsBl.push_back(GetBlocOfI0Alloc(anInd,anInd+1));
      }
   }
   mSys->GSSR_AddNewEquation_Indexe
   (
      &aVSsBl, &aCoefFiltre[0], (int)aCoefFiltre.size(),
      aIndFiltre, aPds, &aCoefFiltre[0], aB,
      NullPCVU
   );
  
   return aRes;
}

cGenSysSurResol * cSetEqFormelles::Sys()
{
     return mSys;
}

cFormQuadCreuse * cSetEqFormelles::FQC()
{
	return mFQC;
}

cSetEqFormelles::~cSetEqFormelles()
{
    for 
    (
         std::list<cObjFormel2Destroy *>::iterator it=mLobj2Kill.begin();
         it != mLobj2Kill.end();
         it++
    )
        delete *it;

    for 
    (
       tContFcteur::iterator it=mLFoncteurs.begin();
       it!=mLFoncteurs.end();
       it++
   )
       delete *it;

    delete mSys;
    delete &mDicHom;
    delete &mDicCam;
    delete &mDicRot;
}

void cSetEqFormelles::AddObj2Kill(cObjFormel2Destroy * anObj)
{
     mLobj2Kill.push_back(anObj);
}

AllocateurDInconnues & cSetEqFormelles::Alloc()
{
   return mAlloc;
}



void cSetEqFormelles::AddElem(cElemEqFormelle & anEq)
{
     AssertUnClosed();
     std::copy
     (
        anEq.AllFonct().begin(),
        anEq.AllFonct().end(),
        std::back_inserter( mLFoncteurs)
     );
    mLEEF.push_back(&anEq);
}

void cSetEqFormelles::AddFonct(cElCompiledFonc * pCF)
{
    AssertUnClosed();
    mLFoncteurs.push_back(pCF);
}

void cSetEqFormelles::AssertClosed()
{
   ELISE_ASSERT(mClosed,"Ilegal Op on UnClosed cSetEqFormelles");
}
void cSetEqFormelles::AssertUnClosed()
{
   ELISE_ASSERT(! mClosed,"Ilegal Op on Closed cSetEqFormelles");
}

bool cSetEqFormelles::IsClosed() const
{
   return mClosed;
}

class cCmpPtrOrderBloc
{
    public :
       bool operator() (cIncIntervale *  aI1,cIncIntervale *  aI2)
       {
           return aI1->Order() < aI2->Order();
       }
};

bool TestPermutVar = false;
bool ShowPermutVar = false;
bool ShowCholesky = false;

int  cSetEqFormelles::NbBloc() const
{
   return (int)mBlocsIncAlloc.size();
}

const std::vector<cIncIntervale *> &   cSetEqFormelles::BlocsIncAlloc() const
{
   return mBlocsIncAlloc;
}



void cManipOrdInc::Init(const std::vector<cIncIntervale *> &aBlocsIncAlloc) 
{
   mBlocsIncSolve = aBlocsIncAlloc;
   for (int aK=0 ; aK<int(mBlocsIncSolve.size()); aK++)
   {
      mBlocsIncSolve[aK]->SetOrder(mBlocsIncSolve[aK]->Order()+ aK/1e9);
      double aOrder = mBlocsIncSolve[aK]->Order();
      if (TestPermutVar)
      {
             // ELISE_ASSERT(false,"TestPermutVar");
             aOrder = NRrandom3();
             // aOrder = -aK;
      }
      if (mBlocsIncSolve[aK]->IsTmp())
      {
          aOrder = 1e10 + aK*1e5;
      }
      mBlocsIncSolve[aK]->SetOrder(aOrder);
   }

   cCmpPtrOrderBloc aCmpBloc;
   std::sort
   (
            mBlocsIncSolve.begin(),
            mBlocsIncSolve.end(),  
            aCmpBloc
   );
   
   mBlocsIncSolve[0]->SetFirstIntervalSolve();
   for (int aK=1 ; aK<int(mBlocsIncSolve.size()); aK++)
      mBlocsIncSolve[aK]->InitntervalSolve(*(mBlocsIncSolve[aK-1]));

   for (int aKB=0 ; aKB<int(mBlocsIncSolve.size()); aKB++)
   {
       int aI0S = mBlocsIncSolve[aKB]->I0Solve();
       int aI1S = mBlocsIncSolve[aKB]->I1Solve();
       int aI0A = mBlocsIncSolve[aKB]->I0Alloc();

       if (::ShowPermutVar)
       {
           std::cout << " INTERVAL Alloc [" << aI0A << " " << mBlocsIncSolve[aKB]->I1Alloc() << "] "
                     << " Solve [" << aI0S << " " << aI1S << "]\n";
       }


       for (int  aKInc=aI0S; aKInc!=aI1S; aKInc++)
       {
            mI02NblSolve.push_back(aKB);
            mSolve2Alloc.push_back(aKInc-aI0S+aI0A);
            mAlloc2Solve.push_back(-1);
       }
   }
   for (int aK=0 ; aK<int(mSolve2Alloc.size()) ; aK++)
   {
     mAlloc2Solve[mSolve2Alloc[aK]] = aK;
   }
   for (int aK=0 ; aK<int(mSolve2Alloc.size()) ; aK++)
   {
         ELISE_ASSERT(mAlloc2Solve[aK]>=0,"mA2S::Inc in NumBloc");
   }
}

Im1D_REAL8 cManipOrdInc::ReordonneSol(Im1D_REAL8 aIm)
{
   Im1D_REAL8 aRes (aIm.tx());
   for (int aK=0 ; aK<aIm.tx() ; aK++)
      aRes.data()[aK] = aIm.data()[mAlloc2Solve[aK]];

   return aRes;
}

std::vector<cIncIntervale *> &  cManipOrdInc::BlocsIncSolve() { return mBlocsIncSolve;}
std::vector<int>             &  cManipOrdInc::I02NblSolve()   { return mI02NblSolve;}
std::vector<int>             &  cManipOrdInc::Alloc2Solve()   { return mAlloc2Solve;}
std::vector<int>             &  cManipOrdInc::Solve2Alloc()   { return mSolve2Alloc;}

cManipOrdInc::cManipOrdInc()
{
}


/*
{
   cCmpPtrOrderBloc aCmpBloc;
   std::sort
   (
            aBlcInc.begin(),
            aBlcInc.end(),  // IL est essentiel que le bloc tmp soit le dernier
            aCmpBloc
   );
   
   aBlcInc[0]->SetFirstIntervalSolve();
   for (int aK=1 ; aK<int(aBlcInc.size()); aK++)
      aBlcInc[aK]->InitntervalSolve(*(aBlcInc[aK-1]));

}
*/


void cSetEqFormelles::TestPbFaisceau(bool doCheck,bool doSVD,bool doV0)
{
    std::vector<cSsBloc> aVSB;
    for (int aKB=0 ; aKB<int(mBlocsIncAlloc.size()) ; aKB++)
       aVSB.push_back(mBlocsIncAlloc[aKB]->SsBlocComplet());

    mSys->VerifGlob(aVSB,doCheck,doSVD,doV0);
    // mSys->VerifGlob(mBlocsIncAlloc);
}


void cSetEqFormelles::NumeroteBloc()
{
   
   if (TestPermutVar)
   {
       NRrandom3InitOfTime();
   }
   // A CHANGER
/*
   mBlocsIncSolve = mBlocsIncAlloc;
   for (int aK=0 ; aK<int(mBlocsIncSolve.size()); aK++)
   {
      // std::cout << "  BIS " << mBlocsIncSolve[aK]->NumBlocAlloc() << " " <<  mBlocsIncSolve[aK]->IsTmp() << "\n";
       // Pour initialiser une ordre identite en cas de non init
       mBlocsIncSolve[aK]->SetOrder(mBlocsIncSolve[aK]->Order()+ aK/1e9);
      double aOrder = mBlocsIncSolve[aK]->Order();
      if (TestPermutVar)
      {
             ELISE_ASSERT(false,"TestPermutVar");
             aOrder = NRrandom3();
             aOrder = -aK;
      }
      if (mBlocsIncSolve[aK]->IsTmp())
      {
          aOrder = 1e10 + aK*1e5;
      }
      mBlocsIncSolve[aK]->SetOrder(aOrder);
   }
*/


 
   mMOI.Init(mBlocsIncAlloc);
/*
   SortAndInitSolve(mBlocsIncSolve);

   for (int aKB=0 ; aKB<int(mBlocsIncSolve.size()); aKB++)
   {
       int aI0S = mBlocsIncSolve[aKB]->I0Solve();
       int aI1S = mBlocsIncSolve[aKB]->I1Solve();
       int aI0A = mBlocsIncSolve[aKB]->I0Alloc();

       if (::ShowPermutVar)
       {
           std::cout << " INTERVAL Alloc [" << aI0A << " " << mBlocsIncSolve[aKB]->I1Alloc() << "] "
                     << " Solve [" << aI0S << " " << aI1S << "]\n";
       }


       for (int  aKInc=aI0S; aKInc!=aI1S; aKInc++)
       {
            mI02NblSolve.push_back(aKB);
            mSolve2Alloc.push_back(aKInc-aI0S+aI0A);
            mAlloc2Solve.push_back(-1);
       }
   }
   for (int aK=0 ; aK<int(mSolve2Alloc.size()) ; aK++)
   {
     mAlloc2Solve[mSolve2Alloc[aK]] = aK;
   }
   for (int aK=0 ; aK<int(mSolve2Alloc.size()) ; aK++)
   {
         ELISE_ASSERT(mAlloc2Solve[aK]>=0,"mA2S::Inc in NumBloc");
   }
*/
}


Im1D_REAL8 cSetEqFormelles::ReordonneSol(Im1D_REAL8 aIm)
{
   return mMOI.ReordonneSol(aIm);
}


void cSetEqFormelles::SetClosed()
{

   AssertUnClosed();
   mNbVar = mAlloc.CurInc();
   if (mBlocsIncAlloc.empty())
   {
       ELISE_ASSERT(mNbVar==0,"cSetEqFormelles::SetClosed BlocInc empty");
   }
   else
   {
       ELISE_ASSERT(mNbVar==mBlocsIncAlloc.back()->I1Alloc(),"cSetEqFormelles::SetClosed BlocInc empty");
   }
   mClosed = true;

   NumeroteBloc();
   for (tContFcteur::iterator itF=mLFoncteurs.begin(); itF!=mLFoncteurs.end() ; itF++)
   {
       (*itF)->InitBloc(*this);
   }


   if (mTypeSys == eSysCreuxMap)
   {
      cElMatCreuseGen * aMatCr = cElMatCreuseGen::StdNewOne(mNbVar,mNbVar,false);
      mFQC = new cFormQuadCreuse(mNbVar,aMatCr);
      mSys = mFQC;
   }
   else if (mTypeSys ==eSysPlein)
   {
      // mSys = new L2SysSurResol(mNbVar);
      mSys = new L2SysSurResol(mNbVar,!mCanUseCstr);
   }
   else if (mTypeSys == eSysCreuxFixe)
   {
      cElMatCreuseGen * aMatCr = cElMatCreuseGen::StdNewOne(mNbVar,mNbVar,true);
      mFQC = new cFormQuadCreuse(mNbVar,aMatCr);
      mSys = mFQC;
   }
   else if (mTypeSys == eSysL1Barrodale)
   {
      mSys = new SystLinSurResolu(mNbVar,mEstimNbEq);
   }
   else if (mTypeSys ==  eSysL2BlocSym)
   {
      cElMatCreuseGen * aMatCr = cElMatCreuseGen::StdBlocSym(mMOI.BlocsIncSolve(),mMOI.I02NblSolve());
      mFQC = new cFormQuadCreuse(mNbVar,aMatCr);
      mSys = mFQC;
   }
   else
   {
      ELISE_ASSERT(false,"Unknown mTypeSys in cSetEqFormelles::SetClosed");
   }
   mSys->GSSR_Reset(true);
   SetPtCur(mAlloc.ValsVar());
}


void cSetEqFormelles::SetPtCur(const double * aPt)
{
     for (INT aK=0 ; aK < mNbVar; aK++)
     {
         mAlloc.SetVar(aPt[aK],aK);
     }
     UpdateFctr();
}

void cSetEqFormelles::UpdateFctr()
{

     for (  tContFcteur::iterator itF=mLFoncteurs.begin();
            itF!=mLFoncteurs.end() ;
            itF++
     )
     {
         (*itF)->SetCoordCur(mAlloc.ValsVar());
     }
}


void  cSetEqFormelles::AddContrainte(const cContrainteEQF & aContr,bool Stricte,double aPds)
{  
     AssertClosed();

     cElCompiledFonc * aFonct = aContr.FctrContrEQF();

     if (aContr.ContrainteIsStricte())
     {
        if (Stricte)
        {
            aFonct->Std_AddEqSysSurResol
            (
                 true,
                 0.0,
                 mAlloc.ValsVar(),
                 *mSys,
                 *this,
                 true,
                 NullPCVU
            );
       }
     }
     else
     {
        if (!Stricte)
        {

            double anEc = ElAbs(ResiduSigne(aFonct));
            if (aPds<0)
               aPds = aContr.PdsOfEcart(anEc);
            VAddEqFonctToSys(aFonct,aPds,false,NullPCVU);
        }
     }
}

cSsBloc cSetEqFormelles::GetBlocOfI0Alloc(int aI0Alloc,int aI1Alloc) const
{
    // ELISE_ASSERT(aBl.Int()==0,"cSetEqFormelles::GetBlocOfI0");
    cIncIntervale * aRes = GetIntervInclusIAlloc(aI0Alloc);

    ELISE_ASSERT
    (
       (aI0Alloc >= aRes->I0Alloc()) && (aI1Alloc<= aRes->I1Alloc()),
       "cSetEqFormelles::GetBlocOfI0"
    );
    cSsBloc aResul(aI0Alloc-aRes->I0Alloc(),aI1Alloc-aRes->I0Alloc());
    aResul.BlocSetInt(*aRes);
    return aResul;
}


cSsBloc cSetEqFormelles::GetBlocOfI0Alloc(const cIncIntervale & anI) const
{
   // NO::BlocSetInt
   return GetBlocOfI0Alloc(anI.I0Alloc(),anI.I1Alloc());
}

// cSsBloc GetBlocOfI0(const cIncIntervale & aBl) const { }
cIncIntervale *  cSetEqFormelles::GetIntervInclusIAlloc(int anI0) const
{
    return  mBlocsIncAlloc[GetNumBlocInclusIAlloc(anI0)];
}

// extern bool MPD_DEBUG_NEW_PT;
int   cSetEqFormelles::GetNumBlocInclusIAlloc(int anI0) const
{
    // if (MPD_DEBUG_NEW_PT) std::cout << "I0= " << anI0 << " SzBl=" << mI02NblAlloc.size() << "\n";
    if ((anI0<0) || (anI0>=int(mI02NblAlloc.size())))
    {
        std::cout << "I0= " << anI0 << " SzBl=" << mI02NblAlloc.size() << "\n";
        ELISE_ASSERT(false, "cSetEqFormelles::GetIntervRef ");
    }
    // ELISE_ASSERT((anI0>=0) && (anI0<int(mI02NblAlloc.size())), "cSetEqFormelles::GetIntervRef ");

    return   mI02NblAlloc[anI0];
}


void cSetEqFormelles::AddABlocAlloc(cIncIntervale * anII)
{
if (DebugCamBil)
{
   std::cout << "AAA cSetEqFormelles::AddABlocAlloc \n"; 
   getchar();
}
   AssertUnClosed();
   if (mBlocsIncAlloc.empty())
   {
      ELISE_ASSERT(anII->I0Alloc()==0,"cSetEqFormelles::AddABloc Firt elem");
   }
   else
   {
      if (anII->I0Alloc()!=mBlocsIncAlloc.back()->I1Alloc())
      {
          std::cout << anII->I0Alloc() << " " << mBlocsIncAlloc.back()->I1Alloc() << "\n";
          ELISE_ASSERT(false,"cSetEqFormelles::AddABloc Next elem");
      }
   }

   int aNumBl = (int)mBlocsIncAlloc.size();
   for (int anI = anII->I0Alloc() ; anI<anII->I1Alloc() ; anI++)
       mI02NblAlloc.push_back(aNumBl);

   anII->SetNumAlloc(aNumBl);
   mBlocsIncAlloc.push_back(anII);
}



void cSetEqFormelles::AddContrainte (const cMultiContEQF & aMC,bool Stricte,double aPds)
{
     for (int aK=0 ;aK<aMC.NbC(); aK++)
     {
        AddContrainte(aMC.KthC(aK),Stricte,aPds);
     }
}

const std::vector<REAL> & cSetEqFormelles::VResiduSigne (cElCompiledFonc * aFonct)
{
     aFonct->SetCoordCur(mAlloc.ValsVar());
     aFonct->ComputeValAndSetIVC();
     return aFonct->Vals();
}

REAL cSetEqFormelles::ResiduSigne (cElCompiledFonc * aFonct)
{
    return VResiduSigne(aFonct)[0];
}
const std::vector<REAL> & cSetEqFormelles::AddEqIndexeToSys
                          (
                              cElCompiledFonc * aFonct,
                              REAL aPds,
                              const std::vector<INT>  & aVInd
                          )
{
     AssertClosed();
     aFonct->SVD_And_AddEqSysSurResol
     (
          false,
          aVInd,
          aPds,
          mAlloc.ValsVar(),
          *mSys,
          *this,
          true,
          NullPCVU
     );
     return aFonct->Vals();
}

//
// En mode indexe on commence par
//    mCompCoord[aK] = Pts[aVInd[aK]];
//
//
//   mCompCoord[aIC] = aRealCoord[mMapComp2Real[aIC]];
// 


const std::vector<REAL> & cSetEqFormelles::VAddEqFonctToSys
     (
                  cElCompiledFonc * aFonct,
                  const std::vector<double> & aVPds,
                  bool WithDerSec,
                  cParamCalcVarUnkEl * aPCVU
     )
{

     AssertClosed();
   aFonct->Std_AddEqSysSurResol
   (
        false,
        aVPds,
        mAlloc.ValsVar(),
        *mSys,
        *this,
        true,
        aPCVU
   );
     const std::vector<REAL> & aRes = aFonct->Vals();

     return aRes;
}

const std::vector<REAL> & cSetEqFormelles::VAddEqFonctToSys
     (
                  cElCompiledFonc * aFonct,
                  REAL aPds,
                  bool WithDerSec,
                  cParamCalcVarUnkEl * aPCVU
     )
{
    return VAddEqFonctToSys(aFonct,MakeVec1(aPds),WithDerSec,aPCVU);
}

REAL cSetEqFormelles::AddEqFonctToSys
     (
                  cElCompiledFonc * aFonct,
                  REAL aPds,
                  bool WithDerSec
     )
{
    REAL aRes = VAddEqFonctToSys(aFonct,aPds,WithDerSec,NullPCVU)[0];


   return aRes;
    
}


REAL cSetEqFormelles::AddEqFonctToSys
     (
                  const tContFcteur & aCont,
                  REAL aPds,
                  bool WithDerSec
      )
{
     REAL aRes = 0;
     for
     (
          tContFcteur::const_iterator anIt=aCont.begin();
          anIt !=aCont.end();
          anIt++
     )
     {
        aRes += AddEqFonctToSys(*anIt,aPds,WithDerSec);
     }
     return aRes;
}

void cSetEqFormelles::ShowVar() 
{
    for (INT aK=0 ; aK < mAlloc.CurInc(); aK++)
    {
        std::cout << "cSEShowV : " << aK 
                  <<  " "  << mAlloc.GetVar(aK) 
                  <<  " &="  << mAlloc.GetAdrVar(aK) 
                  << "\n";
    }
}

void cSetEqFormelles::SolveResetUpdate(double ExpectResidu,bool *OK)
{
if (0)
{
    std::cout << "cSetEqFormelles::SolveResetUpdate " << ExpectResidu << " " << OK << "\n";
    ShowVar();
    ShowSpectrSys(*this);
    for (INT aK=0 ; aK < mAlloc.CurInc(); aK++)
    {
	    std::cout << "DIAG[" << aK <<"]= "  << mSys->GetElemQuad(aK,aK) << "\n";
    }

    getchar();

}
 
    Solve(ExpectResidu,OK);
    ResetUpdate(1.0);
}

bool DoCheckResiduPhgrm=false;


void cSetEqFormelles::Solve(double ExpectResidu,bool *OK)
{
    AssertClosed();

    // Precaution anti degenerescence :
    for 
    ( 
        std::list<cEqfBlocIncTmp *>::iterator itB=mLBlocTmp.begin();
        itB != mLBlocTmp.end();
	itB++
    )
    {
        int I0 = (*itB)->IncInterv().I0Solve();
        int I1 = (*itB)->IncInterv().I1Solve();
        for (int aK=I0; aK<I1 ; aK++)
        {
	    mSys->SetElemQuad(aK,aK,1);
        }
    }

    

    if (false)
    {
        ShowSpectrSys(*this);
        std::cout << "SetEqFormelles::Solve:DoneIntervvvvvvvvvvvvv \n";
        getchar();
    }

    if (::DebugPbCondFaisceau)
    {
/*
        std::cout << "======== BEGIN GLOB  TestPbFaisceau======== \n";
        TestPbFaisceau(true,true,true);
        std::cout << "======== DONE GLOB  TestPbFaisceau======== \n";
*/
    }



    mSolQuad = mSys->GSSR_Solve(OK);
    if (OK)
    {
       ELISE_ASSERT(OK,"Solve pb detected in cSetEqFormelles::SolveResetUpdate");
    }

   if (ExpectResidu >=0)
   {
          Im1D_REAL8  aV0(mNbVar,0.0);
/*
          for (INT aK=0 ; aK < mNbVar; aK++)
          {
              aDC[aK]  = mAlloc.GetVar(aK);
          }
*/
          double aR0 = mSys->ResiduOfSol(aV0.data());
          double aDif = ElAbs(aR0-ExpectResidu) / (1e-8+ExpectResidu);
          if ((aDif>1e-7) && DoCheckResiduPhgrm)
          {
              if (aDif>=1e-1)
              {
                  std::cout 
                     <<  " COST 0 :  " << aR0
                     <<  " COST SOL " <<  mSys->ResiduOfSol(mSolQuad.data())
                     <<  "  EXP  " <<  ExpectResidu
                     <<  "  DIF  " <<  aDif
                     << "\n";
/*
*/

              // cSetEqFormelles::AddEqLineaire : 
                 ELISE_ASSERT(false,"Expect Residu in mSetEq.SolveResetUpdate");
              }
          }

    }
    
    mSolQuad = ReordonneSol(mSolQuad);
    mCurSol.Resize(mSolQuad.tx());

    REAL * aDC = mCurSol.data();
    for (INT aK=0 ; aK < mNbVar; aK++)
    {
        aDC[aK]  = mAlloc.GetVar(aK);
    }
}


void cSetEqFormelles::SetSol(double aLambda)
{

    Im1D_REAL8 aNewSol(mNbVar);
    
    REAL * aDQ = mSolQuad.data();
    REAL * aDC = mCurSol.data();
    REAL * aDN = aNewSol.data();

    // Puisque la solution est calculee en delta 
    // (convention EnPtsCur=true)
    for (INT aK=0 ; aK < mNbVar; aK++)
    {
         aDN[aK] = aDC[aK] + aLambda * aDQ[aK];
        //  aDS[aK] += mAlloc.GetVar(aK);
    }
    SetPtCur(aDN);

    for 
    (
         std::list<cObjFormel2Destroy *>::iterator it=mLobj2Kill.begin();
         it != mLobj2Kill.end();
         it++
    )
    {
        (*it)->Update_0F2D();
    }
    for 
    (
        std::list<cElemEqFormelle *>::iterator itL=mLEEF.begin();
        itL!=mLEEF.end();
        itL++
    )
    {
        (*itL)->SetValInitOnValCur();
    }
}

void cSetEqFormelles::DebugResetSys()
{
    mSys->GSSR_Reset(false);
}

void cSetEqFormelles::ResetUpdate(double aLambda)
{
// getchar();
 /*
*/
    SetSol(aLambda);
    mSys->GSSR_Reset(true);


}

void  cSetEqFormelles::Reinit(INT k0,INT k1)
{
   for (INT aK=k0; aK<k1 ; aK++)
         mAlloc.Reinit(aK);
}

cHomogFormelle * cSetEqFormelles::GetHomFromName(const std::string & aName)
{
    cHomogFormelle * aRes = mDicHom[aName];
    ELISE_ASSERT(aRes!=0,"GetHomFromName");
    return aRes;
}

cCameraFormelle * cSetEqFormelles::GetCamFromName(const std::string & aName)
{
    cCameraFormelle * aRes = mDicCam[aName];
    ELISE_ASSERT(aRes!=0,"GetCamFromName");
    return aRes;
}

cRotationFormelle * cSetEqFormelles::GetRotFromName(const std::string & aName)
{
    cRotationFormelle * aRes = mDicRot[aName];
    ELISE_ASSERT(aRes!=0,"GetRotFromName");
    return aRes;
}



    // ---------------- Allocations statiques ----------------

// void VerifC2MForPIF(bool isDC2M,CamStenope *);

cParamIntrinsequeFormel *  cSetEqFormelles::NewParamIntrNoDist
                           (
			        bool isDC2M,
			        CamStenope * aCamInit,
                                bool ParamVar
                           )
{
   VerifC2MForPIF(isDC2M,aCamInit);

   cParamIntrinsequeFormel * aRes = 
          new cParamIntrinsequeFormel(isDC2M,aCamInit,*this,ParamVar);
   aRes->CloseEEF();
   AddObj2Kill(aRes);
   return aRes;
}

cParamIFDistRadiale *  cSetEqFormelles::NewIntrDistRad
                       (
                            bool                    isDC2M,
			    cCamStenopeDistRadPol * aCam,
                            int aDegFig
                       )
{
   VerifC2MForPIF(isDC2M,aCam);
   cParamIFDistRadiale * aRes = 
          new cParamIFDistRadiale(isDC2M,aCam,*this,aDegFig);
   aRes->CloseEEF();
   AddObj2Kill(aRes);
   return aRes;
}


cParamIFDistStdPhgr *  cSetEqFormelles::NewIntrDistStdPhgr
                       (
                            bool                    isDC2M,
			    cCamStenopeModStdPhpgr * aCam,
                            int aDegFig
                       )
{
   VerifC2MForPIF(isDC2M,aCam);
   cParamIFDistStdPhgr * aRes = 
          new cParamIFDistStdPhgr(isDC2M,aCam,*this,aDegFig);
   aRes->CloseEEF();
   AddObj2Kill(aRes);
   return aRes;
}


cEqHomogFormelle * cSetEqFormelles::NewEqHomog
(
   bool                  InSpaceInit,
   cHomogFormelle &      aHF1,
   cHomogFormelle &      aHF2,
   cDistRadialeFormelle* aDRF,
   bool                  Code2Gen
)
{
     AssertUnClosed();
     cEqHomogFormelle * aRes = new cEqHomogFormelle(InSpaceInit,aHF1,aHF2,aDRF,Code2Gen);

     AddObj2Kill(aRes);
     return aRes;
}

cEqOneHomogFormelle *    cSetEqFormelles::NewOneEqHomog
                         (
                             cHomogFormelle & aHF,
                             bool Code2Gen 
                         )
{
     AssertUnClosed();
     cEqOneHomogFormelle * aRes = new cEqOneHomogFormelle(aHF,Code2Gen);
     AddObj2Kill(aRes);
     return aRes;
}

#if (0)
cEqHomogFormelle * cSetEqFormelles::NewEqHomog
(
   cHomogFormelle &      aHF1,
   cHomogFormelle &      aHF2,
   cDistRadialeFormelle& aDRF,
   bool                  Code2Gen
)
{
     AssertUnClosed();
     cEqHomogFormelle * aRes = new cEqHomogFormelle(aHF1,aHF2,aDRF,Code2Gen);
     AddObj2Kill(aRes);
     return aRes;
}
#endif

cDistRadialeFormelle * cSetEqFormelles::NewDistF
                      (bool doCloseEEF,bool Fige,INT DegFige,
		       const ElDistRadiale_PolynImpair & aDist)
{
  AssertUnClosed();

  cDistRadialeFormelle * pRes =  new cDistRadialeFormelle(doCloseEEF,Fige,DegFige,aDist,*this);
  AddObj2Kill(pRes);
  return pRes;
}

cEqCorrelGrid * cSetEqFormelles::NewEqCorrelGridGen
                (INT aNbPix, bool Im2MoyVar, bool GenCode,bool CanReuse)
{
  cEqCorrelGrid * & aReuse = 
	     mDicEqCorGrid[cEqCorrelGrid::NameType(aNbPix,Im2MoyVar)];
  if (CanReuse && (aReuse != 0))
	  return aReuse;
  // AssertUnClosed();
  cEqCorrelGrid * pRes = new cEqCorrelGrid(*this,aNbPix,Im2MoyVar,GenCode);
  AddObj2Kill(pRes);
  if (aReuse == 0)
     aReuse = pRes;
  return pRes;
}

cEqCorrelGrid * cSetEqFormelles::NewEqCorrelGrid
                (INT aNbPix, bool Im2MoyVar, bool GenCode)
{
   return NewEqCorrelGridGen(aNbPix,Im2MoyVar,GenCode,false);
}

cEqCorrelGrid * cSetEqFormelles::ReuseEqCorrelGrid(INT aNbPix, bool Im2MoyVar)
{
    return NewEqCorrelGridGen(aNbPix,Im2MoyVar,false,Im2MoyVar);
}

cHomogFormelle * cSetEqFormelles::NewHomF
                 (
		      const cElHomographie & anH,
		      eModeContrHom  aModeCtrl,
		      const std::string & aName 
		 )
{
  AssertUnClosed();

  cHomogFormelle *pRes =  new cHomogFormelle(anH,*this,aModeCtrl);
  AddObj2Kill(pRes);
  if (aName != "")
     mDicHom[aName] = pRes;
  return pRes;
}

cParamIFHomogr  * cSetEqFormelles::NewDistHomF
                  (
                       bool                    isDC2M,
                       cCamStenopeDistHomogr * aCam,
		       eModeContrHom          aMode
                  )
{
  VerifC2MForPIF(isDC2M,aCam);
  AssertUnClosed();
  cParamIFHomogr * pRes = new cParamIFHomogr(isDC2M,aCam,*this,aMode);
  pRes->CloseEEF();
  AddObj2Kill(pRes);

  return pRes;
}

// cParamIFHomogr  * NewDistHomF(const cElHomographie &,bool aRotFree);


cEqEllipseImage * cSetEqFormelles::NewEqElIm
                  (
		      const cMirePolygonEtal & aMire,
                      Pt2dr aCentre, REAL  anA, REAL  aB, REAL  aC,
                      REAL  aLarg, REAL  aBlanc, REAL  aNoir,
		      bool Code2Gen
                  )
{
   cEqEllipseImage * pRes = new cEqEllipseImage
	   (*this,aMire,aCentre,anA,aB,aC,aLarg,aBlanc,aNoir,Code2Gen);
  AddObj2Kill(pRes);
  return pRes;
}

cEqEllipseImage * cSetEqFormelles::NewEqElIm(const cMirePolygonEtal & aMire,bool Code2Gen)
{
    bool isNeg =  aMire.IsNegatif();
    int aBlanc = 255;
    int aNoir = 0;
    if (isNeg)
        ElSwap(aBlanc,aNoir);

    return NewEqElIm(aMire,Pt2dr(0,0),1,0,1,0.5,aBlanc,aNoir,Code2Gen);
}


cRotationFormelle * cSetEqFormelles::NewRotationGen
                    (
                         eModeContrRot aMode,
                         ElRotation3D aRC2MInit,
                         cRotationFormelle * pRAtt,
			 const std::string & aName,
                         INT aDegre,
                         bool aVraiBaseU
                    )
{
   AssertUnClosed();
   cRotationFormelle * aRes =  new  cRotationFormelle(aMode,aRC2MInit,*this,pRAtt,aName,aDegre,aVraiBaseU);
   AddObj2Kill(aRes);
   if (aName != "")
      mDicRot[aName] = aRes;
   return aRes;
}

cRotationFormelle * cSetEqFormelles::NewRotation
                    (
                         eModeContrRot aMode,
                         ElRotation3D aRC2MInit,
                         cRotationFormelle * pRAtt,
			 const std::string & aName
                    )
{
   return NewRotationGen(aMode,aRC2MInit,pRAtt,aName,0,false);
}

cRotationFormelle * cSetEqFormelles::NewRotationEvol
                    (
                          ElRotation3D aRC2MInit,
                          INT aDegre,
                          const std::string & aName
                    )
{
   return NewRotationGen
          (
                cNameSpaceEqF::eRotLibre,
                aRC2MInit,
                (cRotationFormelle *)0,
                aName,
                aDegre,
                false
          );
}



cCpleCamFormelle * cSetEqFormelles::NewCpleCam
                   (
                       cCameraFormelle & aCam1,
                       cCameraFormelle & aCam2,
		       eModeResidu      aMode,
		       bool              Code2Gen
                   )
{
   cCpleCamFormelle * aRes =  new cCpleCamFormelle(aCam1,aCam2,aMode,Code2Gen);
   AddObj2Kill(aRes);
   return aRes;
}

cParamIFDistPolynXY  * cSetEqFormelles::NewIntrPolyn(bool isDistC2M,cCamStenopeDistPolyn * aCam)
{  
  VerifC2MForPIF(isDistC2M,aCam);
   cParamIFDistPolynXY * aRes = new cParamIFDistPolynXY(isDistC2M,aCam,*this);
   aRes->CloseEEF();
   AddObj2Kill(aRes);
   return aRes;
}

void cSetEqFormelles::AddCamFormelle
     (cCameraFormelle  * pCam,const std::string & aName)
{
     AddObj2Kill(pCam);
     if (aName != "")
        mDicCam[aName] = pCam;
}

cCpleGridEq * cSetEqFormelles::NewCpleGridEq
              (
                    cTriangulFormelle & aTR1, 
                    cRotationFormelle & aR1,
                    cTriangulFormelle & aTR2,
                    cRotationFormelle & aR2,
		    eModeResidu       aMode,
		    bool              Code2Gen

              )
{
	cCpleGridEq * aRes = new cCpleGridEq(aTR1,aR1,aTR2,aR2,aMode,Code2Gen);
        AddObj2Kill(aRes);
	return aRes;
}


cAppuiGridEq * cSetEqFormelles::NewEqAppuiGrid
               (
                   cTriangulFormelle & aTri,
                   cRotationFormelle & aRot,
                   bool Code2Gen
               )
{
    cAppuiGridEq * aRes = new cAppuiGridEq(aTri,aRot,Code2Gen);
    AddObj2Kill(aRes);
    return aRes;
}


cLIParam_Image *  cSetEqFormelles::NewLIParamImage
                  (
                       Im2D_REAL4 anIm,
                       REAL  aZoom,
                       CamStenope & aCam,
                       cNameSpaceEqF::eModeContrRot aMode
                  )
{
    cLIParam_Image * aRes = new cLIParam_Image(*this,anIm,aZoom,aCam,aMode);
    AddObj2Kill(aRes);
    return aRes;
}

cEqVueLaserImage * cSetEqFormelles::NewLIEqVueLaserIm
                 (
		     cRotationFormelle * aRotPts,
		     bool                Multi,
		     bool                Normalize,
		     INT              aNbPts,
		     cLIParam_Image & anI1,
		     cLIParam_Image & anI2,
		     bool             GenCode 
		  )
{
    ELISE_ASSERT
    (
        (this == &(anI1.Set())) && (this == &(anI2.Set())),
	"Dif Set in cSetEqFormelles::NewLIEqVueLaserIm"
    );
    cEqVueLaserImage * aRes = new cEqVueLaserImage(aRotPts,Multi,Normalize,aNbPts,anI1,anI2,GenCode);
    AddObj2Kill(aRes);
    if (GenCode) return 0;
    return aRes;
}




/************************************************************/
/*                                                          */
/*                    cEqFPtLiaison                         */
/*                                                          */
/************************************************************/


cEqFPtLiaison::cEqFPtLiaison() :
  mMemberX1  ("XL1"),
  mMemberY1  ("YL1"),
  mMemberX2  ("XL2"),
  mMemberY2  ("YL2"),
  mP1        (cVarSpec(0,mMemberX1),cVarSpec(0,mMemberY1)),
  mP2        (cVarSpec(0,mMemberX2),cVarSpec(0,mMemberY2))
{
}

cEqFPtLiaison::~cEqFPtLiaison() 
{
}

ElPackHomologue &  cEqFPtLiaison::StdPack()
{
     return mStdPack;
}


REAL cEqFPtLiaison::AddPackLiaisonP1P2
(
     const ElPackHomologue & aPack,
     bool WithD2,
     cElStatErreur * aStat,
     REAL aPdsGlob ,
     REAL * SomPdsTot
)
{
    REAL aRes  = 0.0;
    REAL aSPds = 0.0;
    for
    (
       ElPackHomologue::const_iterator it=aPack.begin();
       it!=aPack.end();
       it++
    )
    {
       REAL aP = aPdsGlob * it->Pds();
       if (SomPdsTot)
          *SomPdsTot += aP;
       REAL Ecart = ElAbs(AddLiaisonP1P2(it->P1(),it->P2(),aP,WithD2));
       aRes += Ecart * aP;
       aSPds += aP;
       if (aStat)
          aStat->AddErreur(Ecart);
    }
        
    return aRes / aSPds;
}


void cEqFPtLiaison::PondereFromResidu
     (ElPackHomologue & aPack,REAL Ecart,REAL anEcCoupure)
{
     for
     (
          ElPackHomologue::iterator it=aPack.begin();
          it!=aPack.end();
          it++
     )
     {
        REAL aResidu = ResiduNonSigneP1P2(it->P1(),it->P2());
        it->Pds() = 1 / (1+ElSquare(aResidu/Ecart));
        if ((anEcCoupure > 0) && (aResidu > anEcCoupure))
           it->Pds() = 0;
     }
}

/************************************************************/
/*                                                          */
/*                    cSignedEqFPtLiaison                   */
/*                                                          */
/************************************************************/

REAL cSignedEqFPtLiaison::ResiduNonSigneP1P2(Pt2dr aP1,Pt2dr aP2) 
{
    return ElAbs(ResiduSigneP1P2(aP1,aP2));
}


/************************************************************/
/*                                                          */
/*                    cEqFormelleLineaire                   */
/*                                                          */
/************************************************************/
cAllocNameFromInt cEqFormelleLineaire::TheNK("KthEqL_");
std::string cEqFormelleLineaire::TheNameCste = "Cste";
 
cEqFormelleLineaire::cEqFormelleLineaire
(
     cSetEqFormelles & aSet,
     INT aNbInc,
     INT aNbVarTot,
     bool GenCode
) :
  mNbInc     (aNbInc),
  mSet       (aSet),
  mNameType (std::string("cEqLin_")+ToString(aNbInc))
{
     Fonc_Num f = -cVarSpec(1,TheNameCste);
     INT aD = aNbVarTot-aNbInc;
     if (GenCode)
	     aD = 0;
     for (INT aK = 0; aK<aNbInc ; aK++)
     {
         mIntervs.push_back(cIncIntervale(TheNK.NameKth(aK),aK+aD,aK+1+aD,false));
         mLInterv.AddInterv(mIntervs.back(),true);
	 f = f + cVarSpec(0,TheNK.NameKth(aK)) * kth_coord(aK);
     }

     if (GenCode)
     {
         cElCompileFN::DoEverything
         (
               std::string("src")+ELISE_CAR_DIR+"GC_photogram"+ELISE_CAR_DIR,
               mNameType,
               f,
               mLInterv
         );
	 return;
     }
     mFctr = cElCompiledFonc::AllocFromName(mNameType);
     ELISE_ASSERT(mFctr!=0,"Cannot Find Fctr in cEqFormelleLineaire");
     mAdrCste = mFctr->RequireAdrVarLocFromString(TheNameCste);
     for (INT aK = 0; aK<aNbInc ; aK++)
        mAdrCoeff.push_back(mFctr->RequireAdrVarLocFromString(TheNK.NameKth(aK)));
}

cEqFormelleLineaire * cSetEqFormelles::NewEqLin
                      (INT aNInc,INT aNbVarTot,bool GenCode)
{
    return new cEqFormelleLineaire(*this,aNInc,aNbVarTot,GenCode);
}

void cEqFormelleLineaire::AddEqNonIndexee
     (
         REAL Cste,
         REAL * Val,
	 REAL aPds,
	 const std::vector<INT>  & VIncs
     )
{
    ELISE_ASSERT
    (
        INT(VIncs.size()) ==  mNbInc,
        "cEqFormelleLineaire::AddEqNonIndexee"
    );
    *mAdrCste = Cste;
     for (INT aK = 0; aK<mNbInc ; aK++)
     {
	     INT I = VIncs[aK];
	     *mAdrCoeff[aK] = Val[aK] ;
	     mIntervs[aK].SetI0I1Alloc(I,I+1);
             mLInterv.ResetInterv(mIntervs[aK]);
     }
     mFctr->SetMappingCur(mLInterv,&mSet);

     mSet.AddEqFonctToSys(mFctr,aPds,false);
}

void cEqFormelleLineaire::AddEqIndexee
     (
         REAL Cste,
         REAL * Val,
	 REAL aPds,
	 const std::vector<INT>  & VIncs
     )
{
    *mAdrCste = Cste;
     for (INT aK = 0; aK<mNbInc ; aK++)
     {
          *mAdrCoeff[aK] = Val[aK] ;
     }
     mSet.AddEqIndexeToSys(mFctr,aPds,VIncs);
}


/*****************************************************/
/*                                                   */
/*                   cContrainteEQF                  */
/*                                                   */
/*****************************************************/

cContrainteEQF::cContrainteEQF(cElCompiledFonc * aFCtr,double aTol) :
   mFctr   (aFCtr),
   mTol    (aTol),
   mMin    (1e-3),
   mMax    (1e3),
   mPds    (1.0)
{
}

bool cContrainteEQF::ContrainteIsStricte() const
{
   return mTol <= 0;
}

cElCompiledFonc * cContrainteEQF::FctrContrEQF() const
{
   return mFctr;
}

double cContrainteEQF::PdsOfEcart(double anEcart) const
{
    // return mPds * ElMin(ElMax(anEcart,mEcMin),mEcMax);
    return mPds * ElMin(mMax,ElMax(mMin,anEcart/mTol));
}


const double cContrainteEQF::theContrStricte = -1.0;


/*****************************************************/
/*                                                   */
/*                   cMultiContEQF                   */
/*                                                   */
/*****************************************************/

cMultiContEQF::cMultiContEQF()
{
}




void cMultiContEQF::AddAcontrainte(cElCompiledFonc * aFCtr,double aTol)
{
   mContraintes.push_back(cContrainteEQF(aFCtr,aTol));
}

int cMultiContEQF::NbC() const
{
   return (int)mContraintes.size();
}

const cContrainteEQF & cMultiContEQF::KthC(int aKth) const
{
    ELISE_ASSERT
    (
        (aKth>=0) && (aKth<int(mContraintes.size())),
	"cMultiContEQF::KthC"
    );
    return mContraintes[aKth];
}

void cMultiContEQF::Add(const cMultiContEQF & aM2)
{
    for (int aK=0; aK<int(aM2.mContraintes.size()) ; aK++)
       mContraintes.push_back(aM2.mContraintes[aK]);
}


cP3dFormel::cP3dFormel(const Pt3dr & aPt,const std::string & aName,cSetEqFormelles & aSet,cIncListInterv & aLI) :
   cElemEqFormelle(aSet,false),
   mPt   (aPt),
   mFPt  (aSet.Alloc().NewPt3(aName,mPt))
{
   IncInterv().SetName(aName);
   CloseEEF();
   aLI.AddInterv(IncInterv());
}

cP2dFormel::cP2dFormel(const Pt2dr & aPt,const std::string & aName,cSetEqFormelles & aSet,cIncListInterv & aLI) :
   cElemEqFormelle(aSet,false),
   mPt   (aPt),
   mFPt  (aSet.Alloc().NewPt2(aName,mPt))
{
   IncInterv().SetName(aName);
   CloseEEF();
   aLI.AddInterv(IncInterv());
}

cValFormel::cValFormel(const double & aVal,const std::string & aName,cSetEqFormelles & aSet,cIncListInterv & aLI) :
   cElemEqFormelle(aSet,false),
   mVal   (aVal),
   mFVal  (aSet.Alloc().NewF(aName,aName,&mVal))
{
   IncInterv().SetName(aName);
   CloseEEF();
   aLI.AddInterv(IncInterv());
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
