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

const double DefBundle3Im = 1e30;
const double DefScoreTKS() {return   DefBundle3Im / 10;}





#define TypeSysLin cNameSpaceEqF::eSysPlein
// #define TypeSysLin cNameSpaceEqF::eSysL2BlocSym



extern bool AllowUnsortedVarIn_SetMappingCur;

/*
     cPt3dEEF et cScalEEF sont definie afin d'heriter de cElemEqFormelle, cObjFormel2Destroy
    Sinon, on aurai interet a les traiter directement dans cGlobEqLineraiseAngle
*/


/************************************************************/
/*                                                          */
/*                cBundleOptim                              */
/*                                                          */
/************************************************************/



/************************************************************/
/*                                                          */
/*            Bundle classique                              */
/*                                                          */
/************************************************************/

/*
    aVDir[1] = mR2 * Pt3dr(aVPts[1].x,aVPts[1].y,1);
    mI2.SetEtat(ProjStenope(aVDir[1]));
    aVP0.push_back(mB2Cur);
    aVP1.push_back(mB2Cur+aVDir[1]);
    aNbP++;

    mR2 = aRot.Mat();
    mB2Cur =  aRot.tr();

    Pt3d<Fonc_Num> aP2 =  aPIGlob + (mW2->mP ^ aPIGlob) - mB2.PtF() - mC2.PtF()*mc2->mS - mD2.PtF()*md2->mS ;


    R2  :  C2toC1

    P sur  C1M  U1M
    P sur  C2M  U2M




    P sur         C1-> U1
    P +t2  sur  R2 C2-> U2
    P +t3  sur  R3 C3-> U3


    tR2 P sur         tR2 C1-> tR2 U1
    tR2 P + tR2 t2  sur C2-> U2
    tR2 P +  t3  sur  tR2 R3 C3-> U3

   Si P est qqcq  tR2 P  est qcq,
   Si t2 est norem qcq , tR2 te est norme qcq

     Q sur         tR2 C1-> tR2 U1  , tR2 =  (Id + ^W) * tR20  , (Id + ^W) Q sur tR20 U1
     Q + t'2  sur C2-> U2
     Q + t'3  sur  tR2 R3 C3-> U3

*/

// this constant should be static member of cEqBundleBase but it could not be initialized (waiting for c++11)
// VC: error C2864: 'cEqBundleBase::ThePropERRInit': a static data member with an in-class initializer must have non-volatile const integral type
// gcc: warning: non-static data member initializers only available with -std=c++11 or -std=gnu++11
#define ThePropERRInit       0.80

class cEqBundleBase;

class cEqSupBB
{
     public :
       cEqSupBB(cEqBundleBase & ,int aK);
       void AllocInterv();
       void PostInit();
       void InitNewRot(const ElRotation3D & aRot);

       cEqBundleBase *       mBB;
       std::string           mNameEq3;
       cSetEqFormelles *     mSetEq3;  
       cPt3dEEF *            mC3;
       cPt3dEEF *            mW3;
       cP2d_Etat_PhgrF       mI3;  
       cP3d_Etat_PhgrF       mC3Init;
       cEqfP3dIncTmp *       mEq3P3I;
       cIncListInterv        mLInterv3;
       ElMatrix<double>      mR3;
       ElRotation3D          mCurRot;
       Pt3dr                 mCurC3;
       cElCompiledFonc *     mEq;
};




class cEqBundleBase  : public cNameSpaceEqF,
                       public cObjFormel2Destroy
{
    public :
       cEqBundleBase(bool DoGenCode,int aNbCamSup,double aFoc,bool UseAccelCoordCste = false);  // Nb de cam en plus des 2 minim

//  InitNewRot = InitNewR2 , mais interface commune pour 2 image
       void    InitNewRot(const ElRotation3D & aRot);
       void    InitNewR2R3(const ElRotation3D & aR2,const ElRotation3D & aR3);

       ElRotation3D SolveResetUpdate();
       std::vector<ElRotation3D> GenSolveResetUpdate();
       double AddEquation12(const Pt2dr & aP1,const Pt2dr & aP2,double aPds);
       double ResiduEquation12(const Pt2dr & aP1,const Pt2dr & aP2);
       ElRotation3D  CurSol() const;

       double AddEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel,double aPds);
       double ResiduEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel);
       const std::string & NameEq1() const;

       bool UseAccelCoordCste() const {return mUseAccelCoordCste;}
       bool DoGenCode() const {return mDoGenCode;}
       const std::string PostAccel () const {return mPostAccel;}

       cSetEqFormelles * SetEqPrinc() {return  mSetEq;}
       cSetEqFormelles * SetEqSec() {return mDoGenCode ? new cSetEqFormelles(TypeSysLin) : mSetEq;}
       cEqfP3dIncTmp   * PtIncSec(cSetEqFormelles * aSetSec)  {return mDoGenCode ? aSetSec->Pt3dIncTmp() : mEqP3I;}
       double    AddEquationGen(const std::vector<Pt2dr> & aP2,const std::vector<bool> & aVSel, double aPds,bool WithEq,bool BugTK=false);

       bool OkLastI() const {return mOkLastI;}
       Pt3dr LastI() const {ELISE_ASSERT(mOkLastI,"No PI in  cEqBundleBase"); return mLastI;}

    protected :
       void    InitNewR2(const ElRotation3D & aRot);
       void    InitNewR3(int aK,const ElRotation3D & aRot);
       // virtual Pt2dr    AddEquationGen(const Pt3dr & aP1,const Pt3dr & aP2,double aPds,bool WithEq) = 0;

       double    AddEquation12Gen(const Pt2dr & aP1,const Pt2dr & aP2, double aPds,bool WithEq);


       bool  mUseAccelCoordCste;
       bool  mDoGenCode;

       ElMatrix<double> mR2;
       double           mFoc;


       cSetEqFormelles * mSetEq;
       cSetEqFormelles * mSetEq2;  // Galere pour gere la connexist des intervalle en mode GenCode
       cPt3dEEF *        mW2;
       cP3d_Etat_PhgrF   mB2;
       Pt3dr             mB2Cur;
       cScalEEF *        mc2;
       cP3d_Etat_PhgrF   mC2;
       cScalEEF *        md2;
       cP3d_Etat_PhgrF   mD2;

       cP2d_Etat_PhgrF   mI1;  // tRot ^ U1
       cP2d_Etat_PhgrF   mI2;

       cIncListInterv           mLInterv1;
       cIncListInterv           mLInterv2;
       std::vector<cIncListInterv*>  mVLInterv;
       int                   mNbCamSup;
       std::string           mPostAccel;
       std::string           mNameEq1;
       std::string           mNameEq2;

       std::vector<cElCompiledFonc *>         mVFEsResid;
       cEqfP3dIncTmp *                        mEqP3I;
       cEqfP3dIncTmp *                        mEq2P3I;
       // cSubstitueBlocIncTmp *                 mSBIT12;
       std::map<int,cSubstitueBlocIncTmp *>   mMapBufSub;
       ElRotation3D             mCurRot;
       std::vector<cEqSupBB *>  mVEqSup;

       bool                     mOkLastI;
       Pt3dr                    mLastI;
};


cEqSupBB::cEqSupBB(cEqBundleBase & aBB,int aK) :
     mBB       (&aBB),
     mNameEq3  ("cEqBBCamThird" + mBB->PostAccel ()),
     mSetEq3   (mBB->SetEqSec()),
     mC3       (new cPt3dEEF(*mSetEq3,Pt3dr(0,0,0),mBB->UseAccelCoordCste())),
     mW3       (new cPt3dEEF(*mSetEq3,Pt3dr(0,0,0),mBB->UseAccelCoordCste())),
     mI3       ("I" + ToString(aK)),
     mC3Init   ("CInit"+ToString(aK)),
     mR3       (3,3),
     mCurRot   (ElRotation3D::Id)
{
    mC3->IncInterv().SetName("C3");
    mW3->IncInterv().SetName("Omega3");
}

void cEqSupBB::AllocInterv()
{
    mEq3P3I  =   mBB->PtIncSec(mSetEq3);
    mLInterv3.AddInterv(mC3->IncInterv());
    mLInterv3.AddInterv(mW3->IncInterv());
    mLInterv3.AddInterv(mEq3P3I->IncInterv());
}

    
void cEqSupBB::PostInit()
{
    mEq  = cElCompiledFonc::AllocFromName(mNameEq3);
    ELISE_ASSERT(mEq != 0,"Cannot allocate cEqSupBB::PostInit");
    mEq->SetMappingCur(mLInterv3,mBB->SetEqPrinc());
    mI3.InitAdr(*mEq);
    mC3Init.InitAdr(*mEq);
}

void cEqSupBB::InitNewRot(const ElRotation3D & aRot)
{
    mCurRot = aRot;
    mR3 = aRot.Mat();
    mW3->mP0 = Pt3dr(0,0,0);

    mCurC3 = mCurRot.tr();
    mC3->mP0 = Pt3dr(0,0,0);
    mC3Init.SetEtat(mCurC3);
}


extern bool ShowStatMatCond;

cEqBundleBase::cEqBundleBase(bool DoGenCode,int aNbCamSup,double aFoc,bool UAC) :
    mUseAccelCoordCste (UAC),
    mDoGenCode         (DoGenCode),
    mR2         (1,1),
    mFoc        (aFoc),
    mSetEq      (new cSetEqFormelles(TypeSysLin)),
    // mSetEq2     ( DoGenCode ? new cSetEqFormelles(TypeSysLin) : mSetEq),
    mSetEq2     (SetEqSec()),
    mW2         (new cPt3dEEF(*mSetEq2,Pt3dr(0,0,0),mUseAccelCoordCste)),
    mB2         ("VecB2"),
    mc2         (new cScalEEF (*mSetEq2,0,mUseAccelCoordCste)),
    mC2         ("VecC2"),
    md2         (new cScalEEF (*mSetEq2,0,mUseAccelCoordCste)),
    mD2         ("VecD2"),
    mI1         ("I1"),
    mI2         ("I2"),
    mNbCamSup   (DoGenCode ? 1 : aNbCamSup),
    mPostAccel   (mUseAccelCoordCste ? "_AccelCsteCoord" : ""),
    mNameEq1    ("cEqBBCamFirst" + mPostAccel),
    mNameEq2    ("cEqBBCamSecond" + mPostAccel),
    // mEqP3I      (mSetEq->Pt3dIncTmp()),
    // mEq2P3I     (PtIncSec(mSetEq2)), //  (DoGenCode ? mSetEq2->Pt3dIncTmp() : mEqP3I ),
    // mSBIT12     (new cSubstitueBlocIncTmp(*mEqP3I)),
    mCurRot     (ElRotation3D::Id)
    
{
  ShowStatMatCond = false;
  for (int aKES=0 ; aKES<mNbCamSup ; aKES++)
  {
       mVEqSup.push_back(new cEqSupBB(*this,aKES+3));
  }
  mEqP3I    =  mSetEq->Pt3dIncTmp();
  mEq2P3I   =  PtIncSec(mSetEq2);
  //  std::cout << "AcEqBundleBase::cEqBundleBase \n"; getchar();
  AllowUnsortedVarIn_SetMappingCur = true;
  mW2->IncInterv().SetName("Omega2");
  mc2->IncInterv().SetName("C2");
  md2->IncInterv().SetName("D2");

  mLInterv1.AddInterv(mEqP3I->IncInterv());

  mLInterv2.AddInterv(mW2->IncInterv());
  mLInterv2.AddInterv(mc2->IncInterv());
  mLInterv2.AddInterv(md2->IncInterv());
  mLInterv2.AddInterv(mEq2P3I->IncInterv());

  mVLInterv.push_back(&mLInterv1);
  mVLInterv.push_back(&mLInterv2);

  for (int aKI=0 ; aKI<int(mVEqSup.size()) ; aKI++)
  {
       cEqSupBB * aEBB = mVEqSup[aKI];
       aEBB->AllocInterv();
       mVLInterv.push_back(&(aEBB->mLInterv3));
  }


  if (DoGenCode)
  {
      double aFact = 1.0;
      std::vector<Fonc_Num> aVR1;
      std::vector<Fonc_Num> aVR2;
      std::vector<Fonc_Num> aVR3;
      {
          Pt3d<Fonc_Num>  aPIGlob = mEqP3I->PF();
          Pt3d<Fonc_Num> aP1 = aPIGlob ;
          aVR1.push_back(aFact*(mI1.PtF().x - aP1.x / aP1.z));
          aVR1.push_back(aFact*(mI1.PtF().y - aP1.y / aP1.z));
      }
      {
          Pt3d<Fonc_Num>  aPIGlob = mEq2P3I->PF();
          Pt3d<Fonc_Num> aP2 =  aPIGlob + (mW2->mP ^ aPIGlob) - mB2.PtF() - mC2.PtF()*mc2->mS - mD2.PtF()*md2->mS ;
          aVR2.push_back(aFact*(mI2.PtF().x - aP2.x / aP2.z));
          aVR2.push_back(aFact*(mI2.PtF().y - aP2.y / aP2.z));
      }

      cEqSupBB & aESBB = *(mVEqSup[0]);
      {
          Pt3d<Fonc_Num>  aPIGlob =  aESBB.mEq3P3I->PF();
          Pt3d<Fonc_Num> aP3 =  aPIGlob + (aESBB.mW3->mP ^ aPIGlob) - (aESBB.mC3->mP+ aESBB.mC3Init.PtF());
          aVR3.push_back(aFact*(aESBB.mI3.PtF().x - aP3.x / aP3.z));
          aVR3.push_back(aFact*(aESBB.mI3.PtF().y - aP3.y / aP3.z));
      }




      cElCompileFN::DoEverything
      (
          DIRECTORY_GENCODE_FORMEL,
          mNameEq1,
          aVR1,
          mLInterv1  ,
          mUseAccelCoordCste
      );
      cElCompileFN::DoEverything
      (
          DIRECTORY_GENCODE_FORMEL,
          mNameEq2,
          aVR2,
          mLInterv2  ,
          mUseAccelCoordCste
      );
      cElCompileFN::DoEverything
      (
          DIRECTORY_GENCODE_FORMEL,
          aESBB.mNameEq3,
          aVR3,
          aESBB.mLInterv3  ,
          mUseAccelCoordCste
      );




      return;
  }

  //mMapBufSub[3] = new cSubstitueBlocIncTmp(*mEqP3I); 
  for (int aFlag=0 ; aFlag<(1<<mVLInterv.size()) ; aFlag++)
  {
       if (NbBitsOfFlag(aFlag)>=2) 
       {
           cSubstitueBlocIncTmp * aBuf = new cSubstitueBlocIncTmp(*mEqP3I);
           mMapBufSub[aFlag] = aBuf;
           for (int aP=1,aK=0 ; aP<= aFlag  ; aP *=2,aK++)
           {
               if (aP& aFlag)
               {
                  aBuf->AddInc(*(mVLInterv[aK]));
               }
           }
           aBuf->Close();
       }
  }

  // Maintenant, sinon recouvrt avec Tmp

  mVFEsResid.push_back(cElCompiledFonc::AllocFromName(mNameEq1));
  ELISE_ASSERT( mVFEsResid.back() !=0,"Cannot allocate cGlobEqLineraiseAngle");
  mVFEsResid.back() ->SetMappingCur(mLInterv1,mSetEq);
  mI1.InitAdr(*mVFEsResid.back());


  mVFEsResid.push_back(cElCompiledFonc::AllocFromName(mNameEq2));
  ELISE_ASSERT(mVFEsResid.back()!=0,"Cannot allocate cGlobEqLineraiseAngle");
  mVFEsResid.back()->SetMappingCur(mLInterv2,mSetEq);

  mB2.InitAdr(*mVFEsResid.back());
  mC2.InitAdr(*mVFEsResid.back());
  mD2.InitAdr(*mVFEsResid.back());
  mI2.InitAdr(*mVFEsResid.back());



  for (int aK=0 ; aK<int(mVEqSup.size()); aK++)
  {
      mVEqSup[aK]->PostInit();
      mVFEsResid.push_back(mVEqSup[aK]->mEq);
  }

   //======================
  for (int aK=0 ; aK<int(mVFEsResid.size()); aK++)
  {
      mSetEq->AddFonct(mVFEsResid[aK]);
  }
  mSetEq->AddObj2Kill(this);


  mSetEq->SetClosed();
   // mSetEq.
}

const std::string & cEqBundleBase::NameEq1() const {return mNameEq1;}

     // =========================== GESTION ROTATION =================


void  cEqBundleBase::InitNewR2(const ElRotation3D & aRot)
{
     mSetEq->ResetUpdate(1.0);
     mCurRot  = aRot;
     for (int aK=0 ; aK<mSetEq->Alloc().CurInc() ; aK++)
         mSetEq->Alloc().SetVar(0,aK);
     mR2 = aRot.Mat();
     mW2->mP0 = Pt3dr(0,0,0);

     mB2Cur =  aRot.tr();
     Pt3dr aC2,aD2;
     MakeRONWith1Vect(mB2Cur,aC2,aD2);
     mB2.SetEtat(mB2Cur);

     mc2->mS0 = 0;
     mC2.SetEtat(aC2);
     md2->mS0 = 0;
     mD2.SetEtat(aD2);

     mSetEq->SetPhaseEquation();
}



void cEqBundleBase::InitNewR3(int aK,const ElRotation3D & aRot)
{
      mVEqSup[aK]->InitNewRot(aRot);
}

void     cEqBundleBase::InitNewR2R3(const ElRotation3D & aR2,const ElRotation3D & aR3)
{
    ELISE_ASSERT(mNbCamSup==1,"cEqBundleBase::InitNewRot");
    InitNewR2(aR2);
    InitNewR3(0,aR3);

}


void  cEqBundleBase::InitNewRot(const ElRotation3D & aRot)
{
    ELISE_ASSERT(mNbCamSup==0,"cEqBundleBase::InitNewRot");
    InitNewR2(aRot);
}


ElRotation3D  cEqBundleBase::SolveResetUpdate()
{
/*
    mSetEq->SolveResetUpdate();
    Pt3dr aNewB0 =  vunit(mB2.GetVal() + mC2.GetVal()*mc2->mS0 + mD2.GetVal()*md2->mS0);
    ElMatrix<double> aNewR = NearestRotation( gaussj(ElMatrix<double>(3,true)+MatProVect(mW2->mP0)) * mR2);
    return  ElRotation3D (aNewB0,aNewR,true);
*/
    ELISE_ASSERT(mNbCamSup==0,"cEqBundleBase::InitNewRot");
    return GenSolveResetUpdate()[0];
}


std::vector<ElRotation3D> cEqBundleBase::GenSolveResetUpdate()
{
    std::vector<ElRotation3D> aRes;
    mSetEq->SolveResetUpdate();
    Pt3dr aNewB0 =  mB2.GetVal() + mC2.GetVal()*mc2->mS0 + mD2.GetVal()*md2->mS0;
    double aLambda = euclid(aNewB0);
    aNewB0  = aNewB0 / aLambda;
    ElMatrix<double> aNewR = NearestRotation( gaussj(ElMatrix<double>(3,true)+MatProVect(mW2->mP0)) * mR2);
    aRes.push_back(ElRotation3D (aNewB0,aNewR,true));

    for (int aKEB=0 ; aKEB<int(mVEqSup.size()); aKEB++)
    {
       cEqSupBB & aESBB = *(mVEqSup[aKEB]);
       Pt3dr aNewB3 =  (aESBB.mC3Init.GetVal() + aESBB.mC3->mP0)/aLambda;
       ElMatrix<double> aNewR3 = NearestRotation( gaussj(ElMatrix<double>(3,true)+MatProVect(aESBB.mW3->mP0)) * aESBB.mR3);
       aRes.push_back(ElRotation3D (aNewB3,aNewR3,true));
       // (aESBB.mC3->mP+ aESBB.mC3Init.PtF())
    }


    return  aRes;
}


//  ====================== ADD EQUATIONS ===============================




double     cEqBundleBase::AddEquationGen(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel, double aPds,bool WithEq,bool BugTK)
{


   ELISE_ASSERT((2+mNbCamSup) ==  int(aVPts.size()),"cEqBundleBase::AddEquationGen");
   ELISE_ASSERT(aVSel.size() ==  aVPts.size(),"cEqBundleBase::AddEquationGen");

   int aFlag=0;

   std::vector<double> aVPds; // Deux poids, car deux mesures poru chaque camera
   double aRes = 0;
   static Pt3dr    aVDir[9];
   int aNbP=0;
   std::vector<Pt3dr> aVP0;
   std::vector<Pt3dr> aVP1;

   double aDMax2 = 0.0;
   if (aVSel[0])
   {
       aVDir[0] =  Pt3dr(aVPts[0].x,aVPts[0].y,1);
       ElSetMax(aDMax2,square_euclid(aVDir[0]));
       mI1.SetEtat(ProjStenope(aVDir[0]));
       aVP0.push_back(Pt3dr(0,0,0));
       aVP1.push_back(aVDir[0]);
       aNbP++;
   }

   if (aVSel[1])
   {
       aVDir[1] = mR2 * Pt3dr(aVPts[1].x,aVPts[1].y,1);
       ElSetMax(aDMax2,square_euclid(aVDir[1]));
       mI2.SetEtat(ProjStenope(aVDir[1]));
       aVP0.push_back(mB2Cur);
       aVP1.push_back(mB2Cur+aVDir[1]);
       aNbP++;
   }

   for (int aKEB=0 ; aKEB<int(mVEqSup.size()) ; aKEB++)
   {
       int aKIm = aKEB+2;
       if (aVSel[aKIm])
       {
           cEqSupBB & anESB = *(mVEqSup[aKEB]);
           aVDir[aKIm] = anESB.mR3 * Pt3dr(aVPts[aKIm].x,aVPts[aKIm].y,1);
           anESB.mI3.SetEtat(ProjStenope(aVDir[aKIm]));
           aVP0.push_back(anESB.mCurC3);
           aVP1.push_back(anESB.mCurC3+aVDir[aKIm]);
           aNbP++;
       }
   }

   ELISE_ASSERT(aVP0.size()>=2,"cEqBundleBase::AddEquationGen");
   // double aDist;
   mLastI = InterSeg(aVP0,aVP1,mOkLastI);
   if ((! mOkLastI) || (ElAbs(mLastI.z) > 1e9))
   {
      return 0;
   }

   aPds /= aDMax2;
   aVPds.push_back(aPds);
   aVPds.push_back(aPds);

   mEqP3I->InitEqP3iVal(mLastI);

   for (int aK=0 ; aK<int (aVPts.size())  ; aK++)
   {
      if (aVSel[aK])
      {
         aFlag |= (1<<aK);
         mI1.SetEtat(ProjStenope(aVDir[aK]));
         const std::vector<REAL> & aVRES =      WithEq                                              ?
                                                mSetEq->VAddEqFonctToSys(mVFEsResid[aK],aVPds,false,NullPCVU) :
                                                mSetEq->VResiduSigne(mVFEsResid[aK])                 ;

         aRes += euclid(Pt2dr(aVRES[0],aVRES[1]));
      }
   }
   aRes /= aVPts.size();

   if (0)
   {
       // Ok
       std::cout <<  "RRRRRatioBBBB " << aRes / ProjCostMEP(mCurRot,aVPts[0],aVPts[1],-1) << "\n";
   }

   if (WithEq)
   {
      // if (aFlag== 3) mSBIT12->DoSubst();
      cSubstitueBlocIncTmp * aBuf = mMapBufSub[aFlag];
      ELISE_ASSERT(aBuf!=0," Flag in cEqBundleBase::AddEquationGen");
      aBuf->DoSubstBloc(NullPCVU);
   }

if (BugTK)
{
    for (int aK=0 ; aK<int(aVP0.size()) ; aK++)
    {
         ElSeg3D aSeg(aVP0[aK],aVP1[aK]);
          
         std::cout << "     GGGG " << aVP0[aK] << " " << aVP1[aK]  << " TN" << aSeg.TgNormee() << " D=" << aSeg.DistDoite(mLastI) << "\n";


         // InterSeg(aVP0[0],aVP1[0],aVP0[1],aVP1[1],Ok,0);

    }
    std::cout << "cEqBundleBase::AddEquationGen " << aRes << " " << mLastI << "\n";
}

   return aRes;
}




double    cEqBundleBase::AddEquation12Gen(const Pt2dr & aP1,const Pt2dr & aP2, double aPds,bool WithEq)
{
    std::vector<Pt2dr> aVPts;
    std::vector<bool>  aVSel;

    aVPts.push_back(aP1);
    aVPts.push_back(aP2);
    aVSel.push_back(true);
    aVSel.push_back(true);
    return AddEquationGen(aVPts,aVSel,aPds,WithEq);
}


double cEqBundleBase::AddEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel,double aPds)
{
   return AddEquationGen(aVPts,aVSel,aPds,true);
}
double  cEqBundleBase::ResiduEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel)
{
   return AddEquationGen(aVPts,aVSel,1.0,false);
}

double  cEqBundleBase::ResiduEquation12(const Pt2dr & aP1,const Pt2dr & aP2)
{
   return AddEquation12Gen(aP1,aP2,1.0,false);
}

double  cEqBundleBase::AddEquation12(const Pt2dr & aP1,const Pt2dr & aP2,double aPds)
{
   return AddEquation12Gen(aP1,aP2,aPds,true);
}

void GenCodecEqBundleBase()
{
    cEqBundleBase * anEla = new  cEqBundleBase (true,0,0.0,true);
    delete anEla;
    anEla = new  cEqBundleBase (true,0,0.0,false);
    delete anEla;
}


/************************************************************/
/*                                                          */
/*           cBundle3Image                                  */
/*                                                          */
/************************************************************/

typedef std::vector<std::vector<Pt2df> *> tMultiplePF;

class cPairB3Im
{
    public :
        cPairB3Im(const tMultiplePF  & aHom,int aIndA,int aIndB,int aIndC) ;
        tMultiplePF  mHoms;
        int          mIndA;
        int          mIndB;
        int          mIndC; // Exclus
        bool         mPairBugTK;
        int          mNb;
};

class cBundle3Image
{
     public :
         cBundle3Image
         (
               double               aFoc,
               const ElRotation3D & aR12,
               const ElRotation3D & aR13,
               const tMultiplePF  & aH123,
               const tMultiplePF  & aH12,
               const tMultiplePF  & aH13,
               const tMultiplePF  & aH23,
               double  aPds3
         );
        double SomNbPair() const;

        ElRotation3D RotOfInd(int anInd) const;
        ~cBundle3Image();
        double  RobustEr2Glob(double aProp);
        double  RobustEr3(double aProp);

        double OneIter3(double anErStd);
        double OneIter2Glob(double anErStd);


        double PdsErr(const double & anErr,const double & anErrStd) const
        {
           return  1.0 / (1 + ElSquare(anErr/( CoeffPdsErr *anErrStd)));
        }
 

        const cPairB3Im  &  P12() const {return mP12;}
        const cPairB3Im  &  P13() const {return mP13;}
        const cPairB3Im  &  P23() const {return mP23;}

        static const double PropErInit;
        static const double CoeffPdsErr;

        cEqBundleBase&   BB() {return   *mEqBB;}

        std::vector<double>  mXI;
        std::vector<double>  mYI;
        std::vector<double>  mZI;
   private :
        double OneIter2(const cPairB3Im &,double anErStd);

        double  RobustEr2(const cPairB3Im & aPair,double aProp);
        double  AddEq2Pt(const cPairB3Im &,int aK,double aPds);
        double  AddEq3Pt(int aK,double aPds,double * anAccumVerif=0,bool BugKT=false,int IndUnSel=-1);


        double             mFoc;
        double             mScale;
        ElRotation3D       mR12Init;
        ElRotation3D       mR13Init;
        ElRotation3D       mR12Cur;
        ElRotation3D       mR13Cur;
        tMultiplePF        mH123;
        int                mNb123;
        cPairB3Im          mP12;
        cPairB3Im          mP13;
        cPairB3Im          mP23;
        cEqBundleBase*     mEqBB;
        std::vector<Pt2dr> mBufPts;
        std::vector<bool>  mBufSel;
  //  "Sur ponderation" des triplets
        double               mPds3;
};


cPairB3Im::cPairB3Im(const tMultiplePF  & aHom,int aIndA,int aIndB,int aIndC) :
    mHoms      (aHom),
    mIndA      (aIndA),
    mIndB      (aIndB),
    mIndC      (aIndC),
    mPairBugTK (aIndC==0),
    mNb        ((int)aHom[0]->size())
{
}



const double cBundle3Image::PropErInit =  0.75;
const double cBundle3Image::CoeffPdsErr =  2.0;

cBundle3Image::~cBundle3Image()
{
   delete mEqBB;
}

// double     cEqBundleBase::AddEquationGen(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel, double aPds,bool WithEq)
cBundle3Image::cBundle3Image
(
     double aFoc,
     const ElRotation3D & aR12,
     const ElRotation3D & aR13,
     const tMultiplePF  & aH123,
     const tMultiplePF  & aH12,
     const tMultiplePF  & aH13,
     const tMultiplePF  & aH23,
     const double         aSurPds3
) :
   mFoc      (aFoc),
   mScale    (1.0/LongBase(aR12)),
   mR12Init  (ScaleBase(aR12,mScale)),
   mR13Init  (ScaleBase(aR13,mScale)),
   mR12Cur   (mR12Init),
   mR13Cur   (mR13Init),
   mH123     (aH123),
   mNb123    ((int)aH123[0]->size()),
   mP12      (aH12,0,1,2),
   mP13      (aH13,0,2,1),
   mP23      (aH23,1,2,0),
   mEqBB     (new cEqBundleBase(false,1,aFoc,true)),
   mBufPts   (3),
   mBufSel   (3),
   mPds3     (aSurPds3)
{
   mEqBB->InitNewR2R3(mR12Init,mR13Init);
   // mPds3 =  ElMin(MaxSurPond3,(mP12.mNb+mP13.mNb+mP23.mNb)/double(mNb123));
}

Pt2dr F2D(const Pt2df & aP) {return Pt2dr(aP.x,aP.y);}

double  cBundle3Image::AddEq3Pt(int aK,double aPds,double * anAccumVerif,bool BugKT,int anIndUnsel)
{
    mBufPts[0] =   F2D((*(mH123[0]))[aK]);
    mBufPts[1] =   F2D((*(mH123[1]))[aK]);
    mBufPts[2] =   F2D((*(mH123[2]))[aK]);
    mBufSel[0] = mBufSel[1] = mBufSel[2] = true;

    if (BugKT && (anIndUnsel>=0))
    {
       mBufSel[anIndUnsel] = false;
    }
    bool AddEq = aPds>0;
    double aRes =  mEqBB->AddEquationGen(mBufPts,mBufSel,AddEq?aPds:1.0,AddEq,BugKT);


    return aRes;
}
double  cBundle3Image::RobustEr3(double aProp)
{
     std::vector<double>  aVRes;
     for (int aK=0 ; aK<mNb123 ; aK++)
     {
          aVRes.push_back(AddEq3Pt(aK,0));
     }
     return KthValProp(aVRes,aProp);
}


double cBundle3Image::OneIter3(double anErrStd)
{
     mXI.clear();
     mYI.clear();
     mZI.clear();
     double aSomP = 0;
     double aSomEP = 0;
     double aSomVerif = 0;
     for (int aK=0 ; aK<mNb123 ; aK++)
     {
          if (0&& MPD_MM() && (aK==0))
          {
             for (int aInd=0 ; aInd<3 ; aInd++)
             {
                AddEq3Pt(aK,0,(double *)0,true,aInd);
             }
             AddEq3Pt(aK,0,(double *)0,true,-1);
          }
          double anErr = AddEq3Pt(aK,0);
          double aPds =  PdsErr(anErr,anErrStd);

          AddEq3Pt(aK,aPds*mPds3,&aSomVerif);
          aSomP += aPds;
          aSomEP += aPds * anErr;
          if (mEqBB->OkLastI())
          {
               Pt3dr aP = mEqBB->LastI();
               mXI.push_back(aP.x);
               mYI.push_back(aP.y);
               mZI.push_back(aP.z);

          }
     }
     return aSomEP / aSomP;
}


double  cBundle3Image::AddEq2Pt(const cPairB3Im & aP,int aK,double aPds)
{
    mBufPts[aP.mIndA] =   F2D((*(aP.mHoms[0]))[aK]);
    mBufPts[aP.mIndB] =   F2D((*(aP.mHoms[1]))[aK]);
    mBufSel[aP.mIndA] = true;
    mBufSel[aP.mIndB] = true;
    mBufSel[aP.mIndC] = false;
    bool AddEq = aPds>0;
    double aRes =  mEqBB->AddEquationGen(mBufPts,mBufSel,AddEq?aPds:1.0,AddEq);//  , (aK==0)  && (!AddEq));
    return aRes;
}
double  cBundle3Image::RobustEr2(const cPairB3Im & aPair,double aProp)
{
     std::vector<double>  aVRes;
     for (int aK=0 ; aK<aPair.mNb ; aK++)
     {
          aVRes.push_back(AddEq2Pt(aPair,aK,0));
     }
     return KthValProp(aVRes,aProp);
}

ElRotation3D cBundle3Image::RotOfInd(int anInd) const
{
    if (anInd==0) return  ElRotation3D::Id;
    if (anInd==1) return  mR12Cur;
    if (anInd==2) return  mR13Cur;

    ELISE_ASSERT(false,"Bundle3Image::RotOfIn");
    return ElRotation3D::Id;
}


double cBundle3Image::OneIter2(const cPairB3Im & aPair,double anErrStd)
{
     double aSomP = 0;
     double aSomEP = 0;
     for (int aK=0 ; aK<aPair.mNb ; aK++)
     {
          double anErr = AddEq2Pt(aPair,aK,0);
          double aPds =  PdsErr(anErr,anErrStd);
          AddEq2Pt(aPair,aK,aPds);
          aSomP += aPds;
          aSomEP += aPds * anErr;
     }
     return (aSomP==0) ? 0.0 : (aSomEP / aSomP);
}

double cBundle3Image::SomNbPair() const
{
   return ElMax(1.0,double(mP12.mNb + mP13.mNb + mP23.mNb));
}

double  cBundle3Image::RobustEr2Glob(double aProp)
{
    double aRes =    RobustEr2(mP12,aProp)  * mP12.mNb
                  +  RobustEr2(mP13,aProp)  * mP13.mNb
                  +  RobustEr2(mP23,aProp)  * mP23.mNb ;

    return aRes / SomNbPair();
}

double cBundle3Image::OneIter2Glob(double anErStd)
{
    double aRes =    OneIter2(mP12,anErStd)  * mP12.mNb
                  +  OneIter2(mP13,anErStd)  * mP13.mNb
                  +  OneIter2(mP23,anErStd)  * mP23.mNb ;

    return aRes / SomNbPair();
}

ElRotation3D PerturRot(const ElRotation3D & aR,double aMul)
{
    return ElRotation3D
           (
              aR.tr()+Pt3dr(NRrandC(),NRrandC(),NRrandC()) * aMul,
              aR.Mat() * ElMatrix<double>::Rotation(NRrandC()*aMul,NRrandC()*aMul,NRrandC()*aMul),
              true
           );
}

void TestBundle3Image
     (
          double               aFoc,
          const ElRotation3D & aR12,
          const ElRotation3D & aR13,
          const tMultiplePF  & aH123,
          const tMultiplePF  & aH12,
          const tMultiplePF  & aH13,
          const tMultiplePF  & aH23,
          double  aPds3
     )
{

    double aProp = cBundle3Image::PropErInit;
    for (int aK=0 ; aK<5 ; aK++)
    {
        double aMul = 1e-2*pow((float) 10,-aK);
        cBundle3Image aB3(aFoc,PerturRot(aR12,aMul),PerturRot(aR13,aMul),aH123,aH12,aH13,aH23,aPds3);

        double anEr3 = aB3.RobustEr3(aProp);
        double anEr2 = aB3.RobustEr2Glob(aProp);

        for (int anIter = 0 ; anIter<8 ; anIter ++)
        {
           anEr3 = aB3.OneIter3(anEr3); 
           anEr2 = aB3.OneIter2Glob(anEr2); 
           std::vector<ElRotation3D>  aVR = aB3.BB().GenSolveResetUpdate(); 
           aB3.BB().InitNewR2R3(aVR[0],aVR[1]);
           // std::cout << "Er3 " <<  anEr3*aFoc  << " Er2 " << anEr2*aFoc   << "\n";
        }


        std::cout << "  ======================================== \n\n";
        getchar();
    }
}

std::vector<ElRotation3D> VRotB3(const ElRotation3D & aR12,const ElRotation3D &aR13)
{
   std::vector<ElRotation3D> aRes;
   aRes.push_back(ElRotation3D::Id);
   aRes.push_back(aR12);
   aRes.push_back(aR13);

   return aRes;
}



bool SolveBundle3Image
     (
          double               aFoc,
          ElRotation3D & aR12,
          ElRotation3D & aR13,
          Pt3dr &        aPMed,
          double &       aBOnH,
          const tMultiplePF  & aH123,
          const tMultiplePF  & aH12,
          const tMultiplePF  & aH13,
          const tMultiplePF  & aH23,
          double  aPds3,
          cParamCtrlSB3I & aParam
     )
{
    aParam.mRes2 = DefBundle3Im;
    aParam.mRes3 = DefBundle3Im;
    if (LongBase(aR12)==0)
    {
       return false;
    }


    double aDefError = 1e5;

    double aProp = cBundle3Image::PropErInit;
    cBundle3Image aB3(aFoc,aR12,aR13,aH123,aH12,aH13,aH23,aPds3);

    double anEr3 = aParam.mFilterOutlayer ? aB3.RobustEr3(aProp)     : aDefError;
    double anEr2 = aParam.mFilterOutlayer ? aB3.RobustEr2Glob(aProp) : aDefError;

    for (int anIter = 0 ; anIter<aParam.mNbIter ; anIter ++)
    {
           anEr2 = aB3.OneIter2Glob(anEr2); 
           anEr3 = aB3.OneIter3(anEr3); 
           aParam.mRes3 = anEr3;
           aParam.mRes2 = anEr2;
           if ((anEr2<aParam.mResiduStop) && (anEr3<aParam.mResiduStop))
           {
              anIter = aParam.mNbIter;
           }

           std::vector<ElRotation3D>  aVR = aB3.BB().GenSolveResetUpdate(); 
           aB3.BB().InitNewR2R3(aVR[0],aVR[1]);
           aR12 = aVR[0];
           aR13 = aVR[1];
           if (!aParam.mFilterOutlayer)
           {
               anEr3 = aDefError;
               anEr2 = aDefError;
           }
    }
    //  Il doit y avoir plus propre mais je n'ai pas le courage d'intervenir dans le noyau
    for (int aK=0; aK<int(aB3.mXI.size()) ; aK++)
    {
       if (! (IsOkData(aB3.mXI[aK]) && IsOkData(aB3.mYI[aK]) && IsOkData(aB3.mZI[aK])))
          return false;
    }
    if (aParam.mNbIter==0)
    {
       anEr3 = aB3.OneIter3(anEr3); 
    }

    aPMed.x =  MedianeSup(aB3.mXI);
    aPMed.y =  MedianeSup(aB3.mYI);
    aPMed.z =  MedianeSup(aB3.mZI);

    aBOnH = 0.0; 
    double aDMax= 0.0;
    std::vector<Pt3dr> aVC;
    aVC.push_back(Pt3dr(0,0,0));
    aVC.push_back(aR12.tr());
    aVC.push_back(aR13.tr());

    for (int aK=0 ; aK<3 ; aK++)
    {
        Pt3dr aV1 = aVC[aK];
        Pt3dr aV2 = aVC[(aK+1)%3];
        double aDist = euclid(aV1-aV2);
        double  aBonHLoc = aDist/euclid(aPMed);
        aBOnH = ElMax(aBOnH,aBonHLoc);
        aDMax = ElMax(aDMax,aDist);
    }

    aR12.tr() =   aR12.tr() / aDMax;
    aR13.tr() =   aR13.tr() / aDMax;
    aPMed     =   aPMed / aDMax;
    return true;
}


class cFullBundleBase  :  public  cPackInPts2d,
                          public  cInterfBundle2Image
{
    public :
       cFullBundleBase(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCoordCste);

       const std::string & VIB2I_NameType() {return mBB.NameEq1();}
       double  VIB2I_PondK(const int & aK) const {return mVPds[aK];}
       double  VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const;
       double  VIB2I_AddObsK(const int & aK,const double & aPds) ;
       // void    VIB2I_InitNewRot(const ElRotation3D &aRot) {mBB.InitNewRot(aRot);}
       void    VIB2I_InitNewRot(const ElRotation3D &aRot) ;
       ElRotation3D    VIB2I_Solve() {return  mBB.SolveResetUpdate();}
     
    private  :

       cEqBundleBase  mBB;
       bool           mAddCstrDrone;
};

void    cFullBundleBase::VIB2I_InitNewRot(const ElRotation3D &aRot) 
{
    if (mAddCstrDrone) 
    {
       std::cout << "InitNewRotInitNewRot " << aRot.tr() << " " << VIB2I_NameType() << "\n";
    }
    mBB.InitNewRot(aRot);
}

cFullBundleBase::cFullBundleBase(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCoordCste) :
    cPackInPts2d          (aPack),
    cInterfBundle2Image   ((int)mVP1.size(),aFoc),
    mBB                   (false,0,aFoc,UseAccelCoordCste),
    mAddCstrDrone         (false && MPD_MM())
{
    if (mAddCstrDrone)
    {
        std::cout << "ADD CSTRE DRONE \n";
    }
/*
*/
}

double  cFullBundleBase::VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const
{
   return ProjCostMEP(aRot,mVP1[aK],mVP2[aK],-1);
}

double  cFullBundleBase::VIB2I_AddObsK(const int & aK,const double & aPds)
{
   return mBB.AddEquation12(mVP1[aK],mVP2[aK],aPds);
}

cInterfBundle2Image * cInterfBundle2Image::Bundle(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCoordCste)
{
   return new cFullBundleBase(aPack,aFoc,UseAccelCoordCste);
}

/*
*/

/************************************************************/
/*                                                          */
/*              cParamCtrlSB3I                              */
/*                                                          */
/************************************************************/

cParamCtrlSB3I::cParamCtrlSB3I(int aNbIter,bool FilterOutlayer,double aResStop) :
   mNbIter          (aNbIter),
   mResiduStop      (aResStop),
   mFilterOutlayer  (FilterOutlayer)
{
}


/************************************************************/
/*                                                          */
/*           "Mini"-Utilitaires                             */
/*                                                          */
/************************************************************/




/*************************************************************************/
/*                                                                       */
/*               TEST                                                    */
/*                                                                       */
/*************************************************************************/


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
