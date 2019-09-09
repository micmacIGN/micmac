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


extern bool AllowUnsortedVarIn_SetMappingCur;
/*
     cPt3dEEF et cScalEEF sont definie afin d'heriter de cElemEqFormelle, cObjFormel2Destroy
    Sinon, on aurai interet a les traiter directement dans cGlobEqLineraiseAngle
*/

cPt3dEEF::cPt3dEEF(cSetEqFormelles & aSet,const Pt3dr & aP0,bool HasValCste) :
          cElemEqFormelle (aSet,false),
          mP0     (aP0),
          mP      (aSet.Alloc().NewPt3("cPt3dEEF",mP0,HasValCste))
{
           CloseEEF();
           aSet.AddObj2Kill(this);
}


cScalEEF::cScalEEF(cSetEqFormelles & aSet,double aV0,bool HasValCste) :
          cElemEqFormelle (aSet,false),
          mS0     (aV0),
          mS      (aSet.Alloc().NewF("cScalEEF","Scale",&mS0,HasValCste))
{
           CloseEEF();
           aSet.AddObj2Kill(this);
}


/************************************************************/
/*                                                          */
/*           Equation Linearise de l'angle                  */
/*                                                          */
/************************************************************/

/*
class cPt3dEEF : public cElemEqFormelle,
                 public cObjFormel2Destroy
{
    public :
       Pt3dr             mP0;
       Pt3d<Fonc_Num>    mP;

       cPt3dEEF(cSetEqFormelles & aSet,const Pt3dr & aP0,bool HasValCste) :
          cElemEqFormelle (aSet,false),
          mP0     (aP0),
          mP      (aSet.Alloc().NewPt3(mP0,HasValCste))
       {
           CloseEEF();
           aSet.AddObj2Kill(this);
       }
};


class cScalEEF : public cElemEqFormelle,
                     public cObjFormel2Destroy
{
    public :
       double      mS0;
       Fonc_Num    mS;

       cScalEEF(cSetEqFormelles & aSet,double aV0,bool HasValCste) :
          cElemEqFormelle (aSet,false),
          mS0     (aV0),
          mS      (aSet.Alloc().NewF(&mS0,HasValCste))
       {
           CloseEEF();
           aSet.AddObj2Kill(this);
       }
};
*/



#define TypeSysLin cNameSpaceEqF::eSysPlein

class cGlobEqLineraiseAngle : public cNameSpaceEqF,
                              public cObjFormel2Destroy
{
    public :
       cGlobEqLineraiseAngle(bool doGenCode,bool UseAccCst0);
       void    GenCode();
       void    InitNewRot(const ElRotation3D & aRot);
       ElRotation3D SolveResetUpdate();
       Pt2dr    AddEquation(const Pt3dr & aP1,const Pt3dr & aP2,double aPds);
       // Couple d'angle
       Pt2dr  ResiduEquation(const Pt3dr & aP1,const Pt3dr & aP2);

       const std::string & NameType () const ;
    private :
       Pt2dr    AddEquationGen(const Pt3dr & aP1,const Pt3dr & aP2,double aPds,bool WithEq);

       ElMatrix<double> tR0;

       cSetEqFormelles * mSetEq;
       cPt3dEEF *   mW;
       cP3d_Etat_PhgrF   mB0;
       cScalEEF *    mc;
       cP3d_Etat_PhgrF   mC;
       cScalEEF *    md;
       cP3d_Etat_PhgrF   mD;

       cP3d_Etat_PhgrF   mQp1;  // tRot ^ U1
       cP3d_Etat_PhgrF   mQ2;

       std::vector<Fonc_Num> mVFRes;
       cElCompiledFonc *     mFoncEqResidu;
       cIncListInterv        mLInterv;
       std::string           mNameType;
       ElRotation3D          mCurRot;
};





void cGlobEqLineraiseAngle::InitNewRot(const ElRotation3D & aRot)
{
     mSetEq->ResetUpdate(1.0);
     mCurRot = aRot;
     for (int aK=0 ; aK<5 ; aK++)
         mSetEq->Alloc().SetVar(0,aK);

     tR0 = aRot.Mat().transpose();
     mW->mP0 = Pt3dr(0,0,0);

     Pt3dr aB0 = tR0 * aRot.tr();
     Pt3dr aC0,aD0;
     MakeRONWith1Vect(aB0,aC0,aD0);
     mB0.SetEtat(aB0);

     mc->mS0 = 0;
     mC.SetEtat(aC0);
     md->mS0 = 0;
     mD.SetEtat(aD0);


     mSetEq->SetPhaseEquation();
}




ElRotation3D cGlobEqLineraiseAngle::SolveResetUpdate()
{
    mSetEq->SolveResetUpdate();

    ElMatrix<double> aR0 = tR0.transpose();

    Pt3dr aNewB0 =  aR0*vunit(mB0.GetVal() + mC.GetVal()*mc->mS0 + mD.GetVal()*md->mS0);

    ElMatrix<double> aNewR = NearestRotation(aR0*(ElMatrix<double>(3,true)+MatProVect(mW->mP0)));
    return ElRotation3D (aNewB0,aNewR,true);
}
/*
*/

Pt2dr cGlobEqLineraiseAngle::AddEquationGen(const Pt3dr & aP1,const Pt3dr & aP2,double aPds,bool WithEq)
{
    mQp1.SetEtat(tR0*aP1);
    mQ2.SetEtat(aP2);

    std::vector<double> aVPds;
    aVPds.push_back(aPds);
    aVPds.push_back(aPds);
    const std::vector<REAL> & aResidu =      WithEq                                              ?
                                             mSetEq->VAddEqFonctToSys(mFoncEqResidu,aVPds,false,NullPCVU) :
                                             mSetEq->VResiduSigne(mFoncEqResidu)                 ;

    Pt2dr aResult(aResidu[0],aResidu[1]);
    if (0)
    {
         std::cout << "RRRRatioAngle "  << dist4(aResult) / PVCostMEP(mCurRot,aP1,aP2,-1) << "\n";
    }
    return aResult;
}

Pt2dr  cGlobEqLineraiseAngle::ResiduEquation(const Pt3dr & aP1,const Pt3dr & aP2)
{
       return AddEquationGen(aP1,aP2,1.0,false);
}

Pt2dr  cGlobEqLineraiseAngle::AddEquation(const Pt3dr & aP1,const Pt3dr & aP2,double aPds)
{
       return AddEquationGen(aP1,aP2,aPds,true);
}

Pt3d<Fonc_Num> vunit(const Pt3d<Fonc_Num> & aV) {return aV / sqrt(square_euclid(aV));}

cGlobEqLineraiseAngle::cGlobEqLineraiseAngle(bool doGenCode,bool UseAccelCste0) :
    tR0         (1,1),
    mSetEq      (new cSetEqFormelles(TypeSysLin)),
    mW          (new cPt3dEEF(*mSetEq,Pt3dr(0,0,0),UseAccelCste0)),
    mB0         ("VecB0"),
    mc          (new cScalEEF (*mSetEq,0,UseAccelCste0)),
    mC          ("VecC"),
    md          (new cScalEEF (*mSetEq,0,UseAccelCste0)),
    mD          ("VecD"),
    mQp1        ("Qp1"),
    mQ2         ("Q2"),
    mNameType   ("cEqLinariseAngle" + std::string(UseAccelCste0 ? "_AccelCsteCoord" : "")),
    mCurRot     (ElRotation3D::Id)
{


  AllowUnsortedVarIn_SetMappingCur = true;
  mW->IncInterv().SetName("Omega");
  mc->IncInterv().SetName("C");
  md->IncInterv().SetName("D");

  mLInterv.AddInterv(mW->IncInterv());
  mLInterv.AddInterv(mc->IncInterv());
  mLInterv.AddInterv(md->IncInterv());
  //    mBase->IncInterv().SetName("Base");


 //========================
   Pt3d<Fonc_Num>  aQ2 = mQ2.PtF();
   Pt3d<Fonc_Num>  aQp2 = aQ2 + (mW->mP^aQ2);  // Comme W0 = 0 et Q2^W |_ Q2 , pas necessair de normer
   Pt3d<Fonc_Num>  aQp1 = mQp1.PtF();

   Pt3d<Fonc_Num>  aBase = mB0.PtF() + mC.PtF() * mc->mS + mD.PtF() * md->mS;
   // Pt3d<Fonc_Num>  aQp2 = vunit(mQ2.PtF() + (mW.mP^mQ2.PtF()));

   Pt3d<Fonc_Num> aQp1VQp2 = vunit(aQp1 ^ aQp2) ^aBase;

   //  / 4 pour etre coherent avec PVCost, qui lui meme est/4 pour etre similaire a ProjCost et DisCost
   Fonc_Num aDet = Det(aQp1,aQp2,aBase) / 4.0;

   mVFRes.push_back(aDet/scal(aQp1VQp2,aQp1));
   mVFRes.push_back(aDet/scal(aQp1VQp2,aQp2));

   // Fonc_Num aRIm1 = aDet / scal(

   if (doGenCode)
   {
        cElCompileFN::DoEverything
        (
            DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
            mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
            mVFRes,  // expressions formelles
            mLInterv, // intervalle de reference
            UseAccelCste0
        );
       return;
   }
   mFoncEqResidu =  cElCompiledFonc::AllocFromName(mNameType);
   ELISE_ASSERT(mFoncEqResidu!=0,"Cannot allocate cGlobEqLineraiseAngle");
   mFoncEqResidu->SetMappingCur(mLInterv,mSetEq);


   mB0.InitAdr(*mFoncEqResidu);
   mC.InitAdr(*mFoncEqResidu);
   mD.InitAdr(*mFoncEqResidu);
   mQp1.InitAdr(*mFoncEqResidu);
   mQ2.InitAdr(*mFoncEqResidu);
   // mGPS.InitAdr(*mFoncEqResidu);
   mSetEq->AddFonct(mFoncEqResidu);
   mSetEq->AddObj2Kill(this);


    mSetEq->SetClosed();
   // mSetEq.
}

const std::string & cGlobEqLineraiseAngle::NameType () const
{
   return mNameType;
}

void GenCodeEqLinariseAngle()
{
    cGlobEqLineraiseAngle * anEla = new  cGlobEqLineraiseAngle (true,true);
    delete anEla;
     anEla = new  cGlobEqLineraiseAngle (true,false);
    delete anEla;
}

/************************************************************/
/*                                                          */
/*           Linearisation du determinant                   */
/*             cBundleIterLin                               */
/*                                                          */
/************************************************************/

// Equation initiale     [U1,Base, R U2] = 0
//      [U1,Base, R0 dR U2] = 0     R = R0 (Id+dR)    dR ~0  R = (Id + ^W) et W ~ 0
//   [tR0 U1, tR0 Base,U2 + W^U2] = 0 ,
//    tR0 Base = B0 +dB   est un vecteur norme, soit CD tq (B0,C,D) est un Base ortho norme;
//    tR0 U1 = U'1
//   [U'1 ,  B0 + c C + d D , U2 + W ^U2] = 0
//   (U1' ^ (B0 + c C + d D)) . (U2 + W ^U2) = 0
//   (U'1 ^B0  + c U'1^C + d U'1 ^D ) . (U2 + W ^ U2) = 0
//  En supprimant les termes en Wc ou Wd :
//   (U'1 ^ B0) .U2    +  c ((U'1^C).U2) + d ((U'1 ^D).U2)  + (U'1 ^ B0) . (W^U2)
//   (U'1 ^ B0) .U2    +  c ((U'1^C).U2) + d ((U'1 ^D).U2)  +  W.(U2 ^(U'1 ^ B0)) => Verifier Signe permut prod vect


/*
double cOldBundleIterLin::ErrMoy() const {return mSomErr/mSomPds;}

cOldBundleIterLin::cOldBundleIterLin(const ElRotation3D & aRot,const double & anErrStd):
    mRot     (aRot),
    mSysLin5 (5),
    tR0      (aRot.Mat().transpose()),
    mB0      (tR0 * aRot.tr()),
    mSomErr  (0),
    mSomPds  (0),
    mErrStd  (anErrStd)
{
    MakeRONWith1Vect(mB0,mC,mD);
    mSysLin5.GSSR_Reset(false);
}
void cOldBundleIterLin::AddObs(const Pt3dr & aQ1,const Pt3dr& aQ2,const double & aPds)
{
   double aCoef[5];
   Pt3dr aQp1 = tR0 * aQ1;
   Pt3dr aUp1VB0 = aQp1 ^ mB0;

   double aCste =  scal(aQ2,aUp1VB0);
   aCoef[0] = scal(aQ2,aQp1^mC);  // Coeff C
   aCoef[1] = scal(aQ2,aQp1^mD);  // Coeff D
   Pt3dr  a3Prod = aQ2 ^ aUp1VB0;
   aCoef[2] = a3Prod.x;
   aCoef[3] = a3Prod.y;
   aCoef[4] = a3Prod.z;

   mLastPdsCalc = aPds / (1+ElSquare(aCste/mErrStd));
        //    aPair.mLastPdsOfErr = aPds;

   mSomPds += mLastPdsCalc;
   mSomErr += mLastPdsCalc * ElSquare(aCste);

   mSysLin5.GSSR_AddNewEquation(mLastPdsCalc,aCoef,-aCste,0);
   mVRes.push_back(ElAbs(aCste));
}

ElRotation3D  cOldBundleIterLin::CurSol()
{
    Im1D_REAL8   aSol = mSysLin5.GSSR_Solve (0);
    double * aData = aSol.data();

    Pt3dr aNewB0 = mRot.Mat()  * vunit(mB0+mC*aData[0] + mD*aData[1]);

    ElMatrix<double> aNewR = NearestRotation(mRot.Mat() * (ElMatrix<double>(3,true) + MatProVect(Pt3dr(aData[2],aData[3],aData[4]))));
    return ElRotation3D (aNewB0,aNewR,true);
}
*/

    //   ============= cBundleIterLin  remplacera cOldeBundleIterLin des architecture OK


class cBundleIterLin
{
    public :

       cBundleIterLin();
       double AddObs(const Pt3dr & aQ1,const Pt3dr& aQ2,const double & aPds);
       // double  Error(const ElRotation3D &aRot,const Pt3dr & aQ1,const Pt3dr& aQ2);
       ElRotation3D CurSol();
       void InitNewRot(const ElRotation3D & aRot);

    // private :

       ElRotation3D  mRot;
       L2SysSurResol mSysLin5;
       ElMatrix<double> tR0;
       Pt3dr mB0;
       Pt3dr mC,mD;
};

ElRotation3D  cBundleIterLin::CurSol()
{
    Im1D_REAL8   aSol = mSysLin5.GSSR_Solve (0);
    double * aData = aSol.data();

    Pt3dr aNewB0 = mRot.Mat()  * vunit(mB0+mC*aData[0] + mD*aData[1]);

    ElMatrix<double> aNewR = NearestRotation(mRot.Mat() * (ElMatrix<double>(3,true) + MatProVect(Pt3dr(aData[2],aData[3],aData[4]))));
    return ElRotation3D (aNewB0,aNewR,true);
}



void cBundleIterLin::InitNewRot(const ElRotation3D & aRot)
{
   mRot = aRot;
   tR0 = aRot.Mat().transpose();
   mB0 = tR0 * aRot.tr();
   MakeRONWith1Vect(mB0,mC,mD);
   mSysLin5.GSSR_Reset(false);
}


cBundleIterLin::cBundleIterLin() :
    mRot     (ElRotation3D::Id),
    mSysLin5 (5),
    tR0      (1,1)
{
}

double cBundleIterLin::AddObs(const Pt3dr & aQ1,const Pt3dr& aQ2,const double & aPds)
{
   double aCoef[5];
   Pt3dr aQp1 = tR0 * aQ1;
   Pt3dr aUp1VB0 = aQp1 ^ mB0;

   double aCste =  scal(aQ2,aUp1VB0);
   aCoef[0] = scal(aQ2,aQp1^mC);  // Coeff C
   aCoef[1] = scal(aQ2,aQp1^mD);  // Coeff D
   Pt3dr  a3Prod = aQ2 ^ aUp1VB0;
   aCoef[2] = a3Prod.x;
   aCoef[3] = a3Prod.y;
   aCoef[4] = a3Prod.z;

   mSysLin5.GSSR_AddNewEquation(aPds,aCoef,-aCste,0);

   if (0)
   {
       std::cout << "RRRRatioBlin   " << LinearCostMEP(mRot,aQ1,aQ2,-1) / aCste << "\n";  // 1 ou -1
   }
   return aCste;
}


/*************************************************************************/
/*                                                                       */
/*               Interfaces  Bundle generiques                           */
/*                                                                       */
/*************************************************************************/


     // ======================================================
     // ===========   cInterfBundle2Image ====================
     // ======================================================

cInterfBundle2Image::cInterfBundle2Image(int aNbCple,double aFoc) :
   mNbCple (aNbCple),
   mFoc    (aFoc)
{
}

cInterfBundle2Image::~cInterfBundle2Image()
{
}

double cInterfBundle2Image::ErrInitRobuste(const ElRotation3D &aRot,double aProp)
{
  std::vector<double> aVRes;
  for (int aK=0 ; aK< mNbCple ; aK++)
      aVRes.push_back(VIB2I_ErrorK(aRot,aK));
  return KthValProp(aVRes,aProp);
}


void cInterfBundle2Image::OneIterEqGen(const  ElRotation3D &aRot,double & anErrStd,bool AddEq)
{
  VIB2I_InitNewRot(aRot);
  double aSomErr = 0;
  double aSomPds = 0;

  for (int aK=0 ; aK< mNbCple ; aK++)
  {
       double anErr =  VIB2I_ErrorK(aRot,aK);
       double aPds = VIB2I_PondK(aK) / (1 + ElSquare(anErr/(2.0*anErrStd)));
       if (AddEq)
          VIB2I_AddObsK(aK,aPds);
//   std::cout << "EEEEE " << anErr << " " << aE2 << "\n";
       aSomErr += aPds * ElSquare(anErr);
       aSomPds += aPds;
  }

  aSomErr /= aSomPds;
  aSomErr = sqrt(aSomErr);
  anErrStd = aSomErr;
}

ElRotation3D cInterfBundle2Image::OneIterEq(const  ElRotation3D &aRot,double & anErrStd)
{
  OneIterEqGen(aRot,anErrStd,true);
  return VIB2I_Solve() ;
}

double cInterfBundle2Image::ResiduEq(const  ElRotation3D &aRot,const double & anErrStd)
{
    double anErrOut = anErrStd;
    OneIterEqGen(aRot,anErrOut,false);
    return anErrOut;
}

     // ==========================================================
     // ==================   cPackInPts3d  =======================
     // ==================   cPackInPts2d  =======================
     // ==========================================================


cPackInPts3d::cPackInPts3d(const  ElPackHomologue & aPack)
{
           for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
           {
                mVP1.push_back(vunit(PZ1(itP->P1())));
                mVP2.push_back(vunit(PZ1(itP->P2())));
                mVPds.push_back(itP->Pds());
           }
}

cPackInPts2d::cPackInPts2d(const  ElPackHomologue & aPack)
{
           for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
           {
                mVP1.push_back(itP->P1());
                mVP2.push_back(itP->P2());
                mVPds.push_back(itP->Pds());
           }
}



     // ==================================================================
     // ==================   cFullEqLinariseAngle  =======================
     // ==================================================================

class cFullEqLinariseAngle  :  public cPackInPts3d,
                               public cInterfBundle2Image
{
    public :
       cFullEqLinariseAngle(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCste0);

       const std::string & VIB2I_NameType() {return mELA.NameType();}
       double  VIB2I_PondK(const int & aK) const {return mVPds[aK];}
       double  VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const;
       double  VIB2I_AddObsK(const int & aK,const double & aPds) ;
       void    VIB2I_InitNewRot(const ElRotation3D &aRot) {mELA.InitNewRot(aRot);}
       ElRotation3D    VIB2I_Solve() {return  mELA.SolveResetUpdate();}
    private  :

       cGlobEqLineraiseAngle  mELA;
};


cFullEqLinariseAngle::cFullEqLinariseAngle(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCste0) :
   cPackInPts3d           (aPack),
   cInterfBundle2Image  ((int)mVP1.size(),aFoc),
   mELA                 (false,UseAccelCste0)
{
}

           //   Allocateur static

cInterfBundle2Image * cInterfBundle2Image::LineariseAngle(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCste0)
{
   return new cFullEqLinariseAngle(aPack,aFoc,UseAccelCste0);
}

double  cFullEqLinariseAngle::VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const
{
    return PVCostMEP(aRot,mVP1[aK],mVP2[aK],-1);
}

double  cFullEqLinariseAngle::VIB2I_AddObsK(const int & aK,const double & aPds)
{
    double aRes =  dist4(mELA.AddEquation(mVP1[aK],mVP2[aK],aPds));

    return aRes;
}

     // ============================================================
     // ==================   cFullBundleLin  =======================
     // ============================================================

class cFullBundleLin  :  public cPackInPts3d,
                         public cInterfBundle2Image
{
    public :
       cFullBundleLin(const  ElPackHomologue & aPack,double aFoc);

       const std::string & VIB2I_NameType() {return TheName;}
       double  VIB2I_PondK(const int & aK) const {return mVPds[aK];}
       double  VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const;
       double  VIB2I_AddObsK(const int & aK,const double & aPds) ;
       void    VIB2I_InitNewRot(const ElRotation3D &aRot) {mBIL.InitNewRot(aRot);}
       ElRotation3D    VIB2I_Solve() {return  mBIL.CurSol();}
    private  :

       static const std::string TheName;
       cBundleIterLin  mBIL;
};

const std::string cFullBundleLin::TheName = "cFullBundleLin";

cFullBundleLin::cFullBundleLin(const  ElPackHomologue & aPack,double aFoc) :
   cPackInPts3d           (aPack),
   cInterfBundle2Image  ((int)mVP1.size(),aFoc),
   mBIL                 ()
{
}

       //   Allocateur static

cInterfBundle2Image * cInterfBundle2Image::LinearDet(const  ElPackHomologue & aPack,double aFoc)
{
   return new cFullBundleLin(aPack,aFoc);
}

double  cFullBundleLin::VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const
{
   return LinearCostMEP(aRot,mVP1[aK],mVP2[aK],-1);
}

double  cFullBundleLin::VIB2I_AddObsK(const int & aK,const double & aPds)
{
   double aRes =  mBIL.AddObs(mVP1[aK],mVP2[aK],aPds);

   //  std::cout << "RRRBLLL " << aRes /  VIB2I_ErrorK(mBIL.mRot,aK) << "\n"; 1 et - 1
   return aRes;
}





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
