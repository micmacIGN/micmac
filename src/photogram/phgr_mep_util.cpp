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
  On ne gagne rien !!!
Time Euclid 1.42467
Time QEuclid 2.75878
Time 48Euclid 1.47062
*/
#define NBTabQE  100

double QuickDistEuclid(double anX,double anY)
{
    static double Tab[NBTabQE+2];
    static bool  First = true;
    if (First)
    {
       First = false;
       for (int aK=0 ; aK<= (NBTabQE+1) ; aK++)
       {
           Tab[aK] = euclid(Pt2dr(1.0,aK/double(NBTabQE)));
       }
    }



    anX = ElAbs(anX);
    anY = ElAbs(anY);
    if (anX < anY) ElSwap(anX,anY);

    if (anX==0) return 0;
    double aRatio = (anY/anX) * NBTabQE ;

    int anInd = round_down(aRatio);
    double aPds1 = aRatio - anInd;

    return anX * ((1-aPds1) * Tab[anInd] + aPds1 * Tab[anInd+1]);
}

void TestQE()
{
   std::vector<Pt2dr> aVP;
   for (int aK=0 ; aK< 20 ; aK++)
   {
       Pt2dr aP(NRrandC()*10,NRrandC()*10);
       std::cout << "DIST : " << euclid(aP) << " " << QuickDistEuclid(aP.x,aP.y) << " " << dist48_euclid(aP) << "\n";
       aVP.push_back(aP);
   }

   ElTimer aChrono;
   int aNb=1e7;
   double aS1=0;
   for (int aK=0 ; aK<aNb ; aK++)
       for (int aP=0 ; aP<int(aVP.size()) ; aP++)
          aS1+=euclid(aVP[aP]);
   std::cout << "Time Euclid " << aChrono.uval() << "\n";

   double aS2=0;
   ElTimer aChrono2;
   for (int aK=0 ; aK<aNb ; aK++)
       for (int aP=0 ; aP<int(aVP.size()) ; aP++)
          aS2+=QuickDistEuclid(aVP[aP].x,aVP[aP].y);
   std::cout << "Time QEuclid " << aChrono2.uval() << "\n";

   ElTimer aChrono3;
   double aS3=0;
   for (int aK=0 ; aK<aNb ; aK++)
       for (int aP=0 ; aP<int(aVP.size()) ; aP++)
          aS3+=dist48_euclid(aVP[aP]);
   std::cout << "Time 48Euclid " << aChrono3.uval()  << "\n";
   std::cout << "Soms " << aS1 << " " << aS2 << " " << aS3  << "\n";

}


double NewExactCostMEP(Pt3dr &  anI1,const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) ;


Pt3dr InterSeg(const Pt3dr & aP0,const Pt3dr & aP1,const Pt3dr & aQ0,const Pt3dr & aQ1,bool & Ok,double * aSqD)
{
/*
TIMME :
     3.03199   ElSeg3D::L2InterFaisceaux
     0.473224   InterSeg(const Pt3dr & aP0,...
     1.23799    InterSeg(const std::vector<Pt3r>

*/

     // (P0 + p (P1-P0) ) - (Q0+ q(Q1-Q0)) .  (P1-P0) = 0
     // (P0 + p (P1-P0) ) - (Q0+ q(Q1-Q0)) .  (Q1-Q0) = 0

      Pt3dr aP01 = aP1 -aP0;
      Pt3dr aQ01 = aQ1 -aQ0;
      // Normale

      Pt3dr aPQ = aQ0 - aP0;

      //    a p + b q = X
      //    c p + d q = Y

      double a = scal(aP01,aP01);
      double b = -scal(aQ01,aP01);
      double c = -b;
      double d = -scal(aQ01,aQ01);
      double X = scal(aPQ,aP01);
      double Y = scal(aPQ,aQ01);


     //  p      (d  -b)  X
     //  q  =   (-c  a)  Y


      double aDet = a*d - b*c;

      if (aDet==0)
      {
          Ok= false;
          return Pt3dr(0,0,0);
      }
      Ok = true;


      double p = ( d * X - b * Y ) / aDet;
      double q = (-c * X + a * Y ) / aDet;

      Pt3dr aPInt = aP0 + aP01 * p;
      Pt3dr aQInt =  aQ0 +aQ01 * q;

      if (aSqD) *aSqD = square_euclid(aPInt-aQInt);

      return (aPInt + aQInt) / 2.0;

}

Pt3dr InterSeg(const ElRotation3D & aR2to1 ,const Pt3dr & aQ1,const Pt3dr & aQ2,bool & Ok,double * aSquareD)
{
    Pt3dr aBase = aR2to1.tr();

    return InterSeg(Pt3dr(0,0,0),aQ1,aBase,aBase+ aR2to1.Mat()*aQ2,Ok,aSquareD);
}

Pt3dr InterSeg(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,bool & Ok,double * aSquareD)
{
    return InterSeg(aR2to1,PZ1(aP1),PZ1(aP2),Ok,aSquareD);
}

/*
      droite D : P U   ,  soit V W tq (UVW) soit une BON
      d^2 (D,Q) =  [(Q-P) .V] ^2 +  [(Q-P) .W] ^2
                = t(Q-P) [ U tU + W tW] (Q-P)
                =   t(Q -P) [ I - tU U]  (Q-P)

     Maintenta Uk Vk Wk ....

      Mk = I - tUk Uk / ||Uk|| ^2

      Som(Mk) Q = Som ( Mk Pk)

*/

Pt3dr InterSeg(const std::vector<Pt3dr> & aVP0, const std::vector<Pt3dr> & aVP1,bool & Ok)
{
    if (aVP0.size()==2)
       return InterSeg(aVP0[0],aVP1[0],aVP0[1],aVP1[1],Ok,0);
    Ok = true ;  // FAUX => A CHANGER !!!!! 
    static Im2D_REAL8 aImMat(3,3);
    static double ** aDM = aImMat.data();
    static double & aM00 = aDM[0][0];
    static double & aM11 = aDM[1][1];
    static double & aM22 = aDM[2][2];
    static double & aM10 = aDM[0][1];
    static double & aM20 = aDM[0][2];
    static double & aM21 = aDM[1][2];
    aM00  = aM11 = aM22 =  aM10 = aM20 = aM21 = 0;

    static double aV[3];
    aV[0] = aV[1] = aV[2]=0;

    int aNb = aVP0.size();

    for (int aK=0 ; aK<aNb  ; aK++)
    {
        const Pt3dr & aP0 = aVP0[aK];
        Pt3dr aU = (aVP1[aK]-aP0);
        double aX2 = ElSquare(aU.x);
        double aY2 = ElSquare(aU.y);
        double aZ2 = ElSquare(aU.z);
        double aU2 = aX2 + aY2 + aZ2;

        double mxx = 1 -  aX2 / aU2;
        double myy = 1 -  aY2 / aU2;
        double mzz = 1 -  aZ2 / aU2;
        double mxy = - (aU.x * aU.y) / aU2;
        double mxz = - (aU.x * aU.z) / aU2;
        double myz = - (aU.y * aU.z) / aU2;

        aM00 += mxx;
        aM11 += myy;
        aM22 += mzz;
        aM10 += mxy;
        aM20 += mxz;
        aM21 += myz;

        aV[0] +=  mxx * aP0.x + mxy  * aP0.y + mxz * aP0.z;
        aV[1] +=  mxy * aP0.x + myy  * aP0.y + myz * aP0.z;
        aV[2] +=  mxz * aP0.x + myz  * aP0.y + mzz * aP0.z;
    }
    cMSymCoffact3x3<double>  aMCF(aDM);

    return aMCF.CoffVecInv(aV);
}






#define TypeSysLin cNameSpaceEqF::eSysPlein
// #define TypeSysLin cNameSpaceEqF::eSysL2BlocSym



extern bool AllowUnsortedVarIn_SetMappingCur;

/*
     cPt3dEEF et cScalEEF sont definie afin d'heriter de cElemEqFormelle, cObjFormel2Destroy
    Sinon, on aurai interet a les traiter directement dans cGlobEqLineraiseAngle
*/

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

/************************************************************/
/*                                                          */
/*                cBundleOptim                              */
/*                                                          */
/************************************************************/

/*
class cBundleOptim
{
      public :

            cBundleOptim();
     // Retourne le residu ;
     // Si err < 0, fait juste le calcul avec un estimation robuste
     // Sinon mets a jour et estime au moindre carre pondere en fonction de Err
           double AddPackAndSolve(const ElPackHomologue &  aPack,const ElRotation3D  & aRotB,double anErr) ;


     // Derniere solution calculee
           virtual ElRotation3D  VBO_CurSol() const = 0 ;
           virtual double  VBO_AddEquation12(const Pt2dr & aP1,const Pt2dr & aP2,double aPds) = 0;
           virtual double  VBO_ResiduEquation12(const Pt2dr & aP1,const Pt2dr & aP2) = 0;
           virtual void    VBO_InitNewRot(const ElRotation3D & aRot) = 0;
           virtual void    VBO_SolveResetUpdate() = 0;

      protected :
           bool mPhaseEq  ;
           virtual void    VBO_SetPhaseEquation() = 0;
};


cBundleOptim:: cBundleOptim() :
    mPhaseEq (false)
{
}

double cBundleOptim::AddPackAndSolve(const ElPackHomologue &  aPack,const ElRotation3D  & aRotB,double anErr)
{
    if ((! mPhaseEq)  && (anErr>0))
    {
       mPhaseEq = true;
       VBO_SetPhaseEquation();
    }
    double aSomPds=0;
    double aSomErr=0;
    VBO_InitNewRot(aRotB);
    std::vector<double> aVRes;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aRes = VBO_ResiduEquation12(itP->P1(),itP->P2());
         if (anErr >0)
         {
             double aPds = itP->Pds() / (1+ElSquare(aRes/anErr));
             VBO_AddEquation12(itP->P1(),itP->P2(),aPds);
             aSomPds += aPds;
             aSomErr += aPds * ElSquare(aRes);
         }
         else
            aVRes.push_back(aRes);
    }

    if (anErr >0)
    {
       VBO_SolveResetUpdate();
       mPhaseEq = false;
       return sqrt(aSomErr / aSomPds);
    }
    return KthValProp(aVRes,ThePropERRInit);
}
*/


/************************************************************/
/*                                                          */
/*            Bundle classique                              */
/*                                                          */
/************************************************************/

/*
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

class cEqBundleBase  : public cNameSpaceEqF,
                       public cObjFormel2Destroy
{
    public :
       cEqBundleBase(bool DoGenCode,int aNbCamSup,double aFoc,bool UseAccelCoordCste = false);  // Nb de cam en plus des 2 minim
       void    InitNewRot(const ElRotation3D & aRot);
       ElRotation3D SolveResetUpdate();
       double AddEquation12(const Pt2dr & aP1,const Pt2dr & aP2,double aPds);
       double ResiduEquation12(const Pt2dr & aP1,const Pt2dr & aP2);
       ElRotation3D  CurSol() const;

       double AddEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel,double aPds);
       double ResiduEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel);
       const std::string & NameEq1() const;
    protected :
       // virtual Pt2dr    AddEquationGen(const Pt3dr & aP1,const Pt3dr & aP2,double aPds,bool WithEq) = 0;

       double    AddEquationGen(const std::vector<Pt2dr> & aP2,const std::vector<bool> & aVSel, double aPds,bool WithEq);
       double    AddEquation12Gen(const Pt2dr & aP1,const Pt2dr & aP2, double aPds,bool WithEq);

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

       cIncListInterv        mLInterv1;
       cIncListInterv        mLInterv2;
       int                   mNbCamSup;
       std::string           mPostAccel;
       std::string           mNameEq1;
       std::string           mNameEq2;

       std::vector<cElCompiledFonc *>     mVFEsResid;
       cEqfP3dIncTmp *       mEqP3I;
       cEqfP3dIncTmp *       mEq2P3I;
       cSubstitueBlocIncTmp  mSBIT12;
       ElRotation3D          mCurRot;
};




cEqBundleBase::cEqBundleBase(bool DoGenCode,int aNbCamSup,double aFoc,bool UseAccelCoordCste) :
    mR2         (1,1),
    mFoc        (aFoc),
    mSetEq      (new cSetEqFormelles(TypeSysLin)),
    mSetEq2     ( DoGenCode ? new cSetEqFormelles(TypeSysLin) : mSetEq),
    mW2         (new cPt3dEEF(*mSetEq2,Pt3dr(0,0,0),UseAccelCoordCste)),
    mB2         ("VecB2"),
    mc2         (new cScalEEF (*mSetEq2,0,UseAccelCoordCste)),
    mC2         ("VecC2"),
    md2         (new cScalEEF (*mSetEq2,0,UseAccelCoordCste)),
    mD2         ("VecD2"),
    mI1         ("I1"),
    mI2         ("I2"),
    mNbCamSup   (aNbCamSup),
    mPostAccel   (UseAccelCoordCste ? "_AccelCsteCoord" : ""),
    mNameEq1    ("cEqBBCamFirst" + mPostAccel),
    mNameEq2    ("cEqBBCamSecond" + mPostAccel),
    mEqP3I      (mSetEq->Pt3dIncTmp()),
    mEq2P3I     (DoGenCode ? mSetEq2->Pt3dIncTmp() : mEqP3I ),
    mSBIT12     (*mEqP3I),
    mCurRot     (ElRotation3D::Id)
{
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


  if (DoGenCode)
  {
      double aFact = 1.0;
      std::vector<Fonc_Num> aVR1;
      std::vector<Fonc_Num> aVR2;
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
      cElCompileFN::DoEverything
      (
          DIRECTORY_GENCODE_FORMEL,
          mNameEq1,
          aVR1,
          mLInterv1  ,
          UseAccelCoordCste
      );
      cElCompileFN::DoEverything
      (
          DIRECTORY_GENCODE_FORMEL,
          mNameEq2,
          aVR2,
          mLInterv2  ,
          UseAccelCoordCste
      );
      return;
  }

  // Buf de subst d'inconnues
  mSBIT12.AddInc(mLInterv1);
  mSBIT12.AddInc(mLInterv2);
  mSBIT12.Close();

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
   //======================

  mSetEq->AddFonct(mVFEsResid[0]);
  mSetEq->AddFonct(mVFEsResid[1]);
  mSetEq->AddObj2Kill(this);


  mSetEq->SetClosed();
   // mSetEq.
}

const std::string & cEqBundleBase::NameEq1() const {return mNameEq1;}

     // =========================== GESTION ROTATION =================


void  cEqBundleBase::InitNewRot(const ElRotation3D & aRot)
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



ElRotation3D  cEqBundleBase::SolveResetUpdate()
{
    // std::cout << " SOLLLL C2 " << mc2->mS0  << " D2 " << md2->mS0 << "\n";
    // std::cout << " SOL W2 " << mW2->mP0 << "\n";

    mSetEq->SolveResetUpdate();
    Pt3dr aNewB0 =  vunit(mB2.GetVal() + mC2.GetVal()*mc2->mS0 + mD2.GetVal()*md2->mS0);
    ElMatrix<double> aNewR = NearestRotation( gaussj(ElMatrix<double>(3,true)+MatProVect(mW2->mP0)) * mR2);
    return  ElRotation3D (aNewB0,aNewR,true);
}


//  ====================== ADD EQUATIONS ===============================




double     cEqBundleBase::AddEquationGen(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel, double aPds,bool WithEq)
{
   int aFlag=0;

   std::vector<double> aVPds; // Deux poids, car deux mesures poru chaque camera
   aVPds.push_back(aPds);
   aVPds.push_back(aPds);
   double aRes = 0;
   static Pt3dr    aVDir[9];
   int aNbP=0;
   std::vector<Pt3dr> aVP0;
   std::vector<Pt3dr> aVP1;

   if (aVSel[0])
   {
       aVDir[0] =  Pt3dr(aVPts[0].x,aVPts[0].y,1);
       mI1.SetEtat(ProjStenope(aVDir[0]));
       aVP0.push_back(Pt3dr(0,0,0));
       aVP1.push_back(aVDir[0]);
       aNbP++;
   }

   if (aVSel[1])
   {
       aVDir[1] = mR2 * Pt3dr(aVPts[1].x,aVPts[1].y,1);
       mI2.SetEtat(ProjStenope(aVDir[1]));
       aVP0.push_back(mB2Cur);
       aVP1.push_back(mB2Cur+aVDir[1]);
       aNbP++;
   }

   ELISE_ASSERT(aVP0.size()>=2,"cEqBundleBase::AddEquationGen");
   // double aDist;
   bool OkIS;
   Pt3dr aP = InterSeg(aVP0,aVP1,OkIS);
   if ((! OkIS) || (ElAbs(aP.z) > 1e9))
   {
      return 0;
   }



   mEqP3I->InitVal(aP);

   for (int aK=0 ; aK<int (aVPts.size())  ; aK++)
   {
      aFlag |= (1<<aK);
      if (aVSel[aK])
      {
         mI1.SetEtat(ProjStenope(aVDir[aK]));
         const std::vector<REAL> & aVRES =      WithEq                                              ?
                                                mSetEq->VAddEqFonctToSys(mVFEsResid[aK],aVPds,false) :
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
      if (aFlag== 3) mSBIT12.DoSubst();
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
/*           Equation Linearise de l'angle                  */
/*                                                          */
/************************************************************/



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
                                             mSetEq->VAddEqFonctToSys(mFoncEqResidu,aVPds,false) :
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


class  cPackInPts3d
{
     public :
       cPackInPts3d(const  ElPackHomologue & aPack)
       {
           for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
           {
                mVP1.push_back(vunit(PZ1(itP->P1())));
                mVP2.push_back(vunit(PZ1(itP->P2())));
                mVPds.push_back(itP->Pds());
           }
       }
    protected :
       std::vector<Pt3dr> mVP1;
       std::vector<Pt3dr> mVP2;
       std::vector<double> mVPds;
};

class  cPackInPts2d
{
     public :
       cPackInPts2d(const  ElPackHomologue & aPack)
       {
           for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
           {
                mVP1.push_back(itP->P1());
                mVP2.push_back(itP->P2());
                mVPds.push_back(itP->Pds());
           }
       }
    protected :
       std::vector<Pt2dr> mVP1;
       std::vector<Pt2dr> mVP2;
       std::vector<double> mVPds;
};




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
   cInterfBundle2Image  (mVP1.size(),aFoc),
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
   cInterfBundle2Image  (mVP1.size(),aFoc),
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


// cEqBundleBase::cEqBundleBase(bool DoGenCode,int aNbCamSup,double aFoc,bool UseAccelCoordCste) :

     // ============================================================
     // ==================   cFullBundleBase =======================
     // ============================================================

class cFullBundleBase  :  public  cPackInPts2d,
                          public  cInterfBundle2Image
{
    public :
       cFullBundleBase(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCoordCste);

       const std::string & VIB2I_NameType() {return mBB.NameEq1();}
       double  VIB2I_PondK(const int & aK) const {return mVPds[aK];}
       double  VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const;
       double  VIB2I_AddObsK(const int & aK,const double & aPds) ;
       void    VIB2I_InitNewRot(const ElRotation3D &aRot) {mBB.InitNewRot(aRot);}
       ElRotation3D    VIB2I_Solve() {return  mBB.SolveResetUpdate();}
    private  :

       cEqBundleBase  mBB;
};

cFullBundleBase::cFullBundleBase(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCoordCste) :
    cPackInPts2d          (aPack),
    cInterfBundle2Image   (mVP1.size(),aFoc),
    mBB                   (false,0,aFoc,UseAccelCoordCste)
{
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
/*           "Mini"-Utilitaires                             */
/*                                                          */
/************************************************************/

void InitPackME
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr>  &aVP2,
          std::vector<double>  &aVPds,
          const  ElPackHomologue & aPack
     )
{
   for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
   {
      aVP1.push_back(itP->P1());
      aVP2.push_back(itP->P2());
      aVPds.push_back(itP->Pds());
   }
}

//  Formule exacte et programmation simple et claire pour bench, c'est l'angle

// tr + R * M  , tR * (M-tr)

  //   Mesure d'erreur des intersections de faisceaux, "nouvelle" normalisation : c'est la  moyenne des distance de reproj en pixel
  //  C2 => C1 pour tous
  //
  //    ProjCostMEP         => reprojection en pixel
  //    DistDroiteCostMEP   => calcul des intersectio et angles en 3D
  //    PVCostMEP           => calcul par la formule du produit vectoriel
  //    LinearCostMEP       => calcul par le determinant, la moins precise mais la plus rapide

double ProjCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aR2to1.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aR2to1.tr();
   bool Ok;
   Pt3dr anI1 = InterSeg(Pt3dr(0,0,0),aQ1,aBase,aBase+aQ2,Ok);
   if (! Ok) return aTetaMax;

   Pt3dr anI2 =  aR2to1.ImRecAff(anI1);

   Pt2dr anEc1  = aP1 - ProjStenope(anI1);
   Pt2dr anEc2  = aP2 - ProjStenope(anI2);

   return  GenCoutAttenueTetaMax((euclid(anEc1) + euclid(anEc2)) / 2.0  ,aTetaMax);
}

double QuickD48EProjCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aR2to1.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aR2to1.tr();
   bool Ok;
   Pt3dr anI1 = InterSeg(Pt3dr(0,0,0),aQ1,aBase,aBase+aQ2,Ok);
   if (!Ok) return aTetaMax;

   Pt3dr anI2 =  aR2to1.ImRecAff(anI1);

   Pt2dr anEc1  = aP1 - ProjStenope(anI1);
   Pt2dr anEc2  = aP2 - ProjStenope(anI2);

   return  GenCoutAttenueTetaMax((dist48_euclid(anEc1) + dist48_euclid(anEc2)) / 2.0  ,aTetaMax);
}




double DistDroiteCostMEP(const ElRotation3D & aR2to1,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aR2to1.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aR2to1.tr();

   ElSeg3D aS1(Pt3dr(0,0,0),aQ1);
   ElSeg3D aS2(aBase,aBase+aQ2);

   Pt3dr anI = aS1.PseudoInter(aS2);

   double d1 = aS1.DistDoite(anI);
   double d2 = aS2.DistDoite(anI);
   double D1 = euclid(anI);
   double D2 = euclid(aBase-anI);


   double aTeta =  (d1/D1 + d2/D2) / 2;
   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVCostMEP(const ElRotation3D & aR2to1,const Pt3dr & aQ1,const Pt3dr & aQ2Init,double aTetaMax)
{
   Pt3dr aQ2 = aR2to1.Mat() *  aQ2Init;
   Pt3dr aBase  = aR2to1.tr();

   Pt3dr aQ1vQ2vB = vunit(aQ1 ^ aQ2) ^ aBase;

   double aDet = Det(aQ1,aQ2,aBase);

   //   /4.0 : Adatation empirique  aux autres couts
   double aTeta = (ElAbs(aDet/scal(aQ1vQ2vB,aQ1)) +  ElAbs(aDet/scal(aQ1vQ2vB,aQ2))) / 4.0 ;

   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVCostMEP(const ElRotation3D & aR2to1,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
   return PVCostMEP(aR2to1,vunit(PZ1(aP1)), vunit(PZ1(aP2)),aTetaMax);
}

double  LinearCostMEP(const ElRotation3D & aR2to1,const Pt3dr & aQ1,const Pt3dr & aQ2,double aTetaMax)
{
      return GenCoutAttenueTetaMax(ElAbs(Det(aQ1,(aR2to1.Mat()*aQ2),aR2to1.tr())),aTetaMax);
}


double  LinearCostMEP(const ElRotation3D & aR2to1,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
    return LinearCostMEP(aR2to1,vunit(PZ1(aP1)),vunit(PZ1(aP2)),aTetaMax);
}



    // =====================================
    // Fonction sur une pack hom
    // =====================================

/*
#define DECLARE_PACK_COST_MEP (FONC_ELEM)\
double totoooooo(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax) \
{\
 return 0.0;\
}


    double aSomPCost = 0;\
    double aSomPds = 0;\
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)\
    {\
         double aPds = itP->Pds();\
         double aCost = FONC_ELEM(aRot,itP->P1(),itP->P2(),aTetaMax);\
         aSomPds += aPds;\
         aSomPCost += aPds * aCost;\
    }\
    return (aSomPCost / aSomPds);\
}
*/



double QuickD48EProjCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = QuickD48EProjCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double ProjCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = ProjCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double DistDroiteCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = DistDroiteCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double PVCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = PVCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}

double LinearCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax)
{
    double aSomPCost=0,aSomPds = 0;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = LinearCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax);
         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return (aSomPCost / aSomPds);
}





Pt3dr MedianNuage(const ElPackHomologue & aPack,const ElRotation3D & aRot)
{
    std::vector<double>  aVX;
    std::vector<double>  aVY;
    std::vector<double>  aVZ;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
        bool Ok;
        Pt3dr anI = InterSeg(aRot,itP->P1(),itP->P2(),Ok);

        if (Ok)
        {
           aVX.push_back(anI.x);
           aVY.push_back(anI.y);
           aVZ.push_back(anI.z);
        }
    }
    return Pt3dr
           (
                 MedianeSup(aVX),
                 MedianeSup(aVY),
                 MedianeSup(aVZ)
           );
}


/*************************************************************************/
/*                                                                       */
/*               TEST                                                    */
/*                                                                       */
/*************************************************************************/


#define NBTEstTimeBundle  20

void TestLinariseAngle(const  ElPackHomologue & aPack,const ElRotation3D &aRef,double aFoc)
{
   // TestQE(); getchar();
   {
       double anAtten = -1;
       std::cout << "ProjD48 " << aFoc * QuickD48EProjCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "ProEucl " << aFoc * ProjCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "DDroite " << aFoc * DistDroiteCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "ProVec  " << aFoc * PVCostMEP(aPack,aRef,anAtten) << "\n";
       std::cout << "Det     " << aFoc * LinearCostMEP(aPack,aRef,anAtten) << "\n";
   }
   double aNoise = 0.0;
   ElRotation3D aSolF = ElRotation3D::Id;
   for (int aTime =0 ; aTime < 1 ; aTime++)
   {
       ElMatrix<double> aMP =  ElMatrix<double>::Rotation(aNoise*NRrandC(),aNoise*NRrandC(),aNoise*NRrandC());
       ElRotation3D aRInit(aRef.tr(),aRef.Mat()*aMP,true);

       for (int aKTyB=0 ; aKTyB<5 ; aKTyB++)
       {
           ElRotation3D aRot = aRInit;
           cInterfBundle2Image * aIB = 0;
           if (aKTyB<2)
                aIB = cInterfBundle2Image::LineariseAngle(aPack,aFoc,aKTyB==0);
           else if (aKTyB==2)
                aIB = cInterfBundle2Image::LinearDet(aPack,aFoc);
           else if (aKTyB<5)
                aIB = cInterfBundle2Image::Bundle(aPack,aFoc,aKTyB==3);
           else
           {
               ELISE_ASSERT(false,"TestLinariseAngle");
           }


           double anErr = aIB->ErrInitRobuste(aRot);
           double  anErr0 = anErr;

           ElTimer aChrono;
           for (int aIter=0 ; aIter< NBTEstTimeBundle ; aIter++)
           {
               aRot = aIB->OneIterEq(aRot,anErr);
               // std::cout << "ERR " << aIter << " " << anErr * aFoc << "\n";
               if (aIter==0)
                  anErr0 = anErr;
           }
           std::cout << "ERR = " << anErr0*aFoc << " => " << anErr * aFoc
                     << " L2: " << aRot.Mat().L2(aRef.Mat()) << " Tr: " << euclid(aRot.tr()-vunit(aRef.tr()))
                     << " Time " << aChrono.uval()
                     << " CostLi " <<  aFoc * LinearCostMEP(aPack,aRot,-1)
                     << " CostPV " <<  aFoc * PVCostMEP(aPack,aRot,-1)
                     << " Ty " << aIB->VIB2I_NameType() << "\n";
           aSolF = aRot;
       }
       std::cout << "\n";
   }
   {
       double anAtten = -1;
       std::cout << "Det     " << aFoc * LinearCostMEP(aPack,aRef,anAtten)        << "    " << aFoc * LinearCostMEP(aPack,aSolF,anAtten)         <<"\n";
       std::cout << "ProjD48 " << aFoc * QuickD48EProjCostMEP(aPack,aRef,anAtten) << "    " << aFoc * QuickD48EProjCostMEP(aPack,aSolF,anAtten)  << "\n";
       std::cout << "ProEucl " << aFoc * ProjCostMEP(aPack,aRef,anAtten)          << "    " << aFoc * ProjCostMEP(aPack,aSolF,anAtten)           << "\n";
       std::cout << "DDroite " << aFoc * DistDroiteCostMEP(aPack,aRef,anAtten)    << "    " << aFoc * DistDroiteCostMEP(aPack,aSolF,anAtten)     << "\n";
       std::cout << "ProVec  " << aFoc * PVCostMEP(aPack,aRef,anAtten)            << "    " << aFoc * PVCostMEP(aPack,aSolF,anAtten)             << "\n";
   }
/*
 Resultat non explique , pourquoi sur le Ref PVCostMEP est homogenes aux autre, et pas sur la SolF ??

        Det     0.1467    0.302827
        ProjD48 0.155966    0.153702
        ProEucl 0.154485    0.15228
        DDroite 0.154348    0.152146
        ProVec  0.154348    0.152146

*/

   // double aS
   getchar();

/*
   for (int aK=0 ; aK< 2 ; aK++)
   {
       ElMatrix<double> aMP =  ElMatrix<double>::Rotation(aNoise*NRrandC(),aNoise*NRrandC(),aNoise*NRrandC());
       ElRotation3D aRInit(aRef.tr(),aRef.Mat()*aMP,true);

       ElRotation3D aRot = aRInit;
       cEqBundleBase * aBB = new  cEqBundleBase (false,0,aFoc,aK==0);
       ElTimer aChrono;
       double anErr = aBB->AddPackAndSolve(aPack,aRot,-1);
       for (int aKS=0 ; aKS < NBTEstTimeBundle ; aKS++)
       {
             anErr = aBB->AddPackAndSolve(aPack,aRot,anErr);
             aRot = aBB->CurSol();
       }
       std::cout << "Colin Errreur " << anErr * aFoc << " T " << aChrono.uval() << "\n";
   }
*/
}

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
