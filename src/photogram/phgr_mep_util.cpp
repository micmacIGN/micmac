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


std::cout << "Coor " << mCompCoord[0] << "\n";
std::cout << "Coor " << mCompCoord[1] << "\n";
std::cout << "Coor " << mCompCoord[2] << "\n";
std::cout << "Coor " << mCompCoord[3] << "\n";
std::cout << "Coor " << mCompCoord[4] << "\n";

std::cout << "Q2 "  << mLocQ2_x << " " << mLocQ2_y << " " << mLocQ2_z << "\n";
std::cout << "Qp1 "  << mLocQp1_x << " " << mLocQp1_y << " " << mLocQp1_z << "\n";
std::cout << "B0 "  << mLocVecB0_x << " " << mLocVecB0_y << " " << mLocVecB0_z << "\n";
std::cout << "C  "  << mLocVecC_x << " " << mLocVecC_y << " " << mLocVecC_z << "\n";
std::cout << "D  "  << mLocVecD_x << " " << mLocVecD_y << " " << mLocVecD_z << "\n";





std::cout << "======= 5555 ========\n";
std::cout << tmp50_ << " " << tmp51_ << " " << tmp52_ << " " << tmp53_ << " " << tmp54_ << "\n";
std::cout << tmp55_ << " " << tmp56_ << " " << tmp57_ << " " << tmp58_ << " " << tmp59_ << "\n";

*/


#include "StdAfx.h"

/************************************************************/
/*                                                          */
/*           Equation Linearise de l'angle                  */
/*                                                          */
/************************************************************/

extern bool AllowUnsortedVarIn_SetMappingCur;



/*
     cOmegaLineAng et cScalLineAng sont definie afin d'heriter de cElemEqFormelle, cObjFormel2Destroy
    Sinon, on aurai interet a les traiter directement dans cGlobEqLineraiseAngle
*/

class cOmegaLineAng : public cElemEqFormelle,
                      public cObjFormel2Destroy
{
    public : 
       Pt3dr             mP0;
       Pt3d<Fonc_Num>    mP;  

       cOmegaLineAng(cSetEqFormelles & aSet) :
          cElemEqFormelle (aSet,false),
          mP0     (0,0,0),
          mP      (aSet.Alloc().NewPt3(mP0))
       {
           CloseEEF();
           aSet.AddObj2Kill(this);
       }
};


class cScalLineAng : public cElemEqFormelle,
                     public cObjFormel2Destroy
{
    public : 
       double      mS0;
       Fonc_Num    mS;  

       cScalLineAng(cSetEqFormelles & aSet) :
          cElemEqFormelle (aSet,false),
          mS0     (0),
          mS      (aSet.Alloc().NewF(&mS0))
       {
           CloseEEF();
           aSet.AddObj2Kill(this);
       }
};

class cGlobEqLineraiseAngle :    public cNameSpaceEqF,
                            public cObjFormel2Destroy
{
    public :
       cGlobEqLineraiseAngle(bool doGenCode);
       void    GenCode();
       void    InitNewRot(const ElRotation3D & aRot);
       ElRotation3D SolveResetUpdate();
       Pt2dr    AddEquation(const Pt3dr & aP1,const Pt3dr & aP2,double aPds);
       // Couple d'angle
       Pt2dr  ResiduEquation(const Pt3dr & aP1,const Pt3dr & aP2);
       
    private :
       Pt2dr    AddEquationGen(const Pt3dr & aP1,const Pt3dr & aP2,double aPds,bool WithEq);

       ElMatrix<double> tR0;

       cSetEqFormelles * mSetEq;
       cOmegaLineAng *   mW;
       cP3d_Etat_PhgrF   mB0;
       cScalLineAng *    mc;
       cP3d_Etat_PhgrF   mC;
       cScalLineAng *    md;
       cP3d_Etat_PhgrF   mD;
       Pt3d<Fonc_Num>    mResidu;

       cP3d_Etat_PhgrF   mQp1;  // tRot ^ U1
       cP3d_Etat_PhgrF   mQ2;
 
       std::vector<Fonc_Num> mVRes;
       cElCompiledFonc *     mFoncEqResidu;
       cIncListInterv        mLInterv;
       std::string           mNameType;
};

void cGlobEqLineraiseAngle::InitNewRot(const ElRotation3D & aRot)
{
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

    return Pt2dr(aResidu[0],aResidu[1]);
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

cGlobEqLineraiseAngle::cGlobEqLineraiseAngle(bool doGenCode) :
    tR0         (1,1),
    mSetEq      (new cSetEqFormelles(cNameSpaceEqF::eSysPlein)),
    mW          (new cOmegaLineAng(*mSetEq)),
    mB0         ("VecB0"),
    mc          (new cScalLineAng (*mSetEq)),
    mC          ("VecC"),
    md          (new cScalLineAng (*mSetEq)),
    mD          ("VecD"),
    mQp1        ("Qp1"),
    mQ2         ("Q2"),
    mNameType   ("cEqLinariseAngle")
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

   Fonc_Num aDet = Det(aQp1,aQp2,aBase);
  
   mVRes.push_back(aDet/scal(aQp1VQp2,aQp1));
   mVRes.push_back(aDet/scal(aQp1VQp2,aQp2));

   // Fonc_Num aRIm1 = aDet / scal(

   if (doGenCode)
   {
       GenCode();
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

void cGlobEqLineraiseAngle::GenCode()
{
   // Un objet de type equation peux gerer plusieurs equation;
    // il faut passer par un vecteur

    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        mVRes,  // expressions formelles 
        mLInterv  // intervalle de reference
    );

}

void GenCodeEqLinariseAngle()
{
    cGlobEqLineraiseAngle * anEla = new  cGlobEqLineraiseAngle (true);
    delete anEla;
}


/*
   Pt3dr aQ1vQ2vB = vunit(aQ1 ^ aQ2) ^ aBase;

   double aDet = Det(aQ1,aQ2,aBase);

   //   /2 pour etre coherent avec ExactCostMEP
   double aTeta = (ElAbs(aDet/scal(aQ1vQ2vB,aQ1)) +  ElAbs(aDet/scal(aQ1vQ2vB,aQ2))) / 2.0;
*/
   

class cFullEqLinariseAngle 
{
    public :
       cFullEqLinariseAngle(const  ElPackHomologue & aPack,ElRotation3D R,double aFoc);
       double ErrStd(const ElRotation3D &aRot);


       cGlobEqLineraiseAngle * mELA;
       std::vector<Pt3dr> mVP1;
       std::vector<Pt3dr> mVP2;
       std::vector<double> mVPds;
       int                 mNb;
       double              mFoc;
};

double cFullEqLinariseAngle::ErrStd(const ElRotation3D &aRot)
{
  std::vector<double> aVRes;
  for (int aK=0 ; aK< mNb ; aK++)
      aVRes.push_back(PVExactCostMEP(aRot,mVP1[aK],mVP2[aK],-1));
  return KthValProp(aVRes,0.75);
}

cFullEqLinariseAngle::cFullEqLinariseAngle(const  ElPackHomologue & aPack,ElRotation3D  aRot,double aFoc) :
   mFoc (aFoc)
{
  for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
  {
       mVP1.push_back(vunit(PZ1(itP->P1())));
       mVP2.push_back(vunit(PZ1(itP->P2())));
       mVPds.push_back(itP->Pds());
  }
  mNb = mVP1.size();
  mELA = new cGlobEqLineraiseAngle(false);

  mELA->InitNewRot(aRot);


/*
  int aNBIT = 10000;

  ElTimer aCh1;
  for (int aCpt=0 ; aCpt< aNBIT ; aCpt++)
      for (int aK=0 ; aK< mNb ; aK++)
           PVExactCostMEP(aRot,mVP1[aK],mVP2[aK],-1);
  std::cout << "TPV " << aCh1.uval() << "\n";
  getchar();

  ElTimer aCh2;
  for (int aCpt=0 ; aCpt< aNBIT ; aCpt++)
      for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
           PVExactCostMEP(aRot,itP->P1(),itP->P2(),-1);
  std::cout << "TPV " << aCh2.uval() << "\n";
  getchar();


  Pt3dr anI;
  ElTimer aCh3;
  for (int aCpt=0 ; aCpt< aNBIT ; aCpt++)
      for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
           ExactCostMEP(anI,aRot,itP->P1(),itP->P2(),-1);
  std::cout << "TPV " << aCh3.uval() << "\n";
  getchar();
*/



  double  anErStd = ErrStd(aRot);

  for (int aCpt=0 ; aCpt<8; aCpt++)
  {
      double ErrIn = anErStd;
      std::vector<double> aVRes;

      for (int aK=0 ; aK< mNb ; aK++)
      {
          Pt2dr aPRes = mELA->ResiduEquation(mVP1[aK],mVP2[aK]);
          double aRes = ElAbs(aPRes.x) + ElAbs(aPRes.y);
          double aPds = 1/ (1 + ElSquare(aRes/(2*anErStd)));
          mELA->AddEquation(mVP1[aK],mVP2[aK],aPds);
          aVRes.push_back(aRes);

// std::cout <<  aRes / PVExactCostMEP(aRot,mVP1[aK],mVP2[aK],-1) << "\n";

      }
      anErStd = KthValProp(aVRes,0.75);
      aRot = mELA->SolveResetUpdate();
      mELA->InitNewRot(aRot);
      std::cout << "ERR  " << ErrIn*mFoc << " ==> " <<  ErrStd(aRot)*mFoc << "\n";
getchar();
  }
}


void TestLinariseAngle(const  ElPackHomologue & aPack,const ElRotation3D &aRot,double aFoc)
{
   cFullEqLinariseAngle aFELA(aPack,aRot,aFoc);
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

double cBundleIterLin::ErrMoy() const {return mSomErr/mSomPds;}

cBundleIterLin::cBundleIterLin(const ElRotation3D & aRot,const double & anErrStd):
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
void cBundleIterLin::AddObs(const Pt3dr & aQ1,const Pt3dr& aQ2,const double & aPds)
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

ElRotation3D  cBundleIterLin::CurSol()
{
    Im1D_REAL8   aSol = mSysLin5.GSSR_Solve (0);
    double * aData = aSol.data();

    Pt3dr aNewB0 = mRot.Mat()  * vunit(mB0+mC*aData[0] + mD*aData[1]);

    ElMatrix<double> aNewR = NearestRotation(mRot.Mat() * (ElMatrix<double>(3,true) + MatProVect(Pt3dr(aData[2],aData[3],aData[4]))));
    return ElRotation3D (aNewB0,aNewR,true);
}


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

double ExactCostMEP(Pt3dr &  anI,const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) 
{
   Pt3dr aQ1 = Pt3dr(aP1.x,aP1.y,1.0);
   Pt3dr aQ2 = aRot.Mat() * Pt3dr(aP2.x,aP2.y,1.0);
   Pt3dr aBase  = aRot.tr();

   ElSeg3D aS1(Pt3dr(0,0,0),aQ1);
   ElSeg3D aS2(aBase,aBase+aQ2);

   anI = aS1.PseudoInter(aS2);
    
   double d1 = aS1.DistDoite(anI);
   double d2 = aS2.DistDoite(anI);
   double D1 = euclid(anI);
   double D2 = euclid(aBase-anI);


   //   *2 pour etre coherent avec ExactCostMEP
   double aTeta =  (d1/D1 + d2/D2) * 2;
   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVExactCostMEP(const ElRotation3D & aRot,const Pt3dr & aQ1,const Pt3dr & aQ2Init,double aTetaMax) 
{
   Pt3dr aQ2 = aRot.Mat() *  aQ2Init;
   Pt3dr aBase  = aRot.tr();

   Pt3dr aQ1vQ2vB = vunit(aQ1 ^ aQ2) ^ aBase;

   double aDet = Det(aQ1,aQ2,aBase);

   double aTeta = (ElAbs(aDet/scal(aQ1vQ2vB,aQ1)) +  ElAbs(aDet/scal(aQ1vQ2vB,aQ2))) ;
   
   return GenCoutAttenueTetaMax(aTeta,aTetaMax);
}

double PVExactCostMEP(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) 
{
   return PVExactCostMEP(aRot,vunit(PZ1(aP1)), vunit(PZ1(aP2)),aTetaMax);
}



double  LinearCostMEP(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax)
{
      Pt3dr aQ1 = vunit(PZ1(aP1));
      Pt3dr aQ2 = vunit(PZ1(aP2));
      double aDet = scal(aQ1^(aRot.Mat()*aQ2),aRot.tr());
      aDet = ElAbs(aDet);

      return GenCoutAttenueTetaMax(aDet,aTetaMax);
}



double ExactCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax) 
{
    double aSomPCost = 0;
    double aSomPds = 0;
    Pt3dr anI;

    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
         double aPds = itP->Pds();
         double aCost = ExactCostMEP(anI,aRot,itP->P1(),itP->P2(),aTetaMax);

 // std::cout << "RRRrr " << aCost / PVExactCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax)  << " " << aCost /  LinearCostMEP(aRot,itP->P1(),itP->P2(),aTetaMax) << "\n";

         aSomPds += aPds;
         aSomPCost += aPds * aCost;
    }
    return aSomPCost / aSomPds;
}



Pt3dr MedianNuage(const ElPackHomologue & aPack,const ElRotation3D & aRot)
{
    std::vector<double>  aVX;
    std::vector<double>  aVY;
    std::vector<double>  aVZ;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
    {
        Pt3dr                anI;
        ExactCostMEP(anI,aRot,itP->P1(),itP->P2(),0.1);
        aVX.push_back(anI.x);
        aVY.push_back(anI.y);
        aVZ.push_back(anI.z);

// std::cout << "iiiiI " << anI << "\n";
    }
    return Pt3dr
           (
                 MedianeSup(aVX),
                 MedianeSup(aVY),
                 MedianeSup(aVZ)
           );
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
