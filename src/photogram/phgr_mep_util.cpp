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

*/


#include "StdAfx.h"

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

       cPt3dEEF(cSetEqFormelles & aSet,const Pt3dr & aP0) :
          cElemEqFormelle (aSet,false),
          mP0     (aP0),
          mP      (aSet.Alloc().NewPt3(mP0))
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

       cScalEEF(cSetEqFormelles & aSet,double aV0) :
          cElemEqFormelle (aSet,false),
          mS0     (aV0),
          mS      (aSet.Alloc().NewF(&mS0))
       {
           CloseEEF();
           aSet.AddObj2Kill(this);
       }
};


/************************************************************/
/*                                                          */
/*            Bundle classique                              */
/*                                                          */
/************************************************************/

/*
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

class cEqBundleBase  : public cNameSpaceEqF,
                      public cObjFormel2Destroy
{
    public :
       cEqBundleBase(bool DoGenCode,int aNbCamSup);  // Nb de cam en plus des 2 minim
       void    InitNewRot(const ElRotation3D & aRot);
       ElRotation3D SolveResetUpdate();
       
       double AddEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel,double aPds);
       double ResiduEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel);
    protected :
       // virtual Pt2dr    AddEquationGen(const Pt3dr & aP1,const Pt3dr & aP2,double aPds,bool WithEq) = 0;
      
       double    AddEquationGen(const std::vector<Pt2dr> & aP2,const std::vector<bool> & aVSel, double aPds,bool WithEq);

       ElMatrix<double> tR2;

       cSetEqFormelles * mSetEq;
       cSetEqFormelles * mSetEq2;  // Galere pour gere la connexist des intervalle en mode GenCode
       cPt3dEEF *        mW1;
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
       std::string           mNameEq1;
       std::string           mNameEq2;

       std::vector<cElCompiledFonc *>     mVFEsResid;
       cEqfP3dIncTmp *       mEqP3I;
       cEqfP3dIncTmp *       mEq2P3I;
       cSubstitueBlocIncTmp  mSBIT12;
};

void  cEqBundleBase::InitNewRot(const ElRotation3D & aRot)
{
     for (int aK=0 ; aK<mSetEq->Alloc().CurInc() ; aK++) 
         mSetEq->Alloc().SetVar(0,aK);
     tR2 = aRot.Mat().transpose();
     mW1->mP0 = Pt3dr(0,0,0);

     mB2Cur = tR2 * aRot.tr();
     Pt3dr aC2,aD2;
     MakeRONWith1Vect(mB2Cur,aC2,aD2);
     mB2.SetEtat(mB2Cur);

     mc2->mS0 = 0;
     mC2.SetEtat(aC2);
     md2->mS0 = 0;
     mD2.SetEtat(aD2);

     mSetEq->SetPhaseEquation();
/*
*/
}

double     cEqBundleBase::AddEquationGen(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel, double aPds,bool WithEq)
{
   int aFlag=0;
   
   std::vector<double> aVPds;
   aVPds.push_back(aPds);
   aVPds.push_back(aPds);
   double aRes = 0;
   std::vector<ElSeg3D> aVSeg;
   static Pt3dr    aVDir[9];
   int aNbP=0;

   if (aVSel[0])
   {
       aVDir[0] = tR2 * Pt3dr(aVPts[0].x,aVPts[0].y,1);
       mI1.SetEtat(ProjStenope(aVDir[0]));
       aVSeg.push_back(ElSeg3D(Pt3dr(0,0,0),aVDir[0]));
       aNbP++;
   }

   if (aVSel[1])
   {
       aVDir[1] = Pt3dr(aVPts[1].x,aVPts[1].y,1);
       mI2.SetEtat(ProjStenope(aVDir[1]));
       aVSeg.push_back(ElSeg3D(mB2Cur,aVDir[1]));
       aNbP++;
   }
   
   ELISE_ASSERT(aVSeg.size()>=2,"cEqBundleBase::AddEquationGen");
   Pt3dr aP = ElSeg3D::L2InterFaisceaux(0,aVSeg);

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

         aRes += ElAbs(aVRES[0]) + ElAbs(aVRES[1]);
      }
   }

   if (aFlag== 3) mSBIT12.DoSubst();

   return aRes;
}

double cEqBundleBase::AddEquation(const std::vector<Pt2dr> & aVPts,const std::vector<bool> & aVSel,double aPds)
{
   return AddEquationGen(aVPts,aVSel,aPds,true);
}
// Pt2dr  ResiduEquation(const Pt3dr & aP1,const Pt3dr & aP2);

cEqBundleBase::cEqBundleBase(bool DoGenCode,int aNbCamSup) :
    tR2         (1,1),
    mSetEq      (new cSetEqFormelles(cNameSpaceEqF::eSysPlein)),
    mSetEq2     ( DoGenCode ? new cSetEqFormelles(cNameSpaceEqF::eSysPlein) : mSetEq),
    mW1         (new cPt3dEEF(*mSetEq,Pt3dr(0,0,0))),
    mB2         ("VecB2"),
    mc2         (new cScalEEF (*mSetEq2,0)),
    mC2         ("VecC2"),
    md2         (new cScalEEF (*mSetEq2,0)),
    mD2         ("VecD2"),
    mI1         ("I1"),
    mI2         ("I2"),
    mNbCamSup   (aNbCamSup),
    mNameEq1    ("cEqBBCamFirst"),
    mNameEq2    ("cEqBBCamSecond"),
    mEqP3I      (mSetEq->Pt3dIncTmp()),
    mEq2P3I     (DoGenCode ? mSetEq2->Pt3dIncTmp() : mEqP3I ),
    mSBIT12     (*mEqP3I)
{
  AllowUnsortedVarIn_SetMappingCur = true;
  mW1->IncInterv().SetName("Omega1");
  mc2->IncInterv().SetName("C2");
  md2->IncInterv().SetName("D2");

  mLInterv1.AddInterv(mW1->IncInterv());
  mLInterv1.AddInterv(mEqP3I->IncInterv());

  mLInterv2.AddInterv(mc2->IncInterv());
  mLInterv2.AddInterv(md2->IncInterv());
  mLInterv2.AddInterv(mEq2P3I->IncInterv());

  if (DoGenCode)
  {
      {
         {
             std::vector<Fonc_Num> aVR1;
             Pt3d<Fonc_Num>  aPIGlob = mEqP3I->PF();

             Pt3d<Fonc_Num> aP1 = mW1->mP ^ aPIGlob;
             aVR1.push_back(mI1.PtF().x - aP1.x / aP1.z);
             aVR1.push_back(mI1.PtF().y - aP1.y / aP1.z);
             cElCompileFN::DoEverything
             (
                 DIRECTORY_GENCODE_FORMEL, 
                 mNameEq1,  
                 aVR1,  
                 mLInterv1  
             );
         }
         {
             std::vector<Fonc_Num> aVR2;
             Pt3d<Fonc_Num>  aPIGlob = mEq2P3I->PF();
             Pt3d<Fonc_Num> aP2 =  aPIGlob + mB2.PtF() + mC2.PtF()*mc2->mS + mD2.PtF()*md2->mS;
             aVR2.push_back(mI2.PtF().x - aP2.x / aP2.z);
             aVR2.push_back(mI2.PtF().y - aP2.y / aP2.z);
             cElCompileFN::DoEverything
             (
                 DIRECTORY_GENCODE_FORMEL, 
                 mNameEq2,  
                 aVR2,  
                 mLInterv2  
             );
         }
      }

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

void GenCodecEqBundleBase()
{
    cEqBundleBase * anEla = new  cEqBundleBase (true,0);
    delete anEla;

    anEla = new  cEqBundleBase (false,0);
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
    mW          (new cPt3dEEF(*mSetEq,Pt3dr(0,0,0))),
    mB0         ("VecB0"),
    mc          (new cScalEEF (*mSetEq,0)),
    mC          ("VecC"),
    md          (new cScalEEF (*mSetEq,0)),
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
  
   mVFRes.push_back(aDet/scal(aQp1VQp2,aQp1));
   mVFRes.push_back(aDet/scal(aQp1VQp2,aQp2));

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
        mVFRes,  // expressions formelles 
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
