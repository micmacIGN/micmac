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


#define UseTjsDist false

const bool AFocalAcceptNoDist = false;




/************************************************************/
/*                                                          */
/*                 cPolynome1VarFormel                      */
/*                                                          */
/************************************************************/

bool BugGL = false;

cPolynome1VarFormel::cPolynome1VarFormel
(
    cSetEqFormelles    & aSet,
    cVarSpec           aVarTime,
    REAL *             aV0,
    INT                aDegre,
    const std::string& aGrp,
    const std::string& aName0
)  :
    mV0    (aV0),
    mDegre (aDegre)
{
    std::string aName = aName0 + ((aDegre==0) ? "" : ToString(aDegre));
    for (INT aK=0 ; aK<= mDegre ; aK++)
        mValsCur.push_back(0.0);

    mFonc =  aSet.Alloc().NewF(aGrp,aName,mV0);

    for (INT aK=1 ; aK<= mDegre ; aK++)
    {
        mFonc = mFonc +  aSet.Alloc().NewF(aGrp,aName,&(mValsCur[aK])) * PowI(aVarTime,aK);
    }
}

Fonc_Num  cPolynome1VarFormel::Fonc()
{
   return mFonc;
}


REAL  cPolynome1VarFormel::Val(REAL aTime)
{
   REAL aRes = *mV0;
   REAL aPowT = 1.0;

    for (INT aK=1 ; aK<= mDegre ; aK++)
    {
        aPowT *= aTime;
        aRes +=  mValsCur[aK] * aPowT;
    }

    return aRes;
}

/************************************************************/
/************************************************************/
/*                                                          */
/*                 cRotationFormelle                        */
/*                                                          */
/************************************************************/

extern Pt2d<Fonc_Num> operator * (Pt2d<Fonc_Num> aP,Fonc_Num aScal);

cRotationFormelle::~cRotationFormelle()
{
}

cPolynome1VarFormel  * cRotationFormelle::AllocPol(REAL * aValCste,const std::string & aGrp,const std::string  &aName)
{
   return new cPolynome1VarFormel(mSet,mVarTime,aValCste,mDegre,aGrp,aName);
}

INT  cRotationFormelle::Degre() const
{
   return mDegre;
}

const std::string & cRotationFormelle::NameParamTime() 
{
   return mNameParamTime;
}

cRotationFormelle::cRotationFormelle      
(
     eModeContrRot     aModeC,
     ElRotation3D      aRC2MInit,
     cSetEqFormelles & aSet,
     cRotationFormelle * aRAtt,
     const std::string & aName,
     INT                 aDegre,
     bool                aVraiBaseU
) :
  cElemEqFormelle (aSet,false),
  mModeContr      (aModeC),

  mNameParamTime  ("TimeRot"),
  mVarTime        (0,mNameParamTime),
  mDegre          (aDegre),

  mTeta01Init     (aRC2MInit.teta01()),
  mTeta02Init     (aRC2MInit.teta02()),
  mTeta12Init     (aRC2MInit.teta12()),

  mCurTeta01      (mTeta01Init),
  mCurTeta02      (mTeta02Init),
  mCurTeta12      (mTeta12Init),

  mPolTeta01      (AllocPol(&mCurTeta01,aName,"T01")),
  mPolTeta02      (AllocPol(&mCurTeta02,aName,"T02")),
  mPolTeta12      (AllocPol(&mCurTeta12,aName,"T12")),

  mFTeta01        (mPolTeta01->Fonc()),
  mFTeta02        (mPolTeta02->Fonc()),
  mFTeta12        (mPolTeta12->Fonc()),
  mFMatr          (ElMatrix<Fonc_Num>::Rotation(mFTeta01,mFTeta02,mFTeta12)),
  mFMatrInv       (mFMatr.transpose()),

  mCOptInit       (aRC2MInit.tr()),
  mCurCOpt        (mCOptInit),
  mIndAllocCOpt   (mSet.Alloc().CurInc()),
  mPolCoptX       (AllocPol(&mCurCOpt.x,aName,"Cx")),
  mPolCoptY       (AllocPol(&mCurCOpt.y,aName,"Cy")),
  mPolCoptZ       (AllocPol(&mCurCOpt.z,aName,"Cz")),
  mFCOpt          (mPolCoptX->Fonc(),mPolCoptY->Fonc(),mPolCoptZ->Fonc()),

  pRotAttach      (aRAtt),
  pRotPseudoBaseU (0),
  mFcteurRapCoU   (FcteurRapCoU()),
  mName           (aName),
  mFlagAnglFig    (0),
  mTolAng         (cContrainteEQF::theContrStricte),
  mTolCentre      (cContrainteEQF::theContrStricte),
  mModeGL         (false),
  // mSMatriceGL     (0),
  // mPMatriceGL     (0),
  mMGL            (3,true),
  mVraiBaseU      (aVraiBaseU)
{
   // mFcteurRapBaseU (new dd
   AddFoncteurEEF(mFcteurRapCoU);
   CloseEEF();
}

const ElMatrix<REAL> &       cRotationFormelle::MGL() const
{
   return mMGL;
}


/*
cMatr_Etat_PhgrF & cRotationFormelle::MatGL(bool isP) 
{
    cMatr_Etat_PhgrF ** aM = (isP) ? &mPMatriceGL : & mSMatriceGL;
    if (*aM==0)
       *aM = new cMatr_Etat_PhgrF("GL",3,3);
    return **aM;
}
*/

/*
void cRotationFormelle::InitEtatGL(bool isP)
{
   if (IsGL())
     MatGL(isP).SetEtat(mMGL);
}
*/

bool cRotationFormelle::IsGL() const
{
    return mModeGL;
}
void cRotationFormelle::SetGL(bool aModeGL,const ElRotation3D & aGuimb2CM)
{
/*
    if (aModeGL)
    {
       MatGL(true);
       MatGL(false);
    }
*/
    SetCurRot(CurRot(),aGuimb2CM);
    mModeGL = aModeGL;
}
const ElMatrix<Fonc_Num> & cRotationFormelle::MatFGL(int aKForceGL)
{
   if (aKForceGL>=0)
   {
       static std::vector<cMatr_Etat_PhgrF *> aVM;
       // for (int aK=0 ; aK<=aKForceGL ; aK++)
       for (int aK=int(aVM.size()) ; aK<=aKForceGL ; aK++)  // MPD : sinon on augment a chaque fois
           aVM.push_back(new cMatr_Etat_PhgrF("GL_MK"+ToString(aKForceGL),3,3));

       return aVM[aKForceGL]->Mat();
   }

   if (mModeGL)
   {
      static cMatr_Etat_PhgrF aM("GL",3,3);
      return aM.Mat();
   }
   static ElMatrix<Fonc_Num> aMatId(3,true);
   return aMatId;
}

Pt3dr  cRotationFormelle::AddRappOnCentre(const Pt3dr & aPVal,const Pt3dr & aPds,bool WithDerSec)
{
    double aVVals[6];
    aVVals[3]= aPVal.x;
    aVVals[4]= aPVal.y;
    aVVals[5]= aPVal.z;
    tContFcteur aFRap =  FoncRapp(3,6,aVVals);

    Pt3dr aRes;

    aRes.x = aPds.x ? mSet.AddEqFonctToSys(aFRap[0],aPds.x,WithDerSec) : mSet.ResiduSigne(aFRap[0]) ;
    aRes.y = aPds.y ? mSet.AddEqFonctToSys(aFRap[1],aPds.y,WithDerSec) : mSet.ResiduSigne(aFRap[1]);
    aRes.z = aPds.z ? mSet.AddEqFonctToSys(aFRap[2],aPds.z,WithDerSec) : mSet.ResiduSigne(aFRap[2]);

    //  std::cout << mSet.ResiduSigne(aFRap[0]);
    return aRes;
}

void  cRotationFormelle::AddRappOnRot(const ElRotation3D &  aRot,const Pt3dr & aPdsC,const Pt3dr & aPdsRot)
{
    double aVVals[6];

    ELISE_ASSERT(mModeGL,"RotationFormelle::AddRappOnRot no Guibal lock");
    // On veut , d'apres l'utilisation du Guimbal :
    // aRot = ElRotation3D(mCurCOpt.tr(),mMGL*mCurTeta01.Mat(),true);
    // Donc on impose :
    // mCurTeta01 = mMGL.tranp() * aRot
    ElRotation3D aTargetMat (Pt3dr(0,0,0),mMGL.transpose()*aRot.Mat(),true);
    aVVals[0]= aTargetMat.teta01();
    aVVals[1]= aTargetMat.teta02();
    aVVals[2]= aTargetMat.teta12();


    Pt3dr aTargC = aRot.tr();

    aVVals[3]= aTargC.x;
    aVVals[4]= aTargC.y;
    aVVals[5]= aTargC.z;

    tContFcteur aFRap =  FoncRapp(0,6,aVVals);
    mSet.AddEqFonctToSys(aFRap[0],aPdsRot.x,false);
    mSet.AddEqFonctToSys(aFRap[1],aPdsRot.y,false);
    mSet.AddEqFonctToSys(aFRap[2],aPdsRot.z,false);
    mSet.AddEqFonctToSys(aFRap[3],aPdsC.x,false);
    mSet.AddEqFonctToSys(aFRap[4],aPdsC.y,false);
    mSet.AddEqFonctToSys(aFRap[5],aPdsC.z,false);
/*
    
   return aRes;
   aRot = aPdsC;
*/
}


void  cRotationFormelle::SetTolAng(double aTol)
{
   mTolAng = aTol;
}
void  cRotationFormelle::SetTolCentre(double aTol)
{
   mTolCentre = aTol;
}


void cRotationFormelle::SetFlagAnglFige(int aFlag)
{
     mFlagAnglFig = aFlag;
}

cElCompiledFonc *    cRotationFormelle::FcteurRapCoU()
{

  return     ((pRotAttach == 0) || (! mVraiBaseU)) ?
             0            :
             cElCompiledFonc::FoncFixeNormEuclVect
             (
                  &mSet,
                  mIndAllocCOpt,
                  pRotAttach->mIndAllocCOpt,
                  3,
                  euclid(mCurCOpt-pRotAttach->mCurCOpt)
             );
}
void cRotationFormelle::SetValInitOnValCur()
{
     cElemEqFormelle::SetValInitOnValCur();
     ReactuFcteurRapCoU();
}

void cRotationFormelle::ReactuFcteurRapCoU()
{
     if (mFcteurRapCoU)
        mFcteurRapCoU->SetNormValFtcrFixedNormEuclid(euclid(mCurCOpt-pRotAttach->mCurCOpt));
}

void cRotationFormelle::SetCurRot(const ElRotation3D & aR2CM,const ElRotation3D & aGuimb2CM)
{
    AssertDegre0();

    mCurCOpt.x = aR2CM.tr().x;
    mCurCOpt.y = aR2CM.tr().y;
    mCurCOpt.z = aR2CM.tr().z;
    if (mModeGL)
    {
        // ElRotation3D cRotationFormelle::CurRot()
        //  aRes = ElRotation3D(aRes.tr(),mMGL*aRes.Mat(),true);
        //  Mat =    mMGL-1 *  Rot  
        

        // mCurCOpt.x = 0;
        // mCurCOpt.y = 0;
        // mCurCOpt.z = 0;

        mMGL = aGuimb2CM.Mat();
        ElRotation3D aCurDif (Pt3dr(0,0,0), mMGL.transpose()*aR2CM.Mat() ,true);
        mCurTeta01 = aCurDif.teta01();
        mCurTeta02 = aCurDif.teta02();
        mCurTeta12 = aCurDif.teta12();
/*
        mMGL = aR2CM.Mat();
        mCurTeta01 = 0;
        mCurTeta02 = 0;
        mCurTeta12 = 0;
*/
    }
    else
    {
        mMGL = ElMatrix<REAL>(3,true);
        // mCurCOpt.x = aR2CM.tr().x;
        // mCurCOpt.y = aR2CM.tr().y;
        // mCurCOpt.z = aR2CM.tr().z;
        mCurTeta01 = aR2CM.teta01();
        mCurTeta02 = aR2CM.teta02();
        mCurTeta12 = aR2CM.teta12();
    }

    ReinitOnCur();
    ReactuFcteurRapCoU();
}

const std::string & cRotationFormelle::Name() const
{
    return mName;
}



/*
cRotationFormelle * cRotationFormelle::RotAttach()
{
    return pRotAttach;
}
*/

cNameSpaceEqF::eModeContrRot cRotationFormelle::ModeRot() const
{
   return mModeContr;
}

void cRotationFormelle::SetModeRot(eModeContrRot aMode)
{
     mModeContr = aMode;
}

bool cRotationFormelle::IsFiged() const
{
    return mModeContr == eRotFigee;
}

double cRotationFormelle::AddRappelOnCentre
     (
          bool OnCur,
          Pt3dr aTol,
          bool AddEq
     )
{
   std::string aMes="Rappel sur les angles de rotation 3D";


   return   AddRappViscosite ( aMes, OnCur,3, aTol.x,AddEq)
          + AddRappViscosite ( aMes, OnCur,4, aTol.y,AddEq)
          + AddRappViscosite ( aMes, OnCur,5, aTol.z,AddEq) ;
}


double cRotationFormelle::AddRappelOnAngles
     (
          bool OnCur,
          int aK,
          double aTol,
          bool AddEq
     )
{
   return AddRappViscosite
   (
       "Rappel sur les angles de rotation 3D",
       OnCur,aK+0,
       aTol,AddEq
   );
}



double cRotationFormelle::AddRappelOnAngles
     (
         bool OnCur,
         const std::list<int> & aLI,
         double aTol,
         bool AddEq
     )
{
   double aRes = 0.0;
   for 
   (
      std::list<int>::const_iterator itI=aLI.begin();
      itI != aLI.end();
      itI++
   )
   {
      aRes += AddRappelOnAngles(OnCur,*itI,aTol,AddEq);
   }
   return aRes;
}

int cRotationFormelle::NumCentre(int aK) const
{
   return mNumInc0 + 3 + aK;
}

int cRotationFormelle::NumTeta(int aK) const
{
   return mNumInc0  + aK;
}

void  cRotationFormelle::SetRotPseudoBaseU (cRotationFormelle * aRF)
{
    pRotPseudoBaseU = aRF;
}


cMultiContEQF    cRotationFormelle::StdContraintes()
{

  cMultiContEQF  aRes;
  if (mModeContr == eRotLibre)
  {
  }
  else if (mModeContr == eRotAngleFige)
  {
     AddFoncRappInit(aRes,0,3,mTolAng);
  }
  else if (mModeContr == eRotFigee)
  {
     AddFoncRappInit(aRes,0,3,mTolAng);
     AddFoncRappInit(aRes,3,6,mTolCentre);

// std::cout << "YYYYYYYYYYYyyyyyyyyyyyyyyyyyy " << mTolAng << " " << mTolCentre << "\n";
     // aRes =  FoncRappInit(0,6);
  }
  else if (mModeContr == eRotCOptFige)
  {
     AddFoncRappInit(aRes,3,6,mTolCentre);
     // aRes =  FoncRappInit(3,6);
  }
  else if (mModeContr==eRotPseudoBaseU)
  {
      ELISE_ASSERT(pRotPseudoBaseU!=0,"eRotPseudoBaseU sans rattachement ");
      Pt3dr aV01 = CurCOpt()-pRotPseudoBaseU->CurCOpt();
      if (ElAbs(aV01.x) > ElMax(ElAbs(aV01.y),ElAbs(aV01.z)))
      {
           AddFoncRappInit(aRes,3,4,mTolCentre);
      }
      else if (ElAbs(aV01.y) > ElAbs(aV01.z))
      {
           AddFoncRappInit(aRes,4,5,mTolCentre);
      }
      else
      {
           AddFoncRappInit(aRes,5,6,mTolCentre);
      }
  }
  else
  {
     ELISE_ASSERT
     (
           mModeContr==eRotBaseU,
          "Inc in cRotationFormelle::StdContraintes"
     );

     ELISE_ASSERT(pRotAttach!=0,"Attachement Rotatio=0 for eRotBaseU");
  
     mFcteurRapCoU->SetCoordCur(mSet.Alloc().ValsVar());
     aRes.AddAcontrainte(mFcteurRapCoU,mTolCentre);
  }
 
  for (int aBit=0 ; aBit<3 ; aBit++)
  {
     if (mFlagAnglFig & (1<<aBit))
     {
        tContFcteur aCF = FoncRappInit(aBit,aBit+1);
        aRes.AddAcontrainte(aCF[0],mTolAng);
     }
     else
     {
     }
  }

   return aRes;
}



ElMatrix<Fonc_Num>  cRotationFormelle::MatFGLComplete(int aKForceGL)
{
   return  MatFGL(aKForceGL)* mFMatr;
}

Pt3d<Fonc_Num> cRotationFormelle::COpt()
{
    return mFCOpt;
}

Pt3d<Fonc_Num> cRotationFormelle::ImVect(Pt3d<Fonc_Num> aP,int aKForceGL)
{
    return MatFGL(aKForceGL)* mFMatr * aP;
}

Pt3d<Fonc_Num> cRotationFormelle::C2M(Pt3d<Fonc_Num> aP,int aKForceGL)
{
    return MatFGL(aKForceGL)* mFMatr * aP + mFCOpt;
}

Pt3d<Fonc_Num> cRotationFormelle::M2C(Pt3d<Fonc_Num> aP,int aKForceGL)
{
    return  mFMatrInv * MatFGL(aKForceGL).transpose() * (aP - mFCOpt);
}

Pt3d<Fonc_Num> cRotationFormelle::VectM2C(Pt3d<Fonc_Num> aP,int aKForceGL)
{
    return mFMatrInv * MatFGL(aKForceGL).transpose()  * aP ;
}

Pt3d<Fonc_Num> cRotationFormelle::VectC2M(Pt3d<Fonc_Num> aP,int aKForceGL)
{
    return ImVect(aP,aKForceGL);
}

void cRotationFormelle::AssertDegre0() const
{
    ELISE_ASSERT(mDegre==0,"Bas degre in cRotationFormelle::AssertDegre0");
}

Pt3dr  cRotationFormelle::CurCOpt() const
{
   AssertDegre0();
   return mCurCOpt;
}



ElRotation3D cRotationFormelle::CurRot()
{
    AssertDegre0();
    ElRotation3D aRes
           (
              mCurCOpt,
              mCurTeta01,
              mCurTeta02,
              mCurTeta12
           );
   if (mModeGL)
   {
      aRes = ElRotation3D(aRes.tr(),mMGL*aRes.Mat(),true);
   }
   return aRes;
}

Pt3dr  cRotationFormelle::CurCOpt(REAL aT) const
{
    return Pt3dr
           (
                mPolCoptX->Val(aT),
                mPolCoptY->Val(aT),
                mPolCoptZ->Val(aT)
           );
}

ElRotation3D cRotationFormelle::CurRot(REAL aT)
{
    return ElRotation3D
           (
              CurCOpt(aT),
              mPolTeta01->Val(aT),
              mPolTeta02->Val(aT),
              mPolTeta12->Val(aT)
           );
}

/************************************************************/
/*                                                          */
/*                 cGenPDVFormelle                          */
/*                                                          */
/************************************************************/

cSetEqFormelles & cGenPDVFormelle::Set()
{
   return mSet;
}


cGenPDVFormelle::cGenPDVFormelle(cSetEqFormelles & aSet) :
   mSet  (aSet)
{
}



/************************************************************/
/*                                                          */
/*                 cCameraFormelle                          */
/*                                                          */
/************************************************************/


void cCameraFormelle::cEqAppui::GenCode()
{
    cElCompileFN::DoEverything
    (
	DIRECTORY_GENCODE_FORMEL,
	mNameType,
	mEcarts,
	mLInterv
    );
}



/*
 *   Pour des raisons "historique", avec points d'appuis fixe on a 2 fcteur separes
 *   en x et y
*/


cCameraFormelle::cEqAppui::cEqAppui
(
     bool wDist,
     bool isGL,
     bool isProj,
     bool isPTerrainFixe,
     bool Comp,
     cCameraFormelle & aCam,
     bool Code2Gen,
     bool IsEqDroite
)  :
    mCam            (aCam),
    mUseEqNoVar     ((! wDist) && (!UseTjsDist) && (! Code2Gen)),
    mIsPTerrainFixe (isPTerrainFixe),
    mNameType       (
                        std::string("cEqAppui") 
                      + std::string(IsEqDroite ? "_Droite" : "")
                      + std::string(mCam.mIntr.UseAFocal() ? "_AFocal" : "")
                      +std::string(wDist ? "" : "_NoDist_" )
                      +std::string(isGL ? "_GL_" : "")
                      + std::string
                        (
                                isPTerrainFixe ? 
                                std::string("_TerFix_") : 
                                std::string(isProj ? "_PProjInc_" : "_PTInc_")
                        )
		      + std::string(mCam.mIntr.DistIsC2M() ? "C2M" : "M2C")
		      + ((mUseEqNoVar) ? "NoVar" : mCam.mIntr.NameType() )
		    ),
    mNameTerX       ("XTer"),
    mNameTerY       ("YTer"),
    mNameTerZ       ("ZTer"),
    mNameStdScaleN  ("ScNorm"),
    mEqP3I          (isPTerrainFixe ? 0 : mCam.Set().Pt3dIncTmp()),
    mNameImX        ("XIm"),
    mNameImY        ("YIm"),
    mPTerrain       (   isPTerrainFixe ?
                        Pt3d<Fonc_Num>(cVarSpec(0,mNameTerX),cVarSpec(0,mNameTerY),cVarSpec(0,mNameTerZ)):
		        mEqP3I->PF()
                    ),
    mPIm            (  cVarSpec(0,mNameImX),cVarSpec(0,mNameImY)),
    mFScN           (cVarSpec(0,mNameStdScaleN)),
    mEcarts         (),
    mFoncEqResidu   (0),
    mProjP0         (isProj ? new cP3d_Etat_PhgrF("ProjP0"): 0),
    mProjI          (isProj ? new cP3d_Etat_PhgrF("ProjI"): 0),
    mProjJ          (isProj ? new cP3d_Etat_PhgrF("ProjJ"): 0),
    mProjK          (isProj ? new cP3d_Etat_PhgrF("ProjK"): 0),
    mMatriceGL      (isGL ? new cMatr_Etat_PhgrF("GL",3,3) : 0),
    mNDP0           (wDist ? 0 : new cP2d_Etat_PhgrF("NDP0")),
    mNDdx           (wDist ? 0 : new cP2d_Etat_PhgrF("NDdx")),
    mNDdy           (wDist ? 0 : new cP2d_Etat_PhgrF("NDdy")),
    mEqDroite       (IsEqDroite)
{
   if (Code2Gen)  // En mode normal, on ne modifie pas la camera
   {
       mCam.SetGL(isGL,ElRotation3D::Id);
   }

   if (isProj)
   {
       ELISE_ASSERT(!isPTerrainFixe,"Incoh in cCameraFormelle::cEqAppui::cEqAppui");
       mPTerrain =   mProjP0->PtF() 
                   + mProjI->PtF()*mPTerrain.x
                   + mProjJ->PtF()*mPTerrain.y
                   + mProjK->PtF()*mPTerrain.z;
   }

    Pt3d<Fonc_Num> aRay = mPTerrain -  mCam.COptF();
    aRay = mCam.mRot->VectM2C(aRay);

    Pt2d<Fonc_Num> aP1,aP2;

    if (wDist)
    {
       if (mCam.mIntr.DistIsC2M())
       {
           
          mCam.mIntr.AssertNoAFocalParam("Dist Cam->Monde");
          aP1  = mCam.mIntr.DirRayMonde2CorrDist(aRay);
          aP2 =  mCam.mIntr.DistorC2M(mPIm);
       }
       else
       {
           aP1 = mCam.mIntr.DirRayMonde2Cam(aRay,0);
           aP2 = mPIm;
       }
    }
    else
    {
          // Devrait pouvoir marcher, mas apres modif pour etre la differentiel en x,y,z ...
          if (! AFocalAcceptNoDist)
              mCam.mIntr.AssertNoAFocalParam("Cas sans dist pas encore gere");
          aP1 =  Pt2d<Fonc_Num>(aRay.x/aRay.z,aRay.y/aRay.z);
          aP1 = mNDP0->PtF() + mNDdx->PtF()*aP1.x + mNDdy->PtF() *aP1.y;
           // aP1 = y->PtF() *aP1.y;
          // aP1  = mCam.mIntr.DirRayMonde2CorrDist(aRay);
          aP2 = mPIm;
    }

     if (IsEqDroite)
     {
        // Pour reutiliser au max "l'infrastructure" existante on prend la convention que
        // aP2.x -> rho aP2.y -> Theta et que cela code l'equation normale de la droite 
        // rho = cos(Teta) X + sin(Theta) Y
        Fonc_Num fEcart = (aP2.x -cos(aP2.y)*aP1.x - sin(aP2.y)* aP1.y) * mFScN;
        mEcarts.push_back(fEcart);
     }
     else
     {
         Pt2d<Fonc_Num> fEcart = (aP1-aP2) * mFScN;
         // Les ecarts sont des radians !
         mEcarts.push_back(fEcart.x);
         mEcarts.push_back(fEcart.y);
     }


    if (Comp || Code2Gen)
    {
       mCam.PIF().IncInterv().SetName("Intr");
       mCam.RF().IncInterv().SetName("Orient");

       mLInterv.AddInterv(mCam.RF().IncInterv());
       if (! mUseEqNoVar)
       {
          mCam.PIF().AddToListInterval(mLInterv);
          // mLInterv.AddInterv(mCam.PIF().IncInterv());
       }

       if (!isPTerrainFixe)
       {
           mLInterv.AddInterv(mEqP3I->IncInterv());
       }

       mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
       if (Code2Gen)
       {
          GenCode();
	  return;
       }
       if (mFoncEqResidu == 0)
       {
          std::cout << "NAME = " << mNameType << "\n";
	  ELISE_ASSERT(false,"Can Get Code Comp for cCameraFormelle::cEqAppui");
	  mFoncEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,Fonc_Num(0));
       }

       mFoncEqResidu->SetMappingCur(mLInterv,&mCam.Set());

       // Adr = 
       mCam.mSet.AddFonct(mFoncEqResidu);

       if (isPTerrainFixe)
       {
          pAdrXTer =  mFoncEqResidu->RequireAdrVarLocFromString(mNameTerX);
          pAdrYTer =  mFoncEqResidu->RequireAdrVarLocFromString(mNameTerY);
          pAdrZTer =  mFoncEqResidu->RequireAdrVarLocFromString(mNameTerZ);
       }
       else
       {
           pAdrXTer  = pAdrYTer = pAdrZTer =0;
       }

       if (isProj)
       {
           mProjP0->InitAdr(*mFoncEqResidu);
           mProjI->InitAdr(*mFoncEqResidu);
           mProjJ->InitAdr(*mFoncEqResidu);
           mProjK->InitAdr(*mFoncEqResidu);
       }

       if (isGL)
       {
           mMatriceGL->InitAdr(*mFoncEqResidu);
       }

       if (!wDist)
       {
          mNDP0->InitAdr(*mFoncEqResidu);
          mNDdx->InitAdr(*mFoncEqResidu);
          mNDdy->InitAdr(*mFoncEqResidu);
       }


       pAdrXIm = mFoncEqResidu->AdrVarLocFromString(mNameImX);
       pAdrYIm = mFoncEqResidu->AdrVarLocFromString(mNameImY);
       pAdrScN = mFoncEqResidu->RequireAdrVarLocFromString(mNameStdScaleN);

      mCam.PIF().InitStateOfFoncteur(mFoncEqResidu,0);
/*
       if (wDist)
       {
          aCam.PIF().InitStateOfFoncteur(mFoncEqResidu,0);
       }
*/
    }
}

void  cCameraFormelle::cEqAppui::PrepareEqFForPointIm(const Pt2dr & aPIm)
{
     mCam.PrepareEqFForPointIm(mLInterv,mFoncEqResidu,aPIm,mEqDroite,0);
}

Pt2dr cCameraFormelle::cEqAppui::Residu(Pt3dr aPTer,Pt2dr aPIm,REAL aPds)
{
    ELISE_ASSERT(mFoncEqResidu!=0,"cCameraFormelle::cEqAppui::Residu");
    ELISE_ASSERT(mIsPTerrainFixe,"cCameraFormelle::cEqAppui::Residu");
    *pAdrXTer = aPTer.x;
    *pAdrYTer = aPTer.y;
    *pAdrZTer = aPTer.z;
    *pAdrScN = mCam.PIF().StdScaleNNoGrid();


   if (pAdrXIm) 
      *pAdrXIm =  aPIm.x;
   if (pAdrYIm) 
      *pAdrYIm =  aPIm.y;

    if (mCam.IsGL())
    {
       mMatriceGL->SetEtat(mCam.RF().MGL());
    }
    PrepareEqFForPointIm(aPIm);

    // std::cout <<  "cEA::RES " <<  mNameType << "\n";

   std::vector<double> aRes = (aPds > 0)   ?
                            mCam.mSet.VAddEqFonctToSys(mFoncEqResidu,aPds,false,NullPCVU) :
                            mCam.mSet.VResiduSigne(mFoncEqResidu);
/*
   if (aPds > 0)
       return mCam.mSet.AddEqFonctToSys(mFoncEqResidu,aPds,false);

   double aRes = mCam.mSet.ResiduSigne(mFoncEqResidu);
   return aRes;
*/

    // std::cout << pAdrScN  << " " <<  mCam.PIF().StdScaleN() << "\n";;
   // std::cout << "wdfqhtt  " << mNameType << " " << aRes << "\n";
   return Pt2dr(aRes.at(0), mEqDroite ? 0.0  : aRes.at(1));
}


Pt2dr cCameraFormelle::cEqAppui::ResiduPInc(Pt2dr aPIm,REAL aPds,const cParamPtProj & aPPP,cParamCalcVarUnkEl * aPCVU)
{
  // std::cout <<"DEBUG33 " << mNameType << "\n";
  // std::cout <<  "WWWW : " << mNameType << "\n";
    ELISE_ASSERT(mFoncEqResidu!=0,"cCameraFormelle::cEqAppui::Residu");
    ELISE_ASSERT(!mIsPTerrainFixe,"cCameraFormelle::cEqAppui::Residu");
    if (aPPP.mProjIsInit)
    {

        mProjP0->SetEtat(aPPP.mP0);
        mProjI->SetEtat(aPPP.mI);
        mProjJ->SetEtat(aPPP.mJ);
        mProjK->SetEtat(aPPP.mK);
    }
    if (mCam.IsGL())
    {
       mMatriceGL->SetEtat(mCam.RF().MGL());
    }
    if (! aPPP.wDist)
    {
          mNDP0->SetEtat(aPPP.mNDP0);
          mNDdx->SetEtat(aPPP.mNDdx);
          mNDdy->SetEtat(aPPP.mNDdy);
    }

    *pAdrXIm =  aPIm.x;
    *pAdrYIm =  aPIm.y;
    *pAdrScN = mCam.PIF().StdScaleN();
    PrepareEqFForPointIm(aPIm);

    // DEBUG_LSQ
    const std::vector<REAL> & aVals = 
                  (aPds > 0)                                             ?
                  mCam.mSet.VAddEqFonctToSys(mFoncEqResidu,aPds,false,aPCVU)   :
		  mCam.mSet.VResiduSigne(mFoncEqResidu)                  ;

    Pt2dr aRes(aVals[0], mEqDroite ? 0.0 : aVals[1]);

    // std::cout << aPPP.wDist << " " <<  mCam.PIF().StdScaleN() << "wdfqhtt  " << mNameType << " " << aRes << "\n";
    return aRes;
}

cIncListInterv & cCameraFormelle::cEqAppui::LInterv()
{
    return mLInterv;
}

/*
cMatr_Etat_PhgrF &  cCameraFormelle::MatRGL(bool isP)
{
    return mRot->MatGL(isP);
}
*/

void cCameraFormelle::SetGL(bool aModeGL,const ElRotation3D & aGuimb2CM)
{
    mRot->SetGL(aModeGL,aGuimb2CM);
}
bool cCameraFormelle::IsGL() const
{
    return mRot->IsGL();
}

cCameraFormelle::cCameraFormelle
(
     eModeContrRot  aMode,
     ElRotation3D   aRot,
     cParamIntrinsequeFormel & anIntr,
     cCameraFormelle *         aCamAtt,
     const std::string & aName,
     bool  CompEqAppui,
     bool  GenCodeAppui,
     bool  HasEqDroite
)  :
   cGenPDVFormelle(*(anIntr.Set())),
   pCamAttach  (aCamAtt),
   mIntr       (anIntr),
   mRot        (mSet.NewRotation(aMode,aRot, ((pCamAttach==0) ? 0 : pCamAttach->mRot),aName)),
   mName       (aName),
   mNameIm     (""),

   mEqAppuiTerNoGL(NULL),
   mEqAppuiTerGL(NULL),


   mEqAppuiIncXY (0),
   mEqAppuiProjIncXY (0),
   mEqAppuiGLIncXY (0),
   mEqAppuiGLProjIncXY (0),

   mEqAppuiSDistIncXY (0),
   mEqAppuiSDistProjIncXY (0),
   mEqAppuiSDistGLIncXY (0),
   mEqAppuiSDistGLProjIncXY (0),
   mCameraCourante(NULL),
   mHasEqDroite   (HasEqDroite)
{
   for (int aKEqDr=0 ; aKEqDr<TheNbEqDr; aKEqDr++)
   {
            mEqAppuiDroite[aKEqDr] = 0;
   }
	// NO_WARN
   mEqAppuiTerNoGL = new cEqAppui(true,false,false,true,CompEqAppui,*this,GenCodeAppui,false);
   mEqAppuiTerGL	 = new cEqAppui(true,true ,false,true,CompEqAppui,*this,GenCodeAppui,false);
   mCameraCourante	 = CalcCameraCourante();

}


void cCameraFormelle::SetNameIm(const std::string & aNameIm)
{
   mNameIm = aNameIm;
}

const std::string  & cCameraFormelle::NameIm() const
{
   return mNameIm;
}


cCameraFormelle::~cCameraFormelle(){
	// we should delete mEqAppuiTerNoGL and mEqAppuiTerGL but it makes apero crash
	// something is probably Using them after their natural lifetime
	//if ( mEqAppuiTerNoGL_II!=NULL ) delete mEqAppuiTerNoGL;
	//if ( mEqAppuiTerGL_II!=NULL ) delete mEqAppuiTerNoGL;
}

Pt2dr cCameraFormelle::AddEqAppuisInc(const Pt2dr & aPIm,double aPds,cParamPtProj & aPPP,bool IsEqDroite,cParamCalcVarUnkEl * aPCVU)
{
     cCamStenopeGrid * aCSG = mIntr.CamGrid();
     if ( aCSG)
     {
        aPPP.wDist = false;
        Pt2dr aPLoc  = ProjStenope(mCameraCourante->R3toL3(aPPP.mTer));

        aPPP.mNDP0 =aCSG->L2toF2AndDer(aPLoc,aPPP.mNDdx,aPPP.mNDdy);
        // Si on remonte jusqu'a PtImGrid::ValueAndDer, on voit que GradX est la derive de X selon x et y (et non la derive selon x de X et Y)
        ElSwap(aPPP.mNDdx.y,aPPP.mNDdy.x);
        aPPP.mNDP0 = aPPP.mNDP0 - aPPP.mNDdx * aPLoc.x - aPPP.mNDdy * aPLoc.y;

// std::cout << "cCF:AEI Loc " << aPLoc << " " << aPPP.mNDP0 << " " << aPPP.mNDdx <<  " " << aPPP.mNDdy << "\n";

     }
     else
        aPPP.wDist = true;

     cEqAppui*  anEq = AddForUseFctrEqAppuisInc ( false, aPPP.mProjIsInit, aPPP.wDist,IsEqDroite);
     Pt2dr aRes = anEq->ResiduPInc(CorrigePFromDAdd(aPIm,true,IsEqDroite),aPds,aPPP,aPCVU);



     if ( std_isnan(aRes.x) || std_isnan(aRes.y))
     {
         std::cout << anEq->mNameType;
         std::cout << "Im "<<  aPIm 
                   << " ter " << aPPP.mTer 
                   << " Proj " << aPPP.mProjIsInit 
                   << " CentreCam "  << CameraCourante()->VraiOpticalCenter()
                   << "\n";
         ELISE_ASSERT(false,"Nan in cCameraFormelle::AddEqAppuisInc");
     }
// std::cout << "mmmmmMmm " << aRes << mIntr.CamInit()->ResiduMond2Cam(aRes) << "\n";
  //   return aRes;

    return mResiduM2C.IVect(mIntr.CamInit()->ResiduMond2Cam(aRes));
}

ElAffin2D & cCameraFormelle::ResiduM2C()
{
   return mResiduM2C;
}

cIncListInterv & cCameraFormelle::IntervAppuisPtsInc()
{
   int aNbEqDr =(mHasEqDroite  ? 2  : 1);
   for (int aKEqDr=0 ; aKEqDr<aNbEqDr ; aKEqDr++)
   {
       bool WithEqDr = (aKEqDr==1);
       for (int aKDist=0 ; aKDist<2 ; aKDist++)
       {
           bool wDist = (aKDist==0);
           if (wDist || (!mIntr.UseAFocal())  ||   ( AFocalAcceptNoDist))
           {
               AddFctrEqAppuisInc(false,false,false,wDist,WithEqDr);
               AddFctrEqAppuisInc(false,true,false,wDist,WithEqDr);
               AddFctrEqAppuisInc(false,false,true,wDist,WithEqDr);
               AddFctrEqAppuisInc(false,true,true,wDist,WithEqDr);
           }
       }
   }
// std::cout << "GLglgl " << IsGL() << "\n";getchar();

   if (IsGL())
   {
      ELISE_ASSERT
      (
           mEqAppuiGLIncXY->LInterv().Equal(mEqAppuiGLProjIncXY->LInterv()),
           "cCameraFormelle::IntervAppuisPtsInc"
      );
      return mEqAppuiGLProjIncXY->LInterv();
   }

   ELISE_ASSERT
   (
           mEqAppuiIncXY->LInterv().Equal(mEqAppuiProjIncXY->LInterv()),
           "cCameraFormelle::IntervAppuisPtsInc"
   );

  if ( UseTjsDist)
  {
      ELISE_ASSERT
      (
           mEqAppuiIncXY->LInterv().Equal(mEqAppuiSDistIncXY->LInterv()),
           "cCameraFormelle::IntervAppuisPtsInc"
      );
  }

   return mEqAppuiIncXY->LInterv();
}


cCameraFormelle::cEqAppui * cCameraFormelle::AddForUseFctrEqAppuisInc(bool aGenCode,bool isProj,bool wDist,bool IsEqDroite)
{
   return AddFctrEqAppuisInc(aGenCode,isProj,IsGL(),wDist,IsEqDroite);
}

cCameraFormelle::cEqAppui * cCameraFormelle::AddFctrEqAppuisInc(bool aGenCode,bool isProj,bool isGL,bool wDist,bool IsEqDroite)
{
  if (IsEqDroite)
  {
       ELISE_ASSERT(!mIntr.UseAFocal(),"EqDroite incompatible with AFocal\n");
       int aK = (isProj==true) + 2 * (isGL==true) + 4 * (wDist==true);

       if (mEqAppuiDroite[aK] == 0)
           mEqAppuiDroite[aK] =  new cEqAppui(wDist,isGL,isProj,false,true,*this,aGenCode,true);

      return mEqAppuiDroite[aK];
       // Avec ss dist
       // Avec ss Proj
       // Avec ss GL 
  }


  if (wDist  || (mIntr.UseAFocal() && (!AFocalAcceptNoDist)))
  {
      if (isProj)
      {
         if(isGL)
         {
            if (mEqAppuiGLProjIncXY==0)
            {
                mEqAppuiGLProjIncXY = new cEqAppui(wDist,true,true,false,true,*this,aGenCode,false);
            }
            return mEqAppuiGLProjIncXY;
         }
         else
         {
            if (mEqAppuiProjIncXY==0)
            {
                mEqAppuiProjIncXY = new cEqAppui(wDist,false,true,false,true,*this,aGenCode,false);
            }
            return mEqAppuiProjIncXY;
         }
      }
      else
      {
         if(isGL)
         {
             if (mEqAppuiGLIncXY==0) 
             {
                mEqAppuiGLIncXY = new cEqAppui(wDist,true,false,false,true,*this,aGenCode,false);
             }
             return mEqAppuiGLIncXY;
         }
         else
         {
             if (mEqAppuiIncXY==0) 
             {
                mEqAppuiIncXY = new cEqAppui(wDist,false,false,false,true,*this,aGenCode,false);
             }
             return mEqAppuiIncXY;
         }
     }
  }
  else
  {
      if (isProj)
      {
         if(isGL)
         {
            if (mEqAppuiSDistGLProjIncXY==0)
            {
                mEqAppuiSDistGLProjIncXY = new cEqAppui(wDist,true,true,false,true,*this,aGenCode,false);
            }
            return mEqAppuiSDistGLProjIncXY;
         }
         else
         {
            if (mEqAppuiSDistProjIncXY==0)
            {
                mEqAppuiSDistProjIncXY = new cEqAppui(wDist,false,true,false,true,*this,aGenCode,false);
            }
            return mEqAppuiSDistProjIncXY;
         }
      }
      else
      {
         if(isGL)
         {
             if (mEqAppuiSDistGLIncXY==0) 
             {
                mEqAppuiSDistGLIncXY = new cEqAppui(wDist,true,false,false,true,*this,aGenCode,false);
             }
             return mEqAppuiSDistGLIncXY;
         }
         else
         {
             if (mEqAppuiSDistIncXY==0) 
             {
                mEqAppuiSDistIncXY = new cEqAppui(wDist,false,false,false,true,*this,aGenCode,false);
             }
             return mEqAppuiSDistIncXY;
         }
     }
  }


  ELISE_ASSERT(false,"cCameraFormelle::AddFctrEqAppuisInc");
  return 0;
}


void  cCameraFormelle::PrepareEqFForPointIm(const cIncListInterv & anII,cElCompiledFonc * anEq,const Pt2dr & aPIm,bool EqDroite,int aKCam)
{
   mIntr.PrepareEqFForPointIm(anII,anEq,aPIm,EqDroite,aKCam);
}

void cCameraFormelle::TestVB10(const std::string& aMes) const
{
   std::cout << aMes ;
   
   if (mEqAppuiIncXY==0)
     std::cout << " XXXX PtrNull \n";
   else
     std::cout << " VB10=" <<  mEqAppuiIncXY->mFoncEqResidu->ValBrute(10) << "\n";
}


Pt2dr  cCameraFormelle::CorrigePFromDAdd(const Pt2dr & aP,bool UseGrid,bool ModeDr)
{
    if (ModeDr)
    {
        SegComp aSeg = SegComp::FromRhoTeta(aP);
        Pt2dr aP0 = mIntr.CorrigePFromDAdd(aSeg.p0(),UseGrid);
        Pt2dr aP1 = mIntr.CorrigePFromDAdd(aSeg.p1(),UseGrid);

        return SegComp(aP0,aP1).ToRhoTeta();
    }
    else
    {
        return mIntr.CorrigePFromDAdd(aP,UseGrid);
    }
}

Pt2dr  cCameraFormelle::AddAppui(Pt3dr aP,Pt2dr aPIm,REAL aPds)
{
   // ELISE_ASSERT(! mRot->IsGL(),"Do not handle cCameraFormelle::ResiduAppui in mode GL");
   aPIm = CorrigePFromDAdd(aPIm,false,false);
   cEqAppui * anEq = mRot->IsGL() ? mEqAppuiTerGL : mEqAppuiTerNoGL ;
   return anEq->Residu(aP,aPIm,aPds);
}

Pt2dr  cCameraFormelle::ResiduAppui(Pt3dr aP,Pt2dr aPIm)
{
   return AddAppui(aP,aPIm,-1);
/*
   ELISE_ASSERT(! mRot->IsGL(),"Do not handle cCameraFormelle::ResiduAppui in mode GL");
   aPIm = CorrigePFromDAdd(aPIm,false);
   return Pt2dr
	   (
	        mEqAppuiX.Residu(aP,aPIm,-1),
	        mEqAppuiY.Residu(aP,aPIm,-1)
	   );
*/
}




const std::string & cCameraFormelle::Name() const
{
  return mName;
}

cNameSpaceEqF::eModeContrRot cCameraFormelle::ModeRot() const
{
	return mRot->ModeRot();
}

void cCameraFormelle::SetModeRot(eModeContrRot aMode)
{
     mRot->SetModeRot(aMode);
}


Pt3d<Fonc_Num> cCameraFormelle::COptF()
{
  return mRot->COpt();
}

ElRotation3D cCameraFormelle::CurRot()
{
  return mRot->CurRot();
}


CamStenope * cCameraFormelle::CalcCameraCourante()
{
   CamStenope * aCS = mIntr.DupCurPIF();
   aCS->SetIdentCam(mName);
   aCS->SetNameIm(mNameIm);
// std::cout << "Hhhhhhhhhhhjkj  NAME = " << mName << "\n";
   aCS->Dist().SetName(mName.c_str());
   aCS->SetOrientation(CurRot().inv());
   return aCS;
}

CamStenope *  cCameraFormelle::DuplicataCameraCourante() 
{
   return CalcCameraCourante();
}


const CamStenope * cCameraFormelle::CameraCourante()  const
{
   return mCameraCourante;
}

CamStenope * cCameraFormelle::NC_CameraCourante() 
{
   return mCameraCourante;
}


const cBasicGeomCap3D * cCameraFormelle::GPF_CurBGCap3D() const 
{
    return CameraCourante();
}

cBasicGeomCap3D * cCameraFormelle::GPF_NC_CurBGCap3D() 
{
    return NC_CameraCourante();
}



void  cCameraFormelle::Update_0F2D()
{
    mCameraCourante = CalcCameraCourante();
}

void     cCameraFormelle::SetCurRot(const ElRotation3D & aR2CM,const ElRotation3D & aGuimb2CM)
{
     mRot->SetCurRot(aR2CM,aGuimb2CM);
     mCameraCourante->SetOrientation(mRot->CurRot().inv());
     // mCameraCourante->SetOrientation(aR2CM.inv());
     Update_0F2D();
}

/*
CamStenope * cCameraFormelle::CameraCouranteInBuf()
{
}
*/


Pt3d<Fonc_Num>   cCameraFormelle::DirRayonF(Pt2d<Fonc_Num> aP,int aKCam)
{
  return mRot->ImVect(mIntr.Cam2DirRayMonde(aP,aKCam));
}



bool cCameraFormelle::SameIntr(const cCameraFormelle & aCam2) const
{
    return &mIntr == &(aCam2.mIntr);
}

cRotationFormelle & cCameraFormelle::RF()
{
   return *mRot;
}

cParamIntrinsequeFormel & cCameraFormelle::PIF()
{
    return mIntr;
}



/************************************************************/
/*                                                          */
/*                 cCpleCamFormelle                         */
/*                                                          */
/************************************************************/

static Fonc_Num FoncEqCoPlan
                (
                     Pt3d<Fonc_Num>    aBaseNN,
                     Pt3d<Fonc_Num>    aRay1,
                     Pt3d<Fonc_Num>    aRay2
		)
{
	Pt3d<Fonc_Num> u = aRay1 ^ aRay2;

	return scal(u,aBaseNN) / euclid(aBaseNN);
}


static Fonc_Num FoncResIm1(Pt3d<Fonc_Num> B,Pt3d<Fonc_Num> U1,Pt3d<Fonc_Num> U2)
{
	Pt3d<Fonc_Num> V = U1 ^ U2;
	Pt3d<Fonc_Num> BU2 = B ^ U2;

	return (euclid(V)/euclid(U1)) *(scal(BU2,U1)/scal(BU2,V));
}

static Fonc_Num FoncResIm2(Pt3d<Fonc_Num> B,Pt3d<Fonc_Num> U1,Pt3d<Fonc_Num> U2)
{
    return FoncResIm1(B,U2,U1);
}

static Fonc_Num Residu
                (
		     cNameSpaceEqF::eModeResidu aModeResidu,
		     Pt3d<Fonc_Num> B,
		     Pt3d<Fonc_Num> U1,
		     Pt3d<Fonc_Num> U2
                )
{
	switch(aModeResidu)
	{
		case cNameSpaceEqF::eResiduCoplan : return FoncEqCoPlan(B,U1,U2);
		case cNameSpaceEqF::eResiduIm1    : return FoncResIm1(B,U1,U2);
		case cNameSpaceEqF::eResiduIm2    : return FoncResIm2(B,U1,U2);
	}
	ELISE_ASSERT(false,"Bas Residu Names");
	return 0;
}

cCpleCamFormelle::~cCpleCamFormelle()
{
}

cCameraFormelle & cCpleCamFormelle::Cam2() {return mCam2;}
cCameraFormelle & cCpleCamFormelle::Cam1() {return mCam1;}

cCpleCamFormelle::cCpleCamFormelle
(
    cCameraFormelle & aCam1,
    cCameraFormelle & aCam2,
    eModeResidu       aModeResidu,
    bool              Code2Gen
) :
    mCam1       (aCam1),
    mCam2       (aCam2),
    mSet        (aCam1.Set()),
    mBaseNN     (aCam2.COptF() - aCam1.COptF() ),
    mBaseU      (mBaseNN/euclid(mBaseNN)),
    mModeResidu (aModeResidu),
    mEqResidu   (Residu(aModeResidu,mBaseNN,mCam1.DirRayonF(mP1,0),mCam2.DirRayonF(mP2,1))),
    mNameType   (     NameOfTyRes(aModeResidu)
		    + aCam1.PIF().NameType()
		    + (   aCam1.SameIntr(aCam2)  ?
			  "Id"                   :
			  aCam2.PIF().NameType()
		      )
		)
{
    aCam1.RF().IncInterv().SetName("Or1");
    aCam2.RF().IncInterv().SetName("Or2");
    aCam1.PIF().IncInterv().SetName("Intr1");

    mLInterv.AddInterv(aCam1.RF().IncInterv());
    mLInterv.AddInterv(aCam2.RF().IncInterv());
    mLInterv.AddInterv(aCam1.PIF().IncInterv());
    if (! aCam1.SameIntr(aCam2))
    {
        aCam2.PIF().IncInterv().SetName("Intr2");
        mLInterv.AddInterv(aCam2.PIF().IncInterv());
    }

    mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);

    if (Code2Gen)
    {
        GenCode();
	return;
    }
    if (mFoncEqResidu==0)
    {
	    for (INT aK= 0 ; aK< 20 ; aK++)
	       cout << "FONCTEUR DYN FOR " << mNameType << "\n";
	mFoncEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,mEqResidu);
	// ELISE_ASSERT(false,"Dont Get CompVal for  cCpleCamFormelle");
    }

    mFoncEqResidu->SetMappingCur(mLInterv,&mSet);

    pAdrX1 = mFoncEqResidu->RequireAdrVarLocFromString(mMemberX1);
    pAdrY1 = mFoncEqResidu->RequireAdrVarLocFromString(mMemberY1);
    pAdrX2 = mFoncEqResidu->RequireAdrVarLocFromString(mMemberX2);
    pAdrY2 = mFoncEqResidu->RequireAdrVarLocFromString(mMemberY2);
	    
    aCam1.PIF().InitStateOfFoncteur(mFoncEqResidu,0);
    aCam2.PIF().InitStateOfFoncteur(mFoncEqResidu,1);
    mSet.AddFonct(mFoncEqResidu);
}

void cCpleCamFormelle::GenCode()
{
    cElCompileFN::DoEverything
    (
	DIRECTORY_GENCODE_FORMEL,
	mNameType,
	mEqResidu,
	mLInterv
    );
}


void cCpleCamFormelle::CorrigeP1P2FromDAdd(Pt2dr & aP1,Pt2dr & aP2)
{
   aP1 = mCam1.CorrigePFromDAdd(aP1,false,false);
   aP2 = mCam2.CorrigePFromDAdd(aP2,false,false);
}


REAL cCpleCamFormelle::AddLiaisonP1P2(Pt2dr aP1,Pt2dr aP2,REAL aPds,bool WithD2)
{
   CorrigeP1P2FromDAdd(aP1,aP2);
   *pAdrX1 = aP1.x;
   *pAdrY1 = aP1.y;
   *pAdrX2 = aP2.x;
   *pAdrY2 = aP2.y;
   return mSet.AddEqFonctToSys(mFoncEqResidu,aPds,WithD2);

}

REAL cCpleCamFormelle::ResiduSigneP1P2(Pt2dr aP1,Pt2dr aP2)
{
   CorrigeP1P2FromDAdd(aP1,aP2);
   *pAdrX1 = aP1.x;
   *pAdrY1 = aP1.y;
   *pAdrX2 = aP2.x;
   *pAdrY2 = aP2.y;
   return mSet.ResiduSigne(mFoncEqResidu);
}
 

/************************************************************/
/*                                                          */
/*                 cCpleGridEq                              */
/*                                                          */
/************************************************************/

const std::string cCpleGridEq::NamePdsA1 = "PdsA1";
const std::string cCpleGridEq::NamePdsA2 = "PdsA2";
const std::string cCpleGridEq::NamePdsA3 = "PdsA3";

const std::string cCpleGridEq::NamePdsB1 = "PdsB1";
const std::string cCpleGridEq::NamePdsB2 = "PdsB2";
const std::string cCpleGridEq::NamePdsB3 = "PdsB3";





static Pt3d<Fonc_Num> Ray
               (
                    cTriangulFormelle & aTriGul,
                    cRotationFormelle & aRot,
                    const std::string & aName1,
                    const std::string & aName2,
                    const std::string & aName3
                )
{
   ELISE_ASSERT(false,"Grid Dist non retablies !!");
   return Pt3d<Fonc_Num>(0,0,0);
/*
    Fonc_Num PA1 = cVarSpec(0,aName1); 
    Fonc_Num PA2 = cVarSpec(0,aName2); 
    Fonc_Num PA3 = cVarSpec(0,aName3); 
    const cTFI_Triangle & aTri = aTriGul.GetTriFromP(aTriGul.APointInTri());

    Pt2d<Fonc_Num> aP = aTri.PInc1()*PA1 + aTri.PInc2()*PA2 + aTri.PInc3()*PA3;

    Pt3d<Fonc_Num> aRay(aP.x,aP.y,1.0);
    return aRot.ImVect(aRay);
*/
}

Fonc_Num cCpleGridEq::EqCoPlan()
{
    Pt3d<Fonc_Num> aRayA = Ray(mTriA,mRotA,NamePdsA1,NamePdsA2,NamePdsA3);
    Pt3d<Fonc_Num> aRayB = Ray(mTriB,mRotB,NamePdsB1,NamePdsB2,NamePdsB3);
    return Residu
	    (
	       mModeResidu,
               mRotB.COpt() - mRotA.COpt(),
               Ray(mTriA,mRotA,NamePdsA1,NamePdsA2,NamePdsA3),
               Ray(mTriB,mRotB,NamePdsB1,NamePdsB2,NamePdsB3)
	    );

}

void cCpleGridEq::GenCode()
{
    cElCompileFN::DoEverything
    (
	DIRECTORY_GENCODE_FORMEL,
	mNameType,
	EqCoPlan(),
	mLInterv
    );
}




REAL cCpleGridEq::AddLiaisonP1P2(Pt2dr aP1,Pt2dr aP2,REAL aPds,bool WithD2)
{

   SetP1P2(aP1,aP2);
   REAL aRes =  mSet.AddEqFonctToSys(mFoncEqCoP,aPds,WithD2);

   return aRes;
}

REAL cCpleGridEq::ResiduSigneP1P2(Pt2dr aP1,Pt2dr aP2)
{
   SetP1P2(aP1,aP2);
   return mSet.ResiduSigne(mFoncEqCoP);
}

void ShowInterv(const cIncIntervale & I,AllocateurDInconnues & Alloc)
{
	
    cout << "Int::" << I.Id() << "=["
         << I.I0Alloc() << " - " << I.I1Alloc() 
	 << "]" 
	 << " [I0]= " << Alloc.GetVar(I.I0Alloc())
	 << " [I0+1]= " << Alloc.GetVar(I.I0Alloc()+1)
	 << "\n";
}


void cCpleGridEq::SetP1P2(Pt2dr aPA,Pt2dr aPB)
{

     const cTFI_Triangle & aTrianglA = mTriA.GetTriFromP(aPA);
     Pt3dr CoordA = aTrianglA.TriGeom().CoordBarry(aPA);

     const cTFI_Triangle & aTrianglB = mTriB.GetTriFromP(aPB);
     Pt3dr CoordB = aTrianglB.TriGeom().CoordBarry(aPB);


     mLInterv.ResetInterv(aTrianglA.IntervA1());
     mLInterv.ResetInterv(aTrianglA.IntervA2());
     mLInterv.ResetInterv(aTrianglA.IntervA3());

     *pAdrPdsA1 = CoordA.x;
     *pAdrPdsA2 = CoordA.y;
     *pAdrPdsA3 = CoordA.z;



     mLInterv.ResetInterv(aTrianglB.IntervB1());
     mLInterv.ResetInterv(aTrianglB.IntervB2());
     mLInterv.ResetInterv(aTrianglB.IntervB3());

     *pAdrPdsB1 = CoordB.x;
     *pAdrPdsB2 = CoordB.y;
     *pAdrPdsB3 = CoordB.z;

     mFoncEqCoP->SetMappingCur(mLInterv,&mSet);
}



cCpleGridEq::cCpleGridEq
(
     cTriangulFormelle & aTriA, 
     cRotationFormelle & aRotA,
     cTriangulFormelle & aTriB, 
     cRotationFormelle & aRotB,
     eModeResidu         aModeResidu,
     bool                Code2Gen

)  :
   mSet        (aTriA.Set()),
   mTriA       (aTriA),
   mRotA       (aRotA),
   mTriB       (aTriB),
   mRotB       (aRotB),
   mModeResidu (aModeResidu),
   mNameType   (NameOfTyRes(aModeResidu)+"Grid")
{
     mRotA.IncInterv().SetName("OrA");
     mRotB.IncInterv().SetName("OrB");

     const cTFI_Triangle & aTrianglA = mTriA.GetTriFromP(mTriA.APointInTri());
     const cTFI_Triangle & aTrianglB = mTriB.GetTriFromP(mTriB.APointInTri());

     mLInterv.AddInterv(mRotA.IncInterv());
     mLInterv.AddInterv(mRotB.IncInterv());

     mLInterv.AddInterv(aTrianglA.IntervA1(),true);
     mLInterv.AddInterv(aTrianglA.IntervA2(),true);
     mLInterv.AddInterv(aTrianglA.IntervA3(),true);

     mLInterv.AddInterv(aTrianglB.IntervB1(),true);
     mLInterv.AddInterv(aTrianglB.IntervB2(),true);
     mLInterv.AddInterv(aTrianglB.IntervB3(),true);

    mFoncEqCoP = cElCompiledFonc::AllocFromName(mNameType);

    if (Code2Gen)
    {
        GenCode();
	return;
    }
    if (mFoncEqCoP==0)
    {
	mFoncEqCoP = cElCompiledFonc::DynamicAlloc(mLInterv,EqCoPlan());
	// ELISE_ASSERT(false,"Dont Get CompVal for  cCpleCamFormelle");
    }


    mFoncEqCoP->SetMappingCur(mLInterv,&mSet);

    pAdrPdsA1 =  mFoncEqCoP->RequireAdrVarLocFromString(NamePdsA1);
    pAdrPdsA2 =  mFoncEqCoP->RequireAdrVarLocFromString(NamePdsA2);
    pAdrPdsA3 =  mFoncEqCoP->RequireAdrVarLocFromString(NamePdsA3);

    pAdrPdsB1 =  mFoncEqCoP->RequireAdrVarLocFromString(NamePdsB1);
    pAdrPdsB2 =  mFoncEqCoP->RequireAdrVarLocFromString(NamePdsB2);
    pAdrPdsB3 =  mFoncEqCoP->RequireAdrVarLocFromString(NamePdsB3);

    mSet.AddFonct(mFoncEqCoP);
}

/************************************************************/
/*                                                          */
/*                 cAppuiGridEq                             */
/*                                                          */
/************************************************************/



const std::string cAppuiGridEq::NamePds1 = "Pds1";
const std::string cAppuiGridEq::NamePds2 = "Pds2";
const std::string cAppuiGridEq::NamePds3 = "Pds3";

const std::string cAppuiGridEq::NameTerX = "XTer";
const std::string cAppuiGridEq::NameTerY = "YTer";
const std::string cAppuiGridEq::NameTerZ = "ZTer";


Pt2d<Fonc_Num> cAppuiGridEq::ResiduForm()
{
  ELISE_ASSERT(false,"cAppuiGridEq::ResiduForm pas encore retabli");
   return Pt2d<Fonc_Num>(0,0);

/*
   Pt3d<Fonc_Num> aPTerrain(cVarSpec(0,NameTerX),cVarSpec(0,NameTerY),cVarSpec(0,NameTerZ));
   Pt3d<Fonc_Num> aRay = aPTerrain -  mRot.COpt();
   aRay = mRot.VectM2C(aRay);
   aRay = aRay / aRay.z;

          // - * - * - * - * - * - * - * - * - * - *

   Fonc_Num PA1 = cVarSpec(0,NamePds1); 
   Fonc_Num PA2 = cVarSpec(0,NamePds2); 
   Fonc_Num PA3 = cVarSpec(0,NamePds3); 
   const cTFI_Triangle & aTriangle = mTri.GetTriFromP(mTri.APointInTri());
   Pt2d<Fonc_Num> aP = aTriangle.PInc1()*PA1 + aTriangle.PInc2()*PA2 + aTriangle.PInc3()*PA3;

   return  Pt2d<Fonc_Num>(aP.x-aRay.x,aP.y-aRay.y);
  */
}


Pt2dr   cAppuiGridEq::ResiduAppui(Pt3dr aPTer,Pt2dr aPIm)
{
   return AddAppui(aPTer,aPIm,-1);
}

Pt2dr   cAppuiGridEq::AddAppui(Pt3dr aPTer,Pt2dr aPIm,REAL aPds)
{

     const cTFI_Triangle & aTriangle = mTri.GetTriFromP(aPIm);
     Pt3dr aCBary = aTriangle.TriGeom().CoordBarry(aPIm);


     mLInterv.ResetInterv(aTriangle.IntervA1());
     mLInterv.ResetInterv(aTriangle.IntervA2());
     mLInterv.ResetInterv(aTriangle.IntervA3());

     *pAdrPds1 = aCBary.x;
     *pAdrPds2 = aCBary.y;
     *pAdrPds3 = aCBary.z;

     *pAdrXTer = aPTer.x;
     *pAdrYTer = aPTer.y;
     *pAdrZTer = aPTer.z;

     mFoncEq->SetMappingCur(mLInterv,&mSet);


     if (aPds <= 0)
     {


        const std::vector<REAL> &  aV = mSet.VResiduSigne(mFoncEq);
        return Pt2dr(aV[0],aV[1]);
     }
     
     const std::vector<REAL> &  aV = mSet.VAddEqFonctToSys(mFoncEq,aPds,false,NullPCVU);
     return Pt2dr(aV[0],aV[1]);
}


void  cAppuiGridEq::GenCode()
{
    Pt2d<Fonc_Num> aP = ResiduForm();
    std::vector<Fonc_Num> aV;
    aV.push_back(aP.x);
    aV.push_back(aP.y);
    cElCompileFN::DoEverything(DIRECTORY_GENCODE_FORMEL,mNameType,aV,mLInterv);
}


cAppuiGridEq::cAppuiGridEq
(
     cTriangulFormelle & aTri, 
     cRotationFormelle & aRot,
     bool                Code2Gen

)  :
   mSet        (aTri.Set()),
   mTri        (aTri),
   mRot        (aRot),
   mNameType   ("cEqAppuiGrid")
{
     mRot.IncInterv().SetName("Orient");
     mLInterv.AddInterv(mRot.IncInterv());

     const cTFI_Triangle & aTriangle = mTri.GetTriFromP(mTri.APointInTri());

     mLInterv.AddInterv(aTriangle.IntervA1(),true);
     mLInterv.AddInterv(aTriangle.IntervA2(),true);
     mLInterv.AddInterv(aTriangle.IntervA3(),true);


    mFoncEq = cElCompiledFonc::AllocFromName(mNameType);

    if (Code2Gen)
    {
        GenCode();
	return;
    }
    if (mFoncEq==0)
    {
         cout <<"Name Type = [" << mNameType << "]\n";
	 ELISE_ASSERT(false,"cAppuiGridEq::cAppuiGridEq\n");
    }


    mFoncEq->SetMappingCur(mLInterv,&mSet);

    pAdrPds1 =  mFoncEq->RequireAdrVarLocFromString(NamePds1);
    pAdrPds2 =  mFoncEq->RequireAdrVarLocFromString(NamePds2);
    pAdrPds3 =  mFoncEq->RequireAdrVarLocFromString(NamePds3);

    pAdrXTer =  mFoncEq->RequireAdrVarLocFromString(NameTerX);
    pAdrYTer =  mFoncEq->RequireAdrVarLocFromString(NameTerY);
    pAdrZTer =  mFoncEq->RequireAdrVarLocFromString(NameTerZ);

    mSet.AddFonct(mFoncEq);
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
