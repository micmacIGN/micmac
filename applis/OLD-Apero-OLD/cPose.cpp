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
#include "general/all.h"

#include "Apero.h"

namespace NS_ParamApero
{
int PROF_UNDEF() { return -1; }


int cPoseCam::theCpt = 0;

int  TheDefProf2Init = 1000000;

void cPoseCam::SetNameCalib(const std::string & aNameC)
{
   mNameCalib = aNameC;
}

static int theNumCreate =0;


void cPoseCam::C2MCompenseMesureOrInt(Pt2dr & aPC)
{
   aPC = mOrIntC2M(aPC);
}



void cPoseCam::SetOrInt(const cTplValGesInit<cSetOrientationInterne> & aTplSI)
{
  if (! aTplSI.IsInit()) return;

  const cSetOrientationInterne & aSOI = aTplSI.Val();

   cSetName *  aSelector = mAppli.ICNM()->KeyOrPatSelector(aSOI.PatternSel());

   if (! aSelector->IsSetIn(mName))
      return;

  std::string aNameFile =  mAppli.DC() + mAppli.ICNM()->Assoc1To1(aSOI.KeyFile(),mName,true);

   cAffinitePlane aXmlAff = StdGetObjFromFile<cAffinitePlane>
                            (
                                 aNameFile,
                                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 aSOI.Tag().Val(),
                                 "AffinitePlane"
                            );

    ElAffin2D  anAffM2C = Xml2EL(aXmlAff);

   if (! aSOI.M2C())
      anAffM2C = anAffM2C.inv();

   if (aSOI.AddToCur())
      mOrIntM2C= anAffM2C * mOrIntM2C ; 
   else
      mOrIntM2C= anAffM2C;

   //  Si on le fait avec les marques fiduciaires ca ecrase le reste

   mOrIntC2M = mOrIntM2C.inv();
}


cPoseCam::cPoseCam
(
     cAppliApero & anAppli,
     const cPoseCameraInc & aPCI,
     const std::string & aNamePose,
     const std::string & aNameCalib,
     cPoseCam *             aPRat,
     cCompileAOI  *         aCompAOI
)   :
    mAppli   (anAppli),
    mNameCalib (aNameCalib),
    mName    (aNamePose),
    mCpt     (-1),
    mProf2Init (TheDefProf2Init),
    mPdsTmpMST (0.0),
    mPCI     (&aPCI),
    mCalib   (mAppli.CalibFromName(aNameCalib,this)),
    mPoseRat (aPRat),
    mPoseInitMST1 (0),
    mPoseInitMST2 (0),
    mCamRF   (mPoseRat ? mPoseRat->mCF : 0),

    mCF      (mCalib->PIF().NewCam(cNameSpaceEqF::eRotLibre,ElRotation3D::Id,mCamRF,aNamePose,true)),

    mRF      (mCF->RF()),
    mAltiSol     (ALTISOL_UNDEF()),
    mProfondeur  (PROF_UNDEF()),
    mTime        (TIME_UNDEF()),
    mPrioSetAlPr (-1),
    mRotIsInit   (false),
    mLastCP      (0),
    mSom         (0),
    mCompAOI     (aCompAOI),
    mFirstBoxImSet (false),
    mImageLoaded (false),
    mBoxIm       (Pt2di(0,0),Pt2di(0,0)),
    mIm          (1,1),
    mTIm         (mIm),
    mMasqH       (mAppli.MasqHom(aNamePose)),
    mTMasqH      (mMasqH ? new  TIm2DBits<1>(*mMasqH) : 0),
    mPreInit     (false),
    mObsCentre   (0,0,0),
    mHasObsCentre (false),
    mNumTmp       (-12345678),
    mNbPtsMulNN   (-1),
    mNumBande     (0),
    mPrec         (this),
    mNext         (this),
    mNumCreate    (theNumCreate++),
    mCurLayer     (0), 
    mOrIntM2C     (ElAffin2D::Id()),
    mOrIntC2M     (ElAffin2D::Id()),
    mNbPosOfInit  (-1),
    mFidExist     (false)
{
    
   SetOrInt(mAppli.Param().GlobOrInterne());
   SetOrInt(aPCI.OrInterne());

   std::pair<std::string,std::string> aPair = mAppli.ICNM()->Assoc2To1("Key-Assoc-STD-Orientation-Interne",mName,true);
   std::string aNamePtsCam = aPair.first;
   std::string aNamePtsIm  = aPair.second;

   if (ELISE_fp::exist_file(mAppli.DC()+ aNamePtsIm))
   {
       cMesureAppuiFlottant1Im aMesCam = mAppli.StdGetOneMAF(aNamePtsCam);
       cMesureAppuiFlottant1Im aMesIm  = mAppli.StdGetOneMAF(aNamePtsIm);

       ElPackHomologue  aPack = PackFromCplAPF(aMesIm,aMesCam);
       ElAffin2D anAf = ElAffin2D::L2Fit(aPack);

       mOrIntM2C = anAf.inv();
       mFidExist = true;
   }

   mOrIntC2M = mOrIntM2C.inv();
   InitAvantCompens();
}

int cPoseCam::NumCreate() const
{
   return mNumCreate;
}

int & cPoseCam::NumTmp()
{
   return mNumTmp;
}


bool cPoseCam::AcceptPoint(const Pt2dr & aP) const
{
    // std::cout << "SccaaaNN " << mCalib-> CamInit().IsScanned() << mCalib->SzIm() << "\n";

    if (mCalib-> CamInit().IsScanned())
    {
        Pt2dr aSz = Pt2dr(mCalib->SzIm());
        if ((aP.x<=0) || (aP.y <=0) || (aP.x>=aSz.x) || (aP.y>=aSz.y))
           return false;
    }

    return true;
}

bool cPoseCam::FidExist() const
{
   return mFidExist;
}


const ElAffin2D &  cPoseCam::OrIntM2C() const
{
   return mOrIntM2C;
}
const ElAffin2D &  cPoseCam::OrIntC2M() const
{
   return mOrIntC2M;
}


bool cPoseCam::IsInZoneU(const Pt2dr & aP) const
{
   return mCalib->IsInZoneU(aP);
}

void cPoseCam::SetLink(cPoseCam * aPrec,bool OK)
{
   mNumBande =  aPrec->mNumBande;
   if (OK) 
   {
      mPrec     = aPrec;
      aPrec->mNext = this;
   }
   else
   {
      mNumBande++;
   }
/*
    mNumBande =  aNumBande;
    mPrec     = aPrec;
    if (aPrec) aPrec->mNext = this;
*/
}


double  GuimbalAnalyse(const ElRotation3D & aR,bool show)
{
    // aR = aR.inv();
// aR  = ElRotation3D(Pt3dr(0,0,0),0,1.57,0);
    double aTeta01 = aR.teta01();
    double aTeta02 = aR.teta02();
    double aTeta12 = aR.teta12();

    double aEps = 1e-2;
    ElMatrix<double> aM1 = (ElRotation3D(aR.tr(),aTeta01+aEps,aTeta02,aTeta12).Mat()-aR.Mat())*(1/aEps);
    ElMatrix<double> aM2 = (ElRotation3D(aR.tr(),aTeta01,aTeta02+aEps,aTeta12).Mat()-aR.Mat())*(1/aEps);
    ElMatrix<double> aM3 = (ElRotation3D(aR.tr(),aTeta01,aTeta02,aTeta12+aEps).Mat()-aR.Mat())*(1/aEps);

    aM1 = aM1 * (1/sqrt(aM1.L2()));
    aM2 = aM2 * (1/sqrt(aM2.L2()));
    aM3 = aM3 * (1/sqrt(aM3.L2()));

    ElMatrix<double> aU1 = aM1;
    ElMatrix<double> aU2 = aM2 -aU1*aU1.scal(aM2);
    double aD2 = sqrt(aU2.L2());
    aU2 = aU2 *(1/aD2);


    ElMatrix<double> aU3 = aM3 -aU1*aU1.scal(aM3) -aU2*aU2.scal(aM3);

    double aD3 = sqrt(aU3.L2());

    if (show)
    {
        std::cout << aU2.scal(aU1) << " " << aU3.scal(aU1) << " " << aD2*aD3 << "\n";
    

        ShowMatr("M1",aM1);
        ShowMatr("M2",aM2);
        ShowMatr("M3",aM3);
        getchar();
    }
    return aD2 * aD3;
}

void cPoseCam::Trace() const
{
    if (! mAppli.TracePose(*this))
       return;

    std::cout   << mName ;

    if (RotIsInit())
    {
        ElRotation3D  aR = CurRot() ;
        std::cout <<  " C=" << aR.tr() ;
        std::cout <<  " Teta=" << aR.teta01() << " " << aR.teta02() <<  " " << aR.teta12()  ;
       
        if (mAppli.Param().TraceGimbalLock().Val())
        {
            std::cout << " GL-Score=" <<  GuimbalAnalyse(aR,false);
        }
    }
    std::cout << "\n";
}


double & cPoseCam::PdsTmpMST()
{
   return mPdsTmpMST;
}

void cPoseCam::Set0Prof2Init()
{
    mProf2Init = 0;
}


double cPoseCam::Time() const
{
  return mTime;
}

int cPoseCam::Prof2Init() const
{
    return mProf2Init;
}

void cPoseCam::UpdateHeriteProf2Init(const  cPoseCam & aC2)
{
    if (mProf2Init==TheDefProf2Init)
         mProf2Init = 0;

    ElSetMax(mProf2Init,1+aC2.mProf2Init);
}

void cPoseCam::SetSom(tGrApero::TSom & aSom)
{
   mSom = & aSom;
}

tGrApero::TSom * cPoseCam::Som()
{
   return mSom;
}

bool cPoseCam::PreInit() const
{
   return mPreInit;
}

cPoseCam * cPoseCam::PoseInitMST1()
{
   return mPoseInitMST1;
}
void cPoseCam::SetPoseInitMST1(cPoseCam * aPoseInitMST1)
{
   mPoseInitMST1 = aPoseInitMST1;
}


cPoseCam * cPoseCam::PoseInitMST2()
{
   return mPoseInitMST2;
}
void cPoseCam::SetPoseInitMST2(cPoseCam * aPoseInitMST2)
{
   mPoseInitMST2 = aPoseInitMST2;
}



std::string  cPoseCam::CalNameFromL(const cLiaisonsInit & aLI)
{
     std::string aName2 = aLI.NameCam();
     if (aLI.NameCamIsKeyCalc().Val())
     {
         aName2 =  mAppli.ICNM()->Assoc1To1(aName2,mName,aLI.KeyCalcIsIDir().Val());
     }

     return aName2;
}


cRotationFormelle & cPoseCam::RF()
{
   return mRF;
}

ElRotation3D cPoseCam::CurRot() const
{
   return mRF.CurRot();
}

int   cPoseCam::NbPtsMulNN() const 
{
   return mNbPtsMulNN;
}

void  cPoseCam::SetNbPtsMulNN(int aNbNN) 
{
  mNbPtsMulNN = aNbNN;
}


const std::string &  cPoseCam::NameCalib() const
{
   return mCalib->CCI().Name();
}

cAnalyseZoneLiaison  &  cPoseCam::AZL() {return mAZL;}
double               &  cPoseCam::QualAZL() {return mQualAZL;}
int                  &  cPoseCam::NbPLiaisCur() {return mNbPLiaisCur;}



void cPoseCam::SetRattach(const std::string & aNameRat)
{
   if ((mPoseRat==0) || (mPoseRat != mAppli.PoseFromName(aNameRat)))
   {
       std::cout << mPoseRat << "\n";
       std::cout << "Pour rattacher " << mName << " a " << aNameRat << "\n";
       std::cout << "(Momentanement) : le ratachement doit etre prevu a l'initialisation";
       ELISE_ASSERT(false,"Rattachement impossible");
   }
}

void    cPoseCam::InitAvantCompens()
{
    mPMoy = Pt3dr(0,0,0);
    mMoyInvProf =0;
    mSomPM = 0;
}

const CamStenope *  cPoseCam::CurCam() const
{
   return  mCF->CameraCourante() ;
}

void    cPoseCam::AddPMoy(const Pt3dr & aP,double aBSurH)
{
   double aPds = (aBSurH-mAppli.Param().LimInfBSurHPMoy().Val());
   aPds /= (mAppli.Param().LimSupBSurHPMoy().Val() - mAppli.Param().LimInfBSurHPMoy().Val());
   if (aPds<0) return;

    const CamStenope * aCS = mCF->CameraCourante() ;
    if (mTMasqH)
    {
      Pt2di aPIm =  round_ni(aCS->R3toF2(aP));
      
      if (! mTMasqH->get(aPIm,0))
         return;
    }
    double aProf = aCS->ProfondeurDeChamps(aP);

/*
if (aProf<0)
{
    std::cout << " PROF " << aProf << " " << aBSurH << "\n";
}
*/
    
    mPMoy = mPMoy + aP * aPds;
    mMoyInvProf  += (1/aProf) * aPds;
    mSomPM  += aPds ;
 
}

bool     cPoseCam::PMoyIsInit() const
{
   return mSomPM != 0;
}

Pt3dr   cPoseCam::GetPMoy() const
{
   ELISE_ASSERT(PMoyIsInit(),"cPoseCam::GetPMoy");
   return mPMoy / mSomPM;
}


double   cPoseCam::ProfMoyHarmonik() const
{
   ELISE_ASSERT(PMoyIsInit(),"cPoseCam::ProfMoyHarmonik");
   return 1.0 / (mMoyInvProf/mSomPM);
}



void cPoseCam::ActiveContrainte(bool Stricte)
{
    mAppli.SetEq().AddContrainte(mRF.StdContraintes(),Stricte);
}

void ShowResMepRelCoplan(cResMepRelCoplan aRMRC)
{
    std::vector<cElemMepRelCoplan> aV =  aRMRC.VElOk();
    std::cout << "  NB SOL = " << aV.size() << "\n";
    for (int aKS=0; aKS<int(aV.size()) ; aKS++)
    {
        cElemMepRelCoplan aS = aV[aKS];
        std::cout << "ANGLE " << aKS << " " << aS.AngTot() << "\n";
        ElRotation3D aR = aS.Rot();
        std::cout << aR.ImRecAff(Pt3dr(0,0,0)) << " "
                  << aR.teta01() << " "
                  << aR.teta02() << " "
                  << aR.teta12() << " "
                  << "\n";
    }
}


/*
void ShowPrecisionLiaison(ElPackHomologue  aPack,ElRotation3D aR1,ElRotation3D aR2)
{
   CamStenopeIdeale aC1(1.0,Pt2dr(0.0,0.0));
   aC1.SetOrientation(aR1.inv());
   CamStenopeIdeale aC2(1.0,Pt2dr(0.0,0.0));
   aC2.SetOrientation(aR2.inv());

   double aSD=0;
   double aSP=0;
   for (ElPackHomologue::iterator itP= aPack.begin(); itP!=aPack.end() ; itP++)
   {
        ElSeg3D aS1 = aC1.F2toRayonR3(itP->P1());
        ElSeg3D aS2 = aC2.F2toRayonR3(itP->P2());

	Pt3dr aPTer = aS1.PseudoInter(aS2);

	Pt2dr aQ1 = aC1.R3toF2(aPTer);
	Pt2dr aQ2 = aC2.R3toF2(aPTer);

	std::cout << aQ1 << aQ2 << euclid(aQ1,itP->P1()) << " " <<  euclid(aQ2,itP->P2())  << "\n";
	aSP += 2;
	aSD += euclid(aQ1,itP->P1()) + euclid(aQ2,itP->P2());

   }

   std::cout << "DMOYENNE " << aSD/aSP * 1e4 << "/10000\n";
}
*/
 

ElMatrix<double> RFrom3P(const Pt3dr & aP1, const Pt3dr & aP2, const Pt3dr & aP3)
{
    Pt3dr aU = aP2-aP1;
    aU = aU / euclid(aU);
    Pt3dr aW = aU ^(aP3-aP1);
    aW = aW / euclid(aW);
    Pt3dr aV =  aW ^aU;
    return ElMatrix<double>::Rotation(aU,aV,aW);
}

void  TTTT (ElPackHomologue & aPack)
{
//
   int aCpt=0;
   Pt2dr aS1(0,0);
   Pt2dr aS2(0,0);
   double aSP=0;
   std::cout << " ccccccccccc " << aPack.size() << "\n";
   for 
   (
       ElPackHomologue::iterator itP=aPack.begin() ;
       itP!=aPack.end();
       itP++
   )
   {
       aS1 = aS1 + itP->P1() * itP->Pds();
       aS2 = aS2 + itP->P2() * itP->Pds();
       aSP += itP->Pds();
       aCpt++;
       if (euclid(itP->P1()) > 1e5)
          std::cout << itP->P1() << itP->P2() << itP->Pds()  << "\n";
   }
   std::cout << "SOMES ::  "<< (aS1/aSP) << " " << (aS2/aSP) << "\n";
}

void  TTTT (cElImPackHom & anIP)
{
    ElPackHomologue aPack = anIP.ToPackH(0);
    TTTT(aPack);
}

void  cPoseCam::TenteInitAltiProf
      (
           int    aPrio,
           double anAlti,
	   double aProf
      )
{
   if ((aProf != PROF_UNDEF()) && (aPrio>mPrioSetAlPr))
   {
         mProfondeur = aProf;
         mAltiSol = anAlti;
	 mPrioSetAlPr = aPrio;
   }
}


cPoseCam * cPoseCam::Alloc
           (
               cAppliApero & anAppli,
               const cPoseCameraInc & aPCI,
               const std::string & aNamePose,
               const std::string & aNameCalib,
               cCompileAOI * aCompAOI
           )
{

    cPoseCam * aPRat=0;

    if (aPCI.PosesDeRattachement().IsInit())
    {
        aPRat = anAppli.PoseFromNameGen
                (
                    aPCI.PosesDeRattachement().Val(),
                    aPCI.NoErroOnRat().Val()
                );

    }

    cPoseCam * aRes = new cPoseCam(anAppli,aPCI,aNamePose,aNameCalib,aPRat,aCompAOI);
    return aRes;
}

int   cPoseCam::NbPosOfInit(int aDef)
{
   return (mNbPosOfInit>=0) ? mNbPosOfInit : aDef;
}

void  cPoseCam::SetNbPosOfInit(int aNbPosOfInit)
{
   mNbPosOfInit = aNbPosOfInit;
}


void cPoseCam::DoInitIfNow()
{

    mPreInit = true;


    mAppli.AddRotPreInit();
    if (mPCI->InitNow().Val())
    {
       InitRot();
    }
}

/*
void  cPoseCam::AddLink(cPoseCam * aPC)
{
   if (! BoolFind(mVPLinked,aPC))
      mVPLinked.push_back(aPC);
}
*/



void cPoseCam::SetCurRot(const ElRotation3D & aRot)
{
    mCF->SetCurRot(aRot);
}


void  cPoseCam::SetBascRig(const cSolBasculeRig & aSBR)
{

    Pt3dr aP;
    if (mSomPM)
    {
       aP = mPMoy / mSomPM;
       mSomPM = 0;
    }
    else
    {
        const CamStenope *  aCS = CurCam() ;

        aP =  aCS->ImEtProf2Terrain(aCS->Sz()/2.0,mProfondeur);
        aP =  aSBR(aP);
    }

    SetCurRot(aSBR.TransformOriC2M(CurRot()));


    const CamStenope *  aCS = CurCam() ;
    mAltiSol = aP.z;
    mProfondeur = aCS->ProfondeurDeChamps(aP);
}



void cPoseCam::BeforeCompens()
{
   mRF.ReactuFcteurRapCoU();
}

void cPoseCam::InitCpt()
{
    if (mCpt<0)
    {
       mCpt  = theCpt++;
       mAppli.AddPoseInit(mCpt,this);
    }
}

bool cPoseCam::HasObsCentre() const
{
   return mHasObsCentre;
}

const Pt3dr  & cPoseCam::ObsCentre() const
{
   if (!mHasObsCentre)
   {
       std::cout << "Name Pose = " << mName << "\n";
       ELISE_ASSERT
       (
             mHasObsCentre,
             "Observation on centre (GPS) has no been associated to camera"
       );
   }
 
   return mObsCentre;
}


bool cPoseCam::IsId(const ElAffin2D & anAff) const
{
    Pt2dr aSz =  Pt2dr(mCalib->SzIm());
    Box2dr aBox(Pt2dr(0,0),aSz);
    Pt2dr aCoins[4];
    aBox.Corners(aCoins);
    double aDiag = euclid(aSz);

    double aDMax = 0.0;

    for (int aK=0 ; aK<4 ; aK++)
    {
         Pt2dr aP1 = aCoins[aK];
         Pt2dr aP2 = anAff(aP1);
         double aD = euclid(aP1,aP2) / aDiag;
         ElSetMax(aDMax,aD);
    }

    return aDMax < 1e-3;
}
/*
*/


void   cPoseCam::InitRot()
{
   const cLiaisonsInit * theLiasInit = 0;
   mNumInit =  mAppli.NbRotInit();
   std::cout << "NUM " << mNumInit << " FOR " << mName<< "\n";
/*
{

if (mNumInit==90)
{
   BugFE=true;
}
else
{
   std::cout << "END BUG FE \n";
   BugFE=false;
}
 BugFE=true;
}
*/


   if (mPCI->IdBDCentre().IsInit())
   {

//  std::cout << "CCCcccC : " << mName <<  " " << mAppli.HasObsCentre(mPCI->IdBDCentre().Val(),mName) << "\n";
      if (mAppli.HasObsCentre(mPCI->IdBDCentre().Val(),mName))
      {
          mObsCentre = mAppli.ObsCentre(mPCI->IdBDCentre().Val(),mName).mVals;
          mHasObsCentre = true;
      }
   }
  
    // std::cout << mName << "::Prof=" << mProf2Init << "\n";
// std::cout <<  "Init Pose " << aNamePose << "\n";

   InitCpt();


    double aProfI = mPCI->ProfSceneImage().ValWithDef(mAppli.Param().ProfSceneChantier().Val());
    ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
    std::string aNZPl ="";
    double aDZPl = -1;
    double aDZPl2 = -1;
    cPoseCam * aCam2PL = 0;
    double aLambdaRot=1;
    CamStenope & aCS1 =   * (mCalib->PIF().CurPIF());

    double  aProfPose = -1;
    double  anAltiSol = ALTISOL_UNDEF();


   bool isMST = mPCI->MEP_SPEC_MST().IsInit();
   if (isMST)
   {
       ELISE_ASSERT
       (
           mPCI->PoseFromLiaisons().IsInit(),
           "MST requires PoseFromLiaisons"
       );
   }

    if (mPCI->PosId().IsInit())
    {
         aRot =  ElRotation3D(Pt3dr(0,0,0),0,0,-PI);
    }
    else if(mPCI->PosFromBDOrient().IsInit())
    {
	const std::string & anId = mPCI->PosFromBDOrient().Val();
        aRot =mAppli.Orient(anId,mName);
	cObserv1Im<cTypeEnglob_Orient>  &  anObs = mAppli.ObsOrient(anId,mName);

         bool Ok1 = IsId(anObs.mOrIntC2M);
         bool Ok2 = IsId(anObs.mOrIntC2M * mOrIntC2M.inv());
         ELISE_ASSERT
         (
             Ok1 || Ok2,
             "Specicied Internal Orientation is incompatible with Fiducial marks"
         );
         

	aProfPose = anObs.mProfondeur;
	anAltiSol =  anObs.mAltiSol;
        mTime     =  anObs.mTime;
    }
    else if (mPCI->PoseInitFromReperePlan().IsInit())
    {
       cPoseInitFromReperePlan aPP = mPCI->PoseInitFromReperePlan().Val();
       ElPackHomologue aPack;
       std::string  aNamePose2 = aPP.NameCam();
       std::string  aNameCal2 = mAppli.NameCalOfPose(aNamePose2);

       std::string aTestNameFile = mAppli.DC()+aPP.IdBD();
       if (ELISE_fp::exist_file(aTestNameFile))
       {
            ELISE_ASSERT(false,"Obsolet Init Form repere plan");
            // Onsolete, pas cohrent avec orient interen
            // aPack = ElPackHomologue::- FromFile(aTestNameFile);
       }
       else 
           mAppli.InitPack(aPP.IdBD(),aPack,mName,aPP.NameCam());
       CamStenope & aCS2 = mAppli.CalibFromName(aNameCal2,0)->CamInit();

// TTTT(aPack);



       aPack = aCS1.F2toPtDirRayonL3(aPack,&aCS2);
       cResMepRelCoplan aRMRC = aPack.MepRelCoplan(1.0,aPP.L2EstimPlan().Val());
       cElemMepRelCoplan & aSP = aRMRC.BestSol();

       // aM1 aM2 aM3 -> coordonnees monde, specifiees par l'utilisateur
       // aC1 aC2 aC3 -> coordonnees monde

       Pt3dr aM1,aM2,aM3;
       Pt2dr aIm1,aIm2,aIm3;
       Pt3dr aDirPl;
       bool aModeDir = false;
       if (aPP.MesurePIFRP().IsInit())
       {
           aM1 = aPP.Ap1().Ter();
           aM2 = aPP.Ap2().Ter();
           aM3 = aPP.Ap3().Ter();
           aIm1 = aCS1.F2toPtDirRayonL3(aPP.Ap1().Im());
           aIm2 = aCS1.F2toPtDirRayonL3(aPP.Ap2().Im());
           aIm3 = aCS1.F2toPtDirRayonL3(aPP.Ap3().Im());
       }
       else if (aPP.DirPlan().IsInit())
       {
           aModeDir = true;
           aDirPl = aPP.DirPlan().Val();
           aM1 = Pt3dr(0,0,0);

           Pt3dr aW = vunit(aPP.DirPlan().Val());
           aM2 = OneDirOrtho(aW);
           aM3 = aW ^ aM2;
           
           // aIm1 = aCS1.Sz() /2.0;
           // aIm2 = aIm1 + Pt2dr(1.0,0.0);
           // aIm3 = aIm1 + Pt2dr(0.0,1.0);
           aIm1 = Pt2dr(0,0);
           aIm2 = aIm1 + Pt2dr(0.1,0.0);
           aIm3 = aIm1 + Pt2dr(0.0,0.1);
       }



       Pt3dr aC1 = aSP.ImCam1(aIm1);
       Pt3dr aC2 = aSP.ImCam1(aIm2);
       Pt3dr aC3 = aSP.ImCam1(aIm3);

       double aFMult=0;


       if (aPP.DEuclidPlan().IsInit())
           aFMult = aPP.DEuclidPlan().Val() / aSP.DistanceEuclid();
       else
           aFMult = euclid(aM2-aM1)/euclid(aC2-aC1);
        aDZPl = aSP.DPlan() * aFMult;
        aNZPl = aPP.OnZonePlane();

        aC1 = aC1 * aFMult;
        aC2 = aC2 * aFMult;
        aC3 = aC3 * aFMult;

	ElMatrix<double> aMatrM = RFrom3P(aM1,aM2,aM3);
	ElMatrix<double> aMatrC = RFrom3P(aC1,aC2,aC3);
        ElMatrix<double>  aMatrC2M = aMatrM * gaussj(aMatrC);

	Pt3dr aTr = aM1 - aMatrC2M * aC1;

        if (aModeDir)
        {
            Pt3dr aC1 = aMatrC2M*Pt3dr(1,0,0);
            Pt3dr aC2 = aMatrC2M*Pt3dr(0,1,0);
            Pt3dr aC3 = aMatrC2M*Pt3dr(0,0,1);
            double aScal = scal(aC3,aDirPl);

            // std::cout << "CSAL = " << aScal << "\n";
            // std::cout << aMatrC2M*Pt3dr(1,0,0) << aMatrC2M*Pt3dr(0,1,0) <<  aMatrC2M*Pt3dr(0,0,1) << "\n";

            if (aScal <0)
            {
                 aMatrC2M = ElMatrix<double>::Rotation(aC1,aC2*-1,aC3*-1);
            }
            // std::cout << aMatrC2M*Pt3dr(1,0,0) << aMatrC2M*Pt3dr(0,1,0) <<  aMatrC2M*Pt3dr(0,0,1) << "\n";
        }

	aRot = ElRotation3D(aTr,aMatrC2M);

        anAltiSol = aM1.z;
        aProfPose = euclid(aC1);
	mAppli.AddPlan(aNZPl,aM1,aM2,aM3,false);
	//Pt3dr  uM = 
    }
    else if(mPCI->PosFromBDAppuis().IsInit())
    {
         std::cout << "Do Init-by-appuis for " << mName << "\n";
         const cPosFromBDAppuis & aPFA = mPCI->PosFromBDAppuis().Val();
	 const std::string & anId = aPFA.Id();


         std::list<Appar23> aL = mAppli.AppuisPghrm(anId,mName,mCalib);
         tParamAFocal aNoPAF;
         CamStenopeIdeale aCamId(true,1.0,Pt2dr(0.0,0.0),aNoPAF);
	 double aDMin;

         Pt3dr aDirApprox;
         Pt3dr * aPtrDirApprox=0;
         if (aPFA.DirApprox().IsInit())
         {
               aDirApprox = aPFA.DirApprox().Val();
               aPtrDirApprox = &aDirApprox;
         }

	 // aRot = aCamId.CombinatoireOFPA(anAppli.Param().NbMaxAppuisInit().Val(),aL,&aDMin);
	 aRot = aCamId.RansacOFPA(true,aPFA.NbTestRansac(),aL,&aDMin,aPtrDirApprox);

         // std::cout << "DIST-MIN  = " << aDMin << "\n";
/*
*/
	 aRot = aRot.inv();
	 // cObserv1Im<cTypeEnglob_Appuis>  &  anObs = mAppli.ObsAppuis(anId,mName);
	 // Pt3dr aCdg =  anObs.mBarryTer;
         Pt3dr aCdg = BarryImTer(aL).pter;
	 anAltiSol = aCdg.z;
	 Pt3dr aDirVisee = aRot.ImVect(Pt3dr(0,0,1));

	 aProfPose = scal(aDirVisee,aCdg-aRot.ImAff(Pt3dr(0,0,0)));
    }
    else if (mPCI->PoseFromLiaisons().IsInit())
    {
         cResolvAmbiBase * aRAB=0;
         const std::vector<cLiaisonsInit> & aVL = mPCI->PoseFromLiaisons().Val().LiaisonsInit();
	 // bool   aNormIsFixed = false;
	 int aNbL = aVL.size();

         
	 ElPackHomologue  aPack0;
	 ElRotation3D     anOrAbs0 (Pt3dr(0,0,0),0,0,0);

         Pt3dr aP0Pl,aP1Pl,aP2Pl;
	 double aMultPl=1.0;
         
         
	 std::vector<cPoseCam *> aVPC;
	 //cPoseCam * aCam00 = 0;
         if (isMST)
         {
             if (PoseInitMST1()==0)
             {
                std::cout << "For : " << mName << "\n";
                ELISE_ASSERT(false,"MST1 Incoh");
             }
             aNbL = PoseInitMST2() ? 2 : 1;
// std::cout << "isMST " << aNbL << " " << mName << "\n";
             if (aVL[0].OnZonePlane().IsInit())
                aNbL=1;
         }

         bool aRPure = false;
         const cLiaisonsInit * pLI0 = 0;
	 for (int aK=0 ; aK<aNbL ; aK++)
	 {
            const cLiaisonsInit & aLI = aVL[isMST?0:aK];
            if (aK==0) 
            {
               pLI0 = &aLI;
               theLiasInit = pLI0;
            }

	    std::string  aName2;
	    cPoseCam * aCam2 = 0;
            if (isMST)
            {
                aCam2 = (aK==0) ? PoseInitMST1() : PoseInitMST2();
                aName2 = aCam2->Name();

                ELISE_ASSERT
                (
                    aLI.IdBD()==mAppli.SymbPack0(),
                    "MST must be used with first Pack "
                );
            }
            else
            {
	       aName2 = CalNameFromL(aLI);
	       aCam2 = mAppli.PoseFromName(aName2);
            }
            // ElSetMax(mProf2Init,1+aCam2->mProf2Init);


            if (! aCam2->RotIsInit())
            {
               std::cout << "For " << mName << "/" << aName2;
               ELISE_ASSERT(false,"Incohernce : Init based on Cam not init");
            }
	    aVPC.push_back(aCam2);
	    CamStenope &  aCS2 = aCam2->Calib()->CamInit();

	    bool aBaseFixee = false;
	    double aLBase=1.0;
	    if (aLI.LongueurBase().IsInit() && ((!isMST) || (aNbL==1)))
	    {
	      // Ce serait incoherent puisque les liaisons multiples servent a
	      // fixer la longueur de la base
	        ELISE_ASSERT(aNbL==1,"Ne peut fixe la longueur de base avec plusieurs liaisons");
		aBaseFixee = true;
		aLBase  = aLI.LongueurBase().Val();
	    }


	    // if (aK==0) aCam0 = aCam;

            // ElPackHomologue aPack = anAppli.PackPhgrmFromCple(&aCS1,aNamePose,&aCS2,aName2);
	    ElPackHomologue aPack;
	    mAppli.InitPackPhgrm(aLI.IdBD(),aPack,mName,&aCS1,aName2,&aCS2);
	    double aProfC = aLI.ProfSceneCouple().ValWithDef(aProfI);

            if (aK==0)
	    {
               ElRotation3D aOrRel0(Pt3dr(0,0,0),0,0,0);
	       aPack0 = aPack;
	       anOrAbs0 = aCam2->CurRot() ;
               if (aLI.InitOrientPure().Val())
               {
                  ELISE_ASSERT(aNbL==1,"Multiple Liaison with InitOrientPure");
                  ElMatrix<REAL> aMat =  aPack.MepRelCocentrique(aLI.NbTestRansacOrPure().Val(),aLI.NbPtsRansacOrPure().Val());
                  aOrRel0 = ElRotation3D(Pt3dr(0,0,0),aMat);
                  aRPure = true;
                  anAltiSol = aCam2->AltiSol();
                  aProfPose = aCam2->Profondeur();
               }
               else if (aLI.OnZonePlane().IsInit())
	       {
	          aNZPl = aLI.OnZonePlane().Val();
	          cResMepRelCoplan aRMRC = aPack.MepRelCoplan(aLBase,aLI.L2EstimPlan().Val());
		  cElemMepRelCoplan & aSP = aRMRC.BestSol();


		  aP0Pl = aSP.P0();
		  aP1Pl = aSP.P1();
		  aP2Pl = aSP.P2();
		  aOrRel0 = aSP.Rot();

		  if (aNbL==1)
		  {
		      double aMul=0;
		      if (DicBoolFind(aCam2->mDZP,aNZPl))
		      {
	                   ELISE_ASSERT(!aBaseFixee,"Ne peut fixe la longueur de base avec liaison plane");
			   aMul = aCam2->mDZP[aNZPl] /aSP.DPlan2();
		      }
		      else
		      {
		          if (aBaseFixee)
			     aMul = 1;
			  else
			  {
			     aMul = aProfC / aPack.Profondeur(aOrRel0);
                          }
                      }
		      aOrRel0.tr() = aOrRel0.tr() * aMul;
		      aDZPl = aSP.DPlan() * aMul;
		      aCam2->mDZP[aNZPl] = aSP.DPlan2() * aMul;
		      aMultPl = aMul;
		  }
		  else
		  {
		       // Sinon il faudra, une fois connu le multiplicateur donne
		       // par les autres liaisons mettre a jour le plan
		       aDZPl = aSP.DPlan() ;
		       // Et eventuellement initialiser Plan2
		       if (! DicBoolFind(aCam2->mDZP,aNZPl))
		       {
		           aDZPl2 = aSP.DPlan2();
			   aCam2PL = aCam2;
		       }
		  }
	       }
	       else
	       {
                   if ((aNbL<2) && (NbPosOfInit(mAppli.NbRotInit()) >=2))
                   {
                       ELISE_ASSERT 
                       (
                           mAppli.Param().AutoriseToujoursUneSeuleLiaison().Val(),
                           "Une seule liaison pour initialiser la pose au dela de 3"
                       );
                   }
	           bool L2 = aPack.size() > mAppli.Param().SeuilL1EstimMatrEss().Val();
                   double aDGen;
// std::cout << "TEST MEPS STD " << mName << "\n";
	           aOrRel0 = aLI.TestSolPlane().Val()               ? 
                              aPack.MepRelGenSsOpt(aLBase,L2,aDGen) :
                             aPack.MepRelPhysStd(aLBase,L2)         ;
		   if (aNbL==1 && (! aBaseFixee))
                   {
		      aPack.SetProfondeur(aOrRel0,aProfC);
                   }
	       }

	       aOrRel0 = aCam2->CurRot() * aOrRel0;
	       aRot = aOrRel0;
	       aRAB = new cResolvAmbiBase(aCam2->CurRot(),aOrRel0);
            }
	    else
	    {
                
	         ELISE_ASSERT
                 (
                         (!aLI.OnZonePlane().IsInit()) || isMST,
                         "Seule la premiere liaison peut etre plane"
                 );
	         // ELISE_ASSERT(false,"Do not handle multi Liaison");
		 aRAB->AddHom(aPack,aCam2->CurRot());
	    }
	 }

	 if (aNbL > 1)
	 {
	      aRot = aRAB->SolOrient(aLambdaRot);
	      aMultPl= aLambdaRot;
	 }
         delete aRAB;

         // Calcul de l'alti et de la prof
         if (aRPure)
         {
         }
         else
	 {
	     CamStenopeIdeale aC1 = CamStenopeIdeale::CameraId(true,aRot.inv());
	     CamStenopeIdeale aC2 = CamStenopeIdeale::CameraId(true,anOrAbs0.inv());
	     double aD;
	     Pt3dr aCdg = aC1.CdgPseudoInter(aPack0,aC2,aD);


             anAltiSol = aCdg.z;
             aProfPose = aC1.ProfondeurDeChamps(aCdg) ;

	     for (int aK=0 ; aK<int(aVPC.size())  ; aK++)
	     {
	         CamStenopeIdeale aCK = CamStenopeIdeale::CameraId(true,aVPC[aK]->CurRot().inv());
		 double aPrK = aCK.ProfondeurDeChamps(aCdg) ;
		 aVPC[aK]->TenteInitAltiProf
		 (
		     (aK==0) ? 1 : 0,
		     aCdg.z,
		     aPrK
		 );
	     }
	 }

         int aNbRAp = pLI0->NbRansacSolAppui().Val();
         if (aNbRAp>0)
         {
             cObsLiaisonMultiple * anOLM = mAppli.PackMulOfIndAndNale(pLI0->IdBD(),mName);
             anOLM->TestMEPAppuis(mAppli.ZuUseInInit(),aRot,aNbRAp,*pLI0);
         }


	 if (aNZPl!="")
	 {
	    mAppli.AddPlan
	    (
	        aNZPl,
		aRot.ImAff(aP0Pl*aMultPl),
		aRot.ImAff(aP1Pl*aMultPl),
		aRot.ImAff(aP2Pl*aMultPl),
                true
	    );
	 }

    }
    else
    {
       ELISE_ASSERT(false,"cPoseCam::Alloc");
    }

//GUIMBAL

    if (GuimbalAnalyse(aRot,false)<mAppli.Param().LimModeGL().Val())
    {
       std::cout << "GUIMBAL-INIT " << mName << "\n";
       mCF->SetGL(true);
    }

    mCF->SetCurRot(aRot);


    if (aNZPl!="")
    {
       mDZP[aNZPl] = aDZPl * aLambdaRot;
       if (aCam2PL)
       {
           if (! DicBoolFind(aCam2PL->mDZP,aNZPl))
               aCam2PL->mDZP[aNZPl] = aDZPl2 * aLambdaRot;
       }
    }
    TenteInitAltiProf(2,anAltiSol,aProfPose);
    mRotIsInit = true;

/*
    {
        ElRotation3D  aR = CurRot() ;
        const CamStenope * aCS = mCF->CameraCourante() ;
        std::cout << " " << mName <<  " " << aR.tr() <<  "\n";
        std::cout << " " << aCS-> R3toF2(Pt3dr(0,0,10)) <<  " " <<  aCS->F2AndZtoR3(Pt2dr(1000,1000),6) <<  "\n";
        
        getchar();
    }
*/

    if (mCompAOI)
    {
        AffineRot();
    }
    mAppli.AddRotInit();

    mCF->ResiduM2C() = mOrIntM2C;

    Trace();


    if (theLiasInit)
    {
             mAppli.CheckInit(theLiasInit,this);
    }
}

void cPoseCam::AffineRot()
{
   for (int aK=0 ; aK<int(mCompAOI->mPats.size()) ; aK++)
   {
      if (    (mCompAOI->mPats[aK]->Match(mName))
           || (mCompAOI->mPats[aK]->Match(ToString(mCpt)))
         )
      {
           vector<cPoseCam *> aVC;
           aVC.push_back(this);
           vector<eTypeContraintePoseCamera> aVT;
           aVT.push_back(mCompAOI->mCstr[aK]);
std::cout << " Opt " << mName << " :: " << mCpt << "\n";
            mAppli.PowelOptimize(mCompAOI->mParam,aVC,aVT);


           return;
      }
   }
}


bool cPoseCam::RotIsInit() const
{
    return mRotIsInit;
}

bool cPoseCam::CanBeUSedForInit(bool OnInit) const
{
   return OnInit ? RotIsInit() : PreInit() ;
}


void cPoseCam::SetFigee()
{
    mRF.SetTolAng(-1);
    mRF.SetTolCentre(-1);
    mRF.SetModeRot(cNameSpaceEqF::eRotFigee);
}

void cPoseCam::SetDeFigee()
{
   if (mLastCP)
      SetContrainte(*mLastCP);
   else
      mRF.SetModeRot(cNameSpaceEqF::eRotLibre);
}

void cPoseCam::SetContrainte(const cContraintesPoses & aCP)
{
   mLastCP = & aCP;
   switch(aCP.Val())
   {
      case ePoseLibre :
          ELISE_ASSERT
	  (
	       (aCP.TolAng().Val()<=0)&&(aCP.TolCoord().Val()<=0),
	       "Tolerance inutile avec ePoseLibre"
	  );
          mRF.SetModeRot(cNameSpaceEqF::eRotLibre);
      break;

      case ePoseFigee :
           mRF.SetTolAng(aCP.TolAng().Val());
           mRF.SetTolCentre(aCP.TolCoord().Val());
	   mRF.SetModeRot(cNameSpaceEqF::eRotFigee);
      break;


      case eCentreFige :
           ELISE_ASSERT
	   (
	       (aCP.TolAng().Val()<=0),
	       "Tolerance angulaire avec eCentreFige"
	   );
           mRF.SetTolCentre(aCP.TolCoord().Val());
	   mRF.SetModeRot(cNameSpaceEqF::eRotCOptFige);
      break;




      case ePoseBaseNormee :
      case ePoseVraieBaseNormee :
           ELISE_ASSERT
	   (
	       aCP.PoseRattachement().IsInit(),
	       "Rattachement non initialise !"
	   );
          ELISE_ASSERT
	  (
	       (aCP.TolAng().Val()<=0),
	       "Tolerance angle inutile avec ePoseBaseNormee"
	  );

           mRF.SetTolCentre(aCP.TolCoord().Val());
           if (aCP.Val()==ePoseVraieBaseNormee) 
           {
                SetRattach(aCP.PoseRattachement().Val());
	       mRF.SetModeRot(cNameSpaceEqF::eRotBaseU);
           }
           else
           {
               cPoseCam * aPR  = mAppli.PoseFromName(aCP.PoseRattachement().Val());
               mRF.SetRotPseudoBaseU(&(aPR->mRF));
	       mRF.SetModeRot(cNameSpaceEqF::eRotPseudoBaseU);
           }
      break;

   }
}


    //   Gestion image

void cPoseCam::InitIm()
{
    mFirstBoxImSet = false;
    mImageLoaded = false;

}


bool cPoseCam::PtForIm(const Pt3dr & aPTer,const Pt2di & aRab,bool Add)
{
    const CamStenope * aCS = mCF->CameraCourante() ;
    Pt2dr aPIm =  aCS->R3toF2(aPTer);
    
    Box2di aCurBIm(round_down(aPIm)-aRab,round_up(aPIm)+aRab);
    Box2di aFulBox(Pt2di(0,0),aCS->Sz());
   
    if (! aCurBIm.include_in(aFulBox)) 
       return false;

    if (Add)
    {
       ELISE_ASSERT(!mImageLoaded,"cPoseCam::PtForIm : Im Loaded in Add");
       if (mFirstBoxImSet)
       {
          mBoxIm = Sup(mBoxIm,aCurBIm);
       }
       else
       {
          mBoxIm = aCurBIm;
          mFirstBoxImSet = true;
       }
    }

    return true;
}

void cPoseCam::ResetStatR()
{
  mStatRSomP =0;
  mStatRSomPR =0;
  mStatRSom1 =0;
}

void cPoseCam::AddStatR(double aPds,double aRes)
{
  mStatRSomP += aPds;
  mStatRSomPR += aPds * aRes;
  mStatRSom1 += 1;
}

void cPoseCam::GetStatR(double & aSomP,double & aSomPR,double & aSom1) const
{
   aSomP = mStatRSomP;
   aSomPR = mStatRSomPR;
   aSom1  = mStatRSom1;
}


bool cPoseCam::ImageLoaded() const
{
   return mImageLoaded;
}

void cPoseCam::AssertImL() const
{
    if (!mImageLoaded)
    {
       std::cout << "For cam=" << mName << "\n";
       ELISE_ASSERT(false,"Image not Loaded");
    }
}

const Box2di & cPoseCam::BoxIm()
{
   AssertImL();
   return mBoxIm;
}

Im2D_U_INT2  cPoseCam::Im()
{
   AssertImL();
   return mIm;
}


void cPoseCam::CloseAndLoadIm(const Pt2di & aRab)
{
    if (! mFirstBoxImSet) 
       return;
    mImageLoaded = true;

    {
       Pt2di aP0 = Sup(mBoxIm._p0-aRab,Pt2di(0,0));
       Pt2di aP1 = Inf(mBoxIm._p1+aRab,mCF->CameraCourante()->Sz());
       mBoxIm = Box2di(aP0,aP1);
    }
     

    Pt2di aSz = mBoxIm.sz();
    mIm.Resize(aSz);
    mTIm = TIm2D<U_INT2,INT>(mIm);
    Tiff_Im aTF = Tiff_Im::StdConvGen(mAppli.DC()+mName,1,true,false);

    ELISE_COPY
    (
        mIm.all_pts(),
        trans(aTF.in_proj(),mBoxIm._p0),
        mIm.out()
    );
}




    //   ACCESSEURS 

cCalibCam * cPoseCam::Calib() { return mCalib;}
cCameraFormelle * cPoseCam::CF() {return mCF;}
const  std::string & cPoseCam::Name() const {return mName;}
double cPoseCam::AltiSol() const {return mAltiSol;}
double cPoseCam::Profondeur() const {return mProfondeur;}

bool cPoseCam::HasMasqHom() const { return mMasqH !=0; }
int  cPoseCam::NumInit() const {return mNumInit;}


bool &   cPoseCam::MMSelected() { return mMMSelected;}
double & cPoseCam::MMGain()     { return  mMMGain;}
double & cPoseCam::MMAngle()    { return mMMAngle;}
Pt3dr  & cPoseCam::MMDir()      { return mMMDir;}


double & cPoseCam::MMNbPts()    { return  mMMNbPts;}
double & cPoseCam::MMGainAng()  { return  mMMGainAng;}


/*********************************************************/
/*                                                       */
/*                 cAppliApero                           */
/*                                                       */
/*********************************************************/


void   cAppliApero::LoadImageForPtsMul
       (
          Pt2di aRabIncl,
          Pt2di aRabFinal,
          const std::list<cOnePtsMult *> & aLMul
       )
{
    for (int aK=0; aK<int(mVecPose.size()) ; aK++)
    {
        mVecPose[aK]->InitIm();
    } 

    for
    (
        std::list<cOnePtsMult *>::const_iterator itPM=aLMul.begin();
        itPM!=aLMul.end();
        itPM++
    )
    {
         std::vector<double> aVPds;
         const cResiduP3Inc * aRes = (*itPM)->ComputeInter(1.0,aVPds);
         if (aRes)
         {
             for (int aK=0; aK<int(mVecPose.size()) ; aK++)
             {
                 mVecPose[aK]->PtForIm(aRes->mPTer,aRabIncl,true);
             } 
         }
    }

    mVecLoadedPose.clear();
    for (int aK=0; aK<int(mVecPose.size()) ; aK++)
    {
        mVecPose[aK]->CloseAndLoadIm(aRabFinal);
        if (mVecPose[aK]->ImageLoaded())
           mVecLoadedPose.push_back(mVecPose[aK]);
    } 
}


const std::vector<cPoseCam*> &  cAppliApero::VecLoadedPose()
{
    return mVecLoadedPose;
}


std::vector<cPoseCam*>  cAppliApero::VecLoadedPose(const cOnePtsMult & aPM,int aSz)
{
    std::vector<cPoseCam*> aRes;
    std::vector<cPoseCam*> aRes2;

    std::vector<double> aVPds;
    const cResiduP3Inc * aResidu = aPM.ComputeInter(1.0,aVPds);
    Pt2di aPRab(aSz,aSz);

    if (aResidu)
    {
       for (int aKp=0 ; aKp<int(mVecLoadedPose.size()) ; aKp++)
       {
           if (mVecLoadedPose[aKp]->PtForIm(aResidu->mPTer,aPRab,false))
           {
               if (mVecLoadedPose[aKp]==aPM.Pose0())
               {
                   aRes.push_back(mVecLoadedPose[aKp]);
               }
               else
               {
                   aRes2.push_back(mVecLoadedPose[aKp]);
               }
           }
       }
       std::cout << "NB -------  " << aRes.size() << " ## " << aRes2.size() << "\n";
       if ((int(aRes.size())==1) && (int(aRes2.size())>= 1))
       {
          for (int aKp=0 ; aKp<int(aRes2.size()) ; aKp++)
          {
              aRes.push_back(aRes2[aKp]);
          }
       }
       else
       {
          aRes.clear();
       }
    }

    return aRes;
}

bool   cPoseCam::DoAddObsCentre(const cObsCentrePDV & anObs)
{
   if (! mHasObsCentre)
      return false;

   if (  
          (anObs.PatternApply().IsInit())
       && (!anObs.PatternApply().Val()->Match(mName))
      )
      return false;

   return true;
}

Pt3dr cPoseCam::CurCentre() const
{
    return  CurRot().ImAff(Pt3dr(0,0,0));
}

Pt3dr  cPoseCam::AddObsCentre
      (
           const cObsCentrePDV & anObs,
           const cPonderateur &  aPondPlani,
           const cPonderateur &  aPondAlti,
           cStatObs & aSO
      )
{
   ELISE_ASSERT(DoAddObsCentre(anObs),"cPoseCam::AddObsCentre");
   Pt3dr aC0 = CurRot().ImAff(Pt3dr(0,0,0));
   Pt3dr aDif = aC0 - mObsCentre;
   Pt2dr aDifPlani(aDif.x,aDif.y);

   double aPdsP  = aPondPlani.PdsOfError(euclid(aDifPlani)/sqrt(2.));
   double aPdsZ  = aPondAlti.PdsOfError(ElAbs(aDif.z));

   Pt3dr aPPds = aSO.AddEq() ? Pt3dr(aPdsP,aPdsP,aPdsZ) : Pt3dr(0,0,0) ; 
   Pt3dr aRAC = mRF.AddRappOnCentre(mObsCentre,aPPds ,false);


   double aSEP = aPdsP*(ElSquare(aRAC.x)+ElSquare(aRAC.y))+aPdsZ*ElSquare(aRAC.z);
   //  std::cout << "========= SEP " << aSEP << "\n";
   aSO.AddSEP(aSEP);


   mAppli.AddResiducentre(aDif);

   if (aPondPlani.PPM().Show().Val() >=eNSM_Indiv)
   {
        std::cout << mName << " DeltaC " <<  euclid(aDif) << " " << aDif << "\n";
   }

   if (anObs.ShowTestVitesse().Val())
   {
      // Montre le linkage GPS
      {
          ElRotation3D  aR = CurRot() ;
          mAppli.COUT() << aR.teta01()  << " ";
          if (mPrec != mNext)
          {
              Pt3dr aVC = mNext->CurCentre()- mPrec->CurCentre() ;
              double aDT = mNext->mTime -  mPrec->mTime;
              Pt2dr aV2C(aVC.x,aVC.y);

              Pt3dr aVG = mNext->mObsCentre - mPrec->mObsCentre ;
              Pt3dr  aVitG = aVG / aDT;

              Pt3dr aGC = CurCentre() -mObsCentre;

              double aRetard = scal(aGC,aVitG) /scal(aVitG,aVitG);

              mAppli.AddRetard(aRetard);

              Pt2dr aV2G(aVG.x,aVG.y);
              mAppli.COUT() << " Retard " << aRetard
                            << " VIt " << euclid(aVitG)
                            << " Traj " << (aR.teta01() - atan2(aV2C.y,aV2C.x) -PI/2) << " "
                             << " TGps " << (aR.teta01() - atan2(aV2G.y,aV2G.x) -PI/2) << " ";
           
          }
          mAppli.COUT()  << "\n";
      }
/*
      mAppli.COUT().precision(10);
      mAppli.COUT() << "RESIDU CENTRE " << aDif << " pour " << mName 
                     << " AERO=" <<  aC0 << " GPS=" << mObsCentre<< "\n";
*/
   }

   return aDif;
    
}

cOneImageOfLayer * cPoseCam::GetCurLayer()
{
   if (mCurLayer==0)
   {
      std::cout << "FOR NAME POSE " << mName << "\n";
      ELISE_ASSERT(false,"Cannot get layer");
   }
   return mCurLayer;
}

void cPoseCam::SetCurLayer(cLayerImage * aLI)
{
    mCurLayer = aLI->NamePose2Layer(mName);
}

/************************************************************/
/*                                                          */
/*              cClassEquivPose                             */
/*              cRelEquivPose                               */
/*                                                          */
/************************************************************/

             // =======   cClassEquivPose  ====

cClassEquivPose::cClassEquivPose(const std::string & anId) :
   mId (anId)
{
}

void cClassEquivPose::AddAPose(cPoseCam * aPC)
{
    if (BoolFind(mGrp,aPC))
    {
        std::cout << "For Pose : " << aPC->Name() << "\n";
        ELISE_ASSERT(false,"cClassEquivPose::AddAPose multiple name");
    }
    mGrp.push_back(aPC);
}

const std::vector<cPoseCam *> &   cClassEquivPose::Grp() const
{
   return mGrp;
}

const std::string & cClassEquivPose::Id() const
{
    return mId;
}


             // =======   cRelEquivPose  ====

cRelEquivPose::cRelEquivPose(int aNum) :
   mNum (aNum)
{
}

cClassEquivPose * cRelEquivPose::AddAPose(cPoseCam * aPC,const std::string & aName)
{
   cClassEquivPose * & aCEP = mMap[aName];
   if (aCEP==0) 
      aCEP = new cClassEquivPose(aName);
   aCEP->AddAPose(aPC);

   mPos2C[aPC->Name()] = aCEP;
   return aCEP;
}

cClassEquivPose &  cRelEquivPose::ClassOfPose(const cPoseCam & aPC)
{
   cClassEquivPose * aCEP = mPos2C[aPC.Name()];
   if (aCEP==0)
   {
       std::cout << "For Pose " << aPC.Name() << "\n";
       ELISE_ASSERT(false,"Can get Class in cRelEquivPose::ClassOfPose");
   }
   return *aCEP;
}

bool cRelEquivPose::SameClass(const cPoseCam & aPC1,const cPoseCam & aPC2)
{
   return ClassOfPose(aPC1).Id() == ClassOfPose(aPC2).Id();
}



const std::map<std::string,cClassEquivPose *> &  cRelEquivPose::Map() const
{
   return mMap;
}

void cRelEquivPose::Show()
{
    std::cout << "========== REL NUM " << mNum << "==================\n";

   for 
   (
        std::map<std::string,cClassEquivPose *>::const_iterator itM=mMap.begin();
        itM!=mMap.end();
        itM++
   )
   {
          const cClassEquivPose& aCl = *(itM->second);
          const std::vector<cPoseCam *> & aGrp = aCl.Grp() ;

          if (aGrp.size() == 1)
             std::cout << aCl.Id() << " ::  " << aGrp[0]->Name()<< "\n";
          else 
          {
             std::cout << "## " << aCl.Id() << " ##\n";
             for (int aK=0 ; aK<int(aGrp.size()) ; aK++)
                std::cout <<  "  --- "  << aGrp[aK]->Name()<< "\n";
          }
          
   }
}
  

             // =======   cAppliApero  ====

cRelEquivPose * cAppliApero::RelFromId(const std::string & anId)
{
   cRelEquivPose * aRes = mRels[anId];
   if (aRes ==0)
   {
      std::cout << "For Id = " << anId << "\n";
      ELISE_ASSERT(false,"cAppliApero::RelFromId do not exist");
   }

   return aRes;
}


bool cAppliApero::SameClass(const std::string& anId,const cPoseCam & aPC1,const cPoseCam & aPC2)
{
   return RelFromId(anId)->SameClass(aPC1,aPC2);
}



void cAppliApero::AddObservationsRigidGrp(const cObsRigidGrpImage & anORGI,bool IsLastIter,cStatObs & aSO)
{
   cRelEquivPose * aREP = RelFromId(anORGI.RefGrp());
   const std::map<std::string,cClassEquivPose *> &  aMap = aREP->Map();
   for 
   ( 
       std::map<std::string,cClassEquivPose *>::const_iterator itG=aMap.begin();
       itG!=aMap.end();
       itG++
   )
   {
        const std::vector<cPoseCam *> & aGrp = itG->second->Grp();
        int aNb = aGrp.size();
        if (aNb>=2)
        {
            for (int aK1=0 ; aK1<aNb ; aK1++)
            {
                for (int aK2=aK1+1 ; aK2<aNb ; aK2++)
                {
                    cRotationFormelle & aRF1 =  aGrp[aK1]->RF();
                    cRotationFormelle & aRF2 =  aGrp[aK2]->RF();
                    if (anORGI.ORGI_CentreCommun().IsInit())
                    {
                       Pt3dr aPInc = anORGI.ORGI_CentreCommun().Val().Incertitude();
                       double anInc[3];
                       aPInc.to_tab(anInc);
                       for (int aD=0 ; aD<3 ; aD++)
                       {
                           if (anInc[aD]>0)
                           {
                              double aR = mSetEq.AddEqEqualVar(ElSquare(1.0/anInc[aD]),aRF1.NumCentre(aD),aRF2.NumCentre(aD),true);
                              aSO.AddSEP(aR);
                              
                           }
                       }
                    }
                    if (anORGI.ORGI_TetaCommun().IsInit())
                    {
                       Pt3dr aPInc = anORGI.ORGI_TetaCommun().Val().Incertitude();
                       double anInc[3];
                       aPInc.to_tab(anInc);
                       for (int aD=0 ; aD<3 ; aD++)
                       {
                           if (anInc[aD]>0)
                           {
                              double aR  = mSetEq.AddEqEqualVar(ElSquare(1.0/anInc[aD]),aRF1.NumTeta(aD),aRF2.NumTeta(aD),true);
                              aSO.AddSEP(aR);
                           }
                       }
                    }
                }
            }
        }
   }
}


};

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
