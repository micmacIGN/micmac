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

#define BDDL_FIRST 0

namespace NS_ParamApero
{

// Pt2dr BugIM(2591.0926,483.7226);
// Pt3dr BugTER(921804.4349,3212619.4133,889.2173);


// typedef std::list<cCalibrationCameraInc> tLC;
// typedef std::list<cPoseCameraInc> tLP;

cAppliApero::cAppliApero (cResultSubstAndStdGetFile<cParamApero> aParam) : 
   mParam             (*(aParam.mObj)),
   mDC                (aParam.mDC),
   mICNM              (aParam.mICNM),
   mSetEq             (ToNS_EqF(mParam.ModeResolution()),1),
   mAMD               (0),
   mGr                (),
   mMapMaskHom        (StdAllocMn2n(mParam.MapMaskHom(),mICNM)),
   mNbRotInit         (0),
   mNbRotPreInit      (0),
   mProfInit          (0),
   mProfMax           (0),
   mModeMaping        ( IsActive(mParam.SectionMapApero())),
   mShowMes           (   (! mParam.ShowSection().IsInit())
                        || (mParam.ShowSection().Val().ShowMes().Val())
                      ),
   mHasLogF           (false),
   mParamEtal         ( mParam.NameParamEtal().IsInit() ?
                        (new cParamEtal(cParamEtal::FromStr(mDC+mParam.NameParamEtal().Val()))):
                        0
                      ),
   mMTAct             (0),
   mMTRes             (0),
   mAutomTracePose    (
                          mParam.PatternTracePose().IsInit() ?
                          new  cElRegex(mParam.PatternTracePose().Val(),10) :
                          0
                      ),
   mCurPbLiaison      (mParam.DefPbLiaison().PtrVal()),
   mNbEtape           (0),
   mCurSLMGlob        (0),
   mMulSLMGlob        (1.0),
   mCurSLMEtape       (0),
   mMulSLMEtape       (1.0),
   mCurSLMIter        (0),
   mMulSLMIter        (1.0),
   mNumSauvAuto       (0),
   mFpRT              (0)
{
     // ::DebugPbCondFaisceau = mParam.DebugPbCondFaisceau().Val(); => Apero.cpp
     // ::aSeuilMaxCondSubstFaiseau =  mParam.SeuilMaxCondSubstFaiseau().Val();

     ::TestPermutVar = mParam.TestPermutVar().Val();
     ::ShowPermutVar = mParam.ShowPermutVar().Val();
     ::ShowCholesky = mParam.ShowCholesky().Val();
     // ::PermutIndex   = mParam.PermutIndex().Val();

     if (    mParam.ShowSection().IsInit()
          && mParam.ShowSection().Val().LogFile().IsInit()
        )
     {
          mLogName = mDC + mParam.ShowSection().Val().LogFile().Val();
          mHasLogF = true;
          mLogFile.open(mLogName.c_str(),ios::out|ios::ate|ios::app);
     }


     if ( !mModeMaping)
     {
         bool Verbose = ShowMes();


         if (Verbose) COUT() << "BEGIN Pre-compile\n";
         PreCompile();



        if (Verbose) COUT() << "BEGIN Load Observation\n";
        InitBDDObservations();

        if (Verbose)  COUT()<< "BEGIN Init Inconnues\n";
        InitInconnues();

        if (Verbose)  COUT() << "BEGIN Compensation\n";
        CompileObsersvations();

        
        DoAMD();
        mSetEq.SetClosed();

        InitFilters();

        Verifs();

        if (mParam.TimeLinkage().IsInit())
        {
           InitPosesTimeLink(mParam.TimeLinkage().Val());
        }
    }
    InitLVM(mCurSLMGlob,mParam.SLMGlob(),mMulSLMGlob,mParam.MultSLMGlob());
}

FILE *  cAppliApero::FpRT() 
{
    return mFpRT;
}

class cCmpTimePose
{
     public :
       bool operator ()(const cPoseCam * aPC1,const cPoseCam * aPC2) const
       { 
          return aPC1->Time() < aPC2->Time();
       }
};

void  cAppliApero::InitPosesTimeLink(const cTimeLinkage & aTLnk)
{
   mTimeVP = mVecPose;
   cCmpTimePose aCmpTP;
   std::sort(mTimeVP.begin(),mTimeVP.end(),aCmpTP);
   double aDelta = aTLnk.DeltaMax();

   for (int aK=1 ; aK<int(mTimeVP.size()) ; aK++)
   {
       cPoseCam * aPCPred = mTimeVP[aK-1];
       cPoseCam * aPCCur  = mTimeVP[aK];
       double aTPred  = aPCPred->Time();
       double aTCur   = aPCCur->Time();
       bool OKLink =    (aTPred!= TIME_UNDEF())
                     && (aTCur!= TIME_UNDEF())
                     && (ElAbs(aTPred-aTCur) < aDelta);
       aPCCur->SetLink(aPCPred,OKLink);
   }
}


void cAppliApero::DoAMD()
{
   if (mParam.ModeResolution() != eSysL2BlocSym)
      return;

 
   std::cout << "BEGIN AMD \n";

   int aNbBl = mSetEq.NbBloc();
   mAMD = new cAMD_Interf (aNbBl);

   for (int aK=0 ; aK<aNbBl ; aK++)
   {
        mAMD->AddArc(aK,aK,true);
   }

   for
   (
      std::map<std::string,cBdAppuisFlottant *>::iterator it= mDicPF.begin();
      it != mDicPF.end();
      it++
   )
   {
        it->second->DoAMD(mAMD);
   }

   for 
   (
       std::set<std::pair<cPoseCam *,cPoseCam *> >::const_iterator it=mSetLinkedCam.begin();
       it!=mSetLinkedCam.end();
       it++
   )
   {
       int aNums[4];
       cPoseCam * aPC1 = it->first;
       cCalibCam * aCal1 = aPC1->Calib();
       cParamIntrinsequeFormel &  aPIF1 = aCal1->PIF();
       aNums[0] = aPIF1.IncInterv().NumBlocAlloc();
       aNums[1] = aPC1->RF().IncInterv().NumBlocAlloc();


       cPoseCam * aPC2 = it->second;
       cCalibCam * aCal2 = aPC2->Calib();
       cParamIntrinsequeFormel &  aPIF2 = aCal2->PIF();
       aNums[2] = aPIF2.IncInterv().NumBlocAlloc();
       aNums[3] = aPC2->RF().IncInterv().NumBlocAlloc();


       for (int aK1=0 ; aK1 <4 ; aK1++)
           for (int aK2=aK1 ; aK2 <4 ; aK2++)
                mAMD->AddArc(aNums[aK1],aNums[aK2],true);


   }

/*
   for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
   {
       cPoseCam * aPose = mVecPose[aKP];
       cCalibCam *  aCC = aPose->Calib();
       cParamIntrinsequeFormel &  aPIF = aCC->PIF();
       int aNC = aPIF.IncInterv().NumBlocAlloc();
       int aNP = aPose->RF().IncInterv().NumBlocAlloc();

       mAMD->AddArc(aNC,aNP,true);

       const std::vector<cPoseCam *> aVL = aPose->VPLinked();
       for (int aKP2=0 ; aKP2<int(aVL.size()) ; aKP2++)
       {
             int aNP2 = aVL[aKP2]->RF().IncInterv().NumBlocAlloc();
             mAMD->AddArc(aNP,aNP2,true);
       }
   }
*/

   std::vector<int>  anOrder = mAMD->DoRank(::ShowCholesky);
   const std::vector<cIncIntervale *>  & aVI = mSetEq.BlocsIncAlloc();
   for (int aK=0 ; aK<int(aVI.size()) ; aK++)
   {
       aVI[aK]->SetOrder(anOrder[aK]);
       if (mParam.InhibeAMD().Val())
       {
           aVI[aK]->SetOrder(aK);
       }
       // aVI[aK]->SetOrder(aK);
//     std::cout << "aVI " << aK << " " << aVI[aK]->Order() << "\n";
   }

/*
   for (tDiCal::const_iterator it=mDicoCalib.begin(); it!=mDicoCalib.end() ; it++)
   {
        cCalibCam * aCC =  it->second;
        cParamIntrinsequeFormel &  aPIF = aCC->PIF();
        int aNum = aPIF.IncInterv().NumBlocAlloc();
        std::cout << "Num Calib " << aNum << "\n";
        mAMD->AddArc(aNum,aNum);
        // mAMD->
   }
*/

   
   std::cout << "END AMD \n";
   
}

void cAppliApero::AddLinkCam(cPoseCam * aC1,cPoseCam * aC2)
{
   if (aC1>aC2) 
      ElSwap(aC1,aC2);
  mSetLinkedCam.insert(std::pair<cPoseCam *,cPoseCam *>(aC1,aC2)); 
  AddLinkCamCal(aC1,aC1->Calib());
  AddLinkCamCal(aC2,aC2->Calib());
  AddLinkCal(aC1->Calib(),aC2->Calib());
}


void cAppliApero::AddLinkCal(cCalibCam  * aC1,cCalibCam  * aC2)
{
   if (aC1>aC2) 
      ElSwap(aC1,aC2);
  mSetLinkedCal.insert(std::pair<cCalibCam *,cCalibCam *>(aC1,aC2)); 
}

bool cAppliApero::ZuUseInInit() const
{
  return true;
}

void cAppliApero::AddLinkCamCal(cPoseCam *aCam,cCalibCam  * aCalib)
{
   mSetLinkedCamCal.insert(std::pair<cPoseCam *,cCalibCam *>(aCam,aCalib));
}

void cAppliApero::InitFilters()
{
  for 
  (
      std::list<cFilterProj3D>::iterator itF=mParam.FilterProj3D().begin();
      itF!=mParam.FilterProj3D().end();
      itF++
  )
  {
      ELISE_ASSERT(mMapFilters[itF->Id()]==0,"Multiple Filter");
      mMapFilters[itF->Id()] = new cCompFilterProj3D(*this,*itF);
  }
}

cCompFilterProj3D * cAppliApero::FilterOfId(const std::string& aName)
{
   cCompFilterProj3D * aRes = mMapFilters[aName];
   if (aRes==0)
   {
        std::cout << "For name " << aName << "\n";
        ELISE_ASSERT(false,"cAppliApero::FilterOfId");
   }
   return aRes;
}

bool cAppliApero::TracePose(const std::string & aName) const
{
   return mAutomTracePose  && mAutomTracePose->Match(aName);
}


bool cAppliApero::TracePose(const cPoseCam & aCam) const
{
   return TracePose(aCam.Name());
}


bool cAppliApero::AcceptCible(int aNum) const
{
   if (mParamEtal==0) 
      return true;

   return ! BoolFind( mParamEtal->CiblesRejetees(),aNum);
}

void cAppliApero::DoMaping(int argc,char ** argv)
{
   mICNM->SetMapCmp(mParam.SectionMapApero().Val(),argc,argv);
}


ostream &     cAppliApero::COUT()
{
	if (mHasLogF)
		return mLogFile;
	else
		return (std::cout);
}


cAppliApero::~cAppliApero()
{
   if (mHasLogF)
      mLogFile.close();
}

void cAppliApero::AddRotInit()
{
    mNbRotInit++;
}

int  cAppliApero::NbRotInit() const { return mNbRotInit; }
int  cAppliApero::NbRotPreInit() const { return mNbRotPreInit; }

void cAppliApero::AddRotPreInit()
{
    mNbRotPreInit++;
}



bool  cAppliApero::ModeMaping() const
{
   return mModeMaping;
}


bool  cAppliApero::ShowMes() const
{
   return mShowMes;
}



Im2D_Bits<1> * cAppliApero::MasqHom(const std::string & aName)
{
   if (! mMapMaskHom)
     return 0;

   std::string aDef="";
   std::string aNamMasq = mMapMaskHom->map_with_def(aName,aDef);

   if (aNamMasq== aDef)
      return 0;

  Tiff_Im aFile = Tiff_Im::StdConvGen(mDC+aNamMasq,1,true,false);
  Pt2di aSz = aFile.sz();
  Im2D_Bits<1> aRes(aSz.x,aSz.y);
  ELISE_COPY(aRes.all_pts(),aFile.in_bool(),aRes.out());
 
  return new Im2D_Bits<1>(aRes);
}


void cAppliApero::InitLayers()
{
   for
   (
       std::list<cLayerImageToPose>::iterator itL = mParam.LayerImageToPose().begin();
       itL !=  mParam.LayerImageToPose().end();
       itL++
   )
   {
      cLayerImage * & aLI = mMapLayers[itL->Id()];
      ELISE_ASSERT(aLI==0,"Multiple cAppliApero::InitLayers");
      aLI = new cLayerImage(*this,*itL);
   }
}


void cAppliApero::PreCompile()
{
    InitLayers();
    InitCalibCam();

    PreCompilePose();
    InitClassEquiv();
    PreCompileAppuisFlottants();

{
  std::cout << "TEST-COMPILE-NOW-InitAndCompileBDDObsFlottant \n";
  InitAndCompileBDDObsFlottant();
  // getchar();
}
}

void  cAppliApero::InitBDDObservations()
{
    InitBDDLiaisons();
    InitBDDAppuis();
    InitBDDOrient();
    InitBDDCentre();
}


void  cAppliApero::CompileObsersvations()
{
  CompileLiaisons();
  CompileAppuis();
  CompileOsbOr();
  CompileObsCentre();

}

void cAppliApero::Verifs()
{
  VerifSurf();
}


void  cAppliApero::NewSymb(const std::string & aName)
{
  AssertEntreeDicoVide(mLSymbs,aName,"Symboles");
  mLSymbs.insert(aName);
}




//    ACCES AUX DICTIONNAIRES

bool  cAppliApero::CalibExist(const std::string & aName)
{
   return mDicoCalib.find(aName) != mDicoCalib.end();
}

cCalibCam * cAppliApero::CalibFromName(const std::string & aName,cPoseCam * aPC)
{
   {
      tDiCal::const_iterator anIt = mDicoCalib.find(aName);

       if (anIt == mDicoCalib.end())
       {
           if (aPC==0)
           {
              std::cout  << "Name = " << aName << "\n";
              ELISE_ASSERT(false,"Key do not exist cAppliApero::CalibFromName");
           }
       }
       else
       {
          if (anIt->second != 0)
             return anIt->second;
           else
           {
              std::cout  << "Name = " << aName << "\n";
              ELISE_ASSERT(false,"Internal error, 0 in Dic :  cAppliApero::CalibFromName");
           }
       }
      
   }

   if (! aPC)
   {
       std::cout  << "Name = " << aName << "\n";
       ELISE_ASSERT(false,"cAppliApero::CalibFromName Not creation phse");
   }

   tDiArgCab::const_iterator anIt = mDicoArgCalib.find(aName);

   if (anIt == mDicoArgCalib.end())
   {
       std::cout  << "Name = " << aName << "\n";
       ELISE_ASSERT(false,"Key do not exis,  cAppliApero::CalibFromName");
   }

   const cCalibrationCameraInc * aCCI = anIt->second;
   if (! aCCI->CalibPerPose().IsInit())
   {
       std::cout  << "Name = " << aName << "\n";
       ELISE_ASSERT(false,"No init Differee, cAppliApero::CalibFromName");
   }

   const cCalibPerPose & aCPP = aCCI->CalibPerPose().Val();

   std::string aNameCalib = mICNM->Assoc1To1(aCPP.KeyPose2Cal(),aPC->Name(),true);

   cCalibCam *aCC =  mDicoCalib[aNameCalib];
   if (aCC==0)
   {
   
      cCalibrationCameraInc * aNewCCI = new cCalibrationCameraInc(*aCCI);
      aNewCCI->Name() = aNameCalib;
      aCC = cCalibCam::Alloc(aNameCalib,*this,*aNewCCI,aPC);
      mDicoCalib[aNameCalib] = aCC;

      std::cout << "NEW CALIB " << aNameCalib << "\n";
   }
   aPC->SetNameCalib(aNameCalib);
   

   return aCC;
}





cPackObsLiaison * cAppliApero::GetPackOfName(const std::string& anId)
{
   return GetEntreeNonVide(mDicoLiaisons,anId,"cAppliApero::GetPackOfName");
}

bool  cAppliApero::PoseExist(const std::string & aName)
{
   return mDicoPose.find(aName) != mDicoPose.end();
}

cPoseCam * cAppliApero::PoseFromNameGen(const std::string & aName,bool SVP)
{
   tDiPo::iterator iT = mDicoPose.find(aName);
   if (iT==mDicoPose.end())
   {
       iT = mDPByNum.find(aName);
       if (iT != mDPByNum.end())
          return iT->second;
       if (SVP)
          return 0;
       std::cout << "NAME =[" << aName << "]\n";
       ELISE_ASSERT(false,"cAppliApero::PoseFromName");
   }
   return iT->second;
}

cPoseCam * cAppliApero::PoseFromName(const std::string & aName)
{
    return PoseFromNameGen(aName,false);
}
cPoseCam * cAppliApero::PoseFromNameSVP(const std::string & aName)
{
    return PoseFromNameGen(aName,true);
}
 

std::vector<cPoseCam *> cAppliApero::ListPoseOfPattern(const std::string & aPat)
{
   cElRegex anAutom(aPat,20);
   std::vector<cPoseCam *>  aRes;

   for (tDiPo::const_iterator itP=mDicoPose.begin(); itP!=mDicoPose.end(); itP++)
   {
       cPoseCam * aPC= itP->second;
       const std::string & aNP = aPC->Name();
       if (anAutom.Match(aNP))
       {
           aRes.push_back(aPC);
       }
   }
   return aRes;
}





void cAppliApero::AddPoseInit(int aNum,cPoseCam * aPose)
{
   mDPByNum[ToString(aNum)] = aPose;
}

const std::string & cAppliApero::NameCalOfPose(const std::string & aNP)
{
   cPoseCam * aPC = mDicoPose[aNP];
   if (aPC==0)
   {
       std::cout << "POSE=" << aNP << "\n";
       ELISE_ASSERT(false,"cAppliApero::NameCalOfPose");
   }
   return aPC->NameCalib();
}


bool cAppliApero::NamePoseIsKnown(const std::string & aName) const
{
      return mDicoPose.find(aName) != mDicoPose.end();
}


tGrApero &  cAppliApero::Gr()
{
   return mGr;
}

void cAppliApero::VerifSurf()
{
   for 
   (
       std::map<std::string,cSurfParam *>::const_iterator itS=mDicoSurfParam.begin();
       itS!=mDicoSurfParam.end();
       itS++
   )
   {
       itS->second->AssertUsed();
   }
}

cSurfParam * cAppliApero::AddPlan
        (
             const std::string & aName,
             const Pt3dr  &   aP0,
             const Pt3dr  &   aP1,
             const Pt3dr  &   aP2,
             bool  CanExist   
        )
{
    cSurfParam * aRes = mDicoSurfParam[aName];
    if (aRes==0)
    {
	aRes=  mDicoSurfParam[aName] = cSurfParam::NewSurfPlane(*this,aName,aP0,aP1,aP2);
    }
    else
    {
        if (! CanExist)
	{
            std::cout << " ---- For name plan = "<< aName << "\n";
	    ELISE_ASSERT(false,"Multiple definition for plan");
	}
    }

    return aRes;
}


cBdAppuisFlottant *  cAppliApero::BAF_FromName(const std::string & aName,bool CanCreate,bool SVP)
{
   cBdAppuisFlottant * &  aRes =  mDicPF[aName];

   if (aRes == 0)
   {
       if (!CanCreate)
       {
           if (SVP)
           {
              return 0;
           }
           std::cout << "For Name =" << aName << "\n";
	   ELISE_ASSERT(false,"cAppliApero::BAF_FromName");
       }
       aRes = new cBdAppuisFlottant(*this);
   }

   return aRes;
}

double cAppliApero::PdsOfPackForInit(const ElPackHomologue & aPack,int & aNb)
{
   aNb = aPack.size();
   if (aNb==0)
      return 0;
   StatElPackH aStat(aPack);
   return aStat.SomD1() + aStat.SomD2();
}


/*
cSurfParam * cAppliApero::SurfIncFromName(const std::string & aName)
{
    cSurfParam * aSurf = mDicoSurfParam[aName];
    if (aPl==0)
    {
       std::cout << " ---- For name plan = "<< aName << "\n";
       ELISE_ASSERT(false,"Cannot Get plan in EqPlIncFromName");
    }

    cEqPlanInconnuFormel * anEq = aPl-> EqPlInc();
    if (anEq==0)
    {
       std::cout << " ---- For name plan = "<< aName << "\n";
       ELISE_ASSERT(false,"Cannot Get eq in EqPlIncFromName");
    }

    return aSurf;
}
*/


//    ACCESSEURS BASIQUES

const cParamApero & cAppliApero::Param() const {return mParam;}
cSetEqFormelles &   cAppliApero::SetEq()       {return mSetEq;}

const std::string &   cAppliApero::DC() const {return mDC;}

cInterfChantierNameManipulateur * cAppliApero::ICNM()
{
   return mICNM;
}

const std::string & cAppliApero::SymbPack0() const
{
  return mSymbPack0;
}


void cAppliApero::CheckInit(const cLiaisonsInit * aLI,cPoseCam * aPC)
{
    std::string aNC = aPC->Name();
    cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale(aLI->IdBD(),aNC);

    if (! anOLM) return;

     anOLM->CheckInit();


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
