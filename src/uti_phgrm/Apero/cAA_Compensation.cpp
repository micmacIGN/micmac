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
#include "Apero.h"


static bool   mPhaseContrainte = false;

//  AJOUT DES OBSERVATIONS

void cAppliApero::AddStatCam(cGenPoseCam * aCam,double aRes,double aPerc)
{
   if  (aRes>mWorstRes)
   {
       mWorstRes = aRes;
       mPoseWorstRes  = aCam;
   }
   if (aPerc<mWorstPerc)
   {
        mWorstPerc = aPerc;
        mPoseWorstPerc = aCam;
   }
}

cXmlSauvExportAperoOneIter & cAppliApero::CurXmlE(bool SVP)
{
    if (mXMLExport.Iters().empty())
    {
        ELISE_ASSERT(SVP,"cAppliApero::CurXmlE");
       mXMLExport.Iters().push_back(cXmlSauvExportAperoOneIter());
    }
    return mXMLExport.Iters().back();
}

void cAppliApero::AddObservations
     (
          const cSectionObservations & anSO,
          bool IsLastIter,
          cStatObs & aSO
     )
{
   mWorstRes = -1;
   mWorstPerc = 1e10;
   mPoseWorstRes = 0;
   mPoseWorstPerc = 0;

   cXmlSauvExportAperoOneIter aXmlE;
   aXmlE.NumIter() = mNbIterDone;
   aXmlE.NumEtape() = mNbEtape;
   mXMLExport.Iters().push_back(aXmlE);


   if (IsLastIter && anSO.TxtRapDetaille().IsInit())
   {
      InitRapportDetaille(anSO.TxtRapDetaille().Val());
   }
   else
   {
        mFpRT = 0;
   }

//  int aNbIter= MPD MM() ? 1 : 1;  // Completement artificiel, pour tester resultat incertitudes
int aNbIter= 1 ;  // Completement artificiel, pour tester resultat incertitudes
for (int aK=0 ; aK<aNbIter; aK++)
{
   if (aNbIter!=1) std::cout << "ITERRRRRRRR AddObs=" << aK << "\n";
   // On les mets avant pour que AddLevenbergMarkard sache de manier precise si le centre a
   // ete fixe sur CETTE iteration
   {
       //  MajAddCoeffMatrix();
       //  if (NumIterDebug())  MessageDebug("Avant Centre");
       AddObservationsCentres(anSO.ObsCentrePDV(),IsLastIter,aSO);
   }

   {
       for (const auto & aOCIP : anSO.ObsCenterInPlane())
       {
           AddObservationsPlane(*GetDOPOfName(aOCIP.Id()));
       }
   }

   {
        AddObservationsRelGPS(anSO.ObsRelGPS());
   }

   {
      // MajAddCoeffMatrix();
      // if (NumIterDebug())  MessageDebug("Avant LVM");

      AddLevenbergMarkard(aSO);

      // if (NumIterDebug())  MessageDebug("Apres LVM");
   }



   {
       //  MajAddCoeffMatrix();
       //  if (NumIterDebug())  MessageDebug("Avant ApF");

       AddObservationsAppuisFlottants(anSO.ObsAppuisFlottant(),IsLastIter,aSO);
   }

    // ANCIEN AddObservationsCentres

   {
       //  MajAddCoeffMatrix();
       //  if (NumIterDebug())  MessageDebug("Avant Appuis");

       AddObservationsAppuis(anSO.ObsAppuis(),IsLastIter,aSO);
   }

   {
       //  MajAddCoeffMatrix();
       //  if (NumIterDebug())  MessageDebug("Avant Tie-P");

       AddObservationsLiaisons(anSO.ObsLiaisons(),IsLastIter,aSO);
   }

   {
          AddObservationsRigidBlockCam(anSO.ObsBlockCamRig(),IsLastIter,aSO);
   }

   {
          if (anSO.ContrCamConseq().IsInit())
             AddObservationsCamConseq(anSO.ContrCamConseq().Val());
   }

   {
       //  MajAddCoeffMatrix();
       //  if (NumIterDebug())  MessageDebug("Avant RigGrp");

       AddObservationsRigidGrp(anSO.ObsRigidGrpImage(),IsLastIter,aSO);
   }

   {
      for (const auto & aPose : mVecPose)
      {
          aPose->UseRappelOnPose();
      }
   }

   {
       AddObservationsContrCamGenInc(anSO.ContrCamGenInc(),IsLastIter,aSO);
   }
}

   MajAddCoeffMatrix();
   if (NumIterDebug())  MessageDebug("Fin iter Obs");

   if (mFpRT)
   {
       ElFclose(mFpRT);
       mFpRT = 0;
   }
}


void cAppliApero::AddObservationsBaseGpsInit()
{

   for (auto & aPair : mDicoOffGPS)
   {

      cAperoOffsetGPS * anOffs = aPair.second;
      const cGpsOffset &  aGO = anOffs->ParamCreate();
      if (aGO.Inc().IsInit())
      {
          cBaseGPS * aBG =   anOffs->BaseUnk();

          Pt3dr aPInc = aGO.Inc().Val();
          double aTab[3];
          aPInc.to_tab(aTab);
           // Pt3dr aPInc(1e6,1e6,1e6);
          for (int aK=0 ; aK< 3 ; aK++)
          {
              if (aTab[aK] > 0)
              {
                 if (! mPhaseContrainte)
                 {
                      cMultiContEQF  aRes;
                      aBG->AddFoncRappInit(aRes,aK,aK+1,1);
                      mSetEq.AddContrainte(aRes,false,ElSquare(1/aTab[aK]));
                 }
              }
              // Convention "<0" ~ a contrainte stricte
              else if (aTab[aK]<0)
              {
                 if (mPhaseContrainte)
                 {
                     cMultiContEQF  aRes;
                     aBG->AddFoncRappInit(aRes,aK,aK+1,-1);
                     mSetEq.AddContrainte(aRes,true,-1);
                 }
              }

          }
       }
    }
}


void cAppliApero::AddObservationsRigidBlockCam
     (
         const std::list<cObsBlockCamRig> & anOBCR,
         bool IsLastIter,
         cStatObs & aSO
     )
{
    for 
    (
       std::list<cObsBlockCamRig>::const_iterator itO=anOBCR.begin();
       itO !=anOBCR.end();
       itO++
    )
    {
         AddObservationsRigidBlockCam(*itO,IsLastIter,aSO);
    }

}


void cAppliApero::AddObservationsRigidGrp
     (
         const std::list<cObsRigidGrpImage> & aLG,
         bool IsLastIter,
         cStatObs & aSO
     )
{
    for (std::list<cObsRigidGrpImage>::const_iterator itG=aLG.begin(); itG!=aLG.end() ; itG++)
    {
         AddObservationsRigidGrp(*itG,IsLastIter,aSO);
    }
}


void cAppliApero::AddObservationsAppuis(const std::list<cObsAppuis> & aL,bool IsLastIter,cStatObs & aSO)
{
   // La templatisation complique le passage a la "sous-traitance", donc on
   // gere a la main les iterations
   for (std::list<cObsAppuis>::const_iterator itOA= aL.begin(); itOA!=aL.end() ; itOA++)
   {
      tPackAppuis * aPack = GetEntreeNonVide(mDicoAppuis,itOA->NameRef(),"AddObservationsApp"); 
      std::list<cObserv1Im<cTypeEnglob_Appuis> *> & aLobs = aPack->LObs();

      double aSPds = 0;
      double aSResPds = 0;

      double aSRes =0;
      double aNB =0;

      std::vector<cRes1OnsAppui> aVResAp;
      std::vector<cRes1OnsAppui> *pVResAp=0;
      
      const cRapportObsAppui * pRAO=0;
      if (itOA->RapportObsAppui().IsInit())
      {
           const cRapportObsAppui & aROA = itOA->RapportObsAppui().Val();
           if (IsLastIter || (!aROA.OnlyLastIter().Val()))
           {
              pRAO = & aROA;
              pVResAp = &aVResAp;
           }
      }

      for
      (
           std::list<cObserv1Im<cTypeEnglob_Appuis> *>::iterator itAp=aLobs.begin();
	   itAp!= aLobs.end();
	   itAp++
      )
      {
          if ((*itAp)->mPose->RotIsInit())
          {
             double aRes = AddAppuisOnePose(*itOA,*itAp,pVResAp,aSO,aSResPds,aSPds);
	     aNB ++;
	     aSRes += aRes;
          }
      }
      if ((aNB  >0)  && (int(itOA->Pond().Show().Val()) >= int(eNSM_Iter)))
      {
          COUT()  << "| | " << " RESIDU APPUIS MOYENS , Non Pond :"  
	            << sqrt(aSRes/aNB ) 
                    << " Pond " << sqrt(aSResPds/aSPds)
                    << " pour " << itOA->NameRef() << "\n";
      }
      if (pVResAp)
      {
          DoRapportAppuis(*itOA,*pRAO,*pVResAp);
      }
   }
}

void cAppliApero::AddObservationsCamConseq(const cContrCamConseq &  aCCC)
{

    for (tDiCal::iterator itC=mDicoCalib.begin() ; itC!=mDicoCalib.end(); itC++)
    {
        double aRes = itC->second->PIF().AddObsRegulConseq(aCCC.NbGrid(),aCCC.SigmaPix());
        if (aRes >=0)
        {
            std::cout << " ContCamConseq= " << aRes << " for " << itC->first << "\n";
        }
    }
}

void cAppliApero::AddObservationsCentres(const std::list<cObsCentrePDV> & aL,bool IsLastIter,cStatObs & aSO)
{
   for (std::list<cObsCentrePDV>::const_iterator itOC= aL.begin(); itOC!=aL.end() ; itOC++)
   {
        const cObsCentrePDV  & anObs = *itOC;
        cPonderateur aPdsPlani(anObs.Pond(),NbRotInit());
        cPonderateur aPdsAlti(anObs.PondAlti().ValWithDef(anObs.Pond()),NbRotInit());

        for (int aKPose=0 ; aKPose<int(mVecPose.size()) ; aKPose++)
        {
            cPoseCam * aPC = mVecPose[aKPose];

            if (aPC->RotIsInit() && aPC->DoAddObsCentre(anObs))
            {
                 aPC->AddObsCentre(anObs,aPdsPlani,aPdsAlti,aSO);
            }
        }
   }
}


void cAppliApero::AddObservationsLiaisons(const std::list<cObsLiaisons> & aL,bool IsLastIter,cStatObs & aSO)
{
   if (mFpRT)
   {
       fprintf(mFpRT,"\n*Liaisons\n");
       fprintf(mFpRT,"// Xcomp Ycomp Zcomp Pds\n");
       fprintf(mFpRT,"// NomImage XIm YIm ResXIm ResYIm  RayX RayY RayZ\n");
   }


   for (std::list<cObsLiaisons>::const_iterator itOL= aL.begin(); itOL!=aL.end() ; itOL++)
   {
      const std::string & anId = itOL->NameRef();
      if (!CDNP_InavlideUse_StdLiaison(anId))
      {
          cRapOnZ * aRAZ = 0;
          if (itOL->RappelOnZ().IsInit())
          {
              const cRappelOnZ & aRaz = itOL->RappelOnZ().Val();
              double anI = aRaz.IncC();
              aRAZ = new cRapOnZ(aRaz.Z(),anI,aRaz.IncE().ValWithDef(anI),aRaz.LayerMasq().ValWithDef(""),aRaz.KeyGrpApply().ValWithDef(""));
          }

          cPackObsLiaison * aPackL = GetEntreeNonVide(mDicoLiaisons,anId,"AddObservationsLiaisons"); 

          aPackL->AddObs(itOL->Pond(),itOL->PondSurf().PtrVal(),aSO,aRAZ);
      }
      CDNP_Compense(anId,*itOL);
      // delete aRAZ;
   }
}

void cAppliApero::AddObservationsAppuisFlottants(const std::list<cObsAppuisFlottant> & aL,bool IsLastIter,cStatObs & aSO)
{
   if (mFpRT)
   {
       fprintf(mFpRT,"\n*Appuis\n");
       fprintf(mFpRT,"// Xter Yter ZTer Xcomp Ycomp Zcomp  Pds\n");
       fprintf(mFpRT,"// NomImage XIm YIm ResXIm ResYIm  RayX RayY RayZ\n");
   }
   for (std::list<cObsAppuisFlottant>::const_iterator itOAF= aL.begin(); itOAF!=aL.end() ; itOAF++)
   {
      cBdAppuisFlottant * aBAF =  GetEntreeNonVide(mDicPF,itOAF->NameRef(),"AddObservationsAppuisFlottants");

      mNbPtsFlot = 0;
      mMaxDistFlot=0.0;
      mSomDistFlot=0.0;
      mSomEcPtsFlot = Pt3dr(0,0,0);
      mSomAbsEcPtsFlot = Pt3dr(0,0,0);
      mSomRmsEcPtsFlot = Pt3dr(0,0,0);
      mMaxAbsEcPtsFlot = Pt3dr(0,0,0);
      aBAF->AddObs(*itOAF,aSO);

      if (mNbPtsFlot)
      {
          Pt3dr  aMeanEc = mSomEcPtsFlot/mNbPtsFlot;
          double aMeanEcXY = mSomDistXYFlot/mNbPtsFlot;
          double aMeanEcXYZ = mSomDistFlot/mNbPtsFlot;
          Pt3dr  aRMS2 = mSomRmsEcPtsFlot/mNbPtsFlot;
          Pt3dr  aRMS = Pt3dr(sqrt(aRMS2.x),sqrt(aRMS2.y),sqrt(aRMS2.z));
          double aRMSXY = euclid(Pt2dr(aRMS.x,aRMS.y));
          double aCoef = sqrt(mNbPtsFlot/(mNbPtsFlot-1)); //unbiased STD coef
          Pt3dr  aSTDEc = Pt3dr(sqrt(aRMS2.x-pow(aMeanEc.x,2))*aCoef,
                                sqrt(aRMS2.y-pow(aMeanEc.y,2))*aCoef,
                                sqrt(aRMS2.z-pow(aMeanEc.z,2))*aCoef);
          double aSTDEcXY = sqrt(aRMS2.x+aRMS2.y-pow(aMeanEcXY,2))*aCoef;
          double aSTDEcXYZ = sqrt(aRMS2.x+aRMS2.y+aRMS2.z-pow(aMeanEcXYZ,2))*aCoef;
          std::cout << "=== GCP STAT ===  Dist,  Moy="<< (aMeanEcXYZ) << " Max=" << mMaxDistFlot << "\n";
          std::cout <<  "[X,Y,Z],      MoyAbs=" << (mSomAbsEcPtsFlot/mNbPtsFlot) << " Max=" << mMaxAbsEcPtsFlot << " Mean=" << aMeanEc << " STD=" << aSTDEc << " Rms=" << aRMS << "\n";
          std::cout <<  "[Plani,alti], Mean=[" << aMeanEcXY << "," << aMeanEc.z << "] STD=[" << aSTDEcXY << "," << aSTDEc.z << "] RMS=[" << aRMSXY << "," << aRMS.z << "]\n";
          std::cout <<  "Norm,         Mean=" << aMeanEcXYZ << " STD= " << aSTDEcXYZ << " RMS=" << euclid(aRMS) << "\n";
      }
   }
}

void cAppliApero::AddEcPtsFlot(const Pt3dr & anEc)
{
   mNbPtsFlot++;
   double aD = euclid(anEc);
   double aDXY = euclid(Pt2dr(anEc.x,anEc.y));
   mMaxDistFlot= ElMax(mMaxDistFlot,aD);
   mSomDistFlot += aD;
   mSomDistXYFlot += aDXY;
   mSomEcPtsFlot = anEc + mSomEcPtsFlot;
   Pt3dr aEcAbs = Pt3dr(ElAbs(anEc.x),ElAbs(anEc.y),ElAbs(anEc.z));
   mSomAbsEcPtsFlot = aEcAbs + mSomAbsEcPtsFlot;
   Pt3dr aEcSquare = Pt3dr(pow(anEc.x,2.0),pow(anEc.y,2.0),pow(anEc.z,2.0));
   mSomRmsEcPtsFlot = aEcSquare + mSomRmsEcPtsFlot;
   mMaxAbsEcPtsFlot = Sup(mMaxAbsEcPtsFlot,aEcAbs);
}



//    ACTIVATION  DES CONTRAINTES

void cAppliApero::ActiveContraintes(bool Stricte)
{
    // Contraintes sur les calibrations
    for (tDiCal::iterator itD=mDicoCalib.begin() ; itD!=mDicoCalib.end(); itD++)
    {
        itD->second->ActiveContrainte(Stricte);
    }

    // Contraintes sur les poses
    for (tDiPo::iterator itD=mDicoPose.begin() ; itD!=mDicoPose.end(); itD++)
    {
        itD->second->ActiveContrainte(Stricte);
    }

    BlocContraintes(Stricte);
}

#define TheUninitScoreLambda -1e30

double cAppliApero::ScoreLambda(double aLambda)
{
    if (aLambda==0.0) 
    {
       return mScoreLambda0;
    }

    if ((aLambda==1.0)  && (mScoreLambda1 != TheUninitScoreLambda))
    {
       return mScoreLambda1;
    }

    cStatObs  aSO(false);
    mSetEq.ResetUpdate(aLambda);
    AddObservations(mCurEC->SectionObservations(),mIsLastIter,aSO);
    return aSO.SomErPond();
}

double  cAppliApero::NRF1v(double aLambda)
{
   return ScoreLambda(aLambda);
}

bool cAppliApero::NROptF1vContinue() const
{
   if (NROptF1vND::mNbIter <=mParam.SeuilBas_CDD().Val())
      return true;

   if (NROptF1vND::mNbIter > mParam.SeuilHaut_CDD().Val())
      return false;

   return x0 == 0.0;
}

      
       // UNE ITERATION

class cCmpNbNNPose
{
    public :
      bool operator ()(cPoseCam * aP1,cPoseCam * aP2) const
      {
          return aP1->NbPtsMulNN() < aP2->NbPtsMulNN();
      }
};


const std::string & TheNameMatrCorrel()
{
    static std::string TMV="Sensib-MatriceCorrel.tif";
    return TMV;
}
const std::string & TheNameMatrCov()
{
    static std::string TMC="Sensib-MatriceCov.tif";
    return TMC;
}
const std::string & TheNameFileTxtConvName()
{
    static std::string TMC="Sensib-ConvName.txt";
    return TMC;
}

const std::string & TheNameFileXmlConvNameIm()
{
    static std::string TMC="Sensib-ConvName-Im.xml";
    return TMC;
}

std::map<std::string,std::string>  LecSensibName(const std::string & aNameFile,const std::string & aPref)
{
    std::map<std::string,std::string> aRes;
    ELISE_fp  aFile(aNameFile.c_str(),ELISE_fp::READ);

    std::string aPat = std::string(" (") + aPref + ".*) => (.*)";
    cElRegex anAutom(aPat,10);
    bool endof=false;
    std::string aLine;

    while (!endof)
    {
        if (aFile.fgets(aLine,endof))
        {
             if (anAutom.Match(aLine))
             {  
                std::string anId = anAutom.KIemeExprPar(1);
                std::string aVal = anAutom.KIemeExprPar(2);
                aRes[anId] = aVal;
             }
        }
    }
    return aRes;
}
 
std::map<std::string,std::vector<cSensibDateOneInc> >
    LecSensibDicIm(const std::string & aNameConv,const std::string & aNameXml)
{
    std::map<std::string,std::vector<cSensibDateOneInc> > aRes;

    std::map<std::string,std::string> aConv =   LecSensibName ( aNameConv,"Ima");
    cXmlNameSensibs     aXmlSN = StdGetFromAp(aNameXml,XmlNameSensibs);

    for (const auto & aS1I : aXmlSN.SensibDateOneInc())
    {
        auto anIter = aConv.find(aS1I.NameBloc());
        if (anIter != aConv.end())
        {
             aRes[anIter->second].push_back(aS1I);
        }
    }

    return aRes;
}

const cSensibDateOneInc * GetSensib(const std::vector<cSensibDateOneInc> & aVec,const std::string & anId,bool SVP=false)
{
    auto anIter = std::find_if
                  (
                      aVec.begin(),
                      aVec.end(),
                      [anId](const cSensibDateOneInc & aS1) {return aS1.NameInc() == anId;}
                  );
     if (anIter == aVec.end())
     {
         ELISE_ASSERT(SVP,"Cannot find in GetSensib");
         return nullptr;
     }

     return  &(*anIter);
}


std::map<std::string,std::pair<Pt3dr,Pt3dr>> GetSCenterOPK(const std::string & aNameConv,const std::string & aNameXml)
{
     std::map<std::string,std::pair<Pt3dr,Pt3dr> > aRes;

     for (const auto & aVec : LecSensibDicIm(aNameConv,aNameXml))
     {
          Pt3dr aSCenter
                (
                    GetSensib(aVec.second,"Cx")->SensibParamInv(),
                    GetSensib(aVec.second,"Cy")->SensibParamInv(),
                    GetSensib(aVec.second,"Cz")->SensibParamInv()
                 );
          Pt3dr aSOPK
                (
                    GetSensib(aVec.second,"T12")->SensibParamInv(),
                    GetSensib(aVec.second,"T02")->SensibParamInv(),
                    GetSensib(aVec.second,"T01")->SensibParamInv()
                 );
          if (0)
          {
               std::cout << "SSSss " << aVec.first << aSCenter << aSOPK << "\n";
          }
          aRes[aVec.first] = std::pair<Pt3dr,Pt3dr>(aSCenter,aSOPK);
     }

     return aRes;
}
std::map<std::string,std::pair<Pt3dr,Pt3dr>>    StdGetSCenterOPK(const std::string &  aDir)
{
   return GetSCenterOPK(aDir+"/Sensib-ConvName.txt",aDir+"/Sensib-Data.dmp");
}





std::string  TheNameFileExpSens(bool Bin)
{
    return "Sensib-Data" + std::string(Bin ? ".dmp" : ".xml");
}
const std::string & TheNameMatrCorrelInv()
{
    static std::string TMV="Sensib-MatriceCorrelInv.tif";
    return TMV;
}
const std::string & TheNameMatrCorrelDir()
{
    static std::string TMV="Sensib-MatriceCorrelDir.tif";
    return TMV;
}

bool IsMatriceExportBundle(const std::string & aNameIm)
{
   return    (aNameIm==TheNameMatrCorrel()) 
          || (aNameIm==TheNameMatrCov()) 
          || (aNameIm==TheNameMatrCorrelInv())
          || (aNameIm==TheNameMatrCorrelDir());
}

Fonc_Num Correl(Fonc_Num Cov,Fonc_Num Var1, Fonc_Num Var2)
{
   return Max(-1,Min(1,Cov/sqrt(Max(1e-40,Var1*Var2))));
}




void cAppliApero::OneIterationCompensation(const cIterationsCompensation & anIter,const cEtapeCompensation & anEC,bool IsLast)
{
    mPhaseContrainte = true;
    if (mSqueezeDOCOAC)
    {
        ELISE_ASSERT(mSqueezeDOCOAC==1,"Multiple mSqueezeDOCOAC");
        cStatObs  aSO(false);
        mSqueezeDOCOAC++;
        AddObservationsAppuisFlottants(anEC.SectionObservations().ObsAppuisFlottant(),IsLast,aSO);
/*
std::cout << "AOAF : NonO =================================================\n";
        // cSectionObservation
std::cout << "DONNNNE AOAF : NonO =================================================\n";
*/

        // cwanEC.AddObservations().anSO.ObsAppuisFlottant(),IsLast,aSO);
        return;
    }

    mCurEC = & anEC;
    mIsLastIter = IsLast;
    for (tDiPo::iterator itD=mDicoPose.begin() ; itD!=mDicoPose.end(); itD++)
    {
        itD->second->InitAvantCompens();
    }

    for (tDiCal::iterator itC=mDicoCalib.begin() ; itC!=mDicoCalib.end(); itC++)
    {
        itC->second->InitAvantCompens();
    }

    AddObservationsBaseGpsInit();
    ActiveContraintes(true);

    mSetEq.SetPhaseEquation();
    ActiveContraintes(false);
    mPhaseContrainte=false;

    AddObservationsBaseGpsInit();


    for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
    {
       mVecPose[aKP]->SetNbPtsMulNN(0);
    }

    cStatObs  aSO(true);
    AddObservations(anEC.SectionObservations(),IsLast,aSO);
    mStatLastIter = aSO;

    // Eventuel affichage des points des images a peu de liaison
    if (mCurPbLiaison && mCurPbLiaison->Actif().Val())
    {
        std::vector<cPoseCam *> aVP = mVecPose;
        cCmpNbNNPose aCmp;
        std::sort(aVP.begin(),aVP.end(),aCmp);
        bool Got=false;
        for (int aK=int(aVP.size())-1 ; aK>=0 ; aK--)
        {
            if (aVP[aK]->NbPtsMulNN() <mCurPbLiaison->NbMinPtsMul().Val())
            {
                Got = true;
                std::cout << " Pose : " << aVP[aK]->Name()
                          << " PMUL : "  << aVP[aK]->NbPtsMulNN()
                          << "\n";
            }
        }
        if (Got && mCurPbLiaison->GetCharOnPb().Val())
        {
           std::cout << "Enter to continue \n";
           getchar();
        }
    }

    for (tDiCal::iterator itC=mDicoCalib.begin() ; itC!=mDicoCalib.end(); itC++)
    {
        itC->second->PostFinCompens();
    }



    // std::cout  << "=========SOM-POND-ERR " << aSO.SomErPond() << "\n";
    // mSetEq.SolveResetUpdate(aSO.SomErPond());
    // mSetEq.Solve(aSO.SomErPond());

    Im1D_REAL4 aMVar(1);
    Im1D_U_INT1 isCstr(1);
    cXmlNameSensibs aXmlS;
    std::string aPrefESPA;
// IsCstrUniv   bool IsCstrUniv(int anX,double & aVal);

    if (mESPA)
    {
       cGenSysSurResol * aSys = mSetEq.Sys();   
       int aNbV = aSys->NbVar();
       isCstr = Im1D_U_INT1(aNbV);
       for (int aKx=0 ; aKx<aNbV ; aKx++)
       {
           double aVal;
           isCstr.data()[aKx] =  aSys->IsCstrUniv(aKx,aVal);
           //  std::cout << "ISCTRE " << int(isCstr.data()[aKx]) << " " << aKx << "\n";
       }

       aPrefESPA =  DC()+mESPA->Dir();
       ELISE_fp::MkDir(aPrefESPA);
       AllocateurDInconnues &  anAlloc = mSetEq.Alloc();
       std::string aNameConv =  aPrefESPA + TheNameFileTxtConvName();

       ofstream  aStdConvTxt (aNameConv.c_str());
       cSauvegardeNamedRel  aRelIm;
       if (! aStdConvTxt.is_open())
       {
		    std::cout << "FILE=" << aNameConv << "\n";
            ELISE_ASSERT(aStdConvTxt.is_open(),"Open file txt in Analysis Bundle");
       }

// const char* filename, ios_base::openmode mode = ios_base::out);
       //FILE * aFConvTxt = FopenNN(aNameConv.c_str(),"w","Export Sensibilty Analysis in Bundle");

       // fprintf(aFile,"###############  Intrinseque Calibration Correspondance ##############\n")
       aStdConvTxt << "##############  Intrinseque Calibration Correspondance ##############\n";
       for (int aK=0 ; aK<int(mNamesIdCalib.size()) ; aK++)
       {
          aStdConvTxt<< " " << IdOfCalib(aK) << " => " << mNamesIdCalib[aK]  << "\n";
       }
       aStdConvTxt << "##############  Extrinseque Calibration Correspondance ##############\n";
       for (int aK=0 ; aK<int(mNamesIdIm.size()) ; aK++)
       {
          aStdConvTxt<< " " << IdOfIma(aK) << " => " << mNamesIdIm[aK]  << "\n";
          aRelIm.Cple().push_back(cCpleString(IdOfIma(aK), mNamesIdIm[aK]));
       }

       Im2D_REAL4 aMCov(aNbV,aNbV);
       aMVar = Im1D_REAL4(aNbV);
       REAL4 ** aDC = aMCov.data();
       REAL4 *  aDV = aMVar.data();


       for (int aKx=0 ; aKx<aNbV ; aKx++)
       {
            // std::cout << "NAMESSSS " << anAlloc.NamesBlocInc(aKx) << " => " << anAlloc.NamesInc(aKx) << "\n";
            for (int aKy=0 ; aKy<=aKx ; aKy++)
            {
                aDC[aKy][aKx] = aDC[aKx][aKy] = aSys->GetElemQuad(aKx,aKy);
            }
            aDV[aKx] =  aDC[aKx][aKx];
            cSensibDateOneInc aSDOI;
            aSDOI.NameBloc() = anAlloc.NamesBlocInc(aKx);
            aSDOI.NameInc()  = anAlloc.NamesInc(aKx);
            aSDOI.SensibParamInv()  = 0.0;
            aSDOI.SensibParamDir()  = 0.0;
            aXmlS.SensibDateOneInc().push_back(aSDOI);
       }
       // getchar();

       Tiff_Im::CreateFromFonc
       (
            aPrefESPA + TheNameMatrCorrelDir(),
            Pt2di(aNbV,aNbV),
            // aMCov.in()/sqrt(Max(1e-60,aMVar.in()[FX]*aMVar.in()[FY])),
            Correl(aMCov.in(),aMVar.in()[FX],aMVar.in()[FY]),
            GenIm::real4
       );
       Tiff_Im::CreateFromIm(aMCov, aPrefESPA + TheNameMatrCov());

       //fclose(aFConvTxt);
       aStdConvTxt.close();
       MakeFileXML(aRelIm,aPrefESPA+ TheNameFileXmlConvNameIm());
    }

    bool  ExportMMF = mParam.SectionChantier().ExportMatrixMarket().Val() ;
    FILE * aFileEMMF=nullptr;
    // Export to Matrix Market format
    if (ExportMMF)
    {
        cGenSysSurResol * aSys = mSetEq.Sys();   
        int aNbV = aSys->NbVar();
        int aNbNN = 0;
        int aNbTot = 0;
        bool DoSym = true; // If true export 2 way else only for J>=I
        for (int aIter=0 ; aIter<2 ; aIter++)
        {
            if (aIter==1)
            {
                aFileEMMF = FopenNN("Test_SPD.mtx","w","Export Matrix MarketFormat");
                fprintf(aFileEMMF,"%d %d %d\n",aNbV,aNbV,aNbNN);
            }
            for (int aI=0 ; aI<aNbV ; aI++)
            {
                 int aJ0 = (DoSym ? 0 : aI);
                 for (int aJ=aJ0 ; aJ<aNbV ; aJ++)
                 {
                     double aV  = aSys->GetElemQuad(aI,aJ);
                     bool IsNull = (aV==0);
                     if (aIter==0)
                     {
                         aNbTot++;
                         if (! IsNull)
                            aNbNN++;
                     }
                     else
                     {
                         if (! IsNull)
                         {
                             fprintf(aFileEMMF,"%d %d %10.10E\n",aI+1,aJ+1,aV);  // !!! => Fuck Fortran index convention  !!!
                             // fprintf(aFP,"%d %d %lf\n",aI,aJ,aV);
                          }
                     }
                 }
            }
        }
        std::cout << "===========  EXPORT MATRIX MARKET FORMAT =========\n";
        std::cout << "  Densite NN=" << (double(aNbNN) / double(aNbTot)) << " NbVar=" << aNbV << "\n";
    }

    ElTimer aChronoSolve;
    mSetEq.Solve(aSO.SomErPond(),(bool *)0);

    if (ExportMMF)
    {
       fprintf(aFileEMMF,"%% MicMac Cholesky time = %lf\n",aChronoSolve.uval());
       fclose(aFileEMMF);
    }

    if (mESPA)
    {
        cGenSysSurResol * aSys = mSetEq.Sys();   

        if (aSys->InverseIsComputedAfterSolve())
        {
            int aNbV = aSys->NbVar();
            double aRes1 = aSys->ResiduAfterSol();
            double aRes2 = aSys->R2Pond() / aSys->Redundancy();
            for (int aK=0 ; aK<aNbV ; aK++)
            {
                // std::cout << "GGGGG "<< aSys->GetElemInverseQuad(aK,aK) << " " << aMVar.data()[aK] << "\n";
                // double aVal = aSys->GetElemInverseQuad(aK,aK);
                // aVal *= aMVar.data()[aK];
                if (0) std::cout << "=============== RESSSS " << aRes2 << " " << aRes1 << "\n";
                aXmlS.SensibDateOneInc()[aK].SensibParamInv() = sqrt(aRes2*aSys->GetElemInverseQuad(aK,aK));
                aXmlS.SensibDateOneInc()[aK].SensibParamDir() = sqrt(aRes2/aMVar.data()[aK]);
                aXmlS.SensibDateOneInc()[aK].SensibParamVar() = sqrt(aSys->Variance(aK) / aSys->Redundancy());

//   std::cout << " TEST-FUV " << aXmlS.SensibDateOneInc()[aK].SensibParamVar() / sqrt(aSys->GetElemInverseQuad(aK,aK)) << "\n";
            }
            Im2D_REAL8 aMCov(aNbV,aNbV);
            REAL8 ** aDC = aMCov.data();
            Im2D_REAL4 aMCorInv(aNbV,aNbV);
            REAL4 ** aDCI = aMCorInv.data();
            Im1D_REAL4 aMVarI = Im1D_REAL4(aNbV);
            REAL4 *  aDMI = aMVarI.data();
            for (int aKx=0 ; aKx<aNbV ; aKx++)
            {
                for (int aKy=0 ; aKy<=aKx ; aKy++)
                {
                    aDCI[aKy][aKx] = aDCI[aKx][aKy] = aSys->GetElemInverseQuad(aKx,aKy);
                    aDC[aKy][aKx]  = aDC[aKx][aKy] = *(aSys->CoVariance(aKx,aKy));

                }
                aDMI[aKx] = aDCI[aKx][aKx];
            }
            for (int aKx=0 ; aKx<aNbV ; aKx++)
            {
                for (int aKy=0 ; aKy<=aKx ; aKy++)
                {
                    // double aCor = aDC[aKy][aKx] / sqrt(ElMax(1e-10,(double)aDC[aKx][aKx] *aDC[aKy][aKy] ));
                }
            }
            Tiff_Im::CreateFromFonc
            (
                aPrefESPA + TheNameMatrCorrelInv(),
                Pt2di(aNbV,aNbV),
                // aMCorInv.in()/sqrt(Max(1e-60,aMVarI.in()[FX]*aMVarI.in()[FY])),
                Correl(aMCorInv.in(),aMVarI.in()[FX],aMVarI.in()[FY]),
                GenIm::real4
            );
            Tiff_Im::CreateFromFonc
            (
                aPrefESPA + TheNameMatrCorrel(),
                Pt2di(aNbV,aNbV),
                // aMCov.in()/sqrt(Max(1e-60,aMCov.in()[Virgule(FX,FX)]*aMCov.in()[Virgule(FY,FY)])),
                Correl(aMCov.in(),aMCov.in()[Virgule(FX,FX)],aMCov.in()[Virgule(FY,FY)]),
                GenIm::real4
            );
        }
        MakeFileXML(aXmlS,aPrefESPA+TheNameFileExpSens(false));
        MakeFileXML(aXmlS,aPrefESPA+TheNameFileExpSens(true));

        aSys->Show();
    }


    mScoreLambda0 = aSO.SomErPond();

   double aLambdaReset = 1.0;
   eControleDescDic aModeCDD = mParam.ModeControleDescDic().Val();
   if (aModeCDD != eCDD_Jamais)
   {
       mScoreLambda1 = TheUninitScoreLambda;
       mScoreLambda1  =  ScoreLambda(1.0);
       if ((aModeCDD==eCDD_Toujours) || (mScoreLambda1> mScoreLambda0))
       {
           double aVInterm = (mScoreLambda0 < mScoreLambda1) ? 0.38 : 0.62; // Voir Golden
           golden(0.0, aVInterm , 1.0,1e-3,&aLambdaReset);
       }
       std::cout << "LAMBDA MIN = " << aLambdaReset << "\n";
    }


    mSetEq.ResetUpdate(aLambdaReset);
    // mSetEq.SolveResetUpdate(aSO.SomErPond());

    
    for (tDiPo::iterator itD=mDicoPose.begin() ; itD!=mDicoPose.end(); itD++)
    {
        itD->second->Trace();
    }

    for 
    (
        std::list<cXml_EstimateOrientationInitBlockCamera>::const_iterator itE= anIter.EstimateOrientationInitBlockCamera().begin();
        itE != anIter.EstimateOrientationInitBlockCamera().end();
        itE++
    )
    {
       EstimateOIBC(*itE);
    }

    mCptIterCompens ++;

}


//    AJOUT DES CONTRAINTES

void cAppliApero::MAJContrainteCamera(const cContraintesCamerasInc & aC)
{
   // ELISE_ASSERT (aC.TolContrainte().Val()<0,"Ne gere que les contraintes strictes");
   cElRegex  anAutom(aC.PatternNameApply().Val(),10);
   cElRegex * aRef = aC.PatternRefuteur().ValWithDef(0);

   int aNbMatch=0;
   for 
   (
        tDiCal::const_iterator itC = mDicoCalib.begin();
	itC!=  mDicoCalib.end();
	itC++
   )
   {
        if (anAutom.Match(itC->first))
        {
            if ((aRef==0) || (! aRef->Match(itC->first)))
            {
              aNbMatch++;

// std::cout << itC->first   << "  :::  itC->second " << itC->second << "\n";
               itC->second->SetContrainte(aC);
            }
        }
       
   }
   if (aNbMatch==0)
   {
       static bool First = true;
       if (First)
       {
          First = false;
          std::cout << "WARN No Math for ContraintesCamerasInc " << aC.PatternNameApply().Val() << "\n";
          // GetCharOnBrkp();
       }
   }
}


void cAppliApero::MAJContraintePose(const cContraintesPoses & aC)
{
    if (aC.ByPattern().Val())
    {
         cSetName *  aSelector = mICNM->KeyOrPatSelector(aC.NamePose());

         int aNb = 0;

        cSetName * aRefut = 0;
        if (aC.PatternRefuteur().IsInit())
            aRefut = mICNM->KeyOrPatSelector(aC.PatternRefuteur().Val());

        for (int aKP=0; aKP<int(mVecPose.size()) ; aKP++)
        {
            std::string aName = mVecPose[aKP]->Name();
            if (aSelector->IsSetIn(aName))
            {
               if ( (!aRefut) ||(!aRefut->IsSetIn(aName)))
               {
                  mVecPose[aKP]->SetContrainte(aC);
                  aNb++;
               }
            }
        }
        if (aNb==0)
        {
            // std::cout << "WWWWWWWWaarrrrnnnnnnn :  contrainte pose By Pattern, aucun match\n";
        }
    }
    else
    {
         cPoseCam *  aPose =  PoseFromName(aC.NamePose());
         aPose->SetContrainte(aC);
    }
}




typedef std::list<cContraintesCamerasInc> tLCCI;
typedef std::list<cContraintesPoses> tLCCP;

void cAppliApero::InspectCalibs()
{
   for 
   (
        tDiCal::const_iterator itC = mDicoCalib.begin();
	itC!=  mDicoCalib.end();
	itC++
   )
   {
        itC->second->Inspect();
   }
}




void  cAppliApero::MAJContraintes(const cSectionContraintes & aSC)
{
    // ----------------------------------
    // On initialise les contraintes
    // ----------------------------------


    // Contraintes sur les calibrations
    {
       const tLCCI aLC=aSC.ContraintesCamerasInc();
       for (tLCCI::const_iterator anIC=aLC.begin(); anIC!=aLC.end() ; anIC++)
       {
            MAJContrainteCamera(*anIC);
       }
    }
    // Contraintes sur les poses
    {
       const tLCCP aLCP=aSC.ContraintesPoses();
       for (tLCCP::const_iterator anICP=aLCP.begin(); anICP!=aLCP.end() ; anICP++)
       {
            MAJContraintePose(*anICP);
       }
    }

    for (tDiPo::iterator itD=mDicoPose.begin() ; itD!=mDicoPose.end(); itD++)
    {
       if (! itD->second->RotIsInit())
       {
           itD->second->SetFigee();
       }
    }
       

}

//    CONTROLE GLOBAL
//


void cAppliApero:: AddResiducentre(const Pt3dr & aP)
{
   mResiduCentre.push_back(aP);
}

void cAppliApero::AddRetard(double aT)
{
   mRetardGpsC.push_back(aT);
}


void  cAppliApero::ShowRetard()
{
   if (mRetardGpsC.empty()) return;

   std::sort(mRetardGpsC.begin(),mRetardGpsC.end());
   int aNb = 10;
   for (int aK=0 ; aK<=aNb ; aK++)
   {
      double aPerc = (aK*100.0) / aNb;
      std::cout << " %:" << aPerc << " RETARD " << ValPercentile(mRetardGpsC,aPerc) << "\n";
   }
}


void  cAppliApero::ShowResiduCentre()
{
   if (mResiduCentre.size() ==0) return;
   double aResiduMin = 1e30;
   int    aKMin      = -1;

   for (int aKTest=0 ; aKTest <int(mResiduCentre.size()) ; aKTest++)
   {
        double aSom = 0.0;
        Pt3dr  aCTest = mResiduCentre[aKTest];
        for (int aK1=0 ; aK1 <int(mResiduCentre.size()) ; aK1++)
        {
             aSom += euclid(aCTest-mResiduCentre[aK1]);
        }
        if (aSom<aResiduMin)
        {
             aResiduMin = aSom;
             aKMin=aKTest;
        }
   }
   Pt3dr aCMed = mResiduCentre[aKMin];
   std::cout << "CENTRE MEDIAN = " <<  aCMed << "\n";
}

void cAppliApero::DoOneContraintesAndCompens
     (
            const cEtapeCompensation & anEC,
            const cIterationsCompensation &  anIter,
            bool  IsLastIter
     )
{

   ReinitStatCondFaisceau();

   mResiduCentre.clear();
   mRetardGpsC.clear();

   if (! mParam.DoCompensation().Val())
     return;
   for 
   (
      std::list<cSetRayMaxUtileCalib>::const_iterator itS=anIter.SetRayMaxUtileCalib().begin();
      itS!=anIter.SetRayMaxUtileCalib().end();
      itS++
   )
   {
       bool got = false;
       cElRegex anAutom(itS->Name(),10);
       for 
       (
           tDiCal::const_iterator itC=mDicoCalib.begin();
           itC!=mDicoCalib.end();
           itC++
       )
       {
           // if (itC->second && itC->second->CCI().Name()==itS->Name())
           if (itC->second && anAutom.Match(itC->second->CCI().Name()))
           {
                itC->second->SetRMaxU(itS->Ray(),itS->IsRelatifDiag().Val(),itS->ApplyOnlyFE().Val());
                got = true;
           }
       }
      ELISE_ASSERT(got,"No Cam found in SetRayMaxUtileCalib");
  //ELISE_ASSERT(false,"FAIRE MODIF RMAX UTIL POUR CALIB / POSE ");
       // CalibFromName(itS->Name())->SetRMaxU(itS->Ray());
   }

   for
   (
       tDiPo::const_iterator itD=mDicoPose.begin();
       itD!=mDicoPose.end();
       itD++
   )
   {
       cPoseCam * aPC = itD->second;
       aPC->BeforeCompens();
   }

    if (anIter.SectionContraintes().IsInit())
    {
               MAJContraintes(anIter.SectionContraintes().Val());
    }
    OneIterationCompensation(anIter,anEC,IsLastIter);

    ShowResiduCentre();
    ShowRetard();
    if (DebugPbCondFaisceau)
    {
       ShowStatCondFaisceau(true);
    }
}

bool cAppliApero::PIsActif(const Pt2dr & aP) const
{
  return (!mMTAct) || (mMTAct->SelectVal(aP));
}

void cAppliApero::SetSqueezeDOCOAC()
{
   ELISE_ASSERT(mSqueezeDOCOAC<=1,"cAppliApero::SetSqueezeDOCOAC");
   mSqueezeDOCOAC = 1;
}

bool cAppliApero::SqueezeDOCOAC() const
{
   return mSqueezeDOCOAC != 0;
}

void cAppliApero::DoContraintesAndCompens
     (
            const cEtapeCompensation & anEC,
            const cIterationsCompensation &  anIter,
            bool  IsLastIter,
            bool IsLastEtape
     )
{
   mIsLastEtape = IsLastEtape;
   mIsLastEtapeOfLastIter = IsLastIter && IsLastEtape;
/*
   if (mSqueezeDOCOAC)
   {
      ELISE_ASSERT(mSqueezeDOCOAC==1,"Multiple mSqueezeDOCOAC");
      mSqueezeDOCOAC++;
      // AddObservationsAppuisFlottants(anSO.ObsAppuisFlottant(),IsLastIter,aSO);
      return;
   }
*/

   mMTAct = 0;
   if (!anIter.MesureErreurTournante().IsInit())
   {
      DoOneContraintesAndCompens(anEC,anIter,IsLastIter);
      ExportSauvAutom();
      return;
   }
   ELISE_ASSERT(mESPA==0,"Mix Sensib && Erreur Tournante");

  std::cout << "-------------  MESURE ERREUR EXTRAPOLATION  ------------------------\n"; 

  const cMesureErreurTournante & aMET = anIter.MesureErreurTournante().Val();
  int aNbPer = aMET.NbTest().ValWithDef(aMET.Periode()) ;
  int aNbIter = aMET.NbIter().Val();

  ELISE_ASSERT(mMTRes==0,"Multiple mesure tournante");
  mMTAct = new cMTActive(aNbPer);
  mMTRes = new cMTResult;

  for (int aKCur=0 ; aKCur<aNbPer ; aKCur++)
  {
    cStateAllocI aStateVar (mSetEq.Alloc());
     
     mMTAct->SetKCur(aKCur);
     for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
     {
         if (aKIter==(aNbIter-1))
         {
             mMTRes->NewSerie();
             mMTRes->SetActif();
         }
         else
         {
            mMTRes->SetInactif();
         }
         DoOneContraintesAndCompens(anEC,anIter,IsLastIter);
     }
     AddCamsToMTR();

     mSetEq.Alloc().RestoreState(aStateVar);

     std::cout << " Done  " <<  (aKCur+1) << " on " << aNbPer << "\n";
  }

  std::cout << "-------------  ------------------------\n";

  mMTRes->SetInactif();
  delete mMTAct;
  mMTAct = 0;

}



void cAppliApero::TestInteractif(const cTplValGesInit<cTestInteractif> & aTTI,bool Avant)
{
   if (! aTTI.IsInit()) return;
   const cTestInteractif aTI = aTTI.Val();

   if (! (Avant  ? aTI.AvantCompens().Val() : aTI.ApresCompens().Val()))
   {
        return;
   }

   if (aTI.TestF2C2().Val())  
     TestF2C2();

   ResidualStepByStep = aTI.SetStepByStep().Val();

}


// bool ResidualStepByStep = false;

void cAppliApero::TestF2C2()
{

   bool cont = true;
   while (cont)
   {
        std::cout << "Enter Name \n";
        std::string aName;
        std::cin >> aName;
        cPoseCam *  aPC = PoseFromNameSVP(aName);
        if (aPC)
        {
             const CamStenope * aCS = aPC->CurCam();
             Pt2dr aPIm;
             std::cin >> aPIm.x  >> aPIm.y ;
             std::cout << "C2 : " << aCS->F2toC2(aPIm) << aCS->F2AndZtoR3(aPIm,22) << "\n";
        }
   }
}


void  cAppliApero::DoOneEtapeCompensation(const cEtapeCompensation & anEC,bool LastEtape)
{
    delete mMTRes;
    mMTRes = 0;

    InitLVM(mCurSLMGlob,anEC.SLMGlob(),mMulSLMGlob,anEC.MultSLMGlob());
    InitLVM(mCurSLMEtape,anEC.SLMEtape(),mMulSLMEtape,anEC.MultSLMEtape());

    mNbIterDone =0;
    mNbIterTot = 0;

    for (int aK=0 ; aK<int(anEC.IterationsCompensation().size()) ; aK++)
    {
	const cIterationsCompensation &  anIter  = anEC.IterationsCompensation()[aK];
        const cCtrlTimeCompens * aCtrl = anIter.CtrlTimeCompens().PtrVal();
        if (aCtrl)
        {
            mNbIterTot += aCtrl->NbMax()  +1 ;
        }
        else
        {
            mNbIterTot ++;
        }
    }



    for (int aK=0 ; aK<int(anEC.IterationsCompensation().size()) ; aK++)
    {
        bool kIterLast = (aK==((int)anEC.IterationsCompensation().size()-1));
	const cIterationsCompensation &  anIter  = anEC.IterationsCompensation()[aK];

        mESPA =0;
        if (kIterLast && anEC.SectionExport().IsInit())
        {
            mESPA = anEC.SectionExport().Val().ExportSensibParamAero().PtrVal();
            if (mESPA)
               mSetEq.Sys()->SetCalculVariance(true);
        }

        if (anIter.DoIt().Val())
        {
            bool GoOnIter= true;
            int aCptInIter = 0;
            while (GoOnIter)
            {

                TestInteractif(anIter.TestInteractif(),true);

                InitLVM(mCurSLMGlob,anIter.SLMGlob(),mMulSLMGlob,anIter.MultSLMGlob());
                InitLVM(mCurSLMEtape,anIter.SLMEtape(),mMulSLMEtape,anIter.MultSLMEtape());
                InitLVM(mCurSLMIter,anIter.SLMIter(),mMulSLMIter,anIter.MultSLMIter());



                if (anIter.BasculeOrientation().IsInit())
                {
                    Bascule(anIter.BasculeOrientation().Val(),false);
                }

                DoShowPtsMult(anIter.VisuPtsMult());

	        for 
	        (
	           std::list<cVerifAero>::const_iterator itV = anIter.VerifAero().begin();
	           itV != anIter.VerifAero().end();
	           itV++
	        )
                {
                   VerifAero(*itV);
                }
   

                if (anEC.SectionTracage().IsInit())
                {
                    const cSectionTracage & aST = anEC.SectionTracage().Val();

	            for 
	            (
	               std::list<cTraceCpleCam>::const_iterator itT = aST.TraceCpleCam().begin();
	               itT != aST.TraceCpleCam().end();
	               itT++
	            )
	            {
	                 PoseFromName(itT->Cam1())->ShowRel(*itT,*PoseFromName(itT->Cam2()));
	            }

	            if (aST.GetChar().Val())
	            {
	               std::cout << "Stop in trace \n";
	               getchar();
	            }
                }

                for
                (
	            std::list<cExportSimulation>::const_iterator itES=anIter.ExportSimulation().begin();
                    itES!=anIter.ExportSimulation().end();
	            itES++
                )
	        {
	            ExportOneSimule(*itES);
	        }


                if (anIter.Pose2Init().IsInit())
                {
                   ELISE_ASSERT(mESPA==0,"Mix Sensib && Init Pose");
                   const cPose2Init & aP2I = anIter.Pose2Init().Val();
                   bool aShow = aP2I.Show().Val();
                   std::vector<int> mVProfs = aP2I.ProfMin();

                   int aStepC = aP2I.StepComplemAuto().Val();
                   if (aStepC>=0)
                   {
                       if (mVProfs.empty())
                       {
                           mVProfs.push_back(2);
                       }
                       if (mVProfs.size()==1)
                       {
                           mVProfs.push_back(mVProfs[0]+1);
                       }

                       if (aStepC==0)
                       {
                           int aNb = (int)mVProfs.size();
                           aStepC = ElMax(1,mVProfs[aNb-1] - mVProfs[aNb-2]);
                       }
                       while (mVProfs.back() <= mProfMax)
                             mVProfs.push_back(mVProfs.back()+aStepC);
                       while ((mVProfs.size()>=2) && (mVProfs[mVProfs.size()-2] > mProfMax))
                              mVProfs.pop_back();

                       std::cout << "---- PROFS=" ;
                       for (int aK=0 ; aK<int(mVProfs.size()) ; aK++)
                           std::cout << " " << mVProfs[aK];
                       std::cout << "\n" ;
                   }
 
                   for (int aKProf=0 ; aKProf != int(mVProfs.size()) ; aKProf++)
                   {
                       int aProf = mVProfs[aKProf];
                       ELISE_ASSERT(mProfInit<=aProf,"Prof 2 Init non croissante !");

                       for (; mProfInit<aProf; mProfInit++)
                       {
                           if (aShow)
                           {
                              std::cout  << "xProf = " << mProfInit << "\n\n";
                           }
                           for (int aKPose=0 ; aKPose<int(mVecPose.size()) ; aKPose++)
                           {

                              if (
                                      ( mVecPose[aKPose]->Prof2Init() == mProfInit)
                                   && (!mVecPose[aKPose]->RotIsInit())
                                 )
                              {
                                   mVecPose[aKPose]->InitRot();
                                   mVecPose[aKPose]->SetDeFigee();
                                   if (aShow)
                                   {
                                      std::cout << "  Add Pose = " << mVecPose[aKPose]->Name() << "\n";
                                   }
                              }
                           }
                       }
                       bool aKProfLast = (aKProf==((int)mVProfs.size()-1));
                       DoContraintesAndCompens(anEC,anIter,kIterLast&&aKProfLast,LastEtape);
                   }
                }
                else
                {
                   DoContraintesAndCompens(anEC,anIter,kIterLast,LastEtape);
                }

                if (anIter.BasculeOrientation().IsInit())
                {
                    Bascule(anIter.BasculeOrientation().Val(),true);
                }
                if (anIter.FixeEchelle().IsInit())
                {
                    FixeEchelle(anIter.FixeEchelle().Val());
                }

                if (anIter.FixeOrientPlane().IsInit())
                {
                   FixeOrientPlane(anIter.FixeOrientPlane().Val());
                }
                if (anIter.BasicOrPl().IsInit())
                {
                   BasicFixeOrientPlane(anIter.BasicOrPl().Val());
                }

                if (anIter.BlocBascule().IsInit())
                {
                     BasculeBloc(anIter.BlocBascule().Val());
                }




	        const std::list<std::string> & aLM = anIter.Messages();
	        for 
	        (
	             std::list<std::string>::const_iterator itM=aLM.begin();
                     itM != aLM.end();
	             itM++
	        )
	        {
	             COUT()  << *itM << "\n";
	        }

                if (ShowMes())
                {
	            COUT()  << "--- End Iter " << mNbIterDone << " STEP " << mNbEtape << "\n\n";
                }

                TestInteractif(anIter.TestInteractif(),false);

                GoOnIter = false;

                const cCtrlTimeCompens * aCtrl = anIter.CtrlTimeCompens().PtrVal();
                if (aCtrl)
                {
                   if (mStatLastIter.PdsEvol())
                   {
                      double aSeuilMoy = aCtrl->SeuilEvolMoy();
                      double aSeuilMax = aCtrl->SeuilEvolMax().ValWithDef(aSeuilMoy*2.0);
                      GoOnIter = (mStatLastIter.MoyEvol()>=aSeuilMoy) ||  ( mStatLastIter.MaxEvol()>=aSeuilMax);
                      const cAutoAdaptLVM * aAAL = aCtrl->AutoAdaptLVM().PtrVal();
                      if (aAAL)
                      {
                           double aNewV = aSeuilMoy * aAAL->Mult();
                           bool aModeM =  aAAL->ModeMin().Val();
                           UpdateMul(mMulSLMGlob,aNewV,aModeM);
                           UpdateMul(mMulSLMEtape,aNewV,aModeM);
                           UpdateMul(mMulSLMIter,aNewV,aModeM);
                      }
                   }

                   if (aCptInIter <aCtrl->NbMin().Val() )
                   {
                       GoOnIter = true;
                   }
                   else if (aCptInIter >= aCtrl->NbMax())
                   {
                        GoOnIter = false;
                   }
                }
                aCptInIter++;
                mNbIterDone++;
            }
        }
    }

    if (anEC.SectionExport().IsInit())
       Export(anEC.SectionExport().Val());
    mNbEtape++;
}

typedef std::list<cEtapeCompensation> tLEC;
void cAppliApero::DoCompensation()
{
   const tLEC & aLEC =mParam.EtapeCompensation();
   int aNbRest = aLEC.size();
   for ( tLEC::const_iterator itEC=aLEC.begin(); itEC != aLEC.end() ;itEC++)
   {
      
      DoOneEtapeCompensation(*itEC,aNbRest==1);
      aNbRest--;
   }

   
   ExportImageResidu();
   MajAddCoeffMatrix();
   PosesAddMajick();
   MessageDebug("Global End");
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
