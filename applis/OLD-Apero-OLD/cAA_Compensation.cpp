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

#include "ext_stl/numeric.h" 

namespace NS_ParamApero
{

//  AJOUT DES OBSERVATIONS

void cAppliApero::AddObservations
     (
          const cSectionObservations & anSO,
          bool IsLastIter,
          cStatObs & aSO
     )
{
   if (IsLastIter && anSO.TxtRapDetaille().IsInit())
   {
      InitRapportDetaille(anSO.TxtRapDetaille().Val());
   }
   else
   {
        mFpRT = 0;
   }




   AddLevenbergMarkard(aSO);
   AddObservationsAppuisFlottants(anSO.ObsAppuisFlottant(),IsLastIter,aSO);
   AddObservationsCentres(anSO.ObsCentrePDV(),IsLastIter,aSO);
   AddObservationsAppuis(anSO.ObsAppuis(),IsLastIter,aSO);
   AddObservationsLiaisons(anSO.ObsLiaisons(),IsLastIter,aSO);

   AddObservationsRigidGrp(anSO.ObsRigidGrpImage(),IsLastIter,aSO);


   if (mFpRT)
   {
       ElFclose(mFpRT);
       mFpRT = 0;
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
// std::cout << "BBBBB " << aPC->RotIsInit()  << " " <<  aPC->DoAddObsCentre(anObs) << "\n";
            if (aPC->RotIsInit() && aPC->DoAddObsCentre(anObs))
            {
// std::cout << "CCCCCCC\n";
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
      cRapOnZ * aRAZ = 0;
      if (itOL->RappelOnZ().IsInit())
      {
          const cRappelOnZ & aRaz = itOL->RappelOnZ().Val();
          double anI = aRaz.IncC();
          aRAZ = new cRapOnZ(aRaz.Z(),anI,aRaz.IncE().ValWithDef(anI),aRaz.LayerMasq().ValWithDef(""));
      }
      cPackObsLiaison * aPackL = GetEntreeNonVide(mDicoLiaisons,itOL->NameRef(),"AddObservationsLiaisons"); 

      aPackL->AddObs(itOL->Pond(),itOL->PondSurf().PtrVal(),aSO,aRAZ);
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
      aBAF->AddObs(*itOAF,aSO);
   }
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

void cAppliApero::OneIterationCompensation(const cEtapeCompensation & anEC,bool IsLast)
{
    mCurEC = & anEC;
    mIsLastIter = IsLast;
    for (tDiPo::iterator itD=mDicoPose.begin() ; itD!=mDicoPose.end(); itD++)
    {
        itD->second->InitAvantCompens();
    }


    ActiveContraintes(true);
    mSetEq.SetPhaseEquation();
    ActiveContraintes(false);

    cStatObs  aSO(true);

    for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
    {
       mVecPose[aKP]->SetNbPtsMulNN(0);
    }

    AddObservations(anEC.SectionObservations(),IsLast,aSO);

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




    // std::cout  << "=========SOM-POND-ERR " << aSO.SomErPond() << "\n";
    // mSetEq.SolveResetUpdate(aSO.SomErPond());
    // mSetEq.Solve(aSO.SomErPond());

    mSetEq.Solve(aSO.SomErPond(),(bool *)0);
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
          GetCharOnBrkp();
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
    OneIterationCompensation(anEC,IsLastIter);

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


void cAppliApero::DoContraintesAndCompens
     (
            const cEtapeCompensation & anEC,
            const cIterationsCompensation &  anIter,
            bool  IsLastIter
     )
{

   mMTAct = 0;
   if (!anIter.MesureErreurTournante().IsInit())
   {
      DoOneContraintesAndCompens(anEC,anIter,IsLastIter);
      ExportSauvAutom();
      return;
   }

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


void  cAppliApero::DoOneEtapeCompensation(const cEtapeCompensation & anEC)
{
    delete mMTRes;
    mMTRes = 0;

    InitLVM(mCurSLMGlob,anEC.SLMGlob(),mMulSLMGlob,anEC.MultSLMGlob());
    InitLVM(mCurSLMEtape,anEC.SLMEtape(),mMulSLMEtape,anEC.MultSLMEtape());

    for (int aK=0 ; aK<int(anEC.IterationsCompensation().size()) ; aK++)
    {
        bool kIterLast = (aK==((int)anEC.IterationsCompensation().size()-1));
	const cIterationsCompensation &  anIter  = anEC.IterationsCompensation()[aK];

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
                   int aNb = mVProfs.size();
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
               DoContraintesAndCompens(anEC,anIter,kIterLast&&aKProfLast);
           }
        }
        else
        {
           DoContraintesAndCompens(anEC,anIter,kIterLast);
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
	    COUT()  << "--- End Iter " << aK << " ETAPE " << mNbEtape << "\n\n";
        }

        TestInteractif(anIter.TestInteractif(),false);
    }

    if (anEC.SectionExport().IsInit())
       Export(anEC.SectionExport().Val());
    mNbEtape++;
}

typedef std::list<cEtapeCompensation> tLEC;
void cAppliApero::DoCompensation()
{
   const tLEC & aLEC =mParam.EtapeCompensation();
   for ( tLEC::const_iterator itEC=aLEC.begin(); itEC != aLEC.end() ;itEC++)
      DoOneEtapeCompensation(*itEC);
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
