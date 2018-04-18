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


#if ELISE_windows
#include <iterator>
#endif


typedef std::list<cCalibrationCameraInc> tLC;
typedef std::list<cPoseCameraInc> tLP;


void cAppliApero::InitInconnues()
{
    // InitOffsGps();
    InitPoses();
    InitSurf();
    InitBlockCameras();

    InitCalibConseq();
}


void cAppliApero::InitOffsGps()
{
    for (std::list<cGpsOffset>::iterator itO=mParam.GpsOffset().begin() ; itO!=mParam.GpsOffset().end() ;itO++)
    {
        ELISE_ASSERT(!DicBoolFind(mDicoOffGPS,itO->Id()),"Mutiple Base with same name");
        mDicoOffGPS[itO->Id()] = new cAperoOffsetGPS(*itO,*this);
    }
}

        //  Calibs    

void cAppliApero::InitCalibCam()
{
    //  Initialisation des inconnues de calibration
    const tLC & aLC = mParam.CalibrationCameraInc();
    for ( tLC::const_iterator itC = aLC.begin(); itC!=aLC.end() ; itC++)
    {
        AssertEntreeDicoVide(mDicoArgCalib,itC->Name(),"Calibration");
        mDicoArgCalib[itC->Name()] = &(*itC);
        if (itC->CalibPerPose().IsInit())
        {
        }
        else
        {
	   mDicoCalib[itC->Name()] = cCalibCam::Alloc(itC->Name(),*this,*itC,0);
        }

	// std::cout << "CALIB=" << itC->Name() << "\n";
    }
}

class cCreatCtsrConseq
{
    public :
       bool operator < (const cCreatCtsrConseq & aC2) const
       {
            if (mNameEquiv< aC2.mNameEquiv) return true;
            if (mNameEquiv> aC2.mNameEquiv) return false;
            return mNameTime < aC2.mNameTime;
       }
       cCreatCtsrConseq(const std::string & aNE,const std::string & aNT,cPoseCam * aPC) :
            mNameEquiv (aNE),
            mNameTime  (aNT),
            mPC        (aPC)
       {
       }
       std::string mNameEquiv;
       std::string mNameTime;
       cPoseCam *  mPC;
};

void cAppliApero::InitCalibConseq()
{
    cDeclareObsCalConseq * aDOCC = mParam.DeclareObsCalConseq().PtrVal();
    if (!aDOCC) return;

    cSetName * aSel = mICNM->KeyOrPatSelector(aDOCC->PatternSel());
    cSetName * aSelJ =  mICNM->KeyOrPatSelector(aDOCC->KeyJump());
    std::vector<cCreatCtsrConseq> aVCCC;

    for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
    {
         cPoseCam * aPC = mVecPose[aKP];
         if (aPC && aSel->IsSetIn(aPC->Name()))
         {
            std::pair<std::string,std::string> aPair = mICNM->Assoc2To1(aDOCC->Key(),aPC->Name(),true);
            aVCCC.push_back(cCreatCtsrConseq(aPair.second,aPair.first,aPC));
         }
    }
    std::sort(aVCCC.begin(),aVCCC.end());
    
    for (int aKC=0 ; aKC<int(aVCCC.size()-1) ; aKC++)
    {
       cCreatCtsrConseq aCCC1 = aVCCC[aKC];
       cCreatCtsrConseq aCCC2 = aVCCC[aKC+1];
       if (   
                (aCCC1.mNameEquiv==aCCC2.mNameEquiv) 
            &&  (aCCC1.mPC->CalibCam()!= aCCC2.mPC->CalibCam())
            &&  (! aSelJ->IsSetIn(aCCC1.mPC->Name()))
          )
       {
            cParamIntrinsequeFormel & aPIF1 = aCCC1.mPC->CalibCam()->PIF();
            cParamIntrinsequeFormel & aPIF2 = aCCC2.mPC->CalibCam()->PIF();
            aPIF1.AddRegulConseq(&aPIF2,aDOCC->AddFreeRot(),false);
           // ===========
       }
    }
}
    
void cAppliApero::InitPoses()
{
    CompileInitPoseGen(false);
}

void cAppliApero::InitGenPoses()
{
    for (std::list<cCamGenInc>::const_iterator itCG=mParam.CamGenInc().begin() ; itCG!=mParam.CamGenInc().end() ; itCG++)
    {
          InitGenPoses(*itCG);
    }
}

void  cAppliApero::InitGenPoses(const cCamGenInc& aCGI)
{
     std::list<std::string> aLName  = mICNM->StdGetListOfFile(aCGI.PatterName()->NameExpr(),1,aCGI.ErrorWhenEmpytPat().Val());


    for (std::list<std::string>::const_iterator itN=aLName.begin() ; itN!=aLName.end() ; itN++)
    {

         // std::string aNameOri = DC() + "Ori" + aCGI.Orient()  +"/Orientation-" + *itN + ".xml";
         std::string aNameOri = DC() + StdNameCSOrient(aCGI.Orient(),*itN ,false);

         if ((!ELISE_fp::exist_file(aNameOri)) || (!mParam.StenCamSupresGBCam().Val()))
         {
             if (! ELISE_fp::exist_file(StdNameGBOrient(aCGI.Orient(),*itN ,false)))
             {
                 if (aCGI.ErrorWhenNoFileOrient().Val())
                 {
                      std::cout <<  "For ori=" << aCGI.Orient() << " Ima=" << *itN << "\n";
                      ELISE_ASSERT(false,"No file for required GB orient");
                 }
             }
             else
             {
                 cPosePolynGenCam * aPPGC = new cPosePolynGenCam(*this,*itN,aCGI.Orient());
                 mVecPolynPose.push_back(aPPGC);
                 mVecGenPose.push_back(aPPGC);
                 mDicoGenPose[*itN] = aPPGC;
             }
         }

    }

}




void cAppliApero::PreCompilePose()
{
   CompileInitPoseGen(true);
   InitGenPoses();
}



cCompileAOI::cCompileAOI(const cOptimizeAfterInit & anOAI)
{
   mParam = anOAI.ParamOptim();
   for
   (
       std::list<cApplyOAI>::const_iterator itAp= anOAI.ApplyOAI().begin();
       itAp != anOAI.ApplyOAI().end();
       itAp++
   )
   {
       mPats.push_back(new cElRegex(itAp->PatternApply(),10));
       mCstr.push_back(itAp->Cstr());
   }
}

void cAppliApero::CompileInitPoseGen(bool isPrecComp)
{
    //  Initialisation des inconnues d'orientation

    const tLP & aLP = mParam.PoseCameraInc();
    for ( tLP::const_iterator itP = aLP.begin(); itP!=aLP.end() ; itP++)
    {
        bool isMST = itP->MEP_SPEC_MST().IsInit();

        std::list<std::string> aLName;
        for 
        (
           std::list<std::string>::const_iterator itPat=itP->PatternName().begin();
           itPat!=itP->PatternName().end();
           itPat++
        )
        {
           if (itP->ByFile().Val())
           {
              std::list<std::string> aSet = GetListFromSetSauvInFile(mDC+*itPat);
               std::copy(aSet.begin(),aSet.end(),std::back_inserter(aLName));
           }
           else if (itP->ByKey().Val())
           {
               const std::vector<string>  *  aSet = mICNM->Get(*itPat);
               std::copy(aSet->begin(),aSet->end(),std::back_inserter(aLName));
           }
	   else if (itP->ByPattern().Val())
	   {
	      // std::list<std::string > aLName2Add = RegexListFileMatch(DC()+itP->Directory().Val(),*itPat,1,false);

               std::list<std::string > aLName2Add = mICNM->StdGetListOfFile(*itPat,1);

              
	      if (aLName2Add.empty())
	      {
                 std::cout << "For Pattern=["<< *itPat << "]\n";
	         ELISE_ASSERT(false,"No match for this pattern of image names");
	      }

             std::copy(aLName2Add.begin(),aLName2Add.end(),std::back_inserter(aLName));

	   }
	   else
	   {
	       aLName.push_back(*itPat);
	   }
	}



        if (itP->AutomGetImC().IsInit())
        {
            const std::string  & anId = itP->AutomGetImC().Val();
            const cBDD_PtsLiaisons & aBDL = GetBDPtsLiaisonOfId(anId);
            ELISE_ASSERT(aBDL.KeySet().size()==1,"AddAllNameConnectedBy multiple_set");
            // const std::vector<std::string> * aVNL =  ICNM()->Get(aBDL.KeySet()[0]);
            const std::string & aKA = aBDL.KeyAssoc()[0];

            std::map<std::string,double> mCpt;

            for (std::list<std::string>::const_iterator it1 =  aLName.begin() ;it1!=aLName.end() ; it1++)
                mCpt[*it1] = 0;
            for (std::list<std::string>::const_iterator it1 =  aLName.begin() ;it1!=aLName.end() ; it1++)
            {
                  std::list<std::string>::const_iterator it2 = it1; 
                  it2++;
                  for (; it2!=aLName.end() ; it2++)
                  {
                        std::string aNamePack = mDC+ICNM()->Assoc1To2(aKA,*it1,*it2,true);
                        double aNb = sizeofile(aNamePack.c_str());
                        mCpt[*it1] += aNb;
                        mCpt[*it2] +=aNb;
                  }
            }



            double aBestSc = -1e9;
            std::string aBestN;
            for (std::map<std::string,double>::iterator it=mCpt.begin();it!=mCpt.end(); it++)
            {
                if (it->second > aBestSc)
                {
                   aBestSc = it->second;
                   aBestN = it->first;
                }
            }

            aLName.clear();
            aLName.push_back(aBestN);
        }



        if (itP->AddAllNameConnectedBy().IsInit())
        {
            cElRegex * aFilter=0;
            if (itP->FilterConnecBy().IsInit())
                aFilter = new cElRegex(itP->FilterConnecBy().Val(),10);
            const std::string  & anId = itP->AddAllNameConnectedBy().Val();
            const cBDD_PtsLiaisons & aBDL = GetBDPtsLiaisonOfId(anId);
            ELISE_ASSERT(aBDL.KeySet().size()==1,"AddAllNameConnectedBy multiple_set");
            const std::vector<std::string> * aVNL =  ICNM()->Get(aBDL.KeySet()[0]);
            const std::string & aKA = aBDL.KeyAssoc()[0];

             std::list<std::string> aNewL;

            for (int aKL=0;aKL<int(aVNL->size()) ; aKL++)
            {
                std::pair<std::string,std::string> aPair = ICNM()->Assoc2To1(aKA,(*aVNL)[aKL],false);
                const std::string * aNewN=0;
                if (BoolFind(aLName,aPair.first))
                   aNewN = & aPair.second;
                if (BoolFind(aLName,aPair.second))
                   aNewN = & aPair.first;
                if (     aNewN 
                     &&  (!BoolFind(aNewL,*aNewN)) 
                     &&  (!BoolFind(aLName,*aNewN))
                   )
                {
                  if((aFilter==0) || (aFilter->Match(*aNewN)))
                  {
                     bool isNew = isPrecComp                          ?
                                  (! NamePoseCSIsKnown(*aNewN))         :
                                  (      (NamePoseCSIsKnown(*aNewN))
                                     &&  (! PoseFromName(*aNewN)->PreInit()) 
                                  );
                     if (isNew)
                     {
                             aNewL.push_back(*aNewN);
                     }
                  }
                }
            }
            aLName = aNewL;
            delete aFilter;
        }


        if (itP->PatternRefuteur().IsInit())
        {
           std::list<std::string> aNewL;
	   for 
	   (
	      std::list<std::string>::const_iterator itS=aLName.begin();
	      itS != aLName.end();
	      itS++
	   )
	   {
               if (!itP->PatternRefuteur().Val()->Match(*itS))
               {
                  aNewL.push_back(*itS);
               }
           }

           aLName = aNewL;
        }

        if (itP->AutoRefutDupl().Val())
        {
           std::list<std::string> aNewL;
	   for 
	   (
	      std::list<std::string>::const_iterator itS=aLName.begin();
	      itS != aLName.end();
	      itS++
	   )
	   {
               if (!BoolFind(aNewL,*itS))
               {
                  aNewL.push_back(*itS);
               }
           }

           aLName = aNewL;
        }





        if (itP->Filter().IsInit())
        {
           std::list<std::string> aNewL;
           const cNameFilter & aNF = itP->Filter().Val();
	   for 
	   (
	      std::list<std::string>::const_iterator itS=aLName.begin();
	      itS != aLName.end();
	      itS++
	   )
	   {
                   if (NameFilter(mICNM,aNF,*itS))
                   {
                      aNewL.push_back(*itS);
                   }
           }
            
            aLName = aNewL;
        }


        if (itP->ReverseOrderName().Val())
        {
	    aLName.reverse();
        }

        if (itP->KeyTranscriptionName().IsInit())
        {
            std::string aKeyTr  = itP->KeyTranscriptionName().Val();
            std::list<std::string> aNewL;
	   for 
	   (
	      std::list<std::string>::const_iterator itS=aLName.begin();
	      itS != aLName.end();
	      itS++
	   )
	   {
                aNewL.push_back(ICNM()->Assoc1To1(aKeyTr,*itS,true));
           }
           aLName = aNewL;
        }


        // On regarde si il existe un nom bundle gen
        {
             std::list<std::string> aNewL;
	     for 
	     (
	        std::list<std::string>::const_iterator itS=aLName.begin();
	        itS != aLName.end();
	        itS++
	     )
             {
                 bool ExistFileGB = false;
                 for (std::list<cCamGenInc>::const_iterator itGC=mParam.CamGenInc().begin();itGC!=mParam.CamGenInc().end();itGC++)
                 {
                     if (itGC->PatterName()->Match(*itS) && ELISE_fp::exist_file(StdNameGBOrient(itGC->Orient(),*itS,false)))
                        ExistFileGB = true;

/*
                     std::string aNameOri = DC() + "Ori" + itGC->Orient()  +"/GB-Orientation-" + *itS + ".xml";
                     if (ELISE_fp::exist_file(aNameOri))
                        ExistFileGB = true;
*/

                 }
                 if ((!ExistFileGB) || ( !mParam.GBCamSupresStenCam().Val()))
                 {
                      aNewL.push_back(*itS);
                 }
             }
            aLName = aNewL;
        }

        if (isPrecComp)
        {
           cCompileAOI * aCAOI =  
                    itP->OptimizeAfterInit().IsInit()              ?
                    new cCompileAOI(itP->OptimizeAfterInit().Val()):
                    0                                              ;

	   for 
	   (
	      std::list<std::string>::const_iterator itS=aLName.begin();
	      itS != aLName.end();
	      itS++
	   )
	   {
              std::string  aNameCal = "";
              if (itP->CalcNameCalib().IsInit())
              {
                  aNameCal = itP->CalcNameCalib().Val();
                  if (ICNM()->AssocHasKey(aNameCal))
                  {
                       aNameCal = ICNM()->Assoc1To1(aNameCal,*itS,true);
                  }
              }
              // Si les calibrations sont geres par le nouveua systeme
              if (!itP->CalcNameCalibAux().empty())
              {
                  ELISE_ASSERT(!itP->CalcNameCalib().IsInit(),"Choose CalcNameCalib OR CalcNameCalibAux");
                  aNameCal = "";
                  for 
                  (
                       std::list<cCalcNameCalibAux>::const_iterator itCAux = itP->CalcNameCalibAux().begin();
                       (itCAux != itP->CalcNameCalibAux().end()) && (aNameCal=="");
                       itCAux++
                  )
                  {
// std::cout << "HHHHHHHHH " << itCAux->CalcNameOnExistingTag().IsInit() << "\n";
                       if (itCAux->CalcNameOnExistingTag().IsInit())
                       {
                           const cCalcNameOnExistingTag & aCal = itCAux->CalcNameOnExistingTag().Val();
                           std::string aXmlFile =  DC() + ICNM()->Assoc1To1(aCal.KeyCalcFileOriExt(),*itS,true);
                           cElXMLTree aTree (aXmlFile);
                           bool GotTagE = (aTree.GetOneOrZero(aCal.TagExist()) !=0);
                           bool GotTagNonE = (aTree.GetOneOrZero(aCal.TagNotExist()) !=0);

// std::cout << "Auux " << aXmlFile << " " << GotTagE <<  " " << GotTagNonE << "\n";
                           if (aCal.ExigCohTags().Val())
                           {
                               ELISE_ASSERT(GotTagE!=GotTagNonE,"Incoherence in CalcNameOnExistingTag");
                           }
                           if (GotTagE  && (!GotTagNonE))
                           {
                              aNameCal =  ICNM()->Assoc1To1(aCal.KeyCalcName(),*itS,true);
                           }
                       }
                       if (itCAux->KeyCalcNameDef().IsInit())
                       {
                             aNameCal = ICNM()->Assoc1To1(itCAux->KeyCalcNameDef().Val(),*itS,true);
                       }
                  }
                  ELISE_ASSERT(aNameCal!="","Could not find satisfying CalcNameCalibAux");
              }
	   // std::string aNameCal = MatchAndReplace(anAutom,*itS,itP->CalcNameCalib());
	  


              if (DicBoolFind(mDicoPose,*itS))
              {
                 if ( itP->AutoRefutDupl().Val())
                 {
                 }
                 else
                 {
                         AssertEntreeDicoVide(mDicoPose,*itS,"Poses");
                 }
              }
              else
              {
	          cPoseCam * aPC = cPoseCam::Alloc(*this,*itP,*itS,aNameCal,aCAOI);
	          mDicoPose[*itS] = aPC;
	          mDicoGenPose[*itS] = aPC;
                  mVecPose.push_back(aPC);
                  mVecGenPose.push_back(aPC);
                  // tGrApero::TSom & aSom = mGr.new_som(aPC);
                  // aPC->SetSom(aSom);
                  if (! isMST)
                     aPC->InitCpt();

                  for
                  (
                        std::map<std::string,cLayerImage *>::iterator itLI = mMapLayers.begin();
                        itLI != mMapLayers.end();
                        itLI++
                  )
                  {
                          itLI->second->AddLayer(*aPC);
                  }
              }
           }

        }
	else
	{
           if ( itP->AutoRefutDupl().Val())
           {
              std::list<std::string> aNewL;
	      for 
	      (
	         std::list<std::string>::const_iterator itS=aLName.begin();
	         itS != aLName.end();
	         itS++
	      )
	      {
                  cPoseCam * aPC = PoseFromNameSVP(*itS);
                  if (aPC && ! (aPC->PreInit()))
                  {
                     aNewL.push_back(*itS);
                  }
              }
              aLName = aNewL;
           }

           if (isMST)  // L'init est faite "a la volee" dans MST
           {
               ConstructMST(aLName,*itP);
           }
           else
           {
	      for 
	      (
	         std::list<std::string>::const_iterator itS=aLName.begin();
	         itS != aLName.end();
	         itS++
	      )
	      {
                  cPoseCam * aPC = PoseFromName(*itS);
                  if (itP->PoseFromLiaisons().IsInit())
                  {
                      const std::vector<cLiaisonsInit> & aLI = 
                          itP->PoseFromLiaisons().Val().LiaisonsInit();
                      for
                      (
                         std::vector<cLiaisonsInit>::const_iterator itL=aLI.begin();
                         itL!=aLI.end();
                         itL++
                      )
                      {


                            cPoseCam * aPC2 = PoseFromName(aPC->CalNameFromL(*itL));
                            aPC->UpdateHeriteProf2Init(*aPC2);
                      }
                  }
                  else
                  {
                      aPC->Set0Prof2Init();
                      // La valeur par defaut 0 de Prof2Init va tres bien
                  }
                  aPC->DoInitIfNow();
	      }
           }
	}
    }
}

        //  Plans    

void cAppliApero::InitOneSurfParam(const cSurfParamInc & aParamSurf)
{
  if (aParamSurf.InitSurf().ZonePlane().IsInit())
  {
     const std::string & aName  = aParamSurf.InitSurf().ZonePlane().Val();
     cSurfParam * aSurf = mDicoSurfParam[aName];
     if (aSurf==0)
     {
        std::cout << "For name surf = " << aName << "\n";
       ELISE_ASSERT(false,"surf pas initialise");
     }

     aSurf->MakeInconnu(aParamSurf);
  }
  else
  {
      ELISE_ASSERT(false,"SurfParamInc :: ne gere que les plans ");
  }
}

void cAppliApero::InitSurf()
{
    for 
    (
        std::list<cSurfParamInc>::const_iterator itS=mParam.SurfParamInc().begin();
	itS!=mParam.SurfParamInc().end();
	itS++
    )
    {
       InitOneSurfParam(*itS);
    }
}


void cAppliApero::InitClassEquiv()
{
   int aCpt=0;
   for
   (
        std::list<cGroupeDePose>::const_iterator itG=mParam.GroupeDePose().begin();
        itG!=mParam.GroupeDePose().end();
        itG++
   )
   {
       ELISE_ASSERT(false,"cAppliApero::InitClassEquiv to update");
       cRelEquivPose * & aRel = mRels[itG->Id()];

       if (aRel!=0)
       {
          std::cout << "For Name " << itG->Id() << "\n";
          ELISE_ASSERT(false,"conflict name for cGroupeDePose");
       }

       aRel = new cRelEquivPose();

       for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
       {
           cPoseCam * aPC = mVecPose[aKP];
           std::string aName = ICNM()->Assoc1To1(itG->KeyPose2Grp(),aPC->Name(),true);
           aRel->AddAPose(aPC,aName);

       }

       if (itG->ShowCreate().Val())
           aRel->Show();
 
       aCpt++;
   }
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
