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

#if ELISE_windows
#include <iterator>
#endif

#include "Apero.h"

namespace NS_ParamApero
{

typedef std::list<cCalibrationCameraInc> tLC;
typedef std::list<cPoseCameraInc> tLP;


void cAppliApero::InitInconnues()
{
    InitPoses();
    InitSurf();
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
    
void cAppliApero::InitPoses()
{
   CompileInitPoseGen(false);
}

void cAppliApero::PreCompilePose()
{
   CompileInitPoseGen(true);
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
	         ELISE_ASSERT(false,"Aucun match pour ce pattern de nom de pose");
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
            {
                  std::list<std::string>::const_iterator it2 = it1; 
                  it2++;
                  for (; it2!=aLName.end() ; it2++)
                  {
                        std::string aNamePack = mDC+ICNM()->Assoc1To2(aKA,*it1,*it2,true);
                        double aNb = sizeofile(aNamePack.c_str());
 // std::cout << *it1 << " " << *it2 <<  " ===== " << aNb << "\n";
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
                                  (! NamePoseIsKnown(*aNewN))         :
                                  (      (NamePoseIsKnown(*aNewN))
                                     &&  (! PoseFromName(*aNewN)->PreInit()) 
                                  );
                     if (isNew)
                     {
                             aNewL.push_back(*aNewN);
                     }
// std::cout << *aNewN << " " << isNew << "\n";
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
              std::string  aNameCal = itP->CalcNameCalib();
              if (ICNM()->AssocHasKey(aNameCal))
              {
                   aNameCal = ICNM()->Assoc1To1(aNameCal,*itS,true);
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
                  mVecPose.push_back(aPC);
                  tGrApero::TSom & aSom = mGr.new_som(aPC);
                  aPC->SetSom(aSom);
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
//std::cout << "AAAAAA \n";

//std::cout << "NNN  " << aPC->CalNameFromL(*itL);

                            cPoseCam * aPC2 = PoseFromName(aPC->CalNameFromL(*itL));
// std::cout << "BBBB \n";
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
       cRelEquivPose * & aRel = mRels[itG->Id()];

       if (aRel!=0)
       {
          std::cout << "For Name " << itG->Id() << "\n";
          ELISE_ASSERT(false,"conflict name for cGroupeDePose");
       }

       aRel = new cRelEquivPose(aCpt);

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
