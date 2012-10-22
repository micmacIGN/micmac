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
#include "MICMAC.h"

extern void  t(Im2DGen I,int,int);
namespace NS_ParamMICMAC
{


static const std::string PrefAutoM = "AutoMask_";

/*************************************************/
/*                                               */
/*          Global Scope                         */
/*                                               */
/*************************************************/

static void InitArgOneEtapePx
	    (
                const cAppliMICMAC & anAppli,
                std::string    aMode,
	        const  cGeomDiscFPx & aGeomOfEtape,
		int    aNumPx,
		bool   isVraiFirstEtape,
                cArgOneEtapePx & anArg,
		bool  isRealEtape,  // Vraie premiere etape
                bool  VerifInit,    // Versus verif not init
                const std::string & aMes,
                const cEtapeMEC &   anEtape,
                const cTplValGesInit<double> &  aRegul_Quad,
                const cTplValGesInit<double> &  aRegul,
                const cTplValGesInit<double> &  aPas,
                const cTplValGesInit<int> &     aDilatAlti,
	        const cTplValGesInit<int> &     aDilatPlani,
                const cTplValGesInit<double> &  aDilatPlaniProp,
	        const cTplValGesInit<bool> &    aRedrPx,
	        const cTplValGesInit<bool> &    aDeqRedrPx,
                bool                            isOptimDiff
	    )
{

    static int aCpt=0; aCpt++;

    if (VerifInit)
    {
       if ( isRealEtape)
       {
	  if  (
                     (!aRegul.IsInit())
                 ||  (!aPas.IsInit())
                 ||  (!aDilatAlti.IsInit())
                 ||  (!aDilatPlani.IsInit())
	        )
          {
              cout << "Paralaxe-MEC, Pas d'initialisation pour  : " << aMes 
                   << " (DeZoom = " << anEtape.DeZoom() << ")\n";
              ELISE_ASSERT
              (
	             false,
		     "Error dans la partie Paralaxe specif Etapes "
              );
          }
	  else
          {
	      anArg.mRegul_Quad  = aRegul_Quad.ValWithDef(0.0);
	      anArg.mRegul      = aRegul.Val();
	      anArg.mPas        =  isOptimDiff ?  aPas.Val()  : anAppli.AdaptPas(aPas.Val());


	      anArg.mDilatAltiPlus  = aDilatAlti.Val();
	      anArg.mDilatAltiMoins  = aDilatAlti.Val();


	      anArg.mDilatPlani = aDilatPlani.Val();
              anArg.mRedrPx     = aRedrPx.ValWithDef(false);
              anArg.mDeqRedrPx  = aDeqRedrPx.ValWithDef(true);
              anArg.mDilatPlaniProp = aDilatPlaniProp.ValWithDef(0.0);
              if (isOptimDiff)
              {
                  ELISE_ASSERT
                  (
                     anArg.mPas == 1.0,
                     "Optim Diff, Steps must be 1.0"
                  );
              }
              anArg.mIncertPxPlus = aGeomOfEtape.GetEcartInitialPlus(anArg.mPas,aNumPx);
              anArg.mIncertPxMoins = aGeomOfEtape.GetEcartInitialMoins(anArg.mPas,aNumPx);
	      if (isVraiFirstEtape)
	      {
	          anArg.mDilatAltiPlus +=  anArg.mIncertPxPlus ;
	          anArg.mDilatAltiMoins +=  anArg.mIncertPxMoins ;
                  if ((anArg.mIncertPxPlus<0) || (anArg.mIncertPxMoins<0))
                  {
                      std::cout << "anArg.mIncertPx " << anArg.mIncertPxMoins << " " << anArg.mIncertPxPlus<< "\n";
                      ELISE_ASSERT(false,"anArg.mIncertPx");
                  }
	      }

          }
       }
    }
    else
    {
       if (
                 (aRegul_Quad.IsInit())
             ||  (aRegul.IsInit())
             ||  (aPas.IsInit())
             ||  (aDilatAlti.IsInit())
             ||  (aDilatPlani.IsInit())
             ||  (aRedrPx.IsInit())
             ||  (aDeqRedrPx.IsInit())
             ||  (aDilatPlaniProp.IsInit())
          )
       {
          cout << "Paralaxe-MEC, Initialisation interdite pour  : " << aMes 
               << " (DeZoom = " << anEtape.DeZoom() << ")\n";
          ELISE_ASSERT
          (
	       false,
               "Error dans la partie Paralaxe specif Etapes "
          );
       }
    }


}

void Init3ArgOneEtapePx
	    (
	        const  cGeomDiscFPx & aGeomOfEtape,
	        const cAppliMICMAC & anAppli,
                cArgOneEtapePx & aZArg,
                cArgOneEtapePx & aPx1Arg,
                cArgOneEtapePx & aPx2Arg,
		int  aNumEtape,  // Vraie premiere etape
                eModeGeomMEC aModeMEC,
                const cEtapeMEC &   anEtape,
                bool                isEtOpDiff
	    )
{
    InitArgOneEtapePx
    (
          anAppli,
          "Z",
          aGeomOfEtape,
	  0,
	  (aNumEtape==1),
          aZArg,
	  (aNumEtape>=0),
	  (aModeMEC==eGeomMECTerrain) ,
	  "Z",
	  anEtape,
	  anEtape.ZRegul_Quad(),
	  anEtape.ZRegul(),
	  anEtape.ZPas(),
	  anEtape.ZDilatAlti(),
	  anEtape.ZDilatPlani(),
	  anEtape.ZDilatPlaniPropPtsInt(),
          anEtape.ZRedrPx(),
          anEtape.ZDeqRedr(),
          isEtOpDiff
    );

    InitArgOneEtapePx
    (
          anAppli,
          "Px1",
          aGeomOfEtape,
	  0,
	  (aNumEtape==1),
          aPx1Arg,
	  (aNumEtape>=0),
	  (aModeMEC==eGeomMECIm1) ,
	  "Px1",
	  anEtape,
	  anEtape.Px1Regul_Quad(),
	  anEtape.Px1Regul(),
	  anEtape.Px1Pas(),
	  anEtape.Px1DilatAlti(),
	  anEtape.Px1DilatPlani(),
	  anEtape.Px1DilatPlaniPropPtsInt(),
          anEtape.Px1RedrPx(),
          anEtape.Px1DeqRedr(),
          isEtOpDiff
    );

    InitArgOneEtapePx
    (
          anAppli,
          "Px2",
          aGeomOfEtape,
	  1,
	  (aNumEtape==1),
          aPx2Arg,
	  (aNumEtape>=0),
	  (aModeMEC==eGeomMECIm1)  && (anAppli.DimPx()==2),
	  "Px2",
	  anEtape,
	  anEtape.Px2Regul_Quad(),
	  anEtape.Px2Regul(),
	  anEtape.Px2Pas(),
	  anEtape.Px2DilatAlti(),
	  anEtape.Px2DilatPlani(),
	  anEtape.Px2DilatPlaniPropPtsInt(),
          anEtape.Px2RedrPx(),
          anEtape.Px2DeqRedr(),
          isEtOpDiff
    );



}

/*************************************************/
/*                                               */
/*             cEtapeMecComp                     */
/*                                               */
/*************************************************/

typedef std::list<cOneModeleAnalytique> tListMod;


cEtapeMecComp::cEtapeMecComp
(
      cAppliMICMAC & anAppli,
      cEtapeMEC &    anEtape,
      bool                 isLastEtape,
      const cGeomDiscFPx & aGeomTerrain,
      const tContEMC &     aVEtPrec
) :
  mAppli  (anAppli),
  mEtape  (anEtape),
  mDeZoomTer (anEtape.DeZoom()),
  mDeZoomIm  (round_ni(mDeZoomTer*anEtape.RatioDeZoomImage().ValWithDef(1.0))),
  mIsLast (isLastEtape),
  mNum    ((int) aVEtPrec.size()),
  mGeomTer (aGeomTerrain),
  mSsResAlgo (mEtape.SsResolOptim().ValWithDef(1)),
  mPatternModeExcl (true),
  mListAutomDeSel  (),
  mModeleAnToReuse (0),
  mModeleAnImported(0),
  mModelesOut      (),
  mCaracZ          (0),
  mIsOptDiffer     (anEtape.AlgoRegul().ValWithDef(eAlgoCoxRoy)==eAlgoOptimDifferentielle),
  mIsOptDequant    (anEtape.AlgoRegul().ValWithDef(eAlgoCoxRoy)==eAlgoDequant),
  mIsOptimCont     (mIsOptDiffer || mIsOptDequant),
  mGenImageCorrel  ((!mIsOptimCont) && (anEtape.GenImagesCorrel().ValWithDef(mIsLast))),
  mPrec            (aVEtPrec.empty() ?0  : aVEtPrec.back()),
  mPredPC            (0),
  mPredMaskAuto      (0),
  mIsNewProgDyn      (false),
  mArgMaskAuto       (0),
  mEBI               (0),
  mTheModPrgD        (0),
  mTheEtapeNewPrgD   (0),
  mInterpFloat       (0)
{

    
     if (mIsOptimCont)
     {
        ELISE_ASSERT
        (
           isLastEtape,
           "Optimisation differentielle uniquement en derniere etape"
        );
     }
     
     if (!aVEtPrec.empty())
     {
        VerifInit(mEtape.SzW().IsInit(),"SzW");
        VerifInit(mEtape.AlgoRegul().IsInit(),"AlgoRegul");

	mEtape.DynamiqueCorrel().SetValIfNotInit(eCoeffCorrelStd);
	mEtape.AggregCorr().SetValIfNotInit(eAggregSymetrique);
	mEtape.ModeInterpolation().SetValIfNotInit(eInterpolMPD);
     }

      bool iPF = aVEtPrec.empty(); // is Pseudo First
      mGeomTer.SetDeZoom(anEtape.DeZoom());
      mSzFile = mGeomTer.SzDz();
      cArgOneEtapePx aZArg;
      cArgOneEtapePx aPx1Arg;
      cArgOneEtapePx aPx2Arg;

      Init3ArgOneEtapePx
      (
         mGeomTer,
         mAppli,
         aZArg,
         aPx1Arg,
         aPx2Arg,
	 (int) aVEtPrec.size(),
	 anAppli.ModeGeomMEC(),
	 anEtape,
         mIsOptimCont
      );

      if (anAppli.ModeGeomMEC()==eGeomMECTerrain)
      {
          mFilesPx.push_back
          (
	       new cFilePx(aZArg,anAppli,*this,GetPred(aVEtPrec,0),iPF,"Z",0)
          );
      }
      if (anAppli.ModeGeomMEC()==eGeomMECIm1)
      {
          mFilesPx.push_back
          (
	       new cFilePx(aPx1Arg,anAppli,*this,GetPred(aVEtPrec,0),iPF,"Px1",0)
          );
	  if (anAppli.DimPx() == 2)
	  {
            mFilesPx.push_back
            (
	       new cFilePx(aPx2Arg,anAppli,*this,GetPred(aVEtPrec,1),iPF,"Px2",1)
            );
	   }
      }

      // INTERVERTI AVEC L'ETAPE TYU pour prendre en compte
      // la dilat init

      // Pour l'etape 1 (la premiere, une fois
      // enlevee 000) on augment DilatAlti de l'incertitude
      // en Z

              // RAJOUTE apres l'intervretion car sinon le pas relatif (celui du xml) n'est pas initialise
	      // et la dilat init est incorrecte !
/*
      for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
      {
          aVPas[aK] = mFilesPx[aK]->Pas();
      }
      mGeomTer.SetStep(aVPas);

      if (aVEtPrec.size()==1)
      {
         int anEcInit[theDimPxMax];
         mGeomTer.GetEcartInt(anEcInit);
         for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
         {
             mFilesPx[aK]->DilatAlti() += anEcInit[aK];
	     // std::cout << "DAL " << mFilesPx[aK]->DilatAlti() << "\n"; getchar();
         }
      }
      for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
             std::cout << mFilesPx[aK]->DilatAlti() += anEcInit[aK];
*/
        
      double aVPas[theDimPxMax];

     // == TYU ===================
      mNbNappesEp = 0;
      int aNbDilPl = 0;

      for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
      {
          aVPas[aK] = mFilesPx[aK]->Pas();
          if (mFilesPx[aK]->NappeIsEpaisse())
          {
             mNumSeuleNapEp = aK;
             mNbNappesEp++;
          }

          if (mFilesPx[aK]->DilatPlani()!=0)
             aNbDilPl++;
      }

      mAlgoRegul= mEtape.AlgoRegul().Val();
      if (mAlgoRegul==eAlgoCoxRoySiPossible)
      {
         if (mNbNappesEp==1)
         {
            mAlgoRegul = eAlgoCoxRoy;
         }
         else
         {
            mAlgoRegul = mEtape.AlgoWenCxRImpossible().ValWithDef(eAlgo2PrgDyn);
         }
      }

      if (mNbNappesEp > 1)
         mNumSeuleNapEp=-1;
      mGeomTer.SetStep(aVPas);


       if (!mAppli.DoNothingBut().IsInit())
       {
           for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
           {
              mFilesPx[aK]->CreateMNTInit();
           }
       }


      if ((mNbNappesEp==0) && (aVEtPrec.size() > 1))
      {
         cout << "Etape numero " << mNum 
              << " a DeZoom " << DeZoomTer() << "\n";
         ELISE_ASSERT
         (
              false,
	      "Toutes les nappes sans epaisseur !! "
         );
      }

      if ((mNbNappesEp>1) && (mEtape.AlgoRegul().Val()==eAlgoCoxRoy))
      {
         cout << "Etape numero " << mNum 
              << " a DeZoom " << DeZoomTer() << "\n";
         ELISE_ASSERT
         (
              false,
	      "Plusieurs nappes epaisses en Cox Roy"
         );
      }

      if (mEtape.ImageSelecteur().IsInit())
      {
         // cImageSelecteur & aIS = mEtape.ImageSelecteur().Val();
         mPatternModeExcl = mEtape.ImageSelecteur().Val().ModeExclusion();
         std::list<string> aLPat  = mEtape.ImageSelecteur().Val().PatternSel();
         for 
         (
              std::list<string>::iterator itS = aLPat.begin();
              itS != aLPat.end() ;
              itS++
         )
         {
              mListAutomDeSel.push_back(mAppli.ICNM()->KeyOrPatSelector(*itS));
         }
      }

      if (anEtape.SzGeomDerivable().IsInit())
         mSzGeomDerivable = anEtape.SzGeomDerivable().Val();
      else
      {
           mSzGeomDerivable = 4;
           for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
           {
               if (mFilesPx[aK]->RedrPx())
               {
                   mSzGeomDerivable=0;
               }
           }
      }
      // Si la taille est 0, on sauvgarde le booleen a la bonne valeur
      // ensuite on  met une taille + importante pour bufferiser
      // et eviter trop d'appel cLineariseProj::Init, mais le fait
      // de mattre mUseGeomDerivable a 0, inhibe la derivation dans 
      mUseGeomDerivable = (mSzGeomDerivable!=0);
      if (mSzGeomDerivable ==0)
         mSzGeomDerivable = 10;


      if ((mNum !=0) && (anEtape.ModelesAnalytiques().IsInit()))
      {
          const cModelesAnalytiques & aMod = anEtape.ModelesAnalytiques().Val();
          const tListMod & aLMod = aMod.OneModeleAnalytique();
          for(tListMod::const_iterator itM=aLMod.begin();itM!=aLMod.end();itM++)
          {
	     if (itM->UseIt().Val())
	     {
                 cModeleAnalytiqueComp * aMod = new cModeleAnalytiqueComp(mAppli,*itM,*this);
                 if (itM->ReuseModele().ValWithDef(itM->TypeModele()==eTMA_DHomD))
                 {
                    ELISE_ASSERT
                    (
                        mModeleAnToReuse==0,
                        "Plusieurs Modeles analytiques a Reutiliser !"
                    );
                    mModeleAnToReuse = aMod;
                 }
                 mModelesOut.push_back(aMod);
                 // Il faut generer des fichier de paralaxes meme quand il sont
                 // en theorie inutiles car redondants
                 for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
                     mFilesPx[aK]->ForceGenFileMA();
                 if (aMod->ExportGlob())
                    mAppli.SetLastMAnExp(aMod);
             }
          }
      }
      InitModeleImported(aVEtPrec);
 
      // Pour que les eventuels fichier de correlation soient creees dans le processu
      // pere

      if (    (mGenImageCorrel)
           && (!mAppli.DoNothingBut().IsInit())
	 )
      {
         FileCorrel();
      }
    

      // Pour eviter les conflits lorsque les fichiers sont crees par les
      // process en paralelles
      //
      if (mAppli.OneDefCorAllPxDefCor().Val())
      {
          FileMasqOfNoDef();
      }


      if ((mNum !=0) && (mEtape.AlgoRegul().Val()==eAlgo2PrgDyn))
      {
          bool NewPrgDynOblig = false;
          bool NewPrgDynInterdit = false;



          const cTplValGesInit< cModulationProgDyn > &  aTplModul =  mEtape.ModulationProgDyn();
          if ( aTplModul.IsInit())
          {
             mTheModPrgD = &(aTplModul.Val());

             mArgMaskAuto = mTheModPrgD->ArgMaskAuto().PtrVal();

             if (mArgMaskAuto)
             {
                 mEBI = mArgMaskAuto->EtiqBestImage().PtrVal();
             }

             if (
                    (mTheModPrgD->EtapeProgDyn().size()==1)
                 && (( NbNappesEp() ==1) && (NumSeuleNapEp() ==0))
                )
             {
                  mTheEtapeNewPrgD  = &(*(mTheModPrgD->EtapeProgDyn().begin()));
                  if (
                             (mTheEtapeNewPrgD->ModeAgreg()!=ePrgDAgrSomme)
                        ||   (mTheEtapeNewPrgD->Px1MultRegul().IsInit())
                        ||   (mTheEtapeNewPrgD->Px2MultRegul().IsInit())
                     )
                   {
                        NewPrgDynInterdit = true;
                   }
             }
             else
             {
                NewPrgDynInterdit = true;
             }
             if (mTheModPrgD->ArgMaskAuto().IsInit())
             {
                NewPrgDynOblig = true;
             }

             if (mTheModPrgD->ChoixNewProg().IsInit())
             {
                  if (mTheModPrgD->ChoixNewProg().Val())
                     NewPrgDynOblig = true;
                  else
                     NewPrgDynInterdit = true;
             }
          }
         else
          {
             ELISE_ASSERT
             (
                 false,
                 "Ne supporte plus de valeur par defaut pout ModulationProgDyn"
             );
          }

          if (mEtape.CorrelAdHoc().IsInit())
          {
              cTypeCAH aT = mEtape.CorrelAdHoc().Val().TypeCAH();
              if (aT.Correl_PonctuelleCroisee().IsInit())
              {
                  NewPrgDynOblig = true;
              }
          }

          if (NewPrgDynInterdit && NewPrgDynOblig)
          {
                ELISE_ASSERT(false,"Incoherence dans specif choix new prog");
          }

          mIsNewProgDyn = NewPrgDynOblig;

      }




      if (mArgMaskAuto)
      {
          FileMaskAuto();
      }


      //   Calcul l'etape la + proche avec generation de masque autom
     if (mArgMaskAuto) mPredMaskAuto = this;
     for 
     (
         tContEMC::const_reverse_iterator itC= aVEtPrec.rbegin();
         (itC != aVEtPrec.rend()) && (mPredMaskAuto==0) && ((*itC)->mDeZoomTer==mDeZoomTer);
	 itC++
     )
     {
           if (((*itC)->mNum!=0) && ((*itC)->mArgMaskAuto))
           {
              mPredMaskAuto = *itC;
           }
     }

      //   Calcul l'etape la + proche avec parties cachees
     for 
     (
         tContEMC::const_reverse_iterator itC= aVEtPrec.rbegin();
         (itC != aVEtPrec.rend()) && (mPredPC==0);
	 itC++
     )
     {
           if (
                  ((*itC)->mNum!=0)   // Cas special sur le 0 qui est un dedoublemnt "bidon" du 1
               && ((*itC)->mEtape.GenerePartiesCachees().IsInit())
              )
           {
              mPredPC = *itC;
           }
     }
     mUsePC = (mPredPC!=0) && (mEtape.UsePartiesCachee().ValWithDef(true));
}

void cEtapeMecComp::SetCaracOfZoom()
{
   if (mCaracZ==0)
   {
      mCaracZ = mAppli.GetCaracOfDZ(mEtape.DeZoom());
      for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
      {
          mFilesPx[aK]->SetCaracOfZoom(*mCaracZ);
      }
   }
}
cCaracOfDeZoom &  cEtapeMecComp::CaracOfZ()
{
   ELISE_ASSERT(mCaracZ!=0,"cEtapeMecComp::CaracOfZ");
   return *mCaracZ;
}


void cEtapeMecComp::ExportModelesAnalytiques()
{
   for 
   (
       std::list<cModeleAnalytiqueComp*>::iterator itM=mModelesOut.begin();
       itM!=mModelesOut.end();
       itM++
   )
   {
       (*itM)->MakeExport();
   }
}


cInterpolateurIm2D<float> * cEtapeMecComp::InterpFloat()  const
{
   if (mInterpFloat==0)
   {
      mInterpFloat = InterpoleOfEtape(mEtape,(float *)0,(double *)0);
   }
   return mInterpFloat;
}


std::string cEtapeMecComp::NameFileRes(const std::string & aPref) const
{
   return   mAppli.FullDirResult()
          // + std::string("Correl_")
          + aPref
          + mAppli.NameChantier()
          + std::string("_Num_")
          + ToString(mNum)
          + std::string(".tif");
}

std::string cEtapeMecComp::NameFileCorrel() const
{
  return NameFileRes("Correl_");
}


Tiff_Im  cEtapeMecComp::FileRes(GenIm::type_el aTypeEl,const std::string &  aPref,bool NoTile) const
{
   
   L_Arg_Opt_Tiff  aLArg = Tiff_Im::Empty_ARG;
   if (NoTile)
   {
      aLArg = aLArg +  Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(1000000,1000000)));
   }
   bool IsModified;
   return Tiff_Im::CreateIfNeeded
          (
              IsModified,
              NameFileRes(aPref),
              mSzFile,
              aTypeEl,
              Tiff_Im::No_Compr,
              Tiff_Im::BlackIsZero,
              aLArg
          );
}

Tiff_Im  cEtapeMecComp::FileCorrel() const
{
   return FileRes(GenIm::u_int1,"Correl_");
}
Tiff_Im  cEtapeMecComp::FileMaskAuto() const
{
   return FileRes(GenIm::bits1_msbf,PrefAutoM,true);
}




Tiff_Im cEtapeMecComp::LastFileCorrelOK() const
{
   const cEtapeMecComp * anEt = this;

   while (anEt && (!anEt->mGenImageCorrel))
   {
       anEt = anEt->mPrec;
   }
   if (anEt)
   {
      if (anEt->DeZoomTer() == this->DeZoomTer())
         return anEt->FileCorrel();
   }
   ELISE_ASSERT(false,"Cannot Get cEtapeMecComp::LastFileCorrelOK");
   return FileCorrel();
}


bool cEtapeMecComp::SelectImage(cPriseDeVue * aPDV) const
{
    aPDV->SetMaitre(IsModeIm1Maitre(mEtape.AggregCorr().Val()));
    const std::string & aName = aPDV->Name();
    bool aRes = mPatternModeExcl;
    for
    ( 
         std::list<cSetName *>::const_iterator itP = mListAutomDeSel.begin();
         itP != mListAutomDeSel.end();
         itP++
    )
    {
        if ((*itP)->IsSetIn(aName) )
        {
           aRes =  ! mPatternModeExcl ;
        }
    }
    if (aPDV->IsMaitre())
    {
        ELISE_ASSERT
        (
            aRes,
            "ImageSelector : refuse image maitresse"
        );
    }
    return aRes;
}

int cEtapeMecComp::NumSeuleNapEp() const
{
    ELISE_ASSERT(mNumSeuleNapEp>=0,"cEtapeMecComp::NumSeuleNapEp");
    return mNumSeuleNapEp;
}
int cEtapeMecComp::NbNappesEp() const
{
    return mNbNappesEp;
}


bool cEtapeMecComp::KthNapIsEpaisse(int aK) const
{
   return mFilesPx[aK]->NappeIsEpaisse();
}


void cEtapeMecComp::VerifInit(bool aTest,const std::string & aMes)
{
   if (! aTest)
   {
        cout << "No INIT  for " << aMes  
             << " ,etape numero " << mNum 
             << " a DeZoom " << DeZoomTer() 
	     << "\n";
	ELISE_ASSERT(false,"cEtapeMecComp::VerifInit");
   }

}

void cEtapeMecComp::Show() const
{
    for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
        mFilesPx[aK]->Show("cEtapes::Show");
    cout << "------------  N=" << mNum << " ----------\n";
}

cEtapeMecComp::~cEtapeMecComp()
{
      DeleteAndClear(mFilesPx);
      DeleteAndClear(mModelesOut);
}


cFilePx * cEtapeMecComp::GetPred(const tContEMC & aCont,int anInd)
{
    for 
    (
         tContEMC::const_reverse_iterator itC= aCont.rbegin();
         itC != aCont.rend();
	 itC++
    )
        // Appelee avant que Px1IncCalc soit utilise, donc on
        // tient compte du cas ou Num == 0
        if ((*itC)->mFilesPx[anInd]->GenFile())
            return (*itC)->mFilesPx[anInd];
    return 0;
}

cModeleAnalytiqueComp * cEtapeMecComp::ModeleAnImported() const
{
   return mModeleAnImported;
}

bool cEtapeMecComp::PxAfterModAnIsNulle() const
{
   return    (mModeleAnImported!=0)
          && mMAImportPredImm
          && (! mModeleAnImported->Modele().ReuseResiduelle().Val());
}

void  cEtapeMecComp::InitModeleImported(const tContEMC & aCont)
{
    bool PredImm= true;
    for 
    (
         tContEMC::const_reverse_iterator itC= aCont.rbegin();
         itC != aCont.rend();
	 itC++
    )
    {
        if ((*itC)->mModeleAnToReuse)
        {
            mModeleAnImported = (*itC)->mModeleAnToReuse;
            mMAImportPredImm = PredImm;
            return;
        }
        PredImm=false;
    }
}


         // ACCESSEURS      

Pt2di cEtapeMecComp::SzFile() const
{
   return mSzFile;
}

INT   cEtapeMecComp::Num()    const
{
   return mNum;
}

bool  cEtapeMecComp::IsLast() const
{
   return mIsLast;
}

int cEtapeMecComp::DeZoomIm() const
{
    return mDeZoomIm;
}

int cEtapeMecComp::DeZoomTer() const
{
    return mDeZoomTer;
}

cEtapeMecComp* cEtapeMecComp::PredPC() const
{
    return mPredPC;
}
bool  cEtapeMecComp::UsePC() const
{
   return mUsePC;
}

bool   cEtapeMecComp::IsNewProgDyn() const { return mIsNewProgDyn;}
bool   cEtapeMecComp::HasMaskAuto() const { return mArgMaskAuto!=0;}

const cEtiqBestImage *  cEtapeMecComp::EBI() const {return mEBI;}

const cArgMaskAuto &  cEtapeMecComp::ArgMaskAuto() const
{
    ELISE_ASSERT(HasMaskAuto(),"cEtapeMecComp::ArgMaskAuto");
    return *mArgMaskAuto;
}

const cModulationProgDyn *  cEtapeMecComp::TheModPrgD() const {return mTheModPrgD;}
const cEtapeProgDyn *       cEtapeMecComp::TheEtapeNewPrgD() const {return mTheEtapeNewPrgD;}


const cGeomDiscFPx &  cEtapeMecComp::GeomTer() const
{
   return mGeomTer;
}

cGeomDiscFPx &  cEtapeMecComp::GeomTer() 
{
   return mGeomTer;
}

const cFilePx & cEtapeMecComp::KPx(int aK) const
{
   return *mFilesPx[aK];
}

const cEtapeMEC &  cEtapeMecComp::EtapeMEC() const
{
    return mEtape;
}

int cEtapeMecComp::SsResAlgo() const
{
   return mSsResAlgo;
}

int cEtapeMecComp::SzGeomDerivable() const
{
    return mSzGeomDerivable;
}

bool cEtapeMecComp::UseGeomDerivable() const
{
    return mUseGeomDerivable;
}


bool cEtapeMecComp::GenImageCorrel() const
{
   return mGenImageCorrel;
}

eAlgoRegul  cEtapeMecComp::AlgoRegul() const
{
   return mAlgoRegul;
}

bool cEtapeMecComp::IsOptDiffer() const
{
   return mIsOptDiffer;
}
bool cEtapeMecComp::IsOptimCont() const
{
   return mIsOptimCont;
}
bool cEtapeMecComp::IsOptDequant() const
{
   return mIsOptDequant;
}




const cFilePx & cEtapeMecComp::KThPx(int aK) const
{
   ELISE_ASSERT((aK>=0) && (aK<int(mFilesPx.size())),"cEtapeMecComp::KThPx");
   return *(mFilesPx[aK]);
}

Fonc_Num cEtapeMecComp::FoncMasqIn(bool ForceReinj)
{
   Fonc_Num aFoncMasq = mAppli.FoncMasqOfResol(DeZoomTer());

// std::cout << "FMMMMIIII " << mPrec << "\n";
// std::cout << "===FMMMMIIII " <<  mPrec->mArgMaskAuto  << "\n";
// std::cout << "----FMMMMIIII " << mPrec->mArgMaskAuto->ReInjectMask().Val() << "\n";

   if (mPrec && mPrec->mArgMaskAuto && (ForceReinj|| mPrec->mArgMaskAuto->ReInjectMask().Val()))
   {
      
      Tiff_Im aTFM =   mPrec->FileMaskAuto();

      double aRatio  = DeZoomTer() / double(mPrec->DeZoomTer());

      Fonc_Num  aFMAuto = StdFoncChScale_Bilin (
                               aTFM.in(0),
                               Pt2dr(0,0),
                               Pt2dr(aRatio,aRatio),
                               Pt2dr(1,1)
                          );
      aFMAuto = aFMAuto > 0.5;
      int aNbErod = mPrec->mArgMaskAuto->Erod32Mask().ValWithDef((aRatio<1.0) ? 2 : 0);
      if (aNbErod != 0)
      {
          aFMAuto = dilat_32(aFMAuto,aNbErod);
      }

      aFoncMasq =    aFoncMasq && aFMAuto;
   }

   return aFoncMasq;
}


double cEtapeMecComp::LoadNappesAndSetGeom
     (
                   cLoadTer & aLTer ,
                   Box2di aBoxIn
     )
{
   ELISE_ASSERT
   (
        aLTer.NbPx()== int(mFilesPx.size()),
        "Erreur interne dans cEtapeMecComp::LoadNappesAndSetGeom"
   );
   mGeomTer.SetClip(aBoxIn._p0,aBoxIn._p1);

   // Chargement du masque
   Im2D_Bits<1> aIMasq = aLTer.ImMasqTer();


   Fonc_Num aFoncMasq = FoncMasqIn();

/*
   Fonc_Num aFoncMasq = mAppli.FoncMasqOfResol(DeZoomTer());
   if (mPrec && mPrec->mArgMaskAuto && mPrec->mArgMaskAuto->ReInjectMask().Val())
   {
      
      Tiff_Im aTFM =   mPrec->FileMaskAuto();

      double aRatio  = DeZoomTer() / double(mPrec->DeZoomTer());

      Fonc_Num  aFMAuto = StdFoncChScale_Bilin (
                               aTFM.in(0),
                               Pt2dr(0,0),
                               Pt2dr(aRatio,aRatio),
                               Pt2dr(1,1)
                          );
      aFMAuto = aFMAuto > 0.5;
      int aNbErod = mPrec->mArgMaskAuto->Erod32Mask().ValWithDef((aRatio<1.0) ? 2 : 0);
      if (aNbErod != 0)
      {
          aFMAuto = dilat_32(aFMAuto,aNbErod);
      }

      if (0)
      {
         Video_Win aW = Video_Win::WStd(aIMasq.sz(),1);
         ELISE_COPY(aW.all_pts(),trans(aFMAuto+2*aFoncMasq,aBoxIn._p0),aW.odisc());
         std::cout << "FGjhuyiyuyyyyyyyyyyyyy\n"; getchar();
      }

      aFoncMasq =    aFoncMasq && aFMAuto;
   }
*/

   ELISE_COPY
   (
          aIMasq.all_pts(),
          trans(aFoncMasq,aBoxIn._p0),
          aIMasq.out()
   );



/*
   Im2D_Bits<1> aISsPIMasq = aIMasq;
*/

   Im2D_Bits<1> aISsPIMasq = aLTer.ImMasqSsPI();
   ELISE_COPY
   (
          aISsPIMasq.all_pts(),
          trans(mAppli.FoncSsPIMasqOfResol(DeZoomTer()),aBoxIn._p0),
          aISsPIMasq.out()
   );

   ELISE_COPY(aIMasq.border(1),0,aIMasq.out()|aISsPIMasq.out());

   // Le + simple pour traiter de maniere generique les dimensions
   // de paralaxe est parfois d'explorer un intervalle vide
   for (int aK=0 ; aK<  theDimPxMax ; aK++)
   {
      aLTer.PxMin()[aK] = 0;
      aLTer.PxMax()[aK] = 1;
   }
   cResProj32 aRP32 = Projection32(aISsPIMasq.in(),aISsPIMasq.sz());

   Fonc_Num aFNbPx =1;
   double aNbDilAti = 1;
   Pt2di aSz(0,0);

   for (int aK=0 ; aK< int(mFilesPx.size()) ; aK++)
   {
      mFilesPx[aK]->LoadNappeEstim(*this,aRP32,aISsPIMasq,aLTer.KthNap(aK),aBoxIn);
      aLTer.PxMin()[aK] = aLTer.KthNap(aK).mVPxMinAvRedr;
      aLTer.PxMax()[aK] = aLTer.KthNap(aK).mVPxMaxAvRedr;

      aNbDilAti *= 1+(mFilesPx[aK]->DilatAltiPlus()+mFilesPx[aK]->DilatAltiMoins());
      aFNbPx = aFNbPx * (aLTer.KthNap(aK).mImPxMax.in()-aLTer.KthNap(aK).mImPxMin.in());
      aSz = aLTer.KthNap(aK).mImPxMax.sz();
   }

   double aNbPx;
   ELISE_COPY
   (
      rectangle(Pt2di(0,0),aSz),
      Rconv(aFNbPx),
      sigma(aNbPx)
   );

   // double aSurCapa = aNbPx / (aSz.x*aSz.y *aNbDilAti);
   return aNbPx;
}


void cEtapeMecComp::RemplitOri(cFileOriMnt & aFOM) const
{
   mGeomTer.RemplitOri(aFOM);
   if (mFilesPx.size())
      mFilesPx[0]->RemplitOri(aFOM);
   aFOM.NameFileMasque().SetVal(mAppli.NameImageMasqOfResol(mEtape.DeZoom()));
   
}

void PostFiltragePx
     (
         Im2DGen anIm,
         const cPostFiltrageDiscont & aParam,
	 Fonc_Num  aFoncMasq
     )
{
   if (aParam.ValGradAtten().IsInit())
   {
       double aFDer = aParam.DericheFactEPC().Val();
       double aExpGr = aParam.ExposPonderGrad().Val();
       double aValGrAtt = aParam.ValGradAtten().Val();

       Symb_FNum aSGrad(deriche(anIm.in_proj(),aFDer));
       Fonc_Num aG2 =    (Square(aSGrad.v0()) +Square(aSGrad.v1())) 
                      /  ElSquare(aValGrAtt); 

      aFoncMasq = aFoncMasq/(1+pow(aG2,aExpGr/2.0));
   }

   Symb_FNum aFMasq(aFoncMasq);
   Fonc_Num  aFP_P (Virgule(anIm.in(0)*aFMasq,aFMasq));

   int aNbIt = aParam.NbIter().Val();
   double aSzF = aParam.SzFiltre() /sqrt((double) aNbIt);
   double aFact = (2*aSzF+1.0)/(2*aSzF+5.0);

   for (int aK=0 ; aK< aNbIt ; aK++)
   {
       aFP_P = canny_exp_filt(aFP_P,aFact,aFact);
   }
   Symb_FNum aSP_P( aFP_P); 
   Symb_FNum aFP (aSP_P.v0());
   Symb_FNum aP  (aSP_P.v1());

   Symb_FNum aNewF (aFP/Max(1e-5,aP));

   ELISE_COPY
   (
       anIm.all_pts(),
       aNewF,
       anIm.out()
   );
}

Tiff_Im cEtapeMecComp::FileMasqOfNoDef() 
{
   std::string aName =    mAppli.FullDirPyr()
                       +  std::string("Masq_NoDef_")
		       +  mAppli.NameChantier()
		       + std::string("_Num_")
		       + ToString(mNum)
		       +  std::string(".tif");


   if (! ELISE_fp::exist_file(aName))
   {
      Tiff_Im aFile
               (
                     aName.c_str(),
                     mAppli.SzOfResol(mDeZoomTer),
                     GenIm::bits1_msbf,
                     Tiff_Im::No_Compr,
                     Tiff_Im::BlackIsZero
	       );

      return aFile;
   }

   return Tiff_Im(aName.c_str());
}


void cEtapeMecComp::SauvNappes 
     (
        cLoadTer & aVNappes,
        Box2di      aBoxOut,
        Box2di      aBoxIn
     )
{


   cResProj32 aRP32 = Projection32
                      (
		         (! aVNappes.ImOneDefCorr().in()) && aVNappes.ImMasqSsPI().in(),
			 aVNappes.ImOneDefCorr().sz()
                      );

   for (int aK=0 ; aK< int(mFilesPx.size()) ; aK++)
   {
      
      cOneNappePx & aNP = aVNappes.KthNap(aK);
      Im2DGen aIResult =  mIsOptimCont ? 
                          Im2DGen(aNP.mPxRedr) : 
			  Im2DGen(aNP.mPxRes);

      if ((aK==0) && (mEtape.DoStatResult().IsInit()))
      {
           mAppli.StatResultat
           (
               Box2di(Pt2di(0,0),aBoxIn.sz()),
               /// Box2di(aBoxIn._p0,aBoxIn._p1),
               aNP.mPxRedr,
               mEtape.DoStatResult().Val()
           );
      }
      if (mAppli.ReprojPixelNoVal().Val() && (aRP32.IsInit()) && (! aRP32.IsFull()))
      {
           ELISE_COPY
           (
                  aIResult.all_pts(),
                  aIResult.in()[Virgule(aRP32.PX().in(),aRP32.PY().in())],
                  aIResult.out()
          );
      }


      if ( mEtape.PostFiltrageDiscont().IsInit())
      {
          PostFiltragePx
	  (
	      aIResult,
	      mEtape.PostFiltrageDiscont().Val(),
	      aVNappes.ImMasqSsPI().in(0)
	  );
      }

      if (aK==0)
      {
          Pt2di aP0  = Sup(Pt2di(0,0),aBoxOut._p0 - aBoxIn._p0 - Pt2di(5,5));
          Pt2di aP1  = Inf(aIResult.sz(),aBoxOut._p1 - aBoxIn._p0 + Pt2di(5,5));

          AllBasculeMnt
	  (
	     aP0,aP1,
	     aNP.mPxRedr.data(),
	     aNP.mPxRes.data(),
	     aNP.mPxRes.sz()
	  );
      }

      if (0)
      {
         Video_Win aW = Video_Win::WStd(aNP.mPxRedr.sz(),1.0);
         while (1)
         {
              double aDyn;

              std::cout << "REDR \n";
              std::cin >> aDyn;
 
              ELISE_COPY(aW.all_pts(),aNP.mPxRedr.in()*aDyn,aW.ocirc());

              std::cout << "Result \n";
              std::cin >>  aDyn;
              ELISE_COPY(aW.all_pts(),(aIResult.in())*aDyn,aW.ocirc());
         }
      }


      ::t(aIResult,mIsLast,mFilesPx[0]->DilatAltiPlus()+mFilesPx[0]->DilatAltiMoins());

      mFilesPx[aK]->SauvResulPx
      (
          // mIsOptimCont ? aNP.mPxRedr.in() : aNP.mPxRes.in(),
	  aIResult.in(),
          aBoxOut,
          aBoxIn
      );

      
      mFilesPx[aK]->SauvResulPxRel(aNP.mPxRes,aNP.mPxInit,aBoxOut,aBoxIn);


   }
   if (mGenImageCorrel)
   {
       Im2D_U_INT1 aIC = aVNappes.ImCorrelSol();
       ELISE_COPY
       (
           rectangle(aBoxOut._p0,aBoxOut._p1),
           trans(aIC.in(),-aBoxIn._p0),
           FileCorrel().out()
       );
   }
   if (mAppli.OneDefCorAllPxDefCor().Val())
   {
      ELISE_COPY
      (
          rectangle(aBoxOut._p0,aBoxOut._p1),
          trans((! aVNappes.ImOneDefCorr().in()) && aVNappes.ImMasqSsPI().in(),-aBoxIn._p0),
          FileMasqOfNoDef().out()
      );
   }


   {
      for
      (
         std::list<cGenerateProjectionInImages>::const_iterator itG=mEtape.GenerateProjectionInImages().begin();
         itG!=mEtape.GenerateProjectionInImages().end();
         itG++
      )
      {
          std::vector<cPriseDeVue *> aPdvs  = mAppli.AllPDV();
	  for (int aKP=0; aKP<int(aPdvs.size()) ; aKP++)
	  {
	     if (! BoolFind(itG->NumsImageDontApply(),aKP))
	     {
                SauvProjImage(*itG,*aPdvs[aKP],aVNappes,aBoxOut,aBoxIn);
	     }
	  }
      }
   }

   cSurfaceOptimiseur * aSO =mAppli.SurfOpt();
   if (aSO && aSO->MaskCalcDone())
   {
      ELISE_COPY
      (
          rectangle(aBoxOut._p0,aBoxOut._p1),
          trans(aSO->MaskCalc().in(),-aBoxIn._p0),
          FileMaskAuto().out()
      );
   }

}


void cEtapeMecComp::InitPaxResul(cLoadTer & aLT,const  Pt2dr & aP,double * aPx)
{
   for (int aK=0 ; aK< int(mFilesPx.size()) ; aK++)
   {
      aPx[aK] =  aLT.KthNap(aK).ResultPx(aP,mIsOptimCont);
   }
   mGeomTer.PxDisc2PxReel(aPx,aPx);
}

Pt2dr  cEtapeMecComp::ProjectionInImage
     (
         const cGeomImage  & aGeom,
         cLoadTer & aLT,
	 Pt2dr      aP
     )
{
   double aPx[theDimPxMax];
   InitPaxResul(aLT,aP,aPx);
   return aGeom.Objet2ImageInit_Euclid(mGeomTer.RDiscToR2(aP),aPx);  
}



/*
 * [37.1033,100.197][3263.06,2325.82][3227.38,2325.51]VERIF, DIF = [35.6802,0.318277]
 *
*/




void cEtapeMecComp::SauvProjImage
     ( 
         const cGenerateProjectionInImages& aGPI,
         const cPriseDeVue & aPDV,
         cLoadTer & aLT,
         Box2di      aBoxOut,
         Box2di      aBoxIn
     )
{

   Pt2di aSz = aLT.Sz();
   Im2D_REAL4 aImX(aSz.x,aSz.y,0.0);
   TIm2D<REAL4,REAL8> aTX(aImX);
   Im2D_REAL4 aImY(aSz.x,aSz.y,0.0);
   TIm2D<REAL4,REAL8> aTY(aImY);

   Pt2di aP;
   bool aSubXY = aGPI.SubsXY().Val();
   for (aP.x =0 ; aP.x<aSz.x ; aP.x++)
   {
       for (aP.y =0 ; aP.y<aSz.y ; aP.y++)
       {
           Pt2dr aPIm = ProjectionInImage(aPDV.Geom(),aLT,Pt2dr(aP));
           if (aSubXY)
           {
              aPIm = aPIm -mGeomTer.RDiscToR2(Pt2dr(aP));
           }

           aTX.oset(aP,aPIm.x);
           aTY.oset(aP,aPIm.y);
       }
   }

   std::pair<std::string,std::string>   aNames= mAppli.ICNM()->Assoc2To1(aGPI.FCND_CalcProj(),aPDV.Name(),true);
   for (int aK=0 ; aK<2 ; aK++)
   {
      Im2D_REAL4 aRes = (aK==0) ? aImX : aImY;
      std::string aNameRes = mAppli.FullDirResult() + (aK==0 ? aNames.first : aNames.second);
      bool isNew;
      Tiff_Im aTF = Tiff_Im::CreateIfNeeded
                    (
                       isNew,
                       aNameRes,
                       mSzFile,
                       GenIm::real4,
                       Tiff_Im::No_Compr,
                       Tiff_Im::BlackIsZero
                    );
      ELISE_COPY
      (
         rectangle(aBoxOut._p0,aBoxOut._p1),
         trans(aRes.in(),-aBoxIn._p0),
         aTF.out()
      );
   }
}



/*
void cEtapeMecComp::SauvXY(cLoadTer & aLT,cPriseDeVue & aPDV)
{
}
*/


// Size a affiner
int cEtapeMecComp::MemSizeCelluleAlgo() const
{
    switch(AlgoRegul())
    {
         case eAlgoCoxRoy   :
              return 16;
         case eAlgo2PrgDyn  :
              return 4;
         case eAlgoMaxOfScore :
         case eAlgoDequant :
              return 0;
         default :
              ELISE_ASSERT(false,"Optimization non supportee");

    }
    return 0;
}

int cEtapeMecComp::MemSizePixelSsCelluleAlgo() const
{
    switch(AlgoRegul())
    {
         case eAlgoCoxRoy   :
              return 20;
         case eAlgo2PrgDyn  :
              return 20;
         case eAlgoMaxOfScore :
              return (int) (4*mFilesPx.size());
         case eAlgoDequant :
              return (int) (12*mFilesPx.size());
         default :
              ELISE_ASSERT(false,"Optimization non supportee");
    }
    return 0;
}

int cEtapeMecComp::MemSizePixelAlgo() const
{
      int aNbCel = 1;

      for (int aK=0 ; aK<int(mFilesPx.size()) ; aK++)
          aNbCel *= 1 + ( mFilesPx[aK]->DilatAltiMoins() +mFilesPx[aK]->DilatAltiPlus())
		    + mFilesPx[aK]->DilatPlani();

      return     aNbCel* MemSizeCelluleAlgo()
              +  MemSizePixelSsCelluleAlgo();
}


void cEtapeMecComp::DoRemplitXMLNuage() const
{
   DoRemplitXML_MTD_Nuage();
   for
   (
        std::list<cExportNuage>::const_iterator itEN=mEtape.ExportNuage().begin() ;
        itEN !=  mEtape.ExportNuage().end() ;
        itEN++
   )
   {
        DoRemplitXMLNuage(*itEN);
   }
}

void cEtapeMecComp::DoRemplitXML_MTD_Nuage() const
{
   // Prudence pour la generation systematique, ce ne doit pas
   // fonctionner avec toutes les geometries



   cMTD_Nuage_Maille aMTD;
   aMTD.DataInside().SetVal(false);
   aMTD.RatioPseudoConik().SetVal(1000);
   aMTD.KeyNameMTD() = "Key-Assoc-Nuage-ImProf";
   cExportNuage anEN;
   anEN.MTD_Nuage_Maille().SetVal(aMTD);
   DoRemplitXMLNuage(anEN);
}


void cEtapeMecComp::DoRemplitXMLNuage(const cExportNuage & anEN) const
{
    cXML_ParamNuage3DMaille aNuage;
    bool aMTD = anEN.MTD_Nuage_Maille().IsInit();
    RemplitXMLNuage
    (
       anEN.MTD_Nuage_Maille(),
       aNuage,
       ( aMTD && anEN.DataInside().Val()) ?
          eModeCarteProfInterne           : 
          eModeCarteProfExterne
    );
    if (aMTD)
    {
       std::string aName =    mAppli.FullDirMEC()
                           +  mAppli.ICNM()->Assoc1To2
                              (
                                  anEN.KeyNameMTD(),
                                  mAppli.NameChantier(),
                                  ToString(mNum),
                                  true
                              );
        MakeFileXML(aNuage,aName);
    }
    if (anEN.PlyFile().IsInit())
    {
        const cPlyFile & aPlF = anEN.PlyFile().Val();

        cElNuage3DMaille * aN1 = cElNuage3DMaille::FromParam
                                    (aNuage, mAppli.FullDirMEC());


        for
        (
            std::list<cCannauxExportPly>::const_iterator itC=aPlF.CannauxExportPly().begin();
            itC!=aPlF.CannauxExportPly().end();
            itC++
        )
        {
              std::string aName = mAppli.WorkDir()+itC->NameIm();
              if (    (! itC->NamesProperty().empty())
                   || itC->FlagUse().IsInit()
                 )
              {
                 ELISE_ASSERT(false,"aN1->AddAttrFromFile is now private");
/*
                 aN1->AddAttrFromFile
                 (
                   aName,
                   itC->FlagUse().ValWithDef(0xFFFF),
                   itC->NamesProperty()
                 );
*/
              }
              else
              {
                   aN1->Std_AddAttrFromFile(aName);
              }
        }


        double aResol = aPlF.Resolution();
        cElNuage3DMaille * aN2 = aN1->ReScaleAndClip(aResol);

        std::list<std::string> aComs = aPlF.PlyCommentAdd();
        aComs.push_front(std::string("Resolution =")+ToString(aResol));
        aComs.push_front("www.micmac.ign.fr");
        aComs.push_front("Created with MicMac");

        char aSResol[8];
        sprintf(aSResol,"%.1f",aResol);

        std::string aName =    mAppli.FullDirMEC()
                           +  mAppli.ICNM()->Assoc1To2
                              (
                                  aPlF.KeyNamePly().Val(),
                                  mAppli.NameChantier(),
                                  aSResol,
                                  true
                              );

         aN2->PlyPutFile(aName,aComs,aPlF.Binary());
         MakeFileXML(aN2->Params(), StdPrefix(aName)+std::string(".xml"));
    }

   
}

std::string cEtapeMecComp::NameMasqCarteProf() const
{
   if (mPredMaskAuto) 
   {
      return mPredMaskAuto->NameFileRes(PrefAutoM);
   }
   return mAppli.NameImageMasqOfResol(DeZoomTer());
}

void cEtapeMecComp::RemplitXMLNuage
     (
             const cTplValGesInit<cMTD_Nuage_Maille> & aMTD,
             cXML_ParamNuage3DMaille & aNuage,
             eModeExportNuage aMode
     ) const
{
    // Ce sera aux geometries qui ont un besoin de param specif de corriger
    aNuage.NoParamSpecif().SetVal("toto");
    aNuage.NbPixel() = mGeomTer.NbPixel();
    aNuage.SsResolRef().SetVal(DeZoomTer());
    if (mAppli.RC())
    {
       aNuage.RepereGlob().SetVal(mAppli.RC()->El2Xml());
    }

    if (mAppli.XmlAnam())
    {
        aNuage.Anam().SetVal(*(mAppli.XmlAnam()));
    }

    // ELISE_ASSERT(mAppli.DimPx()==1,"cEtapeMecComp::RemplitXMLNuage with DimPx>1");

    if (aMode==eModeNuage3D)
    {
           ELISE_ASSERT(false,"eModeNuage3D,cEtapeMecComp::RemplitXMLNuage");
    }
    else
    {
        if (aMode==eModeCarteProfInterne)
        {
           ELISE_ASSERT(false,"eModeCarteProfExterne,cEtapeMecComp::RemplitXMLNuage");
        }


        aNuage.Image_Point3D().SetNoInit();
        cImage_Profondeur aIP;
        aIP.Image() = NameWithoutDir(KPx(0).NameFile());
        // aIP.Masq() = NameWithoutDir(mAppli.NameImageMasqOfResol(DeZoomTer()));
        aIP.Masq() = NameWithoutDir(NameMasqCarteProf());

        if (mGenImageCorrel)
        {
           aIP.Correl().SetVal(NameWithoutDir(NameFileCorrel()));
        }
        else
        {
           aIP.Correl().SetNoInit();
        }
        aIP.OrigineAlti() = mGeomTer.OrigineAlti();
        aIP.ResolutionAlti() =   mGeomTer.ResolutionAlti();
        // aIP.GeomRestit() = eGeomMNTFaisceauIm1PrCh_Px1D;
        aIP.GeomRestit() = mAppli.GeomMNT();

   

        aNuage.Image_Profondeur().SetVal(aIP);
    }


   if (aMTD.IsInit())
   {
      mAppli.PDV1()->Geom().RemplitOriXMLNuage
      (
          aMTD.Val(),
          mGeomTer,
          aNuage,
          aMode
      );
   }

    cGeomDiscFPx  aGT = GeomTer();
    aGT.SetClipInit();

    Pt2dr  aOriPlani,aResolPlani;
    // mGeomTer.SetOriResolPlani(aOriPlani,aResolPlani);
    aGT.SetOriResolPlani(aOriPlani,aResolPlani);

    if ((mAppli.GeomMNT() == eGeomMNTFaisceauIm1PrCh_Px1D) || (mAppli.GeomMNT()==eGeomMNTFaisceauPrChSpherik))
    {
       ElAffin2D anAff =   ElAffin2D::TransfoImCropAndSousEch(aOriPlani,aResolPlani);
       AddAffinite(aNuage.Orientation(),anAff);
    }


    aNuage.Orientation().Verif().SetNoInit();

    cVerifOrient aVerif;
    Box2dr aBoxDisc (Pt2dr(0,0),Pt2dr(aGT.SzDz()));
    double aIntervPax = (KPx(0).mIncertPxMoins + KPx(0).mIncertPxPlus) /2.0;

    std::vector<Pt3dr> aVE;
    for (int aK=0 ; aK < 10 ; aK++)
    {
         double aPxDisc[theDimPxMax]={0,0};
         double aPxTer[theDimPxMax]={0,0};

         Pt2dr aPDisc =  aBoxDisc.RandomlyGenereInside();
         aPxDisc[0] = NRrandC() * aIntervPax;

         if (aK<2)
         {
              aPxDisc[0] = (aK+1) / (aGT.PasPxRel0() * DeZoomTer());
              aPDisc =  aBoxDisc.FromCoordBar(Pt2dr(0.5,0.5));
         }

         Pt2dr aPTer = aGT.RDiscToR2(aPDisc);
         aGT.PxDisc2PxReel(aPxTer,aPxDisc);

         Pt3dr aPEucl =   mAppli.PDV1()->Geom().Restit2Euclid(aPTer,aPxTer);


         cVerifNuage aVN;
         aVN.IndIm() = aPDisc;
         aVN.Profondeur() = aPxDisc[0];
         aVN.PointEuclid() = aPEucl;

         aNuage.VerifNuage().push_back(aVN);
         aVE.push_back(aPEucl);
    }

     double aSensibility = euclid(aVE[0]-aVE[1]);
     aNuage.TolVerifNuage().SetVal(aSensibility/20.0);
    
    // aIP
    // aIP.Image() =;
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
