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
#include "StdAfx.h"
#include "../src/uti_phgrm/MICMAC/MICMAC.h"
#ifdef MAC
// Modif Greg pour avoir le nom de la machine dans les log
#include <sys/utsname.h>
#endif

// Test commit




const cCWWSImage * GetFromCAWSI(const cChantierAppliWithSetImage & aCAWSI,const std::string & aName)
{
   for (std::list<cCWWSImage>::const_iterator itW=aCAWSI.Images().begin() ; itW!=aCAWSI.Images().end() ; itW++)
   {
       if (itW->NameIm()==aName) 
          return &(*itW);
   }
   return 0;
}

bool GlobDebugMM=false;
Pt2di PBug(47,112);
bool IsPBug(const Pt2di &aP)
{
   return (aP.x==PBug.x) && (aP.y==PBug.y) && (GlobDebugMM);
}


cDebugEscalier * theDE = 0;

// ##

Pt2di ThePtDebug;

std::string  StdNameFromCple
             (
                   cElRegex_Ptr & anAutom,
                   const std::string & aPatSel,
                   const std::string & aPatRes,
                   const std::string & aSep,
                   const std::string & aName1,
                   const std::string & aName2
             )
{
   if (anAutom==0)
   {
      anAutom = new cElRegex(aPatSel,15);
      if (! anAutom->IsOk())
      {
          std::cout << "Autom =[" << aPatSel << "]\n";
          ELISE_ASSERT
          (
             false,
             "Cannot Compile Autom "
          );
      } 
   }
    
   bool aMatch = anAutom->Match(aName1+aSep+aName2);

   if (!aMatch)
   {
      std::cout << "Autom =[" << aPatSel << "]\n";
      std::cout << "Name =[" << aName1+aSep+aName2 << "]\n";
      ELISE_ASSERT(false,"Cannot Match Autom ");
   }
   bool aReplace =  anAutom->Replace(aPatRes);
   if (!aReplace)
   {
      std::cout << "Autom =[" << aPatSel << "]\n";
      std::cout << "Pattern Res =[" << aPatRes << "]\n";
      ELISE_ASSERT(false,"Cannot Replace Autom");
   }

   return anAutom->LastReplaced();
}


bool ModeGeomIsIm1InvarPx(const cParamMICMAC & aParam)
{
    switch(aParam.GeomImages())
    {
         case eGeomImageDHD_Px : 
         case eGeomImage_Hom_Px : 
         case eGeomImageDH_Px_HD : 
         case eGeomImage_Epip : 
         case eGeomImage_EpipolairePure : 

         return true;

         default  : break;
   }


   switch (aParam.GeomMNT())
   {
         case eGeomMNTFaisceauIm1PrCh_Px1D :
         case eGeomMNTFaisceauIm1PrCh_Px2D :

         case eGeomMNTFaisceauIm1ZTerrain_Px1D :
         case eGeomMNTFaisceauIm1ZTerrain_Px2D :

         case eGeomMNTFaisceauPrChSpherik :

         return true;
         default : break;
   }

   return false;

}
             
eModeGeomMEC CalculGeomMEC(const cParamMICMAC & aParam)
{
    switch(aParam.GeomImages())
    {
         case eGeomImageDHD_Px : 
         case eGeomImage_Hom_Px : 
         case eGeomImageDH_Px_HD : 
         case eGeomImage_Epip : 
         case eGeomImage_EpipolairePure : 
         {
              ELISE_ASSERT
              (
                  aParam.GeomMNT()==eGeomPxBiDim,
                  "Combinaison Geometries Image/Mnt incoherente"
              );
              return eGeomMECIm1;
          }
         break;

         case eGeomImageGrille :
         case eGeomImageRTO :  
         case eGeomImageModule :
         case eGeomImageCON :
         case eGeomImageOri :
         case eGeomGen :
         {
              // Quelques cas toleres jusqu'a present mais en fait
              // au comportement pas tres definis
              if (
                          (aParam.GeomImages()==eGeomImageModule)
                      ||  (aParam.GeomImages()==eGeomImageGrille)
                      ||  (aParam.GeomImages()==eGeomImageRTO)
                      ||  (aParam.GeomImages()==eGeomImageCON)
                      ||  (aParam.GeomImages()==eGeomGen)
		)
              {
                 switch (aParam.GeomMNT())
                 {
                    case eGeomMNTCarto :
/*-
                    case eGeomMNTFaisceauIm1PrCh_Px1D :
                    case eGeomMNTFaisceauIm1PrCh_Px2D :
*/
                    {
                        ELISE_ASSERT
                        (
                            false,
                            "Combinsaison Geometries Image/Mnt incoherente"
                        );
                    }
                    default :
                    break;
                 }
              }
              switch (aParam.GeomMNT())
              {
                    case eGeomMNTCarto :
                    case eGeomMNTEuclid :
                    case eGeomMNTFaisceauIm1PrCh_Px1D :
                    case eGeomMNTFaisceauPrChSpherik :
                    case eGeomMNTFaisceauIm1ZTerrain_Px1D :
                         return eGeomMECTerrain;
                    break;
                   
                    case eGeomMNTFaisceauIm1PrCh_Px2D :
                    case eGeomMNTFaisceauIm1ZTerrain_Px2D :
                          return eGeomMECIm1;
                    break;

                    default :
                    break;
              }

              ELISE_ASSERT
              (
                  false,
                  "Combinsaison Geometries Image/Mnt incoherente"
              );
           
         }
         break;

         case eNoGeomIm :
         {
              ELISE_ASSERT
              (
                  aParam.GeomMNT()==eNoGeomMNT,
                  "Combinsaison Geometries Image/Mnt incoherente"
              );
              return eNoGeomMEC;
         }
         break;

         default :
         break;
    }

    ELISE_ASSERT
    (
       false,
        "Combinsaison Geometries Image/Mnt incoherente"
    );
    return eNoGeomMEC;
}


            

/*****************************************/
/*       Constructeur & CO               */
/*****************************************/

bool UseICNM(cParamMICMAC & aParam)
{

   // Aujourd'hui je ne vois plus de raison de ne pas utiliser les ICNM
   // apparemment c'etait une prudence tres conservatrice ?
   return true;
}




     //   cAppliMICMAC::cAppliMICMAC  
     
cAppliMICMAC::cAppliMICMAC
(
    eModeAllocAM         aMode,
    const std::string &  aNameExe,
    const std::string &  aNameXML,
    // const cParamMICMAC & aParam,
    cResultSubstAndStdGetFile<cParamMICMAC>  aParam,
    char **              aArgAux,
    int                  aNbArgAux,
    const std::string  & aNameSpecXML
) :
   cParamMICMAC    (*(aParam.mObj)),
   mICNM           (NULL),
   mUseConstSpecIm1(false),
   mModeAlloc      (aMode),
   mNameExe        (aNameExe),
   mNameXML        (aNameXML),
   mNameSpecXML    (aNameSpecXML),
   mArgAux         (aArgAux),
   mNbArgAux       (aNbArgAux),
   mNbPDV          (0),
   mPDV1           (0),
   mPDV2           (0),
   mAutomNomsHoms  (0),
   mAutomNomPyr    (0),
   mEtape00        (0),
   mCurEtape       (0),
   mPrecEtape      (0),
   mEBI            (0),
   mCurMAI         (0),
   mGeomDFPx       (NULL),
   mGeomDFPxInit   (NULL),
   mCout           (std::cout),
   mTimeTotCorrel  (0.0),
   mTimeTotOptim   (0.0),
   mNbPointsByRect2 (0.0),
   mNbPointsByRectN (0.0),
   mNbPointByRectGen (0.0),
   mNbPointsIsole   (0.0),
   mLastMAnExp      (0),
   mImOkTerCur (1,1),
   mTImOkTerCur (mImOkTerCur),
   mImSzWCor    (1,1),
   mTImSzWCor   (mImSzWCor),
   mImOkTerDil (1,1),
   mTImOkTerDil (mImOkTerDil),
   mAll1ImOkTerDil (1,1),
   mAll1TImOkTerDil (mAll1ImOkTerDil),
   mBufCensusIm2   (),
   mGeoX (1,1),
   mTGeoX (mGeoX),
   mGeoY (1,1),
   mTGeoY (mGeoY),
   mPtrIV    (0),
   mVisu     (0),
   mOriPtLoc_Read (false),
   mMapEquiv (NULL),
   mAnamSA         (0),
   mXmlAnamSA      (0),
   mRepCorrel      (0),
   mRepInvCorrel   (0),
   mFileBoxMasqIsBoxTer (""),
   mInterpolTabule (10,8,0.0,eTabulMPD_EcartMoyen),
   mSurfOpt        (0),
   mCorrelAdHoc    (0),
   mGIm1IsInPax    (ModeGeomIsIm1InvarPx(*aParam.mObj)),
   mGPRed2         (0),
   mDoTheMEC       (true),
   mAnaGeomMNT     (0),
   mMakeMaskImNadir  (0),
   mMaxPrecision     (0),
   mGLOBMasq3D       (0),
   mGLOBNuage        (0),
   mCorrecAlti4ExportIsInit (false),
   mValmCorrecAlti4Export   (0.0)
   // mInterpolTabule (10,8,0.0,eTabul_Bilin)
   // mInterpolTabule (10,8,0.0,eTabul_Bicub)
{

      mDeZoomMax =1;
      mDeZoomMin =1<<20;
      for (std::list<cEtapeMEC>::const_iterator itE=  EtapeMEC().begin() ;  itE!= EtapeMEC().end() ; itE++)
      {
            int aDz = itE->DeZoom();
            if (aDz !=-1)
            {
                 ElSetMax(mDeZoomMax,aDz);
                 ElSetMin(mDeZoomMin,aDz);
            }
       }

       GlobDebugMM = DebugMM().Val();


        mDoTheMEC = DoMEC().Val();
        if (
                  NonExistingFileDoMEC().IsInit()
              &&  ELISE_fp::exist_file(WorkDir()+ NonExistingFileDoMEC().Val())
              &&  (!CalledByProcess().Val())
           )
        {
            mDoTheMEC = false; 
        }
	// NO_WARN
	mICNM			= ( ( UseICNM( (cParamMICMAC &)(*this) ) ) ? aParam.mICNM : NULL );
	mWM				= WithMessage().Val();
	mFullIm1		= ( ChantierFullImage1().Val() || ExportForMultiplePointsHomologues().Val() );
	mModeGeomMEC	= CalculGeomMEC(*this);
	mAucuneGeom     = ( mModeGeomMEC == eNoGeomMEC );
	mInversePx      = ( (GeomMNT()==eGeomMNTFaisceauIm1PrCh_Px1D)
                     || (GeomMNT()==eGeomMNTFaisceauIm1PrCh_Px2D)
                     || (GeomMNT()==eGeomMNTFaisceauPrChSpherik) );
	mDirImagesInit	= WorkDir()+DirImagesOri().ValWithDef("");
	mDirMasqueIms	= WorkDir()+DirMasqueImages().Val();
	mGeomDFPx		= new cGeomDiscFPx(*this);
	mGeomDFPxInit   = new cGeomDiscFPx(*this);
	mShowMes		= ( ByProcess().Val()==0 );
	mNbBoitesToDo	= NbBoitesMEC().Val();
	mFreqPtsInt		= ( EchantillonagePtsInterets().IsInit() ? FreqEchantPtsI() : 1 );
	mDefCorr		= DefCorrelation().Val();
	mEpsCorr		= EpsilonCorrelation().Val();	
	mMapEquiv		= StdAllocMn2n( ClassEquivalenceImage(), mICNM );
	mOutputDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():WorkDir() );
	setInputDirectory( WorkDir() );


  if (RepereCorrel().IsInit() && (RepereCorrel().Val() != "NO-REPERE"))
  {
      cRepereCartesien aRC = StdGetObjFromFile<cRepereCartesien>
                             (
                                  WorkDir()+RepereCorrel().Val(),
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                  TagRepereCorrel().Val(),
                                  "RepereCartesien"
                             );
       mRepCorrel =  new cChCoCart(cChCoCart::Xml2El(aRC));
       mRepInvCorrel = new cChCoCart(mRepCorrel->Inv());
  }


  if (Section_Vrac().SectionDebug().IsInit())
  {
      const cSectionDebug & aSD =  Section_Vrac().SectionDebug().Val();
      if (aSD.DebugEscalier().IsInit())
      {
         theDE = new cDebugEscalier(aSD.DebugEscalier().Val());
      }
  }
  if (Planimetrie().IsInit() && MasqueTerrain().IsInit() && MasqueTerrain().Val().FileBoxMasqIsBoxTer().IsInit())
  {
      mFileBoxMasqIsBoxTer =  WorkDir()
                            + MasqueTerrain().Val().FileBoxMasqIsBoxTer().Val();
  }

  if (FileExportApero2MM().IsInit())
  {
     mExpA2Mm = StdGetObjFromFile<cExportApero2MM>
                (
                    WorkDir()+FileExportApero2MM().Val(),
                     StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "ExportApero2MM",
                    "ExportApero2MM"
                );
  }



   mRecouvrMin  = (GeomMNT()==eGeomMNTFaisceauIm1PrCh_Px2D) ? 0.02 : -1;
   if (cParamMICMAC::Planimetrie().IsInit())
       mRecouvrMin = cParamMICMAC::RecouvrementMinimal().ValWithDef(mRecouvrMin);
   ThePtDebug = PtDebug().ValWithDef(Pt2di(0,0));

   char * aNameExeEnv = getenv("MICMAC_Exe");
   if (aNameExeEnv!=0)
      mNameExe = aNameExeEnv;

   // Parfois besoin de chantier en amont pour Anam ....
   if (!CalcNomChantier().IsInit() &&  NomChantier().IsInit())
      mNameChantier = NomChantier().Val();


   InitDirectories();
   InitAnamSA();
   InitImages();
   InitMemPart();
   ELISE_ASSERT(mNbPDV>=2,"Moins de 2 images selectionnees !!");

   if (CalcNomChantier().IsInit())
   {
       cElRegex * aRegex = 0;
       mNameChantier = StdNameFromCple
                       (
                           aRegex,
                           PatternSelChantier(),
                           PatNameChantier(),
                           SeparateurChantier().Val(),
                           mPDV1->Name(),
                           mPDV2->Name()
                       );
       delete aRegex;
   }
   else
   {
      mNameChantier = NomChantier().Val();
   }

   if ((aMode ==eAllocAM_Saisie) || (aMode==eAllocAM_Batch))
      return;



   VerifEtapes();
   mGeomDFPx->PostInit();
   *mGeomDFPxInit =  *mGeomDFPx;
   double aLogDZ = log2(mGeomDFPxInit->SzDz().XtY() / NbPixDefFilesAux().Val());
   mDeZoomFilesAux = ElMax(DeZoomDefMinFileAux().Val(),(1<<ElMax(0,(round_ni(aLogDZ)))));

   PostInitGeom();
   InitNadirRank();


   VerifImages();


/*
   if (FCND_CalcHomFromI1I2().IsInit())
   {
        DebugPxTrsv(*this);
   }
*/


   if ((aMode ==eAllocAM_VisuSup) || (mModeAlloc==eAllocAM_Surperposition))
      return;


   {
//  std::cout << "CBPBP " << CalledByProcess().Val() << "\n"; getchar();
        if (! CalledByProcess().Val())
           mGPRed2 = new cEl_GPAO;
        TestReducIm(128);
   }
/*
*/

   InitMecComp();
   if ((aMode == eAllocAM_STD) && (! DoNotMasqChantier()))
   {
       FileMasqOfResol(128);
   }
/*
*/
   if (! CalledByProcess().Val())
   {
       MakeFileTA();
   }


   // ShowEtapes();

    // Optionnel permet de generer le fichier apres modif 
    if  ((! CalledByProcess().Val()) && (mModeAlloc==eAllocAM_STD))
    {
/*
        FAIT DANS LE InitMecCOMP pour etre fait avant le CreateMNTInit
        if (! DoNotOriMNT())
           GenereOrientationMnt();
*/
	if (! DoNotExtendParam())
            SauvParam();
        if (! DoNotFDC())
           MakeFileFDC();
    }


    if (    (! CalledByProcess().Val())
         && (Use_MM_EtatAvancement().Val())
         && EtatAvancement().AllDone()
       )
    {
        return;
    }

    if ( (! CalledByProcess().Val()))
    {
        mMemPart.DeZoomLast().SetVal(mEtapesMecComp.back()->DeZoomTer());
        mMemPart.NumLastEtape().SetVal(mEtapesMecComp.back()->Num());
        SauvMemPart();
     }

    {
        SauvEtatAvancement(false);
        DoAllMEC();
        DoPostProcess();
        DoPurgeFile();
        SauvEtatAvancement(true);
    }

    if (DoSimul())
    {
       GenerateSimulations() ;
    }


     cElWarning::ShowWarns(WorkDir() + "WarnMICMAC.txt");

}

cAppliMICMAC::~cAppliMICMAC()
{
    DeleteAndClear(mPrisesDeVue);
    DeleteAndClear(mEtapesMecComp);
    delete mEtape00;
	if ( mGeomDFPx!=NULL ) delete mGeomDFPx;
	if ( mGeomDFPxInit!=NULL ) delete mGeomDFPxInit;
}




cAppliMICMAC * cAppliMICMAC::Alloc(int argc,char ** argv,eModeAllocAM aMode)
{
/* Priorites pour aller rechercher le fichier de specification :

    - dans la ligne de commande NameFileParamMICMAC=MonFichier.xml
    - dans le fichier de parametre effectif sous le tag <NameFileParamMICMAC>
    - dans la variable d'environnement MICMAC_Param
    - en dur, sous applis/MICMAC/ParamMICMAC.xml
*/
	ELISE_ASSERT(argc >= 2,"Not enough arguments");


    const char * aName = 0;
    std::string aNameFromLC;

    if (aName==0)
    {
       if (GetOneModifLC(argc-2,argv+2,"NameFileParamMICMAC",aNameFromLC))
       {
          aName = aNameFromLC.c_str();
       }
    }

    {
        cElXMLTree aTreeParam(argv[1]);
        if (aName==0)
        {
           cElXMLTree * aTrNameParam = aTreeParam.GetOneOrZero("NameFileParamMICMAC");
           if (aTrNameParam !=0)
           {
               aName = aTrNameParam->GetUniqueVal().c_str();
           }
        }
    }
// std::cout << "BBBBBBBBBBBBBBBBBBB\n"; getchar();
    // Creation de l'arbre de specif et de l'arbre des parametres
    if (aName==0) 
    {
        aName = getenv("NameFileParamMICMAC");
    }

    if (aName==0) // Compatibilit\'e 
    {
        aName = getenv("MICMAC_Param");
    }

    if (aName==0) 
    {
       // aName = "applis/MICMAC/ParamMICMAC.xml";
       std::string aStr =  StdGetFileXMLSpec("ParamMICMAC.xml");
       aName = (new std::string(aStr))->c_str();
    }


    cResultSubstAndStdGetFile<cParamMICMAC> aP2
                                           (
                                              argc-2,argv+2,
                                              argv[1],
                                              aName,
                                              "ParamMICMAC",
                                              "ParamMICMAC",
                                              "WorkDir",
                                              "FileChantierNameDescripteur"
                                           );
    Tiff_Im::SetDefTileFile(aP2.mObj->DefTileFile().Val());
    aP2.mObj->WorkDir() = aP2.mDC;

    if (IsActive(aP2.mObj->MapMicMac()))
    {
        aP2.mICNM->SetMapCmp(aP2.mObj->MapMicMac().Val(),argc,argv);
        return 0;
    }


/*
    cElXMLTree aTreeSpec(aName);
    // Verification adequation du param Specif
    aTreeParam.TopVerifMatch(&aTreeSpec,"ParamMICMAC");
    // Traitement des args en ligne de commandes
    aTreeParam.ModifLC(argc-2,argv+2,&aTreeSpec);
    // "Binding"
    cParamMICMAC aParam;
    // xml_init(aParam,&aTreeParam);
    xml_init(aParam,aTreeParam.Get("ParamMICMAC"));
*/

    if (aMode==eAllocAM_Saisie)
    {
        aP2.mObj->GeomMNT() = eNoGeomMNT;
        aP2.mObj->GeomImages() = eNoGeomIm;
    }
	
#if (ELISE_windows)
	// argv[1] is a filename (for the parameters' XML file)
	char *itChar = argv[1];
	while ( *itChar!='\0' )
	{
		if ( *itChar=='\\' ) *itChar='/';
		itChar++;
	}
#endif
	
    return new cAppliMICMAC(aMode,current_program_fullname()+" "+current_program_subcommand(),argv[1],aP2,argv+2,argc-2,aName);
}


    // Initialise les directories

void ViderDir(const std::string & aDir)
{
     ELISE_fp::PurgeDir(aDir);
}

// void MvDir2Dir(cInterfChantierNameManipulateur * aICN,std::string & aPat,

void cAppliMICMAC::InitDirectories()
{
    if (!TmpPyr().IsInit())
       TmpPyr().SetVal(TmpMEC());

    mFullDirMEC =  mOutputDirectory + TmpMEC();
    mFullDirPyr = mOutputDirectory + TmpPyr().Val();
    mFullDirGeom =  mOutputDirectory + TmpGeom().Val();
    mFullDirResult =  mOutputDirectory + TmpResult().Val();
    
    std::string aTmp = mOutputDirectory+ "Tmp-MM-Dir/";
    ELISE_fp::MkDir(mFullDirMEC);
    ELISE_fp::MkDir(mFullDirPyr);
    ELISE_fp::MkDir(mFullDirResult);
    if (TmpGeom().Val() != "")
       ELISE_fp::MkDir(mFullDirGeom);


   if (PurgeMECResultBefore().Val() &&  (!CalledByProcess().Val()))
   {
       const std::vector<std::string> * aSetPres = 0;
       if (PreservedFile().IsInit())
       {
          cInterfChantierNameManipulateur * aICD =  cInterfChantierNameManipulateur::BasicAlloc(mFullDirMEC);
          aSetPres = aICD->Get(PreservedFile().Val());
          std::cout  << "aSETPRES " << aSetPres->size() << "\n";
       }

       std::vector<std::string> aVTMP;
       std::string anUId =  GetUnikId();
       if (aSetPres)
       {
           for (int aK=0 ; aK<int(aSetPres->size()) ; aK++)
           {
               std::string  aNameIm = (*aSetPres)[aK];
               std::string aFileTmp = aTmp + anUId + aNameIm;
               ELISE_fp::MvFile(mFullDirMEC+aNameIm,aFileTmp);
               aVTMP.push_back(aFileTmp);
           }
       }

       ViderDir(mFullDirMEC);

       if (aSetPres)
       {
           for (int aK=0 ; aK<int(aSetPres->size()) ; aK++)
           {
               ELISE_fp::MvFile(aVTMP[aK],mFullDirMEC+(*aSetPres)[aK]);

           }
       }

       if (mFullDirResult != mFullDirMEC)
          ViderDir(mFullDirResult);
   }
}

    // Verification

void cAppliMICMAC::VerifEtapesSucc
     (
          const cEtapeMEC & anEt0,
          const cEtapeMEC & anEt1
     ) 
{
   ELISE_ASSERT
   (
         (anEt0.DeZoom() == anEt1.DeZoom())
     ||  (anEt0.DeZoom() == (2*anEt1.DeZoom())),
     "Resolution successives incoherentes"
   );
}

// A 
void cAppliMICMAC::VerifOneEtapes(const cEtapeMEC & anEt) 
{
     INT aDZ = anEt.DeZoom();
     ELISE_ASSERT
     (
         (aDZ>0) && is_pow_of_2(aDZ),
         "DeZoom non valable"
     );

}


void cAppliMICMAC::VerifEtapes() 
{
//std::cout << "==============================cAppliMICMAC::VerifEtapes \n";
/*
   // DEPLACE PLUS HAUT
   mDeZoomMax =1;
   mDeZoomMin =1<<20;
   for (std::list<cEtapeMEC>::const_iterator itE=  EtapeMEC().begin() ;  itE!= EtapeMEC().end() ; itE++)
   {
         int aDz = itE->DeZoom();
         if (aDz !=-1)
         {
              ElSetMax(mDeZoomMax,aDz);
              ElSetMin(mDeZoomMin,aDz);
         }
    }
*/
   std::list<cEtapeMEC>::const_iterator itE = EtapeMEC().begin();
   ELISE_ASSERT(itE->DeZoom()==-1,"Etape Init, Resol != -1");

   itE++;
   if (itE == EtapeMEC().end())
      return;

    VerifOneEtapes(*itE);
    std::list<cEtapeMEC>::const_iterator itPrec = itE;
    itE++; 
    while (itE != EtapeMEC().end())
    {
         VerifOneEtapes(*itE);
         VerifEtapesSucc(*itPrec,*itE);
         itPrec = itE;
         itE++;
    }

}

void cAppliMICMAC::VerifImages() const
{
   if (mModeAlloc==eAllocAM_Surperposition)
   {
      ELISE_ASSERT
      (
        mPrisesDeVue.size()>=2,
        "Pas au moins 2 images en eGeomMECIm1"
      );
      return;
   }
   switch (ModeGeomMEC() )
   {
       case eNoGeomMEC  :
       case eGeomMECIm1 :
           ELISE_ASSERT
           (
              mPrisesDeVue.size()==2,
              "Pas exactement 2 images en eGeomMECIm1"
           );
       break;

       case eGeomMECTerrain :
           ELISE_ASSERT
           (
              mPrisesDeVue.size() >= 2,
              "Pas au moins 2 images en eGeomMECTerrain"
           );
       break;

   }
}

/*****************************************/
/*       "Compile" les etapes de MEC     */
/*****************************************/

bool  cAppliMICMAC::DoMTDNuage() const
{
   return    (!DoNothingBut().IsInit())
          || (ButDoMTDNuage().Val())
          || (!DoNotOriMNT());
}

bool  cAppliMICMAC::DoNotOriMNT() const
{
   return     DoNothingBut().IsInit() 
           && (! ButDoOriMNT().Val())
           && (! DoSimul());
}
bool  cAppliMICMAC::DoNotExtendParam() const
{
   return     DoNothingBut().IsInit() 
           && (! ButDoExtendParam().Val());
}
bool  cAppliMICMAC::DoNotFDC() const
{
   if   (   DoNothingBut().IsInit() 
           && (! ButDoFDC().Val())
	)
	return true;
   return ! DoFDC().Val();
}



bool  cAppliMICMAC::DoNotMasqChantier() const
{
   return     DoNothingBut().IsInit() 
           && (! ButDoMasqueChantier().Val());
}

bool  cAppliMICMAC::DoNotMemPart() const
{
   return     DoNothingBut().IsInit() 
           && (! ButDoMemPart().Val());
}


bool cAppliMICMAC::DoSimul() const
{
   if (! SectionSimulation().IsInit())
     return false;
   if (!DoNothingBut().IsInit())
    return true;

  return ButDoSimul().Val();
}

void cAppliMICMAC::InitMemPart()
{

   if (DoNotMemPart())
       return;
   mNameFileMemPart = 
           FullDirResult() 
           + std::string("MemPart")
           + (
                 (ModeGeomMEC()==eGeomMECIm1)  ?
                 (
                       std::string("_")
                    + StdPrefixGen(PDV1()->Name())
                    + std::string("_")
                    + StdPrefixGen(PDV2()->Name())
                 )                                 :
                 std::string("")
             )
           + std::string(".xml");

    if (   
            (! CalledByProcess().Val())
         && (! ELISE_fp::exist_file(mNameFileMemPart))
       )
    {
          SauvMemPart();
    }

    cElXMLTree aTree(mNameFileMemPart);
    xml_init(mMemPart,aTree.Get("MemPartMICMAC"));
}

void cAppliMICMAC::SauvMemPart()
{
   if (       (DoNotMemPart())
         ||   ( CalledByProcess().Val())
      )
       return;

    cElXMLTree * aTree = ToXMLTree(mMemPart);
    FILE * aFP2 = ElFopen(mNameFileMemPart.c_str(),"w");
    if (aFP2==0)
    {
         std::cout << "For File =[" << mNameFileMemPart << "]\n";
         ELISE_ASSERT(aFP2!=0,"cAppliMICMAC::SauvMemPart File for _compl.xml");
    }
    aTree->Show("      ",aFP2,0,0,true);
    ElFclose(aFP2);
    delete aTree;
}

/*****************************************/
/*       "Compile" les etapes de MEC     */
/*****************************************/

bool IsModeIm1Maitre(const eModeAggregCorr & aMode)
{
    return    (aMode==eAggregIm1Maitre) 
           || (aMode==eAggregMaxIm1Maitre)
           || (aMode==eAggregMinIm1Maitre)
           || (aMode==eAggregMoyMedIm1Maitre);
}

std::string  cAppliMICMAC::NameFileSzW(int aDz)
{
    return FullDirPyr() + "ImSzW_Dz" + ToString(aDz) + PDV1()->Name() + ".tif";
}




void cAppliMICMAC::InitMecComp()
{
   mHasOneModeIm1Maitre = false;
   std::list<cEtapeMEC>::iterator itE = EtapeMEC().begin();
   itE++;
   int aCpt = (int) EtapeMEC().size()-2;
   bool isFirst=true;
   int aKEtape = 1;
   for (; itE != EtapeMEC().end() ; itE++)
   {
        // La premiere fois on rajoute 2 etape, la toute
        // premiere ne sera pas executee mais contiendra
        // les infos sur les paralaxes initiales (soit que
        // des 0 soit un MNT exhogene)
        if (itE->InterfaceVisualisation().IsInit())
        {
            FirstEtapeMEC().SetVal(aKEtape);
            ByProcess().SetVal(0);
            mPtrIV = & (itE->InterfaceVisualisation().Val());
        }
        int aNb = isFirst ? 2  : 1;
        for (int aK=0; aK<aNb ; aK++)
        {
           cEtapeMecComp * anEt=
                    new cEtapeMecComp
                    (
                        *this,
                        *itE,
                        (aCpt==0),
                        *mGeomDFPx,
		        mEtapesMecComp
                    );
           mEtapesMecComp.push_back ( anEt);
           int aDz = anEt->EtapeMEC().DeZoom();


           if (anEt->UseWAdapt() && (itE->DeZoom()>0) &&  (! CalledByProcess().Val()))
           {
                int aDZ = itE->DeZoom();
                std::string aNameSzW = NameFileSzW(aDz);
                if (! ELISE_fp::exist_file(aNameSzW))
                {
                    std::string aCom =    MM3dBinFile("MMCalcSzWCor ") 
                                       + PDV1()->IMIL()->NameFileOfResol(aDZ) 
                                       + " Out=" + aNameSzW;
                    system_call(aCom.c_str());

                    // std::cout << "CCCcc " << aCom << "\n"; getchar();
                }
           }
        }
        if ( IsModeIm1Maitre(itE->AggregCorr().Val()))
        {
           mHasOneModeIm1Maitre = true;
        }
	aCpt--;
        isFirst = false;
        aKEtape++;
   }
   // On supprime la premier etape (artificielle) de la liste de celles
   // a executer
   if (! isFirst)  // Pour l'instant on essaye encore de gerer un ensemble vide d'etapes
   {
       mEtape00 = mEtapesMecComp.front();
       mEtapesMecComp.pop_front();
   }
   for (int aDz = mDeZoomMin ; aDz<=mDeZoomMax ; aDz*=2)
   {
       mVCaracZ.push_back
       (
           new cCaracOfDeZoom
               (
                   aDz,
                   (aDz == mDeZoomMin)  ? 0 : mVCaracZ.back(),
                   *this
               )
       );
   }

   for
   (
        tContEMC::const_iterator itE = mEtapesMecComp.begin();
        itE != mEtapesMecComp.end();
        itE++
   )
   {
      (*itE)->SetCaracOfZoom();
   }


   if  ((! CalledByProcess().Val()) && (mModeAlloc==eAllocAM_STD) && (! DoNotOriMNT()))
   {
        GenereOrientationMnt();
   }
   if (mEtape00) 
       mEtape00->CreateMNTInit();
   for
   (
        tContEMC::const_iterator itE = mEtapesMecComp.begin();
        itE != mEtapesMecComp.end();
        itE++
   )
   {
      (*itE)->CreateMNTInit();
   }
}

cCaracOfDeZoom * cAppliMICMAC::GetCaracOfDZ(int aDZ) const
{
     for (int aK=0; aK<int(mVCaracZ.size()) ; aK++)
     {
         if (mVCaracZ[aK]->DeZoom() == aDZ)
            return mVCaracZ[aK];
     }
     ELISE_ASSERT
     (
          false,
          "cAppliMICMAC::GetCaracOfDZ"
     );
     return 0;
}


/*****************************************/
/*       Initialise les images           */
/*****************************************/

cAnamorphoseGeometrieMNT * cAppliMICMAC::AnaGeomMNT() const {return mAnaGeomMNT;}
cMakeMaskImNadir * cAppliMICMAC::MMImNadir() const {return mMakeMaskImNadir;}
    
void cAppliMICMAC::InitAnamSA() 
{
    if(! AnamorphoseGeometrieMNT().IsInit())
      return;

    mAnaGeomMNT = AnamorphoseGeometrieMNT().PtrVal();
    mMakeMaskImNadir = mAnaGeomMNT->MakeMaskImNadir().PtrVal();

    ELISE_ASSERT( GeomMNT() == eGeomMNTEuclid,"Anamophose incompatible avec Non-Euclid");

    const cAnamorphoseGeometrieMNT & aAGM = AnamorphoseGeometrieMNT().Val();
    if (aAGM.AnamSurfaceAnalytique().IsInit())
    {
       mXmlAnamSA = new  cXmlOneSurfaceAnalytique;
       const cAnamSurfaceAnalytique & anASA = aAGM.AnamSurfaceAnalytique().Val();
       mNameAnamSA = WorkDir()+anASA.NameFile();
       mAnamSA = SFromFile
               (
                     mNameAnamSA,
                     anASA.Id(),
                     "",
                     mXmlAnamSA
               );
       if (mAnaGeomMNT && mAnaGeomMNT->UnUseAnamXCste().Val())
       {
              mAnamSA->SetUnusedAnamXCSte();
       }
       ELISE_ASSERT(!mRepCorrel,"Anam and RepCorrel incompatibles");
    }
    else if (mRepCorrel!=0)
    {

/*
for (int aK=0 ; aK<10 ; aK++)
        std::cout << mRepCorrel->FromLoc(Pt3dr(0,0,0))
                  << mRepCorrel->FromLoc(Pt3dr(1,0,0)) - mRepCorrel->FromLoc(Pt3dr(0,0,0))
                  << mRepCorrel->FromLoc(Pt3dr(0,1,0)) - mRepCorrel->FromLoc(Pt3dr(0,0,0))
                  << mRepCorrel->FromLoc(Pt3dr(0,0,1)) - mRepCorrel->FromLoc(Pt3dr(0,0,0))
                  << "\n";
*/
        mAnamSA = cInterfSurfaceAnalytique::FromCCC(*mRepCorrel);
    }
    else
    {

      // Il y a un probleme avec l'utilisation des surfaces analytique identite car une surface doit etre telle que
      // la surface moyenne est L=0, donc elle doit etre centree sur le ZMoyen, qui est inconnu ici; repousser la
      // creation des surface semble aussi assez complique; bref ca se mord la queue de facon difficile a contourner,
      // Le choix qui est fait est d'imposer la connaissance du Z moyen dans cette configuration aerienne standard;
      // ceci n'étant utilise que pour la creation d'image nadir, ce sera encapsule dans un appel global, il restera
      // a traiter aussi la cas des repere locaux qui doivente etre considere comme une surface analytique semi triviale
      // c'est un peu une usine a gaz ....

         ELISE_ASSERT(mRepCorrel==0,"Ajouter gestion du repere correl sur Masque Image Nadir");


         double aZMoy = -1e30;
         if (IntervAltimetrie().IsInit())
         {
             cIntervAltimetrie * anIA= IntervAltimetrie().PtrVal();
             if (anIA->ZMoyen().IsInit())
                aZMoy = anIA->ZMoyen().Val();
         }
         ELISE_ASSERT(aZMoy>-1e29,"No ZMoyen in Nadir Masq");


         double aResol = -1e30;
         if (Planimetrie().IsInit())
         {
              cPlanimetrie * anIP  = Planimetrie().PtrVal();
              if (anIP->ResolutionTerrain().IsInit())
              {
                   aResol = StdRound(anIP->ResolutionTerrain().Val()).RVal();
                   double aRR = aResol * mDeZoomMin;
                   aZMoy = round_ni((aZMoy/aRR)) * aRR;
                   anIP->ResolutionTerrain().SetVal(aResol);
              }
         }
         ELISE_ASSERT(aResol>-1e29,"No Resol in Nadir Masq");


// std::cout << "GGGGGGGGGGGg  RR " << aResol  << " ZZZ " << aZMoy << " \n"; getchar();

         mAnamSA = cInterfSurfaceAnalytique::Identite(aZMoy);
         mCorrecAlti4ExportIsInit = true;
         mValmCorrecAlti4Export = aZMoy;
    }
}


void cAppliMICMAC::InitImages() 
{
    if (NomsGeometrieImage().empty())
    {
       if (
               (GeomImages()  == eGeomImage_Epip)
	    || (GeomImages()  == eGeomImage_EpipolairePure)
	    || (mModeAlloc == eAllocAM_Saisie)
	    )
       {
           cNomsGeometrieImage aNGI;
	   aNGI.PatternSel().SetVal(".*");
	   aNGI.PatNameGeom().SetVal("GridDistId");
	   aNGI.AddNumToNameGeom().SetVal(false);
	   NomsGeometrieImage().push_back(aNGI);
       }
       else
       {
          ELISE_ASSERT(false,"NomsGeometrieImage vide,mode non epipolaire");
       }
    }
    for 
    (
         std::list<cNomsGeometrieImage>::iterator itG = NomsGeometrieImage().begin();
         itG != NomsGeometrieImage().end();
         itG++
    )
    {
         mGeoImsComps.push_back(new cGeometrieImageComp(*itG,*this));
    }



    if (Im1().IsInit())
       AddAnImage(Im1().Val());
    if (Im2().IsInit())
       AddAnImage(Im2().Val());
    if (ImSecByDelta().IsInit())
       AddImageByDelta(ImSecByDelta().Val());
    if (FCND_CalcIm2fromIm1().IsInit())
    {
         ELISE_ASSERT(Im1().IsInit(),"No Im1 with FCND_CalcIm2fromIm1");
         AddAnImage
	 (
	     mICNM->Assoc1To1
	     (
	         FCND_CalcIm2fromIm1().Val().I2FromI1Key(),
                 Im1().Val(),
		 FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect()
	     )
	 );
    }


    if (mModeAlloc==eAllocAM_Surperposition)
    {
       if (Im3Superp().IsInit())
          AddAnImage(Im3Superp().Val());
    }
    else 
    {
        ELISE_ASSERT
        (
            !Im3Superp().IsInit(),
            "Im3Superp Init en mode MEC"
        );
    }

    for 
    (
         std::list<string>::const_iterator itP = ImPat().begin();
         itP != ImPat().end();
	 itP++
    )
    {
       PatternAddImages(*itP);
    }


   if (     ImageSecByCAWSI().IsInit()
         && ELISE_fp::exist_file(WorkDir() + ImageSecByCAWSI().Val())
      )
   {
      cChantierAppliWithSetImage aCAWSI = StdGetFromSI(WorkDir()+ ImageSecByCAWSI().Val(),ChantierAppliWithSetImage);
      int aNbPDV = (int)mPrisesDeVue.size();  // Car la taille va augmenter
      for (int aKV=0 ; aKV<aNbPDV ; aKV++)
      {
          const cCWWSImage * aWI = GetFromCAWSI(aCAWSI,mPrisesDeVue[aKV]->Name());
          if (aWI)
          {
              for (std::list<cCWWSIVois>::const_iterator itW=aWI->CWWSIVois().begin() ; itW!=aWI->CWWSIVois().end() ; itW++)
              {
                   AddAnImage(itW->NameVois());
              }
          }
      }
   }
   else if (ImSecCalcApero().IsInit())
   {
       const cImSecCalcApero & aISCA = ImSecCalcApero().Val();
       int aNbPDV = (int)mPrisesDeVue.size();  // Car la taille va augmenter
       int aNbMin = aISCA.NbMin().Val();
       int aNbMax = aISCA.NbMax().Val();
       for (int aKV=0 ; aKV<aNbPDV ; aKV++)
       {
           std::string aNameImSec = WorkDir() + mICNM->Assoc1To1(aISCA.Key(),mPrisesDeVue[aKV]->Name(),true);
           if (ELISE_fp::exist_file(aNameImSec))
           {
                cImSecOfMaster aISOM = StdGetObjFromFile<cImSecOfMaster>
                              (
                                  aNameImSec,
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                 "ImSecOfMaster",
                                 "ImSecOfMaster"
                              );
                 const std::list<std::string > * aList =  GetBestImSec(aISOM,aISCA.Nb().Val(),aNbMin,aNbMax,true);
                 if (aList==0) 
                 {
                     std::cout << "NOT ENOUG IMAGE in ImSecCalcApero for " << aNameImSec << "\n";
                     switch(aISCA.OnEmpty().Val())
                     {
                         case eOEISA_error:
                             ELISE_ASSERT(false,"NOT ENOUG IMAGE is fatal error in this context");
                         break;

                         case eOEISA_exit:
                             exit(0);
                         break;
           
                         case eOEISA_goon:
                         break;
                     }
                 }
                 else
                 {
                      for 
                      (
                           std::list<std::string>::const_iterator itS = aList->begin();
                           itS != aList->end();
                           itS++
                      )
                      {
                          AddAnImage(*itS);
                      }
                 }
            }
       }

   }

   if (RelGlobSelecteur().IsInit())
   {
       int aNbPDV = (int)mPrisesDeVue.size();  // Car la taille va augmenter
       for (int aKV=0 ; aKV<aNbPDV ; aKV++)
       {
           std::vector<std::string> aSBR =  GetStrFromGenStrRel(ICNM(),RelGlobSelecteur().Val(),mPrisesDeVue[aKV]->Name());
           for (int aKV2=0 ; aKV2<int(aSBR.size()) ; aKV2++)
           {
               AddAnImage(aSBR[aKV2]);
           }
       }
   }


   
   if (0)
   {
      for (int aK=0 ; aK<int(mPrisesDeVue.size()) ; aK++)
          std::cout << "====IMAGES : " << mPrisesDeVue[aK]->Name() << "\n";
      getchar();
   }
}


double cAppliMICMAC::AdaptPas(double aPas) const
{
    if ( (! PasIsInPixel().Val()) ||   (mModeGeomMEC != eGeomMECTerrain))
       return aPas;

   ELISE_ASSERT(mMemPart.BSurHGlob().IsInit(),"cAppliMICMAC::AdaptPas");

   double aRes =  aPas / mMemPart.BSurHGlob().Val();

   return aRes;
}


// void Debug(const cAppliMICMAC & anAp,cGeomImage *aGeom);

void cAppliMICMAC::PostInitGeom()
{
     bool DoBSurHGlob = (!CalledByProcess().Val()) &&(PasIsInPixel().Val()) && (!mMemPart.BSurHGlob().IsInit());
     double aSomBSH = 0;
     double aSomSurf = 0;


     for (tIterPDV itFI1=mPrisesDeVue.begin(); itFI1!=mPrisesDeVue.end(); itFI1++)
     {
         cGeomImage * aGeom1 = & ((*itFI1)->Geom());
         aGeom1->PostInit();
         
         cGeomImage * aGeomT = aGeom1->NC_GeoTerrainIntrinseque();
         if (aGeomT != aGeom1)
         {
            aGeomT->PostInit();
         }

         if (DoBSurHGlob)
         {
               // std::cout << "BSurSGlob:: "  << (*itFI1)->Name() <<  "\n";
               for (tIterPDV itFI2=mPrisesDeVue.begin(); itFI2!=itFI1; itFI2++)
               {
                   cGeomImage * aGeom2 = & ((*itFI2)->Geom());

                   Pt2dr aPTer;
                   double aSurf;
                   if (aGeom1->IntersectEmprTer(*aGeom2,aPTer,&aSurf))
                   {
                       double aBSH = aGeom1->BSurH(*aGeom2,aPTer);
                       aSomSurf += aSurf;
                       aSomBSH += aSurf * aBSH;
                   }
               }
         }

     }


     if (DoBSurHGlob)
     {
        ELISE_ASSERT(aSomSurf, "No Cple for computing B/H");
        mMemPart.BSurHGlob().SetVal( aSomBSH / aSomSurf);
        SauvMemPart();
     }
      // Debug(*this,&(mPDV2->Geom()));
}

void cAppliMICMAC::AddImageByDelta(const cListImByDelta & aLIBD)
{
   std::list<std::string> aL = mICNM->GetListImByDelta(aLIBD,PDV1()->Name());
   for (std::list<std::string>::const_iterator iT=aL.begin() ; iT!=aL.end() ; iT++)
   {
        if (ELISE_fp::exist_file(WorkDir()+*iT))
        {
            AddAnImage(*iT); 
        }
   }
}

void cAppliMICMAC::PatternAddImages(const std::string & aPat)
{
/*
    std::list<std::string> aList = RegexListFileMatch(mDirImagesInit,aPat,1,false);
    std::list<std::string > aLName2Add = mICNM->StdGetListOfFile(aPat);
    ELISE_ASSERT(aList.size()==aLName2Add.size(),"cAppliMICMAC::PatternAddImages");
*/

    std::list<std::string > aList = mICNM->StdGetListOfFile(aPat,1);
    for (std::list<std::string>::iterator iT =aList.begin() ; iT!=aList.end() ; iT++)
    {
        if (NameFilter(mICNM,Images().Filter(),*iT))
        {
           AddAnImage(*iT);
        }
    }
}

void cAppliMICMAC::AddAnImage(const std::string & aName)
{
// std::cout << "XXXXjjjI " << aName << "\n";

     if (PDVFromName  (aName,0))
        return;


     if (CreateGrayFileAtBegin().Val())
     {
         Tiff_Im::StdConvGen(WorkDir() + aName, 1,true,true);
     }

     std::string  aNameGeom;
     cGeometrieImageComp * theGotGeom = 0;
     for
     (
        std::list<cGeometrieImageComp *>::iterator itG = mGeoImsComps.begin();
        (itG!= mGeoImsComps.end()) && (theGotGeom==0);
        itG++
     )
     {

           if ((*itG)->AcceptAndTransform(aName,aNameGeom,mNbPDV))
               theGotGeom = *itG;
     }


    if (theGotGeom==0)
    {
        std::cout << "IMAGE = [" << aName <<"]\n";

        ELISE_ASSERT
        (
           theGotGeom !=0 ,
           "Aucun pattern  appariable dans <GeometrieImage><PatternSel>"
        );
    }
    
    if (AutoSelectionneImSec().IsInit())
    {
        ELISE_ASSERT(mICNM!=0,"Use FCND_GeomCalc withe AutoSelectionneImSec");
        //
        //
        const cAutoSelectionneImSec & aSel = AutoSelectionneImSec().Val();
        if ( mNbPDV > 0)
        {
            std::string aN1 = WorkDir()+ mPDV1->NameGeom();
            std::string aN2 = WorkDir()+ aNameGeom;
            ElCamera * aCam1 = Cam_Gen_From_File(aN1,"OrientationConique",mICNM);
            ElCamera * aCam2 = Cam_Gen_From_File(aN2,"OrientationConique",mICNM);
            double aProp = aCam1->RatioInterSol(*aCam2);
            // std::cout << aN1 << " " << aN2 << " " << aProp << "\n";
            if (aProp < aSel.RecouvrMin())
               return;

/*
            Ori3D_Std anO1(aN1.c_str());
            Ori3D_Std anO2(aN2.c_str());
            std::string  aN0 =  mPDV1->Name();
            if (anO1.PropInter(anO2) < aSel.RecouvrMin())
               return;
*/
            // std::cout  << aN0 << " " << aName <<  "  " << anO1.PropInter(anO2) << "\n";
        }
    }

     cInterfModuleImageLoader * aIMIL = GetMIL(theGotGeom,aName);

     mPrisesDeVue.push_back( new cPriseDeVue( *this,
                                              aName,
                                              aIMIL,
                                              mNbPDV,
                                              ( isUsingSeparateDirectories()?MMOutputDirectory()+aNameGeom:aNameGeom ),
                                              theGotGeom->ModG() ) );

// InitAnam
     if (mAnamSA)
     {
         ELISE_ASSERT
         (
              mPrisesDeVue.back()->Geom().AcceptAnamSA(),
              "Acquisition Geometrie do not handle anamorphose\n"
         );
     }
     if ( mNbPDV == 0)
         mPDV1 =  mPrisesDeVue.back();
     if ( mNbPDV== 1)
     {
         // std::cout << "PDVVVVVV  22222222222222\n"; getchar();
         mPDV2 =  mPrisesDeVue.back();
     }
     
     int aDim = mPrisesDeVue.back()->Geom().DimPx();
     if (mNbPDV==0)
     {
        mDimPx = aDim;
     }
     else
     {
        ELISE_ASSERT
        (
            mDimPx == aDim,
            "Geometrie incoherente, Dim de Px variable"
        );
     }
     mNbPDV++;
}

cPriseDeVue * cAppliMICMAC::NC_PDVFromName  
                    (
                                const std::string  & aName,
                                const char * aMesErreur
                    ) 
{
   return const_cast<cPriseDeVue *> (PDVFromName(aName,aMesErreur));
}


const cPriseDeVue * cAppliMICMAC::PDVFromName  
                    (
                                const std::string  & aName,
                                const char * aMesErreur
                    ) const

{
     for (tCsteIterPDV itFI=mPrisesDeVue.begin(); itFI!=mPrisesDeVue.end(); itFI++)
     {
           if (aName == (*itFI)->Name())
	   {
	      return *itFI;
	   }
     }
     ELISE_ASSERT(!aMesErreur,aMesErreur);
     return 0;
}

int  cAppliMICMAC::CurDCAllDC() const
{
   return     (OneDefCorAllPxDefCor().Val())
          &&  (mCurEtape->DeZoomTer() <=ZoomBeginODC_APDC().Val());
}


void cAppliMICMAC::SetContourSpecIm1(const std::vector<Pt2dr> & aCont)
{
   mFullIm1         = true;
   mUseConstSpecIm1 = true;
   mContSpecIm1     = aCont;
}

bool cAppliMICMAC::UseConstSpecIm1() const
{
  return mUseConstSpecIm1;
}

const std::vector<Pt2dr>  & cAppliMICMAC::ContSpecIm1() const
{
   return mContSpecIm1;
}

cEl_GPAO *    cAppliMICMAC::GPRed2() const
{
   return mGPRed2;
}

bool   cAppliMICMAC::CMS_ModeEparse() const
{
    return   mCMS_ModeEparse;
}



/*****************************************/
/*       Valeur Speciale Not Image       */
/*****************************************/

bool  cAppliMICMAC::HasVSNI() const
{
    return ValSpecNotImage().IsInit();
}

int   cAppliMICMAC::VSNI() const
{
   return HasVSNI() ? ValSpecNotImage().Val() : 0;
}

/*****************************************/
/*       ACCESSEURS                      */
/*****************************************/


std::string cAppliMICMAC::NameClassEquiv(const std::string & aName) const
{
   if (mMapEquiv==0) return "XXX";

   std::string aRes = mMapEquiv->map(aName);
   // std::cout << "**************** MAP["<<aName << "]=" << aRes << " **************\n";
   return aRes;
}

const double &  cAppliMICMAC::DefCost() const
{
   return mDefCost;
}

const std::string & cAppliMICMAC::DirImagesInit() const
{
    return mDirImagesInit;
}

const std::string & cAppliMICMAC::DirMasqueIms() const
{
    return mDirMasqueIms;
}

eModeGeomMEC cAppliMICMAC::ModeGeomMEC() const
{
   return mModeGeomMEC;
}

const std::string & cAppliMICMAC::FileBoxMasqIsBoxTer() const
{
   return mFileBoxMasqIsBoxTer;
}
const std::string & cAppliMICMAC::FullDirMEC() const
{
   return mFullDirMEC;
}
const std::string & cAppliMICMAC::FullDirPyr() const
{
   return mFullDirPyr;
}
const std::string & cAppliMICMAC::FullDirGeom() const
{
   return mFullDirGeom;
}
const std::string & cAppliMICMAC::FullDirResult() const
{
   return mFullDirResult;
}

int cAppliMICMAC::NbPDV() const
{
   return mNbPDV;
}

const cExportApero2MM & cAppliMICMAC::ExpA2Mm() const {return mExpA2Mm;}

inline cPriseDeVue * PDVNN(const cPriseDeVue * aPDV,int aNum)
{
   if (aPDV==0)
   {
       std::cout << "FOR VIEW NUM = " << aNum << "\n";
       ELISE_ASSERT(false,"View do not exist");
   }
   return const_cast<cPriseDeVue *>(aPDV);
}


const cEtapeMecComp * cAppliMICMAC::FirstVraiEtape() const
{
   ELISE_ASSERT(!mEtapesMecComp.empty(),"cAppliMICMAC::FirstVraiEtape");
   return mEtapesMecComp.front();
}



const cPriseDeVue * cAppliMICMAC::PDV1() const { return  PDVNN(mPDV1,1); }
const cPriseDeVue * cAppliMICMAC::PDV2() const { return  PDVNN(mPDV2,2); }
cPriseDeVue * cAppliMICMAC::PDV1() { return  PDVNN(mPDV1,1); }
cPriseDeVue * cAppliMICMAC::PDV2() { return  PDVNN(mPDV2,2); }

std::vector<cPriseDeVue *>  cAppliMICMAC::AllPDV()
{
   return std::vector<cPriseDeVue *>(mPrisesDeVue.begin(),mPrisesDeVue.end());
}

double cAppliMICMAC::CurCorrelToCout(double aCor) const
{
   ELISE_ASSERT(mStatGlob!=0,"cAppliMICMAC::CurCorrelToCout");
   return mStatGlob->CorrelToCout(aCor);
}

int cAppliMICMAC::NbVueAct() const
{
   return (int)mPDVBoxGlobAct.size();
}

int cAppliMICMAC::NumImAct2NumImAbs(int aNum) const
{
   return mPDVBoxGlobAct.at(aNum)->Num();
}

tCsteIterPDV cAppliMICMAC::PdvBegin() const
{
   return mPrisesDeVue.begin();
}

tCsteIterPDV cAppliMICMAC::PdvEnd()  const
{
   return mPrisesDeVue.end();
}

int cAppliMICMAC::DimPx() const
{
   return mDimPx;
}

int cAppliMICMAC::NbPdv() const
{
   return mNbPDV;
}

const cEtapeMecComp * cAppliMICMAC::CurEtape() const
{
  return mCurEtape;
}

cCaracOfDeZoom * cAppliMICMAC::GetCurCaracOfDZ() const
{
   return mCurCarDZ;
}

cSurfaceOptimiseur *     cAppliMICMAC::SurfOpt()
{
   return mSurfOpt;
}

cStatGlob  * cAppliMICMAC::StatGlob() {return mStatGlob;}



const cGeomDiscFPx &  cAppliMICMAC::GeomDFPx() const
{
  return *mGeomDFPx;
}

const cGeomDiscFPx &  cAppliMICMAC::GeomDFPxInit() const
{
  return *mGeomDFPxInit;
}

const Pt2di &  cAppliMICMAC::PtSzWFixe() const
{
  return mPtSzWFixe;
}


const cCorrelAdHoc * cAppliMICMAC::CAH() const
{
    return mCorrelAdHoc;
}

const cCorrelMultiScale * cAppliMICMAC::CMS() const
{
   return mCMS;
}

const cCensusCost * cAppliMICMAC::CC() const
{
   return mCC;
}



const cEtiqBestImage *  cAppliMICMAC::EBI() const
{
   return mEBI;
}

const Pt2dr &  cAppliMICMAC::PtSzWReelle() const
{
  return mSzWR;
}

const Pt2di &  cAppliMICMAC::PtSzWMarge() const
{
  return mPtSzWMarge;
}

int  cAppliMICMAC::NbPtsWFixe() const
{
  return mNbPtsWFixe;
}

int  cAppliMICMAC::SzWFixe() const
{
  return mSzWFixe;
}


int cAppliMICMAC::FreqPtsInt() const
{
   return mFreqPtsInt;
}

int cAppliMICMAC::DeZoomMax() const
{
   return mDeZoomMax;
}

int cAppliMICMAC::DeZoomMin() const
{
   return mDeZoomMin;
}

int   cAppliMICMAC::CurNbIterFenSpec() const
{
   return mCurNbIterFenSpec;
}

bool cAppliMICMAC::CurWSpecUseMasqGlob() const
{
   return mCurWSpecUseMasqGlob;
} 


const std::string & cAppliMICMAC::NameChantier() const
{
   return mNameChantier;
}

const cLoadTer * cAppliMICMAC::LoadTer() const
{
   return mLTer;
}

bool cAppliMICMAC::AucuneGeom() const
{
    return mAucuneGeom;
}

eModeAllocAM cAppliMICMAC::ModeAlloc() const
{
    return mModeAlloc;
}

const double  & cAppliMICMAC::RecouvrementMinimal() const
{
   return mRecouvrMin;
}

const std::string & cAppliMICMAC::NameSpecXML() const
{
   return mNameSpecXML;
}

bool cAppliMICMAC::IsOptDiffer() const
{
   return mIsOptDiffer;
}
bool cAppliMICMAC::IsOptimCont() const
{
   return mIsOptimCont;
}
bool cAppliMICMAC::IsOptDequant() const
{
   return mIsOptDequant;
}
bool cAppliMICMAC::IsOptIdentite() const
{
   return mIsOptIdentite;
}

int cAppliMICMAC::CurSurEchWCor() const
{
   return mCurSurEchWCor;
}

bool cAppliMICMAC::DoTheMEC() const
{
   return mDoTheMEC;
}

const cInterfaceVisualisation * cAppliMICMAC::PtrVI() const
{
   return mPtrIV;
}

bool cAppliMICMAC::FullIm1() const
{
   return mFullIm1;
}

cInterfSurfaceAnalytique * cAppliMICMAC::AnamSA() const { return mAnamSA; }
cXmlOneSurfaceAnalytique * cAppliMICMAC::XmlAnamSA() const { return mXmlAnamSA; }
const cChCoCart  * cAppliMICMAC::RC() const { return mRepCorrel; }
const cChCoCart  * cAppliMICMAC::RCI() const { return mRepInvCorrel; }


eTypeWinCorrel cAppliMICMAC::CurTypeWC() const      {return mCurTypeWC;}
Pt2dr          cAppliMICMAC::FactFenetreExp() const {return mFactFenetreExp;}
bool cAppliMICMAC::WM() const {return mWM;}

cInterfChantierNameManipulateur * cAppliMICMAC::ICNM() const 
{
   ELISE_ASSERT(mICNM!=0,"cAppliMICMAC::ICNM");
   return mICNM;
}

bool cAppliMICMAC::UseAlgoSpecifCorrelRect() const
{
   if (mCurEtape->UsePC())
      return false;
   if (mMapEquiv !=0)
      return false;
   if (mCurFenSpec)
      return true;
   int aNbIm = (int) mPDVBoxInterneAct.size();

   if (mModeIm1Maitre)
   {
      return    (aNbIm <=2)
             && (mSzWFixe==mCurSzWInt);
   }

   return    (aNbIm >=2)
          && (aNbIm <=theNbImageMaxAlgoRapide)
          && (mSzWFixe==mCurSzWInt);
}


std::string cAppliMICMAC::NamePackHom
            (
                const std::string & aName1,
                const std::string & aName2
            ) const
{
   if (FCND_CalcHomFromI1I2().IsInit())
   {
       return mICNM->NamePackWithAutoSym(FCND_CalcHomFromI1I2().Val(),aName1,aName2,true);
      // return mICNM->Assoc1To2(FCND_CalcHomFromI1I2().Val(),aName1,aName2,true);
   }

   ELISE_ASSERT
   (
      NomsHomomologues().IsInit(),
      "Pas trouve NomsHomomologues"
   );

   return StdNameFromCple
          (
                   mAutomNomsHoms,
                   NomsHomomologues().Val().PatternSel(),
                   NomsHomomologues().Val().PatNameGeom(),
                   NomsHomomologues().Val().SeparateurHom().Val(),
                   aName1,
                   aName2
          );
}

std::string  cAppliMICMAC::NameFilePyr
             (
                  const std::string & aName1,
                  int   aDZ
             ) const
{
   std::string aKey =  KeyCalNamePyr().Val();
   if (aKey !="")
      return mICNM->Assoc1To2(aKey,aName1,ToString(aDZ),true);
   return StdNameFromCple
          (
                   mAutomNomPyr,
                   PatternSelPyr().Val(),
                   PatternNomPyr().Val(),
                   SeparateurPyr().Val(),
                   aName1,
                   ToString(aDZ)
          );
}

void  cAppliMICMAC::SetLastMAnExp(cModeleAnalytiqueComp * aMAC)
{
    mLastMAnExp = aMAC;
}

cModeleAnalytiqueComp &  cAppliMICMAC::LastMAnExp()
{
   ELISE_ASSERT(mLastMAnExp!=0,"cAppliMICMAC::LastMAnExp");
   return *mLastMAnExp;
}



void cAppliMICMAC::ExeProcessParallelisable
     (
          bool AddNameExeMicMac,
          const  std::list<std::string> & aLProc
     )
{
   if (aLProc.empty())
      return;

   // Modification pour la gestion des espaces dans les nom de repertoire (GM)
   // Remodifie pour fonctionnement sous windows (DB)
   std::string ToAdd("");
   if (AddNameExeMicMac)
   {
#if (ELISE_unix || ELISE_MacOs)
	  // TODO: a better solution for this
	  std::size_t spacePos = mNameExe.rfind( ' ' );
	  if ( spacePos!=std::string::npos )
	  {
		  std::string mmd3_full_name = mNameExe.substr( 0, spacePos );
		  ToAdd = std::string("\"")+mmd3_full_name+std::string("\" MICMAC");  
	  }
	  else
		ToAdd = std::string("\"")+mNameExe+std::string("\"");
#else
		ToAdd = mNameExe;  
#endif
   }

   //std::string ToAdd = AddNameExeMicMac ? (mNameExe ) : "";
   std::string nomAvancement = std::string("\"") + WorkDir()+std::string("avancement-")+NameChantier()+std::string(".txt\"");
   // Version sans parallelisation
   // si ByProcess == 0, on peut quand meme
   // se retrouver la pour des generation de resultats faisant
   // appel a des process extern (to8bits, GrShade ...)
   //
   // Modif MPD : pour ne pas passser par le Makefile quand ByProcess=1
   // (inutile et incompatible Windows)
   if ((ByProcess().Val() == 0) || (ByProcess().Val() == 1))
   {
	int num=0;
	int nbDalles = (int)aLProc.size();
#ifdef MAC
	utsname buf;
	uname(&buf);
	//std::cout << "Nom de machine : " << buf.nodename << std::endl;
#endif
       for
       (
          std::list<std::string>::const_iterator itStr=aLProc.begin();
          itStr!=aLProc.end();
          itStr++,++num
       )
       {
           std::string commande = ToAdd  + " "+ (*itStr);
         mCout << " ---Launch Process="<<commande<< "\n";  
	int aCodeRetour = system_call(commande.c_str());
         if (StopOnEchecFils().Val())
         {
             ELISE_ASSERT(aCodeRetour==0,"Error in child process");
         }
         mCout << " ---End Process\n";

	 if (mWM)
	 {
#ifdef MAC
	    std::string commande_avancement = std::string("echo \"<Top><Machine>")+std::string(buf.nodename)+std::string("</Machine><NumEtape>")
	                                   +ToString(mCurEtape->Num())
                                           +std::string("</NumEtape><NbEtapes>5</NbEtapes><NumTache>")
					   +ToString(num)+std::string("</NumTache><NbTaches>")
					   +ToString(nbDalles)+std::string("</NbTaches></Top>\" >> ")+nomAvancement;
#else
	    std::string commande_avancement = std::string("echo \"<Top><NumEtape>")
                                           +ToString(mCurEtape->Num())
                                           +std::string("</NumEtape><NbEtapes>5</NbEtapes><NumTache>")
                                           +ToString(num)+std::string("</NumTache><NbTaches>")
                                           +ToString(nbDalles)+std::string("</NbTaches></Top>\" >> ")+nomAvancement;
#endif
       	    VoidSystem(commande_avancement.c_str());
	 }
	}
       return;
  }
  // Version parallelisation avec makefile
  // la valeur absolue donne le nombre de job en parallele dans le Makefile (option -j)
  // si ByProcess().Val()>0 --> version classique
  // si ByProcess().Val()<0 --> version qsub (pour le cluster) ou condor
  // Pour la version qsub on utilise un script qsub_synchrone.sh pour 
  // avoir une attente passive lors de la soumission d'un job
  else if (ByProcess().Val()>0)
  {
#ifdef MAC
	utsname buf;
        uname(&buf);
#endif
      // creation d'un Makefile
      // Modif MPD, pour risque potentiel de crash sur MicMac concurent
      std::string nomMakefile = ( isUsingSeparateDirectories()?MMTemporaryDirectory():WorkDir()+TmpMEC() )+std::string("MakefileParallelisation") + GetUnikId();
      std::ofstream fic(nomMakefile.c_str());
      int nbDalles = 0;
      //int numEtape = mCurEtape->Num();
      fic << "all : ";
      for
      (
           size_t i=0;
           i < aLProc.size();
           ++i
       )
       {
           ++nbDalles;
            fic << "Box0Step0_"<< (unsigned int) i<<" ";
       }
       fic << std::endl;
 
#if (ELISE_windows)
	// the 'make' program takes a line beginning by a word + ':' for a rule
	// and Windows' absolute paths match this expression, we need to protect such lines by double quotes
	string mm3dFullName = mNameExe.substr( 0, mNameExe.length()-7 );
	if ( ( ToAdd.length()>1 ) && ( ToAdd[1]==':' ) )
		ToAdd = std::string( "\"" )+mm3dFullName+"\" MICMAC";
#endif

       int num=0;
       for 
       (
           std::list<std::string>::const_iterator itStr=aLProc.begin();
           itStr!=aLProc.end();
           itStr++,++num
       )
       {
           fic << "Box0Step0_"<<ToString(num)<<":"<< std::endl;
	   fic << "\t"<<ToAdd<<" "<<*itStr<<std::endl;
           if (mWM)
           {
#ifdef MAC
		fic << "\techo \"<Top><Machine>"<<buf.nodename<<"</Machine><NumEtape>"
                    << ToString(mCurEtape->Num())
                    << std::string("</NumEtape><NbEtapes></NbEtapes><NumTache>")
                    << ToString(num) << std::string("</NumTache><NbTaches>")
                    << ToString(nbDalles) << std::string("</NbTaches></Top>\" >> ") << nomAvancement << std::endl;
#else
		fic << "\techo \"<Top><NumEtape>"
                    << ToString(mCurEtape->Num())
                    << std::string("</NumEtape><NbEtapes></NbEtapes><NumTache>")
                    << ToString(num) << std::string("</NumTache><NbTaches>")
                    << ToString(nbDalles) << std::string("</NbTaches></Top>\" >> ") << nomAvancement << std::endl;
#endif
               //fic << "\techo Etape "<<ToString(numEtape)
               //    <<" Dalle "<<ToString(num)<<" / "<<ToString(nbDalles)
               //    <<" : Done >> "<< nomAvancement << std::endl;
           }
       } 
       fic.close();
       mCout << " ---Launch processes through the Makefile\n";

	bool makeSucceeded = launchMake( nomMakefile, "", ByProcess().Val() );

       if (StopOnEchecFils().Val())
        {    
            ELISE_ASSERT(makeSucceeded,"Error in child process");
        }
	    mCout << " ---End Process\n";
    }//else 
   else 
   {
       //std::string nomAvancement = WorkDir()+std::string("avancement-")+NameChantier()+std::string(".dot");
       int numEtape = mCurEtape->Num();
	// Modif Greg version sans dag suite a des problemes NFS
	// creation du fichiers de commandes
	std::string nomCmd = WorkDir()+std::string("Micmac_")+ToString(numEtape)+std::string(".cmd");
       std::ofstream fic(nomCmd.c_str(),ios_base::app);
       int num=0;
       for 
           (
            std::list<std::string>::const_iterator itStr=aLProc.begin();
            itStr!=aLProc.end();
            itStr++,++num
           )
           {
               fic << "arguments=\""<<*itStr<<"\""<< std::endl;
	       fic << "queue"<<std::endl; 
           } 
       fic.close();

	/*       
       // creation du DAG
       std::string nomDag = WorkDir()+std::string("Micmac_")+ToString(numEtape)+std::string(".dag");
       std::ofstream fic(nomDag.c_str(),ios_base::app);
       int num=0;
       for 
           (
            std::list<std::string>::const_iterator itStr=aLProc.begin();
            itStr!=aLProc.end();
            itStr++,++num
           )
           {
               fic << "JOB job_"<<ToString(numEtape)<<"_"<<ToString(num)<<" MICMAC.cmd"<< std::endl;
               fic << "VARS job_"<<ToString(numEtape)<<"_"<<ToString(num)<<" Param=\""<<*itStr<<"\""<< std::endl; 
           } 
       fic.close();
	*/
       mCout << " ---End Process\n"; 
   }
}

bool  cAppliMICMAC::InversePx() const
{
    return mInversePx;
}


/*****************************************/
/*                                       */
/*   FONCTION UTILES POUR TEST ET        */
/*   MISE AU POINT                       */
/*                                       */
/*****************************************/

// Pour forcer la creation d'une pyramide
void cAppliMICMAC::TestReducIm(int aDZ)
{
// std::cout << "PDVS " << mPrisesDeVue.size() << "\n";
     for 
     (
          tCsteIterPDV itFI=mPrisesDeVue.begin(); 
          itFI!=mPrisesDeVue.end(); 
          itFI++
     )
     {
          if ((! DoNothingBut().IsInit()) || ButDoPyram().Val())
	  {
	      (*itFI)->Geom();
	      (*itFI)->IMIL()->PreparePyram(aDZ);
	  }
    }

    if (mGPRed2)
    {
       // GetUnikId : sinon pb avec le nouveau launchmake ...
       mGPRed2->ExeParal(mFullDirMEC + "MkRed2MM" + GetUnikId() ,-1);
    }



    for 
    (
          tCsteIterPDV itFI=mPrisesDeVue.begin(); 
          itFI!=mPrisesDeVue.end(); 
          itFI++
    )
    {
          if ((! DoNothingBut().IsInit()) || ButDoMasqIm().Val())
	  {
              (*itFI)->FileImMasqOfResol(aDZ);
	  }
     }
}


void cAppliMICMAC::ShowEtapes() const
{
     for
     (
        tContEMC::const_iterator itE = mEtapesMecComp.begin();
        itE != mEtapesMecComp.end();
        itE++
     )
        (*itE)->Show();
}

#if ELISE_windows
void * cAppliMICMAC::AllocObjFromLibDyn
       (
           const std::string & aNameLibraire,
           const std::string & aNameSymb
       ) const
{
  ELISE_ASSERT
  (
       false,
       "Pas de chargement dynamique sous Windows"
  );

   return 0;
}
#else
#include <dlfcn.h>

void * cAppliMICMAC::AllocObjFromLibDyn
       (
           const std::string & aNameLibraire,
           const std::string & aNameSymb
       ) const
{
   void * handle = dlopen(aNameLibraire.c_str(),RTLD_LAZY|RTLD_GLOBAL);
   if (!handle)
   {
         std::cout << "Erreur dans le chargement du module : "
                   <<aNameLibraire<<std::endl;
         const char * aMes = dlerror();
         std::cout << "dlerror()=["<<(aMes?aMes:"???")<< "]\n";
         ELISE_ASSERT(false,"dlopen");
         return 0;
   }
   void * anObj = dlsym(handle,aNameSymb.c_str());
   if (!anObj)
   {
       std::cout << "Erreur, pas de symbole" << aNameSymb
                 << " dans le module : "<< aNameLibraire
                 <<std::endl;
         ELISE_ASSERT(false,"dlsym");
         return 0;
   }
   return anObj;
}
#endif

int cAppliMICMAC::CodeMicMax2CodeExt(eMicMacCodeRetourErreur aCode) const
{
   return aCode+BaseCodeRetourMicmacErreur().Val();
}
void cAppliMICMAC::MicMacErreur
     (
          eMicMacCodeRetourErreur aCode,
          const std::string & aMes,
          const std::string & aDiag
     ) const
{
   std::cerr << "Une erreur MicMac s'est produite\n";
   std::cerr << "Code erreur " << CodeMicMax2CodeExt(aCode) << "\n";
   std::cerr << "Message interne : " << aMes << "\n";
   if (aDiag != "")
   {
        std::cerr << "Origine probable : " << aDiag << "\n";
   }

   exit(CodeMicMax2CodeExt(aCode));
}

void cAppliMICMAC::AnalyseOri(CamStenope * aCam ) const
{
  Ori3D_Std *  anOri = aCam->CastOliLib();
  if (! ConvertToSameOriPtTgtLoc().ValWithDef(anOri!=0))
  {
     return;
  }
  ELISE_ASSERT(anOri!=0,"ConvertToSameOriPtTgtLoc needs Ori format");
  if (mOriPtLoc_Read)
  {
     anOri->SetOrigineTgtLoc(mOriPtLoc);
     // OO anOri.SetOrigineTgtLoc(mOriPtLoc);
/*
     ELISE_ASSERT
     (
          euclid(mOriPtLoc,anOri.OrigineTgtLoc()) < 1e-5,
          "Ne gere pas Origine Multiple du Rep Tgt Loc"
     );
*/
  }
  else
  {
     mOriPtLoc_Read = true;
     // OO mOriPtLoc= anOri.OrigineTgtLoc();
     mOriPtLoc= anOri->OrigineTgtLoc();
  }
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
