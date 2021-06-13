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
#include <algorithm>
#include "TapasCampari.h"

void UseRegulDist(std::vector<double> & aVRegulDist,std::string & aCom)
{
   if (EAMIsInit(&aVRegulDist))
   {
      ELISE_ASSERT(aVRegulDist.size()>=3,"Not enough parameter in RegulDist")
      double aNbCase = (aVRegulDist.size() >= 4) ? round_ni(aVRegulDist[3])  : 7;
      double aSeuilNbPts = (aVRegulDist.size() >= 5) ? aVRegulDist[4]  : 5.0;
      aCom = aCom  + std::string(" +UseRegulDist=true")
                   + std::string(" +RegDist0=") + ToString(aVRegulDist[0])
                   + std::string(" +RegDist1=") + ToString(aVRegulDist[1])
                   + std::string(" +RegDist2=") + ToString(aVRegulDist[2])
                   + std::string(" +RegDistNbCase=") + ToString(aNbCase)
                   + std::string(" +RegDistSeuil=") + ToString(aSeuilNbPts);
   }
}


class cMemRes
{
   public :
        cMemRes()
        {
            mVAll.reserve(1000000);
        }

        void Init(int aSzMax,int aValInit)
        {
           double aMemTot = 0;
           for (int aExpSz=1 ; aExpSz<=aSzMax ; aExpSz++)
           {
               // int aNbAll = pow(1+aSzMax-aExpSz,4);
               int aNbAll = pow(1.6,aSzMax-aExpSz);
               int aSz = 1 << aExpSz;
               aMemTot += aSz * aNbAll;
               // std::cout << "Mem " << aMemTot << "\n";

               for (int aNb=0 ; aNb<aNbAll ; aNb++)
               {
                  char * aMem = (char *) malloc(aSz);
                  memset(aMem,aValInit,aSz);
                  mVAll.push_back(aMem);
               }
           }
        }
        void Free()
        {
           while (! mVAll.empty())
           {
              free(mVAll.back());
              mVAll.pop_back();
           }
       }
   private:
        std::vector<char *> mVAll;
};








/*
Parametre de Tapas :

   - calibration In : en base de donnees ou deja existantes.


*/

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876



void Tapas_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************************\n";
    std::cout <<  " *     T-ool for                             *\n";
    std::cout <<  " *     A-erotriangulation                    *\n";
    std::cout <<  " *     P-hotogrammetric with                 *\n";
    std::cout <<  " *     A-pero                                *\n";
    std::cout <<  " *     S-implified (hopefully...)            *\n";
    std::cout <<  " *********************************************\n\n";
}


#define  NbModele 30

const char * Modele[NbModele] = {
                                   "RadialBasic",     // 0
                                   "RadialExtended",  // 1
                                   "Fraser",          // 2
                                   "FishEyeEqui",     // 3
                                   "AutoCal",         // 4
                                   "Figee",           // 5
                                   "HemiEqui",        // 6
                                   "RadialStd",       // 7
                                   "FraserBasic",     // 8
                                   "FishEyeBasic",    // 9
                                   "FE_EquiSolBasic",  // 10
                                   "Four7x2",          // 11
                                   "Four11x2",         // 12
                                   "Four15x2",         // 13
                                   "Four19x2",         // 14
                                   "AddFour7x2",          // 15
                                   "AddFour11x2",         // 16
                                   "AddFour15x2",         // 17
                                   "AddFour19x2",         // 18
                                   "AddPolyDeg0",          // 19
                                   "AddPolyDeg1",          // 20
                                   "AddPolyDeg2",          // 21
                                   "AddPolyDeg3",          // 22
                                   "AddPolyDeg4",          // 23
                                   "AddPolyDeg5",          // 24
                                   "AddPolyDeg6",          // 25
                                   "AddPolyDeg7",          // 26
                                   "Ebner",                // 27
                                   "Brown",                 // 28
                                   "FishEyeStereo"          // 29
                                };



void ShowAuthorizedModel()
{
   std::cout << "\n";
   std::cout << "Authorized models : \n";
   for (int aKM=0 ; aKM<NbModele ; aKM++)
       std::cout << "   " << Modele[aKM] << "\n";
}

std::list<std::string> cAppli_Tapas_Campari::GetAuthorizedModel()
{
    std::list<std::string> list;
    for (int aKM=0 ; aKM<NbModele ; aKM++)
        list.push_back(std::string(Modele[aKM]));
    return list;
}


void cAppli_Tapas_Campari::InitVerifModele(const std::string & aMod,cInterfChantierNameManipulateur *)
{

    int aKModele = -1;

    for (int aK=0 ; aK<NbModele ; aK++)
       if (aMod==Modele[aK])
         aKModele = aK;

    if (aKModele==-1)
    {
        ShowAuthorizedModel();
        ELISE_ASSERT(false,"Value is not a correct model\n");
    }

    if (aMod==Modele[0])  // RadialBasic
    {
       eModAutom = "eCalibAutomRadialBasic";

       LocDegGen = 0;
       LocLibDec = false;
       LocDRadMaxUSer = 2;
       LocLibCD = false;
    }
    else if ((aMod==Modele[1]) ||  (aMod==Modele[7]))  // RadialExtended +  RadialStd
    {
       LocDegGen = 0;
       LocLibDec = false;
       LocDRadMaxUSer = (aMod==Modele[1]) ? 5 : 3;
       eModAutom = "eCalibAutomRadial";
    }
    else if (aMod==Modele[2])  //  Fraser
    {
        LocDegGen = 1;
        LocDRadMaxUSer = 3;
        eModAutom = "eCalibAutomPhgrStd";
    }
    else if ((aMod==Modele[3]) || (aMod==Modele[6]))   //   FishEyeEqui +  HemiEqui
    {
        LocDegGen = 2;
        LocDRadMaxUSer = 5;

        eModAutom = "eCalibAutomFishEyeLineaire";
        if (aMod==Modele[6])
        {
            if (PropDiag<0)
               PropDiag = 0.52;
        }
    }
    else if ((aMod==Modele[9]) || (aMod==Modele[10]) ||  (aMod==Modele[29]) ) // "FishEyeBasic" +  "FE_EquiSolBasic"
    {
        LocDegGen = 1;
        LocDRadMaxUSer = 3;
        LocLibDec = false;

        eModAutom = "eCalibAutomFishEyeLineaire";
        if (aMod==Modele[10])
           eModAutom = "eCalibAutomFishEyeEquiSolid";
        else if (aMod==Modele[29])
           eModAutom = "eCalibAutomFishEyeStereographique";
    }
    else if ((aMod==Modele[4]) || (aMod==Modele[5])) // AutoCal  +  Figee
    {
        LocDegGen  = 0;
        LocLibDec = false;
        LocLibCD= false;
        LocDRadMaxUSer = 0;
        LocLibPP  =false;
        LocLibFoc=false;

        IsAutoCal = (aMod==Modele[4]);
        IsFigee   = (aMod==Modele[5]);

        eModAutom = "eCalibAutomNone";
    }
    else if (aMod==Modele[8])  //  FraserBasic
    {
        eModAutom = "eCalibAutomPhgrStdBasic";
        LocDegGen = 1;
        LocLibDec = true;
        LocDRadMaxUSer = 3;
    }
    // Four7x2  => Four19x2
    else if (     (aMod==Modele[11])
              ||  (aMod==Modele[12])
              ||  (aMod==Modele[13])
              ||  (aMod==Modele[14])
            )
    {
        eModAutom = "eCalibAutom" + aMod;

        LocDRadMaxUSer = 3 + (aKModele-11) * 2;
        LocDegGen = 1;
        LocLibDec = false;
    }
    else if ((aKModele>=15) && (aKModele<=26))
    {
        ModeleAdditional= true;
        ModeleAddFour= (aKModele<=18);
        ModeleAddPoly = ! ModeleAddFour;
        if (ModeleAddFour)
        {
              LocDRadMaxUSer = 3 + (aKModele-15) * 2;
              LocDegGen = 2;
              LocLibDec = false;
              TheModelAdd = "eModeleRadFour"+ ToString(7+4*(aKModele-15)) +"x2";
        }
        else if (ModeleAddPoly)
        {
              LocDRadMaxUSer = 0;
              LocLibDec = false;
              LocLibCD = false;
              LocDegGen =  (aKModele-19);
              TheModelAdd = "eModelePolyDeg" +  ToString(LocDegGen);
              LocDegAdd=(aKModele-19);// dans tapas le param degAdd de campari n'existe pas et est "confondu/interchangeable" avec DegGen?

        }
        eModAutom = "eCalibAutomNone";
    }
    else if (     (aMod==Modele[27]) // Ebner
              ||  (aMod==Modele[28]) // Brown
           )
    {
        eModAutom = "eCalibAutom" + aMod;

        LocDRadMaxUSer = 1;
        LocDegGen = 5;
        LocLibDec = false;
    }
    else
    {
        std::cout << "For modele =" << aMod << " KMod=" << aKModele << "\n";
        ELISE_ASSERT(false,"internal error for calib in tapas\n");
    }

   SyncLocAndGlobVar();

}

/*
bool GlobLibFoc=true;
int  GlobDRadMaxUSer = 100;
int  GlobDegGen = 100;
*/

void cAppli_Tapas_Campari::SyncLocAndGlobVar(){

    if (! EAMIsInit(&GlobLibDec))       GlobLibDec = LocLibDec;
    if (! EAMIsInit(&GlobLibPP ))       GlobLibPP = LocLibPP ;
    if (! EAMIsInit(&GlobLibCD ))       GlobLibCD = LocLibCD ;
    if (! EAMIsInit(&GlobLibFoc ))      GlobLibFoc = LocLibFoc ;
    if (! EAMIsInit(&GlobDRadMaxUSer )) GlobDRadMaxUSer = LocDRadMaxUSer ;
    if (! EAMIsInit(&GlobDegGen ))      GlobDegGen = LocDegGen;

    if (EAMIsInit(&GlobLibAff))  ElSetMax(GlobDegGen,(GlobLibAff ? 1 : 0));

    if (! EAMIsInit(&GlobDegAdd ))      GlobDegAdd = LocDegAdd;

}


int Tapas_main_new(int argc,char ** argv)
{
    cAppli_Tapas_Campari anATP;
    NoInit = "#@LL?~~XXXXXXXXXX";

    MMD_InitArgcArgv(argc,argv);

    std::string  aModele,aDir,aPat,aFullDir;
    int ExpTxt=0;
    int DoC=1;
    std::string AeroOut = "";
    std::string AeroIn  = NoInit;
    std::string CalibIn = NoInit;
    std::string ImInit  = NoInit;
    int   aVitesseInit=2;
    int   aDecentre = -1;
    Pt2dr Focales(0,100000);
    Pt2dr aPPDec(-1,-1);
    std::string aSetHom="";
    double  TolLPPCD;

    if ((argc>=2)  && (std::string(argv[1])==std::string("-help")))
    {
        ShowAuthorizedModel();
    }

    int FEAutom= 0;
    double SeuilFEAutom = -1;

    int IsForCalib = -1;


    bool MOI = false;
    int DBF = 0;


    std::string  aRapTxt;
    std::string  aPoseFigee="";
    std::string  aCentreFige="";
    std::string  aRotFige="";
    std::string  aCalFigee="";
    std::string  aCalLibre=".*";
    bool         aFreeCalibInit = false;
    bool Debug = false;
    bool AffineAll = true;
    double EcartMaxFin=5.0;

    std::vector<std::string> aImMinMax;
    Pt2dr EcartInit(100.0,5.0);

    double CondMaxPano = 1e6 ;

    std::vector<std::string>  SinglePos;
    int RankFocale = 3;
    int RankPP     = 4;

    std::vector<double> aVRegulDist;
    double aLVM = 1.0;
    bool MultipleBlock =false;
    bool IsMidle= false;
    bool InitBlocCam = false;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aModele,"Calibration model",eSAM_None,anATP.GetAuthorizedModel())
                    << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile),
        LArgMain()  << EAM(ExpTxt,"ExpTxt",true,"Export in text format (Def=false)",eSAM_IsBool)
                    << EAM(AeroOut,"Out",true, "Directory of Output Orientation", eSAM_IsOutputDirOri)
                    << EAM(CalibIn,"InCal",true,"Directory of Input Internal Orientation (Calibration)",eSAM_IsExistDirOri)
                    << EAM(AeroIn,"InOri",true,"Directory of Input External Orientation",eSAM_IsExistDirOri)
                    << EAM(DoC,"DoC",true,"Do Compensation", eSAM_IsBool)
                    << EAM(IsForCalib,"ForCalib",true,"Is for calibration (Change def value of LMV and prop diag)?")
                    << EAM(Focales,"Focs",true, "Keep images with focal length inside range [A,B] (A,B in mm) (Def=keep all)")
                    << EAM(aVitesseInit,"VitesseInit",true)
                    << EAM(aPPDec,"PPRel",true, "Principal point shift")
                    << EAM(aDecentre,"Decentre",true, "Principal point is shifted (Def=false)")
                    << EAM(anATP.PropDiag,"PropDiag",true, "Hemi-spherik fisheye diameter to diagonal ratio")
                    << EAM(ImInit,"ImInit",true, "Force first image", eSAM_IsExistFile)
                    << EAM(MOI,"MOI",true,"MOI", eSAM_IsBool)
                    << EAM(DBF,"DBF",true,"Debug (internal use : DebugPbCondFaisceau=true) ",eSAM_InternalUse)
                    << EAM(Debug,"Debug",true,"Partial file for debug", eSAM_InternalUse)
                    << EAM(anATP.GlobDRadMaxUSer,"DegRadMax",true,"Max degree of radial, default model dependent")
                    << EAM(anATP.GlobDegGen,"DegGen",true,"Max degree of general polynome, default model dependent (generally 0 or 1)")
                    << EAM(anATP.GlobLibAff,"LibAff",true,"Free affine parameter, Def=true", eSAM_IsBool)
                    << EAM(anATP.GlobLibPP  ,"LibPP",true,"Free principal point, Def=true", eSAM_IsBool)
                    << EAM(anATP.GlobLibFoc,"LibFoc",true,"Free focal, Def=true", eSAM_IsBool)
                    << EAM(aRapTxt,"RapTxt",true, "RapTxt", eSAM_NoInit)
                    << EAM(TolLPPCD,"LinkPPaPPs",true, "Link PPa and PPs (double)", eSAM_NoInit)
                    << EAM(aPoseFigee,"FrozenPoses",true,"List of frozen poses (pattern)", eSAM_IsPatFile)
                    << EAM(aCentreFige,"FrozenCenters",true,"List of frozen centers of poses (pattern)", eSAM_IsPatFile)
                    << EAM(aRotFige,"FrozenOrients",true,"List of frozen orients of poses (pattern)", eSAM_IsPatFile)
                    << EAM(aFreeCalibInit,"FreeCalibInit",true,"Free calibs as soon as created (Def=false)", eSAM_IsPatFile)
                    << EAM(aCalFigee,"FrozenCalibs",true,"List of frozen calibration (pattern)", eSAM_IsPatFile)
                    << EAM(aCalLibre,"FreeCalibs",true,"List of free calibration (pattern, Def=\".*\")", eSAM_IsPatFile)
                    << EAM(aSetHom,"SH",true,"Set of Hom, Def=\"\", give MasqFiltered for result of HomolFilterMasq")
                    << EAM(AffineAll,"RefineAll",true,"More refinement at all step, safer and more accurate, but slower, def=true")
                    << EAM(aImMinMax,"ImMinMax",true,"Image min and max (may avoid tricky pattern ...)")
                    << EAM(EcartMaxFin,"EcMax",true,"Final threshold for residual, def = 5.0 ")
                    << EAM(EcartInit,"EcInit",true,"Inital threshold for residual def = [100,5.0] ")
                    << EAM(CondMaxPano,"CondMaxPano",true,"Precaution for conditionning with Panoramic images, Def=1e4 (old was 0) ")
                    << EAM(SinglePos,"SinglePos",true,"Pattern of single Pos Calib to save [Pose,Calib]")

                    << EAM(RankFocale,"RankInitF",true,"Order of focal initialisation, ref id distotion =2, Def=3 ")
                    << EAM(RankPP,"RankInitPP",true,"Order of Principal point initialisation, ref id distotion =2, Def=4")
                    << EAM(aVRegulDist,"RegulDist",true,"Parameter fo RegulDist [Val,Grad,Hessian,NbCase,SeuilNb]")
                    << EAM(aLVM,"MulLVM",true,"Multipier Levenberg Markard")
                    << EAM(MultipleBlock,"MultipleBlock",true,"Multiple block need special caution (only related to Levenberg Markard)")
                    << EAM(InitBlocCam,"UBR4I","Use Bloc Rigid for Init, def=context dependent")
                    << anATP.ArgATP()
    );


    if (!MMVisualMode)
    {

        if ((AeroIn!= NoInit)  && (CalibIn==NoInit))
            CalibIn = AeroIn;

        #if (ELISE_windows)
            replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
        #endif
        SplitDirAndFile(aDir,aPat,aFullDir);
        setInputDirectory( aDir );

        if (AeroIn!= NoInit)
           StdCorrecNameOrient(AeroIn,aDir);

        if (CalibIn!= NoInit)
           StdCorrecNameOrient(CalibIn,aDir);


        cTplValGesInit<std::string> aTplN;
        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::StdAlloc(0,0,aDir,aTplN);

              MakeXmlXifInfo(aFullDir,aICNM);


         if (EAMIsInit(&ImInit)  && (ImInit=="MIDLE"))
         {
              cInterfChantierNameManipulateur::tSet   aSetGlob = *(aICNM->Get(aPat));
              ELISE_ASSERT(aSetGlob.size()!=0,"Empyt pat of image");
              std::sort(aSetGlob.begin(),aSetGlob.end());
              ImInit  = aSetGlob[aSetGlob.size()/2];
              IsMidle = true;
         }


        if (FEAutom && (SeuilFEAutom<0))
           SeuilFEAutom = 16.5;

        if (IsForCalib<0)
            IsForCalib=(CalibIn==NoInit); // A Changer avec cle de calib

        double TetaLVM = IsForCalib ?     0.1 : 1.5;
        double CentreLVM = IsForCalib ?   1.0 : 10.0;
        double IntrLVM = IsForCalib ?   0.1 : 1.0;


        if (! EAMIsInit(& aVitesseInit)) aVitesseInit = AffineAll ? 2 : 5;


        double RayFEInit = IsForCalib ? 0.85 : 0.95;

    // std::cout << "IFCCCCC " << IsForCalib << " " << CentreLVM << " " << RayFEInit << "\n"; getchar();

        anATP.InitVerifModele(aModele,aICNM);

        if (anATP.PropDiag<0) anATP.PropDiag = 1.0;

        if (AeroOut=="")
           AeroOut = "" +  aModele;



       std::string aNameFileApero = "Apero-Glob-New.xml" ;

       // std::string aParamPatFocSetIm = "@" + aPat + "@" + ToString(Focales.x) + "@" + ToString(Focales.y) ;
       std::string aParamPatFocSetIm = "@[[" + aPat + "]]@" + ToString(Focales.x) + "@" + ToString(Focales.y) ;
       std::string aSetIm = "NKS-Set-OfPatternAndFoc" + aParamPatFocSetIm;

        if (EAMIsInit(&aImMinMax))
        {
            ELISE_ASSERT(aImMinMax.size()==2,"ImMinMax size mut be 2");
            aSetIm =  "NKS-Set-OfPatternAndFocAndInterv" + aParamPatFocSetIm + "@" + aImMinMax[0] + "@" + aImMinMax[1];
        }


       bool SpecFocale = (RankFocale!=3);
       bool SpecPP = (RankPP!=4);


       std::string aCom =     MM3dBinFile_quotes( "Apero" )
                           + ToStrBlkCorr( MMDir()+"include"+ELISE_CAR_DIR+"XML_MicMac"+ELISE_CAR_DIR+ aNameFileApero ) + " "
                           + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                           + std::string(" ") + QUOTE(std::string("+PatternAllIm=") + aPat) + std::string(" ")
                           + std::string(" ") + QUOTE(std::string("+SetIm=") + aSetIm) + std::string(" ")
                           //+ std::string(" +PatternAllIm=") + aPat + std::string(" ")
                           + std::string(" +AeroOut=-") + AeroOut
                           + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                           + std::string(" +ModeleCam=") + anATP.eModAutom
                           + std::string(" DoCompensation=") + ToString(DoC)
                           + std::string(" +SeuilFE=") + ToString(SeuilFEAutom)
                           + std::string(" +TetaLVM=") + ToString(TetaLVM)
                           + std::string(" +CentreLVM=") + ToString(CentreLVM)
                           + std::string(" +IntrLVM=") + ToString(IntrLVM)
               
                           + std::string(" +RayFEInit=") + ToString(RayFEInit)
                           + std::string(" +CalibIn=-") + CalibIn
                           + std::string(" +AeroIn=-") + AeroIn
                           + std::string(" +VitesseInit=") + ToString(2+aVitesseInit)
                           + std::string(" +PropDiagU=") + ToString(anATP.PropDiag)

                           + std::string(" +DegRadMax=") + ToString(anATP.GlobDRadMaxUSer)
                           + std::string(" +LibFoc=") + ToString(anATP.GlobLibFoc && (!SpecFocale))
                           + std::string(" +LibPP=") + ToString(anATP.GlobLibPP && (!SpecPP))
                           + std::string(" +LibCD=") + ToString(anATP.GlobLibCD)
                           + std::string(" +DegGen=") + ToString(anATP.GlobDegGen)
                           + std::string(" +LibDec=") + ToString(anATP.GlobLibDec)
                           + std::string(" +Fast=") + ToString(! AffineAll)
                           + std::string(" +UsePano=true") 
                           + std::string(" +CondMaxPano=") + ToString(CondMaxPano)
                          ;

       if  (SpecPP || SpecFocale)
       {
           std::string aSpecialParam0 = "eLib_PP_CD_10";
           std::string aSpecialParam1 = "eLiberteFocale_1";
           if (RankFocale < RankPP)
              ElSwap(aSpecialParam0,aSpecialParam1);

           aCom = aCom + " +UseSpecialParam0=true +SpecialParam0=" +aSpecialParam0;
           if  (SpecPP &&  SpecFocale)
               aCom = aCom + " +UseSpecialParam1=true +SpecialParam1=" +aSpecialParam1;
       }

        StdCorrecNameHomol(aSetHom,aDir);
        if (EAMIsInit(&aSetHom))
        {
            aCom = aCom + std::string(" +SetHom=") + aSetHom;
        }

        if (MultipleBlock) aCom = aCom + " +OneBlock=false ";

        if (EAMIsInit(&aLVM))
        {
            aCom = aCom + std::string(" +MulLVM=") + ToString(aLVM);
        }

        if (EAMIsInit(&EcartMaxFin))
        {
            aCom = aCom + " +EcartMaxFin=" + ToString(EcartMaxFin);
        }

        if (EAMIsInit(&EcartInit))
        {
            aCom = aCom + " +EcartMaxInit=" + ToString(EcartInit.x) + " +SigmaPondInit=" + ToString(EcartInit.y) ;
        }



        if (anATP.ModeleAdditional)
        {
              aCom = aCom + std::string(" +HasModeleAdd=true")
                          + std::string(" +ModeleAdditionnel=") + anATP.TheModelAdd;
        }


        if (EAMIsInit(&anATP.GlobLibAff) && (!anATP.GlobLibAff))
        {
              aCom = aCom + " +LiberteAff=false ";
        }

        if (EAMIsInit(&TolLPPCD))
           aCom = aCom + " +TolLinkPPCD=" + ToString(TolLPPCD);

       if (aRapTxt!="")
          aCom = aCom +  std::string(" +RapTxt=") + aRapTxt;

       if (DBF)
         aCom  = aCom + " DebugPbCondFaisceau=true";

/* => Dans Common

       if (mSauvAutom!="")
       {
         if (mSauvAutom=="NONE")
            aCom =   aCom + " +DoSauvAutom=false";
         else
            aCom =   aCom + " +SauvAutom="+SauvAutom;
       }
*/

       if (AeroIn!= NoInit)
       {
          aCom =   aCom + " +InitCamBDD=true +InitCamCenter=false";
                 // + " +FileCamInit=InitCamBDD.xml" ;
       }


       if (aPPDec.x>=0)
           aCom =   aCom + " +xPRelPP=" + ToString(aPPDec.x);
       else
          aPPDec.x =0.5;
       if (aPPDec.y>=0)
           aCom =   aCom + " +yPRelPP=" + ToString(aPPDec.y);
       else
          aPPDec.y =0.5;

       if (aDecentre==-1)
       {
            double aDist = euclid(aPPDec,Pt2dr(0.5,0.5));
            aDecentre= (aDist>=0.25);
       }

       if (aDecentre)
       {
            aCom  = aCom + " +ModeCDD=eCDD_OnRemontee";
       }

       if (MOI)
       {
            aCom  = aCom + " +MOI=true";
       }

       if (aPoseFigee!="")
       {
          aCom  = aCom + " +PoseFigee=" + QUOTE(aPoseFigee) + " +WithPoseFigee=true";
       }
       if (aCentreFige!="")
       {
          aCom  = aCom + " +CentreFiges=" + QUOTE(aCentreFige) + " +WithCentreFiges=true";
       }
       if (aRotFige!="")
       {
          aCom  = aCom + " +RotationFigees=" + QUOTE(aRotFige) + " +WithRotationFiges=true";
       }
       if (aFreeCalibInit)
       {
          aCom  = aCom + " +FrozeCalibInit=false";
       }
 

       if (aCalFigee!="")
       {
          aCom  = aCom + " +CalibFigee=" + QUOTE(aCalFigee) ;
       }
       if (EAMIsInit(&aCalLibre))
       {
          aCom  = aCom + " +CalibLibre=" + QUOTE(aCalLibre) ;
       }

       if (anATP.IsAutoCal) aCom  = aCom + " +AutoCal=true";
       if (anATP.IsFigee) aCom  = aCom + " +CalFigee=true";

       if (EAMIsInit(&SinglePos))
       {
           ELISE_ASSERT(SinglePos.size()==2,"SinglePos size must be 2");
           aCom =   aCom 
                  + std::string(" +HasSinglePoseCalibEstim=true")
                  + std::string(" +PatSinglePose=") + QUOTE(SinglePos[0])
                  + std::string(" +PatSingleCalib=") + QUOTE(SinglePos[1]);
       }

       if (EAMIsInit(&aVRegulDist))
       {
           ELISE_ASSERT(aVRegulDist.size()>=3,"Not enough parameter in RegulDist")
           double aNbCase = (aVRegulDist.size() >= 4) ? round_ni(aVRegulDist[3])  : 7;
           double aSeuilNbPts = (aVRegulDist.size() >= 5) ? aVRegulDist[4]  : 5.0;
           aCom = aCom  + std::string(" +UseRegulDist=true")
                        + std::string(" +RegDist0=") + ToString(aVRegulDist[0])
                        + std::string(" +RegDist1=") + ToString(aVRegulDist[1])
                        + std::string(" +RegDist2=") + ToString(aVRegulDist[2])
                        + std::string(" +RegDistNbCase=") + ToString(aNbCase)
                        + std::string(" +RegDistSeuil=") + ToString(aSeuilNbPts);
       }

       anATP.AddParamBloc(aCom);
       if (anATP.mWithBlock)
       {
           if (IsMidle)
           {
              anATP.InitAllImages(aPat,aICNM);
              int aNBInBl =  anATP.NbInBloc();
              const std::vector<std::string> & aVImage   = anATP.BlocImagesByTime();
              const std::vector<std::string> & aVTime = anATP.BlocTimeStamps();
              std::map<std::string,int> & aCptTime =  anATP.BlocCptTime();

              // Recupere l'image la plus proche du milieu et dont le bloc est plein
              int aDistMax = -1;
              int aKMax = -1;
              for (int aK=0 ; aK<int(aVImage.size()) ; aK++)
              {
                 if (aCptTime[aVTime[aK]] == aNBInBl)
                 {
                    int aDist = ElMin(aK,int(aVImage.size()-1-aK)); // Distance to border, max in midle
                    if (aDist>aDistMax)
                    {
                       aDistMax = aDist;
                       aKMax = aK;
                    }
                 }
              }
              ELISE_ASSERT(aDistMax>=0,"No Bloc Full");
              ImInit=  aVImage[aKMax];
           }

/*
           cInterfChantierNameManipulateur::tSet   aSetGlob = *(aICNM->Get(aPat));
           std::vector<std::pair<std::string,std::string> > aVP;
           std::map<std::string,int> aCptTime;
           for (const auto & aS : aSetGlob)
           {
               aVP.push_back(std::pair<std::string,std::string>(anATP.TimeStamp(aS,aICNM),aS));
               aCptTime[aVP.back().first]++;
           }
           if (IsMidle)
           {
              std::sort(aVP.begin(),aVP.end());
              int aDistMax = -1;
              int aKMax = -1;
              for (int aK=0 ; aK<int(aVP.size()) ; aK++)
              {
                 if (aCptTime[aVP[aK].first] == aNBInBl)
                 {
                    int aDist = ElMin(aK,int(aVP.size()-1-aK));
                    if (aDist>aDistMax)
                    {
                       aDistMax = aDist;
                       aKMax = aK;
                    }
                 }
              }
              ELISE_ASSERT(aDistMax>=0,"No Bloc Full");
              ImInit=  aVP[aKMax].second;
           }
*/
           if (EAMIsInit(&ImInit))
           {
               InitBlocCam = true;
               aCom = aCom + " +InitCamCenter=false ";
               ImInit= QUOTE(anATP.ExtendPattern(aPat,ImInit,aICNM));
               aCom =   aCom + " +SetImInit="+ImInit;
           }
           if (! EAMIsInit(&aVitesseInit))
               aCom = aCom + std::string(" +VitesseInit=1") ;
       }
       else if (ImInit!=NoInit)
       {
             aCom =   aCom + " +SetImInit="+ImInit;
             //aCom = aCom + " +FileCamInit=InitCamSpecified.xml";
             aCom = aCom + " +InitCamSpecif=true +InitCamCenter=false";
             ELISE_ASSERT(AeroIn==NoInit,"Incoherence AeroIn/ImInit");
       }


       if (InitBlocCam)
       {
          aCom = aCom + " +InitBlocCam=true ";
       }
       anATP.AddParamBloc(aCom);

       std::cout << "Com = " << aCom << "\n";
       int aRes = 0;

       aRes = TopSystem(aCom);
    /*
       if (MajickTest)
       {
            std::string aNameFile = MMDir() + "DbgAp" + GetUnikId() + ".txt";

            // cMemRes aMR1;
            // cMemRes aMR2;

             aCom = aCom + " +FileDebug=" +  aNameFile;


            for (int aTest=0 ; aTest < 1000000 ; aTest++)
            {

               // int aValInit = (aTest % 17);
               // int aSzMax = 29;
               // aMR1.Init(aSzMax,aValInit);
               // aMR2.Init(aSzMax,aValInit);

               // aMR2.Free();
               aRes = ::System(aCom.c_str(),true,true);
               // aMR1.Free();



               sleep(1); // Pour faciliter l'arret
            }

       }
       else
       {
           aRes = ::System(aCom.c_str(),false,true,true);
       }
    */


       Tapas_Banniere();
       BanniereMM3D();


       return aRes;
   }
   else
       return EXIT_SUCCESS;
}


int New_Tapas_main(int argc,char ** argv)
{
   return  Tapas_main_new(argc,argv);
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
