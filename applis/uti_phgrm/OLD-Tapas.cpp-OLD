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

    MicMa cis an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "general/all.h"
#include "private/all.h"
#include <algorithm>

/*
Parametre de Tapas :
  
   - calibration In : en base de donnees ou deja existantes.


*/

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

#define  NbModele 10


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
                                   "FishEyeBasic"     // 9
                                };


std::string eModAutom;
std::string FileLibere;
double PropDiag = -1.0;


void ShowAuthorizedModel()
{
   std::cout << "\n";
   std::cout << "Authorized models : \n";
   for (int aKM=0 ; aKM<NbModele ; aKM++)
       std::cout << "   " << Modele[aKM] << "\n";
}

void InitVerifModele(const std::string & aMod,cInterfChantierNameManipulateur *)
{
    std::string  aModParam = aMod;


    if (aMod==Modele[0]) 
    {
       eModAutom = "eCalibAutomRadialBasic";
    }
    else if ((aMod==Modele[1]) ||  (aMod==Modele[7]))
    {
       eModAutom = "eCalibAutomRadial";
    }
    else if (aMod==Modele[2])
    {
        eModAutom = "eCalibAutomPhgrStd";
    }
    else if ((aMod==Modele[3]) || (aMod==Modele[6]))
    {
        eModAutom = "eCalibAutomFishEyeLineaire";
        aModParam  = Modele[3];
        if (aMod==Modele[6])
        {
            if (PropDiag<0) 
               PropDiag = 0.52;
        }
    }
    else if (aMod==Modele[9])
    {
        eModAutom = "eCalibAutomFishEyeLineaire";
        aModParam  = Modele[9];
    }
    else if ((aMod==Modele[4]) || (aMod==Modele[5]))
    {
        eModAutom = "eCalibAutomNone";
    }
    else if (aMod==Modele[8]) 
    {
        eModAutom = "eCalibAutomPhgrStdBasic";
        aModParam= Modele[2];
    }
    else
    {
        ShowAuthorizedModel();
        ELISE_ASSERT(false,"Value is not a correct model\n");
    }

    FileLibere = "Param-"+aModParam+".xml";
}


std::string NoInit = "#@LL?~~XXXXXXXXXX";

int main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string  aModele,aDir,aPat,aFullDir;
    int ExpTxt=0;
    int DoC=1;
    std::string AeroOut="";
    std::string AeroIn= NoInit;
    std::string CalibIn= NoInit;
    std::string ImInit=  NoInit;
    int   aVitesseInit=2;
    int   aDecentre = -1;
    Pt2dr Focales(-1,-1);
    Pt2dr aPPDec(-1,-1);
    std::string SauvAutom="";
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

    bool LibAff = true;
    bool LibDec = true;

    std::string  aRapTxt;
    std::string  aPoseFigee="";

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aModele,"Modele of Calibration")
                    << EAMC(aFullDir,"Full Directory (Dir+Pattern)"),
	LArgMain()  << EAM(ExpTxt,"ExpTxt",true,"Export in text format(def=false)")	
                    << EAM(AeroOut,"Out",true)	
                    << EAM(CalibIn,"InCal",true,"Directory of Input Internal Orientation (Calibration)")	
                    << EAM(AeroIn,"InOri",true,"Directory of Input External Orientation")	
                    << EAM(DoC,"DoC",true,"Do Compensation")	
                    << EAM(IsForCalib,"ForCalib",true)	
                    << EAM(Focales,"Focs",true)	
                    << EAM(aVitesseInit,"VitesseInit",true)	
                    << EAM(aPPDec,"PPRel",true)	
                    << EAM(aDecentre,"Decentre",true)	
                    << EAM(PropDiag,"PropDiag",true)	
                    << EAM(SauvAutom,"SauvAutom",true)	
                    << EAM(ImInit,"ImInit",true)	
                    << EAM(MOI,"MOI",true)	
                    << EAM(DBF,"DBF",true,"Debug (internal use : DebugPbCondFaisceau=true) ")	
                    << EAM(LibAff,"LibAff",true)	
                    << EAM(LibDec,"LibDec",true)	
                    << EAM(aRapTxt,"RapTxt",true)	
                    << EAM(TolLPPCD,"LinkPPaPPs",true)	
                    << EAM(aPoseFigee,"FrozenPoses",true,"List of poses frozen")	
    );


     if ((AeroIn!= NoInit)  && (CalibIn==NoInit))
          CalibIn = AeroIn;
       

    SplitDirAndFile(aDir,aPat,aFullDir);

    cTplValGesInit<std::string> aTplN;
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::StdAlloc(0,0,aDir,aTplN);

    if (FEAutom && (SeuilFEAutom<0))
       SeuilFEAutom = 16.5;

    if (IsForCalib<0) 
        IsForCalib=(CalibIn==NoInit); // A Changer avec cle de calib

    double TetaLVM = IsForCalib ? 0.01 : 0.15;
    double CentreLVM = IsForCalib ? 0.1 : 1.0;
    double RayFEInit = IsForCalib ? 0.85 : 0.95;
     

    InitVerifModele(aModele,aICNM);

    if (PropDiag<0) PropDiag = 1.0;

    if (AeroOut=="")
       AeroOut = "" +  aModele;




   std::string aCom =     MMDir() + std::string("bin")+ELISE_CAR_DIR+"Apero "
                       +  MMDir() + std::string("include")+ELISE_CAR_DIR+"XML_MicMac"+ELISE_CAR_DIR+"Apero-Glob.xml "
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" ") + QUOTE(std::string("+PatternAllIm=") + aPat) + std::string(" ")
                       //+ std::string(" +PatternAllIm=") + aPat + std::string(" ")
                       + std::string(" +AeroOut=-") + AeroOut
                       + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                       + std::string(" +ModeleCam=") + eModAutom 
                       + std::string(" +FileLibereParam=") + FileLibere
                       + std::string(" DoCompensation=") + ToString(DoC)
                       + std::string(" +SeuilFE=") + ToString(SeuilFEAutom)
                       + std::string(" +TetaLVM=") + ToString(TetaLVM)
                       + std::string(" +CentreLVM=") + ToString(CentreLVM)
                       + std::string(" +RayFEInit=") + ToString(RayFEInit)
                       + std::string(" +CalibIn=-") + CalibIn
                       + std::string(" +AeroIn=-") + AeroIn
                       + std::string(" +VitesseInit=") + ToString(2+aVitesseInit)
                       + std::string(" +PropDiagU=") + ToString(PropDiag)
                       + std::string(" +ValDec=") + (LibDec ?"eLiberte_Phgr_Std_Dec" : "eFige_Phgr_Std_Dec")
                       + std::string(" +ValDecPP=") + (LibDec ?"eLiberte_Dec1" : "eLiberte_Dec0")
                       + std::string(" +ValAffPP=") + (LibDec ?"eLiberteParamDeg_1" : "eLiberteParamDeg_0")
                       + std::string(" +ValAff=") + (LibAff ?"eLiberte_Phgr_Std_Aff" : "eFige_Phgr_Std_Aff")
                    ;


    if (EAMIsInit(&TolLPPCD))
       aCom = aCom + " +TolLinkPPCD=" + ToString(TolLPPCD);

   if (aRapTxt!="")
      aCom = aCom +  std::string(" +RapTxt=") + aRapTxt;

   if (DBF)
     aCom  = aCom + " DebugPbCondFaisceau=true";


   if (ImInit!=NoInit)
   {
         aCom =   aCom + " +SetImInit="+ImInit;
         aCom = aCom + " +FileCamInit=InitCamSpecified.xml";
         ELISE_ASSERT(AeroIn==NoInit,"Incoherence AeroIn/ImInit");
   }
   if (SauvAutom!="")
     aCom =   aCom + " +SauvAutom="+SauvAutom;

   if (AeroIn!= NoInit)
      aCom =   aCom
             + " +FileCamInit=InitCamBDD.xml" ;

   if (Focales.x>=0)
      aCom =   aCom
             + " +FocMin=" + ToString(Focales.x)
             + " +FocMax=" + ToString(Focales.y);

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
      aCom  = aCom + " +PoseFigee=" + QUOTE(aPoseFigee);
   }

   std::cout << "Com = " << aCom << "\n";
   int aRes = system_call(aCom.c_str());


   Tapas_Banniere();
   
   return aRes;
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
