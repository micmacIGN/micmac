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


int  FixSizeImage(int aZoom,std::string aFile,double aLargMin,int aZoomMax)
{
   cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aFile);
   Pt2di aSz = aMDP.SzImTifOrXif();
   double aLargFile =  ElMin(aSz.x,aSz.y);
   while (  ((aLargFile/aZoom)<(aLargMin*2)) && (aZoom>aZoomMax))
         aZoom /= 2;

   return aZoom;
}


int MMTestOrient_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string anIm1,anIm2;
    std::string AeroIn= "";
    std::string aDir="./";
    int Zoom0=32;
    int ZoomF=2;

    double LargMin=30.0;

    bool mModePB = false;
    std::string mModeOri;
    double aZMoy,aZInc;
    bool    ShowCom = false;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(anIm1,"First Image")
                    << EAMC(anIm2,"Second Images")
                    << EAMC(AeroIn,"Orientation"),
	LArgMain()  << EAM(aDir,"Dir",true,"Directory, Def=./")	
                    << EAM(Zoom0,"Zoom0",true,"Zoom init, pow of 2  in [128,8], Def depend of size")
                    << EAM(ZoomF,"ZoomF",true,"Zoom init,  pow of 2  in [4,1], Def=2")
                    << EAM(mModePB,"PB",true,"Push broom sensor")
                    << EAM(mModeOri,"MOri",true,"Mode Orientation (GRID or RTO) , Mandatory in PB")
                    << EAM(aZMoy,"ZMoy",true,"Average Z,  Mandatory in PB")
                    << EAM(aZInc,"ZInc",true,"Incertitude on Z,  Mandatory in PB")
                    << EAM(ShowCom,"ShowCom",true,"Show MicMac command (tuning purpose)")
    );


    if (!EAMIsInit(&mModePB))
      mModePB = EAMIsInit(&mModeOri);

    std::string aFullModeOri = "eGeomImageOri";
    if (mModePB)
    {
         ELISE_ASSERT(EAMIsInit(&mModeOri) , "MOri is Mandatory in PB");
         ELISE_ASSERT(EAMIsInit(&aZMoy)    , "ZMoy is Mandatory in PB");
         ELISE_ASSERT(EAMIsInit(&aZInc)    , "ZInc is Mandatory in PB");

         if (mModeOri=="GRID")     aFullModeOri= "eGeomImageGrille";
         else if (mModeOri=="RTO") aFullModeOri= "eGeomImageRTO";
         else  {ELISE_ASSERT(false,"Unknown mode ori");}
    }


// eGeomImageRTO
// eGeomImageGrille

    if (! EAMIsInit(&Zoom0))
    {
         Zoom0 = FixSizeImage(128,aDir+anIm1,LargMin,4);
    }

#if (ELISE_windows)
     replace( aDir.begin(), aDir.end(), '\\', '/' );
#endif




   std::string aCom =     MM3dBinFile( "MICMAC" ) + " " 
                       +  Basic_XML_MM_File("MM-PxTransv.xml")
                       +  std::string(" WorkDir=") + aDir + " "
                       +  std::string(" +Im1=") + QUOTE(anIm1) + " " 
                       +  std::string(" +Im2=") + QUOTE(anIm2) + " " 
                       +  std::string(" +AeroIn=-") + AeroIn + " " 
                       +  std::string(" +Zoom0=") + ToString(Zoom0) + " " 
                       +  std::string(" +ZoomF=") + ToString(ZoomF) + " " 
                      ;

    if (mModePB)
    {
         aCom = aCom + " +Conik=false "
                     +  " +ModeOriIm=" + aFullModeOri + std::string(" ")
                     + " +PostFixOri=" + AeroIn     + std::string(" ")
                     + " +Px1Inc=" + ToString(aZInc) + std::string(" ")
                     + " +Px1Moy=" + ToString(aZMoy) + std::string(" ") ;
    }

    if (ShowCom) std::cout << aCom << "\n";



   int aRes = system_call(aCom.c_str());


   BanniereMM3D();

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
