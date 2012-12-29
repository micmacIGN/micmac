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


struct cCamChSys
{
    std::string          mNCam;
    std::string          mNOrIn;
    std::string          mNOrOut;
    cOrientationConique  mXmlIn;
    ElCamera * aCam;
};



int ChgSysCo_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string aFullDir= "";
    std::string AeroIn= "";
    std::string aStrChSys="";
    std::string AeroOut="";
    bool        ForceRot = false;


   std::vector<std::string> GCP;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)")
                    << EAMC(AeroIn,"Input Orientation")
                    << EAMC(aStrChSys,"Change coordinate file")
                    << EAMC(AeroOut,"Output Orientation"),
	LArgMain()  << EAM(ForceRot,"FR",true,"Force orientation matrix to be pure rotation (Def = false)")	
    );

    std::string aKeyIn = "NKS-Assoc-Im2Orient@-" + AeroIn;
    std::string aKeyOut = "NKS-Assoc-Im2Orient@-" + AeroOut;

    std::string aDir,aPat;

#if (ELISE_windows)
     replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
    SplitDirAndFile(aDir,aPat,aFullDir);


    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const cInterfChantierNameManipulateur::tSet * aSetIm = anICNM->Get(aPat);

    std::vector<ElCamera *> aVCam;
    std::vector<cCamChSys *>  aVCS;


    for (int aK=0 ; aK<int(aSetIm->size()) ; aK++)
    {
         aVCS.push_back(new cCamChSys);
         cCamChSys & aCS = *(aVCS.back());
         aCS.mNCam = (*aSetIm)[aK];
         aCS.mNOrIn = anICNM->Assoc1To1(aKeyIn,aCS.mNCam,true);


         std::cout << aCS.mNCam << " " << aCS.mNOrIn << "\n";
    }


    int aRes = 1;

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
