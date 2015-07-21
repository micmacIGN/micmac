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
#include "../uti_phgrm/MICMAC/CameraRPC.h"

//class cSatI_Appli;

void RPCtoBundle_ElInitArgMain(int argc,char ** argv, 
		                   std::string &aFullName,
				   std::string &aModeRPC, 
				   std::string &aDir, 
				   std::string &aPat, 
				   std::string &aCSysOut,
				   Pt2di &aGridSz)
{
    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aFullName,"RPC file full name (Dir+Pat)")
                   << EAMC(aCSysOut,"Output cartographic coordinate system (proj format)"),
	LArgMain() << EAM(aGridSz,"GrSz",true, "No. of grids of bundles, e.g. GrSz=[5,8]")
	           << EAM(aModeRPC, "ModeRPC", true, "The RPC convention (PLEIADE,SPOT,QUICKBIRD,WORLDVIEW,IKONOS,CARTOSAT)")
    );		

    SplitDirAndFile(aDir, aPat, aFullName);

    //validate the RPC mode
    if (aModeRPC=="PLEIADE") {}
    else if (aModeRPC=="SPOT") {}
    else if (aModeRPC=="QUICKBIRD") {}
    else if (aModeRPC=="WORLDVIEW") {}
    else if (aModeRPC=="IKONOS") {}
    else if (aModeRPC=="CARTOSAT") {}
    else {ELISE_ASSERT(false,"Unknown RPC mode");}
}

int RPCtoBundle_main(int argc,char ** argv)
{

    cInterfChantierNameManipulateur * aICNM;
    std::string aFullName;
    std::string aDir;
    std::string aPat;
    std::string aCSysOut;
    std::list<std::string> aListFile;
    std::string aModeRPC;

    Pt2di aGridSz;

    RPCtoBundle_ElInitArgMain(argc,argv,aFullName,aModeRPC,aDir,aPat,aCSysOut,aGridSz);

    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aPat);

    std::cout << "Processed images:" << std::endl;
    for(std::list<std::string>::iterator itL = aListFile.begin(); itL != aListFile.end(); itL++ )   
    {
	std::cout << " - " << *itL << std::endl;
        CameraRPC aCurCam(*itL,aModeRPC,aGridSz);
	aCurCam.ExpImp2Bundle(aCSysOut,*itL);
    }


    return EXIT_SUCCESS;
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
