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
#include <fstream>

/**
 * ProjImPtOnOtherImages: project image points on other images
 *
 * Inputs:
 *  - pattern of images
 *  - Ori
 *  - image points xml file
 *
 * Output:
 *  - 2d points on all images
 *
 * Call example:
 *   mm3d ProjImPtOnOtherImages ".*.tif" Ori-Basc/ Im2D_partial.xml
 *
 *
 *
 * */



int ProjImPtOnOtherImages_main(int argc,char ** argv)
{
  std::string aFullPattern;//pattern of all scanned images
  std::string aImPtFileName;//2d points
  std::string aOriIn;//Orientation containing all images and calibrations

  std::cout<<"ProjImPtOnOtherImages: project image points on other images"<<std::endl;
  ElInitArgMain
    (
     argc,argv,
     //mandatory arguments
     LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                 << EAMC(aOriIn, "Directory orientation",  eSAM_IsExistDirOri)
                 << EAMC(aImPtFileName, "Image points file", eSAM_IsExistFile),
     //optional arguments
     LArgMain()
    );

  if (MMVisualMode) return EXIT_SUCCESS;

  std::string aCom1 = "mm3d TestLib PseudoIntersect " + aFullPattern +  " " + aOriIn + " "+ aImPtFileName ;
  System(aCom1); //output: 3DCoords.xml

  std::string aCom2 = "mm3d SimplePredict " + aFullPattern +  " " + aOriIn + " 3DCoords.xml " ;
  System(aCom2); //output:

  std::cout<<"Quit"<<std::endl;

  return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant a  la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilite au code source et des droits de copie,
   de modification et de redistribution accordes par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
   seule une responsabilite restreinte pese sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concedants successifs.

   A cet egard  l'attention de l'utilisateur est attiree sur les risques
   associes au chargement,  a  l'utilisation,  a  la modification et/ou au
   developpement et a  la reproduction du logiciel par l'utilisateur etant
   donne sa specificite de logiciel libre, qui peut le rendre complexe a
   manipuler et qui le reserve donc a  des developpeurs et des professionnels
   avertis possedant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
   logiciel a  leurs besoins dans des conditions permettant d'assurer la
   securite de leurs systemes et ou de leurs donnees et, plus generalement,
   a l'utiliser et l'exploiter dans les memes conditions de securite.

   Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
