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



   //===========================================

int Genere_Header_TiffFile_main(int argc,char ** argv)
{
    std::string aRes = "";
    std::string aFullFileIn;
    Pt2di aSzTot;

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFullFileIn,"Name of file for tile 0 0 ", eSAM_IsExistFile)
                      << EAMC(aSzTot,"Size of full image"),
           LArgMain() << EAM(aRes,"Out",true,"Name of result")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    std::string aDir,aFileIn;
    SplitDirAndFile(aDir,aFileIn,aFullFileIn);

    if (aRes=="")
    {
         aRes  =  aDir + "Header"+ aFileIn;
    }
    else
       aRes  =  aDir + aRes;


    Tiff_Im aTIn(aFullFileIn.c_str());

    L_Arg_Opt_Tiff aLArgOpt = Tiff_Im::Empty_ARG;

    Pt2di aSz = aTIn.sz();
    Pt2di aSzTile  = aTIn.sz_tile();


    ELISE_ASSERT( ((aSz.x%aSzTile.x)==0) && ((aSz.y%aSzTile.y) ==0),"Sz must be multiple of tile");

    aLArgOpt  = aLArgOpt + Arg_Tiff(Tiff_Im::AFileTiling(-aSz))
                         + Arg_Tiff(Tiff_Im::ATiles(-aSzTile));

    Tiff_Im  aFHeader
             (
                  aRes.c_str(),
                  aSzTot,
                  aTIn.type_el(),
                  aTIn.mode_compr(),
                  aTIn.phot_interp(),
                  aLArgOpt
             );

    return 0;
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
