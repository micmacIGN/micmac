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

#include "TiePHistorical.h"



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
aooter-MicMac-eLiSe-25/06/2007*/

void TransformDSM(std::string aFileOut)
{
    cXml_Map2D aMap;
    std::list<cXml_Map2DElem> aLMEL;
    cXml_Map2DElem aMapEL;
    cXml_Homot Hom;

    std::cout << Hom.Scale() << "\n";

    //double aSc3= aSc1 * aSc2;
    Hom.Scale()=2.0;
       Hom.Tr() = Pt2dr(500,500);
       aMapEL.Homot() = Hom;

      aLMEL.push_back(aMapEL);
      aMap.Maps() = aLMEL;

      MakeFileXML(aMap, aFileOut);
}

int TransformDSM_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aDSMGrayImgDir;

   std::string aDSMGrayImg1;
   std::string aDSMGrayImg2;

   std::string aRGBImgDir;

   std::string aImgList1;
   std::string aImgList2;

   std::string aOri1;
   std::string aOri2;

   std::string aDSMDirL;
   std::string aDSMDirR;
   std::string aDSMFileL = "MMLastNuage.xml";
   std::string aDSMFileR = "MMLastNuage.xml";

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDSMGrayImgDir,"The directory of gray image of DSM")
                    << EAMC(aDSMGrayImg1,"The gray image of DSM of epoch1")
                    << EAMC(aDSMGrayImg2,"The gray image of DSM of epoch2")
                    << EAMC(aRGBImgDir,"The directory of RGB image")
                    << EAMC(aImgList1,"The RGB image list of epoch1")
                    << EAMC(aImgList2,"The RGB image list of epoch2")
               << EAMC(aOri1,"Orientation of images in epoch1")
               << EAMC(aOri2,"Orientation of images in epoch2")
               << EAMC(aDSMDirL,"DSM direcotry of epoch1")
               << EAMC(aDSMDirR,"DSM direcotry of epoch2"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
                    << aCAS3D.ArgCreateGCPs()
                    << EAM(aDSMFileL, "DSMFileL", true, "DSM File of epoch1, Def=MMLastNuage.xml")
                    << EAM(aDSMFileR, "DSMFileR", true, "DSM File of epoch2, Def=MMLastNuage.xml")

    );

   CreateGCPs( aDSMGrayImgDir, aRGBImgDir, aDSMGrayImg1, aDSMGrayImg2, aImgList1, aImgList2, aOri1, aOri2, aCAS3D.mICNM, aDSMDirL, aDSMDirR, aDSMFileL, aDSMFileR, aCAS3D.mOut2DXml1, aCAS3D.mOut2DXml2, aCAS3D.mOut3DXml1, aCAS3D.mOut3DXml2, aCAS3D.mCreateGCPsInSH);

   return EXIT_SUCCESS;
}
