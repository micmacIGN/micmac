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
#include "Casa.h"


void Casa_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     C-alcul                   *\n";
    std::cout <<  " *     A-utomatique de           *\n";
    std::cout <<  " *     S-urfaces                 *\n";
    std::cout <<  " *     A-nalytiques              *\n";
    std::cout <<  " *********************************\n\n";
}

int CASALL_main(int argc,char ** argv)
{
  // cAppliApero * anAppli = cAppliMICMAC::Alloc(argc,argv,eAllocAM_STD);

  //if (0) delete anAppli;

   ELISE_ASSERT(argc>=2,"Not enough arg");

   cElXMLTree aTree(argv[1]);



   cResultSubstAndStdGetFile<cParamCasa> aP2
                                          (
                                              argc-2,argv+2,
                                              argv[1],
                                              StdGetFileXMLSpec("ParamCasa.xml"),
                                              "ParamCasa",
                                              "ParamCasa",
                                              "DirectoryChantier",
                                              "FileChantierNameDescripteur"
                                          );

   cAppli_Casa   anAppli (aP2);

   cElWarning::ShowWarns(aP2.mDC + "CasaWarn.txt");
   return EXIT_SUCCESS;
}
int CASA_main(int argc,char ** argv)
{
    std::string aNameN1;
    std::string aNameN2;
    std::string aNameN3;
    std::string aNameN4;
    std::string Out="TheCyl.xml";
    std::vector<std::string> aVPts;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameN1,"Name of Cloud", eSAM_IsExistFile),
        LArgMain()  << EAM(Out,"Out",true,"Name of result (Def=TheCyl.xml)")
                    <<  EAM(aNameN2,"N2",true,"Name of optional second cloud", eSAM_IsExistFile)
                    <<  EAM(aNameN3,"N3",true,"Name of optional third cloud", eSAM_IsExistFile)
                    <<  EAM(aNameN4,"N4",true,"Name of optional fourth cloud", eSAM_IsExistFile)
                    <<  EAM(aVPts,"PtsOri",true,"[Pts2D.xml,Ori], points and Orientation (used for sizing) to specify surface")
     );

     if (MMVisualMode) return EXIT_SUCCESS;

     std::string aCom =   MM3dBinFile(" TestLib CASALL ")
                       + XML_MM_File("ParamCasa.xml")
                       + " +Out=" + Out
                       + " +N1=" + aNameN1;

     if (EAMIsInit(&aNameN2))
        aCom = aCom + " +UseN2=true +N2=" + aNameN2;

     if (EAMIsInit(&aNameN3))
        aCom = aCom + " +UseN3=true +N3=" + aNameN3;

     if (EAMIsInit(&aNameN4))
        aCom = aCom + " +UseN4=true +N4=" + aNameN4;

     if (EAMIsInit(&aVPts))
     {
         ELISE_ASSERT(aVPts.size()==2,"Require 2 args for PtsOri");
         aCom = aCom + " +Pts=" + aVPts[0] + " +PtsOri=" + aVPts[1] + " +UsePts=true" ;
     }


     System(aCom);

     Casa_Banniere();

     return EXIT_SUCCESS;
}
/*
*/



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
