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

#define DEF_OFSET -12349876

int RepLocBascule_main(int argc,char ** argv)
{
    NoInit = "NoP1P2";
    aNoPt  = Pt2dr(123456,-8765432);

    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);


    std::string  aDir,aPat,aFullDir;
    std::string AeroIn;
    std::string FileMesures ;
    std::string RepereOut;

    bool ExpTxt=false;
    std::string PostPlan="_Masq";
    bool OrthoCyl = false;
    bool P1P2Hor = false;
    bool P1P2HorVertY = false;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile )
                    << EAMC(AeroIn,"Input orientation", eSAM_IsExistDirOri)
                    << EAMC(FileMesures,"Image measures xml file, set \"HOR\" if horizontal line is wanted (HORVy if Y vertical), \"NONE\" if unused ", eSAM_IsExistFile)
                    << EAMC(RepereOut,"Output xml file", eSAM_IsOutputFile),
    LArgMain()
                    << EAM(ExpTxt,"ExpTxt",true,"Are tie points in ascii mode ? (Def=false)")
                    << EAM(PostPlan,"PostPlan",true,"Postfix for plane name, (Def=_Masq)")
                    << EAM(OrthoCyl,"OrthoCyl",true,"Is the coordinate system in ortho-cylindric mode?")

    );
    P1P2Hor = ((FileMesures=="HOR") || (FileMesures=="HORVy"));
    P1P2HorVertY =  (FileMesures=="HORVy");
    if (FileMesures=="NONE") FileMesures= "NoP1P2";

    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);

    if (EAMIsInit(&PostPlan))
    {
        CorrecNameMasq(aDir,aPat,PostPlan);
    }


    StdCorrecNameOrient(AeroIn,aDir);


    MMD_InitArgcArgv(argc,argv);
    std::string aCom =   MM3dBinFile( "Apero" )
                       + MMDir() + std::string("include/XML_MicMac/Apero-RLoc-Bascule.xml ")
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")

                       + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +AeroIn=-") + AeroIn
                       + std::string(" +RepereOut=") +  RepereOut
                       + std::string(" +FileMesures=") + FileMesures

                       + std::string(" +OrthoCyl=") + ToString(OrthoCyl)
                       + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                       + std::string(" +PostMasq=") + PostPlan
                    ;

   if (P1P2Hor) aCom = aCom + " +P1P2Hor=true";
   if (P1P2HorVertY) aCom = aCom + " +P1P2HorVertY=true";



   std::cout << "Com = " << aCom << "\n";
   int aRes = system_call(aCom.c_str());


   return aRes;
}

/*
int AppuiRepLoc_main(int argc,char ** argv)
{
    std::string aFileMes3D;
    std::string aFileRep;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFileMes3D,"Full name (Dir+Pat)", eSAM_IsPatFile ),
        LArgMain()
                    << EAM(aFileRep,"ExpTxt",true,"Are tie points in ascii mode ? (Def=false)")
    );
    if (!EAMIsInit(&aFileRep))
    {
       aFileRep = "Rep-"+aFileMes3D;
    }
}
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
