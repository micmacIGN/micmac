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



Bascule ".*.jpg" RadialExtended B2   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=90
Bascule ".*.jpg" RadialExtended R2.xml   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=45


Bascule ".*.jpg" RadialExtended B2 MesureIm=OutAligned-Im.xml Teta=90
Tarama ".*.jpg" B2

Bascule ".*.jpg" RadialExtended R2.xml MesureIm=OutAligned.xml Teta=180
Tarama ".*.jpg" RadialExtended Repere=R2.xml

Bascule ".*.jpg" RadialExtended R3.xml MesureIm=OutAligned.xml Teta=180

*/

#define DEF_OFSET -12349876

int SBGlobBascule_main(int argc,char ** argv)
{
    NoInit = "NoP1P2";
    aNoPt = Pt2dr(123456,-8765432);

    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;
    bool ExpTxt=false;
    std::string AeroIn;
    std::string AeroOut;
    std::string PostPlan="_Masq";
    std::string FileMesures ;
    std::string TargetRep = "ki" ;
    bool  CPI = false;


    double DistFE = 0;
    //Pt3dr  Normal;
    //Pt3dr  SNormal;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile )
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(FileMesures,"Images measures xml file", eSAM_IsExistFile)
                    << EAMC(AeroOut,"Out : orientation ", eSAM_IsOutputDirOri),
    LArgMain()
                    << EAM(ExpTxt,"ExpTxt",true)
                    << EAM(PostPlan,"PostPlan",true,"Set NONE if no plane")
                    << EAM(DistFE,"DistFS",true,"Distance between Ech1 and Ech2 to fix scale (if not given no scaling)")
                    << EAM(TargetRep,"Rep",true,"Target coordinate system (Def = ki, ie normal is vertical)")
                    << EAM(CPI,"CPI",true,"Calibration Per Image (Def=false)")

    );

    if (!MMVisualMode)
    {
#if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
        SplitDirAndFile(aDir,aPat,aFullDir);
        if (EAMIsInit(&PostPlan) && (PostPlan!="NONE"))
        {
            CorrecNameMasq(aDir,aPat,PostPlan);
        }

        StdCorrecNameOrient(AeroIn,aDir);


        MMD_InitArgcArgv(argc,argv);

        std::string aCom =   MM3dBinFile( "Apero" )
                + MMDir() + std::string("include/XML_MicMac/Apero-SB-Bascule.xml ")
                + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                + std::string(" +AeroOut=-") +  AeroOut
                + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                + std::string(" +AeroIn=-") + AeroIn
                + std::string(" +PostMasq=") + PostPlan
                + std::string(" +DistFE=") + ToString(DistFE)
                + std::string(" +RepNL=") + TargetRep
                + std::string(" +FileMesures=") + FileMesures
                + std::string(" +CPI=") + ToString(CPI)
                + std::string(" +AcceptNoPointInPlan=") + ToString(PostPlan=="NONE")
                ;



        std::cout << "Com = " << aCom << "\n";
        int aRes = system_call(aCom.c_str());


        return aRes;
    }
    else
    {
        return EXIT_SUCCESS;
    }
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
