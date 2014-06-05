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

int AperiCloud_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;

    std::string AeroIn;
    //std::vector<std::string> ImPl;
    int ExpTxt=0;
    int PlyBin=1;
    int CalPerIm=0;
    std::string Out="";

    int RGB = -1;
    double aSeuilEc = 10.0;
    double aLimBsH;
    bool   WithPoints = true;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation directory", eSAM_IsExistDirOri),
    LArgMain()
                    << EAM(ExpTxt,"ExpTxt",true,"Point in txt format ? (Def=false)", eSAM_IsBool)
                    << EAM(Out,"Out",true,"Result (Def=AperiCloud.ply)", eSAM_IsOutputFile)
                    << EAM(PlyBin,"Bin",true,"Ply in binary mode (Def=true)", eSAM_IsBool)
                    << EAM(RGB,"RGB",true,"Use RGB image to texturate points (Def=true)", eSAM_IsBool)
                    << EAM(aSeuilEc,"SeuilEc",true,"Max residual (Def=10)")
                    << EAM(aLimBsH,"LimBsH",false,"Limit ratio base to height (Def=1e-2)")
                    << EAM(WithPoints,"WithPoints",true,"Do we add point cloud? (Def=true) ",eSAM_IsBool)
                    << EAM(CalPerIm,"CalPerIm",true,"If a calibration per image was used (Def=false)",eSAM_IsBool)
    );

    if (!MMVisualMode)
    {
        if (RGB >=0)
        {
            RGB = RGB ? 3  : 1;
        }

        string aXmlName="Apero-Cloud.xml";
        if (CalPerIm)
        {
            aXmlName="Apero-Cloud-PerIm.xml";
        }

#if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
        SplitDirAndFile(aDir,aPat,aFullDir);

        StdCorrecNameOrient(AeroIn,aDir);
        if (Out=="")
        {
            Out="AperiCloud_" + AeroIn + ".ply";
        }


        //std::string aCom =   MMDir() + std::string("bin" ELISE_STR_DIR  "Apero ")
        //                   + MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Cloud.xml ")
        std::string aCom =   MM3dBinFile_quotes("Apero")
                + ToStrBlkCorr( MMDir()+std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR)+ aXmlName)+" "
                + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                + std::string(" +Ext=") + (ExpTxt?"txt":"dat")
                + std::string(" +AeroIn=-") + AeroIn
                + std::string(" +Out=") + Out
                + std::string(" +PlyBin=") + (PlyBin?"true":"false")
                + std::string(" +NbChan=") +  ToString(RGB)
                + std::string(" +SeuilEc=") +  ToString(aSeuilEc)
                ;

        if (! WithPoints)
        {
            aCom = aCom + std::string(" +KeyAssocImage=NKS-Assoc-Cste@NoPoint");
        }

        if (EAMIsInit(&aLimBsH))
            aCom = aCom + std::string(" +LimBsH=") + ToString(aLimBsH);

        std::cout << "Com = " << aCom << "\n";
        int aRes = System(aCom.c_str());

        return aRes;
    }
    else
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
