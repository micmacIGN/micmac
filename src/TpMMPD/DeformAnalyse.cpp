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

int DeformAnalyse_main (int argc, char ** argv)
{
    std::string aPatImgs, aDir, aImgs, aPref, aSH="DM", aModel="Homot", aOutput="Output.txt";
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aPatImgs, "Input image pattern", eSAM_IsExistFile)
                            << EAMC(aPref, "Pref where , Dir=MEC-${Pref}-{Im1}-{Im2}"),
                LArgMain()  << EAM(aSH,"SH",true,"Set of homologue, Def=DM")
                            << EAM(aModel,"Model",true,"Model in [Homot,Simil,Affine,Homogr,Polyn], Def=Homot")
                            << EAM(aOutput,"Out",true,"Output file containing Scale and Tr, Def=Output.txt")
            );
    SplitDirAndFile(aDir,aImgs,aPatImgs);

    // 0.read images

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImg = *(aICNM->Get(aImgs));


    // 1.create output file
    FILE * aResult = FopenNN(aOutput,"w","Output");
    cElemAppliSetFile aEASF(aDir+ELISE_CAR_DIR+aOutput);

    for (uint aV=1; aV < aVImg.size(); aV++)
    {
        // 2.DMatch2Hom
        std::string aOut1 = "PerResidu.txt";
        std::string aCom1 = MM3dBinFile("DMatch2Hom")+" \""
                            +aPref+"\" "
                            +aVImg.at(0)+" "
                            +aVImg.at(aV)+" > "
                            +aOut1;
        system_call(aCom1.c_str());

        // 3. get the vector PerResidu
        ifstream aPR((aDir + aOut1).c_str());
        char *aPerR;
        if(aPR)
        {
            std::string aLine;

            while(!aPR.eof())
            {
                getline(aPR,aLine,'\n');
                getline(aPR,aLine,'\n');
                if(aLine.size() != 0)
                {
                    aPerR = strdup((char*)aLine.c_str());
                }
            }
            aPR.close();
        }
        else
        {
            std::cout<< "Error While opening file" << '\n';
        }

        // 4.CalcMapAnalytik with PerResidu
        std::string aOut2 = aModel + ".xml";
        std::string aCom2 = MM3dBinFile("CalcMapAnalytik") + " "
                            + aVImg.at(0) + " "
                            + aVImg.at(aV) + " "
                            + aModel + " "
                            + aOut2 + " "
                            + "SH="
                            + aSH + " "
                            + aPerR;
        system_call(aCom2.c_str());

        // 5.get Scale and Tr from the result of CalcMapAnalytik
        cXml_Map2D  aDico = StdGetFromSI(aOut2,Xml_Map2D);
        std::list<cXml_Map2DElem> & aLMaps = aDico.Maps();
        double aScale=0;
        Pt2dr aTr;
        for (auto iT=aLMaps.begin(); iT != aLMaps.end(); iT++)
        {
            aScale = (*iT).Homot().Val().Scale();
            aTr = (*iT).Homot().Val().Tr();
        }
        fprintf(aResult,"%f %f %f\n", aScale, aTr.x, aTr.y);

    }

    ElFclose(aResult);


	return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

