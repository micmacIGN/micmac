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
#include <iostream>
#include <string>

struct ImgT{
    std::string ImgName;
    double ImgTemp;
};

class cReechHomol_Appli
{
public :
    cReechHomol_Appli (int argc,char ** argv);
    void ConvertHomol (string & aFullPat, string & aSHIn);
    void CorrHomolFromTemp (string & aDir, string & aSHIn, string & aTempFile, std::vector<ImgT> & aVSIT, string & aExt, string & aPrefix);
    void ReadImgTFile (string & aDir, string aTempFile, std::vector<ImgT> & aVSIT, std::string aExt);
private :
    std::string mDir;
};

void cReechHomol_Appli::ConvertHomol(string & aFullPat, string & aSHIn)
{
    string aComConvertHomol = MM3dBinFile("TestLib Homol2Way ")
            + aFullPat
            + " SH=" + aSHIn
            + " SHOut=_txt"
            + " IntTxt=0"
            + " ExpTxt=1"
            + " OnlyConvert=1";
    system_call(aComConvertHomol.c_str());
}

void cReechHomol_Appli::ReadImgTFile(string & aDir, string aTempFile, std::vector<ImgT> & aVSIT, std::string aExt)
{
    ifstream aFichier((aDir + aTempFile).c_str());
    if(aFichier)
    {
        std::string aLine;
        
        while(!aFichier.eof())
        {
            getline(aFichier,aLine,'\n');
            if(aLine.size() != 0)
            {
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aImage = strtok(aBuffer,"	");
                std::string aTemp = strtok(NULL, "	");

                ImgT aImgT;
                if(aExt != "")
                    aImgT.ImgName = aImage + aExt;
                else
                    aImgT.ImgName = aImage;

                aImgT.ImgTemp = atof(aTemp.c_str());

                aVSIT.push_back(aImgT);
            }
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }
}

void cReechHomol_Appli::CorrHomolFromTemp(string & aDir, string & aSHIn, string & aTempFile, std::vector<ImgT> & aVSIT, string & aExt, string & aPrefix)
{
    cReechHomol_Appli::ReadImgTFile(aDir, aTempFile, aVSIT, aExt); //read all_name_temp.txt

    // get all converted Patis folders
    string aDirHomol = aSHIn + "_txt/";
    std::list<cElFilename> aLPatis;
    ctPath * aPathHomol = new ctPath(aDirHomol);
    aPathHomol->getContent(aLPatis);

    // for one Patis folder
    for (std::list<cElFilename>::iterator iT1 = aLPatis.begin() ; iT1 != aLPatis.end() ; iT1++)
    {
        // master image
        string aImMaster = iT1->m_basename.substr (6,25);
        cout << "Master Image: " << aImMaster << endl;
        std::string aNameMapMaster = "Deg_" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
        cElMap2D * aMapMasterIm = cElMap2D::FromFile(aNameMapMaster);

        // match the temperature for master image
        for (uint aV=0; aV < aVSIT.size(); aV++)
        {
            if (aVSIT.at(aV).ImgName.compare(aImMaster) == 0)
            {
                aNameMapMaster = "Deg_" + ToString(aVSIT.at(aV).ImgTemp) + ".xml";
                * aMapMasterIm->FromFile(aNameMapMaster);
            }
        }

        // get all .txt files of the master image
        string aDirPatis = aDirHomol + iT1->m_basename;
        cInterfChantierNameManipulateur * aICNMP = cInterfChantierNameManipulateur::BasicAlloc(aDirPatis);
        vector<string> aLFileP = *(aICNMP->Get(".*"));

        // matche the temperature for secondary image and correct with maps
        for (uint aL=0; aL < aLFileP.size(); aL++)
        {
            // secondary image
            string aImSecond = aLFileP.at(aL).substr(1,25);
            cout << "Secondary Image: " << aImSecond << endl;
            std::string aNameMapSecond = "Deg_" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
            cElMap2D * aMapSecondIm = cElMap2D::FromFile(aNameMapSecond);

            for (uint aT=0; aT < aVSIT.size(); aT++)
            {
                // match the temperature for secondary image
                if (aVSIT.at(aT).ImgName.compare(aImSecond) == 0)
                {
                    aNameMapSecond = "Deg_" + ToString(aVSIT.at(aT).ImgTemp) + ".xml";
                    * aMapSecondIm->FromFile(aNameMapSecond);
                }
            }

            // read the .txt file and apply maps
            std::vector<std::vector<double> >     aDataBrute;
            std::vector<std::vector<double> >     aDataCorr;

            std::ifstream aFileBrute((aDirPatis+aLFileP.at(aL)).c_str());

            std::string aLine;
            while(std::getline(aFileBrute, aLine))
            {
                std::vector<double>   aLineData;
                std::stringstream  lineStream(aLine);

                double value;
                while(lineStream >> value)
                {
                    aLineData.push_back(value);
                }
                aDataBrute.push_back(aLineData);
            }

            // apply map to raw data
            ElPackHomologue * bb =new ElPackHomologue();

            for (uint i = 0; i < aDataBrute.size(); i++)
            {
                std::vector<double> aLineDataCorr;
                Pt2dr aCorrMasterIm =Pt2dr(aDataBrute[i][0],aDataBrute[i][1])*2-(*aMapMasterIm)(Pt2dr(aDataBrute[i][0],aDataBrute[i][1]));
                Pt2dr aCorrSecondIm =Pt2dr(aDataBrute[i][2],aDataBrute[i][3])*2-(*aMapSecondIm)(Pt2dr(aDataBrute[i][2],aDataBrute[i][3]));
                ElCplePtsHomologues aa (aCorrMasterIm,aCorrSecondIm);
                bb->Cple_Add(aa);
            }

            std::string aImMasterCorr = aPrefix + aImMaster;
            std::string aImSecondCorr = aPrefix + aImSecond;

            std::string aDirHomolCorr = aSHIn + "_Reech/";
            cout << aDirHomol << endl;
            std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                    +  std::string(aDirHomolCorr)
                    +  std::string("@")
                    +  std::string("dat");
            cout << aKHIn << endl;

            cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
            std::string aHmIn= aICNM->Assoc1To2(aKHIn, aImMasterCorr, aImSecondCorr, true);
            cout << aHmIn << endl;
            bb->StdPutInFile(aHmIn);
        }
    }
}

cReechHomol_Appli::cReechHomol_Appli(int argc,char ** argv)
{
    std::string aFullPat, aDir, aPatImgs, aSHIn, aTempFile, aExt = ".thm.tif", aPrefix = "Reech_";
    std::vector<ImgT> aVSIT;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aFullPat, "Full Imgs Pattern", eSAM_IsExistFile)
                            << EAMC(aSHIn, "Input Homol folder", eSAM_IsExistFile)
                            << EAMC(aTempFile, "file containing image name & corresponding temperature", eSAM_IsExistFile),
                LArgMain()  << EAM(aExt,"Ext",true,"Extension of Imgs, Def = .thm.tif")
                            << EAM(aPrefix,"Prefix",true,"Prefix for resampled Imgs, Def = Reech_")
                );
    
    SplitDirAndFile(aDir,aPatImgs,aFullPat);
    
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatImgs));
    
    cout << "File size = " << aSetIm.size() << endl;

    cReechHomol_Appli::ConvertHomol(aFullPat, aSHIn);
    cReechHomol_Appli::CorrHomolFromTemp(aDir, aSHIn, aTempFile, aVSIT, aExt, aPrefix);
}

int ReechHomol_main(int argc, char ** argv)
{

    cReechHomol_Appli anAppli(argc,argv);
    return EXIT_SUCCESS;
}

int ExtraitHomol_main(int argc, char ** argv)
{
    std::string aDir, aPatImgs, aFullPat, aIn="", aOut="_extrait";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullPat, "Full Imgs Pattern", eSAM_IsExistFile),
        LArgMain()  << EAM(aIn,"SHIn",true,"Input of extracted Homol, Def = Homol")
                    << EAM(aOut,"SHOut",true,"Output of extracted Homol, Def = Homol_extrait")
    );

    SplitDirAndFile(aDir,aPatImgs,aFullPat);

    // read corresponding .dat files
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aVImg = *(aICNM->Get(aPatImgs));

    ElPackHomologue aPckIn;
    ElPackHomologue aPckOut;
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aIn)
                       +  std::string("@")
                       +  std::string("dat");
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aOut)
                       +  std::string("@")
                       +  std::string("dat");


    for (uint i=1; i < aVImg.size(); i++)
    {
        std::string aIm1 = aVImg.at(0);
        std::string aIm2 = aVImg.at(i);

        std::string aHmIn= aICNM->Assoc1To2(aKHIn, aIm1, aIm2, true);

        aPckIn = ElPackHomologue::FromFile(aHmIn);
        cout << "File ++ : " << aIm1 << " & " << aIm2 << endl;

        std::string aHmOut= aICNM->Assoc1To2(aKHOut, aIm1, aIm2, true);
        cout << aHmOut << endl;
        aPckOut.StdPutInFile(aHmOut);
    }

    // compare points

    // rewrite selected tie points


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
