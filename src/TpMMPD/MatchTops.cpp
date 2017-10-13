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


const double JD2000 = 2451545.0; 	// J2000 in jd
const double J2000 = 946728000.0; 	// J2000 in seconds starting from 1970-01-01T00:00:00
const double MJD2000 = 51544.5; 	// J2000 en mjd
const double GPS0 = 315964800.0; 	// 1980-01-06T00:00:00 in seconds starting from 1970-01-01T00:00:00
const int LeapSecond = 18;			// GPST-UTC=18s


struct ImgTimes
{
    std::string ImgName;
    long double ImgUT; // system unix time
    long double ImgMJD; // GPS MJD time
};

struct Tops
{
    long double TopsUT; // system unix time
    int TopsWeek; // GPS week
    long double TopsSec; // GPS second of week
    long double TopsMJD; // GPS MJD time
};

std::vector<ImgTimes> ReadImgTimesFile(string & aDir, string aTimeFile, std::string aExt)
{
    std::vector<ImgTimes> aVSIT;
    ifstream aFichier((aDir + aTimeFile).c_str());
    if(aFichier)
    {
        std::string aLine;

        while(!aFichier.eof())
        {
            getline(aFichier,aLine,'\n');
            if(aLine.size() != 0)
            {
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aTime = strtok(aBuffer,"	");
                aTime += "L";
                std::string aImage = strtok(NULL,"	");

                ImgTimes aImgT;
                if(aExt != "")
                    aImgT.ImgName = aImage + aExt;
                else
                    aImgT.ImgName = aImage;

                aImgT.ImgUT = atof(aTime.c_str());

                aVSIT.push_back(aImgT);
            }
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }
    return aVSIT;
}

std::vector<Tops> ReadTopsFile(string & aDir, string aTopsFile, const bool & aUTC)
{
    std::vector<Tops> aVTops;
    ifstream aTopsFichier((aDir + aTopsFile).c_str());
    if(aTopsFichier)
    {
        std::string aTopsLine;
        getline(aTopsFichier,aTopsLine,'\n');
        getline(aTopsFichier,aTopsLine,'\n');

        while(!aTopsFichier.eof())
        {
            getline(aTopsFichier,aTopsLine,'\n');
            if(aTopsLine.size() != 0)
            {
                char *aTopsBuffer = strdup((char*)aTopsLine.c_str());
                std::string aUT = strtok(aTopsBuffer,"  ");
                aUT += "L";
                std::string aWeek = strtok(NULL,"  ");
                aWeek += "L";
                std::string aSec = strtok(NULL,"  ");
                aSec += "L";

                Tops aTops;
                aTops.TopsUT = atof(aUT.c_str());
                aTops.TopsWeek = atof(aWeek.c_str());
                aTops.TopsSec = atof(aSec.c_str());

                if(aUTC)
                    aTops.TopsSec -= LeapSecond;

                long double aS1970 = aTops.TopsWeek * 7 * 86400 + aTops.TopsSec + GPS0;

                long double aMJD = (aS1970 - J2000) / 86400 + MJD2000;

                aTops.TopsMJD=aMJD;

                aVTops.push_back(aTops);
            }
        }
        aTopsFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }
    return aVTops;
}

uint FindIdx(long double aUT, std::vector<Tops> aVTops)
{
    uint aI(0);
    for (uint iV=0; iV < aVTops.size(); iV++)
    {
        long double aRef = abs (aVTops.at(aI).TopsUT-aUT);
        long double aDif = abs (aVTops.at(iV).TopsUT-aUT);
        if (aRef > aDif)
            aI = iV;
    }
    return aI;
}

int MatchTops_main (int argc, char ** argv)
{
    std::string aDir, aNameTF, aTimeFile, aTopsFile, aExt=".thm.tif", aOutFile="ImgTM.xml";
    bool aMJD=1, aUTC=0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aTimeFile, "File of image system unix time (all_name_time.txt)", eSAM_IsExistFile)
                    << EAMC(aTopsFile, "Tops file", eSAM_IsExistFile),
        LArgMain()  << EAM(aExt,"Ext",true,"Extension of Imgs, Def = .thm.tif")
                    << EAM(aOutFile,"Out",true, "Output file, Def = ImgTM.xml")
                    << EAM(aMJD,"MJD",true,"If using MJD time, def=true")
                    << EAM(aUTC,"UTC",true,"If using UTC time, def=false, only useful when MJD=true")
    );

    SplitDirAndFile(aDir,aNameTF,aTimeFile);

    // read temperature file
    std::vector<ImgTimes> aVSIT = ReadImgTimesFile(aDir, aTimeFile, aExt);

    // read Tops file
    std::vector<Tops> aVTops = ReadTopsFile(aDir, aTopsFile, aUTC);

    uint aIdx = FindIdx(aVSIT.at(0).ImgUT,aVTops);

    cDicoImgsTime aDicoIT;
    for (uint iV=0; iV < aVSIT.size(); iV++)
    {
        cCpleImgTime aCpleIT;
        aCpleIT.NameIm() = aVSIT.at(iV).ImgName;
        if(aMJD)
            aCpleIT.TimeIm() = aVTops.at(iV+aIdx).TopsMJD;
        else
            aCpleIT.TimeIm() = aVTops.at(iV+aIdx).TopsSec;

        aDicoIT.CpleImgTime().push_back(aCpleIT);
    }

    MakeFileXML(aDicoIT,aOutFile);

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
