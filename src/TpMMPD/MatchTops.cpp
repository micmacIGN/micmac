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


// in case of rising and falling, test could be necessary to determine if rising was prior to falling or not. But for the moment this is not necessary as only rising time is used in the code

#include "StdAfx.h"
#include <iostream>
#include <string>


const double JD2000 = 2451545.0; 	// J2000 in jd
const double J2000 = 946728000.0; 	// J2000 in seconds starting from 1970-01-01T00:00:00
const double MJD2000 = 51544.5; 	// J2000 en mjd
const double GPS0 = 315964800.0; 	// 1980-01-06T00:00:00 in seconds starting from 1970-01-01T00:00:00
const int LeapSecond = 18;			// GPST-UTC=18s

//struct
struct towTime{
    double GpsWeek;
    double Tow; //or week second
};

struct ImgNameTime
{
    std::string ImgName;
    double ImgCRT; // camera raw time
    double ImgMJD; // MJD time
};

struct Tops
{
    int TopsGpsWeek;
    double TopsTow; //Rising edge
    double TopsCRT; //camera raw time
    double TopsMJD;
};

std::vector<ImgNameTime> ReadImgNameTimeFile(string aINTFile, std::string aExt)
{
    std::cout << "Read file " << aINTFile << ", 2 columns; image name and camera raw time. column delimiter; space or tabulation\n";
    std::vector<ImgNameTime> aVINT;
    std::ifstream aFichier(aINTFile.c_str());
    if(aFichier)
    {
        std::string aImgName;
        double aImgCRT;
        while(aFichier >> aImgName >> aImgCRT)
        {
            ImgNameTime aImgNT;
            aImgNT.ImgName=aImgName;
            aImgNT.ImgCRT=aImgCRT;
            aVINT.push_back(aImgNT);
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }
    std::cout << "Total Image : " << aVINT.size() << endl;
    std::cout << "First : Im = " << aVINT.at(0).ImgName << " CRT = " << aVINT.at(0).ImgCRT << endl;
    std::cout << "Last  : Im = " << aVINT.at(aVINT.size()-1).ImgName << " CRT = " << aVINT.at(aVINT.size()-1).ImgCRT << endl;
    return aVINT;
}


double towTime2MJD(const double GpsWeek, double Tow, const std::string & TimeSys)
{

    if(TimeSys == "UTC")
    {
        Tow += LeapSecond;
    }

    double aS1970 = GpsWeek * 7 * 86400 + Tow + GPS0;

    double aMJD = (aS1970 - J2000) / 86400 + MJD2000;

    return aMJD;
}

// Tops file: file containing signal at the beginning and at the end of the State "ON" of the camera sensor, so in other words: signal tagged with time at the beginning and end of carema trigger.
// signals are send by ublox chip, read by camlight and written in Tops txt file.
std::vector<Tops> ReadTopsFile(string aTops, const std::string & TimeSys)
{
    std::vector<Tops> aVTops;
    ifstream aFichier(aTops.c_str());
    int count_otherFlag(0);
    if(aFichier)
    {
        std::string aLine;

        getline(aFichier,aLine,'\n');
        getline(aFichier,aLine,'\n');

        double aUT, aRE, aFE, aCRT;
        int aWeek, aSeq;
        std::string aFlag;

        while(aFichier>>aUT>>aWeek>>aRE>>aFE>>aFlag>>aSeq>>aCRT)
        {
            // flag send by ublox gps to camlight, written in tops file, 4 situations: have received between 2 epochs 1) 1 rising and 1 falling, 2) One falling 3) One rising 4) more than one rising and one falling
            //rf , rf   , f     , r
            int Hexval(0);
            std::istringstream(aFlag) >> std::hex >> Hexval;
            // the value of the 8th bit of this exadecimal flag is true, which means a newRisingEdge is detected
            // le & simple est une comparaison bit à bit
            if ((Hexval & 0x80 )!=0){

                Tops aTops;
                aTops.TopsGpsWeek = aWeek;
                aTops.TopsTow = aRE;
                aTops.TopsCRT = aCRT;

                aTops.TopsMJD = towTime2MJD(aTops.TopsGpsWeek, aTops.TopsTow, TimeSys);

                aVTops.push_back(aTops);
            }
            else {
                count_otherFlag++;
            }
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }

    if (count_otherFlag!=0)  std::cout << "Other flag messages (removed)  without rising edge : " << count_otherFlag << endl;
    std::cout << "Total Tops : " << aVTops.size() << endl;
    std::cout << "First : CRT = " << aVTops.at(0).TopsCRT << endl;
    std::cout << "Last  : CRT = " << aVTops.at(aVTops.size()-1).TopsCRT << endl;

    return aVTops;
}

int calcul_ecart(std::vector<ImgNameTime> aVINT, std::vector<Tops> aVTops)
{
    int aEcart = -1;
    double aCRT0 = aVINT.at(0).ImgCRT;
    for (uint aV=0; aV<aVTops.size(); aV++)
    {
        if (  ((aVTops.at(aV).TopsCRT > aCRT0) && aV==0)  | ((aVTops.at(aV).TopsCRT > aCRT0) && (aV!=0) && (aVTops.at(aV-1).TopsCRT < aCRT0)))
        {
            aEcart = int(aV);
            break;
        }
    }
    if (aEcart==-1)
        std::cout << "Fail to match files!" << endl;

    std::cout << "Ecart (in number of triggering) = " << aEcart << endl;
    return aEcart;
}

// Tops = logs of the IGN Camlight (from ublox messages), containing camera Raw Time and GPS time
int ImgTMTxt2Xml_main (int argc, char ** argv)
{
    std::string aINTFile, aTops, aExt=".thm.tif", aOut="Img_TM.xml", aTSys="GPS", aPatFile;
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aINTFile, "File of image camera raw time (all_name_rawtime.txt)", eSAM_IsExistFile)
                << EAMC(aTops,"Tops file containing ToW and CRT (tops.txt)",eSAM_IsExistFile),
                LArgMain()  << EAM(aExt,"Ext",true,"Extension of Imgs, Def = .thm.tif")
                << EAM(aTSys,"TSys",true,"Time system, Def=GPS")
                << EAM(aOut,"Out",true,"Output matched file name, Def=Img_TM.xml")
                << EAM(aPatFile,"ImPat",true,"image pattern from which will be extracted camera raw time. If this arguement is provided, the file provided as first compulsory argument is overwritten.")
                );

    if (EAMIsInit(&aPatFile)){
        std::string aTmpFile("Tmp-MTD-CL.txt");
        cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
        std::list<std::string> aVImName = aICNM->StdGetListOfFile(aPatFile);
        FILE * aFOut = FopenNN(aINTFile.c_str(),"w","out");
        for (auto & imName : aVImName){
            // head: read and print the first 1220 bytes of the file (containing the metadata) . grep: option --binary-file=text because otherwise stop functionning
            std::string aCom="head " + imName + " -c 1220 | grep 'CAMERARAWTIME' --binary-files=text > " + aTmpFile;
            // ofset 512
            System(aCom);
            ifstream aFichier(aTmpFile.c_str());
            if(aFichier)
            {
                std::string aLine;
                getline(aFichier,aLine,'\n');
                // recover camera raw time value from the line
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aVal1Str = strtok(aBuffer,"=");
                std::string aCRT = strtok( NULL, " " );
                // save the name of the image and its CRT
                fprintf(aFOut,"%s %s\n",imName.c_str(),aCRT.c_str()); // tab to separate column
                aFichier.close();
                // argument aExt set to null because the above code save name of the image with extension, no need to add it afteward
            } else { std::cout << "Warn, I fail to read file " << aTmpFile << " that should have contains camera raw time from Camlight image " << imName << "\n";}
        }
        ElFclose(aFOut);
        aExt="";
        std::cout << "Camera raw time value extracted from " << aVImName.size() << " images and save in file " << aINTFile << " \n";
    }

    //read aImTimeFile
    std::vector<ImgNameTime> aVINT = ReadImgNameTimeFile(aINTFile, aExt);

    //read aTops
    std::cout << "Read file " << aTops << ", 5 columns; CamUT, week, rising edge, falling edge, flag, seq, cam raw time. column delimiter; space or tabulation\n";
    std::vector<Tops> aVTops = ReadTopsFile(aTops, aTSys);

    //calculate index difference
    int aEcart = calcul_ecart(aVINT, aVTops);
    std::cout << "CRT 1 for 1st image = " << aVINT.at(0).ImgCRT << "       Matched CRT in tops file = " << aVTops.at(aEcart).TopsCRT << endl;

    cDicoImgsTime aDicoIT;

    for(uint iV=0; iV<aVINT.size(); iV++)
    {
        int aV = int(iV+aEcart);
        cCpleImgTime aCpleIT;
        aCpleIT.NameIm() = aVINT.at(iV).ImgName;
        aCpleIT.TimeIm() = aVTops.at(aV).TopsMJD;
        aDicoIT.CpleImgTime().push_back(aCpleIT);
    }
    MakeFileXML(aDicoIT,aOut);
    std::cout << "Save result in file " << aOut << "\n";

    return EXIT_SUCCESS;
}

int GenImgTM_main (int argc, char ** argv)
{
    std::string aDir,aGPSFile,aGPSF,aOutINT="all_name_date.xml",aGPS_S="GPS_selected.txt",aGPS_L="GPS_left.xml";
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(aGPSFile, "File of GPS position and MJD time", eSAM_IsExistFile),
                LArgMain()  << EAM(aOutINT,"OutINT",true,"Output Img name/time couple file, Def = all_name_date.xml")
                << EAM(aGPS_S,"SGPS",true,"Output selected GPS file, Def = GPS_selected.txt")
                << EAM(aGPS_L,"SGPR",true,"Output left GPS file, Def = GPS_left.xml")
                );
    SplitDirAndFile(aDir,aGPSF,aGPSFile);

    //read .xml file
    cDicoGpsFlottant aDico_all = StdGetFromPCP(aGPSF,DicoGpsFlottant);
    std::vector<cOneGpsDGF> aVOneGps = aDico_all.OneGpsDGF();
    std::cout << "size of GPS obs: " << aVOneGps.size() << endl;


    cDicoImgsTime aDicoIT;
    cDicoGpsFlottant aDico_left;

    FILE * aFP = FopenNN(aGPS_S,"w","GenImgTM_main");
    cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aGPS_S);

    for (uint iV=0; iV<aVOneGps.size(); iV++)
    {
        if (iV%2==1)
        {
            cCpleImgTime aCpleIT;
            aCpleIT.NameIm()=aVOneGps.at(iV).NamePt();
            aCpleIT.TimeIm()=aVOneGps.at(iV).TimePt();
            aDicoIT.CpleImgTime().push_back(aCpleIT);
            fprintf(aFP,"%s %lf %lf %lf \n",aCpleIT.NameIm().c_str(), aVOneGps.at(iV).Pt().x, aVOneGps.at(iV).Pt().y, aVOneGps.at(iV).Pt().z);
        }
        else
        {
            cOneGpsDGF aOneGps;
            aOneGps.Pt().x=aVOneGps.at(iV).Pt().x;
            aOneGps.Pt().y=aVOneGps.at(iV).Pt().y;
            aOneGps.Pt().z=aVOneGps.at(iV).Pt().z;
            aOneGps.NamePt()=aVOneGps.at(iV).NamePt();
            aOneGps.TagPt()=aVOneGps.at(iV).TagPt();
            aOneGps.TimePt()=aVOneGps.at(iV).TimePt();
            aOneGps.Incertitude()=aVOneGps.at(iV).Incertitude();
            aDico_left.OneGpsDGF().push_back(aOneGps);
        }


    }

    MakeFileXML(aDicoIT,aOutINT);
    MakeFileXML(aDico_left,aGPS_L);
    ElFclose(aFP);

    return EXIT_SUCCESS;
}

// extract image name and time mark from txt file generated by Joe's (LOEMI) code fusgpsimg, which put GPS time stamp on camlight images

class cFusGPS2DicoImgTime;
class cReadImgTM;

class  cReadImgTM : public cReadObject
{
public :
    cReadImgTM(char aComCar,const std::string & aFormat) :
        cReadObject(aComCar,aFormat,"S"),
        mImName("toto")
    {
        AddString("N",&mImName,true);
        // gps week
        AddDouble("W",&mTopsGpsWeek,false);
        // gps second
        AddDouble("Sec",&mTopsTow,false);
    }
    std::string mImName;
    double mTopsGpsWeek;
    double mTopsTow;
};

class cFusGPS2DicoImgTime{
public:
    cFusGPS2DicoImgTime(int argc,char ** argv);
    cDicoImgsTime mDicoIT;
private:
    bool mDebug;
    std::string mFileIn, mOut;
    std::string mStrType;
    eTypeFichierApp mType;

};

cFusGPS2DicoImgTime::cFusGPS2DicoImgTime(int argc,char ** argv):
    mOut("Img_TM.xml")
{
    ElInitArgMain
            (
                argc,argv,
                LArgMain()  << EAMC(mStrType,"Format specification", eSAM_None, ListOfVal(eNbTypeApp))
                // to do : load several txt file if several flight? could be interresting
                // arg TimeSys
                // debug: std;;cout
                << EAMC(mFileIn, "Txt file with image name and GPS time, as the one generated with fusgpsimg", eSAM_IsExistFile),

                LArgMain()
                << EAM(mDebug,"Debug",true,"help debbuging by printing messages in terminal")
                << EAM(mOut,"Out",true,"Output matched file name, Def=Img_TM.xml")
                );


    if (!MMVisualMode)
    {

        bool Help;
        StdReadEnum(Help,mType,mStrType,eNbTypeApp,true);

        std::string aFormat;
        char        aCom;

        if (mType==eAppInFile)
        {
            bool Ok = cReadObject::ReadFormat(aCom,aFormat,mFileIn,true);
            ELISE_ASSERT(Ok,"File do not begin by format specification");
        }
        else
        {
            bool Ok = cReadObject::ReadFormat(aCom,aFormat,mStrType,false);
            ELISE_ASSERT(Ok,"Arg0 is not a valid format specif (AppInFile or '#F=N_W_I')");
        }
        std::cout << "Comment=[" << aCom<<"]\n";
        std::cout << "Format=[" << aFormat<<"]\n";

        char * aLine;
        int i(0);

        cReadImgTM aReadImgTM(aCom,aFormat);
        ELISE_fp aFIn(mFileIn.c_str(),ELISE_fp::READ);
        while ((aLine = aFIn.std_fgets()))
        {
            if (aReadImgTM.Decode(aLine))
            {
                cCpleImgTime aCpleIT;
                aCpleIT.NameIm() = aReadImgTM.mImName;
                aCpleIT.TimeIm() = towTime2MJD(aReadImgTM.mTopsGpsWeek, aReadImgTM.mTopsTow,"GPS");
                mDicoIT.CpleImgTime().push_back(aCpleIT);
            }
            i ++;
        }
        aFIn.close();

        MakeFileXML(mDicoIT,mOut);
    }
}

int main_Txt2CplImageTime(int argc, char ** argv)
{
    cFusGPS2DicoImgTime(argc,argv);
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
