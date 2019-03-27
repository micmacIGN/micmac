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
#include "ConvertRtk.h"

//lire le fichier de TimeMark
//lire le pattern
//checker le nombre ; si nombre different que faire ?
//lire les exifs des images si elles existent
//convertir en MJD les datations d'images si elles existent
//calculer les différences de temps données par les exifs & le TimeMark
//calculer la corrélation entre les deux
//problème quand aTMPDir existe mais est vide ... ça plante

const std::string aTMPDir = "Tmp-MM-Dir";

//const double JD2000 = 2451545.0; 	// J2000 in jd
//const double J2000 = 946728000.0; 	// J2000 in seconds starting from 1970-01-01T00:00:00
//const double MJD2000 = 51544.5; 	// J2000 en mjd
//const double GPS0 = 315964800.0; 	// 1980-01-06T00:00:00 in seconds starting from 1970-01-01T00:00:00
//const int LeapSecond = 18;			// GPST-UTC=18s

////struct
//struct hmsTime{
//	double Year;
//	double Month;
//	double Day;
//	double Hour;
//	double Minute;
//	double Second;
//};

//struct TimeMark file
struct TM{
	
	//Number
	int Number;
	
	//Time MJD
	double MJD;
	
	//Numer of satellites
	int SatNbr;
	
	//Lat
	double Lat;
	
	//Lon
	double Lon;
	
	//Heigth
	double Heigth;
};

class cMITM_Appli
{
	public :
		cMITM_Appli(int argc,char ** argv);
//		void ShowHmsTime(const hmsTime & Time);
//		double hmsTime2MJD(const hmsTime & Time, const std::string & TimeSys);
		std::vector<TM> ReadTMFile(std::string & aInputTMFile);
		void CheckTmpMMDir();
		std::vector<cXmlDate> ReadExifData();
		std::vector<double> ExifDate2Mjd(const std::vector<cXmlDate> aVExifDate);
		void ExportDicoTmImg(std::vector<double> aTM, std::vector<std::string> aImgName, std::string aOutFile);
		std::vector<double> ComputeSuccDiff(const std::vector<double> aV, bool aShow);
		std::vector<double> GetTmMjdFromVTM(const std::vector<TM> aVTM);
		double CalcSomme(const std::vector<double> aV);
		double CalcCorr(const std::vector<double> aV1, const std::vector<double> aV2);
		
	private :
	    std::string mFullName;
		std::string mDir;
		std::string mPat;
		std::string mFile;
		std::string aOut;
		cInterfChantierNameManipulateur * mICNM;
			
};

//display time
//void cMITM_Appli::ShowHmsTime(const hmsTime & Time)
//{
//	std::cout << "Time.Year   ==> " << Time.Year << std::endl;
//	std::cout << "Time.Month  ==> " << Time.Month << std::endl;
//	std::cout << "Time.Day    ==> " << Time.Day << std::endl;
//	std::cout << "Time.Hour   ==> " << Time.Hour << std::endl;
//	std::cout << "Time.Minute ==> " << Time.Minute << std::endl;
//	std::cout << "Time.Second ==> " << Time.Second << std::endl;
//}

//convert HMS format to MJD (Exif data are given in HMS format)
/*
double cMITM_Appli::hmsTime2MJD(const hmsTime & Time, const std::string & TimeSys)
{
	
	double aYear;
	double aMonth;
	double aSec = Time.Second;
	
	//std::cout << "aSec = " << aSec << std::endl;
	
	if(TimeSys == "UTC")
	{
        aSec += LeapSecond;
	}
	
	//std::cout << "aSec = " << aSec << std::endl;
	
	//2 or 4 digits year management
	if(Time.Year < 80)
	{
		aYear = Time.Year + 2000;
	}
	else if(Time.Year < 100)
	{
		aYear = Time.Year + 1900;
	}
	else
	{
		aYear = Time.Year;
	}
	
	//months
	if(Time.Month <= 2)
	{
		aMonth = Time.Month + 12;
		aYear = Time.Year - 1;
	}
	else
	{
		aMonth = Time.Month;
	}
	
	//std::cout << "aYear = " << aYear << std::endl;
	//std::cout << "aMonth = " << aMonth << std::endl;
	
	double aC = floor(aYear / 100);
	//std::cout << "aC = " << aC << std::endl;
	
	double aB = 2 - aC + floor(aC / 4);
	//std::cout << "aB = " << aB << std::endl;
	
	double aT = (Time.Hour/24) + (Time.Minute/1440) + (aSec/86400);
	//printf("aT = %.15f \n", aT);
	
	double aJD = floor(365.25 * (aYear+4716)) + floor(30.6001 * (aMonth+1)) + Time.Day + aT + aB - 1524.5;
	//printf("aJD = %.15f \n", aJD);
	
	double aS1970 = (aJD - JD2000) * 86400 + J2000; // seconds starting from 1970-01-01T00:00:00
	//printf("aS1970 = %.15f \n", aS1970);
	
	double aMJD = (aS1970 - J2000) / 86400 + MJD2000;
	//printf("aMJD = %.15f \n", aMJD);
	
	return aMJD;
	
}
*/

//read TimeMark File
std::vector<TM> cMITM_Appli::ReadTMFile(std::string & aInputTMFile)
{
	std::vector<TM> aVTM;
	
	//read TimeMark input file
    ifstream aFichier(aInputTMFile.c_str());

    if(aFichier)
    {
		std::string aLigne;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLigne);
			
			//if first string = # ==> it is a comment
			if(aLigne.compare(0,1,"#") == 0)						
            {
				std::cout << "# Comment = " << aLigne << std::endl;
			}
			
			if(!aLigne.empty() && aLigne.compare(0,1,"#") != 0)
			{
				
				char *aBuffer = strdup((char*)aLigne.c_str());
				char *aNbr = strtok(aBuffer," ");
                char *aMjd = strtok( NULL, " " );
                char *aNbSats = strtok( NULL, " " );
                char *aLat = strtok( NULL, " " );
                char *aLon = strtok( NULL, " " );
                char *aH = strtok( NULL, " " );
                
                TM aTM;
                aTM.Number = atoi(aNbr);
                aTM.MJD = atof(aMjd);
                aTM.SatNbr = atoi(aNbSats);
                aTM.Lat = atof(aLat);
                aTM.Lon = atof(aLon);
                aTM.Heigth = atof(aH);
                
                aVTM.push_back(aTM);
			}
		}
		
		aFichier.close(); 
	}
	
	return aVTM;
}

//check Tmp-MM-Dir Directory
void cMITM_Appli::CheckTmpMMDir()
{
	system_call((std::string("mm3d MMXmlXif \"")+mFullName+"\"").c_str());
}

//read exif data, convert to Mjd
std::vector<cXmlDate> cMITM_Appli::ReadExifData()
{
	std::vector<cXmlDate> aVExifDate;
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aTMPDir);
    std::list<std::string> Img_xif = aICNM->StdGetListOfFile(".*xml");

    for(std::list<std::string>::iterator I=Img_xif.begin(); I!=Img_xif.end(); I++)
    {
        cXmlXifInfo aXmlXifInfo=StdGetFromPCP(aTMPDir+"/"+*I,XmlXifInfo);
        
        if(aXmlXifInfo.Date().IsInit())
        {
            aVExifDate.push_back(aXmlXifInfo.Date().Val());
        }
        
        else
        {
            std::cout << "No Date Time in Exif ! " << std::endl;
        }
    }
    
    return aVExifDate;
}


//convert exif data Time to Mjd double values
std::vector<double> cMITM_Appli::ExifDate2Mjd(const std::vector<cXmlDate> aVExifDate)
{
	std::vector<double> aVMjdImgs;

	for(unsigned int aK=0 ; aK<aVExifDate.size() ; aK++)
	{
		hmsTime aTime;
		aTime.Year = aVExifDate.at(aK).Y();
		aTime.Month = aVExifDate.at(aK).M();
		aTime.Day = aVExifDate.at(aK).D();
		aTime.Hour = aVExifDate.at(aK).Hour().H();
		aTime.Minute = aVExifDate.at(aK).Hour().M();
		aTime.Second = aVExifDate.at(aK).Hour().S();
		
		//std::cout << "********************" << std::endl;
		//ShowHmsTime(aTime);
		//std::cout << "********************" << std::endl;
		
		double aMjd =  hmsTime2MJD(aTime,"UTC");
		aVMjdImgs.push_back(aMjd);
	}
	
	return aVMjdImgs;
}


//make .xml export of TimeMark and Images
void  cMITM_Appli::ExportDicoTmImg(std::vector<double> aTM, std::vector<std::string> aImgName, std::string aOutFile)
{
	cDicoImgsTime aDicoIT;
	
	for(unsigned int aK=0; aK < aTM.size() ; aK++)
	{
		cCpleImgTime aCpleIT;
		aCpleIT.NameIm() = aImgName.at(aK);
		aCpleIT.TimeIm() = aTM.at(aK);
		
		aDicoIT.CpleImgTime().push_back(aCpleIT);
	}
	
	MakeFileXML(aDicoIT,aOutFile);
}


//compute form n-dimentional vector (n-1)-dimentional successive differences
std::vector<double> cMITM_Appli::ComputeSuccDiff(const std::vector<double> aV, bool aShow)
{
	std::vector<double> aSuccDiff;
	
	for(unsigned aK=0 ; aK<aV.size()-1 ; aK++)
	{
		double aVal = aV.at(aK+1) - aV.at(aK);
		
		if(aShow)
		{
			printf("Tps2-Tps1 = %lf \n",aVal);
		}
		
		aSuccDiff.push_back(aVal);
	}
	
	return aSuccDiff;
}

//get only double vector TimeMark MJD values from a TM structure vector
std::vector<double> cMITM_Appli::GetTmMjdFromVTM(const std::vector<TM> aVTM)
{
	std::vector<double> aVMjd;
	
	for (unsigned aK=0 ; aK < aVTM.size() ; aK++)
	{
		double aVal = aVTM.at(aK).MJD;
		aVMjd.push_back(aVal);
	}
	
	return aVMjd;
}

//compute sum of a vector
double cMITM_Appli::CalcSomme(const std::vector<double> aV)
{
	double aSum=0;
	
    for (unsigned int aK=0 ; aK<aV.size() ; aK++)
    {
		aSum += aV[aK];
	}
	
	return aSum;
}


//compute corr coeff of 2 1-dimentional vectors
double cMITM_Appli::CalcCorr(const std::vector<double> aV1, const std::vector<double> aV2)
{
	double aCorrCoeff = 0;
	
	if(aV1.size() != aV2.size())
	{
		ELISE_ASSERT(false,"Note same size for both vectors !");
	}
	
	else
	{
		double aSumV1 = CalcSomme(aV1);
		double aSumV2 = CalcSomme(aV2);
	
		double aMeanV1 = aSumV1 / aV1.size();
		double aMeanV2 = aSumV2 / aV2.size();
		
		double aSum1_Pow2 = 0;
		double aSum2_Pow2 = 0;
		
		double aMult = 0;
		
		for (unsigned int aK=0; aK<aV1.size(); aK++)
		{
			aSum1_Pow2 +=  (aV1[aK] - aMeanV1)*(aV1[aK] - aMeanV1);
			aSum2_Pow2 +=  (aV2[aK] - aMeanV2)*(aV2[aK] - aMeanV2);
			
			aMult += (aV1[aK] - aMeanV1)*(aV2[aK] - aMeanV2);
		}
		
		aCorrCoeff = aMult/sqrt(aSum1_Pow2*aSum2_Pow2);
	}
	
	return aCorrCoeff;
}


cMITM_Appli::cMITM_Appli(int argc,char ** argv)
{
	ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(mFullName,"Full Name (Dir+Pat)")
					 << EAMC(mFile, "TimeMark File",  eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Name output file (def=NameFile_MatchImgs.txt)")
     );
     
     SplitDirAndFile(mDir, mPat, mFullName);
     std::cout << "Working dir: " << mDir << std::endl;
     std::cout << "Images pattern: " << mPat << std::endl;
     
     cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(mDir);
     const std::vector<std::string> aSetIm = *(aICNM->Get(mPat));
	 
	 for (unsigned int aK=0 ; aK < aSetIm.size() ; aK++)
	 {
		std::cout << " - " << aSetIm[aK] << std::endl;
	 }
	 
	//name output (.xml) file
    if (aOut=="")
    {
		aOut = StdPrefixGen(mFile) + "_MatchImgs.xml";
    }
    
    //check Tmp-MM-Dir Directory
    CheckTmpMMDir();
    
    //read exif data, convert to Mjd and compare differences of Time
    std::vector<cXmlDate> aVExifDate = ReadExifData();
    
    //convert exif data Time 2 Mjd values
    std::vector<double> aVMjdImgs = ExifDate2Mjd(aVExifDate);
    
    //read TimeMark file
    std::vector<TM> aVTM = ReadTMFile(mFile);
    
    //get only double TimeMark Values from <Tm> structure vector
    std::vector<double> aVTimeMark = GetTmMjdFromVTM(aVTM);
    
    //check coherence with GPS TimeMark
    if(aVTimeMark.size() != aSetIm.size())
    {
		ELISE_ASSERT(false,"Note same number of images and TimeMark !");
	}
    
    //export vector of (TimeMark,Imgs)
    ExportDicoTmImg(aVTimeMark,aSetIm,aOut);
    
    //compute time mark differences in sec (gps)
    std::cout << "***********GNSS**********" << std::endl;
    std::vector<double> aVecSuccDiffTM = ComputeSuccDiff(aVTimeMark,true);

    //if Exif data is available
    if(aVMjdImgs.size() != 0)
    {
		//compute images time differences in sec (exif)
		std::cout << "***********EXIF**********" << std::endl;
		std::vector<double> aVecSuccDiffImgs = ComputeSuccDiff(aVMjdImgs,true);
    
		//give value of coherence
		std::cout << "***********CORR**********" << std::endl;
		double aCorr = CalcCorr(aVTimeMark,aVMjdImgs);
		printf("Correlation = %lf \n",aCorr);
	}
}


int MatchinImgTM_main(int argc,char ** argv)
{
	cMITM_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
associés au chargement,  a  l'utilisation,  a  la modification et/ou au
développement et a  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe a
manipuler et qui le réserve donc a  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités a  charger  et  tester  l'adéquation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
a  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder a  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
