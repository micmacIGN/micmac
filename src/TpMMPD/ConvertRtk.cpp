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

//#include "StdAfx.h"
#include "ConvertRtk.h"

/*
const double JD2000 = 2451545.0; 	// J2000 in jd
const double J2000 = 946728000.0; 	// J2000 in seconds starting from 1970-01-01T00:00:00
const double MJD2000 = 51544.5; 	// J2000 en mjd
const double GPS0 = 315964800.0; 	// 1980-01-06T00:00:00 in seconds starting from 1970-01-01T00:00:00
const int LeapSecond = 18;			// GPST-UTC=18s

//class
class cRPG_Appli;

//struct
struct PosGPS{
	
    //Positions
    Pt3dr Pos;
    
    //Name or Number
    std::string Name;
    
    //vector of Quality
    int Flag;
    
    //Time expressed in Modified Julian day
    double Time;
    
    //Uncertainty
    Pt3dr Ic;
    
    //Correclation terms of Var-CoVar Matrix
    Pt3dr VarCoVar;
    
    //Number of satellites
    int NS;
    
    //Age of Age
    double Age;
    
    //Ratio Factor
    double Ratio;
};

//struct
struct hmsTime{
    double Year;
    double Month;
    double Day;
    double Hour;
    double Minute;
    double Second;
};

//struct
struct towTime{
    double GpsWeek;
    double Tow; //or wsec
};

class cRPG_Appli
{
     public :
          cRPG_Appli(int argc,char ** argv);
          double hmsTime2MJD(const hmsTime & Time, const std::string & TimeSys);
          double towTime2MJD(const towTime & Time, const std::string & TimeSys);
          void ShowHmsTime(const hmsTime & Time);
          void ShowTowTime(const towTime & Time);
     private :
        std::string mDir;
        std::string mFile;
        std::string mOut;
        std::string mStrChSys;
};

template <typename T> string NumberToString(T Number)
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}
*/

void ShowTowTime(const towTime & Time)
{
	std::cout << "Time.GpsWeek ==> " << Time.GpsWeek << std::endl;
	std::cout << "Time.Tow     ==> " << Time.Tow  << std::endl;
}

void ShowHmsTime(const hmsTime & Time)
{
	std::cout << "Time.Year   ==> " << Time.Year << std::endl;
	std::cout << "Time.Month  ==> " << Time.Month << std::endl;
	std::cout << "Time.Day    ==> " << Time.Day << std::endl;
	std::cout << "Time.Hour   ==> " << Time.Hour << std::endl;
	std::cout << "Time.Minute ==> " << Time.Minute << std::endl;
	std::cout << "Time.Second ==> " << Time.Second << std::endl;
}

double hmsTime2MJD(const hmsTime & Time, const std::string & TimeSys)
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

double towTime2MJD(const towTime & Time, const std::string & TimeSys)
{
	double aSec = Time.Tow;
	
    if(TimeSys == "UTC")
	{
        aSec += LeapSecond;
	}
	
	double aS1970 = Time.GpsWeek * 7 * 86400 + aSec + GPS0;
	
	double aMJD = (aS1970 - J2000) / 86400 + MJD2000;
	
	return aMJD;
}

hmsTime ElDate2hmsTime(const cElDate & aDate)
{
    hmsTime ahmsTime;
    ahmsTime.Year = aDate.Y();
    ahmsTime.Month = aDate.M();
    ahmsTime.Day = aDate.D();
    cElHour aHour = aDate.H();
    ahmsTime.Hour = aHour.H();
    ahmsTime.Minute = aHour.M();
    ahmsTime.Second = aHour.S();
    return ahmsTime;
}

cRPG_Appli::cRPG_Appli(int argc,char ** argv)
{
	bool aShowH = false;
	bool aXYZ = false;
	std::string aTimeSys="";
	int aCompt = 0;
	double aTimeF = 0;
    bool aMedian = false;
	
	
	std::vector<PosGPS> aVPosGPS;
	std::vector<Pt3dr> aVSauvPosGPS;
	
    Pt3dr aOffset(0,0,0);
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir, "Directory")
					 << EAMC(mFile, "RTKlib Output (MyFile.txt) file",  eSAM_IsExistFile),
          LArgMain() << EAM(mOut,"Out",false,"Output file name ; Def=MyFile.xml")
					 << EAM(mStrChSys,"ChSys",true,"Change coordinate file")
					 << EAM(aShowH,"ShowH",true,"Show header informations ; Def = false", eSAM_IsBool)
					 << EAM(aXYZ,"tXYZQ",false,"Export tXYZQ format ASCII data ; Def = false", eSAM_IsBool)
                     << EAM(aOffset,"OffSet",true,"Subtract an offset to all points")
                     << EAM(aMedian,"Median",false,"Export the median of coordinates; Def = false")
    );
    
    std::string aFullName = mDir+mFile;
    
    //name output (.xml) file
    if (mOut=="")
    {
		mOut = StdPrefixGen(mFile) + ".xml";
    }
    
    //read rtk input file
    ifstream aFichier(aFullName.c_str());

    if(aFichier)
    {
		std::string aLigne;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLigne);
            
            //if first string = % ==> it is a comment
            if(aLigne.compare(0,1,"%") == 0)						
            {
				
				//check time system (second comparaison to avoid all other lines starting also with '%')
				if(aLigne.compare(0,6,"%  UTC") == 0 && aLigne.compare(1,2,"  ") == 0)
				{
					aTimeSys = "UTC";
				}
				else if(aLigne.compare(0,7,"%  GPST") == 0 && aLigne.compare(1,2,"  ") == 0)
				{
					aTimeSys = "GPST";
				}
				else
				{
					aTimeSys = "NONE";
				}
				
				//show header infos
				if(aShowH)
				{
					std::cout << aLigne << std::endl;
				}
            }

            //if the line is not empty, read data
            //%  GPST                            x-ecef(m)      y-ecef(m)      z-ecef(m)   Q  ns   sdx(m)   sdy(m)   sdz(m)  sdxy(m)  sdyz(m)  sdzx(m) age(s)  ratio
			//2016/06/13 15:05:09.249187296   4214354.4165    172234.1963   4768501.6929   1   6   0.0147   0.0078   0.0133   0.0071   0.0020   0.0083   0.08  147.8
            if(!aLigne.empty() && aLigne.compare(0,1,"%") != 0)
            {	
				if(aTimeSys == "NONE")
				{
					ELISE_ASSERT(false,"Time System Not Supported");
                }

                char *aBuffer = strdup((char*)aLigne.c_str());
                std::string aTimeP1 = strtok(aBuffer," ");
                std::string aTimeP2 = strtok( NULL, " " );
                char *aX = strtok( NULL, " " );
                char *aY = strtok( NULL, " " );
                char *aZ = strtok( NULL, " " );
                char *aQ = strtok( NULL, " " );
                char *aNS = strtok( NULL, " " );
                char *aSDX = strtok( NULL, " " );
                char *aSDY = strtok( NULL, " " );
                char *aSDZ = strtok( NULL, " " );
                char *aSDXY = strtok( NULL, " " );
                char *aSDYZ = strtok( NULL, " " );
                char *aSDZX = strtok( NULL, " " );
                char *aAGE = strtok( NULL, " " );
                char *aRATIO = strtok( NULL, " " );
                
                aCompt++;
                
                //check output format : hms OR tow ?
                if(aTimeP1.size() > 4)
                {
                    //std::cout << "Detected Time Format = hms" << std::endl;
					
					hmsTime aHmsTps;
					aHmsTps.Year = atof(aTimeP1.substr(0,4).c_str());
                    //std::cout << "aHmsTps.Year == " << aHmsTps.Year << std::endl;
					aHmsTps.Month = atof(aTimeP1.substr(5,2).c_str());
                    //std::cout << "aHmsTps.Month == " << aHmsTps.Month << std::endl;
                    aHmsTps.Day = atof(aTimeP1.substr(8,2).c_str());
                    //std::cout << "aHmsTps.Day == " << aHmsTps.Day << std::endl;
					aHmsTps.Hour = atof(aTimeP2.substr(0,2).c_str());
                    //std::cout << "aHmsTps.Hour == " << aHmsTps.Hour << std::endl;
					aHmsTps.Minute = atof(aTimeP2.substr(3,2).c_str());
                    //std::cout << "aHmsTps.Minute == " << aHmsTps.Minute << std::endl;
					aHmsTps.Second = atof(aTimeP2.substr(6,aTimeP2.size()-6).c_str());
                    //std::cout << "aHmsTps.Second == " << aHmsTps.Second << std::endl;
					
					aTimeF = hmsTime2MJD(aHmsTps,aTimeSys);
					
				}
				
				else if(aTimeP1.size() == 4)
				{
					//std::cout << "Detected Time Format = tow" << std::endl;
					
					towTime aTowTps;
					aTowTps.GpsWeek = atof(aTimeP1.c_str());
					aTowTps.Tow = atof(aTimeP2.c_str());
					
					aTimeF = towTime2MJD(aTowTps,aTimeSys);
				}
				
				else
				{
					ELISE_ASSERT(false,"Time Format Not Supported");
				}
                
                Pt3dr aPt(atof(aX),atof(aY),atof(aZ));
                Pt3dr aInc(atof(aSDX),atof(aSDY),atof(aSDZ));
                Pt3dr aVar(atof(aSDXY),atof(aSDYZ),atof(aSDZX));
                
                //printf("aTimeF = %.15f \n", aTimeF);
                
                PosGPS aPosC;
                aPosC.Pos = aPt;
                aPosC.Name = NumberToString(aCompt);
                aPosC.Flag = atoi(aQ);
                aPosC.Time = aTimeF;
                aPosC.Ic = aInc;
                aPosC.VarCoVar = aVar;
                aPosC.NS = atoi(aNS);
                aPosC.Age = atof(aAGE);
                aPosC.Ratio = atof(aRATIO);
                
                aVPosGPS.push_back(aPosC);
                aVSauvPosGPS.push_back(aPt);
                
			}
            
		}
		
		std::cout << "Detected Time System = " << aTimeSys << std::endl;
        aFichier.close();  												
    }
    
    else
    {
		std::cout<< "Error While opening file" << '\n';
	}
			
	 //if changing coordinates system
     cChSysCo * aCSC = 0;
     
     if (mStrChSys!="")
		aCSC = cChSysCo::Alloc(mStrChSys,"");
		
	 if (aCSC!=0)
     {
		aVSauvPosGPS = aCSC->Src2Cibl(aVSauvPosGPS);
     }
	
	cDicoGpsFlottant  aDico;


    std::vector<double> aVCorX;
    std::vector<double> aVCorY;
    std::vector<double> aVCorZ;


	
    for (int aKP=0 ; aKP<int(aVSauvPosGPS.size()) ; aKP++)
    {
		cOneGpsDGF aOAD;
        aOAD.Pt() = aVSauvPosGPS.at(aKP) - aOffset;
        aOAD.NamePt() = aVPosGPS[aKP].Name;
        aOAD.Incertitude() = aVPosGPS[aKP].Ic;
        aOAD.TagPt() = aVPosGPS[aKP].Flag;
        aOAD.TimePt() = aVPosGPS[aKP].Time;

        aDico.OneGpsDGF().push_back(aOAD);

        if (aMedian && (aOAD.TagPt()==1))
        {
            aVCorX.push_back(aOAD.Pt().x);
            aVCorY.push_back(aOAD.Pt().y);
            aVCorZ.push_back(aOAD.Pt().z);
        }
	}

    MakeFileXML(aDico,mOut);

    if (aMedian)
    {
        double aCorX = MedianeSup(aVCorX);
        double aCorY = MedianeSup(aVCorY);
        double aCorZ = MedianeSup(aVCorZ);

        std::cout << std::setprecision(12) << "The median of coordinates = [" << aCorX << "," << aCorY << "," << aCorZ << "]" << endl;
    }
    
    //if (.txt) file export
    if(aXYZ)
    {
		if (!MMVisualMode)
		{
			std::string aTxtOut = StdPrefixGen(mFile) + "_txyz.txt";
			
			FILE * aFP = FopenNN(aTxtOut,"w","ConvertRtk_main");
				
			cElemAppliSetFile aEASF(mDir + ELISE_CAR_DIR + aTxtOut);
				
			for (unsigned int aK=0 ; aK<aVSauvPosGPS.size() ; aK++)
			{
				fprintf(aFP,"%lf %lf %lf %lf %d \n",aVPosGPS[aK].Time,aVSauvPosGPS[aK].x,aVSauvPosGPS[aK].y,aVSauvPosGPS[aK].z, aVPosGPS[aK].Flag);
			}
			
		ElFclose(aFP);
			
		}
	}
    
}

int ConvertRtk_main(int argc,char ** argv)
{
	 cRPG_Appli anAppli(argc,argv);
     return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
