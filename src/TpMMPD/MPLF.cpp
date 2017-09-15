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

template <typename T> string NumberToString(T Number)
{
	ostringstream ss;
    ss << Number;
    return ss.str();
}

template <typename T1, typename T2>
struct t_unpair
{
	T1& a1;
	T2& a2;
	explicit t_unpair( T1& a1, T2& a2 ): a1(a1), a2(a2) { }
	t_unpair<T1,T2>& operator = (const pair<T1,T2>& p)
    {
		a1 = p.first;
		a2 = p.second;
		return *this;
    }
};

template <typename T1, typename T2>
t_unpair<T1,T2> unpair( T1& a1, T2& a2 )
{
  return t_unpair<T1,T2>( a1, a2 );
}

const double JD2000 = 2451545.0; 	// J2000 in jd
const double J2000 = 946728000.0; 	// J2000 in seconds starting from 1970-01-01T00:00:00
const double MJD2000 = 51544.5; 	// J2000 en mjd
const double GPS0 = 315964800.0; 	// 1980-01-06T00:00:00 in seconds starting from 1970-01-01T00:00:00
const int LeapSecond = 18;			// GPST-UTC=18s

struct towTime{
	double GpsWeek;
	double Tow; //or wsec
};
//CAM, QIHLLeeeccC, TimeUS,GPSTime,GPSWeek,Lat,Lng,Alt,RelAlt,GPSAlt,Roll,Pitch,Yaw
//CAM, 1260919656, 220518000, 1954, 48.8444203, 1.4597241, 158.37, 68.20, 158.16, 5.10, -2.80, 243.96
struct CAM{
	double TimeUS;
	double GPSTime; // in milliseconds
	int GPSWeek;
	double Lat;
	double Lng;
	double Alt;
	double RelAlt;
	double GPSAlt;
	double Roll;
	double Pitch;
	double Yaw;
};
//GPS, QBIHBcLLefffB, TimeUS,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ,U
//GPS, 1180441285, 3, 220437600, 1954, 12, 0.91, 48.843854, 1.459903, 94.56, 0.027, 0, 0.004, 0
//GPS2,1260807688, 6, 220518000, 1954, 12, 0.94, 48.8444246,1.4597341, 158.16, 3.331549, 243.3775, -0.6531352, 1
struct GPS{
	double TimeUS;
	int Status;
	double GMS;  // in milliseconds
	int GWk;
	int NSats;
	double HDop;
	double Lat;
	double Lng;
	double Alt;
	double Spd;
	int GCrs;
	double VZ;
	float U;
};
//POS, QLLfff, TimeUS,Lat,Lng,Alt,RelHomeAlt,RelOriginAlt
//POS, 1180414380, 48.8438484, 1.4598758, 90.17, 0.006674996, 0.656675
struct POS{
	double TimeUS;
	double Lat;
	double Lon;
	double Alt;
	double RHAlt;
	double ROAlt;
};

struct hmsTime{
	double Year;
	double Month;
	double Day;
	double Hour;
	double Minute;
	double Second;
};

//class
class cMPLOG_Appli
{
	public:
		cMPLOG_Appli(int argc,char ** argv);
		//~ double towTime2MJD(
						   //~ const towTime & Time, 
						   //~ const std::string & TimeSys
						   //~ );
		
		void ConvCAM2XML(
						 std::vector<CAM> aVCAM, 
						 std::string aXmlOut, 
						 std::string aStrChSys, 
						 Pt3dr aShift
						 );
		
		void ConvPOS2XML(
						 std::vector<POS> aVPos, 
						 std::string aXmlOut, 
						 std::string aStrChSys, 
						 Pt3dr aShift
						 );
						 
		void ConvCam2Traj(
						  std::vector<CAM> aVCAM, 
						  std::vector<Pt3dr> & aVTraj
						  );

		void ConvPos2Traj(
						  std::vector<POS> aVPos, 
						  std::vector<Pt3dr> & aVTraj
						  );
						  
		void ConvGps2Traj(
						  std::vector<GPS> aVCAM, 
						  std::vector<Pt3dr> & aVTraj
						  );
						  
		void ConvGPS2XML(
						 std::vector<GPS> aVGPS, 
						 std::string aXmlOut, 
						 std::string aStrChSys, 
						 Pt3dr aShift
						 );
						 
		void ConvXML2Ply(
						 std::string aXmlIn, 
						 bool aSBFP, 
						 Pt3di aCol
						 );
						 
		void MatchCAMWithPat(
						     std::vector<CAM> aVCAM, 
						     std::string aFullPat, 
						     std::string aOriOut, 
						     bool aShow, 
						     double aSeuil,
						     bool aGenIMTMFile,
						     bool aExportDeltaTime
						     );

		void CheckTmpMMDir(std::string aFullName);

		std::vector<cXmlDate> ReadExifData();

		std::vector<double> ExifDate2Sec(const std::vector<cXmlDate> aVExifDate);

		double hmsTime2Sec(const hmsTime & Time);

		double CalcSomme(const std::vector<double> aV);

		double CalcCorr(
						const std::vector<double> aV1, 
						const std::vector<double> aV2
						);

		std::vector<double> ComputeSuccDiff(
										    const std::vector<double> aV, 
										    bool aShow
										    );

		void ShowHmsTime(const hmsTime & Time);

		pair<int,double> GetIndicMax(std::vector<double> aVD);

		void WriteOriTxtFile(
							 std::vector<std::string> aVIm, 
							 std::vector<Pt3dr> aVPos, 
							 std::string aDir, 
							 std::string aOutFile
							 );

		std::vector<std::string> VImgsFromIndic(
												int aIndic, 
												std::vector<std::string> aVS, 
												unsigned int aSize
												);

		pair<std::vector<Pt3dr>,std::vector<double> > VTmPosFromCam(std::vector<CAM> aVCam);

		void GenrateImTmFile(
							 std::vector<double> aVTM,
							 std::vector<std::string> aVS,
							 std::string aOutputFile
							 );
							 
		void WriteDtimeInFile(
							  std::vector<double> aVD,
							  std::string aDir,
							  std::string aOutputFile
							  );
	private:
		std::string mDir;
		std::string mFile;
		std::string mStrChSys;
};

//write difference of time in a file
void cMPLOG_Appli::WriteDtimeInFile(
									std::vector<double> aVD,
									std::string aDir,
									std::string aOutputFile
									)
{
	 if (!MMVisualMode)
     {

		FILE * aFP = FopenNN(aOutputFile,"w","GetInfosMPLF_main");
        cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOutputFile);
        for(unsigned int aP=0; aP<aVD.size(); aP++)
        {
			fprintf(aFP,"%f\n",aVD[aP]);
        }

        ElFclose(aFP);
        std::cout<< aOutputFile <<" written."<<std::endl;
        }
}

//generate .xml file of Image TimeMark
void cMPLOG_Appli::GenrateImTmFile(
								   std::vector<double> aVTM,
								   std::vector<std::string> aVS,
								   std::string aOutputFile
								   )
{
	if(aVTM.size() != aVS.size())
	{
		ELISE_ASSERT(false, "Size of Tm and Im can't be different")
	}
	
	cDicoImgsTime aDicoIT;
	
	for(unsigned int aK=0; aK < aVS.size() ; aK++)
	{
		cCpleImgTime aCpleIT;
		aCpleIT.NameIm() = aVS[aK];
		aCpleIT.TimeIm() = aVTM[aK];
		
		aDicoIT.CpleImgTime().push_back(aCpleIT);
	}
	
	MakeFileXML(aDicoIT,aOutputFile);
}

//return vector of position from vectro of CAM struct
pair<std::vector<Pt3dr>,std::vector<double> > cMPLOG_Appli::VTmPosFromCam(std::vector<CAM> aVCam)
{
	std::vector<Pt3dr> aVPos;
	std::vector<double> aVTm;
	
	for(unsigned int aP=0; aP<aVCam.size(); aP++)
	{
		Pt3dr aPt;
		aPt.x = aVCam[aP].Lng;
		aPt.y = aVCam[aP].Lat;
		aPt.z = aVCam[aP].Alt;
		
		aVPos.push_back(aPt);
		aVTm.push_back(aVCam[aP].GPSTime);
	}
    // 1 => semble pose probleme sous W (c.f Luc Girod)
    // 2 =>  pose pb sous Linux 
    // 3 semble ok, et ne vois pas la plus value de make pair
	// return make_pair(aInd,aMax);	
	//return make_pair<unsigned int,double>(aInd,aMax);
	return std::pair<std::vector<Pt3dr>,std::vector<double> >(aVPos,aVTm);
}

//return vector strting from given indic
std::vector<std::string> cMPLOG_Appli::VImgsFromIndic(
													  int aIndic,
													  std::vector<std::string> aVS,
													  unsigned int aSize
													  )
{
	std::vector<std::string> aVOut;
	
	for(unsigned int aP=0; aP<aSize; aP++)
	{
		aVOut.push_back(aVS.at(aP+aIndic));
	}
	
	return aVOut;
}

//export a .txt file in OriConvert format
void cMPLOG_Appli::WriteOriTxtFile(
								   std::vector<std::string> aVIm, 
								   std::vector<Pt3dr> aVPos,
								   std::string aDir,
								   std::string aOutFile
								   )
{
	if(aVIm.size() != aVPos.size())
	{
		ELISE_ASSERT(false,"Note same size for images & positions vectors");
	}
	
	FILE * aFP = FopenNN(aOutFile,"w","GetInfosMPLF_main");
				
	cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOutFile);
				
	for (unsigned int aK=0 ; aK<aVIm.size() ; aK++)
	{
		fprintf(aFP,"%s %lf %lf %lf \n",aVIm.at(aK).c_str(), aVPos.at(aK).x,aVPos.at(aK).y,aVPos.at(aK).z);
	}
			
	ElFclose(aFP);
}

//get indice of max value form a vector
pair<int,double> cMPLOG_Appli::GetIndicMax(std::vector<double> aVD)
{
	int aInd=-1;
	
	double aMax = aVD[0];
	
	for(unsigned aK=0; aK<aVD.size(); aK++)
	{
		if(aVD[aK] > aMax)
		{
			aMax = aVD[aK];
			aInd=aK;
		}	
	}

    // 1 => semble pose probleme sous W (c.f Luc Girod)
    // 2 =>  pose pb sous Linux 
    // 3 semble ok, et ne vois pas la plus value de make pair
	// return make_pair(aInd,aMax);	
	//return make_pair<unsigned int,double>(aInd,aMax);
	return std::pair<int,double>(aInd,aMax);
}

//display time
void cMPLOG_Appli::ShowHmsTime(const hmsTime & Time)
{
	std::cout << "Time.Year   ==> " << Time.Year << std::endl;
	std::cout << "Time.Month  ==> " << Time.Month << std::endl;
	std::cout << "Time.Day    ==> " << Time.Day << std::endl;
	std::cout << "Time.Hour   ==> " << Time.Hour << std::endl;
	std::cout << "Time.Minute ==> " << Time.Minute << std::endl;
	std::cout << "Time.Second ==> " << Time.Second << std::endl;
}

//compute form n-dimentional vector (n-1)-dimentional successive differences
std::vector<double> cMPLOG_Appli::ComputeSuccDiff(
												  const std::vector<double> aV, 
												  bool aShow
												  )
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

//compute sum of a vector
double cMPLOG_Appli::CalcSomme(const std::vector<double> aV)
{
	double aSum=0;
	
    for (unsigned int aK=0 ; aK<aV.size() ; aK++)
    {
		aSum += aV[aK];
	}
	
	return aSum;
}

//compute corr coeff of 2 1-dimentional vectors
double cMPLOG_Appli::CalcCorr(
							  const std::vector<double> aV1, 
							  const std::vector<double> aV2
							  )
{
	double aCorrCoeff = 0;
	
	if(aV1.size() != aV2.size())
	{
		ELISE_ASSERT(false,"Not same size for both vectors !");
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

//convert HMS format to MJD (Exif data are given in HMS format)
double cMPLOG_Appli::hmsTime2Sec(const hmsTime & Time)
{
	double aSec;
	
	aSec = (Time.Hour)*3600 + (Time.Minute)*60 + (Time.Second);
	
	return aSec;
	
}

//check Tmp-MM-Dir Directory
void cMPLOG_Appli::CheckTmpMMDir(std::string aFullName)
{
	system_call((std::string("mm3d MMXmlXif \"")+aFullName+"\"").c_str());
}

//read exif data, convert to Mjd
std::vector<cXmlDate> cMPLOG_Appli::ReadExifData()
{
	std::vector<cXmlDate> aVExifDate;
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc("Tmp-MM-Dir/");
    std::list<std::string> Img_xif = aICNM->StdGetListOfFile(".*-MDT-.*xml");
    
    std::cout << "Img_xif.size() = " << Img_xif.size() << std::endl;

    for(std::list<std::string>::iterator I=Img_xif.begin(); I!=Img_xif.end(); I++)
    {
        cXmlXifInfo aXmlXifInfo=StdGetFromPCP("Tmp-MM-Dir/"+*I,XmlXifInfo);
        
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
std::vector<double> cMPLOG_Appli::ExifDate2Sec(const std::vector<cXmlDate> aVExifDate)
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
		
		double aSec =  hmsTime2Sec(aTime);
		aVMjdImgs.push_back(aSec);
	}
	
	return aVMjdImgs;
}

//~ double cMPLOG_Appli::towTime2Sec(const towTime & Time)
//~ {
	//~ double aSec = Time.Tow;
	
	//~ if(TimeSys == "UTC")
	//~ {
		//~ aSec -= LeapSecond;
	//~ }
	
	//~ double aS1970 = Time.GpsWeek * 7 * 86400 + aSec + GPS0;
	
	//~ double aMJD = (aS1970 - J2000) / 86400 + MJD2000;
	
	//~ return aMJD;
//~ }

void cMPLOG_Appli::ConvCam2Traj(
								std::vector<CAM> aVCAM, 
								std::vector<Pt3dr> & aVTraj
								)
{
	for(unsigned int aK=0; aK<aVCAM.size(); aK++)
	{
		Pt3dr aTraj;
		aTraj.x = aVCAM[aK].Lng;
		aTraj.y = aVCAM[aK].Lat;
		aTraj.z = aVCAM[aK].Alt;
		
		aVTraj.push_back(aTraj);
		
	}
}

void cMPLOG_Appli::ConvCAM2XML(
							   std::vector<CAM> aVCAM, 
							   std::string aXmlOut, 
							   std::string aStrChSys, 
							   Pt3dr aShift
							   )
{
	if(aVCAM.size() != 0)
	{
		//if changing coordinates system
		cChSysCo * aCSC = 0;
		 
		 if (mStrChSys!="")
			aCSC = cChSysCo::Alloc(aStrChSys,"");
			
		 std::vector<Pt3dr> aVTraj;
		 ConvCam2Traj(aVCAM,aVTraj);
		 
		 if (aCSC!=0)
		 {
			aVTraj = aCSC->Src2Cibl(aVTraj);
		 }
		
		cDicoGpsFlottant  aDico;
		Pt3dr aIncNULL(0,0,0);
		for(unsigned int aKP=0 ; aKP<aVCAM.size() ; aKP++)
		{
			cOneGpsDGF aOAD;
			Pt3dr aPt;
			aPt.x = aVTraj[aKP].x - aShift.x;
			aPt.y = aVTraj[aKP].y - aShift.y;
			aPt.z = aVTraj[aKP].z - aShift.z;
			aOAD.Pt() = aPt;
			aOAD.NamePt() = NumberToString(aKP);
			aOAD.Incertitude() = aIncNULL;
			aOAD.TagPt() = 0;
			aOAD.TimePt() = aVCAM[aKP].GPSTime;

			aDico.OneGpsDGF().push_back(aOAD);
		}

		MakeFileXML(aDico,aXmlOut);
	}
}

void cMPLOG_Appli::ConvPOS2XML(
							   std::vector<POS> aVPos, 
							   std::string aXmlOut, 
							   std::string aStrChSys, 
							   Pt3dr aShift
							   )
{
	if(aVPos.size() != 0)
	{
		//if changing coordinates system
		cChSysCo * aCSC = 0;
		 
		 if (mStrChSys!="")
			aCSC = cChSysCo::Alloc(aStrChSys,"");
			
		 std::vector<Pt3dr> aVTraj;
		 ConvPos2Traj(aVPos,aVTraj);
		 
		 if (aCSC!=0)
		 {
			aVTraj = aCSC->Src2Cibl(aVTraj);
		 }
		
		cDicoGpsFlottant  aDico;
		Pt3dr aIncNULL(0,0,0);
		for(unsigned int aKP=0 ; aKP<aVPos.size() ; aKP++)
		{
			cOneGpsDGF aOAD;
			Pt3dr aPt;
			aPt.x = aVTraj[aKP].x - aShift.x;
			aPt.y = aVTraj[aKP].y - aShift.y;
			aPt.z = aVTraj[aKP].z - aShift.z;
			aOAD.Pt() = aPt;
			aOAD.NamePt() = NumberToString(aKP);
			aOAD.Incertitude() = aIncNULL;
			aOAD.TagPt() = 0;
			aOAD.TimePt() = aVPos[aKP].TimeUS;

			aDico.OneGpsDGF().push_back(aOAD);
		}

		MakeFileXML(aDico,aXmlOut);
	}
}

void cMPLOG_Appli::ConvGPS2XML(
							   std::vector<GPS> aVGPS, 
							   std::string aXmlOut, 
							   std::string aStrChSys, 
							   Pt3dr aShift
							   )
{
	if(aVGPS.size() != 0)
	{
		//if changing coordinates system
		cChSysCo * aCSC = 0;
		 
		 if (mStrChSys!="")
			aCSC = cChSysCo::Alloc(aStrChSys,"");
			
		 std::vector<Pt3dr> aVTraj;
		 ConvGps2Traj(aVGPS,aVTraj);
		 
		 if (aCSC!=0)
		 {
			aVTraj = aCSC->Src2Cibl(aVTraj);
		 }
		
		cDicoGpsFlottant  aDico;
		Pt3dr aIncNULL(0,0,0);
		for(unsigned int aKP=0 ; aKP<aVGPS.size() ; aKP++)
		{
			cOneGpsDGF aOAD;
			Pt3dr aPt;
			aPt.x = aVTraj[aKP].x - aShift.x;
			aPt.y = aVTraj[aKP].y - aShift.y;
			aPt.z = aVTraj[aKP].z - aShift.z;
			aOAD.Pt() = aPt;
			aOAD.NamePt() = NumberToString(aKP);
			aOAD.Incertitude() = aIncNULL;
			aOAD.TagPt() = 0;
			aOAD.TimePt() = aVGPS[aKP].GMS;

			aDico.OneGpsDGF().push_back(aOAD);
		}

		MakeFileXML(aDico,aXmlOut);
	}
}

void cMPLOG_Appli::ConvXML2Ply(
							   std::string aXmlIn, 
							   bool aSBFP, 
							   Pt3di aCol
							   )
{
	if(ELISE_fp::exist_file(aXmlIn))
	{
		std::string aCom = MMDir()
						   + std::string("bin/mm3d")
						   + std::string(" ")
						   + std::string("TestLib")
						   + std::string(" ")
						   + std::string("Export2Ply")
						   + std::string(" ")
						   + std::string("AppXML")
						   + std::string(" ")
						   + aXmlIn
						   + std::string(" ")
						   + std::string("Ray=0.1")
						   + std::string(" ")
						   + std::string("GpsXML=1")
						   + std::string(" ")
						   + std::string("FixColor=[") + NumberToString(aCol.x)
						   + std::string(",") + NumberToString(aCol.y)
						   + std::string(",") + NumberToString(aCol.z)
						   + std::string("]");
						   
		if(aSBFP)
			aCom = aCom + std::string(" ") + std::string("ShiftBFP=1");
			
		system_call(aCom.c_str());
	}
}

void cMPLOG_Appli::ConvGps2Traj(
								std::vector<GPS> aVGPS, 
								std::vector<Pt3dr> & aVTraj
								)
{
	for(unsigned int aK=0; aK<aVGPS.size(); aK++)
	{
		Pt3dr aTraj;
		aTraj.x = aVGPS[aK].Lng;
		aTraj.y = aVGPS[aK].Lat;
		aTraj.z = aVGPS[aK].Alt;
		
		aVTraj.push_back(aTraj);
	}
}

void cMPLOG_Appli::ConvPos2Traj(
								std::vector<POS> aVPos, 
								std::vector<Pt3dr> & aVTraj
								)
{
	for(unsigned int aK=0; aK<aVPos.size(); aK++)
	{
		Pt3dr aTraj;
		aTraj.x = aVPos[aK].Lon;
		aTraj.y = aVPos[aK].Lat;
		aTraj.z = aVPos[aK].Alt;
		
		aVTraj.push_back(aTraj);
		
	}
}

void cMPLOG_Appli::MatchCAMWithPat(
								   std::vector<CAM> aVCAM, 
								   std::string aFullPat, 
								   std::string aOriOut, 
								   bool aShow, 
								   double aSeuil,
								   bool aGenIMTMFile,
								   bool aExportDeltaTime
								   )
{
	 std::string aDirectory="";
	 std::string aPat="";
	 
	 SplitDirAndFile(aDirectory, aPat, aFullPat);
     std::cout << "Working dir: " << aDirectory << std::endl;
     std::cout << "Images pattern: " << aPat << std::endl;
     
     cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirectory);
     const std::vector<std::string> aSetIm = *(aICNM->Get(aPat));
	 
	 if(aShow)
	 {
		for (unsigned int aK=0 ; aK < aSetIm.size() ; aK++)
		{
			std::cout << " - " << aSetIm[aK] << std::endl;
		}
	 }
	 
	//check Tmp-MM-Dir Directory
    CheckTmpMMDir(aFullPat);
    
    //read exif data, convert to Mjd and compare differences of Time
    std::vector<cXmlDate> aVExifDate = ReadExifData();
    
    //convert exif data Time 2 Mjd values
    std::vector<double> aVMjdImgs = ExifDate2Sec(aVExifDate);
    
    //if Exif data is available
    if(aVMjdImgs.size() != 0 && aVCAM.size() != 0)
    {	
		
		//compute CAM tag time differences in sec : compute once
		std::vector<double> aVTimeCam;
		std::vector<double> aVSuccDiffCAMG; //global
		
		for(unsigned int aP=0; aP<aVCAM.size()-1; aP++)
		{
			double aDiff = aVCAM[aP+1].GPSTime - aVCAM[aP].GPSTime;
			aVSuccDiffCAMG.push_back(aDiff);
			aVTimeCam.push_back(aVCAM[aP].GPSTime);
		}
		aVTimeCam.push_back(aVCAM[aVCAM.size()].GPSTime);
		
		//compute IMG exif time differences in sec : compute once
		std::vector<double> aVSuccDiffImgsG; //global
		for(unsigned int aK=0; aK<aVMjdImgs.size()-1; aK++)
		{
			double aDiff = aVMjdImgs[aK+1] - aVMjdImgs[aK];
			aVSuccDiffImgsG.push_back(aDiff);
		}
		
		//write diff tps in files
		if(aExportDeltaTime)
		{
			WriteDtimeInFile(aVSuccDiffCAMG,aDirectory,"Delta_Time_CAM.txt");
			WriteDtimeInFile(aVSuccDiffImgsG,aDirectory,"Delta_Time_Exif.txt");
		}
		
		//if Nbr CAM == Nbr IMG --> check if correlation of delta time is fine
		if(aVCAM.size() == aSetIm.size())
		{
			std::cout << "Case : Nbr CAM == Nbr IMG " << std::endl;
			
			//check only if correlation >= to seuil
			double aCor = CalcCorr(aVSuccDiffCAMG,aVSuccDiffImgsG);
			std::cout << " aCor = " << aCor << std::endl;
			
			if(aCor > aSeuil)
			{
				std::vector<std::string> aVIFI = VImgsFromIndic(0,aSetIm,aVCAM.size());
				std::vector<Pt3dr> aVPosCam;
				std::vector<double> aVTm;
				unpair(aVPosCam,aVTm) = VTmPosFromCam(aVCAM);
				
				//write file with : IMG X Y Z
				//X Y Z come from CAM TAG
				WriteOriTxtFile(aVIFI,aVPosCam,mDir,aOriOut);
				
				if(aGenIMTMFile)
				{
					std::string aOutputFile = "CAM_IM_TM.xml";
					GenrateImTmFile(aVTm,aVIFI,aOutputFile);
				}
			}
		}
	
		//if Nbr CAM < Nbr IMG --> need to extract a sub sequence of IMG
		if(aVCAM.size() < aSetIm.size())
		{
			std::cout << "Case : Nbr CAM < Nbr IMG " << std::endl;
			
			unsigned int aDec=0;
			//vector containning values of corr coeff
			std::vector<double> aVCorrCoeff;
			
			while(aDec < aSetIm.size()-aVCAM.size()+1)
			{
				std::vector<double> aSubVSetIm; 						//sub vector of images
				for(unsigned int aP=aDec; aP<aVCAM.size()+aDec; aP++)
				{
					aSubVSetIm.push_back(aVMjdImgs.at(aP));
				}
				
				//compute sub images time differences in sec (exif)
				std::vector<double> aVSuccDiffImgs = ComputeSuccDiff(aSubVSetIm,aShow);
				double aCor = CalcCorr(aVSuccDiffCAMG,aVSuccDiffImgs);
				aVCorrCoeff.push_back(aCor);
				
				aDec++;
			}
			
			if(aShow)
			{
				for(unsigned int aP=0; aP<aVCorrCoeff.size(); aP++)
				{
					std::cout << "aVCorrCoeff.at(aP) = " << aVCorrCoeff.at(aP) << std::endl;
				}
			}
			
			//get indice of maximum Corr Coeff
			int aIndic;
			double aMaxVal; 
			unpair(aIndic,aMaxVal) = GetIndicMax(aVCorrCoeff);
			std::cout << "Indic Max Corr = " << aIndic << " & Value = " << aMaxVal << std::endl;
			
			if(aMaxVal > aSeuil)
			{
				std::vector<std::string> aVIFI = VImgsFromIndic(aIndic,aSetIm,aVCAM.size());
				std::vector<Pt3dr> aVPosCam;
				std::vector<double> aVTm;
				unpair(aVPosCam,aVTm) = VTmPosFromCam(aVCAM);
				
				//write file with : IMG X Y Z
				//X Y Z come from CAM TAG
				WriteOriTxtFile(aVIFI,aVPosCam,mDir,aOriOut);
				
				if(aGenIMTMFile)
				{
					std::string aOutputFile = "CAM_IM_TM.xml";
					GenrateImTmFile(aVTm,aVIFI,aOutputFile);
				}
			}
			
		}
	
		//if Nbr CAM > Nbr IMG --> need to extract a sub sequence of CAM that fits best available images 
    	if(aVCAM.size() > aSetIm.size())
		{
			std::cout << "Case : Nbr CAM > Nbr IMG " << std::endl;
			
			unsigned int aDec=0;
			//vector containning values of corr coeff
			std::vector<double> aVCorrCoeff;
			
			while(aDec < aVCAM.size()-aSetIm.size()+1)
			{
				std::cout<< "Dec = " << aDec << " DecMax = " << aVCAM.size()-aSetIm.size() << std::endl;
				
				std::vector<double> aSubVCAM; 							//sub vector of CAM
				std::vector<double> aVSuccDiffCAM;						//vector of succ diffrerence of sub vector CAM
				for(unsigned int aP=aDec; aP<aSetIm.size()+aDec; aP++)
				{
					aSubVCAM.push_back(aVTimeCam.at(aP));
				}
				
				//compute sub CAM time differences in sec (exif)
				aVSuccDiffCAM = ComputeSuccDiff(aSubVCAM,aShow);
				double aCor = CalcCorr(aVSuccDiffImgsG,aVSuccDiffCAM);
				aVCorrCoeff.push_back(aCor);
				
				aDec++;
			}
			
			if(aShow)
			{
				for(unsigned int aP=0; aP<aVCorrCoeff.size(); aP++)
				{
					std::cout << "aVCorrCoeff.at(aP) = " << aVCorrCoeff.at(aP) << std::endl;
				}
			}
			
			//get indice of maximum Corr Coeff
			int aIndic;
			double aMaxVal; 
			unpair(aIndic,aMaxVal) = GetIndicMax(aVCorrCoeff);
			std::cout << "Indic Max Corr = " << aIndic << " & Value = " << aMaxVal << std::endl;
			
			if(aMaxVal > aSeuil)
			{
				
			}
		}
	}
}

cMPLOG_Appli::cMPLOG_Appli(int argc,char ** argv)
{
	bool aShow=false;
	bool aSBFP=false;
	Pt3di aCCAM(255,0,0);
	Pt3di aCGPS(0,0,255);
	Pt3di aCGPS2(0,255,0);
	Pt3di aCPos(0,0,0);
	Pt3dr aShift(0,0,0);
	std::string aFullPat="";
	double aSeuil=0.9;
	std::string aOriOut="";
	bool aGenImTmFile=false;
	bool aExportDeltaTime=true;
	
	ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
	LArgMain()  << EAMC(mDir,"Directory")
				<< EAMC(mFile,"Log File of Mission Planner"),
	LArgMain()  << EAM(aShow, "Show", false, "Display Pattern to use in cmd line ; Def=false",eSAM_IsBool)
				<< EAM(mStrChSys,"ChSys",true,"Change coordinate file")
				<< EAM(aSBFP,"SBFP",true,"substract first point coordinate to each file")
				<< EAM(aCCAM,"ColCAM",true,"Color for CAM Tag ; Def=red")
				<< EAM(aCGPS,"ColGPS",true,"Color for GPS Tag ; Def=blue")
				<< EAM(aCGPS2,"ColGPS2",true,"Color for GPS2 Tag ; Def=green")
				<< EAM(aCPos,"ColPos",true,"Color for POS Tag ; Def=black")
				<< EAM(aShift,"Shift",true,"Add this shift to all coordinates ; Def=(0,0,0)")
				<< EAM(aFullPat,"Pat",true,"Full pattern")
				<< EAM(aSeuil,"Seuil",true,"Correlation coefficient Treshold ; Def=0.9")
				<< EAM(aGenImTmFile,"GenInter",true,"Generate Image Time .xml file ; Def=false")
				<< EAM(aOriOut,"OutOriFile",true,"Output File name for OriConvert ; Def=Ori_N_X_Y_Z.txt")
				<< EAM(aExportDeltaTime,"Exportdt",true,"Export delta time in files ; Def=true")
	);
	
	if(aOriOut == "")
	{
		aOriOut = "Ori_N_X_Y_Z.txt";
	}
	
	std::vector<CAM> aVCAM;
	std::vector<GPS> aVGPS;
	std::vector<GPS> aVGPS2;
	std::vector<POS> aVPos;
	
	//read input file
    ifstream aFichier((mDir + mFile).c_str());

    if(aFichier)
    {
		std::string aLine;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLine);
			
			if(!aLine.empty())
			{
				//CAM, 1260919656, 220518000, 1954, 48.8444203, 1.4597241, 158.37, 68.20, 158.16, 5.10, -2.80, 243.96
				if(aLine.compare(0,4,"CAM,") == 0)						
				{
					char *aBuffer = strdup((char*)aLine.c_str());
					std::string aType = strtok(aBuffer," ");
					std::string aTimeUS = strtok( NULL, " " );
					std::string aGpsTime = strtok( NULL, " " );
					std::string aGpsWeek = strtok( NULL, " " );
					std::string aLat = strtok( NULL, " " );
					std::string aLon = strtok( NULL, " " );
					std::string aAlt = strtok( NULL, " " );
					std::string aRelAlt = strtok( NULL, " " );
					std::string aGpsAlt = strtok( NULL, " " );
					std::string aRoll = strtok( NULL, " " );
					std::string aPitch = strtok( NULL, " " );
					std::string aYaw = strtok( NULL, " " );
					
					CAM aCamInfo;
					aCamInfo.TimeUS = atof(aTimeUS.substr(0,aTimeUS.size()-1).c_str());
					
					//~ towTime aTT;
					//~ aTT.GpsWeek = atoi(aGpsWeek.substr(0,aGpsWeek.size()-1).c_str());
					//~ aTT.Tow = atof(aGpsTime.substr(0,aGpsTime.size()-1).c_str())/1000;
					//~ double aMJD = towTime2MJD(aTT,"GPST");
					
					//~ aCamInfo.MJD = aMJD;
					
					aCamInfo.GPSTime = atof(aGpsTime.substr(0,aGpsTime.size()-1).c_str())/1000; //in sec
					aCamInfo.GPSWeek = atoi(aGpsWeek.substr(0,aGpsWeek.size()-1).c_str());
					aCamInfo.Lat = atof(aLat.substr(0,aLat.size()-1).c_str());
					aCamInfo.Lng = atof(aLon.substr(0,aLon.size()-1).c_str());
					aCamInfo.Alt = atof(aAlt.substr(0,aAlt.size()-1).c_str());
					aCamInfo.RelAlt = atof(aRelAlt.substr(0,aRelAlt.size()-1).c_str());
					aCamInfo.GPSAlt = atof(aGpsAlt.substr(0,aGpsAlt.size()-1).c_str());
					aCamInfo.Roll = atof(aRoll.substr(0,aRoll.size()-1).c_str());
					aCamInfo.Pitch = atof(aPitch.substr(0,aPitch.size()-1).c_str());
					aCamInfo.Yaw = atof(aYaw.substr(0,aYaw.size()-1).c_str());
					
					aVCAM.push_back(aCamInfo);
					
				
				}
				
				//GPS, 1180441285, 3, 220437600, 1954, 12, 0.91, 48.843854, 1.459903, 94.56, 0.027, 0, 0.004, 0
				if(aLine.compare(0,4,"GPS,") == 0)						
				{
					char *aBuffer = strdup((char*)aLine.c_str());
					std::string aType = strtok(aBuffer," ");
					std::string aTimeUS = strtok(NULL," ");
					std::string aStatus = strtok(NULL," ");
					std::string aGpsMilliSec = strtok(NULL," ");
					std::string aGpsWeek = strtok(NULL," ");
					std::string aNbrSat = strtok(NULL," ");
					std::string aHdop = strtok(NULL," ");
					std::string aLat = strtok( NULL, " " );
					std::string aLon = strtok( NULL, " " );
					std::string aAlt = strtok( NULL, " " );
					std::string aSpeed = strtok( NULL, " " );
					std::string aGcrs = strtok( NULL, " " );
					std::string aVZ = strtok( NULL, " " );
					std::string aU = strtok( NULL, " " );
					
					GPS aGPSInfo;
					aGPSInfo.TimeUS = atof(aTimeUS.substr(0,aTimeUS.size()-1).c_str());
					aGPSInfo.Status = atoi(aStatus.substr(0,aStatus.size()-1).c_str());
					aGPSInfo.GMS = atof(aGpsMilliSec.substr(0,aGpsMilliSec.size()-1).c_str())/1000; // in sec
					aGPSInfo.GWk = atoi(aGpsWeek.substr(0,aGpsWeek.size()-1).c_str());
					aGPSInfo.NSats = atoi(aNbrSat.substr(0,aNbrSat.size()-1).c_str());
					aGPSInfo.HDop = atof(aHdop.substr(0,aHdop.size()-1).c_str());
					aGPSInfo.Lat = atof(aLat.substr(0,aLat.size()-1).c_str());
					aGPSInfo.Lng = atof(aLon.substr(0,aLon.size()-1).c_str());
					aGPSInfo.Alt = atof(aAlt.substr(0,aAlt.size()-1).c_str());
					aGPSInfo.Spd = atof(aSpeed.substr(0,aSpeed.size()-1).c_str());
					aGPSInfo.GCrs = atoi(aGcrs.substr(0,aGcrs.size()-1).c_str());
					aGPSInfo.VZ = atof(aVZ.substr(0,aVZ.size()-1).c_str());
					aGPSInfo.U = atof(aU.substr(0,aU.size()-1).c_str());
					
					aVGPS.push_back(aGPSInfo);
					
				}
				
				//GPS2,1260807688, 6, 220518000, 1954, 12, 0.94, 48.8444246,1.4597341, 158.16, 3.331549, 243.3775, -0.6531352, 1
				if(aLine.compare(0,5,"GPS2,") == 0)						
				{
					char *aBuffer = strdup((char*)aLine.c_str());
					std::string aType = strtok(aBuffer," ");
					std::string aTimeUS = strtok(NULL," ");
					std::string aStatus = strtok(NULL," ");
					std::string aGpsMilliSec = strtok(NULL," ");
					std::string aGpsWeek = strtok(NULL," ");
					std::string aNbrSat = strtok(NULL," ");
					std::string aHdop = strtok(NULL," ");
					std::string aLat = strtok( NULL, " " );
					std::string aLon = strtok( NULL, " " );
					std::string aAlt = strtok( NULL, " " );
					std::string aSpeed = strtok( NULL, " " );
					std::string aGcrs = strtok( NULL, " " );
					std::string aVZ = strtok( NULL, " " );
					std::string aU = strtok( NULL, " " );
					
					GPS aGPS2Info;
					aGPS2Info.TimeUS = atof(aTimeUS.substr(0,aTimeUS.size()-1).c_str());
					aGPS2Info.Status = atoi(aStatus.substr(0,aStatus.size()-1).c_str());
					aGPS2Info.GMS = atof(aGpsMilliSec.substr(0,aGpsMilliSec.size()-1).c_str())/1000; //in sec
					aGPS2Info.GWk = atoi(aGpsWeek.substr(0,aGpsWeek.size()-1).c_str());
					aGPS2Info.NSats = atoi(aNbrSat.substr(0,aNbrSat.size()-1).c_str());
					aGPS2Info.HDop = atof(aHdop.substr(0,aHdop.size()-1).c_str());
					aGPS2Info.Lat = atof(aLat.substr(0,aLat.size()-1).c_str());
					aGPS2Info.Lng = atof(aLon.substr(0,aLon.size()-1).c_str());
					aGPS2Info.Alt = atof(aAlt.substr(0,aAlt.size()-1).c_str());
					aGPS2Info.Spd = atof(aSpeed.substr(0,aSpeed.size()-1).c_str());
					aGPS2Info.GCrs = atoi(aGcrs.substr(0,aGcrs.size()-1).c_str());
					aGPS2Info.VZ = atof(aVZ.substr(0,aVZ.size()-1).c_str());
					aGPS2Info.U = atof(aU.substr(0,aU.size()-1).c_str());
					
					aVGPS2.push_back(aGPS2Info);
				}
				
				//POS, 1180414380, 48.8438484, 1.4598758, 90.17, 0.006674996, 0.656675
				if(aLine.compare(0,4,"POS,") == 0)
				{
					char *aBuffer = strdup((char*)aLine.c_str());
					std::string aType = strtok(aBuffer," ");
					std::string aTimeUS = strtok(NULL," ");
					std::string aLat = strtok(NULL," ");
					std::string aLon = strtok(NULL," ");
					std::string aAlt = strtok(NULL," ");
					std::string aRHAlt = strtok(NULL," ");
					std::string aROAlt = strtok(NULL," ");
					
					POS aPosInfo;
					aPosInfo.TimeUS = atof(aTimeUS.substr(0,aTimeUS.size()-1).c_str())/1000; // in sec ?
					aPosInfo.Lat = atof(aLat.substr(0,aLat.size()-1).c_str());
					aPosInfo.Lon = atof(aLon.substr(0,aLon.size()-1).c_str());
					aPosInfo.Alt = atof(aAlt.substr(0,aAlt.size()-1).c_str());
					aPosInfo.RHAlt = atof(aRHAlt.substr(0,aRHAlt.size()-1).c_str());
					aPosInfo.ROAlt = atof(aROAlt.substr(0,aROAlt.size()-1).c_str());
					
					aVPos.push_back(aPosInfo);
					
				}
				
			}
		}
		
	aFichier.close();
	
	}
	
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
	
	//conversion to .xml
	ConvCAM2XML(aVCAM,"Traj_CAM.xml",mStrChSys,aShift);
	ConvGPS2XML(aVGPS,"Traj_GPS.xml",mStrChSys,aShift);
	ConvGPS2XML(aVGPS2,"Traj_GPS2.xml",mStrChSys,aShift);
	ConvPOS2XML(aVPos,"Traj_POS.xml",mStrChSys,aShift);
	
	//conversion to .ply
	ConvXML2Ply("Traj_CAM.xml",aSBFP,aCCAM);
	ConvXML2Ply("Traj_GPS.xml",aSBFP,aCGPS);
	ConvXML2Ply("Traj_GPS2.xml",aSBFP,aCGPS2);
	ConvXML2Ply("Traj_POS.xml",aSBFP,aCPos);
	
	//if generate image Time .xml file
	if(aGenImTmFile && !EAMIsInit(&aFullPat))
	{
		std::cout << "gggggggggggggggggggggggggg" << std::endl;
		std::string aOutputFile = "CAM_IM_TM.xml";
		std::vector<double> aVTM;
		std::vector<std::string> aVS;
		for(unsigned int aP=0; aP<aVCAM.size(); aP++)
		{
			aVTM.push_back(aVCAM[aP].GPSTime);
			aVS.push_back("CAM" + NumberToString(aP+1));
		}
		
		GenrateImTmFile(aVTM,aVS,aOutputFile);
	}
	
	//if a pattern of image is given
	if(EAMIsInit(&aFullPat))
	{
		MatchCAMWithPat(aVCAM, aFullPat, aOriOut, aShow, aSeuil, aGenImTmFile, aExportDeltaTime);
	}
    
    //if we want to print some details
	if(aShow)
	{
	
		for(unsigned int aK=0; aK<aVCAM.size()-1; aK++)
		{
			std::cout << " Difference Time  : CAM" << aK << "-CAM" << aK+1 << " = " << aVCAM[aK+1].GPSTime - aVCAM[aK].GPSTime << " sec" << std::endl;
		}
		
		for(unsigned int aK=0; aK<aVGPS.size(); aK++)
		{
			std::cout << "GPS Status = " << aVGPS[aK].Status << std::endl;
			std::cout << " Difference Time  : GPS" << aK << "-GPS" << aK+1 << " = " << aVGPS[aK+1].GMS - aVGPS[aK].GMS << " sec" << std::endl;
		}
		
		for(unsigned int aK=0; aK<aVGPS2.size(); aK++)
		{
			std::cout << "GPS2 Status = " << aVGPS2[aK].Status << std::endl;
			std::cout << " Difference Time  : GPSP" << aK << "-GPSP" << aK+1 << " = " << aVGPS2[aK+1].GMS - aVGPS2[aK].GMS << " sec" << std::endl;
		}
		
		for(unsigned int aK=0; aK<aVPos.size()-1; aK++)
		{
			std::cout << " Difference Time  : POS" << aK << "-POS" << aK+1 << " = " << aVPos[aK+1].TimeUS - aVPos[aK].TimeUS << " sec" << std::endl;
		}
		
		std::cout << "Number of CAM Infos = " << aVCAM.size() << std::endl; 
		std::cout << "Number of GPS Infos = " << aVGPS.size() << std::endl; 
		std::cout << "Number of GPS2 Infos = " << aVGPS2.size() << std::endl; 
		std::cout << "Number of POS Infos = " << aVPos.size() << std::endl;
		
	}
}

int GetInfosMPLF_main(int argc,char ** argv)
{
	cMPLOG_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}


/* Footer-MicMac-eLiSe-25/06/2007

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
Footer-MicMac-eLiSe-25/06/2007/*/
