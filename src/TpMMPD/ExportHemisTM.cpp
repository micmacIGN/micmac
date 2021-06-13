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

//-------------Structure of a binary message from Hemisphere---------------------------------------------//
typedef struct
{
    char bin[4];
    unsigned short blockid;
    unsigned short datalength;
    char *data;
    unsigned short checksum;
    char cr;
    char lf;
}
msgbin;

//-------------Structure of Bin01 message from Hemisphere---------------------------------------------//
//-------------Bin01 contains TimeMark Informations---------------------------------------------------//
/* length = 8 + 52 + 2 + 2 = 64 */
typedef struct
{
    char bin[4];														/* start of header ( = $BIN) */
    unsigned short blockid;												/* Id of message (1,76,80,94,95) */
    unsigned short datalength;											/* Length of message */
    unsigned char AgeOfDiff;    										/* age of differential, seconds (255 max)*/
    unsigned char NumOfSats;    										/* number of satellites used (12 max)    */
    unsigned short GPSWeek;    											/* GPS week */
    double GPSTimeOfWeek;   											/* GPS tow */
    double    Latitude;    												/* Latitude degrees, -90..90 */
    double    Longitude;    											/* Latitude degrees, -180..180 */
    float    Height;    												/* (m), Altitude ellipsoid */
    float    VNorth;    												/* Velocity north    m/s */
    float    VEast;   													/* Velocity east    m/s */
    float    VUp;    													/* Velocity up m/s */
    float    StdDevResid;    											/* (m), Standard Deviation of Residuals */
    unsigned short NavMode;												/* Navigation mode : 0,1,2,3,4,5,6 */
    unsigned short extAgeOfDiff;    									/* age of diff using 16 bits */
    unsigned short CheckSum;    										/* sum of all bytes of the header and data */
    unsigned short CRLF;    											/* Carriage Return Line Feed */
}
Msg1;

//-------------Structure of a TimeMark Output---------------------------------------------//
typedef struct
{
	int Nbr;
	double MJD;
	int NbrSats;
	double Lat;
	double Lon;
	double Height;
}
TimeMark;

//struct
//struct towTime{
//	double GpsWeek;
//	double Tow; //or wsec
//};

//class
class cEHTM_Appli
{
		public :
			cEHTM_Appli(int argc,char ** argv);
			std::vector<TimeMark> GetTMFromVBin01(std::vector<Msg1> & aVMsgBin01);
			void WriteTMInFile(std::vector<TimeMark> & aVTM, std::string & aOutputNameFile, bool aFormat);
//			double towTime2MJD(const towTime & Time, const std::string & TimeSys);
		
		private :
			std::string mDir;
			std::string mFile;
};

//double cEHTM_Appli::towTime2MJD(const towTime & Time, const std::string & TimeSys)
//{
//	double aSec = Time.Tow;
	
//	if(TimeSys == "UTC")
//	{
//		aSec -= LeapSecond;
//	}
	
//	double aS1970 = Time.GpsWeek * 7 * 86400 + aSec + GPS0;
	
//	double aMJD = (aS1970 - J2000) / 86400 + MJD2000;
	
//	return aMJD;
//}

//get TimeMark vector form a vector of Bin01 Hemisphere 
std::vector<TimeMark> cEHTM_Appli::GetTMFromVBin01(std::vector<Msg1> & aVMsgBin01)
{
	std::vector<TimeMark> aVTM;
	
	int aCmpt=1;
	
	for(unsigned int aK=0; aK < aVMsgBin01.size(); aK++)
	{
		//0=No Fix ; 1=Fix 2D; 2=Fix 3D; 3=Fix 2D & Diff; 4=Fix 3D & Diff; 5=RTK Search; 6=Fix 3D & Diff & RTK solution; Else = Manual TimeMark
		if(aVMsgBin01.at(aK).NavMode != 0 && aVMsgBin01.at(aK).NavMode != 1 && aVMsgBin01.at(aK).NavMode != 2 && aVMsgBin01.at(aK).NavMode != 3 && aVMsgBin01.at(aK).NavMode != 4 && aVMsgBin01.at(aK).NavMode != 5 && aVMsgBin01.at(aK).NavMode != 6)
		{
			double GpsSeconds = aVMsgBin01.at(aK).GPSTimeOfWeek;
			double GpsWeek = aVMsgBin01.at(aK).GPSWeek;
			towTime aTowTime;
			aTowTime.GpsWeek = GpsWeek;
			aTowTime.Tow = GpsSeconds;
			double Mjd = towTime2MJD(aTowTime, "GPST");
			TimeMark aTM;
			aTM.Nbr = aCmpt;
			aTM.MJD = Mjd;
			aTM.NbrSats = aVMsgBin01.at(aK).NumOfSats;
			aTM.Lat = aVMsgBin01.at(aK).Latitude;
			aTM.Lon = aVMsgBin01.at(aK).Longitude;
			aTM.Height = aVMsgBin01.at(aK).Height;
			aVTM.push_back(aTM);
			aCmpt++;
		}
	}
	
	return aVTM;
}

//write into a file TimeMark vector in ASCII format
void cEHTM_Appli::WriteTMInFile(
								std::vector<TimeMark> & aVTM,
								std::string & aOutputNameFile,
								bool aFormat
								)
{
	 FILE* aCible = NULL;
	 aCible=fopen(aOutputNameFile.c_str(),"w");
	 
	 if(aFormat)
	 {
		 std::string aFormat = "# NbrTimeMark MJD NbrSats Lat Lon H";
		 fprintf(aCible,"%s \n",aFormat.c_str());
	 }
	 
	 for(unsigned int aK=0; aK < aVTM.size() ; aK++)
	 {
		 fprintf(aCible,"%d %.9f %d %.6f %.6f %.6f\n",aVTM.at(aK).Nbr, aVTM.at(aK).MJD, aVTM.at(aK).NbrSats, aVTM.at(aK).Lat, aVTM.at(aK).Lon, aVTM.at(aK).Height);
	 }
	 
	 fclose(aCible);
	 
}

bool fread_one(void *aDst, size_t aSize, FILE *aStream)
{
	return fread(aDst, aSize, 1, aStream) == 1;
}

cEHTM_Appli::cEHTM_Appli(int argc,char ** argv)
{
	std::string aOut;
	bool aAddFormat=false;
	
	ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(mDir,"Directory")
					 << EAMC(mFile, "Hemisphere Binary (Containing Bin01) File",  eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Name output file (def=NameFile_TimeMark.txt)")
					 << EAM(aAddFormat,"Format",false,"Add File Format at the begining fo the File")
     );
     
     if(aOut=="")
     {
		aOut=StdPrefixGen(mFile) + "_TimeMark.txt";
     }
      
	 msgbin msg;					//structure of a Hemisphere Binary Faile
     Msg1 msg1;						//structure of Bin01 Hemisphere containing TimeMark
     int countbin1=0;
     FILE* aSource = NULL;
     
	 //open binary file
	 aSource = fopen (mFile.c_str(),"rb");
     if(aSource == NULL) 
     {
		printf("Error Reading File \n");
	 }
     else 
     {
		printf("File Successfuly Opened \n");
	 }
	 
	 std::vector<Msg1> aVMsg1;
	 
	 //read binary file
	 while(!feof(aSource))
     {

		 //-------------Reading of a Binary Hemisphere Message---------------------------------------------//
		 fread_one(&msg.bin,sizeof(msg.bin),aSource);               			//IN
		 fread_one(&msg.blockid,sizeof(msg.blockid),aSource);      			//n°
		 fread_one(&msg.datalength,sizeof(msg.datalength),aSource);			//size
		 char *data = new char[msg.datalength];

		 //if the message is a Bin01 containing TimeMark
		 if (msg.blockid==1)
		 {
			countbin1++;
			fread_one(&msg1.AgeOfDiff,sizeof(msg1.AgeOfDiff),aSource);
			fread_one(&msg1.NumOfSats,sizeof(msg1.NumOfSats),aSource);
			fread_one(&msg1.GPSWeek,sizeof(msg1.GPSWeek),aSource);
			fread_one(&msg1.GPSTimeOfWeek,sizeof(msg1.GPSTimeOfWeek),aSource);
			fread_one(&msg1.Latitude,sizeof(msg1.Latitude),aSource);
			fread_one(&msg1.Longitude,sizeof(msg1.Longitude),aSource);
			fread_one(&msg1.Height,sizeof(msg1.Height),aSource);
			fread_one(&msg1.VNorth,sizeof(msg1.VNorth),aSource);
			fread_one(&msg1.VEast,sizeof(msg1.VEast),aSource);
			fread_one(&msg1.VUp,sizeof(msg1.VUp),aSource);
			fread_one(&msg1.StdDevResid,sizeof(msg1.StdDevResid),aSource);
			fread_one(&msg1.NavMode,sizeof(msg1.NavMode),aSource);                
			fread_one(&msg1.extAgeOfDiff,sizeof(msg1.extAgeOfDiff),aSource);
			fread_one(&msg1.CheckSum,sizeof(msg1.CheckSum),aSource);
			fread_one(&msg1.CRLF,sizeof(msg1.CRLF),aSource);
			
			Msg1 aMsg1;
			
			strcpy(aMsg1.bin, msg1.bin);
			aMsg1.blockid =  msg1.blockid;
			aMsg1.datalength = msg1.datalength;
			aMsg1.AgeOfDiff = msg1.AgeOfDiff;
			aMsg1.NumOfSats = msg1.NumOfSats;
			aMsg1.GPSWeek = msg1.GPSWeek;
			aMsg1.GPSTimeOfWeek = msg1.GPSTimeOfWeek;
			aMsg1.Latitude = msg1.Latitude;
			aMsg1.Longitude = msg1.Longitude;
			aMsg1.Height = msg1.Height;
			aMsg1.VNorth = msg1.VNorth;
			aMsg1.VEast = msg1.VEast;
			aMsg1.VUp = msg1.VUp;
			aMsg1.StdDevResid = msg1.StdDevResid;
			aMsg1.NavMode = msg1.NavMode;
			aMsg1.extAgeOfDiff = msg1.extAgeOfDiff;
			aMsg1.CheckSum = msg1.CheckSum;
			aMsg1.CRLF = msg1.CRLF;
			           
			 aVMsg1.push_back(aMsg1);
			           
		 }
		 
		 else
		 {
			fread_one(data,msg.datalength,aSource);                   		//Data
			fread_one(&msg.checksum,sizeof(msg.checksum),aSource);     		//checksum
			fread_one(&msg.cr,sizeof(msg.cr),aSource);                 		//CR
			fread_one(&msg.lf,sizeof(msg.lf),aSource);                 		//LF
		 }

		 delete [] data;
    }
    
	fclose(aSource);
	
	std::cout << "Number of Bin01 Msgs = " << aVMsg1.size() << std::endl;
	
	//get vector of TimeMark
	std::vector<TimeMark> aVTM = GetTMFromVBin01(aVMsg1);
	
	std::cout << "Number of TimeMark = " << aVTM.size() << std::endl;
	
	//write vector of TimeMark
	WriteTMInFile(aVTM, aOut, aAddFormat);
	
}

int ExportHemisTM_main(int argc,char ** argv)
{
	 cEHTM_Appli anAppli(argc,argv);
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
