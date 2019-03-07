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
#include "hassan/reechantillonnage.h"
#include "RPC.h"



void ASTERGT2MM_Banniere()
{
	std::cout << "\n";
	std::cout << " *********************************\n";
	std::cout << " *     ASTER - GeoTiff           *\n";
	std::cout << " *     2                         *\n";
	std::cout << " *     MicMac Xml                *\n";
	std::cout << " *********************************\n\n";
}

void ASTERGT_Strip_2_MM_Banniere()
{
	std::cout << "\n";
	std::cout << " *********************************\n";
	std::cout << " *     ASTER - GeoTiff           *\n";
	std::cout << " *     Strips of scenes          *\n";
	std::cout << " *     2                         *\n";
	std::cout << " *     MicMac Xml                *\n";
	std::cout << " *********************************\n\n";
}

vector<Pt3dr> ReadCorTable(string nameFile)
{
	vector<Pt3dr> aCorTable;
	std::fstream aFile(nameFile.c_str(), std::ios_base::in);

	// Make sure the file stream is good
	if (!aFile) {
		cout << endl << "Failed to open Radiommetric Correction file " << nameFile;
		return aCorTable;
	}

	double a, b, c;
	while (aFile >> a >> b >> c)
	{
		Pt3dr aPt(a, b, c);
		//cout << aPt << endl;
		aCorTable.push_back(aPt);
	}

	cout << "Size for " << nameFile << " : " << aCorTable.size() << endl;
	return aCorTable;
}

string ReadDate(string aNameFile)
{
	std::fstream aFile(aNameFile.c_str(), std::ios_base::in);
	string aDate = "0000000";

	// Make sure the file stream is good
	if (!aFile) {
		cout << endl << "Failed to open Time obs file " << aNameFile;
		return aDate;
	}

	double a, b, c, d;
	aFile >> a >> b >> c >> d;
	/*
	atime.push_back(a);
	atime.push_back(b);
	atime.push_back(c);
	atime.push_back(d);*/

	//January 0st 1958 is Day 0 in ASTER, Jan 1st 1970 is day 0 in C++ (UNIX Time), so -4384 days difference
	double aTime = a + (b * 100000 + c + d / 1000) / (24 * 3600 * 1000) - 4384;
	time_t _tm = aTime * 86400;

	struct tm * curtime = localtime(&_tm);
	std::ostringstream oss;

#if __cplusplus < 201103L or __GNUC__ < 5
	char buffer[256];
	strftime(buffer, 256, "%Y%m%d", curtime);
	oss << buffer;
#else
	oss << std::put_time(curtime, "%Y%m%d");
#endif
	aDate = oss.str();

	return aDate;
}

void DestripASTER(string aDir, string aNameFile, string aOutDir)
{

	//Reading correction tables
	vector<Pt3dr> Cor_3N = ReadCorTable(aDir + aNameFile + ".VNIR_Band3N.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_3B = ReadCorTable(aDir + aNameFile + ".VNIR_Band3B.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_1 = ReadCorTable(aDir + aNameFile + ".VNIR_Band1.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_2 = ReadCorTable(aDir + aNameFile + ".VNIR_Band2.RadiometricCorrTable.txt");

	//Reading the image and applying correction tables
	Tiff_Im aTF_3N = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band3N.ImageData.tif", 1, false);
	Tiff_Im aTF_3B = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band3B.ImageData.tif", 1, false);
	Tiff_Im aTF_1 = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band1.ImageData.tif", 1, false);
	Tiff_Im aTF_2 = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band2.ImageData.tif", 1, false);

	Pt2di aSz_3N = aTF_3N.sz(); cout << "size of image 3N = " << aSz_3N << endl;
	Im2D_U_INT1  aIm_3N(aSz_3N.x, aSz_3N.y);
	Pt2di aSz_3B = aTF_3B.sz(); cout << "size of image 3B = " << aSz_3B << endl;
	Im2D_U_INT1  aIm_3B(aSz_3B.x, aSz_3B.y);
	Pt2di aSz_1 = aTF_1.sz(); cout << "size of image 1 = " << aSz_1 << endl;
	Im2D_U_INT1  aIm_1(aSz_1.x, aSz_1.y);
	Pt2di aSz_2 = aTF_2.sz(); cout << "size of image 2 = " << aSz_2 << endl;
	Im2D_U_INT1  aIm_2(aSz_2.x, aSz_2.y);

	ELISE_COPY
	(
		aTF_3N.all_pts(),
		aTF_3N.in(),
		aIm_3N.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);

	U_INT1 ** aData_3N = aIm_3N.data();

	ELISE_COPY
	(
		aTF_3B.all_pts(),
		aTF_3B.in(),
		aIm_3B.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);

	U_INT1 ** aData_3B = aIm_3B.data();

	ELISE_COPY
	(
		aTF_1.all_pts(),
		aTF_1.in(),
		aIm_1.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);

	U_INT1 ** aData_1 = aIm_1.data();

	ELISE_COPY
	(
		aTF_2.all_pts(),
		aTF_2.in(),
		aIm_2.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);

	U_INT1 ** aData_2 = aIm_2.data();



	for (size_t aX = 0; aX < Cor_3N.size(); aX++)
	{
		for (int aY = 0; aY < aSz_3N.y; aY++)
		{
			if (Cor_1[aX].y*double(aData_1[aY][aX]) / Cor_1[aX].z + Cor_1[aX].x > 255)
				aData_1[aY][aX] = 255;
			else
				aData_1[aY][aX] = Cor_1[aX].y*double(aData_1[aY][aX]) / Cor_1[aX].z + Cor_1[aX].x;
			if (Cor_2[aX].y*double(aData_2[aY][aX]) / Cor_2[aX].z + Cor_2[aX].x > 255)
				aData_2[aY][aX] = 255;
			else
				aData_2[aY][aX] = Cor_2[aX].y*double(aData_2[aY][aX]) / Cor_2[aX].z + Cor_2[aX].x;
			if (Cor_3N[aX].y*double(aData_3N[aY][aX]) / Cor_3N[aX].z + Cor_3N[aX].x > 255)
				aData_3N[aY][aX] = 255;
			else
				aData_3N[aY][aX] = Cor_3N[aX].y*double(aData_3N[aY][aX]) / Cor_3N[aX].z + Cor_3N[aX].x;
		}
	}

	for (size_t aX = 0; aX < Cor_3B.size(); aX++)
	{
		for (int aY = 0; aY < aSz_3B.y; aY++)
		{
			if (Cor_3B[aX].y*double(aData_3B[aY][aX]) / Cor_3B[aX].z + Cor_3B[aX].x > 255)
				aData_3B[aY][aX] = 255;
			else
				aData_3B[aY][aX] = Cor_3B[aX].y*double(aData_3B[aY][aX]) / Cor_3B[aX].z + Cor_3B[aX].x;
		}
	}


	//Output
	string aNameOut_3N = aOutDir + aNameFile + "_3N.tif";

	Tiff_Im  aTOut_3N
	(
		aNameOut_3N.c_str(),
		aSz_3N,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);


	ELISE_COPY
	(
		aTOut_3N.all_pts(),
		aIm_3N.in(),
		aTOut_3N.out()
	);


	string aNameOut_3B = aOutDir + aNameFile + "_3B.tif";
	Tiff_Im  aTOut_3B
	(
		aNameOut_3B.c_str(),
		aSz_3B,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);


	ELISE_COPY
	(
		aTOut_3B.all_pts(),
		aIm_3B.in(),
		aTOut_3B.out()
	);

	string aNameOut_FC = aOutDir + "FalseColor_" + aNameFile + ".tif";
	Tiff_Im  aTOut_FC
	(
		aNameOut_FC.c_str(),
		aSz_1,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::RGB
	);


	ELISE_COPY
	(
		aTOut_FC.all_pts(),
		Virgule(aIm_3N.in(), aIm_2.in(), aIm_1.in()),
		aTOut_FC.out()
	);


}

vector<Pt2dr> ReadLatticePointsIm(string aNameFile, bool doDistortionCorrection)
{
	vector<Pt2dr> aLatticePoints;

	std::fstream aFile(aNameFile.c_str(), std::ios_base::in);

	// Make sure the file stream is good
	if (!aFile) {
		cout << endl << "Failed to open Lattice Points file " << aNameFile;
		return aLatticePoints;
	}

	double a, b;
	while (aFile >> a >> b)
	{
		Pt2dr aPt(a, b);
		if (doDistortionCorrection)
		{
			//Actual function is upcoming
			aPt.x = aPt.x;//x'=x+f1(x)
			aPt.y = aPt.y;//y'=y+f2(x)
		}
		//cout << aPt << endl;
		aLatticePoints.push_back(aPt);
	}
	return aLatticePoints;

}

vector<Pt3dr> ReadSattelitePos(string aNameFile)
{
	vector<Pt3dr> aSattelitePos;

	std::fstream aFile(aNameFile.c_str(), std::ios_base::in);

	// Make sure the file stream is good
	if (!aFile) {
		cout << endl << "Failed to open Sattelite Positions file " << aNameFile;
		return aSattelitePos;
	}

	double a, b, c;
	while (aFile >> a >> b >> c)
	{
		Pt3dr aPt(a, b, c);
		//cout << "Pt : " << aPt << endl;
		aSattelitePos.push_back(aPt);
	}
	return aSattelitePos;

}

vector<Pt3dr> ReadLatticeECEF(string aNameLonFile, string aNameLatFile)
{

	vector<Pt3dr> aLatticeECEF;

	std::fstream aLonFile(aNameLonFile.c_str(), std::ios_base::in);
	std::fstream aLatFile(aNameLatFile.c_str(), std::ios_base::in);

	// Make sure the file stream is good
	if (!aLonFile || !aLatFile) {
		cout << endl << "Failed to open Longitude or Lattitude file " << aNameLonFile << " or " << aNameLatFile << endl;
		return aLatticeECEF;
	}

	vector<double> aLongitudes;
	vector<double> aLatitudes;
	double aVal;

	while (aLonFile >> aVal)
	{
		aLongitudes.push_back(aVal);
	}

	//cout << "aLongitudes size : " << aLongitudes.size() << endl;

	while (aLatFile >> aVal)
	{
		aLatitudes.push_back(aVal);
	}


	//cout << "aLatitudes size : " << aLatitudes.size() << " x " << aLatitudes[0].size() << endl;
	//cout << aLatitudes[0] << endl;

	// Convert points to ECEF(geocentric euclidian)
	double a = 6378137;
	double b = (1 - 1 / 298.257223563)*a;
	for (u_int i = 0; i < aLatitudes.size(); i++)
	{
		double aSinLat = sin(aLatitudes[i] * M_PI / 180);
		double aCosLat = cos(aLatitudes[i] * M_PI / 180);
		double aSinLon = sin(aLongitudes[i] * M_PI / 180);
		double aCosLon = cos(aLongitudes[i] * M_PI / 180);
		double r = sqrt(a*a*b*b / (a*a*aSinLat*aSinLat + b*b*aCosLat*aCosLat));
		double x = r*aCosLat*aCosLon;
		double y = r*aCosLat*aSinLon;
		double z = r*aSinLat;
		Pt3dr aPt(x, y, z);
		//cout << aPt << endl;
		aLatticeECEF.push_back(aPt);

	}

	return aLatticeECEF;
}


void ASTERXMLWrite(string aNameFile, string aDate, vector<Pt2dr> aLatticePointsIm, vector<Pt3dr> aSatellitePosition, vector<Pt3dr> aLatticeECEF)
{
	std::ofstream fic(aNameFile.c_str());
	fic << std::setprecision(16);

	fic << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>" << endl;
	fic << "<AsterMeta version=\"0.4\">" << endl;
	fic << "\t<Date>" << aDate << "</Date>" << endl;

	fic << "\t<LatticePoints>" << endl;
	fic << "\t\t<NbLattice>" << aLatticePointsIm.size() << "</NbLattice>" << endl;
	for (size_t i = 0; i<aLatticePointsIm.size(); i++)
	{
		fic << "\t\t<LatticePoint_" << i + 1 << ">" << aLatticePointsIm[i].x << " " << aLatticePointsIm[i].y << "</LatticePoint_" << i + 1 << ">" << endl;
	}
	fic << "\t</LatticePoints>" << endl;

	fic << "\t<SatellitePositions>" << endl;
	fic << "\t\t<NbSatPos>" << aSatellitePosition.size() << "</NbSatPos>" << endl;
	for (size_t i = 0; i<aSatellitePosition.size(); i++)
	{
		fic << "\t\t<SatPos_" << i + 1 << ">" << aSatellitePosition[i].x << " " << aSatellitePosition[i].y << " " << aSatellitePosition[i].z << "</SatPos_" << i + 1 << ">" << endl;
	}
	fic << "\t</SatellitePositions>" << endl;

	fic << "\t<ECEFs>" << endl;
	fic << "\t\t<NbECEF>" << aLatticeECEF.size() << "</NbECEF>" << endl;
	for (size_t i = 0; i<aLatticeECEF.size(); i++)
	{
		fic << "\t\t<ECEF_" << i + 1 << ">" << aLatticeECEF[i].x << " " << aLatticeECEF[i].y << " " << aLatticeECEF[i].z << "</ECEF_" << i + 1 << ">" << endl;
	}
	fic << "\t</ECEFs>" << endl;

	fic << "</AsterMeta>" << endl;

	fic.close();
}

int ASTERGT2MM_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aNameFile;
	std::string aOutDir = "../";
	//Reading the arguments
	ElInitArgMain
	(
		argc, argv,
		LArgMain()
		<< EAMC(aNameFile, "Name of ASTER scene", eSAM_IsPatFile),
		LArgMain()
		<< EAM(aOutDir, "DirOut", true, "Output folder (end with /) , def is ../")
	);

	std::string aDir, aPatIm;
	SplitDirAndFile(aDir, aPatIm, aNameFile);

	//Destripping the images using RadiommetricCorrectionTable
	DestripASTER(aDir, aNameFile, aOutDir);

	string aDate = ReadDate(aNameFile + ".VNIR_Band3N.ObservationTime.txt");

	vector<Pt2dr> aLatticePointsIm_3N = ReadLatticePointsIm(aNameFile + ".VNIR_Band3N.LatticePoint.txt", false);
	cout << "LatticePointsIm_3N read (" << aLatticePointsIm_3N.size() << " points)" << endl;
	vector<Pt2dr> aLatticePointsIm_3B = ReadLatticePointsIm(aNameFile + ".VNIR_Band3B.LatticePoint.txt", true);
	cout << "LatticePointsIm_3B read (" << aLatticePointsIm_3B.size() << " points)" << endl;

	vector<Pt3dr> aSatellitePosition_3N = ReadSattelitePos(aNameFile + ".VNIR_Band3N.SatellitePosition.txt");
	cout << "SatellitePosition_3N read (" << aSatellitePosition_3N.size() << " points)" << endl;
	vector<Pt3dr> aSatellitePosition_3B = ReadSattelitePos(aNameFile + ".VNIR_Band3B.SatellitePosition.txt");
	cout << "SatellitePosition_3B read (" << aSatellitePosition_3B.size() << " points)" << endl;

	vector<Pt3dr> aLatticeECEF_3N = ReadLatticeECEF(aNameFile + ".VNIR_Band3N.Longitude.txt", aNameFile + ".VNIR_Band3N.Latitude.txt");
	cout << "LatticeECEF_3N read (" << aLatticeECEF_3N.size() << " points)" << endl;
	vector<Pt3dr> aLatticeECEF_3B = ReadLatticeECEF(aNameFile + ".VNIR_Band3B.Longitude.txt", aNameFile + ".VNIR_Band3B.Latitude.txt");
	cout << "LatticeECEF_3B read (" << aLatticeECEF_3B.size() << " points)" << endl;


	//Write XML
	ASTERXMLWrite(aOutDir + "FalseColor_" + aNameFile + ".xml", aDate, aLatticePointsIm_3N, aSatellitePosition_3N, aLatticeECEF_3N);
	ASTERXMLWrite(aOutDir + aNameFile + "_3N.xml", aDate, aLatticePointsIm_3N, aSatellitePosition_3N, aLatticeECEF_3N);
	ASTERXMLWrite(aOutDir + aNameFile + "_3B.xml", aDate, aLatticePointsIm_3B, aSatellitePosition_3B, aLatticeECEF_3B);



	ASTERGT2MM_Banniere();
	return 0;
}








void ConcatenateASTERImages(string aDir, string aOutDir, vector<int> aVectDistancesBetweenImages3N, vector<int> aVectDistancesBetweenImages3B, list<string> ListScenes, string aStripName)
{

	int nbScenes = (int)ListScenes.size();
	string aNameFile = ListScenes.front();
	aNameFile.erase(aNameFile.end() - 26, aNameFile.end());

	//Reading correction tables
	vector<Pt3dr> Cor_3N = ReadCorTable(aDir + aNameFile + ".VNIR_Band3N.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_3B = ReadCorTable(aDir + aNameFile + ".VNIR_Band3B.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_1 = ReadCorTable(aDir + aNameFile + ".VNIR_Band1.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_2 = ReadCorTable(aDir + aNameFile + ".VNIR_Band2.RadiometricCorrTable.txt");

	//Computing size of output image
	int aSize_3Ny = 4200 + aVectDistancesBetweenImages3N.back();
	std::cout << "Size (1/2/3N)_y:" << aSize_3Ny << endl;

	int aSize_3By = 5400 + aVectDistancesBetweenImages3B.back();
	std::cout << "Size 3B_y:" << aSize_3By << endl;

	Pt2di aSize3N = { 4100,aSize_3Ny };
	Pt2di aSize3B = { 5000,aSize_3By };

	//creating output files
	Im2D_U_INT1  aIm_3N(4100, aSize_3Ny);
	Im2D_U_INT1  aIm_3B(5000, aSize_3By);
	Im2D_U_INT1  aIm_1(4100, aSize_3Ny);
	Im2D_U_INT1  aIm_2(4100, aSize_3Ny);

	U_INT1 ** aData_3N = aIm_3N.data();
	U_INT1 ** aData_3B = aIm_3B.data();
	U_INT1 ** aData_1 = aIm_1.data();
	U_INT1 ** aData_2 = aIm_2.data();

	//for each image
	for (u_int i = 0; int(i) < nbScenes; i++)
	{

		aNameFile = ListScenes.front();
		aNameFile.erase(aNameFile.end() - 26, aNameFile.end());
		ListScenes.pop_front();

		//Reading the image and applying correction tables
		Tiff_Im aTF_3N = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band3N.ImageData.tif", 1, false);
		Tiff_Im aTF_3B = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band3B.ImageData.tif", 1, false);
		Tiff_Im aTF_1 = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band1.ImageData.tif", 1, false);
		Tiff_Im aTF_2 = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band2.ImageData.tif", 1, false);

		Pt2di aSz_3N = aTF_3N.sz();
		Im2D_U_INT1  aIm_3N_loc(aSz_3N.x, aSz_3N.y);
		Pt2di aSz_3B = aTF_3B.sz();
		Im2D_U_INT1  aIm_3B_loc(aSz_3B.x, aSz_3B.y);
		Pt2di aSz_1 = aTF_1.sz();
		Im2D_U_INT1  aIm_1_loc(aSz_1.x, aSz_1.y);
		Pt2di aSz_2 = aTF_2.sz();
		Im2D_U_INT1  aIm_2_loc(aSz_2.x, aSz_2.y);

		ELISE_COPY
		(
			aTF_3N.all_pts(),
			aTF_3N.in(),
			aIm_3N_loc.out()
		);

		U_INT1 ** aData_3N_loc = aIm_3N_loc.data();

		ELISE_COPY
		(
			aTF_3B.all_pts(),
			aTF_3B.in(),
			aIm_3B_loc.out()
		);

		U_INT1 ** aData_3B_loc = aIm_3B_loc.data();

		ELISE_COPY
		(
			aTF_1.all_pts(),
			aTF_1.in(),
			aIm_1_loc.out()
		);

		U_INT1 ** aData_1_loc = aIm_1_loc.data();

		ELISE_COPY
		(
			aTF_2.all_pts(),
			aTF_2.in(),
			aIm_2_loc.out()
		);

		U_INT1 ** aData_2_loc = aIm_2_loc.data();


		//Compute image for 3N
		for (size_t aX = 0; aX < Cor_3N.size(); aX++)
		{
			for (int aY = 0; aY < aSz_3N.y; aY++)
			{
				if (Cor_1[aX].y*double(aData_1_loc[aY][aX]) / Cor_1[aX].z + Cor_1[aX].x > 255)
					aData_1[aY+aVectDistancesBetweenImages3N[i]][aX] = 255;
				else
					aData_1[aY + aVectDistancesBetweenImages3N[i]][aX] = Cor_1[aX].y*double(aData_1_loc[aY][aX]) / Cor_1[aX].z + Cor_1[aX].x;
				if (Cor_2[aX].y*double(aData_2_loc[aY][aX]) / Cor_2[aX].z + Cor_2[aX].x > 255)
					aData_2[aY + aVectDistancesBetweenImages3N[i]][aX] = 255;
				else
					aData_2[aY + aVectDistancesBetweenImages3N[i]][aX] = Cor_2[aX].y*double(aData_2_loc[aY][aX]) / Cor_2[aX].z + Cor_2[aX].x;
				if (Cor_3N[aX].y*double(aData_3N_loc[aY][aX]) / Cor_3N[aX].z + Cor_3N[aX].x > 255)
					aData_3N[aY + aVectDistancesBetweenImages3N[i]][aX] = 255;
				else
					aData_3N[aY + aVectDistancesBetweenImages3N[i]][aX] = Cor_3N[aX].y*double(aData_3N_loc[aY][aX]) / Cor_3N[aX].z + Cor_3N[aX].x;
			}
		}

		for (size_t aX = 0; aX < Cor_3B.size(); aX++)
		{
			for (int aY = 0; aY < aSz_3B.y; aY++)
			{
				if (Cor_3B[aX].y*double(aData_3B_loc[aY][aX]) / Cor_3B[aX].z + Cor_3B[aX].x > 255)
					aData_3B[aY + aVectDistancesBetweenImages3B[i]][aX] = 255;
				else
					aData_3B[aY + aVectDistancesBetweenImages3B[i]][aX] = Cor_3B[aX].y*double(aData_3B_loc[aY][aX]) / Cor_3B[aX].z + Cor_3B[aX].x;
			}
		}
	}

	//Output
	string aNameOut_3N = aOutDir + aStripName + "_3N.tif";

	Tiff_Im  aTOut_3N
	(
		aNameOut_3N.c_str(),
		aSize3N,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);


	ELISE_COPY
	(
		aTOut_3N.all_pts(),
		aIm_3N.in(),
		aTOut_3N.out()
	);


	string aNameOut_3B = aOutDir + aStripName + "_3B.tif";
	Tiff_Im  aTOut_3B
	(
		aNameOut_3B.c_str(),
		aSize3B,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);


	ELISE_COPY
	(
		aTOut_3B.all_pts(),
		aIm_3B.in(),
		aTOut_3B.out()
	);

	string aNameOut_FC = aOutDir + "FalseColor_" + aStripName + ".tif";
	Tiff_Im  aTOut_FC
	(
		aNameOut_FC.c_str(),
		aSize3N,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::RGB
	);


	ELISE_COPY
	(
		aTOut_FC.all_pts(),
		Virgule(aIm_3N.in(), aIm_2.in(), aIm_1.in()),
		aTOut_FC.out()
	);

}






int ASTERGT_strip_2_MM_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aPatScenesInit, aStripName;
	std::string aOutDir = "../";
	//Reading the arguments
	ElInitArgMain
	(
		argc, argv,
		LArgMain()
		<< EAMC(aPatScenesInit, "Regular expression of ASTER scenes names", eSAM_IsPatFile)
		<< EAMC(aStripName, "Name of output ASTER strip", eSAM_IsPatFile),
		LArgMain()
	);

	aPatScenesInit = aPatScenesInit + ".*.VNIR_Band3B.ImageData.tif";
	std::string aDir, aPatScenes;
	SplitDirAndFile(aDir, aPatScenes, aPatScenesInit);


	//Reading input files
	list<string> ListScenes = RegexListFileMatch(aDir, aPatScenes, 1, false);
	int nbScenes = (int)ListScenes.size();
	std::cout << "Scenes in strip: " << nbScenes << endl;
	std::cout << "Strip name: " << aStripName << endl;

	vector<int> aVectDistancesBetweenImages3N = { 0 };
	vector<int> aVectDistancesBetweenImages3B = { 0 };


	string aDate = ListScenes.front().substr(11, 8);

	//Read data for first image of band
	string aNameFile = ListScenes.front();
	aNameFile.erase(aNameFile.end() - 26, aNameFile.end());
	std::cout << "Processing image: " << aNameFile << endl;
	ListScenes.pop_front();

	// Lattice points coordinates in image
	vector<Pt2dr> aLatticePointsIm_3N = ReadLatticePointsIm(aNameFile + ".VNIR_Band3N.LatticePoint.txt", false);
	vector<Pt2dr> aLatticePointsIm_3B = ReadLatticePointsIm(aNameFile + ".VNIR_Band3B.LatticePoint.txt", true);

	// Satellite positions for each lattice lines
	vector<Pt3dr> aSatellitePosition_3N = ReadSattelitePos(aNameFile + ".VNIR_Band3N.SatellitePosition.txt");
	vector<Pt3dr> aSatellitePosition_3B = ReadSattelitePos(aNameFile + ".VNIR_Band3B.SatellitePosition.txt");

	// Ground position for each lattice point
	vector<Pt3dr> aLatticeECEF_3N = ReadLatticeECEF(aNameFile + ".VNIR_Band3N.Longitude.txt", aNameFile + ".VNIR_Band3N.Latitude.txt");
	vector<Pt3dr> aLatticeECEF_3B = ReadLatticeECEF(aNameFile + ".VNIR_Band3B.Longitude.txt", aNameFile + ".VNIR_Band3B.Latitude.txt");

	//Concatenanted data from other images of band (in image and ground geometry, and associated sattellite positions)
	int aDistanceBetweenImages;

	for (u_int i = 0; int(i) < nbScenes - 1; i++)
	{
		//Getting image name
		aNameFile = ListScenes.front();
		aNameFile.erase(aNameFile.end() - 26, aNameFile.end());
		std::cout << "Processing image: " << aNameFile << endl;
		ListScenes.pop_front();

		// Satellite positions for each lattice lines
		vector<Pt3dr> aSatellitePosition_3N_loc = ReadSattelitePos(aNameFile + ".VNIR_Band3N.SatellitePosition.txt");
		//Find duplicate in previous images
		u_int aNbDuplicateN = 0;
		for (u_int j = 0; j < aSatellitePosition_3N.size(); j++)
			{
				if (aSatellitePosition_3N[j] == aSatellitePosition_3N_loc[0]) { aNbDuplicateN = aSatellitePosition_3N.size() - j; }
			}
		aSatellitePosition_3N.erase(aSatellitePosition_3N.end() - aNbDuplicateN, aSatellitePosition_3N.end());
		aSatellitePosition_3N.insert(aSatellitePosition_3N.end(), aSatellitePosition_3N_loc.begin(), aSatellitePosition_3N_loc.end());

		vector<Pt3dr> aSatellitePosition_3B_loc = ReadSattelitePos(aNameFile + ".VNIR_Band3B.SatellitePosition.txt");
		u_int aNbDuplicateB = 0;
		for (u_int j = 0; j < aSatellitePosition_3B.size(); j++)
		{
			if (aSatellitePosition_3B[j] == aSatellitePosition_3B_loc[0]) { aNbDuplicateB = aSatellitePosition_3B.size() - j; }
		}
		aSatellitePosition_3B.erase(aSatellitePosition_3B.end() - aNbDuplicateB, aSatellitePosition_3B.end());
		aSatellitePosition_3B.insert(aSatellitePosition_3B.end(), aSatellitePosition_3B_loc.begin(), aSatellitePosition_3B_loc.end());

		// Ground position for each lattice point
		vector<Pt3dr> aLatticeECEF_3N_loc = ReadLatticeECEF(aNameFile + ".VNIR_Band3N.Longitude.txt", aNameFile + ".VNIR_Band3N.Latitude.txt");
		aLatticeECEF_3N.erase(aLatticeECEF_3N.end() - aNbDuplicateN * 11, aLatticeECEF_3N.end());
		aLatticeECEF_3N.insert(aLatticeECEF_3N.end(), aLatticeECEF_3N_loc.begin(), aLatticeECEF_3N_loc.end());
		vector<Pt3dr> aLatticeECEF_3B_loc = ReadLatticeECEF(aNameFile + ".VNIR_Band3B.Longitude.txt", aNameFile + ".VNIR_Band3B.Latitude.txt");
		aLatticeECEF_3B.erase(aLatticeECEF_3B.end() - aNbDuplicateB * 11, aLatticeECEF_3B.end());
		aLatticeECEF_3B.insert(aLatticeECEF_3B.end(), aLatticeECEF_3B_loc.begin(), aLatticeECEF_3B_loc.end());



		//Computing coordinate of lattice points in the concatenated image
		//3N
		vector<Pt2dr> aLatticePointsIm_3N_loc = ReadLatticePointsIm(aNameFile + ".VNIR_Band3N.LatticePoint.txt", false);
		aDistanceBetweenImages = aLatticePointsIm_3N.back().y - aLatticePointsIm_3N_loc[aNbDuplicateN * 11 - 1].y;
		aVectDistancesBetweenImages3N.push_back(aDistanceBetweenImages);
		//two lines of points are overlaping, 11 pts per line, 22nd element of next image is the last element of previous image

		for (u_int j = 0; j < aLatticePointsIm_3N_loc.size(); j++)
		{
			aLatticePointsIm_3N_loc[j].y = aDistanceBetweenImages + aLatticePointsIm_3N_loc[j].y;
		}
		aLatticePointsIm_3N.erase(aLatticePointsIm_3N.end() - aNbDuplicateN * 11, aLatticePointsIm_3N.end());
		aLatticePointsIm_3N.insert(aLatticePointsIm_3N.end(), aLatticePointsIm_3N_loc.begin(), aLatticePointsIm_3N_loc.end());


		//3B
		vector<Pt2dr> aLatticePointsIm_3B_loc = ReadLatticePointsIm(aNameFile + ".VNIR_Band3B.LatticePoint.txt", true);
		aDistanceBetweenImages = aLatticePointsIm_3B.back().y - aLatticePointsIm_3B_loc[aNbDuplicateB * 11 - 1].y;
		aVectDistancesBetweenImages3B.push_back(aDistanceBetweenImages);
		//six lines of points are overlaping, 11 pts per line, 66nd element of next image is the last element of previous image

		for (u_int j = 0; j < aLatticePointsIm_3B_loc.size(); j++)
		{
			aLatticePointsIm_3B_loc[j].y = aDistanceBetweenImages + aLatticePointsIm_3B_loc[j].y;
		}
		aLatticePointsIm_3B.erase(aLatticePointsIm_3B.end() - aNbDuplicateB * 11, aLatticePointsIm_3B.end());
		aLatticePointsIm_3B.insert(aLatticePointsIm_3B.end(), aLatticePointsIm_3B_loc.begin(), aLatticePointsIm_3B_loc.end());

	}

	// Print counts
	std::cout << "LatticePointsIm_3N read (" << aLatticePointsIm_3N.size() << " points)" << endl;
	std::cout << "LatticePointsIm_3B read (" << aLatticePointsIm_3B.size() << " points)" << endl;
	std::cout << "SatellitePosition_3N read (" << aSatellitePosition_3N.size() << " points)" << endl;
	std::cout << "SatellitePosition_3B read (" << aSatellitePosition_3B.size() << " points)" << endl;
	std::cout << "LatticeECEF_3N read (" << aLatticeECEF_3N.size() << " points)" << endl;
	std::cout << "LatticeECEF_3B read (" << aLatticeECEF_3B.size() << " points)" << endl;









	//Write XML
	ASTERXMLWrite(aOutDir + "FalseColor_" + aStripName + ".xml", aDate, aLatticePointsIm_3N, aSatellitePosition_3N, aLatticeECEF_3N);
	ASTERXMLWrite(aOutDir + aStripName + "_3N.xml", aDate, aLatticePointsIm_3N, aSatellitePosition_3N, aLatticeECEF_3N);
	ASTERXMLWrite(aOutDir + aStripName + "_3B.xml", aDate, aLatticePointsIm_3B, aSatellitePosition_3B, aLatticeECEF_3B);



	// Concatenating image and destripping the images using RadiommetricCorrectionTable (same for all images of a single strip)
	ListScenes = RegexListFileMatch(aDir, aPatScenes, 1, false);
	ConcatenateASTERImages(aDir, aOutDir, aVectDistancesBetweenImages3N, aVectDistancesBetweenImages3B, ListScenes, aStripName);



	ASTERGT_Strip_2_MM_Banniere();
	return 0;
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
