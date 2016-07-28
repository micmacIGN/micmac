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


//AST_L1A_00307062000183135_20160705052720_10177.VNIR_Band3N.ImageData

void ASTERGT2MM_Banniere()
{
	std::cout << "\n";
	std::cout << " *********************************\n";
	std::cout << " *     ASTER - GeoTiff           *\n";
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
	string aDate ="0000000";

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
	time_t _tm = aTime*86400;

	struct tm * curtime = localtime(&_tm);
	std::ostringstream oss;
	
	#if __cplusplus < 201103L
		char buffer[256];
		strftime(buffer, 256, "%Y%m%d", curtime);
		oss << buffer;
	#else
		oss << std::put_time(curtime, "%Y%m%d");
	#endif
	aDate=oss.str();

return aDate;
}

void DestripASTER(string aDir, string aNameFile, string aOutDir)
{

	//Reading correction tables
	vector<Pt3dr> Cor_3N = ReadCorTable(aDir + aNameFile + ".VNIR_Band3N.RadiometricCorrTable.txt");
	vector<Pt3dr> Cor_3B = ReadCorTable(aDir + aNameFile + ".VNIR_Band3B.RadiometricCorrTable.txt");

	//Reading the image and applying correction tables
	Tiff_Im aTF_3N = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band3N.ImageData.tif", 1, false);
	Tiff_Im aTF_3B = Tiff_Im::StdConvGen(aDir + aNameFile + ".VNIR_Band3B.ImageData.tif", 1, false);

	Pt2di aSz_3N = aTF_3N.sz(); cout << "size of image 3N = " << aSz_3N << endl;
	Im2D_U_INT1  aIm_3N(aSz_3N.x, aSz_3N.y);
	Pt2di aSz_3B = aTF_3B.sz(); cout << "size of image 3B = " << aSz_3B << endl;
	Im2D_U_INT1  aIm_3B(aSz_3B.x, aSz_3B.y);

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



	for (size_t aX = 0; aX < Cor_3N.size(); aX++)
	{
		for (int aY = 0; aY < aSz_3N.y; aY++)
		{
			aData_3N[aY][aX] = Cor_3N[aX].y*double(aData_3N[aY][aX]) / Cor_3N[aX].z + Cor_3N[aX].x;
		}
	}

	for (size_t aX = 0; aX < Cor_3B.size(); aX++)
	{
		for (int aY = 0; aY < aSz_3B.y; aY++)
		{
			aData_3B[aY][aX] = Cor_3B[aX].y*double(aData_3B[aY][aX]) / Cor_3B[aX].z + Cor_3B[aX].x;
		}
	}


	//Output
	string aNameOut_3N = aOutDir + aNameFile + ".3N_Destrip.tif";

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


	string aNameOut_3B = aOutDir + aNameFile + ".3B_Destrip.tif";
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


}

vector<Pt2dr> ReadLatticePointsIm(string aNameFile)
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

	vector<vector<double> > aLongitudes;
	vector<vector<double> > aLatitudes;

	double a, b, c, d, e, f, g, h, i, j, k;
	while (aLonFile >> a >> b >> c >> d >> e >> f >> g >> h >> i >> j >> k)
	{
		double aLongitudeLvla[11] = { a, b, c, d, e, f, g, h, i, j, k };
		vector<double> aLongitudeLvl(aLongitudeLvla, aLongitudeLvla + sizeof aLongitudeLvla / sizeof aLongitudeLvla[0]);
		aLongitudes.push_back(aLongitudeLvl);
	}

	//cout << "aLongitudes size : " << aLongitudes.size() << endl;

	while (aLatFile >> a >> b >> c >> d >> e >> f >> g >> h >> i >> j >> k)
	{
		double aLatitudeLvla[11] = { a, b, c, d, e, f, g, h, i, j, k };
		vector<double> aLatitudeLvl(aLatitudeLvla, aLatitudeLvla + sizeof aLatitudeLvla / sizeof aLatitudeLvla[0]);
		aLatitudes.push_back(aLatitudeLvl);
	}


	//cout << "aLatitudes size : " << aLatitudes.size() << " x " << aLatitudes[0].size() << endl;
	//cout << aLatitudes[0] << endl;

	// Convert points to ECEF(geocentric euclidian)
	a = 6378137;
	b = (1 - 1 / 298.257223563)*a;
	for (u_int i = 0; i < aLatitudes.size(); i++)
	{
		//cout << aLatitudes[i] << endl;
		//cout << aLongitudes[i] << endl;
		for (u_int j = 0; j < aLatitudes[0].size(); j++)
		{
			double aSinLat = sin(aLatitudes[i][j]* M_PI / 180);
			double aCosLat = cos(aLatitudes[i][j] * M_PI / 180);
			double aSinLon = sin(aLongitudes[i][j] * M_PI / 180);
			double aCosLon = cos(aLongitudes[i][j] * M_PI / 180);
			double r = sqrt(a*a*b*b / (a*a*aSinLat*aSinLat + b*b*aCosLat*aCosLat));
			double x = r*aCosLat*aCosLon;
			double y = r*aCosLat*aSinLon;
			double z = r*aSinLat;
			Pt3dr aPt(x, y, z);
			//cout << aPt << endl;
			aLatticeECEF.push_back(aPt);
		}

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
	fic << "\t\t<NbLattice>"<< aLatticePointsIm.size() << "</NbLattice>" << endl;
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

	vector<Pt2dr> aLatticePointsIm_3N = ReadLatticePointsIm(aNameFile + ".VNIR_Band3N.LatticePoint.txt");
	cout << "LatticePointsIm_3N read (" << aLatticePointsIm_3N.size() << " points)" << endl;
	vector<Pt2dr> aLatticePointsIm_3B = ReadLatticePointsIm(aNameFile + ".VNIR_Band3B.LatticePoint.txt");
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
	ASTERXMLWrite(aOutDir + aNameFile + "_3N.xml", aDate, aLatticePointsIm_3N, aSatellitePosition_3N, aLatticeECEF_3N);
	ASTERXMLWrite(aOutDir + aNameFile + "_3B.xml", aDate, aLatticePointsIm_3B, aSatellitePosition_3B, aLatticeECEF_3B);



	ASTERGT2MM_Banniere();
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
