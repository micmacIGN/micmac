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


//Important note:
//pt.x is either the column in image space or the longitude in geographic coordinates or the easting  in projected coordinates
//pt.y is either the row    in image space or the latitude  in geographic coordinates or the northing in projected coordinates

Pt2dr TFW_IJ2XY(Pt2dr aIJ, vector<double> aTFW)
{
	Pt2dr aXY;
	aXY.x = aTFW[4] + aIJ.x * aTFW[0] + aIJ.y * aTFW[2];
	aXY.y = aTFW[5] + aIJ.x * aTFW[1] + aIJ.y * aTFW[3];
	return aXY;
}

//(aXY.x - aTFW[4] - aIJ.y * aTFW[2])/ aTFW[0] = aIJ.x ;
//aXY.y - aTFW[5] - (aXY.x - aTFW[4] - aIJ.y * aTFW[2])  * aTFW[1] / (aTFW[0]* aTFW[3]) =  aIJ.y ;


Pt2dr TFW_XY2IJ(Pt2dr aXY, vector<double> aTFW)
{
	Pt2dr aIJ; //[0.01, 0, 0, -0.01, 0, 30]
	aIJ.y = (aXY.y - aTFW[5] - aTFW[1] * (aXY.x - aTFW[4] - aIJ.y * aTFW[2]) / aTFW[0]) / aTFW[3];
	aIJ.x = (aXY.x - aTFW[4] - aIJ.y * aTFW[2]) / aTFW[0];
	return aIJ;
}

bool isInside(Pt2di aBorder, Pt2dr aPt)
{
	return (aPt.x > 0, aPt.y > 0, aPt.x < aBorder.x, aPt.y < aBorder.y);
}

vector<Pt3dr> ComputedZFromDEMAndMask(REAL8** aDEMINData, vector<double> aTFWin, Pt2di aSzIN, string aDEMRefPath, string aMaskPath)
{

	std::cout << "Using Reference DEM : " << aDEMRefPath << endl;
	std::cout << "With Mask           : " << aMaskPath << endl;


	// Read Mask
	string aDir, aMaskName;
	SplitDirAndFile(aDir, aMaskName, aMaskPath);
	Tiff_Im aTF_Mask = Tiff_Im::StdConvGen(aDir + aMaskName, 1, false);
	Pt2di aSz_Mask = aTF_Mask.sz(); cout << "Size of Mask = " << aSz_Mask << endl;
	Im2D_U_INT1  aMask(aSz_Mask.x, aSz_Mask.y);
	ELISE_COPY
	(
		aTF_Mask.all_pts(),
		aTF_Mask.in(),
		aMask.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	U_INT1** aData_Mask = aMask.data();
	// Load Mask georeference data
	string aTFWName = aDir + aMaskName.substr(0, aMaskName.size() - 2) + "fw";
	std::fstream aTFWFile(aTFWName.c_str(), std::ios_base::in);
	double aRes_xEast, aRes_xNorth, aRes_yEast, aRes_yNorth, aCorner_East, aCorner_North;

	// Make sure the file stream is good.
	ELISE_ASSERT(aTFWFile, "Failed to open the mask .tfw file");

	aTFWFile >> aRes_xEast >> aRes_xNorth >> aRes_yEast >> aRes_yNorth >> aCorner_East >> aCorner_North;
	vector<double> aTFWMask = { aRes_xEast,aRes_xNorth,aRes_yEast,aRes_yNorth,aCorner_East,aCorner_North };

	// Load Reference DEM
	string aNameDEMRef;
	SplitDirAndFile(aDir, aNameDEMRef, aDEMRefPath);

	cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string>* aSetIm = aICNM->Get(aNameDEMRef);
	Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameDEMRef, 1, true);
	Pt2di aSzREF = aTF.sz();

	Im2D_REAL8  aDEMRef(aSzREF.x, aSzREF.y);

	ELISE_COPY
	(
		aTF.all_pts(),
		aTF.in(),
		aDEMRef.out()
	);

	REAL8** aDEMREFData = aDEMRef.data();

	// Load Ref DEM georeference data
	aTFWName = aDir + aNameDEMRef.substr(0, aNameDEMRef.size() - 2) + "fw";
	std::fstream aTFWFileRef(aTFWName.c_str(), std::ios_base::in);

	// Make sure the file stream is good.
	ELISE_ASSERT(aTFWFileRef, "Failed to open the reference DEM .tfw file");

	aTFWFileRef >> aRes_xEast >> aRes_xNorth >> aRes_yEast >> aRes_yNorth >> aCorner_East >> aCorner_North;
	vector<double> aTFWRef = { aRes_xEast,aRes_xNorth,aRes_yEast,aRes_yNorth,aCorner_East,aCorner_North };



	vector<Pt3dr> aListXYdZ;

	for (u_int i = 0; i < aSzIN.x; i++)
	{
		for (u_int j = 0; j < aSzIN.y; j++)
		{
			Pt2dr aPosIJ(i, j);
			// get the world coordinate of the current input DEM point
			Pt2dr aPosXY = TFW_IJ2XY(aPosIJ, aTFWin);
			// Get the mask image coordinate for that world coordinate
			Pt2dr aMaskIJ = TFW_XY2IJ(aPosXY, aTFWMask);
			// Get the DEMRef image coordinate for that world coordinate
			Pt2dr aREFIJ = TFW_XY2IJ(aPosXY, aTFWRef);
			// get DEMRef value for that point
			double aREFZ = Reechantillonnage::biline(aDEMREFData, aSzREF.x, aSzREF.y, aREFIJ);
			// if the mask is positive and both the input and ref DEM have data
			if (aDEMINData[j][i] > -9999 && aREFZ > -9999 && aData_Mask[int(aMaskIJ.y)][int(aMaskIJ.x)] == 1)
			{
				// Create point at XY with dZ between DEMin and DEMRef as Z.
				Pt3dr aPtdZ(aPosXY.x, aPosXY.y, aDEMINData[j][i] - aREFZ);
				aListXYdZ.push_back(aPtdZ);
			}

		}

	}

	return aListXYdZ;
}

vector<Pt3dr> ComputedZFromDEMAndXY(REAL8** aDEMINData, vector<double> aTFWin, Pt2di aSzIN, string aDEMRefPath, string aListPointsPath)
{

	std::cout << "Using Reference DEM : " << aDEMRefPath << endl;
	std::cout << "With XY points      : " << aListPointsPath << endl;

	// Load list of points
	vector<Pt2dr> aListXY;
	std::ifstream infile(aListPointsPath);
	double aX, aY;
	while (infile >> aX >> aY)
	{
		Pt2dr aPt(aX, aY);
		aListXY.push_back(aPt);
	}


	// Load Reference DEM
	string aDir, aNameDEMRef;
	SplitDirAndFile(aDir, aNameDEMRef, aDEMRefPath);

	cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string>* aSetIm = aICNM->Get(aNameDEMRef);
	Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameDEMRef, 1, true);
	Pt2di aSzREF = aTF.sz();

	Im2D_REAL8  aDEMRef(aSzREF.x, aSzREF.y);

	ELISE_COPY
	(
		aTF.all_pts(),
		aTF.in(),
		aDEMRef.out()
	);

	REAL8** aDEMREFData = aDEMRef.data();

	// Load Ref DEM georeference data
	string aTFWName = aDir + aNameDEMRef.substr(0, aNameDEMRef.size() - 2) + "fw";
	std::fstream aTFWFile(aTFWName.c_str(), std::ios_base::in);
	double aRes_xEast, aRes_xNorth, aRes_yEast, aRes_yNorth, aCorner_East, aCorner_North;

	// Make sure the file stream is good.
	ELISE_ASSERT(aTFWFile, "Failed to open the reference DEM .tfw file");

	aTFWFile >> aRes_xEast >> aRes_xNorth >> aRes_yEast >> aRes_yNorth >> aCorner_East >> aCorner_North;
	vector<double> aTFWRef = { aRes_xEast,aRes_xNorth,aRes_yEast,aRes_yNorth,aCorner_East,aCorner_North };



	vector<Pt3dr> aListXYdZ;

	// for each XY in aListXY
	for (u_int i = 0; i < aListXY.size(); i++)
	{
		// Get the DEMIn image coordinate for that world coordinate
		Pt2dr aINIJ = TFW_XY2IJ(aListXY[i], aTFWin);
		// Get DEMin value for that point
		double aINZ = Reechantillonnage::biline(aDEMINData, aSzIN.x, aSzIN.y, aINIJ);

		// Get the DEMRef image coordinate for that world coordinate
		Pt2dr aREFIJ = TFW_XY2IJ(aListXY[i], aTFWRef);
		// Get DEMRef value for that point
		double aREFZ = Reechantillonnage::biline(aDEMREFData, aSzREF.x, aSzREF.y, aREFIJ);

		// if the both the input and ref DEM have data at that point
		if (aINZ > -9999 && aREFZ > -9999 && isInside(aSzIN, aINIJ))
		{
			// Create point at XY with dZ between DEMin and DEMRef as Z.
			Pt3dr aPtdZ(aListXY[i].x, aListXY[i].y, aINZ - aREFZ);
			aListXYdZ.push_back(aPtdZ);
		}

	}


	return aListXYdZ;
}

vector<Pt3dr> ComputedZFromGCPs(REAL8** aDEMINData, vector<double> aTFWin, Pt2di aSzIN, string aListGCPsPath)
{

	std::cout << "Using GCPs : " << aListGCPsPath << endl;

	// Load list of points
	vector<Pt3dr> aListXYZ;

	std::ifstream infile(aListGCPsPath);
	double aX, aY, aZ;
	while (infile >> aX >> aY >> aZ)
	{
		Pt3dr aPt(aX, aY, aZ);
		aListXYZ.push_back(aPt);
	}

	vector<Pt3dr> aListXYdZ;

	// for each XY in aListXY
	for (u_int i = 0; i < aListXYZ.size(); i++)
	{
		Pt2dr aPtXY(aListXYZ[i].x, aListXYZ[i].y);
		// Get the DEMIn image coordinate for that world coordinate
		Pt2dr aINIJ = TFW_XY2IJ(aPtXY, aTFWin);
		// Get DEMin value for that point
		double aINZ = Reechantillonnage::biline(aDEMINData, aSzIN.x, aSzIN.y, aINIJ);
		// if the both the input and ref DEM have data at that point
		if (aINZ > -9999 && isInside(aSzIN, aINIJ))
		{
			// Create point at XY with dZ between DEMin and DEMRef as Z.
			Pt3dr aPtdZ(aListXYZ[i].x, aListXYZ[i].y, aINZ - aListXYZ[i].z);
			aListXYdZ.push_back(aPtdZ);
		}

	}
	return aListXYdZ;
}

vector<double> Banana_Solve(vector<Pt3dr> aListXYdZ, int aDeg)
{

	std::cout << "Solving banana with a degree " << aDeg << " polynome." << endl;

	int nbParam;
	vector<double> aPolyBanana = { 0,0,0, 0,0,0,0,0,0,0 };
	double aResidual;


	if (aDeg == 0) {
		nbParam = 1;
		//initialized to 0
		double aBanana[1] = { 0 };
		L2SysSurResol aSysBanana(1);
		for (u_int i = 0; i < aListXYdZ.size(); i++) {

			double X = aListXYdZ[i].x;
			double Y = aListXYdZ[i].y;

			double aPoly[1] = {
				1
			};
			aSysBanana.AddEquation(1, aPoly, aListXYdZ[i].z);
		}

		//Computing the result
		bool Ok;
		Im1D_REAL8 aSolBanana = aSysBanana.GSSR_Solve(&Ok);
		double* aDataBanana = aSolBanana.data();
		for (uint i = 0; i < nbParam; i++) { aPolyBanana[i]=aDataBanana[i]; }
		aResidual = aSysBanana.ResiduAfterSol();

	}

	if (aDeg == 1) {
		nbParam = 3;
		//initialized to 0
		double aBanana[3] = { 0, 0, 0 };
		L2SysSurResol aSysBanana(3);
		for (u_int i = 0; i < aListXYdZ.size(); i++) {

			double X = aListXYdZ[i].x;
			double Y = aListXYdZ[i].y;
			double aPoly[3] = {
				1,
				 X,  Y
			};aSysBanana.AddEquation(1, aPoly, aListXYdZ[i].z);
		}

		//Computing the result
		bool Ok;
		Im1D_REAL8 aSolBanana = aSysBanana.GSSR_Solve(&Ok);
		double* aDataBanana = aSolBanana.data();
		for (uint i = 0; i < nbParam; i++) { aPolyBanana[i] = aDataBanana[i]; }
		aResidual = aSysBanana.ResiduAfterSol();

	}

	if (aDeg == 2) {
		nbParam = 6;
		//initialized to 0
		double aBanana[6] = { 0, 0, 0, 0, 0, 0 };
		L2SysSurResol aSysBanana(6);
		for (u_int i = 0; i < aListXYdZ.size(); i++) {
			double X = aListXYdZ[i].x;
			double Y = aListXYdZ[i].y;
			double aPoly[6] = {
				1,
				 X,  Y,
				 X * Y,  X * X,  Y * Y
			};
			aSysBanana.AddEquation(1, aPoly, aListXYdZ[i].z);
		}

		//Computing the result
		bool Ok;
		Im1D_REAL8 aSolBanana = aSysBanana.GSSR_Solve(&Ok);
		double* aDataBanana = aSolBanana.data();
		for (uint i = 0; i < nbParam; i++) { aPolyBanana[i] = aDataBanana[i]; }
		aResidual = aSysBanana.ResiduAfterSol();
	}


	if (aDeg == 3) {
		nbParam = 10;
		//initialized to 0
		double aBanana[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		L2SysSurResol aSysBanana(10);
		for (u_int i = 0; i < aListXYdZ.size(); i++) {
			double X = aListXYdZ[i].x;
			double Y = aListXYdZ[i].y;
			double aPoly[10] = {
				1,
				 X,  Y,
				 X * Y,  X * X,  Y * Y,
				 X * X * X,  X * Y * Y,  Y * X * X,  Y * Y * Y
			};
			aSysBanana.AddEquation(1, aPoly, aListXYdZ[i].z);
		}

		//Computing the result
		bool Ok;
		Im1D_REAL8 aSolBanana = aSysBanana.GSSR_Solve(&Ok);
		double* aDataBanana = aSolBanana.data();
		for (uint i = 0; i < nbParam; i++) { aPolyBanana[i] = aDataBanana[i]; }
		aResidual = aSysBanana.ResiduAfterSol();
	}

	std::cout << "Solution = " << endl;
	std::cout << aPolyBanana << endl;
	std::cout << "Residual = " << aResidual << endl;
	return aPolyBanana;
}

void Banana_Apply(REAL8** aDEMINData, vector<double> aTFWin, Pt2di aSzIN, vector<double> aBananaPoly, string aNameOut)
{
	Tiff_Im aTF = Tiff_Im(aNameOut.c_str(), aSzIN, GenIm::real4, Tiff_Im::No_Compr, Tiff_Im::BlackIsZero);

	Im2D_REAL4  aDEM(aSzIN.x, aSzIN.y);

	ELISE_COPY
	(
		aTF.all_pts(),
		aTF.in(),
		aDEM.out()
	);

	REAL4** aData = aDEM.data();
	double aCor;
	for (int aY = 0; aY < aSzIN.y; aY++)
	{
		for (int aX = 0; aX < aSzIN.x; aX++)
		{
			Pt2dr aPtIm(aX, aY);
			Pt2dr aPtW = TFW_IJ2XY(aPtIm, aTFWin);
			aCor = aBananaPoly[0] + aPtW.x * aBananaPoly[1] + aPtW.y * aBananaPoly[2]
				+ aPtW.x * aPtW.y * aBananaPoly[3] + aPtW.x * aPtW.x * aBananaPoly[4] + aPtW.y * aPtW.y * aBananaPoly[5]
				+ aPtW.x * aPtW.x * aPtW.x * aBananaPoly[6] + aPtW.x * aPtW.y * aPtW.y * aBananaPoly[7] + aPtW.x * aPtW.x * aPtW.y * aBananaPoly[8] + aPtW.y * aPtW.y * aPtW.y * aBananaPoly[9];
			aData[aY][aX] = aDEMINData[aY][aX]+aCor;
		}
	}

	Tiff_Im  aTOut
	(
		aNameOut.c_str(),
		aSzIN,
		GenIm::real4,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);

	ELISE_COPY
	(
		aTOut.all_pts(),
		aDEM.in(),
		aTOut.out()
	);


}


int Banana_main(int argc, char** argv)
{
	std::string aDEMinPath, aDEMRefPath = "", aMaskPath = "", aListPointsPath = "", aListGCPsPath = "", aNameOut="";
	int aDeg = 2;
	ElInitArgMain
	(
		argc, argv,
		LArgMain() << EAMC(aDEMinPath, "Input DEM to be corrected - DEM must have tfw", eSAM_IsPatFile),
		LArgMain() << EAM(aDeg, "DegPoly", true, "Degree of fitted polynome ([0-3] default = 2)")
		<< EAM(aDEMRefPath, "DEMRef", true, "Reference DEM - DEM must have tfw", eSAM_IsPatFile)
		<< EAM(aMaskPath, "Mask", true, "A binary mask of stable terrain - if value=1 then the point is used, if =0 then unused (to be used with a reference DEM) - mask must have tfw", eSAM_IsPatFile)
		<< EAM(aListPointsPath, "ListPoints", true, "A text file of XY coordinates of stable points (to be used with a reference DEM)", eSAM_IsPatFile)
		<< EAM(aListGCPsPath, "ListGCPs", true, "A text file of XYZ coordinates of stable points (to be used without a reference DEM)", eSAM_IsPatFile)
		<< EAM(aNameOut, "Out", true, "Name of output file (Default : PAT_Corrected.tif)")
	);

	std::cout << "Correcting DEM : " << aDEMinPath << endl;
	// Load Input DEM
	string aDir, aNameDEMin;
	SplitDirAndFile(aDir, aNameDEMin, aDEMinPath);

	cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	const std::vector<std::string>* aSetIm = aICNM->Get(aNameDEMin);
	Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameDEMin, 1, true);
	Pt2di aSz = aTF.sz();

	Im2D_REAL8  aDEMin(aSz.x, aSz.y);

	ELISE_COPY
	(
		aTF.all_pts(),
		aTF.in(),
		aDEMin.out()
	);

	REAL8** aDEMINData = aDEMin.data();

	// Load input DEM georeference data
	string aTFWName = aDir + aNameDEMin.substr(0, aNameDEMin.size() - 2) + "fw";
	std::fstream aTFWFile(aTFWName.c_str(), std::ios_base::in);
	double aRes_xEast, aRes_xNorth, aRes_yEast, aRes_yNorth, aCorner_East, aCorner_North;

	// Make sure the file stream is good.
	ELISE_ASSERT(aTFWFile, "Failed to open the input DEM .tfw file");

	aTFWFile >> aRes_xEast >> aRes_xNorth >> aRes_yEast >> aRes_yNorth >> aCorner_East >> aCorner_North;
	vector<double> aTFW = { aRes_xEast,aRes_xNorth,aRes_yEast,aRes_yNorth,aCorner_East,aCorner_North };


	// Define outfile name
	if (aNameOut == "") { aNameOut = aDir + aNameDEMin.substr(0, aNameDEMin.size() - 4) + "_Corrected.tif"; }

	// Check what inputs are given
	//For each case, a list of XYdZ points (aListXYdZ) is generated.
	vector<Pt3dr> aListXYdZ;
	if (aDEMRefPath != "" && aMaskPath != "") { aListXYdZ = ComputedZFromDEMAndMask(aDEMINData, aTFW, aSz, aDEMRefPath, aMaskPath); }
	else if (aDEMRefPath != "" && aListPointsPath != "") { aListXYdZ = ComputedZFromDEMAndXY(aDEMINData, aTFW, aSz, aDEMRefPath, aListPointsPath); }
	else if (aListGCPsPath != "") { aListXYdZ = ComputedZFromGCPs(aDEMINData, aTFW, aSz, aListGCPsPath); }
	else { ELISE_ASSERT(false, "No valid combination of input given"); }

	//cout << "List of XYdZ:" << endl << aListXYdZ << endl;

	//Compute polynome that would fit that distribution of bias
	vector<double> aBananaPoly = Banana_Solve(aListXYdZ, aDeg);

	//Apply polynome to DEM and export it
	Banana_Apply(aDEMINData, aTFW, aSz, aBananaPoly, aNameOut);


	return EXIT_SUCCESS;
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
