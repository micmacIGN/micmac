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

MicMa cis an open source software specialized in image matching
for research in geographic information. MicMac is built on the
eLiSe image library. MicMac is governed by the  "Cecill-B licence".
See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include <algorithm>
#include "hassan/reechantillonnage.h"
#include "RPC.h"
#include "ALGLIB/interpolation.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"

using namespace alglib;
void function_sumof6sins_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
	// this callback calculates sum of 6 sins
	// where x is a position on X-axis and c is adjustable parameters
	func = c[0] * sin(2 * M_PI*c[1]  * x[0] + c[2])  + c[3]  * sin(2 * M_PI*c[4]  * x[0] + c[5])  + c[6]  * sin(2 * M_PI*c[7]  * x[0] + c[8])
		 + c[9] * sin(2 * M_PI*c[10] * x[0] + c[11]) + c[12] * sin(2 * M_PI*c[13] * x[0] + c[14]) + c[15] * sin(2 * M_PI*c[16] * x[0] + c[17]);
}

void function_sumof8sins_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
	// this callback calculates sum of 8 sins
	// where x is a position on X-axis and c is adjustable parameters
	func = c[0] * sin(2 * M_PI*c[1] * x[0] + c[2]) + c[3] * sin(2 * M_PI*c[4] * x[0] + c[5]) + c[6] * sin(2 * M_PI*c[7] * x[0] + c[8])
		 + c[9] * sin(2 * M_PI*c[10] * x[0] + c[11]) + c[12] * sin(2 * M_PI*c[13] * x[0] + c[14]) + c[15] * sin(2 * M_PI*c[16] * x[0] + c[17])
		 + c[18] * sin(2 * M_PI*c[19] * x[0] + c[20]) + c[21] * sin(2 * M_PI*c[22] * x[0] + c[23]);
}

void function_sumof9sins_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
	// this callback calculates sum of 9 sins
	// where x is a position on X-axis and c is adjustable parameters
	func = c[0] * sin(2 * M_PI*c[1] * x[0] + c[2]) + c[3] * sin(2 * M_PI*c[4] * x[0] + c[5]) + c[6] * sin(2 * M_PI*c[7] * x[0] + c[8])
		+ c[9] * sin(2 * M_PI*c[10] * x[0] + c[11]) + c[12] * sin(2 * M_PI*c[13] * x[0] + c[14]) + c[15] * sin(2 * M_PI*c[16] * x[0] + c[17])
		+ c[18] * sin(2 * M_PI*c[19] * x[0] + c[20]) + c[21] * sin(2 * M_PI*c[22] * x[0] + c[23]) + c[24] * sin(2 * M_PI*c[25] * x[0] + c[26]);
}

void function_sumof12sins_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
	// this callback calculates sum of 12 sins
	// where x is a position on X-axis and c is adjustable parameters
	func = c[0] * sin(2 * M_PI*c[1] * x[0] + c[2]) + c[3] * sin(2 * M_PI*c[4] * x[0] + c[5]) + c[6] * sin(2 * M_PI*c[7] * x[0] + c[8])
		+ c[9] * sin(2 * M_PI*c[10] * x[0] + c[11]) + c[12] * sin(2 * M_PI*c[13] * x[0] + c[14]) + c[15] * sin(2 * M_PI*c[16] * x[0] + c[17])
		+ c[18] * sin(2 * M_PI*c[19] * x[0] + c[20]) + c[21] * sin(2 * M_PI*c[22] * x[0] + c[23]) + c[24] * sin(2 * M_PI*c[25] * x[0] + c[26])
		+ c[27] * sin(2 * M_PI*c[28] * x[0] + c[29]) + c[30] * sin(2 * M_PI*c[31] * x[0] + c[32]) + c[33] * sin(2 * M_PI*c[34] * x[0] + c[35]);
}

/*
//DFT implementation from http://stackoverflow.com/questions/12251453/implementation-of-the-discrete-fourier-transform-fft
#include <complex>
typedef std::complex<double> Complex;
vector< complex<double> > DFT(vector< complex<double> >& theData)
{
	// Define the Size of the read in vector
	const int S = theData.size();

	// Initalise new vector with size of S
	vector< complex<double> > out(S, 0);
	for (unsigned i = 0; (i < S); i++)
	{
		out[i] = complex<double>(0.0, 0.0);
		for (unsigned j = 0; (j < S); j++)
		{
			out[i] += theData[j] * polar<double>(1.0, -2 * PI * i * j / S);
		}
	}

	return out;
}
//End DFT implementation
*/


Im2D_REAL8 FitASTERv2(REAL8 ** aParOrig, string aDir, Pt2di aSz, bool writeFit, string aSceneName)
{
	// timers
	clock_t beginTimer = clock();
	clock_t overallBeginTimer = clock();
	clock_t endTimer;
	double elapsed_secs;


	//Reading the correlation file
	Tiff_Im aTFCorrel = Tiff_Im::StdConvGen(aDir + "GeoI-Px/Correl_Geom-Im_Num_15.tif", 1, false);
	Im2D_REAL8  aCorrel(aSz.x, aSz.y);
	Im2D_REAL8  aCorrel_dilat(aSz.x, aSz.y);
	ELISE_COPY
	(
		aTFCorrel.all_pts(),
		aTFCorrel.in(),
		aCorrel.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	ELISE_COPY
	(
		aTFCorrel.all_pts(),
		aTFCorrel.in(),
		aCorrel_dilat.out()
	);
	REAL8 ** aDatCorrel = aCorrel.data();
	REAL8 ** aDatCorrel_dilat = aCorrel_dilat.data();
	//double aMinCorrel, aMaxCorrel;
	//aCorrel.getMinMax(aMinCorrel, aMaxCorrel);
	//cout << "Min correl = " << aMinCorrel << endl;
	//cout << "Max correl = " << aMaxCorrel << endl;

	// Filter correlation
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			if (aDatCorrel[aY][aX] / 255 < 0.80)
			{
				aDatCorrel[aY][aX] = -9999;
				aDatCorrel_dilat[aY][aX] = -9999;
			}
		}
	}

	//Dilate mask

	int pad = 10;
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			//if on the edge of the image
			if (aX<pad || aX>aSz.x - pad || aY<pad || aY>aSz.y - pad)
			{
				aDatCorrel_dilat[aY][aX] = -9999;
			}
			else
			{
				//Checking pad*pad neighbours (dilation by pad*pad square Structuring element)
				bool aLoop = true;
				for (int adX = aX - pad; adX <= (aX + pad) && aLoop; adX++)
				{
					for (int adY = aY - pad; adY <= (aY + pad) && aLoop; adY++)
					{
						if (aDatCorrel[adY][adX] == -9999)
						{
							aDatCorrel_dilat[aY][aX] = -9999;
							//cout << aDatCorrel_dilat[aY][aX] << " " << aDatCorrel[aY][aX] << endl;
							aLoop = false;
						}
					}
				}
			}
		}
	}


	// Computing mean and number of point conserved
	int countTot = 0;
	int countAccept = 0;
	double aMeanParErr = 0;
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			countTot++;
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
				aMeanParErr = aMeanParErr + aParOrig[aY][aX];
				countAccept++;
			}
		}
	}

	aMeanParErr = aMeanParErr / double(countAccept);

	std::cout << "Rejected : " << countTot - countAccept << " points" << endl;
	std::cout << "Accepted : " << countAccept << " points" << endl;
	std::cout << "Total    : " << countTot << " points" << endl;
	std::cout << "Mean Val : " << aMeanParErr << endl << endl;

	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Input data filtered in " << elapsed_secs << "s" << endl << endl;


	////////////////////////////////////////////////////// 
	//            Read 3N and 3B geometries             //
	//////////////////////////////////////////////////////

	string aSceneInName = aSceneName + "_3N";
	string aSceneOutName = aSceneName + "_3B";

	//Reading the image
	Tiff_Im aTF_In = Tiff_Im::StdConvGen(aDir + aSceneInName + ".tif", 1, false);
	Pt2di aSz_In = aTF_In.sz(); std::cout << "size of image In = " << aSz_In << endl;

	Tiff_Im aTF_Out = Tiff_Im::StdConvGen(aDir + aSceneOutName + ".tif", 1, false);
	Pt2di aSz_Out = aTF_Out.sz(); std::cout << "size of image Out = " << aSz_Out << endl;

	// Loading the GRID files
	std::string aGRIBinInName = aDir + aSceneInName + ".GRIBin";
	std::string aGRIBinOutName = aDir + aSceneOutName + ".GRIBin";
	ElAffin2D oriIntImaM2C_In, oriIntImaM2C_Out;
	Pt2di Sz(100000, 100000);
	ElCamera* mCameraIn = new cCameraModuleOrientation(new OrientationGrille(aGRIBinInName), Sz, oriIntImaM2C_In);
	ElCamera* mCameraOut = new cCameraModuleOrientation(new OrientationGrille(aGRIBinOutName), Sz, oriIntImaM2C_Out);

	// Define area of interest
	Pt2dr aTopLeftCornerOf3N(0, 0);
	Pt2dr aBottomRightCornerOf3N(4100, 4200);
	Pt2dr aTopLeftCornerOf3Nin3B = mCameraOut->Ter2Capteur(mCameraIn->ImEtZ2Terrain(aTopLeftCornerOf3N, 0));
	Pt2dr aBottomRightCornerOf3Nin3B = mCameraOut->Ter2Capteur(mCameraIn->ImEtZ2Terrain(aBottomRightCornerOf3N, 0));

	size_t aXminAOI = max(0, (int)(aTopLeftCornerOf3Nin3B.x));
	size_t aXmaxAOI = min(aSz_Out.x, (int)(aBottomRightCornerOf3Nin3B.x));
	size_t aYminAOI = max(0, (int)(aTopLeftCornerOf3Nin3B.y));
	size_t aYmaxAOI = min(aSz_Out.y, (int)(aBottomRightCornerOf3Nin3B.y));
	std::cout << "Area of interest for computations of columns and rows of 3N in 3B band geometry: " << aXminAOI << " " << aXmaxAOI << " " << aYminAOI << " " << aYmaxAOI << " " << endl;

	////////////////////////////////////////////////////// 
	//       Project columns and rows of 3N in 3B       //
	//////////////////////////////////////////////////////

	// Output map of columns of 3N in 3B
	Im2D_REAL8  aColMap(aSz_Out.x, aSz_Out.y);
	REAL8 ** aData_ColMap = aColMap.data();
	// in integers to limit number of possible different values, hopefully not damaging precision of fit
	//Im2D_INT4  aColMap(aSz_out.x, aSz_out.y);
	//INT4 ** aData_ColMap = aColMap.data();

	// Output map of rows of 3N in 3B
	//Im2D_REAL8  aRowMap(aSz_Out.x, aSz_Out.y);
	//REAL8 ** aData_RowMap = aRowMap.data();
	// in integers to limit number of possible different values, hopefully not damaging precision of fit
	Im2D_INT4  aRowMap(aSz_Out.x, aSz_Out.y);
	INT4 ** aData_RowMap = aRowMap.data();

	std::cout << "Starting computations of columns and rows of 3N in 3B band geometry......";
	beginTimer = clock();

	// Find projected row value for area of interest
	for (size_t aX = aXminAOI; aX < aXmaxAOI; aX++)
	{
		for (size_t aY = aYminAOI; aY < aYmaxAOI; aY++)
		{
			// define points in image 3B
			Pt2dr aPtImOut(aX, aY);

			//Compute image coordinate of points in 3N with GRIBin
			Pt2dr aPtImIn = mCameraIn->Ter2Capteur(mCameraOut->ImEtZ2Terrain(aPtImOut, 0));

			// Record value of columns and rows
			aData_ColMap[aY][aX] = aPtImIn.x;
			aData_RowMap[aY][aX] = aPtImIn.y;
		}
	}
	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Done in " << elapsed_secs << "s" << endl;

	////////////////////////////////////////////////////// 
	//        Export projected rows and columns         //
	//////////////////////////////////////////////////////

	if (false) {
		// Export data Col
		string aNameOut = aDir + "GeoI-Px/ColFrom_" + aSceneInName + "_to_" + aSceneOutName + ".tif";
		std::cout << "Exporting Columns data to : " << aNameOut << "......";
		beginTimer = clock();

		Tiff_Im  aOutCol
		(
			aNameOut.c_str(),
			aSz_Out,
			//GenIm::int4,
			GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aOutCol.all_pts(),
			aColMap.in(),
			aOutCol.out()
		);

		endTimer = clock();
		elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
		std::cout << "Done in " << elapsed_secs << "s" << endl;

		// Export data Row
		aNameOut = aDir + "GeoI-Px/RowFrom_" + aSceneInName + "_to_" + aSceneOutName + ".tif";
		std::cout << "Exporting Rows data to : " << aNameOut << "......";
		beginTimer = clock();

		Tiff_Im  aOutRow
		(
			aNameOut.c_str(),
			aSz_Out,
			GenIm::int4,
			//GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aOutRow.all_pts(),
			aRowMap.in(),
			aOutRow.out()
		);

		endTimer = clock();
		elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
		std::cout << "Done in " << elapsed_secs << "s" << endl;

	}


	////////////////////////////////////////////////////// 
	//          Polynomial fit (5th deg in X)           //
	//////////////////////////////////////////////////////

	std::cout << "Solving polynomial fit (6deg)......";
	beginTimer = clock();
	//degX6
	L2SysSurResol aSysPar6(6);
	//For all points that are not nullified by bad correlation (value=9999) add equation to fit 5th degree polynomials in x to measured paralax
	for (size_t aX = aXminAOI; aX < aXmaxAOI; aX++)
	{
		for (size_t aY = aYminAOI; aY < aYmaxAOI; aY++)
		{
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
				double X = aData_ColMap[aY][aX]; // Columns of 3N in 3B
				//deg5X
				double aEq6[6] = { X, X*X, X*X*X, X*X*X*X, X*X*X*X*X , X*X*X*X*X*X };
				aSysPar6.AddEquation(1, aEq6, aParOrig[aY][aX] - aMeanParErr);
			}
		}
	}

	//Computing the result
	int aCase;
	bool Ok;
	int nbCoef;

	Im1D_REAL8 aSolPar6 = aSysPar6.GSSR_Solve(&Ok);
	double aRes6 = aSysPar6.ResiduOfSol(aSolPar6.data());
	double* aPolyPar = aSolPar6.data(); nbCoef = 6;

	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Done in " << elapsed_secs << "s" << endl << endl;


	std::cout << "Polynomial fit parameters :" << endl
		<< "Cst   = " << aMeanParErr << endl
		<< "Coef = ";
	for (u_int i = 0; i < nbCoef; i++)
	{
		std::cout << aPolyPar[i] << " ";
	}
	std::cout << endl;
	std::cout << "Residual for Poly6 : " << aRes6 << endl << endl;



	//Creating out container
	Im2D_REAL8  aParFit(aSz.x, aSz.y);
	REAL8 ** aDatParFit = aParFit.data();

	//Filling out container
	for (u_int aX = 0; aX < aSz.x; aX++) {
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double X = aData_ColMap[aY][aX]; // Columns of 3N in 3B
			aDatParFit[aY][aX] = aMeanParErr + aPolyPar[0] * X + aPolyPar[1] * X*X + aPolyPar[2] * X*X*X + aPolyPar[3] * X*X*X*X + aPolyPar[4] * X*X*X*X*X + aPolyPar[5] * X*X*X*X*X*X;

		}
	}

	//Output the Fitted paralax file (polynomials)
	if (writeFit) {
		std::cout << "Writing file for 'polynomial' fit......";
		beginTimer = clock();
		string aNameOut = "GeoI-Px/Px2_Num16_DeZoom1_Geom-Im_adjMM1.tif";
		Tiff_Im  aTparralaxFitOut
		(
			aNameOut.c_str(),
			aSz,
			GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aTparralaxFitOut.all_pts(),
			aParFit.in(),
			aTparralaxFitOut.out()
		);

		endTimer = clock();
		elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
		std::cout << "Done in " << elapsed_secs << "s" << endl;
	}

	////////////////////////////////////////////////////// 
	//       Polynomial fit (7th deg in X and Y)        //
	//////////////////////////////////////////////////////
/* 

	std::cout << "Solving polynomial fit (7x7)......";
	beginTimer = clock();
	//deg7*7
	L2SysSurResol aSysPar77(35);
	//For all points that are not nullified by bad correlation (value=9999) add equation to fit 6th degree polynomials in x and y to measured paralax
	for (u_int aX = 0; aX < aSz.x; aX++) {
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
				//deg77
				double aEq77[35] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
					X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
					X*X*X*X*X, X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y, Y*Y*Y*Y*Y,
					X*X*X*X*X*X, X*X*X*X*X*Y, X*X*X*X*Y*Y, X*X*X*Y*Y*Y,  X*X*Y*Y*Y*Y, X*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y ,
					X*X*X*X*X*X*X, X*X*X*X*X*X*Y, X*X*X*X*X*Y*Y, X*X*X*X*Y*Y*Y,  X*X*X*Y*Y*Y*Y, X*X*Y*Y*Y*Y*Y, X*Y*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y*Y };
				aSysPar77.AddEquation(1, aEq77, aParOrig[aY][aX] - aMeanParErr);
			}
		}
	}

	//Computing the result
	int aCase;
	bool Ok;
	int nbCoef;

	Im1D_REAL8 aSolPar77 = aSysPar77.GSSR_Solve(&Ok);
	double aRes77 = aSysPar77.ResiduOfSol(aSolPar77.data());
	double* aPolyPar = aSolPar77.data(); aCase = 77; nbCoef = 35;

	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Done in " << elapsed_secs << "s" << endl << endl;


	std::cout << "Polynomial fit parameters :" << endl
		<< "Cst   = " << aMeanParErr << endl
		<< "Coef = ";
	for (u_int i = 0; i < nbCoef; i++)
	{
		std::cout << aPolyPar[i] << " ";
	}
	std::cout << endl;
	std::cout << "Residual for Poly77 : " << aRes77 << endl << endl;



	//Creating out container
	Im2D_REAL8  aParFit(aSz.x, aSz.y);
	REAL8 ** aDatParFit = aParFit.data();

	//Filling out container
	for (u_int aX = 0; aX < aSz.x; aX++) {
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);

			aDatParFit[aY][aX] = aMeanParErr + aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
				+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
				+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
				+ aPolyPar[14] * X*X*X*X*X + aPolyPar[15] * X*X*X*X*Y + aPolyPar[16] * X*X*X*Y*Y + aPolyPar[17] * X*X*Y*Y*Y + aPolyPar[18] * X*Y*Y*Y*Y + aPolyPar[19] * Y*Y*Y*Y*Y
				+ aPolyPar[20] * X*X*X*X*X*X + aPolyPar[21] * X*X*X*X*X*Y + aPolyPar[22] * X*X*X*X*Y*Y + aPolyPar[23] * X*X*X*Y*Y*Y + aPolyPar[24] * X*X*Y*Y*Y*Y + aPolyPar[25] * X*Y*Y*Y*Y*Y + aPolyPar[26] * Y*Y*Y*Y*Y*Y
				+ aPolyPar[27] * X*X*X*X*X*X*X + aPolyPar[28] * X*X*X*X*X*X*Y + aPolyPar[29] * X*X*X*X*X*Y*Y + aPolyPar[30] * X*X*X*X*Y*Y*Y + aPolyPar[31] * X*X*X*Y*Y*Y*Y + aPolyPar[32] * X*X*Y*Y*Y*Y*Y + aPolyPar[33] * X*Y*Y*Y*Y*Y*Y + aPolyPar[34] * Y*Y*Y*Y*Y*Y*Y;

		}
	}

	//Output the Fitted paralax file (polynomials)
	if (writeFit) {
		std::cout << "Writing file for 'polynomial' fit......";
		beginTimer = clock();
		string aNameOut = "GeoI-Px/Px2_Num16_DeZoom1_Geom-Im_adjMM1.tif";
		Tiff_Im  aTparralaxFitOut
		(
			aNameOut.c_str(),
			aSz,
			GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aTparralaxFitOut.all_pts(),
			aParFit.in(),
			aTparralaxFitOut.out()
		);

		endTimer = clock();
		elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
		std::cout << "Done in " << elapsed_secs << "s" << endl;
	}
*/


	////////////////////////////////////////////////////// 
	//             SIN FITTING                          //
	//////////////////////////////////////////////////////

	//Creating out container
	Im2D_REAL8  aParFit2(aSz.x, aSz.y);
	REAL8 ** aDatParFit2 = aParFit2.data();


	std::cout << "Making the problem 1D for ALGLIB......";
	beginTimer = clock();

	// 2D->1D signal
	vector< double > a1DSignalX;
	vector< double > a1DSignalY;
	vector< int > a1DSignalCounter;

	// Go through image 3B
	std::vector<double>::iterator foundValue;
	int indexOfValue;

	for (size_t aX = aXminAOI; aX < aXmaxAOI; aX++)
	{
		for (size_t aY = aYminAOI; aY < aYmaxAOI; aY++)
		{
			// If in quality mask
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
				foundValue = find(a1DSignalX.begin(), a1DSignalX.end(), aData_RowMap[aY][aX]);
				indexOfValue = std::distance(a1DSignalX.begin(), foundValue);
				if (foundValue == a1DSignalX.end()) // row value is not in a1DSignalX already
				{
					// Add new value to signal vectors (row value in X, signal value in Y)
					a1DSignalX.push_back(aData_RowMap[aY][aX]);
					a1DSignalY.push_back(aParOrig[aY][aX] - aDatParFit[aY][aX]);
					// Add counter for number of 2D points included in this 1D datapoint
					a1DSignalCounter.push_back(1);
				}
				else //row value already in a1DSignalX, at index indexOfValue
				{
					// Add new value to average of existing element of signal vectors with same row (X) value
					a1DSignalY[indexOfValue] = (a1DSignalY[indexOfValue] * a1DSignalCounter[indexOfValue] + (aParOrig[aY][aX] - aDatParFit[aY][aX])) / (a1DSignalCounter[indexOfValue] + 1);
					// Increment counter for number of 2D points included in this 1D datapoint
					a1DSignalCounter[indexOfValue]++;
				}
			}
		}
	}

	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Done in " << elapsed_secs << "s" << endl;

	std::cout << "Size of vector (number of data points) for sin fit " << a1DSignalY.size() << endl << endl;
	//cout << "Vector for sin fit " << a1DSignalY << endl;

	///Sloving using ALGLIB

	std::cout << "Solving sum of sins......";
	beginTimer = clock();

	alglib::real_2d_array AX;
	alglib::real_1d_array AY;
	AX.setcontent(a1DSignalX.size(), 1, &(a1DSignalX[0]));
	AY.setcontent(a1DSignalY.size(), &(a1DSignalY[0]));

	//initialization of parameters
	// c Define the parameters 
	// c=[Amp1,Freq1,Phase1,Amp2,.....,Phase9]
	// s Define scale to have all variables in same approximate range mean(Amp)~0.02, mean(Freq)~0.005, mean(Phase)~pi
	// s=[Amp1_Range,Freq1_Range,Phase1_Range,Amp2_Range,.....,Phase9_Range]
	// Bounds are defined by bndl (lower bounds) and bndu (upper bounds)



	// 9 sins
	// 3 sets of sins with phasse 0, 2*pi/3 and 4*pi/3 at (1) freq 0.0033 (307pix=4.6km) amplitude 0.05, (2) freq 0.0077 (~1300pix~=19.5km), and (3) freq 0.00044 (2266pix~=34km) amplitude 0.2
	real_1d_array c = "[0.05,0.0033,0,  0.05,0.0033,2.1,  0.05,0.0033,4.2, 0.05,0.00077,0,  0.05,0.00077,2.1,  0.05,0.00077,4.2, 0.2,0.00044,0,  0.2,0.00044,2.1,  0.2,0.00044,4.2]";
	real_1d_array s = "[0.1,0.0033,3.14, 0.1,0.0033,3.14, 0.1,0.0033,3.14, 0.1,0.00077,3.14, 0.1,0.00077,3.14, 0.1,0.00077,3.14, 0.1,0.00044,3.14, 0.1,0.00044,3.14, 0.1,0.00044,3.14]";
	real_1d_array bndl = "[0.0,0.0020,0.00, 0.0,0.0020,0.00, 0.0,0.0020,0.00, 0.0,0.0003,0.00, 0.0,0.0003,0.00, 0.0,0.0003,0.00, 0.0,0.0001,0.00, 0.0,0.0001,0.00, 0.0,0.0001,0.00]";
	real_1d_array bndu = "[0.3,0.0045,6.29, 0.3,0.0045,6.29, 0.3,0.0045,6.29, 0.3,0.0015,6.29, 0.3,0.0015,6.29, 0.3,0.0015,6.29, 0.3,0.0010,6.29, 0.3,0.0010,6.29, 0.3,0.0010,6.29]";

	// 2 sets of sins with phasse 0, 2*pi/3 and 4*pi/3 at (1) freq 0.0033 (307pix=4.6km) amplitude 0.05 and (2) freq 0.00044 (2266pix~=34km) amplitude 0.2
	// 6 sins
	//real_1d_array c = "[0.1,0.0033,0,  0.1,0.0033,2.1,  0.1,0.0033,4.2, 0.2,0.00044,0,  0.2,0.00044,2.1,  0.2,0.00044,4.2]";
	//real_1d_array s = "[0.1,0.0033,3.14, 0.1,0.0033,3.14, 0.1,0.0033,3.14, 0.2,0.00044,3.14, 0.2,0.00044,3.14, 0.2,0.00044,3.14]";
	//real_1d_array bndl = "[0,0.002,0, 0,0.002,0, 0,0.002,0, 0,0.0002,0, 0,0.0002,0, 0,0.0002,0]";
	//real_1d_array bndu = "[0.2,0.0045,6.29, 0.2,0.0045,6.29, 0.2,0.0045,6.29, 0.3,0.001,6.29, 0.3,0.001,6.29, 0.3,0.001,6.29]";

	double epsf = 0;
	double epsx = 0.0000001;
	ae_int_t maxits = 0;
	ae_int_t info;
	lsfitstate state;
	lsfitreport rep;
	double diffstep = 0.00001;

	//
	// Fitting without weights
	//
	lsfitcreatef(AX, AY, c, diffstep, state);
	lsfitsetcond(state, epsf, epsx, maxits);
	lsfitsetbc(state, bndl, bndu);
	lsfitsetscale(state, s);
	alglib::lsfitfit(state, function_sumof9sins_func);
	//alglib::lsfitfit(state, function_sumof6sins_func);
	lsfitresults(state, info, c, rep);
	//printf("%s\n", c.tostring(5).c_str());


	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Done in " << elapsed_secs << "s" << endl<< endl;

	cout << "Fitted sins (Amplitude,Frequency,Phase):" << endl;
	for (u_int i = 0; i < 9; i++)
	{
		cout << c[3 * i] << " ; " << c[3 * i + 1] << " ; " << c[3 * i + 2] << endl;
	}
	///END Solve with ALGLIB

	// Computing value for each point in 3B
	cout << "Computing value of sum of sins for each point in 3B......";
	beginTimer = clock();

	// Computing Sum of Sins value for each different row value
	vector<double> aValuesOfSinFunction;
	for (size_t anIndex=0 ; anIndex < a1DSignalX.size(); anIndex++)
	{
		double aVal = 0;
		for (u_int i = 0; i < 9; i++)
		{
			aVal += c[3 * i] * sin(2 * M_PI * c[3 * i + 1] * a1DSignalX[anIndex] + c[3 * i + 2]);
		}
		aValuesOfSinFunction.push_back(aVal);
	}
		
	//Filling out container
	for (u_int aX = 0; aX < aSz.x; aX++)
	{
		for (u_int aY = 0; aY < aSz.y; aY++)
		{
			foundValue = find(a1DSignalX.begin(), a1DSignalX.end(), aData_RowMap[aY][aX]);
			indexOfValue = std::distance(a1DSignalX.begin(), foundValue);

			aDatParFit2[aY][aX] = aDatParFit[aY][aX] + aValuesOfSinFunction[indexOfValue];

		}
	}
	endTimer = clock();
	elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
	std::cout << "Done in " << elapsed_secs << "s" << endl;



	//Output the Fitted paralax file (polynomials + sin)
	if (writeFit) {
		cout << "Writing file for 'polynomial and sum of sins' fit......";
		beginTimer = clock();
		string aNameOut2 = "GeoI-Px/Px2_Num16_DeZoom1_Geom-Im_adjMM2.tif";
		Tiff_Im  aTparralaxFit2Out
		(
			aNameOut2.c_str(),
			aSz,
			GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aTparralaxFit2Out.all_pts(),
			aParFit2.in(),
			aTparralaxFit2Out.out()
		);
		endTimer = clock();
		elapsed_secs = double(endTimer - beginTimer) / CLOCKS_PER_SEC;
		std::cout << "Done in " << elapsed_secs << "s" << endl;
	}

	elapsed_secs = double(endTimer - overallBeginTimer) / CLOCKS_PER_SEC;
	std::cout << endl << "ASTER filter v2 done in " << elapsed_secs << "s" << endl;
	return aParFit2;
}



Im2D_REAL8 FitASTERv1(REAL8 ** aParOrig, string aDir, Pt2di aSz, bool writeFit)
{
	//Reading the correlation file
	Tiff_Im aTFCorrel = Tiff_Im::StdConvGen(aDir + "GeoI-Px/Correl_Geom-Im_Num_15.tif", 1, false);
	Im2D_REAL8  aCorrel(aSz.x, aSz.y);
	Im2D_REAL8  aCorrel_dilat(aSz.x, aSz.y);
	ELISE_COPY
	(
		aTFCorrel.all_pts(),
		aTFCorrel.in(),
		aCorrel.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	ELISE_COPY
	(
		aTFCorrel.all_pts(),
		aTFCorrel.in(),
		aCorrel_dilat.out()
	);
	REAL8 ** aDatCorrel = aCorrel.data();
	REAL8 ** aDatCorrel_dilat = aCorrel_dilat.data();
	//double aMinCorrel, aMaxCorrel;
	//aCorrel.getMinMax(aMinCorrel, aMaxCorrel);
	//cout << "Min correl = " << aMinCorrel << endl;
	//cout << "Max correl = " << aMaxCorrel << endl;

	// Filter correlation
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			if (aDatCorrel[aY][aX] / 255 < 0.80)
			{
				aDatCorrel[aY][aX] = -9999;
				aDatCorrel_dilat[aY][aX] = -9999;
			}
		}
	}

	//Dilate mask

	int pad = 10;
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			//if on the edge of the image
			if (aX<pad || aX>aSz.x - pad || aY<pad || aY>aSz.y - pad)
			{
				aDatCorrel_dilat[aY][aX] = -9999;
			}
			else
			{
				//Checking pad*pad neighbours (dilation by pad*pad square Structuring element)
				bool aLoop = true;
				for (int adX = aX - pad; adX <= (aX + pad) && aLoop; adX++)
				{
					for (int adY = aY - pad; adY <= (aY + pad) && aLoop; adY++)
					{
						if (aDatCorrel[adY][adX] == -9999)
						{
							aDatCorrel_dilat[aY][aX] = -9999;
							//cout << aDatCorrel_dilat[aY][aX] << " " << aDatCorrel[aY][aX] << endl;
							aLoop = false;
						}
					}
				}
			}
		}
	}


	// Computing mean and number of point conserved
	int countTot = 0;
	int countAccept = 0;
	double aMeanParErr = 0;
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			countTot++;
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
				aMeanParErr = aMeanParErr + aParOrig[aY][aX];
				countAccept++;
			}
		}
	}

	aMeanParErr = aMeanParErr / double(countAccept);

	cout << "Rejected : " << countTot - countAccept << " points" << endl;
	cout << "Accepted : " << countAccept << " points" << endl;
	cout << "Total    : " << countTot << " points" << endl;
	cout << "Mean Val : " << aMeanParErr << endl;

	/*
	//deg4*4
	L2SysSurResol aSysPar44(14);
	//deg4*5
	L2SysSurResol aSysPar45(19);
	//deg5*4
	L2SysSurResol aSysPar54(19);
	//deg5*5
	L2SysSurResol aSysPar55(20);
	//deg6*6
	L2SysSurResol aSysPar66(27);
	*/
	//deg7*7
	L2SysSurResol aSysPar77(35);
	//For all points that are not nullified by bad correlation (value=9999) add equation to fit 6th degree polynomials in x and y to measured paralax
	for (u_int aX = 0; aX < aSz.x; aX++) {
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
				/*
				//deg44
				double aEq44[14] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
									 X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y };
				//deg45
				double aEq45[19] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
									 X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
									 X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y, Y*Y*Y*Y*Y };
				//deg54
				double aEq54[19] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
									 X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
									 X*X*X*X*X, X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y };
				//deg55
				double aEq55[20] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
					X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
					X*X*X*X*X, X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y, Y*Y*Y*Y*Y };
				//deg66
				double aEq66[27] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
					X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
					X*X*X*X*X, X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y, Y*Y*Y*Y*Y,
					X*X*X*X*X*X, X*X*X*X*X*Y, X*X*X*X*Y*Y, X*X*X*Y*Y*Y,  X*X*Y*Y*Y*Y, X*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y };
					*/
					//deg77
				double aEq77[35] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
					X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
					X*X*X*X*X, X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y, Y*Y*Y*Y*Y,
					X*X*X*X*X*X, X*X*X*X*X*Y, X*X*X*X*Y*Y, X*X*X*Y*Y*Y,  X*X*Y*Y*Y*Y, X*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y ,
					X*X*X*X*X*X*X, X*X*X*X*X*X*Y, X*X*X*X*X*Y*Y, X*X*X*X*Y*Y*Y,  X*X*X*Y*Y*Y*Y, X*X*Y*Y*Y*Y*Y, X*Y*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y*Y };
				/*
				aSysPar44.AddEquation(1, aEq44, aParOrig[aY][aX] - aMeanParErr);
				aSysPar45.AddEquation(1, aEq45, aParOrig[aY][aX] - aMeanParErr);
				aSysPar54.AddEquation(1, aEq54, aParOrig[aY][aX] - aMeanParErr);
				aSysPar55.AddEquation(1, aEq55, aParOrig[aY][aX] - aMeanParErr);
				aSysPar66.AddEquation(1, aEq66, aParOrig[aY][aX] - aMeanParErr);
				*/
				aSysPar77.AddEquation(1, aEq77, aParOrig[aY][aX] - aMeanParErr);
			}
		}
	}

	//Computing the result
	int aCase;
	bool Ok;
	int nbCoef;
	/*
	Im1D_REAL8 aSolPar44 = aSysPar44.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar45 = aSysPar45.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar54 = aSysPar54.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar55 = aSysPar55.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar66 = aSysPar66.GSSR_Solve(&Ok);
	double aRes44 = aSysPar55.ResiduOfSol(aSolPar44.data());
	double aRes45 = aSysPar55.ResiduOfSol(aSolPar45.data());
	double aRes54 = aSysPar55.ResiduOfSol(aSolPar54.data());
	double aRes55 = aSysPar55.ResiduOfSol(aSolPar55.data());
	double aRes66 = aSysPar66.ResiduOfSol(aSolPar66.data());
	cout << "Residual for Poly44 : " << aRes44 << endl;
	cout << "Residual for Poly45 : " << aRes45 << endl;
	cout << "Residual for Poly54 : " << aRes54 << endl;
	cout << "Residual for Poly55 : " << aRes55 << endl;
	cout << "Residual for Poly66 : " << aRes66 << endl;
	*/
	Im1D_REAL8 aSolPar77 = aSysPar77.GSSR_Solve(&Ok);
	double aRes77 = aSysPar77.ResiduOfSol(aSolPar77.data());
	cout << "Residual for Poly77 : " << aRes77 << endl;
	double* aPolyPar = aSolPar77.data(); aCase = 77; nbCoef = 35;
	/*
	double* aPolyPar;
	if (aRes44 < aRes45 && aRes44 < aRes54 && aRes44 < aRes55 && aRes44 < aRes66 && aRes44 < aRes77) { aPolyPar = aSolPar44.data(); aCase = 44; nbCoef = 14; };
	if (aRes45 < aRes44 && aRes45 < aRes54 && aRes45 < aRes55 && aRes45 < aRes66 && aRes45 < aRes77) { aPolyPar = aSolPar45.data(); aCase = 45; nbCoef = 19; };
	if (aRes54 < aRes45 && aRes54 < aRes44 && aRes54 < aRes55 && aRes54 < aRes66 && aRes54 < aRes77) { aPolyPar = aSolPar54.data(); aCase = 54; nbCoef = 19; };
	if (aRes55 < aRes45 && aRes55 < aRes54 && aRes55 < aRes44 && aRes55 < aRes66 && aRes55 < aRes77) { aPolyPar = aSolPar55.data(); aCase = 55; nbCoef = 20; };
	if (aRes66 < aRes45 && aRes66 < aRes54 && aRes66 < aRes55 && aRes66 < aRes44 && aRes66 < aRes77) { aPolyPar = aSolPar66.data(); aCase = 66; nbCoef = 27; };
	if (aRes77 < aRes45 && aRes77 < aRes54 && aRes77 < aRes55 && aRes77 < aRes66 && aRes77 < aRes44) { aPolyPar = aSolPar77.data(); aCase = 77; nbCoef = 35; };
	*/
	cout << "Polynomial fit (Poly" << aCase << ")" << endl
		<< "Cst   = " << aMeanParErr << endl
		<< "Coef = ";
	for (u_int i = 0; i < nbCoef; i++)
	{
		cout << aPolyPar[i] << " ";
	}
	cout << endl;


	//Creating out container
	Im2D_REAL8  aParFit(aSz.x, aSz.y);
	REAL8 ** aDatParFit = aParFit.data();

	//Filling out container
	for (u_int aX = 0; aX < aSz.x; aX++) {
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);
			double aVal;
			/*
			if (aCase == 44)
			{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y;
			}
			if (aCase == 45)
			{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
					+ aPolyPar[14] * X*X*X*X*Y + aPolyPar[15] * X*X*X*Y*Y + aPolyPar[16] * X*X*Y*Y*Y + aPolyPar[17] * X*Y*Y*Y*Y + aPolyPar[18] * Y*Y*Y*Y*Y;
			}
			if (aCase == 54)
			{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
					+ aPolyPar[14] * X*X*X*X*X + aPolyPar[15] * X*X*X*X*Y + aPolyPar[16] * X*X*X*Y*Y + aPolyPar[17] * X*X*Y*Y*Y + aPolyPar[18] * X*Y*Y*Y*Y;
			}
			if (aCase == 55)
			{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
					+ aPolyPar[14] * X*X*X*X*X + aPolyPar[15] * X*X*X*X*Y + aPolyPar[16] * X*X*X*Y*Y + aPolyPar[17] * X*X*Y*Y*Y + aPolyPar[18] * X*Y*Y*Y*Y + aPolyPar[19] * Y*Y*Y*Y*Y;
			}
			if (aCase == 66)
			{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
					+ aPolyPar[14] * X*X*X*X*X + aPolyPar[15] * X*X*X*X*Y + aPolyPar[16] * X*X*X*Y*Y + aPolyPar[17] * X*X*Y*Y*Y + aPolyPar[18] * X*Y*Y*Y*Y + aPolyPar[19] * Y*Y*Y*Y*Y
					+ aPolyPar[20] * X*X*X*X*X*X + aPolyPar[21] * X*X*X*X*X*Y + aPolyPar[22] * X*X*X*X*Y*Y + aPolyPar[23] * X*X*X*Y*Y*Y + aPolyPar[24] * X*X*Y*Y*Y*Y + aPolyPar[25] * X*Y*Y*Y*Y*Y + aPolyPar[26] * Y*Y*Y*Y*Y*Y;
			}*/
			//if (aCase == 77)
			//{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
					+ aPolyPar[14] * X*X*X*X*X + aPolyPar[15] * X*X*X*X*Y + aPolyPar[16] * X*X*X*Y*Y + aPolyPar[17] * X*X*Y*Y*Y + aPolyPar[18] * X*Y*Y*Y*Y + aPolyPar[19] * Y*Y*Y*Y*Y
					+ aPolyPar[20] * X*X*X*X*X*X + aPolyPar[21] * X*X*X*X*X*Y + aPolyPar[22] * X*X*X*X*Y*Y + aPolyPar[23] * X*X*X*Y*Y*Y + aPolyPar[24] * X*X*Y*Y*Y*Y + aPolyPar[25] * X*Y*Y*Y*Y*Y + aPolyPar[26] * Y*Y*Y*Y*Y*Y
					+ aPolyPar[27] * X*X*X*X*X*X*X + aPolyPar[28] * X*X*X*X*X*X*Y + aPolyPar[29] * X*X*X*X*X*Y*Y + aPolyPar[30] * X*X*X*X*Y*Y*Y + aPolyPar[31] * X*X*X*Y*Y*Y*Y + aPolyPar[32] * X*X*Y*Y*Y*Y*Y + aPolyPar[33] * X*Y*Y*Y*Y*Y*Y + aPolyPar[34] * Y*Y*Y*Y*Y*Y*Y;
			//}

			aDatParFit[aY][aX] = aMeanParErr + aVal;


		}
	}

	//Output the Fitted paralax file (polynomials)
	if (writeFit) {
		cout << "Writing file for polynomial fit" << endl;
		string aNameOut = "GeoI-Px/Px2_Num16_DeZoom1_Geom-Im_adjMM1.tif";
		Tiff_Im  aTparralaxFitOut
		(
			aNameOut.c_str(),
			aSz,
			GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aTparralaxFitOut.all_pts(),
			aParFit.in(),
			aTparralaxFitOut.out()
		);
	}




	///START OF SIN FITTING
	//Creating out container
	Im2D_REAL8  aParFit2(aSz.x, aSz.y);
	REAL8 ** aDatParFit2 = aParFit2.data();

	//Creating sub-container for median computation
	Im2D_REAL8  aParFit2_1(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_2(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_3(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_4(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_5(aSz.x, aSz.y);
	REAL8 ** aDatParFit2_1 = aParFit2_1.data();
	REAL8 ** aDatParFit2_2 = aParFit2_2.data();
	REAL8 ** aDatParFit2_3 = aParFit2_3.data();
	REAL8 ** aDatParFit2_4 = aParFit2_4.data();
	REAL8 ** aDatParFit2_5 = aParFit2_5.data();
	Im2D_REAL8  aParFit2_6(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_7(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_8(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_9(aSz.x, aSz.y);
	Im2D_REAL8  aParFit2_X(aSz.x, aSz.y);
	REAL8 ** aDatParFit2_6 = aParFit2_6.data();
	REAL8 ** aDatParFit2_7 = aParFit2_7.data();
	REAL8 ** aDatParFit2_8 = aParFit2_8.data();
	REAL8 ** aDatParFit2_9 = aParFit2_9.data();
	REAL8 ** aDatParFit2_X = aParFit2_X.data();


	//500pix, overlap (5) 400pix
	//vector<u_int> bands = { 0,0,0,0,0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,
	//	3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5000,5000,5000,5000 };
	//1000pix, overlap (5) 800pix
	//vector<u_int> bands = { 0,0,0,0,0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5000,5000,5000,5000 };
	//1000pix overlap (10) 900pix	
	u_int bandsa[] = { 0,0,0,0,0,0,0,0,0,0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,
		3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000 };
	vector<u_int> bands(bandsa, bandsa + 69);
	int aNbQualityThreshold = 32;
	int nbOverlap = 10;
	//For each band of 500pix with 80% overlap
	for (u_int k = 0; k < bands.size() - nbOverlap; k++) {
		u_int aXmin = bands[k];
		u_int aXmax = bands[k + nbOverlap];
		cout << endl << "Sin fit pass " << k << endl;
		cout << "From X=" << aXmin << " to " << aXmax << endl;
		//Fit sin in y axis (init? aFreq ~= 0.0033, aAmplitude ~= 0.15)

		//FFT to refine frequency (aFreq is around 0.0033)

		// 2D->1D
		int firstValid = 0;
		int lastValid = 0;
		vector<double> a1DSignal;
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double aSum = 0;
			int aCpt = 0;
			for (u_int aX = aXmin; aX < aXmax; aX++) {
				if (aDatCorrel_dilat[aY][aX] != -9999)
				{
					aCpt++;
					aSum += aParOrig[aY][aX] - aDatParFit[aY][aX];
				}
			}
			// If enough data to be reliable
			if (aCpt > aNbQualityThreshold) {
				if (firstValid == 0) { firstValid = aY; }
				aSum = aSum / double(aCpt);
				a1DSignal.push_back(aSum);
				lastValid = aY;
			}
			else {
				a1DSignal.push_back(-9999);
			}
		}

		if (lastValid == 0)
		{
			cout << "Not enough data to fit this slice, keeping polynomial solution" << endl;

#pragma omp parallel for
			for (u_int aX = aXmin; aX < aXmax; aX++) {
				for (u_int aY = 0; aY < aSz.y; aY++) {
					if ((mod(k, nbOverlap)) == 0)
					{
						aDatParFit2_1[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 1)
					{
						aDatParFit2_2[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 2)
					{
						aDatParFit2_3[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 3)
					{
						aDatParFit2_4[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 4)
					{
						aDatParFit2_5[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 5)
					{
						aDatParFit2_6[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 6)
					{
						aDatParFit2_7[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 7)
					{
						aDatParFit2_8[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 8)
					{
						aDatParFit2_9[aY][aX] = aDatParFit[aY][aX];
					}
					if ((mod(k, nbOverlap)) == 9)
					{
						aDatParFit2_X[aY][aX] = aDatParFit[aY][aX];
					}
				}
			}
		}
		else {


			//cout << "Compression to 1D of band finished" << endl;
			//cout << "First Valid = " << firstValid << endl;
			//cout << "Last Valid = " << lastValid << endl;

			//interpolate -9999 values to something that makes sense and Converting to complex type
//			vector< complex<double> > a1DSignalC;
			vector< double > a1DSignalX;
			vector< double > a1DSignalY;

			for (u_int i = firstValid; i <= lastValid; i++)
			{
				if (a1DSignal[i] == -9999) {
					//closest non -9999 value is just before (i-1)
					double aBefore = a1DSignal[i - 1];
					double aAfter = 0;
					int aDist;
					for (u_int j = i + 1; j < lastValid; j++)
					{
						if (a1DSignal[j] != -9999)
						{
							double aAfter = a1DSignal[j];
							aDist = j - i;
						}
					}
					//interpolate
					a1DSignal[i] = (aBefore + aAfter / double(aDist)) / (1 + 1 / double(aDist));
				}
				//Convert to complex type
				//a1DSignalC.push_back(a1DSignal[i]);
				a1DSignalX.push_back(i);
				a1DSignalY.push_back(a1DSignal[i]);
			}
			cout << "Size of vector for sin fit " << a1DSignalY.size() << endl;
			//cout << "Vector for sin fit " << a1DSignalY << endl;



			///Sloving using ALGLIB
			alglib::real_2d_array AX;
			alglib::real_1d_array AY;
			AX.setcontent(a1DSignalX.size(), 1, &(a1DSignalX[0]));
			AY.setcontent(a1DSignalY.size(), &(a1DSignalY[0]));

			//initialization of parameters
			//c=[Amp1,Freq1,Phase1,Amp2,.....,Phase8]
			//real_1d_array c = "[0.05,0.0033,0,  0.05,0.0033,0.3,  0.05,0.0033,0.6,  0.05,0.002,1,  0.05,0.001,0,  0.05,0.00077,0.3,  0.05,0.00077,0.6,  0.05,0.00077,1]";
			real_1d_array c = "[0.05,0.0033,0,  0.05,0.0033,0.003,  0.05,0.0033,0.006,  0.05,0.002,0.01,  0.05,0.001,0,  0.05,0.00077,0.003,  0.05,0.00077,0.006,  0.05,0.00077,0.01]";
			//real_1d_array c = "[2.5,1.65,0, 2.5,1.65,0.3, 2.5,1.65,0.6, 2.5,1,1, 2.5,0.5,0, 2.5,0.385,0.3, 2.5,0.385,0.6, 2.5,0.385,1]";
			//"[0.05,0.002,0,  0.05,0.0033,0.3,  0.05,0.004,0.6,  0.05,0.006,1,  0.005,0.02,0,  0.05,0.0033,0.3,  0.05,0.0033,0.6,  0.05,0.00077,1]";
			real_1d_array s = "[50, 500, 1]";//scale to have all variables in same approximate range mean(Amp)~0.02, mean(Freq)~0.005, mean(Phase)~pi/2
			real_1d_array bndl = "[0, 0.0005, 0,0, 0.0005, 0,0, 0.0005, 0,0, 0.0005, 0,0, 0.0005, 0,0, 0.0005, 0,0, 0.0005, 0,0, 0.0005, 0]";
			real_1d_array bndu = "[0.1, 0.02, 0.031415,0.1, 0.02, 0.031415,0.1, 0.02, 0.031415,0.1, 0.02, 0.031415,0.1, 0.02, 0.031415,0.1, 0.02, 0.031415,0.1, 0.02, 0.031415,0.1, 0.02, 0.031415]";
			double epsf = 0;
			double epsx = 0.0000001;
			ae_int_t maxits = 0;
			ae_int_t info;
			lsfitstate state;
			lsfitreport rep;
			double diffstep = 0.00001;

			//
			// Fitting without weights
			//
			lsfitcreatef(AX, AY, c, diffstep, state);
			lsfitsetcond(state, epsf, epsx, maxits);
			lsfitsetbc(state, bndl, bndu);
			//lsfitsetscale(state, s);
			alglib::lsfitfit(state, function_sumof8sins_func);
			lsfitresults(state, info, c, rep);
			printf("%s\n", c.tostring(5).c_str());


			///END Solve with ALGLIB


			//Filling out container
#pragma omp parallel for
			for (u_int aX = aXmin; aX < aXmax; aX++)
			{
				for (u_int aY = 0; aY < aSz.y; aY++)
				{
					double aVal = 0;
					for (u_int i = 0; i < 9; i++)
					{
						aVal += c[3 * i] * sin(2 * M_PI * c[3 * i + 1] * aY + c[3 * i + 2] * 100);
					}

					if ((mod(k, nbOverlap)) == 0)
					{
						aDatParFit2_1[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 1)
					{
						aDatParFit2_2[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 2)
					{
						aDatParFit2_3[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 3)
					{
						aDatParFit2_4[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 4)
					{
						aDatParFit2_5[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 5)
					{
						aDatParFit2_6[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 6)
					{
						aDatParFit2_7[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 7)
					{
						aDatParFit2_8[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 8)
					{
						aDatParFit2_9[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
					else if ((mod(k, nbOverlap)) == 9)
					{
						aDatParFit2_X[aY][aX] = aDatParFit2[aY][aX] + (aDatParFit[aY][aX] + aVal);
					}
				}
			}


		}


	}

	if (nbOverlap == 2)
	{
		for (u_int aX = 0; aX < aSz.x; aX++) {
			for (u_int aY = 0; aY < aSz.y; aY++) {
				//filter crazy values
				if (abs(aDatParFit2_1[aY][aX] - aMeanParErr) > 0.5 && abs(aDatParFit2_2[aY][aX] - aMeanParErr) > 0.5)
				{
					aDatParFit2[aY][aX] = aDatParFit[aY][aX];
				}
				else if (abs(aDatParFit2_2[aY][aX] - aMeanParErr) > 0.5)
				{
					aDatParFit2[aY][aX] = aDatParFit2_1[aY][aX];
				}
				else if (abs(aDatParFit2_2[aY][aX] - aMeanParErr) > 0.5)
				{
					aDatParFit2[aY][aX] = aDatParFit2_2[aY][aX];
				}
				else
				{
					aDatParFit2[aY][aX] = (aDatParFit2_1[aY][aX] + aDatParFit2_2[aY][aX]) / 2.0;
				}
			}
		}
	}

	//Finding median in stack
	if (nbOverlap == 5)
	{
		for (u_int aX = 0; aX < aSz.x; aX++) {
			for (u_int aY = 0; aY < aSz.y; aY++) {

				double aStacka[] = { aDatParFit2_1[aY][aX],  aDatParFit2_2[aY][aX],  aDatParFit2_3[aY][aX],  aDatParFit2_4[aY][aX],  aDatParFit2_5[aY][aX] };
				vector<double> aStack(aStacka, aStacka + 5);
				std::sort(aStack.begin(), aStack.end());//sort the stack
														//filter crazy values
				if (abs(aStack[2] - aMeanParErr) > 2)
				{
					aDatParFit2[aY][aX] = aDatParFit[aY][aX];
				}
				else if (abs(aStack[1] - aMeanParErr) > 2 || abs(aStack[3] - aMeanParErr) > 2)
				{
					aDatParFit2[aY][aX] = aStack[2];//median of stack (third element of list of 5 elements);}
				}
				else
				{
					aDatParFit2[aY][aX] = (aStack[1] + aStack[2] + aStack[3]) / 3.0;//mean of middle of stack (2-3-4 out of 5, removing extremes);
				}
			}
		}
	}
	//Finding median in stack
	if (nbOverlap == 10)
	{
#pragma omp parallel for
		for (u_int aX = 0; aX < aSz.x; aX++) {
			for (u_int aY = 0; aY < aSz.y; aY++) {

				double aStacka[] = { aDatParFit2_1[aY][aX],  aDatParFit2_2[aY][aX],  aDatParFit2_3[aY][aX],  aDatParFit2_4[aY][aX],  aDatParFit2_5[aY][aX],
										aDatParFit2_6[aY][aX],  aDatParFit2_7[aY][aX],  aDatParFit2_8[aY][aX],  aDatParFit2_9[aY][aX],  aDatParFit2_X[aY][aX] };
				vector<double> aStack(aStacka, aStacka + 10);

				std::sort(aStack.begin(), aStack.end());//sort the stack
														//filter crazy values
				if (abs(aStack[4] - aMeanParErr) > 2)
				{
					aDatParFit2[aY][aX] = aDatParFit[aY][aX];
				}
				else if (abs(aStack[3] - aMeanParErr) > 2 || abs(aStack[5] - aMeanParErr) > 2)
				{
					aDatParFit2[aY][aX] = aStack[4];//median of stack (third element of list of 5 elements);}
				}
				else
				{
					aDatParFit2[aY][aX] = (aStack[3] + aStack[4] + aStack[5]) / 3.0;//mean of middle of stack (4-5-6 out of 10removing extremes);
				}
			}
		}
	}

	//Output the Fitted paralax file (polynomials + sin)
	if (writeFit) {
		cout << "Writing file for polynomial and sum of sins fit" << endl;
		string aNameOut2 = "GeoI-Px/Px2_Num16_DeZoom1_Geom-Im_adjMM2.tif";
		Tiff_Im  aTparralaxFit2Out
		(
			aNameOut2.c_str(),
			aSz,
			GenIm::real8,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero
		);

		ELISE_COPY
		(
			aTparralaxFit2Out.all_pts(),
			aParFit2.in(),
			aTparralaxFit2Out.out()
		);
	}
	return aParFit2;
}


int ApplyParralaxCor_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aNameIm, aNameParallax, aASTERSceneName;
	std::string aNameOut = "";
	int aFitASTER = 0;
	bool  exportFitASTER = false;
	//Reading the arguments
	ElInitArgMain
	(
		argc, argv,
		LArgMain()
		<< EAMC(aNameIm, "Image to be corrected", eSAM_IsPatFile)
		<< EAMC(aNameParallax, "Paralax correction file", eSAM_IsPatFile),
		LArgMain()
		<< EAM(aNameOut, "Out", true, "Name of output image (Def=ImName_corrected.tif)")
		<< EAM(aFitASTER, "FitASTER", true, "Fit functions appropriate for ASTER L1A processing (input '1' or '2' : version number)")
		<< EAM(exportFitASTER, "ExportFitASTER", true, "Export grid from FitASTER (Def=false)")
		<< EAM(aASTERSceneName, "ASTERSceneName", true, "ASTER L1A Scene name (Only for and MANDATORY for FitASTERv2)")
	);

	std::string aDir, aPatIm;
	SplitDirAndFile(aDir, aPatIm, aNameIm);

	cout << "Correcting " << aNameIm << endl;
	if (aNameOut == "")
		aNameOut = aNameIm + "_corrected.tif";

	//Reading the image and creating the objects to be manipulated
	Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameIm, 1, false);

	Pt2di aSz = aTF.sz(); cout << "size of image = " << aSz << endl;
	Im2D_U_INT1  aIm(aSz.x, aSz.y);

	ELISE_COPY
	(
		aTF.all_pts(),
		aTF.in(),
		aIm.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);

	U_INT1 ** aData = aIm.data();

	//Reading the parallax correction file
	Tiff_Im aTFPar = Tiff_Im::StdConvGen(aDir + aNameParallax, 1, false);
	Im2D_REAL8  aPar(aSz.x, aSz.y);
	ELISE_COPY
	(
		aTFPar.all_pts(),
		aTFPar.in(),
		aPar.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	REAL8 ** aDatPar = aPar.data();

	if (aFitASTER == 1)
	{
		Im2D_REAL8 aParFit = FitASTERv1(aDatPar, aDir, aSz, exportFitASTER);
		ELISE_COPY
		(
			aTFPar.all_pts(),
			aParFit.in(),
			aPar.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
		);
		REAL8 ** aDatPar = aPar.data();
		cout << "Data fitted" << endl;
	}
	else if (aFitASTER == 2)
	{
		Im2D_REAL8 aParFit = FitASTERv2(aDatPar, aDir, aSz, exportFitASTER, aASTERSceneName);
		ELISE_COPY
		(
			aTFPar.all_pts(),
			aParFit.in(),
			aPar.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
		);
		REAL8 ** aDatPar = aPar.data();
		cout << "Data fitted" << endl;
	}

	//Output container
	Im2D_U_INT1  aImOut(aSz.x, aSz.y);
	U_INT1 ** aDataOut = aImOut.data();

	//Computing output data
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			Pt2dr ptOut;
			ptOut.x = aX - aDatPar[aY][aX];// *cos(aAngleB);
			ptOut.y = aY;// -aDatPar[aY][aX] * sin(aAngleB);
			aDataOut[aY][aX] = Reechantillonnage::biline(aData, aSz.x, aSz.y, ptOut);
		}
	}



	Tiff_Im  aTOut
	(
		aNameOut.c_str(),
		aSz,
		GenIm::u_int1,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);


	ELISE_COPY
	(
		aTOut.all_pts(),
		aImOut.in(),
		aTOut.out()
	);
	
	cout << "Image corrected" << endl;
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
