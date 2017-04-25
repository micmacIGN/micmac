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
#include "ALGLIB\interpolation.h"

using namespace alglib;
void function_sunof8sins_func(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
	// this callback calculates sum of 8 sins
	// where x is a position on X-axis and c is adjustable parameters
	func = c[0] * sin(2 * M_PI*c[1] * x[0] + c[2] * 100) + c[3] * sin(2 * M_PI*c[4] * x[0] + c[5] * 100) + c[6] * sin(2 * M_PI*c[7] * x[0] + c[8] * 100) + c[9] * sin(2 * M_PI*c[10] * x[0] + c[11] * 100) +
		c[12] * sin(2 * M_PI*c[13] * x[0] + c[14] * 100) + c[15] * sin(2 * M_PI*c[16] * x[0] + c[17] * 100) + c[18] * sin(2 * M_PI*c[19] * x[0] + c[20] * 100) + c[21] * sin(2 * M_PI*c[22] * x[0] + c[23] * 100);
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

Im2D_REAL8 FitASTER(REAL8 ** aParOrig, string aDir, Pt2di aSz, bool writeFit)
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
	//deg7*7
	L2SysSurResol aSysPar77(35);
	//For all points that are not nullified by bad correlation (value=9999) add equation to fit 6th degree polynomials in x and y to measured paralax
	for (u_int aX = 0; aX < aSz.x; aX++) {
		for (u_int aY = 0; aY < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);
			if (aDatCorrel_dilat[aY][aX] != -9999)
			{
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
				//deg77
				double aEq77[35] = { X, Y, X*X, X*Y, Y*Y, X*X*X, X*X*Y, X*Y*Y, Y*Y*Y,
					X*X*X*X,  X*X*X*Y, X*X*Y*Y, X*Y*Y*Y, Y*Y*Y*Y,
					X*X*X*X*X, X*X*X*X*Y, X*X*X*Y*Y, X*X*Y*Y*Y,  X*Y*Y*Y*Y, Y*Y*Y*Y*Y,
					X*X*X*X*X*X, X*X*X*X*X*Y, X*X*X*X*Y*Y, X*X*X*Y*Y*Y,  X*X*Y*Y*Y*Y, X*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y ,
					X*X*X*X*X*X*X, X*X*X*X*X*X*Y, X*X*X*X*X*Y*Y, X*X*X*X*Y*Y*Y,  X*X*X*Y*Y*Y*Y, X*X*Y*Y*Y*Y*Y, X*Y*Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y*Y };

				aSysPar44.AddEquation(1, aEq44, aParOrig[aY][aX] - aMeanParErr);
				aSysPar45.AddEquation(1, aEq45, aParOrig[aY][aX] - aMeanParErr);
				aSysPar54.AddEquation(1, aEq54, aParOrig[aY][aX] - aMeanParErr);
				aSysPar55.AddEquation(1, aEq55, aParOrig[aY][aX] - aMeanParErr);
				aSysPar66.AddEquation(1, aEq66, aParOrig[aY][aX] - aMeanParErr);
				aSysPar77.AddEquation(1, aEq77, aParOrig[aY][aX] - aMeanParErr);
			}
		}
	}

	//Computing the result
	int aCase;
	bool Ok;
	int nbCoef;
	Im1D_REAL8 aSolPar44 = aSysPar44.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar45 = aSysPar45.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar54 = aSysPar54.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar55 = aSysPar55.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar66 = aSysPar66.GSSR_Solve(&Ok);
	Im1D_REAL8 aSolPar77 = aSysPar77.GSSR_Solve(&Ok);
	double aRes44 = aSysPar55.ResiduOfSol(aSolPar44.data());
	double aRes45 = aSysPar55.ResiduOfSol(aSolPar45.data());
	double aRes54 = aSysPar55.ResiduOfSol(aSolPar54.data());
	double aRes55 = aSysPar55.ResiduOfSol(aSolPar55.data());
	double aRes66 = aSysPar66.ResiduOfSol(aSolPar66.data());
	double aRes77 = aSysPar77.ResiduOfSol(aSolPar77.data());
	cout << "Residual for Poly44 : " << aRes44 << endl;
	cout << "Residual for Poly45 : " << aRes45 << endl;
	cout << "Residual for Poly54 : " << aRes54 << endl;
	cout << "Residual for Poly55 : " << aRes55 << endl;
	cout << "Residual for Poly66 : " << aRes66 << endl;
	cout << "Residual for Poly77 : " << aRes77 << endl;
	double* aPolyPar;
	if (aRes44 < aRes45 && aRes44 < aRes54 && aRes44 < aRes55 && aRes44 < aRes66 && aRes44 < aRes77) { aPolyPar = aSolPar44.data(); aCase = 44; nbCoef = 14; };
	if (aRes45 < aRes44 && aRes45 < aRes54 && aRes45 < aRes55 && aRes45 < aRes66 && aRes45 < aRes77) { aPolyPar = aSolPar45.data(); aCase = 45; nbCoef = 19; };
	if (aRes54 < aRes45 && aRes54 < aRes44 && aRes54 < aRes55 && aRes54 < aRes66 && aRes54 < aRes77) { aPolyPar = aSolPar54.data(); aCase = 54; nbCoef = 19; };
	if (aRes55 < aRes45 && aRes55 < aRes54 && aRes55 < aRes44 && aRes55 < aRes66 && aRes55 < aRes77) { aPolyPar = aSolPar55.data(); aCase = 55; nbCoef = 20; };
	if (aRes66 < aRes45 && aRes66 < aRes54 && aRes66 < aRes55 && aRes66 < aRes44 && aRes66 < aRes77) { aPolyPar = aSolPar66.data(); aCase = 66; nbCoef = 27; };
	if (aRes77 < aRes45 && aRes77 < aRes54 && aRes77 < aRes55 && aRes77 < aRes66 && aRes77 < aRes44) { aPolyPar = aSolPar77.data(); aCase = 77; nbCoef = 35; };

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
			}
			if (aCase == 77)
			{
				aVal = aPolyPar[0] * X + aPolyPar[1] * Y + aPolyPar[2] * X*X + aPolyPar[3] * X*Y + aPolyPar[4] * Y*Y
					+ aPolyPar[5] * X*X*X + aPolyPar[6] * X*X*Y + aPolyPar[7] * X*Y*Y + aPolyPar[8] * Y*Y*Y
					+ aPolyPar[9] * X*X*X*X + aPolyPar[10] * X*X*X*Y + aPolyPar[11] * X*X*Y*Y + aPolyPar[12] * X*Y*Y*Y + aPolyPar[13] * Y*Y*Y*Y
					+ aPolyPar[14] * X*X*X*X*X + aPolyPar[15] * X*X*X*X*Y + aPolyPar[16] * X*X*X*Y*Y + aPolyPar[17] * X*X*Y*Y*Y + aPolyPar[18] * X*Y*Y*Y*Y + aPolyPar[19] * Y*Y*Y*Y*Y
					+ aPolyPar[20] * X*X*X*X*X*X + aPolyPar[21] * X*X*X*X*X*Y + aPolyPar[22] * X*X*X*X*Y*Y + aPolyPar[23] * X*X*X*Y*Y*Y + aPolyPar[24] * X*X*Y*Y*Y*Y + aPolyPar[25] * X*Y*Y*Y*Y*Y + aPolyPar[26] * Y*Y*Y*Y*Y*Y
					+ aPolyPar[27] * X*X*X*X*X*X*X + aPolyPar[28] * X*X*X*X*X*X*Y + aPolyPar[29] * X*X*X*X*X*Y*Y + aPolyPar[30] * X*X*X*X*Y*Y*Y + aPolyPar[31] * X*X*X*Y*Y*Y*Y + aPolyPar[32] * X*X*Y*Y*Y*Y*Y + aPolyPar[33] * X*Y*Y*Y*Y*Y*Y + aPolyPar[34] * Y*Y*Y*Y*Y*Y*Y;
			}

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

	//250pix, overlap (5) 200pix
	//vector<u_int> bands = { 0,0,0,0,0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,
	//2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000,3050,3100,3150,3200,3250,3300,3350,3400,3450,3500,3550,3600,3650,3700,3750,3800,3850,3900,3950,4000,
	//4050,4100,4150,4200,4250,4300,4350,4400,4450,4500,4550,4600,4650,4700,4750,4800,4850,4900,4950,5000,5000,5000,5000,5000 };
	//500pix, overlap (5) 400pix
	//vector<u_int> bands = { 0,0,0,0,0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,
	//	3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5000,5000,5000,5000 };
	//500pix, overlap (2) 250pix
	//vector<u_int> bands = { 0,0,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5000 };
	//1000pix, overlap (5) 800pix
	//vector<u_int> bands = { 0,0,0,0,0,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5000,5000,5000,5000 };
	//1000pix overlap (10) 900pix
	vector<u_int> bands = { 0,0,0,0,0,0,0,0,0,0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,
		3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5000,5000,5000,5000,5000,5000,5000,5000,5000 };
	//vector<u_int> bands = { 0,0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5000 };
	//5000pix 5000pixoverlap
	//vector<u_int> bands = { 0,0,5000,5000 };
	int aNbQualityThreshold = 32;// double(countAccept) / (4200.0 * 2 * 50);//"Total acceptable data point"/("typical number of valid lines"*2*"portion of the image in a band (1/10th)")
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
			alglib::lsfitfit(state, function_sunof8sins_func);
			lsfitresults(state, info, c, rep);
			printf("%d\n", int(info));
			printf("%s\n", c.tostring(5).c_str());


			///END Solve with ALGLIB



			//REINSERT HERE
			//Filling out container
			for (u_int aX = aXmin; aX < aXmax; aX++)
			{
				for (u_int aY = 0; aY < aSz.y; aY++)
				{
					double aVal = 0;
					for (u_int i = 0; i < 8; i++)
					{
						aVal += c[3*i] * sin(2 * M_PI * c[3*i+1] * aY + c[3*i+2]*100);
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

				//vector<double> aStack = { aDatParFit2_1[aY][aX],  aDatParFit2_2[aY][aX],  aDatParFit2_3[aY][aX],  aDatParFit2_4[aY][aX],  aDatParFit2_5[aY][aX],
				//						aDatParFit2_6[aY][aX],  aDatParFit2_7[aY][aX],  aDatParFit2_8[aY][aX],  aDatParFit2_9[aY][aX],  aDatParFit2_X[aY][aX] };

				vector<double> aStack = { aDatParFit2_1[aY][aX],  aDatParFit2_2[aY][aX],  aDatParFit2_3[aY][aX],  aDatParFit2_4[aY][aX],  aDatParFit2_5[aY][aX] };
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
		for (u_int aX = 0; aX < aSz.x; aX++) {
			for (u_int aY = 0; aY < aSz.y; aY++) {

				vector<double> aStack = { aDatParFit2_1[aY][aX],  aDatParFit2_2[aY][aX],  aDatParFit2_3[aY][aX],  aDatParFit2_4[aY][aX],  aDatParFit2_5[aY][aX],
										aDatParFit2_6[aY][aX],  aDatParFit2_7[aY][aX],  aDatParFit2_8[aY][aX],  aDatParFit2_9[aY][aX],  aDatParFit2_X[aY][aX] };

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
	std::string aNameIm, aNameParallax;
	std::string aNameOut = "";
	bool aFitASTER = false;
	bool exportFitASTER = false;
	//Reading the arguments
	ElInitArgMain
		(
		argc, argv,
		LArgMain()
			<< EAMC(aNameIm, "Image to be corrected", eSAM_IsPatFile)
			<< EAMC(aNameParallax, "Paralax correction file", eSAM_IsPatFile),
		LArgMain()
			<< EAM(aNameOut, "Out", true, "Name of output image (Def=ImName_corrected.tif)")
			<< EAM(aFitASTER, "FitASTER", true, "EXPERIMENTAL Fit functions appropriate for ASTER L1A processing (Def=false)")
			<< EAM(exportFitASTER, "ExportFitASTER", true, "EXPERIMENTAL export grid from  fitASTER (Def=false)")
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
	
	if(aFitASTER)
	{
		Im2D_REAL8 aParFit = FitASTER(aDatPar,aDir, aSz, exportFitASTER);
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

	//Things needed for RPC angle computation, not main goal of this function
	/*
	//Read RPC 1 (B image)
	RPC aRPC;
	string aNameRPC1 = "RPC_" + StdPrefix(aNameIm) + ".xml";
	aRPC.ReadDimap(aNameRPC1);
	cout << "Dimap File " << aNameRPC1 << " read" << endl;

	
	//Output angle container 1 (B image)
	Im2D_REAL8  aAngleBOut(aSz.x, aSz.y);
	REAL8 ** aDataAngleBOut = aAngleBOut.data();
	string aNameAngleB = "AngleB.tif";
	*/
	/*
	//Reading the DEM file
	Tiff_Im aTFDEM = Tiff_Im::StdConvGen(aDir + aNameDEM, 1, false);
	Im2D_REAL8  aDEM(aSz.x, aSz.y);
	ELISE_COPY
	(
	aTFDEM.all_pts(),
	aTFDEM.in(),
	aDEM.out()
	);
	REAL8 ** aDatDEM = aDEM.data();

	//Read RPC 2 (N image)
	RPC aRPC2;
	string aNameRPC2 = "RPC_" + StdPrefix(aNameIm2) + ".xml";
	aRPC2.ReadDimap(aNameRPC2);
	cout << "Dimap File " << aNameRPC2 << " read" << endl;

	//Output angle container 2 (N image)
	Im2D_REAL8  aAngleNOut(aSz.x, aSz.y);
	REAL8 ** aDataAngleNOut = aAngleNOut.data();
	string aNameAngleN = "AngleN.tif";
	*/
	//Pt3dr PBTest(1500,3000, 0);
	//Pt3dr PWTest = aRPC.DirectRPC(PBTest);
	//Pt3dr PNTest = aRPC2.InverseRPC(PWTest);
	//cout << "PB0 = " << PBTest << endl;
	//cout << "PW0 = " << PWTest << endl;
	//cout << "PN0 = " << PNTest << endl;
	//cout << aRPC.height_scale << " " << aRPC.height_off << endl;
	//PBTest.z=1000;
	//PWTest = aRPC.DirectRPC(PBTest);
	//PNTest = aRPC2.InverseRPC(PWTest);
	//cout << "PB1 = " << PBTest << endl;
	//cout << "PW1 = " << PWTest << endl;
	//cout << "PN1 = " << PNTest << endl;


	//Computing output data
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			//cout << "X = " << aX << " Y = " << aY << endl;
			/*
			//Pt3dr P0B(aX, aY, aDatDEM[aY][aX]);
			Pt3dr P0B(aX, aY, 1000);
			Pt3dr PW0 = aRPC.DirectRPC(P0B);
			Pt3dr PW1 = PW0, PW2 = PW0;
			PW1.z = PW1.z - 100;
			PW2.z = PW2.z + 100;
			Pt3dr P1B = aRPC.InverseRPC(PW1);
			Pt3dr P2B = aRPC.InverseRPC(PW2);
			//Pt3dr P1N = aRPC2.InverseRPC(PW1);
			//Pt3dr P2N = aRPC2.InverseRPC(PW2);
			//Pt3dr P1B(aX, aY, 0);
			//Pt3dr P2B(aX, aY, 10000);
			//Pt3dr P1N = aRPC2.InverseRPC(aRPC.DirectRPC(P1B));
			//Pt3dr P2N = aRPC2.InverseRPC(aRPC.DirectRPC(P2B));
			double aAngleB = atan((P2B.x - P1B.x) / (P2B.y - P1B.y));
			aDataAngleBOut[aY][aX] = aAngleB;
			//double aAngleN = atan((P2N.x - P1N.x) / (P2N.y - P1N.y));
			//aDataAngleNOut[aY][aX] = aAngleN;
			//cout << aX << " " << aY << " " << aAngle << endl;
			//cout << P1N << " " << P2N << " " << aAngle << endl;

			*/
			//THE THINGS COMPUTED ABOVE WILL BE USED IN A FURTHER UPDATE

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
	/*
	Tiff_Im  aTAngleBOut
		(
		aNameAngleB.c_str(),
		aSz,
		GenIm::real8,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
		);


	ELISE_COPY
		(
		aTAngleBOut.all_pts(),
		aAngleBOut.in(),
		aTAngleBOut.out()
		);


	Tiff_Im  aTAngleNOut
		(
		aNameAngleN.c_str(),
		aSz,
		GenIm::real8,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
		);


	ELISE_COPY
		(
		aTAngleNOut.all_pts(),
		aAngleNOut.in(),
		aTAngleNOut.out()
		);
*/
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
