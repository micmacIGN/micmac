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

//FFT implementation
#include <complex>

typedef std::complex<double> Complex;

vector< complex<double> > DFT(vector< complex<double> >& theData)
{
	// Define the Size of the read in vector
	const int S = theData.size();

	// Initalise new vector with size of S
	vector< complex<double> > out(S, 0);
	for (unsigned i = 0; (int(i) < S); i++)
	{
		out[i] = complex<double>(0.0, 0.0);
		for (unsigned j = 0; (int(j) < S); j++)
		{
			out[i] += theData[j] * polar<double>(1.0, -2 * PI * i * j / S);
		}
	}

	return out;
}


Im2D_REAL8 FitASTER(REAL8 ** aParOrig, string aDir, Pt2di aSz)
{
	//Reading the correlation file
	Tiff_Im aTFCorrel = Tiff_Im::StdConvGen(aDir + "GeoI-Px/Correl_Geom-Im_Num_15.tif", 1, false);
	Im2D_REAL8  aCorrel(aSz.x, aSz.y);
	ELISE_COPY
	(
		aTFCorrel.all_pts(),
		aTFCorrel.in(),
		aCorrel.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	REAL8 ** aDatCorrel = aCorrel.data();
	double aMinCorrel, aMaxCorrel;
	aCorrel.getMinMax(aMinCorrel, aMaxCorrel);
	cout << "Min correl = " << aMinCorrel << endl;
	cout << "Max correl = " << aMaxCorrel << endl;
	int countTot = 0;
	int countRej = 0;
	int countAccept = 0;
	double aMeanParErr = 0;

	// Filter by correlation
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			countTot++;
			if (aDatCorrel[aY][aX]/aMaxCorrel<0.80)
			{
			aParOrig[aY][aX] = -9999;
			countRej++;
			}
			else
			{
				aMeanParErr = aMeanParErr + aParOrig[aY][aX];
				countAccept++;
			}
		}
	}
	aMeanParErr = aMeanParErr / double(countAccept);

	cout << "Rejected : " << countRej << " points" << endl;
	cout << "Accepted : " << countAccept << " points" << endl;
	cout << "Total    : " << countTot << " points" << endl;
	cout << "Mean Val : " << aMeanParErr << endl;

	L2SysSurResol aSysPar(12);
	//For all points that are not nullified by bad correlation (value=9999) add equation to fit 6th degree polynomials in x and y to measured paralax
	for (u_int aX = 0; int(aX) < aSz.x; aX++) {
		for (u_int aY = 0; int(aY) < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);
			if (aParOrig[aY][aX] != -9999)
			{
				double aEq[12] = {X, X*X, X*X*X, X*X*X*X, X*X*X*X*X, X*X*X*X*X*X, Y, Y*Y, Y*Y*Y, Y*Y*Y*Y, Y*Y*Y*Y*Y, Y*Y*Y*Y*Y*Y };
				aSysPar.AddEquation(1, aEq, aParOrig[aY][aX]-aMeanParErr);
			}
		}
	}

	//Computing the result
	bool Ok;
	Im1D_REAL8 aSolPar = aSysPar.GSSR_Solve(&Ok);
	double* aPolyPar = aSolPar.data();

	cout << "Polynomial fit " << endl
		 << "Cst   = " << aMeanParErr << endl
		 << "Xcoef = " << aPolyPar[0] << " " << aPolyPar[1] << " " << aPolyPar[2] << " " << aPolyPar[3] << " " << aPolyPar[4] << " " << aPolyPar[5]  << endl
		 << "Ycoef = " << aPolyPar[6] << " " << aPolyPar[7] << " " << aPolyPar[8] << " " << aPolyPar[9] << " " << aPolyPar[10] << " " << aPolyPar[11] <<  endl;

	//Creating out container
	Im2D_REAL8  aParFit(aSz.x, aSz.y);
	REAL8 ** aDatParFit = aParFit.data();

	//Filling out container
	for (u_int aX = 0; int(aX) < aSz.x; aX++) {
		for (u_int aY = 0; int(aY) < aSz.y; aY++) {
			double X = double(aX);
			double Y = double(aY);
				aDatParFit[aY][aX] = aMeanParErr + aPolyPar[0] * X + aPolyPar[1] * X*X + aPolyPar[2] * X*X*X + aPolyPar[3] * X*X*X*X + aPolyPar[4] * X*X*X*X*X + aPolyPar[5] * X*X*X*X*X*X
					+ aPolyPar[6] * Y + aPolyPar[7] * Y*Y + aPolyPar[8] * Y*Y*Y + aPolyPar[9] * Y*Y*Y*Y + aPolyPar[10] * Y*Y*Y*Y*Y + aPolyPar[11] * Y*Y*Y*Y*Y*Y;
		}
	}

	//Output the Fitted paralax file (polynomials)
	
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
	


	
	
	///START OF SIN FITTING
	//Fit sin in y axis (init? aFreq ~= 0.0033, aAmplitude ~= 0.15)
	
	//FFT to refine frequency (aFreq is around 0.0033)
	
	// 2D->1D
	int firstValid=0;
	int lastValid=0;
	vector<double> a1DSignal;
	for (u_int aY = 0; int(aY) < aSz.y; aY++) {
		double aSum = 0;
		int aCpt = 0;
		for (u_int aX = 0; int(aX) < aSz.x; aX++) {
			if (aParOrig[aY][aX] != -9999)
			{
				aCpt++;
				aSum = aSum + aParOrig[aY][aX] - aDatParFit[aY][aX];
			}
		}
		// If enough data to be reliable
		if (aCpt > 2000) {
			if (firstValid == 0) { firstValid = aY; }
			aSum = aSum / double(aCpt);
			a1DSignal.push_back(aSum);
			lastValid = aY;
		}
		else {
			a1DSignal.push_back(-9999);
		}
	}

	//cout << "First Valid = " << firstValid << endl;
	//cout << "Last Valid = " << lastValid << endl;

	//Converting to complex type
	vector< complex<double> > a1DSignalC;
	for (u_int i = firstValid; int(i) < lastValid; ++i)
	{
		a1DSignalC.push_back(a1DSignal[i]);
	}

	//Computing DFT
	vector< complex<double> > aFour = DFT(a1DSignalC);

	//Normalize
	vector<double> normFour;
	for (u_int i = 0; i < aFour.size(); ++i)
	{
		double aReal = aFour[i].real();
		double aImag = aFour[i].imag();
		normFour.push_back(sqrt(aReal * aReal + aImag * aImag));
	}

	//Find main frequency
	std::vector<double>::iterator result = std::max_element(normFour.begin(), normFour.end());
	double aFreq = (std::distance(normFour.begin(), result)+1) / double(normFour.size());


	//Solve for aAmplitude and aPhase with aFreq known
	L2SysSurResol aSysSin(2);
	//For all points that are not nullified by bad correlation (value=9999) add equation to fit sin in the y axis to measured paralax (corrected of the polynome)
	//z=a*sin(b*y+c)=a*sin(c)cos(b*y)+a*cos(c)sin(b*y)=d*COSY+e*SINY
	/*
	for (u_int aY = 0; aY < aSz.y; aY++) {
		double COSY = cos(2 * M_PI * aFreq * double(aY));
		double SINY = sin(2 * M_PI * aFreq * double(aY));
		for (u_int aX = 0; aX < aSz.x; aX++) {
			if (aParOrig[aY][aX] != -9999)
			{
				double aEq[2] = { COSY, SINY };
				aSysSin.AddEquation(1, aEq, aParOrig[aY][aX] - aDatParFit[aY][aX]);
			}
		}
	}
	*/
	for (u_int aY = 0; int(aY) < aSz.y; aY++) {
			double COSY = cos(2 * M_PI * aFreq * double(aY));
			double SINY = sin(2 * M_PI * aFreq * double(aY));
			if (a1DSignal[aY] != -9999)
			{
				double aEq[2] = { COSY, SINY };
				aSysSin.AddEquation(1, aEq, a1DSignal[aY]);
			}
	}

	//Computing the result
	Im1D_REAL8 aSolSin = aSysSin.GSSR_Solve(&Ok);
	double* aSin = aSolSin.data();

	double aAmplitude = sqrt(aSin[0] * aSin[0] + aSin[1] * aSin[1]);
	double aPhase = atan(aSin[0] / aSin[1]);
	
	///END Sin Fitting

	
	cout << "Sin fit " << endl
		<< "Amplitude = " << aAmplitude << endl
		<< "Phase     = " << aPhase << endl
		<< "Frequency = " << aFreq << endl;

	//Creating out container
	Im2D_REAL8  aParFit2(aSz.x, aSz.y);
	REAL8 ** aDatParFit2 = aParFit2.data();

	//Filling out container
	for (u_int aX = 0; int(aX) < aSz.x; aX++) {
		for (u_int aY = 0; int(aY) < aSz.y; aY++) {
			// double X = aX;
			// double Y = aY;
			aDatParFit2[aY][aX] = aDatParFit[aY][aX] + aAmplitude*sin(2 * M_PI * aFreq * aY + aPhase);
		}
	}

	//Output the Fitted paralax file (polynomials + sin)
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
	return aParFit2;
	


	//return aParFit;
}


int ApplyParralaxCor_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aNameIm, aNameParallax;
	std::string aNameOut = "";
	bool aFitASTER = false;
	//Reading the arguments
	ElInitArgMain
		(
		argc, argv,
		LArgMain()
		<< EAMC(aNameIm, "Image to be corrected", eSAM_IsPatFile)
		//<< EAMC(aNameIm2, "Other image", eSAM_IsPatFile)
		<< EAMC(aNameParallax, "Paralax correction file", eSAM_IsPatFile),
		//<< EAMC(aNameDEM, "DEM file", eSAM_IsPatFile),
		LArgMain()
		<< EAM(aNameOut, "Out", true, "Name of output image (Def=ImName_corrected.tif)")
		<< EAM(aFitASTER, "FitASTER", true, "EXPERIMENTAL Fit functions appropriate for ASTER L1A processing (Def=false)")
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
		Im2D_REAL8 aParFit = FitASTER(aDatPar,aDir, aSz);
		ELISE_COPY
		(
			aTFPar.all_pts(),
			aParFit.in(),
			aPar.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
		);
		/*REAL8 ** aDatPar = */  aPar.data();
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
