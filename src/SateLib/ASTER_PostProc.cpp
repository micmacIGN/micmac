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
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"



void ASTER_Banniere()
{
	std::cout << "\n";
	std::cout << " *********************************\n";
	std::cout << " *     ASTER                     *\n";
	std::cout << " *     Post                      *\n";
	std::cout << " *     Processing                *\n";
	std::cout << " *********************************\n\n";
}

int ASTERProjAngle2OtherBand_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aBandIn, aBandOut;
	std::string aOutDir = "../";
	//Reading the arguments
	ElInitArgMain
	(
		argc, argv, 
		LArgMain()
		<< EAMC(aBandIn, "Name of 3N or 3B band where points are taken from (ex : AST_L1A_XXX_3N)", eSAM_IsPatFile)
		<< EAMC(aBandOut, "Name of 3N or 3B band where points are projected to (ex : AST_L1A_XXX_3N)", eSAM_IsPatFile),
		LArgMain()
	);

	std::string aDir, aSceneInName, aSceneOutName;
	SplitDirAndFile(aDir, aSceneInName, aBandIn);
	SplitDirAndFile(aDir, aSceneOutName, aBandOut);

	//Reading the image
	Tiff_Im aTF_In = Tiff_Im::StdConvGen(aDir + aSceneInName + ".tif", 1, false);
	Pt2di aSz_In = aTF_In.sz(); cout << "size of image In = " << aSz_In << endl;

	Tiff_Im aTF_Out = Tiff_Im::StdConvGen(aDir + aSceneOutName + ".tif", 1, false);
	Pt2di aSz_Out = aTF_Out.sz(); cout << "size of image Out = " << aSz_Out << endl;

	// Loading the GRID files
	std::string aGRIBinInName = aDir + aSceneInName + ".GRIBin";
	std::string aGRIBinOutName = aDir + aSceneOutName + ".GRIBin";
	ElAffin2D oriIntImaM2C_In, oriIntImaM2C_Out;
	Pt2di Sz(100000, 100000);
	ElCamera* mCameraIn = new cCameraModuleOrientation(new OrientationGrille(aGRIBinInName), Sz, oriIntImaM2C_In);
	ElCamera* mCameraOut = new cCameraModuleOrientation(new OrientationGrille(aGRIBinOutName), Sz, oriIntImaM2C_Out);

	//Create output
	std::ofstream fic("GeoI-Px/AngleFrom_" + aSceneInName + "_to_" + aSceneOutName + ".txt");
	fic << std::setprecision(15);
	fic << "X1 Y1 X2 Y2 Angle" << std::endl;
	
	// Compute values
	vector<double> aVectAngles;
	vector<Pt3dr> aVectPointsWithAnglesLeft, aVectPointsWithAnglesRight;
	Pt3dr aPt;
	for (size_t row = 0; int(row) < aSz_In.y; row++)
	{

		// define points in image
		Pt2dr aPtIm1(aSz_In.y / 4, row), aPtIm2(3 * aSz_In.y / 4, row);

		//Compute ground coordinate of points with GRIBin
		Pt2dr aPtOut1 = mCameraOut->Ter2Capteur(mCameraIn->ImEtZ2Terrain(aPtIm1, 0));
		Pt2dr aPtOut2 = mCameraOut->Ter2Capteur(mCameraIn->ImEtZ2Terrain(aPtIm2, 0));

		// Computing angles from horizontal
		double aAngle;
		// if left point is bellow right point
		if (aPtOut1.y < aPtOut2.y)
		{
			aAngle = -180 / M_PI * atan(abs((aPtOut2.y - aPtOut1.y) / (aPtOut2.x - aPtOut1.x)));
		}
		// if left point is above right point
		else
		{
			aAngle = 180 / M_PI * atan(abs((aPtOut2.y - aPtOut1.y) / (aPtOut2.x - aPtOut1.x)));
		}

		aVectAngles.push_back(aAngle);

		//Record points for interpolation
		aPt.x = aPtOut1.x; aPt.y = aPtOut1.y; aPt.z = aAngle;
		aVectPointsWithAnglesLeft.push_back(aPt);
		aPt.x = aPtOut2.x; aPt.y = aPtOut2.y; aPt.z = aAngle;
		aVectPointsWithAnglesRight.push_back(aPt);

		//cout << "Angle for row " << row << " : " << aAngle << endl;
		fic << aPtOut1.x << " " << aPtOut1.y << " " << aPtOut2.x << " " << aPtOut2.y << " " <<  aAngle << std::endl;
	}

	//Output map
	Im2D_REAL8  aAngleMap(aSz_Out.x, aSz_Out.y);
	REAL ** aData_AngleMap = aAngleMap.data();

	cout << "Starting interpolation of angles in Out band geometry" << endl;
	double aAngleInterpolated = 1e30; // Warn no init
	for (size_t i = 0;  int(i) < aSz_Out.x; i++)
	{
		for (size_t j = 0; int(j) < aSz_Out.y; j++)
		{
			// if points before the projection of the first points
			if (j < min(aVectPointsWithAnglesLeft[0].y,aVectPointsWithAnglesRight[0].y))
			{
				aAngleInterpolated = aVectPointsWithAnglesLeft[0].z;
			}
			// if points after the projection of the last points
			else if (j > max(aVectPointsWithAnglesLeft[aVectPointsWithAnglesLeft.size()-1].y,aVectPointsWithAnglesRight[aVectPointsWithAnglesRight.size()-1].y))
			{
				aAngleInterpolated = aVectPointsWithAnglesLeft[aVectPointsWithAnglesLeft.size() - 1].z;
			}
			else
			{
				//find point with min distance from points in aVectPointsWithAnglesLeft vectors (could work with aVectPointsWithAnglesRight) 
				double minDist = 99999999;
				double aDist;
				for (size_t k = 0; k < aVectPointsWithAnglesLeft.size(); k++)
				{
					aDist = sqrt(pow((aVectPointsWithAnglesLeft[k].x - (double)i), 2) + pow((aVectPointsWithAnglesLeft[k].y - (double)j), 2));
					if (aDist < minDist)
					{
						aDist = minDist;
						aAngleInterpolated = aVectPointsWithAnglesLeft[k].z;
					}

				}
				
			}
			// Record value
			aData_AngleMap[j][i] = aAngleInterpolated;
		}
	}
        // Test init
        ELISE_ASSERT(aAngleInterpolated!=1e30,"aAngleInterpolated not init");

	// Export data
	string aNameOut = aDir + "GeoI-Px/AngleFrom_" + aSceneInName + "_to_" + aSceneOutName + ".tif";
	cout << "Exporting data to : " << aNameOut << endl;

	Tiff_Im  aOut
	(
		aNameOut.c_str(),
		aSz_Out,
		GenIm::real8,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);

	ELISE_COPY
	(
		aOut.all_pts(),
		aAngleMap.in(),
		aOut.out()
	);



	ASTER_Banniere();

	return 0;
}


int ASTERProjAngle_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aDEMName, aMaskName, aSceneNameInit, aTargetSyst;
	std::string aOutDir = "../";
	//Reading the arguments
	ElInitArgMain
	(
		argc, argv,
		LArgMain()
		<< EAMC(aDEMName, "DEM name without extension (\".tif\" and \".tfw\" are required. ex : MEC-Malt/Z_Num9_DeZoom1_STD-MALT)", eSAM_IsPatFile)
		<< EAMC(aMaskName, "DEM mask name (ex : MEC-Malt/AutoMask_STD-MALT_Num_8.tif)", eSAM_IsPatFile)
		<< EAMC(aSceneNameInit, "Name of 3N or 3B band (ex : AST_L1A_XXX_3N)", eSAM_IsPatFile),
		LArgMain()
	);

	std::string aDir, aSceneName;
	SplitDirAndFile(aDir, aSceneName, aSceneNameInit);

	//Reading the image
	Tiff_Im aTF_3 = Tiff_Im::StdConvGen(aDir + aSceneName + ".tif", 1, false);
	Pt2di aSz_3 = aTF_3.sz(); cout << "size of image = " << aSz_3 << endl;

	// Loading the GRID file
	std::string aGRIBinname = aDir + aSceneName + ".GRIBin";
	ElAffin2D oriIntImaM2C;
	Pt2di Sz(100000, 100000);
	ElCamera* mCamera = new cCameraModuleOrientation(new OrientationGrille(aGRIBinname), Sz, oriIntImaM2C);


	vector<double> aVectAngles;

	for (size_t row = 0; int(row) < aSz_3.y; row++)
	{

		// define points in image
		Pt2dr aPtIm1(aSz_3.y / 4, row), aPtIm2(3 * aSz_3.y / 4, row);

		//Compute ground coordinate of points with GRIBin
		Pt3dr aPtGr1 = mCamera->ImEtZ2Terrain(aPtIm1, 0);
		Pt3dr aPtGr2 = mCamera->ImEtZ2Terrain(aPtIm2, 0);

		// Computing angles
		// Track NE-SW
		if (aPtGr1.x < aPtGr2.x && aPtGr1.y > aPtGr2.y)
		{
			aVectAngles.push_back(180 / M_PI * atan(abs((aPtGr2.y - aPtGr1.y) / (aPtGr2.x - aPtGr1.x))));
		}
		// Track SE-NW
		else if (aPtGr1.x > aPtGr2.x && aPtGr1.y > aPtGr2.y)
		{
			aVectAngles.push_back(90 + 180 / M_PI * atan(abs((aPtGr2.x - aPtGr1.x) / (aPtGr2.y - aPtGr1.y))));
		}
		// Track SW-NE
		else if (aPtGr1.x > aPtGr2.x && aPtGr1.y < aPtGr2.y)
		{
			aVectAngles.push_back(180 + 180 / M_PI * atan(abs((aPtGr2.y - aPtGr1.y) / (aPtGr2.x - aPtGr1.x))));
		}
		// Track NW-SE
		else if (aPtGr1.x < aPtGr2.x && aPtGr1.y < aPtGr2.y)
		{
			aVectAngles.push_back(270 + 180 / M_PI * atan(abs((aPtGr2.x - aPtGr1.x) / (aPtGr2.y - aPtGr1.y))));
		}
	}

	//cout << std::setprecision(15) << aVectAngles << endl;

	// Read DEM data
	Tiff_Im aTF_DEM = Tiff_Im::StdConvGen(aDir + aDEMName + ".tif", 1, false);
	Pt2di aSz_DEM = aTF_DEM.sz(); cout << "Size of DEM = " << aSz_DEM << endl;
	Im2D_REAL16  aDEM(aSz_DEM.x, aSz_DEM.y);
	ELISE_COPY
	(
		aTF_DEM.all_pts(),
		aTF_DEM.in(),
		aDEM.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	REAL16 ** aData_DEM = aDEM.data();

	// Read DEM metadata
	string aTFWName = aDir + aDEMName + ".tfw";
	std::fstream aTFWFile(aTFWName.c_str(), std::ios_base::in);
	double aRes_xEast, aRes_xNorth, aRes_yEast, aRes_yNorth, aCorner_East, aCorner_North;

	// Make sure the file stream is good
	if (!aTFWFile) {
		cout << endl << "Failed to open DEM .tfw file " << aTFWName << endl;
	}

	aTFWFile >> aRes_xEast >> aRes_xNorth >> aRes_yEast >> aRes_yNorth >> aCorner_East >> aCorner_North;


	// Read Mask
	Tiff_Im aTF_Mask = Tiff_Im::StdConvGen(aDir + aMaskName, 1, false);
	Pt2di aSz_Mask = aTF_Mask.sz(); cout << "Size of Mask = " << aSz_Mask << endl;
	Im2D_U_INT1  aMask(aSz_Mask.x, aSz_Mask.y);
	ELISE_COPY
	(
		aTF_Mask.all_pts(),
		aTF_Mask.in(),
		aMask.out()//Virgule(aImR.out(),aImG.out(),aImB.out())
	);
	U_INT1 ** aData_Mask = aMask.data();

	//Output map
	Im2D_REAL8  aAngleMap(aSz_DEM.x, aSz_DEM.y);
	REAL ** aData_AngleMap = aAngleMap.data();

	cout << "Starting interpolation of angle in DEM geometry" << endl;

	for (size_t i = 0; int(i) < aSz_DEM.x; i++)
	{
		for (size_t j = 0; int(j) < aSz_DEM.y; j++)
		{
			if (aData_Mask[j][i] == 0)
			{
				aData_AngleMap[j][i] = 0;
			}
			else
			{
				Pt3dr aPtGr(aCorner_East + i*aRes_xEast + j*aRes_yEast, aCorner_North + i*aRes_xNorth + j*aRes_yNorth, aData_DEM[j][i]);
				Pt2dr aPtIm = mCamera->Ter2Capteur(aPtGr);
				//No need to check if in camera since if not, then it wouldn't be in the DEM

				//Interpolate the angle value
				INT y_floor = (INT)(aPtIm.y);
				INT y_ceil = y_floor + 1;
				REAL aAngle = aVectAngles[y_floor] * (y_ceil - aPtIm.y) + aVectAngles[y_ceil] * (aPtIm.y - y_floor);

				// Record value
				aData_AngleMap[j][i] = aAngle;
			}
		}
	}

	// Export data
	string aNameOut = aDir + "TrackAngleMap.tif";
	cout << "Exporting data to : " << aNameOut << endl;

	Tiff_Im  aOut
	(
		aNameOut.c_str(),
		aSz_DEM,
		GenIm::real8,
		Tiff_Im::No_Compr,
		Tiff_Im::BlackIsZero
	);

	ELISE_COPY
	(
		aOut.all_pts(),
		aAngleMap.in(),
		aOut.out()
	);



	ASTER_Banniere();
	return 0;
}



/*

// On cree un fichier de points geographiques pour les transformer avec proj4
{
std::ofstream fic("processing/direct_ptGeo.txt");
fic << std::setprecision(15);
for (size_t i = 0; i<vAltitude.size(); ++i)
{
double altitude = vAltitude[i];
for (int l = 0; l<nbLine; ++l)
{
for (int c = 0; c<nbSamp; ++c)
{
Pt3dr Pimg(ulcSamp + c * stepPixel, ulcLine + l * stepPixel, altitude);
//pour affiner les coordonnees
Pt3dr PimgRefined = ptRefined(Pimg, vRefineCoef);

Pt3dr Pgeo = DirectRPC(PimgRefined);
fic << Pgeo.x << " " << Pgeo.y << std::endl;
}
}
}
}
// transformation in the ground coordinate system
std::string command;
command = g_externalToolHandler.get("cs2cs").callName() + " " + inputSyst + " +to " + targetSyst + " processing/direct_ptGeo.txt > processing/direct_ptCarto.txt";
int res = system(command.c_str());
if (res != 0) std::cout << "error calling cs2cs in createDirectGrid" << std::endl;
// loading points
std::ifstream fic("processing/direct_ptCarto.txt");
while (!fic.eof() && fic.good())
{
double X, Y, Z;
fic >> X >> Y >> Z;
if (fic.good())
vPtCarto.push_back(Pt2dr(X, Y));
}
std::cout << "Number of points in direct grid : " << vPtCarto.size() << std::endl;



*/





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant ?  la mise en
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
associés au chargement,  ?  l'utilisation,  ?  la modification et/ou au
développement et ?  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe ?
manipuler et qui le réserve donc ?  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités ?  charger  et  tester  l'adéquation  du
logiciel ?  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
?  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder ?  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
