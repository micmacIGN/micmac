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

vector<Pt3dr> ComputedZFromDEMAndMask(REAL8** aDEMINData, vector<double> aTFWin, string aDEMRefPath, string aMaskPath)
{

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
		string aDir, aNameDEMRef;
		SplitDirAndFile(aDir, aNameDEMRef, aDEMRefPath);

		cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string>* aSetIm = aICNM->Get(aNameDEMRef);
		Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameDEMRef, 3, false);
		Pt2di aSz = aTF.sz();

		Im2D_REAL8  aDEMRef(aSz.x, aSz.y);

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


	return aListXYdZ;
}

vector<Pt3dr> ComputedZFromDEMAndXY(REAL8** aDEMINData, vector<double> aTFW, string aDEMRef, string aListPointsPath)
{
	vector<Pt3dr> aListXYdZ;


	return aListXYdZ;
}

vector<Pt3dr> ComputedZFromGCPs(REAL8** aDEMINData, vector<double> aTFW, string aListGCPsPath)
{
	vector<Pt3dr> aListXYdZ;


	return aListXYdZ;
}





int Banana_main(int argc, char ** argv)
{
	string aDEMinPath, aDEMRefPath = "", aMaskPath = "", aListPointsPath = "", aListGCPsPath = "";
	uint aDeg;
	ElInitArgMain
	(
		argc, argv,
		LArgMain() 
		<< EAMC(aDEMinPath, "Input DEM to be corrected - DEM must have tfw"),
		LArgMain()
		<< EAM(aDeg, "DegPoly", true, "Degree of fitted polynome (default = 2)")
		<< EAM(aDEMRefPath, "DEMRef", true, "Reference DEM - DEM must have tfw")
		<< EAM(aMaskPath, "Mask", true, "A binary mask of stable terrain - if value=1 then the point is used, if =0 then unused (to be used with a reference DEM) - mask must have tfw")
		<< EAM(aListPointsPath, "ListPoints", true, "A text file of XY coordinates of stable points (to be used with a reference DEM)")
		<< EAM(aListGCPsPath, "ListGCPs", true, "A text file of XYZ coordinates of stable points (to be used without a reference DEM)")
	);
	
	// Load Input DEM
		string aDir,aNameDEMin;
		SplitDirAndFile(aDir, aNameDEMin, aDEMinPath);

		cInterfChantierNameManipulateur* aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
		const std::vector<std::string>* aSetIm = aICNM->Get(aNameDEMin);
		Tiff_Im aTF = Tiff_Im::StdConvGen(aDir + aNameDEMin, 3, false);
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



	// Check what inputs are given
		vector<Pt3dr> aListXYdZ;
		if (aDEMRefPath != "" && aMaskPath != "") { aListXYdZ = ComputedZFromDEMAndMask(aDEMINData, aTFW, aDEMRefPath, aMaskPath); }
		else if (aDEMRefPath != "" && aListPointsPath != "") { aListXYdZ = ComputedZFromDEMAndXY(aDEMINData, aTFW, aDEMRefPath, aListPointsPath); }
		else if (aListGCPsPath != "") { aListXYdZ = ComputedZFromGCPs(aDEMINData, aTFW, aListGCPsPath); }
		else { ELISE_ASSERT(false, "No valid combination of input given"); }

	//For each case, a list of XYdZ points (aListXYdZ) is generated.

	//Compute polynome that would fit that distribution of bias

	//Apply polynome to DEM

	//Export DEM


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
