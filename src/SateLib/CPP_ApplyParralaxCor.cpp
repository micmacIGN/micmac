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


int ApplyParralaxCor_main(int argc, char ** argv)
{
	//std::string aNameIm, aNameIm2, aNameParallax, aNameDEM;
	std::string aNameIm, aNameParallax;
	std::string aNameOut = "";
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
		<< EAM(aNameOut, "Out", true, "Name of output image (Def=ImName_corrected.tif")
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
	

	//Output container
	Im2D_U_INT1  aImOut(aSz.x, aSz.y);
	U_INT1 ** aDataOut = aImOut.data();

	/*Things needed for RPC angle computation, not main goal of this function
	
	//Read RPCs
	RPC aRPC;
	string aNameRPC1 = "RPC_" + StdPrefix(aNameIm) + ".xml";
	aRPC.ReadDimap(aNameRPC1);
	cout << "Dimap File " << aNameRPC1 << " read" << endl;
	RPC aRPC2;
	string aNameRPC2 = "RPC_" + StdPrefix(aNameIm2) + ".xml";
	aRPC2.ReadDimap(aNameRPC2);
	cout << "Dimap File " << aNameRPC2 << " read" << endl;
	
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

	//Output angle container 1
	Im2D_REAL8  aAngleBOut(aSz.x, aSz.y);
	REAL8 ** aDataAngleBOut = aAngleBOut.data();
	string aNameAngleB = "AngleB.tif";

	//Output angle container 2
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


	cout << "size of image = " << aSz << endl;
	//Computing output data
	for (int aX = 0; aX < aSz.x; aX++)
	{
		for (int aY = 0; aY < aSz.y; aY++)
		{
			/*
			Pt3dr P0B(aX, aY, aDatDEM[aY][aX]);
			Pt3dr PW0 = aRPC.DirectRPC(P0B);
			Pt3dr PW1 = PW0, PW2 = PW0;
			PW1.z = PW1.z - 1;
			PW2.z = PW2.z + 1;
			Pt3dr P1B = aRPC.InverseRPC(PW1);
			Pt3dr P2B = aRPC.InverseRPC(PW2);
			Pt3dr P1N = aRPC2.InverseRPC(PW1);
			Pt3dr P2N = aRPC2.InverseRPC(PW2);
			//Pt3dr P1B(aX, aY, 0);
			//Pt3dr P2B(aX, aY, 10000);
			//Pt3dr P1N = aRPC2.InverseRPC(aRPC.DirectRPC(P1B));
			//Pt3dr P2N = aRPC2.InverseRPC(aRPC.DirectRPC(P2B));
			double aAngleB = atan((P2B.x - P1B.x) / (P2B.y - P1B.y));
			aDataAngleBOut[aY][aX] = aAngleB;
			double aAngleN = atan((P2N.x - P1N.x) / (P2N.y - P1N.y));
			aDataAngleNOut[aY][aX] = aAngleN;
			//cout << aX << " " << aY << " " << aAngle << endl;
			//cout << P1N << " " << P2N << " " << aAngle << endl;

			*/
			//THE THINGS COMPUTED ABOVE WILL BE USED IN A FURTHER UPDATE

			Pt2dr ptOut;
			ptOut.x = aX - aDatPar[aY][aX];// * cos(aAngleB);
			ptOut.y = aY - aDatPar[aY][aX];// * sin(aAngleB);
			aDataOut[aY][aX] = Reechantillonnage::biline(aData, aSz.x, aSz.y, ptOut);
		}
	}
	cout << "size of image = " << aSz << endl;
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
