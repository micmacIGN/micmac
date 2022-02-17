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

// Export vector data after drawing a line in an orthoimage : Export axe long profile //

#include "StdAfx.h"
	
int Panache_main(int argc,char ** argv)
{
	std::string aDir, aOrtho, aMTDOrtho, aXmlLine, aDSM, aZnum, aOut="Profile.xyz", aPly="Profile.ply";
	double aX1, aX2, aY1, aY2;
	bool aBin=true;
	double apixPas=1;
	bool aShow=false;
	Pt3di aCoul(255,0,0);
	std::vector<Pt3di> aVCol;
	std::vector<Pt3dr> aVFPts;
	
	ElInitArgMain
	(
		argc,argv,
		LArgMain()  << EAMC(aDir,"Directory")
					<< EAMC(aOrtho,"Orthoimage", eSAM_IsExistFile)
					<< EAMC(aMTDOrtho,"xml file for Orthoimage params", eSAM_IsExistFile)
					<< EAMC(aXmlLine,"xml file of line image coordinates",eSAM_IsExistFile)
					<< EAMC(aDSM,"xml file NuageImProf_STD-MALT_Etape_X.xml", eSAM_IsExistFile)
					<< EAMC(aZnum,"xml file Z_NumX_DeZoomX_STD-MALT.xml (last etape)",eSAM_IsExistFile),
		LArgMain()  << EAM(apixPas,"pixStep",true,"sampling computation along axis ; def = 1 px")
					<< EAM(aCoul,"FixColor",true,"Fix the color of points ; def = (255,0,0)")
					<< EAM(aShow,"Show",false,"Show some details printed on prompt ; def = false")
					<< EAM(aOut,"Out",true,"Export profile to .xyz format (def = Profile.xyz)")
					<< EAM(aPly,"ExportPly",true,"Export profile to .ply format (def = Profile.ply)")
					<< EAM(aBin,"Bin",true,"Generate Binary or Ascii .ply file (Def=1, Binary)")
	);
	
	std::cout << "aDir = " << aDir << std::endl;
	
	//read aXmlLine : mesures images
	cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aXmlLine,SetOfMesureAppuisFlottants);
	std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();
	std::vector<Pt2dr> aPtx;
	
	for (std::list<cMesureAppuiFlottant1Im>::iterator iT1= aLMAF.begin();iT1 != aLMAF.end();iT1++)
	{
		
		std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
		
		for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
		{
			Pt2dr aPtC;
			aPtC = iT2->PtIm();
			aPtx.push_back(aPtC);
		}
		
	}
	
	std::cout << "aPtx = " << aPtx << std::endl;
		
	//line equation
	ELISE_ASSERT(aPtx.size() == 2, "Size of points for line must be equal to 2 !");
	
	aX1 = aPtx.at(0).x;
	std::cout << "aX1 = " << aX1 << std::endl;
	aX2 = aPtx.at(1).x;
	std::cout << "aX2 = " << aX2 << std::endl;
	aY1 = aPtx.at(0).y;
	std::cout << "aY1 = " << aY1 << std::endl;
	aY2 = aPtx.at(1).y;
	std::cout << "aY2 = " << aY2 << std::endl;
	
	double Coeff =  ( aY2 - aY1) / ( aX2 - aX1);
	double Org = aY1 - Coeff * aX1 ;
	
	std::cout << "Coeff = " << Coeff << std::endl;
	std::cout << "Org = " << Org << std::endl;

	//read aOrtho
	Tiff_Im aTifIm1 = Tiff_Im::StdConvGen(aDir + aOrtho,1,true);
    Pt2di aSz1 = aTifIm1.sz();
    
    std::cout << "aSz1 = " << aSz1 << std::endl;
    
    //read aMTDOrtho
    cFileOriMnt mMTDOrtho = StdGetFromPCP(aMTDOrtho,FileOriMnt);
	Pt2dr mOriginXY = mMTDOrtho.OriginePlani();
	Pt2dr mResoXY = mMTDOrtho.ResolutionPlani();

	//double mOriginZ = mMTDOrtho.OrigineAlti();
	//double mResoZ = mMTDOrtho.ResolutionAlti();
	Pt2di mOrthoSz = mMTDOrtho.NombrePixels();
	
	//compute scale factor
	double aScaleX = mOrthoSz.x / aSz1.x;
	double aScaleY = mOrthoSz.y / aSz1.y;

	ELISE_ASSERT(aScaleX == aScaleY, "Scaling factor is not coherent !");
	
	//generate line in image coordinates
	std::vector<double> aXi, aYi;
	double aStart;
	double aEnd;
	
	if(aX1 < aX2)
	{
		aStart = round_ni(aX1);
		aEnd = round_ni(aX2);
	}
	else
	{
		aStart = round_ni(aX2);
		aEnd = round_ni(aX1);
	}
	
	for (int compt=aStart; compt < aEnd ; compt=compt+apixPas)
	{
		std::cout << "compt = " << compt << std::endl;
		aXi.push_back(compt);
		double aY = Coeff * compt + Org ;
		std::cout << "aY = " << aY << std::endl;
		aYi.push_back(aY);
		//compteur++;
	}
	
	std::cout << "aXi.size() = " << aXi.size() << std::endl;
	std::cout << "aYi.size() = " << aYi.size() << std::endl;
	
	//give to line reel coordinates
	std::vector<Pt2dr> aVRPts;
	
	for (unsigned int Compt=0; Compt < aXi.size(); Compt++)
	{
		Pt2dr aPtCr(mOriginXY.x + (aXi.at(Compt) * mResoXY.x * aScaleX) , mOriginXY.y + (aYi.at(Compt) * mResoXY.y * aScaleY));
		aVRPts.push_back(aPtCr);
	}
	
	//read aZnum xml file
	cFileOriMnt bDico = StdGetFromPCP(aZnum,FileOriMnt);
	Pt2dr aOrgMNT = bDico.OriginePlani();
	Pt2dr aResMNT = bDico.ResolutionPlani();
	
	//read aDSM
	cElNuage3DMaille * DSM = cElNuage3DMaille::FromFileIm(aDSM);
	std::vector<double> aZValues;
	Pt2di  aGCPinMNT;
	
	for (unsigned int iT=0 ; iT < aVRPts.size() ; iT++)
	{
		aGCPinMNT.x = round_ni(( aVRPts.at(iT).x - aOrgMNT.x) / aResMNT.x);
		std::cout << "aGCPinMNT.x = " << aGCPinMNT.x << std::endl;
		
		aGCPinMNT.y = round_ni(( aVRPts.at(iT).y - aOrgMNT.y) / aResMNT.y);
		std::cout << "aGCPinMNT.y = " << aGCPinMNT.y << std::endl;
		
		if (DSM->IndexHasContenu(aGCPinMNT))
		{
			Pt3dr aPTer = DSM->PtOfIndex(aGCPinMNT);
			std::cout << "aPTer = " << aPTer << std::endl;
			double aZval = aPTer.z;
			std::cout << "aZval = " << aZval << std::endl;
			aZValues.push_back(aZval);
			Pt3dr aP2X(aVRPts.at(iT).x,aVRPts.at(iT).y,aZval);
			aVFPts.push_back(aP2X);
			aVCol.push_back(aCoul);
		}
	}
	
	//export to .xyz file
	if (!MMVisualMode)
	{
		FILE * aFP = FopenNN(aOut,"w","Panache_main");
		
		cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);
				

		for (int bK=0; bK < (int) aVFPts.size() ; bK++)
		{
			fprintf(aFP,"%d %lf %lf %lf \n",bK, aVFPts.at(bK).x,aVFPts.at(bK).y,aVFPts.at(bK).z);
		}
			
		ElFclose(aFP);
			
		}
	
	//export to .ply file
	std::list<std::string> aVCom;
	std::vector<const cElNuage3DMaille *> aVNuage;
	
	//~ cElNuage3DMaille * aRes = aNuage;
	//~ aRes->PlyPutFile(aPly,aVCom,(aBin!=0),&aVFPts,true,
	
	//aRes->PlyPutFile( aNameOut, aLComment, (aBin!=0), DoNrm, DoublePrec, anOffset );
	
	cElNuage3DMaille::PlyPutFile
    (
		aPly,
        aVCom,
        aVNuage,
        &aVFPts,
        &aVCol,
        aBin,
        false,
        false
    );
    
    if(aShow)
    {
		std::cout << "NameIm = " << aOrtho << "\n";
		std::cout << "Coeff = " << Coeff << std::endl;
		std::cout << "Org = " << Org << std::endl;
		std::cout << "aSz1 = " << aSz1 << std::endl;
		std::cout << "mOriginXY = " << mOriginXY << std::endl;
		std::cout << "mResoXY = " << mResoXY << std::endl;
		std::cout << "mOrthoSz = " << mOrthoSz << std::endl;
		std::cout << "aScaleX = " << aScaleX << std::endl;
		std::cout << "aScaleY = " << aScaleY << std::endl;
		std::cout << "aOrgMNT = " << aOrgMNT << std::endl;
		std::cout << "aResMNT = " << aResMNT << std::endl;
		std::cout << "aVFPts.size() = " << aVFPts.size() << std::endl;
	}
	
    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
