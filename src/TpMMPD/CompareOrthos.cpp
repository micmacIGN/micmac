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
#include "TiePByMesh/InitOutil.h"



bool IsFileExists( const char * FileName )
{
	#if (ELISE_unix)
		FILE* fp = NULL;
		fp = fopen( FileName, "rb" );
		if( fp != NULL )
		{
			fclose( fp );
			return true;
		}
	#endif
    return false;
}

//***********************************************************************************************
//to do :
//* save correctly image on disq : only dump is done
//* add --> for better visualization;
//* take window size into account : both cases, draw on black image / draw on first ortho image;
//***********************************************************************************************


int CmpOrthos_main(int argc,char ** argv)
{
	std::string aDir, aOrtho1, aOrtho2, aMTDOrtho1, aMTDOrtho2, aOut="", aOutTxt="";
	int aScale = 1;
	bool aFormat = false;
	Pt2di aSizeW(1000,1000);
	bool aDrawOnOrtho = true;
	std::string ahomName("");
	int aFactMult=1;

	ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory")
					<< EAMC(aOrtho1,"Orthoimage 1 : considered as reference", eSAM_IsExistFile)
					<< EAMC(aMTDOrtho1,".xml params file of the first Orthoimage", eSAM_IsExistFile)
					<< EAMC(aOrtho2,"Orthoimage 2", eSAM_IsExistFile)
					<< EAMC(aMTDOrtho2,".xml params file of the second Orthoimage", eSAM_IsExistFile),
        LArgMain()  << EAM(aScale,"Scale",false,"Scale factor for both Orthoimages ; Def=1")
					<< EAM(aOut,"OutImg",true,"Name of output image ; Def=Ortho1_Ortho2.tif")
					<< EAM(aSizeW,"SzW",false,"Size of output window ; Def=[1000,1000]")
					<< EAM(aFormat,"AddFormat",false,"Add format of output data in .txt file header ; Def=false",eSAM_IsBool)
					<< EAM(aOutTxt,"OutFile",false,"Name of output .txt file to Export values ; Def=Ortho1_Ortho2.txt")
					<< EAM(aDrawOnOrtho,"DrawOnOrtho",true,"Draw on Orthoimage 1 ; Def=true",eSAM_IsBool)
					<< EAM(aFactMult,"FactMult",false,"Multiplie delta bay this factor; Def=1")
    );
    
    //read Ortho 1
    Tiff_Im aTifIm1 = Tiff_Im::StdConvGen(aDir + aOrtho1,1,true);
    Pt2di aSz1 = aTifIm1.sz();
    std::cout << "aSz1 " << aOrtho1 << " : " << aSz1 << std::endl;
    
    //read Ortho 2
    Tiff_Im aTifIm2 = Tiff_Im::StdConvGen(aDir + aOrtho2,1,true);
    Pt2di aSz2 = aTifIm2.sz();
    std::cout << "aSz2 " << aOrtho2 << " : " << aSz2 << std::endl;
    
	//read xml Ortho 1
	cFileOriMnt mMTDOrtho1 = StdGetFromPCP(aMTDOrtho1,FileOriMnt);	
	Pt2dr mOriginXY1 = mMTDOrtho1.OriginePlani();
	Pt2dr mResoXY1 = mMTDOrtho1.ResolutionPlani();
	Pt2di mOrthoSz1 = mMTDOrtho1.NombrePixels();
	std::cout << "mOrthoSz1 " << aMTDOrtho1 << " : " << mOrthoSz1 << std::endl;
	
	//read xml Ortho 2
	cFileOriMnt mMTDOrtho2 = StdGetFromPCP(aMTDOrtho2,FileOriMnt);
	Pt2dr mOriginXY2 = mMTDOrtho2.OriginePlani();
	Pt2dr mResoXY2 = mMTDOrtho2.ResolutionPlani();
	Pt2di mOrthoSz2 = mMTDOrtho2.NombrePixels();
	std::cout << "mOrthoSz2 " << aMTDOrtho2 << " : " << mOrthoSz2 << std::endl;
	
	//scale factor between .xml file and image
	//Ortho 1
	double aFactor1X = aSz1.x/static_cast<double>(mOrthoSz1.x);
	std::cout << "Factor1.x = " << aFactor1X << std::endl;
	double aFactor1Y = aSz1.y/static_cast<double>(mOrthoSz1.y);
	std::cout << "Factor1.y = " << aFactor1Y << std::endl;
	
	//Ortho 2
	double aFactor2X = aSz2.x/static_cast<double>(mOrthoSz2.x);
	std::cout << "Factor2.x = " << aFactor2X << std::endl;
	double aFactor2Y = aSz2.y/static_cast<double>(mOrthoSz2.y);
	std::cout << "Factor2.y = " << aFactor2Y << std::endl;
	
	if(aFactor1X != 1 || aFactor1Y != 1 || aFactor2X != 1 || aFactor2Y != 1)
	{
		std::cout << "Warning : Size of image is not the same as the size on its .xml file! " << std::endl; 
	}
    
    
    //reading tie points coordinates
    std::vector<Pt2dr> aPoints1;
    std::vector<Pt2dr> aPoints2;
    
	std::string mPatHom = aDir + "Homol/Pastis" + aOrtho1 + "/" + aOrtho2 + ".dat";
	cInterfChantierNameManipulateur * mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	std::string mKey = "NKS-Assoc-CplIm2Hom@" + ahomName + "@dat";
	
	if (IsFileExists(mPatHom.c_str()))
	{
		std::string aNameH =  aDir + mICNM->Assoc1To2
										(
											mKey,
											aOrtho1,
											aOrtho2,
											true
										);
				
		ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);
		for (
			ElPackHomologue::iterator iTH = aPack.begin();
			iTH != aPack.end();
			iTH++
			)
		{
			Pt2dr aP1 = iTH->P1();
			aPoints1.push_back(aP1);
			Pt2dr aP2 = iTH->P2();
			aPoints2.push_back(aP2);
		}
	}
	
	else
	{
		std::cout << "File " << mPatHom.c_str() << " not found" << std::endl;
	}
	
	ELISE_ASSERT(aPoints1.size() == aPoints2.size(),"Problem with Tapioca step, number of tie points can't be different");
	
    //give to tie points reel coordinates with MTDOrtho.xml
	std::vector<Pt2dr> aPtsIm1;
	std::vector<Pt2dr> aPtsIm2;
	std::vector<Pt2dr> aDelta;
	
	for (u_int aT=0; aT<aPoints1.size() ; aT++)
	{
		double xPtIm1 = mOriginXY1.x + aPoints1.at(aT).x*mResoXY1.x*aFactor1X;
		
		double yPtIm1 = mOriginXY1.y + aPoints1.at(aT).y*mResoXY1.y*aFactor1Y;
		
		Pt2dr PtIm1(xPtIm1,yPtIm1);
		aPtsIm1.push_back(PtIm1);
		
		//std::cout << "aPtsIm1.at(aT) : " << aPtsIm1.at(aT) << std::endl;
		
		double xPtIm2 = mOriginXY2.x + aPoints2.at(aT).x*mResoXY2.x*aFactor2X;
		
		double yPtIm2 = mOriginXY2.y + aPoints2.at(aT).y*mResoXY2.y*aFactor2Y;
		
		Pt2dr PtIm2(xPtIm2,yPtIm2);
		aPtsIm2.push_back(PtIm2);
				
		double xDelta = (xPtIm2 - xPtIm1)*mResoXY1.x*aFactMult;
		//~ std::cout << "xDelta = " << xDelta << std::endl;
		double yDelta = (yPtIm2 - yPtIm1)*mResoXY1.y*aFactMult;
		//~ std::cout << "yDelta = " << yDelta << std::endl;
		
		Pt2dr Delta(xDelta,yDelta);
		aDelta.push_back(Delta);
	}
	
    
    //give output image size of Ortho 1
    if(aSizeW.x == 0 && aSizeW.y ==0)
    {
		aSizeW.x = aSz1.x;
		aSizeW.y = aSz1.y;
	}
    
    //plot in a file
    Video_Win * VW = 0;
    Im2D<U_INT1,INT4> I(aSizeW.x,aSizeW.y);
    
    if(aDrawOnOrtho)
    {
		Tiff_Im mPicTiff = Tiff_Im::StdConvGen(aDir + aOrtho1,1,true);
		TIm2D<U_INT1,INT4> mPic_TIm2D = mPicTiff.sz();
		ELISE_COPY(mPic_TIm2D.all_pts(), mPicTiff.in(), mPic_TIm2D.out());
		I = mPic_TIm2D._the_im;
		VW = display_image(&I, "Displacement_Field", VW, 0.2);
	}
	
	else
	{
		VW = display_image(&I, "Displacement_Field", VW, 0.2);
	}

    for(unsigned int aK=0; aK<aDelta.size(); aK++)
    {
		std::vector<Pt2dr> aVC;
		aVC.push_back(aPoints1.at(aK));
				
		double xDep = aPoints1.at(aK).x+aDelta.at(aK).x;
		double yDep = aPoints1.at(aK).y+aDelta.at(aK).y;
		Pt2dr aPt(xDep,yDep);
		aVC.push_back(aPt);
		
        VW = draw_pts_onVW(aPoints1.at(aK),VW,"red");
        VW = draw_pts_onVW(aPt,VW,"blue");
		VW = draw_polygon_onVW(aVC,VW,Pt3di(0,255,0), false,false);
	}
	
    //write the image in a file
    if(aOut == "")
    {
		aOut = StdPrefixGen(aOrtho1) + "_" + StdPrefixGen(aOrtho2) + ".tif";
	}

	#if ELISE_X11
		VW->DumpImage(aOut);
	#endif

	//~ Im2D<U_INT1,INT> aResIm = I;
            //~ 
    //~ ELISE_COPY
    //~ (
        //~ VW->all_pts(),
        //~ aResIm.in(),
        //~ aResIm.out()
    //~ );
    
    VW->clik_in();
    
    //export in format : Pt i_1 j_1 x_1 y_1 i_2 j_2 x_2 y_2
    if(aOutTxt == "")
    {
		aOutTxt = StdPrefixGen(aOrtho1) + "_" + StdPrefixGen(aOrtho2) + ".txt";
	}
    if (!MMVisualMode)
	{
		FILE * aFP = FopenNN(aOutTxt,"w","CmpOrthos_main");
		cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOutTxt);
		
		if(aFormat)
		{
			std::string Format = "#F=i1_j1_X1_Y1_i2_j2_X2_Y2";
			fprintf(aFP,"%s\n", Format.c_str());
		}

		for (unsigned int aK=0 ; aK<aPoints1.size() ; aK++)
		{
              fprintf(aFP,"%.3f %.3f %.5f %.5f %.3f %.3f %.5f %.5f\n", aPoints1.at(aK).x, aPoints1.at(aK).y, aPtsIm1.at(aK).x, aPtsIm1.at(aK).y, aPoints2.at(aK).x, aPoints2.at(aK).y, aPtsIm2.at(aK).x, aPtsIm2.at(aK).y);       
		}
		
	ElFclose(aFP);
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
