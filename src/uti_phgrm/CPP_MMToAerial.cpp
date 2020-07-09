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

class cCapteur2Capteur
{
	public: 
		cCapteur2Capteur(int argc,char** argv);

		bool ProjectC1toC2();
		void SaveCrop1();
		void CalcH();

		void ReadAsci(std::string& aAsciXY);
		void Show();
	
	private:
		cInterfChantierNameManipulateur * mICNM;
		const std::vector<std::string> *  mSetNameAer;

		std::string mNameI;
		std::string mOri1;
		std::string mOri2;
		std::string mAsciXY;

		cBasicGeomCap3D * mCam1;
		std::map<std::string,cBasicGeomCap3D *> mMapCam2;

		Pt2di              mTransI;//origin of the crop
		Pt2di              mSzBBI;//size of bounding box of the crop 
		std::vector<Pt2dr> mVPtI;
		std::map<std::string,std::vector<Pt2dr>> mMapVPtJ;
		std::map<std::string,std::string>        mMapAsciJ;

		bool mDoReech;
		bool mShow;
};

cCapteur2Capteur::cCapteur2Capteur(int argc,char** argv) :
		mTransI(Pt2di(19000,19000)),
		mSzBBI(Pt2di(0,0)),
		mDoReech(true),
		mShow(false)
{

    std::string aJPat;
    std::string aOri1;
    std::string aOri2;



    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(mNameI,"Name of the terrestrial image I", eSAM_IsExistFile)
                      << EAMC(aJPat,"Pattern of names of the aerial image J", eSAM_IsExistFile)
                      << EAMC(mAsciXY,"Name of the ascii file with (x,y) coordinates in I", eSAM_IsExistFile),
          LArgMain()  << EAM(aOri1,"OriTer",true,"Name of the Ori of the I image")
                      << EAM(aOri2,"OriAer",true,"Name of the Ori of the J images")
					  << EAM(mDoReech,"DoReech",true,"Do resampling, Def=true")
                      << EAM(mShow,"Show",true,"Show computed homographies")
    );

    #if (ELISE_windows)
      replace( mAsciXY.begin(), mAsciXY.end(), '\\', '/' );
      replace( mNameI.begin(), mNameI.end(), '\\', '/' );
      replace( aJPat.begin(), aJPat.end(), '\\', '/' );
      replace( aOri1.begin(), aOri1.end(), '\\', '/' );
      replace( aOri2.begin(), aOri2.end(), '\\', '/' );
    #endif


	// Management of the dataset
	// Set1 (terrestrial)
	std::string aDir,aLocIm1;
    SplitDirAndFile(aDir,aLocIm1,mNameI);
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

	// Set2 (aerial)
	cElemAppliSetFile anEASF(aJPat);
    mSetNameAer = anEASF.SetIm();


	StdCorrecNameOrient(aOri1,aDir,true);
	StdCorrecNameOrient(aOri2,aDir,true);
	
	// Read coordinates in I (terr) image
	ReadAsci(mAsciXY);

	// Read camera for image I 
	mCam1 = mICNM->StdCamGenerikOfNames(aOri1,mNameI);
	
	// Read pattern of cameras for images J (aerial)
	for (auto aJ : *mSetNameAer)
	{
		mMapCam2[aJ] = mICNM->StdCamGenerikOfNames(aOri2,aJ);
	}

}

void cCapteur2Capteur::ReadAsci(std::string& aAsciXY)
{
    ELISE_fp aFIn(aAsciXY.c_str(),ELISE_fp::READ);
    char * aLine;

	Pt2dr aP1;
    while ((aLine = aFIn.std_fgets()))
    {
         int aNb=sscanf(aLine,"%lf  %lf",&aP1.x , &aP1.y);
         ELISE_ASSERT(aNb==2,"Could not read 2 double values");

         //std::cout << aP1  << "\n";
		 mVPtI.push_back(aP1);

    }

}

bool cCapteur2Capteur::ProjectC1toC2()
{

	ELISE_ASSERT(mCam1->AltisSolIsDef(),"Could not the mean Z of the ground");

	double aZMoy = mCam1->GetAltiSol();
	if (mShow)
		std::cout << "ZMoy=" << aZMoy << "\n";

	int aNbPt = int(mVPtI.size());


	Pt2di aPosMax(0,0);

	// Project points in I to ground
	std::vector<Pt3dr> aVPGr;
	for (auto aP : mVPtI)
	{
		// Origin of the selected region
		if (aP.x<mTransI.x)
			mTransI.x=aP.x;
		if (aP.y<mTransI.y)
            mTransI.y=aP.y;

		// End of selected region
		if (aP.x>aPosMax.x)
			aPosMax.x=aP.x;
		if (aP.y>aPosMax.y)
            aPosMax.y=aP.y;

		//1- intersect ascii with ALtiSol
        Pt3dr aP3d = mCam1->ImEtZ2Terrain(aP,aZMoy);
		aVPGr.push_back(aP3d);
		
		if (mShow)
			std::cout << aP3d << "\n";
	}
	mSzBBI = aPosMax - mTransI;//size of bounding box of the crop

	// Iterate over all aerial images
	for (auto aJ : mMapCam2)
	{
		cBasicGeomCap3D * aCam2 = aJ.second;

		std::vector<Pt2dr> aVPtJ;

		// Iterate over all ground points
		for (auto aP3d : aVPGr)
    	{
			if (aCam2->PIsVisibleInImage(aP3d))
			{
				//2- backproject to aerial
				Pt2dr aPAer = aCam2->Ter2Capteur(aP3d);
				aVPtJ.push_back(aPAer);
			}
		}

		// Add only if all points could be projected to image J
		if (aNbPt == int(aVPtJ.size()))
		{		
			mMapVPtJ[aJ.first] = aVPtJ;
		}
	}

	if (mShow)
		Show();

	return EXIT_SUCCESS;

}

void cCapteur2Capteur::SaveCrop1()
{
	// Save image
	std::string aNameImOut = StdPrefix(mAsciXY) + ".tif";

	Tiff_Im aTifI = Tiff_Im::StdConvGen(mNameI.c_str(),-1,true);
    Tiff_Im aTifICrop = Tiff_Im
                       (
                           aNameImOut.c_str(),
                           mSzBBI,
                           aTifI.type_el(),
                           Tiff_Im::No_Compr,
                           aTifI.phot_interp()
                       )                            ;

	ELISE_COPY
    (
    	rectangle(Pt2di(0,0),mSzBBI),
		trans(aTifI.in(),mTransI),
        aTifICrop.out()
    );


	// Save new asci files
	for (auto aJ : mMapVPtJ)
	{

		std::string aAsciXY_NEW = StdPrefix(mAsciXY) + "_" + StdPrefix(aJ.first) + ".txt";
		mMapAsciJ[aJ.first] = aAsciXY_NEW;



		ELISE_fp aFp(aAsciXY_NEW.c_str(),ELISE_fp::WRITE,false,ELISE_fp::eTxtTjs);
    
        aFp.SetFormatDouble("%.3lf");
		
		// iterate over all points
		for (int aK=0; aK<int(aJ.second.size()); aK++)
		{
			aFp.write_REAL8(mVPtI.at(aK).x - mTransI.x);
			aFp.write_REAL8(mVPtI.at(aK).y - mTransI.y);
			aFp.write_REAL8(aJ.second.at(aK).x);
			aFp.write_REAL8(aJ.second.at(aK).y);

			aFp.PutLine();
			
		}
		aFp.close();
	}

}

void cCapteur2Capteur::CalcH()
{
	
    std::list<std::string>  aLCom;

    for (auto aKIm : mMapAsciJ)
    {


        std::string aCom = MM3dBinFile_quotes("TestLib")
                         + " OneReechFromAscii "
                         + aKIm.first
                         + " " + aKIm.second 
                         + " Out=" + StdPrefix(aKIm.second) +".tif"
                         + " Show=" + ToString(mShow) 
						 + " DoReech=" + ToString(mDoReech);
        std::cout << aCom << "\n";
        aLCom.push_back(aCom);
    }

    cEl_GPAO::DoComInSerie(aLCom);

}

void cCapteur2Capteur::Show()
{
	std::cout << "Image I: ";
	for (auto aP : mVPtI)
		std::cout << aP << " ";
	std::cout << "\n";

	for (auto aIm : mMapVPtJ)
	{
		std::cout << aIm.first << "\n";
		for (auto aP : aIm.second)
		{
			std::cout << aP << " ";
		}
		std::cout << "\n";
	}
}

int AllHomMMToAerial_main(int argc,char** argv)
{

	std::string aNameImTerrPat;
	std::string aNameImAerPat;
	std::string aOriTerr;
	std::string aOriAer;
    std::string aASCIINamePat;
    cElemAppliSetFile aEASF;
    bool aShow = false;
	bool aDoReech = true;

    ElInitArgMain
    (
          argc,argv,  
          LArgMain()  << EAMC(aNameImTerrPat,"Pattern of the terrestrial images I", eSAM_IsExistFile)
                      << EAMC(aASCIINamePat,"Pattern of the ascii files with (x,y) coordinates in I", eSAM_IsExistFile)
					  << EAMC(aNameImAerPat,"Pattern of names of the aerial image J", eSAM_IsExistFile),
          LArgMain()  << EAM(aOriTerr,"OriTer",true,"Name of the Ori of the I image")
		              << EAM(aOriAer,"OriAer",true,"Name of the Ori of the J image")
					  << EAM(aDoReech,"DoReech",true,"Do resampling, Def=true")
		              << EAM(aShow,"Show",true,"Show computed homographies")
    );


/*
 * */


    #if (ELISE_windows)
      replace( aASCIINamePat.begin(), aASCIINamePat.end(), '\\', '/' );
      replace( aNameImTerrPat.begin(), aNameImTerrPat.end(), '\\', '/' );
      replace( aNameImAerPat.begin(), aNameImAerPat.end(), '\\', '/' );
      replace( aOriTerr.begin(), aOriTerr.end(), '\\', '/' );
      replace( aOriAer.begin(), aOriAer.end(), '\\', '/' );
    #endif

    aEASF.Init(aNameImTerrPat);

    cElRegex  anAutom(aNameImTerrPat.c_str(),10);

    std::list<std::string>  aLCom;
    for (size_t aKIm=0  ; aKIm< aEASF.SetIm()->size() ; aKIm++)
    {
        std::string aNameIm = (*aEASF.SetIm())[aKIm];

        std::string aNameCorresp  =  MatchAndReplace(anAutom,aNameIm,aASCIINamePat);

        std::string aCom = MM3dBinFile_quotes("TestLib")
                         + " OneMMToAerial "
                         + aNameIm
                         + " " + aNameImAerPat 
						 + " " + aNameCorresp
                         + " OriTer=" + aOriTerr 
                         + " OriAer=" + aOriAer
						 + " DoReech=" + ToString(aDoReech)
						 + " Show=" + ToString(aShow);
        std::cout << aCom << "\n";
        aLCom.push_back(aCom);
    }

    cEl_GPAO::DoComInParal(aLCom);

	return EXIT_SUCCESS;
}


int OneHomMMToAerial_main(int argc,char** argv)
{


	cCapteur2Capteur aProjMM2A(argc,argv);
    aProjMM2A.ProjectC1toC2();
	aProjMM2A.SaveCrop1();	
	aProjMM2A.CalcH();


	return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
