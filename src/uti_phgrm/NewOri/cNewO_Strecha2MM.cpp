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

class cAppliSTRCam
{
	public:
		cAppliSTRCam(int argc, char** argv);

		void Exe();
	private:
		void SaveOC(std::string& aImN, ElMatrix<double>& aR, Pt3dr& aTr);
		void SaveCal(std::string& aImN, double aF, Pt2dr& aPP, Pt2di& aSz);

		std::string mOri;

		cInterfChantierNameManipulateur* mICNM;

		std::list<std::string> mLFile;
};

cAppliSTRCam::cAppliSTRCam(int argc, char** argv)
{
	//read all files
	//iterate over files and create Ori

	std::string aPat;	
	std::string aDir;

	ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aPat,"Pattern of files")
				   << EAMC(mOri,"Orientation directory"),
        LArgMain() 
    );

    #if (ELISE_windows)
        replace( aPat.begin(), aPat.end(), '\\', '/' );
    #endif

	if (mOri.rfind("Ori-") != string::npos)
	{
		mOri = mOri.substr(mOri.rfind("Ori-")+4);
	}

	SplitDirAndFile(aDir,aPat,aPat);
    StdCorrecNameOrient(mOri,aDir,true);

	mICNM  = cInterfChantierNameManipulateur::BasicAlloc(aDir);
	//mOri   = mICNM->StdKeyOrient(mOri);
    mLFile = mICNM->StdGetListOfFile(aPat,1);


	if (mOri.rfind("Ori-") != string::npos)
	{
		if (! ELISE_fp::IsDirectory(mOri))
        	ELISE_fp::MkDir(mOri);	
	}
	else
	{
		if (! ELISE_fp::IsDirectory("Ori-"+mOri))
        	ELISE_fp::MkDir("Ori-"+mOri);	
	
	}

}

void cAppliSTRCam::Exe()
{

	for (auto aF : mLFile)
	{
		std::cout << aF << "\n";

		ELISE_fp aFIn(aF.c_str(),ELISE_fp::READ);
        char * aLine;


		/* Read internal calibration */
		int aNull=0;
        double      aFx=0;
        double      aFy=0;
		Pt2dr       aPP;
		Pt2di       aSz;

		aLine = aFIn.std_fgets();
        int aNb=sscanf(aLine,"%lf %i %lf", &aFx, &aNull, &aPP.x);
        ELISE_ASSERT((aNb==3),"Could not read 3  values");

		aLine = aFIn.std_fgets();
        aNb=sscanf(aLine,"%i %lf %lf", &aNull, &aFy, &aPP.y);
        ELISE_ASSERT((aNb==3),"Could not read 3  values");



		/* Read external calibration */
		ElMatrix<double> aR(3,3,1);
		Pt3dr            aT;

		aLine = aFIn.std_fgets();
		aLine = aFIn.std_fgets();
		
		//rotation
		aLine = aFIn.std_fgets();
        aNb=sscanf(aLine,"%lf %lf %lf", &aR(0,0), &aR(0,1), &aR(0,2));
        ELISE_ASSERT((aNb==3),"Could not read 3  values");

		aLine = aFIn.std_fgets();
        aNb=sscanf(aLine,"%lf %lf %lf", &aR(1,0), &aR(1,1), &aR(1,2));
        ELISE_ASSERT((aNb==3),"Could not read 3  values");

		aLine = aFIn.std_fgets();
        aNb=sscanf(aLine,"%lf %lf %lf", &aR(2,0), &aR(2,1), &aR(2,2));
        ELISE_ASSERT((aNb==3),"Could not read 3  values");

		//translation
		aLine = aFIn.std_fgets();
        aNb=sscanf(aLine,"%lf %lf %lf", &aT.x, &aT.y, &aT.z);
        ELISE_ASSERT((aNb==3),"Could not read 3  values");

		//img size
		aLine = aFIn.std_fgets();
        aNb=sscanf(aLine,"%i %i", &aSz.x, &aSz.y);
        ELISE_ASSERT((aNb==2),"Could not read 2  values");

	

        aFIn.close();

		std::size_t pos = aF.find(".camera"); 
		std::string aImN = aF.substr(0,pos);

		std::cout << aImN <<  " F " << (aFx+aFy)/2 << " aPP=" << aPP << " Sz=" << aSz << aT << "\n";


		/* Save to xml files */
		SaveCal(aImN, (aFx+aFy)/2, aPP, aSz);
		SaveOC(aImN, aR, aT);




	}
}

void cAppliSTRCam::SaveCal(std::string& aImN, double aF, Pt2dr& aPP, Pt2di& aSz)
{

	cCalibrationInternConique aCIO = StdGetObjFromFile<cCalibrationInternConique>
                (
                    Basic_XML_MM_File("Template-Calib-Basic.xml"),
                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                    "CalibrationInternConique",
                    "CalibrationInternConique"
                );
	aCIO.PP()   = aPP ;
    aCIO.F()    = aF ;
    aCIO.SzIm() = aSz ; //SfmInit convention
    aCIO.CalibDistortion()[0].ModRad().Val().CDist() = Pt2dr(0,0);

    MakeFileXML(aCIO,mICNM->StdNameCalib(mOri,aImN));

}

void cAppliSTRCam::SaveOC(std::string& aImN, ElMatrix<double>& aR, Pt3dr& aTr)
{
    /* internal calibration */
    std::string aCalibName = mICNM->StdNameCalib(mOri,aImN);


    /* external */ 
    std::string aKeyOri = mICNM->StdKeyOrient(mOri);
    std::string aFileExterne = mICNM->NameOriStenope(aKeyOri, aImN);
    std::cout << aFileExterne <<  " " << aCalibName << "\n";

    cOrientationExterneRigide aExtern;

    ElMatrix<double> aRotMM = aR;
    Pt3dr            aTrMM  = aTr;


    {
        aExtern.Centre() = aTrMM;
        aExtern.IncCentre() = Pt3dr(1,1,1);


        cTypeCodageMatr aTCRot;
        aTCRot.L1() = Pt3dr(aRotMM(0,0),aRotMM(0,1),aRotMM(0,2));   
        aTCRot.L2() = Pt3dr(aRotMM(1,0),aRotMM(1,1),aRotMM(1,2));   
        aTCRot.L3() = Pt3dr(aRotMM(2,0),aRotMM(2,1),aRotMM(2,2));   
        aTCRot.TrueRot() = true;

        cRotationVect aRV;
        aRV.CodageMatr() = aTCRot;
        aExtern.ParamRotation() = aRV;

        cOrientationConique aOC;
        aOC.ConvOri().KnownConv().SetVal(eConvApero_DistM2C);
        aOC.Externe() = aExtern;
        aOC.FileInterne().SetVal(aCalibName);

        MakeFileXML(aOC,aFileExterne);
    }

}

int CPP_Strecha2MM(int argc, char** argv)
{

	cAppliSTRCam anApp(argc,argv);


	anApp.Exe();


	return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

