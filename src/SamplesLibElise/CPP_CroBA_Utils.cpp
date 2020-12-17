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
#include "../uti_phgrm/TiepTri/MultTieP.h"

/* Read tie-pts 2D */
/* Read EO */
/* Read GCP */


class cAppliCroEIO
{
	public:
		cAppliCroEIO(int argc, char ** argv);

		void DoEOIO();
		bool DoTP();
		bool DoGCP();

	private:
		void SaveOC(std::string& aImN, ElMatrix<double>& aR, ElMatrix<double>& aTr);
        void SaveCal(std::string& aImN, double aF, Pt2dr& aPP, Pt2di& aSz);

		bool ReadTP(std::map<std::string,int> &aMapImToId,
				    std::map<int,std::vector<std::pair<Pt2dr,int>> * > &aMap);

        std::string mOri;
		double      mF;
		Pt2dr       mPP;
		Pt2di       mSz;

		std::string mTiePFile;
		std::string mGCPIm;

		std::string mIm1Name;

        cInterfChantierNameManipulateur* mICNM;

        std::list<std::string> mLFile;

};

cAppliCroEIO::cAppliCroEIO(int argc, char ** argv):
		mSz(Pt2di(0,0))
{
    //iterate over files and create Ori

    std::string aPat;
    std::string aDir;

    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aPat,"Pattern of files")
                   << EAMC(mOri,"Orientation directory"),
        LArgMain() << EAM(mF,"F",true,"Focal length")
		           << EAM(mPP,"PP",true,"Principal point")
				   << EAM(mTiePFile,"TP",true,"Tie-points file")
				   << EAM(mGCPIm,"GCPIm",true,"GCP image observations file")
				   << EAM(mIm1Name,"ImInit",true,"Initial image that will be copied")
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

	if (! EAMIsInit(&mIm1Name))
        mIm1Name = StdPrefix(*(mLFile.begin())) + ".tif";

	/* Create tif */
	for (auto aIm : mLFile)
	{
		std::string aImTif = StdPrefix(aIm) + ".tif";

		if (! ELISE_fp::exist_file(aImTif))
		{
			if (ELISE_fp::exist_file(mIm1Name))
				ELISE_fp::CpFile(mIm1Name,aImTif);
			else
				ELISE_fp::CpFile(aIm,aImTif);
		}		
	}
}

void cAppliCroEIO::DoEOIO()
{
	if (EAMIsInit(&mF))
    {	
		/* Save interior calibration */
		mSz.x = 2*mPP.x;
        mSz.y = 2*mPP.y;
    
        
		SaveCal(mIm1Name, mF, mPP, mSz);//hypothesis that all images share the same IO
    
		ElMatrix<double> NullVec(1,3,0);
    
		ElMatrix<double> K(3,3,0);
		K(0,0) = K(1,1) = mF;
		K(0,2) = mPP.x;
		K(1,2) = mPP.y;
		K(2,2) = 1.0;
    
		ElMatrix<double> KInv(3,3,0);
		KInv = gaussj(K);
		KInv.self_transpose();
    
		/* Save exterior calibration */
		for (auto aF : mLFile)
        {
    
			std::cout << aF << "\n";
    
            ELISE_fp aFIn(aF.c_str(),ELISE_fp::READ);
            char * aLine;
    
			ElMatrix<double> aP (4,3,1);
			ElMatrix<double> aP_(4,3,0);
			ElMatrix<double> aT(1,3,0);
			ElMatrix<double> aR(3,3,0);
			ElMatrix<double> aC(1,3,0);
    
    
    
			aLine = aFIn.std_fgets();
			int aNb=sscanf(aLine,"%lf %lf %lf %lf", &aP(0,0), &aP(1,0), &aP(2,0), &aP(3,0));
            ELISE_ASSERT((aNb==4),"Could not read 4  values");
    
			aLine = aFIn.std_fgets();
			aNb=sscanf(aLine,"%lf %lf %lf %lf", &aP(0,1), &aP(1,1), &aP(2,1), &aP(3,1));
            ELISE_ASSERT((aNb==4),"Could not read 4  values");
    
			aLine = aFIn.std_fgets();
			aNb=sscanf(aLine,"%lf %lf %lf %lf", &aP(0,2), &aP(1,2), &aP(2,2), &aP(3,2));
            ELISE_ASSERT((aNb==4),"Could not read 4  values");
    
			aFIn.close();
    
    
			/* Remove camera calibration */
			aP_ = KInv * aP; //P_ = K^-1 * P
    
    
			/* Translation */
			aT(0,0) = aP_(3,0);
			aT(0,1) = aP_(3,1);
			aT(0,2) = aP_(3,2);
    
    
			
			/* Rotation */
			for (int aK1=0; aK1<3; aK1++)
			{
				for (int aK2=0; aK2<3; aK2++)
				{
					aR(aK1,aK2) = aP_(aK1,aK2);
				}
			}	
    
    
			/* Perspective center */
			aC = NullVec-(gaussj(aR) * aT);
			
			/*std::cout << "P=" << "\n";
			for (int aJ=0; aJ<3; aJ++)
			{
				for (int aI=0; aI<4; aI++)
					std::cout << " " << aP(aI,aJ) << " ";
				std::cout << " \n";
			}
			std::cout << " \n";
			std::cout << "P_=" << "\n";
			for (int aJ=0; aJ<3; aJ++)
			{
				for (int aI=0; aI<4; aI++)
					std::cout << " " << aP_(aI,aJ) << " ";
				std::cout << " \n";
			}
			std::cout << " \n";*/
			//std::cout << "t=" << aT(0,0) << " " << aT(0,1) << " " << aT(0,2) << "\n";
			std::cout << "C=" << aC(0,0) << " " << aC(0,1) << " " << aC(0,2) << "\n";
    
    
			/* Save xml */
			std::string aCurIm = StdPrefix(aF)+".tif";
			SaveOC(aCurIm, aR, aC);
		}
	}
}

void cAppliCroEIO::SaveCal(std::string& aImN, double aF, Pt2dr& aPP, Pt2di& aSz)
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
    aCIO.SzIm() = aSz ; 
    aCIO.CalibDistortion()[0].ModRad().Val().CDist() = Pt2dr(0,0);

	std::cout << mICNM->StdNameCalib(mOri,aImN) << "\n";
    MakeFileXML(aCIO,mICNM->StdNameCalib(mOri,aImN));

}

void cAppliCroEIO::SaveOC(std::string& aImN, ElMatrix<double>& aR, ElMatrix<double>& aTr)
{
    /* internal calibration */
    std::string aCalibName = mICNM->StdNameCalib(mOri,aImN);


    /* external */
    std::string aKeyOri = mICNM->StdKeyOrient(mOri);
    std::string aFileExterne = mICNM->NameOriStenope(aKeyOri, aImN);
    std::cout << aFileExterne <<  " " << aCalibName << "\n";

    cOrientationExterneRigide aExtern;

    ElMatrix<double> aRotMM = aR;
    Pt3dr            aTrMM (aTr(0,0),aTr(0,1),aTr(0,2));


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

bool cAppliCroEIO::ReadTP(std::map<std::string,int> &aMapImToId,
				          std::map<int,std::vector<std::pair<Pt2dr,int>> * > &aMap)
{
    // read the file
	ELISE_fp fptr(mTiePFile.c_str(),ELISE_fp::READ);

	char * aLine;
	int aNb=0;
	
	aLine = fptr.std_fgets();
	bool DoRead = (aLine!= NULL ? true : false );
	while ( DoRead )
	{
		std::string aImName;
		int   		aIdPt;
		Pt2dr 		aPt;
		
		aNb=sscanf(aLine,"%d %lf %lf", &aIdPt, &aPt.x, &aPt.y );
		

		if (aNb==1)
		{
			aImName = aLine;

			// get image id
			int aImId = 0;
			bool ImInPat = false;
			if (DicBoolFind(aMapImToId,aImName))
			{
				aImId = aMapImToId[aImName];
				ImInPat = true;
			}

			aLine = fptr.std_fgets();
			aNb=sscanf(aLine,"%d %lf %lf", &aIdPt, &aPt.x, &aPt.y );
			
			//getchar();


			// collect tie points for that image
			while (aNb==3 && DoRead)
			{
				// if the image is in the pattern mLFile
				if (ImInPat)
				{
					// verify if the map element for that point id exists
					if (!DicBoolFind(aMap,aIdPt))
					{
						// initialise a map element that corresponds to that tie-point
						aMap[aIdPt] = new std::vector<std::pair<Pt2dr,int>>();
					}
					std::vector<std::pair<Pt2dr,int>> * aTPCur = aMap[aIdPt];
			    
					// update the map containing the tie points observations
					aTPCur->push_back(std::pair<Pt2dr,int>(aPt,aImId));

				}

				// next points
				if ((aLine = fptr.std_fgets()) != NULL)
				{			
					aNb=sscanf(aLine,"%d %lf %lf", &aIdPt, &aPt.x, &aPt.y );
				}
				else
				{
					DoRead = false;
				}
			}

			//std::cout << "ImName " << aImName << " " << aNb << "\n";
		}
		else
			std::cout << "Parsing impossible. The file should start with the file name" << aLine << "\n";

	}


	
	
	//int aNb=sscanf(aLine,"%lf %lf %lf %lf", &aP(0,0), &aP(1,0), &aP(2,0), &aP(3,0));
    //ELISE_ASSERT((aNb==4),"Could not read 4  values");
		
	return EXIT_SUCCESS;
}

bool cAppliCroEIO::DoTP()
{
	if (EAMIsInit(&mTiePFile))
	{
		std::cout << mTiePFile << "\n";
		

		// map to store the image name to id correspondences (id coherent with mLFile & aMulPts)
		std::map<std::string,int> aMapImNameToId;

		int Cnt=0;
		for (auto aIm : mLFile)
		{
			aMapImNameToId[StdPrefix(aIm)+".tif"] = Cnt;
			Cnt++;
		}

		//  map tp store the point Id (left) and a vector of its observations (x,y in the image and id of the image)
		std::map<int,std::vector<std::pair<Pt2dr,int>> * > aMapPtIdToObsImId;

		// read the file
		bool OKTP = ReadTP(aMapImNameToId,aMapPtIdToObsImId);
		ELISE_ASSERT(!OKTP,"cAppliCroEIO::DoTP()");

		int aNumPts = aMapPtIdToObsImId.size() ;


		// tracks
		std::map<int,std::vector<Pt2dr>* >  aPtIdTr;
        std::map<int,std::vector<int>* >    aPtIdPId;

		// initialise the tracks
		for (auto aPt : aMapPtIdToObsImId)
    	{
        	aPtIdTr[aPt.first] = new std::vector<Pt2dr>();
        	aPtIdPId[aPt.first] = new std::vector<int>();
    	}

		// fill-in the tracks
		int counter=0; 
		for (auto aPt : aMapPtIdToObsImId)
		{
			std::cout << "== counter ==" << counter++ << "/" << aNumPts  << " PtId" << aPt.first << "\n";
			std::vector<Pt2dr> * aVPtCur = aPtIdTr[aPt.first];
			std::vector<int> * aVIdCur   = aPtIdPId[aPt.first];

			for (auto aMes : *(aPt.second))
			{
				aVPtCur->push_back(aMes.first);
				aVIdCur->push_back(aMes.second);
				std::cout << "== " << aMes.first << "," << aMes.second << "\n";
			}

		}
		std::cout << "== filled ==" << "\n";

		// create the structure that will store and then save the tie points
		std::vector<std::string> aVFile;
		for (auto aIt : mLFile)
			aVFile.push_back(aIt);

		cSetTiePMul * aMulPts = new cSetTiePMul(0,&aVFile);

		std::cout << "before updating MulPts "  << "\n";
		//for (int aK=0; aK<int(aPtIdTr.size()); aK++)
		for (auto aPt : aMapPtIdToObsImId)
    	{
			// ignore single-image observation
			if (aPtIdPId[aPt.first]->size()>1)
			{
        		std::vector<float> aAttr;
        		aMulPts->AddPts(*aPtIdPId[aPt.first],*aPtIdTr[aPt.first],aAttr);
			}
    	}

		ELISE_fp::MkDir("Homol");
    	aMulPts->Save("Homol/PMul.txt");

		/*for (auto aPt : aMapPtIdToObsImId)
		{
			std::cout << "Pt " << aPt.first << "\n";

			for (auto aSaisi : *(aPt.second))
			{
				std::cout << aSaisi.first << " " << aSaisi.second << "\n";
			}
		}*/

	}
	else
		std::cout << "No tie-pts were extracted." << "\n";

	std::cout << "Num images:" << mLFile.size() << "\n";
	return EXIT_SUCCESS;
}

bool cAppliCroEIO::DoGCP()
{

	if (EAMIsInit(&mGCPIm))
	{

		std::cout << "Read GCPs" << mGCPIm << "\n";

		std::map<std::string,std::vector<std::pair<int,Pt2dr>>* > aMImIdPt;

		ELISE_fp fptr(mGCPIm.c_str(),ELISE_fp::READ);

		char *aLine;
		int aNb=0;

		aLine = fptr.std_fgets();
    	bool DoRead = (aLine!= NULL ? true : false );
    	while ( DoRead )
		{
			std::string aImName = aLine;
			std::cout << aImName << "\n";

			if (!DicBoolFind(aMImIdPt,aImName))
            {
                // initialise a map element that corresponds to that GCP
                aMImIdPt[aImName] = new std::vector<std::pair<int,Pt2dr>>();
            }
            std::vector<std::pair<int,Pt2dr>> * aGCPCur = aMImIdPt[aImName];

			int         aIdPt;
        	Pt2dr       aPt;

			// first point
			aLine = fptr.std_fgets();
			bool DoReadPt = (aLine!= NULL ? true : false );

			//case when no observation for an image at EOF or in-file
			if (!DoReadPt)
				DoRead = false;
			else
			{	
				aNb = sscanf(aLine,"%d %lf %lf", &aIdPt, &aPt.x, &aPt.y );
				if (aNb != 3)
					DoReadPt = false;
			}

			while (DoReadPt)
			{
				aNb=sscanf(aLine,"%d %lf %lf", &aIdPt, &aPt.x, &aPt.y );
				aGCPCur->push_back(std::pair<int,Pt2dr>(aIdPt,aPt));


				// next points
                if ((aLine = fptr.std_fgets()) != NULL)
                {
                    aNb=sscanf(aLine,"%d %lf %lf", &aIdPt, &aPt.x, &aPt.y );


					if (aNb==3)
						DoReadPt = true;
					else
						DoReadPt = false;
                }
                else
                {
                    DoRead = false;
                }
			}
		}

		// Save to MicMac structure 
		cSetOfMesureAppuisFlottants aSOMA;

		for (auto aIm : aMImIdPt)
		{
			cMesureAppuiFlottant1Im aMAF;

			aMAF.NameIm() = aIm.first;

			for (auto aPt : *(aIm.second))
			{
				cOneMesureAF1I aOMAF;
				aOMAF.NamePt() = ToString(aPt.first);
				aOMAF.PtIm() = aPt.second;

				aMAF.OneMesureAF1I().push_back(aOMAF);	
			}

			aSOMA.MesureAppuiFlottant1Im().push_back(aMAF);
		}
		MakeFileXML(aSOMA,StdPrefix(mGCPIm)+".xml");

	}

	return EXIT_SUCCESS;
}


int CPP_CroBA_ReadEO(int argc, char ** argv)
{

	cAppliCroEIO aAppliCBA(argc,argv);
	aAppliCBA.DoEOIO();
	aAppliCBA.DoTP();
	aAppliCBA.DoGCP();

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
