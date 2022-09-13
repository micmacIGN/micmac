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

#include "../../MMVII/ExternalInclude/Eigen/Dense"

class cAppliColmap
{

	public:
		cAppliColmap(int argc, char** argv);
		void ToColmap();

	private:
		void ReadImToId(const std::string);

		std::string mPat;
		std::string mOri;
		std::string mOutFile;
		int         mCamId;
		Pt3dr       mOffset;

		std::map<std::string,int> mImToId;

		cInterfChantierNameManipulateur * mICNM;

};

cAppliColmap::cAppliColmap(int argc, char** argv) :
	mOutFile("images.txt"),
	mOffset(0,0,0)
{
	std::string aDir;
	std::string aFileList;

    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aDir,"Working dir. If inside put ./")
				   << EAMC(mPat,"Pattern of images that interests you")
				   << EAMC(aFileList,"List of all images (bundler style)")
		           << EAMC(mOri,"Orientation directory")
		           << EAMC(mCamId,"Camera id in Colmap db"),
        LArgMain() << EAM(mOutFile,"Out",true,"Output filename, Def=images.txt" )
		           << EAM(mOffset,"Offset",true,"X,Y,Z offset" )
    );

	#if (ELISE_windows)
        replace( aDir.begin(), aDir.end(), '\\', '/' );
    #endif

    mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

	StdCorrecNameOrient(mOri,aDir);

	ReadImToId(aFileList);
}

/* Read all images from the list 
 * the order in the list corresponds to the id in Colmap database 
 * the list was exported with colmap_import_features_batch  */
void cAppliColmap::ReadImToId(const std::string aListFile)
{
    ELISE_fp aFIn(aListFile.c_str(),ELISE_fp::READ);
    char * aLine;


	int aId=1;
    while ((aLine = aFIn.std_fgets()))
    {

        char aName[50];

        int aNb=sscanf(aLine,"%s", aName);

        ELISE_ASSERT((aNb==1),"Could not 1 value");


		mImToId[aName] = aId;
		//std::cout << string(aName).size() << "\n";		
		aId++;
    }
    aFIn.close();
    delete aLine;

}

void cAppliColmap::ToColmap()
{
	// save to Colmap images.txt format
	std::ofstream pFileIm(mOutFile, std::ios::trunc);
    pFileIm.precision(15);
	
	std::list<std::string>  aPatList = mICNM->StdGetListOfFile(mPat);

	std::string aKeyOri = mICNM->StdKeyOrient(mOri);
	std::cout << aKeyOri << "\n";

    for (auto aIm : aPatList)
    {

		std::string aNF = mICNM->NameOriStenope(aKeyOri,aIm);

    	if (ELISE_fp::exist_file(aNF))
        {

            /* MicMac */
            Pt3dr aC = StdGetObjFromFile<Pt3dr>
                    (
                        aNF,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        "Centre",
                        "Pt3dr"
                    );
			aC = aC - mOffset;

            cOrientationConique * aCO = OptionalGetObjFromFile_WithLC<cOrientationConique>
                                 (
                                       0,0,
                                       aNF,
                                       StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                       "OrientationConique",
                                       "OrientationConique"
                                 );
			cRotationVect       aRV  = aCO->Externe().ParamRotation();
            ElMatrix<double>    aRot = ElMatrix<double>::Rotation(aRV.CodageMatr().Val().L1(),
                                                                  aRV.CodageMatr().Val().L2(),
                                                                  aRV.CodageMatr().Val().L3());

			/* Colmap */
			Eigen::Matrix3d aMatTmp = Eigen::MatrixXd::Zero(3,3);
			for (int aK1=0; aK1<3; aK1++)
			{
				for (int aK2=0; aK2<3; aK2++)
					aMatTmp(aK1,aK2) = aRot(aK2,aK1);
			}

			Eigen::Quaterniond quat(aMatTmp);
			Eigen::Vector4d Qvec = Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
			
			Eigen::Vector3d Tvec = Eigen::Vector3d(aC.x, aC.y, aC.z);
			Tvec = - (aMatTmp * Tvec);

			std::cout << aIm << " " << aIm.size() << "\n";

			if (DicBoolFind(mImToId,aIm))
			{
				pFileIm << mImToId[aIm] << " " << Qvec(0) << " " << Qvec(1) << " " << Qvec(2) << " " << Qvec(3) << " " << Tvec(0) << " " << Tvec(1) << " " << Tvec(2) << " " << mCamId << " " << aIm << "\n";
              pFileIm << "\n";
			}
			else
				std::cout << aIm << "Not found in  the list file" << "\n";

		}
	}
	pFileIm.close();	
}

class cAppliBundler
{
	public:
		cAppliBundler(int argc, char** argv);

		void FromBundler();
		void ToBundler();

		void ReadList();	
		bool ReadPoses();	

	private:
		 cInterfChantierNameManipulateur * mICNM;
		
		 std::string mNameFile;
		 std::string mCCListAllFile;
		 std::string mCoordsFile;
		 std::string mOri;
		 std::string mSH;
		 std::string mHomExp;
    	 std::vector<std::string> mNameList;//FromBundler
		 std::list<std::string>   mLFile;//ToBundler, should be one var but too lazy to change
		 std::string mNameConvHom;

		 bool ConvHomMM2Bund;

		 std::map<int,Pt2di>      mCamSz;
		 Pt2di                    mUniqueSz;//to serve FromBundler		 

		 template <typename T>
         void FileReadOK(FILE *fptr, const char *format, T *value);

		 bool ReadCoords();
		 void IntCamSz();//to replace ReadCoords in FromBundler
         bool ReadCoordsOneCam(FILE *fptr);
		 void ConvertDR2MM(std::vector<double>& aDR,double& aFoc);
		 void ConvertHom2MM();

		 void SaveOC(ElMatrix<double>& aRotZ, ElMatrix<double>& aR, Pt3dr& aTr, 
				     double aFoc, std::vector<double>& aDr1Dr2, Pt2dr& aPP,int& aCamId);

		 void ConvRotTr(const ElMatrix<double>& dR,Pt3dr& aTr, ElMatrix<double>& aRot, bool IsDirect);
};

cAppliBundler::cAppliBundler(int argc, char** argv) :
	mNameFile(""),
	mCCListAllFile(""),
	mOri("Ori-Bundler/"),
	mSH(""),
	mHomExp("dat"),
	mNameConvHom("-BundlerFormat"),
	ConvHomMM2Bund(false),
	mUniqueSz(0,0)
{

	std::string aDir;
	bool aExpTxt=false;


	ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aDir,"Working dir. If inside put ./"),
        LArgMain() << EAM(mNameFile,"b",true,"bundler.txt, (if FromBundler or ToBundler)" )
                   << EAM(mCCListAllFile,"l",true,"list.txt, (if FromBundler or ToBundler)")
                   << EAM(mCoordsFile,"c",true,"coords.txt ")
                   << EAM(mUniqueSz,"UniSz",true,"Unique PP, (if FromBundler)")
                   << EAM(mOri,"Ori",true,"Orientation directory withoout Ori-, (if FromBundler or ToBundler)")
                   << EAM(mSH,"SH",true,"Homol Postfix")
                   << EAM(aExpTxt,"ExpTxt",true,"Homol in ASCI?")
                   << EAM(ConvHomMM2Bund,"ConvHom",true,"Convert homol to bundler format, Def=false")
    );


	aExpTxt ? mHomExp="txt" : mHomExp="dat";
	
    #if (ELISE_windows)
        replace( aDir.begin(), aDir.end(), '\\', '/' );
    #endif
	
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

	if (! ELISE_fp::IsDirectory(mOri))
            ELISE_fp::MkDir(mOri);

	//list of the images (id coherent with budle file)
    ReadList();

}

template <typename T>
void cAppliBundler::FileReadOK(FILE *fptr, const char *format, T *value)
{
    int OK = fscanf(fptr, format, value);
    if (OK != 1)
        ELISE_ASSERT(false, "cAppliImportSfmInit::FileReadOK")

}

void cAppliBundler::ConvertHom2MM()
{
    std::string aKey = "NKS-Assoc-CplIm2Hom@" + mSH + "@" + mHomExp;
    std::string aKeyNew = "NKS-Assoc-CplIm2Hom@" + mSH+mNameConvHom + "@" + "txt";

	for (int aK1=0; aK1<int(mNameList.size()); aK1++)
	{
		for (int aK2=0; aK2<int(mNameList.size()); aK2++)
		{
			if (aK1!=aK2)
			{
				//  recover tie-pts 
				std::string aName1 = mNameList.at(aK1);
				std::string aName2 = mNameList.at(aK2);
    			std::string aN12  = mICNM->Assoc1To2(aKey,aName1,aName2,true);

				if (ELISE_fp::exist_file(aN12))
				{

					ElPackHomologue aPack = ElPackHomologue::FromFile(aN12);
					ElPackHomologue aPackNew;
			
					for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    				{
						ElCplePtsHomologues aP(Pt2dr(itP->P1().x,-itP->P1().y), Pt2dr(itP->P2().x,-itP->P2().y), 1.0);
						aPackNew.Cple_Add(aP);
					}
					
    				std::string aN12New  = mICNM->Assoc1To2(aKeyNew,aName1,aName2,true);
					aPackNew.StdPutInFile(aN12New);

				}
			}
		}

	}

}

/* Read the list */
void cAppliBundler::ReadList()
{

    {
        ELISE_fp aFIn(mCCListAllFile.c_str(),ELISE_fp::READ);
        char * aLine;


        while ((aLine = aFIn.std_fgets()))
        {

            char aName[50];
            int         aNull=0;
            double      aF=0;

            int aNb=sscanf(aLine,"%s %i %lf", aName, &aNull, &aF);


            ELISE_ASSERT((aNb==3) || (aNb==1),"Could not read 3 or 1 values");

            mNameList.push_back(NameWithoutDir(aName));

        }
        aFIn.close();
        delete aLine;
    }

}

void cAppliBundler::IntCamSz()
{
	int Nb = int(mNameList.size()); 
	for (int aK=0; aK<Nb; aK++)	
		mCamSz[aK] = mUniqueSz;
	
}

//I'm only interested in camera size
bool cAppliBundler::ReadCoordsOneCam(FILE *fptr)
{

	int aCamId; 
    
	char line[50];

    for (int aIt=0; aIt<2; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    FileReadOK(fptr, "%i,", &aCamId);

    for (int aIt=0; aIt<5; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    int aNbKey;
    FileReadOK(fptr, "%i", &aNbKey);


    for (int aIt=0; aIt<3; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }

	Pt2dr aPP;
    FileReadOK(fptr, "%lf,", &aPP.x);


    for (int aIt=0; aIt<2; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    FileReadOK(fptr, "%lf,", &aPP.y);
	mCamSz[aCamId] = Pt2di(aPP.x*2,aPP.y*2);
	//std::cout << "mCamSz[aCamId] " << mCamSz[aCamId] << " " << aCamId << "\n";

    double aF;
    for (int aIt=0; aIt<2; aIt++)
    {
        FileReadOK(fptr, "%s", line);
    }
    FileReadOK(fptr, "%lf", &aF);

	int   aPtId;
    Pt2dr aPt;
    Pt2di aIgnr;
    Pt3di aRGB;
    for (int aK=0; aK<aNbKey; aK++)
    {
        int OK = std::fscanf(fptr,"%i %lf %lf %i %i %i %i %i\n",&aPtId,&aPt.x,&aPt.y,&aIgnr.x,&aIgnr.y,&aRGB.x,&aRGB.y,&aRGB.z);
        if (OK)
        {
			//do nothing
        }
        else
        {
            std::cout << "cAppliBundler::ReadCoordsOneCam could not read a line" << "\n";
            return EXIT_FAILURE;
        }

    }


    return EXIT_SUCCESS;
}

bool cAppliBundler::ReadCoords()
{

    {
        FILE* fptr = fopen(mCoordsFile.c_str(), "r");
        if (fptr == NULL) {
          return false;
        };


        while (!std::feof(fptr)  && !ferror(fptr))
        {

            ReadCoordsOneCam(fptr);
        }

        fclose(fptr);

		return EXIT_SUCCESS;
    }
}

bool cAppliBundler::ReadPoses()
{
	//read the bundle file
	ELISE_fp aFIn(mNameFile.c_str(),ELISE_fp::READ);
	char * aLine;

	//read the bundle comment
	aLine = aFIn.std_fgets();

	//camera number and tie pts number
	int aNbCam;
	int aNbTPt;

	aLine = aFIn.std_fgets();
	int aNb=sscanf(aLine,"%i %i", &aNbCam, &aNbTPt);
    
	ELISE_ASSERT((aNb==2),"Could not read 2 values");
	std::cout << "[Bundler2MM] Nb cams:" << aNbCam << "\n";

	double aFoc;
	std::vector<double> aK1K2(2);
	ElMatrix<double> aRot(3,3);
	Pt3dr            aTr;


	ElMatrix<double> aRotZ(3,3,0);
    aRotZ(0,0) = 1;
    aRotZ(1,1) =-1;
    aRotZ(2,2) =-1;

	if (aNbCam== int(mNameList.size()))
	{
		//read camera at a time and stop after reading aNbCam
		int aCount=0;
		
		while ((aLine = aFIn.std_fgets()) && (aCount<aNbCam))
		{
			//focal, k1, k2
			aNb=sscanf(aLine,"%lf %lf %lf", &aFoc, &aK1K2.at(0), &aK1K2.at(1));
			ELISE_ASSERT((aNb==3),"Could not read 3 values");

			//std::cout << "k1,k2: " << aK1K2 << " f: " << aFoc << "\n";

			//rotation matrix (world to camera convention)
			if ((aLine = aFIn.std_fgets()))
			{
				aNb=sscanf(aLine,"%lf %lf %lf", &aRot(0,0), &aRot(0,1), &aRot(0,2));


				ELISE_ASSERT((aNb==3),"Could not read 3 rotational matrix values");
				//std::cout << "Nb " << aNb << "Rot: " << aRot(0,0) << " " <<  aRot(0,1) << " " <<  aRot(0,2) << "\n";

			}
			if ((aLine = aFIn.std_fgets()))
            {
                aNb=sscanf(aLine,"%lf %lf %lf", &aRot(1,0), &aRot(1,1), &aRot(1,2)) ;

                ELISE_ASSERT((aNb==3),"Could not read 3 rotational matrix values");
                //std::cout << "Nb " << aNb << "Rot: " << aRot(1,0) << " " <<  aRot(1,1) << " " <<  aRot(1,2) << "\n";

            }
			if ((aLine = aFIn.std_fgets()))
            {
                aNb=sscanf(aLine,"%lf %lf %lf", &aRot(2,0), &aRot(2,1), &aRot(2,2)) ;

                ELISE_ASSERT((aNb==3),"Could not read 3 rotational matrix values");
                //std::cout << "Nb " << aNb << "Rot: " << aRot(0,0) << " " <<  aRot(0,1) << " " <<  aRot(0,2) << "\n";

            }

			//translation  (world to camera convention)
			
			if ((aLine = aFIn.std_fgets()))
			{
				aNb=sscanf(aLine,"%lf %lf %lf", &aTr.x, &aTr.y, &aTr.z);
				ELISE_ASSERT((aNb==3),"Could not read 3 translation values");

				//std::cout << "Tr: " << aTr << "\n";
			}



			/* Save to MicMac format */
			Pt2dr aPP(0,0);

			if (DicBoolFind(mCamSz,aCount))
				SaveOC(aRotZ, aRot, aTr, aFoc, aK1K2, aPP, aCount);
			else
			{
				std::cout << "[No orientation in SfmInit for " << mNameList.at(aCount) << " ] \n";
				//std::cout << "poubele/"+mNameList.at(aCount) << "\n";
				//ELISE_fp::MvFile("./"+mNameList.at(aCount),"./poubele/"+mNameList.at(aCount));
			}
		

			aCount++;
		}

        aFIn.close();
	
		return EXIT_SUCCESS;
	}
	else
		return EXIT_FAILURE;

}

void cAppliBundler::SaveOC(ElMatrix<double>& aRotZ, ElMatrix<double>& aR, Pt3dr& aTr, double aFoc, std::vector<double>& aDr1Dr2, Pt2dr& aPP,int& aCamId)
{
	if (aFoc!=0)
	{
		cOrientationConique aOC;
    
		std::string aFileInterne = mOri + NameWithoutDir(mICNM->StdNameCalib("Test",NameWithoutDir(mNameList.at(aCamId))));
        std::string aFileExterne = mOri + "Orientation-" + mNameList.at(aCamId) + ".xml";
    
		//interne
		cCalibrationInternConique aCIO = StdGetObjFromFile<cCalibrationInternConique>
                   (
                       Basic_XML_MM_File("Template-Calib-Basic.xml"),
                       StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                       "CalibrationInternConique",
                       "CalibrationInternConique"
                   );
    
		//pespective center in the middle of the image
        aCIO.PP() = Pt2dr(double(mCamSz[aCamId].x)/2,double(mCamSz[aCamId].y)/2);
        aCIO.F()  = aFoc;
        aCIO.SzIm() = mCamSz[aCamId];
  		
	   	/*cModNoDist aNoDist;
	    aCIO.CalibDistortion()[0].ModNoDist().SetVal(aNoDist);
	    aCIO.CalibDistortion()[0].ModRad().SetNoInit();*/
	
        aCIO.CalibDistortion()[0].ModRad().Val().CDist() = aCIO.PP();
		if (1)
		{
			ConvertDR2MM(aDr1Dr2,aFoc);
			aCIO.CalibDistortion()[0].ModRad().Val().CoeffDist() = aDr1Dr2;
		}
    
        MakeFileXML(aCIO,aFileInterne);	
		
		
		//externe
		cOrientationExterneRigide aExtern;
   

		ConvRotTr(aRotZ, aTr, aR, true);
		

		if ((mNameList.at(aCamId)=="130243089_80519c9d25_o.jpg") || (mNameList.at(aCamId)=="148323927_a290b0c5ab_o.jpg"))
		{
				std::cout << aFileExterne << "\n";
				std::cout << aR(0,0) << " " << aR(0,1) << " " << aR(0,2) << "\n"
						  << aR(1,0) << " " << aR(1,1) << " " << aR(1,2) << "\n"
						  << aR(2,0) << " " << aR(2,1) << " " << aR(2,2) << "\n"
						  << aTr << "\n";
				getchar();
		}


		/* There are images in Bundler solution that are NAN but exist in SfmInit */
        if (! (std::isnan(aTr.x) || std::isnan(aTr.y) || std::isnan(aTr.z)))
		{
			aExtern.Centre() = aTr;//aTrMM;
			aExtern.IncCentre() = Pt3dr(1,1,1);
			
        
			cTypeCodageMatr aTCRot;
			aTCRot.L1() = Pt3dr(aR(0,0),aR(0,1),aR(0,2));   
			aTCRot.L2() = Pt3dr(aR(1,0),aR(1,1),aR(1,2));   
			aTCRot.L3() = Pt3dr(aR(2,0),aR(2,1),aR(2,2));   
			aTCRot.TrueRot() = true;
        
			cRotationVect aRV;
			aRV.CodageMatr() = aTCRot;
			aExtern.ParamRotation() = aRV; 
        
			aOC.ConvOri().KnownConv().SetVal(eConvApero_DistM2C);
			aOC.Externe() = aExtern;	
			aOC.FileInterne().SetVal(aFileInterne);
        
			MakeFileXML(aOC,aFileExterne);
		}
	}
}

void cAppliBundler::ConvertDR2MM(std::vector<double>& aDR,double& aFoc)
{
	for (int aCoef=0; aCoef<int(aDR.size()); aCoef++)
	{
		aDR.at(aCoef) /= std::pow(aFoc,(aCoef+1)*2);

	}
}

/*       
 * * Conventions in MicMac and Bundler :
 *
 *          X - world coordinates
 *          P - camera coordinates
 *          C - position of the camera in the world coordinates
 *          
 *         
 *            MM: X = R * P + C
 *            B : P = R * X + t
 *         
 *            Bundler conv in MM
 *            Rm = R'b
 *            Cm = Rb*t
 *         
 *            therefore, X = Rm * P + Cm => R'b * P + Rb*t
 *         
 * 
 *  Remark: conv MM is Camera2Monde and Bumnder Monde2Camera
 *
 *  Remark: Bundler Z points away from the scene therefore dot product with dR
 *    1  0            0       
 *   0 cos teta=-1   -sin teta=0
 *   0 sin teta=0   cos teta=-1
 *
 * */
void cAppliBundler::ConvRotTr(const ElMatrix<double>& dR, Pt3dr& aTr, ElMatrix<double>& aRot, bool IsDirect)
{



	/* From Bundler */
	if (IsDirect)
	{
		aTr  = -(aRot * aTr);	
		aRot = dR * aRot.transpose() ;
	}
	/* To Bundler */
	else
	{

		aTr = dR * (aRot * (-aTr)) ;
		aRot = aRot.transpose() * dR;		
	}

}


void cAppliBundler::FromBundler()
{


    //read camera sizes (pas top)
    ReadCoords();
//	IntCamSz(); //replacing ReadCoors

    if (ReadPoses())
        std::cout << "[Bundler2MM] Poses done!" << "\n";

    if (ConvHomMM2Bund)
    {
        ConvertHom2MM();
        std::cout << "[Bundler2MM] Tie-points done!" << "\n";
    }
	

}

void cAppliBundler::ToBundler()
{

	/* (1) iterate over images
	 * (2) read MM ori
	 * (3) convert mm ori to bundler
	 * (4) save ori bundler
	 * */


	std::fstream aOut;
    aOut.open(mNameFile.c_str(), std::istream::out);
	
	/* Nb of cameras */
	int aNbCam = int(mNameList.size());
	int aNbTPt = 0;

	aOut << "# Bundle file v0.3\n";	
	aOut << aNbCam << " " << aNbTPt << "\n";



	ElMatrix<double> aRotZ(3,3,0);
    aRotZ(0,0) = 1;
    aRotZ(1,1) =-1;
    aRotZ(2,2) =-1;

	std::string aKeyOri = mICNM->StdKeyOrient(mOri);

	//for (auto aIm : mLFile)
	for (auto aIm : mNameList)
	{


		/* externe */	
		std::string aNF = mICNM->NameOriStenope(aKeyOri,aIm);

        if (ELISE_fp::exist_file(aNF))
        {

			/* interne */
			std::string aNC = mICNM->StdNameCalib(mOri,aIm);


			cCalibrationInternConique aCIO = StdGetObjFromFile<cCalibrationInternConique>
                                (
                                    aNC,
                                    StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                    "CalibrationInternConique",
                                    "CalibrationInternConique"
                                );

			//no distortions taken into account
			aOut << aCIO.F() << " 0 0" << "\n";
	
            Pt3dr aC = StdGetObjFromFile<Pt3dr>
                    (
                        aNF,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        "Centre",
                        "Pt3dr"
                    );


			cOrientationConique * aCO = OptionalGetObjFromFile_WithLC<cOrientationConique>
                                 (
                                       0,0,
                                       aNF,
                                       StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                       "OrientationConique",
                                       "OrientationConique"
                                 );
        	cRotationVect 		aRV  = aCO->Externe().ParamRotation();
			ElMatrix<double>	aRot = ElMatrix<double>::Rotation(aRV.CodageMatr().Val().L1(),
							                                      aRV.CodageMatr().Val().L2(),
																  aRV.CodageMatr().Val().L3());
			

			ConvRotTr(aRotZ,aC,aRot,false);


			aOut << aRot(0,0) << " " << aRot(0,1) << " " << aRot(0,2) << "\n"
				 << aRot(1,0) << " " << aRot(1,1) << " " << aRot(1,2) << "\n"
				 << aRot(2,0) << " " << aRot(2,1) << " " << aRot(2,2) << "\n"
				 << std::setprecision(10) << aC.x << " " << aC.y << " " << aC.z << "\n";

        }
		else
			aOut << "0 0 0" << "\n"
				 << "0 0 0" << "\n"
				 << "0 0 0" << "\n"
				 << "0 0 0" << "\n"
				 << "0 0 0" << "\n";
	
	
	}
	aOut.close();

}

int CPP_Bundler2MM_main(int argc, char** argv)
{
	cAppliBundler anApp(argc,argv);
	anApp.FromBundler();

	return EXIT_SUCCESS;
}

int CPP_MM2Bundler_main(int argc, char** argv)
{
    cAppliBundler anApp(argc,argv);
    anApp.ToBundler();

    return EXIT_SUCCESS;
}

int CPP_MM2Colmap_main(int argc, char** argv)
{
    cAppliColmap anApp(argc,argv);
	anApp.ToColmap();
    
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
