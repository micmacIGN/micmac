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

class cAppliBundler
{
	public:
		cAppliBundler(int argc, char** argv);

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
    	 std::vector<std::string> mNameList;
		 std::string mNameConvHom;

		 bool ConvHomMM2Bund;

		 std::map<int,Pt2di>      mCamSz;

		 template <typename T>
         void FileReadOK(FILE *fptr, const char *format, T *value);

		 bool ReadCoords();
         bool ReadCoordsOneCam(FILE *fptr);
		 void ConvertDR2MM(std::vector<double>& aDR,double& aFoc);
		 void ConvertHom2MM();

		 void SaveOC(ElMatrix<double>& aR, Pt3dr& aTr, double aFoc, std::vector<double>& aDr1Dr2, Pt2dr& aPP,int& aCamId);
};

cAppliBundler::cAppliBundler(int argc, char** argv) :
	mNameFile(""),
	mCCListAllFile(""),
	mOri("Ori-Bundler/"),
	mSH(""),
	mHomExp("dat"),
	mNameConvHom("-BundlerFormat"),
	ConvHomMM2Bund(false)
{

	std::string aDir;
	bool aExpTxt=false;

	ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aDir,"Working dir. If inside put ./")
				   << EAMC(mNameFile,"bundler.txt", eSAM_IsExistFile)
				   << EAMC(mCCListAllFile,"list.txt",eSAM_IsExistFile)
				   << EAMC(mCoordsFile,"coords.txt",eSAM_IsExistFile),
        LArgMain() << EAM(mOri,"Out",true,"Output orientation directory") 
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

	//read camera sizes (pas top)
	ReadCoords();

	if (ReadPoses())
		std::cout << "[Bundler2MM] Poses done!" << "\n";

	if (ConvHomMM2Bund)
	{
		ConvertHom2MM();
		std::cout << "[Bundler2MM] Tie-points done!" << "\n";
	}
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
				SaveOC(aRot, aTr, aFoc, aK1K2, aPP, aCount);
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

void cAppliBundler::SaveOC(ElMatrix<double>& aR, Pt3dr& aTr, double aFoc, std::vector<double>& aDr1Dr2, Pt2dr& aPP,int& aCamId)
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
	
        //aCIO.CalibDistortion()[0].ModRad().Val().CDist() = aCIO.PP();
		if (0)
		{
			ConvertDR2MM(aDr1Dr2,aFoc);
			aCIO.CalibDistortion()[0].ModRad().Val().CoeffDist() = aDr1Dr2;
		}
    
        MakeFileXML(aCIO,aFileInterne);	
		
		
		//externe
		cOrientationExterneRigide aExtern;
   
	    /* Conventions in MicMac and Bundler :
		 * X - world coordinates
		 * P - camera coordinates
		 * C - position of the camera in the world coordinates
		 * 
		 *
		 *   MM: X = R * P + C
		 *   B : P = R * X + t
		 *
		 *   Bundler conv in MM
		 *   Rm = R'b
		 *   Cm = Rb*t
		 *
		 *   therefore, X = Rm * P + Cm => R'b * P + Rb*t
		 *
		 * */	
		//conv MM is Camera2Monde and Bumnder Monde2Camera
		ElMatrix<double> aRotMM = aR.transpose();
		Pt3dr            aTrMM  = - (aR*aTr);	
	
		std::cout << -(aR*aTr) << " " << "\n";

		/* There are images in Bundler solution that are NAN but exist in SfmInit */
		if (! (isnan(aTrMM.x) || isnan(aTrMM.y) || isnan(aTrMM.z)))
		{
			aExtern.Centre() = aTrMM;
			aExtern.IncCentre() = Pt3dr(1,1,1);
			
        
			cTypeCodageMatr aTCRot;
			aTCRot.L1() = Pt3dr(aRotMM(0,0),-aRotMM(0,1),-aRotMM(0,2));     //  1  0            0            conv Bundler: Z points away from the scene
			aTCRot.L2() = Pt3dr(aRotMM(1,0),-aRotMM(1,1),-aRotMM(1,2));   // 0 cos teta=-1   -sin teta=0
			aTCRot.L3() = Pt3dr(aRotMM(2,0),-aRotMM(2,1),-aRotMM(2,2));   // 0 sin teta=0   cos teta=-1
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

int CPP_Bundler2MM_main(int argc, char** argv)
{
	cAppliBundler anApp(argc,argv);

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
