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
#include "spline.h"

class cIIP_Appli
{
	public :
		cIIP_Appli(int argc,char ** argv);
		std::vector<double> GetGps_X_FromFile(std::string GpsFile);
		std::vector<double> GetGps_Y_FromFile(std::string GpsFile);
		std::vector<double> GetGps_Z_FromFile(std::string GpsFile);
		std::vector<double> GetImgTMFromFile(std::string ImgTMFile);
		std::vector<std::string> GetImgNameFromFile(std::string ImgTMFile);
		std::vector<double> GetMjdFromGpsFile(std::string GpsFile);
		void WriteImTmInFile(std::string aOutputNameFile,std::vector<std::string> & aVImName,std::vector<Pt3dr> & aPos,bool aFormat);
		std::vector<Pt3dr> ConvertAxial2Pos(std::vector<double> aP_x, std::vector<double> aP_y, std::vector<double> aP_z);
	
	private :
		std::string mDir;
		std::string mGpsFile;
		std::string mTMFile;
};

std::vector<double> cIIP_Appli::GetMjdFromGpsFile(std::string GpsFile)
{
	std::vector<double> aVMjdGps;
	
	cDicoGpsFlottant aDico = StdGetFromPCP(GpsFile,DicoGpsFlottant);
	std::list<cOneGpsDGF> aOneGpsDGFList = aDico.OneGpsDGF();
		
	for (std::list<cOneGpsDGF>::iterator itP=aOneGpsDGFList.begin(); itP != aOneGpsDGFList.end(); itP ++)
	{
		double aMjdGps = itP->TimePt();
		aVMjdGps.push_back(aMjdGps);
	}
	
	return aVMjdGps;
	
}

std::vector<double> cIIP_Appli::GetGps_X_FromFile(std::string GpsFile)
{
	std::vector<double> aV_X;

    cDicoGpsFlottant aDico = StdGetFromPCP(GpsFile,DicoGpsFlottant);
	std::list<cOneGpsDGF> aOneGpsDGFList = aDico.OneGpsDGF();
		
	for (std::list<cOneGpsDGF>::iterator itP=aOneGpsDGFList.begin(); itP != aOneGpsDGFList.end(); itP ++)
	{
		double aPt_x = itP->Pt().x;
		aV_X.push_back(aPt_x);
	}
	
	return aV_X;
}

std::vector<double> cIIP_Appli::GetGps_Y_FromFile(std::string GpsFile)
{
	std::vector<double> aV_Y;

    cDicoGpsFlottant aDico = StdGetFromPCP(GpsFile,DicoGpsFlottant);
	std::list<cOneGpsDGF> aOneGpsDGFList = aDico.OneGpsDGF();
		
	for (std::list<cOneGpsDGF>::iterator itP=aOneGpsDGFList.begin(); itP != aOneGpsDGFList.end(); itP ++)
	{
		double aPt_y = itP->Pt().y;
		aV_Y.push_back(aPt_y);
	}
	
	return aV_Y;
}

std::vector<double> cIIP_Appli::GetGps_Z_FromFile(std::string GpsFile)
{
	std::vector<double> aV_Z;

    cDicoGpsFlottant aDico = StdGetFromPCP(GpsFile,DicoGpsFlottant);
	std::list<cOneGpsDGF> aOneGpsDGFList = aDico.OneGpsDGF();
		
	for (std::list<cOneGpsDGF>::iterator itP=aOneGpsDGFList.begin(); itP != aOneGpsDGFList.end(); itP ++)
	{
		double aPt_z = itP->Pt().z;
		aV_Z.push_back(aPt_z);
	}
	
	return aV_Z;
}

std::vector<double> cIIP_Appli::GetImgTMFromFile(std::string ImgTMFile)
{
	std::vector<double> aVTMImg;
	
	cDicoImgsTime aDico = StdGetFromPCP(ImgTMFile, DicoImgsTime);
	std::list<cCpleImgTime> aCpleImgTimeList = aDico.CpleImgTime();
	
	for (std::list<cCpleImgTime>::iterator itP=aCpleImgTimeList.begin(); itP != aCpleImgTimeList.end(); itP ++)
	{
		double aTmImg = itP->TimeIm();
		aVTMImg.push_back(aTmImg);
	}
	
	return aVTMImg;
}

std::vector<std::string> cIIP_Appli::GetImgNameFromFile(std::string ImgTMFile)
{
	std::vector<std::string> aVImgName;
	
	cDicoImgsTime aDico = StdGetFromPCP(ImgTMFile, DicoImgsTime);
	std::list<cCpleImgTime> aCpleImgTimeList = aDico.CpleImgTime();
	
	for (std::list<cCpleImgTime>::iterator itP=aCpleImgTimeList.begin(); itP != aCpleImgTimeList.end(); itP ++)
	{
		std::string aImgName = itP->NameIm();
		aVImgName.push_back(aImgName);
	}
	
	return aVImgName;
}



void cIIP_Appli::WriteImTmInFile(
								std::string aOutputNameFile,
								std::vector<std::string> & aVImName,
								std::vector<Pt3dr> & aVPos,
								bool aFormat
								)
{
	 FILE* aCible = NULL;
	 aCible=fopen(aOutputNameFile.c_str(),"w");
	 
	 if(aFormat)
	 {
		 std::string aFormat = "#F=N_X_Y_Z_W_P_K";
		 fprintf(aCible,"%s \n",aFormat.c_str());
	 }
	 
	 for(unsigned int aK=0; aK < aVImName.size() ; aK++)
	 {
		 fprintf(aCible,"%s %.6f %.6f %.6f %.6f %.6f %.6f\n",aVImName.at(aK).c_str(), aVPos.at(aK).x, aVPos.at(aK).y, aVPos.at(aK).z, 0.0, 0.0, 0.0);
	 }
	 
	 fclose(aCible);
	 
}


std::vector<Pt3dr> cIIP_Appli::ConvertAxial2Pos(std::vector<double> aP_x, std::vector<double> aP_y, std::vector<double> aP_z)
{
	std::vector<Pt3dr> aVPosIm;
	
	//check if same size
	if(aP_x.size() != aP_y.size() || aP_x.size() != aP_z.size() || aP_x.size() != aP_z.size())
	{
		ELISE_ASSERT(false,"Note same size for axial components !");
	}
	
	for (unsigned int aK=0; aK < aP_x.size() ; aK++)
	{
		Pt3dr aPt;
		aPt.x = aP_x.at(aK);
		aPt.y = aP_y.at(aK);
		aPt.z = aP_z.at(aK);
		aVPosIm.push_back(aPt);
	}
	
	return aVPosIm;
}

cIIP_Appli::cIIP_Appli(int argc,char ** argv)
{
	std::string aOut;
	bool aAddFormat = false;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir,"Directory")
					 << EAMC(mGpsFile, "GPS .xml file trajectory",  eSAM_IsExistFile)
					 << EAMC(mTMFile, "Image TimeMark .xml file",  eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Name Output File ; Def = GPSFileName-TMFileName.txt")
					 << EAM(aAddFormat,"Format",false,"Add File Format at the begining fo the File ; Def #F=N_X_Y_Z_W_P_K",eSAM_IsBool)
    );
     
    //read .xml GPS trajectory file
	std::vector<double> aTraj_X_Gps = GetGps_X_FromFile(mGpsFile);
	std::vector<double> aTraj_Y_Gps = GetGps_Y_FromFile(mGpsFile);
	std::vector<double> aTraj_Z_Gps = GetGps_Z_FromFile(mGpsFile);
	std::vector<double> aTraj_MJD_Gps = GetMjdFromGpsFile(mGpsFile);
	
	//read .xml Images TimeMark file
	std::vector<double> aTMImgs = GetImgTMFromFile(mTMFile);
	
	printf("****************************************************************\n");
	printf("Gps_MJD[0] = %lf\n", aTraj_MJD_Gps.at(0));  
	printf("Gps_MJD[end] = %lf\n", aTraj_MJD_Gps.at(aTraj_MJD_Gps.size()-1));
	printf("****************************************************************\n");
	printf("****************************************************************\n");
	printf("Img_MJD[0] = %lf\n", aTMImgs.at(0));
	printf("Img_MJD[end] = %lf\n", aTMImgs.at(aTMImgs.size()-1));
	printf("****************************************************************\n");
	
	//check if intervals make sense
	if(aTMImgs.at(0) < aTraj_MJD_Gps.at(0))
	{
		ELISE_ASSERT(false,"First image TM starts before GPS Traj !");
	}
	else if(aTMImgs.at(aTMImgs.size()-1) > aTraj_MJD_Gps.at(aTraj_MJD_Gps.size()-1))
	{
		ELISE_ASSERT(false,"Last image TM ends after GPS Traj !");
	}
	else
	{
		//make interpolation
		tk::spline aS_x;
		tk::spline aS_y;
		tk::spline aS_z;
			
		aS_x.set_points(aTraj_MJD_Gps,aTraj_X_Gps);
		aS_y.set_points(aTraj_MJD_Gps,aTraj_Y_Gps);
		aS_z.set_points(aTraj_MJD_Gps,aTraj_Z_Gps);
			
		std::vector<double> aImgPos_X;
		std::vector<double> aImgPos_Y;
		std::vector<double> aImgPos_Z;
		
		for (unsigned int aK=0; aK<aTMImgs.size(); aK++)
		{
			double aVal_x = aS_x(aTMImgs.at(aK));
			double aVal_y = aS_y(aTMImgs.at(aK));
			double aVal_z = aS_z(aTMImgs.at(aK));
			
			aImgPos_X.push_back(aVal_x);
			aImgPos_Y.push_back(aVal_y);
			aImgPos_Z.push_back(aVal_z);
		}
		
		if(aOut=="")
		{
			aOut=StdPrefixGen(mGpsFile) + "-" + StdPrefixGen(mTMFile) + ".txt";
		}
		
		//generate a txt file to use with oriconvert for example
		std::vector<std::string> aVImNames = GetImgNameFromFile(mTMFile);
		
		std::vector<Pt3dr> aVPosImgs = ConvertAxial2Pos(aImgPos_X,aImgPos_Y,aImgPos_Z);

		WriteImTmInFile(aOut,aVImNames,aVPosImgs,aAddFormat);
	}
}


int InterpImgPos_main(int argc,char ** argv)
{
	cIIP_Appli anAppli(argc,argv);
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
