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

struct ImgT{
	std::string ImgName;
	double ImgTemp;
};

//----------------------------------------------------------------------------
class cThmProc_Appli
{
	public :
		cThmProc_Appli(int argc,char ** argv);
		void Do2DMatching(std::string aDirectory, std::string aPatImgs, std::string aXmlFile, int aPas);
		void DoHomConv(std::string aDirectory, std::string aPatImgs, int aPas);
		void CalcXYT(std::string aDirectory, std::string aPatImgs, int aDegT, int aDegXY, std::string aKey, std::string aHom, std::string aFileName, std::string aExt, int aPas);
		std::string GetPatNoMatser(std::vector<std::string> aSetIm, int aPas); //first image is the master image
		void ReadImgTFile(std::string aDirectory, std::string aFileName, std::vector<ImgT> & aVSIT, std::string aExt);
		void GenerateXmlAssoc(std::vector<ImgT> aVSIT, std::string aDirectory, std::string aOutFileName, std::string aKey);
		void CorrImgsFromTemp(std::string aMap, std::string aDirectory, std::string aFileName, std::string aPatImgs, std::string aExt, std::string aOutFolder);
		std::string GenNameFile(std::vector<ImgT> aVSIT, std::string aNameIm, std::string aPref);
	private :
		std::string mFullPat;
		std::string mDir;
		std::string mPat;
};

std::string cThmProc_Appli::GenNameFile(std::vector<ImgT> aVSIT, std::string aNameIm, std::string aPref)
{
	std::string aNameFile="";
	
	for(unsigned int aK=0; aK<aVSIT.size(); aK++)
	{
		if(aVSIT.at(aK).ImgName.compare(aNameIm) == 0)
		{
			aNameFile = aPref + ToString(aVSIT.at(aK).ImgTemp) + ".xml";
		}
	}
	
	return aNameFile;
}

void cThmProc_Appli::CorrImgsFromTemp(std::string aMap, std::string aDirectory, std::string aFileName, std::string aPatImgs, std::string aExt, std::string aOutFolder)
{
	//step 0 : get pattern of images to be corrected
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirectory);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatImgs));
	
	//step 1 : read file giving img temp for pattern to correct
	std::vector<ImgT> aVI2Corr;
	ReadImgTFile(aDirectory,aFileName,aVI2Corr,aExt);
	
	//step 2 : compute map at each temperature of pattern to correct
	for(unsigned int aP=0; aP<aVI2Corr.size(); aP++)
	{
		std::string aOutT = "Deg_" + ToString(aVI2Corr.at(aP).ImgTemp) + ".xml";
		
		std::string aCom1 = MMDir()
							+ std::string("bin/mm3d")
							+ std::string(" ")
							+ "CalcMapOfT"
							+ std::string(" ")
							+ aMap
							+ std::string(" ")
							+ ToString(aVI2Corr.at(aP).ImgTemp)
							+ std::string(" ")
							+ "Out="
							+ aOutT;
							
		std::cout << "aCom1 = " << aCom1 << std::endl;
		system_call(aCom1.c_str());
	}
	
	//step 3 : generate corrected images in a new folder
	if(aSetIm.size()>aVI2Corr.size())
	{
		ELISE_ASSERT(false, "ERROR: Can't Correct Image without Temperature !");
	}
	
	for(unsigned int aK=0; aK<aSetIm.size(); aK++)
	{
		for(unsigned int aP=0; aP<aVI2Corr.size(); aP++)
		{
			if(aSetIm.at(aK).compare(aVI2Corr.at(aP).ImgName) == 0)
			{
				std::string aCom = MMDir()
							+ std::string("bin/mm3d")
							+ std::string(" ")
							+ "ReechImMap"
							+ std::string(" ")
							+ aDirectory + aSetIm.at(aK)
							+ std::string(" ")
							+ GenNameFile(aVI2Corr,aSetIm.at(aK),"Deg_");
							
				std::cout << "aCom = " << aCom << std::endl;
				system_call(aCom.c_str());
				std::string aNameCorrImg = "Reech_" + aSetIm.at(aK);
				ELISE_fp::MvFile(aDirectory+aNameCorrImg,aOutFolder);
			}
		}
	}
}

void cThmProc_Appli::GenerateXmlAssoc(std::vector<ImgT> aVSIT, std::string aDirectory, std::string aOutFileName, std::string aKey)
{
	FILE * aFP = FopenNN(aOutFileName,"w","ThermikProc_main");
	cElemAppliSetFile aEASF(aOutFileName);
	
	fprintf(aFP,"<ChantierDescripteur>\n");
	fprintf(aFP,"\t<KeyedNamesAssociations>\n");
	
	for(unsigned int aK=0; aK<aVSIT.size(); aK++)
	{
		fprintf(aFP,"\t\t<Calcs>\n");
		fprintf(aFP,"\t\t\t<Arrite> 1 1 </Arrite>\n");
		fprintf(aFP,"\t\t\t\t<Direct>\n");
		fprintf(aFP,"\t\t\t\t\t<PatternTransform>%s</PatternTransform>\n",aVSIT.at(aK).ImgName.c_str());
		fprintf(aFP,"\t\t\t\t\t<CalcName>%f</CalcName>\n",aVSIT.at(aK).ImgTemp);
		fprintf(aFP,"\t\t\t\t</Direct>\n");
		fprintf(aFP,"\t\t</Calcs>\n");
	}
	
	fprintf(aFP,"\t<Key>%s</Key>\n",aKey.c_str());
	fprintf(aFP,"\t</KeyedNamesAssociations>\n");
	fprintf(aFP,"</ChantierDescripteur>\n");
	
	ElFclose(aFP);
	
	
}

void cThmProc_Appli::ReadImgTFile(std::string aDirectory, std::string aFileName, std::vector<ImgT> & aVSIT, std::string aExt)
{
	ifstream aFichier((aDirectory + aFileName).c_str());
	if(aFichier)
    {
		std::string aLine;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLine,'\n');
			if(aLine.size() != 0)
			{
				char *aBuffer = strdup((char*)aLine.c_str());
				std::string aImage = strtok(aBuffer,"	");
				std::string aTemp = strtok(NULL, "	");
				
				ImgT aImgT;
				if(aExt != "")
					aImgT.ImgName = aImage + aExt;
				else
					aImgT.ImgName = aImage;
					
				aImgT.ImgTemp = atof(aTemp.c_str());
				
				aVSIT.push_back(aImgT);
			}
		}
		aFichier.close();
	}
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
}

std::string cThmProc_Appli::GetPatNoMatser(std::vector<std::string> aSetIm, int aPas)
{
	std::string aPatNoMaster="";
    
    //first image is by default the reference image
    for (unsigned int aP=1 ; aP < aSetIm.size() ; aP=aP+aPas)
    {
		if(aP<aSetIm.size())
		{
			aPatNoMaster =  aPatNoMaster + aSetIm.at(aP);
			if(aP != aSetIm.size()-1)
			{
				aPatNoMaster = aPatNoMaster + "|";
			}
		}
		else
			break;
	}
	
	return aPatNoMaster;
}

void cThmProc_Appli::CalcXYT(std::string aDirectory, std::string aPatImgs, int aDegT, int aDegXY, std::string aKey, std::string aHom, std::string aFileName, std::string aExt, int aPas)
{
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirectory);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatImgs));
    
    std::string aMasterIm = aSetIm.at(0);
    std::cout << "aMasterIm = " << aMasterIm << std::endl;
    
    std::string aPatImSec =  GetPatNoMatser(aSetIm,aPas);
    std::cout << "aPatImSec = " << aPatImSec << std::endl;
    
    //read file giving : img1 temp1 ...
    std::vector<ImgT> aVSIT;
    ReadImgTFile(aDirectory,aFileName,aVSIT,aExt);
    
    //generate xml file of associations
    std::string aOutFileName = aDirectory + "MicMac-LocalChantierDescripteur.xml";
    std::cout << "aOutFileName = " << aOutFileName << std::endl;
    
    GenerateXmlAssoc(aVSIT,aDirectory,aOutFileName,aKey);
    
    //launch com
    std::string aCom = MMDir()
						+ std::string("bin/mm3d")
						+ std::string(" ")
						+ "CalcMapXYT"
						+ std::string(" ")
						+ std::string("\"") + aMasterIm + std::string("\"")
						+ std::string(" ")
						+ std::string("\"") + aPatImSec + std::string("\"")
						+ std::string(" ")
						+ ToString(aDegT)
						+ std::string(" ")
						+ ToString(aDegXY)
						+ std::string(" ")
						+ aKey
						+ std::string(" ")
						+ aHom;
	
	std::cout << "aCom = " << aCom << std::endl;
	system_call(aCom.c_str());
}

void cThmProc_Appli::DoHomConv(std::string aDirectory, std::string aPatImgs, int aPas)
{
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirectory);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatImgs));
    
    //first image is by default the reference image
    for (unsigned int aP=1 ; aP < aSetIm.size() ; aP=aP+aPas)
    {
		if(aP < aSetIm.size())
		{
			std::string aCom = MMDir()
								+ std::string("bin/mm3d")
								+ std::string(" ")
								+ "DMatch2Hom"
								+ std::string(" ")
								+ std::string("\"") + std::string("\"")
								+ std::string(" ")
								+ aSetIm.at(0)
								+ std::string(" ")
								+ aSetIm.at(aP);
			
			std::cout << "aCom = " << aCom << std::endl;
			system_call(aCom.c_str());
		}
		else
			break;
	}
}

void cThmProc_Appli::Do2DMatching(std::string aDirectory, std::string aPatImgs, std::string aXmlFile, int aPas)
{
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirectory);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatImgs));
    
    //first image is by default the reference image
    for (unsigned int aP=1 ; aP < aSetIm.size() ; aP=aP+aPas)
    {
		if(aP<aSetIm.size())
		{
			std::string aCom = "MICMAC" + 
							std::string(" ") + aDirectory +
							aXmlFile + 
							std::string(" ") +
							"+Im1=" + aSetIm.at(0) +
							std::string(" ") +
							"+Im2=" + aSetIm.at(aP) + 
							std::string(" ") + "+PrefDir= +WinExp=false +DilAlt=2";
		
			std::cout << "aCom = " << aCom << std::endl;
			//system_call(aCom.c_str());
		}
		else
			break;
	}
}

cThmProc_Appli::cThmProc_Appli(int argc,char ** argv)
{
	
	bool aShow=false;
	std::string aFullPat2="";
	std::string aFullPat3="";
	std::string aFullPat4="";
	std::string aDir2="";
	std::string aDir3="";
	std::string aDir4="";
	std::string aXml2DM="MM-DeformThermik.xml";
	int aDegT=3;
	int aDegXY=4;
	std::string aKey="Loc-Assoc-Temp";
	std::string aHom="DM";
	std::string aFileName="all_name_temp.txt";
	std::string aExt=".thm.tif";
	std::string aFullPatToCorr="";	//pattern of images to correct from temperature after calibration
	std::string aNameFolder="";
	int aPas=1;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mFullPat,"Directory + Pattern of calibration dataset"),
          LArgMain() << EAM(aFullPat2,"Pat2",false,"Directory + Pattern 2")
					 << EAM(aFullPat3,"Pat3",false,"Directory + Pattern 3")
					 << EAM(aFullPat4,"Pat4",false,"Directory + Pattern 4")
					 << EAM(aXml2DM,"Xml2DMF",false,"xml Name File for 2D Matching ; Def=MM-DeformThermik.xml")
                     << EAM(aShow,"Show",false,"Display ... ; Def=false")
                     << EAM(aDegT,"DegT",false,"Degree for temperature ; Def=3")
                     << EAM(aDegXY,"DegXY",false,"Degree for XY ; Def=4")
                     << EAM(aKey,"Key",false,"Key to calc Temperature")
                     << EAM(aHom,"Hom",false,"Set of Hom ; Def=DM")
                     << EAM(aFileName,"FileImgT",false,"Name of file giving img and Temp ; Def=all_name_temp.txt")
                     << EAM(aExt,"Ext",false,"Image extension not given in FileImgT ; Def=.thm.tif")
                     << EAM(aFullPatToCorr,"PatToCorr",false,"pat of imgs to correct from temperature")
                     << EAM(aNameFolder,"OutDir",false,"Name of Directory of corrected images ; Def=CorrTemp_DirName")
                     << EAM(aPas,"Pas",false,"Step between images in 2D matching ; Def=1")
                     
    );
    
	//read the pattern to use as calibration (add as option the possibility to use several patterns)
	SplitDirAndFile(mDir,mPat,mFullPat);
    
    std::cout<<"Working dir: "<<mDir<<std::endl;
    std::cout<<"Images pattern: "<<mPat<<std::endl;
    
	//compute all deformation maps by 2D matching
    Do2DMatching(mDir,mPat,aXml2DM,aPas);

	//convert all deformation maps to Homol format
	DoHomConv(mDir,mPat,aPas);
	
	//compute evolutive map function of temperature
	CalcXYT(mDir,mPat,aDegT,aDegXY,aKey,aHom,aFileName,aExt,aPas);
	std::string aMap = mDir + "PolOfTXY.xml";
	
	//compute map for each tempurature (dataset to process in a photogrammetric workflow) ; give it as option
	//and generate corrected images from temperature to compare with original images
	if(aFullPatToCorr != "")
	{
		std::string aDir2C="";
		std::string aPat2C="";
		SplitDirAndFile(aDir2C,aPat2C,aFullPatToCorr);
		
		if(aNameFolder == "")
		{
			aNameFolder = aDir2C + "CorrTemp_" + StdPrefixGen(aDir2C);
		}
		
		//create new folder where to put corrected images
		ELISE_fp::MkDirSvp(aNameFolder);
		
		CorrImgsFromTemp(aMap,aDir2C,aFileName,aPat2C,aExt,aNameFolder);
	}
}

int ThermikProc_main(int argc,char ** argv)
{
	cThmProc_Appli anAppli(argc,argv);
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
