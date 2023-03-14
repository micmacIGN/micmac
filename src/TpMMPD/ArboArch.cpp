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

#define  NbrFolder 17
#define  NbrTypeImgExt 2

const char * ImgExt[NbrTypeImgExt] = {
										"JPG",								//0
										"jpg"								//1
								     };

const char * Arch[NbrFolder] = {

								"00_Entree",								//0
								"01_Documents",								//1
								"02_Donnees_Brutes",						//2
								"02_Donnees_Brutes/01_Img",					//3
								"02_Donnees_Brutes/02_GCP",					//4
								"02_Donnees_Brutes/03_GNSS",				//5
								"02_Donnees_Brutes/04_Divers",				//6
								"03_Traitements",							//7
								"03_Traitements/01_Img",					//8
								"03_Traitements/02_GCP",					//9
								"03_Traitements/03_GNSS",					//10
								"03_Traitements/04_Processing",				//11
								"03_Traitements/05_Results",				//12
								"03_Traitements/05_Results/01_Point_Cloud",	//13
								"03_Traitements/05_Results/02_Mesh",		//14
								"03_Traitements/05_Results/03_Ortho",		//15
								"04_Rendus"									//16
							};

const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%d%m", &tstruct);
    return buf;
}

struct sensor{
	std::string Label;
	std::string Type;
	Pt2di Resolution;
	Pt2dr PixWH;
	double Focal;
};

//

//option pour renommer toutes les images;
//détecter s'il y a des images avec nom identique --> renomme tout
//si toutes les images ont des noms difféente; --> je garde tel quel ou option spécifiée par le user
//

//input : nom du dossier qui contient les images
//		  structure : 0/ img+file
//		  structure : 1/ Folder1/img+file
//		  			     Folder2/img+file
//		  			     ...


//IMG/Centrale1/Batiment1/FaceNord/Vol1
//IMG/Centrale1/Batiment1/FaceNord/Vol2


//input : indice de profondeur pour genérer l'archive

//générer l'arboresence au niveau du dossier courant

//copier toutes les données au bon endroit

class cArboArch_Appli
{
	public :
		cArboArch_Appli(int argc,char ** argv);
		int GetMaxProf(std::list<ctPath> aLDir);
		int GetNbrBS(std::string aS);
		int GetIndmaxVal(std::vector<int> aVI);
		void GenerateArchStruct(
								int aProf,
								std::list<ctPath> aLDir,
								bool aModeFusion
								);
		void GenStruct(
					  std::string aName
					  );
		std::string GeneNameProj(std::string aName);
		std::string GenPsxName(std::string aName);
		void WriteInPsxFile(
							std::string aName,
							int aLevel
							);
		void GenInFilesFolder(std::string);
		sensor GetSensorFromImg(std::string aNameImg);
	private :
		std::string mDir;
};

//get a sensor struct from an image name
sensor cArboArch_Appli::GetSensorFromImg(std::string aNameImg)
{
	sensor aSensor;
	//launch MMXmlXif xith the image
	
	//read the xml file
	
	//put infos in sensor struct
	
	//return struct
	return aSensor;
}

//generate structure inside .files folder
void cArboArch_Appli::GenInFilesFolder(std::string aName)
{		
	/*******************************************************************/
	//create doc.xml file inside
	std::cout << "aName Level 0 = " << aName << std::endl;
	std::string aNameXmlFile = aName + "/" + "doc.xml";
	std::string aCom1 = ">> " + aNameXmlFile;
	std::cout << "aCom1 = " << aCom1 << std::endl;
	system_call(aCom1.c_str());
	
	//write inside
	WriteInPsxFile(aNameXmlFile,1);
	
	//zip doc.xml file and name it project.zip
	ctPath aWDir = getWorkingDirectory();
	if(setWorkingDirectory(aName))
	{
		std::string aComZ = "zip project.zip doc.xml";
		system_call(aComZ.c_str());
	}
	setWorkingDirectory(aWDir);
	
	//create a folder named "0"
	std::string aNameF = "0";
	ELISE_fp::MkDirSvp(aName + "/" + aNameF);
	/*******************************************************************/
	
	/*******************************************************************/
	//create a doc.xml file inside folder named "0"
	std::cout << "aName Level 1 = " << aName + "/" + aNameF << std::endl;
	std::string aNameXmlFileL1 = aName + "/" + aNameF + "doc.xml";
	
	//write inside
	WriteInPsxFile(aNameXmlFileL1,2);
	
	//create a .zip file again and name it chunk.zip
	
	//create a folder named "0"
	/*******************************************************************/
}

//write in .psx file
void cArboArch_Appli::WriteInPsxFile(
									 std::string aName,
									 int aLevel
									 )
{
	//if file do exist
	if(ELISE_fp::exist_file(aName))
	{
		FILE * aFP = FopenNN(aName,"w","ArboArch_main");
		cElemAppliSetFile aEASF(aName);
		
		if(aLevel == 0)
		{
			fprintf(aFP,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
			fprintf(aFP,"<document version=\"1.2.0\" path=\"{projectname}.files/project.zip\"/>\n");
		}
		
		if(aLevel == 1)
		{
			
			fprintf(aFP,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
			fprintf(aFP,"<document version=\"1.2.0\">\n");
			fprintf(aFP,"\t<chunks next_id=\"1\">\n");
			fprintf(aFP,"\t\t<chunk id=\"0\" path=\"0/chunk.zip\"/>\n");
			fprintf(aFP,"\t</chunks>\n");
			fprintf(aFP,"</document>\n");
		}
		
		if(aLevel == 2)
		{
			fprintf(aFP,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
			fprintf(aFP,"<chunk label=\"Chunk 1\" enabled=\"true\">\n");
		}
		
		else
		{
			//~ ELISE_ASSERT(false, "Problem with level value for .xml writing files !");
		}
		
		ElFclose(aFP);
	}
}

//generate .psx file and folder based on "Code_affaire-Client-Nom_mission"
//output format = Client - NomMission - Date - EtapeCalcul
std::string cArboArch_Appli::GenPsxName(std::string aName)
{
	std::string aPsxName="";
	int aIndic1=-1;
	
	for(unsigned int i=0; i<aName.length()-1; i++)
	{
		if(aName.compare(i,1,"/") == 0)
		{
			aIndic1=i;
		}
	}
	
	if(aIndic1 != -1)
	{
	
		std::string aNamepfolder = aName.substr(aIndic1+1,aName.size()-aIndic1);
		//~ std::replace(aNamepfolder.begin(), aNamepfolder.end(), '/', '\0');
		aNamepfolder = aNamepfolder.substr(0,aNamepfolder.size()-1);
		std::cout << "aNamepfolder = " << aNamepfolder << std::endl;
	
		int aIndic2=-1;
		for(unsigned int i=0; i<aNamepfolder.length(); i++)
		{
			if(aNamepfolder.compare(i,1,"-") == 0)
			{
				aIndic2=i;
				break;
			}
		}
		
		if(aIndic2 != -1)
		{
			std::string aDate = currentDateTime();
			aPsxName = aNamepfolder.substr(aIndic2+1, aNamepfolder.size()-aIndic2-1) + "-" + aDate + "-" + "Aero";
			std::cout << "aPsxName = " << aPsxName << std::endl;
		}
	}
	
	return aPsxName;
}

//function to generate structure of archivage
void cArboArch_Appli::GenStruct(std::string aName)
{
	//create the folder at current dir
	ELISE_fp::MkDirSvp(aName);
	
	//creat all the folders
	for(unsigned int aP=0; aP<NbrFolder; aP++)
	{
		std::string aNameF = aName + "/" + Arch[aP];
		std::cout << "Create folder ---> " << aNameF << std::endl;
		ELISE_fp::MkDirSvp(aNameF);
	}
}

//function to generate proper folder name
std::string cArboArch_Appli::GeneNameProj(std::string aName)
{
	std::string aNameOut="";
	
	int aIndic=-1;
	
	//get position of first "/"
	for(unsigned int i=0; i<aName.length(); i++)
	{
		if(aName.compare(i,1,"/") == 0)
		{
			aIndic=i;
			break;
		}
	}
	
	if(aIndic != -1)
	{
		aNameOut = aName.substr(aIndic+1,aName.length()-aIndic-2);
		std::replace(aNameOut.begin(), aNameOut.end(), '/', '_');
	}
	
	return aNameOut;
}

//generate structure of archive function of Prof
void cArboArch_Appli::GenerateArchStruct(
										 int aProf,
										 std::list<ctPath> aLDirG,
										 bool aModeFusion
										 )
{
	//reduced list of dir (1 project by dir)
	std::vector<std::string> aVReducDirs;
	
	//list of subdirs
	std::vector<std::string> aVsubDirs;
	
	//get list of folder at this specific Prof Level
	for(std::list<ctPath>::iterator iT1 = aLDirG.begin() ; iT1 != aLDirG.end() ; iT1++)
	{
		if(GetNbrBS((*iT1).str())-1 == aProf)
		{
			std::cout << "For Prof = " << aProf << " directory : --> " << (*iT1) << std::endl;
			aVReducDirs.push_back((*iT1).str());
		}
		
		if(GetNbrBS((*iT1).str())-1 > aProf)
		{
			std::cout << "SubDir : --> " << (*iT1) << std::endl;
			aVsubDirs.push_back((*iT1).str());
		}
	}
	
	//display aVReducDirs
	for(unsigned int i=0; i<aVReducDirs.size(); i++)
	{
		std::cout << "aVReducDirs[i] = " << aVReducDirs[i] << std::endl;
	}
	
	//display aVsubDirs
	for(unsigned int i=0; i<aVsubDirs.size(); i++)
	{
		std::cout << "aVsubDirs[i] = " << aVsubDirs[i] << std::endl;
	}
	
	//for size of aVReducDirs generate same number of project
	for(unsigned int aP=0; aP<aVReducDirs.size(); aP++)
	{
		//generate good name for folder of project
		std::string aNameStruc = GeneNameProj(aVReducDirs.at(aP));
		std::cout<<"\n";
		std::cout << "aNameStruc = " << aNameStruc << std::endl;
		std::cout<<"\n";
		
		//create folder of project + generate structure Arch
		GenStruct(aNameStruc);
		
		//copy files at the correct place
		if(!aModeFusion)
		{
			//copy subfolders & put them in Arch[8]
			if(aVsubDirs.size() == 0)
			{
				//it is already the last profondeur; copy directly
				std::list<cElFilename> aLFiles;
				std::list<ctPath> aLDirectories;
					
				ctPath * aPath = new ctPath(aVReducDirs.at(aP));
				aPath->getContent(aLFiles,aLDirectories,true);
				
				ELISE_ASSERT(aLDirectories.size() == 0 , "Incoherence dans les sous dossiers !");
					
				std::cout << "aLFiles.size() == ? " << aLFiles.size() << std::endl;
				std::cout << "aLDirectories.size() == ? " << aLDirectories.size() << std::endl;
					
				for(std::list<cElFilename>::iterator iT2 = aLFiles.begin() ; iT2 != aLFiles.end() ; iT2++)
				{
					std::cout << "Cp FILE " << (*iT2) << " -----> " << (aNameStruc + "/" + Arch[8]) <<  std::endl;
					//get sensor form image
					ELISE_fp::CpFile((*iT2).str(),(aNameStruc + "/" + Arch[8]));
				}
				
				//get vector of unique sensors
			}
			
			else
			{
				//copy subfolders in Arch[8] for each folder at level prof
				std::list<cElFilename> aLFiles;
				std::list<ctPath> aLDirectories;
					
				ctPath * aPath = new ctPath(aVReducDirs.at(aP));
				aPath->getContent(aLFiles,aLDirectories,false);
				
				std::cout << "aLFiles.size() == ? " << aLFiles.size() << std::endl;
				std::cout << "aLDirectories.size() == ? " << aLDirectories.size() << std::endl;
					
				for(std::list<ctPath>::iterator iT2 = aLDirectories.begin() ; iT2 != aLDirectories.end() ; iT2++)
				{
					std::cout << "Cp FOLDER " << (*iT2) << " -----> " << (aNameStruc + "/" + Arch[8]) <<  std::endl;
					ELISE_fp::CpFile((*iT2).str(),(aNameStruc + "/" + Arch[8]));
				}
			}
			
			//check by extension to clean processing folder
			//......
			
			//get name of proc files ; Name format = Client - NomMission - Date - EtapeCalcul
			ctPath aWDir = getWorkingDirectory();
			std::cout << "aWDir = " << aWDir << std::endl;
			std::string aPsxFileName = GenPsxName(aWDir.str());
			
			//create file .psx at Arch[11]
			std::string aFile1 = aWDir.str() + aNameStruc  + "/" + Arch[11] + "/" + aPsxFileName + ".psx";
			std::cout << "Create File -----> " << aFile1 << std::endl;
			std::string aCom1 = ">> " + aFile1;
			std::cout << "aCom1 = " << aCom1 << std::endl;
			system_call(aCom1.c_str());
			
			//write in .psx file
			WriteInPsxFile(aFile1,0);
			
			//creat folder .files at Arch[11]
			std::string aFile2 = aWDir.str() + aNameStruc + "/" + Arch[11] + "/" + aPsxFileName + ".files";
			std::cout << "Create File -----> " << aFile2 << std::endl;
			std::string aCom2 = "mkdir -p " + aFile2;
			std::cout << "aCom2 = " << aCom2 << std::endl;
			system_call(aCom2.c_str());
			
			//generate structure inside the .files folder
			GenInFilesFolder(aFile1);
			GenInFilesFolder(aFile2);
		}
		
		//to do .....
		else
		{
			//get all list of files and copy in right folder based on extension but with unique name
		}
	}
}

//return indic of max value of a vector
int cArboArch_Appli::GetIndmaxVal(std::vector<int> aVI)
{
	int aIndMV=0;
	aIndMV = *max_element(aVI.begin(), aVI.end());
	return aIndMV;
}

//return number of BS in a string
int cArboArch_Appli::GetNbrBS(std::string aS)
{
	int aNbr=0;
	
	for (unsigned int i=0; i<aS.length(); i++)
	{
		//~ std::cout << "aS.substr(i,i+1) = " << aS.substr(i,1) << std::endl;
		if(aS.compare(i,1,"/") == 0)
			aNbr++;
	}
	
	return aNbr;
}

//return max prof of list of directories
int cArboArch_Appli::GetMaxProf(std::list<ctPath> aLDir)
{
	std::vector<int> aVProf;
	
	for(std::list<ctPath>::iterator iT1 = aLDir.begin() ; iT1 != aLDir.end() ; iT1++)
	{
		std::cout << "-------> " << (*iT1) << std::endl;
		aVProf.push_back(GetNbrBS((*iT1).str()));
	}
	
	return GetIndmaxVal(aVProf);
}

cArboArch_Appli::cArboArch_Appli(int argc,char ** argv)
{
	bool aShow=true;
	std::string aDirPrincipal="IMG";
	int aPofS=-1;
	bool aModeF=false;
	
	ElInitArgMain
    (
		argc, argv,
        LArgMain() << EAMC(mDir,"Current Directory")
				   << EAMC(aPofS,"level of split"),
        LArgMain() << EAM(aShow,"Show",false,"Show some details, Def=false")
				   << EAM(aDirPrincipal,"DirP",false,"Principal Directory of all data ; Def=IMG")
				   << EAM(aModeF,"Fusion",false,"Fusion of all images in same folder ; Def=false")
    );
    
    //check if "aDirPrincipal" exists
    //~ if(ELISE_fp::exist_file(aDirPrincipal))
    //~ {
		
		//get the list of folders and files inside "aDirPrincipal" folder
		std::list<cElFilename> aLFiles;
		std::list<ctPath> aLDirectories;
		
		ctPath * aPath = new ctPath(mDir+aDirPrincipal);
		aPath->getContent(aLFiles,aLDirectories,true);
		
		//if print some details
		if(aShow)
		{
			std::cout << "aLDirectories.size() = " << aLDirectories.size() << std::endl;
			std::cout << "aLFiles.size() = " 	   << aLFiles.size() 	   << std::endl;
			std::cout << "\n";
			std::cout << "Display List of Directories : " << std::endl;
			
			for(std::list<ctPath>::iterator iT1 = aLDirectories.begin() ; iT1 != aLDirectories.end() ; iT1++)
			{
				std::cout << "-------> " << (*iT1).str().substr(0,(*iT1).str().size()-1) << std::endl;
			}
			
			std::cout << "\n";
			std::cout << "Display List of Files : " << std::endl;
			
			for(std::list<cElFilename>::iterator iT2 = aLFiles.begin() ; iT2 != aLFiles.end() ; iT2++)
			{
				std::cout << "-------> " << (*iT2) << std::endl;
			}
			
			std::cout << "\n";
		}
		
		//get max profondeur (& Check Coherence ?)
		int aMaxProf = GetMaxProf(aLDirectories);
		ELISE_ASSERT(aPofS <= aMaxProf-1, "aPofS can't be > to aMaxProf")
		
		//if print some details
		if(aShow)
		{
			std::cout << "\n";
			std::cout << "Maximum Prof = " << aMaxProf << std::endl;
			std::cout << "\n";
		}
		
		//use max prof to generate struct of archivage at mDir Level
		GenerateArchStruct(aPofS,aLDirectories,aModeF);
		
		
	//~ }
	//~ else
	//~ {
		//~ std::cout << " Folder " << aDirPrincipal << " do not exist !" << std::endl;
	//~ }
}

int ArboArch_main(int argc,char ** argv)
{
	cArboArch_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
associés au chargement,  a  l'utilisation,  a  la modification et/ou au
développement et a  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe a
manipuler et qui le réserve donc a  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités a  charger  et  tester  l'adéquation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
a  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder a  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
