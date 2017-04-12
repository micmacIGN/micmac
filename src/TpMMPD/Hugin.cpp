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
#include "schnaps.h"

template <typename T>
std::string NumberToString ( T Number )
{
	std::ostringstream ss;
    ss << Number;
	return ss.str();
}
  
//----------------------------------------------------------------------------
struct Couple{
	std::string Img1;
	std::string Img2;
	double BsurH;
};

//faire un export de tous les BsurH

int cleanHomolByBsurH_main(int argc,char ** argv)
{
	std::string aDir, aNameFile, aHomol;
	bool aShow=false;
	double aSeuil=0.01;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aDir,"Directory")
                     << EAMC(aNameFile,"File From Grep")
					 << EAMC(aHomol,"Homol folder where we will delete files"),
          LArgMain() << EAM(aSeuil,"Min",false,"Minimum Value to keep Homol ; Def=0.01")
					 << EAM(aShow,"Show",false,"Print rm commands ; Def=false")
    );
    
    //vector of structure
    std::vector<Couple> aVCouple;
    
    //read file from a Grep of NewOri Folder
    //NewOriTmpCalibQuick/DSC03247.JPG/OriRel-DSC03248.JPG.xml:          <BSurH>0.00291884516651230936</BSurH>
    ifstream aFichier((aDir + aNameFile).c_str());
    
    if(aFichier)
    {
		std::string aLine;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLine,'\n');
			
			if(aLine.size() != 0)
			{
				char *aBuffer = strdup((char*)aLine.c_str());
				std::string aPart1 = strtok(aBuffer," ");
				//~ std::cout << "aPart1 = " << aPart1 << std::endl;
				std::string aPart2 = strtok( NULL, " " );
				//~ std::cout << "aPart2 = " << aPart2 << std::endl;
				
				Couple aCouple;
				aCouple.Img1 = aPart1.substr(20,12);
				std::cout <<"aCouple.Img1 = " << aCouple.Img1 << std::endl;
				
				aCouple.Img2 = aPart1.substr(40,12);
				std::cout <<"aCouple.Img2 = " << aCouple.Img2 << std::endl;
				
				aCouple.BsurH = atof(aPart2.substr(8,10).c_str());
				std::cout <<"aCouple.BsurH = " << aCouple.BsurH << std::endl;
				
				aVCouple.push_back(aCouple);
			}
		}
	aFichier.close();
	}
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
	
	std::vector<std::string> aVFiles;
	
	//check if value < aSuile then generate file to supress
	for(unsigned int aK=0; aK<aVCouple.size(); aK++)
	{
		if(aVCouple.at(aK).BsurH < aSeuil)
		{
			std::cout << "aVCouple.at(aK).BsurH = " << aVCouple.at(aK).BsurH << std::endl;
			std::string aFile = aHomol + "/" + "Pastis" + aVCouple.at(aK).Img1 + "/" + aVCouple.at(aK).Img2 + ".dat";
			aVFiles.push_back(aFile);
		}
	}
	
	//suppress vector of files
	for(unsigned int aP=0; aP<aVFiles.size(); aP++)
	{
		std::cout << "aVFiles.at(aP) = " << aVFiles.at(aP) << std::endl;
		std::string aCom = "rm " + aVFiles.at(aP);
		system_call(aCom.c_str());
		//~ ELISE_fp::RmFile(aVFiles.at(aP));
	}
	
	return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
//Optimize Silos Processing Tie Points Extraction
class cOSPTPE_Appli
{
	public :
		cOSPTPE_Appli(int argc,char ** argv);
		std::string GenPatFromLF(std::list<cElFilename> aLFileName, bool aH);
	private :
		std::string mDir;
		cInterfChantierNameManipulateur * mICNM;
};

std::string cOSPTPE_Appli::GenPatFromLF(std::list<cElFilename> aLFileName, bool aH)
{
	std::string aPat="";
	
	for(std::list<cElFilename>::iterator iT = aLFileName.begin() ; iT != aLFileName.end(); iT++)
	{
		if(aH)
		{
			aPat = aPat + (*iT).m_basename + std::string(" ");
		}
		else
		{
			aPat = aPat + (*iT).m_basename + std::string("|");
		}
	}
	
	int aSz = aPat.size();
	aPat = aPat.substr(0,aSz-1);
	return aPat;
}

cOSPTPE_Appli::cOSPTPE_Appli(int argc,char ** argv)
{
	bool aShow=false;
	std::string aPat;
	std::string aNameProcFolder="Processing";
	double aRes=500;
	int aProj=0;
	double aFov=50;
	bool aFilter=true;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir, "Directory"),
          LArgMain() << EAM(aShow,"Show",false,"Display ... ; Def=false")
					 << EAM(aNameProcFolder,"ProcFolderName",false,"Name of the folder where the processing will be performed")
					 << EAM(aRes,"ResTieP",false,"Resolution for Tie Points Extraction")
					 << EAM(aProj,"Proj",false,"Projection type (default: 0)")
					 << EAM(aFov,"FOV",false,"Horizontal field of view of images (default: 50)")
					 << EAM(aFilter,"FilterTP",true,"Use Schnaps to reduce tie points; Def=true")
    );
    
    // !!!!!!!! sort directories by name !!!!!!!
    
    //get the list of folders and files inside
    std::list<cElFilename> aFiles;
    std::list<ctPath> aDirectories;
    
    ctPath * aPath = new ctPath(mDir);
    aPath->getContent(aFiles,aDirectories, false);
    
    std::cout << "aDirectories.size() = " << aDirectories.size() << std::endl;
    std::cout << "aFiles.size() = " << aFiles.size() << std::endl;
    
    //if "aNameProcFolder" exists
    for(std::list<ctPath>::iterator iT1 = aDirectories.begin() ; iT1 != aDirectories.end() ; iT1++)
    {
		std::cout << "(*iT1).str().substr(0,(*iT1).str().size()-1) = " << (*iT1).str().substr(0,(*iT1).str().size()-1) << std::endl;
		if((*iT1).str().substr(0,(*iT1).str().size()-1).compare(aNameProcFolder) == 0)
		{
			iT1 = aDirectories.erase(iT1);
		}
	}
	
	std::cout << "aDirectories.size() = " << aDirectories.size() << std::endl;
    
    if(aShow)
    {
		std::cout << "Directories : " << std::endl;
		for(std::list<ctPath>::iterator iT1 = aDirectories.begin() ; iT1 != aDirectories.end() ; iT1++)
		{
			std::cout << "---> " << *iT1 << std::endl;
		}
    }
    
    //get the list of files by folder
    std::vector<std::list<cElFilename> > aLFiles;

    for (std::list<ctPath>::iterator iT = aDirectories.begin() ; iT != aDirectories.end() ; iT++)
    {
		ctPath * aPathC = new ctPath(*iT);
		std::list<cElFilename> aLFC;
		aPathC->getContent(aLFC);
		aLFiles.push_back(aLFC);
	}
	
	if(aShow)
	{
		std::cout << "Files : " << std::endl;
		for(unsigned int aP=0; aP<aLFiles.size(); aP++)
		{
			for(std::list<cElFilename>::iterator iT2 = aLFiles.at(aP).begin() ; iT2 != aLFiles.at(aP).end() ; iT2++)
			{
				std::cout << "---> " << *iT2 << std::endl;
			}
		}
	}
    
    //create new folder and copy all files inside to start processing
    ELISE_fp::MkDirSvp(aNameProcFolder);

    for(unsigned int aV=0; aV<aLFiles.size(); aV++)
    {
		std::cout << "Copy images of groupe : " << aV << std::endl;
		int aCompt = 1;
		for(std::list<cElFilename>::iterator iT2 = aLFiles.at(aV).begin() ; iT2 != aLFiles.at(aV).end() ; iT2++)
		{
			std::cout << "---> Copy of file " << aCompt << " out of " << aLFiles.at(aV).size() << std::endl;
			ELISE_fp::CpFile((*iT2).m_path.str() + (*iT2).m_basename,aNameProcFolder); //copy only if no file !!!!
			aCompt++;
		}
	}
    
    //change working directory
    if(setWorkingDirectory(aNameProcFolder))
    {
		//get new working directory
		ctPath aNWDir = getWorkingDirectory();
		std::cout << "aNWDir = " << aNWDir << std::endl;
	}
    
    //pipeline to generate a pano for each 	level N 
	for(unsigned int aP=0; aP<aDirectories.size(); aP++)
	{
		std::cout << "debugggggg" << std::endl;
		std::string aPatL = GenPatFromLF(aLFiles.at(aP),false);
		std::cout << "debugggggg" << std::endl;
		//std::string aPatL2 = GenPatFromLF(aLFiles.at(aP+1),false);
		std::cout << "debugggggg" << std::endl;
		std::string aXmlOutFile = "NameCple_" + NumberToString(aP) + "_" + NumberToString(aP+1) + ".xml";
		std::cout << "aXmlOutFile = " << aXmlOutFile << std::endl;
		
		//generate a .xml file of all couples for level N & level N+1
		std::string aCom1 = MMDir()
							+ std::string("bin/mm3d")
							+ std::string(" ")
							+ "GenPairsFile"
							+ std::string(" ")
							+ std::string("\"") + aPatL + std::string("\"")
							+ std::string(" ")
							+ std::string("\"") + aPatL + std::string("\"")
							+ std::string(" ")
							+ "Out="
							+ aXmlOutFile;
		std::cout << "aCom1 = " << aCom1 << std::endl;
		system_call(aCom1.c_str());
		
		//compute tie points for all images for level N & level n+1
		std::string aCom2 = MMDir()
							+ std::string("bin/mm3d")
							+ std::string(" ")
							+ "Tapioca File"
							+ std::string(" ")
							+ aXmlOutFile
							+ std::string(" ")
							+ NumberToString(aRes);
		std::cout << "aCom2 = " << aCom2 << std::endl;
		system_call(aCom2.c_str());
		
		//if tie points reduction is performed
		if(aFilter)
		{
			std::string aComF = MMDir()
								+ std::string("bin/mm3d")
								+ std::string(" ")
								+ "Schnaps"
								+ std::string(" ")
								+ std::string("\"") + aPatL + std::string("\"")
								+ std::string(" ")
								+ "NbWin=100";
			std::cout << "aComF = " << aComF << std::endl;
			system_call(aComF.c_str());
		}
		
		//generate a Hugin project for level N & level n+1
		std::string aPatLH = GenPatFromLF(aLFiles.at(aP),true);
		//std::string aPatL2H = GenPatFromLF(aLFiles.at(aP+1),true);
		
		std::string aHNameProject = "Hugin_" + NumberToString(aP) + "_" + NumberToString(aP+1) + ".pto";
		std::string aCom3 = "pto_gen -o" 
		                     + std::string(" ") 
		                     + aHNameProject 
		                     + std::string(" ")
							 + aPatLH
							 //+ std::string(" ") + aPatL2H
							 + std::string(" ")
							 + "-p " + NumberToString(aProj)
							 + std::string(" ")
							 + "-f " + NumberToString(aFov);
		std::cout << "aCom3 = " << aCom3 << std::endl;
		system_call(aCom3.c_str());
		
		//specify the variables to be optimized
		std::string aCom4 = "pto_var --opt=\"y, p, r, TrX, TrY, TrZ, Tpy, Tpp, v, a, b, c, d, e, g, t, Eev, Er, Eb, Vb, Vc, Vd, Vx, Vy, Ra, Rb, Rc, Rd, Re\""
		                     + std::string(" ")
		                     + "-o " + aHNameProject
		                     + std::string(" ")
		                     + aHNameProject;
		std::cout << "aCom4 = " << aCom4 << std::endl;
		system_call(aCom4.c_str());
		                     
		//include tie points from MicMac in the project file
		std::string aCom5 = MMDir()
							+ std::string("bin/mm3d")
							+ std::string(" ")
							+ "TestLib GenHuginCp"
							+ std::string(" ")
							+ std::string("\"") + aPatL
							//+ std::string("|") + aPatL2
							+ std::string("\"") + std::string(" ")
							+ aHNameProject
							+ std::string(" ")
							+ "HomolIn=_mini";
		std::cout << "aCom5 = " << aCom5 << std::endl;
		system_call(aCom5.c_str());
		
		//lanch Hugin optimizer
		std::string aCom6 = "autooptimiser -a -l -s -m -v"
		                     + std::string(" ") + NumberToString(aFov)
		                     + std::string(" ")
		                     + "-o " + StdPrefixGen(aHNameProject) + "_Homol.pto"
		                     + std::string(" ") + StdPrefixGen(aHNameProject) + "_Homol.pto";
		std::cout << "aCom6 = " << aCom6 << std::endl;
		system_call(aCom6.c_str());
		
		//pano configuration
		std::string aCom7 = "pano_modify -p " + NumberToString(aProj)
		                    + std::string(" ") + "-o "
		                    + StdPrefixGen(aHNameProject) + "_Homol.pto"
		                    + std::string(" ") + StdPrefixGen(aHNameProject) + "_Homol.pto";
		std::cout << "aCom7 = " << aCom7 << std::endl;
		system_call(aCom7.c_str());
		                    
		//generate individual images to be stiched
		std::string aCom8 = "nona -z LZW -r ldr -m TIFF_m -o ImgsIndiv_" + NumberToString(aP) + "_" + NumberToString(aP+1) + "_" + std::string(" ") + StdPrefixGen(aHNameProject) + "_Homol.pto";
		std::cout << "aCom8 = " << aCom8 << std::endl;
		system_call(aCom8.c_str());
		
		//assembly for level N & level N+1
		std::string aOutMosaic = "Mosaic_" + NumberToString(aP) + "_" + NumberToString(aP+1) + ".jpg";
		std::string aCom9 = "enblend --compression=90 ImgsIndiv_" + NumberToString(aP) + "_" + NumberToString(aP+1) + "*.*tif -o " +  aOutMosaic;
		std::cout << "aCom9 = " << aCom9 << std::endl;
		system_call(aCom9.c_str());
	}
	
	//same pipeline to generate a global panoramique from each individual (level N & level N+1 pano)
	{
		//generate a .xml file of all individual panoramik
		
		//compute tie points for all individual panoramik
		
		//if tie points reduction is performed
		
		//generate a Hugin project for all individual panoramik
		
		//specify the variables to be optimized
		
		//include tie points from MicMac in the project file
		
		//lanch Hugin optimizer
		
		//pano configuration
		
		//generate individual images to be stiched
		
		//assembly for all individual panoramik
		
	}

}

int OptTiePSilo_main(int argc,char ** argv)
{
	cOSPTPE_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
//GHCPFH = Generate Hugin Control Points From Homol
class cGHCFH_Appli
{
	public :
		cGHCFH_Appli(int argc,char ** argv);
		void WriteCpInHFile(std::string aFile, cInterfChantierNameManipulateur * mICNM, std::string mHomol);
		int GetIndiceFromImg(std::string aImg, std::vector<std::string> aVImgs);
		void AddLines(std::vector<std::string> aVImgs, std::vector<Pt2dr> aVPts, std::vector<std::string> aVAllImgs, std::vector<std::string> & aVHL);
		void ShowVs(std::vector<std::string> aVS);
	private :
		std::string mFullName;
		std::string mDir;
		std::string mPat;
		cInterfChantierNameManipulateur * mICNM;
		std::string mHPF; //Hugin Projet File
		std::string mHomol;
		bool mExt;
};

void cGHCFH_Appli::ShowVs(std::vector<std::string> aVS)
{
	for(unsigned int aK=0; aK<aVS.size(); aK++)
	{
		//std::cout << "--> " << aVS.at(aK) << std::endl;
	}
}
int cGHCFH_Appli::GetIndiceFromImg(std::string aImg, std::vector<std::string> aVImgs)
{
	int aIndc=0;
	
	for(unsigned int aK=0; aK<aVImgs.size(); aK++)
	{
		if(aImg.compare(aVImgs.at(aK))==0)
		{
			aIndc = aK;
		}
	}
	
	return aIndc;
}
void cGHCFH_Appli::AddLines(std::vector<std::string> aVImgs, std::vector<Pt2dr> aVPts, std::vector<std::string> aVAllImgs, std::vector<std::string> & aVHL)
{
	for(unsigned int aK=0; aK<aVImgs.size(); aK++)
	{
		for(unsigned int aP=0; aP<aVImgs.size(); aP++)
		{
			if(aK != aP)
			{
				std::string aHLine = "c n" + 
				                    NumberToString(GetIndiceFromImg(aVImgs.at(aK),aVAllImgs)) +
				                    " N" +
				                    NumberToString(GetIndiceFromImg(aVImgs.at(aP),aVAllImgs)) +
				                    " x" +
				                    NumberToString(aVPts.at(aK).x) +
				                    " y" +
				                    NumberToString(aVPts.at(aK).y) +
				                    " X" +
				                    NumberToString(aVPts.at(aP).x) +
				                    " Y" +
				                    NumberToString(aVPts.at(aP).y) +
				                    " t0";
				//std::cout << "aHLine = " << aHLine << std::endl;                    
				aVHL.push_back(aHLine);
			}
		}
	}
}

void cGHCFH_Appli::WriteCpInHFile(std::string aFile, cInterfChantierNameManipulateur * mICNM, std::string mHomol)
{
	
	const std::vector<std::string> aVImgs = *(mICNM->Get(mPat));
	
	//read Homol Directory
    //Init Keys for homol files
    std::list<cHomol> allHomols;
    std::string anExt = mExt ? "txt" : "dat";
    //std::cout << "anExt = " << anExt << std::endl;
    std::cout << "mHomol = " << mHomol << std::endl;
    std::string aKH =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(mHomol)
            +  std::string("@")
            +  std::string(anExt);
            
    CompiledKey2 aCK(mICNM,aKH);
    
    //create pictures list, and pictures size list
    std::map<std::string,cPic*> allPics;
    std::vector<cPicSize*> allPicSizes;
    std::cout<<"Found "<<aVImgs.size()<<" pictures."<<endl;
    computeAllHomol(mICNM,mDir,mPat,aVImgs,allHomols,aCK,allPics,allPicSizes,false,0);
    
    std::vector<std::string> aVHLines;
    
    for (std::list<cHomol>::iterator itHomol=allHomols.begin();itHomol!=allHomols.end();++itHomol)
    {
		int aNbr = itHomol->getPointOnPicsSize();
		std::vector<std::string> aVImgsC;
		std::vector<Pt2dr> aVPtsC;
		//std::cout << "********************** aNbr = " << aNbr << std::endl;
		for(int aK=0; aK<aNbr; aK++)
		{
			cPointOnPic* aPointOnPic = itHomol->getPointOnPic(aK);
			cPic* aPic = aPointOnPic->getPic();
			std::string aName = aPic->getName();
			//std::cout << "aName = " << aName << std::endl;
			Pt2dr& aPt = aPointOnPic->getPt();
			//std::cout << "aPt = " << aPt << std::endl;
			
			aVImgsC.push_back(aName);
			aVPtsC.push_back(aPt);
		}
		//add lines to be written in the .pto file
		AddLines(aVImgsC,aVPtsC,aVImgs,aVHLines);
	}
	
	//std::cout << "allHomols.size() = " << allHomols.size() << std::endl;
	
	std::vector<std::string> aLHeader;
	std::vector<std::string> aLFooter;
	ifstream aFichier((mDir + mHPF).c_str());
	bool aOF=false;
	if(aFichier)
	{
		std::string aLine;    
		while(!aFichier.eof())
		{
			getline(aFichier,aLine,'\n');
			if(aLine.compare("# control points") != 0 && aOF==false)
			{
				aLHeader.push_back(aLine);
			}
			else
			{
				aOF=true;
				if(aLine.compare(0,3,"c n") == 0)
				{
				}
				else if(aLine.compare("# control points") == 0)
				{
					
				}
				else
				{
					aLFooter.push_back(aLine);
				}
			}
		}
	aFichier.close();
	}
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
	
	//std::cout << "Header = " << std::endl;
	ShowVs(aLHeader);
	
	//std::cout << "Footer = " << std::endl;
	ShowVs(aLFooter);
	
	//ecrire dans un nouveau fichier
	std::string aOut= StdPrefixGen(mHPF) + "_Homol.pto";
	if (!MMVisualMode)
	{
		FILE * aFP = FopenNN(aOut,"w","GenHuginCpFromHomol_main");
		cElemAppliSetFile aEASF(mDir + ELISE_CAR_DIR + aOut);
		
		//write header
		for(unsigned int aK=0; aK<aLHeader.size(); aK++)
		{
			fprintf(aFP, "%s\n",aLHeader.at(aK).c_str());
		}
		
		//line # control points
		std::string Line = "# control points";
		fprintf(aFP, "%s\n",Line.c_str());
		
		//write line in hugin format
		for(unsigned int aL=0; aL<aVHLines.size(); aL++)
		{
			fprintf(aFP, "%s\n",aVHLines.at(aL).c_str());
		}
		
		//write footer
		for(unsigned int aF=0; aF<aLFooter.size(); aF++)
		{
			fprintf(aFP, "%s\n",aLFooter.at(aF).c_str());
		}
		
	ElFclose(aFP);
	}
	
}

cGHCFH_Appli::cGHCFH_Appli(int argc,char ** argv)
{
	bool aShow=false;
	mExt=false;
	mHomol="";
	//~ std::string aOut="";
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mFullName, "FullName")
					 << EAMC(mHPF, ".pto Hugin File Project", eSAM_IsExistFile),
          LArgMain() << EAM(aShow,"Show",false,"Display ... ; Def=false")
                     << EAM(mExt,"Ext",false,"Homol extension ; Def=dat")
                     << EAM(mHomol, "HomolIn", true, "Input Homol directory suffix (without \"Homol\")")
                     //~ << EAM(aOut,"Out",false,"Output .pto new file ; Def=File_Homol.pto")
    );
    
    //~ if(aOut="")
    //~ {
		//~ aOut = StdPrefixGen(mHPF) + "_Homol.pto";
	//~ }
    
    SplitDirAndFile(mDir,mPat,mFullName);
    StdCorrecNameHomol(mHomol,mDir);
    //~ std::cout<<"Working dir: "<<aDirImages<<std::endl;
    //~ std::cout<<"Files pattern: "<<aPatIm<<std::endl;

    mICNM=cInterfChantierNameManipulateur::BasicAlloc(mDir);
	WriteCpInHFile(mHPF, mICNM, mHomol);
}

int GenHuginCpFromHomol_main(int argc,char ** argv)
{	
	cGHCFH_Appli anAppli(argc,argv);
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
