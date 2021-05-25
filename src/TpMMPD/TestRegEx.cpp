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


template <typename T> string NumberToString(T Number)
{
	ostringstream ss;
    ss << Number;
    return ss.str();
}
  
//----------------------------------------------------------------------------
int TestRegEx_main(int argc,char ** argv)
{
    std::string aFullPattern;//pattern of all files
    bool aDispPatt=false;
    string aExport;
    
    ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(aFullPattern, "Pattern of files",  eSAM_IsPatFile),
    
    //optional arguments
    LArgMain()  << EAM(aDispPatt, "DispPat", false, "Display Pattern to use in cmd line ; Def=false", eSAM_IsBool)
                << EAM(aExport, "ExpList", false, "Export list image in text file ; Def=false")

  
    );
    
    if (MMVisualMode) return EXIT_SUCCESS;
    
    // Initialize name manipulator & files
    std::string aDirImages,aPatIm;
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Files pattern: "<<aPatIm<<std::endl;

    ofstream aExpListImg;

    if (EAMIsInit(&aExport))
    {
        aExpListImg.open (aExport.c_str());
    }


    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));
    
    std::vector<std::string> aVIm;
    
    std::cout<<"Selected files:"<<std::endl;
    for (unsigned int i=0;i<aSetIm.size();i++)
    {
        std::cout<<" - "<<aSetIm[i]<<std::endl;
        if (EAMIsInit(&aExport))
        {
           aExpListImg<<aSetIm[i]<<endl;
        }
        aVIm.push_back(aSetIm[i]);
    }
    std::cout<<"Total: "<<aSetIm.size()<<" files."<<std::endl;
	
	if(aDispPatt)
	{
		std::string aPat="";
		
		for(unsigned int i=0;i<aVIm.size()-1;i++)
		{
			aPat = aPat + aVIm.at(i) + "|";
		}
		
		aPat = aPat + aVIm.at(aVIm.size()-1);
		
		std::cout << "Pat = \"" << aPat << "\"" << std::endl;
	}
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
int PatFromOri_main(int argc,char ** argv)
{
	std::string aOri;
	bool aShow=false;
	 
	ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
	LArgMain()  << EAMC(aOri, "Ori Folder", eSAM_IsExistDirOri),
	
	LArgMain()  << EAM(aShow, "Show", false, "Display Pattern to use in cmd line ; Def=false",eSAM_IsBool)
	);
	
	if (MMVisualMode) return EXIT_SUCCESS;
	
	std::string aFullName="Orientation-*.*xml";
    cInterfChantierNameManipulateur *ManC=cInterfChantierNameManipulateur::BasicAlloc(aOri);
    std::list<std::string> aFiles=ManC->StdGetListOfFile(aFullName);
    
    std::vector<std::string> aNameIm;
    
    for(std::list<std::string>::iterator I=aFiles.begin();I!=aFiles.end();I++)
    {	
        std::cout << " - " << *I << std::endl;
        aNameIm.push_back(I->substr(12,I->size()-16));
    }
    std::cout<<"Total: "<<aNameIm.size()<<" files."<<std::endl;
    
    std::string aPat="";
    
    for(unsigned int i=0;i<aNameIm.size()-1;i++)
	{
		aPat = aPat + aNameIm.at(i) + "|";
	}
		
	aPat = aPat + aNameIm.at(aNameIm.size()-1);
	
	if(aShow)
		std::cout << "Pat = \"" << aPat << "\"" << std::endl;
    
    
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
//to add : toutes les pairs possibles (en option)
int GenFilePairs_main(int argc,char ** argv)
{
    std::string aImg, aFullPat, aOut="NameCple.xml";
	
	ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(aImg, "Image Name or Pattern")
                << EAMC(aFullPat, "Pattern of Images", eSAM_IsPatFile),
                
    //optional arguments
    LArgMain()  << EAM(aOut, "Out", false, "Output .xml file ; Def=NameCple.xml")
  
    );
    
    if (MMVisualMode) return EXIT_SUCCESS;
    
    // Initialize name manipulator & files
    std::string aDirImages1,aPatIm1, aDirImages2,aPatIm2;
    
    SplitDirAndFile(aDirImages1,aPatIm1,aImg);
    SplitDirAndFile(aDirImages2,aPatIm2,aFullPat);
    
    std::cout<<"Working dir 1: "<<aDirImages1<<std::endl;
    std::cout<<"Working dir 2: "<<aDirImages2<<std::endl;
    
    std::cout<<"Image/Pat 1:"<<aPatIm1<<std::endl;
    std::cout<<"Images pattern 2: "<<aPatIm2<<std::endl;
    
    cInterfChantierNameManipulateur * aICNM1=cInterfChantierNameManipulateur::BasicAlloc(aDirImages1);
    cInterfChantierNameManipulateur * aICNM2=cInterfChantierNameManipulateur::BasicAlloc(aDirImages2);
    
    const std::vector<std::string> aSetIm1 = *(aICNM1->Get(aPatIm1));
    const std::vector<std::string> aSetIm2 = *(aICNM2->Get(aPatIm2));
    
    cSauvegardeNamedRel  aRelIm;
    
    for(unsigned i=0; i<aSetIm1.size(); i++)
    {
		for(unsigned j=0; j<aSetIm2.size(); j++)
		{
			cCpleString aCpl(aSetIm1.at(i),aSetIm2.at(j));
			aRelIm.Cple().push_back(aCpl);
		}
	}
    
      MakeFileXML(aRelIm,aDirImages1+aOut);
	
	return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
int CleanPatByOri_main(int argc,char ** argv)
{
	std::string aFullName, aOri, aDir, aPat, aOut;
	bool aShow=false;
	 
	ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
	LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)")
				<< EAMC(aOri, "Ori Folder", eSAM_IsExistDirOri),
	
	LArgMain()  << EAM(aShow, "Show", false, "Display Pattern to use in cmd line ; Def=false",eSAM_IsBool)
				<< EAM(aOut, "Out", false, "Output Folder Name for Images NOT will be used")
	);
	
	if (MMVisualMode) return EXIT_SUCCESS;
	
	MakeFileDirCompl(aOri);
    ELISE_ASSERT(ELISE_fp::IsDirectory(aOri),"ERROR: Input orientation not found!");

    SplitDirAndFile(aDir, aPat, aFullName);

    cInterfChantierNameManipulateur *aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string> aLFilePat = aICNM->StdGetListOfFile(aPat);
     
     
	std::string aFullPatOri="Orientation-*.*xml";
    cInterfChantierNameManipulateur *ManC=cInterfChantierNameManipulateur::BasicAlloc(aOri);
    std::list<std::string> aLFilesOri=ManC->StdGetListOfFile(aFullPatOri);
    
    std::vector<std::string> aNameImOri;
    
    for(std::list<std::string>::iterator I=aLFilesOri.begin();I!=aLFilesOri.end();I++)
    {	
        //std::cout << " - " << *I << std::endl;
        aNameImOri.push_back(I->substr(12,I->size()-16));
    }
    
    std::cout<<"Total images "<<aOri<<" : "<<aNameImOri.size()<<" files."<<std::endl;
    std::cout<<"Total images input Pattern: "<<aLFilePat.size()<<" files."<<std::endl;
    
    std::vector<std::string> aVImgsTrash;
    
    for(std::list<std::string>::iterator iT=aLFilePat.begin(); iT!=aLFilePat.end(); iT++)
    {
		unsigned int aCmpt=0;
		for (unsigned aP=0; aP<aNameImOri.size(); aP++)
		{
			if(iT->compare(aNameImOri.at(aP)) == 0)
				break;
				
			aCmpt++;
		}
		
		if(aCmpt == aNameImOri.size())
		{
			std::cout << "IMG PAT : " << *iT << " NOT IN ORI " << std::endl;
			aVImgsTrash.push_back(*iT);
		}
		
	}
	std::cout<<"Total images NOT OK: "<<aVImgsTrash.size()<<" files."<<std::endl;
	
	if(aOut == "")
	{
		aOut="TRASH";
	}
	
	if(aVImgsTrash.size() > 0)
	{
		ELISE_fp::MkDirSvp(aOut);
		for (unsigned aK=0; aK<aVImgsTrash.size(); aK++)
		{
			ELISE_fp::MvFile(aVImgsTrash.at(aK),aOut);
		}
	}

	
	return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
int RedImgsByN_main(int argc,char** argv)
{
	std::string aFullName, aDir, aPat, aOut="Selected_Images";
	int aPas;
	bool aShow=false;
	
	ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
	LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pat)")
				<< EAMC(aPas,"1/Pas"),
	LArgMain()  << EAM(aShow, "Show", false, "Display Pattern to use in cmd line ; Def=false",eSAM_IsBool)
				<< EAM(aOut, "Out", false, "Output Folder Name for Images that will be used ; Def=Selected_Images")
	);
	
	if (MMVisualMode) return EXIT_SUCCESS;
	
	//check if Output folder is not existing create it
	ELISE_fp::MkDirSvp(aOut);

    SplitDirAndFile(aDir, aPat, aFullName);

    cInterfChantierNameManipulateur *aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPat));
    
    for (unsigned int aP=0; aP<aSetIm.size(); aP=aP+aPas)
    {
		ELISE_fp::CpFile(aSetIm.at(aP),aOut);
	}
	
	return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
int MvImgsByFile_main(int argc,char** argv)
{
	std::string aDir="";
	std::string aFile=""; //Schnaps_poubelle.txt (format)
	bool aShow=false;
	std::string aTrashName="Poubelle";
	
	ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
	LArgMain()  << EAMC(aDir,"Directory")
				<< EAMC(aFile,"File with images names by line"),
	LArgMain()  << EAM(aShow, "Show", false, "Display Pattern to use in cmd line ; Def=false",eSAM_IsBool)
	);
	
	if (MMVisualMode) return EXIT_SUCCESS;
	
	ELISE_fp::MkDirSvp(aTrashName);
	
	//read input file
    ifstream aFichier((aDir + aFile).c_str());

    if(aFichier)
    {
		std::string aLine;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLine);
			if(!aLine.empty())
			{
				ELISE_fp::MvFile(aLine,aTrashName);
				std::cout << " Move image : " << aLine << "--> " << aTrashName << std::endl;
			}
		}
	aFichier.close();
	}
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
	
	return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
class cTestElParseDir : public ElActionParseDir
{
    public :
        void act(const ElResParseDir & aRPD) 
        {
            //std::cout << aRPD.name() << "\n";
        }
};

int TestElParseDir_main(int argc,char ** argv)
{
     //cTestElParseDir aTPD;
     //ElParseDir("/home/marc/TMP/EPI/Croco/",aTPD,1000);

     return EXIT_SUCCESS;
}


/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
