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

//to do :
//* if name points not given generate them auto (random mode)
//* if all possible combinations asked, generate them too

class cSFGC_Appli
{
	public :
		cSFGC_Appli(int argc,char ** argv);
		void ShowPoints(std::string Tag, std::vector<std::string> aVS);
		void CheckDoublon(std::vector<std::string> aVS1, std::vector<std::string> aVS2);
		void GenerateXmlFile(std::vector<std::string> aVS, std::list<cOneAppuisDAF> aLPts, std::string aOutFile);
		std::vector<std::string> LeftPoints(std::vector<std::string> aVSGlob, std::vector<std::string> aVSSpec);
		std::vector<std::string> GetVPtsName(std::list<cOneAppuisDAF> aLOADAF);
	private :
		std::string mDir;
		std::string mFile;
};

std::vector<std::string> cSFGC_Appli::GetVPtsName(std::list<cOneAppuisDAF> aLOADAF)
{
	std::vector<std::string> aVPts;
	
	for (std::list<cOneAppuisDAF>::iterator iT1 = aLOADAF.begin() ; iT1 != aLOADAF.end() ; iT1++)
	{
		aVPts.push_back(iT1->NamePt());
	}
	return aVPts;
}

std::vector<std::string> cSFGC_Appli::LeftPoints(std::vector<std::string> aVSGlob, std::vector<std::string> aVSSpec)
{
	std::vector<std::string> aVL;
	
	for (unsigned int aP=0; aP<aVSSpec.size(); aP++)
	{
		int aT=0;
		
		for (unsigned int aK=0; aK<aVSGlob.size(); aK++)
		{
			if(aVSSpec.at(aP).compare(aVSGlob.at(aK)) == 0)
				aT++;
		}
		
		if(aT == 0)
			aVL.push_back(aVSSpec.at(aP));
	}
	
	return aVL;
}

void cSFGC_Appli::GenerateXmlFile(std::vector<std::string> aVS, std::list<cOneAppuisDAF> aLPts, std::string aOutFile)
{
	
	//if file is empty do not create it
	if(aVS.size() == 0)
	{
		std::cout << "File " << aOutFile << " is empty! Not Created! " << std::endl;
	}
	
	else
	{
		cDicoAppuisFlottant aDico;
		
		for(std::list<cOneAppuisDAF>::iterator iT = aLPts.begin(); iT != aLPts.end(); iT++)
		{			
			for(unsigned aK=0; aK<aVS.size(); aK++)
			{
				if(aVS.at(aK) == iT->NamePt())
				{
					cOneAppuisDAF aOA;
					aOA.Pt() = iT->Pt();
					aOA.NamePt() = iT->NamePt();
					aOA.Incertitude() = iT->Incertitude();
				
					aDico.OneAppuisDAF().push_back(aOA);
				}
			}
		}
				
		if(aDico.OneAppuisDAF().size() == 0)
		{
			std::cout << "No match between names points and name points in the file!" << std::endl;
		}
		
		else
		{
			 MakeFileXML(aDico,aOutFile);
		}
	}
}

void cSFGC_Appli::CheckDoublon(std::vector<std::string> aVS1, std::vector<std::string> aVS2)
{
	for (unsigned int aK=0; aK<aVS1.size(); aK++)
	{
		for (unsigned int aP=0; aP<aVS2.size(); aP++)
		{
			if(strcmp(aVS1.at(aK).c_str(), aVS2.at(aP).c_str()) == 0)
			{
				std::cout << "Warning : point " << aVS1.at(aK) << " is used as GCPs & CPs !" << std::endl; 
			}
		}
	}
}

void cSFGC_Appli::ShowPoints(std::string Tag, std::vector<std::string> aVS)
{
	std::cout << "Points choosed as " << Tag << " : " << std::endl;
	
	for (unsigned int aK=0; aK<aVS.size(); aK++)
	{
		std::cout << Tag << "[" << aK << "]=" << aVS.at(aK) << std::endl;
	}
}

cSFGC_Appli::cSFGC_Appli(int argc,char ** argv)
{
	std::vector<std::string> aVGCPs, aVCPs;
	std::string aOutGCPs, aOutCPs;
	int aNbrMinGCPs=3;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir, "Directory")
					 << EAMC(mFile, ".xml File", eSAM_IsExistFile),
          LArgMain() << EAM(aVGCPs,"GCPs",false,"Names of points to consider as GCPs")
                     << EAM(aVCPs,"CPs",false,"Names of points to consider as CPs")
                     << EAM(aOutGCPs,"OutGCPs",false,"output GCPs .xml file name ; Def=GCPs-File.xml")
                     << EAM(aOutCPs,"OutCPs",false,"output CPs .xml file name ; Def=CPs-File.xml")
                     << EAM(aNbrMinGCPs,"MinNbrGCPs",false,"Minimum Nubmer of points in GCPs .xml file for random mode; Def=3")
    );
    
    const std::string aTagGCPs = "GCPs";
    const std::string aTagCPs = "CPs";
    
    //correct output name
    if(aOutGCPs=="")
    {
		aOutGCPs = "GCPs-" +  StdPrefixGen(mFile) + ".xml";
	}
	
	//correct output name
	if(aOutCPs=="")
	{
		aOutCPs = "CPs-" +  StdPrefixGen(mFile) + ".xml";
	}
	
	//read .xml file
	cDicoAppuisFlottant aDico = StdGetFromPCP(mFile,DicoAppuisFlottant);
	std::list<cOneAppuisDAF> aOneAppuisDAFList = aDico.OneAppuisDAF();
	std::vector<std::string> aVPts = GetVPtsName(aOneAppuisDAFList);
	
	//if no name given, the programm needs to do it;
	if(aVGCPs.size() == 0 && aVCPs.size() ==0)
	{
		//how ? mode based on geometry to keep certain homogeneité
	}
	
	//classic case where the user choose
	else
	{
		
		//check if there is points used as GCPs and CPs
		CheckDoublon(aVGCPs,aVCPs);
		
		//if at least GCPs vector names are given, the rest is considered as CPs (vice-versa)
		if(aVGCPs.size()==0 && aVCPs.size()!=0)
			aVGCPs=LeftPoints(aVCPs,aVPts);
		else if(aVGCPs.size()!=0 && aVCPs.size()==0)
			aVCPs=LeftPoints(aVGCPs,aVPts);
			
		//show GCPs
		ShowPoints(aTagGCPs,aVGCPs);
		
		//show CPs
		ShowPoints(aTagCPs,aVCPs);
		
		//generate GCPs file if not empty
		GenerateXmlFile(aVGCPs,aOneAppuisDAFList,aOutGCPs);
		
		//generate CPs file if not empty
		GenerateXmlFile(aVCPs,aOneAppuisDAFList,aOutCPs);
	}
}

int SplitGCPsCPs_main(int argc,char **argv)
{
	cSFGC_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}

int ConcateMAF_main(int argc,char **argv)
{
	std::string aDir, aPat, aOut="";
	std::string aPatFiles;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aPatFiles, ".xml Files Full Pattern", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output File Name of concatenated files")
    );
    
    if(aOut == "")
    {
		aOut = "Concat_File.xml";
	}
	
	SplitDirAndFile(aDir, aPat, aPatFiles);
	
	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string> aLFile = aICNM->StdGetListOfFile(aPatFiles);
    
    cSetOfMesureAppuisFlottants aVSMAF;
    
    for (std::list<std::string>::iterator iT1 = aLFile.begin() ; iT1 != aLFile.end() ; iT1++)
    {
		std::cout << "File = " << *iT1 << std::endl;
		cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(*iT1,SetOfMesureAppuisFlottants);
		std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();
		
		for (std::list<cMesureAppuiFlottant1Im>::iterator iT2 = aLMAF.begin() ; iT2 != aLMAF.end() ; iT2++)
		{
			aVSMAF.MesureAppuiFlottant1Im().push_back(*iT2);
		}
	}
	
	MakeFileXML(aVSMAF,aOut);
	
	return EXIT_SUCCESS;
}

int ConvSensXml2Txt_main(int argc,char **argv)
{
	std::string aDir, aInFile, aOut="";
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aDir, "Directory")
					 << EAMC(aInFile, ".xml Sensibility Input File", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output File Name ; Def=File.txt")
    );
    
    if(aOut == "")
    {
		aOut = StdPrefixGen(aInFile) + ".txt";
	}
	
	//read .xml file
    cXmlNameSensibs aDico = StdGetFromPCP(aInFile,XmlNameSensibs);
	std::vector<cSensibDateOneInc> aSensDOI = aDico.SensibDateOneInc();
	
		
	//write data in .txt file
	if(!MMVisualMode)
	{
		FILE * aFP = FopenNN(aOut,"w","ConvSensXml2Txt_main");
		cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);
		
		for (unsigned int itP=0; itP<aSensDOI.size(); itP ++)
		{
			std::cout << aSensDOI.at(itP).NameInc().c_str() << std::endl;
			fprintf(aFP,"%s %s %lf %lf %lf", aSensDOI.at(itP).NameBloc().c_str(), aSensDOI.at(itP).NameInc().c_str(), aSensDOI.at(itP).SensibParamDir(), aSensDOI.at(itP).SensibParamInv(), aSensDOI.at(itP).SensibParamVar());
			
			fprintf(aFP,"\n");
		}
		
		ElFclose(aFP);
	}
	
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
