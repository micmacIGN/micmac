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

template <typename T>
std::string NumberToString(T Number)
{
	ostringstream ss;
    ss << Number;
    return ss.str();
}

struct HomPSOrima{
	std::string ImgName;
	int Id;
	Pt2dr Coord;
	int Flag;
	std::string Status;
};

struct PatisImg{
	int aIdImg1;
	vector<ElPackHomologue> aPack;
	vector<int> aVIdImgs2nd;
};

vector<string> VImgs;
vector<int> VId;
vector<int> VIdUnique;
vector<Pt2dr> VCoords;
vector<string> VImgsUnique;
vector<PatisImg> VHomol;


class cCPSH_Appli
{
	public :
		cCPSH_Appli(int argc,char ** argv);
		void GetVImgsFromId(int aId, std::vector<std::string> & VImgs,std::vector<Pt2dr> & VPts);
		
	private :
		std::string mDir;
		std::string mPSFile;
};

cCPSH_Appli::cCPSH_Appli(int argc,char ** argv)
{
	std::string aOut, aPat;
	bool a2W=true;
	Pt2dr aSize;
	Pt2dr aC;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mDir, "Directory")
					 << EAMC(aC, "[Cx Cy] values from .xml PhotoScan file")
					 << EAMC(aSize, "Size of photosite in mm unit")
					 << EAMC(mPSFile, "PhotoScan ORIMA input File", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output Homol Name ; Def=Homol_PS")
					 << EAM(aPat,"Pat",false,"Pattern of images")
					 << EAM(a2W,"Way",false,"homologue in 2 way", eSAM_IsBool)
    );
    
	//~ if(EAMIsInit(&aSize))
    //~ {
        //~ ELISE_ASSERT(aSize.x >0 ,"Value of Size images needs to be sepicified");
    //~ }
    //~ else
    //~ {
			//~ cout<<"Warning : Image frame in Photoscan format (not compatible with MicMac) - give Size"<<endl;
	//~ }
    //name output homol folder
    if (aOut=="")
    {
		aOut = "_PS";
    }
    
    std::vector<HomPSOrima> aVHPSO;
    
    //read input file : file format to read :    D0003736.JPG               0           7.997          -1.199   0   M
    ifstream aFichier((mDir + mPSFile).c_str());
    if(aFichier)
    {
		std::string aLine;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLine,'\n');
			
			if(aLine.size() != 0)
			{
				char *aBuffer = strdup((char*)aLine.c_str());
				std::string aName = strtok(aBuffer," ");
				char *aId = strtok( NULL, " " );
				char *aI = strtok( NULL, " " );
				char *aJ = strtok( NULL, " " );
				char *aFlag = strtok( NULL, " " );
				char *aStatus = strtok( NULL, " " );
				
				HomPSOrima aHPSO;
				aHPSO.ImgName = aName;
				aHPSO.Id = atoi(aId);
				Pt2dr aC(atof(aI),atof(aJ));
				aHPSO.Coord = aC;
				aHPSO.Flag = atoi(aFlag);
				aHPSO.Status = aStatus;
				
				aVHPSO.push_back(aHPSO);

				VImgs.push_back(aName);
				VId.push_back(atoi(aId));
				VCoords.push_back(aC);
				int aIdU = atoi(aId);
				if ( !(std::find(VImgsUnique.begin(), VImgsUnique.end(), aName) != VImgsUnique.end() ))
				{
					VImgsUnique.push_back(aName);
				}
			    if ( !(std::find(VIdUnique.begin(), VIdUnique.end(), aIdU) != VIdUnique.end() ))
				{
					VIdUnique.push_back(aIdU);
					//~ cout<<aIdU<<endl;
				}
			}
		}
		
		aFichier.close();
        cout<<"Nb Imgs Uniq : " << VImgsUnique.size()<< endl;
        cout<<"Nb IdUnique : " << VIdUnique.size()<< endl;

	}
	
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
	
	//
	for (uint aKImg=0; aKImg<VImgsUnique.size(); aKImg++)
	{
		PatisImg aPatis;
		aPatis.aIdImg1 = aKImg;
		for (uint aKImg=0; aKImg<VImgsUnique.size(); aKImg++)
		{
			aPatis.aPack.push_back(ElPackHomologue());
			aPatis.aVIdImgs2nd.push_back(aKImg);
		}
		VHomol.push_back(aPatis);
	}
	
	//for each unique tieP : get vector of images
	for (uint aKId=0; aKId<VIdUnique.size(); aKId++)
	{
        if (aKId % int(VIdUnique.size()/300) == 0)
          cout<<"["<<(aKId*100.0/VIdUnique.size())<<" %] --> fusion Id : "<<aKId<<"/"<<VIdUnique.size()<<endl;
		std::vector<std::string> RVImgs;
		std::vector<Pt2dr> RVPts;
		cCPSH_Appli::GetVImgsFromId(aKId, RVImgs, RVPts);
		ELISE_ASSERT(RVImgs.size() == RVPts.size(), "ERROR: Incoherent size");
		
		for (uint i=0; i<RVImgs.size(); i++)
		{
			for (uint j=i+1; j<RVImgs.size(); j++ )
			{
				ptrdiff_t pos_img1 = std::find(VImgsUnique.begin(), VImgsUnique.end(), RVImgs[i]) - VImgsUnique.begin();
				ELISE_ASSERT(pos_img1 < int(VImgsUnique.size()), "ERROR: pos_img1");
					//cout<<"Found at " <<index<<RVImgs[i]<<" "<<VImgsUnique[index]<<endl;
					ptrdiff_t pos_img2 = std::find(VImgsUnique.begin(), VImgsUnique.end(), RVImgs[j]) - VImgsUnique.begin();
					ELISE_ASSERT(pos_img2 < int(VImgsUnique.size()), "ERROR: pos_img2");
					
					Pt2dr aPt1;
					aPt1.x = aC.x+(RVPts[i].x/aSize.x);
					aPt1.y = aC.y - (RVPts[i].y/aSize.y);
					
					Pt2dr aPt2;
					aPt2.x = aC.x+(RVPts[j].x/aSize.x);
					aPt2.y = aC.y - (RVPts[j].y/aSize.y);
					
					VHomol[pos_img1].aPack[pos_img2].Cple_Add(ElCplePtsHomologues(aPt1, aPt2));
					VHomol[pos_img1].aVIdImgs2nd.push_back(int(pos_img2));
					
					if (a2W)
					{
						VHomol[pos_img2].aPack[pos_img1].Cple_Add(ElCplePtsHomologues(aPt2, aPt1));
					}
			}
		}
	  }
		//ecriture
		std::string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aOut)
                        +  std::string("@")
                        +  std::string("dat");
        cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(mDir);
		for (uint i=0; i<VHomol.size(); i++)
		{
            if (i % int(VHomol.size()/100) == 0)
              cout<<"["<<(i*100.0/VHomol.size())<<" %] --> write Id : "<<i<<"/"<<VHomol.size()<<endl;
			PatisImg img = VHomol[i];
			for (uint j=0; j<img.aPack.size(); j++)
			{
				std::string clePack = aICNM->Assoc1To2(aKHOutDat, VImgsUnique[img.aIdImg1], VImgsUnique[img.aVIdImgs2nd[j]], true);
				if (img.aPack[j].size() > 0)
					img.aPack[j].StdPutInFile(clePack);
			}
			
		}

		
		
		
		//~ cout<<"Id = "<<aKId<<endl;
		//~ for(uint i=0; i<RVImgs.size(); i++)
		//~ {
			//~ cout<<"	++Img: "<<RVImgs[i]<<" - Pts : "<<RVPts[i]<<endl;
		//~ }
	
    
}

void cCPSH_Appli::GetVImgsFromId(int aId, std::vector<std::string> & RVImgs,std::vector<Pt2dr> & RVPts)
{
	for (uint aKId=0; aKId<VId.size(); aKId++)
	{
		if (VId[aKId] == aId)
		{
			RVImgs.push_back(VImgs[aKId]);
			RVPts.push_back(VCoords[aKId]);
		}
	}
}

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

/***********************************************************************/
int SplitGCPsCPs_main(int argc,char **argv)
{
	cSFGC_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}

/***********************************************************************/
int ConcateMAF_main(int argc,char **argv)
{
	std::string aDir, aPat, aOut="";
	std::string aFullName;
	bool aRmFiles=true;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aFullName, ".xml Files Full Pattern", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output File Name of concatenated files")
					 << EAM(aRmFiles,"aRmFiles",false,"Remove .xml files after concatenation; def=true")
    );
    
    if(aOut == "")
    {
		aOut = "Concat_File.xml";
	}
	
	SplitDirAndFile(aDir, aPat, aFullName);

	cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    
    std::list<std::string> aLFile = aICNM->StdGetListOfFile(aPat);
    
	//              aPts               ImName                   ImMes
	std::map<std::string,std::map<std::string,cOneMesureAF1I> > aDicoMap;
	
	// set of imgs
	std::set<std::string> aSIm; 
    
    for (std::list<std::string>::iterator iT1 = aLFile.begin() ; iT1 != aLFile.end() ; iT1++)
    {
		std::cout << "File = " << *iT1 << std::endl;
		cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aDir + *iT1,SetOfMesureAppuisFlottants);
		std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();
		
		for (std::list<cMesureAppuiFlottant1Im>::iterator iT2 = aLMAF.begin() ; iT2 != aLMAF.end() ; iT2++)
		{
			
			std::list<cOneMesureAF1I> & aLOMA = iT2->OneMesureAF1I();

			for (std::list<cOneMesureAF1I>::iterator iPt = aLOMA.begin() ; iPt!= aLOMA.end() ; iPt++)
			{
				aDicoMap[iPt->NamePt()][iT2->NameIm()] = *iPt;
				//std::cout << "xy=" << iPt->PtIm() << "\n"; 
		
				aSIm.insert(iT2->NameIm());
			}
		}
	}

	//write new
	cSetOfMesureAppuisFlottants  aVSMAF;
	std::list< cMesureAppuiFlottant1Im > aLMAF;
	
	//iterate images	
	for(std::set<std::string>::iterator aM=aSIm.begin(); aM!=aSIm.end(); aM++)
	{
		std::string               aImCur = *aM;
		std::list<cOneMesureAF1I> aLPtIm;
		cMesureAppuiFlottant1Im   aMAFCur;

		aMAFCur.NameIm() = aImCur;		
		
		//iterate points
		std::map<std::string,std::map<std::string,cOneMesureAF1I> >::iterator iPt = aDicoMap.begin();
		for( ; iPt != aDicoMap.end(); iPt++)
		{
			std::map<std::string,cOneMesureAF1I>::iterator iIm = iPt->second.find(aImCur);
			if( iIm != iPt->second.end())
				aLPtIm.push_back(iIm->second);
		}

		aMAFCur.OneMesureAF1I() = aLPtIm;
		aLMAF.push_back(aMAFCur);
	}

	aVSMAF.MesureAppuiFlottant1Im() = aLMAF;

	MakeFileXML(aVSMAF,aDir + aOut);
	
	if(aRmFiles)
	{
		for (std::list<std::string>::iterator iT1 = aLFile.begin() ; iT1 != aLFile.end() ; iT1++)
		{
			std::cout << "Remove File = " << aDir + *iT1 << std::endl;
			ELISE_fp::RmFileIfExist((aDir + *iT1));
		}
	}

	return EXIT_SUCCESS;
}	

/***********************************************************************/
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

struct ImgPS{
	std::string Name;
	Pt3dr Pos;
	Pt3dr Ori;
	double R11;
	double R12;
	double R13;
	double R21;
	double R22;
	double R23;
	double R31; 
	double R32; 
	double R33;
};

int CleanTxtPS_main(int argc,char ** argv)
{
	std::string aDir, aInFile, aOut;
	bool aFormat=false;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aDir, "Directory")
					 << EAMC(aInFile, ".txt Output PhotoScan File", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output File Name ; Def=FileName_NXYZWPK.txt")
                     << EAM(aFormat, "Format", false, "Aff format in file ; Def=false", eSAM_IsBool)
    );
    
    //name output (.xml) file
    if (aOut=="")
    {
		aOut = StdPrefixGen(aInFile) + "_NXYZWPK.txt";
    }
    
    std::vector<ImgPS> aVImgPS;
    
    //read input file
    ifstream aFichier((aDir + aInFile).c_str());
    
    if(aFichier)
    {
		std::string aLine;
        
        while(!aFichier.eof())
        {
			getline(aFichier,aLine,'\n');
			
			if(aLine.size() != 0)
			{
				if (aLine.compare(0,1, "#") == 0)
				{
					std::cout << "SKIP COMMENT = " << aLine << std::endl;
				}
				//# PhotoID, X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33
				else
				{
					char *aBuffer = strdup((char*)aLine.c_str());
					std::string aName = strtok(aBuffer,"	");
					char *aX = strtok( NULL, "	" );
					char *aY = strtok( NULL, "	" );
					char *aZ = strtok( NULL, "	" );
					char *aO = strtok( NULL, "	" );
					char *aP = strtok( NULL, "	" );
					char *aK = strtok( NULL, "	" );
					char *aR11 = strtok( NULL, "	" );
					char *aR12 = strtok( NULL, "	" );
					char *aR13 = strtok( NULL, "	" );
					char *aR21 = strtok( NULL, "	" );
					char *aR22 = strtok( NULL, "	" );
					char *aR23 = strtok( NULL, "	" );
					char *aR31 = strtok( NULL, "	" );
					char *aR32 = strtok( NULL, "	" );
					char *aR33 = strtok( NULL, "	" );
					
					ImgPS aImgPs;
					
					aImgPs.Name = aName;
					Pt3dr aPos(atof(aX),atof(aY),atof(aZ));
					aImgPs.Pos = aPos;
					Pt3dr aOri(atof(aO),atof(aP),atof(aK));
					aImgPs.Ori = aOri;
					aImgPs.R11 = atof(aR11);
					aImgPs.R12 = atof(aR12);
					aImgPs.R13 = atof(aR13);
					aImgPs.R21 = atof(aR21);
					aImgPs.R22 = atof(aR22);
					aImgPs.R23 = atof(aR23);
					aImgPs.R31 = atof(aR31);
					aImgPs.R32 = atof(aR32);
					aImgPs.R33 = atof(aR33);
					
					aVImgPS.push_back(aImgPs);
					
				}
			}
			
		}
		
		aFichier.close();
		
	}
	
	else
    {
		std::cout<< "Error While opening file" << '\n';
	}
	
	
	if (!MMVisualMode)
	{			
		FILE * aFP = FopenNN(aOut,"w","CleanTxtPS_main");
				
		cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);
		
		if (aFormat)
		{
			std::string aAddFormat = "#F=N_X_Y_Z_W_P_K";
			
			fprintf(aFP, "%s \n",  aAddFormat.c_str());
		}
				
		for (unsigned int aK=0 ; aK<aVImgPS.size() ; aK++)
		{
			fprintf(aFP,"%s %lf %lf %lf %lf %lf %lf \n",aVImgPS[aK].Name.c_str(),aVImgPS[aK].Pos.x,aVImgPS[aK].Pos.y,aVImgPS[aK].Pos.z, aVImgPS[aK].Ori.x,aVImgPS[aK].Ori.y,aVImgPS[aK].Ori.z);
		}
		
		ElFclose(aFP);
			
	}
    
	return EXIT_SUCCESS;
}

int CheckPatCple_main(int argc,char ** argv)
{
	std::string aFullName, aInFile, aOut, aDir, aPat;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aFullName,"Full Name (Dir+Pat)")
					 << EAMC(aInFile, ".xml Name Cple File", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output File Name ; Def=Corr-NameFile.xml")
    );
    
    SplitDirAndFile(aDir, aPat, aFullName);

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::list<std::string> aLFile = aICNM->StdGetListOfFile(aPat);
    
    //load .xml file
    cSauvegardeNamedRel aDico = StdGetFromPCP(aInFile,SauvegardeNamedRel);
    std::vector<cCpleString> & aVCple = aDico.Cple();
    
    std::cout << "aVCple.size() = " << aVCple.size() << std::endl;
    
    //list of images to add in .xml file
    std::vector<std::string> aVImgs;
    
    //check for each img if it is present in the .xml file
    for (std::list<std::string>::iterator iT1 = aLFile.begin() ; iT1 != aLFile.end() ; iT1++)
    {
		//~ std::cout << "Image = " << *iT1 << std::endl;
		unsigned int aCmpt=0;
		for (unsigned int aP=0; aP<aVCple.size(); aP++)
		{
			//std::cout << "Cple : " <<  aVCple.at(aP).N1() << " " << aVCple.at(aP).N2() << std::endl;
			if(iT1->compare(aVCple.at(aP).N1())==0 || iT1->compare(aVCple.at(aP).N2())==0)
			{
				std::cout << "IMG PAT : " << *iT1 << " OK " << std::endl;
				break;
			}
			aCmpt++;
		}
		//~ std::cout << "aCmpt = " << aCmpt << std::endl;
		if(aCmpt == aVCple.size())
		{
			std::cout << "IMG PAT : " << *iT1 << " NOT OK " << std::endl;
			aVImgs.push_back(*iT1);
		}
	}
	
	//generate all Cple for aVImgs
	for (unsigned int aK=0; aK<aVImgs.size(); aK++)
	{
		for (std::list<std::string>::iterator iT2 = aLFile.begin() ; iT2 != aLFile.end() ; iT2++)
		{
			if(iT2->compare(aVImgs.at(aK)) != 0)
			{
				cCpleString aCpl(aVImgs.at(aK), *iT2);
				aDico.Cple().push_back(aCpl);
			}
		}
	}
	
	//generate new .xml file
	if(aOut == "")
    {
		aOut = "Corr-" + StdPrefixGen(aInFile) + ".xml";
	}
	
	MakeFileXML(aDico,aOut);
    
	return EXIT_SUCCESS;
}

int ConvPSHomol2MM_main(int argc,char ** argv)
{
	cCPSH_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}


int SplitPatByCam_main(int argc,char ** argv)
{
	std::string aFullName,aPref="Pattern_Cam_",aPat,aDir;
	int aCompt=0;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(aFullName,"Full Name (Dir+Pat)"),
          LArgMain() << EAM(aPref,"Pref",false,"Prefix Of Folders ; Def=Pattern_Cam_")
    );
    
    SplitDirAndFile(aDir, aPat, aFullName);
    
    std::cout << "aDir = " << aDir << std::endl;
    std::cout << "aPat = " << aPat << std::endl;
    std::cout << "aFullName = " << aFullName << std::endl;
    

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    //~ std::vector<std::string> aLFile = aICNM->StdGetListOfFile(aPat);
    std::vector<std::string> aLFile = *(aICNM->Get(aPat));
    
    std::string aTmpDir = "Tmp-MM-Dir";
    
    if(!ELISE_fp::IsDirectory(aTmpDir))
    {
		system_call((std::string("mm3d MMXmlXif \"")+aPat+"\"").c_str());
	}
	
	std::vector<cXmlDate> aVExifDate;
    cInterfChantierNameManipulateur * aICNMX = cInterfChantierNameManipulateur::BasicAlloc(aTmpDir);
    std::vector<std::string> Img_xif = *(aICNMX->Get(".*xml"));
	
	//create first folder: at least one pattern
	std::cout << "aPref = " << aPref << std::endl;
	std::cout << "NumberToString(aCompt) = " << NumberToString(aCompt) << std::endl;
	std::string aFFolder = aPref + NumberToString(aCompt);
	std::cout << "aFFolder = " << aFFolder << std::endl;
	ELISE_fp::MkDirSvp(aFFolder);
	
	//first image
	cXmlXifInfo aXmlXifInfoIC=StdGetFromPCP(aTmpDir+"/"+Img_xif.at(0),XmlXifInfo);
	std::string aTypeIC = aXmlXifInfoIC.Cam().Val();
	std::cout << "aTypeIC = " << aTypeIC << std::endl;
	ELISE_fp::MvFile(aLFile.at(0),aFFolder);
	std::cout << "Deplace Image : "<< aLFile.at(0) << "dans dossier " << aFFolder<<std::endl;
	
	for (unsigned int aK=0; aK<aLFile.size()-1; aK++)
	{
		
		cXmlXifInfo aXmlXifInfoIS=StdGetFromPCP(aTmpDir+"/"+Img_xif.at(aK+1),XmlXifInfo);
		std::string aTypeIS = aXmlXifInfoIS.Cam().Val();
		std::cout << "aTypeIS = " << aTypeIS << std::endl;
		
		if(aTypeIC.compare(aTypeIS) == 0)
		{
			std::cout << "OUI PAREIL !" << std::endl;
			ELISE_fp::MvFile(aLFile.at(aK+1),aFFolder);
			std::cout << "Deplace Image : "<< aLFile.at(aK+1) << "dans dossier " << aFFolder<<std::endl;
			aTypeIC=aTypeIS;
			std::cout << "aTypeIC () = " << aTypeIC << std::endl;
		}
		else
		{
			aCompt++;
			std::cout << "aCompt = " << aCompt << std::endl;
			std::string aNFolder = aPref + NumberToString(aCompt);
			std::cout << "aNFolder = " << aNFolder << std::endl;
			ELISE_fp::MkDirSvp(aNFolder);
			ELISE_fp::MvFile(aLFile.at(aK+1),aNFolder);
			std::cout << "Deplace Image : "<< aLFile.at(aK+1) << "dans dossier " << aNFolder<<std::endl;
			aTypeIC=aTypeIS;
			aFFolder=aNFolder;
			std::cout << "aTypeIC ()() = " << aTypeIC << std::endl;
		}
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
