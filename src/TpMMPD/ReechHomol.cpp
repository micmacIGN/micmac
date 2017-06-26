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
#include <iostream>
#include <string>

struct ImgT{
	std::string ImgName;
	double ImgTemp;
};

class cReechHomol_Appli
{
	public :
		cReechHomol_Appli (int argc,char ** argv);
		void ConvertHomol (string & aFullPat, string & aSHIn);
		void CorrHomolFromTemp (string & aDir, string & aSHIn, string & aTempFile, std::vector<ImgT> & aVSIT, string & aExt);
		void ReadImgTFile (string & aDir, string aTempFile, std::vector<ImgT> & aVSIT, std::string aExt);
	private :
		std::string mDir;
};

void cReechHomol_Appli::ConvertHomol(string & aFullPat, string & aSHIn)
{
	string aComConvertHomol = MM3dBinFile("TestLib Homol2Way ")
								+ aFullPat
                                + " SH=" + aSHIn
                                + " SHOut=_txt"
                                + " IntTxt=0" 
                                + " ExpTxt=1"
                                + " OnlyConvert=1";
    system_call(aComConvertHomol.c_str());
}

void cReechHomol_Appli::ReadImgTFile(string & aDir, string aTempFile, std::vector<ImgT> & aVSIT, std::string aExt)
{
	ifstream aFichier((aDir + aTempFile).c_str());
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

void cReechHomol_Appli::CorrHomolFromTemp(string & aDir, string & aSHIn, string & aTempFile, std::vector<ImgT> & aVSIT, string & aExt)
{
		cReechHomol_Appli::ReadImgTFile(aDir, aTempFile, aVSIT, aExt); //read all_name_temp.txt
			
		// get all converted Patis folders	
		string aDirHomol = aSHIn + "_txt/";
		std::list<cElFilename> aLPatis; 
		ctPath * aPathHomol = new ctPath(aDirHomol);
		aPathHomol->getContent(aLPatis); 
		
		// for one Patis folder
        for (std::list<cElFilename>::iterator iT1 = aLPatis.begin() ; iT1 != aLPatis.end() ; iT1++)
        {
			// master image
			string aImMaster = iT1->m_basename.substr (6,25);
			cout << "Master Image: " << aImMaster << endl;
			std::string aNameMapMaster = "Deg_" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
			cElMap2D * aMapMasterIm = cElMap2D::FromFile(aNameMapMaster);
			
			// match the temperature for master image		
			for (uint aV=0; aV < aVSIT.size(); aV++)
			{
				if (aVSIT.at(aV).ImgName.compare(aImMaster) == 0)
				{
					aNameMapMaster = "Deg_" + ToString(aVSIT.at(aV).ImgTemp) + ".xml";
					* aMapMasterIm->FromFile(aNameMapMaster);
				}
			}
			
			// get all .txt files of the master image			
			string aDirPatis = aDirHomol + iT1->m_basename;			
			cInterfChantierNameManipulateur * aICNMP = cInterfChantierNameManipulateur::BasicAlloc(aDirPatis);
			vector<string> aLFileP = *(aICNMP->Get(".*"));			
	
			// matche the temperature for secondary image and correct with maps
            for (uint aL=0; aL < aLFileP.size(); aL++)
				{	
					// secondary image
					string aImSecond = aLFileP.at(aL).substr(1,25);
					cout << "Secondary Image: " << aImSecond << endl;
					std::string aNameMapSecond = "Deg_" + ToString(aVSIT.at(0).ImgTemp) + ".xml";
					cElMap2D * aMapSecondIm = cElMap2D::FromFile(aNameMapSecond);
					
					for (uint aT=0; aT < aVSIT.size(); aT++)
					{
						// match the temperature for secondary image
						if (aVSIT.at(aT).ImgName.compare(aImSecond) == 0)
						{
							aNameMapSecond = "Deg_" + ToString(aVSIT.at(aT).ImgTemp) + ".xml";
							* aMapSecondIm->FromFile(aNameMapSecond);									
						}						
					}	
								
					    // read the .txt file and apply maps
                        std::vector<std::vector<double> >     aDataBrute;
                        std::vector<std::vector<double> >     aDataCorr;
						
						std::ifstream aFileBrute((aDirPatis+aLFileP.at(aL)).c_str());

						std::string aLine;
						while(std::getline(aFileBrute, aLine))
						{
							std::vector<double>   aLineData;
							std::stringstream  lineStream(aLine);

							double value;
							while(lineStream >> value)
							{
								aLineData.push_back(value);
							}
							aDataBrute.push_back(aLineData);
						}
						
                        // apply map to raw data
                        ElPackHomologue * bb =new ElPackHomologue();

						for (uint i = 0; i < aDataBrute.size(); i++)
						{	
							std::vector<double> aLineDataCorr;
                            Pt2dr aCorrMasterIm =(*aMapMasterIm)(Pt2dr(aDataBrute[i][0],aDataBrute[i][1]));
                            Pt2dr aCorrSecondIm =(*aMapSecondIm)(Pt2dr(aDataBrute[i][2],aDataBrute[i][3]));
                            ElCplePtsHomologues aa (aCorrMasterIm,aCorrSecondIm);
                            bb->Cple_Add(aa);
						}	

                        cout <<"test"<< endl;
                        std::string aDirHomolCorr = aSHIn + "_Reech/";
                        cout << aDirHomol << endl;
                        std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                                           +  std::string(aDirHomolCorr)
                                           +  std::string("@")
                                           +  std::string("dat");
                        cout << aKHIn << endl;

                        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
                        std::string aHmIn= aICNM->Assoc1To2(aKHIn, aImMaster, aImSecond, true);
                        cout << aHmIn << endl;
                        bb->StdPutInFile(aHmIn);
				}
        }
}

cReechHomol_Appli::cReechHomol_Appli(int argc,char ** argv)
{
	std::string aFullPat, aDir, aPatImgs, aSHIn, aTempFile, aExt = ".thm.tif";
	std::vector<ImgT> aVSIT;
    ElInitArgMain
    ( 
        argc,argv,
        LArgMain()  << EAMC(aFullPat, "Full Imgs Pattern", eSAM_IsExistFile)
					<< EAMC(aSHIn, "Input Homol folder", eSAM_IsExistFile)
					<< EAMC(aTempFile, "file containing image name & corresponding temperature", eSAM_IsExistFile),
        LArgMain()  << EAM(aExt,"Ext",true,"Extension of Imgs, Def = .thm.tif")
    );
    
    SplitDirAndFile(aDir,aPatImgs,aFullPat);    
    
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatImgs));
    
	cout << "File size = " << aSetIm.size() << endl;	
	
	//~ cReechHomol_Appli::ConvertHomol(aFullPat, aSHIn);
	cReechHomol_Appli::CorrHomolFromTemp(aDir, aSHIn, aTempFile, aVSIT, aExt);
}

int ReechHomol_main(int argc, char ** argv)
{

	cReechHomol_Appli anAppli(argc,argv);
    return EXIT_SUCCESS;
}
