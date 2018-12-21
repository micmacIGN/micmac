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

#include "schnaps.h"

class cCmpTieP_Appli
{
	public :
		cCmpTieP_Appli(int argc,char ** argv);
		std::vector<CamStenope *> readOri(std::string Ori, const std::vector<std::string> aSetOfIm, cInterfChantierNameManipulateur * ICNM);
		Pt3dr IntersectionFaisceaux(const std::vector<CamStenope *> & aVCS,const std::vector<Pt2dr> & aNPts2D);
		std::vector<Pt3dr> CalcDev(std::vector<Pt3dr> aVPt3d1, std::vector<Pt3dr> aVPt3d2);
		double CalcMean(std::vector<double> & aV);
		double CalcMin(std::vector<double> & aV);
		double CalcMax(std::vector<double> & aV);
		double CalcStd(std::vector<double> & aV);
	private :
		std::string mFullPattern;
		std::string mOri1;
		std::string mOri2;
};

std::vector<Pt3dr> cCmpTieP_Appli::CalcDev(std::vector<Pt3dr> aVPt3d1, std::vector<Pt3dr> aVPt3d2)
{	
	ELISE_ASSERT(aVPt3d1.size() == aVPt3d2.size(),"ERROR: Not Same Homol Size!");
	
	std::vector<Pt3dr> aResDev;
	std::vector<double> aResX;
	std::vector<double> aResY;
	std::vector<double> aResZ;
	
	for(unsigned int aK=0; aK<aVPt3d1.size(); aK++)
	{
		
		aResX.push_back(aVPt3d1.at(aK).x - aVPt3d2.at(aK).x);
		aResY.push_back(aVPt3d1.at(aK).y - aVPt3d2.at(aK).y);
		aResZ.push_back(aVPt3d1.at(aK).z - aVPt3d2.at(aK).z);

		Pt3dr aPt;
		aPt.x = aResX.at(aK);
		aPt.y = aResY.at(aK);
		aPt.z = aResZ.at(aK);

		aResDev.push_back(aPt);

	}
	
	//some statistics
	//calc mean
	std::cout << "***** Mean ***********" << std::endl;
	std::cout << "Mean X = " << CalcMean(aResX) << std::endl;
	std::cout << "Mean Y = " << CalcMean(aResY) << std::endl;
	std::cout << "Mean Z = " << CalcMean(aResZ) << std::endl;
	std::cout << "**********************" << std::endl;
	
	//calc min
	std::cout <<  "***** Min ***********" << std::endl;
	std::cout << "Min X = " << CalcMin(aResX) << std::endl;
	std::cout << "Min Y = " << CalcMin(aResY) << std::endl;
	std::cout << "Min Z = " << CalcMin(aResZ) << std::endl;
	std::cout <<  "*********************" << std::endl;
	
	//calc max
	std::cout <<  "***** Max ***********" << std::endl;
	std::cout << "Max X = " << CalcMax(aResX) << std::endl;
	std::cout << "Max Y = " << CalcMax(aResY) << std::endl;
	std::cout << "Max Z = " << CalcMax(aResZ) << std::endl;
	std::cout <<  "*********************" << std::endl;
	
	//calc std
	std::cout <<  "***** Std ***********" << std::endl;
	std::cout << "Std X = " << CalcStd(aResX) << std::endl;
	std::cout << "Std Y = " << CalcStd(aResY) << std::endl;
	std::cout << "Std Z = " << CalcStd(aResZ) << std::endl;
	std::cout <<  "*********************" << std::endl;
	
	return aResDev;
}

double cCmpTieP_Appli::CalcStd(std::vector<double> & aV)
{
	double aVar=0;
	double aStd=0;
	
	int aN=0;
	int aSize=aV.size();
	
	while(aN < aSize)
	{
		aVar = aVar + ((aV[aN] - CalcMean(aV)) * (aV[aN] - CalcMean(aV)));
		aN++; 
	}
	
	aVar /= aSize;
	aStd = sqrt(aVar);
	
	return aStd;
}

double cCmpTieP_Appli::CalcMax(std::vector<double> & aV)
{
	double aMax = aV[0];
	
	for(unsigned aK=0; aK<aV.size(); aK++)
	{
		if(aV[aK] > aMax)
		{
			aMax = aV[aK];
		}	
	}
	return aMax;
}

double cCmpTieP_Appli::CalcMin(std::vector<double> & aV)
{
	double aMin = aV[0];
	
	for(unsigned aK=0; aK<aV.size(); aK++)
	{
		if(aV[aK] < aMin)
		{
			aMin = aV[aK];
		}	
	}
	return aMin;
}

double cCmpTieP_Appli::CalcMean(std::vector<double>& aV)
{
	double aSum = 0.0;
	
	for(unsigned int aK=0; aK< aV.size(); aK++)
		aSum += aV[aK];
	
	return aSum / static_cast<double>(aV.size());
}

std::vector<CamStenope *> cCmpTieP_Appli::readOri(std::string Ori, const std::vector<std::string> aSetOfIm, cInterfChantierNameManipulateur * ICNM)
{
	std::vector<std::string> aVOriFiles(aSetOfIm.size());
    std::vector<CamStenope *> aVCam(aSetOfIm.size());
    
    for (unsigned int aK=0; aK<aSetOfIm.size(); aK++)
    {
		aVOriFiles.at(aK) = Ori+"Orientation-"+aSetOfIm.at(aK)+".xml";
		aVCam.at(aK) = CamOrientGenFromFile(aVOriFiles.at(aK),ICNM);
	}
	
	return aVCam;
}

Pt3dr cCmpTieP_Appli::IntersectionFaisceaux(const std::vector<CamStenope *> & aVCS,const std::vector<Pt2dr> & aNPts2D)
{
	std::vector<ElSeg3D> aVSeg;
	
	for (int aKR=0 ; aKR < int(aVCS.size()) ; aKR++)
	{
		ElSeg3D aSeg = aVCS.at(aKR)->F2toRayonR3(aNPts2D.at(aKR));
		aVSeg.push_back(aSeg);
	}
	
	Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    return aRes;
}


cCmpTieP_Appli::cCmpTieP_Appli(int argc,char ** argv)
{
	
	std::string aDirImages, aPatIm, aOut;
	std::string aInHomolDirName="";
	bool ExpTxt=false;
	int aMultMin=2;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mFullPattern, "Pattern of images",  eSAM_IsPatFile)
					 << EAMC(mOri1, "First Ori folder", eSAM_IsExistDirOri)
					 << EAMC(mOri2, "Second Ori folder", eSAM_IsExistDirOri),
          LArgMain() << EAM(aOut,"Out",false,"output txt file name ; Def=DeviationsOnTieP.txt")
                     << EAM(aInHomolDirName, "HomolIn", true, "Input Homol directory suffix (without \"Homol\")")
                     << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                     << EAM(aMultMin,"MultMin",false,"Minimum Value for Tie Points Multiplicity ; Def=2")
    );
    
    ELISE_ASSERT(aMultMin > 1,"ERROR: Bad Tie Points Multiplicity Value!");
    
    //name output (.txt) file
    if (aOut=="")
    {
		aOut = "DeviationsOnTieP.txt";
    }
    
    SplitDirAndFile(aDirImages,aPatIm,mFullPattern);
    
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;
    
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));
    
    //read Homol Directory
    //Init Keys for homol files
    std::list<cHomol> allHomols;
    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKH =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aInHomolDirName)
            +  std::string("@")
            +  std::string(anExt);
            
    CompiledKey2 aCK(aICNM,aKH);
    
    //create pictures list, and pictures size list
    std::map<std::string,cPic*> allPics;

    std::cout<<"Found "<<aSetIm.size()<<" pictures."<<endl;

    computeAllHomol(aDirImages,aPatIm,aSetIm,allHomols,aCK,allPics,false,0);
    
    std::vector<Pt3dr> aVPt3DO1;
    std::vector<Pt3dr> aVPt3DO2;
    int aNumBadHomol=0;
    
    for (std::list<cHomol>::iterator itHomol=allHomols.begin();itHomol!=allHomols.end();++itHomol)
    {
		std::vector<CamStenope *> vCSO1;
		std::vector<CamStenope *> vCSO2;
		
		std::vector<Pt2dr> vPt2d;
		
        int aNbr = itHomol->getPointOnPicsSize();
                
        if(aNbr >= aMultMin)
        {
			for(int aK=0; aK<aNbr; aK++)
			{
				cPointOnPic* aPointOnPic = itHomol->getPointOnPic(aK);
				cPic* aPic = aPointOnPic->getPic();
				std::string aName = aPic->getName();
				Pt2dr& aPt = aPointOnPic->getPt();
				
				CamStenope * aCam1 = CamOrientGenFromFile(mOri1+"Orientation-"+aName+".xml",aICNM);
				vCSO1.push_back(aCam1);
				
				CamStenope * aCam2 = CamOrientGenFromFile(mOri2+"Orientation-"+aName+".xml",aICNM);
				vCSO2.push_back(aCam2);
				
				vPt2d.push_back(aPt);
			}
		
		Pt3dr aPt3DO1 = IntersectionFaisceaux(vCSO1,vPt2d);
		aVPt3DO1.push_back(aPt3DO1);
		
		Pt3dr aPt3DO2 = IntersectionFaisceaux(vCSO2,vPt2d);
		aVPt3DO2.push_back(aPt3DO2);
		
		}
        
        if ((itHomol)->isBad()) aNumBadHomol++;
    }

    std::cout<<"Found "<<allHomols.size()<<" Homol points (incl. "<<aNumBadHomol<<" bad ones): "<<100*aNumBadHomol/allHomols.size()<<"% bad!\n";
    
    //compute deviations
    std::vector<Pt3dr> aVDev = CalcDev(aVPt3DO1,aVPt3DO2);    
    
    //export residuals in .txt format
    if (!MMVisualMode)
	{
			
		FILE * aFP = FopenNN(aOut,"w","CompareOriTieP_main");
				
		cElemAppliSetFile aEASF(aDirImages + ELISE_CAR_DIR + aOut);
				
		for (unsigned int aK=0 ; aK<aVDev.size() ; aK++)
		{
			fprintf(aFP,"%lf %lf %lf \n",aVDev.at(aK).x,aVDev.at(aK).y,aVDev.at(aK).z);
		}
			
	ElFclose(aFP);
		
	}

    //cleaning
    std::map<std::string,cPic*>::iterator itPic1;
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
        delete itPic1->second;
    allPics.clear();
}

int CompareOriTieP_main(int argc,char ** argv)
{   
	cCmpTieP_Appli anAppli(argc,argv);
	return EXIT_SUCCESS;
}

int StatsOnFile_main(int argc,char ** argv)
{
	
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
