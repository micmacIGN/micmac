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
	private :
		std::string mFullPattern;
		std::string mOri1;
		std::string mOri2;
};

std::vector<Pt3dr> cCmpTieP_Appli::CalcDev(std::vector<Pt3dr> aVPt3d1, std::vector<Pt3dr> aVPt3d2)
{	
	ELISE_ASSERT(aVPt3d1.size() == aVPt3d2.size(),"ERROR: Not Same Homol Size!");
	
	std::vector<Pt3dr> aResDev;
	
	for(unsigned int aK=0; aK<aVPt3d1.size(); aK++)
	{
		Pt3dr aPt;
		aPt.x = aVPt3d1.at(aK).x - aVPt3d2.at(aK).x;
		aPt.y = aVPt3d1.at(aK).y - aVPt3d2.at(aK).y;
		aPt.z = aVPt3d1.at(aK).z - aVPt3d2.at(aK).z;
		
		aResDev.push_back(aPt);
	}
	
	return aResDev;
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
	//vecteur d'éléments segments 3d
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
	bool veryStrict=false;
	
	ElInitArgMain
    (
          argc, argv,
          LArgMain() << EAMC(mFullPattern, "Pattern of images",  eSAM_IsPatFile)
					 << EAMC(mOri1, "First Ori folder", eSAM_IsExistDirOri)
					 << EAMC(mOri2, "Second Ori folder", eSAM_IsExistDirOri),
          LArgMain() << EAM(aOut,"Out",false,"output txt file name ; Def=DeviationsOnTieP.txt")
                     << EAM(aInHomolDirName, "HomolIn", true, "Input Homol directory suffix (without \"Homol\")")
                     << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                     //~ << EAM(veryStrict,"VeryStrict",true,"Be very strict with homols (remove any suspect), def=false")
    );
    
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
    // Init Keys for homol files
    std::list<cHomol> allHomols;
    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKH =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aInHomolDirName)
            +  std::string("@")
            +  std::string(anExt);
            
    CompiledKey2 aCK(aICNM,aKH);
    
    //create pictures list, and pictures size list
    std::map<std::string,cPic*> allPics;
    
    std::vector<cPicSize> allPicSizes;

    std::cout<<"Found "<<aSetIm.size()<<" pictures."<<endl;


    computeAllHomol(aICNM,aDirImages,aPatIm,aSetIm,allHomols,aCK,allPics,allPicSizes,veryStrict,0);
    
    if (veryStrict)
    {
        //check if homol apear everywhere they should
        cout<<"Checking Homol integrity..";
        //for every homol
        int nbInconsistantHomol=0;
        for (std::list<cHomol>::iterator itHomol=allHomols.begin();itHomol!=allHomols.end();++itHomol)
        {
            cHomol &aHomol=(*itHomol);
            cPic *aPic1=0;
            cPic *aPic2=0;
            #ifdef ReductHomolImage_VeryStrict_DEBUG
            cout<<"For ";
            aHomol.print();
            #endif
            //for every combination of PointOnPic
            for (unsigned int i=0;i<aHomol.getPointOnPicsSize();i++)
            {
                aPic1=aHomol.getPointOnPic(i)->getPic();
                for (unsigned int j=i+1;j<aHomol.getPointOnPicsSize();j++)
                {
                    //if the pack exist
                    aPic2=aHomol.getPointOnPic(j)->getPic();
                    //std::string aNameIn = aDirImages + aICNM->Assoc1To2(aKHIn,aPic1->getName(),aPic2->getName(),true);
                    std::string aNameIn=aCK.get(aPic1->getName(),aPic2->getName());
                    if (ELISE_fp::exist_file(aNameIn))
                    {
                        #ifdef ReductHomolImage_VeryStrict_DEBUG
                        cout<<"   "<<aNameIn<<": ";
                        #endif
                        //check that homol has been seen in this couple of pictures
                        if (!aHomol.appearsOnCouple2way(aPic1,aPic2))
                        {
                            #ifdef ReductHomolImage_VeryStrict_DEBUG
                            cout<<"No!\n";
                            #endif
                            aHomol.setBad();
                            i=aHomol.getPointOnPicsSize();//end second loop
                            nbInconsistantHomol++;
                            break;
                        }
                        #ifdef ReductHomolImage_VeryStrict_DEBUG
                        else cout<<"OK!\n";
                        #endif

                    }
                }
                if (aHomol.isBad()) break;
            }
            if ((aHomol.getId()%1000)==0) cout<<"."<<flush;
        }
        std::cout<<"Done.\n"<<nbInconsistantHomol<<" inconsistant homols found."<<endl;
    }
    
    
    std::vector<Pt3dr> aVPt3DO1;
    std::vector<Pt3dr> aVPt3DO2;
    int aNumBadHomol=0;
    
    for (std::list<cHomol>::iterator itHomol=allHomols.begin();itHomol!=allHomols.end();++itHomol)
    {
		std::vector<CamStenope *> vCSO1;
		std::vector<CamStenope *> vCSO2;
		
		std::vector<Pt2dr> vPt2d;
		
        unsigned int aNbr = itHomol->getPointOnPicsSize();
        
        //~ std::cout << "Homol ID = " << itHomol->getId() << " | PointOnPicsSize() = " << aNbr << std::endl;
        
        if(aNbr > 1)
        {
			for(unsigned int aK=0; aK<aNbr; aK++)
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
		else
		{
			//std::cout << "For Homol id = " << itHomol->getId() << " Not enough images! Bad Homol!" << std::endl; 
		}
        
        if ((itHomol)->isBad()) aNumBadHomol++;
    }

    std::cout<<"Found "<<allHomols.size()<<" Homol points (incl. "<<aNumBadHomol<<" bad ones): "<<100*aNumBadHomol/allHomols.size()<<"% bad!\n";
    
    
    //compute deviations
    std::vector<Pt3dr> aVDev = CalcDev(aVPt3DO1,aVPt3DO2);
    
    //compute statistics
    
    
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
}

int CompareOriTieP_main(int argc,char ** argv)
{   
	cCmpTieP_Appli anAppli(argc,argv);
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
