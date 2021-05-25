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
#include "../uti_phgrm/Apero/cCameraRPC.h"

class cSatI_Appli
{
    public:
        cSatI_Appli(int argc,char ** argv);

        cInterfChantierNameManipulateur * mICNM;
        std::list<std::string> mListFile;

	eTypeImporGenBundle mType;
    const   cSystemeCoord * mChSys;

	Pt2di mGridSz;

	std::vector<std::string> mValidByGCP;
};

cSatI_Appli::cSatI_Appli(int argc,char ** argv) :
	mType(eTIGB_Unknown),
    mGridSz(Pt2di(10,8))
{
    std::string aFullName;
    std::string aDir;
    std::string aPat;
	std::string aCSysOut;

    std::string aNameType;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Orientation file (RPC/SPICE) full name (Dir+Pat)", eSAM_IsExistFile)
	            << EAM(aCSysOut,"proj","true", "Output cartographic coordinate system (proj format)"),
         LArgMain() << EAM(mGridSz,"GrSz",true, "No. of grids of bundles, e.g. GrSz=[10,10]", eSAM_NoInit)
		    << EAM(mValidByGCP, "VGCP", true, "Validate the prj fn with the provided GCPs [GrMes.xml,ImMes.xml]; optical centers not retrieved", eSAM_NoInit )
    );		      

    SplitDirAndFile(aDir, aPat, aFullName);

   // bool aModeHelp;
   // StdReadEnum(aModeHelp,mType,aNameType,eTIGB_NbVals);

    mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    mListFile = mICNM->StdGetListOfFile(aPat);

    mChSys = new cSystemeCoord(StdGetObjFromFile<cSystemeCoord>
            (
                aCSysOut,
                StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                "SystemeCoord",
                "SystemeCoord"
             ));
}


int Recal_RPC_main(int argc,char ** argv)
{
    std::string aFullName, aPat, aDir;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"File name of the RPC in XML_RPC format", eSAM_IsExistFile),
	   LArgMain() 
    );		      
    
    SplitDirAndFile(aDir, aPat, aFullName);

    cRPC::Save2XmlStdMMName(0 ,"",aPat,ElAffin2D::Id());

    
    return EXIT_SUCCESS;
}

int SATtoBundle_main(int argc,char ** argv)
{

     std::string aFullName;

     Pt2di aGridSz;

     ElInitArgMain
     (
        argc, argv,
        LArgMain() << EAMC(aFullName,"Orientation file in Xml_CamGenPolBundle"),
        LArgMain() << EAM(aGridSz,"GrSz",true)
      );

     CameraRPC aCam(aFullName);
     aCam.ExpImp2Bundle();

/*    cSatI_Appli aApps(argc,argv);

    for(std::list<std::string>::iterator itL = aApps.mListFile.begin(); 
		                         itL != aApps.mListFile.end(); 
					 itL++ )   
    {
	//Earth satellite
	if(aApps.mType!=eTIGB_Unknown && aApps.mType!=eTIGB_MMSten)
	{


            CameraRPC aCurCam(*itL,aApps.mType);
	    aCurCam.Exp2BundleInGeoc();
	}
	//other planets
	else
	{
    	    //from SPICE to cXml_ScanLineSensor
    	    //responsible: Antoine Lucas
	}
    }

*/
    return EXIT_SUCCESS;
}


int SatEmpriseSol_main(int argc,char ** argv)
{
	typedef std::vector<Pt2dr> tContour;

	std::string aPat, aDir, aName;
	std::string aOut="Footprints.ply";
	cPlyCloud aPlyFoot;

	cInterfChantierNameManipulateur *aICNM;
	std::list<std::string> aListFile;
	
    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aName,"Pattern of orientation files (in cXml_CamGenPolBundle format)"),
         LArgMain() << EAM(aOut, "Out", true, "Output file name, def=Footprints.ply" )
	);

    SplitDirAndFile(aDir, aPat, aName);

	
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aPat);


	std::list<std::string>::iterator itL = aListFile.begin();
    for( ; itL != aListFile.end(); itL++ )
	{
		int aR= rand() % 256;		
		int aG= rand() % 256;		
		int aB= rand() % 256;		

		CameraRPC aCam(aDir + (*itL));
		double aSolMoy = aCam.GetAltiSol();
		cElPolygone aPolyg = aCam.EmpriseSol();
		std::list<tContour> aVertList = aPolyg.Contours();
				
		
		std::list<tContour>::iterator itV = aVertList.begin();
		for( ; itV!=aVertList.end(); itV++ )
		{
			int aK=0;
			int aSz = int((*itV).size());
			Pt3dr aP1, aP2;
			for( ; aK<aSz-1; aK++ )
			{
				aP1.x = (*itV).at(aK).x; aP1.y = (*itV).at(aK).y; aP1.z = aSolMoy;
				aP2.x = (*itV).at(aK+1).x; aP2.y = (*itV).at(aK+1).y; aP2.z = aSolMoy;
				aPlyFoot.AddSeg(Pt3di(aR,aG,aB), aP1, aP2, 100);
			}
			//close the contour
			aP1.x = (*itV).at(0).x; aP1.y = (*itV).at(0).y; aP1.z = aSolMoy;
			aP2.x = (*itV).at(aSz-1).x; aP2.y = (*itV).at(aSz-1).y; aP2.z = aSolMoy;
			
			aPlyFoot.AddSeg(Pt3di(aR,aG,aB), aP1, aP2, 100);
		}
	}
	aPlyFoot.PutFile(aOut);

    return EXIT_SUCCESS;
} 

int SatBBox_main(int argc,char ** argv)
{
    std::string aGRIName;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aGRIName,"Grid"),
        LArgMain()
     );

    OrientationGrille aOri(aGRIName);

    Pt2dr aDZ = aOri.GetRangeZ();
    double Zmin =max(min(aDZ.x, aDZ.y),0.);

    Pt2dr aDC = aOri.GetRangeCol(); //problems if the grid is much wider than the image...
    Pt2dr aDR = aOri.GetRangeRow();


    Pt2dr aP00(aDC.x,aDR.x);
    Pt2dr aP01(aDC.x,abs(aDR.y));
    Pt2dr aP10(aDC.y,aDR.x);
    Pt2dr aP11(aDC.y,abs(aDR.y));

    Pt2dr aP00Gr,aP01Gr,aP10Gr,aP11Gr;
    aOri.ImageAndPx2Obj(aP00.x,aP00.y,&Zmin,aP00Gr.x,aP00Gr.y);
    aOri.ImageAndPx2Obj(aP01.x,aP01.y,&Zmin,aP01Gr.x,aP01Gr.y);
    aOri.ImageAndPx2Obj(aP10.x,aP10.y,&Zmin,aP10Gr.x,aP10Gr.y);
    aOri.ImageAndPx2Obj(aP11.x,aP11.y,&Zmin,aP11Gr.x,aP11Gr.y);

    double Xmin=min(min(aP00Gr.x, aP01Gr.x), min(aP10Gr.x, aP11Gr.x));
    double Xmax=max(max(aP00Gr.x, aP01Gr.x), max(aP10Gr.x, aP11Gr.x));
    double Ymin=min(min(aP00Gr.y, aP01Gr.y), min(aP10Gr.y, aP11Gr.y));
    double Ymax=max(max(aP00Gr.y, aP01Gr.y), max(aP10Gr.y, aP11Gr.y));

    std::cout << Xmin << " " << Ymin << " " << Xmax << " " << Ymax << "\n";

    return EXIT_SUCCESS;
}


int SATtoOpticalCenter_main(cSatI_Appli &aApps)
{

		for(std::list<std::string>::iterator itL = aApps.mListFile.begin(); 
						itL != aApps.mListFile.end(); 
						itL++ )
		{

				AutoDetermineTypeTIGB(aApps.mType,(const std::string)(*itL));


				//Earth satellite
				if(aApps.mType!=eTIGB_Unknown && aApps.mType!=eTIGB_MMSten)
				{
						//CameraRPC aCamRPC(*itL,aApps.mType,aApps.mCSysOut,aApps.mGridSz,aApps.mMetadata);


						CameraRPC aCamRPC(*itL, aApps.mType,aApps.mChSys);
						aCamRPC.SetGridSz(aApps.mGridSz);
						aCamRPC.OpticalCenterGrid(true);


						/*Pt2dr aa(100.4,1000.0);
						  Pt3dr aaa1 = aCamRPC.OpticalCenterOfPixel(aa);
						  std::cout << "1 " << aaa1 << "\n";

						  aCamRPC.OpticalCenterOfImg();
						  Pt3dr aaa2 = aCamRPC.OpticalCenterOfPixel(aa);
						  std::cout << "2 " << aaa2 << "\n";

						  std::cout << "1-2 " << aaa1 - aaa2 << "\n";
						 */

				}
				//other planets
				else
				{
						std::string aDirTmp = "PBoptCenter";
						std::string aSaveFile = aDirTmp + "/PlanetOpticalCentersTer" + ".txt";

						std::vector<Pt3dr> aOpticalCenters;

						//read the cXml_ScanLineSensor
						cXml_ScanLineSensor aSLS = StdGetFromSI(*itL,Xml_ScanLineSensor);

						std::vector< cXml_OneLineSLS >::iterator aLIt;
						std::vector< cXml_SLSRay >::iterator     aSIt;
						for(aLIt=aSLS.Lines().begin(); aLIt<aSLS.Lines().end(); aLIt++)
						{
								std::vector<ElSeg3D> aVS;
								std::vector<double>  aVPds;

								for(aSIt=aLIt->Rays().begin(); aSIt<aLIt->Rays().end(); aSIt++)
								{
										aVPds.push_back(0.5);
										aVS.push_back( ElSeg3D(aSIt->P1(), aSIt->P2()) );

								}

								//intersect
								bool aIsOK;
								aOpticalCenters.push_back(ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK));

						}

						//save output
						std::ofstream aFO(aSaveFile.c_str());
						aFO << std::setprecision(15);

						unsigned int i;
						for(i=0; i<aOpticalCenters.size(); i++)
								aFO << aOpticalCenters.at(i).x
										<< " " << aOpticalCenters.at(i).y
										<< " " << aOpticalCenters.at(i).z
										<< "\n";

						aFO.close();
				}
		}

		return EXIT_SUCCESS;
}

int SATTrajectory_main(int argc,char ** argv)
{
    cSatI_Appli aApps(argc,argv);
    SATtoOpticalCenter_main(aApps);
	
	return EXIT_SUCCESS;
}

void SATbackrpjGCP_main(cSatI_Appli &aApps)
{
		//input and backprojected image observations
		cSetOfMesureAppuisFlottants aDicoImCmp;

		//read GCPs xy
		cDicoAppuisFlottant aDicoGr = StdGetFromPCP(aApps.mValidByGCP.at(0),
						DicoAppuisFlottant );

		//read GCPs XYZ
		cSetOfMesureAppuisFlottants aDicoIm = StdGetFromPCP(aApps.mValidByGCP.at(1),
						SetOfMesureAppuisFlottants);

		//std::cout << aDicoIm.OneAppuisDAF().begin()->NamePt() << " " << 
		//	         aDicoIm.OneAppuisDAF().begin()->Pt() << "\n";
		//std::cout << aSetDicoMesure.MesureAppuiFlottant1Im().begin()->NameIm() << "\n";

		std::list<cMesureAppuiFlottant1Im> aMAFL;

		for(std::list<std::string>::iterator itL = aApps.mListFile.begin(); 
						itL != aApps.mListFile.end(); 
						itL++ )
		{
				AutoDetermineTypeTIGB(aApps.mType,*itL);


				if(aApps.mType!=eTIGB_Unknown && aApps.mType!=eTIGB_MMSten)
				{
						// CameraRPC aCamRPC(*itL,aApps.mType,aApps.mCSysOut,aApps.mGridSz,aApps.mMetadata);
						CameraRPC aCamRPC(*itL, aApps.mType,aApps.mChSys);

						cMesureAppuiFlottant1Im aImCur;
						aImCur.NameIm() = *itL;
						std::cout << "aImCur.NameIm() " << *itL  << "\n"; 

						//for all images
						std::list<cMesureAppuiFlottant1Im>::const_iterator aImIt;
						for( aImIt=aDicoIm.MesureAppuiFlottant1Im().begin();
										aImIt!=aDicoIm.MesureAppuiFlottant1Im().end(); aImIt++)
						{
								std::cout << "aImIt->NameIm() " << aImIt->NameIm()  << "\n"; 
								if( *itL == aImIt->NameIm() )
								{
										cMesureAppuiFlottant1Im aMAFcur;
										aMAFcur.NameIm() = aImIt->NameIm();
										std::list<cOneMesureAF1I> aOMcur;

										//for all points
										std::list<cOneMesureAF1I>::const_iterator aImPtIt;
										for(aImPtIt=aImIt->OneMesureAF1I().begin(); 
														aImPtIt!=aImIt->OneMesureAF1I().end(); aImPtIt++)
										{
												std::cout << "aImPtIt->NamePt() " << aImPtIt->NamePt()  << "\n"; 
												//find ground coordintes XYZ
												std::list<cOneAppuisDAF>::const_iterator aGrPtIt;
												for(aGrPtIt=aDicoGr.OneAppuisDAF().begin();
																aGrPtIt!=aDicoGr.OneAppuisDAF().end(); aGrPtIt++)
												{
														if(aImPtIt->NamePt() == aGrPtIt->NamePt())
														{

																//tescik
																std::cout << "isVisible " << aCamRPC.PIsVisibleInImage(aGrPtIt->Pt()) << "\n";
																std::cout << "resol " << aCamRPC.ResolSolOfPt(aGrPtIt->Pt()) << "\n";
																//backproject
																cOneMesureAF1I aPtCurCmp;
																aPtCurCmp.NamePt() = aGrPtIt->NamePt() + "_bprj";
																aPtCurCmp.PtIm() = aCamRPC.Ter2Capteur(aGrPtIt->Pt());


																//push the original img observation
																aOMcur.push_back(*aImPtIt);

																//push the backprojected obs
																aOMcur.push_back(aPtCurCmp);
																std::cout << aGrPtIt->NamePt() << aPtCurCmp.PtIm() << aPtCurCmp.PtIm()  << "\n";        
														}
												}
										}
										aMAFcur.OneMesureAF1I() = aOMcur;
										aMAFL.push_back(aMAFcur);

										std::cout << "size " << aMAFL.size() << "\n"; 
		    std::cout << aImIt->NameIm(); 
	        }
	    }
            aDicoImCmp.MesureAppuiFlottant1Im() = aMAFL;


        }
    }
    //export to XML format
    std::string aNameTmp = aApps.mValidByGCP.at(0).substr(0,aApps.mValidByGCP.at(0).size()-4);
    
    MakeFileXML(aDicoImCmp,  
		            aNameTmp + "_bprj.xml");
}

/* Validate the projection function
 * - using the provided GCPs, or
 * - by intersecting picture elements at the optical centers*/
int SATvalid_main(int argc,char ** argv)
{

    cSatI_Appli aApps(argc,argv);

    //validation by GCPs
    if (EAMIsInit(&aApps.mValidByGCP))
    {
        SATbackrpjGCP_main(aApps);
    }
    //validation by intersection
    else
        SATtoOpticalCenter_main(aApps);
    
    return EXIT_SUCCESS;

}

int CPP_TestSystematicResiduals(int argc,char ** argv)
{
    std::string aIm1, aIm2, aDir;
    std::string aNameH, aIm1Name, aIm2Name;

    Pt2dr  aZVal;
    Pt3dr  aPt3D;
    std::vector<double> aVPds;
    bool aExpTxt=0;
    bool aExpPly=0;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aIm1,"First image orientation file (in cXml_CamGenPolBundle format)")
                    << EAMC(aIm2,"Second image orientation file (in cXml_CamGenPolBundle format)"),
         LArgMain() << EAM(aExpTxt, "ExpTxt", true, "Homol in txt, def=false" )
                    << EAM(aExpPly, "ExpPly", true, "Export the intersected tie points in ply, def=false" )
    );
    
    aDir = DirOfFile(aIm1);
    
    std::cout << aIm1 << " " << aIm2 << "\n";

    /* Read cameras */
    CameraRPC aCamRPC1(aIm1);
    CameraRPC aCamRPC2(aIm2);
    
    aZVal = aCamRPC1.GetAltiSolMinMax();
    std::cout << "ZMinMax:" << aZVal << "\n";

    cXml_CamGenPolBundle aXml1 = StdGetFromSI(aIm1,Xml_CamGenPolBundle);
    cXml_CamGenPolBundle aXml2 = StdGetFromSI(aIm2,Xml_CamGenPolBundle);
    
    aIm1Name = aXml1.NameIma();
    aIm2Name = aXml2.NameIma();
    
    /* Read Homol for both images*/
    aNameH = "Homol/Pastis" + aIm1Name + "/" + aIm2Name + (aExpTxt ? ".txt" : ".dat"); 
    ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);

    

    
    /* Intersection + backprojection */
    bool aIsOK;
    Pt3dr aIm1P1, aIm1P2, aIm2P1, aIm2P2;
    Pt2dr aP1B, aP2B;
    Pt2dr aSomD1(0,0), aSomD2(0,0);
    vector<Pt2dr> aRV1, aRV2;
    

    int aNum=0;
    cPlyCloud aPly;
    ElPackHomologue::iterator itPt=aPack.begin();
    for( ; itPt != aPack.end(); itPt++, aNum++ )
    {
        std::vector<ElSeg3D> aVS;
   

        aIm1P1 = aCamRPC1.ImEtZ2Terrain(itPt->P1(),aZVal.x); 
        aIm1P2 = aCamRPC1.ImEtZ2Terrain(itPt->P1(),aZVal.y); 
        
        aIm2P1 = aCamRPC2.ImEtZ2Terrain(itPt->P2(),aZVal.x); 
        aIm2P2 = aCamRPC2.ImEtZ2Terrain(itPt->P2(),aZVal.y);
        
        aVPds.push_back(1);
        aVS.push_back(ElSeg3D(aIm1P1,aIm1P2));
        aVS.push_back(ElSeg3D(aIm2P1,aIm2P2));

        aPt3D = ElSeg3D::L2InterFaisceaux(&aVPds, aVS, &aIsOK);
       
        if( aExpPly==1 )
            aPly.AddPt(Pt3di(255,255,255),aPt3D);
        

        //backproject
        aP1B = aCamRPC1.Ter2Capteur(aPt3D);
        aP2B = aCamRPC2.Ter2Capteur(aPt3D);
        
        aSomD1.x = (aSomD1.x + itPt->P1().x-aP1B.x);
        aSomD1.y = (aSomD1.y + itPt->P1().y-aP1B.y);
        
        aSomD2.x = (aSomD2.x + itPt->P2().x-aP2B.x);
        aSomD2.y = (aSomD2.y + itPt->P2().y-aP2B.y);
        
        //std::cout  << ",dif: " << itPt->P1().x-aP1B.x << "; " << aSomD << "\n";

        aRV1.push_back( Pt2dr(itPt->P1().x-aP1B.x, itPt->P1().y-aP1B.y) );
        aRV2.push_back( Pt2dr(itPt->P2().x-aP2B.x, itPt->P2().y-aP2B.y) );
    }
    
    Pt2dr aMoy1 = (aSomD1)/aNum;
    Pt2dr aMoy2 = (aSomD2)/aNum;
    Pt2dr aStd1(0,0), aStd2(0,0);

    
    for(int aK=0; aK<int(aRV1.size()); aK++)
    {
        aStd1.x = aStd1.x + (aRV1.at(aK).x - aMoy1.x)*(aRV1.at(aK).x - aMoy1.x);
        aStd1.y = aStd1.y + (aRV1.at(aK).y - aMoy1.y)*(aRV1.at(aK).y - aMoy1.y);
        
        aStd2.x = aStd2.x + (aRV2.at(aK).x - aMoy2.x)*(aRV2.at(aK).x - aMoy2.x);
        aStd2.y = aStd2.y + (aRV2.at(aK).y - aMoy2.y)*(aRV2.at(aK).y - aMoy2.y);
    }
    aStd1.x = sqrt(double(aStd1.x)/aNum);
    aStd1.y = sqrt(double(aStd1.y)/aNum);
    aStd2.x = sqrt(double(aStd2.x)/aNum);
    aStd2.y = sqrt(double(aStd2.y)/aNum);

    std::cout << "aSom1=" << aSomD1 << ", aSom2=" << aSomD2 << ", Num=" << aNum << "\n";
    std::cout << "mean1=" << aMoy1  << ", std_dev1=" << aStd1 
              << "mean2=" << aMoy2  << ", std_dev2=" << aStd2 << "\n";


    if( aExpPly==1 )
        aPly.PutFile(aDir + "TiePts.ply");

    return EXIT_SUCCESS;

}


int CPP_TestRPCBackProj(int argc,char ** argv)
{

    cInterfChantierNameManipulateur * aICNM;
    std::list<std::string> aListFile;

    std::string aFullName;
    std::string aDir;
    std::string aPat;

    Pt3dr aPIn;
    Pt2dr aPOut;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Orientation file (or pattern) in cXml_CamGenPolBundle format")
                    << EAMC(aPIn,"3D point in target CS"),
         LArgMain() 
    );		      

    SplitDirAndFile(aDir, aPat, aFullName);

    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aPat);
    
    std::cout << "aPIn " << aPIn << "\n";
    for(std::list<std::string>::iterator itL = aListFile.begin(); 
		                                 itL != aListFile.end();
                                         itL++)
    {
        CameraRPC aCamRPC(aDir + (*itL));

        aPOut = aCamRPC.Ter2Capteur(aPIn);

        std::cout << *itL << " " << aPOut << "\n";
        
    }


    return EXIT_SUCCESS;
}

int CPP_TestRPCDirectGen(int argc,char ** argv)
{

    cInterfChantierNameManipulateur * aICNM;
    std::list<std::string> aListFile;

    std::string aFullName;
    std::string aDir;
    std::string aPat;
    Pt3di aSz(100,100,10);

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Orientation file (or pattern) in cXml_CamGenPolBundle format"),
         LArgMain() << EAM(aSz,"Sz",true,"Size of the verification grid (Def = [100,100,10])")
    );		      

    SplitDirAndFile(aDir, aPat, aFullName);

    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aPat);
    
    for(std::list<std::string>::iterator itL = aListFile.begin(); 
		                                 itL != aListFile.end(); 
					                     itL++ )   
    {
        
        CameraRPC aCRPC(aDir + (*itL));
        cRPCVerf aVf(aCRPC, aSz);
	    
        aVf.Do();
    }

    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

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
Footer-MicMac-eLiSe-25/06/2007*/
