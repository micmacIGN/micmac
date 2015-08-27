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
#include "../uti_phgrm/MICMAC/CameraRPC.h"

class cSatI_Appli
{
    public:
        cSatI_Appli(int argc,char ** argv);

        cInterfChantierNameManipulateur * mICNM;
	std::string mCSysOut;
        std::list<std::string> mListFile;

	eTypeImporGenBundle mType;
	std::string mMetadata;
	Pt2di mGridSz;

	std::vector<std::string> mValidByGCP;
};

cSatI_Appli::cSatI_Appli(int argc,char ** argv) :
	mMetadata(""),
	mGridSz(Pt2di(10,8))
{
    std::string aFullName;
    std::string aDir;
    std::string aPat;

    std::string aNameType;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Orientation file (RPC/SPICE) full name (Dir+Pat)", eSAM_IsExistFile)
                    << EAMC(aNameType,"Type of sensor (see eTypeImporGenBundle)",eSAM_None,ListOfVal(eTT_NbVals,"eTT_")),
	 LArgMain() << EAM(mCSysOut,"proj","true", "Output cartographic coordinate system (proj format)")
                    << EAM(mGridSz,"GrSz",true, "No. of grids of bundles, e.g. GrSz=[10,10]", eSAM_NoInit)
		    << EAM(mMetadata, "Meta", true, "Sensor metadata file, other than the RPC; Valid for IKONOS and CARTOSAT", eSAM_IsExistFile)
		    << EAM(mValidByGCP, "VGCP", true, "Validate the prj fn with the provided GCPs [GrMes.xml,ImMes.xml]; optical centers not retrieved", eSAM_NoInit )
    );		      

    SplitDirAndFile(aDir, aPat, aFullName);

    bool aModeHelp;
    StdReadEnum(aModeHelp,mType,aNameType,eTIGB_NbVals);

    mICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    mListFile = mICNM->StdGetListOfFile(aPat);
}


int SATtoBundle_main(int argc,char ** argv)
{

    cSatI_Appli aApps(argc,argv);

    for(std::list<std::string>::iterator itL = aApps.mListFile.begin(); 
		                         itL != aApps.mListFile.end(); 
					 itL++ )   
    {
	//Earth satellite
	if(aApps.mType!=eTIGB_Unknown && aApps.mType!=eTIGB_MMSten)
	{
            CameraRPC aCurCam(*itL,aApps.mType,aApps.mCSysOut,aApps.mGridSz);
	    aCurCam.ExpImp2Bundle();
	}
	//other planets
	else
	{
    	    //from SPICE to cXml_ScanLineSensor
    	    //responsible: Antoine Lucas
	}
    }


    return EXIT_SUCCESS;
}
 
int SATtoOpticalCenter_main(cSatI_Appli &aApps)
{

    for(std::list<std::string>::iterator itL = aApps.mListFile.begin(); 
		                         itL != aApps.mListFile.end(); 
					 itL++ )
    {
       //Earth satellite
       if(aApps.mType!=eTIGB_Unknown && aApps.mType!=eTIGB_MMSten)
       {
           CameraRPC aCamRPC(*itL,aApps.mType,aApps.mCSysOut,aApps.mGridSz,aApps.mMetadata);
           aCamRPC.OpticalCenterLineGrid(true);

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
	if(aApps.mType!=eTIGB_Unknown && aApps.mType!=eTIGB_MMSten)
	{
            CameraRPC aCamRPC(*itL,aApps.mType,aApps.mCSysOut,aApps.mGridSz,aApps.mMetadata);

            cMesureAppuiFlottant1Im aImCur;
	    aImCur.NameIm() = *itL;

	    //for all images
	    std::list<cMesureAppuiFlottant1Im>::const_iterator aImIt;
	    for( aImIt=aDicoIm.MesureAppuiFlottant1Im().begin();
                 aImIt!=aDicoIm.MesureAppuiFlottant1Im().end(); aImIt++)
	    {
                
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
				std::cout << aGrPtIt->NamePt() << "\n";        
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

int CPP_TestRPCDirectGen(int argc,char ** argv)
{

    cInterfChantierNameManipulateur * aICNM;
    std::list<std::string> aListFile;

    std::string aFullName;
    std::string aDir;
    std::string aPat;

    std::string aNameType;
    eTypeImporGenBundle aType;

    ElInitArgMain
    (
         argc, argv,
         LArgMain()  << EAMC(aFullName,"Orientation file full name (Dir+Pat)", eSAM_IsExistFile)
                     << EAMC(aNameType,"Type of sensor (see eTypeImporGenBundle)",eSAM_None,ListOfVal(eTT_NbVals,"eTT_")),
         LArgMain()
    );		      

    SplitDirAndFile(aDir, aPat, aFullName);

    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aPat);
    
    bool aModeHelp;
    StdReadEnum(aModeHelp,aType,aNameType,eTIGB_NbVals);
  
    
    for(std::list<std::string>::iterator itL = aListFile.begin(); 
		                         itL != aListFile.end(); 
					 itL++ )   
    {
	//Earth satellite
	if(aType!=eTIGB_Unknown && aType!=eTIGB_MMSten)
	{
            CameraRPC aCRPC(*itL,aType);
	    
	}
	//other planets or stenope camera
	else
            ELISE_ASSERT(false,"No eTypeImporGenBundle");

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
