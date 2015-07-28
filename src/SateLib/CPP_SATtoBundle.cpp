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
        std::string mModeOri;
	std::string mMetadata;
	Pt2di mGridSz;
};

cSatI_Appli::cSatI_Appli(int argc,char ** argv) :
	mCSysOut(""),
	mMetadata("")
{
    std::string aFullName;
    std::string aDir;
    std::string aPat;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Orientation file (RPC/SPICE/RAY_BUNDLES) full name (Dir+Pat)", eSAM_IsExistFile),
         LArgMain() << EAM(mModeOri, "ModeOri", true, "The RPC convention (PLEIADE,SPOT,QUICKBIRD,WORLDVIEW,IKONOS,CARTOSAT,SPICE)")
	            << EAM(mCSysOut, "Proj", true, "Output cartographic coordinate system (proj format)")
                    << EAM(mGridSz,"GrSz",true, "No. of grids of bundles, e.g. GrSz=[5,8]", eSAM_NoInit)
		    << EAM(mMetadata, "Meta", true, "Sensor metadata file, other than the RPC; Valid for IKONOS and CARTOSAT", eSAM_IsExistFile)
    );		      

    SplitDirAndFile(aDir, aPat, aFullName);


    //validate the RPC mode 
    if (mModeOri=="PLEIADE") {}
    else if (mModeOri=="SPOT") {}
    else if (mModeOri=="QUICKBIRD") {}
    else if (mModeOri=="WORLDVIEW") {}
    else if (mModeOri=="IKONOS") {}
    else if (mModeOri=="CARTOSAT") {}
    else if (mModeOri=="SPICE") {}
    else {ELISE_ASSERT(false,"Unknown RPC mode");}    


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
	if(aApps.mModeOri!="SPICE")
	{
            CameraRPC aCurCam(*itL,aApps.mModeOri,aApps.mGridSz);
	    aCurCam.ExpImp2Bundle(aApps.mCSysOut);
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
 
int SATtoOpticalCenter_main(int argc,char ** argv)
{

    cSatI_Appli aApps(argc,argv);


    for(std::list<std::string>::iterator itL = aApps.mListFile.begin(); 
		                         itL != aApps.mListFile.end(); 
					 itL++ )
    {
       //Earth satellite
       if(aApps.mModeOri!="SPICE")
       {
           CameraRPC aCamRPC(*itL,aApps.mModeOri,aApps.mGridSz,aApps.mMetadata);
           aCamRPC.OpticalCenterLineTer(aApps.mCSysOut, true);
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
