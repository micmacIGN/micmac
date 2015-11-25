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



//----------------------------------------------------------------------------

int ExtractMesure2D_main(int argc,char ** argv)
{
    std::string a2DMesFileName, aOutputFile;
    std::vector<std::string> aTargetNameList;
    ElInitArgMain			//initialize Elise, set which is mandantory arg and which is optional arg
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(a2DMesFileName, "Input mes2D file",  eSAM_IsExistFile)
                << EAMC(aOutputFile, "Output mes2D file",  eSAM_IsOutputFile)
                << EAMC(aTargetNameList,"List of selected targets. Ex: [target1,target2])"),
    //optional arguments
    LArgMain()

    );
    if (MMVisualMode) return EXIT_SUCCESS;

    std::cout<<"List of selected targets:\n";
    for (unsigned int i=0;i<aTargetNameList.size();i++)
        std::cout<<" * "<<aTargetNameList.at(i)<<"\n";


    cSetOfMesureAppuisFlottants aSetOfMesureAppuisFlottants=StdGetFromPCP(a2DMesFileName,SetOfMesureAppuisFlottants);

    cSetOfMesureAppuisFlottants aNewSetOfMesureAppuisFlottants;
    for( std::list< cMesureAppuiFlottant1Im >::const_iterator iTmes1Im=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().begin();
         iTmes1Im!=aSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().end();          iTmes1Im++    )
    {
        cMesureAppuiFlottant1Im anIm=*iTmes1Im;
        cMesureAppuiFlottant1Im aNewIm;
        aNewIm.NameIm()=anIm.NameIm();
        //std::cout<<anIm.NameIm()<<" ";
        for( std::list< cOneMesureAF1I >::const_iterator iTmes=anIm.OneMesureAF1I().begin();
             iTmes!=anIm.OneMesureAF1I().end();          iTmes++    )
        {
            cOneMesureAF1I aMes=*iTmes;

            std::vector<std::string>::iterator findIter = std::find(aTargetNameList.begin(), aTargetNameList.end(), aMes.NamePt());
            if (findIter!=aTargetNameList.end())
            {
               //std::cout<<aMes.NamePt()<<" ";
               aNewIm.OneMesureAF1I().push_back(aMes);
            }
        }
        //std::cout<<std::endl;
        if (aNewIm.OneMesureAF1I().size()>0)
            aNewSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().push_back(aNewIm);
    }

    //std::cout<<std::endl;

    MakeFileXML(aNewSetOfMesureAppuisFlottants, aOutputFile);
    std::cout<<"Found "<<aNewSetOfMesureAppuisFlottants.MesureAppuiFlottant1Im().size()<<" usable images.\n";
    std::cout<<"Finished!"<<std::endl;

/*

    // Initialize name manipulator & files
    std::string aDirNewImages,aDirRefImages, aPatNewImages,aPatRefImages;
    SplitDirAndFile(aDirNewImages,aPatNewImages,aFullPatternNewImages);	
    SplitDirAndFile(aDirRefImages,aPatRefImages,aFullPatternRefImages);	
	StdCorrecNameOrient(aOriRef,aDirRefImages);//remove "Ori-" if needed
    std::cout<<"New images dir: "<<aDirNewImages<<std::endl;

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirNewImages);
    const std::vector<std::string> aSetNewImages = *(aICNM->Get(aPatNewImages));		//cInterfChantierNameManipulateur::BasicAlloc(aDirImages) have method Get to read path with RegEx
    
    std::cout<<"\nNew images:\n";
    for (unsigned int i=0;i<aSetNewImages.size();i++)
		std::cout<<"  - "<<aSetNewImages[i]<<"\n";
    
    
    aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirRefImages);
    const std::vector<std::string> aSetRefImages = *(aICNM->Get(aPatRefImages));
    
    ELISE_ASSERT(aSetRefImages.size()>1,"Number of reference image must be > 1");
    
    std::cout<<"\nRef images:\n";
		
	std::vector<cOrientationConique> aRefOriList;
	double xBefore=0, yBefore=0, zBefore=0;
	double xAcc = 0, yAcc = 0, zAcc = 0; 
			
	for (unsigned int i=0;i<aSetRefImages.size();i++)
	{
		std::cout<<"  - "<<aSetRefImages[i]<<" ";
		std::string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages[i]+".xml";
		cOrientationConique aOriConique=StdGetFromPCP(aOriRefImage,OrientationConique);
		aRefOriList.push_back(aOriConique);
		std::cout<<aOriConique.Externe().Centre()<<"\n";
		if (i==0)
			{
				xBefore = aOriConique.Externe().Centre().x;
				yBefore = aOriConique.Externe().Centre().y;
				zBefore = aOriConique.Externe().Centre().z;
			}
		xAcc = xAcc + aOriConique.Externe().Centre().x - xBefore;
		yAcc = yAcc + aOriConique.Externe().Centre().y - yBefore;
		zAcc = zAcc + aOriConique.Externe().Centre().z - zBefore;
		xBefore =  aOriConique.Externe().Centre().x;
		yBefore = aOriConique.Externe().Centre().y;
		zBefore = aOriConique.Externe().Centre().z;
	}
	//compute orientation and movement
	double xMov = xAcc/(aSetRefImages.size()-1);
	double yMov = yAcc/(aSetRefImages.size()-1);
	double zMov = zAcc/(aSetRefImages.size()-1);
	cout<<endl<<"Init with vector movement = "<<xMov<<" ; "<<yMov<<" ; "<<zMov<<endl;
	//Create a XML file with class cOrientationConique (define in ParamChantierPhotogram.h)
	double xEstimate = aRefOriList.front().Externe().Centre().x;
	double yEstimate = aRefOriList.front().Externe().Centre().y;
	double zEstimate = aRefOriList.front().Externe().Centre().z;
	cOrientationConique aOriConique = aRefOriList.front();
	//std::cout<<"\nInit Images:\n";
	for (unsigned int i=0;i<aSetNewImages.size();i++)
	{
		//std::cout<<"  - "<<aSetNewImages[i]<<"\n";
		aOriConique.Externe().Centre().x = xEstimate;
		aOriConique.Externe().Centre().y = yEstimate;
		aOriConique.Externe().Centre().z = zEstimate;	
		xEstimate = xEstimate + xMov;
		yEstimate = yEstimate + yMov;
		zEstimate = zEstimate + zMov;
		MakeFileXML(aOriConique, "Ori-"+aOriRef+"/Orientation-"+aSetNewImages[i]+".xml");
	}
    */
    return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

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
Footer-MicMac-eLiSe-25/06/2007/*/
