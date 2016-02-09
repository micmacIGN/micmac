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

int InitOriLinear_main(int argc,char ** argv)
{
    cout<<"************************"<<endl;
    cout<<"*  X : Initial         *"<<endl;
    cout<<"*  X : Orientation     *"<<endl;
    cout<<"*  X : & Position      *"<<endl;
    cout<<"*  X : For Acquisition *"<<endl;
    cout<<"*  X : Linear          *"<<endl;
    cout<<"************************"<<endl;
    vector<std::string> aVecPatternNewImages;
    vector<std::string> aVecPatternRefImages;
    std::string aFullPatternNewImages, aFullPatternRefImages, aOriRef, aPatternCam1, aPatternCam2, aPatternRef1, aPatternRef2;//pattern of all files
    string aOriOut = "InitOut";
    ElInitArgMain			//initialize Elise, set which is mandantory arg and which is optional arg
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  //<< EAMC(aFullPatternNewImages, "Pattern of images to orientate",  eSAM_IsPatFile)
                << EAMC(aOriRef, "Reference Orientation",  eSAM_IsExistDirOri),
    //optional arguments
    LArgMain()  << EAM(aOriOut, "OriOut" , true, "Output initialized ori folder")
                << EAM(aFullPatternRefImages, "Pattern of already-oriented images, used to compute orientation and movement",  eSAM_IsPatFile)
                <<EAM(aVecPatternNewImages, "VPO" , true, "Vector pattern of images to orientate correspondant with each camera VPO=[Pat Cam 1, Pat Cam 2,..]")
                <<EAM(aVecPatternRefImages, "VPR", true, "Vector pattern of Reference Orientation VPR=[Pat Ref 1, Pat Ref 2,..]")
                <<EAM(aPatternCam1, "PatCam1", true, "Vector pattern of images to orientate for cam 1 (a enlever apres)")
                <<EAM(aPatternCam2, "PatCam2", true, "Vector pattern of images to orientate for cam 2 (a enlever apres)")
                <<EAM(aPatternRef1, "PatRef1", true, "Vector pattern of Ref to orientate for cam 1 (a enlever apres)")
                <<EAM(aPatternRef2, "PatRef2", true, "Vector pattern of Ref to orientate for cam 2 (a enlever apres)")

    );
    if (MMVisualMode) return EXIT_SUCCESS;

    //enlever apres fixer VPO VPR
    aVecPatternNewImages.push_back(aPatternCam1);
    if (aPatternCam2.length() > 1)
         {aVecPatternNewImages.push_back(aPatternCam2);}
    aVecPatternRefImages.push_back(aPatternRef1);
    if (aPatternRef2.length() > 1)
    {aVecPatternRefImages.push_back(aPatternRef2);}

    //========//
    cout<<"System with "<<aVecPatternNewImages.size()<<" cameras"<<endl;

    if (aVecPatternNewImages.size() >= 1)
    {
        double xOffsetRef = 0;double yOffsetRef = 0;double zOffsetRef = 0;
        double xOffset = 0;double yOffset = 0;double zOffset = 0;
    for(uint ii=0; ii<aVecPatternNewImages.size(); ii++)
    {
        cout<<"\nInit Cam "<<ii<<" : ";
        aFullPatternNewImages = RequireFromString<string>(aVecPatternNewImages[ii],"Pat Cam");
        aFullPatternRefImages = RequireFromString<string>(aVecPatternRefImages[ii],"Pat Cam");
        cout<<"***"<<aFullPatternNewImages<<"***"<<endl;

        // Initialize name manipulator & files
        std::string aDirNewImages,aDirRefImages, aPatNewImages,aPatRefImages;
        SplitDirAndFile(aDirNewImages,aPatNewImages,aFullPatternNewImages);
        SplitDirAndFile(aDirRefImages,aPatRefImages,aFullPatternRefImages);
        StdCorrecNameOrient(aOriRef,aDirRefImages);//remove "Ori-" if needed

        cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirNewImages);
        const std::vector<std::string> aSetNewImages = *(aICNM->Get(aPatNewImages));

        std::cout<<"\nInit images:\n";
        for (unsigned int ik=0;ik<aSetNewImages.size();ik++)
            std::cout<<"  - "<<aSetNewImages[ik]<<"\n";


        aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirRefImages);
        const std::vector<std::string> aSetRefImages = *(aICNM->Get(aPatRefImages));

        ELISE_ASSERT(aSetRefImages.size()>1,"Number of reference image must be > 1");

        std::cout<<"\nRef images:\n";
        //Read orientation initial (first image in series)
        string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages.back()+".xml";
        cOrientationConique aOriConiqueRef=StdGetFromPCP(aOriRefImage,OrientationConique);

        //init relative position b/w different series image
        if (ii==0) //1st camera as reference
        {
            xOffset = yOffset = zOffset = 0;
            xOffsetRef = aOriConiqueRef.Externe().Centre().x;
            yOffsetRef = aOriConiqueRef.Externe().Centre().y;
            zOffsetRef = aOriConiqueRef.Externe().Centre().z;
        }
        else
        {
            xOffset = aOriConiqueRef.Externe().Centre().x - xOffsetRef;
            yOffset = aOriConiqueRef.Externe().Centre().y - yOffsetRef;
            zOffset = aOriConiqueRef.Externe().Centre().z - zOffsetRef;
            cout<<"Offset = "<<xOffset<<" - "<<yOffset<<" - "<<zOffset;
        }
        std::vector<cOrientationConique> aRefOriList;
        double xBefore=0, yBefore=0, zBefore=0;
        double xAcc = 0, yAcc = 0, zAcc = 0;

        for (unsigned int i=0;i<aSetRefImages.size();i++)
        {
            std::cout<<"  - "<<aSetRefImages[i]<<" ";
            std::string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages[i]+".xml";
            //Pour orientation
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
            xEstimate = xEstimate + xMov + xOffset;
            yEstimate = yEstimate + yMov + yOffset;
            zEstimate = zEstimate + zMov + zOffset;
            aOriConique.Externe().ParamRotation().CodageMatr().SetVal(aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val());
            MakeFileXML(aOriConique, "Ori-"+aOriOut+"/Orientation-"+aSetNewImages[i]+".xml");
        }
    }
    }
    else
    {
        std::cout<<"***"<<aFullPatternNewImages<<"***"<<std::endl;
        std::cout<<"***"<<aFullPatternRefImages<<"***"<<std::endl;

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
            /*aRefCamList.push_back(CamOrientGenFromFile(aOriRefImage,aICNM));
        std::cout<<aRefCamList.back()->VraiOpticalCenter()<<"\n";
        std::cout<<aRefCamList.back()->VraiOpticalCenter()<<"\n";*/
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
    }
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
