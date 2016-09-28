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

#include "Pic.h"
#include "Fast.h"
#include <stdio.h>

extern vector<double> parse_dParam(vector<string> dParam);

    /******************************************************************************
    The main function.
    ******************************************************************************/
int FAST_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    FAST - Detector interest point FAST               *"<<endl;
    cout<<"********************************************************"<<endl;
        string aFullPattern;
        string aDirOut = "PtsInteret";
        vector<string> dParam; dParam.push_back("20");dParam.push_back("3");
        vector<double> aParamD;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile),
                    //optional arguments
                    LArgMain()
                    << EAM(aDirOut, "aDirOut", true, "Output directory for pts interest file, default=PtsInteret")
                    << EAM(dParam, "dParam", true, "detector parameter, default=[20,3]")
                 );

        if (MMVisualMode) return EXIT_SUCCESS;

        string aDirImages, aPatIm;
        SplitDirAndFile(aDirImages, aPatIm, aFullPattern); //Working dir, Images pattern
        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
        vector<string>  aSetIm = *(aICNM->Get(aPatIm));
        ELISE_ASSERT(aSetIm.size()>0,"ERROR: No image found!");

        vector<pic*> aPtrListPic;
        for (uint i=0; i<aSetIm.size(); i++)
        {
            pic *aPic = new pic( &aSetIm[i], "NONE" , aICNM, i);
            aPtrListPic.push_back(aPic);
        }

        aParamD = parse_dParam(dParam); //need to to on arg enter

        for (uint i=0; i<aPtrListPic.size(); i++)
        {
            pic * aPic = aPtrListPic[i];
            Fast aDetecteur(aParamD[0], aParamD[1]);
            vector<Pt2dr> aPtsInterest;
            aDetecteur.detect(*aPic->mPic_Im2D, aPtsInterest);


            string outDigeoFile =  aPic->getNameImgInStr() + "_" + "FAST" + ".dat";
            vector<DigeoPoint> points;
            for (uint j=0; j<aPtsInterest.size(); j++)
            {
                DigeoPoint aPt;
                aPt.x = aPtsInterest[j].x;
                aPt.y = aPtsInterest[j].y;
                points.push_back(aPt);
            }
            string aPtsInteretOutDirName;
            aPtsInteretOutDirName = aDirImages + aDirOut + "/";
            if(!(ELISE_fp::IsDirectory(aPtsInteretOutDirName)))
                ELISE_fp::MkDir(aPtsInteretOutDirName);
            if ( !DigeoPoint::writeDigeoFile(aPtsInteretOutDirName + outDigeoFile, points))
                ELISE_ERROR_EXIT("failed to write digeo file [" << outDigeoFile << "]");
        }


        return EXIT_SUCCESS;
    }
