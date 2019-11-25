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

#include "Sift/Sift.h"


int CPP_SiftExport_main(int argc,char** argv)
{

    const std::vector<std::string> * aSetName;
    std::string aPattern;
    std::string aDir;
    std::string aNameSift;
    std::vector<Siftator::SiftPoint> aVSift;
    int aSzSift = -1;

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aPattern,"Pattern of images"),
        LArgMain() << EAM(aSzSift,"Resol",true,"Resolution of tie-pts extraction") 
    );
    #if (ELISE_windows)
        replace( aPattern.begin(), aPattern.end(), '\\', '/' );
    #endif


    SplitDirAndFile(aDir,aPattern,aPattern);

    cElemAppliSetFile anEASF(aPattern);
    aSetName = anEASF.SetIm();
    
    CamStenope * aCamTmp = anEASF.mICNM->GlobCalibOfName(aSetName->at(0),"",0);
    Pt2di aSzOrg = aCamTmp->SzBasicCapt3D();
    double aSSF =  (aSzSift<0) ? 1.0 :   double( ElMax( aSzOrg.x, aSzOrg.y ) ) / double( aSzSift ) ;

    for (auto aK : *aSetName)
    {
        std::string aFOut = "SIFT_Descr_" + aK + ".txt";
        FILE * aFP=0;

        getPastisGrayscaleFilename(aDir,aK,aSzSift,aNameSift);

        if (EAMIsInit(&aSzSift))
        {
            aNameSift = DirOfFile(aNameSift) + "LBPp" + NameWithoutDir(aNameSift) + ".dat";
        }
        else
        {
            aNameSift  = "LBPp" + aK + ".dat";
            aNameSift = "Pastis/" + aNameSift;//full resolution
        }
        

        bool Ok = read_siftPoint_list(aNameSift,aVSift);

        if (!Ok)
        {
            std::cout << aNameSift << "\n";
            ELISE_ASSERT(false,"CPP_SiftExport_main. Coudn't read SIFT.");
        }


        ELISE_fp::RmFileIfExist(aFOut);        
        aFP = ElFopen(aFOut.c_str(),"w");
        fprintf(aFP,"-------------- x y 128 SIFT descriptors  -----------\n");

        for (auto aPt : aVSift)
        {
            //std::cout << aPt.x << " " << aPt.y << " " << aPt.scale << "\n";
            fprintf(aFP," %lf %lf\n",aPt.x*aSSF,aPt.y*aSSF);
            for (int aDescr=0; aDescr<SIFT_DESCRIPTOR_SIZE; aDescr++)
            {
                //std::cout << " * " << aPt.descriptor[aDescr] ;
                fprintf(aFP," %lf ",aPt.descriptor[aDescr]);
            }
            fprintf(aFP,"\n");
        }
        std::cout << "Saved to: " << aFOut << "\n";
        ElFclose(aFP);
    }

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
