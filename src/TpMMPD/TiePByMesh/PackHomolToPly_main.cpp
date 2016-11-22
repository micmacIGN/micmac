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

#include "InitOutil.h"
#include <stdio.h>

#include "Triangle.h"
#include "Pic.h"
#include "DrawOnMesh.h"
#include "CorrelMesh.h"
#include "PHO_MI.h"

    /******************************************************************************
    The main function.
    ******************************************************************************/

int PackHomolToPly_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    		Draw a pack of homologue in 3D PLY		  *"<<endl;
    cout<<"********************************************************"<<endl;

    string aFullPattern, aOriInput; string aNameHomol = "Homol";
    Pt3dr color(0,255,0);
    vector<string> colorIn;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aFullPattern, "Pattern of images - 2 image have a pack",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri),
                //optional arguments
                LArgMain()
                << EAM(aNameHomol, "SH", true, "homol folder name - default = Homol")
                << EAM(colorIn, "color", true, "[R,B,G] - default = [0,255,0]")
            );

    if (MMVisualMode) return EXIT_SUCCESS;
    if (colorIn.size() > 0)
    {
        vector<double> a = parse_dParam(colorIn);
        color.x = a[0]; color.y = a[1]; color.z = a[2];
    }
    InitOutil * aChain = new InitOutil(aFullPattern, aOriInput, aNameHomol);
    aChain->load_Im();
    vector<pic*> ptrPic = aChain->getmPtrListPic();
    cout<<" ++ Pic:"<<endl;
    for (uint i=0; i<ptrPic.size(); i++)
        cout<<"     ++"<<ptrPic[i]->getNameImgInStr()<<endl;
    DrawOnMesh aDraw(aChain);
    string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aNameHomol)
                       +  std::string("@")
                       +  std::string("dat");
    pic * pic1 = aChain->getmPtrListPic()[0];
    pic * pic2 = aChain->getmPtrListPic()[1];
    string aHomoIn = aChain->getPrivmICNM()->Assoc1To2(aKHIn, pic1->getNameImgInStr(), pic2->getNameImgInStr(), true);
    ElPackHomologue aPack;
    bool Exist= ELISE_fp::exist_file(aHomoIn);
    if (Exist)
    {
        aPack =  ElPackHomologue::FromFile(aHomoIn);
        aDraw.drawPackHomoOnMesh(aPack, pic1, pic2, color, aNameHomol);
    }
    else
    {
        StdCorrecNameHomol_G(aHomoIn, aChain->getPrivmICNM()->Dir());
        Exist= ELISE_fp::exist_file(aHomoIn);
        if (Exist)
        {
            aPack =  ElPackHomologue::FromFile(aHomoIn);
            aDraw.drawPackHomoOnMesh(aPack, pic1, pic2, color, aNameHomol);
        }
        else
            cout<<"Homol pack not found"<<endl;
    }
    cout<<" ++ Verify path homo:"<<endl<<"      ++"<<aHomoIn<<endl;
    return EXIT_SUCCESS;
}



