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

#include <stdio.h>
#include "StdAfx.h"
#include "Triangle.h"
#include "Pic.h"
#include "InitOutil.h"
#include "DrawOnMesh.h"
#include "CorrelMesh.h"

    /******************************************************************************
    The main function.
    ******************************************************************************/

int TiePByMesh_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    Search for tie-point using mesh + correlation     *"<<endl;
    cout<<"********************************************************"<<endl;

    string pathPlyFileS ;
    string method="SubCoor";string aTypeD="HOMOLINIT";
    std::string aFullPattern, aOriInput;
    bool disp = 0;bool disp_glob = 0; bool disp_TriOnPic=0; bool assum1er=0; bool dispPtsInteret=0;
    int SizeWinCorr = 2;int indTri=-1;double corl_seuil = 0.9;bool Test=0;


    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                //optional arguments
                LArgMain()
                << EAM(disp_glob, "disp_glob", true, "display imagette global (of triangle), click on imagette 2 to continue, default = false")
                << EAM(disp, "disp", true, "display imagette for sub-pxl corellation, click on imagette 2 to continue, default = false")
                << EAM(disp_TriOnPic, "disp_TriOnPic", true, "draw Triangle reprojected on image, default = false")
                << EAM(dispPtsInteret, "dispPtsInteret", true, "display pts interest detected on imagette master, default = false")
                << EAM(corl_seuil, "corl_seuil", true, "corellation threshold for imagette global, default = 0.9")
                << EAM(method, "method", true, "Coor, SubCoor, default=SubCoor")
                << EAM(SizeWinCorr, "SzW", true, "1->3*3,2->5*5 size of sub-pxl correlation windows default=2 (5*5)")
                << EAM(indTri, "indTri", true, "process one triangle")
                << EAM(assum1er, "assum1er", true, "always use 1er pose as img master, default=0")
                << EAM(Test, "Test", true, "Test new method - fix size imagette of triangle")
                << EAM(aTypeD, "aTypeD", true, "FAST, DIGEO, HOMOLINIT")
                );

    if (MMVisualMode) return EXIT_SUCCESS;
    vector<double> aParamD; //need to to on arg enter
    InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD);
    aChain->initAll(pathPlyFileS);

    cout<<endl<<" +++ Verify init: +++"<<endl;
    vector<pic*> PtrPic = aChain->getmPtrListPic();
    for (uint i=0; i<PtrPic.size(); i++)
    {
        cout<<PtrPic[i]->getNameImgInStr()<<" has ";
        vector<PackHomo> packHomoWith = PtrPic[i]->mPackHomoWithAnotherPic;
        cout<<packHomoWith.size()<<" homo packs with another pics"<<endl;
        for (uint j=0; j<packHomoWith.size(); j++)
        {
            if (j!=i)
                cout<<" ++ "<< PtrPic[j]->getNameImgInStr()<<" "<<packHomoWith[j].aPack.size()<<" pts"<<endl;
        }
    }

    vector<triangle*> PtrTri = aChain->getmPtrListTri();
    for (uint i=0; i<PtrTri.size(); i++)
    {
        CorrelMesh aCorrel(aChain);
        //aCorrel.correlInTri(i);
    }

    return EXIT_SUCCESS;
}



