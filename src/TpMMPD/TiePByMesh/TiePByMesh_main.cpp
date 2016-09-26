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
#include "InitOutil.h"
#include "Triangle.h"
#include "Pic.h"
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
    cout<<"dParam : param of detector : "<<endl;
    cout<<"     [FAST_Threshold]"<<endl;
    cout<<"     NO"<<endl;

    string pathPlyFileS ;
    string method="SubCoor";string aTypeD="HOMOLINIT";
    string aFullPattern, aOriInput;
    string aHomolOut = "_Filtered";
    bool disp = 0; bool assum1er=0;
    int SzPtCorr = 1;int indTri=-1;double corl_seuil_glob = 0.8;bool Test=0;
    int SzAreaCorr = 5; double corl_seuil_pt = 0.9;
    vector<string> dParam; dParam.push_back("NO");
    bool useExistHomoStruct = false;
    double aAngleF = 90;
    bool justFil = false;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                //optional arguments
                LArgMain()
                << EAM(disp, "disp", true, "display")
                << EAM(corl_seuil_glob, "corl_glob", true, "corellation threshold for imagette global, default = 0.8")
                << EAM(corl_seuil_pt, "corl_pt", true, "corellation threshold for pt interest, default = 0.9")
                << EAM(method, "method", true, "Coor, SubCoor, default=SubCoor")
                << EAM(SzPtCorr, "SzPtCorr", true, "1->3*3,2->5*5 size of cor wind for each pt interet  default=1 (3*3)")
                << EAM(SzAreaCorr, "SzAreaCorr", true, "1->3*3,2->5*5 size of zone autour pt interet for search default=5 (11*11)")
                << EAM(indTri, "indTri", true, "process one triangle")
                << EAM(assum1er, "assum1er", true, "always use 1er pose as img master, default=0")
                << EAM(Test, "Test", true, "Test new method - fix size imagette of triangle")
                << EAM(aTypeD, "aTypeD", true, "FAST, DIGEO, HOMOLINIT - default = HOMOLINIT")
                << EAM(dParam,"dParam",true,"[param1, param2, ..] (selon detector - NO if don't have)", eSAM_NoInit)
                << EAM(aHomolOut, "HomolOut", true, "default = _Filtered")
                << EAM(useExistHomoStruct, "useExist", true, "use exist homol struct - default = false")
                << EAM(aAngleF, "angleV", true, "limit view angle - default = 90 (all triangle is viewable)")
                << EAM(justFil, "justFil", true, "just do filter.")
                );

    if (MMVisualMode) return EXIT_SUCCESS;
    if (justFil)
    {
        useExistHomoStruct = true;
        aTypeD = "HOMOLINIT";
    }
    vector<double> aParamD = parse_dParam(dParam); //need to to on arg enter
    InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                      SzPtCorr, SzAreaCorr,
                                      corl_seuil_glob, corl_seuil_pt, disp, useExistHomoStruct);
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
    cout<<PtrTri.size()<<" tri"<<endl;
    CorrelMesh aCorrel(aChain);
    if (!justFil)
    {
        if (aAngleF == 90)
        {
            for (uint i=0; i<PtrTri.size(); i++)
            {
                if (useExistHomoStruct)
                    aCorrel.correlByCplExist(i);
                else
                    aCorrel.correlInTri(i);
            }
        }
        else
        {
            cout<<"Use condition angle view"<<endl;
            for (uint i=0; i<PtrTri.size(); i++)
            {
                if (useExistHomoStruct)
                    aCorrel.correlByCplExistWithViewAngle(i, aAngleF);
                else
                    aCorrel.correlInTriWithViewAngle(i, aAngleF);
            }
        }
    }
    else
    {
        cout<<"Filter pt homo exist by verif couple pt homo in same triangulation !"<<endl;
        double  pct = PtrTri.size()/100;
        for (uint i=0; i<PtrTri.size(); i++)
        {
           pct = i/PtrTri.size()*100;
           aCorrel.verifCplHomoByTriangulation(i, aAngleF);
           cout<<pct<<" %";
        }
    }
    cout<<endl<<"Tri has pt inside: ";
    for (uint i=0; i<aCorrel.mTriHavePtInteret.size(); i++)
        cout<<" "<<aCorrel.mTriHavePtInteret[i];
    cout<<endl<<endl<<"Tri has Correl super: ";
    for (uint i=0; i<aCorrel.mTriCorrelSuper.size(); i++)
        cout<<" "<<aCorrel.mTriCorrelSuper[i];

    cout<<endl<<"Total "<<aCorrel.countPts<<" cpl homo"<<endl;
    cout<<endl;

    return EXIT_SUCCESS;
}



