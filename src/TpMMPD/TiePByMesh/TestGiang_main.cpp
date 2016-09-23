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

int TestGiang_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    TestGiang                                         *"<<endl;
    cout<<"********************************************************"<<endl;

    string aFullPattern, aOriInput, aHomolOut, aTYpeD;
    double corl_seuil_glob, corl_seuil_pt, SzPtCorr, SzAreaCorr, aAngleF;
    bool useExistHomoStruct;
    vector<string> dParam; dParam.push_back("NO");
    string aTypeD = "DIGEO";

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()  
                << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri),
                //optional arguments
                LArgMain()
                << EAM(corl_seuil_glob, "corl_glob", true, "corellation threshold for imagette global, default = 0.8")
                << EAM(corl_seuil_pt, "corl_pt", true, "corellation threshold for pt interest, default = 0.9")
                << EAM(SzPtCorr, "SzPtCorr", true, "1->3*3,2->5*5 size of cor wind for each pt interet  default=1 (3*3)")
                << EAM(SzAreaCorr, "SzAreaCorr", true, "1->3*3,2->5*5 size of zone autour pt interet for search default=5 (11*11)")
                << EAM(aTypeD, "aTypeD", true, "FAST, DIGEO, HOMOLINIT - default = HOMOLINIT")
                << EAM(dParam,"dParam",true,"[param1, param2, ..] (selon detector - NO if don't have)")
                << EAM(aHomolOut, "HomolOut", true, "default = _Filtered")
                << EAM(useExistHomoStruct, "useExist", true, "use exist homol struct - default = false")
                << EAM(aAngleF, "angleV", true, "limit view angle - default = 90 (all triangle is viewable)")
              );

    if (MMVisualMode) return EXIT_SUCCESS;
    vector<double> aParamD = parse_dParam(dParam);
    InitOutil * aChain = new InitOutil(aFullPattern, aOriInput);
    aChain->initAll();
    vector<pic*> lstPic = aChain->getmPtrListPic();
    for (uint i=0; i<lstPic.size(); i++)
    {
        pic * aPic = lstPic[i];
        Detector * aDetect = new Detector(aTypeD, aParamD, aPic, aChain);
        aDetect->detect();
        aDetect->saveToPicTypeVector(aPic);
        delete aDetect;



    }


    return EXIT_SUCCESS;
}



