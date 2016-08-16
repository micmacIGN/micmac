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
#include "PHO_MI.h"


//   R3 : "reel" coordonnee initiale
//   L3 : "Locale", apres rotation
//   C2 :  camera, avant distortion
//   F2 : finale apres Distortion
//
//       Orientation      Projection      Distortion
//   R3 -------------> L3------------>C2------------->F2

    /******************************************************************************
    The main function.
    ******************************************************************************/

int MeshPartViewable_main(int argc,char ** argv)
{
    cout<<"**********************************************************"<<endl;
    cout<<"*       Which part of mesh is viewable by an image	    *"<<endl;
    cout<<"**********************************************************"<<endl;

    string aMeshIn; string aPicIn; string aOriIn; double mulFactor = 0.05;
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aMeshIn, "Mesh",  eSAM_IsExistFile)
                << EAMC(aPicIn, "Image",  eSAM_IsPatFile)
                << EAMC(aOriIn, "Ori",  eSAM_IsExistDirOri),
                //optional arguments
                LArgMain()
                << EAM(mulFactor, "mF", true, "adjust length of vector - default = 0.05")
            );

    if (MMVisualMode) return EXIT_SUCCESS;
    bool Exist= ELISE_fp::exist_file(aMeshIn);
    InitOutil aChain(aPicIn, aOriIn);
    aChain.load_Im();
    vector<pic*>lstPic = aChain.getmPtrListPic();
    vector<Pt3dr> listPts;
    for (uint i=0; i<lstPic.size(); i++)
    {
        pic * aPic = lstPic[i];
        CamStenope * aCamPic = aPic->mOriPic;
        cout<<"Img: "<<aPicIn<<endl;
        cout<<" ++ VraiOpticalCenter :"<<aCamPic->VraiOpticalCenter()<<endl;
        cout<<" ++ Point Prespective :"<<aCamPic->PP()<<endl;
        cout<<" ++ PP toDirRayonR3 :"<<aCamPic->F2toDirRayonR3(aCamPic->PP())<<endl;
        Pt3dr P1; Pt3dr P2; aCamPic->F2toRayonR3(aCamPic->PP(), P1, P2);
        cout<<" ++ PP toRayonR3 :"<<P1<<P2<<endl;
        listPts.push_back(P1);listPts.push_back(P2);
    }
    DrawOnMesh aDraw;
    aDraw.creatPLYPts3D(listPts, "./PlyVerify/camVerif.ply", Pt3dr(0,255,255));
//    vector<Pt3dr> centre_ge;
//    vector<Pt3dr> vecNormal;
//    if (Exist)
//    {
//        InitOutil * aChain = new InitOutil(aMeshIn);
//        vector<triangle*> aListTri = aChain->getmPtrListTri();
//        for(uint i=0; i<aListTri.size(); i++)
//        {
//            triangle * atri = aListTri[i];
//            Pt3dr centre_geo;
//            Pt3dr aVecNor = atri->CalVecNormal(centre_geo, mulFactor);
//            centre_ge.push_back(centre_geo);
//            vecNormal.push_back(aVecNor);
//        }
//    }
//    else
//        cout<<"Mesh not existed ! "<<aMeshIn<<endl;
    return EXIT_SUCCESS;
}



