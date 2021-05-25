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
    cout<<"*       under a specified angle (not consider ZBuffer)	*"<<endl;
    cout<<"**********************************************************"<<endl;

    string aMeshIn; string aPicIn; string aOriIn; double mulFactor = 0.05; double angleF = 60;
    vector<string> colorIn;
    Pt3dr color(0,255,0);
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aMeshIn, "Mesh",  eSAM_IsExistFile)
                << EAMC(aPicIn, "Image",  eSAM_IsPatFile)
                << EAMC(aOriIn, "Ori",  eSAM_IsExistDirOri)
                << EAMC(angleF, "filter angle (b/w rayon (camera-centre-geo, vecNormal))",  eSAM_None),
                //optional arguments
                LArgMain()
                << EAM(mulFactor, "mF", true, "adjust length of vector - default = 0.05")
                << EAM(colorIn, "color", true, "color RGB color=[R,G,B]")
            );

    if (MMVisualMode) return EXIT_SUCCESS;
    if (colorIn.size() > 0)
    {
        vector<double> a = parse_dParam(colorIn);
        color.x = a[0]; color.y = a[1]; color.z = a[2];
    }
    bool Exist= ELISE_fp::exist_file(aMeshIn);
    InitOutil aChain(aPicIn, aOriIn);
    if (Exist)
        aChain.initAll(aMeshIn);
    else
        cout<<"Mesh not Existed :"<<aMeshIn<<endl;
    vector<pic*>lstPic = aChain.getmPtrListPic();
    vector<triangle*>lstTri = aChain.getmPtrListTri();
    //angleF = pow(cos(angleF*PI/180),2);

if (lstPic.size() == 1)
{
    vector<Pt3dr> listPts;
    string aPlyOutDir;
    aPlyOutDir="./PlyVerify/";
    if(!(ELISE_fp::IsDirectory(aPlyOutDir)))
        ELISE_fp::MkDir(aPlyOutDir);

    vector<Pt3dr> centre_ge;
    vector<Pt3dr> centre_ca;
    vector<Pt3dr> vec_nor;
    vector<triangle*> triVisible;

    for (uint i=0; i<lstPic.size(); i++)
    {
        pic * aPic = lstPic[i];
        CamStenope * aCamPic = aPic->mOriPic;
        cout<<"Img: "<<aPicIn<<endl;
        cout<<" ++ VraiOpticalCenter :"<<aCamPic->VraiOpticalCenter()<<endl;
        cout<<" ++ Point Prespective :"<<aCamPic->PP()<<endl;
        cout<<" ++ PP toDirRayonR3 :"<<aCamPic->F2toDirRayonR3(aCamPic->PP())<<endl;
        Pt3dr centre_cam = aCamPic->VraiOpticalCenter();
        for (uint j=0; j<lstTri.size(); j++)
        {
            triangle * aTri = lstTri[j];
            Pt3dr centre_geo = (aTri->getSommet(0) + aTri->getSommet(1) + aTri->getSommet(2))/ 3;
            Pt3dr Vec1 = centre_cam - centre_geo;
            Pt3dr aVecNor = aTri->CalVecNormal(centre_geo, mulFactor);
            Pt3dr Vec2 = aVecNor - centre_geo;
            bool devant = aCamPic->Devant(centre_geo);
            double angle_deg = (aTri->calAngle(Vec1, Vec2))*180/PI;
            cout<<angle_deg<<endl;
            if ( (angle_deg<angleF) && devant )
            {
                centre_ge.push_back(centre_geo);
                centre_ca.push_back(centre_cam);
                vec_nor.push_back(aVecNor);
                triVisible.push_back(aTri);
            }
        }
    }
    DrawOnMesh aDraw;
    aDraw.drawEdge_p1p2(centre_ge, vec_nor, aPlyOutDir + aPicIn + "_vecNor.ply", Pt3dr(0,255,0), Pt3dr(0,255,0));
    aDraw.drawEdge_p1p2(centre_ge, centre_ca, aPlyOutDir + aPicIn + "_CamRay.ply", Pt3dr(255,255,0), Pt3dr(255,255,0));
    aDraw.drawListTriangle(triVisible, aPlyOutDir + aPicIn + "_meshViewable.ply", color);
}
else
{
    cout<<"Too much image or image not found. Give just 1 image. Bye!"<<endl;
}
    return EXIT_SUCCESS;
}


