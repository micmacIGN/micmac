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
#include "DrawOnMesh.h"
#include "CorrelMesh.h"
#include <stdio.h>
#include "Triangle.h"
#include "Pic.h"
#include "PHO_MI.h"

    /******************************************************************************
    The main function.
    ******************************************************************************/

int DrawVectorNormalMesh_main(int argc,char ** argv)
{
    cout<<"**********************************************************"<<endl;
    cout<<"* Calcul vector normal of each triangle and draw on mesh *"<<endl;
    cout<<"**********************************************************"<<endl;

    string aMeshIn;
    Pt3dr color(0,255,0);
    vector<string> colorIn;
    double mulFactor = 0.05;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aMeshIn, "Mesh",  eSAM_IsExistFile),
                //optional arguments
                LArgMain()
                << EAM(colorIn, "color", true, "[R,B,G] - default = [0,255,0]")
                << EAM(mulFactor, "mF", true, "adjust length of vector - default = 0.05")
            );

    if (MMVisualMode) return EXIT_SUCCESS;
    if (colorIn.size() > 0)
    {
        vector<double> a = parse_dParam(colorIn);
        color.x = a[0]; color.y = a[1]; color.z = a[2];
    }
    vector<Pt3dr> centre_ge;
    vector<Pt3dr> vecNormal;
    bool Exist= ELISE_fp::exist_file(aMeshIn);
    if (Exist)
    {
        InitOutil * aChain = new InitOutil(aMeshIn);
        vector<triangle*> aListTri = aChain->getmPtrListTri();
        for(uint i=0; i<aListTri.size(); i++)
        {
            triangle * atri = aListTri[i];
            Pt3dr centre_geo;
            Pt3dr aVecNor = atri->CalVecNormal(centre_geo, mulFactor);
            centre_ge.push_back(centre_geo);
            vecNormal.push_back(aVecNor);
        }
        DrawOnMesh aDraw(aChain);
        string nameOut = aMeshIn + "_VecNormal.ply";

        string aPlyOutDir;
        aPlyOutDir="./PlyVerify/";
        if(!(ELISE_fp::IsDirectory(aPlyOutDir)))
            ELISE_fp::MkDir(aPlyOutDir);
        nameOut =  aPlyOutDir + nameOut;

        aDraw.drawListPtsOnPly(centre_ge, nameOut, Pt3dr(0,255,0));
        nameOut = aPlyOutDir + aMeshIn + "_VecNormal2.ply";
        aDraw.drawListPtsOnPly(vecNormal, nameOut, Pt3dr(0,255,255));
        nameOut = aPlyOutDir + aMeshIn + "_Vec.ply";
        aDraw.drawEdge_p1p2(centre_ge, vecNormal, nameOut, Pt3dr(255,0,0) , Pt3dr(0,255,0));
    }
    else
        cout<<"Mesh not existed ! "<<aMeshIn<<endl;
    return EXIT_SUCCESS;
}



