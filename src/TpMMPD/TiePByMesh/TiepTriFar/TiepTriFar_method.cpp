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

#include "tieptrifar.h"

cAppliTiepTriFar::cAppliTiepTriFar (cParamTiepTriFar & aParam,
                                    cInterfChantierNameManipulateur * aICNM,
                                    vector<cImgTieTriFar> & vImg,
                                    string & aDir,
                                    string & aOri
                                    ):
    mParam (aParam),
    mvImg  (vImg),
    mDir   (aDir),
    mOri   (aOri),
    mICNM  (aICNM)
{}

void cAppliTiepTriFar::LoadMesh(string & aNameMesh)
{
    cout<<"Lire mesh...";
    cMesh myMesh(aNameMesh, true);
    const int nFaces = myMesh.getFacesNumber();
    for (int aKTri=0; aKTri<nFaces; aKTri++)
    {
        cTriangle* aTri = myMesh.getTriangle(aKTri);
        vector<Pt3dr> aSm;
        aTri->getVertexes(aSm);
        cTri3D aTri3D (   aSm[0],
                          aSm[1],
                          aSm[2],
                          aKTri
                      );
        mVTri3D.push_back(aTri3D);
    }
    cout<<"Finish"<<endl;
}


cImgTieTriFar::cImgTieTriFar(cAppliTiepTriFar &aAppli, string & aName):
    mAppli  (aAppli),
    mTif    (Tiff_Im::UnivConvStd(mAppli.Dir() + mNameIm)),
    mNameIm (aName),
    mMasqIm (1,1),
    mTMasqIm (mMasqIm),
    mCamGen  (mAppli.ICNM()->StdCamGenerikOfNames(mAppli.Ori(),aName)),
    mCamSten (mCamGen->DownCastCS())
{

}

// Peut on definir un mask par convex hull sur le set de reprojection de point 2D ?
void cAppliTiepTriFar::loadMask2D()
{
    for (uint aKTri=0; aKTri < mVTri3D.size(); aKTri++)
    {
        cTri3D aTri3D = mVTri3D[aKTri];
        for (uint aKImg=0; aKImg < mvImg.size(); aKImg++)
        {
             cImgTieTriFar aImg = mvImg[aKImg];
             cTri2D aTri2D = aTri3D.reprj(aImg.CamGen());
             aImg.SetVertices().push_back(aTri2D.P1());
             aImg.SetVertices().push_back(aTri2D.P2());
             aImg.SetVertices().push_back(aTri2D.P3());
        }
    }

    // compute convex hull for each set point 2D on image
    for (uint aKImg=0; aKImg < mvImg.size(); aKImg++)
    {
        cImgTieTriFar aImg = mvImg[aKImg];
        bool aOK = false;
        stack<Pt2dr> aStackP;
        aOK = convexHull(aImg.SetVertices(), aStackP);
        if (aOK)
        {
            while(!aStackP.empty())
            {
                Pt2dr p = aStackP.top();
                cout << "(" << p.x << ", " << p.y <<")" << endl;
                aStackP.pop();
            }
        }
        else
        {
            cout<<"Hull not possible"<<endl;
        }
    }
}

// ===================== CONVEX HULL COMPUTE ===========================//
Pt2dr p0;
Pt2dr nextToTop(stack<Pt2dr> &S)
{
    Pt2dr p = S.top();
    S.pop();
    Pt2dr res = S.top();
    S.push(p);
    return res;
}

int swap(Pt2dr &p1, Pt2dr &p2)
{
    Pt2dr temp = p1;
    p1 = p2;
    p2 = temp;

    return 1;
}

int distSq(Pt2dr p1, Pt2dr p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

int orientation(Pt2dr p, Pt2dr q, Pt2dr r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}


int compare(const void *vp1, const void *vp2)
{
   Pt2dr *p1 = (Pt2dr *)vp1;
   Pt2dr *p2 = (Pt2dr *)vp2;

   int o = orientation(p0, *p1, *p2);
   if (o == 0)
     return (distSq(p0, *p2) >= distSq(p0, *p1))? -1 : 1;

   return (o == 2)? -1: 1;
}


extern bool convexHull(vector<Pt2dr> points, stack<Pt2dr> S)
{
   int nPoints = points.size();
   int ymin = points[0].y, min = 0;
   for (int i = 1; i < nPoints; i++)
   {
     int y = points[i].y;
     if ((y < ymin) || (ymin == y &&
         points[i].x < points[min].x))
        ymin = points[i].y, min = i;
   }

   swap(points[0], points[min]);
   p0 = points[0];
   qsort(&points[1], nPoints-1, sizeof(Pt2dr), compare);
   int m = 1;
   for (int i=1; i<nPoints; i++)
   {
       while (i < nPoints-1 && orientation(p0, points[i],
                                           points[i+1]) == 0)
          i++;


       points[m] = points[i];
       m++;
   }
   if (m < 3) return false;

   S.push(points[0]);
   S.push(points[1]);
   S.push(points[2]);

   for (int i = 3; i < m; i++)
   {
      while (orientation(nextToTop(S), S.top(), points[i]) != 2)
         S.pop();
      S.push(points[i]);
   }
   while (!S.empty())
   {
        return true;
   }
   return false;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
aooter-MicMac-eLiSe-25/06/2007*/
