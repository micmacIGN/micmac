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
#include "../Detector.h"


cParamTiepTriFar::cParamTiepTriFar():
    aDisp (false),
    aZoom (0.2),
    aDispVertices (false),
    aRad (3)
{}

cAppliTiepTriFar::cAppliTiepTriFar (cParamTiepTriFar & aParam,
                                    cInterfChantierNameManipulateur * aICNM,
                                    vector<string> & vNameImg,
                                    string & aDir,
                                    string & aOri
                                    ):
    mParam (aParam),
    mVNameImg  (vNameImg),
    mDir   (aDir),
    mOri   (aOri),
    mICNM  (aICNM)
{
    // create image
    cout<<"In constructor cAppliTiepTriFar..creat "<<mVNameImg.size()<<" img...";
    for (uint aKImg=0; aKImg<mVNameImg.size(); aKImg++)
    {
        cImgTieTriFar * aImg = new cImgTieTriFar(*this, mVNameImg[aKImg]);
        mvImg.push_back(aImg);
    }
    cout<<"done!"<<endl;
}

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
    mNameIm (aName),
    mTif    (Tiff_Im::UnivConvStd(mAppli.Dir() + mNameIm)),
    mMasqIm (1,1),
    mTMasqIm (mMasqIm),
    mCamGen  (mAppli.ICNM()->StdCamGenerikOfNames(mAppli.Ori(),aName)),
    mCamSten (mCamGen->DownCastCS()),
    mVW      (0),
    mImInit  (1,1),
    mTImInit (mImInit),
    mTifZBuf (Tiff_Im::UnivConvStd(aAppli.Param().aDirZBuf + "/" + mNameIm + "/" + mNameIm + "_ZBuffer_DeZoom1.tif"))
{
}

template <typename T> bool cImgTieTriFar::IsInside(Pt2d<T> p, Pt2d<T> aRab)
{
    return
             (p.x >= ( (T)0.0 + aRab.x) )
        &&   (p.y >= ( (T)0.0 + aRab.y) )
        &&   (p.x <  ( (T)mTif.sz().x - aRab.x) )
        &&   (p.y <  ( (T)mTif.sz().y - aRab.x) );
}

int cImgTieTriFar::DetectInterestPts()
{
    ExtremePoint * aDetector = new ExtremePoint(mAppli.Param().aRad);
    int aNbPts = aDetector->template detect<double, double>(mTImInit, mInterestPt, mTMasqIm);
    delete aDetector;
    return aNbPts;
}

// Peut on definir un mask par convex hull sur le set de reprojection de point 2D ?
void cAppliTiepTriFar::loadMask2D()
{
    cout<<"Creat Mask 2D..."<<endl;
    Pt2dr aRab(10.0,10.0);
    for (uint aKTri=0; aKTri < mVTri3D.size(); aKTri++)
    {
        cTri3D aTri3D = mVTri3D[aKTri];
        for (uint aKImg=0; aKImg < mvImg.size(); aKImg++)
        {
             cImgTieTriFar * aImg = mvImg[aKImg];
             bool proj_OK = false;
             cTri2D aTri2D = aTri3D.reprj(aImg->CamGen(), proj_OK);
             if (
                        proj_OK
                     && aImg->IsInside(aTri2D.P1(), aRab)
                     && aImg->IsInside(aTri2D.P2(), aRab)
                     && aImg->IsInside(aTri2D.P3(), aRab)
                )
             {

                 {
                     aImg->SetVertices().push_back(aTri2D.P1());
                     aImg->SetVertices().push_back(aTri2D.P2());
                     aImg->SetVertices().push_back(aTri2D.P3());
                 }
             }
        }
    }

    // compute convex hull for each set point 2D on image
    for (uint aKImg=0; aKImg < mvImg.size(); aKImg++)
    {
        cout<<" + Im "<<aKImg<<"...";
        cImgTieTriFar * aImg = mvImg[aKImg];
        bool aOK = false;
        stack<Pt2dr> aStackP;
        vector<Pt2dr> aBoder;
        aOK = convexHull(aImg->SetVertices(), aStackP);
        if (aOK)
        {
            while(!aStackP.empty())
            {
                Pt2dr p = aStackP.top();
                //cout << "(" << p.x << ", " << p.y <<")" << endl;
                aBoder.push_back(p);
                aStackP.pop();
            }
            if (aBoder.size() > 0)
                cout<<"Hull OK"<<endl;
            else
                cout<<"Hull null point"<<endl;
        }
        else
        {
            cout<<"Hull not possible"<<endl;
        }

        // Remplit l'image de masque avec les point qui sont dans le polygone du convex Hull
        aImg->MasqIm() =  Im2D_Bits<1>(aImg->Tif().sz().x,aImg->Tif().sz().y,0);
        aImg->TMasqIm() = TIm2DBits<1> (aImg->MasqIm());
        ElList<Pt2di>  aLTri;
        for (uint aKBord=0; aKBord<aBoder.size(); aKBord++)
        {
            aLTri = aLTri + round_ni(aBoder[aKBord]);
        }
        ELISE_COPY(polygone(aLTri),1, aImg->MasqIm().oclip());

        aImg->ImInit().Resize(aImg->Tif().sz());
        aImg->TImInit() =  TIm2D<double,double>(aImg->ImInit());
        ELISE_COPY( aImg->ImInit().all_pts(), aImg->Tif().in() , aImg->ImInit().out() ); //masq
        // detection interest pts
        cout<<"  + Detect in mask - nbPts : "<<aImg->DetectInterestPts()<<endl;

        if (aImg->VW() == 0 && Param().aDisp)
        {
             Video_Win * aVW = aImg->VW();
             aVW = Video_Win::PtrWStd(Pt2di(aImg->Tif().sz()*Param().aZoom),true,Pt2dr(Param().aZoom,Param().aZoom));
             aVW->set_sop(Elise_Set_Of_Palette::TheFullPalette());
             std::string aTitle = aImg->NameIm();
             aVW->set_title(aTitle.c_str());
             ELISE_COPY(aVW->all_pts(), aImg->Tif().in_proj(), aVW->ogray());

             cout<<"Disp Hull "<<endl;
             // display convexhull
             aVW->draw_poly(aBoder, aVW->pdisc()(P8COL::green),true);
             cout<<"Nb Vrtc "<<aImg->SetVertices().size()<<endl;
             double r_ptsDraw = ElMax(aImg->Tif().sz().x, aImg->Tif().sz().y)/1000;
             if (Param().aDispVertices)
             {
                 for (uint aKVtc=0; aKVtc<aImg->InterestPt().size(); aKVtc++)
                    {
                        aVW->draw_circle_loc( aImg->InterestPt()[aKVtc],
                                              r_ptsDraw,
                                              aVW->pdisc()(P8COL::red)
                                             );
                    }
             }

             if (aKImg == mvImg.size()-1)
                aVW->clik_in();
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


extern bool convexHull(vector<Pt2dr> points, stack<Pt2dr> & S)
{
    cout<<endl<<"Compute Hull..on "<<points.size()<<" pts.."<<endl;
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
       cout<<" Hull has "<<S.size()<<" pts"<<endl;
       return true;
   }
   cout<<" Hull has "<<S.size()<<" pts"<<endl;
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
