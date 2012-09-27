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


#include "StdAfx.h"          
#include "hough_include.h"


/*
    Contient des algorithmes utiles pour Hough (et developpes
   dans ce cadre) mais dont la portee est a vocation plus generale.
*/

GenMaxLoc::GenMaxLoc() :
  mLastVois (0,0),
  mMarq     (1,1),
  mDataMarq (mMarq.data())
{
}

U_INT1 & GenMaxLoc::Marq(Pt2di pt)  
{
   return mDataMarq[pt.y+mLastVois.y][pt.x+mLastVois.x];
}

void GenMaxLoc::MajOvois()
{
    mOVois.clear();
    INT dx = 2* mLastVois.x;
    INT dy = 2* mLastVois.y;

    bool cont = true;
    while (cont)
    {
         cont =  (dx >1) || (dy > 1);
         for (INT y=-mLastVois.y; y<=mLastVois.y ; y+= dy)
         {
             for (INT x=-mLastVois.x; x<=mLastVois.x ; x+= dx)
             {
                 Pt2di p(x,y);
                 if (Marq(p)==0)
                 {
                     Marq(p)=1;
                     INT Favorise = (y<0) || ((y==0) &&(x<0));
                     mOVois.push_back(Pt3di(x,y,Favorise));
                 }
             }
         }
         dx = ElMax(dx/2,1);
         dy = ElMax(dy/2,1);
    }
}

std::vector<Pt3di> &  GenMaxLoc::OrdVois(Pt2di vois)
{
   if (vois !=  mLastVois)
   {
       if ((vois.x >mLastVois.x) || (vois.y>mLastVois.y))
       {
           mMarq = Im2D_U_INT1(2*vois.x+1,2*vois.y+1,0);
           mDataMarq = mMarq.data();
       }
       mLastVois = vois;
       MajOvois();
   }
   return mOVois;
}

static const INT DefXY = -1<<30;



template <class Type,class TypeBase,class Compare>
          bool CalcMaxLoc<Type,TypeBase,Compare>::CmpTot(Type v1,Type v2,Pt2di p1,Pt2di p2)
{
    if (mCmp(v1,v2)) return true;
    if (mCmp(v2,v1)) return false;

    if (p1.y>p2.y)   return true;
    if (p2.y>p1.y)   return false;

    if (p1.x>p2.x)   return true;
    if (p2.x>p1.x)   return false;

    return false;
}

template <class Type,class TypeBase,class Compare>
        void CalcMaxLoc<Type,TypeBase,Compare>::AllMaxLoc 
             (
                  ElSTDNS vector<Pt2di> &res,
                  Im2D<Type,TypeBase> Im,
                  Pt2di    Vois,
                  Pt2di    p0, Pt2di    p1,
                  TypeBase vMin
             )
{
   Pt2di Sz(Im.tx(),Im.ty());
   Vois.x = ElAbs(Vois.x);
   Vois.y = ElAbs(Vois.y);
   pt_set_min_max(p0,p1);

   p0 = Sup(p0,Vois);
   p1 = Inf(p1,Sz-Vois);

   if ((p1.x<=p0.x) || (p1.y <= p0.y))
      return;

   INT dlx = Vois.x;
   INT dly = Vois.y;
   Type ** data = Im.data();
 

   res.clear();
   ElSTDNS vector<Pt3di> & OVois = OrdVois(Vois);

   for (INT Y0=p0.y ; Y0<p1.y ; Y0+=dly)
   {
       INT Y1 = ElMin(p1.y,Y0+dly);
       for (INT X0=p0.x ; X0<p1.x ; X0+=dlx)
       {
           INT X1 = ElMin(p1.x,X0+dlx);
           Type vMax = vMin;
           INT xMax = DefXY;
           INT yMax = DefXY;

           for (INT y=Y0; y<Y1 ; y++)
           {
               for (INT x=X0; x<X1 ; x++)
               {
                   if (mCmp((TypeBase)vMax,(TypeBase)data[y][x]))
                   {
                       xMax = x;
                       yMax = y;
                       vMax = data[y][x];
                   }
               }
           }
           if (xMax != DefXY)
           {
              bool IsMax = true;
              for 
              ( 
                  std::vector<Pt3di> ::iterator itOV = OVois.begin();
                  IsMax &&(itOV != OVois.end());
                  itOV++
              )
              {
                   Type aVal = data[yMax+itOV->y][xMax+itOV->x];
                   if (itOV->z)
                   {
                      if (!mCmp((TypeBase)aVal,(TypeBase)vMax))
                          IsMax = false;
                   }
                   else
                   {
                      if (mCmp((TypeBase)vMax,(TypeBase)aVal))
                          IsMax = false;
                   }
              }
              if (IsMax)
              {
                 res.push_back(Pt2di(xMax,yMax));

              }
           }
       }
   }
}



template <class Type,class TypeBase,class Compare>
bool CalcMaxLoc<Type,TypeBase,Compare>::BandeConnectedVsup 
     (
            Pt2di p1,
            Pt2di p2,
            Im2D<Type,TypeBase> Im,
            Type  VInf,
            REAL Tol,
            Im2D_U_INT1 Marq   
     )
{
    if (p1==p2)
       return true;

    Pt2di Sz = Inf(Im.sz(),Marq.sz());

    Box2di Box(Pt2di(0,0), Sz);

    if (!(Box.contains(p1) && Box.contains(p2)))
       return false;

    U_INT1 ** dMarq = Marq.data();
    Type **     dIm = Im.data();

    mBufCC.clear();
    mBufCC.push_back(p1);
    dMarq[p1.y][p1.x] = 1;
  
    Pt2dr aPR1(p1.x,p1.y);
    Pt2dr aPR2(p2.x,p2.y);
    SegComp seg12(aPR1,aPR2);

    bool got = false;
    for (INT kBuf=0 ; (kBuf!=(INT)mBufCC.size()) && (! got) ; kBuf++)
    {
         Pt2di pBuf = mBufCC[ kBuf];
         for (INT kV=0 ; kV< 8; kV++)
         {
             Pt2di pVois = pBuf + TAB_8_NEIGH[kV];
             if (
                        Box.contains(pVois)
                   &&  (dMarq[pVois.y][pVois.x] == 0)
                   &&  (! mCmp(dIm[pVois.y][pVois.x],VInf))
                   &&  (seg12.BoxContains(Pt2dr(pVois),1.0,Tol))
                )
             {
                 if (pVois==p2) 
                    got = true;
                 dMarq[pVois.y][pVois.x] = 1;
                 mBufCC.push_back(pVois);
             }
         }
    }

	{
    for (INT kBuf=0 ; kBuf<(INT)mBufCC.size() ; kBuf++)
    {
        Pt2di pBuf = mBufCC[ kBuf];
        dMarq[pBuf.y][pBuf.x] = 0;
    }
	}

    return got;
}


static Pt2di  NoPts(-(1<<30),1<<30);

template <class Type,class TypeBase,class Compare>
void  CalcMaxLoc<Type,TypeBase,Compare>::FiltrMaxLoc_BCVS
      (
          ElSTDNS vector<Pt2di> & Pts,
          Im2D<Type,TypeBase> Im,
          REAL  FactInf,
          REAL  TolGeom,
          Pt2di SzVois,
          Im2D_U_INT1 Marq
      )
{
    Pt2di SzIm = Inf(Im.sz(),Marq.sz());
    Box2di BoxIm(Pt2di(0,0),SzIm);

    Marq.raz();
    mBufFiltr.clear();


    INT4 ** dIm = Im.data();

    for (INT kp1=0; kp1<(INT)Pts.size() ; kp1++)
    {
        ELISE_ASSERT
        (
           BoxIm.contains(Pts[kp1]),
           "GenMaxLoc::FiltrMaxLoc_BCVS"
        );
    }
    
	{
    for (INT kp1=0; kp1<(INT)Pts.size() ; kp1++)
    {
        Pt2di p1 = Pts[kp1];
        Box2di BoxP1(p1-SzVois,p1+SzVois);
        INT v1 = dIm[p1.y][p1.x];
        INT Vinf = round_up(v1*FactInf);
        bool Refut = false;
   

        for (INT kp2=0 ; (kp2<(INT)Pts.size())&&(!Refut)  ; kp2++)
        {
            Pt2di p2 = Pts[kp2];
            if (    (kp1!= kp2)
                 && CmpTot(v1,dIm[p2.y][p2.x],p1,p2)
                 && BoxP1.contains(p2)
                 && BandeConnectedVsup(p1,p2,Im,Vinf,TolGeom,Marq)
               )
               Refut = true;
        }
        if (! Refut)
           mBufFiltr.push_back(p1);
    }
	}
    Pts = mBufFiltr;
}


template class CalcMaxLoc <INT,INT, ElSTDNS less<INT> >;



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
Footer-MicMac-eLiSe-25/06/2007*/
