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

void PolygoneClipBandeVert::ClipPts
     (
          ElFifo<Pt2dr> &Pcliped,
          REAL x0,
          REAL x1,
          const ElFifo<Pt2dr> &POrig
     )
{
    init(x0,x1);
    Pcliped.clear();
    for (INT k=0; k<POrig.nb() ; k++)
         AddSeg(Pcliped,POrig[k],POrig[k+1]);
}


REAL PolygoneClipBandeVert::SurClipPts
     (
        REAL x0,
        REAL x1,
        const ElFifo<Pt2dr> &POrig
     )
{
   mBufClip.set_circ(true);
   mBufClip.clear();
   ClipPts(mBufClip,x0,x1,POrig);
   return surf_or_poly(mBufClip);
}

REAL PolygoneClipBandeVert::SquareSurClipPts(REAL x0,REAL x1,Pt2dr p0,Pt2dr p1)
{
      PtsOfSquare(mBufSquare,p0,p1);
      return SurClipPts(x0,x1,mBufSquare);
}


PolygoneClipBandeVert::tPosBande PolygoneClipBandeVert::PosBande(const Pt2dr & pt) const
{
   if (pt.x<mX0) return PosGauche;
   if (pt.x<mX1) return PosMilieu;
   return PosDroite;
}                     





Pt2dr PolygoneClipBandeVert::InterDrVert(REAL x,const Pt2dr & p0,const Pt2dr & p1)
{
     return Pt2dr(x,p0.y+(x-p0.x)*((p1.y-p0.y)/(p1.x-p0.x)));
}
Pt2dr PolygoneClipBandeVert::IDVGauche(const Pt2dr & p0,const Pt2dr & p1) const
{
   return InterDrVert(mX0,p0,p1);
}
Pt2dr PolygoneClipBandeVert::IDVDroite(const Pt2dr & p0,const Pt2dr & p1) const
{
   return InterDrVert(mX1,p0,p1);
}                                   
                                      


void PolygoneClipBandeVert::AddSeg(ElFifo<Pt2dr> &pts,Pt2dr p0,Pt2dr p1) const
{
    switch (PosBande(p0))
    {
         case PosGauche:
         {
             switch(PosBande(p1))
             {
                  case PosGauche:  
                  break;

                  case PosMilieu: 
                       pts.pushlast(IDVGauche(p0,p1));
                  break;

                  case PosDroite:  
                       pts.pushlast(IDVGauche(p0,p1));
                       pts.pushlast(IDVDroite(p0,p1));
                  break;
             }
         }
         break;

         case PosMilieu:
         {
             pts.pushlast(p0);
             switch(PosBande(p1))
             {
                  case PosGauche:  
                       pts.pushlast(IDVGauche(p0,p1));
                  break;

                  case PosMilieu: 
                  break;

                  case PosDroite:  
                       pts.pushlast(IDVDroite(p0,p1));
                  break;
             }
         }
         break;

         case PosDroite:
         {
             switch(PosBande(p1))
             {
                  case PosGauche:  
                       pts.pushlast(IDVDroite(p0,p1));
                       pts.pushlast(IDVGauche(p0,p1));
                  break;

                  case PosMilieu: 
                       pts.pushlast(IDVDroite(p0,p1));
                  break;

                  case PosDroite:  
                  break;
             }
         }
         break;
    }
}

void PolygoneClipBandeVert::init(REAL x0,REAL x1)
{
    set_min_max(x0,x1);
    mX0 = x0;
    mX1 = x1;
}


void PolygoneClipBandeVert::Bench
     (
         Pt2dr p0,Pt2dr q0,
         REAL  x0,REAL  x1,REAL  step,
         PolygoneClipBandeVert & aPCBV
     )
{
      Pt2dr p1 = p0+q0;

      REAL surf_tot = 0;
      for (REAL x=x0; x<x1; x+=step)
      {
            REAL dsurf = aPCBV.SquareSurClipPts(x,x+step,p0,p1);
            surf_tot += dsurf;
      }


      REAL dif = ElAbs(surf_tot-square_euclid(q0));
     BENCH_ASSERT(dif<1e-5);
}


void PolygoneClipBandeVert::Bench()
{

     PolygoneClipBandeVert aPCBV;


     Bench(Pt2dr(0,0),Pt2dr(0,1),-5,5,1,aPCBV);
     Bench(Pt2dr(0,0),Pt2dr(1,0),-5,5,1,aPCBV);
     Bench(Pt2dr(0,0),Pt2dr(1,1),-5,5,1,aPCBV);
     Bench(Pt2dr(0,0),Pt2dr(1,1),-5,5,10,aPCBV);

     Bench(Pt2dr(0,0),Pt2dr(0,1),-5,5,2,aPCBV);
     Bench(Pt2dr(0,0),Pt2dr(1,0),-5,5,2,aPCBV);
     Bench(Pt2dr(0,0),Pt2dr(1,1),-5,5,2,aPCBV);

     for (INT k=0; k<2000; k++)
     {
        Pt2dr p0 = Pt2dr(NRrandom3()-0.5,NRrandom3()-0.5);
        REAL rho = NRrandom3()*2.0+0.1;
        Pt2dr q0 = Pt2dr::FromPolar(rho,NRrandom3()*20);
        REAL step = 0.4 +NRrandom3();

        Bench(p0,q0,-5,5,step,aPCBV);
     }
     
}





void ElHough::BenchPolygoneClipBandeVert()
{
    PolygoneClipBandeVert::Bench();
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
Footer-MicMac-eLiSe-25/06/2007*/
