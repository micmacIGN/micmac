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


  // Test de Greg

#include "StdAfx.h"

/*
void XXXXX(FILE * aF)
{
  int x;
  int TOTO;
  TOTO = fscanf(aF,"%d",&x);
}
*/



//  TEST MERCURIAL

ElAffin2D::ElAffin2D
(
     Pt2dr im00,  // partie affine
     Pt2dr im10,  // partie vecto
     Pt2dr im01
) :
    mI00 (im00),
    mI10 (im10),
    mI01 (im01)
{
}


ElAffin2D::ElAffin2D() :
    mI00 (0,0),
    mI10 (1,0),
    mI01 (0,1)
{
}

bool ElAffin2D::IsId() const
{
   return 
           (mI00==Pt2dr(0,0))
        && (mI10==Pt2dr(1,0))
        && (mI01==Pt2dr(0,1)) ;
}

ElAffin2D ElAffin2D::Id()
{
   return ElAffin2D();
}

ElAffin2D ElAffin2D::trans(Pt2dr aTr)
{
   return ElAffin2D(aTr,Pt2dr(1,0),Pt2dr(0,1));
}





ElAffin2D::ElAffin2D (const ElSimilitude & aSim) :
    mI00 (aSim(Pt2dr(0,0))),
    mI10 (aSim(Pt2dr(1,0)) -mI00),
    mI01 (aSim(Pt2dr(0,1)) -mI00)
{
}

ElAffin2D ElAffin2D::operator * (const ElAffin2D & sim2) const 
{
    return ElAffin2D
           (
              (*this)(sim2(Pt2dr(0,0))),
              IVect(sim2.IVect(Pt2dr(1,0))),
              IVect(sim2.IVect(Pt2dr(0,1)))
           );

}


ElAffin2D ElAffin2D::inv () const
{
    REAL delta = mI10 ^ mI01;

    Pt2dr  Inv10 = Pt2dr(mI01.y,-mI10.y) /delta;
    Pt2dr  Inv01 = Pt2dr(-mI01.x,mI10.x) /delta;

    return  ElAffin2D
            (
                 -(Inv10*mI00.x+Inv01*mI00.y),
                 Inv10,
                 Inv01
            );
}

ElAffin2D ElAffin2D::TransfoImCropAndSousEch(Pt2dr aTr,Pt2dr aResol,Pt2dr * aSzInOut)
{
   ElAffin2D aRes
             (
                   -Pt2dr(aTr.x/aResol.x,aTr.y/aResol.y),
                   Pt2dr(1.0/aResol.x,0.0),
                   Pt2dr(0.0,1.0/aResol.y)
             );

   if (aSzInOut)
   {
      Box2dr aBoxIn(aTr, aTr+*aSzInOut);
      Box2dr aBoxOut  = aBoxIn.BoxImage(aRes);

      *aSzInOut = aBoxOut.sz();
       aRes = trans(-aBoxOut._p0) * aRes;
   }

   return aRes;
}

ElAffin2D  ElAffin2D::TransfoImCropAndSousEch(Pt2dr aTr,double aResol,Pt2dr * aSzInOut)
{
   return TransfoImCropAndSousEch(aTr,Pt2dr(aResol,aResol),aSzInOut);
}


ElAffin2D  ElAffin2D::L2Fit(const  ElPackHomologue & aPack)
{
   ELISE_ASSERT(aPack.size()>=3,"Less than 3 point in ElAffin2D::L2Fit");

   static L2SysSurResol aSys(6);
   aSys.GSSR_Reset(false);


   //   C0 X1 + C1 Y1 +C2 =  X2     (C0 C1)  (X1)   C2
   //                               (     )  (  ) +
   //   C3 X1 + C4 Y1 +C5 =  Y2     (C3 C4)  (Y1)   C5

  double aCoeffX[6]={1,1,1,0,0,0};
  double aCoeffY[6]={0,0,0,1,1,1};


   for 
   (
        ElPackHomologue::const_iterator it=aPack.begin();
        it!=aPack.end();
        it++
   )
   {
       aCoeffX[0] = it->P1().x;
       aCoeffX[1] = it->P1().y;

       aCoeffY[3] = it->P1().x;
       aCoeffY[4] = it->P1().y;

       aSys.AddEquation(1,aCoeffX, it->P2().x);
       aSys.AddEquation(1,aCoeffY, it->P2().y);
   }

   Im1D_REAL8 aSol = aSys.Solve(0);
   double * aDS = aSol.data();

   Pt2dr aIm00(aDS[2],aDS[5]);
   Pt2dr aIm10(aDS[0],aDS[3]);
   Pt2dr aIm01(aDS[1],aDS[4]);


   ElAffin2D aRes(aIm00,aIm10,aIm01);

/*
   for 
   (
        ElPackHomologue::const_iterator it=aPack.begin();
        it!=aPack.end();
        it++
   )
   {
       std::cout << euclid(aRes(it->P1()),it->P2()) << it->P2()  << "\n";
   }
*/
   return aRes;
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
