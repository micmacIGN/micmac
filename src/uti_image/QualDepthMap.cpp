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
#include "QualDepthMap.h"

#define DEF_OFSET -12349876

/*
   (x,y) -> (x+p(x),y)

   (1+Gx  Gy)     (1/(1+Gx)   -Gy/(1+Gx))
   (0      1)     (0           1)
*/

#define MaxQualDM 1e9

double SquareQualGrad(double aGx,double aGy)
{
   return ElSquare(aGx) + ElSquare(aGy) ;
/*
   
   aGx  = ElMax(-0.66,ElMin(2.0,aGx));

   // if (aGx <=-0.66) return MaxQualDM;


   double InvX = 1/(1+aGx);

   return ElSquare(aGx) + ElSquare(aGy) * (1+ElSquare(InvX)) +ElSquare(1-InvX);
*/
}

Im2D_REAL4 ImageQualityGrad(Im2D_REAL4 aProfInit,Im2D_Bits<1> aMasq,Video_Win * aW,double aResol)
{
    Pt2di aSz = aProfInit.sz();
    Im2D_REAL4 aRes(aSz.x,aSz.y);
    Im2D_REAL4 aProf(aSz.x,aSz.y);
    ELISE_COPY(aProf.all_pts(),aProfInit.in(),aProf.out());


    TIm2D<REAL4,REAL8>   aTRes(aRes);
    TIm2D<REAL4,REAL8>   aTProf(aProf);
    TIm2DBits<1> aTM(aMasq);

    if (0) // (aW)
    {
       // ELISE_COPY(aProf.all_pts(),mod(aProf.in(),256),aW->ogray()); aW->clik_in();
       ELISE_COPY
       (
             aProf.all_pts(),
             Min(255,30.0*Polar_Def_Opun::polar(deriche(aProf.in_proj(),1.0),0).v0()),
             aW->ogray()
        ); 
        ELISE_COPY(select(aMasq.all_pts(),!aMasq.in()),P8COL::yellow,aW->odisc());
        aW->clik_in();
    }

    double aMulG = 1;
    MasqkedFilterGauss(aProf,aMasq,aResol,2);
    Pt2di aP;
    for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
    {
         for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
         {
              double aSomGr = 0;
              int aNbOk=0;
              if (aTM.get(aP))
              {
                  double aV0 = aTProf.get(aP);
                  for (int aSx=-1; aSx<=1 ; aSx++)
                  {
                      Pt2di aQx(aP.x+aSx,aP.y);
                      if (aTM.get(aQx,0))
                      {
                          double aGx = (aTProf.get(aQx)-aV0) * aSx  * aMulG ;
                          for (int aSy=-1; aSy<=1 ; aSy++)
                          {
                              Pt2di aQy(aP.x,aP.y+aSy);
                              if (aTM.get(aQy,0))
                              {
                                  double aGy = (aTProf.get(aQy)-aV0) * aSy * aMulG ;
                                  double aScore = SquareQualGrad(aGx,aGy);
                                   // aScore = aGx;
                                  aSomGr += aScore;
                                  aNbOk++;
                              }
                          }
                      }
                  }
              }
              if (!aNbOk) 
              {
                 aTRes.oset(aP,MaxQualDM);
              }
              else 
              {
                 aTRes.oset(aP,sqrt(aSomGr/aNbOk));
              }
         }
    }
    if (0)//  (aW)
    {
        ELISE_COPY
        (
            aW->all_pts(),
            // Min(255,aRes.in()*20),
            Max(0,Min(255,aRes.in()*200)),
            aW->ogray()
        );
        std::cout << "RESOL " <<  aResol << "\n";
        aW->clik_in();
    }

    
    return aRes;
}

Im2D_REAL4 ImageQualityGrad(Im2D_REAL4 aProf,Im2D_Bits<1> aMasq,Video_Win * aW)
{
   return ImageQualityGrad(aProf,aMasq,aW,4.0);
/*
   ImageQualityGrad(aProf,aMasq,aW,1.0);
   ImageQualityGrad(aProf,aMasq,aW,2.0);
   ImageQualityGrad(aProf,aMasq,aW,4.0);
   ImageQualityGrad(aProf,aMasq,aW,8.0);
   return ImageQualityGrad(aProf,aMasq,aW,1.0);
*/
}

Im2D_REAL4 Fine_ImageQualityGrad(Im2D_REAL4 aProf,Im2D_Bits<1> aMasq,Video_Win *)
{
     Pt2di aSz = aProf.sz();
     Im2D_REAL4 aRes(aSz.x,aSz.y);
     TIm2D<REAL4,REAL8>   aTRes(aRes);
     TIm2DBits<1> aTM(aMasq);
     TIm2D<REAL4,REAL8>   aTProf(aProf);
    
     Pt2di aP;
     for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
     {
         for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
         {
              double aMaxGr = 0;
              bool Ok=false;
              if (aTM.get(aP))
              {
                  double aV0 = aTProf.get(aP);
                  for (int aSx=-1; aSx<=1 ; aSx++)
                  {
                      Pt2di aQx(aP.x+aSx,aP.y);
                      if (aTM.get(aQx,0))
                      {
                          double aGx = (aTProf.get(aQx)-aV0) * aSx;
                          for (int aSy=-1; aSy<=1 ; aSy++)
                          {
                              Pt2di aQy(aP.x,aP.y+aSy);
                              if (aTM.get(aQy,0))
                              {
                                  double aGy = (aTProf.get(aQy)-aV0) * aSy;
                                  double aScore = SquareQualGrad(aGx,aGy);
                                  ElSetMax(aMaxGr,aScore);
                                  Ok = true;
                              }
                          }
                      }
                  }
              }
              if (!Ok) aMaxGr = MaxQualDM;
              aTRes.oset(aP,sqrt(aMaxGr/2.0));
         }
     }
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
