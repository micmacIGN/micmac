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

/*************************************************************/
/*                                                           */
/*           EliseRle                                        */
/*                                                           */
/*************************************************************/

EliseRle::EliseRle
(
    INT x0,
    INT y,
    EliseRle::tIm ** im,
    EliseRle::tIm vTest,
    EliseRle::tIm vSet
)
{
    mY = y;
    mX1=mX0 = x0;
    tIm * l = im[y];
    while (l[mX0-1] == vTest) l[--mX0]=vSet;
    while (l[mX1] == vTest)   l[mX1++]=vSet;

}

void EliseRle::UpdateBox(Pt2di & p0,Pt2di & p1) const
{
    ElSetMin(p0.x,mX0);
    ElSetMin(p0.y,mY);

    ElSetMax(p1.x,mX1);
    ElSetMax(p1.y,mY+1);
}

void EliseRle::SetIm(tIm ** anIm,tIm vSet)
{
   tIm * l = anIm[mY];
   for (INT x=mX0; x<mX1 ; x++)
       l[x] = vSet;
}

                                          
void EliseRle::AddRleVois
     (
          EliseRle::tContainer & V,
          EliseRle::tIm ** im,
          EliseRle::tIm vTest,
          EliseRle::tIm vSet,
          bool   v8
     ) const
{

    INT x0 = v8 ? mX0-1 : mX0;
    INT x1 = v8 ? mX1+1 : mX1;
    for (INT x = x0; x<x1 ; x++)
    {
         if (im[mY-1][x] == vTest)
             V.push_back(EliseRle(x,mY-1,im,vTest,vSet));
         if (im[mY+1][x] == vTest)
             V.push_back(EliseRle(x,mY+1,im,vTest,vSet));
    }

}



Box2di  EliseRle::ConcIfInBox 
        ( 
            Pt2di  pInit,
            EliseRle::tContainer & V,
            EliseRle::tIm ** im,
            EliseRle::tIm vTest,
            EliseRle::tIm vSet,
            bool   v8    ,
            Pt2di  SzBox
        )
{
     Pt2di  pBox0 = pInit;
     Pt2di  pBox1 = pInit+Pt2di(1,1);
      
     V.clear();
     V.push_back(EliseRle(pInit.x,pInit.y,im,vTest,vSet));
     V[0].UpdateBox(pBox0,pBox1);

     INT k0 = 0;
     INT k1 = 1;

     while (
                 (k0 != k1) 
              && (pBox1.x-pBox0.x <= SzBox.x) 
              && (pBox1.y-pBox0.y <= SzBox.y)
           )
     {
          for (INT k=k0; k< k1 ; k++)
          {
               EliseRle rle = V[k]; 
               rle.AddRleVois(V,im,vTest,vSet,v8);
          }

          k0 = k1;
          k1 = V.size();
			
		  {
          for (INT k=k0; k<k1 ; k++)
              V[k].UpdateBox(pBox0,pBox1);
		  }
     }

     if (k0==k1)
        return Box2di(pBox0,pBox1);


    while ((INT) V.size() != (k1-k0))
         V.pop_front();
    
    while (! V.empty())
    {
      INT Sz0 = V.size();
      EliseRle rle = V.front();
      rle.AddRleVois(V,im,vTest,vSet,v8);
      for (INT k=Sz0; k<(INT) V.size() ; k++)
          V[k].UpdateBox(pBox0,pBox1);


      V.pop_front();
    }
    return Box2di(pBox0,pBox1);
}


void  EliseRle::SetIm(EliseRle::tContainer & Rles,EliseRle::tIm ** data,EliseRle::tIm vSet)
{
    for (INT k=0 ; k<(INT) Rles.size() ; k++)
        Rles[k].SetIm(data,vSet);
}


template <class Type> void RleDescOfBinaryImage
     (
          Im2D<Type,INT> anIm,
          EliseRle::tContainer & aVRle
     )
{
     Type ** aData = anIm.data();
     INT aTx = anIm.tx();
     INT aTy = anIm.ty();
     aVRle.clear();


     for (INT anY=0 ; anY<aTy ; anY++)
     {
          Type * aLine = aData[anY];


          for (INT anX=0 ; anX<aTx ;)
          {
              while ((aLine[anX]==0) && (anX<aTx)) anX++;
              INT anX0 = anX;
              while ((aLine[anX]!=0) && (anX<aTx)) anX++;
              if (anX0 != anX)
                 aVRle.push_back(EliseRle(anX0,anX,anY));
          }
     }

}

template void RleDescOfBinaryImage 
              (  
                 Im2D<INT,INT> anIm,
                 EliseRle::tContainer & aRle
              );


/*************************************************************/
/*                                                           */
/*           EliseSurfRle                                    */
/*                                                           */
/*************************************************************/











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
