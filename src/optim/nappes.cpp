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


// TEST THG

/*********************************************/
/*                                           */
/*            cNappe2DGen                    */
/*                                           */
/*********************************************/

void cNappe2DGen::Resize
     (
          INT aSz,
	  Fonc_Num aY1,
	  Fonc_Num aY2
     )
{
    mSz = aSz;

    mIYMin.Resize(aSz);
    mDataYMin   =mIYMin.data();
      
    mIYMax.Resize(aSz);
    mDataYMax   =mIYMax.data();
	    
    mIOffset.Resize(aSz);
    mDataOffset =mIOffset.data();


    Symb_FNum aS1(aY1),aS2(aY2);
    Symb_FNum aYmin(Min(aS1,aS2)),aYmax(Max(aS1,aS2));

    ELISE_COPY
    (
       mIYMin.all_pts(),
       Virgule(aYmin,aYmax,aYmax-aYmin),
       Virgule(mIYMin.out(),mIYMax.out(),sigma(mNbObj))
    );

    mDataOffset[0] = 0;
    for (INT x = 1; x <aSz ; x++)
    {
       mDataOffset[x] =    mDataOffset[x-1] 
	                  + mDataYMax[x-1]-mDataYMin[x-1];
    }
 
    // On presoustrait ZMin a Offset, pour gagner
    // un peu de temps dans OffsetPt
    ELISE_COPY
    (
       mIOffset.all_pts(),
       (mIOffset.in()-mIYMin.in()),
       mIOffset.out()
    );
}


cNappe2DGen::cNappe2DGen
(
     INT aSz,
     Fonc_Num aY1,
     Fonc_Num aY2
)  :
  mIYMin      (aSz),
  mIYMax      (aSz),
  mIOffset    (aSz)
{
   Resize(aSz,aY1,aY2);
}

/*********************************************/
/*                                           */
/*            cNappe3DGen                    */
/*                                           */
/*********************************************/

void cNappe3DGen::Resize
     (
          Pt2di aSz,
	  Fonc_Num aZ1,
	  Fonc_Num aZ2
     )
{
    mSz = aSz;

    mIZMin.Resize(aSz);
    mLineZMin   =mIZMin.data_lin();
    mDataZMin   =mIZMin.data();
      
    mIZMax.Resize(aSz);
    mLineZMax   =mIZMax.data_lin();
    mDataZMax   =mIZMax.data();
	    
    mIOffset.Resize(aSz);
    mLineOffset =mIOffset.data_lin();
    mDataOffset =mIOffset.data();


    Symb_FNum aS1(aZ1),aS2(aZ2);
    Symb_FNum aZmin(Min(aS1,aS2)),aZmax(Max(aS1,aS2));

    ELISE_COPY
    (
       mIZMin.all_pts(),
       Virgule(aZmin,aZmax,aZmax-aZmin),
       Virgule(mIZMin.out(),mIZMax.out(),sigma(mNbObj))
    );

    mLineOffset[0] = 0;
    for (INT xy = 1; xy <aSz.x*aSz.y ; xy++)
    {
       mLineOffset[xy] =    mLineOffset[xy-1] 
	                  + mLineZMax[xy-1]-mLineZMin[xy-1];
    }
 
    // On presoustrait ZMin a Offset, pour gagner
    // un peu de temps dans OffsetPt
    ELISE_COPY
    (
       mIOffset.all_pts(),
       (mIOffset.in()-mIZMin.in()),
       mIOffset.out()
    );
}




cNappe3DGen::cNappe3DGen
(
     Pt2di aSz,
     Fonc_Num aZ1,
     Fonc_Num aZ2
)  :
  mIZMin      (aSz.x,aSz.y),
  mIZMax      (aSz.x,aSz.y),
  mIOffset    (aSz.x,aSz.y)
{
   Resize(aSz,aZ1,aZ2);
}

/*********************************************/
/*                                           */
/*             cTplNape3D<Type>              */
/*                                           */
/*********************************************/








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
