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

#include "NewOri.h"


/*
    Resoud l'equation :
       [aBase   aDirkA  Rot(aDirkB)] =0 ,  K in (1,2,3)
       (aBase ^aDirkA) . Rot(aDirkB) = 0
*/

class cOriFromBundle
{
      public :

           cOriFromBundle
           (
                 Pt3dr aBase,
                 Pt3dr aDir1A,
                 Pt3dr aDir2A,
                 Pt3dr aDir3A,
                 Pt3dr aDir1B,
                 Pt3dr aDir2B,
                 Pt3dr aDir3B
           );

     private :
           void TestTeta(double aTeta);

           Pt3dr mBase;
           Pt3dr mDir1A;
           Pt3dr mDir2A;
           Pt3dr mDir3A;
// Direction |_ a B et au mDirKA
           Pt3dr mDirOr1A;
           Pt3dr mDirOr2A;
           Pt3dr mDirOr3A;

           Pt3dr mDir1B;
           Pt3dr mDir2B;
           Pt3dr mDir3B;
           double mSc12;

  // X12 Y1 est un base du plan |_ a aDirOr1A, X12 est aussi |_ a aDirOr2B,
           Pt3dr mX12;
           Pt3dr mY1;
           Pt3dr mY2;
};

cOriFromBundle::cOriFromBundle
(
      Pt3dr aBase,
      Pt3dr aDir1A,
      Pt3dr aDir2A,
      Pt3dr aDir3A,
      Pt3dr aDir1B,
      Pt3dr aDir2B,
      Pt3dr aDir3B
) :
  mBase    (vunit(aBase)) ,
  mDir1A   (vunit(aDir1A)),
  mDir2A   (vunit(aDir2A)),
  mDir3A   (vunit(aDir3A)),
  mDirOr1A (vunit(mBase^mDir1A)),
  mDirOr2A (vunit(mBase^mDir2A)),
  mDirOr3A (vunit(mBase^mDir3A)),
  mDir1B   (vunit(aDir1B)),
  mDir2B   (vunit(aDir2B)),
  mDir3B   (vunit(aDir3B)),
  mSc12    (scal(mDir1B,mDir2B)),
  mX12     (vunit(mDirOr1A^mDirOr2A)),
  mY1      (vunit(mDirOr1A^mX12)),
  mY2      (vunit(mDirOr2A^mX12))
{
}

void cOriFromBundle::TestTeta(double aTeta)
{
    // L'image de  mDir1B
/*
*/
    double aC = cos(aTeta);
    double aS = sin(aTeta);

    Pt3dr aVA = mX12 * aC + mY1 * aS;
    std::cout << aVA << "\n";
     
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
