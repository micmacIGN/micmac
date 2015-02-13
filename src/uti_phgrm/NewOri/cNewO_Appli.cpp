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

           void TestTeta(double aTeta);
     private :

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

           // mDir1B mYB mZB est _|   mDir1B mYB m plan que mDir1B/mDir2B
           Pt3dr  mZB;
           Pt3dr  mYB;
 // Coordonnee de mDir3B dans mDir1B mYB mZB
           double mSc3X;
           double mSc3Y;
           double mSc3Z;

  // X12 Y1 est une base du plan |_ a aDirOr1A, X12 est aussi |_ a aDirOr2B,
  // X12 Y2 est une base du plan |_ a  aDirOr2B
           Pt3dr mX12;
           Pt3dr mY1;
           Pt3dr mZ1;
           Pt3dr mY2;
           double mCosY2;
           double mSinY2;
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
  mZB      (vunit(mDir1B^mDir2B)),
  mYB      (vunit(mZB^mDir1B)),
  mSc3X    (scal(mDir3B,mDir1B)),
  mSc3Y    (scal(mDir3B,mYB)),
  mSc3Z    (scal(mDir3B,mZB)),
  mX12     (vunit(mDirOr1A^mDirOr2A)),
  mY1      (vunit(mDirOr1A^mX12)),
  mZ1      (mX12 ^ mY1),
  mY2      (vunit(mDirOr2A^mX12)),
  mCosY2   (scal(mY2,mY1)),
  mSinY2   (scal(mY2,mZ1))
{
}

void cOriFromBundle::TestTeta(double aT1)
{
    // L'image de  mDir1B par la rot est dans le plan |_ a mBase et mDir1A donc
    // dans mX12 mY1
    double aC1 = cos(aT1);
    double aS1 = sin(aT1);

    Pt3dr aV1 = mX12 * aC1 + mY1 * aS1;
    std::cout << aV1 << "\n";

    // On V2 = mX12 cos(T2) + mY2 sin (T2) et V2.V1 = mSc12 par conservation
    //   V2 = mX12 C2 +   (mCosY2 mY1 + mSinY2 m Z1) S2
    //  V1.V2  = C1 C2 + S1 S2 mCosY2 = mSc12 = cos(Teta12)

    double  aSP1 = aS1 * mCosY2;

    double aNorm = sqrt(ElSquare(aC1) + ElSquare(aSP1));
    if ((aNorm !=0) && (ElAbs(mSc12<=aNorm)) )
    {

         double aA3 = atan2(aSP1/aNorm,aC1/aNorm);
         double aTeta12 = acos(mSc12/aNorm);

         //  V1.V2 /aNorm   = cos(T2-A3) =  Sc12/ Norm = cos(Teta12)    => T2 = A3 +/- Teta12

         for (int aS=-1 ; aS<=1 ; aS+=2)
         {
               double aT2  = aA3 + aS * aTeta12;
               double aC2 =  cos(aT2);
               double aS2 =  sin(aT2);
               Pt3dr aV2 =  mX12 * aC2 + mY2 * aS2;

               Pt3dr aZ = vunit(aV1^aV2);
               Pt3dr aY = vunit(aZ^aV1);

               Pt3dr aV3 =  aV1*mSc3X + aY*mSc3Y + aZ*mSc3Z ;

               std::cout << "TEST " << scal(aV1,aV2) << " " << mSc12  << " T2 " << aT2 << "\n";
               std::cout << "     " << scal(aV1,aV3) << " " << scal(mDir1B,mDir3B)   << "\n";
               std::cout << "     " << scal(aV2,aV3) << " " << scal(mDir2B,mDir3B)   << "\n";
         }
     }
     else
     {
         std::cout << "IMPOSSIBLE\n";
     }
}



Pt3dr P3dRand()
{
   return Pt3dr(NRrandom3(),NRrandom3(),NRrandom3());
}

void TestOriBundle()
{
      cOriFromBundle anOFB(P3dRand(),P3dRand(),P3dRand(),P3dRand(),P3dRand(),P3dRand(),P3dRand());

      int aNB=100;
      for (int aK=0 ; aK <aNB ; aK++)
      {
           anOFB.TestTeta((2*PI*aK)/aNB);
      }
/*
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
*/
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
