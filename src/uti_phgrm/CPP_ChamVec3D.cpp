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

#define DEF_OFSET -12349876


static Fonc_Num FoncVec(std::vector<Im2D_REAL8> aRes)
{

   return Virgule
          (
              aRes[0].in_proj(),
              aRes[1].in_proj(),
              aRes[2].in_proj()
          );
}
static Output OutVec(std::vector<Im2D_REAL8> aRes)
{

   return Virgule
          (
              aRes[0].out(),
              aRes[1].out(),
              aRes[2].out()
          );
}

static Fonc_Num Grad(Im2D_REAL8 aRes)
{
    return  Polar_Def_Opun::polar(deriche(aRes.in_proj(),1),0);
}

static Fonc_Num Grad(std::vector<Im2D_REAL8> aRes)
{
     return Grad(aRes[0]) + Grad(aRes[1]) + Grad(aRes[2]);
}



int ChamVec3D_main(int argc,char ** argv)
{
   std::string aN3dA;
   std::string aN3dB;
   std::string aVecXB;
   std::string aVecYB;

   double aSeuilG = 5;
   int aSzW = 10;
   int aNbIter = 3;

   double aMulDep3d = 20.0;
   double aMulDep2d = 30.0;
   //std::string aVecXA;
   //std::string aVecYA;


   /*
        aN3dA et aN3dB sont des images de XYZ (codees en RVB) , provenant par ex de Nuage2Ply


       aVecXB aVecYB deplacement image-image inter dates

   */

    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(aN3dA,"First XYZ image name", eSAM_IsExistFile)
                   << EAMC(aN3dB,"Second XYZ image name", eSAM_IsExistFile)
                   << EAMC(aVecXB,"Name of X-mapping ref to second image", eSAM_IsExistFile)
                   << EAMC(aVecYB,"Name of Y-mapping ref to second image", eSAM_IsExistFile),

       LArgMain()   << EAM(aSeuilG,"SeuilG",true,"Threshold for gradient weighting")
                    

/*<< EAM(aVecXA,"XFirst",true,"Name of X-mapping ref to first image")
                    << EAM(aVecYA,"YFirst",true,"Name of X-mapping ref to first image")*/
    );

    if (!MMVisualMode)
    {

    // Charge les images 3D
         std::vector<Im2D_REAL8> aImP3dA;
         Im2D_REAL8::ReadAndPushTif(aImP3dA,Tiff_Im(aN3dA.c_str()));
         ELISE_ASSERT(aImP3dA.size()==3,"Bad size for nuage");

         std::vector<Im2D_REAL8> aImP3dB;
         Im2D_REAL8::ReadAndPushTif(aImP3dB,Tiff_Im(aN3dB.c_str()));
         ELISE_ASSERT(aImP3dB.size()==3,"Bad size for nuage");



    // Charge le deplacement image-image
         Im2D_REAL8 aDepX = Im2D_REAL8::FromFileStd(aVecXB);
         Im2D_REAL8 aDepY = Im2D_REAL8::FromFileStd(aVecYB);

         Pt2di aSz = aDepX.sz();

          // Alloue la place

         std::vector<Im2D_REAL8> aRes;
         std::vector<Im2D_REAL8> aImP3dBInGeomA;
         for (int aK=0 ; aK< 3 ; aK++)
         {
             aRes.push_back(Im2D_REAL8(aSz.x,aSz.y));
             aImP3dBInGeomA.push_back(Im2D_REAL8(aSz.x,aSz.y));
         }


        // Changement de geometrie
        ELISE_COPY
        (
             aRes[0].all_pts(),
             FoncVec(aImP3dB)[Virgule(FX+aDepX.in(),FY+aDepY.in())],
             // Virgule(aDepX.in(),aDepY.in(),0),
             OutVec(aImP3dBInGeomA)
        );


        Im2D_REAL8 aGrad(aSz.x,aSz.y);
        ELISE_COPY
        (
             aRes[0].all_pts(),
             Grad(aImP3dA) + Grad(aImP3dBInGeomA),
             // Virgule(aDepX.in(),aDepY.in(),0),
             aGrad.out()
        );


        Tiff_Im::Create8BFromFonc
        (
            "aGrad.tif",
             aSz,
             aGrad.in()
        );


         // Comme la diff est mal definie la ou il y a du gradient spatial, on fait un
         // filtrage moyenneur en deponderant les points de fort gradient

         Fonc_Num aDif =  FoncVec(aImP3dA) - FoncVec(aImP3dBInGeomA);
         // aDif = MedianBySort(aDif,2);
         Fonc_Num aPds = 1/(1+Square(aGrad.in_proj()/aSeuilG));
         for (int aK=0 ; aK < aNbIter; aK++)
         {
             aDif = rect_som(aDif*aPds,aSzW) /  rect_som(aPds,aSzW);
         }


         ELISE_COPY
         (
              aRes[0].all_pts(),
              aDif,
        // Virgule(aDepX.in(),aDepY.in(),0),
              OutVec(aRes) | Video_Win::WiewAv(aSz)
         );

/*
   ELISE_COPY
   (
        aRes[0].all_pts(),
        MedianBySort(MedianBySort(FoncVec(aRes),5),5),
        // Virgule(aDepX.in(),aDepY.in(),0),
        OutVec(aRes) | Video_Win::WiewAv(aSz)
   );
   double aMulDep3d = 20.0;
   double aMulDep2d = 30.0;
*/


      Tiff_Im::Create8BFromFonc
      (
          "NormDep3d.tif",
           aSz,
           aMulDep3d*sqrt(Square(aRes[0].in())+Square(aRes[1].in())+Square(aRes[2].in()))
      );

      Tiff_Im::Create8BFromFonc
      (
          "NormDep2d.tif",
           aSz,
           aMulDep2d*sqrt(Square(aRes[0].in())+Square(aRes[1].in()))
      );

      Tiff_Im::Create8BFromFonc
      (
          "DepZ.tif",
           aSz,
           128 + 20 * aRes[2].in()
      );

      Tiff_Im::CreateFromFonc
      (
          "Teta.tif",
           aSz,
           Polar_Def_Opun::polar(Virgule(aRes[0].in(),aRes[1].in()),0).v1(),
           GenIm::real4
      );


      L_Arg_Opt_Tiff anArgTif3d = Tiff_Im::Empty_ARG + Arg_Tiff(Tiff_Im::ANoStrip());
      for (int aK=0 ; aK<3 ; aK++)
      {
              const char * aXYZ[3]= {"X","Y","Z"};
              std::string aName =std::string(aXYZ[aK]) + "_Dep.tif";
              Tiff_Im aTifDep3d
              (
                 aName.c_str(),
                 aSz,
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero,
                 anArgTif3d
              );
              // ELISE_COPY(aTifDep3d.all_pts(),FoncVec(aRes),aTifDep3d.out());
              ELISE_COPY(aTifDep3d.all_pts(),aRes[aK].in(),aTifDep3d.out());
       }
    

       BanniereMM3D();

       return EXIT_SUCCESS;

    }
    else return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
