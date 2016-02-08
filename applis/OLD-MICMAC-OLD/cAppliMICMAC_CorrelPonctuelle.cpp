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
#include "general/all.h"
#include "MICMAC.h"
#include "ext_stl/numeric.h"

namespace NS_ParamMICMAC
{


#define NbMaxIm 100

void cAppliMICMAC::StatResultat 
      (
            const Box2di & aBox,
            Im2DGen &      aPxRes,
            const cDoStatResult & aDSR
      )
{
std::cout << "HHHHhjjj " << aBox._p0 << aBox._p1 << "\n";
   bool aDoR2 = aDSR.DoRatio2Im();

   double Vals[NbMaxIm];
   std::vector<double>  aVRatio;

   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {
           // est-on dans le masque des points terrains valide
           if ( IsInTer(anX,anY))
           {
                int aZ = aPxRes.GetI(Pt2di(anX,anY));
                Pt2dr aPTer  = DequantPlani(anX,anY);

                double aZReel  = DequantZ(aZ); // anOrigineZ+ aZInt*aStepZ;
                int aNbImOk = 0;
                    
                for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                {
                       cGPU_LoadedImGeom & aGLI = *(mVLI[aKIm]);
                       const cGeomImage * aGeom=aGLI.Geom();
                       float ** aDataIm =  aGLI.DataIm();
                       
                       if (aGLI.IsVisible(anX,anY))
                       {
                           // On dequantifie la plani 
                           // On projette dans l'image 
                           Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);

                           if (aGLI.IsOk(aPIm.x,aPIm.y))
                           {
                                Vals[aNbImOk++] = mInterpolTabule.GetVal(aDataIm,aPIm);
                           }
                       }
                }

                if ((aNbImOk==2) && aDoR2)
                {
                     aVRatio.push_back(Vals[0] /Vals[1]);
                }
           }
       }
   }

   if (aDoR2)
   {
       std::sort(aVRatio.begin(),aVRatio.end());
       double Vals[5] = {1.0,10.0,50.0,90.0,99.0};
       for (int aK=0 ; aK<5 ; aK++)
           std::cout << "RatioI1I2["  << Vals[aK] << "%]=" << ValPercentile(aVRatio,Vals[aK]) << "\n";
   }
}
    

void cAppliMICMAC::DoCorrel2ImGeomImGen
     (
            const Box2di & aBox,
            double         aRatioI1I2,
            double         aPdsPonct,
            bool           AddCpleRad
     )
{
   
   aPdsPonct *= 2.0;
   ELISE_ASSERT
   (
        (ModeGeomIsIm1InvarPx(*this) && (mNbIm<=2)),
        "DoCorrelPonctuelle2ImGeomI  requires Geom Im1  Inv to Px"
   );

   double Vals[NbMaxIm];


   //  Au boulot !  on balaye le terrain
   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {

           int aZMin = mTabZMin[anY][anX];
           int aZMax = mTabZMax[anY][anX];

           Pt2dr aPTer  = DequantPlani(anX,anY);
           // est-on dans le masque des points terrains valide
           if ( IsInTer(anX,anY))
           {

               // on parcourt l'intervalle de Z compris dans la nappe au point courant
               for (int aZInt=aZMin ;  aZInt< aZMax ; aZInt++)
               {

                   // Pointera sur la derniere imagette OK
                   // Statistique MICMAC
                   mNbPointsIsole++;

                   // On dequantifie le Z 
                   double aZReel  = DequantZ(aZInt); // anOrigineZ+ aZInt*aStepZ;
                    

                   int aNbImOk = 0;

                   // On balaye les images  pour lire les valeur et stocker, par image,
                   // un vecteur des valeurs voisine normalisees en moyenne et ecart type
                   for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                   {
                       cGPU_LoadedImGeom & aGLI = *(mVLI[aKIm]);
                       const cGeomImage * aGeom=aGLI.Geom();
                       float ** aDataIm =  aGLI.DataIm();
       
                       
                       // En cas de gestion parties cachees, un masque terrain 
                       // de visibilite a ete calcule par image
                       if (aGLI.IsVisible(anX,anY))
                       {
                           // On dequantifie la plani 
                           // On projette dans l'image 
                           Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);

                           if (aGLI.IsOk(aPIm.x,aPIm.y))
                           {
                                Vals[aNbImOk++] = mInterpolTabule.GetVal(aDataIm,aPIm);
                           }
                       }
                   }

                   if (aNbImOk==2)
                   {
                     if (AddCpleRad)
                     {
                         mSurfOpt->Local_SetCpleRadiom(Pt2di(anX,anY),&aZInt,Vals[0],Vals[1]);

                     }
                     double aV0 = Vals[0];
                     double aV1 = Vals[1] * aRatioI1I2;
                     double aCost = aPdsPonct  * (ElAbs(aV1-aV0)/(aV1+aV0));
                     // On envoie le resultat a l'optimiseur pour valoir  ce que de droit
                     mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aCost);
                   }
                   else
                   {
                       // Si pas assez d'image, il faut quand meme remplir la case avec qq chose
                       mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,mAhDefCost);
                   }
               }
           }
           else
           {
               for (int aZInt=aZMin ; aZInt< aZMax ; aZInt++)
               {
                    mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,mAhDefCost);
               }
           }
       }
   }
}


void cAppliMICMAC::DoCorrelMultiFen
     (
            const Box2di & aBox,
            const cCorrel_MultiFen & aCMF
     )
{
   
   ELISE_ASSERT
   (
        (ModeGeomIsIm1InvarPx(*this) && (mNbIm<=2)),
        "DoCorrelPonctuelle2ImGeomI  requires Geom Im1  Inv to Px"
   );

   int  aSzVMax = aCMF.NbFen();



   cGPU_LoadedImGeom & aGL1 = *(mVLI[0]);
   const cGeomImage * aGeom1=aGL1.Geom();
   float ** aDataIm1 =  aGL1.DataIm();

   cGPU_LoadedImGeom & aGL2 = *(mVLI[mNbIm-1]);
   const cGeomImage * aGeom2=aGL2.Geom();
   float ** aDataIm2 =  aGL2.DataIm();

   //  Au boulot !  on balaye le terrain
   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {

           int aZMin = mTabZMin[anY][anX];
           int aZMax = mTabZMax[anY][anX];

           // est-on dans le masque des points terrains valide
           if ( IsInTer(anX,anY))
           {

               // on parcourt l'intervalle de Z compris dans la nappe au point courant
               for (int aZInt=aZMin ;  aZInt< aZMax ; aZInt++)
               {

                   double aZReel  = DequantZ(aZInt); // anOrigineZ+ aZInt*aStepZ;
                   double aCostMin = mAhDefCost;
                   for (int aSzV=1; aSzV<= aSzVMax ; aSzV++)
                   {
                       RMat_Inertie aMat;
                       for (int aXV= anX-aSzV ; aXV<=anX+aSzV ;  aXV++)
                       {
                           for (int aYV= anY-aSzV ; aYV<=anY+aSzV ;  aYV++)
                           {
                                if ( (aGL1.IsVisible(aXV,aYV)) && (aGL2.IsVisible(aXV,aYV)))
                                {
                                    Pt2dr aPTer  = DequantPlani(aXV,aYV);
                                    Pt2dr aPIm1  = aGeom1->CurObj2Im(aPTer,&aZReel);
                                    Pt2dr aPIm2  = aGeom2->CurObj2Im(aPTer,&aZReel);

                                    if (aGL1.IsOk(aPIm1.x,aPIm1.y) && aGL2.IsOk(aPIm2.x,aPIm2.y) )
                                    {
                                          aMat.add_pt_en_place
                                          (
                                              mInterpolTabule.GetVal(aDataIm1,aPIm1)  ,
                                              mInterpolTabule.GetVal(aDataIm2,aPIm2)  
                                          );
                                    }
                                }
                           }
                       }
                       if (aMat.s()> ElSquare(aSzV+1))
                       {
                            double aCost = 1-aMat.correlation();
                            ElSetMin(aCostMin,aCost);
                       }
                   }

                   mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aCostMin);
               }
           }
       }
   }
}

void cAppliMICMAC::DoCorrelRobusteNonCentree
     (
            const Box2di & aBox,
            const cCorrel_NC_Robuste & aCNR
     )
{
   
   int aSzV= 1;
   ELISE_ASSERT
   (
        (ModeGeomIsIm1InvarPx(*this) && (mNbIm<=2)),
        "DoCorrelPonctuelle2ImGeomI  requires Geom Im1  Inv to Px"
   );

   cGPU_LoadedImGeom & aGL1 = *(mVLI[0]);
   const cGeomImage * aGeom1=aGL1.Geom();
   float ** aDataIm1 =  aGL1.DataIm();

   cGPU_LoadedImGeom & aGL2 = *(mVLI[mNbIm-1]);
   const cGeomImage * aGeom2=aGL2.Geom();
   float ** aDataIm2 =  aGL2.DataIm();


   //  Au boulot !  on balaye le terrain
   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {

           int aZMin = mTabZMin[anY][anX];
           int aZMax = mTabZMax[anY][anX];

           // est-on dans le masque des points terrains valide
           if ( IsInTer(anX,anY))
           {

               // on parcourt l'intervalle de Z compris dans la nappe au point courant
               for (int aZInt=aZMin ;  aZInt< aZMax ; aZInt++)
               {

                   double aZReel  = DequantZ(aZInt); // anOrigineZ+ aZInt*aStepZ;
                   double aCostMin = mAhDefCost;


                   if ( (aGL1.IsVisible(anX,anY)) && (aGL2.IsVisible(anX,anY)))
                   {
                       Pt2dr aP0Ter  = DequantPlani(anX,anY);
                       Pt2dr aP0Im1  = aGeom1->CurObj2Im(aP0Ter,&aZReel);
                       Pt2dr aP0Im2  = aGeom2->CurObj2Im(aP0Ter,&aZReel);

                       if (aGL1.IsOk(aP0Im1.x,aP0Im1.y) && aGL2.IsOk(aP0Im2.x,aP0Im2.y) )
                       {

                           double aI01 =  mInterpolTabule.GetVal(aDataIm1,aP0Im1);
                           double aI02 =  mInterpolTabule.GetVal(aDataIm2,aP0Im2);

                           if (aI01 && aI02)
                           {
                              aCostMin = 0;

                              for (int aXV= anX-aSzV ; aXV<=anX+aSzV ;  aXV++)
                              {
                                   for (int aYV= anY-aSzV ; aYV<=anY+aSzV ;  aYV++)
                                   {
                                       Pt2dr aPTer  = DequantPlani(aXV,aYV);
                                       Pt2dr aPIm1  = aGeom1->CurObj2Im(aPTer,&aZReel);
                                       Pt2dr aPIm2  = aGeom2->CurObj2Im(aPTer,&aZReel);


                                       if (aGL1.IsOk(aPIm1.x,aPIm1.y) && aGL2.IsOk(aPIm2.x,aPIm2.y) )
                                       {
                                          double aI1 =  mInterpolTabule.GetVal(aDataIm1,aPIm1) / aI01;
                                          double aI2 =  mInterpolTabule.GetVal(aDataIm2,aPIm2) / aI02;
                                          if (aI1 || aI2)
                                          {
                                             double aRatio = (aI1 < aI2) ? (aI1/aI2) : (aI2/aI1);
                                             aRatio = 10*(1-aRatio);
                                             aCostMin+=  ElMin(1.0,aRatio);
                                          }
                                          else
                                          {
                                             aCostMin++;
                                          }
                                       }
                                       else
                                         aCostMin++;
                                   }
                              }

                              aCostMin /= ElSquare(1+2*aSzV );
                           }
                        }
                   }

                   mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aCostMin);
               }
           }
       }
   }
}



void cAppliMICMAC::DoCorrelPonctuelle2ImGeomI
     (
            const Box2di & aBox,
            const cCorrel_Ponctuel2ImGeomI & aCP2
     )
{
    DoCorrel2ImGeomImGen(aBox,aCP2.RatioI1I2().Val(),1.0,false);

}

void cAppliMICMAC::DoCorrelCroisee2ImGeomI
     (
            const Box2di & aBox,
            const cCorrel_PonctuelleCroisee & aCPC
     )
{
    DoCorrel2ImGeomImGen(aBox,aCPC.RatioI1I2().Val(),aCPC.PdsPonctuel(),true);
}

};




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
