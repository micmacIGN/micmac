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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"

class cStatMNE
{
    public :

       cStatMNE() :
          mTot (0)
       {
            for (int aK=0 ; aK<mNbH; aK++)
                 mH[aK] = 0;
       }

       void Add(int aK)
       {
           mH[aK]++;
           mTot++;
       }
       void Show()
       {
            for (int aK=0 ; aK<mNbH; aK++)
                std::cout << "  ---- HISTO[" << aK << "]= " << (mH[aK]/mTot)*100.0 << "%\n";
       }
        

       static const int mNbH = 20;
       double mH[mNbH];
       double mTot;
};


bool PtIsToTest(const Pt2di & aP)
{
   return false;
   // return (aP==Pt2di(220,195)) || (aP==Pt2di(225,195));
}

/*
   Cas ou pour chaque imahge on a calcule des projection du nuage, il est donne
par NuagePredicteur
*/

void cAppliMICMAC::Correl_MNE_ZPredic
     (
            const Box2di & aBox,
            const cCorrel_Correl_MNE_ZPredic&  aParamCCP
     )
{
   static  cStatMNE aStat;

   // Buffer pour pointer sur l'ensmble des vignettes OK
   std::vector<double *> aVecVals(mNbIm);
   double ** aVVals = &(aVecVals[0]);

   double aSeuilDZ = aParamCCP.SeuilDZ();


   //  Au boulot !  on balaye le terrain
   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {
           Pt2dr aPt2C  = DequantPlani(anX,anY);

           int aZMin = mTabZMin[anY][anX];
           int aZMax = mTabZMax[anY][anX];

// bool ToTest = PtIsToTest(Pt2di(anX,anY));

           // est-on dans le masque des points terrains valide
           if ( IsInTer(anX,anY))
           {

               // Bornes du voisinage
               int aX0v = anX-mPtSzWFixe.x;
               int aX1v = anX+mPtSzWFixe.x;
               int aY0v = anY-mPtSzWFixe.y;
               int aY1v = anY+mPtSzWFixe.y;

               // on parcourt l'intervalle de Z compris dans la nappe au point courant
               for (int aZInt=aZMin ;  aZInt< aZMax ; aZInt++)
               {

                   // Pointera sur la derniere imagette OK
                   double ** aVVCur = aVVals;
                   // Statistique MICMAC
                   mNbPointsIsole++;

                   // On dequantifie le Z 
                   double aZReel  = DequantZ(aZInt); // anOrigineZ+ aZInt*aStepZ;

                   //  CALCUL DES IMAGES VISIBLES 
                   std::vector<cGPU_LoadedImGeom * > aSelLI;
                   Pt3dr aPt3C(aPt2C.x,aPt2C.y,aZReel);
                   for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                   {
                       cGPU_LoadedImGeom & aGLI = *(mVLI[aKIm]);
                       double aDZ =  aGLI.DzOverPredic(aPt3C);
                       // aGLI.SetDZ(aDZ);
                       if (aDZ  > aSeuilDZ)
                       {
                            aSelLI.push_back(&aGLI);
                       }
                   }

                   int aNbImOk = 0;
                   int aNbSelIm = (int)aSelLI.size();
                   aStat.Add(aNbSelIm);

                   // On balaye les images  pour lire les valeur et stocker, par image,
                   // un vecteur des valeurs voisine normalisees en moyenne et ecart type
                   for (int aKIm=0 ; aKIm<aNbSelIm ; aKIm++)
                   {
                       cGPU_LoadedImGeom & aGLI = *(aSelLI[aKIm]);
                       const cGeomImage * aGeom=aGLI.Geom();
                       float ** aDataIm =  aGLI.DataIm0();
       
                       // Pour empiler les valeurs
                       double * mValsIm = aGLI.Vals();
                       double * mCurVals = mValsIm;

                       // Pour stocker les moment d'ordre 1 et 2
                       double  aSV = 0;
                       double  aSVV = 0;
                       
                       // En cas de gestion parties cachees, un masque terrain 
                       // de visibilite a ete calcule par image
                       if (aGLI.IsVisible(anX,anY))
                       {
                           // memorise le fait que tout est OK pour le pixel et l'image consideres
                           bool IsOk = true;

                           // Balaye le voisinage
                           for (int aXVois=aX0v ; (aXVois<=aX1v)&&IsOk; aXVois++)
                           {
                               for (int aYVois= aY0v; (aYVois<=aY1v)&&IsOk; aYVois++)
                               {
                                   // On dequantifie la plani 
                                     Pt2dr aPTer  = DequantPlani(aXVois,aYVois);
                                   // On projette dans l'image 
                                     Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);

                                     if (aGLI.IsOk(aPIm.x,aPIm.y))
                                     {
                                        // On utilise l'interpolateur pour lire la valeur image
                                        double aVal =  mInterpolTabule.GetVal(aDataIm,aPIm);
                                        // On "push" la nouvelle valeur de l'image
                                        *(mCurVals++) = aVal;
                                        aSV += aVal;
                                        aSVV += QSquare(aVal) ;
                                        // mValsIm.push_back(mInterpolTabule.GetVal(aDataIm,aPIm));
                                        // *(mTopPts++) = aPIm;
                                     }
                                     else
                                     {
                                        // Si un  seul des voisin n'est pas lisible , on annule tout
                                        IsOk =false;
                                     }
                               }
                           }
                           if (IsOk)
                           {

                             // On normalise en moyenne et ecart type
                              aSV /= mNbPtsWFixe;
                              aSVV /= mNbPtsWFixe;
                              aSVV -=  QSquare(aSV) ;
                              if (aSVV >mAhEpsilon) // Test pour eviter / 0 et sqrt(<0) 
                              {
                                  *(aVVCur++) = mValsIm;
                                   aSVV = sqrt(aSVV);
                                   for (int aKV=0 ; aKV<mNbPtsWFixe; aKV++)
                                       mValsIm[aKV] = (mValsIm[aKV]-aSV)/aSVV;
                              }
                              else
                              {
                                  IsOk = false;
                              }
                           }
                           aNbImOk += IsOk;
                           aGLI.SetOK(IsOk);
                       }
                       else
                       {
                           aGLI.SetOK(false);
                       }
                   }

                   // Calcul "rapide"  de la multi-correlation en utilisant la formule
                   // de Huygens comme decrit en 3.5 de la Doc MicMac
                   if (aNbImOk>=2)
                   {
                      double anEC2 = 0;
                      // Pour chaque pixel
                      for (int aKV=0 ; aKV<mNbPtsWFixe; aKV++)
                      {
                          double aSV=0,aSVV=0;
                          // Pour chaque image, maj des stat 1 et 2
                          for (int aKIm=0 ; aKIm<aNbImOk ; aKIm++)
                          {
                                double aV = aVVals[aKIm][aKV];
                                aSV += aV;
                                aSVV += QSquare(aV);
                          }
                          // Additionner l'ecart type inter imagettes
                          anEC2 += (aSVV-QSquare(aSV)/aNbImOk);
                      }

                     // Normalisation pour le ramener a un equivalent de 1-Correl 
                     double aCost = anEC2 / (( aNbImOk-1) *mNbPtsWFixe);
                     // On envoie le resultat a l'optimiseur pour valoir  ce que de droit
                     mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aCost);
// if (Debug) std::cout << "Z " << aZInt << " Cost " << aCost << "\n";
                   }
                   else
                   {
// if (Debug) std::cout << "Z " << aZInt << " DEF " << aDefCost << "\n";
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
   // aStat.Show();

// std::cout << "DZ MOY " << (aSomDz/aNbDz) <<   " PropOK " <<   aNbPtsOk/aNbPts << "\n";
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
