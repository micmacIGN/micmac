Version initiale, sauvegardee pendant que l'autre sert de test sur les pb de
"marches d'escalier"
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

namespace NS_ParamMICMAC
{


// pour extraire le kieme bit d'un tableau de bits compactes 
// selon les conventions MicMac

static inline   bool GET_Val_BIT(const U_INT1 * aData,int anX)
{
    return (aData[anX/8] >> (7-anX %8) ) & 1;
}


template <class Type> static Type Square(const Type & aV) {return aV*aV;}


// ============================================================
// Classe pour  representer de maniere minimaliste une image
// (acces a ses valeurs et fonction de projection)
//
// Sert aussi pour stocker les info temporaires relatives a chaque
// image (par ex le vecteur des radiometrie dans le voisinage courant)
// ============================================================

class   cGPU_LoadedImGeom
{
   public :
       cGPU_LoadedImGeom(cPriseDeVue*,int aNbVals);

//  Est-ce que un point terrain est visible (si l'option des parties cachees
//  a ete activee)
       bool  IsVisible(int anX,int anY) const
       {
             return   (!mUsePC) || (mImPC[anY][anX] <mSeuilPC);
       }

// Est ce qu'un point image est dans le domaine de definition de l'image
// (dans le rectangle + dans le masque)

      bool IsOk(double aRX,double aRY)
      {
           int anIX = round_ni(aRX);
           int anIY = round_ni(aRY);

           
           return     (anIX>=0)
                   && (anIY>=0)
                   && (anIX<mSzX)
                   && (anIY<mSzY)
                   && (GET_Val_BIT(mImMasq[anIY],anIX));
            
      }

      cGeomImage * Geom() {return mGeom;}
      float ** DataIm()   {return mDataIm;}
      double * Vals()     {return &(mVals[0]);}
      void  SetOK(bool aIsOK) {mIsOK = aIsOK;}
      bool  IsOK() const {return mIsOK;}

   private :

       cPriseDeVue *    mPDV;
       cLoadedImage *   mLI;
       cGeomImage *     mGeom;
       
    //  tampon pour empiler les valeur de l'image sur un voisinage ("imagette")
        std::vector<double>  mVals;

    //  zone de donnee : "l'image" elle meme en fait

        float **         mDataIm;
    //  Masque Image (en geometrie image)
        int              mSzX;
        int              mSzY;
        U_INT1**         mImMasq;

   // Parties cachee :  masque image (en geom terrain), Seuil  et usage 
       U_INT1 **          mImPC;
       int                mSeuilPC;
       bool               mUsePC;
       bool               mIsOK;
};

cGPU_LoadedImGeom::cGPU_LoadedImGeom(cPriseDeVue* aPDV,int aNbVals) :
    mPDV     (aPDV),
    mLI      (&aPDV->LoadedIm()),
    mGeom    (&aPDV->Geom()),
    mVals    (aNbVals),
    mDataIm  (mLI->DataFloatIm()),
    mSzX     (mLI->SzIm().x),
    mSzY     (mLI->SzIm().y),
    mImMasq  (mLI->DataMasqIm()),
    mImPC    (mLI->DataImPC()),
    mSeuilPC (mLI->SeuilPC()),
    mUsePC   (mLI->UsePC())
{
    ELISE_ASSERT
    (
       aPDV->NumEquiv()==0,
       "Ne gere pas les classe d'equiv image en GPU"
    );
}

//
//    Fonction de correlation preparant une version GPU. Pour l'instant on se
//    reduit a la version qui fonctionne pixel par pixel (sans redressement global),
//    de toute facon il faudra l'ecrire et elle est plus simple. 
//
//    Une fois les parametres d'environnement decode et traduits en donnees
//    de bas niveau  ( des tableau bi-dim  de valeur numerique : entier, flottant et bits)
//    les communications, dans le corps de la boucle, avec l'environnement MicMac sont reduites
//    a trois appels :
//
//       [1]   Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);
//
//             Appelle la fonction virtuelle de projection associee a chaque
//             descripteur de geometrie de l'image.
//
//       [2]    mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aDefCost);
//
//             Appelle la fonction virtuelle de remplissage de cout
//             de l'optimiseur actuellement utilise
//
//
//       [3]    double aVal =  mInterpolTabule.GetVal(aDataIm,aPIm);
//
//               Utilise l'interpolateur courant. Pour l'instant l'interpolateur
//               est en dur quand on fonctionne en GPU
//


void cAppliMICMAC::GPU_Correl
     (
            const Box2di & aBox
     )
{
   //  Lecture des parametre d'environnement MicMac : nappes, images, quantification etc ...

   //   Boite Terrain sur laquelle on va effectuer une portion de calcul
   int aX0Ter = aBox._p0.x;
   int aX1Ter = aBox._p1.x;
   int aY0Ter = aBox._p0.y;
   int aY1Ter = aBox._p1.y;

   //   Nappe englobante
   INT2 ** aTabZMin = mLTer->GPULowLevel_ZMin();
   INT2 ** aTabZMax = mLTer->GPULowLevel_ZMax();

   //   Masque des points terrains valides
   U_INT1 **  aTabMasqTER = mLTer->GPULowLevel_MasqTer();

   //   Deux constantes : cout lorque la correlation ne peut etre calculee et
   //   ecart type minmal
   double aDefCost =  mStatGlob->CorrelToCout(mDefCorr);
   double anEpsilon = EpsilonCorrelation().Val();

   //   Parametre de quantification
   double anOrigineZ = mGeomDFPx.OrigineAlti();
   double aStepZ = mGeomDFPx.ResolutionAlti();
   Pt2dr aOriPlani,aStepPlani;
   mGeomDFPx.SetOriResolPlani(aOriPlani,aStepPlani);

   //   Lecture des parametres representant les images
   std::vector<cGPU_LoadedImGeom>  aVLI;
   for
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin();
       itFI!=mPDVBoxInterneAct.end();
       itFI++
   )
   {
      aVLI.push_back(cGPU_LoadedImGeom(*itFI,mNbPtsWFixe));
   }
   int aNbIm = aVLI.size();

   // Buffer pour pointer sur l'ensmble des vignettes OK
   std::vector<double *> aVecVals(aNbIm);
   double ** aVVals = &(aVecVals[0]);

   
   //  Au boulot !  on balaye le terrain
   for (int anX = aX0Ter ; anX <  aX1Ter ; anX++)
   {
       for (int anY = aY0Ter ; anY < aY1Ter ; anY++)
       {
           int aZMin = aTabZMin[anY][anX];
           int aZMax = aTabZMax[anY][anX];

           // est-on dans le masque des points terrains valide
           if ( GET_Val_BIT(aTabMasqTER[anY],anX))
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
                   double aZReel  = anOrigineZ+ aZInt*aStepZ;
                    

                   int aNbImOk = 0;

                   // On balaye les images  pour lire les valeur et stocker, par image,
                   // un vecteur des valeurs voisine normalisees en moyenne et ecart type
                   for (int aKIm=0 ; aKIm<aNbIm ; aKIm++)
                   {
                       cGPU_LoadedImGeom & aGLI = aVLI[aKIm];
                       const cGeomImage * aGeom=aGLI.Geom();
                       float ** aDataIm =  aGLI.DataIm();
       
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
                                    Pt2dr aPTer (
                                                   aOriPlani.x + aStepPlani.x*aXVois,
                                                   aOriPlani.y + aStepPlani.y*aYVois
                                               ); 
                                   // On projette dans l'image 
                                     Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);

                                     if (aGLI.IsOk(aPIm.x,aPIm.y))
                                     {
                                        // On utilise l'interpolateur pour lire la valeur image
                                        double aVal =  mInterpolTabule.GetVal(aDataIm,aPIm);
                                        // On "push" la nouvelle valeur de l'image
                                        *(mCurVals++) = aVal;
                                        aSV += aVal;
                                        aSVV += Square(aVal) ;
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
                              aSVV -=  Square(aSV) ;
                              if (aSVV >anEpsilon) // Test pour eviter / 0 et sqrt(<0) 
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
                                aSVV += Square(aV);
                          }
                          // Additionner l'ecart type inter imagettes
                          anEC2 += (aSVV-Square(aSV)/aNbImOk);
                      }

                     // Normalisation pour le ramener a un equivalent de 1-Correl 
                     double aCost = anEC2 / (( aNbImOk-1) *mNbPtsWFixe);
                     // On envoie le resultat a l'optimiseur pour valoir  ce que de droit
                     mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aCost);
                   }
                   else
                   {
                       // Si pas assez d'image, il faut quand meme remplir la case avec qq chose
                       mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aDefCost);
                   }
               }
           }
           else
           {
               for (int aZInt=aZMin ; aZInt< aZMax ; aZInt++)
               {
                    mSurfOpt->SetCout(Pt2di(anX,anY),&aZInt,aDefCost);
               }
           }
       }
   }
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
