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


#include "im_tpl/elise_ex_oper_assoc_exter.h"

/*
namespace std
{
bool operator < (const Pt3di & aP1,const Pt3di & aP2);
};
*/

namespace NS_ParamMICMAC
{



template <class Type,class TBase> 
        Type ** ImDec
                            (
                               std::vector<Type *> & aV,
                               Im2D<Type,TBase> anIm, 
                               const Box2di & aBox, 
                               const Pt2di & aSzV
                            )
{
   aV.clear();
   Type ** aDIm = anIm.data();
   Pt2di aSzIm = anIm.sz();
   
   for (int aY = 0 ; aY<aSzIm.y  ; aY++)
   {
      aV.push_back(aDIm[aY] - aBox._p0.x+aSzV.x);
   }

   return &(aV[0])    - aBox._p0.y+aSzV.y;
   //return anIm.data();
}

cGPU_LoadedImGeom::cGPU_LoadedImGeom
(
    const cAppliMICMAC & anAppli,
    cPriseDeVue* aPDV,
    int aNbVals,
    const Box2di & aBox,
    const Pt2di  &aSzV
) :
    mAppli   (anAppli),
    mPDV     (aPDV),
    mLI      (&aPDV->LoadedIm()),
    mGeom    (&aPDV->Geom()),
    
    mSzV     (aSzV),
    mSzOrtho (aBox.sz()+ mSzV*2),

    mImOrtho (mSzOrtho.x,mSzOrtho.y),
    mDImOrtho (mImOrtho),
    mDOrtho   (ImDec(mVOrtho,mImOrtho,aBox,aSzV)),

    mImSomO  (mSzOrtho.x,mSzOrtho.y),
    mDImSomO (mImSomO),
    mDSomO   (ImDec(mVSomO,mImSomO,aBox,aSzV)),

    mImSomO2  (mSzOrtho.x,mSzOrtho.y),
    mDImSomO2 (mImSomO2),
    mDSomO2   (ImDec(mVSomO2,mImSomO2,aBox,aSzV)),

    mImSom12  (mSzOrtho.x,mSzOrtho.y),
    mDImSom12 (mImSom12),
    mDSom12   (ImDec(mVSom12,mImSom12,aBox,aSzV)),

    mImOK_Ortho (mSzOrtho.x,mSzOrtho.y),
    mDImOK_Ortho (mImOK_Ortho),
    mDOK_Ortho   (ImDec(mVImOK_Ortho,mImOK_Ortho,aBox,aSzV)),


    mNbVals    ((1+2*aSzV.x) * (1+2*aSzV.y)),

    
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

tGpuF **    cGPU_LoadedImGeom::DataOrtho()   {return mDOrtho;    }
U_INT1 **   cGPU_LoadedImGeom::DataOKOrtho() {return mDOK_Ortho; }
Im2D_U_INT1 cGPU_LoadedImGeom::ImOK_Ortho()  {return mImOK_Ortho;}

tImGpu  cGPU_LoadedImGeom::ImOrtho()  {return mImOrtho; }
tImGpu  cGPU_LoadedImGeom::ImSomO()   {return mImSomO; }
tImGpu  cGPU_LoadedImGeom::ImSomO2()  {return mImSomO2; }
tImGpu  cGPU_LoadedImGeom::ImSom12()  {return mImSom12; }


bool   cGPU_LoadedImGeom::InitValNorms(int anX,int anY)
{
   if (! mDOK_Ortho[anY][anX])
     return false;


   mMoy   = mDSomO[anY][anX] / mNbVals;
   //  double aDMoy = mEpsAddMoy + mMoy * mEpsMulMoy;
   double aDMoy = mAppli.DeltaMoy(mMoy);

   mSigma  = mDSomO2[anY][anX] / mNbVals - QSquare(mMoy) + QSquare(aDMoy);
   mMoy += aDMoy;


   if (mSigma < mAppli.AhEpsilon()) 
      return false;

   mSigma = sqrt(mSigma);

   return true;
}


bool   cGPU_LoadedImGeom::Correl(double & aCorrel,int anX,int anY,const  cGPU_LoadedImGeom & aGeoJ) const
{
   if (! mDOK_Ortho[anY][anX])
     return false;
    double aMI  =  mDSomO [anY][anX] /mNbVals;
    double aDmI = mAppli.DeltaMoy(aMI);
    double aMII =  mDSomO2[anY][anX] /mNbVals - ElSquare(aMI) + ElSquare(aDmI);
    if (aMII < mAppli.AhEpsilon()) 
       return false;

    double aMJ  =  aGeoJ.mDSomO [anY][anX] /mNbVals;
    double aDmJ = mAppli.DeltaMoy(aMJ);
    double aMJJ =  aGeoJ.mDSomO2[anY][anX] /mNbVals - ElSquare(aMJ) + ElSquare(aDmJ);
    if (aMJJ < mAppli.AhEpsilon()) 
       return false;

    double aMIJ =  mDSom12[anY][anX] /mNbVals - aMI * aMJ + aDmI*aDmJ;

    aCorrel = aMIJ / sqrt(aMII*aMJJ);
    return true;
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


void cAppliMICMAC::DoInitAdHoc(const Box2di & aBox,const Pt2di & aSzV)
{
   mX0Ter = aBox._p0.x;
   mX1Ter = aBox._p1.x;
   mY0Ter = aBox._p0.y;
   mY1Ter = aBox._p1.y;


   mCurSzV = aSzV;
   mDilX0Ter = mX0Ter  - aSzV.x;
   mDilY0Ter = mY0Ter  - aSzV.y;
   mDilX1Ter = mX1Ter  + aSzV.x;
   mDilY1Ter = mY1Ter  + aSzV.y;

   mCurSzDil = Pt2di(mDilX1Ter-mDilX0Ter, mDilY1Ter-mDilY0Ter);

   mImOkTerCur.Resize(mCurSzDil);
   mTImOkTerCur = TIm2D<U_INT1,INT> (mImOkTerCur);
   mDOkTer = ImDec(mVDOkTer,mImOkTerCur,aBox,aSzV);

   mImOkTerDil.Resize(mCurSzDil);
   mTImOkTerDil = TIm2D<U_INT1,INT> (mImOkTerDil);
   mDOkTerDil  = ImDec(mVDOkTerDil,mImOkTerDil,aBox,aSzV);

   Pt2di aSzAll1 = mAll1ImOkTerDil.sz();
   if ((aSzAll1.x < mCurSzDil.x ) || (aSzAll1.y<mCurSzDil.y))
   {
      mAll1ImOkTerDil = Im2D_U_INT1(mCurSzDil.x,mCurSzDil.y,1);
   }
   mAll1TImOkTerDil =  TIm2D<U_INT1,INT>(mAll1ImOkTerDil);
   mAll1DOkTerDil = ImDec(mAll1VDOkTerDil,mAll1ImOkTerDil,aBox,aSzV);
   


   mTabZMin = mLTer->GPULowLevel_ZMin();
   mTabZMax = mLTer->GPULowLevel_ZMax();

   mTabMasqTER = mLTer->GPULowLevel_MasqTer();

   mAhDefCost =  mStatGlob->CorrelToCout(mDefCorr);
   mAhEpsilon = EpsilonCorrelation().Val();

   mGeomDFPx.SetOriResolPlani(mOriPlani,mStepPlani);
   mOrigineZ = mGeomDFPx.OrigineAlti();
   mStepZ = mGeomDFPx.ResolutionAlti();

   mFirstZIsInit = false;

   // mVLI.clear();
   DeleteAndClear(mVLI);
   for
   (
       tCsteIterPDV itFI=mPDVBoxGlobAct.begin();
       itFI!=mPDVBoxGlobAct.end();
       itFI++
   )
   {
      mVLI.push_back(new cGPU_LoadedImGeom(*this,*itFI,mNbPtsWFixe,aBox,mPtSzWFixe));
   }
   mNbIm = mVLI.size();


   mZMinGlob = 1e7;
   mZMaxGlob = -1e7;
   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {
           ElSetMin(mZMinGlob,mTabZMin[anY][anX]);
           ElSetMax(mZMaxGlob,mTabZMax[anY][anX]);
       }
  }

  mGpuSzD = 0;
  if (mCurEtape->UseGeomDerivable())
  {
     mGpuSzD = mCurEtape->SzGeomDerivable();
     Pt2di aSzOrtho = aBox.sz() + aSzV * 2;
     Pt2di aSzTab =  Pt2di(3,3) + aSzOrtho/mGpuSzD;
     mGeoX.Resize(aSzTab);
     mGeoY.Resize(aSzTab);
     mTGeoX =   TIm2D<REAL4,REAL8>(mGeoX);
     mTGeoY =   TIm2D<REAL4,REAL8>(mGeoY);
  }

// std::cout << mGpuSzD << "\n";

}


double MAXDIST = 0.0;

bool  cAppliMICMAC::InitZ(int aZ,eModeInitZ aMode)
{
   mZIntCur =aZ;
   mZTerCur  = DequantZ(mZIntCur); 
   
   mImOkTerCur.raz();

   mX0UtiTer = mX1Ter + 1;
   mY0UtiTer = mY1Ter + 1;
   mX1UtiTer = mX0Ter;
   mY1UtiTer = mY0Ter;
   


   for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
   {
       for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
       {
           mDOkTer[anY][anX] =
                                   (mZIntCur >= mTabZMin[anY][anX])
                                && (mZIntCur <  mTabZMax[anY][anX])
                                && IsInTer(anX,anY)
                               ;


            if ( mDOkTer[anY][anX])
            {
                 ElSetMin(mX0UtiTer,anX);
                 ElSetMax(mX1UtiTer,anX);
                 ElSetMin(mY0UtiTer,anY);
                 ElSetMax(mY1UtiTer,anY);
            }
    
       }
   }

   mX1UtiTer ++;
   mY1UtiTer ++;

   if (mX0UtiTer >= mX1UtiTer)
      return false;

   int aKFirstIm = 0;
   U_INT1 ** aDOkIm0TerDil = mDOkTerDil;
   if (mGIm1IsInPax)
   {
       if (mFirstZIsInit)
       {
         aKFirstIm = 1;
       }
       else
       {
         mX0UtiTer = mX0Ter;
         mX1UtiTer = mX1Ter;
         mY0UtiTer = mY0Ter;
         mY1UtiTer = mY1Ter;
         aDOkIm0TerDil = mAll1DOkTerDil;
       }
   }

    mX0UtiDilTer = mX0UtiTer - mCurSzV.x;
    mY0UtiDilTer = mY0UtiTer - mCurSzV.y;
    mX1UtiDilTer = mX1UtiTer + mCurSzV.x;
    mY1UtiDilTer = mY1UtiTer + mCurSzV.y;

    mX0UtiLocIm = mX0UtiTer - mDilX0Ter;
    mX1UtiLocIm = mX1UtiTer - mDilX0Ter;
    mY0UtiLocIm = mY0UtiTer - mDilY0Ter;
    mY1UtiLocIm = mY1UtiTer - mDilY0Ter;

    mX0UtiDilLocIm = mX0UtiDilTer - mDilX0Ter;
    mX1UtiDilLocIm = mX1UtiDilTer - mDilX0Ter;
    mY0UtiDilLocIm = mY0UtiDilTer - mDilY0Ter;
    mY1UtiDilLocIm = mY1UtiDilTer - mDilY0Ter;


   Box2di aBoxUtiLocIm(Pt2di(mX0UtiLocIm,mY0UtiLocIm),Pt2di(mX1UtiLocIm,mY1UtiLocIm));
   Box2di aBoxUtiDilLocIm(Pt2di(mX0UtiDilLocIm,mY0UtiDilLocIm),Pt2di(mX1UtiDilLocIm,mY1UtiDilLocIm));
   

   Dilate(mImOkTerCur,mImOkTerDil,mCurSzV,aBoxUtiDilLocIm);


   cInterpolateurIm2D<float> * anInt = CurEtape()->InterpFloat();



   cGPU_LoadedImGeom * aGLI_00 =  mNbIm ? mVLI[0] : 0 ;
   if (aMode==eModeMom_12_2_22)
   {
      ELISE_ASSERT(aGLI_00!=0,"Incohe eModeMom_12_2_22 with no Im in cAppliMICMAC::InitZ");
   }

   for (int aKIm= aKFirstIm ; aKIm<mNbIm ; aKIm++)
   {
       cGPU_LoadedImGeom & aGLI = *(mVLI[aKIm]);
       const cGeomImage * aGeom=aGLI.Geom();
       float ** aDataIm =  aGLI.DataIm();
       tGpuF ** aDOrtho = aGLI.DataOrtho();
       U_INT1 ** aOkOr =  aGLI.DataOKOrtho();


       // Tabulation des projections image au pas de mGpuSzD
       if (mGpuSzD)
       {
          int aNbX = (mX1UtiDilTer-mX0UtiDilTer +mGpuSzD) / mGpuSzD;
          int aNbY = (mY1UtiDilTer-mY0UtiDilTer +mGpuSzD) / mGpuSzD;
          for (int aKX = 0; aKX <= aNbX ; aKX++)
          {
              for (int aKY = 0; aKY <= aNbY ; aKY++)
              {
                  Pt2dr aPTer  = DequantPlani(mX0UtiDilTer+aKX*mGpuSzD,mY0UtiDilTer+aKY*mGpuSzD);
                  Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&mZTerCur);
                  Pt2di anI(aKX,aKY);
                  mTGeoX.oset(anI,aPIm.x);
                  mTGeoY.oset(anI,aPIm.y);
              }
          }
       }

       U_INT1 ** aDLocOkTerDil = (aKIm==0) ? aDOkIm0TerDil : mDOkTerDil;


       double aStep = 1.0/ElMax(1,mGpuSzD); // Histoire de ne pas diviser par 0
       double anIndX = 0.0;
       for (int anX = mX0UtiDilTer ; anX <  mX1UtiDilTer ; anX++)
       {
           double anIndY = 0.0;
           for (int anY = mY0UtiDilTer ; anY < mY1UtiDilTer ; anY++)
           {
               aOkOr[anY][anX] = 0;
               aDOrtho[anY][anX] = 0.0;
               if (aDLocOkTerDil[anY][anX])
               {
                   Pt2dr aPIm;
                   if (mGpuSzD)
                   {
                      Pt2dr anInd(anIndX,anIndY);
                      aPIm = Pt2dr( mTGeoX.getr(anInd), mTGeoY.getr(anInd)) ;
                   }
                   else
                   {
                      Pt2dr aPTer  = DequantPlani(anX,anY);
                      aPIm = aGeom->CurObj2Im(aPTer,&mZTerCur);
                   }


                   if (aGLI.IsOk(aPIm.x,aPIm.y))
                   {
                      aDOrtho[anY][anX] = anInt->GetVal(aDataIm,aPIm);
                      aOkOr[anY][anX] =  1;
                   }
               }
               anIndY += aStep;

           }
           anIndX += aStep;
       }

       SelfErode(aGLI.ImOK_Ortho(), mCurSzV,aBoxUtiLocIm);
       if (    (aMode==eModeMom_2_22)
            || ((aKIm==0) &&  (aMode==eModeMom_12_2_22))
          )
       {
           MomOrdre2(aGLI.ImOrtho(),aGLI.ImSomO(),aGLI.ImSomO2(),mCurSzV,aBoxUtiLocIm);
       }
       else if (aMode==eModeMom_12_2_22) 
       {
// std::cout << "KIM " << aKIm << "\n";
           Mom12_22
           (
                aGLI_00->ImOrtho(),
                aGLI.ImOrtho(),
                aGLI.ImSom12(),
                aGLI.ImSomO(),
                aGLI.ImSomO2(),
                mCurSzV,
                aBoxUtiLocIm
           );
       }
   }

   mFirstZIsInit = true;

   return true;
}


void cAppliMICMAC::DoOneCorrelSym(int anX,int anY)
{
     double aCost = mAhDefCost;
     std::vector<cGPU_LoadedImGeom *> aCurVLI;
     for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
         cGPU_LoadedImGeom * aGLI = (mVLI[aKIm]);
         if (aGLI->InitValNorms(anX,anY))
         {
              aCurVLI.push_back(aGLI);
         }
     }
     int aNbImCur = aCurVLI.size();
     if (aNbImCur >= 2)
     {
         int aX0 = anX - mCurSzV.x;
         int aX1 = anX + mCurSzV.x;
         int aY0 = anY - mCurSzV.x;
         int aY1 = anY + mCurSzV.x;


         double anEC2 = 0;
         for (int aXV=aX0 ; aXV<=aX1 ; aXV++)
         {
            for (int aYV=aY0 ; aYV<=aY1 ; aYV++)
            {
               double aSV = 0;
               double aSVV = 0;
               for (int aKIm=0 ; aKIm<aNbImCur ; aKIm++)
               {
                   double aV = aCurVLI[aKIm]->ValNorm(aXV,aYV);
                   aSV += aV;
                   aSVV += QSquare(aV) ;
               }
               anEC2 += (aSVV-QSquare(aSV)/aNbImCur);
            }
         }

         aCost = anEC2 / ((aNbImCur -1) * mNbPtsWFixe);
         aCost =  mStatGlob->CorrelToCout(1-aCost);
     }
     mSurfOpt->SetCout(Pt2di(anX,anY),&mZIntCur,aCost);
}

double EcartNormalise(double aI1,double aI2)
{
   // X = I1/I2 
   if (aI1 < aI2)   // X < 1
      return aI1/aI2 -1;   // X -1

   return 1-aI2/aI1;  // 1 -1/X 

}

void cAppliMICMAC::DoOneCorrelIm1Maitre(int anX,int anY,const cMultiCorrelPonctuel * aCMP)
{
     int aNbOk = 0;
     double aSomCorrel = 0;
    
     if (mVLI[0]->OkOrtho(anX,anY))
     {
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
        {
          double aCor;
          if (mVLI[aKIm]->Correl(aCor,anX,anY,*(mVLI[0])))
          {
               aNbOk ++;
               aSomCorrel += aCor;
          }
        }
     }

     if (aCMP)
     {
         std::vector<INT1> aVNorm;
         if (mVLI[0]->OkOrtho(anX,anY))
         {
             tGpuF aV0 = mVLI[0]->ImOrtho(anX,anY);
             for (int aK=1 ; aK<mNbIm ; aK++)
             {
                 if (mVLI[aK]->OkOrtho(anX,anY))
                 {
                     double aVal = EcartNormalise(aV0,mVLI[aK]->ImOrtho(anX,anY));
                     aVNorm.push_back(AdaptCostPonct(round_ni(aVal*127)));

                 }
                 else
                 {
                     aVNorm.push_back(ValUndefCPONT);
                 }
             }
         }
         else
         {
             for (int aK=1 ; aK<mNbIm ; aK++)
             {
                 aVNorm.push_back(ValUndefCPONT);
             }
         }
         mSurfOpt->Local_VecInt1(Pt2di(anX,anY),&mZIntCur,aVNorm);
     }
     
     mSurfOpt->SetCout
     (
         Pt2di(anX,anY),
         &mZIntCur,
         aNbOk ? mStatGlob->CorrelToCout(aSomCorrel/aNbOk) : mAhDefCost
     );
}



void cAppliMICMAC::DoOneCorrelMaxIm1Maitre(int anX,int anY)
{
     if (mEBI) // Etiq Best Image
     {
        if (mNbIm>1)
        {
           for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
           {
               double aCor;
               bool Ok = mVLI[aKIm]->Correl(aCor,anX,anY,*(mVLI[0]));
               aCor  = Ok ?  mStatGlob->CorrelToCout(aCor) : mAhDefCost;
               mSurfOpt->SetCout ( Pt2di(anX,anY),&mZIntCur,aCor, aKIm-1);
           }
        }
        else
        {
           mSurfOpt->SetCout(Pt2di(anX,anY),&mZIntCur,mAhDefCost,0);
        }
     }
     else
     {
         double aMaxCorrel = -2;
    
         if (mVLI[0]->OkOrtho(anX,anY))
         {
            for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
            {
              double aCor;
              if (mVLI[aKIm]->Correl(aCor,anX,anY,*(mVLI[0])))
              {
                   ElSetMax(aMaxCorrel,aCor);
              }
            }
         }
     
         mSurfOpt->SetCout
         (
             Pt2di(anX,anY),
             &mZIntCur,
             (aMaxCorrel>-1) ? mStatGlob->CorrelToCout(aMaxCorrel) : mAhDefCost
         );
     }
}



void cAppliMICMAC::DoGPU_Correl
     (
            const Box2di & aBox,
            const cMultiCorrelPonctuel * aMCP
     )
{
   eModeInitZ aModeInitZ = eModeMom_2_22;
   eModeAggregCorr aModeAgr = mCurEtape->EtapeMEC().AggregCorr().Val();
      
   if (aMCP)
   {
     ELISE_ASSERT(aModeAgr==eAggregIm1Maitre,"MultiCorrelPonctuel requires eAggregIm1Maitre");
   }

   if (aModeAgr==eAggregSymetrique)
   {
   }
   //else if ((aModeAgr==eAggregIm1Maitre) || (aModeAgr==eAggregMaxIm1Maitre))
   else if (IsModeIm1Maitre(aModeAgr))
   {
       aModeInitZ = eModeMom_12_2_22;
   }
   else
   {
        ELISE_ASSERT(false,"Unsupported Mode Aggreg in cAppliMICMAC::DoGPU_Correl");
   }
   
   
   for (int aZ=mZMinGlob ; aZ<mZMaxGlob ; aZ++)
   {
       if (InitZ(aZ,aModeInitZ))
       {
          for (int anX = mX0UtiTer ; anX <  mX1UtiTer ; anX++)
          {
              for (int anY = mY0UtiTer ; anY < mY1UtiTer ; anY++)
              {

                  if (mDOkTer[anY][anX])
                  {
                     switch (aModeAgr)
                     {
                        case eAggregSymetrique :
                             DoOneCorrelSym(anX,anY);
                        break;

                        case eAggregIm1Maitre :
                             DoOneCorrelIm1Maitre(anX,anY,aMCP);
                        break;

                        case  eAggregMaxIm1Maitre :
                             DoOneCorrelMaxIm1Maitre(anX,anY);
                        break;

                        default :
                        break;
                     }
/*
*/
	          }
              }
          }
       }
   }


}


void cAppliMICMAC::DoGPU_Correl_Basik
     (
            const Box2di & aBox
     )
{


   //  Lecture des parametre d'environnement MicMac : nappes, images, quantification etc ...


   //   Masque des points terrains valides
   // U_INT1 **  aTabMasqTER = mLTer->GPULowLevel_MasqTer();

   //   Deux constantes : cout lorque la correlation ne peut etre calculee et
   //   ecart type minmal
   // double aAhDefCost =  mStatGlob->CorrelToCout(mDefCorr);
   // double anAhEpsilon = EpsilonCorrelation().Val();


   // Buffer pour pointer sur l'ensmble des vignettes OK
   std::vector<double *> aVecVals(mNbIm);
   double ** aVVals = &(aVecVals[0]);

   
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
                    

                   int aNbImOk = 0;

                   // On balaye les images  pour lire les valeur et stocker, par image,
                   // un vecteur des valeurs voisine normalisees en moyenne et ecart type
                   for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                   {
                       cGPU_LoadedImGeom & aGLI = *(mVLI[aKIm]);
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
                                     Pt2dr aPTer  = DequantPlani(aXVois,aYVois);
                                   // On projette dans l'image 
                                     Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);
/*
if (MPD_MM())
{
    static int aCpt; aCpt++;
    std::cout  << "Cpt " << aCpt << " " << aPIm << aPTer << " OK " <<  aGLI.IsOk(aPIm.x,aPIm.y) << "\n";
    getchar();
}
*/

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
                     aCost =  mStatGlob->CorrelToCout(1-aCost);
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
}

void cAppliMICMAC::DoCorrelAdHoc
     (
            const Box2di & aBox
     )
{



   if (mEBI)
   {
       ELISE_ASSERT
       (
            mCurEtape->EtapeMEC().AggregCorr().Val() == eAggregMaxIm1Maitre,
            "EtiqBestImage requires eAggregMaxIm1Maitre,"
       );
       /// ELISE_ASSERT(mNb_PDVBoxInterne>,);
   }




   // Pour eventuellement changer si existe algo qui impose une taille
   Pt2di aSzV = mPtSzWFixe;

   DoInitAdHoc(aBox,aSzV);

   const cTypeCAH & aTC  = mCorrelAdHoc->TypeCAH();

   if (aTC.GPU_Correl().IsInit())
   {
          DoGPU_Correl(aBox,(cMultiCorrelPonctuel*)0);
   }
   else if (aTC.GPU_CorrelBasik().IsInit())
   {
          DoGPU_Correl_Basik(aBox);
   }
   else if (aTC.Correl_Ponctuel2ImGeomI().IsInit())
   {
       DoCorrelPonctuelle2ImGeomI(aBox,aTC.Correl_Ponctuel2ImGeomI().Val());
   }
   else if (aTC.Correl_PonctuelleCroisee().IsInit())
   {
       DoCorrelCroisee2ImGeomI(aBox,aTC.Correl_PonctuelleCroisee().Val());
   }
   else if (aTC.Correl_MultiFen().IsInit())
   {
      DoCorrelMultiFen(aBox,aTC.Correl_MultiFen().Val());
   }
   else if (aTC.Correl_Correl_MNE_ZPredic().IsInit())
   {
      Correl_MNE_ZPredic(aBox,aTC.Correl_Correl_MNE_ZPredic().Val());
   }
   else if (aTC.Correl_NC_Robuste().IsInit())
   {
      DoCorrelRobusteNonCentree(aBox,aTC.Correl_NC_Robuste().Val());
   }
   else if (aTC.MultiCorrelPonctuel().IsInit())
   {
       DoGPU_Correl(aBox,(aTC.MultiCorrelPonctuel().PtrVal()));
   }
}


void cAppliMICMAC::GlobDoCorrelAdHoc
     (
            const Box2di & aBox
     )
{

  mEpsAddMoy  =  mCorrelAdHoc->EpsilonAddMoyenne().Val();
  mEpsMulMoy  =  mCorrelAdHoc->EpsilonMulMoyenne().Val();


  cDecoupageInterv2D aDecInterv =
                        cDecoupageInterv2D::SimpleDec
                        (
                            aBox.sz(),
                            mCorrelAdHoc->SzBlocAH().Val(),
                            0
                        );

   for (int aKBox=0 ; aKBox<aDecInterv.NbInterv() ; aKBox++)
   {
        DoCorrelAdHoc(aDecInterv.KthIntervOut(aKBox));
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
