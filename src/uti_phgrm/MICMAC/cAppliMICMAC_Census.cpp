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


namespace NS_ParamMICMAC
{

class cQckInterpolEpip
{
    public :
        cQckInterpolEpip(float X) :
            mX0     (round_down(X)) ,
            mPds1   (X-mX0),
            mPds0   (1-mPds1)
        {
        } 

        int mX0;
        float mPds1;
        float mPds0;

         inline double  GetVal(float *aV)
         {
              return mPds0 * aV[0] + mPds1 * aV[1];
         }

         inline double  GetVal(float **aV, Pt2di aP)
         {
              float * aL = aV[aP.y] + aP.x + mX0;
              return mPds0 * aL[0] + mPds1 * aL[1];
         }
        
};


double CorrelBasic(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV,float anEpsilon)
{
     cQckInterpolEpip aQI2(X2);
     RMat_Inertie aMat;
     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x;
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
               aMat.add_pt_en_place(aL1[aDx],aQI2.GetVal(aL2+aDx));
          }
     }
     return aMat.correlation(anEpsilon);
}

double CensusBasic(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV,float anEpsilon)
{
     cQckInterpolEpip aQI2(X2);
     float aC1 =  Im1[aP1.y][aP1.x];
     float aC2 =  aQI2.GetVal(Im2[aP1.y]+ aQI2.mX0);
     int aNbOk = 0;


     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x;
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
              float aV1 = aL1[aDx];
              float aV2 = aQI2.GetVal(aL2+aDx);
              // if ((aDx==0) && (aDy==0)) std::cout << "TTTt " << (aC1-aV1) << " " << (aC2-aV2) << "\n";

              bool Inf1 = (aV1<aC1);
              bool Inf2 = (aV2<aC2);
              if (Inf1==Inf2) aNbOk++;
          }
     }
     return ((double) aNbOk) / ((1+2*aSzV.x)*(1+2*aSzV.y));
}


static int VX[4] = {1,1,0,-1};
static int VY[4] = {0,1,1,1};





double CensusGraphe(float ** Im1,Pt2di aP1,float ** Im2,float X2,int Y2,Pt2di aSzV,float anEpsilon)
{
     cQckInterpolEpip aQI2(X2);
     int aNbOk = 0;
     int aNbMiss = 0;


     for (int aDy=-aSzV.y ; aDy<=aSzV.y ; aDy++)
     {
          float * aL1 = Im1[aP1.y+aDy] + aP1.x;
          float * aL2 = Im2[Y2+aDy] + aQI2.mX0;
          for (int aDx=-aSzV.x ; aDx<= aSzV.x ; aDx++)
          {
              float aV1 = aL1[aDx];
              float aV2 = aQI2.GetVal(aL2+aDx);
              for (int aK=0 ; aK<4 ; aK++)
              {
                   int aDx2 = aDx+VX[aK];
                   int aDy2 = aDy+VY[aK];
                   if  ((aDx2>=-aSzV.x) && (aDx2<=aSzV.x) && (aDy2>=-aSzV.y) && (aDy2<=aSzV.y))
                   {
                       float aW1 = Im1[aP1.y+aDy2][aP1.x+aDx2];
                       float aW2 = aQI2.GetVal(Im2,Pt2di(aDx2,Y2+aDy2));
                       bool Inf1 = (aV1<aW1);
                       bool Inf2 = (aV2<aW2);
                       
                       if (Inf1==Inf2)
                          aNbOk++;
                       else          
                           aNbMiss++;
                   }
              }

          }
     }
     return ((double) aNbOk) / (aNbOk+aNbMiss);
}




/*
double CensusBasic(float ** Im1,Pt2di aP1,float ** Im2,int X2,double Y2,Pt2di aSzV)


double CensusBasic(float ** Im1,Pt2di aP1,float ** Im2,int X2,double Y2,Pt2di aSzV)
{
}
*/



double TolNbByPix=1e-5;
void cAppliMICMAC::DoCensusCorrel(const Box2di & aBox,const cCensusCost &)
{
 //  ====  VERIFICATION DYNAMIQUE DES ARGUMENTS ==========

 //  ====  1. GEOMETRIE EPIPOLAIRE BASIQUE
    ELISE_ASSERT
    (
         GeomImages() == eGeomImage_EpipolairePure,
         "Not epipolar geometry for census "
    );


 //  ====  1. GEOMETRIE EPIPOLAIRE BASIQUE
    ELISE_ASSERT
    (
       mNbIm <= 2,
       "Image > 2  in Census"
    );

    cGPU_LoadedImGeom &   anI0 = *(mVLI[0]);
    cGPU_LoadedImGeom &   anI1 = *(mVLI[1]);


    double aStepPix = mStepZ / mCurEtape->DeZoomTer();

 //  ====  2. Pas quotient d'entier
    double aRealNbByPix = 1/ aStepPix;
    int mNbByPix = round_ni(aRealNbByPix);

    if (ElAbs(aRealNbByPix-mNbByPix) > TolNbByPix)
    {
         std::cout << "For Step = " << mStepZ  << " GotDif " << aRealNbByPix-mNbByPix << "\n";
         ELISE_ASSERT(false,"in DoCensusCorre step is not 1/INT");
    }
/*
*/


    Pt2di anOff0 = anI0.OffsetIm();
    Pt2di anOff1 = anI1.OffsetIm();


// std::cout << anOff0 << anOff1 << "\n";

    // std::cout << mX0Ter  << " " << mY0Ter << "\n";

    // mCurSzVMax 

    float ** aDataIm0 =  anI0.VDataIm()[0];
    float ** aDataIm1 =  anI1.VDataIm()[0];
    cInterpolateurIm2D<float> * anInt = CurEtape()->InterpFloat();



    
    for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
    {
        for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
        {
               Pt2di aPIm0 = Pt2di(anX,anY) + anOff0;
               bool OkIm0 = anI0.IsOkErod(aPIm0.x,aPIm0.y);


               int aZ0 =  mTabZMin[anY][anX];
               int aZ1 =  mTabZMax[anY][anX];
               for (int aZI=aZ0; aZI< aZ1 ; aZI++)
               {
                   double aZR = aZI * aStepPix;
                   double aCost = mAhDefCost;
                   if (OkIm0)
                   {
                       Pt2dr aPIm1 = Pt2dr(anX,anY) + Pt2dr(anOff1) + Pt2dr(aZR,0);
                       if (anI1.IsOkErod(round_down(aPIm1.x),round_down(aPIm1.y)))
                       {
                            // double aC2 = CorrelBasic(aDataIm0,aPIm0,aDataIm1,anX+anOff1.x+aZR,anY+anOff1.y,mCurSzVMax,mAhEpsilon);
                            // double aC2 = CensusBasic(aDataIm0,aPIm0,aDataIm1,anX+anOff1.x+aZR,anY+anOff1.y,mCurSzVMax,mAhEpsilon);
                            double aC2 = CensusGraphe(aDataIm0,aPIm0,aDataIm1,anX+anOff1.x+aZR,anY+anOff1.y,mCurSzVMax,mAhEpsilon);
/*
                            RMat_Inertie aMat;
                            for (int aDx = -mCurSzVMax.x ; aDx <= mCurSzVMax.x ; aDx++)
                            {
                                for (int aDy = -mCurSzVMax.y ; aDy <= mCurSzVMax.y ; aDy++)
                                {
                                     Pt2dr aPV(aDx,aDy);
                                     Pt2dr aPV0 = Pt2dr(aPIm0) + aPV;
                                     Pt2dr aPV1 = Pt2dr(aPIm1) + aPV;
                                     double aV0 = anInt->GetVal(aDataIm0,aPV0);
                                     double aV1 = anInt->GetVal(aDataIm1,aPV1);
                                     aMat.add_pt_en_place(aV0,aV1);
                                }
                            }
                            double aC1 = aMat.correlation(mAhEpsilon);
*/
                            //std::cout << "COSTE " << aC1 << " " << aC2 << "\n";
                            aCost = mStatGlob->CorrelToCout(aC2);
                             // aCost = mStatGlob->CorrelToCout(aC2);
                       }
                   }
                   mSurfOpt->SetCout(Pt2di(anX,anY),&aZI,aCost);
               }
        }
    }

}

}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à  la mise en
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
associés au chargement,  à  l'utilisation,  à  la modification et/ou au
développement et à  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe à
manipuler et qui le réserve donc à  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à  charger  et  tester  l'adéquation  du
logiciel à  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
à  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder à  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
