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
#include "../NewRechPH/cParamNewRechPH.h"
#include "../NewRechPH/ExternNewRechPH.h"
#include "../NewRechPH/NewRechPH.h"


/***********************************************************/
/*                                                         */
/*      Auto Correl                                        */
/*                                                         */
/***********************************************************/

namespace AimeImageAutoCorrel
{

     // =============== cOneICAIAC  ====   

cOneICAIAC::cOneICAIAC(int aTx,int aTy) :
     mTx     (aTx),
     mTy     (aTy),
     mImCor  (mTx,mTy),
     mTImCor (mImCor),
     mImVis  (mTx,mTy)
{
}

void cOneICAIAC::MakeImVis(bool isRobust) 
{
   mImVis = MakeImI1(isRobust,mImCor);
};
void cOneICAIAC::MakeTiff(const std::string & aName)
{
     Tiff_Im::Create8BFromFonc(aName,mImVis.sz(),mImVis.in()+128);
}


     // =============== cAimeImAutoCorr  ====   
cAimeImAutoCorr::cAimeImAutoCorr(Im2D_INT1 anIm) :
    mSz      (anIm.sz()),
    mNbR     (mSz.x),
    mNbT0    (mSz.y)
{
}



     // cCalcAimeImAutoCorr

double cCalcAimeImAutoCorr::AutoCorrelR0(int aRho,int aDTeta)
{
    ELISE_ASSERT(aDTeta>=0,"cCalcAimeImAutoCorr::AutoCorrel");
    double aRes = 0;

    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aV0 =  mTImInit.get(Pt2di(aRho,aKT));
        double aVD =  mTImInit.get(Pt2di(aRho,(aKT+aDTeta)%mNbT0));

        aRes +=  aV0 * aVD;
    }
    return aRes;
}

double cCalcAimeImAutoCorr::AutoCorrelGT(int aRho,int aDTeta)
{
    ELISE_ASSERT(aDTeta>=0,"cCalcAimeImAutoCorr::AutoCorrel");
    double aRes = 0;

    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aV0 =  mTImInit.get(Pt2di(aRho,aKT));
        double aV1 =  mTImInit.get(Pt2di(aRho,(aKT+1)%mNbT0));
        double aVD =  mTImInit.get(Pt2di(aRho,(aKT+aDTeta)%mNbT0));

        aRes +=  (aV1-aV0) * aVD;
    }
    return aRes;
}

double cCalcAimeImAutoCorr::AutoCorrelGR(int aRho,int aDTeta)
{
    ELISE_ASSERT(aDTeta>=0,"cCalcAimeImAutoCorr::AutoCorrel");
    double aRes = 0;

    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aV0 =  mTImInit.get(Pt2di(aRho,aKT));
        double aV1 =  mTImInit.get(Pt2di(aRho-1,aKT));
        double aVD =  mTImInit.get(Pt2di(aRho,(aKT+aDTeta)%mNbT0));

        aRes +=  (aV1-aV0) * aVD;
    }
    return aRes;
}



#if  PB_LINK_AUTOCOR // PB LINK INCOMPREHENSIBLE  Ann + Micmac + Qt => ?@&#!!!
#else


cCalcAimeImAutoCorr::cCalcAimeImAutoCorr(Im2D_INT1 anIm,bool L1Mode) :
    cAimeImAutoCorr (anIm),
    mImInit         (anIm),
    mTImInit        (anIm),
    mL1Mode         (L1Mode),
    mIR0            (mNbR,mNbT0/2),
    mIGR            (mNbR-1,mNbT0),
    mIGT            (mNbR,mNbT0/2)
/*
    mTImCor         (mImCor),
    mImVis          (1,1)
*/
{
    int aSzTetaR0 =  mIR0.mImCor.sz().y;
    int aSzTetaGT =  mIGT.mImCor.sz().y; // GT being anti symetric, take only half size
    for (int aKT=0 ; aKT<mNbT0 ; aKT++)
    {
        double aS0=0;
        double aS1=0;
        double aS2=0;
        bool DoR0 = aKT < aSzTetaR0;
        bool DoGT = aKT < aSzTetaGT;
        for (int aKR=0 ; aKR<mNbR ; aKR++)
        {
           
            if (DoR0)
            {
                double aC = AutoCorrelR0(aKR,aKT+1);
                aS0 += 1;
                aS1 += aC;
                aS2 += ElSquare(aC);
                mIR0.mTImCor.oset(Pt2di(aKR,aKT),aC);
            }

            // double  aC = AutoCorrelGR(aKR,aKT+1); 
            if (DoGT)
            {
               mIGT.mTImCor.oset(Pt2di(aKR,aKT),AutoCorrelGT(aKR,aKT));
            }
            if (aKR>=1)
            {
               mIGR.mTImCor.oset(Pt2di(aKR-1,aKT),AutoCorrelGR(aKR,aKT));
            }
        }
        if (DoR0)
        {
            aS1 /= aS0;
            aS2 /= aS0;
            aS2 -= ElSquare(aS1);
            double aSig = sqrt(ElMax(1e-10,aS2));
            for (int aKR=0 ; aKR<mNbR ; aKR++)
            {
                double  aC =  mIR0.mTImCor.get(Pt2di(aKR,aKT));
                mIR0.mTImCor.oset(Pt2di(aKR,aKT),(aC-aS1)/aSig);
            }
        }
    }
    mIR0.MakeImVis(mL1Mode);
    mIGT.MakeImVis(mL1Mode);
    mIGR.MakeImVis(mL1Mode);
/*
    CalcImVis(mImCor,mImVis);
    double aVMax,aVMin;
    ELISE_COPY(mImCor.all_pts(),mImCor.in(),VMax(aVMax)|VMin(aVMin));
    double aDyn = 127.0/ ElMax(1e-10,ElMax(-aVMin,aVMax));
    ELISE_COPY(mImCor.all_pts(),round_ni(mImCor.in()*aDyn),mImVis.out());
*/
}
#endif 

}
