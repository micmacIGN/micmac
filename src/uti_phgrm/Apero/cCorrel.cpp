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
#include "Apero.h"



std::vector<cRecorrel*> RC2VCR(cRecorrel & aRC)
{
    std::vector<cRecorrel*> aVRC;
    aVRC.push_back(&aRC);
    return aVRC;
}

cRecorrel::cRecorrel
(
    const cOneVisuPMul & aVPM,
    cPoseCam *          aPose,
    double aSzV,  // 1 Pour 3x3
    double aStep  
) :
   Optim2DParam ( 0.1, -1e5, 1e-6, true),
   mInterp   (aVPM.Interp()),
   mIm       (aPose->Im()),
   mTIm      (mIm),
   mSzIm     (mIm.sz()-Pt2di(1,1)),
   mBestDec   (0.0,0.0),
   mBestCorrel (-1)
{
   const CamStenope * aCS  = aPose->CurCam();

   mPImInitAbs = aCS->R3toF2(aVPM.PTer00());
   mDx = aCS->R3toF2(aVPM.PTer00()+aVPM.X_VecTer()) - mPImInitAbs;
   mDy = aCS->R3toF2(aVPM.PTer00()+aVPM.Y_VecTer()) - mPImInitAbs;

   mPImInitLoc = mPImInitAbs - Pt2dr(aPose->BoxIm()._p0);

   int aNb = round_ni(aSzV/aStep);

   mBox._p0  = mPImInitLoc;
   mBox._p1  = mPImInitLoc;

   for (int aNbX=-aNb ; aNbX<=aNb ; aNbX++)
   {
       for (int aNbY=-aNb ; aNbY<=aNb ; aNbY++)
       {
           Pt2dr aP = mPImInitLoc + mDx * (aNbX*aStep) + mDy * (aNbY*aStep);
           mPInit.push_back(aP);
           mPDec.push_back(aP);  //   Reserve de la place
           mValues.push_back(0); //   Reserve de la place
           mBox._p0.SetInf(aP);
           mBox._p1.SetSup(aP);
       }
   }
   mNbPts =  (int)mPInit.size();
   int  aSzK = mInterp->SzKernel()+1;
   Pt2dr aPSzK(aSzK,aSzK);
   mBox._p0 = mBox._p0-aPSzK;
   mBox._p1 = mBox._p1+aPSzK;


   // Utile pour le decalage de base
   mIsInit = SetValDec(Pt2dr(0,0));


   if (mIsInit)
   {
      mBestValues =  mValues;

      mS1 = 0;
      mS2 = 0;
      for (int aK=0 ; aK<mNbPts  ; aK++)
      {
        mS1 += mValues[aK];
        mS2 += ElSquare(mValues[aK]);
      }
      mS1 /= mNbPts;
      mS2 /= mNbPts;
      mS2 -= ElSquare(mS1);
   }
}

double cRecorrel::BestCorrel() const
{
   return mBestCorrel;
}


Pt2dr cRecorrel::BestPImAbs() const
{
   return mPImInitAbs + mBestDec;
}



bool cRecorrel::SetValDec(Pt2dr aDec)
{
   if (
              (aDec.x+mBox._p0.x <= 0)
           || (aDec.y+mBox._p0.y <= 0)
           || (aDec.x+mBox._p1.x >= mSzIm.x)
           || (aDec.y+mBox._p1.y >= mSzIm.y)
      )
      return false;

   for (int aK=0 ; aK<mNbPts  ; aK++)
         mPDec[aK] = mPInit[aK]+aDec;

   mInterp->GetVals(mIm.data(),&(mPDec[0]),&(mValues[0]),mNbPts);
   return true;
}

double cRecorrel::OneCorrelOfDec(Pt2dr aDec,cRecorrel & aRC)
{
   if ((!mIsInit) || (! aRC.mIsInit))
      return -1;
   if (!SetValDec(aDec)) 
      return -1;

   mS1 = 0;
   mS2 = 0;
   double aS12 = 0;
   for (int aK=0 ; aK<mNbPts  ; aK++)
   {
        mS1 += mValues[aK];
        aS12 += mValues[aK] * aRC. mBestValues[aK];
        mS2 += ElSquare(mValues[aK]);
   }
   mS1 /= mNbPts;
   mS2 /= mNbPts;
   mS2 -= ElSquare(mS1);
   aS12 /=  mNbPts;
   aS12 -= mS1 * aRC.mS1;

   double aRes = aS12/sqrt(ElMax(0.1,mS2*aRC.mS2));

   return aRes;
}

double cRecorrel::TestAndUpdateOneCorrelOfDec
       (
           Pt2dr aDec,
           const std::vector<cRecorrel*> & aVRC
       )
{
    double aSCor = 0;
    for (int aK=0 ; aK<int(aVRC.size()) ; aK++)
        aSCor += OneCorrelOfDec(aDec,*(aVRC[aK]));
    aSCor /= aVRC.size();
    Udpate(aDec,aSCor);
    return aSCor;
}


void cRecorrel::DescAs_Optim2DParam(const std::vector<cRecorrel*> &aVRC)
{
    mVRC  = & aVRC;
    Optim2DParam::optim(mBestDec);
    mVRC = 0;
}

void cRecorrel::DescAs_Optim2DParam(cRecorrel &aRC)
{
   DescAs_Optim2DParam(RC2VCR(aRC));
}


double cRecorrel::Op2DParam_ComputeScore(double aX,double aY)
{
    return TestAndUpdateOneCorrelOfDec(Pt2dr(aX,aY),*mVRC);
}

double cRecorrel::TestAndUpdateOneCorrelOfDec(Pt2dr aDec,cRecorrel & aRC)
{
    return TestAndUpdateOneCorrelOfDec(aDec,RC2VCR(aRC));
}


void cRecorrel::ExploreVois(int aNb,double aStep,const std::vector<cRecorrel*> &aVRC)
{
    Pt2dr aDec = mBestDec;
    for (int aKx=-aNb ; aKx<=aNb ; aKx++)
    {
        for (int aKy=-aNb ; aKy<=aNb ; aKy++)
        {
             TestAndUpdateOneCorrelOfDec(aDec+Pt2dr(aKx*aStep,aKy*aStep),aVRC);
        }
    }
}

void cRecorrel::ExploreVois(int aNb,double aStep,cRecorrel &aRC)
{
     ExploreVois(aNb,aStep,RC2VCR(aRC));
}


/* */


void cRecorrel::Udpate(const Pt2dr & aDec,double aCorrel)
{
   if (aCorrel > mBestCorrel)
   {
        mBestCorrel = aCorrel;
        mBestDec = aDec;
        mBestValues = mValues;
   }
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
