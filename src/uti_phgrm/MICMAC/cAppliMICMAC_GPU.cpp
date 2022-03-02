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

extern bool ERupnik_MM();

/** @addtogroup GpGpuDoc */
/*@{*/

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

template <class Type,class TBase> void  SaveIm(const std::string & aName,Im2D<Type,TBase> anIm,const Box2di & aBox)
{
   Tiff_Im aRes
           (
                 aName.c_str(),
                 aBox.sz(),
                 anIm.TypeEl(),
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
            );

   ELISE_COPY
   (
       aRes.all_pts(),
       trans(anIm.in(),aBox._p0),
       aRes.out()
   );   

}



/********************************************************************/
/*                                                                  */
/*                   cStatOneImage                                  */
/*                                                                  */
/********************************************************************/

cStatOneImage::cStatOneImage()
{
   Reset();
}


void cStatOneImage::Reset()
{
   mS1 = 0.0;
   mS2 = 0.0;
   mVals.clear();
}

void cStatOneImage::Normalise(double aMoy,double aSigma)
{
    int aNb = (int)mVals.size();
    double * aData = &(mVals[0]);

    for (int aK=0 ; aK<aNb ; aK++)
    {
        aData[aK] = (aData[aK] - aMoy) / aSigma;
    }
}

void cStatOneImage::StdNormalise(double aEpsilon)
{
    double aMoy = mS1 / mVals.size();
    double aSigma2  = mS2 / mVals.size() - ElSquare(aMoy);

    Normalise(aMoy,sqrt(ElMax(aEpsilon,aSigma2)));
}


double cStatOneImage::SquareDist(const cStatOneImage & aS2) const
{
    int aNb = (int)mVals.size();
    ELISE_ASSERT(aNb==int(aS2.mVals.size()),"Incoherent size in cStatOneImage::SquareDist");

    const double * aD1 = &(mVals[0]);
    const double * aD2 = &(aS2.mVals[0]);
    double aRes = 0;

    for (int aK=0 ; aK<aNb ; aK++)
        aRes += ElSquare(aD1[aK]-aD2[aK]);

    return aRes;
}


double cStatOneImage::SquareNorm() const
{
    int aNb = (int)mVals.size();

    const double * aD1 = &(mVals[0]);
    double aRes = 0;

    for (int aK=0 ; aK<aNb ; aK++)
        aRes += ElSquare(aD1[aK]);

    return aRes;
}



/********************************************************************/
/*                                                                  */
/*                   cGPU_LoadedImGeom                              */
/*                                                                  */
/********************************************************************/


class cGLI_CalibRadiom
{
     public :
          cGLI_CalibRadiom (const cXML_RatioCorrImage aXml) :
	      mR  (aXml.Ratio()),
	      mRTif(0),
	      mRTIm(0)
	  {
	  }
          cGLI_CalibRadiom (const std::string aName) :
	      mR(1.0),
	      mRTif(new Tiff_Im( Tiff_Im::StdConvGen(aName,-1,true))),
	      mRTIm(new TIm2D<REAL4,REAL>(mRTif->sz()))
          {
		ELISE_COPY(mRTIm->all_pts(), mRTif->in(), mRTIm->out());
          }

	  double                    mR;
	  Tiff_Im            	  * mRTif;
	  TIm2D<REAL4,REAL> 	  * mRTIm;


};

void cAppliMICMAC::ResetCalRad()
{
     for (auto It= mDicCalRad.begin() ; It!= mDicCalRad.end() ; It++)
         delete It->second;
     mDicCalRad.clear();
}


cGPU_LoadedImGeom::cGPU_LoadedImGeom
(
        const cAppliMICMAC & anAppli,
        cPriseDeVue* aPDV,
        const Box2di & aBox,
        const Pt2di  &aSzV0,
        const Pt2di  &aSzVMax,
                bool  Top
) :
      mAppli   (anAppli),
      mTop     (Top),
      mPDV     (aPDV),
      mLI      (&aPDV->LoadedIm()),
      mGeom    (&aPDV->Geom()),

      mSzV0    (aSzV0),
      mSzVMax  (aSzVMax),
      mSzOrtho (aBox.sz()+ mSzVMax*2),

      mOPCms   (0),

      mImOrtho (mSzOrtho.x,mSzOrtho.y),
      mDImOrtho (mImOrtho),
      mDOrtho   (ImDec(mVOrtho,mImOrtho,aBox,mSzVMax)),

      mImSomO  (mSzOrtho.x,mSzOrtho.y),
      mDImSomO (mImSomO),
      mDSomO   (ImDec(mVSomO,mImSomO,aBox,mSzVMax)),

      mImSomO2  (mSzOrtho.x,mSzOrtho.y),
      mDImSomO2 (mImSomO2),
      mDSomO2   (ImDec(mVSomO2,mImSomO2,aBox,mSzVMax)),

      mImSom12  (mSzOrtho.x,mSzOrtho.y),
      mDImSom12 (mImSom12),
      mDSom12   (ImDec(mVSom12,mImSom12,aBox,mSzVMax)),

      mImOK_Ortho (mSzOrtho.x,mSzOrtho.y),
      mDImOK_Ortho (mImOK_Ortho),
      mDOK_Ortho   (ImDec(mVImOK_Ortho,mImOK_Ortho,aBox,mSzVMax)),


      mNbVals    ( mAppli.CMS_ModeEparse() ?  9 : ((1+2*mSzV0.x) * (1+2*mSzV0.y)) ) ,


      mVals    (mNbVals),
      mDataIm  (mLI->DataFloatIm()),
      mLinDIm  (mLI->DataFloatLinIm()),
      mSzX     (mLI->SzIm().x),
      mSzY     (mLI->SzIm().y),
      mImMasq  (mLI->DataMasqIm()),
      mImMasqErod  (mLI->DataMasqImErod()),
      mImPC    (mLI->DataImPC()),
      mSeuilPC (mLI->SeuilPC()),
      mUsePC   (mLI->UsePC())
{


//    ELISE_ASSERT
//    (
//        aPDV->NumEquiv()==0,
//    "Ne gere pas les classe d'equiv image en GPU"
//    );

    if (! Top)
       return;

    mMSGLI.push_back(this);
    const cCorrelMultiScale*  aCMS = anAppli.CMS();

    mOneImage = true;

    if (! aCMS)
    {
       mPdsMS = 1.0;
       mCumSomPdsMS = 1.0;
       return;
    }

    const std::vector<cOneParamCMS> & aVP = aCMS->OneParamCMS();


    double aSomPds = 0;
    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
        if (aK>0)
        {
             mMSGLI.push_back(new cGPU_LoadedImGeom(anAppli,aPDV,aBox,aVP[aK].SzW(),mSzVMax,false));
        }

        mMSGLI[aK]->mOPCms = &(aVP[aK]);

        double aPdsK = aVP[aK].Pds();
        mMSGLI[aK]->mPdsMS = aPdsK/  mMSGLI[aK]->mNbVals;
        aSomPds += aPdsK;
        mMSGLI[aK]->mCumSomPdsMS =aSomPds;
        mMSGLI[aK]->mMyDataIm0 = mDataIm[aK];
        mMSGLI[aK]->mMaster = this;
    }


    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
        mMSGLI[aK]->mTotSomPdsMS = aSomPds;
    }

    mOneImage = (aVP.size()==1);
}

Pt2di cGPU_LoadedImGeom::OffsetIm()
{
   double   aZter[2] ={0,0};
   Pt2dr aPTer = mAppli.DequantPlani(0,0);

   Pt2dr aResR = mGeom->CurObj2Im(aPTer,aZter);

   Pt2di aResI = round_ni(aResR);

   if (euclid(aResR,Pt2dr(aResI)) > 1e-4)
   {
        std::cout << "REEESR " << aResR << "\n";
        ELISE_ASSERT(false,"Non Integer Offset\n");
   }

   return aResI;
}


std::vector<Im2D_REAL4> cGPU_LoadedImGeom::VIm()
{
   std::vector<Im2D_REAL4> aRes;
   for (int aKS=0 ; aKS<int(mMSGLI.size()) ; aKS++)
   {
       aRes.push_back(*FloatIm(aKS));

   }


   return aRes;
}


void cGPU_LoadedImGeom::InitCalibRadiom(cGLI_CalibRadiom * aCal)
{
    mCalR = aCal;
}

double cGPU_LoadedImGeom::CorrRadiom(double aVal)
{
   return aVal / mCalR->mR ;
}

double cGPU_LoadedImGeom::CorrRadiom(double aVal, const Pt2dr &aP)
{
    if(mCalR->mRTIm)
    {
	return aVal / mCalR->mRTIm->Val(aP.x,aP.y) ;
    }
    else
	return aVal / mCalR->mR ;
}

Pt2di  cGPU_LoadedImGeom::SzV0() const
{
   return mSzV0;
}

int  cGPU_LoadedImGeom::NbScale() const
{
    return (int)mMSGLI.size();
}

Im2D_REAL4 * cGPU_LoadedImGeom::FloatIm(int aKScale)
{
   return mLI->FloatIm()[aKScale];
}



cGPU_LoadedImGeom::~cGPU_LoadedImGeom()
{
   if (mTop)
   {
       // mMSGLI[0] contient this, donc commence a 1
       for (int aK=1; aK<int(mMSGLI.size()) ; aK++)
           delete mMSGLI[aK];
   }
}

tGpuF **    cGPU_LoadedImGeom::DataOrtho()   {return mDOrtho;    }
U_INT1 **   cGPU_LoadedImGeom::DataOKOrtho() {return mDOK_Ortho; }
Im2D_U_INT1 cGPU_LoadedImGeom::ImOK_Ortho()  {return mImOK_Ortho;}

tImGpu  cGPU_LoadedImGeom::ImOrtho()  {return mImOrtho; }
tImGpu  cGPU_LoadedImGeom::ImSomO()   {return mImSomO; }
tImGpu  cGPU_LoadedImGeom::ImSomO2()  {return mImSomO2; }
tImGpu  cGPU_LoadedImGeom::ImSom12()  {return mImSom12; }


tGpuF **    cGPU_LoadedImGeom::DataSomO()   {return mDSomO;    }
tGpuF **    cGPU_LoadedImGeom::DataSomO2()  {return mDSomO2;    }

cPriseDeVue * cGPU_LoadedImGeom::PDV()
{
   return mPDV;
}



Pt2dr cGPU_LoadedImGeom::ProjOfPDisc(int anX,int anY,int aZ) const
{
    double aZR = mAppli.DequantZ(aZ);
    Pt2dr aPR = mAppli.DequantPlani(anX,anY);

    return mGeom->CurObj2Im(aPR,&aZR);
}

void cGPU_LoadedImGeom::MakeDeriv(int anX,int anY,int aZ)
{
    mPOfDeriv = Pt3di(anX,anY,aZ);

    mValueP0D = ProjOfPDisc(anX,anY,aZ);
// std::cout << "MkddDD " << mValueP0D << " " << anX << " " << anY << " " << aZ << "\n";
    mDerivX = (ProjOfPDisc(anX+1,anY,aZ)- ProjOfPDisc(anX-1,anY,aZ)) / 2.0;
    mDerivY = (ProjOfPDisc(anX,anY+1,aZ)- ProjOfPDisc(anX,anY-1,aZ)) / 2.0;
    mDerivZ = (ProjOfPDisc(anX,anY,aZ+1)- ProjOfPDisc(anX,anY,aZ-1)) / 2.0;

    mX0Deriv = anX;
    mY0Deriv = anY;
    mZ0Deriv = aZ;
}

Pt2dr cGPU_LoadedImGeom::ProjByDeriv(int anX,int anY,int aZ) const
{

   Pt2dr aRes =    mValueP0D
          +  mDerivX*double(anX-mX0Deriv)
          +  mDerivY*double(anY-mY0Deriv)
          +  mDerivZ*double( aZ-mZ0Deriv) ;


  return aRes;
}


cStatOneImage * cGPU_LoadedImGeom::VignetteDone()
{
    return & mBufVignette;
}

REAL cGPU_LoadedImGeom::GetValOfDisc(int anX,int anY,int aZ)
{
    Pt2dr aP = ProjOfPDisc(anX,anY,aZ);
    cInterpolateurIm2D<float> * anInt = mAppli.CurEtape()->InterpFloat();

    return IsOk(aP.x,aP.y) ? anInt->GetVal(MyDataIm0(),aP) : 0 ;
}

cStatOneImage * cGPU_LoadedImGeom::ValueVignettByDeriv(int anX,int anY,int aZ,int aSzV,Pt2di aPasVig)
{

    cInterpolateurIm2D<float> * anInt = mAppli.CurEtape()->InterpFloat();
    mBufVignette.Reset();
    Pt2dr aP0 = mMaster->ProjByDeriv(anX,anY,aZ);

    Pt2dr aDx = mDerivX * double(aPasVig.x);
    Pt2dr aDy = mDerivY * double(aPasVig.y);

    Pt2dr aDebL =  aP0 - aDx*double(aSzV) - aDy*double(aSzV);
    for (int aKY=-aSzV ; aKY<=aSzV ; aKY++)
    {
        Pt2dr aCur = aDebL;
        for (int aKX=-aSzV ; aKX<=aSzV ; aKX++)
        {
            if (IsOk(aCur.x,aCur.y))
               mBufVignette.Add(anInt->GetVal(MyDataIm0(),aCur));
            else
               return 0;
            aCur += aDx;
        }
        aDebL += aDy;
    }

    return & mBufVignette;
}

bool   cGPU_LoadedImGeom::InitValNorms(int anX,int anY,int aNbScaleIm)
{

    if (! mDOK_Ortho[anY][anX])
    {
       return false;
    }


       mMoy  = MoyIm(anX,anY,aNbScaleIm);
       //mMoy   = mDSomO[anY][anX] / mNbVals;
       //  double aDMoy = mEpsAddMoy + mMoy * mEpsMulMoy;
       // Magouille a cause des mEpsAddMoy, mEpsMulMoy qui en vrai sont tjr == 0 !!!!
       double aDMoy = mAppli.DeltaMoy(mMoy);


       // mSigma  = mDSomO2[anY][anX] / mNbVals - QSquare(mMoy) + QSquare(aDMoy);

       mSigma  = MoyQuadIm(anX,anY,aNbScaleIm)  - QSquare(mMoy) + QSquare(aDMoy);
       mMoy += aDMoy;


// std::cout << "SSSsig " << mSigma  << " " <<  MoyQuadIm(anX,anY,aNbScaleIm) << " " << mMoy << "\n";

      if (mSigma < mAppli.AhEpsilon())
      {
          return false;
      }

       mSigma = sqrt(mSigma);

       // printf("MQI %9.9f %9.9f \n",MoyQuadIm(anX,anY,aNbScaleIm),mDSomO2 [anY][anX]);
       // std::cout << " VALS NORM " << mMoy << " " << mSigma  << " " << aDMoy << " " << mDSomO2 [anY][anX]  << "\n\n";

       for (int aKS=1 ; aKS<int(mMSGLI.size()) ; aKS++)
       {
           mMSGLI[aKS]->mMoy   = mMoy;
           mMSGLI[aKS]->mSigma = mSigma;
       }

       return true;
}

double  cGPU_LoadedImGeom::MoyIm(int anX,int anY,int aNbScaleIm) const
{
    if (! mOPCms)
        return mDSomO [anY][anX] /mNbVals;

    double aRes = 0;
    for (int aK=0 ; aK<aNbScaleIm ; aK++)
    {
        cGPU_LoadedImGeom * aGLI = mMSGLI[aK];
        aRes += aGLI->mDSomO [anY][anX] * aGLI->mPdsMS;

    }

    aRes /= mMSGLI[aNbScaleIm-1]->mCumSomPdsMS;
    if (0)
    {
        std::cout << "Moy IM " << aRes  << " nbS " << aNbScaleIm << " O " << mDSomO [anY][anX]  << " " << mPdsMS << " " << mCumSomPdsMS << "\n";

    }
    return aRes;
}
double  cGPU_LoadedImGeom::MoyQuadIm(int anX,int anY,int aNbScaleIm) const
{
    if (! mOPCms)
        return mDSomO2 [anY][anX] /mNbVals;

    double aRes = 0;
    for (int aK=0 ; aK<aNbScaleIm ; aK++)
    {
        cGPU_LoadedImGeom * aGLI = mMSGLI[aK];
        aRes += aGLI->mDSomO2 [anY][anX] * aGLI->mPdsMS;
    }
    return aRes / mMSGLI[aNbScaleIm-1]->mCumSomPdsMS;
}

double  cGPU_LoadedImGeom::CovIm(int anX,int anY,int aNbScaleIm) const
{
    if (! mOPCms)
        return mDSom12 [anY][anX] /mNbVals;

    double aRes = 0;
    for (int aK=0 ; aK<aNbScaleIm ; aK++)
    {
        cGPU_LoadedImGeom * aGLI = mMSGLI[aK];
        aRes += aGLI->mDSom12 [anY][anX] * aGLI->mPdsMS;
    }
    return aRes / mMSGLI[aNbScaleIm-1]->mCumSomPdsMS;
}

double  cGPU_LoadedImGeom::PdsMS() const
{
   return mPdsMS;
}
double  cGPU_LoadedImGeom::CumSomPdsMS() const
{
   return mCumSomPdsMS;
}
double  cGPU_LoadedImGeom::TotSomPdsMS() const
{
   return mTotSomPdsMS;
}
/*
*/

/*
double MoyQuad() const;
double Cov(const cGPU_LoadedImGeom & aGeoJ) const;
*/


bool   cGPU_LoadedImGeom::CorreCensus(double & aCorrel,int anX,int anY,const  cGPU_LoadedImGeom & aGeoJ,int aNbScaleIm) const
{

   static int aCptCC = 0;

   if (! mDOK_Ortho[anY][anX])
      return false;

   bool ModeQuant = (mAppli.CC()->TypeCost()==eMCC_CensusQuantitatif);

   double anEcGlob=0;
   double aSomPds = 0;
   bool isDense = false;

   for (int aKS=0 ; aKS<aNbScaleIm ; aKS++)
   {
      tDataGpu  aDI = mMSGLI[aKS]->mDOrtho;
      tDataGpu  aDJ = aGeoJ.mMSGLI[aKS]->mDOrtho;
      // const cCorrelMultiScale*  aCMS = mAppli.CMS();
      float aVCI = aDI[anY][anX];
      float aVCJ = aDJ[anY][anX];

      double aScSomEc = 0;
      Pt2di aSzW =  mMSGLI[aKS]->mOPCms->SzW();

      int IncrX = isDense ? 1 : aSzW.x;
      int IncrY = isDense ? 1 : aSzW.y;
      int aNbX =  isDense ? 1+2* aSzW.x  : 3;

      for (int aDY=-aSzW.y ; aDY<=aSzW.y ;aDY+=IncrY)
      {
          float * aLI = aDI[anY+aDY] + anX - aSzW.x;
          float * aLJ = aDJ[anY+aDY] + anX - aSzW.x;
          if (ModeQuant)
          {
              for (int aCpt = aNbX ; aCpt ; aCpt--)
              {
                  aScSomEc += ElAbs(EcartNormalise(aVCI,*aLI)-EcartNormalise(aVCJ,*aLJ));
                  aLI+=IncrX;
                  aLJ+=IncrX;
              }
          }
      }
      double aPds = isDense ? mMSGLI[aKS]->mPdsMS : mMSGLI[aKS]->mOPCms->Pds() ;
      anEcGlob += aScSomEc * aPds;
      aSomPds += aPds;
   }
   // Min pour meme interv que correl
   anEcGlob =  ElMin(2.0,(anEcGlob/aSomPds) * mAppli.CC()->Dyn().Val());
   aCorrel =  1-anEcGlob;
{
// std::cout << " cGPU_LoadedImGeom::Correl " << mAppli.CC()->Dyn().Val() << " " << mCumSomPdsMS  << "\n"; getchar();
}

   aCptCC++;

   return true;
}

bool   cGPU_LoadedImGeom::Correl(double & aCorrel,int anX,int anY,const  cGPU_LoadedImGeom & aGeoJ,int aNbScaleIm) const
{
// if (MPD_MM())

        if (! mDOK_Ortho[anY][anX])
            return false;
        if (mAppli.CC())
        {
            return CorreCensus(aCorrel,anX,anY,aGeoJ,aNbScaleIm);
        }

        double aMI  = MoyIm(anX,anY,aNbScaleIm);
        double aDmI = mAppli.DeltaMoy(aMI);
        double aMII =  MoyQuadIm(anX,anY,aNbScaleIm) - ElSquare(aMI) + ElSquare(aDmI);

//std::cout << "##2## NBSSSS " << aNbScaleIm  << " " <<   aMII << " " <<  MoyIm(anX,anY,aNbScaleIm) << " " << MoyQuadIm(anX,anY,aNbScaleIm)  << "\n";
        if (aMII < mAppli.AhEpsilon())
            return false;

        double aMJ  =  aGeoJ.MoyIm(anX,anY,aNbScaleIm);
        double aDmJ = mAppli.DeltaMoy(aMJ);
        double aMJJ =  aGeoJ.MoyQuadIm(anX,anY,aNbScaleIm) - ElSquare(aMJ) + ElSquare(aDmJ);
        if (aMJJ < mAppli.AhEpsilon())
            return false;

        double aMIJ =  CovIm(anX,anY,aNbScaleIm) - aMI * aMJ + aDmI*aDmJ;

        aCorrel = aMIJ / sqrt(aMII*aMJJ);

///std::cout << "#################  NBSSSS " << aNbScaleIm  << " " <<   mDOK_Ortho[anY][anX] << "\n";

if (0)
{
   static double aNb=0; aNb++;
   static double aNbOut=0;
   static double aMaxC = -10;
   static double aMinC = 10;
   if ((aCorrel>1) || (aCorrel<-1))
   {
      aNbOut++;
   }
   if (aCorrel>aMaxC)
   {
        aMaxC = aCorrel;
        std::cout  << "MAX-MIN-COR   " << aMinC << " " << aMaxC << " OUT " << (aNbOut/aNb) << "\n";
   }
   if (aCorrel<aMinC)
   {
        aMinC = aCorrel;
        std::cout  << "MAX-MIN-COR   " << aMinC << " " << aMaxC << " OUT " << (aNbOut/aNb) << "\n";
   }
}
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



void cAppliMICMAC::DoInitAdHoc(const Box2di & aBox)
{

        mX0Ter = aBox._p0.x;
        mX1Ter = aBox._p1.x;
        mY0Ter = aBox._p0.y;
        mY1Ter = aBox._p1.y;

        mDilX0Ter = mX0Ter  - mCurSzVMax.x;
        mDilY0Ter = mY0Ter  - mCurSzVMax.y;
        mDilX1Ter = mX1Ter  + mCurSzVMax.x;
        mDilY1Ter = mY1Ter  + mCurSzVMax.y;

        mCurSzDil = Pt2di(mDilX1Ter-mDilX0Ter, mDilY1Ter-mDilY0Ter);

        mImOkTerCur.Resize(mCurSzDil);
        mTImOkTerCur = TIm2D<U_INT1,INT> (mImOkTerCur);
        mDOkTer = ImDec(mVDOkTer,mImOkTerCur,aBox,mCurSzVMax);

        mImOkTerDil.Resize(mCurSzDil);
        mTImOkTerDil = TIm2D<U_INT1,INT> (mImOkTerDil);
        mDOkTerDil  = ImDec(mVDOkTerDil,mImOkTerDil,aBox,mCurSzVMax);

        Pt2di aSzAll1 = mAll1ImOkTerDil.sz();
        if ((aSzAll1.x < mCurSzDil.x ) || (aSzAll1.y<mCurSzDil.y))
        {
            mAll1ImOkTerDil = Im2D_U_INT1(mCurSzDil.x,mCurSzDil.y,1);
        }
        mAll1TImOkTerDil =  TIm2D<U_INT1,INT>(mAll1ImOkTerDil);
        mAll1DOkTerDil = ImDec(mAll1VDOkTerDil,mAll1ImOkTerDil,aBox,mCurSzVMax);



        mTabZMin = mLTer->GPULowLevel_ZMin();
        mTabZMax = mLTer->GPULowLevel_ZMax();

        mTabMasqTER = mLTer->GPULowLevel_MasqTer();

        mAhDefCost =  mStatGlob->CorrelToCout(mDefCorr);
        mAhEpsilon = EpsilonCorrelation().Val();

        mGeomDFPx->SetOriResolPlani(mOriPlani,mStepPlani);
        mOrigineZ = mGeomDFPx->OrigineAlti4Compute();
        mStepZ = mGeomDFPx->ResolutionAlti();

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
            mVLI.push_back(new cGPU_LoadedImGeom(*this,*itFI,aBox,mCurSzV0,mCurSzVMax,true));
        }
        mNbIm = (int)mVLI.size();

        mNbScale = mVLI.size() ?  mVLI[0]->NbScale()  : 0;


        mVScaIm.clear();
        for (int aKS=0 ; aKS<mNbScale ; aKS++)
        {
            std::vector<cGPU_LoadedImGeom *> aV;
            mVScaIm.push_back(aV);
        }

        for (int aKS=0 ; aKS<mNbScale ; aKS++)
        {
            for (int aKI=0 ; aKI<mNbIm ; aKI++)
            {
                mVScaIm[aKS].push_back(mVLI[aKI]->KiemeMSGLI(aKS));
            }
        }

        mZMinGlob = (int)1e7;
        mZMaxGlob = (int)(-mZMinGlob);

#if CUDA_ENABLED

        if (mCorrelAdHoc->TypeCAH().GPU_CorrelBasik().IsInit())
        {
            if (!IMmGg.TexturesAreLoaded())//		Mise en calque des images
            {
                IMmGg.SetTexturesAreLoaded(true);

                uint sizeTerGlob  = size(IMmGg.DimTerrainGlob());

                pixel *maskGlobal   = new pixel[sizeTerGlob];
                pixel *maskIML      = new pixel[sizeTerGlob*mNbIm];
                uint   idMask		=  0;

                for (uint anY = 0 ; anY <  IMmGg.DimTerrainGlob().y ; anY++,idMask += IMmGg.DimTerrainGlob().x)
                    for (uint anX = 0 ; anX < IMmGg.DimTerrainGlob().x ; anX++)
                    {
                        uint xId = idMask + anX;
                        if(IsInTer(anX,anY))
                        {
                            maskGlobal[xId] = 1 ;
                            IMmGg.NoMasked = true;
                        }
                        else
                            maskGlobal[xId] = 0 ;

                        for (int aKIm=0 ; aKIm<mNbIm ; aKIm++,xId+=sizeTerGlob)
                            maskIML[xId] = (*(mVLI[aKIm])).IsVisible(anX,anY) ? 255 : 0 ;

                    }

                if(IMmGg.NoMasked)
                {
                    // set textures masques images
                    IMmGg.Data().SetMaskImages(maskIML,IMmGg.DimTerrainGlob(),mNbIm);

                    // set texture masque terrain
                    IMmGg.Data().SetGlobalMask(maskGlobal,IMmGg.DimTerrainGlob());
                    float*	fdataImg1D	= NULL;
                    uint2	dimImgMax	= make_uint2(0,0);

                    ushort  nbCLass     = 1;
                    ushort  pitImage    = 0;

					IMmGg.Data().ReallocConstData(mNbIm);

                    ushort2* hClassEqui = IMmGg.Data().HostClassEqui();
					const ushort2 rayonVignette = make_ushort2(toUi2(mCurSzV0));

                    // Parcourt de toutes les images pour les classes
                    for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                    {
                        // image et orientation
                        cGPU_LoadedImGeom&	aGLI	= *(mVLI[aKIm]);

                        // classe d'équivalence
                        hClassEqui[aKIm].x = aGLI.PDV()->NumEquiv();

                        if(aKIm && hClassEqui[aKIm-1].x != hClassEqui[aKIm].x)
                        {
                            pitImage = aKIm;
                            nbCLass++;
                        }

                        hClassEqui[hClassEqui[aKIm].x].y = pitImage;

						const uint2 sizeImage = toUi2(aGLI.getSizeImage());

						dimImgMax = max(dimImgMax,sizeImage);

						IMmGg.Data().SetZoneImage(aKIm, sizeImage, rayonVignette);
                    }

					IMmGg.Data().SyncConstData();

                    // Pour chaque image nous copions les valeurs dans une structure preparatoire pour les envoyés au GPU
                    for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
                    {
                        // Obtention de l'image courante
                        cGPU_LoadedImGeom&	aGLI	= *(mVLI[aKIm]);

                        // Obtention des donnees images
                        float **aDataIm	= aGLI.DataIm0();
                        float*	data	= aGLI.LinDIm0();
                        uint2 dimImg	= toUi2(aGLI.getSizeImage());

                        if(fdataImg1D == NULL)
                            fdataImg1D	= new float[ size(dimImgMax) * mNbIm ];

                        // Copie du tableau 2d des valeurs de l'image Ameliorer encore la copy de texture, copier les images une à  une dans le device!!!!
                        if (aEq(dimImgMax,dimImg))
                            memcpy(  fdataImg1D + size(dimImgMax)* aKIm , data,  size(dimImg) * sizeof(float));
                        else
                            GpGpuTools::Memcpy2Dto1D(aDataIm ,fdataImg1D + size(dimImgMax) * aKIm, dimImgMax, dimImg );

                        //------------------------------------------------------------------------
                        /*
                        std::string nameFile(GpGpuTools::conca("image_0",aKIm));

                        nameFile+=".pgm";

                        float* pImage = fdataImg1D + size(dimImgMax) * aKIm;

                        pixel* pImage = maskIML;

                        float min = GpGpuTools::getMinArray(pImage,dimImgMax);

                        float* dImage =  GpGpuTools::AddArray(pImage,dimImgMax,-min);

                        GpGpuTools::Array1DtoImageFile(dImage,nameFile.c_str(),dimImgMax,1.f/(GpGpuTools::getMaxArray(dImage,dimImgMax)));


                        //GpGpuTools::Array1DtoImageFile(dImage,nameFile.c_str(),dimImgMax,1.f/(65536.f));
                        //GpGpuTools::Array1DtoImageFile(pImage,nameFile.c_str(),dimImgMax,1.f/(2048.f));
                        GpGpuTools::Array1DtoImageFile(maskIML,nameFile.c_str(),IMmGg.DimTerrainGlob());

                        if(!aKIm)
                        {
                            printf("max = %f;",GpGpuTools::getMaxArray(dImage,dimImgMax));
                            printf("min = %f;",GpGpuTools::getMinArray(dImage,dimImgMax));
                        }
                        delete [] dImage;
                        */
                        //------------------------------------------------------------------------

                    }                                        

                    // Copy Images to device (in layered textures)
                    if ((!(oEq(dimImgMax, 0)|(mNbIm == 0))) && (fdataImg1D != NULL))
                        IMmGg.Data().SetImages(fdataImg1D, dimImgMax, mNbIm);

                    // Delete buffer image temporaire
                    if (fdataImg1D != NULL) delete[] fdataImg1D;

                    // Initialisation des paramètres
					IMmGg.SetParameter(mNbIm,rayonVignette , dimImgMax, (float)mAhEpsilon, SAMPLETERR, INTDEFAULT,nbCLass);

                }

                // delete buffers
                delete[] maskGlobal;
                delete[] maskIML;


            }

            //
            // Génération de volumes, où le calcul est nécessaire.

            IMmGg.MaskVolumeBlock().clear();

            if(IMmGg.NoMasked)
            {
                std::vector<Rect> vCellules;
                //OMP_NT1
                for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
                {
                    //OMP_NT2
                    for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
                    {

                        int2 mZ = make_int2(mTabZMin[anY][anX],mTabZMax[anY][anX]);

                        if (IsInTer(anX,anY))
                        {
                            ushort dZ = abs((int)count(mZ));

                            if(mZMaxGlob == -1e7)
                                vCellules.resize(dZ,MAXIRECT);
                            else
                            {
                                if (mZ.x < mZMinGlob)
                                {
                                    vCellules.insert(vCellules.begin(), abs(mZ.x - mZMinGlob),MAXIRECT);
                                    if(mZ.y < mZMinGlob)
                                        dZ = abs(mZ.x - mZMinGlob);
                                }
                                if (mZ.y > mZMaxGlob)
                                {
                                    vCellules.insert(vCellules.end(),   abs(mZ.y - mZMaxGlob),MAXIRECT);
                                    if(mZ.x > mZMaxGlob)
                                    {
                                        mZ.x = mZMaxGlob;
                                        dZ = abs(mZ.y - mZMaxGlob);
                                    }
                                }
                            }

                            ElSetMin(mZMinGlob,mZ.x);
                            ElSetMax(mZMaxGlob,mZ.y);

                            // Pour chaque Z du volume, nous determinons le rectangle minimum
                            for (int i = 0; i < dZ; ++i)
                            {
                                Rect &box = vCellules[i + abs(mZ.x - mZMinGlob)];

                                box.SetMaxMin(anX,anY);

                            }
                        }
                    }
                }

                if(mZMinGlob == 1e7)
                {
                    mZMinGlob = 0;
                    mZMaxGlob = 0;
                }

                uint Dz = abs(mZMaxGlob-mZMinGlob);

                // Pour chaque intervalle Z INTERZ, nous constituons une box
                if(vCellules.size() > 0)
                {
                    uint cellZmaskVol = iDivUp((int)vCellules.size(), INTERZ);
                    uint reste        = Dz - (((uint)vCellules.size()) / INTERZ) * INTERZ  ;

                    IMmGg.MaskVolumeBlock().resize(cellZmaskVol);

                    if(reste != 0)
                    {
                        cellules &celLast = IMmGg.MaskVolumeBlock().back();
                        celLast.Dz = reste;
                    }

                    for (uint i = 0; i < vCellules.size(); ++i)
                    {
                        uint      sI    = i/INTERZ;
                        cellules &cel   = IMmGg.MaskVolumeBlock()[sI];
                        Rect     &Rec   = vCellules[i];

                        cel.Zone.SetMaxMinInc(Rec);

                    }
                }

            }

        }
        else
#endif
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
            Pt2di aSzOrtho = aBox.sz() + mCurSzVMax * 2;
            Pt2di aSzTab =  Pt2di(3,3) + aSzOrtho/mGpuSzD;
            mGeoX.Resize(aSzTab);
            mGeoY.Resize(aSzTab);
            mTGeoX =   TIm2D<REAL4,REAL8>(mGeoX);
            mTGeoY =   TIm2D<REAL4,REAL8>(mGeoY);
        }
}



Fonc_Num SomVoisCreux(Fonc_Num aF,Pt2di aV)
{
   Fonc_Num aRes = 0;
   for (int aK=0 ; aK<9 ; aK++)
   {
      aRes = aRes + trans(aF,TAB_9_NEIGH[aK].mcbyc(aV));
   }
   return aRes;
}

bool  cAppliMICMAC::InitZ(int aZ,eModeInitZ aMode)
{
    mZIntCur =aZ;
    mZTerCur  = DequantZ(mZIntCur);

    mImOkTerCur.raz();

    // XY01-UtiTer => Box of image at Z level , init at empty box
    mX0UtiTer = mX1Ter + 1;
    mY0UtiTer = mY1Ter + 1;
    mX1UtiTer = mX0Ter;
    mY1UtiTer = mY0Ter;

    // Compute Box &  Masq Terrain
    for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
    {
        for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
        {
             // In ortho if in terrain and Z in intervall
             mDOkTer[anY][anX] =
                                   (mZIntCur >= mTabZMin[anY][anX])
                                   && (mZIntCur <  mTabZMax[anY][anX])
                                   && IsInTer(anX,anY)
                                   ;

	     //  If Ok update the box
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

    // If box was not updated, then it is empty
    if (mX0UtiTer >= mX1UtiTer)
            return false;

    int aKFirstIm = 0;
    U_INT1 ** aDOkIm0TerDil = mDOkTerDil;
    // Case mGIm1IsInPax : Im1 doesnt depend of pax (bundle geom of epip like geometry)
    // generate some optimisation
    if (mGIm1IsInPax)
    {
	    // If we have already been here, we dont neet to reload first image
            if (mFirstZIsInit)
            {
               aKFirstIm = 1;
            }
            else
            {
            // First time we must reload the whole first  images 
                mX0UtiTer = mX0Ter;
                mX1UtiTer = mX1Ter;
                mY0UtiTer = mY0Ter;
                mY1UtiTer = mY1Ter;
                aDOkIm0TerDil = mAll1DOkTerDil;
            }
    }

    //   
    // XY01-UtiDilTer => dilatation of  XY01-UtiTer by  mCurSzVMax
    mX0UtiDilTer = mX0UtiTer - mCurSzVMax.x;
    mY0UtiDilTer = mY0UtiTer - mCurSzVMax.y;
    mX1UtiDilTer = mX1UtiTer + mCurSzVMax.x;
    mY1UtiDilTer = mY1UtiTer + mCurSzVMax.y;

    //  XY01-UtiLocIm =>  Box  of image1 in referentiel of I1
    mX0UtiLocIm = mX0UtiTer - mDilX0Ter;
    mX1UtiLocIm = mX1UtiTer - mDilX0Ter;
    mY0UtiLocIm = mY0UtiTer - mDilY0Ter;
    mY1UtiLocIm = mY1UtiTer - mDilY0Ter;

    // XY01-UtiDilLocIm => dilatation of XY01-UtiDilTer
    mX0UtiDilLocIm = mX0UtiDilTer - mDilX0Ter;
    mX1UtiDilLocIm = mX1UtiDilTer - mDilX0Ter;
    mY0UtiDilLocIm = mY0UtiDilTer - mDilY0Ter;
    mY1UtiDilLocIm = mY1UtiDilTer - mDilY0Ter;


    Box2di aBoxUtiLocIm(Pt2di(mX0UtiLocIm,mY0UtiLocIm),Pt2di(mX1UtiLocIm,mY1UtiLocIm));
    Box2di aBoxUtiDilLocIm(Pt2di(mX0UtiDilLocIm,mY0UtiDilLocIm),Pt2di(mX1UtiDilLocIm,mY1UtiDilLocIm));

    Dilate(mImOkTerCur,mImOkTerDil,mCurSzVMax,aBoxUtiDilLocIm);

    cInterpolateurIm2D<float> * anInt = CurEtape()->InterpFloat();

    cGPU_LoadedImGeom * aGLI_00 =  mNbIm ? mVLI[0] : 0 ;
    if (aMode==eModeMom_12_2_22)
    {
          ELISE_ASSERT(aGLI_00!=0,"Incohe eModeMom_12_2_22 with no Im in cAppliMICMAC::InitZ");
    }


    for (int aKIm= aKFirstIm ; aKIm<mNbIm ; aKIm++)
    {
        cGPU_LoadedImGeom & aGLI_0 = *(mVLI[aKIm]);
        const cGeomImage * aGeom=aGLI_0.Geom();


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

        const std::vector<cGPU_LoadedImGeom *> &  aVGLI = aGLI_0.MSGLI();

        for (int aKScale = 0; aKScale<int(aVGLI.size()) ; aKScale++)
        {
             cGPU_LoadedImGeom & aGLI_K = *(aVGLI[aKScale]);

             ELISE_ASSERT(aGLI_0.VDataIm()==aGLI_K.VDataIm(),"Internal incohe in MulScale correl");
             float ** aDataIm =  aGLI_0.VDataIm()[aKScale];
             tGpuF ** aDOrtho = aGLI_K.DataOrtho();
             U_INT1 ** aOkOr =  aGLI_K.DataOKOrtho();

             U_INT1 ** aDLocOkTerDil = (aKIm==0) ? aDOkIm0TerDil : mDOkTerDil;


             // Pendant longtemps, il y a eu un bug quasi invisible, aSzV0=mCurSzV0 ....
             Pt2di aSzV0 =   aGLI_K.SzV0();
             Pt2di aSzErod = mCurSzVMax;

             // Calcul de l'ortho image et de l'image OK Ortho
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


                           // Peu importe aGLI_0 ou aGLI_K
                           if (aGLI_0.IsOk(aPIm.x,aPIm.y))
                           {
                               aDOrtho[anY][anX] = (tGpuF)anInt->GetVal(aDataIm,aPIm);
                               aOkOr[anY][anX] =  1;
                           }
                       }
                       anIndY += aStep;

                   }
                   anIndX += aStep;
             }

             SelfErode(aGLI_K.ImOK_Ortho(), aSzErod ,aBoxUtiLocIm);

             if (    (aMode==eModeMom_2_22)
                  || ((aKIm==0) &&  (aMode==eModeMom_12_2_22))
             )
             {
                   if (! mCMS_ModeEparse)
                   {
                      MomOrdre2(aGLI_K.ImOrtho(),aGLI_K.ImSomO(),aGLI_K.ImSomO2(),aSzV0 ,aBoxUtiLocIm);
                   }
                   else
                   {
                      MomOrdre2_Creux(aGLI_K.ImOrtho(),aGLI_K.ImSomO(),aGLI_K.ImSomO2(),aSzV0 ,aBoxUtiLocIm);

if (0)
{
  double aSomO= 100;
  double aSomO2= 100;
   ELISE_COPY
   (
        rectangle(aBoxUtiLocIm._p0,aBoxUtiLocIm._p1),
        Abs(aGLI_K.ImSomO().in()-SomVoisCreux(aGLI_K.ImOrtho().in(),aSzV0)),
        sigma(aSomO)
   );
   ELISE_COPY
   (
        rectangle(aBoxUtiLocIm._p0,aBoxUtiLocIm._p1),
        Abs(aGLI_K.ImSomO2().in()-SomVoisCreux(Square(aGLI_K.ImOrtho().in()),aSzV0)),
        sigma(aSomO2)
   );
   // std::cout << "Verif SO " << aSomO  << " SO2 " << aSomO2<< "\n";
   ELISE_ASSERT((aSomO<1e-5) && (aSomO2<1e-5),"Check in SO-SO2\n");
}

                   }
             }
             else if (aMode==eModeMom_12_2_22)
             {
                   // std::cout << "KIM " << aKIm << "\n";
                   if (! mCMS_ModeEparse)
                   {
                       Mom12_22
                       (
                             aGLI_00->KiemeMSGLI(aKScale)->ImOrtho(),
                             aGLI_K.ImOrtho(),
                             aGLI_K.ImSom12(),
                             aGLI_K.ImSomO(),
                             aGLI_K.ImSomO2(),
                             aSzV0 ,
                             aBoxUtiLocIm
                       );
                   }
                   else
                   {
                       Mom12_22_creux
                       (
                             aGLI_00->KiemeMSGLI(aKScale)->ImOrtho(),
                             aGLI_K.ImOrtho(),
                             aGLI_K.ImSom12(),
                             aGLI_K.ImSomO(),
                             aGLI_K.ImSomO2(),
                             aSzV0 ,
                             aBoxUtiLocIm
                       );
                   }
            }
        }
    }

    mFirstZIsInit = true;

    return true;
}

void cAppliMICMAC::DoOneCorrelSym(int anX,int anY,int aNbScaleIm)
{

static int aCpt=0 ; aCpt++;

     double aCost = mAhDefCost;
     std::vector<cGPU_LoadedImGeom *> aCurVLI;
     for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
     {
          cGPU_LoadedImGeom * aGLI = (mVLI[aKIm]);
          if (aGLI->InitValNorms(anX,anY,aNbScaleIm))
          {
              aCurVLI.push_back(aGLI);
          }
     }
     int aNbImCur = (int)aCurVLI.size();
     if (aNbImCur >= 2)
     {
         double anEC2 = 0;
         if (mCMS)
         {
             for (int aKS=0 ; aKS<aNbScaleIm ; aKS++)
             {
                  std::vector<cGPU_LoadedImGeom *> aVSLIm;
                  for (int aKIm=0 ; aKIm<aNbImCur ; aKIm++)
                      aVSLIm.push_back(aCurVLI[aKIm]->KiemeMSGLI(aKS));

                  Pt2di aSzV0 = aVSLIm[0]->SzV0();
                  double aPds = aVSLIm[0]->PdsMS();
                  for (int aDx=-aSzV0.x ;aDx<=aSzV0.x ; aDx+=aSzV0.x)
                  {
                      for (int aDy=-aSzV0.y ;aDy<=aSzV0.y ; aDy+=aSzV0.y)
                      {
                           int aXV = anX+aDx;
                           int aYV = anY+aDy;
                           double aSV = 0;
                           double aSVV = 0;
                           for (int aKIm=0 ; aKIm<aNbImCur ; aKIm++)
                           {
                                double aV = aVSLIm[aKIm]->ValNorm(aXV,aYV);
if (0)
{
  std::cout << "VvV[" << aKIm << "]"
            << " O  " <<  aVSLIm[aKIm]->ValOrthoBasik(aXV,aYV)
            << " So " << aVSLIm[aKIm]->SumO(anX,anY)/9.0
            << " Mi " << aVSLIm[aKIm]->MoyIm(anX,anY,aNbScaleIm)
            << " " << aVSLIm[aKIm]->ValNorm(anX,anY)
            << " " << aVSLIm[aKIm]->MoyCal() << "\n";
}
                                aSV += aV;
                                aSVV += QSquare(aV) ;
                           }
                           anEC2 += (aSVV-QSquare(aSV)/aNbImCur) * aPds;
                      }
                  }
             }
             aCost = anEC2 / ((aNbImCur -1) *  aCurVLI[0]->KiemeMSGLI(aNbScaleIm-1)->CumSomPdsMS ());

if (0)
{
   double aCorStd = 1-aCost ;
   RMat_Inertie aMat;
   Pt2di aSzV0 = aCurVLI[0]->SzV0();
   for (int aDx=-aSzV0.x ;aDx<=aSzV0.x ; aDx+=aSzV0.x)
   {
        for (int aDy=-aSzV0.y ;aDy<=aSzV0.y ; aDy+=aSzV0.y)
        {
              Pt2di aP(anX+aDx,anY+aDy);
              double aV1 = aCurVLI[0]->ValOrthoBasik(aP.x,aP.y);
              double aV2 = aCurVLI[1]->ValOrthoBasik(aP.x,aP.y);

              aMat.add_pt_en_place(aV1,aV2);
        }
   }
   double aDif = ElAbs(aCorStd-aMat.correlation());
   if (aDif >1e-4)
   {
       std::cout << "VeerifCOR " << aCorStd << " " << aMat.correlation() << " D=" << aDif << "\n";
       std::cout << "CPT " << aCpt << "\n";
       getchar();
       // ELISE_ASSERT(false,"VeerifCOR");
   }
}
//std::cout << "CMS " << anEC2 << "\n";
         }
         else
         {
             int aX0 = anX - mCurSzV0.x;
             int aX1 = anX + mCurSzV0.x;
             int aY0 = anY - mCurSzV0.x;
             int aY1 = anY + mCurSzV0.x;

             std::string mode = "normal";
//             /* NORMAL
             std::vector<float> imageM;
             for (int aXV=aX0 ; aXV<=aX1 ; aXV++)
             {
                  for (int aYV=aY0 ; aYV<=aY1 ; aYV++)
                  {
                       double aSV = 0;
                       double aSVV = 0;
                       std::vector<double> vectMediane;
                       for (int aKIm=0 ; aKIm<aNbImCur ; aKIm++)
                       {
                            double aV = aCurVLI[aKIm]->ValNorm(aXV,aYV);
// std::cout << "VvV = " << aV << "\n";
                            aSV += aV;
                            aSVV += QSquare(aV) ;
                            vectMediane.push_back(aV);
                       }

                       if(mode=="normal")
                           anEC2 += (aSVV-QSquare(aSV)/aNbImCur);
                       else if(mode=="moyenne")
                       {
                           aSV/=aNbImCur;
                           imageM.push_back(aSV);
                       }
                       else if (mode == "mediane")
                       {
                           std::sort(vectMediane.begin(), vectMediane.end());
                           if (vectMediane.size()%2==0)
                               imageM.push_back((vectMediane[vectMediane.size()/2]+vectMediane[vectMediane.size()/2-1])/2);
                           else
                               imageM.push_back(vectMediane[(vectMediane.size()-1)/2]);
                       }
                  }
             }

// std::cout << "NOCMS " << anEC2 << "\n";

             if(mode=="normal")
                 aCost = anEC2 / ((aNbImCur -1) * mNbPtsWFixe);
             else
             {
                 double aSVmoy = 0;
                 double aSVVmoy = 0;
                 for (size_t aI=0 ; aI<imageM.size();++aI)
                 {
                     aSVmoy += imageM[aI];
                     aSVVmoy += QSquare(imageM[aI]);
                 }
                 aCost = (aSVVmoy-QSquare(aSVmoy)/aNbImCur)/((aNbImCur -1) *mNbPtsWFixe);

                 if (mode == "moyenne") std::cout << "aCost MOYENNE " << aCost << std::endl;
                 if (mode == "mediane") std::cout << "aCost MEDIANE " << aCost << std::endl;
             }
          }

if (0)
{
   static double aCMax=-10;
   static double aCMin= 10;
   static int aCpt = 0 ; aCpt++;


   if ((aCost>aCMax) || (aCost<aCMin))
   {
       aCMax = ElMax(aCost,aCMax);
       aCMin = ElMin(aCost,aCMin);
       std::cout << "COST " << aCMin << " " << aCMax << "\n";
   }
/*
*/
}

          aCost =  mStatGlob->CorrelToCout(1-aCost);
     }
     mSurfOpt->SetCout(Pt2di(anX,anY),&mZIntCur,aCost);
}

double EcartNormalise(double aI1,double aI2)
{
    // X = I1/I2
    if (aI1 < aI2)   // X < 1
        return aI1/aI2 -1;   // X -1
    // 0<= aI2 <= aI1
    if (aI1==0)
    {
       return 0;
    }

    return 1-aI2/aI1;  // 1 -1/X
}

const double MCPMulCorel = 1.0;


void cAppliMICMAC::DoOneCorrelIm1Maitre(int anX,int anY,const cMultiCorrelPonctuel * aCMP,int aNbScaleIm,bool VireExtre,double aPdsAttPix)
{
    int aNbOk = 0;
    double aSomCorrel = 0;
    double aPdsCorrStd=1.0;

    if (mVLI[0]->OkOrtho(anX,anY))
    {
        double aCMax = -2;
        double aCMin = 2;
        for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
        {
             double aCor;
             if (mVLI[aKIm]->Correl(aCor,anX,anY,*(mVLI[0]),aNbScaleIm))
             {
                 aNbOk ++;
                 aSomCorrel += aCor;
                 ElSetMax(aCMax,aCor);
                 ElSetMin(aCMin,aCor);
             }
        }
        if (VireExtre && (aNbOk>2))
        {
            aSomCorrel -= aCMax + aCMin;
            aNbOk -= 2;
        }
    }

    if (aCMP)
    {
        aPdsCorrStd = aCMP->PdsCorrelStd();
    }

    double aCost = aNbOk ? (mStatGlob->CorrelToCout(aSomCorrel/aNbOk) * aPdsCorrStd): mAhDefCost;

    if (mDoStatCorrel)
    {
        if (aNbOk)
           mStatCNC.push_back(mStatGlob->CorrelToCout(aSomCorrel/aNbOk));
    }

    double aCostPix=0.0;
    int    aNbCostPix = 0;
    if (aCMP)
    {

        std::vector<tMCPVal> aVNorm;
        if (mVLI[0]->OkOrtho(anX,anY))
        {
             tGpuF aV0 = mVLI[0]->ImOrtho(anX,anY);
             for (int aK=1 ; aK<mNbIm ; aK++)
             {
                  if (mVLI[aK]->OkOrtho(anX,anY))
                  {
                       double aVk = mVLI[aK]->ImOrtho(anX,anY);
                       double aVal = EcartNormalise(aV0,aVk);

                       aVNorm.push_back(AdaptCostPonct(round_ni(aVal*TheDynMCP*MCPMulCorel)));
                       if (aPdsAttPix)
                       {
                           aNbCostPix++;
			
			               double aVCorK = mVLI[aK]->CorrRadiom(aVk,mGeomDFPx->RDiscToR2(Pt2dr(anX,anY)));
                           aCostPix += ElAbs(EcartNormalise(aVCorK,aV0));

			               if(ERupnik_MM())
                           {
        			   	        std::cout << "ewelina, " << mVLI[aK]->PDV()->Name()  << ", PTer=" << mGeomDFPx->RDiscToR2(Pt2dr(anX,anY)) 
                                                         << ", aVk=" << aVk << ", aVCorK=" << aVCorK << ", Cor=" << aVk/aVCorK << "\n";
                           }
                       }
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
        mSurfOpt->Local_VecMCP(Pt2di(anX,anY),&mZIntCur,aVNorm);
    }

    if (aNbCostPix)
    {
      aCost += (aCostPix / aNbCostPix) * aPdsAttPix;
      if (mDoStatCorrel)
      {
           mStat1Pix.push_back(aCostPix/aNbCostPix);
      }
    }

    mSurfOpt->SetCout
    (
         Pt2di(anX,anY),
         &mZIntCur,
         aCost
    );
}



void cAppliMICMAC::DoOneCorrelMaxMinIm1Maitre(int anX,int anY,bool aModeMax,int aNbScaleIm)
{
        if (mEBI) // Etiq Best Image
        {
            if (mNbIm>1)
            {
                for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
                {
                    double aCor;
                    bool Ok = mVLI[aKIm]->Correl(aCor,anX,anY,*(mVLI[0]),aNbScaleIm);
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
            double aRes =  aModeMax ? -2 : 2;
                        bool isOk = false;

            if (mVLI[0]->OkOrtho(anX,anY))
            {
                for (int aKIm=1 ; aKIm<mNbIm ; aKIm++)
                {
                    double aCor;
                    if (mVLI[aKIm]->Correl(aCor,anX,anY,*(mVLI[0]),aNbScaleIm))
                    {
                        if (aModeMax)
                                                   ElSetMax(aRes,aCor);
                                                 else
                                                   ElSetMin(aRes,aCor);
                                                isOk = true;
                    }
                }
            }

            mSurfOpt->SetCout
                (
                Pt2di(anX,anY),
                &mZIntCur,
                (isOk) ? mStatGlob->CorrelToCout(aRes) : mAhDefCost
                );
        }
}



void cAppliMICMAC::DoGPU_Correl
        (
            const Box2di & aBox,
            const cMultiCorrelPonctuel * aMCP,
            double aPdsPix
        )
{
        eModeInitZ aModeInitZ = eModeMom_2_22;
        eModeAggregCorr aModeAgr = mCurEtape->EtapeMEC().AggregCorr().Val();

        if (aMCP)
        {
            ELISE_ASSERT(IsModeIm1Maitre(aModeAgr),"MultiCorrelPonctuel requires eAggregIm1Maitre");
        }

        if (aModeAgr==eAggregSymetrique)
        {
        }
        else if (IsModeIm1Maitre(aModeAgr))
        {
            aModeInitZ = eModeMom_12_2_22;
        }
        else
        {
            ELISE_ASSERT(false,"Unsupported Mode Aggreg in cAppliMICMAC::DoGPU_Correl");
        }


        if (mCMS)
        {
            if (! IsModeIm1Maitre(aModeAgr))
            {
                ELISE_ASSERT
                (
                    mCMS_ModeEparse && (! mCurEtUseWAdapt),
                    "Muliscale in ground geom requires : (! ModeDense) && (! UseWAdapt)"
                );
            }
        }

	if (true)
	{
	     // Standard case do the computation in the current process
             for (int aZ=mZMinGlob ; aZ<mZMaxGlob ; aZ++)
             {
                 bool OkZ = InitZ(aZ,aModeInitZ);
                 if (OkZ)
                 {
                     for (int anX = mX0UtiTer ; anX <  mX1UtiTer ; anX++)
                     {
                         for (int anY = mY0UtiTer ; anY < mY1UtiTer ; anY++)
                         {

                             int aNbScaleIm =  NbScaleOfPt(anX,anY);

                             if (mCurEtUseWAdapt)
                             {
                                  ElSetMin(aNbScaleIm,1+mTImSzWCor.get(Pt2di(anX,anY)));
                             }
                             if (mDOkTer[anY][anX])
                             {

                                 switch (aModeAgr)
                                 {
                                     case eAggregSymetrique :
                                         DoOneCorrelSym(anX,anY,aNbScaleIm);
                                     break;

                                     case eAggregIm1Maitre :
                                          DoOneCorrelIm1Maitre(anX,anY,aMCP,aNbScaleIm,false,aPdsPix);
                                     break;

                                     case  eAggregMaxIm1Maitre :
                                         DoOneCorrelMaxMinIm1Maitre(anX,anY,true,aNbScaleIm);
                                     break;

                                     case  eAggregMinIm1Maitre :
                                         DoOneCorrelMaxMinIm1Maitre(anX,anY,false,aNbScaleIm);
                                     break;

                                     case eAggregMoyMedIm1Maitre :
                                          DoOneCorrelIm1Maitre(anX,anY,aMCP,aNbScaleIm,true,aPdsPix);
                                     break;

                                 default :
                                     break;
                                 }
                             }
                         }
                     }
                 }
             }
	}
	// Case where we generate the ortho photos and call processes
	else
	{
		// Moh
             int mDeltaZ = 10;
             std::string aPrefixGlob = "MMV1Ortho_Pid" + ToString(mm_getpid()) ;
             for (int aZ0=mZMinGlob ; aZ0<mZMaxGlob ; aZ0+=mDeltaZ)
             {
                  int aZ1= ElMin(mZMaxGlob,aZ0+mDeltaZ);
                  for (int aZ=aZ0 ; aZ<aZ1 ; aZ++)
                  {
                        std::string aPrefixZ =    aPrefixGlob + "_Z" + ToString(aZ-mZMaxGlob) ;
                        bool OkZ = InitZ(aZ,aModeInitZ);
			if (OkZ)
                        {
                            //SaveIm(aPrefixZ+"Masq.tif",
//template <class Type,class TBase> void  SaveIm(const std::string & aName,Im2D<Type,TBase> anIm,const Box2di & aBox)
			    for (int aKIm=0 ; aKIm<int(mVLI.size()) ; aKIm++)
                            {
                            }
                        }
			else
			{
                            std::string aNameNone  = aPrefixZ + "_NoData";
			    ELISE_fp aFile(aNameNone.c_str(),ELISE_fp::WRITE);
			    aFile.close();
std::cout << aNameNone ;
			}
		  }
             }
	}
}

#ifdef  CUDA_ENABLED

///
/// \brief cAppliMICMAC::Tabul_Projection Pré-calcul des projections des points terrains dans chaque images
/// \param Z        Z initiale
/// \param interZ   interval de Z à pré-calculé
/// \param idBuf    id buffer de pré-calcul
///
	void cAppliMICMAC::Tabul_Projection(short Z, ushort &interZ, ushort idBuf)
    {
#ifdef  NVTOOLS
        GpGpuTools::NvtxR_Push(__FUNCTION__,0xFFAA0033);
#endif
        IMmGg.Data().MemsetHostVolumeProj(IMmGg.Param(idBuf).invPC.IntDefault);

		float2*		pTabProj	=	IMmGg.Data().HostVolumeProj();      // Pointeur sur le buffer des projections
		const Rect  zone        =	IMmGg.Param(idBuf).RDTer();         // Zone Terrain dilaté
		const uint  sample      =	IMmGg.Param(idBuf).invPC.sampProj;  // Sample
		const uint2	dimTabProj	=	zone.dimension();					// Dimension de la zone terrain
		const uint2	dimSTabProj	=	iDivUp(dimTabProj,sample)+1;		// Dimension de la zone terrain sous-echantilloné
		const int2	anB			=	zone.pt0 +  dimSTabProj * sample;

		const Pt2dr stepPlaniSa(mStepPlani.x*sample,mStepPlani.y*sample);
		const Pt2dr cpTerDequan(mOriPlani.x + mStepPlani.x*zone.pt0.x,mOriPlani.y + mStepPlani.y*zone.pt0.y);

		float2* buf_proj		= pTabProj ;//+ (rZ  + aKIm )* sizSTabProj;	// Buffer des projections pre-calculées
		OMP_NT1
		for (short anZ = Z; anZ < (Z + interZ); ++anZ)
		{
			const double aZReel	= DequantZ(anZ);                                                    // Dequantification Z
			OMP_NT2
			for (ushort aKIm = 0 ; aKIm < mNbIm ; ++aKIm)        // Mise en calque des projections pour chaque image
            {
				const cGeomImage* aGeom = ((cGPU_LoadedImGeom*)mVLI[aKIm])->Geom();	// geom image
				int2  pTer;                                 // Debut de la zone de pré-calcul
				Pt2dr pTerDequan;

				for ( pTer.y = zone.pt0.y, pTerDequan.y = cpTerDequan.y; pTer.y < anB.y; pTer.y += sample, pTerDequan.y +=  stepPlaniSa.y)							// Ballayage du terrain
				{
					for (pTer.x = zone.pt0.x, pTerDequan.x = cpTerDequan.x; pTer.x < anB.x ; pTer.x += sample,++buf_proj, pTerDequan.x +=  stepPlaniSa.x)
					{
						const Pt2dr aPIm  = aGeom->CurObj2Im(pTerDequan,&aZReel);		// Calcul de la projection dans l'image aKIm

						//if (aGLI.IsOk( aPIm.x, aPIm.y )) // Le masque image !!
						*buf_proj		= make_float2((float)aPIm.x,(float)aPIm.y);		// affectation dans le

					}
				}
            }
        }
#ifdef  NVTOOLS
		GpGpuTools::Nvtx_RangePop();
#endif
    }


    void cAppliMICMAC::Tabul_Images(int Z, uint &interZ, ushort idBuf)
    {

        CuHostData3D<float> hoValuesImages;

        Rect    zone        = IMmGg.Param(idBuf).RTer();           // Zone Terrain dilaté
        zone.out();
        cout << endl;

        uint2	dimTabProj	= zone.dimension();						// Dimension de la zone terrain
        uint	sizSTabProj	= size(dimTabProj);					// Taille de la zone terrain sous-echantilloné

        int2	anB			= zone.pt0 +  dimTabProj;


        DUMP(anB)
        DUMP(dimTabProj)
        DUMP(IMmGg.Param(idBuf).RDTer().dimension())
        DUMP(IMmGg.Param(idBuf).RTer().dimension())

                IMmGg.Param(idBuf).RTer().out();
                cout << endl;

        hoValuesImages.Malloc(dimTabProj,IMmGg.Param(idBuf).invPC.nbImages*interZ);
        float  *pBufVimg   = hoValuesImages.pData();        // Pointeur sur le buffer des projections

        cInterpolateurIm2D<float> * anInt = CurEtape()->InterpFloat();

        for (int anZ = Z; anZ < (int)(Z + interZ); anZ++)
        {
            int rZ = abs(Z - anZ) * mNbIm;

            for (int aKIm = 0 ; aKIm < mNbIm ; aKIm++ )                     // Mise en calque des projections pour chaque image
            {

                float* buf_ValImages     = pBufVimg + (rZ  + aKIm )* sizSTabProj;// Buffer des projections pre-calculées

                cGPU_LoadedImGeom&	aGLI	= *(mVLI[aKIm]);                // Obtention de l'image aKIm
                const cGeomImage*	aGeom	= aGLI.Geom();

                int2 pTer       = zone.pt0;                                 // Debut de la zone de pré-calcul
                int2 sampTer    = make_int2(0,0);                           // Point retenu
                const double aZReel	= DequantZ(anZ);                                                    // Dequantification Z

                for (pTer.y = zone.pt0.y; pTer.y < anB.y; pTer.y ++, sampTer.y++, sampTer.x = 0)	// Ballayage du terrain

                    for (pTer.x = zone.pt0.x; pTer.x < anB.x ; pTer.x ++, sampTer.x++)
                    {

                        Pt2dr aPTer	= DequantPlani(pTer.x,pTer.y);          // Dequantification  de X, Y
                        Pt2dr aPIm  = aGeom->CurObj2Im(aPTer,&aZReel);      // Calcul de la projection dans l'image aKIm

                        if (aGLI.IsOk( aPIm.x, aPIm.y ))
                            buf_ValImages[to1D(sampTer,dimTabProj)]		= anInt->GetVal(aGLI.DataIm0(),aPIm); // affectation dans le
                        else
                            buf_ValImages[to1D(sampTer,dimTabProj)]		= -1;
                    }
            }
        }


        hoValuesImages.OutputValues(0,XY,Rect(0,0,20,hoValuesImages.GetDimension().y),6);
        hoValuesImages.Dealloc();
    }

	void cAppliMICMAC::setVolumeCost( short z0, short z1,ushort idBuf)
    {
#ifdef  NVTOOLS
        GpGpuTools::NvtxR_Push(__FUNCTION__,0x335A8833);
#endif
		float*		tabCost     = IMmGg.VolumeCost(idBuf);
		const Rect  zone        = IMmGg.Param(idBuf).RTer();
		const float valdefault  = IMmGg.Param(idBuf).invPC.floatDefault;
		const uint2 rDiTer		= zone.dimension();
		const uint  rSiTer		= size(rDiTer);

		OMP_NT1
		for (int anY = zone.pt0.y ; anY < zone.pt1.y; anY++)
		{
			OMP_NT2
			for (int anX = zone.pt0.x ; anX <  zone.pt1.x ; anX++,tabCost++)
			{
				const short anZ0 = max(z0,mTabZMin[anY][anX]);
				const short anZ1 = min(z1,mTabZMax[anY][anX]);
				mNbPointsIsole += abs(anZ1-anZ0);

				float *tCost =  tabCost + rSiTer * abs(anZ0 - (int)z0);

				for (int anZ = anZ0;  anZ < anZ1 ; anZ++,tCost+=rSiTer)
                {

					const double cost = (double)(*tCost);

                    // TODO WARNING les couts init sont stockés dans un ushort mais des couts semblent sup à ushortmax!!!!
					mSurfOpt->SetCout(Pt2di(anX,anY),&anZ, cost != valdefault ? cost : mAhDefCost);
                }
            }
        }

#ifdef  NVTOOLS
			GpGpuTools::Nvtx_RangePop();
#endif

    }

#endif

    void cAppliMICMAC::DoGPU_Correl_Basik
        (
        const Box2di & aBox
        )
    {

#ifdef  CUDA_ENABLED

        // Si le terrain est masque ou aucune image : Aucun calcul
        if (mNbIm == 0 || IMmGg.MaskVolumeBlock().size() == 0) return;

        // Initiation du calcul
        uint interZ = IMmGg.InitCorrelJob(mZMinGlob,mZMaxGlob);

		short anZProjection = mZMinGlob, anZComputed= mZMinGlob;//, ZtoCopy = 0;

        bool idPreBuf = false;

        IMmGg.SetCompute(true);

        int nbCellZ = IMmGg.MaskVolumeBlock().size();

		// Parcourt de l'intervalle de Z compris dans la nappe globale
		if (IMmGg.UseMultiThreading())
		{
			//while( anZComputed < mZMaxGlob )
			int aKCellZ      = 0;
			int aKPreCellZ   = 0;
			while( aKCellZ < nbCellZ )
            {

                // Tabulation des projections si la demande est faite

                //if ( IMmGg.GetPreComp() && anZProjection <= anZComputed + (int)interZ && anZProjection < mZMaxGlob)
                if( aKPreCellZ <= aKCellZ + 1 && aKPreCellZ < nbCellZ &&  IMmGg.GetPreComp() )
                {

                    cellules Mask = IMmGg.MaskVolumeBlock()[aKPreCellZ];

                    IMmGg.Param(idPreBuf).SetDimension(Mask.Zone,Mask.Dz);

                    IMmGg.ReallocHostData(Mask.Dz,idPreBuf);
                    //Tabul_Images(anZProjection, Mask.Dz,idPreBuf);
                    Tabul_Projection( anZProjection, Mask.Dz,idPreBuf);


                    //IMmGg.signalComputeCorrel(Mask.Dz);
                    IMmGg.SetPreComp(false);
                    IMmGg.simpleJob();                    

                    anZProjection+= Mask.Dz;
                    aKPreCellZ++;
                    idPreBuf = !idPreBuf;
                }
                // Affectation des couts si des nouveaux ont ete calcule!

                if (IMmGg.GetDataToCopy())
                {
                    uint ZtoCopy = IMmGg.Param(!IMmGg.GetIdBuf()).ZCInter;
                    setVolumeCost(anZComputed,anZComputed + ZtoCopy,!IMmGg.GetIdBuf());
                    IMmGg.SetDataToCopy(false);
                    anZComputed += ZtoCopy;
                    aKCellZ++;
                }
            }
		}
        else
        {            
            while( anZComputed < mZMaxGlob )
            {
                cellules Mask = IMmGg.MaskVolumeBlock()[abs(anZComputed-mZMinGlob)/INTERZ];

                IMmGg.Param(idPreBuf).SetDimension(Mask.Zone,Mask.Dz);

                IMmGg.ReallocHostData(Mask.Dz,idPreBuf);

                // calcul des projections
                Tabul_Projection( anZComputed,Mask.Dz,0);

                // Kernel Correlation
                IMmGg.BasicCorrelation();

                setVolumeCost(anZComputed,anZComputed + interZ,0);

                anZComputed += interZ;
            }
        }

        IMmGg.freezeCompute();

//        IMmGg.Data().DeallocDeviceData();
//        IMmGg.Data().DeallocHostData();

#else
        ELISE_ASSERT(1,"Sorry, this is not the cuda version");
#endif

    }

void cAppliMICMAC::DoCorrelAdHoc
     (
        const Box2di & aBox
     )
{

      const cTypeCAH & aTC  = mCorrelAdHoc->TypeCAH();
      if (mEBI)
      {
            ELISE_ASSERT
            (
                   mCurEtape->EtapeMEC().AggregCorr().Val() == eAggregMaxIm1Maitre,
                   "EtiqBestImage requires eAggregMaxIm1Maitre,"
            );
            ELISE_ASSERT(aTC.GPU_Correl().IsInit(),"EtiqBestImage requires GPU_Correl");
            /// ELISE_ASSERT(mNb_PDVBoxInterne>,);
      }


        DoInitAdHoc(aBox);

        mCC = aTC.CensusCost().PtrVal();

	if (aTC.ScoreLearnedMMVII().IsInit())
        {
            DoCostLearnedMMVII(aBox,aTC.ScoreLearnedMMVII().Val());
        }
        else if (aTC.GPU_Correl().IsInit())
        {
            DoGPU_Correl(aBox,(cMultiCorrelPonctuel*)0,0);
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
        else if (aTC.MasqueAutoByTieP().IsInit())
        {
            DoMasqueAutoByTieP(aBox,aTC.MasqueAutoByTieP().Val());
        }
        else if (mCC) // (aTC.CensusCost().IsInit())
        {
             ELISE_ASSERT
             (
                 ModeGeomIsIm1InvarPx(*this) ,
                 "Census require ModeGeomIm for now"
             );

             if (GeomImages() == eGeomImage_EpipolairePure)
             {
                DoCensusCorrel(aBox,aTC.CensusCost().Val());
             }
             else
             {
                DoGPU_Correl(aBox,nullptr,0);
                // ELISE_ASSERT ( false, "Not epipolar geometry for census ");
             }
        }

        // On peut avoir a la fois MCP et mCC (par ex)
        if (aTC.MultiCorrelPonctuel().IsInit())
        {
            const cMultiCorrelPonctuel * aMCP = aTC.MultiCorrelPonctuel().PtrVal();
            const cMCP_AttachePixel * aAP = aMCP->MCP_AttachePixel().PtrVal();
            double aPdsPix= 0 ;
            if (aAP)
            {
               aPdsPix=  aAP->Pds() * MCPMulCorel;
               for (int aKIm= 0 ; aKIm<int(mVLI.size()) ; aKIm++)
               {
                    std::string aName = mVLI[aKIm]->PDV()->Name();
                    cGLI_CalibRadiom *  aCalR = mDicCalRad[aName];
                    if (aCalR==0)
                    {
                       std::string aNameF = mICNM->Assoc1To1(aAP->KeyRatio(),aName,true);
                       cGLI_CalibRadiom * aCal = 0;
                       if (StdPostfix(aNameF)=="xml")
                       {
                            cXML_RatioCorrImage aXMLR = StdGetFromMM(aNameF,XML_RatioCorrImage);
                            aCal = new cGLI_CalibRadiom(aXMLR);
                            // std::cout << "RRRR " << aName<< " " << aXMLR.Ratio() << "\n";
                       }
                       else
                       {
			   aCal = new cGLI_CalibRadiom(aNameF); 
                       }
                       mDicCalRad[aName] = aCal;
                    }
                    aCalR = mDicCalRad[aName];
                    mVLI[aKIm]->InitCalibRadiom(aCalR);
               }
            }
            DoGPU_Correl(aBox,aMCP,aPdsPix);
        }

}

void ShowStat(const std::string & aMes,std::vector<float> & aVC)
{
   if(aVC.empty())
   {
        std::cout << aMes << " empty" << "\n";
        return;
   }

   double aV0 = KthValProp(aVC,0.25);
   double aV1 = KthValProp(aVC,0.75);

   std::cout  << aMes 
              << " Ecart " << (aV1-aV0)  
              << " Aver " << (aV0+aV1)/2.0  
              << " Nb " << aVC.size() << "\n";
}

void cAppliMICMAC::GlobDoCorrelAdHoc
        (
        const Box2di & aBoxOut,
        const Box2di & aBoxIn  //  IN
        )
{
        ResetCalRad();

        int aSzDecoupe = mCorrelAdHoc->SzBlocAH().Val();
        // Pour eventuellement changer si existe algo qui impose une taille
        mCurSzV0   = mPtSzWFixe;
        mCurSzVMax = mCurSzV0;
        if (mCMS)
        {
            ELISE_ASSERT
            (
                         mCurSzV0 == mCMS->OneParamCMS()[0].SzW(),
                        "Incoherence in first SzW of OneParamCMS"
            );
            mCurSzVMax = mCurSzV0;
            for (int aK=1 ; aK<int(mCMS->OneParamCMS().size()) ; aK++)
                mCurSzVMax.SetSup(mCMS->OneParamCMS()[aK].SzW());
        }
        const cTypeCAH & aTC  = mCorrelAdHoc->TypeCAH();

        if ((aTC.CensusCost().IsInit()) &&  (GeomImages() == eGeomImage_EpipolairePure))
        {
            int aK=0;
            for
            (
                tCsteIterPDV itFI=mPDVBoxGlobAct.begin();
                itFI!=mPDVBoxGlobAct.end();
                itFI++
            )
            {
                 Pt2di aP0(-mCurSzVMax.x,-mCurSzVMax.y);
                 Pt2di aP1(mCurSzVMax.x+1,mCurSzVMax.y);
                 Box2di aB(aP0,aP1);
                 (*itFI)->LoadedIm().DoMasqErod(aB);
                 aK++;
            }
            mBufCensusIm2.clear();
            mVBufC.clear();
            const std::vector<cMSLoadedIm> & aVSLI = mPDV2->LoadedIm().MSLI();
            for (int aK=0 ; aK<int(aVSLI.size()) ; aK++)
            {
                  const Im2D_REAL4 * anI = aVSLI[aK].Im();
                  Pt2di aSz= anI->sz();
                  mBufCensusIm2.push_back(Im2D_REAL4(aSz.x,aSz.y,0.0));
                  mBufCensusIm2.back().dup(*anI);
                  mVBufC.push_back(mBufCensusIm2.back().data());
            }
            mDataBufC = &(mVBufC[0]);
            // Pour census, afin de faciliter et (marginalement ?) accelerer l'exe, on ne fait qu'une seule boite
            aSzDecoupe = 1000000;
        }

        mMCP = aTC.MultiCorrelPonctuel().PtrVal();
        mDoStatCorrel = false;
        if (mMCP)
        {
            mStatCNC.clear();
            mStat1Pix.clear();
            mDoStatCorrel = true;
        }


        if (aTC.Correl2DLeastSquare().IsInit())
        {
            // ELISE_ASSERT(AlgoRegul()==eAlgoLeastSQ
            DoCorrelLeastQuare(aBoxOut,aBoxIn,aTC.Correl2DLeastSquare().Val());
            return;
        }



        mEpsAddMoy  =  mCorrelAdHoc->EpsilonAddMoyenne().Val();
        mEpsMulMoy  =  mCorrelAdHoc->EpsilonMulMoyenne().Val();


        mSzGlobTer = aBoxIn.sz();

        cDecoupageInterv2D aDecInterv = cDecoupageInterv2D::SimpleDec ( aBoxIn.sz(), aSzDecoupe, 0);

#if CUDA_ENABLED
        IMmGg.DimTerrainGlob().x = aBoxIn.sz().x;
        IMmGg.DimTerrainGlob().y = aBoxIn.sz().y;
        IMmGg.SetProgress(aDecInterv.NbInterv());
#endif
        for (int aKBox=0 ; aKBox<aDecInterv.NbInterv() ; aKBox++)
        {
            DoCorrelAdHoc(aDecInterv.KthIntervOut(aKBox));
            #if CUDA_ENABLED
                IMmGg.IncProgress();
            #endif
        }

        if (mDoStatCorrel)
        {
             ShowStat("Corr NC", mStatCNC);
             ShowStat("1Pix Match", mStat1Pix);
        }
}


/*@}*/


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
