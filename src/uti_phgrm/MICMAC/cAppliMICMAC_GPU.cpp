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

//static bool USE_EPIP=false;
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

template <class Type> void  SaveIm(const std::string & aName,Type ** aDataIn,const Box2di & aBoxIn)
{
   Pt2di aP0 = aBoxIn._p0;
   Pt2di aSz = aBoxIn.sz();

   // Create a temporary image
   Im2D<Type,typename El_CTypeTraits<Type>::tBase >  anImOut(aSz.x,aSz.y);
   Type ** aDataOut = anImOut.data();


   for (int aY=0 ; aY<aSz.y ; aY++)
   {
       memcpy
       (
	    aDataOut[aY],
	    aDataIn[aY+aP0.y]+aP0.x,
	    aSz.x * sizeof(Type)
       );
   }

   L_Arg_Opt_Tiff aLArg;
   aLArg = aLArg + Arg_Tiff(Tiff_Im::ANoStrip());

   Tiff_Im::CreateFromIm(anImOut,aName,aLArg);
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


void cAppliMICMAC::GenerateBoxesImEpip_EpipIm(std::vector<cGPU_LoadedImGeom *> & aVLI, std::string & aNameOrig)
{
  bool IsStenope= (GeomImages()==eGeomImageOri);
  bool IsGen    =  (GeomImages()==eGeomGen);

  if (IsStenope) GenerateGeoPassage_ImEpip_EpipIm_BBox_Stenope(aVLI,aNameOrig);
  //if (IsStenope) GenerateGeoPassage_Homography_BBox_Stenope(aVLI,aNameOrig);
  if (IsGen)     GenerateGeoPassage_ImEpip_EpipIm_BBox(aVLI,aNameOrig);
}

void cAppliMICMAC::GenerateGeoPassage_ImEpip_EpipIm_BBox_Stenope(std::vector<cGPU_LoadedImGeom *> & aVLI, std::string & aNameOrig)
{
  /*TODO: Include scales: scale deformation maps according to the multi-resolution pipeline*/
  Tiff_Im::SetDefTileFile(50000);
  // ONLY ONE SCALE IS TAKEN INTO ACCOUNT
  int mNbIm=aVLI.size();
  // Master image Box
  cGPU_LoadedImGeom & aaGLI_0 = *(aVLI[0]);
  std::string aMasterName=aaGLI_0.PDV()->Name();
  Box2di BoxMaster=aaGLI_0.Geom()->BoxClip();
  //int aDeZoom=aaGLI_0.Geom()->mDeZoom;
  std::cout<<"BoxMaster ==> "<<BoxMaster<<"  "<<aMasterName<<std::endl;
  Pt2di SzMaster=aaGLI_0.getSizeImage();
  for (int aKIm= 1 ; aKIm<mNbIm ; aKIm++)
  {
      aMasterName=aaGLI_0.PDV()->Name();
      std::string aPrefixMaster= aNameOrig + "_I" + ToString(0) + "_S"+ ToString(0)+"_I" + ToString(aKIm) + "_S"+ ToString(0);
      cGPU_LoadedImGeom & aaGLI_K = *(aVLI[aKIm]);
      std::string aSecName=aaGLI_K.PDV()->Name();
      Box2di BoxSec=aaGLI_K.Geom()->BoxClip();
      //int aDeZoom=aaGLI_0.Geom()->mDeZoom;
      std::cout<<"Box Secondary image "<<BoxSec<<"   "<<aSecName<<std::endl;
      Pt2di SzSec=aaGLI_K.getSizeImage();
      std::string aPrefixSec  = aNameOrig + "_I" + ToString(aKIm) + "_S"+ ToString(0);
      std::string aPrefixEpip = aNameOrig + "_I" + ToString(aKIm) +"_I" + ToString(0) + "_S"+ ToString(0);
      // CLIP IMAGES GEOX GEOY EpIm_GEOX EpIm_GEOY
      /*std::string aMasterBare=aMasterName.substr(0, aMasterName.find_last_of("."));
      std::string aSecBare=aSecName.substr(0, aSecName.find_last_of("."));
      bool mCmp=aMasterBare>aSecBare;
      std::string aNameEpip12=mCmp ? "Epi_Im2_Right_"+aSecBare+"_"+aMasterBare+".tif":"Epi_Im1_Left_"+aMasterBare+"_"+aSecBare+".tif";
      std::string aNameEpip21=mCmp ? "Epi_Im1_Left_"+aSecBare+"_"+aMasterBare+".tif":"Epi_Im2_Right_"+aMasterBare+"_"+aSecBare+".tif";
      std::string  aNameBareEpip12=aNameEpip12.substr(0,aNameEpip12.find_last_of(".")); //.tif
      std::string  aNameBareEpip21=aNameEpip21.substr(0,aNameEpip21.find_last_of(".")); //.tif*/
      bool Inverse =aMasterName>aSecName;
      if (Inverse) ElSwap(aMasterName,aSecName);
      CamStenope * aCam1= aaGLI_0.Geom()->GetOriNN();
      CamStenope * aCam2= aaGLI_K.Geom()->GetOriNN();
      double aScale=1.0;
      cCpleEpip aCple("./",
                      aScale,
                      Inverse ? *aCam2:*aCam1,aMasterName,
                      Inverse ? *aCam1:*aCam2,aSecName
                      );
      std::string aNameEpip12=aCple.Dir()+aCple.LocNameImEpi(Inverse ? false : true);
      std::string aNameEpip21=aCple.Dir()+aCple.LocNameImEpi(Inverse ? true : false);
      std::string aNameBareEpip12=aNameEpip12.substr(0,aNameEpip12.find_last_of(".")); //.tif
      std::string aNameBareEpip21=aNameEpip21.substr(0,aNameEpip21.find_last_of(".")); //.tif
      // im1 --> Epip12
     // std::string Im1_to_Epip12_GeoX=aNameBareEpip12+"_GEOX.tif";
     // std::string Im1_to_Epip12_GeoY=aNameBareEpip12+"_GEOY.tif";
     // std::string Im1_to_Epip12_Masq=aNameBareEpip12+"_Masq.tif";

      // im2 --> Epip21
      //std::string Im2_to_Epip21_GeoX=aNameBareEpip21+"_GEOX.tif";
      //std::string Im2_to_Epip21_GeoY=aNameBareEpip21+"_GEOY.tif";
      //std::string Im2_to_Epip21_Masq=aNameBareEpip21+"_Masq.tif";

      // Epip12 --> im1
      std::string Epip12_to_im1_GeoX=aNameBareEpip12+"_EpIm_GEOX.tif";
      std::string Epip12_to_im1_GeoY=aNameBareEpip12+"_EpIm_GEOY.tif";
      std::string Epip12_to_im1_Masq=aNameBareEpip12+"_EpIm_Masq.tif";
      // Epip21 --> im2
      std::string Epip21_to_im2_GeoX=aNameBareEpip21+"_EpIm_GEOX.tif";
      std::string Epip21_to_im2_GeoY=aNameBareEpip21+"_EpIm_GEOY.tif";
      std::string Epip21_to_im2_Masq=aNameBareEpip21+"_EpIm_Masq.tif";

      // calculer les grilles de déformation
      std::cout<<"Master2EpipGeoX   => "<<std::endl;
      Tiff_Im     Master2EpipGeoX(Epip12_to_im1_GeoX.c_str());
      Tiff_Im     Master2EpipGeoY(Epip12_to_im1_GeoY.c_str());
      Tiff_Im     Master2EpipMasq(Epip12_to_im1_Masq.c_str());

      Tiff_Im     Sec2EpipGeoX(Epip21_to_im2_GeoX.c_str());
      Tiff_Im     Sec2EpipGeoY(Epip21_to_im2_GeoY.c_str());
      //std::cout<<"Sec2EpipGeoY==> "<<Sec2EpipGeoY.sz()<<std::endl;
      Tiff_Im     Sec2EpipMasq(Epip21_to_im2_Masq.c_str());

      /* read relevant image boxes for Master */
        TIm2D<float,double> aTImMaster2EpipGeoX(SzMaster);
        TIm2D<float,double> aTImMaster2EpipGeoY(SzMaster);
        Im2D_Bits<1> aImMaster2EpipMasq(SzMaster.x,SzMaster.y);
        //TIm2DBits<1> aTImMaster2EpipMasq(aImMaster2EpipMasq);

        TIm2D<float,double> aTImSec2EpipGeoX(SzSec);
        TIm2D<float,double> aTImSec2EpipGeoY(SzSec);
        Im2D_Bits<1> aImSec2EpipMasq(SzSec.x,SzSec.y);
        //TIm2DBits<1> aTImSec2EpipMasq(aImSec2EpipMasq);

        // Export Epipolars direclty as they have been calculated

        Tiff_Im  anEpip12(aNameEpip12.c_str());
        Tiff_Im  anEpip21(aNameEpip21.c_str());

        // 1. Master : compute box in epipolar domain
       ELISE_COPY
      (
           aTImMaster2EpipGeoX.all_pts(),
           trans(Master2EpipGeoX.in(),BoxMaster._p0),
           aTImMaster2EpipGeoX.out()
      );

       ELISE_COPY
      (
           aTImMaster2EpipGeoY.all_pts(),
           trans(Master2EpipGeoY.in(),BoxMaster._p0),
           aTImMaster2EpipGeoY.out()
      );

       ELISE_COPY
      (
           aImMaster2EpipMasq.all_pts(),
           trans(Master2EpipMasq.in(),BoxMaster._p0),
           aImMaster2EpipMasq.out()
      );

       int HomXmin=1e9;
       int HomYmin=1e9;
       int HomXmax=-1e9;
       int HomYmax=-1e9;

       Pt2di aP=Pt2di(0,0);
       for (aP.x=0;aP.x<SzMaster.x;aP.x++)
         {
           for (aP.y=0;aP.y<SzMaster.y;aP.y++)
             {
               if (aImMaster2EpipMasq.get(aP.x,aP.y))
                 {
                   double aGeox=aTImMaster2EpipGeoX.get(aP);
                   double aGeoy=aTImMaster2EpipGeoY.get(aP);

                   if (aGeox>HomXmax) HomXmax=aGeox;
                   if (aGeox<HomXmin) HomXmin=aGeox;

                   if (aGeoy>HomYmax) HomYmax=aGeoy;
                   if (aGeoy<HomYmin) HomYmin=aGeoy;
                 }
             }
         }

       Box2di aBoxMasterInEpip(round_ni(Pt2dr(HomXmin,HomYmin)),round_ni(Pt2dr(HomXmax,HomYmax)));
       std::cout<<"aBoxMasterInEpip  => "<<aBoxMasterInEpip<<std::endl;

       // 2. Secondary : compute box in epipolar domain

       ELISE_COPY
      (
           aTImSec2EpipGeoX.all_pts(),
           trans(Sec2EpipGeoX.in(),BoxSec._p0),
           aTImSec2EpipGeoX.out()
      );

       ELISE_COPY
      (
           aTImSec2EpipGeoY.all_pts(),
           trans(Sec2EpipGeoY.in(),BoxSec._p0),
           aTImSec2EpipGeoY.out()
      );

       ELISE_COPY
      (
           aImSec2EpipMasq.all_pts(),
           trans(Sec2EpipMasq.in(),BoxSec._p0),
           aImSec2EpipMasq.out()
      );

       HomXmin=1e9;
       HomYmin=1e9;
       HomXmax=-1e9;
       HomYmax=-1e9;

       aP=Pt2di(0,0);
       for (aP.x=0;aP.x<SzSec.x;aP.x++)
         {
           for (aP.y=0;aP.y<SzSec.y;aP.y++)
             {
               if (aImSec2EpipMasq.get(aP.x,aP.y))
                 {
                   double aGeox=aTImSec2EpipGeoX.get(aP);
                   double aGeoy=aTImSec2EpipGeoY.get(aP);

                   if (aGeox>HomXmax) HomXmax=aGeox;
                   if (aGeox<HomXmin) HomXmin=aGeox;

                   if (aGeoy>HomYmax) HomYmax=aGeoy;
                   if (aGeoy<HomYmin) HomYmin=aGeoy;
                 }
             }
         }
       Box2di aBoxSecInEpip(round_ni(Pt2dr(HomXmin,HomYmin)),round_ni(Pt2dr(HomXmax,HomYmax)));
       std::cout<<"aBoxSecInEpip   "<<aBoxSecInEpip<<std::endl;
       // 3. Generate Tiles Im->Epip and Epip->Im

       // create epipolar boxes

       TIm2D<float,double> aTImMasterEpip(aBoxMasterInEpip.sz());
       TIm2D<float,double> aTImSecEpip(aBoxSecInEpip.sz());

       /*3.1 Master*/

       /*
       Tiff_Im Epip2MasterGeoX(Im1_to_Epip12_GeoX.c_str());
       Tiff_Im Epip2MasterGeoY(Im1_to_Epip12_GeoY.c_str());
       Tiff_Im Epip2MasterMasq(Im1_to_Epip12_Masq.c_str());
       */

       /*
       TIm2D<float,double> aTImEpip2MasterGeoX(aBoxMasterInEpip._p1-aBoxMasterInEpip._p0);
       TIm2D<float,double> aTImEpip2MasterGeoY(aBoxMasterInEpip._p1-aBoxMasterInEpip._p0);
       */

       //Pt2di SzInEp=aBoxMasterInEpip._p1-aBoxMasterInEpip._p0;
       //Im2D_Bits<1> aImEpip2MasterMasq(SzInEp.x,SzInEp.y);
       //TIm2DBits<1> aTImEpip2MasterMasq(aImEpip2MasterMasq);

       /*
       ELISE_COPY
      (
           aTImEpip2MasterGeoX.all_pts(),
           trans(Epip2MasterGeoX.in(),aBoxMasterInEpip._p0),
           aTImEpip2MasterGeoX.out()
      );
       ELISE_COPY
      (
           aTImEpip2MasterGeoY.all_pts(),
           trans(Epip2MasterGeoY.in(),aBoxMasterInEpip._p0),
           aTImEpip2MasterGeoY.out()
      );
       */

       ELISE_COPY
      (
           aTImMasterEpip.all_pts(),
           trans(anEpip12.in(),aBoxMasterInEpip._p0),
           aTImMasterEpip.out()
      );

       /*
       ELISE_COPY
      (
           aImEpip2MasterMasq.all_pts(),
           trans(Epip2MasterMasq.in(),aBoxMasterInEpip._p0),
           aImEpip2MasterMasq.out()
      );
      */

       /*
       Tiff_Im Epip2SecGeoX(Im2_to_Epip21_GeoX.c_str());
       Tiff_Im Epip2SecGeoY(Im2_to_Epip21_GeoY.c_str());
       Tiff_Im Epip2SecMasq(Im2_to_Epip21_Masq.c_str());
       */

       /*
       TIm2D<float,double> aTImEpip2SecGeoX(aBoxSecInEpip._p1-aBoxSecInEpip._p0);
       TIm2D<float,double> aTImEpip2SecGeoY(aBoxSecInEpip._p1-aBoxSecInEpip._p0);
       */

       //Pt2di SzInEpSec=aBoxSecInEpip._p1-aBoxSecInEpip._p0;
       //Im2D_Bits<1> aImEpip2SecMasq(SzInEpSec.x,SzInEpSec.y);
       //TIm2DBits<1> aTImEpip2SecMasq(aImEpip2SecMasq);

       /*
       ELISE_COPY
      (
           aTImEpip2SecGeoX.all_pts(),
           trans(Epip2SecGeoX.in(),aBoxSecInEpip._p0),
           aTImEpip2SecGeoX.out()
      );
       ELISE_COPY
      (
           aTImEpip2SecGeoY.all_pts(),
           trans(Epip2SecGeoY.in(),aBoxSecInEpip._p0),
           aTImEpip2SecGeoY.out()
      );
       */

       ELISE_COPY
      (
           aTImSecEpip.all_pts(),
           trans(anEpip21.in(),aBoxSecInEpip._p0),
           aTImSecEpip.out()
      );

       /*
       ELISE_COPY
      (
           aImEpip2SecMasq.all_pts(),
           trans(Epip2SecMasq.in(),aBoxSecInEpip._p0),
           aImEpip2SecMasq.out()
      );
      */
      std::cout<<"Fill Image tiles ==> Im2 to Epip 21 Fill tiles  "<<std::endl;

       //5. Prepare for grid_sample
       /*
          xhmin,yhmin,xhmax,yhmax=Bbox
          divx=xhmax-xhmin
          divy=yhmax-yhmin
          """GeoXB=GeoXBox/W
          GeoXB=GeoXB-(xhmin/W)
          GeoXB=GeoXB*(W/divx)"""

          GeoXB=(2*(GeoXBox-xhmin)/divx)-1
          GeoYB=(2*(GeoYBox-yhmin)/divy)-1
          return GeoXB,GeoYB
      */

       // Image --> Epip
       ELISE_COPY(aTImMaster2EpipGeoX.all_pts(),
                  2*((aTImMaster2EpipGeoX.in()-aBoxMasterInEpip.P0().x)/(aBoxMasterInEpip.P1().x-aBoxMasterInEpip.P0().x-1.0))-1.0,
                  aTImMaster2EpipGeoX.out()
                  );

       ELISE_COPY(aTImMaster2EpipGeoY.all_pts(),
                  2*((aTImMaster2EpipGeoY.in()-aBoxMasterInEpip.P0().y)/(aBoxMasterInEpip.P1().y-aBoxMasterInEpip.P0().y-1.0))-1.0,
                  aTImMaster2EpipGeoY.out()
                  );

       ELISE_COPY(aTImSec2EpipGeoX.all_pts(),
                  2*((aTImSec2EpipGeoX.in()-aBoxSecInEpip.P0().x)/(aBoxSecInEpip.P1().x-aBoxSecInEpip.P0().x-1.0))-1.0,
                  aTImSec2EpipGeoX.out()
                  );

       ELISE_COPY(aTImSec2EpipGeoY.all_pts(),
                  2*((aTImSec2EpipGeoY.in()-aBoxSecInEpip.P0().y)/(aBoxSecInEpip.P1().y-aBoxSecInEpip.P0().y-1.0))-1.0,
                  aTImSec2EpipGeoY.out()
                  );

       // Epip --> Image

       /*
       ELISE_COPY(aTImEpip2MasterGeoX.all_pts(),
                  2*((aTImEpip2MasterGeoX.in()-BoxMaster.P0().x)/(BoxMaster.P1().x-BoxMaster.P0().x-1.0))-1.0,
                  aTImEpip2MasterGeoX.out()
                  );

       ELISE_COPY(aTImEpip2MasterGeoY.all_pts(),
                  2*((aTImEpip2MasterGeoY.in()-BoxMaster.P0().y)/(BoxMaster.P1().y-BoxMaster.P0().y-1.0))-1.0,
                  aTImEpip2MasterGeoY.out()
                  );

       ELISE_COPY(aTImEpip2SecGeoX.all_pts(),
                  2*((aTImEpip2SecGeoX.in()-BoxSec.P0().x)/(BoxSec.P1().x-BoxSec.P0().x-1.0))-1.0,
                  aTImEpip2SecGeoX.out()
                  );

       ELISE_COPY(aTImEpip2SecGeoY.all_pts(),
                  2*((aTImEpip2SecGeoY.in()-BoxSec.P0().y)/(BoxSec.P1().y-BoxSec.P0().y-1.0))-1.0,
                  aTImEpip2SecGeoY.out()
                  );
       */

       // 6. Save
       SaveIm(aPrefixMaster+"_ORIG_EpIm_GEOX.tif",aTImMaster2EpipGeoX._d,Box2di(Pt2di(0,0),SzMaster));
       SaveIm(aPrefixMaster+"_ORIG_EpIm_GEOY.tif",aTImMaster2EpipGeoY._d,Box2di(Pt2di(0,0),SzMaster));
       SaveIm(aPrefixMaster+"_ORIG_EpIm_Masq.tif",aImMaster2EpipMasq.data(),Box2di(Pt2di(0,0),SzMaster));

       SaveIm(aPrefixSec+"_ORIG_EpIm_GEOX.tif",aTImSec2EpipGeoX._d,Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_EpIm_GEOY.tif",aTImSec2EpipGeoY._d,Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_EpIm_Masq.tif",aImSec2EpipMasq.data(),Box2di(Pt2di(0,0),SzSec));

       /*
       SaveIm(aPrefixMaster+"_ORIG_GEOX.tif",aTImEpip2MasterGeoX._d,Box2di(Pt2di(0,0),aBoxMasterInEpip._p1-aBoxMasterInEpip._p0));
       SaveIm(aPrefixMaster+"_ORIG_GEOY.tif",aTImEpip2MasterGeoY._d,Box2di(Pt2di(0,0),aBoxMasterInEpip._p1-aBoxMasterInEpip._p0));
       SaveIm(aPrefixMaster+"_ORIG_Masq.tif",aImEpip2MasterMasq.data(),Box2di(Pt2di(0,0),aBoxMasterInEpip._p1-aBoxMasterInEpip._p0));

       SaveIm(aPrefixSec+"_ORIG_GEOX.tif",aTImEpip2SecGeoX._d,Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       SaveIm(aPrefixSec+"_ORIG_GEOY.tif",aTImEpip2SecGeoY._d,Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       SaveIm(aPrefixSec+"_ORIG_Masq.tif",aImEpip2SecMasq.data(),Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       */

       // Save Epipolars
       SaveIm(aPrefixMaster+"_Epip.tif",aTImMasterEpip._d,Box2di(Pt2di(0,0),aBoxMasterInEpip.sz()));
       SaveIm(aPrefixEpip+"_Epip.tif",aTImSecEpip._d,Box2di(Pt2di(0,0),aBoxSecInEpip.sz()));
  }
}

void cAppliMICMAC::GenerateGeoPassage_ImEpip_EpipIm_BBox(std::vector<cGPU_LoadedImGeom *> & aVLI, std::string & aNameOrig)
{
  /*TODO: Include scales: scale deformation maps according to the multi-resolution pipeline*/
  Tiff_Im::SetDefTileFile(50000);
  // ONLY ONE SCALE IS TAKEN INTO ACCOUNT
  int mNbIm=aVLI.size();
  // Master image Box
  cGPU_LoadedImGeom & aaGLI_0 = *(aVLI[0]);
  std::string aMasterName=aaGLI_0.PDV()->Name();
  Box2di BoxMaster=aaGLI_0.Geom()->BoxClip();
  std::cout<<"BoxMaster ==> "<<BoxMaster<<std::endl;
  Pt2di SzMaster=aaGLI_0.getSizeImage();
  // Orientation to reover epipolar images names
  std::string aNameOri=DirOfFile(aaGLI_0.PDV()->NameGeom());
  std::size_t aPosl=aNameOri.find_first_of("/");
  aNameOri=aNameOri.substr(aPosl+1,-1);
  std::size_t aPosr=aNameOri.find_last_of("/");
  aNameOri=aNameOri.substr(0,aPosr);
  cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc("./");

  for (int aKIm= 1 ; aKIm<mNbIm ; aKIm++)
  {
      std::string aPrefixMaster= aNameOrig + "_I" + ToString(0) + "_S"+ ToString(0)+"_I" + ToString(aKIm) + "_S"+ ToString(0);
      cGPU_LoadedImGeom & aaGLI_0 = *(aVLI[aKIm]);
      std::string aSecName=aaGLI_0.PDV()->Name();
      Box2di BoxSec=aaGLI_0.Geom()->BoxClip();
      std::cout<<"Box Secondary image "<<BoxSec<<std::endl;
      Pt2di SzSec=aaGLI_0.getSizeImage();
      std::string aPrefixSec  = aNameOrig + "_I" + ToString(aKIm) + "_S"+ ToString(0);
      std::string aPrefixEpip = aNameOrig + "_I" + ToString(aKIm) +"_I" + ToString(0) + "_S"+ ToString(0);
      // CLIP IMAGES GEOX GEOY EpIm_GEOX EpIm_GEOY
      std::string NameBareOri=aNameOri.substr(aNameOri.find_first_of("-")+1,-1);
      std::string aNameEpip12=aICNM->NameImEpip(NameBareOri,aMasterName,aSecName);
      std::string aNameEpip21=aICNM->NameImEpip(NameBareOri,aSecName,aMasterName);
      std::string  aNameBareEpip12=aNameEpip12.substr(0,aNameEpip12.find_last_of(".")); //.tif
      std::string  aNameBareEpip21=aNameEpip21.substr(0,aNameEpip21.find_last_of(".")); //.tif
      // im1 --> Epip12
      std::string Im1_to_Epip12_GeoX=aNameBareEpip12+"_GEOX.tif";
      std::string Im1_to_Epip12_GeoY=aNameBareEpip12+"_GEOY.tif";
      std::string Im1_to_Epip12_Masq=aNameBareEpip12+"_Masq.tif";

      // im2 --> Epip21
      std::string Im2_to_Epip21_GeoX=aNameBareEpip21+"_GEOX.tif";
      std::string Im2_to_Epip21_GeoY=aNameBareEpip21+"_GEOY.tif";
      std::string Im2_to_Epip21_Masq=aNameBareEpip21+"_Masq.tif";

      // Epip12 --> im1
      std::string Epip12_to_im1_GeoX=aNameBareEpip12+"_EpIm_GEOX.tif";
      std::string Epip12_to_im1_GeoY=aNameBareEpip12+"_EpIm_GEOY.tif";
      std::string Epip12_to_im1_Masq=aNameBareEpip12+"_EpIm_Masq.tif";
      // Epip21 --> im2
      std::string Epip21_to_im2_GeoX=aNameBareEpip21+"_EpIm_GEOX.tif";
      std::string Epip21_to_im2_GeoY=aNameBareEpip21+"_EpIm_GEOY.tif";
      std::string Epip21_to_im2_Masq=aNameBareEpip21+"_EpIm_Masq.tif";

      // calculer les grilles de déformation
      std::cout<<"Master2EpipGeoX   => "<<std::endl;
      Tiff_Im     Master2EpipGeoX(Epip12_to_im1_GeoX.c_str());
      Tiff_Im     Master2EpipGeoY(Epip12_to_im1_GeoY.c_str());
      Tiff_Im     Master2EpipMasq(Epip12_to_im1_Masq.c_str());

      Tiff_Im     Sec2EpipGeoX(Epip21_to_im2_GeoX.c_str());
      Tiff_Im     Sec2EpipGeoY(Epip21_to_im2_GeoY.c_str());
      //std::cout<<"Sec2EpipGeoY==> "<<Sec2EpipGeoY.sz()<<std::endl;
      Tiff_Im     Sec2EpipMasq(Epip21_to_im2_Masq.c_str());

      /* read relevant image boxes for Master */
        TIm2D<float,double> aTImMaster2EpipGeoX(SzMaster);
        TIm2D<float,double> aTImMaster2EpipGeoY(SzMaster);
        Im2D_Bits<1> aImMaster2EpipMasq(SzMaster.x,SzMaster.y);
        //TIm2DBits<1> aTImMaster2EpipMasq(aImMaster2EpipMasq);

        TIm2D<float,double> aTImSec2EpipGeoX(SzSec);
        TIm2D<float,double> aTImSec2EpipGeoY(SzSec);
        Im2D_Bits<1> aImSec2EpipMasq(SzSec.x,SzSec.y);
        //TIm2DBits<1> aTImSec2EpipMasq(aImSec2EpipMasq);

        // Export Epipolars direclty as they have been calculated

        Tiff_Im  anEpip12(aNameEpip12.c_str());
        Tiff_Im  anEpip21(aNameEpip21.c_str());

        // 1. Master : compute box in epipolar domain
       ELISE_COPY
      (
           aTImMaster2EpipGeoX.all_pts(),
           trans(Master2EpipGeoX.in(),BoxMaster._p0),
           aTImMaster2EpipGeoX.out()
      );

       ELISE_COPY
      (
           aTImMaster2EpipGeoY.all_pts(),
           trans(Master2EpipGeoY.in(),BoxMaster._p0),
           aTImMaster2EpipGeoY.out()
      );

       ELISE_COPY
      (
           aImMaster2EpipMasq.all_pts(),
           trans(Master2EpipMasq.in(),BoxMaster._p0),
           aImMaster2EpipMasq.out()
      );

       int HomXmin=1e9;
       int HomYmin=1e9;
       int HomXmax=-1e9;
       int HomYmax=-1e9;

       Pt2di aP=Pt2di(0,0);
       for (aP.x=0;aP.x<SzMaster.x;aP.x++)
         {
           for (aP.y=0;aP.y<SzMaster.y;aP.y++)
             {
               if (aImMaster2EpipMasq.get(aP.x,aP.y))
                 {
                   double aGeox=aTImMaster2EpipGeoX.get(aP);
                   double aGeoy=aTImMaster2EpipGeoY.get(aP);

                   if (aGeox>HomXmax) HomXmax=aGeox;
                   if (aGeox<HomXmin) HomXmin=aGeox;

                   if (aGeoy>HomYmax) HomYmax=aGeoy;
                   if (aGeoy<HomYmin) HomYmin=aGeoy;
                 }
             }
         }

       Box2di aBoxMasterInEpip(round_ni(Pt2dr(HomXmin,HomYmin)),round_ni(Pt2dr(HomXmax,HomYmax)));
       std::cout<<"aBoxMasterInEpip  => "<<aBoxMasterInEpip<<std::endl;

       // 2. Secondary : compute box in epipolar domain

       ELISE_COPY
      (
           aTImSec2EpipGeoX.all_pts(),
           trans(Sec2EpipGeoX.in(),BoxSec._p0),
           aTImSec2EpipGeoX.out()
      );

       ELISE_COPY
      (
           aTImSec2EpipGeoY.all_pts(),
           trans(Sec2EpipGeoY.in(),BoxSec._p0),
           aTImSec2EpipGeoY.out()
      );

       ELISE_COPY
      (
           aImSec2EpipMasq.all_pts(),
           trans(Sec2EpipMasq.in(),BoxSec._p0),
           aImSec2EpipMasq.out()
      );

       HomXmin=1e9;
       HomYmin=1e9;
       HomXmax=-1e9;
       HomYmax=-1e9;

       aP=Pt2di(0,0);
       for (aP.x=0;aP.x<SzSec.x;aP.x++)
         {
           for (aP.y=0;aP.y<SzSec.y;aP.y++)
             {
               if (aImSec2EpipMasq.get(aP.x,aP.y))
                 {
                   double aGeox=aTImSec2EpipGeoX.get(aP);
                   double aGeoy=aTImSec2EpipGeoY.get(aP);

                   if (aGeox>HomXmax) HomXmax=aGeox;
                   if (aGeox<HomXmin) HomXmin=aGeox;

                   if (aGeoy>HomYmax) HomYmax=aGeoy;
                   if (aGeoy<HomYmin) HomYmin=aGeoy;
                 }
             }
         }
       Box2di aBoxSecInEpip(round_ni(Pt2dr(HomXmin,HomYmin)),round_ni(Pt2dr(HomXmax,HomYmax)));

       // 3. Generate Tiles Im->Epip and Epip->Im

       // create epipolar boxes

       TIm2D<float,double> aTImMasterEpip(aBoxMasterInEpip.sz());
       TIm2D<float,double> aTImSecEpip(aBoxSecInEpip.sz());

       /*3.1 Master*/

       Tiff_Im Epip2MasterGeoX(Im1_to_Epip12_GeoX.c_str());
       Tiff_Im Epip2MasterGeoY(Im1_to_Epip12_GeoY.c_str());
       Tiff_Im Epip2MasterMasq(Im1_to_Epip12_Masq.c_str());

       TIm2D<float,double> aTImEpip2MasterGeoX(aBoxMasterInEpip._p1-aBoxMasterInEpip._p0);
       TIm2D<float,double> aTImEpip2MasterGeoY(aBoxMasterInEpip._p1-aBoxMasterInEpip._p0);

       Pt2di SzInEp=aBoxMasterInEpip._p1-aBoxMasterInEpip._p0;
       Im2D_Bits<1> aImEpip2MasterMasq(SzInEp.x,SzInEp.y);
       //TIm2DBits<1> aTImEpip2MasterMasq(aImEpip2MasterMasq);

       ELISE_COPY
      (
           aTImEpip2MasterGeoX.all_pts(),
           trans(Epip2MasterGeoX.in(),aBoxMasterInEpip._p0),
           aTImEpip2MasterGeoX.out()
      );
       ELISE_COPY
      (
           aTImEpip2MasterGeoY.all_pts(),
           trans(Epip2MasterGeoY.in(),aBoxMasterInEpip._p0),
           aTImEpip2MasterGeoY.out()
      );

       ELISE_COPY
      (
           aTImMasterEpip.all_pts(),
           trans(anEpip12.in(),aBoxMasterInEpip._p0),
           aTImMasterEpip.out()
      );

       ELISE_COPY
      (
           aImEpip2MasterMasq.all_pts(),
           trans(Epip2MasterMasq.in(),aBoxMasterInEpip._p0),
           aImEpip2MasterMasq.out()
      );

       Tiff_Im Epip2SecGeoX(Im2_to_Epip21_GeoX.c_str());
       Tiff_Im Epip2SecGeoY(Im2_to_Epip21_GeoY.c_str());
       Tiff_Im Epip2SecMasq(Im2_to_Epip21_Masq.c_str());


       std::cout<<"aBoxSecInEpip   "<<aBoxSecInEpip<<std::endl;
       TIm2D<float,double> aTImEpip2SecGeoX(aBoxSecInEpip._p1-aBoxSecInEpip._p0);
       TIm2D<float,double> aTImEpip2SecGeoY(aBoxSecInEpip._p1-aBoxSecInEpip._p0);

       Pt2di SzInEpSec=aBoxSecInEpip._p1-aBoxSecInEpip._p0;
       Im2D_Bits<1> aImEpip2SecMasq(SzInEpSec.x,SzInEpSec.y);
       //TIm2DBits<1> aTImEpip2SecMasq(aImEpip2SecMasq);

       ELISE_COPY
      (
           aTImEpip2SecGeoX.all_pts(),
           trans(Epip2SecGeoX.in(),aBoxSecInEpip._p0),
           aTImEpip2SecGeoX.out()
      );
       ELISE_COPY
      (
           aTImEpip2SecGeoY.all_pts(),
           trans(Epip2SecGeoY.in(),aBoxSecInEpip._p0),
           aTImEpip2SecGeoY.out()
      );

       ELISE_COPY
      (
           aTImSecEpip.all_pts(),
           trans(anEpip21.in(),aBoxSecInEpip._p0),
           aTImSecEpip.out()
      );

       ELISE_COPY
      (
           aImEpip2SecMasq.all_pts(),
           trans(Epip2SecMasq.in(),aBoxSecInEpip._p0),
           aImEpip2SecMasq.out()
      );
      std::cout<<"Fill Image tiles ==> Im2 to Epip 21 Fill tiles  "<<std::endl;

       //5. Prepare for grid_sample
       /*
          xhmin,yhmin,xhmax,yhmax=Bbox
          divx=xhmax-xhmin
          divy=yhmax-yhmin
          """GeoXB=GeoXBox/W
          GeoXB=GeoXB-(xhmin/W)
          GeoXB=GeoXB*(W/divx)"""

          GeoXB=(2*(GeoXBox-xhmin)/divx)-1
          GeoYB=(2*(GeoYBox-yhmin)/divy)-1
          return GeoXB,GeoYB
      */

       // Image --> Epip
       ELISE_COPY(aTImMaster2EpipGeoX.all_pts(),
                  2*((aTImMaster2EpipGeoX.in()-aBoxMasterInEpip.P0().x)/(aBoxMasterInEpip.P1().x-aBoxMasterInEpip.P0().x-1.0))-1.0,
                  aTImMaster2EpipGeoX.out()
                  );

       ELISE_COPY(aTImMaster2EpipGeoY.all_pts(),
                  2*((aTImMaster2EpipGeoY.in()-aBoxMasterInEpip.P0().y)/(aBoxMasterInEpip.P1().y-aBoxMasterInEpip.P0().y-1.0))-1.0,
                  aTImMaster2EpipGeoY.out()
                  );

       ELISE_COPY(aTImSec2EpipGeoX.all_pts(),
                  2*((aTImSec2EpipGeoX.in()-aBoxSecInEpip.P0().x)/(aBoxSecInEpip.P1().x-aBoxSecInEpip.P0().x-1.0))-1.0,
                  aTImSec2EpipGeoX.out()
                  );

       ELISE_COPY(aTImSec2EpipGeoY.all_pts(),
                  2*((aTImSec2EpipGeoY.in()-aBoxSecInEpip.P0().y)/(aBoxSecInEpip.P1().y-aBoxSecInEpip.P0().y-1.0))-1.0,
                  aTImSec2EpipGeoY.out()
                  );

       // Epip --> Image

       ELISE_COPY(aTImEpip2MasterGeoX.all_pts(),
                  2*((aTImEpip2MasterGeoX.in()-BoxMaster.P0().x)/(BoxMaster.P1().x-BoxMaster.P0().x-1.0))-1.0,
                  aTImEpip2MasterGeoX.out()
                  );

       ELISE_COPY(aTImEpip2MasterGeoY.all_pts(),
                  2*((aTImEpip2MasterGeoY.in()-BoxMaster.P0().y)/(BoxMaster.P1().y-BoxMaster.P0().y-1.0))-1.0,
                  aTImEpip2MasterGeoY.out()
                  );

       ELISE_COPY(aTImEpip2SecGeoX.all_pts(),
                  2*((aTImEpip2SecGeoX.in()-BoxSec.P0().x)/(BoxSec.P1().x-BoxSec.P0().x-1.0))-1.0,
                  aTImEpip2SecGeoX.out()
                  );

       ELISE_COPY(aTImEpip2SecGeoY.all_pts(),
                  2*((aTImEpip2SecGeoY.in()-BoxSec.P0().y)/(BoxSec.P1().y-BoxSec.P0().y-1.0))-1.0,
                  aTImEpip2SecGeoY.out()
                  );

       // 6. Save
       SaveIm(aPrefixMaster+"_ORIG_EpIm_GEOX.tif",aTImMaster2EpipGeoX._d,Box2di(Pt2di(0,0),SzMaster));
       SaveIm(aPrefixMaster+"_ORIG_EpIm_GEOY.tif",aTImMaster2EpipGeoY._d,Box2di(Pt2di(0,0),SzMaster));
       SaveIm(aPrefixMaster+"_ORIG_EpIm_Masq.tif",aImMaster2EpipMasq.data(),Box2di(Pt2di(0,0),SzMaster));

       SaveIm(aPrefixSec+"_ORIG_EpIm_GEOX.tif",aTImSec2EpipGeoX._d,Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_EpIm_GEOY.tif",aTImSec2EpipGeoY._d,Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_EpIm_Masq.tif",aImSec2EpipMasq.data(),Box2di(Pt2di(0,0),SzSec));

       /*
       SaveIm(aPrefixMaster+"_ORIG_GEOX.tif",aTImEpip2MasterGeoX._d,Box2di(Pt2di(0,0),aBoxMasterInEpip._p1-aBoxMasterInEpip._p0));
       SaveIm(aPrefixMaster+"_ORIG_GEOY.tif",aTImEpip2MasterGeoY._d,Box2di(Pt2di(0,0),aBoxMasterInEpip._p1-aBoxMasterInEpip._p0));
       SaveIm(aPrefixMaster+"_ORIG_Masq.tif",aImEpip2MasterMasq.data(),Box2di(Pt2di(0,0),aBoxMasterInEpip._p1-aBoxMasterInEpip._p0));

       SaveIm(aPrefixSec+"_ORIG_GEOX.tif",aTImEpip2SecGeoX._d,Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       SaveIm(aPrefixSec+"_ORIG_GEOY.tif",aTImEpip2SecGeoY._d,Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       SaveIm(aPrefixSec+"_ORIG_Masq.tif",aImEpip2SecMasq.data(),Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       */

       // Save Epipolars
       SaveIm(aPrefixMaster+"_Epip.tif",aTImMasterEpip._d,Box2di(Pt2di(0,0),aBoxMasterInEpip.sz()));
       SaveIm(aPrefixEpip+"_Epip.tif",aTImSecEpip._d,Box2di(Pt2di(0,0),aBoxSecInEpip.sz()));
  }
}


void cAppliMICMAC::GenerateGeoPassage_Homography_BBox_Stenope(std::vector<cGPU_LoadedImGeom *> & aVLI, std::string & aNameOrig)
{
  /*TODO: Include scales: scale deformation maps according to the multi-resolution pipeline*/
  Tiff_Im::SetDefTileFile(50000);
  // ONLY ONE SCALE IS TAKEN INTO ACCOUNT
  int mNbIm=aVLI.size();
  // Master image Box
  cGPU_LoadedImGeom & aaGLI_0 = *(aVLI[0]);
  std::string aMasterName=aaGLI_0.PDV()->Name();
  Box2di BoxMaster=aaGLI_0.Geom()->BoxClip();
  std::cout<<"BoxMaster ==> "<<BoxMaster<<std::endl;
  for (int aKIm= 1 ; aKIm<mNbIm ; aKIm++)
  {
      cGPU_LoadedImGeom & aaGLI_K = *(aVLI[aKIm]);
      std::string aSecName=aaGLI_K.PDV()->Name();
      Box2di BoxSec=aaGLI_K.Geom()->BoxClip();
      //int aDeZoom=aaGLI_0.Geom()->mDeZoom;
      std::cout<<"Box Secondary image "<<BoxSec<<std::endl;
      Pt2di SzSec=aaGLI_K.getSizeImage();
      std::string aPrefixSec  = aNameOrig + "_I" + ToString(aKIm) + "_S"+ ToString(0);
      std::string aPrefixEpip = aNameOrig + "_I" + ToString(aKIm) +"_I" + ToString(0) + "_S"+ ToString(0);

      std::string NameBareMaster=aMasterName.substr(0,aMasterName.find_last_of("."));
      std::string NameBareSec  = aSecName.substr(0,aSecName.find_last_of("."));

      std::string NameOut=NameBareMaster+"_"+NameBareSec;
      std::string aNameEpip21=NameOut+".tif";

      // im2 --> Epip21
      std::string Im2_to_Epip21_GeoX=NameOut+"_GEOX.tif";
      std::string Im2_to_Epip21_GeoY=NameOut+"_GEOY.tif";
      std::string Im2_to_Epip21_Masq=NameOut+"_Masq.tif";
      // Epip21 --> im2
      std::string Epip21_to_im2_GeoX=NameOut+"_EpIm_GEOX.tif";
      std::string Epip21_to_im2_GeoY=NameOut+"_EpIm_GEOY.tif";
      std::string Epip21_to_im2_Masq=NameOut+"_EpIm_Masq.tif";

      // calculer les grilles de déformation
      Tiff_Im     Sec2EpipGeoX(Epip21_to_im2_GeoX.c_str());
      Tiff_Im     Sec2EpipGeoY(Epip21_to_im2_GeoY.c_str());
      //std::cout<<"Sec2EpipGeoY==> "<<Sec2EpipGeoY.sz()<<std::endl;
      Tiff_Im     Sec2EpipMasq(Epip21_to_im2_Masq.c_str());
      TIm2D<float,double> aTImSec2EpipGeoX(SzSec);
      TIm2D<float,double> aTImSec2EpipGeoY(SzSec);
      Im2D_Bits<1> aImSec2EpipMasq(SzSec.x,SzSec.y);
      // Export Epipolars direclty as they have been calculated
      Tiff_Im  anEpip21(aNameEpip21.c_str());

       // 2. Secondary : compute box in epipolar domain

       ELISE_COPY
      (
           aTImSec2EpipGeoX.all_pts(),
           trans(Sec2EpipGeoX.in(),BoxSec._p0),
           aTImSec2EpipGeoX.out()
      );

       ELISE_COPY
      (
           aTImSec2EpipGeoY.all_pts(),
           trans(Sec2EpipGeoY.in(),BoxSec._p0),
           aTImSec2EpipGeoY.out()
      );

       ELISE_COPY
      (
           aImSec2EpipMasq.all_pts(),
           trans(Sec2EpipMasq.in(),BoxSec._p0),
           aImSec2EpipMasq.out()
      );

       int HomXmin=1e9;
       int HomYmin=1e9;
       int HomXmax=-1e9;
       int HomYmax=-1e9;

       Pt2di aP=Pt2di(0,0);
       for (aP.x=0;aP.x<SzSec.x;aP.x++)
         {
           for (aP.y=0;aP.y<SzSec.y;aP.y++)
             {
               if (aImSec2EpipMasq.get(aP.x,aP.y))
                 {
                   double aGeox=aTImSec2EpipGeoX.get(aP);
                   double aGeoy=aTImSec2EpipGeoY.get(aP);

                   if (aGeox>HomXmax) HomXmax=aGeox;
                   if (aGeox<HomXmin) HomXmin=aGeox;

                   if (aGeoy>HomYmax) HomYmax=aGeoy;
                   if (aGeoy<HomYmin) HomYmin=aGeoy;
                 }
             }
         }
       Box2di aBoxSecInEpip(round_ni(Pt2dr(HomXmin,HomYmin)),round_ni(Pt2dr(HomXmax,HomYmax)));
       std::cout<<"aBoxSecInEpip   "<<aBoxSecInEpip<<std::endl;
       TIm2D<float,double> aTImSecEpip(aBoxSecInEpip.sz());

       /*3.1 Master*/
       Tiff_Im Epip2SecGeoX(Im2_to_Epip21_GeoX.c_str());
       Tiff_Im Epip2SecGeoY(Im2_to_Epip21_GeoY.c_str());
       Tiff_Im Epip2SecMasq(Im2_to_Epip21_Masq.c_str());

       TIm2D<float,double> aTImEpip2SecGeoX(aBoxSecInEpip._p1-aBoxSecInEpip._p0);
       TIm2D<float,double> aTImEpip2SecGeoY(aBoxSecInEpip._p1-aBoxSecInEpip._p0);

       Pt2di SzInEpSec=aBoxSecInEpip._p1-aBoxSecInEpip._p0;
       Im2D_Bits<1> aImEpip2SecMasq(SzInEpSec.x,SzInEpSec.y);
       //TIm2DBits<1> aTImEpip2SecMasq(aImEpip2SecMasq);

       ELISE_COPY
      (
           aTImEpip2SecGeoX.all_pts(),
           trans(Epip2SecGeoX.in(),aBoxSecInEpip._p0),
           aTImEpip2SecGeoX.out()
      );
       ELISE_COPY
      (
           aTImEpip2SecGeoY.all_pts(),
           trans(Epip2SecGeoY.in(),aBoxSecInEpip._p0),
           aTImEpip2SecGeoY.out()
      );

       ELISE_COPY
      (
           aTImSecEpip.all_pts(),
           trans(anEpip21.in(),aBoxSecInEpip._p0),
           aTImSecEpip.out()
      );

       ELISE_COPY
      (
           aImEpip2SecMasq.all_pts(),
           trans(Epip2SecMasq.in(),aBoxSecInEpip._p0),
           aImEpip2SecMasq.out()
      );
      std::cout<<"Fill Image tiles ==> Im2 to Epip 21 Fill tiles  "<<std::endl;

       // Image --> Homography warped image
       ELISE_COPY(aTImSec2EpipGeoX.all_pts(),
                  2*((aTImSec2EpipGeoX.in()-aBoxSecInEpip.P0().x)/(aBoxSecInEpip.P1().x-aBoxSecInEpip.P0().x-1.0))-1.0,
                  aTImSec2EpipGeoX.out()
                  );

       ELISE_COPY(aTImSec2EpipGeoY.all_pts(),
                  2*((aTImSec2EpipGeoY.in()-aBoxSecInEpip.P0().y)/(aBoxSecInEpip.P1().y-aBoxSecInEpip.P0().y-1.0))-1.0,
                  aTImSec2EpipGeoY.out()
                  );

       // Homography warped image back to --> Image
       ELISE_COPY(aTImEpip2SecGeoX.all_pts(),
                  2*((aTImEpip2SecGeoX.in()-BoxSec.P0().x)/(BoxSec.P1().x-BoxSec.P0().x-1.0))-1.0,
                  aTImEpip2SecGeoX.out()
                  );

       ELISE_COPY(aTImEpip2SecGeoY.all_pts(),
                  2*((aTImEpip2SecGeoY.in()-BoxSec.P0().y)/(BoxSec.P1().y-BoxSec.P0().y-1.0))-1.0,
                  aTImEpip2SecGeoY.out()
                  );

       // 6. Save

       SaveIm(aPrefixSec+"_ORIG_EpIm_GEOX.tif",aTImSec2EpipGeoX._d,Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_EpIm_GEOY.tif",aTImSec2EpipGeoY._d,Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_EpIm_Masq.tif",aImSec2EpipMasq.data(),Box2di(Pt2di(0,0),SzSec));
       SaveIm(aPrefixSec+"_ORIG_GEOX.tif",aTImEpip2SecGeoX._d,Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       SaveIm(aPrefixSec+"_ORIG_GEOY.tif",aTImEpip2SecGeoY._d,Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       SaveIm(aPrefixSec+"_ORIG_Masq.tif",aImEpip2SecMasq.data(),Box2di(Pt2di(0,0),aBoxSecInEpip._p1-aBoxSecInEpip._p0));
       // Save Epipolars
       SaveIm(aPrefixEpip+"_Epip.tif",aTImSecEpip._d,Box2di(Pt2di(0,0),aBoxSecInEpip.sz()));
  }
}


void cAppliMICMAC::DoEstimHomWarpers()
{
  if (mZoomChanged)
    {
        int NbVues=mPrisesDeVue.size();
        //ELISE_ASSERT(NbVues>=2,"Nombre de vues non suffisant pour la correlation ");
        const cPriseDeVue * aMaster= mPrisesDeVue.at(0);
        std::string aMasterName=aMaster->Name();
        if (NbVues>=2)
          {
            std::list<std::string> aLCom;
            bool WithHomol=true;
            if (WithHomol)
              {
                for (int itSecIm=1; itSecIm<NbVues;itSecIm++)
                  {

                    const cPriseDeVue * aSec=mPrisesDeVue.at(itSecIm);
                    std::string aSecName= aSec->Name();

                    if (GeomImages()==eGeomImageOri)
                      {

                        std::string NameBareMaster=aMasterName.substr(0,aMasterName.find_last_of("."));
                        std::string NameBareSec  = aSecName.substr(0,aSecName.find_last_of("."));

                        std::string NameOut=NameBareMaster+"_"+NameBareSec;
                        // im2 --> Epip21
                        /*std::string Im2_to_Epip21_GeoX=NameOut+"_GEOX.tif";
                        std::string Im2_to_Epip21_GeoY=NameOut+"_GEOY.tif";
                        std::string Im2_to_Epip21_Masq=NameOut+"_Masq.tif";
                        // Epip21 --> im2
                        std::string Epip21_to_im2_GeoX=NameOut+"_EpIm_GEOX.tif";
                        std::string Epip21_to_im2_GeoY=NameOut+"_EpIm_GEOX.tif";
                        std::string Epip21_to_im2_EpMasq=NameOut+"_EpIm_Masq.tif";

                        bool AllExist=ELISE_fp::exist_file(Im2_to_Epip21_GeoX) &&
                                      ELISE_fp::exist_file(Im2_to_Epip21_GeoY) &&
                                      ELISE_fp::exist_file(Epip21_to_im2_GeoX) &&
                                      ELISE_fp::exist_file(Epip21_to_im2_GeoY) &&
                                      ELISE_fp::exist_file(Im2_to_Epip21_Masq) &&
                                      ELISE_fp::exist_file(Epip21_to_im2_EpMasq);
                        int aCurDeZoom=this->CurEtape()->DeZoomTer();
                        bool SizesNoMatch=false;
                        // Compare Sizes to decide wether to recompute epipolars
                        if (ELISE_fp::exist_file(Epip21_to_im2_GeoX))
                          {
                             Tiff_Im ImEpipolarEpGeox(Epip21_to_im2_GeoX.c_str());
                             Pt2di aSzZoom=Std2Elise(aSec->IMIL()->Sz(aCurDeZoom));
                             std::cout<<"taille de limage dezoom "<<aSec->IMIL()->Sz(aCurDeZoom)<<std::endl;
                             SizesNoMatch=(aSzZoom!=ImEpipolarEpGeox.sz());
                             std::cout<<"SZ IMAGE "<<aMaster->LoadedIm().SzIm()<<"   "<<ImEpipolarEpGeox.sz()<<" CUD DEZOOM "<<aCurDeZoom<<std::endl;
                          }
                        if (!AllExist||SizesNoMatch)
                          {*/

                            int aCurDeZoom=this->CurEtape()->DeZoomTer();
                              std::string aComHom=MMBinFile(MM3DStr)               + BLANK +
                                                  "TestLib"                        + BLANK +
                                                  "OneReechHom"                    + BLANK +
                                                  aMasterName                      + BLANK +
                                                  aSecName                         + BLANK +
                                                  NameOut+".tif"                   + BLANK +
                                                  "ExportGeoXY=1"                 + BLANK +
                                                  "PostMasq="+"_Masq"             + BLANK +
                                                  "ScaleReech=" + ToString(aCurDeZoom)  ;
                              aLCom.push_back(aComHom);
                              std::cout<<"COMMANDE HOMOGRAPHY "<<aComHom<<std::endl;

                      }
                  }
              }
            if (aLCom.size()>=1)
              {
               cEl_GPAO::DoComInSerie(aLCom);
               aLCom.clear();
              }

            /*
            bool WithOri=false;
            if (WithOri)
              {
                for (int itSecIm=1; itSecIm<NbVues;itSecIm++)
                  {

                    const cPriseDeVue * aSec=mPDVBoxGlobAct.at(itSecIm);
                    CamStenope * CamSec=aSec->GetOri();

                    ElRotation3D aRM=CamMaster->Orient();
                    ElRotation3D aRS=CamSec->Orient();

                    // ROT C1==>C2! C1-->M-->C2
                    ElRotation3D aRot12=aRS*aRM.inv();

                    //ROT C2==>C1
                    ElRotation3D aRot21=aRot12.inv();
                    aRot21.tr()=vunit(aRot21.tr());

                    // OriRel == aRot21
                    //Calcul Matrice Essentielle
                    ElMatrix<double> aR  = aRot12.Mat();
                    Pt3dr            aTr = aRot21.tr();

                    std::cout<<" Base :::: "<<aTr<<std::endl;
                    ElMatrix<double> aT(1,3);
                    aT(0,0) = aTr.x;
                    aT(0,1) = aTr.y;
                    aT(0,2) = aTr.z;

                    //R * [R^t t]x  , o? x c'est skew matrix
                    ElMatrix<double> aRtT = aR.transpose() * aT;


                    ElMatrix<double> aRtTx(3,3);
                    aRtTx(0,1) = -aRtT(0,2);
                    aRtTx(1,0) =  aRtT(0,2);
                    aRtTx(0,2) =  aRtT(0,1);
                    aRtTx(2,0) = -aRtT(0,1);
                    aRtTx(1,2) = -aRtT(0,0);
                    aRtTx(2,1) =  aRtT(0,0);

                    ElMatrix<double> aMatEss = aR * aRtTx;
                    std::cout << "size MatEss=" << aMatEss.Sz() << "\n";
                    double ** dataMat=aMatEss.data();
                    std::cout<<"Matrice Essentielle "<<std::endl;
                    for (int LN=0;LN<3;LN++)
                      {
                       double *LINE=dataMat[LN];
                       std::cout<<LINE[0]<<"  "<<LINE[1]<<" "<<LINE[2]<<std::endl;
                      }

                    // Save Mat Essentielle as Homographies
                    mH1To2=cElHomographie::FromMatrix(aMatEss);
                    mH2To1=mH1To2.Inverse();
                    AVecOfGeoTransfo.push_back(std::make_pair(mH1To2,mH2To1));

                  }
              }
            */

          }
        mZoomChanged=false;
  }
}
void cAppliMICMAC::DoEstimWarpersPDVs()
{
  if (mZoomChanged)
    {
        // compute epipolar pairs and generate Epipolar guiding grids
        int NbVues=mPrisesDeVue.size();
        ELISE_ASSERT(NbVues>=2,"Nombre de vues non suffisant pour la correlation ");
        const cPriseDeVue * aMaster=mPrisesDeVue.at(0);
        std::string aNameOri=DirOfFile(aMaster->NameGeom());
        if (GeomImages()==eGeomGen)
          {
            std::size_t aPosl=aNameOri.find_first_of("/");
            aNameOri=aNameOri.substr(aPosl+1,-1);
            std::size_t aPosr=aNameOri.find_last_of("/");
            aNameOri=aNameOri.substr(0,aPosr);

          }
        if (GeomImages()==eGeomImageOri)
          {
            std::size_t aPosr=aNameOri.find_last_of("/");
            aNameOri=aNameOri.substr(0,aPosr);
          }

        //std::cout<<DirOfFile(aMaster->NameGeom())<<std::endl;
        //bool IsExistFile = ELISE_fp::exist_file(aFullName);

        // compute Epipolar images names and check if they already exist
         std::list<std::string> aLCom;
         //cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc("./");
         for (int itSecIm=1; itSecIm<NbVues;itSecIm++)
             {
              std::string NameBareOri=aNameOri.substr(aNameOri.find_first_of("-")+1,-1);
              std::cout<<"Name ori bare ==> "<<NameBareOri<<" name ori "<<aNameOri<<std::endl;
               const cPriseDeVue * aSec=mPrisesDeVue.at(itSecIm);


               /*
               std::string aNameBareEpip12,aNameBareEpip21,aNameEpip12,aNameEpip21;

               if (GeomImages()==eGeomGen)
                 {
                     aNameEpip12=aICNM->NameImEpip(NameBareOri,aMaster->Name(),aSec->Name());
                     aNameEpip21=aICNM->NameImEpip(NameBareOri,aSec->Name(),aMaster->Name());
                     //without extension
                     aNameBareEpip12=aNameEpip12.substr(0,aNameEpip12.find_last_of(".")); //.tif
                     aNameBareEpip21=aNameEpip21.substr(0,aNameEpip21.find_last_of(".")); //.tif
                 }
               if (GeomImages()==eGeomImageOri)
                 {
                   std::string aMasterN=aMaster->Name();
                   std::string aSecN=aSec->Name();
                   bool Inverse=aMasterN>aSecN;
                   if (Inverse) ElSwap(aMasterN,aSecN);

                   std::cout<<"Master "<<aMasterN<<"  aSec "<<aSecN<<std::endl;
                   CamStenope * aCam1= aMaster->Geom().GetOriNN();
                   CamStenope * aCam2= aSec->Geom().GetOriNN();
                   double aScale=1.0;
                   cCpleEpip aCple("./",
                                   aScale,
                                   Inverse ? *aCam2:*aCam1,aMasterN,
                                   Inverse ? *aCam1:*aCam2,aSecN
                                   );
                   aNameEpip12=aCple.Dir()+aCple.LocNameImEpi(Inverse ? false : true);
                   aNameEpip21=aCple.Dir()+aCple.LocNameImEpi(Inverse ? true : false);
                   aNameBareEpip12=aNameEpip12.substr(0,aNameEpip12.find_last_of(".")); //.tif
                   aNameBareEpip21=aNameEpip21.substr(0,aNameEpip21.find_last_of(".")); //.tif
                 }
               std::cout<<"Name Epip 12 ==> "<<aNameEpip12<<std::endl;
               std::cout<<"Name Epip 21 ==> "<<aNameEpip21<<std::endl;
               // im1 --> Epip12
               std::string Im1_to_Epip12_GeoX=aNameBareEpip12+"_GEOX.tif";
               std::string Im1_to_Epip12_GeoY=aNameBareEpip12+"_GEOY.tif";
               // im2 --> Epip21
               std::string Im2_to_Epip21_GeoX=aNameBareEpip21+"_GEOX.tif";
               std::string Im2_to_Epip21_GeoY=aNameBareEpip21+"_GEOY.tif";

               // Epip12 --> im1
               std::string Epip12_to_im1_GeoX=aNameBareEpip12+"_EpIm_GEOX.tif";
               std::string Epip12_to_im1_GeoY=aNameBareEpip12+"_EpIm_GEOY.tif";

               // Epip21 --> im2
               std::string Epip21_to_im2_GeoX=aNameBareEpip21+"_EpIm_GEOX.tif";
               std::string Epip21_to_im2_GeoY=aNameBareEpip21+"_EpIm_GEOX.tif";

               bool AllExist=ELISE_fp::exist_file(aNameEpip12) &&
                             ELISE_fp::exist_file(aNameEpip21) &&
                             ELISE_fp::exist_file(Im1_to_Epip12_GeoX) &&
                             ELISE_fp::exist_file(Im1_to_Epip12_GeoY) &&
                             ELISE_fp::exist_file(Im2_to_Epip21_GeoX) &&
                             ELISE_fp::exist_file(Im2_to_Epip21_GeoY) &&
                             ELISE_fp::exist_file(Epip12_to_im1_GeoX) &&
                             ELISE_fp::exist_file(Epip12_to_im1_GeoY) &&
                             ELISE_fp::exist_file(Epip21_to_im2_GeoX) &&
                             ELISE_fp::exist_file(Epip21_to_im2_GeoY);

               bool SizesNoMatch=false;
               // Compare Sizes to decide wether to recompute epipolars
               if (ELISE_fp::exist_file(Epip12_to_im1_GeoX))
                 {
                    Tiff_Im ImEpipolarEpGeox(Epip12_to_im1_GeoX.c_str());
                    SizesNoMatch=(aMaster->SzIm()!=ImEpipolarEpGeox.sz());
                    std::cout<<"SZ IMAGE "<<aMaster->LoadedIm().SzIm()<<"   "<<ImEpipolarEpGeox.sz()<<" CUD DEZOOM "<<aCurDeZoom<<std::endl;
                 }
               if (!AllExist || SizesNoMatch)
                   {

                   */

                     int aCurDeZoom=this->CurEtape()->DeZoomTer();
                     std::string aComEpip=MMBinFile(MM3DStr)          + BLANK +
                                          "CreateEpip"                + BLANK +
                                          aMaster->Name()             + BLANK +
                                          aSec->Name()                + BLANK +
                                          aNameOri                    + BLANK +
                                          "XCorrecOri=1"              + BLANK +
                                         "Scale=" + ToString(aCurDeZoom) + BLANK +
                                          "InParal=1"                 + BLANK +
                                          "DoEpiAbs=0"                + BLANK+
                                          "Export12WayGeoxy=1";
                     aLCom.push_back(aComEpip);
                     std::cout<<"COMMANDE EPIP "<<aComEpip<<std::endl;
                  //}
             }

         // Or apply homography to secondary images
         /*
         int NBVUES=mPDVBoxGlobAct.size();
           if (NBVUES>=2) // More than one image
             {
               std::string aName1=mPDVBoxGlobAct.at(0)->Name();
               cElemAppliSetFile anEASF;
               anEASF.Init(aName1);
               std::list<std::string> aLCom;
               for (int im2=1;im2<NBVUES;im2++)
                 {
                   std::string aName2=mPDVBoxGlobAct.at(im2)->Name();
                   std::string aNameRes = anEASF.mDir +  "RegHom_" +aName2 + ".tif";
                   if (! ELISE_fp::exist_file(aNameRes))
                   {
                       // std::cout << "RES = " << aNameRes << "\n";
                       std::string aCom =  MM3dBinFile_quotes("TestLib")
                                           + " OneReechHom "
                                           +   aName1
                                           +  " " + anEASF.mDir + aName2
                                           +  " " +  aNameRes
                                           +  " PostMasq=" + "_Masq";

                       aLCom.push_back(aCom);
                       // std::cout << "COM= " << aCom << "\n";
                   }
                 }
               cEl_GPAO::DoComInParal(aLCom);
             }
           */
         if (aLCom.size()>=1)
           {
            cEl_GPAO::DoComInSerie(aLCom);
           }
         mZoomChanged=false;
    }
}

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

        //std::cout<<"mStepPlani   "<<mStepPlani<<"  mStepZ "<<mStepZ<<std::endl;

        //std::cout<<"ZZZZZZMINNNN   "<<mLTer->PxMin() <<"   ZZZZZMMMMMAX   "<<mLTer->PxMax()<<std::endl;
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

            /*Im2D_Bits<1> aNewMasq= mLTer->ImMasqTer();
            int aCurDeZoom=this->CurEtape()->DeZoomTer();
            ELISE_COPY
             (
               aNewMasq.all_pts(),
               erod_d8(aNewMasq.in(0)==1,7*aCurDeZoom),
               aNewMasq.out()
             );
            Tiff_Im::CreateFromIm(aNewMasq,"MASQSSSSSSSS"+to_string(aCurDeZoom)+".tif");*/


            for (int anX = mX0Ter ; anX <  mX1Ter ; anX++)
            {
                for (int anY = mY0Ter ; anY < mY1Ter ; anY++)
                {
                    if (IsInTer(anX,anY))
                      {
                        ElSetMin(mZMinGlob,mTabZMin[anY][anX]);
                        ElSetMax(mZMaxGlob,mTabZMax[anY][anX]);
                      }
                }
            }


        mGpuSzD = 0;
        if (mCurEtape->UseGeomDerivable())
        {
            //std::cout<<" SZ GEOM DERIVABLE "<<std::endl;
            mGpuSzD = mCurEtape->SzGeomDerivable();
            std::cout<<"   SZZZZZZZ  "<<mGpuSzD<<std::endl;
            Pt2di aSzOrtho = aBox.sz() + mCurSzVMax * 2;
            Pt2di aSzTab =  Pt2di(3,3) + aSzOrtho/mGpuSzD;
            //std::cout<<" SSIZE IMAGE OFFSETS "<<aSzTab<<"   $$$$$$$$$$$$$$$$$$$$$ "<<std::endl;
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

    //int RandInd=std::rand();
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

             //std::cout<<"  SZ000   "<<aGLI_0.getSizeImage()<<"   SZKKK    "<<aGLI_K.getSizeImage()<<"  EROD  "<<aSzErod<<std::endl;

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
                       //if (aDLocOkTerDil[anY][anX])
                       //{
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
                               if (aDLocOkTerDil[anY][anX])
                                 {
                                    aOkOr[anY][anX] =  1;
                                 }
                           }
                       //}
                       anIndY += aStep;

                   }
                   anIndX += aStep;
             }

             // Save mappings locations that are stored in mGeoX and mGeoY
             /*std::string aPrefX   = FullDirMEC() + "MMV1Ortho_Pid" + ToString(mm_getpid()) + "_Z" + ToString(aZ-aZ0) ;
             std::string aPrefixGeo = aPrefX + "_I" + ToString(aKIm) + "_S"+ ToString(aKScale);
             Box2di  aBoxD(Pt2di(mX0UtiDilTer,mY0UtiDilTer),Pt2di(mX1UtiDilTer,mY1UtiDilTer));
             SaveIm(aPrefixGeo+"_GEOX.tif",mGeoX.data(),aBoxD);
             SaveIm(aPrefixGeo+"_GEOY.tif",mGeoY.data(),aBoxD);*/

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

bool  cAppliMICMAC::InitZRef(int aZ,int aZRef, std::string aPrefX, eModeInitZ aMode)
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

    //int RandInd=std::rand();
    for (int aKIm= aKFirstIm ; aKIm<mNbIm ; aKIm++)
    {
        /*if (aKIm==0)
        {
                std::cout<<" WWWWWWWWWW E HAVE COMPUTED THE FIRST MASTER ORTHO =========>  "<<std::endl;
        }*/
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

             //std::cout<<"  SZ000   "<<aGLI_0.getSizeImage()<<"   SZKKK    "<<aGLI_K.getSizeImage()<<"  EROD  "<<aSzErod<<std::endl;
             // Calcul de l'ortho image et de l'image OK Ortho
             double aStep = 1.0/ElMax(1,mGpuSzD); // Histoire de ne pas diviser par 0
             double anIndX = 0.0;
             //std::cout<<mX0UtiDilTer<<"  "<<mX1UtiDilTer<<"  "<<mY0UtiDilTer<<"   "<<mY1UtiDilTer<<std::endl;
             for (int anX = mX0UtiDilTer ; anX <  mX1UtiDilTer ; anX++)
             {
                   double anIndY = 0.0;
                   //std::cout<<"anX   "<<anX<<std::endl;
                   for (int anY = mY0UtiDilTer ; anY < mY1UtiDilTer ; anY++)
                   {
                     aOkOr[anY][anX] = 0;
                     aDOrtho[anY][anX] = 0.0;
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
                             if (aDLocOkTerDil[anY][anX])
                               {
                                  aOkOr[anY][anX] =  1;
                               }
                         }

                       /*if (aDLocOkTerDil[anY][anX])
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

                            aDOrtho[anY][anX] = (tGpuF)anInt->GetVal(aDataIm,aPIm);
                           // Peu importe aGLI_0 ou aGLI_K
                           if (aGLI_0.IsOk(aPIm.x,aPIm.y))
                           {
                               //aDOrtho[anY][anX] = (tGpuF)anInt->GetVal(aDataIm,aPIm);
                               aOkOr[anY][anX] =  1;
                           }
                       }
                       */
                       anIndY += aStep;

                   }
                   anIndX += aStep;
             }
              //std::cout<<"fill for each scale "<<std::endl;

             // Save mappings locations that are stored in mGeoX and mGeoY
             if (mGpuSzD)
                 {
                     //std::cout<<" akimmmmmmmmmmmmmmmmmmmm   ===  "<<aKIm<<" PID "<<aPrefX<<std::endl;
                     std::string aPrefixGeo = aPrefX+ "_Z" + ToString(aZ-aZRef)  + "_I" + ToString(aKIm) + "_S"+ ToString(aKScale);
                     int NbX = (mX1UtiDilTer-mX0UtiDilTer + mGpuSzD) / mGpuSzD;
                     int NbY = (mY1UtiDilTer-mY0UtiDilTer + mGpuSzD) / mGpuSzD;
                     Box2di  aBoxD(Pt2di(0,0),Pt2di(NbX,NbY));
                     //std::cout<<mGeoX.sz()<< "  "<<aBoxD.sz()<<std::endl;
                     //std::cout<<mGeoY.sz()<< "  "<<aBoxD.sz()<<std::endl;
                     SaveIm(aPrefixGeo+"_GEOX.tif",mGeoX.data(),aBoxD);
                     SaveIm(aPrefixGeo+"_GEOY.tif",mGeoY.data(),aBoxD);
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

    mFirstZIsInit = false; // recomputing master ortho although not necessary

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
            const Box2di & ,// aBox,
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

	{
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
	// Case where we generate the ortho photos and call processes
	else if (aTC.MutiCorrelOrthoExt().IsInit())
	//	MutiCorrelOrthoExt
	{
	      std::cout<<"======== ZPAS ========   "<<GeomDFPx().PasPxRel0()<<std::endl;
	      std::cout<<" ZMIN GLOBAL "<<mZMinGlob<<" ZMAX GLOBAL "<<mZMaxGlob<<std::endl;
	      // Generate grids to compute descriptors in the epipolar geometry
	      std::string aPrefixGlobIm = FullDirMEC() + "MMV1Ortho_Pid" + ToString(mm_getpid()) ;
             const cMutiCorrelOrthoExt aMCOE = aTC.MutiCorrelOrthoExt().Val();
             int mDeltaZ = aMCOE.DeltaZ().Val();
             std::string aPrefixGlob = aPrefixGlobIm; //FullDirMEC() + "MMV1Ortho_Pid" + ToString(mm_getpid()) ;
             for (int aZ0=mZMinGlob ; aZ0<mZMaxGlob ; aZ0+=mDeltaZ)
             {
                 /********************************************************************************/

                 // Save Original images because they will ommitted every loop between aZ0 and aZ1
                 for (int aKIm= 0 ; aKIm<mNbIm ; aKIm++)
                 {
                     cGPU_LoadedImGeom & aaGLI_0 = *(mVLI[aKIm]);
                     const std::vector<cGPU_LoadedImGeom *> &  aaVGLI = aaGLI_0.MSGLI();

                     for (int aKScale = 0; aKScale<int(aaVGLI.size()) ; aKScale++)
                     {
                          cGPU_LoadedImGeom & aaGLI_K = *(aaVGLI[aKScale]);
                          ELISE_ASSERT(aaGLI_0.VDataIm()==aaGLI_K.VDataIm(),"Internal incohe in MulScale correl");
                          float ** aaDataIm =  aaGLI_0.VDataIm()[aKScale];
                          std::string aPrefiX= aPrefixGlobIm + "_I" + ToString(aKIm) + "_S"+ ToString(aKScale);
                          Pt2di SzImage=aaGLI_0.getSizeImage();
                          Box2di ABox(Pt2di(0,0),SzImage);
                          std::cout<<"SAVE IMAGE :::::: "<<aPrefiX<<std::endl;
                          SaveIm(aPrefiX+"_ORIG.tif",aaDataIm,ABox);
                     }
                 }
                 if (aMCOE.OrthFileModeleArch().IsInit())
                   {
                     // save transformations from ORIG images to Epipolar images or homography corrected images
                     if (
                         aMCOE.OrthoResol().IsInit())
                        // && (std::stof(aMCOE.OrthoResol().Val())==1.0)

                     GenerateBoxesImEpip_EpipIm(mVLI,aPrefixGlobIm);
                   }

                 // export nappes sup et min
                 /*
                 Im2D_INT2  aImZMin = mLTer->KthNap(0).mImPxMin;
                 Im2D_INT2  aImZMax = mLTer->KthNap(0).mImPxMax;

                 std::string aNameZMin=aPrefixGlobIm+"_ZMINNN.tif";
                 std::string aNameZMax=aPrefixGlobIm+"_ZMAXXX.tif";

                 Tiff_Im::CreateFromIm(aImZMin,aNameZMin);
                 Tiff_Im::CreateFromIm(aImZMax,aNameZMax);
                 */
                 /********************************************************************************/
                  int aZ1= ElMin(mZMaxGlob,aZ0+mDeltaZ);
		  Box2di  aBoxEmpty(Pt2di(0,0),Pt2di(0,0));
		  std::vector<Box2di>  aVecBoxDil;
		  std::vector<Box2di>  aVecBoxUti;
		  bool allOkZ=true;
                  for (int aZ=aZ0 ; aZ<aZ1 ; aZ++)
                  {
                        std::string aPrefixZ =    aPrefixGlob + "_Z" + ToString(aZ-aZ0) ;
                        bool OkZ= InitZRef(aZ,aZ0, aPrefixGlob,eModeNoMom); // create deformation maps
                        allOkZ= allOkZ && OkZ;
                        //std::cout<<"Init Z REF DONE !"<<std::endl;
                        if (OkZ)
                        {
                            Box2di  aBoxDil(Pt2di(mX0UtiDilTer,mY0UtiDilTer),Pt2di(mX1UtiDilTer,mY1UtiDilTer));
                            Box2di  aBoxUti(Pt2di(mX0UtiTer,mY0UtiTer),Pt2di(mX1UtiTer,mY1UtiTer));

			    // Memorize vector of boxes
			    aVecBoxDil.push_back(aBoxDil);
			    aVecBoxUti.push_back(aBoxUti);

                            SaveIm(aPrefixZ+"_OkT.tif",mDOkTer,aBoxUti);
                            //std::vector<std::vector<cGPU_LoadedImGeom *> > mVScaIm
			    //  Save ortho and Masks  for all images
			    for (int aKIm=0 ; aKIm<int(mVLI.size()) ; aKIm++)
                            {
                                 for (int aKScale=0; aKScale<mNbScale ; aKScale++)
                                 {
                                     // cGPU_LoadedImGeom & aGLI_0 = *(mVLI[aKIm]);
                                     cGPU_LoadedImGeom & aGLI_K =  *(mVScaIm[aKScale][aKIm]);
                                     std::string aPrefixZIm = aPrefixZ + "_I" + ToString(aKIm) + "_S"+ ToString(aKScale);
                                     //SaveIm(aPrefixZIm+"_O.tif",aGLI_K.DataOrtho(),aBoxDil);
				     SaveIm(aPrefixZIm+"_M.tif",aGLI_K.DataOKOrtho(),aBoxDil);
                                 }
                            }

                        }
			else
			{
                            // Generate information for no data
			    aVecBoxDil.push_back(aBoxEmpty);
			    aVecBoxUti.push_back(aBoxEmpty);
                            std::string aNameNone  = aPrefixZ + "_NoData";
			    ELISE_fp aFile(aNameNone.c_str(),ELISE_fp::WRITE);
			    aFile.close();
			}
		  }

		  if (allOkZ) // sure to have already writtem images and grids to warp descriptors
		    {

			//  Call external command
			std::string   aCom =  aMCOE.Cmd().Val() // "MMVII  DM4MatchMultipleOrtho "
			  +  " " + aPrefixGlob
					      +  " " + ToString(aZ1-aZ0)          // Number of Ortho
					      +  " " + ToString(int(mVLI.size()))  // Number of Images
					      +  " " + ToString(mNbScale)  // Number of Scale
					      +  " " + ToString(  mCurSzV0)     // Size of Window
					      +  " " + ToString( mGIm1IsInPax)     // Are we in mode Im1 Master
				       ;
			if (aMCOE.Options().IsInit())
			   aCom = aCom + " " + QUOTE(aMCOE.Options().Val());
			//ADDED STATEMENT ON MODEL ARCHITECTURE AND MODELS PARAMETERES FOR INFERENCE
			if (aMCOE.OrthFileModeleArch().IsInit())
			   aCom = aCom + " " +  "CNNArch=" + QUOTE(aMCOE.OrthFileModeleArch().Val());
			if (aMCOE.OrthFileModeleParams().IsInit())
			   aCom = aCom + " " +  "CNNParams=" + QUOTE(aMCOE.OrthFileModeleParams().Val());
			if (aMCOE.OrthoResol().IsInit())
			   aCom = aCom + " " +  "RESOL=" + QUOTE(aMCOE.OrthoResol().Val());
			// add cuda
			if (aMCOE.Cuda().IsInit())
			  {
			    aCom= aCom + " "+"UseCuda=" +QUOTE(ToString(aMCOE.Cuda().Val()));
			  }
			else
			  {
			    aCom= aCom + " "+"UseCuda=" +ToString(0);
			  }
			std::cout<<"COMMAND 2 TEST "<<aCom<<std::endl;
			System(aCom);

                        // Fill cube with computed similarities
                        for (int aZ=aZ0 ; aZ<aZ1 ; aZ++)
                        {
                            int aKBox = (aZ-aZ0);
                            const Box2di & aBoxU = aVecBoxUti.at(aKBox);
                            const Box2di & aBoxDil = aVecBoxDil.at(aKBox);
                            bool  CorDone = (aBoxU.sz() != Pt2di(0,0));
                            if (CorDone)
                            {
                                // Read similarity
                                std::string aNameSim =    aPrefixGlob + "_Z" + ToString(aZ-aZ0) + "_Sim.tif"  ;
                                Im2D_REAL4  aImSim = Im2D_REAL4::FromFileStd(aNameSim);
                                TIm2D<REAL4,REAL8> aTImSim(aImSim);

                                    // Read masq terrain
                                std::string aNameOkT =    aPrefixGlob + "_Z" + ToString(aZ-aZ0) + "_OkT.tif"  ;
                                Im2D_U_INT1  aImOkT = Im2D_U_INT1::FromFileStd(aNameOkT);
                                TIm2D<U_INT1,INT4> aTImOkT(aImOkT);


                                // Parse image to fill cost for optimizer
                                Pt2di aPUti;
                                for (aPUti.x = aBoxU._p0.x ; aPUti.x <  aBoxU._p1.x ; aPUti.x++)
                                {
                                     for (aPUti.y=aBoxU._p0.y ; aPUti.y<aBoxU._p1.y ; aPUti.y++)
                                     {
                                           bool Ok1 = aTImOkT.get(aPUti-aBoxU._p0);
                                           /*
                                           bool Ok2 = mDOkTer[aPUti.y][aPUti.x];
                                           ELISE_ASSERT(Ok1==Ok2,"aImOkT.get coh");
                                           */
                                           if (Ok1)
                                           {
                                               Pt2di aPDil = aPUti - aBoxDil._p0;
                                               double aSim =  aTImSim.get(aPDil);
                                               mSurfOpt->SetCout(aPUti,&aZ,aSim);
                                           }
                                     }
                                }
                            }
                        }
                    }
                        // Purge temporary files
                        std::string aComPurge = SYS_RM + std::string(" ") + aPrefixGlob + "*";
                        System(aComPurge);
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
