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
#include "im_tpl/cPtOfCorrel.h"






         //================================================
         //================================================
         //================================================

void cRepartPtInteret::UpdatePond(Pt2di aP0,Im2D_REAL4 aIPond,bool Ajout)
{
   Pt2di aSzP = aIPond.sz();

   TIm2D<REAL4,REAL8> aTIPond(aIPond);
   for (int anX=-aSzP.x+1 ; anX<aSzP.x ; anX++)
   {
       for (int anY=-aSzP.y+1 ; anY<aSzP.y ; anY++)
       {
           Pt2di aP = aP0 + Pt2di(anX,anY);
           if (mTPondGlob.inside(aP))
           {
               double aMod = aTIPond.get(Pt2di(ElAbs(anX),ElAbs(anY)));
               double aPds = mTPondGlob.get(aP);
               if (Ajout)
                  aPds *= aMod;
               else
                  aPds /= aMod;
                mTPondGlob.oset(aP,aPds);
           }
       }
   }
}

cRepartPtInteret::cRepartPtInteret
(
    Im2D_REAL4       aImSc,
    Im2D_Bits<1>     aMasq,
    const cEquiv1D & anEqX,
    const cEquiv1D & anEqY,
    double           aDistRejet,
    double           aDistEvitement
) :
  mNbC      (anEqX.NbClasses(),anEqY.NbClasses()),
  mImSc     (aImSc),
  mTSc      (mImSc),
  mMasq     (aMasq),
  mTMasq    (mMasq),
  mSzGlob   (aImSc.sz()),
  mPondGlob (mSzGlob.x,mSzGlob.y,1.0),
  mTPondGlob (mPondGlob),
  mEqX      (anEqX),
  mEqY      (anEqY),
  mPX       (mNbC.x,mNbC.y,-1),
  mTPx      (mPX),
  mPY       (mNbC.x,mNbC.y,-1),
  mTPy      (mPY),
  mValRejet (1e-2),
  mDRejMax  (aDistRejet),
  mDEvMax   (aDistEvitement),
  mLastImPond (1,1)
{
}

Im2D_Bits<1> cRepartPtInteret::ItereAndResult()
{

    for (int aK=0 ; aK < 5 ; aK++)
    {
       OnePasse(ElMin(1.0,aK/3.0));
    }
    Im2D_Bits<1> aRes(mSzGlob.x,mSzGlob.y,0);
    TIm2DBits<1> aTRes(aRes);

    Pt2di aPC;
    for (aPC.x=0 ; aPC.x<mNbC.x ; aPC.x++)
    {
        for (aPC.y=0 ; aPC.y<mNbC.y ; aPC.y++)
        {
             Pt2di aP(mTPx.get(aPC),mTPy.get(aPC));
             if (aP.x != -1)
                aTRes.oset(aP,1);
        }
    }

    return aRes;
}
        


void cRepartPtInteret::OnePasse(double aProp)
{
    int aNb = round_up(mDEvMax);
    Im2D_REAL4 aIPond(aNb,aNb);

    double aDRej = mDRejMax*aProp;
    double aDEv = mDEvMax*aProp;
    for (int anX=0 ; anX <aNb; anX++)
    {
       for (int anY=0 ; anY <aNb; anY++)
       {
            double aV=0;
            double aD = hypot(anX,anY);
            if (aD < aDRej)
               aV = mValRejet;
            else if (aD <aDEv)
               aV =   mValRejet
                    + (1-mValRejet) * ((aD-aDRej)/(aDEv-aDRej));
            else
               aV=1.0;
            aIPond.data()[anY][anX] = (float) aV;
       }
    }

    Pt2di aPC;
    for (aPC.x=0;aPC.x<mNbC.x ;aPC.x++)
    {
        for (aPC.y=0;aPC.y<mNbC.y ;aPC.y++)
        {
            {
               Pt2di aP(mTPx.get(aPC),mTPy.get(aPC));
               if (aP.x!=-1)
                  UpdatePond(aP,mLastImPond,false);
            }

            int aX0,aX1;
            mEqX.ClasseOfNum(aX0,aX1,aPC.x);
            int aY0,aY1;
            mEqY.ClasseOfNum(aY0,aY1,aPC.y);

            Pt2di aPMax(-1,-1);
            double aVMax = -1e9;
            Pt2di aP;
            for (aP.x =aX0; aP.x<aX1 ; aP.x++)
            {
                for (aP.y =aY0; aP.y<aY1 ; aP.y++)
                {
                    if (mTMasq.get(aP))
                    {
                        double aV = mTSc.get(aP) * mTPondGlob.get(aP);
                        if (aV>aVMax)
                        {
                            aVMax = aV;
                            aPMax = aP;
                        }
                    }
                }
            }
            if (aPMax.x!=-1)
            {
                UpdatePond(aPMax,aIPond,true);
                mTPx.oset(aPC,aPMax.x);
                mTPy.oset(aPC,aPMax.y);
            }
        }
    }

    mLastImPond = aIPond;
}


         //================================================
         //================================================
         //================================================

double  cPtOfCorrel::PdsCorrel(double anX,double anY)
{
   return 1.0;
   // return 0.5+ ElMin(mVCor-ElAbs(anX),mVCor-ElAbs(anY));
}


#include "im_tpl/oper_assoc_exter.h"

class cQAC_XYV
{
public :
    double mPds;
    double mX;
    double mY;
    double mV;
};

class cQAC_Stat_XYV
{
public :
      cQAC_Stat_XYV();
      void AddCumul(int aSigne,const cQAC_Stat_XYV & aCum);
      void AddElem(int aSigne,const cQAC_XYV & anElem);
      void AutoCor_Gen(double & aLambda);

      double mSP;

      double mSx;
      double mSy;
      double mSv;

      double mSxx;
      double mSyy;
      double mSvv;

      double mSxy;
      double mSxv;
      double mSyv;
};

void cQAC_Stat_XYV::AddElem(int aSigne,const cQAC_XYV & anElem)
{
   double aP = anElem.mPds * aSigne;

    mSP += aP;
   
    mSx +=  aP * anElem.mX;
    mSy +=  aP * anElem.mY;
    mSv +=  aP * anElem.mV;

    mSxx +=  aP * anElem.mX * anElem.mX;
    mSyy +=  aP * anElem.mY * anElem.mY;
    mSvv +=  aP * anElem.mV * anElem.mV;

    mSxy +=  aP * anElem.mX * anElem.mY;
    mSxv +=  aP * anElem.mX * anElem.mV;
    mSyv +=  aP * anElem.mY * anElem.mV;
}

void cQAC_Stat_XYV::AddCumul(int aSigne,const cQAC_Stat_XYV & aCum)
{
    mSP += aSigne* aCum.mSP;

    mSx += aSigne* aCum.mSx;
    mSy += aSigne* aCum.mSy;
    mSv += aSigne* aCum.mSv;

    mSxx += aSigne* aCum.mSxx;
    mSyy += aSigne* aCum.mSyy;
    mSvv += aSigne* aCum.mSvv;

    mSxy += aSigne* aCum.mSxy;
    mSxv += aSigne* aCum.mSxv;
    mSyv += aSigne* aCum.mSyv;
}


cQAC_Stat_XYV::cQAC_Stat_XYV() :
    mSP (0.0),
    mSx (0.0),
    mSy (0.0),
    mSv (0.0),
    mSxx(0.0),
    mSyy(0.0),
    mSvv(0.0),
    mSxy(0.0),
    mSxv(0.0),
    mSyv(0.0)
{
}

void cPtOfCorrel::Init(const std::complex<int> & aPC,cQAC_XYV & aXYV)
{
   Pt2di aP(aPC.real(),aPC.imag());
   aXYV.mPds = 1;
   aXYV.mX = mTGrX.getproj(aP) ;
   aXYV.mY = mTGrY.getproj(aP);
   aXYV.mV = mTIn.getproj(aP);
}

void cQAC_Stat_XYV::AutoCor_Gen(double & aLambda)
{
    mSv /= mSP;
    mSvv /= mSP;
    mSvv -= ElSquare(mSv);

    mSx /= mSP;
    mSxx /= mSP;
    mSxx -= ElSquare(mSx);

    mSy /= mSP;
    mSyy /= mSP;
    mSyy -= ElSquare(mSy);

    mSxy /= mSP;
    mSxy -= mSx * mSy;

    mSxv /= mSP;
    mSxv -= mSx * mSv;

    mSyv /= mSP;
    mSyv -= mSy * mSv;


    ELISE_ASSERT(mSvv>-1e-5,"cPtOfCorrel::QuickAutoCor");
    mSvv = ElMax(1e-9,mSvv);
    double aA = mSxx - ElSquare(mSxv)/mSvv;
    double aB = mSxy - (mSxv*mSyv) / mSvv;
    double aC = mSyy -  ElSquare(mSyv)/mSvv;

    double aDelta = ElSquare(aA+aC)-4*(aA*aC-ElSquare(aB));
    ELISE_ASSERT(aDelta>-1e-5,"cPtOfCorrel::QuickAutoCor");
    aLambda = (aA+aC-sqrt(ElMax(0.0,aDelta))) / 2.0;
}


void cPtOfCorrel::QuickAutoCor_Gen(Pt2di aP0,double & aLambda,double & aSvv)
{
    double aS0  = 0.0;
    double aSv  = 0.0;
    aSvv = 0.0;

    double aSx  = 0.0;
    double aSxx = 0.0;
    double aSy  = 0.0;
    double aSyy = 0.0;

    double aSxy = 0.0;
    double aSxv = 0.0;
    double aSyv = 0.0;

    for (int aDx=-mVCor ; aDx <=mVCor ; aDx++)
    {
        for (int aDy=-mVCor ; aDy <=mVCor ; aDy++)
        {
            Pt2di aPV(aP0.x+aDx,aP0.y+aDy);
            double aV = mTIn.getproj(aPV);
            double aGx =   mTGrX.getproj(aPV);
            double aGy =   mTGrY.getproj(aPV);

            double aPds = PdsCorrel(aDx,aDy);
            aS0 += aPds;
            aSv += aV* aPds;
            aSvv += ElSquare(aV)* aPds;

            aSx += aGx* aPds;
            aSxx += ElSquare(aGx)* aPds;
            aSy += aGy* aPds;
            aSyy += ElSquare(aGy)* aPds;

            aSxy += aGx * aGy* aPds;
            aSxv += aGx * aV* aPds;
            aSyv += aGy * aV* aPds;
        }
    }
    aSv /= aS0;
    aSvv /= aS0;
    aSvv -= ElSquare(aSv);

    aSx /= aS0;
    aSxx /= aS0;
    aSxx -= ElSquare(aSx);

    aSy /= aS0;
    aSyy /= aS0;
    aSyy -= ElSquare(aSy);

    aSxy /= aS0;
    aSxy -= aSx * aSy;

    aSxv /= aS0;
    aSxv -= aSx * aSv;

    aSyv /= aS0;
    aSyv -= aSy * aSv;


    ELISE_ASSERT(aSvv>-1e-5,"cPtOfCorrel::QuickAutoCor");
    aSvv = ElMax(1e-9,aSvv);
    double aA = aSxx - ElSquare(aSxv)/aSvv;
    double aB = aSxy - (aSxv*aSyv) / aSvv;
    double aC = aSyy -  ElSquare(aSyv)/aSvv;

    double aDelta = ElSquare(aA+aC)-4*(aA*aC-ElSquare(aB));
    ELISE_ASSERT(aDelta>-1e-5,"cPtOfCorrel::QuickAutoCor");
    aLambda = (aA+aC-sqrt(ElMax(0.0,aDelta))) / 2.0;

// std::cout << "Deter " << (aSxx-aLambda)*(aSyy-aLambda)-ElSquare(aSxy) << "\n";
}

double cPtOfCorrel::QuickAutoCor(Pt2di aP0)
{
    double aLambda,aSvv;
    QuickAutoCor_Gen(aP0,aLambda,aSvv);
    return aLambda / aSvv;
}


double cPtOfCorrel::QuickAutoCorWithNoise(Pt2di aP0,double aBr)
{
    double aLambda,aSvv;
    QuickAutoCor_Gen(aP0,aLambda,aSvv);
    return aLambda / (aSvv+ElSquare(aBr));
}


double cPtOfCorrel::AutoCorTeta(Pt2di aP0,double aTeta,double anEpsilon)
{
    RMat_Inertie aMat;
    Pt2dr aDir = Pt2dr::FromPolar(anEpsilon,aTeta);

    for (int aDx=-mVCor ; aDx <=mVCor ; aDx++)
    {
        for (int aDy=-mVCor ; aDy <=mVCor ; aDy++)
        {
            Pt2di aPV(aP0.x+aDx,aP0.y+aDy);
            double aV = mTIn.getproj(aPV);
            double aVt =   aV
                         + aDir.x * mTGrX.getproj(aPV)
                         + aDir.y * mTGrY.getproj(aPV);
            double aPds = PdsCorrel(aDx,aDy);
            aMat.add_pt_en_place(aV,aVt,aPds);
        }
    }
    return (2*(1-aMat.correlation())) / ElSquare(anEpsilon);
}


double cPtOfCorrel::BrutaleAutoCor(Pt2di aP,int aNbTeta,double anEpsilon)
{
    double aRes = 1e9;
    for (int aK=0 ; aK<aNbTeta ; aK++)
    {
        double aCor = AutoCorTeta(aP,(aK*2*PI)/aNbTeta,anEpsilon);
        aRes = ElMin(aRes,aCor);
    }
    return aRes;
}




cPtOfCorrel::cPtOfCorrel(Pt2di aSz,Fonc_Num aF,int aVCor) :
   mSz       (aSz),
   mVCor     (aVCor),
   mImIn     (mSz.x,mSz.y),
   mTIn      (mImIn),
   mGrX      (mSz.x,mSz.y),
   mTGrX     (mGrX),
   mGrY      (mSz.x,mSz.y),
   mTGrY     (mGrY)
{
   double anAlphaDeriche = 1.0;
   //  int    aSzVCor = 2;
   //  double aEB = 10.0;  // estimation du bruit

   ELISE_COPY(mImIn.all_pts(), aF, mImIn.out());

   ELISE_COPY
   (
       mImIn.all_pts(),
       deriche(mImIn.in_proj(),anAlphaDeriche,10),
       Virgule(mGrX.out(),mGrY.out())
   );

}

void cPtOfCorrel::OnNewLine(int)
{
}

void cPtOfCorrel::UseAggreg
     (
            const std::complex<int> & aPC,
            const cQAC_Stat_XYV & aCVC
     )
{
    Pt2di aP = Std2Elise(aPC);
    cQAC_Stat_XYV aC2 = aCVC;
    double aL2;
    aC2.AutoCor_Gen(aL2);
    double aSvv2  = aC2.mSvv;


    if (mEstBr >= 0)
       mISc->oset(aP,aL2/(aSvv2+ElSquare(mEstBr)));
    else
       mISc->oset(aP,aL2);
    mMasq->oset(aP,(aSvv2>ElSquare(mSeuilEcart)) && (aL2>mSeuilVP*aSvv2));
}



void cPtOfCorrel::MakeScoreAndMasq 
     (
                   Im2D_REAL4 & aISc,
                   double  aEstBr,
                   Im2D_Bits<1> & aMasq,
                   double aSeuilVp,
                   double aSeuilEcart
     )
{
   aISc = Im2D_REAL4(mSz.x,mSz.y);
   aMasq= Im2D_Bits<1>(mSz.x,mSz.y);

   mISc = new TIm2D<REAL4,REAL8>(aISc);
   mMasq = new TIm2DBits<1>(aMasq);
   mEstBr = aEstBr;
   mSeuilVP = aSeuilVp;
   mSeuilEcart = aSeuilEcart;

   Pt2di aSzC(mVCor,mVCor);
   cTplOpbBufImage<cPtOfCorrel>
                  aOpBuf
                  (
                       *this,
                       Elise2Std(Pt2di(0,0)),
                       Elise2Std(mSz),
                       Elise2Std(-aSzC),
                       Elise2Std(aSzC)
                  );
   aOpBuf.DoIt();

   delete mISc;
   delete mMasq;
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
