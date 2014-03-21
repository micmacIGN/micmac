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


const double TheNormMinLSQ = 0.3;
double       TheMult = 1.5;
const int    TheNbIterLine  = 10;
const int    TheNbIter2D    = 4;



//      Im2  + Kx DIm2/Dx + Ky  DIm2/Dy = K0 + K1 Im1

class cOneImLSQ;
class cOneTestLSQ;


class cOneImLSQ
{
   public :

       //  friend class cOneTestLSQ;

       cOneImLSQ(float **,const Pt2di & aSz,cOneTestLSQ &);
       Pt2dr PVois(int aKx,int aKy) const;
       bool OK(const Pt2dr &) const;
       void SetPtIm(const Pt2dr&);
       void SetPtImCur(const Pt2dr&);
       const Pt2dr& PImCur() const;

       double GetVal(const Pt2dr &) const;
       Pt3dr  GetValDer(const Pt2dr &) const;
   private :
       cOneTestLSQ &  mTest;
       Pt2di          mSz;
       float **       mData;
       Pt2dr          mPImInit;
       Pt2dr          mPImCur;
};




class cOneTestLSQ : public NROptF1vND
{
      public :
         //  friend class cOneImLSQ;

         REAL NRF1v(REAL) ;

         void SetPIm1(const Pt2dr & aPt) {mIm1->SetPtIm(aPt);}
         void SetPIm2(const Pt2dr & aPt) {mIm2->SetPtIm(aPt);}


         cOneTestLSQ
         (
             int aNbIterLine,
             float ** aData1,const Pt2di & aSzI1,
             float ** aData2,const Pt2di & aSzI2,
             cInterpolateurIm2D<float> *,
             const cCorrel2DLeastSquare &
         );
		 ~cOneTestLSQ();
         void DoEstim();
         const double & Step() const {return mStep;}
         const int    & NbW() const {return mNbW;}
         cInterpolateurIm2D<float> * Interp() const {return mInterp;}
         cOneImLSQ & Im1() {return *mIm1;}
         cOneImLSQ & Im2() {return *mIm2;}

         bool MinimByLSQandGolden(const Pt2dr& aP1,const Pt2dr&aP2);
      private :

         double Correl(const Pt2dr & aDec) const;
         double CorrelOnLine(const double & aLambda) const;
         bool   OneOptimOnLine();


         cCorrel2DLeastSquare       mCLSQ;
         cInterpolateurIm2D<float> * mInterp;
         double mStep;
         int mNbW;
         cOneImLSQ  *mIm1;
         cOneImLSQ  *mIm2;


         typedef double tSom;
         bool OK(const Pt2dr &) const;

         void ClearStat();
         void InitP1();
         bool OneLSQItere();

        
         static const int NbInc = 4;
         static const int IndK0 = 0;
         static const int IndK1 = 1;
         static const int IndKx = 2;
         static const int IndKy = 3;

         tSom  mCov[NbInc][NbInc];
         tSom  mSomI2[NbInc];
         double mSomP;
         double mSom1;
         double mSom11;

         std::vector<double> mValsIm1;
         std::vector<Pt2dr>  mDeps;
         ElMatrix<double>  mMatCov;
         ElMatrix<double>  mMatI2;
         ElMatrix<double>  mVecP;
         ElMatrix<double>  mValP;
         ElMatrix<double>  mSolLSQ;
         Pt2dr             mDepLSQ;
         double            mCorel0LSQ;

};

//    ========================= cOneImLSQ:: ======================


bool cOneImLSQ::OK(const Pt2dr  & aP) const
{
    double  aSzW = mTest.Interp()->SzKernel() + mTest.NbW()*mTest.Step() +2;

    Pt2dr aQ = aP+mPImCur;

    return       (aQ.x > aSzW)
            &&   (aQ.y > aSzW)
            &&   (aQ.x < mSz.x - aSzW)
            &&   (aQ.y < mSz.y - aSzW);
}

Pt2dr cOneImLSQ::PVois(int aKx,int aKy) const
{
   return mPImCur + Pt2dr(aKx*mTest.Step(),aKy*mTest.Step());
}

cOneImLSQ::cOneImLSQ(float ** aData,const Pt2di & aSz,cOneTestLSQ & aTest) :
   mTest (aTest),
   mSz   (aSz),
   mData (aData)
{
}

void cOneImLSQ::SetPtIm(const Pt2dr& aPtIm)
{
   mPImCur = aPtIm;
   mPImInit = aPtIm;
}
void cOneImLSQ::SetPtImCur(const Pt2dr& aPtIm)
{
   mPImCur = aPtIm;
}

const Pt2dr& cOneImLSQ::PImCur() const
{
   return mPImCur;
}


double cOneImLSQ::GetVal(const Pt2dr  & aP) const
{
   return mTest.Interp()->GetVal(mData,aP);
}

Pt3dr cOneImLSQ::GetValDer(const Pt2dr  & aP) const
{
   return mTest.Interp()->GetValDer(mData,aP);
}


//    ========================= cOneTestLSQ:: ======================

cOneTestLSQ::cOneTestLSQ
(
   int aNbIterLine,
   float ** aData1,const Pt2di & aSzI1,
   float ** aData2,const Pt2di & aSzI2,
   cInterpolateurIm2D<float> * anInterp,
   const cCorrel2DLeastSquare &  aCLSQ
) :
   NROptF1vND (aNbIterLine),
   mCLSQ   (aCLSQ),
   mInterp (anInterp),
   mStep   (aCLSQ.Step().Val()),
   mNbW    (round_ni(aCLSQ.SzW()/mStep)),
   mIm1    (NULL),
   mIm2    (NULL),
   mMatCov (NbInc,NbInc),
   mMatI2  (1,NbInc),
   mVecP   (NbInc,NbInc),
   mValP   (NbInc,NbInc),
   mSolLSQ (1,NbInc)
{
	// NO_WARN
	mIm1 = new cOneImLSQ( aData1,aSzI1, *this );
	mIm2 = new cOneImLSQ( aData2,aSzI2, *this );
}

cOneTestLSQ::~cOneTestLSQ(){
	if ( mIm1!=NULL ) delete mIm1;
	if ( mIm2!=NULL ) delete mIm2;
}

bool cOneTestLSQ::OK(const Pt2dr & aP) const
{
    return mIm1->OK(Pt2dr(0,0)) && mIm2->OK(aP) ;
}



void cOneTestLSQ::ClearStat()
{
   for(int aK1=0; aK1<NbInc; aK1++)
   {
       mSomI2[aK1] = 0;
       for(int aK2=0; aK2<NbInc; aK2++)
       {
           mCov[aK1][aK2] = 0;
       }
   }
}


void cOneTestLSQ::InitP1()
{
   mValsIm1.clear();
   mDeps.clear();

   mSomP=0;
   mSom1=0;
   mSom11=0;
   for(int aKx=-mNbW; aKx<=mNbW; aKx++)
   {
       for(int aKy=-mNbW; aKy<=mNbW; aKy++)
       {
             
            Pt2dr aP1 = mIm1->PVois(aKx,aKy);
            double aV1 = mIm1->GetVal(aP1);
            mSom1 += aV1;
            mSom11 += ElSquare(aV1);
            mSomP++;
            mValsIm1.push_back(aV1);
       }
   }
   mSom1 /= mSomP;
   mSom11 /= mSomP;
   mSom11 -= ElSquare(mSom1);
}

double  cOneTestLSQ::NRF1v(REAL aV)
{
   return -CorrelOnLine(aV);
}


double cOneTestLSQ::CorrelOnLine(const double & aLambda) const
{
    return Correl(mDepLSQ*aLambda);
}

double cOneTestLSQ::Correl(const Pt2dr & aDec) const
{
   if (! OK(aDec)) return -1;

   double aSom2= 0;
   double aSom22= 0;
   double aSom12= 0;
   int aCpt=0;

   for(int aKx=-mNbW; aKx<=mNbW; aKx++)
   {
       for(int aKy=-mNbW; aKy<=mNbW; aKy++)
       {
          Pt2dr aP2 = aDec+mIm2->PVois(aKx,aKy);

          double aV2 =  mIm2->GetVal(aP2);
          double aV1 = mValsIm1[aCpt];
          aSom2 += aV2;
          aSom12 += aV1*aV2;
          aSom22 += ElSquare(aV2);
         
          aCpt++;
       }
    }

    aSom2 /= mSomP;
    aSom12 /= mSomP;
    aSom22 /= mSomP;

    aSom12 -= mSom1 * aSom2;
    aSom22 -= ElSquare(aSom2);


    return aSom12 * sqrt(ElMax(1e-5,aSom22*mSom11));
}
 
bool  cOneTestLSQ::OneOptimOnLine()
{
   if (! OneLSQItere())
      return false;

   double aL0 = 0;
   double aC0 =  mCorel0LSQ;

   double aL1 = 1;
   double aC1 =  CorrelOnLine(aL1);
   if (aC1 == -1) 
      return false;

   if (aC1<aC0) 
      return false;

   double aL2 = ElMax(2.0,TheNormMinLSQ/euclid(mDepLSQ));
   double aC2 =  CorrelOnLine(aL2);

   if (aC2 == -1) 
      return false;

   while (aC2 > aC1)
   {
       aL0 = aL1;
       aC0 = aC1;

       aL1 = aL2;
       aC1 = aC2;
 
       aL2 *= TheMult;
       aC2 =  CorrelOnLine(aL2);
       if (aC2 == -1) 
           return false;
   }

   double aLMin;
   golden(aL0,aL1,aL2,1e-3,&aLMin);

   SetPIm2(mIm2->PVois(0,0) +mDepLSQ *aLMin);
   return true;

}

bool cOneTestLSQ::MinimByLSQandGolden(const Pt2dr& aP1,const Pt2dr& aP2)
{
   SetPIm1(aP1);
   SetPIm2(aP2);
   InitP1();


   for (int aK=0 ; aK<TheNbIter2D ; aK++)
   {
     
      if (! OneLSQItere())
      {
         return (aK>0);
      }
      if (!  OneOptimOnLine())
      {
         return (aK>0);
      }
   }
   return true;
}


/*
*/


bool  cOneTestLSQ::OneLSQItere()
{
   if (!OK(Pt2dr(0,0))) return false;

   ClearStat();
   int aCpt=0;

   RMat_Inertie aMat;
   double aDif = 0;

   for(int aKx=-mNbW; aKx<=mNbW; aKx++)
   {
       for(int aKy=-mNbW; aKy<=mNbW; aKy++)
       {
          Pt2dr aP2 = mIm2->PVois(aKx,aKy);

          Pt3dr aGV2 =  mIm2->GetValDer(aP2);


           double aV1 = mValsIm1[aCpt];
          // double aV1 = mIm1.GetVal(mIm1.PVois(aKx,aKy));
          double aV2 = aGV2.z;
          double aGx = aGV2.x;
          double aGy = aGV2.y;

          aDif += ElSquare(aV1-aV2);
          aMat.add_pt_en_place(aV1,aV2);

          // Pour verifier la justesse des moindres carres
          if (0)
          {
              aV2 = 50 + 2 * aV1  - 0.5* aGx -0.25 * aGy;
          }

          mCov[IndK0][IndK0] += 1;
          mCov[IndK0][IndK1] += aV1;
          mCov[IndK0][IndKx] -= aGx;
          mCov[IndK0][IndKy] -= aGy;

          mCov[IndK1][IndK1] += aV1*aV1;
          mCov[IndK1][IndKx] -= aV1*aGx;
          mCov[IndK1][IndKy] -= aV1*aGy;

          mCov[IndKx][IndKx] += aGx*aGx;
          mCov[IndKx][IndKy] += aGx*aGy;

          mCov[IndKy][IndKy] += aGy*aGy;

          mSomI2[IndK0] += aV2;
          mSomI2[IndK1] += aV1 * aV2;
          mSomI2[IndKx] -= aV2 * aGx;
          mSomI2[IndKy] -= aV2 * aGy;

          aCpt++;
       }
   }

   for(int aK1=0; aK1<NbInc; aK1++)
   {
       mMatI2(0,aK1) = mSomI2[aK1];
       for(int aK2=0; aK2<=aK1; aK2++)
       {
           mMatCov(aK1,aK2) =  mMatCov(aK2,aK1) = mCov[aK2][aK1];
       }
   }


   jacobi_diag(mMatCov,mValP,mVecP);

   double aVPMin = 1e10;
   for(int aK1=0; aK1<NbInc; aK1++)
      aVPMin = ElMin(aVPMin,mValP(aK1,aK1));

   if (aVPMin<1e-8) return false;


   mSolLSQ = gaussj(mMatCov) * mMatI2;
   mCorel0LSQ = aMat.correlation();


   mDepLSQ = Pt2dr(mSolLSQ(0,2), mSolLSQ(0,3));



   mDeps.push_back(mDepLSQ);

   return true;
}


void  cOneTestLSQ::DoEstim()
{
   if (! OK(Pt2dr(0,0)))
      return;

   InitP1();

   for (int aK=0 ; aK<90 ; aK++)
   {
       if (!  OneLSQItere())
       {
           std::cout << " Failled !!!\n";
           return;
       }
       mIm2->SetPtImCur(mIm2->PImCur()+mDepLSQ);
   }

   for (int aK=0 ; aK < int(mDeps.size()) ; aK++)
       std::cout << mDeps[aK] << " ";

    std::cout << "\n";
    getchar();
}


/*
         static const int NbInc = 4;
         static const int IndK0 = 0;
         static const int IndK1 = 1;
         static const int IndKx = 2;
         static const int IndKy = 2;
*/


// ============================ cAppliMICMAC:: ========================


void cAppliMICMAC::DoCorrelLeastQuare(const Box2di &  aBoxOut,const Box2di & aBoxIn,const cCorrel2DLeastSquare & aClsq)
{
    int aPer = aClsq.PeriodEch();


    cOneTestLSQ  aTest
                 (
                     TheNbIterLine,
                     PDV1()->LoadedIm().DataFloatIm()[0],PDV1()->LoadedIm().SzIm(),
                     PDV2()->LoadedIm().DataFloatIm()[0],PDV2()->LoadedIm().SzIm(),
                     CurEtape()->InterpFloat(),
                     aClsq
                 );


    Pt2di aP0Red = round_up(Pt2dr(aBoxOut._p0) / double(aPer));
    Pt2di aP1Red = round_up(Pt2dr(aBoxOut._p1) / double(aPer));
    Pt2di aSzRed = aP1Red - aP0Red;

    Im2D_REAL8 aImDepX(aSzRed.x,aSzRed.y);
    Im2D_REAL8 aImDepY(aSzRed.x,aSzRed.y);

    Pt2di aPRed;

    const cOneNappePx & aPx1 = mLTer->KthNap(0);
    const cOneNappePx & aPx2 = mLTer->KthNap(1);

    for (aPRed.x=aP0Red.x  ; aPRed.x<aP1Red.x ; aPRed.x++)
    {
        for (aPRed.y=aP0Red.y ; aPRed.y<aP1Red.y ; aPRed.y++)
        {
             // std::cout <<"REST " << aP1Red - aPRed << "\n";

              Pt2di aP = aPRed  * aPer;
              Pt2di aPLoc = aP-aBoxIn._p0;
              double aPx[2];
              aPx[0] = aPx1.mTPxInit.get(aPLoc);
              aPx[1] = aPx2.mTPxInit.get(aPLoc);

              mCurEtape->GeomTer().PxDisc2PxReel(aPx,aPx);

              //aTest.SetPIm1(PDV1()->Geom().CurObj2Im(Pt2dr(aP),aPx));
              //aTest.SetPIm2(PDV2()->Geom().CurObj2Im(Pt2dr(aP),aPx));

              aTest.MinimByLSQandGolden
              (
                   PDV1()->Geom().CurObj2Im(Pt2dr(aP),aPx),
                   PDV2()->Geom().CurObj2Im(Pt2dr(aP),aPx)
              );

              double aPx0[2]={0,0};
              Pt2dr aP20 = PDV2()->Geom().CurObj2Im(Pt2dr(aP),aPx0);
              Pt2dr aP2 = aTest.Im2().PImCur();

              aImDepX.SetR(aPRed-aP0Red,aP2.x-aP20.x);
              aImDepY.SetR(aPRed-aP0Red,aP2.y-aP20.y);
         }
    }

    Tiff_Im   aFileRX = mCurEtape->KPx(0).FileIm();
    Tiff_Im   aFileRY = mCurEtape->KPx(1).FileIm();


    ELISE_COPY
    (
         rectangle(aP0Red,aP1Red),
         trans(aImDepX.in(),-aP0Red),
         aFileRX.out()
    );
    ELISE_COPY
    (
         rectangle(aP0Red,aP1Red),
         trans(aImDepY.in(),-aP0Red),
         aFileRY.out()
    );
/*
*/

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
