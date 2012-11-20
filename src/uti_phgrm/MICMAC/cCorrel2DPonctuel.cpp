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

namespace NS_ParamMICMAC
{




//     K0 +  K1 Im2  + Kx DIm2/Dx + Ky  DIm2/Dy = Im1

class cOneImLSQ;
class cOneTestLSQ;


class cOneImLSQ
{
   public :

       //  friend class cOneTestLSQ;

       cOneImLSQ(float **,const Pt2di & aSz,cOneTestLSQ &);
       Pt2dr PVois(int aKx,int aKy) const;
       bool OK(const cOneTestLSQ &) const;
       void SetPtIm(const Pt2dr&);
       void SetPtImCur(const Pt2dr&);
       const Pt2dr& PImCur() const;

       double GetVal(const Pt2dr &);
       Pt3dr  GetValDer(const Pt2dr &);
   private :
       cOneTestLSQ &  mTest;
       Pt2di          mSz;
       float **       mData;
       Pt2dr          mPImInit;
       Pt2dr          mPImCur;
};




class cOneTestLSQ
{
      public :
         //  friend class cOneImLSQ;

         void SetPIm1(const Pt2dr & aPt) {mIm1.SetPtIm(aPt);}
         void SetPIm2(const Pt2dr & aPt) {mIm2.SetPtIm(aPt);}


         cOneTestLSQ
         (
             float ** aData1,const Pt2di & aSzI1,
             float ** aData2,const Pt2di & aSzI2,
             cInterpolateurIm2D<float> *,
             int aSzW, double aStep
         );
         void DoEstim();
         const double & Step() const {return mStep;}
         const int    & NbW() const {return mNbW;}
         cInterpolateurIm2D<float> * Interp() const {return mInterp;}
         cOneImLSQ & Im1() {return mIm1;}
         cOneImLSQ & Im2() {return mIm2;}

      private :

         cInterpolateurIm2D<float> * mInterp;
         double mStep;
         int mNbW;
         cOneImLSQ  mIm1;
         cOneImLSQ  mIm2;


         typedef double tSom;
         bool OK() const;

         void ClearStat();
         void Init();
         bool OneItere();

        
         static const int NbInc = 4;
         static const int IndK0 = 0;
         static const int IndK1 = 1;
         static const int IndKx = 2;
         static const int IndKy = 3;

         tSom  mCov[NbInc][NbInc];
         tSom  mSomI2[NbInc];

         std::vector<double> mValsIm1;
         std::vector<Pt2dr>  mDeps;
         ElMatrix<double>  mMatCov;
         ElMatrix<double>  mMatI2;
         ElMatrix<double>  mVecP;
         ElMatrix<double>  mValP;
};

//    ========================= cOneImLSQ:: ======================


bool cOneImLSQ::OK(const cOneTestLSQ & aT) const
{
    double  aSzW = aT.Interp()->SzKernel() + aT.NbW()*aT.Step() +2;

    return       (mPImCur.x > aSzW)
            &&   (mPImCur.y > aSzW)
            &&   (mPImCur.x < mSz.x - aSzW)
            &&   (mPImCur.y < mSz.y - aSzW);
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


double cOneImLSQ::GetVal(const Pt2dr  & aP)
{
   return mTest.Interp()->GetVal(mData,aP);
}

Pt3dr cOneImLSQ::GetValDer(const Pt2dr  & aP)
{
   return mTest.Interp()->GetValDer(mData,aP);
}


//    ========================= cOneTestLSQ:: ======================

cOneTestLSQ:: cOneTestLSQ
(
   float ** aData1,const Pt2di & aSzI1,
   float ** aData2,const Pt2di & aSzI2,
   cInterpolateurIm2D<float> * anInterp,
   int aSzW, double aStep
) :
   mInterp (anInterp),
   mStep   (aStep),
   mIm1    (aData1,aSzI1,*this),
   mIm2    (aData2,aSzI2,*this),
   mNbW    (round_ni(aSzW/aStep)),
   mMatCov (NbInc,NbInc),
   mMatI2  (1,NbInc),
   mVecP   (NbInc,NbInc),
   mValP   (NbInc,NbInc)
{
}


bool cOneTestLSQ::OK() const
{
    return mIm1.OK(*this) && mIm2.OK(*this) ;
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


void cOneTestLSQ::Init()
{
   mValsIm1.clear();
   mDeps.clear();
   for(int aKx=-mNbW; aKx<=mNbW; aKx++)
   {
       for(int aKy=-mNbW; aKy<=mNbW; aKy++)
       {
             
            Pt2dr aP1 = mIm1.PVois(aKx,aKy);
            double aV1 = mIm1.GetVal(aP1);
            mValsIm1.push_back(aV1);
       }
   }
}

bool  cOneTestLSQ::OneItere()
{
   if (!OK()) return false;

   ClearStat();
   int aCpt=0;

   RMat_Inertie aMat;
   double aDif = 0;

   for(int aKx=-mNbW; aKx<=mNbW; aKx++)
   {
       for(int aKy=-mNbW; aKy<=mNbW; aKy++)
       {
          Pt2dr aP2 = mIm2.PVois(aKx,aKy);

          Pt3dr aGV2 =  mIm2.GetValDer(aP2);


          // double aV1 = mValsIm1[aCpt];
          double aV1 = mIm1.GetVal(mIm1.PVois(aKx,aKy));
          double aV2 = aGV2.z;
          double aGx = aGV2.x;
          double aGy = aGV2.y;

          aDif += ElSquare(aV1-aV2);
          aMat.add_pt_en_place(aV1,aV2);

if (0)
{
/*
  aV1 = NRrandom3();
  aGx = NRrandom3();
  aGy = NRrandom3();
*/

  aV2 = 50 + 2 * aV1  - 0.5* aGx -0.25 * aGy;
}

//     Im2 -K0 -K1 Im1 + Kx DIm2/Dx + Ky  DIm2/Dy =  0

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
    // std::cout << aVPMin << " vp :: "<< mValP(0,0) << " " << mValP(1,1) << " " << mValP(2,2)<< " " << mValP(3,3) << "\n";
   if (aVPMin<1e-8) return false;


   ElMatrix<double> aSol = gaussj(mMatCov) * mMatI2;

   // std::cout << "Sol "<< aSol(0,0) << " " << aSol(0,1) << "  x " << aSol(0,2)<< " y " << aSol(0,3) << "\n";
   // std::cout << mIm2.PImCur()  << mIm2.PVois(0,0) << " dif " << aDif<< " cor " << aMat.correlation() << "\n";


   Pt2dr aDP2( aSol(0,2), aSol(0,3));

   mDeps.push_back(aDP2);
   mIm2.SetPtImCur(mIm2.PImCur()+aDP2);

   return true;
}


void  cOneTestLSQ::DoEstim()
{
   if (! OK())
      return;

   Init();

   for (int aK=0 ; aK<90 ; aK++)
   {
       if (!  OneItere())
       {
           std::cout << " Failled !!!\n";
           return;
       }
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

void cAppliMICMAC::DoCorrelLeastQuare(const Box2di &  aBoxOut,const Box2di & aBoxIn,const cCorrel2DLeastSquare &)
{
while(1)
{
    double aSzW = 32.0;
    double aStep = 1.0;
    int aPer = 10;



    cOneTestLSQ  aTest
                 (
                     PDV1()->LoadedIm().DataFloatIm(),PDV1()->LoadedIm().SzIm(),
                     PDV2()->LoadedIm().DataFloatIm(),PDV2()->LoadedIm().SzIm(),
                     CurEtape()->InterpFloat(),
                     aSzW,aStep
                 );




/*
    aTest.mInterp = CurEtape()->InterpFloat();
    aTest.mNbW = round_ni(aSzW/aStep);
    aTest.mStep = aStep;

    aTest.mIm1.mData = PDV1()->LoadedIm().DataFloatIm();
    aTest.mIm1.mSz = PDV1()->LoadedIm().SzIm();
    aTest.mIm2.mData = PDV2()->LoadedIm().DataFloatIm();
    aTest.mIm2.mSz = PDV2()->LoadedIm().SzIm();
*/

    std::cout << "DoCorrelLeastQuare " << aBoxOut._p0 << aBoxOut._p1 << "\n";
    std::cout << "DoCorrelLeastQuare " << aBoxIn._p0 << aBoxIn._p1 << "\n";
    std::cout << "ssssssssssssss " << aBoxOut.sz() << aBoxIn.sz()  << "\n";



    // ===================

    Pt2di aP0Red = round_up(Pt2dr(aBoxOut._p0) / double(aPer));
    Pt2di aP1Red = round_up(Pt2dr(aBoxOut._p1) / double(aPer));
    Pt2di aSzRed = aP1Red - aP0Red;


    Pt2di aPRed;

    const cOneNappePx & aPx1 = mLTer->KthNap(0);
    const cOneNappePx & aPx2 = mLTer->KthNap(1);

/*
    for (aPRed.x=aP0Red.x  ; aPRed.x<aP1Red.x ; aPRed.x++)
    {
        for (aPRed.y=aP0Red.y ; aPRed.y<aP1Red.y ; aPRed.y++)
        {
*/
              Pt2di aP = aPRed  * aPer;

std::cout << "Enter value for P\n";
std::cin >> aP.x >> aP.y ;

              Pt2di aPLoc = aP-aBoxIn._p0;

              double aPx[2];
std::cout << "in::PL " << aPLoc  << " pp " << aP <<"\n";
              aPx[0] = aPx1.mTPxInit.get(aPLoc);
              aPx[1] = aPx2.mTPxInit.get(aPLoc);
std::cout << "out::PL " << aPLoc <<"\n";

              mCurEtape->GeomTer().PxDisc2PxReel(aPx,aPx);

              aTest.SetPIm1(PDV1()->Geom().CurObj2Im(Pt2dr(aP),aPx));
              aTest.SetPIm2(PDV2()->Geom().CurObj2Im(Pt2dr(aP),aPx));

 Pt2dr aQ2 = aTest.Im2().PImCur();


              std::cout << "  in ppxxx " << aPx[0] << " " << aPx[1] << aP << aTest.Im1().PImCur() << aTest.Im2().PImCur() << "\n";
              aTest.DoEstim();
              std::cout << "  out ppxxx " << aPx[0] << " " << aPx[1] << aP << aTest.Im1().PImCur() << aTest.Im2().PImCur() << "\n";

std::cout << "Delta p2 " << aTest.Im2().PImCur() - aQ2 << "\n";

// getchar();
              // Pt2di aP0 = aP -Pt2di(aSzW,aSzW);
              // Pt2di aP1 = aP +Pt2di(aSzW+1,aSzW+1);
             
/*
        }
    }
*/


    //cOneSolLSQ aSol(aBox,aBox.sz(),PDV1(),PDV2());


//    for (int anX==


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
