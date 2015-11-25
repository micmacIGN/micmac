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


/*
class cSolTmpME
{
    public :
        cSolTmpME(double aCost,const double * aSol) :
            mCost (aCost),
            mMat (ME_Lign2Mat(aSol)),
            mRot (ElRotation3D::Id)
        {
        }

        double mCost;
        ElMatrix<double> mMat;
        ElRotation3D     mRot;
};
class cCmpcSolTmpME
{
   public :
       bool operator () (const cSolTmpME & aS1, const cSolTmpME & aS2)
       {
           return aS1.mCost > aS2.mCost;
       }
};

cCmpcSolTmpME TheCmpSolTmpME;
*/

/*
typedef cTplPrioByOther<ElRotation3D,double> tRotPrio;
typedef cCmpSupPBO<ElRotation3D,double>      tCmpRotPrio;
*/

tCmpRotPrio  TheCmpROT;



/*
 Time MicroSec 516.365

    aRot  = NEW_Mat Ess2Rot(aMat,mPack30);  => 20.5  MicroSec
       svdcmp_diag(aMEss,aSvd1,aDiag,aSvd2,true); => 1.5 MicroSec
       NEW_ SignInters(aPack,aSol,aNb1,aNb2);      =>  11 MicroSec

   ProjCostMEP(mPack150,aRot,0.005)  => 20.5 micro sec

    Pour le calcul de Mat ess => 9 Micro second

Avec 20 Iterations : 460 Micro sec sur la min


9 sec pour le calucl MatEss

*/

// void C



double  NEW_SignInters(const ElPackHomologue & aPack,const ElRotation3D & aR2to1,int & NbP1,int & NbP2)
{
     NbP1 = 0;
     NbP2 = 0;

     double aSomD=0;
     int    aNbD=0;
     for (ElPackHomologue::const_iterator it=aPack.begin(); it!=aPack.end() ; it++)
     {
          bool Ok;
          double aD;
          Pt3dr aQ1 = InterSeg(aR2to1,it->P1(),it->P2(),Ok,&aD);

          if (Ok)
          {
              Pt3dr aQ2 = aR2to1.ImRecAff(aQ1);
              NbP1 += (aQ1.z>0) ? 1 : - 1;
              NbP2 += (aQ2.z>0) ? 1 : - 1;
              aNbD++;
              aSomD += sqrt(aD);
          }
     }
     return aSomD / aNbD;
}


ElRotation3D  NEW_MatEss2Rot(const  ElMatrix<REAL> & aMEss,const ElPackHomologue & aPack,double * aResDisMin)
{
   static std::vector<ElMatrix<REAL> > TheVTeta;
   if (TheVTeta.empty())
   {
      for (INT kTeta = 0; kTeta<2 ; kTeta++)
      {
           REAL aTeta2 = PI * (kTeta+0.5);
           TheVTeta.push_back( ElMatrix<REAL>::Rotation(0,0,aTeta2));
      }
   }


   ElRotation3D aRes = ElRotation3D::Id;
   INT aScMin = -20 * aPack.size();
   double LBase = 1;

   ElMatrix<REAL> aSvd1(3,3),aDiag(3,3),aSvd2(3,3);
   svdcmp_diag(aMEss,aSvd1,aDiag,aSvd2,true);


   aSvd1.self_transpose();
   double  aDistMin = 1e20;

   for (INT sign = -1; sign <=1 ; sign += 2)
   {
      for (INT kTeta = 0; kTeta<2 ; kTeta++)
      {

           ElMatrix<REAL> aR1 = aSvd1;
           ElMatrix<REAL> aR2 =  TheVTeta[kTeta] * aSvd2;


           ElMatrix<REAL> aR1T = aR1.transpose();
           ElRotation3D aSol(aR1T *Pt3dr(-LBase*sign,0,0),aR1T*aR2,true);

            
           INT aNb1,aNb2;
           double aDist = NEW_SignInters(aPack,aSol,aNb1,aNb2);


           INT aScore = aNb1+aNb2;
           if (aScore > aScMin)
           {
               aRes = aSol;
               aScMin = aScore;
               aDistMin = aDist;
           }
      }
   }
   if (aResDisMin) 
      *aResDisMin = aDistMin;
   return aRes;
}


/***********************************************************************/

class cTestCost
{
   public :

      double mCost0;
      int    mRank0;
      double mCost1;
      double mRank1;
};
struct cTCCmp0
{
    bool operator() (const cTestCost & aCA,const cTestCost & aCB) {return aCA.mCost0 < aCB.mCost0;}
};
struct cTCCmp1
{
    bool operator() (const cTestCost & aCA,const cTestCost & aCB) {return aCA.mCost1 < aCB.mCost1;}
};





class cRanscMinimMatEss
{
     public :
         cRanscMinimMatEss( bool aQuick,
                           const ElPackHomologue & aPack,
                           const ElPackHomologue & aPackRed,
                           const ElPackHomologue & aPack150,
                           const ElPackHomologue & aPack30,
                           double aFoc
          );


          void OneFiltrage(int aNbOut,int aNbIter,cInterfBundle2Image * anIbi);
          void OneTest(int aK);


         bool AllmostEq(const ElRotation3D & aR1,const ElRotation3D & aR2) {return DistRot(aR1,aR2) < mSeuilDist;}

         const ElPackHomologue & mPackAll;
         const ElPackHomologue & mPack500;
         const ElPackHomologue & mPack150;
         const ElPackHomologue & mPack30;

         bool                    mQuick;
         int                     mNbTir0;
         int                     mNbPresel0;
         const ElPackHomologue & mPackTir0;

         
         std::vector<ElCplePtsHomologues>  mDal[3][3];
         double                mFoc;
         ElRotation3D          mRotMin;
         double                mCostMin;
         ElRotation3D          mRotMinAbs;
         double                mCostMinAbs;
         int                   mPerBundle;
         int                   mNbBundle;
         int                   mMinBundle;
         int                   mMaxBundle;
         int                   mNbSeuilStable;

         double                mSeuilDist;
         int                   mNbUnderSD;

         cInterfBundle2Image*  mLinIBI;
         cInterfBundle2Image*  mIBI30;
         cInterfBundle2Image*  mIBI150;
         ElMatrix<double>      mMatCstrEss;
         double **             mDataME;

        std::vector<ElRotation3D> mSolInterm;
        std::vector<cTestCost>    mVTC;
        cTplKPluGrand<tRotPrio,tCmpRotPrio> mKBest;
};





extern bool ShowStatMatCond;
void cRanscMinimMatEss::OneTest(int aCpt)
{
    ShowStatMatCond = false;

    /* Calcul de la matrice essentielle; */

    double aDS[9];
    int aKUnused = NRrandom3(13);
    if (aKUnused >=9) aKUnused = 4;  // On privilegie l'elim la case centrale
    int aNumLig=0;
    for (int aKCase=0 ; aKCase<9 ; aKCase++)
    {
        if (aKCase!=aKUnused)
        {
            const std::vector<ElCplePtsHomologues> & aVC = mDal[aKCase/3][aKCase%3];
            int aKP = NRrandom3((int)aVC.size());

            const Pt2dr & aP1 = aVC[aKP].P1();
            const Pt2dr & aP2 = aVC[aKP].P2();
            const double & x1 = aP1.x;
            const double & y1 = aP1.y;
            const double & x2 = aP2.x;
            const double & y2 = aP2.y;

            double * aDL = mDataME[aNumLig++];
            aDL[0] = x1 * x2;
            aDL[1] = x1 * y2;
            aDL[2] = x1 *  1;
            aDL[3] = y1 * x2;
            aDL[4] = y1 * y2;
            aDL[5] = y1 *  1;
            aDL[6] =  1 * x2;
            aDL[7] =  1 * y2;

        }
    }
    bool Ok;
    Ok = self_gaussj_svp(mMatCstrEss);
    if (! Ok)
    {
        return;
    }
    for (int aK=0 ; aK< 8 ; aK++)
    {
        double * aDL = mDataME[aK];
        aDS[aK] =  -(aDL[0]+aDL[1]+aDL[2]+aDL[3]+aDL[4]+aDL[5]+aDL[6]+aDL[7]);
    }
    aDS[8] = 1.0;


    ElMatrix<REAL> aMat = ME_Lign2Mat(aDS);

    ElRotation3D aRot  = NEW_MatEss2Rot(aMat,mPack30);
    double aCost = 0;
    aCost = ProjCostMEP(mPackTir0,aRot,0.1) * mFoc;


    tRotPrio aRP(aRot,aCost);
    mKBest.push(aRP);

    return;

/*
    for (int aK=0 ; aK<100 ; aK++)
    {
       ElRotation3D aRot  = NEW_MatEss2Rot(aMat,mPack30);
       double aCost = 0;
       aCost = ProjCostMEP(mPack30,aRot,0.1) * mFoc;
    }

*/


/*
     aRot = mIBI30->OneIterEq(aRot,aCost);
     aRot = mIBI30->OneIterEq(aRot,aCost);
*/

// for (int aK=0 ; aK<100 ; aK++) aCost = ProjCostMEP(mPack150,aRot,0.005) * mFoc;

     if (aCost< mCostMin)
     {
        mCostMin = aCost;
        mRotMin  = aRot;
     }

     if ((aCpt%mPerBundle) == (mPerBundle-1))
     {
        mNbBundle++;
        double anErStd = mIBI150->ErrInitRobuste(mRotMin);
        for (int aK=0 ; aK<10 ; aK++)
        {
              // std::cout << "     Errrrss  " << anErStd *mFoc << "\n";
              mRotMin = mIBI150->OneIterEq(mRotMin,anErStd);
        }
         // std::cout << "cRanscMinimMatEss::OneTest " << aCpt << " C0 " << mCostMin   << " End " << anErStd * mFoc << "\n";
        mSolInterm.push_back(mRotMin);
        if (anErStd < mCostMinAbs)
        {
           mCostMinAbs= anErStd;
           mRotMinAbs = mRotMin;
           mNbUnderSD=0;
           for (int aK=0 ; aK<int(mSolInterm.size()) ; aK++)
               if (AllmostEq(mRotMinAbs,mSolInterm[aK]))
                  mNbUnderSD++;
        }
        else
        {
             if ( AllmostEq(mRotMinAbs,mRotMin))
                 mNbUnderSD++;
        }
        mCostMin = 1e5;

        if (0)
        {
            cTestCost aTC;
            aTC.mCost0 = mCostMin;
            aTC.mCost1 = anErStd;
            mVTC.push_back(aTC);
        }
     }
 

     if (0)
     {
         int aPerG = mPerBundle *1;
     
         if ((aCpt%aPerG) == (aPerG-1))
         {
             double aSom = 0;
             for (int aK = 0 ; aK<int( mSolInterm.size()) ; aK++)
             {
                  double aDist = DistRot(mSolInterm[aK],mRotMinAbs);
                  aSom += (aDist<0.01);
             }



             cTCCmp0 aCmp0;
             std::sort(mVTC.begin(),mVTC.end(),aCmp0);
             for (int aK=0 ; aK<int(mVTC.size()) ; aK++)
                 mVTC[aK].mRank0 = aK;

             cTCCmp1 aCmp1;
             std::sort(mVTC.begin(),mVTC.end(),aCmp1);
             for (int aK=0 ; aK<int(mVTC.size()) ; aK++)
                 mVTC[aK].mRank1 = aK;


             double aSomDif=0;
             for (int aK=0 ; aK<int(mVTC.size()) ; aK++)
             {
                 aSomDif += ElAbs(mVTC[aK].mRank0- mVTC[aK].mRank1);
             }
             aSomDif /= ElSquare(mVTC.size());

             std::cout << "pppppppppppppppPROP " << aSom/  mSolInterm.size() << " " << aSom << " RANK " << aSomDif << " "<< mNbUnderSD<< " NB " << mSolInterm.size()<< " CMIN " << mCostMinAbs *mFoc << "\n";

         }
     }
/*
*/
}

void cRanscMinimMatEss::OneFiltrage(int aNbOut,int aNbIter,cInterfBundle2Image * anIbi)
{
   std::vector<tRotPrio>  aV0 = mKBest.Els();
   mKBest.ClearAndSetK(ElMax(1,aNbOut));


   for (int aK=0 ; aK<int(aV0.size()) ; aK++)
   {
       ElRotation3D aRot = aV0[aK].mVal;
       double aCost = anIbi->ErrInitRobuste(aRot);
       for (int aK=0 ; aK< aNbIter ; aK++)
       {
           aRot = anIbi->OneIterEq(aRot,aCost);
       }
       tRotPrio  aRP(aRot,aCost*mFoc);
       mKBest.push(aRP);
   }
}





cRanscMinimMatEss::cRanscMinimMatEss
(
      bool aQuick,
      const ElPackHomologue & aPackAll,
      const ElPackHomologue & aPack500,
      const ElPackHomologue & aPack150,
      const ElPackHomologue & aPack30,
      double aFoc
) :
    mPackAll   (aPackAll),
    mPack500   (aPack500),
    mPack150   (aPack150),
    mPack30    (aPack30 ),
    mQuick     (aQuick),
    mNbTir0    (mQuick ? 200 : 800),
    mNbPresel0 (mQuick ? 20 : 40),
    mPackTir0  (mQuick ? mPack30 : mPack150),
    mFoc     (aFoc),
    mRotMin  (ElRotation3D::Id),
    mCostMin (1e5),
    mRotMinAbs  (ElRotation3D::Id),
    mCostMinAbs (1e5),
    mPerBundle  (30),
    mNbBundle   (0),
    mMinBundle  (20),
    mMaxBundle  (100),
    mNbSeuilStable (8),
    mSeuilDist  (0.01),
    mNbUnderSD  (0),
    mLinIBI  (cInterfBundle2Image::LinearDet(mPack150,aFoc)),
    mIBI30   (cInterfBundle2Image::Bundle(mPack30,aFoc,true)),
    mIBI150     (cInterfBundle2Image::Bundle(mPack150,aFoc,true)),
    mMatCstrEss (8,8),
    mDataME     (mMatCstrEss.data()),
    mKBest      (TheCmpROT,mNbPresel0)
{
    std::vector<double> aVx;
    for (ElPackHomologue::const_iterator itP=mPackAll.begin(); itP!=mPackAll.end() ; itP++)
    {
          aVx.push_back(itP->P1().x);
    }

    double aSX0 = KthValProp(aVx,1/3.0);
    double aSX1 = KthValProp(aVx,2/3.0);

    std::vector<double> aVyX0;
    std::vector<double> aVyX1;
    std::vector<double> aVyX2;


    for (ElPackHomologue::const_iterator itP=mPackAll.begin(); itP!=mPackAll.end() ; itP++)
    {
        double anX = itP->P1().x;
        double anY = itP->P1().y;
        if (anX<aSX0) 
           aVyX0.push_back(anY);
        else if (anX<aSX1) 
           aVyX1.push_back(anY);
        else
           aVyX2.push_back(anY);
    }

    double aSy0X0 =  KthValProp(aVyX0,1/3.0);
    double aSy1X0 =  KthValProp(aVyX0,2/3.0);
    double aSy0X1 =  KthValProp(aVyX1,1/3.0);
    double aSy1X1 =  KthValProp(aVyX1,2/3.0);
    double aSy0X2 =  KthValProp(aVyX2,1/3.0);
    double aSy1X2 =  KthValProp(aVyX2,2/3.0);

    for (ElPackHomologue::const_iterator itP=mPackAll.begin(); itP!=mPackAll.end() ; itP++)
    {
        double anX = itP->P1().x;
        double anY = itP->P1().y;
        if (anX<aSX0) 
        {
           if      (anY<aSy0X0)  mDal[0][0].push_back(itP->ToCple());
           else if (anY<aSy1X0)  mDal[1][0].push_back(itP->ToCple());
           else                  mDal[2][0].push_back(itP->ToCple());
        }
        else if (anX<aSX1) 
        {
           if      (anY<aSy0X1)  mDal[0][1].push_back(itP->ToCple());
           else if (anY<aSy1X1)  mDal[1][1].push_back(itP->ToCple());
           else                  mDal[2][1].push_back(itP->ToCple());
        }
        else 
        {
           if      (anY<aSy0X2)  mDal[0][2].push_back(itP->ToCple());
           else if (anY<aSy1X2)  mDal[1][2].push_back(itP->ToCple());
           else                  mDal[2][2].push_back(itP->ToCple());
        }
    
    }


   for (int aK=0 ; aK<mNbTir0 ; aK++)
        OneTest(aK);

    OneFiltrage((int)(mKBest.Els().size() / 2),mQuick ? 2 : 3 , mQuick ? mIBI30 : mIBI150);
    OneFiltrage((int)(mKBest.Els().size() / 2),mQuick ? 2 : 3 , mQuick ? mIBI30 : mIBI150);
    OneFiltrage(1,2,mIBI150);
    mRotMinAbs = mKBest.Els()[0].mVal;

    // double aSY0X0 = KthValProp
}


ElRotation3D TestcRanscMinimMatEss
     (
          bool                    aQuick,
          const ElPackHomologue & aPackAll,
          const ElPackHomologue & aPack500,
          const ElPackHomologue & aPack150,
          const ElPackHomologue & aPack30,
          double aFoc
     )
{
    cRanscMinimMatEss aRMME(aQuick,aPackAll,aPack500,aPack150,aPack30,aFoc);

   return aRMME.mRotMinAbs;
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
