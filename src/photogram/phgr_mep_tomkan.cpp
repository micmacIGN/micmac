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

const double DefScoreTKS();
// static double DefScoreTKS = 1e20;

//================================================================================



class cBasicCamOrtho;
class cTomKanCamUnk;
class cTomKanSolver;

static Pt3dr RanP3C()  {return Pt3dr(NRrandC(),NRrandC(),NRrandC());}
static Pt2dr RanP2C()  {return Pt2dr(NRrandC(),NRrandC());}


class cBasicCamOrtho
{
   public :
      cBasicCamOrtho(const Pt3dr & aI,const Pt3dr & aJ,const double & aScale,const Pt2dr & aPP);
      static cBasicCamOrtho  RandomCam();
      Pt2dr Proj(const Pt3dr &) const;
   private :

        Pt3dr mI;
        Pt3dr mJ;
        Pt3dr mK;
        Pt2dr mPP;
};


class cTomKanCamUnk
{
    public :
        friend class cTomKanSolver;
        cTomKanCamUnk(const std::vector<Pt2dr> & aVpt,double aDist,CamStenopeIdeale * aRef); // Dist >0 -> Pts Photogram, utilise pour normaliser
        const cBasicCamOrtho & SolOrth() const;
        void SetCamOrtho(const Pt3dr & aI,const Pt3dr & aJ);
        const Pt2dr & VInitK(int) const;
        const Pt2dr & VCenterK(int) const;

        CamStenopeIdeale& CamI(bool Ref);
        double & ResOFPA();
        std::vector<Pt2df> & VFInit();
    private :
        std::vector<Pt2dr>   mVInit;
        std::vector<Pt2dr>   mVNorm;
        std::vector<Pt2dr>   mVCenter;
        int                  mNbPts;
        Pt2dr                mCDG;
        cBasicCamOrtho       mSolOrth;
        CamStenopeIdeale     mCamI;
        double               mResOFPA;
        std::vector<Pt2df>   mVFInit;
        CamStenopeIdeale *   mCamRef;
};


class cTomKanSolver
{
    public :
        cTomKanSolver(const std::vector<std::vector<Pt2dr> > & aVVPt,double aDist,std::vector<CamStenopeIdeale> * aVRef);

        // double OrientCamStenope(int aNbTest,CamStenope & aCam,int aKCam,int aSign);
        void   OrientCamStenope(int aNbTest,int aKCam,int aSign);
        double   OrientAllCamStenope(int aNbTest,int aSign,std::vector<ElRotation3D> & aSol);
        double   OrientAllCamStenopeAllSigne(int aNbTest,std::vector<ElRotation3D> & aSol);
        bool VPNeg() const {return mVPNeg;}

        ElMatrix<double> OrientOrtho(int aK);
        Pt3dr GetPGround(int aIP) const;

        double TestSolLin(const std::vector<Pt3dr> & , ElMatrix<double> & aMat,Pt3dr & aTr);
        cSolBasculeRig TestSolSim(const std::vector<Pt3dr> &,int aSign, double & aPrec);

        double   SomResOFPA() const; 
        const tMultiplePF &  VPF3() const;
    private :
        void FillCoeffW(double *,int aIP1,int aIP2) const;
        Pt3dr GetPtIJ(int aIP) const;
        Pt3dr GetPtI(int aIP) const;
        Pt3dr GetPtJ(int aIP) const;

        double                     mDist;
        int                        mNbCam;
        int                        mNbPts;
        std::vector<cTomKanCamUnk> mVC;
        int                        mSzMat;
     //  mMatUV0 init non square
        ElMatrix<double>           mMatUV0;
        ElMatrix<double>           mMatIJ0;
        ElMatrix<double>           mDiag0;
        ElMatrix<double>           mMatP0;
     // Square Matrix for svd lib
        ElMatrix<double>           mMatUVSq;
        ElMatrix<double>           mMatIJSq;
        ElMatrix<double>           mDiagSq;
        ElMatrix<double>           mMatPSq;

     // 3x3 matrixes
        ElMatrix<double>           mW;
        ElMatrix<double>           mRotW;
        ElMatrix<double>           mDiagW;
        ElMatrix<double>           mQ;

      // 
        std::vector<Pt3dr>        mVTer; // Vector of ground coordinates (=> mMatP)
        bool                      mVPNeg;
        double                    mSomResOFPA; // Residu de l'orient / appuis

        bool                          mHasRef;
        std::vector<CamStenopeIdeale>* mVRef;
        tMultiplePF                    mVPF3;
};

class cTKS_OptGlob
{
    public :
          cTKS_OptGlob(const tMultiplePF & aVPF3);
          void  OneTest(int aNb);
          void  Show(double aTime);
          double ScMin() const;
          const std::vector<ElRotation3D> & BestR() const;
    private :
          tMultiplePF               mVPF3;
          int                       mNbPts;
          int                       mNbCam;
          double                    mBestScore;
          std::vector<ElRotation3D> mBestR;
          double                    mScMin;
          double                    mScMax;
          std::vector<double>       mVSc;
          int                       mCptTot;
          int                       mCptNeg;
};

/*****************************************************/
/*                                                   */
/*              cTKS_OptGlob                         */
/*                                                   */
/*****************************************************/



cTKS_OptGlob::cTKS_OptGlob(const tMultiplePF & aVPF3) :
   mVPF3       (aVPF3),
   mNbPts      (mVPF3[0]->size()),
   mNbCam      (mVPF3.size()),
   mBestScore  (1e20),
   mBestR      (),
   mScMin      (1e20),
   mScMax      (0),
   mCptTot     (0),
   mCptNeg     (0)
{
}

void cTKS_OptGlob::Show(double aTime)
{
   int aNb = mVSc.size();
   std::cout << "SCORE"
             << " ; Min " << mScMin 
             << " ; 2th " << KthVal(mVSc,2) 
             << " ; Med " << KthVal(mVSc,aNb/2) 
             << " ; Max " << mScMax 
             << "  ### T=" << aTime 
             << "  Neg=" << (100.0*mCptNeg) / mCptTot
             << "\n";
}



void  cTKS_OptGlob::OneTest(int aNbLoc)
{

   int aDist = 10000;

   std::vector<std::vector<Pt2dr> >  aVVPt(mNbCam);
   aNbLoc = ElMax(6,aNbLoc);
   cRandNParmiQ aSel(aNbLoc,mNbPts);

   for (int aKp=0 ; aKp<mNbPts ; aKp++)
   {
       if (aSel.GetNext())
       {
          for (int aKc=0 ; aKc<mNbCam ; aKc++)
          {
              aVVPt[aKc].push_back(ToPt2dr( (*(mVPF3[aKc]))[aKp]));
          } 
       }
   }
   aNbLoc  = aVVPt[0].size();

   cTomKanSolver aTKS(aVVPt,aDist,0);

   std::vector<ElRotation3D> aVR;
   // double aScore0 = 
   double aS0 = aTKS.OrientAllCamStenopeAllSigne(ElMin(30,(aNbLoc*(aNbLoc+1)/2)),aVR);
   double aScoreQ =  (aS0!= DefScoreTKS()) ? QualInterSeg(aVR,mVPF3) :  DefScoreTKS();

   // std::cout << "SCORE = " << aScore0  << " SCROT=" <<  aScoreQ << "\n";
   mScMax = ElMax(aScoreQ,mScMax);
   if (aScoreQ < mScMin)
   {
      mScMin = aScoreQ;
      mBestR = aVR;
   }
   mVSc.push_back(aScoreQ);

   mCptTot++;
   if (aTKS.VPNeg())
      mCptNeg++;
}

double cTKS_OptGlob::ScMin() const
{
   return mScMin;
}

const std::vector<ElRotation3D> & cTKS_OptGlob::BestR() const
{
   return mBestR;
}
/*****************************************************/
/*                                                   */
/*              cBasicCamOrtho                       */
/*                                                   */
/*****************************************************/


cBasicCamOrtho::cBasicCamOrtho(const Pt3dr & aI,const Pt3dr & aJ,const double & aScale,const Pt2dr & aPP) :
    mI      (aI),
    mJ      (aJ),
    mK      (SchmitComplMakeOrthon(mI,mJ)),
    mPP     (aPP)
{
   mI = mI * aScale;
   mJ = mJ * aScale;

}

cBasicCamOrtho cBasicCamOrtho::RandomCam()
{
    return cBasicCamOrtho ( RanP3C(), RanP3C(), 2+NRrandC(), RanP2C());
}

Pt2dr cBasicCamOrtho::Proj(const Pt3dr & aP) const
{
    return mPP +  Pt2dr(scal(mI,aP),scal(mJ,aP));
}

/*****************************************************/
/*                                                   */
/*              cTomKanCamUnk                        */
/*                                                   */
/*****************************************************/


// Si les points sont des "points photogrammetriques" (i.e. (x,y,1) est un dir de rayon)

static std::vector<Pt2dr> ApproxFuseau(const std::vector<Pt2dr> & aV0,double aDist)
{
    int aNbPts = aV0.size();
    Pt3dr aDirK(0,0,0);
    std::vector<Pt3dr> aVDir;
    for (int aKP=0 ; aKP<aNbPts ; aKP++)
    { 
        aVDir.push_back(vunit(Pt3dr(aV0[aKP].x,aV0[aKP].y,1.0)));
        aDirK = aDirK + aVDir.back();
    }
    aDirK = aDirK / aNbPts;
        
    Pt3dr aDirI,aDirJ;
    MakeRONWith1Vect(aDirK,aDirI,aDirJ);
    ElMatrix<double> aMat = MatFromCol(aDirI,aDirJ,aDirK).transpose();

    std::vector<Pt2dr> aVRes;
    for (int aKP=0 ; aKP<aNbPts ; aKP++)
    { 
        Pt3dr aDir = aMat * aVDir[aKP];
        Pt2dr aQ(aDir.x/aDir.z,aDir.y/aDir.z);
        aVRes.push_back(aQ*aDist);

    }
    return aVRes;
}



static std::vector<double> ThePAF;

cTomKanCamUnk::cTomKanCamUnk(const std::vector<Pt2dr> & aVPt,double aDist,CamStenopeIdeale * aRef) :
   mVInit   (aVPt),
   mVNorm   ((aDist>0) ? ApproxFuseau(mVInit,aDist) : mVInit),
   mVCenter (mVNorm),
   mNbPts   (aVPt.size()),
   mCDG     (0,0),
   mSolOrth (Pt3dr(1,0,0),Pt3dr(0,1,0),1.0,Pt2dr(0,0)),
   mCamI    (true,1.0,Pt2dr(0,0),ThePAF),
   mCamRef  (aRef)
{
   ConvertContainer(mVFInit,mVInit);
   for (int aK=0 ; aK<mNbPts ; aK++)
      mCDG = mCDG + mVCenter[aK];

   mCDG = mCDG / mNbPts;
   for (int aK=0 ; aK<mNbPts ; aK++)
       mVCenter[aK] = mVCenter[aK] - mCDG;

}



void cTomKanCamUnk::SetCamOrtho(const Pt3dr & aI,const Pt3dr & aJ)
{
   double aS = (euclid(aI) + euclid(aJ)) / 2.0;

   mSolOrth  = cBasicCamOrtho(vunit(aI),vunit(aJ),aS,mCDG);
}

const Pt2dr & cTomKanCamUnk::VInitK(int aKP) const   { return mVInit.at(aKP); }
const Pt2dr & cTomKanCamUnk::VCenterK(int aKP) const { return mVCenter.at(aKP);}



const cBasicCamOrtho & cTomKanCamUnk::SolOrth() const
{
   return mSolOrth;
}
CamStenopeIdeale& cTomKanCamUnk::CamI(bool Ref) 
{
   if (! Ref)
      return mCamI;
   ELISE_ASSERT(mCamRef!=0,"cTomKanCamUnk::CamI");
   return * mCamRef;
}
double & cTomKanCamUnk::ResOFPA()       {return mResOFPA;}
std::vector<Pt2df> & cTomKanCamUnk::VFInit() {return mVFInit;}

/*****************************************************/
/*                                                   */
/*              cTomKanSolver                        */
/*                                                   */
/*****************************************************/

class cCmpAbsMatDiag
{
    public :
        cCmpAbsMatDiag(ElMatrix<double>  & aMat) :
            mMat(aMat)
        {
        }
        bool operator () (const int & aI1,const int & aI2)
        {
           return ElAbs(mMat(aI1,0)) >  ElAbs(mMat(aI2,0)) ;
        }
    private :
        ElMatrix<double>  & mMat;

};

// Fill the coeff 
//                 C1  C4  C5   X2               C1*X2  + C4*Y2 + C5*Z2
//                 C4  C2  C6   Y2               C4*X2  + C2*Y2 + C6*Z2
//    X1 Y1 Z1     C5  C6  C3   Z2  =  X1 Y1 Z1  C5*X2  + C6*Y2 + C3*Z2


Pt3dr cTomKanSolver::GetPtIJ(int aIP) const
{
    return Pt3dr(mMatIJ0(0,aIP),mMatIJ0(1,aIP),mMatIJ0(2,aIP));
}

Pt3dr cTomKanSolver::GetPtI(int aIP) const { return GetPtIJ(aIP); }
Pt3dr cTomKanSolver::GetPtJ(int aIP) const { return GetPtIJ(aIP+mNbCam); }

Pt3dr cTomKanSolver::GetPGround(int aIP) const
{
    return Pt3dr(mMatP0(aIP,0),mMatP0(aIP,1),mMatP0(aIP,2));
}

void cTomKanSolver::FillCoeffW(double * aCoef,int aIP1,int aIP2) const
{
   Pt3dr aP1 = GetPtIJ(aIP1);
   Pt3dr aP2 = GetPtIJ(aIP2);

   aCoef[0] = aP1.x * aP2.x;
   aCoef[1] = aP1.y * aP2.y;
   aCoef[2] = aP1.z * aP2.z;
   aCoef[3] = aP1.x * aP2.y + aP1.y * aP2.x;
   aCoef[4] = aP1.x * aP2.z + aP1.z * aP2.x;
   aCoef[5] = aP1.y * aP2.z + aP1.z * aP2.y;
}



cTomKanSolver::cTomKanSolver
(
      const std::vector<std::vector<Pt2dr> > & aVVPt,
      double aDist,
      std::vector<CamStenopeIdeale> * aVRef
) :
   mDist   (aDist),
   mNbCam  (aVVPt.size()),
   mMatUV0  (1,1),
   mMatIJ0  (1,1),
   mDiag0   (1,1),
   mMatP0   (1,1),
   mMatUVSq (1,1),
   mMatIJSq (1,1),
   mDiagSq  (1,1),
   mMatPSq  (1,1),
   mW       (3,3),
   mRotW    (3,3),
   mDiagW   (3,3),
   mQ       (3,3),
   mVRef    (aVRef)
{
   // ====================================================
   //  [0]  Allocate Matrixes
   // ====================================================
   ELISE_ASSERT(mNbCam!=0,"cTomKanSolver no cam");

   mNbPts = aVVPt[0].size();

   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       mVC.push_back(cTomKanCamUnk(aVVPt[aKC],aDist,aVRef ? &(aVRef->at(aKC)) : 0 ));
       ELISE_ASSERT(mNbPts==mVC.back().mNbPts,"cTomKanSolver size Pts Diff");
   }
   mSzMat = ElMax(mNbPts,2*mNbCam);
   mMatUVSq = ElMatrix<double>(mSzMat,mSzMat,0.0);

   mMatUV0 =  ElMatrix<double>(mNbPts,2*mNbCam);
   mMatIJ0 =  ElMatrix<double>(3,2*mNbCam);
   mDiag0 =  ElMatrix<double>(3,3,0.0);
   mMatP0 =  ElMatrix<double>(mNbPts,3);

   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       mVPF3.push_back(&(mVC.at(aKC).VFInit()));
   }

   // ====================================================
   //  [1]  Fill UV
   // ====================================================
   ELISE_ASSERT(mNbCam!=0,"cTomKanSolver no cam");
   // Fill MatUV
   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       const std::vector<Pt2dr> & aVP = mVC[aKC].mVCenter;
       for (int aKP=0 ; aKP<mNbPts ; aKP++)
       {
           mMatUV0(aKP,aKC) = mMatUVSq(aKP,aKC       ) = aVP[aKP].x;
           mMatUV0(aKP,aKC+mNbCam) = mMatUVSq(aKP,aKC+mNbCam) = aVP[aKP].y;
       }
   }

   // ====================================================
   //  [2]  SVD decompose
   // ====================================================

     // 2.1

   svdcmp(mMatUVSq,mMatIJSq,mDiagSq,mMatPSq,false);

   // 2.2 Get the 3 index of  highest value in their initial order
   std::vector<int> aVInd;
   for (int aK=0 ; aK<mSzMat ; aK++)
   {
      aVInd.push_back(aK);
   }
   cCmpAbsMatDiag aCmp(mDiagSq);
   std::sort(aVInd.begin(),aVInd.end(),aCmp);  // Sort the index by value
   std::sort(aVInd.begin(),aVInd.begin()+3);   // Sort the 3 highest by their initial order


   // 2.3 On remets dans mDiag0 ... ce qu'on aurait si pas une SVD carre
   for (int aKVp=0 ; aKVp<3 ; aKVp++)
   {
       int aIVp = aVInd[aKVp];
       mDiag0(aKVp,aKVp)  = mDiagSq(aIVp,0);
       for (int aKC=0 ; aKC < mNbCam ; aKC++)
       {
           mMatIJ0(aKVp,aKC)        = mMatIJSq(aIVp,aKC);
           mMatIJ0(aKVp,aKC+mNbCam) = mMatIJSq(aIVp,aKC+mNbCam);
       }
       for (int aKP=0 ; aKP < mNbPts ; aKP++)
       {
           mMatP0(aKP,aKVp) = mMatPSq(aKP,aIVp);
       }
   }
   mMatP0 =  mDiag0 * mMatP0;

   // Check the decomposition mMatUV0 = mMatIJ0 * mDiag0 * mMatP0 
   // with mDiag0 3x3 ...
   if (0)
   {
      for (int aK=0 ; aK<mSzMat ; aK++)
      {
          std::cout << "DIAG="  << aVInd[aK] << " => " << mDiagSq(aVInd[aK],0) << "\n";
      }
      ElMatrix<double> aChk = mMatUV0 - mMatIJ0 *  mMatP0;
      std::cout << "CHEK Dec " << aChk.L2() << "\n";
      getchar();
   }

   // ====================================================
   //  [3]  Compute W symetric Matrix
   // ====================================================

      // 3.1 fill the equation
   L2SysSurResol aSys(6);
   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
        double aCoef1[6];
        double aCoef2[6];

        // ti0 W i0 = 1 = tj0 W j0
        if (aKC==0)
        {
           FillCoeffW(aCoef1,0,0);
           aSys.AddEquation(1.0,aCoef1,1.0);
           FillCoeffW(aCoef1,mNbCam,mNbCam);
           aSys.AddEquation(1.0,aCoef1,1.0);
        }
        else // tik W ik - tjk W jk = 0
        {
           FillCoeffW(aCoef1,aKC,aKC);
           FillCoeffW(aCoef2,aKC+mNbCam,aKC+mNbCam);
           for (int aK=0 ; aK<6 ; aK++)
           {
               aCoef1[aK] -= aCoef2[aK];
           }
           aSys.AddEquation(1.0,aCoef1,0);
        }

        // tik W jk = 0
        FillCoeffW(aCoef1,aKC,aKC+mNbCam);
        aSys.AddEquation(1.0,aCoef1,0);
   }


      // 3.2 solve and compute
   Im1D_REAL8  aSol = aSys.Solve((bool *) 0);
   double * aDS = aSol.data();
   
   mW(0,0) = aDS[0];
   mW(1,1) = aDS[1];
   mW(2,2) = aDS[2];
   mW(1,0) =  mW(0,1) = aDS[3];
   mW(2,0) =  mW(0,2) = aDS[4];
   mW(2,1) =  mW(1,2) = aDS[5];

   if (0)  // Check  metrics :  Ik . Jk ....
   {
       for (int aKC=0 ; aKC<mNbCam ; aKC++)
       {
           Pt3dr aI = GetPtI(aKC);
           Pt3dr aJ = GetPtJ(aKC);
           std::cout << scal(aI,mW*aI) << " " << scal(aJ,mW*aJ)  << " " << scal(aI,mW*aJ) << "\n";
       }
   }
   
   // ====================================================
   //  [4]  Compute Q matrix
   // ====================================================

   jacobi_diag(mW,mDiagW,mRotW);

   if (0)
   {
      ElMatrix<double> aCh = mW - mRotW * mDiagW * mRotW.transpose();
      std::cout << "DiiAgg " << mDiagW(0,0)  << " " << mDiagW(1,1)  << " " <<  mDiagW(2,2) << " L2=" << aCh.L2() << "\n";
      std::cout << "Det " << mRotW.Det() << "\n";
   }

   mVPNeg = false;
   for (int aK=0 ; aK<3 ; aK++)
   {
      // mDiagW(aK,aK) = sqrt(ElMax(0.0,mDiagW(aK,aK)));
      double aVP = mDiagW(aK,aK);
      mDiagW(aK,aK) = sqrt(ElAbs(aVP));
      if (aVP<0) 
         mVPNeg = true;
   }

   mQ = mRotW * mDiagW;
   
   mMatIJ0 = mMatIJ0 * mQ;
   mMatP0  = gaussj(mQ) * mMatP0;

   if (0)    // Check metrics
   {
       for (int aKC=0 ; aKC<mNbCam ; aKC++)
       {
           Pt3dr aI = GetPtI(aKC);
           Pt3dr aJ = GetPtJ(aKC);
           std::cout << "Metrics : " << scal(aI,aI) << " " << scal(aJ,aJ)  << " " << scal(aI,aJ) << "\n";
       }
   }
   if (0)  // Check factorisation
   {
       ElMatrix<double> aM = mMatUV0 - mMatIJ0*mMatP0;
       std::cout << "Factorisation, L2= " << aM.L2() << "\n";
   }
   if (0)  // Check projection
   {
      for (int aKC=0 ; aKC<mNbCam ; aKC++)
      {
          double aSomD = 0.0;
          for (int aKP=0 ; aKP<mNbPts ; aKP++)
          {
              Pt2dr aP1 = mVC[aKC].VCenterK(aKP);
              Pt3dr aPTer = GetPGround(aKP);
              Pt3dr aI = GetPtI(aKC);
              Pt3dr aJ = GetPtJ(aKC);
 
              aSomD += euclid(aP1,Pt2dr(scal(aPTer,aI),scal(aPTer,aJ)));
              
          }
          std::cout << " SomD[" << aKC << "]=" << aSomD << "\n";
      }
   }

   // ====================================================
   //  [5]  Export result
   // ====================================================

       //  5.1 Ground Pts
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
      // mVTer.push_back(Pt3dr(mMatP0(aKP,0),mMatP0(aKP,1),mMatP0(aKP,2)));
      mVTer.push_back(GetPGround(aKP));
   }
       //  5.2 Ortho camera
   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       Pt3dr aI = GetPtI(aKC);
       Pt3dr aJ = GetPtJ(aKC);
       mVC[aKC].SetCamOrtho(aI,aJ);
   }

   if (1)
   {
      double aSomDGlob=0.0;
      for (int aKC=0 ; aKC<mNbCam ; aKC++)
      {
          double aSomD = 0.0;
          for (int aKP=0 ; aKP<mNbPts ; aKP++)
          {
              Pt2dr aP1 = mVC[aKC].VInitK(aKP);
              Pt2dr aP2 = mVC[aKC].SolOrth().Proj(mVTer[aKP]);
              aSomD += euclid(aP1-aP2);
          }
          aSomDGlob += aSomD;
      }
   }
}

ElMatrix<double> cTomKanSolver::OrientOrtho(int aK)
{
    Pt3dr aI = GetPtI(aK);
    Pt3dr aJ = GetPtJ(aK);

    return MatFromCol(aI,aJ,aI^aJ).transpose();
}

void  cTomKanSolver::OrientCamStenope(int aNbTest,int aKCam,int aSign)
{
   std::vector<double> aPAF;
   cTomKanCamUnk & aTKC = mVC.at(aKCam);
   CamStenopeIdeale & aCam = aTKC.CamI(false);
   double & aSomD = aTKC.ResOFPA();

   std::list<Appar23> aL23;

   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       // Pt2dr aPProj = aCam.F2toPtDirRayonL3(aTKC.VInitK(aKP));
       Pt2dr aPProj = aTKC.VInitK(aKP);
       aL23.push_back(Appar23(aPProj*double(1),mVTer[aKP]*double(aSign)));
   }

   double  aDMin;
   ElRotation3D aR = aCam.RansacOFPA(true,aNbTest,aL23,&aDMin);

   // aCamUnused.SetOrientation(aR);
   aCam.SetOrientation(aR);

   aSomD = 0;
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       Pt2dr aPProjTh = aCam.F2toPtDirRayonL3(aTKC.VInitK(aKP));
       Pt2dr aPProj = aCam.Ter2Capteur(mVTer[aKP]);
       aSomD += euclid(aPProjTh,aPProj);
   }
   
   aSomD /= mNbPts;
}

double   cTomKanSolver::OrientAllCamStenope(int aNbTest,int aSign,std::vector<ElRotation3D> & aVSol)
{
    aVSol.clear();
    // Initial orientation with GCP
    for (int aKC=0 ; aKC<mNbCam ; aKC++)
    {
         OrientCamStenope(aNbTest,aKC,aSign);
    }

    // Bundle
    if (mNbCam ==3)
    {
    
       tMultiplePF aVPF3;
       for (int aKC=0 ; aKC<mNbCam ; aKC++)
       {
          aVPF3.push_back(&(mVC.at(aKC).VFInit()));
       }


       std::vector<Pt2df> aVNop;
       tMultiplePF aNoVPF2;
       for (int aKC=0 ; aKC<2 ; aKC++)
       {
          aNoVPF2.push_back(&aVNop);
       }
       
       bool TestRef = false;

       
    
       ElRotation3D aRWtoC1 = mVC.at(0).CamI(TestRef).Orient();
       ElRotation3D aRWtoC2 = mVC.at(1).CamI(TestRef).Orient();
       ElRotation3D aRWtoC3 = mVC.at(2).CamI(TestRef).Orient();

       ElRotation3D aRC2toC1 = (aRWtoC1 * aRWtoC2.inv());
       ElRotation3D aRC3toC1 = (aRWtoC1 * aRWtoC3.inv());


       Pt3dr  aPMed;
       double aBOnH;
       cParamCtrlSB3I aParamSB3I(10,false,1e-5);
       SolveBundle3Image
       (
          mDist,
          aRC2toC1, aRC3toC1,
          aPMed,aBOnH,
          aVPF3,
          aNoVPF2, aNoVPF2,aNoVPF2,
          1.0,
          aParamSB3I
      );

      aVSol.push_back(ElRotation3D::Id);
      aVSol.push_back(aRC2toC1);
      aVSol.push_back(aRC3toC1);

      return aParamSB3I.mRes3;
    }


    // Quality measurement
    std::vector<ElRotation3D>         aVRotInit;
    std::vector<ElRotation3D>         aVRotAfter;
    mSomResOFPA = 0;
    for (int aKC=0 ; aKC<mNbCam ; aKC++)
    {
         aVSol.push_back(mVC.at(aKC).CamI(false).Orient().inv());
         mSomResOFPA +=  mVC.at(aKC).ResOFPA();
/*
         if (mVRef)
         {
            aVRotInit.push_back((*mVRef)[aKC].Orient().inv());
            aVRotAfter.push_back(mVC.at(aKC).CamI(false).Orient().inv());
         }
*/
    }
    mSomResOFPA /= mNbCam;


/*

    // Print quality
    if (mVRef)
    {
        cSolBasculeRig aSol =  cSolBasculeRig::SolM2ToM1(aVRotInit,aVRotAfter);
        double aDMatr,aDCentr;
        aSol.QualitySol(aVRotInit,aVRotAfter,aDMatr,aDCentr);
        std::cout 
                      << " S=" << ((aSign>0) ? "+" : "-") 
                      << " STEN=" << mSomResOFPA 
                      << " DRef(M,Tr)=" << aDMatr << " " << aDCentr
                      << "\n";
        // getchar();
    }
*/

    return mSomResOFPA;

}

double   cTomKanSolver::OrientAllCamStenopeAllSigne(int aNbTest,std::vector<ElRotation3D> & aBestSol)
{
     double aDefScore = DefScoreTKS();
     double aBestScore = aDefScore;
     for (int aSign=-1 ; aSign<=1 ; aSign+=2)
     {
          std::vector<ElRotation3D> aVSol;
          double aScore = OrientAllCamStenope(aNbTest,aSign,aVSol);
          if (aScore<aBestScore)
          {
              aBestScore = aScore;
              aBestSol = aVSol;
          }
     }

     // En fait il y a des cas degenere rare qui engendre ces config, pas de raison de tout bloquer
     // ELISE_ASSERT(aBestScore<aDefScore,"cTomKanSolver::OrientAllCamStenopeAllSigne");
     return aBestScore;
}

cSolBasculeRig cTomKanSolver::TestSolSim(const std::vector<Pt3dr> & aV3D,int aSign, double & aSomDPts)
{
   std::vector<Pt3dr>   aVP3dApres;
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       aVP3dApres.push_back(GetPGround(aKP) * double(aSign));
   }

   cSolBasculeRig  aSBRPts = cSolBasculeRig::StdSolFromPts(aV3D,aVP3dApres);

   aSomDPts = 0.0;
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       ///aSomDPts += euclid(aVP3dAvant[aKP]-aSBRPts(aVP3dApres[aKP]));
       aSomDPts += euclid(aVP3dApres[aKP]-aSBRPts(aV3D[aKP]));
   }
   
   return aSBRPts;
}

double cTomKanSolver::TestSolLin(const std::vector<Pt3dr> & aV3d, ElMatrix<double> & aMat,Pt3dr & aTr)
{
// Solution lineaire
   L2SysSurResol aSys(12);
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       double aTabAp[3];
       GetPGround(aKP).to_tab(aTabAp);
       // aVP3dApres[aKP].to_tab(aTabAp);
       double * aDataAp = aTabAp;
       for (int aKxyz = 0 ; aKxyz<12; aKxyz+=4)
       {
           std::vector<double> aVCoeff(12,0.0);
           aVCoeff[aKxyz]   = aV3d[aKP].x;
           aVCoeff[aKxyz+1] = aV3d[aKP].y;
           aVCoeff[aKxyz+2] = aV3d[aKP].z;
           aVCoeff[aKxyz+3] = 1.0;
           aSys.AddEquation(1.0,VData(aVCoeff),*aDataAp);
           aDataAp++;
       }
   }

   Im1D_REAL8 aDSol = aSys.Solve(0);
   double * aDS = aDSol.data();
   double aVPts[3];
   for (int aKxyz = 0 ; aKxyz<12; aKxyz+=4)
   {
       aMat(0,aKxyz/4 ) = aDS[aKxyz];
       aMat(1,aKxyz/4 ) = aDS[aKxyz+1];
       aMat(2,aKxyz/4 ) = aDS[aKxyz+2];
       aVPts[aKxyz/4] = aDS[aKxyz+3];
   }
   aTr = Pt3dr::FromTab(aVPts);

   double aSomDPts = 0.0;
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       aSomDPts += euclid(GetPGround(aKP)-(aTr+aMat*aV3d[aKP]));
   }

   return aSomDPts / mNbPts;
}

double   cTomKanSolver::SomResOFPA() const
{
    return mSomResOFPA;
}


const tMultiplePF &  cTomKanSolver::VPF3() const 
{
   return mVPF3;
}


std::vector<ElRotation3D> OrientTomasiKanade
                          (
                             double &            aPrec,
                             const tMultiplePF & aVPF3,
                             int                 aNbMin,
                             int                 aNbMax,
                             double              aPrecCible,
                             std::vector<ElRotation3D> * aVRotInit
                          )
{
   ElTimer aChrono;
   cTKS_OptGlob aTOp(aVPF3);

   bool cont = true;
   int aCpt=0;

   while (cont)
   {
             aCpt++;
             aTOp.OneTest(6 );
             if (aCpt>=aNbMin)
             {
                if ((aTOp.ScMin() < aPrecCible) || (aCpt>=aNbMax))
                {
                   cont = false;
                }
             }
   }

   aPrec = aTOp.ScMin() ;
   if (aVRotInit)
   {
      aTOp.Show(aChrono.uval());

      cSolBasculeRig aSol =  cSolBasculeRig::SolM2ToM1(aTOp.BestR(),*aVRotInit);
      double aDMatr,aDCentr;
      aSol.QualitySol(aTOp.BestR(),*aVRotInit,aDMatr,aDCentr);
      std::cout << " DRef(M,Tr)=" << aDMatr << " " << aDCentr << "\n";

      getchar();
   }

   return aTOp.BestR();
}



void OneTestTomKan(int aNbC,int aNbPts,double aDist,double aNoise,double anOutLayer)
{
    // std::cin >> aNbC >> aNbPts  >> aDist >> aNoise;
    std::cout << "Nb Cam " << aNbC << " ; Nb Pts " << aNbPts  << " Dist " << aDist << " Noise=" << aNoise << " \n";

    std::vector<cBasicCamOrtho> aVCamO;
    std::vector<CamStenopeIdeale> aVCamI;
    std::vector<CamStenopeIdeale> aVCamINonOr;
    std::vector<std::vector<Pt2dr> > aVVPt0;
    std::vector<Pt3dr>               aVP3dAvant;

    std::vector<ElRotation3D>        aVRotInit;
    // std::vector<std::vector<Pt2df> > aVV3(aNbC);
    tMultiplePF aVPF3;

    for (int aKC=0 ;  aKC<aNbC ; aKC++)
    {
        aVPF3.push_back(new std::vector<Pt2df>);
        aVCamO.push_back(cBasicCamOrtho::RandomCam());
        aVVPt0.push_back(std::vector<Pt2dr>());

        if (aDist > 0)
        {
            double aRay = aDist * (1+ NRrandC()/7.0);
            Pt2dr  aPP = RanP2C();
// aPP = Pt2dr(0,0);

            Pt3dr aC = RanP3C();
            aC = vunit(aC) * aRay;
            Pt3dr aK = -vunit(aC + RanP3C() );
//Pt3dr aK = -vunit(aC);
            Pt3dr aI = vunit(aK ^ RanP3C());
            Pt3dr aJ = aK ^ aI;
            // ElMatrix<double> aRot = MatFromCol(aI,aJ,aK);
            std::vector<double> aPAF;
            CamStenopeIdeale aCam(true,aDist * (1+ NRrandC()/7.0) ,aPP,aPAF);
            aVCamINonOr.push_back(aCam);
            ElRotation3D aRot(aC,MatFromCol(aI,aJ,aK),true);
            aCam.SetOrientation(aRot.inv());
            aVCamI.push_back(aCam);
            aVRotInit.push_back(aRot);
        }
    }

    for (int aKP=0 ;  aKP<aNbPts ; aKP++)
    {
       Pt3dr aP = RanP3C();
       aVP3dAvant.push_back(aP);
        
       for (int aKC=0 ;  aKC<aNbC ; aKC++)
       {
           Pt2dr aProj = (aDist>0) ? aVCamI[aKC].Ter2Capteur(aP) : aVCamO[aKC].Proj(aP) ;
           Pt2dr aPNoise =  RanP2C()*aNoise;;
           aProj = aProj + aPNoise;
           if (NRrandom3() < anOutLayer)
           {
               aProj = aProj + RanP2C();  //  * NRrandC() * NRrandC() * NRrandC ();
           }
           if (aDist>0)
              aProj = aVCamI[aKC].F2toPtDirRayonL3(aProj);
           aVVPt0[aKC].push_back(aProj);
           aVPF3[aKC]->push_back(ToPt2df(aProj));
       }
    }
  
    double aPrec;
    std::vector<ElRotation3D>  aVR =OrientTomasiKanade (aPrec,aVPF3,3,50,1e-5,&aVRotInit);
}


int Test_TomCan(int argc,char ** argv)
{
    int aNbC=3;
    int aNbPts=4;
    double aDist=10;
    double aNoise= 1e-3;
    double anOutLayer= 0.0;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNbC,"Number Cam")
                    << EAMC(aNbPts,"Number Pts")
                    << EAMC(aDist,"Dist (+or- 1/FOV in radian)")
                    << EAMC(aNoise,"Noise added to image measurement")
                    << EAMC(anOutLayer,"Proba of out layer")
        ,
        LArgMain()  
    );

    while (1)
    {
       OneTestTomKan(aNbC,aNbPts,aDist,aNoise,anOutLayer);
    }
    return EXIT_SUCCESS;
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
