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
        cTomKanCamUnk(const std::vector<Pt2dr> & aVpt);
        const cBasicCamOrtho & SolOrth() const;
        void SetCamOrtho(const Pt3dr & aI,const Pt3dr & aJ);
        const Pt2dr & V0K(int) const;
        const Pt2dr & VCK(int) const;
    private :
        std::vector<Pt2dr> mV0;
        std::vector<Pt2dr> mVPt;
        int                mNbPts;
        Pt2dr              mCDG;
        cBasicCamOrtho     mSolOrth;
};


class cTomKanSolver
{
    public :
        cTomKanSolver(const std::vector<std::vector<Pt2dr> > & aVVPt);

        double OrientCamStenope(int aNbTest,CamStenope & aCam,int aKCam);
        bool VPNeg() const {return mVPNeg;}
    private :
        void FillCoeffW(double *,int aIP1,int aIP2) const;
        Pt3dr GetPIJ(int aIP) const;
        Pt3dr GetPGround(int aIP) const;

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

};



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

   // std::cout << " CCCO " << scal(mI,mJ) << " " << euclid(mI) << " " << euclid(mJ) << " " << aScale << "\n";
   // std::cout << "   co " << scal(mI,aI) / (euclid(aI)*euclid(mI)) << " " << scal(mJ,aJ) / (euclid(aJ)*euclid(mJ)) << "\n";
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

cTomKanCamUnk::cTomKanCamUnk(const std::vector<Pt2dr> & aVPt) :
   mV0      (aVPt),
   mVPt     (aVPt),
   mNbPts   (aVPt.size()),
   mCDG     (0,0),
   mSolOrth (Pt3dr(1,0,0),Pt3dr(0,1,0),1.0,Pt2dr(0,0))
{
   for (int aK=0 ; aK<mNbPts ; aK++)
      mCDG = mCDG + mVPt[aK];

   mCDG = mCDG / mNbPts;
   for (int aK=0 ; aK<mNbPts ; aK++)
       mVPt[aK] = mVPt[aK] - mCDG;

}


void cTomKanCamUnk::SetCamOrtho(const Pt3dr & aI,const Pt3dr & aJ)
{
   double aS = (euclid(aI) + euclid(aJ)) / 2.0;

   mSolOrth  = cBasicCamOrtho(vunit(aI),vunit(aJ),aS,mCDG);
}

const Pt2dr & cTomKanCamUnk::V0K(int aKP) const { return mV0.at(aKP); }
const Pt2dr & cTomKanCamUnk::VCK(int aKP) const { return mVPt.at(aKP);}


const cBasicCamOrtho & cTomKanCamUnk::SolOrth() const
{
   return mSolOrth;
}


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


Pt3dr cTomKanSolver::GetPIJ(int aIP) const
{
    return Pt3dr(mMatIJ0(0,aIP),mMatIJ0(1,aIP),mMatIJ0(2,aIP));
}

Pt3dr cTomKanSolver::GetPGround(int aIP) const
{
    return Pt3dr(mMatP0(aIP,0),mMatP0(aIP,1),mMatP0(aIP,2));
}

void cTomKanSolver::FillCoeffW(double * aCoef,int aIP1,int aIP2) const
{
   Pt3dr aP1 = GetPIJ(aIP1);
   Pt3dr aP2 = GetPIJ(aIP2);

   aCoef[0] = aP1.x * aP2.x;
   aCoef[1] = aP1.y * aP2.y;
   aCoef[2] = aP1.z * aP2.z;
   aCoef[3] = aP1.x * aP2.y + aP1.y * aP2.x;
   aCoef[4] = aP1.x * aP2.z + aP1.z * aP2.x;
   aCoef[5] = aP1.y * aP2.z + aP1.z * aP2.y;
}



cTomKanSolver::cTomKanSolver(const std::vector<std::vector<Pt2dr> > & aVVPt) :
   mNbCam (aVVPt.size()),
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
   mQ       (3,3)
{
   // ====================================================
   //  [0]  Allocate Matrixes
   // ====================================================
   ELISE_ASSERT(mNbCam!=0,"cTomKanSolver no cam");

   mNbPts = aVVPt[0].size();

   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       mVC.push_back(cTomKanCamUnk(aVVPt[aKC]));
       ELISE_ASSERT(mNbPts==mVC.back().mNbPts,"cTomKanSolver size Pts Diff");
   }
   mSzMat = ElMax(mNbPts,2*mNbCam);
   mMatUVSq = ElMatrix<double>(mSzMat,mSzMat,0.0);

   mMatUV0 =  ElMatrix<double>(mNbPts,2*mNbCam);
   mMatIJ0 =  ElMatrix<double>(3,2*mNbCam);
   mDiag0 =  ElMatrix<double>(3,3,0.0);
   mMatP0 =  ElMatrix<double>(mNbPts,3);


   // ====================================================
   //  [1]  Fill UV
   // ====================================================
   ELISE_ASSERT(mNbCam!=0,"cTomKanSolver no cam");
   // Fill MatUV
   for (int aKC=0 ; aKC<mNbCam ; aKC++)
   {
       const std::vector<Pt2dr> & aVP = mVC[aKC].mVPt;
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
          // std::cout << "DIAG="  << aVInd[aK] << " => " << mDiagSq(aVInd[aK],0) << "\n";
      }
      ElMatrix<double> aChk = mMatUV0 - mMatIJ0 *  mMatP0;
      std::cout << "CHEK Dec " << aChk.L2() << "\n";
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
           Pt3dr aI = GetPIJ(aKC);
           Pt3dr aJ = GetPIJ(aKC+mNbCam);
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
           Pt3dr aI = GetPIJ(aKC);
           Pt3dr aJ = GetPIJ(aKC+mNbCam);
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
              Pt2dr aP1 = mVC[aKC].VCK(aKP);
              Pt3dr aPTer = GetPGround(aKP);
              Pt3dr aI = GetPIJ(aKC);
              Pt3dr aJ = GetPIJ(aKC+mNbCam);
 
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
       Pt3dr aI = GetPIJ(aKC);
       Pt3dr aJ = GetPIJ(aKC+mNbCam);
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
              Pt2dr aP1 = mVC[aKC].V0K(aKP);
              Pt2dr aP2 = mVC[aKC].SolOrth().Proj(mVTer[aKP]);
              aSomD += euclid(aP1-aP2);
          }
          // std::cout << "   SomD[" << aKC << "]=" << aSomD << "\n";
          aSomDGlob += aSomD;
      }
      std::cout << "MoyDGlob=" << aSomDGlob / (mNbCam*mNbPts) << "\n";
   }
}

double  cTomKanSolver::OrientCamStenope(int aNbTest,CamStenope & aCam,int aKCam)
{
   cTomKanCamUnk & aTKC = mVC.at(aKCam);
   std::list<Appar23> aL23;

   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       Pt2dr aPProj = aCam.F2toPtDirRayonL3(aTKC.VCK(aKP));
       aL23.push_back(Appar23(aPProj,mVTer[aKP]));
   }

   double  aDMin;
   ElRotation3D aR = aCam.RansacOFPA(true,aNbTest,aL23,&aDMin);

   aCam.SetOrientation(aR);

   double aSomD = 0;
   for (int aKP=0 ; aKP<mNbPts ; aKP++)
   {
       Pt2dr aPProjTh = aCam.F2toPtDirRayonL3(aTKC.VCK(aKP));
       Pt2dr aPProj = aCam.Ter2Capteur(mVTer[aKP]);
       aSomD += euclid(aPProjTh,aPProj);
   }
   return aSomD / mNbPts;
}


void OneTestTomKan() 
{
    int aNbC=3;
    int aNbPts=4;
    double aDist=10;
     // Dist joue +ou- le role de focale, +ou- FOV en radian de 1/Dist
     // Si dist <=0  => on utilise les camera orthos
    double aNoise= 1e-3;
    std::cout << "Enter NbCam NbPts Dist Noise \n";

    // std::cin >> aNbC >> aNbPts  >> aDist >> aNoise;
    std::cout << "Nb Cam " << aNbC << " ; Nb Pts " << aNbPts  << " Dist " << aDist << " Noise=" << aNoise << " \n";

    std::vector<cBasicCamOrtho> aVCamO;
    std::vector<CamStenopeIdeale> aVCamI;
    std::vector<CamStenopeIdeale> aVCamINonOr;
    std::vector<std::vector<Pt2dr> > aVVPt;
    std::vector<Pt3dr>               aVP3d;

    for (int aKC=0 ;  aKC<aNbC ; aKC++)
    {
        aVCamO.push_back(cBasicCamOrtho::RandomCam());
        aVVPt.push_back(std::vector<Pt2dr>());

        if (aDist > 0)
        {
            double aRay = aDist * (1+ NRrandC()/3.0);
            Pt2dr  aPP = RanP2C();

            Pt3dr aC = RanP3C();
            aC = vunit(aC) * aRay;
            Pt3dr aK = -vunit(aC + RanP3C() );
            Pt3dr aI = vunit(aK ^ RanP3C());
            Pt3dr aJ = aK ^ aI;
            // ElMatrix<double> aRot = MatFromCol(aI,aJ,aK);
            std::vector<double> aPAF;
            CamStenopeIdeale aCam(true,aDist * (1+ NRrandC()/5.0) ,aPP,aPAF);
            aVCamINonOr.push_back(aCam);
            ElRotation3D aRot(aC,MatFromCol(aI,aJ,aK),true);
            aCam.SetOrientation(aRot.inv());
            aVCamI.push_back(aCam);
        }
    }


    for (int aKP=0 ;  aKP<aNbPts ; aKP++)
    {
       Pt3dr aP = RanP3C();
       aVP3d.push_back(aP);
        
       for (int aKC=0 ;  aKC<aNbC ; aKC++)
       {
           Pt2dr aProj = (aDist>0) ? aVCamI[aKC].Ter2Capteur(aP) : aVCamO[aKC].Proj(aP) ;
           aVVPt[aKC].push_back(aProj + RanP2C()*aNoise);
           // std::cout << "PROJ " << aProj << "\n";
       }
       // std::cout << "\n";
    }


   cTomKanSolver aTKS(aVVPt);

   if (aDist>0)
   {
      for (int aKC=0 ; aKC<aNbC ; aKC++)
      {
          double aRes = aTKS.OrientCamStenope(ElMin(30,(aNbPts*(aNbPts+1)/2)), aVCamINonOr[aKC],aKC);

          std::cout << "RESIDU STEN = " << aRes << "\n";
      }
   }
   if (aTKS.VPNeg()) 
      getchar();
}

int Test_TomCan(int argc,char ** argv)
{
    while (1)
    {
       OneTestTomKan();
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
