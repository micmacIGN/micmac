#include "include/MMVII_all.h"
#include "include/MMVII_Derivatives.h"


namespace MMVII
{
static const int SzJetCam=17;

/** \file TestJetsCam.cpp
    \brief Make benchmark on jets vs  with camera projection

    This file contain test on efficiency of jets , sparse and dense,
  compared to generated code. Ceres implementation is still to add.

*/

/* ************************************** */
/*                                        */
/*           cVarEpsNum<int N>            */
/*                                        */
/* ************************************** */

double cVarEpsNum::BufEps[SzBuf];
bool   cVarEpsNum::IsInit = false;



/* ************************************** */
/*                                        */
/*           cProjCamRad<Type>            */
/*                                        */
/* ************************************** */

/** Template class for implementing the projection formula,
    can be used with any type having the 4 operation "+*-/"
    (i.e double, jets, sparse jets ...)
*/
template <class Type>  class cProjCamRad 
{
    public :
       void Compute (Type * Residual);

        cProjCamRad();
        cDenseMatrix<double>  mRotCur;
        cPt2dr  mPix;

        Type mParameter[SzJetCam];
};

template <class Type>  cProjCamRad<Type>::cProjCamRad() :
    mRotCur(3,3,eModeInitImage::eMIA_MatrixId),
    mPix   (0,0)
{
}

template <class Type>  void cProjCamRad<Type>::Compute(Type * Residu)
{
    // Ground Coordinates of projected point
    Type & XTer = mParameter[0];
    Type & YTer = mParameter[1];
    Type & ZTer = mParameter[2];

    // Coordinate of camera center
    Type & C_XCam = mParameter[3];
    Type & C_YCam = mParameter[4];
    Type & C_ZCam = mParameter[5];

    // Coordinate of Omega vector coding the unknown "tiny" rotation
    Type & Wx = mParameter[6];
    Type & Wy = mParameter[7];
    Type & Wz = mParameter[8];

    // Coordinate Center of distorstion
    Type & xCD = mParameter[9];
    Type & yCD = mParameter[10];

    // Distortions coefficients
    Type & k2D = mParameter[11];
    Type & k4D = mParameter[12];
    Type & k6D = mParameter[13];

    // PP and Focal
    Type & xPP = mParameter[14];
    Type & yPP = mParameter[15];
    Type & zPP = mParameter[16]; // also named as focal

    // Vector P->Cam
    Type  XPC = XTer-C_XCam;
    Type  YPC = YTer-C_YCam;
    Type  ZPC = ZTer-C_ZCam;


    // Coordinate of points in  camera coordinate system, do not integrate "tiny" rotation

    Type  XCam0 = mRotCur(0,0)*XPC +  mRotCur(1,0)*YPC +  mRotCur(2,0)*ZPC;
    Type  YCam0 = mRotCur(0,1)*XPC +  mRotCur(1,1)*YPC +  mRotCur(2,1)*ZPC;
    Type  ZCam0 = mRotCur(0,2)*XPC +  mRotCur(1,2)*YPC +  mRotCur(2,2)*ZPC;
     
    // Now "tiny" rotation
    //  Wx      X      Wy * Z - Wz * Y
    //  Wy  ^   Y  =   Wz * X - Wx * Z
    //  Wz      Z      Wx * Y - Wy * X
   
     //  P =  P0 + W ^ P0 
    
    Type  XCam = XCam0 + Wy * ZCam0 - Wz * YCam0;
    Type  YCam = YCam0 + Wz * XCam0 - Wx * ZCam0;
    Type  ZCam = ZCam0 + Wx * YCam0 - Wy * XCam0;

    // Projection 

    Type xPi =  XCam/ZCam;
    Type yPi =  YCam/ZCam;


    // Coordinate relative to distorsion center
    Type xC =  xPi-xCD;
    Type yC =  yPi-yCD;
    Type Rho2C = Square(xC) + Square(yC);

   // Compute the distorsion
    Type Dist = k2D*Rho2C + k4D * Square(Rho2C) + k6D*Cube(Rho2C);
    
    Type xDist =  xPi + xC * Dist;
    Type yDist =  yPi + yC * Dist;
    
   // Use principal point and focal
    Type xIm =  xPP  + zPP  * xDist;
    Type yIm =  yPP  + zPP  * yDist;

    Residu[0] = xIm - mPix.x();
    Residu[1] = yIm - mPix.y();
}

/* ************************************** */
/*                                        */
/*         cVarJetsTestCam                */
/*                                        */
/* ************************************** */

/** Class using sparse jets to implement camera projection & derivatives */

class  cVarJetsTestCam : public cInterfaceTestCam
{
    public : 
       void  InitFromParams(const std::vector<double> &) override;
       void  Compute(std::vector<double> & Vals,std::vector<std::vector<double> > & ) override;
       void  Compute(int aNb) override;
    private : 
       cProjCamRad<cVarEpsNum> mVarJetsCam;
};

void  cVarJetsTestCam::InitFromParams(const std::vector<double> & aVParam)
{
     for (int aK=0 ; aK<int(aVParam.size()) ; aK++)
        mVarJetsCam.mParameter[aK] =  cVarEpsNum(aVParam[aK],aK);
}

void  cVarJetsTestCam::Compute(std::vector<double> & Vals,std::vector<std::vector<double> > & aDer) 
{
      Vals = std::vector<double>(2);
      aDer.clear();

      cVarEpsNum  aResidu[2];
      mVarJetsCam.Compute(aResidu);
      for (int aDimXY=0 ; aDimXY<2 ; aDimXY++)
      {
           const cVarEpsNum & aR=   aResidu[aDimXY];
           Vals[aDimXY] = aR.mNum;
           aDer.push_back(std::vector<double>(SzJetCam,0.0));
           for (int aKParam=0 ; aKParam<int(aR.mVInd.size()) ; aKParam++)
           {
               aDer[aDimXY][aR.mVInd[aKParam]] =  aR.mVEps[aKParam];
           }
      }
}
void  cVarJetsTestCam::Compute(int aNb)
{
   cVarEpsNum  aResidu[2];
   for (int aK=0 ; aK<aNb ; aK++)
      mVarJetsCam.Compute(aResidu);
}


/* ************************************** */
/*                                        */
/*           cJetsTestCam                 */
/*                                        */
/* ************************************** */

class  cJetsTestCam : public cInterfaceTestCam
{
    public : 
       void  InitFromParams(const std::vector<double> &) override;
       void  Compute(std::vector<double> & Vals,std::vector<std::vector<double> > & ) override;
       void  Compute(int aNb) override;
    private : 
       cProjCamRad<cEpsNum<SzJetCam> > mJetsCam;

};

void  cJetsTestCam::InitFromParams(const std::vector<double> & aVParam)
{
     for (int aK=0 ; aK<SzJetCam ; aK++)
        mJetsCam.mParameter[aK] =  cEpsNum<SzJetCam>(aVParam[aK],aK);
}
   
void  cJetsTestCam::Compute(std::vector<double> & Vals,std::vector<std::vector<double> > & aDer) 
{
      Vals = std::vector<double>(2);
      aDer.clear();

      cEpsNum<SzJetCam>  aResidu[2];
      mJetsCam.Compute(aResidu);
      for (int aDimXY=0 ; aDimXY<2 ; aDimXY++)
      {
           Vals[aDimXY] = aResidu[aDimXY].mNum;
           aDer.push_back(std::vector<double>(SzJetCam));
           for (int aKParam=0 ; aKParam<SzJetCam ; aKParam++)
           {
               aDer[aDimXY][aKParam] =  aResidu[aDimXY].mEps[aKParam];
           }
      }
}

void  cJetsTestCam::Compute(int aNb)
{
   cEpsNum<SzJetCam>  aResidu[2];
   for (int aK=0 ; aK<aNb ; aK++)
      mJetsCam.Compute(aResidu);
}

/* ************************************** */
/*                                        */
/*                ::                      */
/*                                        */
/* ************************************** */

/**
     Generate a Random vector "realistic" of parameter for the camera projection
    test
*/

std::vector<double>    StdParamTestCam(double AmplNoise)
{
    std::vector<double> aRes;

    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //X-Gr
    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //Y-Gr
    aRes.push_back(1.0+0.01*RandUnif_C()*AmplNoise);  //Z-Gr

    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //X-Cam
    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //Y-Cam
    aRes.push_back(0.0+0.01*RandUnif_C()*AmplNoise);  //Z-Cam

    aRes.push_back(0.0);   // W-x   Mandotary 000 as it is the complementary rotation
    aRes.push_back(0.0);   // W-y
    aRes.push_back(0.0);   // W-z

    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // Centre dist X
    aRes.push_back(0.02 *RandUnif_C()*AmplNoise);   // Centre dist Y


    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // K1
    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // K2
    aRes.push_back(0.01 *RandUnif_C()*AmplNoise);   // K3


    aRes.push_back(3000 * (1+ 0.01 *RandUnif_C()*AmplNoise));   // PPx
    aRes.push_back(2000 * (1+ 0.01 *RandUnif_C()*AmplNoise));   // PPy
    aRes.push_back(6000 * (1+ 0.01 *RandUnif_C()*AmplNoise));   // PPz / Focale

    return aRes;

}

void BenchJetsCam(cParamExeBench & aParam)
{
   if (! aParam.NewBench("CamJetDerivative")) return;

   int aNb = 1+aParam.Level();

   for (int aK=0 ; aK<aNb ; aK++)
   {
      std::vector<double>  aVParam =  StdParamTestCam(1.0);
      std::vector<double> aValJet,aValV1,aVarValJet;
      std::vector<std::vector<double> > aDerJet,aDerV1,aVarDerJet;

      cJetsTestCam aCamJet;
      aCamJet.InitFromParams(aVParam);
      aCamJet.Compute(aValJet,aDerJet);

      cVarJetsTestCam aVarCamJet;
      aVarCamJet.InitFromParams(aVParam);
      aVarCamJet.Compute(aVarValJet,aVarDerJet);


      std::shared_ptr<cInterfaceTestCam>  aCamV1 (cInterfaceTestCam::AllocMMV1());
      aCamV1->InitFromParams(aVParam);
      aCamV1->Compute(aValV1,aDerV1);

      for (int aKXY=0 ; aKXY<2 ; aKXY++)
      {
           double aDif = aValJet[aKXY]-aValV1[aKXY];
           MMVII_INTERNAL_ASSERT_bench(std::abs(aDif)<1e-5,"Jets/V1 vals");

           aDif = aVarValJet[aKXY]-aValV1[aKXY];

           // StdOut() << "DDD=" << aDif<< " " << aValJet[aKXY] << " " << aValV1[aKXY] << "\n";
           for (int aKD=0 ; aKD<SzJetCam ; aKD++)
           {
                double aDif = aDerV1[aKXY][aKD]-aDerJet[aKXY][aKD];
                MMVII_INTERNAL_ASSERT_bench(std::abs(aDif)<1e-5,"Jets/V1 deriv");
                aDif = aDerV1[aKXY][aKD]-aVarDerJet[aKXY][aKD];
                MMVII_INTERNAL_ASSERT_bench(std::abs(aDif)<1e-5,"Jets/V1 deriv");
                // StdOut() << " Der=" << aDif << " " << aDerV1[aKXY][aKD]  << "\n";
           }
      }
      int aNbTest=1000 * (1+std::min(20,aParam.Level()));
      double aT0 = cMMVII_Appli::CurrentAppli().SecFromT0();
      aCamJet.Compute(aNbTest);
      double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
      aCamV1->Compute(aNbTest);
      double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();
      // Now we now that sparse jets are slow save some time
      double aT3 = aT2;
      if (aK==0)
      {
         aVarCamJet.Compute(aNbTest);
         aT3 = cMMVII_Appli::CurrentAppli().SecFromT0();
      }

      if (aParam.Show())
      {
         StdOut() << "TimeJets=" << (aT1-aT0) 
                  << " TimeV1=" << (aT2-aT1) 
                  << " TVar=" << (aT3-aT2) 
                  << "\n";
      }
   }
   aParam.EndBench();
}


};
