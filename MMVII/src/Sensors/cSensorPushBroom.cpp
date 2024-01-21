#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Tpl_Images.h"


/**
   \file cSensorCamPC.cpp  

   \brief file for push-broom sensor, will have several reorg ....
*/

namespace MMVII
{

tPoseR   cSensorImage::GetPoseLineSensor(tREAL8 aXY,bool isLineX,int aNbSample,bool * IsOk,std::vector<double> * aResCenterLine) const
{
   // let be optimistic of "Ok" initialization , sometime it works
   if (*IsOk)
   {
      *IsOk = true;
   }

   // P0 and P1 are the extremity of the line
   cPt2dr aP0 =  isLineX ? cPt2dr(aXY,0.0     )  :  cPt2dr(0.0     ,aXY);
   cPt2dr aP1 =  isLineX ? cPt2dr(aXY,Sz().y())  :  cPt2dr(Sz().x(),aXY);

   std::vector<tSeg3dr>  aVBundles;  // vector of valid bundles

   cPt3dr aAxeK = cPt3dr::PCste(0.0);  // will generate an average of the average viewing direction
   cPt3dr aAxeJ = cPt3dr::PCste(0.0);  // so that JK will generate the plane of direction

   // Parse to generate aNbSample regularly sampled on line P0-P1
   for (int aKPt=0 ; aKPt< aNbSample ; aKPt++)
   {
        tREAL8 aWeight0 = 1.0- (aKPt+0.5) / aNbSample ;
        cPt2dr aP = Centroid(aWeight0,aP0,aP1);
        if (IsVisibleOnImFrame(aP))
        {
             tSeg3dr  aBundle =  Image2Bundle(aP);
             aVBundles.push_back(aBundle);
             
             cPt3dr aVU = VUnit(aBundle.V12());
             aAxeK += aVU;
             aAxeJ += (aWeight0-0.5) * aVU;
        }
   }

   // Cannot do anything if no enough points
   if (aVBundles.size() <2)
   {
       // if user can manage error, signal it and return
       if (*IsOk)
       {
            *IsOk = false;
            return tPoseR(); // fake value
       }
       // else generate an error 
       MMVII_INTERNAL_ERROR("Not enough valid bundles in cSensorImage::GetRepairLoc");
   }

   aAxeK = VUnit(aAxeK);  // make K a unitary vector
   cPt3dr aAxeI = VUnit(aAxeJ ^ aAxeK);  // I => is orthonale to plane JK
   aAxeJ = VUnit(aAxeK ^ aAxeI); // to assure rigourous orthogonalisation

   cPt3dr aCenter =   BundleInters(aVBundles);

   // is user whish to have resiudual indicating how much the bundles where coplanar
   if (aResCenterLine)
   {
      aResCenterLine->resize(2,0.0);
      for (const auto & aB : aVBundles)
      {
          cSegmentCompiled<tREAL8,3> aSC(aB);
          aResCenterLine->at(0) += aSC.Dist(aCenter);  // idealy all the bunle cross the center
          aResCenterLine->at(1) += std::abs(Scal(aSC.Tgt(),aAxeI)); // ideally all the bundle are orthog to I
      }
      for (auto & aV : *aResCenterLine)
      {
          aV /= aVBundles.size();
      }
   }

   return tPoseR(aCenter,tRotR(aAxeI,aAxeJ,aAxeK,false));
}


cDenseMatrix<tREAL8>   cSensorImage::CalcDiffProjRot
                       (
                             const cPt3dr & aPXYZ,
                             const tPoseR & aPose,
                             const cPt3dr& aPtEpsIJZ,
                             const tREAL8& aEpsWPK
                       ) const
{
   cDenseMatrix<tREAL8> aJacXYZ(3,3);
   cDenseMatrix<tREAL8> aJacWPK(3,3);
   cPt3dr aPtIJZ = Ground2ImageAndZ(aPXYZ);
   cPt3dr aPLoc = aPose.Inverse(aPXYZ);

   for (size_t aK=0 ; aK<3 ; aK++)
   {
       tREAL8 aEpsXYZ = aPtEpsIJZ[aK];  // extract the good epsilon
       cPt3dr aEpsK = cPt3dr::P1Coord(aK,aEpsXYZ); // point in given direction
       cPt3dr  aDerXYZ = (ImageAndZ2Ground(aPtIJZ+aEpsK) - ImageAndZ2Ground(aPtIJZ-aEpsK)) /(2*aEpsXYZ);  // 
       SetCol(aJacXYZ,aK,aDerXYZ);

       cDenseMatrix<tREAL8>  aMat =  cRotation3D<tREAL8>::Rot1WPK(aK,aEpsWPK);
       //  PlocPert = aMat*aPLoc  ,  PGroundPert = aPose.Value(PlocPert) ,  PImPert = ImageAndZ2Ground(PGroundPert)
       cPt3dr aPLocPlus  =   ImageAndZ2Ground(aPose.Value(aMat*aPLoc));
       // As aMat is rotation, transpose is inverse, and v*M is in fact t(tV*M) = tM *v
       cPt3dr aPLocMoins =   ImageAndZ2Ground(aPose.Value(aPLoc*aMat));
       cPt3dr  aDerWPK  =  (aPLocPlus-aPLocMoins) / aEpsWPK;
       SetCol(aJacWPK,aK,aDerWPK);
   }

   return -aJacXYZ.Inverse() * aJacWPK;
}

void BenchStenopeSat(int aNum)
{
    cPerspCamIntrCalib * aCalib  = cPerspCamIntrCalib::RandomCalib(eProjPC::eStenope,aNum);

    cPt3dr aC  = cPt3dr::PRandC();
    tRotR aRot =  tRotR::RandomRot(0.1);

    cSensorCamPC aCamPC("TestStenopeSat",tPoseR(aC,aRot),aCalib);
    tREAL8 aFoc =  aCalib->F();

    cPt3dr aWPK =  cPt3dr::PRandC() / aFoc;
    tRotR aRotPert = tRotR::RotFromWPK(aWPK);
    cSensorCamPC aCamPCPert("TestStenopeSat",tPoseR(aC,aRot*aRotPert),aCalib);

FakeUseIt(aCamPCPert);
StdOut() <<  "BenchStenopeSatBenchStenopeSat " << aFoc << "\n";

    tREAL8 aMargin = 2.0;
    for (int aX = aMargin ; aX < aCalib->SzPix().x() - aMargin ; aX+=20)
    {
        bool Ok;
        std::vector<double> aRes;
        tPoseR  aPose=  aCamPC.GetPoseLineSensor(aX,true,30,&Ok,&aRes);

        MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Tr() - aC)<1e-5,"bundle ref in BenchStenopeSat");
        MMVII_INTERNAL_ASSERT_bench(aRes.at(0)<1e-5,"Residual bunles in BenchStenopeSat");
        MMVII_INTERNAL_ASSERT_bench(aRes.at(1)<1e-5,"Residual plane in BenchStenopeSat");

    }

    delete aCalib;
}
void BenchStenopeSat()
{
    for (int aK=0 ; aK<10 ; aK++)
    {
        BenchStenopeSat(0);
        BenchStenopeSat(3);
    }
}




}; // MMVII

