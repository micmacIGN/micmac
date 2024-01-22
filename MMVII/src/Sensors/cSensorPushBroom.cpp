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
   tREAL8 aZ = aPXYZ.z();
   const cPt3dr & aCenter = aPose.Tr();

// StdOut() << "PLL = " << aPLoc << "\n";

   for (size_t aK=0 ; aK<3 ; aK++)
   {
       tREAL8 aEpsXYZ = aPtEpsIJZ[aK];  // extract the good epsilon
       cPt3dr aEpsK = cPt3dr::P1Coord(aK,aEpsXYZ); // epsilon point in given direction
       cPt3dr  aDerXYZ = (ImageAndZ2Ground(aPtIJZ+aEpsK) - ImageAndZ2Ground(aPtIJZ-aEpsK)) /(2*aEpsXYZ);  // 
       SetCol(aJacXYZ,aK,aDerXYZ);
       // SetLine(aK,aJacXYZ,aDerXYZ);

       cDenseMatrix<tREAL8>  aMat =  cRotation3D<tREAL8>::Rot1WPK(aK,aEpsWPK);
       //  PlocPert = aMat*aPLoc  ,  PGroundPert = aPose.Value(PlocPert) 
       cPt3dr aPLocPlus  =   BundleFixZ(tSeg3dr(aCenter,aPose.Value(aMat*aPLoc)),aZ);
       // As aMat is rotation, transpose is inverse, and v*M is in fact t(tV*M) = tM *v
       cPt3dr aPLocMoins =    BundleFixZ(tSeg3dr(aCenter,aPose.Value(aPLoc*aMat)),aZ);
 //ImageAndZ2Ground(aPose.Value(aPLoc*aMat));

   // aPLocPlus = aPose.Value(aMat*aPLoc);
   // aPLocMoins = aPose.Value(aPLoc*aMat);
//StdOut() << "D=" <<  aDerXYZ   <<  "PTS=" << aPLocPlus <<  aPXYZ <<  aPLocMoins<< "\n";
       cPt3dr  aDerWPK  =  (aPLocPlus-aPLocMoins) / (2*aEpsWPK);
       SetCol(aJacWPK,aK,aDerWPK);
       // SetLine(aK,aJacWPK,aDerWPK);
   }

   return -aJacXYZ.Inverse() * aJacWPK;
}

class cBenchStenopeSat
{
    public :
};

void BenchStenopeSat(int aNum,bool isX)
{
   // cPerspCamIntrCalib * aCalib  = cPerspCamIntrCalib::RandomCalib(eProjPC::eStenope,aNum);
    cDataPerspCamIntrCalib aDPCIC("BStenSat",eProjPC::eStenope,cPt3di(0,0,1),1000, cPt2di(1000,1000));
    cPerspCamIntrCalib * aCalib  = new cPerspCamIntrCalib(aDPCIC);

    cPt3dr aC  = cPt3dr::PRandC();
    tRotR aRot =  tRotR::RandomRot(0.1);

    cSensorCamPC aCamPC("TestStenopeSat",tPoseR(aC,aRot),aCalib);
    tREAL8 aFoc =  aCalib->F();

    cPt3dr aWPK =  cPt3dr::PRandC() / (aFoc*1);
    tRotR aRotPert = tRotR::RotFromWPK(aWPK);

/*
cDenseMatrix<tREAL8>  aMatP =     tRotR::RotOmega(aWPK.x())
                               +  tRotR::RotPhi(aWPK.y())
                               +  tRotR::RotKappa(aWPK.z()) ;
     aMatP = aMatP - cDenseMatrix<tREAL8>::Identity(3)*2.0;
     StdOut() << "aWPKaWPK " << aWPK  << tRotR(aMatP,true).ToWPK() -aWPK << "\n";
     aRotPert = tRotR(aMatP,true);
*/

    cSensorCamPC aCamPCPertPlus("TestStenopeSat",tPoseR(aC,aRot*aRotPert),aCalib);
    cSensorCamPC aCamPCPertMinus("TestStenopeSat",tPoseR(aC,aRot*aRotPert.MapInverse()),aCalib);

    // tREAL8 aMargin = 2.0;
    int aSz0 = isX ?  aCalib->SzPix().x() :  aCalib->SzPix().y();
    int aSz1 = isX ?  aCalib->SzPix().y() :  aCalib->SzPix().x();

    int aNb0 = 3;
    int aNb1 = 3;
    //for (tREAL8 aCoord0=aMargin ; aCoord0<aSz0-aMargin ; aCoord0+=20)
    for (int aK0= 0 ; aK0< aNb0 ; aK0++)
    {
        bool Ok;
        tREAL8 aCoord0 =  aSz0 * ( aK0+0.5) / aNb0;
        std::vector<double> aRes;
        tPoseR  aPose=  aCamPC.GetPoseLineSensor(aCoord0,isX,30,&Ok,&aRes);

        MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Tr() - aC)<1e-5,"bundle ref in BenchStenopeSat");
        MMVII_INTERNAL_ASSERT_bench(aRes.at(0)<1e-5,"Residual bunles in BenchStenopeSat");
        MMVII_INTERNAL_ASSERT_bench(aRes.at(1)<1e-5,"Residual plane in BenchStenopeSat");

        // tREAL8 aCoord1 = RandInInterval(aMargin,aSz1-aMargin);

        for (int aK1= 0 ; aK1< aNb1 ; aK1++)
        {
             tREAL8 aCoord1 =  aSz1 * ( aK1+0.5) / aNb1;
             cPt2dr aPtIJ = isX ? cPt2dr(aCoord0,aCoord1)  : cPt2dr(aCoord1,aCoord0);
             tREAL8 aDepth = RandInInterval(1,10.0);
             cPt3dr aPXYZ = aCamPC.ImageAndDepth2Ground(TP3z(aPtIJ,aDepth));

             cPt3dr   aEpsPt (1,1,aDepth/(aSz0+aSz1));

             auto aMat = aCamPC.CalcDiffProjRot(aPXYZ,aPose,aEpsPt, 1/aFoc);

             //cPt2dr aGTDif = aCamPCPertPlus.Ground2Image(aPXYZ)-aPtIJ;
             cPt2dr aGTDif = ( aCamPCPertPlus.Ground2Image(aPXYZ)-aCamPCPertMinus.Ground2Image(aPXYZ) ) /2.0;
             cPt3dr aDerDif = aMat*aWPK ;

             StdOut() << "GT=" <<    aGTDif   << " DerDif" << aDerDif <<  " Delta="  <<  Proj(aDerDif) + aGTDif << "\n"; 
         }
         StdOut() << "===============  ENDLINE ============\n"; // getchar();
    }
    StdOut() << "ENDCAM \n"; getchar();

    delete aCalib;
}

// WPPkkk:: [-0.089067,0.0490246,-1.81514e-17]  [0.0884715,-0.0497044]     WPPkkk:: [-0.000595493,-0.00067973]
// WPPkkk:: [-0.89067,0.490246,-1.81514e-16]    [0.884704,-0.497031]       WPPkkk:: [-0.00596524,-0.0067843]

//  WPK sym :   WPPkkk:: [-0.00596756,-0.00678868] [0.884702,-0.497035]

// WPPkkk:: [-8.9067,4.90246,-1.81514e-15]      [8.84601,-4.96901]                  [-0.0606832,-0.0665438]


void BenchStenopeSat()
{
    for (int aK=0 ; aK<10 ; aK++)
    {
        BenchStenopeSat(0,true);
        BenchStenopeSat(3,true);
        // BenchStenopeSat(0,false);
        // BenchStenopeSat(3,false);
    }
}




}; // MMVII

