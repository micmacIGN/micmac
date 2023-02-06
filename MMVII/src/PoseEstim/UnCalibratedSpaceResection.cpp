#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"


// #include "MMVII_nums.h"
// #include "MMVII_Geom3D.h"
// #include "cMMVII_Appli.h"


/**
   \file UnCalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in uncalibrated case

 */

/* We have the image formula w/o distorsion:

   (u v)  = PI0   R (P-c)

   we consider a  calibraytion with linear distorsion :

   (I) =  (PPx + F (u  p1 u + p2 v))  =  (F(1+p1)   p2F  PPx) (u)  =  (a b c) (u) =  C (u)
   (J)    (PPy + F v               )     (0         F    PPy) (v)     (d e f) (v)      (v)


   (u v) =  PI0 (u v 1)       C (u)  =  PI0 (a  b  c) (u)
                                (v)         (d  e  f) (v)
                                            (0  0  1) (1)
   (I)  =  C PI0 R (P-c)
   (J)
*/

namespace MMVII
{
bool MMVII_IN_DEBUG = false;

template <class Type,const int Dim> class cAffineForm
{
    public :
       typedef cPtxd<Type,Dim>      tPt;

       Type Value(const tPt & aP) const {return Scal(mForm,aP) + mCste;}

       cAffineForm(const Type * aV);

       const tPt&  Form() const {return mForm;}
       const Type& Cste() const {return mCste;}
    private :
       tPt  mForm;
       Type mCste;
};

template <class Type,const int Dim> 
   cAffineForm<Type,Dim>::cAffineForm(const Type * aV) :
       mForm   (tPt(aV)),
       mCste   (aV[Dim])
{
}


template <class Type> class cHomog2D3D
{
    public :
       static constexpr int       TheDim=3;

       typedef cPtxd<Type,3>      tPtIn;
       typedef cPtxd<Type,2>      tPtOut;

       tPtOut  Value(const tPtIn & aP)  const  
       {
	       return tPtOut(mFx.Value(aP),mFy.Value(aP)) / mFz.Value(aP);
       }
       tPt2dr  RValue(const tPt3dr & aP)const  {return ToR(Value(tPtIn::FromPtR(aP)));}

       cHomog2D3D(const Type *);
       cHomog2D3D();

       cDenseMatrix<Type>  Mat() const;
       tPtIn               Tr()  const;
       const cAffineForm<Type,3> & Fx() const;
       const cAffineForm<Type,3> & Fy() const;
       const cAffineForm<Type,3> & Fz() const;

    private :

       cAffineForm<Type,3>  mFx;
       cAffineForm<Type,3>  mFy;
       cAffineForm<Type,3>  mFz;

};

template <class Type> 
   cHomog2D3D<Type>::cHomog2D3D(const Type * aV) :
	mFx  (aV+0),
	mFy  (aV+4),
	mFz  (aV+8)
{
}

template <class Type> cHomog2D3D<Type>::cHomog2D3D() :
	cHomog2D3D(std::vector<Type>({1,0,0,0,  0,1,0,0,  0,0,0,1}).data())
{
}

template <class Type>  cDenseMatrix<Type> cHomog2D3D<Type>::Mat() const
{
     return  M3x3FromLines(mFx.Form(),mFy.Form(),mFz.Form());
}

template <class Type> cPtxd<Type,3> cHomog2D3D<Type>::Tr() const
{
    return cPtxd<Type,3> (mFx.Cste(),mFy.Cste(),mFz.Cste());
}
template <class Type>   const cAffineForm<Type,3> &  cHomog2D3D<Type>::Fx() const {return mFx;}
template <class Type>   const cAffineForm<Type,3> &  cHomog2D3D<Type>::Fy() const {return mFy;}
template <class Type>   const cAffineForm<Type,3> &  cHomog2D3D<Type>::Fz() const {return mFz;}

// template <class TMap>  tREAL8  RValue(const TMap& aMap)

template <class TMap>  tREAL8  AvgReProj(const cSet2D3D & aSet,const TMap& aMap)
{
    cWeightAv<tREAL8>  aWAvg;

    for (const auto & aPair : aSet.Pairs())
    {
        aWAvg.Add(aPair.mWeight,SqN2(aPair.mP2-aMap.RValue(aPair.mP3)));
    }

    return aWAvg.Average();
}


/**
 *
 * Class for solving the "11 parameter" equation, AKA uncalibrated resection
 */
template <class Type>  class cUncalibSpaceRessection
{
      public :
           cUncalibSpaceRessection
           (
	       const cPt2di & aSz,
	       const cSet2D3D & aSetProj,
	       const cSensorCamPC * aGTCam = nullptr
	   );

       private :

	   /// Compute least square system w/o any constraint on sols
	   void  CalcLeastSquare_WOConstr();

	   /// Test all constraint on all possible variable and store the best
	   void  Test_WithAllConstr();
	   
	   ///  Compute Parameters
	   void  ComputeParameters();


	   static void SetVect(cDenseVect<Type> & aV,int anInd,const tPt3dr & aP,double aMul);

	   void CalcSolOneVarFixed(int aK);

	   void AddOneEquation(const cWeightedPair2D3D & aPair);

           cPt2di            mSz;
	   cSet2D3D          mSet;
	   cLeasSqtAA<Type>  mSys0;
	   cPair2D3D         mCentroid;
	   Type              mSumW;
	   cDenseVect<Type>  mVecW;
           cWhichMin<cHomog2D3D<Type>,Type>  mBestH;
	   const cSensorCamPC * mGTCam;
};


template <class Type>  
    cUncalibSpaceRessection<Type>::cUncalibSpaceRessection(const cPt2di& aSz,const cSet2D3D & aSet,const cSensorCamPC * aGTCam) :
        mSz   (aSz),
	mSet  (aSet),
        mSys0 (12),
	mCentroid (mSet.Centroid()),
	mSumW (0),
        mVecW (12, eModeInitImage::eMIA_Null),
	mBestH (cHomog2D3D<Type>(),1e30),
	mGTCam (aGTCam)
{
    // substract centroid to make more stable system with "big" coords
    mSet.Substract(mCentroid);

    //  LSQ W/o  any constraint
    CalcLeastSquare_WOConstr();

    //  Test all possible constraint to make the system well defined
    Test_WithAllConstr();

    ComputeParameters();
}

/*   ========================================
 *   Methods for computing Var/Cov
 *   ======================================== */

template <class Type>  
    void cUncalibSpaceRessection<Type>::SetVect
         (
            cDenseVect<Type> & aV,
            int aInd,
            const tPt3dr & aP,
            double aMul
	  )
{
	aV(aInd+0) = aP.x() * aMul;
	aV(aInd+1) = aP.y() * aMul;
	aV(aInd+2) = aP.z() * aMul;
	aV(aInd+3) =          aMul;
}

/*
      I =  (v0 x + v1 y + v2 z + v3) /  (v8 x + v9 y + v10 z + v11)
      J =  (v4 x + v5 y + v6 z + v7) /  (v8 x + v9 y + v10 z + v11)

      0 =  (v0 x + v1 y + v2 z + v3)  - I *  (v8 x + v9 y + v10 z + v11)
      0 =  (v4 x + v5 y + v6 z + v7)  - J *  (v8 x + v9 y + v10 z + v11)
*/


template <class Type>  void  cUncalibSpaceRessection<Type>::AddOneEquation(const cWeightedPair2D3D & aPair)
{
    for (const auto & IsX : {true,false})
    {
       double aW = aPair.mWeight;
       cPt3dr aP3 = aPair.mP3 ;

       cDenseVect<Type>  aVect (12, eModeInitImage::eMIA_Null);
       SetVect( aVect , (IsX ? 0 : 4) , aP3 ,  1.0                                   );
       SetVect( aVect ,             8 , aP3 , - (IsX ?aPair.mP2.x() : aPair.mP2.y()) );
       mSys0.AddObservation(aW,aVect,0.0);

       mSumW += aW;
       for (int aK=0 ; aK<12  ; aK++)
           mVecW(aK) += aW * Square(aVect(aK));
    }
}
template <class Type>  void    cUncalibSpaceRessection<Type>::CalcLeastSquare_WOConstr()
{
    for (const auto & aPair : mSet.Pairs())
        AddOneEquation(aPair);

}

/*   ========================================
 *   Methods for computing sols of homographie
 *   ======================================== */

template <class Type>  void cUncalibSpaceRessection<Type>::CalcSolOneVarFixed(int aKV)
{

     cLeasSqtAA<Type>  aSys = mSys0.Dup();
     double aW = mSumW * std::sqrt(mVecW(aKV)/mSumW);
     aSys.AddObsFixVar(aW,aKV,1.0);

     cDenseVect<Type>  aSol = aSys.Solve();
     cHomog2D3D<Type>  aHom(aSol.RawData());

     tREAL8 aScore = AvgReProj(mSet,aHom);
    // StdOut()  << " LLLxxxx " << aSys.tAA().DIm().L2Norm()  <<  " ww=" << aW << "\n";
    // StdOut()  << " SSSS " << aSol(0)  << " " << aSol(1) << " " << aSol(11) << "\n";
     mBestH.Add(aHom, aScore);
}

template <class Type>  void    cUncalibSpaceRessection<Type>::Test_WithAllConstr()
{

    for (int aKV=0 ; aKV<12 ; aKV++)
    {
        CalcSolOneVarFixed(aKV);
    }

    if (mGTCam) 
    {
       double aResidual =  mBestH.ValExtre();
       MMVII_INTERNAL_ASSERT_bench(aResidual < tElemNumTrait<Type>::Accuracy()*1e-2,"Residual homogr in cUncalibSpaceRessection");
    }
}

/*
     (i)            (v0 v1 v2 )  (x)    (v3 )
     (j)  =  Pi0 {  (v4 v5 v6 )  (y)  + (v7 ) }  = Pi0 ( M P + Q) = Pi0 (M (P-  (-M^(-1) Q)))
                    (v8 v9 v10)  (z)    (v11)
*/

template <class Type>  void    cUncalibSpaceRessection<Type>::ComputeParameters()
{
    cDenseMatrix<Type> aMat = mBestH.IndexExtre().Mat();
    cPtxd<Type,3>      aTr  = mBestH.IndexExtre().Tr();
    cPt3dr      aCLoc   =  ToR(SolveCol(aMat,aTr)) * -1.0 ;
    cPt3dr      aCAbs   =  aCLoc+ mCentroid.mP3;

    // cPt3dr aC =  mGTCam->Center() - mCentroid.mP3;

    // StdOut() << " CCC=" <<  aTr    << aMat*aC << "\n"; 
    if (mGTCam)
    {
       double aD =  Norm2(mGTCam->Center() - aCAbs) ;
       // StdOut() << " CCC=" <<  Norm2(mGTCam->Center() - aCAbs) << "\n"; 
       if (0) // (aD >= 1e-2)
       {
            StdOut() <<  "DIST CENTERS= " << aD << "\n";
            MMVII_INTERNAL_ASSERT_bench(aD < tElemNumTrait<Type>::Accuracy()*1e-2,"Center in cUncalibSpaceRessection");
       }
    }
    
    cResulRQ_Decomp<tREAL8>  aRQ = Convert((tREAL8*)nullptr,aMat).RQ_Decomposition();

    cDenseMatrix<tREAL8>  aRot = aRQ.Q_Matrix().Transpose(); // transpose to change W->C into Cam->Word for pose
    if (aRot.Det() <0) // remember  matrix are udefined up to scale
       aRot = aRot * -1;

    // matrix beign defined up to a scale, fix R(2,2) = 1 
    aRQ.R_Matrix().DIm() *=  (1.0/aRQ.R_Matrix().GetElem(2,2));

    /*
    StdOut() << " LLLL " << aRot.L2Dist(mGTCam->Pose().Rot().Mat()) << "\n";
    StdOut() << " QQQQQQQQQQ  \n";
    aRot.Show();
    StdOut() << " MMMMMMMMMM  \n";
    mGTCam->Pose().Rot().Mat().Show();
    StdOut() << " RRRR  "  << aRQ.R_Matrix().Det() << "\n";
    aRQ.R_Matrix().Show();
    */

    const cDenseMatrix<tREAL8> & aR = aRQ.R_Matrix();
    double aF = aR.GetElem(1,1);
    double  aB1 = (aR.GetElem(0,0)- aF)/aF;
    double  aB2 = aR.GetElem(1,0)/aF;
    cPt2dr  aPP =  mCentroid.mP2 + cPt2dr(aR.GetElem(2,0),aR.GetElem(2,1)) ;

    // StdOut() << "   F ,PP  "  << aF << " " << aPP << "\n";
    // StdOut() << " B1B2  "  << aB1 << " " << aB2 << "\n";

    cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::Alloc
                                 (
                                         cDataPerspCamIntrCalib
                                         (
                                               "UncalibSpaceRessection",
                                                eProjPC::eStenope,
                                                cPt3di(0,0,1),
                                                std::vector<double>(),
                                                cCalibStenPerfect(aF,aPP),
                                                cDataPixelDomain(mSz),
                                                cPt3di(0,0,1),
                                                10
                                         )
                                 );
      cMMVII_Appli::AddObj2DelAtEnd(aCalib);

      aCalib->SetParamDist("b1",aB1);
      aCalib->SetParamDist("b2",aB2);

      cIsometry3D<tREAL8> aPose(aCAbs,cRotation3D<tREAL8>(aRot,false));
      cSensorCamPC aCam("Camera_UncalibResection",aPose,aCalib);

      if (mGTCam) 
      {
          cWeightAv<tREAL8>  aAvgDiff;
          for (const auto & aPair : mSet.Pairs())
	  {
              double aDif =  Norm2(aPair.mP2+ mCentroid.mP2 - aCam.Ground2Image(aPair.mP3+mCentroid.mP3) );
              aAvgDiff.Add(aPair.mWeight,aDif);
	  }
	  double aAvR = aAvgDiff.Average() ;
          //  StdOut() << "AVVVV " << aAvgDiff.Average() << "\n";
          MMVII_INTERNAL_ASSERT_bench(aAvR < 1e-5,"Residual cam in cUncalibSpaceRessection");
          // StdOut()<< mGTCam->Center() << " " << aCam.Center() << "\n";
      }
      //delete aCalib;
}
/*
   (A  B C)  (x        A x/z + B y/z + C
   (0  D E)  (y   = 
   (0  0 1)  (z
 
   x = x+b1x+b2y  (1+b1    b2)
                  (        1)
 */



void OneBenchUnCalibResection(int aKTest)
{

     // StdOut() << "KKK=" << aKTest << "\n";

     cPt2di  aSz(2000+RandUnif_0_1()*2000,2000+RandUnif_0_1()*2000);
     tREAL8  aFoc = 1000 + RandUnif_0_1() * 10000;
     //cPt2dr  aPP(1250.0,1100.0);
     cPt2dr  aPP =  ToR(aSz/2) + cPt2dr::PRandC() * 4000.0;

     double  aB1 = -0.5 + RandUnif_0_1();  ///   0 <1+B1
     double  aB2 = RandUnif_C();
// aB1=aB2=0;



      cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::Alloc
                                  (
                                         cDataPerspCamIntrCalib
                                         (
                                               "Calib_BenchUncalibResection",
                                                eProjPC::eStenope,
                                                 cPt3di(0,0,1),
                                                std::vector<double>(),
                                                cCalibStenPerfect(aFoc,aPP),
                                                cDataPixelDomain(aSz),
                                                 cPt3di(0,0,1),
                                                10
                                         )
                                  );



      aCalib->SetParamDist("b1",aB1);
      aCalib->SetParamDist("b2",aB2);
      if (MMVII_IN_DEBUG)  // just in case we need to check params
      {
          for (const auto & aName : {"b1","b2","p1","toto"})
          {
              int aInd= aCalib->IndParamDistFromName(aName,true);
              StdOut()  <<  aName << " I=" << aInd ;
	      if (aInd>=0)
	      {
                  const auto & aDesc = aCalib->VDescDist().at(aInd);
                  StdOut()  << " LN=" << aDesc.mLongName  << " Deg=" << aDesc.mDegMon << " V=" <<aCalib->ParamDist(aName);
	      }
              StdOut()  << "\n";
          }
      }

      if (1) // (aKTest >= 40)
      {
          cSensorCamPC aCam("Camera_BenchUncalibResection",cIsometry3D<tREAL8>::RandomIsom3D(100.0),aCalib);

          std::vector<double> aVDepts({1,2});
          cSet2D3D  aSetCorresp  =  aCam.SyntheticsCorresp3D2D(10,aVDepts) ;

          cUncalibSpaceRessection<tREAL8>  aResec8(aSz,aSetCorresp,&aCam);
      }
      delete aCalib;
}

void BenchUnCalibResection()
{
    // Maybe because bad conditionning ? but inversion do not pass eigen test, 
    // BTW they pass "my" test on residual, so ....
    PushErrorEigenErrorLevel(eLevelCheck::Warning);

    for (int aK=0 ; aK<200 ; aK++)
    {
         OneBenchUnCalibResection(aK);
    }
    PopErrorEigenErrorLevel();
}




}; // MMVII

