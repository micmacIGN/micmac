#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
// #include "MMVII_nums.h"
// #include "MMVII_Geom3D.h"
// #include "cMMVII_Appli.h"


/**
   \file CalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in calibrated case

 */

namespace MMVII
{

template <class Type,const int Dim> class cAffineForm
{
    public :
       typedef cPtxd<Type,Dim>      tPt;

       Type Value(const tPt & aP) const {return Scal(mForm,aP) + mCste;}

       cAffineForm(const Type * aV);

       const tPt&  Form() {return mForm;}
       const Type& Cste() {return mCste;}
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
	       const cSet2D3D &,
	       bool  ForBench
	   );

       private :
	   static void SetVect(cDenseVect<Type> & aV,int anInd,const tPt3dr & aP,double aMul);

	   void CalcSolOneVarFixed(int aK);

	   void AddOneEquation(const cWeightedPair2D3D & aPair);

	   cLeasSqtAA<Type>  mSys0;
	   cSet2D3D          mSet;
	   cPair2D3D         mCentroid;
	   Type              mSumW;
	   cDenseVect<Type>  mVecW;
           cWhichMin<cHomog2D3D<Type>,Type>  mBestH;
	   bool                              mForBench;
};



template <class Type>  
    cUncalibSpaceRessection<Type>::cUncalibSpaceRessection(const cSet2D3D & aSet,bool forBench) :
        mSys0 (12),
	mSet  (aSet),
	mCentroid (mSet.Centroid()),
	mSumW (0),
        mVecW (12, eModeInitImage::eMIA_Null),
	mBestH (cHomog2D3D<Type>(),1e30),
	mForBench (forBench)
{
    mSet.Substract(mCentroid);
    for (const auto & aPair : mSet.Pairs())
        AddOneEquation(aPair);



    for (int aKV=0 ; aKV<12 ; aKV++)
    {
        CalcSolOneVarFixed(aKV);
    }
// StdOut()  <<  "SSsssssss " << mBestH.ValExtre() << "\n";
}

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

void BenchUnCalibResection()
{
     cPt2di  aSz(3000,3000);
     tREAL8  aFoc(4000);
     cPt2dr  aPP(1250.0,1100.0);
     double  aB1 = 0.3;
     double  aB2 = -0.2;



      cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::Alloc
                                  (
                                         cDataPerspCamIntrCalib
                                         (
                                               "Calib_BenchUncalibResection",
                                                eProjPC::eStenope,
                                                 cPt3di(3,1,1),
                                                std::vector<double>(),
                                                cCalibStenPerfect(aFoc,aPP),
                                                cDataPixelDomain(aSz),
                                                 cPt3di(3,1,1),
                                                100
                                         )
                                  );


      aCalib->SetParamDist("b1",aB1);
      aCalib->SetParamDist("b2",aB2);
      if (0)  // just in case we need to check params
      {
          for (const auto & aName : {"b1","b2","p1","toto"})
          {
              int aInd= aCalib->IndParamDistFromName(aName,true);
              StdOut()  <<  aName << " I=" << aInd ;
	      if (aInd>0)
	      {
                  const auto & aDesc = aCalib->VDescDist().at(aInd);
                  StdOut()  << " LN=" << aDesc.mLongName  << " Deg=" << aDesc.mDegMon << " V=" <<aCalib->ParamDist(aName);
	      }
              StdOut()  << "\n";
          }
      }

      cSensorCamPC aCam("Calib_BenchUncalibResection",cIsometry3D<tREAL8>::RandomIsom3D(100.0),aCalib);

      //std::vector<double> aVDepts({1,2,3});
      std::vector<double> aVDepts({1,2});
      cSet2D3D  aSetCorresp  =  aCam.SyntheticsCorresp3D2D(10,aVDepts) ;

      cUncalibSpaceRessection<tREAL4>  aResec4(aSetCorresp,true);
//StdOut() << "----------------------------------------\n\n";
      cUncalibSpaceRessection<tREAL8>  aResec8(aSetCorresp,true);
//StdOut() << "----------------------------------------\n\n";
      cUncalibSpaceRessection<tREAL16>  aResec16(aSetCorresp,true);


      delete aCalib;
}




}; // MMVII

