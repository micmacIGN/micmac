#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_BundleAdj.h"


// #include "MMVII_nums.h"
// #include "MMVII_Geom3D.h"
// #include "cMMVII_Appli.h"

//  Test git

/**
   \file UnCalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in uncalibrated case

 */

/* We have the image formula w/o distorsion:

   (u v)  = PI0   R (P-C0)
   Or
   (u v 1) = Lambda  PI0   R (P-C0)

   we consider a  calibraytion with linear distorsion :

   (I)    (PPx + F (u  p1 u + p2 v))     (F(1+p1)   p2F  PPx) (u)     (a b c) (u)      (u) [EqCal]
   (J) ~  (PPy + F v               )  =  (0         F    PPy) (v)  =  (0 e f) (v) =  C (v)
   (1)    (                       1)     (0         0     1)  (1)     (0 0 1) (1)      (1)

   (I)                                           (M00 M10 M20) (X) + (xm0)
   (J)  ~  C R (P-C0)  = M (P-C0) =  MP -MC0 =   (M01 M11 M21) (Y) + (ym0)  = M +Tr
   (1)                                           (M02 M12 M22) (Z) + (zm0)

   I =  (M00 X + M10 Y + M20 Z + xm0) / (M02 X + M12 Y + M22 Z + zm0)   [EqHom]
   J =  (M01 X + M11 Y + M21 Z + xm0) / (M02 X + M12 Y + M22 Z + zm0)


   0  =  (M00 X + M10 Y + M20 Z + xm0) - I (M02 X + M12 Y + M22 Z + zm0)  [XEqL]  | [EqLin]  
   0  =  (M01 X + M11 Y + M21 Z + xm0) - J (M02 X + M12 Y + M22 Z + zm0)  [YEqL]  |


   In [EqLin]  we know X,Y,Z,I,F for each projection. We have a system of linear equation in M00,... M22, xm0, ym0 , zm0


   We want to solve  [EqLin] by least square, but if do it directly we will get the null solution. So we
   have to add an arbitray constraint as M11=1  or xm0=1..  but which one ? If we select one for which 0 is the
   "natural" solution we will  not add any constraint and will get again the null vector, and even if not exactly void,
   we can get a very noisi solution ...

   => to turn over, we test all posible constraint (try M00=1 , then M11=1 ...), and select the solution that
   give the best residual.  Note that this is fast as the normal matrix & vector is computed only once.

   Once we have estimate M00,...,xm0 ...  we must recover the physicall parameters.  
   
   
   For the center its easy, we have Tr = -MC0 and then  :
   
                    C0 = - M-1 Tr     [EqCenter]


   For internal calib & orientaion, it's theoretically easy ..., but practically a bit tricky.
   Knowing M, we can extract R & C, using a RQ decomposition : ie write M = R * Q where Q
   id orthogonal and R is triangular sup, however :

      *  eigen provide QR and not RQ, so there is a class that make basic permuation
    
      *  RQ decomposition is ambiguous on sign, let S be any sign-matrix (diag with +-1) we 
         have  RQ = RSSQ = (RS) (SQ) , and stil RS is up-triangular ans SQ orthogobal,
	 in our implemtatuion we fix the ambiguity by imposing that all diag elem of R
	 are >=0

     * M and Tr are  defined up to a scale factor, so are R and Q :

         - for R  we solve that by imposing M22=1 (ie divide initial M by M22)

	 - for Q , is we multiply Q by -1, its still orthogonal, we solve that by
	   multiply Q by -1 if its initial determinant is <0

         - for C0 it's not a problem as in [EqCenter] , the arbitrary scale of M and T is naturally
           absorbed;

*/

namespace MMVII
{


template <class Type> class cHomog2D3D; // class for homography 3D->2D as in [EqHom]
template <class Type,const int Dim> class cAffineForm;  // H3D2 are made of 3 Affine forms 
						
template <class Type>  class cUncalibSpaceRessection; // class for computing the uncalibrated space resection




/** Helper for cHomog2D3D, represent an affine form R3->R */
template <class Type,const int Dim> class cAffineForm
{
    public :
       typedef cPtxd<Type,Dim>      tPt;

       /// compute value of the function
       Type Value(const tPt & aP) const {return Scal(mLinear,aP) + mCste;}

       /// constructor from 
       cAffineForm(const Type * aV);

       const tPt&  Linear() const {return mLinear;} /// Accessor
       const Type& Cste() const {return mCste;} ///< Accessor
    private :
       tPt  mLinear; ///< Linear part, a point (by "duality")
       Type mCste;   ///< constant part, a scalar
};



/**  Class for represnting a 3D->2D homography */

template <class Type> class cHomog2D3D
{
    public :
       static constexpr int       TheDim=3;

       typedef cPtxd<Type,3>      tPtIn;
       typedef cPtxd<Type,2>      tPtOut;

       /// Compute value of the function
       tPtOut  Value(const tPtIn & aP)  const  
       {
	       return tPtOut(mFx.Value(aP),mFy.Value(aP)) / mFz.Value(aP);
       }

       /// Adaptor to have with point operating on REAL8
       tPt2dr  RValue(const tPt3dr & aP)const  {return ToR(Value(tPtIn::FromPtR(aP)));}

       /// Constructor from a raw data, used for creating from least sq sol
       cHomog2D3D(const Type *);
       /// Default constuctor required by WhichMin
       cHomog2D3D();

       /// extact matrix from the homography
       cDenseMatrix<Type>  Mat() const;
       /// extract translation from the homography
       tPtIn               Tr()  const;

       const cAffineForm<Type,3> & Fx() const;  ///< Accessor
       const cAffineForm<Type,3> & Fy() const;  ///< Accessor
       const cAffineForm<Type,3> & Fz() const;  ///< Accessor

    private :

       cAffineForm<Type,3>  mFx;  ///< X composant
       cAffineForm<Type,3>  mFy;  ///< Y composant
       cAffineForm<Type,3>  mFz;  ///< Z composant

};

/**
 *
 * Class for solving the "11 parameter" equation, AKA uncalibrated resection
 */
template <class Type>  class cUncalibSpaceRessection
{
      public :
           static constexpr int TheNbVar = 12;

           cUncalibSpaceRessection
           (
	       const cPt2di & aSz,           // sz of the camera to generate at end
	       const cSet2D3D & aSetProj,    // set of corresponsdance used for estimation
	       const cSensorCamPC * aGTCam = nullptr // ground truth in bench mode
	   );
	   ///  Compute Parameters
	   cSensorCamPC *  ComputeParameters(const std::string & aNameCam);

       private :

	   /// Compute least square system w/o any constraint on sols (dont try to solve it)
	   void  CalcLeastSquare_WOConstr();
           /// Add on equation corresponding to one correspondance
	   void AddOneEquation(const cWeightedPair2D3D & aPair);
           /// Put in dense vect [XEqL] or [YEqL], Mul can be 1,I or J
	   static void SetOneAffineForm(cDenseVect<Type> & aV,int anInd,const tPt3dr & aP,double aMul);

	   /// Test all constraint on all possible variable and store the best (call "CalcSolOneVarFixed")
	   void  Test_WithAllConstr();
           /// Add one constraint then solve least square
	   void CalcSolOneVarFixed(int aK);

                //==============  DATA ==================

                      //  --  Copy of constructor param -------------

           cPt2di            mSz;  ///< Size of cam to generate
	   cSet2D3D          mSet;  ///< Set of correspondances
	   const cSensorCamPC * mGTCam;  ///< Ground truth cam

                      //  --  Internal param -------------

	   cLeasSqtAA<Type>  mSys0;      ///< Least Sq Syst for equation [EqLin]
	   cPair2D3D         mCentroid;  ///< Centroid, for 2D/3D correspond
	   Type              mSumW;      ///< Sum of weight  of all obs
	   cDenseVect<Type>  mVecW;      ///< Weighted sum of  square coeff
           cWhichMin<cHomog2D3D<Type>,Type>  mBestH;  ///< memorize best homography of all possible var
};

/* ************************************************* */
/*                                                   */
/*                     MMVII                         */
/*                                                   */
/* ************************************************* */

/// compute average reproj of a map
template <class TMap>  tREAL8  AvgReProj(const cSet2D3D & aSet,const TMap& aMap)
{
    cWeightAv<tREAL8>  aWAvg;

    for (const auto & aPair : aSet.Pairs())
    {
        aWAvg.Add(aPair.mWeight,SqN2(aPair.mP2-aMap.RValue(aPair.mP3)));
    }

    return aWAvg.Average();
}



/* ************************************************* */
/*                                                   */
/*         cAffineForm                               */
/*                                                   */
/* ************************************************* */

template <class Type,const int Dim> 
   cAffineForm<Type,Dim>::cAffineForm(const Type * aV) :
       mLinear   (tPt(aV)),
       mCste   (aV[Dim])
{
}

/* ************************************************* */
/*                                                   */
/*         cHomog2D3D                                */
/*                                                   */
/* ************************************************* */


// A bit trick, but default Cstr of  cWitchMin calls cHomog2D3D with 0 as param, 
// which is interpreted as nullptr, so make it work with nullptr

template <class Type> 
   cHomog2D3D<Type>::cHomog2D3D(const Type * aV) :
	mFx  ( aV ? (aV+0) : std::vector<Type>({1,0,0,0}).data()),
	mFy  ( aV ? (aV+4) : std::vector<Type>({0,1,0,0}).data()),
	mFz  ( aV ? (aV+8) : std::vector<Type>({0,0,0,1}).data())
{
}

// calls nullptr => generate identity

template <class Type> cHomog2D3D<Type>::cHomog2D3D() :
	cHomog2D3D(nullptr)
{
}

     /// extract linear part to form a matrix
template <class Type>  cDenseMatrix<Type> cHomog2D3D<Type>::Mat() const
{
     return  M3x3FromLines(mFx.Linear(),mFy.Linear(),mFz.Linear());
}

     /// extract constant part to form the translation
template <class Type> cPtxd<Type,3> cHomog2D3D<Type>::Tr() const
{
    return cPtxd<Type,3> (mFx.Cste(),mFy.Cste(),mFz.Cste());
}
template <class Type>   const cAffineForm<Type,3> &  cHomog2D3D<Type>::Fx() const {return mFx;}
template <class Type>   const cAffineForm<Type,3> &  cHomog2D3D<Type>::Fy() const {return mFy;}
template <class Type>   const cAffineForm<Type,3> &  cHomog2D3D<Type>::Fz() const {return mFz;}


/* ************************************************* */
/*                                                   */
/*         cUncalibSpaceRessection                   */
/*                                                   */
/* ************************************************* */


template <class Type>  
    cUncalibSpaceRessection<Type>::cUncalibSpaceRessection(const cPt2di& aSz,const cSet2D3D & aSet,const cSensorCamPC * aGTCam) :
        mSz   (aSz),
	mSet  (aSet),
	mGTCam (aGTCam),

        mSys0 (TheNbVar),
	mCentroid (mSet.Centroid()),
	mSumW (0),
        mVecW (TheNbVar, eModeInitImage::eMIA_Null)
{
    // substract centroid to make more stable system with "big" coords
    mSet.Substract(mCentroid);

    //  LSQ W/o  any constraint
    CalcLeastSquare_WOConstr();

    //  Test all possible constraint to make the system well defined
    Test_WithAllConstr();

}

/*   ========================================
 *   Methods for computing Var/Cov
 *   ======================================== */

//   Implemenr EqLin : for example
//     (v0 x + v1 y + v2 z + v3)  =>   Ind=0, Mul =1
//    - J *  (v8 x + v9 y + v10 z + v11)  => Ind=8  Mul=-J
template <class Type>  
    void cUncalibSpaceRessection<Type>::SetOneAffineForm
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



template <class Type>  void  cUncalibSpaceRessection<Type>::AddOneEquation(const cWeightedPair2D3D & aPair)
{
    // IsX=true => XEqL   , IsX=false => YEqL
    for (const auto & IsX : {true,false})
    {
       // Extract Params
       double aW = aPair.mWeight;
       cPt3dr aP3 = aPair.mP3 ;

       // Put Equations in vect
       cDenseVect<Type>  aVect (TheNbVar, eModeInitImage::eMIA_Null);
       SetOneAffineForm( aVect , (IsX ? 0 : 4) , aP3 ,  1.0                                   );
       SetOneAffineForm( aVect ,             8 , aP3 , - (IsX ?aPair.mP2.x() : aPair.mP2.y()) );
       mSys0.AddObservation(aW,aVect,0.0);

       // update weigthings
       mSumW += aW;
       for (int aK=0 ; aK<TheNbVar  ; aK++)
           mVecW(aK) += aW * Square(aVect(aK));
    }
}
template <class Type>  void    cUncalibSpaceRessection<Type>::CalcLeastSquare_WOConstr()
{
    // Just add equations for all pairs
    for (const auto & aPair : mSet.Pairs())
        AddOneEquation(aPair);

}

/*   ========================================
 *   Methods for computing sols of homographie
 *   ======================================== */

template <class Type>  void cUncalibSpaceRessection<Type>::CalcSolOneVarFixed(int aKV)
{
     cLeasSqtAA<Type>  aSys = mSys0.Dup(); // Make a duplication of the system without constraints

     double aW =  std::sqrt(mVecW(aKV)/mSumW);  // weighted average of square coeff
     aW *= mSumW ;  // now weithed summ
     aSys.AddObsFixVar(aW,aKV,1.0);  // Add a constraint

     cDenseVect<Type>  aSol = aSys.Solve();  // extract the least square sol with constraint
     cHomog2D3D<Type>  aHom(aSol.RawData()); // make an homography of the sol

     tREAL8 aScore = AvgReProj(mSet,aHom); // compute the residual
     mBestH.Add(aHom, aScore);  // update with a possibly better solution
}

template <class Type>  void    cUncalibSpaceRessection<Type>::Test_WithAllConstr()
{
    // Test the contsrained solution with all possible variable
    for (int aKV=0 ; aKV<TheNbVar ; aKV++)
    {
        CalcSolOneVarFixed(aKV);
    }

    // If ground truth exist, make a firt check on linear solution
    if (mGTCam) 
    {
       double aResidual =  mBestH.ValExtre();
       // Low accurracy required, experimentally tREAL4 has numerical problem
       MMVII_INTERNAL_ASSERT_bench
       (
             aResidual < tElemNumTrait<Type>::Accuracy()*1e-2,
             "Residual homogr in cUncalibSpaceRessection"
       );
    }
}

/*   ===================================================
 *   Extract the "physicall" parameters from homography
 *   =================================================== */

template <class Type>  
   cSensorCamPC *    cUncalibSpaceRessection<Type>::ComputeParameters(const std::string & aNameIm)
{
    cDenseMatrix<Type> aMat = mBestH.IndexExtre().Mat(); // Matrix
    cPtxd<Type,3>      aTr  = mBestH.IndexExtre().Tr();  // Translation

    cPt3dr      aCLoc   =  ToR(SolveCol(aMat,aTr)) * -1.0 ;  // implementation of [EqCenter]
    cPt3dr      aCAbs   =  aCLoc+ mCentroid.mP3;  // invert initial centering

    // optional test on center accuracy, 
    if (mGTCam)
    {
       double aD =  Norm2(mGTCam->Center() - aCAbs) ;
       if (0) // (aD >= 1e-2)
       {
            StdOut() <<  "DIST CENTERS= " << aD << std::endl;
            MMVII_INTERNAL_ASSERT_bench(aD < tElemNumTrait<Type>::Accuracy()*1e-2,"Center in cUncalibSpaceRessection");
       }
    }
    
    //  make a decompositio  M = R Q with, R Triangular sup, Q orthogonal
    cResulRQ_Decomp<tREAL8>  aRQ = Convert((tREAL8*)nullptr,aMat).RQ_Decomposition();

    cDenseMatrix<tREAL8>  aRot = aRQ.Q_Matrix().Transpose(); // transpose to change W->C into Cam->Word for pose

    //  matrix are udefined up to scale, there is a sign ambiguity on aRot
    aRot.SetDirectBySign();

    // matrix beign defined up to a scale, fix R(2,2) = 1 
    aRQ.R_Matrix().DIm() *=  (1.0/aRQ.R_Matrix().GetElem(2,2));

    // Extract physicall internal parameters usign [EqCal]
    const cDenseMatrix<tREAL8> & aR = aRQ.R_Matrix();
    double aF = aR.GetElem(1,1);
    double  aB1 = (aR.GetElem(0,0)- aF)/aF;
    double  aB2 = aR.GetElem(1,0)/aF;
    cPt2dr  aPP =  mCentroid.mP2 + cPt2dr(aR.GetElem(2,0),aR.GetElem(2,1)) ; // Invert initial centering


    // Create camera with  2 linear parameters, initially no dist
    cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::Alloc
                                 (
                                         cDataPerspCamIntrCalib
                                         (
                                                cPerspCamIntrCalib::PrefixName()  + aNameIm,
                                                eProjPC::eStenope,
                                                cPt3di(0,0,1),
                                                std::vector<double>(),
                                                cMapPProj2Im(aF,aPP),
                                                cDataPixelDomain(mSz),
                                                cPt3di(0,0,1),
                                                10
                                         )
                                 );

      cMMVII_Appli::AddObj2DelAtEnd(aCalib); // Not sure of this

      // Now fix distorsion
      aCalib->SetParamDist("b1",aB1);  
      aCalib->SetParamDist("b2",aB2);

      // Compute pose & finally the camera
      cIsometry3D<tREAL8> aPose(aCAbs,cRotation3D<tREAL8>(aRot,false)); 
      cSensorCamPC* aCam = new cSensorCamPC(aNameIm,aPose,aCalib);

      // If grond truth camera, check accuracy
      if (mGTCam) 
      {
          cWeightAv<tREAL8>  aAvgDiff;
          for (const auto & aPair : mSet.Pairs())
	  {
              double aDif =  Norm2(aPair.mP2+ mCentroid.mP2 - aCam->Ground2Image(aPair.mP3+mCentroid.mP3) );
              aAvgDiff.Add(aPair.mWeight,aDif);
	  }
	  double aAvR = aAvgDiff.Average() ;
          //  StdOut() << "AVVVV " << aAvgDiff.Average() << std::endl;
          MMVII_INTERNAL_ASSERT_bench(aAvR < 1e-5,"Residual cam in cUncalibSpaceRessection");
          // StdOut()<< mGTCam->Center() << " " << aCam.Center() << std::endl;
      }
      return aCam;
}

/* ************************************************* */
/*                                                   */
/*                     cSensorCamPC                  */
/*                                                   */
/* ************************************************* */

cSensorCamPC * 
    cSensorCamPC::CreateUCSR
    (
         const cSet2D3D& aSetCorresp,
         const cPt2di & aSzCam,
         const std::string & aNameIm,
         bool Real16
    )
{
   cSensorCamPC * aCamCalc = nullptr;
   if (Real16)
   {
       cUncalibSpaceRessection<tREAL16>  aResec(aSzCam,aSetCorresp);
       aCamCalc = aResec.ComputeParameters(aNameIm);
   }
   else
   {
       cUncalibSpaceRessection<tREAL8>  aResec(aSzCam,aSetCorresp);
       aCamCalc = aResec.ComputeParameters(aNameIm);
   }

   return aCamCalc;
}


/* ************************************************* */
/*                                                   */
/*                     ::MMVII                       */
/*                                                   */
/* ************************************************* */


void OneBenchUnCalibResection(int aKTest)
{

     // StdOut() << "KKK=" << aKTest << std::endl;

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
                                                cMapPProj2Im(aFoc,aPP),
                                                cDataPixelDomain(aSz),
                                                 cPt3di(0,0,1),
                                                10
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
	      if (aInd>=0)
	      {
                  const auto & aDesc = aCalib->VDescDist().at(aInd);
                  StdOut()  << " LN=" << aDesc.mLongName  << " Deg=" << aDesc.mDegMon << " V=" <<aCalib->ParamDist(aName);
	      }
              StdOut()  << std::endl;
          }
      }

      if (1) // (aKTest >= 40)
      {
          cSensorCamPC aCam("Camera_BenchUncalibResection",cIsometry3D<tREAL8>::RandomIsom3D(100.0),aCalib);

          std::vector<double> aVDepts({1,2});
          cSet2D3D  aSetCorresp  =  aCam.SyntheticsCorresp3D2D(10,aVDepts) ;

          cUncalibSpaceRessection<tREAL8>  aResec8(aSz,aSetCorresp,&aCam);
          cSensorCamPC * aCamCalc = aResec8.ComputeParameters("NoIm_UCSR");
          delete aCamCalc;
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

/* ==================================================== */
/*                                                      */
/*          cAppli_UncalibSpaceResection                */
/*                                                      */
/* ==================================================== */

class cAppli_UncalibSpaceResection : public cMMVII_Appli
{
     public :
        typedef std::vector<cPerspCamIntrCalib *> tVCal;

        cAppli_UncalibSpaceResection(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
	int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
	///  compute a model of calibration different from linear one (more or less parameter)
        cSensorCamPC * ChgModel(cSensorCamPC * aCam);

	/// In case multiple pose for same camera try a robust compromise for each value
        void  DoMedianCalib();

	std::string              mSpecImIn;   ///  Pattern of xml file
	cPhotogrammetricProject  mPhProj;
        cSet2D3D                 mSet23 ;
	bool                     mShow;
	bool                     mReal16;
	cPt3di                   mDegDist;
        std::string              mPatParFrozen;
        cPt2dr                   mValFixPP;
	bool                     mDoMedianCalib;
};

cAppli_UncalibSpaceResection::cAppli_UncalibSpaceResection(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec):
	cMMVII_Appli   (aVArgs,aSpec),
        mPhProj        (*this),
        mShow          (false),
	mReal16        (false),
	mDoMedianCalib (true)
{
}

cCollecSpecArg2007 & cAppli_UncalibSpaceResection::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_UncalibSpaceResection::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return anArgOpt
	       << AOpt2007(mDegDist,"DegDist","Degree of distorsion, if model different of linear one")
	       << AOpt2007(mShow,"ShowNP","Show possible names of param for distorsion",{eTA2007::Tuning,eTA2007::HDV})
	       << AOpt2007(mPatParFrozen,"PatFrozen","Pattern for frozen parameters",{eTA2007::PatParamCalib})
	       << AOpt2007(mValFixPP,"ValPPRel","Fix value of PP in relative to image size ([0.5,0.5] for middle)")
	       << AOpt2007(mDoMedianCalib,"DoMedianCalib","Export for a median calib for multiple images",{eTA2007::HDV})
           ;
}

cSensorCamPC * cAppli_UncalibSpaceResection::ChgModel(cSensorCamPC * aCam0)
{
    tREAL8 aR0 =  aCam0->AvgSqResidual(mSet23);
    cPerspCamIntrCalib * aCal0 = aCam0->InternalCalib();

    bool  PPFrozen=false;
    cPt2dr aPP = aCal0->PP();
    cPt2di aSzPix = aCal0->SzPix();

    if (IsInit(&mValFixPP))
    {
        aPP = MulCByC(mValFixPP,ToR(aSzPix));
        PPFrozen=true;
    }

    
    // Create a calibration with adequate degree, same paramater as is init, except dist=0
    cDataPerspCamIntrCalib  aData
                            (
                                   aCal0->Name(),
                                   eProjPC::eStenope,
                                   mDegDist,
                                   std::vector<double>(),
                                   cMapPProj2Im(aCal0->F(),aPP),
                                   cDataPixelDomain(aSzPix),
                                   mDegDist,
                                   10
			    );

     cPerspCamIntrCalib * aCal1 = new cPerspCamIntrCalib(aData);

     cMMVII_Appli::AddObj2DelAtEnd(aCal1); // Not sure of this

     cSensorCamPC * aCam1 = new cSensorCamPC(aCam0->NameImage(),aCam0->Pose(),aCal1);
     delete aCam0;

     if (mShow)
     {
         cGetAdrInfoParam<tREAL8>::ShowAllParam(*aCam1);
     }

     tREAL8 aR1Init = aCam1->AvgSqResidual(mSet23);
     cCorresp32_BA  aBA(aCam1,mSet23);


     if (IsInit(&mPatParFrozen))
     {
        aBA.SetFrozenVarOfPattern(mPatParFrozen);
     }
     if (PPFrozen)
     {
        aBA.SetFrozenVarOfPattern("PP.*");
     }


     for (int aK=0 ; aK<10 ; aK++)
     {
         aBA.OneIteration();
     }
     tREAL8 aR1Final = aCam1->AvgSqResidual(mSet23);

     if (mShow)
     {
        StdOut() << "RESIDUAL, R0=" << aR0 << " R1Init=" << aR1Init << " R1Final=" << aR1Final << std::endl;
     }
     return aCam1;
}

void cAppli_UncalibSpaceResection::DoMedianCalib()
{
     // [1]   Extract all the calibration, group the one having same NameCalib
     std::map<std::string,tVCal> aMapCal;
     for (const auto &  aNameIm : VectMainSet(0))
     {
         std::string aNameCal = mPhProj.DPOrient().FullDirOut() + cPerspCamIntrCalib::PrefixName()  + aNameIm  + "." +TaggedNameDefSerial();
         if (ExistFile(aNameCal))
         {
             cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::FromFile(aNameCal);

	     cMetaDataImage  aMDI = mPhProj.GetMetaData(DirProject()+aNameIm);
	     aMapCal[aMDI.InternalCalibGeomIdent()].push_back(aCalib);
	     StdOut() << "NIIII  " << aNameIm << " F=" << aCalib->F()   << std::endl;
         }
         else
         {
             // No reason dont exit
             MMVII_UsersErrror(eTyUEr::eOpenFile,"No calib file found");
         }
     }

     for (const auto & aNameCal : aMapCal)
     {
          // [2]  Extract a vector that for each param contains a vector of all its values in different calib
          std::string  aName = aNameCal.first;
	  const tVCal &  aVCal = aNameCal.second;
          cPerspCamIntrCalib & aCal0 = *(aVCal.at(0));
          cGetAdrInfoParam<tREAL8>  aGAIP0(".*",aCal0); // Structure for extract param by names, all here
          size_t aNbParam = aGAIP0.VAdrs().size();

          std::vector<std::vector<double> > aVVParam(aNbParam); // will store all the value of a given param
          for (const auto & aPCal : aVCal)
          {
                cGetAdrInfoParam<tREAL8>  aGAIPK(".*",*(aPCal));
                for (size_t aKP=0 ; aKP<aNbParam ; aKP++)
                {
                     aVVParam.at(aKP).push_back(*aGAIPK.VAdrs().at(aKP));
                }
          }

     //std::vector<cPerspCamIntrCalib *> aVCal;

          StdOut() << " ####  " <<  aName   << " ####" << std::endl;
          for (size_t aKP=0 ; aKP< aNbParam ; aKP++)
          {
               StdOut() << " " <<  aGAIP0.VNames()[aKP] ;
	       tREAL8 aVMed = NonConstMediane(aVVParam.at(aKP));
	       tREAL8 aV20 = NC_KthVal(aVVParam.at(aKP),0.2);
	       tREAL8 aV80 = NC_KthVal(aVVParam.at(aKP),0.8);
               StdOut() <<  ": V=" << aVMed;
               StdOut() <<  ": DISP=" << (aV80-aV20);
               StdOut() <<  std::endl;

	       *(aGAIP0.VAdrs()[aKP]) = aVMed;
          }

          aCal0.SetName(aName);
          mPhProj.SaveCalibPC(aCal0);
     }
}

int cAppli_UncalibSpaceResection::Exe()
{
    mPhProj.FinishInit();

    if (RunMultiSet(0,0))  
    {
        int aResult = ResultMultiSet();

	if (aResult != EXIT_SUCCESS)
           return aResult;

	if (mDoMedianCalib)
	{
            DoMedianCalib();
	}
        mPhProj.CpSysIn2Out(false,true);

        return EXIT_SUCCESS;
    }

    std::string aNameIm =FileOfPath(mSpecImIn);

    mSet23 =mPhProj.LoadSet32(aNameIm);

    StdOut() <<  "Nb Measures=" << mSet23.NbPair() << std::endl;


    cPt2di aSz =  cDataFileIm2D::Create(aNameIm,false).Sz();
    cSensorCamPC *  aCam0 =  cSensorCamPC::CreateUCSR(mSet23,aSz,aNameIm,mReal16);

     
    if (IsInit(&mDegDist))
    {
       aCam0 = ChgModel(aCam0);
    }

    mPhProj.SaveCamPC(*aCam0);

    delete aCam0;
    return EXIT_SUCCESS;
};

/* ==================================================== */
/*                                                      */
/*                                                      */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_UncalibSpaceResection(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_UncalibSpaceResection(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriUncalibSpaceResection
(
     "OriPoseEstim11P",
      Alloc_UncalibSpaceResection,
      "Pose estimation from GCP, uncalibrated case",
      {eApF::Ori},
      {eApDT::GCP},
      {eApDT::Orient},
      __FILE__
);



}; // MMVII




