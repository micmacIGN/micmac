#ifndef  _MMVII_Interpolators_H_
#define  _MMVII_Interpolators_H_

#include "MMVII_Geom2D.h"

namespace MMVII
{

class cInterpolator1D ;
class cDiffInterpolator1D ;
class cLinearInterpolator ;
class cCubicInterpolator ;
class cSinCApodInterpolator ;

class cEpsDiffFctr ;
class cTabulatedInterpolator ;
class cTabulatedDiffInterpolator ;
class cMMVII2Inperpol;

/*  ********************************* */
/*       Kernels                      */
/* ********************************** */

/// A kernel, approximating "gauss"

/**  a quick kernel, derivable, with support in [-1,1], coinciding with bicub in [-1,1] 
     not really gauss but has a "bell shape"
     1 +2X^3 -3X^2  , it's particular of bicub with Deriv(1) = 0
*/

/// If we dont need any kernel interface keep it simple 
tREAL8 CubAppGaussVal(const tREAL8&);

tREAL8  GaussLaw(const tREAL8& aVal,const tREAL8& aAvg,const tREAL8& aStdDev);

/** Class for doing "very fine" ressampling , high cost because compute a kernel for each pixel
 * typically for simulation with scale change, used for target simulation*/

class cRessampleWeigth
{
    public :
         static cRessampleWeigth  GaussBiCub(const cPt2dr & aPtsIn,const cAff2D_r & aMapIn, double aSzK);
    // private :
         std::vector<cPt2di>  mVPts;  // Pts used
         std::vector<double>  mVWeight;
};

/**  Virtual Base class for  all interpolator */

class cInterpolator1D : public cMemCheck
{
      public :
        /// Constructor, Weight is assumed to be 0 outside [-aSzK,+aSzK]
        cInterpolator1D(const tREAL8 & aSzKernel,const std::vector<std::string> & aVNames);
        ///  virtual destructor for a pure virtual class
        virtual ~cInterpolator1D();

        ///  fundamental method described the weight for each X, supposed to be even
        virtual tREAL8  Weight(tREAL8  anX) const = 0;
        const tREAL8 & SzKernel() const;  ///< accessor

        const std::vector<std::string> & VNames() const ; ///< Accessor

	/// create a tabulated interpol
        static cInterpolator1D *  TabulatedInterp(const cInterpolator1D &,int aNbTabul,bool BilinInterp);
      protected :
        tREAL8 mSzKernel;
        std::vector<std::string>   mVNames;
};

/** A diffenrantiable inteporlator indicate the derivate of weigthing function, it is used for
 * computing both the value and the derivates of the interpolation of an image */

class cDiffInterpolator1D : public cInterpolator1D
{
       public :

            /// constructor required for initializin base class
            cDiffInterpolator1D(tREAL8 aSzK,const std::vector<std::string>& aName);
            /// indicate dW/dx
            virtual tREAL8  DiffWeight(tREAL8  anX) const =0;
            /// Sometime its more optimized to compute both value simultaneously, default calls "Weight"&"DiffWeight"
            virtual std::pair<tREAL8,tREAL8>  WAndDiff(tREAL8  anX) const ;

            /// Allocator from a vector of names, usefull if interpolator is parametrized by command line
	    static cDiffInterpolator1D * AllocFromNames(const std::vector<std::string> & aVName);

	    // ---- For allocation inside a programm it may be more convenient to have type allocator
		      
	         /// create a tabulated interpolator from an existing (analytical)
             static cDiffInterpolator1D *  TabulatedInterp(const cInterpolator1D &,int aNbTabul);
	         /// create a tabulated interpolator from an existing (analytical) + delete the interp
             static cDiffInterpolator1D *  TabulatedInterp(cInterpolator1D *,int aNbTabul);

       protected :
	    static const std::string & Get(const std::vector<std::string> & aVName,size_t aK0);

	    static void AssertEndParse(const std::vector<std::string> & aVName,size_t aK0);
            /// Internal method , K0 will vary during parsing
	    static cDiffInterpolator1D * AllocFromNames(const std::vector<std::string> & aVName,size_t aK0);
};


/** Linear interpolator, not very usefull as  function GetVBL do the jobs
 * inline and faster, but used in unitary test
 */

class cLinearInterpolator : public cDiffInterpolator1D
{
      public :
        static const std::string TheNameInterpol;
        cLinearInterpolator();
        tREAL8  Weight(tREAL8  anX) const override ;
        /// Derivability can be discussed ...
        tREAL8  DiffWeight(tREAL8  anX) const override ;
};


/** Cubic  interpolator, we make it differentiable essentially for
 * unitary test, because pratically the tabulated is probably more efficient
 *
 *   mA = value of derivate in 1
 *   when mA=-0.5, interpolation of linear image is linear function
 *   when mA=0, kernel is [-1,1] on weight is >=0, its "CubAppGaussVal"
 *   for theoreticall reason (I forgot) we should have  0 >= mA >= -3
 */

class cCubicInterpolator : public cDiffInterpolator1D
{
        public :
             static const std::string TheNameInterpol;
             /// Contructor with value of derivate in 1
             cCubicInterpolator(tREAL8 aParam);
             /// Classical cubic weighting
             tREAL8  Weight(tREAL8  anX) const override ;
             /// Analyticall differential of weight
             tREAL8  DiffWeight(tREAL8  anX) const override ;
             /// Optimized simultaneous compute of Value & Derivate
             std::pair<tREAL8,tREAL8>   WAndDiff(tREAL8  anX) const override;
       private :
             tREAL8 mA;

};

/**  Apodized Sinus Cardinal Intepolator.  For time effciency we cannot use
 * the full "SinC" which has infinite support. So we use a troncated value;
 *  but we cannot have abrupt transition which would have undesirable effect.
 *
 *  So use a smooth transition (aka "apodisation") with two parameters :
 *
 *      - aSzSinc : size of window where we use full sinus cardinal
 *      - aSzAppod : size of window where smooth transition from 0 to 1 is used
 *
 *  Concretely "(5,5)" parametrization is a good  heuristic choice.
 *
 *  Implementation is very slow + it's not a partition of unity, so we highly recommand its
 *  use as tabulated interpolator.
 */

class cSinCApodInterpolator : public cDiffInterpolator1D
{
       public :
            static const std::string TheNameInterpol;
            cSinCApodInterpolator(tREAL8 aSzSinC,tREAL8 aSzAppod);
            tREAL8  Weight(tREAL8  anX) const override ;
            tREAL8  DiffWeight(tREAL8  anX) const override ;
       public :
            tREAL8 mSzSinC;
            tREAL8 mSzAppod;
};

/** See in the code (grep "cMMVII2Inperpol::cMMVII2Inperpol") the definition of this
 * interpolator */

class cMMVII2Inperpol : public cDiffInterpolator1D
{
        public :
             static const std::string TheNameInterpol;
             /// Contructor with value of derivate in 1
             cMMVII2Inperpol();
             /// Classical cubic weighting
             tREAL8  Weight(tREAL8  anX) const override ;
             /// Analyticall differential of weight
             tREAL8  DiffWeight(tREAL8  anX) const override ;
             /// Optimized simultaneous compute of Value & Derivate
       private :
};

/** generalisation of "cMMVII2Inperpol", where the spreading of the distribution is not variance but
 * sum(Avg-X)^Exp , cMMVIIKInterpol(2) is same as cMMVII2Inperpol. Implementation is very slow,
 * (solve a linear system for each value), so use as tabulation is highly recommande
 */

class cMMVIIKInterpol : public cDiffInterpolator1D
{
        public :
            ///
            static const std::string TheNameInterpol;
            cMMVIIKInterpol(tREAL8 aParam);

            tREAL8  Weight(tREAL8  anX) const override ;
             /// Analyticall differential of weight
             tREAL8  DiffWeight(tREAL8  anX) const override ;
        private :
            tREAL8 mExp;
};



/** Transformate an interpolator in a differentiable one using a
 * finite difference schema (with parameter "aEps").  Not very used in pratice because
 * "cTabulatedDiffInterpolator" directly does the "stuff".
 */

class cEpsDiffFctr : public cDiffInterpolator1D
{
      public :
         /// Constructor take the interpolator to make differentiable and "epsilon value" for finite difference
          cEpsDiffFctr(const cInterpolator1D & anInt,tREAL8 aEps) ;
          tREAL8  Weight(tREAL8  anX) const override ;
          tREAL8  DiffWeight(tREAL8  anX) const override;
      private :
           const cInterpolator1D & mInt;
           tREAL8    mEps;
};
/** The exact kernel value computation of a given interpolator  can be time consuming.
 *  For ex with sinC appodized of parameter (5,5), kernel is computed "441" time for each
 *  pixel it is used.
 *
 *  To accelerate this computation this class offer the possibiliby of "tabulating" the possible
 *  values, as kernel is "1 dim" a fin step can be used if necessary.  Also at slight cost it is
 *  also possible to use a linear interpolation for kernel computation.
 *
 *   A property of interpolator is that for any x in R we have ;
 *
 *      Sum{k in Z} W(x+k) = 1  (N1)
 *
 *   This is required, for example, for the interpolation of constant to be a constant. For the
 *   derivate "W'" of weighting we then have :
 *
 *      Sum{k in Z} W'(x+k) = 0 (N0)
 *
 *   As this properties are not necessarily true of all kernel (for example with apodized sinc),
 *   it can be enforced as a post-processing in  the method DoNomalise , depending from param
 *   "ForDeriv"
 *
 *      - if "false", then "N1" is enforced by dividing the W(x+k) by their sum
 *      - if "true",  then "N0" is enforced by substracting their average to the W'(x+k)
 *
 */

class cTabulatedInterpolator : public cInterpolator1D
{
      public :
          friend class cTabulatedDiffInterpolator;

          /// Constructor : Intepol 2 tabulate, Nb of tabul by  unity segment, do we interplate values
          cTabulatedInterpolator(const cInterpolator1D &,int aNbTabul,bool InterpolTab);
          /// Weight computed usinf tabulation
          tREAL8  Weight(tREAL8  anX) const override ;

      private :

          /// Constructo specifying the normalization
          cTabulatedInterpolator(const cInterpolator1D &,int aNbTabul,bool InterpolTab,bool DoNorm);
          /// activate the normaliation in sum 1 or sum 0 (for deriv)
          void DoNormalize(bool ForDeriv);
          /// compute the values as derivative from an existing tab (used by "cTabulatedDiffInterpolator")
          void SetDiff(const cTabulatedInterpolator & anInt);

          bool               mInterpolTab;  ///< Are value interpolated
          int                mNbTabul;      ///< number of value / unity
          int                mSzTot;        ///< Total number of values
          cIm1D<double>      mIm;           ///< Image tabulating
          cDataIm1D<double>* mDIm;          ///< Data image

};

/** A differenatiable version of tabuled interpolator.  Is implemented using
 * 2 cTabulatedInterpolator  ("mTabW" for values, "mTabDifW" for diff),
 *
 *  mTabDifW is computed from mTabW, so it not required the interpolator given
 *  to constructor be a "cDiffInterpolator1D"
 */

class cTabulatedDiffInterpolator : public cDiffInterpolator1D
{
      public :
          static const std::string TheNameInterpol;
          /// constructor : interpol 2 tabluate , nb of value / unity
          cTabulatedDiffInterpolator(const cInterpolator1D &,int aNbTabul=1000);
          ///  idem but delete the interpolator
          cTabulatedDiffInterpolator(cInterpolator1D *,int aNbTabul=1000);

          ///  Weight access to mTabW
          tREAL8  Weight(tREAL8  anX) const override ;
          ///  Differential of Weight access to mTabDifW
          tREAL8  DiffWeight(tREAL8  anX) const override;
          ///  Optmize version for simultaneous compute of weight and diff
          std::pair<tREAL8,tREAL8>   WAndDiff(tREAL8  anX) const override;
      private :
          cTabulatedInterpolator  mTabW;      ///< Tabulation of weighting
          cTabulatedInterpolator  mTabDifW;   ///< Tabulation of derivate of weighting
          int                     mNbTabul;   ///<  Nb Tabul/unity
          int                     mSzTot;     ///< Total nb tabul
          const tREAL8 *          mRawW;      ///< Raw data of values
          const tREAL8 *          mRawDifW;   ///< Raw data of derivates
};





};

#endif  //  _MMVII_Interpolators_H_
