#ifndef  _MMVII_Linear2DFiltering_H_
#define  _MMVII_Linear2DFiltering_H_
namespace MMVII
{

/** \file  MMVII_Linear2DFiltering.h
    \brief Classes for linear filterings and its representation
*/


/**************************************/
/*           Gaussian  FILTERING      */
/**************************************/

/// return the variance of  exponential distribution of parameter "a" ( i.e proportiona to  "a^|x|")
double Sigma2FromFactExp(double a);
/// Inverse function, the usefull one 
double FactExpFromSigma2(double aS2);


/** General function, 
      Normalise => If true, limit size effect and Cste => Cste (else lowe close to border)
      cRect2 => foot print, not necessarily whole image
      Fx,Fy => factor in x and y can be different, (if one is equal 0 => speed up)
*/

template <class Type>
void  ExponentialFilter(bool Normalise,cDataIm2D<Type> & aIm,int   aNbIter,const cRect2 &,double Fx,double Fy);


/** Standard parameters, normalised, whole image, Fx==Fy */
template <class Type>
void  ExponentialFilter(cDataIm2D<Type> & aIm,int   aNbIter,double aFact);
/** More naturel parametrisation, specify the standard deviation of FINAL filter (including iterations),
   the NbIter parameter allow to be more or less close to a gaussian */
template <class Type>
void  ExpFilterOfStdDev(cDataIm2D<Type> & aIm,int   aNbIter,double aStdDev);


/**************************************/
/*           Gaussian  Pyramid        */
/**************************************/

template <class Type> class cGP_OneImage;
template <class Type> class cGP_OneOctave;
template <class Type> class cGaussianPyramid;
struct cParamGP;

/// Class to store one image of a gaussian pyram
/**
     This class contain a image which the result of gaussian
    convolution of original image. It contains also many links
    to other images, octaves and Pyrams it belongs to
*/

template <class Type> class cGP_OneImage : public cMemCheck
{
    public :
        typedef cGP_OneImage<Type>     tGPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef cGaussianPyramid<Type> tPyr;
        cGP_OneImage(tOct * anOct,int NumInOct,tGPIm * mUp);
    private :
        cGP_OneImage(const tGPIm &) = delete;
        tOct*         mOct;        ///< Octave it belongs to
        tPyr*         mPyr;        ///< Pyramid it belongs to
        int           mNumInOct;   ///< Number inside octave
        tGPIm *       mUp;         ///< Possible image up in the   octave
        tGPIm *       mDown;       ///< Possible image down in the octave
        cIm2D<Type>   mImG;        ///< Gaussian image
        double        mSigmAbs;    ///< Sigma of Gauss "Absolute" 
        double        mSigmInO;    ///< Sigma of Gauss  In octave
};

/// Class to store one octabe of a gaussian pyram
/**
     This class contain an octave, it contains essentially a vector of all image
     having the same size, for practicall reason there will
     currently be an overlap between consecutive octaves
*/
template <class Type> class cGP_OneOctave : public cMemCheck
{
    public :
        typedef cGP_OneImage<Type>     tGPIm;
        typedef std::shared_ptr<tGPIm> tSP_GPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef cGaussianPyramid<Type> tPyr;

        cGP_OneOctave(tPyr * aPyr,int aNum,tOct * aUp);
        //  ====  Accessors  ===========
        tPyr*          Pyram() const ;    ///< Accessor to Pyram
        const cPt2di &  SzIm() const;     ///<  mSzIm
        const double &  Sigm0Abs() const; ///< mSigm0Abs
    private :
        cGP_OneOctave(const tOct &) = delete;
        tPyr*              mPyram;        ///< Pyramid it belongs to
        tOct *             mUp;           ///< Possible octave up in the pyramid
        tOct *             mDown;         ///< Possible octave down in the pyramid
        int                mNumInPyr;     ///< Number inside Pyram
        cPt2di             mSzIm;         ///< Size of images in this octave
        double             mSigm0Abs;     ///< Sigma Abs of first image in octave
        std::vector<std::shared_ptr<tGPIm>> mVIms;       ///< Images of the Pyramid
};


/// Struct for parametrization of Gaussian Pyramid

/**  As  cGaussianPyramid may evolve with many parameters
     it's a "good" precaution to use a struct of parametrisation
*/

struct cGP_Params
{
     public :
         cGP_Params(const cPt2di & aSzIm0,int aNbOct,int aNbLevByOct,int aOverlap);

         const cPt2di mSzIm0;    ///< Sz of Image at full resol
         const int  mNbOct;      ///< Number of octave
         const int  mNbLevByOct;  ///< Number of level per octave (dont include overlap)
         const int  mNbOverlap;  ///< Number of overlap
};

/// Struct to store a gaussian pyram

template <class Type> class  cGaussianPyramid : public cMemCheck
{
    public :
      // Typedef section 
        typedef cGP_OneImage<Type>     tGPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef std::shared_ptr<tOct>  tSP_Oct;
        typedef cGaussianPyramid<Type> tPyr;

        cGaussianPyramid(const cGP_Params &);
      // Accessors
        const cGP_Params & Params() const;  ///< Parameters of pyramid
        const double & MulSigm() const;    ///< Multiplier sigm conseq
        const double & Sigma0() const;     ///< mSigma0
        int  NbImByOct() const;            ///< mNbImByOct + mNbOverlap
        const cPt2di & SzIm0() const;      ///< Size of most resolved images
    private :
        cGaussianPyramid(const tPyr &) = delete;
        cGP_Params          mParams;   ///< Memorize parameters
        std::vector<tSP_Oct>   mVOct;  ///< Vector of octaves
        double   mMulSigm;  ///<  Sigma of gaussian multiplier between two consecutive gaussian
        double   mSigma0;    ///< Sigma of first image, conventionnaly by default
};

};





#endif  //   _MMVII_Linear2DFiltering_H_
