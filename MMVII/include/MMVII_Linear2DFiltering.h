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

/// Sigma of convol sqrt(A^2+B^2)
double SomSigm(double  aS1,double a2);
/// "Inverse" of SomSigm  sqrt(A^2-B^2)
double DifSigm(double  aS1,double a2);


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
template <class Type>
void  ExpFilterOfStdDev(cDataIm2D<Type> & aIOut,const cDataIm2D<Type> & aImIn,int aNbIter,double aStdDev);


/**************************************/
/*           Gaussian  Pyramid        */
/**************************************/

template <class Type> class cGP_OneImage;
template <class Type> class cGP_OneOctave;
template <class Type> class cGaussianPyramid;
struct cGP_Params;

/// Class to store one image of a gaussian pyram
/**
     This class contain a image which the result of gaussian
    convolution of original image. It contains also many links
    to other images, octaves and Pyrams it belongs to
*/

template <class Type> class cGP_OneImage : public cMemCheck
{
    public :
        typedef cIm2D<Type>            tIm;
        typedef cDataIm2D<Type>        tDIm;
        typedef cGP_OneImage<Type>     tGPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef cGaussianPyramid<Type> tPyr;

        cGP_OneImage(tOct * anOct,int NumInOct,tGPIm * mUp); ///< Constructor

              // =======   Image processing for creation
        void  ComputGaussianFilter();  ///< Generate computation of gauss image
        void  MakeDiff(const tGPIm & ); ///< Put in this the difference between anIm and anIm.mDown
        void  MakeCorner(); ///< Compute an indice of corner image 
        void  MakeOrigNorm(const tGPIm & ); ///< Create an image, almost orig, but normalized

              // =======   Description
        void SaveInFile() const;  ///< Save image on file, tuning/teaching
        void Show() const;  ///< Show Image in text format, test and debug
 
              // =======   Utilitaries
        std::string  ShortId() const;  ///< Helper to create Id avoid 
        std::string  Id() const; ///< Some identifier, may usefull in debuging
        bool   IsTopOct() const;   ///< Is it top image in its octave
        cPt2dr Im2File(const cPt2dr &) const; ///< To geomtry of global file

              // =======   Accessors
        tIm ImG();  ///< Get gaussian image
        bool   IsTopPyr() const;   ///< Is it top image of top octave
        double ScaleAbs() const;   ///< Scale of Gauss "Absolute" 
        double ScaleInO() const;   ///< Scale of Gauss "Absolute" 
        tGPIm * Up() const;         ///< Possible image up in the   octave
        tGPIm * Down() const;       ///< Possible image down in the octave
        const std::string &   NameSave() const;
        tOct*  Oct() const;        ///< Octave it belongs to
        int    NumInOct () const;        ///< Number inside octave

    private :

        cGP_OneImage(const tGPIm &) = delete;
        tOct*         mOct;        ///< Octave it belongs to
        tPyr*         mPyr;        ///< Pyramid it belongs to
        int           mNumInOct;   ///< Number inside octave
        tGPIm *       mUp;         ///< Possible image up in the   octave
        tGPIm *       mDown;       ///< Possible image down in the octave
        tGPIm *       mOverlapUp;  ///< Overlaping image (== same sigma) , if exist, in up octave
        cIm2D<Type>   mImG;        ///< Gaussian image
        double        mScaleAbs;   ///< Scale of Gauss "Absolute" , used for global analyse
        double        mScaleInO;   ///< Scale of Gauss  In octave , used for image processing
        double        mTargetSigmAbs;  ///< Sigma Abs of gaussian we need to reach
        double        mTargetSigmInO;  ///< Sigma , in octave, of gaussian we need to reach
        bool          mIsTopPyr;   ///< Is it top image of top octave
        std::string   mNameSave;   ///< Name generated for saving the image when required
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
        typedef cIm2D<Type>            tIm;
        typedef cGP_OneImage<Type>     tGPIm;
        typedef std::shared_ptr<tGPIm> tSP_GPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef cGaussianPyramid<Type> tPyr;

        cGP_OneOctave(tPyr * aPyr,int aNum,tOct * aUp); ///< Constructor

        tGPIm * GPImTop() const ; ///< For init or save, need to access to highest resolution image
        tIm ImTop() const; ///< For initalisation , need to access to top image of the pyramid
        tGPIm * ImageOfScaleAbs(double aScale, double aTolRel=1e-5) ; ///< Return image having given sigma
        cPt2dr Oct2File(const cPt2dr &) const; ///< To geomtry of global file

        void Show() const;  ///< Show octave in text format, test and debug
        void ComputGaussianFilter();  ///< Generate computation of gauss pyram
  
        /** Put in all image of this, the  image  whic are differences of consecutive image in anOct */
        void  MakeDiff(const tOct & anOct);

        /**  Put in all image of this, image original normalized */
        void  MakeOrigNorm(const tOct & anOct);

        
        //  ====  Accessors  ===========
        tPyr*          Pyram() const ;      ///< Accessor to Pyram
        const cPt2di &  SzIm() const;       ///<  mSzIm
        const double &  Scale0Abs() const;  ///< Scale of Top Image
        tOct *          Up() const;         ///< Possible octave up in the pyramid, 0 if dont exist
        const int &     NumInPyr() const;   ///< Number inside Pyram
        const std::vector<tSP_GPIm>& VIms()const ;       ///< Images of the Pyramid
       
    private :
        cGP_OneOctave(const tOct &) = delete;
        tPyr*              mPyram;        ///< Pyramid it belongs to
        tOct *             mUp;           ///< Possible octave up in the pyramid
        tOct *             mDown;         ///< Possible octave down in the pyramid
        int                mNumInPyr;     ///< Number inside Pyram
        cPt2di             mSzIm;         ///< Size of images in this octave
        double             mScale0Abs;     ///<  Abs Scale of first image in octave
        std::vector<tSP_GPIm> mVIms;       ///< Images of the Pyramid
};


/// Struct for parametrization of Gaussian Pyramid

/**  As  cGaussianPyramid may evolve with many parameters
     it's a "good" precaution to use a struct of parametrisation
*/

struct cGP_Params
{
     public :
         cGP_Params(const cPt2di & aSzIm0,int aNbOct,int aNbLevByOct,int aOverlap);

      // Parameters having no def values
         cPt2di mSzIm0;    ///< Sz of Image at full resol
         int  mNbOct;      ///< Number of octave
         int  mNbLevByOct;  ///< Number of level per octave (dont include overlap)
         int  mNbOverlap;   ///< Number of overlap
         
      // Parameters with def value, can be changed
         double      mConvolIm0;  ///< Possible additionnal convolution to first image  def 0.0
         int         mNbIter1;    ///<  Number of iteration of first Gaussian              def 4
         int         mNbIterMin;  ///<  Min number if iteration                            def 2
         double      mConvolC0;   ///<  Additional convolution for corner pyramid
         double      mScaleDirOrig;   ///<  Scale multiplier for diff on Orig carac point
         double      mEstimSigmInitIm0;   ///< Estimation of sigma0 of first image
};

/// Struct to store a gaussian pyram

template <class Type> class  cGaussianPyramid : public cMemCheck
{
    public :
      // Typedef section 
        typedef cIm2D<Type>            tIm;
        typedef cGP_OneImage<Type>     tGPIm;
        typedef std::shared_ptr<tGPIm> tSP_GPIm;
        typedef cGP_OneOctave<Type>    tOct;
        typedef std::shared_ptr<tOct>  tSP_Oct;
        typedef cGaussianPyramid<Type> tPyr;
        typedef std::shared_ptr<tPyr>  tSP_Pyr;

        static tSP_Pyr Alloc(const cGP_Params &,const std::string & aNameIm,const std::string & aPref,const cRect2 & aBIn,const cRect2 & aBOut); ///< Allocator
        /** Generate a Pyramid made of the difference, typically for laplacian from gaussian */
        tSP_Pyr  PyramDiff() ;
        /** Generate a Pyramid of corner points */
        tSP_Pyr  PyramCorner() ;
        /** Generate a Pyramid "almost" original but with normalized values */
        tSP_Pyr  PyramOrigNormalize() ;

       
        cPt2dr Pyr2File(const cPt2dr &) const; ///< To geomtry of global file

        void Show() const;  ///< Show pyramid in text format, test and debug
        tIm ImTop() const; ///< For initalisation , need to access to top image of the pyramid
        tGPIm * GPImTop() const ; ///< For init or save ..., need to access to highest resolution image in high oct
        tOct * OctHom(tOct *) ;  ///< return the homologue octave (from another pyramid)
        tGPIm * ImHom(tGPIm *) ;  ///< return the homologue image (from another pyramid)

        void ComputGaussianFilter();  ///< Generate gauss in image of octave
        void SaveInFile(int aPowSPr,bool ForInstpect) const;  ///< Save images  
      // Accessors
        const cGP_Params & Params() const;  ///< Parameters of pyramid
        const double & MulScale() const;    ///< Multiplier sigm conseq
        const double & Scale0() const;     ///< mScale0 scale of first image, 1.0 in 99.99 %
        int  NbImByOct() const;            ///< mNbImByOct + mNbOverlap
        const cPt2di & SzIm0() const;      ///< Size of most resolved images
        const double & SigmIm0() const;    ///< Sigma of first image after possible convolution
        const std::vector<tSP_Oct>&  VOcts() const; ///< vector of octaves
        const std::vector<tSP_GPIm>&  VAllIms () const;  ///< Vector of All Images of All Octaves
        eTyPyrTieP TypePyr () const;    ///< Type in enum possibility (Laplapcian of Gauss, Corner ...)
        const std::string & NameIm() const; ///<  Name of image
        const std::string & Prefix() const; ///<  Name of image

    private :
        cGaussianPyramid(const cGP_Params &,tPyr * aOrig,eTyPyrTieP,const std::string & aNameI,const std::string & aPref,const cRect2 & aBIn,const cRect2 & aBOut); ///< Constructor
        cGaussianPyramid(const tPyr &) = delete;

        tPyr *   mPyrOrig;    ///< The pyramide that contain the original images
        eTyPyrTieP mTypePyr;    ///< Type in enum possibility (Laplapcian of Gauss, Corner ...)
        std::string mNameIm; ///<  Name of image
        std::string mPrefix; ///<  To add to identifier for output
        cRect2      mBoxIn; /// Box of Input
        cRect2      mBoxOut; ///  Box of Output
        cGP_Params          mParams;   ///< Memorize parameters
        std::vector<tSP_Oct>   mVOcts;  ///< Vector of octaves
        std::vector<tSP_GPIm>  mVAllIms;  ///< Vector of All Images of All Octaves

        double   mMulScale;  ///<  Scale of gaussian multiplier between two consecutive gaussian
        double   mScale0;    ///< Scale of first image, conventionnaly 1 
        /** This one is a bit tricky, because rigourously speaking , we should know the initial
        FTM, conventionnaly we select a well sampled image with Sigm = DefStdDevImWellSample */
        double   mEstimSigmInitIm0;  
        double   mSigmIm0; ///<   Sigma of first image after possible convolution
};

};





#endif  //   _MMVII_Linear2DFiltering_H_
