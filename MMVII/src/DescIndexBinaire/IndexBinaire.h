#ifndef _INDEXBINAIRE_H_
#define _INDEXBINAIRE_H_

namespace MMVII
{

typedef bool tVBool;

// Global Application class, in one process deal with only one type of carac points
class cAppli_ComputeParamIndexBinaire ;
// Class for representing information on all file on one invariant
class cDataOneInvRad;
// Class for representing information on one file, it's only meta data, as the global image is not memmorized
class cMetaDataOneFileInvRad ;
/// Class for storing info/vector on one Pts Carac
class  cVecInvRad ;
     /*=====   Class specific to bit vector processing =====*/
// Virtual Class for computing on bit
class cIB_LineFoncBool;
//  Class to store computed values of given bit computer
class cVecBool;
//  Class to store the stat on number of bits equal
class cStatDifBits;


typedef std::shared_ptr<cVecInvRad> tPtVIR;
typedef std::vector<tPtVIR>            tVPtVIR;

typedef std::shared_ptr<cVecBool> tPtVBool;

/// Global class to process computation of binary index

/**
     Global class to process computation of binary index.
   
     Probably the application will be used for other learning task (as learning
     decison by SVM, RF, RNN ... from invariant or LogPol)
*/

class cAppli_ComputeParamIndexBinaire : public cMMVII_Appli
{
     public :
        virtual ~cAppli_ComputeParamIndexBinaire();
        cAppli_ComputeParamIndexBinaire(const std::vector<std::string> &,const cSpecMMVII_Appli &);
  
        // Pure virtual methods of cMMVII_Appli
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        
        // void   SaveFileData(); Not implemented for now
        // Accessors
        const std::string & DirCurPC() const ;
        cVecInvRad*  IR(int aK);
        double PropFile() const;
        const cStrStat2<tREAL8> & Stat2();
        cDenseVect<tREAL8> &    TmpVect();
        const cResulSymEigenValue<tREAL8> &  Eigen() const;
        int    ZoomImg() const;
        cIm2D<tU_INT1> MakeImSz(cIm2D<tU_INT1>);

     private :
         void ShowStat();
         void ChangeVB(double aSc,tPtVBool,int);
         void TestNewParamLinear(const std::vector<tPtVBool>&,int aK0);
         void OptimiseScoreAPriori(tPtVBool ,const tVPtVIR & aVIR);
         tPtVBool VecBoolFromSol(const  cDenseVect<double> & aSol,int aIndex);


         template <class Type> void AddOneEqParamLin(double aPds,const cDataIm1D<Type> & aCdg,double aVal,int aNb);
         void AddVecOneEqParamLin(double aPds,cVecInvRad & aCdg,double aVal);
         void AddEqProxCur(double aPdsCloseCur,tPtVBool aVB0);




         void ProcessOneDir(const std::string &);
         void TestNewSol(const std::vector<int> &,const std::vector<tPtVBool> &);
         double ScoreSol(int & aKMax,const  std::vector<tPtVBool> &aNewVB);

         double  ScoreAPrioiri(const std::vector<int>& aSet) const;


         void ComputeIndexBinaire();
         void OneIterComputeIndexBinaire();
         std::vector<tPtVBool> IndexToVB(const std::vector<int>&) const;
         std::vector<tPtVBool> GenereVB(int aNbTest,std::vector<int>&,int aK,int aNb) const;
         void  TestRandom();
         void TestNbBit() const;
         void TestNbBit(int aNb) const;

         std::string    mDirGlob;   ///< Directory 4 data
         std::string    mPatPCar;   ///< Pattern for carac to process, if several run indepently in paral
         std::string    mPatInvRad; ///< Pattern for radial invariant used
         tREAL8         mNbPixTot;  ///< Number of pixel, juste 4 info

            // Parameter that fix the combinatory (for tuning essentially)
         double         mPropFile;  ///< Prop of selected file, tuning to go faster in test mode
         int            mNbIterBits;  ///< Number of iteration for bits selection
         int            mNbEigenVal; ///< Number of Eigen Value selected
         int            mNbTestCombInit; ///< Number of test in initial combinatory
         int            mNbOptCombLoc; ///< Number of test combinatory 4 local optim
         bool           mQuickBits;    ///< Fix all parameter to low values for test
         int            mZoomImg;  ///< Zoom of decimation of images

         std::vector<std::string>  mLDirPC;   ///< List of directory containg PCar
         std::string               mDirCurPC; ///< Containe Current PCar, as only one is processed simultaneously
         std::vector<eTyInvRad>    mVecTyIR;  ///< List of invariand radial used
         std::vector<std::unique_ptr<cDataOneInvRad> > mVecDIR;  ///< List of class to organize of radial inv
         int  mNbPts;   ///< Number of PCar 
         int                    mNbValByP;  ///< Number of value by point, for dimensionning
         tVPtVIR                mVIR;
         tVPtVIR                mVIR0;

         cDenseVect<tREAL8>    mTmpVect;

         cStrStat2<tREAL8>     mStat2;
         cLeasSqtAA<tREAL8>    mLSQOpt;
         const cResulSymEigenValue<tREAL8>  *mEigen;
         std::vector<tPtVBool>              mVVBool;  ///< Memorized vector of bool

         std::vector<cPt2di>  mVecTrueP;  ///< True pairs
         std::vector<cPt2di>  mVecFalseP; ///< Pairs of non hom, random but memorized to have always same
         std::string          mSaveFileSelVectIR;  ///< To Save file of selected IR, usefull if we rerun from previous comp
              // 3 variable used to store curent solution of bits selection
         double                       mBestSc;  ///< Current score (initialize - inft)
         std::vector<tPtVBool>       mBestVB;   ///< Vector of bool
         std::vector<int>             mBestIndex; ///< Index of these vectors
         bool                         mMedian; ///< Median instead of average
         double                       mWFP;  ///< Weigth False Positive
         int                          mNbVecBit; ///< Size of bit vector
         int                          mNbMaxNeigh; ///< Used for avoiding error in RandNeighSet
         bool                         mOptimAPrio; ///< Use a priori optimization of Bit Vect
};


/// Class for storing info/vector on one Pts Carac
/**  Essentially this class is a vector of "real value", pack to tU_INT1 */
class  cVecInvRad : public cMemCheck
{
      public :
         cVecInvRad(int aNbVal);
      
         cIm1D<tU_INT1>  mVec;  ///< Values
         bool            mSelected; ///< When we iterate binary index, use this to "supress" IR that satisy the test
};



class cMetaDataOneFileInvRad : public cMemCheck
{
     public :
        cMetaDataOneFileInvRad(cDataOneInvRad &,const std::string &);
        void CheckCoherence(const cMetaDataOneFileInvRad&) const;

        void SetNbPair();  ///< called to compute number of patch when mDOIR has enough info (know size of patch)
        /// Read file and add carac points
        void  AddPCar() ;

        cDataOneInvRad * mDOIR; ///<  Radial invariant upper one given folder
        std::string            mName; ///<  short name (without folder)
        cDataFileIm2D          mDFIm; ///<  Information on a single file 
        int                    mNbPair;  ///<  Number of pair
};

class cDataOneInvRad  : public cMemCheck
{
    public :
       cDataOneInvRad(cAppli_ComputeParamIndexBinaire & anAppli,cDataOneInvRad * aPrev,eTyInvRad);

       ///  May differ from rough size of pix if we decide to do some sub-sampling
       int        NbValByP() const; 
       /// All the folder must have the same structure (files, number of points ...) : check it
       void CheckCoherence(const cDataOneInvRad&) const;
       /// Read files and add carac points
       void  AddPCar() ;

       // Accessors
       cAppli_ComputeParamIndexBinaire  &   Appli() ;
       const std::string&   Dir() const;
       const cPt2di &  SzP0Init() const;  
       const cPt2di &  SzP0Final() const;  
       int  NbPixByP() const;
       eTyInvRad  TIR() const;
       tREAL8     NbPixTot() const;
       int        NbPatch() const;  
       int        PosInVect() const;  
       int &      KFill();
    private :
       void SetSzP0Final(const cPt2di &);
       cDataOneInvRad(const cDataOneInvRad &) = delete;

       cAppli_ComputeParamIndexBinaire  &   mAppli;  ///< Memorize global application
       eTyInvRad                            mTIR;  ///< Kind of radial invariant
       std::string                          mDir; ///< Directory with file "Cple.*tif"
       std::vector<cMetaDataOneFileInvRad>  mMDOFIR;
       cPt2di                               mSzP0Init;   ///< Size Patch Init, before decimate
       cPt2di                               mSzP0Final;  ///< Size Patch Final, after possible decimation

       int                                  mNbPixByP; ///< Nb of Pix/Patch Can differ if reduced
       tREAL8                               mNbPixTot;  ///< Number of Pixel, for stat/info
       int                                  mNbPatch;  ///<  Number of patch = 2 *NbPair
       int                                  mPosInVect; ///< Cumul of previous NbPixByP() to know where pack in Vect
       int                                  mKFill; ///< Current number of PCar being filled by  files
};




class cIB_LinearFoncBool : public cMemCheck
{
     public :
        bool Calc(const cVecInvRad &) const ;
        double RCalc(const cVecInvRad &) const ;
        cIB_LinearFoncBool
        (
             cAppli_ComputeParamIndexBinaire & anAppli,
             const cDenseVect<double>&   aVect,
             double                aThresh 
        );
        cIB_LinearFoncBool
        (
             cAppli_ComputeParamIndexBinaire & anAppli,
             const cDenseVect<double>&   aVect
        );
        cIB_LinearFoncBool
        (
             cAppli_ComputeParamIndexBinaire & anAppli,
             int aK
        );
        const cDenseVect<double>& Vect() const;
     private :
        cIB_LinearFoncBool (const cIB_LinearFoncBool &) = delete;
        cAppli_ComputeParamIndexBinaire & mAppli;
        // int                   mK;
        cDenseVect<double>    mVect;
        double                mThresh;
};


class cVecBool  : public cMemCheck
{
     public :
         cVecBool(int Index,bool Med,cIB_LinearFoncBool * aFB,const tVPtVIR &);
         void CalcCdgScoreAPriori (const tVPtVIR & aVIR);

         bool  KBit(int aK) const {return mVB.at(aK);}
         const tREAL4 & Score() const {return mScore;}

         cIB_LinearFoncBool&  FB();
         const cDenseVect<tREAL4>& Cdg0() const;
         int   Nb0() const;
         const cDenseVect<tREAL4>& Cdg1() const;
         int   Nb1() const;
         int   Index() const;
         
     private :
         cVecBool(const cVecBool &) = delete;

         int                                  mIndex;
         std::shared_ptr<cIB_LinearFoncBool>  mFB;
         std::vector<bool> mVB;

         cDenseVect<tREAL4>  mCdg0;
         int                 mNb0;
         cDenseVect<tREAL4>  mCdg1;
         int                 mNb1;
         tREAL4              mScore;
};

int NbbBitDif(const std::vector<tPtVBool> & aVVB,const cPt2di & aPair);

class cStatDifBits
{
     public :
        cStatDifBits(const std::vector<cPt2di> & aVPair,const std::vector<tPtVBool> & aVB);
        std::vector<int>     mHNbBitDif;
        std::vector<double>  mStatR;
        std::vector<double>  mStatRCum;
        double Score(const cStatDifBits & aStatFalse,double aPdsFalse,int &aKMax) const;
        void  Show(const cStatDifBits & aStatFalse,int aK1,int K2,double aMaxTrue) const;
};




};

#endif // _INDEXBINAIRE_H_

