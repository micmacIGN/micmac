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
// Class for storing info/vector on one Pts Carac
class  cVecInvRad ;
// Virtual Class for computing on bit
class cIB_FoncBool ;
// Virtual Class for computing on bit
class cIB_LineFoncBool;
//  Class to store computed values of given bit computer
class cVecBool;
//  Class to store the stat on number of bits equal
class cStatDifBits;


typedef std::shared_ptr<cVecInvRad> tPtVIR;
typedef std::vector<tPtVIR>            tVPtVIR;

typedef std::unique_ptr<cVecBool> tPtVBool;

/**
     Global class to process computation of binary index, will be used probably for
     other learning.
*/

class cAppli_ComputeParamIndexBinaire : public cMMVII_Appli
{
     public :
        cAppli_ComputeParamIndexBinaire(int argc,char** argv,const cSpecMMVII_Appli &);
  
        // Pure virtual methods of cMMVII_Appli
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        
        // Accessors
        const std::string & DirCurPC() const ;
        cVecInvRad*  IR(int aK);
        double PropFile() const;
        const cStrStat2<tREAL8> & Stat2();
        cDenseVect<tREAL8> &    TmpVect();
        const cResulSymEigenValue<tREAL8> &  Eigen() const;
        void   SaveFileData();
     private :
         void ProcessOneDir(const std::string &);
         void TestNewSol(const std::vector<int> &,const std::vector<const cVecBool*> &);

         void ComputeIndexBinaire();
         std::vector<const cVecBool*> IndexToVB(const std::vector<int>&) const;
         std::vector<const cVecBool*> GenereVB(std::vector<int>&,int aK,int aNb) const;
         void  TestRandom(int aNbTest,int aNb); 
         void TestNbBit() const;
         void TestNbBit(int aNb) const;

         std::string    mDirGlob;   ///< Directory 4 data
         std::string    mPatPCar;   ///< Pattern for carac to process, if several run indepently in paral
         std::string    mPatInvRad; ///< Pattern for radial invariant used
         tREAL8         mNbPixTot;  ///< Number of pixel, juste 4 info
         double         mPropFile;  ///< Prop of selected file, tuning to go faster in test mode

         std::vector<std::string>  mLDirPC;   ///< List of directory containg PCar
         std::string               mDirCurPC; ///< Containe Current PCar, as only one is processed simultaneously
         std::vector<eTyInvRad>    mVecTyIR;  ///< List of invariand radial used
         std::vector<std::unique_ptr<cDataOneInvRad> > mVecDIR;  ///< List of class to organize of radial inv
         int  mNbPts;   ///< Number of PCar 
         int                    mNbValByP;  ///< Number of value by point, for dimensionning
         tVPtVIR                mVIR;

         cDenseVect<tREAL8>    mTmpVect;

         cStrStat2<tREAL8>     mStat2;
         const cResulSymEigenValue<tREAL8>  *mEigen;
         std::vector<tPtVBool>              mVVBool;  ///< Memorized vector of bool

         std::vector<cPt2di>  mVecTrueP;  ///< True pairs
         std::vector<cPt2di>  mVecFalseP; ///< Pairs of non hom, random but memorized to have always same
         std::string          mSaveFileSelVectIR;  ///< To Save file of selected IR, usefull if we rerun from previous comp
         double                       mBestSc;
         std::vector<const cVecBool*> mBestVB;
         std::vector<int>             mBestIndex;
};

class  cVecInvRad : public cMemCheck
{
      public :
         cVecInvRad(int aNbVal);
      
         cIm1D<tU_INT1>  mVec;
         bool            mSelected;
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

class cDataOneInvRad : public cMemCheck
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
       const cPt2di &  SzP0() const;  
       int  NbPixByP() const;
       eTyInvRad  TIR() const;
       tREAL8     NbPixTot() const;
       int        NbPatch() const;  
       int        PosInVect() const;  
       int &      KFill();
    private :
       cDataOneInvRad(const cDataOneInvRad &) = delete;

       cAppli_ComputeParamIndexBinaire  &   mAppli;  ///< Memorize global application
       eTyInvRad                            mTIR;  ///< Kind of radial invariant
       std::string                          mDir; ///< Directory with file "Cple.*tif"
       std::vector<cMetaDataOneFileInvRad>  mMDOFIR;
       cPt2di                               mSzP0;  ///< Size Patch Init
       int                                  mNbPixByP; ///< Nb of Pix/Patch Can differ if reduced
       tREAL8                               mNbPixTot;  ///< Number of Pixel, for stat/info
       int                                  mNbPatch;  ///<  Number of patch = 2 *NbPair
       int                                  mPosInVect; ///< Cumul of previous NbPixByP() to know where pack in Vect
       int                                  mKFill; ///< Current number of PCar being filled by  files
};



class cIB_FoncBool : public cMemCheck
{
     public :
        virtual bool   Calc(const cVecInvRad &) const =0; ///< Compute one bit
        virtual double RCalc(const cVecInvRad &) const;   ///< Continuous version def -0.5 / 0.5
        virtual ~cIB_FoncBool();
        cIB_FoncBool(cAppli_ComputeParamIndexBinaire & anAppli);
     protected :
        cAppli_ComputeParamIndexBinaire & mAppli;
     private :
        cIB_FoncBool (const cIB_FoncBool &) = delete;
};

class cIB_LinearFoncBool : public cIB_FoncBool
{
     public :
        bool Calc(const cVecInvRad &) const override;
        double RCalc(const cVecInvRad &) const override;
        cIB_LinearFoncBool
        (
             cAppli_ComputeParamIndexBinaire & anAppli,
             int aK,
             double aTreshold
        );
     private :
        cIB_LinearFoncBool (const cIB_LinearFoncBool &) = delete;
        int                   mK;
        double                mThresh;
};


class cVecBool : public cMemCheck
{
     public :
         cVecBool(cIB_FoncBool * aFB,const tVPtVIR &);
         bool  KBit(int aK) const {return mVB.at(aK);}
     private :
         cVecBool(const cVecBool &) = delete;
         std::shared_ptr<cIB_FoncBool>  mFB;
         std::vector<tU_INT1> mVB;
};

int NbbBitDif(const std::vector<const cVecBool*> & aVVB,const cPt2di & aPair);

class cStatDifBits
{
     public :
        cStatDifBits(const std::vector<cPt2di> & aVPair,const std::vector<const cVecBool*> & aVB);
        std::vector<int>     mHNbBitDif;
        std::vector<double>  mStatR;
        std::vector<double>  mStatRCum;
        double Score(const cStatDifBits & aStatFalse,double aPdsFalse,int &aKMax) const;
        void  Show(const cStatDifBits & aStatFalse,int aK1,int K2) const;
};




};

#endif // _INDEXBINAIRE_H_

