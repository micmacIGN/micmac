#ifndef _INDEXBINAIRE_H_
#define _INDEXBINAIRE_H_

namespace MMVII
{

// Global Application class, in one process deal with only one type of carac points
class cAppli_ComputeParamIndexBinaire ;
// Class for representing information on all file on one invariant
class cDataOneInvRad;
// Class for representing information on one file, it's only meta data, as the global image is not memmorized
class cMetaDataOneFileInvRad ;
// Class for storing info/vector on one Pts Carac
class  cVecInvRad ;




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
     public :
         void ProcessOneDir(const std::string &);

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
         std::vector<std::unique_ptr<cVecInvRad> > mVIR;

         cDenseVect<tREAL4>    mTmpVect;
         cDenseVect<tREAL8>    mMoyVect;
         cDenseMatrix<tREAL8>  mCovVect;
};

class  cVecInvRad : public cMemCheck
{
      public :
         cVecInvRad(int aNbVal);
      
         cIm1D<tU_INT1>  mVec;
         void  Add2Stat(cDenseVect<tREAL4>& aTmp,cDenseVect<tREAL8>& aMoy,cDenseMatrix<tREAL8> & aCov) const;
         ///  Normalise by pds
         static void PostStat(cDenseVect<tREAL8>& aMoy,cDenseMatrix<tREAL8> & aCov,double aPdsTot);
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

};

#endif // _INDEXBINAIRE_H_

