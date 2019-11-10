#ifndef _AIME_TIEP_H_
#define _AIME_TIEP_H_

namespace MMVII
{
class cAimeDescriptor;
class cAimePCar;

/// Proto-class for Aime TieP

/**  This very basic class is made to export result of MMV2 in a way they can
     be inspected in MMV1 (as MMV2 will not have tools equivalent to X11-MMV1 before a long)
     Certainly Aime TieP will be much more complex than this proto class
*/
template <class Type> struct cProtoAimeTieP : public cMemCheck
{
    public :
        typedef cGP_OneImage<Type> tGPI;
        typedef cGaussianPyramid<Type> tPyr;
        bool  FillAPC(const cFilterPCar&,cAimePCar &,bool ForTest);
        bool  TestFillAPC(const cFilterPCar&); // Just to know if the point is OK for filling it
        // cProtoAimeTieP(const cPt2dr & aP,int aNumOct,int aNumIm,float aScaleInO,float aScaleAbs);
        cProtoAimeTieP(cGP_OneImage<Type> *,const cPt2di & aPImInit,bool ChgMaj);

        // void SetPt(const cPt2dr & );
        // const cPt2dr & Pt() const;
        int   NumOct() const;
        int   NumIm() const;
        float ScaleInO() const;
        float ScaleAbs() const;
        const tPyr & Pyram() const;
        const cGP_Params& Params() const;
        

        tGPI *               mGPI;
        bool                 mChgMaj;  ///< Image changed to major, tuning
        cPt2di               mPImInit;      ///<  in image octave coordinate (comes from extrema detection)
        cPt2dr               mPFileInit;    ///< idem, but global file coordinate
        cPt2dr               mPFileRefined; ///< after refinement
        int                  mId;           ///< For debug essentially
        bool                 mOkAutoCor;    ///< Is it OK regarding auto correl threshold
        double               mAutoCor;      ///< Self correlation
        int                  mNumOutAutoC;  ///< Id of exit in self correl
        // double               mCritFastStd;  ///< Fast criterion without connexion
        // double               mCritFastCnx;  ///< Fast criteraion with connexion constraint
        double               mStdDev;       ///< Standadr deviation of radiometrie = contrast
        double               mScoreInit;    ///< Initial score (agreg of var,autoc,scale ...)
        double               mScoreRel;     ///< Relative score, after modification by neighboors
    // Temporary data for computing 
        bool                 mSFSelected;   ///< Spatial Filtering Flag to know if has already been selected
        bool                 mStable;   ///< Is it stable vs refinement
        bool                 mOKLP;   ///< Is it OK for LogPol
        int                  mHeapIndexe;   ///< Data for "indexed heap" stuff
        int                  mNumAPC;  ///< Num pointing inside Vec APC, need to maitain Prop even if reftued

    private :
};

/// Interface class for export Proto Aime

/**  As I want to maintain the principle that only a very minimum
   of code of MMV2 communicate with MMV1, this interface class was created
*/

template <class Type> class cInterf_ExportAimeTiep : public cMemCheck
{
     public :
         static cInterf_ExportAimeTiep<Type> * Alloc(const cPt2di & aSzIm0,bool IsMin,eTyPyrTieP TypePt,const std::string & aName,bool ForInspect,const cGP_Params & );
         virtual ~cInterf_ExportAimeTiep();
         virtual void AddAimeTieP(cProtoAimeTieP<Type>  aPATP ) = 0;
         virtual void Export(const std::string &,bool SaveV1) = 0;
         virtual void FiltrageSpatialPts() = 0; 
     protected :

};

/**  Class to store Aime descriptor, independantly of it caracterization
*/ 
class cAimeDescriptor : public cMemCheck
{
     public :
         cAimeDescriptor();  ///< Need a default descriptor (serialization ...)
         cIm2D<tU_INT1>   ILP();   ///< Accesor to log pol image

         const std::vector<double> & DirPrinc() const; ///< const accesor to main directions
         std::vector<double> & DirPrinc() ;            ///< non const accessor
     private :
        cIm2D<tU_INT1>      mILP;   ///< mImLogPol
        std::vector<double> mDirPrinc ; ///< Principal directions  options
};

/**  Class to store Aime Pts Car = Descriptor + localization
*/ 
class cAimePCar
{
     public :
        cAimeDescriptor & Desc();
        cPt2dr&         Pt();
     private :
        cAimeDescriptor mDesc;  ///< Descriptor
        cPt2dr          mPt;    ///<  Localization
};

/**  Class to store  aSet of AimePcar = vector<PC> + some common caracteritic on type
*/ 
class cSetAimePCAR
{
     public :
        // cSetAimePCAR();
        cSetAimePCAR(eTyPyrTieP aType,bool IsMax);
        int &                   IType();
        eTyPyrTieP              Type();
        bool&                   IsMax();
        std::vector<cAimePCar>& VPC();
        void SaveInFile(const std::string &) const;
     private :
        int                     mType;  ///< Type registered as int, easier for AddData, in fact a eTyPyrTieP
        bool                    mIsMax; ///< Is it a maxima or a minima of its caracteristic
        std::vector<cAimePCar>  mVPC;   ///< Vector of Aime points
   
};



};

#endif // _AIME_TIEP_H_

