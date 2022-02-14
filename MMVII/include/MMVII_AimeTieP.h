#ifndef _AIME_TIEP_H_
#define _AIME_TIEP_H_

namespace MMVII
{
class cAimeDescriptor;
class cAimePCar;
class cSetAimePCAR;

/// Proto-class for Aime TieP

/**  This very basic class is made to export result of MMV2 in a way they can
     be inspected in MMV1 (as MMV2 will not have tools equivalent to X11-MMV1 before a long)
     Certainly Aime TieP will be much more complex than this proto class
*/
template <class Type> class cProtoAimeTieP : public cMemCheck
{
    public :
        typedef cGP_OneImage<Type> tGPI;
        typedef cGaussianPyramid<Type> tPyr;
        bool  FillAPC(const cFilterPCar&,cAimePCar &,bool ForTest);
        bool  TestFillAPC(const cFilterPCar&); // Just to know if the point is OK for filling it
        // cProtoAimeTieP(const cPt2dr & aP,int aNumOct,int aNumIm,float aScaleInO,float aScaleAbs);

        // With Integer P, used in Aime
        cProtoAimeTieP(cGP_OneImage<Type> *,const cPt2di & aPImInit,bool ChgMaj);
        // With Real P, used in dense match, we directly have the "refined" point
        cProtoAimeTieP(cGP_OneImage<Type> *,const cPt2dr & aPImInit);

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
        cPt2dr               mPRImRefined;      ///<  Real point 
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
         //  aDILPr.SetV(cPt2di(aKTeta,IndRhoLP),aV);

         cAimeDescriptor DupLPIm() const;  // Duplicate only LP IM
         cAimeDescriptor();  ///< Need a default descriptor (serialization ...)
         cIm2D<tU_INT1>   ILP() const;   ///< Accesor to log pol image

         const std::vector<double> & DirPrinc() const; ///< const accesor to main directions
         std::vector<double> & DirPrinc() ;            ///< non const accessor

         ///  this[x] / AD2[x+aShift]
         double  DistanceFromIShift(const cAimeDescriptor & aAD2,int aShift,const cSetAimePCAR & aSet) const;
         // X1 and X2 are the peek, supposed to be homologous in 
         double  DistanceFrom2RPeek(double aX1,const cAimeDescriptor & aAD2,double aX2,const cSetAimePCAR & aSet) const;
         // IPeek is an index from  DirPrinc
         double  DistanceFromStdPeek(int aIPeek,const cAimeDescriptor & aAD2,const cSetAimePCAR & aSet) const;
         //  Compute best match from all Dir Princ
         cWhitchMin<int,double>  DistanceFromBestPeek(const cAimeDescriptor & aAD2,const cSetAimePCAR & aSet) const;

     private :
        cIm2D<tU_INT1>      mILP;       ///< mImLogPol
        std::vector<double> mDirPrinc ; ///< Principal directions  options
};

/**  Class to store Aime Pts Car = Descriptor + localization
*/ 
class cAimePCar : public cMemCheck
{
     public :
        cAimeDescriptor & Desc();
        const cAimeDescriptor & Desc() const;
        cPt2dr&         Pt();
        const cPt2dr&   Pt() const;
        cPt2dr&         PtIm();
        const cPt2dr&   PtIm() const;

        cAimePCar       DupLPIm() const; // Duplicate only LP IM
        double          L1Dist(const cAimePCar&) const;
     private :
        cAimeDescriptor mDesc;  ///< Descriptor
        cPt2dr          mPt;    ///<  Localization
        cPt2dr          mPtIm;    ///<  Localization in image (offseted), not realy usefull as Tie-P, but required in dense match
};

/**  Class to store  aSet of AimePcar = vector<PC> + some common caracteritic on type
*/ 
class cSetAimePCAR : public cMemCheck
{
     public :
        // cSetAimePCAR();
        cSetAimePCAR(eTyPyrTieP aType,bool IsMax); ///< "Real" constructor
        cSetAimePCAR(); ///< Sometime need a default constructor
        int &                   IType();
        eTyPyrTieP              Type();
        bool&                   IsMax();
        bool&                   Census();
        const bool&             Census() const;
        double&                 Ampl2N();
        const double&           Ampl2N() const;
        std::vector<cAimePCar>& VPC();
        void SaveInFile(const std::string &) const;
        void InitFromFile(const std::string &) ;
        // For census, as the value are strictly in [-1,1] we can use a universall value for normalize
        static const double TheCensusMult;
     private :
        int                     mType;   ///< Type registered as int, easier for AddData, in fact a eTyPyrTieP
        bool                    mIsMax;  ///< Is it a maxima or a minima of its caracteristic
        std::vector<cAimePCar>  mVPC;    ///< Vector of Aime points
        bool                    mCensus; ///<  Is it Census mode
        double                  mAmpl2N;   ///< Ampl between the normalized value  IPL = Norm*Ampl
};



};

#endif // _AIME_TIEP_H_

