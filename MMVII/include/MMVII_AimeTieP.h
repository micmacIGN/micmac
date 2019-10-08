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
        void  FillAPC(cAimePCar &);
        // cProtoAimeTieP(const cPt2dr & aP,int aNumOct,int aNumIm,float aScaleInO,float aScaleAbs);
        cProtoAimeTieP(cGP_OneImage<Type> *,const cPt2di & aPImInit);

        // void SetPt(const cPt2dr & );
        // const cPt2dr & Pt() const;
        int   NumOct() const;
        int   NumIm() const;
        float ScaleInO() const;
        float ScaleAbs() const;

        cGP_OneImage<Type> * mGPI;
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
        int                  mHeapIndexe;   ///< Data for "indexed heap" stuff

    private :
};

/// Interface class for export Proto Aime

/**  As I want to maintain the principle that only a very minimum
   of code of MMV2 communicate with MMV1, this interface class was created
*/

template <class Type> class cInterf_ExportAimeTiep : public cMemCheck
{
     public :
         static cInterf_ExportAimeTiep<Type> * Alloc(const cPt2di & aSzIm0,bool IsMin,int ATypePt,const std::string & aName,bool ForInspect,const cGP_Params & );
         virtual ~cInterf_ExportAimeTiep();
         virtual void AddAimeTieP(cProtoAimeTieP<Type>  aPATP ) = 0;
         virtual void Export(const std::string &) = 0;
         virtual void FiltrageSpatialPts() = 0; 
     protected :

};

/*
class cAimeDescriptor
{
     public :
         cAimeDescriptor();
     private :
        cIm2D<tU_INT1>   mILP;   ///< mImLogPol
};

class cAimePCar
{
     public :
     private :
};
*/


};

#endif // _AIME_TIEP_H_

