#ifndef _CODED_TARGET_H_
#define _CODED_TARGET_H_
#include "include/MMVII_all.h"
#include "include/MMVII_SetITpl.h"


namespace MMVII
{

template <class Type>  class  cFilterDCT : public cMemCheck
{
    public :
           typedef cIm2D<Type>     tIm;
           typedef cDataIm2D<Type> tDIm;


           static cFilterDCT<Type> * AllocSym(tIm anIm,double aR0,double aR1,double aEpsilon);

           cFilterDCT(bool IsCumul,eDCTFilters,tIm anIm,bool IsSym,double aR0,double aR1,eDCTFilters aMode,double aThickN=1.5);

           virtual void Reset() = 0;
           virtual void Add(const Type & aWeight,const cPt2dr & aNeigh) = 0;
           virtual double Compute() =0;

           double ComputeVal(const cPt2dr & aP);
           tIm    ComputeIm();
           double ComputeValMaxCrown(const cPt2dr & aP,const double& aThreshold);
           tIm    ComputeImMaxCrown(const double& aThreshold);

           eDCTFilters  TypeF() const;

    protected  :
           cFilterDCT (const cFilterDCT<Type> &) = delete;

           void IncrK0(int & aK0);
           void IncrK1(int & aK1,const int & aK0);

           bool                 mIsCumul;
           eDCTFilters          mTypeF;
           tIm                  mIm;
           tDIm&                mDIm;
           cPt2di               mSz;
           bool                 mIsSym;
           double               mR0;
           double               mR1;
           double               mThickN;
           cPt2dr               mCurC;
           eDCTFilters          mMode;
           std::vector<cPt2di>  mIVois;
           std::vector<cPt2dr>  mRVois;
           double               mRhoEnd;

           std::vector<cPt2di>  mVK0K1;
};


template<class TypeEl>
   double IndBinarity(const  cDataIm2D<TypeEl> & aDIm,const cPt2di & aP0,const std::vector<cPt2di> & aVectVois);

template<class TypeEl> cIm2D<TypeEl> ImBinarity(const  cDataIm2D<TypeEl> & aDIm,double aR0,double aR1,double Epsilon);

std::vector<cPt2dr> VecDir(const  std::vector<cPt2di>&  aVectVois);
template<class TypeEl> double Starity
                              (
                                  const  cImGrad<TypeEl> & aImGrad,
                                  const cPt2dr & aP0,
                                  const  std::vector<cPt2di>&  aVectVois ,
                                  const  std::vector<cPt2dr>&  aVecDir,
                                  double Epsilon
                              );


template<class TypeEl> cIm2D<TypeEl> ImStarity(const  cImGrad<TypeEl> & aImGrad,double aR0,double aR1,double Epsilon);

template<class TypeEl> cIm2D<TypeEl> ImSymMin(cIm2D<TypeEl>  aImIn,double aR0,double aR1,double Epsilon);




template<class TypeEl> cIm2D<TypeEl> ImSymetricity(bool DoCheck,cIm2D<TypeEl> anImIn,double aR0,double aR1,double Epsilon);




namespace  cNS_CodedTarget
{

typedef cSetISingleFixed<tU_INT4>  tBinCodeTarg;
typedef std::vector<tBinCodeTarg> tVSetICT;
typedef cIm2D<tU_INT1>     tImTarget;
typedef cDataIm2D<tU_INT1> tDataImT;


/*  *********************************************************** */
/*                                                              */
/*                      cParamCodedTarget                       */
/*                                                              */
/*  *********************************************************** */

class cSetCodeOf1Circle
{
    public :
      cSetCodeOf1Circle(const std::vector<int> & aCards,int aN);
      int  NbSub() const;
      const tBinCodeTarg & CodeOfNum(int aNum) const;
      int N() const;
    private :
      std::vector<int>   mVCards;
      int      mN;
      tVSetICT mVSet ;  //   All the binary code of one target 
};


class cCodesOf1Target
{
   public :
      cCodesOf1Target(int aNum);

      void AddOneCode(const tBinCodeTarg &);
      void  Show();
      const tBinCodeTarg & CodeOfNumC(int) const;
      int   Num() const;
   private :
      int                        mNum;
      std::vector<tBinCodeTarg>  mCodes;
};




class cParamCodedTarget
{
    public :
       cParamCodedTarget();
       void InitFromFile(const std::string & aNameFile);

       int &     NbRedond();  // Redundancy = number of repetition of a pattern in a circle
       int &     NbCircle();  // Redundancy = number of repetition of a pattern in a circle
       double &  RatioBar();  // Ratio on codin bar
       void      Finish();

       int NbCodeAvalaible() const;         // Number of different code we can generate
       int BaseForNum() const;         // Base used for converting integer to string
       cCodesOf1Target CodesOfNum(int);     // One combinaison of binary code
       tImTarget  MakeIm(const cCodesOf1Target &);  // Generate the image of 1 combinaison
       tImTarget  MakeImCodeExt(const cCodesOf1Target &);  // Generate the image of 1 combinaison

       void AddData(const cAuxAr2007 & anAux);

       bool CodeBinOfPts(double aRho,double aTeta,const cCodesOf1Target & aSetCodesOfT,double aRho0,double aThRho);

       std::string NameOfNum(int) const; ///  Juste the apha num
       std::string NameFileOfNum(int) const; ///  Juste the apha num


       cPt2dr    mCenterF;   // symetry center at end
       cPt2di    mSzF;       // sz at end
       cPt2dr    mMidle;
    // private :

       cPt2dr    Pix2Norm(const cPt2di &) const;
       cPt2dr    Norm2PixR(const cPt2dr &) const;
       cPt2di    Norm2PixI(const cPt2dr &) const;


       int       mNbBit;  // Do not include parity, so 5=> gives 16 if parity used
       bool      mWithParity;  // Do we use parirty check
       int       mNbRedond;  // Redundancy = number of repetition of a pattern in a circle
       int       mNbCircle;  // Number of circles encoding information
       int       mNbPixelBin;        // Number of pixel  Binary image
       const double    mSz_CCB;      // size of central chekcboard/target , everything prop to it, 1 by convention

       double    mThickN_WInt;  // Thickness white circle separating code/
       double    mThickN_Code;  // Thickness of coding part
       double    mThickN_WExt;  // Thickness of white separatio,
       double    mThickN_Car;  // thickness of black border (needed only on pannel)
       double    mThickN_BExt;  // thickness of black border (needed only on pannel)


       double          mRho_0_EndCCB;// End of Central CB , here Rho=ThickN ...
       double          mRho_1_BeginCode;// ray where begins the coding stuff
       double          mRho_2_EndCode;// ray where begins the coding stuff
       double          mRho_3_BeginCar;// ray where begins the coding stuff
       double          mRho_4_EndCar;  // ray where begins the coding stuff


       cPt2di    mSzBin;
       double    mScale;  // Sz of Pixel in normal coord

       std::vector<cSetCodeOf1Circle>     mVecSetOfCode;
       cDecomposPAdikVar                  mDecP;
};

void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT);


};
};
#endif // _CODED_TARGET_H_

