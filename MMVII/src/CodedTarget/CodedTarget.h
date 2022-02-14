#ifndef _CODED_TARGET_H_
#define _CODED_TARGET_H_
#include "include/MMVII_all.h"
#include "include/MMVII_SetITpl.h"


namespace MMVII
{

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
      cSetCodeOf1Circle(double aRho0,int aK,int aN);
      int  NbSub() const;
      const tBinCodeTarg & CodeOfNum(int aNum) const;
      int N() const;
      int K() const;
    private :
      double   mRho0;
      int      mK;
      int      mN;
      tVSetICT mVSet ;
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
       cCodesOf1Target CodesOfNum(int);     // One combinaison of binary code
       tImTarget  MakeIm(const cCodesOf1Target &);  // Generate the image of 1 combinaison

       void AddData(const cAuxAr2007 & anAux);


    private :

       cPt2dr    Pix2Norm(const cPt2di &) const;
       cPt2dr    Norm2PixR(const cPt2dr &) const;
       cPt2di    Norm2PixI(const cPt2dr &) const;

       int       mNbRedond;  // Redundancy = number of repetition of a pattern in a circle
       int       mNbCircle;  // Number of circles encoding information

       double    mThTargetC;  // Thickness of central target
       double    mThStars;  //   Thickness of "star" pattern
       double    mThBlCircExt;  //   Thickness of External Black circle
       double    mThBrdWhiteInt;    // Thickness of white internal border
       double    mThBrdBlack;    // Thickness of black border
       double    mThBrdWhiteExt;    // Thickness of black border

       double    mScaleTopo;         // Scale used to create identifiable center 4 toto
       int       mNbPixelBin;        // Number of pixel  Binary image

       std::vector<double> mTetasQ;  // Tetas of first quarter

       double    mThRing ;      // Thickness of each ring of star : mThStars/mNbCircle

       double    mRhoEndTargetC;  // Rho when central targe ends
       double    mRhoEndStar;      // Rho when ends stars pattern
       double    mRhoEndBlackCircle;      // Rho when ends external black circle
       double    mRhoEnBrdWhiteInt;   // Rho where ends interior white border
       double    mRhoEndBrdBlack;   // Rho where ends black border
       double    mRhoEndBrdWhiteExt;   // Rho where ends white border


       cPt2di    mSzBin;
       cPt2dr    mMidle;
       double    mScale;  // Sz of Pixel in normal coord

       std::vector<cSetCodeOf1Circle>     mVecSetOfCode;
       cDecomposPAdikVar                  mDecP;
};

void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT);


};
};
#endif // _CODED_TARGET_H_

