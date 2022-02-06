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
   private :
      int                        mNum;
      std::vector<tBinCodeTarg>  mCodes;
};




class cParamCodedTarget
{
    public :
       cParamCodedTarget();
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
       double    mRatioBar;  // Ratio H/l on coding rect, def 1, 2-> mean more bar
       double    mRhoWhite0;  // Central circle, used to compute affinity
       double    mRhoBlack0;  // Black circle, used for detection
       int       mNbCircle;  // Number of circles encoding information
       double    mThCircle;  //   Thickness of each circle
       double    mDistMarkFid;    // Dist between Fid mark &  codage
       double    mBorderMarkFid;  // Dist between Bord & FidMark
       double    mRadiusFidMark;  // Radius of Fid Mark
       double    mTetaCenterFid;   // Teta init 
       int       mNbPaqFid;        // Number of group in "Fid Mark" By defaut==mNbRedond

       int       mNbFidByPaq;        // Number of Fiducial by quarter
       double    mGapFid;            // Size of gab in fiducial repeat
       double    mScaleTopo;         // Scale used to create identifiable center 4 toto
       int       mNbPixelBin;        // Number of pixel  Binary image


       std::vector<double> mTetasQ;  // Tetas of first quarter

       double    mRhoCodage0;   // Rho when begin binarie code
       double    mRhoCodage1;   // Rho when ends binarie code
       double    mRhoFidMark;   // Rho where are located Fid Mark
       double    mRhoEnd ;      // Rho where are finish the target


       double mRho_00_TopoB   ;  // Circle for topo ident
       double mRho_000_TopoW  ;  // Circle for topo ident
       double mRho_0000_TopoB ;  // Circle for topo ident

       cPt2di    mSzBin;
       cPt2dr    mMidle;
       double    mScale;  // Sz of Pixel in normal coord

       std::vector<cSetCodeOf1Circle>     mVecSetOfCode;
       cDecomposPAdikVar                  mDecP;
};

};
};
#endif // _CODED_TARGET_H_

