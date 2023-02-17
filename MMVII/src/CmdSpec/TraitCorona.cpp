
// #include "include/MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Matrix.h"


/** \file cMMVII_CalcSet.cpp
    \brief Command for set calculation

    This file contain the command that compute a  set of file from File/Regex
  It's also the first "real" command of MMVII, so an occasion for tuning a 
  a lot of thing.

*/

namespace MMVII
{

typedef tREAL4             tValIm;
typedef cIm2D<tValIm>      tIm;
typedef cDataIm2D<tValIm>  tDIm;

/// Class to store static of rectangle in   cAppli_CalcDiscIm

class cACDI_Stat1Im
{
   public :
       cACDI_Stat1Im  (bool aLeft) :
          mLeft  (aLeft),
          mIm    (cPt2di(1,1))
       {
       }

       bool    mLeft;            ///< Is it left
       tIm     mIm;              ///< Image to store part of file
       double  mMoy;             ///< Average
       double  mMed;             ///<  Median
       std::vector<double> mVy;  ///<  Stacking along the line
       cPt2di   mP0;             ///<  Uper left corner of rectangle
       cPt2di   mP1;             ///< Bottom right corner
};

/// Class to compute discontinuities in image

/**  Created initially to compute discontinuities in Corona image resulting
     from scanning artefact.

      Handle only vertical/horizontal line.  May evolve or not, not sure there is
     interest for much more complex tool (i.e polyline ...) in a command line 
*/

class cAppli_CalcDiscIm : public cMMVII_Appli
{
     public :


        cAppli_CalcDiscIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli &);  ///< constructor
        int Exe() override;                                             ///< execute action
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override; ///< return spec of  mandatory args
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override; ///< return spec of optional args

     private :
         void MakeOneLine(int aCenter,int aNum);
         void CalcBox(cACDI_Stat1Im & aStat);
         void LoadOneIm(cACDI_Stat1Im & aStat);
         void InitC(int aXYC);

      // Read mandatory parameters
         std::string       mNameFile;    ///< Name of Image File
         std::vector<int>  mCenters;     ///< Centers of lines, can be X or Y
         cPt2di            mDP0;         ///<  Top Left corner, relative to centrale position
         cPt2di            mDP1;         ///<  Bottom Right corner



      // Internal variable
         cPt2di         mSz;         ///<  Size of image
         bool           mVert;       ///< Is it vertical lines
         cACDI_Stat1Im  mLeft;     ///<  Statistic on left  (L/R => for vertical)
         cACDI_Stat1Im  mRight;    ///<  Statitic on Right rectangle
         cPt2di         mPC;         ///<  P Center
         cPt2di         mMulLeft;    ///<  Multiplier of coordinates to go on the left
         int            mXYC;        ///< Position of curent line
         int            mNum;        ///< Num of Current line
         
};

cCollecSpecArg2007 &  cAppli_CalcDiscIm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl 
          << Arg2007(mNameFile,"Name of file", {eTA2007::FileDirProj})
          << Arg2007(mCenters,"Centers" )
          << Arg2007(mDP0,"Upper left offset")
          << Arg2007(mDP1,"Bottom right offset")
      ;
}

cCollecSpecArg2007 & cAppli_CalcDiscIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
      ;
}

cAppli_CalcDiscIm::cAppli_CalcDiscIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mLeft        (true),
  mRight       (false)
{

}

void  cAppli_CalcDiscIm::CalcBox(cACDI_Stat1Im & aStat)
{
     cPt2di aMul = aStat.mLeft ? mMulLeft : cPt2di(1,1) ;
     aStat.mP0 = mPC +  MulCByC(aMul,mDP0);
     aStat.mP1 = mPC +  MulCByC(aMul,mDP1);
     MakeBox(aStat.mP0,aStat.mP1);
}
void  cAppli_CalcDiscIm::LoadOneIm(cACDI_Stat1Im & aStat)
{
     aStat.mIm.DIm().Resize(mSz);
     CalcBox(aStat);

     cDataFileIm2D aDFI = cDataFileIm2D::Create(mNameFile,true);

     aStat.mIm.Read(aDFI,aStat.mP0);

     tIm aIm =  mVert ? aStat.mIm : aStat.mIm.Transpose() ;
     tDIm & aDIm = aIm.DIm();
     cPt2di aSz = aIm.DIm().Sz();
     // StdOut()  <<  (aStat.mLeft ? "  L " : "  R ") <<   mP0 << " " << mP1 << "\n";

     aStat.mMoy = 0;
     int aNb =0 ;
     std::vector<double> aVVals;
     cPt2di aP;
     aStat.mVy.clear();
     for (aP.y()=0 ;  aP.y() <aSz.y() ; aP.y()++)
     {
        std::vector<double> aVx;
        for (aP.x()=0 ;  aP.x() <aSz.x() ; aP.x()++)
        {
           double  aV = aDIm.GetV(aP);
           aVx.push_back(aV);
           aStat.mMoy += aV;
           aNb++;
           aVVals.push_back(aV);
        }
        aStat.mVy.push_back(NonConstMediane(aVx));
     }
     aStat.mMoy /= aNb;
     aStat.mMed = NonConstMediane(aVVals);
}

void  cAppli_CalcDiscIm::InitC(int aXYC)
{
    mXYC = aXYC;
    mPC       =  mVert ? cPt2di(aXYC,0) :  cPt2di(0,aXYC);
}

void cAppli_CalcDiscIm::MakeOneLine(int aXYC,int aNum)
{
    mMulLeft  =  mVert ? cPt2di(-1,1)   :  cPt2di(1,-1);
    mNum = aNum;

    InitC(aXYC);
    // mXYC = aXYC;
    // mPC       =  mVert ? cPt2di(aXYC,0) :  cPt2di(0,aXYC);

    StdOut()  << "========== XYC " << mXYC << " Num " << mNum << " ==========\n";

    LoadOneIm(mLeft);
    LoadOneIm(mRight);

    std::vector<double> aVDif;
    int aNby = mLeft.mVy.size();
    for (int aY=0 ; aY<aNby ; aY++)
       aVDif.push_back( mRight.mVy[aY] - mLeft.mVy[aY]);

    std::sort(aVDif.begin(),aVDif.end());
    double aSomP = 0;
    double aSomDif = 0;
    for (int aY=0 ; aY<aNby ; aY++)
    {
         double aPds = std::min(aY,aNby-aY);
         aSomP += aPds;
         aSomDif += aPds * aVDif[aY];
    }
    aSomDif /= aSomP;
 
    StdOut() 
             << " DMoy=" <<  mRight.mMoy-mLeft.mMoy 
             << " DMed=" <<  mRight.mMed-mLeft.mMed  
             << " DMy=" <<  aVDif[aNby/2]
             << " DMy2=" <<  aSomDif
             << " LMoy="<<  mLeft.mMoy 
             << " RMoy=" << mRight.mMoy << "\n";
}

int cAppli_CalcDiscIm::Exe()
{
   MakeBox(mDP0,mDP1);
   mSz = mDP1 - mDP0;
   mVert =  mSz.x() < mSz.y();

   for (int aK=0 ; aK<int(mCenters.size())  ; aK++)
   {
       MakeOneLine(mCenters.at(aK),aK);
   }
   
   return EXIT_SUCCESS;
}



tMMVII_UnikPApli Alloc_CalcDiscIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CalcDiscIm(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCalcDiscIm
(
     "ImageCalcDisc",
      Alloc_CalcDiscIm,
      "Compute value of discontinuities in images",
      {eApF::ImProc},
      {eApDT::Image,eApDT::Console},
      {eApDT::Console},
      __FILE__
);

};

