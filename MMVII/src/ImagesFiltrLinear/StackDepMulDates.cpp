#include "MMVII_Image2D.h"
#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Interpolators.h"


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*              cAppli_StackDep                          */
/*                                                      */
/* ==================================================== */

class cOneDepOfStack
{
    public :
       typedef cIm2D<tREAL4> tIm;
       typedef cDataIm2D<tREAL4> tDIm;

       cOneDepOfStack
       (
	   int aK1,int aK2,
	   cIm2D<tU_INT1>      aMasqNoDepl,
           const std::string & aP1,
           const std::string & aP2,
           const std::string & aScore
       );

       int    mK1;
       int    mK2;
       tIm    mImPx1;
       tDIm & mDImPx1;
       tIm    mImPx2;
       tDIm & mDImPx2;
       tIm    mImScore;
       tDIm & mDImScore;
};

cOneDepOfStack::cOneDepOfStack
(
     int aK1,
     int aK2,
     cIm2D<tU_INT1>      aMasqNoDepl,
     const std::string & aP1,
     const std::string & aP2,
     const std::string & aScore
) :
    mK1      (aK1),
    mK2      (aK2),
    mImPx1   (tIm::FromFile(aP1)),
    mDImPx1  (mImPx1.DIm()),
    mImPx2   (tIm::FromFile(aP2)),
    mDImPx2  (mImPx2.DIm()),
    mImScore (tIm::FromFile(aScore)),
    mDImScore(mImScore.DIm())
{

     // preprocessing average =0 
     {
         cWeightAv<tREAL8,cPt2dr> aAvgDep;
         const auto & aDMask = aMasqNoDepl.DIm();
         for (const auto &  aPix : aDMask)
         {
              if (aDMask.GetV(aPix))
	      {
                  aAvgDep.Add(1.0,cPt2dr(mDImPx1.GetV(aPix),mDImPx2.GetV(aPix)));
	      }
         }
	 cPt2dr aAvg = aAvgDep.Average();

         for (const auto &  aPix : aDMask)
         {
             mDImPx1.AddVal(aPix,-aAvg.x());
             mDImPx2.AddVal(aPix,-aAvg.y());
         }
     }
}




class cAppli_StackDep : public cMMVII_Appli
{
     public :

        typedef cIm2D<tREAL4> tIm;
        typedef cDataIm2D<tREAL4> tDIm;

        cAppli_StackDep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
	int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :

	std::string NameIm(int aK1,int aK2,const std::string & aPost) const;
	std::string NamePx1(int aK1,int aK2) const;
	std::string NamePx2(int aK1,int aK2) const;
	std::string NameScore(int aK1,int aK2) const;

	void Do1Pixel(const cPt2di & aPix);


        std::vector<std::string>     mArgInterpol;
        cDiffInterpolator1D *        mInterpol;
	bool                         mDoL2;
	cLinearOverCstrSys<tREAL8>*  mSys;
        int                          mNbIm ;
        int                          mNbVar ;
        int                          mKRef ;
	std::string                  mSpecImIn;
	std::vector<cOneDepOfStack>  mVecDepl;
};

cAppli_StackDep::cAppli_StackDep(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec):
	cMMVII_Appli   (aVArgs,aSpec),
	mNbIm          (10)
{
}

cCollecSpecArg2007 & cAppli_StackDep::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return anArgObl
              << Arg2007(mArgInterpol,"Argument interpolator ")
              << Arg2007(mDoL2,"L2/L1 compensation")
/*
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
*/
           ;
}

cCollecSpecArg2007 & cAppli_StackDep::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return anArgOpt;
}


std::string cAppli_StackDep::NameIm(int aK1,int aK2,const std::string & aPost) const
{
	return ToStr(aK1) + ToStr(aK2) + "_" + aPost + ".tif";
}

std::string cAppli_StackDep::NamePx1(int aK1,int aK2) const {return NameIm(aK1,aK2,"px1");}
std::string cAppli_StackDep::NamePx2(int aK1,int aK2) const {return NameIm(aK1,aK2,"px2");}
std::string cAppli_StackDep::NameScore(int aK1,int aK2) const {return NameIm(aK1,aK2,"corrscore");}


void  cAppli_StackDep::Do1Pixel(const cPt2di & aPix)
{
	mSys->Reset();
}




int cAppli_StackDep::Exe()
{
    mInterpol = cDiffInterpolator1D::AllocFromNames(mArgInterpol);
    mNbVar = mNbIm - 1;
    mKRef = mNbIm /2 ;

    mSys = mDoL2 ? new cLeasSqtAA<tREAL8>(mNbVar) : AllocL1_Barrodale<tREAL8>(mNbVar);

    cIm2D<tU_INT1> aMasq = cIm2D<tU_INT1>::FromFile("mask.tif");
    for (int aK1=0 ; aK1<mNbIm ; aK1++)
    {
        for (int aK2=0 ; aK2<mNbIm ; aK2++)
        {
            if (ExistFile(NamePx1(aK1,aK2)) && ExistFile(NamePx2(aK1,aK2)) && ExistFile(NameScore(aK1,aK2)))
            {
                cOneDepOfStack aDepl(aK1,aK2,aMasq,NamePx1(aK1,aK2),NamePx2(aK1,aK2) ,NameScore(aK1,aK2));
		mVecDepl.push_back(aDepl);
            }
        }
    }

    delete mInterpol;
    delete mSys;
    return EXIT_SUCCESS;
};




/* ==================================================== */
/*                                                      */
/*                                                      */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_StackDep(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_StackDep(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_StackDep
(
     "DeplStack",
      Alloc_StackDep,
      "Stack a serie multi date displacment",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);



}; // MMVII




