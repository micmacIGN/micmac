#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Linear2DFiltering.h"


// Test git branch

namespace MMVII
{

/*  *********************************************************** */
/*                                                              */
/*              cAppliCheckBoardTargetExtract                   */
/*                                                              */
/*  *********************************************************** */


class cAppliCheckBoardTargetExtract : public cMMVII_Appli
{
     public :
        typedef tREAL4            tElem;
        typedef cIm2D<tElem>      tIm;
        typedef cDataIm2D<tElem>  tDIm;
        typedef cAffin2D<tREAL8>  tAffMap;


        cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	void DoOneImage() ;
        int  IsTopoSaddlePoint(const cPt2di & aPt) const;


        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============

                //  --

	std::vector<int>  mShowQuickSaddleF;

        // =========== Internal param ============
        tIm       mImIn;        ///< Input global image
	tDIm *    mDImIn;
};


/* *************************************************** */
/*                                                     */
/*              cAppliCheckBoardTargetExtract                  */
/*                                                     */
/* *************************************************** */

cAppliCheckBoardTargetExtract::cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mImIn            (cPt2di(1,1)),
   mDImIn           (nullptr)
{
}



cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameSpecif,"Name of target file")
   ;
}


cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<   AOpt2007(mShowQuickSaddleF,"ShowQSF","Vector to show quick saddle filters")
   ;
}
#if (0)
#endif

// cPt2di FreemanV8[8] = {{


int cAppliCheckBoardTargetExtract::IsTopoSaddlePoint(const cPt2di & aPt) const
{
   tREAL4 aV0 = mDImIn->GetV(aPt);
   bool Sup0[8];

   for (int aK=0 ; aK<4 ; aK++)
   {
      Sup0[aK] = mDImIn->GetV(aPt+FreemanV8[aK]) >  aV0;
   }
   for (int aK=4 ; aK<8 ; aK++)
   {
      Sup0[aK] = mDImIn->GetV(aPt+FreemanV8[aK]) >= aV0;
   }

   int aNbDist = 0;
   for (int aK=0 ; aK<8 ; aK++)
   {
       if (Sup0[aK]!=Sup0[(aK+1)%8])
          aNbDist++;
   }


   return aNbDist;
}

void cAppliCheckBoardTargetExtract::DoOneImage() 
{
    mImIn =  tIm::FromFile(mNameIm);
    mDImIn = &mImIn.DIm() ;


    new cIm2D<tU_INT1>(cPt2di(2,2));
    new cIm2D<tU_INT1>(cPt2di(2,2));

    // ExpFilterOfStdDev(*mDImIn,4,2.0);

    SquareAvgFilter(*mDImIn,4,1,1);
    mDImIn->ToFile("tooooottttoto.tif");

    cRect2 aRectInt = mDImIn->Dilate(-1);
    int aNbExtre=0;
    int aNbSaddle=0;
    int aNbStd=0;
    int aNbTot=0;
    for (const auto & aPix : aRectInt)
    {
        aNbTot++;
        int aNb = IsTopoSaddlePoint(aPix);
	if (aNb==0)  aNbExtre++;
	if (aNb==2)  aNbStd++;
	if (aNb==4)  aNbSaddle++;
    }

    StdOut() << "Saddle = " << aNbSaddle  << " %=" << (100.0*aNbSaddle)/double(aNbTot) << "\n";
}



int  cAppliCheckBoardTargetExtract::Exe()
{
   if (RunMultiSet(0,0))
   {
       return ResultMultiSet();
   }

   DoOneImage();

   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CheckBoardCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCheckBoardTargetExtract(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCheckBoardTarget
(
     "CodedTargetCheckBoardExtract",
      Alloc_CheckBoardCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


};
