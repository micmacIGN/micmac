#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Linear2DFiltering.h"
#include <bitset>


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
	void MakeImageSaddlePoints(const tDIm &) const;
        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============

                //  --

	std::vector<int>  mShowQuickSaddleF;

        // =========== Internal param ============
        tIm       mImIn;        ///< Input global image
        cPt2di    mSzIm;
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



/** Compute, as a flagt of bit, the set of Fremaan-8 neighboor that are > over a pixel,
 * for value==, the comparison is done on Y then X 
 */


template <class Type>  tU_INT1   FlagSup8Neigh(const cDataIm2D<Type> & aDIm,const cPt2di & aPt)
{
   Type aV0 = aDIm.GetV(aPt);
   tU_INT1 aResult = 0;
   //  for freeman 1,4  Y is positive, for freeman 0, X is positive
   for (int aK=0 ; aK<4 ; aK++)
   {
      if (aDIm.GetV(aPt+FreemanV8[aK]) >=  aV0)
         aResult |=  1<< aK;
   }
   for (int aK=4 ; aK<8 ; aK++)
   {
      if (aDIm.GetV(aPt+FreemanV8[aK]) >  aV0)
         aResult |=  1<< aK;
   }
   return aResult;
}

class cCCOfOrdNeigh
{
     public :
        bool mIsSup;  // is it a component with value > pixel
	int  mBit0;   // First neighboorr
	int  mBit1;   // last neighboor 
};

/*
void  ComputeCCOfeNeigh(std::vector<std::vector<cCCOfOrdNeigh>> & aVCC)
{
     for (size_t aFlag=0 ; aFlag<256 ; aFlag++)
     {
         for (size_t aBit=0 ; aBit<8 ; aBit++)
         {
              int aBitPrec = (7+aBit) % 8;
              bool ThisIs0 = (aFlag& (1<<aBit)) == 0;
              bool PrecIs0 = (aFlag& (1<<aBitPrec)) == 0;
	      if (ThisIs0!=PrecIs0)
	      {
                   int aBitNext = aBit+1;
	      }
	 }
     }
}

void FlagToVect(std::vector<bool> & aVBool,size_t aFlag)
{
}
*/

void ComputeNbbCCOfFlag(tU_INT1 * aTabFlag)
{
     for (size_t aFlag=0 ; aFlag<256 ; aFlag++)
     {
         tU_INT1 aNbCC = 0;
         for (size_t aBit=0 ; aBit<8 ; aBit++)
         {
              int aNextBit = (1+aBit) % 8;
              bool ThisIs0 = (aFlag& (1<<aBit)) == 0;
              bool NextIs0 = (aFlag& (1<<aNextBit)) == 0;

	      if (ThisIs0!=NextIs0)
	          aNbCC ++;
	 }
	 aTabFlag[aFlag] = aNbCC;
     }
}

int NbbCCOfFlag(tU_INT1 aFlag)
{
     static tU_INT1  TabFlag[256];
     static bool First=true;
     if (First)
     {
        First = false;
        ComputeNbbCCOfFlag(TabFlag);
     }
     return TabFlag[aFlag];
}
/*
template <class Type>  tREAL8  SadleDecision(const cDataIm2D<Type> & aDIm,const cPt2di & aPt)
{

}
*/


void ConnectedComponent
     (
         std::vector<cPt2di> & aVPts,
         cDataIm2D<tU_INT1>  & aDIm,
         const std::vector<cPt2di> & aNeighbourhood,
         const cPt2di& aSeed,
         int aMarqInit=1,
         int aNewMarq=0
     )
{
    aVPts.clear();
    if (aDIm.GetV(aSeed) != aMarqInit)
       return;

    aDIm.SetV(aSeed,aNewMarq);
    aVPts.push_back(aSeed);
    size_t aIndBottom = 0;

    while (aIndBottom!=aVPts.size())
    {
          cPt2di aP0 = aVPts[aIndBottom];
	  for (const auto & aDelta : aNeighbourhood)
	  {
              cPt2di aNeigh = aP0 + aDelta;
	      if (aDIm.GetV(aNeigh)==aMarqInit)
	      {
                  aDIm.SetV(aNeigh,aNewMarq);
                  aVPts.push_back(aNeigh);
	      }
	  }
	  aIndBottom++;
    }
}




int cAppliCheckBoardTargetExtract::IsTopoSaddlePoint(const cPt2di & aPt) const
{
   tREAL4 aV0 = mDImIn->GetV(aPt);
   std::vector<int> Sup0(8);

   for (int aK=0 ; aK<4 ; aK++)
   {
      Sup0[aK] = mDImIn->GetV(aPt+FreemanV8[aK]) >=  aV0;
   }
   for (int aK=4 ; aK<8 ; aK++)
   {
      Sup0[aK] = mDImIn->GetV(aPt+FreemanV8[aK]) > aV0;
   }

   int aNbDist = 0;
   for (int aK=0 ; aK<8 ; aK++)
   {
       if (Sup0[aK]!=Sup0[(aK+1)%8])
          aNbDist++;
   }


   return aNbDist;
}

void cAppliCheckBoardTargetExtract::MakeImageSaddlePoints(const tDIm & aDIm) const
{
    cRGBImage  aRGB = RGBImFromGray<tElem>(aDIm);
    // cRect2 aRectInt = aDIm->Dilate(-1);

    for (const auto & aPix : cRect2(aDIm.Dilate(-1)))
    {
       if (NbbCCOfFlag(FlagSup8Neigh(aDIm,aPix))>=4)
       {
          aRGB.SetRGBPix(aPix,cRGBImage::Red);
       }
    }
    aRGB.ToFile("Saddles.tif");
}
void cAppliCheckBoardTargetExtract::DoOneImage() 
{
    mImIn =  tIm::FromFile(mNameIm);
    mDImIn = &mImIn.DIm() ;
    mSzIm = mDImIn->Sz();

    StdOut() << "BITSET , 4 " << sizeof(std::bitset<4>) << " 8:" << sizeof(std::bitset<8>) << " 9:" << sizeof(std::bitset<8>) << "\n";
    StdOut() << "END READ IMAGE \n";

    SquareAvgFilter(*mDImIn,4,1,1);

    StdOut() << "END FILTER \n";

    // mDImIn->ToFile("tooooottttoto.tif");

    cRect2 aRectInt = mDImIn->Dilate(-1);
    int aNbSaddle=0;
    int aNbTot=0;

    // MakeImageSaddlePoints(*mDImIn);

    cIm2D<tU_INT1>  aImMasq(mSzIm,nullptr,eModeInitImage::eMIA_Null);
    cDataIm2D<tU_INT1> &  aDMasq = aImMasq.DIm();
    for (const auto & aPix : aRectInt)
    {
        if (NbbCCOfFlag(FlagSup8Neigh(*mDImIn,aPix)) >=4)
	{
            aImMasq.DIm().SetV(aPix,1);
	    aNbSaddle++;
	}
        aNbTot++;
    }

    int aNbCCSad=0;
    std::vector<cPt2di>  aVCC;
    const std::vector<cPt2di> & aV8 = Alloc8Neighbourhood();
    for (const auto& aPix : aDMasq)
    {
         if (aDMasq.GetV(aPix)==1)
	 {
             aNbCCSad++;
             ConnectedComponent(aVCC,aDMasq,aV8,aPix,1,0);
	     for (const auto & aPixCC : aVCC)
	     {
		     FakeUseIt(aPixCC);
	     }
	 }
    }


    StdOut() << "NBS " << (100.0*aNbSaddle)/aNbTot << " " <<  (100.0*aNbCCSad)/aNbTot << "\n";
    /*
    for (const auto & aPix : aRectInt)
    {
    }
    */
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
