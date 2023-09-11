#include "MMVII_Bench.h"
#include "MMVII_Class4Bench.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_Geom2D.h"
#include "MMVII_PCSens.h"
#include "Serial.h"


/** \file DemoSerial.cpp
    \brief file for generating spec+samples of serialization

*/

namespace MMVII
{
/*
class cAppliSpecSerial : public cMMVII_Appli
{
     public :

        cAppliSpecSerial(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
};


cAppliSpecSerial::cAppliSpecSerial
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli   (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliSpecSerial::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<   Arg2007(mSpec.mType  ,"Type among enumerated values",{AC_ListVal<eTyCodeTarget>()})
           <<   Arg2007(mSpec.mNbBits,"Number of bits for the code")
   ;
}


cCollecSpecArg2007 & cAppliGenerateEncoding::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
                  anArgOpt
               << AOpt2007(mSpec.mMinHammingD,"MinHamD","Minimal Hamming Dist (def depend of type)")
               << AOpt2007(mSpec.mMaxRunL,"MaxRunL","Maximal Run Length (def depend of type)")
               << AOpt2007(mSpec.mFreqCircEq,"FreqCircEq","Freq for generating circular permuts (conventionnaly 0->highest) (def depend of type)")
               << AOpt2007(mSpec.mParity,"Parity","Parity check , 1 odd, 2 even, 3 all (def depend of type)")
               << AOpt2007(mSpec.mMaxNb,"MaxNb","Max number of codes",{eTA2007::HDV})
               << AOpt2007(mSpec.mBase4Name,"Base4N","Base for name",{eTA2007::HDV})
               << AOpt2007(mSpec.mNbDigit,"NbDig","Number of digit for name (default depend of max num & base)")
               << AOpt2007(mSpec.mUseHammingCode,"UHC","Use Hamming code")
               << AOpt2007(mSpec.mPrefix,"Prefix","Prefix for output files")
               << AOpt2007(mMiror,"Mir","Unify mirro codes")
          ;
}

int  cAppliGenerateEncoding::Exe()
{
   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_GenerateEncoding(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenerateEncoding(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecGenerateEncoding
(
     "CodedTargetGenerateEncoding",
      Alloc_GenerateEncoding,
      "Generate en encoding for coded target, according to some specification",
      {eApF::CodedTarget},
      {eApDT::None},
      {eApDT::Xml},
      __FILE__
);

*/




};
