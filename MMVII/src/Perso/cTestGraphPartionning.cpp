
#include "include/MMVII_2Include_Serial_Tpl.h"
#include<map>

/** \file 
    \brief 
*/


namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_TestGraphPart                             */
/*                                                      */
/* ==================================================== */


/** Application for concatenating videos */

class cAppli_TestGraphPart : public cMMVII_Appli
{
     public :
        cAppli_TestGraphPart(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
         bool        mExec;
         bool        mAppend;
         std::string mNameFoF;     ///< name of file of files
         std::string mNameResult;  ///< name of Resulting media
         bool        mVideoMode;       ///< is it video, change def options

         std::string mOptions;     ///< is it video, change options

         size_t  mNbVertex;
         size_t  mNbClass;
         cIm1D<tINT4>       mGTClass;
         cDataIm1D<tINT4>*  mDGTC;
};



cCollecSpecArg2007 & cAppli_TestGraphPart::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
         << Arg2007(mNbVertex,"Number of vertex")

   ;
}

cCollecSpecArg2007 & cAppli_TestGraphPart::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mNbClass,"NbClass","Number of classes)",{eTA2007::HDV})
   ;
}


cAppli_TestGraphPart::cAppli_TestGraphPart
(
      const std::vector<std::string> &  aVArgs,
      const cSpecMMVII_Appli & aSpec
) :
  cMMVII_Appli    (aVArgs,aSpec),
  mNbClass        (5),
  mGTClass        (1)
{
}

int cAppli_TestGraphPart::Exe()
{
   mGTClass =  cIm1D<tINT4>(mNbVertex);
   mDGTC = &(mGTClass.DIm());

   for (size_t aKv = 0 ; aKv < mNbVertex ; aKv++)
   {
       size_t  aClass = aKv % mNbClass;

        mDGTC->SetVa(aKv,aClass);
   }

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestGraphPart(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TestGraphPart(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecTestGraphPart
(
     "TestGraphPart",
      Alloc_TestGraphPart,
      "This command is to make some test on graph partionning",
      {eApF::Perso},
      {eApDT::Media},
      {eApDT::Media},
      __FILE__
);

};

