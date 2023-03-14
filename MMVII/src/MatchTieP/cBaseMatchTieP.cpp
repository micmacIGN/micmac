#include "MatchTieP.h"

namespace MMVII
{


/* =============================================== */
/*                                                 */
/*               cBaseMatchTieP                   */
/*                                                 */
/* =============================================== */

cBaseMatchTieP::cBaseMatchTieP(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli  (aVArgs,aSpec,{eSharedPO::eSPO_CarPI})
{
}

/// Im1 and Im2 are the only mandatory args
cCollecSpecArg2007 & cBaseMatchTieP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIm1,"Name of input file")
          <<   Arg2007(mNameIm2,"Name of input file")
   ;
}

/// This was a test to check that  optional args are "inherited"
cCollecSpecArg2007 & cBaseMatchTieP::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mTestOPtBase,"TOBase","Test Optionnal Base",{eTA2007::HDV})
   ;
}

/**   Once the args have been initiazed by virtual method (not in constrtuctor !)  ArgObl-ArgOpt
    we can  read the 
*/
void  cBaseMatchTieP::PostInit()
{
   // Parse all the possible labels
   for (int aKTy=0 ; aKTy<int(eTyPyrTieP::eNbVals) ; aKTy ++)
   {
       eTyPyrTieP aType = eTyPyrTieP(aKTy);
       // Parse Min and Max hypothesis
       for (int aKMax=0 ; aKMax<2 ; aKMax++)
       {
            bool IsMax = (aKMax==0);
            std::string aNamePC1 =  StdNamePCarIn(mNameIm1,aType,IsMax);
            std::string aNamePC2 =  StdNamePCarIn(mNameIm2,aType,IsMax);
            // Files do not necessary exist 
            if (ExistFile(aNamePC1) && ExistFile(aNamePC2))
            {
                // Create empty set
                mVSAPc1.push_back(cSetAimePCAR());
                mVSAPc2.push_back(cSetAimePCAR());

                // Call method that will use serialization to init objects
                mVSAPc1.back().InitFromFile(aNamePC1);
                mVSAPc2.back().InitFromFile(aNamePC2);
                // StdOut() << " " << mSAPc1.Ampl2N() << "\n";
                StdOut() << aNamePC1 << " Nb=" << mVSAPc1.back().VPC().size() << "\n"  
                         << aNamePC2 << "\n\n";
            }
       }
   }
}


};
