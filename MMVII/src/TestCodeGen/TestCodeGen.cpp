#include "TestCodeGenTpl.h"

#include "include/CodeGen_IncludeAll.h"

#include "Formula_Fraser_Test.h"
#include "Formula_Primitives_Test.h"
#include "Formula_Ratkowskyresidual.h"
#include "Formula_Eqcollinearity.h"

#include "include/MMVII_all.h"


namespace CG = CodeGen;

/* ==================================================== */
/*                                                      */
/*          cAppli_TestCodeGen                          */
/*                                                      */
/* ==================================================== */
namespace MMVII
{

class  TestPrimitives : public cCodeGenTest<cPrimitivesTest,CG::PrimitivesTest<double>,CG::PrimitivesTestDevel<double>>
{
public:
    TestPrimitives(size_t nbTest) : cCodeGenTest<cPrimitivesTest,CG::PrimitivesTest<double>,CG::PrimitivesTestDevel<double>>(nbTest)
    {
        mVUk[0] = 1;
        mVUk[1] = 2;
    }
};




class cAppli_TestCodeGen : public cMMVII_Appli
{
     public :
        cAppli_TestCodeGen(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override {return anArgObl;}
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
        int mSizeBuf;
        int mThreads;
};


cCollecSpecArg2007 & cAppli_TestCodeGen::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt << AOpt2007(mSizeBuf,"b","Buffer Size",{})
                    << AOpt2007(mThreads,"t","Nb Thread Max",{});
}

cAppli_TestCodeGen::cAppli_TestCodeGen(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),mSizeBuf(0),mThreads(0)
{
}


int cAppli_TestCodeGen::Exe()
{
//    TestFraser test(1000000);
//   TestPrimitives test(1000000);
//    TestRatkowsky test(1000000);

    std::vector<double>  uk(cEqCoLinearity<cTplPolDist<7>>::NbUk(),0);
    std::vector<double> obs(cEqCoLinearity<cTplPolDist<7>>::NbObs(),0);
    uk[11] = 1.0;
    // In obs, we set the current matrix to Id
    obs[0] = obs[4] = obs[8] = 1;

    // Here we initiate with "peferct" projection, to check something
    // Fix X,Y,Z
    uk[0] = 1.0;
    uk[1] = 2.0;
    uk[2] = 10.0;
    // Fix I,J (pixel projection)
    obs[ 9] = 0.101;
    obs[10] = 0.2;

//     cCodeGenTest<cFraserCamColinear,CG::Fraser<double>,CG::FraserDevel<double>> test(1000000);
    cCodeGenTest<cEqCoLinearity<cTplPolDist<7>>,CG::EqColLinearityXYPol_Deg7<double>,CG::EqColLinearityXYPol_Deg7Devel<double>> test(100000);
//     cCodeGenTest<cEqCoLinearity<cTplFraserDist>,CG::EqColLinearityFraser<double>,CG::EqColLinearityFraserDevel<double>> test(1000000);
     test.mVUk = uk;
     test.mVObs =  obs;
     test.checkAll();
    if (mSizeBuf == 0 || mThreads==0) {
        test.benchMark();
    } else {
        test.oneShot(mThreads,mSizeBuf);
    }
   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestCodeGen(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TestCodeGen(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpecTestCodeGen
(
     "TestCodeGen",
      Alloc_TestCodeGen,
      "Code Generator test",
      {eApF::Perso},
      {eApDT::FileSys},
      {eApDT::Xml},
      __FILE__
);

}



