#include "MMVII_Image2D.h"

namespace MMVII
{

namespace {         // private

class cAppliTestAPBI : public cMMVII_Appli,
                        public cAppliParseBoxIm<tREAL4>
{
public :
    typedef cAppliParseBoxIm<tREAL4> tAPBI;
    typedef tAPBI::tIm               tImAPBI;
    typedef tAPBI::tDataIm           tDImAPBI;
    typedef cIm2D<tREAL4>            tImPx;
    typedef cIm2D<tU_INT1>           tImMasq;

    // typedef cIm2D

    // =========== Declaration ========
    // --- Method to be a MMVII application
    cAppliTestAPBI(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
    int Exe() override;
    // --- Method to be a cAppliParseBoxIm<tREAL4>
    int ExeOnParsedBox() override; ///< Action to exec for each box, When the appli parse a big file

    static std::string ImNameIn() { return cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() + "TestAPBI.tif"; }
    static std::string ImNameOut() { return cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() + "TestAPBI_out.tif"; }
    static tU_INT1 valFromBox(const cPixBox<2>& box) { return (box.P0().x() + box.P0().y()) % 255;}

private:
    cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
    cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;
};



cAppliTestAPBI::cAppliTestAPBI
    (
        const std::vector<std::string> &  aVArgs,
        const cSpecMMVII_Appli &          aSpec
        )  :
    cMMVII_Appli               (aVArgs,aSpec) ,
    cAppliParseBoxIm<tREAL4>   (*this,eForceGray::Yes,cPt2di(0,0),cPt2di(0,0),true)
{
}

cCollecSpecArg2007 & cAppliTestAPBI::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl;
//    return APBI_ArgObl(anArgObl); <-- should be used, but for bench, it's simplier to don't
}

cCollecSpecArg2007 & cAppliTestAPBI::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return APBI_ArgOpt(anArgOpt);
}


int cAppliTestAPBI::ExeOnParsedBox()
{
    cIm2D<tU_INT1>  aImTest(CurSzIn());
    tU_INT1 aVal = valFromBox(CurBoxIn());
    auto &imSrc = LoadI(CurBoxIn());
    for (const auto & aPix : aImTest.DIm())
        aImTest.DIm().SetVTrunc(aPix,(aVal + (int)imSrc.GetV(aPix)) % 255);
    APBI_WriteIm(ImNameOut(),aImTest);

    return EXIT_SUCCESS;
}

int cAppliTestAPBI::Exe()
{
    mNameIm = ImNameIn();
    APBI_ExecAll(true);
    return EXIT_SUCCESS;
}


/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_cAppliTestAPBI(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppliTestAPBI(aVArgs,aSpec));
}


}         // namespace private

cSpecMMVII_Appli  TheSpecAppliBenchAPBI
    (
        "TestAPBI",
        Alloc_cAppliTestAPBI,
        "Internal only !! Used by APBI bench",
        {eApF::Test,eApF::NoGui},
        {eApDT::None},
        {eApDT::Image},
        __FILE__
        );



void BenchAPBI(cParamExeBench & aParam)
{
    if (! aParam.NewBench("APBI")) return;


    cDataFileIm2D::Create(cAppliTestAPBI::ImNameIn(),eTyNums::eTN_U_INT1,
                          cPt2di(2000+RandUnif_M_N(-100,100),2000+RandUnif_M_N(-100,100)),
                          1
                          );
    auto aIm2d = cIm2D<tU_INT1>(cPt2di(2000+RandUnif_M_N(-100,100),2000+RandUnif_M_N(-100,100)));
    for (const auto& aPix : aIm2d.DIm()) {
        aIm2d.DIm().SetV(aPix,(aPix.x()+aPix.y()) % 255);
    }
    cDataFileIm2D  aDF = cDataFileIm2D::Create(cAppliTestAPBI::ImNameIn(),eTyNums::eTN_U_INT1,aIm2d.DIm().Sz(),1);
    aIm2d.Write(aDF,cPt2di(0,0));
    auto aBoxSize = cPt2di(200+RandUnif_M_N(-10,10),200+RandUnif_M_N(-10,10));

    cMMVII_Appli & anAp = cMMVII_Appli::CurrentAppli();

    anAp.ExeCallMMVII
        (
            TheSpecAppliBenchAPBI,
            anAp.StrObl(),
            anAp.StrOpt() << std::make_pair("SzTiles",cStrIO<cPt2di>::ToStr(aBoxSize))
            );


    auto mIm2d = cIm2D<tU_INT1>::FromFile(cAppliTestAPBI::ImNameOut());
    cParseBoxInOut<2> aPBIO =  cParseBoxInOut<2>::CreateFromSize(mIm2d.DIm(),aBoxSize);
    for (const auto & aPixI : aPBIO.BoxIndex()) {
        auto aBox = aPBIO.BoxOut(aPixI);
        tU_INT1 aVal = cAppliTestAPBI::valFromBox(aBox);
        for (const auto &aPix : aBox) {
            MMVII_INTERNAL_ASSERT_bench(mIm2d.DIm().GetV(aPix) == (aVal + (aPix.x() + aPix.y()) % 255)%255,"BenchAPBI failed");
        }
    }
    aParam.EndBench();
}

} // namespace MMVII

