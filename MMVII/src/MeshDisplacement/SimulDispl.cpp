#include "MMVII_Image2D.h"
#include "cMMVII_Appli.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Tpl_Images.h"

/**
   \file SimulDispl.cpp

   \brief file for generating simulation of smooth displacement
 **/

namespace MMVII
{
    /* ======================================== */
    /*                                          */
    /*          cAppli_SimulDispl       	    */
    /*                                          */
    /* ======================================== */

    class cAppli_SimulDispl : public cMMVII_Appli
    {
    public:
        typedef cIm2D<tREAL4> tIm;
        typedef cDataIm2D<tREAL4> tDIm;
        typedef cIm2D<tREAL4> tImDispl;
        typedef cDataIm2D<tREAL4> tDImDispl;

        cAppli_SimulDispl(const std::vector<std::string> &aVArgs, 
                          const cSpecMMVII_Appli &);

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

    private:
        tImDispl GenerateSmoothRandDispl();

        // ==   Mandatory args ====
        std::string mNameImage; ///< name of the input image to deform

        // ==   Optionnal args ====

        tREAL8 mAmplDef;
        bool mWithDisc;

        // ==    Internal variables ====
        tIm mImIn;    ///<  memory representation of the image
        tDIm *mDImIn; ///<  memory representation of the image
        cPt2di mSz;
        tIm mImOut;    ///<  memory representation of the image
        tDIm *mDImOut; ///<  memory representation of the image
    };

    cAppli_SimulDispl::cAppli_SimulDispl(
        const std::vector<std::string> &aVArgs,
        const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                         mAmplDef(2.0),
                                         mWithDisc(false),
                                         mImIn(cPt2di(1, 1)),
                                         mDImIn(nullptr),
                                         mImOut(cPt2di(1, 1)),
                                         mDImOut(nullptr)
    {
    }

    cCollecSpecArg2007 &cAppli_SimulDispl::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNameImage, "Name of image to deform", {{eTA2007::FileImage}, 
                                                                  {eTA2007::FileDirProj}});
    }

    cCollecSpecArg2007 &cAppli_SimulDispl::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {

        return anArgOpt
               << AOpt2007(mAmplDef, "Ampl", "Amplitude of deformation", {eTA2007::HDV})
               << AOpt2007(mWithDisc, "WithDisc", "Do we add disconinuities", {eTA2007::HDV});
    }

    //================================================

    cAppli_SimulDispl::tImDispl cAppli_SimulDispl::GenerateSmoothRandDispl()
    {
        const tREAL8 aDeZoom = 10.0;
        const tREAL8 aNbBlob = 10.0;

        const cPt2di aSzRed = Pt_round_up(ToR(mSz) / aDeZoom);

        tImDispl aResSsEch(aSzRed);

        for (const cPt2di &aPix : aResSsEch.DIm())
            aResSsEch.DIm().SetV(aPix, RandUnif_C());

        ExpFilterOfStdDev(aResSsEch.DIm(), 5, Norm2(aSzRed) / aNbBlob);
        NormalizedAvgDev(aResSsEch.DIm(), 1e-10, mAmplDef);

        tImDispl aRes(mSz);
        for (const cPt2di &aPix : aRes.DIm())
        {
            const cPt2dr aPixSE = ToR(aPix) / aDeZoom;
            aRes.DIm().SetV(aPix, aResSsEch.DIm().DefGetVBL(aPixSE, 0));
        }

        return aRes;
    }

    int cAppli_SimulDispl::Exe()
    {
        mImIn = tIm::FromFile(mNameImage);
        cDataFileIm2D aDescFile = cDataFileIm2D::Create(mNameImage, false);

        mDImIn = &mImIn.DIm();
        mSz = mDImIn->Sz();

        mImOut = tIm(mSz);
        mDImOut = &mImOut.DIm();

        for (const cPt2di &aPix : *mDImIn)
            mDImOut->SetV(aPix, 255 - mDImIn->GetV(aPix));

        tImDispl aImDispx = GenerateSmoothRandDispl();
        tImDispl aImDispy = GenerateSmoothRandDispl();
        tImDispl aImRegion = GenerateSmoothRandDispl();

        if (mWithDisc)
        {
            for (const cPt2di &aPix : aImRegion.DIm())
            {
                aImRegion.DIm().SetV(aPix, aImRegion.DIm().GetV(aPix) > 0);
                if (aImRegion.DIm().GetV(aPix))
                    std::swap(aImDispx.DIm().GetReference_V(aPix),
                              aImDispy.DIm().GetReference_V(aPix));
           }
        }

        aImDispx.DIm().ToFile("DeplX.tif");
        aImDispy.DIm().ToFile("DeplY.tif");
        aImRegion.DIm().ToFile("Region.tif");

        for (const auto &aPix : mImOut.DIm())
        {
            const tREAL8 aDx = aImDispx.DIm().GetV(aPix);
            const tREAL8 aDy = aImDispy.DIm().GetV(aPix);
            const cPt2dr aPixR = ToR(aPix) - cPt2dr(aDx, aDy);

            mDImOut->SetV(aPix, mDImIn->DefGetVBL(aPixR, 0));
        }

        mDImOut->ToFile("image_post.tif", aDescFile.Type());

        StdOut() << "Size of image = [" << mImIn.DIm().Sz().x() 
                 << ", " << mDImIn->SzY() << "]" << std::endl;
        return EXIT_SUCCESS;
    }

    /* ====================================== */
    /*                                        */
    /*               MMVII                    */
    /*                                        */
    /* ====================================== */

    tMMVII_UnikPApli Alloc_SimulDispl(const std::vector<std::string> &aVArgs, 
                                      const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_SimulDispl(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_SimulDispl(
        "SimulDispl",
        Alloc_SimulDispl,
        "Generate smooth displacement and deformed image",
        {eApF::ImProc},
        {eApDT::Image},
        {eApDT::Image},
        __FILE__);

}; // MMVII
