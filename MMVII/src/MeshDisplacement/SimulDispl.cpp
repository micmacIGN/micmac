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
        typedef cIm2D<tREAL4> tImDispl;
        typedef cDataIm2D<tREAL4> tDImDispl;

        cAppli_SimulDispl(const std::vector<std::string> &aVArgs,
                          const cSpecMMVII_Appli &);

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

        tImDispl GenerateSmoothRandDispl();

    private:
        // ==   Mandatory args ====
        std::string mNameImage; // name of the input image to deform

        // ==   Optionnal args ====

        tREAL8 mAmplDef;                      // Amplitude of deformation
        bool mWithDisc;                       // Generate image with discontinuities
        bool mGenerateDispImageFromUserMaps;  // Generate displaced image from user defined displacement map
        std::string mUserDefinedDispXMapName; // Filename of user defined x-displacement map
        std::string mUserDefinedDispYMapName; // Filename of user defined y-displacement map

        // ==    Internal variables ====
        tImDispl mImIn;     // memory representation of the image
        tDImDispl *mDImIn;  // memory representation of the image
        cPt2di mSz;         // Size of image
        tImDispl mImOut;    // memory representation of the image
        tDImDispl *mDImOut; // memory representation of the image
    };

    cAppli_SimulDispl::cAppli_SimulDispl(
        const std::vector<std::string> &aVArgs,
        const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                         mAmplDef(2.0),
                                         mWithDisc(false),
                                         mGenerateDispImageFromUserMaps(false),
                                         mUserDefinedDispXMapName("UserDeplX.tif"),
                                         mUserDefinedDispYMapName("UserDeplY.tif"),
                                         mImIn(cPt2di(1, 1)),
                                         mDImIn(nullptr),
                                         mImOut(cPt2di(1, 1)),
                                         mDImOut(nullptr)
    {
    }

    cCollecSpecArg2007 &cAppli_SimulDispl::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNameImage, "Name of image to deform", {{eTA2007::FileImage}, {eTA2007::FileDirProj}});
    }

    cCollecSpecArg2007 &cAppli_SimulDispl::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {

        return anArgOpt
               << AOpt2007(mAmplDef, "Ampl", "Amplitude of deformation.", {eTA2007::HDV})
               << AOpt2007(mWithDisc, "WithDisc", "Do we add disconinuities.", {eTA2007::HDV})
               << AOpt2007(mGenerateDispImageFromUserMaps, "GenerateDispImageFromUserMaps",
                           "Generate post deformation image from user defined displacement maps.", {eTA2007::HDV})
               << AOpt2007(mUserDefinedDispXMapName, "UserDispXMapName", "Name of user defined x-displacement map.", {eTA2007::HDV, eTA2007::FileImage})
               << AOpt2007(mUserDefinedDispYMapName, "UserDispYMapName", "Name of user defined y-displacement map.", {eTA2007::HDV, eTA2007::FileImage});
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
            const tPt2dr aPixSE = ToR(aPix) / aDeZoom;
            aRes.DIm().SetV(aPix, aResSsEch.DIm().DefGetVBL(aPixSE, 0));
        }

        return aRes;
    }

    int cAppli_SimulDispl::Exe()
    {
        mImIn = tImDispl::FromFile(mNameImage);
        cDataFileIm2D aDescFile = cDataFileIm2D::Create(mNameImage, false);

        mDImIn = &mImIn.DIm();
        mSz = mDImIn->Sz();

        mImOut = tImDispl(mSz);
        mDImOut = &mImOut.DIm();

        for (const cPt2di &aPix : *mDImIn)
            mDImOut->SetV(aPix, 255 - mDImIn->GetV(aPix));

        tImDispl aImDispx = tImDispl(mSz);
        tImDispl aImDispy = tImDispl(mSz);
        tImDispl aImRegion = tImDispl(mSz);

        if (mGenerateDispImageFromUserMaps)
        {
            aImDispx = tImDispl::FromFile(mUserDefinedDispXMapName);
            aImDispy = tImDispl::FromFile(mUserDefinedDispYMapName);
        }
        else
        {
            aImDispx = GenerateSmoothRandDispl();
            aImDispy = GenerateSmoothRandDispl();
        }

        tDImDispl *aDImDispx = &aImDispx.DIm();
        tDImDispl *aDImDispy = &aImDispy.DIm();

        if (mWithDisc)
        {
            aImRegion = GenerateSmoothRandDispl();
            for (const cPt2di &aPix : aImRegion.DIm())
            {
                aImRegion.DIm().SetV(aPix, aImRegion.DIm().GetV(aPix) > 0);
                if (aImRegion.DIm().GetV(aPix))
                    std::swap(aDImDispx->GetReference_V(aPix),
                              aDImDispy->GetReference_V(aPix));
            }
            aImRegion.DIm().ToFile("Region.tif");
        }

        aDImDispx->ToFile("DeplX.tif");
        aDImDispy->ToFile("DeplY.tif");

        for (const tPt2di &aPix : mImOut.DIm())
        {
            const tREAL8 aDx = aDImDispx->GetV(aPix);
            const tREAL8 aDy = aDImDispy->GetV(aPix);
            const tPt2dr aPixR = ToR(aPix) - tPt2dr(aDx, aDy);

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
