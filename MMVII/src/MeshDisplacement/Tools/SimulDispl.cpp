#include "MMVII_Image2D.h"
#include "cMMVII_Appli.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"

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

		// Constructor
		cAppli_SimulDispl(const std::vector<std::string> &aVArgs,
						  const cSpecMMVII_Appli &aSpec);
		// Destructor
		~cAppli_SimulDispl();

		// Exe and argument methods
		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		// Generates smooth random displacement
        tImDispl GenerateSmoothRandDispl();
		// Loads user defined displacement maps or generates a smooth random one
		tImDispl LoadOrGenerateDisplacementMap(const std::string &aUserDefinedMapFileName);
		// Initialises interpolator with default arguments
        cDiffInterpolator1D *InitUserInterpolator();
		// Generates discontinuities on images if desired
		void GenerateDiscontinuity(tDImDispl *&aDImDispx, tDImDispl *&aDImDispy);
		// Builds displaced image from displacement maps
		void BuildDisplacedOutputImage(const bool aIsBilinearInterp, std::unique_ptr<const cDiffInterpolator1D> &anInterp,
									   tDImDispl *&aDImDispx, tDImDispl *&aDImDispy);

	private:

		// ==== Mandatory args ====

		std::string mNameImage; // Name of the input image to deform

		// ==== Optionnal args ====

        tREAL8 mAmplDef;                      	// Amplitude of deformation
        bool mWithDisc;                       	// Generate image with discontinuities
        bool mGenerateDispImageFromUserMaps;  	// Generate displaced image from user defined displacement map
        std::string mUserDefinedDispXMapName; 	// Filename of user defined x-displacement map
        std::string mUserDefinedDispYMapName; 	// Filename of user defined y-displacement map
        std::string mInterpName;                // Interpolator name (bicubic, sinc, ..)
        std::vector<std::string> mInterpParams; // Interpolator's parameters
		std::string mFileNameDispXMap;			// File name to use to save simulated x-displacement map
		std::string mFileNameDispYMap;			// File name to use to save simulated y-displacement map
		std::string mFileNameOutputImage;		// File name to use to save displaced image by displacement maps

		// ==== Internal variables ====

		tImDispl mImIn;		// Memory representation of the image
		tDImDispl *mDImIn;	// Memory representation of the image
		cPt2di mSz;			// Size of image
		tImDispl mImOut;	// Memory representation of the image
		tDImDispl *mDImOut; // Memory representation of the image
	};

    cAppli_SimulDispl::cAppli_SimulDispl(
        const std::vector<std::string> &aVArgs,
        const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                         mAmplDef(2.0),
                                         mWithDisc(false),
                                         mGenerateDispImageFromUserMaps(false),
                                         mUserDefinedDispXMapName("UserDeplX.tif"),
                                         mUserDefinedDispYMapName("UserDeplY.tif"),
                                         mInterpName("Cubic"),
                                         mInterpParams({"Tabul", "1000", "Cubic", "-0.5"}),
										 mFileNameDispXMap("DeplX.tif"),
										 mFileNameDispYMap("DeplY.tif"),
										 mFileNameOutputImage("image_post.tif"),
                                         mImIn(tPt2di(1, 1)),
                                         mDImIn(nullptr),
                                         mImOut(tPt2di(1, 1)),
                                         mDImOut(nullptr)
    {
    }

	cAppli_SimulDispl::~cAppli_SimulDispl()
	{
	}

	cCollecSpecArg2007 &cAppli_SimulDispl::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl << Arg2007(mNameImage, "Name of image to deform", {{eTA2007::FileImage}, {eTA2007::FileDirProj}});
	}

	cCollecSpecArg2007 &cAppli_SimulDispl::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{

        return anArgOpt << AOpt2007(mAmplDef, "Ampl", "Amplitude of deformation.", {eTA2007::HDV})
               			<< AOpt2007(mWithDisc, "WithDisc", "Do we add disconinuities.", {eTA2007::HDV})
               			<< AOpt2007(mInterpName,"Inter","Interpolator's name type, \"Bilinear\", \"Cubic\", \"SinCApod\", \"MMVIIK\" ", {eTA2007::HDV})
               			<< AOpt2007(mInterpParams,"InterParams","Interpolator's parameters", {eTA2007::HDV})
               			<< AOpt2007(mGenerateDispImageFromUserMaps, "GenerateDispImageFromUserMaps",
                           			"Generate post deformation image from user defined displacement maps.", {eTA2007::HDV})
               			<< AOpt2007(mUserDefinedDispXMapName, "UserDispXMapName", "Name of user defined x-displacement map.", {eTA2007::HDV, eTA2007::FileImage})
               			<< AOpt2007(mUserDefinedDispYMapName, "UserDispYMapName", "Name of user defined y-displacement map.", {eTA2007::HDV, eTA2007::FileImage})
						<< AOpt2007(mFileNameDispXMap, "FilenameToSaveDispXMap", "Filename to use to save x-displacement map.", {eTA2007::HDV})
						<< AOpt2007(mFileNameDispYMap, "FilenameToSaveDispYMap", "Filename to use to save y-displacement map.", {eTA2007::HDV})
						<< AOpt2007(mFileNameOutputImage, "FilenameToSaveDispXMap", "Filename to use to save output image displaced by displacement maps.", {eTA2007::HDV});
    }

	//================================================

    cDiffInterpolator1D *cAppli_SimulDispl::InitUserInterpolator()
    {
        std::vector<std::string> aParamDef;
        cDiffInterpolator1D *anInterp = nullptr;
        if (mInterpName == "Cubic")
            aParamDef = {"Tabul", "1000", "Cubic", "-0.5"};
        else if (mInterpName == "SinCApod")
            aParamDef = {"Tabul", "10000", "SinCApod", "10", "10"};
        else if (mInterpName == "MMVIIK")
            aParamDef = {"Tabul", "1000", "MMVIIK", "2"};
        else
            MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError, "A misspelled interpolator name ?");

        anInterp = cDiffInterpolator1D::AllocFromNames( IsInit(&mInterpParams) ? mInterpParams : aParamDef);

        return anInterp;
    }

	cAppli_SimulDispl::tImDispl cAppli_SimulDispl::GenerateSmoothRandDispl()
	{
		const tREAL8 aDeZoom = 10;
		const tREAL8 aNbBlob = 10;

		const tPt2di aSzRed = Pt_round_up(ToR(mSz) / aDeZoom);

		tImDispl aResSsEch(aSzRed);

		for (const tPt2di &aPix : aResSsEch.DIm())
			aResSsEch.DIm().SetV(aPix, RandUnif_C());

		ExpFilterOfStdDev(aResSsEch.DIm(), 5, Norm2(aSzRed) / aNbBlob);
		NormalizedAvgDev(aResSsEch.DIm(), 1e-10, mAmplDef);

		tImDispl aRes(mSz);
		for (const tPt2di &aPix : aRes.DIm())
		{
			const tPt2dr aPixSE = ToR(aPix) / aDeZoom;
			aRes.DIm().SetV(aPix, aResSsEch.DIm().DefGetVBL(aPixSE, 0));
		}

		return aRes;
	}

	cAppli_SimulDispl::tImDispl cAppli_SimulDispl::LoadOrGenerateDisplacementMap(const std::string &aUserDefinedMapFileName)
	{
		tImDispl aImDisp = (mGenerateDispImageFromUserMaps) ? tImDispl::FromFile(aUserDefinedMapFileName)
															: GenerateSmoothRandDispl();
		return aImDisp;
	}

	void cAppli_SimulDispl::GenerateDiscontinuity(tDImDispl *&aDImDispx, tDImDispl *&aDImDispy)
	{
		tImDispl aImRegion = tImDispl(mSz);
		aImRegion = GenerateSmoothRandDispl();
		for (const tPt2di &aPix : aImRegion.DIm())
		{
			aImRegion.DIm().SetV(aPix, aImRegion.DIm().GetV(aPix) > 0);
			if (aImRegion.DIm().GetV(aPix))
				std::swap(aDImDispx->GetReference_V(aPix),
						  aDImDispy->GetReference_V(aPix));
		}
		aImRegion.DIm().ToFile("Region.tif");
	}

	void cAppli_SimulDispl::BuildDisplacedOutputImage(const bool aIsBilinearInterp, std::unique_ptr<const cDiffInterpolator1D> &anInterp,
													  tDImDispl *&aDImDispx, tDImDispl *&aDImDispy)
	{
		for (const tPt2di &aPix : *mDImOut)
		{
			const tREAL8 aDx = aDImDispx->GetV(aPix);
			const tREAL8 aDy = aDImDispy->GetV(aPix);
			const tPt2dr aPixR = ToR(aPix) - tPt2dr(aDx, aDy);

			const bool aPixIn = (aIsBilinearInterp) ? mDImIn->InsideBL(aPixR) : mDImIn->InsideInterpolator(*anInterp, aPixR, 0);

            if (aPixIn)
            {
                const tREAL4 aValNew = (aIsBilinearInterp) ? mDImIn->DefGetVBL(aPixR, 0) : mDImIn->GetValueInterpol(*anInterp, aPixR);
                mDImOut->SetV(aPix, aValNew);
            }
        }
	}

	int cAppli_SimulDispl::Exe()
	{
		const bool aIsBilinearInterp = (mInterpName == "Bilinear");
		std::unique_ptr<const cDiffInterpolator1D> anInterp = (!aIsBilinearInterp) ? std::unique_ptr<const cDiffInterpolator1D>(InitUserInterpolator()) : nullptr;

		mImIn = tImDispl::FromFile(mNameImage);
		cDataFileIm2D aDescFile = cDataFileIm2D::Create(mNameImage, false);

		mDImIn = &mImIn.DIm();
		mSz = mDImIn->Sz();

		mImOut = tImDispl(mSz);
		mDImOut = &mImOut.DIm();

		for (const tPt2di &aPix : *mDImIn)
			mDImOut->SetV(aPix, 255 - mDImIn->GetV(aPix));

		tImDispl aImDispx = LoadOrGenerateDisplacementMap(mUserDefinedDispXMapName); 
        tImDispl aImDispy = LoadOrGenerateDisplacementMap(mUserDefinedDispYMapName);
        tImDispl aImRegion = tImDispl(mSz);

		tDImDispl *aDImDispx = &aImDispx.DIm();
		tDImDispl *aDImDispy = &aImDispy.DIm();

        if (mWithDisc)
			GenerateDiscontinuity(aDImDispx, aDImDispy);

		aDImDispx->ToFile(mFileNameDispXMap);
		aDImDispy->ToFile(mFileNameDispYMap);

		BuildDisplacedOutputImage(aIsBilinearInterp, anInterp, aDImDispx, aDImDispy);

		mDImOut->ToFile(mFileNameOutputImage, aDescFile.Type());

		StdOut() << "Size of image = [" << mImIn.DIm().Sz().x()
				 << ", " << mDImIn->SzY() << "]" << std::endl;

		return EXIT_SUCCESS;
	}

	/* =============================== */
	/*                                 */
	/*             MMVII               */
	/*                                 */
	/* =============================== */

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
