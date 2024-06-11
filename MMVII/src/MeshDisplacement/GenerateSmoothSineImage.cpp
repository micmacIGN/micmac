#include "cMMVII_Appli.h"
#include "MMVII_Image2D.h"

/**
	\file GenerateSmoothSineImage.cpp

	\brief file for generating a smooth 2D image thanks to sine waves with
	equation a(x, y)xI(x, y) + b(x, y) wherer a(x, y), b(x, y) are sine functions
	and I(x, y) the input image
 **/

namespace MMVII
{
	/******************************************/
	/*                                        */
	/*    cAppli_GenerateSmoothSineImage      */
	/*                                        */
	/******************************************/

	class cAppli_GenerateSmoothSineImage : public cMMVII_Appli
	{
	public:
		typedef cIm2D<tREAL8> tIm;
		typedef cDataIm2D<tREAL8> tDIm;

		cAppli_GenerateSmoothSineImage(const std::vector<std::string> &aVArgs,
									   const cSpecMMVII_Appli &aSpec);

		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		void GenerateSmoothedSineImage(); // Generates smooth image with sine waves

	private:
		// ==== Mandatory args ====

		std::string mNameImage; // Name of the input image to deform

		// ==== Optional args ====

		tREAL8 mXAxisFrequency;		 // Applied frequency on x-axis of image
		tREAL8 mYAxisFrequency;		 // Applied frequency on y-axis of image
		tREAL8 mRadTranslationPhase; // Radiometry translation phase of cosine function
		tREAL8 mRadScalingPhase;	 // Radiometry scaling phase of sine function
		tREAL8 mAmplRadTranslation;	 // Amplitude of radiomety translation function
		tREAL8 mAmplRadScaling;		 // Amplitude of radiometry scaling function

		// ==== Internal variables ====

		tPt2di mSz;	   // Size of image
		tIm mImIn;	   // Memory representation of the image
		tDIm *mDImIn;  // Memory representation of the image
		tIm mImOut;	   // Memory representation of the image
		tDIm *mDImOut; // Memory representation of the image
	};

	cAppli_GenerateSmoothSineImage::cAppli_GenerateSmoothSineImage(
		const std::vector<std::string> &aVArgs,
		const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
										 mXAxisFrequency(0.1),
										 mYAxisFrequency(0.1),
										 mRadTranslationPhase(0),
										 mRadScalingPhase(0),
										 mAmplRadTranslation(2),
										 mAmplRadScaling(2),
										 mSz(tPt2di(1, 1)),
										 mImIn(mSz),
										 mDImIn(nullptr),
										 mImOut(mSz),
										 mDImOut(nullptr)
	{
	}

	cCollecSpecArg2007 &cAppli_GenerateSmoothSineImage::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl
			   << Arg2007(mNameImage, "Name of image to deform", {{eTA2007::FileImage}, {eTA2007::FileDirProj}});
	}

	cCollecSpecArg2007 &cAppli_GenerateSmoothSineImage::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{

		return anArgOpt
			   << AOpt2007(mXAxisFrequency, "XAxisFrequency", "Frequency used by wave functions on x-axis of image", {eTA2007::HDV})
			   << AOpt2007(mYAxisFrequency, "YAxisFrequency", "Frequency used by wave functions on y-axis of image", {eTA2007::HDV})
			   << AOpt2007(mRadTranslationPhase, "RadiometryTranslationPhase", "Phase used for radiometry translation of cosine function.", {eTA2007::HDV})
			   << AOpt2007(mRadScalingPhase, "RadiometryScalingPhase", "Phase used for radiometry scaling of sine function.", {eTA2007::HDV})
			   << AOpt2007(mAmplRadTranslation, "AmplRadiometryTranslation", "Amplitude used for radiometry translation cosine function", {eTA2007::HDV})
			   << AOpt2007(mAmplRadScaling, "AmplRadiometryScaling", "Amplitude used for radiometry scaling sine function", {eTA2007::HDV});
	}

	//------------------------------------------//

	void cAppli_GenerateSmoothSineImage::GenerateSmoothedSineImage()
	{
		for (const tPt2di &aOutPix : *mDImOut)
		{
			const tREAL8 aSineArgument = mXAxisFrequency * aOutPix.x() + mYAxisFrequency * aOutPix.y() + mRadScalingPhase;
			const tREAL8 aCosineArgument = mXAxisFrequency * aOutPix.x() + mYAxisFrequency * aOutPix.y() + mRadTranslationPhase;
			const tREAL8 aOutValue = mAmplRadScaling * std::sin(aSineArgument) * mDImIn->GetV(aOutPix) +
									 mAmplRadTranslation * std::cos(aCosineArgument);
			mDImOut->SetV(aOutPix, aOutValue);
		}
	}

	int cAppli_GenerateSmoothSineImage::Exe()
	{
		mImIn = tIm::FromFile(mNameImage);

		mDImIn = &mImIn.DIm();
		mSz = mDImIn->Sz();

		mImOut = tIm(mSz);
		mDImOut = &mImOut.DIm();

		GenerateSmoothedSineImage();

		mDImOut->ToFile("smooth_sine_image_frequency_" + ToStr(mXAxisFrequency) + "_" +
						ToStr(mYAxisFrequency) + "_phase_" +
						ToStr(mRadTranslationPhase) + "_" +
						ToStr(mRadScalingPhase) + "_amplitude_" +
						ToStr(mAmplRadTranslation) + "_" + ToStr(mAmplRadScaling) + ".tif");

		StdOut() << "Generated image is of size : " << mSz.x() << " x " << mSz.y() << std::endl;

		return EXIT_SUCCESS;
	}

	/********************************************/
	//              ::MMVII                     //
	/********************************************/

	tMMVII_UnikPApli Alloc_cAppli_GenerateSmoothSineImage(const std::vector<std::string> &aVArgs,
														  const cSpecMMVII_Appli &aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_GenerateSmoothSineImage(aVArgs, aSpec));
	}

	cSpecMMVII_Appli TheSpec_GenerateSmoothSineImage(
		"GenerateSmoothSineImage",
		Alloc_cAppli_GenerateSmoothSineImage,
		"Generates smooth sine 2D image",
		{eApF::ImProc}, // category
		{eApDT::Image}, // input
		{eApDT::Image}, // output
		__FILE__);

}; // namespace MMVII
