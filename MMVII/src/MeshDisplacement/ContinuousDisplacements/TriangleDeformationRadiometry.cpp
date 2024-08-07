#include "TriangleDeformationRadiometry.h"

/**
   \file TriangleDeformationRadiometry.cpp

   \brief file computing radiometric deformations
   between 2 images using triangular meshes.
**/

namespace MMVII
{
	/****************************************************/
	/*                                                  */
	/*       cAppli_TriangleDeformationRadiometry       */
	/*                                                  */
	/****************************************************/

	cAppli_TriangleDeformationRadiometry::cAppli_TriangleDeformationRadiometry(const std::vector<std::string> &aVArgs,
																			   const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
																												mNumberOfLines(1),
																												mNumberOfCols(1),
																												mShow(true),
																												mUseMultiScaleApproach(false),
																												mGenerateOutputImage(true),
																												mBuildRandomUniformGrid(false),
																												mUseMMV2Interpolators(true),
																												mInterpolArgs({"Tabul", "1000", "Cubic", "-0.5"}),
																												mSerialiseTriangleNodes(false),
																												mNameMultipleTriangleNodes("TriangulationNodes.xml"),
																												mInitialiseWithUserValues(true),
																												mInitialiseRadTrValue(0),
																												mInitialiseRadScValue(1),
																												mHardFreezeRadTranslation(false),
																												mHardFreezeRadScale(false),
																												mWeightRadTranslation(-1),
																												mWeightRadScale(-1),
																												mUserDefinedFolderNameToSaveResult(""),
																												mDisplayLastRadiometryValues(false),
																												mSigmaGaussFilterStep(1),
																												mNumberOfIterGaussFilter(3),
																												mNumberOfEndIterations(2),
																												mSzImPre(tPt2di(1, 1)),
																												mImPre(mSzImPre),
																												mDImPre(nullptr),
																												mSzImPost(tPt2di(1, 1)),
																												mImPost(mSzImPost),
																												mDImPost(nullptr),
																												mSzImOut(tPt2di(1, 1)),
																												mImOut(mSzImOut),
																												mDImOut(nullptr),
																												mDelTri({tPt2dr(0, 0)}),
																												mInterpolRad(nullptr),
																												mSysRadiometry(nullptr),
																												mEqRadiometryTri(nullptr)

	{
	}

	cAppli_TriangleDeformationRadiometry::~cAppli_TriangleDeformationRadiometry()
	{
		delete mSysRadiometry;
		delete mEqRadiometryTri;
		delete mInterpolRad;
	}

	cCollecSpecArg2007 &cAppli_TriangleDeformationRadiometry::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl
			   << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
			   << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
			   << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
			   << Arg2007(mNumberOfIterations, "Total number of scales to run in multi-scale approach or iterations if multi-scale approach is not applied in optimisation process.");
	}

	cCollecSpecArg2007 &cAppli_TriangleDeformationRadiometry::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{
		return anArgOpt
			   << AOpt2007(mNumberOfCols, "RandomUniformLawUpperBoundXAxis",
						   "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mNumberOfLines, "RandomUniformLawUpperBoundYAxis",
						   "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mGenerateOutputImage, "GenerateOutputImage",
						   "Whether to generate and save the output image with computed radiometry", {eTA2007::HDV})
			   << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
						   "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
			   << AOpt2007(mUseMMV2Interpolators, "UseMMV2Interpolators",
						   "Use MMVII interpolators instead of usual bilinear interpolation.", {eTA2007::HDV})
			   << AOpt2007(mInterpolArgs, "InterpolationArguments", "Arguments used for interpolation", {eTA2007::HDV})
			   << AOpt2007(mSerialiseTriangleNodes, "SerialiseTriangleNodes", "Whether to serialise triangle nodes to .xml file or not", {eTA2007::HDV})
			   << AOpt2007(mNameMultipleTriangleNodes, "NameOfMultipleTriangleNodes", "File name to use when saving all triangle nodes values to .xml file", {eTA2007::HDV})
			   << AOpt2007(mInitialiseWithUserValues, "InitialiseWithUserValues",
						   "Whether the user wishes or not to initialise unknowns with personalised values.", {eTA2007::HDV})
			   << AOpt2007(mInitialiseRadTrValue, "InitialeRadiometryTranslationValue",
						   "Value to use for initialising radiometry translation unknown values", {eTA2007::HDV})
			   << AOpt2007(mInitialiseRadScValue, "InitialeRadiometryScalingValue",
						   "Value to use for initialising radiometry scaling unknown values", {eTA2007::HDV})
			   << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
			   << AOpt2007(mHardFreezeRadTranslation, "FreezeRadTranslation",
						   "Whether to freeze radiometry translation factor in computation or not.", {eTA2007::HDV})
			   << AOpt2007(mHardFreezeRadScale, "FreezeRadScaling",
						   "Whether to freeze radiometry scaling factor in computation or not.", {eTA2007::HDV})
			   << AOpt2007(mWeightRadTranslation, "WeightRadiometryTranslation",
						   "A value to weight radiometry translation for soft freezing of coefficient.", {eTA2007::HDV})
			   << AOpt2007(mWeightRadScale, "WeightRadiometryScaling",
						   "A value to weight radiometry scaling for soft freezing of coefficient.", {eTA2007::HDV})
			   << AOpt2007(mUserDefinedFolderNameToSaveResult, "FolderNameToSaveResults",
						   "Folder name where to store produced results", {eTA2007::HDV})
			   << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
						   "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV})
			   << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep",
						   "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
						   "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
						   "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning});
	}

	void cAppli_TriangleDeformationRadiometry::LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber, const int aTotalNumberOfIters,
																							  const bool aNonEmptyFolderName)
	{
		//----------- allocate vec of obs :
		// 6 for ImagePre and 5 for ImagePost in linear gradient case and 6 in bilinear case
		const int aNumberOfObsRad = (mUseMMV2Interpolators) ? TriangleDisplacement_GradInterpol_NbObs : TriangleDisplacement_Bilin_NbObs;
		tDoubleVect aVObsRad(6 + aNumberOfObsRad, 0);

		//----------- extract current parameters
		tDenseVect aVCurSolRad = mSysRadiometry->CurGlobSol(); // Get current solution.

		tIm aCurPreIm = tIm(mSzImPre);
		tDIm *aCurPreDIm = nullptr;
		tIm aCurPostIm = tIm(mSzImPost);
		tDIm *aCurPostDIm = nullptr;

		mIsLastIters = false;

		if (mUseMultiScaleApproach)
			mIsLastIters = ManageDifferentCasesOfEndIterations(aIterNumber, mNumberOfIterations, mNumberOfEndIterations,
															   mIsLastIters, mImPre, mImPost, aCurPreIm, aCurPreDIm,
															   aCurPostIm, aCurPostDIm);
		else
		{
			LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", mImPre, mImPost);
			LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", mImPre, mImPost);
		}

		if (mUseMultiScaleApproach && !mIsLastIters)
		{
			aCurPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
			aCurPostIm = mImPost.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);

			aCurPreDIm = &aCurPreIm.DIm();
			aCurPostDIm = &aCurPostIm.DIm();

			mSigmaGaussFilter -= 1;

			const bool aSaveGaussImage = false;
			if (aSaveGaussImage)
				aCurPreDIm->ToFile("GaussFilteredImPre_iter_" + ToStr(aIterNumber) + ".tif");
		}

		//----------- Declaration of indicator of convergence
		size_t aNbOut = 0;	// Number of translated pixels out of image

		cStdStatRes aStarResObjRad;
		// Id of points
		int aNodeCounterRad = 0;
		std::unique_ptr<cMultipleTriangleNodesSerialiser> aVectorOfTriangleNodesRad = (mSerialiseTriangleNodes) ? cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(mNameMultipleTriangleNodes) : nullptr;

		if (mHardFreezeRadTranslation || mHardFreezeRadScale)
		{
			const int aSolStepRad = 2;
			for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
			{
				const tPt3di aIndicesOfTriKnotsRad = mDelTri.KthFace(aTr);

				tIntVect aVecIndRad;
				GetIndicesVector(aVecIndRad, aIndicesOfTriKnotsRad, 2);

				if (mHardFreezeRadTranslation)
				{
					const int aSolStartRadTranslation = 0;
					ApplyHardConstraintsToMultipleUnknowns(aSolStartRadTranslation, aSolStepRad,
														   aVecIndRad, aVCurSolRad, mSysRadiometry);
				}

				if (mHardFreezeRadScale)
				{
					const int aSolStartRadScaling = 1;
					ApplyHardConstraintsToMultipleUnknowns(aSolStartRadScaling, aSolStepRad,
														   aVecIndRad, aVCurSolRad, mSysRadiometry);
				}
			}
		}

		// Loop over all triangles to add the observations on each point
		for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
		{
			const tTri2dr aTriRad = mDelTri.KthTri(aTr);
			const cPt3di aIndicesOfTriKnotsRad = mDelTri.KthFace(aTr);

			const cTriangle2DCompiled aCompTri(aTriRad);

			std::vector<tPt2di> aVectorToFillWithInsidePixels;
			aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // Get pixels inside triangle

			//----------- Index of unknown, finds the associated pixels of current triangle knots
			tIntVect aVecIndRad;
			GetIndicesVector(aVecIndRad, aIndicesOfTriKnotsRad, 2);

			// current translation on radiometry 1st point of triangle
			const tREAL8 aCurRadTrPointA = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0)
																	  : LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0,
																																  aNodeCounterRad, aIndicesOfTriKnotsRad, true,
																																  aVectorOfTriangleNodesRad);
			// current scale on radiometry 1st point of triangle
			const tREAL8 aCurRadScPointA = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0)
																	  : LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0,
																	  														  aNodeCounterRad, aIndicesOfTriKnotsRad, false,
																															  aVectorOfTriangleNodesRad);
			// current translation on radiometry 2nd point of triangle
			const tREAL8 aCurRadTrPointB = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1)
																	  : LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1,
																																  aNodeCounterRad, aIndicesOfTriKnotsRad, true,
																																  aVectorOfTriangleNodesRad);
			// current scale on radiometry 2nd point of triangle
			const tREAL8 aCurRadScPointB = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1)
																	  : LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1,
																	  														  aNodeCounterRad + 1, aIndicesOfTriKnotsRad, false,
																															  aVectorOfTriangleNodesRad);
			// current translation on radiometry 3rd point of triangle
			const tREAL8 aCurRadTrPointC = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2)
																	  : LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2,
																																  aNodeCounterRad, aIndicesOfTriKnotsRad, true,
																																  aVectorOfTriangleNodesRad);
			// current scale on radiometry 3rd point of triangle
			const tREAL8 aCurRadScPointC = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2)
																	  : LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2,
																	  														  aNodeCounterRad + 2, aIndicesOfTriKnotsRad, false,
																															  aVectorOfTriangleNodesRad);
			aNodeCounterRad = (!mSerialiseTriangleNodes) ? 0 : aNodeCounterRad + 3;			

			if (mWeightRadTranslation > 0 || mWeightRadScale > 0)
			{
				const int aSolStepRad = 2;
				// Soft constraint radiometric translation
				if (mWeightRadTranslation > 0)
				{
					const int aSolStartRadTranslation = 0;
					ApplySoftConstraintToMultipleUnknown(aSolStartRadTranslation, aSolStepRad, aVecIndRad,
												 mSysRadiometry, aVCurSolRad, mWeightRadTranslation);
				}

				// Soft constraint radiometric scaling
				if (mWeightRadScale > 0)
				{
					const int aSolStartRadScaling = 1;
					ApplySoftConstraintToMultipleUnknown(aSolStartRadScaling, aSolStepRad, aVecIndRad,
												 mSysRadiometry, aVCurSolRad, mWeightRadScale);
				}
			}

			// Loop over all pixels inside triangle
			// size_t is necessary as there can be a lot of pixels in triangles
			for (size_t aFilledPixel = 0; aFilledPixel < aVectorToFillWithInsidePixels.size(); aFilledPixel++)
			{
				const cPtInsideTriangles aPixInsideTriangle = cPtInsideTriangles(aCompTri, aVectorToFillWithInsidePixels,
																				 aFilledPixel, aCurPreDIm, mInterpolRad);
				// Prepare for barycenter translation formula by filling aVObsRad with different coordinates
				FormalInterpBarycenter_SetObs(aVObsRad, 0, aPixInsideTriangle);

				// Radiometry translation of pixel by current radiometry translation of triangle knots
				const tREAL8 aRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aCurRadTrPointA,
																												aCurRadTrPointB,
																												aCurRadTrPointC,
																												aVObsRad);

				// Radiometry scaling of pixel by current radiometry scaling of triangle knots
				const tREAL8 aRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aCurRadScPointA,
																										aCurRadScPointB,
																										aCurRadScPointC,
																										aVObsRad);

				const tPt2dr aInsideTrianglePoint = aPixInsideTriangle.GetCartesianCoordinates();
				const tPt2di aEastTranslatedPoint = (mUseMMV2Interpolators) ? tPt2di(aInsideTrianglePoint.x(), aInsideTrianglePoint.y()) + tPt2di(1, 0) : tPt2di(0, 0);
				const tPt2di aSouthTranslatedPoint = (mUseMMV2Interpolators) ? tPt2di(aInsideTrianglePoint.x(), aInsideTrianglePoint.y()) + tPt2di(0, 1) : tPt2di(0, 0);

				const bool aPixInside = (mUseMMV2Interpolators) ? aCurPostDIm->InsideInterpolator(*mInterpolRad, aInsideTrianglePoint, 0) : (aCurPostDIm->InsideBL(tPt2dr(aEastTranslatedPoint.x(), aEastTranslatedPoint.y())) && aCurPostDIm->InsideBL(tPt2dr(aSouthTranslatedPoint.x(), aSouthTranslatedPoint.y())));
				if (aPixInside)
				{
					(mUseMMV2Interpolators) ?
											 // Prepare for application of linear gradient formula
						FormalGradInterpolTri_SetObs(aVObsRad, TriangleDisplacement_NbObs_ImPre, aInsideTrianglePoint,
													 aCurPostDIm, mInterpolRad)
											 :
											 // Prepare for application of bilinear formula
						FormalBilinTri_SetObs(aVObsRad, TriangleDisplacement_NbObs_ImPre, aInsideTrianglePoint, aCurPostDIm);

					// Now add observation
					mSysRadiometry->CalcAndAddObs(mEqRadiometryTri, aVecIndRad, aVObsRad);

					const tREAL8 aInterpolatedValue = (mUseMMV2Interpolators) ? aCurPostDIm->GetValueInterpol(*mInterpolRad, aInsideTrianglePoint) : aCurPostDIm->GetVBL(aInsideTrianglePoint);
					// Compute indicators
					const tREAL8 aRadiomValueImPre = aRadiometryScaling * aVObsRad[5] + aRadiometryTranslation;
					const tREAL8 aDif = aRadiomValueImPre - aInterpolatedValue; // residual
					aStarResObjRad.Add(std::abs(aDif));
				}
				else
					aNbOut++;
			}
		}

		// Update all parameter taking into account previous observation
		mSysRadiometry->SolveUpdateReset();

		// Save all triangle nodes to .xml file
		if (mSerialiseTriangleNodes && aVectorOfTriangleNodesRad != nullptr)
			aVectorOfTriangleNodesRad->MultipleNodesToFile(mNameMultipleTriangleNodes);

		if (mShow)
			StdOut() << aIterNumber + 1 << ", " << aStarResObjRad.Avg()
					 << ", " << aNbOut << std::endl;
	}

	void cAppli_TriangleDeformationRadiometry::GenerateOutputImage(const tDenseVect &aVFinalSol, const int aTotalNumberOfIterations,
																   const bool aNonEmptyFolderName)
	{
		// Initialise output image
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImOut, mDImOut, mSzImOut);

		tIm aLastPreIm = tIm(mSzImPre);
		tDIm *aLastPreDIm = nullptr;
		LoadPrePostImageAndData(aLastPreIm, aLastPreDIm, "pre", mImPre, mImPost);

		std::unique_ptr<cMultipleTriangleNodesSerialiser> aLastVectorOfTriangleNodesRad = nullptr;

		if (mUseMultiScaleApproach && !mIsLastIters)
		{
			aLastPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
			aLastPreDIm = &aLastPreIm.DIm();
		}

		int aLastNodeCounterRad = 0;

		for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
		{
			const tTri2dr aLastTriRad = mDelTri.KthTri(aLTr);
			const cPt3di aLastIndicesOfTriKnotsRad = mDelTri.KthFace(aLTr);

			const cTriangle2DCompiled aLastCompTri(aLastTriRad);

			std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
			aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

			tIntVect aLastVecIndRad;
			GetIndicesVector(aLastVecIndRad, aLastIndicesOfTriKnotsRad, 2);

			// Last radiometry translation of 1st point
			const tREAL8 aLastRadTrPointA = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0)
																	   : LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad,0,
																	   															   aLastNodeCounterRad, aLastIndicesOfTriKnotsRad, false,
																																   aLastVectorOfTriangleNodesRad);
			// Last radiometry scaling of 1st point
			const tREAL8 aLastRadScPointA = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0)
																	   : LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0,
																	   														   aLastNodeCounterRad, aLastIndicesOfTriKnotsRad, false,
																															   aLastVectorOfTriangleNodesRad);
			// Last radiometry translation of 2nd point
			const tREAL8 aLastRadTrPointB = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1)
																	   : LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1,
																	   															   aLastNodeCounterRad + 1, aLastIndicesOfTriKnotsRad, false,
																																   aLastVectorOfTriangleNodesRad);
			// Last radiometry scaling of 2nd point
			const tREAL8 aLastRadScPointB = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1)
																	   : LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1,
																	   														   aLastNodeCounterRad + 1, aLastIndicesOfTriKnotsRad, false,
																															   aLastVectorOfTriangleNodesRad);
			// Last radiometry translation of 3rd point
			const tREAL8 aLastRadTrPointC = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2)
																	   : LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2,
																	   															   aLastNodeCounterRad + 2, aLastIndicesOfTriKnotsRad, false,
																																   aLastVectorOfTriangleNodesRad);
			// Last radiometry scaling of 3rd point
			const tREAL8 aLastRadScPointC = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2)
																	   : LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2,
																	   														   aLastNodeCounterRad + 2, aLastIndicesOfTriKnotsRad, false,
																															   aLastVectorOfTriangleNodesRad);

			aLastNodeCounterRad = (!mSerialiseTriangleNodes) ? 0 : aLastNodeCounterRad + 3;

			const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

			for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
			{
				const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
																					 aLastFilledPixel, aLastPreDIm, mInterpolRad);

				// Radiometry translation of pixel by current radiometry translation
				const tREAL8 aLastRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aLastRadTrPointA,
																													aLastRadTrPointB,
																													aLastRadTrPointC,
																													aLastPixInsideTriangle);

				const tREAL8 aLastRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aLastRadScPointA,
																											aLastRadScPointB,
																											aLastRadScPointC,
																											aLastPixInsideTriangle);

				FillOutputImageRadiometry(aLastPixInsideTriangle, aLastRadiometryTranslation,
										  aLastRadiometryScaling, mDImOut);
			}
		}

		// Save output image with calculated radiometries to image file
		SaveOutputImageToFile(mDImOut, aNonEmptyFolderName, mUserDefinedFolderNameToSaveResult, "OutputImage",
							  mNumberPointsToGenerate, aTotalNumberOfIterations);
	}

	void cAppli_TriangleDeformationRadiometry::DoOneIterationRadiometry(const int aIterNumber, const int aTotalNumberOfIterations,
																		const tDenseVect &aVInitSol, const bool aNonEmptyFolderName)
	{
		LoopOverTrianglesAndUpdateParametersRadiometry(aIterNumber, aTotalNumberOfIterations, aNonEmptyFolderName); // Iterate over triangles and solve system

		tDenseVect aVFinalSol = mSysRadiometry->CurGlobSol();

		// Show final translation results and produce displacement maps
		if (aIterNumber == (aTotalNumberOfIterations - 1))
		{
			if (mGenerateOutputImage)
				GenerateOutputImage(aVFinalSol, aTotalNumberOfIterations, aNonEmptyFolderName);
			// Display last computed values of radiometry unknowns
			if (mDisplayLastRadiometryValues)
				DisplayFirstAndLastUnknownValuesAndComputeStatisticsTwoUnknowns(aVFinalSol, aVInitSol);
		}
	}

	//-----------------------------------------

	int cAppli_TriangleDeformationRadiometry::Exe()
	{
		// Read pre and post images and their sizes
		ReadImageFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
		ReadImageFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

		const bool aNonEmptyFolderName = CheckFolderExistence(mUserDefinedFolderNameToSaveResult);

		if (mUseMultiScaleApproach)
			mSigmaGaussFilter = mNumberOfIterations * mSigmaGaussFilterStep;

		if (mShow)
			StdOut() << "Iter, "
					 << "Diff, "
					 << "NbOut" << std::endl;

		DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
														mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);

		InitialiseInterpolationAndEquationRadiometry(mEqRadiometryTri, mInterpolRad, mInterpolArgs, mUseMMV2Interpolators);

		InitialiseWithUserValuesRadiometry(mDelTri.NbPts(), mSysRadiometry, mInitialiseWithUserValues,
										   mInitialiseRadTrValue, mInitialiseRadScValue);

		const tDenseVect aVInitSolRad = mSysRadiometry->CurGlobSol().Dup(); // Duplicate initial solution

		const int aTotalNumberOfIterations = GetTotalNumberOfIterations(mUseMultiScaleApproach, mNumberOfIterations,
																		mNumberOfEndIterations);

		for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
			DoOneIterationRadiometry(aIterNumber, aTotalNumberOfIterations, aVInitSolRad, aNonEmptyFolderName);

		return EXIT_SUCCESS;
	}

	/********************************************/
	//              ::MMVII                     //
	/********************************************/

	tMMVII_UnikPApli Alloc_cAppli_TriangleDeformationRadiometry(const std::vector<std::string> &aVArgs,
																const cSpecMMVII_Appli &aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_TriangleDeformationRadiometry(aVArgs, aSpec));
	}

	cSpecMMVII_Appli TheSpec_ComputeTriangleDeformationRadiometry(
		"ComputeTriangleDeformationRadiometry",
		Alloc_cAppli_TriangleDeformationRadiometry,
		"Compute radiometric deformation between images using triangular mesh",
		{eApF::ImProc}, // Category
		{eApDT::Image}, // Input
		{eApDT::Image}, // Output
		__FILE__);

}; // namespace MMVII
