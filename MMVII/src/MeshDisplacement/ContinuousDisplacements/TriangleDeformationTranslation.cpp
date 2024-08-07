#include "TriangleDeformationTranslation.h"

/**
   \file TriangleDeformationTranslation.cpp

   \brief file for computing 2D translation between 2 images
   thanks to triangular meshes.
**/

namespace MMVII
{
	/************************************************/
	/*                                              */
	/*    cAppli_TriangleDeformationTranslation     */
	/*                                              */
	/************************************************/

	cAppli_TriangleDeformationTranslation::cAppli_TriangleDeformationTranslation(const std::vector<std::string> &aVArgs,
																				 const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
																												  mNumberOfLines(1),
																												  mNumberOfCols(1),
																												  mShow(true),
																												  mUseMultiScaleApproach(false),
																												  mBuildRandomUniformGrid(false),
																												  mUseMMV2Interpolators(false),
																												  mInterpolArgs({"Tabul", "1000", "Cubic", "-0.5"}),
																												  mSerialiseTriangleNodes(false),
																												  mNameMultipleTriangleNodes("TriangulationNodes.xml"),
																												  mInitialiseTranslationWithPreviousExecution(false),
																												  mInitialiseWithUserValues(true),
																												  mInitialiseXTranslationValue(0),
																												  mInitialiseYTranslationValue(0),
																												  mInitialiseWithMMVI(false),
																												  mNameInitialDepX("InitialDispXMap.tif"),
																												  mNameInitialDepY("InitialDispYMap.tif"),
																												  mNameIntermediateDepX("IntermediateDispXMap.tif"),
																												  mNameIntermediateDepY("IntermediateDispYMap.tif"),
																												  mNameCorrelationMaskMMVI("CorrelationMask.tif"),
																												  mIsFirstExecution(false),
																												  mSigmaGaussFilterStep(1),
																												  mGenerateDisplacementImage(true),
																												  mHardFreezeTranslationX(false),
																												  mHardFreezeTranslationY(false),
																												  mWeightTranslationX(-1),
																												  mWeightTranslationY(-1),
																												  mUserDefinedFolderNameToSaveResult(""),
																												  mNumberOfIterGaussFilter(3),
																												  mNumberOfEndIterations(2),
																												  mDisplayLastTranslationValues(false),
																												  mSzImPre(tPt2di(1, 1)),
																												  mImPre(mSzImPre),
																												  mDImPre(nullptr),
																												  mSzImPost(tPt2di(1, 1)),
																												  mImPost(mSzImPost),
																												  mDImPost(nullptr),
																												  mSzImOut(tPt2di(1, 1)),
																												  mImOut(mSzImOut),
																												  mDImOut(nullptr),
																												  mSzImDepX(tPt2di(1, 1)),
																												  mImDepX(mSzImDepX),
																												  mDImDepX(nullptr),
																												  mSzImDepY(tPt2di(1, 1)),
																												  mImDepY(mSzImDepY),
																												  mDImDepY(nullptr),
																												  mSzImIntermediateDepX(tPt2di(1, 1)),
																												  mImIntermediateDepX(mSzImIntermediateDepX),
																												  mDImIntermediateDepX(nullptr),
																												  mSzImIntermediateDepY(tPt2di(1, 1)),
																												  mImIntermediateDepY(mSzImIntermediateDepY),
																												  mDImIntermediateDepY(nullptr),
																												  mSzCorrelationMask(tPt2di(1, 1)),
																												  mImCorrelationMask(mSzImIntermediateDepY),
																												  mDImCorrelationMask(nullptr),
																												  mDelTri({tPt2dr(0, 0)}),
																												  mInterpolTr(nullptr),
																												  mSysTranslation(nullptr),
																												  mEqTranslationTri(nullptr)
	{
	}

	cAppli_TriangleDeformationTranslation::~cAppli_TriangleDeformationTranslation()
	{
		delete mSysTranslation;
		delete mEqTranslationTri;
		delete mInterpolTr;
	}

	cCollecSpecArg2007 &cAppli_TriangleDeformationTranslation::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl
			   << Arg2007(mNamePreImage, "Name of pre-image file.", {eTA2007::FileImage, eTA2007::FileDirProj})
			   << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
			   << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
			   << Arg2007(mNumberOfIterations, "Total number of scales to run in multi-scale approach or iterations if multi-scale approach is not applied in optimisation process.");
	}

	cCollecSpecArg2007 &cAppli_TriangleDeformationTranslation::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{
		return anArgOpt
			   << AOpt2007(mNumberOfCols, "MaximumValueNumberOfCols",
						   "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mNumberOfLines, "MaximumValueNumberOfLines",
						   "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
			   << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
						   "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
			   << AOpt2007(mUseMMV2Interpolators, "UseMMV2Interpolators",
						   "Use MMVII interpolators instead of usual bilinear interpolation.", {eTA2007::HDV})
			   << AOpt2007(mInterpolArgs, "InterpolationArguments", "Which arguments to use for interpolation.", {eTA2007::HDV})
			   << AOpt2007(mSerialiseTriangleNodes, "SerialiseTriangleNodes", "Whether to serialise triangle nodes to .xml file or not.", {eTA2007::HDV})
			   << AOpt2007(mNameMultipleTriangleNodes, "NameOfMultipleTriangleNodes", "File name to use when saving all triangle nodes values to .xml file.", {eTA2007::HDV})
			   << AOpt2007(mInitialiseTranslationWithPreviousExecution, "InitialiseTranslationWithPreviousExecution",
						   "Whether to initialise or not with unknown values obtained at previous algorithm execution.", {eTA2007::HDV})
			   << AOpt2007(mInitialiseWithUserValues, "InitialiseWithUserValues",
						   "Whether the user wishes or not to initialise unknowns with personalised values.", {eTA2007::HDV})
			   << AOpt2007(mInitialiseXTranslationValue, "InitialXTranslationValue",
						   "Value to use for initialising x-translation unknowns.", {eTA2007::HDV})
			   << AOpt2007(mInitialiseYTranslationValue, "InitialYTranslationValue",
						   "Value to use for initialising y-translation unknowns.", {eTA2007::HDV})
			   << AOpt2007(mInitialiseWithMMVI, "InitialiseWithMMVI",
						   "Whether to initialise or not values of unknowns with pre-computed values from MicMacV1 at first execution.", {eTA2007::HDV})
			   << AOpt2007(mNameInitialDepX, "NameOfInitialDispXMap", "Name of file of initial X-displacement map.", {eTA2007::HDV, eTA2007::FileImage})
			   << AOpt2007(mNameInitialDepY, "NameOfInitialDispYMap", "Name of file of initial Y-displacement map.", {eTA2007::HDV, eTA2007::FileImage})
			   << AOpt2007(mNameIntermediateDepX, "NameForIntermediateDispXMap",
						   "File name to use when saving intermediate x-displacement maps between executions.", {eTA2007::HDV, eTA2007::FileImage, eTA2007::Tuning})
			   << AOpt2007(mNameIntermediateDepY, "NameForIntermediateDisYpMap",
						   "File name to use when saving intermediate y-displacement maps between executions.", {eTA2007::HDV, eTA2007::FileImage, eTA2007::Tuning})
			   << AOpt2007(mNameCorrelationMaskMMVI, "NameOfCorrelationMask",
						   "File name of mask file from MMVI giving locations where correlation is computed.", {eTA2007::HDV, eTA2007::FileImage})
			   << AOpt2007(mIsFirstExecution, "IsFirstExecution",
						   "Whether this is the first execution of optimisation algorithm or not", {eTA2007::HDV})
			   << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep", "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mGenerateDisplacementImage, "GenerateDisplacementImage",
						   "Whether to generate and save an image having been translated.", {eTA2007::HDV})
			   << AOpt2007(mHardFreezeTranslationX, "FreezeXTranslation",
						   "Whether to freeze or not x-translation to certain value during computation.", {eTA2007::HDV})
			   << AOpt2007(mHardFreezeTranslationY, "FreezeYTranslation",
						   "Whether to freeze or not y-translation to certain value during computation.", {eTA2007::HDV})
			   << AOpt2007(mWeightTranslationX, "WeightTranslationX",
						   "A value to weight x-translation for soft freezing of coefficient.", {eTA2007::HDV})
			   << AOpt2007(mWeightTranslationY, "WeightTranslationY",
						   "A value to weight y-translation for soft freezing of coefficient.", {eTA2007::HDV})
			   << AOpt2007(mUserDefinedFolderNameToSaveResult, "FolderNameToSaveResults",
						   "Folder name where to store produced results", {eTA2007::HDV})
			   << AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationValues",
						   "Whether to display the final coordinates of the trainslated points.", {eTA2007::HDV})
			   << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
						   "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
			   << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
						   "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning});
	}

	void cAppli_TriangleDeformationTranslation::LoopOverTrianglesAndUpdateParametersTranslation(const int aIterNumber,
																								const int aTotalNumberOfIterations,
																								const bool aNonEmptyPathToFolder)
	{
		//----------- Allocate vec of obs :
		// 6 for ImagePre and 5 for ImagePost in linear gradient case and 6 in bilinear case
		const int aNumberOfObsTr = mUseMMV2Interpolators ? TriangleDisplacement_GradInterpol_NbObs : TriangleDisplacement_Bilin_NbObs;
		tDoubleVect aVObsTr(6 + aNumberOfObsTr, 0);

		//----------- Extract current parameters
		tDenseVect aVCurSolTr = mSysTranslation->CurGlobSol(); // Get current solution.

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
		size_t aNbOut = 0; // Number of translated pixels out of image

		cStdStatRes aStatResObjTr;
		// Id of points
		int aNodeCounterTr = 0;

		std::unique_ptr<cMultipleTriangleNodesSerialiser> aVectorOfTriangleNodesTr = (mSerialiseTriangleNodes) ? cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(mNameMultipleTriangleNodes) 
																											   : nullptr;

		if (mHardFreezeTranslationX || mHardFreezeTranslationY)
		{
			const int aSolStepTr = 2;
			for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
			{
				const tPt3di aIndicesOfTriKnotsTr = mDelTri.KthFace(aTr);

				tIntVect aVecIndTr;
				GetIndicesVector(aVecIndTr, aIndicesOfTriKnotsTr, 2);

				if (mHardFreezeTranslationX)
				{
					const int aSolStartTranslationX = 0;
					ApplyHardConstraintsToMultipleUnknowns(aSolStartTranslationX, aSolStepTr, aVecIndTr,
														   aVCurSolTr, mSysTranslation);
				}

				if (mHardFreezeTranslationY)
				{
					const int aSolStartTranslationY = 1;
					ApplyHardConstraintsToMultipleUnknowns(aSolStartTranslationY, aSolStepTr, aVecIndTr,
														   aVCurSolTr, mSysTranslation);
				}
			}
		}

		// Loop over all triangles to add the observations on each point
		for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
		{
			const tTri2dr aTriTr = mDelTri.KthTri(aTr);
			const tPt3di aIndicesOfTriKnotsTr = mDelTri.KthFace(aTr);

			const cTriangle2DCompiled aCompTri(aTriTr);

			std::vector<tPt2di> aVectorToFillWithInsidePixels;
			aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

			//----------- Index of unknown, finds the associated pixels of current triangle
			tIntVect aVecIndTr;
			GetIndicesVector(aVecIndTr, aIndicesOfTriKnotsTr, 2);
			
			// Current translation 1st point of triangle
			const tPt2dr aCurTrPointA = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 0, 1, 0, 1, aTriTr, 0)
																   : LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 0, 1, 0, 1, aTriTr, 0,
																   													  aNodeCounterTr, aIndicesOfTriKnotsTr, true,
																													  aVectorOfTriangleNodesTr);
			// Current translation 2nd point of triangle
			const tPt2dr aCurTrPointB = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 2, 3, 2, 3, aTriTr, 1)
																   : LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 2, 3, 2, 3, aTriTr, 1,
																   													  aNodeCounterTr + 1, aIndicesOfTriKnotsTr, true,
																													  aVectorOfTriangleNodesTr);
			// Current translation 3rd point of triangle
			const tPt2dr aCurTrPointC = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 4, 5, 4, 5, aTriTr, 2)
																   : LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 4, 5, 4, 5, aTriTr, 2,
																   													  aNodeCounterTr + 2, aIndicesOfTriKnotsTr, true,
																													  aVectorOfTriangleNodesTr);

			aNodeCounterTr = (!mSerialiseTriangleNodes) ? 0 : aNodeCounterTr + 3;

			// Soft constraint x-translation
			if (mWeightTranslationX > 0 || mWeightTranslationY > 0)
			{
				const int aSolStepTranslation = 2;
				if (mWeightTranslationX > 0)
				{
					const int aSolStartTranslationX = 0;
					ApplySoftConstraintToMultipleUnknown(aSolStartTranslationX, aSolStepTranslation, aVecIndTr,
														 mSysTranslation, aVCurSolTr, mWeightTranslationX);
				}

				// Soft constraint y-translation
				if (mWeightTranslationY > 0)
				{
					const int aSolStartTranslationY = 1;
					ApplySoftConstraintToMultipleUnknown(aSolStartTranslationY, aSolStepTranslation, aVecIndTr,
														 mSysTranslation, aVCurSolTr, mWeightTranslationY);
				}
			}

			// Loop over all pixels inside triangle
			// size_t is necessary as there can be a lot of pixels in triangles
			for (size_t aFilledPixel = 0; aFilledPixel < aVectorToFillWithInsidePixels.size(); aFilledPixel++)
			{
				const cPtInsideTriangles aPixInsideTriangle = cPtInsideTriangles(aCompTri, aVectorToFillWithInsidePixels,
																				 aFilledPixel, aCurPreDIm, mInterpolTr);
				// Prepare for barycenter translation formula by filling aVObsTr with different coordinates
				FormalInterpBarycenter_SetObs(aVObsTr, 0, aPixInsideTriangle);

				// Image of a point in triangle by current translation
				const tPt2dr aTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aCurTrPointA, aCurTrPointB,
																									 aCurTrPointC, aVObsTr);

				const bool aPixInside = (mUseMMV2Interpolators) ? aCurPostDIm->InsideInterpolator(*mInterpolTr, aTranslatedFilledPoint, 0)
																: aCurPostDIm->InsideBL(aTranslatedFilledPoint);
				if (aPixInside)
				{
					(mUseMMV2Interpolators) ?
											// Prepare for application of linear gradient formula
						FormalGradInterpolTri_SetObs(aVObsTr, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint,
													 aCurPostDIm, mInterpolTr)
											:
											// Prepare for application of bilinear formula
						FormalBilinTri_SetObs(aVObsTr, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint, aCurPostDIm);

					// Now add observation
					mSysTranslation->CalcAndAddObs(mEqTranslationTri, aVecIndTr, aVObsTr);

					const tREAL8 anInterpolatedValue = (mUseMMV2Interpolators) ? aCurPostDIm->GetValueInterpol(*mInterpolTr, aTranslatedFilledPoint)
																			   : aCurPostDIm->GetVBL(aTranslatedFilledPoint);
					// Compute indicators
					const tREAL8 aDif = aVObsTr[5] - anInterpolatedValue; // residual
					aStatResObjTr.Add(std::abs(aDif));
				}
				else
					aNbOut++;
			}
		}

		if (mUseMultiScaleApproach && !mIsLastIters && aIterNumber != 0)
		{
			const bool aGenerateIntermediateMaps = false;
			if (aGenerateIntermediateMaps)
				GenerateDisplacementMaps(aVCurSolTr, aIterNumber, aTotalNumberOfIterations, aNonEmptyPathToFolder);
		}

		// Update all parameter taking into account previous observation
		mSysTranslation->SolveUpdateReset();

		// Save all triangle nodes to .xml file
		if (mSerialiseTriangleNodes && aVectorOfTriangleNodesTr != nullptr && (aIterNumber == (aTotalNumberOfIterations - 1)))
			aVectorOfTriangleNodesTr->MultipleNodesToFile(mNameMultipleTriangleNodes);

		if (mShow)
			StdOut() << aIterNumber + 1 << ", " << aStatResObjTr.Avg()
					 << ", " << aNbOut << std::endl;
	}

	void cAppli_TriangleDeformationTranslation::GenerateDisplacementMaps(const tDenseVect &aVFinalSolTr, const int aIterNumber,
																		 const int aTotalNumberOfIterations, const bool aNonEmptyPathToFolder)
	{
		// Initialise output image, x and y displacement maps
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImOut, mDImOut, mSzImOut);
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImDepX, mDImDepX, mSzImDepX);
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImDepY, mDImDepY, mSzImDepY);

		tIm aLastPreIm = tIm(mSzImPre);
		tDIm *aLastPreDIm = nullptr;
		LoadPrePostImageAndData(aLastPreIm, aLastPreDIm, "pre", mImPre, mImPost);

		if (mUseMultiScaleApproach && !mIsLastIters)
		{
			aLastPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
			aLastPreDIm = &aLastPreIm.DIm();
		}

		int aLastNodeCounterTr = 0;

		std::unique_ptr<cMultipleTriangleNodesSerialiser> aLastVectorOfTriangleNodesTr = nullptr;

		for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
		{
			const tTri2dr aLastTriTr = mDelTri.KthTri(aLTr);
			const tPt3di aLastIndicesOfTriKnotsTr = mDelTri.KthFace(aLTr);

			const cTriangle2DCompiled aLastCompTri(aLastTriTr);

			std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
			aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

			tIntVect aLastVecIndTr;
			GetIndicesVector(aLastVecIndTr, aLastIndicesOfTriKnotsTr, 2);

			// Last translation 1st point of triangle
			const tPt2dr aLastTrPointA = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 0, 1, 0, 1, aLastTriTr, 0)
																	: LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 0, 1, 0, 1, aLastTriTr, 0,
																													   aLastNodeCounterTr, aLastIndicesOfTriKnotsTr, false,
																													   aLastVectorOfTriangleNodesTr);
			// Last translation 2nd point of triangle
			const tPt2dr aLastTrPointB = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 2, 3, 2, 3, aLastTriTr, 1)
																	: LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 2, 3, 2, 3, aLastTriTr, 1,
																													   aLastNodeCounterTr + 1, aLastIndicesOfTriKnotsTr, false,
																													   aLastVectorOfTriangleNodesTr);
			// Last translation 3rd point of triangle
			const tPt2dr aLastTrPointC = (!mSerialiseTriangleNodes) ? LoadNodeAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 4, 5, 4, 5, aLastTriTr, 2)
																	: LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 4, 5, 4, 5, aLastTriTr, 2,
																													   aLastNodeCounterTr + 2, aLastIndicesOfTriKnotsTr, false,
																													   aLastVectorOfTriangleNodesTr);
			aLastNodeCounterTr = (!mSerialiseTriangleNodes) ? 0 : aLastNodeCounterTr + 3;

			const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

			for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
			{
				const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
																					 aLastFilledPixel, aLastPreDIm, mInterpolTr);

				// Image of a point in triangle by current translation
				const tPt2dr aLastTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
																										 aLastTrPointC, aLastPixInsideTriangle);

				FillDisplacementMapsTranslation(aLastPixInsideTriangle, aLastTranslatedFilledPoint,
												mSzImOut, mDImDepX, mDImDepY, mDImOut, mInterpolTr);
			}
		}

		// save displacement maps in x and y to image files
		if (mUseMultiScaleApproach)
		{
			SaveMultiScaleDisplacementMapsToFile(mDImDepX, mDImDepY, aNonEmptyPathToFolder, mUserDefinedFolderNameToSaveResult,
												 "DisplacedPixelsX_iter", "DisplacedPixelsY_iter", aIterNumber, mNumberPointsToGenerate,
												 aTotalNumberOfIterations);
			if (aIterNumber == aTotalNumberOfIterations - 1)
				SaveOutputImageToFile(mDImOut, aNonEmptyPathToFolder, mUserDefinedFolderNameToSaveResult, "DisplacedPixels",
									  mNumberPointsToGenerate, aTotalNumberOfIterations);
		}
		if (mInitialiseTranslationWithPreviousExecution)
		{
			mDImDepX->ToFile(mNameIntermediateDepX);
			mDImDepY->ToFile(mNameIntermediateDepY);
		}
		if (!mUseMultiScaleApproach && (aIterNumber == aTotalNumberOfIterations - 1))
		{
			SaveFinalDisplacementMapsToFile(mDImDepX, mDImDepY, aNonEmptyPathToFolder, mUserDefinedFolderNameToSaveResult, "DisplacedPixelsX",
											"DisplacedPixelsY", mNumberPointsToGenerate, aTotalNumberOfIterations);
			SaveOutputImageToFile(mDImOut, aNonEmptyPathToFolder, mUserDefinedFolderNameToSaveResult, "DisplacedPixels",
								  mNumberPointsToGenerate, aTotalNumberOfIterations);
		}
	}

	void cAppli_TriangleDeformationTranslation::GenerateDisplacementMapsAndDisplayLastTranslatedPoints(const int aIterNumber,
																									   const int aTotalNumberOfIterations,
																									   const tDenseVect &aVinitVecSolTr,
																									   const bool aNonEmptyPathToFolder)
	{
		tDenseVect aVFinalSolTr = mSysTranslation->CurGlobSol();

		if (mGenerateDisplacementImage)
			GenerateDisplacementMaps(aVFinalSolTr, aIterNumber, aTotalNumberOfIterations, aNonEmptyPathToFolder);

		if (mDisplayLastTranslationValues)
			DisplayFirstAndLastUnknownValuesAndComputeStatisticsTwoUnknowns(aVFinalSolTr, aVinitVecSolTr);
	}

	void cAppli_TriangleDeformationTranslation::DoOneIterationTranslation(const int aIterNumber, const int aTotalNumberOfIterations,
																		  const tDenseVect &aVInitVecSol, const bool aNonEmptyPathToFolder)
	{
		LoopOverTrianglesAndUpdateParametersTranslation(aIterNumber, aTotalNumberOfIterations, aNonEmptyPathToFolder); // Iterate over triangles and solve system

		// Show final translation results and produce displacement maps
		if (aIterNumber == (aTotalNumberOfIterations - 1))
			GenerateDisplacementMapsAndDisplayLastTranslatedPoints(aIterNumber, aTotalNumberOfIterations, aVInitVecSol, aNonEmptyPathToFolder);
	}

	//-----------------------------------------

	int cAppli_TriangleDeformationTranslation::Exe()
	{
		// Read pre and post images and update their sizes
		ReadImageFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
		ReadImageFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

		const bool aNonEmptyFolderName = CheckFolderExistence(mUserDefinedFolderNameToSaveResult);

		if (mUseMultiScaleApproach)
			mSigmaGaussFilter = mNumberOfIterations * mSigmaGaussFilterStep;

		if (mShow)
			StdOut() << "Iter, "
					 << "Diff, "
					 << "NbOut" << std::endl;

		if (!mIsFirstExecution && mSerialiseTriangleNodes)
			RegenerateTriangulatedGridFromSerialisation(mDelTri, mNameMultipleTriangleNodes);
		else if (!mIsFirstExecution || (mIsFirstExecution && mInitialiseWithMMVI) || (mIsFirstExecution && mSerialiseTriangleNodes))
			DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
															mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);

		InitialiseInterpolationAndEquationTranslation(mEqTranslationTri, mInterpolTr, mInterpolArgs, mUseMMV2Interpolators);

		if ((!mIsFirstExecution && mInitialiseWithUserValues && !mInitialiseTranslationWithPreviousExecution) ||
			(mIsFirstExecution && mSerialiseTriangleNodes))
			InitialiseWithUserValuesTranslation(mDelTri.NbPts(), mSysTranslation, mInitialiseWithUserValues,
												mInitialiseXTranslationValue, mInitialiseYTranslationValue);
		else
		{
			if (mIsFirstExecution && mInitialiseWithMMVI && mInitialiseTranslationWithPreviousExecution)
				InitialiseWithPreviousExecutionValuesTranslationMMVI(mDelTri, mSysTranslation, mInterpolTr,
																	 mNameInitialDepX, mImIntermediateDepX,
																	 mDImIntermediateDepX, mSzImIntermediateDepX,
																	 mNameInitialDepY, mImIntermediateDepY,
																	 mDImIntermediateDepY, mSzImIntermediateDepY,
																	 mNameCorrelationMaskMMVI, mImCorrelationMask,
																	 mDImCorrelationMask, mSzCorrelationMask);

			else if (!mIsFirstExecution && mInitialiseWithMMVI && mInitialiseTranslationWithPreviousExecution)
				InitialiseWithPreviousExecutionValuesTranslationMMVI(mDelTri, mSysTranslation, mInterpolTr,
																	 mNameIntermediateDepX, mImIntermediateDepX,
																	 mDImIntermediateDepX, mSzImIntermediateDepX,
																	 mNameIntermediateDepY, mImIntermediateDepY,
																	 mDImIntermediateDepY, mSzImIntermediateDepY,
																	 mNameCorrelationMaskMMVI, mImCorrelationMask,
																	 mDImCorrelationMask, mSzCorrelationMask);
			else if (!mIsFirstExecution && mSerialiseTriangleNodes && mInitialiseTranslationWithPreviousExecution)
				InitialiseWithPreviousExecutionValuesSerialisation(mDelTri.NbPts(), mSysTranslation, mNameMultipleTriangleNodes);
		}

		const tDenseVect aVInitSolTr = mSysTranslation->CurGlobSol().Dup(); // Duplicate initial solution

		const int aTotalNumberOfIterations = GetTotalNumberOfIterations(mUseMultiScaleApproach, mNumberOfIterations,
																		mNumberOfEndIterations);

		for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
			DoOneIterationTranslation(aIterNumber, aTotalNumberOfIterations, aVInitSolTr, aNonEmptyFolderName);

		return EXIT_SUCCESS;
	}

	/********************************************/
	//              ::MMVII                     //
	/********************************************/

	tMMVII_UnikPApli Alloc_cAppli_TriangleDeformationTranslation(const std::vector<std::string> &aVArgs,
																 const cSpecMMVII_Appli &aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_TriangleDeformationTranslation(aVArgs, aSpec));
	}

	cSpecMMVII_Appli TheSpec_ComputeTriangleDeformationTranslation(
		"ComputeTriangleDeformationTranslation",
		Alloc_cAppli_TriangleDeformationTranslation,
		"Compute 2D translation deformations between images using triangular mesh",
		{eApF::ImProc}, // Category
		{eApDT::Image}, // Input
		{eApDT::Image}, // Output
		__FILE__);

}; // namespace MMVII
