#include "ConstrainedTriangleDeformation.h"

/**
   \file TriangleDeformation.cpp

   \brief file for computing 2D deformations between 2 images
   thanks to triangular meshes.
**/

namespace MMVII
{
	/*********************************************/
	/*                                           */
	/*   cAppli_ConstrainedTriangleDeformation   */
	/*                                           */
	/*********************************************/

cAppli_ConstrainedTriangleDeformation::cAppli_ConstrainedTriangleDeformation(const std::vector<std::string> &aVArgs,
														   const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
																							mShow(true),
																							mUseMMV2Interpolators(true),
																							mInterpolArgs({"Tabul", "1000", "Cubic", "-0.5"}),
																							mUseConvexHull(true),
																							mNumberPointsToGenerate(0),
																							mInitialiseWithUserValues(true),
																							mInitialiseXTranslationValue(0),
																							mInitialiseYTranslationValue(0),
																							mInitialiseRadTrValue(0),
																							mInitialiseRadScValue(1),
																							mGenerateDisplacementImage(true),
																							mSaveTriangulation(false),
																							mTriangulationFileName("ConstrainedTriangulation.off"),
																							mUserDefinedFolderNameSaveResult(""),
																							mDisplayLastTranslationValues(false),
																							mDisplayLastRadiometryValues(false),
																							mDisplayStatisticsOnUnkValues(false),
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
																							mTriConstrGenerator(),
																							mInterpol(nullptr),
																							mSys(nullptr),
																							mEqTriDeform(nullptr)
	{
	}

	cAppli_ConstrainedTriangleDeformation::~cAppli_ConstrainedTriangleDeformation()
	{
		delete mSys;
		delete mEqTriDeform;
		delete mInterpol;
	}

	cCollecSpecArg2007 &cAppli_ConstrainedTriangleDeformation::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
						<< Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
						<< Arg2007(mNumberOfIterations, "Total number of iterations to run in optimisation process.")
						<< Arg2007(mUseConstrainedTriangulation, "Constrain the triangulation with segments.")
               			<< Arg2007(mVectorOfNodeCoordinates, "Vector containing coordinates of mesh nodes.")
						<< Arg2007(mLinkConstraintsWithIds, "Whether to link points of constraints by point ids or use point coordinates.")
            			<< Arg2007(mVectorOfConstraintSegments, "Ids or points coordinates of constraint nodes.");
	}

	cCollecSpecArg2007 &cAppli_ConstrainedTriangleDeformation::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{
		return anArgOpt << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mUseMMV2Interpolators, "UseMMV2Interpolators",
									"Use MMVII interpolators instead of usual bilinear interpolation.", {eTA2007::HDV})
						<< AOpt2007(mInterpolArgs, "InterpolationArguments", "Input arguments for MMVII interpolation use.", {eTA2007::HDV})
						<< AOpt2007(mUseConvexHull, "UseConvexHull", "Whether or not to use convex hull during triangulation", {eTA2007::HDV})
						<< AOpt2007(mNumberPointsToGenerate, "NumberOfPointsToGenerate", "Number of points you want to generate for triangulation.", {eTA2007::HDV})
						<< AOpt2007(mInitialiseWithUserValues, "InitialiseWithUserValues",
									"Whether the user wishes or not to initialise unknowns with personalised values.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mInitialiseXTranslationValue, "InitialTranslationXValue",
									"Value to use for initialising x-translation unknowns.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mInitialiseYTranslationValue, "InitialTranslationYValue",
									"Value to use for initialising y-translation unknowns.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mInitialiseRadTrValue, "InitialeRadiometryTranslationValue",
									"Value to use for initialising radiometry translation unknown values.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mInitialiseRadScValue, "InitialeRadiometryScalingValue",
									"Value to use for initialising radiometry scaling unknown values.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mGenerateDisplacementImage, "GenerateDisplacementImage",
									"Whether to generate and save an image having been translated.", {eTA2007::HDV})
						<< AOpt2007(mSaveTriangulation, "SaveTriangulationResult", "Whether or not to save triangulation result in a .off file", {eTA2007::HDV})
						<< AOpt2007(mTriangulationFileName, "TriangulationFileName", "Name to use for saving triangulation operation in a .off file", {eTA2007::HDV})
						<< AOpt2007(mUserDefinedFolderNameSaveResult, "FolderNameToSaveResults",
									"Folder name where to store produced results.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationsValues",
									"Whether to display the final values of unknowns linked to point translation.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
									"Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV, eTA2007::Tuning})
						<< AOpt2007(mDisplayStatisticsOnUnkValues, "DisplayStatistics", "Display statistics : min, max, mean and std on final values of solution.", {eTA2007::HDV, eTA2007::Tuning});
	}

	void cAppli_ConstrainedTriangleDeformation::LoopOverTrianglesAndUpdateConstrainedParameters(const int aIterNumber, const bool aUserDefinedFolderName)
	{
		//----------- Allocate vector of observations :
		// 6 for ImagePre and 5 for ImagePost in linear gradient case and 6 in bilinear case
		const int aNumberOfObs = mUseMMV2Interpolators ? TriangleDisplacement_GradInterpol_NbObs : TriangleDisplacement_Bilin_NbObs;
		tDoubleVect aVObs(6 + aNumberOfObs, 0);

		//----------- Extract current parameters
		tDenseVect aVCurSol = mSys->CurGlobSol(); // Get current solution.

		tIm aCurPreIm = tIm(mSzImPre);
		tDIm *aCurPreDIm = nullptr;
		tIm aCurPostIm = tIm(mSzImPost);
		tDIm *aCurPostDIm = nullptr;

		LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", mImPre, mImPost);
		LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", mImPre, mImPost);

		//----------- Declaration of indicators of convergence
		size_t aNbOut = 0; // Number of translated pixels out of image

		cStdStatRes aStatResObj;

		// Loop over all triangles to add the observations on each point
		for (tpp::FaceIterator aFaceiterator = mTriConstrGenerator.fbegin(); aFaceiterator != mTriConstrGenerator.fend(); ++aFaceiterator)
		{
			tTri2dr aTri = cTriangle(tPt2dr(0, 0), tPt2dr(0, 0), tPt2dr(0, 0));
			tPt3di anIndicesOfTriKnots = tPt3di(-1, -1, -1);
			GetFaceAndTriangleFromDelaunayPoints(aFaceiterator, mTriConstrGenerator, aTri, anIndicesOfTriKnots);

			const cTriangle2DCompiled aCompTri(aTri);

			std::vector<tPt2di> aVectorToFillWithInsidePixels;
			aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

			//----------- Index of unknown, finds the associated pixels of current triangle
			tIntVect aVecInd;
			GetIndicesVector(aVecInd, anIndicesOfTriKnots, 4);

			// Current translation 1st point of triangle
			const tPt2dr aCurTrPointA = LoadNodeAndReturnCurrentDisplacement(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0);
			// Current translation 2nd point of triangle
			const tPt2dr aCurTrPointB = LoadNodeAndReturnCurrentDisplacement(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1);
			// Current translation 3rd point of triangle
			const tPt2dr aCurTrPointC = LoadNodeAndReturnCurrentDisplacement(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2);
			// Current translation on radiometry 1st point of triangle
			const tREAL8 aCurRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0);
			// Current scale on radiometry 1st point of triangle
			const tREAL8 aCurRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0);
			// Current translation on radiometry 2nd point of triangle
			const tREAL8 aCurRadTrPointB = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1);
			// Current scale on radiometry 2nd point of triangle
			const tREAL8 aCurRadScPointB = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1);
			// Current translation on radiometry 3rd point of triangle
			const tREAL8 aCurRadTrPointC = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2);
			// Current scale on radiometry 3rd point of triangle
			const tREAL8 aCurRadScPointC = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2);


			// Loop over all pixels inside triangle
			// size_t is necessary as there can be a lot of pixels in triangles
			for (size_t aFilledPixel = 0; aFilledPixel < aVectorToFillWithInsidePixels.size(); aFilledPixel++)
			{
				const cPtInsideTriangles aPixInsideTriangle = cPtInsideTriangles(aCompTri, aVectorToFillWithInsidePixels,
																				 aFilledPixel, aCurPreDIm, mInterpol);
				// Prepare for barycenter translation formula by filling aVObs with different coordinates
				FormalInterpBarycenter_SetObs(aVObs, 0, aPixInsideTriangle);

				// Image of a point in triangle by current geometric translation of triangle knots
				const tPt2dr aTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aCurTrPointA, aCurTrPointB,
																									 aCurTrPointC, aVObs);

				// Radiometry translation of pixel by current radiometry translation of triangle knots
				const tREAL8 aRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aCurRadTrPointA,
																												aCurRadTrPointB,
																												aCurRadTrPointC,
																												aVObs);
				// Radiometry translation of pixel by current radiometry scaling of triangle knots
				const tREAL8 aRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aCurRadScPointA,
																										aCurRadScPointB,
																										aCurRadScPointC,
																										aVObs);
				// Check if pixel is inside image and therefore can be used as an observation
				const bool aPixInside = (mUseMMV2Interpolators) ? aCurPostDIm->InsideInterpolator(*mInterpol, aTranslatedFilledPoint, 0)
																: aCurPostDIm->InsideBL(aTranslatedFilledPoint);
				if (aPixInside)
				{
					(mUseMMV2Interpolators) ?
											// Prepare for application of MMVII interpolators
						FormalGradInterpolTri_SetObs(aVObs, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint,
													 aCurPostDIm, mInterpol)
											:
											// Prepare for application of bilinear interpolators
						FormalBilinTri_SetObs(aVObs, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint, aCurPostDIm);

					// Now add observations
					mSys->CalcAndAddObs(mEqTriDeform, aVecInd, aVObs);

					// Get interpolated value
					const tREAL8 anInterpolatedValue = (mUseMMV2Interpolators) ? aCurPostDIm->GetValueInterpol(*mInterpol, aTranslatedFilledPoint)
																			   : aCurPostDIm->GetVBL(aTranslatedFilledPoint);
					const tREAL8 aRadValueImPre = aRadiometryScaling * aVObs[5] + aRadiometryTranslation;
					// Compute indicators
					const tREAL8 aDif = aRadValueImPre - anInterpolatedValue; // residual

					aStatResObj.Add(std::abs(aDif));
				}
				else
					aNbOut++;
			}
		}

		// Update all parameters by taking into account previous observation
		mSys->SolveUpdateReset();

		if (mShow)
			StdOut() << aIterNumber + 1 << ", " << aStatResObj.Avg()
					 << ", " << aNbOut << std::endl;
	}

	void cAppli_ConstrainedTriangleDeformation::GenerateConstrainedDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber,
											                    								   const bool aNonEmptyPathToFolder)
	{
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImOut, mDImOut, mSzImOut);
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImDepX, mDImDepX, mSzImDepX);
		InitialiseDisplacementMapsAndOutputImage(mSzImPre, mImDepY, mDImDepY, mSzImDepY);

		tIm aLastPreIm = tIm(mSzImPre);
		tDIm *aLastPreDIm = nullptr;
		LoadPrePostImageAndData(aLastPreIm, aLastPreDIm, "pre", mImPre, mImPost);

		tIm aLastPostIm = tIm(mSzImPost);
		tDIm *aLastPostDIm = nullptr;
		LoadPrePostImageAndData(aLastPostIm, aLastPostDIm, "post", mImPre, mImPost);

		// Last loop over all triangles and pixels inside triangles
		for (tpp::FaceIterator aFaceiterator = mTriConstrGenerator.fbegin(); aFaceiterator != mTriConstrGenerator.fend(); ++aFaceiterator)
		{
			tTri2dr aLastTri = cTriangle(tPt2dr(0, 0), tPt2dr(0, 0), tPt2dr(0, 0));
			tPt3di aLastIndicesOfTriKnots = tPt3di(-1, -1, -1);
			GetFaceAndTriangleFromDelaunayPoints(aFaceiterator, mTriConstrGenerator, aLastTri, aLastIndicesOfTriKnots);

			const cTriangle2DCompiled aLastCompTri(aLastTri);

			std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
			aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

			tIntVect aLastVecInd;
			GetIndicesVector(aLastVecInd, aLastIndicesOfTriKnots, 4);

			// Last translation 1st point of triangle
			const tPt2dr aLastTrPointA = LoadNodeAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
			// Last translation 2nd point of triangle
			const tPt2dr aLastTrPointB = LoadNodeAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
			// Last translation 3rd point of triangle
			const tPt2dr aLastTrPointC = LoadNodeAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);
			// Last radiometry translation of 1st point
			const tREAL8 aLastRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
			// Last radiometry scaling of 1st point
			const tREAL8 aLastRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
			// Last radiometry translation of 2nd point
			const tREAL8 aLastRadTrPointB = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
			// Last radiometry scaling of 2nd point
			const tREAL8 aLastRadScPointB = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
			// Last radiometry translation of 3rd point
			const tREAL8 aLastRadTrPointC = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);
			// Last radiometry scaling of 3rd point
			const tREAL8 aLastRadScPointC = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);

			const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

			for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
			{
				const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
																					 aLastFilledPixel, aLastPreDIm, mInterpol);

				// Pixel coordinates in triangle by final geometric translation
				const tPt2dr aLastTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
																										 aLastTrPointC, aLastPixInsideTriangle);
				// Pixel radiometry translation value in triangle by final radiometry translation
				const tREAL8 aLastRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aLastRadTrPointA,
																													aLastRadTrPointB,
																													aLastRadTrPointC,
																													aLastPixInsideTriangle);
				// Pixel radiometry scaling value in triangle by final radiometry scaling
				const tREAL8 aLastRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aLastRadScPointA,
																											aLastRadScPointB,
																											aLastRadScPointC,
																											aLastPixInsideTriangle);
				// Build displacement maps and output image
				FillDisplacementMapsAndOutputImage(aLastPixInsideTriangle, aLastTranslatedFilledPoint,
												   aLastRadiometryTranslation, aLastRadiometryScaling,
												   mSzImOut, mDImDepX, mDImDepY, mDImOut, aLastPostDIm, mInterpol);
			
			}
		}
	}

	void cAppli_ConstrainedTriangleDeformation::GenerateConstrainedDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const bool aDisplayLastRadiometryValues,
																			 									const bool aDisplayLastTranslationValues, const bool aUserDefinedFolderName)
	{
		tDenseVect aVFinalSol = mSys->CurGlobSol();

		if (mGenerateDisplacementImage)
			GenerateConstrainedDisplacementMapsAndOutputImages(aVFinalSol, aIterNumber, aUserDefinedFolderName);

		if (aDisplayLastRadiometryValues || aDisplayLastTranslationValues || mDisplayStatisticsOnUnkValues)
			DisplayLastUnknownValuesAndComputeStatistics(aVFinalSol, aDisplayLastRadiometryValues,
														 aDisplayLastTranslationValues, mDisplayStatisticsOnUnkValues);	
	}

	void cAppli_ConstrainedTriangleDeformation::DoOneConstrainedIteration(const int aIterNumber, const bool aNonEmptyPathToFolder)
	{
		LoopOverTrianglesAndUpdateConstrainedParameters(aIterNumber, aNonEmptyPathToFolder); // Iterate over triangles and solve system

		// Show final translation results and produce displacement maps
		if (aIterNumber == (mNumberOfIterations - 1))
			GenerateConstrainedDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber, mDisplayLastRadiometryValues,
																			mDisplayLastTranslationValues, aNonEmptyPathToFolder);
	}

	//-----------------------------------------

	int cAppli_ConstrainedTriangleDeformation::Exe()
	{
		// Read pre and post images and update their sizes
		ReadImageFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
		ReadImageFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

		const bool aNonEmptyFolderName = CheckFolderExistence(mUserDefinedFolderNameSaveResult);

		if (mShow)
			StdOut() << "Iter, "
					 << "Diff, "
					 << "NbOut" << std::endl;

		// Generate triangulated knots coordinates and constraints
		GenerateConstrainedTriangulationGridAndConstraints(mTriConstrGenerator, mVectorOfNodeCoordinates,
														   mUseConstrainedTriangulation, mNumberPointsToGenerate, 
														   mSzImPre, mLinkConstraintsWithIds, mUseConvexHull,
														   mSaveTriangulation, mTriangulationFileName);

		// Initialise equation and interpolators if needed
		InitialiseInterpolationAndEquation(mEqTriDeform, mInterpol, mInterpolArgs, mUseMMV2Interpolators);

		InitialisationWithUserValues(mTriConstrGenerator.verticeCount(), mSys, mInitialiseWithUserValues, mInitialiseXTranslationValue,
									 mInitialiseYTranslationValue, mInitialiseRadTrValue, mInitialiseRadScValue);

		for (int aIterNumber = 0; aIterNumber < mNumberOfIterations; aIterNumber++)
			DoOneConstrainedIteration(aIterNumber, aNonEmptyFolderName);

		return EXIT_SUCCESS;
	}

	/********************************************/
	//              ::MMVII                     //
	/********************************************/

	tMMVII_UnikPApli Alloc_cAppli_ConstrainedTriangleDeformation(const std::vector<std::string> &aVArgs,
											                     const cSpecMMVII_Appli &aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_ConstrainedTriangleDeformation(aVArgs, aSpec));
	}

	cSpecMMVII_Appli TheSpec_ComputeConstrainedTriangleDeformation(
		"ComputeConstrainedTriangleDeformation",
		Alloc_cAppli_ConstrainedTriangleDeformation,
		"Compute 2D deformation between images using constrained triangular mesh",
		{eApF::ImProc}, // Category
		{eApDT::Image}, // Input
		{eApDT::Image}, // Output
		__FILE__);

}; // namespace MMVII
