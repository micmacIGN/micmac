#include "TriangleDeformation.h"

/**
   \file TriangleDeformation.cpp

   \brief file for computing 2D deformations between 2 images
   thanks to triangular meshes.
**/

namespace MMVII
{
    /******************************************/
    /*                                        */
    /*       cAppli_TriangleDeformation       */
    /*                                        */
    /******************************************/

    cAppli_TriangleDeformation::cAppli_TriangleDeformation(const std::vector<std::string> &aVArgs,
                                                           const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                                                                            mNumberOfLines(1),
                                                                                            mNumberOfCols(1),
                                                                                            mShow(true),
                                                                                            mComputeAvgMax(false),
                                                                                            mUseMultiScaleApproach(false),
                                                                                            mBuildRandomUniformGrid(false),
                                                                                            mUseMMVIIInterpolators(false),
                                                                                            mInterpolArgs({"Tabul", "1000", "Cubic", "-0.5"}),
                                                                                            mInitialiseTranslationWithPreviousExecution(false),
                                                                                            mInitialiseRadiometryWithPreviousExecution(false),
                                                                                            mInitialiseWithUserValues(true),
                                                                                            mInitialiseXTranslationValue(0),
                                                                                            mInitialiseYTranslationValue(0),
                                                                                            mInitialiseRadTrValue(0),
                                                                                            mInitialiseRadScValue(1),
                                                                                            mInitialiseWithMMVI(false),
                                                                                            mNameInitialDepX("InitialXDisplacementMap.tif"),
                                                                                            mNameInitialDepY("InitialYDisplacementMap.tif"),
                                                                                            mNameIntermediateDepX("IntermediateDispXMap.tif"),
                                                                                            mNameIntermediateDepY("IntermediateDispYMap.tif"),
                                                                                            mIsFirstExecution(false),
                                                                                            mSigmaGaussFilterStep(1),
                                                                                            mGenerateDisplacementImage(true),
                                                                                            mFreezeTranslationX(false),
                                                                                            mFreezeTranslationY(false),
                                                                                            mFreezeRadTranslation(false),
                                                                                            mFreezeRadScale(false),
                                                                                            mWeightTranslationX(-1),
                                                                                            mWeightTranslationY(-1),
                                                                                            mWeightRadTranslation(-1),
                                                                                            mWeightRadScale(-1),
                                                                                            mNumberOfItersToHardFreezeTranslation(0),
                                                                                            mNumberOfItersToHardFreezeRadiometry(0),
                                                                                            mNumberOfItersToSoftFreezeTranslation(0),
                                                                                            mNumberOfItersToSoftFreezeRadiometry(0),
                                                                                            mNumberOfIterGaussFilter(3),
                                                                                            mNumberOfEndIterations(2),
                                                                                            mUserDefinedFolderNameSaveResult(""),
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
                                                                                            mSzIntermediateImOut(tPt2di(1, 1)),
                                                                                            mImIntermediateOut(mSzIntermediateImOut),
                                                                                            mDImIntermediateOut(nullptr),
                                                                                            mSzImIntermediateDepX(tPt2di(1, 1)),
                                                                                            mImIntermediateDepX(mSzImIntermediateDepX),
                                                                                            mDImIntermediateDepX(nullptr),
                                                                                            mSzImIntermediateDepY(tPt2di(1, 1)),
                                                                                            mImIntermediateDepY(mSzImIntermediateDepY),
                                                                                            mDImIntermediateDepY(nullptr),
                                                                                            mSzCorrelationMask(tPt2di(1, 1)),
                                                                                            mImCorrelationMask(mSzImIntermediateDepY),
                                                                                            mDImCorrelationMask(nullptr),
                                                                                            mSzImDiff(tPt2di(1, 1)),
                                                                                            mImDiff(mSzImDiff),
                                                                                            mDImDiff(nullptr),
                                                                                            mDelTri({tPt2dr(0, 0)}),
                                                                                            mInterpol(nullptr),
                                                                                            mSys(nullptr),
                                                                                            mEqTriDeform(nullptr)
    {
    }

    cAppli_TriangleDeformation::~cAppli_TriangleDeformation()
    {
        delete mSys;
        delete mEqTriDeform;
        delete mInterpol;
    }

    cCollecSpecArg2007 &cAppli_TriangleDeformation::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach or iterations if multi-scale approach is not applied in optimisation process.");
    }

    cCollecSpecArg2007 &cAppli_TriangleDeformation::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mNumberOfCols, "MaximumValueNumberOfCols",
                           "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfLines, "MaximumValueNumberOfLines",
                           "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mComputeAvgMax, "ComputeAvgMaxDiffIm",
                           "Whether to compute the average and maximum pixel value of the difference image between post and pre image or not.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
               << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
                           "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
               << AOpt2007(mUseMMVIIInterpolators, "UseMMVIIInterpolators",
                           "Use MMVII interpolators instead of usual bilinear interpolation.", {eTA2007::HDV})
               << AOpt2007(mInterpolArgs, "InterpolationArguments", "Which type of interpolation to use : cubic, sinc or MMVIIK", {eTA2007::HDV})
               << AOpt2007(mInitialiseTranslationWithPreviousExecution, "InitialiseTranslationWithPreviousExecution",
                           "Whether to initialise or not with unknown translation values obtained at previous execution", {eTA2007::HDV})
               << AOpt2007(mInitialiseRadiometryWithPreviousExecution, "InitialiseRadiometryWithPreviousExecution",
                           "Whether to initialise or not with unknown radiometry values obtained at previous execution", {eTA2007::HDV})
               << AOpt2007(mInitialiseWithUserValues, "InitialiseWithUserValues",
                           "Whether the user wishes or not to initialise unknowns with personalised values.", {eTA2007::HDV})
               << AOpt2007(mInitialiseXTranslationValue, "InitialXTranslationValue",
                           "Value to use for initialising x-translation unknowns.", {eTA2007::HDV})
               << AOpt2007(mInitialiseYTranslationValue, "InitialYTranslationValue",
                           "Value to use for initialising y-translation unknowns.", {eTA2007::HDV})
               << AOpt2007(mInitialiseRadTrValue, "InitialeRadiometryTranslationValue",
                           "Value to use for initialising radiometry translation unknown values", {eTA2007::HDV})
               << AOpt2007(mInitialiseRadScValue, "InitialeRadiometryScalingValue",
                           "Value to use for initialising radiometry scaling unknown values", {eTA2007::HDV})
               << AOpt2007(mInitialiseWithMMVI, "InitialiseWithMMVI",
                           "Whether to initialise or not values of unknowns with pre-computed values from MicMacV1 at first execution", {eTA2007::HDV})
               << AOpt2007(mNameInitialDepX, "InitialDispXMapFilename", "Name of file of initial X-displacement map", {eTA2007::HDV, eTA2007::FileImage})
               << AOpt2007(mNameInitialDepY, "InitialDispYMapFilename", "Name of file of initial Y-displacement map", {eTA2007::HDV, eTA2007::FileImage})
               << AOpt2007(mNameIntermediateDepX, "NameForIntermediateXDispMap",
                           "File name to use when saving intermediate x-displacement maps between executions", {eTA2007::HDV, eTA2007::FileImage, eTA2007::Tuning})
               << AOpt2007(mNameIntermediateDepY, "NameForIntermediateYDispMap",
                           "File name to use when saving intermediate y-displacement maps between executions", {eTA2007::HDV, eTA2007::FileImage, eTA2007::Tuning})
               << AOpt2007(mNameCorrelationMaskMMVI, "NameOfCorrelationMask",
                           "File name of mask file from MMVI giving locations where correlation is computed", {eTA2007::HDV, eTA2007::FileImage})
               << AOpt2007(mIsFirstExecution, "IsFirstExecution",
                           "Whether this is the first execution of optimisation algorithm or not", {eTA2007::HDV})
               << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep", "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mGenerateDisplacementImage, "GenerateDisplacementImage",
                           "Whether to generate and save an image having been translated.", {eTA2007::HDV})
               << AOpt2007(mFreezeTranslationX, "FreezeXTranslation",
                           "Whether to freeze or not x-translation to certain value during computation.", {eTA2007::HDV})
               << AOpt2007(mFreezeTranslationY, "FreezeYTranslation",
                           "Whether to freeze or not y-translation to certain value during computation.", {eTA2007::HDV})
               << AOpt2007(mFreezeRadTranslation, "FreezeRadTranslation",
                           "Whether to freeze radiometry translation factor in computation or not.", {eTA2007::HDV})
               << AOpt2007(mFreezeRadScale, "FreezeRadScaling",
                           "Whether to freeze radiometry scaling factor in computation or not.", {eTA2007::HDV})
               << AOpt2007(mWeightTranslationX, "WeightTranslationX",
                           "A value to weight x-translation for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mWeightTranslationY, "WeightTranslationY",
                           "A value to weight y-translation for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mWeightRadTranslation, "WeightRadiometryTranslation",
                           "A value to weight radiometry translation for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mWeightRadScale, "WeightRadiometryScaling",
                           "A value to weight radiometry scaling for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mNumberOfItersToHardFreezeTranslation, "NumberOfHardFrozenTranslationIter",
                           "Freeze x and y translation unknowns for a certain number of the first iterations", {eTA2007::HDV})
               << AOpt2007(mNumberOfItersToHardFreezeRadiometry, "NumberOfHardFrozenRadiometryIter",
                           "Freeze radiometry translation and scaling unknowns for a certain number of the first iterations", {eTA2007::HDV})
               << AOpt2007(mNumberOfItersToSoftFreezeTranslation, "NumberOfSoftFrozenTranslationIter",
                           "Apply soft constraints to translation unknowns for a certain number of the first iterations", {eTA2007::HDV})
               << AOpt2007(mNumberOfItersToSoftFreezeRadiometry, "NumberOfSoftFrozenRadiometryIter",
                           "Apply soft contraints to radiometry translation and scaling unknowns for a certain number of the first iterations", {eTA2007::HDV})
               << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mUserDefinedFolderNameSaveResult, "FolderNameToSaveResults",
                           "Folder name where to store produced results", {eTA2007::HDV})
               << AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationsValues",
                           "Whether to display the final values of unknowns linked to point translation.", {eTA2007::HDV})
               << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                           "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV});
    }

    void cAppli_TriangleDeformation::LoopOverTrianglesAndUpdateParameters(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                          const bool aNonEmptyPathToFolder, const bool aHardFreezeForFirstItersTranslation,
                                                                          const bool aHardFreezeForFirstItersRadiometry, const bool aSoftFreezeForFirstItersTranslation,
                                                                          const bool aSoftFreezeForFirstItersRadiometry)
    {
        //----------- allocate vec of obs :
        // 6 for ImagePre and 5 for ImagePost in linear gradient case and 6 in bilinear case
        const int aNumberOfObs = mUseMMVIIInterpolators ? TriangleDisplacement_GradInterpol_NbObs : TriangleDisplacement_Bilin_NbObs;
        tDoubleVect aVObs(6 + aNumberOfObs, 0);

        //----------- extract current parameters
        tDenseVect aVCurSol = mSys->CurGlobSol(); // Get current solution.

        tIm aCurPreIm = tIm(mSzImPre);
        tDIm *aCurPreDIm = nullptr;
        tIm aCurPostIm = tIm(mSzImPost);
        tDIm *aCurPostDIm = nullptr;

        mIsLastIters = false;

        if (mUseMultiScaleApproach)
            mIsLastIters = ManageDifferentCasesOfEndIterations(aIterNumber, mNumberOfScales, mNumberOfEndIterations,
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

            const bool aSaveGaussImage = false; // if a save of image filtered by Gauss filter is wanted
            if (aSaveGaussImage)
            {
                if (aNonEmptyPathToFolder)
                    aCurPreDIm->ToFile(mUserDefinedFolderNameSaveResult + "/GaussFilteredImPre_iter_" + ToStr(aIterNumber) + ".tif");
                else
                    aCurPreDIm->ToFile("GaussFilteredImPre_iter_" + ToStr(aIterNumber) + ".tif");
            }
        }
        else if (mUseMultiScaleApproach && mIsLastIters)
        {
            LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", mImPre, mImPost);
            LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", mImPre, mImPost);
        }

        //----------- declaration of indicators of convergence
        tREAL8 aSomDif = 0; // sum of difference between untranslated pixel and translated one.
        size_t aNbOut = 0;  // number of translated pixels out of image

        // Count number of pixels inside triangles for normalisation
        size_t aTotalNumberOfInsidePixels = 0;
        // Id of points
        int aNodeCounter = 0;

        std::unique_ptr<cMultipleTriangleNodesSerialiser> aVectorOfTriangleNodes = nullptr;

        if (mSerialiseTriangleNodes && aVectorOfTriangleNodes == nullptr)
            aVectorOfTriangleNodes = cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(mNameMultipleTriangleNodes);

        // hard constraint : freeze radiometric coefficients
        if (mFreezeRadTranslation || mFreezeRadScale ||
            mFreezeTranslationX || mFreezeTranslationY)
        {
            for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
            {
                const tPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

                tIntVect aVecInd;
                GetIndicesVector(aVecInd, aIndicesOfTriKnots, 4);
                if (mFreezeTranslationX || mFreezeTranslationY)
                {
                    const bool aHardConstraintsApplicationTranslation = ((!aHardFreezeForFirstItersTranslation) || (aHardFreezeForFirstItersTranslation && aIterNumber < mNumberOfItersToHardFreezeTranslation));
                    if (mFreezeTranslationX)
                    {
                        if (aHardConstraintsApplicationTranslation)
                            ApplyHardConstraintsToMultipleUnknowns(0, 4, 8, aVecInd, aVCurSol, mSys);
                        if (aHardFreezeForFirstItersTranslation && aIterNumber >= mNumberOfItersToHardFreezeTranslation)
                            UnfreezeMultipleUnknowns(0, 4, 8, aVecInd, mSys);
                    }
                    if (mFreezeTranslationY)
                    {
                        if (aHardConstraintsApplicationTranslation)
                            ApplyHardConstraintsToMultipleUnknowns(1, 5, 9, aVecInd, aVCurSol, mSys);
                        if (aHardFreezeForFirstItersTranslation && aIterNumber >= mNumberOfItersToHardFreezeTranslation)
                            UnfreezeMultipleUnknowns(1, 5, 9, aVecInd, mSys);
                    }
                }
                if (mFreezeRadTranslation || mFreezeRadScale)
                {
                    const bool aHardConstraintsApplicationRadiometry = ((!aHardFreezeForFirstItersRadiometry) || (aHardFreezeForFirstItersRadiometry && aIterNumber < mNumberOfItersToHardFreezeRadiometry));
                    if (mFreezeRadTranslation)
                    {
                        if (aHardConstraintsApplicationRadiometry)
                            ApplyHardConstraintsToMultipleUnknowns(2, 6, 10, aVecInd, aVCurSol, mSys);
                        if (aHardFreezeForFirstItersRadiometry && aIterNumber >= mNumberOfItersToHardFreezeRadiometry)
                            UnfreezeMultipleUnknowns(2, 6, 10, aVecInd, mSys);
                    }
                    if (mFreezeRadScale)
                    {
                        if (aHardConstraintsApplicationRadiometry)
                            ApplyHardConstraintsToMultipleUnknowns(3, 7, 11, aVecInd, aVCurSol, mSys);
                        if (aHardFreezeForFirstItersRadiometry && aIterNumber >= mNumberOfItersToHardFreezeRadiometry)
                            UnfreezeMultipleUnknowns(3, 7, 11, aVecInd, mSys);
                    }
                }
            }
        }

        // Loop over all triangles to add the observations on each point
        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTri = mDelTri.KthTri(aTr);
            const tPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

            const cTriangle2DCompiled aCompTri(aTri);

            std::vector<tPt2di> aVectorToFillWithInsidePixels;
            aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

            //----------- index of unknown, finds the associated pixels of current triangle
            tIntVect aVecInd;
            GetIndicesVector(aVecInd, aIndicesOfTriKnots, 4);

            tPt2dr aCurTrPointA = tPt2dr(0, 0);
            tPt2dr aCurTrPointB = tPt2dr(0, 0);
            tPt2dr aCurTrPointC = tPt2dr(0, 0);
            tREAL8 aCurRadTrPointA = 0;
            tREAL8 aCurRadScPointA = 1;
            tREAL8 aCurRadTrPointB = 0;
            tREAL8 aCurRadScPointB = 1;
            tREAL8 aCurRadTrPointC = 0;
            tREAL8 aCurRadScPointC = 1;

            if (!mSerialiseTriangleNodes && aVectorOfTriangleNodes == nullptr)
            {
                aCurTrPointA = LoadNodeAndReturnCurrentDisplacement(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0);   // current translation 1st point of triangle
                aCurTrPointB = LoadNodeAndReturnCurrentDisplacement(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1);   // current translation 2nd point of triangle
                aCurTrPointC = LoadNodeAndReturnCurrentDisplacement(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2); // current translation 3rd point of triangle

                aCurRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0);    // current translation on radiometry 1st point of triangle
                aCurRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0);        // current scale on radiometry 1st point of triangle
                aCurRadTrPointB = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1);    // current translation on radiometry 2nd point of triangle
                aCurRadScPointB = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1);        // current scale on radiometry 2nd point of triangle
                aCurRadTrPointC = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2);  // current translation on radiometry 3rd point of triangle
                aCurRadScPointC = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2);      // current scale on radiometry 3rd point of triangle
            }
            else if (mSerialiseTriangleNodes && aVectorOfTriangleNodes != nullptr)
            {
                // current translation 1st point of triangle
                aCurTrPointA = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0, aNodeCounter,
                                                                                aIndicesOfTriKnots, true, aVectorOfTriangleNodes);
                // current translation 2nd point of triangle
                aCurTrPointB = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1, aNodeCounter + 1,
                                                                                aIndicesOfTriKnots, true, aVectorOfTriangleNodes);
                // current translation 3rd point of triangle
                aCurTrPointC = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2, aNodeCounter + 2,
                                                                                aIndicesOfTriKnots, true, aVectorOfTriangleNodes);

                // current translation on radiometry 1st point of triangle
                aCurRadTrPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0, aNodeCounter,
                                                                                            aIndicesOfTriKnots, false, aVectorOfTriangleNodes);
                // current scaling on radiometry 1st point of triangle
                aCurRadScPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 0, 1, 2, 3, aTri, 0, aNodeCounter,
                                                                                        aIndicesOfTriKnots, false, aVectorOfTriangleNodes);
                // current translation on radiometry 2nd point of triangle
                aCurRadTrPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1, aNodeCounter + 1,
                                                                                            aIndicesOfTriKnots, false, aVectorOfTriangleNodes);
                // current scaling on radiometry 2nd point of triangle
                aCurRadScPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 4, 5, 6, 7, aTri, 1, aNodeCounter + 1,
                                                                                        aIndicesOfTriKnots, false, aVectorOfTriangleNodes);
                // current translation on radiometry 3rd point of triangle
                aCurRadTrPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2, aNodeCounter + 2,
                                                                                            aIndicesOfTriKnots, false, aVectorOfTriangleNodes);
                // current scaling on radiometry 3rd point of triangle
                aCurRadScPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSol, aVecInd, 8, 9, 10, 11, aTri, 2, aNodeCounter + 2,
                                                                                        aIndicesOfTriKnots, false, aVectorOfTriangleNodes);
                aNodeCounter += 3;
            }

            // soft constraint x-translation
            if (!mFreezeTranslationX)
            {
                const bool aSoftApplicationTranslationX = ((mWeightTranslationX >= 0 && !aSoftFreezeForFirstItersTranslation) ||
                                                           (aSoftFreezeForFirstItersTranslation && mWeightTranslationX >= 0 &&
                                                            aIterNumber < mNumberOfItersToSoftFreezeTranslation));
                if (aSoftApplicationTranslationX)
                    ApplySoftConstraintToUnknown(0, 4, aVecInd, mSys, aVCurSol, mWeightTranslationX);
            }

            // soft constraint y-translation
            if (!mFreezeTranslationY)
            {
                const bool aSoftApplicationTranslationY = ((mWeightTranslationY >= 0 && !aSoftFreezeForFirstItersTranslation) ||
                                                           (aSoftFreezeForFirstItersTranslation && mWeightTranslationX >= 0 &&
                                                            aIterNumber < mNumberOfItersToSoftFreezeTranslation));
                if (aSoftApplicationTranslationY)
                    ApplySoftConstraintToUnknown(1, 4, aVecInd, mSys, aVCurSol, mWeightTranslationY);
            }

            // soft constraint radiometric translation
            if (!mFreezeRadTranslation)
            {
                const bool aSoftApplicationRadiometryTranslation = ((mWeightRadTranslation >=0 && !aSoftFreezeForFirstItersRadiometry) ||
                                                                    (aSoftFreezeForFirstItersRadiometry && mWeightRadTranslation && mWeightRadTranslation >= 0 &&
                                                                     aIterNumber < mNumberOfItersToHardFreezeRadiometry));
                if (aSoftApplicationRadiometryTranslation)
                    ApplySoftConstraintToUnknown(2, 4, aVecInd, mSys, aVCurSol, mWeightRadTranslation);
            }

            // soft constraint radiometric scaling
            if (!mFreezeRadScale)
            {
                const bool aSoftApplicationRadiometryScale = ((mWeightRadScale >=0 && !aSoftFreezeForFirstItersRadiometry) ||
                                                              (aSoftFreezeForFirstItersRadiometry && mWeightRadScale && mWeightRadScale >= 0 &&
                                                               aIterNumber < mNumberOfItersToHardFreezeRadiometry));
                if (aSoftApplicationRadiometryScale)
                    ApplySoftConstraintToUnknown(3, 4, aVecInd, mSys, aVCurSol, mWeightRadScale);
            }

            const size_t aNumberOfInsidePixels = aVectorToFillWithInsidePixels.size();

            // Loop over all pixels inside triangle
            // size_t is necessary as there can be a lot of pixels in triangles
            for (size_t aFilledPixel = 0; aFilledPixel < aNumberOfInsidePixels; aFilledPixel++)
            {
                const cPtInsideTriangles aPixInsideTriangle = cPtInsideTriangles(aCompTri, aVectorToFillWithInsidePixels,
                                                                                 aFilledPixel, aCurPreDIm);
                // prepare for barycenter translation formula by filling aVObs with different coordinates
                FormalInterpBarycenter_SetObs(aVObs, 0, aPixInsideTriangle);

                // image of a point in triangle by current translation
                const tPt2dr aTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aCurTrPointA, aCurTrPointB,
                                                                                                     aCurTrPointC, aVObs);

                // radiometry translation of pixel by current radiometry translation of triangle knots
                const tREAL8 aRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aCurRadTrPointA,
                                                                                                                aCurRadTrPointB,
                                                                                                                aCurRadTrPointC,
                                                                                                                aVObs);
                // radiometry translation of pixel by current radiometry scaling of triangle knots
                const tREAL8 aRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aCurRadScPointA,
                                                                                                        aCurRadScPointB,
                                                                                                        aCurRadScPointC,
                                                                                                        aVObs);

                const bool aPixInside = (mUseMMVIIInterpolators) ? aCurPostDIm->InsideInterpolator(*mInterpol, aTranslatedFilledPoint, 0) : aCurPostDIm->InsideBL(aTranslatedFilledPoint);
                if (aPixInside)
                {
                    (mUseMMVIIInterpolators) ?
                                             // prepare for application of linear gradient formula
                        FormalGradInterpolTri_SetObs(aVObs, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint,
                                                     aCurPostDIm, mInterpol)
                                             :
                                             // prepare for application of bilinear formula
                        FormalBilinTri_SetObs(aVObs, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint, aCurPostDIm);

                    // Now add observations
                    mSys->CalcAndAddObs(mEqTriDeform, aVecInd, aVObs);

                    const tREAL8 anInterpolatedValue = (mUseMMVIIInterpolators) ? aCurPostDIm->GetValueInterpol(*mInterpol, aTranslatedFilledPoint) : aCurPostDIm->GetVBL(aTranslatedFilledPoint);
                    // compute indicators
                    const tREAL8 aRadValueImPre = aRadiometryScaling * aVObs[5] + aRadiometryTranslation;
                    const tREAL8 aDif = aRadValueImPre - anInterpolatedValue; // residual
                    aSomDif += std::abs(aDif);
                }
                else
                    aNbOut++;

                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
            }
        }

        if (mUseMultiScaleApproach && !mIsLastIters && aIterNumber != 0)
        {
            const bool aGenerateIntermediateMaps = false; // if a generating intermediate displacement maps is wanted
            if (aGenerateIntermediateMaps)
                GenerateDisplacementMapsAndOutputImages(aVCurSol, aIterNumber, aTotalNumberOfIterations, aNonEmptyPathToFolder);
        }

        // Update all parameter taking into account previous observation
        mSys->SolveUpdateReset();

        // Save all triangle nodes to .xml file
        if (mSerialiseTriangleNodes && aVectorOfTriangleNodes != nullptr)
            aVectorOfTriangleNodes->MultipleNodesToFile(mNameMultipleTriangleNodes);

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_TriangleDeformation::GenerateDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber,
                                                                             const int aTotalNumberOfIterations, const bool aNonEmptyPathToFolder)
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

        std::unique_ptr<cMultipleTriangleNodesSerialiser> aLastVectorOfTriangleNodes = nullptr;

        if (mUseMultiScaleApproach && !mIsLastIters)
        {
            aLastPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
            aLastPreDIm = &aLastPreIm.DIm();
        }

        int aLastNodeCounter = 0;

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTri = mDelTri.KthTri(aLTr);
            const tPt3di aLastIndicesOfTriKnots = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTri);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            tIntVect aLastVecInd;
            GetIndicesVector(aLastVecInd, aLastIndicesOfTriKnots, 4);

            tPt2dr aLastTrPointA = tPt2dr(0, 0);
            tPt2dr aLastTrPointB = tPt2dr(0, 0);
            tPt2dr aLastTrPointC = tPt2dr(0, 0);
            tREAL8 aLastRadTrPointA = 0;
            tREAL8 aLastRadScPointA = 1;
            tREAL8 aLastRadTrPointB = 0;
            tREAL8 aLastRadScPointB = 1;
            tREAL8 aLastRadTrPointC = 0;
            tREAL8 aLastRadScPointC = 1;

            if (!mSerialiseTriangleNodes)
            {
                // last translation 1st point of triangle
                aLastTrPointA = LoadNodeAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
                // last translation 2nd point of triangle
                aLastTrPointB = LoadNodeAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
                // last translation 3rd point of triangle
                aLastTrPointC = LoadNodeAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);

                // last radiometry translation of 1st point
                aLastRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
                // last radiometry scaling of 1st point
                aLastRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
                // last radiometry translation of 2nd point
                aLastRadTrPointB = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
                // last radiometry scaling of 2nd point
                aLastRadScPointB = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
                // last radiometry translation of 3rd point
                aLastRadTrPointC = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);
                // last radiometry scaling of 3rd point
                aLastRadScPointC = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);
            }
            else
            {
                // last translation 1st point of triangle
                aLastTrPointA = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0, aLastNodeCounter,
                                                                                 aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last translation 2nd point of triangle
                aLastTrPointB = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1, aLastNodeCounter + 1,
                                                                                 aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last translation 3rd point of triangle
                aLastTrPointC = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2, aLastNodeCounter + 2,
                                                                                 aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);

                // last radiometry translation of 1st point
                aLastRadTrPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0, aLastNodeCounter,
                                                                                             aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last radiometry scaling of 1st point
                aLastRadScPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0, aLastNodeCounter,
                                                                                         aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last radiometry translation of 2nd point
                aLastRadTrPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1, aLastNodeCounter + 1,
                                                                                             aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last radiometry scaling of 2nd point
                aLastRadScPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1, aLastNodeCounter + 1,
                                                                                         aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last radiometry translation of 3rd point
                aLastRadTrPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2, aLastNodeCounter + 2,
                                                                                             aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                // last radiometry scaling of 3rd point
                aLastRadScPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2, aLastNodeCounter + 2,
                                                                                         aLastIndicesOfTriKnots, false, aLastVectorOfTriangleNodes);
                aLastNodeCounter += 3;
            }

            const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

            for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
            {
                const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                     aLastFilledPixel, aLastPreDIm);
 
                // image of a point in triangle by current translation
                const tPt2dr aLastTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
                                                                                                         aLastTrPointC, aLastPixInsideTriangle);

                const tREAL8 aLastRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aLastRadTrPointA,
                                                                                                                    aLastRadTrPointB,
                                                                                                                    aLastRadTrPointC,
                                                                                                                    aLastPixInsideTriangle);

                const tREAL8 aLastRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aLastRadScPointA,
                                                                                                            aLastRadScPointB,
                                                                                                            aLastRadScPointC,
                                                                                                            aLastPixInsideTriangle);

                FillDisplacementMapsAndOutputImage(aLastPixInsideTriangle, aLastTranslatedFilledPoint,
                                                   aLastRadiometryTranslation, aLastRadiometryScaling,
                                                   mSzImOut, mDImDepX, mDImDepY, mDImOut, aLastPostDIm);
            }
        }

        // save displacement maps in x and y to image files
        if (mUseMultiScaleApproach)
        {
            SaveMultiScaleDisplacementMapsToFile(mDImDepX, mDImDepY, aNonEmptyPathToFolder,
                                                 mUserDefinedFolderNameSaveResult, "DisplacedPixelsX_iter", "DisplacedPixelsY_iter", aIterNumber,
                                                 mNumberPointsToGenerate, aTotalNumberOfIterations);
            if (aIterNumber == aTotalNumberOfIterations - 1)
                SaveOutputImageToFile(mDImOut, aNonEmptyPathToFolder, mUserDefinedFolderNameSaveResult, "DisplacedPixels",
                                      mNumberPointsToGenerate, aTotalNumberOfIterations);
        }
        if (mInitialiseTranslationWithPreviousExecution)
        {
            mDImDepX->ToFile(mNameIntermediateDepX + ".tif");
            mDImDepY->ToFile(mNameIntermediateDepY + ".tif");
        }
        if (!mUseMultiScaleApproach && (aIterNumber == aTotalNumberOfIterations - 1))
        {
            SaveFinalDisplacementMapsToFile(mDImDepX, mDImDepY, aNonEmptyPathToFolder, mUserDefinedFolderNameSaveResult, "DisplacedPixelsX",
                                            "DisplacedPixelsY", mNumberPointsToGenerate, aTotalNumberOfIterations);
            SaveOutputImageToFile(mDImOut, aNonEmptyPathToFolder, mUserDefinedFolderNameSaveResult, "DisplacedPixels",
                                  mNumberPointsToGenerate, aTotalNumberOfIterations);
        }
    }

    void cAppli_TriangleDeformation::GenerateDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                                          const bool aDisplayLastRadiometryValues, 
                                                                                          const bool aDisplayLastTranslationValues,
                                                                                          const bool aNonEmptyPathToFolder)
    {
        tDenseVect aVFinalSol = mSys->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMapsAndOutputImages(aVFinalSol, aIterNumber, aTotalNumberOfIterations, aNonEmptyPathToFolder);

        if (aDisplayLastRadiometryValues || aDisplayLastTranslationValues)
            DisplayLastUnknownValuesAndComputeStatistics(aVFinalSol, aDisplayLastRadiometryValues, aDisplayLastTranslationValues, 4);
    }

    void cAppli_TriangleDeformation::DoOneIteration(const int aIterNumber, const int aTotalNumberOfIterations,
                                                    const bool aNonEmptyPathToFolder, const bool aHardFreezeForFirstItersTranslation,
                                                    const bool aHardFreezeForFirstItersRadiometry, const bool aSoftFreezeForFirstItersTranslation,
                                                    const bool aSoftFreezeForFirstItersRadiometry)
    {
        LoopOverTrianglesAndUpdateParameters(aIterNumber, aTotalNumberOfIterations,
                                             aNonEmptyPathToFolder, aHardFreezeForFirstItersTranslation,
                                             aHardFreezeForFirstItersRadiometry, aSoftFreezeForFirstItersTranslation,
                                             aSoftFreezeForFirstItersRadiometry); // Iterate over triangles and solve system

        // Show final translation results and produce displacement maps
        if (aIterNumber == (aTotalNumberOfIterations - 1))
            GenerateDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber, aTotalNumberOfIterations,
                                                                 mDisplayLastRadiometryValues, mDisplayLastTranslationValues,
                                                                 aNonEmptyPathToFolder);
    }

    //-----------------------------------------

    int cAppli_TriangleDeformation::Exe()
    {
        // read pre and post images and update their sizes
        ReadImageFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
        ReadImageFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

        const bool aNonEmptyFolderName = CheckFolderExistence(mUserDefinedFolderNameSaveResult);

        if (mUseMultiScaleApproach)
            mSigmaGaussFilter = mNumberOfScales * mSigmaGaussFilterStep;

        if (mComputeAvgMax)
            SubtractPrePostImageAndComputeAvgAndMax(mImDiff, mDImDiff, mDImPre,
                                                    mDImPost, mSzImPre); // Size of ImPre and ImPost are the same

        bool aHardFreezeForFirstItersTranslation = false;
        bool aHardFreezeForFirstItersRadiometry = false;
        bool aSoftFreezeForFirstItersTranslation = false;
        bool aSoftFreezeForFirstItersRadiometry = false;
        if (mNumberOfItersToHardFreezeTranslation > 0)
            aHardFreezeForFirstItersTranslation = true;
        if (mNumberOfItersToHardFreezeRadiometry > 0)
            aHardFreezeForFirstItersRadiometry = true;
        if (mNumberOfItersToSoftFreezeTranslation)
            aSoftFreezeForFirstItersTranslation = true;
        if (mNumberOfItersToSoftFreezeRadiometry)
            aSoftFreezeForFirstItersRadiometry = true;

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        // Generate triangulated knots coordinates
        DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
                                                        mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);
        // Initialise equation and interpolators if needed
        InitialiseInterpolationAndEquation(mEqTriDeform, mInterpol, mInterpolArgs, mUseMMVIIInterpolators);

        // If initialisation with previous excution is not wanted initialise the problem with zeros everywhere apart from radiometry scaling, with one
        if ((!mInitialiseTranslationWithPreviousExecution && !mInitialiseRadiometryWithPreviousExecution) || mInitialiseWithUserValues)
            InitialisationWithUserValues(mDelTri, mSys, mInitialiseWithUserValues, mInitialiseXTranslationValue,
                                         mInitialiseYTranslationValue, mInitialiseRadTrValue, mInitialiseRadScValue);
        else
        {
            if (mIsFirstExecution && mInitialiseWithMMVI)
                InitialiseWithPreviousExecutionValues(mDelTri, mSys, mNameInitialDepX, mImIntermediateDepX,
                                                      mDImIntermediateDepX, mSzImIntermediateDepX, mNameInitialDepY,
                                                      mImIntermediateDepY, mDImIntermediateDepY, mSzImDepY,
                                                      mNameCorrelationMaskMMVI, mImCorrelationMask,
                                                      mDImCorrelationMask, mSzCorrelationMask);
            else if (!mIsFirstExecution && mInitialiseWithMMVI)
                InitialiseWithPreviousExecutionValues(mDelTri, mSys, mNameInitialDepX, mImIntermediateDepX,
                                                      mDImIntermediateDepX, mSzImIntermediateDepX, mNameInitialDepY,
                                                      mImIntermediateDepY, mDImIntermediateDepY, mSzImDepY,
                                                      mNameCorrelationMaskMMVI, mImCorrelationMask,
                                                      mDImCorrelationMask, mSzCorrelationMask);
        }

        const tDenseVect aVInitSol = mSys->CurGlobSol().Dup(); // Duplicate initial solution

        int aTotalNumberOfIterations = 0;
        (mUseMultiScaleApproach) ? aTotalNumberOfIterations = mNumberOfScales + mNumberOfEndIterations : aTotalNumberOfIterations = mNumberOfScales;

        for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
            DoOneIteration(aIterNumber, aTotalNumberOfIterations, aNonEmptyFolderName,
                           aHardFreezeForFirstItersTranslation, aHardFreezeForFirstItersRadiometry,
                           aSoftFreezeForFirstItersTranslation, aSoftFreezeForFirstItersRadiometry);

        return EXIT_SUCCESS;
    }

    /********************************************/
    //              ::MMVII                     //
    /********************************************/

    tMMVII_UnikPApli Alloc_cAppli_TriangleDeformation(const std::vector<std::string> &aVArgs,
                                                      const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_TriangleDeformation(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_ComputeTriangleDeformation(
        "ComputeTriangleDeformation",
        Alloc_cAppli_TriangleDeformation,
        "Compute 2D deformation between images using triangular mesh",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII
