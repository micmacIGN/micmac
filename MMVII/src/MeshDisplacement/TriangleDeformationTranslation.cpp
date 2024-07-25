#include "TriangleDeformationTranslation.h"

/**
   \file TriangleDeformationTranslation.cpp

   \brief file for computing 2D translation between 2 images
   thanks to triangles.
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
                                                                                                                  mUseLinearGradInterpolation(false),
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
                                                                                                                  mFreezeTranslationX(false),
                                                                                                                  mFreezeTranslationY(false),
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
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach optimisation process.");
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
               << AOpt2007(mUseLinearGradInterpolation, "UseLinearGradientInterpolation",
                           "Use linear gradient interpolation instead of bilinear interpolation.", {eTA2007::HDV})
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
               << AOpt2007(mFreezeTranslationX, "FreezeXTranslation",
                           "Whether to freeze or not x-translation to certain value during computation.", {eTA2007::HDV})
               << AOpt2007(mFreezeTranslationY, "FreezeYTranslation",
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
        //----------- allocate vec of obs :
        // 6 for ImagePre and 5 for ImagePost in linear gradient case and 6 in bilinear case
        const int aNumberOfObsTr = mUseLinearGradInterpolation ? TriangleDisplacement_GradInterpol_NbObs : TriangleDisplacement_Bilin_NbObs;
        tDoubleVect aVObsTr(6 + aNumberOfObsTr, 0);

        //----------- extract current parameters
        tDenseVect aVCurSolTr = mSysTranslation->CurGlobSol(); // Get current solution.

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

            const bool aSaveGaussImage = false;
            if (aSaveGaussImage)
                aCurPreDIm->ToFile("GaussFilteredImPre_iter_" + ToStr(aIterNumber) + ".tif");
        }

        //----------- declaration of indicator of convergence
        tREAL8 aSomDif = 0; // sum of difference between untranslated pixel and translated one.
        size_t aNbOut = 0;  // number of translated pixels out of image

        // Count number of pixels inside triangles for normalisation
        size_t aTotalNumberOfInsidePixels = 0;
        // Id of points
        int aNodeCounterTr = 0;

        std::unique_ptr<cMultipleTriangleNodesSerialiser> aVectorOfTriangleNodesTr = nullptr;

        if (mSerialiseTriangleNodes && aVectorOfTriangleNodesTr == nullptr)
            aVectorOfTriangleNodesTr = cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(mNameMultipleTriangleNodes);

        if (mFreezeTranslationX || mFreezeTranslationY)
        {
            for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
            {
                const tPt3di aIndicesOfTriKnotsTr = mDelTri.KthFace(aTr);

                const tIntVect aVecIndTr = {2 * aIndicesOfTriKnotsTr.x(), 2 * aIndicesOfTriKnotsTr.x() + 1,
                                            2 * aIndicesOfTriKnotsTr.y(), 2 * aIndicesOfTriKnotsTr.y() + 1,
                                            2 * aIndicesOfTriKnotsTr.z(), 2 * aIndicesOfTriKnotsTr.z() + 1};

                if (mFreezeTranslationX)
                {
                    const int aFirstTrXIndices = aVecIndTr.at(0);
                    const int aSecondTrXIndices = aVecIndTr.at(2);
                    const int aThirdTrXIndices = aVecIndTr.at(4);
                    mSysTranslation->SetFrozenVar(aFirstTrXIndices, aVCurSolTr(aFirstTrXIndices));
                    mSysTranslation->SetFrozenVar(aSecondTrXIndices, aVCurSolTr(aSecondTrXIndices));
                    mSysTranslation->SetFrozenVar(aThirdTrXIndices, aVCurSolTr(aThirdTrXIndices));
                }
                if (mFreezeTranslationY)
                {
                    const int aFirstTrYIndices = aVecIndTr.at(1);
                    const int aSecondTrYIndices = aVecIndTr.at(3);
                    const int aThirdTrYIndices = aVecIndTr.at(5);
                    mSysTranslation->SetFrozenVar(aFirstTrYIndices, aVCurSolTr(aFirstTrYIndices));
                    mSysTranslation->SetFrozenVar(aSecondTrYIndices, aVCurSolTr(aSecondTrYIndices));
                    mSysTranslation->SetFrozenVar(aThirdTrYIndices, aVCurSolTr(aThirdTrYIndices));
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

            //----------- index of unknown, finds the associated pixels of current triangle
            const tIntVect aVecIndTr = {2 * aIndicesOfTriKnotsTr.x(), 2 * aIndicesOfTriKnotsTr.x() + 1,
                                        2 * aIndicesOfTriKnotsTr.y(), 2 * aIndicesOfTriKnotsTr.y() + 1,
                                        2 * aIndicesOfTriKnotsTr.z(), 2 * aIndicesOfTriKnotsTr.z() + 1};

            tPt2dr aCurTrPointA = tPt2dr(0, 0);
            tPt2dr aCurTrPointB = tPt2dr(0, 0);
            tPt2dr aCurTrPointC = tPt2dr(0, 0);

            if (!mSerialiseTriangleNodes && aVectorOfTriangleNodesTr == nullptr)
            {
                // current translation 1st point of triangle
                aCurTrPointA = LoadNodeAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 0, 1, 0, 1, aTriTr, 0);
                // current translation 2nd point of triangle
                aCurTrPointB = LoadNodeAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 2, 3, 2, 3, aTriTr, 1);
                // current translation 3rd point of triangle
                aCurTrPointC = LoadNodeAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 4, 5, 4, 5, aTriTr, 2);
            }
            else if (mSerialiseTriangleNodes && aVectorOfTriangleNodesTr != nullptr)
            {
                // current translation 1st point of triangle
                aCurTrPointA = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 0, 1, 0, 1, aTriTr, 0, aNodeCounterTr, 
                                                                                aIndicesOfTriKnotsTr, true, aVectorOfTriangleNodesTr);
                // current translation 2nd point of triangle
                aCurTrPointB = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 2, 3, 2, 3, aTriTr, 1, aNodeCounterTr + 1,
                                                                                aIndicesOfTriKnotsTr, true, aVectorOfTriangleNodesTr);
                // current translation 3rd point of triangle
                aCurTrPointC = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVCurSolTr, aVecIndTr, 4, 5, 4, 5, aTriTr, 2, aNodeCounterTr + 2, 
                                                                                aIndicesOfTriKnotsTr, true, aVectorOfTriangleNodesTr);
                aNodeCounterTr += 3;
            }

            // soft constraint x-translation
            if (!mFreezeTranslationX)
            {
                if (mWeightTranslationX >= 0)
                {
                    const int aSolStart = 0;
                    const int aSolStep = 2; // adapt step to solution vector configuration
                    for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecIndTr.size() - 1; aIndCurSol += aSolStep)
                    {
                        const int aIndices = aVecIndTr.at(aIndCurSol);
                        mSysTranslation->AddEqFixCurVar(aIndices, mWeightTranslationX);
                    }
                }
            }

            // soft constraint y-translation
            if (!mFreezeTranslationY)
            {
                if (mWeightTranslationY >= 0)
                {
                    const int aSolStart = 1;
                    const int aSolStep = 2; // adapt step to solution vector configuration
                    for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecIndTr.size(); aIndCurSol += aSolStep)
                    {
                        const int aIndices = aVecIndTr.at(aIndCurSol);
                        mSysTranslation->AddEqFixCurVar(aIndices, mWeightTranslationY);
                    }
                }
            }

            const size_t aNumberOfInsidePixels = aVectorToFillWithInsidePixels.size();

            // Loop over all pixels inside triangle
            // size_t is necessary as there can be a lot of pixels in triangles
            for (size_t aFilledPixel = 0; aFilledPixel < aNumberOfInsidePixels; aFilledPixel++)
            {
                const cPtInsideTriangles aPixInsideTriangle = cPtInsideTriangles(aCompTri, aVectorToFillWithInsidePixels,
                                                                                 aFilledPixel, aCurPreDIm);
                // prepare for barycenter translation formula by filling aVObsTr with different coordinates
                FormalInterpBarycenter_SetObs(aVObsTr, 0, aPixInsideTriangle);

                // image of a point in triangle by current translation
                const tPt2dr aTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aCurTrPointA, aCurTrPointB,
                                                                                                     aCurTrPointC, aVObsTr);

                const bool aPixInside = (mUseLinearGradInterpolation) ? aCurPostDIm->InsideInterpolator(*mInterpolTr, aTranslatedFilledPoint, 0) : aCurPostDIm->InsideBL(aTranslatedFilledPoint);
                if (aPixInside)
                {
                    if (mUseLinearGradInterpolation)
                        // prepare for application of linear gradient formula
                        FormalGradInterpol_SetObs(aVObsTr, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint,
                                                  *aCurPostDIm, *mInterpolTr);
                    else
                        // prepare for application of bilinear formula
                        FormalBilinTri_SetObs(aVObsTr, TriangleDisplacement_NbObs_ImPre, aTranslatedFilledPoint, *aCurPostDIm);

                    // Now add observation
                    mSysTranslation->CalcAndAddObs(mEqTranslationTri, aVecIndTr, aVObsTr);

                    const tREAL8 aInterpolatedValue = (mUseLinearGradInterpolation) ? aCurPostDIm->GetValueInterpol(*mInterpolTr, aTranslatedFilledPoint) : aCurPostDIm->GetVBL(aTranslatedFilledPoint);
                    // compute indicators
                    const tREAL8 aDif = aVObsTr[5] - aInterpolatedValue; // residual
                    aSomDif += std::abs(aDif);
                }
                else
                    aNbOut++;

                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
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
        if (mSerialiseTriangleNodes && aVectorOfTriangleNodesTr != nullptr)
            aVectorOfTriangleNodesTr->MultipleNodesToFile(mNameMultipleTriangleNodes);

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_TriangleDeformationTranslation::GenerateDisplacementMaps(const tDenseVect &aVFinalSolTr, const int aIterNumber,
                                                                         const int aTotalNumberOfIterations, const bool aNonEmptyPathToFolder)
    {
        // Initialise output image, x and y displacement maps
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = mDImOut->Sz();

        InitialiseDisplacementMaps(mSzImPre, mImDepX, mDImDepX, mSzImDepX);
        InitialiseDisplacementMaps(mSzImPre, mImDepY, mDImDepY, mSzImDepY);

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

        // prefill output image with ImPre pixels to not have null values
        for (const tPt2di &aOutPix : *mDImOut)
            mDImOut->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTriTr = mDelTri.KthTri(aLTr);
            const tPt3di aLastIndicesOfTriKnotsTr = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTriTr);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecIndTr = {2 * aLastIndicesOfTriKnotsTr.x(), 2 * aLastIndicesOfTriKnotsTr.x() + 1,
                                            2 * aLastIndicesOfTriKnotsTr.y(), 2 * aLastIndicesOfTriKnotsTr.y() + 1,
                                            2 * aLastIndicesOfTriKnotsTr.z(), 2 * aLastIndicesOfTriKnotsTr.z() + 1};

            tPt2dr aLastTrPointA = tPt2dr(0, 0);
            tPt2dr aLastTrPointB = tPt2dr(0, 0);
            tPt2dr aLastTrPointC = tPt2dr(0, 0);

            if (!mSerialiseTriangleNodes)
            {
                aLastTrPointA = LoadNodeAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 0, 1, 0, 1, aLastTriTr, 0);   // last translation 1st point of triangle
                aLastTrPointB = LoadNodeAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 2, 3, 2, 3, aLastTriTr, 1);   // last translation 2nd point of triangle
                aLastTrPointC = LoadNodeAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 4, 5, 4, 5, aLastTriTr, 2);   // last translation 3rd point of triangle
            }
            else
            {
                // last translation 1st point of triangle
                aLastTrPointA = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 0, 1, 0, 1, aLastTriTr, 0, aLastNodeCounterTr,
                                                                                 aLastIndicesOfTriKnotsTr, false, aLastVectorOfTriangleNodesTr);
                // last translation 2nd point of triangle
                aLastTrPointB = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 2, 3, 2, 3, aLastTriTr, 1, aLastNodeCounterTr + 1,
                                                                                 aLastIndicesOfTriKnotsTr, false, aLastVectorOfTriangleNodesTr);
                // last translation 3rd point of triangle
                aLastTrPointC = LoadNodeAppendVectorAndReturnCurrentDisplacement(aVFinalSolTr, aLastVecIndTr, 4, 5, 4, 5, aLastTriTr, 2, aLastNodeCounterTr + 2,
                                                                                 aLastIndicesOfTriKnotsTr, false, aLastVectorOfTriangleNodesTr);
                aLastNodeCounterTr += 3;
            }

            const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

            for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
            {
                const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                     aLastFilledPixel, aLastPreDIm);

                // image of a point in triangle by current translation
                const tPt2dr aLastTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
                                                                                                         aLastTrPointC, aLastPixInsideTriangle);

                FillDisplacementMapsTranslation(aLastPixInsideTriangle, aLastTranslatedFilledPoint,
                                                mSzImOut, mDImDepX, mDImDepY, mDImOut);
            }
        }

        // save displacement maps in x and y to image files
        if (mUseMultiScaleApproach)
        {
            SaveMultiScaleDisplacementMapsToFile(mDImDepX, mDImDepY, aNonEmptyPathToFolder, mUserDefinedFolderNameToSaveResult, "DisplacedPixelsX_iter",
                                                 "DisplacedPixelsY_iter", aIterNumber, mNumberPointsToGenerate, aTotalNumberOfIterations);
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
                                                                                                       const tDenseVect &aVinitVecSol,
                                                                                                       const bool aNonEmptyPathToFolder)
    {
        tDenseVect aVFinalSolTr = mSysTranslation->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMaps(aVFinalSolTr, aIterNumber, aTotalNumberOfIterations, aNonEmptyPathToFolder);

        if (mDisplayLastTranslationValues)
            DisplayLastUnknownValuesAndComputeStatistics(aVFinalSolTr, aVinitVecSol);
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
        // read pre and post images and update their sizes
        ReadFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
        ReadFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

        bool aNonEmptyPathToFolder = false;
        if (!mUserDefinedFolderNameToSaveResult.empty())
        {
            aNonEmptyPathToFolder = true;
            if (!ExistFile(mUserDefinedFolderNameToSaveResult))
                CreateDirectories(mUserDefinedFolderNameToSaveResult, aNonEmptyPathToFolder);
        }

        if (mUseMultiScaleApproach)
            mSigmaGaussFilter = mNumberOfScales * mSigmaGaussFilterStep;

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
                                                        mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);

        InitialiseInterpolationAndEquationTranslation(mEqTranslationTri, mInterpolTr, mInterpolArgs, mUseLinearGradInterpolation);

        if (!mInitialiseTranslationWithPreviousExecution || mInitialiseWithUserValues)
            InitialiseWithUserValuesTranslation(mDelTri, mSysTranslation, mInitialiseWithUserValues,
                                                mInitialiseXTranslationValue, mInitialiseYTranslationValue);
        else
        {
            if (mIsFirstExecution && mInitialiseWithMMVI)
                InitialiseWithPreviousExecutionValuesTranslation(mDelTri, mSysTranslation,
                                                                 mNameInitialDepX, mImIntermediateDepX,
                                                                 mDImIntermediateDepX, mSzImIntermediateDepX,
                                                                 mNameInitialDepY, mImIntermediateDepY,
                                                                 mDImIntermediateDepY, mSzImIntermediateDepY,
                                                                 mNameCorrelationMaskMMVI, mImCorrelationMask,
                                                                 mDImCorrelationMask, mSzCorrelationMask);

            else if (!mIsFirstExecution && mInitialiseWithMMVI)
                InitialiseWithPreviousExecutionValuesTranslation(mDelTri, mSysTranslation,
                                                                 mNameIntermediateDepX, mImIntermediateDepX,
                                                                 mDImIntermediateDepX, mSzImIntermediateDepX,
                                                                 mNameIntermediateDepY, mImIntermediateDepY,
                                                                 mDImIntermediateDepY, mSzImIntermediateDepY,
                                                                 mNameCorrelationMaskMMVI, mImCorrelationMask,
                                                                 mDImCorrelationMask, mSzCorrelationMask);
        }

        const tDenseVect aVInitSolTr = mSysTranslation->CurGlobSol().Dup(); // Duplicate initial solution

        int aTotalNumberOfIterations = 0;
        (mUseMultiScaleApproach) ? aTotalNumberOfIterations = mNumberOfScales + mNumberOfEndIterations : aTotalNumberOfIterations = mNumberOfScales;

        for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
            DoOneIterationTranslation(aIterNumber, aTotalNumberOfIterations, aVInitSolTr, aNonEmptyPathToFolder);

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
        "Compute 2D translation deformations between images using triangles",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII
