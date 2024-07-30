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
                                                                                 const cSpecMMVII_Appli &aSpec) : cAppli_TriangleDeformation(aVArgs, aSpec),
                                                                                                                  mNumberOfLines(1),
                                                                                                                  mNumberOfCols(1),
                                                                                                                  mShow(true),
                                                                                                                  mUseMultiScaleApproach(false),
                                                                                                                  mBuildRandomUniformGrid(false),
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
                                                                                                                  mSysTranslation(nullptr),
                                                                                                                  mEqTranslationTri(nullptr)
    {
        mEqTranslationTri = EqDeformTriTranslation(true, 1); // true means with derivative, 1 is size of buffer
    }

    cAppli_TriangleDeformationTranslation::~cAppli_TriangleDeformationTranslation()
    {
        delete mSysTranslation;
        delete mEqTranslationTri;
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
               << AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationValues",
                           "Whether to display the final coordinates of the trainslated points.", {eTA2007::HDV})
               << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning});
    }

    void cAppli_TriangleDeformationTranslation::LoopOverTrianglesAndUpdateParametersTranslation(const int aIterNumber,
                                                                                                const int aTotalNumberOfIterations)
    {
        //----------- allocate vec of obs :
        tDoubleVect aVObsTr(12, 0.0); // 6 for ImagePre and 6 for ImagePost

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
                    mSysTranslation->SetFrozenVar(aVecIndTr.at(0), aVCurSolTr(aVecIndTr.at(0)));
                    mSysTranslation->SetFrozenVar(aVecIndTr.at(2), aVCurSolTr(aVecIndTr.at(2)));
                    mSysTranslation->SetFrozenVar(aVecIndTr.at(4), aVCurSolTr(aVecIndTr.at(4)));
                }
                if (mFreezeTranslationY)
                {
                    mSysTranslation->SetFrozenVar(aVecIndTr.at(1), aVCurSolTr(aVecIndTr.at(1)));
                    mSysTranslation->SetFrozenVar(aVecIndTr.at(3), aVCurSolTr(aVecIndTr.at(3)));
                    mSysTranslation->SetFrozenVar(aVecIndTr.at(5), aVCurSolTr(aVecIndTr.at(5)));
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

            const cNodeOfTriangles aFirstPointOfTriTr = cNodeOfTriangles(aVCurSolTr, aVecIndTr, 0, 1, 0, 1, aTriTr, 0);
            const cNodeOfTriangles aSecondPointOfTriTr = cNodeOfTriangles(aVCurSolTr, aVecIndTr, 2, 3, 2, 3, aTriTr, 1);
            const cNodeOfTriangles aThirdPointOfTriTr = cNodeOfTriangles(aVCurSolTr, aVecIndTr, 4, 5, 4, 5, aTriTr, 2);

            const tPt2dr aCurTrPointA = aFirstPointOfTriTr.GetCurrentXYDisplacementValues();    // current translation 1st point of triangle
            const tPt2dr aCurTrPointB = aSecondPointOfTriTr.GetCurrentXYDisplacementValues();   // current translation 2nd point of triangle
            const tPt2dr aCurTrPointC = aThirdPointOfTriTr.GetCurrentXYDisplacementValues();    // current translation 3rd point of triangle

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

                if (aCurPostDIm->InsideBL(aTranslatedFilledPoint)) // avoid errors
                {
                    // prepare for application of bilinear formula
                    FormalBilinTri_SetObs(aVObsTr, TriangleDisplacement_NbObs, aTranslatedFilledPoint, *aCurPostDIm);

                    // Now add observation
                    mSysTranslation->CalcAndAddObs(mEqTranslationTri, aVecIndTr, aVObsTr);

                    // compute indicators
                    const tREAL8 aDif = aVObsTr[5] - aCurPostDIm->GetVBL(aTranslatedFilledPoint); // residual - aValueImPre - aCurPostDIm->GetVBL(aTranslatedFilledPoint)
                    aSomDif += std::abs(aDif);
                }
                else
                    aNbOut++; // Count number of pixels translated outside post image

                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
            }
        }

        if (mUseMultiScaleApproach && !mIsLastIters && aIterNumber != 0)
        {
            const bool aGenerateIntermediateMaps = false;
            if (aGenerateIntermediateMaps)
                GenerateDisplacementMaps(aVCurSolTr, aIterNumber, aTotalNumberOfIterations);
        }

        // Update all parameter taking into account previous observation
        mSysTranslation->SolveUpdateReset();

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_TriangleDeformationTranslation::GenerateDisplacementMaps(const tDenseVect &aVFinalSol, const int aIterNumber,
                                                                         const int aTotalNumberOfIterations)
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

        // prefill output image with ImPre pixels to not have null values
        for (const tPt2di &aOutPix : *mDImOut)
            mDImOut->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTri = mDelTri.KthTri(aLTr);
            const tPt3di aLastIndicesOfTriKnots = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTri);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecInd = {2 * aLastIndicesOfTriKnots.x(), 2 * aLastIndicesOfTriKnots.x() + 1,
                                          2 * aLastIndicesOfTriKnots.y(), 2 * aLastIndicesOfTriKnots.y() + 1,
                                          2 * aLastIndicesOfTriKnots.z(), 2 * aLastIndicesOfTriKnots.z() + 1};

            const cNodeOfTriangles aLastFirstPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 0, 1, 0, 1, aLastTri, 0);
            const cNodeOfTriangles aLastSecondPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 2, 3, 2, 3, aLastTri, 1);
            const cNodeOfTriangles aLastThirdPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 4, 5, 4, 5, aLastTri, 2);

            const tPt2dr aLastTrPointA = aLastFirstPointOfTri.GetCurrentXYDisplacementValues();  // last translation 1st point of triangle
            const tPt2dr aLastTrPointB = aLastSecondPointOfTri.GetCurrentXYDisplacementValues(); // last translation 2nd point of triangle
            const tPt2dr aLastTrPointC = aLastThirdPointOfTri.GetCurrentXYDisplacementValues();  // last translation 3rd point of triangle

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
            mDImDepX->ToFile("DisplacedPixelsX_iter_" + ToStr(aIterNumber) + "_" +
                             ToStr(mNumberPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            mDImDepY->ToFile("DisplacedPixelsY_iter_" + ToStr(aIterNumber) + "_" +
                             ToStr(mNumberPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            if (aIterNumber == aTotalNumberOfIterations - 1)
                mDImOut->ToFile("DisplacedPixels_iter_" + ToStr(aIterNumber) + "_" +
                                ToStr(mNumberPointsToGenerate) + "_" +
                                ToStr(aTotalNumberOfIterations) + ".tif");
        }
        if (mInitialiseTranslationWithPreviousExecution)
        {
            mDImDepX->ToFile(mNameIntermediateDepX);
            mDImDepY->ToFile(mNameIntermediateDepY);
        }
        if (!mUseMultiScaleApproach && (aIterNumber == aTotalNumberOfIterations - 1))
        {
            mDImDepX->ToFile("DisplacedPixelsX_" + ToStr(mNumberPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            mDImDepY->ToFile("DisplacedPixelsY_" + ToStr(mNumberPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            mDImOut->ToFile("DisplacedPixels_" + ToStr(mNumberPointsToGenerate) + "_" +
                            ToStr(aTotalNumberOfIterations) + ".tif");
        }
    }

    void cAppli_TriangleDeformationTranslation::GenerateDisplacementMapsAndDisplayLastTranslatedPoints(const int aIterNumber,
                                                                                                       const int aTotalNumberOfIterations,
                                                                                                       const tDenseVect &aVinitVecSol)
    {
        tDenseVect aVFinalSol = mSysTranslation->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMaps(aVFinalSol, aIterNumber, aTotalNumberOfIterations);

        if (mDisplayLastTranslationValues)
            DisplayLastUnknownValuesAndComputeStatistics(aVFinalSol, aVinitVecSol);
    }

    void cAppli_TriangleDeformationTranslation::DoOneIterationTranslation(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                          const tDenseVect &aVInitVecSol)
    {
        LoopOverTrianglesAndUpdateParametersTranslation(aIterNumber, aTotalNumberOfIterations); // Iterate over triangles and solve system

        // Show final translation results and produce displacement maps
        if (aIterNumber == (aTotalNumberOfIterations - 1))
            GenerateDisplacementMapsAndDisplayLastTranslatedPoints(aIterNumber, aTotalNumberOfIterations, aVInitVecSol);
    }

    //-----------------------------------------

    int cAppli_TriangleDeformationTranslation::Exe()
    {
        // read pre and post images and update their sizes
        ReadFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
        ReadFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

        if (mUseMultiScaleApproach)
            mSigmaGaussFilter = mNumberOfScales * mSigmaGaussFilterStep;

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
                                                        mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);

        if (!mInitialiseTranslationWithPreviousExecution || mInitialiseWithUserValues)
            InitialisationAfterExeTranslation(mDelTri, mSysTranslation, mInitialiseWithUserValues,
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
            DoOneIterationTranslation(aIterNumber, aTotalNumberOfIterations, aVInitSolTr);

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
