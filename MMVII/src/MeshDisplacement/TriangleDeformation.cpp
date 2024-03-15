#include "TriangleDeformation.h"

/**
   \file TriangleDeformation.cpp

   \brief file for computing 2D deformations between 2 images
   thanks to triangles.
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
                                                                                            mWeightRadTranslation(-1),
                                                                                            mWeightRadScale(-1),
                                                                                            mNumberOfIterGaussFilter(3),
                                                                                            mNumberOfEndIterations(2),
                                                                                            mFolderSaveResult(""),
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
                                                                                            mSys(nullptr),
                                                                                            mEqTriDeform(nullptr)
    {
        mEqTriDeform = EqDeformTri(true, 1); // true means with derivative, 1 is size of buffer
    }

    cAppli_TriangleDeformation::~cAppli_TriangleDeformation()
    {
        delete mSys;
        delete mEqTriDeform;
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
                << AOpt2007(mWeightRadTranslation, "WeightRadiometryTranslation",
                            "A value to weight radiometry translation for soft freezing of coefficient.", {eTA2007::HDV})
                << AOpt2007(mWeightRadScale, "WeightRadiometryScaling",
                            "A value to weight radiometry scaling for soft freezing of coefficient.", {eTA2007::HDV})
                << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                            "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
                << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                            "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning})
                << AOpt2007(mFolderSaveResult, "FolderToSaveResults",
                            "Folder name where to store produced results", {eTA2007::HDV})
                << AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationsValues",
                            "Whether to display the final values of unknowns linked to point translation.", {eTA2007::HDV})
                << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                            "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV});
    }

    void cAppli_TriangleDeformation::LoopOverTrianglesAndUpdateParameters(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                          const bool aUserDefinedFolderName)
    {
        //----------- allocate vec of obs :
        tDoubleVect aVObs(12, 0.0); // 6 for ImagePre and 6 for ImagePost

        //----------- extract current parameters
        tDenseVect aVCur = mSys->CurGlobSol(); // Get current solution.

        /*
        for (int aUnk=0; aUnk<aVCur.DIm().Sz(); aUnk++)
            StdOut() << aVCur(aUnk) << " " ;
        StdOut() << std::endl;
        */

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
                if (aUserDefinedFolderName)
                    aCurPreDIm->ToFile(mFolderSaveResult + "/GaussFilteredImPre_iter_" + ToStr(aIterNumber) + ".tif");
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

        // hard constraint : freeze radiometric coefficients
        if (mFreezeRadTranslation || mFreezeRadScale ||
            mFreezeTranslationX || mFreezeTranslationY)
        {
            for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
            {
                const tPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

                const tIntVect aVecInd = {4 * aIndicesOfTriKnots.x(), 4 * aIndicesOfTriKnots.x() + 1,
                                          4 * aIndicesOfTriKnots.x() + 2, 4 * aIndicesOfTriKnots.x() + 3,
                                          4 * aIndicesOfTriKnots.y(), 4 * aIndicesOfTriKnots.y() + 1,
                                          4 * aIndicesOfTriKnots.y() + 2, 4 * aIndicesOfTriKnots.y() + 3,
                                          4 * aIndicesOfTriKnots.z(), 4 * aIndicesOfTriKnots.z() + 1,
                                          4 * aIndicesOfTriKnots.z() + 2, 4 * aIndicesOfTriKnots.z() + 3};

                if (mFreezeTranslationX)
                {
                    mSys->SetFrozenVar(aVecInd.at(0), aVCur(aVecInd.at(0)));
                    mSys->SetFrozenVar(aVecInd.at(4), aVCur(aVecInd.at(4)));
                    mSys->SetFrozenVar(aVecInd.at(8), aVCur(aVecInd.at(8)));
                }
                if (mFreezeTranslationY)
                {
                    mSys->SetFrozenVar(aVecInd.at(1), aVCur(aVecInd.at(1)));
                    mSys->SetFrozenVar(aVecInd.at(5), aVCur(aVecInd.at(5)));
                    mSys->SetFrozenVar(aVecInd.at(9), aVCur(aVecInd.at(9)));
                }
                if (mFreezeRadTranslation)
                {
                    mSys->SetFrozenVar(aVecInd.at(2), aVCur(aVecInd.at(2)));
                    mSys->SetFrozenVar(aVecInd.at(6), aVCur(aVecInd.at(6)));
                    mSys->SetFrozenVar(aVecInd.at(10), aVCur(aVecInd.at(10)));
                }
                if (mFreezeRadScale)
                {
                    mSys->SetFrozenVar(aVecInd.at(3), aVCur(aVecInd.at(3)));
                    mSys->SetFrozenVar(aVecInd.at(7), aVCur(aVecInd.at(7)));
                    mSys->SetFrozenVar(aVecInd.at(11), aVCur(aVecInd.at(11)));
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
            const tIntVect aVecInd = {4 * aIndicesOfTriKnots.x(), 4 * aIndicesOfTriKnots.x() + 1,
                                      4 * aIndicesOfTriKnots.x() + 2, 4 * aIndicesOfTriKnots.x() + 3,
                                      4 * aIndicesOfTriKnots.y(), 4 * aIndicesOfTriKnots.y() + 1,
                                      4 * aIndicesOfTriKnots.y() + 2, 4 * aIndicesOfTriKnots.y() + 3,
                                      4 * aIndicesOfTriKnots.z(), 4 * aIndicesOfTriKnots.z() + 1,
                                      4 * aIndicesOfTriKnots.z() + 2, 4 * aIndicesOfTriKnots.z() + 3};

            const cNodeOfTriangles aFirstPointOfTri = cNodeOfTriangles(aVCur, aVecInd, 0, 1, 2, 3, aTri, 0);
            const cNodeOfTriangles aSecondPointOfTri = cNodeOfTriangles(aVCur, aVecInd, 4, 5, 6, 7, aTri, 1);
            const cNodeOfTriangles aThirdPointOfTri = cNodeOfTriangles(aVCur, aVecInd, 8, 9, 10, 11, aTri, 2);

            const tPt2dr aCurTrPointA = aFirstPointOfTri.GetCurrentXYDisplacementValues();  // current translation 1st point of triangle
            const tPt2dr aCurTrPointB = aSecondPointOfTri.GetCurrentXYDisplacementValues(); // current translation 2nd point of triangle
            const tPt2dr aCurTrPointC = aThirdPointOfTri.GetCurrentXYDisplacementValues();  // current translation 3rd point of triangle

            const tREAL8 aCurRadTrPointA = aFirstPointOfTri.GetCurrentRadiometryTranslation();  // current translation on radiometry 1st point of triangle
            const tREAL8 aCurRadScPointA = aFirstPointOfTri.GetCurrentRadiometryScaling();      // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointB = aSecondPointOfTri.GetCurrentRadiometryTranslation(); // current translation on radiometry 2nd point of triangle
            const tREAL8 aCurRadScPointB = aSecondPointOfTri.GetCurrentRadiometryScaling();     // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointC = aThirdPointOfTri.GetCurrentRadiometryTranslation();  // current translation on radiometry 3rd point of triangle
            const tREAL8 aCurRadScPointC = aThirdPointOfTri.GetCurrentRadiometryScaling();      // current scale on radiometry 3rd point of triangle

            // soft constraint radiometric translation
            if (!mFreezeRadTranslation)
            {
                if (mWeightRadTranslation >= 0)
                {
                    const int aSolStep = 4; // adapt step to solution vector configuration
                    const int aSolStart = 2;
                    for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size() - 1; aIndCurSol += aSolStep)
                    {
                        const int aIndices = aVecInd.at(aIndCurSol);
                        mSys->AddEqFixVar(aIndices, aVCur(aIndices), mWeightRadTranslation);
                    }
                }
            }

            // soft constraint radiometric scaling
            if (!mFreezeRadScale)
            {
                if (mWeightRadScale >= 0)
                {
                    const int aSolStep = 4; // adapt step to solution vector configuration
                    const int aSolStart = 3;
                    for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size(); aIndCurSol += aSolStep)
                    {
                        const int aIndices = aVecInd.at(aIndCurSol);
                        mSys->AddEqFixVar(aIndices, aVCur(aIndices), mWeightRadScale);
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

                if (aCurPostDIm->InsideBL(aTranslatedFilledPoint)) // avoid errors
                {
                    // prepare for application of bilinear formula
                    FormalBilinTri_SetObs(aVObs, TriangleDisplacement_NbObs, aTranslatedFilledPoint, *aCurPostDIm);

                    // Now add observation
                    mSys->CalcAndAddObs(mEqTriDeform, aVecInd, aVObs);

                    // compute indicators
                    const tREAL8 aRadiomValueImPre = aRadiometryScaling * aVObs[5] + aRadiometryTranslation;
                    const tREAL8 aDif = aRadiomValueImPre - aCurPostDIm->GetVBL(aTranslatedFilledPoint); // residual : IntensiteImPreWithRadiometry - TranslatedCoordImPost
                    aSomDif += std::abs(aDif);
                }
                else
                    aNbOut++; // Count number of pixels translated outside post image

                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
            }
        }

        if (mUseMultiScaleApproach && !mIsLastIters && aIterNumber != 0)
        {
            const bool aGenerateIntermediateMaps = false; // if a generating intermediate displacement maps is wanted
            if (aGenerateIntermediateMaps)
                GenerateDisplacementMapsAndOutputImages(aVCur, aIterNumber, aTotalNumberOfIterations, aUserDefinedFolderName);
        }

        // Update all parameter taking into account previous observation
        mSys->SolveUpdateReset();

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_TriangleDeformation::GenerateDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber,
                                                                             const int aTotalNumberOfIterations, const bool aUserDefinedFolderName)
    {
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

        tDoubleVect aLastVObs(12, 0.0);

        // Prefill output image with ImPre to not have null values
        for (const tPt2di &aOutPix : *mDImOut)
            mDImOut->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTri = mDelTri.KthTri(aLTr);
            const tPt3di aLastIndicesOfTriKnots = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTri);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecInd = {4 * aLastIndicesOfTriKnots.x(), 4 * aLastIndicesOfTriKnots.x() + 1,
                                          4 * aLastIndicesOfTriKnots.x() + 2, 4 * aLastIndicesOfTriKnots.x() + 3,
                                          4 * aLastIndicesOfTriKnots.y(), 4 * aLastIndicesOfTriKnots.y() + 1,
                                          4 * aLastIndicesOfTriKnots.y() + 2, 4 * aLastIndicesOfTriKnots.y() + 3,
                                          4 * aLastIndicesOfTriKnots.z(), 4 * aLastIndicesOfTriKnots.z() + 1,
                                          4 * aLastIndicesOfTriKnots.z() + 2, 4 * aLastIndicesOfTriKnots.z() + 3};

            const cNodeOfTriangles aLastFirstPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
            const cNodeOfTriangles aLastSecondPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
            const cNodeOfTriangles aLastThirdPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);

            const tPt2dr aLastTrPointA = aLastFirstPointOfTri.GetCurrentXYDisplacementValues();
            const tPt2dr aLastTrPointB = aLastSecondPointOfTri.GetCurrentXYDisplacementValues();
            const tPt2dr aLastTrPointC = aLastThirdPointOfTri.GetCurrentXYDisplacementValues();

            const tREAL8 aLastRadTrPointA = aLastFirstPointOfTri.GetCurrentRadiometryTranslation();
            const tREAL8 aLastRadScPointA = aLastFirstPointOfTri.GetCurrentRadiometryScaling();
            const tREAL8 aLastRadTrPointB = aLastSecondPointOfTri.GetCurrentRadiometryTranslation();
            const tREAL8 aLastRadScPointB = aLastSecondPointOfTri.GetCurrentRadiometryScaling();
            const tREAL8 aLastRadTrPointC = aLastThirdPointOfTri.GetCurrentRadiometryTranslation();
            const tREAL8 aLastRadScPointC = aLastThirdPointOfTri.GetCurrentRadiometryScaling();

            const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

            for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
            {
                const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                     aLastFilledPixel, aLastPreDIm);
                // prepare for barycenter translation formula by filling aVObs with different coordinates
                FormalInterpBarycenter_SetObs(aLastVObs, 0, aLastPixInsideTriangle);

                // image of a point in triangle by current translation
                const tPt2dr aLastTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
                                                                                                         aLastTrPointC, aLastVObs);

                const tREAL8 aLastRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aLastRadTrPointA,
                                                                                                                    aLastRadTrPointB,
                                                                                                                    aLastRadTrPointC,
                                                                                                                    aLastVObs);

                const tREAL8 aLastRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aLastRadScPointA,
                                                                                                            aLastRadScPointB,
                                                                                                            aLastRadScPointC,
                                                                                                            aLastVObs);

                FillDisplacementMapsAndOutputImage(aLastPixInsideTriangle, aLastTranslatedFilledPoint,
                                                   aLastRadiometryTranslation, aLastRadiometryScaling, 
                                                   mSzImOut, mDImDepX, mDImDepY, mDImOut);
            }
        }

        // save displacement maps in x and y to image files
        if (mUseMultiScaleApproach)
        {
            if (aUserDefinedFolderName)
            {
                mDImDepX->ToFile(mFolderSaveResult + "/DisplacedPixelsX_iter_" + ToStr(aIterNumber) + "_" +
                                 ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile(mFolderSaveResult + "/DisplacedPixelsY_iter_" + ToStr(aIterNumber) + "_" +
                                 ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
            }
            else
            {
                mDImDepX->ToFile("DisplacedPixelsX_iter_" + ToStr(aIterNumber) + "_" +
                                 ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile("DisplacedPixelsY_iter_" + ToStr(aIterNumber) + "_" +
                                 ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
            }
            if (aIterNumber == aTotalNumberOfIterations - 1)
            {
                if (aUserDefinedFolderName)
                    mDImOut->ToFile(mFolderSaveResult + "/DisplacedPixels_iter_" + ToStr(aIterNumber) + "_" +
                                    ToStr(mNumberPointsToGenerate) + "_" +
                                    ToStr(aTotalNumberOfIterations) + ".tif");
                else
                    mDImOut->ToFile("DisplacedPixels_iter_" + ToStr(aIterNumber) + "_" +
                                    ToStr(mNumberPointsToGenerate) + "_" +
                                    ToStr(aTotalNumberOfIterations) + ".tif");
            }
        }
        else if (mInitialiseTranslationWithPreviousExecution)
        {
            mDImDepX->ToFile(mNameIntermediateDepX + ".tif");
            mDImDepY->ToFile(mNameIntermediateDepY + ".tif");
        }
        else
        {
            if (aUserDefinedFolderName)
            {
                mDImDepX->ToFile(mFolderSaveResult + "/DisplacedPixelsX_" + ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile(mFolderSaveResult + "/DisplacedPixelsY_" + ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
                mDImOut->ToFile(mFolderSaveResult + "/DisplacedPixels_" + ToStr(mNumberPointsToGenerate) + "_" +
                                ToStr(aTotalNumberOfIterations) + ".tif");
            }
            else
            {
                mDImDepX->ToFile("DisplacedPixelsX_" + ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile("DisplacedPixelsY_" + ToStr(mNumberPointsToGenerate) + "_" +
                                 ToStr(aTotalNumberOfIterations) + ".tif");
                mDImOut->ToFile("DisplacedPixels_" + ToStr(mNumberPointsToGenerate) + "_" +
                                ToStr(aTotalNumberOfIterations) + ".tif");
            }
        }
    }

    void cAppli_TriangleDeformation::GenerateDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                                          const bool aDisplayLastRadiometryValues, const bool aDisplayLastTranslationValues,
                                                                                          const bool aUserDefinedFolderName)
    {
        tDenseVect aVFinalSol = mSys->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMapsAndOutputImages(aVFinalSol, aIterNumber, aTotalNumberOfIterations, aUserDefinedFolderName);

        if (aDisplayLastRadiometryValues || aDisplayLastTranslationValues)
            DisplayLastUnknownValues(aVFinalSol, aDisplayLastRadiometryValues, aDisplayLastTranslationValues);
    }

    void cAppli_TriangleDeformation::DoOneIteration(const int aIterNumber, const int aTotalNumberOfIterations,
                                                    const bool aUserDefinedFolderName)
    {
        LoopOverTrianglesAndUpdateParameters(aIterNumber, aTotalNumberOfIterations,
                                             aUserDefinedFolderName); // Iterate over triangles and solve system

        // Show final translation results and produce displacement maps
        if (aIterNumber == (aTotalNumberOfIterations - 1))
            GenerateDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber, aTotalNumberOfIterations,
                                                                 mDisplayLastRadiometryValues, mDisplayLastTranslationValues,
                                                                 aUserDefinedFolderName);
    }

    //-----------------------------------------

    int cAppli_TriangleDeformation::Exe()
    {
        // read pre and post images and update their sizes
        ReadFileNameLoadData(mNamePreImage, mImPre, mDImPre, mSzImPre);
        ReadFileNameLoadData(mNamePostImage, mImPost, mDImPost, mSzImPost);

        bool aUserDefinedFolderName = false;
        if (!mFolderSaveResult.empty())
        {
            aUserDefinedFolderName = true;
            if (!ExistFile(mFolderSaveResult))
                CreateDirectories(mFolderSaveResult, aUserDefinedFolderName);
        }

        if (mUseMultiScaleApproach)
            mSigmaGaussFilter = mNumberOfScales * mSigmaGaussFilterStep;

        if (mComputeAvgMax)
            SubtractPrePostImageAndComputeAvgAndMax(mImDiff, mDImDiff, mDImPre,
                                                    mDImPost, mSzImPre);

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        // Generate triangulated knots coordinates
        DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
                                                        mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);

        // If initialisation with previous excution is not wanted initialise the problem with zeros everywhere apart from radiometry scaling, with one
        if ((!mInitialiseTranslationWithPreviousExecution && !mInitialiseRadiometryWithPreviousExecution) || mInitialiseWithUserValues)
            InitialisationAfterExe(mDelTri, mSys, mInitialiseWithUserValues, mInitialiseXTranslationValue,
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

        int aTotalNumberOfIterations = 0;
        (mUseMultiScaleApproach) ? aTotalNumberOfIterations = mNumberOfScales + mNumberOfEndIterations : aTotalNumberOfIterations = mNumberOfScales;

        for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
            DoOneIteration(aIterNumber, aTotalNumberOfIterations, aUserDefinedFolderName);

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
        "Compute 2D deformation between images using triangles",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII
