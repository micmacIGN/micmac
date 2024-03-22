#include "cMMVII_Appli.h"

#include "MMVII_TplSymbTriangle.h"
#include "TriangleDeformation.h"

#include "MMVII_util.h"
#include "TriangleDeformationUtils.h"

/**
   \file TriangleDeformation.cpp

   \brief file for computing 2D deformations between 2 images
   thanks to triangles.
**/

namespace MMVII
{
    /******************************************/
    /*                                        */
    /*          cTriangleDeformation          */
    /*                                        */
    /******************************************/

    cAppli_cTriangleDeformation::cAppli_cTriangleDeformation(const std::vector<std::string> &aVArgs,
                                                             const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                                                                              mRandomUniformLawUpperBoundLines(1),
                                                                                              mRandomUniformLawUpperBoundCols(1),
                                                                                              mShow(true),
                                                                                              mComputeAvgMax(false),
                                                                                              mUseMultiScaleApproach(true),
                                                                                              mInitialiseTranslationWithPreviousExecution(true),
                                                                                              mInitialiseRadiometryWithPreviousExecution(true),
                                                                                              mInitialiseWithMMVI(false),
                                                                                              mNameFileInitialDepX("InitialXDisplacementMap.tif"),
                                                                                              mNameFileInitialDepY("InitialYDisplacementMap.tif"),
                                                                                              mNameIntermediateDepX("IntermediateXDisplacementMap.tif"),
                                                                                              mNameIntermediateDepY("IntermediateYDisplacementMap.tif"),
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
                                                                                              mVectorPts({tPt2dr(0, 0)}),
                                                                                              mDelTri(mVectorPts),
                                                                                              mSys(nullptr),
                                                                                              mEqTriDeform(nullptr)
    {
        mEqTriDeform = EqDeformTri(true, 1); // true means with derivative, 1 is size of buffer
    }

    cAppli_cTriangleDeformation::~cAppli_cTriangleDeformation()
    {
        delete mSys;
        delete mEqTriDeform;
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformation::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach or iterations if multi-scale approach is not applied in optimisation process.");
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformation::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mRandomUniformLawUpperBoundCols, "RandomUniformLawUpperBoundXAxis",
                           "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV})
               << AOpt2007(mRandomUniformLawUpperBoundLines, "RandomUniformLawUpperBoundYAxis",
                           "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV})
               << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV})
               << AOpt2007(mComputeAvgMax, "ComputeAvgMaxDiffIm",
                           "Whether to compute the average and maximum pixel value of the difference image between post and pre image or not.", {eTA2007::HDV})
               << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
               << AOpt2007(mInitialiseTranslationWithPreviousExecution, "InitialiseTranslationWithPreviousExecution",
                           "Whether to initialise or not with unknown translation values obtained at previous execution", {eTA2007::HDV})
               << AOpt2007(mInitialiseRadiometryWithPreviousExecution, "InitialiseRadiometryWithPreviousExecution",
                           "Whether to initialise or not with unknown radiometry values obtained at previous execution", {eTA2007::HDV})
               << AOpt2007(mInitialiseWithMMVI, "InitialiseWithMMVI",
                           "Whether to initialise or not values of unknowns with pre-computed values from MicMacV1 at first execution", {eTA2007::HDV})
               << AOpt2007(mNameFileInitialDepX, "InitialDepXMapFilename", "Name of file of initial X-displacement map", {eTA2007::HDV})
               << AOpt2007(mNameFileInitialDepY, "InitialDepYMapFilename", "Name of file of initial Y-displacement map", {eTA2007::HDV})
               << AOpt2007(mNameIntermediateDepX, "NameForIntermediateXDisplacementMap",
                           "File name to use when saving intermediate x-displacement maps between executions", {eTA2007::HDV})
               << AOpt2007(mNameIntermediateDepY, "NameForIntermediateYDisplacementMap",
                           "File name to use when saving intermediate y-displacement maps between executions", {eTA2007::HDV})
               << AOpt2007(mNameCorrelationMaskMMVI, "NameOfCorrelationMask",
                           "File name of mask file from MMVI giving locations where correlation is computed", {eTA2007::HDV})
               << AOpt2007(mIsFirstExecution, "IsFirstExecution",
                           "Whether this is the first execution of optimisation algorithm or not", {eTA2007::HDV})
               << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep", "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV})
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
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV})
               << AOpt2007(mFolderSaveResult, "FolderToSaveResults",
                           "Folder name where to store produced results", {eTA2007::HDV})
               << AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationsValues",
                           "Whether to display the final values of unknowns linked to point translation.", {eTA2007::HDV})
               << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                           "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV});
    }

    void cAppli_cTriangleDeformation::InitialiseWithPreviousExecutionValues(const cTriangulation2D<tREAL8> &aDelTri,
                                                                            cResolSysNonLinear<tREAL8> *&aSys)
    {
        tDenseVect aVInit(4 * aDelTri.NbPts(), eModeInitImage::eMIA_Null);

        if (!mIsFirstExecution)
        {
            ReadFileNameLoadData(mNameIntermediateDepX, mImIntermediateDepX,
                                 mDImIntermediateDepX, mSzImIntermediateDepX);
            ReadFileNameLoadData(mNameIntermediateDepY, mImIntermediateDepY,
                                 mDImIntermediateDepY, mSzImIntermediateDepY);
        }

        ReadFileNameLoadData(mNameCorrelationMaskMMVI, mImCorrelationMask,
                             mDImCorrelationMask, mSzCorrelationMask);

        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aInitTri = mDelTri.KthTri(aTr);
            const tPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

            //----------- index of unknown, finds the associated pixels of current triangle
            const tIntVect aInitVecInd = {4 * aIndicesOfTriKnots.x(), 4 * aIndicesOfTriKnots.x() + 1,
                                      4 * aIndicesOfTriKnots.x() + 2, 4 * aIndicesOfTriKnots.x() + 3,
                                      4 * aIndicesOfTriKnots.y(), 4 * aIndicesOfTriKnots.y() + 1,
                                      4 * aIndicesOfTriKnots.y() + 2, 4 * aIndicesOfTriKnots.y() + 3,
                                      4 * aIndicesOfTriKnots.z(), 4 * aIndicesOfTriKnots.z() + 1,
                                      4 * aIndicesOfTriKnots.z() + 2, 4 * aIndicesOfTriKnots.z() + 3};

            // Get points coordinates associated to triangle
            const cKtOfTriangles aFirstInitPointOfTri = cKtOfTriangles(aVInit, aInitVecInd, 0, 1, 2, 3, aInitTri, 0);
            const cKtOfTriangles aSecondInitPointOfTri = cKtOfTriangles(aVInit, aInitVecInd, 4, 5, 6, 7, aInitTri, 1);
            const cKtOfTriangles aThirdInitPointOfTri = cKtOfTriangles(aVInit, aInitVecInd, 8, 9, 10, 11, aInitTri, 2);

            // Check if correlation is computed for these points
            const bool aFirstPointIsValid = CheckValidCorrelationValue(mDImCorrelationMask, aFirstInitPointOfTri);
            const bool aSecondPointIsValid = CheckValidCorrelationValue(mDImCorrelationMask, aSecondInitPointOfTri);
            const bool aThirdPointIsValid = CheckValidCorrelationValue(mDImCorrelationMask, aThirdInitPointOfTri);

            if (mInitialiseTranslationWithPreviousExecution)
            {
                aVInit(aInitVecInd.at(0)) = ReturnCorrectInitialisationValue(aFirstPointIsValid, mDImIntermediateDepX,
                                                                                        aFirstInitPointOfTri, 0);
                aVInit(aInitVecInd.at(1)) = ReturnCorrectInitialisationValue(aFirstPointIsValid, mDImIntermediateDepY,
                                                                                        aFirstInitPointOfTri, 0);
                aVInit(aInitVecInd.at(2)) = ReturnCorrectInitialisationValue(aSecondPointIsValid, mDImIntermediateDepX,
                                                                                        aSecondInitPointOfTri, 0);
                aVInit(aInitVecInd.at(3)) = ReturnCorrectInitialisationValue(aSecondPointIsValid, mDImIntermediateDepY,
                                                                                        aSecondInitPointOfTri, 0);
                aVInit(aInitVecInd.at(4)) = ReturnCorrectInitialisationValue(aThirdPointIsValid, mDImIntermediateDepX,
                                                                                        aThirdInitPointOfTri, 0);
                aVInit(aInitVecInd.at(5)) = ReturnCorrectInitialisationValue(aThirdPointIsValid, mDImIntermediateDepY,
                                                                                        aThirdInitPointOfTri, 0);
            } /*
             else if (mInitialisationRadiometryWithPreviousExecution)
             {
                 aVInit(aVecInd.at(2)) = mDImIntermediateRadTr->GetV(aFirstPointTri + tPt2di(mDImIntermediateDepX->GetV(aFirstPointTri),
                                                                                             mDImIntermediateDepY->GetV(aFirstPointTri));
                 aVInit(aVecInd.at(3)) = mDImIntermediateRadSc->GetV(aFirstPointTri);
                 aVInit(aVecInd.at(6)) = mDImIntermediateRadTr->GetV(aSecondPointTri);
                 aVInit(aVecInd.at(7)) = mDImIntermediateRadSc->GetV(aSecondPointTri);
                 aVInit(aVecInd.at(10)) = mDImIntermediateRadTr->GetV(aThirdPointTri);
                 aVInit(aVecInd.at(11)) = mDImIntermediateRadSc->GetV(aThirdPointTri);
             }*/
        }

        aSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInit);
    }

    void cAppli_cTriangleDeformation::LoopOverTrianglesAndUpdateParameters(const int aIterNumber, const int aTotalNumberOfIterations,
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
                    aCurPreDIm->ToFile(mFolderSaveResult + "/GaussFilteredImPre_iter_" + std::to_string(aIterNumber) + ".tif");
                else
                    aCurPreDIm->ToFile("GaussFilteredImPre_iter_" + std::to_string(aIterNumber) + ".tif");
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

            const cKtOfTriangles aFirstPointOfTri = cKtOfTriangles(aVCur, aVecInd, 0, 1, 2, 3, aTri, 0);
            const cKtOfTriangles aSecondPointOfTri = cKtOfTriangles(aVCur, aVecInd, 4, 5, 6, 7, aTri, 1);
            const cKtOfTriangles aThirdPointOfTri = cKtOfTriangles(aVCur, aVecInd, 8, 9, 10, 11, aTri, 2);
/*
            const tPt2dr aCurTrPointA = tPt2dr(aVCur(aVecInd.at(0)),
                                               aVCur(aVecInd.at(1))); 
            const tPt2dr aCurTrPointB = tPt2dr(aVCur(aVecInd.at(4)),
                                               aVCur(aVecInd.at(5))); 
            const tPt2dr aCurTrPointC = tPt2dr(aVCur(aVecInd.at(8)),
                                               aVCur(aVecInd.at(9)));
*/
            const tPt2dr aCurTrPointA = aFirstPointOfTri.GetCurrentXYDisplacementVector(); // current translation 1st point of triangle
            const tPt2dr aCurTrPointB = aSecondPointOfTri.GetCurrentXYDisplacementVector(); // current translation 2nd point of triangle
            const tPt2dr aCurTrPointC = aThirdPointOfTri.GetCurrentXYDisplacementVector(); // current translation 3rd point of triangle
            /*
            const tREAL8 aCurRadTrPointA = aVCur(aVecInd.at(2));  // current translation on radiometry 1st point of triangle
            const tREAL8 aCurRadScPointA = aVCur(aVecInd.at(3));  // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointB = aVCur(aVecInd.at(6));  // current translation on radiometry 2nd point of triangle
            const tREAL8 aCurRadScPointB = aVCur(aVecInd.at(7));  // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointC = aVCur(aVecInd.at(10)); // current translation on radiometry 3rd point of triangle
            const tREAL8 aCurRadScPointC = aVCur(aVecInd.at(11)); // current scale on radiometry 3rd point of triangle
            */
            const tREAL8 aCurRadTrPointA = aFirstPointOfTri.GetCurrentRadiometryTranslation();  // current translation on radiometry 1st point of triangle
            const tREAL8 aCurRadScPointA = aFirstPointOfTri.GetCurrentRadiometryScaling();  // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointB = aSecondPointOfTri.GetCurrentRadiometryTranslation();  // current translation on radiometry 2nd point of triangle
            const tREAL8 aCurRadScPointB = aSecondPointOfTri.GetCurrentRadiometryScaling();  // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointC = aThirdPointOfTri.GetCurrentRadiometryTranslation(); // current translation on radiometry 3rd point of triangle
            const tREAL8 aCurRadScPointC = aThirdPointOfTri.GetCurrentRadiometryScaling(); // current scale on radiometry 3rd point of triangle

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
                                                                                 aFilledPixel, *aCurPreDIm);
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
                    const tREAL8 aDif = aRadiomValueImPre - aCurPostDIm->GetVBL(aTranslatedFilledPoint); // residual
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

    void cAppli_cTriangleDeformation::FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                                                         const tPt2dr &aLastTranslatedFilledPoint,
                                                                         const tREAL8 aLastRadiometryTranslation,
                                                                         const tREAL8 aLastRadiometryScaling)
    {
        const tREAL8 aLastXCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().x();
        const tREAL8 aLastYCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().y();
        const tREAL8 aLastPixelValue = aLastPixInsideTriangle.GetPixelValue();

        const tPt2di aLastCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);
        mDImDepX->SetV(aLastCoordinate,
                       aLastTranslatedFilledPoint.x() - aLastXCoordinate);
        mDImDepY->SetV(aLastCoordinate,
                       aLastTranslatedFilledPoint.y() - aLastYCoordinate);
        const tREAL8 aLastXTranslatedCoord = aLastXCoordinate + mDImDepX->GetV(aLastCoordinate);
        const tREAL8 aLastYTranslatedCoord = aLastYCoordinate + mDImDepY->GetV(aLastCoordinate);

        const tREAL8 aLastRadiometryValue = aLastRadiometryScaling * aLastPixelValue +
                                            aLastRadiometryTranslation;

        // Build image with intensities displaced
        // deal with different cases of pixel being translated out of image
        if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(0, 0)));
        else if (aLastXTranslatedCoord >= mSzImOut.x() && aLastYTranslatedCoord >= mSzImOut.y())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(mSzImOut.x() - 1, mSzImOut.y() - 1)));
        else if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord >= mSzImOut.y())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(0, mSzImOut.y() - 1)));
        else if (aLastXTranslatedCoord >= mSzImOut.x() && aLastYTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(mSzImOut.x() - 1, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < mSzImOut.x() &&
                 aLastYTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(aLastXTranslatedCoord, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < mSzImOut.x() &&
                 aLastYTranslatedCoord > mSzImOut.y())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(aLastXTranslatedCoord, mSzImOut.y() - 1)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < mSzImOut.y() &&
                 aLastXTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(0, aLastYTranslatedCoord)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < mSzImOut.y() &&
                 aLastXTranslatedCoord > mSzImOut.x())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(tPt2di(mSzImOut.x() - 1, aLastYTranslatedCoord)));
        else
            // at the translated pixel the untranslated pixel value is given computed with the right radiometry values
            mDImOut->SetV(tPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord), aLastRadiometryValue);
    }

    void cAppli_cTriangleDeformation::GenerateDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber,
                                                                              const int aTotalNumberOfIterations, const bool aUserDefinedFolderName)
    {
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = mDImOut->Sz();

        mImDepX = tIm(mSzImPre, 0, eModeInitImage::eMIA_Null);
        mDImDepX = &mImDepX.DIm();

        mImDepY = tIm(mSzImPre, 0, eModeInitImage::eMIA_Null);
        mDImDepY = &mImDepY.DIm();

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
            /*
            const tPt2dr aLastTrPointA = tPt2dr(aVFinalSol(aLastVecInd.at(0)),
                                                aVFinalSol(aLastVecInd.at(1))); // last translation 1st point of triangle
            const tPt2dr aLastTrPointB = tPt2dr(aVFinalSol(aLastVecInd.at(4)),
                                                aVFinalSol(aLastVecInd.at(5))); // last translation 2nd point of triangle
            const tPt2dr aLastTrPointC = tPt2dr(aVFinalSol(aLastVecInd.at(8)),
                                                aVFinalSol(aLastVecInd.at(9))); // last translation 3rd point of triangle

            const tREAL8 aLastRadTrPointA = aVFinalSol(aLastVecInd.at(2));  // last translation on radiometry 1st point of triangle
            const tREAL8 aLastRadScPointA = aVFinalSol(aLastVecInd.at(3));  // last scale on radiometry 3rd point of triangle
            const tREAL8 aLastRadTrPointB = aVFinalSol(aLastVecInd.at(6));  // last translation on radiometry 2nd point of triangle
            const tREAL8 aLastRadScPointB = aVFinalSol(aLastVecInd.at(7));  // last scale on radiometry 3rd point of triangle
            const tREAL8 aLastRadTrPointC = aVFinalSol(aLastVecInd.at(10)); // last translation on radiometry 3rd point of triangle
            const tREAL8 aLastRadScPointC = aVFinalSol(aLastVecInd.at(11)); // last scale on radiometry 3rd point of triangle
            */
            const cKtOfTriangles aLastFirstPointOfTri = cKtOfTriangles(aVFinalSol, aLastVecInd, 0, 1, 2, 3, aLastTri, 0);
            const cKtOfTriangles aLastSecondPointOfTri = cKtOfTriangles(aVFinalSol, aLastVecInd, 4, 5, 6, 7, aLastTri, 1);
            const cKtOfTriangles aLastThirdPointOfTri = cKtOfTriangles(aVFinalSol, aLastVecInd, 8, 9, 10, 11, aLastTri, 2);

            const tPt2dr aLastTrPointA = aLastFirstPointOfTri.GetCurrentXYDisplacementVector();
            const tPt2dr aLastTrPointB = aLastSecondPointOfTri.GetCurrentXYDisplacementVector();
            const tPt2dr aLastTrPointC = aLastThirdPointOfTri.GetCurrentXYDisplacementVector();

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
                                                                                     aLastFilledPixel, *aLastPreDIm);
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
                                                   aLastRadiometryTranslation, aLastRadiometryScaling);
            }
        }

        // save displacement maps in x and y to image files
        if (mUseMultiScaleApproach)
        {
            if (aUserDefinedFolderName)
            {
                mDImDepX->ToFile(mFolderSaveResult + "/DisplacedPixelsX_iter_" + std::to_string(aIterNumber) + "_" +
                                 std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile(mFolderSaveResult + "/DisplacedPixelsY_iter_" + std::to_string(aIterNumber) + "_" +
                                 std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
            }
            else
            {
                mDImDepX->ToFile("DisplacedPixelsX_iter_" + std::to_string(aIterNumber) + "_" +
                                 std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile("DisplacedPixelsY_iter_" + std::to_string(aIterNumber) + "_" +
                                 std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
            }
            if (aIterNumber == aTotalNumberOfIterations - 1)
            {
                if (aUserDefinedFolderName)
                    mDImOut->ToFile(mFolderSaveResult + "/DisplacedPixels_iter_" + std::to_string(aIterNumber) + "_" +
                                    std::to_string(mNumberPointsToGenerate) + "_" +
                                    std::to_string(aTotalNumberOfIterations) + ".tif");
                else
                    mDImOut->ToFile("DisplacedPixels_iter_" + std::to_string(aIterNumber) + "_" +
                                    std::to_string(mNumberPointsToGenerate) + "_" +
                                    std::to_string(aTotalNumberOfIterations) + ".tif");
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
                mDImDepX->ToFile(mFolderSaveResult + "/DisplacedPixelsX_" + std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile(mFolderSaveResult + "/DisplacedPixelsY_" + std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
                mDImOut->ToFile(mFolderSaveResult + "/DisplacedPixels_" + std::to_string(mNumberPointsToGenerate) + "_" +
                                std::to_string(aTotalNumberOfIterations) + ".tif");
            }
            else
            {
                mDImDepX->ToFile("DisplacedPixelsX_" + std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
                mDImDepY->ToFile("DisplacedPixelsY_" + std::to_string(mNumberPointsToGenerate) + "_" +
                                 std::to_string(aTotalNumberOfIterations) + ".tif");
                mDImOut->ToFile("DisplacedPixels_" + std::to_string(mNumberPointsToGenerate) + "_" +
                                std::to_string(aTotalNumberOfIterations) + ".tif");
            }
        }
    }

    void cAppli_cTriangleDeformation::GenerateDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                                           const bool aDisplayLastRadiometryValues, const bool aDisplayLastTranslationValues,
                                                                                           const bool aUserDefinedFolderName)
    {
        tDenseVect aVFinalSol = mSys->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMapsAndOutputImages(aVFinalSol, aIterNumber, aTotalNumberOfIterations, aUserDefinedFolderName);

        if (aDisplayLastRadiometryValues || aDisplayLastTranslationValues)
            DisplayLastUnknownValues(aVFinalSol, aDisplayLastRadiometryValues, aDisplayLastTranslationValues);
    }

    void cAppli_cTriangleDeformation::DoOneIteration(const int aIterNumber, const int aTotalNumberOfIterations,
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

    int cAppli_cTriangleDeformation::Exe()
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
        GeneratePointsForDelaunay(mVectorPts, mNumberPointsToGenerate, mRandomUniformLawUpperBoundLines,
                                  mRandomUniformLawUpperBoundCols, mDelTri, mSzImPre);

        // If initialisation with previous excution is not wanted initialise the problem with zeros everywhere apart from radiometry scaling, with one
        if (!mInitialiseTranslationWithPreviousExecution && !mInitialiseRadiometryWithPreviousExecution)
            InitialisationAfterExe(mDelTri, mSys);
        else
        {
            if (mIsFirstExecution && mInitialiseWithMMVI)
            {
                ReadFileNameLoadData(mNameFileInitialDepX, mImIntermediateDepX,
                                     mDImIntermediateDepX, mSzImIntermediateDepX);
                ReadFileNameLoadData(mNameFileInitialDepY, mImIntermediateDepY,
                                     mDImIntermediateDepY, mSzImIntermediateDepY);

                InitialiseWithPreviousExecutionValues(mDelTri, mSys);
            }
            else if (!mIsFirstExecution && mInitialiseWithMMVI)
                InitialiseWithPreviousExecutionValues(mDelTri, mSys);
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

    tMMVII_UnikPApli Alloc_cTriangleDeformation(const std::vector<std::string> &aVArgs,
                                                const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_cTriangleDeformation(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_ComputeTriangleDeformation(
        "ComputeTriangleDeformation",
        Alloc_cTriangleDeformation,
        "Compute 2D deformation of triangles between images using triangles",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII