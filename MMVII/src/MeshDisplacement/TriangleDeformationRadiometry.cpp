#include "TriangleDeformationRadiometry.h"

/**
   \file TriangleDeformationRadiometry.cpp

   \brief file computing radiometric deformations
   between 2 images using triangles.
**/

namespace MMVII
{
    /****************************************************/
    /*                                                  */
    /*       cAppli_TriangleDeformationRadiometry       */
    /*                                                  */
    /****************************************************/

    cAppli_TriangleDeformationRadiometry::cAppli_TriangleDeformationRadiometry(const std::vector<std::string> &aVArgs,
                                                                               const cSpecMMVII_Appli &aSpec) : cAppli_TriangleDeformation(aVArgs, aSpec),
                                                                                                                mNumberOfLines(1),
                                                                                                                mNumberOfCols(1),
                                                                                                                mShow(true),
                                                                                                                mGenerateOutputImage(true),
                                                                                                                mInitialiseWithUserValues(true),
                                                                                                                mInitialiseRadTrValue(0),
                                                                                                                mInitialiseRadScValue(1),
                                                                                                                mUseMultiScaleApproach(false),
                                                                                                                mWeightRadTranslation(-1),
                                                                                                                mWeightRadScale(-1),
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
                                                                                                                mSysRadiometry(nullptr),
                                                                                                                mEqRadiometryTri(nullptr)

    {
        mEqRadiometryTri = EqDeformTriRadiometry(true, 1); // true means with derivative, 1 is size of buffer
    }

    cAppli_TriangleDeformationRadiometry::~cAppli_TriangleDeformationRadiometry()
    {
        delete mSysRadiometry;
        delete mEqRadiometryTri;
    }

    cCollecSpecArg2007 &cAppli_TriangleDeformationRadiometry::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach optimisation process.");
    }

    cCollecSpecArg2007 &cAppli_TriangleDeformationRadiometry::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mNumberOfCols, "RandomUniformLawUpperBoundXAxis",
                           "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfLines, "RandomUniformLawUpperBoundYAxis",
                           "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mGenerateOutputImage, "GenerateDisplacementImage",
                           "Whether to generate and save the output image with computed radiometry", {eTA2007::HDV})
               << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
                           "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
               << AOpt2007(mInitialiseWithUserValues, "InitialiseWithUserValues",
                           "Whether the user wishes or not to initialise unknowns with personalised values.", {eTA2007::HDV})
               << AOpt2007(mInitialiseRadTrValue, "InitialeRadiometryTranslationValue",
                           "Value to use for initialising radiometry translation unknown values", {eTA2007::HDV})
               << AOpt2007(mInitialiseRadScValue, "InitialeRadiometryScalingValue",
                           "Value to use for initialising radiometry scaling unknown values", {eTA2007::HDV})
               << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
               << AOpt2007(mWeightRadTranslation, "WeightRadiometryTranslation",
                           "A value to weight radiometry translation for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mWeightRadScale, "WeightRadiometryScaling",
                           "A value to weight radiometry scaling for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                           "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV})
               << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep",
                           "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning});
    }

    void cAppli_TriangleDeformationRadiometry::LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber)
    {
        //----------- allocate vec of obs :
        tDoubleVect aVObsRad(12, 0.0); // 6 for ImagePre and 6 for ImagePost

        //----------- extract current parameters
        tDenseVect aVCurSolRad = mSysRadiometry->CurGlobSol(); // Get current solution.

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

        // Loop over all triangles to add the observations on each point
        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTriRad = mDelTri.KthTri(aTr);
            const cPt3di aIndicesOfTriKnotsRad = mDelTri.KthFace(aTr);

            const cTriangle2DCompiled aCompTri(aTriRad);

            std::vector<tPt2di> aVectorToFillWithInsidePixels;
            aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

            //----------- index of unknown, finds the associated pixels of current triangle knots
            const tIntVect aVecInd = {2 * aIndicesOfTriKnotsRad.x(), 2 * aIndicesOfTriKnotsRad.x() + 1,
                                      2 * aIndicesOfTriKnotsRad.y(), 2 * aIndicesOfTriKnotsRad.y() + 1,
                                      2 * aIndicesOfTriKnotsRad.z(), 2 * aIndicesOfTriKnotsRad.z() + 1};

            const cNodeOfTriangles aFirstPointOfTriRad = cNodeOfTriangles(aVCurSolRad, aVecInd, 0, 1, 0, 1, aTriRad, 0);
            const cNodeOfTriangles aSecondPointOfTriRad = cNodeOfTriangles(aVCurSolRad, aVecInd, 2, 3, 2, 3, aTriRad, 1);
            const cNodeOfTriangles aThirdPointOfTriRad = cNodeOfTriangles(aVCurSolRad, aVecInd, 4, 5, 4, 5, aTriRad, 2);

            const tREAL8 aCurRadTrPointA = aFirstPointOfTriRad.GetCurrentRadiometryTranslation();  // current translation on radiometry 1st point of triangle
            const tREAL8 aCurRadScPointA = aFirstPointOfTriRad.GetCurrentRadiometryScaling();      // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointB = aSecondPointOfTriRad.GetCurrentRadiometryTranslation(); // current translation on radiometry 2nd point of triangle
            const tREAL8 aCurRadScPointB = aSecondPointOfTriRad.GetCurrentRadiometryScaling();     // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointC = aThirdPointOfTriRad.GetCurrentRadiometryTranslation();  // current translation on radiometry 3rd point of triangle
            const tREAL8 aCurRadScPointC = aThirdPointOfTriRad.GetCurrentRadiometryScaling();      // current scale on radiometry 3rd point of triangle

            // soft constraint radiometric translation
            if (mWeightRadTranslation >= 0)
            {
                const int aSolStart = 0;
                const int aSolStep = 2; // adapt step to solution vector configuration
                for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size() - 1; aIndCurSol += aSolStep)
                {
                    const int aIndices = aVecInd.at(aIndCurSol);
                    mSysRadiometry->AddEqFixVar(aIndices, aVCurSolRad(aIndices), mWeightRadTranslation);
                }
            }

            // soft constraint radiometric scaling
            if (mWeightRadScale >= 0)
            {
                const int aSolStart = 1;
                const int aSolStep = 2; // adapt step to solution vector configuration
                for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size(); aIndCurSol += aSolStep)
                {
                    const int aIndices = aVecInd.at(aIndCurSol);
                    mSysRadiometry->AddEqFixVar(aIndices, aVCurSolRad(aIndices), mWeightRadScale);
                }
            }

            const size_t aNumberOfInsidePixels = aVectorToFillWithInsidePixels.size();

            // Loop over all pixels inside triangle
            // size_t is necessary as there can be a lot of pixels in triangles
            for (size_t aFilledPixel = 0; aFilledPixel < aNumberOfInsidePixels; aFilledPixel++)
            {
                const cPtInsideTriangles aPixInsideTriangle = cPtInsideTriangles(aCompTri, aVectorToFillWithInsidePixels,
                                                                                 aFilledPixel, aCurPreDIm);
                // prepare for barycenter translation formula by filling aVObsRad with different coordinates
                FormalInterpBarycenter_SetObs(aVObsRad, 0, aPixInsideTriangle);

                // radiometry translation of pixel by current radiometry translation of triangle knots
                const tREAL8 aRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aCurRadTrPointA,
                                                                                                                aCurRadTrPointB,
                                                                                                                aCurRadTrPointC,
                                                                                                                aVObsRad);

                // radiometry scaling of pixel by current radiometry scaling of triangle knots
                const tREAL8 aRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aCurRadScPointA,
                                                                                                        aCurRadScPointB,
                                                                                                        aCurRadScPointC,
                                                                                                        aVObsRad);

                const tPt2dr aInsideTrianglePoint = aPixInsideTriangle.GetCartesianCoordinates();
                const tPt2di aEastTranslatedPoint = tPt2di(aInsideTrianglePoint.x(), aInsideTrianglePoint.y()) + tPt2di(1, 0);
                const tPt2di aSouthTranslatedPoint = tPt2di(aInsideTrianglePoint.x(), aInsideTrianglePoint.y()) + tPt2di(0, 1);

                if (aCurPostDIm->InsideBL(tPt2dr(aEastTranslatedPoint.x(), aEastTranslatedPoint.y()))) // avoid errors
                {
                    if (aCurPostDIm->InsideBL(tPt2dr(aSouthTranslatedPoint.x(), aSouthTranslatedPoint.y()))) // avoid errors
                    {
                        // prepare for application of bilinear formula
                        FormalBilinTri_SetObs(aVObsRad, TriangleDisplacement_NbObs, aInsideTrianglePoint, *aCurPostDIm);

                        // Now add observation
                        mSysRadiometry->CalcAndAddObs(mEqRadiometryTri, aVecInd, aVObsRad);

                        // compute indicators
                        const tREAL8 aRadiomValueImPre = aRadiometryScaling * aVObsRad[5] + aRadiometryTranslation;
                        const tREAL8 aDif = aRadiomValueImPre - aCurPostDIm->GetVBL(aInsideTrianglePoint); // residual
                        aSomDif += std::abs(aDif);
                    }
                }
                else
                    aNbOut++;
                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
            }
        }

        // Update all parameter taking into account previous observation
        mSysRadiometry->SolveUpdateReset();

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_TriangleDeformationRadiometry::GenerateOutputImageAndDisplayLastRadiometryValues(const tDenseVect &aVFinalSol, const int aIterNumber)
    {
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = mDImOut->Sz();

        tIm aLastPreIm = tIm(mSzImPre);
        tDIm *aLastPreDIm = nullptr;
        LoadPrePostImageAndData(aLastPreIm, aLastPreDIm, "pre", mImPre, mImPost);

        if (mUseMultiScaleApproach && !mIsLastIters)
        {
            aLastPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
            aLastPreDIm = &aLastPreIm.DIm();
        }

        for (const tPt2di &aOutPix : *mDImOut) // Initialise output image
            mDImOut->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTriRad = mDelTri.KthTri(aLTr);
            const cPt3di aLastIndicesOfTriKnotsTr = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTriRad);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecInd = {2 * aLastIndicesOfTriKnotsTr.x(), 2 * aLastIndicesOfTriKnotsTr.x() + 1,
                                          2 * aLastIndicesOfTriKnotsTr.y(), 2 * aLastIndicesOfTriKnotsTr.y() + 1,
                                          2 * aLastIndicesOfTriKnotsTr.z(), 2 * aLastIndicesOfTriKnotsTr.z() + 1};

            const cNodeOfTriangles aLastFirstPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 0, 1, 0, 1, aLastTriRad, 0);
            const cNodeOfTriangles aLastSecondPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 2, 3, 2, 3, aLastTriRad, 1);
            const cNodeOfTriangles aLastThirdPointOfTri = cNodeOfTriangles(aVFinalSol, aLastVecInd, 4, 5, 4, 5, aLastTriRad, 2);

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

                // radiometry translation of pixel by current radiometry translation
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

        // save output image with calculated radiometries to image file
        mDImOut->ToFile("OutputImage_" + ToStr(mNumberPointsToGenerate) + "_" + ToStr(mNumberOfScales) + ".tif");
    }

    void cAppli_TriangleDeformationRadiometry::DoOneIterationRadiometry(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                        const tDenseVect &aVInitSol)
    {
        LoopOverTrianglesAndUpdateParametersRadiometry(aIterNumber); // Iterate over triangles and solve system

        tDenseVect aVFinalSol = mSysRadiometry->CurGlobSol();

        // Show final translation results and produce displacement maps
        if (aIterNumber == (aTotalNumberOfIterations - 1))
        {
            GenerateOutputImageAndDisplayLastRadiometryValues(aVFinalSol, aIterNumber);
            // Display last computed values of radiometry unknowns
            if (mDisplayLastRadiometryValues)
                DisplayLastUnknownValuesAndComputeStatistics(aVFinalSol, aVInitSol);
        }
    }

    //-----------------------------------------

    int cAppli_TriangleDeformationRadiometry::Exe()
    {
        // read pre and post images and their sizes
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

        InitialisationAfterExeRadiometry(mDelTri, mSysRadiometry, mInitialiseWithUserValues,
                                         mInitialiseRadTrValue, mInitialiseRadScValue);

        const tDenseVect aVInitSolRad = mSysRadiometry->CurGlobSol().Dup(); // Duplicate initial solution

        int aTotalNumberOfIterations = 0;
        (mUseMultiScaleApproach) ? aTotalNumberOfIterations = mNumberOfScales + mNumberOfEndIterations : aTotalNumberOfIterations = mNumberOfScales;

        for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
            DoOneIterationRadiometry(aIterNumber, aTotalNumberOfIterations, aVInitSolRad);

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
        "Compute radiometric deformation between images using triangles",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII
