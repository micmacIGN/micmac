#include "MMVII_TplSymbTriangle.h"

#include "TriangleDeformationRad.h"

/**
   \file TriangleDeformationRad.cpp

   \brief file computing radiometric deformations
   between 2 images using triangles.
**/

namespace MMVII
{

    /**********************************************/
    /*                                            */
    /*          cTriangleDeformationRad           */
    /*                                            */
    /**********************************************/

    cAppli_cTriangleDeformationRad::cAppli_cTriangleDeformationRad(const std::vector<std::string> &aVArgs,
                                                                   const cSpecMMVII_Appli &aSpec) : cAppli_cTriangleDeformationRadiometry(aVArgs, aSpec),
                                                                                                    mRandomUniformLawUpperBoundLines(1),
                                                                                                    mRandomUniformLawUpperBoundCols(1),
                                                                                                    mShow(true),
                                                                                                    mGenerateOutputImage(true),
                                                                                                    mUseMultiScaleApproach(false),
                                                                                                    mWeightRadTranslation(-1),
                                                                                                    mWeightRadScale(-1),
                                                                                                    mDisplayLastRadiometryValues(false),
                                                                                                    mSigmaGaussFilterStep(1),
                                                                                                    mNumberOfIterGaussFilter(3),
                                                                                                    mNumberOfEndIterations(2),
                                                                                                    mSzImPre(cPt2di(1, 1)),
                                                                                                    mImPre(mSzImPre),
                                                                                                    mDImPre(nullptr),
                                                                                                    mSzImPost(cPt2di(1, 1)),
                                                                                                    mImPost(mSzImPost),
                                                                                                    mDImPost(nullptr),
                                                                                                    mSzImOut(cPt2di(1, 1)),
                                                                                                    mImOut(mSzImOut),
                                                                                                    mDImOut(nullptr),
                                                                                                    mVectorPts({cPt2dr(0, 0)}),
                                                                                                    mDelTri(mVectorPts),
                                                                                                    mSys(nullptr),
                                                                                                    mEqRadTri(nullptr)

    {
        mEqRadTri = EqDeformTriRad(true, 1); // true means with derivative, 1 is size of buffer
        // mEqRadiometryTri->SetDebugEnabled(true);
    }

    cAppli_cTriangleDeformationRad::~cAppli_cTriangleDeformationRad()
    {
        delete mSys;
        delete mEqRadTri;
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformationRad::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach optimisation process.");
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformationRad::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mRandomUniformLawUpperBoundCols, "RandomUniformLawUpperBoundXAxis",
                           "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV})
               << AOpt2007(mRandomUniformLawUpperBoundLines, "RandomUniformLawUpperBoundYAxis",
                           "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV})
               << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV})
               << AOpt2007(mGenerateOutputImage, "GenerateDisplacementImage",
                           "Whether to generate and save the output image with computed radiometry", {eTA2007::HDV})
               << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
               << AOpt2007(mWeightRadTranslation, "WeightRadiometryTranslation",
                           "A value to weight radiometry translation for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mWeightRadScale, "WeightRadiometryScaling",
                           "A value to weight radiometry scaling for soft freezing of coefficient.", {eTA2007::HDV})
               << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                           "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV})
               << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep",
                           "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV})
               << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV});
    }

    void cAppli_cTriangleDeformationRad::LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber)
    {
        //----------- allocate vec of obs :
        tDoubleVect aVObs(12, 0.0); // 6 for ImagePre and 6 for ImagePost

        //----------- extract current parameters
        tDenseVect aVCur = mSys->CurGlobSol(); // Get current solution.

        tIm aCurPreIm = tIm(mSzImPre);
        tDIm *aCurPreDIm = nullptr;
        tIm aCurPostIm = tIm(mSzImPost);
        tDIm *aCurPostDIm = nullptr;

        mIsLastIters = false;

        if (mUseMultiScaleApproach)
            mIsLastIters = cAppli_cTriangleDeformation::ManageDifferentCasesOfEndIterations(aIterNumber, mNumberOfScales, mNumberOfEndIterations,
                                                                                            mIsLastIters, mImPre, mImPost, aCurPreIm, aCurPreDIm,
                                                                                            aCurPostIm, aCurPostDIm);
        else
        {
            cAppli_cTriangleDeformation::LoadImageAndData(aCurPreIm, aCurPreDIm, "pre", mImPre, mImPost);
            cAppli_cTriangleDeformation::LoadImageAndData(aCurPostIm, aCurPostDIm, "post", mImPre, mImPost);
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
                aCurPreDIm->ToFile("GaussFilteredImPre_iter_" + std::to_string(aIterNumber) + ".tif");
        }
        else if (mUseMultiScaleApproach && mIsLastIters)
        {
            LoadImageAndData(aCurPreIm, aCurPreDIm, "pre", mImPre, mImPost);
            LoadImageAndData(aCurPostIm, aCurPostDIm, "post", mImPre, mImPost);
        }

        //----------- declaration of indicator of convergence
        tREAL8 aSomDif = 0; // sum of difference between untranslated pixel and translated one.
        size_t aNbOut = 0;  // number of translated pixels out of image

        // Count number of pixels inside triangles for normalisation
        size_t aTotalNumberOfInsidePixels = 0;

        // Loop over all triangles to add the observations on each point
        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTri = mDelTri.KthTri(aTr);
            const cPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

            const cTriangle2DCompiled aCompTri(aTri);

            std::vector<cPt2di> aVectorToFillWithInsidePixels;
            aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

            //----------- index of unknown, finds the associated pixels of current triangle knots
            const tIntVect aVecInd = {2 * aIndicesOfTriKnots.x(), 2 * aIndicesOfTriKnots.x() + 1,
                                      2 * aIndicesOfTriKnots.y(), 2 * aIndicesOfTriKnots.y() + 1,
                                      2 * aIndicesOfTriKnots.z(), 2 * aIndicesOfTriKnots.z() + 1};

            const tREAL8 aCurRadTrPointA = aVCur(aVecInd.at(0)); // current translation on radiometry 1st point of triangle
            const tREAL8 aCurRadScPointA = aVCur(aVecInd.at(1)); // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointB = aVCur(aVecInd.at(2)); // current translation on radiometry 2nd point of triangle
            const tREAL8 aCurRadScPointB = aVCur(aVecInd.at(3)); // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointC = aVCur(aVecInd.at(4)); // current translation on radiometry 3rd point of triangle
            const tREAL8 aCurRadScPointC = aVCur(aVecInd.at(5)); // current scale on radiometry 3rd point of triangle

            // soft constraint radiometric translation
            if (mWeightRadTranslation >= 0)
            {
                const int aSolStep = 2; // adapt step to solution vector configuration
                const int aSolStart = 0;
                for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size() - 1; aIndCurSol += aSolStep)
                {
                    const int aIndices = aVecInd.at(aIndCurSol);
                    mSys->AddEqFixVar(aIndices, aVCur(aIndices), mWeightRadTranslation);
                }
            }

            // soft constraint radiometric scaling
            if (mWeightRadScale >= 0)
            {
                const int aSolStep = 2; // adapt step to solution vector configuration
                const int aSolStart = 1;
                for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size(); aIndCurSol += aSolStep)
                {
                    const int aIndices = aVecInd.at(aIndCurSol);
                    mSys->AddEqFixVar(aIndices, aVCur(aIndices), mWeightRadScale);
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

                // radiometry translation of pixel by current radiometry translation of triangle knots
                const tREAL8 aRadiometryTranslation = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForTranslationRadiometry(aCurRadTrPointA,
                                                                                                                                             aCurRadTrPointB,
                                                                                                                                             aCurRadTrPointC,
                                                                                                                                             aVObs);

                // radiometry scaling of pixel by current radiometry scaling of triangle knots
                const tREAL8 aRadiometryScaling = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForScalingRadiometry(aCurRadScPointA,
                                                                                                                                     aCurRadScPointB,
                                                                                                                                     aCurRadScPointC,
                                                                                                                                     aVObs);

                const cPt2dr aInsideTrianglePoint = aPixInsideTriangle.GetCartesianCoordinates();
                const cPt2di aEastTranslatedPoint = cPt2di(aInsideTrianglePoint.x(), aInsideTrianglePoint.y()) + cPt2di(1, 0);
                const cPt2di aSouthTranslatedPoint = cPt2di(aInsideTrianglePoint.x(), aInsideTrianglePoint.y()) + cPt2di(0, 1);

                if (aCurPostDIm->InsideBL(cPt2dr(aEastTranslatedPoint.x(), aEastTranslatedPoint.y()))) // avoid errors
                {
                    if (aCurPostDIm->InsideBL(cPt2dr(aSouthTranslatedPoint.x(), aSouthTranslatedPoint.y()))) // avoid errors
                    {
                        // prepare for application of bilinear formula
                        FormalBilinTri_SetObs(aVObs, TriangleDisplacement_NbObs, aInsideTrianglePoint, *aCurPostDIm);

                        // Now add observation
                        mSys->CalcAndAddObs(mEqRadTri, aVecInd, aVObs);

                        // compute indicators
                        const tREAL8 aBilinearRadiomValue = aRadiometryScaling * aCurPostDIm->GetVBL(aPixInsideTriangle.GetCartesianCoordinates()) +
                                                            aRadiometryTranslation;
                        const tREAL8 aDif = aVObs[5] - aBilinearRadiomValue; // residual : aValueImPre - aBilinearRadiomValue
                        aSomDif += std::abs(aDif);
                    }
                }
                else
                    aNbOut++;
                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
            }
        }

        // Update all parameter taking into account previous observation
        mSys->SolveUpdateReset();

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_cTriangleDeformationRad::GenerateOutputImageAndDisplayLastRadiometryValues(const tDenseVect &aVFinalSol, const int aIterNumber)
    {
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = cPt2di(mDImOut->Sz().x(), mDImOut->Sz().y());

        tIm aLastPostIm = tIm(mSzImPost);
        tDIm *aLastPostDIm = nullptr;
        cAppli_cTriangleDeformation::LoadImageAndData(aLastPostIm, aLastPostDIm, "post", mImPre, mImPost);

        if (mUseMultiScaleApproach && !mIsLastIters)
        {
            aLastPostIm = mImPost.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
            aLastPostDIm = &aLastPostIm.DIm();
        }

        for (const cPt2di &aOutPix : *mDImOut) // Initialise output image
            mDImOut->SetV(aOutPix, aLastPostDIm->GetV(aOutPix));

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTri = mDelTri.KthTri(aLTr);
            const cPt3di aLastIndicesOfTriKnots = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTri);

            std::vector<cPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecInd = {2 * aLastIndicesOfTriKnots.x(), 2 * aLastIndicesOfTriKnots.x() + 1,
                                          2 * aLastIndicesOfTriKnots.y(), 2 * aLastIndicesOfTriKnots.y() + 1,
                                          2 * aLastIndicesOfTriKnots.z(), 2 * aLastIndicesOfTriKnots.z() + 1};

            const tREAL8 aLastRadTrPointA = aVFinalSol(aLastVecInd.at(0)); // last translation on radiometry 1st point of triangle
            const tREAL8 aLastRadScPointA = aVFinalSol(aLastVecInd.at(1)); // last scale on radiometry 1st point of triangle
            const tREAL8 aLastRadTrPointB = aVFinalSol(aLastVecInd.at(2)); // last translation on radiometry 2nd point of triangle
            const tREAL8 aLastRadScPointB = aVFinalSol(aLastVecInd.at(3)); // last scale on radiometry 2nd point of triangle
            const tREAL8 aLastRadTrPointC = aVFinalSol(aLastVecInd.at(4)); // last translation on radiometry 3rd point of triangle
            const tREAL8 aLastRadScPointC = aVFinalSol(aLastVecInd.at(5)); // last scale on radiometry 3rd point of triangle

            const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

            for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
            {
                const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                     aLastFilledPixel, *aLastPostDIm);

                // radiometry translation of pixel by current radiometry translation
                const tREAL8 aLastRadiometryTranslation = cAppli_cTriangleDeformationRadiometry::ApplyLastBarycenterInterpolationFormulaRadiometryTranslation(aLastRadTrPointA,
                                                                                                                                                              aLastRadTrPointB,
                                                                                                                                                              aLastRadTrPointC,
                                                                                                                                                              aLastPixInsideTriangle);

                const tREAL8 aLastRadiometryScaling = cAppli_cTriangleDeformationRadiometry::ApplyLastBarycenterInterpolationFormulaRadiometryScaling(aLastRadScPointA,
                                                                                                                                                      aLastRadScPointB,
                                                                                                                                                      aLastRadScPointC,
                                                                                                                                                      aLastPixInsideTriangle);

                cAppli_cTriangleDeformationRadiometry::FillOutputImage(aLastPixInsideTriangle, aLastRadiometryTranslation,
                                                                       aLastRadiometryScaling);
            }
        }

        // save output image with calculated radiometries to image file
        mDImOut->ToFile("OutputImage_" + std::to_string(mNumberOfScales) + "_" + std::to_string(mNumberPointsToGenerate) + ".tif");

        if (mDisplayLastRadiometryValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 2 == 1 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
    }

    void cAppli_cTriangleDeformationRad::DoOneIterationRadiometry(const int aIterNumber)
    {
        LoopOverTrianglesAndUpdateParametersRadiometry(aIterNumber); // Iterate over triangles and solve system

        tDenseVect aVFinalSol = mSys->CurGlobSol();
        // Show final translation results and produce displacement maps
        if (mUseMultiScaleApproach)
        {
            if (aIterNumber == (mNumberOfScales + mNumberOfEndIterations - 1))
                GenerateOutputImageAndDisplayLastRadiometryValues(aVFinalSol, aIterNumber);
        }
        else
        {
            if (aIterNumber == (mNumberOfScales - 1))
                GenerateOutputImageAndDisplayLastRadiometryValues(aVFinalSol, aIterNumber);
        }
    }

    //-----------------------------------------

    int cAppli_cTriangleDeformationRad::Exe()
    {
        // read pre and post images and their sizes
        mImPre = tIm::FromFile(mNamePreImage);
        mImPost = tIm::FromFile(mNamePostImage);

        mDImPre = &mImPre.DIm();
        mSzImPre = mDImPre->Sz();

        mDImPost = &mImPost.DIm();
        mSzImPost = mDImPost->Sz();

        if (mUseMultiScaleApproach)
            mSigmaGaussFilter = mNumberOfScales * mSigmaGaussFilterStep;

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        cAppli_cTriangleDeformation::GeneratePointsForDelaunay(mVectorPts, mNumberPointsToGenerate, mRandomUniformLawUpperBoundLines,
                                                               mRandomUniformLawUpperBoundCols, mDelTri, mSzImPre);

        cAppli_cTriangleDeformationRadiometry::InitialisationAfterExeRadiometry(mDelTri, mSys);

        if (mUseMultiScaleApproach)
        {
            for (int aIterNumber = 0; aIterNumber < mNumberOfScales + mNumberOfEndIterations; aIterNumber++)
                DoOneIterationRadiometry(aIterNumber);
        }
        else
        {
            for (int aIterNumber = 0; aIterNumber < mNumberOfScales; aIterNumber++)
                DoOneIterationRadiometry(aIterNumber);
        }

        return EXIT_SUCCESS;
    }

    /********************************************/
    //              ::MMVII                     //
    /********************************************/

    tMMVII_UnikPApli Alloc_cTriangleDeformationRad(const std::vector<std::string> &aVArgs,
                                                   const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_cTriangleDeformationRad(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_ComputeTriangleDeformationRad(
        "ComputeTriangleDeformationRad",
        Alloc_cTriangleDeformationRad,
        "Compute radiometric deformation between images using triangles and alternative equation",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII