#include "cMMVII_Appli.h"

#include "MMVII_TplSymbTriangle.h"

#include "TriangleDeformationTrRad.h"

/**
   \file TriangleDeformationTrRad.cpp

   \brief file for computing 2D deformations (translation and radiometry)
   between 2 images thanks to triangles.
**/

namespace MMVII
{
    /******************************************/
    /*                                        */
    /*   cAppli_cTriangleDeformationTrRad     */
    /*                                        */
    /******************************************/

    cAppli_cTriangleDeformationTrRad::cAppli_cTriangleDeformationTrRad(const std::vector<std::string> &aVArgs,
                                                                       const cSpecMMVII_Appli &aSpec) : cAppli_cTriangleDeformation(aVArgs, aSpec),
                                                                                                        mRandomUniformLawUpperBoundLines(1),
                                                                                                        mRandomUniformLawUpperBoundCols(1),
                                                                                                        mShow(true),
                                                                                                        mUseMultiScaleApproach(true),
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
                                                                                                        mDisplayLastTranslationValues(false),
                                                                                                        mDisplayLastRadiometryValues(false),
                                                                                                        mSzImPre(cPt2di(1, 1)),
                                                                                                        mImPre(mSzImPre),
                                                                                                        mDImPre(nullptr),
                                                                                                        mSzImPost(cPt2di(1, 1)),
                                                                                                        mImPost(mSzImPost),
                                                                                                        mDImPost(nullptr),
                                                                                                        mSzImOut(cPt2di(1, 1)),
                                                                                                        mImOut(mSzImOut),
                                                                                                        mDImOut(nullptr),
                                                                                                        mSzImDepX(cPt2di(1, 1)),
                                                                                                        mImDepX(mSzImDepX),
                                                                                                        mDImDepX(nullptr),
                                                                                                        mSzImDepY(cPt2di(1, 1)),
                                                                                                        mImDepY(mSzImDepY),
                                                                                                        mDImDepY(nullptr),
                                                                                                        mVectorPts({cPt2dr(0, 0)}),
                                                                                                        mDelTri(mVectorPts),
                                                                                                        mSys(nullptr),
                                                                                                        mEqTriDeformTrRad(nullptr)
    {
        mEqTriDeformTrRad = EqDeformTriTrRad(true, 1); // true means with derivative, 1 is size of buffer
    }

    cAppli_cTriangleDeformationTrRad::~cAppli_cTriangleDeformationTrRad()
    {
        delete mSys;
        delete mEqTriDeformTrRad;
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformationTrRad::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach optimisation process.");
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformationTrRad::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mRandomUniformLawUpperBoundCols, "RandomUniformLawUpperBoundXAxis",
                           "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV})
               << AOpt2007(mRandomUniformLawUpperBoundLines, "RandomUniformLawUpperBoundYAxis",
                           "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV})
               << AOpt2007(mShow, "Show", "Whether to print minimisation results.", {eTA2007::HDV})
               << AOpt2007(mUseMultiScaleApproach, "UseMultiScaleApproach", "Whether to use multi-scale approach or not.", {eTA2007::HDV})
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
               << AOpt2007(mDisplayLastTranslationValues, "DisplayLastTranslationsValues",
                           "Whether to display the final values of unknowns linked to point translation.", {eTA2007::HDV})
               << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                           "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV});
    }

    void cAppli_cTriangleDeformationTrRad::LoopOverTrianglesAndUpdateParameters(const int aIterNumber)
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
            cAppli_cTriangleDeformation::LoadImageAndData(aCurPreIm, aCurPreDIm, "pre", mImPre, mImPost);
            cAppli_cTriangleDeformation::LoadImageAndData(aCurPostIm, aCurPostDIm, "post", mImPre, mImPost);
        }

        //----------- declaration of indicator of convergence
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
                const cPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

                const tIntVect aVecInd = {4 * aIndicesOfTriKnots.x(), 4 * aIndicesOfTriKnots.x() + 1,
                                          4 * aIndicesOfTriKnots.x() + 2, 4 * aIndicesOfTriKnots.x() + 3,
                                          4 * aIndicesOfTriKnots.y(), 4 * aIndicesOfTriKnots.y() + 1,
                                          4 * aIndicesOfTriKnots.y() + 2, 4 * aIndicesOfTriKnots.y() + 3,
                                          4 * aIndicesOfTriKnots.z(), 4 * aIndicesOfTriKnots.z() + 1,
                                          4 * aIndicesOfTriKnots.z() + 2, 4 * aIndicesOfTriKnots.z() + 3};

                if (mFreezeTranslationX)
                {
                    mSys->SetFrozenVar(aVecInd.at(2), aVCur(0));
                    mSys->SetFrozenVar(aVecInd.at(6), aVCur(4));
                    mSys->SetFrozenVar(aVecInd.at(10), aVCur(8));
                }
                if (mFreezeTranslationY)
                {
                    mSys->SetFrozenVar(aVecInd.at(2), aVCur(1));
                    mSys->SetFrozenVar(aVecInd.at(6), aVCur(5));
                    mSys->SetFrozenVar(aVecInd.at(10), aVCur(9));
                }
                if (mFreezeRadTranslation)
                {
                    mSys->SetFrozenVar(aVecInd.at(2), aVCur(2));
                    mSys->SetFrozenVar(aVecInd.at(6), aVCur(6));
                    mSys->SetFrozenVar(aVecInd.at(10), aVCur(10));
                }
                if (mFreezeRadScale)
                {
                    mSys->SetFrozenVar(aVecInd.at(3), aVCur(3));
                    mSys->SetFrozenVar(aVecInd.at(7), aVCur(7));
                    mSys->SetFrozenVar(aVecInd.at(11), aVCur(11));
                }
            }
        }

        // Loop over all triangles to add the observations on each point
        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTri = mDelTri.KthTri(aTr);
            const cPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

            const cTriangle2DCompiled aCompTri(aTri);

            std::vector<cPt2di> aVectorToFillWithInsidePixels;
            aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

            //----------- index of unknown, finds the associated pixels of current triangle
            const tIntVect aVecInd = {4 * aIndicesOfTriKnots.x(), 4 * aIndicesOfTriKnots.x() + 1,
                                      4 * aIndicesOfTriKnots.x() + 2, 4 * aIndicesOfTriKnots.x() + 3,
                                      4 * aIndicesOfTriKnots.y(), 4 * aIndicesOfTriKnots.y() + 1,
                                      4 * aIndicesOfTriKnots.y() + 2, 4 * aIndicesOfTriKnots.y() + 3,
                                      4 * aIndicesOfTriKnots.z(), 4 * aIndicesOfTriKnots.z() + 1,
                                      4 * aIndicesOfTriKnots.z() + 2, 4 * aIndicesOfTriKnots.z() + 3};

            const cPt2dr aCurTrPointA = cPt2dr(aVCur(aVecInd.at(0)),
                                               aVCur(aVecInd.at(1))); // current translation 1st point of triangle
            const cPt2dr aCurTrPointB = cPt2dr(aVCur(aVecInd.at(4)),
                                               aVCur(aVecInd.at(5))); // current translation 2nd point of triangle
            const cPt2dr aCurTrPointC = cPt2dr(aVCur(aVecInd.at(8)),
                                               aVCur(aVecInd.at(9))); // current translation 3rd point of triangle

            const tREAL8 aCurRadTrPointA = aVCur(aVecInd.at(2));  // current translation on radiometry 1st point of triangle
            const tREAL8 aCurRadScPointA = aVCur(aVecInd.at(3));  // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointB = aVCur(aVecInd.at(6));  // current translation on radiometry 2nd point of triangle
            const tREAL8 aCurRadScPointB = aVCur(aVecInd.at(7));  // current scale on radiometry 3rd point of triangle
            const tREAL8 aCurRadTrPointC = aVCur(aVecInd.at(10)); // current translation on radiometry 3rd point of triangle
            const tREAL8 aCurRadScPointC = aVCur(aVecInd.at(11)); // current scale on radiometry 3rd point of triangle

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
                const cPt2dr aTranslatedFilledPoint = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaToFilledPixel(aCurTrPointA, aCurTrPointB,
                                                                                                                                  aCurTrPointC, aVObs);
                // radiometry translation of pixel by current radiometry translation of triangle knots
                const tREAL8 aRadiometryTranslation = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForTranslationRadiometry(aCurRadTrPointA,
                                                                                                                                             aCurRadTrPointB,
                                                                                                                                             aCurRadTrPointC,
                                                                                                                                             aVObs);
                // radiometry translation of pixel by current radiometry scaling of triangle knots
                const tREAL8 aRadiometryScaling = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForScalingRadiometry(aCurRadScPointA,
                                                                                                                                     aCurRadScPointB,
                                                                                                                                     aCurRadScPointC,
                                                                                                                                     aVObs);

                if (aCurPostDIm->InsideBL(aTranslatedFilledPoint)) // avoid errors
                {
                    // prepare for application of bilinear formula
                    FormalBilinTri_SetObs(aVObs, TriangleDisplacement_NbObs, aTranslatedFilledPoint, *aCurPostDIm);

                    // Now add observation
                    mSys->CalcAndAddObs(mEqTriDeformTrRad, aVecInd, aVObs);

                    // compute indicators
                    const tREAL8 aBilinearRadiomValue = aRadiometryScaling * aCurPostDIm->GetVBL(aTranslatedFilledPoint) + aRadiometryTranslation;
                    const tREAL8 aDif = aVObs[5] - aBilinearRadiomValue; // residual : aValueImPre - aBilinearRadiomValue
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
                GenerateDisplacementMapsAndOutputImages(aVCur, aIterNumber);
        }

        // Update all parameter taking into account previous observation
        mSys->SolveUpdateReset();

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_cTriangleDeformationTrRad::FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                                                              const cPt2dr &aLastTranslatedFilledPoint,
                                                                              const tREAL8 aLastRadiometryTranslation,
                                                                              const tREAL8 aLastRadiometryScaling)
    {
        const tREAL8 aLastXCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().x();
        const tREAL8 aLastYCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().y();

        const cPt2di aLastCoordinate = cPt2di(aLastXCoordinate, aLastYCoordinate);
        mDImDepX->SetV(aLastCoordinate,
                       aLastTranslatedFilledPoint.x() - aLastXCoordinate);
        mDImDepY->SetV(aLastCoordinate,
                       aLastTranslatedFilledPoint.y() - aLastYCoordinate);
        const tREAL8 aLastXTranslatedCoord = aLastXCoordinate + mDImDepX->GetV(aLastCoordinate);
        const tREAL8 aLastYTranslatedCoord = aLastYCoordinate + mDImDepY->GetV(aLastCoordinate);

        const tREAL8 aLastRadiometryValue = aLastRadiometryScaling * aLastPixInsideTriangle.GetPixelValue() +
                                            aLastRadiometryTranslation;

        // Build image with intensities displaced
        // deal with different cases of pixel being translated out of image
        if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(0, 0)));
        else if (aLastXTranslatedCoord >= mSzImOut.x() && aLastYTranslatedCoord >= mSzImOut.y())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(mSzImOut.x() - 1, mSzImOut.y() - 1)));
        else if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord >= mSzImOut.y())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(0, mSzImOut.y() - 1)));
        else if (aLastXTranslatedCoord >= mSzImOut.x() && aLastYTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(mSzImOut.x() - 1, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < mSzImOut.x() &&
                 aLastYTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(aLastXTranslatedCoord, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < mSzImOut.x() &&
                 aLastYTranslatedCoord > mSzImOut.y())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(aLastXTranslatedCoord, mSzImOut.y() - 1)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < mSzImOut.y() &&
                 aLastXTranslatedCoord < 0)
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(0, aLastYTranslatedCoord)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < mSzImOut.y() &&
                 aLastXTranslatedCoord > mSzImOut.x())
            mDImOut->SetV(aLastCoordinate, mDImOut->GetV(cPt2di(mSzImOut.x() - 1, aLastYTranslatedCoord)));
        else
            // at the translated pixel the untranslated pixel value is given computed with the right radiometry values
            mDImOut->SetV(cPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord), aLastRadiometryValue);
    }

    void cAppli_cTriangleDeformationTrRad::GenerateDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber)
    {
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = cPt2di(mDImOut->Sz().x(), mDImOut->Sz().y());

        mImDepX = tIm(mSzImPre, 0, eModeInitImage::eMIA_Null);
        mDImDepX = &mImDepX.DIm();

        mImDepY = tIm(mSzImPre, 0, eModeInitImage::eMIA_Null);
        mDImDepY = &mImDepY.DIm();

        tIm aLastPostIm = tIm(mSzImPost);
        tDIm *aLastPostDIm = nullptr;
        cAppli_cTriangleDeformation::LoadImageAndData(aLastPostIm, aLastPostDIm, "post", mImPre, mImPost);

        if (mUseMultiScaleApproach && !mIsLastIters)
        {
            aLastPostIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
            aLastPostDIm = &aLastPostIm.DIm();
        }

        tDoubleVect aLastVObs(12, 0.0);

        for (const cPt2di &aOutPix : *mDImOut) // Initialise output image
            mDImOut->SetV(aOutPix, aLastPostDIm->GetV(aOutPix));

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTri = mDelTri.KthTri(aLTr);
            const cPt3di aLastIndicesOfTriKnots = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTri);

            std::vector<cPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecInd = {
                4 * aLastIndicesOfTriKnots.x(),
                4 * aLastIndicesOfTriKnots.x() + 1,
                4 * aLastIndicesOfTriKnots.x() + 2,
                4 * aLastIndicesOfTriKnots.x() + 3,
                4 * aLastIndicesOfTriKnots.y(),
                4 * aLastIndicesOfTriKnots.y() + 1,
                4 * aLastIndicesOfTriKnots.y() + 2,
                4 * aLastIndicesOfTriKnots.y() + 3,
                4 * aLastIndicesOfTriKnots.z(),
                4 * aLastIndicesOfTriKnots.z() + 1,
                4 * aLastIndicesOfTriKnots.z() + 2,
                4 * aLastIndicesOfTriKnots.z() + 3,
            };

            const cPt2dr aLastTrPointA = cPt2dr(aVFinalSol(aLastVecInd.at(0)),
                                                aVFinalSol(aLastVecInd.at(1))); // last translation 1st point of triangle
            const cPt2dr aLastTrPointB = cPt2dr(aVFinalSol(aLastVecInd.at(4)),
                                                aVFinalSol(aLastVecInd.at(5))); // last translation 2nd point of triangle
            const cPt2dr aLastTrPointC = cPt2dr(aVFinalSol(aLastVecInd.at(8)),
                                                aVFinalSol(aLastVecInd.at(9))); // last translation 3rd point of triangle

            const tREAL8 aLastRadTrPointA = aVFinalSol(aLastVecInd.at(2));  // last translation on radiometry 1st point of triangle
            const tREAL8 aLastRadScPointA = aVFinalSol(aLastVecInd.at(3));  // last scale on radiometry 3rd point of triangle
            const tREAL8 aLastRadTrPointB = aVFinalSol(aLastVecInd.at(6));  // last translation on radiometry 2nd point of triangle
            const tREAL8 aLastRadScPointB = aVFinalSol(aLastVecInd.at(7));  // last scale on radiometry 3rd point of triangle
            const tREAL8 aLastRadTrPointC = aVFinalSol(aLastVecInd.at(10)); // last translation on radiometry 3rd point of triangle
            const tREAL8 aLastRadScPointC = aVFinalSol(aLastVecInd.at(11)); // last scale on radiometry 3rd point of triangle

            const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

            for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
            {
                const cPtInsideTriangles aLastPixInsideTriangle = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                     aLastFilledPixel, *aLastPostDIm);
                // prepare for barycenter translation formula by filling aVObs with different coordinates
                FormalInterpBarycenter_SetObs(aLastVObs, 0, aLastPixInsideTriangle);

                // image of a point in triangle by current translation
                const cPt2dr aLastTranslatedFilledPoint = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
                                                                                                                                      aLastTrPointC, aLastVObs);

                const tREAL8 aLastRadiometryTranslation = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForTranslationRadiometry(aLastRadTrPointA,
                                                                                                                                                 aLastRadTrPointB,
                                                                                                                                                 aLastRadTrPointC,
                                                                                                                                                 aLastVObs);

                const tREAL8 aLastRadiometryScaling = cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForScalingRadiometry(aLastRadScPointA,
                                                                                                                                         aLastRadScPointB,
                                                                                                                                         aLastRadScPointC,
                                                                                                                                         aLastVObs);

                FillDisplacementMapsAndOutputImage(aLastPixInsideTriangle, aLastTranslatedFilledPoint, aLastRadiometryTranslation,
                                                   aLastRadiometryScaling);
            }
        }

        // save displacement maps in x and y to image files
        if (mUseMultiScaleApproach)
        {
            mDImDepX->ToFile("DisplacedPixelsX_iter_" + std::to_string(aIterNumber) + "_" +
                             std::to_string(mNumberPointsToGenerate) + "_" +
                             std::to_string(mNumberOfScales + mNumberOfEndIterations) + ".tif");
            mDImDepY->ToFile("DisplacedPixelsY_iter_" + std::to_string(aIterNumber) + "_" +
                             std::to_string(mNumberPointsToGenerate) + "_" +
                             std::to_string(mNumberOfScales + mNumberOfEndIterations) + ".tif");
            if (aIterNumber == mNumberOfScales + mNumberOfEndIterations - 1)
                mDImOut->ToFile("DisplacedPixels.tif");
        }
        else
        {
            mDImDepX->ToFile("DisplacedPixelsX_" + std::to_string(mNumberPointsToGenerate) + "_" +
                             std::to_string(mNumberOfScales) + ".tif");
            mDImDepY->ToFile("DisplacedPixelsY_" + std::to_string(mNumberPointsToGenerate) + "_" +
                             std::to_string(mNumberOfScales) + ".tif");
            mDImOut->ToFile("DisplacedPixels.tif");
        }
    }

    void cAppli_cTriangleDeformationTrRad::GenerateDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const bool aDisplayLastRadiometryValues,
                                                                                                const bool aDisplayLastTranslationValues)
    {
        tDenseVect aVFinalSol = mSys->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMapsAndOutputImages(aVFinalSol, aIterNumber);

        if (aDisplayLastRadiometryValues || aDisplayLastTranslationValues)
            cAppli_cTriangleDeformation::DisplayLastUnknownValues(aVFinalSol, aDisplayLastRadiometryValues, 
                                                                  aDisplayLastTranslationValues);
    }

    void cAppli_cTriangleDeformationTrRad::DoOneIteration(const int aIterNumber)
    {
        LoopOverTrianglesAndUpdateParameters(aIterNumber); // Iterate over triangles and solve system

        // Show final translation results and produce displacement maps
        if (mUseMultiScaleApproach)
        {
            if (aIterNumber == (mNumberOfScales + mNumberOfEndIterations - 1))
                GenerateDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber, mDisplayLastRadiometryValues,
                                                                     mDisplayLastTranslationValues);
        }
        else
        {
            if (aIterNumber == (mNumberOfScales - 1))
                GenerateDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber, mDisplayLastRadiometryValues,
                                                                     mDisplayLastTranslationValues);
        }
    }

    //-----------------------------------------

    int cAppli_cTriangleDeformationTrRad::Exe()
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

        cAppli_cTriangleDeformation::InitialisationAfterExe(mDelTri, mSys);

        if (mUseMultiScaleApproach)
        {
            for (int aIterNumber = 0; aIterNumber < mNumberOfScales + mNumberOfEndIterations; aIterNumber++)
                DoOneIteration(aIterNumber);
        }
        else
        {
            for (int aIterNumber = 0; aIterNumber < mNumberOfScales; aIterNumber++)
                DoOneIteration(aIterNumber);
        }

        return EXIT_SUCCESS;
    }

    /********************************************/
    //              ::MMVII                     //
    /********************************************/

    tMMVII_UnikPApli Alloc_cTriangleDeformationTrRad(const std::vector<std::string> &aVArgs,
                                                     const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_cTriangleDeformationTrRad(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_ComputeTriangleDeformationTrRad(
        "ComputeTriangleDeformationTrRad",
        Alloc_cTriangleDeformationTrRad,
        "Compute 2D deformation of triangles between images using triangles and alternative formula",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII