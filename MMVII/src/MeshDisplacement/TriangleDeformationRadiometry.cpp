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
                                                                               const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                                                                                                mNumberOfLines(1),
                                                                                                                mNumberOfCols(1),
                                                                                                                mShow(true),
                                                                                                                mUseMultiScaleApproach(false),
                                                                                                                mGenerateOutputImage(true),
                                                                                                                mBuildRandomUniformGrid(false),
                                                                                                                mUseLinearGradInterpolation(false),
                                                                                                                mInterpolArgs({"Tabul", "1000", "Cubic", "-0.5"}),
                                                                                                                mSerialiseTriangleNodes(false),
                                                                                                                mNameMultipleTriangleNodes("TriangulationNodes.xml"),
                                                                                                                mInitialiseWithUserValues(true),
                                                                                                                mInitialiseRadTrValue(0),
                                                                                                                mInitialiseRadScValue(1),
                                                                                                                mWeightRadTranslation(-1),
                                                                                                                mWeightRadScale(-1),
                                                                                                                mUserDefinedFolderNameToSaveResult(""),
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
                                                                                                                mInterpolRad(nullptr),
                                                                                                                mSysRadiometry(nullptr),
                                                                                                                mEqRadiometryTri(nullptr)

    {
    }

    cAppli_TriangleDeformationRadiometry::~cAppli_TriangleDeformationRadiometry()
    {
        delete mSysRadiometry;
        delete mEqRadiometryTri;
        delete mInterpolRad;
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
               << AOpt2007(mGenerateOutputImage, "GenerateOutputImage",
                           "Whether to generate and save the output image with computed radiometry", {eTA2007::HDV})
               << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
                           "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
               << AOpt2007(mUseLinearGradInterpolation, "UseLinearGradientInterpolation",
                           "Use linear gradient interpolation instead of bilinear interpolation.", {eTA2007::HDV})
               << AOpt2007(mInterpolArgs, "InterpolationName", "Which type of interpolation to use : cubic, sinc or MMVIIK", {eTA2007::HDV})
               << AOpt2007(mSerialiseTriangleNodes, "SerialiseTriangleNodes", "Whether to serialise triangle nodes to .xml file or not", {eTA2007::HDV})
               << AOpt2007(mNameMultipleTriangleNodes, "NameOfMultipleTriangleNodes", "File name to use when saving all triangle nodes values to .xml file", {eTA2007::HDV})
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
               << AOpt2007(mUserDefinedFolderNameToSaveResult, "FolderNameToSaveResults",
                           "Folder name where to store produced results", {eTA2007::HDV})
               << AOpt2007(mDisplayLastRadiometryValues, "DisplayLastRadiometryValues",
                           "Whether to display or not the last values of radiometry unknowns after optimisation process.", {eTA2007::HDV})
               << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep",
                           "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV, eTA2007::Tuning});
    }

    void cAppli_TriangleDeformationRadiometry::LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber, const int aTotalNumberOfIters,
                                                                                              const bool aNonEmptyFolderName)
    {
        //----------- allocate vec of obs :
        // 6 for ImagePre and 5 for ImagePost in linear gradient case and 6 in bilinear case
        const int aNumberOfObsRad = mUseLinearGradInterpolation ? TriangleDisplacement_GradInterpol_NbObs : TriangleDisplacement_Bilin_NbObs;
        tDoubleVect aVObsRad(6 + aNumberOfObsRad, 0);

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
        int aNodeCounterRad = 0;
        std::unique_ptr<cMultipleTriangleNodesSerialiser> aVectorOfTriangleNodesRad = nullptr;

        if (mSerialiseTriangleNodes && aVectorOfTriangleNodesRad == nullptr)
            aVectorOfTriangleNodesRad = cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(mNameMultipleTriangleNodes);

        // Loop over all triangles to add the observations on each point
        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTriRad = mDelTri.KthTri(aTr);
            const cPt3di aIndicesOfTriKnotsRad = mDelTri.KthFace(aTr);

            const cTriangle2DCompiled aCompTri(aTriRad);

            std::vector<tPt2di> aVectorToFillWithInsidePixels;
            aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

            //----------- index of unknown, finds the associated pixels of current triangle knots
            const tIntVect aVecIndRad = {2 * aIndicesOfTriKnotsRad.x(), 2 * aIndicesOfTriKnotsRad.x() + 1,
                                         2 * aIndicesOfTriKnotsRad.y(), 2 * aIndicesOfTriKnotsRad.y() + 1,
                                         2 * aIndicesOfTriKnotsRad.z(), 2 * aIndicesOfTriKnotsRad.z() + 1};

            tREAL8 aCurRadTrPointA = 0;
            tREAL8 aCurRadScPointA = 1;
            tREAL8 aCurRadTrPointB = 0;
            tREAL8 aCurRadScPointB = 1;
            tREAL8 aCurRadTrPointC = 0;
            tREAL8 aCurRadScPointC = 1;

            if (!mSerialiseTriangleNodes && aVectorOfTriangleNodesRad == nullptr)
            {
                // current translation on radiometry 1st point of triangle
                aCurRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0);
                // current scale on radiometry 1st point of triangle
                aCurRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0);
                // current translation on radiometry 2nd point of triangle
                aCurRadTrPointB = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1);
                // current scale on radiometry 2nd point of triangle
                aCurRadScPointB = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1);
                // current translation on radiometry 3rd point of triangle
                aCurRadTrPointC = LoadNodeAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2);
                // current scale on radiometry 3rd point of triangle
                aCurRadScPointC = LoadNodeAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2);
            }
            else if (mSerialiseTriangleNodes && aVectorOfTriangleNodesRad != nullptr)
            {
                // current translation on radiometry 1st point of triangle
                aCurRadTrPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0,
                                                                                            aNodeCounterRad, aIndicesOfTriKnotsRad, true, aVectorOfTriangleNodesRad);
                // current scale on radiometry 1st point of triangle
                aCurRadScPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 0, 1, 0, 1, aTriRad, 0, aNodeCounterRad,
                                                                                        aIndicesOfTriKnotsRad, false, aVectorOfTriangleNodesRad);
                // current translation on radiometry 2nd point of triangle
                aCurRadTrPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1,
                                                                                            aNodeCounterRad, aIndicesOfTriKnotsRad, true, aVectorOfTriangleNodesRad);
                // current scale on radiometry 2nd point of triangle
                aCurRadScPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 2, 3, 2, 3, aTriRad, 1, aNodeCounterRad + 1,
                                                                                        aIndicesOfTriKnotsRad, false, aVectorOfTriangleNodesRad);
                // current translation on radiometry 3rd point of triangle
                aCurRadTrPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2,
                                                                                            aNodeCounterRad, aIndicesOfTriKnotsRad, true, aVectorOfTriangleNodesRad);
                // current scale on radiometry 3rd point of triangle
                aCurRadScPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVCurSolRad, aVecIndRad, 4, 5, 4, 5, aTriRad, 2, aNodeCounterRad + 2,
                                                                                        aIndicesOfTriKnotsRad, false, aVectorOfTriangleNodesRad);
                aNodeCounterRad += 3;
            }

            // soft constraint radiometric translation
            if (mWeightRadTranslation >= 0)
            {
                const int aSolStart = 0;
                const int aSolStep = 2; // adapt step to solution vector configuration
                for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecIndRad.size() - 1; aIndCurSol += aSolStep)
                {
                    const int aIndices = aVecIndRad.at(aIndCurSol);
                    mSysRadiometry->AddEqFixVar(aIndices, aVCurSolRad(aIndices), mWeightRadTranslation);
                }
            }

            // soft constraint radiometric scaling
            if (mWeightRadScale >= 0)
            {
                const int aSolStart = 1;
                const int aSolStep = 2; // adapt step to solution vector configuration
                for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecIndRad.size(); aIndCurSol += aSolStep)
                {
                    const int aIndices = aVecIndRad.at(aIndCurSol);
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

                const bool aPixInside = (mUseLinearGradInterpolation) ? aCurPostDIm->InsideInterpolator(*mInterpolRad, aInsideTrianglePoint, 0) : (aCurPostDIm->InsideBL(tPt2dr(aEastTranslatedPoint.x(), aEastTranslatedPoint.y())) && aCurPostDIm->InsideBL(tPt2dr(aSouthTranslatedPoint.x(), aSouthTranslatedPoint.y())));
                if (aPixInside)
                {
                    if (mUseLinearGradInterpolation)
                        // prepare for application of linear gradient formula
                        FormalGradInterpol_SetObs(aVObsRad, TriangleDisplacement_NbObs_ImPre, aInsideTrianglePoint,
                                                  *aCurPostDIm, *mInterpolRad);
                    else
                        // prepare for application of bilinear formula
                        FormalBilinTri_SetObs(aVObsRad, TriangleDisplacement_NbObs_ImPre, aInsideTrianglePoint, *aCurPostDIm);

                    // Now add observation
                    mSysRadiometry->CalcAndAddObs(mEqRadiometryTri, aVecIndRad, aVObsRad);

                    const tREAL8 aInterpolatedValue = (mUseLinearGradInterpolation) ? aCurPostDIm->GetValueInterpol(*mInterpolRad, aInsideTrianglePoint) : aCurPostDIm->GetVBL(aInsideTrianglePoint);
                    // compute indicators
                    const tREAL8 aRadiomValueImPre = aRadiometryScaling * aVObsRad[5] + aRadiometryTranslation;
                    const tREAL8 aDif = aRadiomValueImPre - aInterpolatedValue; // residual
                    aSomDif += std::abs(aDif);
                }
                else
                    aNbOut++;

                aTotalNumberOfInsidePixels += aNumberOfInsidePixels;
            }
        }

        // Update all parameter taking into account previous observation
        mSysRadiometry->SolveUpdateReset();

        // Save all triangle nodes to .xml file
        if (mSerialiseTriangleNodes && aVectorOfTriangleNodesRad != nullptr)
            aVectorOfTriangleNodesRad->MultipleNodesToFile(mNameMultipleTriangleNodes);

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_TriangleDeformationRadiometry::GenerateOutputImage(const tDenseVect &aVFinalSol, const int aTotalNumberOfIterations,
                                                                   const bool aNonEmptyFolderName)
    {
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = mDImOut->Sz();

        tIm aLastPreIm = tIm(mSzImPre);
        tDIm *aLastPreDIm = nullptr;
        LoadPrePostImageAndData(aLastPreIm, aLastPreDIm, "pre", mImPre, mImPost);

        std::unique_ptr<cMultipleTriangleNodesSerialiser> aLastVectorOfTriangleNodesRad = nullptr;

        if (mUseMultiScaleApproach && !mIsLastIters)
        {
            aLastPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
            aLastPreDIm = &aLastPreIm.DIm();
        }

        for (const tPt2di &aOutPix : *mDImOut) // Initialise output image
            mDImOut->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));

        int aLastNodeCounter = 0;

        for (size_t aLTr = 0; aLTr < mDelTri.NbFace(); aLTr++)
        {
            const tTri2dr aLastTriRad = mDelTri.KthTri(aLTr);
            const cPt3di aLastIndicesOfTriKnotsRad = mDelTri.KthFace(aLTr);

            const cTriangle2DCompiled aLastCompTri(aLastTriRad);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tIntVect aLastVecIndRad = {2 * aLastIndicesOfTriKnotsRad.x(), 2 * aLastIndicesOfTriKnotsRad.x() + 1,
                                             2 * aLastIndicesOfTriKnotsRad.y(), 2 * aLastIndicesOfTriKnotsRad.y() + 1,
                                             2 * aLastIndicesOfTriKnotsRad.z(), 2 * aLastIndicesOfTriKnotsRad.z() + 1};
            tREAL8 aLastRadTrPointA = 0;
            tREAL8 aLastRadScPointA = 1;
            tREAL8 aLastRadTrPointB = 0;
            tREAL8 aLastRadScPointB = 1;
            tREAL8 aLastRadTrPointC = 0;
            tREAL8 aLastRadScPointC = 1;

            if (!mSerialiseTriangleNodes)
            {
                // last radiometry translation of 1st point
                aLastRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0);
                // last radiometry scaling of 1st point
                aLastRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0);
                 // last radiometry translation of 2nd point
                aLastRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1);
                // last radiometry scaling of 2nd point
                aLastRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1);
                 // last radiometry translation of 3rd point
                aLastRadTrPointA = LoadNodeAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2);
                // last radiometry scaling of 3rd point
                aLastRadScPointA = LoadNodeAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2);
            }
            else
            {
                // last radiometry translation of 1st point
                aLastRadTrPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0, aLastNodeCounter,
                                                                                             aLastIndicesOfTriKnotsRad, false, aLastVectorOfTriangleNodesRad);
                // last radiometry scaling of 1st point
                aLastRadScPointA = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 0, 1, 0, 1, aLastTriRad, 0, aLastNodeCounter,
                                                                                         aLastIndicesOfTriKnotsRad, false, aLastVectorOfTriangleNodesRad);
                // last radiometry translation of 2nd point
                aLastRadTrPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1, aLastNodeCounter + 1,
                                                                                             aLastIndicesOfTriKnotsRad, false, aLastVectorOfTriangleNodesRad);
                // last radiometry scaling of 2nd point
                aLastRadScPointB = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 2, 3, 2, 3, aLastTriRad, 1, aLastNodeCounter + 1,
                                                                                         aLastIndicesOfTriKnotsRad, false, aLastVectorOfTriangleNodesRad);
                // last radiometry translation of 3rd point
                aLastRadTrPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2, aLastNodeCounter + 2,
                                                                                             aLastIndicesOfTriKnotsRad, false, aLastVectorOfTriangleNodesRad);
                // last radiometry scaling of 3rd point
                aLastRadScPointC = LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(aVFinalSol, aLastVecIndRad, 4, 5, 4, 5, aLastTriRad, 2, aLastNodeCounter + 2,
                                                                                         aLastIndicesOfTriKnotsRad, false, aLastVectorOfTriangleNodesRad);
                aLastNodeCounter += 3;
            }

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
        SaveOutputImageToFile(mDImOut, aNonEmptyFolderName, mUserDefinedFolderNameToSaveResult, "OutputImage",
                              mNumberPointsToGenerate, aTotalNumberOfIterations);
    }

    void cAppli_TriangleDeformationRadiometry::DoOneIterationRadiometry(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                        const tDenseVect &aVInitSol, const bool aNonEmptyFolderName)
    {
        LoopOverTrianglesAndUpdateParametersRadiometry(aIterNumber, aTotalNumberOfIterations, aNonEmptyFolderName); // Iterate over triangles and solve system

        tDenseVect aVFinalSol = mSysRadiometry->CurGlobSol();

        // Show final translation results and produce displacement maps
        if (aIterNumber == (aTotalNumberOfIterations - 1))
        {
            GenerateOutputImage(aVFinalSol, aTotalNumberOfIterations, aNonEmptyFolderName);
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

        bool aNonEmptyFolderName = false;
        if (!mUserDefinedFolderNameToSaveResult.empty())
        {
            aNonEmptyFolderName = true;
            if (!ExistFile(mUserDefinedFolderNameToSaveResult))
                CreateDirectories(mUserDefinedFolderNameToSaveResult, aNonEmptyFolderName);
        }

        if (mUseMultiScaleApproach)
            mSigmaGaussFilter = mNumberOfScales * mSigmaGaussFilterStep;

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
                                                        mNumberOfCols, mDelTri, mSzImPre, mBuildRandomUniformGrid);

        InitialiseInterpolationAndEquationRadiometry(mEqRadiometryTri, mInterpolRad, mInterpolArgs, mUseLinearGradInterpolation);

        InitialiseWithUserValuesRadiometry(mDelTri, mSysRadiometry, mInitialiseWithUserValues,
                                           mInitialiseRadTrValue, mInitialiseRadScValue);

        const tDenseVect aVInitSolRad = mSysRadiometry->CurGlobSol().Dup(); // Duplicate initial solution

        int aTotalNumberOfIterations = 0;
        (mUseMultiScaleApproach) ? aTotalNumberOfIterations = mNumberOfScales + mNumberOfEndIterations : aTotalNumberOfIterations = mNumberOfScales;

        for (int aIterNumber = 0; aIterNumber < aTotalNumberOfIterations; aIterNumber++)
            DoOneIterationRadiometry(aIterNumber, aTotalNumberOfIterations, aVInitSolRad, aNonEmptyFolderName);

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
