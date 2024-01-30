#include "cMMVII_Appli.h"

#include "MMVII_TplSymbTriangle.h"

#include "TriangleDeformation.h"

/**
   \file TriangleDeformation.cpp

   \brief file for computing 2D deformations between 2 images
   thanks to triangles.
**/

namespace MMVII
{
    /****************************************/
    /*                                      */
    /*         cPtInsideTriangles           */
    /*                                      */
    /****************************************/

    cPtInsideTriangles::cPtInsideTriangles(const cTriangle2DCompiled<tREAL8> &aCompTri,              // a compiled triangle
                                           const std::vector<cPt2di> &aVectorFilledwithInsidePixels, // vector containing pixels insisde triangles
                                           const size_t aFilledPixel,                                // a counter that is looping over pixels in triangles
                                           const cDataIm2D<tREAL8> &aDIm)                            // image
    {
        mFilledIndices = cPt2dr(aVectorFilledwithInsidePixels[aFilledPixel].x(), aVectorFilledwithInsidePixels[aFilledPixel].y());
        mBarycenterCoordinatesOfPixel = aCompTri.CoordBarry(mFilledIndices);
        mValueOfPixel = aDIm.GetV(cPt2di(mFilledIndices.x(), mFilledIndices.y()));
    }

    cPt3dr cPtInsideTriangles::GetBarycenterCoordinates() const { return mBarycenterCoordinatesOfPixel; } // Accessor
    cPt2dr cPtInsideTriangles::GetCartesianCoordinates() const { return mFilledIndices; }                 // Accessor
    tREAL8 cPtInsideTriangles::GetPixelValue() const { return mValueOfPixel; }                            // Accessor

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
                                                                                              mSigmaGaussFilterStep(1),
                                                                                              mGenerateDisplacementImage(true),
                                                                                              mFreezeRadTranslation(false),
                                                                                              mFreezeRadScale(false),
                                                                                              mWeightRadTranslation(-1),
                                                                                              mWeightRadScale(-1),
                                                                                              mNumberOfIterGaussFilter(3),
                                                                                              mNumberOfEndIterations(2),
                                                                                              mDisplayLastTranslationValues(false),
                                                                                              mSzImPre(cPt2di(1, 1)),
                                                                                              mImPre(mSzImPre),
                                                                                              mDImPre(nullptr),
                                                                                              mSzImPost(cPt2di(1, 1)),
                                                                                              mImPost(mSzImPost),
                                                                                              mDImPost(nullptr),
                                                                                              mSzImOutTr(cPt2di(1, 1)),
                                                                                              mImOutTr(mSzImOutTr),
                                                                                              mDImOutTr(nullptr),
                                                                                              mSzImOutRad(cPt2di(1, 1)),
                                                                                              mImOutRad(mSzImOutTr),
                                                                                              mDImOutRad(nullptr),
                                                                                              mSzImDiff(cPt2di(1, 1)),
                                                                                              mImDiff(mSzImDiff),
                                                                                              mDImDiff(nullptr),
                                                                                              mSzImDepX(cPt2di(1, 1)),
                                                                                              mImDepX(mSzImDepX),
                                                                                              mDImDepX(nullptr),
                                                                                              mSzImDepY(cPt2di(1, 1)),
                                                                                              mImDepY(mSzImDepY),
                                                                                              mDImDepY(nullptr),
                                                                                              mVectorPts({cPt2dr(0, 0)}),
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
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach optimisation process.");
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
               << AOpt2007(mSigmaGaussFilterStep, "SigmaGaussFilterStep", "Sigma value to use for Gauss filter in multi-stage approach.", {eTA2007::HDV})
               << AOpt2007(mGenerateDisplacementImage, "GenerateDisplacementImage",
                           "Whether to generate and save an image having been translated.", {eTA2007::HDV})
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

    void cAppli_cTriangleDeformation::ConstructUniformRandomVectorAndApplyDelaunay()
    {
        // Use current time as seed for random generator
        // srand(time(0));

        mVectorPts.pop_back(); // eliminate initialisation values
        // Generate coordinates from drawing lines and columns of coordinates from a uniform distribution
        for (int aNbPt = 0; aNbPt < mNumberPointsToGenerate; aNbPt++)
        {
            const tREAL8 aUniformRandomLine = RandUnif_N(mRandomUniformLawUpperBoundLines);
            const tREAL8 aUniformRandomCol = RandUnif_N(mRandomUniformLawUpperBoundCols);
            const cPt2dr aUniformRandomPt(aUniformRandomCol, aUniformRandomLine); // cPt2dr format
            mVectorPts.push_back(aUniformRandomPt);
        }
        mDelTri = mVectorPts;

        mDelTri.MakeDelaunay(); // Delaunay triangulate randomly generated points.
    }

    void cAppli_cTriangleDeformation::GeneratePointsForDelaunay()
    {
        // If user hasn't defined another value than the default value, it is changed
        if (mRandomUniformLawUpperBoundLines == 1 && mRandomUniformLawUpperBoundCols == 1)
        {
            // Maximum value of coordinates are drawn from [0, NumberOfImageLines[ for lines
            mRandomUniformLawUpperBoundLines = mSzImPre.y();
            // Maximum value of coordinates are drawn from [0, NumberOfImageColumns[ for columns
            mRandomUniformLawUpperBoundCols = mSzImPre.x();
        }
        else
        {
            if (mRandomUniformLawUpperBoundLines != 1 && mRandomUniformLawUpperBoundCols == 1)
                mRandomUniformLawUpperBoundCols = mSzImPre.x();
            else
            {
                if (mRandomUniformLawUpperBoundLines == 1 && mRandomUniformLawUpperBoundCols != 1)
                    mRandomUniformLawUpperBoundLines = mSzImPre.y();
            }
        }

        ConstructUniformRandomVectorAndApplyDelaunay();
    }

    void cAppli_cTriangleDeformation::InitialisationAfterExe()
    {
        tDenseVect aVInit(4 * mDelTri.NbPts(), eModeInitImage::eMIA_Null);

        for (size_t aKtNumber = 0; aKtNumber < 4 * mDelTri.NbPts(); aKtNumber++)
        {
            if (aKtNumber % 4 == 3)
                aVInit(aKtNumber) = 1;
        }

        mSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInit);
    }

    void cAppli_cTriangleDeformation::SubtractPrePostImageAndComputeAvgAndMax()
    {
        mImDiff = tIm(mSzImPre);
        mDImDiff = &mImDiff.DIm();

        for (const cPt2di &aDiffPix : *mDImDiff)
            mDImDiff->SetV(aDiffPix, mDImPre->GetV(aDiffPix) - mDImPost->GetV(aDiffPix));
        const int aNumberOfPixelsInImage = mSzImPre.x() * mSzImPre.y();

        tREAL8 aSumPixelValuesInDiffImage = 0;
        tREAL8 aMaxPixelValuesInDiffImage = 0;
        tREAL8 aDiffImPixelValue = 0;
        for (const cPt2di &aDiffPix : *mDImDiff)
        {
            aDiffImPixelValue = mDImDiff->GetV(aDiffPix);
            aSumPixelValuesInDiffImage += aDiffImPixelValue;
            if (aDiffImPixelValue > aMaxPixelValuesInDiffImage)
                aMaxPixelValuesInDiffImage = aDiffImPixelValue;
        }
        StdOut() << "The average value of the difference image between the Pre and Post images is : "
                 << aSumPixelValuesInDiffImage / (tREAL8)aNumberOfPixelsInImage << std::endl;
        StdOut() << "The maximum value of the difference image between the Pre and Post images is : "
                 << aMaxPixelValuesInDiffImage << std::endl;
    }

    cPt2dr cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaToFilledPixel(const cPt2dr &aCurrentTranslationPointA,
                                                                                       const cPt2dr &aCurrentTranslationPointB,
                                                                                       const cPt2dr &aCurrentTranslationPointC,
                                                                                       const tDoubleVect &aVObs)
    {
        // apply current barycenter translation formula for x and y on current observations.
        const tREAL8 aXTriCoord = aVObs[0] + aVObs[2] * aCurrentTranslationPointA.x() + aVObs[3] * aCurrentTranslationPointB.x() +
                                  aVObs[4] * aCurrentTranslationPointC.x();
        const tREAL8 aYTriCoord = aVObs[1] + aVObs[2] * aCurrentTranslationPointA.y() + aVObs[3] * aCurrentTranslationPointB.y() +
                                  aVObs[4] * aCurrentTranslationPointC.y();

        const cPt2dr aCurrentTranslatedPixel = cPt2dr(aXTriCoord, aYTriCoord);

        return aCurrentTranslatedPixel;
    }

    tREAL8 cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                                                  const tREAL8 aCurrentRadTranslationPointB,
                                                                                                  const tREAL8 aCurrentRadTranslationPointC,
                                                                                                  const tDoubleVect &aVObs)
    {
        const tREAL8 aCurentRadTranslation = aVObs[2] * aCurrentRadTranslationPointA + aVObs[3] * aCurrentRadTranslationPointB +
                                             aVObs[4] * aCurrentRadTranslationPointC;
        return aCurentRadTranslation;
    }

    tREAL8 cAppli_cTriangleDeformation::ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                                              const tREAL8 aCurrentRadScalingPointB,
                                                                                              const tREAL8 aCurrentRadScalingPointC,
                                                                                              const tDoubleVect &aVObs)
    {
        const tREAL8 aCurrentRadScaling = aVObs[2] * aCurrentRadScalingPointA + aVObs[3] * aCurrentRadScalingPointB +
                                          aVObs[4] * aCurrentRadScalingPointC;
        return aCurrentRadScaling;
    }

    void cAppli_cTriangleDeformation::LoadImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage)
    {
        (aPreOrPostImage == "pre") ? aCurIm = mImPre : aCurIm = mImPost;
        aCurDIm = &aCurIm.DIm();
    }

    void cAppli_cTriangleDeformation::ManageDifferentCasesOfEndIterations(const int aIterNumber, tIm aCurPreIm, tDIm *aCurPreDIm,
                                                                          tIm aCurPostIm, tDIm *aCurPostDIm)
    {
        switch (mNumberOfEndIterations)
        {
        case 1: // one last iteration
            if (aIterNumber == mNumberOfScales)
            {
                mIsLastIters = true;
                LoadImageAndData(aCurPreIm, aCurPreDIm, "pre");
                LoadImageAndData(aCurPostIm, aCurPostDIm, "post");
            }
            break;
        case 2: // two last iterations
            if ((aIterNumber == mNumberOfScales) || (aIterNumber == mNumberOfScales + mNumberOfEndIterations - 1))
            {
                mIsLastIters = true;
                LoadImageAndData(aCurPreIm, aCurPreDIm, "pre");
                LoadImageAndData(aCurPostIm, aCurPostDIm, "post");
            }
            break;
        case 3: //  three last iterations
            if ((aIterNumber == mNumberOfScales) || (aIterNumber == mNumberOfScales + mNumberOfEndIterations - 2) ||
                (aIterNumber == mNumberOfScales + mNumberOfEndIterations - 1))
            {
                mIsLastIters = true;
                LoadImageAndData(aCurPreIm, aCurPreDIm, "pre");
                LoadImageAndData(aCurPostIm, aCurPostDIm, "post");
            }
            break;
        default: // default is two last iterations
            if ((aIterNumber == mNumberOfScales) || (aIterNumber == mNumberOfScales + mNumberOfEndIterations - 1))
            {
                mIsLastIters = true;
                LoadImageAndData(aCurPreIm, aCurPreDIm, "pre");
                LoadImageAndData(aCurPostIm, aCurPostDIm, "post");
            }
            break;
        }
    }

    void cAppli_cTriangleDeformation::LoopOverTrianglesAndUpdateParameters(const int aIterNumber)
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
            ManageDifferentCasesOfEndIterations(aIterNumber, aCurPreIm, aCurPreDIm,
                                                aCurPostIm, aCurPostDIm);
        else
        {
            LoadImageAndData(aCurPreIm, aCurPreDIm, "pre");
            LoadImageAndData(aCurPostIm, aCurPostDIm, "post");
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

        //----------- declaration of indicator of convergence
        tREAL8 aSomDif = 0; // sum of difference between untranslated pixel and translated one.
        size_t aNbOut = 0;  // number of translated pixels out of image

        // Count number of pixels inside triangles for normalisation
        size_t aTotalNumberOfInsidePixels = 0;

        // hard constraint : freeze radiometric coefficients
        if (mFreezeRadTranslation || mFreezeRadScale)
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
                const cPt2dr aTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aCurTrPointA, aCurTrPointB,
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
            const bool aGenerateIntermediateMaps = false;
            if (aGenerateIntermediateMaps)
                GenerateDisplacementMaps(aVCur, aIterNumber);
        }

        // Update all parameter taking into account previous observation
        mSys->SolveUpdateReset();

        if (mShow)
            StdOut() << aIterNumber + 1 << ", " << aSomDif / aTotalNumberOfInsidePixels
                     << ", " << aNbOut << std::endl;
    }

    void cAppli_cTriangleDeformation::FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                                                         const cPt2dr &aLastTranslatedFilledPoint,
                                                                         const tREAL8 aLastRadiometryTranslation,
                                                                         const tREAL8 aLastRadiometryScaling)
    {
        const tREAL8 aLastXCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().x();
        const tREAL8 aLastYCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().y();
        const tREAL8 aLastPixelValue = aLastPixInsideTriangle.GetPixelValue();

        const cPt2di aLastCoordinate = cPt2di(aLastXCoordinate, aLastYCoordinate);
        mDImDepX->SetV(aLastCoordinate,
                       aLastTranslatedFilledPoint.x() - aLastXCoordinate);
        mDImDepY->SetV(aLastCoordinate,
                       aLastTranslatedFilledPoint.y() - aLastYCoordinate);
        const tREAL8 aLastXTranslatedCoord = aLastXCoordinate + mDImDepX->GetV(aLastCoordinate);
        const tREAL8 aLastYTranslatedCoord = aLastYCoordinate + mDImDepY->GetV(aLastCoordinate);

        const tREAL8 aLastRadiometryValue = aLastRadiometryScaling * aLastPixelValue +
                                            aLastRadiometryTranslation;
        mDImOutRad->SetV(aLastCoordinate, aLastRadiometryValue);

        // Build image with intensities displaced
        // deal with different cases of pixel being translated out of image
        if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord < 0)
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(0, 0)));
        else if (aLastXTranslatedCoord >= mSzImOutTr.x() && aLastYTranslatedCoord >= mSzImOutTr.y())
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(mSzImOutTr.x() - 1, mSzImOutTr.y() - 1)));
        else if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord >= mSzImOutTr.y())
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(0, mSzImOutTr.y() - 1)));
        else if (aLastXTranslatedCoord >= mSzImOutTr.x() && aLastYTranslatedCoord < 0)
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(mSzImOutTr.x() - 1, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < mSzImOutTr.x() &&
                 aLastYTranslatedCoord < 0)
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(aLastXTranslatedCoord, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < mSzImOutTr.x() &&
                 aLastYTranslatedCoord > mSzImOutTr.y())
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(aLastXTranslatedCoord, mSzImOutTr.y() - 1)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < mSzImOutTr.y() &&
                 aLastXTranslatedCoord < 0)
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(0, aLastYTranslatedCoord)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < mSzImOutTr.y() &&
                 aLastXTranslatedCoord > mSzImOutTr.x())
            mDImOutTr->SetV(aLastCoordinate, mDImOutTr->GetV(cPt2di(mSzImOutTr.x() - 1, aLastYTranslatedCoord)));
        else
            // at the translated pixel the untranslated pixel value is given computed with the right radiometry values
            mDImOutTr->SetV(cPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord), aLastPixelValue);
    }

    void cAppli_cTriangleDeformation::GenerateDisplacementMaps(const tDenseVect &aVFinalSol, const int aIterNumber)
    {
        mImOutTr = tIm(mSzImPre);
        mDImOutTr = &mImOutTr.DIm();
        mSzImOutTr = cPt2di(mDImOutTr->Sz().x(), mDImOutTr->Sz().y());

        mImOutRad = tIm(mSzImPre);
        mDImOutRad = &mImOutRad.DIm();
        mSzImOutRad = cPt2di(mDImOutRad->Sz().x(), mDImOutRad->Sz().y());

        mImDepX = tIm(mSzImPre, 0, eModeInitImage::eMIA_Null);
        mDImDepX = &mImDepX.DIm();

        mImDepY = tIm(mSzImPre, 0, eModeInitImage::eMIA_Null);
        mDImDepY = &mImDepY.DIm();

        tIm aLastPreIm = tIm(mSzImPre);
        tDIm *aLastPreDIm = nullptr;
        LoadImageAndData(aLastPreIm, aLastPreDIm, "pre");

        if (mUseMultiScaleApproach && !mIsLastIters)
        {
            aLastPreIm = mImPre.GaussFilter(mSigmaGaussFilter, mNumberOfIterGaussFilter);
            aLastPreDIm = &aLastPreIm.DIm();
        }

        tDoubleVect aLastVObs(12, 0.0);

        for (const cPt2di &aOutPix : *mDImOutTr) // Initialise output images
        {
            mDImOutTr->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));
            mDImOutRad->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));
        }

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
                                                                                     aLastFilledPixel, *aLastPreDIm);
                // prepare for barycenter translation formula by filling aVObs with different coordinates
                FormalInterpBarycenter_SetObs(aLastVObs, 0, aLastPixInsideTriangle);

                // image of a point in triangle by current translation
                const cPt2dr aLastTranslatedFilledPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aLastTrPointA, aLastTrPointB,
                                                                                                         aLastTrPointC, aLastVObs);

                const tREAL8 aLastRadiometryTranslation = ApplyBarycenterTranslationFormulaForTranslationRadiometry(aLastRadTrPointA,
                                                                                                                    aLastRadTrPointB,
                                                                                                                    aLastRadTrPointC,
                                                                                                                    aLastVObs);

                const tREAL8 aLastRadiometryScaling = ApplyBarycenterTranslationFormulaForScalingRadiometry(aLastRadScPointA,
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
            {
                mDImOutTr->ToFile("DisplacedPixels_iter_" + std::to_string(aIterNumber) + "_" +
                                  std::to_string(mNumberPointsToGenerate) + "_" +
                                  std::to_string(mNumberOfScales + mNumberOfEndIterations) + ".tif");
                mDImOutRad->ToFile("OutputImage_iter_" + std::to_string(aIterNumber) + "_" +
                                   std::to_string(mNumberPointsToGenerate) + "_" +
                                   std::to_string(mNumberOfScales + mNumberOfEndIterations) + ".tif");
            }
        }
        else
        {
            mDImDepX->ToFile("DisplacedPixelsX_" + std::to_string(mNumberPointsToGenerate) + "_" +
                             std::to_string(mNumberOfScales) + ".tif");
            mDImDepY->ToFile("DisplacedPixelsY_" + std::to_string(mNumberPointsToGenerate) + "_" +
                             std::to_string(mNumberOfScales) + ".tif");
            mDImOutTr->ToFile("DisplacedPixels_" + std::to_string(mNumberPointsToGenerate) + "_" +
                              std::to_string(mNumberOfScales) + ".tif");
            mDImOutRad->ToFile("OutputImage_" + std::to_string(mNumberPointsToGenerate) + "_" +
                               std::to_string(mNumberOfScales) + ".tif");
        }
    }

    void cAppli_cTriangleDeformation::GenerateDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber)
    {
        tDenseVect aVFinalSol = mSys->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMaps(aVFinalSol, aIterNumber);

        if (mDisplayLastRadiometryValues && mDisplayLastTranslationValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        if (mDisplayLastTranslationValues && !mDisplayLastRadiometryValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                if (aFinalUnk % 4 == 0 || aFinalUnk % 4 == 1)
                    StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        else if (mDisplayLastRadiometryValues && !mDisplayLastTranslationValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                if (aFinalUnk % 4 == 2 || aFinalUnk % 4 == 3)
                    StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
    }

    void cAppli_cTriangleDeformation::DoOneIteration(const int aIterNumber)
    {
        LoopOverTrianglesAndUpdateParameters(aIterNumber); // Iterate over triangles and solve system

        // Show final translation results and produce displacement maps
        if (mUseMultiScaleApproach)
        {
            if (aIterNumber == (mNumberOfScales + mNumberOfEndIterations - 1))
                GenerateDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber);
        }
        else
        {
            if (aIterNumber == (mNumberOfScales - 1))
                GenerateDisplacementMapsAndDisplayLastValuesUnknowns(aIterNumber);
        }
    }

    //-----------------------------------------

    int cAppli_cTriangleDeformation::Exe()
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

        if (mComputeAvgMax)
            SubtractPrePostImageAndComputeAvgAndMax();

        if (mShow)
            StdOut() << "Iter, "
                     << "Diff, "
                     << "NbOut" << std::endl;

        GeneratePointsForDelaunay();

        InitialisationAfterExe();

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