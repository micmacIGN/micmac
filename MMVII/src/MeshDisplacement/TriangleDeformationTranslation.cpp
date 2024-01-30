#include "cMMVII_Appli.h"

#include "MMVII_TplSymbTriangle.h"

#include "TriangleDeformationTranslation.h"

/**
   \file TriangleDeformationTranslation.cpp

   \brief file for computing 2D translation between 2 images
   thanks to triangles.
**/

namespace MMVII
{

    /******************************************/
    /*                                        */
    /*    cTriangleDeformationTranslation     */
    /*                                        */
    /******************************************/

    cAppli_cTriangleDeformationTranslation::cAppli_cTriangleDeformationTranslation(const std::vector<std::string> &aVArgs,
                                                                                   const cSpecMMVII_Appli &aSpec) : cAppli_cTriangleDeformation(aVArgs, aSpec),
                                                                                                                    mRandomUniformLawUpperBoundLines(1),
                                                                                                                    mRandomUniformLawUpperBoundCols(1),
                                                                                                                    mShow(true),
                                                                                                                    mUseMultiScaleApproach(true),
                                                                                                                    mSigmaGaussFilterStep(1),
                                                                                                                    mGenerateDisplacementImage(true),
                                                                                                                    mNumberOfIterGaussFilter(3),
                                                                                                                    mNumberOfEndIterations(2),
                                                                                                                    mDisplayLastTranslatedPointsCoordinates(false),
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
                                                                                                                    mEqTranslationTri(nullptr)
    {
        mEqTranslationTri = EqDeformTriTranslation(true, 1); // true means with derivative, 1 is size of buffer
        // mEqTranslationTri->SetDebugEnabled(true);
    }

    cAppli_cTriangleDeformationTranslation::~cAppli_cTriangleDeformationTranslation()
    {
        delete mSys;
        delete mEqTranslationTri;
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformationTranslation::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNamePreImage, "Name of pre-image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               << Arg2007(mNamePostImage, "Name of post-image file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.")
               << Arg2007(mNumberOfScales, "Total number of scales to run in multi-scale approach optimisation process.");
    }

    cCollecSpecArg2007 &cAppli_cTriangleDeformationTranslation::ArgOpt(cCollecSpecArg2007 &anArgOpt)
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
               << AOpt2007(mDisplayLastTranslatedPointsCoordinates, "DisplayLastTranslatedCoordinates",
                           "Whether to display the final coordinates of the trainslated points.", {eTA2007::HDV})    
               << AOpt2007(mNumberOfIterGaussFilter, "NumberOfIterationsGaussFilter",
                           "Number of iterations to run in Gauss filter algorithm.", {eTA2007::HDV})
               << AOpt2007(mNumberOfEndIterations, "NumberOfEndIterations",
                           "Number of iterations to run on original images in multi-scale approach.", {eTA2007::HDV});
    }

    void cAppli_cTriangleDeformationTranslation::InitialisationAfterExe()
    {
        tDenseVect aVInit(2 * mDelTri.NbPts(), eModeInitImage::eMIA_Null);

        mSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInit);
    }

    void cAppli_cTriangleDeformationTranslation::ConstructUniformRandomVectorAndApplyDelaunay()
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

    void cAppli_cTriangleDeformationTranslation::GeneratePointsForDelaunay()
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

    void cAppli_cTriangleDeformationTranslation::LoadImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage)
    {
        (aPreOrPostImage == "pre") ? aCurIm = mImPre : aCurIm = mImPost;
        aCurDIm = &aCurIm.DIm();
    }

    void cAppli_cTriangleDeformationTranslation::ManageDifferentCasesOfEndIterations(const int aIterNumber, tIm aCurPreIm, tDIm * aCurPreDIm,
                                                                                     tIm aCurPostIm, tDIm * aCurPostDIm)
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

    void cAppli_cTriangleDeformationTranslation::LoopOverTrianglesAndUpdateParameters(const int aIterNumber)
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

        // Loop over all triangles to add the observations on each point
        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTri = mDelTri.KthTri(aTr);
            const cPt3di aIndicesOfTriKnots = mDelTri.KthFace(aTr);

            const cTriangle2DCompiled aCompTri(aTri);

            std::vector<cPt2di> aVectorToFillWithInsidePixels;
            aCompTri.PixelsInside(aVectorToFillWithInsidePixels); // get pixels inside triangle

            //----------- index of unknown, finds the associated pixels of current triangle
            const tIntVect aVecInd = {2 * aIndicesOfTriKnots.x(), 2 * aIndicesOfTriKnots.x() + 1,
                                      2 * aIndicesOfTriKnots.y(), 2 * aIndicesOfTriKnots.y() + 1,
                                      2 * aIndicesOfTriKnots.z(), 2 * aIndicesOfTriKnots.z() + 1};

            const cPt2dr aCurTrPointA = cPt2dr(aVCur(aVecInd.at(0)),
                                               aVCur(aVecInd.at(1))); // current translation 1st point of triangle
            const cPt2dr aCurTrPointB = cPt2dr(aVCur(aVecInd.at(2)),
                                               aVCur(aVecInd.at(3))); // current translation 2nd point of triangle
            const cPt2dr aCurTrPointC = cPt2dr(aVCur(aVecInd.at(4)),
                                               aVCur(aVecInd.at(5))); // current translation 3rd point of triangle

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

                if (aCurPostDIm->InsideBL(aTranslatedFilledPoint)) // avoid errors
                {
                    // prepare for application of bilinear formula
                    FormalBilinTri_SetObs(aVObs, TriangleDisplacement_NbObs, aTranslatedFilledPoint, *aCurPostDIm);

                    // Now add observation
                    mSys->CalcAndAddObs(mEqTranslationTri, aVecInd, aVObs);

                    // compute indicators
                    const tREAL8 aDif = aVObs[5] - aCurPostDIm->GetVBL(aTranslatedFilledPoint); // residual - aValueImPre - aCurPostDIm->GetVBL(aTranslatedFilledPoint)
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

    void cAppli_cTriangleDeformationTranslation::FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                                                                    const cPt2dr &aLastTranslatedFilledPoint)
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
            // at the translated pixel the untranslated pixel value is given
            mDImOut->SetV(cPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord), aLastPixelValue);
    }

    void cAppli_cTriangleDeformationTranslation::GenerateDisplacementMaps(const tDenseVect &aVFinalSol, const int aIterNumber)
    {   
        mImOut = tIm(mSzImPre);
        mDImOut = &mImOut.DIm();
        mSzImOut = cPt2di(mDImOut->Sz().x(), mDImOut->Sz().y());

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

        for (const cPt2di &aOutPix : *mDImOut) // Initialise output image
            mDImOut->SetV(aOutPix, aLastPreDIm->GetV(aOutPix));

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

            const cPt2dr aLastTrPointA = cPt2dr(aVFinalSol(aLastVecInd.at(0)),
                                                aVFinalSol(aLastVecInd.at(1))); // last translation 1st point of triangle
            const cPt2dr aLastTrPointB = cPt2dr(aVFinalSol(aLastVecInd.at(2)),
                                                aVFinalSol(aLastVecInd.at(3))); // last translation 2nd point of triangle
            const cPt2dr aLastTrPointC = cPt2dr(aVFinalSol(aLastVecInd.at(4)),
                                                aVFinalSol(aLastVecInd.at(5))); // last translation 3rd point of triangle

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

                FillDisplacementMapsAndOutputImage(aLastPixInsideTriangle, aLastTranslatedFilledPoint);
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

    void cAppli_cTriangleDeformationTranslation::GenerateDisplacementMapsAndLastTranslatedPoints(const int aIterNumber)
    {
        tDenseVect aVFinalSol = mSys->CurGlobSol();

        if (mGenerateDisplacementImage)
            GenerateDisplacementMaps(aVFinalSol, aIterNumber);

        if (mDisplayLastTranslatedPointsCoordinates)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk %2 == 1 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
    }

    void cAppli_cTriangleDeformationTranslation::DoOneIterationTranslation(const int aIterNumber)
    {
        LoopOverTrianglesAndUpdateParameters(aIterNumber); // Iterate over triangles and solve system

        // Show final translation results and produce displacement maps
        if (mUseMultiScaleApproach)
        {
            if (aIterNumber == (mNumberOfScales + mNumberOfEndIterations - 1))
                GenerateDisplacementMapsAndLastTranslatedPoints(aIterNumber);
        }
        else
        {
            if (aIterNumber == (mNumberOfScales - 1))
                GenerateDisplacementMapsAndLastTranslatedPoints(aIterNumber);
        }
    }

    //-----------------------------------------

    int cAppli_cTriangleDeformationTranslation::Exe()
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

        GeneratePointsForDelaunay();

        InitialisationAfterExe();

        if (mUseMultiScaleApproach)
        {
            for (int aIterNumber = 0; aIterNumber < mNumberOfScales + mNumberOfEndIterations; aIterNumber++)
                DoOneIterationTranslation(aIterNumber);
        }
        else
        {
            for (int aIterNumber = 0; aIterNumber < mNumberOfScales; aIterNumber++)
                DoOneIterationTranslation(aIterNumber);
        }

        return EXIT_SUCCESS;
    }

    /********************************************/
    //              ::MMVII                     //
    /********************************************/

    tMMVII_UnikPApli Alloc_cTriangleDeformationTranslation(const std::vector<std::string> &aVArgs,
                                                           const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_cTriangleDeformationTranslation(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_ComputeTriangleDeformationTranslation(
        "ComputeTriangleDeformationTranslation",
        Alloc_cTriangleDeformationTranslation,
        "Compute 2D translation deformations of triangles between images using triangles",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // namespace MMVII