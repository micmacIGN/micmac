#include "ComputeNodesPrecision.h"

/**
   \file ComputeNodesPrecision.cpp

   \brief file for computing finding the best compromise between
    how many nodes to use in triangulation and the possible displacement precision
**/

namespace MMVII
{
    /************************************************/
    /*                                              */
    /*          cAppli_ComputeNodesPrecision        */
    /*                                              */
    /************************************************/

    cAppli_ComputeNodesPrecision::cAppli_ComputeNodesPrecision(const std::vector<std::string> &aVArgs,
                                                               const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
                                                                                                mNumberOfLines(1),
                                                                                                mNumberOfCols(1),
                                                                                                mBuildRandomUniformGrid(false),
                                                                                                mComputeDiffDispMaps(true),
                                                                                                mComputeInterpTranslationDispMaps(true),
                                                                                                mNameDiffDispX("DiffComputedDispX"),
                                                                                                mNameDiffDispY("DiffComputedDispY"),
                                                                                                mNameComputedTranslatedDispX("ComputedTranslationDispX"),
                                                                                                mNameComputedTranslatedDispY("ComputedTranslationDispY"),
                                                                                                mSzImDispX(tPt2di(1, 1)),
                                                                                                mImDispX(mSzImDispX),
                                                                                                mDImDispX(nullptr),
                                                                                                mSzImDispY(tPt2di(1, 1)),
                                                                                                mImDispY(mSzImDispY),
                                                                                                mDImDispY(nullptr),
                                                                                                mSzImDiffDispX(tPt2di(1, 1)),
                                                                                                mImDiffDispX(mSzImDiffDispX),
                                                                                                mDImDiffDispX(nullptr),
                                                                                                mSzImDiffDispY(tPt2di(1, 1)),
                                                                                                mImDiffDispY(mSzImDiffDispY),
                                                                                                mDImDiffDispY(nullptr),
                                                                                                mSzImTranslatedDispX(tPt2di(1, 1)),
                                                                                                mImTranslatedDispX(mSzImTranslatedDispX),
                                                                                                mDImTranslatedDispX(nullptr),
                                                                                                mSzImTranslatedDispY(tPt2di(1, 1)),
                                                                                                mImTranslatedDispY(mSzImTranslatedDispY),
                                                                                                mDImTranslatedDispY(nullptr),
                                                                                                mDelTri({tPt2dr(0, 0)})
    {
    }

    cAppli_ComputeNodesPrecision::~cAppli_ComputeNodesPrecision()
    {
    }

    cCollecSpecArg2007 &cAppli_ComputeNodesPrecision::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
        return anArgObl
               << Arg2007(mNameDispXMap, "Name of x-displacement ground-truth file.", {eTA2007::FileImage, eTA2007::FileDirProj})
               << Arg2007(mNameDispYMap, "Name of y-displacement ground-truth file.", {eTA2007::FileImage})
               << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.");
    }

    cCollecSpecArg2007 &cAppli_ComputeNodesPrecision::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
        return anArgOpt
               << AOpt2007(mNumberOfLines, "MaximumValueNumberOfLines",
                           "Maximum value that the uniform law can draw from for on the y-axis.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mNumberOfCols, "MaximumValueNumberOfCols",
                           "Maximum value that the uniform law can draw from on the x-axis.", {eTA2007::HDV, eTA2007::Tuning})
               << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
                           "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
               << AOpt2007(mComputeDiffDispMaps, "ComputeDiffDisplacementMaps",
                           "Whether to compute difference displacement maps or not.", {eTA2007::HDV})
               << AOpt2007(mComputeInterpTranslationDispMaps, "ComputeInterpTranslationDispMaps",
                           "Whether to compute displacement maps containing translations by barycentric interpolation formula or not.", {eTA2007::HDV})
               << AOpt2007(mNameDiffDispX, "NameXDisplacementDiffMap",
                           "File name to use to save the difference x-displacement map.", {eTA2007::HDV})
               << AOpt2007(mNameDiffDispY, "NameYDisplacementDiffMap",
                           "File name to use to save the difference y-displacement map.", {eTA2007::HDV})
               << AOpt2007(mNameComputedTranslatedDispX, "NameXTranslatedDispMap",
                           "File name to use to save the x-translated displacement map.", {eTA2007::HDV})
               << AOpt2007(mNameComputedTranslatedDispY, "NameYTranslatedDispMap",
                           "File name to use to save the y-translated displacement map.", {eTA2007::HDV});
    }

    void cAppli_ComputeNodesPrecision::LoopOverTrianglesAndGetDiffDispMaps()
    {
        if (mComputeDiffDispMaps)
        {
            InitialiseDisplacementMaps(mSzImDispX, mImDiffDispX, mDImDiffDispX, mSzImDiffDispX);
            InitialiseDisplacementMaps(mSzImDispY, mImDiffDispY, mDImDiffDispY, mSzImDiffDispY);
        }

        if (mComputeInterpTranslationDispMaps)
        {
            InitialiseDisplacementMaps(mSzImDispX, mImTranslatedDispX, mDImTranslatedDispX, mSzImTranslatedDispX);
            InitialiseDisplacementMaps(mSzImDispY, mImTranslatedDispY, mDImTranslatedDispY, mSzImTranslatedDispY);
        }

        for (size_t aTr = 0; aTr < mDelTri.NbFace(); aTr++)
        {
            const tTri2dr aTri = mDelTri.KthTri(aTr);

            const cTriangle2DCompiled aLastCompTri(aTri);

            std::vector<tPt2di> aLastVectorToFillWithInsidePixels;
            aLastCompTri.PixelsInside(aLastVectorToFillWithInsidePixels);

            const tPt2dr aRealCoordPointA = aTri.Pt(0); // Get real value coordinates 1st point of triangle
            const tPt2dr aRealCoordPointB = aTri.Pt(1); // Get real value coordinates 2nd point of triangle
            const tPt2dr aRealCoordPointC = aTri.Pt(2); // Get real value coordinates 3rd point of triangle

            const tPt2di aIntCoordPointA = tPt2di(aRealCoordPointA.x(), aRealCoordPointA.y()); // Get integer value coordinates 1st point of triangle
            const tPt2di aIntCoordPointB = tPt2di(aRealCoordPointB.x(), aRealCoordPointB.y()); // Get integer value coordinates 2nd point of triangle
            const tPt2di aIntCoordPointC = tPt2di(aRealCoordPointC.x(), aRealCoordPointC.y()); // Get integer value coordinates 3rd point of triangle

            const tPt2dr aTrPointA = CheckReturnOfBilinearValue(mDImDispX, mDImDispY, aRealCoordPointA, aIntCoordPointA); // Get value value from real or int coordinates 1st point
            const tPt2dr aTrPointB = CheckReturnOfBilinearValue(mDImDispX, mDImDispY, aRealCoordPointB, aIntCoordPointB); // Get value value from real or int coordinates 2nd point
            const tPt2dr aTrPointC = CheckReturnOfBilinearValue(mDImDispX, mDImDispY, aRealCoordPointC, aIntCoordPointC); // Get value value from real or int coordinates 3rd point

            const size_t aLastNumberOfInsidePixels = aLastVectorToFillWithInsidePixels.size();

            for (size_t aLastFilledPixel = 0; aLastFilledPixel < aLastNumberOfInsidePixels; aLastFilledPixel++)
            {
                const cPtInsideTriangles aPixInsideTriangleDispX = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                      aLastFilledPixel, mDImDispX);
                const cPtInsideTriangles aPixInsideTriangleDispY = cPtInsideTriangles(aLastCompTri, aLastVectorToFillWithInsidePixels,
                                                                                      aLastFilledPixel, mDImDispY);

                // apply barycentric interpolation formula
                const tPt2dr aTranslatedDispPoint = ApplyBarycenterTranslationFormulaToFilledPixel(aTrPointA, aTrPointB,
                                                                                                   aTrPointC, aPixInsideTriangleDispX);

                FillDiffDisplacementMap(mDImDispX, mDImDiffDispX, mDImTranslatedDispX, aPixInsideTriangleDispX,
                                        aPixInsideTriangleDispX.GetCartesianCoordinates().x(), aTranslatedDispPoint.x(),
                                        mComputeDiffDispMaps, mComputeInterpTranslationDispMaps);
                FillDiffDisplacementMap(mDImDispY, mDImDiffDispY, mDImTranslatedDispY, aPixInsideTriangleDispY,
                                        aPixInsideTriangleDispY.GetCartesianCoordinates().y(), aTranslatedDispPoint.y(),
                                        mComputeDiffDispMaps, mComputeInterpTranslationDispMaps);
                // aTranslatedDispPoint has same translated coordinates as aTranslatedDispYPoint, the computation of formula doesn't need to be done twice
            }
        }

        if (mComputeDiffDispMaps)
        {
            mDImDiffDispX->ToFile(mNameDiffDispX + "_" + ToStr(mNumberPointsToGenerate) + ".tif");
            mDImDiffDispX->ToFile(mNameDiffDispY + "_" + ToStr(mNumberPointsToGenerate) + ".tif");
        }
        if (mComputeInterpTranslationDispMaps)
        {
            mDImTranslatedDispX->ToFile(mNameComputedTranslatedDispX + "_" + ToStr(mNumberPointsToGenerate) + ".tif");
            mDImTranslatedDispY->ToFile(mNameComputedTranslatedDispY + "_" + ToStr(mNumberPointsToGenerate) + ".tif");
        }
    }

    int cAppli_ComputeNodesPrecision::Exe()
    {
        // read pre and post images and update their sizes
        ReadFileNameLoadData(mNameDispXMap, mImDispX, mDImDispX, mSzImDispX);
        ReadFileNameLoadData(mNameDispYMap, mImDispY, mDImDispY, mSzImDispY);

        // Build uniform or rectangular grid
        DefineValueLimitsForPointGenerationAndBuildGrid(mNumberPointsToGenerate, mNumberOfLines,
                                                        mNumberOfCols, mDelTri, mSzImDispX, mBuildRandomUniformGrid);

        LoopOverTrianglesAndGetDiffDispMaps();

        return EXIT_SUCCESS;
    }

    /********************************************/
    //              ::MMVII                     //
    /********************************************/

    tMMVII_UnikPApli Alloc_cAppli_ComputeNodesPrecision(const std::vector<std::string> &aVArgs,
                                                        const cSpecMMVII_Appli &aSpec)
    {
        return tMMVII_UnikPApli(new cAppli_ComputeNodesPrecision(aVArgs, aSpec));
    }

    cSpecMMVII_Appli TheSpec_ComputeNodesPrecision(
        "ComputeNodesPrecision",
        Alloc_cAppli_ComputeNodesPrecision,
        "Compute compromise between number of nodes used in triangulation and obtained precision",
        {eApF::ImProc}, // category
        {eApDT::Image}, // input
        {eApDT::Image}, // output
        __FILE__);

}; // MMVII