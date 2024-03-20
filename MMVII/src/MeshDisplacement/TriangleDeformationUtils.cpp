#include "TriangleDeformationUtils.h"

namespace MMVII
{
    /****************************************/
    /*                                      */
    /*          cPtInsideTriangles          */
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

    //---------------------------------------------//

    void ConstructUniformRandomVectorAndApplyDelaunay(std::vector<cPt2dr> &aVectorPts, const int aNumberOfPointsToGenerate,
                                                      const int aRandomUniformLawUpperBoundLines, const int aRandomUniformLawUpperBoundCols,
                                                      cTriangulation2D<tREAL8> &aDelaunayTri)
    {
        aVectorPts.pop_back(); // eliminate initialisation values
        // Generate coordinates from drawing lines and columns of coordinates from a uniform distribution
        for (int aNbPt = 0; aNbPt < aNumberOfPointsToGenerate; aNbPt++)
        {
            const tREAL8 aUniformRandomLine = RandUnif_N(aRandomUniformLawUpperBoundLines);
            const tREAL8 aUniformRandomCol = RandUnif_N(aRandomUniformLawUpperBoundCols);
            const cPt2dr aUniformRandomPt(aUniformRandomCol, aUniformRandomLine); // cPt2dr format
            aVectorPts.push_back(aUniformRandomPt);
        }
        aDelaunayTri = aVectorPts;

        aDelaunayTri.MakeDelaunay(); // Delaunay triangulate randomly generated points.
    }

    void GeneratePointsForDelaunay(std::vector<cPt2dr> &aVectorPts, const int aNumberOfPointsToGenerate,
                                   int aRandomUniformLawUpperBoundLines, int aRandomUniformLawUpperBoundCols,
                                   cTriangulation2D<tREAL8> &aDelaunayTri, const cPt2di &aSzImPre)
    {
        // If user hasn't defined another value than the default value, it is changed
        if (aRandomUniformLawUpperBoundLines == 1 && aRandomUniformLawUpperBoundCols == 1)
        {
            // Maximum value of coordinates are drawn from [0, NumberOfImageLines[ for lines
            aRandomUniformLawUpperBoundLines = aSzImPre.y();
            // Maximum value of coordinates are drawn from [0, NumberOfImageColumns[ for columns
            aRandomUniformLawUpperBoundCols = aSzImPre.x();
        }
        else
        {
            if (aRandomUniformLawUpperBoundLines != 1 && aRandomUniformLawUpperBoundCols == 1)
                aRandomUniformLawUpperBoundCols = aSzImPre.x();
            else
            {
                if (aRandomUniformLawUpperBoundLines == 1 && aRandomUniformLawUpperBoundCols != 1)
                    aRandomUniformLawUpperBoundLines = aSzImPre.y();
            }
        }

        ConstructUniformRandomVectorAndApplyDelaunay(aVectorPts, aNumberOfPointsToGenerate,
                                                     aRandomUniformLawUpperBoundLines, aRandomUniformLawUpperBoundCols,
                                                     aDelaunayTri);
    }

    void InitialisationAfterExe(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                cResolSysNonLinear<tREAL8> *&aSys)
    {
        const size_t aStartNumberPts = 4 * aDelaunayTri.NbPts();
        tDenseVect aVInit(aStartNumberPts, eModeInitImage::eMIA_Null);

        for (size_t aStartKtNumber = 0; aStartKtNumber < aStartNumberPts; aStartKtNumber++)
        {
            if (aStartKtNumber % 4 == 3)
                aVInit(aStartKtNumber) = 1;
        }

        aSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInit);
    }

    void InitialisationAfterExeTranslation(cTriangulation2D<tREAL8> &aDelaunayTri,
                                           cResolSysNonLinear<tREAL8> *&aSysTranslation)
    {
        tDenseVect aVInitTranslation(2 * aDelaunayTri.NbPts(), eModeInitImage::eMIA_Null);

        aSysTranslation = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInitTranslation);
    }

    void InitialisationAfterExeRadiometry(cTriangulation2D<tREAL8> &aDelaunayTri,
                                          cResolSysNonLinear<tREAL8> *&aSysRadiometry)
    {
        const size_t aNumberPts = 2 * aDelaunayTri.NbPts();
        tDenseVect aVInitRadiometry(aNumberPts, eModeInitImage::eMIA_Null); // eMIA_V1

        for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
        {
            if (aKtNumber % 2 == 1)
                aVInitRadiometry(aKtNumber) = 1;
        }

        aSysRadiometry = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInitRadiometry);
    }

    bool CheckValidCorrelationValue(tDIm * aMask, const cPt2di &aPointTri)
    {
        bool aIsValidCorrelPoint;
        (aMask->GetV(aPointTri) == 1) ? aIsValidCorrelPoint = true : aIsValidCorrelPoint = false;
        return aIsValidCorrelPoint;
    }

    tREAL8 ReturnCorrectInitialisationValue(const bool aIsValidCorrelation, tDIm *aIntermediateDispMap,
                                            const cPt2di &aTriPoint, const tREAL8 aValueToReturnIfFalse)
    {
        tREAL8 aInitialisationValue;
        (aIsValidCorrelation) ? aInitialisationValue = aIntermediateDispMap->GetV(aTriPoint) : aInitialisationValue = aValueToReturnIfFalse;
        return aInitialisationValue;
    }

    void SubtractPrePostImageAndComputeAvgAndMax(tIm aImDiff, tDIm *aDImDiff, tDIm *aDImPre,
                                                 tDIm *aDImPost, cPt2di aSzImPre)
    {
        aImDiff = tIm(aSzImPre);
        aDImDiff = &aImDiff.DIm();

        for (const cPt2di &aDiffPix : *aDImDiff)
            aDImDiff->SetV(aDiffPix, aDImPre->GetV(aDiffPix) - aDImPost->GetV(aDiffPix));
        const int aNumberOfPixelsInImage = aSzImPre.x() * aSzImPre.y();

        tREAL8 aSumPixelValuesInDiffImage = 0;
        tREAL8 aMaxPixelValuesInDiffImage = 0;
        tREAL8 aDiffImPixelValue = 0;
        for (const cPt2di &aDiffPix : *aDImDiff)
        {
            aDiffImPixelValue = aDImDiff->GetV(aDiffPix);
            aSumPixelValuesInDiffImage += aDiffImPixelValue;
            if (aDiffImPixelValue > aMaxPixelValuesInDiffImage)
                aMaxPixelValuesInDiffImage = aDiffImPixelValue;
        }
        StdOut() << "The average value of the difference image between the Pre and Post images is : "
                 << aSumPixelValuesInDiffImage / (tREAL8)aNumberOfPixelsInImage << std::endl;
        StdOut() << "The maximum value of the difference image between the Pre and Post images is : "
                 << aMaxPixelValuesInDiffImage << std::endl;
    }

    cPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const cPt2dr &aCurrentTranslationPointA,
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

    tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                     const tREAL8 aCurrentRadTranslationPointB,
                                                                     const tREAL8 aCurrentRadTranslationPointC,
                                                                     const tDoubleVect &aVObs)
    {
        const tREAL8 aCurentRadTranslation = aVObs[2] * aCurrentRadTranslationPointA + aVObs[3] * aCurrentRadTranslationPointB +
                                             aVObs[4] * aCurrentRadTranslationPointC;
        return aCurentRadTranslation;
    }

    tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                 const tREAL8 aCurrentRadScalingPointB,
                                                                 const tREAL8 aCurrentRadScalingPointC,
                                                                 const tDoubleVect &aVObs)
    {
        const tREAL8 aCurrentRadScaling = aVObs[2] * aCurrentRadScalingPointA + aVObs[3] * aCurrentRadScalingPointB +
                                          aVObs[4] * aCurrentRadScalingPointC;
        return aCurrentRadScaling;
    }
    
    void ReadFileNameLoadData(const std::string aImageFilename, tIm &aImage,
                              tDIm *&aDataImage, cPt2di &aSzIm)
    {
        aImage = tIm::FromFile(aImageFilename);

        aDataImage = &aImage.DIm();
        aSzIm = aDataImage->Sz();
    }

    void LoadPrePostImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage, tIm &aImPre, tIm &aImPost)
    {
        (aPreOrPostImage == "pre") ? aCurIm = aImPre : aCurIm = aImPost;
        aCurDIm = &aCurIm.DIm();
    }

    bool ManageDifferentCasesOfEndIterations(const int aIterNumber, const int aNumberOfScales, const int aNumberOfEndIterations,
                                             bool aIsLastIters, tIm &aImPre, tIm &aImPost, tIm aCurPreIm, tDIm *aCurPreDIm,
                                             tIm aCurPostIm, tDIm *aCurPostDIm)
    {
        switch (aNumberOfEndIterations)
        {
        case 1: // one last iteration
            if (aIterNumber == aNumberOfScales)
            {
                aIsLastIters = true;
                LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
                LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
            }
            break;
        case 2: // two last iterations
            if ((aIterNumber == aNumberOfScales) || (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 1))
            {
                aIsLastIters = true;
                LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
                LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
            }
            break;
        case 3: //  three last iterations
            if ((aIterNumber == aNumberOfScales) || (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 2) ||
                (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 1))
            {
                aIsLastIters = true;
                LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
                LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
            }
            break;
        default: // default is two last iterations
            if ((aIterNumber == aNumberOfScales) || (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 1))
            {
                aIsLastIters = true;
                LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
                LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
            }
            break;
        }
        return aIsLastIters;
    }

    void DisplayLastUnknownValues(const tDenseVect &aVFinalSol, const bool aDisplayLastRadiometryValues,
                                  const bool aDisplayLastTranslationValues)
    {
        if (aDisplayLastRadiometryValues && aDisplayLastTranslationValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        if (aDisplayLastRadiometryValues && !aDisplayLastTranslationValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                if (aFinalUnk % 4 == 0 || aFinalUnk % 4 == 1)
                    StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        else if (aDisplayLastRadiometryValues && !aDisplayLastTranslationValues)
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

}; // namespace MMVII