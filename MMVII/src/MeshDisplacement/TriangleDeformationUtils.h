#ifndef _TRIANGLEDEFORMATIONUTILS_H_
#define _TRIANGLEDEFORMATIONUTILS_H_

#include "MMVII_Geom2D.h"

namespace MMVII
{
    typedef cIm2D<tREAL8> tIm;
    typedef cDataIm2D<tREAL8> tDIm;
    typedef cDenseVect<double> tDenseVect;
    typedef std::vector<double> tDoubleVect;

    class cPtInsideTriangles
    {
    public:
        cPtInsideTriangles(const cTriangle2DCompiled<tREAL8> &aCompTri,              // a compiled triangle
                           const std::vector<cPt2di> &aVectorFilledwithInsidePixels, // vector containing pixels inside triangles
                           const size_t aFilledPixel,                                // a counter that is looping over pixels in triangles
                           const cDataIm2D<tREAL8> &aDIm);                           // image
        cPt3dr GetBarycenterCoordinates() const;                                     // Accessor for barycenter coordinates
        cPt2dr GetCartesianCoordinates() const;                                      // Accessor for cartesian coordinates
        tREAL8 GetPixelValue() const;                                                // Accessor for pixel value at coordinates

    private:
        cPt3dr mBarycenterCoordinatesOfPixel; // Barycentric coordinates of pixel.
        cPt2dr mFilledIndices;                // 2D cartesian coordinates of pixel.
        tREAL8 mValueOfPixel;                 // Intensity in image at pixel.
    };

    //-------------------------------------------//

    // Build uniform vector of coordinates and apply Delaunay triangulation
    void ConstructUniformRandomVectorAndApplyDelaunay(std::vector<cPt2dr> &aVectorPts, const int aNumberOfPointsToGenerate,
                                                      const int aRandomUniformLawUpperBoundLines, const int aRandomUniformLawUpperBoundCols,
                                                      cTriangulation2D<tREAL8> &aDelaunayTri);
    // Generate coordinates from uniform law for Delaunay triangulation application
    void GeneratePointsForDelaunay(std::vector<cPt2dr> &aVectorPts, const int aNumberOfPointsToGenerate,
                                   int aRandomUniformLawUpperBoundLines, int aRandomUniformLawUpperBoundCols,
                                   cTriangulation2D<tREAL8> &aDelaunayTri, const cPt2di &aSzImPre);
    // Initialise values of unknowns at the beginning of optimisation process after user has input information
    void InitialisationAfterExe(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                cResolSysNonLinear<tREAL8> *&aSys);
    // Initialise problem after user has input information for translation
    void InitialisationAfterExeTranslation(cTriangulation2D<tREAL8> &aDelaunayTri,
                                           cResolSysNonLinear<tREAL8> *&aSys);
    // Initialise problem after user has input information for radiometry
    void InitialisationAfterExeRadiometry(cTriangulation2D<tREAL8> &aDelaunayTri,
                                            cResolSysNonLinear<tREAL8> *&aSys);
    // Check whether point has a correlation value or not thanks to MMVI correlation mask
    bool CheckValidCorrelationValue(tDIm * aMask, const cPt2di &aPointTri);
    // Return correct value for initalisation depending on mask
    tREAL8 ReturnCorrectInitialisationValue(const bool aIsValidCorrelation, tDIm *aIntermediateDispMap,
                                            const cPt2di &aTriPoint, const tREAL8 aValueToReturnIfFalse);
    // Construct difference image and compute average and max pixel value on ths image
    void SubtractPrePostImageAndComputeAvgAndMax(tIm aImDiff, tDIm *aDImDiff, tDIm *aDImPre, 
                                                 tDIm *aDImPost, cPt2di aSzImPre);
    // Read image filename and loads into MMVII data
    void ReadFileNameLoadData(const std::string aImageFilename, tIm &aImage,
                              tDIm *&aDataImage, cPt2di &aSzIm);
    // Loads current pre and post images
    void LoadPrePostImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage, tIm &aImPre, tIm &aImPost);
    // Load image and data according to number of iterations to optimise on original image
    bool ManageDifferentCasesOfEndIterations(const int aIterNumber, const int aNumberOfScales, const int aNumberOfEndIterations,
                                                bool aIsLastIters, tIm &aImPre, tIm &aImPost, tIm aCurPreIm, tDIm *aCurPreDIm,
                                                tIm aCurPostIm, tDIm *aCurPostDIm);
    // Display values of unknowns at last iteration of optimisation process
    void DisplayLastUnknownValues(const tDenseVect &aVFinalSol, const bool aDisplayLastRadiometryValues,
                                    const bool aDisplayLastTranslationValues);
    // Apply barycentric translation formula to current translation values
    cPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const cPt2dr &aCurrentTranslationPointA,
                                                            const cPt2dr &aCurrentTranslationPointB,
                                                            const cPt2dr &aCurrentTranslationPointC,
                                                            const tDoubleVect &aVObs);
    // Apply barycentric translation formula to current radiometric translation values
    tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                        const tREAL8 aCurrentRadTranslationPointB,
                                                                        const tREAL8 aCurrentRadTranslationPointC,
                                                                        const tDoubleVect &aVObs);
    // Apply barycentric translation formula to current radiometric scaling values
    tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                    const tREAL8 aCurrentRadScalingPointB,
                                                                    const tREAL8 aCurrentRadScalingPointC,
                                                                    const tDoubleVect &aVObs);

}

#endif // _TRIANGLEDEFORMATIONUTILS_H_