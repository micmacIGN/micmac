#ifndef _TRIANGLEDEFORMATIONUTILS_H_
#define _TRIANGLEDEFORMATIONUTILS_H_

#include "MMVII_Geom2D.h"

namespace MMVII
{
    typedef cIm2D<tREAL8> tIm;
    typedef cDataIm2D<tREAL8> tDIm;
    typedef cDenseVect<double> tDenseVect;
    typedef std::vector<double> tDoubleVect;

    //----------------------------------------//

    /****************************************/
    /*                                      */
    /*          cPtInsideTriangles          */
    /*                                      */
    /****************************************/

    class cPtInsideTriangles
    {
    public:
        cPtInsideTriangles(const cTriangle2DCompiled<tREAL8> &aCompTri,              // a compiled triangle
                           const std::vector<tPt2di> &aVectorFilledwithInsidePixels, // vector containing pixels inside triangles
                           const size_t aFilledPixel,                                // a counter that is looping over pixels in triangles
                           const cDataIm2D<tREAL8> &aDIm);                           // image
        cPt3dr GetBarycenterCoordinates() const;                                     // Accessor for barycenter coordinates
        tPt2dr GetCartesianCoordinates() const;                                      // Accessor for cartesian coordinates
        tREAL8 GetPixelValue() const;                                                // Accessor for pixel value at coordinates

    private:
        cPt3dr mBarycenterCoordinatesOfPixel; // Barycentric coordinates of pixel.
        tPt2dr mFilledIndices;                // 2D cartesian coordinates of pixel.
        tREAL8 mValueOfPixel;                 // Intensity in image at pixel.
    };

    /****************************************/
    /*                                      */
    /*           cNodeOfTriangles           */
    /*                                      */
    /****************************************/

    class cNodeOfTriangles
    {
        typedef cDenseVect<double> tDenseVect;
        typedef std::vector<int> tIntVect;

    public:
        cNodeOfTriangles(const tDenseVect &aVecSol,    // Current solution vector
                         const tIntVect &aIndicesVec,  // Indices of current triangle in solution vector
                         const int adXIndex,           // Index for current x-displacement in solution vector
                         const int adYIndex,           // Index for current y-displacement in solution vector
                         const tREAL8 aRadTrIndex,     // Index for current radiometry translation in solution vector
                         const tREAL8 aRadScIndex,     // Index for current radiometry scaling in solution vector
                         const tTri2dr &aTri,          // Current triangle
                         const int aPointNumberInTri); // Index of point in triangle : 0, 1 or 2

        tPt2dr GetInitialNodeCoordinates() const;       // Accessor
        tPt2dr GetCurrentXYDisplacementVector() const;  // Accessor
        tREAL8 GetCurrentRadiometryTranslation() const; // Accessor
        tREAL8 &GetCurrentRadiometryTranslation();      // Accessor
        tREAL8 GetCurrentRadiometryScaling() const;     // Accessor
        tREAL8 &GetCurrentRadiometryScaling();          // Accessor

    private:
        tPt2dr mInitialNodeCoordinates;  // Coordinates of knot before displacement
        tPt2dr mCurXYDisplacementVector; // Vector containing current dx, dy displacement values
        tREAL8 mCurRadSc;                // Current radiometry scaling value
        tREAL8 mCurRadTr;                // Current radiometry translation value
    };

    //-------------------------------------------//

    // Build uniform vector of coordinates and apply Delaunay triangulation
    void ConstructUniformRandomVectorAndApplyDelaunay(std::vector<tPt2dr> &aVectorPts, const int aNumberOfPointsToGenerate,
                                                      const int aRandomUniformLawUpperBoundLines, const int aRandomUniformLawUpperBoundCols,
                                                      cTriangulation2D<tREAL8> &aDelaunayTri);
    // Generate coordinates from uniform law for Delaunay triangulation application
    void GeneratePointsForDelaunay(std::vector<tPt2dr> &aVectorPts, const int aNumberOfPointsToGenerate,
                                   int aRandomUniformLawUpperBoundLines, int aRandomUniformLawUpperBoundCols,
                                   cTriangulation2D<tREAL8> &aDelaunayTri, const tPt2di &aSzImPre);
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
    bool CheckValidCorrelationValue(tDIm *aMask, const cNodeOfTriangles &aPtOfTri);
    // Return correct value for initalisation depending on mask
    tREAL8 ReturnCorrectInitialisationValue(const bool aIsValidCorrelation, tDIm *aIntermediateDispMap,
                                            const cNodeOfTriangles &aPtOfTri, const tREAL8 aValueToReturnIfFalse);
    // Construct difference image and compute average and max pixel value on ths image
    void SubtractPrePostImageAndComputeAvgAndMax(tIm &aImDiff, tDIm *aDImDiff, tDIm *aDImPre,
                                                 tDIm *aDImPost, tPt2di &aSzImPre);
    // Read image filename and loads into MMVII data
    void ReadFileNameLoadData(const std::string &aImageFilename, tIm &aImage,
                              tDIm *&aDataImage, tPt2di &aSzIm);
    // Loads current pre and post images
    void LoadPrePostImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage, tIm &aImPre, tIm &aImPost);
    // Load image and data according to number of iterations to optimise on original image
    bool ManageDifferentCasesOfEndIterations(const int aIterNumber, const int aNumberOfScales, const int aNumberOfEndIterations,
                                             bool aIsLastIters, tIm &aImPre, tIm &aImPost, tIm &aCurPreIm, tDIm *aCurPreDIm,
                                             tIm &aCurPostIm, tDIm *aCurPostDIm);
    // Display values of unknowns at last iteration of optimisation process
    void DisplayLastUnknownValues(const tDenseVect &aVFinalSol, const bool aDisplayLastRadiometryValues,
                                  const bool aDisplayLastTranslationValues);
    // Apply barycentric translation formula to current translation values
    tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aCurrentTranslationPointA,
                                                          const tPt2dr &aCurrentTranslationPointB,
                                                          const tPt2dr &aCurrentTranslationPointC,
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