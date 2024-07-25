#ifndef _TRIANGLEDEFORMATIONUTILS_H_
#define _TRIANGLEDEFORMATIONUTILS_H_

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_Interpolators.h"

using namespace NS_SymbolicDerivative;

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
                           cDataIm2D<tREAL8> *&aDIm);                                // image
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
        cNodeOfTriangles(); // default constructor

        cNodeOfTriangles(const tDenseVect &aVecSol,    // Current solution vector
                         const tIntVect &aIndicesVec,  // Indices of current triangle in solution vector
                         const int adXIndex,           // Index for current x-displacement in solution vector
                         const int adYIndex,           // Index for current y-displacement in solution vector
                         const int aRadTrIndex,        // Index for current radiometry translation in solution vector
                         const int aRadScIndex,        // Index for current radiometry scaling in solution vector
                         const tTri2dr &aTri,          // Current triangle
                         const int aPointNumberInTri); // Index of point in triangle : 0, 1 or 2

        cNodeOfTriangles(const tDenseVect &aVecSol,   // Current solution vector
                         const tIntVect &aIndicesVec, // Indices of current triangle in solution vector
                         const int aXIndices,         // Index for current x-displacement in solution vector
                         const int aYIndices,         // Index for current y-displacement in solution vector
                         const int aRadTrIndices,     // Index for current radiometry translation in solution vector
                         const int aRadScIndices,     // Index for current radiometry scaling in solution vector
                         const tTri2dr &aTri,         // Current triangle
                         const tPt3di &aFace,         // Current face of triangle
                         const int aPointNumberInTri, // Index of point in triangle : 0, 1 or 2
                         const int anIdOfNode);       // Id of point when looping over triangles

        tPt2dr GetInitialNodeCoordinates() const;       // Accessor
        tPt2dr GetCurrentXYDisplacementValues() const;  // Accessor
        tREAL8 GetCurrentRadiometryTranslation() const; // Accessor
        tREAL8 &GetCurrentRadiometryTranslation();      // Accessor
        tREAL8 GetCurrentRadiometryScaling() const;     // Accessor
        tREAL8 &GetCurrentRadiometryScaling();          // Accessor

        int GetPointId() const;      // Accessor
        int &GetPointId();           // Accessor
        int GetTriangleFace() const; // Accessor
        int &GetTriangleFace();      // Accessor

        void AddData(const cAuxAr2007 &anAux, cNodeOfTriangles &aPtToSerialise); // Add data to xml file
        void SaveTriangleNodeToFile() const;                                     // Save to xml file
        void ShowTriangleNodeCarateristics() const;                              // Display information about node
        // Adds ability to re-read already saved node
        static std::unique_ptr<cNodeOfTriangles> ReadSerialisedTriangleNode(const tDenseVect &aVecSol, const tIntVect &aIndVec,
                                                                            const int aXInd, const int aYInd, const int aRadTrInd,
                                                                            const int aRadScInd, const tTri2dr &aTriangle, const tPt3di &aFace,
                                                                            const int aPointNumberInTri, const int anIdOfPoint);

        static std::string NameFileToSaveNode(const int anId); // Gives name to saved file

    private:
        tPt2dr mInitialNodeCoordinates;  // Coordinates of knot before displacement
        tPt2dr mCurXYDisplacementVector; // Vector containing current dx, dy displacement values
        tREAL8 mCurRadSc;                // Current radiometry scaling value
        tREAL8 mCurRadTr;                // Current radiometry translation value
        int mIdOfPt;                     // Id of point when looping over triangles
        int mFaceOfTriangle;             // The face number associated to the point

        friend std::ostream &operator<<(std::ostream &os, const cNodeOfTriangles &obj);
    };

    // void AddData(const cAuxAr2007 &anAux, cNodeOfTriangles &aPtToSerialise);

    /************************************************/
    /*                                              */
    /*       cMultipleTriangleNodesSerialiser       */
    /*                                              */
    /************************************************/

    class cMultipleTriangleNodesSerialiser
    {
    public:
        cMultipleTriangleNodesSerialiser();                         // default constructor
        cMultipleTriangleNodesSerialiser(const std::string &aName); // constructor

        void MultipleNodesToFile(const std::string &aFileName) const; // Save triangle nodes to .xml file

        // Creates new object able to append multiple triangle nodes
        static std::unique_ptr<cMultipleTriangleNodesSerialiser> NewMultipleTriangleNodes(const std::string &aName);
        // Read a saved vector of triangle nodes into vector
        static std::unique_ptr<cMultipleTriangleNodesSerialiser> ReadVectorOfTriangleNodes(const std::string &aFileName);

        std::string GetName() const; // Accessor
        // Appends triangle nodes in vector
        void PushInVector(std::unique_ptr<cNodeOfTriangles> &aTriangleNode);
        void AddData(const cAuxAr2007 &anAux); // Add data method

        void ShowAllTriangleNodes(const std::string aAllOrSingularValue, int aNodeNumber = 0) const; // Display contents of vector containing all triangle nodes

    private:
        std::string mName;                                  // File name to use to save/read vector of triangle nodes
        std::vector<cNodeOfTriangles> mVectorTriangleNodes; // Vector containing all triangle nodes
    };

    //-------------------------------------------//

    // Build uniform vector of coordinates and apply Delaunay triangulation
    void BuildUniformRandomVectorAndApplyDelaunay(const int aNumberOfPointsToGenerate, const int aRandomUniformLawUpperBoundLines,
                                                  const int aRandomUniformLawUpperBoundCols, cTriangulation2D<tREAL8> &aDelaunayTri);
    // Build points vector that have coordinates that make up a grid made of rectangles
    void GeneratePointsForRectangleGrid(const int aNumberOfPoints, const int aGridSizeLines,
                                        const int aGridSizeCols, cTriangulation2D<tREAL8> &aDelaunayTri);
    // Generate coordinates from uniform law for Delaunay triangulation application
    void DefineValueLimitsForPointGenerationAndBuildGrid(const int aNumberOfPointsToGenerate, int aRandomUniformLawUpperBoundLines,
                                                         int aRandomUniformLawUpperBoundCols, cTriangulation2D<tREAL8> &aDelaunayTri,
                                                         const tPt2di &aSzImPre, const bool aBuildUniformVector);

    // Initialise equation and interpolator if needed
    void InitialiseInterpolationAndEquation(cCalculator<tREAL8> *&aEqDeformTri, cDiffInterpolator1D *&aInterpol,
                                            const std::vector<std::string> aArgsVectorInterpol, const bool aUseLinearGradInterpolation);
    // Initialise values of unknowns at the beginning of optimisation process after user has input information
    void InitialisationWithUserValues(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                      cResolSysNonLinear<tREAL8> *&aSys,
                                      const bool aUserInitialisation,
                                      const tREAL8 aXTranslationInitVal,
                                      const tREAL8 aYTranslationInitVal,
                                      const tREAL8 aRadTranslationInitVal,
                                      const tREAL8 aRadScaleInitVal);
    // Initialise unknowns values with values obtained at previous execution
    void InitialiseWithPreviousExecutionValues(const cTriangulation2D<tREAL8> &aDelTri,
                                               cResolSysNonLinear<tREAL8> *&aSys,
                                               const std::string &aNameDepXFile, tIm &aImDepX,
                                               tDIm *&aDImDepX, tPt2di &aSzImDepX,
                                               const std::string &aNameDepYFile, tIm &aImDepY,
                                               tDIm *&aDImDepY, tPt2di &aSzImDepY,
                                               const std::string &aNameCorrelationMask,
                                               tIm &aImCorrelationMask, tDIm *&aDImCorrelationMask,
                                               tPt2di &aSzCorrelationMask);
    // Initialise equation and interpolation if needed for translation
    void InitialiseInterpolationAndEquationTranslation(cCalculator<tREAL8> *&aEqTranslationTri, cDiffInterpolator1D *&aInterpolTr,
                                                       const std::vector<std::string> &aArgsVectorInterpolTr, const bool aUseLinearGradInterpolation);
    // Initialise problem after user has input information for translation
    void InitialiseWithUserValuesTranslation(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                             cResolSysNonLinear<tREAL8> *&aSysTranslation,
                                             const bool aUserInitialisation,
                                             const tREAL8 aXTranslationInitVal,
                                             const tREAL8 aYTranslationInitVal);
    // Initialise unknowns values with values obtained at previous execution for translation
    void InitialiseWithPreviousExecutionValuesTranslation(const cTriangulation2D<tREAL8> &aDelTri,
                                                          cResolSysNonLinear<tREAL8> *&aSysTranslation,
                                                          const std::string &aNameDepXFile, tIm &aImDepX,
                                                          tDIm *&aDImDepX, tPt2di &aSzImDepX,
                                                          const std::string &aNameDepYFile, tIm &aImDepY,
                                                          tDIm *&aDImDepY, tPt2di &aSzImDepY,
                                                          const std::string &aNameCorrelationMask,
                                                          tIm &aImCorrelationMask, tDIm *&aDImCorrelationMask,
                                                          tPt2di &aSzCorrelationMask);
    // Initialise equation and interpolation if needed for radiometry
    void InitialiseInterpolationAndEquationRadiometry(cCalculator<tREAL8> *&aEqRadiometryTri, cDiffInterpolator1D *&aInterpolRad,
                                                      const std::vector<std::string> aArgsVectorInterpolRad, const bool aUseLinearGradInterpolation);
    // Initialise problem after user has input information for radiometry
    void InitialiseWithUserValuesRadiometry(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                            cResolSysNonLinear<tREAL8> *&aSysRadiometry,
                                            const bool aUserInitialisation,
                                            const tREAL8 aRadTranslationInitVal,
                                            const tREAL8 aRadScaleInitVal);
    // Create unique pointer to triangle node to ease serialisation
    std::unique_ptr<cNodeOfTriangles> DefineNewTriangleNode(const tDenseVect &aVecSol, const std::vector<int> aIndVec, const int aXInd,
                                                            const int aYInd, const int aRadTrInd, const int aRadScInd, const tTri2dr &aTriangle,
                                                            const tPt3di &aFace, const int aPointNumberInTri, const int anIdOfPoint);
    // Check whether point has a correlation value or not thanks to MMVI correlation mask
    bool CheckValidCorrelationValue(tDIm *aMask, const cNodeOfTriangles &aPtOfTri);
    // Return correct value for initalisation depending on mask
    tREAL8 ReturnCorrectInitialisationValue(tDIm *aMask, tDIm *aIntermediateDispMap,
                                            const cNodeOfTriangles &aPtOfTri, const tREAL8 aValueToReturnIfFalse);
    // Check if interpolated bilinear value in displacement map can be returned or if integer value is needed
    tPt2dr CheckReturnOfBilinearValue(tDIm *&aDImDispXMap, tDIm *&aDImDispYMap,
                                      const tPt2dr &aRealCoordPoint, const tPt2di &aIntCoordPoint);
    // Construct difference image and compute average and max pixel value on ths image
    void SubtractPrePostImageAndComputeAvgAndMax(tIm &aImDiff, tDIm *aDImDiff, tDIm *aDImPre,
                                                 tDIm *aDImPost, tPt2di &aSzImPre);
    // Read image filename and loads into MMVII data
    void ReadFileNameLoadData(const std::string &aImageFilename, tIm &aImage,
                              tDIm *&aDataImage, tPt2di &aSzIm);
    // Loads current pre and post images
    void LoadPrePostImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage, tIm &aImPre, tIm &aImPost);
    // Initialise displacement maps with null coefficients
    void InitialiseDisplacementMaps(tPt2di &aSzIm, tIm &aImDispMap, tDIm *&aDImDispMap, tPt2di &aSzImDispMap);
    ;
    // Load node of triangulation, get current displacement values point
    tPt2dr LoadNodeAndReturnCurrentDisplacement(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri);
    // Load node of triangulation, get current displacement values point and append node to vector of all nodes
    tPt2dr LoadNodeAppendVectorAndReturnCurrentDisplacement(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                            const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                            const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
                                                            const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
                                                            std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes);
    // Load node of triangulation, get current radiometry translation value
    tREAL8 LoadNodeAndReturnCurrentRadiometryTranslation(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                         const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                         const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri);
    // Load node of triangulation, get current radiometry translation value and append node to vector of all nodes
    tREAL8 LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                                     const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                                     const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
                                                                     const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
                                                                     std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes);
    // Load node of triangulation, get current radiometry scaling value
    tREAL8 LoadNodeAndReturnCurrentRadiometryScaling(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                     const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                     const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri);
    // Load node of triangulation, get current radiometry scaling value and append node to vector of all nodes
    tREAL8 LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                                 const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                                 const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
                                                                 const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
                                                                 std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfMultipleNodes);
    // Load image and data according to number of iterations to optimise on original image
    bool ManageDifferentCasesOfEndIterations(const int aIterNumber, const int aNumberOfScales, const int aNumberOfEndIterations,
                                             bool aIsLastIters, tIm &aImPre, tIm &aImPost, tIm &aCurPreIm, tDIm *&aCurPreDIm,
                                             tIm &aCurPostIm, tDIm *&aCurPostDIm);
    // Apply barycentric translation formula to current translation values with observations
    tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aCurrentTranslationPointA,
                                                          const tPt2dr &aCurrentTranslationPointB,
                                                          const tPt2dr &aCurrentTranslationPointC,
                                                          const tDoubleVect &aVObs);
    // Apply barycentric translation formula to current translation values with observations with inside pixels
    tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aLastTranslationPointA,
                                                          const tPt2dr &aLastTranslationPointB,
                                                          const tPt2dr &aLastTranslationPointC,
                                                          const cPtInsideTriangles &aLastPixInsideTriangle);
    // Apply barycentric translation formula to current radiometric translation values
    tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                     const tREAL8 aCurrentRadTranslationPointB,
                                                                     const tREAL8 aCurrentRadTranslationPointC,
                                                                     const tDoubleVect &aVObs);
    tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                     const tREAL8 aCurrentRadTranslationPointB,
                                                                     const tREAL8 aCurrentRadTranslationPointC,
                                                                     const cPtInsideTriangles &aPixInsideTriangle);
    // Apply barycentric translation formula to current radiometric scaling values
    tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                 const tREAL8 aCurrentRadScalingPointB,
                                                                 const tREAL8 aCurrentRadScalingPointC,
                                                                 const tDoubleVect &aVObs);
    tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                 const tREAL8 aCurrentRadScalingPointB,
                                                                 const tREAL8 aCurrentRadScalingPointC,
                                                                 const cPtInsideTriangles &aLastPixInsideTriangle);
    // Produces output image with computed radiometry and displacement maps with computed translations
    void FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                            const tPt2dr &aLastTranslatedFilledPoint,
                                            const tREAL8 aLastRadiometryTranslation,
                                            const tREAL8 aLastRadiometryScaling, tPt2di &aSzImOut,
                                            tDIm *&aDImDepX, tDIm *&aDImDepY, tDIm *&aDImOut);
    // Produces output image when computing just radiometry
    void FillOutputImageRadiometry(const cPtInsideTriangles &aLastPixInsideTriangle,
                                   const tREAL8 aLastRadiometryTranslation,
                                   const tREAL8 aLastRadiometryScaling,
                                   tDIm *&aDImOut);
    // Produces x and y displacement maps for computing just translation
    void FillDisplacementMapsTranslation(const cPtInsideTriangles &aLastPixInsideTriangle,
                                         const tPt2dr &aLastTranslatedFilledPoint, tPt2di &aSzImOut,
                                         tDIm *&aDImDepX, tDIm *&aDImDepY, tDIm *&aDImOut);
    // Fills displacement map with difference between ground truth and interpolated pixel in ground truth
    void FillDiffDisplacementMap(tDIm *&aDImDispMap, tDIm *&aDImDiffMap,
                                 tDIm *&aDImTranslatedDispMap,
                                 const cPtInsideTriangles &aPixInsideTriangle,
                                 const tREAL8 aPixInsideTriangleCoord,
                                 const tREAL8 aTranslatedRealDispMapPointCoord,
                                 const bool aComputeDiffDispMaps, const bool aComputeInterpTranslationDispMaps);
    // Save intermediate displacement maps in different folder or not
    void SaveMultiScaleDisplacementMapsToFile(tDIm *&aDImDepX, tDIm *&aDImDepY, const bool aUserDefinedFolderName,
                                              const std::string &aFolderPathToSave, const std::string &aDepXFileNameToSave,
                                              const std::string &aDepYFileNameToSave, const int aIterNumber,
                                              const int aNumberOfPointsToGenerate, const int aTotalNumberOfIterations);
    // Save final displacement maps in different folder or not
    void SaveFinalDisplacementMapsToFile(tDIm *&aDImDepX, tDIm *&aDImDepY, const bool aUserDefinedFolderName,
                                         const std::string &aFolderPathToSave, const std::string &aDepXFileNameToSave,
                                         const std::string &aDepYFileNameToSave, const int aNumberOfPointsToGenerate,
                                         const int aTotalNumberOfIterations);
    // Save intermediate output image in different folder or not
    void SaveOutputImageToFile(tDIm *&aDImOut, const bool aUserDefinedFolderName, const std::string &aFolderPathToSave,
                               const std::string aOutputImageFileNameToSave,
                               const int aNumberOfPointsToGenerate, const int aTotalNumberOfIterations);
    // Display last values of unknowns and compute statistics on these values : min, mean, max and variance
    void DisplayLastUnknownValuesAndComputeStatistics(const tDenseVect &aVFinalSol, const tDenseVect &aVInitSol);
    // Display values of unknowns at last iteration of optimisation process
    void DisplayLastUnknownValues(const tDenseVect &aVFinalSol, const bool aDisplayLastRadiometryValues,
                                  const bool aDisplayLastTranslationValues);

}

#endif // _TRIANGLEDEFORMATIONUTILS_H_