#include "TriangleDeformationUtils.h"

/**
 \file TriangleDeformationUtils.cpp
 \brief File containing annexe methods that can be used by
 other classes linked to triangle deformation computation.
**/

namespace MMVII
{
    /****************************************/
    /*                                      */
    /*          cPtInsideTriangles          */
    /*                                      */
    /****************************************/

    cPtInsideTriangles::cPtInsideTriangles(const cTriangle2DCompiled<tREAL8> &aCompTri,              // a compiled triangle
                                           const std::vector<tPt2di> &aVectorFilledwithInsidePixels, // vector containing pixels insisde triangles
                                           const size_t aFilledPixel,                                // a counter that is looping over pixels in triangles
                                           cDataIm2D<tREAL8> *&aDIm)                                 // image
    {
        mFilledIndices = tPt2dr(aVectorFilledwithInsidePixels[aFilledPixel].x(), aVectorFilledwithInsidePixels[aFilledPixel].y());
        mBarycenterCoordinatesOfPixel = aCompTri.CoordBarry(mFilledIndices);
        if (aDIm->InsideBL(mFilledIndices))
            mValueOfPixel = aDIm->GetVBL(mFilledIndices);
        else
            mValueOfPixel = aDIm->GetV(tPt2di(mFilledIndices.x(), mFilledIndices.y()));
    }

    cPt3dr cPtInsideTriangles::GetBarycenterCoordinates() const { return mBarycenterCoordinatesOfPixel; } // Accessor
    tPt2dr cPtInsideTriangles::GetCartesianCoordinates() const { return mFilledIndices; }                 // Accessor
    tREAL8 cPtInsideTriangles::GetPixelValue() const { return mValueOfPixel; }                            // Accessor

    /****************************************/
    /*                                      */
    /*           cNodeOfTriangles           */
    /*                                      */
    /****************************************/

    cNodeOfTriangles::cNodeOfTriangles()
    {
    }

    cNodeOfTriangles::cNodeOfTriangles(const tDenseVect &aVecSol,
                                       const tIntVect &aIndicesVec,
                                       const int aXIndices,
                                       const int aYIndices,
                                       const int aRadTrIndices,
                                       const int aRadScIndices,
                                       const tTri2dr &aTri,
                                       const int aPointNumberInTri)
    {
        mInitialNodeCoordinates = aTri.Pt(aPointNumberInTri);
        mCurXYDisplacementVector = tPt2dr(aVecSol(aIndicesVec.at(aXIndices)),
                                          aVecSol(aIndicesVec.at(aYIndices)));
        mCurRadTr = aVecSol(aIndicesVec.at(aRadTrIndices));
        mCurRadSc = aVecSol(aIndicesVec.at(aRadScIndices));
    }

    cNodeOfTriangles::cNodeOfTriangles(const tDenseVect &aVecSol,
                                       const tIntVect &aIndicesVec,
                                       const int aXIndices,
                                       const int aYIndices,
                                       const int aRadTrIndices,
                                       const int aRadScIndices,
                                       const tTri2dr &aTri,
                                       const tPt3di &aFace,
                                       const int aPointNumberInTri,
                                       const int anIdOfPoint) : mIdOfPt(anIdOfPoint)
    {
        mInitialNodeCoordinates = aTri.Pt(aPointNumberInTri);
        mCurXYDisplacementVector = tPt2dr(aVecSol(aIndicesVec.at(aXIndices)),
                                          aVecSol(aIndicesVec.at(aYIndices)));
        mCurRadTr = aVecSol(aIndicesVec.at(aRadTrIndices));
        mCurRadSc = aVecSol(aIndicesVec.at(aRadScIndices));

        if (aPointNumberInTri == 0)
            mFaceOfTriangle = aFace.x();
        else if (aPointNumberInTri == 1)
            mFaceOfTriangle = aFace.y();
        else
            mFaceOfTriangle = aFace.z();
    }

    tPt2dr cNodeOfTriangles::GetInitialNodeCoordinates() const { return mInitialNodeCoordinates; }       // Accessor
    tPt2dr cNodeOfTriangles::GetCurrentXYDisplacementValues() const { return mCurXYDisplacementVector; } // Accessor
    tREAL8 cNodeOfTriangles::GetCurrentRadiometryScaling() const { return mCurRadSc; }                   // Accessor
    tREAL8 cNodeOfTriangles::GetCurrentRadiometryTranslation() const { return mCurRadTr; }               // Accessor
    tREAL8 &cNodeOfTriangles::GetCurrentRadiometryTranslation() { return mCurRadTr; }                    // Accessor
    tREAL8 &cNodeOfTriangles::GetCurrentRadiometryScaling() { return mCurRadSc; }                        // Accessor

    std::ostream &operator<<(std::ostream &os, const cNodeOfTriangles &obj)
    {
        obj.ShowTriangleNodeCarateristics();
        return os;
    }

    void cNodeOfTriangles::AddData(const cAuxAr2007 &anAux, cNodeOfTriangles &aPtToSerialise)
    {
        MMVII::AddData(cAuxAr2007("Id", anAux), aPtToSerialise.GetPointId());
        MMVII::AddData(cAuxAr2007("Face", anAux), aPtToSerialise.GetTriangleFace());
        MMVII::AddData(cAuxAr2007("x", anAux), aPtToSerialise.GetInitialNodeCoordinates().x());
        MMVII::AddData(cAuxAr2007("y", anAux), aPtToSerialise.GetInitialNodeCoordinates().y());
        MMVII::AddData(cAuxAr2007("dx", anAux), aPtToSerialise.GetCurrentXYDisplacementValues().x());
        MMVII::AddData(cAuxAr2007("dy", anAux), aPtToSerialise.GetCurrentXYDisplacementValues().y());
        MMVII::AddData(cAuxAr2007("RadiometryTranslation", anAux), aPtToSerialise.GetCurrentRadiometryTranslation());
        MMVII::AddData(cAuxAr2007("RadiometryScaling", anAux), aPtToSerialise.GetCurrentRadiometryScaling());
    }

    void AddData(const cAuxAr2007 &anAux, cNodeOfTriangles &aPtToSerialise) { aPtToSerialise.AddData(anAux, aPtToSerialise); }

    void cNodeOfTriangles::ShowTriangleNodeCarateristics() const
    {
        StdOut() << "Id of this point : " << this->GetPointId() << std::endl;
        StdOut() << "Face of triangle associated to point : " << this->GetTriangleFace() << std::endl;
        StdOut() << "Initial node coordinates : " << this->GetInitialNodeCoordinates() << "." << std::endl;
        StdOut() << "Current displacement coefficient values : " << this->GetCurrentXYDisplacementValues()
                 << "." << std::endl;
        StdOut() << "Current radiometric coefficient values : " << this->GetCurrentRadiometryTranslation()
                 << " for translation and " << this->GetCurrentRadiometryScaling() << " for scaling." << std::endl;
    }

    void cNodeOfTriangles::SaveTriangleNodeToFile() const
    {
        SaveInFile(*this, NameFileToSaveNode(mIdOfPt));
    }

    std::unique_ptr<cNodeOfTriangles> cNodeOfTriangles::ReadSerialisedTriangleNode(const tDenseVect &aVecSol, const tIntVect &aIndVec,
                                                                                   const int aXInd, const int aYInd, const int aRadTrInd,
                                                                                   const int aRadScInd, const tTri2dr &aTriangle, const tPt3di &aFace,
                                                                                   const int aPointNumberInTri, const int anIdOfPoint)
    {
        std::unique_ptr<cNodeOfTriangles> aReReadSerialisedObj = std::make_unique<cNodeOfTriangles>(aVecSol, aIndVec, aXInd, aYInd, aRadTrInd,
                                                                                                    aRadScInd, aTriangle, aFace, aPointNumberInTri,
                                                                                                    anIdOfPoint);
        ReadFromFile(*aReReadSerialisedObj, NameFileToSaveNode(anIdOfPoint));

        return aReReadSerialisedObj;
    }

    std::string cNodeOfTriangles::NameFileToSaveNode(const int anId)
    {
        return "Id_" + ToStr(anId) + ".xml";
    }

    int cNodeOfTriangles::GetPointId() const
    {
        return mIdOfPt;
    }

    int cNodeOfTriangles::GetTriangleFace() const
    {
        return mFaceOfTriangle;
    }

    int &cNodeOfTriangles::GetPointId()
    {
        return mIdOfPt;
    }

    int &cNodeOfTriangles::GetTriangleFace()
    {
        return mFaceOfTriangle;
    }

    /************************************************/
    /*                                              */
    /*       cMultipleTriangleNodesSerialiser       */
    /*                                              */
    /************************************************/

    cMultipleTriangleNodesSerialiser::cMultipleTriangleNodesSerialiser()
    {
    }

    cMultipleTriangleNodesSerialiser::cMultipleTriangleNodesSerialiser(const std::string &aFileName) : mName(aFileName)
    {
    }

    std::unique_ptr<cMultipleTriangleNodesSerialiser> cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(const std::string &aName)
    {
        std::unique_ptr<cMultipleTriangleNodesSerialiser> aNewMultipleTriangleNodes = std::make_unique<cMultipleTriangleNodesSerialiser>(aName);
        return aNewMultipleTriangleNodes;
    }

    void cMultipleTriangleNodesSerialiser::AddData(const cAuxAr2007 &anAux)
    {
        MMVII::AddData(cAuxAr2007("VectorOfTriangleNodes", anAux), mVectorTriangleNodes);
    }
    void AddData(const cAuxAr2007 &anAux, cMultipleTriangleNodesSerialiser &aSetOfObjsToSerialise) { aSetOfObjsToSerialise.AddData(anAux); }

    void cMultipleTriangleNodesSerialiser::MultipleNodesToFile(const std::string &aFileName) const
    {
        SaveInFile(*this, aFileName);
    }

    void cMultipleTriangleNodesSerialiser::PushInVector(std::unique_ptr<cNodeOfTriangles> &aTriangleDeformationObj)
    {
        mVectorTriangleNodes.push_back(*aTriangleDeformationObj);
    }

    std::unique_ptr<cMultipleTriangleNodesSerialiser> cMultipleTriangleNodesSerialiser::ReadVectorOfTriangleNodes(const std::string &aFileName)
    {
        std::unique_ptr<cMultipleTriangleNodesSerialiser> aNewSetOfMultipleTriangleDeformations = NewMultipleTriangleNodes(aFileName);
        ReadFromFile(*aNewSetOfMultipleTriangleDeformations, aFileName);

        return aNewSetOfMultipleTriangleDeformations;
    }

    std::string cMultipleTriangleNodesSerialiser::GetName() const { return mName; } // Acessor

    void cMultipleTriangleNodesSerialiser::ShowAllTriangleNodes(const std::string &aAllOrSingularValue, int aNodeNumber) const
    {
        if (aAllOrSingularValue == "all")
            StdOut() << "The carateristics of the nodes are : " << mVectorTriangleNodes << std::endl;
        else
            StdOut() << "The carateristics of node number " << aNodeNumber << " are : "
                     << mVectorTriangleNodes.at(aNodeNumber) << std::endl;
    }

    //---------------------------------------------//

    void BuildUniformRandomVectorAndApplyDelaunay(const int aNumberOfPointsToGenerate, const int aRandomUniformLawUpperBoundLines,
                                                  const int aRandomUniformLawUpperBoundCols, cTriangulation2D<tREAL8> &aDelaunayTri)
    {
        std::vector<tPt2dr> aVectorPts;
        // Generate coordinates from drawing lines and columns of coordinates from a uniform distribution
        for (int aNbPt = 0; aNbPt < aNumberOfPointsToGenerate; aNbPt++)
        {
            const tREAL8 aUniformRandomLine = RandUnif_N(aRandomUniformLawUpperBoundLines);
            const tREAL8 aUniformRandomCol = RandUnif_N(aRandomUniformLawUpperBoundCols);
            const tPt2dr aUniformRandomPt(aUniformRandomCol, aUniformRandomLine); // tPt2dr format
            aVectorPts.push_back(aUniformRandomPt);
        }
        aDelaunayTri = aVectorPts;

        aDelaunayTri.MakeDelaunay(); // Delaunay triangulate randomly generated points.
    }

    void DefineValueLimitsForPointGenerationAndBuildGrid(const int aNumberOfPointsToGenerate, int aNumberOfLines, int aNumberOfCols,
                                                         cTriangulation2D<tREAL8> &aDelaunayTri, const tPt2di &aSzIm,
                                                         const bool aBuildUniformVector)
    {
        // If user hasn't defined another value than the default value, it is changed
        if (aNumberOfLines == 1 && aNumberOfCols == 1)
        {
            // Maximum value of coordinates are drawn from [0, NumberOfImageLines[ for lines
            aNumberOfLines = aSzIm.y();
            // Maximum value of coordinates are drawn from [0, NumberOfImageColumns[ for columns
            aNumberOfCols = aSzIm.x();
        }
        else
        {
            if (aNumberOfLines != 1 && aNumberOfCols == 1)
                aNumberOfCols = aSzIm.x();
            else
            {
                if (aNumberOfLines == 1 && aNumberOfCols != 1)
                    aNumberOfLines = aSzIm.y();
            }
        }

        if (aBuildUniformVector)
            BuildUniformRandomVectorAndApplyDelaunay(aNumberOfPointsToGenerate,
                                                     aNumberOfLines, aNumberOfCols,
                                                     aDelaunayTri);
        else
            GeneratePointsForRectangleGrid(aNumberOfPointsToGenerate, aNumberOfLines,
                                           aNumberOfCols, aDelaunayTri);
    }

    void GeneratePointsForRectangleGrid(const int aNumberOfPoints, const int aGridSizeLines,
                                        const int aGridSizeCols, cTriangulation2D<tREAL8> &aDelaunayTri)
    {
        std::vector<tPt2dr> aGridVector;
        const int anEdge = 10; // To take away variations linked to edges

        // be aware that step between grid points are integers
        const int aDistanceLines = aGridSizeLines / std::sqrt(aNumberOfPoints);
        const int aDistanceCols = aGridSizeCols / std::sqrt(aNumberOfPoints);

        const int anEndLinesLoop = aGridSizeLines - anEdge;
        const int anEndColsLoop = aGridSizeCols - anEdge;

        for (int aLineNumber = anEdge; aLineNumber < anEndLinesLoop; aLineNumber += aDistanceLines)
        {
            for (int aColNumber = anEdge; aColNumber < anEndColsLoop; aColNumber += aDistanceCols)
            {
                const tPt2dr aGridPt = tPt2dr(aColNumber, aLineNumber); // tPt2dr format
                aGridVector.push_back(aGridPt);
            }
        }

        aDelaunayTri = aGridVector;

        aDelaunayTri.MakeDelaunay(); // Delaunay triangulate randomly generated points.
    }

    void InitialiseInterpolationAndEquation(cCalculator<tREAL8> *&aEqDeformTri, cDiffInterpolator1D *&aInterpol,
                                            const std::vector<std::string> &aArgsVectorInterpol, const bool aUseLinearGradInterpolation)
    {
        if (aUseLinearGradInterpolation)
            aInterpol = cDiffInterpolator1D::AllocFromNames(aArgsVectorInterpol);
        // true means with derivative, 1 is size of buffer
        aEqDeformTri = aUseLinearGradInterpolation ? EqDeformTriLinearGrad(true, 1) : EqDeformTriBilin(true, 1);
    }

    void InitialisationWithUserValues(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                      cResolSysNonLinear<tREAL8> *&aSys,
                                      const bool aUserInitialisation,
                                      const tREAL8 aXTranslationInitVal,
                                      const tREAL8 aYTranslationInitVal,
                                      const tREAL8 aRadTranslationInitVal,
                                      const tREAL8 aRadScaleInitVal)
    {
        const size_t aStartNumberPts = 4 * aDelaunayTri.NbPts();
        tDenseVect aVInit(aStartNumberPts, eModeInitImage::eMIA_Null);

        if (aUserInitialisation && aXTranslationInitVal != 0 && aYTranslationInitVal != 0
            && aRadTranslationInitVal != 0 && aRadScaleInitVal != 1)
        {
            for (size_t aKtNumber = 0; aKtNumber < aStartNumberPts; aKtNumber++)
            {
                if (aKtNumber % 4 == 0 && aXTranslationInitVal != 0)
                    aVInit(aKtNumber) = aXTranslationInitVal;
                if (aKtNumber % 4 == 1 && aYTranslationInitVal != 0)
                    aVInit(aKtNumber) = aYTranslationInitVal;
                if (aKtNumber % 4 == 2 && aRadTranslationInitVal != 0)
                    aVInit(aKtNumber) = aRadTranslationInitVal;
                if (aKtNumber % 4 == 3)
                    aVInit(aKtNumber) = aRadScaleInitVal;
            }
        }
        else
        {
            for (size_t aKtNumber = 0; aKtNumber < aStartNumberPts; aKtNumber++)
            {
                if (aKtNumber % 4 == 3)
                    aVInit(aKtNumber) = 1;
            }
        }

        aSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInit);
    }

    void InitialiseWithPreviousExecutionValues(const cTriangulation2D<tREAL8> &aDelTri,
                                               cResolSysNonLinear<tREAL8> *&aSys,
                                               const std::string &aNameDepXFile, tIm &aImDepX,
                                               tDIm *&aDImDepX, tPt2di &aSzImDepX,
                                               const std::string &aNameDepYFile, tIm &aImDepY,
                                               tDIm *&aDImDepY, tPt2di &aSzImDepY,
                                               const std::string &aNameCorrelationMask,
                                               tIm &aImCorrelationMask, tDIm *&aDImCorrelationMask,
                                               tPt2di &aSzCorrelationMask)
    {
        tDenseVect aVInit(4 * aDelTri.NbPts(), eModeInitImage::eMIA_Null);

        ReadImageFileNameLoadData(aNameDepXFile, aImDepX,
                                  aDImDepX, aSzImDepX);
        ReadImageFileNameLoadData(aNameDepYFile, aImDepY,
                                  aDImDepY, aSzImDepY);

        ReadImageFileNameLoadData(aNameCorrelationMask, aImCorrelationMask,
                                  aDImCorrelationMask, aSzCorrelationMask);

        for (size_t aTr = 0; aTr < aDelTri.NbFace(); aTr++)
        {
            const tTri2dr aInitTri = aDelTri.KthTri(aTr);
            const tPt3di aIndicesOfTriKnots = aDelTri.KthFace(aTr);

            //----------- index of unknown, finds the associated pixels of current triangle
            std::vector<int> aInitVecInd;
            GetIndicesVector(aInitVecInd, aIndicesOfTriKnots, 4);

            // Get points coordinates associated to triangle
            const cNodeOfTriangles aFirstInitPointOfTri = cNodeOfTriangles(aVInit, aInitVecInd, 0, 1, 2, 3, aInitTri, 0);
            const cNodeOfTriangles aSecondInitPointOfTri = cNodeOfTriangles(aVInit, aInitVecInd, 4, 5, 6, 7, aInitTri, 1);
            const cNodeOfTriangles aThirdInitPointOfTri = cNodeOfTriangles(aVInit, aInitVecInd, 8, 9, 10, 11, aInitTri, 2);

            aVInit(aInitVecInd.at(0)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepX,
                                                                         aFirstInitPointOfTri, 0);
            aVInit(aInitVecInd.at(1)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepY,
                                                                         aFirstInitPointOfTri, 0);
            aVInit(aInitVecInd.at(2)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepX,
                                                                         aSecondInitPointOfTri, 0);
            aVInit(aInitVecInd.at(3)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepY,
                                                                         aSecondInitPointOfTri, 0);
            aVInit(aInitVecInd.at(4)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepX,
                                                                         aThirdInitPointOfTri, 0);
            aVInit(aInitVecInd.at(5)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepY,
                                                                         aThirdInitPointOfTri, 0);
        }

        aSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInit);
    }

    void InitialiseInterpolationAndEquationTranslation(cCalculator<tREAL8> *&aEqTranslationTri, cDiffInterpolator1D *&aInterpolTr,
                                                       const std::vector<std::string> &aArgsVectorInterpolTr, const bool aUseLinearGradInterpolation)
    {
        if (aUseLinearGradInterpolation)
            aInterpolTr = cDiffInterpolator1D::AllocFromNames(aArgsVectorInterpolTr);

        // true means with derivative, 1 is size of buffer
        aEqTranslationTri = aUseLinearGradInterpolation ? EqDeformTriTranslationLinearGrad(true, 1) : EqDeformTriTranslationBilin(true, 1);
    }

    void InitialiseWithUserValuesTranslation(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                             cResolSysNonLinear<tREAL8> *&aSysTranslation,
                                             const bool aUserInitialisation,
                                             const tREAL8 aXTranslationInitVal,
                                             const tREAL8 aYTranslationInitVal)
    {
        const size_t aNumberPts = 2 * aDelaunayTri.NbPts();
        tDenseVect aVInitTranslation(aNumberPts, eModeInitImage::eMIA_Null);

        if (aUserInitialisation && aXTranslationInitVal != 0 && aYTranslationInitVal != 0)
        {
            for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
            {
                if (aKtNumber % 2 == 0 && aXTranslationInitVal != 0)
                    aVInitTranslation(aKtNumber) = aXTranslationInitVal;
                if (aKtNumber % 2 == 1 && aYTranslationInitVal != 0)
                    aVInitTranslation(aKtNumber) = aYTranslationInitVal;
            }
        }

        aSysTranslation = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInitTranslation);
    }

    void InitialiseWithPreviousExecutionValuesTranslation(const cTriangulation2D<tREAL8> &aDelTri,
                                                          cResolSysNonLinear<tREAL8> *&aSysTranslation,
                                                          const std::string &aNameDepXFile, tIm &aImDepX,
                                                          tDIm *&aDImDepX, tPt2di &aSzImDepX,
                                                          const std::string &aNameDepYFile, tIm &aImDepY,
                                                          tDIm *&aDImDepY, tPt2di &aSzImDepY,
                                                          const std::string &aNameCorrelationMask,
                                                          tIm &aImCorrelationMask, tDIm *&aDImCorrelationMask,
                                                          tPt2di &aSzCorrelationMask)
    {
        tDenseVect aVInitTranslation(2 * aDelTri.NbPts(), eModeInitImage::eMIA_Null);

        ReadImageFileNameLoadData(aNameDepXFile, aImDepX,
                                  aDImDepX, aSzImDepX);
        ReadImageFileNameLoadData(aNameDepYFile, aImDepY,
                                  aDImDepY, aSzImDepY);

        ReadImageFileNameLoadData(aNameCorrelationMask, aImCorrelationMask,
                                  aDImCorrelationMask, aSzCorrelationMask);

        for (size_t aTr = 0; aTr < aDelTri.NbFace(); aTr++)
        {
            const tTri2dr aInitTriTr = aDelTri.KthTri(aTr);
            const tPt3di aInitIndicesOfTriKnots = aDelTri.KthFace(aTr);

            //----------- index of unknown, finds the associated pixels of current triangle
            std::vector<int> aInitVecInd;
            GetIndicesVector(aInitVecInd, aInitIndicesOfTriKnots, 2);

            // Get nodes associated to triangle
            const cNodeOfTriangles aFirstInitPointOfTri = cNodeOfTriangles(aVInitTranslation, aInitVecInd,
                                                                           0, 1, 0, 1, aInitTriTr, 0);
            const cNodeOfTriangles aSecondInitPointOfTri = cNodeOfTriangles(aVInitTranslation, aInitVecInd,
                                                                            2, 3, 2, 3, aInitTriTr, 1);
            const cNodeOfTriangles aThirdInitPointOfTri = cNodeOfTriangles(aVInitTranslation, aInitVecInd,
                                                                           4, 5, 4, 5, aInitTriTr, 2);

            aVInitTranslation(aInitVecInd.at(0)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepX,
                                                                                    aFirstInitPointOfTri, 0);
            aVInitTranslation(aInitVecInd.at(1)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepY,
                                                                                    aFirstInitPointOfTri, 0);
            aVInitTranslation(aInitVecInd.at(2)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepX,
                                                                                    aSecondInitPointOfTri, 0);
            aVInitTranslation(aInitVecInd.at(3)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepY,
                                                                                    aSecondInitPointOfTri, 0);
            aVInitTranslation(aInitVecInd.at(4)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepX,
                                                                                    aThirdInitPointOfTri, 0);
            aVInitTranslation(aInitVecInd.at(5)) = ReturnCorrectInitialisationValue(aDImCorrelationMask, aDImDepY,
                                                                                    aThirdInitPointOfTri, 0);
        }

        aSysTranslation = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInitTranslation);
    }

    void InitialiseInterpolationAndEquationRadiometry(cCalculator<tREAL8> *&aEqRadiometryTri, cDiffInterpolator1D *&aInterpolRad,
                                                      const std::vector<std::string> &aArgsVectorInterpolRad, const bool aUseLinearGradInterpolation)
    {
        if (aUseLinearGradInterpolation)
            aInterpolRad = cDiffInterpolator1D::AllocFromNames(aArgsVectorInterpolRad);

        aEqRadiometryTri = aUseLinearGradInterpolation ? EqDeformTriRadiometryLinearGrad(true, 1) : EqDeformTriRadiometryBilin(true, 1); // true means with derivative, 1 is size of buffer
    }

    void InitialiseWithUserValuesRadiometry(const cTriangulation2D<tREAL8> &aDelaunayTri,
                                            cResolSysNonLinear<tREAL8> *&aSysRadiometry,
                                            const bool aUserInitialisation,
                                            const tREAL8 aRadTranslationInitVal,
                                            const tREAL8 aRadScaleInitVal)
    {
        const size_t aNumberPts = 2 * aDelaunayTri.NbPts();
        tDenseVect aVInitRadiometry(aNumberPts, eModeInitImage::eMIA_Null);

        if (aUserInitialisation && aRadTranslationInitVal != 0 && aRadScaleInitVal != 1)
        {
            for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
            {
                if (aKtNumber % 2 == 0 && aRadTranslationInitVal != 0)
                    aVInitRadiometry(aKtNumber) = aRadTranslationInitVal;
                if (aKtNumber % 2 == 1)
                    aVInitRadiometry(aKtNumber) = aRadScaleInitVal;
            }
        }
        else
        {
            for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
            {
                if (aKtNumber % 2 == 1)
                    aVInitRadiometry(aKtNumber) = 1;
            }
        }

        aSysRadiometry = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense, aVInitRadiometry);
    }

    std::unique_ptr<cNodeOfTriangles> DefineNewTriangleNode(const tDenseVect &aVecSol, const std::vector<int> &aIndVec, const int aXInd,
                                                            const int aYInd, const int aRadTrInd, const int aRadScInd, const tTri2dr &aTriangle,
                                                            const tPt3di &aFace, const int aPointNumberInTri, const int anIdOfPoint)
    {
        std::unique_ptr<cNodeOfTriangles> aNewTriangleNode = std::make_unique<cNodeOfTriangles>(aVecSol, aIndVec, aXInd, aYInd, aRadTrInd, aRadScInd, aTriangle, aFace,
                                                                                                aPointNumberInTri, anIdOfPoint);
        return aNewTriangleNode;
    }

    bool CheckValidCorrelationValue(tDIm *&aMask, const cNodeOfTriangles &aPtOfTri)
    {
        // const tPt2di aCoordKt = tPt2di(aPtOfTri.GetInitialNodeCoordinates().x(), aPtOfTri.GetInitialNodeCoordinates().y());
        const tPt2dr aCoordNode = aPtOfTri.GetInitialNodeCoordinates();
        bool aIsValidCorrelPoint;
        if (aMask->InsideBL(aCoordNode))
            (aMask->GetVBL(aCoordNode) == 1) ? aIsValidCorrelPoint = true : aIsValidCorrelPoint = false;
        else
            aIsValidCorrelPoint = false;
        return aIsValidCorrelPoint;
    }

    tREAL8 ReturnCorrectInitialisationValue(tDIm *&aMask, tDIm *&aDispMap,
                                            const cNodeOfTriangles &aPtOfTri, const tREAL8 aValueToReturnIfFalse)
    {
        // const tPt2di aCoordKt = tPt2di(aPtOfTri.GetInitialNodeCoordinates().x(), aPtOfTri.GetInitialNodeCoordinates().y());
        const tPt2dr aCoordNode = aPtOfTri.GetInitialNodeCoordinates();
        // Check if correlation is computed for the point
        const bool aPointIsValid = CheckValidCorrelationValue(aMask, aPtOfTri);
        tREAL8 aInitialisationValue;
        if (aDispMap->InsideBL(aCoordNode))
            (aPointIsValid) ? aInitialisationValue = aDispMap->GetVBL(aCoordNode) : aInitialisationValue = aValueToReturnIfFalse;
        else
            aInitialisationValue = aValueToReturnIfFalse;
        return aInitialisationValue;
    }

    tPt2dr CheckReturnOfBilinearValue(tDIm *&aDImDispXMap, tDIm *&aDImDispYMap,
                                      const tPt2dr &aRealCoordPoint, const tPt2di &aIntCoordPoint)
    {
        const bool aRealCoordPointInsideDispX = aDImDispXMap->InsideBL(aRealCoordPoint);
        const bool aRealCoordPointInsideDispY = aDImDispYMap->InsideBL(aRealCoordPoint);
        if (aRealCoordPointInsideDispX && aRealCoordPointInsideDispY)
            return tPt2dr(aDImDispXMap->GetVBL(aRealCoordPoint), aDImDispYMap->GetVBL(aRealCoordPoint));
        else if (aRealCoordPointInsideDispX && !aRealCoordPointInsideDispY)
            return tPt2dr(aDImDispXMap->GetVBL(aRealCoordPoint), aDImDispYMap->GetV(aIntCoordPoint));
        else if (!aRealCoordPointInsideDispX && aRealCoordPointInsideDispY)
            return tPt2dr(aDImDispXMap->GetV(aIntCoordPoint), aDImDispYMap->GetVBL(aRealCoordPoint));
        else if (!aRealCoordPointInsideDispX && !aRealCoordPointInsideDispY)
            return tPt2dr(aDImDispXMap->GetV(aIntCoordPoint), aDImDispYMap->GetV(aIntCoordPoint));
        else
            return tPt2dr(0, 0); // so compiler accepts function
    }

    bool CheckFolderExistence(const std::string &aUserDefinedFolderNameToSaveResult)
    {
        bool aNonEmptyFolderName = false;
        if (!aUserDefinedFolderNameToSaveResult.empty())
        {
            aNonEmptyFolderName = true;
            if (!ExistFile(aUserDefinedFolderNameToSaveResult))
                CreateDirectories(aUserDefinedFolderNameToSaveResult, aNonEmptyFolderName);
        }
        return aNonEmptyFolderName;
    }

    void GetIndicesVector(std::vector<int> &aVecInd, const tPt3di &aIndicesOfTriKnots, const int aIsTwoOrFour)
    {
        if (aIsTwoOrFour == 2)
            aVecInd = {2 * aIndicesOfTriKnots.x(), 2 * aIndicesOfTriKnots.x() + 1,
                       2 * aIndicesOfTriKnots.y(), 2 * aIndicesOfTriKnots.y() + 1,
                       2 * aIndicesOfTriKnots.z(), 2 * aIndicesOfTriKnots.z() + 1};
        else
            aVecInd = {4 * aIndicesOfTriKnots.x(), 4 * aIndicesOfTriKnots.x() + 1,
                       4 * aIndicesOfTriKnots.x() + 2, 4 * aIndicesOfTriKnots.x() + 3,
                       4 * aIndicesOfTriKnots.y(), 4 * aIndicesOfTriKnots.y() + 1,
                       4 * aIndicesOfTriKnots.y() + 2, 4 * aIndicesOfTriKnots.y() + 3,
                       4 * aIndicesOfTriKnots.z(), 4 * aIndicesOfTriKnots.z() + 1,
                       4 * aIndicesOfTriKnots.z() + 2, 4 * aIndicesOfTriKnots.z() + 3};
    }

    void SubtractPrePostImageAndComputeAvgAndMax(tIm &aImDiff, tDIm *&aDImDiff, tDIm *&aDImPre,
                                                 tDIm *&aDImPost, const tPt2di &aSzImPre)
    {
        aImDiff = tIm(aSzImPre);
        aDImDiff = &aImDiff.DIm();

        for (const tPt2di &aDiffPix : *aDImDiff)
            aDImDiff->SetV(aDiffPix, aDImPre->GetV(aDiffPix) - aDImPost->GetV(aDiffPix));
        const int aNumberOfPixelsInImage = aSzImPre.x() * aSzImPre.y();

        tREAL4 aSumPixelValuesInDiffImage = 0;
        tREAL4 aMaxPixelValuesInDiffImage = 0;
        tREAL4 aDiffImPixelValue = 0;

        for (const tPt2di &aDiffPix : *aDImDiff)
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

    void ReadImageFileNameLoadData(const std::string &aImageFilename, tIm &aImage,
                                   tDIm *&aDataImage, tPt2di &aSzIm)
    {
        aImage = tIm::FromFile(aImageFilename);

        aDataImage = &aImage.DIm();
        aSzIm = aDataImage->Sz();
    }

    void LoadPrePostImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage, const tIm &aImPre, const tIm &aImPost)
    {
        (aPreOrPostImage == "pre") ? aCurIm = aImPre : aCurIm = aImPost;
        aCurDIm = &aCurIm.DIm();
    }

    void InitialiseDisplacementMapsAndOutputImage(const tPt2di &aSzImIn, tIm &aImOut, tDIm *&aDImOut, tPt2di &aSzImOut)
    {
        aImOut = tIm(aSzImIn, 0, eModeInitImage::eMIA_Null);
        aDImOut = &aImOut.DIm();
        aSzImOut = aDImOut->Sz();
    }

    tPt2dr LoadNodeAndReturnCurrentDisplacement(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri)
    {
        const cNodeOfTriangles aTriNode = cNodeOfTriangles(aVCurSol, aVecInd, aXDispInd, aYDispInd,
                                                           aRadTrInd, aRadScInd, aTri, aPtInNumberTri);
        return aTriNode.GetCurrentXYDisplacementValues(); // current translation of node
    }

    tREAL8 LoadNodeAndReturnCurrentRadiometryTranslation(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                         const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                         const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri)
    {
        const cNodeOfTriangles aTriNode = cNodeOfTriangles(aVCurSol, aVecInd, aXDispInd, aYDispInd,
                                                           aRadTrInd, aRadScInd, aTri, aPtInNumberTri);
        return aTriNode.GetCurrentRadiometryTranslation(); // current radiometry translation of node
    }

    tREAL8 LoadNodeAndReturnCurrentRadiometryScaling(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                     const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                     const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri)
    {
        const cNodeOfTriangles aTriNode = cNodeOfTriangles(aVCurSol, aVecInd, aXDispInd, aYDispInd,
                                                           aRadTrInd, aRadScInd, aTri, aPtInNumberTri);
        return aTriNode.GetCurrentRadiometryScaling(); // current radiometry scaling of node
    }

    tPt2dr LoadNodeAppendVectorAndReturnCurrentDisplacement(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                            const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                            const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
                                                            const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
                                                            std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes)
    {
        std::unique_ptr<cNodeOfTriangles> aNodeOfTri = DefineNewTriangleNode(aVCurSol, aVecInd, aXDispInd, aYDispInd, aRadTrInd, aRadScInd,
                                                                             aTri, aFace, aPtInNumberTri, aNodeCounter);
        if (anAppend)
            aVectorOfTriangleNodes->PushInVector(aNodeOfTri);
        return aNodeOfTri->GetCurrentXYDisplacementValues(); // current translation of node
    }

    tREAL8 LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                                     const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                                     const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
                                                                     const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
                                                                     std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes)
    {
        std::unique_ptr<cNodeOfTriangles> aNodeOfTri = DefineNewTriangleNode(aVCurSol, aVecInd, aXDispInd, aYDispInd, aRadTrInd, aRadScInd,
                                                                             aTri, aFace, aPtInNumberTri, aNodeCounter);
        if (anAppend)
            aVectorOfTriangleNodes->PushInVector(aNodeOfTri);
        return aNodeOfTri->GetCurrentRadiometryTranslation(); // current radiometry translation of node
    }

    tREAL8 LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(const tDenseVect &aVCurSol, const std::vector<int> &aVecInd,
                                                                 const int aXDispInd, const int aYDispInd, const int aRadTrInd,
                                                                 const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
                                                                 const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
                                                                 std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes)
    {
        std::unique_ptr<cNodeOfTriangles> aNodeOfTri = DefineNewTriangleNode(aVCurSol, aVecInd, aXDispInd, aYDispInd, aRadTrInd, aRadScInd,
                                                                             aTri, aFace, aPtInNumberTri, aNodeCounter);
        if (anAppend)
            aVectorOfTriangleNodes->PushInVector(aNodeOfTri);
        return aNodeOfTri->GetCurrentRadiometryScaling(); // current radiometry scaling of node
    }

    bool ManageDifferentCasesOfEndIterations(const int aIterNumber, const int aNumberOfScales, const int aNumberOfEndIterations,
                                             bool aIsLastIters, const tIm &aImPre, const tIm &aImPost, tIm &aCurPreIm, tDIm *&aCurPreDIm,
                                             tIm &aCurPostIm, tDIm *&aCurPostDIm)
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

    void ApplyHardConstraintToUnknown(const int anIndices, const std::vector<int> &aVecInd,
                                      const tDenseVect &aVCurSol, cResolSysNonLinear<tREAL8> *&aSys)
    {
        const int aIndicesToFreeze = aVecInd.at(anIndices);
        aSys->SetFrozenVar(aIndicesToFreeze, aVCurSol(aIndicesToFreeze));
    }

    void ApplyHardConstraintsToMultipleUnknowns(const int aFirstIndToFreeze, const int aSecondIndToFreeze,
                                                const int aThirdIndToFreeze, const std::vector<int> &aVecInd,
                                                const tDenseVect &aVCurSol, cResolSysNonLinear<tREAL8> *&aSys)
    {
        ApplyHardConstraintToUnknown(aFirstIndToFreeze, aVecInd, aVCurSol, aSys);
        ApplyHardConstraintToUnknown(aSecondIndToFreeze, aVecInd, aVCurSol, aSys);
        ApplyHardConstraintToUnknown(aThirdIndToFreeze, aVecInd, aVCurSol, aSys);
    }

    void UnfreezeUnknown(const int anIndices, const std::vector<int> &aVecInd,
                         cResolSysNonLinear<tREAL8> *&aSys)
    {
        const int anIndicesToUnfreeze = aVecInd.at(anIndices);
        aSys->SetUnFrozen(anIndicesToUnfreeze);
    }

    void UnfreezeMultipleUnknowns(const int aFirstIndToUnfreeze, const int aSecondIndToUnfreeze,
                                  const int aThirdIndToUnfreeze, const std::vector<int> &aVecInd,
                                  cResolSysNonLinear<tREAL8> *&aSys)
    {
        UnfreezeUnknown(aFirstIndToUnfreeze, aVecInd, aSys);
        UnfreezeUnknown(aSecondIndToUnfreeze, aVecInd, aSys);
        UnfreezeUnknown(aThirdIndToUnfreeze, aVecInd, aSys);
    }

    void ApplySoftConstraintToUnknown(const int aSolStart, const int aSolStep, const std::vector<int> &aVecInd,
                                      cResolSysNonLinear<tREAL8> *&aSys, const tDenseVect &aVCurSol, const tREAL8 aWeight)
    {
        for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size() - 1; aIndCurSol += aSolStep)
        {
            const int aIndices = aVecInd.at(aIndCurSol);
            aSys->AddEqFixVar(aIndices, aVCurSol(aIndices), aWeight);
        }
    }

    tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aCurrentTranslationPointA,
                                                          const tPt2dr &aCurrentTranslationPointB,
                                                          const tPt2dr &aCurrentTranslationPointC,
                                                          const tDoubleVect &aVObs)
    {
        // apply current barycenter translation formula for x and y on current observations.
        const tREAL8 aXTriCoord = aVObs[0] + aVObs[2] * aCurrentTranslationPointA.x() + aVObs[3] * aCurrentTranslationPointB.x() +
                                  aVObs[4] * aCurrentTranslationPointC.x();
        const tREAL8 aYTriCoord = aVObs[1] + aVObs[2] * aCurrentTranslationPointA.y() + aVObs[3] * aCurrentTranslationPointB.y() +
                                  aVObs[4] * aCurrentTranslationPointC.y();

        const tPt2dr aCurrentTranslatedPixel = tPt2dr(aXTriCoord, aYTriCoord);

        return aCurrentTranslatedPixel;
    }

    tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aCurrentTranslationPointA,
                                                          const tPt2dr &aCurrentTranslationPointB,
                                                          const tPt2dr &aCurrentTranslationPointC,
                                                          const cPtInsideTriangles &aPixInsideTriangle)
    {
        const tPt2dr aCartesianCoordinates = aPixInsideTriangle.GetCartesianCoordinates();
        const cPt3dr aBarycenterCoordinates = aPixInsideTriangle.GetBarycenterCoordinates();
        // apply current barycenter translation formula for x and y on current observations.
        const tREAL8 aXTriCoord = aCartesianCoordinates.x() + aBarycenterCoordinates.x() * aCurrentTranslationPointA.x() +
                                  aBarycenterCoordinates.y() * aCurrentTranslationPointB.x() + aBarycenterCoordinates.z() * aCurrentTranslationPointC.x();
        const tREAL8 aYTriCoord = aCartesianCoordinates.y() + aBarycenterCoordinates.x() * aCurrentTranslationPointA.y() +
                                  aBarycenterCoordinates.y() * aCurrentTranslationPointB.y() + aBarycenterCoordinates.z() * aCurrentTranslationPointC.y();

        const tPt2dr aTranslatedPixel = tPt2dr(aXTriCoord, aYTriCoord);

        return aTranslatedPixel;
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

    tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                     const tREAL8 aCurrentRadTranslationPointB,
                                                                     const tREAL8 aCurrentRadTranslationPointC,
                                                                     const cPtInsideTriangles &aPixInsideTriangle)
    {
        const cPt3dr BarycenterCoordinateOfPoint = aPixInsideTriangle.GetBarycenterCoordinates();
        const tREAL8 aCurentRadTranslation = BarycenterCoordinateOfPoint.x() * aCurrentRadTranslationPointA + BarycenterCoordinateOfPoint.y() * aCurrentRadTranslationPointB +
                                             BarycenterCoordinateOfPoint.z() * aCurrentRadTranslationPointC;
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

    tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                 const tREAL8 aCurrentRadScalingPointB,
                                                                 const tREAL8 aCurrentRadScalingPointC,
                                                                 const cPtInsideTriangles &aLastPixInsideTriangle)
    {
        const cPt3dr BarycenterCoordinateOfPoint = aLastPixInsideTriangle.GetBarycenterCoordinates();
        const tREAL8 aCurrentRadScaling = BarycenterCoordinateOfPoint.x() * aCurrentRadScalingPointA + BarycenterCoordinateOfPoint.y() * aCurrentRadScalingPointB +
                                          BarycenterCoordinateOfPoint.z() * aCurrentRadScalingPointC;
        return aCurrentRadScaling;
    }

    void SetValueIfTranslatedOutOfImage(const tREAL8 anIntensityValue, const tREAL8 aLastXTranslatedCoord,
                                        const tREAL8 aLastYTranslatedCoord, const tPt2di &aLastCoordinate,
                                        const tPt2di &aSzImOut, tDIm *&aDImOut)
    {
        // Build image with intensities displaced
        // deal with different cases of pixel being translated out of image
        if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord < 0)
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(0, 0)));
        else if (aLastXTranslatedCoord >= aSzImOut.x() && aLastYTranslatedCoord >= aSzImOut.y())
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(aSzImOut.x() - 1, aSzImOut.y() - 1)));
        else if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord >= aSzImOut.y())
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(0, aSzImOut.y() - 1)));
        else if (aLastXTranslatedCoord >= aSzImOut.x() && aLastYTranslatedCoord < 0)
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(aSzImOut.x() - 1, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < aSzImOut.x() &&
                 aLastYTranslatedCoord < 0)
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(aLastXTranslatedCoord, 0)));
        else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < aSzImOut.x() &&
                 aLastYTranslatedCoord > aSzImOut.y())
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(aLastXTranslatedCoord, aSzImOut.y() - 1)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < aSzImOut.y() &&
                 aLastXTranslatedCoord < 0)
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(0, aLastYTranslatedCoord)));
        else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < aSzImOut.y() &&
                 aLastXTranslatedCoord > aSzImOut.x())
            aDImOut->SetV(aLastCoordinate, aDImOut->GetV(tPt2di(aSzImOut.x() - 1, aLastYTranslatedCoord)));
        else
            // at the translated pixel the untranslated pixel value is given computed with the right radiometry values
            aDImOut->SetV(tPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord), anIntensityValue);
    }

    void FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                            const tPt2dr &aLastTranslatedFilledPoint,
                                            const tREAL8 aLastRadiometryTranslation,
                                            const tREAL8 aLastRadiometryScaling, const tPt2di &aSzImOut,
                                            tDIm *&aDImDepX, tDIm *&aDImDepY, tDIm *&aDImOut, tDIm *&aDImPost)
    { /*
      const tPt2dr aLastRealCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates();
      const tREAL8 aLastXCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().x();
      const tREAL8 aLastYCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().y();
      const tREAL8 aLastPixelValue = aLastPixInsideTriangle.GetPixelValue();

      const tPt2di aLastIntCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

      aDImDepX->SetV(aLastIntCoordinate,
                     aLastTranslatedFilledPoint.x() - aLastXCoordinate);
      aDImDepY->SetV(aLastIntCoordinate,
                     aLastTranslatedFilledPoint.y() - aLastYCoordinate);

      tREAL8 aLastXTranslatedCoord = aLastXCoordinate + aDImDepX->GetV(aLastIntCoordinate);
      tREAL8 aLastYTranslatedCoord = aLastYCoordinate + aDImDepY->GetV(aLastIntCoordinate);
      if (aDImDepX->InsideBL(aLastRealCoordinate))
          aLastXTranslatedCoord = aLastXCoordinate + aDImDepX->GetVBL(aLastRealCoordinate);
      if (aDImDepY->InsideBL(aLastRealCoordinate))
          aLastYTranslatedCoord = aLastYCoordinate + aDImDepY->GetVBL(aLastRealCoordinate);

      const tREAL8 aLastRadiometryValue = aLastRadiometryScaling * aLastPixelValue +
                                          aLastRadiometryTranslation;

      SetValueIfTranslatedOutOfImage(aLastRadiometryValue, aLastXTranslatedCoord, aLastYTranslatedCoord,
                                     aLastIntCoordinate, aSzImOut, aDImOut);
      */
        const tPt2dr aLastRealCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates();
        const tREAL8 aLastXCoordinate = aLastRealCoordinate.x();
        const tREAL8 aLastYCoordinate = aLastRealCoordinate.y();

        const tPt2di aLastIntCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

        aDImDepX->SetV(aLastIntCoordinate,
                       aLastTranslatedFilledPoint.x() - aLastXCoordinate);
        aDImDepY->SetV(aLastIntCoordinate,
                       aLastTranslatedFilledPoint.y() - aLastYCoordinate);

        tREAL8 aLastXTranslatedCoord = aLastXCoordinate + aDImDepX->GetV(aLastIntCoordinate);
        tREAL8 aLastYTranslatedCoord = aLastYCoordinate + aDImDepY->GetV(aLastIntCoordinate);
        if (aDImDepX->InsideBL(aLastRealCoordinate))
            aLastXTranslatedCoord = aLastXCoordinate + aDImDepX->GetVBL(aLastRealCoordinate);
        if (aDImDepY->InsideBL(aLastRealCoordinate))
            aLastYTranslatedCoord = aLastYCoordinate + aDImDepY->GetVBL(aLastRealCoordinate);

        tREAL8 aLastPixelValue = aDImPost->GetV(tPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord));

        if (aDImOut->InsideBL(tPt2dr(aLastXTranslatedCoord, aLastYTranslatedCoord)))
            aLastPixelValue = aDImPost->GetVBL(tPt2dr(aLastXTranslatedCoord, aLastYTranslatedCoord));

        const tREAL8 aLastRadiometryValue = (aLastPixelValue - aLastRadiometryTranslation) / aLastRadiometryScaling;

        aDImOut->SetV(aLastIntCoordinate, aLastRadiometryValue);
        /*
        SetValueIfTranslatedOutOfImage(aLastRadiometryValue, aLastXTranslatedCoord, aLastYTranslatedCoord,
                                       aLastIntCoordinate, aSzImOut, aDImOut);
        */
    }

    void FillDisplacementMapsTranslation(const cPtInsideTriangles &aLastPixInsideTriangle,
                                         const tPt2dr &aLastTranslatedFilledPoint, const tPt2di &aSzImOut,
                                         tDIm *&aDImDepX, tDIm *&aDImDepY, tDIm *&aDImOut)
    {
        const tPt2dr aLastRealCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates();
        const tREAL8 aLastXCoordinate = aLastRealCoordinate.x();
        const tREAL8 aLastYCoordinate = aLastRealCoordinate.y();
        const tREAL8 aLastPixelValue = aLastPixInsideTriangle.GetPixelValue();

        const tPt2di aLastIntCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

        aDImDepX->SetV(aLastIntCoordinate,
                       aLastTranslatedFilledPoint.x() - aLastXCoordinate);
        aDImDepY->SetV(aLastIntCoordinate,
                       aLastTranslatedFilledPoint.y() - aLastYCoordinate);

        tREAL8 aLastXTranslatedCoord = aLastXCoordinate + aDImDepX->GetV(aLastIntCoordinate);
        tREAL8 aLastYTranslatedCoord = aLastYCoordinate + aDImDepY->GetV(aLastIntCoordinate);
        if (aDImDepX->InsideBL(aLastRealCoordinate))
            aLastXTranslatedCoord = aLastXCoordinate + aDImDepX->GetVBL(aLastRealCoordinate);
        if (aDImDepY->InsideBL(aLastRealCoordinate))
            aLastYTranslatedCoord = aLastYCoordinate + aDImDepY->GetVBL(aLastRealCoordinate);

        SetValueIfTranslatedOutOfImage(aLastPixelValue, aLastXTranslatedCoord, aLastYTranslatedCoord,
                                       aLastIntCoordinate, aSzImOut, aDImOut);
    }

    void FillOutputImageRadiometry(const cPtInsideTriangles &aLastPixInsideTriangle,
                                   const tREAL8 aLastRadiometryTranslation,
                                   const tREAL8 aLastRadiometryScaling,
                                   tDIm *&aDImOut)
    {
        const tREAL8 aLastXCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().x();
        const tREAL8 aLastYCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().y();

        const tPt2di aLastCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

        const tREAL8 aLastRadiometryValue = aLastRadiometryScaling * aLastPixInsideTriangle.GetPixelValue() +
                                            aLastRadiometryTranslation;

        // Build image with radiometric intensities
        aDImOut->SetV(aLastCoordinate, aLastRadiometryValue);
    }

    void FillDiffDisplacementMap(tDIm *&aDImDispMap, tDIm *&aDImDiffMap,
                                 tDIm *&aDImTranslatedDispMap,
                                 const cPtInsideTriangles &aPixInsideTriangle,
                                 const tREAL8 aPixInsideTriangleCoord,
                                 const tREAL8 aTranslatedDispMapPointCoord,
                                 const bool aComputeDiffDispMaps, const bool aComputeInterpTranslationDispMaps)
    {
        const tPt2di aIntCoordinate = tPt2di(aPixInsideTriangle.GetCartesianCoordinates().x(), aPixInsideTriangle.GetCartesianCoordinates().y());
        const tREAL8 aDiffBarycentricInterpTranslation = aTranslatedDispMapPointCoord - aPixInsideTriangleCoord;
        if (aComputeDiffDispMaps)
            // compute difference between value of pixel in ground-truth map and value of barycentric interpolated pixel from displacement map
            aDImDiffMap->SetV(aIntCoordinate, std::abs(aDiffBarycentricInterpTranslation - aPixInsideTriangle.GetPixelValue()));
        if (aComputeInterpTranslationDispMaps)
            // Sets value to difference between pixel's original position in displacement map and interpolated value by barycentric interpolation formula
            aDImTranslatedDispMap->SetV(aIntCoordinate, aDiffBarycentricInterpTranslation);
    }

    void SaveMultiScaleDisplacementMapsToFile(tDIm *&aDImDepX, tDIm *&aDImDepY, const bool aUserDefinedFolderName,
                                              const std::string &aFolderPathToSave, const std::string &aDepXFileNameToSave,
                                              const std::string &aDepYFileNameToSave, const int aIterNumber,
                                              const int aNumberOfPointsToGenerate, const int aTotalNumberOfIterations)
    {
        if (aUserDefinedFolderName)
        {
            aDImDepX->ToFile(aFolderPathToSave + "/" + aDepXFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            aDImDepY->ToFile(aFolderPathToSave + "/" + aDepYFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
        }
        else
        {
            aDImDepX->ToFile(aDepXFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            aDImDepY->ToFile(aDepYFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
        }
    }

    void SaveFinalDisplacementMapsToFile(tDIm *&aDImDepX, tDIm *&aDImDepY, const bool aUserDefinedFolderName,
                                         const std::string &aFolderPathToSave, const std::string &aDepXFileNameToSave,
                                         const std::string &aDepYFileNameToSave, const int aNumberOfPointsToGenerate,
                                         const int aTotalNumberOfIterations)
    {
        if (aUserDefinedFolderName)
        {
            aDImDepX->ToFile(aFolderPathToSave + "/" + aDepXFileNameToSave + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            aDImDepY->ToFile(aFolderPathToSave + "/" + aDepYFileNameToSave + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
        }
        else
        {
            aDImDepX->ToFile(aDepXFileNameToSave + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
            aDImDepY->ToFile(aDepYFileNameToSave + "_" +
                             ToStr(aNumberOfPointsToGenerate) + "_" +
                             ToStr(aTotalNumberOfIterations) + ".tif");
        }
    }

    void SaveOutputImageToFile(tDIm *&aDImOut, const bool aUserDefinedFolderName, const std::string &aFolderPathToSave,
                               const std::string &aOutputImageFileNameToSave,
                               const int aNumberOfPointsToGenerate, const int aTotalNumberOfIterations)
    {
        if (aUserDefinedFolderName)
            aDImOut->ToFile(aFolderPathToSave + "/" + aOutputImageFileNameToSave + ToStr(aNumberOfPointsToGenerate) + "_" +
                            ToStr(aTotalNumberOfIterations) + ".tif");
        else
            aDImOut->ToFile(aOutputImageFileNameToSave + "_" + ToStr(aNumberOfPointsToGenerate) + "_" +
                            ToStr(aTotalNumberOfIterations) + ".tif");
    }

    tREAL8 CheckMinMaxValueIsChanged(const tREAL8 aMinMaxValue, tREAL8 aPossiblyChangedValue, const int aNbChanges)
    {
        if (aNbChanges > 0)
            return aPossiblyChangedValue;
        else
        {
            aPossiblyChangedValue = aMinMaxValue;
            return aPossiblyChangedValue;
        }
    }

    void DisplayFirstAndLastUnknownValuesAndComputeStatisticsTwoUnknowns(const tDenseVect &aVFinalSol, const tDenseVect &aVInitSol,
                                                                         const int aNbUnk)
    {
        tREAL8 aMaxFirstUnk = INT_MAX, aMinFirstUnk = INT_MIN, aMeanFirstUnk = 0, aVarianceMeanFirstUnk = 0;     // aVarFirstUnk = 0;
        tREAL8 aMaxSecondUnk = INT_MAX, aMinSecondUnk = INT_MIN, aMeanSecondUnk = 0, aVarianceMeanSecondUnk = 0; // aVarSecondUnk = 0;
        const tREAL8 aVecSolSz = aVFinalSol.DIm().Sz();
        const tREAL8 aDividedVecSolSz = aVecSolSz / aNbUnk;

        int aNbChangesMinFirstUnk = 0;
        int aNbChangesMaxFirstUnk = 0;
        int aNbChangesMinSecondUnk = 0;
        int aNbChangesMaxSecondUnk = 0;

        for (int aFinalUnk = 0; aFinalUnk < aVecSolSz; aFinalUnk++)
        {
            const tREAL8 aFinalSolValue = aVFinalSol(aFinalUnk);
            const tREAL8 aInitSolValue = aVInitSol(aFinalUnk);

            StdOut() << aFinalSolValue << " " << aInitSolValue << " ";
            if (aFinalUnk % aNbUnk == 1 && aFinalUnk != 0)
                StdOut() << std::endl;

            if (aFinalUnk % aNbUnk == 0)
            {
                aMeanFirstUnk += aFinalSolValue;
                if (aFinalSolValue > aMaxFirstUnk)
                {
                    aMaxFirstUnk = aFinalSolValue;
                    aNbChangesMaxFirstUnk++;
                }
                if (aFinalSolValue < aMinFirstUnk)
                {
                    aMinFirstUnk = aFinalSolValue;
                    aNbChangesMinFirstUnk++;
                }
            }
            else
            {
                aMeanSecondUnk += aFinalSolValue;
                if (aFinalSolValue > aMaxSecondUnk)
                {
                    aMaxSecondUnk = aFinalSolValue;
                    aNbChangesMaxSecondUnk++;
                }
                if (aFinalSolValue < aMinSecondUnk)
                {
                    aMinSecondUnk = aFinalSolValue;
                    aNbChangesMinSecondUnk++;
                }
            }
        }

        aMinFirstUnk = CheckMinMaxValueIsChanged(aMaxFirstUnk, aMinFirstUnk, aNbChangesMinFirstUnk);
        aMaxFirstUnk = CheckMinMaxValueIsChanged(aMinFirstUnk, aMaxFirstUnk, aNbChangesMaxFirstUnk);
        aMinSecondUnk = CheckMinMaxValueIsChanged(aMaxSecondUnk, aMinSecondUnk, aNbChangesMinSecondUnk);
        aMaxSecondUnk = CheckMinMaxValueIsChanged(aMinSecondUnk, aMaxSecondUnk, aNbChangesMaxSecondUnk);

        aMeanFirstUnk /= aDividedVecSolSz;
        aMeanSecondUnk /= aDividedVecSolSz;

        /*
        for (tREAL8 var: aVFinalSol)
        {
            if (aFinalUnk % 2 == 0)
                aVarFirstUnk += (var - aMeanFirstUnk) * (var - aMeanFirstUnk);
            if (aFinalUnk % 2 == 0)
                aVarSecondUnk += (var - aMeanSecondUnk) * (var - aMeanSecondUnk);
        }

        aVarFirstUnk /= (aDividedVecSolSz - 1);
        aVarSecondUnk /= (aDividedVecSolSz - 1);
        */

        std::vector<tREAL8> aVarianceVectorFirstUnk;
        std::vector<tREAL8> aVarianceVectorSecondUnk;

        for (int aSolNumber = 0; aSolNumber < aVecSolSz; aSolNumber++)
        {
            const tREAL8 aFinalSolValue = aVFinalSol(aSolNumber);
            if (aSolNumber % aNbUnk == 0)
                aVarianceVectorFirstUnk.push_back(std::abs(aFinalSolValue - aMeanFirstUnk) * std::abs(aFinalSolValue - aMeanFirstUnk));
            else
                aVarianceVectorSecondUnk.push_back(std::abs(aFinalSolValue - aMeanSecondUnk) * std::abs(aFinalSolValue - aMeanSecondUnk));
        }

        for (int aSolNumber = 0; aSolNumber < aDividedVecSolSz; aSolNumber++)
        {
            if (aSolNumber % aNbUnk == 0)
                aVarianceMeanFirstUnk += aVarianceVectorFirstUnk[aSolNumber];
            else
                aVarianceMeanSecondUnk += aVarianceVectorSecondUnk[aSolNumber];
        }

        aVarianceMeanFirstUnk /= aDividedVecSolSz;
        aVarianceMeanSecondUnk /= aDividedVecSolSz;

        StdOut() << "The minimum value for the first unknown is : " << aMinFirstUnk << " and the maximum value is : "
                 << aMaxFirstUnk << std::endl;
        StdOut() << "The minimum  value for the second unknown is : " << aMinSecondUnk << " and the maximum value is : "
                 << aMaxSecondUnk << std::endl;
        StdOut() << "The mean value for the first unknown is : " << aMeanFirstUnk << " and the standard deviation value is : "
                 << std::sqrt(aVarianceMeanFirstUnk) << std::endl;
        StdOut() << "The mean value for the second unknown is : " << aMeanSecondUnk << " and the standard deviation value is : "
                 << std::sqrt(aVarianceMeanSecondUnk) << std::endl;
    }

    void ComputeStatisticsFourUnknowns(const tDenseVect &aVFinalSol, const int aNbUnk)
    {
        tREAL8 aMinFirstUnk = INT_MAX, aMaxFirstUnk = INT_MIN, aMeanFirstUnk = 0, aVarianceMeanFirstUnk = 0;
        tREAL8 aMinSecondUnk = INT_MAX, aMaxSecondUnk = INT_MIN, aMeanSecondUnk = 0, aVarianceMeanSecondUnk = 0;
        tREAL8 aMinThirdUnk = INT_MAX, aMaxThirdUnk = INT_MIN, aMeanThirdUnk = 0, aVarianceMeanThirdUnk = 0;
        tREAL8 aMinFourthUnk = INT_MAX, aMaxFourthUnk = INT_MIN, aMeanFourthUnk = 0, aVarianceMeanFourthUnk = 0;

        const tREAL8 aVecSolSz = aVFinalSol.DIm().Sz();
        const tREAL8 aDividedVecSolSz = aVecSolSz / aNbUnk;

        int aNbChangesMinFirstUnk = 0;
        int aNbChangesMaxFirstUnk = 0;
        int aNbChangesMinSecondUnk = 0;
        int aNbChangesMaxSecondUnk = 0;
        int aNbChangesMinThirdUnk = 0;
        int aNbChangesMaxThirdUnk = 0;
        int aNbChangesMinFourthUnk = 0;
        int aNbChangesMaxFourthUnk = 0;

        for (int aFinalUnk = 0; aFinalUnk < aVecSolSz; aFinalUnk++)
        {
            const tREAL8 aFinalSolValue = aVFinalSol(aFinalUnk);

            if (aFinalUnk % aNbUnk == 0)
            {
                aMeanFirstUnk += aFinalSolValue;
                if (aFinalSolValue > aMaxFirstUnk)
                {
                    aMaxFirstUnk = aFinalSolValue;
                    aNbChangesMaxFirstUnk++;
                }
                if (aFinalSolValue < aMinFirstUnk)
                {
                    aMinFirstUnk = aFinalSolValue;
                    aNbChangesMinFirstUnk++;
                }
            }
            else if (aFinalUnk % aNbUnk == 1)
            {
                aMeanSecondUnk += aFinalSolValue;
                if (aFinalSolValue > aMaxSecondUnk)
                {
                    aMaxSecondUnk = aFinalSolValue;
                    aNbChangesMaxSecondUnk++;
                }
                if (aFinalSolValue < aMinSecondUnk)
                {
                    aMinSecondUnk = aFinalSolValue;
                    aNbChangesMinSecondUnk++;
                }
            }
            else if (aFinalUnk % aNbUnk == 2)
            {
                aMeanThirdUnk += aFinalSolValue;
                if (aFinalSolValue > aMaxThirdUnk)
                {
                    aMaxThirdUnk = aFinalSolValue;
                    aNbChangesMaxThirdUnk++;
                }
                if (aFinalSolValue < aMinThirdUnk)
                {
                    aMinThirdUnk = aFinalSolValue;
                    aNbChangesMinThirdUnk++;
                }
            }
            else
            {
                aMeanFourthUnk += aFinalSolValue;
                if (aFinalSolValue > aMaxFourthUnk)
                {
                    aMaxFourthUnk = aFinalSolValue;
                    aNbChangesMaxFourthUnk++;
                }
                if (aFinalSolValue < aMinFourthUnk)
                {
                    aMinFourthUnk = aFinalSolValue;
                    aNbChangesMinFourthUnk++;
                }
            }
        }

        aMinFirstUnk = CheckMinMaxValueIsChanged(aMaxFirstUnk, aMinFirstUnk, aNbChangesMinFirstUnk);
        aMaxFirstUnk = CheckMinMaxValueIsChanged(aMinFirstUnk, aMaxFirstUnk, aNbChangesMaxFirstUnk);
        aMinSecondUnk = CheckMinMaxValueIsChanged(aMaxSecondUnk, aMinSecondUnk, aNbChangesMinSecondUnk);
        aMaxSecondUnk = CheckMinMaxValueIsChanged(aMinSecondUnk, aMaxSecondUnk, aNbChangesMaxSecondUnk);
        aMinThirdUnk = CheckMinMaxValueIsChanged(aMaxThirdUnk, aMinThirdUnk, aNbChangesMinThirdUnk);
        aMaxThirdUnk = CheckMinMaxValueIsChanged(aMinThirdUnk, aMaxThirdUnk, aNbChangesMaxThirdUnk);
        aMinFourthUnk = CheckMinMaxValueIsChanged(aMaxFourthUnk, aMinFourthUnk, aNbChangesMinFourthUnk);
        aMaxFourthUnk = CheckMinMaxValueIsChanged(aMinFourthUnk, aMaxFourthUnk, aNbChangesMaxFourthUnk);

        aMeanFirstUnk /= aDividedVecSolSz;
        aMeanSecondUnk /= aDividedVecSolSz;
        aMeanThirdUnk /= aDividedVecSolSz;
        aMeanFourthUnk /= aDividedVecSolSz;

        std::vector<tREAL8> aVarianceVectorFirstUnk;
        std::vector<tREAL8> aVarianceVectorSecondUnk;
        std::vector<tREAL8> aVarianceVectorThirdUnk;
        std::vector<tREAL8> aVarianceVectorFourthUnk;

        for (int aSolNumber = 0; aSolNumber < aVecSolSz; aSolNumber++)
        {
            const tREAL8 aFinalSolValue = aVFinalSol(aSolNumber);
            if (aSolNumber % aNbUnk == 0)
                aVarianceVectorFirstUnk.push_back(std::abs(aFinalSolValue - aMeanFirstUnk) * std::abs(aFinalSolValue - aMeanFirstUnk));
            else if (aSolNumber % aNbUnk == 1)
                aVarianceVectorSecondUnk.push_back(std::abs(aFinalSolValue - aMeanSecondUnk) * std::abs(aFinalSolValue - aMeanSecondUnk));
            else if (aSolNumber % aNbUnk == 2)
                aVarianceVectorThirdUnk.push_back(std::abs(aFinalSolValue - aMeanThirdUnk) * std::abs(aFinalSolValue - aMeanThirdUnk));
            else
                aVarianceVectorFourthUnk.push_back(std::abs(aFinalSolValue - aMeanFourthUnk) * std::abs(aFinalSolValue - aMeanFourthUnk));
        }

        for (int aSolNumber = 0; aSolNumber < aDividedVecSolSz; aSolNumber++)
        {
            if (aSolNumber % aNbUnk == 0)
                aVarianceMeanFirstUnk += aVarianceVectorFirstUnk[aSolNumber];
            else if (aSolNumber % aNbUnk == 1)
                aVarianceMeanSecondUnk += aVarianceVectorSecondUnk[aSolNumber];
            else if (aSolNumber % aNbUnk == 2)
                aVarianceMeanThirdUnk += aVarianceVectorThirdUnk[aSolNumber];
            else
                aVarianceMeanFourthUnk += aVarianceVectorFourthUnk[aSolNumber];
        }

        aVarianceMeanFirstUnk /= aDividedVecSolSz;
        aVarianceMeanSecondUnk /= aDividedVecSolSz;
        aVarianceMeanThirdUnk /= aDividedVecSolSz;
        aVarianceMeanFourthUnk /= aDividedVecSolSz;

        StdOut() << "The minimum value for the first unknown is : " << aMinFirstUnk << " and the maximum value is : "
                 << aMaxFirstUnk << std::endl;
        StdOut() << "The minimum value for the second unknown is : " << aMinSecondUnk << " and the maximum value is : "
                 << aMaxSecondUnk << std::endl;
        StdOut() << "The minimum value for the third unknown is : " << aMinThirdUnk << " and the maximum value is : "
                 << aMaxThirdUnk << std::endl;
        StdOut() << "The minimum value for the fourth unknown is : " << aMinFourthUnk << " and the maximum value is : "
                 << aMaxFourthUnk << std::endl;
        StdOut() << "The mean value for the first unknown is : " << aMeanFirstUnk << " and the standard deviation value is : "
                 << std::sqrt(aVarianceMeanFirstUnk) << std::endl;
        StdOut() << "The mean value for the second unknown is : " << aMeanSecondUnk << " and the standard deviation value is : "
                 << std::sqrt(aVarianceMeanSecondUnk) << std::endl;
        StdOut() << "The mean value for the third unknown is : " << aMeanThirdUnk << " and the standard deviation value is : "
                 << std::sqrt(aVarianceMeanThirdUnk) << std::endl;
        StdOut() << "The mean value for the fourth unknown is : " << aMeanFourthUnk << " and the standard deviation value is : "
                 << std::sqrt(aVarianceMeanFourthUnk) << std::endl;
    }

    void DisplayLastUnknownValuesAndComputeStatistics(const tDenseVect &aVFinalSol, const bool aDisplayLastRadiometryValues,
                                                      const bool aDisplayLastTranslationValues, const int aNbUnk)
    {
        if (aDisplayLastTranslationValues && aDisplayLastRadiometryValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        else if (aDisplayLastTranslationValues && !aDisplayLastRadiometryValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                if (aFinalUnk % 4 == 0 || aFinalUnk % 4 == 1)
                    StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        else if (!aDisplayLastTranslationValues && aDisplayLastRadiometryValues)
        {
            for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
            {
                if (aFinalUnk % 4 == 2 || aFinalUnk % 4 == 3)
                    StdOut() << aVFinalSol(aFinalUnk) << " ";
                if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
                    StdOut() << std::endl;
            }
        }
        ComputeStatisticsFourUnknowns(aVFinalSol, aNbUnk);
    }

}; // namespace MMVII