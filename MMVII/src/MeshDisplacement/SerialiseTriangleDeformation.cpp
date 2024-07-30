#include "SerialiseTriangleDeformation.h"

/**
 \file SerialiseDeformation.cpp
 \brief File containing methods to serialise objects
 linked to triangle deformation between optimisation executions
 **/

namespace MMVII
{
    cSerialiseTriangleDeformation::cSerialiseTriangleDeformation()
    {
    }

    cSerialiseTriangleDeformation::cSerialiseTriangleDeformation(const tDenseVect &aVecSol,
                                                                 const tIntVect &aIndicesVec,
                                                                 const int aXIndices,
                                                                 const int aYIndices,
                                                                 const int aRadTrIndices,
                                                                 const int aRadScIndices,
                                                                 const tTri2dr &aTri,
                                                                 const tPt3di &aFace,
                                                                 const int aPointNumberInTri,
                                                                 const int anIdOfPoint) : cNodeOfTriangles(aVecSol, aIndicesVec, aXIndices,
                                                                                                           aYIndices, aRadTrIndices, aRadScIndices,
                                                                                                           aTri, aPointNumberInTri),
                                                                                          mIdOfPt(anIdOfPoint)
    {
        if (aPointNumberInTri == 0)
            mFaceOfTriangle = aFace.x();
        else if (aPointNumberInTri == 1)
            mFaceOfTriangle = aFace.y();
        else
            mFaceOfTriangle = aFace.z();
    }

    cSerialiseTriangleDeformation::~cSerialiseTriangleDeformation()
    {
    }

    std::ostream& operator<<(std::ostream& os, const cSerialiseTriangleDeformation& obj)
    {
        obj.ShowTriangleDeformationObjectCarateristics();
        return os;
    }

    void cSerialiseTriangleDeformation::AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise)
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

    void AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise) { aPtToSerialise.AddData(anAux, aPtToSerialise); }

    void cSerialiseTriangleDeformation::ShowTriangleDeformationObjectCarateristics() const
    {
        StdOut() << "Id of this point : " << this->GetPointId() << std::endl;
        StdOut() << "Face of triangle associated to point : " << this->GetTriangleFace() << std::endl;
        StdOut() << "Initial node coordinates : " << this->GetInitialNodeCoordinates() << "." << std::endl;
        StdOut() << "Current displacement coefficient values : " << this->GetCurrentXYDisplacementValues()
                 << "." << std::endl;
        StdOut() << "Current radiometric coefficient values : " << this->GetCurrentRadiometryTranslation()
                 << " for translation and " << this->GetCurrentRadiometryScaling() << " for scaling." << std::endl;
    }

    void cSerialiseTriangleDeformation::SaveTriangleDeformationObjectToFile() const
    {
        SaveInFile(*this, NameFileToSaveOneObj(mIdOfPt));
    }

    std::unique_ptr<cSerialiseTriangleDeformation> cSerialiseTriangleDeformation::ReadSerialisedTriangleDeformation(const tDenseVect &aVecSol, const tIntVect &aIndVec,
                                                                                                                    const int aXInd, const int aYInd, const int aRadTrInd,
                                                                                                                    const int aRadScInd, const tTri2dr &aTriangle, const tPt3di &aFace,
                                                                                                                    const int aPointNumberInTri, const int anIdOfPoint)
    {
        std::unique_ptr<cSerialiseTriangleDeformation> aReReadSerialisedObj = std::make_unique<cSerialiseTriangleDeformation>(aVecSol, aIndVec, aXInd, aYInd, aRadTrInd, 
                                                                                                                              aRadScInd, aTriangle, aFace, aPointNumberInTri, 
                                                                                                                              anIdOfPoint);
        ReadFromFile(*aReReadSerialisedObj, NameFileToSaveOneObj(anIdOfPoint));

        return aReReadSerialisedObj;
    }

    std::string cSerialiseTriangleDeformation::NameFileToSaveOneObj(const int anId)
    {
        return "Id_" + ToStr(anId) + ".xml";
    }

    int cSerialiseTriangleDeformation::GetPointId() const
    {
        return mIdOfPt;
    }

    int cSerialiseTriangleDeformation::GetTriangleFace() const
    {
        return mFaceOfTriangle;
    }

    int& cSerialiseTriangleDeformation::GetPointId()
    {
        return mIdOfPt;
    }

    int& cSerialiseTriangleDeformation::GetTriangleFace()
    {
        return mFaceOfTriangle;
    }

}; // MMVII
