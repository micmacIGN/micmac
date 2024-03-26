#include "SerialiseTriangleDeformation.h"

/**
 \file SerialiseDeformation.cpp
 \brief File containing methods to serialise objects
 linked to triangle deformation between optimisation executions
 **/

namespace MMVII
{
    cSerialiseTriangleDeformation::cSerialiseTriangleDeformation(const tDenseVect &aVecSol,
                                                                 const tIntVect &aIndicesVec,
                                                                 const int aXIndices,
                                                                 const int aYIndices,
                                                                 const tREAL8 aRadTrIndices,
                                                                 const tREAL8 aRadScIndices,
                                                                 const tTri2dr &aTri,
                                                                 const int aPointNumberInTri,
                                                                 const int anIdOfPoint,
                                                                 const cTriangulation2D<tREAL8> &aDelTri) : cNodeOfTriangles(aVecSol, aIndicesVec, aXIndices,
                                                                                                                             aYIndices, aRadTrIndices, aRadScIndices,
                                                                                                                             aTri, aPointNumberInTri),
                                                                                                            mIdOfPt(anIdOfPoint)

    {
    }

    cSerialiseTriangleDeformation::~cSerialiseTriangleDeformation()
    {
    }

    void cSerialiseTriangleDeformation::AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise)
    {
        MMVII::AddData(cAuxAr2007("Id", anAux), mIdOfPt);
        MMVII::AddData(cAuxAr2007("x", anAux), aPtToSerialise.GetInitialNodeCoordinates().x());
        MMVII::AddData(cAuxAr2007("y", anAux), aPtToSerialise.GetInitialNodeCoordinates().y());
        MMVII::AddData(cAuxAr2007("dx", anAux), aPtToSerialise.GetCurrentXYDisplacementVector().x());
        MMVII::AddData(cAuxAr2007("dy", anAux), aPtToSerialise.GetCurrentXYDisplacementVector().y());
        MMVII::AddData(cAuxAr2007("RadiometryTranslation", anAux), aPtToSerialise.GetCurrentRadiometryTranslation());
        MMVII::AddData(cAuxAr2007("RadiometryScaling", anAux), aPtToSerialise.GetCurrentRadiometryScaling());
    }

    void AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise) { aPtToSerialise.AddData(anAux, aPtToSerialise); }

    void cSerialiseTriangleDeformation::SaveAuto() const
    {
        SaveInFile(*this, NameFile(mIdOfPt));
    }

    std::unique_ptr<cSerialiseTriangleDeformation> cSerialiseTriangleDeformation::ReadSerialisedTriangleDeformation(int anId)
    {
        std::unique_ptr<cSerialiseTriangleDeformation> aRes;
        ReadFromFile(*aRes, NameFile(anId));

        return aRes;
    }

}; // MMVII
