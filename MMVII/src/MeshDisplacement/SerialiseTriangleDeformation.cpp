#include "MMVII_2Include_Serial_Tpl.h"
#include "TriangleDeformationUtils.h"

/**
 \file SerialiseDeformation.cpp
 \brief File containing methods to serialise objects
 linked to triangle deformation between optimisation executions
 **/

namespace MMVII
{

    class cSerialiseTriangleDeformation : public cNodeOfTriangles
    {
        typedef cDenseVect<double> tDenseVect;
        typedef std::vector<int> tIntVect;

    public:
        cSerialiseTriangleDeformation(const tDenseVect &aVecSol,   // Current solution vector
                                      const tIntVect &aIndicesVec, // Indices of current triangle in solution vector
                                      const int aXIndices,         // Index for current x-displacement in solution vector
                                      const int aYIndices,         // Index for current y-displacement in solution vector
                                      const tREAL8 aRadTrIndices,  // Index for current radiometry translation in solution vector
                                      const tREAL8 aRadScIndices,  // Index for current radiometry scaling in solution vector
                                      const tTri2dr &aTri,         // Current triangle
                                      const int aPointNumberInTri, // Index of point in triangle : 0, 1 or 2
                                      const int IdOfPoint);        // Id of point when looping over triangles
        ~cSerialiseTriangleDeformation();                          // destructor

        void AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise); // Add data to xml file
        void SaveAuto() const;                                                                // Save to xml file
        std::unique_ptr<cSerialiseTriangleDeformation> ReadSerialisedTriangleDeformation(int anId);

        static std::string NameFile(int anId) { return "Id_" + ToStr(anId) + ".xml"; }

    private:
        int mIdOfPt; // Id of point when looping over triangles
    };

    cSerialiseTriangleDeformation::cSerialiseTriangleDeformation(const tDenseVect &aVecSol,
                                                                 const tIntVect &aIndicesVec,
                                                                 const int aXIndices,
                                                                 const int aYIndices,
                                                                 const tREAL8 aRadTrIndices,
                                                                 const tREAL8 aRadScIndices,
                                                                 const tTri2dr &aTri,
                                                                 const int aPointNumberInTri,
                                                                 const int IdOfPoint) : cNodeOfTriangles(aVecSol, aIndicesVec, aXIndices,
                                                                                                       aYIndices, aRadTrIndices, aRadScIndices,
                                                                                                       aTri, aPointNumberInTri),
                                                                                        mIdOfPt(IdOfPoint)

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
