#include "MMVII_2Include_Serial_Tpl.h"
#include "TriangleDeformationUtils.h"

namespace MMVII
{
    /******************************************/
    /*                                        */
    /*      cSerialiseTriangleDeformation     */
    /*                                        */
    /******************************************/

    class cSerialiseTriangleDeformation : public cNodeOfTriangles
    {
            typedef cDenseVect<double> tDenseVect;
            typedef std::vector<int> tIntVect;

        public:
            cSerialiseTriangleDeformation(const tDenseVect &aVecSol,                // Current solution vector
                                        const tIntVect &aIndicesVec,                // Indices of current triangle in solution vector
                                        const int aXIndices,                        // Index for current x-displacement in solution vector
                                        const int aYIndices,                        // Index for current y-displacement in solution vector
                                        const tREAL8 aRadTrIndices,                 // Index for current radiometry translation in solution vector
                                        const tREAL8 aRadScIndices,                 // Index for current radiometry scaling in solution vector
                                        const tTri2dr &aTri,                        // Current triangle
                                        const int aPointNumberInTri,                // Index of point in triangle : 0, 1 or 2
                                        const int IdOfPoint,                        // Id of point when looping over triangles
                                        const cTriangulation2D<tREAL8> &aDelTri);   // a triangulation

            ~cSerialiseTriangleDeformation();                        // destructor

            void AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise); // Add data to xml file
            void SaveAuto() const;                                                                // Save to xml file
            std::unique_ptr<cSerialiseTriangleDeformation> ReadSerialisedTriangleDeformation(int anId);

            static std::string NameFile(int anId) { return "Id_" + ToStr(anId) + ".xml"; }

        private:
            int mIdOfPt; // Id of point when looping over triangles
    };
}; // MMVII