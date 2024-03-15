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

    public:
        typedef cDenseVect<double> tDenseVect;
        typedef std::vector<int> tIntVect;

        cSerialiseTriangleDeformation();    // default constructor
        ~cSerialiseTriangleDeformation();   // destructor

        cSerialiseTriangleDeformation(const tDenseVect &aVecSol,    // Current solution vector
                                      const tIntVect &aIndicesVec,  // Indices of current triangle in solution vector
                                      const int aXIndices,          // Index for current x-displacement in solution vector
                                      const int aYIndices,          // Index for current y-displacement in solution vector
                                      const int aRadTrIndices,      // Index for current radiometry translation in solution vector
                                      const int aRadScIndices,      // Index for current radiometry scaling in solution vector
                                      const tTri2dr &aTri,          // Current triangle
                                      const tPt3di &aFace,          // Current face of triangle
                                      const int aPointNumberInTri,  // Index of point in triangle : 0, 1 or 2
                                      const int IdOfPoint);         // Id of point when looping over triangles

        void AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise);   // Add data to xml file
        void SaveTriangleDeformationObjectToFile() const;                                       // Save to xml file
        void ShowTriangleDeformationObjectCarateristics() const;                                // Display information about object
        // Adds ability to re-read already saved object
        static std::unique_ptr<cSerialiseTriangleDeformation> ReadSerialisedTriangleDeformation(const tDenseVect &aVecSol, const tIntVect &aIndVec, 
                                                                                                const int aXInd, const int aYInd, const int aRadTrInd, 
                                                                                                const int aRadScInd, const tTri2dr &aTriangle, const tPt3di &aFace,
                                                                                                const int aPointNumberInTri, const int anIdOfPoint);

        static std::string NameFileToSaveOneObj(const int anId);    // Gives name to saved file
        int GetPointId() const;
        int& GetPointId();
        int GetTriangleFace() const;
        int& GetTriangleFace();

    private:
        int mIdOfPt;            // Id of point when looping over triangles
        int mFaceOfTriangle;    // The face number associated to the point

        friend std::ostream& operator<<(std::ostream& os, const cSerialiseTriangleDeformation& obj);
    };

    void AddData(const cAuxAr2007 &anAux, cSerialiseTriangleDeformation &aPtToSerialise);
}; // MMVII