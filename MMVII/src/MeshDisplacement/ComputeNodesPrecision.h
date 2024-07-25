#ifndef _COMPUTENODESPRECISION_H_
#define _COMPUTENODESPRECISION_H_

#include "cMMVII_Appli.h"

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"

#include "TriangleDeformationUtils.h"

namespace MMVII
{

    /************************************************/
    /*                                              */
    /*          cAppli_ComputeNodesPrecision        */
    /*                                              */
    /************************************************/

    class cAppli_ComputeNodesPrecision : public cMMVII_Appli
    {
    public:
        typedef cIm2D<tREAL8> tIm;
        typedef cDataIm2D<tREAL8> tDIm;
        typedef cTriangle<tREAL8, 2> tTri2dr;

        cAppli_ComputeNodesPrecision(const std::vector<std::string> &aVArgs,
                                     const cSpecMMVII_Appli &aSpec);
        ~cAppli_ComputeNodesPrecision();

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

        void LoopOverTrianglesAndGetDiffDispMaps(); // Loop over triangles and save displacement map between ground truth and interpolation value

    private:
        // ==  Mandatory args ==

        std::string mNameDispXMap;   // Name of given pre-image
        std::string mNameDispYMap;   // Name of given post-image
        int mNumberPointsToGenerate; // Number of generated points

        // == Optionnal args ==

        int mNumberOfLines;                         // Uniform law generates random coordinates in interval [0, mNumberOfLines [
        int mNumberOfCols;                          // Uniform law generates random coordinates in interval [0, mNumberOfCols [
        bool mBuildRandomUniformGrid;               // Whether to triangulate grid made of points whose coordinates follow a uniform law or have coordinates that form rectangles
        bool mComputeDiffDispMaps;                  // Whether to compute difference displacement maps or not
        bool mComputeInterpTranslationDispMaps;     // Whether to compute translated dispalcement maps or not
        std::string mNameDiffDispX;                 // File name to use to save the x-displacement difference map between interpolation ground truth and ground truth
        std::string mNameDiffDispY;                 // File name to use to save the y-displacement difference map between interpolation ground truth and ground truth
        std::string mNameComputedTranslatedDispX;   // File name to use to save the x-translated displacement map
        std::string mNameComputedTranslatedDispY;   // File name to use to save the y-translated displacement map

        // == Internal variables ==

        tPt2di mSzImDispX; // size of image
        tIm mImDispX;      // memory representation of the image
        tDIm *mDImDispX;   // memory representation of the image

        tPt2di mSzImDispY; // size of image
        tIm mImDispY;      // memory representation of the image
        tDIm *mDImDispY;   // memory representation of the image

        tPt2di mSzImDiffDispX; // size of image
        tIm mImDiffDispX;      // memory representation of the image
        tDIm *mDImDiffDispX;   // memory representation of the image

        tPt2di mSzImDiffDispY; // size of image
        tIm mImDiffDispY;      // memory representation of the image
        tDIm *mDImDiffDispY;   // memory representation of the image

        tPt2di mSzImTranslatedDispX; // size of image
        tIm mImTranslatedDispX;      // memory representation of the image
        tDIm *mDImTranslatedDispX;   // memory representation of the image

        tPt2di mSzImTranslatedDispY; // size of image
        tIm mImTranslatedDispY;      // memory representation of the image
        tDIm *mDImTranslatedDispY;   // memory representation of the image

        cTriangulation2D<tREAL8> mDelTri; // A Delaunay triangle
    };
} // MMVII

#endif // _COMPUTENODESPRECISION_H_