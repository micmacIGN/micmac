#ifndef _TRIANGLEDEFORMATIONRADIOMETRY_H_
#define _TRIANGLEDEFORMATIONRADIOMETRY_H_

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_TplSymbTriangle.h"
#include "TriangleDeformation.h"


using namespace NS_SymbolicDerivative;

namespace MMVII
{

    /*************************************************/
    /*                                               */
    /*     cAppli_TriangleDeformationRadiometry      */
    /*                                               */
    /*************************************************/

    class cAppli_TriangleDeformationRadiometry : public cAppli_TriangleDeformation
    {
    public:
        typedef cIm2D<tREAL8> tIm;
        typedef cDataIm2D<tREAL8> tDIm;
        typedef cTriangle<tREAL8, 2> tTri2dr;
        typedef cDenseVect<tREAL8> tDenseVect;
        typedef std::vector<int> tIntVect;
        typedef std::vector<tREAL8> tDoubleVect;

        cAppli_TriangleDeformationRadiometry(const std::vector<std::string> &aVArgs,
                                             const cSpecMMVII_Appli &aSpec);
        ~cAppli_TriangleDeformationRadiometry();

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

        // Iterate of triangles and inside pixels
        void DoOneIterationRadiometry(const int aIterNumber, const int aTotalNumberOfIterations,
                                      const tDenseVect &aVinitSol);
        // Loops over all triangles and solves system to update parameters at end of iteration
        void LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber);
        // Generate displacement maps of last solution
        void GenerateOutputImageAndDisplayLastRadiometryValues(const tDenseVect &aVFinalSol, const int aIterNumber);

    private:
        // ==  Mandatory args ====

        std::string mNamePreImage;   // Name of given pre-image
        std::string mNamePostImage;  // Name of given post-image
        int mNumberPointsToGenerate; // number of generated points
        int mNumberOfScales;         // number of iterations in optimisation process

        // ==  Optionnal args ====

        int mNumberOfLines;   // Uniform law generates random coordinates in interval [0, mNumberOfLines [
        int mNumberOfCols;    // Uniform law generates random coordinates in interval [0, mNumberOfCols [
        bool mShow;                             // Print result, export image ...
        bool mGenerateOutputImage;              // Generate image with computed radiometric pixels
        bool mBuildRandomUniformGrid;           // Whether to triangulate grid made of points whose coordinates follow a uniform law or have coordinates that form rectangles
        bool mInitialiseWithUserValues;         // Initalise or not with values given by user
        tREAL8 mInitialiseRadTrValue;           // Value given by user to initialise radiometry transltion unknowns
        tREAL8 mInitialiseRadScValue;           // Value given by user to initialise radiometry scaling unknowns
        bool mUseMultiScaleApproach;            // Apply multi-scale approach or not
        tREAL8 mWeightRadTranslation;           // Weight given to radiometry translation if soft freezing is applied (default : negative => not applied)
        tREAL8 mWeightRadScale;                 // Weight given to radiometry scaling if soft freezing is applied (default : negative => not applied)
        bool mDisplayLastRadiometryValues;      // Display final values of radiometry unknowns at last iteration of optimisation process
        int mSigmaGaussFilterStep;              // Decreasing step of sigma value during iterations
        int mNumberOfIterGaussFilter;           // Number of iterations to be done in Gauss filter algorithm
        int mNumberOfEndIterations;             // Number of iterations to do while using original image in multi-scale approach

        // ==  Internal variables ====

        cPt2di mSzImPre; //  size of image
        tIm mImPre;      //  memory representation of the image
        tDIm *mDImPre;   //  memory representation of the image

        cPt2di mSzImPost; //  size of image
        tIm mImPost;      //  memory representation of the image
        tDIm *mDImPost;   //  memory representation of the image

        cPt2di mSzImOut; //  size of image
        tIm mImOut;      //  memory representation of the image
        tDIm *mDImOut;   //  memory representation of the image

        tREAL8 mSigmaGaussFilter; // Value of sigma in gauss filter
        bool mIsLastIters;        // Determines whether optimisation process is at last iters to optimise on original image

        cTriangulation2D<tREAL8> mDelTri; // A Delaunay triangle

        cResolSysNonLinear<tREAL8> *mSysRadiometry;      // Non Linear Sys for solving problem
        cCalculator<tREAL8> *mEqRadiometryTri; // calculator giving access to values and derivatives
    };
}

#endif // _TRIANGLEDEFORMATIONRADIOMETRY_H_
