#ifndef _TRIANGLEDEFORMATIONRAD_H_
#define _TRIANGLEDEFORMATIONRAD_H_

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"
#include "TriangleDeformation.h"
#include "TriangleDeformationRadiometry.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

    /******************************************************/
    /*                                                    */
    /*           cAppli_cTriangleDeformationRad           */
    /*                                                    */
    /******************************************************/

    class cAppli_cTriangleDeformationRad : public cAppli_cTriangleDeformationRadiometry
    {
    public:
        typedef cIm2D<tREAL8> tIm;
        typedef cDataIm2D<tREAL8> tDIm;
        typedef cTriangle<tREAL8, 2> tTri2dr;
        typedef cDenseVect<double> tDenseVect;
        typedef std::vector<int> tIntVect;
        typedef std::vector<double> tDoubleVect;

        cAppli_cTriangleDeformationRad(const std::vector<std::string> &aVArgs,
                                       const cSpecMMVII_Appli &aSpec);
        ~cAppli_cTriangleDeformationRad();

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

        // Iterate of triangles and inside pixels
        virtual void DoOneIterationRadiometry(const int aIterNumber);
        // Loops over all triangles and solves system to update parameters at end of iteration
        virtual void LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber);
        // Generate displacement maps of last solution
        virtual void GenerateOutputImageAndDisplayLastRadiometryValues(const tDenseVect &aVFinalSol, const int aIterNumber);

    private:
        // ==  Mandatory args ====

        std::string mNamePreImage;   // Name of given pre-image
        std::string mNamePostImage;  // Name of given post-image
        int mNumberPointsToGenerate; // number of generated points
        int mNumberOfScales;         // number of iterations in optimisation process

        // ==  Optionnal args ====

        int mRandomUniformLawUpperBoundLines; // Uniform law generates random coordinates in interval [0, mRandomUniformLawUpperBoundLines [
        int mRandomUniformLawUpperBoundCols;  // Uniform law generates random coordinates in interval [0, mRandomUniformLawUpperBoundCols [
        bool mShow;                           // Print result, export image ...
        bool mGenerateOutputImage;            // Generate image with displaced pixels
        bool mUseMultiScaleApproach;          // Apply multi-scale approach or not
        tREAL8 mWeightRadTranslation;         // Weight given to radiometry translation if soft freezing is applied (default : negative => not applied)
        tREAL8 mWeightRadScale;               // Weight given to radiometry scaling if soft freezing is applied (default : negative => not applied)
        bool mDisplayLastRadiometryValues;    // Display final values of radiometry unknowns at last iteration of optimisation process
        int mSigmaGaussFilterStep;            // Decreasing step of sigma value during iterations
        int mNumberOfIterGaussFilter;         // Number of iterations to be done in Gauss filter algorithm
        int mNumberOfEndIterations;           // Number of iterations to do while using original image in multi-scale approach

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

        std::vector<cPt2dr> mVectorPts;   // A vector containing a set of points
        cTriangulation2D<tREAL8> mDelTri; // A Delaunay triangle

        cResolSysNonLinear<tREAL8> *mSys; // Non Linear Sys for solving problem
        cCalculator<double> *mEqRadTri;   // calculator giving access to values and derivatives
    };
}

#endif // _TRIANGLEDEFORMATIONRAD_H_
