#ifndef _TRIANGLEDEFORMATIONTRAD_H_
#define _TRIANGLEDEFORMATIONTRAD_H_

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"
#include "TriangleDeformation.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

    /******************************************/
    /*                                        */
    /*          cTriangleDeformation          */
    /*                                        */
    /******************************************/

    class cAppli_cTriangleDeformationTrRad : public cAppli_cTriangleDeformation
    {
    public:
        typedef cIm2D<tREAL8> tIm;
        typedef cDataIm2D<tREAL8> tDIm;
        typedef cTriangle<tREAL8, 2> tTri2dr;
        typedef cDenseVect<double> tDenseVect;
        typedef std::vector<int> tIntVect;
        typedef std::vector<double> tDoubleVect;

        cAppli_cTriangleDeformationTrRad(const std::vector<std::string> &aVArgs,
                                    const cSpecMMVII_Appli &aSpec);
        ~cAppli_cTriangleDeformationTrRad();

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

        // Build uniform vector of coordinates and apply Delaunay triangulation
        virtual void ConstructUniformRandomVectorAndApplyDelaunay();
        // Generate coordinates from uniform law for Delaunay triangulation application
        virtual void GeneratePointsForDelaunay();
        // Construct difference image and compute average and max pixel value on ths image
        void SubtractPrePostImageAndComputeAvgAndMax();
        // Iterate of triangles and inside pixels
        void DoOneIteration(const int aIterNumber);
        // Loops over all triangles and solves system to update parameters at end of iteration
        virtual void LoopOverTrianglesAndUpdateParameters(const int aIterNumber);
        // Generate displacement maps of last solution
        virtual void GenerateDisplacementMaps(const tDenseVect &aVFinalSol, const int aIterNumber);
        // Generates Displacement maps and coordinates of points in triangulation at last iteration
        virtual void GenerateDisplacementMapsAndLastTranslatedPoints(const int aIterNumber);
        // Initialise problem after user has input information
        virtual void InitialisationAfterExe();
        // Loads current pre and post images
        virtual void LoadImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage);
        // Load image and data according to number of iterations to optimise on original image
        virtual void ManageDifferentCasesOfEndIterations(const int aIterNumber, tIm aCurPreIm, tDIm * aCurPreDIm,
                                                 tIm aCurPostIm, tDIm * aCurPostDIm);
        // Fill displacement maps and output image
        virtual void FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
                                                        const cPt2dr &aLastTranslatedFilledPoint,
                                                        const tREAL8 aLastRadiometryTranslation,
                                                        const tREAL8 aLastRadiometryScaling);
        // Apply barycentric translation formula to current translation values
        cPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const cPt2dr &aCurrentTranslationPointA,
                                                              const cPt2dr &aCurrentTranslationPointB,
                                                              const cPt2dr &aCurrentTranslationPointC,
                                                              const tDoubleVect &aVObs);
        // Apply barycentric translation formula to current radiometric translation values
        tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
                                                                         const tREAL8 aCurrentRadTranslationPointB,
                                                                         const tREAL8 aCurrentRadTranslationPointC,
                                                                         const tDoubleVect &aVObs);
        // Apply barycentric translation formula to current radiometric scaling values
        tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
                                                                     const tREAL8 aCurrentRadScalingPointB,
                                                                     const tREAL8 aCurrentRadScalingPointC,
                                                                     const tDoubleVect &aVObs);

    private:
        // ==  Mandatory args ====

        std::string mNamePreImage;   // Name of given pre-image
        std::string mNamePostImage;  // Name of given post-image
        int mNumberPointsToGenerate; // number of generated points
        int mNumberOfScales;         // number of iterations in optimisation process

        // ==  Optionnal args ====

        int mRandomUniformLawUpperBoundLines;         // Uniform law generates random coordinates in interval [0, mRandomUniformLawUpperBoundLines [
        int mRandomUniformLawUpperBoundCols;          // Uniform law generates random coordinates in interval [0, mRandomUniformLawUpperBoundCols [
        bool mShow;                                   // Print result, export image ...
        bool mComputeAvgMax;                          // Compute average and maximum pixel value of difference image between pre and post images
        bool mUseMultiScaleApproach;                  // Apply multi-scale approach or not
        int mSigmaGaussFilterStep;                    // Decreasing step of sigma value during iterations
        bool mGenerateDisplacementImage;              // Generate image with displaced pixels
        bool mFreezeRadTranslation;                   // Freeze radiometry translation in computation
        bool mFreezeRadScale;                         // Freeze radiometry scaling in computation
        double mWeightRadTranslation;                 // Weight given to radiometry translation if soft freezing is applied (default : negative => not applied)
        double mWeightRadScale;                       // Weight given to radiometry scaling if soft freezing is applied (default : negative => not applied)
        int mNumberOfIterGaussFilter;                 // Number of iterations to be done in Gauss filter algorithm
        int mNumberOfEndIterations;                   // Number of iterations to do while using original image in multi-scale approach
        bool mDisplayLastTranslatedPointsCoordinates; // Whether to display the final coordinates of the translated points

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

        cPt2di mSzImDepX; //  size of image
        tIm mImDepX;      //  memory representation of the image
        tDIm *mDImDepX;   //  memory representation of the image

        cPt2di mSzImDepY; //  size of image
        tIm mImDepY;      //  memory representation of the image
        tDIm *mDImDepY;   //  memory representation of the image

        std::vector<cPt2dr> mVectorPts;   // A vector containing a set of points
        cTriangulation2D<tREAL8> mDelTri; // A Delaunay triangle

        double mSigmaGaussFilter; // Value of sigma in gauss filter
        bool mIsLastIters;        // Determines whether optimisation process is at last iters to optimise on original image

        cResolSysNonLinear<tREAL8> *mSys;  // Non Linear Sys for solving problem
        cCalculator<double> *mEqTriDeformTrRad; // calculator giving access to values and derivatives
    };
}

#endif // _TRIANGLEDEFORMATIONTRAD_H_
