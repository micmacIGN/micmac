#ifndef _TRIANGLEDEFORMATION_H_
#define _TRIANGLEDEFORMATION_H_

#include "cMMVII_Appli.h"
#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"

#include "MMVII_util.h"
#include "MMVII_TplSymbTriangle.h"
#include "TriangleDeformationUtils.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{
    /******************************************/
    /*                                        */
    /*      cAppli_TriangleDeformation        */
    /*                                        */
    /******************************************/

    class cAppli_TriangleDeformation : public cMMVII_Appli
    {
    public:
        typedef cIm2D<tREAL8> tIm;
        typedef cDataIm2D<tREAL8> tDIm;
        typedef cDenseVect<tREAL8> tDenseVect;
        typedef std::vector<int> tIntVect;

        cAppli_TriangleDeformation(const std::vector<std::string> &aVArgs,
                                   const cSpecMMVII_Appli &aSpec);
        ~cAppli_TriangleDeformation();

        int Exe() override;
        cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
        cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

        // Iterate of triangles and inside pixels
        void DoOneIteration(const int aIterNumber, const int aTotalNumberOfIterations,
                            const bool aUserDefinedFolderName);
        // Loops over all triangles and solves system to update parameters at end of iteration
        void LoopOverTrianglesAndUpdateParameters(const int aIterNumber, const int aTotalNumberOfIterations,
                                                  const bool aUserDefinedFolderName);
        // Generate displacement maps of last solution
        void GenerateDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber,
                                                     const int aTotalNumberOfIterations, const bool aUserDefinedFolderName);
        // Generates Displacement maps and coordinates of points in triangulation at last iteration
        void GenerateDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const int aTotalNumberOfIterations,
                                                                  const bool aDisplayLastRadiometryValues, const bool aDisplayLastTranslationValues,
                                                                  const bool aUserDefinedFolderName);

    private:
        // ==  Mandatory args ====

        std::string mNamePreImage;      // Name of given pre-image
        std::string mNamePostImage;     // Name of given post-image
        int mNumberPointsToGenerate;    // number of generated points
        int mNumberOfScales;            // number of iterations in optimisation process

        // ==  Optionnal args ====

        int mNumberOfLines;                                 // Uniform law generates random coordinates in interval [0, mNumberOfLines [
        int mNumberOfCols;                                  // Uniform law generates random coordinates in interval [0, mNumberOfCols [
        bool mShow;                                         // Print result, export image ...
        bool mComputeAvgMax;                                // Compute average and maximum pixel value of difference image between pre and post images
        bool mUseMultiScaleApproach;                        // Apply multi-scale approach or not
        bool mBuildRandomUniformGrid;                       // Whether to triangulate grid made of points whose coordinates follow a uniform law or have coordinates that form rectangles
        bool mInitialiseTranslationWithPreviousExecution;   // Initialise values of translation unknowns with values obtained at previous algorithm execution
        bool mInitialiseRadiometryWithPreviousExecution;    // Initialise values of radiometry unknowns with values obtained at previous algorithm execution
        bool mInitialiseWithUserValues;                     // Initalise or not with values given by user
        tREAL8 mInitialiseXTranslationValue;                // Value given by user to initialise x-translation unknowns
        tREAL8 mInitialiseYTranslationValue;                // Value given by user to initialise y-translation unknowns
        tREAL8 mInitialiseRadTrValue;                       // Value given by user to initialise radiometry transltion unknowns
        tREAL8 mInitialiseRadScValue;                       // Value given by user to initialise radiometry scaling unknowns
        bool mInitialiseWithMMVI;                           // Whether to initialise values of unknowns with pre-computed values from MicMacV1 or not
        std::string mNameInitialDepX;                       // File name of initial X-displacement map
        std::string mNameInitialDepY;                       // File name of initial Y-displacement map
        std::string mNameIntermediateDepX;                  // File name to save to of intermediate X-displacement map between executions if initialisation with previous unknown values is true
        std::string mNameIntermediateDepY;                  // File name to save to of intermediate Y-displacement map between executions if initialisation with previous unknown values is true
        std::string mNameCorrelationMaskMMVI;               // File name of mask file produced by MMVI that gives pixel locations where correlation was computed
        bool mIsFirstExecution;                             // Whether current execution of algorithm is first execution or not
        int mSigmaGaussFilterStep;                          // Decreasing step of sigma value during iterations
        bool mGenerateDisplacementImage;                    // Generate image with displaced pixels
        bool mFreezeTranslationX;                           // Freeze x-translation or not during optimisation
        bool mFreezeTranslationY;                           // Freeze y-translation or not during optimisation
        bool mFreezeRadTranslation;                         // Freeze radiometry translation or not during optimisation
        bool mFreezeRadScale;                               // Freeze radiometry scaling or not during optimisation
        tREAL8 mWeightRadTranslation;                       // Weight given to radiometry translation if soft freezing is applied (default : negative => not applied)
        tREAL8 mWeightRadScale;                             // Weight given to radiometry scaling if soft freezing is applied (default : negative => not applied)
        int mNumberOfIterGaussFilter;                       // Number of iterations to be done in Gauss filter algorithm
        int mNumberOfEndIterations;                         // Number of iterations to do while using original image in multi-scale approach
        std::string mFolderSaveResult;                      // Folder name to save results
        bool mDisplayLastTranslationValues;                 // Whether to display the final coordinates of the translated points
        bool mDisplayLastRadiometryValues;                  // Display final values of radiometry unknowns at last iteration of optimisation process

        // ==  Internal variables ====

        tPt2di mSzImPre;    // size of image
        tIm mImPre;         // memory representation of the image
        tDIm *mDImPre;      // memory representation of the image

        tPt2di mSzImPost;   // size of image
        tIm mImPost;        // memory representation of the image
        tDIm *mDImPost;     // memory representation of the image

        tPt2di mSzImOut;    // size of image
        tIm mImOut;         // memory representation of the image
        tDIm *mDImOut;      // memory representation of the image

        tPt2di mSzImDepX;   // size of image
        tIm mImDepX;        // memory representation of the image
        tDIm *mDImDepX;     // memory representation of the image

        tPt2di mSzImDepY;   // size of image
        tIm mImDepY;        // memory representation of the image
        tDIm *mDImDepY;     // memory representation of the image

        tPt2di mSzIntermediateImOut;    // size of image
        tIm mImIntermediateOut;         // memory representation of the image
        tDIm *mDImIntermediateOut;      // memory representation of the image

        tPt2di mSzImIntermediateDepX;   // size of image
        tIm mImIntermediateDepX;        // memory representation of the image
        tDIm *mDImIntermediateDepX;     // memory representation of the image

        tPt2di mSzImIntermediateDepY;   // size of image
        tIm mImIntermediateDepY;        // memory representation of the image
        tDIm *mDImIntermediateDepY;     // memory representation of the image

        tPt2di mSzCorrelationMask;  // size of image
        tIm mImCorrelationMask;     // memory representation of the image
        tDIm *mDImCorrelationMask;  // memory representation of the image

        tPt2di mSzImDiff;   // size of image
        tIm mImDiff;        // memory representation of the image
        tDIm *mDImDiff;     // memory representation of the image

        //std::vector<tPt2dr> mVectorPts;     // A vector containing a set of points
        cTriangulation2D<tREAL8> mDelTri;   // A Delaunay triangle

        tREAL8 mSigmaGaussFilter;   // Value of sigma in gauss filter
        bool mIsLastIters;          // Determines whether optimisation process is at last iters to optimise on original image

        cResolSysNonLinear<tREAL8> *mSys;   // Non Linear Sys for solving problem
        cCalculator<tREAL8> *mEqTriDeform;  // calculator giving access to values and derivatives
    };
}

#endif // _TRIANGLEDEFORMATION_H_
