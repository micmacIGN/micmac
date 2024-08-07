#ifndef _TRIANGLEDEFORMATIONRADIOMETRY_H_
#define _TRIANGLEDEFORMATIONRADIOMETRY_H_

#include "MMVII_TplSymbTriangle.h"
#include "TriangleDeformationUtils.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

	/*************************************************/
	/*                                               */
	/*     cAppli_TriangleDeformationRadiometry      */
	/*                                               */
	/*************************************************/

	class cAppli_TriangleDeformationRadiometry : public cMMVII_Appli
	{
	public:
		typedef cIm2D<tREAL8> tIm;
		typedef cDataIm2D<tREAL8> tDIm;
		typedef cTriangle<tREAL8, 2> tTri2dr;
		typedef cDenseVect<tREAL8> tDenseVect;
		typedef std::vector<int> tIntVect;
		typedef std::vector<tREAL8> tDoubleVect;

		// Constructor
		cAppli_TriangleDeformationRadiometry(const std::vector<std::string> &aVArgs,
											 const cSpecMMVII_Appli &aSpec);
		// Destructor
		~cAppli_TriangleDeformationRadiometry();

		// Exe and argument methods
		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		// Does one iteration of optimisation
		void DoOneIterationRadiometry(const int aIterNumber, const int aTotalNumberOfIterations,
									  const tDenseVect &aVInitSol, const bool aNonEmptyFolderName);
		// Loops over all triangles and solves system to update parameters at end of iteration
		void LoopOverTrianglesAndUpdateParametersRadiometry(const int aIterNumber, const int aTotalNumberOfIters,
															const bool aNonEmptyPathToFolder);
		// Generates displacement maps of last solution
		void GenerateOutputImage(const tDenseVect &aVFinalSol, const int aTotalNumberOfIterations,
								 const bool aNonEmptyFolderName);

	private:
		// ==  Mandatory args ====

		std::string mNamePreImage;	 // Name of given pre-image
		std::string mNamePostImage;	 // Name of given post-image
		int mNumberPointsToGenerate; // Number of generated points
		int mNumberOfIterations;		 // Number of iterations in optimisation process

		// ==  Optionnal args ====

		int mNumberOfLines;								// Uniform law generates random coordinates in interval [0, mNumberOfLines [
		int mNumberOfCols;								// Uniform law generates random coordinates in interval [0, mNumberOfCols [
		bool mShow;										// Print result, export image ...
		bool mUseMultiScaleApproach;					// Apply multi-scale approach or not
		bool mGenerateOutputImage;						// Generate image with computed radiometric pixels
		bool mBuildRandomUniformGrid;					// Whether to triangulate grid made of points whose coordinates follow a uniform law or have coordinates that form rectangles
		bool mUseMMV2Interpolators;						// Whether to use interpolators from MicMacV2 instead of usual bilinear interpolation
		std::vector<std::string> mInterpolArgs;			// Arguments to use if linear gradient interpolation is used
		bool mSerialiseTriangleNodes;					// Whether to serialise nodes to .xml file or not
		std::string mNameMultipleTriangleNodes;			// File name to use when saving all to triangle nodes to .xml file
		bool mInitialiseWithUserValues;					// Initalise or not with values given by user
		tREAL8 mInitialiseRadTrValue;					// Value given by user to initialise radiometry transltion unknowns
		tREAL8 mInitialiseRadScValue;					// Value given by user to initialise radiometry scaling unknowns
		bool mHardFreezeRadTranslation;					// Freeze radiometry translation or not during optimisation
		bool mHardFreezeRadScale;						// Freeze radiometry scaling or not during optimisation
		tREAL8 mWeightRadTranslation;					// Weight given to radiometry translation if soft freezing is applied (default : negative => not applied)
		tREAL8 mWeightRadScale;							// Weight given to radiometry scaling if soft freezing is applied (default : negative => not applied)
		std::string mUserDefinedFolderNameToSaveResult; // Folder name to save results in, if wanted
		bool mDisplayLastRadiometryValues;				// Display final values of radiometry unknowns at last iteration of optimisation process
		int mSigmaGaussFilterStep;						// Decreasing step of sigma value during iterations
		int mNumberOfIterGaussFilter;					// Number of iterations to be done in Gauss filter algorithm
		int mNumberOfEndIterations;						// Number of iterations to do while using original image in multi-scale approach

		// ==  Internal variables ====

		cPt2di mSzImPre; // Size of image
		tIm mImPre;		 // Memory representation of the image
		tDIm *mDImPre;	 // Memory representation of the image

		cPt2di mSzImPost; // Size of image
		tIm mImPost;	  // Memory representation of the image
		tDIm *mDImPost;	  // Memory representation of the image

		cPt2di mSzImOut; // Size of image
		tIm mImOut;		 // Memory representation of the image
		tDIm *mDImOut;	 // Memory representation of the image

		tREAL8 mSigmaGaussFilter; // Value of sigma in gauss filter
		bool mIsLastIters;		  // Determines whether optimisation process is at last iters to optimise on original image

		cTriangulation2D<tREAL8> mDelTri;	// A Delaunay triangle

		cDiffInterpolator1D *mInterpolRad;			// Interpolator, if exist use linear/grad instead of bilinear one
		cResolSysNonLinear<tREAL8> *mSysRadiometry; // Non Linear Sys for solving problem
		cCalculator<tREAL8> *mEqRadiometryTri;		// Calculator giving access to values and derivatives
	};
}

#endif // _TRIANGLEDEFORMATIONRADIOMETRY_H_
