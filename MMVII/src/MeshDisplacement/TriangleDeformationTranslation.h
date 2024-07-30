#ifndef _TRIANGLEDEFORMATIONTRANSLATION_H_
#define _TRIANGLEDEFORMATIONTRANSLATION_H_

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_Interpolators.h"

#include "MMVII_TplSymbTriangle.h"
#include "TriangleDeformationUtils.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

	/************************************************/
	/*                                              */
	/*    cAppli_TriangleDeformationTranslation     */
	/*                                              */
	/************************************************/

	class cAppli_TriangleDeformationTranslation : public cMMVII_Appli
	{
	public:
		typedef cIm2D<tREAL8> tIm;
		typedef cDataIm2D<tREAL8> tDIm;
		typedef cTriangle<tREAL8, 2> tTri2dr;
		typedef cDenseVect<tREAL8> tDenseVect;
		typedef std::vector<int> tIntVect;
		typedef std::vector<tREAL8> tDoubleVect;

		cAppli_TriangleDeformationTranslation(const std::vector<std::string> &aVArgs,
											  const cSpecMMVII_Appli &aSpec);
		~cAppli_TriangleDeformationTranslation();

		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		// Iterate of triangles and inside pixels
		// void DoOneIterationTranslation(const int aIterNumber, const int aTotalNumberOfIterations);
		void DoOneIterationTranslation(const int aIterNumber, const int aTotalNumberOfIterations,
									   const tDenseVect &aVInitVecSol, const bool aUserDefinedFolder);
		// Loops over all triangles and solves system to update parameters at end of iteration
		void LoopOverTrianglesAndUpdateParametersTranslation(const int aIterNumber, const int aTotalNumberOfIterations, const bool aUserDefinedFolder);
		// Generate displacement maps of last solution
		void GenerateDisplacementMaps(const tDenseVect &aVFinalSolTr, const int aIterNumber,
									  const int aTotalNumberOfIterations, const bool aUserDefinedFolder);
		// Generates Displacement maps and coordinates of points in triangulation at last iteration
		void GenerateDisplacementMapsAndDisplayLastTranslatedPoints(const int aIterNumber, const int aTotalNumberOfIterations,
																	const tDenseVect &aVinitVecSol, const bool aUserDefinedFolder);

	private:
		// ==  Mandatory args ====

		std::string mNamePreImage;	 // Name of given pre-image
		std::string mNamePostImage;	 // Name of given post-image
		int mNumberPointsToGenerate; // Number of generated points
		int mNumberOfIterations;		 // Number of iterations in optimisation process or scales is use of multi-scale approach

		// ==  Optionnal args ====

		int mNumberOfLines;								  // Uniform law generates random coordinates in interval [0, mNumberOfLines [
		int mNumberOfCols;								  // Uniform law generates random coordinates in interval [0, mNumberOfCols [
		bool mShow;										  // Print results
		bool mUseMultiScaleApproach;					  // Apply multi-scale approach or not
		bool mBuildRandomUniformGrid;					  // Whether to triangulate grid made of points whose coordinates follow a uniform law or have coordinates that form rectangles
		bool mUseMMV2Interpolators;					  // Whether to use interpolators from MicMacV2 instead of usual bilinear interpolation
		std::vector<std::string> mInterpolArgs;			  // Arguments to use if linear gradient interpolation is used
		bool mSerialiseTriangleNodes;					  // Whether to serialise nodes to .xml file or not
		std::string mNameMultipleTriangleNodes;			  // File name to use when saving all to triangle nodes to .xml file
		bool mInitialiseTranslationWithPreviousExecution; // Initialise values of unknowns with values obtained at previous algorithm execution
		bool mInitialiseWithUserValues;					  // Initalise or not with values given by user
		tREAL8 mInitialiseXTranslationValue;			  // Value given by user to initialise x-translation unknowns
		tREAL8 mInitialiseYTranslationValue;			  // Value given by user to initialise y-translation unknowns
		bool mInitialiseWithMMVI;						  // Whether to initialise values of unknowns with pre-computed values from MicMacV1 or not
		std::string mNameInitialDepX;					  // File name of initial X-displacement map
		std::string mNameInitialDepY;					  // File name of initial Y-displacement map
		std::string mNameIntermediateDepX;				  // File name to save to of intermediate X-displacement map between executions if initialisation with previous unknown values is true
		std::string mNameIntermediateDepY;				  // File name to save to of intermediate Y-displacement map between executions if initialisation with previous unknown values is true
		std::string mNameCorrelationMaskMMVI;			  // File name of mask file produced by MMVI that gives pixel locations where correlation was computed
		bool mIsFirstExecution;							  // Whether current execution of algorithm is first execution or not
		int mSigmaGaussFilterStep;						  // Decreasing step of sigma value during iterations
		bool mGenerateDisplacementImage;				  // Generate image with displaced pixels
		bool mHardFreezeTranslationX;					  // Freeze x-translation or not during optimisation
		bool mHardFreezeTranslationY;					  // Freeze y-translation or not during optimisation
		tREAL8 mWeightTranslationX;						  // Weight given to x-translation if soft freezing is applied (default : negative => not applied)
		tREAL8 mWeightTranslationY;						  // Weight given to y-translation if soft freezing is applied (default : negative => not applied)
		std::string mUserDefinedFolderNameToSaveResult;	  // Folder name to save results in, if wanted
		int mNumberOfIterGaussFilter;					  // Number of iterations to be done in Gauss filter algorithm
		int mNumberOfEndIterations;						  // Number of iterations to do while using original image in multi-scale approach
		bool mDisplayLastTranslationValues;				  // Whether to display the final coordinates of the translated points

		// ==  Internal variables ====

		tPt2di mSzImPre; // size of image
		tIm mImPre;		 // Memory representation of the image
		tDIm *mDImPre;	 // Memory representation of the image

		tPt2di mSzImPost; // Size of image
		tIm mImPost;	  // Memory representation of the image
		tDIm *mDImPost;	  // Memory representation of the image

		tPt2di mSzImOut; // Size of image
		tIm mImOut;		 // Memory representation of the image
		tDIm *mDImOut;	 // Memory representation of the image

		tPt2di mSzImDepX; // Size of image
		tIm mImDepX;	  // Memory representation of the image
		tDIm *mDImDepX;	  // Memory representation of the image

		tPt2di mSzImDepY; // Size of image
		tIm mImDepY;	  // Memory representation of the image
		tDIm *mDImDepY;	  // Memory representation of the image

		tPt2di mSzImIntermediateDepX; // Size of image
		tIm mImIntermediateDepX;	  // Memory representation of the image
		tDIm *mDImIntermediateDepX;	  // Memory representation of the image

		tPt2di mSzImIntermediateDepY; // Size of image
		tIm mImIntermediateDepY;	  // Memory representation of the image
		tDIm *mDImIntermediateDepY;	  // Memory representation of the image

		tPt2di mSzCorrelationMask; // Size of image
		tIm mImCorrelationMask;	   // Memory representation of the image
		tDIm *mDImCorrelationMask; // Memory representation of the image

		tREAL8 mSigmaGaussFilter; // Value of sigma in gauss filter
		bool mIsLastIters;		  // Determines whether optimisation process is at last iters to optimise on original image

		cTriangulation2D<tREAL8> mDelTri;	// A Delaunay triangle

		cDiffInterpolator1D *mInterpolTr;			 // Interpolator, if it exists use linear/grad instead of bilinear one
		cResolSysNonLinear<tREAL8> *mSysTranslation; // Non Linear Sys for solving problem
		cCalculator<tREAL8> *mEqTranslationTri;		 // Calculator giving access to values and derivatives
	};
}

#endif // _TRIANGLEDEFORMATIONTRANSLATION_H_