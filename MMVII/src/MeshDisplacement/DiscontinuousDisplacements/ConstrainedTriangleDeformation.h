#ifndef _CONSTRAINEDTRIANGLEDEFORMATION_H_
#define _CONSTRAINEDTRIANGLEDEFORMATION_H_

#include "MMVII_TplSymbTriangle.h"
#include "../ContinuousDisplacements/TriangleDeformationUtils.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{
	/*********************************************/
	/*                                           */
	/*   cAppli_ConstrainedTriangleDeformation   */
	/*                                           */
	/*********************************************/

	class cAppli_ConstrainedTriangleDeformation : public cMMVII_Appli
	{
	public:
        using Point = tpp::Delaunay::Point;

		typedef cIm2D<tREAL8> tIm;
		typedef cDataIm2D<tREAL8> tDIm;
		typedef cDenseVect<tREAL8> tDenseVect;
		typedef std::vector<int> tIntVect;

		// Constructor
		cAppli_ConstrainedTriangleDeformation(const std::vector<std::string> &aVArgs,
											  const cSpecMMVII_Appli &aSpec);
		// Destructor
		~cAppli_ConstrainedTriangleDeformation();

		// Exe and argument methods
		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		// Does one Iteration of the constrained optimisation process
		void DoOneConstrainedIteration(const int aIterNumber, const bool aNonEmptyPathToFolder);
		// Loops over all triangles and solves system to update parameters at end of iteration
		void LoopOverTrianglesAndUpdateConstrainedParameters(const int aIterNumber, const bool aUserDefinedFolderName);
		// Generate displacement maps of last solution
		void GenerateConstrainedDisplacementMapsAndOutputImages(const tDenseVect &aVFinalSol, const int aIterNumber,
											                    const bool aNonEmptyPathToFolder);
		// Generates Displacement maps and coordinates of points in triangulation at last iteration
		void GenerateConstrainedDisplacementMapsAndDisplayLastValuesUnknowns(const int aIterNumber, const bool aDisplayLastRadiometryValues,
																			 const bool aDisplayLastTranslationValues, const bool aUserDefinedFolderName);

	private:
		// ==== Mandatory args ====

		std::string mNamePreImage;	 		// Name of given pre-image
		std::string mNamePostImage;	 		// Name of given post-image
		int mNumberOfIterations;	 		// Number of iterations in optimisation process
		bool mUseConstrainedTriangulation;	// Constrain the triangulation with segments
        // Vector of point coordinates of mesh in the form [x1-coordinate, y1-coordinate, x2-coordinate, y2-coordinate, ..., xn-coordinate, yn-coordinate]
        std::vector<int> mVectorOfNodeCoordinates;
        bool mLinkConstraintsWithIds;     				// Whether the segments are drawn by ids of points or with coordinates of points
        std::vector<int> mVectorOfConstraintSegments;	// Vector containing Ids of Points used for constraints or tpp::Delaunay::Points coordinates

		// ==== Optional args ====

		bool mShow; 										// Print result, export image ...
		bool mUseMMV2Interpolators;						  	// Whether to use interpolators from MicMacV2 instead of usual bilinear interpolation
		std::vector<std::string> mInterpolArgs;			  	// Arguments to use if linear gradient interpolation is used
		bool mUseConvexHull;			  					// Whether or not to use convex hull during triangulation
		int mNumberPointsToGenerate; 						// Number of generated points
		bool mInitialiseWithUserValues;					  	// Initalise or not with values given by user
		tREAL8 mInitialiseXTranslationValue;			  	// Value given by user to initialise x-translation unknowns
		tREAL8 mInitialiseYTranslationValue;			  	// Value given by user to initialise y-translation unknowns
		tREAL8 mInitialiseRadTrValue;					  	// Value given by user to initialise radiometry transltion unknowns
		tREAL8 mInitialiseRadScValue;					  	// Value given by user to initialise radiometry scaling unknowns
		bool mGenerateDisplacementImage;				  	// Generate image with displaced pixels
		bool mSaveTriangulation;							// Whether or not to save triangulation result in a .off file
		std::string mTriangulationFileName;					// Name to use for saving triangulation operation in a .off file
		std::string mUserDefinedFolderNameSaveResult;	  	// Folder name to save results
		bool mDisplayLastTranslationValues;				  	// Whether to display the final coordinates of the translated points
		bool mDisplayLastRadiometryValues;				  	// Display final values of radiometry unknowns at last iteration of optimisation process
		bool mDisplayStatisticsOnUnkValues;					// Display statististics : min, max, mean and std on final values of solution

		// ==== Internal variables ====

		tPt2di mSzImPre; // Size of image
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

		tpp::Delaunay mTriConstrGenerator;

		cDiffInterpolator1D *mInterpol;	   // Interpolator, if it exists use linear/grad instead of bilinear one
		cResolSysNonLinear<tREAL8> *mSys;  // Non Linear system for solving problem
		cCalculator<tREAL8> *mEqTriDeform; // Calculator giving access to values and derivatives
	};
}

#endif // _CONSTRAINEDTRIANGLEDEFORMATION_H_