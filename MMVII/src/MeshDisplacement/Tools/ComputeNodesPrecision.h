#ifndef _COMPUTENODESPRECISION_H_
#define _COMPUTENODESPRECISION_H_

#include "MMVII_Geom2D.h"
#include "MMVII_PhgrDist.h"

#include "../ContinuousDisplacements/TriangleDeformationUtils.h"

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

		// Constructor
		cAppli_ComputeNodesPrecision(const std::vector<std::string> &aVArgs,
									 const cSpecMMVII_Appli &aSpec);
		// Destructor
		~cAppli_ComputeNodesPrecision();

		// Exe and argument methods
		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		// Loops over triangles and save displacement map between ground truth and interpolation value
		void LoopOverTrianglesAndGetDiffDispMaps();
		// Checks if interpolated bilinear value in displacement map can be returned or if integer value is needed
		tPt2dr CheckReturnOfInterpolatorValue(const tPt2dr &aRealCoordPoint, const tPt2di &aIntCoordPoint);

	private:
		// == Mandatory args ==

		std::string mNameDispXMap;	 // Name of given pre-image
		std::string mNameDispYMap;	 // Name of given post-image
		int mNumberPointsToGenerate; // Number of generated points

		// == Optionnal args ==

		int mNumberOfLines;						  // Uniform law generates random coordinates in interval [0, mNumberOfLines [
		int mNumberOfCols;						  // Uniform law generates random coordinates in interval [0, mNumberOfCols [
		bool mBuildRandomUniformGrid;			  // Whether to triangulate grid made of points whose coordinates follow a uniform law or have coordinates that form rectangles
		std::vector<std::string> mInterpolArgs;	  // Arguments to use if linear gradient interpolation is used
		bool mComputeDiffDispMaps;                // Whether to compute difference displacement maps or not
		bool mComputeInterpTranslationDispMaps;	  // Whether to compute translated dispalcement maps or not
		std::string mNameDiffDispX;				  // File name to use to save the x-displacement difference map between interpolation ground truth and ground truth
		std::string mNameDiffDispY;				  // File name to use to save the y-displacement difference map between interpolation ground truth and ground truth
		std::string mNameComputedTranslatedDispX; // File name to use to save the x-translated displacement map
		std::string mNameComputedTranslatedDispY; // File name to use to save the y-translated displacement map

		// == Internal variables ==

		tPt2di mSzImDispX; // Size of image
		tIm mImDispX;	   // Memory representation of the image
		tDIm *mDImDispX;   // Memory representation of the image

		tPt2di mSzImDispY; // Size of image
		tIm mImDispY;	   // Memory representation of the image
		tDIm *mDImDispY;   // Memory representation of the image

		tPt2di mSzImDiffDispX; // Size of image
		tIm mImDiffDispX;	   // Memory representation of the image
		tDIm *mDImDiffDispX;   // Memory representation of the image

		tPt2di mSzImDiffDispY; // Size of image
		tIm mImDiffDispY;	   // Memory representation of the image
		tDIm *mDImDiffDispY;   // Memory representation of the image

		tPt2di mSzImTranslatedDispX; // Size of image
		tIm mImTranslatedDispX;		 // Memory representation of the image
		tDIm *mDImTranslatedDispX;	 // Memory representation of the image

		tPt2di mSzImTranslatedDispY; // Size of image
		tIm mImTranslatedDispY;		 // Memory representation of the image
		tDIm *mDImTranslatedDispY;	 // Memory representation of the image

		cTriangulation2D<tREAL8> mDelTri;	// A Delaunay triangle

		cDiffInterpolator1D *mInterpol;	   	// Interpolator, if it exists use linear/grad instead of bilinear one
	};
}	// MMVII

#endif // _COMPUTENODESPRECISION_H_