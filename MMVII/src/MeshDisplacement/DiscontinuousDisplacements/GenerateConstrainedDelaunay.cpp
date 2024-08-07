#include "cMMVII_Appli.h"
#include "MMVII_Image2D.h"

#include "MMVII_TrianglePP.h"

/**
	\file GenerateConstrainedDelaunay.cpp

	\brief file for generating a constrained Delaunay triangulation with
    a MicMacV2 command using the TrianglePP library. The library is
    available at the following link : https://github.com/mrkkrj/TrianglePP
 **/

namespace MMVII
{
	/******************************************/
	/*                                        */
	/*   cAppli_GenerateConstrainedDelaunay   */
	/*                                        */
	/******************************************/

	class cAppli_GenerateConstrainedDelaunay : public cMMVII_Appli
	{
	public:
        using Point = tpp::Delaunay::Point;

		typedef cIm2D<tREAL8> tIm;
		typedef cDataIm2D<tREAL8> tDIm;

        // Constructor
		cAppli_GenerateConstrainedDelaunay(const std::vector<std::string> &aVArgs,
									       const cSpecMMVII_Appli &aSpec);
        // Destructor
        ~cAppli_GenerateConstrainedDelaunay();

		// Exe and argument methods
		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;
		
		// Gets vector of integer given by user and loads a vector with tpp::Delaunay::Points
		void TransformVectorOfIntegerCoordinatesToPointCoordinatesVector(std::vector<Point> &aVectorOfPoints,
																		 const std::vector<int> &aVectorOfCoordinates);
		// Generates rectangular grid if user doesn't want to enter coordinates manually
		void GeneratePointsForRectangleGrid(const int aNumberOfPoints, const int aGridSizeLines,
											const int aGridSizeCols, std::vector<int> &aRectGridVector);

	private:

		// ==== Mandatory args ====

		std::string mNameImage;             			// Name of the input image to draw segment line on
        // Vector of point coordinates of mesh in the form [x1-coordinate, y1-coordinate, x2-coordinate, y2-coordinate, ..., xn-coordinate, yn-coordinate]
        std::vector<int> mVectorOfNodeCoordinates;
        bool mLinkConstraintsWithIds;     				// Whether the segments are drawn by ids of points or with coordinates of points
        std::vector<int> mVectorOfConstraintSegments;	// Vector containing Ids of Points used for constraints or tpp::Delaunay::Points coordinates

		// ==== Optional args ====

		bool mUseConvexHull;			  	// Whether or not to use convex hull during triangulation
		bool mSaveSegments;					// Whether or not to save segments in a .poly file
		std::string mSegmentFileName;		// Name to use for saving segments in a .poly file
		bool mSaveTriangulation;			// Whether or not to save triangulation result in a .off file
		std::string mTriangulationFileName;	// Name to use for saving triangulation operation in a .off file
		int mNumberPointsToGenerate; 		// Number of generated points

		// ==== Internal variables ====

		tPt2di mSz;	   // Size of image
		tIm mImIn;	   // Memory representation of the image
		tDIm *mDImIn;  // Memory representation of the image
	};

	cAppli_GenerateConstrainedDelaunay::cAppli_GenerateConstrainedDelaunay(
		const std::vector<std::string> &aVArgs,
		const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
										 mUseConvexHull(true),
										 mSaveSegments(false),
										 mSegmentFileName("UsedSegments.poly"),
										 mSaveTriangulation(false),
										 mTriangulationFileName("ConstrainedTriangulation.off"),
										 mNumberPointsToGenerate(0),
										 mSz(tPt2di(1, 1)),
										 mImIn(mSz),
										 mDImIn(nullptr)
	{
	}

    cAppli_GenerateConstrainedDelaunay::~cAppli_GenerateConstrainedDelaunay()
    {
    }

	cCollecSpecArg2007 &cAppli_GenerateConstrainedDelaunay::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl << Arg2007(mNameImage, "Name of image on which to set a discontinuity.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
               			<< Arg2007(mVectorOfNodeCoordinates, "Vector containing coordinates of mesh nodes.")
						<< Arg2007(mLinkConstraintsWithIds, "Whether to link points of constraints by point ids or use point coordinates.")
            			<< Arg2007(mVectorOfConstraintSegments, "Ids or points coordinates of constraint nodes.");
	}

	cCollecSpecArg2007 &cAppli_GenerateConstrainedDelaunay::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{
		return anArgOpt << AOpt2007(mUseConvexHull, "UseConvexHull", "Whether or not to use convex hull during triangulation", {eTA2007::HDV})
						<< AOpt2007(mSaveSegments, "SaveSegments", "Whether to save drawn segments or not in .poly file.", {eTA2007::HDV})
						<< AOpt2007(mSegmentFileName, "SegmentFileName", "Name to use for saving segments in a .poly file", {eTA2007::HDV})
						<< AOpt2007(mSaveTriangulation, "SaveTriangulationResult", "Whether or not to save triangulation result in a .off file", {eTA2007::HDV})
						<< AOpt2007(mTriangulationFileName, "TriangulationFileName", "Name to use for saving triangulation operation in a .off file", {eTA2007::HDV})
						<< AOpt2007(mNumberPointsToGenerate, "NumberOfPointsToGenerate", "Number of points you want to generate for triangulation.", {eTA2007::HDV});
	}

	//------------------------------------------//

	void cAppli_GenerateConstrainedDelaunay::GeneratePointsForRectangleGrid(const int aNumberOfPoints, const int aGridSizeLines,
																			const int aGridSizeCols, std::vector<int> &aRectGridVector)
	{
		const int anEdge = 10; // To take away variations linked to edges

		// Be aware that step between grid points are integers
		const int aDistanceLines = aGridSizeLines / std::sqrt(aNumberOfPoints);
		const int aDistanceCols = aGridSizeCols / std::sqrt(aNumberOfPoints);

		const int anEndLinesLoop = aGridSizeLines - anEdge;
		const int anEndColsLoop = aGridSizeCols - anEdge;

		for (int aLineNumber = anEdge; aLineNumber < anEndLinesLoop; aLineNumber += aDistanceLines)
		{
			for (int aColNumber = anEdge; aColNumber < anEndColsLoop; aColNumber += aDistanceCols)
			{
				const int aRectGridColCoordinate = aColNumber;
				const int aRectGridLineCoordinate = aLineNumber;
				aRectGridVector.push_back(aRectGridColCoordinate);
				aRectGridVector.push_back(aRectGridLineCoordinate);
			}
		}
	}

	void cAppli_GenerateConstrainedDelaunay::TransformVectorOfIntegerCoordinatesToPointCoordinatesVector(std::vector<Point> &aVectorOfPoints,
																										 const std::vector<int> &aVectorOfNodeCoordinates)
	{
        for (size_t aNodeCoordinate=0; aNodeCoordinate < aVectorOfNodeCoordinates.size() - 1; aNodeCoordinate += 2)
            aVectorOfPoints.push_back(Point(aVectorOfNodeCoordinates[aNodeCoordinate], aVectorOfNodeCoordinates[aNodeCoordinate + 1]));
	}

	//------------------------------------------//

	int cAppli_GenerateConstrainedDelaunay::Exe()
	{
		mImIn = tIm::FromFile(mNameImage);

		mDImIn = &mImIn.DIm();
		mSz = mDImIn->Sz();

        std::vector<Point> aPSLGDelaunayInput;

		if (mVectorOfNodeCoordinates.empty())
			GeneratePointsForRectangleGrid(mNumberPointsToGenerate, mSz.y(), mSz.x(), mVectorOfNodeCoordinates);

		TransformVectorOfIntegerCoordinatesToPointCoordinatesVector(aPSLGDelaunayInput, mVectorOfNodeCoordinates);

	   	// Initialise a constrained triangle object
        tpp::Delaunay aTriConstrGenerator(aPSLGDelaunayInput);

        std::vector<int> aConstrainedDelaunaySegmentsIds;
		std::vector<Point> aConstrainedDelaunaySegmentsPoints;
		
		if (mLinkConstraintsWithIds)
		{
			for (size_t aConstraintId = 0; aConstraintId < mVectorOfConstraintSegments.size(); aConstraintId++)
        		aConstrainedDelaunaySegmentsIds.push_back(mVectorOfConstraintSegments[aConstraintId]);
		}
		else
			TransformVectorOfIntegerCoordinatesToPointCoordinatesVector(aConstrainedDelaunaySegmentsPoints, mVectorOfConstraintSegments);

        // segment-constrained triangulation
        aTriConstrGenerator.setSegmentConstraint(aConstrainedDelaunaySegmentsIds);

		if (mUseConvexHull)
        	aTriConstrGenerator.useConvexHullWithSegments(true); // don't remove concavities!

        aTriConstrGenerator.Triangulate();

		if (mSaveSegments)
			aTriConstrGenerator.saveSegments(mSegmentFileName);
		if (mSaveTriangulation)
        	aTriConstrGenerator.writeoff(mTriangulationFileName);

		return EXIT_SUCCESS;
	}

	/********************************************/
	//              ::MMVII                     //
	/********************************************/

	tMMVII_UnikPApli Alloc_cAppli_GenerateConstrainedDelaunay(const std::vector<std::string> &aVArgs,
														      const cSpecMMVII_Appli &aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_GenerateConstrainedDelaunay(aVArgs, aSpec));
	}

	cSpecMMVII_Appli TheSpec_GenerateConstrainedDelaunay(
		"GenerateConstrainedDelaunay",
		Alloc_cAppli_GenerateConstrainedDelaunay,
		"Generates Delaunay triangulation with constrained segment line",
		{eApF::ImProc}, // category
		{eApDT::Image}, // input
		{eApDT::Image}, // output
		__FILE__);

}; // namespace MMVII
