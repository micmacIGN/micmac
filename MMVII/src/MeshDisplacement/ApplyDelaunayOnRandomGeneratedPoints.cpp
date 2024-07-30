#include "cMMVII_Appli.h"
#include "MMVII_Geom2D.h"

/**
   \file ApplyDelaunayOnRandomGeneratedPoints.cpp

   \brief file for generating random points distributed uniformely
   and applying 2D Delaunay triangulation.

**/

namespace MMVII
{

	/* ==================================================== */
	/*                                                      */
	/*          cAppli_RandomGeneratedDelaunay              */
	/*                                                      */
	/* ==================================================== */

	class cAppli_RandomGeneratedDelaunay : public cMMVII_Appli
	{
	public:
		typedef cIm2D<tREAL4> tIm;
		typedef cDataIm2D<tREAL4> tDIm;
		typedef cTriangulation2D<tREAL8> tTriangule2dr;

		cAppli_RandomGeneratedDelaunay(const std::vector<std::string> &aVArgs,
									   const cSpecMMVII_Appli &aSpec);

		int Exe() override;
		cCollecSpecArg2007 &ArgObl(cCollecSpecArg2007 &anArgObl) override;
		cCollecSpecArg2007 &ArgOpt(cCollecSpecArg2007 &anArgOpt) override;

		void ApplyAndSaveDelaunayTriangulationOnPoints(); // Apply triangulation and save to .ply file
		void DefineValueLimitsForPointGeneration();		  // Define limits of values for uniform law
		void ConstructUniformRandomVector();			  // Build vector with node coordinates drawn from uniform law
		void GeneratePointsForRectangularGrid();		  // Build vector with node coordinates that form a rectangular grid
		// Shift triangles by a certain quantity defined in method
		void ComputeShiftedTrianglesAndPoints(const cTriangle<tREAL8, 2> &aTri, std::vector<cPt2dr> &ShiftedTriangleCoordinates,
											  const tREAL8 aUniformRandomRedChannel, const tREAL8 aUniformRandomGreenChannel,
											  const tREAL8 aUniformRandomBlueChannel);

	private:
		// ==   Mandatory args ====

		std::string mNameInputImage; // Name of input image
		std::string mNamePlyFile;	 // Name of .ply file to save delaunay triangles to
		int mNumberPointsToGenerate; // number of generated points

		// ==   Optionnal args ====

		int mNumberOfCols;						 // Uniform law generate numbers from [0, mRandomUniformLawUpperBound [ for x-axis
		int mNumberOfLines;						 // Uniform law generate numbers from [0, mRandomUniformLawUpperBound [ for y-axis
		bool mBuildRandomUniformGrid;			 // Whether to draw coordinates of nodes from uniform law or by rectangular grid
		bool mShiftTriangles;					 // Whether or not to shift triangles
		bool mPlyFileisBinary;					 // Whether the .ply file is binary or not
		std::string mNameModifedTrianglePlyFile; // Name of .ply file to which the shifted triangles can be saved

		// ==    Internal variables ====

		cPt2di mSzImIn;					  // Size of images
		tIm mImIn;						  // memory representation of the image
		tDIm *mDImIn;					  // memory representation of the image
		std::vector<cPt2dr> mVectorPts;	  // A vector containing a set of points
		cTriangulation2D<tREAL8> mDelTri; // A delaunay triangle
	};

	cAppli_RandomGeneratedDelaunay::cAppli_RandomGeneratedDelaunay(const std::vector<std::string> &aVArgs,
																   const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
																									mNumberOfCols(1),
																									mNumberOfLines(1),
																									mBuildRandomUniformGrid(true),
																									mShiftTriangles(true),
																									mPlyFileisBinary(false),
																									mNameModifedTrianglePlyFile("ShiftedTriangles.ply"),
																									mSzImIn(cPt2di(1, 1)),
																									mImIn(mSzImIn),
																									mDImIn(nullptr),
																									mVectorPts({tPt2dr(0, 0)}),
																									mDelTri(mVectorPts)
	{
	}

	cCollecSpecArg2007 &cAppli_RandomGeneratedDelaunay::ArgObl(cCollecSpecArg2007 &anArgObl)
	{
		return anArgObl
			   << Arg2007(mNameInputImage, "Name of input image file.", {{eTA2007::FileImage}, {eTA2007::FileDirProj}})
			   << Arg2007(mNamePlyFile, "Name of main triangulation file to save in .ply format.", {{eTA2007::FileCloud}})
			   << Arg2007(mNumberPointsToGenerate, "Number of points you want to generate for triangulation.");
	}

	cCollecSpecArg2007 &cAppli_RandomGeneratedDelaunay::ArgOpt(cCollecSpecArg2007 &anArgOpt)
	{
		return anArgOpt
			   << AOpt2007(mNumberOfCols, "MaximumValueNumberOfCols", "Maximum value that the uniform law can draw from for x-axis.", {eTA2007::HDV})
			   << AOpt2007(mNumberOfLines, "MaximumValueNumberOfLines", "Maximum value that the uniform law can draw from for y-axis.", {eTA2007::HDV})
			   << AOpt2007(mBuildRandomUniformGrid, "GenerateRandomUniformGrid",
						   "Whether to build a grid to be triangulated thanks to points generated randomly with a uniform law or build a grid made of rectangles.", {eTA2007::HDV})
			   << AOpt2007(mShiftTriangles, "ShiftTriangles", "Whether to shift points of triangles after application of Delaunay triangulation.", {eTA2007::HDV})
			   << AOpt2007(mPlyFileisBinary, "PlyFileIsBinary", "Whether to save the .ply file binarised or not.", {eTA2007::HDV})
			   << AOpt2007(mNameModifedTrianglePlyFile, "NamePlyFileShiftedTriangles", "Name of .ply file for shifted triangles.", {eTA2007::HDV, eTA2007::FileCloud});
	}

	//=========================================================

	void cAppli_RandomGeneratedDelaunay::ApplyAndSaveDelaunayTriangulationOnPoints()
	{
		mDelTri.MakeDelaunay();

		std::vector<cPt2dr> ShiftedTriangleCoordinates;

		const int aMaxValueUniformLaw = 256;

		// Loop over all triangle
		for (size_t aKt = 0; aKt < mDelTri.NbFace(); aKt++)
		{
			const cTriangle<tREAL8, 2> aTri = mDelTri.KthTri(aKt);

			if (mShiftTriangles)
			{
				// for colouring points in representation
				const double aUniformRandomRedChannel = RandUnif_N(aMaxValueUniformLaw);
				const double aUniformRandomGreenChannel = RandUnif_N(aMaxValueUniformLaw);
				const double aUniformRandomBlueChannel = RandUnif_N(aMaxValueUniformLaw);
				ComputeShiftedTrianglesAndPoints(aTri, ShiftedTriangleCoordinates, aUniformRandomRedChannel,
												 aUniformRandomGreenChannel, aUniformRandomBlueChannel);
			}
		}
		if (mShiftTriangles)
		{
			tTriangule2dr aModifiedDelTri(ShiftedTriangleCoordinates);

			aModifiedDelTri.MakeDelaunay();
			// Save files to .ply format
			aModifiedDelTri.WriteFile(mNameModifedTrianglePlyFile, mPlyFileisBinary);
		}

		// Save files to .ply format
		mDelTri.WriteFile(mNamePlyFile, mPlyFileisBinary);
	}

	void cAppli_RandomGeneratedDelaunay::ConstructUniformRandomVector()
	{
		// Use current time as seed for random generator
		srand(time(0));
		// Generate coordinates from drawing lines and columns of coordinates from a uniform distribution
		for (int aNbPt = 0; aNbPt < mNumberPointsToGenerate; aNbPt++)
		{
			const double aUniformRandomXAxis = RandUnif_N(mNumberOfCols);
			const double aUniformRandomYAxis = RandUnif_N(mNumberOfLines);
			const cPt2dr aUniformRandomPt(aUniformRandomXAxis, aUniformRandomYAxis); // cPt2dr format
			mVectorPts.push_back(aUniformRandomPt);
		}

		mDelTri = mVectorPts;
	}

	void cAppli_RandomGeneratedDelaunay::GeneratePointsForRectangularGrid()
	{
		std::vector<tPt2dr> aGridVector;
		const int anEdge = 10; // To take away variations linked to edges

		const int aDistanceLines = mNumberOfLines / std::sqrt(mNumberPointsToGenerate);
		const int aDistanceCols = mNumberOfCols / std::sqrt(mNumberPointsToGenerate);

		for (int aLineNumber = anEdge; aLineNumber < mNumberOfLines; aLineNumber += aDistanceLines)
		{
			for (int aColNumber = anEdge; aColNumber < mNumberOfCols; aColNumber += aDistanceCols)
			{
				const tPt2dr aGridPt = tPt2dr(aColNumber, aLineNumber);
				aGridVector.push_back(aGridPt);
			}
		}

		mDelTri = aGridVector;
	}

	void cAppli_RandomGeneratedDelaunay::DefineValueLimitsForPointGeneration()
	{
		// If user hasn't defined another value than the default value, it is changed
		if (mNumberOfCols == 1 && mNumberOfCols == 1)
		{
			// Maximum value of coordinates are drawn from [0, NumberOfImageLines[
			mNumberOfCols = mSzImIn.x();
			mNumberOfLines = mSzImIn.y();
		}
		else
		{
			if (mNumberOfCols != 1 && mNumberOfLines == 1)
				mNumberOfLines = mSzImIn.y();
			else
			{
				if (mNumberOfCols == 1 && mNumberOfLines != 1)
					mNumberOfCols = mSzImIn.x();
			}
		}
		if (mBuildRandomUniformGrid)
			ConstructUniformRandomVector();
		else
			GeneratePointsForRectangularGrid();

		ApplyAndSaveDelaunayTriangulationOnPoints(); // Apply Delaunay triangulation on generated points.
	}

	void cAppli_RandomGeneratedDelaunay::ComputeShiftedTrianglesAndPoints(const cTriangle<tREAL8, 2> &aTri,
																		  std::vector<cPt2dr> &ShiftedTriangleCoordinates,
																		  const tREAL8 aUniformRandomRedChannel,
																		  const tREAL8 aUniformRandomGreenChannel,
																		  const tREAL8 aUniformRandomBlueChannel)
	{
		const cTriangle2DCompiled aCompTri(aTri);

		// Compute shifts depending on point of triangle
		const cPt2dr PercentDiffA = 0.01 * (aTri.Pt(0) - aTri.Pt(2));
		const cPt2dr PercentDiffB = 0.015 * (aTri.Pt(1) - aTri.Pt(0)); // arbitrary values are chosen for displacement
		const cPt2dr PercentDiffC = 0.02 * (aTri.Pt(2) - aTri.Pt(1));

		ShiftedTriangleCoordinates.push_back(aTri.Pt(0) + PercentDiffA);
		ShiftedTriangleCoordinates.push_back(aTri.Pt(1) + PercentDiffB);
		ShiftedTriangleCoordinates.push_back(aTri.Pt(2) + PercentDiffC);

		// Get pixels inside each triangle and shift them
		std::vector<cPt2di> aVectorToFillwithInsidePixels;
		aCompTri.PixelsInside(aVectorToFillwithInsidePixels);
		for (size_t aFilledPixel = 0; aFilledPixel < aVectorToFillwithInsidePixels.size(); aFilledPixel++)
		{
			if (aFilledPixel % 40 == 0)
			{
				const cPt2dr aFilledPoint(aVectorToFillwithInsidePixels[aFilledPixel].x(), aVectorToFillwithInsidePixels[aFilledPixel].y());
				const cPt3dr barycenter_coordinates = aCompTri.CoordBarry(aFilledPoint);
				const cPt2dr ShiftedInsidePixels = cPt2dr(aFilledPoint.x() + barycenter_coordinates.x() * PercentDiffA.x() +
															  barycenter_coordinates.y() * PercentDiffB.x() + barycenter_coordinates.z() * PercentDiffC.x(),
														  aFilledPoint.y() + barycenter_coordinates.x() * PercentDiffA.y() +
															  barycenter_coordinates.y() * PercentDiffB.y() + barycenter_coordinates.z() * PercentDiffC.y());

				StdOut() << aFilledPoint.x() << " " << aFilledPoint.y() << " " << aUniformRandomRedChannel
						 << " " << aUniformRandomGreenChannel << " " << aUniformRandomBlueChannel << std::endl;
				StdOut() << ShiftedInsidePixels.x() << " " << ShiftedInsidePixels.y() << " "
						 << aUniformRandomRedChannel << " " << aUniformRandomGreenChannel << " " << aUniformRandomBlueChannel << std::endl;
			}
		}
	}

	//----------------------------------------

	int cAppli_RandomGeneratedDelaunay::Exe()
	{
		/*
		MMVII RandomGeneratedDelaunay pair18_im1_720.png OriginalTriangles.ply 20 50 NamePlyFileShiftedTriangles=ShiftedTriangles.ply > OriginalAndShiftedPoints.xyz
		awk NR%2==1 < OriginalAndShiftedPoints.xyz > OriginalPoints.xyz
		awk NR%2==0 < OriginalAndShiftedPoints.xyz > ShiftedPoints.xyz
		*/

		mImIn = tIm::FromFile(mNameInputImage);

		mDImIn = &mImIn.DIm();
		mSzImIn = mDImIn->Sz();

		DefineValueLimitsForPointGeneration();

		return EXIT_SUCCESS;
	}

	/* ================================== */
	/*                                    */
	/*               MMVII                */
	/*                                    */
	/* ================================== */

	tMMVII_UnikPApli Alloc_RandomGeneratedDelaunay(const std::vector<std::string> &aVArgs, const cSpecMMVII_Appli &aSpec)
	{
		return tMMVII_UnikPApli(new cAppli_RandomGeneratedDelaunay(aVArgs, aSpec));
	}

	cSpecMMVII_Appli TheSpec_RandomGeneratedDelaunay(
		"RandomGeneratedDelaunay",
		Alloc_RandomGeneratedDelaunay,
		"Generate random points thanks to uniform law and apply Delaunay triangulation",
		{eApF::ImProc}, // category
		{eApDT::Image}, // input
		{eApDT::Ply},	// output
		__FILE__);

}; // MMVII
