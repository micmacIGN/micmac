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

		void ApplyAndSaveDelaunayTriangulationOnPoints();
		void GeneratePointsForDelaunay();
		void ConstructUniformRandomVector();
		void ComputeShiftedTrianglesAndPoints(const cTriangle<tREAL8, 2> & aTri, std::vector<cPt2dr> & ShiftedTriangleCoordinates,
											  const double aUniformRandomRedChannel, const double aUniformRandomGreenChannel,
											  const double aUniformRandomBlueChannel);

	private:
		// ==   Mandatory args ====
		
		std::string mNameInputImage;
		std::string mNamePlyFile;
		int mNumberPointsToGenerate;	 // number of generated points

		// ==   Optionnal args ====

		int mRandomUniformLawUpperBoundXAxis; // Uniform law generate numbers from [0, mRandomUniformLawUpperBound [ for x-axis
		int mRandomUniformLawUpperBoundYAxis; // Uniform law generate numbers from [0, mRandomUniformLawUpperBound [ for y-axis
		bool mPlyFileisBinary;
		bool mShiftTriangles;
		std::string mNameModifedTrianglePlyFile;

		// ==    Internal variables ====
		
		tIm mImIn;    ///<  memory representation of the image
        tDIm *mDImIn; ///<  memory representation of the image
        cPt2di mSz; // Size of images
		std::vector<cPt2dr> mVectorPts; // A vector containing a set of points
	};

	cAppli_RandomGeneratedDelaunay::cAppli_RandomGeneratedDelaunay(const std::vector<std::string> &aVArgs,
																   const cSpecMMVII_Appli &aSpec) : cMMVII_Appli(aVArgs, aSpec),
																   									mRandomUniformLawUpperBoundXAxis(1),
																									mRandomUniformLawUpperBoundYAxis(1),
																									mPlyFileisBinary(false),
																									mShiftTriangles(true),
																									mImIn(cPt2di(1, 1)),
																									mDImIn(nullptr)
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
				<< AOpt2007(mRandomUniformLawUpperBoundXAxis, "RandomUniformLawXAxis", "Maximum value that the uniform law can draw from for x-axis.", {eTA2007::HDV})
				<< AOpt2007(mRandomUniformLawUpperBoundYAxis, "RandomUniformLawYAxis", "Maximum value that the uniform law can draw from for y-axis.", {eTA2007::HDV})
				<< AOpt2007(mShiftTriangles, "ShiftTriangles", "Whether to shift points of triangles after application of Delaunay triangulation.", {eTA2007::HDV})
				<< AOpt2007(mPlyFileisBinary, "PlyFileIsBinary", "Whether to save the .ply file binarised or not.", {eTA2007::HDV})
				<< AOpt2007(mNameModifedTrianglePlyFile, "NamePlyFileShiftedTriangles", "Name of .ply file for shifted triangles.", {eTA2007::FileCloud});
	}

	//=========================================================

	void cAppli_RandomGeneratedDelaunay::ComputeShiftedTrianglesAndPoints(const cTriangle<tREAL8, 2> & aTri,
																		  std::vector<cPt2dr> & ShiftedTriangleCoordinates,
																		  const double aUniformRandomRedChannel, 
																		  const double aUniformRandomGreenChannel,
																		  const double aUniformRandomBlueChannel)
	{
		const cTriangle2DCompiled aCompTri(aTri);

		// Compute shifts depending on point of triangle
		const cPt2dr PercentDiffA = 0.01 * (aTri.Pt(0) - aTri.Pt(2));
		const cPt2dr PercentDiffB = 0.015 * (aTri.Pt(1) - aTri.Pt(0));
		const cPt2dr PercentDiffC = 0.02 * (aTri.Pt(2) - aTri.Pt(1));

		ShiftedTriangleCoordinates.push_back(aTri.Pt(0) + PercentDiffA);
		ShiftedTriangleCoordinates.push_back(aTri.Pt(1) + PercentDiffB);
		ShiftedTriangleCoordinates.push_back(aTri.Pt(2) + PercentDiffC);

		// Get pixels inside each triangle and shift them
		std::vector<cPt2di> aVectorToFillwithInsidePixels;
		aCompTri.PixelsInside(aVectorToFillwithInsidePixels);
		for (long unsigned int FilledPixel=0; FilledPixel < aVectorToFillwithInsidePixels.size(); FilledPixel++)
		{
			if (FilledPixel % 10 == 0)
			{
				const cPt2dr aFilledPoint(aVectorToFillwithInsidePixels[FilledPixel].x(), aVectorToFillwithInsidePixels[FilledPixel].y());
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

	void cAppli_RandomGeneratedDelaunay::ApplyAndSaveDelaunayTriangulationOnPoints()
	{
		tTriangule2dr aDelTri(mVectorPts);

		aDelTri.MakeDelaunay();

		std::vector<cPt2dr> ShiftedTriangleCoordinates;

		// Loop over all triangle
		for (size_t aKt = 0; aKt < aDelTri.NbFace(); aKt++)
		{
			const cTriangle<tREAL8, 2> aTri = aDelTri.KthTri(aKt);

			if (mShiftTriangles)
			{
				// for colouring points in representation
				const double aUniformRandomRedChannel = RandUnif_N(256);
				const double aUniformRandomGreenChannel = RandUnif_N(256);
				const double aUniformRandomBlueChannel = RandUnif_N(256);
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
		aDelTri.WriteFile(mNamePlyFile, mPlyFileisBinary);
	}

	void cAppli_RandomGeneratedDelaunay::ConstructUniformRandomVector()
	{
		// Use current time as seed for random generator 
    	srand(time(0)); 
		// Generate coordinates from drawing lines and columns of coordinates from a uniform distribution
		for (int aNbPt = 0; aNbPt < mNumberPointsToGenerate; aNbPt++)
		{
			const double aUniformRandomXAxis = RandUnif_N(mRandomUniformLawUpperBoundXAxis);
			const double aUniformRandomYAxis = RandUnif_N(mRandomUniformLawUpperBoundYAxis);
			const cPt2dr aUniformRandomPt(aUniformRandomXAxis, aUniformRandomYAxis); // cPt2dr format
			mVectorPts.push_back(aUniformRandomPt);
		}
	}

	void cAppli_RandomGeneratedDelaunay::GeneratePointsForDelaunay()
	{
		// If user hasn't defined another value than the default value, it is changed
        if (mRandomUniformLawUpperBoundXAxis == 1 && mRandomUniformLawUpperBoundXAxis == 1)
        {
            // Maximum value of coordinates are drawn from [0, NumberOfImageLines[
            mRandomUniformLawUpperBoundXAxis = mSz.x();
			mRandomUniformLawUpperBoundYAxis = mSz.y();            
        }
        else
        {
            if (mRandomUniformLawUpperBoundXAxis != 1 && mRandomUniformLawUpperBoundYAxis == 1)
                mRandomUniformLawUpperBoundYAxis = mSz.y();
            else
            {
                if (mRandomUniformLawUpperBoundXAxis == 1 && mRandomUniformLawUpperBoundYAxis != 1)
                    mRandomUniformLawUpperBoundXAxis = mSz.x();
            }
        }

		ConstructUniformRandomVector();

		ApplyAndSaveDelaunayTriangulationOnPoints(); // Apply Delaunay triangulation on generated points.
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
		mSz = mDImIn->Sz();

		GeneratePointsForDelaunay();

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
		"Generate random points and apply Delaunay triangulation",
		{eApF::ImProc}, // category
		{eApDT::Image}, // input
		{eApDT::Ply},	// output
		__FILE__);

}; // MMVII
