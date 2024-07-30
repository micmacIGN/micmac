#include "TriangleDeformationUtils.h"

/**
 \file TriangleDeformationUtils.cpp
 \brief File containing annexe methods that can be used by
 other classes linked to triangle deformation computation.
**/

namespace MMVII
{
	/****************************************/
	/*                                      */
	/*          cPtInsideTriangles          */
	/*                                      */
	/****************************************/

	cPtInsideTriangles::cPtInsideTriangles(const cTriangle2DCompiled<tREAL8> &aCompTri,				 // A compiled triangle
										   const std::vector<tPt2di> &aVectorFilledwithInsidePixels, // Vector containing pixels insisde triangles
										   const size_t aFilledPixel,								 // A counter that is looping over pixels in triangles
										   tDIm *&aDIm,												 // Image
										   cDiffInterpolator1D *&anInterpolator)					 // An interpolator
	{
		mFilledIndices = tPt2dr(aVectorFilledwithInsidePixels[aFilledPixel].x(), aVectorFilledwithInsidePixels[aFilledPixel].y());
		mBarycenterCoordinatesOfPixel = aCompTri.CoordBarry(mFilledIndices);
		(aDIm->InsideInterpolator(*anInterpolator, mFilledIndices, 0)) ? mValueOfPixel = aDIm->GetValueInterpol(*anInterpolator, mFilledIndices) : mValueOfPixel = aDIm->GetV(tPt2di(mFilledIndices.x(), mFilledIndices.y()));
	}

	cPt3dr cPtInsideTriangles::GetBarycenterCoordinates() const { return mBarycenterCoordinatesOfPixel; } // Accessor
	tPt2dr cPtInsideTriangles::GetCartesianCoordinates() const { return mFilledIndices; }				  // Accessor
	tREAL8 cPtInsideTriangles::GetPixelValue() const { return mValueOfPixel; }							  // Accessor

	/****************************************/
	/*                                      */
	/*           cNodeOfTriangles           */
	/*                                      */
	/****************************************/

	cNodeOfTriangles::cNodeOfTriangles()
	{
	}

	cNodeOfTriangles::cNodeOfTriangles(const tDenseVect &aVecSol,
									   const tIntVect &aIndicesVec,
									   const int aXIndices,
									   const int aYIndices,
									   const int aRadTrIndices,
									   const int aRadScIndices,
									   const tTri2dr &aTri,
									   const int aPointNumberInTri)
	{
		mInitialNodeCoordinates = aTri.Pt(aPointNumberInTri);
		mCurXYDisplacementVector = tPt2dr(aVecSol(aIndicesVec.at(aXIndices)),
										  aVecSol(aIndicesVec.at(aYIndices)));
		mCurRadTr = aVecSol(aIndicesVec.at(aRadTrIndices));
		mCurRadSc = aVecSol(aIndicesVec.at(aRadScIndices));
	}

	cNodeOfTriangles::cNodeOfTriangles(const tDenseVect &aVecSol,
									   const tIntVect &aIndicesVec,
									   const int aXIndices,
									   const int aYIndices,
									   const int aRadTrIndices,
									   const int aRadScIndices,
									   const tTri2dr &aTri,
									   const tPt3di &aFace,
									   const int aPointNumberInTri,
									   const int anIdOfPoint) : mIdOfPt(anIdOfPoint)
	{
		mInitialNodeCoordinates = aTri.Pt(aPointNumberInTri);
		mCurXYDisplacementVector = tPt2dr(aVecSol(aIndicesVec.at(aXIndices)),
										  aVecSol(aIndicesVec.at(aYIndices)));
		mCurRadTr = aVecSol(aIndicesVec.at(aRadTrIndices));
		mCurRadSc = aVecSol(aIndicesVec.at(aRadScIndices));

		if (aPointNumberInTri == 0)
			mFaceOfTriangle = aFace.x();
		else if (aPointNumberInTri == 1)
			mFaceOfTriangle = aFace.y();
		else
			mFaceOfTriangle = aFace.z();
	}

	cNodeOfTriangles::~cNodeOfTriangles()
	{
	}

	tPt2dr cNodeOfTriangles::GetInitialNodeCoordinates() const { return mInitialNodeCoordinates; }		 // Accessor
	tPt2dr cNodeOfTriangles::GetCurrentXYDisplacementValues() const { return mCurXYDisplacementVector; } // Accessor
	tREAL8 cNodeOfTriangles::GetCurrentRadiometryScaling() const { return mCurRadSc; }					 // Accessor
	tREAL8 cNodeOfTriangles::GetCurrentRadiometryTranslation() const { return mCurRadTr; }				 // Accessor
	int cNodeOfTriangles::GetPointId() const { return mIdOfPt; }										 // Accessor
	int cNodeOfTriangles::GetTriangleFace() const { return mFaceOfTriangle; }							 // Accessor

	tPt2dr &cNodeOfTriangles::GetInitialNodeCoordinates() { return mInitialNodeCoordinates; }		// Accessor
	tPt2dr &cNodeOfTriangles::GetCurrentXYDisplacementValues() { return mCurXYDisplacementVector; } // Accessor
	tREAL8 &cNodeOfTriangles::GetCurrentRadiometryTranslation() { return mCurRadTr; }				// Accessor
	tREAL8 &cNodeOfTriangles::GetCurrentRadiometryScaling() { return mCurRadSc; }					// Accessor
	int &cNodeOfTriangles::GetPointId() { return mIdOfPt; }											// Accessor
	int &cNodeOfTriangles::GetTriangleFace() { return mFaceOfTriangle; }							// Accessor

	std::ostream &operator<<(std::ostream &os, const cNodeOfTriangles &obj)
	{
		obj.ShowTriangleNodeCarateristics();
		return os;
	}

	void cNodeOfTriangles::AddData(const cAuxAr2007 &anAux, cNodeOfTriangles &aPtToSerialise)
	{
		MMVII::AddData(cAuxAr2007("Id", anAux), aPtToSerialise.GetPointId());
		MMVII::AddData(cAuxAr2007("Face", anAux), aPtToSerialise.GetTriangleFace());
		MMVII::AddData(cAuxAr2007("x", anAux), aPtToSerialise.GetInitialNodeCoordinates().x());
		MMVII::AddData(cAuxAr2007("y", anAux), aPtToSerialise.GetInitialNodeCoordinates().y());
		MMVII::AddData(cAuxAr2007("dx", anAux), aPtToSerialise.GetCurrentXYDisplacementValues().x());
		MMVII::AddData(cAuxAr2007("dy", anAux), aPtToSerialise.GetCurrentXYDisplacementValues().y());
		MMVII::AddData(cAuxAr2007("RadiometryTranslation", anAux), aPtToSerialise.GetCurrentRadiometryTranslation());
		MMVII::AddData(cAuxAr2007("RadiometryScaling", anAux), aPtToSerialise.GetCurrentRadiometryScaling());
	}

	void AddData(const cAuxAr2007 &anAux, cNodeOfTriangles &aPtToSerialise) { aPtToSerialise.AddData(anAux, aPtToSerialise); }

	void cNodeOfTriangles::ShowTriangleNodeCarateristics() const
	{
		StdOut() << "Id of this point : " << this->GetPointId() << std::endl;
		StdOut() << "Face of triangle associated to point : " << this->GetTriangleFace() << std::endl;
		StdOut() << "Initial node coordinates : " << this->GetInitialNodeCoordinates() << "." << std::endl;
		StdOut() << "Current displacement coefficient values : " << this->GetCurrentXYDisplacementValues()
				 << "." << std::endl;
		StdOut() << "Current radiometric coefficient values : " << this->GetCurrentRadiometryTranslation()
				 << " for translation and " << this->GetCurrentRadiometryScaling() << " for scaling." << std::endl;
	}

	void cNodeOfTriangles::SaveTriangleNodeToFile() const
	{
		SaveInFile(*this, NameFileToSaveNode(mIdOfPt));
	}

	std::unique_ptr<cNodeOfTriangles> cNodeOfTriangles::ReadSerialisedTriangleNode(const tDenseVect &aVecSol, const tIntVect &aIndVec,
																				   const int aXInd, const int aYInd, const int aRadTrInd,
																				   const int aRadScInd, const tTri2dr &aTriangle, const tPt3di &aFace,
																				   const int aPointNumberInTri, const int anIdOfPoint)
	{
		std::unique_ptr<cNodeOfTriangles> aReReadSerialisedObj = std::make_unique<cNodeOfTriangles>(aVecSol, aIndVec, aXInd, aYInd, aRadTrInd,
																									aRadScInd, aTriangle, aFace, aPointNumberInTri,
																									anIdOfPoint);
		ReadFromFile(*aReReadSerialisedObj, NameFileToSaveNode(anIdOfPoint));

		return aReReadSerialisedObj;
	}

	std::string cNodeOfTriangles::NameFileToSaveNode(const int anId)
	{
		return "Id_" + ToStr(anId) + ".xml";
	}

	/************************************************/
	/*                                              */
	/*       cMultipleTriangleNodesSerialiser       */
	/*                                              */
	/************************************************/

	cMultipleTriangleNodesSerialiser::cMultipleTriangleNodesSerialiser()
	{
	}

	cMultipleTriangleNodesSerialiser::cMultipleTriangleNodesSerialiser(const std::string &aFileName) : mName(aFileName)
	{
	}

	cMultipleTriangleNodesSerialiser::~cMultipleTriangleNodesSerialiser()
	{
	}

	std::unique_ptr<cMultipleTriangleNodesSerialiser> cMultipleTriangleNodesSerialiser::NewMultipleTriangleNodes(const std::string &aName)
	{
		std::unique_ptr<cMultipleTriangleNodesSerialiser> aNewMultipleTriangleNodes = std::make_unique<cMultipleTriangleNodesSerialiser>(aName);
		return aNewMultipleTriangleNodes;
	}

	void cMultipleTriangleNodesSerialiser::AddData(const cAuxAr2007 &anAux)
	{
		MMVII::AddData(cAuxAr2007("VectorOfTriangleNodes", anAux), mVectorTriangleNodes);
	}
	void AddData(const cAuxAr2007 &anAux, cMultipleTriangleNodesSerialiser &aSetOfObjsToSerialise) { aSetOfObjsToSerialise.AddData(anAux); }

	void cMultipleTriangleNodesSerialiser::MultipleNodesToFile(const std::string &aFileName) const
	{
		SaveInFile(*this, aFileName);
	}

	void cMultipleTriangleNodesSerialiser::PushInVector(const std::unique_ptr<const cNodeOfTriangles> &aTriangleDeformationObj)
	{
		mVectorTriangleNodes.push_back(*aTriangleDeformationObj);
	}

	std::unique_ptr<cMultipleTriangleNodesSerialiser> cMultipleTriangleNodesSerialiser::ReadVectorOfTriangleNodes(const std::string &aFileName)
	{
		std::unique_ptr<cMultipleTriangleNodesSerialiser> aNewSetOfMultipleTriangleDeformations = NewMultipleTriangleNodes(aFileName);
		ReadFromFile(*aNewSetOfMultipleTriangleDeformations, aFileName);

		return aNewSetOfMultipleTriangleDeformations;
	}

	std::string cMultipleTriangleNodesSerialiser::GetName() const { return mName; } // Accessor

	size_t cMultipleTriangleNodesSerialiser::GetNumberOfVectorTriangleNodes() const { return mVectorTriangleNodes.size(); } // Accessor

	std::vector<cNodeOfTriangles> cMultipleTriangleNodesSerialiser::GetVectorOfNodes() const { return mVectorTriangleNodes; } // Accessor

	void cMultipleTriangleNodesSerialiser::ShowAllTriangleNodes(const std::string &aAllOrSingularValue, const int aNodeNumber) const
	{
		if (aAllOrSingularValue == "all")
			StdOut() << "The carateristics of the nodes are : " << mVectorTriangleNodes << std::endl;
		else
			StdOut() << "The carateristics of node number " << aNodeNumber << " are : "
					 << mVectorTriangleNodes.at(aNodeNumber) << std::endl;
	}

	//---------------------------------------------//

	void BuildUniformRandomVectorAndApplyDelaunay(const int aNumberOfPointsToGenerate, const int aRandomUniformLawUpperBoundLines,
												  const int aRandomUniformLawUpperBoundCols, cTriangulation2D<tREAL8> &aDelaunayTri)
	{
		std::vector<tPt2dr> aVectorPts;
		// Generate coordinates from drawing lines and columns of coordinates from a uniform distribution
		for (int aNbPt = 0; aNbPt < aNumberOfPointsToGenerate; aNbPt++)
		{
			const tREAL8 aUniformRandomLine = RandUnif_N(aRandomUniformLawUpperBoundLines);
			const tREAL8 aUniformRandomCol = RandUnif_N(aRandomUniformLawUpperBoundCols);
			const tPt2dr aUniformRandomPt(aUniformRandomCol, aUniformRandomLine); // tPt2dr format
			aVectorPts.push_back(aUniformRandomPt);
		}
		aDelaunayTri = aVectorPts;

		aDelaunayTri.MakeDelaunay(); // Delaunay triangulate randomly generated points.
	}

	void DefineValueLimitsForPointGenerationAndBuildGrid(const int aNumberOfPointsToGenerate, int aNumberOfLines, int aNumberOfCols,
														 cTriangulation2D<tREAL8> &aDelaunayTri, const tPt2di &aSzIm,
														 const bool aBuildUniformVector)
	{
		// If user hasn't defined another value than the default value, it is changed
		if (aNumberOfLines == 1 && aNumberOfCols == 1)
		{
			// Maximum value of coordinates are drawn from [0, NumberOfImageLines[ for lines
			aNumberOfLines = aSzIm.y();
			// Maximum value of coordinates are drawn from [0, NumberOfImageColumns[ for columns
			aNumberOfCols = aSzIm.x();
		}
		else
		{
			if (aNumberOfLines != 1 && aNumberOfCols == 1)
				aNumberOfCols = aSzIm.x();
			else
			{
				if (aNumberOfLines == 1 && aNumberOfCols != 1)
					aNumberOfLines = aSzIm.y();
			}
		}

		if (aBuildUniformVector)
			BuildUniformRandomVectorAndApplyDelaunay(aNumberOfPointsToGenerate,
													 aNumberOfLines, aNumberOfCols,
													 aDelaunayTri);
		else
			GeneratePointsForRectangleGrid(aNumberOfPointsToGenerate, aNumberOfLines,
										   aNumberOfCols, aDelaunayTri);
	}

	void GeneratePointsForRectangleGrid(const int aNumberOfPoints, const int aGridSizeLines,
										const int aGridSizeCols, cTriangulation2D<tREAL8> &aDelaunayTri)
	{
		std::vector<tPt2dr> aRectGridVector;
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
				const tPt2dr aRectGridPt = tPt2dr(aColNumber, aLineNumber); // tPt2dr format
				aRectGridVector.push_back(aRectGridPt);
			}
		}

		aDelaunayTri = aRectGridVector;

		aDelaunayTri.MakeDelaunay(); // Delaunay triangulate rectangular grid generated points.
	}

	void RegenerateTriangulatedGridFromSerialisation(cTriangulation2D<tREAL8> &aDelaunayTri,
													 const std::string &aNameMultipleNodesFile)
	{
		std::vector<tPt2dr> aGridVector;

		const std::unique_ptr<const cMultipleTriangleNodesSerialiser> aReReadMultipleNodeObj = cMultipleTriangleNodesSerialiser::ReadVectorOfTriangleNodes(aNameMultipleNodesFile);

		const std::vector<cNodeOfTriangles> aReReadVectorOfNodes = aReReadMultipleNodeObj->GetVectorOfNodes();

		for (size_t aNodeNumber = 0; aNodeNumber < aReReadVectorOfNodes.size(); aNodeNumber++)
		{
			const tPt2dr aGridPt = tPt2dr(aReReadVectorOfNodes.at(aNodeNumber).GetInitialNodeCoordinates().x(),
										  aReReadVectorOfNodes.at(aNodeNumber).GetInitialNodeCoordinates().y());
			if (!(std::find(aGridVector.begin(), aGridVector.end(), aGridPt) != aGridVector.end()))
				// if aGridPt isn't contained in aGridVector : add it
				aGridVector.push_back(aGridPt);
		}

		aDelaunayTri = aGridVector;

		aDelaunayTri.MakeDelaunay(); // Delaunay triangulate rectangular grid generated points.
	}

	void InitialiseInterpolationAndEquation(cCalculator<tREAL8> *&aEqDeformTri, cDiffInterpolator1D *&aInterpol,
											const std::vector<std::string> &aArgsVectorInterpol, const bool aUseLinearGradInterpolation)
	{
		if (aUseLinearGradInterpolation)
			aInterpol = cDiffInterpolator1D::AllocFromNames(aArgsVectorInterpol);
		// True means with derivative, 1 is size of buffer
		aEqDeformTri = aUseLinearGradInterpolation ? EqDeformTriLinearGrad(true, 1) : EqDeformTriBilin(true, 1);
	}

	void InitialisationWithUserValues(const cTriangulation2D<tREAL8> &aDelaunayTri,
									  tSys *&aSys,
									  const bool aUserInitialisation,
									  const tREAL8 aXTranslationInitVal,
									  const tREAL8 aYTranslationInitVal,
									  const tREAL8 aRadTranslationInitVal,
									  const tREAL8 aRadScaleInitVal)
	{
		const int aNbUnkPerNode = 4;
		const size_t aStartNumberPts = aNbUnkPerNode * aDelaunayTri.NbPts();
		tDenseVect aVInit(aStartNumberPts, eModeInitImage::eMIA_Null);

		const bool aDiffTranslationInitValue = (aXTranslationInitVal != 0 || aYTranslationInitVal != 0) ? true : false;
		const bool aDiffRadiometryInitValue = (aRadTranslationInitVal != 0 || aRadScaleInitVal != 1) ? true : false;

		if (aUserInitialisation && (aDiffTranslationInitValue || aDiffRadiometryInitValue))
		{
			for (size_t aKtNumber = 0; aKtNumber < aStartNumberPts; aKtNumber++)
			{
				if (aKtNumber % aNbUnkPerNode == 0 && aXTranslationInitVal != 0)
					aVInit(aKtNumber) = aXTranslationInitVal;
				if (aKtNumber % aNbUnkPerNode == 1 && aYTranslationInitVal != 0)
					aVInit(aKtNumber) = aYTranslationInitVal;
				if (aKtNumber % aNbUnkPerNode == 2 && aRadTranslationInitVal != 0)
					aVInit(aKtNumber) = aRadTranslationInitVal;
				if (aKtNumber % aNbUnkPerNode == 3)
					aVInit(aKtNumber) = aRadScaleInitVal;
			}
		}
		else
		{
			for (size_t aKtNumber = 0; aKtNumber < aStartNumberPts; aKtNumber++)
			{
				if (aKtNumber % aNbUnkPerNode == 3)
					aVInit(aKtNumber) = 1;
			}
		}

		aSys = new tSys(eModeSSR::eSSR_LsqDense, aVInit);
	}

	void InitialiseWithPreviousExecutionValuesMMVI(const cTriangulation2D<tREAL8> &aDelTri,
												   tSys *&aSys,
												   cDiffInterpolator1D *&anInterpolator,
												   const std::string &aNameDepXFile, tIm &aImDepX,
												   tDIm *&aDImDepX, tPt2di &aSzImDepX,
												   const std::string &aNameDepYFile, tIm &aImDepY,
												   tDIm *&aDImDepY, tPt2di &aSzImDepY,
												   const std::string &aNameCorrelationMask,
												   tIm &aImCorrelationMask, tDIm *&aDImCorrelationMask,
												   tPt2di &aSzCorrelationMask)
	{
		tDenseVect aVInit(4 * aDelTri.NbPts(), eModeInitImage::eMIA_Null);

		ReadImageFileNameLoadData(aNameDepXFile, aImDepX,
								  aDImDepX, aSzImDepX);
		ReadImageFileNameLoadData(aNameDepYFile, aImDepY,
								  aDImDepY, aSzImDepY);

		ReadImageFileNameLoadData(aNameCorrelationMask, aImCorrelationMask,
								  aDImCorrelationMask, aSzCorrelationMask);

		const int aRadiometryScalingInitialValue = 1;

		for (size_t aTr = 0; aTr < aDelTri.NbFace(); aTr++)
		{
			const tTri2dr aInitTri = aDelTri.KthTri(aTr);
			const tPt3di aIndicesOfTriKnots = aDelTri.KthFace(aTr);

			//----------- Index of unknown, finds the associated pixels of current triangle
			tIntVect aInitVecInd;
			GetIndicesVector(aInitVecInd, aIndicesOfTriKnots, 4);

			// Get points coordinates associated to triangle
			const cNodeOfTriangles aFirstInitPointOfTri = cNodeOfTriangles(aVInit, aInitVecInd, 0, 1, 2, 3, aInitTri, 0);
			const cNodeOfTriangles aSecondInitPointOfTri = cNodeOfTriangles(aVInit, aInitVecInd, 4, 5, 6, 7, aInitTri, 1);
			const cNodeOfTriangles aThirdInitPointOfTri = cNodeOfTriangles(aVInit, aInitVecInd, 8, 9, 10, 11, aInitTri, 2);
			// Initialisation of geometrical translation and radiometry scaling is necessary but not radiometry translation
			// as the initiale value of radiometry translation is zero

			aVInit(aInitVecInd.at(0)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepX,
																			 aFirstInitPointOfTri, 0, anInterpolator);
			aVInit(aInitVecInd.at(1)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepY,
																			 aFirstInitPointOfTri, 0, anInterpolator);
			aVInit(aInitVecInd.at(3)) = aRadiometryScalingInitialValue;
			aVInit(aInitVecInd.at(4)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepX,
																			 aSecondInitPointOfTri, 0, anInterpolator);
			aVInit(aInitVecInd.at(5)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepY,
																			 aSecondInitPointOfTri, 0, anInterpolator);
			aVInit(aInitVecInd.at(7)) = aRadiometryScalingInitialValue;
			aVInit(aInitVecInd.at(8)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepX,
																			 aThirdInitPointOfTri, 0, anInterpolator);
			aVInit(aInitVecInd.at(9)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepY,
																			 aThirdInitPointOfTri, 0, anInterpolator);
			aVInit(aInitVecInd.at(11)) = aRadiometryScalingInitialValue;
		}

		aSys = new tSys(eModeSSR::eSSR_LsqDense, aVInit);
	}

	void InitialiseWithPreviousExecutionValuesSerialisation(const cTriangulation2D<tREAL8> &aDelTri,
															tSys *&aSys,
															const std::string &aMultipleNodesFilename)
	{
		const int aNbUnkPerNode = 4;
		tDenseVect aVInit(aNbUnkPerNode * aDelTri.NbPts(), eModeInitImage::eMIA_Null);

		std::unique_ptr<const cMultipleTriangleNodesSerialiser> aReReadNodeFile = cMultipleTriangleNodesSerialiser::ReadVectorOfTriangleNodes(aMultipleNodesFilename);

		const std::vector<cNodeOfTriangles> aReReadVectorOfNodes = aReReadNodeFile->GetVectorOfNodes();

		for (size_t aNodeNumber = 0; aNodeNumber < aReReadVectorOfNodes.size(); aNodeNumber += 1)
		{
			const cNodeOfTriangles aInitPointOfTri = aReReadVectorOfNodes.at(aNodeNumber);

			aVInit(aNbUnkPerNode * aInitPointOfTri.GetTriangleFace()) = aInitPointOfTri.GetCurrentXYDisplacementValues().x();
			aVInit(aNbUnkPerNode * aInitPointOfTri.GetTriangleFace() + 1) = aInitPointOfTri.GetCurrentXYDisplacementValues().y();
			aVInit(aNbUnkPerNode * aInitPointOfTri.GetTriangleFace() + 2) = aInitPointOfTri.GetCurrentRadiometryTranslation();
			aVInit(aNbUnkPerNode * aInitPointOfTri.GetTriangleFace() + 3) = aInitPointOfTri.GetCurrentRadiometryScaling();
		}

		aSys = new tSys(eModeSSR::eSSR_LsqDense, aVInit);
	}

	void InitialiseInterpolationAndEquationTranslation(cCalculator<tREAL8> *&aEqTranslationTri, cDiffInterpolator1D *&aInterpolTr,
													   const std::vector<std::string> &aArgsVectorInterpolTr, const bool aUseLinearGradInterpolation)
	{
		if (aUseLinearGradInterpolation)
			aInterpolTr = cDiffInterpolator1D::AllocFromNames(aArgsVectorInterpolTr);

		// True means with derivative, 1 is size of buffer
		aEqTranslationTri = aUseLinearGradInterpolation ? EqDeformTriTranslationLinearGrad(true, 1) : EqDeformTriTranslationBilin(true, 1);
	}

	void InitialiseWithUserValuesTranslation(const cTriangulation2D<tREAL8> &aDelaunayTri,
											 tSys *&aSysTranslation,
											 const bool aUserInitialisation,
											 const tREAL8 aXTranslationInitVal,
											 const tREAL8 aYTranslationInitVal)
	{
		const int aNbUnkPerNode = 2;
		const size_t aNumberPts = aNbUnkPerNode * aDelaunayTri.NbPts();
		tDenseVect aVInitTranslation(aNumberPts, eModeInitImage::eMIA_Null);

		if (aUserInitialisation && aXTranslationInitVal != 0 && aYTranslationInitVal != 0)
		{
			for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
			{
				if (aKtNumber % aNbUnkPerNode == 0 && aXTranslationInitVal != 0)
					aVInitTranslation(aKtNumber) = aXTranslationInitVal;
				if (aKtNumber % aNbUnkPerNode == 1 && aYTranslationInitVal != 0)
					aVInitTranslation(aKtNumber) = aYTranslationInitVal;
			}
		}

		aSysTranslation = new tSys(eModeSSR::eSSR_LsqDense, aVInitTranslation);
	}

	void InitialiseWithPreviousExecutionValuesTranslationMMVI(const cTriangulation2D<tREAL8> &aDelTri,
															  tSys *&aSysTranslation,
															  cDiffInterpolator1D *&anInterpolator,
															  const std::string &aNameDepXFile, tIm &aImDepX,
															  tDIm *&aDImDepX, tPt2di &aSzImDepX,
															  const std::string &aNameDepYFile, tIm &aImDepY,
															  tDIm *&aDImDepY, tPt2di &aSzImDepY,
															  const std::string &aNameCorrelationMask,
															  tIm &aImCorrelationMask, tDIm *&aDImCorrelationMask,
															  tPt2di &aSzCorrelationMask)
	{
		tDenseVect aVInitTranslation(2 * aDelTri.NbPts(), eModeInitImage::eMIA_Null);

		ReadImageFileNameLoadData(aNameDepXFile, aImDepX,
								  aDImDepX, aSzImDepX);
		ReadImageFileNameLoadData(aNameDepYFile, aImDepY,
								  aDImDepY, aSzImDepY);

		ReadImageFileNameLoadData(aNameCorrelationMask, aImCorrelationMask,
								  aDImCorrelationMask, aSzCorrelationMask);

		for (size_t aTr = 0; aTr < aDelTri.NbFace(); aTr++)
		{
			const tTri2dr aInitTriTr = aDelTri.KthTri(aTr);
			const tPt3di aInitIndicesOfTriKnots = aDelTri.KthFace(aTr);

			//----------- Index of unknown, finds the associated pixels of current triangle
			tIntVect aInitVecInd;
			GetIndicesVector(aInitVecInd, aInitIndicesOfTriKnots, 2);

			// Get nodes associated to triangle
			const cNodeOfTriangles aFirstInitPointOfTri = cNodeOfTriangles(aVInitTranslation, aInitVecInd,
																		   0, 1, 0, 1, aInitTriTr, 0);
			const cNodeOfTriangles aSecondInitPointOfTri = cNodeOfTriangles(aVInitTranslation, aInitVecInd,
																			2, 3, 2, 3, aInitTriTr, 1);
			const cNodeOfTriangles aThirdInitPointOfTri = cNodeOfTriangles(aVInitTranslation, aInitVecInd,
																		   4, 5, 4, 5, aInitTriTr, 2);

			aVInitTranslation(aInitVecInd.at(0)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepX,
																						aFirstInitPointOfTri, 0, anInterpolator);
			aVInitTranslation(aInitVecInd.at(1)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepY,
																						aFirstInitPointOfTri, 0, anInterpolator);
			aVInitTranslation(aInitVecInd.at(2)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepX,
																						aSecondInitPointOfTri, 0, anInterpolator);
			aVInitTranslation(aInitVecInd.at(3)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepY,
																						aSecondInitPointOfTri, 0, anInterpolator);
			aVInitTranslation(aInitVecInd.at(4)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepX,
																						aThirdInitPointOfTri, 0, anInterpolator);
			aVInitTranslation(aInitVecInd.at(5)) = ReturnCorrectInitialisationValueMMVI(aDImCorrelationMask, aDImDepY,
																						aThirdInitPointOfTri, 0, anInterpolator);
		}

		aSysTranslation = new tSys(eModeSSR::eSSR_LsqDense, aVInitTranslation);
	}

	void InitialiseInterpolationAndEquationRadiometry(cCalculator<tREAL8> *&anEqRadiometryTri, cDiffInterpolator1D *&anInterpolRad,
													  const std::vector<std::string> &anArgsVectorInterpolRad, const bool aUseLinearGradInterpolation)
	{
		if (aUseLinearGradInterpolation)
			anInterpolRad = cDiffInterpolator1D::AllocFromNames(anArgsVectorInterpolRad);
		// True means with derivative, 1 is size of buffer
		anEqRadiometryTri = aUseLinearGradInterpolation ? EqDeformTriRadiometryLinearGrad(true, 1) : EqDeformTriRadiometryBilin(true, 1);
	}

	void InitialiseWithUserValuesRadiometry(const cTriangulation2D<tREAL8> &aDelaunayTri,
											tSys *&aSysRadiometry,
											const bool aUserInitialisation,
											const tREAL8 aRadTranslationInitVal,
											const tREAL8 aRadScaleInitVal)
	{
		const int aNbUnkPerNodeRad = 2;
		const size_t aNumberPts = aNbUnkPerNodeRad * aDelaunayTri.NbPts();
		tDenseVect aVInitRadiometry(aNumberPts, eModeInitImage::eMIA_Null);

		if (aUserInitialisation && aRadTranslationInitVal != 0 && aRadScaleInitVal != 1)
		{
			for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
			{
				if (aKtNumber % aNbUnkPerNodeRad == 0 && aRadTranslationInitVal != 0)
					aVInitRadiometry(aKtNumber) = aRadTranslationInitVal;
				if (aKtNumber % aNbUnkPerNodeRad == 1)
					aVInitRadiometry(aKtNumber) = aRadScaleInitVal;
			}
		}
		else
		{
			for (size_t aKtNumber = 0; aKtNumber < aNumberPts; aKtNumber++)
			{
				if (aKtNumber % aNbUnkPerNodeRad == 1)
					aVInitRadiometry(aKtNumber) = 1;
			}
		}

		aSysRadiometry = new tSys(eModeSSR::eSSR_LsqDense, aVInitRadiometry);
	}

	std::unique_ptr<const cNodeOfTriangles> DefineNewTriangleNode(const tDenseVect &aVecSol, const tIntVect &aIndVec, const int aXInd,
																  const int aYInd, const int aRadTrInd, const int aRadScInd, const tTri2dr &aTriangle,
																  const tPt3di &aFace, const int aPointNumberInTri, const int anIdOfPoint)
	{
		std::unique_ptr<const cNodeOfTriangles> aNewTriangleNode = std::make_unique<const cNodeOfTriangles>(aVecSol, aIndVec, aXInd, aYInd, aRadTrInd, aRadScInd, aTriangle, aFace,
																											aPointNumberInTri, anIdOfPoint);
		return aNewTriangleNode;
	}

	bool CheckValidCorrelationValue(tDIm *&aMask, const tPt2dr &aCoordNode,
									cDiffInterpolator1D *&anInterpolator)
	{
		bool aValidCorrelPoint;
		if (aMask->InsideInterpolator(*anInterpolator, aCoordNode, 0))
			aValidCorrelPoint = (aMask->GetValueInterpol(*anInterpolator, aCoordNode) == 255) ? true : false;
		else
			aValidCorrelPoint = false;
		return aValidCorrelPoint;
	}

	tREAL8 ReturnCorrectInitialisationValueMMVI(tDIm *&aMask, tDIm *&aDispMap, const cNodeOfTriangles &aPtOfTri,
												const tREAL8 aValueToReturnIfFalse, cDiffInterpolator1D *&anInterpolator)
	{
		const tPt2dr aCoordNode = aPtOfTri.GetInitialNodeCoordinates();
		// Check if correlation is computed for the point
		const bool aPointIsValid = CheckValidCorrelationValue(aMask, aCoordNode, anInterpolator);
		tREAL8 anInitialisationValue;
		if (aDispMap->InsideInterpolator(*anInterpolator, aCoordNode, 0))
			anInitialisationValue = (aPointIsValid) ? aDispMap->GetValueInterpol(*anInterpolator, aCoordNode) : aValueToReturnIfFalse;
		else
			anInitialisationValue = aValueToReturnIfFalse;
		return anInitialisationValue;
	}

	bool CheckFolderExistence(const std::string &aUserDefinedFolderNameToSaveResult)
	{
		bool aNonEmptyFolderName = false;
		if (!aUserDefinedFolderNameToSaveResult.empty())
		{
			aNonEmptyFolderName = true;
			if (!ExistFile(aUserDefinedFolderNameToSaveResult))
				CreateDirectories(aUserDefinedFolderNameToSaveResult, aNonEmptyFolderName);
		}
		return aNonEmptyFolderName;
	}

	int GetTotalNumberOfIterations(const bool aUseOfMultiScaleApproach, const int aNumberOfIterations,
								   const int aNumberOfEndIterations)
	{
		const int aTotalNumberOfIterations = (aUseOfMultiScaleApproach) ? aNumberOfIterations + aNumberOfEndIterations : aNumberOfIterations;
		return aTotalNumberOfIterations;
	}

	void GetIndicesVector(tIntVect &aVecInd, const tPt3di &aIndicesOfTriKnots, const int aIsTwoOrFour)
	{
		const int aCaseWithTwoUnk = 2;
		const int aCaseWithFourUnk = 4;
		(aIsTwoOrFour == aCaseWithTwoUnk) ? aVecInd = {aCaseWithTwoUnk * aIndicesOfTriKnots.x(), aCaseWithTwoUnk * aIndicesOfTriKnots.x() + 1,
													   aCaseWithTwoUnk * aIndicesOfTriKnots.y(), aCaseWithTwoUnk * aIndicesOfTriKnots.y() + 1,
													   aCaseWithTwoUnk * aIndicesOfTriKnots.z(), aCaseWithTwoUnk * aIndicesOfTriKnots.z() + 1}
										  : aVecInd = {aCaseWithFourUnk * aIndicesOfTriKnots.x(), aCaseWithFourUnk * aIndicesOfTriKnots.x() + 1,
													   aCaseWithFourUnk * aIndicesOfTriKnots.x() + 2, aCaseWithFourUnk * aIndicesOfTriKnots.x() + 3,
													   aCaseWithFourUnk * aIndicesOfTriKnots.y(), aCaseWithFourUnk * aIndicesOfTriKnots.y() + 1,
													   aCaseWithFourUnk * aIndicesOfTriKnots.y() + 2, aCaseWithFourUnk * aIndicesOfTriKnots.y() + 3,
													   aCaseWithFourUnk * aIndicesOfTriKnots.z(), aCaseWithFourUnk * aIndicesOfTriKnots.z() + 1,
													   aCaseWithFourUnk * aIndicesOfTriKnots.z() + 2, aCaseWithFourUnk * aIndicesOfTriKnots.z() + 3};
	}

	void SubtractPrePostImageAndComputeAvgAndMax(tIm &aImDiff, tDIm *&aDImDiff, tDIm *&aDImPre,
												 tDIm *&aDImPost, const tPt2di &aSzImPre)
	{
		aImDiff = tIm(aSzImPre);
		aDImDiff = &aImDiff.DIm();

		for (const tPt2di &aDiffPix : *aDImDiff)
			aDImDiff->SetV(aDiffPix, aDImPre->GetV(aDiffPix) - aDImPost->GetV(aDiffPix));
		const int aNumberOfPixelsInImage = aSzImPre.x() * aSzImPre.y();

		tREAL4 aSumPixelValuesInDiffImage = 0;
		tREAL4 aMaxPixelValuesInDiffImage = INT_MIN;
		tREAL4 aDiffImPixelValue = 0;

		for (const tPt2di &aDiffPix : *aDImDiff)
		{
			aDiffImPixelValue = aDImDiff->GetV(aDiffPix);
			aSumPixelValuesInDiffImage += aDiffImPixelValue;
			if (aDiffImPixelValue > aMaxPixelValuesInDiffImage)
				aMaxPixelValuesInDiffImage = aDiffImPixelValue;
		}
		StdOut() << "The average value of the difference image between the Pre and Post images is : "
				 << aSumPixelValuesInDiffImage / (tREAL8)aNumberOfPixelsInImage << std::endl;
		StdOut() << "The maximum value of the difference image between the Pre and Post images is : "
				 << aMaxPixelValuesInDiffImage << std::endl;
	}

	void ReadImageFileNameLoadData(const std::string &aImageFilename, tIm &aImage,
								   tDIm *&aDataImage, tPt2di &aSzIm)
	{
		aImage = tIm::FromFile(aImageFilename);

		aDataImage = &aImage.DIm();
		aSzIm = aDataImage->Sz();
	}

	void LoadPrePostImageAndData(tIm &aCurIm, tDIm *&aCurDIm, const std::string &aPreOrPostImage,
								 const tIm &aImPre, const tIm &aImPost)
	{
		(aPreOrPostImage == "pre") ? aCurIm = aImPre : aCurIm = aImPost;
		aCurDIm = &aCurIm.DIm();
	}

	void InitialiseDisplacementMapsAndOutputImage(const tPt2di &aSzImIn, tIm &aImOut,
												  tDIm *&aDImOut, tPt2di &aSzImOut)
	{
		aImOut = tIm(aSzImIn, 0, eModeInitImage::eMIA_Null);
		aDImOut = &aImOut.DIm();
		aSzImOut = aDImOut->Sz();
	}

	bool CheckIfConstraintsAreAppliedInFirstItersOfCurOptimsation(const int aNumberOfConstrainedFirstItersUnk)
	{
		const bool aApplicationOfConstraintsForFirstIters = (aNumberOfConstrainedFirstItersUnk > 0) ? true : false;
		return aApplicationOfConstraintsForFirstIters;
	}

	bool CheckIfCurIterIsFirstIterWithConstraint(const int aCurIterNumber, const int aNumberOfIterToFreeze)
	{
		const bool aCurFirstIterWithConstraintApplication = (aCurIterNumber < aNumberOfIterToFreeze) ? true : false;
		return aCurFirstIterWithConstraintApplication;
	}

	bool CheckIfCurIterIsOverMaxNumberOfConstrainedIterations(const int aCurIterNumber,
															  const int aNumberOfFirstIterWhereConstraintsAreAppliedTranslation,
															  const int aNumberOfFirstIterWhereConstraintsAreAppliedRadiometry)
	{
		const int aMaxNumberOfFirstIterToConstrain = std::max(aNumberOfFirstIterWhereConstraintsAreAppliedTranslation,
															  aNumberOfFirstIterWhereConstraintsAreAppliedRadiometry);
		const bool aIterWhereUnknownsAreConstrained = (aCurIterNumber < aMaxNumberOfFirstIterToConstrain) ? true : false;
		return aIterWhereUnknownsAreConstrained;
	}

	bool CheckIfHardConstraintsAreAppliedInCurrentIteration(const bool aHardFreezeForFirstItersUnk, const bool aCurFirstIterWhereHardFreezingUnkIsApplied,
															const bool aCurFirstIterWhereHardFreezingOtherUnkIsApplied, const bool aHardFreezeUnkAfterFirstIters,
															const bool aHardFreezeFirstUnk, const bool aHardFreezeSecondUnk, const bool aIterWhereUnknownsAreHardFrozen)
	{
		const bool aCurIterWithHardFrozenUnknowns = ((aHardFreezeForFirstItersUnk && aCurFirstIterWhereHardFreezingUnkIsApplied) ||
													 (!aCurFirstIterWhereHardFreezingOtherUnkIsApplied && aHardFreezeUnkAfterFirstIters) ||
													 (!aCurFirstIterWhereHardFreezingUnkIsApplied && !aHardFreezeUnkAfterFirstIters &&
													  aHardFreezeFirstUnk && aHardFreezeSecondUnk && !aIterWhereUnknownsAreHardFrozen && !aHardFreezeForFirstItersUnk))
														? true
														: false;
		return aCurIterWithHardFrozenUnknowns;
	}

	bool CheckIfSoftConstraintsAreAppliedInCurrentIteration(const bool aSoftFreezeForFirstItersUnk, const bool aSoftFreezeForFirstItersOtherUnk,
															const bool aCurFirstIterWhereSoftFreezingUnkIsApplied,
															const bool aCurFirstIterWhereHardFreezingUnkIsApplied, const bool aSoftFreezeUnkAfterFirstIters,
															const bool aIterWhereUnknownsAreSoftFrozen, const bool aCurIterWithHardFrozenUnk,
															const bool aCurIterWithFreedUnk, const bool aCurIterWithHardFrozenOtherUnk,
															const bool aCurFirstIterWhereHardFreezingOtherUnkIsApplied,
															const bool aHardFreezeForFirstItersUnk, const bool aHardFreezeForFirstItersOtherUnk)
	{
		// Case where soft constaints want to be applied for a certain amount of first iterations
		const bool aCurIterWithSoftFrozenUnknowns = ((aSoftFreezeForFirstItersUnk && aCurFirstIterWhereSoftFreezingUnkIsApplied) ||
													 // Case where soft constraints are always applied in optimisation process
													 (!aCurFirstIterWhereHardFreezingUnkIsApplied && !aSoftFreezeUnkAfterFirstIters &&
													  !aSoftFreezeForFirstItersUnk && !aSoftFreezeForFirstItersOtherUnk &&
													  !aIterWhereUnknownsAreSoftFrozen && !aCurFirstIterWhereSoftFreezingUnkIsApplied &&
													  !aCurIterWithHardFrozenUnk && !aCurIterWithFreedUnk && !aCurIterWithHardFrozenOtherUnk &&
													  !aHardFreezeForFirstItersUnk && !aHardFreezeForFirstItersOtherUnk) ||
													 // Case where soft constraints are applied after freeing other unknown
													 (!aCurFirstIterWhereHardFreezingUnkIsApplied && aSoftFreezeUnkAfterFirstIters &&
													  !aCurIterWithFreedUnk && !aCurFirstIterWhereSoftFreezingUnkIsApplied && !aCurFirstIterWhereHardFreezingOtherUnkIsApplied) ||
													 // Case where hard constraints have been freed and user now wants to apply soft constraints
													 (aCurIterWithFreedUnk && aSoftFreezeUnkAfterFirstIters &&
													  !aCurFirstIterWhereHardFreezingUnkIsApplied && !aIterWhereUnknownsAreSoftFrozen))
														? true
														: false;
		return aCurIterWithSoftFrozenUnknowns;
	}

	bool CheckIfUnknownsAreFreedInCurrentIteration(const bool aCurFirstIterWhereHardFreezingUnkIsApplied, const bool aHardFreezeForFirstItersUnk,
												   const bool aCurIterWithHardFrozenUnk, const bool aIterWhereUnknownsAreHardFrozen, const bool aSoftFreezeUnkAfterFirstIters,
												   const bool aHardFreezeUnkAfterFirstIters, const bool aSoftFreezeForFirstItersUnk, const bool aIterWhereUnknownsAreSoftFrozen,
												   const bool aCurFirstIterWhereSoftFreezingUnkIsApplied)
	{
		const bool aCurIterWithFreedUnknowns = ((!aCurFirstIterWhereHardFreezingUnkIsApplied && aHardFreezeForFirstItersUnk) ||
												(!aCurIterWithHardFrozenUnk && !aCurFirstIterWhereHardFreezingUnkIsApplied && !aIterWhereUnknownsAreHardFrozen &&
												 aHardFreezeForFirstItersUnk && aSoftFreezeUnkAfterFirstIters) ||
												(!aCurFirstIterWhereHardFreezingUnkIsApplied && !aSoftFreezeUnkAfterFirstIters && !aHardFreezeUnkAfterFirstIters &&
												 !aSoftFreezeForFirstItersUnk && !aIterWhereUnknownsAreSoftFrozen && !aCurFirstIterWhereSoftFreezingUnkIsApplied &&
												 !aCurIterWithHardFrozenUnk && aHardFreezeForFirstItersUnk))
												   ? true
												   : false;
		return aCurIterWithFreedUnknowns;
	}

	tPt2dr LoadNodeAndReturnCurrentDisplacement(const tDenseVect &aVCurSol, const tIntVect &aVecInd,
												const int aXDispInd, const int aYDispInd, const int aRadTrInd,
												const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri)
	{
		const cNodeOfTriangles aTriNode = cNodeOfTriangles(aVCurSol, aVecInd, aXDispInd, aYDispInd,
														   aRadTrInd, aRadScInd, aTri, aPtInNumberTri);
		return aTriNode.GetCurrentXYDisplacementValues(); // Current translation of node
	}

	tREAL8 LoadNodeAndReturnCurrentRadiometryTranslation(const tDenseVect &aVCurSol, const tIntVect &aVecInd,
														 const int aXDispInd, const int aYDispInd, const int aRadTrInd,
														 const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri)
	{
		const cNodeOfTriangles aTriNode = cNodeOfTriangles(aVCurSol, aVecInd, aXDispInd, aYDispInd,
														   aRadTrInd, aRadScInd, aTri, aPtInNumberTri);
		return aTriNode.GetCurrentRadiometryTranslation(); // Current radiometry translation of node
	}

	tREAL8 LoadNodeAndReturnCurrentRadiometryScaling(const tDenseVect &aVCurSol, const tIntVect &aVecInd,
													 const int aXDispInd, const int aYDispInd, const int aRadTrInd,
													 const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri)
	{
		const cNodeOfTriangles aTriNode = cNodeOfTriangles(aVCurSol, aVecInd, aXDispInd, aYDispInd,
														   aRadTrInd, aRadScInd, aTri, aPtInNumberTri);
		return aTriNode.GetCurrentRadiometryScaling(); // Current radiometry scaling of node
	}

	tPt2dr LoadNodeAppendVectorAndReturnCurrentDisplacement(const tDenseVect &aVCurSol, const tIntVect &aVecInd,
															const int aXDispInd, const int aYDispInd, const int aRadTrInd,
															const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
															const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
															const std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes)
	{
		const std::unique_ptr<const cNodeOfTriangles> aNodeOfTri = DefineNewTriangleNode(aVCurSol, aVecInd, aXDispInd, aYDispInd, aRadTrInd, aRadScInd,
																						 aTri, aFace, aPtInNumberTri, aNodeCounter);
		if (anAppend)
			aVectorOfTriangleNodes->PushInVector(aNodeOfTri);
		return aNodeOfTri->GetCurrentXYDisplacementValues(); // Current translation of node
	}

	tREAL8 LoadNodeAppendVectorAndReturnCurrentRadiometryTranslation(const tDenseVect &aVCurSol, const tIntVect &aVecInd,
																	 const int aXDispInd, const int aYDispInd, const int aRadTrInd,
																	 const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
																	 const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
																	 const std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes)
	{
		const std::unique_ptr<const cNodeOfTriangles> aNodeOfTri = DefineNewTriangleNode(aVCurSol, aVecInd, aXDispInd, aYDispInd, aRadTrInd, aRadScInd,
																						 aTri, aFace, aPtInNumberTri, aNodeCounter);
		if (anAppend)
			aVectorOfTriangleNodes->PushInVector(aNodeOfTri);
		return aNodeOfTri->GetCurrentRadiometryTranslation(); // Current radiometry translation of node
	}

	tREAL8 LoadNodeAppendVectorAndReturnCurrentRadiometryScaling(const tDenseVect &aVCurSol, const tIntVect &aVecInd,
																 const int aXDispInd, const int aYDispInd, const int aRadTrInd,
																 const int aRadScInd, const tTri2dr &aTri, const int aPtInNumberTri,
																 const int aNodeCounter, const tPt3di &aFace, const bool anAppend,
																 const std::unique_ptr<cMultipleTriangleNodesSerialiser> &aVectorOfTriangleNodes)
	{
		const std::unique_ptr<const cNodeOfTriangles> aNodeOfTri = DefineNewTriangleNode(aVCurSol, aVecInd, aXDispInd, aYDispInd, aRadTrInd, aRadScInd,
																						 aTri, aFace, aPtInNumberTri, aNodeCounter);
		if (anAppend)
			aVectorOfTriangleNodes->PushInVector(aNodeOfTri);
		return aNodeOfTri->GetCurrentRadiometryScaling(); // Current radiometry scaling of node
	}

	bool ManageDifferentCasesOfEndIterations(const int aIterNumber, const int aNumberOfScales, const int aNumberOfEndIterations,
											 bool aIsLastIters, const tIm &aImPre, const tIm &aImPost, tIm &aCurPreIm, tDIm *&aCurPreDIm,
											 tIm &aCurPostIm, tDIm *&aCurPostDIm)
	{
		switch (aNumberOfEndIterations)
		{
		case 1: // One last iteration
			if (aIterNumber == aNumberOfScales)
			{
				aIsLastIters = true;
				LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
				LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
			}
			break;
		case 2: // Two last iterations
			if ((aIterNumber == aNumberOfScales) || (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 1))
			{
				aIsLastIters = true;
				LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
				LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
			}
			break;
		case 3: //  Three last iterations
			if ((aIterNumber == aNumberOfScales) || (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 2) ||
				(aIterNumber == aNumberOfScales + aNumberOfEndIterations - 1))
			{
				aIsLastIters = true;
				LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
				LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
			}
			break;
		default: // Default is two last iterations
			if ((aIterNumber == aNumberOfScales) || (aIterNumber == aNumberOfScales + aNumberOfEndIterations - 1))
			{
				aIsLastIters = true;
				LoadPrePostImageAndData(aCurPreIm, aCurPreDIm, "pre", aImPre, aImPost);
				LoadPrePostImageAndData(aCurPostIm, aCurPostDIm, "post", aImPre, aImPost);
			}
			break;
		}
		return aIsLastIters;
	}

	void ApplyHardConstraintsToMultipleUnknowns(const int aSolStart, const int aSolStep, const tIntVect &aVecInd,
												const tDenseVect &aVCurSol, tSys *&aSys)
	{
		for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size(); aIndCurSol += aSolStep)
		{
			const int aIndicesToFreeze = aVecInd.at(aIndCurSol);
			aSys->SetFrozenVar(aIndicesToFreeze, aVCurSol(aIndicesToFreeze));
		}
	}

	void UnfreezeUnknown(const int anIndices, const tIntVect &aVecInd,
						 tSys *&aSys)
	{
		const int anIndicesToUnfreeze = aVecInd.at(anIndices);
		aSys->SetUnFrozen(anIndicesToUnfreeze);
	}

	void UnfreezeMultipleUnknowns(const int aSolStart, const int aSolStep,
								  const tIntVect &aVecInd, tSys *&aSys)
	{
		for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size(); aIndCurSol += aSolStep)
		{
			const int anIndicesToUnfreeze = aVecInd.at(aIndCurSol);
			aSys->SetUnFrozen(anIndicesToUnfreeze);
		}
	}

	void FreezeOrUnfreezeUnknown(const bool aHardConstrainApplication, const bool aCurIterWithFreedUnk,
								 const int aSolStart, const int aSolStep, const tIntVect &aVecInd,
								 const tDenseVect &aVCurSol, tSys *&aSys)
	{
		if (aHardConstrainApplication)
			ApplyHardConstraintsToMultipleUnknowns(aSolStart, aSolStep, aVecInd, aVCurSol, aSys);
		if (aCurIterWithFreedUnk)
			UnfreezeMultipleUnknowns(aSolStart, aSolStep, aVecInd, aSys);
	}

	void ApplySoftConstraintToMultipleUnknown(const int aSolStart, const int aSolStep, const tIntVect &aVecInd, tSys *&aSys,
											  const tDenseVect &aVCurSol, const tREAL8 aWeight)
	{
		for (size_t aIndCurSol = aSolStart; aIndCurSol < aVecInd.size(); aIndCurSol += aSolStep)
		{
			const int aIndices = aVecInd.at(aIndCurSol);
			aSys->AddEqFixVar(aIndices, aVCurSol(aIndices), aWeight);
		}
	}

	void ApplySoftConstraintWithCondition(const tREAL8 aWeight, const bool aCurIterWithSoftConstraintApplication,
										  const int aSolStart, const int aSolStep, const tIntVect &aVecInd,
										  const tDenseVect &aVCurSol, tSys *&aSys)
	{
		const bool aSoftApplication = (aWeight > 0 && aCurIterWithSoftConstraintApplication);
		if (aSoftApplication)
			ApplySoftConstraintToMultipleUnknown(aSolStart, aSolStep, aVecInd, aSys, aVCurSol, aWeight);
	}

	tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aCurrentTranslationPointA,
														  const tPt2dr &aCurrentTranslationPointB,
														  const tPt2dr &aCurrentTranslationPointC,
														  const tDoubleVect &aVObs)
	{
		// Apply current barycenter translation formula for x and y on current observations.
		const tREAL8 aXTriCoord = aVObs[0] + aVObs[2] * aCurrentTranslationPointA.x() + aVObs[3] * aCurrentTranslationPointB.x() +
								  aVObs[4] * aCurrentTranslationPointC.x();
		const tREAL8 aYTriCoord = aVObs[1] + aVObs[2] * aCurrentTranslationPointA.y() + aVObs[3] * aCurrentTranslationPointB.y() +
								  aVObs[4] * aCurrentTranslationPointC.y();

		const tPt2dr aCurrentTranslatedPixel = tPt2dr(aXTriCoord, aYTriCoord);

		return aCurrentTranslatedPixel;
	}

	tPt2dr ApplyBarycenterTranslationFormulaToFilledPixel(const tPt2dr &aCurrentTranslationPointA,
														  const tPt2dr &aCurrentTranslationPointB,
														  const tPt2dr &aCurrentTranslationPointC,
														  const cPtInsideTriangles &aPixInsideTriangle)
	{
		const tPt2dr aCartesianCoordinates = aPixInsideTriangle.GetCartesianCoordinates();
		const cPt3dr aBarycenterCoordinates = aPixInsideTriangle.GetBarycenterCoordinates();
		// Apply current barycenter translation formula for x and y on current observations.
		const tREAL8 aXTriCoord = aCartesianCoordinates.x() + aBarycenterCoordinates.x() * aCurrentTranslationPointA.x() +
								  aBarycenterCoordinates.y() * aCurrentTranslationPointB.x() + aBarycenterCoordinates.z() * aCurrentTranslationPointC.x();
		const tREAL8 aYTriCoord = aCartesianCoordinates.y() + aBarycenterCoordinates.x() * aCurrentTranslationPointA.y() +
								  aBarycenterCoordinates.y() * aCurrentTranslationPointB.y() + aBarycenterCoordinates.z() * aCurrentTranslationPointC.y();

		const tPt2dr aTranslatedPixel = tPt2dr(aXTriCoord, aYTriCoord);

		return aTranslatedPixel;
	}

	tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
																	 const tREAL8 aCurrentRadTranslationPointB,
																	 const tREAL8 aCurrentRadTranslationPointC,
																	 const tDoubleVect &aVObs)
	{
		const tREAL8 aCurentRadTranslation = aVObs[2] * aCurrentRadTranslationPointA + aVObs[3] * aCurrentRadTranslationPointB +
											 aVObs[4] * aCurrentRadTranslationPointC;
		return aCurentRadTranslation;
	}

	tREAL8 ApplyBarycenterTranslationFormulaForTranslationRadiometry(const tREAL8 aCurrentRadTranslationPointA,
																	 const tREAL8 aCurrentRadTranslationPointB,
																	 const tREAL8 aCurrentRadTranslationPointC,
																	 const cPtInsideTriangles &aPixInsideTriangle)
	{
		const cPt3dr BarycenterCoordinateOfPoint = aPixInsideTriangle.GetBarycenterCoordinates();
		const tREAL8 aCurentRadTranslation = BarycenterCoordinateOfPoint.x() * aCurrentRadTranslationPointA + BarycenterCoordinateOfPoint.y() * aCurrentRadTranslationPointB +
											 BarycenterCoordinateOfPoint.z() * aCurrentRadTranslationPointC;
		return aCurentRadTranslation;
	}

	tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
																 const tREAL8 aCurrentRadScalingPointB,
																 const tREAL8 aCurrentRadScalingPointC,
																 const tDoubleVect &aVObs)
	{
		const tREAL8 aCurrentRadScaling = aVObs[2] * aCurrentRadScalingPointA + aVObs[3] * aCurrentRadScalingPointB +
										  aVObs[4] * aCurrentRadScalingPointC;
		return aCurrentRadScaling;
	}

	tREAL8 ApplyBarycenterTranslationFormulaForScalingRadiometry(const tREAL8 aCurrentRadScalingPointA,
																 const tREAL8 aCurrentRadScalingPointB,
																 const tREAL8 aCurrentRadScalingPointC,
																 const cPtInsideTriangles &aLastPixInsideTriangle)
	{
		const cPt3dr BarycenterCoordinateOfPoint = aLastPixInsideTriangle.GetBarycenterCoordinates();
		const tREAL8 aCurrentRadScaling = BarycenterCoordinateOfPoint.x() * aCurrentRadScalingPointA + BarycenterCoordinateOfPoint.y() * aCurrentRadScalingPointB +
										  BarycenterCoordinateOfPoint.z() * aCurrentRadScalingPointC;
		return aCurrentRadScaling;
	}

	tPt2di GetCoordinatesToSetValueForTranslatedImage(const tREAL8 aLastXTranslatedCoord, const tREAL8 aLastYTranslatedCoord,
													  const tPt2di &aLastCoordinate, const tPt2di &aSzImOut, tDIm *&aDImOut)
	{
		// Build image with intensities displaced
		// deal with different cases of pixel being translated out of image
		if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord < 0)
			return tPt2di(0, 0);
		else if (aLastXTranslatedCoord >= aSzImOut.x() && aLastYTranslatedCoord >= aSzImOut.y())
			return tPt2di(aSzImOut.x() - 1, aSzImOut.y() - 1);
		else if (aLastXTranslatedCoord < 0 && aLastYTranslatedCoord >= aSzImOut.y())
			return tPt2di(0, aSzImOut.y() - 1);
		else if (aLastXTranslatedCoord >= aSzImOut.x() && aLastYTranslatedCoord < 0)
			return tPt2di(aSzImOut.x() - 1, 0);
		else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < aSzImOut.x() &&
				 aLastYTranslatedCoord < 0)
			return tPt2di(aLastXTranslatedCoord, 0);
		else if (aLastXTranslatedCoord >= 0 && aLastXTranslatedCoord < aSzImOut.x() &&
				 aLastYTranslatedCoord > aSzImOut.y())
			return tPt2di(aLastXTranslatedCoord, aSzImOut.y() - 1);
		else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < aSzImOut.y() &&
				 aLastXTranslatedCoord < 0)
			return tPt2di(0, aLastYTranslatedCoord);
		else if (aLastYTranslatedCoord >= 0 && aLastYTranslatedCoord < aSzImOut.y() &&
				 aLastXTranslatedCoord > aSzImOut.x())
			return tPt2di(aSzImOut.x() - 1, aLastYTranslatedCoord);
		else
			// At the translated pixel the untranslated pixel value is given computed with the right radiometry values
			return tPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord);
	}

	tREAL8 GetInterpolatedcCoordinatesIfInsideInterpolator(const tREAL8 aLastCoordinate, const tPt2di &aLastIntCoordinate,
														   const tPt2dr &aLastRealCoordinate, tDIm *&aDImDepMap,
														   cDiffInterpolator1D *&anInterpolator)
	{
		const tREAL8 aLastTranslatedCoord = (aDImDepMap->InsideInterpolator(*anInterpolator, aLastRealCoordinate, 0)) ? aLastCoordinate + aDImDepMap->GetValueInterpol(*anInterpolator, aLastRealCoordinate) : aLastCoordinate + aDImDepMap->GetV(aLastIntCoordinate);
		return aLastTranslatedCoord;
	}

	void FillDisplacementMapsAndOutputImage(const cPtInsideTriangles &aLastPixInsideTriangle,
											const tPt2dr &aLastTranslatedFilledPoint,
											const tREAL8 aLastRadiometryTranslation,
											const tREAL8 aLastRadiometryScaling, const tPt2di &aSzImOut,
											tDIm *&aDImDepX, tDIm *&aDImDepY, tDIm *&aDImOut, tDIm *&aDImPost,
											cDiffInterpolator1D *&anInterpolator)
	{
		const tPt2dr aLastRealCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates();
		const tREAL8 aLastXCoordinate = aLastRealCoordinate.x();
		const tREAL8 aLastYCoordinate = aLastRealCoordinate.y();

		const tPt2di aLastIntCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

		aDImDepX->SetV(aLastIntCoordinate,
					   aLastTranslatedFilledPoint.x() - aLastXCoordinate);
		aDImDepY->SetV(aLastIntCoordinate,
					   aLastTranslatedFilledPoint.y() - aLastYCoordinate);

		const tREAL8 aLastXTranslatedCoord = GetInterpolatedcCoordinatesIfInsideInterpolator(aLastXCoordinate,
																							 aLastIntCoordinate,
																							 aLastRealCoordinate,
																							 aDImDepX, anInterpolator);
		const tREAL8 aLastYTranslatedCoord = GetInterpolatedcCoordinatesIfInsideInterpolator(aLastYCoordinate,
																							 aLastIntCoordinate,
																							 aLastRealCoordinate,
																							 aDImDepY, anInterpolator);

		const tPt2di aCorrectCoordToPlaceValueInImage = GetCoordinatesToSetValueForTranslatedImage(aLastXTranslatedCoord,
																								   aLastYTranslatedCoord,
																								   aLastIntCoordinate,
																								   aSzImOut, aDImOut);

		const tPt2dr aLastRealTranslatedPoint = tPt2dr(aLastXTranslatedCoord, aLastYTranslatedCoord);

		const tREAL8 aLastPixelValue = aDImPost->InsideInterpolator(*anInterpolator, aLastRealTranslatedPoint, 0) ? aDImPost->GetValueInterpol(*anInterpolator, aLastRealTranslatedPoint) : aDImPost->GetV(aCorrectCoordToPlaceValueInImage);

		const tREAL8 aLastRadiometryValue = (aLastPixelValue - aLastRadiometryTranslation) / aLastRadiometryScaling;

		aDImOut->SetV(aLastIntCoordinate, aLastRadiometryValue);
	}

	void FillDisplacementMapsTranslation(const cPtInsideTriangles &aLastPixInsideTriangle,
										 const tPt2dr &aLastTranslatedFilledPoint, const tPt2di &aSzImOut,
										 tDIm *&aDImDepX, tDIm *&aDImDepY, tDIm *&aDImOut,
										 cDiffInterpolator1D *&anInterpolator)
	{
		const tPt2dr aLastRealCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates();
		const tREAL8 aLastXCoordinate = aLastRealCoordinate.x();
		const tREAL8 aLastYCoordinate = aLastRealCoordinate.y();
		const tREAL8 aLastPixelValue = aLastPixInsideTriangle.GetPixelValue();

		const tPt2di aLastIntCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

		aDImDepX->SetV(aLastIntCoordinate,
					   aLastTranslatedFilledPoint.x() - aLastXCoordinate);
		aDImDepY->SetV(aLastIntCoordinate,
					   aLastTranslatedFilledPoint.y() - aLastYCoordinate);

		const tREAL8 aLastXTranslatedCoord = GetInterpolatedcCoordinatesIfInsideInterpolator(aLastXCoordinate,
																							 aLastIntCoordinate,
																							 aLastRealCoordinate,
																							 aDImDepX, anInterpolator);
		const tREAL8 aLastYTranslatedCoord = GetInterpolatedcCoordinatesIfInsideInterpolator(aLastYCoordinate,
																							 aLastIntCoordinate,
																							 aLastRealCoordinate,
																							 aDImDepY, anInterpolator);

		const tPt2di aCorrectCoordToPlaceValueInImage = GetCoordinatesToSetValueForTranslatedImage(aLastXTranslatedCoord,
																								   aLastYTranslatedCoord,
																								   aLastIntCoordinate,
																								   aSzImOut, aDImOut);

		(tPt2di(aCorrectCoordToPlaceValueInImage) == tPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord)) ? aDImOut->SetV(tPt2di(aLastXTranslatedCoord, aLastYTranslatedCoord), aLastPixelValue)
																										   : aDImOut->SetV(aLastIntCoordinate, aDImOut->GetV(aCorrectCoordToPlaceValueInImage));
	}

	void FillOutputImageRadiometry(const cPtInsideTriangles &aLastPixInsideTriangle,
								   const tREAL8 aLastRadiometryTranslation,
								   const tREAL8 aLastRadiometryScaling,
								   tDIm *&aDImOut)
	{
		const tREAL8 aLastXCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().x();
		const tREAL8 aLastYCoordinate = aLastPixInsideTriangle.GetCartesianCoordinates().y();

		const tPt2di aLastCoordinate = tPt2di(aLastXCoordinate, aLastYCoordinate);

		const tREAL8 aLastRadiometryValue = aLastRadiometryScaling * aLastPixInsideTriangle.GetPixelValue() +
											aLastRadiometryTranslation;

		// Build image with radiometric intensities
		aDImOut->SetV(aLastCoordinate, aLastRadiometryValue);
	}

	void FillDiffDisplacementMap(tDIm *&aDImDispMap, tDIm *&aDImDiffMap,
								 tDIm *&aDImTranslatedDispMap,
								 const cPtInsideTriangles &aPixInsideTriangle,
								 const tREAL8 aPixInsideTriangleCoord,
								 const tREAL8 aTranslatedDispMapPointCoord,
								 const bool aComputeDiffDispMaps, const bool aComputeInterpTranslationDispMaps)
	{
		const tPt2di aIntCoordinate = tPt2di(aPixInsideTriangle.GetCartesianCoordinates().x(), aPixInsideTriangle.GetCartesianCoordinates().y());
		const tREAL8 aDiffBarycentricInterpTranslation = aTranslatedDispMapPointCoord - aPixInsideTriangleCoord;
		if (aComputeDiffDispMaps)
			// Compute difference between value of pixel in ground-truth map and value of barycentric interpolated pixel from displacement map
			aDImDiffMap->SetV(aIntCoordinate, std::abs(aDiffBarycentricInterpTranslation - aPixInsideTriangle.GetPixelValue()));
		if (aComputeInterpTranslationDispMaps)
			// Sets value to difference between pixel's original position in displacement map and interpolated value by barycentric interpolation formula
			aDImTranslatedDispMap->SetV(aIntCoordinate, aDiffBarycentricInterpTranslation);
	}

	void SaveMultiScaleDisplacementMapsToFile(tDIm *&aDImDepX, tDIm *&aDImDepY, const bool aUserDefinedFolderName,
											  const std::string &aFolderPathToSave, const std::string &aDepXFileNameToSave,
											  const std::string &aDepYFileNameToSave, const int aIterNumber,
											  const int aNumberOfPointsToGenerate, const int aTotalNumberOfIterations)
	{
		if (aUserDefinedFolderName)
		{
			aDImDepX->ToFile(aFolderPathToSave + "/" + aDepXFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
			aDImDepY->ToFile(aFolderPathToSave + "/" + aDepYFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
		}
		else
		{
			aDImDepX->ToFile(aDepXFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
			aDImDepY->ToFile(aDepYFileNameToSave + "_" + ToStr(aIterNumber) + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
		}
	}

	void SaveFinalDisplacementMapsToFile(tDIm *&aDImDepX, tDIm *&aDImDepY, const bool aUserDefinedFolderName,
										 const std::string &aFolderPathToSave, const std::string &aDepXFileNameToSave,
										 const std::string &aDepYFileNameToSave, const int aNumberOfPointsToGenerate,
										 const int aTotalNumberOfIterations)
	{
		if (aUserDefinedFolderName)
		{
			aDImDepX->ToFile(aFolderPathToSave + "/" + aDepXFileNameToSave + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
			aDImDepY->ToFile(aFolderPathToSave + "/" + aDepYFileNameToSave + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
		}
		else
		{
			aDImDepX->ToFile(aDepXFileNameToSave + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
			aDImDepY->ToFile(aDepYFileNameToSave + "_" +
							 ToStr(aNumberOfPointsToGenerate) + "_" +
							 ToStr(aTotalNumberOfIterations) + ".tif");
		}
	}

	void SaveOutputImageToFile(tDIm *&aDImOut, const bool aUserDefinedFolderName, const std::string &aFolderPathToSave,
							   const std::string &aOutputImageFileNameToSave,
							   const int aNumberOfPointsToGenerate, const int aTotalNumberOfIterations)
	{
		(aUserDefinedFolderName) ? aDImOut->ToFile(aFolderPathToSave + "/" + aOutputImageFileNameToSave + ToStr(aNumberOfPointsToGenerate) + "_" +
												   ToStr(aTotalNumberOfIterations) + ".tif")
								 : aDImOut->ToFile(aOutputImageFileNameToSave + "_" + ToStr(aNumberOfPointsToGenerate) + "_" +
												   ToStr(aTotalNumberOfIterations) + ".tif");
	}

	tREAL8 CheckMinMaxValueIsChanged(const tREAL8 aMinMaxValue, tREAL8 aPossiblyChangedValue, const int aNbChanges)
	{
		if (aNbChanges > 0)
			return aPossiblyChangedValue;
		else
		{
			aPossiblyChangedValue = aMinMaxValue;
			return aPossiblyChangedValue;
		}
	}

	void DisplayFirstAndLastUnknownValuesAndComputeStatisticsTwoUnknowns(const tDenseVect &aVFinalSol, const tDenseVect &aVInitSol)
	{
		tREAL8 aMaxFirstUnk = INT_MAX, aMinFirstUnk = INT_MIN, aMeanFirstUnk = 0, aVarianceMeanFirstUnk = 0;	 // aVarFirstUnk = 0;
		tREAL8 aMaxSecondUnk = INT_MAX, aMinSecondUnk = INT_MIN, aMeanSecondUnk = 0, aVarianceMeanSecondUnk = 0; // aVarSecondUnk = 0;
		const tREAL8 aVecSolSz = aVFinalSol.DIm().Sz();
		const int aNbUnk = 2;
		const tREAL8 aDividedVecSolSz = aVecSolSz / aNbUnk;

		int aNbChangesMinFirstUnk = 0;
		int aNbChangesMaxFirstUnk = 0;
		int aNbChangesMinSecondUnk = 0;
		int aNbChangesMaxSecondUnk = 0;

		for (int aFinalUnk = 0; aFinalUnk < aVecSolSz; aFinalUnk++)
		{
			const tREAL8 aFinalSolValue = aVFinalSol(aFinalUnk);
			const tREAL8 aInitSolValue = aVInitSol(aFinalUnk);

			StdOut() << aFinalSolValue << " " << aInitSolValue << " ";
			if (aFinalUnk % aNbUnk == 1 && aFinalUnk != 0)
				StdOut() << std::endl;

			if (aFinalUnk % aNbUnk == 0)
			{
				aMeanFirstUnk += aFinalSolValue;
				if (aFinalSolValue > aMaxFirstUnk)
				{
					aMaxFirstUnk = aFinalSolValue;
					aNbChangesMaxFirstUnk++;
				}
				if (aFinalSolValue < aMinFirstUnk)
				{
					aMinFirstUnk = aFinalSolValue;
					aNbChangesMinFirstUnk++;
				}
			}
			else
			{
				aMeanSecondUnk += aFinalSolValue;
				if (aFinalSolValue > aMaxSecondUnk)
				{
					aMaxSecondUnk = aFinalSolValue;
					aNbChangesMaxSecondUnk++;
				}
				if (aFinalSolValue < aMinSecondUnk)
				{
					aMinSecondUnk = aFinalSolValue;
					aNbChangesMinSecondUnk++;
				}
			}
		}

		aMinFirstUnk = CheckMinMaxValueIsChanged(aMaxFirstUnk, aMinFirstUnk, aNbChangesMinFirstUnk);
		aMaxFirstUnk = CheckMinMaxValueIsChanged(aMinFirstUnk, aMaxFirstUnk, aNbChangesMaxFirstUnk);
		aMinSecondUnk = CheckMinMaxValueIsChanged(aMaxSecondUnk, aMinSecondUnk, aNbChangesMinSecondUnk);
		aMaxSecondUnk = CheckMinMaxValueIsChanged(aMinSecondUnk, aMaxSecondUnk, aNbChangesMaxSecondUnk);

		aMeanFirstUnk /= aDividedVecSolSz;
		aMeanSecondUnk /= aDividedVecSolSz;

		std::vector<tREAL8> aVarianceVectorFirstUnk;
		std::vector<tREAL8> aVarianceVectorSecondUnk;

		for (int aSolNumber = 0; aSolNumber < aVecSolSz; aSolNumber++)
		{
			const tREAL8 aFinalSolValue = aVFinalSol(aSolNumber);
			if (aSolNumber % aNbUnk == 0)
				aVarianceVectorFirstUnk.push_back(std::abs(aFinalSolValue - aMeanFirstUnk) * std::abs(aFinalSolValue - aMeanFirstUnk));
			else
				aVarianceVectorSecondUnk.push_back(std::abs(aFinalSolValue - aMeanSecondUnk) * std::abs(aFinalSolValue - aMeanSecondUnk));
		}

		for (int aSolNumber = 0; aSolNumber < aDividedVecSolSz; aSolNumber++)
		{
			if (aSolNumber % aNbUnk == 0)
				aVarianceMeanFirstUnk += aVarianceVectorFirstUnk[aSolNumber];
			else
				aVarianceMeanSecondUnk += aVarianceVectorSecondUnk[aSolNumber];
		}

		aVarianceMeanFirstUnk /= aDividedVecSolSz;
		aVarianceMeanSecondUnk /= aDividedVecSolSz;

		StdOut() << "The minimum value for the first unknown is : " << aMinFirstUnk << " and the maximum value is : "
				 << aMaxFirstUnk << std::endl;
		StdOut() << "The minimum  value for the second unknown is : " << aMinSecondUnk << " and the maximum value is : "
				 << aMaxSecondUnk << std::endl;
		StdOut() << "The mean value for the first unknown is : " << aMeanFirstUnk << " and the standard deviation value is : "
				 << std::sqrt(aVarianceMeanFirstUnk) << std::endl;
		StdOut() << "The mean value for the second unknown is : " << aMeanSecondUnk << " and the standard deviation value is : "
				 << std::sqrt(aVarianceMeanSecondUnk) << std::endl;
	}

	void ComputeStatisticsFourUnknowns(const tDenseVect &aVFinalSol)
	{
		tREAL8 aMinFirstUnk = INT_MAX, aMaxFirstUnk = INT_MIN, aMeanFirstUnk = 0, aVarianceMeanFirstUnk = 0;
		tREAL8 aMinSecondUnk = INT_MAX, aMaxSecondUnk = INT_MIN, aMeanSecondUnk = 0, aVarianceMeanSecondUnk = 0;
		tREAL8 aMinThirdUnk = INT_MAX, aMaxThirdUnk = INT_MIN, aMeanThirdUnk = 0, aVarianceMeanThirdUnk = 0;
		tREAL8 aMinFourthUnk = INT_MAX, aMaxFourthUnk = INT_MIN, aMeanFourthUnk = 0, aVarianceMeanFourthUnk = 0;

		const int aNbUnk = 4;

		const tREAL8 aVecSolSz = aVFinalSol.DIm().Sz();
		const tREAL8 aDividedVecSolSz = aVecSolSz / aNbUnk;

		int aNbChangesMinFirstUnk = 0;
		int aNbChangesMaxFirstUnk = 0;
		int aNbChangesMinSecondUnk = 0;
		int aNbChangesMaxSecondUnk = 0;
		int aNbChangesMinThirdUnk = 0;
		int aNbChangesMaxThirdUnk = 0;
		int aNbChangesMinFourthUnk = 0;
		int aNbChangesMaxFourthUnk = 0;

		for (int aFinalUnk = 0; aFinalUnk < aVecSolSz; aFinalUnk++)
		{
			const tREAL8 aFinalSolValue = aVFinalSol(aFinalUnk);

			if (aFinalUnk % aNbUnk == 0)
			{
				aMeanFirstUnk += aFinalSolValue;
				if (aFinalSolValue > aMaxFirstUnk)
				{
					aMaxFirstUnk = aFinalSolValue;
					aNbChangesMaxFirstUnk++;
				}
				if (aFinalSolValue < aMinFirstUnk)
				{
					aMinFirstUnk = aFinalSolValue;
					aNbChangesMinFirstUnk++;
				}
			}
			else if (aFinalUnk % aNbUnk == 1)
			{
				aMeanSecondUnk += aFinalSolValue;
				if (aFinalSolValue > aMaxSecondUnk)
				{
					aMaxSecondUnk = aFinalSolValue;
					aNbChangesMaxSecondUnk++;
				}
				if (aFinalSolValue < aMinSecondUnk)
				{
					aMinSecondUnk = aFinalSolValue;
					aNbChangesMinSecondUnk++;
				}
			}
			else if (aFinalUnk % aNbUnk == 2)
			{
				aMeanThirdUnk += aFinalSolValue;
				if (aFinalSolValue > aMaxThirdUnk)
				{
					aMaxThirdUnk = aFinalSolValue;
					aNbChangesMaxThirdUnk++;
				}
				if (aFinalSolValue < aMinThirdUnk)
				{
					aMinThirdUnk = aFinalSolValue;
					aNbChangesMinThirdUnk++;
				}
			}
			else
			{
				aMeanFourthUnk += aFinalSolValue;
				if (aFinalSolValue > aMaxFourthUnk)
				{
					aMaxFourthUnk = aFinalSolValue;
					aNbChangesMaxFourthUnk++;
				}
				if (aFinalSolValue < aMinFourthUnk)
				{
					aMinFourthUnk = aFinalSolValue;
					aNbChangesMinFourthUnk++;
				}
			}
		}

		aMinFirstUnk = CheckMinMaxValueIsChanged(aMaxFirstUnk, aMinFirstUnk, aNbChangesMinFirstUnk);
		aMaxFirstUnk = CheckMinMaxValueIsChanged(aMinFirstUnk, aMaxFirstUnk, aNbChangesMaxFirstUnk);
		aMinSecondUnk = CheckMinMaxValueIsChanged(aMaxSecondUnk, aMinSecondUnk, aNbChangesMinSecondUnk);
		aMaxSecondUnk = CheckMinMaxValueIsChanged(aMinSecondUnk, aMaxSecondUnk, aNbChangesMaxSecondUnk);
		aMinThirdUnk = CheckMinMaxValueIsChanged(aMaxThirdUnk, aMinThirdUnk, aNbChangesMinThirdUnk);
		aMaxThirdUnk = CheckMinMaxValueIsChanged(aMinThirdUnk, aMaxThirdUnk, aNbChangesMaxThirdUnk);
		aMinFourthUnk = CheckMinMaxValueIsChanged(aMaxFourthUnk, aMinFourthUnk, aNbChangesMinFourthUnk);
		aMaxFourthUnk = CheckMinMaxValueIsChanged(aMinFourthUnk, aMaxFourthUnk, aNbChangesMaxFourthUnk);

		aMeanFirstUnk /= aDividedVecSolSz;
		aMeanSecondUnk /= aDividedVecSolSz;
		aMeanThirdUnk /= aDividedVecSolSz;
		aMeanFourthUnk /= aDividedVecSolSz;

		std::vector<tREAL8> aVarianceVectorFirstUnk;
		std::vector<tREAL8> aVarianceVectorSecondUnk;
		std::vector<tREAL8> aVarianceVectorThirdUnk;
		std::vector<tREAL8> aVarianceVectorFourthUnk;

		for (int aSolNumber = 0; aSolNumber < aVecSolSz; aSolNumber++)
		{
			const tREAL8 aFinalSolValue = aVFinalSol(aSolNumber);
			if (aSolNumber % aNbUnk == 0)
				aVarianceVectorFirstUnk.push_back(std::abs(aFinalSolValue - aMeanFirstUnk) * std::abs(aFinalSolValue - aMeanFirstUnk));
			else if (aSolNumber % aNbUnk == 1)
				aVarianceVectorSecondUnk.push_back(std::abs(aFinalSolValue - aMeanSecondUnk) * std::abs(aFinalSolValue - aMeanSecondUnk));
			else if (aSolNumber % aNbUnk == 2)
				aVarianceVectorThirdUnk.push_back(std::abs(aFinalSolValue - aMeanThirdUnk) * std::abs(aFinalSolValue - aMeanThirdUnk));
			else
				aVarianceVectorFourthUnk.push_back(std::abs(aFinalSolValue - aMeanFourthUnk) * std::abs(aFinalSolValue - aMeanFourthUnk));
		}

		for (int aSolNumber = 0; aSolNumber < aDividedVecSolSz; aSolNumber++)
		{
			if (aSolNumber % aNbUnk == 0)
				aVarianceMeanFirstUnk += aVarianceVectorFirstUnk[aSolNumber];
			else if (aSolNumber % aNbUnk == 1)
				aVarianceMeanSecondUnk += aVarianceVectorSecondUnk[aSolNumber];
			else if (aSolNumber % aNbUnk == 2)
				aVarianceMeanThirdUnk += aVarianceVectorThirdUnk[aSolNumber];
			else
				aVarianceMeanFourthUnk += aVarianceVectorFourthUnk[aSolNumber];
		}

		aVarianceMeanFirstUnk /= aDividedVecSolSz;
		aVarianceMeanSecondUnk /= aDividedVecSolSz;
		aVarianceMeanThirdUnk /= aDividedVecSolSz;
		aVarianceMeanFourthUnk /= aDividedVecSolSz;

		StdOut() << "The minimum value for the first unknown is : " << aMinFirstUnk << " and the maximum value is : "
				 << aMaxFirstUnk << std::endl;
		StdOut() << "The minimum value for the second unknown is : " << aMinSecondUnk << " and the maximum value is : "
				 << aMaxSecondUnk << std::endl;
		StdOut() << "The minimum value for the third unknown is : " << aMinThirdUnk << " and the maximum value is : "
				 << aMaxThirdUnk << std::endl;
		StdOut() << "The minimum value for the fourth unknown is : " << aMinFourthUnk << " and the maximum value is : "
				 << aMaxFourthUnk << std::endl;
		StdOut() << "The mean value for the first unknown is : " << aMeanFirstUnk << " and the standard deviation value is : "
				 << std::sqrt(aVarianceMeanFirstUnk) << std::endl;
		StdOut() << "The mean value for the second unknown is : " << aMeanSecondUnk << " and the standard deviation value is : "
				 << std::sqrt(aVarianceMeanSecondUnk) << std::endl;
		StdOut() << "The mean value for the third unknown is : " << aMeanThirdUnk << " and the standard deviation value is : "
				 << std::sqrt(aVarianceMeanThirdUnk) << std::endl;
		StdOut() << "The mean value for the fourth unknown is : " << aMeanFourthUnk << " and the standard deviation value is : "
				 << std::sqrt(aVarianceMeanFourthUnk) << std::endl;
	}

	void DisplayLastUnknownValuesAndComputeStatistics(const tDenseVect &aVFinalSol, const bool aDisplayLastRadiometryValues,
													  const bool aDisplayLastTranslationValues, const bool aDisplayStatistics)
	{
		if (aDisplayLastTranslationValues && aDisplayLastRadiometryValues)
		{
			for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
			{
				StdOut() << aVFinalSol(aFinalUnk) << " ";
				if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
					StdOut() << std::endl;
			}
		}
		else if (aDisplayLastTranslationValues && !aDisplayLastRadiometryValues)
		{
			for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
			{
				if (aFinalUnk % 4 == 0 || aFinalUnk % 4 == 1)
					StdOut() << aVFinalSol(aFinalUnk) << " ";
				if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
					StdOut() << std::endl;
			}
		}
		else if (!aDisplayLastTranslationValues && aDisplayLastRadiometryValues)
		{
			for (int aFinalUnk = 0; aFinalUnk < aVFinalSol.DIm().Sz(); aFinalUnk++)
			{
				if (aFinalUnk % 4 == 2 || aFinalUnk % 4 == 3)
					StdOut() << aVFinalSol(aFinalUnk) << " ";
				if (aFinalUnk % 4 == 3 && aFinalUnk != 0)
					StdOut() << std::endl;
			}
		}
		if (aDisplayStatistics)
			ComputeStatisticsFourUnknowns(aVFinalSol);
	}

}; // namespace MMVII