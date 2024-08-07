/**
   @file  tpp_delaunay.hpp

   @brief  Declaration of the main Delaunay class of the Triangle++ wrapper

	 Original Triangle/TriLib code: http://www.cs.cmu.edu/~quake/triangle.html
	 Paper about Triangle/TriLib impl.: http://www.cs.cmu.edu/~quake-papers/triangle.ps

   @author  Marek Krajewski (mrkkrj), www.ib-krajewski.de
   @author  Piyush Kumar (piyush), http://compgeom.com/~piyush
   @author  Jonathan Richard Shewchuk (TriLib!!!), https://people.eecs.berkeley.edu/~jrs/
*/

#ifndef TRPP_DELAUNAY
#define TRPP_DELAUNAY

#include "dpoint.hpp"

#include <vector>
#include <string>
#include <unordered_map>

class Triwrap;
struct triangulateio;

namespace tpp
{
	class FaceIterator;
	class VertexIterator;
	class VoronoiVertexIterator;
	class VoronoiEdgeIterator;

	class TriangulationMesh;
	struct FacesList;
	struct VertexList;

	enum DebugOutputLevel // OPEN TODO:: forward-decl.
	{
		None,	// mute
		Info,	// most useful; it gives information on algorithmic progress and much more detailed statistics
		Vertex, // gives vertex-by-vertex details, and prints so much that Triangle runs much more slowly (!!!)
		Debug	// gives information only a debugger could love
	};

	enum AlgorithmType // OPEN TODO:: forward-decl.
	{
		DivideConquer, // the default!
		Incremental,
		Sweepline
	};

	/**
	   @brief: The main Delaunay class that wraps original Triangle (aka TriLib) code by J.R. Shewchuk

	   Use this class to produce Delaunay triangulations (and more...), for example:

		 Delaunay d(inputPoints);
		 d.Triangulate();
		 for(const auto& f: faces()) { ... } // iterate over triangles

	   @note: Currently the dpoint class by Piyush Kumar is used: a d-dimensional reviver::dpoint class
			  with d=2. If you want to use your own point class, you might have to work hard :-(...
	 */
	class TRPP_LIB_EXPORT Delaunay
	{
	public:
		typedef reviver::dpoint<double, 2> Point;  // OPEN TODO:: decouple from this dependency!
		typedef reviver::dpoint<double, 4> Point4; // OPEN TODO:: decouple from this dependency!

		/**
		   @brief: constructor

		   @param points: vector of 2 dimensional points to be used as input
		   @param enableMeshIndexing: enables incremental numbering of resulting vertices while iterating over
				  resulting faces/triangles (@see fIterator/FaceIterator)
		 */
		Delaunay(const std::vector<Point> &points = std::vector<Point>(), bool enableMeshIndexing = false);

		/**
		   @brief: destructor
		 */
		~Delaunay();

		//---------------------------------
		//  main API
		//---------------------------------

		/**
		   @brief: Delaunay triangulate the input points

		   This function triangulates points given as input to the constructor of this class. A quality
		   triangualtion can be also created here.

		   If segment constraints are set, this method creates a constrained Delaunay triangulation where
		   each PSLG segment is present as a single edge in the triangulation. Note that some of the resulting
		   triangles might *not be Delaunay*! In quality triangulation *additional* vertices called Steiner
		   points may be created.

		   @param quality: enforce minimal angle (default: 20�) and minimal area (default: none)
		   @param traceLvl: enable traces
		 */
		void Triangulate(bool quality = false, DebugOutputLevel traceLvl = None);

		/**
		  @brief: Convenience method
		 */
		void Triangulate(DebugOutputLevel traceLvl) { Triangulate(false, traceLvl); }

		/**
		   @brief: Conforming Delaunay triangulate the input points

		   This function triangulates points given as input to the constructor of this class, using the segments
		   set with setSegmentConstraint() as constraints. Here a conforming triangualtion will be created.

		   A conforming Delaunay triangulation is a *true Delaunay* triangulation in which each constraining
		   segment may have been *subdivided* into several edges by the insertion of *additional* vertices, called
		   Steiner points (@see: http://www.cs.cmu.edu/~quake/triangle.defs.html)

		   @param quality: enforce minimal angle (default: 20�) and minimal area (default: none)
		   @param traceLvl: enable traces
		 */
		void TriangulateConf(bool quality = false, DebugOutputLevel traceLvl = None);

		/**
		  @brief: Convenience method
		 */
		void TriangulateConf(DebugOutputLevel traceLvl) { TriangulateConf(false, traceLvl); }

		/**
			@brief: Voronoi tesselate the input points

			This function creates a Voronoi diagram for points given as input to the constructor of this
			class. Note that a Voronoi diagram can be only created if the underlying triangulation is convex
			and doesn't have holes!

			@param useConformingDelaunay: use conforming Delaunay triangulation as base for the Voronoi diagram
			@param traceLvl: enable traces
		  */
		void Tesselate(bool useConformingDelaunay = false, DebugOutputLevel traceLvl = None);

		/**
		  @brief: Enable incremental numbering of vertices in the triangulation while iterating over faces

		  @note: must be set before Triangulate() was called to take effect
		 */
		void enableMeshIndexGeneration();

		/**
		  @brief: Change the triangulation algorithm
		 */
		void setAlgorithm(AlgorithmType alg);

		//---------------------------------
		//  constraints API
		//---------------------------------

		/**
		  @brief: Set quality constraints for triangulation

		  @param angle: min. resulting angle, if angle <= 0, the constraint will be removed
		  @param area:  max. triangle area, if area <= 0, the constraint will be removed
		 */
		void setQualityConstraints(float angle, float area);

		/**
		  @brief: Convenience method
		 */
		void setMinAngle(float angle) { m_minAngle = angle; }

		/**
		  @brief: Convenience method
		 */
		void setMaxArea(float area) { m_maxArea = area; }

		/**
		  @brief: Convenience method
		 */
		void removeQualityConstraints() { setQualityConstraints(-1, -1); };

		/**
		  @brief: Set the segment constraints for triangulation

		  @param segments: vector of 2 dimensional points where each consecutive pair of points describes
						   a single segment. Both endpoints of every segment are vertices of the input vector,
						   and a segment may intersect other segments and vertices only at its endpoints!
		  @return: true if the input is valid, false otherwise
		 */
		bool setSegmentConstraint(const std::vector<Point> &segments);

		/**
		  @brief: Same as above, but using indexes of the input points
		 */
		bool setSegmentConstraint(const std::vector<int> &segmentPointIndexes, DebugOutputLevel traceLvl = None);

		/**
		  @brief: Use convex hull with constraining segments

		  @param useConvexHull: if true - generate convex hull using all specified points, the constraining
								segments are guaranteed to be included in the triangulation
		 */
		void useConvexHullWithSegments(bool useConvexHull);

		/**
		  @brief: Set holes to constrain the triangulation

		  @param holes: vector of 2 dimensional points where each points marks a hole, i.e. it infects all
						triangles around in until it sees a segment
		  @return: true if the input is valid, false otherwise
		 */
		bool setHolesConstraint(const std::vector<Point> &holes);

		/**
		  @brief: Set region constraints for the triangulation

		  @param regions: vector of 2 dimensional points where each points marks a regions, i.e. it infects all
						  triangles around in until it sees a segment
		  @param areas:  max. triangle area for the region with the same index in the regions vector
		  @return: true if the input is valid, false otherwise
		 */
		bool setRegionsConstraint(const std::vector<Point> &regions, const std::vector<float> &areas);

		/**
		  @brief: Convenience method
		 */
		bool setRegionsConstraint(const std::vector<Point4> &regionConstr); // OPEN TODO::: remove???

		/**
		   @brief:  Set a user test function for the triangulation
					OPEN TODO::: NYI!!!
		 */
		void setUserConstraint(bool (*f)()) { /* NYI !!!!! */ }

		/**
		  @brief: Are the quality constraints acceptable?

		  @param possible: set to true, if is highly *probable* for triangualtion to succeed
		  @return: true if triangualtion is *guaranteed* to succeed
		 */
		bool checkConstraints(bool &possible) const;

		/**
		  @brief: Are the quality constraints acceptable?

		  @param relaxed: report highly probable as correct too, as error otherwise
		  @return: true if triangualtion is guaranteed or higly probable to succeed
		 */
		bool checkConstraintsOpt(bool relaxed) const;

		/**
		  @brief: Get the acceptable ranges for quality constraints

		  @param guaranteed: up to this value triangualtion is guaranteed to succeed
		  @param possible: up to this value it is highly probable for triangualtion to succeed
		 */
		static void getMinAngleBoundaries(float &guaranteed, float &possible);

		//---------------------------------
		//  results API
		//---------------------------------

		/**
		  @brief: Is the triangulation completed?
		 */
		bool hasTriangulation() const;

		/**
		  @brief: Triangulation results, counts of entities:
		 */
		int edgeCount() const;
		int triangleCount() const;
		int verticeCount() const;
		int hullSize() const;
		int holeCount() const;

		/**
		  @brief: Min-max point coordinate values in the resulting triangulation
		 */
		void getMinMaxPoints(double &minX, double &minY, double &maxX, double &maxY) const;

		/**
		  @brief: Iterate over resulting faces (i.e. triangles) and vertices
		 */
		FaceIterator fbegin();
		FaceIterator fend();
		VertexIterator vbegin();
		VertexIterator vend();

		FacesList faces();
		VertexList vertices();

		/**
		  @brief: Tesselation results, counts of entities:
		 */
		int voronoiPointCount() const;
		int voronoiEdgeCount() const;

		/**
		  @brief: Iterate over Voronoi vertices and edges
		 */
		VoronoiVertexIterator vvbegin();
		VoronoiVertexIterator vvend();
		VoronoiEdgeIterator vebegin();
		VoronoiEdgeIterator veend();

		/**
		  @brief: Get a class for operations on oriented triangles (faces)
		 */
		TriangulationMesh mesh();

		/**
		   @brief: Helper - given a vertex index, return the actual Point from the input data
		 */
		const Point &pointAtVertexId(int vertexId) const;

		//---------------------------------
		//  file I/O API
		//---------------------------------

		/**
		  @brief: Write the current vertices to a text file in TriLib's .node file format.

		  @param filePath: directory and the name of file to be written
		  @return: true if file written, false otherwise
		 */
		bool savePoints(const std::string &filePath);

		/**
		  @brief: Write the current vertices and segments to a text file in TriLib's .poly file format.

		  @param filePath: directory and the name of file to be written
		  @return: true if file written, false otherwise
		 */
		bool saveSegments(const std::string &filePath);

		/**
		  @brief: Write the triangulation to an .off file
		  @note: OFF stands for the "Object File Format", a format used by Geometry Center's "Geomview" package.
		 */
		void writeoff(std::string &fname);

		/**
		  @brief: Read vertices from a text file in TriLib's .node file format.

		  @param filePath: directory and the name of file to be read
		  @param points: vertices read from the file
		  @return: true if file read, false otherwise
		 */
		bool readPoints(const std::string &filePath, std::vector<Point> &points);

		/**
		  @brief: Read vertices from a text file in TriLib's .poly file format.

		  @param filePath: directory and the name of file to be read
		  @param points: vertices read from the file
		  @param segmentEndpoints: indexes of the point pairs defining the segments, relative to the points vector
		  @param holeMarkers: coordinates of hole marker points
		  @param regionConstr: coordinates of region marker points, plus region attribute, plus the max area constraint for the region
		  @param duplicatePointCount: (optional) how many duplicate points were removed from input?
		  @param traceLvl: enable traces
		  @return: true if file read, false otherwise
		 */
		bool readSegments(const std::string &filePath, std::vector<Point> &points, std::vector<int> &segmentEndpoints,
						  std::vector<Delaunay::Point> &holeMarkers, std::vector<Point4> &regionConstr,
						  int *duplicatePointCount = nullptr, DebugOutputLevel traceLvl = None);

		/**
		   @brief: debug helper, works only if TRIANGLE_DBG_TO_FILE is set!
		 */
		void enableFileIOTrace(bool enable);

		//---------------------------------
		//  misc. API
		//---------------------------------

		/**
		   @brief: helper, use it to sort the Points first on their X then on Y coord.
				   OPEN:: compiler cannot instantiate less<> with operator<() for Point class?!
		 */
		struct OrderPoints
		{
			bool operator()(const Point &lhs, const Point &rhs) const;
		};

	private:
		void invokeTriLib(std::string &triswitches);
		void setQualityOptions(std::string &options, bool quality);
		void setDebugLevelOption(std::string &options, DebugOutputLevel traceLvl);
		void sanitizeInputData(std::unordered_map<int, int> duplicatePointsMap, DebugOutputLevel traceLvl = None);
		void freeTriangleDataStructs();
		void initTriangleDataForPoints();
		void initTriangleInputData(triangulateio *pin, const std::vector<Point> &points);
		void readPointsFromMesh(std::vector<Point> &points) const;
		void readSegmentsFromMesh(std::vector<int> &segmentEndpoints) const;
		void static SetPoint(Point &point, /*Triwrap::vertex*/ double *vertexptr);

		bool readSegmentsFromFile(char *polyfileName, FILE *polyfile, std::vector<int> &segmentEndpoints);
		void readHolesFromFile(char *polyfileName, FILE *polyfile, std::vector<Point> &holeMarkers, std::vector<Point4> &regionConstr) const;
		std::unordered_map<int, int> checkForDuplicatePoints() const;
		int GetFirstIndexNumber() const;

		friend class VertexIterator;
		friend class FaceIterator;
		friend class VoronoiVertexIterator;
		friend class VoronoiEdgeIterator;
		friend class TriangulationMesh;

	private:
		Triwrap *m_triangleWrap; // inner helper class grouping the original TriLib's C functions.

		void *m_in; // pointers to TriLib's intput, mesh & behavior
		void *m_pmesh;
		void *m_pbehavior;
		void *m_vorout; // pointer to TriLib's Voronoi output

		AlgorithmType m_triAlgorithm;
		float m_minAngle;
		float m_maxArea;
		bool m_convexHullWithSegments;
		bool m_extraVertexAttr;
		bool m_triangulated;

		std::vector<Point> m_pointList;
		std::vector<int> m_segmentList;
		std::vector<Point> m_holesList;
		std::vector<double> m_defaultExtraAttrs;
		std::vector<Point4> m_regionsConstrList;
	};

}

#endif
