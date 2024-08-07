 /** 
    @file  tpp_iterators.hpp    

    @brief Declaration of the Iterators for the Triangle++ wrapper.
           Iterators allow lazy access to the results of the triangualtion/tesselation

    @author  Marek Krajewski (mrkkrj), www.ib-krajewski.de
    @author  Piyush Kumar (piyush), http://compgeom.com/~piyush
 */

#ifndef TRPP_ITERATORS
#define TRPP_ITERATORS

 // OPEN TODO:: decouple Point class from Delaunay, then use forward declarations only!!!
#include "tpp_delaunay.hpp" // for Delaunay::Point


namespace tpp
{
   /**
      @brief: The face iterator for a Delaunay triangulation

        Implements access to the resulting oriented triangles (aka faces).

        A triangle abc has an origin (Org) a, a destination (Dest) b, and apex (Apex) c.
        These vertices occur in counterclockwise order about the triangle.
    */
   class TRPP_LIB_EXPORT FaceIterator
   {
   public:
      FaceIterator& operator++();
      FaceIterator operator++(int);

      FaceIterator() : m_delaunay(nullptr), meshPointCount(0) { floop.tri = nullptr; }

      bool empty() const;    // points to no triangle?  
      bool isdummy() const;  // deprecated!!!! --> pointing to a ghost triangle?
      bool isGhost() const;  // pointing to a ghost triangle?
      bool hasSteinerPoints() const;

      /**
         @brief: Get the origin point of the triangle

         @param point: if specified - the cordinates of the vertex
         @return: index of the vertex in the input vector, or -1 if a new vertex was created
       */
      int Org(Delaunay::Point* point = nullptr) const;
      int Dest(Delaunay::Point* point = nullptr) const;
      int Apex(Delaunay::Point* point = nullptr) const;

      /**
         @brief: Get the origin point of the triangle (@see Org() above) and its *mesh index*

         @param point: the cordinates of the vertex
         @param meshIndex: Index of the vertex in mesh (in order of iteration!)
       */
      void Org(Delaunay::Point& point, int& meshIndex) const;
      void Dest(Delaunay::Point& point, int& meshIndex) const;
      void Apex(Delaunay::Point& point, int& meshIndex) const;

      /**
         @brief: Calculate the area of the triangle
       */
      double area() const;

      // support for iterator dereferencing
      struct Face
      {
         Face(FaceIterator* iter) : m_iter(iter) {}

         // gets index in the input array
         int Org(Delaunay::Point* point = nullptr)   const { return m_iter->Org(point); }
         int Dest(Delaunay::Point* point = nullptr)  const { return m_iter->Dest(point); }
         int Apex(Delaunay::Point* point = nullptr)  const { return m_iter->Apex(point); }

         // gets index in the resulting mesh
         void Org(Delaunay::Point& point, int& meshIndex)   const { m_iter->Org(point, meshIndex); }
         void Dest(Delaunay::Point& point, int& meshIndex)  const { m_iter->Dest(point, meshIndex); }
         void Apex(Delaunay::Point& point, int& meshIndex)  const { m_iter->Apex(point, meshIndex); }

         // misc
         double area() const { return m_iter->area(); }

      private:
         FaceIterator* m_iter;
      };

      Face operator*() { return Face(this); }

      friend bool TRPP_LIB_EXPORT operator==(FaceIterator const&, FaceIterator const&);
      friend bool TRPP_LIB_EXPORT operator!=(FaceIterator const&, FaceIterator const&);
      friend bool TRPP_LIB_EXPORT operator<(FaceIterator const&, FaceIterator const&);

   private:
      struct tdata // TriLib's internal data
      {
         double*** tri;
         int orient;
      };

      typedef struct tdata poface; // = ptr. to oriented face

      FaceIterator(Delaunay* triangulator);

      int getVertexIndex(/*Triwrap::vertex*/ double* vertexptr) const;
      int getMeshVertexIndex(/*Triwrap::vertex*/ double* vertexptr) const;

      Delaunay* m_delaunay;         
      poface floop;                // TriLib's internal data
      mutable int meshPointCount;  // Used for numbering vertices in a complete triangulation   

      friend struct Face;
      friend class Delaunay;
      friend class TriangulationMesh;
   };
         

   /**
      @brief: This class supports iteration over faces in a foreach() loop
    */
   struct TRPP_LIB_EXPORT FacesList
   {
      FacesList(Delaunay* triangulator) : m_delaunay(triangulator) {}

      FaceIterator begin();
      FaceIterator end();

   private:
      Delaunay* m_delaunay;
   };


   /**
      @brief: The vertex iterator for a Delaunay triangulation

        Implements access to the resulting vertices of the triangulation
    */
   class TRPP_LIB_EXPORT VertexIterator
   {
   public:
      VertexIterator operator++();
      Delaunay::Point& operator*() const;

      VertexIterator() : vloop(nullptr), m_delaunay(nullptr) {}

      int vertexId() const; // standard internal numbering! ---> OPEN TODO::: mesh numbering, -1 for Steiner points ???!!!
      double x() const;
      double y() const;

      friend class Delaunay;
      friend bool TRPP_LIB_EXPORT operator==(VertexIterator const&, VertexIterator const&);
      friend bool TRPP_LIB_EXPORT operator!=(VertexIterator const&, VertexIterator const&);

   private:
      VertexIterator(Delaunay* triangulator);   

      void* vloop;  // TriLib's internal data
      Delaunay* m_delaunay;   
   };


   /**
      @brief: This class supports iteration over vertices in a foreach() loop
    */
   struct TRPP_LIB_EXPORT VertexList
   {
      VertexList(Delaunay* triangulator) : m_delaunay(triangulator) {}

      struct VertexListIterator : public VertexIterator
      {
         VertexListIterator(VertexIterator vit) : VertexIterator(vit) {}

         VertexListIterator operator++() {
            return VertexIterator::operator++();
         }

         const VertexIterator& operator*() const {
            return *this;
         }         
      };

      VertexListIterator begin();
      VertexListIterator end();

   private:
      Delaunay* m_delaunay;
   };


   /**
      @brief: The vertex iterator for a Voronoi tesselation

        Implements access to the resulting Voronoi points
    */
   class TRPP_LIB_EXPORT VoronoiVertexIterator
   {
   public:
      VoronoiVertexIterator operator++();
      Delaunay::Point& operator*() const;

      VoronoiVertexIterator();
      void advance(int steps);

      friend class Delaunay;
      friend bool TRPP_LIB_EXPORT operator==(VoronoiVertexIterator const&, VoronoiVertexIterator const&);
      friend bool TRPP_LIB_EXPORT operator!=(VoronoiVertexIterator const&, VoronoiVertexIterator const&);

   private:
      VoronoiVertexIterator(Delaunay* tiangulator);

      Delaunay* m_delaunay;   

      void* vvloop; // TriLib's internal data
      int vvindex;
      int vvcount;
   };


   /**
      @brief: The edges iterator for a Voronoi tesselation

        Implements access to the resulting connections between Voronoi points
    */
   class TRPP_LIB_EXPORT VoronoiEdgeIterator
   {
   public:    
      VoronoiEdgeIterator operator++();

      VoronoiEdgeIterator();

      // OPEN TODO:: comment!
      int startPointId() const;
      int endPointId(Delaunay::Point& normvec) const;

      /**
         @brief: Access the origin vertex (i.e. start point) of a Voronoi edge
         @return: the start point of the edge
       */
      const Delaunay::Point& Org();

      /**
         @brief: Access the destination vertex (i.e. end point) of a Voronoi edge

         @param finiteEdge: true for finite edges, false for inifinte rays
         @return: the end point of the edge, for infinite rays - the *normal vector* of the ray!
       */
      Delaunay::Point Dest(bool& finiteEdge);

      friend class Delaunay;
      friend bool TRPP_LIB_EXPORT operator==(VoronoiEdgeIterator const&, VoronoiEdgeIterator const&);
      friend bool TRPP_LIB_EXPORT operator!=(VoronoiEdgeIterator const&, VoronoiEdgeIterator const&);

   private:
      VoronoiEdgeIterator(Delaunay* tiangulator);

      Delaunay* m_delaunay;   
      void* veloop;  // TriLib's internal data
      int veindex;
      int vecount;
   };

} 

#endif
