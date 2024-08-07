 /** 
    @file  tpp_triangulation_mesh.hpp    
    @brief Declaration of the TriangulationMesh class for the Triangle++ wrapper.

    @author  Marek Krajewski (mrkkrj), www.ib-krajewski.de
 */

#ifndef TRPP_TRIANGULATION_MESH
#define TRPP_TRIANGULATION_MESH

#include <vector>

namespace tpp
{
   class Delaunay;
   class FaceIterator;

   /**
      @brief: Class for operations on oriented triangles (faces) of a triangulation mesh

        A triangle abc has an origin (Org) a, a destination (Dest) b, and apex (Apex) c.
        These vertices occur in counterclockwise order about the triangle.

        ::: OPEN TODO :::
         dnext:  Find the next edge counterclockwise with the same destination.
         dnext(abc) -> *ba

         dprev:  Find the next edge clockwise with the same destination.
         dprev(abc) -> cb*

         rnext:  Find the next edge (counterclockwise) of the adjacent triangle.
         rnext(abc) -> *a*

         rprev:  Find the previous edge (clockwise) of the adjacent triangle.
         rprev(abc) -> b**
    */
   class TRPP_LIB_EXPORT TriangulationMesh
   {
   public:
      /**
         @brief: Access the triangle adjoining edge N

         Example:
           Sym(abc, N = 0) -> ba*
           Sym(abc, N = 1) -> cb*
           Sym(abc, N = 2) -> ac*

         Here '*' stands for the farthest vertex on the adjoining triangle whose index is returned

         @param fit: face iterator
         @param i: edge number (N)
         @return: The vertex on the opposite face, or -1 if the edge is part of the convex hull
                  (@see FaceIterator::Org() above)
       */
      int Sym(FaceIterator const& fit, char i) const;

      /**
         @brief: Access the triangle opposite to current edge of the face

         @param fit: face iterator
         @return: iterator of the opposite face. It is empty if the edge is on the convex hull
       */
      FaceIterator Sym(FaceIterator const& fit) const;

      /**
         @brief: Find the next edge (counterclockwise) of a triangle

         @param fit: face iterator
         @return: iterator corresponding to the next counterclockwise edge of a triangle,
                  Lnext(abc) -> bca
       */
      FaceIterator Lnext(FaceIterator const& fit);

      /**
         @brief: Find the previous edge (clockwise) of a triangle

         @param fit: face iterator
         @return: iterator corresponding to the previous clockwise edge of a triangle,
                  Lprev(abc) -> cab
       */
      FaceIterator Lprev(FaceIterator const& fit);

      /**
         @brief: Find the next edge (counterclockwise) of a triangle with the same origin

         @param fit: face iterator
         @return: iterator corresponding to the next edge counterclockwise with the same origin,
                  Onext(abc) -> ac* (@see Sym() above)
       */
      FaceIterator Onext(FaceIterator const& fit);

      /**
         @brief: Find the next edge clockwise with the same origin

         @param fit: face iterator
         @return: iterator corresponding to the next edge clockwise with the same origin,
                  Oprev(abc) -> a*b (@see Sym() above)
       */
      FaceIterator Oprev(FaceIterator const& fit);

      /**
         @brief: Calculate incident triangles around a vertex

         Note that behaviour is undefined if vertexId is greater than number of vertices - 1.
         All triangles returned have Org(triangle) = vertexId and are in counterclockwise order.

         @param vertexId: the vertex for which you want incident triangles
         @param ivv: triangles around a vertex in counterclockwise order
       */
      void trianglesAroundVertex(int vertexId, std::vector<int>& ivv);

      /**
         @brief:  Point-locate a vertex V

         @param vertexId: the vertex
         @return: a face iterator whose origin is V
       */
      FaceIterator locate(int vertexId);

   private:
      TriangulationMesh(Delaunay* triangulator);

      Delaunay* m_delaunay;

      friend class Delaunay;
   };

} 

#endif
