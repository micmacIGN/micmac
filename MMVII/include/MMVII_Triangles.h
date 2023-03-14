#ifndef  _MMVII_Triangles_H_
#define  _MMVII_Triangles_H_

#include "MMVII_Ptxd.h"


namespace MMVII
{

class cEdgeDual
{
     public :
        static constexpr int NO_INIT=-1;
        cEdgeDual();
        cEdgeDual(int aS1,int aS2,int aF1);
        /// return 1 or 2, -1 if none and none allowed
        int GetOtherSom(int aS,bool AllowNone)  const { return GetOtherObj(mS,aS,AllowNone); }
        int GetOtherFace(int aF,bool AllowNone) const { return GetOtherObj(mF,aF,AllowNone); }
        void SetFace2(int aF) ;
     private :
        typedef int  tTab2I[2];
        int GetOtherObj(const tTab2I &aTab,int aObj,bool AllowNone)  const
        {
            if (aTab[0]==aObj) return aTab[1];
            if (aTab[1]==aObj) return aTab[0];
            MMVII_INTERNAL_ASSERT_tiny(AllowNone,"cEdgeDual::GetNumSom");
            return NO_INIT;
        }

        tTab2I  mS;
        tTab2I  mF;
};

class cGraphDual
{
     public :
         typedef std::list<cEdgeDual *> tListAdj;
         typedef cPt3di tFace;

         cGraphDual();
         void Init(int aNbSom,const std::vector<tFace>&);
         void  AddTri(int aFace,const cPt3di &);

         cEdgeDual * GetEdgeOfSoms(int aS1,int aS2) const; ///< return egde s1->s2 if it exist, else return null
	 void  GetSomsNeighOfSom(std::vector<int> & aRes,int aS1) const;
	 void  GetFacesNeighOfFace(std::vector<int> & aRes,int aF1) const;
     private :
         void  AddEdge(int aFace,int aS1,int aS2);
         cGraphDual(const cGraphDual &) = delete;

         std::vector<tListAdj>     mSomNeigh;
         std::vector<tListAdj>     mFaceNeigh;
         std::list<cEdgeDual>      mReserve;
};



/// Class for storing  basic triangle in 2 or 3 D
template <class Type,const int Dim> class  cTriangle
{
     public :
       typedef cPtxd<Type,Dim>     tPt;
       typedef cTriangle<Type,Dim> tTri;

       cTriangle(const tPt & aP0,const tPt & aP1,const tPt & aP2);

       /// some time we need a fake init, safer to have it explict rather than using a default constructor
       static tTri  Tri000();

       tTri  TriSwapPt(int aK0) const; ///< "same" but with different orientation by swap K0/1+K0

       static tTri  RandomTri(const Type & aSz,const Type & aRegulMin = Type(1e-2));
       /// aWeight  encode in a point the 3 weights
       tPt  FromCoordBarry(const cPtxd<Type,3> & aWeight) const;
       /// Barrycenter with equal weights
       tPt  Barry() const;

       ///  return K such that Pt(K)Pt(K+1) is the longest
       int  IndexLongestSeg() const;

       /// How much is it a non degenerate triangle,  without unity, 0=> degenerate (point aligned)
       Type Regularity() const;
       /// High degeneracy, 0 indicate two point indentics
       Type MinDist() const;
       /// Area of the triangle
       Type Area() const;
       /// Point equidistant to 3 point,  To finish for dim 3
       tPt CenterInscribedCircle() const;
       const tPt & Pt(int aK) const;   ///< Accessor
       const tPt & PtCirc(int aK) const;   ///<  %3, always correc even >3 or <0
       tPt KVect(int aK) const;   ///<   Pk->Pk+1
       cTplBox<Type,Dim>  BoxEngl() const;
       cTplBox<int,Dim>     BoxPixEngl() const;  // May be a bit bigger


     protected :
       tPt  mPts[3];
};
typedef   cTriangle<tREAL8,2>  tTri2dr;
typedef   cTriangle<tREAL8,3>  tTri3dr;


/// return 2 elementay triangle both oriented, DiagCrois : diag contain 00->11 , else 01->10
template <class Type> const std::vector<cTriangle<Type,2> > &  SplitPixIn2(bool DiagCrois);

template <class Type,const int Dim>  cTriangle<Type,Dim>  TriFromFace(const std::vector<cPtxd<Type,Dim>> &, const cPt3di &);

/// return K such Face[K] = NumS, if  not found : -1 if SVP, error if not
int  IndOfSomInFace(const cPt3di & aFace,int aNumS,bool SVP=false);

template <class Type,const int Dim> class cTriangulation
{
     public :
          typedef Type                  tCoord;
          typedef cPtxd<tCoord,Dim>     tPt;
          typedef cTriangle<tCoord,Dim> tTri;
          typedef cPt3di                tFace;
          typedef std::vector<tPt>      tVPt;
          typedef std::vector<tFace>    tVFace;

	  tPt PAvg() const; ///< return an average point 
	  int   IndexClosestFace(const tPt& aPClose) const; ///< Face closest to a given point
	  /**  return a Face more or less at the center,  4 now ompute face closest to Avg , not perfect
	   * but work with simple suface, if necessary will evolve as a real geodetic center */
	  int   IndexCenterFace() const;

	  //  ============ Accessor ========================
	  //
          const std::vector<tPt> &   VPts() const;  ///< Standard Accessor
          const std::vector<tFace> & VFaces() const;  ///< Standard Accessor

          size_t  NbFace() const;  ///< Number of faces
          size_t  NbPts() const;   ///< Number of points
          const tFace &  KthFace(size_t aK) const;  ///<  Faces number K
	  const tPt  & KthPts(size_t aK) const;  ///< Points number K
	  tPt  & KthPts(size_t aK) ;  ///< Points number K

          tTri  KthTri(int aK) const;  ///< Triangle corresponding to the face
	  bool  ValidFace(const tFace &) const;  ///< is it a valide face (i.e. : all index in [0,NbPts[)

	  /// Create a sub tri of vertices belonging to the set, require 1,2 or 3 vertice in each tri
	  void Filter(const cDataBoundedSet<tREAL8,Dim> &,int aNbVertixThres=3) ;
	  /// Box of Pts, error when empty, FactMargin make it slightly bigger
	  cTplBox<tCoord,Dim>  BoxEngl(Type aFactMargin = 1e-2) const;

	  /// Equality is difficiult, because of permutation,just make heuristik test
	  bool  HeuristikAlmostEqual (const cTriangulation<Type,Dim> &,Type TolPt,Type TolFace)  const;

	  ///  Generate topology of dual graphe => implemanted ; used in mesh dev (so +or- tested)
	  void MakeTopo();
          const cGraphDual & DualGr() const;

	  ///  Make some (basic) test on correction of a triangulation, eventually correct some default
          bool CheckAndCorrect(bool Correct);


	  std::vector<size_t> IndexPts3D(size_t aNpPtsTot);

     protected :
	  /// More a
	  bool  HeuristikAlmostInclude (const cTriangulation<Type,Dim> &,Type TolPt,Type TolFace)  const;

          cTriangulation(const tVPt& =tVPt(),const tVFace & =tVFace());
          void AddFace(const tFace &);
          void ResetTopo();

          std::vector<tPt>    mVPts;
          std::vector<tFace>  mVFaces;
          cGraphDual          mDualGr;
};

};

#endif  //  _MMVII_Triangles_H_
