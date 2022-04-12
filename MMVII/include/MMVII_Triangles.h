#ifndef  _MMVII_Triangles_H_
#define  _MMVII_Triangles_H_
namespace MMVII
{

class cEdgeDual
{
     public :
        static constexpr int NO_INIT=-1;
        cEdgeDual();
        cEdgeDual(int aS1,int aS2,int aF1);
        /// return 1 or 2, -1 if none and none allowed
        int GetOtherSom(int aS,bool AllowNone)  const
        {
            if (mI1==aS) return mI2;
            if (mI2==aS) return mI1;
            MMVII_INTERNAL_ASSERT_tiny(AllowNone,"cEdgeDual::GetNumSom");
            return NO_INIT;
        }
        void SetFace2(int aF) ;
     private :
        int mI1;
        int mI2;
        int mF1;
        int mF2;
};

class cGraphDual
{
     public :
         typedef std::list<cEdgeDual *> tListAdj;
         typedef cPt3di tFace;

         cGraphDual();
         void Init(int aNbSom,const std::vector<tFace>&);
         void  AddEdge(int aFace,int aS1,int aS2);
         void  AddTri(int aFace,const cPt3di &);

	 std::list<int>  SuccOfSom(int aS1) const;
     private :
         cEdgeDual * GetEdge(int aS1,int aS2); ///< return egde s1->s2 if it exist, else return null

         std::vector<tListAdj>     mSomNeigh;
         std::vector<tListAdj>     mFaceNeigh;
         std::vector<cEdgeDual> mReserve;
};



/// Class for storing  basic triangle in 2 or 3 D
template <class Type,const int Dim> class  cTriangle
{
     public :
       typedef cPtxd<Type,Dim>     tPt;
       typedef cTriangle<Type,Dim> tTri;

       cTriangle(const tPt & aP0,const tPt & aP1,const tPt & aP2);

       tTri  TriSwapPt(int aK0) const; ///< "same" but with different orientation by swap K0/1+K0

       static tTri  RandomTri(const Type & aSz,const Type & aRegulMin = Type(1e-2));
       /// aWeight  encode in a point the 3 weights
       tPt  FromCoordBarry(const cPtxd<Type,3> & aWeight) const;
       /// Barrycenter with equal weights
       tPt  Barry() const;

       /// How much is it a non degenerate triangle,  without unity, 0=> degenerate
       Type Regularity() const;
       /// Area of the triangle
       Type Area() const;
       /// Point equidistant to 3 point,  To finish for dim 3
       tPt CenterInscribedCircle() const;
       const tPt & Pt(int aK) const;   ///< Accessor
       tPt KVect(int aK) const;   ///<   Pk->Pk+1
       cTplBox<Type,Dim>  BoxEngl() const;
       cTplBox<int,Dim>     BoxPixEngl() const;  // May be a bit bigger

     protected :
       tPt  mPts[3];
};


/// return 2 elementay triangle both oriented, DiagCrois : diag contain 00->11 , else 01->10
template <class Type> const std::vector<cTriangle<Type,2> > &  SplitPixIn2(bool DiagCrois);



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

          int  NbFace() const;  ///< Number of faces
          int  NbPts() const;   ///< Number of points
          const tFace &  KthFace(int aK) const;  ///<  Faces number K
	  const tPt  & KthPts(int aK) const;  ///< Points number K

          tTri  KthTri(int aK) const;  ///< Triangle corresponding to the face
	  bool  ValidFace(const tFace &) const;  ///< is it a valide face (i.e. : all index in [0,NbPts[)

	  /// Create a sub tri of vertices belonging to the set, require 1,2 or 3 vertice in each tri
	  void Filter(const cDataBoundedSet<tREAL8,Dim> &,int aNbVertixThres=3) ;
	  /// Box of Pts, error when empty, FactMargin make it slightly bigger
	  cTplBox<tCoord,Dim>  BoxEngl(Type aFactMargin = 1e-2) const;

	  /// Equality is difficiult, because of permutation,just make heuristik test
	  bool  HeuristikAlmostEqual (const cTriangulation<Type,Dim> &,Type TolPt,Type TolFace)  const;

	  ///  Generate topology of dual graphe => implemanted but 4 now, nor used nor tested ...
	  void MakeTopo();
          const cGraphDual & DualGr() const;
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
