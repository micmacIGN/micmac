#ifndef  _MMVII_GEOM3D_H_
#define  _MMVII_GEOM3D_H_

#include "MMVII_Triangles.h"

namespace MMVII
{

template<class T> cPtxd<T,3>  PFromNumAxe(int aNum); ///< return I,J or K according to 0,1 or 2
/// use the 3 "colum vector" to compute the matrix
template<class T> cDenseMatrix<T> MatFromCols(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2);
/// use the 3 "line vector" to compute the matrix
template<class T> cDenseMatrix<T> MatFromLines(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2);
/// Vector product 
template <class T>  cPtxd<T,3> operator ^ (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2)
{
   return cPtxd<T,3>
          (
               aP1.y() * aP2.z() -aP1.z()*aP2.y(),
               aP1.z() * aP2.x() -aP1.x()*aP2.z(),
               aP1.x() * aP2.y() -aP1.y()*aP2.x()
          );
}

/// Matrix corresponf to P ->  W ^ P
template<class T> cDenseMatrix<T> MatProdVect(const cPtxd<T,3>& aW);

// Return one vector orthog,  choice is not univoque , quikcly select on stable
template<class T> cPtxd<T,3>  VOrthog(const cPtxd<T,3> & aP);

template<class Type> cPtxd<Type,3> NormalUnit(const cTriangle<Type,3> &);  // V01 ^ V02
template<class Type> cPtxd<Type,3> Normal(const cTriangle<Type,3> &);  // V01 ^ V02

// ===============================================================================
//  Quaternion part  : I use them essentially for interface with other library,
//  BTW, I implement a bit of elementary quat-algebra that allow to check the
//  correctness . I dont create a new class, it's just an interpration of 
//  4D points. 
// ===============================================================================

/// Quaternion multiplication
template<class T> cPtxd<T,4>  operator * (const cPtxd<T,4> & aP1,const cPtxd<T,4> & aP2); 
/// Quaternion => Rotation
template<class T> cDenseMatrix<T>  Quat2MatrRot  (const cPtxd<T,4> & aPt);
///  Rotation => Quaternion
template<class T> cPtxd<T,4>  MatrRot2Quat  (const cDenseMatrix<T> &);
/*
/// Quaternion => Matrix 4x4 , with isomorphisme between * on both
template<class T> cDenseMatrix<T>  Quat2Matr4x4  (const cPtxd<T,4> & aPt);
///  Matrix 4x4 => Quaternion
template<class T> cPtxd<T,4>  Matr4x42Quat  (const cDenseMatrix<T>&);
*/


/** \file MMVII_Geom3D.h
    \brief contain classes for geometric manipulation, specific to 3D space :
           3D line, 3D plane, rotation, ...
*/


/**  Class for 3D rotation of vector, internal representation data contains only the matrix, 
     manipulation with quaternion, angles ... are (will be) used only for import/export

     This class is also used for reprensting ortho normal repair as it the same object
     with two different interpretation
     

     Also they will be almost always used with double, they are templated in case
     long double would be necessary.
*/

template <class Type> class cRotation3D
{
    public :
       static constexpr int       TheDim=3;
       typedef cPtxd<Type,3>      tPt;
       typedef Type               tTypeElem;
       typedef cRotation3D<Type>  tTypeMap;
       typedef cRotation3D<Type>  tTypeMapInv;
       static const int NbDOF()   {return 3;}


       // RefineIt : if true, assume not fully orthog and compute closest one
       cRotation3D(const cDenseMatrix<Type> &,bool RefineIt);
       const cDenseMatrix<Type> & Mat() const {return mMat;}

       tPt   Value(const tPt & aPt) const  {return mMat * aPt;}
       tPt   Inverse(const tPt & aPt) const {return aPt  * mMat ;}  // Work as M tM = Id
       // tTypeMapInv  MapInverse() const {return cRotation3D(mMat.Transpose(),false);}
       tTypeMapInv  MapInverse() const;
       tTypeMap  operator* (const tTypeMap &) const;
       static tTypeMap Identity();

       tPt   AxeI() const ;
       tPt   AxeJ() const ;
       tPt   AxeK() const ;

       // Compute a normal repair, first vector being colinear to Pt
       static cRotation3D<Type> CompleteRON(const tPt & aPt);
       // Compute a normal repair, first vector being colinear to P1, second in the plane P1,P2
       static cRotation3D<Type> CompleteRON(const tPt & aP0,const tPt & aP1);
       // Compute a rotation arround a given axe and with a given angle
       static cRotation3D<Type> RotFromAxe(const tPt & anAxe,Type aTeta);
       //  Axiator close to Rot From but teta=Norm !!  exp(Mat(^Axe))
       static cRotation3D<Type> RotFromAxiator(const tPt & anAxe);
       // Compute a random rotation for test/bench
       static cRotation3D<Type> RandomRot();
       // Compute a "small" random rot controlled by ampl
       static cRotation3D<Type> RandomRot(const Type & aAmpl);

       // Extract Axes of a rotation and compute its angle 
       void ExtractAxe(tPt & anAxe,Type & aTeta);

       // conversion to Omega Phi Kapa
       static cRotation3D<Type>  RotFromWPK(const tPt & aWPK);
       tPt                       ToWPK() const;

       // conversion to Yaw Pitch Roll
       static cRotation3D<Type>  RotFromYPR(const tPt & aWPK);
       tPt                       ToYPR() const;

    private :
       cDenseMatrix<Type>  mMat;
};

/**  Class for 3D "affine" rotation of vector

*/

template <class Type> class cIsometry3D
{
    public :
       static constexpr int       TheDim=3;
       typedef cPtxd<Type,3>      tPt;
       typedef cPtxd<Type,2>      tPt2;
       typedef cTriangle<Type,3>  tTri;
       typedef cTriangle<Type,2>  tTri2d;
       typedef Type               tTypeElem;
       typedef cIsometry3D<Type> tTypeMap;
       typedef cIsometry3D<Type> tTypeMapInv;
       static const int NbDOF()   {return 6;}


       cIsometry3D(const tPt& aTr,const cRotation3D<Type> &);
       tTypeMapInv  MapInverse() const; // {return cIsometry3D(-mRot.Inverse(mTr),mRot.MapInverse());}
       tTypeMap  operator* (const tTypeMap &) const;
       static tTypeMap Identity();

       /// Return Isometrie with given Rot such I(PTin) = I(PTout)
       static cIsometry3D<Type> FromRotAndInOut(const cRotation3D<Type> &,const tPt& aPtIn,const tPt& aPtOut );
       /// Return Isome such thqt I(InJ) = OutK ;  In(InJJp1) // OutKKp1 ; In(Norm0) = NormOut
       static cIsometry3D<Type> FromTriInAndOut(int aKIn,const tTri  & aTriIn,int aKOut,const tTri  & aTriOut);
       /// Idem put use canonique tri = 0,I,J as input
       static cIsometry3D<Type> FromTriOut(int aKOut,const tTri  & aTriOut,bool Direct=true);

       /// return a 2D triangle isometric to 3d, PK in 0,0  PK->PK1 // to Ox
       static tTri2d ToPlaneZ0(int aKOut,const tTri  & aTriOut,bool Direct=true);


       void SetRotation(const cRotation3D<Type> &);

       const cRotation3D<Type> & Rot() const {return mRot;}  ///< Accessor
       const tPt &Tr() const {return mTr;}  ///< Accessor
       tPt &Tr() {return mTr;}  ///< Accessor

       tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt);}
       tPt   Inverse(const tPt & aPt) const {return mRot.Inverse(aPt-mTr) ;}  // Work as M tM = Id

       cSimilitud3D<Type>  ToSimil() const; ///< make a similitude with scale 1

    private :
       tPt                mTr;
       cRotation3D<Type>  mRot;
};

template <class Type> class cSimilitud3D
{
    public :
       static constexpr int       TheDim=3;
       typedef cPtxd<Type,3>      tPt;
       typedef cPtxd<Type,2>      tPt2;
       typedef cTriangle<Type,3>  tTri;
       typedef Type               tTypeElem;
       typedef cSimilitud3D<Type> tTypeMap;
       typedef cSimilitud3D<Type> tTypeMapInv;
       static const int NbDOF()   {return 7;}


       cSimilitud3D(const Type & aScale,const tPt& aTr,const cRotation3D<Type> &);
       tTypeMapInv  MapInverse() const; // {return cIsometry3D(-mRot.Inverse(mTr),mRot.MapInverse());}
       tTypeMap  operator* (const tTypeMap &) const;

       /// Return Similitud with given Rot such I(PTin) = I(PTout)
       static tTypeMap FromScaleRotAndInOut(const Type&,const cRotation3D<Type> &,const tPt& aPtIn,const tPt& aPtOut );
       /// Return Similitud such thqt I(InJ) = OutK ;  In(InJJp1) // OutKKp1 ; In(Norm0) = NormOut
       static tTypeMap FromTriInAndOut(int aKIn,const tTri  & aTriIn,int aKOut,const tTri  & aTriOut);
       /// Idem put use canonique tri = 0,I,J as input
       static tTypeMap FromTriOut(int aKOut,const tTri  & aTriOut);
       /*  Create a cIsom that aline seg KKp1 of tri on P1->P2  and the normal oriented on axe Z
       */
       static cSimilitud3D<Type> FromTriInAndSeg(const tPt2&aP1,const tPt2&aP2,int aKIn,const tTri  & aTriIn);

       const cRotation3D<Type> & Rot() const {return mRot;}  ///< Accessor
       const tPt & Tr() const {return mTr;}  ///< Accessor
       const Type & Scale() const {return mScale;}  ///< Accessor

       tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt)*mScale;}
       tPt   Inverse(const tPt & aPt) const {return mRot.Inverse((aPt-mTr)/mScale) ;}  // Work as M tM = Id

    private :
       tTypeElem          mScale;
       tPt                mTr;
       cRotation3D<Type>  mRot;
};

/**  Class to store the devlopment planar of two adjacent faces      :          P2
 * adjacent face . At the end we have two 2Dtriangle   with  P0P1    :         /    \   [T1] 
 * identic, P2 of each side.  T1 direct , T2 indirect (if there      :       /       \
 * is no orientation problem                                         :      P0 -------P1
 *                                                                   :       |     _ /
 *                                                                   :        P2 -      [T2]
 * */

template <class Type> class cDevBiFaceMesh
{
   public :
      typedef  cPtxd<Type,2> tPt;

      cDevBiFaceMesh(const cTriangle<Type,2> & aT1, const cTriangle<Type,2> & aT2);
      cDevBiFaceMesh();
      bool Ok() const;
      bool WellOriented() const;
      const cTriangle<Type,2> & T1() const;
      const cTriangle<Type,2> & T2() const;

   private :
      void AssertOk() const;
      bool              mOk;
      bool              mWellOriented;
      cTriangle<Type,2> mT1;
      cTriangle<Type,2> mT2;
};




template <class Type> class cTriangulation3D : public cTriangulation<Type,3>
{
        public :
           typedef cPtxd<Type,3>      tPt;
           typedef cPt3di             tFace;
           typedef std::vector<tPt>   tVPt;
           typedef std::vector<tFace> tVFace;

           typedef cTriangulation3D<Type>  tTriangulation3D;
           /// Constructor from file, include ply format, maybe later others (internals?)  if required
           cTriangulation3D(const std::string &);
           cTriangulation3D(const tVPt&,const tVFace &);
           void WriteFile(const std::string &,bool isBinary) const;

	   static void Bench();

	   void CheckOri3D();
	   void CheckOri2D();

	   cTriangle<Type,2>     TriDevlpt(int aKF,int aNumSom) const;  // aNumSom in [0,1,2]
	   cDevBiFaceMesh<Type>  DoDevBiFace(int aKF1,int aNumSom) const;  // aNumSom in [0,1,2]
        private :
           /// Read/Write in ply format using
           void PlyInit(const std::string &);
           void PlyWrite(const std::string &,bool isBinary) const;
};


};

#endif  //  _MMVII_GEOM3D_H_
