#ifndef  _MMVII_GEOM3D_H_
#define  _MMVII_GEOM3D_H_

namespace MMVII
{

template<class T> cPtxd<T,3>  PFromNumAxe(int aNum); ///< return I,J or K according to 0,1 or 2
/// use the 3 "colum vector" to compute the matrix
template<class T> cDenseMatrix<T> MatFromCols(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2);
/// use the 3 "line vector" to compute the matrix
template<class T> cDenseMatrix<T> MatFromLines(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2);
/// Vector product 
template <class T>  cPtxd<T,3> operator ^ (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2);
// Return one vector orthog,  choice is not univoque , quikcly select on stable
template<class T> cPtxd<T,3>  VOrthog(const cPtxd<T,3> & aP);

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

       tPt   AxeI() const ;
       tPt   AxeJ() const ;
       tPt   AxeK() const ;

       // Compute a normal repair, first vector being colinear to Pt
       static cRotation3D<Type> CompleteRON(const tPt & aPt);
       // Compute a normal repair, first vector being colinear to P1, second in the plane P1,P2
       static cRotation3D<Type> CompleteRON(const tPt & aP0,const tPt & aP1);
       // Compute a rotation arround a given axe and with a given angle
       static cRotation3D<Type> RotFromAxe(const tPt & anAxe,Type aTeta);
       // Compute a random rotation for test/bench
       static cRotation3D<Type> RandomRot();

       // Extract Axes of a rotation and compute its angle 
       void ExtractAxe(tPt & anAxe,Type & aTeta);

    private :
       cDenseMatrix<Type>  mMat;
};

/**  Class for 3D "affine" rotation of vector

*/

template <class Type> class cIsometrie3D
{
    public :
       static constexpr int       TheDim=3;
       typedef cPtxd<Type,3>      tPt;
       typedef cTriangle<Type,3>  tTri;
       typedef Type               tTypeElem;
       typedef cIsometrie3D<Type> tTypeMap;
       typedef cIsometrie3D<Type> tTypeMapInv;
       static const int NbDOF()   {return 6;}


       cIsometrie3D(const tPt& aTr,const cRotation3D<Type> &);
       tTypeMapInv  MapInverse() const; // {return cIsometrie3D(-mRot.Inverse(mTr),mRot.MapInverse());}

       static cIsometrie3D<Type> FromRotAndInOut(const cRotation3D<Type> &,const tPt& aPtIn,const tPt& aPtOut );
       static cIsometrie3D<Type> FromTriInAndOut(const tTri  & aTriIn,const tTri  & aTriOut);

       const cRotation3D<Type> & Rot() const {return mRot;}  ///< Accessor
       const tPt Tr() const {return mTr;}  ///< Accessor

       tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt);}
       tPt   Inverse(const tPt & aPt) const {return mRot.Inverse(aPt-mTr) ;}  // Work as M tM = Id

    private :
       tPt                mTr;
       cRotation3D<Type>  mRot;
};


template <class Type> class cTriangulation3D : public cTriangulation<Type,3>
{
        public :
           typedef cPtxd<Type,3>  tPt;
           typedef cTriangulation3D<Type>  tTriangulation3D;
           /// Constructor from file, include ply format, maybe later others (internals?)  if required
           cTriangulation3D(const std::string &);
           void WriteFile(const std::string &,bool isBinary) const;

	   static void Bench();

        private :
           /// Read/Write in ply format using
           void PlyInit(const std::string &);
           void PlyWrite(const std::string &,bool isBinary) const;
};


};

#endif  //  _MMVII_GEOM3D_H_
