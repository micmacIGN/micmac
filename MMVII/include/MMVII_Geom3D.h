#ifndef  _MMVII_GEOM3D_H_
#define  _MMVII_GEOM3D_H_

namespace MMVII
{

template<class T> cPtxd<T,3>  PFromNumAxe(int aNum);
// template<class T> cPtxd<T,3>  VUnit(const cPtxd<T,3> &);
template<class T> cDenseMatrix<T> MatFromCols(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2);
template<class T> cDenseMatrix<T> MatFromLines(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2);
/// Vector product 
template <class T>  cPtxd<T,3> operator ^ (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2);
// Return one vector orthog, as the choice is not univoque just avoid degenerency
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
       typedef cPtxd<Type,3> tPt;
       // RefineIt : if true, assume not fully orthog and compute closest one
       cRotation3D(const cDenseMatrix<Type> &,bool RefineIt);
       const cDenseMatrix<Type> & Mat() const {return mMat;}
       tPt   Direct(const tPt & aPt) const  {return mMat * aPt;}
       tPt   Inverse(const tPt & aPt) const {return aPt  * mMat ;}  // Work as M tM = Id

       tPt   AxeI() const  {return tPt::Col(mMat,0);}
       tPt   AxeJ() const  {return tPt::Col(mMat,1);}
       tPt   AxeK() const  {return tPt::Col(mMat,2);}

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

class cTriangulation3D : public cTriangulation<3>
{
        public :
            typedef cTriangulation<3>::tPt  tPt;
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
