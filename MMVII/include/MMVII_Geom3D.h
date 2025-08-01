#ifndef  _MMVII_GEOM3D_H_
#define  _MMVII_GEOM3D_H_

#include "MMVII_Triangles.h"
#include "MMVII_Matrix.h"

namespace MMVII
{

typedef cSegment<tREAL8,3> tSeg3dr;

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

template <class Type> inline cPtxd<Type,3> PSymXY (const cPtxd<Type,3> & aP)
{
    return cPtxd<Type,3>(aP.y(),aP.x(),aP.z());
}


///< compute determinant  as A.(B ^ C)
template <class T>  T  Determinant (const cPtxd<T,3> &,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3);
///< compute regularity of tetraedr (including 0,0,0),  FactEps * limit is used for too small coords
template <class T>  T  TetraReg (const cPtxd<T,3> &,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const T& FactEps=1e5);
///< Idem with4 points
template <class T>  T  TetraReg (const cPtxd<T,3> &,const cPtxd<T,3> &,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const T& FactEps=1e5);

/// Matrix corresponf to P ->  W ^ P
template<class T> cDenseMatrix<T> MatProdVect(const cPtxd<T,3>& aW);

// Return one vector orthog,  choice is not univoque , quikcly select on stable
template<class T> cPtxd<T,3>  VOrthog(const cPtxd<T,3> & aP);

template<class Type> cPtxd<Type,3> NormalUnit(const cTriangle<Type,3> &);  // V01 ^ V02
template<class Type> cPtxd<Type,3> Normal(const cTriangle<Type,3> &);  // V01 ^ V02
								       
template<class Type>  cTriangle<Type,3> RandomTriangRegul(Type aRegulMin,Type aAmpl);
template<class Type>  cTriangle<Type,3> RandomTetraTriangRegul(Type aRegulMin,Type aAmpl);

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
       static constexpr int       NbDOF = 3;
       static constexpr int       NbPtsMin = 2;  // != NbDOF/TheDim

       typedef cPtxd<Type,3>      tPt;
       typedef Type               tTypeElem;
       typedef cRotation3D<Type>  tTypeMap;
       typedef cRotation3D<Type>  tTypeMapInv;
       typedef std::vector<tPt>   tVPts;
       typedef const tVPts&       tCRVPts;
       typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac

       /// Create a "dummy" rotation, initialized with null matrix (to force problem if not init later)
       cRotation3D();

       /// RefineIt : if true, assume not fully orthog and compute closest one
       cRotation3D(const cDenseMatrix<Type> &,bool RefineIt);
       /// Create rotation from 3 vector I,J,K
       cRotation3D(const tPt &,const tPt &,const tPt &,bool RefineIt);

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

       /// Compute a normal repair, first vector being colinear to Pt (or second or last according to aNumP0)
       static cRotation3D<Type> CompleteRON(const tPt & aPt,int aNumP0=0);
       /// Compute a normal repair, first vector being colinear to P1, second in the plane P1,P2
       static cRotation3D<Type> CompleteRON(const tPt & aP0, const tPt & aP1, bool SVP=false); // SVP=return Identity if impossible
       /// Compute a rotation arround a given axe and with a given angle
       static cRotation3D<Type> RotFromAxe(const tPt & anAxe,Type aTeta);
       ///  Axiator close to Rot From but teta=Norm !!  exp(Mat(^Axe))
       static cRotation3D<Type> RotFromAxiator(const tPt & anAxe);
       /// Compute a random rotation for test/bench
       static cRotation3D<Type> RandomRot();
       /// Compute a "small" random rot controlled by ampl
       static cRotation3D<Type> RandomRot(const Type & aAmpl);
       /// create rotation from  string like "ijk" "i-kj" ... if sth like "ikj" => error !, so last is redundant but necessary
       static cRotation3D RotFromCanonicalAxes(const std::string&);

        // ================  RandomRot, but with standard names for standard group interface ==========

        /// return a random elem,  +or- with uniform density (whatever it means)
        static cRotation3D<Type> RandomElem();

        /// return a random small elem,  +or- with uniform density on tangent space (whatever it means)
        static cRotation3D<Type> RandomSmallElem(const Type & aAmpl);
        /// return a random rot with nomr in [V0,V1]
        static cRotation3D<Type> RandomInInterval(const Type & aV0,const Type & aV1);

        /// distance between 2 rotation; uses matrixes
        Type Dist(const  cRotation3D<Type> &) const;
        /// Max possible distance
        static Type MaxDist() {return 2.0 * std::sqrt(2);}

        /// this function must defined for differentiable group considered as variety
        static tTypeMap  Centroid(const std::vector<tTypeMap> & aV,const std::vector<Type> &);
        /// Average of 2 rotation
        tTypeMap  Centroid(const tTypeMap & aR2) const;

        /// Select the pose minimizing the sum of distance to other
        static tTypeMap  PseudoMediane(const std::vector<tTypeMap> & aV,int aSzProgr=-1);
        /// Select the pose minimizing the sum of distance to other in the interval K0,K1
        // static tTypeMap  PseudoMediane(const std::vector<tTypeMap> & aV,int aK0,int aK1);


        ///  Make a "robust" weighted average, starting from S0, W=[s0,A,B]-> 1/(1+R/s0^A)^B,defA=2, defB=1/A
        static tTypeMap  RobustAvg(const std::vector<tTypeMap> & aV,const tTypeMap & , const std::vector<tREAL8>& aWeight);
        ///  Make an iterate robust estimator
        static tTypeMap  RobustAvg
                         (    const std::vector<tTypeMap> & aV,tTypeMap  , const std::vector<tREAL8>& aWeight,
                              int aNbIterMin,
                              tREAL8 aDistStab = 1e30,
                              int aNbIterMax   = -1
                         );
        ///  Call robust avh with PseudoMediane initialization
        static tTypeMap  RobustMedAvg(const std::vector<tTypeMap> & aV,const std::vector<tREAL8> aWeight,int aNbIter,int aNbProg=-1);

       //  0-> arround I, 1->arround J ...
        static cRotation3D RotArroundKthAxe(int aNum);
       
       //// Compute a normal repair, first vector being colinear to P1, second in the plane P1,P2
      // static cRotation3D<Type> CompleteRON(const tPt & aP0,const tPt & aP1);

       /// Extract Axes of a rotation and compute its angle 
       void ExtractAxe(tPt & anAxe,Type & aTeta) const;
       /// More modern interface to ExtractAxe
       std::pair<tPt,Type> ExtractAxe() const;

       /// Convenient if you only need Angle, but slow else
       Type Angle() const;
       /// Convenient if you only need Axe, but slow else
       tPt    Axe() const;

       /// conversion to Omega Phi Kapa
       static cRotation3D<Type>  RotFromWPK(const tPt & aWPK);
       /// extrecat Omega Phi Kapa from rotation
       tPt                       ToWPK() const;

       /// Rotation arround X
       static cDenseMatrix<Type> RotOmega(const tREAL8 & aOmega);
       /// Rotation arround Y
       static cDenseMatrix<Type> RotPhi(const tREAL8 & aPhi);
       /// Rotation arround Z
       static cDenseMatrix<Type> RotKappa(const tREAL8 & aKappa);

       ///  0-> Omega   1->Phi  2-> Kappa
       static cDenseMatrix<Type> Rot1WPK(int aK,const tREAL8 & aOmega);

       // conversion to Yaw Pitch Roll
       static cRotation3D<Type>  RotFromYPR(const tPt & aWPK);
       tPt                       ToYPR() const;

    private :
       cDenseMatrix<Type>  mMat;
};

typedef cRotation3D<tREAL8> tRotR; 
void AddData(const  cAuxAr2007 & anAux,tRotR&);


/**  Class for 3D "affine" rotation of vector

*/

template <class Type> class cIsometry3D
{
    public :
       static constexpr int       TheDim=3;
       static constexpr int       NbDOF= 6;
       static constexpr int       NbPtsMin = 3;  // != NbDOF/TheDim


       typedef cPtxd<Type,3>      tPt;
       typedef cRotation3D<Type>  tRot;
       typedef cPtxd<Type,2>      tPt2;
       typedef cTriangle<Type,3>  tTri;
       typedef cTriangle<Type,2>  tTri2d;
       typedef Type               tTypeElem;
       typedef std::vector<tPt>   tVPts;
       typedef const tVPts&       tCRVPts;
       typedef std::vector<Type> tVVals;
       typedef const tVVals *    tCPVVals;

       typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac

       typedef cIsometry3D<Type> tTypeMap;
       typedef cIsometry3D<Type> tTypeMapInv;

       /// Default constructor is only provided for serialization, it initialize with dummy stuff
       cIsometry3D();

       cIsometry3D(const tPt& aTr,const tRot &);
       tTypeMapInv  MapInverse() const; // {return cIsometry3D(-mRot.Inverse(mTr),mRot.MapInverse());}
       tTypeMap  operator* (const tTypeMap &) const;
       static tTypeMap Identity();

       ///  Distance with normalisation to unity on center, W= weight of center dist vs rot
       Type DistPoseRel(const tTypeMap & aIsom2,const Type & aWTr) const;

       ///  Idem but dont normalize to unity
       Type DistPose(const tTypeMap & aIsom2,const Type & aWTr) const;

       ///  Estimate using ransac 
       static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);
      /// 
      tTypeMap LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals=nullptr)const;
      /// 
      tTypeMap StdGlobEstimate ( tCRVPts aVIn, tCRVPts aVOut, tTypeElem* aRes, tCPVVals   aVW, cParamCtrlOpt aParam);

       /// Estimate from 3 point , interface to "FromTriOut"  for  "RansacL1Estimate"
       static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);
      /// Basic   Value(aPIn) - aPOUt 
      tPt DiffInOut(const tPt & aPIn,const tPt & aPOUt) const;
      /// compute the vector used in least square equation
      static void ToEqParam(tPt & aRHS,std::vector<cDenseVect<Type>>&,const tPt &In,const tPt & Out);
      ///  evaluate from a vec [TrX,TrY,ScX,ScY], typycally result of mean square
      static tTypeMap  FromParam(const cDenseVect<Type> &);




       /// Return Isometrie with given Rot such I(PTin) = I(PTout)
       static cIsometry3D<Type> FromRotAndInOut(const tRot &,const tPt& aPtIn,const tPt& aPtOut );
       /// Return Isome such thqt I(InJ) = OutK ;  In(InJJp1) // OutKKp1 ; In(Norm0) = NormOut
       static cIsometry3D<Type> FromTriInAndOut(int aKIn, const tTri  & aTriIn, int aKOut, const tTri  & aTriOut, bool SVP=false); // SVP: do not crash if impossible, return Id
       /// Idem put use canonique tri = 0,I,J as input
       static cIsometry3D<Type> FromTriOut(int aKOut, const tTri  & aTriOut, bool Direct=true, bool SVP=false); // SVP: do not crash if impossible, return Id

       /// return a 2D triangle isometric to 3d, PK in 0,0  PK->PK1 // to Ox
       static tTri2d ToPlaneZ0(int aKOut,const tTri  & aTriOut,bool Direct=true);

       /// return a random isometry, amplt fix size of randomization for tr
       static cIsometry3D<Type> RandomIsom3D(const Type & AmplPt);

       /// this function must defined for differentiable group considered as variety
       static tTypeMap  Centroid(const std::vector<tTypeMap> & aV,const std::vector<Type> &);

       void SetRotation(const tRot &);

       const tRot & Rot() const {return mRot;}  ///< Accessor
       tRot & Rot() {return mRot;}  ///< Accessor
       const tPt &Tr() const {return mTr;}  ///< Accessor
       tPt &Tr() {return mTr;}  ///< Accessor

       tPt   Value(const tPt & aPt) const  {return mTr + mRot.Value(aPt);}
       tPt   Inverse(const tPt & aPt) const {return mRot.Inverse(aPt-mTr) ;}  // Work as M tM = Id

       cSimilitud3D<Type>  ToSimil() const; ///< make a similitude with scale 1

    private :
       tPt          mTr;
       tRot         mRot;
};
typedef cIsometry3D<tREAL8> tPoseR; 
void AddData(const cAuxAr2007 & anAux,tPoseR & aPose);


template <class Type> cRotation3D<tREAL8>  ToReal8(const cRotation3D<Type>  &);
template <class Type> cIsometry3D<tREAL8>  ToReal8(const cIsometry3D<Type>  &);

template <class Type> class cSimilitud3D
{
    public :
       static constexpr int       TheDim=3;
       static constexpr int       NbDOF=7;
       static constexpr int       NbPtsMin = 3;  // == NbDOF/TheDim

       typedef cPtxd<Type,3>      tPt;
       typedef cPtxd<Type,2>      tPt2;
       typedef cTriangle<Type,3>  tTri;
       typedef Type               tTypeElem;
       typedef std::vector<tPt>   tVPts;
       typedef const tVPts&       tCRVPts;
       typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac
       typedef cSimilitud3D<Type> tTypeMap;
       typedef cSimilitud3D<Type> tTypeMapInv;


       cSimilitud3D(const Type & aScale,const tPt& aTr,const cRotation3D<Type> &);
       tTypeMapInv  MapInverse() const; // {return cIsometry3D(-mRot.Inverse(mTr),mRot.MapInverse());}
       tTypeMap  operator* (const tTypeMap &) const;

       /// Return Similitud with given Rot such I(PTin) = I(PTout)
       static tTypeMap FromScaleRotAndInOut(const Type&,const cRotation3D<Type> &,const tPt& aPtIn,const tPt& aPtOut );
       /// Return Similitud such thqt I(InJ) = OutK ;  In(InJJp1) // OutKKp1 ; In(Norm0) = NormOut
       static tTypeMap FromTriInAndOut(int aKIn,const tTri  & aTriIn,int aKOut,const tTri  & aTriOut);
       /// Idem put use canonique tri = 0,I,J as input
       static tTypeMap FromTriOut(int aKOut,const tTri  & aTriOut);
       ///  Random similitud
       static tTypeMap RandomSim3D(Type aLevelScale,Type aLevelTr);
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
typedef cSimilitud3D<tREAL8>  tSim3dR;
template <class Type> cIsometry3D<Type>    TransfoPose(const cSimilitud3D<Type> & aSim,const cIsometry3D<Type> & aR);

/// V1 and V2 being pose "identic" but in different repair, compute the transfer similitude that best align
std::pair<tREAL8,tSim3dR>   EstimateSimTransfertFromPoses(const std::vector<tPoseR> & aV1,const std::vector<tPoseR> & aV2);



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

           cBox2dr  Box2D() const;

           void MakePatches(std::list<std::vector<int> > & ,tREAL8 aDistNeigh,tREAL8 aDistReject,int aSzMin) const;


        private :
           /// Read/Write in ply format using
           void PlyInit(const std::string &);
           void PlyWrite(const std::string &,bool isBinary) const;
};


class cPlane3D
{
     public :
	 /* ---------------------  Static constructor ---------------------------*/
	     /// construct with 1 point inside, and 2 vector inside
         static cPlane3D FromP0And2V(const cPt3dr & aP0,const cPt3dr& aAxeI , const cPt3dr& aAxeJ);
	     /// construct with 1 point inside, and the normal direction
         static cPlane3D FromPtAndNormal(const cPt3dr & aP0,const cPt3dr& aAxeK);
	     /// construct with 3 point inside
         static cPlane3D From3Point(const cPt3dr & aP0, const cPt3dr & aP1, const cPt3dr &aP2);

	 /// Estimate from a set of point,
         static std::pair<cPlane3D,tREAL8> RansacEstimate(const std::vector<cPt3dr> & aP0,bool AvgOrMax,int aNbTest=-1,tREAL8 aRegulMinTri =1e-3);
	 /// Return the indexes of the "best" plane
         static std::pair<cPt3di,tREAL8>  IndexRansacEstimate(const std::vector<cPt3dr> & aP0,bool AvgOrMax,int aNbTest=-1,tREAL8 aRegulMinTri =1e-3);

	 ///   Avegrage distance 
	 tREAL8 AvgDist(const std::vector<cPt3dr> &) const;
	 tREAL8 MaxDist(const std::vector<cPt3dr> &) const;

         cPt3dr  ToLocCoord(const cPt3dr &) const;
         cPt3dr  FromCoordLoc(const cPt3dr &) const;
         tREAL8  Dist(const cPt3dr &) const;
         cPt3dr  Inter(const cPt3dr&aP0,const cPt3dr&aP1) const;
         cPt3dr  Inter(const tSeg3dr& ) const;

         // return 3 point for random plane
         static std::vector<cPt3dr>  RandParam();

         const cPt3dr& P0() const; ///< Accessor
         const cPt3dr& AxeI() const; ///< Accessor
         const cPt3dr& AxeJ() const; ///< Accessor
         const cPt3dr& AxeK() const; ///< Accessor
	 
	 /** Return the intersection of the planes consider  as vector space , used eigen decomposition
	  * to get the best solutuob if Nb>2, if Sz=1 or 0 and aSzMin is Ok, return a random acceptable solution */
	 static cPt3dr DirInterPlane(const std::vector<const cPlane3D*>& aVPlanes,int aSzMin=2);
	 static cPt3dr DirInterPlane(const std::vector<cPlane3D>& aVPlanes,int aSzMin=2);

	 static tSeg3dr InterPlane(const std::vector<const cPlane3D*>& aVPlanes,int aSzMin=2,tREAL8 aWeithStab=1e-10);
	 static tSeg3dr InterPlane(const std::vector<cPlane3D>& aVPlanes,int aSzMin=2,tREAL8 aWeithStab=1e-10);
     private :
         cPlane3D(const cPt3dr & aP0,const cPt3dr& aAxeI , const cPt3dr& aAxeJ);
         cPt3dr mP0;
         cPt3dr mAxeI;
         cPt3dr mAxeJ;
         cPt3dr mAxeK;
};

/// Planarity index using ratio of eigen value 0/2 of moment matrix
tREAL8 L2_PlanarityIndex(const std::vector<cPt3dr> & aVPt);
/// Linearity index using ratio of eigen value 1/2 of moment matrix
tREAL8 L2_LinearityIndex(const std::vector<cPt3dr> & aVPt);



cPt3dr  BundleInters(const std::vector<tSeg3dr> & aVSeg,const std::vector<tREAL8> * aVWeight = nullptr);
///  Specialization for 2 lines, supposed to be faster,  Weight: 1.0 ->on line 1 , 0.0 -> one line 2
cPt3dr  BundleInters(const tSeg3dr & aSeg1,const tSeg3dr & aSeg2,tREAL8 aW12=0.5);
///  If we want to have the coeff of intersection
cPt3dr  BundleInters(cPt3dr & aCoeff,const tSeg3dr & aSeg1,const tSeg3dr & aSeg2,tREAL8 aW12=0.5);
///   Return point on bundle having given Z Value
cPt3dr  BundleFixZ(const tSeg3dr & aSeg1,const tREAL8 &);

///  Compute intersection on all pairs, and return the one minimizing sum of euclidian distances
cPt3dr  RobustBundleInters(const std::vector<tSeg3dr> & aVSeg);

/// Compute bundle intersection using a L1 criteria with barodale, "NbSegCompl" handle to be closer to euclidian distance
// cPt3dr  L1_BundleInters(const std::vector<tSeg3dr> & aVSeg,int NbSegCompl=0,const std::vector<tREAL8> * aVWeight = nullptr);

/// Compute isometry between two 3D points vectors. at least 3 points
tPoseR RobustIsometry(const std::vector<cPt3dr> & aPtsA, const std::vector<cPt3dr> & aPtsB);

/**  Class for sampling the space of quaternion/quaternion.  Method :
 *
 *     - we sample the sphere of R4
 *     - for this we sample the frontier of hypercube of R4 and normalize it lenghth, not uniform
 *     (impossible to realize by th way) but bounded anisoptropy
 *     - for this we sample regularly the  cube R3 and make the cartesian profuct with all the 8 (4*2)
 *    "principal" direction  (+- ijkt)
 *     - also we must be aware that Q and -Q correspond to same rotation, so if want to sample
 *     the space of rotation, which is generaly the case, we will take only one of both and
 *     explore + ijkt only
 *
 *    => possible amelioration make a non regular sampling in each direction ?
 *    => other amelioration  ???  Tabulate the result  with more sophisticated computation
 */


class cSampleQuat
{
      public :
         cSampleQuat(int aNbElem,bool ForRot);  ///< constructor indicating  number of step
         static cSampleQuat FromNbRot(int aNbRot,bool ForRot);  ///< constructor indicating the min number of rotation expected


         cPt4dr  KthQuat(int aK) const; ///< Main method return the number of rot
         size_t NbRot() const;  ///< Accessor
         size_t NbStep() const;  ///< Accessor

         //  compute vector, for NbTest random point, of  minimal distance to all sampled rot/quat
         std::vector<tREAL8 >  TestVecMinDist(size_t aNbTest) const;

         //  compute statistics, for NbTest random point, of  minimal distance to all sampled rot/quat
         cStdStatRes  TestStatMinDist(size_t aNbTest) const;

         // return the min distance bewteen all pair of sampled quat
         tREAL8 TestMinDistPairQuat() const;

	 // generate all the quaternions sampled
         std::vector<cPt4dr>  VecAllQuat() const;
      private :


         tREAL8 Int2Coord(int aK) const;  ///< convert on 1 direction [0,NbStep] => [-1,1]

         bool    m4R;     // is it for rotation
         size_t  mNbF;    // number of face
         size_t  mNbStep;
         size_t  mNbRot;
};

/// class for sampling regularly the hypercube, useful for sampling not so irregularly the hyper sphere
class cSampleHyperCube
{
   public :
        cSampleHyperCube(int aDim,int aNbStep,bool isProj = false);
        /// number of sampled points
        int NbSamples() const;
        ///  Main method return the Kth point
        void  KthPt(std::vector<tREAL8> & aPts,int aK) const;

   private :
        tREAL8 Int2Coord(int aK) const;  ///< convert on 1 direction [0,NbStep] => [-1,1]

        int  mDim;        // dimension of the cube
        int  mNbStep;     // memo nuber of step
        bool mIsProj;     // is it projective (i P ~ -P , and only of 2 is returned)
        int  mNbF;        // number of face
        int  mNbSamples;  // number of sample of the cube
};

///  Class for sampling point +- regularly on the sphere
class cSampleSphere3D
{
   public :
      cSampleSphere3D(int aNbStep);
      cPt3dr KthPt(int aK) const;
      int NbSamples() const;
   private :
      cSampleHyperCube mSHC;
};





};

#endif  //  _MMVII_GEOM3D_H_
