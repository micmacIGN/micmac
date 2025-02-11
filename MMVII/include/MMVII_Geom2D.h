#ifndef  _MMVII_GEOM2D_H_
#define  _MMVII_GEOM2D_H_

#include "MMVII_Matrix.h"
#include "MMVII_Triangles.h"
#include "MMVII_ImageInfoExtract.h"


namespace MMVII
{

typedef cSegment<tREAL8,2> tSeg2dr;
typedef cSegmentCompiled<tREAL8,2> tSegComp2dr;



/** \file MMVII_Geom2D.h
    \brief contain classes for geometric manipulation, specific to 2D space :
           2D line, 2D plane, rotation, ...
*/

     // Complex and  polar function dedicatde
///   Complex multiplication 
template <class Type> inline  cPtxd<Type,2>  operator * (const  cPtxd<Type,2>  &aP1,const  cPtxd<Type,2>  & aP2)
{
   return  cPtxd<Type,2> (aP1.x()*aP2.x()-aP1.y()*aP2.y(),aP1.x()*aP2.y()+aP1.y()*aP2.x());
}

template <class Type> inline  cPtxd<Type,2>  conj (const  cPtxd<Type,2>  &aP1) {return cPtxd<Type,2>(aP1.x(),-aP1.y());}
template <class Type> inline  cPtxd<Type,2>  inv (const  cPtxd<Type,2>  &aP1) 
{  
   AssertNonNul(aP1); 
   return conj(aP1) / Type(SqN2(aP1));
}
template <class Type> inline  cPtxd<Type,2>   Rot90  (const cPtxd<Type,2> &aP) {return cPtxd<Type,2>(-aP.y(),aP.x());}
template <class Type> inline  cPtxd<Type,2> operator/(const cPtxd<Type,2> &aP1,const cPtxd<Type,2> & aP2) {return aP1*inv(aP2);}


template <class T>   T operator ^ (const cPtxd<T,2> & aP1,const cPtxd<T,2> & aP2)
{
    return aP1.x()*aP2.y()-aP1.y()*aP2.x();
}

template <class T>   cPtxd<T,3> TP3z  (const cPtxd<T,2> & aPt,const T&);
template <class T>   cPtxd<T,3> TP3z0  (const cPtxd<T,2> & aPt);
template <class T>   cPtxd<T,2> Proj   (const cPtxd<T,3> & aPt);
template <class T>   cTriangle<T,3> TP3z0  (const cTriangle<T,2> & aPt);
template <class T>   cTriangle<T,2> Proj   (const cTriangle<T,3> & aPt);


template <class T>  inline cPtxd<T,2> ToPolar(const cPtxd<T,2> & aP1)  ///<  From x,y to To rho,teta
{
   AssertNonNul(aP1);
   return  cPtxd<T,2>(std::hypot(aP1.x(),aP1.y()),std::atan2(aP1.y(),aP1.x()));
}
template <class T>  inline T Teta(const cPtxd<T,2> & aP1)  ///<  From x,y to To rho,teta
{
   AssertNonNul(aP1);
   return  std::atan2(aP1.y(),aP1.x());
}

/// return the "line" angle : i.e angle  between 2  non oriented direction, it's always in [0,PI/2] 
template <class T>  T LineAngles(const cPtxd<T,2> & aDir1,const cPtxd<T,2> & aDir2);


template <class T> inline cPtxd<T,2> ToPolar(const cPtxd<T,2> & aP1,T aDefTeta)  ///<  With Def value 4 teta
{
    return IsNotNull(aP1) ? ToPolar(aP1) : cPtxd<T,2>(0,aDefTeta);
}
template <class T> inline cPtxd<T,2> FromPolar(const T & aRho,const T & aTeta)
{
    return cPtxd<T,2>(aRho*cos(aTeta),aRho*sin(aTeta));
}
template <class T> inline cPtxd<T,2> FromPolar(const cPtxd<T,2> & aP)
{
    return FromPolar(aP.x(),aP.y());
}

template <class Type> inline cPtxd<Type,2> PSymXY (const cPtxd<Type,2> & aP) 
{ 
    return cPtxd<Type,2>(aP.y(),aP.x()); 
}

/// return twice the oriented area of the polygon
template <class Type> Type DbleAreaPolygOriented(const std::vector<cPtxd<Type,2>> &aPolyg);

///  matrix of  linear function  q -> q * aP
template <class Type> cDenseMatrix<Type> MatOfMul (const cPtxd<Type,2> & aP);

/**  This specialization is specific to dim 2, as the normal to a vector is 
 * specific to d2
 */
template <class Type> class cSegment2DCompiled : public cSegmentCompiled<Type,2>
{
    public :
       typedef cPtxd<Type,2>   tPt;
       cSegment2DCompiled(const tPt& aP1,const tPt& aP2);
       cSegment2DCompiled(const cSegment<Type,2>&);
       tPt  ToCoordLoc(const tPt&) const;
       tPt  FromCoordLoc(const tPt&) const;
       Type  DistLine(const tPt&) const; ///< distance between the line and the point
       Type  DistClosedSeg(const tPt&) const; ///< distance between the point and closed segment
       Type  SignedDist(const tPt& aPt) const; ///< Signed dist to the line (= y of local coordinates)
       Type  Dist(const tPt& aPt) const; ///< Faster than upper class
       const tPt & Normal() const {return mNorm;}


       tPt InterSeg(const cSegment2DCompiled<Type> &,tREAL8 aMinAngle=1e-5,bool *IsOk=nullptr);
    private :
       tPt     mNorm;
};

/** this class a represent a "closed" segment , it has same data than cSegment2DCompiled,
 * but as a set/geometric primitive, it is limited by extremities
 */

class cClosedSeg2D
{
   public :
      bool  InfEqDist(const cPt2dr & aPt,tREAL8 aDist) const;
      cClosedSeg2D(const cPt2dr & aP0,const cPt2dr & aP1);
      cBox2dr GetBoxEnglob() const;

      const cSegment2DCompiled<tREAL8> & Seg() const;
   private :
      cSegment2DCompiled<tREAL8>  mSeg;
};



/*  Class of 2D mapping having the same interface, usable for ransac & least square */
template <class Type>  class cHomot2D;
template <class Type>  class cSim2D;
template <class Type>  class cRot2D;
template <class Type>  class cAffin2D;
template <class TypeMap> class  cLeastSquareEstimate;


/** This class represent 2D Homotetie , it can aussi be used for an non
   distorted camera with :
       * mTr -> principal point
       * mSc -> focale
*/

template <class Type>  class cHomot2D
{
      public :
          static constexpr int TheDim=2;
          static constexpr int NbDOF = 3;
          static std::string Name() {return "Homot2D";}
          static constexpr int NbPtsMin = DIV_SUP(NbDOF,TheDim);

          typedef Type  tTypeElem;
          typedef cHomot2D<Type>  tTypeMap;
          typedef cHomot2D<Type>  tTypeMapInv;

          typedef cPtxd<Type,2>     tPt;
          typedef std::vector<tPt>  tVPts;
          typedef const tVPts&      tCRVPts;
          typedef std::vector<Type> tVVals;
          typedef const tVVals *    tCPVVals;
          typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac

          ///  evaluate from a vec [TrX,TrY,ScX,ScY], typycally result of mean square
          static tTypeMap  FromParam(const cDenseVect<Type> &);  
          /// compute the vector used in least square equation
          static void ToEqParam(tPt & aRHS,cDenseVect<Type>&,cDenseVect<Type> &,const tPt &In,const tPt & Out);
          /// compute by least square the mapping such that Hom(PIn[aK]) = POut[aK]
          static tTypeMap StdGlobEstimate(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals aVWeight=nullptr);
          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);

	  /// Compute a random homotethy, assuring that Amplitude of scale has a minimal value
          static tTypeMap RandomHomotInv(const Type&AmplTr,const Type&AmplSc,const Type&AmplMinSc);

          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);

          cHomot2D(const tPt & aTr,const Type & aSc)  :
              mSc (aSc),
              mTr (aTr)
          {
          }
          cHomot2D() :  cHomot2D<Type>(tPt(0.0,0.0),1.0) {};
          inline tPt  Value(const tPt & aP) const   {return mTr + aP * mSc;}
          inline tPt  Inverse(const tPt & aP) const {return (aP-mTr)/mSc  ;}
          tTypeMapInv MapInverse() const {return cHomot2D<Type>(-mTr/mSc,1.0/mSc);}
	  tTypeMap operator *(const tTypeMap&aS2) const {return tTypeMap(mTr+mSc*aS2.mTr,mSc*aS2.mSc);}

          inline const tPt&     Tr() const   {return mTr;}
          inline const Type &   Sc() const   {return mSc;}
          inline tPt&     Tr() {return mTr;}
          inline Type &   Sc() {return mSc;}
          /// Basic   Value(aPIn) - aPOUt 
          tPt DiffInOut(const tPt & aPIn,const tPt & aPOUt) const;
          /// Basic   1.0
          Type Divisor(const tPt & aPInt) const;

      private :
          Type mSc;
          tPt mTr;
};

/** Usefull when we want to visualize objects : compute box of visu + Mapping Visu/Init */
cBox2di BoxAndCorresp(cHomot2D<tREAL8> & aHomIn2Image,const cBox2dr & aBox,int aSzIm,int aMargeImage);

/** Class for a similitude 2D  P ->  Tr + Sc * P

       * Tr is the translation
       * Sc is the both homthethy and rotation as is used the complex number for point multiplication
*/

template <class Type>  class cSim2D
{
      public :
          static constexpr int TheDim=2;
          static constexpr int  NbDOF = 4;
          static std::string Name() {return "Sim2D";}
          static constexpr int  NbPtsMin = DIV_SUP(NbDOF,TheDim);

          typedef Type          tTypeElem;
          typedef cSim2D<Type>  tTypeMap;
          typedef cSim2D<Type>  tTypeMapInv;

          typedef cPtxd<Type,2> tPt;
          typedef std::vector<tPt>  tVPts;
          typedef const tVPts&      tCRVPts;
          typedef std::vector<Type> tVVals;
          typedef const tVVals *    tCPVVals;
          typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac

          cSim2D(const tPt & aTr,const tPt & aSc)  :
              mTr (aTr),
              mSc (aSc)
          {
          }
          cSim2D() : cSim2D<Type>(tPt(0,0),tPt(1,0)) {}
          
          ///  evaluate from a vec [TrX,TrY,ScX,ScY], typycally result of mean square
          static tTypeMap  FromParam(const cDenseVect<Type> &);  
          /// compute the vectors and constants used in least square equation
          static void ToEqParam(tPt& aRHS,cDenseVect<Type>&,cDenseVect<Type> &,const tPt & aPtIn,const tPt & aPtOut);
          /// Degree of freedoom


          inline tPt  Value(const tPt & aP) const {return mTr + aP * mSc;}
          inline tPt  Inverse(const tPt & aP) const {return (aP-mTr)/mSc  ;}
          tTypeMapInv  MapInverse() const {return cSim2D<Type>(-mTr/mSc,tPt(1.0,0.0)/mSc);}
	  tTypeMap operator *(const tTypeMap&aS2) const {return tTypeMap(mTr+mSc*aS2.mTr,mSc*aS2.mSc);}
          

          static tTypeMap FromMinimalSamples(const tPt&aP0In,const tPt&aP1In,const tPt&aP0Out,const tPt&aP1Out);
          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);

          /// compute by least square the mapping such that Sim(PIn[aK]) = POut[aK]
          static tTypeMap StdGlobEstimate(tCRVPts & aVIn,tCRVPts& aVOut,Type * aRes2=nullptr,tCPVVals aVWeight=nullptr);
          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);
	  /// Compute a random similitude, assuring that Amplitude of scale has a minimal value
          static cSim2D RandomSimInv(const Type&AmplTr,const Type&AmplSc,const Type&AmplMinSc);

          inline const tPt &  Tr() const {return mTr ;}
          inline const tPt &  Sc() const {return mSc ;}

          /// Basic   Value(aPIn) - aPOUt 
          tPt DiffInOut(const tPt & aPIn,const tPt & aPOUt) const;
          /// Basic   1.0
          Type Divisor(const tPt & aPInt) const;
                
	  ///  Generate the 3D-Sim having same impact in the plane X,Y
	  cSimilitud3D<Type> Ext3D() const;
      private :
          tPt mTr;
          tPt mSc;
};

/** Class for a rotation  2D , implemeted using similitude with ||Sc|| = 1 */

template <class Type>  class cRot2D
{
      public :
          static constexpr int TheDim=2;
          static constexpr int  NbDOF = 3;
          static std::string Name() {return "Rot2D";}
          static constexpr int  NbPtsMin = DIV_SUP(NbDOF,TheDim);

          typedef Type          tTypeElem;
          typedef cRot2D<Type>  tTypeMap;
          typedef cRot2D<Type>  tTypeMapInv;

          typedef cPtxd<Type,2>     tPt;
          typedef std::vector<tPt>  tVPts;
          typedef const tVPts&      tCRVPts;
          typedef std::vector<Type> tVVals;
          typedef const tVVals *    tCPVVals;
          typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac


          cRot2D(const tPt & aTr,const Type & aTeta)  :
              mTeta (aTeta),
              mSim  (aTr,FromPolar(Type(1.0),aTeta))
          {
          }
          cRot2D() : cRot2D<Type> (tPt(0,0),0) {}
          


          inline tPt  Value(const tPt & aP) const {return mSim.Value(aP);}
          inline tPt  Inverse(const tPt & aP) const {return mSim.Inverse(aP)  ;}
          tTypeMapInv  MapInverse() const {return cRot2D<Type>(-Tr()/Sc(),-mTeta);}
	  tTypeMap operator *(const tTypeMap&aS2) const {return tTypeMap(Tr()+Sc()*aS2.Tr(),mTeta+aS2.mTeta);}

          inline const tPt&  Tr() const {return mSim.Tr() ;}
          inline const tPt&  Sc() const {return mSim.Sc() ;}
          inline const Type& Teta() const {return mTeta ;}
          inline const cSim2D<Type> & Sim() const {return mSim;}

          static tTypeMap RandomRot(const Type&AmplTr);


          ///  evaluate from a vec [TrX,TrY,ScX,ScY], typycally result of mean square
          static tTypeMap  FromParam(const cDenseVect<Type> &);  
          /// compute the vectors and constants used in least square equation
          static void ToEqParam(tPt& aRHS,cDenseVect<Type>&,cDenseVect<Type> &,const tPt & aPtIn,const tPt & aPtOut);

          /// Refine an existing solution using least square
          tTypeMap LeastSquareRefine(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals=nullptr)const;
          /// compute least square , after a ransac init, the mapping such that Sim(PIn[aK]) = POut[aK]
          static tTypeMap StdGlobEstimate 
                          (
                             tCRVPts aVIn,
                             tCRVPts aVOut,
                             Type * aRes2=nullptr,
                             tCPVVals aVWeight=nullptr,
                             cParamCtrlOpt=cParamCtrlOpt::Default()
                          );
          /// compute with minimal number of samples
          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);
          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);
          /// compute a quick estimate, assuming no outlayers, +or- generalization of FromMinimalSamples
          static tTypeMap QuickEstimate(tCRVPts aVIn,tCRVPts aVOut);
          /// Basic   Value(aPIn) - aPOUt 
          tPt DiffInOut(const tPt & aPIn,const tPt & aPOUt) const;
          /// Basic   1.0
          Type Divisor(const tPt & aPInt) const;
      private :
          Type          mTeta;
          cSim2D<Type>  mSim;
};



template <class Type>  class cAffin2D
{
      public :
          static constexpr int    TheDim= 2;
          static constexpr int    NbDOF = 6;
          static std::string Name() {return "Affin2D";}
          static constexpr int    NbPtsMin = DIV_SUP(NbDOF,TheDim);

          typedef Type            tTypeElem;
          typedef cAffin2D<Type>  tTypeMap;
          typedef cAffin2D<Type>  tTypeMapInv;

          typedef cPtxd<Type,2> tPt;
          typedef std::vector<tPt>  tVPts;
          typedef const tVPts&      tCRVPts;
          typedef std::vector<Type> tVVals;
          typedef const tVVals *    tCPVVals;
          typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac

          typedef cTriangle<Type,2>  tTri;

          cAffin2D(const tPt & aTr,const tPt & aImX,const tPt aImY) ; 
          cAffin2D();
          tPt  Value(const tPt & aP) const ;
          tPt  Inverse(const tPt & aP) const ;
          tTypeMapInv  MapInverse() const ;

	  // ========== Accesors =================
	  const Type &  Delta() const;  ///<  Accessor
	  const tPt &   VX() const;  ///<  Accessor
	  const tPt &   VY() const;  ///<  Accessor
	  const tPt &   Tr() const;  ///<  Accessor
	  const tPt &   VInvX() const;  ///<  Accessor
	  const tPt &   VInvY() const;  ///<  Accessor

	  tTypeMap operator *(const tTypeMap&) const;
          tPt  VecInverse(const tPt & aP) const ;
          tPt  VecValue(const tPt & aP) const ;
	  
	  //  allocator static
          static  tTypeMap  AllocRandom(const Type & aDeltaMin);
          static  tTypeMap  Translation(const tPt  & aTr);
          static  tTypeMap  Rotation(const Type & aScale);
          static  tTypeMap  Homot(const Type & aScale);
          static  tTypeMap  HomotXY(const Type & aScaleX,const Type & aScaleY);
                
          ///  evaluate from a vec [TrX,TrY,ScX,ScY], typycally result of mean square
          static tTypeMap  FromParam(const cDenseVect<Type> &);  
          /// compute the vectors and constants used in least square equation
          static void ToEqParam(tPt& aRHS,cDenseVect<Type>&,cDenseVect<Type> &,const tPt & aPtIn,const tPt & aPtOut);
          /// compute with minimal number of samples
          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);
          /// Affity transforming a triangle in another ~ FromMinimalSamples, just interface
          static tTypeMap Tri2Tri(const tTri& aTriIn,const tTri& aTriOut);

          /// compute by least square the mapping such that Hom(PIn[aK]) = POut[aK]
          static tTypeMap StdGlobEstimate(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals aVWeight=nullptr);

          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);

          /// compute the minimal resolution in all possible direction
          Type  MinResolution() const;
          /// Basic   Value(aPIn) - aPOUt 
          tPt DiffInOut(const tPt & aPIn,const tPt & aPOUt) const;
          /// Basic   1.0
          Type Divisor(const tPt & aPInt) const;

      private :
          tPt   mTr;
          tPt   mVX;
          tPt   mVY;
	  Type  mDelta;
          tPt   mVInvX;
          tPt   mVInvY;
};
typedef  cAffin2D<tREAL8>  cAff2D_r;
cBox2dr  ImageOfBox(const cAff2D_r & aAff,const cBox2dr & aBox);


template <class Type>  class cHomogr2D
{
      public :
          static constexpr int TheDim=2;
          static constexpr int NbDOF = 8;
          static std::string Name() {return "Homogr2D";}
          static constexpr int NbPtsMin = DIV_SUP(NbDOF,TheDim);

          typedef Type                 tTypeElem;
          typedef cHomogr2D<Type>      tTypeMap;
          typedef cHomogr2D<Type>      tTypeMapInv;
          // typedef cElemHomogr2D<Type>  tElemH;
          typedef cPtxd<Type,3>     tElemH;

          typedef cPtxd<Type,2>     tPt;
          typedef std::vector<tPt>  tVPts;
          typedef const tVPts&      tCRVPts;
          typedef std::vector<Type> tVVals;
          typedef const tVVals *    tCPVVals;
          typedef tPt   tTabMin[NbPtsMin];  // Used for estimate with min number of point=> for ransac
          ///  evaluate from a vec [TrX,TrY,ScX,ScY], typycally result of mean square
          static tTypeMap  FromParam(const cDenseVect<Type> &);

//==================
          cHomogr2D(const tElemH & aHX,const tElemH & aHY,const tElemH & aHZ);
          cHomogr2D() ;

          cDenseMatrix<Type>  Mat() const;
          static tTypeMap  FromMat(const cDenseMatrix<Type> &);

          tTypeMap operator *(const tTypeMap&aS2) const ;
          tTypeMapInv MapInverse() const ;

	  tTypeElem  AvgDistL1(tCRVPts aVIn,tCRVPts aVOut);

          inline tPt  Value(const tPt & aP) const   {return tPt(S(mHX,aP),S(mHY,aP)) / S(mHZ,aP);}
          inline tPt  Inverse(const tPt & aP) const {return tPt(S(mIHX,aP),S(mIHY,aP)) / S(mIHZ,aP);}
          /// compute the vector used in least square equation
          static void ToEqParam(tPt & aRHS,cDenseVect<Type>&,cDenseVect<Type> &,const tPt &In,const tPt & Out);
          /// Creat an homotethy from 4 example
          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);

          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);
          /// compute by least square the mapping such that Hom(PIn[aK]) = POut[aK]
          static tTypeMap StdGlobEstimate(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals aVWeight=nullptr);

	  /// compute the homography, assuming we know it up to a shift of Z (devlopped in CERN pannel context)
          tTypeMap LeastSqParalPlaneShift(tCRVPts aVIn,tCRVPts aVOut) const;
	  /// call LeastSqParalPlaneShift for a robust approach
          tTypeMap RansacParalPlaneShift(tCRVPts aVIn,tCRVPts aVOut,int aNbMin=2,int aNbMax=2) const;

          const tElemH &  Hx() const {return mHX;}
          const tElemH &  Hy() const {return mHY;}
          const tElemH &  Hz() const {return mHZ;}
          tElemH &  Hx()  {return mHX;}
          tElemH &  Hy()  {return mHY;}
          tElemH &  Hz()  {return mHZ;}

          Type S(const tElemH & aH,const tPt & aP) const {return aH.x()*aP.x() + aH.y()*aP.y() + aH.z();}

          ///   Amplitue of the square where inversible
          static  tTypeMap  AllocRandom(const Type & aAmpl);

          /// !! NON BASIC !! ~   (Value(aPIn) - aPOUt) * Hz
          tPt DiffInOut(const tPt & aPIn,const tPt & aPOUt) const;
          /// !! NON BASIC !! ~   S( Hz)
          Type Divisor(const tPt & aPInt) const;
      private :

          tElemH  mHX;
          tElemH  mHY;
          tElemH  mHZ;

          tElemH  mIHX;
          tElemH  mIHY;
          tElemH  mIHZ;
};




//template <class Type,class TMap>  cTplBox<2,Type>  ImageOfBox();

/// Image of an tri by a mapping
template <class Type,class tMap>  cTriangle<Type,2>  ImageOfTri(const cTriangle<Type,2> &,const tMap &);



template <class Type> class  cTriangle2DCompiled : public cTriangle<Type,2>
{
       public :
           typedef cPtxd<Type,2>      tPt;
           typedef cPtxd<Type,3>      t3Val;
           typedef cTriangle<Type,2>  tTri;

           cTriangle2DCompiled(const tTri & aTri);
           cTriangle2DCompiled(const tPt & aP0,const tPt & aP1,const tPt & aP2);

           bool  Regular() const;  ///<  Non degenerate i.e  delta !=0
           t3Val  CoordBarry(const     tPt & aP) const; ///< Barrycentric coordinates
           Type ValueInterpol(const   tPt & aP,const t3Val & aValues) const;  ///< Interpolated value
           tPt GradientVI(const t3Val& aValues) const;  ///< Gradient of Interpolated value

           static cTriangle2DCompiled<Type> RandomRegularTri(Type aSz,Type aEps=Type(1e-3));

           Type Insideness(const tPt &) const; // <0 out, > inside, 0 : frontier
           bool   Insides(const tPt &,Type aTol=0.0) const; // Tol<0 give more points

	   /// generate all the pixels inside a triangle, aVWeight to get barrycentric weighting, Tol to have all pixels
           void PixelsInside(std::vector<cPt2di> & aRes,double aTol=-1e-5,std::vector<t3Val> * aVWeight = nullptr) const;

	   Type Delta() const {return mDelta;}

       private :
           void  AssertRegular() const;  //  Non degenerate i.e  delta !=0
           /*  
              For barycentrique coord, we have :
	      {L1 = (CX1  CY1)}   (X1-X0   X2-X0)  =  (1  0)
	      {L2 = (CX2  CY2)} * (Y1-Y0   Y2-Y0)     (0  1)
           */
           Type  mDelta;
           tPt   mL1;
           tPt   mL2;
};

// std::pair<cTriangle2D,cPt3dr> Mqk=////


template<class Type> class cTriangulation2D : public cTriangulation<Type,2>
{
	public :
           typedef cPtxd<Type,2>      tPt;

           cTriangulation2D(const std::vector<tPt>&);
	   /// create by flatening to z=0 the points
           cTriangulation2D(const cTriangulation<Type,3>&);
           cTriangulation2D(const std::string &);
           void WriteFile(const std::string &,bool isBinary) const;

       void  MakeDelaunay();

    private :
       void PlyWrite(const std::string &,bool isBinary) const;

};


/**  Class for modelization of an ellipse */

class cEllipse
{
     public :
       static void BenchEllispe();

       /// Create from a vector of parameter ABCEF such elipse is definedby  :  Axx+2Bxy+Cyy+Dx+Fy=1
       cEllipse(cDenseVect<tREAL8> aDV,const cPt2dr & aC0);
       ///  A more physicall creation
       cEllipse(const cPt2dr & aCenter,tREAL8 aTeta,tREAL8 aLGa,tREAL8 aLSa);
       /// Create a circle
       cEllipse (const cPt2dr & aCenter,tREAL8 aRay);

       void AddData(const  cAuxAr2007 & anAux);


       double NonEuclidDist(const cPt2dr& aP) const;  /// Dist to non euclid proj (on radius)
       double EuclidDist(const cPt2dr& aP) const;  /// rigourous  distance, use projection (long ?)
       double SignedEuclidDist(const cPt2dr& aP) const;  /// rigourous signed distance

       double ApproxSigneDist(const cPt2dr& aP) const;
       double ApproxDist(const cPt2dr& aP) const;

       double SignedQF_D2(const cPt2dr& aP) const;  ///  computed frm quadratic form , in D2 at infty
       double QF_Dist(const cPt2dr & aP) const;     ///  computed frm quadratic form ,  in sqrt(D) at 0

       double   Norm() const  {return std::sqrt(1/ mNorm);}
       bool Ok() const;  ///< Accessor
       tREAL8 LGa() const;  ///< Accessor
       tREAL8 LSa() const;  ///< Accessor
       tREAL8 RayMoy() const;  ///< Accessor
       const cPt2dr &  Center() const; ///< Accessor
       const cPt2dr &  VGa() const; ///< Accessor
       const cPt2dr &  VSa() const; ///< Accessor
       double TetaGa() const; /// Teta great axe
       tREAL8  EVP() const ;  /// Are Eigen value positive


       cPt2dr  PtOfTeta(tREAL8 aTeta,tREAL8 aMulRho=1.0) const; /// return on ellipse with param A cos(T) + B sin(T)
       cPt2dr  PtAndGradOfTeta(tREAL8 aTeta,cPt2dr &,tREAL8 aMulRho=1.0) const;  /// return also the gradien of belong function

       cPt2dr  ToCoordLoc(const cPt2dr &) const; /// in a sys when ellipse is unity circle
       cPt2dr  VectToCoordLoc(const cPt2dr &) const; ///  for vector (dont use center)
       cPt2dr  FromCoordLoc(const cPt2dr &) const; /// in a sys when ellipse is unity circle
       cPt2dr  VectFromCoordLoc(const cPt2dr &) const; /// for vector (dont use center)in a sys when ellipse is unity circle
       cPt2dr  ToRhoTeta(const cPt2dr &) const; /// Invert function of PtOfTeta

       cPt2dr  ProjOnEllipse(const cPt2dr &) const;
       cPt2dr  ProjNonEuclOnEllipse(const cPt2dr &) const;   // project with ellispe norm

       cPt2dr  Tgt(const cPt2dr &) const;
       cPt2dr  NormalInt(const cPt2dr &) const;

       cPt2dr InterSemiLine(tREAL8 aTeta) const;    /// compute the intesection of 1/2 line of direction teta with the ellipse

       /// get points on ellipse that are +- less regularly sampled at a given step
       void GetTetasRegularSample(std::vector<tREAL8> & aVTetas,const tREAL8 & aDist);


    private :
       inline void AssertOk() const;

       void OneBenchEllispe();
       cDenseVect<tREAL8>     mV;
       double                 mNorm;
       cPt2dr                 mC0;
       cDenseMatrix<tREAL8>   mQF; // Matrix of quadratic form
       cPt2dr                 mCenter;
       double                 mCste;
       double                 mRRR;
       bool                   mOk;
       tREAL8                 mLGa;  ///< Length Great Axe
       cPt2dr                 mVGa;  ///< Vector Great Axe
       tREAL8                 mLSa;  ///< Lenght Small Axe
       cPt2dr                 mVSa;  ///< Vector Great Axe
       tREAL8                 mRayMoy;
       tREAL8                 mSqRatio;
};
void AddData(const  cAuxAr2007 & anAux,cEllipse &);

class cEllipse_Estimate
{
//  A X2 + BXY + C Y2 + DX + EY = 1
      public :
        cLeasSqtAA<tREAL8> & Sys();

        // indicate a rough center, for better numerical accuracy
        cEllipse_Estimate(const cPt2dr & aC0,bool isCenterFree=true,bool isCircle=false);
        void AddPt(cPt2dr aP) ;

        cEllipse Compute() ;
        ~cEllipse_Estimate();
      private :
	 bool               mIsCenterFree;
	 bool               mIsCircle;
         cLeasSqtAA<tREAL8> *mSys;
         cPt2dr             mC0;

	 std::vector<cPt2dr>  mVObs;
};



/// Return random point that are not degenerated, +or- pertubation of unity roots
template <class Type> std::vector<cPtxd<Type,2> > RandomPtsOnCircle(int aNbPts);
/// Specialze for homogr, avoid singlularity in [-1,1]^2 
template <class Type> std::pair<std::vector<cPtxd<Type,2> >,std::vector<cPtxd<Type,2>>> RandomPtsHomgr(Type aAmpl=1.5);
/// Point regularly positionned on circle, to generate mapping close to Identity 
template <class Type> std::pair<std::vector<cPtxd<Type,2> >,std::vector<cPtxd<Type,2>>>  RandomPtsId(int aNb,Type aEpsId);
/// generate a map, "close to Id", using RandomPtsId
template <class tMap>  tMap RandomMapId(typename tMap::tTypeElem aEpsId);





// geometric   Flux of pixel

typedef std::vector<cPt2di> tResFlux;

void      GetPts_Circle(tResFlux & aRes,const cPt2dr & aC,double aRay,bool with8Neigh);
tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh);

void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2);
void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2,tREAL8 Width);

void  GetPts_Ellipse(tResFlux & aRes,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh,tREAL8 aDilate);
void  GetPts_Ellipse(tResFlux & aRes,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh);


/**  Class to store the extraction of an ellipse, contain the seed-point + geometric ellipse itself +
 * different quality indicator
 */

struct cExtractedEllipse
{
     public :
        cSeedBWTarget    mSeed;
        cEllipse         mEllipse;

        cExtractedEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse);
        void  ShowOnFile(const std::string & aNameIm,int aZoom,const std::string& aPrefName) const; // make a accurate visu

        tREAL8               mDist; /// Dist of frontier point to ellispse
        tREAL8               mDistPond; /// Dist attenuated "very empirically" by size of ellipse
        tREAL8               mEcartAng;  /// Angular diff between image gradient an theoreticall ellipse normal
        bool                 mValidated;  /// Is the ellipse validated
        std::vector<cPt2dr>  mVFront;
};

/** Class for extracting B/W ellipse, herits from B/W target for component analysis and add ellipse recognition */

class cExtract_BW_Ellipse  : public cExtract_BW_Target
{
        public :
             cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest);

             void AnalyseAllConnectedComponents(const std::string & aNameIm);
             bool AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm);

             const std::list<cExtractedEllipse> & ListExtEl() const;  ///< Accessor

             void   ComputeBlurr();  /// experimental, to review later, maybe ...
        private :
             std::list<cExtractedEllipse> mListExtEl;
};

class cCircTargExtr;

/** Minimal struct to save the result of an ellipse extracted in image */

struct cSaveExtrEllipe
{
     public :
          cSaveExtrEllipe (const cCircTargExtr &,const std::string & aNameCode);
          cSaveExtrEllipe ();
          static std::string NameFile(const cPhotogrammetricProject & ,const cSetMesPtOf1Im &,bool Input);

          cEllipse  mEllipse;
          std::string mNameCode;
          tREAL4 mBlack;
          tREAL4 mWhite;
};
void AddData(const  cAuxAr2007 & anAux, cSaveExtrEllipe & aCTE);



};

#endif  //  _MMVII_GEOM2D_H_
