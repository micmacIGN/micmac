#ifndef  _MMVII_GEOM2D_H_
#define  _MMVII_GEOM2D_H_

#include "MMVII_Matrix.h"
#include "MMVII_Triangles.h"
#include "MMVII_ImageInfoExtract.h"


namespace MMVII
{


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

template <class T>   cPtxd<T,3> TP3z0  (const cPtxd<T,2> & aPt);
template <class T>   cPtxd<T,2> Proj   (const cPtxd<T,3> & aPt);
template <class T>   cTriangle<T,3> TP3z0  (const cTriangle<T,2> & aPt);
template <class T>   cTriangle<T,2> Proj   (const cTriangle<T,3> & aPt);


template <class T>  inline cPtxd<T,2> ToPolar(const cPtxd<T,2> & aP1)  ///<  From x,y to To rho,teta
{
   AssertNonNul(aP1);
   return  cPtxd<T,2>(std::hypot(aP1.x(),aP1.y()),std::atan2(aP1.y(),aP1.x()));
}
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

///  matrix of  linear function  q -> q * aP
template <class Type> cDenseMatrix<Type> MatOfMul (const cPtxd<Type,2> & aP);

template <class Type> class cSegment2DCompiled : public cSegmentCompiled<Type,2>
{
    public :
       typedef cPtxd<Type,2>   tPt;
       cSegment2DCompiled(const tPt& aP1,const tPt& aP2);
       tPt  ToCoordLoc(const tPt&) const;
       tPt  FromCoordLoc(const tPt&) const;
       Type  DistLine(const tPt&) const; ///< distance between the line and the point
    private :
       tPt     mNorm;
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
          static tTypeMap StdGlobEstimate(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals=nullptr);
          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);

	  /// Compute a random homotethy, assuring that Amplitude of scale has a minimal value
          static tTypeMap RandomHomotInv(const Type&AmplTr,const Type&AmplSc,const Type&AmplMinSc);

          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);

          cHomot2D(const tPt & aTr,const Type & aSc)  :
              mTr (aTr),
              mSc (aSc)
          {
          }
          cHomot2D() :  cHomot2D<Type>(tPt(0.0,0.0),1.0) {};
          inline tPt  Value(const tPt & aP) const   {return mTr + aP * mSc;}
          inline tPt  Inverse(const tPt & aP) const {return (aP-mTr)/mSc  ;}
          tTypeMapInv MapInverse() const {return cHomot2D<Type>(-mTr/mSc,1.0/mSc);}
	  tTypeMap operator *(const tTypeMap&aS2) const {return tTypeMap(mTr+mSc*aS2.mTr,mSc*aS2.mSc);}

          inline const tPt&  Tr() const   {return mTr;}
          inline Type        Sc() const   {return mSc;}
      private :
          tPt mTr;
          Type mSc;
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
          static tTypeMap StdGlobEstimate(tCRVPts & aVIn,tCRVPts& aVOut,Type * aRes2=nullptr,tCPVVals=nullptr);
          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);
	  /// Compute a random similitude, assuring that Amplitude of scale has a minimal value
          static cSim2D RandomSimInv(const Type&AmplTr,const Type&AmplSc,const Type&AmplMinSc);

          inline const tPt &  Tr() const {return mTr ;}
          inline const tPt &  Sc() const {return mSc ;}

                
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
                             tCPVVals=nullptr,
                             cParamCtrlOpt=cParamCtrlOpt::Default()
                          );
          /// compute with minimal number of samples
          static tTypeMap FromMinimalSamples(const tTabMin&,const tTabMin&);
          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);
          /// compute a quick estimate, assuming no outlayers, +or- generalization of FromMinimalSamples
          static tTypeMap QuickEstimate(tCRVPts aVIn,tCRVPts aVOut);
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
          static tTypeMap StdGlobEstimate(tCRVPts aVIn,tCRVPts aVOut,Type * aRes2=nullptr,tCPVVals=nullptr);

          /// compute by ransac the map minizing Sum |Map(VIn[K])-VOut[K]|
          static tTypeMap RansacL1Estimate(tCRVPts aVIn,tCRVPts aVOut,int aNbTest);

          /// compute the minimal resolution in all possible direction
          Type  MinResolution() const;

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
	   void  MakeDelaunay();
	public :
};


/**  Class for modelization of an ellipse */

class cEllipse
{
     public :
       /// Create from a vector of parameter ABCEF such elipse is definedby  :  Axx+2Bxy+Cyy+Dx+Fy=1
       cEllipse(cDenseVect<tREAL8> aDV,const cPt2dr & aC0);
       double ApproxSigneDist(cPt2dr aP) const;

       double SignedD2(cPt2dr aP) const;
       double Dist(const cPt2dr & aP) const;
       double   Norm() const  {return std::sqrt(1/ mNorm);}


       bool Ok() const;  ///< Accessor
       tREAL8 LGa() const;  ///< Accessor
       tREAL8 LSa() const;  ///< Accessor
       tREAL8 RayMoy() const;  ///< Accessor
       const cPt2dr &  Center() const; ///< Accessor
       double TetaGa() const; /// Teta great axe

       cPt2dr  PtOfTeta(tREAL8 aTeta,tREAL8 aMulRho=1.0) const; /// return on ellipse with param A cos(T) + B sin(T)
       cPt2dr  PtAndGradOfTeta(tREAL8 aTeta,cPt2dr &,tREAL8 aMulRho=1.0) const;  /// return also the gradien of belong function

    private :
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

class cEllipse_Estimate
{
//  A X2 + BXY + C Y2 + DX + EY = 1
      public :
        cLeasSqtAA<tREAL8> & Sys();

        // indicate a rough center, for better numerical accuracy
        cEllipse_Estimate(const cPt2dr & aC0);
        void AddPt(cPt2dr aP) ;

        cEllipse Compute() ;
        ~cEllipse_Estimate();
      private :
         cLeasSqtAA<tREAL8> *mSys;
         cPt2dr             mC0;

	 std::vector<cPt2dr>  mVObs;
};



/// Return random point that are not degenerated, +or- pertubation of unity roots
template <class Type> std::vector<cPtxd<Type,2> > RandomPtsOnCircle(int aNbPts);

// geometric   Flux of pixel

typedef std::vector<cPt2di> tResFlux;

void      GetPts_Circle(tResFlux & aRes,const cPt2dr & aC,double aRay,bool with8Neigh);
tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh);
void  GetPts_Ellipse(tResFlux & aRes,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh);
void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2);



struct cExtractedEllipse
{
     public :
        cSeedBWTarget    mSeed;
        cEllipse         mEllipse;

        cExtractedEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse);
        void  ShowOnFile(const std::string & aNameIm,int aZoom,const std::string& aPrefName) const; // make a accurate visu

        tREAL8               mDist;
        tREAL8               mDistPond;
        tREAL8               mEcartAng;
        bool                 mValidated;
        std::vector<cPt2dr>  mVFront;
};

class cExtract_BW_Ellipse  : public cExtract_BW_Target
{
        public :
             cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest);

             void AnalyseAllConnectedComponents(const std::string & aNameIm);
             bool AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm);

             const std::list<cExtractedEllipse> & ListExtEl() const;  ///< Accessor

        private :
             std::list<cExtractedEllipse> mListExtEl;
};



};

#endif  //  _MMVII_GEOM2D_H_
