#ifndef  _MMVII_GEOM2D_H_
#define  _MMVII_GEOM2D_H_

namespace MMVII
{


/** \file MMVII_Geom2D.h
    \brief contain classes for geometric manipulation, specific to 2D space :
           2D line, 2D plane, rotation, ...
*/

     // Complex and  polar function dedicatde
///   Complex multiplication 
inline cPt2dr operator * (const cPt2dr &aP1,const cPt2dr & aP2)
{
   return cPt2dr(aP1.x()*aP2.x()-aP1.y()*aP2.y(),aP1.x()*aP2.y()+aP1.y()*aP2.x());
}
inline cPt2dr conj  (const cPt2dr &aP1) {return cPt2dr(aP1.x(),-aP1.y());}
inline cPt2dr inv   (const cPt2dr &aP1)
{  
   AssertNonNul(aP1); 
   return conj(aP1) / SqN2(aP1);
}
inline cPt2dr Rot90  (const cPt2dr &aP) {return cPt2dr(-aP.y(),aP.x());}

inline cPt2dr operator / (const cPt2dr &aP1,const cPt2dr & aP2) {return aP1 * inv(aP2);}


template <class T>   T operator ^ (const cPtxd<T,2> & aP1,const cPtxd<T,2> & aP2)
{
    return aP1.x()*aP2.y()-aP1.y()*aP2.x();
}


inline cPt2dr ToPolar(const cPt2dr & aP1)  ///<  From x,y to To rho,teta
{
   AssertNonNul(aP1);
   return  cPt2dr(hypot(aP1.x(),aP1.y()),atan2(aP1.y(),aP1.x()));
}

inline cPt2dr ToPolar(const cPt2dr & aP1,double aDefTeta)  ///<  With Def value 4 teta
{
    return IsNotNull(aP1) ? ToPolar(aP1) : cPt2dr(0,aDefTeta);
}
inline cPt2dr FromPolar(const double & aRho,const double & aTeta)
{
    return cPt2dr(aRho*cos(aTeta),aRho*sin(aTeta));
}
inline cPt2dr FromPolar(const cPt2dr & aP)
{
    return FromPolar(aP.x(),aP.y());
}

template <class Type> inline cPtxd<Type,2> PSymXY (const cPtxd<Type,2> & aP) 
{ 
    return cPtxd<Type,2>(aP.y(),aP.x()); 
}

/** This class represent 2D Homotetie , it can aussi be used for an non
   distorted camera with :
       * mTr -> principal point
       * mSc -> focale
*/

template <class Type>  class cHomot2D
{
      public :
          static constexpr int TheDim=2;
          typedef Type  tTypeElem;
          typedef cHomot2D<Type>  tTypeMap;
          typedef cHomot2D<Type>  tTypeMapInv;

          typedef cPtxd<Type,2> tPt;

          static const int NbDOF() {return 3;}
          cHomot2D(const tPt & aTr,const Type & aSc)  :
              mTr (aTr),
              mSc (aSc)
          {
          }
          inline tPt  Value(const tPt & aP) const   {return mTr + aP * mSc;}
          inline tPt  Inverse(const tPt & aP) const {return (aP-mTr)/mSc  ;}
          tTypeMapInv MapInverse() const {return cHomot2D<Type>(-mTr/mSc,1.0/mSc);}
      private :
          tPt mTr;
          Type mSc;
};

template <class Type>  class cSim2D
{
      public :
          static constexpr int TheDim=2;
          typedef Type          tTypeElem;
          typedef cSim2D<Type>  tTypeMap;
          typedef cSim2D<Type>  tTypeMapInv;

          typedef cPtxd<Type,2> tPt;

          cSim2D(const tPt & aTr,const tPt & aSc)  :
              mTr (aTr),
              mSc (aSc)
          {
          }
          static const int NbDOF() {return 4;}

          inline tPt  Value(const tPt & aP) const {return mTr + aP * mSc;}
          inline tPt  Inverse(const tPt & aP) const {return (aP-mTr)/mSc  ;}

          tTypeMapInv  MapInverse() const {return cSim2D<Type>(-mTr/mSc,tPt(1.0,0.0)/mSc);}
                
      private :
          tPt mTr;
          tPt mSc;
};



class  cTriangle2DCompiled : public cTriangle2D
{
       public :
           cTriangle2DCompiled(const cTriangle2D & aTri);
           cTriangle2DCompiled(const cPt2dr & aP0,const cPt2dr & aP1,const cPt2dr & aP2);

           bool  Regular() const;  ///<  Non degenerate i.e  delta !=0
           cPt3dr  CoordBarry(const     cPt2dr & aP) const; ///< Barrycentric coordinates
           double ValueInterpol(const     cPt2dr & aP,const cPt3dr & aValues) const;  ///< Interpolated value
           cPt2dr GradientVI(const cPt3dr& aValues) const;  ///< Gradient of Interpolated value

           static cTriangle2DCompiled RandomRegularTri(double aSz,double aEps=1e-3);

           double Insideness(const cPt2dr &) const; // <0 out, > inside, 0 : frontier
           bool   Insides(const cPt2dr &,double aTol=0.0) const; // Tol<0 give more points
           void PixelsInside(std::vector<cPt2di> & aRes,double aTol=-1e-5) const;

       private :
           void  AssertRegular() const;  //  Non degenerate i.e  delta !=0
           /*  
              For barycentrique coord, we have :
              L1 = (CX1  CY1)   (X1-X0   X2-X0)  =  (1  0)
              L2 = (CX2  CY20   (Y1-Y0   Y2-Y0)     (0  1)
           */
           double mDelta;
           cPt2dr mL1;
           cPt2dr mL2;
};

// std::pair<cTriangle2D,cPt3dr> Mqk=////


class cTriangulation2D : public cTriangulation<2>
{
	public :
           cTriangulation2D(const std::vector<tPt>&);
	   void  MakeDelaunay();
	public :
};


// geometric   Flux of pixel

typedef std::vector<cPt2di> tResFlux;

void      GetPts_Circle(tResFlux & aRes,const cPt2dr & aC,double aRay,bool with8Neigh);
tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh);



};

#endif  //  _MMVII_GEOM2D_H_
