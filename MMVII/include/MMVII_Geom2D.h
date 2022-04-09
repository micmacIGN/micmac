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
    private :
       tPt     mNorm;
};


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
          static cSim2D FromExample(const tPt & aP0In,const tPt & aP1In,const tPt & aP0Out,const tPt & aP1Out )  ;
          static const int NbDOF() {return 4;}

          inline tPt  Value(const tPt & aP) const {return mTr + aP * mSc;}
          inline tPt  Inverse(const tPt & aP) const {return (aP-mTr)/mSc  ;}

          tTypeMapInv  MapInverse() const {return cSim2D<Type>(-mTr/mSc,tPt(1.0,0.0)/mSc);}
                
	  ///  Generate the 3D-Sim having same impact in the plane X,Y
	  cSimilitud3D<Type> Ext3D() const;
      private :
          tPt mTr;
          tPt mSc;
};



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
           void PixelsInside(std::vector<cPt2di> & aRes,double aTol=-1e-5) const;

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
	   void  MakeDelaunay();
	public :
};


// geometric   Flux of pixel

typedef std::vector<cPt2di> tResFlux;

void      GetPts_Circle(tResFlux & aRes,const cPt2dr & aC,double aRay,bool with8Neigh);
tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh);



};

#endif  //  _MMVII_GEOM2D_H_
