#ifndef  _MMVII_MAPPINGS_H_
#define  _MMVII_MAPPINGS_H_

namespace MMVII
{
template <class Type,const int Dim> class cDataBoundedSet ;
template <class Type,const int DimIn,const int DimOut> class cDataMapping;

/** \file MMVII_Mappings.h
    \brief contain interface class for continuous mapping

   Most probably this will evolve a lot, with several reengenering 
  phases. 

*/

template <class Type,const int DimIn,const int DimOut> class cMapping
{
    public :
      typedef cDataMapping<Type,DimIn,DimOut> tDataMap;
      cMapping(tDataMap * aDM);
      tDataMap * DM() {return  mRawPtr;}
      const tDataMap * DM() const {return  mRawPtr;}

    private :
      std::shared_ptr<tDataMap>   mPtr; 
      tDataMap*                   mRawPtr;
};

template <class Type,const int Dim> class cDataBoundedSet : public cMemCheck
{
    public :
      typedef  cPtxd<Type,Dim>  tPt;

      /// Does it belong to the set;  default true
      virtual bool Inside(const tPt &) const;
       
    private :
       cTplBox<Type,Dim> mBox;
};

/// Class that represent a continous mapping R^k -> R^n



template <class Type,const int DimIn,const int DimOut> class cDataMapping : public cMemCheck
{
    public :
      typedef  cMapping<Type,DimIn,DimOut> tMap;
      typedef  cPtxd<Type,DimOut>          tPtOut;
      typedef  cPtxd<Type,DimIn>           tPtIn;
      typedef  std::vector<tPtIn>          tVecIn;
      typedef  std::vector<tPtOut>         tVecOut;
      typedef  cDenseMatrix<Type>          tJac;  ///< jacobian (DimIn DimOut); DimOut=1 =>line vector/linear form
      typedef  std::vector<tJac>         tVecJac;
      typedef std::pair<tPtOut ,tJac>                    tResJac;
      typedef std::pair<const tVecOut *,const tVecJac*>  tResVecJac;


           // ========== Computation of values ==============
      virtual  const  tVecOut &  Direct(const tVecIn & ) const;
      virtual  tPtOut Direct(const tPtIn &) const;

      virtual tResVecJac  Jacobian(const tVecIn &) const;
      virtual tResJac     Jacobian(const tPtIn &) const;


      // virtual  void ComputeValAndJac() const;
      /// Compute image in direct sens
      // virtual  const std::pair<tPtOut,tJac> &  ComputeValAndGrad(const tPtIn &) const;

      /// compute diffenrentiable method , default = erreur
    protected :
       /// This one can compute jacobian
       cDataMapping();
       /**  EpsJac is used to compute the jacobian by finite difference, */
       cDataMapping(const tPtIn & aEps);

       tPtIn               mEpsJac;
       bool                mJacByFiniteDif;
       // std::vector<tJac>   mGrads;
       // std::vector<tJac>   mResGrads;

       mutable tVecOut  mBufOut;
       mutable tVecOut  mJBufOut;
       mutable tVecIn   mBufIn;
       mutable tVecIn   mBufIn1Val;
       mutable tVecJac  mJacReserve;
       mutable tVecJac  mJacResult;

       inline tVecOut&  BufOut()    const {return mBufOut;}
       inline tVecOut&  BufOutCleared()    const {mBufOut.clear();return mBufOut;}
       inline tVecOut&  JBufOut()   const {return mJBufOut;}
       inline tVecOut&  JBufOutCleared()   const {mJBufOut.clear();return mJBufOut;}
       inline tVecIn&   BufIn()     const {return mBufIn;}
       inline tVecIn&   BufInCleared()  const {mBufIn.clear(); return mBufIn;}
       inline tVecIn &  BufIn1Val() const {return mBufIn1Val;}
       tVecJac & BufJac(tU_INT4 aSz) const ; /// Sz<0 => means clear buf
};

template <class Type,const int Dim> class cInvertDIMByIter;

template <class Type,const int Dim> class cDataInvertibleMapping :  public cDataMapping<Type,Dim,Dim>
{
    public :
      friend class cInvertDIMByIter <Type,Dim>;

      typedef cDataMapping<Type,Dim,Dim>     tDataMap;
      typedef cMapping<Type,Dim,Dim>         tMap;
      typedef typename  tDataMap::tPtIn      tPt;
      typedef typename  tDataMap::tVecIn     tVecPt;
      typedef typename  tDataMap::tResVecJac tResVecJac;

      void SetRoughInv(tMap,const Type& aDistTol,int aNbIterMax);
      const tDataMap *    RoughInv() const ;
      virtual  const  tVecPt &  ComputeInverse(const tVecPt &) const;
    protected :
      cDataInvertibleMapping();

      tMap                mRoughInv;
      Type                mDTolInv;
      int                 mNbIterMaxInv;
};
/*
*/

template <class Type,const int Dim> class cMappingIdentity :  public cDataMapping<Type,Dim,Dim>
{
    public :
      typedef cDataMapping<Type,Dim,Dim> tDataMap;
      typedef typename  tDataMap::tPtIn  tPt;
      typedef typename  tDataMap::tVecIn tVecPt;
      tPt Direct(const tPt &) const override;
      const  tVecPt &  Direct(const tVecPt & ) const override;
};

/*
*/

/**  We coul expect that DimIn=DimOut, but in fact in can be used
to represent a mapping from a part of the plane to a part of the sphere */

/*

template <class Type,const int DimIn,const int DimOut> 
         class cInvertibleMapping : public cMapping<Type,DimIn,DimOut>
{
    public :
      typedef  cPtxd<Type,DimOut> tPtOut;
      typedef  cPtxd<Type,DimIn>  tPtIn;
      typedef  cDenseMatrix<Type> tGrad;  ///< For each 
      virtual  tPtOut  Inverse(const tPtOut &) const = 0;
};

template <class Type> class  cImageSensor : public  cMapping<Type,3,2>
{
    public :
};

template <class Type> class cImagePose : public cInvertibleMapping<Type,3,3>
{
    public :
      typedef  cPtxd<Type,3>  tPt;
      /// Coordinate Cam -> Word ; Pt =>  mC + mOrient * Pt
      tPt  Direct(const tPt &)  const override;  
      /// Coordinate Cam -> Word ; Pt =>  (Pt-mC) * mOrient 
      tPt  Inverse(const tPt &) const override;  // 
      // cImagePose();

    private :
       cDenseMatrix<Type>  mOrient;
       tPt                 mC;
};
*/

/*
Avec R=N(x,y,z) et r=N(x,y)

  Ter -> Cam

       Proj Cam normale :  x/z y/z   ou  
       Proj fisheye equiline :   atan(r/z) * (x/r,y/r) = z/r atan(r/z) (x/z,y/z)

            ou Asin(r/R) *(x/r,y/r) = (Asin(r/R)*R/r) (x/R,y/R)

       Si appel AsinC(X) = asin(X)/X, fonction prolongeable en 0 par AsinC(0) =1 et Cinfini,
       alors
 
         Proj = AsinC(r/R) * (x/R,y/R)   avec l'avantage de n'avoir que des terme borne
         (en fait compris entre -1 et 1) et bien defini meme en x,y=0,0  
         Mais pb car deriv infini en 1 et ne gere pas bien aude la de pi/2

         Qhi(x,y,z) = atan2(r,z)/r  * (x,y)

      Autre method   ATC(A,B) = atan2(A,B)/A     =>  ATC(r,z) * (x,y)

     Encore une autre :
          Si z>r     (Asin(r/R)*R/r) (x/R,y/R)
          Si r>z>-r  (Acos(z/R) (x/r,y/r)
          Si -r>     (Asin(r/R)*R/r) (x/R,y/R)
      Cela fait intervenir les fonc
           *  AsinC pronlogeable en 0 , et

*/


};

#endif  //  _MMVII_MAPPINGS_H_
