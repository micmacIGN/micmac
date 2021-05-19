#ifndef  _MMVII_MAPPINGS_H_
#define  _MMVII_MAPPINGS_H_

/* For in & out computation of vector of points, do we use static buffer or 
   do each object has its own buffer. First option is more economic, but can lead to
   sides effect if buffering is not well understood . So for now I maintain the possibility
   to have the two option
*/

#define MAP_STATIC_BUF true


 /*  These macro are for now the only way I found for detecting infinite recursion
      in case user did define neither buffered nor unbuffered methods for Values, Jacobian, Inverse ...

      Not very proud of that, but can help to detect the error
   */


#define  MACRO_CHECK_RECURS_BEGIN\
 static int  aCPT_CHECK_RECURS=0;\
 MMVII_INTERNAL_ASSERT_strong((aCPT_CHECK_RECURS==0),"Forbiden Recursive Call");\
 aCPT_CHECK_RECURS++;

#define  MACRO_CHECK_RECURS_END\
 aCPT_CHECK_RECURS--;


namespace MMVII
{
template <class Type,const int Dim> class cDataBoundedSet ;
template <class Type,const int DimIn,const int DimOut> class cDataMapping;

/** \file MMVII_Mappings.h
    \brief contain interface class for continuous mapping

   Most probably this will evolve a lot, with several reengenering 
  phases. 

  Mapping are object for representing mathematicall smooth function from R^K to R^N. 

  [1]  WHAT
  The typicall mapping used in photogrammetry are :
      * distorsion (fraser, brown,   polynomial)  R^2 => R^2
      * projection (stenope,fisheye, pushbroom)   R^3 => R^2
      * polynomial model    R^K => R ^N
      * parametric maping for estimating global transformation between repair, image ...
         # External orientation, helmert transform ...  R^3 => R^3
         # parametric 2D maping for estimating global image tranform (rotation,affine,homography, polynomial)
      * generic epipolar ressempling  R^2 => R^1

  [2] WHY
  The typicall thing we want to do with mappings are :
      * compute their value  M(P)
      * when N=K: 
          # for a given P,  compute the invert value Q such that M(Q) = P
          #  for certain mapping belong to a given group, compute its exact invert mapping 
             (i.e affinty, homogra ...)
          # compute M' an approximate inverse of M such that M'(M(P)) ~ P, typically M' can be polynomial
            and approximation can be made by least square
      * compute the value of the differential in given point
      * estimate a mapping M from a set of example (Pi,Qi) such that M(Pi)~Qi , taking, or not,
        into account the presence of outlayers
      * use a mapping to re-sample an image

  [3] HOW
  The implementation in MMVII plane to have the following objective :
      * be general in WHAT and WHY
      * be relatively simple to use, i.e. for defining a new mapping the user has
       the minimum to do : define the direct mapping
      * be efficient in term of memory and time by offering the user to define more
       than the minimum (own jacobian, own inverse ...)

  Typically, the idea is that main services, like estimation, inversion, resampling ...
  are implemanted using virtual base classes, and user can benefit of these services with his 
  new mapping functions, as soon as they redefine a minimum of methods.

  I hope this implimentation complies with the above requirement, hower it comes to certain
  complexity on my side (implementation of the general class).

  For the trade-off between efficiency and simplicity on user side, the general philosphy is to offer the
  user the possibility to choice. For example :

      * if simplicity is priviligiated, user can just define a methode operating on a single
        value and he will have acces to derivative (with finite difference) and to the inverse
        (if he provide an additionnal rough inverse) with an iterative methods

      * is efficiency is priviligiated, user can define a method computing value and jacobian
        operating and vectors for taking benefit of his parallel implementation.
  

*/

/** Shared pointer on the real class, later it will be the main class accessible by non derived
    In the devlopment step  cDataMapping is still accessible.
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

/** Class for defining a bounded subset of R^n. TO DEVLOP ...
*/


template <class Type,const int Dim> class cDataBoundedSet : public cMemCheck
{
    public :
      typedef  cPtxd<Type,Dim>   tPt;
      typedef  std::vector<tPt>  tVecPt;
      typedef  cTplBox<Type,Dim> tBox;

      cDataBoundedSet(const tBox &);
      /// Does it belong to the set;  default =belong to box
      virtual bool InsideWithBox(const tPt &) const;
      /// Does it belong to the set;  default =true
      virtual bool Inside(const tPt &) const;

       const tBox & Box() const;

       void GridPointInsideAtStep(tVecPt&,Type aStepMoy) const;
       void GridPointInsideOfNbPoints(tVecPt&,int aStepMoy) const;
       
    private :
       cTplBox<Type,Dim> mBox;
};



/**   Mother base classe for defining a mapping  R^k => R^n with k=DimIn and n=DimOut

      The virtual methods "Values" and "Value" compute the image of poinst by the mapping.
      The derived classe MUST override  at least "Values" or "Value"
    
         * Value works on single points, if not overrided, Values is called with one element vector
         * Values works on vector, if not override, Value is called for each element of the vector

      If nor Value neither Values are defined, we have a potential infinite recursion (which is
      cathed by Macros  MACRO_CHECK_RECURS_BEGIN / MACRO_CHECK_RECURS_END to avoid an unclear
      core-dump).  Conversely it is possible (but probably not very usefull) to override both method 
      for very fine optimzation.

      There is two methods values :
         V2 : virtual const vec&  Values(vec&,cons vec&)
         V1 : const vec&  Values(cons vec&)
      The medod V2 is the "fundental" one which must be overrided. V1 is just a facility that use
      the internal buffer.

           ----------------------------------------------------------- 

      There is exactly the same contruction with Jacobian(s) which are methods for computing the 
      jacobian(s) AND the value(s) of point(s).  

          * Jacobian  => return a pair (Point,Matrix)
          * Jacobians => return a pair (pointer on vector of Point,pointer on vector of Matrix)

      However there is a TRICKY thing to take care of :

           * between different calls to Values or Jacobians, if user use V1 of J1 methods ,
             the results are returns as a reference to an internal buffer
           * so afetr each  calls , the result overwrite the previous one
           * for values, if the vector must be memorized, it is sufficient to do something
                 tVecPt V=Values(..)
             and it will create a copy that will not be overide
           * for Jacobian this would not work, the copy would be made, but as the matrix
             are stored as pointer (shared) to the real data, it will be always the same matrices
             that will be used.

      Maybe later, offer an additionale safe copy version....
  
           ----------------------------------------------------------- 
      This design is motivated by the following reasons :

         * use buffered mode (on vector) and compute simultaneously values & jacobian to 
           take benfit of the opportunity of code generation in MMVII

         * offers non buffered mode when simplicity is privilegiated on efficiency
*/



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
      typedef std::pair<const tVecOut *,const tVecJac*>  tCsteResVecJac;
      typedef std::pair<tVecOut *,tVecJac*>  tResVecJac;


           // ========== Computation of values ==============
      virtual  const  tVecOut &  Values(tVecOut &,const tVecIn & ) const;  //V2
      const  tVecOut &  Values(const tVecIn & ) const;   //  V1
      virtual  tPtOut Value(const tPtIn &) const;

      virtual tCsteResVecJac  Jacobian(tResVecJac,const tVecIn &) const;  //J2
      tCsteResVecJac  Jacobian(const tVecIn &) const;  //J1
      virtual tResJac     Jacobian(const tPtIn &) const;



      /// compute diffenrentiable method , default = erreur
    protected :
       /// This one can compute jacobian
       cDataMapping();
       /**  EpsJac is used to compute the jacobian by finite difference, */
       cDataMapping(const tPtIn & aEps);
       /**  call "cDataMapping(tPt)" with cste point */
       cDataMapping(const Type & aEps);

       tPtIn               mEpsJac;
       bool                mJacByFiniteDif;
       // std::vector<tJac>   mGrads;
       // std::vector<tJac>   mResGrads;

#if (MAP_STATIC_BUF)
       static tVecOut&  BufOut()         {static tVecOut aRes; return aRes;}
       static tVecOut&  JBufOut()        {static tVecOut aRes; return aRes;}
       static tVecIn&   BufIn()          {static tVecIn  aRes; return aRes;}
       static tVecIn&   JBufIn()         {static tVecIn  aRes; return aRes;}

       static tVecOut&  BufOutCleared()  { BufOut().clear() ; return  BufOut();}
       static tVecOut&  JBufOutCleared() {JBufOut().clear() ; return JBufOut();}
       static tVecIn&   BufInCleared()   { BufIn().clear()  ; return  BufIn(); }
       static tVecIn&   JBufInCleared()  {JBufIn().clear()  ; return JBufIn(); }

       static tVecIn &  BufIn1Val()  {static tVecIn  aRes{tPtIn()}; return aRes;}
       static tVecJac & BufJac(tU_INT4 aSz) ; 
#else  // !MAP_STATIC_BUF
    private :
       mutable tVecOut  mBufOut;
       mutable tVecOut  mJBufOut;
       mutable tVecIn   mBufIn;
       mutable tVecIn   mJBufIn;
       mutable tVecIn   mBufIn1Val;
       mutable tVecJac  mJacReserve;
       mutable tVecJac  mJacResult;

    protected :
       inline tVecOut&  BufOut()    const {return mBufOut;}
       inline tVecOut&  BufOutCleared()    const {mBufOut.clear();return mBufOut;}
       inline tVecOut&  JBufOut()   const {return mJBufOut;}
       inline tVecOut&  JBufOutCleared()   const {mJBufOut.clear();return mJBufOut;}
       inline tVecIn&   BufIn()     const {return mBufIn;}
       inline tVecIn&   BufInCleared()  const {mBufIn.clear(); return mBufIn;}
       inline tVecIn&   JBufIn()     const {return mJBufIn;}
       inline tVecIn&   JBufInCleared()  const {mJBufIn.clear(); return mJBufIn;}
       inline tVecIn &  BufIn1Val() const {return mBufIn1Val;}
       tVecJac & BufJac(tU_INT4 aSz) const ; 
#endif // MAP_STATIC_BUF
};

/**   This is the mother class of maping that can compute the inverse of a point.

      The method comptuing the inverse are "Inverse(s)", and we have the same behaviour as with Value(s).
*/

template <class Type,const int Dim> class cDataInvertibleMapping :  public cDataMapping<Type,Dim,Dim>
{
    public :
      typedef cDataMapping<Type,Dim,Dim>     tDataMap;
      typedef typename  tDataMap::tPtIn      tPt;
      typedef typename  tDataMap::tVecIn     tVecPt;


           // ========== Constructors ==============
      cDataInvertibleMapping(const tPt &);
      cDataInvertibleMapping();

           // ========== Computation of inverses ==============
      virtual  const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const;
      const  tVecPt &  Inverses(const tVecPt & ) const;
      virtual  tPt Inverse(const tPt &) const;

    private :
#if (MAP_STATIC_BUF)
       static tVecPt&  BufInvOut()         {static tVecPt aRes; return aRes;}
       static tVecPt&  BufInvOutCleared()  { BufInvOut().clear() ; return  BufInvOut();}
#else  // !MAP_STATIC_BUF
       mutable tVecPt  mBufInvOut;
       inline tVecPt&  BufInvOut()    const {return mBufInvOut;}
       inline tVecPt&  BufInvOutCleared()    const {mBufInvOut.clear();return mBufOut;}
#endif
};




/* class doing the real iterative computation */

template <class Type,const int Dim> class cInvertDIMByIter;
/**   This class offer a concrete computation of the inverse by a iterative method, relying
    on jacobian computation. A first "guess" of the inverse must be given. If the mapping is
    very smooth globally close to a linear mapping the guess is of no importance for the convergence.
    On the other hand, giving a relatively accurate guess will accelerate the result and, more
    importantly, will be safer for the convergence.
*/

template <class Type,const int Dim> class cDataIterInvertMapping :  public cDataInvertibleMapping<Type,Dim>
{
    public :
      typedef cInvertDIMByIter<Type,Dim>     tHelperInvertIter;
      friend  tHelperInvertIter;

      typedef cDataInvertibleMapping<Type,Dim> tDataInvMap;
      typedef typename  tDataInvMap::tPt       tPt;
      typedef typename  tDataInvMap::tVecPt    tVecPt;

      typedef cMapping<Type,Dim,Dim>         tMap;
      typedef cDataMapping<Type,Dim,Dim>     tDataMap;
      typedef typename  tDataMap::tResVecJac tResVecJac;


      const  tVecPt &  Inverses(tVecPt &,const tVecPt &) const override;
      // Accessors 
      const tDataMap *     RoughInv() const ;
      const Type & DTolInv() const;
    protected :
      cDataIterInvertMapping(const tPt &,tMap,const Type& aDistTol,int aNbIterMax);
      cDataIterInvertMapping(tMap,const Type& aDistTol,int aNbIterMax);

    private :
      tHelperInvertIter *  StrInvertIter() const;

      mutable std::shared_ptr<tHelperInvertIter> mStrInvertIter;
      tMap                mRoughInv;
      Type                mDTolInv;
      int                 mNbIterMaxInv;
};

/** When we have an existing mapping, we want to invert it by iteration, we cannot inherit
if we cannot modify, so we make it member .  The methods just call method of mMap...
*/

template <class Type,const int Dim> class cDataIIMFromMap : public cDataIterInvertMapping<Type,Dim>
{
    public :
      typedef cDataIterInvertMapping<Type,Dim> tDataIIMap;
      typedef cMapping<Type,Dim,Dim>         tMap;

      using typename tDataIIMap::tPt;
      using typename tDataIIMap::tVecPt;
      using typename tDataIIMap::tCsteResVecJac;
      using typename tDataIIMap::tResVecJac;

      cDataIIMFromMap(tMap aMap,const tPt &,tMap aRoughInv,const Type& aDistTol,int aNbIterMax);
      cDataIIMFromMap(tMap aMap,tMap aRoughInv,const Type& aDistTol,int aNbIterMax);

      const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;  //V2
      tCsteResVecJac  Jacobian(tResVecJac,const tVecPt &) const override;  //J2
    private :
      tMap                mMap;
};

/** Represntation of identity as a mapping */

template <class Type,const int Dim> class cMappingIdentity :  public cDataMapping<Type,Dim,Dim>
{
    public :
      typedef cDataMapping<Type,Dim,Dim> tDataMap;
      typedef typename  tDataMap::tPtIn  tPt;
      typedef typename  tDataMap::tVecIn tVecPt;
      tPt Value(const tPt &) const override;
      const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;
};

template <class Type,const int DimIn,const int DimOut>
    class cDataMapCalcSymbDer : public cDataMapping<Type,DimIn,DimOut>
{   
    public :
      typedef typename NS_SymbolicDerivative::cCalculator<Type> tCalc;
      typedef cDataMapping<Type,DimIn,DimOut> tDataMap;

      using typename tDataMap::tVecIn;
      using typename tDataMap::tVecOut;
      using typename tDataMap::tCsteResVecJac;
      using typename tDataMap::tResVecJac;
      using typename tDataMap::tVecJac;
      using typename tDataMap::tPtIn;
      using typename tDataMap::tPtOut;

       const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;  //V2
       tCsteResVecJac  Jacobian(tResVecJac,const tVecIn &) const override;  //J2

       cDataMapCalcSymbDer(tCalc  * aCalcVal,tCalc  * aCalcDer,const std::vector<Type> & aVObs);
       void SetObs(const std::vector<Type> &);

    private  :
       void CheckDim(tCalc *,bool Derive);
       tCalc  *           mCalcVal;
       tCalc  *           mCalcDer;
       std::vector<Type>  mVObs;
};

/**
    Sometime we want to manipulate "small" maping directly, with no virtual interface,
  an after reuse these code in global mappings, this is possible if the "small" mapping
  complies with the following "contract" :

      define the type :    typedef Type  TheType;
      degines its dimension :    static constexpr int TheDim=2;
      defines its degree of freedom :    static const int NbDOF() {return 4;}
      defiens direct mapping :    inline tPt  Value(const tPt & aP) 
      defines inverst mapping for a point :    inline tPt  Inverse(const tPt & aP) 
      defines global invert map :    cSim2D<Type>  MapInverse() const

   See class cSim2D<Type> for an example of  elementary mapping.

*/

template <class cMapElem,class cIMapElem> class cInvertMappingFromElem :  public 
       cDataInvertibleMapping<typename cMapElem::TheType,cMapElem::TheDim>
{
    public :
         static constexpr int     Dim=cMapElem::TheDim;
         typedef typename  cMapElem::TheType TheType;
         static_assert(cMapElem::TheDim==cIMapElem::TheDim);

         typedef cDataInvertibleMapping<TheType,Dim>  tDataIMap;
         typedef typename  tDataIMap::tPt             tPt;
         typedef typename  tDataIMap::tVecPt          tVecPt;

         const  tVecPt &  Values(tVecPt & aRes,const tVecPt & aVIn ) const override;
         tPt Value(const tPt &) const override;
         const  tVecPt &  Inverses(tVecPt & aRes,const tVecPt & aVIn ) const override;
         tPt  Inverse(const tPt &) const override;

         cInvertMappingFromElem(const cMapElem & aMap,const cIMapElem & aIMap); 
         cInvertMappingFromElem(const cMapElem & aMap);  // requires that aMap can compute its inverse
    private :
         cMapElem   mMap;  // Map
         cIMapElem  mIMap; // Map inverse
};

/**
     We have a set of function F1,  .. Fp     R^k => R ^n

     We have samples      Km , Nm 

     We want to solve by  lest square  :
            Sum   Al Fl (Km) = Nm
*/
template <class Type,const int  DimIn,const int DimOut> class cLeastSqComputeMaps
{
     public :
         typedef  cPtxd<Type,DimOut>          tPtOut;
         typedef  cPtxd<Type,DimIn>           tPtIn;
         typedef  std::vector<tPtIn>          tVecIn;
         typedef  std::vector<tPtOut>         tVecOut;

         cLeastSqComputeMaps(size_t aNbFunc);
         void AddObs(const tPtIn & aPt,const tPtOut & aValue,const tPtOut & aPds);
         void AddObs(const tPtIn & aPt,const tPtOut & aValue,const Type & aPds);
         void AddObs(const tPtIn & aPt,const tPtOut & aValue);
     private :
         virtual void ComputeValFuncsBase(tVecOut &,const tPtIn & aPt) = 0;
         size_t             mNbFunc;
         cLeasSqtAA<Type>   mLSQ;
         cDenseVect<Type>   mCoeffs;
         tVecOut            mBufPOut;
};




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
