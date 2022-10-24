#ifndef  _MMVII_MAPPINGS_H_
#define  _MMVII_MAPPINGS_H_


#include "MMVII_ImageInfoExtract.h"

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
template <class Type,const int Dim> class cSphereBoundedSet;//   cDataBoundedSet<Type,Dim>



template <class Type,const int DimIn,const int DimOut> class cMapping;
template <class Type,const int DimIn,const int DimOut> class cDataMapping;
template <class Type,const int Dim> class cDataInvertibleMapping ;// :  public cDataMapping<Type,Dim,Dim>
template <class Type,const int Dim> class cDataIterInvertMapping ;// :  public cDataInvertibleMapping<Type,Dim>
template <class Type,const int Dim> class cDataIIMFromMap ; // : public cDataIterInvertMapping<Type,Dim>

template <class Type,const int Dim> class cMappingIdentity ; // :  public cDataMapping<Type,Dim,Dim>
template <class Type,const int DimIn,const int DimOut> class cDataMapCalcSymbDer ;// : public cDataMapping<Type,DimIn,DimOut>
template <class cMapElem> class cInvertMappingFromElem ;
    // :  public cDataInvertibleMapping<typename cMapElem::TheType,cMapElem::TheDim>
template <class Type,const int  DimIn,const int DimOut> class cLeastSqComputeMaps;
template <class Type,const int DimIn,const int DimOut> class cLeastSqCompMapCalcSymb; 

template <class Type,const int Dim> class cBijAffMapElem;


    //  : public cLeastSqComputeMaps<Type,DimIn,DimOut>

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

/*
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
*/

/** Class for defining a bounded subset of R^n. TO DEVLOP ...
*/

template <class Type,const int Dim> class cDataBoundedSet : public cMemCheck
{
    public :
      typedef  cPtxd<Type,Dim>   tPt;
      typedef  std::vector<tPt>  tVecPt;
      typedef  cTplBox<Type,Dim> tBox;

      cDataBoundedSet(const tBox &);
      virtual ~cDataBoundedSet<Type,Dim>();

      /// quantitative  + inside, - outside , 0 at the frontier
      tREAL8 InsidenessWithBox(const tPt &) const;
      ///  default return many (infinite) because
      virtual tREAL8 Insideness(const tPt &) const;

      /// Does it belong to the set;  default =belong to box,  defined from InsidenessWithBox
      bool InsideWithBox(const tPt &) const;
      /// Does it belong to the set;  default =belong to box,  defined from InsidenessWithBox
      bool InsideWithBox(const cTriangle<Type,Dim> &) const;
      /// Does it belong to the set;  default =true, defined form Insideness
      bool Inside(const tPt &) const;


      const tBox & Box() const;

      /// Generate random point inside 
      tPt GeneratePointInside() const;

       /// Generate regular grid at given step, not used for now
       void GridPointInsideAtStep(tVecPt&,Type aStepMoy) const;
       /// idem, but fix the number of point (total number, not for each direction)
       void GridPointInsideOfNbPoints(tVecPt&,int aStepMoy) const;
       
    private :
       cDataBoundedSet(const cDataBoundedSet<Type,Dim> & ) = delete;
       cTplBox<Type,Dim> mBox;
};

cDataBoundedSet<tREAL8,3> *  MMV1_Masq(const cBox3dr &,const std::string & aNameFile);

/** specialization of BoundedSet to sphere, used for example to define convergence domain
    of radial function
 */

template <class Type,const int Dim> class cSphereBoundedSet : public cDataBoundedSet<Type,Dim>
{
     public :
         typedef  cTplBox<Type,Dim> tBox;
         typedef  cPtxd<Type,Dim>   tPt;
         /// construct from Box, center and radius
         cSphereBoundedSet(const tBox & aBox,const tPt & , const Type & aRadius);
         tREAL8 Insideness(const tPt &) const override;
     private :
         tPt  mCenter;  ///<  center of the spher
         Type mRadius;      ///< square of ray ,
};

/** Class for defining image of bounded set by a mapping
*/
template <class Type,const int Dim> class  cDataMappedBoundedSet : public cDataBoundedSet<Type,Dim>
{
      public :
           typedef cDataBoundedSet<Type,Dim>         tDBSet;
           typedef cDataInvertibleMapping<Type,Dim>  tDIMap;
           typedef cPtxd<Type,Dim>                   tPt;

           cDataMappedBoundedSet (const tDBSet *,const tDIMap *,bool AdoptSet,bool AdoptMap);
           ~cDataMappedBoundedSet ();
           double Insideness(const tPt & aPt) const override;  ///<  Map-1(Pt) belong to Set
      private :
           const tDBSet * mSet;
           const tDIMap * mMap;
           bool           mAdoptSet;
           bool           mAdoptMap;
};




/*
     Convention for gradiant vs/line columns  :  tJac(DimIn,DimOut)  SzX = DimIn = Number of col
      For examlpe   R^3 (x,yz)  -> R^2  (u,v)

            d/dx    d/dy  d/dz
        u    *       *     *
        v    *       *     *
 
*/


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
      virtual ~cDataMapping<Type,DimIn,DimOut>();
      // typedef  cMapping<Type,DimIn,DimOut> tMap;
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

      ///  buffered method, call unbeferred one
      virtual  const  tVecOut &  Values(tVecOut &,const tVecIn & ) const;  //V2
      ///  unbuffered method, call unbeferred one ...
      virtual  tPtOut Value(const tPtIn &) const;
      /// buffered, calls V2 with  own buffer
      const  tVecOut &  Values(const tVecIn & ) const;   //  V1

      /// PRE ALLOCATED VALUES ;  Pts is clear and must be pushed back, Jacob contain already the matrixes
      virtual tCsteResVecJac  Jacobian(tResVecJac,const tVecIn &) const;  //J2
      tCsteResVecJac  Jacobian(const tVecIn &) const;  //J1
      virtual tResJac     Jacobian(const tPtIn &) const;

      /// 
      cTplBox<Type,DimOut> BoxOfCorners(const cTplBox<Type,DimIn>&) const;


      cTriangle<Type,DimOut>  TriValue(const cTriangle<Type,DimIn> &) const;

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
       /// return a "Buffer" of jacobian, satic becaus alloc in class
       static tVecJac & BufJac(tU_INT4 aSz) ; 
#else  // !MAP_STATIC_BUF
    private :
       cDataMapping(const cDataMapping<Type,DimIn,DimOut> & ) = delete;
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

       /// return a "Buffer" of jacobian, on own ressources, const -> modify mutable var
       tVecJac & BufJac(tU_INT4 aSz) const ; 
#endif // MAP_STATIC_BUF
};

/** Specialization for DimIn=DimOut , introduce because we want to force invertible mapping
    to have DimIn==DimOut
*/

template <class Type,const int Dim> class cDataNxNMapping : public cDataMapping<Type,Dim,Dim>
{
    public :
      typedef  cDataMapping<Type,Dim,Dim> tDMap;
      typedef  cPtxd<Type,Dim>            tPt;
      using typename tDMap::tResJac;
      using typename tDMap::tJac;

      cDataNxNMapping(const tPt &);  ///< just initialize cDataMapping
      cDataNxNMapping();  ///< just initialize cDataMapping
      /// return bijective differential application , used for ex in BoxInByJacobian
      cBijAffMapElem<Type,Dim>  Linearize(const tPt & aPt) const;
};

/**   This is the mother class of maping that can compute the inverse of a point.

      The method comptuing the inverse are "Inverse(s)", and we have the same behaviour as with Value(s) :
        * No inverse is really defined, the 3 methods recall each-others
        * if no methods is redefined a fatal error will occurs (infinite recursion)
        * benefit is that  user has only to redefine buffered or unbuffered
      
*/

template <class Type,const int Dim> class cDataInvertibleMapping :  public cDataNxNMapping<Type,Dim>
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
      cDataInvertibleMapping(const cDataInvertibleMapping<Type,Dim> & ) = delete;
#if (MAP_STATIC_BUF) 
       static tVecPt&  BufInvOut()         {static tVecPt aRes; return aRes;}
       static tVecPt&  BufInvOutCleared()  { BufInvOut().clear() ; return  BufInvOut();}
#else  // !MAP_STATIC_BUF
       mutable tVecPt  mBufInvOut;
       inline tVecPt&  BufInvOut()    const {return mBufInvOut;}
       inline tVecPt&  BufInvOutCleared()    const {mBufInvOut.clear();return mBufOut;}
#endif
};

template <class Type,const int Dim> class cDataInvertOfMapping : public cDataInvertibleMapping <Type,Dim>
{
   public :
         typedef  cDataInvertibleMapping<Type,Dim>  tIMap;
         typedef cPtxd<Type,Dim>                    tPt;
         typedef std::vector<tPt>                   tVecPt;

         cDataInvertOfMapping(const tIMap * aMapToInv,bool toAdopt);
         ~cDataInvertOfMapping();

         const  tVecPt &  Inverses(tVecPt &,const tVecPt & ) const;
         const  tVecPt &  Values(tVecPt &,const tVecPt & ) const;
   private :
         const tIMap * mMapToInv;
         bool          mAdopted;
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

      // typedef cMapping<Type,Dim,Dim>         tMap;
      typedef cDataMapping<Type,Dim,Dim>     tDataMap;
      typedef typename  tDataMap::tResVecJac tResVecJac;


      const  tVecPt &  Inverses(tVecPt &,const tVecPt &) const override;
      // Accessors 
      const tDataMap *     RoughInv() const ;
      const Type & DTolInv() const;
      /// Access to the structure, only needed in some bench to create artificial difficult situations
      tHelperInvertIter *  StrInvertIter() const;

      ~cDataIterInvertMapping();
    protected :
      cDataIterInvertMapping(const tPt & aEpsDiff,tDataMap * aRoughInv,const Type& aDistTol,int aNbIterMax,bool AdoptRoughInv);
      cDataIterInvertMapping(tDataMap * aRoughInv,const Type& aDistTol,int aNbIterMax,bool AdoptRoughInv);

    private :
      cDataIterInvertMapping(const cDataIterInvertMapping<Type,Dim> & ) = delete;

      // mutable std::shared_ptr<tHelperInvertIter> mStrInvertIter;
      mutable tHelperInvertIter* mStrInvertIter;
      tDataMap *          mRoughInv;
      Type                mDTolInv;
      int                 mNbIterMaxInv;
      bool                mAdoptRoughInv;
};

/** When we have an existing mapping, we want to invert it by iteration, if we cannot inherit
    (we only get the object) we cannot modify; so we make the object a member .  The methods just call method of mMap...
*/

template <class Type,const int Dim> class cDataIIMFromMap : public cDataIterInvertMapping<Type,Dim>
{
    public :
      typedef cDataIterInvertMapping<Type,Dim> tDataIIMap;
      typedef cDataMapping<Type,Dim,Dim>     tDataMap;
      // typedef cMapping<Type,Dim,Dim>         tMap;

      using typename tDataIIMap::tPt;
      using typename tDataIIMap::tVecPt;
      using typename tDataIIMap::tCsteResVecJac;
      using typename tDataIIMap::tResVecJac;

      cDataIIMFromMap(tDataMap * aMap,const tPt &,tDataMap * aRoughInv,const Type& aDistTol,int aNbIterMax,bool AdoptMap,bool AdoptRIM);
      cDataIIMFromMap(tDataMap * aMap,tDataMap * aRoughInv,const Type& aDistTol,int aNbIterMax,bool AdoptMap,bool AdoptRIM);

      const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;  //V2
      tCsteResVecJac  Jacobian(tResVecJac,const tVecPt &) const override;  //J2
      ~cDataIIMFromMap();
    private :
      cDataIIMFromMap(const cDataIIMFromMap<Type,Dim> & ) = delete;
      tDataMap *                mMap;									   
      bool                      mAdoptMap;
};

/** Represntation of identity as a mapping */

// template <class Type,const int Dim> class cMappingIdentity :  public cDataMapping<Type,Dim,Dim>
template <class Type,const int Dim> class cMappingIdentity :  public cDataNxNMapping<Type,Dim>
{
    public :
      typedef cDataMapping<Type,Dim,Dim> tDataMap;
      typedef typename  tDataMap::tPtIn  tPt;
      typedef typename  tDataMap::tVecIn tVecPt;

      tPt Value(const tPt &) const override;
      const  tVecPt &  Values(tVecPt &,const tVecPt & ) const override;
      /// Initialize with eps=1 (no importance), later maybe add jacobian
      cMappingIdentity();
};

/** Create an interface for using a generated code as a mapping.
*/
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

      virtual ~cDataMapCalcSymbDer<Type,DimIn,DimOut>();

       const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;  ///< V2 : use mCalc to fill values
       tCsteResVecJac  Jacobian(tResVecJac,const tVecIn &) const override;  ///< J2 : use mCalcDer to compute derivative

       cDataMapCalcSymbDer(tCalc  * aCalcVal,tCalc  * aCalcDer,const std::vector<Type> & aVObs,bool ToDelete);
       void SetObs(const std::vector<Type> &); ///< just modify mVObs
       // void SetDeleteCalc(bool);

       const std::vector<Type> & VObs() const;
       std::vector<Type> & VObs();
    private  :
       cDataMapCalcSymbDer(const cDataMapCalcSymbDer<Type,DimIn,DimOut> & ) = delete;
       void CheckDim(tCalc *,bool Derive); ///< used to check that both calculator have adequate structure
       tCalc  *           mCalcVal;
       tCalc  *           mCalcDer;
       std::vector<Type>  mVObs;
       bool               mDeleteCalc;
};

/** Sometime we need to have type where DimIn=DimOut */
template <class Type,int Dim> class  cDataNxNMapCalcSymbDer  : public cDataNxNMapping<Type,Dim>
{
    public :
      typedef typename NS_SymbolicDerivative::cCalculator<Type> tCalc;
      typedef cDataMapping<Type,Dim,Dim> tDataMap;

      using typename tDataMap::tVecIn;
      using typename tDataMap::tVecOut;
      using typename tDataMap::tCsteResVecJac;
      using typename tDataMap::tResVecJac;
      using typename tDataMap::tVecJac;
      using typename tDataMap::tPtIn;
      using typename tDataMap::tPtOut; 
      const  tVecOut &  Values(tVecOut &,const tVecIn & ) const override;  //V2
      tCsteResVecJac  Jacobian(tResVecJac,const tVecIn &) const override;  //J2

      cDataNxNMapCalcSymbDer(tCalc  * aCalcVal,tCalc  * aCalcDer,const std::vector<Type> & aVObs,bool DeleteCalc);
      void SetObs(const std::vector<Type> &);
      const std::vector<Type> & VObs() const;
      std::vector<Type> & VObs();
    public :
       cDataMapCalcSymbDer<Type,Dim,Dim>  mDMS;

};

/**  Helper for extending map invere near frontier */

template <class Type,const int Dim>  struct cPtsExtendCMI
{
     public :
         typedef cPtxd<Type,Dim> tPt;
         cPtsExtendCMI(const tPt & aCurP,const tPt & aDir) ;

         tPt  mCurP;
         tPt  mDir;
};


/**   Class for computing an inverse mapping from :
         * the direct mapping to invert  EIn => EOut
         * a set of base function that linerly code the invert
         * a validity domain on the output space EOut
         * a "seed" point in input space

      The criterion for validity if   || J(Seed)^-1 * J(p) -Id ||  <  Threshold   J=Jacobian

       This is adapted to distorsion where :
          * we know the output space -> sensor space + an optional validty (masq image, circle ...)
          * we jo

       The method make grow a space where the mapping can reasonnabily be expect to be invertible,
  the critrion being for this is to ensure that the jacobian is always sufficiently close to the jacobian
  at the seed (pushed to the limit, when equals it means that function is linear).

       The growing is made on a grid by a connected component analysis starting from the seed.

       At the end, due to the sampling we may have few or no point close to the border/frontier. This
    is no good as we know that extrapolation do not work well, so we have a step  were we add
    a prolongation to go nearer to the frontier
*/



template <class Type,const int Dim> class  cComputeMapInverse
{
    public :
            //  aCMaxRel => define the zone relatively to the rho max
        friend void OneBench_CMI(double aCMaxRel);
        // using enum eLabelIm_CMI;
        typedef cLeastSqComputeMaps<Type,Dim,Dim> tLSQ;
        typedef cDataBoundedSet<Type,Dim>         tSet;
        typedef cDataNxNMapping<Type,Dim>         tMap;
        typedef cPtxd<Type,Dim>                   tPtR;
        typedef cPtxd<int,Dim>                    tPtI;
        typedef cTplBox<Type,Dim>                 tBoxR;
        typedef cPtsExtendCMI<Type,Dim>           tExtent;
        typedef typename tMap::tCsteResVecJac     tCsteResVecJac;

        /// Constructor, essentially memorize parameters
        cComputeMapInverse
        (
             const Type & aThreshJac, ///< Threshold on jacobian to ensure inversability
             const tPtR& aSeed,       ///< Seed point, in input space
             const int & aNbPtsIn,    ///< Approximate number of point (in the biggest size)
             tSet &,  ///< Set of validity, in output space
             tMap&,   ///< Maping to invert : InputSpace -> OutputSpace
             tLSQ&,  ///< Structure for computing the invert on base of function using least square   
             bool Test=false
        );
        void  DoAll(std::vector<Type> & aVSol);

        static int constexpr  TheNbIterByStep = 3;
        static Type constexpr TheStepFrontLim = 3e-2;

   private :
        cComputeMapInverse(const cComputeMapInverse<Type,Dim> &) = delete;
        /** Compute an approximation of Input box as reciproque of output box, use jacobian as
            we dont know inverse (else we would not be here ...) */
        tBoxR  BoxInByJacobian() const;
        /** From input real space to grid space */
        tPtI ToPix(const tPtR& aPR) const;
        /** From grid space to input real space*/
        tPtR FromPix(const tPtI& aPI) const;
        /// Is the jacobian sufficently close to its value on seed ?
        bool ValideJac(const cDenseMatrix<Type> & aMat) const;

        /// Add a Pixel in the queue if has not already be visited
        void Add1PixelTopo(const tPtI & aPix);
        /// Filters pixel geometrically OK (Jac+domain) and add them as obs for least square
        void FilterAndAddPixelsGeom();

        /// Make on iteration, at given step, to have point closer to the frontier
        void OneStepFront(const Type & aStepFront);

        /// Validate (POut/Jac) if in domain and jacobian is OK
        bool ValidateK(const tCsteResVecJac & aVecPJ,int aKp);
        /// Add one observtion for computing inverse, IsFront used for memo in test mode
        void AddObsMapDirect(const tPtR & aPIn,const tPtR & aPOut,bool IsFront);

	         // Copy of parameters
        Type          mThresholdJac;
        tPtR          mPSeed; //  seed point that is waranteed to be inside the domain
        tSet &        mSet;   // Definition set of Output space
        tMap &        mMap;   // Map to invert
        tLSQ &        mLSQ;   // systeme to compute the inverse as a linear composition of given base functions (using least square)
          // Created members
        tBoxR         mBoxByJac; ///< Box computed assuming that Map is equal to its jacobian in PSeed
        tBoxR         mBoxMaj;  ///< Majoration of box, taking into account possible  unstability and jacobian threshold
        Type          mStep;    ///< Step on the grid
        cPixBox<Dim>              mBoxPix;  ///< Pixel box to make image processing stuff
        cDataTypedIm<tU_INT1,Dim> mMarker;  ///< Marker image to make growing
        std::vector<tPtI>         mNextGen; ///< Next generation of pixel in growing region
        cDenseMatrix<Type>        mJacInv0; ///< Matrix invert of Jacobian in PSeed
        cDenseMatrix<Type>        mMatId;   ///< Id Matrix, helper for computing Jacobian criteria
        const std::vector<tPtI> &       mNeigh; ///< Neighbourhood for image-morpho-operation
        std::vector<tExtent>      mVExt; ///< Vector of "extension" to the frontier
        bool                      mTest; ///< Are we in test mode ?
    public :
        Type                      mStepFrontLim; // TheStepFrontLim
        std::vector<tPtR>         mVPtsInt; ///< For test, memo point interior
        std::vector<tPtR>         mVPtsFr;  ///< For test, memo point frontier

};





/**
    Sometime we want to manipulate "small" maping directly, with no virtual interface,
  an after reuse these code in global mappings, this is possible if the "small" mapping
  complies with the following "contract" :

      //  => NONNN : define the type :    typedef Type  TheType;
      degines its dimension :    static constexpr int TheDim=2;
      defines its degree of freedom :    static const int NbDOF() {return 4;}
      defiens direct mapping :    inline tPt  Value(const tPt & aP) 
      defines inverst mapping for a point :    inline tPt  Inverse(const tPt & aP) 
      defines global invert map :    cSim2D<Type>  MapInverse() const

   See class cSim2D<Type> for an example of  elementary mapping.

*/

template <class cMapElem> class cInvertMappingFromElem :  public 
       cDataInvertibleMapping<typename cMapElem::tTypeElem,cMapElem::TheDim>
{
    public :
         static constexpr int     Dim=cMapElem::TheDim;
         typedef typename  cMapElem::tTypeElem  tTypeElem;
         typedef cMapElem                       tMap;
         typedef typename cMapElem::tTypeMapInv tMapInv;


         static_assert(cMapElem::TheDim==tMapInv::TheDim);

         typedef cDataInvertibleMapping<tTypeElem,Dim>  tDataIMap;
         typedef typename  tDataIMap::tPt             tPt;
         typedef typename  tDataIMap::tVecPt          tVecPt;

         const  tVecPt &  Values(tVecPt & aRes,const tVecPt & aVIn ) const override;
         tPt Value(const tPt &) const override;
         const  tVecPt &  Inverses(tVecPt & aRes,const tVecPt & aVIn ) const override;
         tPt  Inverse(const tPt &) const override;

         cInvertMappingFromElem(const cMapElem & aMap,const tMapInv & aIMap); 
         cInvertMappingFromElem(const cMapElem & aMap);  // requires that aMap can compute its inverse
    private :
         cMapElem   mMap;  // Map
         tMapInv  mIMap; // Map inverse
};

/** Specialization when cMapElem is linear => constant jacobian */

template <class cMapElem> class cIMElemLinear :  public 
           cInvertMappingFromElem<cMapElem>
{
    public :
         typedef cInvertMappingFromElem<cMapElem> tIMap;
         typedef typename  tIMap::tTypeElem  tTypeElem;
         static constexpr int     Dim=tIMap::Dim;
         typedef cDataMapping<tTypeElem,Dim,Dim>  tDataMap;
         typedef cDenseMatrix<tTypeElem> tMat;
         using typename tDataMap::tVecIn;
         using typename tDataMap::tCsteResVecJac;
         using typename tDataMap::tResVecJac;
         using typename tDataMap::tResJac;
         using typename tDataMap::tPtIn;


         cIMElemLinear(const cMapElem & aMap,tMat & aMat);  // requires that aMap can compute its inverse
         tCsteResVecJac  Jacobian(tResVecJac,const tVecIn &) const override;  //J2
         tResJac     Jacobian(const tPtIn &) const override;
    private :
         tMat  mMat;
};

/**
     We have a set of function F1,  .. Fp     R^k => R ^n, we want to estimate F  as a linear combination:
           F =  Sum   Al Fl 
     We have samples      Km , Nm 
     We want to solve by  lest square  :   F (Km) = Nm

     The computation is not optimized (not parallized) as it would add a complexity important for
     a probable low gain :
         * there is not so many samples
         * we dont compute the derivative
         * the filling of matrix is however a cost not parallized
*/


template <class Type,const int  DimIn,const int DimOut> class cLeastSqComputeMaps
{
     public :
         typedef  cPtxd<Type,DimOut>          tPtOut;
         typedef  cPtxd<Type,DimIn>           tPtIn;
         typedef  std::vector<tPtIn>          tVecIn;
         typedef  std::vector<tPtOut>         tVecOut;

         cLeastSqComputeMaps(size_t aNbFunc);
         virtual  ~cLeastSqComputeMaps();
         /// Add obs 
         void AddObs(const tPtIn & aPt,const tPtOut & aValue,const tPtOut & aPds);
         void AddObs(const tPtIn & aPt,const tPtOut & aValue,const Type & aPds);
         void AddObs(const tPtIn & aPt,const tPtOut & aValue);
         // =========== ACCESSOR ==============
         const size_t &  NbFunc() const {return mNbFunc;}

         void ComputeSol(std::vector<Type>&);
         void ComputeSolNotClear(std::vector<Type>&) ;
     private :
         /// Must indicate for point Pt the value on all elements of bases by filling VOut, which is already sized
         virtual void ComputeValFuncsBase(tVecOut & aVOut,const tPtIn & aPt) = 0;
         size_t             mNbFunc;
         cLeasSqtAA<Type>   mLSQ;
         cDenseVect<Type>   mCoeffs;
         tVecOut            mBufPOut;
};

template <class Type,const int DimIn,const int DimOut>
    class cLeastSqCompMapCalcSymb : public cLeastSqComputeMaps<Type,DimIn,DimOut>
{
    public :
         typedef  cLeastSqComputeMaps<Type,DimIn,DimOut>  tSuper;
         using typename tSuper::tPtOut;
         using typename tSuper::tPtIn;
         using typename tSuper::tVecIn;
         using typename tSuper::tVecOut;

         typedef typename NS_SymbolicDerivative::cCalculator<Type> tCalc;
         cLeastSqCompMapCalcSymb(tCalc *);  ///< create from a calculator : Not Adopted for deletion
         virtual ~cLeastSqCompMapCalcSymb();
         void ComputeValFuncsBase(tVecOut &,const tPtIn & aPt) override;
    private :
         cLeastSqCompMapCalcSymb(const cLeastSqCompMapCalcSymb<Type,DimIn,DimOut>&)=delete;

         tCalc * mCalc;
         std::vector<Type> mVUk;
         std::vector<Type> mVObs;
};



/**  Bijective Affine Mapping Elementary */

template <class Type,const int Dim> class cBijAffMapElem
{
     public :
        typedef Type  tTypeElem;
        static constexpr int TheDim=Dim;
        typedef cBijAffMapElem<Type,Dim> tTypeMapInv;
        static constexpr int NbDOF() {return Dim * (1+Dim);}

        typedef cDenseMatrix<Type> tMat;
        typedef cPtxd<Type,Dim>    tPt;
        cBijAffMapElem(const tMat & aMat ,const tPt& aTr) ;

        tPt  Value(const tPt & aP)   const;
        tPt  Inverse(const tPt & aP) const;

        cBijAffMapElem<Type,Dim>  MapInverse() const;

     private :
        tMat  mMat;
        tPt   mTr;
        tMat  mMatInv;
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
