#ifndef  _MMVII_SysSurR_H_
#define  _MMVII_SysSurR_H_

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Matrix.h"

namespace MMVII
{
/** \file MMVII_SysSurR.h
    \brief Classes for linear redundant system
*/

template <class Type> class  cInputOutputRSNL;
template <class Type> class  cSetIORSNL_SameTmp;
template <class Type> class  cLinearOverCstrSys  ;
template <class Type> class  cLeasSq ;
template <class Type> class  cLeasSqtAA ;
template <class Type> class  cBufSchurSubst;
template <class Type> class  cSetIORSNL_SameTmp;
template <class Type> class cResidualWeighter;
template <class Type> class cObjWithUnkowns;
template <class Type> class cSetInterUK_MultipeObj;

/// Index to use in vector of index indicating a variable to substituate
static constexpr int RSL_INDEX_SUBST_TMP = -1;


/**  Class for solving non linear system of equations
 */
template <class Type> class cResolSysNonLinear
{
      public :
          typedef tREAL8                                        tNumCalc;
          typedef NS_SymbolicDerivative::cCalculator<tNumCalc>  tCalc;
          typedef std::vector<tNumCalc>                         tStdCalcVect;
          typedef cInputOutputRSNL<Type>                        tIO_RSNL;
          typedef cSetIORSNL_SameTmp<Type>                      tSetIO_ST;


          typedef cLinearOverCstrSys<Type>                      tLinearSysSR;
          typedef cDenseVect<Type>                              tDVect;
          typedef cSparseVect<Type>                             tSVect;
          typedef std::vector<Type>                             tStdVect;
          typedef std::vector<int>                              tVectInd;
          typedef cResolSysNonLinear<Type>                      tRSNL;
          typedef cResidualWeighter<Type>                       tResidualW;
          typedef cObjWithUnkowns<Type>                         tObjWUk;

	  /// basic constructor, using a mode of matrix + a solution  init
          cResolSysNonLinear(eModeSSR,const tDVect & aInitSol);
	  ///  constructor  using linear system, allow finer control
          cResolSysNonLinear(tLinearSysSR *,const tDVect & aInitSol);
	  /// destructor 
          ~cResolSysNonLinear();

          /// Accessor
          const tDVect  &    CurGlobSol() const;
   
          /// Accessor
          int NbVar() const;  
          /// Value of a given num var
          const Type  &    CurSol(int aNumV) const;
          /// Set value, usefull for ex in dev-mesh because variable are activated stepby step
          void SetCurSol(int aNumV,const Type&) ;

          tLinearSysSR *  SysLinear() ;

          /// Solve solution,  update the current solution, Reset the least square system
          const tDVect  &    SolveUpdateReset() ;

          /// Add 1 equation fixing variable
          void   AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight);
          /// Add equation to fix variable to current value
          void   AddEqFixCurVar(const int & aNumV,const Type& aWeight);

          /// Basic Add 1 equation , no bufferistion, no schur complement
          void   CalcAndAddObs(tCalc *,const tVectInd &,const tStdVect& aVObs,const tResidualW & = tResidualW());

          /**  Add 1 equation in structure aSetIO ,  who will accumulate all equation of a given temporary set of unknowns
	       relatively basic 4 now because don't use parallelism of tCalc
	  */
          void  AddEq2Subst (tSetIO_ST & aSetIO,tCalc *,const tVectInd &,
                             const tStdVect& aVObs,const tResidualW & = tResidualW());

	  /** Once "aSetIO" has been filled by multiple calls to  "AddEq2Subst",  do it using for exemple schur complement
	   */
          void  AddObsWithTmpUK (const tSetIO_ST & aSetIO);

	       //    frozen  checking

	   void  SetFrozenVar(int aK,const  Type &);  ///< seti var var frozen /unfrozen
	   void  SetFrozenVarCurVal(int aK);  ///< idem to current val
	       // frozen for  cObjWithUnkowns
	   void  SetFrozenVar(tObjWUk & anObj,const  Type & aVal);  ///< Froze the value aVal, that must belong to anObj
	   void  SetFrozenVar(tObjWUk & anObj,const  Type * Vals,size_t aNb);  ///< Froze Nb values aVal, that must belong to anObj
	   void  SetFrozenVar(tObjWUk & anObj,const tStdVect & aVect);  ///< Froze aVect, that must belong to anObj
	   void  SetFrozenVar(tObjWUk & anObj,const cPtxd<Type,3> & aPt);  ///< Froze aPt that must belong to anObj
	   void  SetFrozenVar(tObjWUk & anObj,const cPtxd<Type,2> & aPt);  ///< Froze aPt that must belong to anObj
	   void  SetFrozenAllCurrentValues(tObjWUk & anObj);  ///< Froze all the value beloning to an anObj

           void AddObservationLinear(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS)  ;
           void AddObservationLinear(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) ;


	   void  SetUnFrozen(int aK);  ///< indicate it var must be frozen /unfrozen
	   void  UnfrozeAll() ;                       ///< indicate it var must be frozen /unfrozen
	   bool  VarIsFrozen(int aK) const;           ///< indicate it var must be frozen /unfrozen
	   int   CountFreeVariables() const;          ///< number of free variables
	   void  AssertNotInEquation() const;         ///< verify that we are notin equation step (to allow froze modification)
	   int   GetNbObs() const;                    ///< get number of observations (last iteration if after reset, or current number if after AddObs)

     private :
          cResolSysNonLinear(const tRSNL & ) = delete;

	  ///  Modify equations to take into account var is frozen
	  void  ModifyFrozenVar (tIO_RSNL&);

          /// Add observations as computed by CalcVal
          void   AddObs(const std::vector<tIO_RSNL>&);

          /** Bases function of calculating derivatives, dont modify the system as is
              to avoid in case  of schur complement */
          void   CalcVal(tCalc *,std::vector<tIO_RSNL>&,const tStdVect & aVTmp,bool WithDer,const tResidualW & );

          int        mNbVar;       ///< Number of variable, facility
          tDVect     mCurGlobSol;  ///< Curent solution
          tLinearSysSR*    mSysLinear;         ///< Sys to solve equations, equation are concerning the differences with current solution

	  bool                 mInPhaseAddEq;      ///< check that dont modify val fixed after adding  equations
	  std::vector<bool>    mVarIsFrozen;       ///< indicate for each var is it is frozen
	  std::vector<Type>    mValueFrozenVar;    ///< indicate for each var the possible value where it is frozen
	  int lastNbObs;                           ///< number of observations of last solving
	  int currNbObs;                           ///< number of observations currently added
};

/**  Class for weighting residuals : compute the vector of weight from a 
     vector of residual; default return {1.0,1.0,...}
 */
template <class Type> class cResidualWeighter
{
       public :
            typedef std::vector<Type>     tStdVect;

            cResidualWeighter(const Type & aVal=1.0);
            virtual tStdVect WeightOfResidual(const tStdVect &) const;
       private :
             Type mVal;

};


/**  class for communinication  input and ouptut of equations in 
   cResolSysNonLinear
 */
template <class Type> class cInputOutputRSNL
{
     public :
          typedef std::vector<Type>  tStdVect;
          typedef std::vector<int>   tVectInd;

	  /// Create Input data w/o temporay
	  cInputOutputRSNL(const tVectInd&,const tStdVect & aVObs);
	  /// Create Input data with temporary temporay
	  // cInputOutputRSNL(const tVectInd&,const tStdVect &aVTmp,const tStdVect & aVObs);

	  /// Give the weight, handle case where mWeights is empty (1.0) or size 1 (considered constant)
	  Type WeightOfKthResisual(int aK) const;
	  /// Check all the size are coherent
	  bool IsOk() const;
	  ///  Real unknowns + Temporary
	  size_t NbUkTot() const;

          tVectInd   mGlobVInd;    ///<  index of unknown in the system + TMP (with -1)   mVIndGlob
          tStdVect   mVObs;     ///< Observation (i.e constants)
          tStdVect                mWeights;  ///< Weights of eq, size can equal mVals or be 1 (cste) or 0 (all 1.0) 
          tStdVect                mVals;     ///< values of fctr, i.e. residuals
          std::vector<tStdVect>   mDers;     ///< derivate of fctr
	  size_t                  mNbTmpUk;

};

/**  class for grouping all the observation relative to a temporary variable,
     this is necessary because all must be processed simultaneously in schur elimination

     Basically this only a set of "cInputOutputRSNL"
 */
template <class Type> class cSetIORSNL_SameTmp
{
	public :
            typedef cInputOutputRSNL<Type>  tIO_OneEq;
	    typedef std::vector<tIO_OneEq>  tIO_AllEq;
            typedef std::vector<Type>  tStdVect;
            typedef std::vector<int>   tVectInd;

	    /** Constructor :  take value of tmp+ optional vector of fixed var (given wih negatives value like {-1,-2})
	        if aValFix is not given, default is current value
		It can be strange to fix value of tmp var, but it's usefull for example if we want to use the same
		equation with unknown variable and fix value. This is the case in bundle adj when the same colinearity 
		equation can be use with tie-point or with a known GCP.
	     */
	    cSetIORSNL_SameTmp(const tStdVect & aValTmpUk,const tVectInd & aVFix={},const tStdVect & aValFix ={});

            /// Force tmp to its current value,  aNum is the same index (negative) than in AddEq2Subst
            void  AddFixCurVarTmp (int aNum,const Type& aWeight);
            /// Force tmp to a different value, probably not usefull in real (if we have a better value, why not use it), execpt for test
            void  AddFixVarTmp (int aNum,const Type& aVal,const Type& aWeight);

	    /// Add and equation,  check the coherence
	    void AddOneEq(const tIO_OneEq &);
	    ///  Accesor to all equations
	    const tIO_AllEq & AllEq() const;
	    /// To be Ok must have at least 1 eq, and number of eq must be >= to unkwnonw
	    void  AssertOk() const;

	    ///  Number of temporary unkown
	    size_t  NbTmpUk() const;
	    const tStdVect & ValTmpUk() const;
	    Type  Val1TmpUk(int aInd) const; ///make the correction of negativeness

	    static size_t ToIndTmp(int ) ;
	    static bool   IsIndTmp(int ) ;

	private :
	    cSetIORSNL_SameTmp(const cSetIORSNL_SameTmp&) = delete;
	    tIO_AllEq        mVEq;

	    bool               mOk;
	    size_t             mNbTmpUk;
	    tStdVect           mValTmpUk;
	    std::vector<bool>  mVarTmpIsFrozen;       ///< indicate for each temporary var if it is frozen
	    tStdVect           mValueFrozenVarTmp;    ///< value of frozen tmp
	    size_t             mNbEq;
            cSetIntDyn         mSetIndTmpUk;
};


/** Virtual base classe for solving an over resolved system of linear equation;
    Typical derived classes can be :
        * l1 sys using barodale methods
        * least square using covariance system
        * least square not using covariance ...
*/

template <class Type> class cLinearOverCstrSys  : public cMemCheck
{
    public :
       ///  Basic Cstr
       cLinearOverCstrSys(int aNbVar);
       ///  static allocator
       static cLinearOverCstrSys<Type> * AllocSSR(eModeSSR,int aNbVar);



       /// Virtual methods => virtaul ~X()
       virtual ~cLinearOverCstrSys();
       /// Add  aPds (  aCoeff .X = aRHS) 
       virtual void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) = 0;
       /// Add  aPds (  aCoeff .X = aRHS) , version sparse
       virtual void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) = 0;

       /**  This the method for adding observation with temporaray unknown, the class can have various answer
	     -  eliminate the temporay via schur complement
             - treat temporary as unknowns and increase the size of their unknowns
	     - refuse to process =>default is error ...
	*/
       virtual void AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&);

       /// "Purge" all accumulated equations
       virtual void Reset() = 0;
       /// Compute a solution
       virtual cDenseVect<Type>  Solve() = 0;
       ///  May contain a specialization for sparse system, default use generik
       virtual cDenseVect<Type>  SparseSolve() ;

       /** Return for a given "solution" the weighted residual of a given observation
           Typically can be square, abs ....
           Usefull for bench at least (check that solution is minimum, or least < to neighboor)
        */
       
       virtual Type Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const = 0;
       
       //  ============ Fix value of variable =============
            ///  Fix value of curent variable, 1 variable
       virtual void AddObsFixVar(const Type& aWeight,int aIndVal,const Type & aVal);
            ///  Fix value of curent variable, N variable
       virtual void AddObsFixVar(const Type& aWeight,const cSparseVect<Type> & aVVarVals);
            ///  Fix value of curent variable, All variable
       virtual void AddObsFixVar(const Type& aWeight,const cDenseVect<Type>  &  aVRHS); 

       /// Accessor
       int NbVar() const;

      /// Normal Matrix defined 4 now only in cLeasSqtAA, maybe later defined in other classe, else error
      virtual cDenseMatrix<Type>  V_tAA() const;
      /// Idem  "normal" vector
      virtual cDenseVect<Type>    V_tARhs() const;  
      /// Indicate if it gives acces to these normal "stuff"
      virtual bool   Acces2NormalEq() const;  


      virtual void   AddCov(const cDenseMatrix<Type> &,const cDenseVect<Type>& ,const std::vector<int> &aVInd);

    protected :
       int mNbVar;
};

/** Class for fine parametrisation  allocation of normal sparse system */

class cParamSparseNormalLstSq
{
      public :
          cParamSparseNormalLstSq();
          cParamSparseNormalLstSq(double aPerEmptyBuf,size_t aNbMaxRangeDense,size_t aNbBufDense);

	  /// Def=4,  equation are buffered "as is" and at some frequency put in normal matrix
	  double mPerEmptyBuf;
	  /** Def={} , even with sparse system, it can happen that a small subset of variable are better handled as dense one, 
	    typically it could be the intrinsic parameters in bundle */
	  std::vector<size_t> mVecIndDense;

	  /** Def=0 ; it is recommandend that mIndDense correpond to low index, in the case where they are in fact range [0,N]
	       mIndMaxRangeDense allow an easier parametrization 
	   */
	  size_t mIndMaxRangeDense;

	  /** Def=13 the systeme can maintain a certain number of non dense variable in temporary  dense mode, def is purely arbitrary ...*/
	  size_t mNbBufDense;
};

/** Not sure the many thing common to least square system, at least there is the
    residual (sum of square ...)
*/
template <class Type> class  cLeasSq  :  public cLinearOverCstrSys<Type>
{
    public :
       cLeasSq(int aNbVar);
       Type Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const override;
       
       /// Dont use normal equation 
       static cLeasSq<Type>*  AllocSparseGCLstSq(int aNbVar);
       
       /// Adpated to"very sparse" system like in finite element, probably also ok for photogrammetry
       static cLeasSq<Type>*  AllocSparseNormalLstSq(int aNbVar,const cParamSparseNormalLstSq & aParam);
       
       /// Basic dense systems  => cLeasSqtAA
       static cLeasSq<Type>*  AllocDenseLstSq(int aNbVar);


};

/**  Implemant least by suming tA A ,  simple and efficient, by the way known to have
  a conditionning problem 
*/

template <class Type> class  cLeasSqtAA  :  public cLeasSq<Type>
{
    public :
       cLeasSqtAA(int aNbVar);
       cLeasSqtAA<Type> Dup() const;

       virtual ~cLeasSqtAA();
       void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
       void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;
       void Reset() override;
       /// Compute a solution
       cDenseVect<Type>  Solve() override;
       /// Use  sparse cholesky , usefull for "sparse dense" system ...
       cDenseVect<Type>  SparseSolve() override ;

       void AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&) override;

       //  ================  Accessor used in Schur elim ========  :
       
       const cDenseMatrix<Type> & tAA   () const;   ///< Accessor 
       const cDenseVect<Type>   & tARhs () const;   ///< Accessor 
       cDenseMatrix<Type> & tAA   () ;         ///< Accessor 
       cDenseVect<Type>   & tARhs () ;         ///< Accessor 

      /// access to tAA via virtual interface
      cDenseMatrix<Type>  V_tAA() const override;
      /// access to tARhs via virtual interface
      cDenseVect<Type>    V_tARhs() const override;  
      /// true because acces is given
      bool   Acces2NormalEq() const override;  

      void   AddCov(const cDenseMatrix<Type> &,const cDenseVect<Type>& ,const std::vector<int> &aVInd) override;
    private :
       cDenseMatrix<Type>  mtAA;    /// Som(W tA A)
       cDenseVect<Type>    mtARhs;  /// Som(W tA Rhs)
       cBufSchurSubst<Type> * mBSC;
      
};


/**  Class for compute elimination of temporay equation using schur complement.

     For vector and matrix, We use notation of MMV1 documentation   L= Lamda

     (L      B   .. )     (X)   (A )
     (tB     M11 .. )  *  (Y) = (C1)   =>     (M11- tB L-1 B    ...)  (Y) = C1 - tB L-1 A
       ...                (Z)    ..           (                 ...)  (Z)
*/


template <class Type> class  cBufSchurSubst
{
     public :
          typedef cSetIORSNL_SameTmp<Type>  tSetEq;

          /// constructor , just alloc the vector to compute subset of unknosn
          cBufSchurSubst(size_t aNbVar);
          /// Make the computation from the set of equations
          void CompileSubst(const tSetEq &);

          //  ==== 3 accessors to the result

          /// return normal matrix after schur substitution ie : M11- tB L-1 B
          const cDenseMatrix<Type> & tAASubst() const;
          ///  return normal vector after schur subst  ie : C1 - tB L-1 A
          const cDenseVect<Type> & tARhsSubst() const;
          ///  return list of indexes  used to put compress mat/vect in "big" matrixes/vect
          const std::vector<size_t> & VIndexUsed() const;
     private :

          size_t            mNbVar;
          cSetIntDyn        mSetInd;
          cLeasSqtAA<Type>  mSysRed;
          cSparseVect<Type> mSV;
          size_t            mNbTmp;
          size_t            mNbUk;
          size_t            mNbUkTot;

          cDenseMatrix<Type> mL;
          // cDenseMatrix<Type> mLInv;
          cDenseMatrix<Type> mtB;
          cDenseMatrix<Type> mtB_LInv;
          cDenseMatrix<Type> mLInv_B;
          cDenseMatrix<Type> mB;
          cDenseMatrix<Type> mtB_LInv_B;
          cDenseMatrix<Type> mM11;

          cDenseVect<Type>   mA;
          cDenseVect<Type>   mC1;
          cDenseVect<Type>   mtB_LInv_A;
};


    /*  =======================   classes used for facilitating numerotation of unkwons ================= 
     *
     *   cObjWithUnkowns     ->  base-class for object having  unknows 
     *
     *   cOneInteralvUnkown  ->  an interval of Type*  , simply a pair  Type* - int
     *   cSetIntervUK_OneObj -> all the interval of an object  + the object itself
     *   cSetInterUK_MultipeObj -> all the object interacting in one system
     *
     * =================================================================================================== */

template <class Type> class cOneInteralvUnkown;
template <class Type> class cSetIntervUK_OneObj;
template <class Type> class cSetInterUK_MultipeObj;
template <class Type> class cObjWithUnkowns;

/*  Typical scenario
 
     //  for object having unknowns, make them inherit of cObjWithUnkowns, describe behaviour with PutUknowsInSetInterval
     class  cObj: public cObjWithUnkowns
     {
	  double mUK1[4];  // first set of unknowns
	  int    mToto;
	  double mUK2[7];  // secon set unk
       
	   ...  do stuff specific to cObj ...

          void PutUknowsInSetInterval() override 
	  {
	       mSetInterv->AddOneInterv(mUK1,4);
	       mSetInterv->AddOneInterv(mUK2,7);
	  }
     };

     {
        cObj aO1,aO2;
        cSetInterUK_MultipeObj<Type>  aSet;    //  create the object
        aSet.AddOneObj(aO1); // in this call aSet will call O1->PutUknowsInSetInterval()
        aSet.AddOneObj(aO2);

	// create a sys with the vector of all unkwnon
	cResolSysNonLinear<double> * aSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv.GetVUnKnowns());


	const auto & aVectSol = mSys->SolveUpdateReset();
	// modify all unkowns with new solution, call the method OnUpdate in case object have something to do
        mSetInterv.SetVUnKnowns(aVectSol);
     }

*/



///  represent one interval of consecutive unkown
template <class Type> class cOneInteralvUnkown
{
     public :
        Type * mVUk;
        size_t mNb;
        cOneInteralvUnkown(Type * aVUk,size_t aNb)  : mVUk (aVUk) , mNb (aNb) {}
};

///  represent all the unkown interval of one object
template <class Type> class cSetIntervUK_OneObj
{
     public :
         cSetIntervUK_OneObj(cObjWithUnkowns<Type>   *anObj) : mObj (anObj) {}

         cObjWithUnkowns<Type>   *         mObj;
         std::vector<cOneInteralvUnkown<Type>>   mVInterv;

};

///  represent all the object with unknown of a given system of equation

template <class Type> class cSetInterUK_MultipeObj
{
        public :
           friend class cObjWithUnkowns<Type>;

           cSetInterUK_MultipeObj(); /// constructor, init mNbUk
           ~cSetInterUK_MultipeObj();  /// indicate to all object that they are no longer active

	   /// This method is used to add the unknowns of one object
           void  AddOneObj(cObjWithUnkowns<Type> *);

	   ///  return a DenseVect filled with all unknowns  as expected to create a cResolSysNonLinear
           cDenseVect<Type>  GetVUnKnowns() const;

	   ///  fills all unknown of object with a vector as created by cResolSysNonLinear::SolveUpdateReset()
           void  SetVUnKnowns(const cDenseVect<Type> &);

	        // different method for adding intervalls

           void AddOneInterv(Type * anAdr,size_t aSz) ; ///<  generall method
           void AddOneInterv(Type & anAdr) ;            ///<  call with a single value
           void AddOneInterv(std::vector<Type> & aV) ;  ///<  call previous with a vector
           void AddOneInterv(cPtxd<Type,2> &);          ///<  call previous wih a point
           void AddOneInterv(cPtxd<Type,3> &);          ///<  call previous wih a point
        private :

	   void Reset();  /// Maybe private later, now used for tricky destruction order

	   size_t IndOfVal(const cObjWithUnkowns<Type>&,const Type *) const;

           cSetInterUK_MultipeObj(const cSetInterUK_MultipeObj<Type> &) = delete;

           void IO_UnKnowns(cDenseVect<Type> & aV,bool isSetUK);

           std::vector<cSetIntervUK_OneObj<Type> >  mVVInterv;
           size_t                                    mNbUk;
};

template <class Type> class cObjWithUnkowns
{
       public :
          friend class cSetInterUK_MultipeObj<Type>;

	  /// defautl constructor, put non init in all vars
          cObjWithUnkowns();
	  ///  check that object is no longer referenced when destroyd
          virtual ~cObjWithUnkowns();
	  
          /// Fundamental methos :  the object put it sets on unknowns intervals  in the glob struct
          virtual void PutUknowsInSetInterval() = 0;

          /// This callbak method is called after update, used when modification of linear var is not enough (see cSensorCamPC)
          virtual void OnUpdate();

	  ///  Push in vector all the number of unknowns
          void PushIndexes(std::vector<int> &);

	  ///  indicate if the object has been initialized
          bool  UkIsInit() const;

	  /// recover the index from a value
	  size_t IndOfVal(const Type *) const;

          int   IndUk0() const;   ///< Accessor
          int   IndUk1() const;   ///< Accessor

       protected :
	  /// defautl constructor, put non init in all vars
          void Reset();
          cObjWithUnkowns(const cObjWithUnkowns<Type> &) = delete;


          cSetInterUK_MultipeObj<Type> *  mSetInterv;
          int   mNumObj;
          int   mIndUk0;
          int   mIndUk1;
};

};

#endif  //  _MMVII_SysSurR_H_
