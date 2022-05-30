#ifndef  _MMVII_SysSurR_H_
#define  _MMVII_SysSurR_H_
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
template <class Type> class  cBufSchurrSubst; 
template <class Type> class  cSetIORSNL_SameTmp;
template <class Type> class cResidualWeighter;

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


          typedef cLinearOverCstrSys<Type>                      tSysSR;
          typedef cDenseVect<Type>                              tDVect;
          typedef cSparseVect<Type>                             tSVect;
          typedef std::vector<Type>                             tStdVect;
          typedef std::vector<int>                              tVectInd;
          typedef cResolSysNonLinear<Type>                      tRSNL;
          typedef cResidualWeighter<Type>                       tResidualW;

	  /// basic constructor, using a mode of matrix + a solution  init
          cResolSysNonLinear(eModeSSR,const tDVect & aInitSol);
	  ///  constructor  using linear system, allow finer control
          cResolSysNonLinear(tSysSR *,const tDVect & aInitSol);
	  /// destructor 
          ~cResolSysNonLinear();

          /// Accessor
          const tDVect  &    CurGlobSol() const;
          /// Value of a given num var
          const Type  &    CurSol(int aNumV) const;

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
          void  AddEq2Subst (tSetIO_ST & aSetIO,tCalc *,const tVectInd &,const tStdVect& aVTmp,
                             const tStdVect& aVObs,const tResidualW & = tResidualW());
	  /** Once "aSetIO" has been filled by multiple calls to  "AddEq2Subst",  do it using for exemple schurr complement
	   */
          void  AddObsWithTmpUK (const tSetIO_ST & aSetIO);
     private :
          cResolSysNonLinear(const tRSNL & ) = delete;

          /// Add observations as computed by CalcVal
          void   AddObs(const std::vector<tIO_RSNL>&);

          /** Bases function of calculating derivatives, dont modify the system as is
              to avoid in case  of schur complement */
          void   CalcVal(tCalc *,std::vector<tIO_RSNL>&,bool WithDer,const tResidualW & );

          int        mNbVar;       ///< Number of variable, facility
          tDVect     mCurGlobSol;  ///< Curent solution
          tSysSR*    mSys;         ///< Sys to solve equations, equation are concerning the differences with current solution
};

/**  Class for weighting residuals : compute the vector of weight from a 
     vector of residual; default return {1.0,1.0,...}
 */
template <class Type> class cResidualWeighter
{
       public :
            typedef std::vector<Type>     tStdVect;

            cResidualWeighter();
            virtual tStdVect WeightOfResidual(const tStdVect &) const;
       private :

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
	  cInputOutputRSNL(const tVectInd&,const tStdVect &aVTmp,const tStdVect & aVObs);

	  /// Give the weight, handle case where mWeights is empty (1.0) or size 1 (considered constant)
	  Type WeightOfKthResisual(int aK) const;
	  /// Check all the size are coherent
	  bool IsOk() const;
	  ///  Real unknowns + Temporary
	  size_t NbUkTot() const;

          tVectInd   mVIndUk;    ///<  index of unknown in the system , no TMP
          tStdVect   mVTmpUK;   ///< possible value of temporary unknown,that would be eliminated by schur complement
          tVectInd   mVIndGlob;    ///<  index of unknown in the system + TMP (with -1)
          tStdVect   mVObs;     ///< Observation (i.e constants)

          tStdVect                mWeights;  ///< Weights of eq, size can equal mVals or be 1 (cste) or 0 (all 1.0) 
          tStdVect                mVals;     ///< values of fctr, i.e. residuals
          std::vector<tStdVect>   mDers;     ///< derivate of fctr

};

/**  class for grouping all the observation relative to a temporary variable,
     this is necessary because all must be processed simultaneously in schurr elimination

     Basically this only a set of "cInputOutputRSNL"
 */
template <class Type> class cSetIORSNL_SameTmp
{
	public :
            typedef cInputOutputRSNL<Type>  tIO_OneEq;
	    typedef std::vector<tIO_OneEq>  tIO_AllEq;

	    /// Constructor :  Create an empty set
	    cSetIORSNL_SameTmp();

	    /// Add and equation,  check the coherence
	    void AddOneEq(const tIO_OneEq &);
	    ///  Accesor to all equations
	    const tIO_AllEq & AllEq() const;
	    /// To be Ok must have at least 1 eq, and number of eq must be >= to unkwnonw
	    void  AssertOk() const;

	    ///  Number of temporary unkown
	    size_t  NbTmpUk() const;

	private :
	    tIO_AllEq        mVEq;
	    bool             mOk;
	    size_t           mNbEq;
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
       virtual ~cLeasSqtAA();
       void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
       void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;
       void Reset() override;
       /// Compute a solution
       cDenseVect<Type>  Solve() override;
       /// Use  sparse cholesky , usefull for "sparse dense" system ...
       cDenseVect<Type>  SparseSolve() override ;

       void AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&) override;

       //  ================  Accessor used in Schurr elim ========  :
       
       const cDenseMatrix<Type> & tAA   () const;   ///< Accessor 
       const cDenseVect<Type>   & tARhs () const;   ///< Accessor 
       cDenseMatrix<Type> & tAA   () ;         ///< Accessor 
       cDenseVect<Type>   & tARhs () ;         ///< Accessor 

    private :
       cDenseMatrix<Type>  mtAA;    /// Som(W tA A)
       cDenseVect<Type>    mtARhs;  /// Som(W tA Rhs)
       cBufSchurrSubst<Type> * mBSC;
      
};


/**  Class for compute elimination of temporay equation using schurr complement.

     For vector and matrix, We use notation of MMV1 documentation   L= Lamda

     (L      B   .. )     (X)   (A )
     (tB     M11 .. )  *  (Y) = (C1)   =>     (M11- tB L-1 B    ...)  (Y) = C1 - tB L-1 A
       ...                (Z)    ..           (                 ...)  (Z)
*/


template <class Type> class  cBufSchurrSubst
{
     public :
          typedef cSetIORSNL_SameTmp<Type>  tSetEq;

          /// constructor , just alloc the vector to compute subset of unknosn
          cBufSchurrSubst(size_t aNbVar);
          /// Make the computation from the set of equations
          void CompileSubst(const tSetEq &);

          //  ==== 3 accessors to the result

          /// return normal matrix after schurr substitution ie : M11- tB L-1 B  
          const cDenseMatrix<Type> & tAASubst() const;
          ///  return normal vector after schurr subst  ie : C1 - tB L-1 A
          const cDenseVect<Type> & tARhsSubst() const;
          ///  return list of indexes  used to put compress mat/vect in "big" matrixes/vect
          const std::vector<size_t> & VIndexUsed() const;
     private :

          size_t            mNbVar;
          std::vector<size_t>  mNumComp;
          cSetIntDyn        mSetInd;
          cLeasSqtAA<Type>  mSysRed;
          cSparseVect<Type> mSV;
          size_t            mNbTmp;
          size_t            mNbUk;
          size_t            mNbUkTot;

          cDenseMatrix<Type> mL;
          cDenseMatrix<Type> mLInv;
          cDenseMatrix<Type> mtB;
          cDenseMatrix<Type> mtB_LInv;
          cDenseMatrix<Type> mB;
          cDenseMatrix<Type> mtB_LInv_B;
          cDenseMatrix<Type> mM11;

          cDenseVect<Type>   mA;
          cDenseVect<Type>   mC1;
          cDenseVect<Type>   mtB_LInv_A;
};





};

#endif  //  _MMVII_SysSurR_H_
