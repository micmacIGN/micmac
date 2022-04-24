#ifndef  _MMVII_SysSurR_H_
#define  _MMVII_SysSurR_H_
namespace MMVII
{

/** \file MMVII_SysSurR.h
    \brief Classes for linear redundant system
*/


/**  class for communinication  input and ouptut of equations in 
   cResolSysNonLinear
 */
template <class Type> class cInputOutputRSNL
{
     public :
          typedef std::vector<Type>  tStdVect;
          typedef std::vector<int>   tVectInd;

	  cInputOutputRSNL(const tVectInd&,const tStdVect & aVObs);
	  cInputOutputRSNL(const tVectInd&,const tStdVect &aVTmp,const tStdVect & aVObs);

	  Type WeightOfKthResisual(int aK) const;
	  bool IsOk() const;
	  size_t NbUkTot() const;

          tVectInd   mVInd;    ///<  index of unknown in the system
          tStdVect   mTmpUK;   ///< possible value of temporary unknown,that would be eliminated by schur complement
          tStdVect   mObs;     ///< Observation (i.e constants)

          tStdVect                mWeights;  ///< Weights of eq, size can equal mVals or be 1 (cste) or 0 (all 1.0) 
          tStdVect                mVals;     ///< values of fctr, i.e. residuals
          std::vector<tStdVect>   mDers;     ///< derivate of fctr

};

template <class Type> class cSetIORSNL_SameTmp
{
	public :
            typedef cInputOutputRSNL<Type>  tIO_OneEq;
	    typedef std::vector<tIO_OneEq>  tIO_AllEq;

	    cSetIORSNL_SameTmp();

	    void AddOneEq(const tIO_OneEq &);
	    const tIO_AllEq & AllEq() const;
	    /// To be Ok must have at least 1 eq, and number of eq must be >= to unkwnonw
	    void  AssertOk() const;

	private :
	    tIO_AllEq        mVEq;
	    bool             mOk;
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
       virtual void AddObsWithTmpK(const cSetIORSNL_SameTmp<Type>&);

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
       static cLeasSq<Type>*  AllocSparseNormalLstSq(int aNbVar,double aPerEmptyBuf=4.0);
       
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
       void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
       void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) override;
       void Reset() override;
       /// Compute a solution
       cDenseVect<Type>  Solve() override;
       /// Use  sparse cholesky 
       cDenseVect<Type>  SparseSolve() override ;
       // Accessor, at least for debug (else why ?)
       const cDenseMatrix<Type> & tAA   () const;
       const cDenseVect<Type>   & tARhs () const;
    private :
       cDenseMatrix<Type>  mtAA;    /// Som(W tA A)
       cDenseVect<Type>    mtARhs;  /// Som(W tA Rhs)
      
};





};

#endif  //  _MMVII_SysSurR_H_
