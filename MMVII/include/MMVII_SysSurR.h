#ifndef  _MMVII_SysSurR_H_
#define  _MMVII_SysSurR_H_
namespace MMVII
{

/** \file MMVII_SysSurR.h
    \brief Classes for linear redundant system
*/

/** Virtual base classe for solving an over resolved system of linear equation;
    Typical derived classes can be :
        * l1 sys using barodale methods
        * least square using covariance system
        * least square not using covariance ...
*/

template <class Type> class cSysSurResolu
{
    public :
       cSysSurResolu(int aNbVar);
       /// Virtual methods => virtaul ~X()
       virtual ~cSysSurResolu();
       /// Add  aPds (  aCoeff .X = aRHS) 
       virtual void AddObservation(const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) = 0;
       /// Add  aPds (  aCoeff .X = aRHS) , version sparse
       virtual void AddObservation(const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS) = 0;
       /// "Purge" all accumulated equations
       virtual void Reset() = 0;
       /// Compute a solution
       virtual cDenseVect<Type>  Solve() = 0;

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


    private :
       int mNbVar;
};

/** Not sure the many thing common to least square system, at least there is the
    residual (sum of square ...)
*/
template <class Type> class  cLeasSq  :  public cSysSurResolu<Type>
{
    public :
       cLeasSq(int aNbVar);
       Type Residual(const cDenseVect<Type> & aVect,const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS) const override;
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
       // Accessor, at least for debug (else why ?)
       const cDenseMatrix<Type> & tAA   () const;
       const cDenseVect<Type>   & tARhs () const;
    private :
       cDenseMatrix<Type>  mtAA;    /// Som(W tA A)
       cDenseVect<Type>    mtARhs;  /// Som(W tA Rhs)
      
};

};

#endif  //  _MMVII_SysSurR_H_
