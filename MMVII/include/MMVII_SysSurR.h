#ifndef  _MMVII_SysSurR_H_
#define  _MMVII_SysSurR_H_
namespace MMVII
{

/** \file MMVII_SysSurR.h
    \brief Classes for matrix manipulation, 
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
       /// Add  aPds (  aCoeff .X = aRHS) 
       virtual void AddObservation(const Type& aPds,const cDenseVect<Type> & aCoeff,const Type &  aRHS) = 0;
       /// "Purge" all accumulated equations
       virtual void Reset() = 0;
       /// Compute a solution
       virtual cDenseVect<Type>  Solve() = 0;
    private :
       int mNbVar;
};

template <class Type> class  cLeasSqtAA  :  public cSysSurResolu<Type>
{
    public :
       cLeasSqtAA(int aNbVar);
       void AddObservation(const Type& aPds,const cDenseVect<Type> & aCoeff,const Type &  aRHS) override;
       void Reset() override;
       /// Compute a solution
       cDenseVect<Type>  Solve() override;
    private :
       cDenseMatrix<Type>  mtAA;
       cDenseVect<Type>    mtARhs;
      
};

};

#endif  //  _MMVII_SysSurR_H_
