#include "include/V1VII.h"
#include "include/ext_stl/numeric.h"

namespace MMVII
{

double Average(const std::vector<double> &aVec)
{
    cWeightAv<tREAL8>  aWA;

    for (const auto & aVal : aVec)
       aWA.Add(1.0,aVal);

    return aWA.Average();
}

double NC_KthVal(std::vector<double> & aV, double aProportion)
{
   return ::KthValProp(aV,aProportion);
}


double Cst_KthVal(const std::vector<double> & aV, double aProportion)
{
     std::vector<double> aDup= aV;
     return NC_KthVal(aDup,aProportion);
}

template <class Type>  
    std::vector<Type>  V1RealRoots(const std::vector<Type> &  aVCoef, Type aTol,int aNbMaxIter)
{
    ElPolynome<Type>  aV1Pol((char*)0,aVCoef.size()-1);
    for (size_t aK=0 ; aK<aVCoef.size() ; aK++)
	    aV1Pol[aK] = aVCoef[aK];

    std::vector<Type>  aVRoots;
    RealRootsOfRealPolynome(aVRoots,aV1Pol,aTol,aNbMaxIter);

    return aVRoots;
}

#define INST_V1ROOTS(TYPE)\
template std::vector<TYPE>  V1RealRoots(const std::vector<TYPE> &  aVCoef, TYPE aTol,int aNbMaxIter);

INST_V1ROOTS(tREAL4)
INST_V1ROOTS(tREAL8)
INST_V1ROOTS(tREAL16)


/*
template <class Type> void
     RealRootsOfRealPolynome
     (
         ElSTDNS vector<Type> &  Sols,
         const ElPolynome<Type>  &aPol,
         Type                    tol,
         INT                     ItMax
     )
     */


};
