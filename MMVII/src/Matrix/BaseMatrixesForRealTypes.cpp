#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Tpl_Images.h"

namespace MMVII
{

template <class Type>  cDenseMatrix<Type>  cDenseVect<Type>::MatLineOfVect(const std::vector<tDV>  & aVVect)
{
    int aDimV = AllDimComon(aVVect);
    int aDimMat = std::max(aDimV,int(aVVect.size()));
    cDenseMatrix<Type> aMat(aDimMat,eModeInitImage::eMIA_Null);
    for (size_t aKY=0 ; aKY<aVVect.size() ; aKY++)
        aMat.WriteLine(aKY,aVVect.at(aKY),true);

    return aMat;
}

template <class Type>  Type  cDenseVect<Type>::DegenDegree(const std::vector<tDV>  & aVVect) 
{
     cDenseMatrix<Type> aMat = MatLineOfVect(aVVect); 
     cResulSVDDecomp<Type> aSVD = aMat.SVD();
     int aDimMin = std::min((int)aVVect.size(),AllDimComon(aVVect));
    
     // singular values are >=0 and decreasing

     return aSVD.SingularValues()(aDimMin-1);
}

template <class Type> std::vector<cDenseVect<Type>>  
       cDenseVect<Type>::GenerateVectNonColin(int aDim,int aNbVect,tREAL8 aMaxDeg)
{
    for (int aKTry=0; aKTry<10000 ; aKTry++)
    {
        std::vector<cDenseVect<Type>>  aRes;
        for (int aKV=0 ; aKV<aNbVect ; aKV++)
            aRes.push_back(cDenseVect<Type>(aDim,eModeInitImage::eMIA_RandCenter));
        if (DegenDegree(aRes) > aMaxDeg)
           return aRes;
    }

    MMVII_INTERNAL_ERROR("Too many test in GenerateVectNonColin");
    return {};
}



 
#define INSTANTIATE_DENSE_VECT(Type)\
template  class  cDenseVect<Type>;

INSTANTIATE_DENSE_VECT(tREAL4)
INSTANTIATE_DENSE_VECT(tREAL8)
INSTANTIATE_DENSE_VECT(tREAL16)


};


/* ========================== */
/*          cMatrix           */
/* ========================== */

