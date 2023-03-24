#include "MMVII_enums.h"
#include "MMVII_util_tpl.h"


namespace MMVII
{
/* ********************************************* */
/*                                               */
/*             cBijectiveMapI2O                  */
/*                                               */
/* ********************************************* */

template <class Type>
   int cBijectiveMapI2O<Type>::Add(const Type & anObj,bool OkExist)
{
    if (mObj2I.find(anObj) != mObj2I.end())
    {
       MMVII_INTERNAL_ASSERT_tiny(OkExist,"cBijectiveMapI2O multiple add");
       return -1;
    }

    size_t aNum = mI2Obj.size();
    mObj2I[anObj] = aNum;
    mI2Obj.push_back(anObj);

    return aNum;
}

template <class Type>
   Type * cBijectiveMapI2O<Type>::I2Obj(int anInd)
{
   if ( (anInd<0) || (anInd>=int(mObj2I.size())) )
      return nullptr;

   return & mI2Obj.at(anInd);
}


template <class Type>
   int  cBijectiveMapI2O<Type>::Obj2I(const Type & anObj,bool SVP)
{
    const auto & anIt = mObj2I.find(anObj) ;

    if (anIt== mObj2I.end())
    {
        MMVII_INTERNAL_ASSERT_tiny(SVP,"Obj2I : object dont exist");
        return -1;
    }

    return anIt->second;
}

template class cBijectiveMapI2O<std::string>;

/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/

template <class Type> int LexicoCmp(const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
    int aSz = std::min(aV1.size(),aV2.size());
    for (int aK=0 ; aK<aSz ; aK++)
    {
        if (aV1.at(aK) < aV2.at(aK))  return -1;
        if (aV1.at(aK) > aV2.at(aK))  return  1;
    }
    if (aV1.size() < aV2.size()) return -1;
    if (aV1.size() > aV2.size()) return 1;
    return 0;
}
template <class Type> bool operator < (const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   return LexicoCmp(aV1,aV2) == -1;
}
template <class Type> bool operator == (const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   return LexicoCmp(aV1,aV2) == 0;
}
template <class Type> bool operator != (const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   return LexicoCmp(aV1,aV2) != 0;
}



#define INSTANTIATE_LEXICO_COMP(TYPE)\
template int LexicoCmp(const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);\
template bool operator < (const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);\
template bool operator ==(const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);\
template bool operator !=(const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);

INSTANTIATE_LEXICO_COMP(size_t);
INSTANTIATE_LEXICO_COMP(int);
INSTANTIATE_LEXICO_COMP(double);
INSTANTIATE_LEXICO_COMP(eModeCaracMatch);
INSTANTIATE_LEXICO_COMP(std::string);

};

