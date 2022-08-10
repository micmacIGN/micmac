#include "include/MMVII_all.h"

namespace MMVII
{

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


/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/

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

