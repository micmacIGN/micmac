#include "MMVII_enums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"


namespace MMVII
{
/* ********************************************* */
/*                                               */
/*             cBijectiveMapI2O                  */
/*                                               */
/* ********************************************* */

template <class Type>
   int cBijectiveMapI2O<Type>::Add(const Type & anObj,bool OkExist,const std::string & aMsgError)
{
    if (mObj2I.find(anObj) != mObj2I.end())
    {
       if  (!OkExist)
       {
           MMVII_INTERNAL_ASSERT_tiny(false,"cBijectiveMapI2O multiple add : " + aMsgError);
       }
       return -1;
    }

    size_t aNum = mI2Obj.size();
    mObj2I[anObj] = aNum;
    mI2Obj.push_back(anObj);

    return aNum;
}

template <class Type> size_t  cBijectiveMapI2O<Type>::size() const
{
    return mI2Obj.size();
}


template <class Type>
   const Type * cBijectiveMapI2O<Type>::I2Obj(int anInd,bool SVP) const
{
   if ( (anInd<0) || (anInd>=int(mObj2I.size())) )
   {
      MMVII_INTERNAL_ASSERT_tiny(SVP,"I2Obj : object dont exist");
      return nullptr;
   }

   return & mI2Obj.at(anInd);
}

template <class Type>
   Type * cBijectiveMapI2O<Type>::I2Obj(int anInd,bool SVP) 
{
   if ( (anInd<0) || (anInd>=int(mObj2I.size())) )
   {
      MMVII_INTERNAL_ASSERT_tiny(SVP,"I2Obj : object dont exist");
      return nullptr;
   }

   return & mI2Obj.at(anInd);
}

template <class Type>
   int  cBijectiveMapI2O<Type>::Obj2I(const Type & anObj,bool SVP) const
{
    const auto & anIt = mObj2I.find(anObj) ;

    if (anIt== mObj2I.end())
    {
        if (! SVP)
	{
              MMVII_INTERNAL_ASSERT_tiny(SVP,"Obj2I : object dont exist,  Obj=[" +ToStr(anObj) + "]");
	}
        return -1;
    }

    return anIt->second;
}

template <class Type>
   int  cBijectiveMapI2O<Type>::Obj2IWithMsg(const Type & anObj,const std::string & aMesg) const
{
    int aK = Obj2I(anObj,true);

    if (aK<0)
    {
         StdOut()  << "For context : " << aMesg << std::endl;
         MMVII_INTERNAL_ASSERT_tiny(false,"Obj2I : object dont exist ===> [ " + aMesg + "]" );
    }

    return aK;
}

template class cBijectiveMapI2O<std::string>;

/****************************************************/
/*                                                  */
/*                                                  */
/*                                                  */
/****************************************************/

template <class Type> int IterLexicoCmp(Type  aB1,Type  aE1,Type  aB2,Type  aE2)
{
    // for(auto It1=aB1,It2=aB2 ; (It1!=aE1) && (It2!=aE2) ; It1++,It2++)
    for(; (aB1!=aE1) && (aB2!=aE2) ; aB1++,aB2++)
    {
        if ( *aB1 < *aB2)  return -1;
        if (*aB1 > *aB2)  return  1;
    }

    if ((aB1==aE1) && (aB2!=aE2))
       return -1;

    if ((aB1!=aE1) && (aB2==aE2))
       return 1;

    return 0;
}

template <class Type> int bLexicoCmp(const std::vector<Type> & aV1,const std::vector<Type> & aV2)
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

template <class Type> int VecLexicoCmp(const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
    int aR0 = IterLexicoCmp(aV1.begin(),aV1.end(),aV2.begin(),aV2.end());
/*
    int aR1 = bLexicoCmp(aV1,aV2);
    std::cout << "llLexicoCmp" << aR0 << aR1 << "\n";
    MMVII_INTERNAL_ASSERT_tiny(aR0==aR1,"cBijectiveMapI2O multiple add");
*/
    return aR0;
}


template <class Type> bool operator < (const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   return VecLexicoCmp(aV1,aV2) == -1;
}
template <class Type> bool operator == (const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   return VecLexicoCmp(aV1,aV2) == 0;
}
template <class Type> bool operator != (const std::vector<Type> & aV1,const std::vector<Type> & aV2)
{
   return VecLexicoCmp(aV1,aV2) != 0;
}

template <class Type,const int Dim>  bool operator < (const cPtxd<Type,Dim> & aP1, const cPtxd<Type,Dim> & aP2)
{
   return IterLexicoCmp(aP1.PtRawData(),aP1.PtRawData()+Dim,aP2.PtRawData(),aP2.PtRawData()+Dim) == -1;
}
template <class Type,const int Dim>  bool operator > (const cPtxd<Type,Dim> & aP1, const cPtxd<Type,Dim> & aP2)
{
   return IterLexicoCmp(aP1.PtRawData(),aP1.PtRawData()+Dim,aP2.PtRawData(),aP2.PtRawData()+Dim) ==  1;
}


#define INSTANTIATE_LEXICO_COMP(TYPE)\
template int VecLexicoCmp(const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);\
template bool operator < (const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);\
template bool operator ==(const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);\
template bool operator !=(const std::vector<TYPE> & aV1,const std::vector<TYPE> & aV2);

INSTANTIATE_LEXICO_COMP(size_t);
INSTANTIATE_LEXICO_COMP(int);
INSTANTIATE_LEXICO_COMP(double);
INSTANTIATE_LEXICO_COMP(eModeCaracMatch);
INSTANTIATE_LEXICO_COMP(std::string);

// INSTANTIATE_LEXICO_COMP(tPt2dr);
#define VIRGULE ,

// INSTANTIATE_LEXICO_COMP(cPtxd<tREAL8 VIRGULE 3>);

#define INSTANTIATE_CMP_TYPE_DIM(TYPE,DIM)\
INSTANTIATE_LEXICO_COMP(std::vector<cPtxd<TYPE VIRGULE DIM>>);\
INSTANTIATE_LEXICO_COMP(cPtxd<TYPE VIRGULE DIM>);\
template bool operator < (const cPtxd<TYPE,DIM> & aP1, const cPtxd<TYPE,DIM> & aP2);\
template bool operator > (const cPtxd<TYPE,DIM> & aP1, const cPtxd<TYPE,DIM> & aP2);

#define INSTANTIATE_CMP_TYPE(TYPE)\
INSTANTIATE_CMP_TYPE_DIM(TYPE,1)\
INSTANTIATE_CMP_TYPE_DIM(TYPE,2)\
INSTANTIATE_CMP_TYPE_DIM(TYPE,3)\

INSTANTIATE_CMP_TYPE(tREAL8)
};

