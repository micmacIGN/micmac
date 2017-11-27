#include "include/MMVII_all.h"
#include <map>

/** \file uti_e2string.cpp
    \brief Implementation enum <=> string conversion

    Probably sooner or later this file will be generated 
automatically. Waiting for that, I do it quick and dirty
by hand.
*/


namespace MMVII
{
    /* =========================================== */
    /*                                             */
    /*              cE2Str<TypeEnum>               */
    /*                                             */
    /* =========================================== */

/// Those will never be automatized


template <class TypeEnum> class cE2Str
{
   public :
     static const std::string & E2Str(const TypeEnum & anE)
     {
         typename tMapE2Str::iterator anIt = mE2S.find(anE);
         if (anIt == mE2S.end())
            MMVII_INTERNAL_ASSERT_always(false,"E2Str for enum : " + ToStr(int(anE)));
         return anIt->second;
     }

     static const TypeEnum &  Str2E(const std::string & aStr)
     {
         if (mS2E==0)
         {
            // mS2E = new tMapStr2E;
            mS2E.reset(new tMapStr2E);
            for (const auto & it : mE2S)
                (*mS2E)[it.second] = it.first;
         }
         typename tMapStr2E::iterator anIt = mS2E->find(aStr);
         if (anIt == mS2E->end())
            MMVII_INTERNAL_ASSERT_always(false,"Str2E for enum : " + aStr);
         return anIt->second;
     }

     static const std::string   StrAllVal()
     {
         std::string aRes;
         for (const auto & it : mE2S)
         {
             if (aRes!="") aRes += " ";
             aRes += it.second;
         }
         return aRes;
     }

   private :
     typedef std::map<TypeEnum,std::string> tMapE2Str;
     typedef std::map<std::string,TypeEnum> tMapStr2E;

     static tMapE2Str                   mE2S;
     static std::unique_ptr<tMapStr2E > mS2E;
};
// template<class TypeEnum>  TypeEnum cE2Str<TypeEnum>::E2s


#define TPL_ENUM_2_STRING(TypeEnum)\
template<> std::unique_ptr<cE2Str<TypeEnum>::tMapStr2E > cE2Str<TypeEnum>::mS2E = nullptr;\
const std::string & E2Str(const TypeEnum & anOp)\
{\
   return cE2Str<TypeEnum>::E2Str(anOp);\
}\
template <> const TypeEnum & Str2E<TypeEnum>(const std::string & aName)\
{\
   return cE2Str<TypeEnum>::Str2E(aName);\
}\
template <> std::string   StrAllVall<TypeEnum>()\
{\
   return cE2Str<TypeEnum>::StrAllVal();\
}

TPL_ENUM_2_STRING(eOpAff);
TPL_ENUM_2_STRING(eTySC);



// This part must be redefined for each
template<> cE2Str<eOpAff>::tMapE2Str cE2Str<eOpAff>::mE2S
           {
                           {eOpAff::ePlusEq,"+="},
                           {eOpAff::eMulEq,"*="},
                           {eOpAff::eMinusEq,"-="},
                           {eOpAff::eEq,"="},
                           {eOpAff::eReset,"=0"}
           };
template<> cE2Str<eTySC>::tMapE2Str cE2Str<eTySC>::mE2S
           {
                           {eTySC::NonInit,MMVII_NONE},
                           {eTySC::US,"unordered"},
           };


/****************************  BENCH **************************/

/// Bench enum, template

template<class TypeEnum> void TplBenchEnum()
{
   for (int aK=0 ; aK<int(TypeEnum::eNbVals) ; aK++)
   {
       TypeEnum anE = (TypeEnum) aK;
       const std::string & aStr = E2Str(anE); 
       TypeEnum anE2 = Str2E<TypeEnum>(aStr);
       MMVII_INTERNAL_ASSERT_always(int(anE2)==aK,"TplBenchEnum");
   }
}

/// Bench enum
void BenchEnum()
{
   TplBenchEnum<eOpAff>();
   TplBenchEnum<eTySC>();
}


};

