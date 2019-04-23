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
         /// In this direction no need to create
         typename tMapE2Str::iterator anIt = mE2S.find(anE);
         // Enum to string is not user error (user do not create enum)
         if (anIt == mE2S.end())
            MMVII_INTERNAL_ASSERT_always(false,"E2Str for enum : " + ToStr(int(anE)));
         return anIt->second;
     }

     static const TypeEnum &  Str2E(const std::string & aStr)
     {
         /// If first time we create mS2E by inverting the  mE2S
         if (mS2E==0)
         {
            // mS2E = new tMapStr2E;
            mS2E.reset(new tMapStr2E);
            for (const auto & it : mE2S)
                (*mS2E)[it.second] = it.first;
         }
         typename tMapStr2E::iterator anIt = mS2E->find(aStr);
         // String to enum is probably a user error (programm create enum)
         if (anIt == mS2E->end())
         {
            MMVII_INTERNAL_ASSERT_user(eTyUEr::eBadEnum,"Str2E for : " + aStr + " ; valid are : " + StrAllVal() );
         }
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




// This part must be redefined for each
template<> cE2Str<eOpAff>::tMapE2Str cE2Str<eOpAff>::mE2S
           {
                           {eOpAff::ePlusEq,"+="},
                           {eOpAff::eMulEq,"*="},
                           {eOpAff::eMinusEq,"-="},
                           {eOpAff::eEq,"="},
                           {eOpAff::eReset,"=0"}
           };
TPL_ENUM_2_STRING(eOpAff);

template<> cE2Str<eTySC>::tMapE2Str cE2Str<eTySC>::mE2S
           {
                           {eTySC::NonInit,MMVII_NONE},
                           {eTySC::US,"unordered"},
           };
TPL_ENUM_2_STRING(eTySC);


template<> cE2Str<eTA2007>::tMapE2Str cE2Str<eTA2007>::mE2S
           {
                {eTA2007::DirProject,"DP"},
                {eTA2007::FileDirProj,"FDP"},
                {eTA2007::MPatIm,"MPI"},
                {eTA2007::Internal,"##Intern"},
                {eTA2007::Common,"##Com"},
                {eTA2007::HDV,"##HDV"},
                {eTA2007::FFI,"FFI"}
           };
TPL_ENUM_2_STRING(eTA2007);



template<> cE2Str<eTyNums>::tMapE2Str cE2Str<eTyNums>::mE2S
           {
                {eTyNums::eTN_INT1,"INT1"},
                {eTyNums::eTN_U_INT1,"U_INT1"},
                {eTyNums::eTN_INT2,"INT2"},
                {eTyNums::eTN_U_INT2,"U_INT2"},
                {eTyNums::eTN_INT4,"INT4"},
                {eTyNums::eTN_U_INT4,"U_INT4"},
                {eTyNums::eTN_INT8,"INT8"},
                {eTyNums::eTN_REAL4,"REAL4"},
                {eTyNums::eTN_REAL8,"REAL8"}
           };
TPL_ENUM_2_STRING(eTyNums);


template<> cE2Str<eTyUEr>::tMapE2Str cE2Str<eTyUEr>::mE2S
           {
                {eTyUEr::eCreateDir,"MkDir"},
                {eTyUEr::eRemoveFile,"RmFile"},
                {eTyUEr::eBadFileSetName,"FileSetN"},
                {eTyUEr::eBadFileRelName,"FileRelN"},
                {eTyUEr::eOpenFile,"OpenFile"},
                {eTyUEr::eWriteFile,"WriteFile"},
                {eTyUEr::eReadFile,"ReadFile"},
                {eTyUEr::eBadBool,"BadBool"},
                {eTyUEr::eBadEnum,"BadEnum"},
                {eTyUEr::eMulOptParam,"MultOptP"},
                {eTyUEr::eBadOptParam,"BadOptP"},
                {eTyUEr::eInsufNbParam,"InsufP"},
                {eTyUEr::eIntervWithoutSet,"IntWithoutS"},
                {eTyUEr::eTooBig4NbDigit,"TooBig4NbDigit"},
                {eTyUEr::eNoModeInEditRel,"NoModeInEditRel"},
                {eTyUEr::eMultiModeInEditRel,"MultiModeInEditRel"},
                {eTyUEr::e2PatInModeLineEditRel,"2PatInModeLineEditRel"}
           };
TPL_ENUM_2_STRING(eTyUEr);





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
    TplBenchEnum<eTA2007>();
    TplBenchEnum<eTyUEr>();
    TplBenchEnum<eTyNums>();
}


};

