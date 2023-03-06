
#include <map>
#include "MMVII_util.h"
#include "MMVII_enums.h"
#include "MMVII_Stringifier.h"
#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"

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
            MMVII_UsersErrror(eTyUEr::eBadEnum,"Str2E for : "+aStr+" ; valids are : "+ StrAllVal() );
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

     static std::vector<TypeEnum> VecOfPat(const std::string & aPat,bool AcceptEmpy)
     {
          std::vector<TypeEnum> aRes;
          tNameSelector  aSel =  AllocRegex(aPat);
          for (const auto & it : mE2S)
          {
              if (aSel.Match(it.second))
              {
                 aRes.push_back(it.first);
              }
          }
          if ((!AcceptEmpy) && aRes.empty())
          {
             MMVII_UsersErrror
             (
                eTyUEr::eEmptyPattern,
                "No value for enum, allowed are :"+StrAllVall<TypeEnum>()
             );

          }

          return aRes;
     }

     static std::vector<bool> VecBoolOfPat(const std::string & aPat,bool AcceptEmpy)
     {
         std::vector<TypeEnum>  aVEnum = VecOfPat(aPat,AcceptEmpy);
	 std::vector<bool> aResult(size_t(TypeEnum::eNbVals)+1,false);

	 for (const auto & aLab : aVEnum)
		 aResult.at(size_t(aLab)) = true;
	 return aResult;
     }

   private :
     typedef std::map<TypeEnum,std::string> tMapE2Str;
     typedef std::map<std::string,TypeEnum> tMapStr2E;

     static tMapE2Str                   mE2S;
     static std::unique_ptr<tMapStr2E > mS2E;
};

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
}\
template <> std::vector<TypeEnum> SubOfPat<TypeEnum>(const std::string & aPat,bool AcceptEmpty)\
{\
   return cE2Str<TypeEnum>::VecOfPat(aPat,AcceptEmpty);\
}\
template <> tSemA2007  AC_ListVal<TypeEnum>()\
{\
   return {eTA2007::AddCom,"Allowed values for this enum:{"+StrAllVall<TypeEnum>()+"}"};\
}\
template <> std::vector<bool> VBoolOfPat<TypeEnum>(const std::string & aPat,bool AcceptEmpty)\
{\
   return cE2Str<TypeEnum>::VecBoolOfPat(aPat,AcceptEmpty);\
}\


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


template<> cE2Str<eProjPC>::tMapE2Str cE2Str<eProjPC>::mE2S
           {
               {eProjPC::eStenope,"Stenope"},
               {eProjPC::eFE_EquiDist,"FE_EquiDist"},
               {eProjPC::eFE_EquiSolid,"FE_EquiSolid"},
               {eProjPC::eStereroGraphik,"StereroGraphik"},
               {eProjPC::eOrthoGraphik,"OrthoGraphik"},
               {eProjPC::eEquiRect,"eEquiRect"}
           };
TPL_ENUM_2_STRING(eProjPC);




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
                {eTA2007::FileImage,"Im"},
                {eTA2007::FileCloud,"Cloud"},
                {eTA2007::File3DRegion,"3DReg"},
                {eTA2007::MPatFile,"MPF"},
                {eTA2007::Orient,"Ori"},
                {eTA2007::Radiom,"Rad"},
                {eTA2007::MeshDev,"MeshDev"},
                {eTA2007::Mask,"Mask"},
                {eTA2007::Input,"In"},
                {eTA2007::Output,"Out"},
                {eTA2007::OptionalExist,"OptEx"},
                {eTA2007::AddCom,"AddCom"},
                {eTA2007::Internal,"##Intern"},
                {eTA2007::Tuning,"##Tune"},
                {eTA2007::Global,"##Glob"},
                {eTA2007::Shared,"##Shar"},
                {eTA2007::HDV,"##HDV"},
                {eTA2007::ISizeV,"##ISizeV"},
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
                {eTyNums::eTN_REAL8,"REAL8"},
                {eTyNums::eTN_REAL16,"REAL16"},
                {eTyNums::eTN_UnKnown,"Unknown"}
           };
TPL_ENUM_2_STRING(eTyNums);


template<> cE2Str<eTyUEr>::tMapE2Str cE2Str<eTyUEr>::mE2S
           {
                {eTyUEr::eCreateDir,"MkDir"},
                {eTyUEr::eRemoveFile,"RmFile"},
                {eTyUEr::eEmptyPattern,"EmptyPattern"},
                {eTyUEr::eBadFileSetName,"FileSetN"},
                {eTyUEr::eBadFileRelName,"FileRelN"},
                {eTyUEr::eOpenFile,"OpenFile"},
                {eTyUEr::eWriteFile,"WriteFile"},
                {eTyUEr::eReadFile,"ReadFile"},
                {eTyUEr::eBadBool,"BadBool"},
                {eTyUEr::eBadInt,"BadInt"},
                {eTyUEr::eBadEnum,"BadEnum"},
                {eTyUEr::eMulOptParam,"MultOptP"},
                {eTyUEr::eBadOptParam,"BadOptP"},
                {eTyUEr::eInsufNbParam,"InsufP"},
                {eTyUEr::eIntervWithoutSet,"IntWithoutS"},
                {eTyUEr::eTooBig4NbDigit,"TooBig4NbDigit"},
                {eTyUEr::eNoModeInEditRel,"NoModeInEditRel"},
                {eTyUEr::eMultiModeInEditRel,"MultiModeInEditRel"},
                {eTyUEr::e2PatInModeLineEditRel,"2PatInModeLineEditRel"},
                {eTyUEr::eParseError,"ParseError"},
                {eTyUEr::eBadDimForPt,"BadDimension4Pts"},
                {eTyUEr::eBadDimForBox,"BadDimension4Box"},
                {eTyUEr::eBadSize4Vect,"BadSize4Vector"},
                {eTyUEr::eMultiplePostifx,"MultiplePostifx"},
                {eTyUEr::eBadPostfix,"BadPostifx"},
                {eTyUEr::eNoAperture,"NoAperture"},
                {eTyUEr::eNoFocale,"NoFocale"},
                {eTyUEr::eNoFocaleEqui35,"NoFocaleEqui35"},
                {eTyUEr::eUnClassedError,"UnClassedError"}
           };

TPL_ENUM_2_STRING(eTyUEr);

template<> cE2Str<eTyInvRad>::tMapE2Str cE2Str<eTyInvRad>::mE2S
           {
                {eTyInvRad::eTVIR_ACGR,"eTVIR_ACGR"},
                {eTyInvRad::eTVIR_ACGT,"eTVIR_ACGT"},
                {eTyInvRad::eTVIR_ACR0,"eTVIR_ACR0"},
                {eTyInvRad::eTVIR_Curve,"eTVIR_Curve"}
           };
TPL_ENUM_2_STRING(eTyInvRad);

template<> cE2Str<eTyPyrTieP>::tMapE2Str cE2Str<eTyPyrTieP>::mE2S
           {
                {eTyPyrTieP::eTPTP_Init,"Init"},
                {eTyPyrTieP::eTPTP_LaplG,"LaplG"},
                {eTyPyrTieP::eTPTP_Corner,"Corner"},
                {eTyPyrTieP::eTPTP_OriNorm,"OriNorm"}
           };
TPL_ENUM_2_STRING(eTyPyrTieP);

template<> cE2Str<eModeEpipMatch>::tMapE2Str cE2Str<eModeEpipMatch>::mE2S
           {
                {eModeEpipMatch::eMEM_MMV1,"MMV1"},
                {eModeEpipMatch::eMEM_PSMNet,"PSMNet"},
                {eModeEpipMatch::eMEM_NoMatch,"NoMatch"}
           };
TPL_ENUM_2_STRING(eModeEpipMatch);

template<> cE2Str<eModeTestPropCov>::tMapE2Str cE2Str<eModeTestPropCov>::mE2S
           {
                {eModeTestPropCov::eMTPC_MatCovRFix  ,"MatCovRFix"},
                {eModeTestPropCov::eMTPC_SomL2RUk    ,"SomL2RUk"},
                {eModeTestPropCov::eMTPC_PtsRFix     ,"PtsRFix"},
                {eModeTestPropCov::eMTPC_PtsRUk      ,"PtsRUk"}
           };
TPL_ENUM_2_STRING(eModeTestPropCov);



template<> cE2Str<eModePaddingEpip>::tMapE2Str cE2Str<eModePaddingEpip>::mE2S
           {
                {eModePaddingEpip::eMPE_NoPad,"NoPad"},
                {eModePaddingEpip::eMPE_PxPos,"PxPos"},
                {eModePaddingEpip::eMPE_PxNeg,"PxNeg"},
                {eModePaddingEpip::eMPE_SzEq,"SzEq"}
           };
TPL_ENUM_2_STRING(eModePaddingEpip);

template<> cE2Str<eDCTFilters>::tMapE2Str cE2Str<eDCTFilters>::mE2S
           {
                {eDCTFilters::eSym,"Sym"},
                {eDCTFilters::eBin,"Bin"},
                {eDCTFilters::eRad,"Rad"},
                {eDCTFilters::eGrad,"Grad"}
           };
TPL_ENUM_2_STRING(eDCTFilters);


template<> cE2Str<eTyCodeTarget>::tMapE2Str cE2Str<eTyCodeTarget>::mE2S
           {
                {eTyCodeTarget::eIGNIndoor,"IGNIndoor"},
                {eTyCodeTarget::eIGNDroneSym,"IGNDroneSym"},
                {eTyCodeTarget::eIGNDroneTop,"IGNDroneTop"},
                {eTyCodeTarget::eCERN,"CERN"}
           };
TPL_ENUM_2_STRING(eTyCodeTarget);



template<> cE2Str<eModeCaracMatch>::tMapE2Str cE2Str<eModeCaracMatch>::mE2S
           {
                {eModeCaracMatch::eMS_CQ1,"MS_CQ1"},
                {eModeCaracMatch::eMS_CQ2,"MS_CQ2"},
                {eModeCaracMatch::eMS_CQ3,"MS_CQ3"},
                {eModeCaracMatch::eMS_CQ4,"MS_CQ4"},
                {eModeCaracMatch::eMS_CQW,"MS_CQW"},
                {eModeCaracMatch::eMS_CQA,"MS_CQA"},
             //---------------------------------------
                {eModeCaracMatch::eMS_Cen1,"MS_Cen1"},
                {eModeCaracMatch::eMS_Cen2,"MS_Cen2"},
                {eModeCaracMatch::eMS_Cen3,"MS_Cen3"},
                {eModeCaracMatch::eMS_Cen4,"MS_Cen4"},
                {eModeCaracMatch::eMS_CenW,"MS_CenW"},
                {eModeCaracMatch::eMS_CenA,"MS_CenA"},
             //---------------------------------------
                {eModeCaracMatch::eMS_Cor1,"MS_Cor1"},
                {eModeCaracMatch::eMS_Cor2,"MS_Cor2"},
                {eModeCaracMatch::eMS_Cor3,"MS_Cor3"},
                {eModeCaracMatch::eMS_Cor4,"MS_Cor4"},
                {eModeCaracMatch::eMS_CorW,"MS_CorW"},
                {eModeCaracMatch::eMS_CorA,"MS_CorA"},
             //---------------------------------------
                {eModeCaracMatch::eMS_WorstCorrel2,"MS_WorstCor2"},
                {eModeCaracMatch::eMS_WorstCorrel3,"MS_WorstCor3"},
                {eModeCaracMatch::eMS_BestCorrel2,"MS_BestCor2"},
                {eModeCaracMatch::eMS_BestCorrel3,"MS_BestCor3"},
                {eModeCaracMatch::eMS_BestCorrel4,"MS_BestCor4"},
                {eModeCaracMatch::eMS_BestCorrel5,"MS_BestCor5"},
             //---------------------------------------
                {eModeCaracMatch::eMS_BestCQ2,"MS_BestCQ2"},
                {eModeCaracMatch::eMS_BestCQ3,"MS_BestCQ3"},
                {eModeCaracMatch::eMS_WorstCQ2,"MS_WorstCQ2"},
                {eModeCaracMatch::eMS_WorstCQ3,"MS_WorstCQ3"},
                {eModeCaracMatch::eMS_WorstCQ4,"MS_WorstCQ4"},
                {eModeCaracMatch::eMS_WorstCQ5,"MS_WorstCQ5"},
             //---------------------------------------
                {eModeCaracMatch::eMS_CornW180,"MS_CornW180"},
                {eModeCaracMatch::eMS_CornW90 ,"MS_CornW90"},
             //---------------------------------------
                {eModeCaracMatch::eMS_MinStdDev1,"MS_MinStdDev1"},
                {eModeCaracMatch::eMS_MinStdDev2,"MS_MinStdDev2"},
                {eModeCaracMatch::eMS_MinStdDev3,"MS_MinStdDev3"},
                {eModeCaracMatch::eMS_MinStdDev4,"MS_MinStdDev4"},
                {eModeCaracMatch::eMS_MinStdDevW,"MS_MinStdDevW"},

      // -----########## NORMALIZED IMAGES ##############

             //---------------------------------------
                {eModeCaracMatch::eNI_DifGray,"NI_DifGray"},
                {eModeCaracMatch::eNI_MinGray,"NI_MinGray"},

             //---------------------------------------
                {eModeCaracMatch::eNI_Diff1,"NI_Diff1"}, // 5
                {eModeCaracMatch::eNI_Diff2,"NI_Diff2"},
                {eModeCaracMatch::eNI_Diff3,"NI_Diff3"},
                {eModeCaracMatch::eNI_Diff5,"NI_Diff5"},
                {eModeCaracMatch::eNI_Diff7,"NI_Diff7"},

      // -----########## STD IMAGES ##############

             //---------------------------------------
                {eModeCaracMatch::eSTD_Cor1,"STD_Cor1"}, // 5 + 4 = 9
                {eModeCaracMatch::eSTD_Cor2,"STD_Cor2"},
                {eModeCaracMatch::eSTD_Cor3,"STD_Cor3"},
                {eModeCaracMatch::eSTD_Cor4,"STD_Cor4"},

             //---------------------------------------
                {eModeCaracMatch::eSTD_NCCor1,"STD_NCCor1"}, // 9 + 4 = 13
                {eModeCaracMatch::eSTD_NCCor2,"STD_NCCor2"},
                {eModeCaracMatch::eSTD_NCCor3,"STD_NCCor3"},
                {eModeCaracMatch::eSTD_NCCor4,"STD_NCCor4"},

             //---------------------------------------
                {eModeCaracMatch::eSTD_Diff1,"STD_Diff1"}, // 13 + 5 = 18
                {eModeCaracMatch::eSTD_Diff2,"STD_Diff2"},
                {eModeCaracMatch::eSTD_Diff3,"STD_Diff3"},
                {eModeCaracMatch::eSTD_Diff5,"STD_Diff5"},
                {eModeCaracMatch::eSTD_Diff7,"STD_Diff7"},

             //---------------------------------------
                {eModeCaracMatch::eSTD_CQ2,"STD_CQ2"},  // 18 + 4 = 22
                {eModeCaracMatch::eSTD_CQ4,"STD_CQ4"},
                {eModeCaracMatch::eSTD_CQ6,"STD_CQ6"},
                {eModeCaracMatch::eSTD_CQ8,"STD_CQ8"},

             //---------------------------------------
                {eModeCaracMatch::eSTD_Cen2,"STD_Cen2"},  // 22 + 4 = 26
                {eModeCaracMatch::eSTD_Cen4,"STD_Cen4"},
                {eModeCaracMatch::eSTD_Cen6,"STD_Cen6"},
                {eModeCaracMatch::eSTD_Cen8,"STD_Cen8"}

           };
TPL_ENUM_2_STRING(eModeCaracMatch);
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
void BenchEnum(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Enum")) return;

    TplBenchEnum<eProjPC>();
    TplBenchEnum<eOpAff>();
    TplBenchEnum<eTySC>();
    TplBenchEnum<eTA2007>();
    TplBenchEnum<eTyUEr>();
    TplBenchEnum<eTyNums>();
    TplBenchEnum<eTyInvRad>();
    TplBenchEnum<eTyPyrTieP>();
    TplBenchEnum<eModeEpipMatch>();
    TplBenchEnum<eModeTestPropCov>();
    TplBenchEnum<eModePaddingEpip>();
    TplBenchEnum<eModeCaracMatch>();
    TplBenchEnum<eDCTFilters>();

    aParam.EndBench();
}


};



