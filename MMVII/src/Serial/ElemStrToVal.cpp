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

     //static const TypeEnum &  Str2E(const std::string & aStr,bool WithDef)
     static TypeEnum   Str2E(const std::string & aStr,bool WithDef)
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
            if (WithDef) 
                return TypeEnum::eNbVals;
            MMVII_UsersErrror(eTyUEr::eBadEnum,"Str2E for : "+aStr+" ; valids are : "+ StrAllVal() );
         }
         return anIt->second;
     }

     static const std::string   StrAllVal()
     {
         std::string aRes;
         for (const auto & it : mE2S)
         {
             if (aRes!="") aRes += ",";
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
template <> TypeEnum  Str2E<TypeEnum>(const std::string & aName,bool WithDef)\
{\
   return cE2Str<TypeEnum>::Str2E(aName,WithDef);\
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
   return {eTA2007::AllowedValues,StrAllVall<TypeEnum>()};\
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


template<> cE2Str<eProjPC>::tMapE2Str cE2Str<eProjPC>::mE2S
           {
               {eProjPC::eStenope,"Stenope"},
               {eProjPC::eFE_EquiDist,"FE_EquiDist"},
               {eProjPC::eFE_EquiSolid,"FE_EquiSolid"},
               {eProjPC::eStereroGraphik,"StereroGraphik"},
               {eProjPC::eOrthoGraphik,"OrthoGraphik"},
               {eProjPC::eEquiRect,"eEquiRect"}
           };

template<> cE2Str<eSysCoGeo>::tMapE2Str cE2Str<eSysCoGeo>::mE2S
           {
               {eSysCoGeo::eLambert93,"Lambert93"},
               {eSysCoGeo::eRTL,"RTL"},
               {eSysCoGeo::eGeoC,"GeoC"},
               {eSysCoGeo::eUndefined,"Undefined"}
           };


template<> cE2Str<eTySC>::tMapE2Str cE2Str<eTySC>::mE2S
           {
                           {eTySC::NonInit,MMVII_NONE},
                           {eTySC::US,"unordered"},
           };


template<> cE2Str<eTA2007>::tMapE2Str cE2Str<eTA2007>::mE2S
           {
                {eTA2007::DirProject,"DP"},
                {eTA2007::FileDirProj,"FDP"},
                {eTA2007::FileImage,"Im"},
                {eTA2007::FileCloud,"Cloud"},
                {eTA2007::File3DRegion,"3DReg"},
                {eTA2007::MPatFile,"MPF"},
                {eTA2007::Orient,"Ori"},
                {eTA2007::RadiomData,"RadData"},
                {eTA2007::RadiomModel,"RadModel"},
                {eTA2007::MeshDev,"MeshDev"},
                {eTA2007::Mask,"Mask"},
                {eTA2007::MetaData,"MetaData"},
                {eTA2007::PointsMeasure,"PointsMeasure"},
                {eTA2007::TieP,"TieP"},
                {eTA2007::MulTieP,"MulTieP"},
                {eTA2007::RigBlock,"RigBlock"},
                {eTA2007::SysCo,"SysCo"},
                {eTA2007::Input,"In"},
                {eTA2007::Output,"Out"},
                {eTA2007::OptionalExist,"OptEx"},
                {eTA2007::PatParamCalib,"ParamCalib"},
                {eTA2007::AddCom,"AddCom"},
                {eTA2007::AllowedValues,"Allowed"},
                {eTA2007::Internal,"##Intern"},
                {eTA2007::Tuning,"##Tune"},
                {eTA2007::Global,"##Glob"},
                {eTA2007::Shared,"##Shar"},
                {eTA2007::HDV,"##HDV"},
                {eTA2007::ISizeV,"##ISizeV"},
                {eTA2007::XmlOfTopTag,"##XmlOfTopTag"},
                {eTA2007::Range,"##Range"},
                {eTA2007::FFI,"FFI"}
           };

template<> cE2Str<eApF>::tMapE2Str cE2Str<eApF>::mE2S
           {
                {eApF::ManMMVII,"ManMMVII"},
                {eApF::Project,"Project"},
                {eApF::Test,"Test"},
                {eApF::ImProc,"ImProc"},
                {eApF::Radiometry,"Radiometry"},
                {eApF::Ori,"Ori"},
                {eApF::SysCo,"SysCo"},
                {eApF::Match,"Match"},
                {eApF::GCP,"GCP"},
                {eApF::TieP,"TieP"},
                {eApF::TiePLearn,"TiePLearn"},
                {eApF::Cloud,"Cloud"},
                {eApF::CodedTarget,"CodedTarget"},
                {eApF::Topo,"Topo"},
                {eApF::NoGui,"NoGui"},
                {eApF::Perso,"Perso"}
           };


template<> cE2Str<eApDT>::tMapE2Str cE2Str<eApDT>::mE2S
           {
                {eApDT::Ori,"Ori"},
                {eApDT::PCar,"PCar"},
                {eApDT::TieP,"TieP"},
                {eApDT::GCP,"GCP"},
                {eApDT::Image,"Image"},
                {eApDT::Orient,"Orient"},
                {eApDT::SysCo,"SysCo"},
                {eApDT::Radiom,"Radiom"},
                {eApDT::Ply,"Ply"},
                {eApDT::None,"None"},
                {eApDT::ToDef,"ToDef"},
                {eApDT::Console,"Console"},
                {eApDT::Xml,"Xml"},
                {eApDT::Csv,"Csv"},
                {eApDT::FileSys,"FileSys"},
                {eApDT::Media,"Media"},
           };


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


template<> cE2Str<eTyUEr>::tMapE2Str cE2Str<eTyUEr>::mE2S
           {
                {eTyUEr::eCreateDir,"MkDir"},
                {eTyUEr::eRemoveFile,"RmFile"},
                {eTyUEr::eEmptyPattern,"EmptyPattern"},
                {eTyUEr::eBadXmlTopTag,"XmlTopTag"},
                {eTyUEr::eParseBadClose,"ParseBadClose"},
                {eTyUEr::eJSonBadPunct,"JSonBadPunct"},
                {eTyUEr::eBadFileSetName,"FileSetN"},
                {eTyUEr::eBadFileRelName,"FileRelN"},
                {eTyUEr::eOpenFile,"OpenFile"},
                {eTyUEr::eWriteFile,"WriteFile"},
                {eTyUEr::eReadFile,"ReadFile"},
                {eTyUEr::eBadBool,"BadBool"},
                {eTyUEr::eBadInt,"BadInt"},
                {eTyUEr::eBadDegreeDist,"BadDegreeDist"},
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
                {eTyUEr::eNoNumberPixel,"NoNumberPixel"},
                {eTyUEr::eNoCameraName,"NoCameraName"},
                {eTyUEr::eUnClassedError,"UnClassedError"}
           };


template<> cE2Str<eTyInvRad>::tMapE2Str cE2Str<eTyInvRad>::mE2S
           {
                {eTyInvRad::eTVIR_ACGR,"eTVIR_ACGR"},
                {eTyInvRad::eTVIR_ACGT,"eTVIR_ACGT"},
                {eTyInvRad::eTVIR_ACR0,"eTVIR_ACR0"},
                {eTyInvRad::eTVIR_Curve,"eTVIR_Curve"}
           };

template<> cE2Str<eTyPyrTieP>::tMapE2Str cE2Str<eTyPyrTieP>::mE2S
           {
                {eTyPyrTieP::eTPTP_Init,"Init"},
                {eTyPyrTieP::eTPTP_LaplG,"LaplG"},
                {eTyPyrTieP::eTPTP_Corner,"Corner"},
                {eTyPyrTieP::eTPTP_OriNorm,"OriNorm"}
           };

template<> cE2Str<eModeEpipMatch>::tMapE2Str cE2Str<eModeEpipMatch>::mE2S
           {
                {eModeEpipMatch::eMEM_MMV1,"MMV1"},
                {eModeEpipMatch::eMEM_PSMNet,"PSMNet"},
                {eModeEpipMatch::eMEM_NoMatch,"NoMatch"}
           };

template<> cE2Str<eTyUnitAngle>::tMapE2Str cE2Str<eTyUnitAngle>::mE2S
           {
                {eTyUnitAngle::eUA_radian,"radian"},
                {eTyUnitAngle::eUA_degree,"degree"},
                {eTyUnitAngle::eUA_gon,"gon"}
           };

template<> cE2Str<eModeTestPropCov>::tMapE2Str cE2Str<eModeTestPropCov>::mE2S
           {
                {eModeTestPropCov::eMTPC_MatCovRFix  ,"MatCovRFix"},
                {eModeTestPropCov::eMTPC_SomL2RUk    ,"SomL2RUk"},
                {eModeTestPropCov::eMTPC_PtsRFix     ,"PtsRFix"},
                {eModeTestPropCov::eMTPC_PtsRUk      ,"PtsRUk"}
           };



template<> cE2Str<eModePaddingEpip>::tMapE2Str cE2Str<eModePaddingEpip>::mE2S
           {
                {eModePaddingEpip::eMPE_NoPad,"NoPad"},
                {eModePaddingEpip::eMPE_PxPos,"PxPos"},
                {eModePaddingEpip::eMPE_PxNeg,"PxNeg"},
                {eModePaddingEpip::eMPE_SzEq,"SzEq"}
           };

template<> cE2Str<eDCTFilters>::tMapE2Str cE2Str<eDCTFilters>::mE2S
           {
                {eDCTFilters::eSym,"Sym"},
                {eDCTFilters::eBin,"Bin"},
                {eDCTFilters::eRad,"Rad"},
                {eDCTFilters::eGrad,"Grad"}
           };


template<> cE2Str<eTyCodeTarget>::tMapE2Str cE2Str<eTyCodeTarget>::mE2S
           {
                {eTyCodeTarget::eIGNIndoor,"IGNIndoor"},
                {eTyCodeTarget::eIGNDroneSym,"IGNDroneSym"},
                {eTyCodeTarget::eIGNDroneTop,"IGNDroneTop"},
                {eTyCodeTarget::eCERN,"CERN"}
           };

template<> cE2Str<eMTDIm>::tMapE2Str cE2Str<eMTDIm>::mE2S
           {
                {eMTDIm::eFocalmm,"Focalmm"},
                {eMTDIm::eAperture,"Aperture"},
                {eMTDIm::eModelCam,"ModelCam"},
                {eMTDIm::eNbPixel,"NbPix"},
                {eMTDIm::eAdditionalName,"AdditionalName"}
           };

template<> cE2Str<eFormatExtern>::tMapE2Str cE2Str<eFormatExtern>::mE2S
           {
                {eFormatExtern::eMMV1,"MMV1"},
                {eFormatExtern::eMeshRoom,"MeshRoom"},
                {eFormatExtern::eColMap,"ColMap"}
           };


template<> cE2Str<eTypeSerial>::tMapE2Str cE2Str<eTypeSerial>::mE2S
           {
                {eTypeSerial::exml,"xml"},
                {eTypeSerial::exml2,"xml2"},
                {eTypeSerial::edmp,"dmp"},
                {eTypeSerial::etxt,"txt"},
                {eTypeSerial::etagt,"tagt"},
                {eTypeSerial::ejson,"json"},
                {eTypeSerial::ecsv,"csv"}
           };

template<> cE2Str<eTAAr>::tMapE2Str cE2Str<eTAAr>::mE2S
           {
                {eTAAr::eStd,"Std"},
                {eTAAr::eSzCont,"SzCont"},
                {eTAAr::eFixTabNum,"FixTabNum"},
                {eTAAr::ePtxd,"Ptxd"},
                {eTAAr::eCont,"Cont"},
                {eTAAr::eElemCont,"ElemC"},
                {eTAAr::eMap,"Map"},
                {eTAAr::ePairMap,"PairM"},
                {eTAAr::eKeyMap,"KeyM"},
                {eTAAr::eValMap,"ValM"},
                {eTAAr::eUndef,"???"}
           };


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
    TplBenchEnum<eSysCoGeo>();
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
    TplBenchEnum<eTyCodeTarget>();
    TplBenchEnum<eTypeSerial>();
    TplBenchEnum<eTAAr>();
    TplBenchEnum<eMTDIm>();

    aParam.EndBench();
}


/* ========================== */
/*          cEnumAttr         */
/* ========================== */

template <class TypeEnum> cEnumAttr<TypeEnum>::cEnumAttr(TypeEnum aType,const std::string & anAux) :
   mType (aType),
   mAux  (anAux)
{
}
template <class TypeEnum> cEnumAttr<TypeEnum>::cEnumAttr(TypeEnum aType) :
   cEnumAttr<TypeEnum>(aType,"")
{
}
template <class TypeEnum> TypeEnum            cEnumAttr<TypeEnum>::Type() const {return mType;}
template <class TypeEnum> const std::string & cEnumAttr<TypeEnum>::Aux()  const {return mAux;}


/* ========================== */
/*    cES_PropertyList        */
/* ========================== */

template <class TypeEnum> cES_PropertyList<TypeEnum>::cES_PropertyList(const tAllPairs & aAllPairs) :
   mAllPairs (aAllPairs)
{
}

template <class TypeEnum> const typename  cES_PropertyList<TypeEnum>::tAllPairs & cES_PropertyList<TypeEnum>::AllPairs() const
{
   return mAllPairs;
}


template class cEnumAttr<eTA2007>;
template class cES_PropertyList<eTA2007>;


/* ========================== */
/*             ::             */
/* ========================== */


std::string  Name4Help(const tSemA2007 & aSem)
{
   if (int(aSem.Type()) < int(eTA2007::AddCom))
   {
      return E2Str(aSem.Type()) + aSem.Aux();
   }

   return "";
}



/* ========================== */
/*          cSpecOneArg2007   */
/* ========================== */

const std::vector<tSemA2007>   cSpecOneArg2007::TheEmptySem;

cSpecOneArg2007::cSpecOneArg2007(const std::string & aName,const std::string & aCom,const tAllSemPL & aVPL) :
   mName  (aName),
   mCom   (aCom),
   mSemPL (aVPL)
{
    ReInit();
}

cSpecOneArg2007::~cSpecOneArg2007()
{
}

void cSpecOneArg2007::ReInit()
{
   mNbMatch = 0;
}


std::string  cSpecOneArg2007::Name4Help() const
{
   std::string aRes;
   int aNb=0;
   for (const auto & aSem : SemPL())
   {
      std::string aStr = MMVII::Name4Help(aSem);
      if (aStr!="")
      {
         if (aNb==0)
            aRes = " [";
         else
            aRes = aRes + ",";
         aRes = aRes + aStr;
         aNb++;
      }
   }
   if (aNb!=0)
      aRes += "]";
   return aRes;
}

std::list<std::string>  cSpecOneArg2007::AddComs() const
{
   std::list<std::string> aRes;
   for (const auto & aSem : SemPL())
   {
      if (aSem.Type()== eTA2007::AddCom)
         aRes.push_back(aSem.Aux());
      if (aSem.Type()== eTA2007::AllowedValues)
         aRes.push_back("Allowed values for this enum:{" + aSem.Aux() + "}");
      if (aSem.Type()== eTA2007::Range)
         aRes.push_back("Allowed values range:" + aSem.Aux());
   }
   return aRes;
}


void cSpecOneArg2007::IncrNbMatch()
{
   mNbMatch++;
}

int  cSpecOneArg2007::NbMatch () const
{
   return mNbMatch;
}

const cSpecOneArg2007::tAllSemPL & cSpecOneArg2007::SemPL() const
{
   return mSemPL.AllPairs();
}

bool cSpecOneArg2007::HasType(const eTA2007 & aType,std::string * aValue) const
{
    for (const auto & aSem : SemPL())
    {
       if (aSem.Type() == aType)
       {
          if (aValue) 
             *aValue =  aSem.Aux();
          return true;
       }
   }

    return false;
}

const std::string  & cSpecOneArg2007::Name() const
{
   return mName;
}

const std::string  & cSpecOneArg2007::Value() const
{
   return mValue;
}

const std::string  & cSpecOneArg2007::Com() const
{
   return mCom;
}

void  cSpecOneArg2007::InitParam(const std::string & aStr) 
{
   mValue = aStr;
   V_InitParam(aStr);
}


/* ============================ */
/*          cCollecSpecArg2007  */
/* ============================ */


cCollecSpecArg2007 & cCollecSpecArg2007::operator << (tPtrArg2007 aVal)
{
    mV.push_back(aVal);
    return *this; 
}

cCollecSpecArg2007::cCollecSpecArg2007()
{
}

size_t cCollecSpecArg2007::size() const
{
   return mV.size();
}

tPtrArg2007 cCollecSpecArg2007::operator [] (int aK) const
{
   return mV.at(aK);
}

void cCollecSpecArg2007::clear()
{
   mV.clear();
}

tVecArg2007 & cCollecSpecArg2007::Vec()
{
   return mV;
}




/* ============================================== */
/*                                                */
/*       cInstReadOneArgCL2007                    */
/*                                                */
/* ============================================== */

template<typename T> struct is_vector : public std::false_type {};

template<typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};


template <class Type> void  GlobCheckSize(const Type & ,const std::string & anArg) 
{
    MMVII_INTERNAL_ASSERT_always(false,"Check size vect for non vect arg");
}

template <class Type> void  GlobCheckSize(const std::vector<Type> & aVal,const std::string & anArg) 
{
    cPt2di aSz = cStrIO<cPt2di>::FromStr(anArg);
    if ((int(aVal.size()) < aSz.x()) || ((int(aVal.size()) > aSz.y()))) 
    {
       MMVII_UsersErrror(eTyUEr::eBadSize4Vect,"IntervalOk=" + anArg + " Got=" + ToStr(int(aVal.size())));
    }
}


template <class Type> class cInstReadOneArgCL2007 : public cSpecOneArg2007
{
    public :

       void  CheckSize(const std::string & anArg) const override 
       {
               GlobCheckSize(mVal,anArg);
       }

       bool IsVector() const override
       {
           return  is_vector<Type>::value;
       }



        void V_InitParam(const std::string & aStr) override
        {
            mVal = cStrIO<Type>::FromStr(aStr);
        }
        cInstReadOneArgCL2007 (Type & aVal,const std::string & aName,const std::string & aCom,const tAllSemPL & aVSem) :
              cSpecOneArg2007(aName,aCom,aVSem),
              mVal         (aVal)
        {
        }
        const std::string & NameType() const override 
        {
            return  cStrIO<Type>::msNameType;
        }
        void * AdrParam() override {return &mVal;}
        std::string NameValue() const override {return ToStr(mVal);}

    private :
        Type &          mVal;
};


template <class Type> tPtrArg2007 Arg2007(Type & aVal, const std::string & aCom,const cSpecOneArg2007::tAllSemPL & aVSem )
{
   return tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,"",aCom,aVSem));
}




template <class Type> tPtrArg2007 AOpt2007(Type & aVal,const std::string & aName, const std::string &aCom,const cSpecOneArg2007::tAllSemPL & aVSem)
{
   return  tPtrArg2007(new cInstReadOneArgCL2007<Type>(aVal,aName,aCom,aVSem));
}

#define MACRO_INSTANTIATE_ARG2007(Type)\
template tPtrArg2007 Arg2007<Type>(Type &, const std::string & aCom,const cSpecOneArg2007::tAllSemPL & aVSem);\
template tPtrArg2007 AOpt2007<Type>(Type &,const std::string & aName, const std::string & aCom,const cSpecOneArg2007::tAllSemPL & aVSem);

MACRO_INSTANTIATE_ARG2007(char)
MACRO_INSTANTIATE_ARG2007(size_t)
MACRO_INSTANTIATE_ARG2007(int)
MACRO_INSTANTIATE_ARG2007(double)
MACRO_INSTANTIATE_ARG2007(bool)
MACRO_INSTANTIATE_ARG2007(std::string)
MACRO_INSTANTIATE_ARG2007(std::vector<std::string>)
MACRO_INSTANTIATE_ARG2007(std::vector<int>)
MACRO_INSTANTIATE_ARG2007(std::vector<double>)
MACRO_INSTANTIATE_ARG2007(cPt2di)
MACRO_INSTANTIATE_ARG2007(cPt2dr)
MACRO_INSTANTIATE_ARG2007(cPt3di)
MACRO_INSTANTIATE_ARG2007(cPt3dr)
MACRO_INSTANTIATE_ARG2007(cBox2di)
MACRO_INSTANTIATE_ARG2007(cBox2dr)
MACRO_INSTANTIATE_ARG2007(cBox3di)
MACRO_INSTANTIATE_ARG2007(cBox3dr)




/**
    This file contains the implemenation of conversion between strings and 
   atomic and some non atomic object
*/

/* ==================================== */
/*                                      */
/*         std::vector<T>               */
/*                                      */
/* ==================================== */

static char BufStrIO[1000];

//  vector<int>  => [1,2,3]

template <class Type>  std::string Vect2Str(const std::vector<Type>  & aV)
{
   std::string aRes ="[";
   for (int aK=0 ; aK<(int)aV.size() ; aK++)
   {
      if (aK>0)
         aRes += ",";
      aRes += ToStr(aV[aK]);
   }
   aRes += "]";
   return aRes;
}

template <class Type>  std::vector<Type> Str2Vec(const std::string & aStrGlob)
{
   std::vector<Type> aRes;
   const char * aC=aStrGlob.c_str();
   if (*aC!='[')
       MMVII_UsersErrror(eTyUEr::eParseError,"expected [ at beging of vect");
   aC++;
   while((*aC) && *aC!=']')
   {
       std::string aStrV;
       while ((*aC) && (*aC!=',') && (*aC!=']'))
          aStrV += *(aC++);
       if (!(*aC))
          MMVII_UsersErrror(eTyUEr::eParseError,"unexpected end of string while expecting \",\"");
       aRes.push_back(cStrIO<Type>::FromStr(aStrV)); 
       if (*aC==',')
          aC++;
   }
   if (*aC!=']')
      MMVII_UsersErrror(eTyUEr::eParseError,"unexpected end of string while expecting \"]\"");
   aC++;

   return  aRes;
}

                          //   - - std::vector<Type>  - -

#define MACRO_INSTANTITATE_STRIO_VECT_TYPE(TYPE)\
template <>  std::string cStrIO<std::vector<TYPE>>::ToStr(const std::vector<TYPE>  & aV)\
{\
   return  Vect2Str(aV);\
}\
template <>  std::vector<TYPE> cStrIO<std::vector<TYPE> >::FromStr(const std::string & aStr)\
{\
    return Str2Vec<TYPE>(aStr);\
}\
template <>  const std::string cStrIO<std::vector<TYPE>>::msNameType = "std::vector<"  #TYPE  ">";\

MACRO_INSTANTITATE_STRIO_VECT_TYPE(std::string)
MACRO_INSTANTITATE_STRIO_VECT_TYPE(int)
MACRO_INSTANTITATE_STRIO_VECT_TYPE(double)

/* ==================================== */
/*                                      */
/*         cPtxd                        */
/*                                      */
/* ==================================== */

                          //   - - cPtxd  - -


#define MACRO_INSTANTITATE_STRIO_CPTXD(TYPE,DIM)\
template <>  std::string cStrIO<cTplBox<TYPE,DIM> >::ToStr(const cTplBox<TYPE,DIM>  & aV)\
{\
  std::vector<TYPE> aVec;\
  for (int aD=0; aD<DIM; aD++) aVec.push_back(aV.P0()[aD]);\
  for (int aD=0; aD<DIM; aD++) aVec.push_back(aV.P1()[aD]);\
  return Vect2Str(aVec);\
}\
template <>  std::string cStrIO<cPtxd<TYPE,DIM> >::ToStr(const cPtxd<TYPE,DIM>  & aV)\
{\
  return Vect2Str(std::vector<TYPE>(aV.PtRawData(),aV.PtRawData()+cPtxd<TYPE,DIM>::TheDim));\
}\
template <>  cPtxd<TYPE,DIM> cStrIO<cPtxd<TYPE,DIM> >::FromStr(const std::string & aStr)\
{\
    std::vector<TYPE> aV = cStrIO<std::vector<TYPE>>::FromStr(aStr);\
    if (aV.size()!=DIM)\
       MMVII_UsersErrror(eTyUEr::eBadDimForPt,"Expect="+ MMVII::ToStr(DIM) + " Got=" + MMVII::ToStr(int(aV.size())) );\
    cPtxd<TYPE,DIM> aRes;\
    for (int aK=0 ; aK<DIM ; aK++)\
        aRes[aK] = aV[aK];\
    return aRes;\
}\
template <>  cTplBox<TYPE,DIM> cStrIO<cTplBox<TYPE,DIM> >::FromStr(const std::string & aStr)\
{\
    std::vector<TYPE> aV = cStrIO<std::vector<TYPE>>::FromStr(aStr);\
    if (aV.size()!=2*DIM)\
       MMVII_UsersErrror(eTyUEr::eBadDimForBox,"Expect="+ MMVII::ToStr(2*DIM) + " Got=" + MMVII::ToStr(int(aV.size())) );\
    cPtxd<TYPE,DIM> aP0,aP1;\
    for (int aK=0 ; aK<DIM ; aK++){\
        aP0[aK] = aV[aK];\
        aP1[aK] = aV[aK+DIM];\
    }\
    return cTplBox<TYPE,DIM>(aP0,aP1);\
}\
template <>  const std::string cStrIO<cPtxd<TYPE,DIM> >::msNameType = "cPtxd<" #TYPE ","  #DIM ">";\
template <>  const std::string cStrIO<cTplBox<TYPE,DIM> >::msNameType = "cTplBox<" #TYPE ","  #DIM ">";\

MACRO_INSTANTITATE_STRIO_CPTXD(int,2)
MACRO_INSTANTITATE_STRIO_CPTXD(double,2)
MACRO_INSTANTITATE_STRIO_CPTXD(int,3)
MACRO_INSTANTITATE_STRIO_CPTXD(double,3)





void OneBenchStrIO(std::string aStr,const  std::vector<std::string> & aV)
{
   // std::string aStr =
   std::vector<std::string> aV2 =  cStrIO<std::vector<std::string> >::FromStr(aStr);
   if(aV!=aV2)
   {
      StdOut() << "STR=" << aStr << std::endl;
      StdOut() << "VEC=[";
      for (int aK=0 ; aK<int(aV2.size()) ; aK++)
      {
         if (aK!=0) StdOut() << ",";
         StdOut() << "{" << aV2[aK] << "}";
      }
      StdOut() << "]" << std::endl;
      MMVII_INTERNAL_ASSERT_bench((aV==aV2),"OneBenchStrIO");
   }
}

void BenchStrIO(cParamExeBench & aParam)
{
   if (! aParam.NewBench("StrIO")) return;
   OneBenchStrIO("[1,2,3]",{"1","2","3"});
   OneBenchStrIO("[1]",{"1"});
   OneBenchStrIO("[]",{});
   OneBenchStrIO("[,1]",{"","1"});
   OneBenchStrIO("[,,,1]",{"","","","1"});
   OneBenchStrIO("[,]",{""});
   OneBenchStrIO("[1,2,]",{"1","2"});
   OneBenchStrIO("[1,2,,]",{"1","2",""});
   OneBenchStrIO("[1,2,,,,]",{"1","2","","",""});
   // OneBenchStrIO("[,,1,,3,]",{"","","1","","3",""});
   // Check that we get an error catched
   //    OneBenchStrIO("[",{});
   //    OneBenchStrIO("[1,2",{});
   // getchar();
   aParam.EndBench();
}

/* ==================================== */
/*                                      */
/*          Enumerated type             */
/*    eOpAff,                           */
/*                                      */
/* ==================================== */


#define MACRO_INSTANTITATE_STRIO_ENUM(ETYPE,ENAME)\
MACRO_INSTANTIATE_ARG2007(ETYPE)\
TPL_ENUM_2_STRING(ETYPE)\
template <>  std::string cStrIO<ETYPE>::ToStr(const ETYPE & anEnum) { return  E2Str(anEnum); }\
template <>  ETYPE cStrIO<ETYPE>::FromStr(const std::string & aStr) { return Str2E<ETYPE>(aStr); }\
template <>  const std::string cStrIO<ETYPE>::msNameType = ENAME;

MACRO_INSTANTITATE_STRIO_ENUM(eApF,"ApF")
MACRO_INSTANTITATE_STRIO_ENUM(eApDT,"ApDT")
MACRO_INSTANTITATE_STRIO_ENUM(eTyNums,"TypeNum")
MACRO_INSTANTITATE_STRIO_ENUM(eTyUEr,"TyUEr")
MACRO_INSTANTITATE_STRIO_ENUM(eTyInvRad,"TyInvRad")
MACRO_INSTANTITATE_STRIO_ENUM(eTyPyrTieP,"TyPyrTieP")
MACRO_INSTANTITATE_STRIO_ENUM(eProjPC,"ProjPC")
MACRO_INSTANTITATE_STRIO_ENUM(eSysCoGeo,"SysCoGeo")
MACRO_INSTANTITATE_STRIO_ENUM(eOpAff,"OpAff")
MACRO_INSTANTITATE_STRIO_ENUM(eModeEpipMatch,"ModeEpiMatch")
MACRO_INSTANTITATE_STRIO_ENUM(eModePaddingEpip,"ModePadEpip")
MACRO_INSTANTITATE_STRIO_ENUM(eModeCaracMatch,"ModeCaracMatch")
MACRO_INSTANTITATE_STRIO_ENUM(eDCTFilters,"DCTFilters")
MACRO_INSTANTITATE_STRIO_ENUM(eTyCodeTarget,"TypeCodedTarget")
MACRO_INSTANTITATE_STRIO_ENUM(eModeTestPropCov,"TestPropCov")
MACRO_INSTANTITATE_STRIO_ENUM(eMTDIm,"TypeMTDIm")
MACRO_INSTANTITATE_STRIO_ENUM(eFormatExtern,"ExternalFormat")
MACRO_INSTANTITATE_STRIO_ENUM(eTypeSerial,"TypeSerial")
MACRO_INSTANTITATE_STRIO_ENUM(eTAAr,"TypeAAr")
MACRO_INSTANTITATE_STRIO_ENUM(eTA2007,"TA2007")
MACRO_INSTANTITATE_STRIO_ENUM(eTySC,"TySC")
MACRO_INSTANTITATE_STRIO_ENUM(eTyUnitAngle,"AngleUnit")


/* ==================================== */
/*                                      */
/*         Atomic native type           */
/*  bool, int, double, std::string      */
/*                                      */
/* ==================================== */

   // ================  bool ==============================================

template <>  std::string cStrIO<bool>::ToStr(const bool & anI)
{
   return  anI ? "true" : "false";
}
template <>  bool cStrIO<bool>::FromStr(const std::string & aStr)
{
    if ((aStr=="1") || UCaseEqual(aStr,"true")) return true;
    if ((aStr=="0") || UCaseEqual(aStr,"false")) return false;

    MMVII_UsersErrror(eTyUEr::eBadBool,"Bad value for boolean :["+aStr+"]");

    return false;
}

template <>  const std::string cStrIO<bool>::msNameType = "bool";

   // ================  char ==============================================

template <>  std::string cStrIO<char>::ToStr(const char & anI)
{
 
   std::string aStrI;
   aStrI += anI;
   return   aStrI ;
}
template <>  char cStrIO<char>::FromStr(const std::string & aStr)
{
    MMVII_INTERNAL_ASSERT_User(aStr.size()==1,eTyUEr::eUnClassedError,"String size shoul be 1 for char create");

    return aStr[0];
}

template <>  const std::string cStrIO<char>::msNameType = "char";






   // ================  size_t ==============================================

template <>  std::string cStrIO<size_t>::ToStr(const size_t & aSz)
{
   sprintf(BufStrIO,"%zu",aSz);
   return BufStrIO;
}
template <>  size_t cStrIO<size_t>::FromStr(const std::string & aStr)
{
    // can be convenient that empty string correspond to zero
    if (aStr.empty())
       return 0;
    size_t aSz;
    int aNb= sscanf(aStr.c_str(),"%zu",&aSz);

    MMVII_INTERNAL_ASSERT_User((aNb!=0),eTyUEr::eBadInt,"String is not a valid size_t")
    return aSz;
}
template <>  const std::string cStrIO<size_t>::msNameType = "size_t";


   // ================  int ==============================================

template <>  std::string cStrIO<int>::ToStr(const int & anI)
{
   sprintf(BufStrIO,"%d",anI);
   return BufStrIO;
}
template <>  int cStrIO<int>::FromStr(const std::string & aStr)
{
    // can be convenient that empty string correspond to zero
    if (aStr.empty())
       return 0;
    int anI;
    int aNb= sscanf(aStr.c_str(),"%d",&anI);

    if (aNb==0)
    {
         MMVII_INTERNAL_ASSERT_User((aNb!=0),eTyUEr::eBadInt,"String=["+ aStr +"] is not a valid int")
    }
    return anI;
}
template <>  const std::string cStrIO<int>::msNameType = "int";

std::string ToStr(int aVal,int aSzMin)
{
   std::string aRes = ToStr(std::abs(aVal));
   while (int(aRes.size())<aSzMin)
       aRes = "0" + aRes;
   if (aVal<0)
       aRes = "-" + aRes;
   return aRes;
}



/*
std::string  ToS_NbDigit(int aNb,int aNbDig,bool AcceptOverFlow)
{
   std::string aRes = ToS(aNb);
   int aSz = (int)aRes.size();
   if ((!AcceptOverFlow) && (aSz>aNbDig))
   {
       MMVII_INTERNAL_ASSERT_user(eTyUEr::eTooBig4NbDigit,"Pas assez de digit dans ToStringNBD")
   }
   for (;aSz<aNbDig ; aSz++)
   {
       aRes = "0" + aRes;
   }
   return aRes;
}
*/


   // ================  double ==============================================

static std::vector<size_t> The_VecPrecTxtSerial = {15};
void PushPrecTxtSerial(size_t aPrec) { The_VecPrecTxtSerial.push_back(aPrec); }
void PopPrecTxtSerial() { The_VecPrecTxtSerial.pop_back(); }


template <>  std::string cStrIO<double>::ToStr(const double & aD)
{
    if (int(aD) == aD) return cStrIO<int>::ToStr (int(aD));

    std::ostringstream out;
    out.precision(The_VecPrecTxtSerial.back());
    out << std::fixed << aD;

    std::string aRes = std::move(out).str();

    if (aRes.back() != '0') return aRes;

    int aL = aRes.size()-1;

    while ((aL>=0) && (aRes[aL] == '0'))
          aL--;

    std::string aNewRes = aRes.substr(0,aL+1);

    if (RelativeSafeDifference(aD,FromStr(aNewRes)) < 1e-10)
	    return aNewRes;

    return aRes;
	/*
   sprintf(BufStrIO,"%lf",aD);
   return BufStrIO;
    */
   // return std::to_string(aD);
}
template <>  double cStrIO<double>::FromStr(const std::string & aStr)
{
    double anI;
    sscanf(aStr.c_str(),"%lf",&anI);
    return anI;
}
template <>  const std::string cStrIO<double>::msNameType = "double";

std::string FixDigToStr(double aSignedVal,int aNbDig)
{
   std::string aFormat = "%."+ToS(aNbDig) + "f";
   char aBuf[100];
   sprintf(aBuf,aFormat.c_str(),aSignedVal);
   return aBuf;
}

std::string FixDigToStr(double aSignedVal,int aNbBef,int aNbAfter)
{
   std::string aFormat = "%0" + ToS(aNbBef+aNbAfter+1) + "."+ToS(aNbAfter) + "f";
   char aBuf[100];
   sprintf(aBuf,aFormat.c_str(),aSignedVal);
   return aBuf;
}



   // ================  std::string ==============================================

template <>  std::string cStrIO<std::string>::ToStr(const std::string & aStr)
{
   return aStr;
}
template <>  std::string cStrIO<std::string>::FromStr(const std::string & aStr)
{
    return aStr;
}

template <>  const std::string cStrIO<std::string>::msNameType = "string";

};
