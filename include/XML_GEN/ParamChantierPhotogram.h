#include "StdAfx.h"
//#include "general/all.h"
//#include "private/all.h"
#ifndef Define_NotPCP
#define Define_NotPCP
// NO MORE
typedef enum
{
  esingle,
  edgps,
  ekinematic,
  estatic,
  emovingbase,
  efixed,
  eppp_kine,
  eppp_static,
  eNbTypeGpsMod
} eTypeGpsMod;
void xml_init(eTypeGpsMod & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsMod & aVal);

eTypeGpsMod  Str2eTypeGpsMod(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsMod & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsMod &);

std::string  Mangling( eTypeGpsMod *);

void  BinaryUnDumpFromFile(eTypeGpsMod &,ELISE_fp &);

typedef enum
{
  ellh,
  exyz,
  eenu,
  enmea,
  eNbTypeGpsSolFormat
} eTypeGpsSolFormat;
void xml_init(eTypeGpsSolFormat & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsSolFormat & aVal);

eTypeGpsSolFormat  Str2eTypeGpsSolFormat(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsSolFormat & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsSolFormat &);

std::string  Mangling( eTypeGpsSolFormat *);

void  BinaryUnDumpFromFile(eTypeGpsSolFormat &,ELISE_fp &);

typedef enum
{
  eellipsoidal,
  egeodetic,
  eNbTypeGpsHeight
} eTypeGpsHeight;
void xml_init(eTypeGpsHeight & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsHeight & aVal);

eTypeGpsHeight  Str2eTypeGpsHeight(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsHeight & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsHeight &);

std::string  Mangling( eTypeGpsHeight *);

void  BinaryUnDumpFromFile(eTypeGpsHeight &,ELISE_fp &);

typedef enum
{
  einternal,
  eegm96,
  eegm08_2_5,
  eegm08_1,
  egsi2000,
  eNbTypeGeoid
} eTypeGeoid;
void xml_init(eTypeGeoid & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGeoid & aVal);

eTypeGeoid  Str2eTypeGeoid(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGeoid & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGeoid &);

std::string  Mangling( eTypeGeoid *);

void  BinaryUnDumpFromFile(eTypeGeoid &,ELISE_fp &);

typedef enum
{
  enone,
  estate,
  eresidual,
  eNbTypeRtkOutStats
} eTypeRtkOutStats;
void xml_init(eTypeRtkOutStats & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeRtkOutStats & aVal);

eTypeRtkOutStats  Str2eTypeRtkOutStats(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeRtkOutStats & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeRtkOutStats &);

std::string  Mangling( eTypeRtkOutStats *);

void  BinaryUnDumpFromFile(eTypeRtkOutStats &,ELISE_fp &);

typedef enum
{
  eall,
  eone,
  eNbTypeGpsSolStatic
} eTypeGpsStaticSol;
void xml_init(eTypeGpsStaticSol & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsStaticSol & aVal);

eTypeGpsStaticSol  Str2eTypeGpsStaticSol(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsStaticSol & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsStaticSol &);

std::string  Mangling( eTypeGpsStaticSol *);

void  BinaryUnDumpFromFile(eTypeGpsStaticSol &,ELISE_fp &);

typedef enum
{
  egpst,
  eutc,
  ejst,
  eNbTypeGpsTimeSys
} eTypeGpsTimeSys;
void xml_init(eTypeGpsTimeSys & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsTimeSys & aVal);

eTypeGpsTimeSys  Str2eTypeGpsTimeSys(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsTimeSys & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsTimeSys &);

std::string  Mangling( eTypeGpsTimeSys *);

void  BinaryUnDumpFromFile(eTypeGpsTimeSys &,ELISE_fp &);

typedef enum
{
  etow,
  ehms,
  eNbTypeGpsTimeFormat
} eTypeGpsTimeFormat;
void xml_init(eTypeGpsTimeFormat & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsTimeFormat & aVal);

eTypeGpsTimeFormat  Str2eTypeGpsTimeFormat(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsTimeFormat & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsTimeFormat &);

std::string  Mangling( eTypeGpsTimeFormat *);

void  BinaryUnDumpFromFile(eTypeGpsTimeFormat &,ELISE_fp &);

typedef enum
{
  edeg,
  edms,
  eNbTypeGpsDegFormat
} eTypeGpsDegFormat;
void xml_init(eTypeGpsDegFormat & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsDegFormat & aVal);

eTypeGpsDegFormat  Str2eTypeGpsDegFormat(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsDegFormat & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsDegFormat &);

std::string  Mangling( eTypeGpsDegFormat *);

void  BinaryUnDumpFromFile(eTypeGpsDegFormat &,ELISE_fp &);

typedef enum
{
  el1,
  el1_l2,
  el1_l2_l5,
  el1_l2_l5_l6,
  el1_l2_l5_l6_l7,
  eNbTypeGpsFreq
} eTypeGpsFreq;
void xml_init(eTypeGpsFreq & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsFreq & aVal);

eTypeGpsFreq  Str2eTypeGpsFreq(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsFreq & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsFreq &);

std::string  Mangling( eTypeGpsFreq *);

void  BinaryUnDumpFromFile(eTypeGpsFreq &,ELISE_fp &);

typedef enum
{
  eforward,
  ebackward,
  ecombined,
  eNbTypeGpsSol
} eTypeGpsSol;
void xml_init(eTypeGpsSol & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsSol & aVal);

eTypeGpsSol  Str2eTypeGpsSol(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsSol & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsSol &);

std::string  Mangling( eTypeGpsSol *);

void  BinaryUnDumpFromFile(eTypeGpsSol &,ELISE_fp &);

typedef enum
{
  eNav,
  eprecise,
  ebrdc_sbas,
  ebrdc_ssrapc,
  ebrdc_ssrcom,
  eNbTypeGpsEphe
} eTypeGpsEphe;
void xml_init(eTypeGpsEphe & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsEphe & aVal);

eTypeGpsEphe  Str2eTypeGpsEphe(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsEphe & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsEphe &);

std::string  Mangling( eTypeGpsEphe *);

void  BinaryUnDumpFromFile(eTypeGpsEphe &,ELISE_fp &);

typedef enum
{
  eNONE,
  econtinuous,
  einstantaneous,
  efix_and_hold,
  eNbTypeGpsAmbRes
} eTypeGpsAmbRes;
void xml_init(eTypeGpsAmbRes & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsAmbRes & aVal);

eTypeGpsAmbRes  Str2eTypeGpsAmbRes(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsAmbRes & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsAmbRes &);

std::string  Mangling( eTypeGpsAmbRes *);

void  BinaryUnDumpFromFile(eTypeGpsAmbRes &,ELISE_fp &);

typedef enum
{
  eOFF,
  eon,
  eautocal,
  eNbTypeGloAmbRes
} eTypeGloAmbRes;
void xml_init(eTypeGloAmbRes & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGloAmbRes & aVal);

eTypeGloAmbRes  Str2eTypeGloAmbRes(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGloAmbRes & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGloAmbRes &);

std::string  Mangling( eTypeGloAmbRes *);

void  BinaryUnDumpFromFile(eTypeGloAmbRes &,ELISE_fp &);

typedef enum
{
  eoff,
  ebrdc,
  esbas,
  edual_freq,
  eest_stec,
  eionex_tec,
  eqzs_brdc,
  eqzs_lex,
  evtec_sf,
  evtec_ef,
  egtec,
  eNbTypeGpsIonoCorr
} eTypeGpsIonoCorr;
void xml_init(eTypeGpsIonoCorr & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsIonoCorr & aVal);

eTypeGpsIonoCorr  Str2eTypeGpsIonoCorr(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsIonoCorr & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsIonoCorr &);

std::string  Mangling( eTypeGpsIonoCorr *);

void  BinaryUnDumpFromFile(eTypeGpsIonoCorr &,ELISE_fp &);

typedef enum
{
  enull,
  esaas,
  eSBAS,
  eest_ztd,
  eest_ztdgrad,
  eNbTypeGpsTropoCorr
} eTypeGpsTropoCorr;
void xml_init(eTypeGpsTropoCorr & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsTropoCorr & aVal);

eTypeGpsTropoCorr  Str2eTypeGpsTropoCorr(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsTropoCorr & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsTropoCorr &);

std::string  Mangling( eTypeGpsTropoCorr *);

void  BinaryUnDumpFromFile(eTypeGpsTropoCorr &,ELISE_fp &);

typedef enum
{
  eLLH,
  eXYZ,
  ecode,
  eposfile,
  erinexhead,
  ertcm,
  eNbGpsAntPos
} eTypeGpsAntPos;
void xml_init(eTypeGpsAntPos & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeGpsAntPos & aVal);

eTypeGpsAntPos  Str2eTypeGpsAntPos(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeGpsAntPos & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeGpsAntPos &);

std::string  Mangling( eTypeGpsAntPos *);

void  BinaryUnDumpFromFile(eTypeGpsAntPos &,ELISE_fp &);

typedef enum
{
  eTMalt_Ortho,
  eTMalt_UrbanMNE,
  eTMalt_GeomImage,
  eTMalt_NbVals
} eNewTypeMalt;
void xml_init(eNewTypeMalt & aVal,cElXMLTree * aTree);
std::string  eToString(const eNewTypeMalt & aVal);

eNewTypeMalt  Str2eNewTypeMalt(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eNewTypeMalt & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eNewTypeMalt &);

std::string  Mangling( eNewTypeMalt *);

void  BinaryUnDumpFromFile(eNewTypeMalt &,ELISE_fp &);

typedef enum
{
  eTT_RadialBasic,
  eTT_RadialExtended,
  eTT_Fraser,
  eTT_FishEyeEqui,
  eTT_AutoCal,
  eTT_Figee,
  eTT_HemiEqui,
  eTT_RadialStd,
  eTT_FraserBasic,
  eTT_FishEyeBasic,
  eTT_FE_EquiSolBasic,
  eTT_RadGen7x2,
  eTT_RadGen11x2,
  eTT_RadGen15x2,
  eTT_RadGen19x2,
  eTT_NbVals
} eTypeTapas;
void xml_init(eTypeTapas & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeTapas & aVal);

eTypeTapas  Str2eTypeTapas(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeTapas & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeTapas &);

std::string  Mangling( eTypeTapas *);

void  BinaryUnDumpFromFile(eTypeTapas &,ELISE_fp &);

typedef enum
{
  eBBA,
  eSBBA,
  eSBBAFus,
  eUndefVal
} eTypeOriVid;
void xml_init(eTypeOriVid & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeOriVid & aVal);

eTypeOriVid  Str2eTypeOriVid(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeOriVid & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeOriVid &);

std::string  Mangling( eTypeOriVid *);

void  BinaryUnDumpFromFile(eTypeOriVid &,ELISE_fp &);

typedef enum
{
  eGround,
  eStatue,
  eForest,
  eTestIGN,
  eQuickMac,
  eMicMac,
  eBigMac,
  eMTDTmp,
  eNbTypeMMByP
} eTypeMMByP;
void xml_init(eTypeMMByP & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeMMByP & aVal);

eTypeMMByP  Str2eTypeMMByP(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeMMByP & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeMMByP &);

std::string  Mangling( eTypeMMByP *);

void  BinaryUnDumpFromFile(eTypeMMByP &,ELISE_fp &);

typedef enum
{
  eQual_High,
  eQual_Average,
  eQual_Low,
  eNbTypeQual
} eTypeQuality;
void xml_init(eTypeQuality & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeQuality & aVal);

eTypeQuality  Str2eTypeQuality(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeQuality & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeQuality &);

std::string  Mangling( eTypeQuality *);

void  BinaryUnDumpFromFile(eTypeQuality &,ELISE_fp &);

typedef enum
{
  eOrtho,
  eUrbanMNE,
  eGeomImage,
  eNbTypesMNE
} eTypeMalt;
void xml_init(eTypeMalt & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeMalt & aVal);

eTypeMalt  Str2eTypeMalt(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeMalt & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeMalt &);

std::string  Mangling( eTypeMalt *);

void  BinaryUnDumpFromFile(eTypeMalt &,ELISE_fp &);

typedef enum
{
  eAppEgels,
  eAppGeoCub,
  eAppInFile,
  eAppXML,
  eNbTypeApp
} eTypeFichierApp;
void xml_init(eTypeFichierApp & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeFichierApp & aVal);

eTypeFichierApp  Str2eTypeFichierApp(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeFichierApp & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeFichierApp &);

std::string  Mangling( eTypeFichierApp *);

void  BinaryUnDumpFromFile(eTypeFichierApp &,ELISE_fp &);

typedef enum
{
  eOriTxtAgiSoft,
  eOriBluh,
  eOriTxtInFile,
  eNbTypeOriTxt
} eTypeFichierOriTxt;
void xml_init(eTypeFichierOriTxt & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeFichierOriTxt & aVal);

eTypeFichierOriTxt  Str2eTypeFichierOriTxt(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeFichierOriTxt & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeFichierOriTxt &);

std::string  Mangling( eTypeFichierOriTxt *);

void  BinaryUnDumpFromFile(eTypeFichierOriTxt &,ELISE_fp &);

typedef enum
{
  eImpaintL2,
  eImpaintMNT
} eImpaintMethod;
void xml_init(eImpaintMethod & aVal,cElXMLTree * aTree);
std::string  eToString(const eImpaintMethod & aVal);

eImpaintMethod  Str2eImpaintMethod(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eImpaintMethod & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eImpaintMethod &);

std::string  Mangling( eImpaintMethod *);

void  BinaryUnDumpFromFile(eImpaintMethod &,ELISE_fp &);

typedef enum
{
  eTN_u_int1,
  eTN_int1,
  eTN_u_int2,
  eTN_int2,
  eTN_int4,
  eTN_float,
  eTN_double,
  eTN_Bits1MSBF
} eTypeNumerique;
void xml_init(eTypeNumerique & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeNumerique & aVal);

eTypeNumerique  Str2eTypeNumerique(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeNumerique & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeNumerique &);

std::string  Mangling( eTypeNumerique *);

void  BinaryUnDumpFromFile(eTypeNumerique &,ELISE_fp &);

typedef enum
{
  eComprTiff_None,
  eComprTiff_LZW,
  eComprTiff_FAX4,
  eComprTiff_PackBits
} eComprTiff;
void xml_init(eComprTiff & aVal,cElXMLTree * aTree);
std::string  eToString(const eComprTiff & aVal);

eComprTiff  Str2eComprTiff(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eComprTiff & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eComprTiff &);

std::string  Mangling( eComprTiff *);

void  BinaryUnDumpFromFile(eComprTiff &,ELISE_fp &);

typedef enum
{
  ePCR_Atgt,
  ePCR_2SinAtgtS2,
  ePCR_Stereographik
} eTypePreCondRad;
void xml_init(eTypePreCondRad & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypePreCondRad & aVal);

eTypePreCondRad  Str2eTypePreCondRad(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePreCondRad & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypePreCondRad &);

std::string  Mangling( eTypePreCondRad *);

void  BinaryUnDumpFromFile(eTypePreCondRad &,ELISE_fp &);

typedef enum
{
  eDEM,
  eOrthoIm,
  eNbTypeVals
} eTypeSake;
void xml_init(eTypeSake & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeSake & aVal);

eTypeSake  Str2eTypeSake(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeSake & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeSake &);

std::string  Mangling( eTypeSake *);

void  BinaryUnDumpFromFile(eTypeSake &,ELISE_fp &);

typedef enum
{
  eGeomMNTCarto,
  eGeomMNTEuclid,
  eGeomMNTFaisceauIm1PrCh_Px1D,
  eGeomMNTFaisceauIm1PrCh_Px2D,
  eGeomMNTFaisceauIm1ZTerrain_Px1D,
  eGeomMNTFaisceauIm1ZTerrain_Px2D,
  eGeomPxBiDim,
  eNoGeomMNT,
  eGeomMNTFaisceauPrChSpherik
} eModeGeomMNT;
void xml_init(eModeGeomMNT & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeGeomMNT & aVal);

eModeGeomMNT  Str2eModeGeomMNT(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeGeomMNT & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModeGeomMNT &);

std::string  Mangling( eModeGeomMNT *);

void  BinaryUnDumpFromFile(eModeGeomMNT &,ELISE_fp &);

typedef enum
{
  eModeLeBrisPP,
  eModeAutopano
} eModeBinSift;
void xml_init(eModeBinSift & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeBinSift & aVal);

eModeBinSift  Str2eModeBinSift(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeBinSift & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModeBinSift &);

std::string  Mangling( eModeBinSift *);

void  BinaryUnDumpFromFile(eModeBinSift &,ELISE_fp &);

typedef enum
{
  eSysPlein,
  eSysCreuxMap,
  eSysCreuxFixe,
  eSysL1Barrodale,
  eSysL2BlocSym
} eModeSolveurEq;
void xml_init(eModeSolveurEq & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeSolveurEq & aVal);

eModeSolveurEq  Str2eModeSolveurEq(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeSolveurEq & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModeSolveurEq &);

std::string  Mangling( eModeSolveurEq *);

void  BinaryUnDumpFromFile(eModeSolveurEq &,ELISE_fp &);

typedef enum
{
  eUniteAngleDegre,
  eUniteAngleGrade,
  eUniteAngleRadian,
  eUniteAngleUnknown
} eUniteAngulaire;
void xml_init(eUniteAngulaire & aVal,cElXMLTree * aTree);
std::string  eToString(const eUniteAngulaire & aVal);

eUniteAngulaire  Str2eUniteAngulaire(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eUniteAngulaire & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eUniteAngulaire &);

std::string  Mangling( eUniteAngulaire *);

void  BinaryUnDumpFromFile(eUniteAngulaire &,ELISE_fp &);

typedef enum
{
  eCPPFiges,
  eCPPLies,
  eCPPLibres
} eDegreLiberteCPP;
void xml_init(eDegreLiberteCPP & aVal,cElXMLTree * aTree);
std::string  eToString(const eDegreLiberteCPP & aVal);

eDegreLiberteCPP  Str2eDegreLiberteCPP(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eDegreLiberteCPP & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eDegreLiberteCPP &);

std::string  Mangling( eDegreLiberteCPP *);

void  BinaryUnDumpFromFile(eDegreLiberteCPP &,ELISE_fp &);

typedef enum
{
  eModeleEbner,
  eModeleDCBrown,
  eModelePolyDeg2,
  eModelePolyDeg3,
  eModelePolyDeg4,
  eModelePolyDeg5,
  eModelePolyDeg6,
  eModelePolyDeg7,
  eModele_FishEye_10_5_5,
  eModele_EquiSolid_FishEye_10_5_5,
  eModele_DRad_PPaEqPPs,
  eModele_Fraser_PPaEqPPs,
  eModeleRadFour7x2,
  eModeleRadFour11x2,
  eModeleRadFour15x2,
  eModeleRadFour19x2,
  eModelePolyDeg0,
  eModelePolyDeg1,
  eModele_Stereographik_FishEye_10_5_5
} eModelesCalibUnif;
void xml_init(eModelesCalibUnif & aVal,cElXMLTree * aTree);
std::string  eToString(const eModelesCalibUnif & aVal);

eModelesCalibUnif  Str2eModelesCalibUnif(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModelesCalibUnif & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModelesCalibUnif &);

std::string  Mangling( eModelesCalibUnif *);

void  BinaryUnDumpFromFile(eModelesCalibUnif &,ELISE_fp &);

typedef enum
{
  eProjStenope,
  eProjOrthographique,
  eProjGrid
} eTypeProjectionCam;
void xml_init(eTypeProjectionCam & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeProjectionCam & aVal);

eTypeProjectionCam  Str2eTypeProjectionCam(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeProjectionCam & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeProjectionCam &);

std::string  Mangling( eTypeProjectionCam *);

void  BinaryUnDumpFromFile(eTypeProjectionCam &,ELISE_fp &);

typedef enum
{
  eTC_WGS84,
  eTC_GeoCentr,
  eTC_RTL,
  eTC_Polyn,
  eTC_Unknown,
  eTC_Lambert93,
  eTC_LambertCC,
  eTC_Proj4
} eTypeCoord;
void xml_init(eTypeCoord & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeCoord & aVal);

eTypeCoord  Str2eTypeCoord(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeCoord & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeCoord &);

std::string  Mangling( eTypeCoord *);

void  BinaryUnDumpFromFile(eTypeCoord &,ELISE_fp &);

class cMicMacConfiguration
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMicMacConfiguration & anObj,cElXMLTree * aTree);


        std::string & DirInstall();
        const std::string & DirInstall()const ;

        int & NbProcess();
        const int & NbProcess()const ;
    private:
        std::string mDirInstall;
        int mNbProcess;
};
cElXMLTree * ToXMLTree(const cMicMacConfiguration &);

void  BinaryDumpInFile(ELISE_fp &,const cMicMacConfiguration &);

void  BinaryUnDumpFromFile(cMicMacConfiguration &,ELISE_fp &);

std::string  Mangling( cMicMacConfiguration *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBasicSystemeCoord
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasicSystemeCoord & anObj,cElXMLTree * aTree);


        eTypeCoord & TypeCoord();
        const eTypeCoord & TypeCoord()const ;

        std::vector< double > & AuxR();
        const std::vector< double > & AuxR()const ;

        std::vector< int > & AuxI();
        const std::vector< int > & AuxI()const ;

        std::vector< std::string > & AuxStr();
        const std::vector< std::string > & AuxStr()const ;

        cTplValGesInit< bool > & ByFile();
        const cTplValGesInit< bool > & ByFile()const ;

        std::vector< eUniteAngulaire > & AuxRUnite();
        const std::vector< eUniteAngulaire > & AuxRUnite()const ;
    private:
        eTypeCoord mTypeCoord;
        std::vector< double > mAuxR;
        std::vector< int > mAuxI;
        std::vector< std::string > mAuxStr;
        cTplValGesInit< bool > mByFile;
        std::vector< eUniteAngulaire > mAuxRUnite;
};
cElXMLTree * ToXMLTree(const cBasicSystemeCoord &);

void  BinaryDumpInFile(ELISE_fp &,const cBasicSystemeCoord &);

void  BinaryUnDumpFromFile(cBasicSystemeCoord &,ELISE_fp &);

std::string  Mangling( cBasicSystemeCoord *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSystemeCoord
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSystemeCoord & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Comment();
        const cTplValGesInit< std::string > & Comment()const ;

        std::vector< cBasicSystemeCoord > & BSC();
        const std::vector< cBasicSystemeCoord > & BSC()const ;
    private:
        cTplValGesInit< std::string > mComment;
        std::vector< cBasicSystemeCoord > mBSC;
};
cElXMLTree * ToXMLTree(const cSystemeCoord &);

void  BinaryDumpInFile(ELISE_fp &,const cSystemeCoord &);

void  BinaryUnDumpFromFile(cSystemeCoord &,ELISE_fp &);

std::string  Mangling( cSystemeCoord *);

/******************************************************/
/******************************************************/
/******************************************************/
class cChangementCoordonnees
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChangementCoordonnees & anObj,cElXMLTree * aTree);


        cSystemeCoord & SystemeSource();
        const cSystemeCoord & SystemeSource()const ;

        cSystemeCoord & SystemeCible();
        const cSystemeCoord & SystemeCible()const ;
    private:
        cSystemeCoord mSystemeSource;
        cSystemeCoord mSystemeCible;
};
cElXMLTree * ToXMLTree(const cChangementCoordonnees &);

void  BinaryDumpInFile(ELISE_fp &,const cChangementCoordonnees &);

void  BinaryUnDumpFromFile(cChangementCoordonnees &,ELISE_fp &);

std::string  Mangling( cChangementCoordonnees *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFileOriMnt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFileOriMnt & anObj,cElXMLTree * aTree);


        std::string & NameFileMnt();
        const std::string & NameFileMnt()const ;

        cTplValGesInit< std::string > & NameFileMasque();
        const cTplValGesInit< std::string > & NameFileMasque()const ;

        Pt2di & NombrePixels();
        const Pt2di & NombrePixels()const ;

        Pt2dr & OriginePlani();
        const Pt2dr & OriginePlani()const ;

        Pt2dr & ResolutionPlani();
        const Pt2dr & ResolutionPlani()const ;

        double & OrigineAlti();
        const double & OrigineAlti()const ;

        double & ResolutionAlti();
        const double & ResolutionAlti()const ;

        cTplValGesInit< int > & NumZoneLambert();
        const cTplValGesInit< int > & NumZoneLambert()const ;

        eModeGeomMNT & Geometrie();
        const eModeGeomMNT & Geometrie()const ;

        cTplValGesInit< Pt2dr > & OrigineTgtLoc();
        const cTplValGesInit< Pt2dr > & OrigineTgtLoc()const ;

        cTplValGesInit< int > & Rounding();
        const cTplValGesInit< int > & Rounding()const ;
    private:
        std::string mNameFileMnt;
        cTplValGesInit< std::string > mNameFileMasque;
        Pt2di mNombrePixels;
        Pt2dr mOriginePlani;
        Pt2dr mResolutionPlani;
        double mOrigineAlti;
        double mResolutionAlti;
        cTplValGesInit< int > mNumZoneLambert;
        eModeGeomMNT mGeometrie;
        cTplValGesInit< Pt2dr > mOrigineTgtLoc;
        cTplValGesInit< int > mRounding;
};
cElXMLTree * ToXMLTree(const cFileOriMnt &);

void  BinaryDumpInFile(ELISE_fp &,const cFileOriMnt &);

void  BinaryUnDumpFromFile(cFileOriMnt &,ELISE_fp &);

std::string  Mangling( cFileOriMnt *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRefPlani
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRefPlani & anObj,cElXMLTree * aTree);


        Pt2dr & Origine();
        const Pt2dr & Origine()const ;

        Pt2dr & Resolution();
        const Pt2dr & Resolution()const ;
    private:
        Pt2dr mOrigine;
        Pt2dr mResolution;
};
cElXMLTree * ToXMLTree(const cRefPlani &);

void  BinaryDumpInFile(ELISE_fp &,const cRefPlani &);

void  BinaryUnDumpFromFile(cRefPlani &,ELISE_fp &);

std::string  Mangling( cRefPlani *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRefAlti
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRefAlti & anObj,cElXMLTree * aTree);


        double & Origine();
        const double & Origine()const ;

        double & Resolution();
        const double & Resolution()const ;
    private:
        double mOrigine;
        double mResolution;
};
cElXMLTree * ToXMLTree(const cRefAlti &);

void  BinaryDumpInFile(ELISE_fp &,const cRefAlti &);

void  BinaryUnDumpFromFile(cRefAlti &,ELISE_fp &);

std::string  Mangling( cRefAlti *);

class cGestionAltimetrie
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGestionAltimetrie & anObj,cElXMLTree * aTree);


        cTplValGesInit< cRefAlti > & RefAlti();
        const cTplValGesInit< cRefAlti > & RefAlti()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;
    private:
        cTplValGesInit< cRefAlti > mRefAlti;
        cTplValGesInit< double > mZMoyen;
};
cElXMLTree * ToXMLTree(const cGestionAltimetrie &);

void  BinaryDumpInFile(ELISE_fp &,const cGestionAltimetrie &);

void  BinaryUnDumpFromFile(cGestionAltimetrie &,ELISE_fp &);

std::string  Mangling( cGestionAltimetrie *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlGeoRefFile
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlGeoRefFile & anObj,cElXMLTree * aTree);


        cTplValGesInit< cSystemeCoord > & SysCo();
        const cTplValGesInit< cSystemeCoord > & SysCo()const ;

        cRefPlani & RefPlani();
        const cRefPlani & RefPlani()const ;

        cTplValGesInit< cRefAlti > & RefAlti();
        const cTplValGesInit< cRefAlti > & RefAlti()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        cGestionAltimetrie & GestionAltimetrie();
        const cGestionAltimetrie & GestionAltimetrie()const ;
    private:
        cTplValGesInit< cSystemeCoord > mSysCo;
        cRefPlani mRefPlani;
        cGestionAltimetrie mGestionAltimetrie;
};
cElXMLTree * ToXMLTree(const cXmlGeoRefFile &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlGeoRefFile &);

void  BinaryUnDumpFromFile(cXmlGeoRefFile &,ELISE_fp &);

std::string  Mangling( cXmlGeoRefFile *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSpecExtractFromFile
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSpecExtractFromFile & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        std::string & NameTag();
        const std::string & NameTag()const ;

        cTplValGesInit< bool > & AutorizeNonExisting();
        const cTplValGesInit< bool > & AutorizeNonExisting()const ;
    private:
        std::string mNameFile;
        std::string mNameTag;
        cTplValGesInit< bool > mAutorizeNonExisting;
};
cElXMLTree * ToXMLTree(const cSpecExtractFromFile &);

void  BinaryDumpInFile(ELISE_fp &,const cSpecExtractFromFile &);

void  BinaryUnDumpFromFile(cSpecExtractFromFile &,ELISE_fp &);

std::string  Mangling( cSpecExtractFromFile *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSpecifFormatRaw
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSpecifFormatRaw & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NameFile();
        const cTplValGesInit< std::string > & NameFile()const ;

        Pt2di & Sz();
        const Pt2di & Sz()const ;

        bool & MSBF();
        const bool & MSBF()const ;

        int & NbBitsParPixel();
        const int & NbBitsParPixel()const ;

        bool & IntegerType();
        const bool & IntegerType()const ;

        bool & SignedType();
        const bool & SignedType()const ;

        cTplValGesInit< int > & Offset();
        const cTplValGesInit< int > & Offset()const ;

        cTplValGesInit< std::string > & Camera();
        const cTplValGesInit< std::string > & Camera()const ;

        cTplValGesInit< std::string > & BayPat();
        const cTplValGesInit< std::string > & BayPat()const ;

        cTplValGesInit< double > & Focalmm();
        const cTplValGesInit< double > & Focalmm()const ;

        cTplValGesInit< double > & FocalEqui35();
        const cTplValGesInit< double > & FocalEqui35()const ;
    private:
        cTplValGesInit< std::string > mNameFile;
        Pt2di mSz;
        bool mMSBF;
        int mNbBitsParPixel;
        bool mIntegerType;
        bool mSignedType;
        cTplValGesInit< int > mOffset;
        cTplValGesInit< std::string > mCamera;
        cTplValGesInit< std::string > mBayPat;
        cTplValGesInit< double > mFocalmm;
        cTplValGesInit< double > mFocalEqui35;
};
cElXMLTree * ToXMLTree(const cSpecifFormatRaw &);

void  BinaryDumpInFile(ELISE_fp &,const cSpecifFormatRaw &);

void  BinaryUnDumpFromFile(cSpecifFormatRaw &,ELISE_fp &);

std::string  Mangling( cSpecifFormatRaw *);

/******************************************************/
/******************************************************/
/******************************************************/
typedef enum
{
  eTotoGeomMECIm1
} eTotoModeGeomMEC;
void xml_init(eTotoModeGeomMEC & aVal,cElXMLTree * aTree);
std::string  eToString(const eTotoModeGeomMEC & aVal);

eTotoModeGeomMEC  Str2eTotoModeGeomMEC(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTotoModeGeomMEC & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTotoModeGeomMEC &);

std::string  Mangling( eTotoModeGeomMEC *);

void  BinaryUnDumpFromFile(eTotoModeGeomMEC &,ELISE_fp &);

class cCM_Set
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCM_Set & anObj,cElXMLTree * aTree);


        std::string & KeySet();
        const std::string & KeySet()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        std::string & NameVarMap();
        const std::string & NameVarMap()const ;
    private:
        std::string mKeySet;
        cTplValGesInit< std::string > mKeyAssoc;
        std::string mNameVarMap;
};
cElXMLTree * ToXMLTree(const cCM_Set &);

void  BinaryDumpInFile(ELISE_fp &,const cCM_Set &);

void  BinaryUnDumpFromFile(cCM_Set &,ELISE_fp &);

std::string  Mangling( cCM_Set *);

class cModeCmdMapeur
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModeCmdMapeur & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & CM_One();
        const cTplValGesInit< std::string > & CM_One()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        std::string & NameVarMap();
        const std::string & NameVarMap()const ;

        cTplValGesInit< cCM_Set > & CM_Set();
        const cTplValGesInit< cCM_Set > & CM_Set()const ;
    private:
        cTplValGesInit< std::string > mCM_One;
        cTplValGesInit< cCM_Set > mCM_Set;
};
cElXMLTree * ToXMLTree(const cModeCmdMapeur &);

void  BinaryDumpInFile(ELISE_fp &,const cModeCmdMapeur &);

void  BinaryUnDumpFromFile(cModeCmdMapeur &,ELISE_fp &);

std::string  Mangling( cModeCmdMapeur *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCmdMapRel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCmdMapRel & anObj,cElXMLTree * aTree);


        std::string & KeyRel();
        const std::string & KeyRel()const ;

        std::string & NameArc();
        const std::string & NameArc()const ;
    private:
        std::string mKeyRel;
        std::string mNameArc;
};
cElXMLTree * ToXMLTree(const cCmdMapRel &);

void  BinaryDumpInFile(ELISE_fp &,const cCmdMapRel &);

void  BinaryUnDumpFromFile(cCmdMapRel &,ELISE_fp &);

std::string  Mangling( cCmdMapRel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCMVA
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCMVA & anObj,cElXMLTree * aTree);


        std::list< cCpleString > & NV();
        const std::list< cCpleString > & NV()const ;
    private:
        std::list< cCpleString > mNV;
};
cElXMLTree * ToXMLTree(const cCMVA &);

void  BinaryDumpInFile(ELISE_fp &,const cCMVA &);

void  BinaryUnDumpFromFile(cCMVA &,ELISE_fp &);

std::string  Mangling( cCMVA *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCmdMappeur
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCmdMappeur & anObj,cElXMLTree * aTree);


        bool & ActivateCmdMap();
        const bool & ActivateCmdMap()const ;

        cTplValGesInit< std::string > & CM_One();
        const cTplValGesInit< std::string > & CM_One()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        std::string & NameVarMap();
        const std::string & NameVarMap()const ;

        cTplValGesInit< cCM_Set > & CM_Set();
        const cTplValGesInit< cCM_Set > & CM_Set()const ;

        cModeCmdMapeur & ModeCmdMapeur();
        const cModeCmdMapeur & ModeCmdMapeur()const ;

        std::string & KeyRel();
        const std::string & KeyRel()const ;

        std::string & NameArc();
        const std::string & NameArc()const ;

        cTplValGesInit< cCmdMapRel > & CmdMapRel();
        const cTplValGesInit< cCmdMapRel > & CmdMapRel()const ;

        std::list< cCMVA > & CMVA();
        const std::list< cCMVA > & CMVA()const ;

        cTplValGesInit< std::string > & ByMkF();
        const cTplValGesInit< std::string > & ByMkF()const ;

        cTplValGesInit< std::string > & KeyTargetMkF();
        const cTplValGesInit< std::string > & KeyTargetMkF()const ;
    private:
        bool mActivateCmdMap;
        cModeCmdMapeur mModeCmdMapeur;
        cTplValGesInit< cCmdMapRel > mCmdMapRel;
        std::list< cCMVA > mCMVA;
        cTplValGesInit< std::string > mByMkF;
        cTplValGesInit< std::string > mKeyTargetMkF;
};
cElXMLTree * ToXMLTree(const cCmdMappeur &);

void  BinaryDumpInFile(ELISE_fp &,const cCmdMappeur &);

void  BinaryUnDumpFromFile(cCmdMappeur &,ELISE_fp &);

std::string  Mangling( cCmdMappeur *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneCmdPar
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneCmdPar & anObj,cElXMLTree * aTree);


        std::list< std::string > & OneCmdSer();
        const std::list< std::string > & OneCmdSer()const ;
    private:
        std::list< std::string > mOneCmdSer;
};
cElXMLTree * ToXMLTree(const cOneCmdPar &);

void  BinaryDumpInFile(ELISE_fp &,const cOneCmdPar &);

void  BinaryUnDumpFromFile(cOneCmdPar &,ELISE_fp &);

std::string  Mangling( cOneCmdPar *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCmdExePar
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCmdExePar & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NameMkF();
        const cTplValGesInit< std::string > & NameMkF()const ;

        std::list< cOneCmdPar > & OneCmdPar();
        const std::list< cOneCmdPar > & OneCmdPar()const ;
    private:
        cTplValGesInit< std::string > mNameMkF;
        std::list< cOneCmdPar > mOneCmdPar;
};
cElXMLTree * ToXMLTree(const cCmdExePar &);

void  BinaryDumpInFile(ELISE_fp &,const cCmdExePar &);

void  BinaryUnDumpFromFile(cCmdExePar &,ELISE_fp &);

std::string  Mangling( cCmdExePar *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPt3drEntries
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPt3drEntries & anObj,cElXMLTree * aTree);


        std::string & Key();
        const std::string & Key()const ;

        Pt3dr & Val();
        const Pt3dr & Val()const ;
    private:
        std::string mKey;
        Pt3dr mVal;
};
cElXMLTree * ToXMLTree(const cPt3drEntries &);

void  BinaryDumpInFile(ELISE_fp &,const cPt3drEntries &);

void  BinaryUnDumpFromFile(cPt3drEntries &,ELISE_fp &);

std::string  Mangling( cPt3drEntries *);

class cBasesPt3dr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasesPt3dr & anObj,cElXMLTree * aTree);


        std::string & NameBase();
        const std::string & NameBase()const ;

        std::list< cPt3drEntries > & Pt3drEntries();
        const std::list< cPt3drEntries > & Pt3drEntries()const ;
    private:
        std::string mNameBase;
        std::list< cPt3drEntries > mPt3drEntries;
};
cElXMLTree * ToXMLTree(const cBasesPt3dr &);

void  BinaryDumpInFile(ELISE_fp &,const cBasesPt3dr &);

void  BinaryUnDumpFromFile(cBasesPt3dr &,ELISE_fp &);

std::string  Mangling( cBasesPt3dr *);

/******************************************************/
/******************************************************/
/******************************************************/
class cScalEntries
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cScalEntries & anObj,cElXMLTree * aTree);


        std::string & Key();
        const std::string & Key()const ;

        double & Val();
        const double & Val()const ;
    private:
        std::string mKey;
        double mVal;
};
cElXMLTree * ToXMLTree(const cScalEntries &);

void  BinaryDumpInFile(ELISE_fp &,const cScalEntries &);

void  BinaryUnDumpFromFile(cScalEntries &,ELISE_fp &);

std::string  Mangling( cScalEntries *);

class cBasesScal
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasesScal & anObj,cElXMLTree * aTree);


        std::string & NameBase();
        const std::string & NameBase()const ;

        std::list< cScalEntries > & ScalEntries();
        const std::list< cScalEntries > & ScalEntries()const ;
    private:
        std::string mNameBase;
        std::list< cScalEntries > mScalEntries;
};
cElXMLTree * ToXMLTree(const cBasesScal &);

void  BinaryDumpInFile(ELISE_fp &,const cBasesScal &);

void  BinaryUnDumpFromFile(cBasesScal &,ELISE_fp &);

std::string  Mangling( cBasesScal *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBaseDataCD
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBaseDataCD & anObj,cElXMLTree * aTree);


        std::list< cBasesPt3dr > & BasesPt3dr();
        const std::list< cBasesPt3dr > & BasesPt3dr()const ;

        std::list< cBasesScal > & BasesScal();
        const std::list< cBasesScal > & BasesScal()const ;
    private:
        std::list< cBasesPt3dr > mBasesPt3dr;
        std::list< cBasesScal > mBasesScal;
};
cElXMLTree * ToXMLTree(const cBaseDataCD &);

void  BinaryDumpInFile(ELISE_fp &,const cBaseDataCD &);

void  BinaryUnDumpFromFile(cBaseDataCD &,ELISE_fp &);

std::string  Mangling( cBaseDataCD *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamVolChantierPhotogram
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamVolChantierPhotogram & anObj,cElXMLTree * aTree);


        std::string & Directory();
        const std::string & Directory()const ;

        cTplValGesInit< std::string > & DirOrientations();
        const cTplValGesInit< std::string > & DirOrientations()const ;

        cElRegex_Ptr & NameSelector();
        const cElRegex_Ptr & NameSelector()const ;

        cElRegex_Ptr & BandeIdSelector();
        const cElRegex_Ptr & BandeIdSelector()const ;

        std::string & NomBandeId();
        const std::string & NomBandeId()const ;

        std::string & NomIdInBande();
        const std::string & NomIdInBande()const ;

        std::string & NomImage();
        const std::string & NomImage()const ;

        cTplValGesInit< std::string > & DirImages();
        const cTplValGesInit< std::string > & DirImages()const ;
    private:
        std::string mDirectory;
        cTplValGesInit< std::string > mDirOrientations;
        cElRegex_Ptr mNameSelector;
        cElRegex_Ptr mBandeIdSelector;
        std::string mNomBandeId;
        std::string mNomIdInBande;
        std::string mNomImage;
        cTplValGesInit< std::string > mDirImages;
};
cElXMLTree * ToXMLTree(const cParamVolChantierPhotogram &);

void  BinaryDumpInFile(ELISE_fp &,const cParamVolChantierPhotogram &);

void  BinaryUnDumpFromFile(cParamVolChantierPhotogram &,ELISE_fp &);

std::string  Mangling( cParamVolChantierPhotogram *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamChantierPhotogram
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamChantierPhotogram & anObj,cElXMLTree * aTree);


        std::list< cParamVolChantierPhotogram > & ParamVolChantierPhotogram();
        const std::list< cParamVolChantierPhotogram > & ParamVolChantierPhotogram()const ;
    private:
        std::list< cParamVolChantierPhotogram > mParamVolChantierPhotogram;
};
cElXMLTree * ToXMLTree(const cParamChantierPhotogram &);

void  BinaryDumpInFile(ELISE_fp &,const cParamChantierPhotogram &);

void  BinaryUnDumpFromFile(cParamChantierPhotogram &,ELISE_fp &);

std::string  Mangling( cParamChantierPhotogram *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPDV
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPDV & anObj,cElXMLTree * aTree);


        std::string & Im();
        const std::string & Im()const ;

        std::string & Orient();
        const std::string & Orient()const ;

        std::string & IdInBande();
        const std::string & IdInBande()const ;

        std::string & Bande();
        const std::string & Bande()const ;
    private:
        std::string mIm;
        std::string mOrient;
        std::string mIdInBande;
        std::string mBande;
};
cElXMLTree * ToXMLTree(const cPDV &);

void  BinaryDumpInFile(ELISE_fp &,const cPDV &);

void  BinaryUnDumpFromFile(cPDV &,ELISE_fp &);

std::string  Mangling( cPDV *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBandesChantierPhotogram
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBandesChantierPhotogram & anObj,cElXMLTree * aTree);


        std::string & IdBande();
        const std::string & IdBande()const ;

        std::list< cPDV > & PDVs();
        const std::list< cPDV > & PDVs()const ;
    private:
        std::string mIdBande;
        std::list< cPDV > mPDVs;
};
cElXMLTree * ToXMLTree(const cBandesChantierPhotogram &);

void  BinaryDumpInFile(ELISE_fp &,const cBandesChantierPhotogram &);

void  BinaryUnDumpFromFile(cBandesChantierPhotogram &,ELISE_fp &);

std::string  Mangling( cBandesChantierPhotogram *);

class cVolChantierPhotogram
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVolChantierPhotogram & anObj,cElXMLTree * aTree);


        std::list< cBandesChantierPhotogram > & BandesChantierPhotogram();
        const std::list< cBandesChantierPhotogram > & BandesChantierPhotogram()const ;
    private:
        std::list< cBandesChantierPhotogram > mBandesChantierPhotogram;
};
cElXMLTree * ToXMLTree(const cVolChantierPhotogram &);

void  BinaryDumpInFile(ELISE_fp &,const cVolChantierPhotogram &);

void  BinaryUnDumpFromFile(cVolChantierPhotogram &,ELISE_fp &);

std::string  Mangling( cVolChantierPhotogram *);

/******************************************************/
/******************************************************/
/******************************************************/
class cChantierPhotogram
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChantierPhotogram & anObj,cElXMLTree * aTree);


        std::list< cVolChantierPhotogram > & VolChantierPhotogram();
        const std::list< cVolChantierPhotogram > & VolChantierPhotogram()const ;
    private:
        std::list< cVolChantierPhotogram > mVolChantierPhotogram;
};
cElXMLTree * ToXMLTree(const cChantierPhotogram &);

void  BinaryDumpInFile(ELISE_fp &,const cChantierPhotogram &);

void  BinaryUnDumpFromFile(cChantierPhotogram &,ELISE_fp &);

std::string  Mangling( cChantierPhotogram *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCplePDV
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCplePDV & anObj,cElXMLTree * aTree);


        int & Id1();
        const int & Id1()const ;

        int & Id2();
        const int & Id2()const ;
    private:
        int mId1;
        int mId2;
};
cElXMLTree * ToXMLTree(const cCplePDV &);

void  BinaryDumpInFile(ELISE_fp &,const cCplePDV &);

void  BinaryUnDumpFromFile(cCplePDV &,ELISE_fp &);

std::string  Mangling( cCplePDV *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGraphePdv
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGraphePdv & anObj,cElXMLTree * aTree);


        Box2dr & BoxCh();
        const Box2dr & BoxCh()const ;

        std::vector< cPDV > & PDVs();
        const std::vector< cPDV > & PDVs()const ;

        std::list< cCplePDV > & CplePDV();
        const std::list< cCplePDV > & CplePDV()const ;
    private:
        Box2dr mBoxCh;
        std::vector< cPDV > mPDVs;
        std::list< cCplePDV > mCplePDV;
};
cElXMLTree * ToXMLTree(const cGraphePdv &);

void  BinaryDumpInFile(ELISE_fp &,const cGraphePdv &);

void  BinaryUnDumpFromFile(cGraphePdv &,ELISE_fp &);

std::string  Mangling( cGraphePdv *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCercleRelief
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCercleRelief & anObj,cElXMLTree * aTree);


        double & Rayon();
        const double & Rayon()const ;

        double & Profondeur();
        const double & Profondeur()const ;
    private:
        double mRayon;
        double mProfondeur;
};
cElXMLTree * ToXMLTree(const cCercleRelief &);

void  BinaryDumpInFile(ELISE_fp &,const cCercleRelief &);

void  BinaryUnDumpFromFile(cCercleRelief &,ELISE_fp &);

std::string  Mangling( cCercleRelief *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCibleCalib
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCibleCalib & anObj,cElXMLTree * aTree);


        int & Id();
        const int & Id()const ;

        cTplValGesInit< bool > & Negatif();
        const cTplValGesInit< bool > & Negatif()const ;

        Pt3dr & Position();
        const Pt3dr & Position()const ;

        Pt3dr & Normale();
        const Pt3dr & Normale()const ;

        std::vector< double > & Rayons();
        const std::vector< double > & Rayons()const ;

        bool & Ponctuel();
        const bool & Ponctuel()const ;

        bool & ReliefIsSortant();
        const bool & ReliefIsSortant()const ;

        std::vector< cCercleRelief > & CercleRelief();
        const std::vector< cCercleRelief > & CercleRelief()const ;

        std::string & NomType();
        const std::string & NomType()const ;

        int & Qualite();
        const int & Qualite()const ;

        cTplValGesInit< double > & FacteurElargRechCorrel();
        const cTplValGesInit< double > & FacteurElargRechCorrel()const ;

        cTplValGesInit< double > & FacteurElargRechRaffine();
        const cTplValGesInit< double > & FacteurElargRechRaffine()const ;
    private:
        int mId;
        cTplValGesInit< bool > mNegatif;
        Pt3dr mPosition;
        Pt3dr mNormale;
        std::vector< double > mRayons;
        bool mPonctuel;
        bool mReliefIsSortant;
        std::vector< cCercleRelief > mCercleRelief;
        std::string mNomType;
        int mQualite;
        cTplValGesInit< double > mFacteurElargRechCorrel;
        cTplValGesInit< double > mFacteurElargRechRaffine;
};
cElXMLTree * ToXMLTree(const cCibleCalib &);

void  BinaryDumpInFile(ELISE_fp &,const cCibleCalib &);

void  BinaryUnDumpFromFile(cCibleCalib &,ELISE_fp &);

std::string  Mangling( cCibleCalib *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPolygoneCalib
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPolygoneCalib & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        std::vector< cCibleCalib > & Cibles();
        const std::vector< cCibleCalib > & Cibles()const ;
    private:
        std::string mName;
        std::vector< cCibleCalib > mCibles;
};
cElXMLTree * ToXMLTree(const cPolygoneCalib &);

void  BinaryDumpInFile(ELISE_fp &,const cPolygoneCalib &);

void  BinaryUnDumpFromFile(cPolygoneCalib &,ELISE_fp &);

std::string  Mangling( cPolygoneCalib *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPointesCibleAC
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPointesCibleAC & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        Pt2dr & PtIm();
        const Pt2dr & PtIm()const ;
    private:
        std::string mNameIm;
        Pt2dr mPtIm;
};
cElXMLTree * ToXMLTree(const cPointesCibleAC &);

void  BinaryDumpInFile(ELISE_fp &,const cPointesCibleAC &);

void  BinaryUnDumpFromFile(cPointesCibleAC &,ELISE_fp &);

std::string  Mangling( cPointesCibleAC *);

class cCibleACalcByLiaisons
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCibleACalcByLiaisons & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        std::list< cPointesCibleAC > & PointesCibleAC();
        const std::list< cPointesCibleAC > & PointesCibleAC()const ;
    private:
        std::string mName;
        std::list< cPointesCibleAC > mPointesCibleAC;
};
cElXMLTree * ToXMLTree(const cCibleACalcByLiaisons &);

void  BinaryDumpInFile(ELISE_fp &,const cCibleACalcByLiaisons &);

void  BinaryUnDumpFromFile(cCibleACalcByLiaisons &,ELISE_fp &);

std::string  Mangling( cCibleACalcByLiaisons *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCible2Rech
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCible2Rech & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        std::vector< int > & Id();
        const std::vector< int > & Id()const ;
    private:
        cTplValGesInit< bool > mUseIt;
        std::vector< int > mId;
};
cElXMLTree * ToXMLTree(const cCible2Rech &);

void  BinaryDumpInFile(ELISE_fp &,const cCible2Rech &);

void  BinaryUnDumpFromFile(cCible2Rech &,ELISE_fp &);

std::string  Mangling( cCible2Rech *);

/******************************************************/
/******************************************************/
/******************************************************/
class cIm2Select
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cIm2Select & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        std::vector< std::string > & Id();
        const std::vector< std::string > & Id()const ;
    private:
        cTplValGesInit< bool > mUseIt;
        std::vector< std::string > mId;
};
cElXMLTree * ToXMLTree(const cIm2Select &);

void  BinaryDumpInFile(ELISE_fp &,const cIm2Select &);

void  BinaryUnDumpFromFile(cIm2Select &,ELISE_fp &);

std::string  Mangling( cIm2Select *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImageUseDirectPointeManuel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImageUseDirectPointeManuel & anObj,cElXMLTree * aTree);


        std::list< cElRegex_Ptr > & Id();
        const std::list< cElRegex_Ptr > & Id()const ;
    private:
        std::list< cElRegex_Ptr > mId;
};
cElXMLTree * ToXMLTree(const cImageUseDirectPointeManuel &);

void  BinaryDumpInFile(ELISE_fp &,const cImageUseDirectPointeManuel &);

void  BinaryUnDumpFromFile(cImageUseDirectPointeManuel &,ELISE_fp &);

std::string  Mangling( cImageUseDirectPointeManuel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cExportAppuisAsDico
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportAppuisAsDico & anObj,cElXMLTree * aTree);


        std::string & NameDico();
        const std::string & NameDico()const ;

        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;
    private:
        std::string mNameDico;
        Pt3dr mIncertitude;
};
cElXMLTree * ToXMLTree(const cExportAppuisAsDico &);

void  BinaryDumpInFile(ELISE_fp &,const cExportAppuisAsDico &);

void  BinaryUnDumpFromFile(cExportAppuisAsDico &,ELISE_fp &);

std::string  Mangling( cExportAppuisAsDico *);

/******************************************************/
/******************************************************/
/******************************************************/
class cComplParamEtalPoly
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cComplParamEtalPoly & anObj,cElXMLTree * aTree);


        std::list< cCibleACalcByLiaisons > & CibleACalcByLiaisons();
        const std::list< cCibleACalcByLiaisons > & CibleACalcByLiaisons()const ;

        cTplValGesInit< cCible2Rech > & Cible2Rech();
        const cTplValGesInit< cCible2Rech > & Cible2Rech()const ;

        cTplValGesInit< cIm2Select > & Im2Select();
        const cTplValGesInit< cIm2Select > & Im2Select()const ;

        cTplValGesInit< cImageUseDirectPointeManuel > & ImageUseDirectPointeManuel();
        const cTplValGesInit< cImageUseDirectPointeManuel > & ImageUseDirectPointeManuel()const ;

        std::string & NameDico();
        const std::string & NameDico()const ;

        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;

        cTplValGesInit< cExportAppuisAsDico > & ExportAppuisAsDico();
        const cTplValGesInit< cExportAppuisAsDico > & ExportAppuisAsDico()const ;
    private:
        std::list< cCibleACalcByLiaisons > mCibleACalcByLiaisons;
        cTplValGesInit< cCible2Rech > mCible2Rech;
        cTplValGesInit< cIm2Select > mIm2Select;
        cTplValGesInit< cImageUseDirectPointeManuel > mImageUseDirectPointeManuel;
        cTplValGesInit< cExportAppuisAsDico > mExportAppuisAsDico;
};
cElXMLTree * ToXMLTree(const cComplParamEtalPoly &);

void  BinaryDumpInFile(ELISE_fp &,const cComplParamEtalPoly &);

void  BinaryUnDumpFromFile(cComplParamEtalPoly &,ELISE_fp &);

std::string  Mangling( cComplParamEtalPoly *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneAppuisDAF
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneAppuisDAF & anObj,cElXMLTree * aTree);


        Pt3dr & Pt();
        const Pt3dr & Pt()const ;

        std::string & NamePt();
        const std::string & NamePt()const ;

        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;

        cTplValGesInit< bool > & UseForRTA();
        const cTplValGesInit< bool > & UseForRTA()const ;

        cTplValGesInit< Pt3dr > & Norm2Surf();
        const cTplValGesInit< Pt3dr > & Norm2Surf()const ;

        cTplValGesInit< double > & TetaN2SHor();
        const cTplValGesInit< double > & TetaN2SHor()const ;
    private:
        Pt3dr mPt;
        std::string mNamePt;
        Pt3dr mIncertitude;
        cTplValGesInit< bool > mUseForRTA;
        cTplValGesInit< Pt3dr > mNorm2Surf;
        cTplValGesInit< double > mTetaN2SHor;
};
cElXMLTree * ToXMLTree(const cOneAppuisDAF &);

void  BinaryDumpInFile(ELISE_fp &,const cOneAppuisDAF &);

void  BinaryUnDumpFromFile(cOneAppuisDAF &,ELISE_fp &);

std::string  Mangling( cOneAppuisDAF *);

/******************************************************/
/******************************************************/
/******************************************************/
class cDicoAppuisFlottant
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDicoAppuisFlottant & anObj,cElXMLTree * aTree);


        std::list< cOneAppuisDAF > & OneAppuisDAF();
        const std::list< cOneAppuisDAF > & OneAppuisDAF()const ;
    private:
        std::list< cOneAppuisDAF > mOneAppuisDAF;
};
cElXMLTree * ToXMLTree(const cDicoAppuisFlottant &);

void  BinaryDumpInFile(ELISE_fp &,const cDicoAppuisFlottant &);

void  BinaryUnDumpFromFile(cDicoAppuisFlottant &,ELISE_fp &);

std::string  Mangling( cDicoAppuisFlottant *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCpleImgTime
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCpleImgTime & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        double & TimeIm();
        const double & TimeIm()const ;
    private:
        std::string mNameIm;
        double mTimeIm;
};
cElXMLTree * ToXMLTree(const cCpleImgTime &);

void  BinaryDumpInFile(ELISE_fp &,const cCpleImgTime &);

void  BinaryUnDumpFromFile(cCpleImgTime &,ELISE_fp &);

std::string  Mangling( cCpleImgTime *);

/******************************************************/
/******************************************************/
/******************************************************/
class cDicoImgsTime
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDicoImgsTime & anObj,cElXMLTree * aTree);


        std::vector< cCpleImgTime > & CpleImgTime();
        const std::vector< cCpleImgTime > & CpleImgTime()const ;
    private:
        std::vector< cCpleImgTime > mCpleImgTime;
};
cElXMLTree * ToXMLTree(const cDicoImgsTime &);

void  BinaryDumpInFile(ELISE_fp &,const cDicoImgsTime &);

void  BinaryUnDumpFromFile(cDicoImgsTime &,ELISE_fp &);

std::string  Mangling( cDicoImgsTime *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneGpsDGF
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneGpsDGF & anObj,cElXMLTree * aTree);


        Pt3dr & Pt();
        const Pt3dr & Pt()const ;

        std::string & NamePt();
        const std::string & NamePt()const ;

        int & TagPt();
        const int & TagPt()const ;

        double & TimePt();
        const double & TimePt()const ;

        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;
    private:
        Pt3dr mPt;
        std::string mNamePt;
        int mTagPt;
        double mTimePt;
        Pt3dr mIncertitude;
};
cElXMLTree * ToXMLTree(const cOneGpsDGF &);

void  BinaryDumpInFile(ELISE_fp &,const cOneGpsDGF &);

void  BinaryUnDumpFromFile(cOneGpsDGF &,ELISE_fp &);

std::string  Mangling( cOneGpsDGF *);

/******************************************************/
/******************************************************/
/******************************************************/
class cDicoGpsFlottant
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDicoGpsFlottant & anObj,cElXMLTree * aTree);


        std::vector< cOneGpsDGF > & OneGpsDGF();
        const std::vector< cOneGpsDGF > & OneGpsDGF()const ;
    private:
        std::vector< cOneGpsDGF > mOneGpsDGF;
};
cElXMLTree * ToXMLTree(const cDicoGpsFlottant &);

void  BinaryDumpInFile(ELISE_fp &,const cDicoGpsFlottant &);

void  BinaryUnDumpFromFile(cDicoGpsFlottant &,ELISE_fp &);

std::string  Mangling( cDicoGpsFlottant *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneModifIPF
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneModifIPF & anObj,cElXMLTree * aTree);


        std::string & KeyName();
        const std::string & KeyName()const ;

        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;

        cTplValGesInit< bool > & IsMult();
        const cTplValGesInit< bool > & IsMult()const ;
    private:
        std::string mKeyName;
        Pt3dr mIncertitude;
        cTplValGesInit< bool > mIsMult;
};
cElXMLTree * ToXMLTree(const cOneModifIPF &);

void  BinaryDumpInFile(ELISE_fp &,const cOneModifIPF &);

void  BinaryUnDumpFromFile(cOneModifIPF &,ELISE_fp &);

std::string  Mangling( cOneModifIPF *);

/******************************************************/
/******************************************************/
/******************************************************/
class cModifIncPtsFlottant
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModifIncPtsFlottant & anObj,cElXMLTree * aTree);


        std::list< cOneModifIPF > & OneModifIPF();
        const std::list< cOneModifIPF > & OneModifIPF()const ;
    private:
        std::list< cOneModifIPF > mOneModifIPF;
};
cElXMLTree * ToXMLTree(const cModifIncPtsFlottant &);

void  BinaryDumpInFile(ELISE_fp &,const cModifIncPtsFlottant &);

void  BinaryUnDumpFromFile(cModifIncPtsFlottant &,ELISE_fp &);

std::string  Mangling( cModifIncPtsFlottant *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneMesureAF1I
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneMesureAF1I & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & PrecPointe();
        const cTplValGesInit< double > & PrecPointe()const ;

        std::string & NamePt();
        const std::string & NamePt()const ;

        Pt2dr & PtIm();
        const Pt2dr & PtIm()const ;
    private:
        cTplValGesInit< double > mPrecPointe;
        std::string mNamePt;
        Pt2dr mPtIm;
};
cElXMLTree * ToXMLTree(const cOneMesureAF1I &);

void  BinaryDumpInFile(ELISE_fp &,const cOneMesureAF1I &);

void  BinaryUnDumpFromFile(cOneMesureAF1I &,ELISE_fp &);

std::string  Mangling( cOneMesureAF1I *);

class cMesureAppuiFlottant1Im
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMesureAppuiFlottant1Im & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        cTplValGesInit< double > & PrecPointeByIm();
        const cTplValGesInit< double > & PrecPointeByIm()const ;

        std::list< cOneMesureAF1I > & OneMesureAF1I();
        const std::list< cOneMesureAF1I > & OneMesureAF1I()const ;
    private:
        std::string mNameIm;
        cTplValGesInit< double > mPrecPointeByIm;
        std::list< cOneMesureAF1I > mOneMesureAF1I;
};
cElXMLTree * ToXMLTree(const cMesureAppuiFlottant1Im &);

void  BinaryDumpInFile(ELISE_fp &,const cMesureAppuiFlottant1Im &);

void  BinaryUnDumpFromFile(cMesureAppuiFlottant1Im &,ELISE_fp &);

std::string  Mangling( cMesureAppuiFlottant1Im *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetOfMesureAppuisFlottants
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetOfMesureAppuisFlottants & anObj,cElXMLTree * aTree);


        std::list< cMesureAppuiFlottant1Im > & MesureAppuiFlottant1Im();
        const std::list< cMesureAppuiFlottant1Im > & MesureAppuiFlottant1Im()const ;
    private:
        std::list< cMesureAppuiFlottant1Im > mMesureAppuiFlottant1Im;
};
cElXMLTree * ToXMLTree(const cSetOfMesureAppuisFlottants &);

void  BinaryDumpInFile(ELISE_fp &,const cSetOfMesureAppuisFlottants &);

void  BinaryUnDumpFromFile(cSetOfMesureAppuisFlottants &,ELISE_fp &);

std::string  Mangling( cSetOfMesureAppuisFlottants *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneMesureSegDr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneMesureSegDr & anObj,cElXMLTree * aTree);


        std::list< std::string > & NamePt();
        const std::list< std::string > & NamePt()const ;

        Pt2dr & Pt1Im();
        const Pt2dr & Pt1Im()const ;

        Pt2dr & Pt2Im();
        const Pt2dr & Pt2Im()const ;
    private:
        std::list< std::string > mNamePt;
        Pt2dr mPt1Im;
        Pt2dr mPt2Im;
};
cElXMLTree * ToXMLTree(const cOneMesureSegDr &);

void  BinaryDumpInFile(ELISE_fp &,const cOneMesureSegDr &);

void  BinaryUnDumpFromFile(cOneMesureSegDr &,ELISE_fp &);

std::string  Mangling( cOneMesureSegDr *);

class cMesureAppuiSegDr1Im
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMesureAppuiSegDr1Im & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        std::list< cOneMesureSegDr > & OneMesureSegDr();
        const std::list< cOneMesureSegDr > & OneMesureSegDr()const ;
    private:
        std::string mNameIm;
        std::list< cOneMesureSegDr > mOneMesureSegDr;
};
cElXMLTree * ToXMLTree(const cMesureAppuiSegDr1Im &);

void  BinaryDumpInFile(ELISE_fp &,const cMesureAppuiSegDr1Im &);

void  BinaryUnDumpFromFile(cMesureAppuiSegDr1Im &,ELISE_fp &);

std::string  Mangling( cMesureAppuiSegDr1Im *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetOfMesureSegDr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetOfMesureSegDr & anObj,cElXMLTree * aTree);


        std::list< cMesureAppuiSegDr1Im > & MesureAppuiSegDr1Im();
        const std::list< cMesureAppuiSegDr1Im > & MesureAppuiSegDr1Im()const ;
    private:
        std::list< cMesureAppuiSegDr1Im > mMesureAppuiSegDr1Im;
};
cElXMLTree * ToXMLTree(const cSetOfMesureSegDr &);

void  BinaryDumpInFile(ELISE_fp &,const cSetOfMesureSegDr &);

void  BinaryUnDumpFromFile(cSetOfMesureSegDr &,ELISE_fp &);

std::string  Mangling( cSetOfMesureSegDr *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMesureAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMesureAppuis & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & Num();
        const cTplValGesInit< int > & Num()const ;

        Pt2dr & Im();
        const Pt2dr & Im()const ;

        Pt3dr & Ter();
        const Pt3dr & Ter()const ;
    private:
        cTplValGesInit< int > mNum;
        Pt2dr mIm;
        Pt3dr mTer;
};
cElXMLTree * ToXMLTree(const cMesureAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cMesureAppuis &);

void  BinaryUnDumpFromFile(cMesureAppuis &,ELISE_fp &);

std::string  Mangling( cMesureAppuis *);

/******************************************************/
/******************************************************/
/******************************************************/
class cListeAppuis1Im
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cListeAppuis1Im & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NameImage();
        const cTplValGesInit< std::string > & NameImage()const ;

        std::list< cMesureAppuis > & Mesures();
        const std::list< cMesureAppuis > & Mesures()const ;
    private:
        cTplValGesInit< std::string > mNameImage;
        std::list< cMesureAppuis > mMesures;
};
cElXMLTree * ToXMLTree(const cListeAppuis1Im &);

void  BinaryDumpInFile(ELISE_fp &,const cListeAppuis1Im &);

void  BinaryUnDumpFromFile(cListeAppuis1Im &,ELISE_fp &);

std::string  Mangling( cListeAppuis1Im *);

/******************************************************/
/******************************************************/
/******************************************************/
class cVerifOrient
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVerifOrient & anObj,cElXMLTree * aTree);


        double & Tol();
        const double & Tol()const ;

        cTplValGesInit< bool > & ShowMes();
        const cTplValGesInit< bool > & ShowMes()const ;

        std::list< cMesureAppuis > & Appuis();
        const std::list< cMesureAppuis > & Appuis()const ;

        cTplValGesInit< bool > & IsTest();
        const cTplValGesInit< bool > & IsTest()const ;

        cTplValGesInit< cListeAppuis1Im > & AppuisConv();
        const cTplValGesInit< cListeAppuis1Im > & AppuisConv()const ;
    private:
        double mTol;
        cTplValGesInit< bool > mShowMes;
        std::list< cMesureAppuis > mAppuis;
        cTplValGesInit< bool > mIsTest;
        cTplValGesInit< cListeAppuis1Im > mAppuisConv;
};
cElXMLTree * ToXMLTree(const cVerifOrient &);

void  BinaryDumpInFile(ELISE_fp &,const cVerifOrient &);

void  BinaryUnDumpFromFile(cVerifOrient &,ELISE_fp &);

std::string  Mangling( cVerifOrient *);

/******************************************************/
/******************************************************/
/******************************************************/
typedef enum
{
  eConvInconnue,
  eConvApero_DistC2M,
  eConvApero_DistM2C,
  eConvOriLib,
  eConvMatrPoivillier_E,
  eConvAngErdas,
  eConvAngErdas_Grade,
  eConvAngAvionJaune,
  eConvAngSurvey,
  eConvAngPhotoMDegre,
  eConvAngPhotoMGrade,
  eConvAngLPSDegre,
  eConvMatrixInpho
} eConventionsOrientation;
void xml_init(eConventionsOrientation & aVal,cElXMLTree * aTree);
std::string  eToString(const eConventionsOrientation & aVal);

eConventionsOrientation  Str2eConventionsOrientation(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eConventionsOrientation & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eConventionsOrientation &);

std::string  Mangling( eConventionsOrientation *);

void  BinaryUnDumpFromFile(eConventionsOrientation &,ELISE_fp &);

typedef enum
{
  eEO_MMM,
  eEO_AMM,
  eEO_WPK,
  eEO_NbVals
} eExportOri;
void xml_init(eExportOri & aVal,cElXMLTree * aTree);
std::string  eToString(const eExportOri & aVal);

eExportOri  Str2eExportOri(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eExportOri & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eExportOri &);

std::string  Mangling( eExportOri *);

void  BinaryUnDumpFromFile(eExportOri &,ELISE_fp &);

class cJPPTest
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cJPPTest & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        std::list< int > & LN();
        const std::list< int > & LN()const ;
    private:
        std::string mName;
        std::list< int > mLN;
};
cElXMLTree * ToXMLTree(const cJPPTest &);

void  BinaryDumpInFile(ELISE_fp &,const cJPPTest &);

void  BinaryUnDumpFromFile(cJPPTest &,ELISE_fp &);

std::string  Mangling( cJPPTest *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibrationInterneGridDef
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationInterneGridDef & anObj,cElXMLTree * aTree);


        Pt2dr & P0();
        const Pt2dr & P0()const ;

        Pt2dr & P1();
        const Pt2dr & P1()const ;

        Pt2di & Nb();
        const Pt2di & Nb()const ;

        std::vector< Pt2dr > & PGr();
        const std::vector< Pt2dr > & PGr()const ;
    private:
        Pt2dr mP0;
        Pt2dr mP1;
        Pt2di mNb;
        std::vector< Pt2dr > mPGr;
};
cElXMLTree * ToXMLTree(const cCalibrationInterneGridDef &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationInterneGridDef &);

void  BinaryUnDumpFromFile(cCalibrationInterneGridDef &,ELISE_fp &);

std::string  Mangling( cCalibrationInterneGridDef *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibrationInterneRadiale
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationInterneRadiale & anObj,cElXMLTree * aTree);


        Pt2dr & CDist();
        const Pt2dr & CDist()const ;

        std::vector< double > & CoeffDist();
        const std::vector< double > & CoeffDist()const ;

        std::vector< double > & CoeffDistInv();
        const std::vector< double > & CoeffDistInv()const ;

        cTplValGesInit< double > & RatioDistInv();
        const cTplValGesInit< double > & RatioDistInv()const ;

        cTplValGesInit< bool > & PPaEqPPs();
        const cTplValGesInit< bool > & PPaEqPPs()const ;
    private:
        Pt2dr mCDist;
        std::vector< double > mCoeffDist;
        std::vector< double > mCoeffDistInv;
        cTplValGesInit< double > mRatioDistInv;
        cTplValGesInit< bool > mPPaEqPPs;
};
cElXMLTree * ToXMLTree(const cCalibrationInterneRadiale &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationInterneRadiale &);

void  BinaryUnDumpFromFile(cCalibrationInterneRadiale &,ELISE_fp &);

std::string  Mangling( cCalibrationInterneRadiale *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibrationInternePghrStd
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationInternePghrStd & anObj,cElXMLTree * aTree);


        cCalibrationInterneRadiale & RadialePart();
        const cCalibrationInterneRadiale & RadialePart()const ;

        cTplValGesInit< double > & P1();
        const cTplValGesInit< double > & P1()const ;

        cTplValGesInit< double > & P2();
        const cTplValGesInit< double > & P2()const ;

        cTplValGesInit< double > & b1();
        const cTplValGesInit< double > & b1()const ;

        cTplValGesInit< double > & b2();
        const cTplValGesInit< double > & b2()const ;
    private:
        cCalibrationInterneRadiale mRadialePart;
        cTplValGesInit< double > mP1;
        cTplValGesInit< double > mP2;
        cTplValGesInit< double > mb1;
        cTplValGesInit< double > mb2;
};
cElXMLTree * ToXMLTree(const cCalibrationInternePghrStd &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationInternePghrStd &);

void  BinaryUnDumpFromFile(cCalibrationInternePghrStd &,ELISE_fp &);

std::string  Mangling( cCalibrationInternePghrStd *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibrationInterneUnif
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationInterneUnif & anObj,cElXMLTree * aTree);


        eModelesCalibUnif & TypeModele();
        const eModelesCalibUnif & TypeModele()const ;

        std::vector< double > & Params();
        const std::vector< double > & Params()const ;

        std::vector< double > & Etats();
        const std::vector< double > & Etats()const ;
    private:
        eModelesCalibUnif mTypeModele;
        std::vector< double > mParams;
        std::vector< double > mEtats;
};
cElXMLTree * ToXMLTree(const cCalibrationInterneUnif &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationInterneUnif &);

void  BinaryUnDumpFromFile(cCalibrationInterneUnif &,ELISE_fp &);

std::string  Mangling( cCalibrationInterneUnif *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTestNewGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTestNewGrid & anObj,cElXMLTree * aTree);


        std::string & A();
        const std::string & A()const ;

        Im2D_INT1 & Im();
        const Im2D_INT1 & Im()const ;

        std::string & Z();
        const std::string & Z()const ;
    private:
        std::string mA;
        Im2D_INT1 mIm;
        std::string mZ;
};
cElXMLTree * ToXMLTree(const cTestNewGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cTestNewGrid &);

void  BinaryUnDumpFromFile(cTestNewGrid &,ELISE_fp &);

std::string  Mangling( cTestNewGrid *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGridDeform2D
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGridDeform2D & anObj,cElXMLTree * aTree);


        Pt2dr & Origine();
        const Pt2dr & Origine()const ;

        Pt2dr & Step();
        const Pt2dr & Step()const ;

        Im2D_REAL8 & ImX();
        const Im2D_REAL8 & ImX()const ;

        Im2D_REAL8 & ImY();
        const Im2D_REAL8 & ImY()const ;
    private:
        Pt2dr mOrigine;
        Pt2dr mStep;
        Im2D_REAL8 mImX;
        Im2D_REAL8 mImY;
};
cElXMLTree * ToXMLTree(const cGridDeform2D &);

void  BinaryDumpInFile(ELISE_fp &,const cGridDeform2D &);

void  BinaryUnDumpFromFile(cGridDeform2D &,ELISE_fp &);

std::string  Mangling( cGridDeform2D *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGridDirecteEtInverse
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGridDirecteEtInverse & anObj,cElXMLTree * aTree);


        cGridDeform2D & Directe();
        const cGridDeform2D & Directe()const ;

        cGridDeform2D & Inverse();
        const cGridDeform2D & Inverse()const ;

        bool & AdaptStep();
        const bool & AdaptStep()const ;
    private:
        cGridDeform2D mDirecte;
        cGridDeform2D mInverse;
        bool mAdaptStep;
};
cElXMLTree * ToXMLTree(const cGridDirecteEtInverse &);

void  BinaryDumpInFile(ELISE_fp &,const cGridDirecteEtInverse &);

void  BinaryUnDumpFromFile(cGridDirecteEtInverse &,ELISE_fp &);

std::string  Mangling( cGridDirecteEtInverse *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPreCondRadial
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPreCondRadial & anObj,cElXMLTree * aTree);


        Pt2dr & C();
        const Pt2dr & C()const ;

        double & F();
        const double & F()const ;

        eTypePreCondRad & Mode();
        const eTypePreCondRad & Mode()const ;
    private:
        Pt2dr mC;
        double mF;
        eTypePreCondRad mMode;
};
cElXMLTree * ToXMLTree(const cPreCondRadial &);

void  BinaryDumpInFile(ELISE_fp &,const cPreCondRadial &);

void  BinaryUnDumpFromFile(cPreCondRadial &,ELISE_fp &);

std::string  Mangling( cPreCondRadial *);

class cPreCondGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPreCondGrid & anObj,cElXMLTree * aTree);


        Pt2dr & C();
        const Pt2dr & C()const ;

        double & F();
        const double & F()const ;

        eTypePreCondRad & Mode();
        const eTypePreCondRad & Mode()const ;

        cTplValGesInit< cPreCondRadial > & PreCondRadial();
        const cTplValGesInit< cPreCondRadial > & PreCondRadial()const ;
    private:
        cTplValGesInit< cPreCondRadial > mPreCondRadial;
};
cElXMLTree * ToXMLTree(const cPreCondGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cPreCondGrid &);

void  BinaryUnDumpFromFile(cPreCondGrid &,ELISE_fp &);

std::string  Mangling( cPreCondGrid *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibrationInterneGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationInterneGrid & anObj,cElXMLTree * aTree);


        Pt2dr & C();
        const Pt2dr & C()const ;

        double & F();
        const double & F()const ;

        eTypePreCondRad & Mode();
        const eTypePreCondRad & Mode()const ;

        cTplValGesInit< cPreCondRadial > & PreCondRadial();
        const cTplValGesInit< cPreCondRadial > & PreCondRadial()const ;

        cTplValGesInit< cPreCondGrid > & PreCondGrid();
        const cTplValGesInit< cPreCondGrid > & PreCondGrid()const ;

        cGridDirecteEtInverse & Grid();
        const cGridDirecteEtInverse & Grid()const ;
    private:
        cTplValGesInit< cPreCondGrid > mPreCondGrid;
        cGridDirecteEtInverse mGrid;
};
cElXMLTree * ToXMLTree(const cCalibrationInterneGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationInterneGrid &);

void  BinaryUnDumpFromFile(cCalibrationInterneGrid &,ELISE_fp &);

std::string  Mangling( cCalibrationInterneGrid *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSimilitudePlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSimilitudePlane & anObj,cElXMLTree * aTree);


        Pt2dr & Scale();
        const Pt2dr & Scale()const ;

        Pt2dr & Trans();
        const Pt2dr & Trans()const ;
    private:
        Pt2dr mScale;
        Pt2dr mTrans;
};
cElXMLTree * ToXMLTree(const cSimilitudePlane &);

void  BinaryDumpInFile(ELISE_fp &,const cSimilitudePlane &);

void  BinaryUnDumpFromFile(cSimilitudePlane &,ELISE_fp &);

std::string  Mangling( cSimilitudePlane *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAffinitePlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAffinitePlane & anObj,cElXMLTree * aTree);


        Pt2dr & I00();
        const Pt2dr & I00()const ;

        Pt2dr & V10();
        const Pt2dr & V10()const ;

        Pt2dr & V01();
        const Pt2dr & V01()const ;
    private:
        Pt2dr mI00;
        Pt2dr mV10;
        Pt2dr mV01;
};
cElXMLTree * ToXMLTree(const cAffinitePlane &);

void  BinaryDumpInFile(ELISE_fp &,const cAffinitePlane &);

void  BinaryUnDumpFromFile(cAffinitePlane &,ELISE_fp &);

std::string  Mangling( cAffinitePlane *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOrIntGlob
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOrIntGlob & anObj,cElXMLTree * aTree);


        cAffinitePlane & Affinite();
        const cAffinitePlane & Affinite()const ;

        bool & C2M();
        const bool & C2M()const ;
    private:
        cAffinitePlane mAffinite;
        bool mC2M;
};
cElXMLTree * ToXMLTree(const cOrIntGlob &);

void  BinaryDumpInFile(ELISE_fp &,const cOrIntGlob &);

void  BinaryUnDumpFromFile(cOrIntGlob &,ELISE_fp &);

std::string  Mangling( cOrIntGlob *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamForGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamForGrid & anObj,cElXMLTree * aTree);


        Pt2dr & StepGrid();
        const Pt2dr & StepGrid()const ;

        double & RayonInv();
        const double & RayonInv()const ;
    private:
        Pt2dr mStepGrid;
        double mRayonInv;
};
cElXMLTree * ToXMLTree(const cParamForGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cParamForGrid &);

void  BinaryUnDumpFromFile(cParamForGrid &,ELISE_fp &);

std::string  Mangling( cParamForGrid *);

/******************************************************/
/******************************************************/
/******************************************************/
class cModNoDist
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModNoDist & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Inutile();
        const cTplValGesInit< std::string > & Inutile()const ;
    private:
        cTplValGesInit< std::string > mInutile;
};
cElXMLTree * ToXMLTree(const cModNoDist &);

void  BinaryDumpInFile(ELISE_fp &,const cModNoDist &);

void  BinaryUnDumpFromFile(cModNoDist &,ELISE_fp &);

std::string  Mangling( cModNoDist *);

class cCalibDistortion
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibDistortion & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Inutile();
        const cTplValGesInit< std::string > & Inutile()const ;

        cTplValGesInit< cModNoDist > & ModNoDist();
        const cTplValGesInit< cModNoDist > & ModNoDist()const ;

        cTplValGesInit< cCalibrationInterneRadiale > & ModRad();
        const cTplValGesInit< cCalibrationInterneRadiale > & ModRad()const ;

        cTplValGesInit< cCalibrationInternePghrStd > & ModPhgrStd();
        const cTplValGesInit< cCalibrationInternePghrStd > & ModPhgrStd()const ;

        cTplValGesInit< cCalibrationInterneUnif > & ModUnif();
        const cTplValGesInit< cCalibrationInterneUnif > & ModUnif()const ;

        cTplValGesInit< cCalibrationInterneGrid > & ModGrid();
        const cTplValGesInit< cCalibrationInterneGrid > & ModGrid()const ;

        cTplValGesInit< cCalibrationInterneGridDef > & ModGridDef();
        const cTplValGesInit< cCalibrationInterneGridDef > & ModGridDef()const ;
    private:
        cTplValGesInit< cModNoDist > mModNoDist;
        cTplValGesInit< cCalibrationInterneRadiale > mModRad;
        cTplValGesInit< cCalibrationInternePghrStd > mModPhgrStd;
        cTplValGesInit< cCalibrationInterneUnif > mModUnif;
        cTplValGesInit< cCalibrationInterneGrid > mModGrid;
        cTplValGesInit< cCalibrationInterneGridDef > mModGridDef;
};
cElXMLTree * ToXMLTree(const cCalibDistortion &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibDistortion &);

void  BinaryUnDumpFromFile(cCalibDistortion &,ELISE_fp &);

std::string  Mangling( cCalibDistortion *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCorrectionRefractionAPosteriori
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCorrectionRefractionAPosteriori & anObj,cElXMLTree * aTree);


        std::string & FileEstimCam();
        const std::string & FileEstimCam()const ;

        cTplValGesInit< std::string > & NameTag();
        const cTplValGesInit< std::string > & NameTag()const ;

        double & CoeffRefrac();
        const double & CoeffRefrac()const ;

        cTplValGesInit< bool > & IntegreDist();
        const cTplValGesInit< bool > & IntegreDist()const ;
    private:
        std::string mFileEstimCam;
        cTplValGesInit< std::string > mNameTag;
        double mCoeffRefrac;
        cTplValGesInit< bool > mIntegreDist;
};
cElXMLTree * ToXMLTree(const cCorrectionRefractionAPosteriori &);

void  BinaryDumpInFile(ELISE_fp &,const cCorrectionRefractionAPosteriori &);

void  BinaryUnDumpFromFile(cCorrectionRefractionAPosteriori &,ELISE_fp &);

std::string  Mangling( cCorrectionRefractionAPosteriori *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibrationInternConique
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationInternConique & anObj,cElXMLTree * aTree);


        cTplValGesInit< eConventionsOrientation > & KnownConv();
        const cTplValGesInit< eConventionsOrientation > & KnownConv()const ;

        std::vector< double > & ParamAF();
        const std::vector< double > & ParamAF()const ;

        Pt2dr & PP();
        const Pt2dr & PP()const ;

        double & F();
        const double & F()const ;

        Pt2di & SzIm();
        const Pt2di & SzIm()const ;

        cTplValGesInit< Pt2dr > & PixelSzIm();
        const cTplValGesInit< Pt2dr > & PixelSzIm()const ;

        cTplValGesInit< double > & RayonUtile();
        const cTplValGesInit< double > & RayonUtile()const ;

        std::vector< bool > & ComplIsC2M();
        const std::vector< bool > & ComplIsC2M()const ;

        cTplValGesInit< bool > & ScannedAnalogik();
        const cTplValGesInit< bool > & ScannedAnalogik()const ;

        cAffinitePlane & Affinite();
        const cAffinitePlane & Affinite()const ;

        bool & C2M();
        const bool & C2M()const ;

        cTplValGesInit< cOrIntGlob > & OrIntGlob();
        const cTplValGesInit< cOrIntGlob > & OrIntGlob()const ;

        Pt2dr & StepGrid();
        const Pt2dr & StepGrid()const ;

        double & RayonInv();
        const double & RayonInv()const ;

        cTplValGesInit< cParamForGrid > & ParamForGrid();
        const cTplValGesInit< cParamForGrid > & ParamForGrid()const ;

        std::vector< cCalibDistortion > & CalibDistortion();
        const std::vector< cCalibDistortion > & CalibDistortion()const ;

        std::string & FileEstimCam();
        const std::string & FileEstimCam()const ;

        cTplValGesInit< std::string > & NameTag();
        const cTplValGesInit< std::string > & NameTag()const ;

        double & CoeffRefrac();
        const double & CoeffRefrac()const ;

        cTplValGesInit< bool > & IntegreDist();
        const cTplValGesInit< bool > & IntegreDist()const ;

        cTplValGesInit< cCorrectionRefractionAPosteriori > & CorrectionRefractionAPosteriori();
        const cTplValGesInit< cCorrectionRefractionAPosteriori > & CorrectionRefractionAPosteriori()const ;
    private:
        cTplValGesInit< eConventionsOrientation > mKnownConv;
        std::vector< double > mParamAF;
        Pt2dr mPP;
        double mF;
        Pt2di mSzIm;
        cTplValGesInit< Pt2dr > mPixelSzIm;
        cTplValGesInit< double > mRayonUtile;
        std::vector< bool > mComplIsC2M;
        cTplValGesInit< bool > mScannedAnalogik;
        cTplValGesInit< cOrIntGlob > mOrIntGlob;
        cTplValGesInit< cParamForGrid > mParamForGrid;
        std::vector< cCalibDistortion > mCalibDistortion;
        cTplValGesInit< cCorrectionRefractionAPosteriori > mCorrectionRefractionAPosteriori;
};
cElXMLTree * ToXMLTree(const cCalibrationInternConique &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationInternConique &);

void  BinaryUnDumpFromFile(cCalibrationInternConique &,ELISE_fp &);

std::string  Mangling( cCalibrationInternConique *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRepereCartesien
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRepereCartesien & anObj,cElXMLTree * aTree);


        Pt3dr & Ori();
        const Pt3dr & Ori()const ;

        Pt3dr & Ox();
        const Pt3dr & Ox()const ;

        Pt3dr & Oy();
        const Pt3dr & Oy()const ;

        Pt3dr & Oz();
        const Pt3dr & Oz()const ;
    private:
        Pt3dr mOri;
        Pt3dr mOx;
        Pt3dr mOy;
        Pt3dr mOz;
};
cElXMLTree * ToXMLTree(const cRepereCartesien &);

void  BinaryDumpInFile(ELISE_fp &,const cRepereCartesien &);

void  BinaryUnDumpFromFile(cRepereCartesien &,ELISE_fp &);

std::string  Mangling( cRepereCartesien *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTypeCodageMatr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTypeCodageMatr & anObj,cElXMLTree * aTree);


        Pt3dr & L1();
        const Pt3dr & L1()const ;

        Pt3dr & L2();
        const Pt3dr & L2()const ;

        Pt3dr & L3();
        const Pt3dr & L3()const ;

        cTplValGesInit< bool > & TrueRot();
        const cTplValGesInit< bool > & TrueRot()const ;
    private:
        Pt3dr mL1;
        Pt3dr mL2;
        Pt3dr mL3;
        cTplValGesInit< bool > mTrueRot;
};
cElXMLTree * ToXMLTree(const cTypeCodageMatr &);

void  BinaryDumpInFile(ELISE_fp &,const cTypeCodageMatr &);

void  BinaryUnDumpFromFile(cTypeCodageMatr &,ELISE_fp &);

std::string  Mangling( cTypeCodageMatr *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRotationVect
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRotationVect & anObj,cElXMLTree * aTree);


        cTplValGesInit< cTypeCodageMatr > & CodageMatr();
        const cTplValGesInit< cTypeCodageMatr > & CodageMatr()const ;

        cTplValGesInit< Pt3dr > & CodageAngulaire();
        const cTplValGesInit< Pt3dr > & CodageAngulaire()const ;

        cTplValGesInit< std::string > & CodageSymbolique();
        const cTplValGesInit< std::string > & CodageSymbolique()const ;
    private:
        cTplValGesInit< cTypeCodageMatr > mCodageMatr;
        cTplValGesInit< Pt3dr > mCodageAngulaire;
        cTplValGesInit< std::string > mCodageSymbolique;
};
cElXMLTree * ToXMLTree(const cRotationVect &);

void  BinaryDumpInFile(ELISE_fp &,const cRotationVect &);

void  BinaryUnDumpFromFile(cRotationVect &,ELISE_fp &);

std::string  Mangling( cRotationVect *);

/******************************************************/
/******************************************************/
/******************************************************/
class cDatePDV
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDatePDV & anObj,cElXMLTree * aTree);


        int & Annee();
        const int & Annee()const ;

        int & Mois();
        const int & Mois()const ;

        int & Jour();
        const int & Jour()const ;

        int & Heure();
        const int & Heure()const ;

        int & Minute();
        const int & Minute()const ;

        double & Seconde();
        const double & Seconde()const ;
    private:
        int mAnnee;
        int mMois;
        int mJour;
        int mHeure;
        int mMinute;
        double mSeconde;
};
cElXMLTree * ToXMLTree(const cDatePDV &);

void  BinaryDumpInFile(ELISE_fp &,const cDatePDV &);

void  BinaryUnDumpFromFile(cDatePDV &,ELISE_fp &);

std::string  Mangling( cDatePDV *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlHour
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlHour & anObj,cElXMLTree * aTree);


        int & H();
        const int & H()const ;

        int & M();
        const int & M()const ;

        double & S();
        const double & S()const ;
    private:
        int mH;
        int mM;
        double mS;
};
cElXMLTree * ToXMLTree(const cXmlHour &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlHour &);

void  BinaryUnDumpFromFile(cXmlHour &,ELISE_fp &);

std::string  Mangling( cXmlHour *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlDate
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlDate & anObj,cElXMLTree * aTree);


        int & Y();
        const int & Y()const ;

        int & M();
        const int & M()const ;

        int & D();
        const int & D()const ;

        cXmlHour & Hour();
        const cXmlHour & Hour()const ;
    private:
        int mY;
        int mM;
        int mD;
        cXmlHour mHour;
};
cElXMLTree * ToXMLTree(const cXmlDate &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlDate &);

void  BinaryUnDumpFromFile(cXmlDate &,ELISE_fp &);

std::string  Mangling( cXmlDate *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOrientationExterneRigide
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOrientationExterneRigide & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & AltiSol();
        const cTplValGesInit< double > & AltiSol()const ;

        cTplValGesInit< double > & Profondeur();
        const cTplValGesInit< double > & Profondeur()const ;

        cTplValGesInit< double > & Time();
        const cTplValGesInit< double > & Time()const ;

        cTplValGesInit< cXmlDate > & Date();
        const cTplValGesInit< cXmlDate > & Date()const ;

        cTplValGesInit< eConventionsOrientation > & KnownConv();
        const cTplValGesInit< eConventionsOrientation > & KnownConv()const ;

        Pt3dr & Centre();
        const Pt3dr & Centre()const ;

        cTplValGesInit< Pt3dr > & OffsetCentre();
        const cTplValGesInit< Pt3dr > & OffsetCentre()const ;

        cTplValGesInit< Pt3dr > & Vitesse();
        const cTplValGesInit< Pt3dr > & Vitesse()const ;

        cTplValGesInit< bool > & VitesseFiable();
        const cTplValGesInit< bool > & VitesseFiable()const ;

        cTplValGesInit< Pt3dr > & IncCentre();
        const cTplValGesInit< Pt3dr > & IncCentre()const ;

        cRotationVect & ParamRotation();
        const cRotationVect & ParamRotation()const ;
    private:
        cTplValGesInit< double > mAltiSol;
        cTplValGesInit< double > mProfondeur;
        cTplValGesInit< double > mTime;
        cTplValGesInit< cXmlDate > mDate;
        cTplValGesInit< eConventionsOrientation > mKnownConv;
        Pt3dr mCentre;
        cTplValGesInit< Pt3dr > mOffsetCentre;
        cTplValGesInit< Pt3dr > mVitesse;
        cTplValGesInit< bool > mVitesseFiable;
        cTplValGesInit< Pt3dr > mIncCentre;
        cRotationVect mParamRotation;
};
cElXMLTree * ToXMLTree(const cOrientationExterneRigide &);

void  BinaryDumpInFile(ELISE_fp &,const cOrientationExterneRigide &);

void  BinaryUnDumpFromFile(cOrientationExterneRigide &,ELISE_fp &);

std::string  Mangling( cOrientationExterneRigide *);

/******************************************************/
/******************************************************/
/******************************************************/
class cModuleOrientationFile
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModuleOrientationFile & anObj,cElXMLTree * aTree);


        std::string & NameFileOri();
        const std::string & NameFileOri()const ;
    private:
        std::string mNameFileOri;
};
cElXMLTree * ToXMLTree(const cModuleOrientationFile &);

void  BinaryDumpInFile(ELISE_fp &,const cModuleOrientationFile &);

void  BinaryUnDumpFromFile(cModuleOrientationFile &,ELISE_fp &);

std::string  Mangling( cModuleOrientationFile *);

/******************************************************/
/******************************************************/
/******************************************************/
class cConvExplicite
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cConvExplicite & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & SensYVideo();
        const cTplValGesInit< bool > & SensYVideo()const ;

        cTplValGesInit< bool > & DistSenC2M();
        const cTplValGesInit< bool > & DistSenC2M()const ;

        cTplValGesInit< bool > & MatrSenC2M();
        const cTplValGesInit< bool > & MatrSenC2M()const ;

        cTplValGesInit< Pt3dr > & ColMul();
        const cTplValGesInit< Pt3dr > & ColMul()const ;

        cTplValGesInit< Pt3dr > & LigMul();
        const cTplValGesInit< Pt3dr > & LigMul()const ;

        cTplValGesInit< eUniteAngulaire > & UniteAngles();
        const cTplValGesInit< eUniteAngulaire > & UniteAngles()const ;

        cTplValGesInit< Pt3di > & NumAxe();
        const cTplValGesInit< Pt3di > & NumAxe()const ;

        cTplValGesInit< bool > & SensCardan();
        const cTplValGesInit< bool > & SensCardan()const ;

        cTplValGesInit< eConventionsOrientation > & Convention();
        const cTplValGesInit< eConventionsOrientation > & Convention()const ;
    private:
        cTplValGesInit< bool > mSensYVideo;
        cTplValGesInit< bool > mDistSenC2M;
        cTplValGesInit< bool > mMatrSenC2M;
        cTplValGesInit< Pt3dr > mColMul;
        cTplValGesInit< Pt3dr > mLigMul;
        cTplValGesInit< eUniteAngulaire > mUniteAngles;
        cTplValGesInit< Pt3di > mNumAxe;
        cTplValGesInit< bool > mSensCardan;
        cTplValGesInit< eConventionsOrientation > mConvention;
};
cElXMLTree * ToXMLTree(const cConvExplicite &);

void  BinaryDumpInFile(ELISE_fp &,const cConvExplicite &);

void  BinaryUnDumpFromFile(cConvExplicite &,ELISE_fp &);

std::string  Mangling( cConvExplicite *);

class cConvOri
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cConvOri & anObj,cElXMLTree * aTree);


        cTplValGesInit< eConventionsOrientation > & KnownConv();
        const cTplValGesInit< eConventionsOrientation > & KnownConv()const ;

        cTplValGesInit< bool > & SensYVideo();
        const cTplValGesInit< bool > & SensYVideo()const ;

        cTplValGesInit< bool > & DistSenC2M();
        const cTplValGesInit< bool > & DistSenC2M()const ;

        cTplValGesInit< bool > & MatrSenC2M();
        const cTplValGesInit< bool > & MatrSenC2M()const ;

        cTplValGesInit< Pt3dr > & ColMul();
        const cTplValGesInit< Pt3dr > & ColMul()const ;

        cTplValGesInit< Pt3dr > & LigMul();
        const cTplValGesInit< Pt3dr > & LigMul()const ;

        cTplValGesInit< eUniteAngulaire > & UniteAngles();
        const cTplValGesInit< eUniteAngulaire > & UniteAngles()const ;

        cTplValGesInit< Pt3di > & NumAxe();
        const cTplValGesInit< Pt3di > & NumAxe()const ;

        cTplValGesInit< bool > & SensCardan();
        const cTplValGesInit< bool > & SensCardan()const ;

        cTplValGesInit< eConventionsOrientation > & Convention();
        const cTplValGesInit< eConventionsOrientation > & Convention()const ;

        cTplValGesInit< cConvExplicite > & ConvExplicite();
        const cTplValGesInit< cConvExplicite > & ConvExplicite()const ;
    private:
        cTplValGesInit< eConventionsOrientation > mKnownConv;
        cTplValGesInit< cConvExplicite > mConvExplicite;
};
cElXMLTree * ToXMLTree(const cConvOri &);

void  BinaryDumpInFile(ELISE_fp &,const cConvOri &);

void  BinaryUnDumpFromFile(cConvOri &,ELISE_fp &);

std::string  Mangling( cConvOri *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOrientationConique
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOrientationConique & anObj,cElXMLTree * aTree);


        std::string & NameFileOri();
        const std::string & NameFileOri()const ;

        cTplValGesInit< cModuleOrientationFile > & ModuleOrientationFile();
        const cTplValGesInit< cModuleOrientationFile > & ModuleOrientationFile()const ;

        cTplValGesInit< cAffinitePlane > & OrIntImaM2C();
        const cTplValGesInit< cAffinitePlane > & OrIntImaM2C()const ;

        cTplValGesInit< eTypeProjectionCam > & TypeProj();
        const cTplValGesInit< eTypeProjectionCam > & TypeProj()const ;

        cTplValGesInit< bool > & ZoneUtileInPixel();
        const cTplValGesInit< bool > & ZoneUtileInPixel()const ;

        cTplValGesInit< cCalibrationInternConique > & Interne();
        const cTplValGesInit< cCalibrationInternConique > & Interne()const ;

        cTplValGesInit< std::string > & FileInterne();
        const cTplValGesInit< std::string > & FileInterne()const ;

        cTplValGesInit< bool > & RelativeNameFI();
        const cTplValGesInit< bool > & RelativeNameFI()const ;

        cOrientationExterneRigide & Externe();
        const cOrientationExterneRigide & Externe()const ;

        cTplValGesInit< cVerifOrient > & Verif();
        const cTplValGesInit< cVerifOrient > & Verif()const ;

        cTplValGesInit< eConventionsOrientation > & KnownConv();
        const cTplValGesInit< eConventionsOrientation > & KnownConv()const ;

        cTplValGesInit< bool > & SensYVideo();
        const cTplValGesInit< bool > & SensYVideo()const ;

        cTplValGesInit< bool > & DistSenC2M();
        const cTplValGesInit< bool > & DistSenC2M()const ;

        cTplValGesInit< bool > & MatrSenC2M();
        const cTplValGesInit< bool > & MatrSenC2M()const ;

        cTplValGesInit< Pt3dr > & ColMul();
        const cTplValGesInit< Pt3dr > & ColMul()const ;

        cTplValGesInit< Pt3dr > & LigMul();
        const cTplValGesInit< Pt3dr > & LigMul()const ;

        cTplValGesInit< eUniteAngulaire > & UniteAngles();
        const cTplValGesInit< eUniteAngulaire > & UniteAngles()const ;

        cTplValGesInit< Pt3di > & NumAxe();
        const cTplValGesInit< Pt3di > & NumAxe()const ;

        cTplValGesInit< bool > & SensCardan();
        const cTplValGesInit< bool > & SensCardan()const ;

        cTplValGesInit< eConventionsOrientation > & Convention();
        const cTplValGesInit< eConventionsOrientation > & Convention()const ;

        cTplValGesInit< cConvExplicite > & ConvExplicite();
        const cTplValGesInit< cConvExplicite > & ConvExplicite()const ;

        cConvOri & ConvOri();
        const cConvOri & ConvOri()const ;
    private:
        cTplValGesInit< cModuleOrientationFile > mModuleOrientationFile;
        cTplValGesInit< cAffinitePlane > mOrIntImaM2C;
        cTplValGesInit< eTypeProjectionCam > mTypeProj;
        cTplValGesInit< bool > mZoneUtileInPixel;
        cTplValGesInit< cCalibrationInternConique > mInterne;
        cTplValGesInit< std::string > mFileInterne;
        cTplValGesInit< bool > mRelativeNameFI;
        cOrientationExterneRigide mExterne;
        cTplValGesInit< cVerifOrient > mVerif;
        cConvOri mConvOri;
};
cElXMLTree * ToXMLTree(const cOrientationConique &);

void  BinaryDumpInFile(ELISE_fp &,const cOrientationConique &);

void  BinaryUnDumpFromFile(cOrientationConique &,ELISE_fp &);

std::string  Mangling( cOrientationConique *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMNT2Cmp
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMNT2Cmp & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        cTplValGesInit< std::string > & NameXml();
        const cTplValGesInit< std::string > & NameXml()const ;

        cTplValGesInit< int > & IdIsRef();
        const cTplValGesInit< int > & IdIsRef()const ;

        cTplValGesInit< std::string > & ShorName();
        const cTplValGesInit< std::string > & ShorName()const ;
    private:
        std::string mNameIm;
        cTplValGesInit< std::string > mNameXml;
        cTplValGesInit< int > mIdIsRef;
        cTplValGesInit< std::string > mShorName;
};
cElXMLTree * ToXMLTree(const cMNT2Cmp &);

void  BinaryDumpInFile(ELISE_fp &,const cMNT2Cmp &);

void  BinaryUnDumpFromFile(cMNT2Cmp &,ELISE_fp &);

std::string  Mangling( cMNT2Cmp *);

/******************************************************/
/******************************************************/
/******************************************************/
class cContourPolyCM
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContourPolyCM & anObj,cElXMLTree * aTree);


        std::list< Pt2di > & Pts();
        const std::list< Pt2di > & Pts()const ;
    private:
        std::list< Pt2di > mPts;
};
cElXMLTree * ToXMLTree(const cContourPolyCM &);

void  BinaryDumpInFile(ELISE_fp &,const cContourPolyCM &);

void  BinaryUnDumpFromFile(cContourPolyCM &,ELISE_fp &);

std::string  Mangling( cContourPolyCM *);

class cEnvellopeZoneCM
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEnvellopeZoneCM & anObj,cElXMLTree * aTree);


        std::list< Pt2di > & Pts();
        const std::list< Pt2di > & Pts()const ;

        cTplValGesInit< cContourPolyCM > & ContourPolyCM();
        const cTplValGesInit< cContourPolyCM > & ContourPolyCM()const ;

        cTplValGesInit< Box2dr > & BoxContourCM();
        const cTplValGesInit< Box2dr > & BoxContourCM()const ;
    private:
        cTplValGesInit< cContourPolyCM > mContourPolyCM;
        cTplValGesInit< Box2dr > mBoxContourCM;
};
cElXMLTree * ToXMLTree(const cEnvellopeZoneCM &);

void  BinaryDumpInFile(ELISE_fp &,const cEnvellopeZoneCM &);

void  BinaryUnDumpFromFile(cEnvellopeZoneCM &,ELISE_fp &);

std::string  Mangling( cEnvellopeZoneCM *);

class cZoneCmpMnt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cZoneCmpMnt & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NomZone();
        const cTplValGesInit< std::string > & NomZone()const ;

        std::list< Pt2di > & Pts();
        const std::list< Pt2di > & Pts()const ;

        cTplValGesInit< cContourPolyCM > & ContourPolyCM();
        const cTplValGesInit< cContourPolyCM > & ContourPolyCM()const ;

        cTplValGesInit< Box2dr > & BoxContourCM();
        const cTplValGesInit< Box2dr > & BoxContourCM()const ;

        cEnvellopeZoneCM & EnvellopeZoneCM();
        const cEnvellopeZoneCM & EnvellopeZoneCM()const ;
    private:
        cTplValGesInit< std::string > mNomZone;
        cEnvellopeZoneCM mEnvellopeZoneCM;
};
cElXMLTree * ToXMLTree(const cZoneCmpMnt &);

void  BinaryDumpInFile(ELISE_fp &,const cZoneCmpMnt &);

void  BinaryUnDumpFromFile(cZoneCmpMnt &,ELISE_fp &);

std::string  Mangling( cZoneCmpMnt *);

/******************************************************/
/******************************************************/
/******************************************************/
class cEcartZ
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEcartZ & anObj,cElXMLTree * aTree);


        double & DynVisu();
        const double & DynVisu()const ;
    private:
        double mDynVisu;
};
cElXMLTree * ToXMLTree(const cEcartZ &);

void  BinaryDumpInFile(ELISE_fp &,const cEcartZ &);

void  BinaryUnDumpFromFile(cEcartZ &,ELISE_fp &);

std::string  Mangling( cEcartZ *);

class cCorrelPente
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCorrelPente & anObj,cElXMLTree * aTree);


        double & SzWCP();
        const double & SzWCP()const ;

        double & GrMinCP();
        const double & GrMinCP()const ;
    private:
        double mSzWCP;
        double mGrMinCP;
};
cElXMLTree * ToXMLTree(const cCorrelPente &);

void  BinaryDumpInFile(ELISE_fp &,const cCorrelPente &);

void  BinaryUnDumpFromFile(cCorrelPente &,ELISE_fp &);

std::string  Mangling( cCorrelPente *);

class cMesureCmptMnt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMesureCmptMnt & anObj,cElXMLTree * aTree);


        double & DynVisu();
        const double & DynVisu()const ;

        cTplValGesInit< cEcartZ > & EcartZ();
        const cTplValGesInit< cEcartZ > & EcartZ()const ;

        double & SzWCP();
        const double & SzWCP()const ;

        double & GrMinCP();
        const double & GrMinCP()const ;

        cTplValGesInit< cCorrelPente > & CorrelPente();
        const cTplValGesInit< cCorrelPente > & CorrelPente()const ;

        cTplValGesInit< bool > & EcartPente();
        const cTplValGesInit< bool > & EcartPente()const ;
    private:
        cTplValGesInit< cEcartZ > mEcartZ;
        cTplValGesInit< cCorrelPente > mCorrelPente;
        cTplValGesInit< bool > mEcartPente;
};
cElXMLTree * ToXMLTree(const cMesureCmptMnt &);

void  BinaryDumpInFile(ELISE_fp &,const cMesureCmptMnt &);

void  BinaryUnDumpFromFile(cMesureCmptMnt &,ELISE_fp &);

std::string  Mangling( cMesureCmptMnt *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCompareMNT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCompareMNT & anObj,cElXMLTree * aTree);


        Pt2dr & ResolutionPlaniTerrain();
        const Pt2dr & ResolutionPlaniTerrain()const ;

        cTplValGesInit< int > & RabLoad();
        const cTplValGesInit< int > & RabLoad()const ;

        std::string & NameFileRes();
        const std::string & NameFileRes()const ;

        cTplValGesInit< bool > & VisuInter();
        const cTplValGesInit< bool > & VisuInter()const ;

        std::list< cMNT2Cmp > & MNT2Cmp();
        const std::list< cMNT2Cmp > & MNT2Cmp()const ;

        cTplValGesInit< std::string > & MasqGlobalCM();
        const cTplValGesInit< std::string > & MasqGlobalCM()const ;

        std::list< cZoneCmpMnt > & ZoneCmpMnt();
        const std::list< cZoneCmpMnt > & ZoneCmpMnt()const ;

        double & DynVisu();
        const double & DynVisu()const ;

        cTplValGesInit< cEcartZ > & EcartZ();
        const cTplValGesInit< cEcartZ > & EcartZ()const ;

        double & SzWCP();
        const double & SzWCP()const ;

        double & GrMinCP();
        const double & GrMinCP()const ;

        cTplValGesInit< cCorrelPente > & CorrelPente();
        const cTplValGesInit< cCorrelPente > & CorrelPente()const ;

        cTplValGesInit< bool > & EcartPente();
        const cTplValGesInit< bool > & EcartPente()const ;

        cMesureCmptMnt & MesureCmptMnt();
        const cMesureCmptMnt & MesureCmptMnt()const ;
    private:
        Pt2dr mResolutionPlaniTerrain;
        cTplValGesInit< int > mRabLoad;
        std::string mNameFileRes;
        cTplValGesInit< bool > mVisuInter;
        std::list< cMNT2Cmp > mMNT2Cmp;
        cTplValGesInit< std::string > mMasqGlobalCM;
        std::list< cZoneCmpMnt > mZoneCmpMnt;
        cMesureCmptMnt mMesureCmptMnt;
};
cElXMLTree * ToXMLTree(const cCompareMNT &);

void  BinaryDumpInFile(ELISE_fp &,const cCompareMNT &);

void  BinaryUnDumpFromFile(cCompareMNT &,ELISE_fp &);

std::string  Mangling( cCompareMNT *);

/******************************************************/
/******************************************************/
/******************************************************/
class cDataBaseNameTransfo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDataBaseNameTransfo & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & AddFocMul();
        const cTplValGesInit< double > & AddFocMul()const ;

        cTplValGesInit< std::string > & Separateur();
        const cTplValGesInit< std::string > & Separateur()const ;

        cTplValGesInit< std::string > & NewKeyId();
        const cTplValGesInit< std::string > & NewKeyId()const ;

        cTplValGesInit< std::string > & NewKeyIdAdd();
        const cTplValGesInit< std::string > & NewKeyIdAdd()const ;

        cTplValGesInit< bool > & NewAddNameCam();
        const cTplValGesInit< bool > & NewAddNameCam()const ;

        cTplValGesInit< double > & NewFocMul();
        const cTplValGesInit< double > & NewFocMul()const ;
    private:
        cTplValGesInit< double > mAddFocMul;
        cTplValGesInit< std::string > mSeparateur;
        cTplValGesInit< std::string > mNewKeyId;
        cTplValGesInit< std::string > mNewKeyIdAdd;
        cTplValGesInit< bool > mNewAddNameCam;
        cTplValGesInit< double > mNewFocMul;
};
cElXMLTree * ToXMLTree(const cDataBaseNameTransfo &);

void  BinaryDumpInFile(ELISE_fp &,const cDataBaseNameTransfo &);

void  BinaryUnDumpFromFile(cDataBaseNameTransfo &,ELISE_fp &);

std::string  Mangling( cDataBaseNameTransfo *);

/******************************************************/
/******************************************************/
/******************************************************/
class cInterpoleGrille
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cInterpoleGrille & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Directory();
        const cTplValGesInit< std::string > & Directory()const ;

        std::string & Grille1();
        const std::string & Grille1()const ;

        std::string & Grille2();
        const std::string & Grille2()const ;

        std::string & Grille0();
        const std::string & Grille0()const ;

        cTplValGesInit< Pt2dr > & StepGrid();
        const cTplValGesInit< Pt2dr > & StepGrid()const ;

        double & Focale1();
        const double & Focale1()const ;

        double & Focale2();
        const double & Focale2()const ;

        double & Focale0();
        const double & Focale0()const ;

        cTplValGesInit< int > & NbPtsByIter();
        const cTplValGesInit< int > & NbPtsByIter()const ;

        cTplValGesInit< int > & DegPoly();
        const cTplValGesInit< int > & DegPoly()const ;

        cTplValGesInit< eDegreLiberteCPP > & LiberteCPP();
        const cTplValGesInit< eDegreLiberteCPP > & LiberteCPP()const ;
    private:
        cTplValGesInit< std::string > mDirectory;
        std::string mGrille1;
        std::string mGrille2;
        std::string mGrille0;
        cTplValGesInit< Pt2dr > mStepGrid;
        double mFocale1;
        double mFocale2;
        double mFocale0;
        cTplValGesInit< int > mNbPtsByIter;
        cTplValGesInit< int > mDegPoly;
        cTplValGesInit< eDegreLiberteCPP > mLiberteCPP;
};
cElXMLTree * ToXMLTree(const cInterpoleGrille &);

void  BinaryDumpInFile(ELISE_fp &,const cInterpoleGrille &);

void  BinaryUnDumpFromFile(cInterpoleGrille &,ELISE_fp &);

std::string  Mangling( cInterpoleGrille *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneCalib2Visu
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneCalib2Visu & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        Pt3dr & Coul();
        const Pt3dr & Coul()const ;
    private:
        std::string mName;
        Pt3dr mCoul;
};
cElXMLTree * ToXMLTree(const cOneCalib2Visu &);

void  BinaryDumpInFile(ELISE_fp &,const cOneCalib2Visu &);

void  BinaryUnDumpFromFile(cOneCalib2Visu &,ELISE_fp &);

std::string  Mangling( cOneCalib2Visu *);

/******************************************************/
/******************************************************/
/******************************************************/
class cVisuCalibZoom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVisuCalibZoom & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Directory();
        const cTplValGesInit< std::string > & Directory()const ;

        Pt2dr & SzIm();
        const Pt2dr & SzIm()const ;

        std::list< cOneCalib2Visu > & OneCalib2Visu();
        const std::list< cOneCalib2Visu > & OneCalib2Visu()const ;
    private:
        cTplValGesInit< std::string > mDirectory;
        Pt2dr mSzIm;
        std::list< cOneCalib2Visu > mOneCalib2Visu;
};
cElXMLTree * ToXMLTree(const cVisuCalibZoom &);

void  BinaryDumpInFile(ELISE_fp &,const cVisuCalibZoom &);

void  BinaryUnDumpFromFile(cVisuCalibZoom &,ELISE_fp &);

std::string  Mangling( cVisuCalibZoom *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFilterLocalisation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFilterLocalisation & anObj,cElXMLTree * aTree);


        std::string & KeyAssocOrient();
        const std::string & KeyAssocOrient()const ;

        std::string & NameMasq();
        const std::string & NameMasq()const ;

        std::string & NameMTDMasq();
        const std::string & NameMTDMasq()const ;
    private:
        std::string mKeyAssocOrient;
        std::string mNameMasq;
        std::string mNameMTDMasq;
};
cElXMLTree * ToXMLTree(const cFilterLocalisation &);

void  BinaryDumpInFile(ELISE_fp &,const cFilterLocalisation &);

void  BinaryUnDumpFromFile(cFilterLocalisation &,ELISE_fp &);

std::string  Mangling( cFilterLocalisation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cKeyExistingFile
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cKeyExistingFile & anObj,cElXMLTree * aTree);


        std::list< std::string > & KeyAssoc();
        const std::list< std::string > & KeyAssoc()const ;

        bool & RequireExist();
        const bool & RequireExist()const ;

        bool & RequireForAll();
        const bool & RequireForAll()const ;
    private:
        std::list< std::string > mKeyAssoc;
        bool mRequireExist;
        bool mRequireForAll;
};
cElXMLTree * ToXMLTree(const cKeyExistingFile &);

void  BinaryDumpInFile(ELISE_fp &,const cKeyExistingFile &);

void  BinaryUnDumpFromFile(cKeyExistingFile &,ELISE_fp &);

std::string  Mangling( cKeyExistingFile *);

/******************************************************/
/******************************************************/
/******************************************************/
class cNameFilter
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNameFilter & anObj,cElXMLTree * aTree);


        std::list< Pt2drSubst > & FocMm();
        const std::list< Pt2drSubst > & FocMm()const ;

        cTplValGesInit< std::string > & Min();
        const cTplValGesInit< std::string > & Min()const ;

        cTplValGesInit< std::string > & Max();
        const cTplValGesInit< std::string > & Max()const ;

        cTplValGesInit< int > & SizeMinFile();
        const cTplValGesInit< int > & SizeMinFile()const ;

        std::list< cKeyExistingFile > & KeyExistingFile();
        const std::list< cKeyExistingFile > & KeyExistingFile()const ;

        cTplValGesInit< cFilterLocalisation > & KeyLocalisation();
        const cTplValGesInit< cFilterLocalisation > & KeyLocalisation()const ;
    private:
        std::list< Pt2drSubst > mFocMm;
        cTplValGesInit< std::string > mMin;
        cTplValGesInit< std::string > mMax;
        cTplValGesInit< int > mSizeMinFile;
        std::list< cKeyExistingFile > mKeyExistingFile;
        cTplValGesInit< cFilterLocalisation > mKeyLocalisation;
};
cElXMLTree * ToXMLTree(const cNameFilter &);

void  BinaryDumpInFile(ELISE_fp &,const cNameFilter &);

void  BinaryUnDumpFromFile(cNameFilter &,ELISE_fp &);

std::string  Mangling( cNameFilter *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBasicAssocNameToName
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasicAssocNameToName & anObj,cElXMLTree * aTree);


        std::string & PatternTransform();
        const std::string & PatternTransform()const ;

        cTplValGesInit< cDataBaseNameTransfo > & NameTransfo();
        const cTplValGesInit< cDataBaseNameTransfo > & NameTransfo()const ;

        cTplValGesInit< std::string > & PatternSelector();
        const cTplValGesInit< std::string > & PatternSelector()const ;

        std::vector< std::string > & CalcName();
        const std::vector< std::string > & CalcName()const ;

        cTplValGesInit< std::string > & Separateur();
        const cTplValGesInit< std::string > & Separateur()const ;

        cTplValGesInit< cNameFilter > & Filter();
        const cTplValGesInit< cNameFilter > & Filter()const ;
    private:
        std::string mPatternTransform;
        cTplValGesInit< cDataBaseNameTransfo > mNameTransfo;
        cTplValGesInit< std::string > mPatternSelector;
        std::vector< std::string > mCalcName;
        cTplValGesInit< std::string > mSeparateur;
        cTplValGesInit< cNameFilter > mFilter;
};
cElXMLTree * ToXMLTree(const cBasicAssocNameToName &);

void  BinaryDumpInFile(ELISE_fp &,const cBasicAssocNameToName &);

void  BinaryUnDumpFromFile(cBasicAssocNameToName &,ELISE_fp &);

std::string  Mangling( cBasicAssocNameToName *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAssocNameToName
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAssocNameToName & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt2di > & Arrite();
        const cTplValGesInit< Pt2di > & Arrite()const ;

        cBasicAssocNameToName & Direct();
        const cBasicAssocNameToName & Direct()const ;

        cTplValGesInit< cBasicAssocNameToName > & Inverse();
        const cTplValGesInit< cBasicAssocNameToName > & Inverse()const ;

        cTplValGesInit< bool > & AutoInverseBySym();
        const cTplValGesInit< bool > & AutoInverseBySym()const ;
    private:
        cTplValGesInit< Pt2di > mArrite;
        cBasicAssocNameToName mDirect;
        cTplValGesInit< cBasicAssocNameToName > mInverse;
        cTplValGesInit< bool > mAutoInverseBySym;
};
cElXMLTree * ToXMLTree(const cAssocNameToName &);

void  BinaryDumpInFile(ELISE_fp &,const cAssocNameToName &);

void  BinaryUnDumpFromFile(cAssocNameToName &,ELISE_fp &);

std::string  Mangling( cAssocNameToName *);

/******************************************************/
/******************************************************/
/******************************************************/
class cEtatPims
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEtatPims & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NameOri();
        const cTplValGesInit< std::string > & NameOri()const ;
    private:
        cTplValGesInit< std::string > mNameOri;
};
cElXMLTree * ToXMLTree(const cEtatPims &);

void  BinaryDumpInFile(ELISE_fp &,const cEtatPims &);

void  BinaryUnDumpFromFile(cEtatPims &,ELISE_fp &);

std::string  Mangling( cEtatPims *);

/******************************************************/
/******************************************************/
/******************************************************/
class cListOfName
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cListOfName & anObj,cElXMLTree * aTree);


        std::list< std::string > & Name();
        const std::list< std::string > & Name()const ;
    private:
        std::list< std::string > mName;
};
cElXMLTree * ToXMLTree(const cListOfName &);

void  BinaryDumpInFile(ELISE_fp &,const cListOfName &);

void  BinaryUnDumpFromFile(cListOfName &,ELISE_fp &);

std::string  Mangling( cListOfName *);

/******************************************************/
/******************************************************/
/******************************************************/
class cModLin
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModLin & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        double & a();
        const double & a()const ;

        double & b();
        const double & b()const ;
    private:
        std::string mNameIm;
        double ma;
        double mb;
};
cElXMLTree * ToXMLTree(const cModLin &);

void  BinaryDumpInFile(ELISE_fp &,const cModLin &);

void  BinaryUnDumpFromFile(cModLin &,ELISE_fp &);

std::string  Mangling( cModLin *);

/******************************************************/
/******************************************************/
/******************************************************/
class cListOfRadiomEgalModel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cListOfRadiomEgalModel & anObj,cElXMLTree * aTree);


        std::list< cModLin > & ModLin();
        const std::list< cModLin > & ModLin()const ;
    private:
        std::list< cModLin > mModLin;
};
cElXMLTree * ToXMLTree(const cListOfRadiomEgalModel &);

void  BinaryDumpInFile(ELISE_fp &,const cListOfRadiomEgalModel &);

void  BinaryUnDumpFromFile(cListOfRadiomEgalModel &,ELISE_fp &);

std::string  Mangling( cListOfRadiomEgalModel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetNameDescriptor
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetNameDescriptor & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & AddDirCur();
        const cTplValGesInit< bool > & AddDirCur()const ;

        std::list< std::string > & PatternAccepteur();
        const std::list< std::string > & PatternAccepteur()const ;

        std::list< std::string > & PatternRefuteur();
        const std::list< std::string > & PatternRefuteur()const ;

        cTplValGesInit< int > & NivSubDir();
        const cTplValGesInit< int > & NivSubDir()const ;

        cTplValGesInit< bool > & NameCompl();
        const cTplValGesInit< bool > & NameCompl()const ;

        cTplValGesInit< std::string > & SubDir();
        const cTplValGesInit< std::string > & SubDir()const ;

        std::list< std::string > & Name();
        const std::list< std::string > & Name()const ;

        std::list< std::string > & NamesFileLON();
        const std::list< std::string > & NamesFileLON()const ;

        cTplValGesInit< std::string > & Min();
        const cTplValGesInit< std::string > & Min()const ;

        cTplValGesInit< std::string > & Max();
        const cTplValGesInit< std::string > & Max()const ;

        cTplValGesInit< cNameFilter > & Filter();
        const cTplValGesInit< cNameFilter > & Filter()const ;
    private:
        cTplValGesInit< bool > mAddDirCur;
        std::list< std::string > mPatternAccepteur;
        std::list< std::string > mPatternRefuteur;
        cTplValGesInit< int > mNivSubDir;
        cTplValGesInit< bool > mNameCompl;
        cTplValGesInit< std::string > mSubDir;
        std::list< std::string > mName;
        std::list< std::string > mNamesFileLON;
        cTplValGesInit< std::string > mMin;
        cTplValGesInit< std::string > mMax;
        cTplValGesInit< cNameFilter > mFilter;
};
cElXMLTree * ToXMLTree(const cSetNameDescriptor &);

void  BinaryDumpInFile(ELISE_fp &,const cSetNameDescriptor &);

void  BinaryUnDumpFromFile(cSetNameDescriptor &,ELISE_fp &);

std::string  Mangling( cSetNameDescriptor *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImMatrixStructuration
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImMatrixStructuration & anObj,cElXMLTree * aTree);


        std::string & KeySet();
        const std::string & KeySet()const ;

        Pt2di & Period();
        const Pt2di & Period()const ;

        bool & XCroissants();
        const bool & XCroissants()const ;

        bool & YCroissants();
        const bool & YCroissants()const ;

        bool & XVarieFirst();
        const bool & XVarieFirst()const ;
    private:
        std::string mKeySet;
        Pt2di mPeriod;
        bool mXCroissants;
        bool mYCroissants;
        bool mXVarieFirst;
};
cElXMLTree * ToXMLTree(const cImMatrixStructuration &);

void  BinaryDumpInFile(ELISE_fp &,const cImMatrixStructuration &);

void  BinaryUnDumpFromFile(cImMatrixStructuration &,ELISE_fp &);

std::string  Mangling( cImMatrixStructuration *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFiltreEmprise
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFiltreEmprise & anObj,cElXMLTree * aTree);


        std::string & KeyOri();
        const std::string & KeyOri()const ;

        double & RatioMin();
        const double & RatioMin()const ;

        cTplValGesInit< bool > & MemoFile();
        const cTplValGesInit< bool > & MemoFile()const ;

        cTplValGesInit< std::string > & Tag();
        const cTplValGesInit< std::string > & Tag()const ;
    private:
        std::string mKeyOri;
        double mRatioMin;
        cTplValGesInit< bool > mMemoFile;
        cTplValGesInit< std::string > mTag;
};
cElXMLTree * ToXMLTree(const cFiltreEmprise &);

void  BinaryDumpInFile(ELISE_fp &,const cFiltreEmprise &);

void  BinaryUnDumpFromFile(cFiltreEmprise &,ELISE_fp &);

std::string  Mangling( cFiltreEmprise *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFiltreByRelSsEch
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFiltreByRelSsEch & anObj,cElXMLTree * aTree);


        std::string & KeySet();
        const std::string & KeySet()const ;

        std::string & KeyAssocCple();
        const std::string & KeyAssocCple()const ;

        IntSubst & SeuilBasNbPts();
        const IntSubst & SeuilBasNbPts()const ;

        IntSubst & SeuilHautNbPts();
        const IntSubst & SeuilHautNbPts()const ;

        IntSubst & NbMinCple();
        const IntSubst & NbMinCple()const ;
    private:
        std::string mKeySet;
        std::string mKeyAssocCple;
        IntSubst mSeuilBasNbPts;
        IntSubst mSeuilHautNbPts;
        IntSubst mNbMinCple;
};
cElXMLTree * ToXMLTree(const cFiltreByRelSsEch &);

void  BinaryDumpInFile(ELISE_fp &,const cFiltreByRelSsEch &);

void  BinaryUnDumpFromFile(cFiltreByRelSsEch &,ELISE_fp &);

std::string  Mangling( cFiltreByRelSsEch *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFiltreDeRelationOrient
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFiltreDeRelationOrient & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyEquiv();
        const cTplValGesInit< std::string > & KeyEquiv()const ;

        std::string & KeyOri();
        const std::string & KeyOri()const ;

        double & RatioMin();
        const double & RatioMin()const ;

        cTplValGesInit< bool > & MemoFile();
        const cTplValGesInit< bool > & MemoFile()const ;

        cTplValGesInit< std::string > & Tag();
        const cTplValGesInit< std::string > & Tag()const ;

        cTplValGesInit< cFiltreEmprise > & FiltreEmprise();
        const cTplValGesInit< cFiltreEmprise > & FiltreEmprise()const ;

        cTplValGesInit< std::string > & FiltreAdjMatrix();
        const cTplValGesInit< std::string > & FiltreAdjMatrix()const ;

        cTplValGesInit< Pt2di > & EcartFiltreMatr();
        const cTplValGesInit< Pt2di > & EcartFiltreMatr()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        std::string & KeyAssocCple();
        const std::string & KeyAssocCple()const ;

        IntSubst & SeuilBasNbPts();
        const IntSubst & SeuilBasNbPts()const ;

        IntSubst & SeuilHautNbPts();
        const IntSubst & SeuilHautNbPts()const ;

        IntSubst & NbMinCple();
        const IntSubst & NbMinCple()const ;

        cTplValGesInit< cFiltreByRelSsEch > & FiltreByRelSsEch();
        const cTplValGesInit< cFiltreByRelSsEch > & FiltreByRelSsEch()const ;
    private:
        cTplValGesInit< std::string > mKeyEquiv;
        cTplValGesInit< cFiltreEmprise > mFiltreEmprise;
        cTplValGesInit< std::string > mFiltreAdjMatrix;
        cTplValGesInit< Pt2di > mEcartFiltreMatr;
        cTplValGesInit< cFiltreByRelSsEch > mFiltreByRelSsEch;
};
cElXMLTree * ToXMLTree(const cFiltreDeRelationOrient &);

void  BinaryDumpInFile(ELISE_fp &,const cFiltreDeRelationOrient &);

void  BinaryUnDumpFromFile(cFiltreDeRelationOrient &,ELISE_fp &);

std::string  Mangling( cFiltreDeRelationOrient *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSauvegardeNamedRel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSauvegardeNamedRel & anObj,cElXMLTree * aTree);


        std::vector< cCpleString > & Cple();
        const std::vector< cCpleString > & Cple()const ;
    private:
        std::vector< cCpleString > mCple;
};
cElXMLTree * ToXMLTree(const cSauvegardeNamedRel &);

void  BinaryDumpInFile(ELISE_fp &,const cSauvegardeNamedRel &);

void  BinaryUnDumpFromFile(cSauvegardeNamedRel &,ELISE_fp &);

std::string  Mangling( cSauvegardeNamedRel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSauvegardeSetString
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSauvegardeSetString & anObj,cElXMLTree * aTree);


        std::list< std::string > & Name();
        const std::list< std::string > & Name()const ;
    private:
        std::list< std::string > mName;
};
cElXMLTree * ToXMLTree(const cSauvegardeSetString &);

void  BinaryDumpInFile(ELISE_fp &,const cSauvegardeSetString &);

void  BinaryUnDumpFromFile(cSauvegardeSetString &,ELISE_fp &);

std::string  Mangling( cSauvegardeSetString *);

/******************************************************/
/******************************************************/
/******************************************************/
class cClassEquivDescripteur
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cClassEquivDescripteur & anObj,cElXMLTree * aTree);


        std::string & KeySet();
        const std::string & KeySet()const ;

        std::string & KeyAssocRep();
        const std::string & KeyAssocRep()const ;

        std::string & KeyClass();
        const std::string & KeyClass()const ;
    private:
        std::string mKeySet;
        std::string mKeyAssocRep;
        std::string mKeyClass;
};
cElXMLTree * ToXMLTree(const cClassEquivDescripteur &);

void  BinaryDumpInFile(ELISE_fp &,const cClassEquivDescripteur &);

void  BinaryUnDumpFromFile(cClassEquivDescripteur &,ELISE_fp &);

std::string  Mangling( cClassEquivDescripteur *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneSpecDelta
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneSpecDelta & anObj,cElXMLTree * aTree);


        std::vector<std::string> & Soms();
        const std::vector<std::string> & Soms()const ;

        std::vector<int>  & Delta();
        const std::vector<int>  & Delta()const ;
    private:
        std::vector<std::string> mSoms;
        std::vector<int>  mDelta;
};
cElXMLTree * ToXMLTree(const cOneSpecDelta &);

void  BinaryDumpInFile(ELISE_fp &,const cOneSpecDelta &);

void  BinaryUnDumpFromFile(cOneSpecDelta &,ELISE_fp &);

std::string  Mangling( cOneSpecDelta *);

class cGrByDelta
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGrByDelta & anObj,cElXMLTree * aTree);


        std::string & KeySet();
        const std::string & KeySet()const ;

        std::list< cOneSpecDelta > & OneSpecDelta();
        const std::list< cOneSpecDelta > & OneSpecDelta()const ;
    private:
        std::string mKeySet;
        std::list< cOneSpecDelta > mOneSpecDelta;
};
cElXMLTree * ToXMLTree(const cGrByDelta &);

void  BinaryDumpInFile(ELISE_fp &,const cGrByDelta &);

void  BinaryUnDumpFromFile(cGrByDelta &,ELISE_fp &);

std::string  Mangling( cGrByDelta *);

class cRelByGrapheExpl
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRelByGrapheExpl & anObj,cElXMLTree * aTree);


        std::list< cCpleString > & Cples();
        const std::list< cCpleString > & Cples()const ;

        std::list< std::vector<std::string> > & CpleSymWithFirt();
        const std::list< std::vector<std::string> > & CpleSymWithFirt()const ;

        std::list< std::vector<std::string> > & ProdCartesien();
        const std::list< std::vector<std::string> > & ProdCartesien()const ;

        cTplValGesInit< std::string > & Prefix2Name();
        const cTplValGesInit< std::string > & Prefix2Name()const ;

        cTplValGesInit< std::string > & Postfix2Name();
        const cTplValGesInit< std::string > & Postfix2Name()const ;

        std::list< cGrByDelta > & GrByDelta();
        const std::list< cGrByDelta > & GrByDelta()const ;
    private:
        std::list< cCpleString > mCples;
        std::list< std::vector<std::string> > mCpleSymWithFirt;
        std::list< std::vector<std::string> > mProdCartesien;
        cTplValGesInit< std::string > mPrefix2Name;
        cTplValGesInit< std::string > mPostfix2Name;
        std::list< cGrByDelta > mGrByDelta;
};
cElXMLTree * ToXMLTree(const cRelByGrapheExpl &);

void  BinaryDumpInFile(ELISE_fp &,const cRelByGrapheExpl &);

void  BinaryUnDumpFromFile(cRelByGrapheExpl &,ELISE_fp &);

std::string  Mangling( cRelByGrapheExpl *);

/******************************************************/
/******************************************************/
/******************************************************/
class cByAdjDeGroupes
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cByAdjDeGroupes & anObj,cElXMLTree * aTree);


        std::vector< std::string > & KeySets();
        const std::vector< std::string > & KeySets()const ;

        int & DeltaMin();
        const int & DeltaMin()const ;

        int & DeltaMax();
        const int & DeltaMax()const ;
    private:
        std::vector< std::string > mKeySets;
        int mDeltaMin;
        int mDeltaMax;
};
cElXMLTree * ToXMLTree(const cByAdjDeGroupes &);

void  BinaryDumpInFile(ELISE_fp &,const cByAdjDeGroupes &);

void  BinaryUnDumpFromFile(cByAdjDeGroupes &,ELISE_fp &);

std::string  Mangling( cByAdjDeGroupes *);

class cByGroupesDImages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cByGroupesDImages & anObj,cElXMLTree * aTree);


        std::list< cCpleString > & CplesKey();
        const std::list< cCpleString > & CplesKey()const ;

        std::list< cByAdjDeGroupes > & ByAdjDeGroupes();
        const std::list< cByAdjDeGroupes > & ByAdjDeGroupes()const ;

        cTplValGesInit< cFiltreDeRelationOrient > & Filtre();
        const cTplValGesInit< cFiltreDeRelationOrient > & Filtre()const ;

        cTplValGesInit< bool > & Sym();
        const cTplValGesInit< bool > & Sym()const ;

        cTplValGesInit< bool > & Reflexif();
        const cTplValGesInit< bool > & Reflexif()const ;
    private:
        std::list< cCpleString > mCplesKey;
        std::list< cByAdjDeGroupes > mByAdjDeGroupes;
        cTplValGesInit< cFiltreDeRelationOrient > mFiltre;
        cTplValGesInit< bool > mSym;
        cTplValGesInit< bool > mReflexif;
};
cElXMLTree * ToXMLTree(const cByGroupesDImages &);

void  BinaryDumpInFile(ELISE_fp &,const cByGroupesDImages &);

void  BinaryUnDumpFromFile(cByGroupesDImages &,ELISE_fp &);

std::string  Mangling( cByGroupesDImages *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFiltreDelaunay
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFiltreDelaunay & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & DMaxDelaunay();
        const cTplValGesInit< double > & DMaxDelaunay()const ;
    private:
        cTplValGesInit< double > mDMaxDelaunay;
};
cElXMLTree * ToXMLTree(const cFiltreDelaunay &);

void  BinaryDumpInFile(ELISE_fp &,const cFiltreDelaunay &);

void  BinaryUnDumpFromFile(cFiltreDelaunay &,ELISE_fp &);

std::string  Mangling( cFiltreDelaunay *);

class cFiltreDist
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFiltreDist & anObj,cElXMLTree * aTree);


        double & DistMax();
        const double & DistMax()const ;
    private:
        double mDistMax;
};
cElXMLTree * ToXMLTree(const cFiltreDist &);

void  BinaryDumpInFile(ELISE_fp &,const cFiltreDist &);

void  BinaryUnDumpFromFile(cFiltreDist &,ELISE_fp &);

std::string  Mangling( cFiltreDist *);

class cModeFiltreSpatial
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModeFiltreSpatial & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & DMaxDelaunay();
        const cTplValGesInit< double > & DMaxDelaunay()const ;

        cTplValGesInit< cFiltreDelaunay > & FiltreDelaunay();
        const cTplValGesInit< cFiltreDelaunay > & FiltreDelaunay()const ;

        double & DistMax();
        const double & DistMax()const ;

        cTplValGesInit< cFiltreDist > & FiltreDist();
        const cTplValGesInit< cFiltreDist > & FiltreDist()const ;
    private:
        cTplValGesInit< cFiltreDelaunay > mFiltreDelaunay;
        cTplValGesInit< cFiltreDist > mFiltreDist;
};
cElXMLTree * ToXMLTree(const cModeFiltreSpatial &);

void  BinaryDumpInFile(ELISE_fp &,const cModeFiltreSpatial &);

void  BinaryUnDumpFromFile(cModeFiltreSpatial &,ELISE_fp &);

std::string  Mangling( cModeFiltreSpatial *);

class cByFiltreSpatial
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cByFiltreSpatial & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & ByFileTrajecto();
        const cTplValGesInit< std::string > & ByFileTrajecto()const ;

        std::string & KeyOri();
        const std::string & KeyOri()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        cTplValGesInit< std::string > & TagCentre();
        const cTplValGesInit< std::string > & TagCentre()const ;

        cTplValGesInit< bool > & Sym();
        const cTplValGesInit< bool > & Sym()const ;

        cTplValGesInit< cFiltreDeRelationOrient > & FiltreSup();
        const cTplValGesInit< cFiltreDeRelationOrient > & FiltreSup()const ;

        cTplValGesInit< double > & DMaxDelaunay();
        const cTplValGesInit< double > & DMaxDelaunay()const ;

        cTplValGesInit< cFiltreDelaunay > & FiltreDelaunay();
        const cTplValGesInit< cFiltreDelaunay > & FiltreDelaunay()const ;

        double & DistMax();
        const double & DistMax()const ;

        cTplValGesInit< cFiltreDist > & FiltreDist();
        const cTplValGesInit< cFiltreDist > & FiltreDist()const ;

        cModeFiltreSpatial & ModeFiltreSpatial();
        const cModeFiltreSpatial & ModeFiltreSpatial()const ;
    private:
        cTplValGesInit< std::string > mByFileTrajecto;
        std::string mKeyOri;
        std::string mKeySet;
        cTplValGesInit< std::string > mTagCentre;
        cTplValGesInit< bool > mSym;
        cTplValGesInit< cFiltreDeRelationOrient > mFiltreSup;
        cModeFiltreSpatial mModeFiltreSpatial;
};
cElXMLTree * ToXMLTree(const cByFiltreSpatial &);

void  BinaryDumpInFile(ELISE_fp &,const cByFiltreSpatial &);

void  BinaryUnDumpFromFile(cByFiltreSpatial &,ELISE_fp &);

std::string  Mangling( cByFiltreSpatial *);

/******************************************************/
/******************************************************/
/******************************************************/
class cByAdjacence
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cByAdjacence & anObj,cElXMLTree * aTree);


        std::vector< std::string > & KeySets();
        const std::vector< std::string > & KeySets()const ;

        cTplValGesInit< IntSubst > & DeltaMax();
        const cTplValGesInit< IntSubst > & DeltaMax()const ;

        cTplValGesInit< IntSubst > & DeltaMin();
        const cTplValGesInit< IntSubst > & DeltaMin()const ;

        cTplValGesInit< IntSubst > & Sampling();
        const cTplValGesInit< IntSubst > & Sampling()const ;

        cTplValGesInit< cFiltreDeRelationOrient > & Filtre();
        const cTplValGesInit< cFiltreDeRelationOrient > & Filtre()const ;

        cTplValGesInit< bool > & Sym();
        const cTplValGesInit< bool > & Sym()const ;

        cTplValGesInit< BoolSubst > & Circ();
        const cTplValGesInit< BoolSubst > & Circ()const ;
    private:
        std::vector< std::string > mKeySets;
        cTplValGesInit< IntSubst > mDeltaMax;
        cTplValGesInit< IntSubst > mDeltaMin;
        cTplValGesInit< IntSubst > mSampling;
        cTplValGesInit< cFiltreDeRelationOrient > mFiltre;
        cTplValGesInit< bool > mSym;
        cTplValGesInit< BoolSubst > mCirc;
};
cElXMLTree * ToXMLTree(const cByAdjacence &);

void  BinaryDumpInFile(ELISE_fp &,const cByAdjacence &);

void  BinaryUnDumpFromFile(cByAdjacence &,ELISE_fp &);

std::string  Mangling( cByAdjacence *);

/******************************************************/
/******************************************************/
/******************************************************/
class cNameRelDescriptor
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNameRelDescriptor & anObj,cElXMLTree * aTree);


        std::list< std::string > & NameFileIn();
        const std::list< std::string > & NameFileIn()const ;

        cTplValGesInit< bool > & Reflexif();
        const cTplValGesInit< bool > & Reflexif()const ;

        cTplValGesInit< std::string > & NameFileSauvegarde();
        const cTplValGesInit< std::string > & NameFileSauvegarde()const ;

        std::list< cRelByGrapheExpl > & RelByGrapheExpl();
        const std::list< cRelByGrapheExpl > & RelByGrapheExpl()const ;

        std::list< cByGroupesDImages > & ByGroupesDImages();
        const std::list< cByGroupesDImages > & ByGroupesDImages()const ;

        std::list< cByFiltreSpatial > & ByFiltreSpatial();
        const std::list< cByFiltreSpatial > & ByFiltreSpatial()const ;

        std::list< cByAdjacence > & ByAdjacence();
        const std::list< cByAdjacence > & ByAdjacence()const ;

        std::list< cCpleString > & CplesExcl();
        const std::list< cCpleString > & CplesExcl()const ;
    private:
        std::list< std::string > mNameFileIn;
        cTplValGesInit< bool > mReflexif;
        cTplValGesInit< std::string > mNameFileSauvegarde;
        std::list< cRelByGrapheExpl > mRelByGrapheExpl;
        std::list< cByGroupesDImages > mByGroupesDImages;
        std::list< cByFiltreSpatial > mByFiltreSpatial;
        std::list< cByAdjacence > mByAdjacence;
        std::list< cCpleString > mCplesExcl;
};
cElXMLTree * ToXMLTree(const cNameRelDescriptor &);

void  BinaryDumpInFile(ELISE_fp &,const cNameRelDescriptor &);

void  BinaryUnDumpFromFile(cNameRelDescriptor &,ELISE_fp &);

std::string  Mangling( cNameRelDescriptor *);

/******************************************************/
/******************************************************/
/******************************************************/
class cExeRequired
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExeRequired & anObj,cElXMLTree * aTree);


        std::string & Exe();
        const std::string & Exe()const ;

        std::string & Make();
        const std::string & Make()const ;
    private:
        std::string mExe;
        std::string mMake;
};
cElXMLTree * ToXMLTree(const cExeRequired &);

void  BinaryDumpInFile(ELISE_fp &,const cExeRequired &);

void  BinaryUnDumpFromFile(cExeRequired &,ELISE_fp &);

std::string  Mangling( cExeRequired *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFileRequired
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFileRequired & anObj,cElXMLTree * aTree);


        std::list< std::string > & Pattern();
        const std::list< std::string > & Pattern()const ;

        cTplValGesInit< int > & NbMin();
        const cTplValGesInit< int > & NbMin()const ;

        cTplValGesInit< int > & NbMax();
        const cTplValGesInit< int > & NbMax()const ;
    private:
        std::list< std::string > mPattern;
        cTplValGesInit< int > mNbMin;
        cTplValGesInit< int > mNbMax;
};
cElXMLTree * ToXMLTree(const cFileRequired &);

void  BinaryDumpInFile(ELISE_fp &,const cFileRequired &);

void  BinaryUnDumpFromFile(cFileRequired &,ELISE_fp &);

std::string  Mangling( cFileRequired *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBatchRequirement
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBatchRequirement & anObj,cElXMLTree * aTree);


        std::list< cExeRequired > & ExeRequired();
        const std::list< cExeRequired > & ExeRequired()const ;

        std::list< cFileRequired > & FileRequired();
        const std::list< cFileRequired > & FileRequired()const ;
    private:
        std::list< cExeRequired > mExeRequired;
        std::list< cFileRequired > mFileRequired;
};
cElXMLTree * ToXMLTree(const cBatchRequirement &);

void  BinaryDumpInFile(ELISE_fp &,const cBatchRequirement &);

void  BinaryUnDumpFromFile(cBatchRequirement &,ELISE_fp &);

std::string  Mangling( cBatchRequirement *);

/******************************************************/
/******************************************************/
/******************************************************/
class cExportApero2MM
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportApero2MM & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt3dr > & DirVertLoc();
        const cTplValGesInit< Pt3dr > & DirVertLoc()const ;

        cTplValGesInit< double > & ProfInVertLoc();
        const cTplValGesInit< double > & ProfInVertLoc()const ;
    private:
        cTplValGesInit< Pt3dr > mDirVertLoc;
        cTplValGesInit< double > mProfInVertLoc;
};
cElXMLTree * ToXMLTree(const cExportApero2MM &);

void  BinaryDumpInFile(ELISE_fp &,const cExportApero2MM &);

void  BinaryUnDumpFromFile(cExportApero2MM &,ELISE_fp &);

std::string  Mangling( cExportApero2MM *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlXifInfo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlXifInfo & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & HGRev();
        const cTplValGesInit< int > & HGRev()const ;

        cTplValGesInit< std::string > & GITRev();
        const cTplValGesInit< std::string > & GITRev()const ;

        cTplValGesInit< double > & FocMM();
        const cTplValGesInit< double > & FocMM()const ;

        cTplValGesInit< double > & Foc35();
        const cTplValGesInit< double > & Foc35()const ;

        cTplValGesInit< double > & ExpTime();
        const cTplValGesInit< double > & ExpTime()const ;

        cTplValGesInit< double > & Diaph();
        const cTplValGesInit< double > & Diaph()const ;

        cTplValGesInit< double > & IsoSpeed();
        const cTplValGesInit< double > & IsoSpeed()const ;

        cTplValGesInit< Pt2di > & Sz();
        const cTplValGesInit< Pt2di > & Sz()const ;

        cTplValGesInit< double > & GPSLat();
        const cTplValGesInit< double > & GPSLat()const ;

        cTplValGesInit< double > & GPSLon();
        const cTplValGesInit< double > & GPSLon()const ;

        cTplValGesInit< double > & GPSAlt();
        const cTplValGesInit< double > & GPSAlt()const ;

        cTplValGesInit< std::string > & Cam();
        const cTplValGesInit< std::string > & Cam()const ;

        cTplValGesInit< std::string > & BayPat();
        const cTplValGesInit< std::string > & BayPat()const ;

        cTplValGesInit< cXmlDate > & Date();
        const cTplValGesInit< cXmlDate > & Date()const ;

        cTplValGesInit< std::string > & Orientation();
        const cTplValGesInit< std::string > & Orientation()const ;

        cTplValGesInit< std::string > & CameraOrientation();
        const cTplValGesInit< std::string > & CameraOrientation()const ;

        cTplValGesInit< int > & NbBits();
        const cTplValGesInit< int > & NbBits()const ;
    private:
        cTplValGesInit< int > mHGRev;
        cTplValGesInit< std::string > mGITRev;
        cTplValGesInit< double > mFocMM;
        cTplValGesInit< double > mFoc35;
        cTplValGesInit< double > mExpTime;
        cTplValGesInit< double > mDiaph;
        cTplValGesInit< double > mIsoSpeed;
        cTplValGesInit< Pt2di > mSz;
        cTplValGesInit< double > mGPSLat;
        cTplValGesInit< double > mGPSLon;
        cTplValGesInit< double > mGPSAlt;
        cTplValGesInit< std::string > mCam;
        cTplValGesInit< std::string > mBayPat;
        cTplValGesInit< cXmlDate > mDate;
        cTplValGesInit< std::string > mOrientation;
        cTplValGesInit< std::string > mCameraOrientation;
        cTplValGesInit< int > mNbBits;
};
cElXMLTree * ToXMLTree(const cXmlXifInfo &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlXifInfo &);

void  BinaryUnDumpFromFile(cXmlXifInfo &,ELISE_fp &);

std::string  Mangling( cXmlXifInfo *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMIC_IndicAutoCorrel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMIC_IndicAutoCorrel & anObj,cElXMLTree * aTree);


        double & AutoC();
        const double & AutoC()const ;

        double & SzCalc();
        const double & SzCalc()const ;
    private:
        double mAutoC;
        double mSzCalc;
};
cElXMLTree * ToXMLTree(const cMIC_IndicAutoCorrel &);

void  BinaryDumpInFile(ELISE_fp &,const cMIC_IndicAutoCorrel &);

void  BinaryUnDumpFromFile(cMIC_IndicAutoCorrel &,ELISE_fp &);

std::string  Mangling( cMIC_IndicAutoCorrel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMTDImCalc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMTDImCalc & anObj,cElXMLTree * aTree);


        std::list< cMIC_IndicAutoCorrel > & MIC_IndicAutoCorrel();
        const std::list< cMIC_IndicAutoCorrel > & MIC_IndicAutoCorrel()const ;
    private:
        std::list< cMIC_IndicAutoCorrel > mMIC_IndicAutoCorrel;
};
cElXMLTree * ToXMLTree(const cMTDImCalc &);

void  BinaryDumpInFile(ELISE_fp &,const cMTDImCalc &);

void  BinaryUnDumpFromFile(cMTDImCalc &,ELISE_fp &);

std::string  Mangling( cMTDImCalc *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCameraEntry
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCameraEntry & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        Pt2dr & SzCaptMm();
        const Pt2dr & SzCaptMm()const ;

        std::string & ShortName();
        const std::string & ShortName()const ;

        cTplValGesInit< bool > & BayerSwapRB();
        const cTplValGesInit< bool > & BayerSwapRB()const ;

        cTplValGesInit< bool > & DevRawBasic();
        const cTplValGesInit< bool > & DevRawBasic()const ;
    private:
        std::string mName;
        Pt2dr mSzCaptMm;
        std::string mShortName;
        cTplValGesInit< bool > mBayerSwapRB;
        cTplValGesInit< bool > mDevRawBasic;
};
cElXMLTree * ToXMLTree(const cCameraEntry &);

void  BinaryDumpInFile(ELISE_fp &,const cCameraEntry &);

void  BinaryUnDumpFromFile(cCameraEntry &,ELISE_fp &);

std::string  Mangling( cCameraEntry *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMMCameraDataBase
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMMCameraDataBase & anObj,cElXMLTree * aTree);


        std::list< cCameraEntry > & CameraEntry();
        const std::list< cCameraEntry > & CameraEntry()const ;
    private:
        std::list< cCameraEntry > mCameraEntry;
};
cElXMLTree * ToXMLTree(const cMMCameraDataBase &);

void  BinaryDumpInFile(ELISE_fp &,const cMMCameraDataBase &);

void  BinaryUnDumpFromFile(cMMCameraDataBase &,ELISE_fp &);

std::string  Mangling( cMMCameraDataBase *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMakeDataBase
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMakeDataBase & anObj,cElXMLTree * aTree);


        std::string & KeySetCollectXif();
        const std::string & KeySetCollectXif()const ;

        std::list< std::string > & KeyAssocNameSup();
        const std::list< std::string > & KeyAssocNameSup()const ;

        cTplValGesInit< std::string > & NameFile();
        const cTplValGesInit< std::string > & NameFile()const ;
    private:
        std::string mKeySetCollectXif;
        std::list< std::string > mKeyAssocNameSup;
        cTplValGesInit< std::string > mNameFile;
};
cElXMLTree * ToXMLTree(const cMakeDataBase &);

void  BinaryDumpInFile(ELISE_fp &,const cMakeDataBase &);

void  BinaryUnDumpFromFile(cMakeDataBase &,ELISE_fp &);

std::string  Mangling( cMakeDataBase *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBatchChantDesc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBatchChantDesc & anObj,cElXMLTree * aTree);


        cTplValGesInit< cBatchRequirement > & Requirement();
        const cTplValGesInit< cBatchRequirement > & Requirement()const ;

        std::string & Key();
        const std::string & Key()const ;

        std::list< std::string > & Line();
        const std::list< std::string > & Line()const ;
    private:
        cTplValGesInit< cBatchRequirement > mRequirement;
        std::string mKey;
        std::list< std::string > mLine;
};
cElXMLTree * ToXMLTree(const cBatchChantDesc &);

void  BinaryDumpInFile(ELISE_fp &,const cBatchChantDesc &);

void  BinaryUnDumpFromFile(cBatchChantDesc &,ELISE_fp &);

std::string  Mangling( cBatchChantDesc *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneShowChantDesc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneShowChantDesc & anObj,cElXMLTree * aTree);


        std::list< std::string > & LineBefore();
        const std::list< std::string > & LineBefore()const ;

        cTplValGesInit< bool > & ShowKeys();
        const cTplValGesInit< bool > & ShowKeys()const ;

        std::list< std::string > & KeyRels();
        const std::list< std::string > & KeyRels()const ;

        std::list< std::string > & KeySets();
        const std::list< std::string > & KeySets()const ;

        std::list< std::string > & LineAfter();
        const std::list< std::string > & LineAfter()const ;
    private:
        std::list< std::string > mLineBefore;
        cTplValGesInit< bool > mShowKeys;
        std::list< std::string > mKeyRels;
        std::list< std::string > mKeySets;
        std::list< std::string > mLineAfter;
};
cElXMLTree * ToXMLTree(const cOneShowChantDesc &);

void  BinaryDumpInFile(ELISE_fp &,const cOneShowChantDesc &);

void  BinaryUnDumpFromFile(cOneShowChantDesc &,ELISE_fp &);

std::string  Mangling( cOneShowChantDesc *);

class cShowChantDesc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cShowChantDesc & anObj,cElXMLTree * aTree);


        std::list< cOneShowChantDesc > & OneShowChantDesc();
        const std::list< cOneShowChantDesc > & OneShowChantDesc()const ;

        std::string & File();
        const std::string & File()const ;
    private:
        std::list< cOneShowChantDesc > mOneShowChantDesc;
        std::string mFile;
};
cElXMLTree * ToXMLTree(const cShowChantDesc &);

void  BinaryDumpInFile(ELISE_fp &,const cShowChantDesc &);

void  BinaryUnDumpFromFile(cShowChantDesc &,ELISE_fp &);

std::string  Mangling( cShowChantDesc *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMatrixSplitBox
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMatrixSplitBox & anObj,cElXMLTree * aTree);


        std::string & KeyMatr();
        const std::string & KeyMatr()const ;

        cTplValGesInit< double > & Rab();
        const cTplValGesInit< double > & Rab()const ;
    private:
        std::string mKeyMatr;
        cTplValGesInit< double > mRab;
};
cElXMLTree * ToXMLTree(const cMatrixSplitBox &);

void  BinaryDumpInFile(ELISE_fp &,const cMatrixSplitBox &);

void  BinaryUnDumpFromFile(cMatrixSplitBox &,ELISE_fp &);

std::string  Mangling( cMatrixSplitBox *);

class cContenuAPrioriImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContenuAPrioriImage & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyAutoAdaptScale();
        const cTplValGesInit< std::string > & KeyAutoAdaptScale()const ;

        cTplValGesInit< double > & PdsMaxAdaptScale();
        const cTplValGesInit< double > & PdsMaxAdaptScale()const ;

        cTplValGesInit< double > & Scale();
        const cTplValGesInit< double > & Scale()const ;

        cTplValGesInit< double > & Teta();
        const cTplValGesInit< double > & Teta()const ;

        cTplValGesInit< Box2di > & BoiteEnglob();
        const cTplValGesInit< Box2di > & BoiteEnglob()const ;

        cTplValGesInit< std::string > & ElInt_CaPImAddedSet();
        const cTplValGesInit< std::string > & ElInt_CaPImAddedSet()const ;

        cTplValGesInit< std::string > & ElInt_CaPImMyKey();
        const cTplValGesInit< std::string > & ElInt_CaPImMyKey()const ;

        std::string & KeyMatr();
        const std::string & KeyMatr()const ;

        cTplValGesInit< double > & Rab();
        const cTplValGesInit< double > & Rab()const ;

        cTplValGesInit< cMatrixSplitBox > & MatrixSplitBox();
        const cTplValGesInit< cMatrixSplitBox > & MatrixSplitBox()const ;
    private:
        cTplValGesInit< std::string > mKeyAutoAdaptScale;
        cTplValGesInit< double > mPdsMaxAdaptScale;
        cTplValGesInit< double > mScale;
        cTplValGesInit< double > mTeta;
        cTplValGesInit< Box2di > mBoiteEnglob;
        cTplValGesInit< std::string > mElInt_CaPImAddedSet;
        cTplValGesInit< std::string > mElInt_CaPImMyKey;
        cTplValGesInit< cMatrixSplitBox > mMatrixSplitBox;
};
cElXMLTree * ToXMLTree(const cContenuAPrioriImage &);

void  BinaryDumpInFile(ELISE_fp &,const cContenuAPrioriImage &);

void  BinaryUnDumpFromFile(cContenuAPrioriImage &,ELISE_fp &);

std::string  Mangling( cContenuAPrioriImage *);

class cAPrioriImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAPrioriImage & anObj,cElXMLTree * aTree);


        std::list< std::string > & Names();
        const std::list< std::string > & Names()const ;

        cTplValGesInit< std::string > & KeyedAddedSet();
        const cTplValGesInit< std::string > & KeyedAddedSet()const ;

        cTplValGesInit< std::string > & Key();
        const cTplValGesInit< std::string > & Key()const ;

        cTplValGesInit< std::string > & KeyAutoAdaptScale();
        const cTplValGesInit< std::string > & KeyAutoAdaptScale()const ;

        cTplValGesInit< double > & PdsMaxAdaptScale();
        const cTplValGesInit< double > & PdsMaxAdaptScale()const ;

        cTplValGesInit< double > & Scale();
        const cTplValGesInit< double > & Scale()const ;

        cTplValGesInit< double > & Teta();
        const cTplValGesInit< double > & Teta()const ;

        cTplValGesInit< Box2di > & BoiteEnglob();
        const cTplValGesInit< Box2di > & BoiteEnglob()const ;

        cTplValGesInit< std::string > & ElInt_CaPImAddedSet();
        const cTplValGesInit< std::string > & ElInt_CaPImAddedSet()const ;

        cTplValGesInit< std::string > & ElInt_CaPImMyKey();
        const cTplValGesInit< std::string > & ElInt_CaPImMyKey()const ;

        std::string & KeyMatr();
        const std::string & KeyMatr()const ;

        cTplValGesInit< double > & Rab();
        const cTplValGesInit< double > & Rab()const ;

        cTplValGesInit< cMatrixSplitBox > & MatrixSplitBox();
        const cTplValGesInit< cMatrixSplitBox > & MatrixSplitBox()const ;

        cContenuAPrioriImage & ContenuAPrioriImage();
        const cContenuAPrioriImage & ContenuAPrioriImage()const ;
    private:
        std::list< std::string > mNames;
        cTplValGesInit< std::string > mKeyedAddedSet;
        cTplValGesInit< std::string > mKey;
        cContenuAPrioriImage mContenuAPrioriImage;
};
cElXMLTree * ToXMLTree(const cAPrioriImage &);

void  BinaryDumpInFile(ELISE_fp &,const cAPrioriImage &);

void  BinaryUnDumpFromFile(cAPrioriImage &,ELISE_fp &);

std::string  Mangling( cAPrioriImage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cKeyedNamesAssociations
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cKeyedNamesAssociations & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & IsParametrized();
        const cTplValGesInit< bool > & IsParametrized()const ;

        std::list< cAssocNameToName > & Calcs();
        const std::list< cAssocNameToName > & Calcs()const ;

        std::string & Key();
        const std::string & Key()const ;

        cTplValGesInit< std::string > & SubDirAutoMake();
        const cTplValGesInit< std::string > & SubDirAutoMake()const ;

        cTplValGesInit< bool > & SubDirAutoMakeRec();
        const cTplValGesInit< bool > & SubDirAutoMakeRec()const ;
    private:
        cTplValGesInit< bool > mIsParametrized;
        std::list< cAssocNameToName > mCalcs;
        std::string mKey;
        cTplValGesInit< std::string > mSubDirAutoMake;
        cTplValGesInit< bool > mSubDirAutoMakeRec;
};
cElXMLTree * ToXMLTree(const cKeyedNamesAssociations &);

void  BinaryDumpInFile(ELISE_fp &,const cKeyedNamesAssociations &);

void  BinaryUnDumpFromFile(cKeyedNamesAssociations &,ELISE_fp &);

std::string  Mangling( cKeyedNamesAssociations *);

/******************************************************/
/******************************************************/
/******************************************************/
class cKeyedSetsOfNames
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cKeyedSetsOfNames & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & IsParametrized();
        const cTplValGesInit< bool > & IsParametrized()const ;

        cSetNameDescriptor & Sets();
        const cSetNameDescriptor & Sets()const ;

        std::string & Key();
        const std::string & Key()const ;
    private:
        cTplValGesInit< bool > mIsParametrized;
        cSetNameDescriptor mSets;
        std::string mKey;
};
cElXMLTree * ToXMLTree(const cKeyedSetsOfNames &);

void  BinaryDumpInFile(ELISE_fp &,const cKeyedSetsOfNames &);

void  BinaryUnDumpFromFile(cKeyedSetsOfNames &,ELISE_fp &);

std::string  Mangling( cKeyedSetsOfNames *);

/******************************************************/
/******************************************************/
/******************************************************/
class cKeyedSetsORels
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cKeyedSetsORels & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & IsParametrized();
        const cTplValGesInit< bool > & IsParametrized()const ;

        cNameRelDescriptor & Sets();
        const cNameRelDescriptor & Sets()const ;

        std::string & Key();
        const std::string & Key()const ;
    private:
        cTplValGesInit< bool > mIsParametrized;
        cNameRelDescriptor mSets;
        std::string mKey;
};
cElXMLTree * ToXMLTree(const cKeyedSetsORels &);

void  BinaryDumpInFile(ELISE_fp &,const cKeyedSetsORels &);

void  BinaryUnDumpFromFile(cKeyedSetsORels &,ELISE_fp &);

std::string  Mangling( cKeyedSetsORels *);

/******************************************************/
/******************************************************/
/******************************************************/
class cKeyedMatrixStruct
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cKeyedMatrixStruct & anObj,cElXMLTree * aTree);


        cImMatrixStructuration & Matrix();
        const cImMatrixStructuration & Matrix()const ;

        std::string & Key();
        const std::string & Key()const ;
    private:
        cImMatrixStructuration mMatrix;
        std::string mKey;
};
cElXMLTree * ToXMLTree(const cKeyedMatrixStruct &);

void  BinaryDumpInFile(ELISE_fp &,const cKeyedMatrixStruct &);

void  BinaryUnDumpFromFile(cKeyedMatrixStruct &,ELISE_fp &);

std::string  Mangling( cKeyedMatrixStruct *);

/******************************************************/
/******************************************************/
/******************************************************/
class cChantierDescripteur
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChantierDescripteur & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & ExitOnBrkp();
        const cTplValGesInit< bool > & ExitOnBrkp()const ;

        std::list< std::string > & Symb();
        const std::list< std::string > & Symb()const ;

        std::list< std::string > & eSymb();
        const std::list< std::string > & eSymb()const ;

        cTplValGesInit< cMMCameraDataBase > & LocCamDataBase();
        const cTplValGesInit< cMMCameraDataBase > & LocCamDataBase()const ;

        std::string & KeySetCollectXif();
        const std::string & KeySetCollectXif()const ;

        std::list< std::string > & KeyAssocNameSup();
        const std::list< std::string > & KeyAssocNameSup()const ;

        cTplValGesInit< std::string > & NameFile();
        const cTplValGesInit< std::string > & NameFile()const ;

        cTplValGesInit< cMakeDataBase > & MakeDataBase();
        const cTplValGesInit< cMakeDataBase > & MakeDataBase()const ;

        cTplValGesInit< std::string > & KeySuprAbs2Rel();
        const cTplValGesInit< std::string > & KeySuprAbs2Rel()const ;

        std::list< cBatchChantDesc > & BatchChantDesc();
        const std::list< cBatchChantDesc > & BatchChantDesc()const ;

        std::list< cShowChantDesc > & ShowChantDesc();
        const std::list< cShowChantDesc > & ShowChantDesc()const ;

        std::list< cAPrioriImage > & APrioriImage();
        const std::list< cAPrioriImage > & APrioriImage()const ;

        std::list< cKeyedNamesAssociations > & KeyedNamesAssociations();
        const std::list< cKeyedNamesAssociations > & KeyedNamesAssociations()const ;

        std::list< cKeyedSetsOfNames > & KeyedSetsOfNames();
        const std::list< cKeyedSetsOfNames > & KeyedSetsOfNames()const ;

        std::list< cKeyedSetsORels > & KeyedSetsORels();
        const std::list< cKeyedSetsORels > & KeyedSetsORels()const ;

        std::list< cKeyedMatrixStruct > & KeyedMatrixStruct();
        const std::list< cKeyedMatrixStruct > & KeyedMatrixStruct()const ;

        std::list< cClassEquivDescripteur > & KeyedClassEquiv();
        const std::list< cClassEquivDescripteur > & KeyedClassEquiv()const ;

        cTplValGesInit< cBaseDataCD > & BaseDatas();
        const cTplValGesInit< cBaseDataCD > & BaseDatas()const ;

        std::list< std::string > & FilesDatas();
        const std::list< std::string > & FilesDatas()const ;
    private:
        cTplValGesInit< bool > mExitOnBrkp;
        std::list< std::string > mSymb;
        std::list< std::string > meSymb;
        cTplValGesInit< cMMCameraDataBase > mLocCamDataBase;
        cTplValGesInit< cMakeDataBase > mMakeDataBase;
        cTplValGesInit< std::string > mKeySuprAbs2Rel;
        std::list< cBatchChantDesc > mBatchChantDesc;
        std::list< cShowChantDesc > mShowChantDesc;
        std::list< cAPrioriImage > mAPrioriImage;
        std::list< cKeyedNamesAssociations > mKeyedNamesAssociations;
        std::list< cKeyedSetsOfNames > mKeyedSetsOfNames;
        std::list< cKeyedSetsORels > mKeyedSetsORels;
        std::list< cKeyedMatrixStruct > mKeyedMatrixStruct;
        std::list< cClassEquivDescripteur > mKeyedClassEquiv;
        cTplValGesInit< cBaseDataCD > mBaseDatas;
        std::list< std::string > mFilesDatas;
};
cElXMLTree * ToXMLTree(const cChantierDescripteur &);

void  BinaryDumpInFile(ELISE_fp &,const cChantierDescripteur &);

void  BinaryUnDumpFromFile(cChantierDescripteur &,ELISE_fp &);

std::string  Mangling( cChantierDescripteur *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXML_Date
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXML_Date & anObj,cElXMLTree * aTree);


        int & year();
        const int & year()const ;

        int & month();
        const int & month()const ;

        int & day();
        const int & day()const ;

        int & hour();
        const int & hour()const ;

        int & minute();
        const int & minute()const ;

        int & second();
        const int & second()const ;

        std::string & time_system();
        const std::string & time_system()const ;
    private:
        int myear;
        int mmonth;
        int mday;
        int mhour;
        int mminute;
        int msecond;
        std::string mtime_system;
};
cElXMLTree * ToXMLTree(const cXML_Date &);

void  BinaryDumpInFile(ELISE_fp &,const cXML_Date &);

void  BinaryUnDumpFromFile(cXML_Date &,ELISE_fp &);

std::string  Mangling( cXML_Date *);

/******************************************************/
/******************************************************/
/******************************************************/
class cpt3d
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cpt3d & anObj,cElXMLTree * aTree);


        double & x();
        const double & x()const ;

        double & y();
        const double & y()const ;

        double & z();
        const double & z()const ;
    private:
        double mx;
        double my;
        double mz;
};
cElXMLTree * ToXMLTree(const cpt3d &);

void  BinaryDumpInFile(ELISE_fp &,const cpt3d &);

void  BinaryUnDumpFromFile(cpt3d &,ELISE_fp &);

std::string  Mangling( cpt3d *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXML_LinePt3d
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXML_LinePt3d & anObj,cElXMLTree * aTree);


        double & x();
        const double & x()const ;

        double & y();
        const double & y()const ;

        double & z();
        const double & z()const ;

        cpt3d & pt3d();
        const cpt3d & pt3d()const ;
    private:
        cpt3d mpt3d;
};
cElXMLTree * ToXMLTree(const cXML_LinePt3d &);

void  BinaryDumpInFile(ELISE_fp &,const cXML_LinePt3d &);

void  BinaryUnDumpFromFile(cXML_LinePt3d &,ELISE_fp &);

std::string  Mangling( cXML_LinePt3d *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneSolImageSec
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneSolImageSec & anObj,cElXMLTree * aTree);


        std::list< std::string > & Images();
        const std::list< std::string > & Images()const ;

        double & Coverage();
        const double & Coverage()const ;

        double & Score();
        const double & Score()const ;
    private:
        std::list< std::string > mImages;
        double mCoverage;
        double mScore;
};
cElXMLTree * ToXMLTree(const cOneSolImageSec &);

void  BinaryDumpInFile(ELISE_fp &,const cOneSolImageSec &);

void  BinaryUnDumpFromFile(cOneSolImageSec &,ELISE_fp &);

std::string  Mangling( cOneSolImageSec *);

/******************************************************/
/******************************************************/
/******************************************************/
class cISOM_Vois
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cISOM_Vois & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        double & Angle();
        const double & Angle()const ;

        double & Nb();
        const double & Nb()const ;

        cTplValGesInit< double > & RatioVis();
        const cTplValGesInit< double > & RatioVis()const ;
    private:
        std::string mName;
        double mAngle;
        double mNb;
        cTplValGesInit< double > mRatioVis;
};
cElXMLTree * ToXMLTree(const cISOM_Vois &);

void  BinaryDumpInFile(ELISE_fp &,const cISOM_Vois &);

void  BinaryUnDumpFromFile(cISOM_Vois &,ELISE_fp &);

std::string  Mangling( cISOM_Vois *);

class cISOM_AllVois
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cISOM_AllVois & anObj,cElXMLTree * aTree);


        std::list< cISOM_Vois > & ISOM_Vois();
        const std::list< cISOM_Vois > & ISOM_Vois()const ;
    private:
        std::list< cISOM_Vois > mISOM_Vois;
};
cElXMLTree * ToXMLTree(const cISOM_AllVois &);

void  BinaryDumpInFile(ELISE_fp &,const cISOM_AllVois &);

void  BinaryUnDumpFromFile(cISOM_AllVois &,ELISE_fp &);

std::string  Mangling( cISOM_AllVois *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImSecOfMaster
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImSecOfMaster & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & UsedPenal();
        const cTplValGesInit< double > & UsedPenal()const ;

        std::string & Master();
        const std::string & Master()const ;

        std::list< cOneSolImageSec > & Sols();
        const std::list< cOneSolImageSec > & Sols()const ;

        cTplValGesInit< cISOM_AllVois > & ISOM_AllVois();
        const cTplValGesInit< cISOM_AllVois > & ISOM_AllVois()const ;
    private:
        cTplValGesInit< double > mUsedPenal;
        std::string mMaster;
        std::list< cOneSolImageSec > mSols;
        cTplValGesInit< cISOM_AllVois > mISOM_AllVois;
};
cElXMLTree * ToXMLTree(const cImSecOfMaster &);

void  BinaryDumpInFile(ELISE_fp &,const cImSecOfMaster &);

void  BinaryUnDumpFromFile(cImSecOfMaster &,ELISE_fp &);

std::string  Mangling( cImSecOfMaster *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamOrientSHC
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamOrientSHC & anObj,cElXMLTree * aTree);


        std::string & IdGrp();
        const std::string & IdGrp()const ;

        Pt3dr & Vecteur();
        const Pt3dr & Vecteur()const ;

        cTypeCodageMatr & Rot();
        const cTypeCodageMatr & Rot()const ;
    private:
        std::string mIdGrp;
        Pt3dr mVecteur;
        cTypeCodageMatr mRot;
};
cElXMLTree * ToXMLTree(const cParamOrientSHC &);

void  BinaryDumpInFile(ELISE_fp &,const cParamOrientSHC &);

void  BinaryUnDumpFromFile(cParamOrientSHC &,ELISE_fp &);

std::string  Mangling( cParamOrientSHC *);

class cLiaisonsSHC
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLiaisonsSHC & anObj,cElXMLTree * aTree);


        std::list< cParamOrientSHC > & ParamOrientSHC();
        const std::list< cParamOrientSHC > & ParamOrientSHC()const ;
    private:
        std::list< cParamOrientSHC > mParamOrientSHC;
};
cElXMLTree * ToXMLTree(const cLiaisonsSHC &);

void  BinaryDumpInFile(ELISE_fp &,const cLiaisonsSHC &);

void  BinaryUnDumpFromFile(cLiaisonsSHC &,ELISE_fp &);

std::string  Mangling( cLiaisonsSHC *);

/******************************************************/
/******************************************************/
/******************************************************/
class cStructBlockCam
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cStructBlockCam & anObj,cElXMLTree * aTree);


        std::string & KeyIm2TimeCam();
        const std::string & KeyIm2TimeCam()const ;

        cTplValGesInit< std::string > & MasterGrp();
        const cTplValGesInit< std::string > & MasterGrp()const ;

        std::list< cParamOrientSHC > & ParamOrientSHC();
        const std::list< cParamOrientSHC > & ParamOrientSHC()const ;

        cTplValGesInit< cLiaisonsSHC > & LiaisonsSHC();
        const cTplValGesInit< cLiaisonsSHC > & LiaisonsSHC()const ;
    private:
        std::string mKeyIm2TimeCam;
        cTplValGesInit< std::string > mMasterGrp;
        cTplValGesInit< cLiaisonsSHC > mLiaisonsSHC;
};
cElXMLTree * ToXMLTree(const cStructBlockCam &);

void  BinaryDumpInFile(ELISE_fp &,const cStructBlockCam &);

void  BinaryUnDumpFromFile(cStructBlockCam &,ELISE_fp &);

std::string  Mangling( cStructBlockCam *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlExivEntry
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlExivEntry & anObj,cElXMLTree * aTree);


        std::list< std::string > & Names();
        const std::list< std::string > & Names()const ;

        double & Focale();
        const double & Focale()const ;
    private:
        std::list< std::string > mNames;
        double mFocale;
};
cElXMLTree * ToXMLTree(const cXmlExivEntry &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlExivEntry &);

void  BinaryUnDumpFromFile(cXmlExivEntry &,ELISE_fp &);

std::string  Mangling( cXmlExivEntry *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlDataBase
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlDataBase & anObj,cElXMLTree * aTree);


        int & MajNumVers();
        const int & MajNumVers()const ;

        int & MinNumVers();
        const int & MinNumVers()const ;

        std::list< cXmlExivEntry > & Exiv();
        const std::list< cXmlExivEntry > & Exiv()const ;
    private:
        int mMajNumVers;
        int mMinNumVers;
        std::list< cXmlExivEntry > mExiv;
};
cElXMLTree * ToXMLTree(const cXmlDataBase &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlDataBase &);

void  BinaryUnDumpFromFile(cXmlDataBase &,ELISE_fp &);

std::string  Mangling( cXmlDataBase *);

/******************************************************/
/******************************************************/
/******************************************************/
class cListImByDelta
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cListImByDelta & anObj,cElXMLTree * aTree);


        std::string & KeySplitName();
        const std::string & KeySplitName()const ;

        std::list< int > & Delta();
        const std::list< int > & Delta()const ;
    private:
        std::string mKeySplitName;
        std::list< int > mDelta;
};
cElXMLTree * ToXMLTree(const cListImByDelta &);

void  BinaryDumpInFile(ELISE_fp &,const cListImByDelta &);

void  BinaryUnDumpFromFile(cListImByDelta &,ELISE_fp &);

std::string  Mangling( cListImByDelta *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMMUserEnvironment
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMMUserEnvironment & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & TiePDetect();
        const cTplValGesInit< std::string > & TiePDetect()const ;

        cTplValGesInit< std::string > & TiePMatch();
        const cTplValGesInit< std::string > & TiePMatch()const ;

        cTplValGesInit< std::string > & UserName();
        const cTplValGesInit< std::string > & UserName()const ;

        cTplValGesInit< int > & NbMaxProc();
        const cTplValGesInit< int > & NbMaxProc()const ;

        cTplValGesInit< bool > & UseSeparateDirectories();
        const cTplValGesInit< bool > & UseSeparateDirectories()const ;

        cTplValGesInit< std::string > & OutputDirectory();
        const cTplValGesInit< std::string > & OutputDirectory()const ;

        cTplValGesInit< std::string > & LogDirectory();
        const cTplValGesInit< std::string > & LogDirectory()const ;

        cTplValGesInit< int > & VersionNameCam();
        const cTplValGesInit< int > & VersionNameCam()const ;
    private:
        cTplValGesInit< std::string > mTiePDetect;
        cTplValGesInit< std::string > mTiePMatch;
        cTplValGesInit< std::string > mUserName;
        cTplValGesInit< int > mNbMaxProc;
        cTplValGesInit< bool > mUseSeparateDirectories;
        cTplValGesInit< std::string > mOutputDirectory;
        cTplValGesInit< std::string > mLogDirectory;
        cTplValGesInit< int > mVersionNameCam;
};
cElXMLTree * ToXMLTree(const cMMUserEnvironment &);

void  BinaryDumpInFile(ELISE_fp &,const cMMUserEnvironment &);

void  BinaryUnDumpFromFile(cMMUserEnvironment &,ELISE_fp &);

std::string  Mangling( cMMUserEnvironment *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMTDCoher
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMTDCoher & anObj,cElXMLTree * aTree);


        Pt2di & Dec2();
        const Pt2di & Dec2()const ;
    private:
        Pt2di mDec2;
};
cElXMLTree * ToXMLTree(const cMTDCoher &);

void  BinaryDumpInFile(ELISE_fp &,const cMTDCoher &);

void  BinaryUnDumpFromFile(cMTDCoher &,ELISE_fp &);

std::string  Mangling( cMTDCoher *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_sommet
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_sommet & anObj,cElXMLTree * aTree);


        double & easting();
        const double & easting()const ;

        double & northing();
        const double & northing()const ;

        double & altitude();
        const double & altitude()const ;
    private:
        double measting;
        double mnorthing;
        double maltitude;
};
cElXMLTree * ToXMLTree(const cXmlMatis_sommet &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_sommet &);

void  BinaryUnDumpFromFile(cXmlMatis_sommet &,ELISE_fp &);

std::string  Mangling( cXmlMatis_sommet *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_pt3d
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_pt3d & anObj,cElXMLTree * aTree);


        double & x();
        const double & x()const ;

        double & y();
        const double & y()const ;

        double & z();
        const double & z()const ;
    private:
        double mx;
        double my;
        double mz;
};
cElXMLTree * ToXMLTree(const cXmlMatis_pt3d &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_pt3d &);

void  BinaryUnDumpFromFile(cXmlMatis_pt3d &,ELISE_fp &);

std::string  Mangling( cXmlMatis_pt3d *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_FormeLin
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_FormeLin & anObj,cElXMLTree * aTree);


        cXmlMatis_pt3d & pt3d();
        const cXmlMatis_pt3d & pt3d()const ;
    private:
        cXmlMatis_pt3d mpt3d;
};
cElXMLTree * ToXMLTree(const cXmlMatis_FormeLin &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_FormeLin &);

void  BinaryUnDumpFromFile(cXmlMatis_FormeLin &,ELISE_fp &);

std::string  Mangling( cXmlMatis_FormeLin *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_mat3d
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_mat3d & anObj,cElXMLTree * aTree);


        cXmlMatis_FormeLin & l1();
        const cXmlMatis_FormeLin & l1()const ;

        cXmlMatis_FormeLin & l2();
        const cXmlMatis_FormeLin & l2()const ;

        cXmlMatis_FormeLin & l3();
        const cXmlMatis_FormeLin & l3()const ;
    private:
        cXmlMatis_FormeLin ml1;
        cXmlMatis_FormeLin ml2;
        cXmlMatis_FormeLin ml3;
};
cElXMLTree * ToXMLTree(const cXmlMatis_mat3d &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_mat3d &);

void  BinaryUnDumpFromFile(cXmlMatis_mat3d &,ELISE_fp &);

std::string  Mangling( cXmlMatis_mat3d *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_quaternion
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_quaternion & anObj,cElXMLTree * aTree);


        double & x();
        const double & x()const ;

        double & y();
        const double & y()const ;

        double & z();
        const double & z()const ;

        double & w();
        const double & w()const ;
    private:
        double mx;
        double my;
        double mz;
        double mw;
};
cElXMLTree * ToXMLTree(const cXmlMatis_quaternion &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_quaternion &);

void  BinaryUnDumpFromFile(cXmlMatis_quaternion &,ELISE_fp &);

std::string  Mangling( cXmlMatis_quaternion *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_rotation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_rotation & anObj,cElXMLTree * aTree);


        bool & Image2Ground();
        const bool & Image2Ground()const ;

        cXmlMatis_mat3d & mat3d();
        const cXmlMatis_mat3d & mat3d()const ;

        cTplValGesInit< cXmlMatis_quaternion > & quaternion();
        const cTplValGesInit< cXmlMatis_quaternion > & quaternion()const ;
    private:
        bool mImage2Ground;
        cXmlMatis_mat3d mmat3d;
        cTplValGesInit< cXmlMatis_quaternion > mquaternion;
};
cElXMLTree * ToXMLTree(const cXmlMatis_rotation &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_rotation &);

void  BinaryUnDumpFromFile(cXmlMatis_rotation &,ELISE_fp &);

std::string  Mangling( cXmlMatis_rotation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_extrinseque
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_extrinseque & anObj,cElXMLTree * aTree);


        std::string & systeme();
        const std::string & systeme()const ;

        std::string & grid_alti();
        const std::string & grid_alti()const ;

        cXmlMatis_sommet & sommet();
        const cXmlMatis_sommet & sommet()const ;

        cXmlMatis_rotation & rotation();
        const cXmlMatis_rotation & rotation()const ;
    private:
        std::string msysteme;
        std::string mgrid_alti;
        cXmlMatis_sommet msommet;
        cXmlMatis_rotation mrotation;
};
cElXMLTree * ToXMLTree(const cXmlMatis_extrinseque &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_extrinseque &);

void  BinaryUnDumpFromFile(cXmlMatis_extrinseque &,ELISE_fp &);

std::string  Mangling( cXmlMatis_extrinseque *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_P2d_cl
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_P2d_cl & anObj,cElXMLTree * aTree);


        double & c();
        const double & c()const ;

        double & l();
        const double & l()const ;
    private:
        double mc;
        double ml;
};
cElXMLTree * ToXMLTree(const cXmlMatis_P2d_cl &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_P2d_cl &);

void  BinaryUnDumpFromFile(cXmlMatis_P2d_cl &,ELISE_fp &);

std::string  Mangling( cXmlMatis_P2d_cl *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_ppa
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_ppa & anObj,cElXMLTree * aTree);


        double & c();
        const double & c()const ;

        double & l();
        const double & l()const ;

        double & focale();
        const double & focale()const ;
    private:
        double mc;
        double ml;
        double mfocale;
};
cElXMLTree * ToXMLTree(const cXmlMatis_ppa &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_ppa &);

void  BinaryUnDumpFromFile(cXmlMatis_ppa &,ELISE_fp &);

std::string  Mangling( cXmlMatis_ppa *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_distortion
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_distortion & anObj,cElXMLTree * aTree);


        cXmlMatis_P2d_cl & pps();
        const cXmlMatis_P2d_cl & pps()const ;

        double & r3();
        const double & r3()const ;

        double & r5();
        const double & r5()const ;

        double & r7();
        const double & r7()const ;
    private:
        cXmlMatis_P2d_cl mpps;
        double mr3;
        double mr5;
        double mr7;
};
cElXMLTree * ToXMLTree(const cXmlMatis_distortion &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_distortion &);

void  BinaryUnDumpFromFile(cXmlMatis_distortion &,ELISE_fp &);

std::string  Mangling( cXmlMatis_distortion *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_image_size
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_image_size & anObj,cElXMLTree * aTree);


        int & width();
        const int & width()const ;

        int & height();
        const int & height()const ;
    private:
        int mwidth;
        int mheight;
};
cElXMLTree * ToXMLTree(const cXmlMatis_image_size &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_image_size &);

void  BinaryUnDumpFromFile(cXmlMatis_image_size &,ELISE_fp &);

std::string  Mangling( cXmlMatis_image_size *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_sensor
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_sensor & anObj,cElXMLTree * aTree);


        std::string & name();
        const std::string & name()const ;

        std::string & calibration_date();
        const std::string & calibration_date()const ;

        std::string & serial_number();
        const std::string & serial_number()const ;

        cXmlMatis_image_size & image_size();
        const cXmlMatis_image_size & image_size()const ;

        cXmlMatis_ppa & ppa();
        const cXmlMatis_ppa & ppa()const ;

        cXmlMatis_distortion & distortion();
        const cXmlMatis_distortion & distortion()const ;

        double & pixel_size();
        const double & pixel_size()const ;
    private:
        std::string mname;
        std::string mcalibration_date;
        std::string mserial_number;
        cXmlMatis_image_size mimage_size;
        cXmlMatis_ppa mppa;
        cXmlMatis_distortion mdistortion;
        double mpixel_size;
};
cElXMLTree * ToXMLTree(const cXmlMatis_sensor &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_sensor &);

void  BinaryUnDumpFromFile(cXmlMatis_sensor &,ELISE_fp &);

std::string  Mangling( cXmlMatis_sensor *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_frame
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_frame & anObj,cElXMLTree * aTree);


        double & lambda_min();
        const double & lambda_min()const ;

        double & lambda_max();
        const double & lambda_max()const ;

        double & phi_min();
        const double & phi_min()const ;

        double & phi_max();
        const double & phi_max()const ;
    private:
        double mlambda_min;
        double mlambda_max;
        double mphi_min;
        double mphi_max;
};
cElXMLTree * ToXMLTree(const cXmlMatis_frame &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_frame &);

void  BinaryUnDumpFromFile(cXmlMatis_frame &,ELISE_fp &);

std::string  Mangling( cXmlMatis_frame *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_spherique
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_spherique & anObj,cElXMLTree * aTree);


        cXmlMatis_image_size & image_size();
        const cXmlMatis_image_size & image_size()const ;

        cXmlMatis_P2d_cl & ppa();
        const cXmlMatis_P2d_cl & ppa()const ;

        cXmlMatis_frame & frame();
        const cXmlMatis_frame & frame()const ;
    private:
        cXmlMatis_image_size mimage_size;
        cXmlMatis_P2d_cl mppa;
        cXmlMatis_frame mframe;
};
cElXMLTree * ToXMLTree(const cXmlMatis_spherique &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_spherique &);

void  BinaryUnDumpFromFile(cXmlMatis_spherique &,ELISE_fp &);

std::string  Mangling( cXmlMatis_spherique *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_intrinseque
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_intrinseque & anObj,cElXMLTree * aTree);


        cTplValGesInit< cXmlMatis_sensor > & sensor();
        const cTplValGesInit< cXmlMatis_sensor > & sensor()const ;

        cTplValGesInit< cXmlMatis_spherique > & spherique();
        const cTplValGesInit< cXmlMatis_spherique > & spherique()const ;
    private:
        cTplValGesInit< cXmlMatis_sensor > msensor;
        cTplValGesInit< cXmlMatis_spherique > mspherique;
};
cElXMLTree * ToXMLTree(const cXmlMatis_intrinseque &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_intrinseque &);

void  BinaryUnDumpFromFile(cXmlMatis_intrinseque &,ELISE_fp &);

std::string  Mangling( cXmlMatis_intrinseque *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_geometry
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_geometry & anObj,cElXMLTree * aTree);


        cXmlMatis_extrinseque & extrinseque();
        const cXmlMatis_extrinseque & extrinseque()const ;

        cXmlMatis_intrinseque & intrinseque();
        const cXmlMatis_intrinseque & intrinseque()const ;
    private:
        cXmlMatis_extrinseque mextrinseque;
        cXmlMatis_intrinseque mintrinseque;
};
cElXMLTree * ToXMLTree(const cXmlMatis_geometry &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_geometry &);

void  BinaryUnDumpFromFile(cXmlMatis_geometry &,ELISE_fp &);

std::string  Mangling( cXmlMatis_geometry *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_image_date
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_image_date & anObj,cElXMLTree * aTree);


        int & year();
        const int & year()const ;

        int & month();
        const int & month()const ;

        int & day();
        const int & day()const ;

        std::string & time_system();
        const std::string & time_system()const ;

        int & hour();
        const int & hour()const ;

        int & minute();
        const int & minute()const ;

        double & second();
        const double & second()const ;
    private:
        int myear;
        int mmonth;
        int mday;
        std::string mtime_system;
        int mhour;
        int mminute;
        double msecond;
};
cElXMLTree * ToXMLTree(const cXmlMatis_image_date &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_image_date &);

void  BinaryUnDumpFromFile(cXmlMatis_image_date &,ELISE_fp &);

std::string  Mangling( cXmlMatis_image_date *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlMatis_auxiliarydata
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlMatis_auxiliarydata & anObj,cElXMLTree * aTree);


        std::string & image_name();
        const std::string & image_name()const ;

        XmlXml & stereopolis();
        const XmlXml & stereopolis()const ;

        cXmlMatis_image_date & image_date();
        const cXmlMatis_image_date & image_date()const ;
    private:
        std::string mimage_name;
        XmlXml mstereopolis;
        cXmlMatis_image_date mimage_date;
};
cElXMLTree * ToXMLTree(const cXmlMatis_auxiliarydata &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlMatis_auxiliarydata &);

void  BinaryUnDumpFromFile(cXmlMatis_auxiliarydata &,ELISE_fp &);

std::string  Mangling( cXmlMatis_auxiliarydata *);

/******************************************************/
/******************************************************/
/******************************************************/
class corientation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(corientation & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & version();
        const cTplValGesInit< std::string > & version()const ;

        cXmlMatis_auxiliarydata & auxiliarydata();
        const cXmlMatis_auxiliarydata & auxiliarydata()const ;

        cXmlMatis_geometry & geometry();
        const cXmlMatis_geometry & geometry()const ;
    private:
        cTplValGesInit< std::string > mversion;
        cXmlMatis_auxiliarydata mauxiliarydata;
        cXmlMatis_geometry mgeometry;
};
cElXMLTree * ToXMLTree(const corientation &);

void  BinaryDumpInFile(ELISE_fp &,const corientation &);

void  BinaryUnDumpFromFile(corientation &,ELISE_fp &);

std::string  Mangling( corientation *);

/******************************************************/
/******************************************************/
/******************************************************/
typedef enum
{
  eDynVinoModulo,
  eDynVinoColCirc,
  eDynVinoMaxMin,
  eDynVinoStat2,
  eDynVinoEqual,
  eDynVinoNbVals
} eTypeDynVino;
void xml_init(eTypeDynVino & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeDynVino & aVal);

eTypeDynVino  Str2eTypeDynVino(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeDynVino & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeDynVino &);

std::string  Mangling( eTypeDynVino *);

void  BinaryUnDumpFromFile(eTypeDynVino &,ELISE_fp &);

class cXml_StatVino
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_StatVino & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        eTypeDynVino & Type();
        const eTypeDynVino & Type()const ;

        bool & IsInit();
        const bool & IsInit()const ;

        double & Nb();
        const double & Nb()const ;

        std::vector< double > & Soms();
        const std::vector< double > & Soms()const ;

        std::vector< double > & Soms2();
        const std::vector< double > & Soms2()const ;

        std::vector< double > & ECT();
        const std::vector< double > & ECT()const ;

        std::vector< double > & VMax();
        const std::vector< double > & VMax()const ;

        std::vector< double > & VMin();
        const std::vector< double > & VMin()const ;

        Pt2dr & IntervDyn();
        const Pt2dr & IntervDyn()const ;

        double & MulDyn();
        const double & MulDyn()const ;

        double & VMinHisto();
        const double & VMinHisto()const ;

        double & StepHisto();
        const double & StepHisto()const ;
    private:
        std::string mNameFile;
        eTypeDynVino mType;
        bool mIsInit;
        double mNb;
        std::vector< double > mSoms;
        std::vector< double > mSoms2;
        std::vector< double > mECT;
        std::vector< double > mVMax;
        std::vector< double > mVMin;
        Pt2dr mIntervDyn;
        double mMulDyn;
        double mVMinHisto;
        double mStepHisto;
};
cElXMLTree * ToXMLTree(const cXml_StatVino &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_StatVino &);

void  BinaryUnDumpFromFile(cXml_StatVino &,ELISE_fp &);

std::string  Mangling( cXml_StatVino *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_EnvVino
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_EnvVino & anObj,cElXMLTree * aTree);


        Pt2di & SzW();
        const Pt2di & SzW()const ;

        cTplValGesInit< double > & SzLimSsEch();
        const cTplValGesInit< double > & SzLimSsEch()const ;

        int & LargAsc();
        const int & LargAsc()const ;

        Pt2di & SzIncr();
        const Pt2di & SzIncr()const ;

        bool & ZoomBilin();
        const bool & ZoomBilin()const ;

        double & SpeedZoomGrab();
        const double & SpeedZoomGrab()const ;

        double & SpeedZoomMolette();
        const double & SpeedZoomMolette()const ;

        bool & ForceGray();
        const bool & ForceGray()const ;

        int & NumCrop();
        const int & NumCrop()const ;

        std::list< cXml_StatVino > & Stats();
        const std::list< cXml_StatVino > & Stats()const ;
    private:
        Pt2di mSzW;
        cTplValGesInit< double > mSzLimSsEch;
        int mLargAsc;
        Pt2di mSzIncr;
        bool mZoomBilin;
        double mSpeedZoomGrab;
        double mSpeedZoomMolette;
        bool mForceGray;
        int mNumCrop;
        std::list< cXml_StatVino > mStats;
};
cElXMLTree * ToXMLTree(const cXml_EnvVino &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_EnvVino &);

void  BinaryUnDumpFromFile(cXml_EnvVino &,ELISE_fp &);

std::string  Mangling( cXml_EnvVino *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ParamBoxReducTieP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ParamBoxReducTieP & anObj,cElXMLTree * aTree);


        Box2dr & Box();
        const Box2dr & Box()const ;

        Box2dr & BoxRab();
        const Box2dr & BoxRab()const ;

        std::vector< std::string > & Ims();
        const std::vector< std::string > & Ims()const ;

        cTplValGesInit< std::string > & MasterIm();
        const cTplValGesInit< std::string > & MasterIm()const ;
    private:
        Box2dr mBox;
        Box2dr mBoxRab;
        std::vector< std::string > mIms;
        cTplValGesInit< std::string > mMasterIm;
};
cElXMLTree * ToXMLTree(const cXml_ParamBoxReducTieP &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ParamBoxReducTieP &);

void  BinaryUnDumpFromFile(cXml_ParamBoxReducTieP &,ELISE_fp &);

std::string  Mangling( cXml_ParamBoxReducTieP *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ResOneImReducTieP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ResOneImReducTieP & anObj,cElXMLTree * aTree);


        Box2dr & BoxIm();
        const Box2dr & BoxIm()const ;

        double & Resol();
        const double & Resol()const ;
    private:
        Box2dr mBoxIm;
        double mResol;
};
cElXMLTree * ToXMLTree(const cXml_ResOneImReducTieP &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ResOneImReducTieP &);

void  BinaryUnDumpFromFile(cXml_ResOneImReducTieP &,ELISE_fp &);

std::string  Mangling( cXml_ResOneImReducTieP *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ParamGlobReducTieP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ParamGlobReducTieP & anObj,cElXMLTree * aTree);


        double & Resol();
        const double & Resol()const ;
    private:
        double mResol;
};
cElXMLTree * ToXMLTree(const cXml_ParamGlobReducTieP &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ParamGlobReducTieP &);

void  BinaryUnDumpFromFile(cXml_ParamGlobReducTieP &,ELISE_fp &);

std::string  Mangling( cXml_ParamGlobReducTieP *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ParamSubcommandTiepRed
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ParamSubcommandTiepRed & anObj,cElXMLTree * aTree);


        int & NumInit();
        const int & NumInit()const ;

        int & NumSubcommands();
        const int & NumSubcommands()const ;

        int & MaxNumRelated();
        const int & MaxNumRelated()const ;

        std::vector< std::string > & Images();
        const std::vector< std::string > & Images()const ;
    private:
        int mNumInit;
        int mNumSubcommands;
        int mMaxNumRelated;
        std::vector< std::string > mImages;
};
cElXMLTree * ToXMLTree(const cXml_ParamSubcommandTiepRed &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ParamSubcommandTiepRed &);

void  BinaryUnDumpFromFile(cXml_ParamSubcommandTiepRed &,ELISE_fp &);

std::string  Mangling( cXml_ParamSubcommandTiepRed *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ParamBascRigide
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ParamBascRigide & anObj,cElXMLTree * aTree);


        cTypeCodageMatr & ParamRotation();
        const cTypeCodageMatr & ParamRotation()const ;

        Pt3dr & Trans();
        const Pt3dr & Trans()const ;

        double & Scale();
        const double & Scale()const ;
    private:
        cTypeCodageMatr mParamRotation;
        Pt3dr mTrans;
        double mScale;
};
cElXMLTree * ToXMLTree(const cXml_ParamBascRigide &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ParamBascRigide &);

void  BinaryUnDumpFromFile(cXml_ParamBascRigide &,ELISE_fp &);

std::string  Mangling( cXml_ParamBascRigide *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ResiduBascule
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ResiduBascule & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        Pt3dr & Offset();
        const Pt3dr & Offset()const ;

        double & Dist();
        const double & Dist()const ;
    private:
        std::string mName;
        Pt3dr mOffset;
        double mDist;
};
cElXMLTree * ToXMLTree(const cXml_ResiduBascule &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ResiduBascule &);

void  BinaryUnDumpFromFile(cXml_ResiduBascule &,ELISE_fp &);

std::string  Mangling( cXml_ResiduBascule *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_SolBascRigide
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_SolBascRigide & anObj,cElXMLTree * aTree);


        cXml_ParamBascRigide & Param();
        const cXml_ParamBascRigide & Param()const ;

        double & MoyenneDist();
        const double & MoyenneDist()const ;

        double & MoyenneDistAlti();
        const double & MoyenneDistAlti()const ;

        double & MoyenneDistPlani();
        const double & MoyenneDistPlani()const ;

        cXml_ResiduBascule & Worst();
        const cXml_ResiduBascule & Worst()const ;

        std::list< cXml_ResiduBascule > & Residus();
        const std::list< cXml_ResiduBascule > & Residus()const ;
    private:
        cXml_ParamBascRigide mParam;
        double mMoyenneDist;
        double mMoyenneDistAlti;
        double mMoyenneDistPlani;
        cXml_ResiduBascule mWorst;
        std::list< cXml_ResiduBascule > mResidus;
};
cElXMLTree * ToXMLTree(const cXml_SolBascRigide &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_SolBascRigide &);

void  BinaryUnDumpFromFile(cXml_SolBascRigide &,ELISE_fp &);

std::string  Mangling( cXml_SolBascRigide *);

/******************************************************/
/******************************************************/
/******************************************************/
typedef enum
{
  eCmf_Control,
  eCmf_Convert,
  eCmf_ImProc,
  eCmf_Interf,
  eCmf_Orient,
  eCmf_OriAbs,
  eCmf_OriSat,
  eCmf_TiePoints,
  eCmf_ImMatch,
  eCmf_Map2D,
  eCmf_TrajGnss,
  eCmf_NbVals
} eCmdMM_Feature;
void xml_init(eCmdMM_Feature & aVal,cElXMLTree * aTree);
std::string  eToString(const eCmdMM_Feature & aVal);

eCmdMM_Feature  Str2eCmdMM_Feature(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eCmdMM_Feature & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eCmdMM_Feature &);

std::string  Mangling( eCmdMM_Feature *);

void  BinaryUnDumpFromFile(eCmdMM_Feature &,ELISE_fp &);

typedef enum
{
  eCmDt_Orient,
  eCmDt_Images,
  eCmDt_TiePoints,
  eCmDt_CloudXML,
  eCmDt_FileXML,
  eCmDt_FileTxt,
  eCmDt_GCP,
  eCmDt_PtImMes,
  eCmDt_Any,
  eCmDt_None,
  eCmDt_Ply,
  eCmDt_Map2D,
  eCmDt_NbVals
} eCmdMM_DataType;
void xml_init(eCmdMM_DataType & aVal,cElXMLTree * aTree);
std::string  eToString(const eCmdMM_DataType & aVal);

eCmdMM_DataType  Str2eCmdMM_DataType(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eCmdMM_DataType & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eCmdMM_DataType &);

std::string  Mangling( eCmdMM_DataType *);

void  BinaryUnDumpFromFile(eCmdMM_DataType &,ELISE_fp &);

typedef enum
{
  eCmGrp_mm3d,
  eCmGrp_TestLib,
  eCmGrp_SateLib,
  eCmGrp_SimuLib,
  eCmGrp_XLib,
  eCmGrp_NbVals
} eCmdMM_Group;
void xml_init(eCmdMM_Group & aVal,cElXMLTree * aTree);
std::string  eToString(const eCmdMM_Group & aVal);

eCmdMM_Group  Str2eCmdMM_Group(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eCmdMM_Group & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eCmdMM_Group &);

std::string  Mangling( eCmdMM_Group *);

void  BinaryUnDumpFromFile(eCmdMM_Group &,ELISE_fp &);

class cXml_Specif1MMCmd
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_Specif1MMCmd & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        eCmdMM_Feature & MainFeature();
        const eCmdMM_Feature & MainFeature()const ;

        eCmdMM_DataType & MainInput();
        const eCmdMM_DataType & MainInput()const ;

        eCmdMM_DataType & MainOutput();
        const eCmdMM_DataType & MainOutput()const ;

        cTplValGesInit< eCmdMM_Group > & Group();
        const cTplValGesInit< eCmdMM_Group > & Group()const ;

        cTplValGesInit< std::string > & Option();
        const cTplValGesInit< std::string > & Option()const ;

        std::list< eCmdMM_Feature > & OtherFeature();
        const std::list< eCmdMM_Feature > & OtherFeature()const ;

        std::list< eCmdMM_DataType > & OtherInput();
        const std::list< eCmdMM_DataType > & OtherInput()const ;

        std::list< eCmdMM_DataType > & OtherOutput();
        const std::list< eCmdMM_DataType > & OtherOutput()const ;

        cTplValGesInit< Pt3di > & CreationDate();
        const cTplValGesInit< Pt3di > & CreationDate()const ;

        cTplValGesInit< Pt3di > & ModifDate();
        const cTplValGesInit< Pt3di > & ModifDate()const ;

        std::list< std::string > & DependOf();
        const std::list< std::string > & DependOf()const ;

        std::list< std::string > & UsedBy();
        const std::list< std::string > & UsedBy()const ;
    private:
        std::string mName;
        eCmdMM_Feature mMainFeature;
        eCmdMM_DataType mMainInput;
        eCmdMM_DataType mMainOutput;
        cTplValGesInit< eCmdMM_Group > mGroup;
        cTplValGesInit< std::string > mOption;
        std::list< eCmdMM_Feature > mOtherFeature;
        std::list< eCmdMM_DataType > mOtherInput;
        std::list< eCmdMM_DataType > mOtherOutput;
        cTplValGesInit< Pt3di > mCreationDate;
        cTplValGesInit< Pt3di > mModifDate;
        std::list< std::string > mDependOf;
        std::list< std::string > mUsedBy;
};
cElXMLTree * ToXMLTree(const cXml_Specif1MMCmd &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_Specif1MMCmd &);

void  BinaryUnDumpFromFile(cXml_Specif1MMCmd &,ELISE_fp &);

std::string  Mangling( cXml_Specif1MMCmd *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_SpecifAllMMCmd
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_SpecifAllMMCmd & anObj,cElXMLTree * aTree);


        std::list< cXml_Specif1MMCmd > & OneCmd();
        const std::list< cXml_Specif1MMCmd > & OneCmd()const ;
    private:
        std::list< cXml_Specif1MMCmd > mOneCmd;
};
cElXMLTree * ToXMLTree(const cXml_SpecifAllMMCmd &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_SpecifAllMMCmd &);

void  BinaryUnDumpFromFile(cXml_SpecifAllMMCmd &,ELISE_fp &);

std::string  Mangling( cXml_SpecifAllMMCmd *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGS_OneLinear
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGS_OneLinear & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & Period();
        const cTplValGesInit< int > & Period()const ;

        int & DeltaMin();
        const int & DeltaMin()const ;

        int & DeltaMax();
        const int & DeltaMax()const ;

        std::list< cCpleString > & CpleGrp();
        const std::list< cCpleString > & CpleGrp()const ;
    private:
        cTplValGesInit< int > mPeriod;
        int mDeltaMin;
        int mDeltaMax;
        std::list< cCpleString > mCpleGrp;
};
cElXMLTree * ToXMLTree(const cGS_OneLinear &);

void  BinaryDumpInFile(ELISE_fp &,const cGS_OneLinear &);

void  BinaryUnDumpFromFile(cGS_OneLinear &,ELISE_fp &);

std::string  Mangling( cGS_OneLinear *);

class cGS_SectionLinear
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGS_SectionLinear & anObj,cElXMLTree * aTree);


        std::list< cGS_OneLinear > & GS_OneLinear();
        const std::list< cGS_OneLinear > & GS_OneLinear()const ;
    private:
        std::list< cGS_OneLinear > mGS_OneLinear;
};
cElXMLTree * ToXMLTree(const cGS_SectionLinear &);

void  BinaryDumpInFile(ELISE_fp &,const cGS_SectionLinear &);

void  BinaryUnDumpFromFile(cGS_SectionLinear &,ELISE_fp &);

std::string  Mangling( cGS_SectionLinear *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGS_SectionCross
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGS_SectionCross & anObj,cElXMLTree * aTree);


        double & DistMax();
        const double & DistMax()const ;

        double & DistCurvMin();
        const double & DistCurvMin()const ;

        double & AngleMinSpeed();
        const double & AngleMinSpeed()const ;

        double & DistMinTraj();
        const double & DistMinTraj()const ;

        std::list< std::string > & ListCam();
        const std::list< std::string > & ListCam()const ;
    private:
        double mDistMax;
        double mDistCurvMin;
        double mAngleMinSpeed;
        double mDistMinTraj;
        std::list< std::string > mListCam;
};
cElXMLTree * ToXMLTree(const cGS_SectionCross &);

void  BinaryDumpInFile(ELISE_fp &,const cGS_SectionCross &);

void  BinaryUnDumpFromFile(cGS_SectionCross &,ELISE_fp &);

std::string  Mangling( cGS_SectionCross *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneInterv_OT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneInterv_OT & anObj,cElXMLTree * aTree);


        double & DistMax();
        const double & DistMax()const ;

        std::list< cCpleString > & CpleGrp();
        const std::list< cCpleString > & CpleGrp()const ;
    private:
        double mDistMax;
        std::list< cCpleString > mCpleGrp;
};
cElXMLTree * ToXMLTree(const cOneInterv_OT &);

void  BinaryDumpInFile(ELISE_fp &,const cOneInterv_OT &);

void  BinaryUnDumpFromFile(cOneInterv_OT &,ELISE_fp &);

std::string  Mangling( cOneInterv_OT *);

class cGS_SectionOverlapingTraj
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGS_SectionOverlapingTraj & anObj,cElXMLTree * aTree);


        double & AngleMaxSpeed();
        const double & AngleMaxSpeed()const ;

        double & DistMaxTraj();
        const double & DistMaxTraj()const ;

        std::list< cOneInterv_OT > & OneInterv_OT();
        const std::list< cOneInterv_OT > & OneInterv_OT()const ;
    private:
        double mAngleMaxSpeed;
        double mDistMaxTraj;
        std::list< cOneInterv_OT > mOneInterv_OT;
};
cElXMLTree * ToXMLTree(const cGS_SectionOverlapingTraj &);

void  BinaryDumpInFile(ELISE_fp &,const cGS_SectionOverlapingTraj &);

void  BinaryUnDumpFromFile(cGS_SectionOverlapingTraj &,ELISE_fp &);

std::string  Mangling( cGS_SectionOverlapingTraj *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ParamGraphStereopolis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ParamGraphStereopolis & anObj,cElXMLTree * aTree);


        std::string & NameGrpC();
        const std::string & NameGrpC()const ;

        std::list< cGS_OneLinear > & GS_OneLinear();
        const std::list< cGS_OneLinear > & GS_OneLinear()const ;

        cTplValGesInit< cGS_SectionLinear > & GS_SectionLinear();
        const cTplValGesInit< cGS_SectionLinear > & GS_SectionLinear()const ;

        double & DistMax();
        const double & DistMax()const ;

        double & DistCurvMin();
        const double & DistCurvMin()const ;

        double & AngleMinSpeed();
        const double & AngleMinSpeed()const ;

        double & DistMinTraj();
        const double & DistMinTraj()const ;

        std::list< std::string > & ListCam();
        const std::list< std::string > & ListCam()const ;

        cTplValGesInit< cGS_SectionCross > & GS_SectionCross();
        const cTplValGesInit< cGS_SectionCross > & GS_SectionCross()const ;

        double & AngleMaxSpeed();
        const double & AngleMaxSpeed()const ;

        double & DistMaxTraj();
        const double & DistMaxTraj()const ;

        std::list< cOneInterv_OT > & OneInterv_OT();
        const std::list< cOneInterv_OT > & OneInterv_OT()const ;

        cTplValGesInit< cGS_SectionOverlapingTraj > & GS_SectionOverlapingTraj();
        const cTplValGesInit< cGS_SectionOverlapingTraj > & GS_SectionOverlapingTraj()const ;
    private:
        std::string mNameGrpC;
        cTplValGesInit< cGS_SectionLinear > mGS_SectionLinear;
        cTplValGesInit< cGS_SectionCross > mGS_SectionCross;
        cTplValGesInit< cGS_SectionOverlapingTraj > mGS_SectionOverlapingTraj;
};
cElXMLTree * ToXMLTree(const cXml_ParamGraphStereopolis &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ParamGraphStereopolis &);

void  BinaryUnDumpFromFile(cXml_ParamGraphStereopolis &,ELISE_fp &);

std::string  Mangling( cXml_ParamGraphStereopolis *);

/******************************************************/
/******************************************************/
/******************************************************/
typedef enum
{
  eR3D,
  eR2D,
  eNbTypeRHP
} eRANSAC_HistoP;
void xml_init(eRANSAC_HistoP & aVal,cElXMLTree * aTree);
std::string  eToString(const eRANSAC_HistoP & aVal);

eRANSAC_HistoP  Str2eRANSAC_HistoP(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eRANSAC_HistoP & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eRANSAC_HistoP &);

std::string  Mangling( eRANSAC_HistoP *);

void  BinaryUnDumpFromFile(eRANSAC_HistoP &,ELISE_fp &);

typedef enum
{
  eBruteForce,
  eGuided,
  eNbTypePPHP
} eGetPatchPair_HistoP;
void xml_init(eGetPatchPair_HistoP & aVal,cElXMLTree * aTree);
std::string  eToString(const eGetPatchPair_HistoP & aVal);

eGetPatchPair_HistoP  Str2eGetPatchPair_HistoP(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eGetPatchPair_HistoP & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eGetPatchPair_HistoP &);

std::string  Mangling( eGetPatchPair_HistoP *);

void  BinaryUnDumpFromFile(eGetPatchPair_HistoP &,ELISE_fp &);

// };
#endif // Define_NotPCP
