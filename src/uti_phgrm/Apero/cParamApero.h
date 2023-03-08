#include "StdAfx.h"
#ifndef Define_NotApero
#define Define_NotApero
// #include "XML_GEN/all.h"
// NO MORE ...
typedef enum
{
  eAllParamLibres,
  eAllParamFiges,
  eLiberteParamDeg_0,
  eLiberteParamDeg_1,
  eLiberteParamDeg_2,
  eLiberteParamDeg_3,
  eLiberteParamDeg_4,
  eLiberteParamDeg_5,
  eLiberteParamDeg_6,
  eLiberteParamDeg_7,
  eLiberteParamDeg_2_NoAff,
  eLiberteParamDeg_3_NoAff,
  eLiberteParamDeg_4_NoAff,
  eLiberteParamDeg_5_NoAff,
  eLiberteFocale_0,
  eLiberteFocale_1,
  eLib_PP_CD_00,
  eLib_PP_CD_10,
  eLib_PP_CD_01,
  eLib_PP_CD_11,
  eLib_PP_CD_Lies,
  eLiberte_DR0,
  eLiberte_DR1,
  eLiberte_DR2,
  eLiberte_DR3,
  eLiberte_DR4,
  eLiberte_DR5,
  eLiberte_DR6,
  eLiberte_DR7,
  eLiberte_DR8,
  eLiberte_DR9,
  eLiberte_DR10,
  eLiberte_Dec0,
  eLiberte_Dec1,
  eLiberte_Dec2,
  eLiberte_Dec3,
  eLiberte_Dec4,
  eLiberte_Dec5,
  eLiberte_Phgr_Std_Aff,
  eLiberte_Phgr_Std_Dec,
  eFige_Phgr_Std_Aff,
  eFige_Phgr_Std_Dec,
  eLiberte_AFocal0,
  eLiberte_AFocal1,
  eFige_AFocal0,
  eFige_AFocal1
} eTypeContrainteCalibCamera;
void xml_init(eTypeContrainteCalibCamera & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeContrainteCalibCamera & aVal);

eTypeContrainteCalibCamera  Str2eTypeContrainteCalibCamera(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeContrainteCalibCamera & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeContrainteCalibCamera &);

std::string  Mangling( eTypeContrainteCalibCamera *);

void  BinaryUnDumpFromFile(eTypeContrainteCalibCamera &,ELISE_fp &);

typedef enum
{
  eCalibAutomRadial,
  eCalibAutomPhgrStd,
  eCalibAutomFishEyeLineaire,
  eCalibAutomFishEyeEquiSolid,
  eCalibAutomRadialBasic,
  eCalibAutomPhgrStdBasic,
  eCalibAutomFour7x2,
  eCalibAutomFour11x2,
  eCalibAutomFour15x2,
  eCalibAutomFour19x2,
  eCalibAutomEbner,
  eCalibAutomBrown,
  eCalibAutomFishEyeStereographique,
  eCalibAutomNone
} eTypeCalibAutom;
void xml_init(eTypeCalibAutom & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeCalibAutom & aVal);

eTypeCalibAutom  Str2eTypeCalibAutom(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeCalibAutom & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeCalibAutom &);

std::string  Mangling( eTypeCalibAutom *);

void  BinaryUnDumpFromFile(eTypeCalibAutom &,ELISE_fp &);

typedef enum
{
  ePoseLibre,
  ePoseFigee,
  ePoseBaseNormee,
  ePoseVraieBaseNormee,
  eCentreFige,
  eAnglesFiges
} eTypeContraintePoseCamera;
void xml_init(eTypeContraintePoseCamera & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeContraintePoseCamera & aVal);

eTypeContraintePoseCamera  Str2eTypeContraintePoseCamera(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeContraintePoseCamera & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeContraintePoseCamera &);

std::string  Mangling( eTypeContraintePoseCamera *);

void  BinaryUnDumpFromFile(eTypeContraintePoseCamera &,ELISE_fp &);

typedef enum
{
  eVerifDZ,
  eVerifResPerIm
} eTypeVerif;
void xml_init(eTypeVerif & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeVerif & aVal);

eTypeVerif  Str2eTypeVerif(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeVerif & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeVerif &);

std::string  Mangling( eTypeVerif *);

void  BinaryUnDumpFromFile(eTypeVerif &,ELISE_fp &);

typedef enum
{
  eTRPB_Ok,
  eTRPB_InsufPoseInit,
  eTRPB_PdsResNull,
  eTRPB_NotInMasq3D,
  eTRPB_BSurH,
  eTRPB_Behind,
  eTRPB_VisibIm,
  eTRPB_OutIm,
  eTRPB_PbInterBundle,
  eTRPB_RatioDistP2Cam,
  eTRPB_Unknown,
  eTRPB_NbVals
} eTypeResulPtsBundle;
void xml_init(eTypeResulPtsBundle & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeResulPtsBundle & aVal);

eTypeResulPtsBundle  Str2eTypeResulPtsBundle(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeResulPtsBundle & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeResulPtsBundle &);

std::string  Mangling( eTypeResulPtsBundle *);

void  BinaryUnDumpFromFile(eTypeResulPtsBundle &,ELISE_fp &);

typedef enum
{
  eMST_PondCard
} eTypePondMST_MEP;
void xml_init(eTypePondMST_MEP & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypePondMST_MEP & aVal);

eTypePondMST_MEP  Str2eTypePondMST_MEP(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePondMST_MEP & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypePondMST_MEP &);

std::string  Mangling( eTypePondMST_MEP *);

void  BinaryUnDumpFromFile(eTypePondMST_MEP &,ELISE_fp &);

typedef enum
{
  eCDD_Jamais,
  eCDD_OnRemontee,
  eCDD_Toujours
} eControleDescDic;
void xml_init(eControleDescDic & aVal,cElXMLTree * aTree);
std::string  eToString(const eControleDescDic & aVal);

eControleDescDic  Str2eControleDescDic(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eControleDescDic & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eControleDescDic &);

std::string  Mangling( eControleDescDic *);

void  BinaryUnDumpFromFile(eControleDescDic &,ELISE_fp &);

typedef enum
{
  ePondL2,
  ePondL1,
  ePondLK,
  ePondGauss,
  eL1Secured
} eModePonderationRobuste;
void xml_init(eModePonderationRobuste & aVal,cElXMLTree * aTree);
std::string  eToString(const eModePonderationRobuste & aVal);

eModePonderationRobuste  Str2eModePonderationRobuste(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModePonderationRobuste & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModePonderationRobuste &);

std::string  Mangling( eModePonderationRobuste *);

void  BinaryUnDumpFromFile(eModePonderationRobuste &,ELISE_fp &);

typedef enum
{
  eUME_Radian,
  eUME_Image,
  eUME_Terrain,
  eUME_Naturel
} eUniteMesureErreur;
void xml_init(eUniteMesureErreur & aVal,cElXMLTree * aTree);
std::string  eToString(const eUniteMesureErreur & aVal);

eUniteMesureErreur  Str2eUniteMesureErreur(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eUniteMesureErreur & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eUniteMesureErreur &);

std::string  Mangling( eUniteMesureErreur *);

void  BinaryUnDumpFromFile(eUniteMesureErreur &,ELISE_fp &);

typedef enum
{
  eNSM_None,
  eNSM_Iter,
  eNSM_Paquet,
  eNSM_Percentile,
  eNSM_CpleIm,
  eNSM_Indiv
} eNiveauShowMessage;
void xml_init(eNiveauShowMessage & aVal,cElXMLTree * aTree);
std::string  eToString(const eNiveauShowMessage & aVal);

eNiveauShowMessage  Str2eNiveauShowMessage(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eNiveauShowMessage & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eNiveauShowMessage &);

std::string  Mangling( eNiveauShowMessage *);

void  BinaryUnDumpFromFile(eNiveauShowMessage &,ELISE_fp &);

typedef enum
{
  eMPL_DbleCoplanIm,
  eMPL_PtTerrainInc
} eModePointLiaison;
void xml_init(eModePointLiaison & aVal,cElXMLTree * aTree);
std::string  eToString(const eModePointLiaison & aVal);

eModePointLiaison  Str2eModePointLiaison(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModePointLiaison & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModePointLiaison &);

std::string  Mangling( eModePointLiaison *);

void  BinaryUnDumpFromFile(eModePointLiaison &,ELISE_fp &);

class cPowPointLiaisons
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPowPointLiaisons & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        int & NbTot();
        const int & NbTot()const ;

        cTplValGesInit< double > & Pds();
        const cTplValGesInit< double > & Pds()const ;
    private:
        std::string mId;
        int mNbTot;
        cTplValGesInit< double > mPds;
};
cElXMLTree * ToXMLTree(const cPowPointLiaisons &);

void  BinaryDumpInFile(ELISE_fp &,const cPowPointLiaisons &);

void  BinaryUnDumpFromFile(cPowPointLiaisons &,ELISE_fp &);

std::string  Mangling( cPowPointLiaisons *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOptimizationPowel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOptimizationPowel & anObj,cElXMLTree * aTree);


        std::list< cPowPointLiaisons > & PowPointLiaisons();
        const std::list< cPowPointLiaisons > & PowPointLiaisons()const ;
    private:
        std::list< cPowPointLiaisons > mPowPointLiaisons;
};
cElXMLTree * ToXMLTree(const cOptimizationPowel &);

void  BinaryDumpInFile(ELISE_fp &,const cOptimizationPowel &);

void  BinaryUnDumpFromFile(cOptimizationPowel &,ELISE_fp &);

std::string  Mangling( cOptimizationPowel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cShowPbLiaison
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cShowPbLiaison & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NbMinPtsMul();
        const cTplValGesInit< int > & NbMinPtsMul()const ;

        cTplValGesInit< bool > & Actif();
        const cTplValGesInit< bool > & Actif()const ;

        cTplValGesInit< bool > & GetCharOnPb();
        const cTplValGesInit< bool > & GetCharOnPb()const ;
    private:
        cTplValGesInit< int > mNbMinPtsMul;
        cTplValGesInit< bool > mActif;
        cTplValGesInit< bool > mGetCharOnPb;
};
cElXMLTree * ToXMLTree(const cShowPbLiaison &);

void  BinaryDumpInFile(ELISE_fp &,const cShowPbLiaison &);

void  BinaryUnDumpFromFile(cShowPbLiaison &,ELISE_fp &);

std::string  Mangling( cShowPbLiaison *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPonderationPackMesure
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPonderationPackMesure & anObj,cElXMLTree * aTree);


        double & EcartMesureIndiv();
        const double & EcartMesureIndiv()const ;

        cTplValGesInit< bool > & Add2Compens();
        const cTplValGesInit< bool > & Add2Compens()const ;

        cTplValGesInit< eModePonderationRobuste > & ModePonderation();
        const cTplValGesInit< eModePonderationRobuste > & ModePonderation()const ;

        cTplValGesInit< double > & EcartMax();
        const cTplValGesInit< double > & EcartMax()const ;

        cTplValGesInit< double > & ExposantLK();
        const cTplValGesInit< double > & ExposantLK()const ;

        cTplValGesInit< double > & SigmaPond();
        const cTplValGesInit< double > & SigmaPond()const ;

        cTplValGesInit< double > & NbMax();
        const cTplValGesInit< double > & NbMax()const ;

        cTplValGesInit< eNiveauShowMessage > & Show();
        const cTplValGesInit< eNiveauShowMessage > & Show()const ;

        cTplValGesInit< bool > & GetChar();
        const cTplValGesInit< bool > & GetChar()const ;

        cTplValGesInit< int > & NbMinMultShowIndiv();
        const cTplValGesInit< int > & NbMinMultShowIndiv()const ;

        cTplValGesInit< std::vector<double> > & ShowPercentile();
        const cTplValGesInit< std::vector<double> > & ShowPercentile()const ;

        cTplValGesInit< double > & ExposantPoidsMult();
        const cTplValGesInit< double > & ExposantPoidsMult()const ;

        cTplValGesInit< std::string > & IdFilter3D();
        const cTplValGesInit< std::string > & IdFilter3D()const ;
    private:
        double mEcartMesureIndiv;
        cTplValGesInit< bool > mAdd2Compens;
        cTplValGesInit< eModePonderationRobuste > mModePonderation;
        cTplValGesInit< double > mEcartMax;
        cTplValGesInit< double > mExposantLK;
        cTplValGesInit< double > mSigmaPond;
        cTplValGesInit< double > mNbMax;
        cTplValGesInit< eNiveauShowMessage > mShow;
        cTplValGesInit< bool > mGetChar;
        cTplValGesInit< int > mNbMinMultShowIndiv;
        cTplValGesInit< std::vector<double> > mShowPercentile;
        cTplValGesInit< double > mExposantPoidsMult;
        cTplValGesInit< std::string > mIdFilter3D;
};
cElXMLTree * ToXMLTree(const cPonderationPackMesure &);

void  BinaryDumpInFile(ELISE_fp &,const cPonderationPackMesure &);

void  BinaryUnDumpFromFile(cPonderationPackMesure &,ELISE_fp &);

std::string  Mangling( cPonderationPackMesure *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamEstimPlan
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamEstimPlan & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & AttrSup();
        const cTplValGesInit< std::string > & AttrSup()const ;

        cTplValGesInit< std::string > & KeyCalculMasq();
        const cTplValGesInit< std::string > & KeyCalculMasq()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        cTplValGesInit< double > & LimBSurH();
        const cTplValGesInit< double > & LimBSurH()const ;

        cTplValGesInit< bool > & AcceptDefPlanIfNoPoint();
        const cTplValGesInit< bool > & AcceptDefPlanIfNoPoint()const ;
    private:
        cTplValGesInit< std::string > mAttrSup;
        cTplValGesInit< std::string > mKeyCalculMasq;
        std::string mIdBdl;
        cPonderationPackMesure mPond;
        cTplValGesInit< double > mLimBSurH;
        cTplValGesInit< bool > mAcceptDefPlanIfNoPoint;
};
cElXMLTree * ToXMLTree(const cParamEstimPlan &);

void  BinaryDumpInFile(ELISE_fp &,const cParamEstimPlan &);

void  BinaryUnDumpFromFile(cParamEstimPlan &,ELISE_fp &);

std::string  Mangling( cParamEstimPlan *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRigidBlockWeighting
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRigidBlockWeighting & anObj,cElXMLTree * aTree);


        double & PondOnTr();
        const double & PondOnTr()const ;

        double & PondOnRot();
        const double & PondOnRot()const ;

        cTplValGesInit< double > & PondOnTrFinal();
        const cTplValGesInit< double > & PondOnTrFinal()const ;

        cTplValGesInit< double > & PondOnRotFinal();
        const cTplValGesInit< double > & PondOnRotFinal()const ;
    private:
        double mPondOnTr;
        double mPondOnRot;
        cTplValGesInit< double > mPondOnTrFinal;
        cTplValGesInit< double > mPondOnRotFinal;
};
cElXMLTree * ToXMLTree(const cRigidBlockWeighting &);

void  BinaryDumpInFile(ELISE_fp &,const cRigidBlockWeighting &);

void  BinaryUnDumpFromFile(cRigidBlockWeighting &,ELISE_fp &);

std::string  Mangling( cRigidBlockWeighting *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGpsRelativeWeighting
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGpsRelativeWeighting & anObj,cElXMLTree * aTree);


        double & SigmaPerSec();
        const double & SigmaPerSec()const ;

        double & SigmaMin();
        const double & SigmaMin()const ;

        cTplValGesInit< double > & MaxResidu();
        const cTplValGesInit< double > & MaxResidu()const ;
    private:
        double mSigmaPerSec;
        double mSigmaMin;
        cTplValGesInit< double > mMaxResidu;
};
cElXMLTree * ToXMLTree(const cGpsRelativeWeighting &);

void  BinaryDumpInFile(ELISE_fp &,const cGpsRelativeWeighting &);

void  BinaryUnDumpFromFile(cGpsRelativeWeighting &,ELISE_fp &);

std::string  Mangling( cGpsRelativeWeighting *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_OneObsPlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_OneObsPlane & anObj,cElXMLTree * aTree);


        double & Sigma();
        const double & Sigma()const ;

        double & Cste();
        const double & Cste()const ;

        Pt3dr & Vect();
        const Pt3dr & Vect()const ;
    private:
        double mSigma;
        double mCste;
        Pt3dr mVect;
};
cElXMLTree * ToXMLTree(const cXml_OneObsPlane &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_OneObsPlane &);

void  BinaryUnDumpFromFile(cXml_OneObsPlane &,ELISE_fp &);

std::string  Mangling( cXml_OneObsPlane *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_ObsPlaneOnPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_ObsPlaneOnPose & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        std::list< cXml_OneObsPlane > & Obs1Plane();
        const std::list< cXml_OneObsPlane > & Obs1Plane()const ;
    private:
        std::string mNameIm;
        std::list< cXml_OneObsPlane > mObs1Plane;
};
cElXMLTree * ToXMLTree(const cXml_ObsPlaneOnPose &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_ObsPlaneOnPose &);

void  BinaryUnDumpFromFile(cXml_ObsPlaneOnPose &,ELISE_fp &);

std::string  Mangling( cXml_ObsPlaneOnPose *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_FileObsPlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_FileObsPlane & anObj,cElXMLTree * aTree);


        std::map< std::string,cXml_ObsPlaneOnPose > & Obs1Im();
        const std::map< std::string,cXml_ObsPlaneOnPose > & Obs1Im()const ;
    private:
        std::map< std::string,cXml_ObsPlaneOnPose > mObs1Im;
};
cElXMLTree * ToXMLTree(const cXml_FileObsPlane &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_FileObsPlane &);

void  BinaryUnDumpFromFile(cXml_FileObsPlane &,ELISE_fp &);

std::string  Mangling( cXml_FileObsPlane *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAperoPointeStereo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAperoPointeStereo & anObj,cElXMLTree * aTree);


        Pt2dr & P1();
        const Pt2dr & P1()const ;

        std::string & Im1();
        const std::string & Im1()const ;

        Pt2dr & P2();
        const Pt2dr & P2()const ;

        std::string & Im2();
        const std::string & Im2()const ;
    private:
        Pt2dr mP1;
        std::string mIm1;
        Pt2dr mP2;
        std::string mIm2;
};
cElXMLTree * ToXMLTree(const cAperoPointeStereo &);

void  BinaryDumpInFile(ELISE_fp &,const cAperoPointeStereo &);

void  BinaryUnDumpFromFile(cAperoPointeStereo &,ELISE_fp &);

std::string  Mangling( cAperoPointeStereo *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAperoPointeMono
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAperoPointeMono & anObj,cElXMLTree * aTree);


        Pt2dr & Pt();
        const Pt2dr & Pt()const ;

        std::string & Im();
        const std::string & Im()const ;
    private:
        Pt2dr mPt;
        std::string mIm;
};
cElXMLTree * ToXMLTree(const cAperoPointeMono &);

void  BinaryDumpInFile(ELISE_fp &,const cAperoPointeMono &);

void  BinaryUnDumpFromFile(cAperoPointeMono &,ELISE_fp &);

std::string  Mangling( cAperoPointeMono *);

/******************************************************/
/******************************************************/
/******************************************************/
class cApero2PointeFromFile
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cApero2PointeFromFile & anObj,cElXMLTree * aTree);


        std::string & File();
        const std::string & File()const ;

        std::string & NameP1();
        const std::string & NameP1()const ;

        std::string & NameP2();
        const std::string & NameP2()const ;
    private:
        std::string mFile;
        std::string mNameP1;
        std::string mNameP2;
};
cElXMLTree * ToXMLTree(const cApero2PointeFromFile &);

void  BinaryDumpInFile(ELISE_fp &,const cApero2PointeFromFile &);

void  BinaryUnDumpFromFile(cApero2PointeFromFile &,ELISE_fp &);

std::string  Mangling( cApero2PointeFromFile *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamForceRappel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamForceRappel & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & PatternNameApply();
        const cElRegex_Ptr & PatternNameApply()const ;

        std::vector< double > & Incertitude();
        const std::vector< double > & Incertitude()const ;

        cTplValGesInit< bool > & OnCur();
        const cTplValGesInit< bool > & OnCur()const ;
    private:
        cElRegex_Ptr mPatternNameApply;
        std::vector< double > mIncertitude;
        cTplValGesInit< bool > mOnCur;
};
cElXMLTree * ToXMLTree(const cParamForceRappel &);

void  BinaryDumpInFile(ELISE_fp &,const cParamForceRappel &);

void  BinaryUnDumpFromFile(cParamForceRappel &,ELISE_fp &);

std::string  Mangling( cParamForceRappel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRappelOnAngles
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRappelOnAngles & anObj,cElXMLTree * aTree);


        cParamForceRappel & ParamF();
        const cParamForceRappel & ParamF()const ;

        std::vector< int > & TetaApply();
        const std::vector< int > & TetaApply()const ;
    private:
        cParamForceRappel mParamF;
        std::vector< int > mTetaApply;
};
cElXMLTree * ToXMLTree(const cRappelOnAngles &);

void  BinaryDumpInFile(ELISE_fp &,const cRappelOnAngles &);

void  BinaryUnDumpFromFile(cRappelOnAngles &,ELISE_fp &);

std::string  Mangling( cRappelOnAngles *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRappelOnCentres
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRappelOnCentres & anObj,cElXMLTree * aTree);


        cParamForceRappel & ParamF();
        const cParamForceRappel & ParamF()const ;

        cTplValGesInit< bool > & OnlyWhenNoCentreInit();
        const cTplValGesInit< bool > & OnlyWhenNoCentreInit()const ;
    private:
        cParamForceRappel mParamF;
        cTplValGesInit< bool > mOnlyWhenNoCentreInit;
};
cElXMLTree * ToXMLTree(const cRappelOnCentres &);

void  BinaryDumpInFile(ELISE_fp &,const cRappelOnCentres &);

void  BinaryUnDumpFromFile(cRappelOnCentres &,ELISE_fp &);

std::string  Mangling( cRappelOnCentres *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRappelOnIntrinseque
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRappelOnIntrinseque & anObj,cElXMLTree * aTree);


        cParamForceRappel & ParamF();
        const cParamForceRappel & ParamF()const ;
    private:
        cParamForceRappel mParamF;
};
cElXMLTree * ToXMLTree(const cRappelOnIntrinseque &);

void  BinaryDumpInFile(ELISE_fp &,const cRappelOnIntrinseque &);

void  BinaryUnDumpFromFile(cRappelOnIntrinseque &,ELISE_fp &);

std::string  Mangling( cRappelOnIntrinseque *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlSLM_RappelOnPt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlSLM_RappelOnPt & anObj,cElXMLTree * aTree);


        double & CondMax();
        const double & CondMax()const ;
    private:
        double mCondMax;
};
cElXMLTree * ToXMLTree(const cXmlSLM_RappelOnPt &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlSLM_RappelOnPt &);

void  BinaryUnDumpFromFile(cXmlSLM_RappelOnPt &,ELISE_fp &);

std::string  Mangling( cXmlSLM_RappelOnPt *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionLevenbergMarkard
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionLevenbergMarkard & anObj,cElXMLTree * aTree);


        std::list< cRappelOnAngles > & RappelOnAngles();
        const std::list< cRappelOnAngles > & RappelOnAngles()const ;

        std::list< cRappelOnCentres > & RappelOnCentres();
        const std::list< cRappelOnCentres > & RappelOnCentres()const ;

        std::list< cRappelOnIntrinseque > & RappelOnIntrinseque();
        const std::list< cRappelOnIntrinseque > & RappelOnIntrinseque()const ;

        double & CondMax();
        const double & CondMax()const ;

        cTplValGesInit< cXmlSLM_RappelOnPt > & XmlSLM_RappelOnPt();
        const cTplValGesInit< cXmlSLM_RappelOnPt > & XmlSLM_RappelOnPt()const ;
    private:
        std::list< cRappelOnAngles > mRappelOnAngles;
        std::list< cRappelOnCentres > mRappelOnCentres;
        std::list< cRappelOnIntrinseque > mRappelOnIntrinseque;
        cTplValGesInit< cXmlSLM_RappelOnPt > mXmlSLM_RappelOnPt;
};
cElXMLTree * ToXMLTree(const cSectionLevenbergMarkard &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionLevenbergMarkard &);

void  BinaryUnDumpFromFile(cSectionLevenbergMarkard &,ELISE_fp &);

std::string  Mangling( cSectionLevenbergMarkard *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_SigmaRot
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_SigmaRot & anObj,cElXMLTree * aTree);


        double & Ang();
        const double & Ang()const ;

        double & Center();
        const double & Center()const ;
    private:
        double mAng;
        double mCenter;
};
cElXMLTree * ToXMLTree(const cXml_SigmaRot &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_SigmaRot &);

void  BinaryUnDumpFromFile(cXml_SigmaRot &,ELISE_fp &);

std::string  Mangling( cXml_SigmaRot *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetOrientationInterne
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetOrientationInterne & anObj,cElXMLTree * aTree);


        std::string & KeyFile();
        const std::string & KeyFile()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< std::string > & Tag();
        const cTplValGesInit< std::string > & Tag()const ;

        bool & AddToCur();
        const bool & AddToCur()const ;

        bool & M2C();
        const bool & M2C()const ;
    private:
        std::string mKeyFile;
        cTplValGesInit< std::string > mPatternSel;
        cTplValGesInit< std::string > mTag;
        bool mAddToCur;
        bool mM2C;
};
cElXMLTree * ToXMLTree(const cSetOrientationInterne &);

void  BinaryDumpInFile(ELISE_fp &,const cSetOrientationInterne &);

void  BinaryUnDumpFromFile(cSetOrientationInterne &,ELISE_fp &);

std::string  Mangling( cSetOrientationInterne *);

/******************************************************/
/******************************************************/
/******************************************************/
class cExportAsNewGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportAsNewGrid & anObj,cElXMLTree * aTree);


        Pt2dr & Step();
        const Pt2dr & Step()const ;

        cTplValGesInit< double > & RayonInv();
        const cTplValGesInit< double > & RayonInv()const ;

        cTplValGesInit< double > & RayonInvRelFE();
        const cTplValGesInit< double > & RayonInvRelFE()const ;
    private:
        Pt2dr mStep;
        cTplValGesInit< double > mRayonInv;
        cTplValGesInit< double > mRayonInvRelFE;
};
cElXMLTree * ToXMLTree(const cExportAsNewGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cExportAsNewGrid &);

void  BinaryUnDumpFromFile(cExportAsNewGrid &,ELISE_fp &);

std::string  Mangling( cExportAsNewGrid *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlPondRegDist
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlPondRegDist & anObj,cElXMLTree * aTree);


        double & Pds0();
        const double & Pds0()const ;

        double & Pds1();
        const double & Pds1()const ;

        double & Pds2();
        const double & Pds2()const ;

        double & NbCase();
        const double & NbCase()const ;

        double & SeuilNbPtsByCase();
        const double & SeuilNbPtsByCase()const ;
    private:
        double mPds0;
        double mPds1;
        double mPds2;
        double mNbCase;
        double mSeuilNbPtsByCase;
};
cElXMLTree * ToXMLTree(const cXmlPondRegDist &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlPondRegDist &);

void  BinaryUnDumpFromFile(cXmlPondRegDist &,ELISE_fp &);

std::string  Mangling( cXmlPondRegDist *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_EstimateOrientationInitBlockCamera
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_EstimateOrientationInitBlockCamera & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;
    private:
        std::string mId;
        cTplValGesInit< bool > mShow;
};
cElXMLTree * ToXMLTree(const cXml_EstimateOrientationInitBlockCamera &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_EstimateOrientationInitBlockCamera &);

void  BinaryUnDumpFromFile(cXml_EstimateOrientationInitBlockCamera &,ELISE_fp &);

std::string  Mangling( cXml_EstimateOrientationInitBlockCamera *);

/******************************************************/
/******************************************************/
/******************************************************/
class cShowSection
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cShowSection & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & ShowMes();
        const cTplValGesInit< bool > & ShowMes()const ;

        cTplValGesInit< std::string > & LogFile();
        const cTplValGesInit< std::string > & LogFile()const ;
    private:
        cTplValGesInit< bool > mShowMes;
        cTplValGesInit< std::string > mLogFile;
};
cElXMLTree * ToXMLTree(const cShowSection &);

void  BinaryDumpInFile(ELISE_fp &,const cShowSection &);

void  BinaryUnDumpFromFile(cShowSection &,ELISE_fp &);

std::string  Mangling( cShowSection *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSzImForInvY
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSzImForInvY & anObj,cElXMLTree * aTree);


        Pt2dr & SzIm1();
        const Pt2dr & SzIm1()const ;

        Pt2dr & SzIm2();
        const Pt2dr & SzIm2()const ;
    private:
        Pt2dr mSzIm1;
        Pt2dr mSzIm2;
};
cElXMLTree * ToXMLTree(const cSzImForInvY &);

void  BinaryDumpInFile(ELISE_fp &,const cSzImForInvY &);

void  BinaryUnDumpFromFile(cSzImForInvY &,ELISE_fp &);

std::string  Mangling( cSzImForInvY *);

class cSplitLayer
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSplitLayer & anObj,cElXMLTree * aTree);


        std::string & IdLayer();
        const std::string & IdLayer()const ;

        std::string & KeyCalHomSplit();
        const std::string & KeyCalHomSplit()const ;
    private:
        std::string mIdLayer;
        std::string mKeyCalHomSplit;
};
cElXMLTree * ToXMLTree(const cSplitLayer &);

void  BinaryDumpInFile(ELISE_fp &,const cSplitLayer &);

void  BinaryUnDumpFromFile(cSplitLayer &,ELISE_fp &);

std::string  Mangling( cSplitLayer *);

class cBDD_PtsLiaisons
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBDD_PtsLiaisons & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & TestForMatin();
        const cTplValGesInit< int > & TestForMatin()const ;

        cTplValGesInit< bool > & UseAsPtMultiple();
        const cTplValGesInit< bool > & UseAsPtMultiple()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< std::string > & IdFilterSameGrp();
        const cTplValGesInit< std::string > & IdFilterSameGrp()const ;

        cTplValGesInit< bool > & AutoSuprReflexif();
        const cTplValGesInit< bool > & AutoSuprReflexif()const ;

        std::vector< std::string > & KeySet();
        const std::vector< std::string > & KeySet()const ;

        std::vector< std::string > & KeyAssoc();
        const std::vector< std::string > & KeyAssoc()const ;

        std::list< std::string > & XMLKeySetOrPat();
        const std::list< std::string > & XMLKeySetOrPat()const ;

        Pt2dr & SzIm1();
        const Pt2dr & SzIm1()const ;

        Pt2dr & SzIm2();
        const Pt2dr & SzIm2()const ;

        cTplValGesInit< cSzImForInvY > & SzImForInvY();
        const cTplValGesInit< cSzImForInvY > & SzImForInvY()const ;

        std::string & IdLayer();
        const std::string & IdLayer()const ;

        std::string & KeyCalHomSplit();
        const std::string & KeyCalHomSplit()const ;

        cTplValGesInit< cSplitLayer > & SplitLayer();
        const cTplValGesInit< cSplitLayer > & SplitLayer()const ;
    private:
        cTplValGesInit< int > mTestForMatin;
        cTplValGesInit< bool > mUseAsPtMultiple;
        std::string mId;
        cTplValGesInit< std::string > mIdFilterSameGrp;
        cTplValGesInit< bool > mAutoSuprReflexif;
        std::vector< std::string > mKeySet;
        std::vector< std::string > mKeyAssoc;
        std::list< std::string > mXMLKeySetOrPat;
        cTplValGesInit< cSzImForInvY > mSzImForInvY;
        cTplValGesInit< cSplitLayer > mSplitLayer;
};
cElXMLTree * ToXMLTree(const cBDD_PtsLiaisons &);

void  BinaryDumpInFile(ELISE_fp &,const cBDD_PtsLiaisons &);

void  BinaryUnDumpFromFile(cBDD_PtsLiaisons &,ELISE_fp &);

std::string  Mangling( cBDD_PtsLiaisons *);

class cBDD_NewPtMul
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBDD_NewPtMul & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & SH();
        const std::string & SH()const ;

        bool & BinaryMode();
        const bool & BinaryMode()const ;

        bool & SupressStdHom();
        const bool & SupressStdHom()const ;
    private:
        std::string mId;
        std::string mSH;
        bool mBinaryMode;
        bool mSupressStdHom;
};
cElXMLTree * ToXMLTree(const cBDD_NewPtMul &);

void  BinaryDumpInFile(ELISE_fp &,const cBDD_NewPtMul &);

void  BinaryUnDumpFromFile(cBDD_NewPtMul &,ELISE_fp &);

std::string  Mangling( cBDD_NewPtMul *);

class cBddApp_AutoNum
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBddApp_AutoNum & anObj,cElXMLTree * aTree);


        double & DistFusion();
        const double & DistFusion()const ;

        double & DistAmbiguite();
        const double & DistAmbiguite()const ;
    private:
        double mDistFusion;
        double mDistAmbiguite;
};
cElXMLTree * ToXMLTree(const cBddApp_AutoNum &);

void  BinaryDumpInFile(ELISE_fp &,const cBddApp_AutoNum &);

void  BinaryUnDumpFromFile(cBddApp_AutoNum &,ELISE_fp &);

std::string  Mangling( cBddApp_AutoNum *);

class cBDD_PtsAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBDD_PtsAppuis & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< Pt2dr > & SzImForInvY();
        const cTplValGesInit< Pt2dr > & SzImForInvY()const ;

        cTplValGesInit< bool > & InvXY();
        const cTplValGesInit< bool > & InvXY()const ;

        cTplValGesInit< Pt3dr > & ToSubstract();
        const cTplValGesInit< Pt3dr > & ToSubstract()const ;

        cTplValGesInit< std::string > & TagExtract();
        const cTplValGesInit< std::string > & TagExtract()const ;

        double & DistFusion();
        const double & DistFusion()const ;

        double & DistAmbiguite();
        const double & DistAmbiguite()const ;

        cTplValGesInit< cBddApp_AutoNum > & BddApp_AutoNum();
        const cTplValGesInit< cBddApp_AutoNum > & BddApp_AutoNum()const ;
    private:
        std::string mId;
        std::string mKeySet;
        std::string mKeyAssoc;
        cTplValGesInit< Pt2dr > mSzImForInvY;
        cTplValGesInit< bool > mInvXY;
        cTplValGesInit< Pt3dr > mToSubstract;
        cTplValGesInit< std::string > mTagExtract;
        cTplValGesInit< cBddApp_AutoNum > mBddApp_AutoNum;
};
cElXMLTree * ToXMLTree(const cBDD_PtsAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cBDD_PtsAppuis &);

void  BinaryUnDumpFromFile(cBDD_PtsAppuis &,ELISE_fp &);

std::string  Mangling( cBDD_PtsAppuis *);

class cBDD_ObsAppuisFlottant
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBDD_ObsAppuisFlottant & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt2dr > & OffsetIm();
        const cTplValGesInit< Pt2dr > & OffsetIm()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< std::string > & KeySetOrPat();
        const cTplValGesInit< std::string > & KeySetOrPat()const ;

        cTplValGesInit< std::string > & NameAppuiSelector();
        const cTplValGesInit< std::string > & NameAppuiSelector()const ;

        cTplValGesInit< bool > & AcceptNoGround();
        const cTplValGesInit< bool > & AcceptNoGround()const ;

        cTplValGesInit< std::string > & KeySetSegDroite();
        const cTplValGesInit< std::string > & KeySetSegDroite()const ;
    private:
        cTplValGesInit< Pt2dr > mOffsetIm;
        std::string mId;
        cTplValGesInit< std::string > mKeySetOrPat;
        cTplValGesInit< std::string > mNameAppuiSelector;
        cTplValGesInit< bool > mAcceptNoGround;
        cTplValGesInit< std::string > mKeySetSegDroite;
};
cElXMLTree * ToXMLTree(const cBDD_ObsAppuisFlottant &);

void  BinaryDumpInFile(ELISE_fp &,const cBDD_ObsAppuisFlottant &);

void  BinaryUnDumpFromFile(cBDD_ObsAppuisFlottant &,ELISE_fp &);

std::string  Mangling( cBDD_ObsAppuisFlottant *);

class cBDD_Orient
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBDD_Orient & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< eConventionsOrientation > & ConvOr();
        const cTplValGesInit< eConventionsOrientation > & ConvOr()const ;
    private:
        std::string mId;
        std::string mKeySet;
        std::string mKeyAssoc;
        cTplValGesInit< eConventionsOrientation > mConvOr;
};
cElXMLTree * ToXMLTree(const cBDD_Orient &);

void  BinaryDumpInFile(ELISE_fp &,const cBDD_Orient &);

void  BinaryUnDumpFromFile(cBDD_Orient &,ELISE_fp &);

std::string  Mangling( cBDD_Orient *);

class cCalcOffsetCentre
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalcOffsetCentre & anObj,cElXMLTree * aTree);


        std::string & IdBase();
        const std::string & IdBase()const ;

        std::string & KeyCalcBande();
        const std::string & KeyCalcBande()const ;

        cTplValGesInit< bool > & OffsetUnknown();
        const cTplValGesInit< bool > & OffsetUnknown()const ;
    private:
        std::string mIdBase;
        std::string mKeyCalcBande;
        cTplValGesInit< bool > mOffsetUnknown;
};
cElXMLTree * ToXMLTree(const cCalcOffsetCentre &);

void  BinaryDumpInFile(ELISE_fp &,const cCalcOffsetCentre &);

void  BinaryUnDumpFromFile(cCalcOffsetCentre &,ELISE_fp &);

std::string  Mangling( cCalcOffsetCentre *);

class cBDD_Centre
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBDD_Centre & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & KeySet();
        const std::string & KeySet()const ;

        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< std::string > & Tag();
        const cTplValGesInit< std::string > & Tag()const ;

        cTplValGesInit< std::string > & ByFileTrajecto();
        const cTplValGesInit< std::string > & ByFileTrajecto()const ;

        cTplValGesInit< std::string > & PatternFileTrajecto();
        const cTplValGesInit< std::string > & PatternFileTrajecto()const ;

        cTplValGesInit< std::string > & PatternRefutFileTrajecto();
        const cTplValGesInit< std::string > & PatternRefutFileTrajecto()const ;

        std::string & IdBase();
        const std::string & IdBase()const ;

        std::string & KeyCalcBande();
        const std::string & KeyCalcBande()const ;

        cTplValGesInit< bool > & OffsetUnknown();
        const cTplValGesInit< bool > & OffsetUnknown()const ;

        cTplValGesInit< cCalcOffsetCentre > & CalcOffsetCentre();
        const cTplValGesInit< cCalcOffsetCentre > & CalcOffsetCentre()const ;
    private:
        std::string mId;
        std::string mKeySet;
        std::string mKeyAssoc;
        cTplValGesInit< std::string > mTag;
        cTplValGesInit< std::string > mByFileTrajecto;
        cTplValGesInit< std::string > mPatternFileTrajecto;
        cTplValGesInit< std::string > mPatternRefutFileTrajecto;
        cTplValGesInit< cCalcOffsetCentre > mCalcOffsetCentre;
};
cElXMLTree * ToXMLTree(const cBDD_Centre &);

void  BinaryDumpInFile(ELISE_fp &,const cBDD_Centre &);

void  BinaryUnDumpFromFile(cBDD_Centre &,ELISE_fp &);

std::string  Mangling( cBDD_Centre *);

class cFilterProj3D
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFilterProj3D & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & AttrSup();
        const std::string & AttrSup()const ;

        std::string & KeyCalculMasq();
        const std::string & KeyCalculMasq()const ;
    private:
        std::string mId;
        std::string mPatternSel;
        std::string mAttrSup;
        std::string mKeyCalculMasq;
};
cElXMLTree * ToXMLTree(const cFilterProj3D &);

void  BinaryDumpInFile(ELISE_fp &,const cFilterProj3D &);

void  BinaryUnDumpFromFile(cFilterProj3D &,ELISE_fp &);

std::string  Mangling( cFilterProj3D *);

class cLayerTerrain
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLayerTerrain & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyAssocGeoref();
        const cTplValGesInit< std::string > & KeyAssocGeoref()const ;

        std::string & KeyAssocOrImage();
        const std::string & KeyAssocOrImage()const ;

        std::string & SysCoIm();
        const std::string & SysCoIm()const ;

        cTplValGesInit< std::string > & TagOri();
        const cTplValGesInit< std::string > & TagOri()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;
    private:
        cTplValGesInit< std::string > mKeyAssocGeoref;
        std::string mKeyAssocOrImage;
        std::string mSysCoIm;
        cTplValGesInit< std::string > mTagOri;
        cTplValGesInit< double > mZMoyen;
};
cElXMLTree * ToXMLTree(const cLayerTerrain &);

void  BinaryDumpInFile(ELISE_fp &,const cLayerTerrain &);

void  BinaryUnDumpFromFile(cLayerTerrain &,ELISE_fp &);

std::string  Mangling( cLayerTerrain *);

class cLayerImageToPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLayerImageToPose & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & KeyCalculImage();
        const std::string & KeyCalculImage()const ;

        int & FactRed();
        const int & FactRed()const ;

        cTplValGesInit< std::string > & KeyNameRed();
        const cTplValGesInit< std::string > & KeyNameRed()const ;

        cTplValGesInit< int > & FactCoherence();
        const cTplValGesInit< int > & FactCoherence()const ;

        std::vector< int > & EtiqPrio();
        const std::vector< int > & EtiqPrio()const ;

        cTplValGesInit< std::string > & KeyAssocGeoref();
        const cTplValGesInit< std::string > & KeyAssocGeoref()const ;

        std::string & KeyAssocOrImage();
        const std::string & KeyAssocOrImage()const ;

        std::string & SysCoIm();
        const std::string & SysCoIm()const ;

        cTplValGesInit< std::string > & TagOri();
        const cTplValGesInit< std::string > & TagOri()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        cTplValGesInit< cLayerTerrain > & LayerTerrain();
        const cTplValGesInit< cLayerTerrain > & LayerTerrain()const ;
    private:
        std::string mId;
        std::string mKeyCalculImage;
        int mFactRed;
        cTplValGesInit< std::string > mKeyNameRed;
        cTplValGesInit< int > mFactCoherence;
        std::vector< int > mEtiqPrio;
        cTplValGesInit< cLayerTerrain > mLayerTerrain;
};
cElXMLTree * ToXMLTree(const cLayerImageToPose &);

void  BinaryDumpInFile(ELISE_fp &,const cLayerImageToPose &);

void  BinaryUnDumpFromFile(cLayerImageToPose &,ELISE_fp &);

std::string  Mangling( cLayerImageToPose *);

class cDeclareObsRelGPS
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDeclareObsRelGPS & anObj,cElXMLTree * aTree);


        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & Id();
        const std::string & Id()const ;
    private:
        std::string mPatternSel;
        std::string mId;
};
cElXMLTree * ToXMLTree(const cDeclareObsRelGPS &);

void  BinaryDumpInFile(ELISE_fp &,const cDeclareObsRelGPS &);

void  BinaryUnDumpFromFile(cDeclareObsRelGPS &,ELISE_fp &);

std::string  Mangling( cDeclareObsRelGPS *);

class cDeclareObsCalConseq
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDeclareObsCalConseq & anObj,cElXMLTree * aTree);


        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & Key();
        const std::string & Key()const ;

        cTplValGesInit< std::string > & KeyJump();
        const cTplValGesInit< std::string > & KeyJump()const ;

        bool & AddFreeRot();
        const bool & AddFreeRot()const ;
    private:
        std::string mPatternSel;
        std::string mKey;
        cTplValGesInit< std::string > mKeyJump;
        bool mAddFreeRot;
};
cElXMLTree * ToXMLTree(const cDeclareObsCalConseq &);

void  BinaryDumpInFile(ELISE_fp &,const cDeclareObsCalConseq &);

void  BinaryUnDumpFromFile(cDeclareObsCalConseq &,ELISE_fp &);

std::string  Mangling( cDeclareObsCalConseq *);

class cSectionBDD_Observation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionBDD_Observation & anObj,cElXMLTree * aTree);


        std::list< cBDD_PtsLiaisons > & BDD_PtsLiaisons();
        const std::list< cBDD_PtsLiaisons > & BDD_PtsLiaisons()const ;

        std::list< cBDD_NewPtMul > & BDD_NewPtMul();
        const std::list< cBDD_NewPtMul > & BDD_NewPtMul()const ;

        std::list< cBDD_PtsAppuis > & BDD_PtsAppuis();
        const std::list< cBDD_PtsAppuis > & BDD_PtsAppuis()const ;

        std::list< cBDD_ObsAppuisFlottant > & BDD_ObsAppuisFlottant();
        const std::list< cBDD_ObsAppuisFlottant > & BDD_ObsAppuisFlottant()const ;

        std::list< cBDD_Orient > & BDD_Orient();
        const std::list< cBDD_Orient > & BDD_Orient()const ;

        std::list< cBDD_Centre > & BDD_Centre();
        const std::list< cBDD_Centre > & BDD_Centre()const ;

        std::list< cFilterProj3D > & FilterProj3D();
        const std::list< cFilterProj3D > & FilterProj3D()const ;

        std::list< cLayerImageToPose > & LayerImageToPose();
        const std::list< cLayerImageToPose > & LayerImageToPose()const ;

        cTplValGesInit< double > & LimInfBSurHPMoy();
        const cTplValGesInit< double > & LimInfBSurHPMoy()const ;

        cTplValGesInit< double > & LimSupBSurHPMoy();
        const cTplValGesInit< double > & LimSupBSurHPMoy()const ;

        std::list< cDeclareObsRelGPS > & DeclareObsRelGPS();
        const std::list< cDeclareObsRelGPS > & DeclareObsRelGPS()const ;

        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & Key();
        const std::string & Key()const ;

        cTplValGesInit< std::string > & KeyJump();
        const cTplValGesInit< std::string > & KeyJump()const ;

        bool & AddFreeRot();
        const bool & AddFreeRot()const ;

        cTplValGesInit< cDeclareObsCalConseq > & DeclareObsCalConseq();
        const cTplValGesInit< cDeclareObsCalConseq > & DeclareObsCalConseq()const ;
    private:
        std::list< cBDD_PtsLiaisons > mBDD_PtsLiaisons;
        std::list< cBDD_NewPtMul > mBDD_NewPtMul;
        std::list< cBDD_PtsAppuis > mBDD_PtsAppuis;
        std::list< cBDD_ObsAppuisFlottant > mBDD_ObsAppuisFlottant;
        std::list< cBDD_Orient > mBDD_Orient;
        std::list< cBDD_Centre > mBDD_Centre;
        std::list< cFilterProj3D > mFilterProj3D;
        std::list< cLayerImageToPose > mLayerImageToPose;
        cTplValGesInit< double > mLimInfBSurHPMoy;
        cTplValGesInit< double > mLimSupBSurHPMoy;
        std::list< cDeclareObsRelGPS > mDeclareObsRelGPS;
        cTplValGesInit< cDeclareObsCalConseq > mDeclareObsCalConseq;
};
cElXMLTree * ToXMLTree(const cSectionBDD_Observation &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionBDD_Observation &);

void  BinaryUnDumpFromFile(cSectionBDD_Observation &,ELISE_fp &);

std::string  Mangling( cSectionBDD_Observation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGpsOffset
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGpsOffset & anObj,cElXMLTree * aTree);


        Pt3dr & ValInit();
        const Pt3dr & ValInit()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< Pt3dr > & Inc();
        const cTplValGesInit< Pt3dr > & Inc()const ;
    private:
        Pt3dr mValInit;
        std::string mId;
        cTplValGesInit< Pt3dr > mInc;
};
cElXMLTree * ToXMLTree(const cGpsOffset &);

void  BinaryDumpInFile(ELISE_fp &,const cGpsOffset &);

void  BinaryUnDumpFromFile(cGpsOffset &,ELISE_fp &);

std::string  Mangling( cGpsOffset *);

class cDataObsPlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDataObsPlane & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        cTplValGesInit< double > & Weight();
        const cTplValGesInit< double > & Weight()const ;

        cXml_FileObsPlane & Data();
        const cXml_FileObsPlane & Data()const ;
    private:
        std::string mId;
        std::string mNameFile;
        cTplValGesInit< double > mWeight;
        cXml_FileObsPlane mData;
};
cElXMLTree * ToXMLTree(const cDataObsPlane &);

void  BinaryDumpInFile(ELISE_fp &,const cDataObsPlane &);

void  BinaryUnDumpFromFile(cDataObsPlane &,ELISE_fp &);

std::string  Mangling( cDataObsPlane *);

class cCalibAutomNoDist
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibAutomNoDist & anObj,cElXMLTree * aTree);


        eTypeCalibAutom & TypeDist();
        const eTypeCalibAutom & TypeDist()const ;

        cTplValGesInit< std::string > & NameIm();
        const cTplValGesInit< std::string > & NameIm()const ;

        cTplValGesInit< std::string > & KeyFileSauv();
        const cTplValGesInit< std::string > & KeyFileSauv()const ;

        cTplValGesInit< Pt2dr > & PositionRelPP();
        const cTplValGesInit< Pt2dr > & PositionRelPP()const ;
    private:
        eTypeCalibAutom mTypeDist;
        cTplValGesInit< std::string > mNameIm;
        cTplValGesInit< std::string > mKeyFileSauv;
        cTplValGesInit< Pt2dr > mPositionRelPP;
};
cElXMLTree * ToXMLTree(const cCalibAutomNoDist &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibAutomNoDist &);

void  BinaryUnDumpFromFile(cCalibAutomNoDist &,ELISE_fp &);

std::string  Mangling( cCalibAutomNoDist *);

class cCalValueInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalValueInit & anObj,cElXMLTree * aTree);


        cTplValGesInit< cCalibrationInternConique > & CalFromValues();
        const cTplValGesInit< cCalibrationInternConique > & CalFromValues()const ;

        cTplValGesInit< cSpecExtractFromFile > & CalFromFileExtern();
        const cTplValGesInit< cSpecExtractFromFile > & CalFromFileExtern()const ;

        cTplValGesInit< bool > & CalibFromMmBD();
        const cTplValGesInit< bool > & CalibFromMmBD()const ;

        eTypeCalibAutom & TypeDist();
        const eTypeCalibAutom & TypeDist()const ;

        cTplValGesInit< std::string > & NameIm();
        const cTplValGesInit< std::string > & NameIm()const ;

        cTplValGesInit< std::string > & KeyFileSauv();
        const cTplValGesInit< std::string > & KeyFileSauv()const ;

        cTplValGesInit< Pt2dr > & PositionRelPP();
        const cTplValGesInit< Pt2dr > & PositionRelPP()const ;

        cTplValGesInit< cCalibAutomNoDist > & CalibAutomNoDist();
        const cTplValGesInit< cCalibAutomNoDist > & CalibAutomNoDist()const ;
    private:
        cTplValGesInit< cCalibrationInternConique > mCalFromValues;
        cTplValGesInit< cSpecExtractFromFile > mCalFromFileExtern;
        cTplValGesInit< bool > mCalibFromMmBD;
        cTplValGesInit< cCalibAutomNoDist > mCalibAutomNoDist;
};
cElXMLTree * ToXMLTree(const cCalValueInit &);

void  BinaryDumpInFile(ELISE_fp &,const cCalValueInit &);

void  BinaryUnDumpFromFile(cCalValueInit &,ELISE_fp &);

std::string  Mangling( cCalValueInit *);

class cAddParamAFocal
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAddParamAFocal & anObj,cElXMLTree * aTree);


        std::vector< double > & Coeffs();
        const std::vector< double > & Coeffs()const ;
    private:
        std::vector< double > mCoeffs;
};
cElXMLTree * ToXMLTree(const cAddParamAFocal &);

void  BinaryDumpInFile(ELISE_fp &,const cAddParamAFocal &);

void  BinaryUnDumpFromFile(cAddParamAFocal &,ELISE_fp &);

std::string  Mangling( cAddParamAFocal *);

class cCalibPerPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibPerPose & anObj,cElXMLTree * aTree);


        std::string & KeyPose2Cal();
        const std::string & KeyPose2Cal()const ;

        cTplValGesInit< std::string > & KeyInitFromPose();
        const cTplValGesInit< std::string > & KeyInitFromPose()const ;
    private:
        std::string mKeyPose2Cal;
        cTplValGesInit< std::string > mKeyInitFromPose;
};
cElXMLTree * ToXMLTree(const cCalibPerPose &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibPerPose &);

void  BinaryUnDumpFromFile(cCalibPerPose &,ELISE_fp &);

std::string  Mangling( cCalibPerPose *);

class cCalibrationCameraInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalibrationCameraInc & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< eConventionsOrientation > & ConvCal();
        const cTplValGesInit< eConventionsOrientation > & ConvCal()const ;

        cTplValGesInit< std::string > & Directory();
        const cTplValGesInit< std::string > & Directory()const ;

        cTplValGesInit< bool > & AddDirCur();
        const cTplValGesInit< bool > & AddDirCur()const ;

        cTplValGesInit< cCalibrationInternConique > & CalFromValues();
        const cTplValGesInit< cCalibrationInternConique > & CalFromValues()const ;

        cTplValGesInit< cSpecExtractFromFile > & CalFromFileExtern();
        const cTplValGesInit< cSpecExtractFromFile > & CalFromFileExtern()const ;

        cTplValGesInit< bool > & CalibFromMmBD();
        const cTplValGesInit< bool > & CalibFromMmBD()const ;

        eTypeCalibAutom & TypeDist();
        const eTypeCalibAutom & TypeDist()const ;

        cTplValGesInit< std::string > & NameIm();
        const cTplValGesInit< std::string > & NameIm()const ;

        cTplValGesInit< std::string > & KeyFileSauv();
        const cTplValGesInit< std::string > & KeyFileSauv()const ;

        cTplValGesInit< Pt2dr > & PositionRelPP();
        const cTplValGesInit< Pt2dr > & PositionRelPP()const ;

        cTplValGesInit< cCalibAutomNoDist > & CalibAutomNoDist();
        const cTplValGesInit< cCalibAutomNoDist > & CalibAutomNoDist()const ;

        cCalValueInit & CalValueInit();
        const cCalValueInit & CalValueInit()const ;

        cTplValGesInit< cCalibDistortion > & DistortionAddInc();
        const cTplValGesInit< cCalibDistortion > & DistortionAddInc()const ;

        std::vector< double > & Coeffs();
        const std::vector< double > & Coeffs()const ;

        cTplValGesInit< cAddParamAFocal > & AddParamAFocal();
        const cTplValGesInit< cAddParamAFocal > & AddParamAFocal()const ;

        cTplValGesInit< double > & RayMaxUtile();
        const cTplValGesInit< double > & RayMaxUtile()const ;

        cTplValGesInit< bool > & RayIsRelatifDiag();
        const cTplValGesInit< bool > & RayIsRelatifDiag()const ;

        cTplValGesInit< bool > & RayApplyOnlyFE();
        const cTplValGesInit< bool > & RayApplyOnlyFE()const ;

        cTplValGesInit< double > & PropDiagUtile();
        const cTplValGesInit< double > & PropDiagUtile()const ;

        std::string & KeyPose2Cal();
        const std::string & KeyPose2Cal()const ;

        cTplValGesInit< std::string > & KeyInitFromPose();
        const cTplValGesInit< std::string > & KeyInitFromPose()const ;

        cTplValGesInit< cCalibPerPose > & CalibPerPose();
        const cTplValGesInit< cCalibPerPose > & CalibPerPose()const ;
    private:
        std::string mName;
        cTplValGesInit< eConventionsOrientation > mConvCal;
        cTplValGesInit< std::string > mDirectory;
        cTplValGesInit< bool > mAddDirCur;
        cCalValueInit mCalValueInit;
        cTplValGesInit< cCalibDistortion > mDistortionAddInc;
        cTplValGesInit< cAddParamAFocal > mAddParamAFocal;
        cTplValGesInit< double > mRayMaxUtile;
        cTplValGesInit< bool > mRayIsRelatifDiag;
        cTplValGesInit< bool > mRayApplyOnlyFE;
        cTplValGesInit< double > mPropDiagUtile;
        cTplValGesInit< cCalibPerPose > mCalibPerPose;
};
cElXMLTree * ToXMLTree(const cCalibrationCameraInc &);

void  BinaryDumpInFile(ELISE_fp &,const cCalibrationCameraInc &);

void  BinaryUnDumpFromFile(cCalibrationCameraInc &,ELISE_fp &);

std::string  Mangling( cCalibrationCameraInc *);

class cBlockGlobalBundle
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBlockGlobalBundle & anObj,cElXMLTree * aTree);


        cTplValGesInit< cXml_SigmaRot > & SigmaV0();
        const cTplValGesInit< cXml_SigmaRot > & SigmaV0()const ;

        cTplValGesInit< bool > & V0Stricte();
        const cTplValGesInit< bool > & V0Stricte()const ;

        cTplValGesInit< double > & SigmaSimDist();
        const cTplValGesInit< double > & SigmaSimDist()const ;
    private:
        cTplValGesInit< cXml_SigmaRot > mSigmaV0;
        cTplValGesInit< bool > mV0Stricte;
        cTplValGesInit< double > mSigmaSimDist;
};
cElXMLTree * ToXMLTree(const cBlockGlobalBundle &);

void  BinaryDumpInFile(ELISE_fp &,const cBlockGlobalBundle &);

void  BinaryUnDumpFromFile(cBlockGlobalBundle &,ELISE_fp &);

std::string  Mangling( cBlockGlobalBundle *);

class cUseForBundle
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cUseForBundle & anObj,cElXMLTree * aTree);


        cTplValGesInit< cXml_SigmaRot > & SigmaV0();
        const cTplValGesInit< cXml_SigmaRot > & SigmaV0()const ;

        cTplValGesInit< bool > & V0Stricte();
        const cTplValGesInit< bool > & V0Stricte()const ;

        cTplValGesInit< double > & SigmaSimDist();
        const cTplValGesInit< double > & SigmaSimDist()const ;

        cTplValGesInit< cBlockGlobalBundle > & BlockGlobalBundle();
        const cTplValGesInit< cBlockGlobalBundle > & BlockGlobalBundle()const ;

        bool & RelTimeBundle();
        const bool & RelTimeBundle()const ;

        cTplValGesInit< bool > & RelDistTimeBundle();
        const cTplValGesInit< bool > & RelDistTimeBundle()const ;

        cTplValGesInit< bool > & GlobDistTimeBundle();
        const cTplValGesInit< bool > & GlobDistTimeBundle()const ;
    private:
        cTplValGesInit< cBlockGlobalBundle > mBlockGlobalBundle;
        bool mRelTimeBundle;
        cTplValGesInit< bool > mRelDistTimeBundle;
        cTplValGesInit< bool > mGlobDistTimeBundle;
};
cElXMLTree * ToXMLTree(const cUseForBundle &);

void  BinaryDumpInFile(ELISE_fp &,const cUseForBundle &);

void  BinaryUnDumpFromFile(cUseForBundle &,ELISE_fp &);

std::string  Mangling( cUseForBundle *);

class cBlockCamera
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBlockCamera & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        cTplValGesInit< std::string > & Id();
        const cTplValGesInit< std::string > & Id()const ;

        cTplValGesInit< cXml_SigmaRot > & SigmaV0();
        const cTplValGesInit< cXml_SigmaRot > & SigmaV0()const ;

        cTplValGesInit< bool > & V0Stricte();
        const cTplValGesInit< bool > & V0Stricte()const ;

        cTplValGesInit< double > & SigmaSimDist();
        const cTplValGesInit< double > & SigmaSimDist()const ;

        cTplValGesInit< cBlockGlobalBundle > & BlockGlobalBundle();
        const cTplValGesInit< cBlockGlobalBundle > & BlockGlobalBundle()const ;

        bool & RelTimeBundle();
        const bool & RelTimeBundle()const ;

        cTplValGesInit< bool > & RelDistTimeBundle();
        const cTplValGesInit< bool > & RelDistTimeBundle()const ;

        cTplValGesInit< bool > & GlobDistTimeBundle();
        const cTplValGesInit< bool > & GlobDistTimeBundle()const ;

        cTplValGesInit< cUseForBundle > & UseForBundle();
        const cTplValGesInit< cUseForBundle > & UseForBundle()const ;
    private:
        std::string mNameFile;
        cTplValGesInit< std::string > mId;
        cTplValGesInit< cUseForBundle > mUseForBundle;
};
cElXMLTree * ToXMLTree(const cBlockCamera &);

void  BinaryDumpInFile(ELISE_fp &,const cBlockCamera &);

void  BinaryUnDumpFromFile(cBlockCamera &,ELISE_fp &);

std::string  Mangling( cBlockCamera *);

class cCamGenInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCamGenInc & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & PatterName();
        const cElRegex_Ptr & PatterName()const ;

        std::string & Orient();
        const std::string & Orient()const ;

        cTplValGesInit< bool > & ErrorWhenEmpytPat();
        const cTplValGesInit< bool > & ErrorWhenEmpytPat()const ;

        cTplValGesInit< bool > & ErrorWhenNoFileOrient();
        const cTplValGesInit< bool > & ErrorWhenNoFileOrient()const ;
    private:
        cElRegex_Ptr mPatterName;
        std::string mOrient;
        cTplValGesInit< bool > mErrorWhenEmpytPat;
        cTplValGesInit< bool > mErrorWhenNoFileOrient;
};
cElXMLTree * ToXMLTree(const cCamGenInc &);

void  BinaryDumpInFile(ELISE_fp &,const cCamGenInc &);

void  BinaryUnDumpFromFile(cCamGenInc &,ELISE_fp &);

std::string  Mangling( cCamGenInc *);

class cMEP_SPEC_MST
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMEP_SPEC_MST & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & MSTBlockRigid();
        const cTplValGesInit< std::string > & MSTBlockRigid()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< int > & MinNbPtsInit();
        const cTplValGesInit< int > & MinNbPtsInit()const ;

        cTplValGesInit< double > & ExpDist();
        const cTplValGesInit< double > & ExpDist()const ;

        cTplValGesInit< double > & ExpNb();
        const cTplValGesInit< double > & ExpNb()const ;

        cTplValGesInit< bool > & MontageOnInit();
        const cTplValGesInit< bool > & MontageOnInit()const ;

        cTplValGesInit< int > & NbInitMinBeforeUnconnect();
        const cTplValGesInit< int > & NbInitMinBeforeUnconnect()const ;
    private:
        cTplValGesInit< std::string > mMSTBlockRigid;
        cTplValGesInit< bool > mShow;
        cTplValGesInit< int > mMinNbPtsInit;
        cTplValGesInit< double > mExpDist;
        cTplValGesInit< double > mExpNb;
        cTplValGesInit< bool > mMontageOnInit;
        cTplValGesInit< int > mNbInitMinBeforeUnconnect;
};
cElXMLTree * ToXMLTree(const cMEP_SPEC_MST &);

void  BinaryDumpInFile(ELISE_fp &,const cMEP_SPEC_MST &);

void  BinaryUnDumpFromFile(cMEP_SPEC_MST &,ELISE_fp &);

std::string  Mangling( cMEP_SPEC_MST *);

class cApplyOAI
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cApplyOAI & anObj,cElXMLTree * aTree);


        eTypeContraintePoseCamera & Cstr();
        const eTypeContraintePoseCamera & Cstr()const ;

        std::string & PatternApply();
        const std::string & PatternApply()const ;
    private:
        eTypeContraintePoseCamera mCstr;
        std::string mPatternApply;
};
cElXMLTree * ToXMLTree(const cApplyOAI &);

void  BinaryDumpInFile(ELISE_fp &,const cApplyOAI &);

void  BinaryUnDumpFromFile(cApplyOAI &,ELISE_fp &);

std::string  Mangling( cApplyOAI *);

class cOptimizeAfterInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOptimizeAfterInit & anObj,cElXMLTree * aTree);


        cOptimizationPowel & ParamOptim();
        const cOptimizationPowel & ParamOptim()const ;

        std::list< cApplyOAI > & ApplyOAI();
        const std::list< cApplyOAI > & ApplyOAI()const ;
    private:
        cOptimizationPowel mParamOptim;
        std::list< cApplyOAI > mApplyOAI;
};
cElXMLTree * ToXMLTree(const cOptimizeAfterInit &);

void  BinaryDumpInFile(ELISE_fp &,const cOptimizeAfterInit &);

void  BinaryUnDumpFromFile(cOptimizeAfterInit &,ELISE_fp &);

std::string  Mangling( cOptimizeAfterInit *);

class cCalcNameOnExistingTag
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalcNameOnExistingTag & anObj,cElXMLTree * aTree);


        std::string & KeyCalcFileOriExt();
        const std::string & KeyCalcFileOriExt()const ;

        std::string & KeyCalcName();
        const std::string & KeyCalcName()const ;

        std::string & TagExist();
        const std::string & TagExist()const ;

        std::string & TagNotExist();
        const std::string & TagNotExist()const ;

        cTplValGesInit< bool > & ExigCohTags();
        const cTplValGesInit< bool > & ExigCohTags()const ;
    private:
        std::string mKeyCalcFileOriExt;
        std::string mKeyCalcName;
        std::string mTagExist;
        std::string mTagNotExist;
        cTplValGesInit< bool > mExigCohTags;
};
cElXMLTree * ToXMLTree(const cCalcNameOnExistingTag &);

void  BinaryDumpInFile(ELISE_fp &,const cCalcNameOnExistingTag &);

void  BinaryUnDumpFromFile(cCalcNameOnExistingTag &,ELISE_fp &);

std::string  Mangling( cCalcNameOnExistingTag *);

class cCalcNameCalibAux
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalcNameCalibAux & anObj,cElXMLTree * aTree);


        std::string & KeyCalcFileOriExt();
        const std::string & KeyCalcFileOriExt()const ;

        std::string & KeyCalcName();
        const std::string & KeyCalcName()const ;

        std::string & TagExist();
        const std::string & TagExist()const ;

        std::string & TagNotExist();
        const std::string & TagNotExist()const ;

        cTplValGesInit< bool > & ExigCohTags();
        const cTplValGesInit< bool > & ExigCohTags()const ;

        cTplValGesInit< cCalcNameOnExistingTag > & CalcNameOnExistingTag();
        const cTplValGesInit< cCalcNameOnExistingTag > & CalcNameOnExistingTag()const ;

        cTplValGesInit< std::string > & KeyCalcNameDef();
        const cTplValGesInit< std::string > & KeyCalcNameDef()const ;
    private:
        cTplValGesInit< cCalcNameOnExistingTag > mCalcNameOnExistingTag;
        cTplValGesInit< std::string > mKeyCalcNameDef;
};
cElXMLTree * ToXMLTree(const cCalcNameCalibAux &);

void  BinaryDumpInFile(ELISE_fp &,const cCalcNameCalibAux &);

void  BinaryUnDumpFromFile(cCalcNameCalibAux &,ELISE_fp &);

std::string  Mangling( cCalcNameCalibAux *);

class cPosFromBDAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPosFromBDAppuis & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        int & NbTestRansac();
        const int & NbTestRansac()const ;

        cTplValGesInit< Pt3dr > & DirApprox();
        const cTplValGesInit< Pt3dr > & DirApprox()const ;
    private:
        std::string mId;
        int mNbTestRansac;
        cTplValGesInit< Pt3dr > mDirApprox;
};
cElXMLTree * ToXMLTree(const cPosFromBDAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cPosFromBDAppuis &);

void  BinaryUnDumpFromFile(cPosFromBDAppuis &,ELISE_fp &);

std::string  Mangling( cPosFromBDAppuis *);

class cLiaisonsInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLiaisonsInit & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & OnZonePlane();
        const cTplValGesInit< std::string > & OnZonePlane()const ;

        cTplValGesInit< bool > & TestSolPlane();
        const cTplValGesInit< bool > & TestSolPlane()const ;

        cTplValGesInit< int > & NbRansacSolAppui();
        const cTplValGesInit< int > & NbRansacSolAppui()const ;

        cTplValGesInit< bool > & InitOrientPure();
        const cTplValGesInit< bool > & InitOrientPure()const ;

        cTplValGesInit< int > & NbPtsRansacOrPure();
        const cTplValGesInit< int > & NbPtsRansacOrPure()const ;

        cTplValGesInit< int > & NbTestRansacOrPure();
        const cTplValGesInit< int > & NbTestRansacOrPure()const ;

        cTplValGesInit< int > & NbMinPtsRanAp();
        const cTplValGesInit< int > & NbMinPtsRanAp()const ;

        cTplValGesInit< int > & NbMaxPtsRanAp();
        const cTplValGesInit< int > & NbMaxPtsRanAp()const ;

        cTplValGesInit< double > & PropMinPtsMult();
        const cTplValGesInit< double > & PropMinPtsMult()const ;

        std::string & NameCam();
        const std::string & NameCam()const ;

        cTplValGesInit< bool > & NameCamIsKeyCalc();
        const cTplValGesInit< bool > & NameCamIsKeyCalc()const ;

        cTplValGesInit< bool > & KeyCalcIsIDir();
        const cTplValGesInit< bool > & KeyCalcIsIDir()const ;

        std::string & IdBD();
        const std::string & IdBD()const ;

        cTplValGesInit< double > & ProfSceneCouple();
        const cTplValGesInit< double > & ProfSceneCouple()const ;

        cTplValGesInit< bool > & L2EstimPlan();
        const cTplValGesInit< bool > & L2EstimPlan()const ;

        cTplValGesInit< double > & LongueurBase();
        const cTplValGesInit< double > & LongueurBase()const ;
    private:
        cTplValGesInit< std::string > mOnZonePlane;
        cTplValGesInit< bool > mTestSolPlane;
        cTplValGesInit< int > mNbRansacSolAppui;
        cTplValGesInit< bool > mInitOrientPure;
        cTplValGesInit< int > mNbPtsRansacOrPure;
        cTplValGesInit< int > mNbTestRansacOrPure;
        cTplValGesInit< int > mNbMinPtsRanAp;
        cTplValGesInit< int > mNbMaxPtsRanAp;
        cTplValGesInit< double > mPropMinPtsMult;
        std::string mNameCam;
        cTplValGesInit< bool > mNameCamIsKeyCalc;
        cTplValGesInit< bool > mKeyCalcIsIDir;
        std::string mIdBD;
        cTplValGesInit< double > mProfSceneCouple;
        cTplValGesInit< bool > mL2EstimPlan;
        cTplValGesInit< double > mLongueurBase;
};
cElXMLTree * ToXMLTree(const cLiaisonsInit &);

void  BinaryDumpInFile(ELISE_fp &,const cLiaisonsInit &);

void  BinaryUnDumpFromFile(cLiaisonsInit &,ELISE_fp &);

std::string  Mangling( cLiaisonsInit *);

class cPoseFromLiaisons
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPoseFromLiaisons & anObj,cElXMLTree * aTree);


        std::vector< cLiaisonsInit > & LiaisonsInit();
        const std::vector< cLiaisonsInit > & LiaisonsInit()const ;
    private:
        std::vector< cLiaisonsInit > mLiaisonsInit;
};
cElXMLTree * ToXMLTree(const cPoseFromLiaisons &);

void  BinaryDumpInFile(ELISE_fp &,const cPoseFromLiaisons &);

void  BinaryUnDumpFromFile(cPoseFromLiaisons &,ELISE_fp &);

std::string  Mangling( cPoseFromLiaisons *);

class cMesurePIFRP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMesurePIFRP & anObj,cElXMLTree * aTree);


        cMesureAppuis & Ap1();
        const cMesureAppuis & Ap1()const ;

        cMesureAppuis & Ap2();
        const cMesureAppuis & Ap2()const ;

        cMesureAppuis & Ap3();
        const cMesureAppuis & Ap3()const ;
    private:
        cMesureAppuis mAp1;
        cMesureAppuis mAp2;
        cMesureAppuis mAp3;
};
cElXMLTree * ToXMLTree(const cMesurePIFRP &);

void  BinaryDumpInFile(ELISE_fp &,const cMesurePIFRP &);

void  BinaryUnDumpFromFile(cMesurePIFRP &,ELISE_fp &);

std::string  Mangling( cMesurePIFRP *);

class cInitPIFRP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cInitPIFRP & anObj,cElXMLTree * aTree);


        cMesureAppuis & Ap1();
        const cMesureAppuis & Ap1()const ;

        cMesureAppuis & Ap2();
        const cMesureAppuis & Ap2()const ;

        cMesureAppuis & Ap3();
        const cMesureAppuis & Ap3()const ;

        cTplValGesInit< cMesurePIFRP > & MesurePIFRP();
        const cTplValGesInit< cMesurePIFRP > & MesurePIFRP()const ;

        cTplValGesInit< Pt3dr > & DirPlan();
        const cTplValGesInit< Pt3dr > & DirPlan()const ;
    private:
        cTplValGesInit< cMesurePIFRP > mMesurePIFRP;
        cTplValGesInit< Pt3dr > mDirPlan;
};
cElXMLTree * ToXMLTree(const cInitPIFRP &);

void  BinaryDumpInFile(ELISE_fp &,const cInitPIFRP &);

void  BinaryUnDumpFromFile(cInitPIFRP &,ELISE_fp &);

std::string  Mangling( cInitPIFRP *);

class cPoseInitFromReperePlan
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPoseInitFromReperePlan & anObj,cElXMLTree * aTree);


        std::string & OnZonePlane();
        const std::string & OnZonePlane()const ;

        cTplValGesInit< bool > & L2EstimPlan();
        const cTplValGesInit< bool > & L2EstimPlan()const ;

        std::string & IdBD();
        const std::string & IdBD()const ;

        std::string & NameCam();
        const std::string & NameCam()const ;

        cTplValGesInit< double > & DEuclidPlan();
        const cTplValGesInit< double > & DEuclidPlan()const ;

        cMesureAppuis & Ap1();
        const cMesureAppuis & Ap1()const ;

        cMesureAppuis & Ap2();
        const cMesureAppuis & Ap2()const ;

        cMesureAppuis & Ap3();
        const cMesureAppuis & Ap3()const ;

        cTplValGesInit< cMesurePIFRP > & MesurePIFRP();
        const cTplValGesInit< cMesurePIFRP > & MesurePIFRP()const ;

        cTplValGesInit< Pt3dr > & DirPlan();
        const cTplValGesInit< Pt3dr > & DirPlan()const ;

        cInitPIFRP & InitPIFRP();
        const cInitPIFRP & InitPIFRP()const ;
    private:
        std::string mOnZonePlane;
        cTplValGesInit< bool > mL2EstimPlan;
        std::string mIdBD;
        std::string mNameCam;
        cTplValGesInit< double > mDEuclidPlan;
        cInitPIFRP mInitPIFRP;
};
cElXMLTree * ToXMLTree(const cPoseInitFromReperePlan &);

void  BinaryDumpInFile(ELISE_fp &,const cPoseInitFromReperePlan &);

void  BinaryUnDumpFromFile(cPoseInitFromReperePlan &,ELISE_fp &);

std::string  Mangling( cPoseInitFromReperePlan *);

class cPosValueInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPosValueInit & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PosId();
        const cTplValGesInit< std::string > & PosId()const ;

        cTplValGesInit< std::string > & PosFromBDOrient();
        const cTplValGesInit< std::string > & PosFromBDOrient()const ;

        cTplValGesInit< std::string > & PosFromBlockRigid();
        const cTplValGesInit< std::string > & PosFromBlockRigid()const ;

        std::string & Id();
        const std::string & Id()const ;

        int & NbTestRansac();
        const int & NbTestRansac()const ;

        cTplValGesInit< Pt3dr > & DirApprox();
        const cTplValGesInit< Pt3dr > & DirApprox()const ;

        cTplValGesInit< cPosFromBDAppuis > & PosFromBDAppuis();
        const cTplValGesInit< cPosFromBDAppuis > & PosFromBDAppuis()const ;

        std::vector< cLiaisonsInit > & LiaisonsInit();
        const std::vector< cLiaisonsInit > & LiaisonsInit()const ;

        cTplValGesInit< cPoseFromLiaisons > & PoseFromLiaisons();
        const cTplValGesInit< cPoseFromLiaisons > & PoseFromLiaisons()const ;

        std::string & OnZonePlane();
        const std::string & OnZonePlane()const ;

        cTplValGesInit< bool > & L2EstimPlan();
        const cTplValGesInit< bool > & L2EstimPlan()const ;

        std::string & IdBD();
        const std::string & IdBD()const ;

        std::string & NameCam();
        const std::string & NameCam()const ;

        cTplValGesInit< double > & DEuclidPlan();
        const cTplValGesInit< double > & DEuclidPlan()const ;

        cMesureAppuis & Ap1();
        const cMesureAppuis & Ap1()const ;

        cMesureAppuis & Ap2();
        const cMesureAppuis & Ap2()const ;

        cMesureAppuis & Ap3();
        const cMesureAppuis & Ap3()const ;

        cTplValGesInit< cMesurePIFRP > & MesurePIFRP();
        const cTplValGesInit< cMesurePIFRP > & MesurePIFRP()const ;

        cTplValGesInit< Pt3dr > & DirPlan();
        const cTplValGesInit< Pt3dr > & DirPlan()const ;

        cInitPIFRP & InitPIFRP();
        const cInitPIFRP & InitPIFRP()const ;

        cTplValGesInit< cPoseInitFromReperePlan > & PoseInitFromReperePlan();
        const cTplValGesInit< cPoseInitFromReperePlan > & PoseInitFromReperePlan()const ;
    private:
        cTplValGesInit< std::string > mPosId;
        cTplValGesInit< std::string > mPosFromBDOrient;
        cTplValGesInit< std::string > mPosFromBlockRigid;
        cTplValGesInit< cPosFromBDAppuis > mPosFromBDAppuis;
        cTplValGesInit< cPoseFromLiaisons > mPoseFromLiaisons;
        cTplValGesInit< cPoseInitFromReperePlan > mPoseInitFromReperePlan;
};
cElXMLTree * ToXMLTree(const cPosValueInit &);

void  BinaryDumpInFile(ELISE_fp &,const cPosValueInit &);

void  BinaryUnDumpFromFile(cPosValueInit &,ELISE_fp &);

std::string  Mangling( cPosValueInit *);

class cPoseCameraInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPoseCameraInc & anObj,cElXMLTree * aTree);


        cTplValGesInit< cSetOrientationInterne > & OrInterne();
        const cTplValGesInit< cSetOrientationInterne > & OrInterne()const ;

        cTplValGesInit< std::string > & IdBDCentre();
        const cTplValGesInit< std::string > & IdBDCentre()const ;

        cTplValGesInit< std::string > & IdOffsetGPS();
        const cTplValGesInit< std::string > & IdOffsetGPS()const ;

        cTplValGesInit< bool > & InitNow();
        const cTplValGesInit< bool > & InitNow()const ;

        cTplValGesInit< double > & ProfSceneImage();
        const cTplValGesInit< double > & ProfSceneImage()const ;

        cTplValGesInit< std::string > & Directory();
        const cTplValGesInit< std::string > & Directory()const ;

        std::list< std::string > & PatternName();
        const std::list< std::string > & PatternName()const ;

        cTplValGesInit< std::string > & AutomGetImC();
        const cTplValGesInit< std::string > & AutomGetImC()const ;

        cTplValGesInit< std::string > & TestFewerTiePoints();
        const cTplValGesInit< std::string > & TestFewerTiePoints()const ;

        cTplValGesInit< cNameFilter > & Filter();
        const cTplValGesInit< cNameFilter > & Filter()const ;

        cTplValGesInit< cElRegex_Ptr > & PatternRefuteur();
        const cTplValGesInit< cElRegex_Ptr > & PatternRefuteur()const ;

        cTplValGesInit< bool > & AutoRefutDupl();
        const cTplValGesInit< bool > & AutoRefutDupl()const ;

        cTplValGesInit< std::string > & KeyTranscriptionName();
        const cTplValGesInit< std::string > & KeyTranscriptionName()const ;

        cTplValGesInit< std::string > & AddAllNameConnectedBy();
        const cTplValGesInit< std::string > & AddAllNameConnectedBy()const ;

        cTplValGesInit< std::string > & FilterConnecBy();
        const cTplValGesInit< std::string > & FilterConnecBy()const ;

        cTplValGesInit< std::string > & MSTBlockRigid();
        const cTplValGesInit< std::string > & MSTBlockRigid()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< int > & MinNbPtsInit();
        const cTplValGesInit< int > & MinNbPtsInit()const ;

        cTplValGesInit< double > & ExpDist();
        const cTplValGesInit< double > & ExpDist()const ;

        cTplValGesInit< double > & ExpNb();
        const cTplValGesInit< double > & ExpNb()const ;

        cTplValGesInit< bool > & MontageOnInit();
        const cTplValGesInit< bool > & MontageOnInit()const ;

        cTplValGesInit< int > & NbInitMinBeforeUnconnect();
        const cTplValGesInit< int > & NbInitMinBeforeUnconnect()const ;

        cTplValGesInit< cMEP_SPEC_MST > & MEP_SPEC_MST();
        const cTplValGesInit< cMEP_SPEC_MST > & MEP_SPEC_MST()const ;

        cOptimizationPowel & ParamOptim();
        const cOptimizationPowel & ParamOptim()const ;

        std::list< cApplyOAI > & ApplyOAI();
        const std::list< cApplyOAI > & ApplyOAI()const ;

        cTplValGesInit< cOptimizeAfterInit > & OptimizeAfterInit();
        const cTplValGesInit< cOptimizeAfterInit > & OptimizeAfterInit()const ;

        cTplValGesInit< bool > & ReverseOrderName();
        const cTplValGesInit< bool > & ReverseOrderName()const ;

        cTplValGesInit< std::string > & CalcNameCalib();
        const cTplValGesInit< std::string > & CalcNameCalib()const ;

        std::list< cCalcNameCalibAux > & CalcNameCalibAux();
        const std::list< cCalcNameCalibAux > & CalcNameCalibAux()const ;

        cTplValGesInit< std::string > & PosesDeRattachement();
        const cTplValGesInit< std::string > & PosesDeRattachement()const ;

        cTplValGesInit< bool > & NoErroOnRat();
        const cTplValGesInit< bool > & NoErroOnRat()const ;

        cTplValGesInit< bool > & ByPattern();
        const cTplValGesInit< bool > & ByPattern()const ;

        cTplValGesInit< std::string > & KeyFilterExistingFile();
        const cTplValGesInit< std::string > & KeyFilterExistingFile()const ;

        cTplValGesInit< bool > & ByKey();
        const cTplValGesInit< bool > & ByKey()const ;

        cTplValGesInit< bool > & ByFile();
        const cTplValGesInit< bool > & ByFile()const ;

        cTplValGesInit< std::string > & PosId();
        const cTplValGesInit< std::string > & PosId()const ;

        cTplValGesInit< std::string > & PosFromBDOrient();
        const cTplValGesInit< std::string > & PosFromBDOrient()const ;

        cTplValGesInit< std::string > & PosFromBlockRigid();
        const cTplValGesInit< std::string > & PosFromBlockRigid()const ;

        std::string & Id();
        const std::string & Id()const ;

        int & NbTestRansac();
        const int & NbTestRansac()const ;

        cTplValGesInit< Pt3dr > & DirApprox();
        const cTplValGesInit< Pt3dr > & DirApprox()const ;

        cTplValGesInit< cPosFromBDAppuis > & PosFromBDAppuis();
        const cTplValGesInit< cPosFromBDAppuis > & PosFromBDAppuis()const ;

        std::vector< cLiaisonsInit > & LiaisonsInit();
        const std::vector< cLiaisonsInit > & LiaisonsInit()const ;

        cTplValGesInit< cPoseFromLiaisons > & PoseFromLiaisons();
        const cTplValGesInit< cPoseFromLiaisons > & PoseFromLiaisons()const ;

        std::string & OnZonePlane();
        const std::string & OnZonePlane()const ;

        cTplValGesInit< bool > & L2EstimPlan();
        const cTplValGesInit< bool > & L2EstimPlan()const ;

        std::string & IdBD();
        const std::string & IdBD()const ;

        std::string & NameCam();
        const std::string & NameCam()const ;

        cTplValGesInit< double > & DEuclidPlan();
        const cTplValGesInit< double > & DEuclidPlan()const ;

        cMesureAppuis & Ap1();
        const cMesureAppuis & Ap1()const ;

        cMesureAppuis & Ap2();
        const cMesureAppuis & Ap2()const ;

        cMesureAppuis & Ap3();
        const cMesureAppuis & Ap3()const ;

        cTplValGesInit< cMesurePIFRP > & MesurePIFRP();
        const cTplValGesInit< cMesurePIFRP > & MesurePIFRP()const ;

        cTplValGesInit< Pt3dr > & DirPlan();
        const cTplValGesInit< Pt3dr > & DirPlan()const ;

        cInitPIFRP & InitPIFRP();
        const cInitPIFRP & InitPIFRP()const ;

        cTplValGesInit< cPoseInitFromReperePlan > & PoseInitFromReperePlan();
        const cTplValGesInit< cPoseInitFromReperePlan > & PoseInitFromReperePlan()const ;

        cPosValueInit & PosValueInit();
        const cPosValueInit & PosValueInit()const ;
    private:
        cTplValGesInit< cSetOrientationInterne > mOrInterne;
        cTplValGesInit< std::string > mIdBDCentre;
        cTplValGesInit< std::string > mIdOffsetGPS;
        cTplValGesInit< bool > mInitNow;
        cTplValGesInit< double > mProfSceneImage;
        cTplValGesInit< std::string > mDirectory;
        std::list< std::string > mPatternName;
        cTplValGesInit< std::string > mAutomGetImC;
        cTplValGesInit< std::string > mTestFewerTiePoints;
        cTplValGesInit< cNameFilter > mFilter;
        cTplValGesInit< cElRegex_Ptr > mPatternRefuteur;
        cTplValGesInit< bool > mAutoRefutDupl;
        cTplValGesInit< std::string > mKeyTranscriptionName;
        cTplValGesInit< std::string > mAddAllNameConnectedBy;
        cTplValGesInit< std::string > mFilterConnecBy;
        cTplValGesInit< cMEP_SPEC_MST > mMEP_SPEC_MST;
        cTplValGesInit< cOptimizeAfterInit > mOptimizeAfterInit;
        cTplValGesInit< bool > mReverseOrderName;
        cTplValGesInit< std::string > mCalcNameCalib;
        std::list< cCalcNameCalibAux > mCalcNameCalibAux;
        cTplValGesInit< std::string > mPosesDeRattachement;
        cTplValGesInit< bool > mNoErroOnRat;
        cTplValGesInit< bool > mByPattern;
        cTplValGesInit< std::string > mKeyFilterExistingFile;
        cTplValGesInit< bool > mByKey;
        cTplValGesInit< bool > mByFile;
        cPosValueInit mPosValueInit;
};
cElXMLTree * ToXMLTree(const cPoseCameraInc &);

void  BinaryDumpInFile(ELISE_fp &,const cPoseCameraInc &);

void  BinaryUnDumpFromFile(cPoseCameraInc &,ELISE_fp &);

std::string  Mangling( cPoseCameraInc *);

class cGroupeDePose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGroupeDePose & anObj,cElXMLTree * aTree);


        std::string & KeyPose2Grp();
        const std::string & KeyPose2Grp()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< bool > & ShowCreate();
        const cTplValGesInit< bool > & ShowCreate()const ;
    private:
        std::string mKeyPose2Grp;
        std::string mId;
        cTplValGesInit< bool > mShowCreate;
};
cElXMLTree * ToXMLTree(const cGroupeDePose &);

void  BinaryDumpInFile(ELISE_fp &,const cGroupeDePose &);

void  BinaryUnDumpFromFile(cGroupeDePose &,ELISE_fp &);

std::string  Mangling( cGroupeDePose *);

class cLiaisonsApplyContrainte
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLiaisonsApplyContrainte & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;

        std::string & PatternI1();
        const std::string & PatternI1()const ;

        cTplValGesInit< std::string > & PatternI2();
        const cTplValGesInit< std::string > & PatternI2()const ;
    private:
        std::string mNameRef;
        std::string mPatternI1;
        cTplValGesInit< std::string > mPatternI2;
};
cElXMLTree * ToXMLTree(const cLiaisonsApplyContrainte &);

void  BinaryDumpInFile(ELISE_fp &,const cLiaisonsApplyContrainte &);

void  BinaryUnDumpFromFile(cLiaisonsApplyContrainte &,ELISE_fp &);

std::string  Mangling( cLiaisonsApplyContrainte *);

class cInitSurf
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cInitSurf & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & ZonePlane();
        const cTplValGesInit< std::string > & ZonePlane()const ;
    private:
        cTplValGesInit< std::string > mZonePlane;
};
cElXMLTree * ToXMLTree(const cInitSurf &);

void  BinaryDumpInFile(ELISE_fp &,const cInitSurf &);

void  BinaryUnDumpFromFile(cInitSurf &,ELISE_fp &);

std::string  Mangling( cInitSurf *);

class cSurfParamInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSurfParamInc & anObj,cElXMLTree * aTree);


        std::list< cLiaisonsApplyContrainte > & LiaisonsApplyContrainte();
        const std::list< cLiaisonsApplyContrainte > & LiaisonsApplyContrainte()const ;

        cTplValGesInit< std::string > & ZonePlane();
        const cTplValGesInit< std::string > & ZonePlane()const ;

        cInitSurf & InitSurf();
        const cInitSurf & InitSurf()const ;
    private:
        std::list< cLiaisonsApplyContrainte > mLiaisonsApplyContrainte;
        cInitSurf mInitSurf;
};
cElXMLTree * ToXMLTree(const cSurfParamInc &);

void  BinaryDumpInFile(ELISE_fp &,const cSurfParamInc &);

void  BinaryUnDumpFromFile(cSurfParamInc &,ELISE_fp &);

std::string  Mangling( cSurfParamInc *);

class cPointFlottantInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPointFlottantInc & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & KeySetOrPat();
        const std::string & KeySetOrPat()const ;

        cTplValGesInit< cModifIncPtsFlottant > & ModifInc();
        const cTplValGesInit< cModifIncPtsFlottant > & ModifInc()const ;
    private:
        std::string mId;
        std::string mKeySetOrPat;
        cTplValGesInit< cModifIncPtsFlottant > mModifInc;
};
cElXMLTree * ToXMLTree(const cPointFlottantInc &);

void  BinaryDumpInFile(ELISE_fp &,const cPointFlottantInc &);

void  BinaryUnDumpFromFile(cPointFlottantInc &,ELISE_fp &);

std::string  Mangling( cPointFlottantInc *);

class cSectionInconnues
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionInconnues & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & SeuilAutomFE();
        const cTplValGesInit< double > & SeuilAutomFE()const ;

        cTplValGesInit< bool > & AutoriseToujoursUneSeuleLiaison();
        const cTplValGesInit< bool > & AutoriseToujoursUneSeuleLiaison()const ;

        cTplValGesInit< cMapName2Name > & MapMaskHom();
        const cTplValGesInit< cMapName2Name > & MapMaskHom()const ;

        cTplValGesInit< bool > & SauvePMoyenOnlyWithMasq();
        const cTplValGesInit< bool > & SauvePMoyenOnlyWithMasq()const ;

        std::list< cGpsOffset > & GpsOffset();
        const std::list< cGpsOffset > & GpsOffset()const ;

        std::list< cDataObsPlane > & DataObsPlane();
        const std::list< cDataObsPlane > & DataObsPlane()const ;

        std::list< cCalibrationCameraInc > & CalibrationCameraInc();
        const std::list< cCalibrationCameraInc > & CalibrationCameraInc()const ;

        cTplValGesInit< int > & SeuilL1EstimMatrEss();
        const cTplValGesInit< int > & SeuilL1EstimMatrEss()const ;

        std::list< cBlockCamera > & BlockCamera();
        const std::list< cBlockCamera > & BlockCamera()const ;

        cTplValGesInit< cSetOrientationInterne > & GlobOrInterne();
        const cTplValGesInit< cSetOrientationInterne > & GlobOrInterne()const ;

        std::list< cCamGenInc > & CamGenInc();
        const std::list< cCamGenInc > & CamGenInc()const ;

        std::list< cPoseCameraInc > & PoseCameraInc();
        const std::list< cPoseCameraInc > & PoseCameraInc()const ;

        std::list< cGroupeDePose > & GroupeDePose();
        const std::list< cGroupeDePose > & GroupeDePose()const ;

        std::list< cSurfParamInc > & SurfParamInc();
        const std::list< cSurfParamInc > & SurfParamInc()const ;

        std::list< cPointFlottantInc > & PointFlottantInc();
        const std::list< cPointFlottantInc > & PointFlottantInc()const ;
    private:
        cTplValGesInit< double > mSeuilAutomFE;
        cTplValGesInit< bool > mAutoriseToujoursUneSeuleLiaison;
        cTplValGesInit< cMapName2Name > mMapMaskHom;
        cTplValGesInit< bool > mSauvePMoyenOnlyWithMasq;
        std::list< cGpsOffset > mGpsOffset;
        std::list< cDataObsPlane > mDataObsPlane;
        std::list< cCalibrationCameraInc > mCalibrationCameraInc;
        cTplValGesInit< int > mSeuilL1EstimMatrEss;
        std::list< cBlockCamera > mBlockCamera;
        cTplValGesInit< cSetOrientationInterne > mGlobOrInterne;
        std::list< cCamGenInc > mCamGenInc;
        std::list< cPoseCameraInc > mPoseCameraInc;
        std::list< cGroupeDePose > mGroupeDePose;
        std::list< cSurfParamInc > mSurfParamInc;
        std::list< cPointFlottantInc > mPointFlottantInc;
};
cElXMLTree * ToXMLTree(const cSectionInconnues &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionInconnues &);

void  BinaryUnDumpFromFile(cSectionInconnues &,ELISE_fp &);

std::string  Mangling( cSectionInconnues *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRappelPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRappelPose & anObj,cElXMLTree * aTree);


        std::string & IdOrient();
        const std::string & IdOrient()const ;

        double & SigmaC();
        const double & SigmaC()const ;

        double & SigmaR();
        const double & SigmaR()const ;

        cElRegex_Ptr & PatternApply();
        const cElRegex_Ptr & PatternApply()const ;
    private:
        std::string mIdOrient;
        double mSigmaC;
        double mSigmaR;
        cElRegex_Ptr mPatternApply;
};
cElXMLTree * ToXMLTree(const cRappelPose &);

void  BinaryDumpInFile(ELISE_fp &,const cRappelPose &);

void  BinaryUnDumpFromFile(cRappelPose &,ELISE_fp &);

std::string  Mangling( cRappelPose *);

class cUseExportImageResidu
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cUseExportImageResidu & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & SzByPair();
        const cTplValGesInit< double > & SzByPair()const ;

        cTplValGesInit< double > & SzByPose();
        const cTplValGesInit< double > & SzByPose()const ;

        cTplValGesInit< double > & SzByCam();
        const cTplValGesInit< double > & SzByCam()const ;

        cTplValGesInit< double > & NbMesByCase();
        const cTplValGesInit< double > & NbMesByCase()const ;

        std::string & AeroExport();
        const std::string & AeroExport()const ;

        cTplValGesInit< bool > & GeneratePly();
        const cTplValGesInit< bool > & GeneratePly()const ;

        cTplValGesInit< int > & SzOrtho();
        const cTplValGesInit< int > & SzOrtho()const ;
    private:
        cTplValGesInit< double > mSzByPair;
        cTplValGesInit< double > mSzByPose;
        cTplValGesInit< double > mSzByCam;
        cTplValGesInit< double > mNbMesByCase;
        std::string mAeroExport;
        cTplValGesInit< bool > mGeneratePly;
        cTplValGesInit< int > mSzOrtho;
};
cElXMLTree * ToXMLTree(const cUseExportImageResidu &);

void  BinaryDumpInFile(ELISE_fp &,const cUseExportImageResidu &);

void  BinaryUnDumpFromFile(cUseExportImageResidu &,ELISE_fp &);

std::string  Mangling( cUseExportImageResidu *);

class cTimeLinkage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTimeLinkage & anObj,cElXMLTree * aTree);


        double & DeltaMax();
        const double & DeltaMax()const ;
    private:
        double mDeltaMax;
};
cElXMLTree * ToXMLTree(const cTimeLinkage &);

void  BinaryDumpInFile(ELISE_fp &,const cTimeLinkage &);

void  BinaryUnDumpFromFile(cTimeLinkage &,ELISE_fp &);

std::string  Mangling( cTimeLinkage *);

class cSectionChantier
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionChantier & anObj,cElXMLTree * aTree);


        std::string & IdOrient();
        const std::string & IdOrient()const ;

        double & SigmaC();
        const double & SigmaC()const ;

        double & SigmaR();
        const double & SigmaR()const ;

        cElRegex_Ptr & PatternApply();
        const cElRegex_Ptr & PatternApply()const ;

        cTplValGesInit< cRappelPose > & RappelPose();
        const cTplValGesInit< cRappelPose > & RappelPose()const ;

        cTplValGesInit< int > & NumAttrPdsNewF();
        const cTplValGesInit< int > & NumAttrPdsNewF()const ;

        cTplValGesInit< double > & RatioMaxDistCS();
        const cTplValGesInit< double > & RatioMaxDistCS()const ;

        cTplValGesInit< std::string > & DebugVecElimTieP();
        const cTplValGesInit< std::string > & DebugVecElimTieP()const ;

        cTplValGesInit< int > & DoStatElimBundle();
        const cTplValGesInit< int > & DoStatElimBundle()const ;

        cTplValGesInit< double > & SzByPair();
        const cTplValGesInit< double > & SzByPair()const ;

        cTplValGesInit< double > & SzByPose();
        const cTplValGesInit< double > & SzByPose()const ;

        cTplValGesInit< double > & SzByCam();
        const cTplValGesInit< double > & SzByCam()const ;

        cTplValGesInit< double > & NbMesByCase();
        const cTplValGesInit< double > & NbMesByCase()const ;

        std::string & AeroExport();
        const std::string & AeroExport()const ;

        cTplValGesInit< bool > & GeneratePly();
        const cTplValGesInit< bool > & GeneratePly()const ;

        cTplValGesInit< int > & SzOrtho();
        const cTplValGesInit< int > & SzOrtho()const ;

        cTplValGesInit< cUseExportImageResidu > & UseExportImageResidu();
        const cTplValGesInit< cUseExportImageResidu > & UseExportImageResidu()const ;

        cTplValGesInit< bool > & UseRegulDist();
        const cTplValGesInit< bool > & UseRegulDist()const ;

        cTplValGesInit< bool > & GBCamSupresStenCam();
        const cTplValGesInit< bool > & GBCamSupresStenCam()const ;

        cTplValGesInit< bool > & StenCamSupresGBCam();
        const cTplValGesInit< bool > & StenCamSupresGBCam()const ;

        cTplValGesInit< bool > & IsAperiCloud();
        const cTplValGesInit< bool > & IsAperiCloud()const ;

        cTplValGesInit< bool > & IsChoixImSec();
        const cTplValGesInit< bool > & IsChoixImSec()const ;

        cTplValGesInit< std::string > & FileSauvParam();
        const cTplValGesInit< std::string > & FileSauvParam()const ;

        cTplValGesInit< bool > & GenereErreurOnContraineCam();
        const cTplValGesInit< bool > & GenereErreurOnContraineCam()const ;

        cTplValGesInit< double > & ProfSceneChantier();
        const cTplValGesInit< double > & ProfSceneChantier()const ;

        cTplValGesInit< std::string > & DirectoryChantier();
        const cTplValGesInit< std::string > & DirectoryChantier()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        cTplValGesInit< std::string > & NameParamEtal();
        const cTplValGesInit< std::string > & NameParamEtal()const ;

        cTplValGesInit< std::string > & PatternTracePose();
        const cTplValGesInit< std::string > & PatternTracePose()const ;

        cTplValGesInit< bool > & TraceGimbalLock();
        const cTplValGesInit< bool > & TraceGimbalLock()const ;

        cTplValGesInit< double > & MaxDistErrorPtsTerr();
        const cTplValGesInit< double > & MaxDistErrorPtsTerr()const ;

        cTplValGesInit< double > & MaxDistWarnPtsTerr();
        const cTplValGesInit< double > & MaxDistWarnPtsTerr()const ;

        cTplValGesInit< cShowPbLiaison > & DefPbLiaison();
        const cTplValGesInit< cShowPbLiaison > & DefPbLiaison()const ;

        cTplValGesInit< bool > & DoCompensation();
        const cTplValGesInit< bool > & DoCompensation()const ;

        double & DeltaMax();
        const double & DeltaMax()const ;

        cTplValGesInit< cTimeLinkage > & TimeLinkage();
        const cTplValGesInit< cTimeLinkage > & TimeLinkage()const ;

        cTplValGesInit< bool > & DebugPbCondFaisceau();
        const cTplValGesInit< bool > & DebugPbCondFaisceau()const ;

        cTplValGesInit< std::string > & SauvAutom();
        const cTplValGesInit< std::string > & SauvAutom()const ;

        cTplValGesInit< bool > & SauvAutomBasic();
        const cTplValGesInit< bool > & SauvAutomBasic()const ;

        cTplValGesInit< double > & ThresholdWarnPointsBehind();
        const cTplValGesInit< double > & ThresholdWarnPointsBehind()const ;

        cTplValGesInit< bool > & ExportMatrixMarket();
        const cTplValGesInit< bool > & ExportMatrixMarket()const ;

        cTplValGesInit< double > & ExtensionIntervZ();
        const cTplValGesInit< double > & ExtensionIntervZ()const ;
    private:
        cTplValGesInit< cRappelPose > mRappelPose;
        cTplValGesInit< int > mNumAttrPdsNewF;
        cTplValGesInit< double > mRatioMaxDistCS;
        cTplValGesInit< std::string > mDebugVecElimTieP;
        cTplValGesInit< int > mDoStatElimBundle;
        cTplValGesInit< cUseExportImageResidu > mUseExportImageResidu;
        cTplValGesInit< bool > mUseRegulDist;
        cTplValGesInit< bool > mGBCamSupresStenCam;
        cTplValGesInit< bool > mStenCamSupresGBCam;
        cTplValGesInit< bool > mIsAperiCloud;
        cTplValGesInit< bool > mIsChoixImSec;
        cTplValGesInit< std::string > mFileSauvParam;
        cTplValGesInit< bool > mGenereErreurOnContraineCam;
        cTplValGesInit< double > mProfSceneChantier;
        cTplValGesInit< std::string > mDirectoryChantier;
        cTplValGesInit< string > mFileChantierNameDescripteur;
        cTplValGesInit< std::string > mNameParamEtal;
        cTplValGesInit< std::string > mPatternTracePose;
        cTplValGesInit< bool > mTraceGimbalLock;
        cTplValGesInit< double > mMaxDistErrorPtsTerr;
        cTplValGesInit< double > mMaxDistWarnPtsTerr;
        cTplValGesInit< cShowPbLiaison > mDefPbLiaison;
        cTplValGesInit< bool > mDoCompensation;
        cTplValGesInit< cTimeLinkage > mTimeLinkage;
        cTplValGesInit< bool > mDebugPbCondFaisceau;
        cTplValGesInit< std::string > mSauvAutom;
        cTplValGesInit< bool > mSauvAutomBasic;
        cTplValGesInit< double > mThresholdWarnPointsBehind;
        cTplValGesInit< bool > mExportMatrixMarket;
        cTplValGesInit< double > mExtensionIntervZ;
};
cElXMLTree * ToXMLTree(const cSectionChantier &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionChantier &);

void  BinaryUnDumpFromFile(cSectionChantier &,ELISE_fp &);

std::string  Mangling( cSectionChantier *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionSolveur
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionSolveur & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & AllMatSym();
        const cTplValGesInit< bool > & AllMatSym()const ;

        eModeSolveurEq & ModeResolution();
        const eModeSolveurEq & ModeResolution()const ;

        cTplValGesInit< eControleDescDic > & ModeControleDescDic();
        const cTplValGesInit< eControleDescDic > & ModeControleDescDic()const ;

        cTplValGesInit< int > & SeuilBas_CDD();
        const cTplValGesInit< int > & SeuilBas_CDD()const ;

        cTplValGesInit< int > & SeuilHaut_CDD();
        const cTplValGesInit< int > & SeuilHaut_CDD()const ;

        cTplValGesInit< bool > & InhibeAMD();
        const cTplValGesInit< bool > & InhibeAMD()const ;

        cTplValGesInit< bool > & AMDSpecInterne();
        const cTplValGesInit< bool > & AMDSpecInterne()const ;

        cTplValGesInit< bool > & ShowCholesky();
        const cTplValGesInit< bool > & ShowCholesky()const ;

        cTplValGesInit< bool > & TestPermutVar();
        const cTplValGesInit< bool > & TestPermutVar()const ;

        cTplValGesInit< bool > & ShowPermutVar();
        const cTplValGesInit< bool > & ShowPermutVar()const ;

        cTplValGesInit< bool > & PermutIndex();
        const cTplValGesInit< bool > & PermutIndex()const ;

        cTplValGesInit< bool > & NormaliseEqSc();
        const cTplValGesInit< bool > & NormaliseEqSc()const ;

        cTplValGesInit< bool > & NormaliseEqTr();
        const cTplValGesInit< bool > & NormaliseEqTr()const ;

        cTplValGesInit< double > & LimBsHProj();
        const cTplValGesInit< double > & LimBsHProj()const ;

        cTplValGesInit< double > & LimBsHRefut();
        const cTplValGesInit< double > & LimBsHRefut()const ;

        cTplValGesInit< double > & LimModeGL();
        const cTplValGesInit< double > & LimModeGL()const ;

        cTplValGesInit< bool > & GridOptimKnownDist();
        const cTplValGesInit< bool > & GridOptimKnownDist()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob()const ;

        cTplValGesInit< double > & MultSLMGlob();
        const cTplValGesInit< double > & MultSLMGlob()const ;

        cTplValGesInit< cElRegex_Ptr > & Im2Aff();
        const cTplValGesInit< cElRegex_Ptr > & Im2Aff()const ;

        cTplValGesInit< cXmlPondRegDist > & RegDistGlob();
        const cTplValGesInit< cXmlPondRegDist > & RegDistGlob()const ;
    private:
        cTplValGesInit< bool > mAllMatSym;
        eModeSolveurEq mModeResolution;
        cTplValGesInit< eControleDescDic > mModeControleDescDic;
        cTplValGesInit< int > mSeuilBas_CDD;
        cTplValGesInit< int > mSeuilHaut_CDD;
        cTplValGesInit< bool > mInhibeAMD;
        cTplValGesInit< bool > mAMDSpecInterne;
        cTplValGesInit< bool > mShowCholesky;
        cTplValGesInit< bool > mTestPermutVar;
        cTplValGesInit< bool > mShowPermutVar;
        cTplValGesInit< bool > mPermutIndex;
        cTplValGesInit< bool > mNormaliseEqSc;
        cTplValGesInit< bool > mNormaliseEqTr;
        cTplValGesInit< double > mLimBsHProj;
        cTplValGesInit< double > mLimBsHRefut;
        cTplValGesInit< double > mLimModeGL;
        cTplValGesInit< bool > mGridOptimKnownDist;
        cTplValGesInit< cSectionLevenbergMarkard > mSLMGlob;
        cTplValGesInit< double > mMultSLMGlob;
        cTplValGesInit< cElRegex_Ptr > mIm2Aff;
        cTplValGesInit< cXmlPondRegDist > mRegDistGlob;
};
cElXMLTree * ToXMLTree(const cSectionSolveur &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionSolveur &);

void  BinaryUnDumpFromFile(cSectionSolveur &,ELISE_fp &);

std::string  Mangling( cSectionSolveur *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAutoAdaptLVM
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAutoAdaptLVM & anObj,cElXMLTree * aTree);


        double & Mult();
        const double & Mult()const ;

        cTplValGesInit< bool > & ModeMin();
        const cTplValGesInit< bool > & ModeMin()const ;
    private:
        double mMult;
        cTplValGesInit< bool > mModeMin;
};
cElXMLTree * ToXMLTree(const cAutoAdaptLVM &);

void  BinaryDumpInFile(ELISE_fp &,const cAutoAdaptLVM &);

void  BinaryUnDumpFromFile(cAutoAdaptLVM &,ELISE_fp &);

std::string  Mangling( cAutoAdaptLVM *);

class cCtrlTimeCompens
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCtrlTimeCompens & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NbMin();
        const cTplValGesInit< int > & NbMin()const ;

        int & NbMax();
        const int & NbMax()const ;

        double & SeuilEvolMoy();
        const double & SeuilEvolMoy()const ;

        cTplValGesInit< double > & SeuilEvolMax();
        const cTplValGesInit< double > & SeuilEvolMax()const ;

        double & Mult();
        const double & Mult()const ;

        cTplValGesInit< bool > & ModeMin();
        const cTplValGesInit< bool > & ModeMin()const ;

        cTplValGesInit< cAutoAdaptLVM > & AutoAdaptLVM();
        const cTplValGesInit< cAutoAdaptLVM > & AutoAdaptLVM()const ;
    private:
        cTplValGesInit< int > mNbMin;
        int mNbMax;
        double mSeuilEvolMoy;
        cTplValGesInit< double > mSeuilEvolMax;
        cTplValGesInit< cAutoAdaptLVM > mAutoAdaptLVM;
};
cElXMLTree * ToXMLTree(const cCtrlTimeCompens &);

void  BinaryDumpInFile(ELISE_fp &,const cCtrlTimeCompens &);

void  BinaryUnDumpFromFile(cCtrlTimeCompens &,ELISE_fp &);

std::string  Mangling( cCtrlTimeCompens *);

class cPose2Init
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPose2Init & anObj,cElXMLTree * aTree);


        std::vector<int> & ProfMin();
        const std::vector<int> & ProfMin()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< int > & StepComplemAuto();
        const cTplValGesInit< int > & StepComplemAuto()const ;
    private:
        std::vector<int> mProfMin;
        cTplValGesInit< bool > mShow;
        cTplValGesInit< int > mStepComplemAuto;
};
cElXMLTree * ToXMLTree(const cPose2Init &);

void  BinaryDumpInFile(ELISE_fp &,const cPose2Init &);

void  BinaryUnDumpFromFile(cPose2Init &,ELISE_fp &);

std::string  Mangling( cPose2Init *);

class cSetRayMaxUtileCalib
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetRayMaxUtileCalib & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        double & Ray();
        const double & Ray()const ;

        cTplValGesInit< bool > & IsRelatifDiag();
        const cTplValGesInit< bool > & IsRelatifDiag()const ;

        cTplValGesInit< bool > & ApplyOnlyFE();
        const cTplValGesInit< bool > & ApplyOnlyFE()const ;
    private:
        std::string mName;
        double mRay;
        cTplValGesInit< bool > mIsRelatifDiag;
        cTplValGesInit< bool > mApplyOnlyFE;
};
cElXMLTree * ToXMLTree(const cSetRayMaxUtileCalib &);

void  BinaryDumpInFile(ELISE_fp &,const cSetRayMaxUtileCalib &);

void  BinaryUnDumpFromFile(cSetRayMaxUtileCalib &,ELISE_fp &);

std::string  Mangling( cSetRayMaxUtileCalib *);

class cBascOnCentre
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBascOnCentre & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< bool > & EstimateSpeed();
        const cTplValGesInit< bool > & EstimateSpeed()const ;

        cTplValGesInit< double > & ForceVertical();
        const cTplValGesInit< double > & ForceVertical()const ;
    private:
        cTplValGesInit< std::string > mPoseCentrale;
        cTplValGesInit< bool > mEstimateSpeed;
        cTplValGesInit< double > mForceVertical;
};
cElXMLTree * ToXMLTree(const cBascOnCentre &);

void  BinaryDumpInFile(ELISE_fp &,const cBascOnCentre &);

void  BinaryUnDumpFromFile(cBascOnCentre &,ELISE_fp &);

std::string  Mangling( cBascOnCentre *);

class cBascOnAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBascOnAppuis & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;
    private:
        std::string mNameRef;
};
cElXMLTree * ToXMLTree(const cBascOnAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cBascOnAppuis &);

void  BinaryUnDumpFromFile(cBascOnAppuis &,ELISE_fp &);

std::string  Mangling( cBascOnAppuis *);

class cAerialDeformNonLin
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAerialDeformNonLin & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & FlagX();
        const cTplValGesInit< int > & FlagX()const ;

        cTplValGesInit< int > & FlagY();
        const cTplValGesInit< int > & FlagY()const ;

        cTplValGesInit< int > & FlagZ();
        const cTplValGesInit< int > & FlagZ()const ;

        cTplValGesInit< bool > & ForceTrueRot();
        const cTplValGesInit< bool > & ForceTrueRot()const ;

        cTplValGesInit< std::string > & PattEstim();
        const cTplValGesInit< std::string > & PattEstim()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;
    private:
        cTplValGesInit< int > mFlagX;
        cTplValGesInit< int > mFlagY;
        cTplValGesInit< int > mFlagZ;
        cTplValGesInit< bool > mForceTrueRot;
        cTplValGesInit< std::string > mPattEstim;
        cTplValGesInit< bool > mShow;
};
cElXMLTree * ToXMLTree(const cAerialDeformNonLin &);

void  BinaryDumpInFile(ELISE_fp &,const cAerialDeformNonLin &);

void  BinaryUnDumpFromFile(cAerialDeformNonLin &,ELISE_fp &);

std::string  Mangling( cAerialDeformNonLin *);

class cBasculeOnPoints
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasculeOnPoints & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & ForceSol();
        const cTplValGesInit< std::string > & ForceSol()const ;

        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< bool > & EstimateSpeed();
        const cTplValGesInit< bool > & EstimateSpeed()const ;

        cTplValGesInit< double > & ForceVertical();
        const cTplValGesInit< double > & ForceVertical()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

        cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin();
        const cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin()const ;

        cTplValGesInit< std::string > & NameExport();
        const cTplValGesInit< std::string > & NameExport()const ;
    private:
        cTplValGesInit< std::string > mForceSol;
        cTplValGesInit< cBascOnCentre > mBascOnCentre;
        cTplValGesInit< cBascOnAppuis > mBascOnAppuis;
        cTplValGesInit< bool > mModeL2;
        cTplValGesInit< cAerialDeformNonLin > mAerialDeformNonLin;
        cTplValGesInit< std::string > mNameExport;
};
cElXMLTree * ToXMLTree(const cBasculeOnPoints &);

void  BinaryDumpInFile(ELISE_fp &,const cBasculeOnPoints &);

void  BinaryUnDumpFromFile(cBasculeOnPoints &,ELISE_fp &);

std::string  Mangling( cBasculeOnPoints *);

class cOrientInPlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOrientInPlane & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & DistFixEch();
        const cTplValGesInit< double > & DistFixEch()const ;

        std::string & FileMesures();
        const std::string & FileMesures()const ;

        cTplValGesInit< std::string > & AlignOn();
        const cTplValGesInit< std::string > & AlignOn()const ;
    private:
        cTplValGesInit< double > mDistFixEch;
        std::string mFileMesures;
        cTplValGesInit< std::string > mAlignOn;
};
cElXMLTree * ToXMLTree(const cOrientInPlane &);

void  BinaryDumpInFile(ELISE_fp &,const cOrientInPlane &);

void  BinaryUnDumpFromFile(cOrientInPlane &,ELISE_fp &);

std::string  Mangling( cOrientInPlane *);

class cBasculeLiaisonOnPlan
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasculeLiaisonOnPlan & anObj,cElXMLTree * aTree);


        cParamEstimPlan & EstimPl();
        const cParamEstimPlan & EstimPl()const ;

        cTplValGesInit< double > & DistFixEch();
        const cTplValGesInit< double > & DistFixEch()const ;

        std::string & FileMesures();
        const std::string & FileMesures()const ;

        cTplValGesInit< std::string > & AlignOn();
        const cTplValGesInit< std::string > & AlignOn()const ;

        cTplValGesInit< cOrientInPlane > & OrientInPlane();
        const cTplValGesInit< cOrientInPlane > & OrientInPlane()const ;
    private:
        cParamEstimPlan mEstimPl;
        cTplValGesInit< cOrientInPlane > mOrientInPlane;
};
cElXMLTree * ToXMLTree(const cBasculeLiaisonOnPlan &);

void  BinaryDumpInFile(ELISE_fp &,const cBasculeLiaisonOnPlan &);

void  BinaryUnDumpFromFile(cBasculeLiaisonOnPlan &,ELISE_fp &);

std::string  Mangling( cBasculeLiaisonOnPlan *);

class cModeBascule
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModeBascule & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & ForceSol();
        const cTplValGesInit< std::string > & ForceSol()const ;

        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< bool > & EstimateSpeed();
        const cTplValGesInit< bool > & EstimateSpeed()const ;

        cTplValGesInit< double > & ForceVertical();
        const cTplValGesInit< double > & ForceVertical()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

        cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin();
        const cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin()const ;

        cTplValGesInit< std::string > & NameExport();
        const cTplValGesInit< std::string > & NameExport()const ;

        cTplValGesInit< cBasculeOnPoints > & BasculeOnPoints();
        const cTplValGesInit< cBasculeOnPoints > & BasculeOnPoints()const ;

        cParamEstimPlan & EstimPl();
        const cParamEstimPlan & EstimPl()const ;

        cTplValGesInit< double > & DistFixEch();
        const cTplValGesInit< double > & DistFixEch()const ;

        std::string & FileMesures();
        const std::string & FileMesures()const ;

        cTplValGesInit< std::string > & AlignOn();
        const cTplValGesInit< std::string > & AlignOn()const ;

        cTplValGesInit< cOrientInPlane > & OrientInPlane();
        const cTplValGesInit< cOrientInPlane > & OrientInPlane()const ;

        cTplValGesInit< cBasculeLiaisonOnPlan > & BasculeLiaisonOnPlan();
        const cTplValGesInit< cBasculeLiaisonOnPlan > & BasculeLiaisonOnPlan()const ;
    private:
        cTplValGesInit< cBasculeOnPoints > mBasculeOnPoints;
        cTplValGesInit< cBasculeLiaisonOnPlan > mBasculeLiaisonOnPlan;
};
cElXMLTree * ToXMLTree(const cModeBascule &);

void  BinaryDumpInFile(ELISE_fp &,const cModeBascule &);

void  BinaryUnDumpFromFile(cModeBascule &,ELISE_fp &);

std::string  Mangling( cModeBascule *);

class cBasculeOrientation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBasculeOrientation & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & AfterCompens();
        const cTplValGesInit< bool > & AfterCompens()const ;

        cTplValGesInit< std::string > & PatternNameApply();
        const cTplValGesInit< std::string > & PatternNameApply()const ;

        cTplValGesInit< std::string > & PatternNameEstim();
        const cTplValGesInit< std::string > & PatternNameEstim()const ;

        cTplValGesInit< std::string > & FileExportDir();
        const cTplValGesInit< std::string > & FileExportDir()const ;

        cTplValGesInit< std::string > & FileExportInv();
        const cTplValGesInit< std::string > & FileExportInv()const ;

        cTplValGesInit< std::string > & ForceSol();
        const cTplValGesInit< std::string > & ForceSol()const ;

        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< bool > & EstimateSpeed();
        const cTplValGesInit< bool > & EstimateSpeed()const ;

        cTplValGesInit< double > & ForceVertical();
        const cTplValGesInit< double > & ForceVertical()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

        cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin();
        const cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin()const ;

        cTplValGesInit< std::string > & NameExport();
        const cTplValGesInit< std::string > & NameExport()const ;

        cTplValGesInit< cBasculeOnPoints > & BasculeOnPoints();
        const cTplValGesInit< cBasculeOnPoints > & BasculeOnPoints()const ;

        cParamEstimPlan & EstimPl();
        const cParamEstimPlan & EstimPl()const ;

        cTplValGesInit< double > & DistFixEch();
        const cTplValGesInit< double > & DistFixEch()const ;

        std::string & FileMesures();
        const std::string & FileMesures()const ;

        cTplValGesInit< std::string > & AlignOn();
        const cTplValGesInit< std::string > & AlignOn()const ;

        cTplValGesInit< cOrientInPlane > & OrientInPlane();
        const cTplValGesInit< cOrientInPlane > & OrientInPlane()const ;

        cTplValGesInit< cBasculeLiaisonOnPlan > & BasculeLiaisonOnPlan();
        const cTplValGesInit< cBasculeLiaisonOnPlan > & BasculeLiaisonOnPlan()const ;

        cModeBascule & ModeBascule();
        const cModeBascule & ModeBascule()const ;
    private:
        cTplValGesInit< bool > mAfterCompens;
        cTplValGesInit< std::string > mPatternNameApply;
        cTplValGesInit< std::string > mPatternNameEstim;
        cTplValGesInit< std::string > mFileExportDir;
        cTplValGesInit< std::string > mFileExportInv;
        cModeBascule mModeBascule;
};
cElXMLTree * ToXMLTree(const cBasculeOrientation &);

void  BinaryDumpInFile(ELISE_fp &,const cBasculeOrientation &);

void  BinaryUnDumpFromFile(cBasculeOrientation &,ELISE_fp &);

std::string  Mangling( cBasculeOrientation *);

class cStereoFE
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cStereoFE & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeStereo > & HomFE();
        const std::vector< cAperoPointeStereo > & HomFE()const ;
    private:
        std::vector< cAperoPointeStereo > mHomFE;
};
cElXMLTree * ToXMLTree(const cStereoFE &);

void  BinaryDumpInFile(ELISE_fp &,const cStereoFE &);

void  BinaryUnDumpFromFile(cStereoFE &,ELISE_fp &);

std::string  Mangling( cStereoFE *);

class cModeFE
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModeFE & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeStereo > & HomFE();
        const std::vector< cAperoPointeStereo > & HomFE()const ;

        cTplValGesInit< cStereoFE > & StereoFE();
        const cTplValGesInit< cStereoFE > & StereoFE()const ;

        cTplValGesInit< cApero2PointeFromFile > & FEFromFile();
        const cTplValGesInit< cApero2PointeFromFile > & FEFromFile()const ;
    private:
        cTplValGesInit< cStereoFE > mStereoFE;
        cTplValGesInit< cApero2PointeFromFile > mFEFromFile;
};
cElXMLTree * ToXMLTree(const cModeFE &);

void  BinaryDumpInFile(ELISE_fp &,const cModeFE &);

void  BinaryUnDumpFromFile(cModeFE &,ELISE_fp &);

std::string  Mangling( cModeFE *);

class cFixeEchelle
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFixeEchelle & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeStereo > & HomFE();
        const std::vector< cAperoPointeStereo > & HomFE()const ;

        cTplValGesInit< cStereoFE > & StereoFE();
        const cTplValGesInit< cStereoFE > & StereoFE()const ;

        cTplValGesInit< cApero2PointeFromFile > & FEFromFile();
        const cTplValGesInit< cApero2PointeFromFile > & FEFromFile()const ;

        cModeFE & ModeFE();
        const cModeFE & ModeFE()const ;

        double & DistVraie();
        const double & DistVraie()const ;
    private:
        cModeFE mModeFE;
        double mDistVraie;
};
cElXMLTree * ToXMLTree(const cFixeEchelle &);

void  BinaryDumpInFile(ELISE_fp &,const cFixeEchelle &);

void  BinaryUnDumpFromFile(cFixeEchelle &,ELISE_fp &);

std::string  Mangling( cFixeEchelle *);

class cHorFOP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cHorFOP & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeMono > & VecFOH();
        const std::vector< cAperoPointeMono > & VecFOH()const ;

        cTplValGesInit< double > & Z();
        const cTplValGesInit< double > & Z()const ;
    private:
        std::vector< cAperoPointeMono > mVecFOH;
        cTplValGesInit< double > mZ;
};
cElXMLTree * ToXMLTree(const cHorFOP &);

void  BinaryDumpInFile(ELISE_fp &,const cHorFOP &);

void  BinaryUnDumpFromFile(cHorFOP &,ELISE_fp &);

std::string  Mangling( cHorFOP *);

class cModeFOP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModeFOP & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeMono > & VecFOH();
        const std::vector< cAperoPointeMono > & VecFOH()const ;

        cTplValGesInit< double > & Z();
        const cTplValGesInit< double > & Z()const ;

        cTplValGesInit< cHorFOP > & HorFOP();
        const cTplValGesInit< cHorFOP > & HorFOP()const ;

        cTplValGesInit< cApero2PointeFromFile > & HorFromFile();
        const cTplValGesInit< cApero2PointeFromFile > & HorFromFile()const ;
    private:
        cTplValGesInit< cHorFOP > mHorFOP;
        cTplValGesInit< cApero2PointeFromFile > mHorFromFile;
};
cElXMLTree * ToXMLTree(const cModeFOP &);

void  BinaryDumpInFile(ELISE_fp &,const cModeFOP &);

void  BinaryUnDumpFromFile(cModeFOP &,ELISE_fp &);

std::string  Mangling( cModeFOP *);

class cFixeOrientPlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFixeOrientPlane & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeMono > & VecFOH();
        const std::vector< cAperoPointeMono > & VecFOH()const ;

        cTplValGesInit< double > & Z();
        const cTplValGesInit< double > & Z()const ;

        cTplValGesInit< cHorFOP > & HorFOP();
        const cTplValGesInit< cHorFOP > & HorFOP()const ;

        cTplValGesInit< cApero2PointeFromFile > & HorFromFile();
        const cTplValGesInit< cApero2PointeFromFile > & HorFromFile()const ;

        cModeFOP & ModeFOP();
        const cModeFOP & ModeFOP()const ;

        Pt2dr & Vecteur();
        const Pt2dr & Vecteur()const ;
    private:
        cModeFOP mModeFOP;
        Pt2dr mVecteur;
};
cElXMLTree * ToXMLTree(const cFixeOrientPlane &);

void  BinaryDumpInFile(ELISE_fp &,const cFixeOrientPlane &);

void  BinaryUnDumpFromFile(cFixeOrientPlane &,ELISE_fp &);

std::string  Mangling( cFixeOrientPlane *);

class cBlocBascule
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBlocBascule & anObj,cElXMLTree * aTree);


        std::string & Pattern1();
        const std::string & Pattern1()const ;

        std::string & Pattern2();
        const std::string & Pattern2()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;
    private:
        std::string mPattern1;
        std::string mPattern2;
        std::string mIdBdl;
};
cElXMLTree * ToXMLTree(const cBlocBascule &);

void  BinaryDumpInFile(ELISE_fp &,const cBlocBascule &);

void  BinaryUnDumpFromFile(cBlocBascule &,ELISE_fp &);

std::string  Mangling( cBlocBascule *);

class cMesureErreurTournante
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMesureErreurTournante & anObj,cElXMLTree * aTree);


        int & Periode();
        const int & Periode()const ;

        cTplValGesInit< int > & NbTest();
        const cTplValGesInit< int > & NbTest()const ;

        cTplValGesInit< int > & NbIter();
        const cTplValGesInit< int > & NbIter()const ;

        cTplValGesInit< bool > & ApplyAppuis();
        const cTplValGesInit< bool > & ApplyAppuis()const ;

        cTplValGesInit< bool > & ApplyLiaisons();
        const cTplValGesInit< bool > & ApplyLiaisons()const ;
    private:
        int mPeriode;
        cTplValGesInit< int > mNbTest;
        cTplValGesInit< int > mNbIter;
        cTplValGesInit< bool > mApplyAppuis;
        cTplValGesInit< bool > mApplyLiaisons;
};
cElXMLTree * ToXMLTree(const cMesureErreurTournante &);

void  BinaryDumpInFile(ELISE_fp &,const cMesureErreurTournante &);

void  BinaryUnDumpFromFile(cMesureErreurTournante &,ELISE_fp &);

std::string  Mangling( cMesureErreurTournante *);

class cContraintesCamerasInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContraintesCamerasInc & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & TolContrainte();
        const cTplValGesInit< double > & TolContrainte()const ;

        cTplValGesInit< std::string > & PatternNameApply();
        const cTplValGesInit< std::string > & PatternNameApply()const ;

        std::list< eTypeContrainteCalibCamera > & Val();
        const std::list< eTypeContrainteCalibCamera > & Val()const ;

        cTplValGesInit< cElRegex_Ptr > & PatternRefuteur();
        const cTplValGesInit< cElRegex_Ptr > & PatternRefuteur()const ;
    private:
        cTplValGesInit< double > mTolContrainte;
        cTplValGesInit< std::string > mPatternNameApply;
        std::list< eTypeContrainteCalibCamera > mVal;
        cTplValGesInit< cElRegex_Ptr > mPatternRefuteur;
};
cElXMLTree * ToXMLTree(const cContraintesCamerasInc &);

void  BinaryDumpInFile(ELISE_fp &,const cContraintesCamerasInc &);

void  BinaryUnDumpFromFile(cContraintesCamerasInc &,ELISE_fp &);

std::string  Mangling( cContraintesCamerasInc *);

class cContraintesPoses
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContraintesPoses & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & ByPattern();
        const cTplValGesInit< bool > & ByPattern()const ;

        cTplValGesInit< std::string > & PatternRefuteur();
        const cTplValGesInit< std::string > & PatternRefuteur()const ;

        cTplValGesInit< double > & TolAng();
        const cTplValGesInit< double > & TolAng()const ;

        cTplValGesInit< double > & TolCoord();
        const cTplValGesInit< double > & TolCoord()const ;

        std::string & NamePose();
        const std::string & NamePose()const ;

        eTypeContraintePoseCamera & Val();
        const eTypeContraintePoseCamera & Val()const ;

        cTplValGesInit< std::string > & PoseRattachement();
        const cTplValGesInit< std::string > & PoseRattachement()const ;
    private:
        cTplValGesInit< bool > mByPattern;
        cTplValGesInit< std::string > mPatternRefuteur;
        cTplValGesInit< double > mTolAng;
        cTplValGesInit< double > mTolCoord;
        std::string mNamePose;
        eTypeContraintePoseCamera mVal;
        cTplValGesInit< std::string > mPoseRattachement;
};
cElXMLTree * ToXMLTree(const cContraintesPoses &);

void  BinaryDumpInFile(ELISE_fp &,const cContraintesPoses &);

void  BinaryUnDumpFromFile(cContraintesPoses &,ELISE_fp &);

std::string  Mangling( cContraintesPoses *);

class cSectionContraintes
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionContraintes & anObj,cElXMLTree * aTree);


        std::list< cContraintesCamerasInc > & ContraintesCamerasInc();
        const std::list< cContraintesCamerasInc > & ContraintesCamerasInc()const ;

        std::list< cContraintesPoses > & ContraintesPoses();
        const std::list< cContraintesPoses > & ContraintesPoses()const ;
    private:
        std::list< cContraintesCamerasInc > mContraintesCamerasInc;
        std::list< cContraintesPoses > mContraintesPoses;
};
cElXMLTree * ToXMLTree(const cSectionContraintes &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionContraintes &);

void  BinaryUnDumpFromFile(cSectionContraintes &,ELISE_fp &);

std::string  Mangling( cSectionContraintes *);

class cVisuPtsMult
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVisuPtsMult & anObj,cElXMLTree * aTree);


        std::string & Cam1();
        const std::string & Cam1()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< int > & SzWPrinc();
        const cTplValGesInit< int > & SzWPrinc()const ;

        cTplValGesInit< int > & SzWAux();
        const cTplValGesInit< int > & SzWAux()const ;

        cTplValGesInit< int > & ZoomWAux();
        const cTplValGesInit< int > & ZoomWAux()const ;

        Pt2di & NbWAux();
        const Pt2di & NbWAux()const ;

        bool & AuxEnDessous();
        const bool & AuxEnDessous()const ;

        cTplValGesInit< double > & MaxDistReproj();
        const cTplValGesInit< double > & MaxDistReproj()const ;

        cTplValGesInit< double > & MaxDistSift();
        const cTplValGesInit< double > & MaxDistSift()const ;

        cTplValGesInit< double > & MaxDistProjCorr();
        const cTplValGesInit< double > & MaxDistProjCorr()const ;

        cTplValGesInit< double > & SeuilCorrel();
        const cTplValGesInit< double > & SeuilCorrel()const ;
    private:
        std::string mCam1;
        std::string mId;
        cTplValGesInit< int > mSzWPrinc;
        cTplValGesInit< int > mSzWAux;
        cTplValGesInit< int > mZoomWAux;
        Pt2di mNbWAux;
        bool mAuxEnDessous;
        cTplValGesInit< double > mMaxDistReproj;
        cTplValGesInit< double > mMaxDistSift;
        cTplValGesInit< double > mMaxDistProjCorr;
        cTplValGesInit< double > mSeuilCorrel;
};
cElXMLTree * ToXMLTree(const cVisuPtsMult &);

void  BinaryDumpInFile(ELISE_fp &,const cVisuPtsMult &);

void  BinaryUnDumpFromFile(cVisuPtsMult &,ELISE_fp &);

std::string  Mangling( cVisuPtsMult *);

class cVerifAero
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVerifAero & anObj,cElXMLTree * aTree);


        std::string & PatternApply();
        const std::string & PatternApply()const ;

        std::string & IdBdLiaison();
        const std::string & IdBdLiaison()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        std::string & Prefixe();
        const std::string & Prefixe()const ;

        eTypeVerif & TypeVerif();
        const eTypeVerif & TypeVerif()const ;

        double & SeuilTxt();
        const double & SeuilTxt()const ;

        double & Resol();
        const double & Resol()const ;

        double & PasR();
        const double & PasR()const ;

        double & PasB();
        const double & PasB()const ;
    private:
        std::string mPatternApply;
        std::string mIdBdLiaison;
        cPonderationPackMesure mPond;
        std::string mPrefixe;
        eTypeVerif mTypeVerif;
        double mSeuilTxt;
        double mResol;
        double mPasR;
        double mPasB;
};
cElXMLTree * ToXMLTree(const cVerifAero &);

void  BinaryDumpInFile(ELISE_fp &,const cVerifAero &);

void  BinaryUnDumpFromFile(cVerifAero &,ELISE_fp &);

std::string  Mangling( cVerifAero *);

class cGPtsTer_By_ImProf
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGPtsTer_By_ImProf & anObj,cElXMLTree * aTree);


        Pt3dr & Origine();
        const Pt3dr & Origine()const ;

        Pt3dr & Step();
        const Pt3dr & Step()const ;

        int & NbPts();
        const int & NbPts()const ;

        bool & OnGrid();
        const bool & OnGrid()const ;

        std::string & File();
        const std::string & File()const ;

        cTplValGesInit< double > & RandomizeInGrid();
        const cTplValGesInit< double > & RandomizeInGrid()const ;

        cTplValGesInit< std::string > & ImMaitresse();
        const cTplValGesInit< std::string > & ImMaitresse()const ;

        cTplValGesInit< bool > & DTMIsZ();
        const cTplValGesInit< bool > & DTMIsZ()const ;
    private:
        Pt3dr mOrigine;
        Pt3dr mStep;
        int mNbPts;
        bool mOnGrid;
        std::string mFile;
        cTplValGesInit< double > mRandomizeInGrid;
        cTplValGesInit< std::string > mImMaitresse;
        cTplValGesInit< bool > mDTMIsZ;
};
cElXMLTree * ToXMLTree(const cGPtsTer_By_ImProf &);

void  BinaryDumpInFile(ELISE_fp &,const cGPtsTer_By_ImProf &);

void  BinaryUnDumpFromFile(cGPtsTer_By_ImProf &,ELISE_fp &);

std::string  Mangling( cGPtsTer_By_ImProf *);

class cGeneratePointsTerrains
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGeneratePointsTerrains & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & GPtsTer_By_File();
        const cTplValGesInit< std::string > & GPtsTer_By_File()const ;

        Pt3dr & Origine();
        const Pt3dr & Origine()const ;

        Pt3dr & Step();
        const Pt3dr & Step()const ;

        int & NbPts();
        const int & NbPts()const ;

        bool & OnGrid();
        const bool & OnGrid()const ;

        std::string & File();
        const std::string & File()const ;

        cTplValGesInit< double > & RandomizeInGrid();
        const cTplValGesInit< double > & RandomizeInGrid()const ;

        cTplValGesInit< std::string > & ImMaitresse();
        const cTplValGesInit< std::string > & ImMaitresse()const ;

        cTplValGesInit< bool > & DTMIsZ();
        const cTplValGesInit< bool > & DTMIsZ()const ;

        cTplValGesInit< cGPtsTer_By_ImProf > & GPtsTer_By_ImProf();
        const cTplValGesInit< cGPtsTer_By_ImProf > & GPtsTer_By_ImProf()const ;
    private:
        cTplValGesInit< std::string > mGPtsTer_By_File;
        cTplValGesInit< cGPtsTer_By_ImProf > mGPtsTer_By_ImProf;
};
cElXMLTree * ToXMLTree(const cGeneratePointsTerrains &);

void  BinaryDumpInFile(ELISE_fp &,const cGeneratePointsTerrains &);

void  BinaryUnDumpFromFile(cGeneratePointsTerrains &,ELISE_fp &);

std::string  Mangling( cGeneratePointsTerrains *);

class cGenerateLiaisons
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenerateLiaisons & anObj,cElXMLTree * aTree);


        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< std::string > & FilterIm1();
        const cTplValGesInit< std::string > & FilterIm1()const ;

        cTplValGesInit< std::string > & FilterIm2();
        const cTplValGesInit< std::string > & FilterIm2()const ;

        double & BruitIm1();
        const double & BruitIm1()const ;

        double & BruitIm2();
        const double & BruitIm2()const ;
    private:
        std::string mKeyAssoc;
        cTplValGesInit< std::string > mFilterIm1;
        cTplValGesInit< std::string > mFilterIm2;
        double mBruitIm1;
        double mBruitIm2;
};
cElXMLTree * ToXMLTree(const cGenerateLiaisons &);

void  BinaryDumpInFile(ELISE_fp &,const cGenerateLiaisons &);

void  BinaryUnDumpFromFile(cGenerateLiaisons &,ELISE_fp &);

std::string  Mangling( cGenerateLiaisons *);

class cExportSimulation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportSimulation & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & GPtsTer_By_File();
        const cTplValGesInit< std::string > & GPtsTer_By_File()const ;

        Pt3dr & Origine();
        const Pt3dr & Origine()const ;

        Pt3dr & Step();
        const Pt3dr & Step()const ;

        int & NbPts();
        const int & NbPts()const ;

        bool & OnGrid();
        const bool & OnGrid()const ;

        std::string & File();
        const std::string & File()const ;

        cTplValGesInit< double > & RandomizeInGrid();
        const cTplValGesInit< double > & RandomizeInGrid()const ;

        cTplValGesInit< std::string > & ImMaitresse();
        const cTplValGesInit< std::string > & ImMaitresse()const ;

        cTplValGesInit< bool > & DTMIsZ();
        const cTplValGesInit< bool > & DTMIsZ()const ;

        cTplValGesInit< cGPtsTer_By_ImProf > & GPtsTer_By_ImProf();
        const cTplValGesInit< cGPtsTer_By_ImProf > & GPtsTer_By_ImProf()const ;

        cGeneratePointsTerrains & GeneratePointsTerrains();
        const cGeneratePointsTerrains & GeneratePointsTerrains()const ;

        std::list< cGenerateLiaisons > & GenerateLiaisons();
        const std::list< cGenerateLiaisons > & GenerateLiaisons()const ;
    private:
        cGeneratePointsTerrains mGeneratePointsTerrains;
        std::list< cGenerateLiaisons > mGenerateLiaisons;
};
cElXMLTree * ToXMLTree(const cExportSimulation &);

void  BinaryDumpInFile(ELISE_fp &,const cExportSimulation &);

void  BinaryUnDumpFromFile(cExportSimulation &,ELISE_fp &);

std::string  Mangling( cExportSimulation *);

class cTestInteractif
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTestInteractif & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & AvantCompens();
        const cTplValGesInit< bool > & AvantCompens()const ;

        cTplValGesInit< bool > & ApresCompens();
        const cTplValGesInit< bool > & ApresCompens()const ;

        cTplValGesInit< bool > & TestF2C2();
        const cTplValGesInit< bool > & TestF2C2()const ;

        cTplValGesInit< bool > & SetStepByStep();
        const cTplValGesInit< bool > & SetStepByStep()const ;
    private:
        cTplValGesInit< bool > mAvantCompens;
        cTplValGesInit< bool > mApresCompens;
        cTplValGesInit< bool > mTestF2C2;
        cTplValGesInit< bool > mSetStepByStep;
};
cElXMLTree * ToXMLTree(const cTestInteractif &);

void  BinaryDumpInFile(ELISE_fp &,const cTestInteractif &);

void  BinaryUnDumpFromFile(cTestInteractif &,ELISE_fp &);

std::string  Mangling( cTestInteractif *);

class cIterationsCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cIterationsCompensation & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NbMin();
        const cTplValGesInit< int > & NbMin()const ;

        int & NbMax();
        const int & NbMax()const ;

        double & SeuilEvolMoy();
        const double & SeuilEvolMoy()const ;

        cTplValGesInit< double > & SeuilEvolMax();
        const cTplValGesInit< double > & SeuilEvolMax()const ;

        double & Mult();
        const double & Mult()const ;

        cTplValGesInit< bool > & ModeMin();
        const cTplValGesInit< bool > & ModeMin()const ;

        cTplValGesInit< cAutoAdaptLVM > & AutoAdaptLVM();
        const cTplValGesInit< cAutoAdaptLVM > & AutoAdaptLVM()const ;

        cTplValGesInit< cCtrlTimeCompens > & CtrlTimeCompens();
        const cTplValGesInit< cCtrlTimeCompens > & CtrlTimeCompens()const ;

        cTplValGesInit< bool > & DoIt();
        const cTplValGesInit< bool > & DoIt()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMIter();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMIter()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMEtape();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMEtape()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob()const ;

        cTplValGesInit< double > & MultSLMIter();
        const cTplValGesInit< double > & MultSLMIter()const ;

        cTplValGesInit< double > & MultSLMEtape();
        const cTplValGesInit< double > & MultSLMEtape()const ;

        cTplValGesInit< double > & MultSLMGlob();
        const cTplValGesInit< double > & MultSLMGlob()const ;

        std::vector<int> & ProfMin();
        const std::vector<int> & ProfMin()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< int > & StepComplemAuto();
        const cTplValGesInit< int > & StepComplemAuto()const ;

        cTplValGesInit< cPose2Init > & Pose2Init();
        const cTplValGesInit< cPose2Init > & Pose2Init()const ;

        std::list< cSetRayMaxUtileCalib > & SetRayMaxUtileCalib();
        const std::list< cSetRayMaxUtileCalib > & SetRayMaxUtileCalib()const ;

        cTplValGesInit< bool > & AfterCompens();
        const cTplValGesInit< bool > & AfterCompens()const ;

        cTplValGesInit< std::string > & PatternNameApply();
        const cTplValGesInit< std::string > & PatternNameApply()const ;

        cTplValGesInit< std::string > & PatternNameEstim();
        const cTplValGesInit< std::string > & PatternNameEstim()const ;

        cTplValGesInit< std::string > & FileExportDir();
        const cTplValGesInit< std::string > & FileExportDir()const ;

        cTplValGesInit< std::string > & FileExportInv();
        const cTplValGesInit< std::string > & FileExportInv()const ;

        cTplValGesInit< std::string > & ForceSol();
        const cTplValGesInit< std::string > & ForceSol()const ;

        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< bool > & EstimateSpeed();
        const cTplValGesInit< bool > & EstimateSpeed()const ;

        cTplValGesInit< double > & ForceVertical();
        const cTplValGesInit< double > & ForceVertical()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

        cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin();
        const cTplValGesInit< cAerialDeformNonLin > & AerialDeformNonLin()const ;

        cTplValGesInit< std::string > & NameExport();
        const cTplValGesInit< std::string > & NameExport()const ;

        cTplValGesInit< cBasculeOnPoints > & BasculeOnPoints();
        const cTplValGesInit< cBasculeOnPoints > & BasculeOnPoints()const ;

        cParamEstimPlan & EstimPl();
        const cParamEstimPlan & EstimPl()const ;

        cTplValGesInit< double > & DistFixEch();
        const cTplValGesInit< double > & DistFixEch()const ;

        std::string & FileMesures();
        const std::string & FileMesures()const ;

        cTplValGesInit< std::string > & AlignOn();
        const cTplValGesInit< std::string > & AlignOn()const ;

        cTplValGesInit< cOrientInPlane > & OrientInPlane();
        const cTplValGesInit< cOrientInPlane > & OrientInPlane()const ;

        cTplValGesInit< cBasculeLiaisonOnPlan > & BasculeLiaisonOnPlan();
        const cTplValGesInit< cBasculeLiaisonOnPlan > & BasculeLiaisonOnPlan()const ;

        cModeBascule & ModeBascule();
        const cModeBascule & ModeBascule()const ;

        cTplValGesInit< cBasculeOrientation > & BasculeOrientation();
        const cTplValGesInit< cBasculeOrientation > & BasculeOrientation()const ;

        std::vector< cAperoPointeStereo > & HomFE();
        const std::vector< cAperoPointeStereo > & HomFE()const ;

        cTplValGesInit< cStereoFE > & StereoFE();
        const cTplValGesInit< cStereoFE > & StereoFE()const ;

        cTplValGesInit< cApero2PointeFromFile > & FEFromFile();
        const cTplValGesInit< cApero2PointeFromFile > & FEFromFile()const ;

        cModeFE & ModeFE();
        const cModeFE & ModeFE()const ;

        double & DistVraie();
        const double & DistVraie()const ;

        cTplValGesInit< cFixeEchelle > & FixeEchelle();
        const cTplValGesInit< cFixeEchelle > & FixeEchelle()const ;

        std::vector< cAperoPointeMono > & VecFOH();
        const std::vector< cAperoPointeMono > & VecFOH()const ;

        cTplValGesInit< double > & Z();
        const cTplValGesInit< double > & Z()const ;

        cTplValGesInit< cHorFOP > & HorFOP();
        const cTplValGesInit< cHorFOP > & HorFOP()const ;

        cTplValGesInit< cApero2PointeFromFile > & HorFromFile();
        const cTplValGesInit< cApero2PointeFromFile > & HorFromFile()const ;

        cModeFOP & ModeFOP();
        const cModeFOP & ModeFOP()const ;

        Pt2dr & Vecteur();
        const Pt2dr & Vecteur()const ;

        cTplValGesInit< cFixeOrientPlane > & FixeOrientPlane();
        const cTplValGesInit< cFixeOrientPlane > & FixeOrientPlane()const ;

        cTplValGesInit< std::string > & BasicOrPl();
        const cTplValGesInit< std::string > & BasicOrPl()const ;

        std::string & Pattern1();
        const std::string & Pattern1()const ;

        std::string & Pattern2();
        const std::string & Pattern2()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< cBlocBascule > & BlocBascule();
        const cTplValGesInit< cBlocBascule > & BlocBascule()const ;

        std::list< cXml_EstimateOrientationInitBlockCamera > & EstimateOrientationInitBlockCamera();
        const std::list< cXml_EstimateOrientationInitBlockCamera > & EstimateOrientationInitBlockCamera()const ;

        int & Periode();
        const int & Periode()const ;

        cTplValGesInit< int > & NbTest();
        const cTplValGesInit< int > & NbTest()const ;

        cTplValGesInit< int > & NbIter();
        const cTplValGesInit< int > & NbIter()const ;

        cTplValGesInit< bool > & ApplyAppuis();
        const cTplValGesInit< bool > & ApplyAppuis()const ;

        cTplValGesInit< bool > & ApplyLiaisons();
        const cTplValGesInit< bool > & ApplyLiaisons()const ;

        cTplValGesInit< cMesureErreurTournante > & MesureErreurTournante();
        const cTplValGesInit< cMesureErreurTournante > & MesureErreurTournante()const ;

        std::list< cContraintesCamerasInc > & ContraintesCamerasInc();
        const std::list< cContraintesCamerasInc > & ContraintesCamerasInc()const ;

        std::list< cContraintesPoses > & ContraintesPoses();
        const std::list< cContraintesPoses > & ContraintesPoses()const ;

        cTplValGesInit< cSectionContraintes > & SectionContraintes();
        const cTplValGesInit< cSectionContraintes > & SectionContraintes()const ;

        std::list< std::string > & Messages();
        const std::list< std::string > & Messages()const ;

        std::list< cVisuPtsMult > & VisuPtsMult();
        const std::list< cVisuPtsMult > & VisuPtsMult()const ;

        std::list< cVerifAero > & VerifAero();
        const std::list< cVerifAero > & VerifAero()const ;

        std::list< cExportSimulation > & ExportSimulation();
        const std::list< cExportSimulation > & ExportSimulation()const ;

        cTplValGesInit< bool > & AvantCompens();
        const cTplValGesInit< bool > & AvantCompens()const ;

        cTplValGesInit< bool > & ApresCompens();
        const cTplValGesInit< bool > & ApresCompens()const ;

        cTplValGesInit< bool > & TestF2C2();
        const cTplValGesInit< bool > & TestF2C2()const ;

        cTplValGesInit< bool > & SetStepByStep();
        const cTplValGesInit< bool > & SetStepByStep()const ;

        cTplValGesInit< cTestInteractif > & TestInteractif();
        const cTplValGesInit< cTestInteractif > & TestInteractif()const ;
    private:
        cTplValGesInit< cCtrlTimeCompens > mCtrlTimeCompens;
        cTplValGesInit< bool > mDoIt;
        cTplValGesInit< cSectionLevenbergMarkard > mSLMIter;
        cTplValGesInit< cSectionLevenbergMarkard > mSLMEtape;
        cTplValGesInit< cSectionLevenbergMarkard > mSLMGlob;
        cTplValGesInit< double > mMultSLMIter;
        cTplValGesInit< double > mMultSLMEtape;
        cTplValGesInit< double > mMultSLMGlob;
        cTplValGesInit< cPose2Init > mPose2Init;
        std::list< cSetRayMaxUtileCalib > mSetRayMaxUtileCalib;
        cTplValGesInit< cBasculeOrientation > mBasculeOrientation;
        cTplValGesInit< cFixeEchelle > mFixeEchelle;
        cTplValGesInit< cFixeOrientPlane > mFixeOrientPlane;
        cTplValGesInit< std::string > mBasicOrPl;
        cTplValGesInit< cBlocBascule > mBlocBascule;
        std::list< cXml_EstimateOrientationInitBlockCamera > mEstimateOrientationInitBlockCamera;
        cTplValGesInit< cMesureErreurTournante > mMesureErreurTournante;
        cTplValGesInit< cSectionContraintes > mSectionContraintes;
        std::list< std::string > mMessages;
        std::list< cVisuPtsMult > mVisuPtsMult;
        std::list< cVerifAero > mVerifAero;
        std::list< cExportSimulation > mExportSimulation;
        cTplValGesInit< cTestInteractif > mTestInteractif;
};
cElXMLTree * ToXMLTree(const cIterationsCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cIterationsCompensation &);

void  BinaryUnDumpFromFile(cIterationsCompensation &,ELISE_fp &);

std::string  Mangling( cIterationsCompensation *);

class cTraceCpleHom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTraceCpleHom & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;
    private:
        std::string mId;
};
cElXMLTree * ToXMLTree(const cTraceCpleHom &);

void  BinaryDumpInFile(ELISE_fp &,const cTraceCpleHom &);

void  BinaryUnDumpFromFile(cTraceCpleHom &,ELISE_fp &);

std::string  Mangling( cTraceCpleHom *);

class cTraceCpleCam
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTraceCpleCam & anObj,cElXMLTree * aTree);


        std::string & Cam1();
        const std::string & Cam1()const ;

        std::string & Cam2();
        const std::string & Cam2()const ;

        std::list< cTraceCpleHom > & TraceCpleHom();
        const std::list< cTraceCpleHom > & TraceCpleHom()const ;
    private:
        std::string mCam1;
        std::string mCam2;
        std::list< cTraceCpleHom > mTraceCpleHom;
};
cElXMLTree * ToXMLTree(const cTraceCpleCam &);

void  BinaryDumpInFile(ELISE_fp &,const cTraceCpleCam &);

void  BinaryUnDumpFromFile(cTraceCpleCam &,ELISE_fp &);

std::string  Mangling( cTraceCpleCam *);

class cSectionTracage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionTracage & anObj,cElXMLTree * aTree);


        std::list< cTraceCpleCam > & TraceCpleCam();
        const std::list< cTraceCpleCam > & TraceCpleCam()const ;

        cTplValGesInit< bool > & GetChar();
        const cTplValGesInit< bool > & GetChar()const ;
    private:
        std::list< cTraceCpleCam > mTraceCpleCam;
        cTplValGesInit< bool > mGetChar;
};
cElXMLTree * ToXMLTree(const cSectionTracage &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionTracage &);

void  BinaryUnDumpFromFile(cSectionTracage &,ELISE_fp &);

std::string  Mangling( cSectionTracage *);

class cContrCamConseq
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContrCamConseq & anObj,cElXMLTree * aTree);


        double & SigmaPix();
        const double & SigmaPix()const ;

        int & NbGrid();
        const int & NbGrid()const ;
    private:
        double mSigmaPix;
        int mNbGrid;
};
cElXMLTree * ToXMLTree(const cContrCamConseq &);

void  BinaryDumpInFile(ELISE_fp &,const cContrCamConseq &);

void  BinaryUnDumpFromFile(cContrCamConseq &,ELISE_fp &);

std::string  Mangling( cContrCamConseq *);

class cContrCamGenInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContrCamGenInc & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PatternApply();
        const cTplValGesInit< std::string > & PatternApply()const ;

        cTplValGesInit< double > & PdsAttachToId();
        const cTplValGesInit< double > & PdsAttachToId()const ;

        cTplValGesInit< double > & PdsAttachToLast();
        const cTplValGesInit< double > & PdsAttachToLast()const ;

        cTplValGesInit< double > & PdsAttachRGLob();
        const cTplValGesInit< double > & PdsAttachRGLob()const ;
    private:
        cTplValGesInit< std::string > mPatternApply;
        cTplValGesInit< double > mPdsAttachToId;
        cTplValGesInit< double > mPdsAttachToLast;
        cTplValGesInit< double > mPdsAttachRGLob;
};
cElXMLTree * ToXMLTree(const cContrCamGenInc &);

void  BinaryDumpInFile(ELISE_fp &,const cContrCamGenInc &);

void  BinaryUnDumpFromFile(cContrCamGenInc &,ELISE_fp &);

std::string  Mangling( cContrCamGenInc *);

class cObsBlockCamRig
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsBlockCamRig & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< cRigidBlockWeighting > & GlobalPond();
        const cTplValGesInit< cRigidBlockWeighting > & GlobalPond()const ;

        cTplValGesInit< cRigidBlockWeighting > & RelTimePond();
        const cTplValGesInit< cRigidBlockWeighting > & RelTimePond()const ;

        cTplValGesInit< cRigidBlockWeighting > & GlobalDistPond();
        const cTplValGesInit< cRigidBlockWeighting > & GlobalDistPond()const ;

        cTplValGesInit< cRigidBlockWeighting > & RelTimeDistPond();
        const cTplValGesInit< cRigidBlockWeighting > & RelTimeDistPond()const ;
    private:
        std::string mId;
        cTplValGesInit< bool > mShow;
        cTplValGesInit< cRigidBlockWeighting > mGlobalPond;
        cTplValGesInit< cRigidBlockWeighting > mRelTimePond;
        cTplValGesInit< cRigidBlockWeighting > mGlobalDistPond;
        cTplValGesInit< cRigidBlockWeighting > mRelTimeDistPond;
};
cElXMLTree * ToXMLTree(const cObsBlockCamRig &);

void  BinaryDumpInFile(ELISE_fp &,const cObsBlockCamRig &);

void  BinaryUnDumpFromFile(cObsBlockCamRig &,ELISE_fp &);

std::string  Mangling( cObsBlockCamRig *);

class cObsCenterInPlane
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsCenterInPlane & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;
    private:
        std::string mId;
};
cElXMLTree * ToXMLTree(const cObsCenterInPlane &);

void  BinaryDumpInFile(ELISE_fp &,const cObsCenterInPlane &);

void  BinaryUnDumpFromFile(cObsCenterInPlane &,ELISE_fp &);

std::string  Mangling( cObsCenterInPlane *);

class cROA_FichierImg
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cROA_FichierImg & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        double & Sz();
        const double & Sz()const ;

        double & Exag();
        const double & Exag()const ;

        cTplValGesInit< bool > & VisuVideo();
        const cTplValGesInit< bool > & VisuVideo()const ;
    private:
        std::string mName;
        double mSz;
        double mExag;
        cTplValGesInit< bool > mVisuVideo;
};
cElXMLTree * ToXMLTree(const cROA_FichierImg &);

void  BinaryDumpInFile(ELISE_fp &,const cROA_FichierImg &);

void  BinaryUnDumpFromFile(cROA_FichierImg &,ELISE_fp &);

std::string  Mangling( cROA_FichierImg *);

class cRapportObsAppui
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRapportObsAppui & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & OnlyLastIter();
        const cTplValGesInit< bool > & OnlyLastIter()const ;

        std::string & FichierTxt();
        const std::string & FichierTxt()const ;

        cTplValGesInit< bool > & ColPerPose();
        const cTplValGesInit< bool > & ColPerPose()const ;

        cTplValGesInit< double > & SeuilColOut();
        const cTplValGesInit< double > & SeuilColOut()const ;

        std::string & Name();
        const std::string & Name()const ;

        double & Sz();
        const double & Sz()const ;

        double & Exag();
        const double & Exag()const ;

        cTplValGesInit< bool > & VisuVideo();
        const cTplValGesInit< bool > & VisuVideo()const ;

        cTplValGesInit< cROA_FichierImg > & ROA_FichierImg();
        const cTplValGesInit< cROA_FichierImg > & ROA_FichierImg()const ;
    private:
        cTplValGesInit< bool > mOnlyLastIter;
        std::string mFichierTxt;
        cTplValGesInit< bool > mColPerPose;
        cTplValGesInit< double > mSeuilColOut;
        cTplValGesInit< cROA_FichierImg > mROA_FichierImg;
};
cElXMLTree * ToXMLTree(const cRapportObsAppui &);

void  BinaryDumpInFile(ELISE_fp &,const cRapportObsAppui &);

void  BinaryUnDumpFromFile(cRapportObsAppui &,ELISE_fp &);

std::string  Mangling( cRapportObsAppui *);

class cObsAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsAppuis & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        cTplValGesInit< bool > & OnlyLastIter();
        const cTplValGesInit< bool > & OnlyLastIter()const ;

        std::string & FichierTxt();
        const std::string & FichierTxt()const ;

        cTplValGesInit< bool > & ColPerPose();
        const cTplValGesInit< bool > & ColPerPose()const ;

        cTplValGesInit< double > & SeuilColOut();
        const cTplValGesInit< double > & SeuilColOut()const ;

        std::string & Name();
        const std::string & Name()const ;

        double & Sz();
        const double & Sz()const ;

        double & Exag();
        const double & Exag()const ;

        cTplValGesInit< bool > & VisuVideo();
        const cTplValGesInit< bool > & VisuVideo()const ;

        cTplValGesInit< cROA_FichierImg > & ROA_FichierImg();
        const cTplValGesInit< cROA_FichierImg > & ROA_FichierImg()const ;

        cTplValGesInit< cRapportObsAppui > & RapportObsAppui();
        const cTplValGesInit< cRapportObsAppui > & RapportObsAppui()const ;
    private:
        std::string mNameRef;
        cPonderationPackMesure mPond;
        cTplValGesInit< cRapportObsAppui > mRapportObsAppui;
};
cElXMLTree * ToXMLTree(const cObsAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cObsAppuis &);

void  BinaryUnDumpFromFile(cObsAppuis &,ELISE_fp &);

std::string  Mangling( cObsAppuis *);

class cObsAppuisFlottant
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsAppuisFlottant & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;

        cPonderationPackMesure & PondIm();
        const cPonderationPackMesure & PondIm()const ;

        std::list< cElRegex_Ptr > & PtsShowDet();
        const std::list< cElRegex_Ptr > & PtsShowDet()const ;

        cTplValGesInit< bool > & DetShow3D();
        const cTplValGesInit< bool > & DetShow3D()const ;

        cTplValGesInit< double > & NivAlerteDetail();
        const cTplValGesInit< double > & NivAlerteDetail()const ;

        cTplValGesInit< bool > & ShowMax();
        const cTplValGesInit< bool > & ShowMax()const ;

        cTplValGesInit< bool > & ShowSom();
        const cTplValGesInit< bool > & ShowSom()const ;

        cTplValGesInit< bool > & ShowUnused();
        const cTplValGesInit< bool > & ShowUnused()const ;
    private:
        std::string mNameRef;
        cPonderationPackMesure mPondIm;
        std::list< cElRegex_Ptr > mPtsShowDet;
        cTplValGesInit< bool > mDetShow3D;
        cTplValGesInit< double > mNivAlerteDetail;
        cTplValGesInit< bool > mShowMax;
        cTplValGesInit< bool > mShowSom;
        cTplValGesInit< bool > mShowUnused;
};
cElXMLTree * ToXMLTree(const cObsAppuisFlottant &);

void  BinaryDumpInFile(ELISE_fp &,const cObsAppuisFlottant &);

void  BinaryUnDumpFromFile(cObsAppuisFlottant &,ELISE_fp &);

std::string  Mangling( cObsAppuisFlottant *);

class cRappelOnZ
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRappelOnZ & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyGrpApply();
        const cTplValGesInit< std::string > & KeyGrpApply()const ;

        double & Z();
        const double & Z()const ;

        double & IncC();
        const double & IncC()const ;

        cTplValGesInit< double > & IncE();
        const cTplValGesInit< double > & IncE()const ;

        cTplValGesInit< double > & SeuilR();
        const cTplValGesInit< double > & SeuilR()const ;

        cTplValGesInit< std::string > & LayerMasq();
        const cTplValGesInit< std::string > & LayerMasq()const ;
    private:
        cTplValGesInit< std::string > mKeyGrpApply;
        double mZ;
        double mIncC;
        cTplValGesInit< double > mIncE;
        cTplValGesInit< double > mSeuilR;
        cTplValGesInit< std::string > mLayerMasq;
};
cElXMLTree * ToXMLTree(const cRappelOnZ &);

void  BinaryDumpInFile(ELISE_fp &,const cRappelOnZ &);

void  BinaryUnDumpFromFile(cRappelOnZ &,ELISE_fp &);

std::string  Mangling( cRappelOnZ *);

class cObsLiaisons
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsLiaisons & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        cTplValGesInit< cPonderationPackMesure > & PondSurf();
        const cTplValGesInit< cPonderationPackMesure > & PondSurf()const ;

        cTplValGesInit< std::string > & KeyGrpApply();
        const cTplValGesInit< std::string > & KeyGrpApply()const ;

        double & Z();
        const double & Z()const ;

        double & IncC();
        const double & IncC()const ;

        cTplValGesInit< double > & IncE();
        const cTplValGesInit< double > & IncE()const ;

        cTplValGesInit< double > & SeuilR();
        const cTplValGesInit< double > & SeuilR()const ;

        cTplValGesInit< std::string > & LayerMasq();
        const cTplValGesInit< std::string > & LayerMasq()const ;

        cTplValGesInit< cRappelOnZ > & RappelOnZ();
        const cTplValGesInit< cRappelOnZ > & RappelOnZ()const ;
    private:
        std::string mNameRef;
        cPonderationPackMesure mPond;
        cTplValGesInit< cPonderationPackMesure > mPondSurf;
        cTplValGesInit< cRappelOnZ > mRappelOnZ;
};
cElXMLTree * ToXMLTree(const cObsLiaisons &);

void  BinaryDumpInFile(ELISE_fp &,const cObsLiaisons &);

void  BinaryUnDumpFromFile(cObsLiaisons &,ELISE_fp &);

std::string  Mangling( cObsLiaisons *);

class cObsCentrePDV
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsCentrePDV & anObj,cElXMLTree * aTree);


        cTplValGesInit< cElRegex_Ptr > & PatternApply();
        const cTplValGesInit< cElRegex_Ptr > & PatternApply()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        cTplValGesInit< cPonderationPackMesure > & PondAlti();
        const cTplValGesInit< cPonderationPackMesure > & PondAlti()const ;

        cTplValGesInit< bool > & ShowTestVitesse();
        const cTplValGesInit< bool > & ShowTestVitesse()const ;
    private:
        cTplValGesInit< cElRegex_Ptr > mPatternApply;
        cPonderationPackMesure mPond;
        cTplValGesInit< cPonderationPackMesure > mPondAlti;
        cTplValGesInit< bool > mShowTestVitesse;
};
cElXMLTree * ToXMLTree(const cObsCentrePDV &);

void  BinaryDumpInFile(ELISE_fp &,const cObsCentrePDV &);

void  BinaryUnDumpFromFile(cObsCentrePDV &,ELISE_fp &);

std::string  Mangling( cObsCentrePDV *);

class cORGI_CentreCommun
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cORGI_CentreCommun & anObj,cElXMLTree * aTree);


        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;
    private:
        Pt3dr mIncertitude;
};
cElXMLTree * ToXMLTree(const cORGI_CentreCommun &);

void  BinaryDumpInFile(ELISE_fp &,const cORGI_CentreCommun &);

void  BinaryUnDumpFromFile(cORGI_CentreCommun &,ELISE_fp &);

std::string  Mangling( cORGI_CentreCommun *);

class cORGI_TetaCommun
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cORGI_TetaCommun & anObj,cElXMLTree * aTree);


        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;
    private:
        Pt3dr mIncertitude;
};
cElXMLTree * ToXMLTree(const cORGI_TetaCommun &);

void  BinaryDumpInFile(ELISE_fp &,const cORGI_TetaCommun &);

void  BinaryUnDumpFromFile(cORGI_TetaCommun &,ELISE_fp &);

std::string  Mangling( cORGI_TetaCommun *);

class cObsRigidGrpImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsRigidGrpImage & anObj,cElXMLTree * aTree);


        std::string & RefGrp();
        const std::string & RefGrp()const ;

        cTplValGesInit< cORGI_CentreCommun > & ORGI_CentreCommun();
        const cTplValGesInit< cORGI_CentreCommun > & ORGI_CentreCommun()const ;

        cTplValGesInit< cORGI_TetaCommun > & ORGI_TetaCommun();
        const cTplValGesInit< cORGI_TetaCommun > & ORGI_TetaCommun()const ;
    private:
        std::string mRefGrp;
        cTplValGesInit< cORGI_CentreCommun > mORGI_CentreCommun;
        cTplValGesInit< cORGI_TetaCommun > mORGI_TetaCommun;
};
cElXMLTree * ToXMLTree(const cObsRigidGrpImage &);

void  BinaryDumpInFile(ELISE_fp &,const cObsRigidGrpImage &);

void  BinaryUnDumpFromFile(cObsRigidGrpImage &,ELISE_fp &);

std::string  Mangling( cObsRigidGrpImage *);

class cTxtRapDetaille
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTxtRapDetaille & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;
    private:
        std::string mNameFile;
};
cElXMLTree * ToXMLTree(const cTxtRapDetaille &);

void  BinaryDumpInFile(ELISE_fp &,const cTxtRapDetaille &);

void  BinaryUnDumpFromFile(cTxtRapDetaille &,ELISE_fp &);

std::string  Mangling( cTxtRapDetaille *);

class cObsRelGPS
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cObsRelGPS & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        cGpsRelativeWeighting & Pond();
        const cGpsRelativeWeighting & Pond()const ;
    private:
        std::string mId;
        cGpsRelativeWeighting mPond;
};
cElXMLTree * ToXMLTree(const cObsRelGPS &);

void  BinaryDumpInFile(ELISE_fp &,const cObsRelGPS &);

void  BinaryUnDumpFromFile(cObsRelGPS &,ELISE_fp &);

std::string  Mangling( cObsRelGPS *);

class cSectionObservations
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionObservations & anObj,cElXMLTree * aTree);


        double & SigmaPix();
        const double & SigmaPix()const ;

        int & NbGrid();
        const int & NbGrid()const ;

        cTplValGesInit< cContrCamConseq > & ContrCamConseq();
        const cTplValGesInit< cContrCamConseq > & ContrCamConseq()const ;

        std::list< cContrCamGenInc > & ContrCamGenInc();
        const std::list< cContrCamGenInc > & ContrCamGenInc()const ;

        std::list< cObsBlockCamRig > & ObsBlockCamRig();
        const std::list< cObsBlockCamRig > & ObsBlockCamRig()const ;

        std::list< cObsCenterInPlane > & ObsCenterInPlane();
        const std::list< cObsCenterInPlane > & ObsCenterInPlane()const ;

        std::list< cObsAppuis > & ObsAppuis();
        const std::list< cObsAppuis > & ObsAppuis()const ;

        std::list< cObsAppuisFlottant > & ObsAppuisFlottant();
        const std::list< cObsAppuisFlottant > & ObsAppuisFlottant()const ;

        std::list< cObsLiaisons > & ObsLiaisons();
        const std::list< cObsLiaisons > & ObsLiaisons()const ;

        std::list< cObsCentrePDV > & ObsCentrePDV();
        const std::list< cObsCentrePDV > & ObsCentrePDV()const ;

        std::list< cObsRigidGrpImage > & ObsRigidGrpImage();
        const std::list< cObsRigidGrpImage > & ObsRigidGrpImage()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        cTplValGesInit< cTxtRapDetaille > & TxtRapDetaille();
        const cTplValGesInit< cTxtRapDetaille > & TxtRapDetaille()const ;

        std::list< cObsRelGPS > & ObsRelGPS();
        const std::list< cObsRelGPS > & ObsRelGPS()const ;
    private:
        cTplValGesInit< cContrCamConseq > mContrCamConseq;
        std::list< cContrCamGenInc > mContrCamGenInc;
        std::list< cObsBlockCamRig > mObsBlockCamRig;
        std::list< cObsCenterInPlane > mObsCenterInPlane;
        std::list< cObsAppuis > mObsAppuis;
        std::list< cObsAppuisFlottant > mObsAppuisFlottant;
        std::list< cObsLiaisons > mObsLiaisons;
        std::list< cObsCentrePDV > mObsCentrePDV;
        std::list< cObsRigidGrpImage > mObsRigidGrpImage;
        cTplValGesInit< cTxtRapDetaille > mTxtRapDetaille;
        std::list< cObsRelGPS > mObsRelGPS;
};
cElXMLTree * ToXMLTree(const cSectionObservations &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionObservations &);

void  BinaryUnDumpFromFile(cSectionObservations &,ELISE_fp &);

std::string  Mangling( cSectionObservations *);

class cExportAsGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportAsGrid & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DoExport();
        const cTplValGesInit< bool > & DoExport()const ;

        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< std::string > & XML_Supl();
        const cTplValGesInit< std::string > & XML_Supl()const ;

        cTplValGesInit< bool > & XML_Autonome();
        const cTplValGesInit< bool > & XML_Autonome()const ;

        cTplValGesInit< Pt2dr > & RabPt();
        const cTplValGesInit< Pt2dr > & RabPt()const ;

        cTplValGesInit< Pt2dr > & Step();
        const cTplValGesInit< Pt2dr > & Step()const ;
    private:
        cTplValGesInit< bool > mDoExport;
        std::string mName;
        cTplValGesInit< std::string > mXML_Supl;
        cTplValGesInit< bool > mXML_Autonome;
        cTplValGesInit< Pt2dr > mRabPt;
        cTplValGesInit< Pt2dr > mStep;
};
cElXMLTree * ToXMLTree(const cExportAsGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cExportAsGrid &);

void  BinaryUnDumpFromFile(cExportAsGrid &,ELISE_fp &);

std::string  Mangling( cExportAsGrid *);

class cExportCalib
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportCalib & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< std::string > & Prefix();
        const cTplValGesInit< std::string > & Prefix()const ;

        cTplValGesInit< std::string > & Postfix();
        const cTplValGesInit< std::string > & Postfix()const ;

        cTplValGesInit< bool > & KeyIsName();
        const cTplValGesInit< bool > & KeyIsName()const ;

        cTplValGesInit< bool > & DoExport();
        const cTplValGesInit< bool > & DoExport()const ;

        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< std::string > & XML_Supl();
        const cTplValGesInit< std::string > & XML_Supl()const ;

        cTplValGesInit< bool > & XML_Autonome();
        const cTplValGesInit< bool > & XML_Autonome()const ;

        cTplValGesInit< Pt2dr > & RabPt();
        const cTplValGesInit< Pt2dr > & RabPt()const ;

        cTplValGesInit< Pt2dr > & Step();
        const cTplValGesInit< Pt2dr > & Step()const ;

        cTplValGesInit< cExportAsGrid > & ExportAsGrid();
        const cTplValGesInit< cExportAsGrid > & ExportAsGrid()const ;

        cTplValGesInit< cExportAsNewGrid > & ExportAsNewGrid();
        const cTplValGesInit< cExportAsNewGrid > & ExportAsNewGrid()const ;
    private:
        cTplValGesInit< std::string > mPatternSel;
        std::string mKeyAssoc;
        cTplValGesInit< std::string > mPrefix;
        cTplValGesInit< std::string > mPostfix;
        cTplValGesInit< bool > mKeyIsName;
        cTplValGesInit< cExportAsGrid > mExportAsGrid;
        cTplValGesInit< cExportAsNewGrid > mExportAsNewGrid;
};
cElXMLTree * ToXMLTree(const cExportCalib &);

void  BinaryDumpInFile(ELISE_fp &,const cExportCalib &);

void  BinaryUnDumpFromFile(cExportCalib &,ELISE_fp &);

std::string  Mangling( cExportCalib *);

class cForce2ObsOnC
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cForce2ObsOnC & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & WhenExist();
        const cTplValGesInit< bool > & WhenExist()const ;
    private:
        cTplValGesInit< bool > mWhenExist;
};
cElXMLTree * ToXMLTree(const cForce2ObsOnC &);

void  BinaryDumpInFile(ELISE_fp &,const cForce2ObsOnC &);

void  BinaryUnDumpFromFile(cForce2ObsOnC &,ELISE_fp &);

std::string  Mangling( cForce2ObsOnC *);

class cExportPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportPose & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & WhenExist();
        const cTplValGesInit< bool > & WhenExist()const ;

        cTplValGesInit< cForce2ObsOnC > & Force2ObsOnC();
        const cTplValGesInit< cForce2ObsOnC > & Force2ObsOnC()const ;

        cTplValGesInit< std::string > & ChC();
        const cTplValGesInit< std::string > & ChC()const ;

        cTplValGesInit< bool > & ChCForceRot();
        const cTplValGesInit< bool > & ChCForceRot()const ;

        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< std::string > & StdNameMMDir();
        const cTplValGesInit< std::string > & StdNameMMDir()const ;

        cTplValGesInit< bool > & AddCalib();
        const cTplValGesInit< bool > & AddCalib()const ;

        cTplValGesInit< cExportAsNewGrid > & ExportAsNewGrid();
        const cTplValGesInit< cExportAsNewGrid > & ExportAsNewGrid()const ;

        cTplValGesInit< std::string > & FileExtern();
        const cTplValGesInit< std::string > & FileExtern()const ;

        cTplValGesInit< bool > & FileExternIsKey();
        const cTplValGesInit< bool > & FileExternIsKey()const ;

        cTplValGesInit< bool > & CalcKeyFromCalib();
        const cTplValGesInit< bool > & CalcKeyFromCalib()const ;

        cTplValGesInit< bool > & RelativeNameFE();
        const cTplValGesInit< bool > & RelativeNameFE()const ;

        cTplValGesInit< bool > & ModeAngulaire();
        const cTplValGesInit< bool > & ModeAngulaire()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< int > & NbVerif();
        const cTplValGesInit< int > & NbVerif()const ;

        cTplValGesInit< Pt3di > & VerifDeterm();
        const cTplValGesInit< Pt3di > & VerifDeterm()const ;

        cTplValGesInit< bool > & ShowWhenVerif();
        const cTplValGesInit< bool > & ShowWhenVerif()const ;

        cTplValGesInit< double > & TolWhenVerif();
        const cTplValGesInit< double > & TolWhenVerif()const ;
    private:
        cTplValGesInit< cForce2ObsOnC > mForce2ObsOnC;
        cTplValGesInit< std::string > mChC;
        cTplValGesInit< bool > mChCForceRot;
        std::string mKeyAssoc;
        cTplValGesInit< std::string > mStdNameMMDir;
        cTplValGesInit< bool > mAddCalib;
        cTplValGesInit< cExportAsNewGrid > mExportAsNewGrid;
        cTplValGesInit< std::string > mFileExtern;
        cTplValGesInit< bool > mFileExternIsKey;
        cTplValGesInit< bool > mCalcKeyFromCalib;
        cTplValGesInit< bool > mRelativeNameFE;
        cTplValGesInit< bool > mModeAngulaire;
        cTplValGesInit< std::string > mPatternSel;
        cTplValGesInit< int > mNbVerif;
        cTplValGesInit< Pt3di > mVerifDeterm;
        cTplValGesInit< bool > mShowWhenVerif;
        cTplValGesInit< double > mTolWhenVerif;
};
cElXMLTree * ToXMLTree(const cExportPose &);

void  BinaryDumpInFile(ELISE_fp &,const cExportPose &);

void  BinaryUnDumpFromFile(cExportPose &,ELISE_fp &);

std::string  Mangling( cExportPose *);

class cExportAttrPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportAttrPose & anObj,cElXMLTree * aTree);


        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

        cTplValGesInit< std::string > & AttrSup();
        const cTplValGesInit< std::string > & AttrSup()const ;

        std::string & PatternApply();
        const std::string & PatternApply()const ;

        cTplValGesInit< cParamEstimPlan > & ExportDirVerticaleLocale();
        const cTplValGesInit< cParamEstimPlan > & ExportDirVerticaleLocale()const ;
    private:
        std::string mKeyAssoc;
        cTplValGesInit< std::string > mAttrSup;
        std::string mPatternApply;
        cTplValGesInit< cParamEstimPlan > mExportDirVerticaleLocale;
};
cElXMLTree * ToXMLTree(const cExportAttrPose &);

void  BinaryDumpInFile(ELISE_fp &,const cExportAttrPose &);

void  BinaryUnDumpFromFile(cExportAttrPose &,ELISE_fp &);

std::string  Mangling( cExportAttrPose *);

class cExportOrthoCyl
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportOrthoCyl & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        cTplValGesInit< std::string > & PatternEstimAxe();
        const cTplValGesInit< std::string > & PatternEstimAxe()const ;

        bool & AngulCorr();
        const bool & AngulCorr()const ;

        cTplValGesInit< bool > & L2EstimAxe();
        const cTplValGesInit< bool > & L2EstimAxe()const ;
    private:
        cTplValGesInit< bool > mUseIt;
        cTplValGesInit< std::string > mPatternEstimAxe;
        bool mAngulCorr;
        cTplValGesInit< bool > mL2EstimAxe;
};
cElXMLTree * ToXMLTree(const cExportOrthoCyl &);

void  BinaryDumpInFile(ELISE_fp &,const cExportOrthoCyl &);

void  BinaryUnDumpFromFile(cExportOrthoCyl &,ELISE_fp &);

std::string  Mangling( cExportOrthoCyl *);

class cExportRepereLoc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportRepereLoc & anObj,cElXMLTree * aTree);


        std::string & NameRepere();
        const std::string & NameRepere()const ;

        std::string & PatternEstimPl();
        const std::string & PatternEstimPl()const ;

        cParamEstimPlan & EstimPlanHor();
        const cParamEstimPlan & EstimPlanHor()const ;

        cTplValGesInit< std::string > & ImP1P2();
        const cTplValGesInit< std::string > & ImP1P2()const ;

        Pt2dr & P1();
        const Pt2dr & P1()const ;

        Pt2dr & P2();
        const Pt2dr & P2()const ;

        cTplValGesInit< Pt2dr > & AxeDef();
        const cTplValGesInit< Pt2dr > & AxeDef()const ;

        cTplValGesInit< Pt2dr > & Origine();
        const cTplValGesInit< Pt2dr > & Origine()const ;

        cTplValGesInit< std::string > & NameImOri();
        const cTplValGesInit< std::string > & NameImOri()const ;

        cTplValGesInit< bool > & P1P2Hor();
        const cTplValGesInit< bool > & P1P2Hor()const ;

        cTplValGesInit< bool > & P1P2HorYVert();
        const cTplValGesInit< bool > & P1P2HorYVert()const ;

        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        cTplValGesInit< std::string > & PatternEstimAxe();
        const cTplValGesInit< std::string > & PatternEstimAxe()const ;

        bool & AngulCorr();
        const bool & AngulCorr()const ;

        cTplValGesInit< bool > & L2EstimAxe();
        const cTplValGesInit< bool > & L2EstimAxe()const ;

        cTplValGesInit< cExportOrthoCyl > & ExportOrthoCyl();
        const cTplValGesInit< cExportOrthoCyl > & ExportOrthoCyl()const ;
    private:
        std::string mNameRepere;
        std::string mPatternEstimPl;
        cParamEstimPlan mEstimPlanHor;
        cTplValGesInit< std::string > mImP1P2;
        Pt2dr mP1;
        Pt2dr mP2;
        cTplValGesInit< Pt2dr > mAxeDef;
        cTplValGesInit< Pt2dr > mOrigine;
        cTplValGesInit< std::string > mNameImOri;
        cTplValGesInit< bool > mP1P2Hor;
        cTplValGesInit< bool > mP1P2HorYVert;
        cTplValGesInit< cExportOrthoCyl > mExportOrthoCyl;
};
cElXMLTree * ToXMLTree(const cExportRepereLoc &);

void  BinaryDumpInFile(ELISE_fp &,const cExportRepereLoc &);

void  BinaryUnDumpFromFile(cExportRepereLoc &,ELISE_fp &);

std::string  Mangling( cExportRepereLoc *);

class cExportBlockCamera
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportBlockCamera & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        cTplValGesInit< cXml_EstimateOrientationInitBlockCamera > & Estimate();
        const cTplValGesInit< cXml_EstimateOrientationInitBlockCamera > & Estimate()const ;
    private:
        std::string mId;
        std::string mNameFile;
        cTplValGesInit< cXml_EstimateOrientationInitBlockCamera > mEstimate;
};
cElXMLTree * ToXMLTree(const cExportBlockCamera &);

void  BinaryDumpInFile(ELISE_fp &,const cExportBlockCamera &);

void  BinaryUnDumpFromFile(cExportBlockCamera &,ELISE_fp &);

std::string  Mangling( cExportBlockCamera *);

class cCartes2Export
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCartes2Export & anObj,cElXMLTree * aTree);


        std::list< std::string > & Im1();
        const std::list< std::string > & Im1()const ;

        std::string & Nuage();
        const std::string & Nuage()const ;

        std::list< std::string > & ImN();
        const std::list< std::string > & ImN()const ;

        cTplValGesInit< std::string > & FilterIm2();
        const cTplValGesInit< std::string > & FilterIm2()const ;
    private:
        std::list< std::string > mIm1;
        std::string mNuage;
        std::list< std::string > mImN;
        cTplValGesInit< std::string > mFilterIm2;
};
cElXMLTree * ToXMLTree(const cCartes2Export &);

void  BinaryDumpInFile(ELISE_fp &,const cCartes2Export &);

void  BinaryUnDumpFromFile(cCartes2Export &,ELISE_fp &);

std::string  Mangling( cCartes2Export *);

class cExportMesuresFromCarteProf
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportMesuresFromCarteProf & anObj,cElXMLTree * aTree);


        std::list< cCartes2Export > & Cartes2Export();
        const std::list< cCartes2Export > & Cartes2Export()const ;

        std::string & IdBdLiaisonIn();
        const std::string & IdBdLiaisonIn()const ;

        cTplValGesInit< std::string > & KeyAssocLiaisons12();
        const cTplValGesInit< std::string > & KeyAssocLiaisons12()const ;

        cTplValGesInit< std::string > & KeyAssocLiaisons21();
        const cTplValGesInit< std::string > & KeyAssocLiaisons21()const ;

        cTplValGesInit< std::string > & KeyAssocAppuis();
        const cTplValGesInit< std::string > & KeyAssocAppuis()const ;

        cTplValGesInit< bool > & AppuisModeAdd();
        const cTplValGesInit< bool > & AppuisModeAdd()const ;

        cTplValGesInit< bool > & LiaisonModeAdd();
        const cTplValGesInit< bool > & LiaisonModeAdd()const ;
    private:
        std::list< cCartes2Export > mCartes2Export;
        std::string mIdBdLiaisonIn;
        cTplValGesInit< std::string > mKeyAssocLiaisons12;
        cTplValGesInit< std::string > mKeyAssocLiaisons21;
        cTplValGesInit< std::string > mKeyAssocAppuis;
        cTplValGesInit< bool > mAppuisModeAdd;
        cTplValGesInit< bool > mLiaisonModeAdd;
};
cElXMLTree * ToXMLTree(const cExportMesuresFromCarteProf &);

void  BinaryDumpInFile(ELISE_fp &,const cExportMesuresFromCarteProf &);

void  BinaryUnDumpFromFile(cExportMesuresFromCarteProf &,ELISE_fp &);

std::string  Mangling( cExportMesuresFromCarteProf *);

class cExportVisuConfigGrpPose
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportVisuConfigGrpPose & anObj,cElXMLTree * aTree);


        std::list< std::string > & PatternSel();
        const std::list< std::string > & PatternSel()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;
    private:
        std::list< std::string > mPatternSel;
        std::string mNameFile;
};
cElXMLTree * ToXMLTree(const cExportVisuConfigGrpPose &);

void  BinaryDumpInFile(ELISE_fp &,const cExportVisuConfigGrpPose &);

void  BinaryUnDumpFromFile(cExportVisuConfigGrpPose &,ELISE_fp &);

std::string  Mangling( cExportVisuConfigGrpPose *);

class cExportPtsFlottant
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportPtsFlottant & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< std::string > & NameFileXml();
        const cTplValGesInit< std::string > & NameFileXml()const ;

        cTplValGesInit< std::string > & NameFileTxt();
        const cTplValGesInit< std::string > & NameFileTxt()const ;

        cTplValGesInit< std::string > & NameFileJSON();
        const cTplValGesInit< std::string > & NameFileJSON()const ;

        cTplValGesInit< std::string > & TextComplTxt();
        const cTplValGesInit< std::string > & TextComplTxt()const ;
    private:
        cTplValGesInit< std::string > mPatternSel;
        cTplValGesInit< std::string > mNameFileXml;
        cTplValGesInit< std::string > mNameFileTxt;
        cTplValGesInit< std::string > mNameFileJSON;
        cTplValGesInit< std::string > mTextComplTxt;
};
cElXMLTree * ToXMLTree(const cExportPtsFlottant &);

void  BinaryDumpInFile(ELISE_fp &,const cExportPtsFlottant &);

void  BinaryUnDumpFromFile(cExportPtsFlottant &,ELISE_fp &);

std::string  Mangling( cExportPtsFlottant *);

class cResidusIndiv
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cResidusIndiv & anObj,cElXMLTree * aTree);


        std::string & Pattern();
        const std::string & Pattern()const ;

        std::string & Name();
        const std::string & Name()const ;
    private:
        std::string mPattern;
        std::string mName;
};
cElXMLTree * ToXMLTree(const cResidusIndiv &);

void  BinaryDumpInFile(ELISE_fp &,const cResidusIndiv &);

void  BinaryUnDumpFromFile(cResidusIndiv &,ELISE_fp &);

std::string  Mangling( cResidusIndiv *);

class cExportImResiduLiaison
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportImResiduLiaison & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & Signed();
        const cTplValGesInit< bool > & Signed()const ;

        std::string & PatternGlobCalIm();
        const std::string & PatternGlobCalIm()const ;

        std::string & NameGlobCalIm();
        const std::string & NameGlobCalIm()const ;

        double & ScaleIm();
        const double & ScaleIm()const ;

        double & DynIm();
        const double & DynIm()const ;

        std::string & Pattern();
        const std::string & Pattern()const ;

        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< cResidusIndiv > & ResidusIndiv();
        const cTplValGesInit< cResidusIndiv > & ResidusIndiv()const ;
    private:
        cTplValGesInit< bool > mSigned;
        std::string mPatternGlobCalIm;
        std::string mNameGlobCalIm;
        double mScaleIm;
        double mDynIm;
        cTplValGesInit< cResidusIndiv > mResidusIndiv;
};
cElXMLTree * ToXMLTree(const cExportImResiduLiaison &);

void  BinaryDumpInFile(ELISE_fp &,const cExportImResiduLiaison &);

void  BinaryUnDumpFromFile(cExportImResiduLiaison &,ELISE_fp &);

std::string  Mangling( cExportImResiduLiaison *);

class cExportRedressement
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportRedressement & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & Dyn();
        const cTplValGesInit< double > & Dyn()const ;

        cTplValGesInit< double > & Gamma();
        const cTplValGesInit< double > & Gamma()const ;

        cTplValGesInit< eTypeNumerique > & TypeNum();
        const cTplValGesInit< eTypeNumerique > & TypeNum()const ;

        cTplValGesInit< double > & Offset();
        const cTplValGesInit< double > & Offset()const ;

        cTplValGesInit< cElRegex_Ptr > & PatternSel();
        const cTplValGesInit< cElRegex_Ptr > & PatternSel()const ;

        cTplValGesInit< std::string > & KeyAssocIn();
        const cTplValGesInit< std::string > & KeyAssocIn()const ;

        cTplValGesInit< Pt2dr > & OffsetIm();
        const cTplValGesInit< Pt2dr > & OffsetIm()const ;

        cTplValGesInit< double > & ScaleIm();
        const cTplValGesInit< double > & ScaleIm()const ;

        std::string & KeyAssocOut();
        const std::string & KeyAssocOut()const ;

        cTplValGesInit< double > & ZSol();
        const cTplValGesInit< double > & ZSol()const ;

        double & Resol();
        const double & Resol()const ;

        bool & ResolIsRel();
        const bool & ResolIsRel()const ;

        cTplValGesInit< bool > & DoTFW();
        const cTplValGesInit< bool > & DoTFW()const ;

        double & TetaLimite();
        const double & TetaLimite()const ;

        cTplValGesInit< Pt3dr > & DirTetaLim();
        const cTplValGesInit< Pt3dr > & DirTetaLim()const ;

        cTplValGesInit< bool > & DoOnlyIfNew();
        const cTplValGesInit< bool > & DoOnlyIfNew()const ;
    private:
        cTplValGesInit< double > mDyn;
        cTplValGesInit< double > mGamma;
        cTplValGesInit< eTypeNumerique > mTypeNum;
        cTplValGesInit< double > mOffset;
        cTplValGesInit< cElRegex_Ptr > mPatternSel;
        cTplValGesInit< std::string > mKeyAssocIn;
        cTplValGesInit< Pt2dr > mOffsetIm;
        cTplValGesInit< double > mScaleIm;
        std::string mKeyAssocOut;
        cTplValGesInit< double > mZSol;
        double mResol;
        bool mResolIsRel;
        cTplValGesInit< bool > mDoTFW;
        double mTetaLimite;
        cTplValGesInit< Pt3dr > mDirTetaLim;
        cTplValGesInit< bool > mDoOnlyIfNew;
};
cElXMLTree * ToXMLTree(const cExportRedressement &);

void  BinaryDumpInFile(ELISE_fp &,const cExportRedressement &);

void  BinaryUnDumpFromFile(cExportRedressement &,ELISE_fp &);

std::string  Mangling( cExportRedressement *);

class cExportNuageByImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportNuageByImage & anObj,cElXMLTree * aTree);


        std::string & KeyCalc();
        const std::string & KeyCalc()const ;

        cTplValGesInit< bool > & SymPts();
        const cTplValGesInit< bool > & SymPts()const ;
    private:
        std::string mKeyCalc;
        cTplValGesInit< bool > mSymPts;
};
cElXMLTree * ToXMLTree(const cExportNuageByImage &);

void  BinaryDumpInFile(ELISE_fp &,const cExportNuageByImage &);

void  BinaryUnDumpFromFile(cExportNuageByImage &,ELISE_fp &);

std::string  Mangling( cExportNuageByImage *);

class cNuagePutCam
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNuagePutCam & anObj,cElXMLTree * aTree);


        Pt3di & ColCadre();
        const Pt3di & ColCadre()const ;

        cTplValGesInit< Pt3di > & ColRay();
        const cTplValGesInit< Pt3di > & ColRay()const ;

        double & Long();
        const double & Long()const ;

        double & StepSeg();
        const double & StepSeg()const ;

        cTplValGesInit< std::string > & KeyCalName();
        const cTplValGesInit< std::string > & KeyCalName()const ;

        cTplValGesInit< double > & StepImage();
        const cTplValGesInit< double > & StepImage()const ;

        cTplValGesInit< std::string > & HomolRay();
        const cTplValGesInit< std::string > & HomolRay()const ;

        cTplValGesInit< Pt3di > & ColRayHomol();
        const cTplValGesInit< Pt3di > & ColRayHomol()const ;
    private:
        Pt3di mColCadre;
        cTplValGesInit< Pt3di > mColRay;
        double mLong;
        double mStepSeg;
        cTplValGesInit< std::string > mKeyCalName;
        cTplValGesInit< double > mStepImage;
        cTplValGesInit< std::string > mHomolRay;
        cTplValGesInit< Pt3di > mColRayHomol;
};
cElXMLTree * ToXMLTree(const cNuagePutCam &);

void  BinaryDumpInFile(ELISE_fp &,const cNuagePutCam &);

void  BinaryUnDumpFromFile(cNuagePutCam &,ELISE_fp &);

std::string  Mangling( cNuagePutCam *);

class cNuagePutInterPMul
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNuagePutInterPMul & anObj,cElXMLTree * aTree);


        std::string & NamePMul();
        const std::string & NamePMul()const ;

        double & StepDr();
        const double & StepDr()const ;

        cTplValGesInit< double > & RabDr();
        const cTplValGesInit< double > & RabDr()const ;

        Pt3di & ColRayInter();
        const Pt3di & ColRayInter()const ;

        cTplValGesInit< double > & Epais();
        const cTplValGesInit< double > & Epais()const ;
    private:
        std::string mNamePMul;
        double mStepDr;
        cTplValGesInit< double > mRabDr;
        Pt3di mColRayInter;
        cTplValGesInit< double > mEpais;
};
cElXMLTree * ToXMLTree(const cNuagePutInterPMul &);

void  BinaryDumpInFile(ELISE_fp &,const cNuagePutInterPMul &);

void  BinaryUnDumpFromFile(cNuagePutInterPMul &,ELISE_fp &);

std::string  Mangling( cNuagePutInterPMul *);

class cNuagePutGCPCtrl
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNuagePutGCPCtrl & anObj,cElXMLTree * aTree);


        std::string & NameGCPIm();
        const std::string & NameGCPIm()const ;

        std::string & NameGCPTerr();
        const std::string & NameGCPTerr()const ;

        double & ScaleVec();
        const double & ScaleVec()const ;
    private:
        std::string mNameGCPIm;
        std::string mNameGCPTerr;
        double mScaleVec;
};
cElXMLTree * ToXMLTree(const cNuagePutGCPCtrl &);

void  BinaryDumpInFile(ELISE_fp &,const cNuagePutGCPCtrl &);

void  BinaryUnDumpFromFile(cNuagePutGCPCtrl &,ELISE_fp &);

std::string  Mangling( cNuagePutGCPCtrl *);

class cExportNuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportNuage & anObj,cElXMLTree * aTree);


        std::string & NameOut();
        const std::string & NameOut()const ;

        std::string & KeyCalc();
        const std::string & KeyCalc()const ;

        cTplValGesInit< bool > & SymPts();
        const cTplValGesInit< bool > & SymPts()const ;

        cTplValGesInit< cExportNuageByImage > & ExportNuageByImage();
        const cTplValGesInit< cExportNuageByImage > & ExportNuageByImage()const ;

        cTplValGesInit< bool > & PlyModeBin();
        const cTplValGesInit< bool > & PlyModeBin()const ;

        cTplValGesInit< bool > & SavePtsCol();
        const cTplValGesInit< bool > & SavePtsCol()const ;

        std::list< std::string > & NameRefLiaison();
        const std::list< std::string > & NameRefLiaison()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        std::string & KeyFileColImage();
        const std::string & KeyFileColImage()const ;

        cTplValGesInit< int > & NbChan();
        const cTplValGesInit< int > & NbChan()const ;

        cTplValGesInit< Pt3dr > & DirCol();
        const cTplValGesInit< Pt3dr > & DirCol()const ;

        cTplValGesInit< double > & PerCol();
        const cTplValGesInit< double > & PerCol()const ;

        cTplValGesInit< double > & LimBSurH();
        const cTplValGesInit< double > & LimBSurH()const ;

        cTplValGesInit< std::string > & ImExpoRef();
        const cTplValGesInit< std::string > & ImExpoRef()const ;

        Pt3di & ColCadre();
        const Pt3di & ColCadre()const ;

        cTplValGesInit< Pt3di > & ColRay();
        const cTplValGesInit< Pt3di > & ColRay()const ;

        double & Long();
        const double & Long()const ;

        double & StepSeg();
        const double & StepSeg()const ;

        cTplValGesInit< std::string > & KeyCalName();
        const cTplValGesInit< std::string > & KeyCalName()const ;

        cTplValGesInit< double > & StepImage();
        const cTplValGesInit< double > & StepImage()const ;

        cTplValGesInit< std::string > & HomolRay();
        const cTplValGesInit< std::string > & HomolRay()const ;

        cTplValGesInit< Pt3di > & ColRayHomol();
        const cTplValGesInit< Pt3di > & ColRayHomol()const ;

        cTplValGesInit< cNuagePutCam > & NuagePutCam();
        const cTplValGesInit< cNuagePutCam > & NuagePutCam()const ;

        std::string & NamePMul();
        const std::string & NamePMul()const ;

        double & StepDr();
        const double & StepDr()const ;

        cTplValGesInit< double > & RabDr();
        const cTplValGesInit< double > & RabDr()const ;

        Pt3di & ColRayInter();
        const Pt3di & ColRayInter()const ;

        cTplValGesInit< double > & Epais();
        const cTplValGesInit< double > & Epais()const ;

        cTplValGesInit< cNuagePutInterPMul > & NuagePutInterPMul();
        const cTplValGesInit< cNuagePutInterPMul > & NuagePutInterPMul()const ;

        std::string & NameGCPIm();
        const std::string & NameGCPIm()const ;

        std::string & NameGCPTerr();
        const std::string & NameGCPTerr()const ;

        double & ScaleVec();
        const double & ScaleVec()const ;

        cTplValGesInit< cNuagePutGCPCtrl > & NuagePutGCPCtrl();
        const cTplValGesInit< cNuagePutGCPCtrl > & NuagePutGCPCtrl()const ;

        cTplValGesInit< int > & NormByC();
        const cTplValGesInit< int > & NormByC()const ;
    private:
        std::string mNameOut;
        cTplValGesInit< cExportNuageByImage > mExportNuageByImage;
        cTplValGesInit< bool > mPlyModeBin;
        cTplValGesInit< bool > mSavePtsCol;
        std::list< std::string > mNameRefLiaison;
        cTplValGesInit< std::string > mPatternSel;
        cPonderationPackMesure mPond;
        std::string mKeyFileColImage;
        cTplValGesInit< int > mNbChan;
        cTplValGesInit< Pt3dr > mDirCol;
        cTplValGesInit< double > mPerCol;
        cTplValGesInit< double > mLimBSurH;
        cTplValGesInit< std::string > mImExpoRef;
        cTplValGesInit< cNuagePutCam > mNuagePutCam;
        cTplValGesInit< cNuagePutInterPMul > mNuagePutInterPMul;
        cTplValGesInit< cNuagePutGCPCtrl > mNuagePutGCPCtrl;
        cTplValGesInit< int > mNormByC;
};
cElXMLTree * ToXMLTree(const cExportNuage &);

void  BinaryDumpInFile(ELISE_fp &,const cExportNuage &);

void  BinaryUnDumpFromFile(cExportNuage &,ELISE_fp &);

std::string  Mangling( cExportNuage *);

class cChoixImSec
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChoixImSec & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyExistingFile();
        const cTplValGesInit< std::string > & KeyExistingFile()const ;

        cTplValGesInit< std::string > & FileImSel();
        const cTplValGesInit< std::string > & FileImSel()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< int > & CardMaxSub();
        const cTplValGesInit< int > & CardMaxSub()const ;

        cTplValGesInit< double > & PenalNbIm();
        const cTplValGesInit< double > & PenalNbIm()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMinPreSel();
        const cTplValGesInit< double > & TetaMinPreSel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< double > & RatioDistMin();
        const cTplValGesInit< double > & RatioDistMin()const ;

        cTplValGesInit< double > & RatioStereoVertMax();
        const cTplValGesInit< double > & RatioStereoVertMax()const ;

        cTplValGesInit< double > & Teta2Min();
        const cTplValGesInit< double > & Teta2Min()const ;

        cTplValGesInit< double > & Teta2Max();
        const cTplValGesInit< double > & Teta2Max()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbTestPrecis();
        const cTplValGesInit< int > & NbTestPrecis()const ;

        cTplValGesInit< int > & NbCellOccAng();
        const cTplValGesInit< int > & NbCellOccAng()const ;

        cTplValGesInit< int > & NbCaseIm();
        const cTplValGesInit< int > & NbCaseIm()const ;

        cTplValGesInit< std::string > & Masq3D();
        const cTplValGesInit< std::string > & Masq3D()const ;
    private:
        cTplValGesInit< std::string > mKeyExistingFile;
        cTplValGesInit< std::string > mFileImSel;
        cTplValGesInit< std::string > mKeyAssoc;
        cTplValGesInit< std::string > mPatternSel;
        cTplValGesInit< int > mCardMaxSub;
        cTplValGesInit< double > mPenalNbIm;
        int mNbMin;
        std::string mIdBdl;
        cTplValGesInit< int > mNbMinPtsHom;
        cTplValGesInit< double > mTetaMinPreSel;
        cTplValGesInit< double > mTetaOpt;
        cTplValGesInit< double > mTetaMaxPreSel;
        cTplValGesInit< double > mRatioDistMin;
        cTplValGesInit< double > mRatioStereoVertMax;
        cTplValGesInit< double > mTeta2Min;
        cTplValGesInit< double > mTeta2Max;
        cTplValGesInit< int > mNbMaxPresel;
        cTplValGesInit< int > mNbTestPrecis;
        cTplValGesInit< int > mNbCellOccAng;
        cTplValGesInit< int > mNbCaseIm;
        cTplValGesInit< std::string > mMasq3D;
};
cElXMLTree * ToXMLTree(const cChoixImSec &);

void  BinaryDumpInFile(ELISE_fp &,const cChoixImSec &);

void  BinaryUnDumpFromFile(cChoixImSec &,ELISE_fp &);

std::string  Mangling( cChoixImSec *);

class cChoixImMM
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChoixImMM & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyExistingFile();
        const cTplValGesInit< std::string > & KeyExistingFile()const ;

        cTplValGesInit< std::string > & FileImSel();
        const cTplValGesInit< std::string > & FileImSel()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< int > & CardMaxSub();
        const cTplValGesInit< int > & CardMaxSub()const ;

        cTplValGesInit< double > & PenalNbIm();
        const cTplValGesInit< double > & PenalNbIm()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMinPreSel();
        const cTplValGesInit< double > & TetaMinPreSel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< double > & RatioDistMin();
        const cTplValGesInit< double > & RatioDistMin()const ;

        cTplValGesInit< double > & RatioStereoVertMax();
        const cTplValGesInit< double > & RatioStereoVertMax()const ;

        cTplValGesInit< double > & Teta2Min();
        const cTplValGesInit< double > & Teta2Min()const ;

        cTplValGesInit< double > & Teta2Max();
        const cTplValGesInit< double > & Teta2Max()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbTestPrecis();
        const cTplValGesInit< int > & NbTestPrecis()const ;

        cTplValGesInit< int > & NbCellOccAng();
        const cTplValGesInit< int > & NbCellOccAng()const ;

        cTplValGesInit< int > & NbCaseIm();
        const cTplValGesInit< int > & NbCaseIm()const ;

        cTplValGesInit< std::string > & Masq3D();
        const cTplValGesInit< std::string > & Masq3D()const ;

        cChoixImSec & ChoixImSec();
        const cChoixImSec & ChoixImSec()const ;
    private:
        cChoixImSec mChoixImSec;
};
cElXMLTree * ToXMLTree(const cChoixImMM &);

void  BinaryDumpInFile(ELISE_fp &,const cChoixImMM &);

void  BinaryUnDumpFromFile(cChoixImMM &,ELISE_fp &);

std::string  Mangling( cChoixImMM *);

class cExportSensibParamAero
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cExportSensibParamAero & anObj,cElXMLTree * aTree);


        std::string & Dir();
        const std::string & Dir()const ;
    private:
        std::string mDir;
};
cElXMLTree * ToXMLTree(const cExportSensibParamAero &);

void  BinaryDumpInFile(ELISE_fp &,const cExportSensibParamAero &);

void  BinaryUnDumpFromFile(cExportSensibParamAero &,ELISE_fp &);

std::string  Mangling( cExportSensibParamAero *);

class cSectionExport
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionExport & anObj,cElXMLTree * aTree);


        std::list< cExportCalib > & ExportCalib();
        const std::list< cExportCalib > & ExportCalib()const ;

        std::list< cExportPose > & ExportPose();
        const std::list< cExportPose > & ExportPose()const ;

        std::list< cExportAttrPose > & ExportAttrPose();
        const std::list< cExportAttrPose > & ExportAttrPose()const ;

        std::list< cExportRepereLoc > & ExportRepereLoc();
        const std::list< cExportRepereLoc > & ExportRepereLoc()const ;

        std::list< cExportBlockCamera > & ExportBlockCamera();
        const std::list< cExportBlockCamera > & ExportBlockCamera()const ;

        std::list< cExportMesuresFromCarteProf > & ExportMesuresFromCarteProf();
        const std::list< cExportMesuresFromCarteProf > & ExportMesuresFromCarteProf()const ;

        std::list< cExportVisuConfigGrpPose > & ExportVisuConfigGrpPose();
        const std::list< cExportVisuConfigGrpPose > & ExportVisuConfigGrpPose()const ;

        cTplValGesInit< cExportPtsFlottant > & ExportPtsFlottant();
        const cTplValGesInit< cExportPtsFlottant > & ExportPtsFlottant()const ;

        std::list< cExportImResiduLiaison > & ExportImResiduLiaison();
        const std::list< cExportImResiduLiaison > & ExportImResiduLiaison()const ;

        std::list< cExportRedressement > & ExportRedressement();
        const std::list< cExportRedressement > & ExportRedressement()const ;

        std::list< cExportNuage > & ExportNuage();
        const std::list< cExportNuage > & ExportNuage()const ;

        cTplValGesInit< std::string > & KeyExistingFile();
        const cTplValGesInit< std::string > & KeyExistingFile()const ;

        cTplValGesInit< std::string > & FileImSel();
        const cTplValGesInit< std::string > & FileImSel()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< int > & CardMaxSub();
        const cTplValGesInit< int > & CardMaxSub()const ;

        cTplValGesInit< double > & PenalNbIm();
        const cTplValGesInit< double > & PenalNbIm()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMinPreSel();
        const cTplValGesInit< double > & TetaMinPreSel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< double > & RatioDistMin();
        const cTplValGesInit< double > & RatioDistMin()const ;

        cTplValGesInit< double > & RatioStereoVertMax();
        const cTplValGesInit< double > & RatioStereoVertMax()const ;

        cTplValGesInit< double > & Teta2Min();
        const cTplValGesInit< double > & Teta2Min()const ;

        cTplValGesInit< double > & Teta2Max();
        const cTplValGesInit< double > & Teta2Max()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbTestPrecis();
        const cTplValGesInit< int > & NbTestPrecis()const ;

        cTplValGesInit< int > & NbCellOccAng();
        const cTplValGesInit< int > & NbCellOccAng()const ;

        cTplValGesInit< int > & NbCaseIm();
        const cTplValGesInit< int > & NbCaseIm()const ;

        cTplValGesInit< std::string > & Masq3D();
        const cTplValGesInit< std::string > & Masq3D()const ;

        cChoixImSec & ChoixImSec();
        const cChoixImSec & ChoixImSec()const ;

        cTplValGesInit< cChoixImMM > & ChoixImMM();
        const cTplValGesInit< cChoixImMM > & ChoixImMM()const ;

        cTplValGesInit< std::string > & ExportResiduXml();
        const cTplValGesInit< std::string > & ExportResiduXml()const ;

        std::string & Dir();
        const std::string & Dir()const ;

        cTplValGesInit< cExportSensibParamAero > & ExportSensibParamAero();
        const cTplValGesInit< cExportSensibParamAero > & ExportSensibParamAero()const ;
    private:
        std::list< cExportCalib > mExportCalib;
        std::list< cExportPose > mExportPose;
        std::list< cExportAttrPose > mExportAttrPose;
        std::list< cExportRepereLoc > mExportRepereLoc;
        std::list< cExportBlockCamera > mExportBlockCamera;
        std::list< cExportMesuresFromCarteProf > mExportMesuresFromCarteProf;
        std::list< cExportVisuConfigGrpPose > mExportVisuConfigGrpPose;
        cTplValGesInit< cExportPtsFlottant > mExportPtsFlottant;
        std::list< cExportImResiduLiaison > mExportImResiduLiaison;
        std::list< cExportRedressement > mExportRedressement;
        std::list< cExportNuage > mExportNuage;
        cTplValGesInit< cChoixImMM > mChoixImMM;
        cTplValGesInit< std::string > mExportResiduXml;
        cTplValGesInit< cExportSensibParamAero > mExportSensibParamAero;
};
cElXMLTree * ToXMLTree(const cSectionExport &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionExport &);

void  BinaryUnDumpFromFile(cSectionExport &,ELISE_fp &);

std::string  Mangling( cSectionExport *);

class cEtapeCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEtapeCompensation & anObj,cElXMLTree * aTree);


        std::vector< cIterationsCompensation > & IterationsCompensation();
        const std::vector< cIterationsCompensation > & IterationsCompensation()const ;

        std::list< cTraceCpleCam > & TraceCpleCam();
        const std::list< cTraceCpleCam > & TraceCpleCam()const ;

        cTplValGesInit< bool > & GetChar();
        const cTplValGesInit< bool > & GetChar()const ;

        cTplValGesInit< cSectionTracage > & SectionTracage();
        const cTplValGesInit< cSectionTracage > & SectionTracage()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMEtape();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMEtape()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob()const ;

        cTplValGesInit< double > & MultSLMEtape();
        const cTplValGesInit< double > & MultSLMEtape()const ;

        cTplValGesInit< double > & MultSLMGlob();
        const cTplValGesInit< double > & MultSLMGlob()const ;

        double & SigmaPix();
        const double & SigmaPix()const ;

        int & NbGrid();
        const int & NbGrid()const ;

        cTplValGesInit< cContrCamConseq > & ContrCamConseq();
        const cTplValGesInit< cContrCamConseq > & ContrCamConseq()const ;

        std::list< cContrCamGenInc > & ContrCamGenInc();
        const std::list< cContrCamGenInc > & ContrCamGenInc()const ;

        std::list< cObsBlockCamRig > & ObsBlockCamRig();
        const std::list< cObsBlockCamRig > & ObsBlockCamRig()const ;

        std::list< cObsCenterInPlane > & ObsCenterInPlane();
        const std::list< cObsCenterInPlane > & ObsCenterInPlane()const ;

        std::list< cObsAppuis > & ObsAppuis();
        const std::list< cObsAppuis > & ObsAppuis()const ;

        std::list< cObsAppuisFlottant > & ObsAppuisFlottant();
        const std::list< cObsAppuisFlottant > & ObsAppuisFlottant()const ;

        std::list< cObsLiaisons > & ObsLiaisons();
        const std::list< cObsLiaisons > & ObsLiaisons()const ;

        std::list< cObsCentrePDV > & ObsCentrePDV();
        const std::list< cObsCentrePDV > & ObsCentrePDV()const ;

        std::list< cObsRigidGrpImage > & ObsRigidGrpImage();
        const std::list< cObsRigidGrpImage > & ObsRigidGrpImage()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        cTplValGesInit< cTxtRapDetaille > & TxtRapDetaille();
        const cTplValGesInit< cTxtRapDetaille > & TxtRapDetaille()const ;

        std::list< cObsRelGPS > & ObsRelGPS();
        const std::list< cObsRelGPS > & ObsRelGPS()const ;

        cSectionObservations & SectionObservations();
        const cSectionObservations & SectionObservations()const ;

        std::list< cExportCalib > & ExportCalib();
        const std::list< cExportCalib > & ExportCalib()const ;

        std::list< cExportPose > & ExportPose();
        const std::list< cExportPose > & ExportPose()const ;

        std::list< cExportAttrPose > & ExportAttrPose();
        const std::list< cExportAttrPose > & ExportAttrPose()const ;

        std::list< cExportRepereLoc > & ExportRepereLoc();
        const std::list< cExportRepereLoc > & ExportRepereLoc()const ;

        std::list< cExportBlockCamera > & ExportBlockCamera();
        const std::list< cExportBlockCamera > & ExportBlockCamera()const ;

        std::list< cExportMesuresFromCarteProf > & ExportMesuresFromCarteProf();
        const std::list< cExportMesuresFromCarteProf > & ExportMesuresFromCarteProf()const ;

        std::list< cExportVisuConfigGrpPose > & ExportVisuConfigGrpPose();
        const std::list< cExportVisuConfigGrpPose > & ExportVisuConfigGrpPose()const ;

        cTplValGesInit< cExportPtsFlottant > & ExportPtsFlottant();
        const cTplValGesInit< cExportPtsFlottant > & ExportPtsFlottant()const ;

        std::list< cExportImResiduLiaison > & ExportImResiduLiaison();
        const std::list< cExportImResiduLiaison > & ExportImResiduLiaison()const ;

        std::list< cExportRedressement > & ExportRedressement();
        const std::list< cExportRedressement > & ExportRedressement()const ;

        std::list< cExportNuage > & ExportNuage();
        const std::list< cExportNuage > & ExportNuage()const ;

        cTplValGesInit< std::string > & KeyExistingFile();
        const cTplValGesInit< std::string > & KeyExistingFile()const ;

        cTplValGesInit< std::string > & FileImSel();
        const cTplValGesInit< std::string > & FileImSel()const ;

        cTplValGesInit< std::string > & KeyAssoc();
        const cTplValGesInit< std::string > & KeyAssoc()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< int > & CardMaxSub();
        const cTplValGesInit< int > & CardMaxSub()const ;

        cTplValGesInit< double > & PenalNbIm();
        const cTplValGesInit< double > & PenalNbIm()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMinPreSel();
        const cTplValGesInit< double > & TetaMinPreSel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< double > & RatioDistMin();
        const cTplValGesInit< double > & RatioDistMin()const ;

        cTplValGesInit< double > & RatioStereoVertMax();
        const cTplValGesInit< double > & RatioStereoVertMax()const ;

        cTplValGesInit< double > & Teta2Min();
        const cTplValGesInit< double > & Teta2Min()const ;

        cTplValGesInit< double > & Teta2Max();
        const cTplValGesInit< double > & Teta2Max()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbTestPrecis();
        const cTplValGesInit< int > & NbTestPrecis()const ;

        cTplValGesInit< int > & NbCellOccAng();
        const cTplValGesInit< int > & NbCellOccAng()const ;

        cTplValGesInit< int > & NbCaseIm();
        const cTplValGesInit< int > & NbCaseIm()const ;

        cTplValGesInit< std::string > & Masq3D();
        const cTplValGesInit< std::string > & Masq3D()const ;

        cChoixImSec & ChoixImSec();
        const cChoixImSec & ChoixImSec()const ;

        cTplValGesInit< cChoixImMM > & ChoixImMM();
        const cTplValGesInit< cChoixImMM > & ChoixImMM()const ;

        cTplValGesInit< std::string > & ExportResiduXml();
        const cTplValGesInit< std::string > & ExportResiduXml()const ;

        std::string & Dir();
        const std::string & Dir()const ;

        cTplValGesInit< cExportSensibParamAero > & ExportSensibParamAero();
        const cTplValGesInit< cExportSensibParamAero > & ExportSensibParamAero()const ;

        cTplValGesInit< cSectionExport > & SectionExport();
        const cTplValGesInit< cSectionExport > & SectionExport()const ;
    private:
        std::vector< cIterationsCompensation > mIterationsCompensation;
        cTplValGesInit< cSectionTracage > mSectionTracage;
        cTplValGesInit< cSectionLevenbergMarkard > mSLMEtape;
        cTplValGesInit< cSectionLevenbergMarkard > mSLMGlob;
        cTplValGesInit< double > mMultSLMEtape;
        cTplValGesInit< double > mMultSLMGlob;
        cSectionObservations mSectionObservations;
        cTplValGesInit< cSectionExport > mSectionExport;
};
cElXMLTree * ToXMLTree(const cEtapeCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cEtapeCompensation &);

void  BinaryUnDumpFromFile(cEtapeCompensation &,ELISE_fp &);

std::string  Mangling( cEtapeCompensation *);

class cSectionCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionCompensation & anObj,cElXMLTree * aTree);


        std::list< cEtapeCompensation > & EtapeCompensation();
        const std::list< cEtapeCompensation > & EtapeCompensation()const ;
    private:
        std::list< cEtapeCompensation > mEtapeCompensation;
};
cElXMLTree * ToXMLTree(const cSectionCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionCompensation &);

void  BinaryUnDumpFromFile(cSectionCompensation &,ELISE_fp &);

std::string  Mangling( cSectionCompensation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamApero
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamApero & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        cTplValGesInit< std::string > & FileDebug();
        const cTplValGesInit< std::string > & FileDebug()const ;

        cTplValGesInit< bool > & ShowMes();
        const cTplValGesInit< bool > & ShowMes()const ;

        cTplValGesInit< std::string > & LogFile();
        const cTplValGesInit< std::string > & LogFile()const ;

        cTplValGesInit< cShowSection > & ShowSection();
        const cTplValGesInit< cShowSection > & ShowSection()const ;

        cTplValGesInit< bool > & CalledByItself();
        const cTplValGesInit< bool > & CalledByItself()const ;

        cTplValGesInit< cCmdMappeur > & SectionMapApero();
        const cTplValGesInit< cCmdMappeur > & SectionMapApero()const ;

        std::list< cBDD_PtsLiaisons > & BDD_PtsLiaisons();
        const std::list< cBDD_PtsLiaisons > & BDD_PtsLiaisons()const ;

        std::list< cBDD_NewPtMul > & BDD_NewPtMul();
        const std::list< cBDD_NewPtMul > & BDD_NewPtMul()const ;

        std::list< cBDD_PtsAppuis > & BDD_PtsAppuis();
        const std::list< cBDD_PtsAppuis > & BDD_PtsAppuis()const ;

        std::list< cBDD_ObsAppuisFlottant > & BDD_ObsAppuisFlottant();
        const std::list< cBDD_ObsAppuisFlottant > & BDD_ObsAppuisFlottant()const ;

        std::list< cBDD_Orient > & BDD_Orient();
        const std::list< cBDD_Orient > & BDD_Orient()const ;

        std::list< cBDD_Centre > & BDD_Centre();
        const std::list< cBDD_Centre > & BDD_Centre()const ;

        std::list< cFilterProj3D > & FilterProj3D();
        const std::list< cFilterProj3D > & FilterProj3D()const ;

        std::list< cLayerImageToPose > & LayerImageToPose();
        const std::list< cLayerImageToPose > & LayerImageToPose()const ;

        cTplValGesInit< double > & LimInfBSurHPMoy();
        const cTplValGesInit< double > & LimInfBSurHPMoy()const ;

        cTplValGesInit< double > & LimSupBSurHPMoy();
        const cTplValGesInit< double > & LimSupBSurHPMoy()const ;

        std::list< cDeclareObsRelGPS > & DeclareObsRelGPS();
        const std::list< cDeclareObsRelGPS > & DeclareObsRelGPS()const ;

        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & Key();
        const std::string & Key()const ;

        cTplValGesInit< std::string > & KeyJump();
        const cTplValGesInit< std::string > & KeyJump()const ;

        bool & AddFreeRot();
        const bool & AddFreeRot()const ;

        cTplValGesInit< cDeclareObsCalConseq > & DeclareObsCalConseq();
        const cTplValGesInit< cDeclareObsCalConseq > & DeclareObsCalConseq()const ;

        cSectionBDD_Observation & SectionBDD_Observation();
        const cSectionBDD_Observation & SectionBDD_Observation()const ;

        cTplValGesInit< double > & SeuilAutomFE();
        const cTplValGesInit< double > & SeuilAutomFE()const ;

        cTplValGesInit< bool > & AutoriseToujoursUneSeuleLiaison();
        const cTplValGesInit< bool > & AutoriseToujoursUneSeuleLiaison()const ;

        cTplValGesInit< cMapName2Name > & MapMaskHom();
        const cTplValGesInit< cMapName2Name > & MapMaskHom()const ;

        cTplValGesInit< bool > & SauvePMoyenOnlyWithMasq();
        const cTplValGesInit< bool > & SauvePMoyenOnlyWithMasq()const ;

        std::list< cGpsOffset > & GpsOffset();
        const std::list< cGpsOffset > & GpsOffset()const ;

        std::list< cDataObsPlane > & DataObsPlane();
        const std::list< cDataObsPlane > & DataObsPlane()const ;

        std::list< cCalibrationCameraInc > & CalibrationCameraInc();
        const std::list< cCalibrationCameraInc > & CalibrationCameraInc()const ;

        cTplValGesInit< int > & SeuilL1EstimMatrEss();
        const cTplValGesInit< int > & SeuilL1EstimMatrEss()const ;

        std::list< cBlockCamera > & BlockCamera();
        const std::list< cBlockCamera > & BlockCamera()const ;

        cTplValGesInit< cSetOrientationInterne > & GlobOrInterne();
        const cTplValGesInit< cSetOrientationInterne > & GlobOrInterne()const ;

        std::list< cCamGenInc > & CamGenInc();
        const std::list< cCamGenInc > & CamGenInc()const ;

        std::list< cPoseCameraInc > & PoseCameraInc();
        const std::list< cPoseCameraInc > & PoseCameraInc()const ;

        std::list< cGroupeDePose > & GroupeDePose();
        const std::list< cGroupeDePose > & GroupeDePose()const ;

        std::list< cSurfParamInc > & SurfParamInc();
        const std::list< cSurfParamInc > & SurfParamInc()const ;

        std::list< cPointFlottantInc > & PointFlottantInc();
        const std::list< cPointFlottantInc > & PointFlottantInc()const ;

        cSectionInconnues & SectionInconnues();
        const cSectionInconnues & SectionInconnues()const ;

        std::string & IdOrient();
        const std::string & IdOrient()const ;

        double & SigmaC();
        const double & SigmaC()const ;

        double & SigmaR();
        const double & SigmaR()const ;

        cElRegex_Ptr & PatternApply();
        const cElRegex_Ptr & PatternApply()const ;

        cTplValGesInit< cRappelPose > & RappelPose();
        const cTplValGesInit< cRappelPose > & RappelPose()const ;

        cTplValGesInit< int > & NumAttrPdsNewF();
        const cTplValGesInit< int > & NumAttrPdsNewF()const ;

        cTplValGesInit< double > & RatioMaxDistCS();
        const cTplValGesInit< double > & RatioMaxDistCS()const ;

        cTplValGesInit< std::string > & DebugVecElimTieP();
        const cTplValGesInit< std::string > & DebugVecElimTieP()const ;

        cTplValGesInit< int > & DoStatElimBundle();
        const cTplValGesInit< int > & DoStatElimBundle()const ;

        cTplValGesInit< double > & SzByPair();
        const cTplValGesInit< double > & SzByPair()const ;

        cTplValGesInit< double > & SzByPose();
        const cTplValGesInit< double > & SzByPose()const ;

        cTplValGesInit< double > & SzByCam();
        const cTplValGesInit< double > & SzByCam()const ;

        cTplValGesInit< double > & NbMesByCase();
        const cTplValGesInit< double > & NbMesByCase()const ;

        std::string & AeroExport();
        const std::string & AeroExport()const ;

        cTplValGesInit< bool > & GeneratePly();
        const cTplValGesInit< bool > & GeneratePly()const ;

        cTplValGesInit< int > & SzOrtho();
        const cTplValGesInit< int > & SzOrtho()const ;

        cTplValGesInit< cUseExportImageResidu > & UseExportImageResidu();
        const cTplValGesInit< cUseExportImageResidu > & UseExportImageResidu()const ;

        cTplValGesInit< bool > & UseRegulDist();
        const cTplValGesInit< bool > & UseRegulDist()const ;

        cTplValGesInit< bool > & GBCamSupresStenCam();
        const cTplValGesInit< bool > & GBCamSupresStenCam()const ;

        cTplValGesInit< bool > & StenCamSupresGBCam();
        const cTplValGesInit< bool > & StenCamSupresGBCam()const ;

        cTplValGesInit< bool > & IsAperiCloud();
        const cTplValGesInit< bool > & IsAperiCloud()const ;

        cTplValGesInit< bool > & IsChoixImSec();
        const cTplValGesInit< bool > & IsChoixImSec()const ;

        cTplValGesInit< std::string > & FileSauvParam();
        const cTplValGesInit< std::string > & FileSauvParam()const ;

        cTplValGesInit< bool > & GenereErreurOnContraineCam();
        const cTplValGesInit< bool > & GenereErreurOnContraineCam()const ;

        cTplValGesInit< double > & ProfSceneChantier();
        const cTplValGesInit< double > & ProfSceneChantier()const ;

        cTplValGesInit< std::string > & DirectoryChantier();
        const cTplValGesInit< std::string > & DirectoryChantier()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        cTplValGesInit< std::string > & NameParamEtal();
        const cTplValGesInit< std::string > & NameParamEtal()const ;

        cTplValGesInit< std::string > & PatternTracePose();
        const cTplValGesInit< std::string > & PatternTracePose()const ;

        cTplValGesInit< bool > & TraceGimbalLock();
        const cTplValGesInit< bool > & TraceGimbalLock()const ;

        cTplValGesInit< double > & MaxDistErrorPtsTerr();
        const cTplValGesInit< double > & MaxDistErrorPtsTerr()const ;

        cTplValGesInit< double > & MaxDistWarnPtsTerr();
        const cTplValGesInit< double > & MaxDistWarnPtsTerr()const ;

        cTplValGesInit< cShowPbLiaison > & DefPbLiaison();
        const cTplValGesInit< cShowPbLiaison > & DefPbLiaison()const ;

        cTplValGesInit< bool > & DoCompensation();
        const cTplValGesInit< bool > & DoCompensation()const ;

        double & DeltaMax();
        const double & DeltaMax()const ;

        cTplValGesInit< cTimeLinkage > & TimeLinkage();
        const cTplValGesInit< cTimeLinkage > & TimeLinkage()const ;

        cTplValGesInit< bool > & DebugPbCondFaisceau();
        const cTplValGesInit< bool > & DebugPbCondFaisceau()const ;

        cTplValGesInit< std::string > & SauvAutom();
        const cTplValGesInit< std::string > & SauvAutom()const ;

        cTplValGesInit< bool > & SauvAutomBasic();
        const cTplValGesInit< bool > & SauvAutomBasic()const ;

        cTplValGesInit< double > & ThresholdWarnPointsBehind();
        const cTplValGesInit< double > & ThresholdWarnPointsBehind()const ;

        cTplValGesInit< bool > & ExportMatrixMarket();
        const cTplValGesInit< bool > & ExportMatrixMarket()const ;

        cTplValGesInit< double > & ExtensionIntervZ();
        const cTplValGesInit< double > & ExtensionIntervZ()const ;

        cSectionChantier & SectionChantier();
        const cSectionChantier & SectionChantier()const ;

        cTplValGesInit< bool > & AllMatSym();
        const cTplValGesInit< bool > & AllMatSym()const ;

        eModeSolveurEq & ModeResolution();
        const eModeSolveurEq & ModeResolution()const ;

        cTplValGesInit< eControleDescDic > & ModeControleDescDic();
        const cTplValGesInit< eControleDescDic > & ModeControleDescDic()const ;

        cTplValGesInit< int > & SeuilBas_CDD();
        const cTplValGesInit< int > & SeuilBas_CDD()const ;

        cTplValGesInit< int > & SeuilHaut_CDD();
        const cTplValGesInit< int > & SeuilHaut_CDD()const ;

        cTplValGesInit< bool > & InhibeAMD();
        const cTplValGesInit< bool > & InhibeAMD()const ;

        cTplValGesInit< bool > & AMDSpecInterne();
        const cTplValGesInit< bool > & AMDSpecInterne()const ;

        cTplValGesInit< bool > & ShowCholesky();
        const cTplValGesInit< bool > & ShowCholesky()const ;

        cTplValGesInit< bool > & TestPermutVar();
        const cTplValGesInit< bool > & TestPermutVar()const ;

        cTplValGesInit< bool > & ShowPermutVar();
        const cTplValGesInit< bool > & ShowPermutVar()const ;

        cTplValGesInit< bool > & PermutIndex();
        const cTplValGesInit< bool > & PermutIndex()const ;

        cTplValGesInit< bool > & NormaliseEqSc();
        const cTplValGesInit< bool > & NormaliseEqSc()const ;

        cTplValGesInit< bool > & NormaliseEqTr();
        const cTplValGesInit< bool > & NormaliseEqTr()const ;

        cTplValGesInit< double > & LimBsHProj();
        const cTplValGesInit< double > & LimBsHProj()const ;

        cTplValGesInit< double > & LimBsHRefut();
        const cTplValGesInit< double > & LimBsHRefut()const ;

        cTplValGesInit< double > & LimModeGL();
        const cTplValGesInit< double > & LimModeGL()const ;

        cTplValGesInit< bool > & GridOptimKnownDist();
        const cTplValGesInit< bool > & GridOptimKnownDist()const ;

        cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob();
        const cTplValGesInit< cSectionLevenbergMarkard > & SLMGlob()const ;

        cTplValGesInit< double > & MultSLMGlob();
        const cTplValGesInit< double > & MultSLMGlob()const ;

        cTplValGesInit< cElRegex_Ptr > & Im2Aff();
        const cTplValGesInit< cElRegex_Ptr > & Im2Aff()const ;

        cTplValGesInit< cXmlPondRegDist > & RegDistGlob();
        const cTplValGesInit< cXmlPondRegDist > & RegDistGlob()const ;

        cSectionSolveur & SectionSolveur();
        const cSectionSolveur & SectionSolveur()const ;

        std::list< cEtapeCompensation > & EtapeCompensation();
        const std::list< cEtapeCompensation > & EtapeCompensation()const ;

        cSectionCompensation & SectionCompensation();
        const cSectionCompensation & SectionCompensation()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cTplValGesInit< std::string > mFileDebug;
        cTplValGesInit< cShowSection > mShowSection;
        cTplValGesInit< bool > mCalledByItself;
        cTplValGesInit< cCmdMappeur > mSectionMapApero;
        cSectionBDD_Observation mSectionBDD_Observation;
        cSectionInconnues mSectionInconnues;
        cSectionChantier mSectionChantier;
        cSectionSolveur mSectionSolveur;
        cSectionCompensation mSectionCompensation;
};
cElXMLTree * ToXMLTree(const cParamApero &);

void  BinaryDumpInFile(ELISE_fp &,const cParamApero &);

void  BinaryUnDumpFromFile(cParamApero &,ELISE_fp &);

std::string  Mangling( cParamApero *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlSauvExportAperoOneIm
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlSauvExportAperoOneIm & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        double & Residual();
        const double & Residual()const ;

        double & PercOk();
        const double & PercOk()const ;

        int & NbPts();
        const int & NbPts()const ;

        int & NbPtsMul();
        const int & NbPtsMul()const ;
    private:
        std::string mName;
        double mResidual;
        double mPercOk;
        int mNbPts;
        int mNbPtsMul;
};
cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneIm &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlSauvExportAperoOneIm &);

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneIm &,ELISE_fp &);

std::string  Mangling( cXmlSauvExportAperoOneIm *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlSauvExportAperoOneAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlSauvExportAperoOneAppuis & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< Pt3dr > & EcartFaiscTerrain();
        const cTplValGesInit< Pt3dr > & EcartFaiscTerrain()const ;

        cTplValGesInit< double > & DistFaiscTerrain();
        const cTplValGesInit< double > & DistFaiscTerrain()const ;

        cTplValGesInit< double > & EcartImMoy();
        const cTplValGesInit< double > & EcartImMoy()const ;

        cTplValGesInit< double > & EcartImMax();
        const cTplValGesInit< double > & EcartImMax()const ;

        cTplValGesInit< std::string > & NameImMax();
        const cTplValGesInit< std::string > & NameImMax()const ;
    private:
        std::string mName;
        cTplValGesInit< Pt3dr > mEcartFaiscTerrain;
        cTplValGesInit< double > mDistFaiscTerrain;
        cTplValGesInit< double > mEcartImMoy;
        cTplValGesInit< double > mEcartImMax;
        cTplValGesInit< std::string > mNameImMax;
};
cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlSauvExportAperoOneAppuis &);

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneAppuis &,ELISE_fp &);

std::string  Mangling( cXmlSauvExportAperoOneAppuis *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlSauvExportAperoOneMult
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlSauvExportAperoOneMult & anObj,cElXMLTree * aTree);


        int & Multiplicity();
        const int & Multiplicity()const ;

        double & Residual();
        const double & Residual()const ;

        int & NbPts();
        const int & NbPts()const ;

        double & PercOk();
        const double & PercOk()const ;
    private:
        int mMultiplicity;
        double mResidual;
        int mNbPts;
        double mPercOk;
};
cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneMult &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlSauvExportAperoOneMult &);

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneMult &,ELISE_fp &);

std::string  Mangling( cXmlSauvExportAperoOneMult *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlSauvExportAperoOneIter
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlSauvExportAperoOneIter & anObj,cElXMLTree * aTree);


        std::list< cXmlSauvExportAperoOneAppuis > & OneAppui();
        const std::list< cXmlSauvExportAperoOneAppuis > & OneAppui()const ;

        std::list< cXmlSauvExportAperoOneIm > & OneIm();
        const std::list< cXmlSauvExportAperoOneIm > & OneIm()const ;

        std::list< cXmlSauvExportAperoOneMult > & OneMult();
        const std::list< cXmlSauvExportAperoOneMult > & OneMult()const ;

        double & AverageResidual();
        const double & AverageResidual()const ;

        int & NumIter();
        const int & NumIter()const ;

        int & NumEtape();
        const int & NumEtape()const ;

        cTplValGesInit< double > & EvolMax();
        const cTplValGesInit< double > & EvolMax()const ;

        cTplValGesInit< double > & EvolMoy();
        const cTplValGesInit< double > & EvolMoy()const ;

        cTplValGesInit< std::string > & ImWorstRes();
        const cTplValGesInit< std::string > & ImWorstRes()const ;

        cTplValGesInit< double > & WorstRes();
        const cTplValGesInit< double > & WorstRes()const ;
    private:
        std::list< cXmlSauvExportAperoOneAppuis > mOneAppui;
        std::list< cXmlSauvExportAperoOneIm > mOneIm;
        std::list< cXmlSauvExportAperoOneMult > mOneMult;
        double mAverageResidual;
        int mNumIter;
        int mNumEtape;
        cTplValGesInit< double > mEvolMax;
        cTplValGesInit< double > mEvolMoy;
        cTplValGesInit< std::string > mImWorstRes;
        cTplValGesInit< double > mWorstRes;
};
cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneIter &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlSauvExportAperoOneIter &);

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneIter &,ELISE_fp &);

std::string  Mangling( cXmlSauvExportAperoOneIter *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlSauvExportAperoGlob
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlSauvExportAperoGlob & anObj,cElXMLTree * aTree);


        std::list< cXmlSauvExportAperoOneIter > & Iters();
        const std::list< cXmlSauvExportAperoOneIter > & Iters()const ;
    private:
        std::list< cXmlSauvExportAperoOneIter > mIters;
};
cElXMLTree * ToXMLTree(const cXmlSauvExportAperoGlob &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlSauvExportAperoGlob &);

void  BinaryUnDumpFromFile(cXmlSauvExportAperoGlob &,ELISE_fp &);

std::string  Mangling( cXmlSauvExportAperoGlob *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlOneResultRTA
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlOneResultRTA & anObj,cElXMLTree * aTree);


        double & Mult();
        const double & Mult()const ;

        double & MoyErr();
        const double & MoyErr()const ;

        std::list< cXmlSauvExportAperoOneAppuis > & OneAppui();
        const std::list< cXmlSauvExportAperoOneAppuis > & OneAppui()const ;
    private:
        double mMult;
        double mMoyErr;
        std::list< cXmlSauvExportAperoOneAppuis > mOneAppui;
};
cElXMLTree * ToXMLTree(const cXmlOneResultRTA &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlOneResultRTA &);

void  BinaryUnDumpFromFile(cXmlOneResultRTA &,ELISE_fp &);

std::string  Mangling( cXmlOneResultRTA *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlResultRTA
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlResultRTA & anObj,cElXMLTree * aTree);


        double & BestMult();
        const double & BestMult()const ;

        double & BestMoyErr();
        const double & BestMoyErr()const ;

        std::list< cXmlOneResultRTA > & RTA();
        const std::list< cXmlOneResultRTA > & RTA()const ;
    private:
        double mBestMult;
        double mBestMoyErr;
        std::list< cXmlOneResultRTA > mRTA;
};
cElXMLTree * ToXMLTree(const cXmlResultRTA &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlResultRTA &);

void  BinaryUnDumpFromFile(cXmlResultRTA &,ELISE_fp &);

std::string  Mangling( cXmlResultRTA *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSensibDateOneInc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSensibDateOneInc & anObj,cElXMLTree * aTree);


        std::string & NameBloc();
        const std::string & NameBloc()const ;

        std::string & NameInc();
        const std::string & NameInc()const ;

        double & SensibParamDir();
        const double & SensibParamDir()const ;

        double & SensibParamInv();
        const double & SensibParamInv()const ;

        double & SensibParamVar();
        const double & SensibParamVar()const ;
    private:
        std::string mNameBloc;
        std::string mNameInc;
        double mSensibParamDir;
        double mSensibParamInv;
        double mSensibParamVar;
};
cElXMLTree * ToXMLTree(const cSensibDateOneInc &);

void  BinaryDumpInFile(ELISE_fp &,const cSensibDateOneInc &);

void  BinaryUnDumpFromFile(cSensibDateOneInc &,ELISE_fp &);

std::string  Mangling( cSensibDateOneInc *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlNameSensibs
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlNameSensibs & anObj,cElXMLTree * aTree);


        std::vector< cSensibDateOneInc > & SensibDateOneInc();
        const std::vector< cSensibDateOneInc > & SensibDateOneInc()const ;
    private:
        std::vector< cSensibDateOneInc > mSensibDateOneInc;
};
cElXMLTree * ToXMLTree(const cXmlNameSensibs &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlNameSensibs &);

void  BinaryUnDumpFromFile(cXmlNameSensibs &,ELISE_fp &);

std::string  Mangling( cXmlNameSensibs *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlOneContourCamera
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlOneContourCamera & anObj,cElXMLTree * aTree);


        std::vector< Pt2dr > & Pt();
        const std::vector< Pt2dr > & Pt()const ;
    private:
        std::vector< Pt2dr > mPt;
};
cElXMLTree * ToXMLTree(const cXmlOneContourCamera &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlOneContourCamera &);

void  BinaryUnDumpFromFile(cXmlOneContourCamera &,ELISE_fp &);

std::string  Mangling( cXmlOneContourCamera *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotApero
