#include "general/all.h"
#include "private/all.h"
#ifndef Define_NotApero
#define Define_NotApero
#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;
namespace NS_ParamApero{
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

typedef enum
{
  eCalibAutomRadial,
  eCalibAutomPhgrStd,
  eCalibAutomFishEyeLineaire,
  eCalibAutomFishEyeEquiSolid,
  eCalibAutomRadialBasic,
  eCalibAutomPhgrStdBasic,
  eCalibAutomNone
} eTypeCalibAutom;
void xml_init(eTypeCalibAutom & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeCalibAutom & aVal);

eTypeCalibAutom  Str2eTypeCalibAutom(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeCalibAutom & anObj);

typedef enum
{
  ePoseLibre,
  ePoseFigee,
  ePoseBaseNormee,
  ePoseVraieBaseNormee,
  eCentreFige
} eTypeContraintePoseCamera;
void xml_init(eTypeContraintePoseCamera & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeContraintePoseCamera & aVal);

eTypeContraintePoseCamera  Str2eTypeContraintePoseCamera(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeContraintePoseCamera & anObj);

typedef enum
{
  eVerifDZ,
  eVerifResPerIm
} eTypeVerif;
void xml_init(eTypeVerif & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeVerif & aVal);

eTypeVerif  Str2eTypeVerif(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeVerif & anObj);

typedef enum
{
  eMST_PondCard
} eTypePondMST_MEP;
void xml_init(eTypePondMST_MEP & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypePondMST_MEP & aVal);

eTypePondMST_MEP  Str2eTypePondMST_MEP(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePondMST_MEP & anObj);

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

typedef enum
{
  eMPL_DbleCoplanIm,
  eMPL_PtTerrainInc
} eModePointLiaison;
void xml_init(eModePointLiaison & aVal,cElXMLTree * aTree);
std::string  eToString(const eModePointLiaison & aVal);

eModePointLiaison  Str2eModePointLiaison(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModePointLiaison & anObj);

class cPowPointLiaisons
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cOptimizationPowel
{
    public:
        friend void xml_init(cOptimizationPowel & anObj,cElXMLTree * aTree);


        std::list< cPowPointLiaisons > & PowPointLiaisons();
        const std::list< cPowPointLiaisons > & PowPointLiaisons()const ;
    private:
        std::list< cPowPointLiaisons > mPowPointLiaisons;
};
cElXMLTree * ToXMLTree(const cOptimizationPowel &);

/******************************************************/
/******************************************************/
/******************************************************/
class cShowPbLiaison
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cPonderationPackMesure
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cParamEstimPlan
{
    public:
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
    private:
        cTplValGesInit< std::string > mAttrSup;
        cTplValGesInit< std::string > mKeyCalculMasq;
        std::string mIdBdl;
        cPonderationPackMesure mPond;
        cTplValGesInit< double > mLimBSurH;
};
cElXMLTree * ToXMLTree(const cParamEstimPlan &);

/******************************************************/
/******************************************************/
/******************************************************/
class cAperoPointeStereo
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cAperoPointeMono
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cApero2PointeFromFile
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cParamForceRappel
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cRappelOnAngles
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cRappelOnCentres
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionLevenbergMarkard
{
    public:
        friend void xml_init(cSectionLevenbergMarkard & anObj,cElXMLTree * aTree);


        std::list< cRappelOnAngles > & RappelOnAngles();
        const std::list< cRappelOnAngles > & RappelOnAngles()const ;

        std::list< cRappelOnCentres > & RappelOnCentres();
        const std::list< cRappelOnCentres > & RappelOnCentres()const ;
    private:
        std::list< cRappelOnAngles > mRappelOnAngles;
        std::list< cRappelOnCentres > mRappelOnCentres;
};
cElXMLTree * ToXMLTree(const cSectionLevenbergMarkard &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetOrientationInterne
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cExportAsNewGrid
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cShowSection
{
    public:
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

/******************************************************/
/******************************************************/
/******************************************************/
class cSzImForInvY
{
    public:
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

class cSplitLayer
{
    public:
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

class cBDD_PtsLiaisons
{
    public:
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

class cBddApp_AutoNum
{
    public:
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

class cBDD_PtsAppuis
{
    public:
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

class cBDD_ObsAppuisFlottant
{
    public:
        friend void xml_init(cBDD_ObsAppuisFlottant & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt2dr > & OffsetIm();
        const cTplValGesInit< Pt2dr > & OffsetIm()const ;

        std::string & Id();
        const std::string & Id()const ;

        std::string & KeySetOrPat();
        const std::string & KeySetOrPat()const ;

        cTplValGesInit< std::string > & NameAppuiSelector();
        const cTplValGesInit< std::string > & NameAppuiSelector()const ;

        cTplValGesInit< bool > & AcceptNoGround();
        const cTplValGesInit< bool > & AcceptNoGround()const ;
    private:
        cTplValGesInit< Pt2dr > mOffsetIm;
        std::string mId;
        std::string mKeySetOrPat;
        cTplValGesInit< std::string > mNameAppuiSelector;
        cTplValGesInit< bool > mAcceptNoGround;
};
cElXMLTree * ToXMLTree(const cBDD_ObsAppuisFlottant &);

class cBDD_Orient
{
    public:
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

class cCalcOffsetCentre
{
    public:
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

class cBDD_Centre
{
    public:
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

class cFilterProj3D
{
    public:
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

class cLayerTerrain
{
    public:
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

class cLayerImageToPose
{
    public:
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

class cSectionBDD_Observation
{
    public:
        friend void xml_init(cSectionBDD_Observation & anObj,cElXMLTree * aTree);


        std::list< cBDD_PtsLiaisons > & BDD_PtsLiaisons();
        const std::list< cBDD_PtsLiaisons > & BDD_PtsLiaisons()const ;

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
    private:
        std::list< cBDD_PtsLiaisons > mBDD_PtsLiaisons;
        std::list< cBDD_PtsAppuis > mBDD_PtsAppuis;
        std::list< cBDD_ObsAppuisFlottant > mBDD_ObsAppuisFlottant;
        std::list< cBDD_Orient > mBDD_Orient;
        std::list< cBDD_Centre > mBDD_Centre;
        std::list< cFilterProj3D > mFilterProj3D;
        std::list< cLayerImageToPose > mLayerImageToPose;
        cTplValGesInit< double > mLimInfBSurHPMoy;
        cTplValGesInit< double > mLimSupBSurHPMoy;
};
cElXMLTree * ToXMLTree(const cSectionBDD_Observation &);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalibAutomNoDist
{
    public:
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

class cCalValueInit
{
    public:
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

class cAddParamAFocal
{
    public:
        friend void xml_init(cAddParamAFocal & anObj,cElXMLTree * aTree);


        std::vector< double > & Coeffs();
        const std::vector< double > & Coeffs()const ;
    private:
        std::vector< double > mCoeffs;
};
cElXMLTree * ToXMLTree(const cAddParamAFocal &);

class cCalibPerPose
{
    public:
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

class cCalibrationCameraInc
{
    public:
        friend void xml_init(cCalibrationCameraInc & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< eConventionsOrientation > & ConvCal();
        const cTplValGesInit< eConventionsOrientation > & ConvCal()const ;

        cTplValGesInit< std::string > & Directory();
        const cTplValGesInit< std::string > & Directory()const ;

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

class cMEP_SPEC_MST
{
    public:
        friend void xml_init(cMEP_SPEC_MST & anObj,cElXMLTree * aTree);


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
        cTplValGesInit< bool > mShow;
        cTplValGesInit< int > mMinNbPtsInit;
        cTplValGesInit< double > mExpDist;
        cTplValGesInit< double > mExpNb;
        cTplValGesInit< bool > mMontageOnInit;
        cTplValGesInit< int > mNbInitMinBeforeUnconnect;
};
cElXMLTree * ToXMLTree(const cMEP_SPEC_MST &);

class cApplyOAI
{
    public:
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

class cOptimizeAfterInit
{
    public:
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

class cPosFromBDAppuis
{
    public:
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

class cLiaisonsInit
{
    public:
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

class cPoseFromLiaisons
{
    public:
        friend void xml_init(cPoseFromLiaisons & anObj,cElXMLTree * aTree);


        std::vector< cLiaisonsInit > & LiaisonsInit();
        const std::vector< cLiaisonsInit > & LiaisonsInit()const ;
    private:
        std::vector< cLiaisonsInit > mLiaisonsInit;
};
cElXMLTree * ToXMLTree(const cPoseFromLiaisons &);

class cMesurePIFRP
{
    public:
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

class cInitPIFRP
{
    public:
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

class cPoseInitFromReperePlan
{
    public:
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

class cPosValueInit
{
    public:
        friend void xml_init(cPosValueInit & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PosId();
        const cTplValGesInit< std::string > & PosId()const ;

        cTplValGesInit< std::string > & PosFromBDOrient();
        const cTplValGesInit< std::string > & PosFromBDOrient()const ;

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
        cTplValGesInit< cPosFromBDAppuis > mPosFromBDAppuis;
        cTplValGesInit< cPoseFromLiaisons > mPoseFromLiaisons;
        cTplValGesInit< cPoseInitFromReperePlan > mPoseInitFromReperePlan;
};
cElXMLTree * ToXMLTree(const cPosValueInit &);

class cPoseCameraInc
{
    public:
        friend void xml_init(cPoseCameraInc & anObj,cElXMLTree * aTree);


        cTplValGesInit< cSetOrientationInterne > & OrInterne();
        const cTplValGesInit< cSetOrientationInterne > & OrInterne()const ;

        cTplValGesInit< std::string > & IdBDCentre();
        const cTplValGesInit< std::string > & IdBDCentre()const ;

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

        std::string & CalcNameCalib();
        const std::string & CalcNameCalib()const ;

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
        std::string mCalcNameCalib;
        cTplValGesInit< std::string > mPosesDeRattachement;
        cTplValGesInit< bool > mNoErroOnRat;
        cTplValGesInit< bool > mByPattern;
        cTplValGesInit< std::string > mKeyFilterExistingFile;
        cTplValGesInit< bool > mByKey;
        cTplValGesInit< bool > mByFile;
        cPosValueInit mPosValueInit;
};
cElXMLTree * ToXMLTree(const cPoseCameraInc &);

class cGroupeDePose
{
    public:
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

class cLiaisonsApplyContrainte
{
    public:
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

class cInitSurf
{
    public:
        friend void xml_init(cInitSurf & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & ZonePlane();
        const cTplValGesInit< std::string > & ZonePlane()const ;
    private:
        cTplValGesInit< std::string > mZonePlane;
};
cElXMLTree * ToXMLTree(const cInitSurf &);

class cSurfParamInc
{
    public:
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

class cPointFlottantInc
{
    public:
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

class cSectionInconnues
{
    public:
        friend void xml_init(cSectionInconnues & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & SeuilAutomFE();
        const cTplValGesInit< double > & SeuilAutomFE()const ;

        cTplValGesInit< bool > & AutoriseToujoursUneSeuleLiaison();
        const cTplValGesInit< bool > & AutoriseToujoursUneSeuleLiaison()const ;

        cTplValGesInit< cMapName2Name > & MapMaskHom();
        const cTplValGesInit< cMapName2Name > & MapMaskHom()const ;

        cTplValGesInit< bool > & SauvePMoyenOnlyWithMasq();
        const cTplValGesInit< bool > & SauvePMoyenOnlyWithMasq()const ;

        std::list< cCalibrationCameraInc > & CalibrationCameraInc();
        const std::list< cCalibrationCameraInc > & CalibrationCameraInc()const ;

        cTplValGesInit< int > & SeuilL1EstimMatrEss();
        const cTplValGesInit< int > & SeuilL1EstimMatrEss()const ;

        cTplValGesInit< cSetOrientationInterne > & GlobOrInterne();
        const cTplValGesInit< cSetOrientationInterne > & GlobOrInterne()const ;

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
        std::list< cCalibrationCameraInc > mCalibrationCameraInc;
        cTplValGesInit< int > mSeuilL1EstimMatrEss;
        cTplValGesInit< cSetOrientationInterne > mGlobOrInterne;
        std::list< cPoseCameraInc > mPoseCameraInc;
        std::list< cGroupeDePose > mGroupeDePose;
        std::list< cSurfParamInc > mSurfParamInc;
        std::list< cPointFlottantInc > mPointFlottantInc;
};
cElXMLTree * ToXMLTree(const cSectionInconnues &);

/******************************************************/
/******************************************************/
/******************************************************/
class cTimeLinkage
{
    public:
        friend void xml_init(cTimeLinkage & anObj,cElXMLTree * aTree);


        double & DeltaMax();
        const double & DeltaMax()const ;
    private:
        double mDeltaMax;
};
cElXMLTree * ToXMLTree(const cTimeLinkage &);

class cSectionChantier
{
    public:
        friend void xml_init(cSectionChantier & anObj,cElXMLTree * aTree);


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
    private:
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
};
cElXMLTree * ToXMLTree(const cSectionChantier &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionSolveur
{
    public:
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
};
cElXMLTree * ToXMLTree(const cSectionSolveur &);

/******************************************************/
/******************************************************/
/******************************************************/
class cPose2Init
{
    public:
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

class cSetRayMaxUtileCalib
{
    public:
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

class cBascOnCentre
{
    public:
        friend void xml_init(cBascOnCentre & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;
    private:
        cTplValGesInit< std::string > mPoseCentrale;
};
cElXMLTree * ToXMLTree(const cBascOnCentre &);

class cBascOnAppuis
{
    public:
        friend void xml_init(cBascOnAppuis & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;
    private:
        std::string mNameRef;
};
cElXMLTree * ToXMLTree(const cBascOnAppuis &);

class cBasculeOnPoints
{
    public:
        friend void xml_init(cBasculeOnPoints & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;
    private:
        cTplValGesInit< cBascOnCentre > mBascOnCentre;
        cTplValGesInit< cBascOnAppuis > mBascOnAppuis;
        cTplValGesInit< bool > mModeL2;
};
cElXMLTree * ToXMLTree(const cBasculeOnPoints &);

class cOrientInPlane
{
    public:
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

class cBasculeLiaisonOnPlan
{
    public:
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

class cModeBascule
{
    public:
        friend void xml_init(cModeBascule & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

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

class cBasculeOrientation
{
    public:
        friend void xml_init(cBasculeOrientation & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & AfterCompens();
        const cTplValGesInit< bool > & AfterCompens()const ;

        cTplValGesInit< std::string > & PatternNameApply();
        const cTplValGesInit< std::string > & PatternNameApply()const ;

        cTplValGesInit< std::string > & PatternNameEstim();
        const cTplValGesInit< std::string > & PatternNameEstim()const ;

        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

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
        cModeBascule mModeBascule;
};
cElXMLTree * ToXMLTree(const cBasculeOrientation &);

class cStereoFE
{
    public:
        friend void xml_init(cStereoFE & anObj,cElXMLTree * aTree);


        std::vector< cAperoPointeStereo > & HomFE();
        const std::vector< cAperoPointeStereo > & HomFE()const ;
    private:
        std::vector< cAperoPointeStereo > mHomFE;
};
cElXMLTree * ToXMLTree(const cStereoFE &);

class cModeFE
{
    public:
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

class cFixeEchelle
{
    public:
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

class cHorFOP
{
    public:
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

class cModeFOP
{
    public:
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

class cFixeOrientPlane
{
    public:
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

class cBlocBascule
{
    public:
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

class cMesureErreurTournante
{
    public:
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

class cContraintesCamerasInc
{
    public:
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

class cContraintesPoses
{
    public:
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

class cSectionContraintes
{
    public:
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

class cVisuPtsMult
{
    public:
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

class cVerifAero
{
    public:
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

class cGPtsTer_By_ImProf
{
    public:
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

class cGeneratePointsTerrains
{
    public:
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

class cGenerateLiaisons
{
    public:
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

class cExportSimulation
{
    public:
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

class cTestInteractif
{
    public:
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

class cIterationsCompensation
{
    public:
        friend void xml_init(cIterationsCompensation & anObj,cElXMLTree * aTree);


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

        cTplValGesInit< std::string > & PoseCentrale();
        const cTplValGesInit< std::string > & PoseCentrale()const ;

        cTplValGesInit< cBascOnCentre > & BascOnCentre();
        const cTplValGesInit< cBascOnCentre > & BascOnCentre()const ;

        std::string & NameRef();
        const std::string & NameRef()const ;

        cTplValGesInit< cBascOnAppuis > & BascOnAppuis();
        const cTplValGesInit< cBascOnAppuis > & BascOnAppuis()const ;

        cTplValGesInit< bool > & ModeL2();
        const cTplValGesInit< bool > & ModeL2()const ;

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
        cTplValGesInit< cMesureErreurTournante > mMesureErreurTournante;
        cTplValGesInit< cSectionContraintes > mSectionContraintes;
        std::list< std::string > mMessages;
        std::list< cVisuPtsMult > mVisuPtsMult;
        std::list< cVerifAero > mVerifAero;
        std::list< cExportSimulation > mExportSimulation;
        cTplValGesInit< cTestInteractif > mTestInteractif;
};
cElXMLTree * ToXMLTree(const cIterationsCompensation &);

class cTraceCpleHom
{
    public:
        friend void xml_init(cTraceCpleHom & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;
    private:
        std::string mId;
};
cElXMLTree * ToXMLTree(const cTraceCpleHom &);

class cTraceCpleCam
{
    public:
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

class cSectionTracage
{
    public:
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

class cROA_FichierImg
{
    public:
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

class cRapportObsAppui
{
    public:
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

class cObsAppuis
{
    public:
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

class cObsAppuisFlottant
{
    public:
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
    private:
        std::string mNameRef;
        cPonderationPackMesure mPondIm;
        std::list< cElRegex_Ptr > mPtsShowDet;
        cTplValGesInit< bool > mDetShow3D;
        cTplValGesInit< double > mNivAlerteDetail;
        cTplValGesInit< bool > mShowMax;
        cTplValGesInit< bool > mShowSom;
};
cElXMLTree * ToXMLTree(const cObsAppuisFlottant &);

class cRappelOnZ
{
    public:
        friend void xml_init(cRappelOnZ & anObj,cElXMLTree * aTree);


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
        double mZ;
        double mIncC;
        cTplValGesInit< double > mIncE;
        cTplValGesInit< double > mSeuilR;
        cTplValGesInit< std::string > mLayerMasq;
};
cElXMLTree * ToXMLTree(const cRappelOnZ &);

class cObsLiaisons
{
    public:
        friend void xml_init(cObsLiaisons & anObj,cElXMLTree * aTree);


        std::string & NameRef();
        const std::string & NameRef()const ;

        cPonderationPackMesure & Pond();
        const cPonderationPackMesure & Pond()const ;

        cTplValGesInit< cPonderationPackMesure > & PondSurf();
        const cTplValGesInit< cPonderationPackMesure > & PondSurf()const ;

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

class cObsCentrePDV
{
    public:
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

class cORGI_CentreCommun
{
    public:
        friend void xml_init(cORGI_CentreCommun & anObj,cElXMLTree * aTree);


        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;
    private:
        Pt3dr mIncertitude;
};
cElXMLTree * ToXMLTree(const cORGI_CentreCommun &);

class cORGI_TetaCommun
{
    public:
        friend void xml_init(cORGI_TetaCommun & anObj,cElXMLTree * aTree);


        Pt3dr & Incertitude();
        const Pt3dr & Incertitude()const ;
    private:
        Pt3dr mIncertitude;
};
cElXMLTree * ToXMLTree(const cORGI_TetaCommun &);

class cObsRigidGrpImage
{
    public:
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

class cTxtRapDetaille
{
    public:
        friend void xml_init(cTxtRapDetaille & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;
    private:
        std::string mNameFile;
};
cElXMLTree * ToXMLTree(const cTxtRapDetaille &);

class cSectionObservations
{
    public:
        friend void xml_init(cSectionObservations & anObj,cElXMLTree * aTree);


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
    private:
        std::list< cObsAppuis > mObsAppuis;
        std::list< cObsAppuisFlottant > mObsAppuisFlottant;
        std::list< cObsLiaisons > mObsLiaisons;
        std::list< cObsCentrePDV > mObsCentrePDV;
        std::list< cObsRigidGrpImage > mObsRigidGrpImage;
        cTplValGesInit< cTxtRapDetaille > mTxtRapDetaille;
};
cElXMLTree * ToXMLTree(const cSectionObservations &);

class cExportAsGrid
{
    public:
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

class cExportCalib
{
    public:
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

class cExportPose
{
    public:
        friend void xml_init(cExportPose & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChangementCoordonnees > & ChC();
        const cTplValGesInit< cChangementCoordonnees > & ChC()const ;

        std::string & KeyAssoc();
        const std::string & KeyAssoc()const ;

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
        cTplValGesInit< cChangementCoordonnees > mChC;
        std::string mKeyAssoc;
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

class cExportAttrPose
{
    public:
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

class cExportOrthoCyl
{
    public:
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

class cExportRepereLoc
{
    public:
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
        cTplValGesInit< cExportOrthoCyl > mExportOrthoCyl;
};
cElXMLTree * ToXMLTree(const cExportRepereLoc &);

class cCartes2Export
{
    public:
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

class cExportMesuresFromCarteProf
{
    public:
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

class cExportVisuConfigGrpPose
{
    public:
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

class cExportPtsFlottant
{
    public:
        friend void xml_init(cExportPtsFlottant & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< std::string > & NameFileXml();
        const cTplValGesInit< std::string > & NameFileXml()const ;

        cTplValGesInit< std::string > & NameFileTxt();
        const cTplValGesInit< std::string > & NameFileTxt()const ;

        cTplValGesInit< std::string > & TextComplTxt();
        const cTplValGesInit< std::string > & TextComplTxt()const ;
    private:
        cTplValGesInit< std::string > mPatternSel;
        cTplValGesInit< std::string > mNameFileXml;
        cTplValGesInit< std::string > mNameFileTxt;
        cTplValGesInit< std::string > mTextComplTxt;
};
cElXMLTree * ToXMLTree(const cExportPtsFlottant &);

class cResidusIndiv
{
    public:
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

class cExportImResiduLiaison
{
    public:
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

class cExportRedressement
{
    public:
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

class cNuagePutCam
{
    public:
        friend void xml_init(cNuagePutCam & anObj,cElXMLTree * aTree);


        Pt3di & ColCadre();
        const Pt3di & ColCadre()const ;

        cTplValGesInit< Pt3di > & ColRay();
        const cTplValGesInit< Pt3di > & ColRay()const ;

        double & Long();
        const double & Long()const ;

        double & StepSeg();
        const double & StepSeg()const ;

        cTplValGesInit< double > & StepImage();
        const cTplValGesInit< double > & StepImage()const ;
    private:
        Pt3di mColCadre;
        cTplValGesInit< Pt3di > mColRay;
        double mLong;
        double mStepSeg;
        cTplValGesInit< double > mStepImage;
};
cElXMLTree * ToXMLTree(const cNuagePutCam &);

class cExportNuage
{
    public:
        friend void xml_init(cExportNuage & anObj,cElXMLTree * aTree);


        std::string & NameOut();
        const std::string & NameOut()const ;

        cTplValGesInit< bool > & PlyModeBin();
        const cTplValGesInit< bool > & PlyModeBin()const ;

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

        double & Seg();
        const double & StepSeg()const ;

        cTplValGesInit< double > & StepImage();
        const cTplValGesInit< double > & StepImage()const ;

        cTplValGesInit< cNuagePutCam > & NuagePutCam();
        const cTplValGesInit< cNuagePutCam > & NuagePutCam()const ;
    private:
        std::string mNameOut;
        cTplValGesInit< bool > mPlyModeBin;
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
};
cElXMLTree * ToXMLTree(const cExportNuage &);

class cChoixImSec
{
    public:
        friend void xml_init(cChoixImSec & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< int > & NbMinPresel();
        const cTplValGesInit< int > & NbMinPresel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;
    private:
        cTplValGesInit< std::string > mPatternSel;
        int mNbMin;
        std::string mIdBdl;
        cTplValGesInit< int > mNbMaxPresel;
        cTplValGesInit< int > mNbMinPtsHom;
        cTplValGesInit< double > mTetaMaxPreSel;
        cTplValGesInit< int > mNbMinPresel;
        cTplValGesInit< double > mTetaOpt;
};
cElXMLTree * ToXMLTree(const cChoixImSec &);

class cChoixImMM
{
    public:
        friend void xml_init(cChoixImMM & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< int > & NbMinPresel();
        const cTplValGesInit< int > & NbMinPresel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cChoixImSec & ChoixImSec();
        const cChoixImSec & ChoixImSec()const ;
    private:
        cChoixImSec mChoixImSec;
};
cElXMLTree * ToXMLTree(const cChoixImMM &);

class cSectionExport
{
    public:
        friend void xml_init(cSectionExport & anObj,cElXMLTree * aTree);


        std::list< cExportCalib > & ExportCalib();
        const std::list< cExportCalib > & ExportCalib()const ;

        std::list< cExportPose > & ExportPose();
        const std::list< cExportPose > & ExportPose()const ;

        std::list< cExportAttrPose > & ExportAttrPose();
        const std::list< cExportAttrPose > & ExportAttrPose()const ;

        std::list< cExportRepereLoc > & ExportRepereLoc();
        const std::list< cExportRepereLoc > & ExportRepereLoc()const ;

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

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< int > & NbMinPresel();
        const cTplValGesInit< int > & NbMinPresel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cChoixImSec & ChoixImSec();
        const cChoixImSec & ChoixImSec()const ;

        cTplValGesInit< cChoixImMM > & ChoixImMM();
        const cTplValGesInit< cChoixImMM > & ChoixImMM()const ;
    private:
        std::list< cExportCalib > mExportCalib;
        std::list< cExportPose > mExportPose;
        std::list< cExportAttrPose > mExportAttrPose;
        std::list< cExportRepereLoc > mExportRepereLoc;
        std::list< cExportMesuresFromCarteProf > mExportMesuresFromCarteProf;
        std::list< cExportVisuConfigGrpPose > mExportVisuConfigGrpPose;
        cTplValGesInit< cExportPtsFlottant > mExportPtsFlottant;
        std::list< cExportImResiduLiaison > mExportImResiduLiaison;
        std::list< cExportRedressement > mExportRedressement;
        std::list< cExportNuage > mExportNuage;
        cTplValGesInit< cChoixImMM > mChoixImMM;
};
cElXMLTree * ToXMLTree(const cSectionExport &);

class cEtapeCompensation
{
    public:
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

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        int & NbMin();
        const int & NbMin()const ;

        std::string & IdBdl();
        const std::string & IdBdl()const ;

        cTplValGesInit< int > & NbMaxPresel();
        const cTplValGesInit< int > & NbMaxPresel()const ;

        cTplValGesInit< int > & NbMinPtsHom();
        const cTplValGesInit< int > & NbMinPtsHom()const ;

        cTplValGesInit< double > & TetaMaxPreSel();
        const cTplValGesInit< double > & TetaMaxPreSel()const ;

        cTplValGesInit< int > & NbMinPresel();
        const cTplValGesInit< int > & NbMinPresel()const ;

        cTplValGesInit< double > & TetaOpt();
        const cTplValGesInit< double > & TetaOpt()const ;

        cChoixImSec & ChoixImSec();
        const cChoixImSec & ChoixImSec()const ;

        cTplValGesInit< cChoixImMM > & ChoixImMM();
        const cTplValGesInit< cChoixImMM > & ChoixImMM()const ;

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

class cSectionCompensation
{
    public:
        friend void xml_init(cSectionCompensation & anObj,cElXMLTree * aTree);


        std::list< cEtapeCompensation > & EtapeCompensation();
        const std::list< cEtapeCompensation > & EtapeCompensation()const ;
    private:
        std::list< cEtapeCompensation > mEtapeCompensation;
};
cElXMLTree * ToXMLTree(const cSectionCompensation &);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamApero
{
    public:
        friend void xml_init(cParamApero & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

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

        std::list< cCalibrationCameraInc > & CalibrationCameraInc();
        const std::list< cCalibrationCameraInc > & CalibrationCameraInc()const ;

        cTplValGesInit< int > & SeuilL1EstimMatrEss();
        const cTplValGesInit< int > & SeuilL1EstimMatrEss()const ;

        cTplValGesInit< cSetOrientationInterne > & GlobOrInterne();
        const cTplValGesInit< cSetOrientationInterne > & GlobOrInterne()const ;

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

        cSectionSolveur & SectionSolveur();
        const cSectionSolveur & SectionSolveur()const ;

        std::list< cEtapeCompensation > & EtapeCompensation();
        const std::list< cEtapeCompensation > & EtapeCompensation()const ;

        cSectionCompensation & SectionCompensation();
        const cSectionCompensation & SectionCompensation()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
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

/******************************************************/
/******************************************************/
/******************************************************/
};
#endif // Define_NotApero
