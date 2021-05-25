#include "general/all.h"
#include "private/all.h"
#ifndef Define_NotMicMac
#define Define_NotMicMac
#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;
namespace NS_ParamMICMAC{
typedef enum
{
  eGeomMECIm1,
  eGeomMECTerrain,
  eNoGeomMEC
} eModeGeomMEC;
void xml_init(eModeGeomMEC & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeGeomMEC & aVal);

eModeGeomMEC  Str2eModeGeomMEC(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeGeomMEC & anObj);

typedef enum
{
  eTMA_Homologues,
  eTMA_DHomD,
  eTMA_Ori,
  eTMA_Nuage3D
} eTypeModeleAnalytique;
void xml_init(eTypeModeleAnalytique & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeModeleAnalytique & aVal);

eTypeModeleAnalytique  Str2eTypeModeleAnalytique(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeModeleAnalytique & anObj);

typedef enum
{
  eGeomImageOri,
  eGeomImageModule,
  eGeomImageGrille,
  eGeomImageRTO,
  eGeomImageCON,
  eGeomImageDHD_Px,
  eGeomImage_Hom_Px,
  eGeomImageDH_Px_HD,
  eGeomImage_Epip,
  eGeomImage_EpipolairePure,
  eNoGeomIm
} eModeGeomImage;
void xml_init(eModeGeomImage & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeGeomImage & aVal);

eModeGeomImage  Str2eModeGeomImage(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeGeomImage & anObj);

typedef enum
{
  eAggregSymetrique,
  eAggregIm1Maitre,
  eAggregInfoMut,
  eAggregMaxIm1Maitre
} eModeAggregCorr;
void xml_init(eModeAggregCorr & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeAggregCorr & aVal);

eModeAggregCorr  Str2eModeAggregCorr(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeAggregCorr & anObj);

typedef enum
{
  eCoeffCorrelStd,
  eCoeffAngle
} eModeDynamiqueCorrel;
void xml_init(eModeDynamiqueCorrel & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeDynamiqueCorrel & aVal);

eModeDynamiqueCorrel  Str2eModeDynamiqueCorrel(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeDynamiqueCorrel & anObj);

typedef enum
{
  eUInt8Bits,
  eUInt16Bits,
  eFloat32Bits
} eTypeImPyram;
void xml_init(eTypeImPyram & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeImPyram & aVal);

eTypeImPyram  Str2eTypeImPyram(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeImPyram & anObj);

typedef enum
{
  eAlgoCoxRoy,
  eAlgo2PrgDyn,
  eAlgoMaxOfScore,
  eAlgoCoxRoySiPossible,
  eAlgoOptimDifferentielle,
  eAlgoDequant
} eAlgoRegul;
void xml_init(eAlgoRegul & aVal,cElXMLTree * aTree);
std::string  eToString(const eAlgoRegul & aVal);

eAlgoRegul  Str2eAlgoRegul(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eAlgoRegul & anObj);

typedef enum
{
  eInterpolPPV,
  eInterpolBiLin,
  eInterpolBiCub,
  eInterpolSinCard,
  eOldInterpolSinCard,
  eInterpolMPD,
  eInterpolBicubOpt
} eModeInterpolation;
void xml_init(eModeInterpolation & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeInterpolation & aVal);

eModeInterpolation  Str2eModeInterpolation(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeInterpolation & anObj);

typedef enum
{
  eFiltrageMedian,
  eFiltrageMoyenne,
  eFiltrageDeriche,
  eFiltrageGamma,
  eFiltrageEqLoc
} eTypeFiltrage;
void xml_init(eTypeFiltrage & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeFiltrage & aVal);

eTypeFiltrage  Str2eTypeFiltrage(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeFiltrage & anObj);

typedef enum
{
  eApplyPx1,
  eApplyPx2,
  eApplyPx12
} ePxApply;
void xml_init(ePxApply & aVal,cElXMLTree * aTree);
std::string  eToString(const ePxApply & aVal);

ePxApply  Str2ePxApply(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const ePxApply & anObj);

typedef enum
{
  ePrgDAgrSomme,
  ePrgDAgrMax,
  ePrgDAgrReinject,
  ePrgDAgrProgressif
} eModeAggregProgDyn;
void xml_init(eModeAggregProgDyn & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeAggregProgDyn & aVal);

eModeAggregProgDyn  Str2eModeAggregProgDyn(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeAggregProgDyn & anObj);

typedef enum
{
  eErrNbPointInEqOriRel,
  eErrImageFileEmpty,
  eErrPtHomHorsImage,
  eErrRecouvrInsuffisant,
  eErrGrilleInverseNonDisponible
} eMicMacCodeRetourErreur;
void xml_init(eMicMacCodeRetourErreur & aVal,cElXMLTree * aTree);
std::string  eToString(const eMicMacCodeRetourErreur & aVal);

eMicMacCodeRetourErreur  Str2eMicMacCodeRetourErreur(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eMicMacCodeRetourErreur & anObj);

typedef enum
{
  eWInCorrelFixe,
  eWInCorrelExp,
  eWInCorrelRectSpec
} eTypeWinCorrel;
void xml_init(eTypeWinCorrel & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeWinCorrel & aVal);

eTypeWinCorrel  Str2eTypeWinCorrel(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeWinCorrel & anObj);

typedef enum
{
  eModeEchantRegulier,
  eModeEchantNonAutoCor,
  eModeEchantAleatoire,
  eModeEchantPtsIntByComandeExterne
} eTypeModeEchantPtsI;
void xml_init(eTypeModeEchantPtsI & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeModeEchantPtsI & aVal);

eTypeModeEchantPtsI  Str2eTypeModeEchantPtsI(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeModeEchantPtsI & anObj);

typedef enum
{
  eSLL_Geom_X,
  eSLL_Geom_Y,
  eSLL_Geom_Z,
  eSLL_Geom_dir_X,
  eSLL_Geom_dir_Y,
  eSLL_Geom_dir_Z,
  eSLL_Radiom_R,
  eSLL_Radiom_G,
  eSLL_Radiom_B,
  eSLL_Radiom_Panchro,
  eSLL_Radiom_Pir,
  eSLL_Radiom_Lidar,
  eSLL_Radiom_Unknown,
  eSLL_Unknown
} eSemantiqueLL;
void xml_init(eSemantiqueLL & aVal,cElXMLTree * aTree);
std::string  eToString(const eSemantiqueLL & aVal);

eSemantiqueLL  Str2eSemantiqueLL(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eSemantiqueLL & anObj);

class cSpecFitrageImage
{
    public:
        friend void xml_init(cSpecFitrageImage & anObj,cElXMLTree * aTree);


        eTypeFiltrage & TypeFiltrage();
        const eTypeFiltrage & TypeFiltrage()const ;

        double & SzFiltrage();
        const double & SzFiltrage()const ;

        cTplValGesInit< double > & SzFiltrNonAd();
        const cTplValGesInit< double > & SzFiltrNonAd()const ;

        cTplValGesInit< ePxApply > & PxApply();
        const cTplValGesInit< ePxApply > & PxApply()const ;

        cTplValGesInit< cElRegex_Ptr > & PatternSelFiltre();
        const cTplValGesInit< cElRegex_Ptr > & PatternSelFiltre()const ;

        cTplValGesInit< int > & NbIteration();
        const cTplValGesInit< int > & NbIteration()const ;

        cTplValGesInit< double > & AmplitudeSignal();
        const cTplValGesInit< double > & AmplitudeSignal()const ;

        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;
    private:
        eTypeFiltrage mTypeFiltrage;
        double mSzFiltrage;
        cTplValGesInit< double > mSzFiltrNonAd;
        cTplValGesInit< ePxApply > mPxApply;
        cTplValGesInit< cElRegex_Ptr > mPatternSelFiltre;
        cTplValGesInit< int > mNbIteration;
        cTplValGesInit< double > mAmplitudeSignal;
        cTplValGesInit< bool > mUseIt;
};
cElXMLTree * ToXMLTree(const cSpecFitrageImage &);

/******************************************************/
/******************************************************/
/******************************************************/
class cCorrectionPxTransverse
{
    public:
        friend void xml_init(cCorrectionPxTransverse & anObj,cElXMLTree * aTree);


        Pt2dr & DirPx();
        const Pt2dr & DirPx()const ;

        Im2D_REAL4 & ValeurPx();
        const Im2D_REAL4 & ValeurPx()const ;

        double & SsResol();
        const double & SsResol()const ;
    private:
        Pt2dr mDirPx;
        Im2D_REAL4 mValeurPx;
        double mSsResol;
};
cElXMLTree * ToXMLTree(const cCorrectionPxTransverse &);

/******************************************************/
/******************************************************/
/******************************************************/
class cLidarLayer
{
    public:
        friend void xml_init(cLidarLayer & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        eSemantiqueLL & Semantic();
        const eSemantiqueLL & Semantic()const ;

        cTplValGesInit< double > & LongueurDOnde();
        const cTplValGesInit< double > & LongueurDOnde()const ;

        cTplValGesInit< double > & OffsetValues();
        const cTplValGesInit< double > & OffsetValues()const ;

        cTplValGesInit< double > & StepValues();
        const cTplValGesInit< double > & StepValues()const ;

        bool & IntegerValues();
        const bool & IntegerValues()const ;

        bool & SignedValues();
        const bool & SignedValues()const ;

        int & BytePerValues();
        const int & BytePerValues()const ;

        int & OffsetDataInFile();
        const int & OffsetDataInFile()const ;
    private:
        std::string mNameFile;
        eSemantiqueLL mSemantic;
        cTplValGesInit< double > mLongueurDOnde;
        cTplValGesInit< double > mOffsetValues;
        cTplValGesInit< double > mStepValues;
        bool mIntegerValues;
        bool mSignedValues;
        int mBytePerValues;
        int mOffsetDataInFile;
};
cElXMLTree * ToXMLTree(const cLidarLayer &);

class cGeometrieAffineApprochee
{
    public:
        friend void xml_init(cGeometrieAffineApprochee & anObj,cElXMLTree * aTree);


        Pt2dr & ImTerain_P00();
        const Pt2dr & ImTerain_P00()const ;

        Pt2dr & DerImTerain_Di();
        const Pt2dr & DerImTerain_Di()const ;

        Pt2dr & DerImTerain_Dj();
        const Pt2dr & DerImTerain_Dj()const ;
    private:
        Pt2dr mImTerain_P00;
        Pt2dr mDerImTerain_Di;
        Pt2dr mDerImTerain_Dj;
};
cElXMLTree * ToXMLTree(const cGeometrieAffineApprochee &);

class cLidarStrip
{
    public:
        friend void xml_init(cLidarStrip & anObj,cElXMLTree * aTree);


        std::list< cLidarLayer > & LidarLayer();
        const std::list< cLidarLayer > & LidarLayer()const ;

        bool & FileIs2DStructured();
        const bool & FileIs2DStructured()const ;

        Pt2dr & ImTerain_P00();
        const Pt2dr & ImTerain_P00()const ;

        Pt2dr & DerImTerain_Di();
        const Pt2dr & DerImTerain_Di()const ;

        Pt2dr & DerImTerain_Dj();
        const Pt2dr & DerImTerain_Dj()const ;

        cTplValGesInit< cGeometrieAffineApprochee > & GeometrieAffineApprochee();
        const cTplValGesInit< cGeometrieAffineApprochee > & GeometrieAffineApprochee()const ;

        Box2dr & BoiteEnglob();
        const Box2dr & BoiteEnglob()const ;
    private:
        std::list< cLidarLayer > mLidarLayer;
        bool mFileIs2DStructured;
        cTplValGesInit< cGeometrieAffineApprochee > mGeometrieAffineApprochee;
        Box2dr mBoiteEnglob;
};
cElXMLTree * ToXMLTree(const cLidarStrip &);

/******************************************************/
/******************************************************/
/******************************************************/
class cLidarFlight
{
    public:
        friend void xml_init(cLidarFlight & anObj,cElXMLTree * aTree);


        std::string & SystemeCoordonnees();
        const std::string & SystemeCoordonnees()const ;

        std::list< cLidarStrip > & LidarStrip();
        const std::list< cLidarStrip > & LidarStrip()const ;

        Box2dr & BoiteEnglob();
        const Box2dr & BoiteEnglob()const ;
    private:
        std::string mSystemeCoordonnees;
        std::list< cLidarStrip > mLidarStrip;
        Box2dr mBoiteEnglob;
};
cElXMLTree * ToXMLTree(const cLidarFlight &);

/******************************************************/
/******************************************************/
/******************************************************/
class cMemPartMICMAC
{
    public:
        friend void xml_init(cMemPartMICMAC & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NbMaxImageOn1Point();
        const cTplValGesInit< int > & NbMaxImageOn1Point()const ;

        cTplValGesInit< double > & BSurHGlob();
        const cTplValGesInit< double > & BSurHGlob()const ;
    private:
        cTplValGesInit< int > mNbMaxImageOn1Point;
        cTplValGesInit< double > mBSurHGlob;
};
cElXMLTree * ToXMLTree(const cMemPartMICMAC &);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamMasqAnam
{
    public:
        friend void xml_init(cParamMasqAnam & anObj,cElXMLTree * aTree);


        Box2dr & BoxTer();
        const Box2dr & BoxTer()const ;

        double & Resol();
        const double & Resol()const ;
    private:
        Box2dr mBoxTer;
        double mResol;
};
cElXMLTree * ToXMLTree(const cParamMasqAnam &);

/******************************************************/
/******************************************************/
/******************************************************/
class cMM_EtatAvancement
{
    public:
        friend void xml_init(cMM_EtatAvancement & anObj,cElXMLTree * aTree);


        bool & AllDone();
        const bool & AllDone()const ;
    private:
        bool mAllDone;
};
cElXMLTree * ToXMLTree(const cMM_EtatAvancement &);

/******************************************************/
/******************************************************/
/******************************************************/
class cImageFDC
{
    public:
        friend void xml_init(cImageFDC & anObj,cElXMLTree * aTree);


        std::string & FDCIm();
        const std::string & FDCIm()const ;

        cTplValGesInit< Pt2dr > & DirEpipTransv();
        const cTplValGesInit< Pt2dr > & DirEpipTransv()const ;
    private:
        std::string mFDCIm;
        cTplValGesInit< Pt2dr > mDirEpipTransv;
};
cElXMLTree * ToXMLTree(const cImageFDC &);

/******************************************************/
/******************************************************/
/******************************************************/
class cCouplesFDC
{
    public:
        friend void xml_init(cCouplesFDC & anObj,cElXMLTree * aTree);


        std::string & FDCIm1();
        const std::string & FDCIm1()const ;

        std::string & FDCIm2();
        const std::string & FDCIm2()const ;

        cTplValGesInit< double > & BSurH();
        const cTplValGesInit< double > & BSurH()const ;
    private:
        std::string mFDCIm1;
        std::string mFDCIm2;
        cTplValGesInit< double > mBSurH;
};
cElXMLTree * ToXMLTree(const cCouplesFDC &);

/******************************************************/
/******************************************************/
/******************************************************/
class cFileDescriptionChantier
{
    public:
        friend void xml_init(cFileDescriptionChantier & anObj,cElXMLTree * aTree);


        std::list< cImageFDC > & ImageFDC();
        const std::list< cImageFDC > & ImageFDC()const ;

        std::list< cCouplesFDC > & CouplesFDC();
        const std::list< cCouplesFDC > & CouplesFDC()const ;
    private:
        std::list< cImageFDC > mImageFDC;
        std::list< cCouplesFDC > mCouplesFDC;
};
cElXMLTree * ToXMLTree(const cFileDescriptionChantier &);

/******************************************************/
/******************************************************/
/******************************************************/
class cBoxMasqIsBoxTer
{
    public:
        friend void xml_init(cBoxMasqIsBoxTer & anObj,cElXMLTree * aTree);


        Box2dr & Box();
        const Box2dr & Box()const ;
    private:
        Box2dr mBox;
};
cElXMLTree * ToXMLTree(const cBoxMasqIsBoxTer &);

/******************************************************/
/******************************************************/
/******************************************************/
class cMNT_Init
{
    public:
        friend void xml_init(cMNT_Init & anObj,cElXMLTree * aTree);


        std::string & MNT_Init_Image();
        const std::string & MNT_Init_Image()const ;

        std::string & MNT_Init_Xml();
        const std::string & MNT_Init_Xml()const ;

        cTplValGesInit< double > & MNT_Offset();
        const cTplValGesInit< double > & MNT_Offset()const ;
    private:
        std::string mMNT_Init_Image;
        std::string mMNT_Init_Xml;
        cTplValGesInit< double > mMNT_Offset;
};
cElXMLTree * ToXMLTree(const cMNT_Init &);

class cEnveloppeMNT_INIT
{
    public:
        friend void xml_init(cEnveloppeMNT_INIT & anObj,cElXMLTree * aTree);


        std::string & ZInf();
        const std::string & ZInf()const ;

        std::string & ZSup();
        const std::string & ZSup()const ;
    private:
        std::string mZInf;
        std::string mZSup;
};
cElXMLTree * ToXMLTree(const cEnveloppeMNT_INIT &);

class cIntervAltimetrie
{
    public:
        friend void xml_init(cIntervAltimetrie & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        double & ZIncCalc();
        const double & ZIncCalc()const ;

        cTplValGesInit< bool > & ZIncIsProp();
        const cTplValGesInit< bool > & ZIncIsProp()const ;

        cTplValGesInit< double > & ZIncZonage();
        const cTplValGesInit< double > & ZIncZonage()const ;

        std::string & MNT_Init_Image();
        const std::string & MNT_Init_Image()const ;

        std::string & MNT_Init_Xml();
        const std::string & MNT_Init_Xml()const ;

        cTplValGesInit< double > & MNT_Offset();
        const cTplValGesInit< double > & MNT_Offset()const ;

        cTplValGesInit< cMNT_Init > & MNT_Init();
        const cTplValGesInit< cMNT_Init > & MNT_Init()const ;

        std::string & ZInf();
        const std::string & ZInf()const ;

        std::string & ZSup();
        const std::string & ZSup()const ;

        cTplValGesInit< cEnveloppeMNT_INIT > & EnveloppeMNT_INIT();
        const cTplValGesInit< cEnveloppeMNT_INIT > & EnveloppeMNT_INIT()const ;
    private:
        cTplValGesInit< double > mZMoyen;
        double mZIncCalc;
        cTplValGesInit< bool > mZIncIsProp;
        cTplValGesInit< double > mZIncZonage;
        cTplValGesInit< cMNT_Init > mMNT_Init;
        cTplValGesInit< cEnveloppeMNT_INIT > mEnveloppeMNT_INIT;
};
cElXMLTree * ToXMLTree(const cIntervAltimetrie &);

class cIntervParalaxe
{
    public:
        friend void xml_init(cIntervParalaxe & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & Px1Moy();
        const cTplValGesInit< double > & Px1Moy()const ;

        cTplValGesInit< double > & Px2Moy();
        const cTplValGesInit< double > & Px2Moy()const ;

        double & Px1IncCalc();
        const double & Px1IncCalc()const ;

        cTplValGesInit< double > & Px1PropProf();
        const cTplValGesInit< double > & Px1PropProf()const ;

        cTplValGesInit< double > & Px2IncCalc();
        const cTplValGesInit< double > & Px2IncCalc()const ;

        cTplValGesInit< double > & Px1IncZonage();
        const cTplValGesInit< double > & Px1IncZonage()const ;

        cTplValGesInit< double > & Px2IncZonage();
        const cTplValGesInit< double > & Px2IncZonage()const ;
    private:
        cTplValGesInit< double > mPx1Moy;
        cTplValGesInit< double > mPx2Moy;
        double mPx1IncCalc;
        cTplValGesInit< double > mPx1PropProf;
        cTplValGesInit< double > mPx2IncCalc;
        cTplValGesInit< double > mPx1IncZonage;
        cTplValGesInit< double > mPx2IncZonage;
};
cElXMLTree * ToXMLTree(const cIntervParalaxe &);

class cIntervSpecialZInv
{
    public:
        friend void xml_init(cIntervSpecialZInv & anObj,cElXMLTree * aTree);


        double & MulZMin();
        const double & MulZMin()const ;

        double & MulZMax();
        const double & MulZMax()const ;
    private:
        double mMulZMin;
        double mMulZMax;
};
cElXMLTree * ToXMLTree(const cIntervSpecialZInv &);

class cListePointsInclus
{
    public:
        friend void xml_init(cListePointsInclus & anObj,cElXMLTree * aTree);


        std::list< Pt2dr > & Pt();
        const std::list< Pt2dr > & Pt()const ;

        std::string & Im();
        const std::string & Im()const ;
    private:
        std::list< Pt2dr > mPt;
        std::string mIm;
};
cElXMLTree * ToXMLTree(const cListePointsInclus &);

class cMasqueTerrain
{
    public:
        friend void xml_init(cMasqueTerrain & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & FileBoxMasqIsBoxTer();
        const cTplValGesInit< std::string > & FileBoxMasqIsBoxTer()const ;

        std::string & MT_Image();
        const std::string & MT_Image()const ;

        std::string & MT_Xml();
        const std::string & MT_Xml()const ;
    private:
        cTplValGesInit< std::string > mFileBoxMasqIsBoxTer;
        std::string mMT_Image;
        std::string mMT_Xml;
};
cElXMLTree * ToXMLTree(const cMasqueTerrain &);

class cPlanimetrie
{
    public:
        friend void xml_init(cPlanimetrie & anObj,cElXMLTree * aTree);


        cTplValGesInit< Box2dr > & BoxTerrain();
        const cTplValGesInit< Box2dr > & BoxTerrain()const ;

        std::list< cListePointsInclus > & ListePointsInclus();
        const std::list< cListePointsInclus > & ListePointsInclus()const ;

        cTplValGesInit< double > & RatioResolImage();
        const cTplValGesInit< double > & RatioResolImage()const ;

        cTplValGesInit< double > & ResolutionTerrain();
        const cTplValGesInit< double > & ResolutionTerrain()const ;

        cTplValGesInit< std::string > & FilterEstimTerrain();
        const cTplValGesInit< std::string > & FilterEstimTerrain()const ;

        cTplValGesInit< std::string > & FileBoxMasqIsBoxTer();
        const cTplValGesInit< std::string > & FileBoxMasqIsBoxTer()const ;

        std::string & MT_Image();
        const std::string & MT_Image()const ;

        std::string & MT_Xml();
        const std::string & MT_Xml()const ;

        cTplValGesInit< cMasqueTerrain > & MasqueTerrain();
        const cTplValGesInit< cMasqueTerrain > & MasqueTerrain()const ;

        cTplValGesInit< double > & RecouvrementMinimal();
        const cTplValGesInit< double > & RecouvrementMinimal()const ;
    private:
        cTplValGesInit< Box2dr > mBoxTerrain;
        std::list< cListePointsInclus > mListePointsInclus;
        cTplValGesInit< double > mRatioResolImage;
        cTplValGesInit< double > mResolutionTerrain;
        cTplValGesInit< std::string > mFilterEstimTerrain;
        cTplValGesInit< cMasqueTerrain > mMasqueTerrain;
        cTplValGesInit< double > mRecouvrementMinimal;
};
cElXMLTree * ToXMLTree(const cPlanimetrie &);

class cRugositeMNT
{
    public:
        friend void xml_init(cRugositeMNT & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & EnergieExpCorrel();
        const cTplValGesInit< double > & EnergieExpCorrel()const ;

        cTplValGesInit< double > & EnergieExpRegulPlani();
        const cTplValGesInit< double > & EnergieExpRegulPlani()const ;

        cTplValGesInit< double > & EnergieExpRegulAlti();
        const cTplValGesInit< double > & EnergieExpRegulAlti()const ;
    private:
        cTplValGesInit< double > mEnergieExpCorrel;
        cTplValGesInit< double > mEnergieExpRegulPlani;
        cTplValGesInit< double > mEnergieExpRegulAlti;
};
cElXMLTree * ToXMLTree(const cRugositeMNT &);

class cSection_Terrain
{
    public:
        friend void xml_init(cSection_Terrain & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & RatioAltiPlani();
        const cTplValGesInit< double > & RatioAltiPlani()const ;

        cTplValGesInit< bool > & EstimPxPrefZ2Prof();
        const cTplValGesInit< bool > & EstimPxPrefZ2Prof()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        double & ZIncCalc();
        const double & ZIncCalc()const ;

        cTplValGesInit< bool > & ZIncIsProp();
        const cTplValGesInit< bool > & ZIncIsProp()const ;

        cTplValGesInit< double > & ZIncZonage();
        const cTplValGesInit< double > & ZIncZonage()const ;

        std::string & MNT_Init_Image();
        const std::string & MNT_Init_Image()const ;

        std::string & MNT_Init_Xml();
        const std::string & MNT_Init_Xml()const ;

        cTplValGesInit< double > & MNT_Offset();
        const cTplValGesInit< double > & MNT_Offset()const ;

        cTplValGesInit< cMNT_Init > & MNT_Init();
        const cTplValGesInit< cMNT_Init > & MNT_Init()const ;

        std::string & ZInf();
        const std::string & ZInf()const ;

        std::string & ZSup();
        const std::string & ZSup()const ;

        cTplValGesInit< cEnveloppeMNT_INIT > & EnveloppeMNT_INIT();
        const cTplValGesInit< cEnveloppeMNT_INIT > & EnveloppeMNT_INIT()const ;

        cTplValGesInit< cIntervAltimetrie > & IntervAltimetrie();
        const cTplValGesInit< cIntervAltimetrie > & IntervAltimetrie()const ;

        cTplValGesInit< double > & Px1Moy();
        const cTplValGesInit< double > & Px1Moy()const ;

        cTplValGesInit< double > & Px2Moy();
        const cTplValGesInit< double > & Px2Moy()const ;

        double & Px1IncCalc();
        const double & Px1IncCalc()const ;

        cTplValGesInit< double > & Px1PropProf();
        const cTplValGesInit< double > & Px1PropProf()const ;

        cTplValGesInit< double > & Px2IncCalc();
        const cTplValGesInit< double > & Px2IncCalc()const ;

        cTplValGesInit< double > & Px1IncZonage();
        const cTplValGesInit< double > & Px1IncZonage()const ;

        cTplValGesInit< double > & Px2IncZonage();
        const cTplValGesInit< double > & Px2IncZonage()const ;

        cTplValGesInit< cIntervParalaxe > & IntervParalaxe();
        const cTplValGesInit< cIntervParalaxe > & IntervParalaxe()const ;

        double & MulZMin();
        const double & MulZMin()const ;

        double & MulZMax();
        const double & MulZMax()const ;

        cTplValGesInit< cIntervSpecialZInv > & IntervSpecialZInv();
        const cTplValGesInit< cIntervSpecialZInv > & IntervSpecialZInv()const ;

        cTplValGesInit< Box2dr > & BoxTerrain();
        const cTplValGesInit< Box2dr > & BoxTerrain()const ;

        std::list< cListePointsInclus > & ListePointsInclus();
        const std::list< cListePointsInclus > & ListePointsInclus()const ;

        cTplValGesInit< double > & RatioResolImage();
        const cTplValGesInit< double > & RatioResolImage()const ;

        cTplValGesInit< double > & ResolutionTerrain();
        const cTplValGesInit< double > & ResolutionTerrain()const ;

        cTplValGesInit< std::string > & FilterEstimTerrain();
        const cTplValGesInit< std::string > & FilterEstimTerrain()const ;

        cTplValGesInit< std::string > & FileBoxMasqIsBoxTer();
        const cTplValGesInit< std::string > & FileBoxMasqIsBoxTer()const ;

        std::string & MT_Image();
        const std::string & MT_Image()const ;

        std::string & MT_Xml();
        const std::string & MT_Xml()const ;

        cTplValGesInit< cMasqueTerrain > & MasqueTerrain();
        const cTplValGesInit< cMasqueTerrain > & MasqueTerrain()const ;

        cTplValGesInit< double > & RecouvrementMinimal();
        const cTplValGesInit< double > & RecouvrementMinimal()const ;

        cTplValGesInit< cPlanimetrie > & Planimetrie();
        const cTplValGesInit< cPlanimetrie > & Planimetrie()const ;

        cTplValGesInit< std::string > & FileOriMnt();
        const cTplValGesInit< std::string > & FileOriMnt()const ;

        cTplValGesInit< double > & EnergieExpCorrel();
        const cTplValGesInit< double > & EnergieExpCorrel()const ;

        cTplValGesInit< double > & EnergieExpRegulPlani();
        const cTplValGesInit< double > & EnergieExpRegulPlani()const ;

        cTplValGesInit< double > & EnergieExpRegulAlti();
        const cTplValGesInit< double > & EnergieExpRegulAlti()const ;

        cTplValGesInit< cRugositeMNT > & RugositeMNT();
        const cTplValGesInit< cRugositeMNT > & RugositeMNT()const ;
    private:
        cTplValGesInit< double > mRatioAltiPlani;
        cTplValGesInit< bool > mEstimPxPrefZ2Prof;
        cTplValGesInit< cIntervAltimetrie > mIntervAltimetrie;
        cTplValGesInit< cIntervParalaxe > mIntervParalaxe;
        cTplValGesInit< cIntervSpecialZInv > mIntervSpecialZInv;
        cTplValGesInit< cPlanimetrie > mPlanimetrie;
        cTplValGesInit< std::string > mFileOriMnt;
        cTplValGesInit< cRugositeMNT > mRugositeMNT;
};
cElXMLTree * ToXMLTree(const cSection_Terrain &);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneMasqueImage
{
    public:
        friend void xml_init(cOneMasqueImage & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & PatternSel();
        const cElRegex_Ptr & PatternSel()const ;

        std::string & NomMasq();
        const std::string & NomMasq()const ;
    private:
        cElRegex_Ptr mPatternSel;
        std::string mNomMasq;
};
cElXMLTree * ToXMLTree(const cOneMasqueImage &);

class cMasqImageIn
{
    public:
        friend void xml_init(cMasqImageIn & anObj,cElXMLTree * aTree);


        std::list< cOneMasqueImage > & OneMasqueImage();
        const std::list< cOneMasqueImage > & OneMasqueImage()const ;
    private:
        std::list< cOneMasqueImage > mOneMasqueImage;
};
cElXMLTree * ToXMLTree(const cMasqImageIn &);

class cModuleGeomImage
{
    public:
        friend void xml_init(cModuleGeomImage & anObj,cElXMLTree * aTree);


        std::string & NomModule();
        const std::string & NomModule()const ;

        std::string & NomGeometrie();
        const std::string & NomGeometrie()const ;
    private:
        std::string mNomModule;
        std::string mNomGeometrie;
};
cElXMLTree * ToXMLTree(const cModuleGeomImage &);

class cFCND_CalcIm2fromIm1
{
    public:
        friend void xml_init(cFCND_CalcIm2fromIm1 & anObj,cElXMLTree * aTree);


        std::string & I2FromI1Key();
        const std::string & I2FromI1Key()const ;

        bool & I2FromI1SensDirect();
        const bool & I2FromI1SensDirect()const ;
    private:
        std::string mI2FromI1Key;
        bool mI2FromI1SensDirect;
};
cElXMLTree * ToXMLTree(const cFCND_CalcIm2fromIm1 &);

class cAutoSelectionneImSec
{
    public:
        friend void xml_init(cAutoSelectionneImSec & anObj,cElXMLTree * aTree);


        double & RecouvrMin();
        const double & RecouvrMin()const ;
    private:
        double mRecouvrMin;
};
cElXMLTree * ToXMLTree(const cAutoSelectionneImSec &);

class cImages
{
    public:
        friend void xml_init(cImages & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Im1();
        const cTplValGesInit< std::string > & Im1()const ;

        cTplValGesInit< std::string > & Im2();
        const cTplValGesInit< std::string > & Im2()const ;

        std::string & I2FromI1Key();
        const std::string & I2FromI1Key()const ;

        bool & I2FromI1SensDirect();
        const bool & I2FromI1SensDirect()const ;

        cTplValGesInit< cFCND_CalcIm2fromIm1 > & FCND_CalcIm2fromIm1();
        const cTplValGesInit< cFCND_CalcIm2fromIm1 > & FCND_CalcIm2fromIm1()const ;

        std::list< std::string > & ImPat();
        const std::list< std::string > & ImPat()const ;

        cTplValGesInit< cNameFilter > & Filter();
        const cTplValGesInit< cNameFilter > & Filter()const ;

        double & RecouvrMin();
        const double & RecouvrMin()const ;

        cTplValGesInit< cAutoSelectionneImSec > & AutoSelectionneImSec();
        const cTplValGesInit< cAutoSelectionneImSec > & AutoSelectionneImSec()const ;

        cTplValGesInit< cListImByDelta > & ImSecByDelta();
        const cTplValGesInit< cListImByDelta > & ImSecByDelta()const ;

        cTplValGesInit< std::string > & Im3Superp();
        const cTplValGesInit< std::string > & Im3Superp()const ;
    private:
        cTplValGesInit< std::string > mIm1;
        cTplValGesInit< std::string > mIm2;
        cTplValGesInit< cFCND_CalcIm2fromIm1 > mFCND_CalcIm2fromIm1;
        std::list< std::string > mImPat;
        cTplValGesInit< cNameFilter > mFilter;
        cTplValGesInit< cAutoSelectionneImSec > mAutoSelectionneImSec;
        cTplValGesInit< cListImByDelta > mImSecByDelta;
        cTplValGesInit< std::string > mIm3Superp;
};
cElXMLTree * ToXMLTree(const cImages &);

class cFCND_Mode_GeomIm
{
    public:
        friend void xml_init(cFCND_Mode_GeomIm & anObj,cElXMLTree * aTree);


        std::string & FCND_GeomCalc();
        const std::string & FCND_GeomCalc()const ;

        cTplValGesInit< std::string > & FCND_GeomApply();
        const cTplValGesInit< std::string > & FCND_GeomApply()const ;
    private:
        std::string mFCND_GeomCalc;
        cTplValGesInit< std::string > mFCND_GeomApply;
};
cElXMLTree * ToXMLTree(const cFCND_Mode_GeomIm &);

class cModuleImageLoader
{
    public:
        friend void xml_init(cModuleImageLoader & anObj,cElXMLTree * aTree);


        std::string & NomModule();
        const std::string & NomModule()const ;

        std::string & NomLoader();
        const std::string & NomLoader()const ;
    private:
        std::string mNomModule;
        std::string mNomLoader;
};
cElXMLTree * ToXMLTree(const cModuleImageLoader &);

class cCropAndScale
{
    public:
        friend void xml_init(cCropAndScale & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & Scale();
        const cTplValGesInit< double > & Scale()const ;

        cTplValGesInit< Pt2dr > & Crop();
        const cTplValGesInit< Pt2dr > & Crop()const ;

        cTplValGesInit< double > & ScaleY();
        const cTplValGesInit< double > & ScaleY()const ;
    private:
        cTplValGesInit< double > mScale;
        cTplValGesInit< Pt2dr > mCrop;
        cTplValGesInit< double > mScaleY;
};
cElXMLTree * ToXMLTree(const cCropAndScale &);

class cGeom
{
    public:
        friend void xml_init(cGeom & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & Scale();
        const cTplValGesInit< double > & Scale()const ;

        cTplValGesInit< Pt2dr > & Crop();
        const cTplValGesInit< Pt2dr > & Crop()const ;

        cTplValGesInit< double > & ScaleY();
        const cTplValGesInit< double > & ScaleY()const ;

        cTplValGesInit< cCropAndScale > & CropAndScale();
        const cTplValGesInit< cCropAndScale > & CropAndScale()const ;

        cTplValGesInit< std::string > & NamePxTr();
        const cTplValGesInit< std::string > & NamePxTr()const ;
    private:
        cTplValGesInit< cCropAndScale > mCropAndScale;
        cTplValGesInit< std::string > mNamePxTr;
};
cElXMLTree * ToXMLTree(const cGeom &);

class cModifieurGeometrie
{
    public:
        friend void xml_init(cModifieurGeometrie & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & Scale();
        const cTplValGesInit< double > & Scale()const ;

        cTplValGesInit< Pt2dr > & Crop();
        const cTplValGesInit< Pt2dr > & Crop()const ;

        cTplValGesInit< double > & ScaleY();
        const cTplValGesInit< double > & ScaleY()const ;

        cTplValGesInit< cCropAndScale > & CropAndScale();
        const cTplValGesInit< cCropAndScale > & CropAndScale()const ;

        cTplValGesInit< std::string > & NamePxTr();
        const cTplValGesInit< std::string > & NamePxTr()const ;

        cGeom & Geom();
        const cGeom & Geom()const ;

        cTplValGesInit< cElRegex_Ptr > & Apply();
        const cTplValGesInit< cElRegex_Ptr > & Apply()const ;
    private:
        cGeom mGeom;
        cTplValGesInit< cElRegex_Ptr > mApply;
};
cElXMLTree * ToXMLTree(const cModifieurGeometrie &);

class cNomsGeometrieImage
{
    public:
        friend void xml_init(cNomsGeometrieImage & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        cTplValGesInit< std::string > & PatternSel();
        const cTplValGesInit< std::string > & PatternSel()const ;

        cTplValGesInit< std::string > & PatNameGeom();
        const cTplValGesInit< std::string > & PatNameGeom()const ;

        cTplValGesInit< std::string > & PatternNameIm1Im2();
        const cTplValGesInit< std::string > & PatternNameIm1Im2()const ;

        std::string & FCND_GeomCalc();
        const std::string & FCND_GeomCalc()const ;

        cTplValGesInit< std::string > & FCND_GeomApply();
        const cTplValGesInit< std::string > & FCND_GeomApply()const ;

        cTplValGesInit< cFCND_Mode_GeomIm > & FCND_Mode_GeomIm();
        const cTplValGesInit< cFCND_Mode_GeomIm > & FCND_Mode_GeomIm()const ;

        cTplValGesInit< bool > & AddNumToNameGeom();
        const cTplValGesInit< bool > & AddNumToNameGeom()const ;

        std::string & NomModule();
        const std::string & NomModule()const ;

        std::string & NomLoader();
        const std::string & NomLoader()const ;

        cTplValGesInit< cModuleImageLoader > & ModuleImageLoader();
        const cTplValGesInit< cModuleImageLoader > & ModuleImageLoader()const ;

        std::list< int > & GenereOriDeZoom();
        const std::list< int > & GenereOriDeZoom()const ;

        std::list< cModifieurGeometrie > & ModifieurGeometrie();
        const std::list< cModifieurGeometrie > & ModifieurGeometrie()const ;
    private:
        cTplValGesInit< bool > mUseIt;
        cTplValGesInit< std::string > mPatternSel;
        cTplValGesInit< std::string > mPatNameGeom;
        cTplValGesInit< std::string > mPatternNameIm1Im2;
        cTplValGesInit< cFCND_Mode_GeomIm > mFCND_Mode_GeomIm;
        cTplValGesInit< bool > mAddNumToNameGeom;
        cTplValGesInit< cModuleImageLoader > mModuleImageLoader;
        std::list< int > mGenereOriDeZoom;
        std::list< cModifieurGeometrie > mModifieurGeometrie;
};
cElXMLTree * ToXMLTree(const cNomsGeometrieImage &);

class cNomsHomomologues
{
    public:
        friend void xml_init(cNomsHomomologues & anObj,cElXMLTree * aTree);


        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & PatNameGeom();
        const std::string & PatNameGeom()const ;

        cTplValGesInit< std::string > & SeparateurHom();
        const cTplValGesInit< std::string > & SeparateurHom()const ;
    private:
        std::string mPatternSel;
        std::string mPatNameGeom;
        cTplValGesInit< std::string > mSeparateurHom;
};
cElXMLTree * ToXMLTree(const cNomsHomomologues &);

class cSection_PriseDeVue
{
    public:
        friend void xml_init(cSection_PriseDeVue & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & BordImage();
        const cTplValGesInit< int > & BordImage()const ;

        cTplValGesInit< bool > & ConvertToSameOriPtTgtLoc();
        const cTplValGesInit< bool > & ConvertToSameOriPtTgtLoc()const ;

        cTplValGesInit< int > & ValSpecNotImage();
        const cTplValGesInit< int > & ValSpecNotImage()const ;

        cTplValGesInit< std::string > & PrefixMasqImRes();
        const cTplValGesInit< std::string > & PrefixMasqImRes()const ;

        cTplValGesInit< std::string > & DirMasqueImages();
        const cTplValGesInit< std::string > & DirMasqueImages()const ;

        std::list< cMasqImageIn > & MasqImageIn();
        const std::list< cMasqImageIn > & MasqImageIn()const ;

        std::list< cSpecFitrageImage > & FiltreImageIn();
        const std::list< cSpecFitrageImage > & FiltreImageIn()const ;

        eModeGeomImage & GeomImages();
        const eModeGeomImage & GeomImages()const ;

        std::string & NomModule();
        const std::string & NomModule()const ;

        std::string & NomGeometrie();
        const std::string & NomGeometrie()const ;

        cTplValGesInit< cModuleGeomImage > & ModuleGeomImage();
        const cTplValGesInit< cModuleGeomImage > & ModuleGeomImage()const ;

        cTplValGesInit< std::string > & Im1();
        const cTplValGesInit< std::string > & Im1()const ;

        cTplValGesInit< std::string > & Im2();
        const cTplValGesInit< std::string > & Im2()const ;

        std::string & I2FromI1Key();
        const std::string & I2FromI1Key()const ;

        bool & I2FromI1SensDirect();
        const bool & I2FromI1SensDirect()const ;

        cTplValGesInit< cFCND_CalcIm2fromIm1 > & FCND_CalcIm2fromIm1();
        const cTplValGesInit< cFCND_CalcIm2fromIm1 > & FCND_CalcIm2fromIm1()const ;

        std::list< std::string > & ImPat();
        const std::list< std::string > & ImPat()const ;

        cTplValGesInit< cNameFilter > & Filter();
        const cTplValGesInit< cNameFilter > & Filter()const ;

        double & RecouvrMin();
        const double & RecouvrMin()const ;

        cTplValGesInit< cAutoSelectionneImSec > & AutoSelectionneImSec();
        const cTplValGesInit< cAutoSelectionneImSec > & AutoSelectionneImSec()const ;

        cTplValGesInit< cListImByDelta > & ImSecByDelta();
        const cTplValGesInit< cListImByDelta > & ImSecByDelta()const ;

        cTplValGesInit< std::string > & Im3Superp();
        const cTplValGesInit< std::string > & Im3Superp()const ;

        cImages & Images();
        const cImages & Images()const ;

        std::list< cNomsGeometrieImage > & NomsGeometrieImage();
        const std::list< cNomsGeometrieImage > & NomsGeometrieImage()const ;

        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & PatNameGeom();
        const std::string & PatNameGeom()const ;

        cTplValGesInit< std::string > & SeparateurHom();
        const cTplValGesInit< std::string > & SeparateurHom()const ;

        cTplValGesInit< cNomsHomomologues > & NomsHomomologues();
        const cTplValGesInit< cNomsHomomologues > & NomsHomomologues()const ;

        cTplValGesInit< std::string > & FCND_CalcHomFromI1I2();
        const cTplValGesInit< std::string > & FCND_CalcHomFromI1I2()const ;

        cTplValGesInit< bool > & SingulariteInCorresp_I1I2();
        const cTplValGesInit< bool > & SingulariteInCorresp_I1I2()const ;

        cTplValGesInit< cMapName2Name > & ClassEquivalenceImage();
        const cTplValGesInit< cMapName2Name > & ClassEquivalenceImage()const ;
    private:
        cTplValGesInit< int > mBordImage;
        cTplValGesInit< bool > mConvertToSameOriPtTgtLoc;
        cTplValGesInit< int > mValSpecNotImage;
        cTplValGesInit< std::string > mPrefixMasqImRes;
        cTplValGesInit< std::string > mDirMasqueImages;
        std::list< cMasqImageIn > mMasqImageIn;
        std::list< cSpecFitrageImage > mFiltreImageIn;
        eModeGeomImage mGeomImages;
        cTplValGesInit< cModuleGeomImage > mModuleGeomImage;
        cImages mImages;
        std::list< cNomsGeometrieImage > mNomsGeometrieImage;
        cTplValGesInit< cNomsHomomologues > mNomsHomomologues;
        cTplValGesInit< std::string > mFCND_CalcHomFromI1I2;
        cTplValGesInit< bool > mSingulariteInCorresp_I1I2;
        cTplValGesInit< cMapName2Name > mClassEquivalenceImage;
};
cElXMLTree * ToXMLTree(const cSection_PriseDeVue &);

/******************************************************/
/******************************************************/
/******************************************************/
class cEchantillonagePtsInterets
{
    public:
        friend void xml_init(cEchantillonagePtsInterets & anObj,cElXMLTree * aTree);


        int & FreqEchantPtsI();
        const int & FreqEchantPtsI()const ;

        eTypeModeEchantPtsI & ModeEchantPtsI();
        const eTypeModeEchantPtsI & ModeEchantPtsI()const ;

        cTplValGesInit< std::string > & KeyCommandeExterneInteret();
        const cTplValGesInit< std::string > & KeyCommandeExterneInteret()const ;

        cTplValGesInit< int > & SzVAutoCorrel();
        const cTplValGesInit< int > & SzVAutoCorrel()const ;

        cTplValGesInit< double > & EstmBrAutoCorrel();
        const cTplValGesInit< double > & EstmBrAutoCorrel()const ;

        cTplValGesInit< double > & SeuilLambdaAutoCorrel();
        const cTplValGesInit< double > & SeuilLambdaAutoCorrel()const ;

        cTplValGesInit< double > & SeuilEcartTypeAutoCorrel();
        const cTplValGesInit< double > & SeuilEcartTypeAutoCorrel()const ;

        cTplValGesInit< double > & RepartExclusion();
        const cTplValGesInit< double > & RepartExclusion()const ;

        cTplValGesInit< double > & RepartEvitement();
        const cTplValGesInit< double > & RepartEvitement()const ;
    private:
        int mFreqEchantPtsI;
        eTypeModeEchantPtsI mModeEchantPtsI;
        cTplValGesInit< std::string > mKeyCommandeExterneInteret;
        cTplValGesInit< int > mSzVAutoCorrel;
        cTplValGesInit< double > mEstmBrAutoCorrel;
        cTplValGesInit< double > mSeuilLambdaAutoCorrel;
        cTplValGesInit< double > mSeuilEcartTypeAutoCorrel;
        cTplValGesInit< double > mRepartExclusion;
        cTplValGesInit< double > mRepartEvitement;
};
cElXMLTree * ToXMLTree(const cEchantillonagePtsInterets &);

class cAdapteDynCov
{
    public:
        friend void xml_init(cAdapteDynCov & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & CovLim();
        const cTplValGesInit< double > & CovLim()const ;

        cTplValGesInit< double > & TermeDecr();
        const cTplValGesInit< double > & TermeDecr()const ;

        cTplValGesInit< int > & SzRef();
        const cTplValGesInit< int > & SzRef()const ;

        cTplValGesInit< double > & ValRef();
        const cTplValGesInit< double > & ValRef()const ;
    private:
        cTplValGesInit< double > mCovLim;
        cTplValGesInit< double > mTermeDecr;
        cTplValGesInit< int > mSzRef;
        cTplValGesInit< double > mValRef;
};
cElXMLTree * ToXMLTree(const cAdapteDynCov &);

class cGPU_Correl
{
    public:
        friend void xml_init(cGPU_Correl & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Unused();
        const cTplValGesInit< std::string > & Unused()const ;
    private:
        cTplValGesInit< std::string > mUnused;
};
cElXMLTree * ToXMLTree(const cGPU_Correl &);

class cGPU_CorrelBasik
{
    public:
        friend void xml_init(cGPU_CorrelBasik & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Unused();
        const cTplValGesInit< std::string > & Unused()const ;
    private:
        cTplValGesInit< std::string > mUnused;
};
cElXMLTree * ToXMLTree(const cGPU_CorrelBasik &);

class cMultiCorrelPonctuel
{
    public:
        friend void xml_init(cMultiCorrelPonctuel & anObj,cElXMLTree * aTree);


        double & PdsCorrelStd();
        const double & PdsCorrelStd()const ;

        double & PdsCorrelPonct();
        const double & PdsCorrelPonct()const ;

        cTplValGesInit< double > & DefCost();
        const cTplValGesInit< double > & DefCost()const ;

        cTplValGesInit< std::string > & UnUsedTest();
        const cTplValGesInit< std::string > & UnUsedTest()const ;
    private:
        double mPdsCorrelStd;
        double mPdsCorrelPonct;
        cTplValGesInit< double > mDefCost;
        cTplValGesInit< std::string > mUnUsedTest;
};
cElXMLTree * ToXMLTree(const cMultiCorrelPonctuel &);

class cCorrel_Ponctuel2ImGeomI
{
    public:
        friend void xml_init(cCorrel_Ponctuel2ImGeomI & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & RatioI1I2();
        const cTplValGesInit< double > & RatioI1I2()const ;
    private:
        cTplValGesInit< double > mRatioI1I2;
};
cElXMLTree * ToXMLTree(const cCorrel_Ponctuel2ImGeomI &);

class cCorrel_PonctuelleCroisee
{
    public:
        friend void xml_init(cCorrel_PonctuelleCroisee & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & RatioI1I2();
        const cTplValGesInit< double > & RatioI1I2()const ;

        double & PdsPonctuel();
        const double & PdsPonctuel()const ;

        double & PdsCroisee();
        const double & PdsCroisee()const ;
    private:
        cTplValGesInit< double > mRatioI1I2;
        double mPdsPonctuel;
        double mPdsCroisee;
};
cElXMLTree * ToXMLTree(const cCorrel_PonctuelleCroisee &);

class cCorrel_MultiFen
{
    public:
        friend void xml_init(cCorrel_MultiFen & anObj,cElXMLTree * aTree);


        int & NbFen();
        const int & NbFen()const ;
    private:
        int mNbFen;
};
cElXMLTree * ToXMLTree(const cCorrel_MultiFen &);

class cCorrel_Correl_MNE_ZPredic
{
    public:
        friend void xml_init(cCorrel_Correl_MNE_ZPredic & anObj,cElXMLTree * aTree);


        double & SeuilDZ();
        const double & SeuilDZ()const ;
    private:
        double mSeuilDZ;
};
cElXMLTree * ToXMLTree(const cCorrel_Correl_MNE_ZPredic &);

class cCorrel_NC_Robuste
{
    public:
        friend void xml_init(cCorrel_NC_Robuste & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Unused();
        const cTplValGesInit< std::string > & Unused()const ;
    private:
        cTplValGesInit< std::string > mUnused;
};
cElXMLTree * ToXMLTree(const cCorrel_NC_Robuste &);

class cTypeCAH
{
    public:
        friend void xml_init(cTypeCAH & anObj,cElXMLTree * aTree);


        cTplValGesInit< cGPU_Correl > & GPU_Correl();
        const cTplValGesInit< cGPU_Correl > & GPU_Correl()const ;

        cTplValGesInit< cGPU_CorrelBasik > & GPU_CorrelBasik();
        const cTplValGesInit< cGPU_CorrelBasik > & GPU_CorrelBasik()const ;

        cTplValGesInit< cMultiCorrelPonctuel > & MultiCorrelPonctuel();
        const cTplValGesInit< cMultiCorrelPonctuel > & MultiCorrelPonctuel()const ;

        cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & Correl_Ponctuel2ImGeomI();
        const cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & Correl_Ponctuel2ImGeomI()const ;

        cTplValGesInit< cCorrel_PonctuelleCroisee > & Correl_PonctuelleCroisee();
        const cTplValGesInit< cCorrel_PonctuelleCroisee > & Correl_PonctuelleCroisee()const ;

        cTplValGesInit< cCorrel_MultiFen > & Correl_MultiFen();
        const cTplValGesInit< cCorrel_MultiFen > & Correl_MultiFen()const ;

        cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & Correl_Correl_MNE_ZPredic();
        const cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & Correl_Correl_MNE_ZPredic()const ;

        cTplValGesInit< cCorrel_NC_Robuste > & Correl_NC_Robuste();
        const cTplValGesInit< cCorrel_NC_Robuste > & Correl_NC_Robuste()const ;
    private:
        cTplValGesInit< cGPU_Correl > mGPU_Correl;
        cTplValGesInit< cGPU_CorrelBasik > mGPU_CorrelBasik;
        cTplValGesInit< cMultiCorrelPonctuel > mMultiCorrelPonctuel;
        cTplValGesInit< cCorrel_Ponctuel2ImGeomI > mCorrel_Ponctuel2ImGeomI;
        cTplValGesInit< cCorrel_PonctuelleCroisee > mCorrel_PonctuelleCroisee;
        cTplValGesInit< cCorrel_MultiFen > mCorrel_MultiFen;
        cTplValGesInit< cCorrel_Correl_MNE_ZPredic > mCorrel_Correl_MNE_ZPredic;
        cTplValGesInit< cCorrel_NC_Robuste > mCorrel_NC_Robuste;
};
cElXMLTree * ToXMLTree(const cTypeCAH &);

class cCorrelAdHoc
{
    public:
        friend void xml_init(cCorrelAdHoc & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & EpsilonAddMoyenne();
        const cTplValGesInit< double > & EpsilonAddMoyenne()const ;

        cTplValGesInit< double > & EpsilonMulMoyenne();
        const cTplValGesInit< double > & EpsilonMulMoyenne()const ;

        cTplValGesInit< int > & SzBlocAH();
        const cTplValGesInit< int > & SzBlocAH()const ;

        cTplValGesInit< cGPU_Correl > & GPU_Correl();
        const cTplValGesInit< cGPU_Correl > & GPU_Correl()const ;

        cTplValGesInit< cGPU_CorrelBasik > & GPU_CorrelBasik();
        const cTplValGesInit< cGPU_CorrelBasik > & GPU_CorrelBasik()const ;

        cTplValGesInit< cMultiCorrelPonctuel > & MultiCorrelPonctuel();
        const cTplValGesInit< cMultiCorrelPonctuel > & MultiCorrelPonctuel()const ;

        cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & Correl_Ponctuel2ImGeomI();
        const cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & Correl_Ponctuel2ImGeomI()const ;

        cTplValGesInit< cCorrel_PonctuelleCroisee > & Correl_PonctuelleCroisee();
        const cTplValGesInit< cCorrel_PonctuelleCroisee > & Correl_PonctuelleCroisee()const ;

        cTplValGesInit< cCorrel_MultiFen > & Correl_MultiFen();
        const cTplValGesInit< cCorrel_MultiFen > & Correl_MultiFen()const ;

        cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & Correl_Correl_MNE_ZPredic();
        const cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & Correl_Correl_MNE_ZPredic()const ;

        cTplValGesInit< cCorrel_NC_Robuste > & Correl_NC_Robuste();
        const cTplValGesInit< cCorrel_NC_Robuste > & Correl_NC_Robuste()const ;

        cTypeCAH & TypeCAH();
        const cTypeCAH & TypeCAH()const ;
    private:
        cTplValGesInit< double > mEpsilonAddMoyenne;
        cTplValGesInit< double > mEpsilonMulMoyenne;
        cTplValGesInit< int > mSzBlocAH;
        cTypeCAH mTypeCAH;
};
cElXMLTree * ToXMLTree(const cCorrelAdHoc &);

class cDoStatResult
{
    public:
        friend void xml_init(cDoStatResult & anObj,cElXMLTree * aTree);


        bool & DoRatio2Im();
        const bool & DoRatio2Im()const ;
    private:
        bool mDoRatio2Im;
};
cElXMLTree * ToXMLTree(const cDoStatResult &);

class cMasqOfEtape
{
    public:
        friend void xml_init(cMasqOfEtape & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & PatternApply();
        const cElRegex_Ptr & PatternApply()const ;

        cTplValGesInit< Box2dr > & RectInclus();
        const cTplValGesInit< Box2dr > & RectInclus()const ;
    private:
        cElRegex_Ptr mPatternApply;
        cTplValGesInit< Box2dr > mRectInclus;
};
cElXMLTree * ToXMLTree(const cMasqOfEtape &);

class cEtapeProgDyn
{
    public:
        friend void xml_init(cEtapeProgDyn & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::vector<double> > & Px1MultRegul();
        const cTplValGesInit< std::vector<double> > & Px1MultRegul()const ;

        cTplValGesInit< std::vector<double> > & Px2MultRegul();
        const cTplValGesInit< std::vector<double> > & Px2MultRegul()const ;

        cTplValGesInit< int > & NbDir();
        const cTplValGesInit< int > & NbDir()const ;

        eModeAggregProgDyn & ModeAgreg();
        const eModeAggregProgDyn & ModeAgreg()const ;

        cTplValGesInit< double > & Teta0();
        const cTplValGesInit< double > & Teta0()const ;
    private:
        cTplValGesInit< std::vector<double> > mPx1MultRegul;
        cTplValGesInit< std::vector<double> > mPx2MultRegul;
        cTplValGesInit< int > mNbDir;
        eModeAggregProgDyn mModeAgreg;
        cTplValGesInit< double > mTeta0;
};
cElXMLTree * ToXMLTree(const cEtapeProgDyn &);

class cEtiqBestImage
{
    public:
        friend void xml_init(cEtiqBestImage & anObj,cElXMLTree * aTree);


        double & CostChangeEtiq();
        const double & CostChangeEtiq()const ;

        cTplValGesInit< bool > & SauvEtiq();
        const cTplValGesInit< bool > & SauvEtiq()const ;
    private:
        double mCostChangeEtiq;
        cTplValGesInit< bool > mSauvEtiq;
};
cElXMLTree * ToXMLTree(const cEtiqBestImage &);

class cArgMaskAuto
{
    public:
        friend void xml_init(cArgMaskAuto & anObj,cElXMLTree * aTree);


        double & ValDefCorrel();
        const double & ValDefCorrel()const ;

        double & CostTrans();
        const double & CostTrans()const ;

        cTplValGesInit< bool > & ReInjectMask();
        const cTplValGesInit< bool > & ReInjectMask()const ;

        cTplValGesInit< double > & AmplKLPostTr();
        const cTplValGesInit< double > & AmplKLPostTr()const ;

        cTplValGesInit< int > & Erod32Mask();
        const cTplValGesInit< int > & Erod32Mask()const ;

        cTplValGesInit< int > & SzOpen32();
        const cTplValGesInit< int > & SzOpen32()const ;

        cTplValGesInit< int > & SeuilZC();
        const cTplValGesInit< int > & SeuilZC()const ;

        double & CostChangeEtiq();
        const double & CostChangeEtiq()const ;

        cTplValGesInit< bool > & SauvEtiq();
        const cTplValGesInit< bool > & SauvEtiq()const ;

        cTplValGesInit< cEtiqBestImage > & EtiqBestImage();
        const cTplValGesInit< cEtiqBestImage > & EtiqBestImage()const ;
    private:
        double mValDefCorrel;
        double mCostTrans;
        cTplValGesInit< bool > mReInjectMask;
        cTplValGesInit< double > mAmplKLPostTr;
        cTplValGesInit< int > mErod32Mask;
        cTplValGesInit< int > mSzOpen32;
        cTplValGesInit< int > mSeuilZC;
        cTplValGesInit< cEtiqBestImage > mEtiqBestImage;
};
cElXMLTree * ToXMLTree(const cArgMaskAuto &);

class cModulationProgDyn
{
    public:
        friend void xml_init(cModulationProgDyn & anObj,cElXMLTree * aTree);


        std::list< cEtapeProgDyn > & EtapeProgDyn();
        const std::list< cEtapeProgDyn > & EtapeProgDyn()const ;

        cTplValGesInit< double > & Px1PenteMax();
        const cTplValGesInit< double > & Px1PenteMax()const ;

        cTplValGesInit< double > & Px2PenteMax();
        const cTplValGesInit< double > & Px2PenteMax()const ;

        cTplValGesInit< bool > & ChoixNewProg();
        const cTplValGesInit< bool > & ChoixNewProg()const ;

        double & ValDefCorrel();
        const double & ValDefCorrel()const ;

        double & CostTrans();
        const double & CostTrans()const ;

        cTplValGesInit< bool > & ReInjectMask();
        const cTplValGesInit< bool > & ReInjectMask()const ;

        cTplValGesInit< double > & AmplKLPostTr();
        const cTplValGesInit< double > & AmplKLPostTr()const ;

        cTplValGesInit< int > & Erod32Mask();
        const cTplValGesInit< int > & Erod32Mask()const ;

        cTplValGesInit< int > & SzOpen32();
        const cTplValGesInit< int > & SzOpen32()const ;

        cTplValGesInit< int > & SeuilZC();
        const cTplValGesInit< int > & SeuilZC()const ;

        double & CostChangeEtiq();
        const double & CostChangeEtiq()const ;

        cTplValGesInit< bool > & SauvEtiq();
        const cTplValGesInit< bool > & SauvEtiq()const ;

        cTplValGesInit< cEtiqBestImage > & EtiqBestImage();
        const cTplValGesInit< cEtiqBestImage > & EtiqBestImage()const ;

        cTplValGesInit< cArgMaskAuto > & ArgMaskAuto();
        const cTplValGesInit< cArgMaskAuto > & ArgMaskAuto()const ;
    private:
        std::list< cEtapeProgDyn > mEtapeProgDyn;
        cTplValGesInit< double > mPx1PenteMax;
        cTplValGesInit< double > mPx2PenteMax;
        cTplValGesInit< bool > mChoixNewProg;
        cTplValGesInit< cArgMaskAuto > mArgMaskAuto;
};
cElXMLTree * ToXMLTree(const cModulationProgDyn &);

class cPostFiltragePx
{
    public:
        friend void xml_init(cPostFiltragePx & anObj,cElXMLTree * aTree);


        std::list< cSpecFitrageImage > & OneFitragePx();
        const std::list< cSpecFitrageImage > & OneFitragePx()const ;
    private:
        std::list< cSpecFitrageImage > mOneFitragePx;
};
cElXMLTree * ToXMLTree(const cPostFiltragePx &);

class cPostFiltrageDiscont
{
    public:
        friend void xml_init(cPostFiltrageDiscont & anObj,cElXMLTree * aTree);


        double & SzFiltre();
        const double & SzFiltre()const ;

        cTplValGesInit< int > & NbIter();
        const cTplValGesInit< int > & NbIter()const ;

        cTplValGesInit< double > & ExposPonderGrad();
        const cTplValGesInit< double > & ExposPonderGrad()const ;

        cTplValGesInit< double > & DericheFactEPC();
        const cTplValGesInit< double > & DericheFactEPC()const ;

        cTplValGesInit< double > & ValGradAtten();
        const cTplValGesInit< double > & ValGradAtten()const ;

        cTplValGesInit< double > & ExposPonderCorr();
        const cTplValGesInit< double > & ExposPonderCorr()const ;
    private:
        double mSzFiltre;
        cTplValGesInit< int > mNbIter;
        cTplValGesInit< double > mExposPonderGrad;
        cTplValGesInit< double > mDericheFactEPC;
        cTplValGesInit< double > mValGradAtten;
        cTplValGesInit< double > mExposPonderCorr;
};
cElXMLTree * ToXMLTree(const cPostFiltrageDiscont &);

class cImageSelecteur
{
    public:
        friend void xml_init(cImageSelecteur & anObj,cElXMLTree * aTree);


        bool & ModeExclusion();
        const bool & ModeExclusion()const ;

        std::list< std::string > & PatternSel();
        const std::list< std::string > & PatternSel()const ;
    private:
        bool mModeExclusion;
        std::list< std::string > mPatternSel;
};
cElXMLTree * ToXMLTree(const cImageSelecteur &);

class cGenerateProjectionInImages
{
    public:
        friend void xml_init(cGenerateProjectionInImages & anObj,cElXMLTree * aTree);


        std::list< int > & NumsImageDontApply();
        const std::list< int > & NumsImageDontApply()const ;

        std::string & FCND_CalcProj();
        const std::string & FCND_CalcProj()const ;

        cTplValGesInit< bool > & SubsXY();
        const cTplValGesInit< bool > & SubsXY()const ;
    private:
        std::list< int > mNumsImageDontApply;
        std::string mFCND_CalcProj;
        cTplValGesInit< bool > mSubsXY;
};
cElXMLTree * ToXMLTree(const cGenerateProjectionInImages &);

class cGenCorPxTransv
{
    public:
        friend void xml_init(cGenCorPxTransv & anObj,cElXMLTree * aTree);


        double & SsResolPx();
        const double & SsResolPx()const ;

        std::string & NameXMLFile();
        const std::string & NameXMLFile()const ;
    private:
        double mSsResolPx;
        std::string mNameXMLFile;
};
cElXMLTree * ToXMLTree(const cGenCorPxTransv &);

class cSimulFrac
{
    public:
        friend void xml_init(cSimulFrac & anObj,cElXMLTree * aTree);


        double & CoutFrac();
        const double & CoutFrac()const ;
    private:
        double mCoutFrac;
};
cElXMLTree * ToXMLTree(const cSimulFrac &);

class cInterfaceVisualisation
{
    public:
        friend void xml_init(cInterfaceVisualisation & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & VisuTerrainIm();
        const cTplValGesInit< bool > & VisuTerrainIm()const ;

        cTplValGesInit< int > & SzWTerr();
        const cTplValGesInit< int > & SzWTerr()const ;

        std::list< std::string > & UnSelectedImage();
        const std::list< std::string > & UnSelectedImage()const ;

        Pt2di & CentreVisuTerrain();
        const Pt2di & CentreVisuTerrain()const ;

        cTplValGesInit< int > & ZoomTerr();
        const cTplValGesInit< int > & ZoomTerr()const ;

        cTplValGesInit< int > & NbDiscHistoPartieFrac();
        const cTplValGesInit< int > & NbDiscHistoPartieFrac()const ;

        double & CoutFrac();
        const double & CoutFrac()const ;

        cTplValGesInit< cSimulFrac > & SimulFrac();
        const cTplValGesInit< cSimulFrac > & SimulFrac()const ;
    private:
        cTplValGesInit< bool > mVisuTerrainIm;
        cTplValGesInit< int > mSzWTerr;
        std::list< std::string > mUnSelectedImage;
        Pt2di mCentreVisuTerrain;
        cTplValGesInit< int > mZoomTerr;
        cTplValGesInit< int > mNbDiscHistoPartieFrac;
        cTplValGesInit< cSimulFrac > mSimulFrac;
};
cElXMLTree * ToXMLTree(const cInterfaceVisualisation &);

class cMTD_Nuage_Maille
{
    public:
        friend void xml_init(cMTD_Nuage_Maille & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DataInside();
        const cTplValGesInit< bool > & DataInside()const ;

        std::string & KeyNameMTD();
        const std::string & KeyNameMTD()const ;

        cTplValGesInit< double > & RatioPseudoConik();
        const cTplValGesInit< double > & RatioPseudoConik()const ;
    private:
        cTplValGesInit< bool > mDataInside;
        std::string mKeyNameMTD;
        cTplValGesInit< double > mRatioPseudoConik;
};
cElXMLTree * ToXMLTree(const cMTD_Nuage_Maille &);

class cCannauxExportPly
{
    public:
        friend void xml_init(cCannauxExportPly & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        std::vector< std::string > & NamesProperty();
        const std::vector< std::string > & NamesProperty()const ;

        cTplValGesInit< int > & FlagUse();
        const cTplValGesInit< int > & FlagUse()const ;
    private:
        std::string mNameIm;
        std::vector< std::string > mNamesProperty;
        cTplValGesInit< int > mFlagUse;
};
cElXMLTree * ToXMLTree(const cCannauxExportPly &);

class cPlyFile
{
    public:
        friend void xml_init(cPlyFile & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & KeyNamePly();
        const cTplValGesInit< std::string > & KeyNamePly()const ;

        bool & Binary();
        const bool & Binary()const ;

        double & Resolution();
        const double & Resolution()const ;

        std::list< std::string > & PlyCommentAdd();
        const std::list< std::string > & PlyCommentAdd()const ;

        std::list< cCannauxExportPly > & CannauxExportPly();
        const std::list< cCannauxExportPly > & CannauxExportPly()const ;
    private:
        cTplValGesInit< std::string > mKeyNamePly;
        bool mBinary;
        double mResolution;
        std::list< std::string > mPlyCommentAdd;
        std::list< cCannauxExportPly > mCannauxExportPly;
};
cElXMLTree * ToXMLTree(const cPlyFile &);

class cExportNuage
{
    public:
        friend void xml_init(cExportNuage & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DataInside();
        const cTplValGesInit< bool > & DataInside()const ;

        std::string & KeyNameMTD();
        const std::string & KeyNameMTD()const ;

        cTplValGesInit< double > & RatioPseudoConik();
        const cTplValGesInit< double > & RatioPseudoConik()const ;

        cTplValGesInit< cMTD_Nuage_Maille > & MTD_Nuage_Maille();
        const cTplValGesInit< cMTD_Nuage_Maille > & MTD_Nuage_Maille()const ;

        cTplValGesInit< std::string > & KeyNamePly();
        const cTplValGesInit< std::string > & KeyNamePly()const ;

        bool & Binary();
        const bool & Binary()const ;

        double & Resolution();
        const double & Resolution()const ;

        std::list< std::string > & PlyCommentAdd();
        const std::list< std::string > & PlyCommentAdd()const ;

        std::list< cCannauxExportPly > & CannauxExportPly();
        const std::list< cCannauxExportPly > & CannauxExportPly()const ;

        cTplValGesInit< cPlyFile > & PlyFile();
        const cTplValGesInit< cPlyFile > & PlyFile()const ;
    private:
        cTplValGesInit< cMTD_Nuage_Maille > mMTD_Nuage_Maille;
        cTplValGesInit< cPlyFile > mPlyFile;
};
cElXMLTree * ToXMLTree(const cExportNuage &);

class cReCalclCorrelMultiEchelle
{
    public:
        friend void xml_init(cReCalclCorrelMultiEchelle & anObj,cElXMLTree * aTree);


        bool & UseIt();
        const bool & UseIt()const ;

        std::list< Pt2di > & ScaleSzW();
        const std::list< Pt2di > & ScaleSzW()const ;

        cTplValGesInit< bool > & AgregMin();
        const cTplValGesInit< bool > & AgregMin()const ;

        cTplValGesInit< bool > & DoImg();
        const cTplValGesInit< bool > & DoImg()const ;

        double & Seuil();
        const double & Seuil()const ;
    private:
        bool mUseIt;
        std::list< Pt2di > mScaleSzW;
        cTplValGesInit< bool > mAgregMin;
        cTplValGesInit< bool > mDoImg;
        double mSeuil;
};
cElXMLTree * ToXMLTree(const cReCalclCorrelMultiEchelle &);

class cOneModeleAnalytique
{
    public:
        friend void xml_init(cOneModeleAnalytique & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        cTplValGesInit< std::string > & KeyNuage3D();
        const cTplValGesInit< std::string > & KeyNuage3D()const ;

        eTypeModeleAnalytique & TypeModele();
        const eTypeModeleAnalytique & TypeModele()const ;

        cTplValGesInit< bool > & HomographieL2();
        const cTplValGesInit< bool > & HomographieL2()const ;

        cTplValGesInit< bool > & PolynomeL2();
        const cTplValGesInit< bool > & PolynomeL2()const ;

        cTplValGesInit< int > & DegrePol();
        const cTplValGesInit< int > & DegrePol()const ;

        std::list< int > & NumsAngleFiges();
        const std::list< int > & NumsAngleFiges()const ;

        cTplValGesInit< bool > & L1CalcOri();
        const cTplValGesInit< bool > & L1CalcOri()const ;

        cTplValGesInit< std::string > & AutomSelExportOri();
        const cTplValGesInit< std::string > & AutomSelExportOri()const ;

        cTplValGesInit< std::string > & AutomNamesExportOri1();
        const cTplValGesInit< std::string > & AutomNamesExportOri1()const ;

        cTplValGesInit< std::string > & AutomNamesExportOri2();
        const cTplValGesInit< std::string > & AutomNamesExportOri2()const ;

        cTplValGesInit< std::string > & AutomNamesExportHomXml();
        const cTplValGesInit< std::string > & AutomNamesExportHomXml()const ;

        cTplValGesInit< std::string > & AutomNamesExportHomTif();
        const cTplValGesInit< std::string > & AutomNamesExportHomTif()const ;

        cTplValGesInit< std::string > & AutomNamesExportHomBin();
        const cTplValGesInit< std::string > & AutomNamesExportHomBin()const ;

        cTplValGesInit< bool > & AffineOrient();
        const cTplValGesInit< bool > & AffineOrient()const ;

        cTplValGesInit< std::string > & KeyNamesExportHomXml();
        const cTplValGesInit< std::string > & KeyNamesExportHomXml()const ;

        cTplValGesInit< double > & SigmaPixPdsExport();
        const cTplValGesInit< double > & SigmaPixPdsExport()const ;

        cTplValGesInit< bool > & FiltreByCorrel();
        const cTplValGesInit< bool > & FiltreByCorrel()const ;

        cTplValGesInit< double > & SeuilFiltreCorrel();
        const cTplValGesInit< double > & SeuilFiltreCorrel()const ;

        cTplValGesInit< bool > & UseFCBySeuil();
        const cTplValGesInit< bool > & UseFCBySeuil()const ;

        cTplValGesInit< double > & ExposantPondereCorrel();
        const cTplValGesInit< double > & ExposantPondereCorrel()const ;

        std::list< cReCalclCorrelMultiEchelle > & ReCalclCorrelMultiEchelle();
        const std::list< cReCalclCorrelMultiEchelle > & ReCalclCorrelMultiEchelle()const ;

        int & PasCalcul();
        const int & PasCalcul()const ;

        cTplValGesInit< bool > & PointUnique();
        const cTplValGesInit< bool > & PointUnique()const ;

        cTplValGesInit< bool > & ReuseModele();
        const cTplValGesInit< bool > & ReuseModele()const ;

        cTplValGesInit< bool > & MakeExport();
        const cTplValGesInit< bool > & MakeExport()const ;

        cTplValGesInit< std::string > & NameExport();
        const cTplValGesInit< std::string > & NameExport()const ;

        cTplValGesInit< bool > & ExportImage();
        const cTplValGesInit< bool > & ExportImage()const ;

        cTplValGesInit< bool > & ReuseResiduelle();
        const cTplValGesInit< bool > & ReuseResiduelle()const ;

        cTplValGesInit< std::string > & FCND_ExportModeleGlobal();
        const cTplValGesInit< std::string > & FCND_ExportModeleGlobal()const ;

        cTplValGesInit< double > & MailleExport();
        const cTplValGesInit< double > & MailleExport()const ;

        cTplValGesInit< bool > & UseHomologueReference();
        const cTplValGesInit< bool > & UseHomologueReference()const ;

        cTplValGesInit< bool > & MakeImagePxRef();
        const cTplValGesInit< bool > & MakeImagePxRef()const ;

        cTplValGesInit< int > & NbPtMinValideEqOriRel();
        const cTplValGesInit< int > & NbPtMinValideEqOriRel()const ;
    private:
        cTplValGesInit< bool > mUseIt;
        cTplValGesInit< std::string > mKeyNuage3D;
        eTypeModeleAnalytique mTypeModele;
        cTplValGesInit< bool > mHomographieL2;
        cTplValGesInit< bool > mPolynomeL2;
        cTplValGesInit< int > mDegrePol;
        std::list< int > mNumsAngleFiges;
        cTplValGesInit< bool > mL1CalcOri;
        cTplValGesInit< std::string > mAutomSelExportOri;
        cTplValGesInit< std::string > mAutomNamesExportOri1;
        cTplValGesInit< std::string > mAutomNamesExportOri2;
        cTplValGesInit< std::string > mAutomNamesExportHomXml;
        cTplValGesInit< std::string > mAutomNamesExportHomTif;
        cTplValGesInit< std::string > mAutomNamesExportHomBin;
        cTplValGesInit< bool > mAffineOrient;
        cTplValGesInit< std::string > mKeyNamesExportHomXml;
        cTplValGesInit< double > mSigmaPixPdsExport;
        cTplValGesInit< bool > mFiltreByCorrel;
        cTplValGesInit< double > mSeuilFiltreCorrel;
        cTplValGesInit< bool > mUseFCBySeuil;
        cTplValGesInit< double > mExposantPondereCorrel;
        std::list< cReCalclCorrelMultiEchelle > mReCalclCorrelMultiEchelle;
        int mPasCalcul;
        cTplValGesInit< bool > mPointUnique;
        cTplValGesInit< bool > mReuseModele;
        cTplValGesInit< bool > mMakeExport;
        cTplValGesInit< std::string > mNameExport;
        cTplValGesInit< bool > mExportImage;
        cTplValGesInit< bool > mReuseResiduelle;
        cTplValGesInit< std::string > mFCND_ExportModeleGlobal;
        cTplValGesInit< double > mMailleExport;
        cTplValGesInit< bool > mUseHomologueReference;
        cTplValGesInit< bool > mMakeImagePxRef;
        cTplValGesInit< int > mNbPtMinValideEqOriRel;
};
cElXMLTree * ToXMLTree(const cOneModeleAnalytique &);

class cModelesAnalytiques
{
    public:
        friend void xml_init(cModelesAnalytiques & anObj,cElXMLTree * aTree);


        std::list< cOneModeleAnalytique > & OneModeleAnalytique();
        const std::list< cOneModeleAnalytique > & OneModeleAnalytique()const ;
    private:
        std::list< cOneModeleAnalytique > mOneModeleAnalytique;
};
cElXMLTree * ToXMLTree(const cModelesAnalytiques &);

class cByFileNomChantier
{
    public:
        friend void xml_init(cByFileNomChantier & anObj,cElXMLTree * aTree);


        std::string & Prefixe();
        const std::string & Prefixe()const ;

        cTplValGesInit< bool > & NomChantier();
        const cTplValGesInit< bool > & NomChantier()const ;

        std::string & Postfixe();
        const std::string & Postfixe()const ;

        cTplValGesInit< std::string > & NameTag();
        const cTplValGesInit< std::string > & NameTag()const ;
    private:
        std::string mPrefixe;
        cTplValGesInit< bool > mNomChantier;
        std::string mPostfixe;
        cTplValGesInit< std::string > mNameTag;
};
cElXMLTree * ToXMLTree(const cByFileNomChantier &);

class cOri
{
    public:
        friend void xml_init(cOri & anObj,cElXMLTree * aTree);


        cTplValGesInit< cFileOriMnt > & Explicite();
        const cTplValGesInit< cFileOriMnt > & Explicite()const ;

        std::string & Prefixe();
        const std::string & Prefixe()const ;

        cTplValGesInit< bool > & NomChantier();
        const cTplValGesInit< bool > & NomChantier()const ;

        std::string & Postfixe();
        const std::string & Postfixe()const ;

        cTplValGesInit< std::string > & NameTag();
        const cTplValGesInit< std::string > & NameTag()const ;

        cTplValGesInit< cByFileNomChantier > & ByFileNomChantier();
        const cTplValGesInit< cByFileNomChantier > & ByFileNomChantier()const ;
    private:
        cTplValGesInit< cFileOriMnt > mExplicite;
        cTplValGesInit< cByFileNomChantier > mByFileNomChantier;
};
cElXMLTree * ToXMLTree(const cOri &);

class cBasculeRes
{
    public:
        friend void xml_init(cBasculeRes & anObj,cElXMLTree * aTree);


        cTplValGesInit< cFileOriMnt > & Explicite();
        const cTplValGesInit< cFileOriMnt > & Explicite()const ;

        std::string & Prefixe();
        const std::string & Prefixe()const ;

        cTplValGesInit< bool > & NomChantier();
        const cTplValGesInit< bool > & NomChantier()const ;

        std::string & Postfixe();
        const std::string & Postfixe()const ;

        cTplValGesInit< std::string > & NameTag();
        const cTplValGesInit< std::string > & NameTag()const ;

        cTplValGesInit< cByFileNomChantier > & ByFileNomChantier();
        const cTplValGesInit< cByFileNomChantier > & ByFileNomChantier()const ;

        cOri & Ori();
        const cOri & Ori()const ;

        cTplValGesInit< double > & OutValue();
        const cTplValGesInit< double > & OutValue()const ;
    private:
        cOri mOri;
        cTplValGesInit< double > mOutValue;
};
cElXMLTree * ToXMLTree(const cBasculeRes &);

class cVisuSuperposMNT
{
    public:
        friend void xml_init(cVisuSuperposMNT & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        double & Seuil();
        const double & Seuil()const ;
    private:
        std::string mNameFile;
        double mSeuil;
};
cElXMLTree * ToXMLTree(const cVisuSuperposMNT &);

class cMakeMTDMaskOrtho
{
    public:
        friend void xml_init(cMakeMTDMaskOrtho & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NameFileSauv();
        const cTplValGesInit< std::string > & NameFileSauv()const ;

        cMasqMesures & Mesures();
        const cMasqMesures & Mesures()const ;
    private:
        cTplValGesInit< std::string > mNameFileSauv;
        cMasqMesures mMesures;
};
cElXMLTree * ToXMLTree(const cMakeMTDMaskOrtho &);

class cMakeOrthoParImage
{
    public:
        friend void xml_init(cMakeOrthoParImage & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & DirOrtho();
        const cTplValGesInit< std::string > & DirOrtho()const ;

        cTplValGesInit< std::string > & FileMTD();
        const cTplValGesInit< std::string > & FileMTD()const ;

        cTplValGesInit< std::string > & NameFileSauv();
        const cTplValGesInit< std::string > & NameFileSauv()const ;

        cMasqMesures & Mesures();
        const cMasqMesures & Mesures()const ;

        cTplValGesInit< cMakeMTDMaskOrtho > & MakeMTDMaskOrtho();
        const cTplValGesInit< cMakeMTDMaskOrtho > & MakeMTDMaskOrtho()const ;

        cTplValGesInit< double > & OrthoBiCub();
        const cTplValGesInit< double > & OrthoBiCub()const ;

        cTplValGesInit< double > & ScaleBiCub();
        const cTplValGesInit< double > & ScaleBiCub()const ;

        cTplValGesInit< double > & ResolRelOrhto();
        const cTplValGesInit< double > & ResolRelOrhto()const ;

        cTplValGesInit< double > & ResolAbsOrtho();
        const cTplValGesInit< double > & ResolAbsOrtho()const ;

        cTplValGesInit< Pt2dr > & PixelTerrainPhase();
        const cTplValGesInit< Pt2dr > & PixelTerrainPhase()const ;

        std::string & KeyCalcInput();
        const std::string & KeyCalcInput()const ;

        std::string & KeyCalcOutput();
        const std::string & KeyCalcOutput()const ;

        cTplValGesInit< int > & NbChan();
        const cTplValGesInit< int > & NbChan()const ;

        cTplValGesInit< std::string > & KeyCalcIncidHor();
        const cTplValGesInit< std::string > & KeyCalcIncidHor()const ;

        cTplValGesInit< double > & SsResolIncH();
        const cTplValGesInit< double > & SsResolIncH()const ;

        cTplValGesInit< bool > & CalcIncAZMoy();
        const cTplValGesInit< bool > & CalcIncAZMoy()const ;

        cTplValGesInit< bool > & ImageIncIsDistFront();
        const cTplValGesInit< bool > & ImageIncIsDistFront()const ;

        cTplValGesInit< int > & RepulsFront();
        const cTplValGesInit< int > & RepulsFront()const ;

        cTplValGesInit< double > & ResolIm();
        const cTplValGesInit< double > & ResolIm()const ;

        cTplValGesInit< Pt2di > & TranslateIm();
        const cTplValGesInit< Pt2di > & TranslateIm()const ;
    private:
        cTplValGesInit< std::string > mDirOrtho;
        cTplValGesInit< std::string > mFileMTD;
        cTplValGesInit< cMakeMTDMaskOrtho > mMakeMTDMaskOrtho;
        cTplValGesInit< double > mOrthoBiCub;
        cTplValGesInit< double > mScaleBiCub;
        cTplValGesInit< double > mResolRelOrhto;
        cTplValGesInit< double > mResolAbsOrtho;
        cTplValGesInit< Pt2dr > mPixelTerrainPhase;
        std::string mKeyCalcInput;
        std::string mKeyCalcOutput;
        cTplValGesInit< int > mNbChan;
        cTplValGesInit< std::string > mKeyCalcIncidHor;
        cTplValGesInit< double > mSsResolIncH;
        cTplValGesInit< bool > mCalcIncAZMoy;
        cTplValGesInit< bool > mImageIncIsDistFront;
        cTplValGesInit< int > mRepulsFront;
        cTplValGesInit< double > mResolIm;
        cTplValGesInit< Pt2di > mTranslateIm;
};
cElXMLTree * ToXMLTree(const cMakeOrthoParImage &);

class cGenerePartiesCachees
{
    public:
        friend void xml_init(cGenerePartiesCachees & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        cTplValGesInit< double > & PasDisc();
        const cTplValGesInit< double > & PasDisc()const ;

        double & SeuilUsePC();
        const double & SeuilUsePC()const ;

        cTplValGesInit< std::string > & KeyCalcPC();
        const cTplValGesInit< std::string > & KeyCalcPC()const ;

        cTplValGesInit< bool > & AddChantierKPC();
        const cTplValGesInit< bool > & AddChantierKPC()const ;

        cTplValGesInit< bool > & SupresExtChantierKPC();
        const cTplValGesInit< bool > & SupresExtChantierKPC()const ;

        cTplValGesInit< bool > & Dequant();
        const cTplValGesInit< bool > & Dequant()const ;

        cTplValGesInit< bool > & ByMkF();
        const cTplValGesInit< bool > & ByMkF()const ;

        cTplValGesInit< std::string > & PatternApply();
        const cTplValGesInit< std::string > & PatternApply()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        double & Seuil();
        const double & Seuil()const ;

        cTplValGesInit< cVisuSuperposMNT > & VisuSuperposMNT();
        const cTplValGesInit< cVisuSuperposMNT > & VisuSuperposMNT()const ;

        cTplValGesInit< bool > & BufXYZ();
        const cTplValGesInit< bool > & BufXYZ()const ;

        cTplValGesInit< bool > & DoOnlyWhenNew();
        const cTplValGesInit< bool > & DoOnlyWhenNew()const ;

        cTplValGesInit< int > & SzBloc();
        const cTplValGesInit< int > & SzBloc()const ;

        cTplValGesInit< int > & SzBord();
        const cTplValGesInit< int > & SzBord()const ;

        cTplValGesInit< bool > & ImSuperpMNT();
        const cTplValGesInit< bool > & ImSuperpMNT()const ;

        cTplValGesInit< double > & ZMoy();
        const cTplValGesInit< double > & ZMoy()const ;

        cTplValGesInit< cElRegex_Ptr > & FiltreName();
        const cTplValGesInit< cElRegex_Ptr > & FiltreName()const ;

        cTplValGesInit< std::string > & DirOrtho();
        const cTplValGesInit< std::string > & DirOrtho()const ;

        cTplValGesInit< std::string > & FileMTD();
        const cTplValGesInit< std::string > & FileMTD()const ;

        cTplValGesInit< std::string > & NameFileSauv();
        const cTplValGesInit< std::string > & NameFileSauv()const ;

        cMasqMesures & Mesures();
        const cMasqMesures & Mesures()const ;

        cTplValGesInit< cMakeMTDMaskOrtho > & MakeMTDMaskOrtho();
        const cTplValGesInit< cMakeMTDMaskOrtho > & MakeMTDMaskOrtho()const ;

        cTplValGesInit< double > & OrthoBiCub();
        const cTplValGesInit< double > & OrthoBiCub()const ;

        cTplValGesInit< double > & ScaleBiCub();
        const cTplValGesInit< double > & ScaleBiCub()const ;

        cTplValGesInit< double > & ResolRelOrhto();
        const cTplValGesInit< double > & ResolRelOrhto()const ;

        cTplValGesInit< double > & ResolAbsOrtho();
        const cTplValGesInit< double > & ResolAbsOrtho()const ;

        cTplValGesInit< Pt2dr > & PixelTerrainPhase();
        const cTplValGesInit< Pt2dr > & PixelTerrainPhase()const ;

        std::string & KeyCalcInput();
        const std::string & KeyCalcInput()const ;

        std::string & KeyCalcOutput();
        const std::string & KeyCalcOutput()const ;

        cTplValGesInit< int > & NbChan();
        const cTplValGesInit< int > & NbChan()const ;

        cTplValGesInit< std::string > & KeyCalcIncidHor();
        const cTplValGesInit< std::string > & KeyCalcIncidHor()const ;

        cTplValGesInit< double > & SsResolIncH();
        const cTplValGesInit< double > & SsResolIncH()const ;

        cTplValGesInit< bool > & CalcIncAZMoy();
        const cTplValGesInit< bool > & CalcIncAZMoy()const ;

        cTplValGesInit< bool > & ImageIncIsDistFront();
        const cTplValGesInit< bool > & ImageIncIsDistFront()const ;

        cTplValGesInit< int > & RepulsFront();
        const cTplValGesInit< int > & RepulsFront()const ;

        cTplValGesInit< double > & ResolIm();
        const cTplValGesInit< double > & ResolIm()const ;

        cTplValGesInit< Pt2di > & TranslateIm();
        const cTplValGesInit< Pt2di > & TranslateIm()const ;

        cTplValGesInit< cMakeOrthoParImage > & MakeOrthoParImage();
        const cTplValGesInit< cMakeOrthoParImage > & MakeOrthoParImage()const ;
    private:
        cTplValGesInit< bool > mUseIt;
        cTplValGesInit< double > mPasDisc;
        double mSeuilUsePC;
        cTplValGesInit< std::string > mKeyCalcPC;
        cTplValGesInit< bool > mAddChantierKPC;
        cTplValGesInit< bool > mSupresExtChantierKPC;
        cTplValGesInit< bool > mDequant;
        cTplValGesInit< bool > mByMkF;
        cTplValGesInit< std::string > mPatternApply;
        cTplValGesInit< cVisuSuperposMNT > mVisuSuperposMNT;
        cTplValGesInit< bool > mBufXYZ;
        cTplValGesInit< bool > mDoOnlyWhenNew;
        cTplValGesInit< int > mSzBloc;
        cTplValGesInit< int > mSzBord;
        cTplValGesInit< bool > mImSuperpMNT;
        cTplValGesInit< double > mZMoy;
        cTplValGesInit< cElRegex_Ptr > mFiltreName;
        cTplValGesInit< cMakeOrthoParImage > mMakeOrthoParImage;
};
cElXMLTree * ToXMLTree(const cGenerePartiesCachees &);

class cRedrLocAnam
{
    public:
        friend void xml_init(cRedrLocAnam & anObj,cElXMLTree * aTree);


        std::string & NameOut();
        const std::string & NameOut()const ;

        std::string & NameMasq();
        const std::string & NameMasq()const ;

        std::string & NameOriGlob();
        const std::string & NameOriGlob()const ;

        cTplValGesInit< int > & XRecouvrt();
        const cTplValGesInit< int > & XRecouvrt()const ;

        cTplValGesInit< double > & MemAvalaible();
        const cTplValGesInit< double > & MemAvalaible()const ;

        cTplValGesInit< double > & FilterMulLargY();
        const cTplValGesInit< double > & FilterMulLargY()const ;

        cTplValGesInit< double > & NbIterFilterY();
        const cTplValGesInit< double > & NbIterFilterY()const ;

        cTplValGesInit< int > & FilterXY();
        const cTplValGesInit< int > & FilterXY()const ;

        cTplValGesInit< int > & NbIterXY();
        const cTplValGesInit< int > & NbIterXY()const ;

        cTplValGesInit< double > & DensityHighThresh();
        const cTplValGesInit< double > & DensityHighThresh()const ;

        cTplValGesInit< double > & DensityLowThresh();
        const cTplValGesInit< double > & DensityLowThresh()const ;

        cTplValGesInit< bool > & UseAutoMask();
        const cTplValGesInit< bool > & UseAutoMask()const ;
    private:
        std::string mNameOut;
        std::string mNameMasq;
        std::string mNameOriGlob;
        cTplValGesInit< int > mXRecouvrt;
        cTplValGesInit< double > mMemAvalaible;
        cTplValGesInit< double > mFilterMulLargY;
        cTplValGesInit< double > mNbIterFilterY;
        cTplValGesInit< int > mFilterXY;
        cTplValGesInit< int > mNbIterXY;
        cTplValGesInit< double > mDensityHighThresh;
        cTplValGesInit< double > mDensityLowThresh;
        cTplValGesInit< bool > mUseAutoMask;
};
cElXMLTree * ToXMLTree(const cRedrLocAnam &);

class cNuagePredicteur
{
    public:
        friend void xml_init(cNuagePredicteur & anObj,cElXMLTree * aTree);


        std::string & KeyAssocIm2Nuage();
        const std::string & KeyAssocIm2Nuage()const ;

        cTplValGesInit< std::string > & Selector();
        const cTplValGesInit< std::string > & Selector()const ;

        double & ScaleNuage();
        const double & ScaleNuage()const ;
    private:
        std::string mKeyAssocIm2Nuage;
        cTplValGesInit< std::string > mSelector;
        double mScaleNuage;
};
cElXMLTree * ToXMLTree(const cNuagePredicteur &);

class cEtapeMEC
{
    public:
        friend void xml_init(cEtapeMEC & anObj,cElXMLTree * aTree);


        int & DeZoom();
        const int & DeZoom()const ;

        cTplValGesInit< double > & EpsilonAddMoyenne();
        const cTplValGesInit< double > & EpsilonAddMoyenne()const ;

        cTplValGesInit< double > & EpsilonMulMoyenne();
        const cTplValGesInit< double > & EpsilonMulMoyenne()const ;

        cTplValGesInit< int > & SzBlocAH();
        const cTplValGesInit< int > & SzBlocAH()const ;

        cTplValGesInit< cGPU_Correl > & GPU_Correl();
        const cTplValGesInit< cGPU_Correl > & GPU_Correl()const ;

        cTplValGesInit< cGPU_CorrelBasik > & GPU_CorrelBasik();
        const cTplValGesInit< cGPU_CorrelBasik > & GPU_CorrelBasik()const ;

        cTplValGesInit< cMultiCorrelPonctuel > & MultiCorrelPonctuel();
        const cTplValGesInit< cMultiCorrelPonctuel > & MultiCorrelPonctuel()const ;

        cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & Correl_Ponctuel2ImGeomI();
        const cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & Correl_Ponctuel2ImGeomI()const ;

        cTplValGesInit< cCorrel_PonctuelleCroisee > & Correl_PonctuelleCroisee();
        const cTplValGesInit< cCorrel_PonctuelleCroisee > & Correl_PonctuelleCroisee()const ;

        cTplValGesInit< cCorrel_MultiFen > & Correl_MultiFen();
        const cTplValGesInit< cCorrel_MultiFen > & Correl_MultiFen()const ;

        cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & Correl_Correl_MNE_ZPredic();
        const cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & Correl_Correl_MNE_ZPredic()const ;

        cTplValGesInit< cCorrel_NC_Robuste > & Correl_NC_Robuste();
        const cTplValGesInit< cCorrel_NC_Robuste > & Correl_NC_Robuste()const ;

        cTypeCAH & TypeCAH();
        const cTypeCAH & TypeCAH()const ;

        cTplValGesInit< cCorrelAdHoc > & CorrelAdHoc();
        const cTplValGesInit< cCorrelAdHoc > & CorrelAdHoc()const ;

        bool & DoRatio2Im();
        const bool & DoRatio2Im()const ;

        cTplValGesInit< cDoStatResult > & DoStatResult();
        const cTplValGesInit< cDoStatResult > & DoStatResult()const ;

        std::list< cMasqOfEtape > & MasqOfEtape();
        const std::list< cMasqOfEtape > & MasqOfEtape()const ;

        cTplValGesInit< int > & SzRecouvrtDalles();
        const cTplValGesInit< int > & SzRecouvrtDalles()const ;

        cTplValGesInit< int > & SzDalleMin();
        const cTplValGesInit< int > & SzDalleMin()const ;

        cTplValGesInit< int > & SzDalleMax();
        const cTplValGesInit< int > & SzDalleMax()const ;

        cTplValGesInit< eModeDynamiqueCorrel > & DynamiqueCorrel();
        const cTplValGesInit< eModeDynamiqueCorrel > & DynamiqueCorrel()const ;

        cTplValGesInit< eModeAggregCorr > & AggregCorr();
        const cTplValGesInit< eModeAggregCorr > & AggregCorr()const ;

        cTplValGesInit< double > & SzW();
        const cTplValGesInit< double > & SzW()const ;

        cTplValGesInit< bool > & WSpecUseMasqGlob();
        const cTplValGesInit< bool > & WSpecUseMasqGlob()const ;

        cTplValGesInit< eTypeWinCorrel > & TypeWCorr();
        const cTplValGesInit< eTypeWinCorrel > & TypeWCorr()const ;

        cTplValGesInit< double > & SzWy();
        const cTplValGesInit< double > & SzWy()const ;

        cTplValGesInit< int > & NbIterFenSpec();
        const cTplValGesInit< int > & NbIterFenSpec()const ;

        std::list< cSpecFitrageImage > & FiltreImageLoc();
        const std::list< cSpecFitrageImage > & FiltreImageLoc()const ;

        cTplValGesInit< int > & SzWInt();
        const cTplValGesInit< int > & SzWInt()const ;

        cTplValGesInit< int > & SurEchWCor();
        const cTplValGesInit< int > & SurEchWCor()const ;

        cTplValGesInit< eAlgoRegul > & AlgoRegul();
        const cTplValGesInit< eAlgoRegul > & AlgoRegul()const ;

        cTplValGesInit< eAlgoRegul > & AlgoWenCxRImpossible();
        const cTplValGesInit< eAlgoRegul > & AlgoWenCxRImpossible()const ;

        cTplValGesInit< bool > & CoxRoy8Cnx();
        const cTplValGesInit< bool > & CoxRoy8Cnx()const ;

        cTplValGesInit< bool > & CoxRoyUChar();
        const cTplValGesInit< bool > & CoxRoyUChar()const ;

        std::list< cEtapeProgDyn > & EtapeProgDyn();
        const std::list< cEtapeProgDyn > & EtapeProgDyn()const ;

        cTplValGesInit< double > & Px1PenteMax();
        const cTplValGesInit< double > & Px1PenteMax()const ;

        cTplValGesInit< double > & Px2PenteMax();
        const cTplValGesInit< double > & Px2PenteMax()const ;

        cTplValGesInit< bool > & ChoixNewProg();
        const cTplValGesInit< bool > & ChoixNewProg()const ;

        double & ValDefCorrel();
        const double & ValDefCorrel()const ;

        double & CostTrans();
        const double & CostTrans()const ;

        cTplValGesInit< bool > & ReInjectMask();
        const cTplValGesInit< bool > & ReInjectMask()const ;

        cTplValGesInit< double > & AmplKLPostTr();
        const cTplValGesInit< double > & AmplKLPostTr()const ;

        cTplValGesInit< int > & Erod32Mask();
        const cTplValGesInit< int > & Erod32Mask()const ;

        cTplValGesInit< int > & SzOpen32();
        const cTplValGesInit< int > & SzOpen32()const ;

        cTplValGesInit< int > & SeuilZC();
        const cTplValGesInit< int > & SeuilZC()const ;

        double & CostChangeEtiq();
        const double & CostChangeEtiq()const ;

        cTplValGesInit< bool > & SauvEtiq();
        const cTplValGesInit< bool > & SauvEtiq()const ;

        cTplValGesInit< cEtiqBestImage > & EtiqBestImage();
        const cTplValGesInit< cEtiqBestImage > & EtiqBestImage()const ;

        cTplValGesInit< cArgMaskAuto > & ArgMaskAuto();
        const cTplValGesInit< cArgMaskAuto > & ArgMaskAuto()const ;

        cTplValGesInit< cModulationProgDyn > & ModulationProgDyn();
        const cTplValGesInit< cModulationProgDyn > & ModulationProgDyn()const ;

        cTplValGesInit< int > & SsResolOptim();
        const cTplValGesInit< int > & SsResolOptim()const ;

        cTplValGesInit< double > & RatioDeZoomImage();
        const cTplValGesInit< double > & RatioDeZoomImage()const ;

        cTplValGesInit< int > & NdDiscKerInterp();
        const cTplValGesInit< int > & NdDiscKerInterp()const ;

        cTplValGesInit< eModeInterpolation > & ModeInterpolation();
        const cTplValGesInit< eModeInterpolation > & ModeInterpolation()const ;

        cTplValGesInit< double > & CoefInterpolationBicubique();
        const cTplValGesInit< double > & CoefInterpolationBicubique()const ;

        cTplValGesInit< double > & SzSinCard();
        const cTplValGesInit< double > & SzSinCard()const ;

        cTplValGesInit< double > & SzAppodSinCard();
        const cTplValGesInit< double > & SzAppodSinCard()const ;

        cTplValGesInit< int > & TailleFenetreSinusCardinal();
        const cTplValGesInit< int > & TailleFenetreSinusCardinal()const ;

        cTplValGesInit< bool > & ApodisationSinusCardinal();
        const cTplValGesInit< bool > & ApodisationSinusCardinal()const ;

        cTplValGesInit< int > & SzGeomDerivable();
        const cTplValGesInit< int > & SzGeomDerivable()const ;

        cTplValGesInit< double > & SeuilAttenZRegul();
        const cTplValGesInit< double > & SeuilAttenZRegul()const ;

        cTplValGesInit< double > & AttenRelatifSeuilZ();
        const cTplValGesInit< double > & AttenRelatifSeuilZ()const ;

        cTplValGesInit< double > & ZRegul_Quad();
        const cTplValGesInit< double > & ZRegul_Quad()const ;

        cTplValGesInit< double > & ZRegul();
        const cTplValGesInit< double > & ZRegul()const ;

        cTplValGesInit< double > & ZPas();
        const cTplValGesInit< double > & ZPas()const ;

        cTplValGesInit< int > & RabZDilatAltiMoins();
        const cTplValGesInit< int > & RabZDilatAltiMoins()const ;

        cTplValGesInit< int > & RabZDilatPlaniMoins();
        const cTplValGesInit< int > & RabZDilatPlaniMoins()const ;

        cTplValGesInit< int > & ZDilatAlti();
        const cTplValGesInit< int > & ZDilatAlti()const ;

        cTplValGesInit< int > & ZDilatPlani();
        const cTplValGesInit< int > & ZDilatPlani()const ;

        cTplValGesInit< double > & ZDilatPlaniPropPtsInt();
        const cTplValGesInit< double > & ZDilatPlaniPropPtsInt()const ;

        cTplValGesInit< bool > & ZRedrPx();
        const cTplValGesInit< bool > & ZRedrPx()const ;

        cTplValGesInit< bool > & ZDeqRedr();
        const cTplValGesInit< bool > & ZDeqRedr()const ;

        cTplValGesInit< int > & RedrNbIterMed();
        const cTplValGesInit< int > & RedrNbIterMed()const ;

        cTplValGesInit< int > & RedrSzMed();
        const cTplValGesInit< int > & RedrSzMed()const ;

        cTplValGesInit< bool > & RedrSauvBrut();
        const cTplValGesInit< bool > & RedrSauvBrut()const ;

        cTplValGesInit< int > & RedrNbIterMoy();
        const cTplValGesInit< int > & RedrNbIterMoy()const ;

        cTplValGesInit< int > & RedrSzMoy();
        const cTplValGesInit< int > & RedrSzMoy()const ;

        cTplValGesInit< double > & Px1Regul_Quad();
        const cTplValGesInit< double > & Px1Regul_Quad()const ;

        cTplValGesInit< double > & Px1Regul();
        const cTplValGesInit< double > & Px1Regul()const ;

        cTplValGesInit< double > & Px1Pas();
        const cTplValGesInit< double > & Px1Pas()const ;

        cTplValGesInit< int > & Px1DilatAlti();
        const cTplValGesInit< int > & Px1DilatAlti()const ;

        cTplValGesInit< int > & Px1DilatPlani();
        const cTplValGesInit< int > & Px1DilatPlani()const ;

        cTplValGesInit< double > & Px1DilatPlaniPropPtsInt();
        const cTplValGesInit< double > & Px1DilatPlaniPropPtsInt()const ;

        cTplValGesInit< bool > & Px1RedrPx();
        const cTplValGesInit< bool > & Px1RedrPx()const ;

        cTplValGesInit< bool > & Px1DeqRedr();
        const cTplValGesInit< bool > & Px1DeqRedr()const ;

        cTplValGesInit< double > & Px2Regul_Quad();
        const cTplValGesInit< double > & Px2Regul_Quad()const ;

        cTplValGesInit< double > & Px2Regul();
        const cTplValGesInit< double > & Px2Regul()const ;

        cTplValGesInit< double > & Px2Pas();
        const cTplValGesInit< double > & Px2Pas()const ;

        cTplValGesInit< int > & Px2DilatAlti();
        const cTplValGesInit< int > & Px2DilatAlti()const ;

        cTplValGesInit< int > & Px2DilatPlani();
        const cTplValGesInit< int > & Px2DilatPlani()const ;

        cTplValGesInit< double > & Px2DilatPlaniPropPtsInt();
        const cTplValGesInit< double > & Px2DilatPlaniPropPtsInt()const ;

        cTplValGesInit< bool > & Px2RedrPx();
        const cTplValGesInit< bool > & Px2RedrPx()const ;

        cTplValGesInit< bool > & Px2DeqRedr();
        const cTplValGesInit< bool > & Px2DeqRedr()const ;

        std::list< cSpecFitrageImage > & OneFitragePx();
        const std::list< cSpecFitrageImage > & OneFitragePx()const ;

        cTplValGesInit< cPostFiltragePx > & PostFiltragePx();
        const cTplValGesInit< cPostFiltragePx > & PostFiltragePx()const ;

        double & SzFiltre();
        const double & SzFiltre()const ;

        cTplValGesInit< int > & NbIter();
        const cTplValGesInit< int > & NbIter()const ;

        cTplValGesInit< double > & ExposPonderGrad();
        const cTplValGesInit< double > & ExposPonderGrad()const ;

        cTplValGesInit< double > & DericheFactEPC();
        const cTplValGesInit< double > & DericheFactEPC()const ;

        cTplValGesInit< double > & ValGradAtten();
        const cTplValGesInit< double > & ValGradAtten()const ;

        cTplValGesInit< double > & ExposPonderCorr();
        const cTplValGesInit< double > & ExposPonderCorr()const ;

        cTplValGesInit< cPostFiltrageDiscont > & PostFiltrageDiscont();
        const cTplValGesInit< cPostFiltrageDiscont > & PostFiltrageDiscont()const ;

        bool & ModeExclusion();
        const bool & ModeExclusion()const ;

        std::list< std::string > & PatternSel();
        const std::list< std::string > & PatternSel()const ;

        cTplValGesInit< cImageSelecteur > & ImageSelecteur();
        const cTplValGesInit< cImageSelecteur > & ImageSelecteur()const ;

        cTplValGesInit< bool > & Gen8Bits_Px1();
        const cTplValGesInit< bool > & Gen8Bits_Px1()const ;

        cTplValGesInit< int > & Offset8Bits_Px1();
        const cTplValGesInit< int > & Offset8Bits_Px1()const ;

        cTplValGesInit< double > & Dyn8Bits_Px1();
        const cTplValGesInit< double > & Dyn8Bits_Px1()const ;

        cTplValGesInit< bool > & Gen8Bits_Px2();
        const cTplValGesInit< bool > & Gen8Bits_Px2()const ;

        cTplValGesInit< int > & Offset8Bits_Px2();
        const cTplValGesInit< int > & Offset8Bits_Px2()const ;

        cTplValGesInit< double > & Dyn8Bits_Px2();
        const cTplValGesInit< double > & Dyn8Bits_Px2()const ;

        std::list< std::string > & ArgGen8Bits();
        const std::list< std::string > & ArgGen8Bits()const ;

        cTplValGesInit< bool > & GenFilePxRel();
        const cTplValGesInit< bool > & GenFilePxRel()const ;

        cTplValGesInit< bool > & GenImagesCorrel();
        const cTplValGesInit< bool > & GenImagesCorrel()const ;

        std::list< cGenerateProjectionInImages > & GenerateProjectionInImages();
        const std::list< cGenerateProjectionInImages > & GenerateProjectionInImages()const ;

        double & SsResolPx();
        const double & SsResolPx()const ;

        std::string & NameXMLFile();
        const std::string & NameXMLFile()const ;

        cTplValGesInit< cGenCorPxTransv > & GenCorPxTransv();
        const cTplValGesInit< cGenCorPxTransv > & GenCorPxTransv()const ;

        std::list< cGenereModeleRaster2Analytique > & ExportAsModeleDist();
        const std::list< cGenereModeleRaster2Analytique > & ExportAsModeleDist()const ;

        cTplValGesInit< ePxApply > & OptDif_PxApply();
        const cTplValGesInit< ePxApply > & OptDif_PxApply()const ;

        cTplValGesInit< bool > & VisuTerrainIm();
        const cTplValGesInit< bool > & VisuTerrainIm()const ;

        cTplValGesInit< int > & SzWTerr();
        const cTplValGesInit< int > & SzWTerr()const ;

        std::list< std::string > & UnSelectedImage();
        const std::list< std::string > & UnSelectedImage()const ;

        Pt2di & CentreVisuTerrain();
        const Pt2di & CentreVisuTerrain()const ;

        cTplValGesInit< int > & ZoomTerr();
        const cTplValGesInit< int > & ZoomTerr()const ;

        cTplValGesInit< int > & NbDiscHistoPartieFrac();
        const cTplValGesInit< int > & NbDiscHistoPartieFrac()const ;

        double & CoutFrac();
        const double & CoutFrac()const ;

        cTplValGesInit< cSimulFrac > & SimulFrac();
        const cTplValGesInit< cSimulFrac > & SimulFrac()const ;

        cTplValGesInit< cInterfaceVisualisation > & InterfaceVisualisation();
        const cTplValGesInit< cInterfaceVisualisation > & InterfaceVisualisation()const ;

        std::list< cExportNuage > & ExportNuage();
        const std::list< cExportNuage > & ExportNuage()const ;

        std::list< cOneModeleAnalytique > & OneModeleAnalytique();
        const std::list< cOneModeleAnalytique > & OneModeleAnalytique()const ;

        cTplValGesInit< cModelesAnalytiques > & ModelesAnalytiques();
        const cTplValGesInit< cModelesAnalytiques > & ModelesAnalytiques()const ;

        std::list< cBasculeRes > & BasculeRes();
        const std::list< cBasculeRes > & BasculeRes()const ;

        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;

        cTplValGesInit< double > & PasDisc();
        const cTplValGesInit< double > & PasDisc()const ;

        double & SeuilUsePC();
        const double & SeuilUsePC()const ;

        cTplValGesInit< std::string > & KeyCalcPC();
        const cTplValGesInit< std::string > & KeyCalcPC()const ;

        cTplValGesInit< bool > & AddChantierKPC();
        const cTplValGesInit< bool > & AddChantierKPC()const ;

        cTplValGesInit< bool > & SupresExtChantierKPC();
        const cTplValGesInit< bool > & SupresExtChantierKPC()const ;

        cTplValGesInit< bool > & Dequant();
        const cTplValGesInit< bool > & Dequant()const ;

        cTplValGesInit< bool > & ByMkF();
        const cTplValGesInit< bool > & ByMkF()const ;

        cTplValGesInit< std::string > & PatternApply();
        const cTplValGesInit< std::string > & PatternApply()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        double & Seuil();
        const double & Seuil()const ;

        cTplValGesInit< cVisuSuperposMNT > & VisuSuperposMNT();
        const cTplValGesInit< cVisuSuperposMNT > & VisuSuperposMNT()const ;

        cTplValGesInit< bool > & BufXYZ();
        const cTplValGesInit< bool > & BufXYZ()const ;

        cTplValGesInit< bool > & DoOnlyWhenNew();
        const cTplValGesInit< bool > & DoOnlyWhenNew()const ;

        cTplValGesInit< int > & SzBloc();
        const cTplValGesInit< int > & SzBloc()const ;

        cTplValGesInit< int > & SzBord();
        const cTplValGesInit< int > & SzBord()const ;

        cTplValGesInit< bool > & ImSuperpMNT();
        const cTplValGesInit< bool > & ImSuperpMNT()const ;

        cTplValGesInit< double > & ZMoy();
        const cTplValGesInit< double > & ZMoy()const ;

        cTplValGesInit< cElRegex_Ptr > & FiltreName();
        const cTplValGesInit< cElRegex_Ptr > & FiltreName()const ;

        cTplValGesInit< std::string > & DirOrtho();
        const cTplValGesInit< std::string > & DirOrtho()const ;

        cTplValGesInit< std::string > & FileMTD();
        const cTplValGesInit< std::string > & FileMTD()const ;

        cTplValGesInit< std::string > & NameFileSauv();
        const cTplValGesInit< std::string > & NameFileSauv()const ;

        cMasqMesures & Mesures();
        const cMasqMesures & Mesures()const ;

        cTplValGesInit< cMakeMTDMaskOrtho > & MakeMTDMaskOrtho();
        const cTplValGesInit< cMakeMTDMaskOrtho > & MakeMTDMaskOrtho()const ;

        cTplValGesInit< double > & OrthoBiCub();
        const cTplValGesInit< double > & OrthoBiCub()const ;

        cTplValGesInit< double > & ScaleBiCub();
        const cTplValGesInit< double > & ScaleBiCub()const ;

        cTplValGesInit< double > & ResolRelOrhto();
        const cTplValGesInit< double > & ResolRelOrhto()const ;

        cTplValGesInit< double > & ResolAbsOrtho();
        const cTplValGesInit< double > & ResolAbsOrtho()const ;

        cTplValGesInit< Pt2dr > & PixelTerrainPhase();
        const cTplValGesInit< Pt2dr > & PixelTerrainPhase()const ;

        std::string & KeyCalcInput();
        const std::string & KeyCalcInput()const ;

        std::string & KeyCalcOutput();
        const std::string & KeyCalcOutput()const ;

        cTplValGesInit< int > & NbChan();
        const cTplValGesInit< int > & NbChan()const ;

        cTplValGesInit< std::string > & KeyCalcIncidHor();
        const cTplValGesInit< std::string > & KeyCalcIncidHor()const ;

        cTplValGesInit< double > & SsResolIncH();
        const cTplValGesInit< double > & SsResolIncH()const ;

        cTplValGesInit< bool > & CalcIncAZMoy();
        const cTplValGesInit< bool > & CalcIncAZMoy()const ;

        cTplValGesInit< bool > & ImageIncIsDistFront();
        const cTplValGesInit< bool > & ImageIncIsDistFront()const ;

        cTplValGesInit< int > & RepulsFront();
        const cTplValGesInit< int > & RepulsFront()const ;

        cTplValGesInit< double > & ResolIm();
        const cTplValGesInit< double > & ResolIm()const ;

        cTplValGesInit< Pt2di > & TranslateIm();
        const cTplValGesInit< Pt2di > & TranslateIm()const ;

        cTplValGesInit< cMakeOrthoParImage > & MakeOrthoParImage();
        const cTplValGesInit< cMakeOrthoParImage > & MakeOrthoParImage()const ;

        cTplValGesInit< cGenerePartiesCachees > & GenerePartiesCachees();
        const cTplValGesInit< cGenerePartiesCachees > & GenerePartiesCachees()const ;

        std::string & NameOut();
        const std::string & NameOut()const ;

        std::string & NameMasq();
        const std::string & NameMasq()const ;

        std::string & NameOriGlob();
        const std::string & NameOriGlob()const ;

        cTplValGesInit< int > & XRecouvrt();
        const cTplValGesInit< int > & XRecouvrt()const ;

        cTplValGesInit< double > & MemAvalaible();
        const cTplValGesInit< double > & MemAvalaible()const ;

        cTplValGesInit< double > & FilterMulLargY();
        const cTplValGesInit< double > & FilterMulLargY()const ;

        cTplValGesInit< double > & NbIterFilterY();
        const cTplValGesInit< double > & NbIterFilterY()const ;

        cTplValGesInit< int > & FilterXY();
        const cTplValGesInit< int > & FilterXY()const ;

        cTplValGesInit< int > & NbIterXY();
        const cTplValGesInit< int > & NbIterXY()const ;

        cTplValGesInit< double > & DensityHighThresh();
        const cTplValGesInit< double > & DensityHighThresh()const ;

        cTplValGesInit< double > & DensityLowThresh();
        const cTplValGesInit< double > & DensityLowThresh()const ;

        cTplValGesInit< bool > & UseAutoMask();
        const cTplValGesInit< bool > & UseAutoMask()const ;

        cTplValGesInit< cRedrLocAnam > & RedrLocAnam();
        const cTplValGesInit< cRedrLocAnam > & RedrLocAnam()const ;

        cTplValGesInit< bool > & UsePartiesCachee();
        const cTplValGesInit< bool > & UsePartiesCachee()const ;

        cTplValGesInit< std::string > & NameVisuTestPC();
        const cTplValGesInit< std::string > & NameVisuTestPC()const ;

        std::string & KeyAssocIm2Nuage();
        const std::string & KeyAssocIm2Nuage()const ;

        cTplValGesInit< std::string > & Selector();
        const cTplValGesInit< std::string > & Selector()const ;

        double & ScaleNuage();
        const double & ScaleNuage()const ;

        cTplValGesInit< cNuagePredicteur > & NuagePredicteur();
        const cTplValGesInit< cNuagePredicteur > & NuagePredicteur()const ;
    private:
        int mDeZoom;
        cTplValGesInit< cCorrelAdHoc > mCorrelAdHoc;
        cTplValGesInit< cDoStatResult > mDoStatResult;
        std::list< cMasqOfEtape > mMasqOfEtape;
        cTplValGesInit< int > mSzRecouvrtDalles;
        cTplValGesInit< int > mSzDalleMin;
        cTplValGesInit< int > mSzDalleMax;
        cTplValGesInit< eModeDynamiqueCorrel > mDynamiqueCorrel;
        cTplValGesInit< eModeAggregCorr > mAggregCorr;
        cTplValGesInit< double > mSzW;
        cTplValGesInit< bool > mWSpecUseMasqGlob;
        cTplValGesInit< eTypeWinCorrel > mTypeWCorr;
        cTplValGesInit< double > mSzWy;
        cTplValGesInit< int > mNbIterFenSpec;
        std::list< cSpecFitrageImage > mFiltreImageLoc;
        cTplValGesInit< int > mSzWInt;
        cTplValGesInit< int > mSurEchWCor;
        cTplValGesInit< eAlgoRegul > mAlgoRegul;
        cTplValGesInit< eAlgoRegul > mAlgoWenCxRImpossible;
        cTplValGesInit< bool > mCoxRoy8Cnx;
        cTplValGesInit< bool > mCoxRoyUChar;
        cTplValGesInit< cModulationProgDyn > mModulationProgDyn;
        cTplValGesInit< int > mSsResolOptim;
        cTplValGesInit< double > mRatioDeZoomImage;
        cTplValGesInit< int > mNdDiscKerInterp;
        cTplValGesInit< eModeInterpolation > mModeInterpolation;
        cTplValGesInit< double > mCoefInterpolationBicubique;
        cTplValGesInit< double > mSzSinCard;
        cTplValGesInit< double > mSzAppodSinCard;
        cTplValGesInit< int > mTailleFenetreSinusCardinal;
        cTplValGesInit< bool > mApodisationSinusCardinal;
        cTplValGesInit< int > mSzGeomDerivable;
        cTplValGesInit< double > mSeuilAttenZRegul;
        cTplValGesInit< double > mAttenRelatifSeuilZ;
        cTplValGesInit< double > mZRegul_Quad;
        cTplValGesInit< double > mZRegul;
        cTplValGesInit< double > mZPas;
        cTplValGesInit< int > mRabZDilatAltiMoins;
        cTplValGesInit< int > mRabZDilatPlaniMoins;
        cTplValGesInit< int > mZDilatAlti;
        cTplValGesInit< int > mZDilatPlani;
        cTplValGesInit< double > mZDilatPlaniPropPtsInt;
        cTplValGesInit< bool > mZRedrPx;
        cTplValGesInit< bool > mZDeqRedr;
        cTplValGesInit< int > mRedrNbIterMed;
        cTplValGesInit< int > mRedrSzMed;
        cTplValGesInit< bool > mRedrSauvBrut;
        cTplValGesInit< int > mRedrNbIterMoy;
        cTplValGesInit< int > mRedrSzMoy;
        cTplValGesInit< double > mPx1Regul_Quad;
        cTplValGesInit< double > mPx1Regul;
        cTplValGesInit< double > mPx1Pas;
        cTplValGesInit< int > mPx1DilatAlti;
        cTplValGesInit< int > mPx1DilatPlani;
        cTplValGesInit< double > mPx1DilatPlaniPropPtsInt;
        cTplValGesInit< bool > mPx1RedrPx;
        cTplValGesInit< bool > mPx1DeqRedr;
        cTplValGesInit< double > mPx2Regul_Quad;
        cTplValGesInit< double > mPx2Regul;
        cTplValGesInit< double > mPx2Pas;
        cTplValGesInit< int > mPx2DilatAlti;
        cTplValGesInit< int > mPx2DilatPlani;
        cTplValGesInit< double > mPx2DilatPlaniPropPtsInt;
        cTplValGesInit< bool > mPx2RedrPx;
        cTplValGesInit< bool > mPx2DeqRedr;
        cTplValGesInit< cPostFiltragePx > mPostFiltragePx;
        cTplValGesInit< cPostFiltrageDiscont > mPostFiltrageDiscont;
        cTplValGesInit< cImageSelecteur > mImageSelecteur;
        cTplValGesInit< bool > mGen8Bits_Px1;
        cTplValGesInit< int > mOffset8Bits_Px1;
        cTplValGesInit< double > mDyn8Bits_Px1;
        cTplValGesInit< bool > mGen8Bits_Px2;
        cTplValGesInit< int > mOffset8Bits_Px2;
        cTplValGesInit< double > mDyn8Bits_Px2;
        std::list< std::string > mArgGen8Bits;
        cTplValGesInit< bool > mGenFilePxRel;
        cTplValGesInit< bool > mGenImagesCorrel;
        std::list< cGenerateProjectionInImages > mGenerateProjectionInImages;
        cTplValGesInit< cGenCorPxTransv > mGenCorPxTransv;
        std::list< cGenereModeleRaster2Analytique > mExportAsModeleDist;
        cTplValGesInit< ePxApply > mOptDif_PxApply;
        cTplValGesInit< cInterfaceVisualisation > mInterfaceVisualisation;
        std::list< cExportNuage > mExportNuage;
        cTplValGesInit< cModelesAnalytiques > mModelesAnalytiques;
        std::list< cBasculeRes > mBasculeRes;
        cTplValGesInit< cGenerePartiesCachees > mGenerePartiesCachees;
        cTplValGesInit< cRedrLocAnam > mRedrLocAnam;
        cTplValGesInit< bool > mUsePartiesCachee;
        cTplValGesInit< std::string > mNameVisuTestPC;
        cTplValGesInit< cNuagePredicteur > mNuagePredicteur;
};
cElXMLTree * ToXMLTree(const cEtapeMEC &);

class cTypePyramImage
{
    public:
        friend void xml_init(cTypePyramImage & anObj,cElXMLTree * aTree);


        int & Resol();
        const int & Resol()const ;

        cTplValGesInit< int > & DivIm();
        const cTplValGesInit< int > & DivIm()const ;

        eTypeImPyram & TypeEl();
        const eTypeImPyram & TypeEl()const ;
    private:
        int mResol;
        cTplValGesInit< int > mDivIm;
        eTypeImPyram mTypeEl;
};
cElXMLTree * ToXMLTree(const cTypePyramImage &);

class cSection_MEC
{
    public:
        friend void xml_init(cSection_MEC & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & PasIsInPixel();
        const cTplValGesInit< bool > & PasIsInPixel()const ;

        cTplValGesInit< Box2dr > & ProportionClipMEC();
        const cTplValGesInit< Box2dr > & ProportionClipMEC()const ;

        cTplValGesInit< bool > & ClipMecIsProp();
        const cTplValGesInit< bool > & ClipMecIsProp()const ;

        cTplValGesInit< double > & ZoomClipMEC();
        const cTplValGesInit< double > & ZoomClipMEC()const ;

        cTplValGesInit< int > & NbMinImagesVisibles();
        const cTplValGesInit< int > & NbMinImagesVisibles()const ;

        cTplValGesInit< bool > & OneDefCorAllPxDefCor();
        const cTplValGesInit< bool > & OneDefCorAllPxDefCor()const ;

        cTplValGesInit< int > & ZoomBeginODC_APDC();
        const cTplValGesInit< int > & ZoomBeginODC_APDC()const ;

        cTplValGesInit< double > & DefCorrelation();
        const cTplValGesInit< double > & DefCorrelation()const ;

        cTplValGesInit< bool > & ReprojPixelNoVal();
        const cTplValGesInit< bool > & ReprojPixelNoVal()const ;

        cTplValGesInit< double > & EpsilonCorrelation();
        const cTplValGesInit< double > & EpsilonCorrelation()const ;

        int & FreqEchantPtsI();
        const int & FreqEchantPtsI()const ;

        eTypeModeEchantPtsI & ModeEchantPtsI();
        const eTypeModeEchantPtsI & ModeEchantPtsI()const ;

        cTplValGesInit< std::string > & KeyCommandeExterneInteret();
        const cTplValGesInit< std::string > & KeyCommandeExterneInteret()const ;

        cTplValGesInit< int > & SzVAutoCorrel();
        const cTplValGesInit< int > & SzVAutoCorrel()const ;

        cTplValGesInit< double > & EstmBrAutoCorrel();
        const cTplValGesInit< double > & EstmBrAutoCorrel()const ;

        cTplValGesInit< double > & SeuilLambdaAutoCorrel();
        const cTplValGesInit< double > & SeuilLambdaAutoCorrel()const ;

        cTplValGesInit< double > & SeuilEcartTypeAutoCorrel();
        const cTplValGesInit< double > & SeuilEcartTypeAutoCorrel()const ;

        cTplValGesInit< double > & RepartExclusion();
        const cTplValGesInit< double > & RepartExclusion()const ;

        cTplValGesInit< double > & RepartEvitement();
        const cTplValGesInit< double > & RepartEvitement()const ;

        cTplValGesInit< cEchantillonagePtsInterets > & EchantillonagePtsInterets();
        const cTplValGesInit< cEchantillonagePtsInterets > & EchantillonagePtsInterets()const ;

        cTplValGesInit< bool > & ChantierFullImage1();
        const cTplValGesInit< bool > & ChantierFullImage1()const ;

        cTplValGesInit< bool > & ChantierFullMaskImage1();
        const cTplValGesInit< bool > & ChantierFullMaskImage1()const ;

        cTplValGesInit< bool > & ExportForMultiplePointsHomologues();
        const cTplValGesInit< bool > & ExportForMultiplePointsHomologues()const ;

        cTplValGesInit< double > & CovLim();
        const cTplValGesInit< double > & CovLim()const ;

        cTplValGesInit< double > & TermeDecr();
        const cTplValGesInit< double > & TermeDecr()const ;

        cTplValGesInit< int > & SzRef();
        const cTplValGesInit< int > & SzRef()const ;

        cTplValGesInit< double > & ValRef();
        const cTplValGesInit< double > & ValRef()const ;

        cTplValGesInit< cAdapteDynCov > & AdapteDynCov();
        const cTplValGesInit< cAdapteDynCov > & AdapteDynCov()const ;

        std::list< cEtapeMEC > & EtapeMEC();
        const std::list< cEtapeMEC > & EtapeMEC()const ;

        std::list< cTypePyramImage > & TypePyramImage();
        const std::list< cTypePyramImage > & TypePyramImage()const ;

        cTplValGesInit< bool > & Correl16Bits();
        const cTplValGesInit< bool > & Correl16Bits()const ;
    private:
        cTplValGesInit< bool > mPasIsInPixel;
        cTplValGesInit< Box2dr > mProportionClipMEC;
        cTplValGesInit< bool > mClipMecIsProp;
        cTplValGesInit< double > mZoomClipMEC;
        cTplValGesInit< int > mNbMinImagesVisibles;
        cTplValGesInit< bool > mOneDefCorAllPxDefCor;
        cTplValGesInit< int > mZoomBeginODC_APDC;
        cTplValGesInit< double > mDefCorrelation;
        cTplValGesInit< bool > mReprojPixelNoVal;
        cTplValGesInit< double > mEpsilonCorrelation;
        cTplValGesInit< cEchantillonagePtsInterets > mEchantillonagePtsInterets;
        cTplValGesInit< bool > mChantierFullImage1;
        cTplValGesInit< bool > mChantierFullMaskImage1;
        cTplValGesInit< bool > mExportForMultiplePointsHomologues;
        cTplValGesInit< cAdapteDynCov > mAdapteDynCov;
        std::list< cEtapeMEC > mEtapeMEC;
        cEtapeMEC mGlobEtapeMEC;
        std::list< cTypePyramImage > mTypePyramImage;
        cTplValGesInit< bool > mCorrel16Bits;
};
cElXMLTree * ToXMLTree(const cSection_MEC &);

/******************************************************/
/******************************************************/
/******************************************************/
class cDoNothingBut
{
    public:
        friend void xml_init(cDoNothingBut & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & ButDoPyram();
        const cTplValGesInit< bool > & ButDoPyram()const ;

        cTplValGesInit< bool > & ButDoMasqIm();
        const cTplValGesInit< bool > & ButDoMasqIm()const ;

        cTplValGesInit< bool > & ButDoMemPart();
        const cTplValGesInit< bool > & ButDoMemPart()const ;

        cTplValGesInit< bool > & ButDoTA();
        const cTplValGesInit< bool > & ButDoTA()const ;

        cTplValGesInit< bool > & ButDoMasqueChantier();
        const cTplValGesInit< bool > & ButDoMasqueChantier()const ;

        cTplValGesInit< bool > & ButDoOriMNT();
        const cTplValGesInit< bool > & ButDoOriMNT()const ;

        cTplValGesInit< bool > & ButDoFDC();
        const cTplValGesInit< bool > & ButDoFDC()const ;

        cTplValGesInit< bool > & ButDoExtendParam();
        const cTplValGesInit< bool > & ButDoExtendParam()const ;

        cTplValGesInit< bool > & ButDoGenCorPxTransv();
        const cTplValGesInit< bool > & ButDoGenCorPxTransv()const ;

        cTplValGesInit< bool > & ButDoPartiesCachees();
        const cTplValGesInit< bool > & ButDoPartiesCachees()const ;

        cTplValGesInit< bool > & ButDoOrtho();
        const cTplValGesInit< bool > & ButDoOrtho()const ;

        cTplValGesInit< bool > & ButDoSimul();
        const cTplValGesInit< bool > & ButDoSimul()const ;

        cTplValGesInit< bool > & ButDoRedrLocAnam();
        const cTplValGesInit< bool > & ButDoRedrLocAnam()const ;
    private:
        cTplValGesInit< bool > mButDoPyram;
        cTplValGesInit< bool > mButDoMasqIm;
        cTplValGesInit< bool > mButDoMemPart;
        cTplValGesInit< bool > mButDoTA;
        cTplValGesInit< bool > mButDoMasqueChantier;
        cTplValGesInit< bool > mButDoOriMNT;
        cTplValGesInit< bool > mButDoFDC;
        cTplValGesInit< bool > mButDoExtendParam;
        cTplValGesInit< bool > mButDoGenCorPxTransv;
        cTplValGesInit< bool > mButDoPartiesCachees;
        cTplValGesInit< bool > mButDoOrtho;
        cTplValGesInit< bool > mButDoSimul;
        cTplValGesInit< bool > mButDoRedrLocAnam;
};
cElXMLTree * ToXMLTree(const cDoNothingBut &);

class cFoncPer
{
    public:
        friend void xml_init(cFoncPer & anObj,cElXMLTree * aTree);


        Pt2dr & Per();
        const Pt2dr & Per()const ;

        double & Ampl();
        const double & Ampl()const ;

        cTplValGesInit< bool > & AmplIsDer();
        const cTplValGesInit< bool > & AmplIsDer()const ;
    private:
        Pt2dr mPer;
        double mAmpl;
        cTplValGesInit< bool > mAmplIsDer;
};
cElXMLTree * ToXMLTree(const cFoncPer &);

class cMNTPart
{
    public:
        friend void xml_init(cMNTPart & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt2dr > & PenteGlob();
        const cTplValGesInit< Pt2dr > & PenteGlob()const ;

        std::list< cFoncPer > & FoncPer();
        const std::list< cFoncPer > & FoncPer()const ;
    private:
        cTplValGesInit< Pt2dr > mPenteGlob;
        std::list< cFoncPer > mFoncPer;
};
cElXMLTree * ToXMLTree(const cMNTPart &);

class cSimulBarres
{
    public:
        friend void xml_init(cSimulBarres & anObj,cElXMLTree * aTree);


        int & Nb();
        const int & Nb()const ;

        cTplValGesInit< double > & PowDistLongueur();
        const cTplValGesInit< double > & PowDistLongueur()const ;

        Pt2dr & IntervLongeur();
        const Pt2dr & IntervLongeur()const ;

        Pt2dr & IntervLargeur();
        const Pt2dr & IntervLargeur()const ;

        Pt2dr & IntervPentes();
        const Pt2dr & IntervPentes()const ;

        Pt2dr & IntervHauteur();
        const Pt2dr & IntervHauteur()const ;

        cTplValGesInit< double > & ProbSortant();
        const cTplValGesInit< double > & ProbSortant()const ;
    private:
        int mNb;
        cTplValGesInit< double > mPowDistLongueur;
        Pt2dr mIntervLongeur;
        Pt2dr mIntervLargeur;
        Pt2dr mIntervPentes;
        Pt2dr mIntervHauteur;
        cTplValGesInit< double > mProbSortant;
};
cElXMLTree * ToXMLTree(const cSimulBarres &);

class cMNEPart
{
    public:
        friend void xml_init(cMNEPart & anObj,cElXMLTree * aTree);


        std::list< cSimulBarres > & SimulBarres();
        const std::list< cSimulBarres > & SimulBarres()const ;
    private:
        std::list< cSimulBarres > mSimulBarres;
};
cElXMLTree * ToXMLTree(const cMNEPart &);

class cSimulRelief
{
    public:
        friend void xml_init(cSimulRelief & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DoItR();
        const cTplValGesInit< bool > & DoItR()const ;

        cTplValGesInit< Pt2dr > & PenteGlob();
        const cTplValGesInit< Pt2dr > & PenteGlob()const ;

        std::list< cFoncPer > & FoncPer();
        const std::list< cFoncPer > & FoncPer()const ;

        cMNTPart & MNTPart();
        const cMNTPart & MNTPart()const ;

        std::list< cSimulBarres > & SimulBarres();
        const std::list< cSimulBarres > & SimulBarres()const ;

        cMNEPart & MNEPart();
        const cMNEPart & MNEPart()const ;
    private:
        cTplValGesInit< bool > mDoItR;
        cMNTPart mMNTPart;
        cMNEPart mMNEPart;
};
cElXMLTree * ToXMLTree(const cSimulRelief &);

class cTexturePart
{
    public:
        friend void xml_init(cTexturePart & anObj,cElXMLTree * aTree);


        std::string & Texton();
        const std::string & Texton()const ;

        std::string & ImRes();
        const std::string & ImRes()const ;
    private:
        std::string mTexton;
        std::string mImRes;
};
cElXMLTree * ToXMLTree(const cTexturePart &);

class cProjImPart
{
    public:
        friend void xml_init(cProjImPart & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & PatternSel();
        const cElRegex_Ptr & PatternSel()const ;

        cTplValGesInit< int > & SzBloc();
        const cTplValGesInit< int > & SzBloc()const ;

        cTplValGesInit< int > & SzBrd();
        const cTplValGesInit< int > & SzBrd()const ;

        cTplValGesInit< double > & RatioSurResol();
        const cTplValGesInit< double > & RatioSurResol()const ;

        cTplValGesInit< std::string > & KeyProjMNT();
        const cTplValGesInit< std::string > & KeyProjMNT()const ;

        cTplValGesInit< std::string > & KeyIm();
        const cTplValGesInit< std::string > & KeyIm()const ;

        cTplValGesInit< double > & BicubParam();
        const cTplValGesInit< double > & BicubParam()const ;

        cTplValGesInit< bool > & ReprojInverse();
        const cTplValGesInit< bool > & ReprojInverse()const ;

        cTplValGesInit< double > & SzFTM();
        const cTplValGesInit< double > & SzFTM()const ;

        cTplValGesInit< double > & Bruit();
        const cTplValGesInit< double > & Bruit()const ;
    private:
        cElRegex_Ptr mPatternSel;
        cTplValGesInit< int > mSzBloc;
        cTplValGesInit< int > mSzBrd;
        cTplValGesInit< double > mRatioSurResol;
        cTplValGesInit< std::string > mKeyProjMNT;
        cTplValGesInit< std::string > mKeyIm;
        cTplValGesInit< double > mBicubParam;
        cTplValGesInit< bool > mReprojInverse;
        cTplValGesInit< double > mSzFTM;
        cTplValGesInit< double > mBruit;
};
cElXMLTree * ToXMLTree(const cProjImPart &);

class cSectionSimulation
{
    public:
        friend void xml_init(cSectionSimulation & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DoItR();
        const cTplValGesInit< bool > & DoItR()const ;

        cTplValGesInit< Pt2dr > & PenteGlob();
        const cTplValGesInit< Pt2dr > & PenteGlob()const ;

        std::list< cFoncPer > & FoncPer();
        const std::list< cFoncPer > & FoncPer()const ;

        cMNTPart & MNTPart();
        const cMNTPart & MNTPart()const ;

        std::list< cSimulBarres > & SimulBarres();
        const std::list< cSimulBarres > & SimulBarres()const ;

        cMNEPart & MNEPart();
        const cMNEPart & MNEPart()const ;

        cSimulRelief & SimulRelief();
        const cSimulRelief & SimulRelief()const ;

        std::string & Texton();
        const std::string & Texton()const ;

        std::string & ImRes();
        const std::string & ImRes()const ;

        cTexturePart & TexturePart();
        const cTexturePart & TexturePart()const ;

        cElRegex_Ptr & PatternSel();
        const cElRegex_Ptr & PatternSel()const ;

        cTplValGesInit< int > & SzBloc();
        const cTplValGesInit< int > & SzBloc()const ;

        cTplValGesInit< int > & SzBrd();
        const cTplValGesInit< int > & SzBrd()const ;

        cTplValGesInit< double > & RatioSurResol();
        const cTplValGesInit< double > & RatioSurResol()const ;

        cTplValGesInit< std::string > & KeyProjMNT();
        const cTplValGesInit< std::string > & KeyProjMNT()const ;

        cTplValGesInit< std::string > & KeyIm();
        const cTplValGesInit< std::string > & KeyIm()const ;

        cTplValGesInit< double > & BicubParam();
        const cTplValGesInit< double > & BicubParam()const ;

        cTplValGesInit< bool > & ReprojInverse();
        const cTplValGesInit< bool > & ReprojInverse()const ;

        cTplValGesInit< double > & SzFTM();
        const cTplValGesInit< double > & SzFTM()const ;

        cTplValGesInit< double > & Bruit();
        const cTplValGesInit< double > & Bruit()const ;

        cProjImPart & ProjImPart();
        const cProjImPart & ProjImPart()const ;
    private:
        cSimulRelief mSimulRelief;
        cTexturePart mTexturePart;
        cProjImPart mProjImPart;
};
cElXMLTree * ToXMLTree(const cSectionSimulation &);

class cAnamSurfaceAnalytique
{
    public:
        friend void xml_init(cAnamSurfaceAnalytique & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        std::string & Id();
        const std::string & Id()const ;
    private:
        std::string mNameFile;
        std::string mId;
};
cElXMLTree * ToXMLTree(const cAnamSurfaceAnalytique &);

class cAnamorphoseGeometrieMNT
{
    public:
        friend void xml_init(cAnamorphoseGeometrieMNT & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< cAnamSurfaceAnalytique > & AnamSurfaceAnalytique();
        const cTplValGesInit< cAnamSurfaceAnalytique > & AnamSurfaceAnalytique()const ;

        cTplValGesInit< int > & AnamDeZoomMasq();
        const cTplValGesInit< int > & AnamDeZoomMasq()const ;

        cTplValGesInit< double > & AnamLimAngleVisib();
        const cTplValGesInit< double > & AnamLimAngleVisib()const ;
    private:
        cTplValGesInit< cAnamSurfaceAnalytique > mAnamSurfaceAnalytique;
        cTplValGesInit< int > mAnamDeZoomMasq;
        cTplValGesInit< double > mAnamLimAngleVisib;
};
cElXMLTree * ToXMLTree(const cAnamorphoseGeometrieMNT &);

class cColorimetriesCanaux
{
    public:
        friend void xml_init(cColorimetriesCanaux & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & CanalSelector();
        const cElRegex_Ptr & CanalSelector()const ;

        cTplValGesInit< double > & ValBlanc();
        const cTplValGesInit< double > & ValBlanc()const ;

        cTplValGesInit< double > & ValNoir();
        const cTplValGesInit< double > & ValNoir()const ;
    private:
        cElRegex_Ptr mCanalSelector;
        cTplValGesInit< double > mValBlanc;
        cTplValGesInit< double > mValNoir;
};
cElXMLTree * ToXMLTree(const cColorimetriesCanaux &);

class cSuperpositionImages
{
    public:
        friend void xml_init(cSuperpositionImages & anObj,cElXMLTree * aTree);


        Pt3di & OrdreChannels();
        const Pt3di & OrdreChannels()const ;

        cTplValGesInit< Pt2di > & PtBalanceBlancs();
        const cTplValGesInit< Pt2di > & PtBalanceBlancs()const ;

        cTplValGesInit< Pt2di > & P0Sup();
        const cTplValGesInit< Pt2di > & P0Sup()const ;

        cTplValGesInit< Pt2di > & SzSup();
        const cTplValGesInit< Pt2di > & SzSup()const ;

        cElRegex_Ptr & PatternSelGrid();
        const cElRegex_Ptr & PatternSelGrid()const ;

        std::string & PatternNameGrid();
        const std::string & PatternNameGrid()const ;

        std::list< cColorimetriesCanaux > & ColorimetriesCanaux();
        const std::list< cColorimetriesCanaux > & ColorimetriesCanaux()const ;

        cTplValGesInit< double > & GammaCorrection();
        const cTplValGesInit< double > & GammaCorrection()const ;

        cTplValGesInit< double > & MultiplicateurBlanc();
        const cTplValGesInit< double > & MultiplicateurBlanc()const ;

        cTplValGesInit< bool > & GenFileImages();
        const cTplValGesInit< bool > & GenFileImages()const ;
    private:
        Pt3di mOrdreChannels;
        cTplValGesInit< Pt2di > mPtBalanceBlancs;
        cTplValGesInit< Pt2di > mP0Sup;
        cTplValGesInit< Pt2di > mSzSup;
        cElRegex_Ptr mPatternSelGrid;
        std::string mPatternNameGrid;
        std::list< cColorimetriesCanaux > mColorimetriesCanaux;
        cTplValGesInit< double > mGammaCorrection;
        cTplValGesInit< double > mMultiplicateurBlanc;
        cTplValGesInit< bool > mGenFileImages;
};
cElXMLTree * ToXMLTree(const cSuperpositionImages &);

class cSection_Results
{
    public:
        friend void xml_init(cSection_Results & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & Use_MM_EtatAvancement();
        const cTplValGesInit< bool > & Use_MM_EtatAvancement()const ;

        cTplValGesInit< bool > & ButDoPyram();
        const cTplValGesInit< bool > & ButDoPyram()const ;

        cTplValGesInit< bool > & ButDoMasqIm();
        const cTplValGesInit< bool > & ButDoMasqIm()const ;

        cTplValGesInit< bool > & ButDoMemPart();
        const cTplValGesInit< bool > & ButDoMemPart()const ;

        cTplValGesInit< bool > & ButDoTA();
        const cTplValGesInit< bool > & ButDoTA()const ;

        cTplValGesInit< bool > & ButDoMasqueChantier();
        const cTplValGesInit< bool > & ButDoMasqueChantier()const ;

        cTplValGesInit< bool > & ButDoOriMNT();
        const cTplValGesInit< bool > & ButDoOriMNT()const ;

        cTplValGesInit< bool > & ButDoFDC();
        const cTplValGesInit< bool > & ButDoFDC()const ;

        cTplValGesInit< bool > & ButDoExtendParam();
        const cTplValGesInit< bool > & ButDoExtendParam()const ;

        cTplValGesInit< bool > & ButDoGenCorPxTransv();
        const cTplValGesInit< bool > & ButDoGenCorPxTransv()const ;

        cTplValGesInit< bool > & ButDoPartiesCachees();
        const cTplValGesInit< bool > & ButDoPartiesCachees()const ;

        cTplValGesInit< bool > & ButDoOrtho();
        const cTplValGesInit< bool > & ButDoOrtho()const ;

        cTplValGesInit< bool > & ButDoSimul();
        const cTplValGesInit< bool > & ButDoSimul()const ;

        cTplValGesInit< bool > & ButDoRedrLocAnam();
        const cTplValGesInit< bool > & ButDoRedrLocAnam()const ;

        cTplValGesInit< cDoNothingBut > & DoNothingBut();
        const cTplValGesInit< cDoNothingBut > & DoNothingBut()const ;

        cTplValGesInit< int > & Paral_Pc_IdProcess();
        const cTplValGesInit< int > & Paral_Pc_IdProcess()const ;

        cTplValGesInit< int > & Paral_Pc_NbProcess();
        const cTplValGesInit< int > & Paral_Pc_NbProcess()const ;

        cTplValGesInit< double > & X_DirPlanInterFaisceau();
        const cTplValGesInit< double > & X_DirPlanInterFaisceau()const ;

        cTplValGesInit< double > & Y_DirPlanInterFaisceau();
        const cTplValGesInit< double > & Y_DirPlanInterFaisceau()const ;

        cTplValGesInit< double > & Z_DirPlanInterFaisceau();
        const cTplValGesInit< double > & Z_DirPlanInterFaisceau()const ;

        eModeGeomMNT & GeomMNT();
        const eModeGeomMNT & GeomMNT()const ;

        cTplValGesInit< cSectionSimulation > & SectionSimulation();
        const cTplValGesInit< cSectionSimulation > & SectionSimulation()const ;

        cTplValGesInit< bool > & Prio2OwnAltisolForEmprise();
        const cTplValGesInit< bool > & Prio2OwnAltisolForEmprise()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< cAnamSurfaceAnalytique > & AnamSurfaceAnalytique();
        const cTplValGesInit< cAnamSurfaceAnalytique > & AnamSurfaceAnalytique()const ;

        cTplValGesInit< int > & AnamDeZoomMasq();
        const cTplValGesInit< int > & AnamDeZoomMasq()const ;

        cTplValGesInit< double > & AnamLimAngleVisib();
        const cTplValGesInit< double > & AnamLimAngleVisib()const ;

        cTplValGesInit< cAnamorphoseGeometrieMNT > & AnamorphoseGeometrieMNT();
        const cTplValGesInit< cAnamorphoseGeometrieMNT > & AnamorphoseGeometrieMNT()const ;

        cTplValGesInit< std::string > & RepereCorrel();
        const cTplValGesInit< std::string > & RepereCorrel()const ;

        cTplValGesInit< std::string > & TagRepereCorrel();
        const cTplValGesInit< std::string > & TagRepereCorrel()const ;

        cTplValGesInit< bool > & DoMEC();
        const cTplValGesInit< bool > & DoMEC()const ;

        cTplValGesInit< bool > & DoFDC();
        const cTplValGesInit< bool > & DoFDC()const ;

        cTplValGesInit< bool > & GenereXMLComp();
        const cTplValGesInit< bool > & GenereXMLComp()const ;

        cTplValGesInit< int > & ZoomMakeTA();
        const cTplValGesInit< int > & ZoomMakeTA()const ;

        cTplValGesInit< double > & SaturationTA();
        const cTplValGesInit< double > & SaturationTA()const ;

        cTplValGesInit< bool > & OrthoTA();
        const cTplValGesInit< bool > & OrthoTA()const ;

        cTplValGesInit< int > & ZoomMakeMasq();
        const cTplValGesInit< int > & ZoomMakeMasq()const ;

        cTplValGesInit< bool > & MakeImCptTA();
        const cTplValGesInit< bool > & MakeImCptTA()const ;

        cTplValGesInit< std::string > & FilterTA();
        const cTplValGesInit< std::string > & FilterTA()const ;

        cTplValGesInit< double > & GammaVisu();
        const cTplValGesInit< double > & GammaVisu()const ;

        cTplValGesInit< int > & ZoomVisuLiaison();
        const cTplValGesInit< int > & ZoomVisuLiaison()const ;

        cTplValGesInit< double > & TolerancePointHomInImage();
        const cTplValGesInit< double > & TolerancePointHomInImage()const ;

        cTplValGesInit< double > & FiltragePointHomInImage();
        const cTplValGesInit< double > & FiltragePointHomInImage()const ;

        cTplValGesInit< int > & BaseCodeRetourMicmacErreur();
        const cTplValGesInit< int > & BaseCodeRetourMicmacErreur()const ;

        Pt3di & OrdreChannels();
        const Pt3di & OrdreChannels()const ;

        cTplValGesInit< Pt2di > & PtBalanceBlancs();
        const cTplValGesInit< Pt2di > & PtBalanceBlancs()const ;

        cTplValGesInit< Pt2di > & P0Sup();
        const cTplValGesInit< Pt2di > & P0Sup()const ;

        cTplValGesInit< Pt2di > & SzSup();
        const cTplValGesInit< Pt2di > & SzSup()const ;

        cElRegex_Ptr & PatternSelGrid();
        const cElRegex_Ptr & PatternSelGrid()const ;

        std::string & PatternNameGrid();
        const std::string & PatternNameGrid()const ;

        std::list< cColorimetriesCanaux > & ColorimetriesCanaux();
        const std::list< cColorimetriesCanaux > & ColorimetriesCanaux()const ;

        cTplValGesInit< double > & GammaCorrection();
        const cTplValGesInit< double > & GammaCorrection()const ;

        cTplValGesInit< double > & MultiplicateurBlanc();
        const cTplValGesInit< double > & MultiplicateurBlanc()const ;

        cTplValGesInit< bool > & GenFileImages();
        const cTplValGesInit< bool > & GenFileImages()const ;

        cTplValGesInit< cSuperpositionImages > & SuperpositionImages();
        const cTplValGesInit< cSuperpositionImages > & SuperpositionImages()const ;
    private:
        cTplValGesInit< bool > mUse_MM_EtatAvancement;
        cTplValGesInit< cDoNothingBut > mDoNothingBut;
        cTplValGesInit< int > mParal_Pc_IdProcess;
        cTplValGesInit< int > mParal_Pc_NbProcess;
        cTplValGesInit< double > mX_DirPlanInterFaisceau;
        cTplValGesInit< double > mY_DirPlanInterFaisceau;
        cTplValGesInit< double > mZ_DirPlanInterFaisceau;
        eModeGeomMNT mGeomMNT;
        cTplValGesInit< cSectionSimulation > mSectionSimulation;
        cTplValGesInit< bool > mPrio2OwnAltisolForEmprise;
        cTplValGesInit< cAnamorphoseGeometrieMNT > mAnamorphoseGeometrieMNT;
        cTplValGesInit< std::string > mRepereCorrel;
        cTplValGesInit< std::string > mTagRepereCorrel;
        cTplValGesInit< bool > mDoMEC;
        cTplValGesInit< bool > mDoFDC;
        cTplValGesInit< bool > mGenereXMLComp;
        cTplValGesInit< int > mZoomMakeTA;
        cTplValGesInit< double > mSaturationTA;
        cTplValGesInit< bool > mOrthoTA;
        cTplValGesInit< int > mZoomMakeMasq;
        cTplValGesInit< bool > mMakeImCptTA;
        cTplValGesInit< std::string > mFilterTA;
        cTplValGesInit< double > mGammaVisu;
        cTplValGesInit< int > mZoomVisuLiaison;
        cTplValGesInit< double > mTolerancePointHomInImage;
        cTplValGesInit< double > mFiltragePointHomInImage;
        cTplValGesInit< int > mBaseCodeRetourMicmacErreur;
        cTplValGesInit< cSuperpositionImages > mSuperpositionImages;
};
cElXMLTree * ToXMLTree(const cSection_Results &);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalcNomChantier
{
    public:
        friend void xml_init(cCalcNomChantier & anObj,cElXMLTree * aTree);


        std::string & PatternSelChantier();
        const std::string & PatternSelChantier()const ;

        std::string & PatNameChantier();
        const std::string & PatNameChantier()const ;

        cTplValGesInit< std::string > & SeparateurChantier();
        const cTplValGesInit< std::string > & SeparateurChantier()const ;
    private:
        std::string mPatternSelChantier;
        std::string mPatNameChantier;
        cTplValGesInit< std::string > mSeparateurChantier;
};
cElXMLTree * ToXMLTree(const cCalcNomChantier &);

class cPurgeFiles
{
    public:
        friend void xml_init(cPurgeFiles & anObj,cElXMLTree * aTree);


        std::string & PatternSelPurge();
        const std::string & PatternSelPurge()const ;

        bool & PurgeToSupress();
        const bool & PurgeToSupress()const ;
    private:
        std::string mPatternSelPurge;
        bool mPurgeToSupress;
};
cElXMLTree * ToXMLTree(const cPurgeFiles &);

class cSection_WorkSpace
{
    public:
        friend void xml_init(cSection_WorkSpace & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & FileExportApero2MM();
        const cTplValGesInit< std::string > & FileExportApero2MM()const ;

        cTplValGesInit< bool > & UseProfInVertLoc();
        const cTplValGesInit< bool > & UseProfInVertLoc()const ;

        cTplValGesInit< std::string > & NameFileParamMICMAC();
        const cTplValGesInit< std::string > & NameFileParamMICMAC()const ;

        std::string & WorkDir();
        const std::string & WorkDir()const ;

        cTplValGesInit< std::string > & DirImagesOri();
        const cTplValGesInit< std::string > & DirImagesOri()const ;

        std::string & TmpMEC();
        const std::string & TmpMEC()const ;

        cTplValGesInit< std::string > & TmpPyr();
        const cTplValGesInit< std::string > & TmpPyr()const ;

        cTplValGesInit< std::string > & TmpGeom();
        const cTplValGesInit< std::string > & TmpGeom()const ;

        cTplValGesInit< std::string > & TmpResult();
        const cTplValGesInit< std::string > & TmpResult()const ;

        cTplValGesInit< bool > & CalledByProcess();
        const cTplValGesInit< bool > & CalledByProcess()const ;

        cTplValGesInit< bool > & Visu();
        const cTplValGesInit< bool > & Visu()const ;

        cTplValGesInit< int > & ByProcess();
        const cTplValGesInit< int > & ByProcess()const ;

        cTplValGesInit< bool > & StopOnEchecFils();
        const cTplValGesInit< bool > & StopOnEchecFils()const ;

        cTplValGesInit< int > & AvalaibleMemory();
        const cTplValGesInit< int > & AvalaibleMemory()const ;

        cTplValGesInit< int > & SzRecouvrtDalles();
        const cTplValGesInit< int > & SzRecouvrtDalles()const ;

        cTplValGesInit< int > & SzDalleMin();
        const cTplValGesInit< int > & SzDalleMin()const ;

        cTplValGesInit< int > & SzDalleMax();
        const cTplValGesInit< int > & SzDalleMax()const ;

        cTplValGesInit< double > & NbCelluleMax();
        const cTplValGesInit< double > & NbCelluleMax()const ;

        cTplValGesInit< int > & SzMinDecomposCalc();
        const cTplValGesInit< int > & SzMinDecomposCalc()const ;

        cTplValGesInit< bool > & AutorizeSplitRec();
        const cTplValGesInit< bool > & AutorizeSplitRec()const ;

        cTplValGesInit< int > & DefTileFile();
        const cTplValGesInit< int > & DefTileFile()const ;

        cTplValGesInit< double > & NbPixDefFilesAux();
        const cTplValGesInit< double > & NbPixDefFilesAux()const ;

        cTplValGesInit< int > & DeZoomDefMinFileAux();
        const cTplValGesInit< int > & DeZoomDefMinFileAux()const ;

        cTplValGesInit< int > & FirstEtapeMEC();
        const cTplValGesInit< int > & FirstEtapeMEC()const ;

        cTplValGesInit< int > & LastEtapeMEC();
        const cTplValGesInit< int > & LastEtapeMEC()const ;

        cTplValGesInit< int > & FirstBoiteMEC();
        const cTplValGesInit< int > & FirstBoiteMEC()const ;

        cTplValGesInit< int > & NbBoitesMEC();
        const cTplValGesInit< int > & NbBoitesMEC()const ;

        cTplValGesInit< std::string > & NomChantier();
        const cTplValGesInit< std::string > & NomChantier()const ;

        std::string & PatternSelChantier();
        const std::string & PatternSelChantier()const ;

        std::string & PatNameChantier();
        const std::string & PatNameChantier()const ;

        cTplValGesInit< std::string > & SeparateurChantier();
        const cTplValGesInit< std::string > & SeparateurChantier()const ;

        cTplValGesInit< cCalcNomChantier > & CalcNomChantier();
        const cTplValGesInit< cCalcNomChantier > & CalcNomChantier()const ;

        cTplValGesInit< std::string > & PatternSelPyr();
        const cTplValGesInit< std::string > & PatternSelPyr()const ;

        cTplValGesInit< std::string > & PatternNomPyr();
        const cTplValGesInit< std::string > & PatternNomPyr()const ;

        cTplValGesInit< std::string > & SeparateurPyr();
        const cTplValGesInit< std::string > & SeparateurPyr()const ;

        cTplValGesInit< std::string > & KeyCalNamePyr();
        const cTplValGesInit< std::string > & KeyCalNamePyr()const ;

        cTplValGesInit< bool > & ActivePurge();
        const cTplValGesInit< bool > & ActivePurge()const ;

        std::list< cPurgeFiles > & PurgeFiles();
        const std::list< cPurgeFiles > & PurgeFiles()const ;

        cTplValGesInit< bool > & PurgeMECResultBefore();
        const cTplValGesInit< bool > & PurgeMECResultBefore()const ;

        cTplValGesInit< bool > & UseChantierNameDescripteur();
        const cTplValGesInit< bool > & UseChantierNameDescripteur()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        cTplValGesInit< cCmdMappeur > & MapMicMac();
        const cTplValGesInit< cCmdMappeur > & MapMicMac()const ;

        cTplValGesInit< cCmdExePar > & PostProcess();
        const cTplValGesInit< cCmdExePar > & PostProcess()const ;

        cTplValGesInit< eComprTiff > & ComprMasque();
        const cTplValGesInit< eComprTiff > & ComprMasque()const ;

        cTplValGesInit< NS_ParamChantierPhotogram::eTypeNumerique > & TypeMasque();
        const cTplValGesInit< NS_ParamChantierPhotogram::eTypeNumerique > & TypeMasque()const ;
    private:
        cTplValGesInit< std::string > mFileExportApero2MM;
        cTplValGesInit< bool > mUseProfInVertLoc;
        cTplValGesInit< std::string > mNameFileParamMICMAC;
        std::string mWorkDir;
        cTplValGesInit< std::string > mDirImagesOri;
        std::string mTmpMEC;
        cTplValGesInit< std::string > mTmpPyr;
        cTplValGesInit< std::string > mTmpGeom;
        cTplValGesInit< std::string > mTmpResult;
        cTplValGesInit< bool > mCalledByProcess;
        cTplValGesInit< bool > mVisu;
        cTplValGesInit< int > mByProcess;
        cTplValGesInit< bool > mStopOnEchecFils;
        cTplValGesInit< int > mAvalaibleMemory;
        cTplValGesInit< int > mSzRecouvrtDalles;
        cTplValGesInit< int > mSzDalleMin;
        cTplValGesInit< int > mSzDalleMax;
        cTplValGesInit< double > mNbCelluleMax;
        cTplValGesInit< int > mSzMinDecomposCalc;
        cTplValGesInit< bool > mAutorizeSplitRec;
        cTplValGesInit< int > mDefTileFile;
        cTplValGesInit< double > mNbPixDefFilesAux;
        cTplValGesInit< int > mDeZoomDefMinFileAux;
        cTplValGesInit< int > mFirstEtapeMEC;
        cTplValGesInit< int > mLastEtapeMEC;
        cTplValGesInit< int > mFirstBoiteMEC;
        cTplValGesInit< int > mNbBoitesMEC;
        cTplValGesInit< std::string > mNomChantier;
        cTplValGesInit< cCalcNomChantier > mCalcNomChantier;
        cTplValGesInit< std::string > mPatternSelPyr;
        cTplValGesInit< std::string > mPatternNomPyr;
        cTplValGesInit< std::string > mSeparateurPyr;
        cTplValGesInit< std::string > mKeyCalNamePyr;
        cTplValGesInit< bool > mActivePurge;
        std::list< cPurgeFiles > mPurgeFiles;
        cTplValGesInit< bool > mPurgeMECResultBefore;
        cTplValGesInit< bool > mUseChantierNameDescripteur;
        cTplValGesInit< string > mFileChantierNameDescripteur;
        cTplValGesInit< cCmdMappeur > mMapMicMac;
        cTplValGesInit< cCmdExePar > mPostProcess;
        cTplValGesInit< eComprTiff > mComprMasque;
        cTplValGesInit< NS_ParamChantierPhotogram::eTypeNumerique > mTypeMasque;
};
cElXMLTree * ToXMLTree(const cSection_WorkSpace &);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneBatch
{
    public:
        friend void xml_init(cOneBatch & anObj,cElXMLTree * aTree);


        std::string & PatternSelImBatch();
        const std::string & PatternSelImBatch()const ;

        std::list< std::string > & PatternCommandeBatch();
        const std::list< std::string > & PatternCommandeBatch()const ;
    private:
        std::string mPatternSelImBatch;
        std::list< std::string > mPatternCommandeBatch;
};
cElXMLTree * ToXMLTree(const cOneBatch &);

class cSectionBatch
{
    public:
        friend void xml_init(cSectionBatch & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & ExeBatch();
        const cTplValGesInit< bool > & ExeBatch()const ;

        std::list< cOneBatch > & OneBatch();
        const std::list< cOneBatch > & OneBatch()const ;

        std::list< std::string > & NextMicMacFile2Exec();
        const std::list< std::string > & NextMicMacFile2Exec()const ;
    private:
        cTplValGesInit< bool > mExeBatch;
        std::list< cOneBatch > mOneBatch;
        std::list< std::string > mNextMicMacFile2Exec;
};
cElXMLTree * ToXMLTree(const cSectionBatch &);

/******************************************************/
/******************************************************/
/******************************************************/
class cListTestCpleHomol
{
    public:
        friend void xml_init(cListTestCpleHomol & anObj,cElXMLTree * aTree);


        Pt2dr & PtIm1();
        const Pt2dr & PtIm1()const ;

        Pt2dr & PtIm2();
        const Pt2dr & PtIm2()const ;
    private:
        Pt2dr mPtIm1;
        Pt2dr mPtIm2;
};
cElXMLTree * ToXMLTree(const cListTestCpleHomol &);

class cDebugEscalier
{
    public:
        friend void xml_init(cDebugEscalier & anObj,cElXMLTree * aTree);


        Pt2di & P1();
        const Pt2di & P1()const ;

        Pt2di & P2();
        const Pt2di & P2()const ;

        cTplValGesInit< bool > & ShowDerivZ();
        const cTplValGesInit< bool > & ShowDerivZ()const ;
    private:
        Pt2di mP1;
        Pt2di mP2;
        cTplValGesInit< bool > mShowDerivZ;
};
cElXMLTree * ToXMLTree(const cDebugEscalier &);

class cSectionDebug
{
    public:
        friend void xml_init(cSectionDebug & anObj,cElXMLTree * aTree);


        Pt2di & P1();
        const Pt2di & P1()const ;

        Pt2di & P2();
        const Pt2di & P2()const ;

        cTplValGesInit< bool > & ShowDerivZ();
        const cTplValGesInit< bool > & ShowDerivZ()const ;

        cTplValGesInit< cDebugEscalier > & DebugEscalier();
        const cTplValGesInit< cDebugEscalier > & DebugEscalier()const ;
    private:
        cTplValGesInit< cDebugEscalier > mDebugEscalier;
};
cElXMLTree * ToXMLTree(const cSectionDebug &);

class cSection_Vrac
{
    public:
        friend void xml_init(cSection_Vrac & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & SL_XSzW();
        const cTplValGesInit< int > & SL_XSzW()const ;

        cTplValGesInit< int > & SL_YSzW();
        const cTplValGesInit< int > & SL_YSzW()const ;

        cTplValGesInit< bool > & SL_Epip();
        const cTplValGesInit< bool > & SL_Epip()const ;

        cTplValGesInit< int > & SL_YDecEpip();
        const cTplValGesInit< int > & SL_YDecEpip()const ;

        cTplValGesInit< std::string > & SL_PackHom0();
        const cTplValGesInit< std::string > & SL_PackHom0()const ;

        cTplValGesInit< bool > & SL_RedrOnCur();
        const cTplValGesInit< bool > & SL_RedrOnCur()const ;

        cTplValGesInit< bool > & SL_NewRedrCur();
        const cTplValGesInit< bool > & SL_NewRedrCur()const ;

        cTplValGesInit< std::vector<std::string> > & SL_FILTER();
        const cTplValGesInit< std::vector<std::string> > & SL_FILTER()const ;

        cTplValGesInit< bool > & SL_TJS_FILTER();
        const cTplValGesInit< bool > & SL_TJS_FILTER()const ;

        cTplValGesInit< double > & SL_Step_Grid();
        const cTplValGesInit< double > & SL_Step_Grid()const ;

        cTplValGesInit< std::string > & SL_Name_Grid_Exp();
        const cTplValGesInit< std::string > & SL_Name_Grid_Exp()const ;

        cTplValGesInit< double > & VSG_DynImRed();
        const cTplValGesInit< double > & VSG_DynImRed()const ;

        cTplValGesInit< int > & VSG_DeZoomContr();
        const cTplValGesInit< int > & VSG_DeZoomContr()const ;

        cTplValGesInit< Pt2di > & PtDebug();
        const cTplValGesInit< Pt2di > & PtDebug()const ;

        cTplValGesInit< bool > & DumpNappesEnglob();
        const cTplValGesInit< bool > & DumpNappesEnglob()const ;

        cTplValGesInit< bool > & InterditAccelerationCorrSpec();
        const cTplValGesInit< bool > & InterditAccelerationCorrSpec()const ;

        cTplValGesInit< bool > & InterditCorrelRapide();
        const cTplValGesInit< bool > & InterditCorrelRapide()const ;

        cTplValGesInit< bool > & ForceCorrelationByRect();
        const cTplValGesInit< bool > & ForceCorrelationByRect()const ;

        std::list< cListTestCpleHomol > & ListTestCpleHomol();
        const std::list< cListTestCpleHomol > & ListTestCpleHomol()const ;

        std::list< Pt3dr > & ListeTestPointsTerrain();
        const std::list< Pt3dr > & ListeTestPointsTerrain()const ;

        cTplValGesInit< bool > & WithMessage();
        const cTplValGesInit< bool > & WithMessage()const ;

        cTplValGesInit< bool > & ShowLoadedImage();
        const cTplValGesInit< bool > & ShowLoadedImage()const ;

        Pt2di & P1();
        const Pt2di & P1()const ;

        Pt2di & P2();
        const Pt2di & P2()const ;

        cTplValGesInit< bool > & ShowDerivZ();
        const cTplValGesInit< bool > & ShowDerivZ()const ;

        cTplValGesInit< cDebugEscalier > & DebugEscalier();
        const cTplValGesInit< cDebugEscalier > & DebugEscalier()const ;

        cTplValGesInit< cSectionDebug > & SectionDebug();
        const cTplValGesInit< cSectionDebug > & SectionDebug()const ;
    private:
        cTplValGesInit< int > mSL_XSzW;
        cTplValGesInit< int > mSL_YSzW;
        cTplValGesInit< bool > mSL_Epip;
        cTplValGesInit< int > mSL_YDecEpip;
        cTplValGesInit< std::string > mSL_PackHom0;
        cTplValGesInit< bool > mSL_RedrOnCur;
        cTplValGesInit< bool > mSL_NewRedrCur;
        cTplValGesInit< std::vector<std::string> > mSL_FILTER;
        cTplValGesInit< bool > mSL_TJS_FILTER;
        cTplValGesInit< double > mSL_Step_Grid;
        cTplValGesInit< std::string > mSL_Name_Grid_Exp;
        cTplValGesInit< double > mVSG_DynImRed;
        cTplValGesInit< int > mVSG_DeZoomContr;
        cTplValGesInit< Pt2di > mPtDebug;
        cTplValGesInit< bool > mDumpNappesEnglob;
        cTplValGesInit< bool > mInterditAccelerationCorrSpec;
        cTplValGesInit< bool > mInterditCorrelRapide;
        cTplValGesInit< bool > mForceCorrelationByRect;
        std::list< cListTestCpleHomol > mListTestCpleHomol;
        std::list< Pt3dr > mListeTestPointsTerrain;
        cTplValGesInit< bool > mWithMessage;
        cTplValGesInit< bool > mShowLoadedImage;
        cTplValGesInit< cSectionDebug > mSectionDebug;
};
cElXMLTree * ToXMLTree(const cSection_Vrac &);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamMICMAC
{
    public:
        friend void xml_init(cParamMICMAC & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        cTplValGesInit< double > & RatioAltiPlani();
        const cTplValGesInit< double > & RatioAltiPlani()const ;

        cTplValGesInit< bool > & EstimPxPrefZ2Prof();
        const cTplValGesInit< bool > & EstimPxPrefZ2Prof()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        double & ZIncCalc();
        const double & ZIncCalc()const ;

        cTplValGesInit< bool > & ZIncIsProp();
        const cTplValGesInit< bool > & ZIncIsProp()const ;

        cTplValGesInit< double > & ZIncZonage();
        const cTplValGesInit< double > & ZIncZonage()const ;

        std::string & MNT_Init_Image();
        const std::string & MNT_Init_Image()const ;

        std::string & MNT_Init_Xml();
        const std::string & MNT_Init_Xml()const ;

        cTplValGesInit< double > & MNT_Offset();
        const cTplValGesInit< double > & MNT_Offset()const ;

        cTplValGesInit< cMNT_Init > & MNT_Init();
        const cTplValGesInit< cMNT_Init > & MNT_Init()const ;

        std::string & ZInf();
        const std::string & ZInf()const ;

        std::string & ZSup();
        const std::string & ZSup()const ;

        cTplValGesInit< cEnveloppeMNT_INIT > & EnveloppeMNT_INIT();
        const cTplValGesInit< cEnveloppeMNT_INIT > & EnveloppeMNT_INIT()const ;

        cTplValGesInit< cIntervAltimetrie > & IntervAltimetrie();
        const cTplValGesInit< cIntervAltimetrie > & IntervAltimetrie()const ;

        cTplValGesInit< double > & Px1Moy();
        const cTplValGesInit< double > & Px1Moy()const ;

        cTplValGesInit< double > & Px2Moy();
        const cTplValGesInit< double > & Px2Moy()const ;

        double & Px1IncCalc();
        const double & Px1IncCalc()const ;

        cTplValGesInit< double > & Px1PropProf();
        const cTplValGesInit< double > & Px1PropProf()const ;

        cTplValGesInit< double > & Px2IncCalc();
        const cTplValGesInit< double > & Px2IncCalc()const ;

        cTplValGesInit< double > & Px1IncZonage();
        const cTplValGesInit< double > & Px1IncZonage()const ;

        cTplValGesInit< double > & Px2IncZonage();
        const cTplValGesInit< double > & Px2IncZonage()const ;

        cTplValGesInit< cIntervParalaxe > & IntervParalaxe();
        const cTplValGesInit< cIntervParalaxe > & IntervParalaxe()const ;

        double & MulZMin();
        const double & MulZMin()const ;

        double & MulZMax();
        const double & MulZMax()const ;

        cTplValGesInit< cIntervSpecialZInv > & IntervSpecialZInv();
        const cTplValGesInit< cIntervSpecialZInv > & IntervSpecialZInv()const ;

        cTplValGesInit< Box2dr > & BoxTerrain();
        const cTplValGesInit< Box2dr > & BoxTerrain()const ;

        std::list< cListePointsInclus > & ListePointsInclus();
        const std::list< cListePointsInclus > & ListePointsInclus()const ;

        cTplValGesInit< double > & RatioResolImage();
        const cTplValGesInit< double > & RatioResolImage()const ;

        cTplValGesInit< double > & ResolutionTerrain();
        const cTplValGesInit< double > & ResolutionTerrain()const ;

        cTplValGesInit< std::string > & FilterEstimTerrain();
        const cTplValGesInit< std::string > & FilterEstimTerrain()const ;

        cTplValGesInit< std::string > & FileBoxMasqIsBoxTer();
        const cTplValGesInit< std::string > & FileBoxMasqIsBoxTer()const ;

        std::string & MT_Image();
        const std::string & MT_Image()const ;

        std::string & MT_Xml();
        const std::string & MT_Xml()const ;

        cTplValGesInit< cMasqueTerrain > & MasqueTerrain();
        const cTplValGesInit< cMasqueTerrain > & MasqueTerrain()const ;

        cTplValGesInit< double > & RecouvrementMinimal();
        const cTplValGesInit< double > & RecouvrementMinimal()const ;

        cTplValGesInit< cPlanimetrie > & Planimetrie();
        const cTplValGesInit< cPlanimetrie > & Planimetrie()const ;

        cTplValGesInit< std::string > & FileOriMnt();
        const cTplValGesInit< std::string > & FileOriMnt()const ;

        cTplValGesInit< double > & EnergieExpCorrel();
        const cTplValGesInit< double > & EnergieExpCorrel()const ;

        cTplValGesInit< double > & EnergieExpRegulPlani();
        const cTplValGesInit< double > & EnergieExpRegulPlani()const ;

        cTplValGesInit< double > & EnergieExpRegulAlti();
        const cTplValGesInit< double > & EnergieExpRegulAlti()const ;

        cTplValGesInit< cRugositeMNT > & RugositeMNT();
        const cTplValGesInit< cRugositeMNT > & RugositeMNT()const ;

        cSection_Terrain & Section_Terrain();
        const cSection_Terrain & Section_Terrain()const ;

        cTplValGesInit< int > & BordImage();
        const cTplValGesInit< int > & BordImage()const ;

        cTplValGesInit< bool > & ConvertToSameOriPtTgtLoc();
        const cTplValGesInit< bool > & ConvertToSameOriPtTgtLoc()const ;

        cTplValGesInit< int > & ValSpecNotImage();
        const cTplValGesInit< int > & ValSpecNotImage()const ;

        cTplValGesInit< std::string > & PrefixMasqImRes();
        const cTplValGesInit< std::string > & PrefixMasqImRes()const ;

        cTplValGesInit< std::string > & DirMasqueImages();
        const cTplValGesInit< std::string > & DirMasqueImages()const ;

        std::list< cMasqImageIn > & MasqImageIn();
        const std::list< cMasqImageIn > & MasqImageIn()const ;

        std::list< cSpecFitrageImage > & FiltreImageIn();
        const std::list< cSpecFitrageImage > & FiltreImageIn()const ;

        eModeGeomImage & GeomImages();
        const eModeGeomImage & GeomImages()const ;

        std::string & NomModule();
        const std::string & NomModule()const ;

        std::string & NomGeometrie();
        const std::string & NomGeometrie()const ;

        cTplValGesInit< cModuleGeomImage > & ModuleGeomImage();
        const cTplValGesInit< cModuleGeomImage > & ModuleGeomImage()const ;

        cTplValGesInit< std::string > & Im1();
        const cTplValGesInit< std::string > & Im1()const ;

        cTplValGesInit< std::string > & Im2();
        const cTplValGesInit< std::string > & Im2()const ;

        std::string & I2FromI1Key();
        const std::string & I2FromI1Key()const ;

        bool & I2FromI1SensDirect();
        const bool & I2FromI1SensDirect()const ;

        cTplValGesInit< cFCND_CalcIm2fromIm1 > & FCND_CalcIm2fromIm1();
        const cTplValGesInit< cFCND_CalcIm2fromIm1 > & FCND_CalcIm2fromIm1()const ;

        std::list< std::string > & ImPat();
        const std::list< std::string > & ImPat()const ;

        cTplValGesInit< cNameFilter > & Filter();
        const cTplValGesInit< cNameFilter > & Filter()const ;

        double & RecouvrMin();
        const double & RecouvrMin()const ;

        cTplValGesInit< cAutoSelectionneImSec > & AutoSelectionneImSec();
        const cTplValGesInit< cAutoSelectionneImSec > & AutoSelectionneImSec()const ;

        cTplValGesInit< cListImByDelta > & ImSecByDelta();
        const cTplValGesInit< cListImByDelta > & ImSecByDelta()const ;

        cTplValGesInit< std::string > & Im3Superp();
        const cTplValGesInit< std::string > & Im3Superp()const ;

        cImages & Images();
        const cImages & Images()const ;

        std::list< cNomsGeometrieImage > & NomsGeometrieImage();
        const std::list< cNomsGeometrieImage > & NomsGeometrieImage()const ;

        std::string & PatternSel();
        const std::string & PatternSel()const ;

        std::string & PatNameGeom();
        const std::string & PatNameGeom()const ;

        cTplValGesInit< std::string > & SeparateurHom();
        const cTplValGesInit< std::string > & SeparateurHom()const ;

        cTplValGesInit< cNomsHomomologues > & NomsHomomologues();
        const cTplValGesInit< cNomsHomomologues > & NomsHomomologues()const ;

        cTplValGesInit< std::string > & FCND_CalcHomFromI1I2();
        const cTplValGesInit< std::string > & FCND_CalcHomFromI1I2()const ;

        cTplValGesInit< bool > & SingulariteInCorresp_I1I2();
        const cTplValGesInit< bool > & SingulariteInCorresp_I1I2()const ;

        cTplValGesInit< cMapName2Name > & ClassEquivalenceImage();
        const cTplValGesInit< cMapName2Name > & ClassEquivalenceImage()const ;

        cSection_PriseDeVue & Section_PriseDeVue();
        const cSection_PriseDeVue & Section_PriseDeVue()const ;

        cTplValGesInit< bool > & PasIsInPixel();
        const cTplValGesInit< bool > & PasIsInPixel()const ;

        cTplValGesInit< Box2dr > & ProportionClipMEC();
        const cTplValGesInit< Box2dr > & ProportionClipMEC()const ;

        cTplValGesInit< bool > & ClipMecIsProp();
        const cTplValGesInit< bool > & ClipMecIsProp()const ;

        cTplValGesInit< double > & ZoomClipMEC();
        const cTplValGesInit< double > & ZoomClipMEC()const ;

        cTplValGesInit< int > & NbMinImagesVisibles();
        const cTplValGesInit< int > & NbMinImagesVisibles()const ;

        cTplValGesInit< bool > & OneDefCorAllPxDefCor();
        const cTplValGesInit< bool > & OneDefCorAllPxDefCor()const ;

        cTplValGesInit< int > & ZoomBeginODC_APDC();
        const cTplValGesInit< int > & ZoomBeginODC_APDC()const ;

        cTplValGesInit< double > & DefCorrelation();
        const cTplValGesInit< double > & DefCorrelation()const ;

        cTplValGesInit< bool > & ReprojPixelNoVal();
        const cTplValGesInit< bool > & ReprojPixelNoVal()const ;

        cTplValGesInit< double > & EpsilonCorrelation();
        const cTplValGesInit< double > & EpsilonCorrelation()const ;

        int & FreqEchantPtsI();
        const int & FreqEchantPtsI()const ;

        eTypeModeEchantPtsI & ModeEchantPtsI();
        const eTypeModeEchantPtsI & ModeEchantPtsI()const ;

        cTplValGesInit< std::string > & KeyCommandeExterneInteret();
        const cTplValGesInit< std::string > & KeyCommandeExterneInteret()const ;

        cTplValGesInit< int > & SzVAutoCorrel();
        const cTplValGesInit< int > & SzVAutoCorrel()const ;

        cTplValGesInit< double > & EstmBrAutoCorrel();
        const cTplValGesInit< double > & EstmBrAutoCorrel()const ;

        cTplValGesInit< double > & SeuilLambdaAutoCorrel();
        const cTplValGesInit< double > & SeuilLambdaAutoCorrel()const ;

        cTplValGesInit< double > & SeuilEcartTypeAutoCorrel();
        const cTplValGesInit< double > & SeuilEcartTypeAutoCorrel()const ;

        cTplValGesInit< double > & RepartExclusion();
        const cTplValGesInit< double > & RepartExclusion()const ;

        cTplValGesInit< double > & RepartEvitement();
        const cTplValGesInit< double > & RepartEvitement()const ;

        cTplValGesInit< cEchantillonagePtsInterets > & EchantillonagePtsInterets();
        const cTplValGesInit< cEchantillonagePtsInterets > & EchantillonagePtsInterets()const ;

        cTplValGesInit< bool > & ChantierFullImage1();
        const cTplValGesInit< bool > & ChantierFullImage1()const ;

        cTplValGesInit< bool > & ChantierFullMaskImage1();
        const cTplValGesInit< bool > & ChantierFullMaskImage1()const ;

        cTplValGesInit< bool > & ExportForMultiplePointsHomologues();
        const cTplValGesInit< bool > & ExportForMultiplePointsHomologues()const ;

        cTplValGesInit< double > & CovLim();
        const cTplValGesInit< double > & CovLim()const ;

        cTplValGesInit< double > & TermeDecr();
        const cTplValGesInit< double > & TermeDecr()const ;

        cTplValGesInit< int > & SzRef();
        const cTplValGesInit< int > & SzRef()const ;

        cTplValGesInit< double > & ValRef();
        const cTplValGesInit< double > & ValRef()const ;

        cTplValGesInit< cAdapteDynCov > & AdapteDynCov();
        const cTplValGesInit< cAdapteDynCov > & AdapteDynCov()const ;

        std::list< cEtapeMEC > & EtapeMEC();
        const std::list< cEtapeMEC > & EtapeMEC()const ;

        std::list< cTypePyramImage > & TypePyramImage();
        const std::list< cTypePyramImage > & TypePyramImage()const ;

        cTplValGesInit< bool > & Correl16Bits();
        const cTplValGesInit< bool > & Correl16Bits()const ;

        cSection_MEC & Section_MEC();
        const cSection_MEC & Section_MEC()const ;

        cTplValGesInit< bool > & Use_MM_EtatAvancement();
        const cTplValGesInit< bool > & Use_MM_EtatAvancement()const ;

        cTplValGesInit< bool > & ButDoPyram();
        const cTplValGesInit< bool > & ButDoPyram()const ;

        cTplValGesInit< bool > & ButDoMasqIm();
        const cTplValGesInit< bool > & ButDoMasqIm()const ;

        cTplValGesInit< bool > & ButDoMemPart();
        const cTplValGesInit< bool > & ButDoMemPart()const ;

        cTplValGesInit< bool > & ButDoTA();
        const cTplValGesInit< bool > & ButDoTA()const ;

        cTplValGesInit< bool > & ButDoMasqueChantier();
        const cTplValGesInit< bool > & ButDoMasqueChantier()const ;

        cTplValGesInit< bool > & ButDoOriMNT();
        const cTplValGesInit< bool > & ButDoOriMNT()const ;

        cTplValGesInit< bool > & ButDoFDC();
        const cTplValGesInit< bool > & ButDoFDC()const ;

        cTplValGesInit< bool > & ButDoExtendParam();
        const cTplValGesInit< bool > & ButDoExtendParam()const ;

        cTplValGesInit< bool > & ButDoGenCorPxTransv();
        const cTplValGesInit< bool > & ButDoGenCorPxTransv()const ;

        cTplValGesInit< bool > & ButDoPartiesCachees();
        const cTplValGesInit< bool > & ButDoPartiesCachees()const ;

        cTplValGesInit< bool > & ButDoOrtho();
        const cTplValGesInit< bool > & ButDoOrtho()const ;

        cTplValGesInit< bool > & ButDoSimul();
        const cTplValGesInit< bool > & ButDoSimul()const ;

        cTplValGesInit< bool > & ButDoRedrLocAnam();
        const cTplValGesInit< bool > & ButDoRedrLocAnam()const ;

        cTplValGesInit< cDoNothingBut > & DoNothingBut();
        const cTplValGesInit< cDoNothingBut > & DoNothingBut()const ;

        cTplValGesInit< int > & Paral_Pc_IdProcess();
        const cTplValGesInit< int > & Paral_Pc_IdProcess()const ;

        cTplValGesInit< int > & Paral_Pc_NbProcess();
        const cTplValGesInit< int > & Paral_Pc_NbProcess()const ;

        cTplValGesInit< double > & X_DirPlanInterFaisceau();
        const cTplValGesInit< double > & X_DirPlanInterFaisceau()const ;

        cTplValGesInit< double > & Y_DirPlanInterFaisceau();
        const cTplValGesInit< double > & Y_DirPlanInterFaisceau()const ;

        cTplValGesInit< double > & Z_DirPlanInterFaisceau();
        const cTplValGesInit< double > & Z_DirPlanInterFaisceau()const ;

        eModeGeomMNT & GeomMNT();
        const eModeGeomMNT & GeomMNT()const ;

        cTplValGesInit< cSectionSimulation > & SectionSimulation();
        const cTplValGesInit< cSectionSimulation > & SectionSimulation()const ;

        cTplValGesInit< bool > & Prio2OwnAltisolForEmprise();
        const cTplValGesInit< bool > & Prio2OwnAltisolForEmprise()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< cAnamSurfaceAnalytique > & AnamSurfaceAnalytique();
        const cTplValGesInit< cAnamSurfaceAnalytique > & AnamSurfaceAnalytique()const ;

        cTplValGesInit< int > & AnamDeZoomMasq();
        const cTplValGesInit< int > & AnamDeZoomMasq()const ;

        cTplValGesInit< double > & AnamLimAngleVisib();
        const cTplValGesInit< double > & AnamLimAngleVisib()const ;

        cTplValGesInit< cAnamorphoseGeometrieMNT > & AnamorphoseGeometrieMNT();
        const cTplValGesInit< cAnamorphoseGeometrieMNT > & AnamorphoseGeometrieMNT()const ;

        cTplValGesInit< std::string > & RepereCorrel();
        const cTplValGesInit< std::string > & RepereCorrel()const ;

        cTplValGesInit< std::string > & TagRepereCorrel();
        const cTplValGesInit< std::string > & TagRepereCorrel()const ;

        cTplValGesInit< bool > & DoMEC();
        const cTplValGesInit< bool > & DoMEC()const ;

        cTplValGesInit< bool > & DoFDC();
        const cTplValGesInit< bool > & DoFDC()const ;

        cTplValGesInit< bool > & GenereXMLComp();
        const cTplValGesInit< bool > & GenereXMLComp()const ;

        cTplValGesInit< int > & ZoomMakeTA();
        const cTplValGesInit< int > & ZoomMakeTA()const ;

        cTplValGesInit< double > & SaturationTA();
        const cTplValGesInit< double > & SaturationTA()const ;

        cTplValGesInit< bool > & OrthoTA();
        const cTplValGesInit< bool > & OrthoTA()const ;

        cTplValGesInit< int > & ZoomMakeMasq();
        const cTplValGesInit< int > & ZoomMakeMasq()const ;

        cTplValGesInit< bool > & MakeImCptTA();
        const cTplValGesInit< bool > & MakeImCptTA()const ;

        cTplValGesInit< std::string > & FilterTA();
        const cTplValGesInit< std::string > & FilterTA()const ;

        cTplValGesInit< double > & GammaVisu();
        const cTplValGesInit< double > & GammaVisu()const ;

        cTplValGesInit< int > & ZoomVisuLiaison();
        const cTplValGesInit< int > & ZoomVisuLiaison()const ;

        cTplValGesInit< double > & TolerancePointHomInImage();
        const cTplValGesInit< double > & TolerancePointHomInImage()const ;

        cTplValGesInit< double > & FiltragePointHomInImage();
        const cTplValGesInit< double > & FiltragePointHomInImage()const ;

        cTplValGesInit< int > & BaseCodeRetourMicmacErreur();
        const cTplValGesInit< int > & BaseCodeRetourMicmacErreur()const ;

        Pt3di & OrdreChannels();
        const Pt3di & OrdreChannels()const ;

        cTplValGesInit< Pt2di > & PtBalanceBlancs();
        const cTplValGesInit< Pt2di > & PtBalanceBlancs()const ;

        cTplValGesInit< Pt2di > & P0Sup();
        const cTplValGesInit< Pt2di > & P0Sup()const ;

        cTplValGesInit< Pt2di > & SzSup();
        const cTplValGesInit< Pt2di > & SzSup()const ;

        cElRegex_Ptr & PatternSelGrid();
        const cElRegex_Ptr & PatternSelGrid()const ;

        std::string & PatternNameGrid();
        const std::string & PatternNameGrid()const ;

        std::list< cColorimetriesCanaux > & ColorimetriesCanaux();
        const std::list< cColorimetriesCanaux > & ColorimetriesCanaux()const ;

        cTplValGesInit< double > & GammaCorrection();
        const cTplValGesInit< double > & GammaCorrection()const ;

        cTplValGesInit< double > & MultiplicateurBlanc();
        const cTplValGesInit< double > & MultiplicateurBlanc()const ;

        cTplValGesInit< bool > & GenFileImages();
        const cTplValGesInit< bool > & GenFileImages()const ;

        cTplValGesInit< cSuperpositionImages > & SuperpositionImages();
        const cTplValGesInit< cSuperpositionImages > & SuperpositionImages()const ;

        cSection_Results & Section_Results();
        const cSection_Results & Section_Results()const ;

        cTplValGesInit< std::string > & FileExportApero2MM();
        const cTplValGesInit< std::string > & FileExportApero2MM()const ;

        cTplValGesInit< bool > & UseProfInVertLoc();
        const cTplValGesInit< bool > & UseProfInVertLoc()const ;

        cTplValGesInit< std::string > & NameFileParamMICMAC();
        const cTplValGesInit< std::string > & NameFileParamMICMAC()const ;

        std::string & WorkDir();
        const std::string & WorkDir()const ;

        cTplValGesInit< std::string > & DirImagesOri();
        const cTplValGesInit< std::string > & DirImagesOri()const ;

        std::string & TmpMEC();
        const std::string & TmpMEC()const ;

        cTplValGesInit< std::string > & TmpPyr();
        const cTplValGesInit< std::string > & TmpPyr()const ;

        cTplValGesInit< std::string > & TmpGeom();
        const cTplValGesInit< std::string > & TmpGeom()const ;

        cTplValGesInit< std::string > & TmpResult();
        const cTplValGesInit< std::string > & TmpResult()const ;

        cTplValGesInit< bool > & CalledByProcess();
        const cTplValGesInit< bool > & CalledByProcess()const ;

        cTplValGesInit< bool > & Visu();
        const cTplValGesInit< bool > & Visu()const ;

        cTplValGesInit< int > & ByProcess();
        const cTplValGesInit< int > & ByProcess()const ;

        cTplValGesInit< bool > & StopOnEchecFils();
        const cTplValGesInit< bool > & StopOnEchecFils()const ;

        cTplValGesInit< int > & AvalaibleMemory();
        const cTplValGesInit< int > & AvalaibleMemory()const ;

        cTplValGesInit< int > & SzRecouvrtDalles();
        const cTplValGesInit< int > & SzRecouvrtDalles()const ;

        cTplValGesInit< int > & SzDalleMin();
        const cTplValGesInit< int > & SzDalleMin()const ;

        cTplValGesInit< int > & SzDalleMax();
        const cTplValGesInit< int > & SzDalleMax()const ;

        cTplValGesInit< double > & NbCelluleMax();
        const cTplValGesInit< double > & NbCelluleMax()const ;

        cTplValGesInit< int > & SzMinDecomposCalc();
        const cTplValGesInit< int > & SzMinDecomposCalc()const ;

        cTplValGesInit< bool > & AutorizeSplitRec();
        const cTplValGesInit< bool > & AutorizeSplitRec()const ;

        cTplValGesInit< int > & DefTileFile();
        const cTplValGesInit< int > & DefTileFile()const ;

        cTplValGesInit< double > & NbPixDefFilesAux();
        const cTplValGesInit< double > & NbPixDefFilesAux()const ;

        cTplValGesInit< int > & DeZoomDefMinFileAux();
        const cTplValGesInit< int > & DeZoomDefMinFileAux()const ;

        cTplValGesInit< int > & FirstEtapeMEC();
        const cTplValGesInit< int > & FirstEtapeMEC()const ;

        cTplValGesInit< int > & LastEtapeMEC();
        const cTplValGesInit< int > & LastEtapeMEC()const ;

        cTplValGesInit< int > & FirstBoiteMEC();
        const cTplValGesInit< int > & FirstBoiteMEC()const ;

        cTplValGesInit< int > & NbBoitesMEC();
        const cTplValGesInit< int > & NbBoitesMEC()const ;

        cTplValGesInit< std::string > & NomChantier();
        const cTplValGesInit< std::string > & NomChantier()const ;

        std::string & PatternSelChantier();
        const std::string & PatternSelChantier()const ;

        std::string & PatNameChantier();
        const std::string & PatNameChantier()const ;

        cTplValGesInit< std::string > & SeparateurChantier();
        const cTplValGesInit< std::string > & SeparateurChantier()const ;

        cTplValGesInit< cCalcNomChantier > & CalcNomChantier();
        const cTplValGesInit< cCalcNomChantier > & CalcNomChantier()const ;

        cTplValGesInit< std::string > & PatternSelPyr();
        const cTplValGesInit< std::string > & PatternSelPyr()const ;

        cTplValGesInit< std::string > & PatternNomPyr();
        const cTplValGesInit< std::string > & PatternNomPyr()const ;

        cTplValGesInit< std::string > & SeparateurPyr();
        const cTplValGesInit< std::string > & SeparateurPyr()const ;

        cTplValGesInit< std::string > & KeyCalNamePyr();
        const cTplValGesInit< std::string > & KeyCalNamePyr()const ;

        cTplValGesInit< bool > & ActivePurge();
        const cTplValGesInit< bool > & ActivePurge()const ;

        std::list< cPurgeFiles > & PurgeFiles();
        const std::list< cPurgeFiles > & PurgeFiles()const ;

        cTplValGesInit< bool > & PurgeMECResultBefore();
        const cTplValGesInit< bool > & PurgeMECResultBefore()const ;

        cTplValGesInit< bool > & UseChantierNameDescripteur();
        const cTplValGesInit< bool > & UseChantierNameDescripteur()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        cTplValGesInit< cCmdMappeur > & MapMicMac();
        const cTplValGesInit< cCmdMappeur > & MapMicMac()const ;

        cTplValGesInit< cCmdExePar > & PostProcess();
        const cTplValGesInit< cCmdExePar > & PostProcess()const ;

        cTplValGesInit< eComprTiff > & ComprMasque();
        const cTplValGesInit< eComprTiff > & ComprMasque()const ;

        cTplValGesInit< NS_ParamChantierPhotogram::eTypeNumerique > & TypeMasque();
        const cTplValGesInit< NS_ParamChantierPhotogram::eTypeNumerique > & TypeMasque()const ;

        cSection_WorkSpace & Section_WorkSpace();
        const cSection_WorkSpace & Section_WorkSpace()const ;

        cTplValGesInit< bool > & ExeBatch();
        const cTplValGesInit< bool > & ExeBatch()const ;

        std::list< cOneBatch > & OneBatch();
        const std::list< cOneBatch > & OneBatch()const ;

        std::list< std::string > & NextMicMacFile2Exec();
        const std::list< std::string > & NextMicMacFile2Exec()const ;

        cTplValGesInit< cSectionBatch > & SectionBatch();
        const cTplValGesInit< cSectionBatch > & SectionBatch()const ;

        cTplValGesInit< int > & SL_XSzW();
        const cTplValGesInit< int > & SL_XSzW()const ;

        cTplValGesInit< int > & SL_YSzW();
        const cTplValGesInit< int > & SL_YSzW()const ;

        cTplValGesInit< bool > & SL_Epip();
        const cTplValGesInit< bool > & SL_Epip()const ;

        cTplValGesInit< int > & SL_YDecEpip();
        const cTplValGesInit< int > & SL_YDecEpip()const ;

        cTplValGesInit< std::string > & SL_PackHom0();
        const cTplValGesInit< std::string > & SL_PackHom0()const ;

        cTplValGesInit< bool > & SL_RedrOnCur();
        const cTplValGesInit< bool > & SL_RedrOnCur()const ;

        cTplValGesInit< bool > & SL_NewRedrCur();
        const cTplValGesInit< bool > & SL_NewRedrCur()const ;

        cTplValGesInit< std::vector<std::string> > & SL_FILTER();
        const cTplValGesInit< std::vector<std::string> > & SL_FILTER()const ;

        cTplValGesInit< bool > & SL_TJS_FILTER();
        const cTplValGesInit< bool > & SL_TJS_FILTER()const ;

        cTplValGesInit< double > & SL_Step_Grid();
        const cTplValGesInit< double > & SL_Step_Grid()const ;

        cTplValGesInit< std::string > & SL_Name_Grid_Exp();
        const cTplValGesInit< std::string > & SL_Name_Grid_Exp()const ;

        cTplValGesInit< double > & VSG_DynImRed();
        const cTplValGesInit< double > & VSG_DynImRed()const ;

        cTplValGesInit< int > & VSG_DeZoomContr();
        const cTplValGesInit< int > & VSG_DeZoomContr()const ;

        cTplValGesInit< Pt2di > & PtDebug();
        const cTplValGesInit< Pt2di > & PtDebug()const ;

        cTplValGesInit< bool > & DumpNappesEnglob();
        const cTplValGesInit< bool > & DumpNappesEnglob()const ;

        cTplValGesInit< bool > & InterditAccelerationCorrSpec();
        const cTplValGesInit< bool > & InterditAccelerationCorrSpec()const ;

        cTplValGesInit< bool > & InterditCorrelRapide();
        const cTplValGesInit< bool > & InterditCorrelRapide()const ;

        cTplValGesInit< bool > & ForceCorrelationByRect();
        const cTplValGesInit< bool > & ForceCorrelationByRect()const ;

        std::list< cListTestCpleHomol > & ListTestCpleHomol();
        const std::list< cListTestCpleHomol > & ListTestCpleHomol()const ;

        std::list< Pt3dr > & ListeTestPointsTerrain();
        const std::list< Pt3dr > & ListeTestPointsTerrain()const ;

        cTplValGesInit< bool > & WithMessage();
        const cTplValGesInit< bool > & WithMessage()const ;

        cTplValGesInit< bool > & ShowLoadedImage();
        const cTplValGesInit< bool > & ShowLoadedImage()const ;

        Pt2di & P1();
        const Pt2di & P1()const ;

        Pt2di & P2();
        const Pt2di & P2()const ;

        cTplValGesInit< bool > & ShowDerivZ();
        const cTplValGesInit< bool > & ShowDerivZ()const ;

        cTplValGesInit< cDebugEscalier > & DebugEscalier();
        const cTplValGesInit< cDebugEscalier > & DebugEscalier()const ;

        cTplValGesInit< cSectionDebug > & SectionDebug();
        const cTplValGesInit< cSectionDebug > & SectionDebug()const ;

        cSection_Vrac & Section_Vrac();
        const cSection_Vrac & Section_Vrac()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cSection_Terrain mSection_Terrain;
        cSection_PriseDeVue mSection_PriseDeVue;
        cSection_MEC mSection_MEC;
        cSection_Results mSection_Results;
        cSection_WorkSpace mSection_WorkSpace;
        cTplValGesInit< cSectionBatch > mSectionBatch;
        cSection_Vrac mSection_Vrac;
};
cElXMLTree * ToXMLTree(const cParamMICMAC &);

/******************************************************/
/******************************************************/
/******************************************************/
};
#endif // Define_NotMicMac
