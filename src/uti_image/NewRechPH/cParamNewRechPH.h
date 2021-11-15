#ifndef Define_NotRechNewPH
#define Define_NotRechNewPH
// NOMORE ...
typedef enum
{
  eTPR_LaplMax,
  eTPR_LaplMin,
  eTPR_GrayMax,
  eTPR_GrayMin,
  eTPR_BifurqMax,
  eTPR_BifurqMin,
  eTPR_NoLabel,
  eTPR_GraySadl,
  eTPR_BifurqSadl
} eTypePtRemark;
void xml_init(eTypePtRemark & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypePtRemark & aVal);

eTypePtRemark  Str2eTypePtRemark(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePtRemark & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypePtRemark &);

std::string  Mangling( eTypePtRemark *);

void  BinaryUnDumpFromFile(eTypePtRemark &,ELISE_fp &);

typedef enum
{
  eTVIR_Curve,
  eTVIR_ACR0,
  eTVIR_ACGT,
  eTVIR_ACGR,
  eTVIR_LogPol,
  eTVIR_NoLabel
} eTypeVecInvarR;
void xml_init(eTypeVecInvarR & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeVecInvarR & aVal);

eTypeVecInvarR  Str2eTypeVecInvarR(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeVecInvarR & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeVecInvarR &);

std::string  Mangling( eTypeVecInvarR *);

void  BinaryUnDumpFromFile(eTypeVecInvarR &,ELISE_fp &);

typedef enum
{
  eTIR_Radiom,
  eTIR_GradRad,
  eTIR_GradCroise,
  eTIR_GradTan,
  eTIR_GradTanPiS2,
  eTIR_GradTanPi,
  eTIR_LaplRad,
  eTIR_LaplTan,
  eTIR_LaplCrois,
  eTIR_DiffOpposePi,
  eTIR_DiffOpposePiS2,
  eTIR_Sq_Radiom,
  eTIR_Sq_GradRad,
  eTIR_Sq_GradCroise,
  eTIR_Sq_GradTan,
  eTIR_Sq_GradTanPiS2,
  eTIR_Sq_GradTanPi,
  eTIR_Sq_LaplRad,
  eTIR_Sq_LaplTan,
  eTIR_Sq_LaplCrois,
  eTIR_Sq_DiffOpposePi,
  eTIR_Sq_DiffOpposePiS2,
  eTIR_Cub_Radiom,
  eTIR_Cub_GradRad,
  eTIR_Cub_GradCroise,
  eTIR_Cub_GradTan,
  eTIR_Cub_GradTanPiS2,
  eTIR_Cub_GradTanPi,
  eTIR_Cub_LaplRad,
  eTIR_Cub_LaplTan,
  eTIR_Cub_LaplCrois,
  eTIR_Cub_DiffOpposePi,
  eTIR_Cub_DiffOpposePiS2,
  eTIR_NoLabel
} eTypeInvRad;
void xml_init(eTypeInvRad & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeInvRad & aVal);

eTypeInvRad  Str2eTypeInvRad(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeInvRad & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeInvRad &);

std::string  Mangling( eTypeInvRad *);

void  BinaryUnDumpFromFile(eTypeInvRad &,ELISE_fp &);

class cPtSc
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPtSc & anObj,cElXMLTree * aTree);


        Pt2dr & Pt();
        const Pt2dr & Pt()const ;

        double & Scale();
        const double & Scale()const ;
    private:
        Pt2dr mPt;
        double mScale;
};
cElXMLTree * ToXMLTree(const cPtSc &);

void  BinaryDumpInFile(ELISE_fp &,const cPtSc &);

void  BinaryUnDumpFromFile(cPtSc &,ELISE_fp &);

std::string  Mangling( cPtSc *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml_TestDMP
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_TestDMP & anObj,cElXMLTree * aTree);


        Im2D_INT2 & PxMin();
        const Im2D_INT2 & PxMin()const ;

        Im2D_INT2 & PxMax();
        const Im2D_INT2 & PxMax()const ;

        Im2D_INT4 & ImCpt();
        const Im2D_INT4 & ImCpt()const ;

        Im2D_U_INT2 & DataIm();
        const Im2D_U_INT2 & DataIm()const ;

        double & StepPx();
        const double & StepPx()const ;

        double & DynPx();
        const double & DynPx()const ;
    private:
        Im2D_INT2 mPxMin;
        Im2D_INT2 mPxMax;
        Im2D_INT4 mImCpt;
        Im2D_U_INT2 mDataIm;
        double mStepPx;
        double mDynPx;
};
cElXMLTree * ToXMLTree(const cXml_TestDMP &);

void  BinaryDumpInFile(ELISE_fp &,const cXml_TestDMP &);

void  BinaryUnDumpFromFile(cXml_TestDMP &,ELISE_fp &);

std::string  Mangling( cXml_TestDMP *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneInvRad
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneInvRad & anObj,cElXMLTree * aTree);


        Im2D_INT1 & ImRad();
        const Im2D_INT1 & ImRad()const ;

        Im2D_U_INT2 & CodeBinaire();
        const Im2D_U_INT2 & CodeBinaire()const ;
    private:
        Im2D_INT1 mImRad;
        Im2D_U_INT2 mCodeBinaire;
};
cElXMLTree * ToXMLTree(const cOneInvRad &);

void  BinaryDumpInFile(ELISE_fp &,const cOneInvRad &);

void  BinaryUnDumpFromFile(cOneInvRad &,ELISE_fp &);

std::string  Mangling( cOneInvRad *);

/******************************************************/
/******************************************************/
/******************************************************/
class cProfilRad
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cProfilRad & anObj,cElXMLTree * aTree);


        Im2D_INT1 & ImProfil();
        const Im2D_INT1 & ImProfil()const ;
    private:
        Im2D_INT1 mImProfil;
};
cElXMLTree * ToXMLTree(const cProfilRad &);

void  BinaryDumpInFile(ELISE_fp &,const cProfilRad &);

void  BinaryUnDumpFromFile(cProfilRad &,ELISE_fp &);

std::string  Mangling( cProfilRad *);

/******************************************************/
/******************************************************/
/******************************************************/
class cRotInvarAutoCor
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cRotInvarAutoCor & anObj,cElXMLTree * aTree);


        Im2D_INT1 & IR0();
        const Im2D_INT1 & IR0()const ;

        Im2D_INT1 & IGT();
        const Im2D_INT1 & IGT()const ;

        Im2D_INT1 & IGR();
        const Im2D_INT1 & IGR()const ;
    private:
        Im2D_INT1 mIR0;
        Im2D_INT1 mIGT;
        Im2D_INT1 mIGR;
};
cElXMLTree * ToXMLTree(const cRotInvarAutoCor &);

void  BinaryDumpInFile(ELISE_fp &,const cRotInvarAutoCor &);

void  BinaryUnDumpFromFile(cRotInvarAutoCor &,ELISE_fp &);

std::string  Mangling( cRotInvarAutoCor *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOnePCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOnePCarac & anObj,cElXMLTree * aTree);


        eTypePtRemark & Kind();
        const eTypePtRemark & Kind()const ;

        Pt2dr & Pt();
        const Pt2dr & Pt()const ;

        Pt2dr & Pt0();
        const Pt2dr & Pt0()const ;

        int & NivScale();
        const int & NivScale()const ;

        double & Scale();
        const double & Scale()const ;

        double & ScaleStab();
        const double & ScaleStab()const ;

        double & ScaleNature();
        const double & ScaleNature()const ;

        Pt2dr & DirMS();
        const Pt2dr & DirMS()const ;

        Pt2dr & DirAC();
        const Pt2dr & DirAC()const ;

        double & Contraste();
        const double & Contraste()const ;

        double & ContrasteRel();
        const double & ContrasteRel()const ;

        double & AutoCorrel();
        const double & AutoCorrel()const ;

        bool & OK();
        const bool & OK()const ;

        cOneInvRad & InvR();
        const cOneInvRad & InvR()const ;

        double & MoyLP();
        const double & MoyLP()const ;

        Im2D_INT1 & ImLogPol();
        const Im2D_INT1 & ImLogPol()const ;

        std::vector<double> & VectRho();
        const std::vector<double> & VectRho()const ;

        cProfilRad & ProfR();
        const cProfilRad & ProfR()const ;

        cRotInvarAutoCor & RIAC();
        const cRotInvarAutoCor & RIAC()const ;

        int & Id();
        const int & Id()const ;

        int & HeapInd();
        const int & HeapInd()const ;

        double & Prio();
        const double & Prio()const ;
    private:
        eTypePtRemark mKind;
        Pt2dr mPt;
        Pt2dr mPt0;
        int mNivScale;
        double mScale;
        double mScaleStab;
        double mScaleNature;
        Pt2dr mDirMS;
        Pt2dr mDirAC;
        double mContraste;
        double mContrasteRel;
        double mAutoCorrel;
        bool mOK;
        cOneInvRad mInvR;
        double mMoyLP;
        Im2D_INT1 mImLogPol;
        std::vector<double> mVectRho;
        cProfilRad mProfR;
        cRotInvarAutoCor mRIAC;
        int mId;
        int mHeapInd;
        double mPrio;
};
cElXMLTree * ToXMLTree(const cOnePCarac &);

void  BinaryDumpInFile(ELISE_fp &,const cOnePCarac &);

void  BinaryUnDumpFromFile(cOnePCarac &,ELISE_fp &);

std::string  Mangling( cOnePCarac *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetPCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetPCarac & anObj,cElXMLTree * aTree);


        std::vector< cOnePCarac > & OnePCarac();
        const std::vector< cOnePCarac > & OnePCarac()const ;
    private:
        std::vector< cOnePCarac > mOnePCarac;
};
cElXMLTree * ToXMLTree(const cSetPCarac &);

void  BinaryDumpInFile(ELISE_fp &,const cSetPCarac &);

void  BinaryUnDumpFromFile(cSetPCarac &,ELISE_fp &);

std::string  Mangling( cSetPCarac *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSRPC_Truth
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSRPC_Truth & anObj,cElXMLTree * aTree);


        cOnePCarac & P1();
        const cOnePCarac & P1()const ;

        cOnePCarac & P2();
        const cOnePCarac & P2()const ;
    private:
        cOnePCarac mP1;
        cOnePCarac mP2;
};
cElXMLTree * ToXMLTree(const cSRPC_Truth &);

void  BinaryDumpInFile(ELISE_fp &,const cSRPC_Truth &);

void  BinaryUnDumpFromFile(cSRPC_Truth &,ELISE_fp &);

std::string  Mangling( cSRPC_Truth *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetRefPCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetRefPCarac & anObj,cElXMLTree * aTree);


        std::vector< cSRPC_Truth > & SRPC_Truth();
        const std::vector< cSRPC_Truth > & SRPC_Truth()const ;

        std::vector< cOnePCarac > & SRPC_Rand();
        const std::vector< cOnePCarac > & SRPC_Rand()const ;
    private:
        std::vector< cSRPC_Truth > mSRPC_Truth;
        std::vector< cOnePCarac > mSRPC_Rand;
};
cElXMLTree * ToXMLTree(const cSetRefPCarac &);

void  BinaryDumpInFile(ELISE_fp &,const cSetRefPCarac &);

void  BinaryUnDumpFromFile(cSetRefPCarac &,ELISE_fp &);

std::string  Mangling( cSetRefPCarac *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCBOneBit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCBOneBit & anObj,cElXMLTree * aTree);


        std::vector<double> & Coeff();
        const std::vector<double> & Coeff()const ;

        std::vector<int> & IndInV();
        const std::vector<int> & IndInV()const ;

        int & IndBit();
        const int & IndBit()const ;
    private:
        std::vector<double> mCoeff;
        std::vector<int> mIndInV;
        int mIndBit;
};
cElXMLTree * ToXMLTree(const cCBOneBit &);

void  BinaryDumpInFile(ELISE_fp &,const cCBOneBit &);

void  BinaryUnDumpFromFile(cCBOneBit &,ELISE_fp &);

std::string  Mangling( cCBOneBit *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCBOneVect
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCBOneVect & anObj,cElXMLTree * aTree);


        int & IndVec();
        const int & IndVec()const ;

        std::vector< cCBOneBit > & CBOneBit();
        const std::vector< cCBOneBit > & CBOneBit()const ;
    private:
        int mIndVec;
        std::vector< cCBOneBit > mCBOneBit;
};
cElXMLTree * ToXMLTree(const cCBOneVect &);

void  BinaryDumpInFile(ELISE_fp &,const cCBOneVect &);

void  BinaryUnDumpFromFile(cCBOneVect &,ELISE_fp &);

std::string  Mangling( cCBOneVect *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFullParamCB
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFullParamCB & anObj,cElXMLTree * aTree);


        std::vector< cCBOneVect > & CBOneVect();
        const std::vector< cCBOneVect > & CBOneVect()const ;
    private:
        std::vector< cCBOneVect > mCBOneVect;
};
cElXMLTree * ToXMLTree(const cFullParamCB &);

void  BinaryDumpInFile(ELISE_fp &,const cFullParamCB &);

void  BinaryUnDumpFromFile(cFullParamCB &,ELISE_fp &);

std::string  Mangling( cFullParamCB *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCompCBOneBit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCompCBOneBit & anObj,cElXMLTree * aTree);


        std::vector<double> & Coeff();
        const std::vector<double> & Coeff()const ;

        std::vector<int> & IndX();
        const std::vector<int> & IndX()const ;

        std::vector<int> & IndY();
        const std::vector<int> & IndY()const ;

        int & IndBit();
        const int & IndBit()const ;
    private:
        std::vector<double> mCoeff;
        std::vector<int> mIndX;
        std::vector<int> mIndY;
        int mIndBit;
};
cElXMLTree * ToXMLTree(const cCompCBOneBit &);

void  BinaryDumpInFile(ELISE_fp &,const cCompCBOneBit &);

void  BinaryUnDumpFromFile(cCompCBOneBit &,ELISE_fp &);

std::string  Mangling( cCompCBOneBit *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCompCB
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCompCB & anObj,cElXMLTree * aTree);


        int & BitThresh();
        const int & BitThresh()const ;

        std::vector< cCompCBOneBit > & CompCBOneBit();
        const std::vector< cCompCBOneBit > & CompCBOneBit()const ;
    private:
        int mBitThresh;
        std::vector< cCompCBOneBit > mCompCBOneBit;
};
cElXMLTree * ToXMLTree(const cCompCB &);

void  BinaryDumpInFile(ELISE_fp &,const cCompCB &);

void  BinaryUnDumpFromFile(cCompCB &,ELISE_fp &);

std::string  Mangling( cCompCB *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFitsOneBin
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFitsOneBin & anObj,cElXMLTree * aTree);


        std::string & PrefName();
        const std::string & PrefName()const ;

        cTplValGesInit< std::string > & PostName();
        const cTplValGesInit< std::string > & PostName()const ;

        cTplValGesInit< cCompCB > & CCB();
        const cTplValGesInit< cCompCB > & CCB()const ;
    private:
        std::string mPrefName;
        cTplValGesInit< std::string > mPostName;
        cTplValGesInit< cCompCB > mCCB;
};
cElXMLTree * ToXMLTree(const cFitsOneBin &);

void  BinaryDumpInFile(ELISE_fp &,const cFitsOneBin &);

void  BinaryUnDumpFromFile(cFitsOneBin &,ELISE_fp &);

std::string  Mangling( cFitsOneBin *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFitsOneLabel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFitsOneLabel & anObj,cElXMLTree * aTree);


        eTypePtRemark & KindOf();
        const eTypePtRemark & KindOf()const ;

        cFitsOneBin & BinIndexed();
        const cFitsOneBin & BinIndexed()const ;

        cFitsOneBin & BinDecisionShort();
        const cFitsOneBin & BinDecisionShort()const ;

        cFitsOneBin & BinDecisionLong();
        const cFitsOneBin & BinDecisionLong()const ;
    private:
        eTypePtRemark mKindOf;
        cFitsOneBin mBinIndexed;
        cFitsOneBin mBinDecisionShort;
        cFitsOneBin mBinDecisionLong;
};
cElXMLTree * ToXMLTree(const cFitsOneLabel &);

void  BinaryDumpInFile(ELISE_fp &,const cFitsOneLabel &);

void  BinaryUnDumpFromFile(cFitsOneLabel &,ELISE_fp &);

std::string  Mangling( cFitsOneLabel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSeuilFitsParam
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSeuilFitsParam & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & SeuilCorrDR();
        const cTplValGesInit< double > & SeuilCorrDR()const ;

        cTplValGesInit< double > & SeuilInc();
        const cTplValGesInit< double > & SeuilInc()const ;

        cTplValGesInit< double > & SeuilCorrLP();
        const cTplValGesInit< double > & SeuilCorrLP()const ;

        cTplValGesInit< double > & ExposantPdsDistGrad();
        const cTplValGesInit< double > & ExposantPdsDistGrad()const ;

        cTplValGesInit< double > & SeuilDistGrad();
        const cTplValGesInit< double > & SeuilDistGrad()const ;

        cTplValGesInit< double > & SeuilCorrelRatio12();
        const cTplValGesInit< double > & SeuilCorrelRatio12()const ;

        cTplValGesInit< double > & SeuilGradRatio12();
        const cTplValGesInit< double > & SeuilGradRatio12()const ;
    private:
        cTplValGesInit< double > mSeuilCorrDR;
        cTplValGesInit< double > mSeuilInc;
        cTplValGesInit< double > mSeuilCorrLP;
        cTplValGesInit< double > mExposantPdsDistGrad;
        cTplValGesInit< double > mSeuilDistGrad;
        cTplValGesInit< double > mSeuilCorrelRatio12;
        cTplValGesInit< double > mSeuilGradRatio12;
};
cElXMLTree * ToXMLTree(const cSeuilFitsParam &);

void  BinaryDumpInFile(ELISE_fp &,const cSeuilFitsParam &);

void  BinaryUnDumpFromFile(cSeuilFitsParam &,ELISE_fp &);

std::string  Mangling( cSeuilFitsParam *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFitsParam
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFitsParam & anObj,cElXMLTree * aTree);


        cFitsOneLabel & DefInit();
        const cFitsOneLabel & DefInit()const ;

        std::list< cFitsOneLabel > & GenLabs();
        const std::list< cFitsOneLabel > & GenLabs()const ;

        cSeuilFitsParam & SeuilOL();
        const cSeuilFitsParam & SeuilOL()const ;

        cSeuilFitsParam & SeuilGen();
        const cSeuilFitsParam & SeuilGen()const ;
    private:
        cFitsOneLabel mDefInit;
        std::list< cFitsOneLabel > mGenLabs;
        cSeuilFitsParam mSeuilOL;
        cSeuilFitsParam mSeuilGen;
};
cElXMLTree * ToXMLTree(const cFitsParam &);

void  BinaryDumpInFile(ELISE_fp &,const cFitsParam &);

void  BinaryUnDumpFromFile(cFitsParam &,ELISE_fp &);

std::string  Mangling( cFitsParam *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXAPA_OneMatch
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXAPA_OneMatch & anObj,cElXMLTree * aTree);


        std::string & Master();
        const std::string & Master()const ;

        std::string & Pattern();
        const std::string & Pattern()const ;

        std::string & PatternRef();
        const std::string & PatternRef()const ;
    private:
        std::string mMaster;
        std::string mPattern;
        std::string mPatternRef;
};
cElXMLTree * ToXMLTree(const cXAPA_OneMatch &);

void  BinaryDumpInFile(ELISE_fp &,const cXAPA_OneMatch &);

void  BinaryUnDumpFromFile(cXAPA_OneMatch &,ELISE_fp &);

std::string  Mangling( cXAPA_OneMatch *);

class cXAPA_PtCar
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXAPA_PtCar & anObj,cElXMLTree * aTree);


        std::string & Pattern();
        const std::string & Pattern()const ;
    private:
        std::string mPattern;
};
cElXMLTree * ToXMLTree(const cXAPA_PtCar &);

void  BinaryDumpInFile(ELISE_fp &,const cXAPA_PtCar &);

void  BinaryUnDumpFromFile(cXAPA_PtCar &,ELISE_fp &);

std::string  Mangling( cXAPA_PtCar *);

class cXlmAimeOneDir
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXlmAimeOneDir & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DoIt();
        const cTplValGesInit< bool > & DoIt()const ;

        cTplValGesInit< bool > & DoMatch();
        const cTplValGesInit< bool > & DoMatch()const ;

        cTplValGesInit< bool > & DoPtCar();
        const cTplValGesInit< bool > & DoPtCar()const ;

        cTplValGesInit< bool > & DoRef();
        const cTplValGesInit< bool > & DoRef()const ;

        cTplValGesInit< int > & ZoomF();
        const cTplValGesInit< int > & ZoomF()const ;

        cTplValGesInit< int > & NumMatch();
        const cTplValGesInit< int > & NumMatch()const ;

        std::string & Dir();
        const std::string & Dir()const ;

        std::string & Ori();
        const std::string & Ori()const ;

        std::list< cXAPA_OneMatch > & XAPA_OneMatch();
        const std::list< cXAPA_OneMatch > & XAPA_OneMatch()const ;

        std::string & Pattern();
        const std::string & Pattern()const ;

        cXAPA_PtCar & XAPA_PtCar();
        const cXAPA_PtCar & XAPA_PtCar()const ;
    private:
        cTplValGesInit< bool > mDoIt;
        cTplValGesInit< bool > mDoMatch;
        cTplValGesInit< bool > mDoPtCar;
        cTplValGesInit< bool > mDoRef;
        cTplValGesInit< int > mZoomF;
        cTplValGesInit< int > mNumMatch;
        std::string mDir;
        std::string mOri;
        std::list< cXAPA_OneMatch > mXAPA_OneMatch;
        cXAPA_PtCar mXAPA_PtCar;
};
cElXMLTree * ToXMLTree(const cXlmAimeOneDir &);

void  BinaryDumpInFile(ELISE_fp &,const cXlmAimeOneDir &);

void  BinaryUnDumpFromFile(cXlmAimeOneDir &,ELISE_fp &);

std::string  Mangling( cXlmAimeOneDir *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXlmAimeOneApprent
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXlmAimeOneApprent & anObj,cElXMLTree * aTree);


        double & PdsW();
        const double & PdsW()const ;

        int & NbBB();
        const int & NbBB()const ;

        cTplValGesInit< int > & BitM();
        const cTplValGesInit< int > & BitM()const ;
    private:
        double mPdsW;
        int mNbBB;
        cTplValGesInit< int > mBitM;
};
cElXMLTree * ToXMLTree(const cXlmAimeOneApprent &);

void  BinaryDumpInFile(ELISE_fp &,const cXlmAimeOneApprent &);

void  BinaryUnDumpFromFile(cXlmAimeOneApprent &,ELISE_fp &);

std::string  Mangling( cXlmAimeOneApprent *);

class cXlmAimeApprent
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXlmAimeApprent & anObj,cElXMLTree * aTree);


        int & NbExEt0();
        const int & NbExEt0()const ;

        int & NbExEt1();
        const int & NbExEt1()const ;

        cTplValGesInit< double > & TimeOut();
        const cTplValGesInit< double > & TimeOut()const ;

        std::list< cXlmAimeOneApprent > & XlmAimeOneApprent();
        const std::list< cXlmAimeOneApprent > & XlmAimeOneApprent()const ;
    private:
        int mNbExEt0;
        int mNbExEt1;
        cTplValGesInit< double > mTimeOut;
        std::list< cXlmAimeOneApprent > mXlmAimeOneApprent;
};
cElXMLTree * ToXMLTree(const cXlmAimeApprent &);

void  BinaryDumpInFile(ELISE_fp &,const cXlmAimeApprent &);

void  BinaryUnDumpFromFile(cXlmAimeApprent &,ELISE_fp &);

std::string  Mangling( cXlmAimeApprent *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlAimeParamApprentissage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlAimeParamApprentissage & anObj,cElXMLTree * aTree);


        std::string & AbsDir();
        const std::string & AbsDir()const ;

        cTplValGesInit< bool > & DefDoIt();
        const cTplValGesInit< bool > & DefDoIt()const ;

        cTplValGesInit< bool > & DefDoMatch();
        const cTplValGesInit< bool > & DefDoMatch()const ;

        cTplValGesInit< bool > & DefDoPtCar();
        const cTplValGesInit< bool > & DefDoPtCar()const ;

        cTplValGesInit< bool > & DefDoRef();
        const cTplValGesInit< bool > & DefDoRef()const ;

        cTplValGesInit< bool > & DefDoApprComb();
        const cTplValGesInit< bool > & DefDoApprComb()const ;

        cTplValGesInit< bool > & DefDoApprLocal1();
        const cTplValGesInit< bool > & DefDoApprLocal1()const ;

        cTplValGesInit< bool > & DefDoApprLocal2();
        const cTplValGesInit< bool > & DefDoApprLocal2()const ;

        cTplValGesInit< std::string > & DefParamPtCar();
        const cTplValGesInit< std::string > & DefParamPtCar()const ;

        std::list< cXlmAimeOneDir > & XlmAimeOneDir();
        const std::list< cXlmAimeOneDir > & XlmAimeOneDir()const ;

        int & NbExEt0();
        const int & NbExEt0()const ;

        int & NbExEt1();
        const int & NbExEt1()const ;

        cTplValGesInit< double > & TimeOut();
        const cTplValGesInit< double > & TimeOut()const ;

        std::list< cXlmAimeOneApprent > & XlmAimeOneApprent();
        const std::list< cXlmAimeOneApprent > & XlmAimeOneApprent()const ;

        cXlmAimeApprent & XlmAimeApprent();
        const cXlmAimeApprent & XlmAimeApprent()const ;
    private:
        std::string mAbsDir;
        cTplValGesInit< bool > mDefDoIt;
        cTplValGesInit< bool > mDefDoMatch;
        cTplValGesInit< bool > mDefDoPtCar;
        cTplValGesInit< bool > mDefDoRef;
        cTplValGesInit< bool > mDefDoApprComb;
        cTplValGesInit< bool > mDefDoApprLocal1;
        cTplValGesInit< bool > mDefDoApprLocal2;
        cTplValGesInit< std::string > mDefParamPtCar;
        std::list< cXlmAimeOneDir > mXlmAimeOneDir;
        cXlmAimeApprent mXlmAimeApprent;
};
cElXMLTree * ToXMLTree(const cXmlAimeParamApprentissage &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlAimeParamApprentissage &);

void  BinaryUnDumpFromFile(cXmlAimeParamApprentissage &,ELISE_fp &);

std::string  Mangling( cXmlAimeParamApprentissage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml2007Pt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml2007Pt & anObj,cElXMLTree * aTree);


        Pt2dr & PtInit();
        const Pt2dr & PtInit()const ;

        Pt2dr & PtAff();
        const Pt2dr & PtAff()const ;

        int & Id();
        const int & Id()const ;

        int & NumOct();
        const int & NumOct()const ;

        int & NumIm();
        const int & NumIm()const ;

        double & ScaleInO();
        const double & ScaleInO()const ;

        double & ScaleAbs();
        const double & ScaleAbs()const ;

        double & Score();
        const double & Score()const ;

        double & ScoreRel();
        const double & ScoreRel()const ;

        std::vector<double> & VectRho();
        const std::vector<double> & VectRho()const ;

        std::vector<double> & VectDir();
        const std::vector<double> & VectDir()const ;

        double & Var();
        const double & Var()const ;

        double & AutoCor();
        const double & AutoCor()const ;

        int & NumChAC();
        const int & NumChAC()const ;

        bool & OKAc();
        const bool & OKAc()const ;

        bool & OKLP();
        const bool & OKLP()const ;

        bool & SFSelected();
        const bool & SFSelected()const ;

        bool & Stable();
        const bool & Stable()const ;

        bool & ChgMaj();
        const bool & ChgMaj()const ;

        Im2D_U_INT1 & ImLP();
        const Im2D_U_INT1 & ImLP()const ;
    private:
        Pt2dr mPtInit;
        Pt2dr mPtAff;
        int mId;
        int mNumOct;
        int mNumIm;
        double mScaleInO;
        double mScaleAbs;
        double mScore;
        double mScoreRel;
        std::vector<double> mVectRho;
        std::vector<double> mVectDir;
        double mVar;
        double mAutoCor;
        int mNumChAC;
        bool mOKAc;
        bool mOKLP;
        bool mSFSelected;
        bool mStable;
        bool mChgMaj;
        Im2D_U_INT1 mImLP;
};
cElXMLTree * ToXMLTree(const cXml2007Pt &);

void  BinaryDumpInFile(ELISE_fp &,const cXml2007Pt &);

void  BinaryUnDumpFromFile(cXml2007Pt &,ELISE_fp &);

std::string  Mangling( cXml2007Pt *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXml2007SetPtOneType
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml2007SetPtOneType & anObj,cElXMLTree * aTree);


        std::vector< cXml2007Pt > & Pts();
        const std::vector< cXml2007Pt > & Pts()const ;

        bool & IsMax();
        const bool & IsMax()const ;

        int & TypePt();
        const int & TypePt()const ;

        std::string & NameTypePt();
        const std::string & NameTypePt()const ;
    private:
        std::vector< cXml2007Pt > mPts;
        bool mIsMax;
        int mTypePt;
        std::string mNameTypePt;
};
cElXMLTree * ToXMLTree(const cXml2007SetPtOneType &);

void  BinaryDumpInFile(ELISE_fp &,const cXml2007SetPtOneType &);

void  BinaryUnDumpFromFile(cXml2007SetPtOneType &,ELISE_fp &);

std::string  Mangling( cXml2007SetPtOneType *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotRechNewPH
