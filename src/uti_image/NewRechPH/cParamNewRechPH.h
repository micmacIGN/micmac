#ifndef Define_NotRechNewPH
#define Define_NotRechNewPH
// NOMORE ...
typedef enum
{
  eTPR_LaplMax,
  eTPR_LaplMin,
  eTPR_GrayMax,
  eTPR_GrayMin,
  eTPR_GraySadl,
  eTPR_NoLabel
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

        Im2D_INT1 & ImLogPol();
        const Im2D_INT1 & ImLogPol()const ;

        std::vector<double> & VectRho();
        const std::vector<double> & VectRho()const ;

        cProfilRad & ProfR();
        const cProfilRad & ProfR()const ;
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
        Im2D_INT1 mImLogPol;
        std::vector<double> mVectRho;
        cProfilRad mProfR;
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


        std::vector< cCompCBOneBit > & CompCBOneBit();
        const std::vector< cCompCBOneBit > & CompCBOneBit()const ;
    private:
        std::vector< cCompCBOneBit > mCompCBOneBit;
};
cElXMLTree * ToXMLTree(const cCompCB &);

void  BinaryDumpInFile(ELISE_fp &,const cCompCB &);

void  BinaryUnDumpFromFile(cCompCB &,ELISE_fp &);

std::string  Mangling( cCompCB *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotRechNewPH
