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

        std::vector<double> & CoeffRadiom();
        const std::vector<double> & CoeffRadiom()const ;

        std::vector<double> & CoeffRadiom2();
        const std::vector<double> & CoeffRadiom2()const ;

        std::vector<double> & CoeffGradRadial();
        const std::vector<double> & CoeffGradRadial()const ;

        std::vector<double> & CoeffGradTangent();
        const std::vector<double> & CoeffGradTangent()const ;

        std::vector<double> & CoeffGradTangentPiS4();
        const std::vector<double> & CoeffGradTangentPiS4()const ;

        std::vector<double> & CoeffGradTangentPiS2();
        const std::vector<double> & CoeffGradTangentPiS2()const ;

        Im2D_REAL4 & ImRad();
        const Im2D_REAL4 & ImRad()const ;

        std::vector<double> & VectRho();
        const std::vector<double> & VectRho()const ;
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
        std::vector<double> mCoeffRadiom;
        std::vector<double> mCoeffRadiom2;
        std::vector<double> mCoeffGradRadial;
        std::vector<double> mCoeffGradTangent;
        std::vector<double> mCoeffGradTangentPiS4;
        std::vector<double> mCoeffGradTangentPiS2;
        Im2D_REAL4 mImRad;
        std::vector<double> mVectRho;
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
// };
#endif // Define_NotRechNewPH
