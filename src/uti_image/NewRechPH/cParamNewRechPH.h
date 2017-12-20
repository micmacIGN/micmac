#ifndef Define_NotRechNewPH
#define Define_NotRechNewPH
// NOMORE ...
typedef enum
{
  eTPR_Max,
  eTPR_Min,
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

        double & Scale();
        const double & Scale()const ;

        Pt2dr & Dir();
        const Pt2dr & Dir()const ;

        double & Contrast();
        const double & Contrast()const ;

        double & AutoCorrel();
        const double & AutoCorrel()const ;
    private:
        eTypePtRemark mKind;
        Pt2dr mPt;
        double mScale;
        Pt2dr mDir;
        double mContrast;
        double mAutoCorrel;
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


        std::list< cOnePCarac > & OnePCarac();
        const std::list< cOnePCarac > & OnePCarac()const ;
    private:
        std::list< cOnePCarac > mOnePCarac;
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
