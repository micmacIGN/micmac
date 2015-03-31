// #include "general/all.h"
// #include "private/all.h"
#ifndef Define_NotSupIm
#define Define_NotSupIm
// #include "general/all.h"
// #include "private/all.h"
// #include "XML_GEN/ParamChantierPhotogram.h"
//
typedef enum
{
  eTSA_CylindreRevolution
} eTypeSurfaceAnalytique;
void xml_init(eTypeSurfaceAnalytique & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeSurfaceAnalytique & aVal);

eTypeSurfaceAnalytique  Str2eTypeSurfaceAnalytique(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeSurfaceAnalytique & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTypeSurfaceAnalytique &);

std::string  Mangling( eTypeSurfaceAnalytique *);

void  BinaryUnDumpFromFile(eTypeSurfaceAnalytique &,ELISE_fp &);

typedef enum
{
  eMBF_Union,
  eMBF_Inter,
  eMBF_First
} eModeBoxFusion;
void xml_init(eModeBoxFusion & aVal,cElXMLTree * aTree);
std::string  eToString(const eModeBoxFusion & aVal);

eModeBoxFusion  Str2eModeBoxFusion(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeBoxFusion & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eModeBoxFusion &);

std::string  Mangling( eModeBoxFusion *);

void  BinaryUnDumpFromFile(eModeBoxFusion &,ELISE_fp &);

typedef enum
{
  eQC_Out,
  eQC_ZeroCohBrd,
  eQC_ZeroCoh,
  eQC_ZeroCohImMul,
  eQC_GradFort,
  eQC_GradFaibleC1,
  eQC_Bord,
  eQC_Coh1,
  eQC_GradFaibleC2,
  eQC_Coh2,
  eQC_Coh3,
  eQC_NonAff
} eQualCloud;
void xml_init(eQualCloud & aVal,cElXMLTree * aTree);
std::string  eToString(const eQualCloud & aVal);

eQualCloud  Str2eQualCloud(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eQualCloud & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eQualCloud &);

std::string  Mangling( eQualCloud *);

void  BinaryUnDumpFromFile(eQualCloud &,ELISE_fp &);

class cIntervLutConvertion
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cIntervLutConvertion & anObj,cElXMLTree * aTree);


        int & NivIn();
        const int & NivIn()const ;

        int & NivOut();
        const int & NivOut()const ;
    private:
        int mNivIn;
        int mNivOut;
};
cElXMLTree * ToXMLTree(const cIntervLutConvertion &);

void  BinaryDumpInFile(ELISE_fp &,const cIntervLutConvertion &);

void  BinaryUnDumpFromFile(cIntervLutConvertion &,ELISE_fp &);

std::string  Mangling( cIntervLutConvertion *);

/******************************************************/
/******************************************************/
/******************************************************/
class cLutConvertion
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLutConvertion & anObj,cElXMLTree * aTree);


        std::vector< cIntervLutConvertion > & IntervLutConvertion();
        const std::vector< cIntervLutConvertion > & IntervLutConvertion()const ;
    private:
        std::vector< cIntervLutConvertion > mIntervLutConvertion;
};
cElXMLTree * ToXMLTree(const cLutConvertion &);

void  BinaryDumpInFile(ELISE_fp &,const cLutConvertion &);

void  BinaryUnDumpFromFile(cLutConvertion &,ELISE_fp &);

std::string  Mangling( cLutConvertion *);

/******************************************************/
/******************************************************/
/******************************************************/
class cWindowSelection
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cWindowSelection & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & AllPts();
        const cTplValGesInit< std::string > & AllPts()const ;

        cTplValGesInit< std::string > & PtsCenter();
        const cTplValGesInit< std::string > & PtsCenter()const ;

        cTplValGesInit< double > & Percent();
        const cTplValGesInit< double > & Percent()const ;
    private:
        cTplValGesInit< std::string > mAllPts;
        cTplValGesInit< std::string > mPtsCenter;
        cTplValGesInit< double > mPercent;
};
cElXMLTree * ToXMLTree(const cWindowSelection &);

void  BinaryDumpInFile(ELISE_fp &,const cWindowSelection &);

void  BinaryUnDumpFromFile(cWindowSelection &,ELISE_fp &);

std::string  Mangling( cWindowSelection *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMasqTerrain
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMasqTerrain & anObj,cElXMLTree * aTree);


        std::string & Image();
        const std::string & Image()const ;

        std::string & XML();
        const std::string & XML()const ;

        cWindowSelection & SelectPts();
        const cWindowSelection & SelectPts()const ;
    private:
        std::string mImage;
        std::string mXML;
        cWindowSelection mSelectPts;
};
cElXMLTree * ToXMLTree(const cMasqTerrain &);

void  BinaryDumpInFile(ELISE_fp &,const cMasqTerrain &);

void  BinaryUnDumpFromFile(cMasqTerrain &,ELISE_fp &);

std::string  Mangling( cMasqTerrain *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBoxPixMort
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBoxPixMort & anObj,cElXMLTree * aTree);


        Pt2di & HautG();
        const Pt2di & HautG()const ;

        Pt2di & BasD();
        const Pt2di & BasD()const ;
    private:
        Pt2di mHautG;
        Pt2di mBasD;
};
cElXMLTree * ToXMLTree(const cBoxPixMort &);

void  BinaryDumpInFile(ELISE_fp &,const cBoxPixMort &);

void  BinaryUnDumpFromFile(cBoxPixMort &,ELISE_fp &);

std::string  Mangling( cBoxPixMort *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFlattField
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFlattField & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        std::vector< double > & RefValue();
        const std::vector< double > & RefValue()const ;
    private:
        std::string mNameFile;
        std::vector< double > mRefValue;
};
cElXMLTree * ToXMLTree(const cFlattField &);

void  BinaryDumpInFile(ELISE_fp &,const cFlattField &);

void  BinaryUnDumpFromFile(cFlattField &,ELISE_fp &);

std::string  Mangling( cFlattField *);

/******************************************************/
/******************************************************/
/******************************************************/
class cChannelCmpCol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChannelCmpCol & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & Dyn();
        const cTplValGesInit< double > & Dyn()const ;

        cTplValGesInit< double > & Offset();
        const cTplValGesInit< double > & Offset()const ;

        int & In();
        const int & In()const ;

        int & Out();
        const int & Out()const ;

        cTplValGesInit< double > & Pds();
        const cTplValGesInit< double > & Pds()const ;

        cTplValGesInit< double > & ParamBiCub();
        const cTplValGesInit< double > & ParamBiCub()const ;
    private:
        cTplValGesInit< double > mDyn;
        cTplValGesInit< double > mOffset;
        int mIn;
        int mOut;
        cTplValGesInit< double > mPds;
        cTplValGesInit< double > mParamBiCub;
};
cElXMLTree * ToXMLTree(const cChannelCmpCol &);

void  BinaryDumpInFile(ELISE_fp &,const cChannelCmpCol &);

void  BinaryUnDumpFromFile(cChannelCmpCol &,ELISE_fp &);

std::string  Mangling( cChannelCmpCol *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImageCmpCol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImageCmpCol & anObj,cElXMLTree * aTree);


        std::string & NameOrKey();
        const std::string & NameOrKey()const ;

        cTplValGesInit< eTypeNumerique > & TypeTmpIn();
        const cTplValGesInit< eTypeNumerique > & TypeTmpIn()const ;

        cTplValGesInit< std::string > & KeyCalcNameImOfGeom();
        const cTplValGesInit< std::string > & KeyCalcNameImOfGeom()const ;

        Pt2di & HautG();
        const Pt2di & HautG()const ;

        Pt2di & BasD();
        const Pt2di & BasD()const ;

        cTplValGesInit< cBoxPixMort > & BoxPixMort();
        const cTplValGesInit< cBoxPixMort > & BoxPixMort()const ;

        std::string & NameFile();
        const std::string & NameFile()const ;

        std::vector< double > & RefValue();
        const std::vector< double > & RefValue()const ;

        cTplValGesInit< cFlattField > & FlattField();
        const cTplValGesInit< cFlattField > & FlattField()const ;

        std::list< cChannelCmpCol > & ChannelCmpCol();
        const std::list< cChannelCmpCol > & ChannelCmpCol()const ;

        cTplValGesInit< int > & NbFilter();
        const cTplValGesInit< int > & NbFilter()const ;

        cTplValGesInit< int > & SzFilter();
        const cTplValGesInit< int > & SzFilter()const ;
    private:
        std::string mNameOrKey;
        cTplValGesInit< eTypeNumerique > mTypeTmpIn;
        cTplValGesInit< std::string > mKeyCalcNameImOfGeom;
        cTplValGesInit< cBoxPixMort > mBoxPixMort;
        cTplValGesInit< cFlattField > mFlattField;
        std::list< cChannelCmpCol > mChannelCmpCol;
        cTplValGesInit< int > mNbFilter;
        cTplValGesInit< int > mSzFilter;
};
cElXMLTree * ToXMLTree(const cImageCmpCol &);

void  BinaryDumpInFile(ELISE_fp &,const cImageCmpCol &);

void  BinaryUnDumpFromFile(cImageCmpCol &,ELISE_fp &);

std::string  Mangling( cImageCmpCol *);

/******************************************************/
/******************************************************/
/******************************************************/
class cShowCalibsRel
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cShowCalibsRel & anObj,cElXMLTree * aTree);


        std::vector< int > & Channel();
        const std::vector< int > & Channel()const ;

        cTplValGesInit< double > & MaxRatio();
        const cTplValGesInit< double > & MaxRatio()const ;
    private:
        std::vector< int > mChannel;
        cTplValGesInit< double > mMaxRatio;
};
cElXMLTree * ToXMLTree(const cShowCalibsRel &);

void  BinaryDumpInFile(ELISE_fp &,const cShowCalibsRel &);

void  BinaryUnDumpFromFile(cShowCalibsRel &,ELISE_fp &);

std::string  Mangling( cShowCalibsRel *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImResultCC_Gray
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImResultCC_Gray & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & Channel();
        const cTplValGesInit< int > & Channel()const ;
    private:
        cTplValGesInit< int > mChannel;
};
cElXMLTree * ToXMLTree(const cImResultCC_Gray &);

void  BinaryDumpInFile(ELISE_fp &,const cImResultCC_Gray &);

void  BinaryUnDumpFromFile(cImResultCC_Gray &,ELISE_fp &);

std::string  Mangling( cImResultCC_Gray *);

class cImResultCC_RVB
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImResultCC_RVB & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt3di > & Channel();
        const cTplValGesInit< Pt3di > & Channel()const ;
    private:
        cTplValGesInit< Pt3di > mChannel;
};
cElXMLTree * ToXMLTree(const cImResultCC_RVB &);

void  BinaryDumpInFile(ELISE_fp &,const cImResultCC_RVB &);

void  BinaryUnDumpFromFile(cImResultCC_RVB &,ELISE_fp &);

std::string  Mangling( cImResultCC_RVB *);

class cImResultCC_Cnes
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImResultCC_Cnes & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & ModeMedian();
        const cTplValGesInit< bool > & ModeMedian()const ;

        cTplValGesInit< Pt2di > & SzF();
        const cTplValGesInit< Pt2di > & SzF()const ;

        cTplValGesInit< std::string > & ValueF();
        const cTplValGesInit< std::string > & ValueF()const ;

        cTplValGesInit< int > & ChannelHF();
        const cTplValGesInit< int > & ChannelHF()const ;

        cTplValGesInit< std::vector<int> > & ChannelBF();
        const cTplValGesInit< std::vector<int> > & ChannelBF()const ;

        cTplValGesInit< int > & NbIterFCSte();
        const cTplValGesInit< int > & NbIterFCSte()const ;

        cTplValGesInit< int > & SzIterFCSte();
        const cTplValGesInit< int > & SzIterFCSte()const ;
    private:
        cTplValGesInit< bool > mModeMedian;
        cTplValGesInit< Pt2di > mSzF;
        cTplValGesInit< std::string > mValueF;
        cTplValGesInit< int > mChannelHF;
        cTplValGesInit< std::vector<int> > mChannelBF;
        cTplValGesInit< int > mNbIterFCSte;
        cTplValGesInit< int > mSzIterFCSte;
};
cElXMLTree * ToXMLTree(const cImResultCC_Cnes &);

void  BinaryDumpInFile(ELISE_fp &,const cImResultCC_Cnes &);

void  BinaryUnDumpFromFile(cImResultCC_Cnes &,ELISE_fp &);

std::string  Mangling( cImResultCC_Cnes *);

class cImResultCC_PXs
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImResultCC_PXs & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::vector<int> > & Channel();
        const cTplValGesInit< std::vector<int> > & Channel()const ;

        cTplValGesInit< Pt3dr > & AxeRGB();
        const cTplValGesInit< Pt3dr > & AxeRGB()const ;

        cTplValGesInit< double > & Cste();
        const cTplValGesInit< double > & Cste()const ;

        cTplValGesInit< bool > & ApprentisageAxeRGB();
        const cTplValGesInit< bool > & ApprentisageAxeRGB()const ;

        std::list< std::string > & UnusedAppr();
        const std::list< std::string > & UnusedAppr()const ;
    private:
        cTplValGesInit< std::vector<int> > mChannel;
        cTplValGesInit< Pt3dr > mAxeRGB;
        cTplValGesInit< double > mCste;
        cTplValGesInit< bool > mApprentisageAxeRGB;
        std::list< std::string > mUnusedAppr;
};
cElXMLTree * ToXMLTree(const cImResultCC_PXs &);

void  BinaryDumpInFile(ELISE_fp &,const cImResultCC_PXs &);

void  BinaryUnDumpFromFile(cImResultCC_PXs &,ELISE_fp &);

std::string  Mangling( cImResultCC_PXs *);

class cPondThom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPondThom & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & PondExp();
        const cTplValGesInit< double > & PondExp()const ;

        cTplValGesInit< int > & PondCste();
        const cTplValGesInit< int > & PondCste()const ;
    private:
        cTplValGesInit< double > mPondExp;
        cTplValGesInit< int > mPondCste;
};
cElXMLTree * ToXMLTree(const cPondThom &);

void  BinaryDumpInFile(ELISE_fp &,const cPondThom &);

void  BinaryUnDumpFromFile(cPondThom &,ELISE_fp &);

std::string  Mangling( cPondThom *);

class cThomBidouille
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cThomBidouille & anObj,cElXMLTree * aTree);


        double & VMin();
        const double & VMin()const ;

        double & PourCent();
        const double & PourCent()const ;
    private:
        double mVMin;
        double mPourCent;
};
cElXMLTree * ToXMLTree(const cThomBidouille &);

void  BinaryDumpInFile(ELISE_fp &,const cThomBidouille &);

void  BinaryUnDumpFromFile(cThomBidouille &,ELISE_fp &);

std::string  Mangling( cThomBidouille *);

class cMPDBidouille
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMPDBidouille & anObj,cElXMLTree * aTree);


        double & EcartMin();
        const double & EcartMin()const ;
    private:
        double mEcartMin;
};
cElXMLTree * ToXMLTree(const cMPDBidouille &);

void  BinaryDumpInFile(ELISE_fp &,const cMPDBidouille &);

void  BinaryUnDumpFromFile(cMPDBidouille &,ELISE_fp &);

std::string  Mangling( cMPDBidouille *);

class cThomAgreg
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cThomAgreg & anObj,cElXMLTree * aTree);


        double & VMin();
        const double & VMin()const ;

        double & PourCent();
        const double & PourCent()const ;

        cTplValGesInit< cThomBidouille > & ThomBidouille();
        const cTplValGesInit< cThomBidouille > & ThomBidouille()const ;

        double & EcartMin();
        const double & EcartMin()const ;

        cTplValGesInit< cMPDBidouille > & MPDBidouille();
        const cTplValGesInit< cMPDBidouille > & MPDBidouille()const ;
    private:
        cTplValGesInit< cThomBidouille > mThomBidouille;
        cTplValGesInit< cMPDBidouille > mMPDBidouille;
};
cElXMLTree * ToXMLTree(const cThomAgreg &);

void  BinaryDumpInFile(ELISE_fp &,const cThomAgreg &);

void  BinaryUnDumpFromFile(cThomAgreg &,ELISE_fp &);

std::string  Mangling( cThomAgreg *);

class cImResultCC_Thom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImResultCC_Thom & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & PondExp();
        const cTplValGesInit< double > & PondExp()const ;

        cTplValGesInit< int > & PondCste();
        const cTplValGesInit< int > & PondCste()const ;

        cPondThom & PondThom();
        const cPondThom & PondThom()const ;

        cTplValGesInit< int > & NbIterPond();
        const cTplValGesInit< int > & NbIterPond()const ;

        cTplValGesInit< bool > & SupressCentre();
        const cTplValGesInit< bool > & SupressCentre()const ;

        cTplValGesInit< int > & ChannelHF();
        const cTplValGesInit< int > & ChannelHF()const ;

        cTplValGesInit< std::vector<int> > & ChannelBF();
        const cTplValGesInit< std::vector<int> > & ChannelBF()const ;

        double & VMin();
        const double & VMin()const ;

        double & PourCent();
        const double & PourCent()const ;

        cTplValGesInit< cThomBidouille > & ThomBidouille();
        const cTplValGesInit< cThomBidouille > & ThomBidouille()const ;

        double & EcartMin();
        const double & EcartMin()const ;

        cTplValGesInit< cMPDBidouille > & MPDBidouille();
        const cTplValGesInit< cMPDBidouille > & MPDBidouille()const ;

        cThomAgreg & ThomAgreg();
        const cThomAgreg & ThomAgreg()const ;
    private:
        cPondThom mPondThom;
        cTplValGesInit< int > mNbIterPond;
        cTplValGesInit< bool > mSupressCentre;
        cTplValGesInit< int > mChannelHF;
        cTplValGesInit< std::vector<int> > mChannelBF;
        cThomAgreg mThomAgreg;
};
cElXMLTree * ToXMLTree(const cImResultCC_Thom &);

void  BinaryDumpInFile(ELISE_fp &,const cImResultCC_Thom &);

void  BinaryUnDumpFromFile(cImResultCC_Thom &,ELISE_fp &);

std::string  Mangling( cImResultCC_Thom *);

class cImResultCC
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImResultCC & anObj,cElXMLTree * aTree);


        cTplValGesInit< cImResultCC_Gray > & ImResultCC_Gray();
        const cTplValGesInit< cImResultCC_Gray > & ImResultCC_Gray()const ;

        cTplValGesInit< cImResultCC_RVB > & ImResultCC_RVB();
        const cTplValGesInit< cImResultCC_RVB > & ImResultCC_RVB()const ;

        cTplValGesInit< cImResultCC_Cnes > & ImResultCC_Cnes();
        const cTplValGesInit< cImResultCC_Cnes > & ImResultCC_Cnes()const ;

        cTplValGesInit< cImResultCC_PXs > & ImResultCC_PXs();
        const cTplValGesInit< cImResultCC_PXs > & ImResultCC_PXs()const ;

        cTplValGesInit< cImResultCC_Thom > & ImResultCC_Thom();
        const cTplValGesInit< cImResultCC_Thom > & ImResultCC_Thom()const ;
    private:
        cTplValGesInit< cImResultCC_Gray > mImResultCC_Gray;
        cTplValGesInit< cImResultCC_RVB > mImResultCC_RVB;
        cTplValGesInit< cImResultCC_Cnes > mImResultCC_Cnes;
        cTplValGesInit< cImResultCC_PXs > mImResultCC_PXs;
        cTplValGesInit< cImResultCC_Thom > mImResultCC_Thom;
};
cElXMLTree * ToXMLTree(const cImResultCC &);

void  BinaryDumpInFile(ELISE_fp &,const cImResultCC &);

void  BinaryUnDumpFromFile(cImResultCC &,ELISE_fp &);

std::string  Mangling( cImResultCC *);

class cResultCompCol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cResultCompCol & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & GamaExport();
        const cTplValGesInit< double > & GamaExport()const ;

        cTplValGesInit< double > & RefGama();
        const cTplValGesInit< double > & RefGama()const ;

        cTplValGesInit< cLutConvertion > & LutExport();
        const cTplValGesInit< cLutConvertion > & LutExport()const ;

        std::string & KeyName();
        const std::string & KeyName()const ;

        cTplValGesInit< eTypeNumerique > & Type();
        const cTplValGesInit< eTypeNumerique > & Type()const ;

        cTplValGesInit< cImResultCC_Gray > & ImResultCC_Gray();
        const cTplValGesInit< cImResultCC_Gray > & ImResultCC_Gray()const ;

        cTplValGesInit< cImResultCC_RVB > & ImResultCC_RVB();
        const cTplValGesInit< cImResultCC_RVB > & ImResultCC_RVB()const ;

        cTplValGesInit< cImResultCC_Cnes > & ImResultCC_Cnes();
        const cTplValGesInit< cImResultCC_Cnes > & ImResultCC_Cnes()const ;

        cTplValGesInit< cImResultCC_PXs > & ImResultCC_PXs();
        const cTplValGesInit< cImResultCC_PXs > & ImResultCC_PXs()const ;

        cTplValGesInit< cImResultCC_Thom > & ImResultCC_Thom();
        const cTplValGesInit< cImResultCC_Thom > & ImResultCC_Thom()const ;

        cImResultCC & ImResultCC();
        const cImResultCC & ImResultCC()const ;
    private:
        cTplValGesInit< double > mGamaExport;
        cTplValGesInit< double > mRefGama;
        cTplValGesInit< cLutConvertion > mLutExport;
        std::string mKeyName;
        cTplValGesInit< eTypeNumerique > mType;
        cImResultCC mImResultCC;
};
cElXMLTree * ToXMLTree(const cResultCompCol &);

void  BinaryDumpInFile(ELISE_fp &,const cResultCompCol &);

void  BinaryUnDumpFromFile(cResultCompCol &,ELISE_fp &);

std::string  Mangling( cResultCompCol *);

/******************************************************/
/******************************************************/
/******************************************************/
class cEspaceResultSuperpCol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEspaceResultSuperpCol & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & EnglobImMaitre();
        const cTplValGesInit< std::string > & EnglobImMaitre()const ;

        cTplValGesInit< std::string > & EnglobAll();
        const cTplValGesInit< std::string > & EnglobAll()const ;

        cTplValGesInit< Box2di > & EnglobBoxMaitresse();
        const cTplValGesInit< Box2di > & EnglobBoxMaitresse()const ;
    private:
        cTplValGesInit< std::string > mEnglobImMaitre;
        cTplValGesInit< std::string > mEnglobAll;
        cTplValGesInit< Box2di > mEnglobBoxMaitresse;
};
cElXMLTree * ToXMLTree(const cEspaceResultSuperpCol &);

void  BinaryDumpInFile(ELISE_fp &,const cEspaceResultSuperpCol &);

void  BinaryUnDumpFromFile(cEspaceResultSuperpCol &,ELISE_fp &);

std::string  Mangling( cEspaceResultSuperpCol *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImages2Verif
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImages2Verif & anObj,cElXMLTree * aTree);


        std::string & X();
        const std::string & X()const ;

        std::string & Y();
        const std::string & Y()const ;

        double & ExagXY();
        const double & ExagXY()const ;
    private:
        std::string mX;
        std::string mY;
        double mExagXY;
};
cElXMLTree * ToXMLTree(const cImages2Verif &);

void  BinaryDumpInFile(ELISE_fp &,const cImages2Verif &);

void  BinaryUnDumpFromFile(cImages2Verif &,ELISE_fp &);

std::string  Mangling( cImages2Verif *);

class cVisuEcart
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVisuEcart & anObj,cElXMLTree * aTree);


        double & SzW();
        const double & SzW()const ;

        double & Exag();
        const double & Exag()const ;

        cTplValGesInit< std::string > & NameFile();
        const cTplValGesInit< std::string > & NameFile()const ;

        std::string & X();
        const std::string & X()const ;

        std::string & Y();
        const std::string & Y()const ;

        double & ExagXY();
        const double & ExagXY()const ;

        cTplValGesInit< cImages2Verif > & Images2Verif();
        const cTplValGesInit< cImages2Verif > & Images2Verif()const ;
    private:
        double mSzW;
        double mExag;
        cTplValGesInit< std::string > mNameFile;
        cTplValGesInit< cImages2Verif > mImages2Verif;
};
cElXMLTree * ToXMLTree(const cVisuEcart &);

void  BinaryDumpInFile(ELISE_fp &,const cVisuEcart &);

void  BinaryUnDumpFromFile(cVisuEcart &,ELISE_fp &);

std::string  Mangling( cVisuEcart *);

class cVerifHoms
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVerifHoms & anObj,cElXMLTree * aTree);


        std::string & NameOrKeyHomologues();
        const std::string & NameOrKeyHomologues()const ;

        double & SzW();
        const double & SzW()const ;

        double & Exag();
        const double & Exag()const ;

        cTplValGesInit< std::string > & NameFile();
        const cTplValGesInit< std::string > & NameFile()const ;

        std::string & X();
        const std::string & X()const ;

        std::string & Y();
        const std::string & Y()const ;

        double & ExagXY();
        const double & ExagXY()const ;

        cTplValGesInit< cImages2Verif > & Images2Verif();
        const cTplValGesInit< cImages2Verif > & Images2Verif()const ;

        cTplValGesInit< cVisuEcart > & VisuEcart();
        const cTplValGesInit< cVisuEcart > & VisuEcart()const ;
    private:
        std::string mNameOrKeyHomologues;
        cTplValGesInit< cVisuEcart > mVisuEcart;
};
cElXMLTree * ToXMLTree(const cVerifHoms &);

void  BinaryDumpInFile(ELISE_fp &,const cVerifHoms &);

void  BinaryUnDumpFromFile(cVerifHoms &,ELISE_fp &);

std::string  Mangling( cVerifHoms *);

class cImSec
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImSec & anObj,cElXMLTree * aTree);


        cImageCmpCol & Im();
        const cImageCmpCol & Im()const ;

        std::string & KeyCalcNameCorresp();
        const std::string & KeyCalcNameCorresp()const ;

        cTplValGesInit< Pt2dr > & OffsetPt();
        const cTplValGesInit< Pt2dr > & OffsetPt()const ;

        cTplValGesInit< std::string > & DirCalcCorrep();
        const cTplValGesInit< std::string > & DirCalcCorrep()const ;

        std::string & NameOrKeyHomologues();
        const std::string & NameOrKeyHomologues()const ;

        double & SzW();
        const double & SzW()const ;

        double & Exag();
        const double & Exag()const ;

        cTplValGesInit< std::string > & NameFile();
        const cTplValGesInit< std::string > & NameFile()const ;

        std::string & X();
        const std::string & X()const ;

        std::string & Y();
        const std::string & Y()const ;

        double & ExagXY();
        const double & ExagXY()const ;

        cTplValGesInit< cImages2Verif > & Images2Verif();
        const cTplValGesInit< cImages2Verif > & Images2Verif()const ;

        cTplValGesInit< cVisuEcart > & VisuEcart();
        const cTplValGesInit< cVisuEcart > & VisuEcart()const ;

        cTplValGesInit< cVerifHoms > & VerifHoms();
        const cTplValGesInit< cVerifHoms > & VerifHoms()const ;

        cTplValGesInit< int > & NbTestRansacEstimH();
        const cTplValGesInit< int > & NbTestRansacEstimH()const ;

        cTplValGesInit< int > & NbPtsRansacEstimH();
        const cTplValGesInit< int > & NbPtsRansacEstimH()const ;

        cTplValGesInit< bool > & L2EstimH();
        const cTplValGesInit< bool > & L2EstimH()const ;

        cTplValGesInit< bool > & L1EstimH();
        const cTplValGesInit< bool > & L1EstimH()const ;

        std::list< Pt2dr > & PonderaL2Iter();
        const std::list< Pt2dr > & PonderaL2Iter()const ;
    private:
        cImageCmpCol mIm;
        std::string mKeyCalcNameCorresp;
        cTplValGesInit< Pt2dr > mOffsetPt;
        cTplValGesInit< std::string > mDirCalcCorrep;
        cTplValGesInit< cVerifHoms > mVerifHoms;
        cTplValGesInit< int > mNbTestRansacEstimH;
        cTplValGesInit< int > mNbPtsRansacEstimH;
        cTplValGesInit< bool > mL2EstimH;
        cTplValGesInit< bool > mL1EstimH;
        std::list< Pt2dr > mPonderaL2Iter;
};
cElXMLTree * ToXMLTree(const cImSec &);

void  BinaryDumpInFile(ELISE_fp &,const cImSec &);

void  BinaryUnDumpFromFile(cImSec &,ELISE_fp &);

std::string  Mangling( cImSec *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCreateCompColoree
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCreateCompColoree & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        std::list< cCmdMappeur > & MapCCC();
        const std::list< cCmdMappeur > & MapCCC()const ;

        cTplValGesInit< double > & ParamBiCub();
        const cTplValGesInit< double > & ParamBiCub()const ;

        double & StepGrid();
        const double & StepGrid()const ;

        std::string & WorkDir();
        const std::string & WorkDir()const ;

        std::list< cShowCalibsRel > & ShowCalibsRel();
        const std::list< cShowCalibsRel > & ShowCalibsRel()const ;

        std::list< cResultCompCol > & ResultCompCol();
        const std::list< cResultCompCol > & ResultCompCol()const ;

        std::string & KeyCalcNameCalib();
        const std::string & KeyCalcNameCalib()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        cImageCmpCol & ImMaitresse();
        const cImageCmpCol & ImMaitresse()const ;

        cTplValGesInit< std::string > & EnglobImMaitre();
        const cTplValGesInit< std::string > & EnglobImMaitre()const ;

        cTplValGesInit< std::string > & EnglobAll();
        const cTplValGesInit< std::string > & EnglobAll()const ;

        cTplValGesInit< Box2di > & EnglobBoxMaitresse();
        const cTplValGesInit< Box2di > & EnglobBoxMaitresse()const ;

        cEspaceResultSuperpCol & EspaceResultSuperpCol();
        const cEspaceResultSuperpCol & EspaceResultSuperpCol()const ;

        cTplValGesInit< Box2di > & BoxCalc();
        const cTplValGesInit< Box2di > & BoxCalc()const ;

        cTplValGesInit< int > & TailleBloc();
        const cTplValGesInit< int > & TailleBloc()const ;

        cTplValGesInit< int > & KBoxParal();
        const cTplValGesInit< int > & KBoxParal()const ;

        cTplValGesInit< int > & ByProcess();
        const cTplValGesInit< int > & ByProcess()const ;

        cTplValGesInit< bool > & CorDist();
        const cTplValGesInit< bool > & CorDist()const ;

        cTplValGesInit< double > & ScaleFus();
        const cTplValGesInit< double > & ScaleFus()const ;

        std::list< cImSec > & ImSec();
        const std::list< cImSec > & ImSec()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        std::list< cCmdMappeur > mMapCCC;
        cTplValGesInit< double > mParamBiCub;
        double mStepGrid;
        std::string mWorkDir;
        std::list< cShowCalibsRel > mShowCalibsRel;
        std::list< cResultCompCol > mResultCompCol;
        std::string mKeyCalcNameCalib;
        cTplValGesInit< string > mFileChantierNameDescripteur;
        cImageCmpCol mImMaitresse;
        cEspaceResultSuperpCol mEspaceResultSuperpCol;
        cTplValGesInit< Box2di > mBoxCalc;
        cTplValGesInit< int > mTailleBloc;
        cTplValGesInit< int > mKBoxParal;
        cTplValGesInit< int > mByProcess;
        cTplValGesInit< bool > mCorDist;
        cTplValGesInit< double > mScaleFus;
        std::list< cImSec > mImSec;
};
cElXMLTree * ToXMLTree(const cCreateCompColoree &);

void  BinaryDumpInFile(ELISE_fp &,const cCreateCompColoree &);

void  BinaryUnDumpFromFile(cCreateCompColoree &,ELISE_fp &);

std::string  Mangling( cCreateCompColoree *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSauvegardeMR2A
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSauvegardeMR2A & anObj,cElXMLTree * aTree);


        std::string & NameSauvMR2A();
        const std::string & NameSauvMR2A()const ;

        double & StepGridMR2A();
        const double & StepGridMR2A()const ;

        cTplValGesInit< std::string > & SauvImgMR2A();
        const cTplValGesInit< std::string > & SauvImgMR2A()const ;
    private:
        std::string mNameSauvMR2A;
        double mStepGridMR2A;
        cTplValGesInit< std::string > mSauvImgMR2A;
};
cElXMLTree * ToXMLTree(const cSauvegardeMR2A &);

void  BinaryDumpInFile(ELISE_fp &,const cSauvegardeMR2A &);

void  BinaryUnDumpFromFile(cSauvegardeMR2A &,ELISE_fp &);

std::string  Mangling( cSauvegardeMR2A *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGenereModeleRaster2Analytique
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenereModeleRaster2Analytique & anObj,cElXMLTree * aTree);


        std::string & Dir();
        const std::string & Dir()const ;

        std::string & Im1();
        const std::string & Im1()const ;

        std::string & Im2();
        const std::string & Im2()const ;

        double & SsResol();
        const double & SsResol()const ;

        Pt2dr & Pas();
        const Pt2dr & Pas()const ;

        cTplValGesInit< Pt2dr > & Tr0();
        const cTplValGesInit< Pt2dr > & Tr0()const ;

        cTplValGesInit< bool > & AutoCalcTr0();
        const cTplValGesInit< bool > & AutoCalcTr0()const ;

        cTplValGesInit< double > & RoundTr0();
        const cTplValGesInit< double > & RoundTr0()const ;

        int & DegPoly();
        const int & DegPoly()const ;

        bool & CLibre();
        const bool & CLibre()const ;

        bool & Dequant();
        const bool & Dequant()const ;

        std::string & NameSauvMR2A();
        const std::string & NameSauvMR2A()const ;

        double & StepGridMR2A();
        const double & StepGridMR2A()const ;

        cTplValGesInit< std::string > & SauvImgMR2A();
        const cTplValGesInit< std::string > & SauvImgMR2A()const ;

        cTplValGesInit< cSauvegardeMR2A > & SauvegardeMR2A();
        const cTplValGesInit< cSauvegardeMR2A > & SauvegardeMR2A()const ;
    private:
        std::string mDir;
        std::string mIm1;
        std::string mIm2;
        double mSsResol;
        Pt2dr mPas;
        cTplValGesInit< Pt2dr > mTr0;
        cTplValGesInit< bool > mAutoCalcTr0;
        cTplValGesInit< double > mRoundTr0;
        int mDegPoly;
        bool mCLibre;
        bool mDequant;
        cTplValGesInit< cSauvegardeMR2A > mSauvegardeMR2A;
};
cElXMLTree * ToXMLTree(const cGenereModeleRaster2Analytique &);

void  BinaryDumpInFile(ELISE_fp &,const cGenereModeleRaster2Analytique &);

void  BinaryUnDumpFromFile(cGenereModeleRaster2Analytique &,ELISE_fp &);

std::string  Mangling( cGenereModeleRaster2Analytique *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBayerGridDirecteEtInverse
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBayerGridDirecteEtInverse & anObj,cElXMLTree * aTree);


        std::string & Ch1();
        const std::string & Ch1()const ;

        std::string & Ch2();
        const std::string & Ch2()const ;

        cGridDirecteEtInverse & Grid();
        const cGridDirecteEtInverse & Grid()const ;
    private:
        std::string mCh1;
        std::string mCh2;
        cGridDirecteEtInverse mGrid;
};
cElXMLTree * ToXMLTree(const cBayerGridDirecteEtInverse &);

void  BinaryDumpInFile(ELISE_fp &,const cBayerGridDirecteEtInverse &);

void  BinaryUnDumpFromFile(cBayerGridDirecteEtInverse &,ELISE_fp &);

std::string  Mangling( cBayerGridDirecteEtInverse *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBayerCalibGeom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBayerCalibGeom & anObj,cElXMLTree * aTree);


        std::list< cBayerGridDirecteEtInverse > & Grids();
        const std::list< cBayerGridDirecteEtInverse > & Grids()const ;

        cTplValGesInit< Pt3dr > & WB();
        const cTplValGesInit< Pt3dr > & WB()const ;

        cTplValGesInit< Pt3dr > & PG();
        const cTplValGesInit< Pt3dr > & PG()const ;
    private:
        std::list< cBayerGridDirecteEtInverse > mGrids;
        cTplValGesInit< Pt3dr > mWB;
        cTplValGesInit< Pt3dr > mPG;
};
cElXMLTree * ToXMLTree(const cBayerCalibGeom &);

void  BinaryDumpInFile(ELISE_fp &,const cBayerCalibGeom &);

void  BinaryUnDumpFromFile(cBayerCalibGeom &,ELISE_fp &);

std::string  Mangling( cBayerCalibGeom *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSpecifEtalRelOneChan
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSpecifEtalRelOneChan & anObj,cElXMLTree * aTree);


        int & DegreOwn();
        const int & DegreOwn()const ;

        int & DegreOther();
        const int & DegreOther()const ;
    private:
        int mDegreOwn;
        int mDegreOther;
};
cElXMLTree * ToXMLTree(const cSpecifEtalRelOneChan &);

void  BinaryDumpInFile(ELISE_fp &,const cSpecifEtalRelOneChan &);

void  BinaryUnDumpFromFile(cSpecifEtalRelOneChan &,ELISE_fp &);

std::string  Mangling( cSpecifEtalRelOneChan *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSpecifEtalRadiom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSpecifEtalRadiom & anObj,cElXMLTree * aTree);


        std::list< cSpecifEtalRelOneChan > & Channel();
        const std::list< cSpecifEtalRelOneChan > & Channel()const ;
    private:
        std::list< cSpecifEtalRelOneChan > mChannel;
};
cElXMLTree * ToXMLTree(const cSpecifEtalRadiom &);

void  BinaryDumpInFile(ELISE_fp &,const cSpecifEtalRadiom &);

void  BinaryUnDumpFromFile(cSpecifEtalRadiom &,ELISE_fp &);

std::string  Mangling( cSpecifEtalRadiom *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPolyNRadiom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPolyNRadiom & anObj,cElXMLTree * aTree);


        std::vector< int > & Degre();
        const std::vector< int > & Degre()const ;

        double & Val();
        const double & Val()const ;
    private:
        std::vector< int > mDegre;
        double mVal;
};
cElXMLTree * ToXMLTree(const cPolyNRadiom &);

void  BinaryDumpInFile(ELISE_fp &,const cPolyNRadiom &);

void  BinaryUnDumpFromFile(cPolyNRadiom &,ELISE_fp &);

std::string  Mangling( cPolyNRadiom *);

/******************************************************/
/******************************************************/
/******************************************************/
class cEtalRelOneChan
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEtalRelOneChan & anObj,cElXMLTree * aTree);


        std::vector< cPolyNRadiom > & PolyNRadiom();
        const std::vector< cPolyNRadiom > & PolyNRadiom()const ;
    private:
        std::vector< cPolyNRadiom > mPolyNRadiom;
};
cElXMLTree * ToXMLTree(const cEtalRelOneChan &);

void  BinaryDumpInFile(ELISE_fp &,const cEtalRelOneChan &);

void  BinaryUnDumpFromFile(cEtalRelOneChan &,ELISE_fp &);

std::string  Mangling( cEtalRelOneChan *);

/******************************************************/
/******************************************************/
/******************************************************/
class cColorCalib
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cColorCalib & anObj,cElXMLTree * aTree);


        std::vector< cEtalRelOneChan > & CalibChannel();
        const std::vector< cEtalRelOneChan > & CalibChannel()const ;
    private:
        std::vector< cEtalRelOneChan > mCalibChannel;
};
cElXMLTree * ToXMLTree(const cColorCalib &);

void  BinaryDumpInFile(ELISE_fp &,const cColorCalib &);

void  BinaryUnDumpFromFile(cColorCalib &,ELISE_fp &);

std::string  Mangling( cColorCalib *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneGridECG
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneGridECG & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        bool & Direct();
        const bool & Direct()const ;
    private:
        std::string mName;
        bool mDirect;
};
cElXMLTree * ToXMLTree(const cOneGridECG &);

void  BinaryDumpInFile(ELISE_fp &,const cOneGridECG &);

void  BinaryUnDumpFromFile(cOneGridECG &,ELISE_fp &);

std::string  Mangling( cOneGridECG *);

/******************************************************/
/******************************************************/
/******************************************************/
class cEvalComposeGrid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEvalComposeGrid & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Directory();
        const cTplValGesInit< std::string > & Directory()const ;

        double & Dyn();
        const double & Dyn()const ;

        double & Resol();
        const double & Resol()const ;

        std::list< cOneGridECG > & OneGridECG();
        const std::list< cOneGridECG > & OneGridECG()const ;

        cTplValGesInit< std::string > & NameNorm();
        const cTplValGesInit< std::string > & NameNorm()const ;
    private:
        cTplValGesInit< std::string > mDirectory;
        double mDyn;
        double mResol;
        std::list< cOneGridECG > mOneGridECG;
        cTplValGesInit< std::string > mNameNorm;
};
cElXMLTree * ToXMLTree(const cEvalComposeGrid &);

void  BinaryDumpInFile(ELISE_fp &,const cEvalComposeGrid &);

void  BinaryUnDumpFromFile(cEvalComposeGrid &,ELISE_fp &);

std::string  Mangling( cEvalComposeGrid *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalcNomFromCouple
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalcNomFromCouple & anObj,cElXMLTree * aTree);


        std::string & Pattern2Match();
        const std::string & Pattern2Match()const ;

        cTplValGesInit< std::string > & Separateur();
        const cTplValGesInit< std::string > & Separateur()const ;

        std::string & NameCalculated();
        const std::string & NameCalculated()const ;
    private:
        std::string mPattern2Match;
        cTplValGesInit< std::string > mSeparateur;
        std::string mNameCalculated;
};
cElXMLTree * ToXMLTree(const cCalcNomFromCouple &);

void  BinaryDumpInFile(ELISE_fp &,const cCalcNomFromCouple &);

void  BinaryUnDumpFromFile(cCalcNomFromCouple &,ELISE_fp &);

std::string  Mangling( cCalcNomFromCouple *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCalcNomFromOne
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCalcNomFromOne & anObj,cElXMLTree * aTree);


        std::string & Pattern2Match();
        const std::string & Pattern2Match()const ;

        std::string & NameCalculated();
        const std::string & NameCalculated()const ;
    private:
        std::string mPattern2Match;
        std::string mNameCalculated;
};
cElXMLTree * ToXMLTree(const cCalcNomFromOne &);

void  BinaryDumpInFile(ELISE_fp &,const cCalcNomFromOne &);

void  BinaryUnDumpFromFile(cCalcNomFromOne &,ELISE_fp &);

std::string  Mangling( cCalcNomFromOne *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneResync
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneResync & anObj,cElXMLTree * aTree);


        std::string & Dir();
        const std::string & Dir()const ;

        std::string & PatSel();
        const std::string & PatSel()const ;

        std::string & PatRename();
        const std::string & PatRename()const ;

        std::string & Rename();
        const std::string & Rename()const ;
    private:
        std::string mDir;
        std::string mPatSel;
        std::string mPatRename;
        std::string mRename;
};
cElXMLTree * ToXMLTree(const cOneResync &);

void  BinaryDumpInFile(ELISE_fp &,const cOneResync &);

void  BinaryUnDumpFromFile(cOneResync &,ELISE_fp &);

std::string  Mangling( cOneResync *);

/******************************************************/
/******************************************************/
/******************************************************/
class cReSynchronImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cReSynchronImage & anObj,cElXMLTree * aTree);


        std::list< cOneResync > & OneResync();
        const std::list< cOneResync > & OneResync()const ;

        double & EcartMin();
        const double & EcartMin()const ;

        double & EcartMax();
        const double & EcartMax()const ;

        cTplValGesInit< double > & EcartRechAuto();
        const cTplValGesInit< double > & EcartRechAuto()const ;

        cTplValGesInit< double > & SigmaRechAuto();
        const cTplValGesInit< double > & SigmaRechAuto()const ;

        cTplValGesInit< double > & EcartCalcMoyRechAuto();
        const cTplValGesInit< double > & EcartCalcMoyRechAuto()const ;
    private:
        std::list< cOneResync > mOneResync;
        double mEcartMin;
        double mEcartMax;
        cTplValGesInit< double > mEcartRechAuto;
        cTplValGesInit< double > mSigmaRechAuto;
        cTplValGesInit< double > mEcartCalcMoyRechAuto;
};
cElXMLTree * ToXMLTree(const cReSynchronImage &);

void  BinaryDumpInFile(ELISE_fp &,const cReSynchronImage &);

void  BinaryUnDumpFromFile(cReSynchronImage &,ELISE_fp &);

std::string  Mangling( cReSynchronImage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlCylindreRevolution
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlCylindreRevolution & anObj,cElXMLTree * aTree);


        Pt3dr & P0();
        const Pt3dr & P0()const ;

        Pt3dr & P1();
        const Pt3dr & P1()const ;

        Pt3dr & POnCyl();
        const Pt3dr & POnCyl()const ;
    private:
        Pt3dr mP0;
        Pt3dr mP1;
        Pt3dr mPOnCyl;
};
cElXMLTree * ToXMLTree(const cXmlCylindreRevolution &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlCylindreRevolution &);

void  BinaryUnDumpFromFile(cXmlCylindreRevolution &,ELISE_fp &);

std::string  Mangling( cXmlCylindreRevolution *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlToreRevol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlToreRevol & anObj,cElXMLTree * aTree);


        cXmlCylindreRevolution & Cyl();
        const cXmlCylindreRevolution & Cyl()const ;

        Pt3dr & POriTore();
        const Pt3dr & POriTore()const ;
    private:
        cXmlCylindreRevolution mCyl;
        Pt3dr mPOriTore;
};
cElXMLTree * ToXMLTree(const cXmlToreRevol &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlToreRevol &);

void  BinaryUnDumpFromFile(cXmlToreRevol &,ELISE_fp &);

std::string  Mangling( cXmlToreRevol *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlOrthoCyl
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlOrthoCyl & anObj,cElXMLTree * aTree);


        cRepereCartesien & Repere();
        const cRepereCartesien & Repere()const ;

        Pt3dr & P0();
        const Pt3dr & P0()const ;

        Pt3dr & P1();
        const Pt3dr & P1()const ;

        bool & AngulCorr();
        const bool & AngulCorr()const ;
    private:
        cRepereCartesien mRepere;
        Pt3dr mP0;
        Pt3dr mP1;
        bool mAngulCorr;
};
cElXMLTree * ToXMLTree(const cXmlOrthoCyl &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlOrthoCyl &);

void  BinaryUnDumpFromFile(cXmlOrthoCyl &,ELISE_fp &);

std::string  Mangling( cXmlOrthoCyl *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlDescriptionAnalytique
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlDescriptionAnalytique & anObj,cElXMLTree * aTree);


        cTplValGesInit< cXmlCylindreRevolution > & Cyl();
        const cTplValGesInit< cXmlCylindreRevolution > & Cyl()const ;

        cTplValGesInit< cXmlOrthoCyl > & OrthoCyl();
        const cTplValGesInit< cXmlOrthoCyl > & OrthoCyl()const ;

        cTplValGesInit< cXmlToreRevol > & Tore();
        const cTplValGesInit< cXmlToreRevol > & Tore()const ;
    private:
        cTplValGesInit< cXmlCylindreRevolution > mCyl;
        cTplValGesInit< cXmlOrthoCyl > mOrthoCyl;
        cTplValGesInit< cXmlToreRevol > mTore;
};
cElXMLTree * ToXMLTree(const cXmlDescriptionAnalytique &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlDescriptionAnalytique &);

void  BinaryUnDumpFromFile(cXmlDescriptionAnalytique &,ELISE_fp &);

std::string  Mangling( cXmlDescriptionAnalytique *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlOneSurfaceAnalytique
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlOneSurfaceAnalytique & anObj,cElXMLTree * aTree);


        cXmlDescriptionAnalytique & XmlDescriptionAnalytique();
        const cXmlDescriptionAnalytique & XmlDescriptionAnalytique()const ;

        std::string & Id();
        const std::string & Id()const ;

        bool & VueDeLExterieur();
        const bool & VueDeLExterieur()const ;
    private:
        cXmlDescriptionAnalytique mXmlDescriptionAnalytique;
        std::string mId;
        bool mVueDeLExterieur;
};
cElXMLTree * ToXMLTree(const cXmlOneSurfaceAnalytique &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlOneSurfaceAnalytique &);

void  BinaryUnDumpFromFile(cXmlOneSurfaceAnalytique &,ELISE_fp &);

std::string  Mangling( cXmlOneSurfaceAnalytique *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlModeleSurfaceComplexe
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlModeleSurfaceComplexe & anObj,cElXMLTree * aTree);


        std::list< cXmlOneSurfaceAnalytique > & XmlOneSurfaceAnalytique();
        const std::list< cXmlOneSurfaceAnalytique > & XmlOneSurfaceAnalytique()const ;
    private:
        std::list< cXmlOneSurfaceAnalytique > mXmlOneSurfaceAnalytique;
};
cElXMLTree * ToXMLTree(const cXmlModeleSurfaceComplexe &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlModeleSurfaceComplexe &);

void  BinaryUnDumpFromFile(cXmlModeleSurfaceComplexe &,ELISE_fp &);

std::string  Mangling( cXmlModeleSurfaceComplexe *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMapByKey
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMapByKey & anObj,cElXMLTree * aTree);


        std::string & Key();
        const std::string & Key()const ;

        cTplValGesInit< bool > & DefIfFileNotExisting();
        const cTplValGesInit< bool > & DefIfFileNotExisting()const ;
    private:
        std::string mKey;
        cTplValGesInit< bool > mDefIfFileNotExisting;
};
cElXMLTree * ToXMLTree(const cMapByKey &);

void  BinaryDumpInFile(ELISE_fp &,const cMapByKey &);

void  BinaryUnDumpFromFile(cMapByKey &,ELISE_fp &);

std::string  Mangling( cMapByKey *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneAutomMapN2N
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneAutomMapN2N & anObj,cElXMLTree * aTree);


        cElRegex_Ptr & MatchPattern();
        const cElRegex_Ptr & MatchPattern()const ;

        cTplValGesInit< cElRegex_Ptr > & AutomSel();
        const cTplValGesInit< cElRegex_Ptr > & AutomSel()const ;

        std::string & Result();
        const std::string & Result()const ;
    private:
        cElRegex_Ptr mMatchPattern;
        cTplValGesInit< cElRegex_Ptr > mAutomSel;
        std::string mResult;
};
cElXMLTree * ToXMLTree(const cOneAutomMapN2N &);

void  BinaryDumpInFile(ELISE_fp &,const cOneAutomMapN2N &);

void  BinaryUnDumpFromFile(cOneAutomMapN2N &,ELISE_fp &);

std::string  Mangling( cOneAutomMapN2N *);

class cMapN2NByAutom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMapN2NByAutom & anObj,cElXMLTree * aTree);


        std::vector< cOneAutomMapN2N > & OneAutomMapN2N();
        const std::vector< cOneAutomMapN2N > & OneAutomMapN2N()const ;
    private:
        std::vector< cOneAutomMapN2N > mOneAutomMapN2N;
};
cElXMLTree * ToXMLTree(const cMapN2NByAutom &);

void  BinaryDumpInFile(ELISE_fp &,const cMapN2NByAutom &);

void  BinaryUnDumpFromFile(cMapN2NByAutom &,ELISE_fp &);

std::string  Mangling( cMapN2NByAutom *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMapName2Name
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMapName2Name & anObj,cElXMLTree * aTree);


        std::string & Key();
        const std::string & Key()const ;

        cTplValGesInit< bool > & DefIfFileNotExisting();
        const cTplValGesInit< bool > & DefIfFileNotExisting()const ;

        cTplValGesInit< cMapByKey > & MapByKey();
        const cTplValGesInit< cMapByKey > & MapByKey()const ;

        std::vector< cOneAutomMapN2N > & OneAutomMapN2N();
        const std::vector< cOneAutomMapN2N > & OneAutomMapN2N()const ;

        cTplValGesInit< cMapN2NByAutom > & MapN2NByAutom();
        const cTplValGesInit< cMapN2NByAutom > & MapN2NByAutom()const ;
    private:
        cTplValGesInit< cMapByKey > mMapByKey;
        cTplValGesInit< cMapN2NByAutom > mMapN2NByAutom;
};
cElXMLTree * ToXMLTree(const cMapName2Name &);

void  BinaryDumpInFile(ELISE_fp &,const cMapName2Name &);

void  BinaryUnDumpFromFile(cMapName2Name &,ELISE_fp &);

std::string  Mangling( cMapName2Name *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImage_Point3D
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImage_Point3D & anObj,cElXMLTree * aTree);


        std::string & Image();
        const std::string & Image()const ;

        std::string & Masq();
        const std::string & Masq()const ;
    private:
        std::string mImage;
        std::string mMasq;
};
cElXMLTree * ToXMLTree(const cImage_Point3D &);

void  BinaryDumpInFile(ELISE_fp &,const cImage_Point3D &);

void  BinaryUnDumpFromFile(cImage_Point3D &,ELISE_fp &);

std::string  Mangling( cImage_Point3D *);

class cImage_Profondeur
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImage_Profondeur & anObj,cElXMLTree * aTree);


        std::string & Image();
        const std::string & Image()const ;

        std::string & Masq();
        const std::string & Masq()const ;

        cTplValGesInit< std::string > & Correl();
        const cTplValGesInit< std::string > & Correl()const ;

        double & OrigineAlti();
        const double & OrigineAlti()const ;

        double & ResolutionAlti();
        const double & ResolutionAlti()const ;

        eModeGeomMNT & GeomRestit();
        const eModeGeomMNT & GeomRestit()const ;
    private:
        std::string mImage;
        std::string mMasq;
        cTplValGesInit< std::string > mCorrel;
        double mOrigineAlti;
        double mResolutionAlti;
        eModeGeomMNT mGeomRestit;
};
cElXMLTree * ToXMLTree(const cImage_Profondeur &);

void  BinaryDumpInFile(ELISE_fp &,const cImage_Profondeur &);

void  BinaryUnDumpFromFile(cImage_Profondeur &,ELISE_fp &);

std::string  Mangling( cImage_Profondeur *);

class cPN3M_Nuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPN3M_Nuage & anObj,cElXMLTree * aTree);


        cTplValGesInit< cImage_Point3D > & Image_Point3D();
        const cTplValGesInit< cImage_Point3D > & Image_Point3D()const ;

        cTplValGesInit< cImage_Profondeur > & Image_Profondeur();
        const cTplValGesInit< cImage_Profondeur > & Image_Profondeur()const ;

        cTplValGesInit< bool > & EmptyPN3M();
        const cTplValGesInit< bool > & EmptyPN3M()const ;
    private:
        cTplValGesInit< cImage_Point3D > mImage_Point3D;
        cTplValGesInit< cImage_Profondeur > mImage_Profondeur;
        cTplValGesInit< bool > mEmptyPN3M;
};
cElXMLTree * ToXMLTree(const cPN3M_Nuage &);

void  BinaryDumpInFile(ELISE_fp &,const cPN3M_Nuage &);

void  BinaryUnDumpFromFile(cPN3M_Nuage &,ELISE_fp &);

std::string  Mangling( cPN3M_Nuage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAttributsNuage3D
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAttributsNuage3D & anObj,cElXMLTree * aTree);


        std::string & NameFileImage();
        const std::string & NameFileImage()const ;

        cTplValGesInit< bool > & AddDir2Name();
        const cTplValGesInit< bool > & AddDir2Name()const ;

        cTplValGesInit< double > & Dyn();
        const cTplValGesInit< double > & Dyn()const ;

        cTplValGesInit< double > & Scale();
        const cTplValGesInit< double > & Scale()const ;
    private:
        std::string mNameFileImage;
        cTplValGesInit< bool > mAddDir2Name;
        cTplValGesInit< double > mDyn;
        cTplValGesInit< double > mScale;
};
cElXMLTree * ToXMLTree(const cAttributsNuage3D &);

void  BinaryDumpInFile(ELISE_fp &,const cAttributsNuage3D &);

void  BinaryUnDumpFromFile(cAttributsNuage3D &,ELISE_fp &);

std::string  Mangling( cAttributsNuage3D *);

/******************************************************/
/******************************************************/
/******************************************************/
class cModeFaisceauxImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModeFaisceauxImage & anObj,cElXMLTree * aTree);


        Pt3dr & DirFaisceaux();
        const Pt3dr & DirFaisceaux()const ;

        bool & ZIsInverse();
        const bool & ZIsInverse()const ;

        cTplValGesInit< bool > & IsSpherik();
        const cTplValGesInit< bool > & IsSpherik()const ;

        cTplValGesInit< Pt2dr > & DirTrans();
        const cTplValGesInit< Pt2dr > & DirTrans()const ;
    private:
        Pt3dr mDirFaisceaux;
        bool mZIsInverse;
        cTplValGesInit< bool > mIsSpherik;
        cTplValGesInit< Pt2dr > mDirTrans;
};
cElXMLTree * ToXMLTree(const cModeFaisceauxImage &);

void  BinaryDumpInFile(ELISE_fp &,const cModeFaisceauxImage &);

void  BinaryUnDumpFromFile(cModeFaisceauxImage &,ELISE_fp &);

std::string  Mangling( cModeFaisceauxImage *);

class cPM3D_ParamSpecifs
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPM3D_ParamSpecifs & anObj,cElXMLTree * aTree);


        Pt3dr & DirFaisceaux();
        const Pt3dr & DirFaisceaux()const ;

        bool & ZIsInverse();
        const bool & ZIsInverse()const ;

        cTplValGesInit< bool > & IsSpherik();
        const cTplValGesInit< bool > & IsSpherik()const ;

        cTplValGesInit< Pt2dr > & DirTrans();
        const cTplValGesInit< Pt2dr > & DirTrans()const ;

        cTplValGesInit< cModeFaisceauxImage > & ModeFaisceauxImage();
        const cTplValGesInit< cModeFaisceauxImage > & ModeFaisceauxImage()const ;

        cTplValGesInit< std::string > & NoParamSpecif();
        const cTplValGesInit< std::string > & NoParamSpecif()const ;
    private:
        cTplValGesInit< cModeFaisceauxImage > mModeFaisceauxImage;
        cTplValGesInit< std::string > mNoParamSpecif;
};
cElXMLTree * ToXMLTree(const cPM3D_ParamSpecifs &);

void  BinaryDumpInFile(ELISE_fp &,const cPM3D_ParamSpecifs &);

void  BinaryUnDumpFromFile(cPM3D_ParamSpecifs &,ELISE_fp &);

std::string  Mangling( cPM3D_ParamSpecifs *);

/******************************************************/
/******************************************************/
/******************************************************/
class cVerifNuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cVerifNuage & anObj,cElXMLTree * aTree);


        Pt2dr & IndIm();
        const Pt2dr & IndIm()const ;

        double & Profondeur();
        const double & Profondeur()const ;

        Pt3dr & PointEuclid();
        const Pt3dr & PointEuclid()const ;
    private:
        Pt2dr mIndIm;
        double mProfondeur;
        Pt3dr mPointEuclid;
};
cElXMLTree * ToXMLTree(const cVerifNuage &);

void  BinaryDumpInFile(ELISE_fp &,const cVerifNuage &);

void  BinaryUnDumpFromFile(cVerifNuage &,ELISE_fp &);

std::string  Mangling( cVerifNuage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXML_ParamNuage3DMaille
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXML_ParamNuage3DMaille & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & SsResolRef();
        const cTplValGesInit< double > & SsResolRef()const ;

        cTplValGesInit< bool > & Empty();
        const cTplValGesInit< bool > & Empty()const ;

        Pt2di & NbPixel();
        const Pt2di & NbPixel()const ;

        cTplValGesInit< cImage_Point3D > & Image_Point3D();
        const cTplValGesInit< cImage_Point3D > & Image_Point3D()const ;

        cTplValGesInit< cImage_Profondeur > & Image_Profondeur();
        const cTplValGesInit< cImage_Profondeur > & Image_Profondeur()const ;

        cTplValGesInit< bool > & EmptyPN3M();
        const cTplValGesInit< bool > & EmptyPN3M()const ;

        cPN3M_Nuage & PN3M_Nuage();
        const cPN3M_Nuage & PN3M_Nuage()const ;

        std::list< cAttributsNuage3D > & AttributsNuage3D();
        const std::list< cAttributsNuage3D > & AttributsNuage3D()const ;

        cTplValGesInit< cRepereCartesien > & RepereGlob();
        const cTplValGesInit< cRepereCartesien > & RepereGlob()const ;

        cTplValGesInit< cXmlOneSurfaceAnalytique > & Anam();
        const cTplValGesInit< cXmlOneSurfaceAnalytique > & Anam()const ;

        cOrientationConique & Orientation();
        const cOrientationConique & Orientation()const ;

        cTplValGesInit< double > & RatioResolAltiPlani();
        const cTplValGesInit< double > & RatioResolAltiPlani()const ;

        Pt3dr & DirFaisceaux();
        const Pt3dr & DirFaisceaux()const ;

        bool & ZIsInverse();
        const bool & ZIsInverse()const ;

        cTplValGesInit< bool > & IsSpherik();
        const cTplValGesInit< bool > & IsSpherik()const ;

        cTplValGesInit< Pt2dr > & DirTrans();
        const cTplValGesInit< Pt2dr > & DirTrans()const ;

        cTplValGesInit< cModeFaisceauxImage > & ModeFaisceauxImage();
        const cTplValGesInit< cModeFaisceauxImage > & ModeFaisceauxImage()const ;

        cTplValGesInit< std::string > & NoParamSpecif();
        const cTplValGesInit< std::string > & NoParamSpecif()const ;

        cPM3D_ParamSpecifs & PM3D_ParamSpecifs();
        const cPM3D_ParamSpecifs & PM3D_ParamSpecifs()const ;

        cTplValGesInit< double > & TolVerifNuage();
        const cTplValGesInit< double > & TolVerifNuage()const ;

        std::list< cVerifNuage > & VerifNuage();
        const std::list< cVerifNuage > & VerifNuage()const ;
    private:
        cTplValGesInit< double > mSsResolRef;
        cTplValGesInit< bool > mEmpty;
        Pt2di mNbPixel;
        cPN3M_Nuage mPN3M_Nuage;
        std::list< cAttributsNuage3D > mAttributsNuage3D;
        cTplValGesInit< cRepereCartesien > mRepereGlob;
        cTplValGesInit< cXmlOneSurfaceAnalytique > mAnam;
        cOrientationConique mOrientation;
        cTplValGesInit< double > mRatioResolAltiPlani;
        cPM3D_ParamSpecifs mPM3D_ParamSpecifs;
        cTplValGesInit< double > mTolVerifNuage;
        std::list< cVerifNuage > mVerifNuage;
};
cElXMLTree * ToXMLTree(const cXML_ParamNuage3DMaille &);

void  BinaryDumpInFile(ELISE_fp &,const cXML_ParamNuage3DMaille &);

void  BinaryUnDumpFromFile(cXML_ParamNuage3DMaille &,ELISE_fp &);

std::string  Mangling( cXML_ParamNuage3DMaille *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMasqMesures
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMasqMesures & anObj,cElXMLTree * aTree);


        std::string & NameFile();
        const std::string & NameFile()const ;

        std::string & NameMTD();
        const std::string & NameMTD()const ;
    private:
        std::string mNameFile;
        std::string mNameMTD;
};
cElXMLTree * ToXMLTree(const cMasqMesures &);

void  BinaryDumpInFile(ELISE_fp &,const cMasqMesures &);

void  BinaryUnDumpFromFile(cMasqMesures &,ELISE_fp &);

std::string  Mangling( cMasqMesures *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCielVisible
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCielVisible & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & UnUsed();
        const cTplValGesInit< std::string > & UnUsed()const ;
    private:
        cTplValGesInit< std::string > mUnUsed;
};
cElXMLTree * ToXMLTree(const cCielVisible &);

void  BinaryDumpInFile(ELISE_fp &,const cCielVisible &);

void  BinaryUnDumpFromFile(cCielVisible &,ELISE_fp &);

std::string  Mangling( cCielVisible *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXML_ParamOmbrageNuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXML_ParamOmbrageNuage & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & ScaleMaxPyr();
        const cTplValGesInit< int > & ScaleMaxPyr()const ;

        cTplValGesInit< double > & StepScale();
        const cTplValGesInit< double > & StepScale()const ;

        cTplValGesInit< double > & RatioOct();
        const cTplValGesInit< double > & RatioOct()const ;

        cTplValGesInit< std::string > & UnUsed();
        const cTplValGesInit< std::string > & UnUsed()const ;

        cTplValGesInit< cCielVisible > & CielVisible();
        const cTplValGesInit< cCielVisible > & CielVisible()const ;
    private:
        cTplValGesInit< int > mScaleMaxPyr;
        cTplValGesInit< double > mStepScale;
        cTplValGesInit< double > mRatioOct;
        cTplValGesInit< cCielVisible > mCielVisible;
};
cElXMLTree * ToXMLTree(const cXML_ParamOmbrageNuage &);

void  BinaryDumpInFile(ELISE_fp &,const cXML_ParamOmbrageNuage &);

void  BinaryUnDumpFromFile(cXML_ParamOmbrageNuage &,ELISE_fp &);

std::string  Mangling( cXML_ParamOmbrageNuage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFTrajParamInit2Actuelle
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFTrajParamInit2Actuelle & anObj,cElXMLTree * aTree);


        double & Lambda();
        const double & Lambda()const ;

        cOrientationExterneRigide & Orient();
        const cOrientationExterneRigide & Orient()const ;
    private:
        double mLambda;
        cOrientationExterneRigide mOrient;
};
cElXMLTree * ToXMLTree(const cFTrajParamInit2Actuelle &);

void  BinaryDumpInFile(ELISE_fp &,const cFTrajParamInit2Actuelle &);

void  BinaryUnDumpFromFile(cFTrajParamInit2Actuelle &,ELISE_fp &);

std::string  Mangling( cFTrajParamInit2Actuelle *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPtTrajecto
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPtTrajecto & anObj,cElXMLTree * aTree);


        Pt3dr & Pt();
        const Pt3dr & Pt()const ;

        std::string & IdImage();
        const std::string & IdImage()const ;

        std::string & IdBande();
        const std::string & IdBande()const ;

        double & Time();
        const double & Time()const ;
    private:
        Pt3dr mPt;
        std::string mIdImage;
        std::string mIdBande;
        double mTime;
};
cElXMLTree * ToXMLTree(const cPtTrajecto &);

void  BinaryDumpInFile(ELISE_fp &,const cPtTrajecto &);

void  BinaryUnDumpFromFile(cPtTrajecto &,ELISE_fp &);

std::string  Mangling( cPtTrajecto *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFichier_Trajecto
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFichier_Trajecto & anObj,cElXMLTree * aTree);


        std::string & NameInit();
        const std::string & NameInit()const ;

        double & Lambda();
        const double & Lambda()const ;

        cOrientationExterneRigide & Orient();
        const cOrientationExterneRigide & Orient()const ;

        cFTrajParamInit2Actuelle & FTrajParamInit2Actuelle();
        const cFTrajParamInit2Actuelle & FTrajParamInit2Actuelle()const ;

        std::map< std::string,cPtTrajecto > & PtTrajecto();
        const std::map< std::string,cPtTrajecto > & PtTrajecto()const ;
    private:
        std::string mNameInit;
        cFTrajParamInit2Actuelle mFTrajParamInit2Actuelle;
        std::map< std::string,cPtTrajecto > mPtTrajecto;
};
cElXMLTree * ToXMLTree(const cFichier_Trajecto &);

void  BinaryDumpInFile(ELISE_fp &,const cFichier_Trajecto &);

void  BinaryUnDumpFromFile(cFichier_Trajecto &,ELISE_fp &);

std::string  Mangling( cFichier_Trajecto *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionEntree
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionEntree & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & FileMNT();
        const cTplValGesInit< std::string > & FileMNT()const ;

        std::string & KeySetIm();
        const std::string & KeySetIm()const ;

        std::string & KeyAssocMetaData();
        const std::string & KeyAssocMetaData()const ;

        std::string & KeyAssocNamePC();
        const std::string & KeyAssocNamePC()const ;

        std::string & KeyAssocNameIncH();
        const std::string & KeyAssocNameIncH()const ;

        cTplValGesInit< std::string > & KeyAssocPriorite();
        const cTplValGesInit< std::string > & KeyAssocPriorite()const ;

        std::list< cMasqMesures > & ListMasqMesures();
        const std::list< cMasqMesures > & ListMasqMesures()const ;

        std::list< std::string > & FileExterneMasqMesures();
        const std::list< std::string > & FileExterneMasqMesures()const ;
    private:
        cTplValGesInit< std::string > mFileMNT;
        std::string mKeySetIm;
        std::string mKeyAssocMetaData;
        std::string mKeyAssocNamePC;
        std::string mKeyAssocNameIncH;
        cTplValGesInit< std::string > mKeyAssocPriorite;
        std::list< cMasqMesures > mListMasqMesures;
        std::list< std::string > mFileExterneMasqMesures;
};
cElXMLTree * ToXMLTree(const cSectionEntree &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionEntree &);

void  BinaryUnDumpFromFile(cSectionEntree &,ELISE_fp &);

std::string  Mangling( cSectionEntree *);

/******************************************************/
/******************************************************/
/******************************************************/
class cBoucheTrou
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cBoucheTrou & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & SeuilVisib();
        const cTplValGesInit< int > & SeuilVisib()const ;

        cTplValGesInit< int > & SeuilVisibBT();
        const cTplValGesInit< int > & SeuilVisibBT()const ;

        cTplValGesInit< double > & CoeffPondAngul();
        const cTplValGesInit< double > & CoeffPondAngul()const ;
    private:
        cTplValGesInit< int > mSeuilVisib;
        cTplValGesInit< int > mSeuilVisibBT;
        cTplValGesInit< double > mCoeffPondAngul;
};
cElXMLTree * ToXMLTree(const cBoucheTrou &);

void  BinaryDumpInFile(ELISE_fp &,const cBoucheTrou &);

void  BinaryUnDumpFromFile(cBoucheTrou &,ELISE_fp &);

std::string  Mangling( cBoucheTrou *);

class cSectionFiltrageIn
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionFiltrageIn & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & SaturThreshold();
        const cTplValGesInit< double > & SaturThreshold()const ;

        cTplValGesInit< int > & SzDilatPC();
        const cTplValGesInit< int > & SzDilatPC()const ;

        cTplValGesInit< int > & SzOuvPC();
        const cTplValGesInit< int > & SzOuvPC()const ;

        cTplValGesInit< int > & SeuilVisib();
        const cTplValGesInit< int > & SeuilVisib()const ;

        cTplValGesInit< int > & SeuilVisibBT();
        const cTplValGesInit< int > & SeuilVisibBT()const ;

        cTplValGesInit< double > & CoeffPondAngul();
        const cTplValGesInit< double > & CoeffPondAngul()const ;

        cTplValGesInit< cBoucheTrou > & BoucheTrou();
        const cTplValGesInit< cBoucheTrou > & BoucheTrou()const ;
    private:
        cTplValGesInit< double > mSaturThreshold;
        cTplValGesInit< int > mSzDilatPC;
        cTplValGesInit< int > mSzOuvPC;
        cTplValGesInit< cBoucheTrou > mBoucheTrou;
};
cElXMLTree * ToXMLTree(const cSectionFiltrageIn &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionFiltrageIn &);

void  BinaryUnDumpFromFile(cSectionFiltrageIn &,ELISE_fp &);

std::string  Mangling( cSectionFiltrageIn *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionSorties
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionSorties & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & TestDiff();
        const cTplValGesInit< bool > & TestDiff()const ;

        std::string & NameOrtho();
        const std::string & NameOrtho()const ;

        cTplValGesInit< std::string > & NameLabels();
        const cTplValGesInit< std::string > & NameLabels()const ;

        cTplValGesInit< Box2di > & BoxCalc();
        const cTplValGesInit< Box2di > & BoxCalc()const ;

        int & SzDalle();
        const int & SzDalle()const ;

        int & SzBrd();
        const int & SzBrd()const ;

        cTplValGesInit< int > & SzTileResult();
        const cTplValGesInit< int > & SzTileResult()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< double > & DynGlob();
        const cTplValGesInit< double > & DynGlob()const ;
    private:
        cTplValGesInit< bool > mTestDiff;
        std::string mNameOrtho;
        cTplValGesInit< std::string > mNameLabels;
        cTplValGesInit< Box2di > mBoxCalc;
        int mSzDalle;
        int mSzBrd;
        cTplValGesInit< int > mSzTileResult;
        cTplValGesInit< bool > mShow;
        cTplValGesInit< double > mDynGlob;
};
cElXMLTree * ToXMLTree(const cSectionSorties &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionSorties &);

void  BinaryUnDumpFromFile(cSectionSorties &,ELISE_fp &);

std::string  Mangling( cSectionSorties *);

/******************************************************/
/******************************************************/
/******************************************************/
class cNoiseSSI
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNoiseSSI & anObj,cElXMLTree * aTree);


        double & Ampl();
        const double & Ampl()const ;

        bool & Unif();
        const bool & Unif()const ;

        int & Iter();
        const int & Iter()const ;

        int & Sz();
        const int & Sz()const ;
    private:
        double mAmpl;
        bool mUnif;
        int mIter;
        int mSz;
};
cElXMLTree * ToXMLTree(const cNoiseSSI &);

void  BinaryDumpInFile(ELISE_fp &,const cNoiseSSI &);

void  BinaryUnDumpFromFile(cNoiseSSI &,ELISE_fp &);

std::string  Mangling( cNoiseSSI *);

class cSectionSimulImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionSimulImage & anObj,cElXMLTree * aTree);


        Pt2dr & Per1();
        const Pt2dr & Per1()const ;

        cTplValGesInit< Pt2dr > & Per2();
        const cTplValGesInit< Pt2dr > & Per2()const ;

        cTplValGesInit< double > & Ampl();
        const cTplValGesInit< double > & Ampl()const ;

        std::list< cNoiseSSI > & NoiseSSI();
        const std::list< cNoiseSSI > & NoiseSSI()const ;
    private:
        Pt2dr mPer1;
        cTplValGesInit< Pt2dr > mPer2;
        cTplValGesInit< double > mAmpl;
        std::list< cNoiseSSI > mNoiseSSI;
};
cElXMLTree * ToXMLTree(const cSectionSimulImage &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionSimulImage &);

void  BinaryUnDumpFromFile(cSectionSimulImage &,ELISE_fp &);

std::string  Mangling( cSectionSimulImage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGlobRappInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGlobRappInit & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DoGlob();
        const cTplValGesInit< bool > & DoGlob()const ;

        std::vector< Pt2di > & Degres();
        const std::vector< Pt2di > & Degres()const ;

        std::vector< Pt2di > & DegresSec();
        const std::vector< Pt2di > & DegresSec()const ;

        cTplValGesInit< std::string > & PatternApply();
        const cTplValGesInit< std::string > & PatternApply()const ;

        cTplValGesInit< bool > & RapelOnEgalPhys();
        const cTplValGesInit< bool > & RapelOnEgalPhys()const ;
    private:
        cTplValGesInit< bool > mDoGlob;
        std::vector< Pt2di > mDegres;
        std::vector< Pt2di > mDegresSec;
        cTplValGesInit< std::string > mPatternApply;
        cTplValGesInit< bool > mRapelOnEgalPhys;
};
cElXMLTree * ToXMLTree(const cGlobRappInit &);

void  BinaryDumpInFile(ELISE_fp &,const cGlobRappInit &);

void  BinaryUnDumpFromFile(cGlobRappInit &,ELISE_fp &);

std::string  Mangling( cGlobRappInit *);

class cSectionEgalisation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionEgalisation & anObj,cElXMLTree * aTree);


        cTplValGesInit< cMasqTerrain > & MasqApprent();
        const cTplValGesInit< cMasqTerrain > & MasqApprent()const ;

        cTplValGesInit< int > & PeriodEchant();
        const cTplValGesInit< int > & PeriodEchant()const ;

        cTplValGesInit< double > & NbPEqualMoyPerImage();
        const cTplValGesInit< double > & NbPEqualMoyPerImage()const ;

        int & SzVois();
        const int & SzVois()const ;

        std::string & NameFileMesures();
        const std::string & NameFileMesures()const ;

        cTplValGesInit< bool > & UseFileMesure();
        const cTplValGesInit< bool > & UseFileMesure()const ;

        std::vector< Pt2di > & DegresEgalVois();
        const std::vector< Pt2di > & DegresEgalVois()const ;

        std::vector< Pt2di > & DegresEgalVoisSec();
        const std::vector< Pt2di > & DegresEgalVoisSec()const ;

        cTplValGesInit< double > & PdsRappelInit();
        const cTplValGesInit< double > & PdsRappelInit()const ;

        cTplValGesInit< double > & PdsSingularite();
        const cTplValGesInit< double > & PdsSingularite()const ;

        cTplValGesInit< bool > & DoGlob();
        const cTplValGesInit< bool > & DoGlob()const ;

        std::vector< Pt2di > & Degres();
        const std::vector< Pt2di > & Degres()const ;

        std::vector< Pt2di > & DegresSec();
        const std::vector< Pt2di > & DegresSec()const ;

        cTplValGesInit< std::string > & PatternApply();
        const cTplValGesInit< std::string > & PatternApply()const ;

        cTplValGesInit< bool > & RapelOnEgalPhys();
        const cTplValGesInit< bool > & RapelOnEgalPhys()const ;

        cGlobRappInit & GlobRappInit();
        const cGlobRappInit & GlobRappInit()const ;

        bool & EgaliseSomCh();
        const bool & EgaliseSomCh()const ;

        cTplValGesInit< int > & SzMaxVois();
        const cTplValGesInit< int > & SzMaxVois()const ;

        cTplValGesInit< bool > & Use4Vois();
        const cTplValGesInit< bool > & Use4Vois()const ;

        cTplValGesInit< double > & CorrelThreshold();
        const cTplValGesInit< double > & CorrelThreshold()const ;

        cTplValGesInit< bool > & AdjL1ByCple();
        const cTplValGesInit< bool > & AdjL1ByCple()const ;

        cTplValGesInit< double > & PercCutAdjL1();
        const cTplValGesInit< double > & PercCutAdjL1()const ;

        cTplValGesInit< double > & FactMajorByCutGlob();
        const cTplValGesInit< double > & FactMajorByCutGlob()const ;
    private:
        cTplValGesInit< cMasqTerrain > mMasqApprent;
        cTplValGesInit< int > mPeriodEchant;
        cTplValGesInit< double > mNbPEqualMoyPerImage;
        int mSzVois;
        std::string mNameFileMesures;
        cTplValGesInit< bool > mUseFileMesure;
        std::vector< Pt2di > mDegresEgalVois;
        std::vector< Pt2di > mDegresEgalVoisSec;
        cTplValGesInit< double > mPdsRappelInit;
        cTplValGesInit< double > mPdsSingularite;
        cGlobRappInit mGlobRappInit;
        bool mEgaliseSomCh;
        cTplValGesInit< int > mSzMaxVois;
        cTplValGesInit< bool > mUse4Vois;
        cTplValGesInit< double > mCorrelThreshold;
        cTplValGesInit< bool > mAdjL1ByCple;
        cTplValGesInit< double > mPercCutAdjL1;
        cTplValGesInit< double > mFactMajorByCutGlob;
};
cElXMLTree * ToXMLTree(const cSectionEgalisation &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionEgalisation &);

void  BinaryUnDumpFromFile(cSectionEgalisation &,ELISE_fp &);

std::string  Mangling( cSectionEgalisation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCreateOrtho
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCreateOrtho & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        std::string & WorkDir();
        const std::string & WorkDir()const ;

        cTplValGesInit< int > & KBox0();
        const cTplValGesInit< int > & KBox0()const ;

        cTplValGesInit< std::string > & FileMNT();
        const cTplValGesInit< std::string > & FileMNT()const ;

        std::string & KeySetIm();
        const std::string & KeySetIm()const ;

        std::string & KeyAssocMetaData();
        const std::string & KeyAssocMetaData()const ;

        std::string & KeyAssocNamePC();
        const std::string & KeyAssocNamePC()const ;

        std::string & KeyAssocNameIncH();
        const std::string & KeyAssocNameIncH()const ;

        cTplValGesInit< std::string > & KeyAssocPriorite();
        const cTplValGesInit< std::string > & KeyAssocPriorite()const ;

        std::list< cMasqMesures > & ListMasqMesures();
        const std::list< cMasqMesures > & ListMasqMesures()const ;

        std::list< std::string > & FileExterneMasqMesures();
        const std::list< std::string > & FileExterneMasqMesures()const ;

        cSectionEntree & SectionEntree();
        const cSectionEntree & SectionEntree()const ;

        cTplValGesInit< double > & SaturThreshold();
        const cTplValGesInit< double > & SaturThreshold()const ;

        cTplValGesInit< int > & SzDilatPC();
        const cTplValGesInit< int > & SzDilatPC()const ;

        cTplValGesInit< int > & SzOuvPC();
        const cTplValGesInit< int > & SzOuvPC()const ;

        cTplValGesInit< int > & SeuilVisib();
        const cTplValGesInit< int > & SeuilVisib()const ;

        cTplValGesInit< int > & SeuilVisibBT();
        const cTplValGesInit< int > & SeuilVisibBT()const ;

        cTplValGesInit< double > & CoeffPondAngul();
        const cTplValGesInit< double > & CoeffPondAngul()const ;

        cTplValGesInit< cBoucheTrou > & BoucheTrou();
        const cTplValGesInit< cBoucheTrou > & BoucheTrou()const ;

        cSectionFiltrageIn & SectionFiltrageIn();
        const cSectionFiltrageIn & SectionFiltrageIn()const ;

        cTplValGesInit< bool > & TestDiff();
        const cTplValGesInit< bool > & TestDiff()const ;

        std::string & NameOrtho();
        const std::string & NameOrtho()const ;

        cTplValGesInit< std::string > & NameLabels();
        const cTplValGesInit< std::string > & NameLabels()const ;

        cTplValGesInit< Box2di > & BoxCalc();
        const cTplValGesInit< Box2di > & BoxCalc()const ;

        int & SzDalle();
        const int & SzDalle()const ;

        int & SzBrd();
        const int & SzBrd()const ;

        cTplValGesInit< int > & SzTileResult();
        const cTplValGesInit< int > & SzTileResult()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< double > & DynGlob();
        const cTplValGesInit< double > & DynGlob()const ;

        cSectionSorties & SectionSorties();
        const cSectionSorties & SectionSorties()const ;

        Pt2dr & Per1();
        const Pt2dr & Per1()const ;

        cTplValGesInit< Pt2dr > & Per2();
        const cTplValGesInit< Pt2dr > & Per2()const ;

        cTplValGesInit< double > & Ampl();
        const cTplValGesInit< double > & Ampl()const ;

        std::list< cNoiseSSI > & NoiseSSI();
        const std::list< cNoiseSSI > & NoiseSSI()const ;

        cTplValGesInit< cSectionSimulImage > & SectionSimulImage();
        const cTplValGesInit< cSectionSimulImage > & SectionSimulImage()const ;

        cTplValGesInit< cMasqTerrain > & MasqApprent();
        const cTplValGesInit< cMasqTerrain > & MasqApprent()const ;

        cTplValGesInit< int > & PeriodEchant();
        const cTplValGesInit< int > & PeriodEchant()const ;

        cTplValGesInit< double > & NbPEqualMoyPerImage();
        const cTplValGesInit< double > & NbPEqualMoyPerImage()const ;

        int & SzVois();
        const int & SzVois()const ;

        std::string & NameFileMesures();
        const std::string & NameFileMesures()const ;

        cTplValGesInit< bool > & UseFileMesure();
        const cTplValGesInit< bool > & UseFileMesure()const ;

        std::vector< Pt2di > & DegresEgalVois();
        const std::vector< Pt2di > & DegresEgalVois()const ;

        std::vector< Pt2di > & DegresEgalVoisSec();
        const std::vector< Pt2di > & DegresEgalVoisSec()const ;

        cTplValGesInit< double > & PdsRappelInit();
        const cTplValGesInit< double > & PdsRappelInit()const ;

        cTplValGesInit< double > & PdsSingularite();
        const cTplValGesInit< double > & PdsSingularite()const ;

        cTplValGesInit< bool > & DoGlob();
        const cTplValGesInit< bool > & DoGlob()const ;

        std::vector< Pt2di > & Degres();
        const std::vector< Pt2di > & Degres()const ;

        std::vector< Pt2di > & DegresSec();
        const std::vector< Pt2di > & DegresSec()const ;

        cTplValGesInit< std::string > & PatternApply();
        const cTplValGesInit< std::string > & PatternApply()const ;

        cTplValGesInit< bool > & RapelOnEgalPhys();
        const cTplValGesInit< bool > & RapelOnEgalPhys()const ;

        cGlobRappInit & GlobRappInit();
        const cGlobRappInit & GlobRappInit()const ;

        bool & EgaliseSomCh();
        const bool & EgaliseSomCh()const ;

        cTplValGesInit< int > & SzMaxVois();
        const cTplValGesInit< int > & SzMaxVois()const ;

        cTplValGesInit< bool > & Use4Vois();
        const cTplValGesInit< bool > & Use4Vois()const ;

        cTplValGesInit< double > & CorrelThreshold();
        const cTplValGesInit< double > & CorrelThreshold()const ;

        cTplValGesInit< bool > & AdjL1ByCple();
        const cTplValGesInit< bool > & AdjL1ByCple()const ;

        cTplValGesInit< double > & PercCutAdjL1();
        const cTplValGesInit< double > & PercCutAdjL1()const ;

        cTplValGesInit< double > & FactMajorByCutGlob();
        const cTplValGesInit< double > & FactMajorByCutGlob()const ;

        cTplValGesInit< cSectionEgalisation > & SectionEgalisation();
        const cTplValGesInit< cSectionEgalisation > & SectionEgalisation()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cTplValGesInit< string > mFileChantierNameDescripteur;
        std::string mWorkDir;
        cTplValGesInit< int > mKBox0;
        cSectionEntree mSectionEntree;
        cSectionFiltrageIn mSectionFiltrageIn;
        cSectionSorties mSectionSorties;
        cTplValGesInit< cSectionSimulImage > mSectionSimulImage;
        cTplValGesInit< cSectionEgalisation > mSectionEgalisation;
};
cElXMLTree * ToXMLTree(const cCreateOrtho &);

void  BinaryDumpInFile(ELISE_fp &,const cCreateOrtho &);

void  BinaryUnDumpFromFile(cCreateOrtho &,ELISE_fp &);

std::string  Mangling( cCreateOrtho *);

/******************************************************/
/******************************************************/
/******************************************************/
class cMetaDataPartiesCachees
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMetaDataPartiesCachees & anObj,cElXMLTree * aTree);


        bool & Done();
        const bool & Done()const ;

        Pt2di & Offset();
        const Pt2di & Offset()const ;

        Pt2di & Sz();
        const Pt2di & Sz()const ;

        double & Pas();
        const double & Pas()const ;

        int & SeuilUse();
        const int & SeuilUse()const ;

        cTplValGesInit< double > & SsResolIncH();
        const cTplValGesInit< double > & SsResolIncH()const ;
    private:
        bool mDone;
        Pt2di mOffset;
        Pt2di mSz;
        double mPas;
        int mSeuilUse;
        cTplValGesInit< double > mSsResolIncH;
};
cElXMLTree * ToXMLTree(const cMetaDataPartiesCachees &);

void  BinaryDumpInFile(ELISE_fp &,const cMetaDataPartiesCachees &);

void  BinaryUnDumpFromFile(cMetaDataPartiesCachees &,ELISE_fp &);

std::string  Mangling( cMetaDataPartiesCachees *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPVPN_Orientation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPVPN_Orientation & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt3dr > & AngleCardan();
        const cTplValGesInit< Pt3dr > & AngleCardan()const ;
    private:
        cTplValGesInit< Pt3dr > mAngleCardan;
};
cElXMLTree * ToXMLTree(const cPVPN_Orientation &);

void  BinaryDumpInFile(ELISE_fp &,const cPVPN_Orientation &);

void  BinaryUnDumpFromFile(cPVPN_Orientation &,ELISE_fp &);

std::string  Mangling( cPVPN_Orientation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPVPN_ImFixe
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPVPN_ImFixe & anObj,cElXMLTree * aTree);


        cPVPN_Orientation & Orient();
        const cPVPN_Orientation & Orient()const ;

        std::string & Name();
        const std::string & Name()const ;
    private:
        cPVPN_Orientation mOrient;
        std::string mName;
};
cElXMLTree * ToXMLTree(const cPVPN_ImFixe &);

void  BinaryDumpInFile(ELISE_fp &,const cPVPN_ImFixe &);

void  BinaryUnDumpFromFile(cPVPN_ImFixe &,ELISE_fp &);

std::string  Mangling( cPVPN_ImFixe *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPVPN_Camera
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPVPN_Camera & anObj,cElXMLTree * aTree);


        Pt2di & NbPixel();
        const Pt2di & NbPixel()const ;

        double & AngleDiag();
        const double & AngleDiag()const ;
    private:
        Pt2di mNbPixel;
        double mAngleDiag;
};
cElXMLTree * ToXMLTree(const cPVPN_Camera &);

void  BinaryDumpInFile(ELISE_fp &,const cPVPN_Camera &);

void  BinaryUnDumpFromFile(cPVPN_Camera &,ELISE_fp &);

std::string  Mangling( cPVPN_Camera *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPVPN_Fond
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPVPN_Fond & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt3dr > & FondConstant();
        const cTplValGesInit< Pt3dr > & FondConstant()const ;
    private:
        cTplValGesInit< Pt3dr > mFondConstant;
};
cElXMLTree * ToXMLTree(const cPVPN_Fond &);

void  BinaryDumpInFile(ELISE_fp &,const cPVPN_Fond &);

void  BinaryUnDumpFromFile(cPVPN_Fond &,ELISE_fp &);

std::string  Mangling( cPVPN_Fond *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPVPN_Nuages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPVPN_Nuages & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;
    private:
        std::string mName;
};
cElXMLTree * ToXMLTree(const cPVPN_Nuages &);

void  BinaryDumpInFile(ELISE_fp &,const cPVPN_Nuages &);

void  BinaryUnDumpFromFile(cPVPN_Nuages &,ELISE_fp &);

std::string  Mangling( cPVPN_Nuages *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamVisuProjNuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamVisuProjNuage & anObj,cElXMLTree * aTree);


        std::string & WorkDir();
        const std::string & WorkDir()const ;

        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        cTplValGesInit< string > & FileChantierNameDescripteur();
        const cTplValGesInit< string > & FileChantierNameDescripteur()const ;

        cPVPN_Orientation & Orient();
        const cPVPN_Orientation & Orient()const ;

        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< cPVPN_ImFixe > & PVPN_ImFixe();
        const cTplValGesInit< cPVPN_ImFixe > & PVPN_ImFixe()const ;

        Pt2di & NbPixel();
        const Pt2di & NbPixel()const ;

        double & AngleDiag();
        const double & AngleDiag()const ;

        cPVPN_Camera & PVPN_Camera();
        const cPVPN_Camera & PVPN_Camera()const ;

        cTplValGesInit< Pt3dr > & FondConstant();
        const cTplValGesInit< Pt3dr > & FondConstant()const ;

        cPVPN_Fond & PVPN_Fond();
        const cPVPN_Fond & PVPN_Fond()const ;

        std::list< cPVPN_Nuages > & PVPN_Nuages();
        const std::list< cPVPN_Nuages > & PVPN_Nuages()const ;

        cTplValGesInit< double > & SousEchQuickN();
        const cTplValGesInit< double > & SousEchQuickN()const ;
    private:
        std::string mWorkDir;
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cTplValGesInit< string > mFileChantierNameDescripteur;
        cTplValGesInit< cPVPN_ImFixe > mPVPN_ImFixe;
        cPVPN_Camera mPVPN_Camera;
        cPVPN_Fond mPVPN_Fond;
        std::list< cPVPN_Nuages > mPVPN_Nuages;
        cTplValGesInit< double > mSousEchQuickN;
};
cElXMLTree * ToXMLTree(const cParamVisuProjNuage &);

void  BinaryDumpInFile(ELISE_fp &,const cParamVisuProjNuage &);

void  BinaryUnDumpFromFile(cParamVisuProjNuage &,ELISE_fp &);

std::string  Mangling( cParamVisuProjNuage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPoinAvionJaune
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPoinAvionJaune & anObj,cElXMLTree * aTree);


        double & x();
        const double & x()const ;

        double & y();
        const double & y()const ;
    private:
        double mx;
        double my;
};
cElXMLTree * ToXMLTree(const cPoinAvionJaune &);

void  BinaryDumpInFile(ELISE_fp &,const cPoinAvionJaune &);

void  BinaryUnDumpFromFile(cPoinAvionJaune &,ELISE_fp &);

std::string  Mangling( cPoinAvionJaune *);

/******************************************************/
/******************************************************/
/******************************************************/
class cValueAvionJaune
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cValueAvionJaune & anObj,cElXMLTree * aTree);


        std::string & unit();
        const std::string & unit()const ;

        cTplValGesInit< std::string > & source();
        const cTplValGesInit< std::string > & source()const ;

        cTplValGesInit< double > & biaisCorrige();
        const cTplValGesInit< double > & biaisCorrige()const ;

        double & value();
        const double & value()const ;
    private:
        std::string munit;
        cTplValGesInit< std::string > msource;
        cTplValGesInit< double > mbiaisCorrige;
        double mvalue;
};
cElXMLTree * ToXMLTree(const cValueAvionJaune &);

void  BinaryDumpInFile(ELISE_fp &,const cValueAvionJaune &);

void  BinaryUnDumpFromFile(cValueAvionJaune &,ELISE_fp &);

std::string  Mangling( cValueAvionJaune *);

/******************************************************/
/******************************************************/
/******************************************************/
class cValueXYAvionJaune
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cValueXYAvionJaune & anObj,cElXMLTree * aTree);


        std::string & unit();
        const std::string & unit()const ;

        cTplValGesInit< std::string > & source();
        const cTplValGesInit< std::string > & source()const ;

        cTplValGesInit< double > & biaisCorrige();
        const cTplValGesInit< double > & biaisCorrige()const ;

        double & xvalue();
        const double & xvalue()const ;

        double & yvalue();
        const double & yvalue()const ;
    private:
        std::string munit;
        cTplValGesInit< std::string > msource;
        cTplValGesInit< double > mbiaisCorrige;
        double mxvalue;
        double myvalue;
};
cElXMLTree * ToXMLTree(const cValueXYAvionJaune &);

void  BinaryDumpInFile(ELISE_fp &,const cValueXYAvionJaune &);

void  BinaryUnDumpFromFile(cValueXYAvionJaune &,ELISE_fp &);

std::string  Mangling( cValueXYAvionJaune *);

/******************************************************/
/******************************************************/
/******************************************************/
class cnavigation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cnavigation & anObj,cElXMLTree * aTree);


        std::string & systemeGeodesique();
        const std::string & systemeGeodesique()const ;

        std::string & projection();
        const std::string & projection()const ;

        cPoinAvionJaune & sommet();
        const cPoinAvionJaune & sommet()const ;

        cValueAvionJaune & altitude();
        const cValueAvionJaune & altitude()const ;

        cValueAvionJaune & capAvion();
        const cValueAvionJaune & capAvion()const ;

        cValueAvionJaune & roulisAvion();
        const cValueAvionJaune & roulisAvion()const ;

        cValueAvionJaune & tangageAvion();
        const cValueAvionJaune & tangageAvion()const ;

        cValueAvionJaune & tempsAutopilote();
        const cValueAvionJaune & tempsAutopilote()const ;
    private:
        std::string msystemeGeodesique;
        std::string mprojection;
        cPoinAvionJaune msommet;
        cValueAvionJaune maltitude;
        cValueAvionJaune mcapAvion;
        cValueAvionJaune mroulisAvion;
        cValueAvionJaune mtangageAvion;
        cValueAvionJaune mtempsAutopilote;
};
cElXMLTree * ToXMLTree(const cnavigation &);

void  BinaryDumpInFile(ELISE_fp &,const cnavigation &);

void  BinaryUnDumpFromFile(cnavigation &,ELISE_fp &);

std::string  Mangling( cnavigation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cimage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cimage & anObj,cElXMLTree * aTree);


        cValueAvionJaune & focale();
        const cValueAvionJaune & focale()const ;

        cValueAvionJaune & ouverture();
        const cValueAvionJaune & ouverture()const ;

        cValueAvionJaune & tempsDExposition();
        const cValueAvionJaune & tempsDExposition()const ;
    private:
        cValueAvionJaune mfocale;
        cValueAvionJaune mouverture;
        cValueAvionJaune mtempsDExposition;
};
cElXMLTree * ToXMLTree(const cimage &);

void  BinaryDumpInFile(ELISE_fp &,const cimage &);

void  BinaryUnDumpFromFile(cimage &,ELISE_fp &);

std::string  Mangling( cimage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cgeometrieAPriori
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cgeometrieAPriori & anObj,cElXMLTree * aTree);


        cValueAvionJaune & hauteur();
        const cValueAvionJaune & hauteur()const ;

        cValueXYAvionJaune & resolution();
        const cValueXYAvionJaune & resolution()const ;

        cValueAvionJaune & orientationAPN();
        const cValueAvionJaune & orientationAPN()const ;

        std::vector< cPoinAvionJaune > & coin();
        const std::vector< cPoinAvionJaune > & coin()const ;
    private:
        cValueAvionJaune mhauteur;
        cValueXYAvionJaune mresolution;
        cValueAvionJaune morientationAPN;
        std::vector< cPoinAvionJaune > mcoin;
};
cElXMLTree * ToXMLTree(const cgeometrieAPriori &);

void  BinaryDumpInFile(ELISE_fp &,const cgeometrieAPriori &);

void  BinaryUnDumpFromFile(cgeometrieAPriori &,ELISE_fp &);

std::string  Mangling( cgeometrieAPriori *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAvionJauneDocument
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAvionJauneDocument & anObj,cElXMLTree * aTree);


        std::string & numeroImage();
        const std::string & numeroImage()const ;

        std::string & systemeGeodesique();
        const std::string & systemeGeodesique()const ;

        std::string & projection();
        const std::string & projection()const ;

        cPoinAvionJaune & sommet();
        const cPoinAvionJaune & sommet()const ;

        cValueAvionJaune & altitude();
        const cValueAvionJaune & altitude()const ;

        cValueAvionJaune & capAvion();
        const cValueAvionJaune & capAvion()const ;

        cValueAvionJaune & roulisAvion();
        const cValueAvionJaune & roulisAvion()const ;

        cValueAvionJaune & tangageAvion();
        const cValueAvionJaune & tangageAvion()const ;

        cValueAvionJaune & tempsAutopilote();
        const cValueAvionJaune & tempsAutopilote()const ;

        cnavigation & navigation();
        const cnavigation & navigation()const ;

        cValueAvionJaune & focale();
        const cValueAvionJaune & focale()const ;

        cValueAvionJaune & ouverture();
        const cValueAvionJaune & ouverture()const ;

        cValueAvionJaune & tempsDExposition();
        const cValueAvionJaune & tempsDExposition()const ;

        cimage & image();
        const cimage & image()const ;

        cValueAvionJaune & hauteur();
        const cValueAvionJaune & hauteur()const ;

        cValueXYAvionJaune & resolution();
        const cValueXYAvionJaune & resolution()const ;

        cValueAvionJaune & orientationAPN();
        const cValueAvionJaune & orientationAPN()const ;

        std::vector< cPoinAvionJaune > & coin();
        const std::vector< cPoinAvionJaune > & coin()const ;

        cgeometrieAPriori & geometrieAPriori();
        const cgeometrieAPriori & geometrieAPriori()const ;
    private:
        std::string mnumeroImage;
        cnavigation mnavigation;
        cimage mimage;
        cgeometrieAPriori mgeometrieAPriori;
};
cElXMLTree * ToXMLTree(const cAvionJauneDocument &);

void  BinaryDumpInFile(ELISE_fp &,const cAvionJauneDocument &);

void  BinaryUnDumpFromFile(cAvionJauneDocument &,ELISE_fp &);

std::string  Mangling( cAvionJauneDocument *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTrAJ2_GenerateOrient
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_GenerateOrient & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & Teta1FromCap();
        const cTplValGesInit< bool > & Teta1FromCap()const ;

        cTplValGesInit< double > & CorrecDelayGps();
        const cTplValGesInit< double > & CorrecDelayGps()const ;

        cTplValGesInit< bool > & ModeMatrix();
        const cTplValGesInit< bool > & ModeMatrix()const ;

        std::list< std::string > & KeyName();
        const std::list< std::string > & KeyName()const ;

        cSystemeCoord & SysCible();
        const cSystemeCoord & SysCible()const ;

        std::string & NameCalib();
        const std::string & NameCalib()const ;

        double & AltiSol();
        const double & AltiSol()const ;
    private:
        cTplValGesInit< bool > mTeta1FromCap;
        cTplValGesInit< double > mCorrecDelayGps;
        cTplValGesInit< bool > mModeMatrix;
        std::list< std::string > mKeyName;
        cSystemeCoord mSysCible;
        std::string mNameCalib;
        double mAltiSol;
};
cElXMLTree * ToXMLTree(const cTrAJ2_GenerateOrient &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_GenerateOrient &);

void  BinaryUnDumpFromFile(cTrAJ2_GenerateOrient &,ELISE_fp &);

std::string  Mangling( cTrAJ2_GenerateOrient *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTrAJ2_ModeliseVitesse
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_ModeliseVitesse & anObj,cElXMLTree * aTree);


        double & DeltaTimeMax();
        const double & DeltaTimeMax()const ;
    private:
        double mDeltaTimeMax;
};
cElXMLTree * ToXMLTree(const cTrAJ2_ModeliseVitesse &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_ModeliseVitesse &);

void  BinaryUnDumpFromFile(cTrAJ2_ModeliseVitesse &,ELISE_fp &);

std::string  Mangling( cTrAJ2_ModeliseVitesse *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTrAJ2_SectionImages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_SectionImages & anObj,cElXMLTree * aTree);


        cTplValGesInit< eConventionsOrientation > & ConvOrCam();
        const cTplValGesInit< eConventionsOrientation > & ConvOrCam()const ;

        cRotationVect & OrientationCamera();
        const cRotationVect & OrientationCamera()const ;

        std::string & KeySetIm();
        const std::string & KeySetIm()const ;

        std::string & Id();
        const std::string & Id()const ;
    private:
        cTplValGesInit< eConventionsOrientation > mConvOrCam;
        cRotationVect mOrientationCamera;
        std::string mKeySetIm;
        std::string mId;
};
cElXMLTree * ToXMLTree(const cTrAJ2_SectionImages &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_SectionImages &);

void  BinaryUnDumpFromFile(cTrAJ2_SectionImages &,ELISE_fp &);

std::string  Mangling( cTrAJ2_SectionImages *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGenerateTabExemple
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenerateTabExemple & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        Pt3di & Nb();
        const Pt3di & Nb()const ;

        cTplValGesInit< double > & ZMin();
        const cTplValGesInit< double > & ZMin()const ;

        cTplValGesInit< double > & ZMax();
        const cTplValGesInit< double > & ZMax()const ;

        cTplValGesInit< double > & DIntervZ();
        const cTplValGesInit< double > & DIntervZ()const ;

        cTplValGesInit< bool > & RandomXY();
        const cTplValGesInit< bool > & RandomXY()const ;

        cTplValGesInit< bool > & RandomZ();
        const cTplValGesInit< bool > & RandomZ()const ;
    private:
        std::string mName;
        Pt3di mNb;
        cTplValGesInit< double > mZMin;
        cTplValGesInit< double > mZMax;
        cTplValGesInit< double > mDIntervZ;
        cTplValGesInit< bool > mRandomXY;
        cTplValGesInit< bool > mRandomZ;
};
cElXMLTree * ToXMLTree(const cGenerateTabExemple &);

void  BinaryDumpInFile(ELISE_fp &,const cGenerateTabExemple &);

void  BinaryUnDumpFromFile(cGenerateTabExemple &,ELISE_fp &);

std::string  Mangling( cGenerateTabExemple *);

class cFullDate
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFullDate & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & KYear();
        const cTplValGesInit< int > & KYear()const ;

        cTplValGesInit< int > & DefYear();
        const cTplValGesInit< int > & DefYear()const ;

        cTplValGesInit< int > & KMonth();
        const cTplValGesInit< int > & KMonth()const ;

        cTplValGesInit< int > & DefMonth();
        const cTplValGesInit< int > & DefMonth()const ;

        cTplValGesInit< int > & KDay();
        const cTplValGesInit< int > & KDay()const ;

        cTplValGesInit< int > & DefDay();
        const cTplValGesInit< int > & DefDay()const ;

        int & KHour();
        const int & KHour()const ;

        int & KMin();
        const int & KMin()const ;

        int & KSec();
        const int & KSec()const ;

        cTplValGesInit< double > & DivSec();
        const cTplValGesInit< double > & DivSec()const ;

        cTplValGesInit< int > & KMiliSec();
        const cTplValGesInit< int > & KMiliSec()const ;

        cTplValGesInit< double > & DivMiliSec();
        const cTplValGesInit< double > & DivMiliSec()const ;
    private:
        cTplValGesInit< int > mKYear;
        cTplValGesInit< int > mDefYear;
        cTplValGesInit< int > mKMonth;
        cTplValGesInit< int > mDefMonth;
        cTplValGesInit< int > mKDay;
        cTplValGesInit< int > mDefDay;
        int mKHour;
        int mKMin;
        int mKSec;
        cTplValGesInit< double > mDivSec;
        cTplValGesInit< int > mKMiliSec;
        cTplValGesInit< double > mDivMiliSec;
};
cElXMLTree * ToXMLTree(const cFullDate &);

void  BinaryDumpInFile(ELISE_fp &,const cFullDate &);

void  BinaryUnDumpFromFile(cFullDate &,ELISE_fp &);

std::string  Mangling( cFullDate *);

class cSectionTime
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionTime & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NoTime();
        const cTplValGesInit< std::string > & NoTime()const ;

        cTplValGesInit< int > & KTime();
        const cTplValGesInit< int > & KTime()const ;

        cTplValGesInit< int > & KYear();
        const cTplValGesInit< int > & KYear()const ;

        cTplValGesInit< int > & DefYear();
        const cTplValGesInit< int > & DefYear()const ;

        cTplValGesInit< int > & KMonth();
        const cTplValGesInit< int > & KMonth()const ;

        cTplValGesInit< int > & DefMonth();
        const cTplValGesInit< int > & DefMonth()const ;

        cTplValGesInit< int > & KDay();
        const cTplValGesInit< int > & KDay()const ;

        cTplValGesInit< int > & DefDay();
        const cTplValGesInit< int > & DefDay()const ;

        int & KHour();
        const int & KHour()const ;

        int & KMin();
        const int & KMin()const ;

        int & KSec();
        const int & KSec()const ;

        cTplValGesInit< double > & DivSec();
        const cTplValGesInit< double > & DivSec()const ;

        cTplValGesInit< int > & KMiliSec();
        const cTplValGesInit< int > & KMiliSec()const ;

        cTplValGesInit< double > & DivMiliSec();
        const cTplValGesInit< double > & DivMiliSec()const ;

        cTplValGesInit< cFullDate > & FullDate();
        const cTplValGesInit< cFullDate > & FullDate()const ;
    private:
        cTplValGesInit< std::string > mNoTime;
        cTplValGesInit< int > mKTime;
        cTplValGesInit< cFullDate > mFullDate;
};
cElXMLTree * ToXMLTree(const cSectionTime &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionTime &);

void  BinaryUnDumpFromFile(cSectionTime &,ELISE_fp &);

std::string  Mangling( cSectionTime *);

class cTrajAngles
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrajAngles & anObj,cElXMLTree * aTree);


        eUniteAngulaire & Unites();
        const eUniteAngulaire & Unites()const ;

        eConventionsOrientation & ConvOr();
        const eConventionsOrientation & ConvOr()const ;

        int & KTeta1();
        const int & KTeta1()const ;

        int & KTeta2();
        const int & KTeta2()const ;

        int & KTeta3();
        const int & KTeta3()const ;

        cTplValGesInit< double > & OffsetTeta1();
        const cTplValGesInit< double > & OffsetTeta1()const ;

        cTplValGesInit< double > & OffsetTeta2();
        const cTplValGesInit< double > & OffsetTeta2()const ;

        cTplValGesInit< double > & OffsetTeta3();
        const cTplValGesInit< double > & OffsetTeta3()const ;

        cTplValGesInit< cRotationVect > & RefOrTrajI2C();
        const cTplValGesInit< cRotationVect > & RefOrTrajI2C()const ;
    private:
        eUniteAngulaire mUnites;
        eConventionsOrientation mConvOr;
        int mKTeta1;
        int mKTeta2;
        int mKTeta3;
        cTplValGesInit< double > mOffsetTeta1;
        cTplValGesInit< double > mOffsetTeta2;
        cTplValGesInit< double > mOffsetTeta3;
        cTplValGesInit< cRotationVect > mRefOrTrajI2C;
};
cElXMLTree * ToXMLTree(const cTrajAngles &);

void  BinaryDumpInFile(ELISE_fp &,const cTrajAngles &);

void  BinaryUnDumpFromFile(cTrajAngles &,ELISE_fp &);

std::string  Mangling( cTrajAngles *);

class cGetImInLog
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGetImInLog & anObj,cElXMLTree * aTree);


        int & KIm();
        const int & KIm()const ;
    private:
        int mKIm;
};
cElXMLTree * ToXMLTree(const cGetImInLog &);

void  BinaryDumpInFile(ELISE_fp &,const cGetImInLog &);

void  BinaryUnDumpFromFile(cGetImInLog &,ELISE_fp &);

std::string  Mangling( cGetImInLog *);

class cTrAJ2_SectionLog
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_SectionLog & anObj,cElXMLTree * aTree);


        std::list< cGenerateTabExemple > & GenerateTabExemple();
        const std::list< cGenerateTabExemple > & GenerateTabExemple()const ;

        cTplValGesInit< double > & TimeMin();
        const cTplValGesInit< double > & TimeMin()const ;

        cTplValGesInit< int > & KLogT0();
        const cTplValGesInit< int > & KLogT0()const ;

        std::string & File();
        const std::string & File()const ;

        std::string & Autom();
        const std::string & Autom()const ;

        cSystemeCoord & SysCoord();
        const cSystemeCoord & SysCoord()const ;

        std::string & Id();
        const std::string & Id()const ;

        cTplValGesInit< std::string > & NoTime();
        const cTplValGesInit< std::string > & NoTime()const ;

        cTplValGesInit< int > & KTime();
        const cTplValGesInit< int > & KTime()const ;

        cTplValGesInit< int > & KYear();
        const cTplValGesInit< int > & KYear()const ;

        cTplValGesInit< int > & DefYear();
        const cTplValGesInit< int > & DefYear()const ;

        cTplValGesInit< int > & KMonth();
        const cTplValGesInit< int > & KMonth()const ;

        cTplValGesInit< int > & DefMonth();
        const cTplValGesInit< int > & DefMonth()const ;

        cTplValGesInit< int > & KDay();
        const cTplValGesInit< int > & KDay()const ;

        cTplValGesInit< int > & DefDay();
        const cTplValGesInit< int > & DefDay()const ;

        int & KHour();
        const int & KHour()const ;

        int & KMin();
        const int & KMin()const ;

        int & KSec();
        const int & KSec()const ;

        cTplValGesInit< double > & DivSec();
        const cTplValGesInit< double > & DivSec()const ;

        cTplValGesInit< int > & KMiliSec();
        const cTplValGesInit< int > & KMiliSec()const ;

        cTplValGesInit< double > & DivMiliSec();
        const cTplValGesInit< double > & DivMiliSec()const ;

        cTplValGesInit< cFullDate > & FullDate();
        const cTplValGesInit< cFullDate > & FullDate()const ;

        cSectionTime & SectionTime();
        const cSectionTime & SectionTime()const ;

        int & KCoord1();
        const int & KCoord1()const ;

        cTplValGesInit< double > & DivCoord1();
        const cTplValGesInit< double > & DivCoord1()const ;

        int & KCoord2();
        const int & KCoord2()const ;

        cTplValGesInit< double > & DivCoord2();
        const cTplValGesInit< double > & DivCoord2()const ;

        int & KCoord3();
        const int & KCoord3()const ;

        cTplValGesInit< double > & DivCoord3();
        const cTplValGesInit< double > & DivCoord3()const ;

        std::vector< eUniteAngulaire > & UnitesCoord();
        const std::vector< eUniteAngulaire > & UnitesCoord()const ;

        eUniteAngulaire & Unites();
        const eUniteAngulaire & Unites()const ;

        eConventionsOrientation & ConvOr();
        const eConventionsOrientation & ConvOr()const ;

        int & KTeta1();
        const int & KTeta1()const ;

        int & KTeta2();
        const int & KTeta2()const ;

        int & KTeta3();
        const int & KTeta3()const ;

        cTplValGesInit< double > & OffsetTeta1();
        const cTplValGesInit< double > & OffsetTeta1()const ;

        cTplValGesInit< double > & OffsetTeta2();
        const cTplValGesInit< double > & OffsetTeta2()const ;

        cTplValGesInit< double > & OffsetTeta3();
        const cTplValGesInit< double > & OffsetTeta3()const ;

        cTplValGesInit< cRotationVect > & RefOrTrajI2C();
        const cTplValGesInit< cRotationVect > & RefOrTrajI2C()const ;

        cTplValGesInit< cTrajAngles > & TrajAngles();
        const cTplValGesInit< cTrajAngles > & TrajAngles()const ;

        int & KIm();
        const int & KIm()const ;

        cTplValGesInit< cGetImInLog > & GetImInLog();
        const cTplValGesInit< cGetImInLog > & GetImInLog()const ;
    private:
        std::list< cGenerateTabExemple > mGenerateTabExemple;
        cTplValGesInit< double > mTimeMin;
        cTplValGesInit< int > mKLogT0;
        std::string mFile;
        std::string mAutom;
        cSystemeCoord mSysCoord;
        std::string mId;
        cSectionTime mSectionTime;
        int mKCoord1;
        cTplValGesInit< double > mDivCoord1;
        int mKCoord2;
        cTplValGesInit< double > mDivCoord2;
        int mKCoord3;
        cTplValGesInit< double > mDivCoord3;
        std::vector< eUniteAngulaire > mUnitesCoord;
        cTplValGesInit< cTrajAngles > mTrajAngles;
        cTplValGesInit< cGetImInLog > mGetImInLog;
};
cElXMLTree * ToXMLTree(const cTrAJ2_SectionLog &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_SectionLog &);

void  BinaryUnDumpFromFile(cTrAJ2_SectionLog &,ELISE_fp &);

std::string  Mangling( cTrAJ2_SectionLog *);

/******************************************************/
/******************************************************/
/******************************************************/
class cLearnByExample
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLearnByExample & anObj,cElXMLTree * aTree);


        std::string & Im0();
        const std::string & Im0()const ;

        int & Log0();
        const int & Log0()const ;

        int & DeltaMinRech();
        const int & DeltaMinRech()const ;

        int & DeltaMaxRech();
        const int & DeltaMaxRech()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< bool > & ShowPerc();
        const cTplValGesInit< bool > & ShowPerc()const ;
    private:
        std::string mIm0;
        int mLog0;
        int mDeltaMinRech;
        int mDeltaMaxRech;
        cTplValGesInit< bool > mShow;
        cTplValGesInit< bool > mShowPerc;
};
cElXMLTree * ToXMLTree(const cLearnByExample &);

void  BinaryDumpInFile(ELISE_fp &,const cLearnByExample &);

void  BinaryUnDumpFromFile(cLearnByExample &,ELISE_fp &);

std::string  Mangling( cLearnByExample *);

class cLearnByStatDiff
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLearnByStatDiff & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & MaxEcart();
        const cTplValGesInit< double > & MaxEcart()const ;
    private:
        cTplValGesInit< double > mMaxEcart;
};
cElXMLTree * ToXMLTree(const cLearnByStatDiff &);

void  BinaryDumpInFile(ELISE_fp &,const cLearnByStatDiff &);

void  BinaryUnDumpFromFile(cLearnByStatDiff &,ELISE_fp &);

std::string  Mangling( cLearnByStatDiff *);

class cLearnOffset
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cLearnOffset & anObj,cElXMLTree * aTree);


        std::string & Im0();
        const std::string & Im0()const ;

        int & Log0();
        const int & Log0()const ;

        int & DeltaMinRech();
        const int & DeltaMinRech()const ;

        int & DeltaMaxRech();
        const int & DeltaMaxRech()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< bool > & ShowPerc();
        const cTplValGesInit< bool > & ShowPerc()const ;

        cTplValGesInit< cLearnByExample > & LearnByExample();
        const cTplValGesInit< cLearnByExample > & LearnByExample()const ;

        cTplValGesInit< double > & MaxEcart();
        const cTplValGesInit< double > & MaxEcart()const ;

        cTplValGesInit< cLearnByStatDiff > & LearnByStatDiff();
        const cTplValGesInit< cLearnByStatDiff > & LearnByStatDiff()const ;
    private:
        cTplValGesInit< cLearnByExample > mLearnByExample;
        cTplValGesInit< cLearnByStatDiff > mLearnByStatDiff;
};
cElXMLTree * ToXMLTree(const cLearnOffset &);

void  BinaryDumpInFile(ELISE_fp &,const cLearnOffset &);

void  BinaryUnDumpFromFile(cLearnOffset &,ELISE_fp &);

std::string  Mangling( cLearnOffset *);

class cMatchNearestIm
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMatchNearestIm & anObj,cElXMLTree * aTree);


        double & TolMatch();
        const double & TolMatch()const ;

        double & TolAmbig();
        const double & TolAmbig()const ;
    private:
        double mTolMatch;
        double mTolAmbig;
};
cElXMLTree * ToXMLTree(const cMatchNearestIm &);

void  BinaryDumpInFile(ELISE_fp &,const cMatchNearestIm &);

void  BinaryUnDumpFromFile(cMatchNearestIm &,ELISE_fp &);

std::string  Mangling( cMatchNearestIm *);

class cMatchByName
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cMatchByName & anObj,cElXMLTree * aTree);


        std::string & KeyLog2Im();
        const std::string & KeyLog2Im()const ;
    private:
        std::string mKeyLog2Im;
};
cElXMLTree * ToXMLTree(const cMatchByName &);

void  BinaryDumpInFile(ELISE_fp &,const cMatchByName &);

void  BinaryUnDumpFromFile(cMatchByName &,ELISE_fp &);

std::string  Mangling( cMatchByName *);

class cAlgoMatch
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAlgoMatch & anObj,cElXMLTree * aTree);


        double & TolMatch();
        const double & TolMatch()const ;

        double & TolAmbig();
        const double & TolAmbig()const ;

        cTplValGesInit< cMatchNearestIm > & MatchNearestIm();
        const cTplValGesInit< cMatchNearestIm > & MatchNearestIm()const ;

        std::string & KeyLog2Im();
        const std::string & KeyLog2Im()const ;

        cTplValGesInit< cMatchByName > & MatchByName();
        const cTplValGesInit< cMatchByName > & MatchByName()const ;
    private:
        cTplValGesInit< cMatchNearestIm > mMatchNearestIm;
        cTplValGesInit< cMatchByName > mMatchByName;
};
cElXMLTree * ToXMLTree(const cAlgoMatch &);

void  BinaryDumpInFile(ELISE_fp &,const cAlgoMatch &);

void  BinaryUnDumpFromFile(cAlgoMatch &,ELISE_fp &);

std::string  Mangling( cAlgoMatch *);

class cTrAJ2_SectionMatch
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_SectionMatch & anObj,cElXMLTree * aTree);


        std::string & IdIm();
        const std::string & IdIm()const ;

        std::string & IdLog();
        const std::string & IdLog()const ;

        std::string & Im0();
        const std::string & Im0()const ;

        int & Log0();
        const int & Log0()const ;

        int & DeltaMinRech();
        const int & DeltaMinRech()const ;

        int & DeltaMaxRech();
        const int & DeltaMaxRech()const ;

        cTplValGesInit< bool > & Show();
        const cTplValGesInit< bool > & Show()const ;

        cTplValGesInit< bool > & ShowPerc();
        const cTplValGesInit< bool > & ShowPerc()const ;

        cTplValGesInit< cLearnByExample > & LearnByExample();
        const cTplValGesInit< cLearnByExample > & LearnByExample()const ;

        cTplValGesInit< double > & MaxEcart();
        const cTplValGesInit< double > & MaxEcart()const ;

        cTplValGesInit< cLearnByStatDiff > & LearnByStatDiff();
        const cTplValGesInit< cLearnByStatDiff > & LearnByStatDiff()const ;

        cTplValGesInit< cLearnOffset > & LearnOffset();
        const cTplValGesInit< cLearnOffset > & LearnOffset()const ;

        double & TolMatch();
        const double & TolMatch()const ;

        double & TolAmbig();
        const double & TolAmbig()const ;

        cTplValGesInit< cMatchNearestIm > & MatchNearestIm();
        const cTplValGesInit< cMatchNearestIm > & MatchNearestIm()const ;

        std::string & KeyLog2Im();
        const std::string & KeyLog2Im()const ;

        cTplValGesInit< cMatchByName > & MatchByName();
        const cTplValGesInit< cMatchByName > & MatchByName()const ;

        cAlgoMatch & AlgoMatch();
        const cAlgoMatch & AlgoMatch()const ;

        cTplValGesInit< cTrAJ2_ModeliseVitesse > & ModeliseVitesse();
        const cTplValGesInit< cTrAJ2_ModeliseVitesse > & ModeliseVitesse()const ;

        cTplValGesInit< cTrAJ2_GenerateOrient > & GenerateOrient();
        const cTplValGesInit< cTrAJ2_GenerateOrient > & GenerateOrient()const ;
    private:
        std::string mIdIm;
        std::string mIdLog;
        cTplValGesInit< cLearnOffset > mLearnOffset;
        cAlgoMatch mAlgoMatch;
        cTplValGesInit< cTrAJ2_ModeliseVitesse > mModeliseVitesse;
        cTplValGesInit< cTrAJ2_GenerateOrient > mGenerateOrient;
};
cElXMLTree * ToXMLTree(const cTrAJ2_SectionMatch &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_SectionMatch &);

void  BinaryUnDumpFromFile(cTrAJ2_SectionMatch &,ELISE_fp &);

std::string  Mangling( cTrAJ2_SectionMatch *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTraJ2_FilesInputi_Appuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTraJ2_FilesInputi_Appuis & anObj,cElXMLTree * aTree);


        std::string & KeySetOrPat();
        const std::string & KeySetOrPat()const ;

        cElRegex_Ptr & Autom();
        const cElRegex_Ptr & Autom()const ;

        bool & GetMesTer();
        const bool & GetMesTer()const ;

        bool & GetMesIm();
        const bool & GetMesIm()const ;

        int & KIdPt();
        const int & KIdPt()const ;
    private:
        std::string mKeySetOrPat;
        cElRegex_Ptr mAutom;
        bool mGetMesTer;
        bool mGetMesIm;
        int mKIdPt;
};
cElXMLTree * ToXMLTree(const cTraJ2_FilesInputi_Appuis &);

void  BinaryDumpInFile(ELISE_fp &,const cTraJ2_FilesInputi_Appuis &);

void  BinaryUnDumpFromFile(cTraJ2_FilesInputi_Appuis &,ELISE_fp &);

std::string  Mangling( cTraJ2_FilesInputi_Appuis *);

class cTrAJ2_ConvertionAppuis
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_ConvertionAppuis & anObj,cElXMLTree * aTree);


        std::string & Id();
        const std::string & Id()const ;

        std::list< cTraJ2_FilesInputi_Appuis > & TraJ2_FilesInputi_Appuis();
        const std::list< cTraJ2_FilesInputi_Appuis > & TraJ2_FilesInputi_Appuis()const ;

        cTplValGesInit< std::string > & OutMesTer();
        const cTplValGesInit< std::string > & OutMesTer()const ;

        cTplValGesInit< std::string > & OutMesIm();
        const cTplValGesInit< std::string > & OutMesIm()const ;

        cTplValGesInit< cElRegex_Ptr > & AutomComment();
        const cTplValGesInit< cElRegex_Ptr > & AutomComment()const ;

        std::vector< eUniteAngulaire > & UnitesCoord();
        const std::vector< eUniteAngulaire > & UnitesCoord()const ;

        cTplValGesInit< int > & KIncertPlani();
        const cTplValGesInit< int > & KIncertPlani()const ;

        cTplValGesInit< int > & KIncertAlti();
        const cTplValGesInit< int > & KIncertAlti()const ;

        cTplValGesInit< double > & ValIncertPlani();
        const cTplValGesInit< double > & ValIncertPlani()const ;

        cTplValGesInit< double > & ValIncertAlti();
        const cTplValGesInit< double > & ValIncertAlti()const ;

        int & KxTer();
        const int & KxTer()const ;

        int & KyTer();
        const int & KyTer()const ;

        int & KzTer();
        const int & KzTer()const ;

        int & KIIm();
        const int & KIIm()const ;

        int & KJIm();
        const int & KJIm()const ;

        int & KIdIm();
        const int & KIdIm()const ;

        cTplValGesInit< Pt2di > & OffsetIm();
        const cTplValGesInit< Pt2di > & OffsetIm()const ;

        std::string & KeyId2Im();
        const std::string & KeyId2Im()const ;

        cSystemeCoord & SystemeIn();
        const cSystemeCoord & SystemeIn()const ;

        cSystemeCoord & SystemeOut();
        const cSystemeCoord & SystemeOut()const ;
    private:
        std::string mId;
        std::list< cTraJ2_FilesInputi_Appuis > mTraJ2_FilesInputi_Appuis;
        cTplValGesInit< std::string > mOutMesTer;
        cTplValGesInit< std::string > mOutMesIm;
        cTplValGesInit< cElRegex_Ptr > mAutomComment;
        std::vector< eUniteAngulaire > mUnitesCoord;
        cTplValGesInit< int > mKIncertPlani;
        cTplValGesInit< int > mKIncertAlti;
        cTplValGesInit< double > mValIncertPlani;
        cTplValGesInit< double > mValIncertAlti;
        int mKxTer;
        int mKyTer;
        int mKzTer;
        int mKIIm;
        int mKJIm;
        int mKIdIm;
        cTplValGesInit< Pt2di > mOffsetIm;
        std::string mKeyId2Im;
        cSystemeCoord mSystemeIn;
        cSystemeCoord mSystemeOut;
};
cElXMLTree * ToXMLTree(const cTrAJ2_ConvertionAppuis &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_ConvertionAppuis &);

void  BinaryUnDumpFromFile(cTrAJ2_ConvertionAppuis &,ELISE_fp &);

std::string  Mangling( cTrAJ2_ConvertionAppuis *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTrAJ2_ExportProjImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTrAJ2_ExportProjImage & anObj,cElXMLTree * aTree);


        std::string & NameFileOut();
        const std::string & NameFileOut()const ;

        std::string & KeySetOrPatIm();
        const std::string & KeySetOrPatIm()const ;

        std::string & NameAppuis();
        const std::string & NameAppuis()const ;

        std::string & KeyAssocIm2Or();
        const std::string & KeyAssocIm2Or()const ;

        cTplValGesInit< std::string > & KeyGenerateTxt();
        const cTplValGesInit< std::string > & KeyGenerateTxt()const ;
    private:
        std::string mNameFileOut;
        std::string mKeySetOrPatIm;
        std::string mNameAppuis;
        std::string mKeyAssocIm2Or;
        cTplValGesInit< std::string > mKeyGenerateTxt;
};
cElXMLTree * ToXMLTree(const cTrAJ2_ExportProjImage &);

void  BinaryDumpInFile(ELISE_fp &,const cTrAJ2_ExportProjImage &);

void  BinaryUnDumpFromFile(cTrAJ2_ExportProjImage &,ELISE_fp &);

std::string  Mangling( cTrAJ2_ExportProjImage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParam_Traj_AJ
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParam_Traj_AJ & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        std::list< cTrAJ2_SectionImages > & TrAJ2_SectionImages();
        const std::list< cTrAJ2_SectionImages > & TrAJ2_SectionImages()const ;

        cTplValGesInit< cElRegex_Ptr > & TraceImages();
        const cTplValGesInit< cElRegex_Ptr > & TraceImages()const ;

        cTplValGesInit< cElRegex_Ptr > & TraceLogs();
        const cTplValGesInit< cElRegex_Ptr > & TraceLogs()const ;

        std::list< cTrAJ2_SectionLog > & TrAJ2_SectionLog();
        const std::list< cTrAJ2_SectionLog > & TrAJ2_SectionLog()const ;

        std::list< cTrAJ2_SectionMatch > & TrAJ2_SectionMatch();
        const std::list< cTrAJ2_SectionMatch > & TrAJ2_SectionMatch()const ;

        std::list< cTrAJ2_ConvertionAppuis > & TrAJ2_ConvertionAppuis();
        const std::list< cTrAJ2_ConvertionAppuis > & TrAJ2_ConvertionAppuis()const ;

        std::list< cTrAJ2_ExportProjImage > & TrAJ2_ExportProjImage();
        const std::list< cTrAJ2_ExportProjImage > & TrAJ2_ExportProjImage()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        std::list< cTrAJ2_SectionImages > mTrAJ2_SectionImages;
        cTplValGesInit< cElRegex_Ptr > mTraceImages;
        cTplValGesInit< cElRegex_Ptr > mTraceLogs;
        std::list< cTrAJ2_SectionLog > mTrAJ2_SectionLog;
        std::list< cTrAJ2_SectionMatch > mTrAJ2_SectionMatch;
        std::list< cTrAJ2_ConvertionAppuis > mTrAJ2_ConvertionAppuis;
        std::list< cTrAJ2_ExportProjImage > mTrAJ2_ExportProjImage;
};
cElXMLTree * ToXMLTree(const cParam_Traj_AJ &);

void  BinaryDumpInFile(ELISE_fp &,const cParam_Traj_AJ &);

void  BinaryUnDumpFromFile(cParam_Traj_AJ &,ELISE_fp &);

std::string  Mangling( cParam_Traj_AJ *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamGenereStr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamGenereStr & anObj,cElXMLTree * aTree);


        std::list< std::string > & KeySet();
        const std::list< std::string > & KeySet()const ;

        std::list< std::string > & KeyString();
        const std::list< std::string > & KeyString()const ;
    private:
        std::list< std::string > mKeySet;
        std::list< std::string > mKeyString;
};
cElXMLTree * ToXMLTree(const cParamGenereStr &);

void  BinaryDumpInFile(ELISE_fp &,const cParamGenereStr &);

void  BinaryUnDumpFromFile(cParamGenereStr &,ELISE_fp &);

std::string  Mangling( cParamGenereStr *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamGenereStrVois
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamGenereStrVois & anObj,cElXMLTree * aTree);


        std::list< std::string > & KeyRel();
        const std::list< std::string > & KeyRel()const ;

        std::list< std::string > & KeyString();
        const std::list< std::string > & KeyString()const ;

        std::list< std::string > & KeySet();
        const std::list< std::string > & KeySet()const ;

        cTplValGesInit< bool > & UseIt();
        const cTplValGesInit< bool > & UseIt()const ;
    private:
        std::list< std::string > mKeyRel;
        std::list< std::string > mKeyString;
        std::list< std::string > mKeySet;
        cTplValGesInit< bool > mUseIt;
};
cElXMLTree * ToXMLTree(const cParamGenereStrVois &);

void  BinaryDumpInFile(ELISE_fp &,const cParamGenereStrVois &);

void  BinaryUnDumpFromFile(cParamGenereStrVois &,ELISE_fp &);

std::string  Mangling( cParamGenereStrVois *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamFiltreDetecRegulProf
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamFiltreDetecRegulProf & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & SzCC();
        const cTplValGesInit< int > & SzCC()const ;

        cTplValGesInit< double > & PondZ();
        const cTplValGesInit< double > & PondZ()const ;

        cTplValGesInit< double > & Pente();
        const cTplValGesInit< double > & Pente()const ;

        cTplValGesInit< double > & SeuilReg();
        const cTplValGesInit< double > & SeuilReg()const ;

        cTplValGesInit< bool > & V4();
        const cTplValGesInit< bool > & V4()const ;

        cTplValGesInit< int > & NbCCInit();
        const cTplValGesInit< int > & NbCCInit()const ;

        cTplValGesInit< std::string > & NameTest();
        const cTplValGesInit< std::string > & NameTest()const ;
    private:
        cTplValGesInit< int > mSzCC;
        cTplValGesInit< double > mPondZ;
        cTplValGesInit< double > mPente;
        cTplValGesInit< double > mSeuilReg;
        cTplValGesInit< bool > mV4;
        cTplValGesInit< int > mNbCCInit;
        cTplValGesInit< std::string > mNameTest;
};
cElXMLTree * ToXMLTree(const cParamFiltreDetecRegulProf &);

void  BinaryDumpInFile(ELISE_fp &,const cParamFiltreDetecRegulProf &);

void  BinaryUnDumpFromFile(cParamFiltreDetecRegulProf &,ELISE_fp &);

std::string  Mangling( cParamFiltreDetecRegulProf *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionName
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionName & anObj,cElXMLTree * aTree);


        std::string & KeyNuage();
        const std::string & KeyNuage()const ;

        std::string & KeyResult();
        const std::string & KeyResult()const ;

        cTplValGesInit< bool > & KeyResultIsLoc();
        const cTplValGesInit< bool > & KeyResultIsLoc()const ;

        cTplValGesInit< std::string > & ModeleNuageResult();
        const cTplValGesInit< std::string > & ModeleNuageResult()const ;

        cTplValGesInit< std::string > & KeyNuage2Im();
        const cTplValGesInit< std::string > & KeyNuage2Im()const ;
    private:
        std::string mKeyNuage;
        std::string mKeyResult;
        cTplValGesInit< bool > mKeyResultIsLoc;
        cTplValGesInit< std::string > mModeleNuageResult;
        cTplValGesInit< std::string > mKeyNuage2Im;
};
cElXMLTree * ToXMLTree(const cSectionName &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionName &);

void  BinaryUnDumpFromFile(cSectionName &,ELISE_fp &);

std::string  Mangling( cSectionName *);

/******************************************************/
/******************************************************/
/******************************************************/
class cScoreMM1P
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cScoreMM1P & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & MakeFileResult();
        const cTplValGesInit< bool > & MakeFileResult()const ;

        cTplValGesInit< double > & PdsAR();
        const cTplValGesInit< double > & PdsAR()const ;

        cTplValGesInit< double > & PdsDistor();
        const cTplValGesInit< double > & PdsDistor()const ;

        cTplValGesInit< double > & AmplImDistor();
        const cTplValGesInit< double > & AmplImDistor()const ;

        cTplValGesInit< double > & SeuilDist();
        const cTplValGesInit< double > & SeuilDist()const ;

        cTplValGesInit< double > & PdsDistBord();
        const cTplValGesInit< double > & PdsDistBord()const ;

        cTplValGesInit< double > & SeuilDisBord();
        const cTplValGesInit< double > & SeuilDisBord()const ;
    private:
        cTplValGesInit< bool > mMakeFileResult;
        cTplValGesInit< double > mPdsAR;
        cTplValGesInit< double > mPdsDistor;
        cTplValGesInit< double > mAmplImDistor;
        cTplValGesInit< double > mSeuilDist;
        cTplValGesInit< double > mPdsDistBord;
        cTplValGesInit< double > mSeuilDisBord;
};
cElXMLTree * ToXMLTree(const cScoreMM1P &);

void  BinaryDumpInFile(ELISE_fp &,const cScoreMM1P &);

void  BinaryUnDumpFromFile(cScoreMM1P &,ELISE_fp &);

std::string  Mangling( cScoreMM1P *);

class cSectionScoreQualite
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionScoreQualite & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & MakeFileResult();
        const cTplValGesInit< bool > & MakeFileResult()const ;

        cTplValGesInit< double > & PdsAR();
        const cTplValGesInit< double > & PdsAR()const ;

        cTplValGesInit< double > & PdsDistor();
        const cTplValGesInit< double > & PdsDistor()const ;

        cTplValGesInit< double > & AmplImDistor();
        const cTplValGesInit< double > & AmplImDistor()const ;

        cTplValGesInit< double > & SeuilDist();
        const cTplValGesInit< double > & SeuilDist()const ;

        cTplValGesInit< double > & PdsDistBord();
        const cTplValGesInit< double > & PdsDistBord()const ;

        cTplValGesInit< double > & SeuilDisBord();
        const cTplValGesInit< double > & SeuilDisBord()const ;

        cTplValGesInit< cScoreMM1P > & ScoreMM1P();
        const cTplValGesInit< cScoreMM1P > & ScoreMM1P()const ;
    private:
        cTplValGesInit< cScoreMM1P > mScoreMM1P;
};
cElXMLTree * ToXMLTree(const cSectionScoreQualite &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionScoreQualite &);

void  BinaryUnDumpFromFile(cSectionScoreQualite &,ELISE_fp &);

std::string  Mangling( cSectionScoreQualite *);

/******************************************************/
/******************************************************/
/******************************************************/
class cFMNT_GesNoVal
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFMNT_GesNoVal & anObj,cElXMLTree * aTree);


        double & PenteMax();
        const double & PenteMax()const ;

        double & CostNoVal();
        const double & CostNoVal()const ;

        double & Trans();
        const double & Trans()const ;
    private:
        double mPenteMax;
        double mCostNoVal;
        double mTrans;
};
cElXMLTree * ToXMLTree(const cFMNT_GesNoVal &);

void  BinaryDumpInFile(ELISE_fp &,const cFMNT_GesNoVal &);

void  BinaryUnDumpFromFile(cFMNT_GesNoVal &,ELISE_fp &);

std::string  Mangling( cFMNT_GesNoVal *);

class cFMNT_ProgDyn
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFMNT_ProgDyn & anObj,cElXMLTree * aTree);


        double & Regul();
        const double & Regul()const ;

        double & Sigma0();
        const double & Sigma0()const ;

        int & NbDir();
        const int & NbDir()const ;

        double & PenteMax();
        const double & PenteMax()const ;

        double & CostNoVal();
        const double & CostNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;
    private:
        double mRegul;
        double mSigma0;
        int mNbDir;
        cTplValGesInit< cFMNT_GesNoVal > mFMNT_GesNoVal;
};
cElXMLTree * ToXMLTree(const cFMNT_ProgDyn &);

void  BinaryDumpInFile(ELISE_fp &,const cFMNT_ProgDyn &);

void  BinaryUnDumpFromFile(cFMNT_ProgDyn &,ELISE_fp &);

std::string  Mangling( cFMNT_ProgDyn *);

class cSpecAlgoFMNT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSpecAlgoFMNT & anObj,cElXMLTree * aTree);


        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        double & SeuilMaxLoc();
        const double & SeuilMaxLoc()const ;

        double & SeuilCptOk();
        const double & SeuilCptOk()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

        cTplValGesInit< int > & NBMaxMaxLoc();
        const cTplValGesInit< int > & NBMaxMaxLoc()const ;

        cTplValGesInit< bool > & QuickExp();
        const cTplValGesInit< bool > & QuickExp()const ;

        double & Regul();
        const double & Regul()const ;

        double & Sigma0();
        const double & Sigma0()const ;

        int & NbDir();
        const int & NbDir()const ;

        double & PenteMax();
        const double & PenteMax()const ;

        double & CostNoVal();
        const double & CostNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;

        cTplValGesInit< cParamFiltreDetecRegulProf > & ParamRegProf();
        const cTplValGesInit< cParamFiltreDetecRegulProf > & ParamRegProf()const ;
    private:
        double mSigmaPds;
        cTplValGesInit< double > mSigmaZ;
        double mSeuilMaxLoc;
        double mSeuilCptOk;
        cTplValGesInit< double > mMaxDif;
        cTplValGesInit< int > mNBMaxMaxLoc;
        cTplValGesInit< bool > mQuickExp;
        cTplValGesInit< cFMNT_ProgDyn > mFMNT_ProgDyn;
        cTplValGesInit< cParamFiltreDetecRegulProf > mParamRegProf;
};
cElXMLTree * ToXMLTree(const cSpecAlgoFMNT &);

void  BinaryDumpInFile(ELISE_fp &,const cSpecAlgoFMNT &);

void  BinaryUnDumpFromFile(cSpecAlgoFMNT &,ELISE_fp &);

std::string  Mangling( cSpecAlgoFMNT *);

class cParamAlgoFusionMNT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamAlgoFusionMNT & anObj,cElXMLTree * aTree);


        double & FMNTSeuilCorrel();
        const double & FMNTSeuilCorrel()const ;

        double & FMNTGammaCorrel();
        const double & FMNTGammaCorrel()const ;

        cTplValGesInit< std::string > & KeyPdsNuage();
        const cTplValGesInit< std::string > & KeyPdsNuage()const ;

        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        double & SeuilMaxLoc();
        const double & SeuilMaxLoc()const ;

        double & SeuilCptOk();
        const double & SeuilCptOk()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

        cTplValGesInit< int > & NBMaxMaxLoc();
        const cTplValGesInit< int > & NBMaxMaxLoc()const ;

        cTplValGesInit< bool > & QuickExp();
        const cTplValGesInit< bool > & QuickExp()const ;

        double & Regul();
        const double & Regul()const ;

        double & Sigma0();
        const double & Sigma0()const ;

        int & NbDir();
        const int & NbDir()const ;

        double & PenteMax();
        const double & PenteMax()const ;

        double & CostNoVal();
        const double & CostNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;

        cTplValGesInit< cParamFiltreDetecRegulProf > & ParamRegProf();
        const cTplValGesInit< cParamFiltreDetecRegulProf > & ParamRegProf()const ;

        cSpecAlgoFMNT & SpecAlgoFMNT();
        const cSpecAlgoFMNT & SpecAlgoFMNT()const ;
    private:
        double mFMNTSeuilCorrel;
        double mFMNTGammaCorrel;
        cTplValGesInit< std::string > mKeyPdsNuage;
        cSpecAlgoFMNT mSpecAlgoFMNT;
};
cElXMLTree * ToXMLTree(const cParamAlgoFusionMNT &);

void  BinaryDumpInFile(ELISE_fp &,const cParamAlgoFusionMNT &);

void  BinaryUnDumpFromFile(cParamAlgoFusionMNT &,ELISE_fp &);

std::string  Mangling( cParamAlgoFusionMNT *);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionGestionChantier
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionGestionChantier & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & SzDalles();
        const cTplValGesInit< int > & SzDalles()const ;

        cTplValGesInit< int > & RecouvrtDalles();
        const cTplValGesInit< int > & RecouvrtDalles()const ;

        cTplValGesInit< std::string > & ParalMkF();
        const cTplValGesInit< std::string > & ParalMkF()const ;

        cTplValGesInit< bool > & ByProcess();
        const cTplValGesInit< bool > & ByProcess()const ;

        cTplValGesInit< bool > & InterneCalledByProcess();
        const cTplValGesInit< bool > & InterneCalledByProcess()const ;

        cTplValGesInit< std::string > & InterneSingleImage();
        const cTplValGesInit< std::string > & InterneSingleImage()const ;

        cTplValGesInit< int > & InterneSingleBox();
        const cTplValGesInit< int > & InterneSingleBox()const ;

        cTplValGesInit< std::string > & WorkDirPFM();
        const cTplValGesInit< std::string > & WorkDirPFM()const ;

        cTplValGesInit< Box2di > & BoxTest();
        const cTplValGesInit< Box2di > & BoxTest()const ;
    private:
        cTplValGesInit< int > mSzDalles;
        cTplValGesInit< int > mRecouvrtDalles;
        cTplValGesInit< std::string > mParalMkF;
        cTplValGesInit< bool > mByProcess;
        cTplValGesInit< bool > mInterneCalledByProcess;
        cTplValGesInit< std::string > mInterneSingleImage;
        cTplValGesInit< int > mInterneSingleBox;
        cTplValGesInit< std::string > mWorkDirPFM;
        cTplValGesInit< Box2di > mBoxTest;
};
cElXMLTree * ToXMLTree(const cSectionGestionChantier &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionGestionChantier &);

void  BinaryUnDumpFromFile(cSectionGestionChantier &,ELISE_fp &);

std::string  Mangling( cSectionGestionChantier *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamFusionMNT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamFusionMNT & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        std::string & KeyNuage();
        const std::string & KeyNuage()const ;

        std::string & KeyResult();
        const std::string & KeyResult()const ;

        cTplValGesInit< bool > & KeyResultIsLoc();
        const cTplValGesInit< bool > & KeyResultIsLoc()const ;

        cTplValGesInit< std::string > & ModeleNuageResult();
        const cTplValGesInit< std::string > & ModeleNuageResult()const ;

        cTplValGesInit< std::string > & KeyNuage2Im();
        const cTplValGesInit< std::string > & KeyNuage2Im()const ;

        cSectionName & SectionName();
        const cSectionName & SectionName()const ;

        cTplValGesInit< bool > & MakeFileResult();
        const cTplValGesInit< bool > & MakeFileResult()const ;

        cTplValGesInit< double > & PdsAR();
        const cTplValGesInit< double > & PdsAR()const ;

        cTplValGesInit< double > & PdsDistor();
        const cTplValGesInit< double > & PdsDistor()const ;

        cTplValGesInit< double > & AmplImDistor();
        const cTplValGesInit< double > & AmplImDistor()const ;

        cTplValGesInit< double > & SeuilDist();
        const cTplValGesInit< double > & SeuilDist()const ;

        cTplValGesInit< double > & PdsDistBord();
        const cTplValGesInit< double > & PdsDistBord()const ;

        cTplValGesInit< double > & SeuilDisBord();
        const cTplValGesInit< double > & SeuilDisBord()const ;

        cTplValGesInit< cScoreMM1P > & ScoreMM1P();
        const cTplValGesInit< cScoreMM1P > & ScoreMM1P()const ;

        cTplValGesInit< cSectionScoreQualite > & SectionScoreQualite();
        const cTplValGesInit< cSectionScoreQualite > & SectionScoreQualite()const ;

        double & FMNTSeuilCorrel();
        const double & FMNTSeuilCorrel()const ;

        double & FMNTGammaCorrel();
        const double & FMNTGammaCorrel()const ;

        cTplValGesInit< std::string > & KeyPdsNuage();
        const cTplValGesInit< std::string > & KeyPdsNuage()const ;

        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        double & SeuilMaxLoc();
        const double & SeuilMaxLoc()const ;

        double & SeuilCptOk();
        const double & SeuilCptOk()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

        cTplValGesInit< int > & NBMaxMaxLoc();
        const cTplValGesInit< int > & NBMaxMaxLoc()const ;

        cTplValGesInit< bool > & QuickExp();
        const cTplValGesInit< bool > & QuickExp()const ;

        double & Regul();
        const double & Regul()const ;

        double & Sigma0();
        const double & Sigma0()const ;

        int & NbDir();
        const int & NbDir()const ;

        double & PenteMax();
        const double & PenteMax()const ;

        double & CostNoVal();
        const double & CostNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;

        cTplValGesInit< cParamFiltreDetecRegulProf > & ParamRegProf();
        const cTplValGesInit< cParamFiltreDetecRegulProf > & ParamRegProf()const ;

        cSpecAlgoFMNT & SpecAlgoFMNT();
        const cSpecAlgoFMNT & SpecAlgoFMNT()const ;

        cParamAlgoFusionMNT & ParamAlgoFusionMNT();
        const cParamAlgoFusionMNT & ParamAlgoFusionMNT()const ;

        cParamGenereStr & GenereRes();
        const cParamGenereStr & GenereRes()const ;

        cParamGenereStrVois & GenereInput();
        const cParamGenereStrVois & GenereInput()const ;

        cTplValGesInit< int > & SzDalles();
        const cTplValGesInit< int > & SzDalles()const ;

        cTplValGesInit< int > & RecouvrtDalles();
        const cTplValGesInit< int > & RecouvrtDalles()const ;

        cTplValGesInit< std::string > & ParalMkF();
        const cTplValGesInit< std::string > & ParalMkF()const ;

        cTplValGesInit< bool > & ByProcess();
        const cTplValGesInit< bool > & ByProcess()const ;

        cTplValGesInit< bool > & InterneCalledByProcess();
        const cTplValGesInit< bool > & InterneCalledByProcess()const ;

        cTplValGesInit< std::string > & InterneSingleImage();
        const cTplValGesInit< std::string > & InterneSingleImage()const ;

        cTplValGesInit< int > & InterneSingleBox();
        const cTplValGesInit< int > & InterneSingleBox()const ;

        cTplValGesInit< std::string > & WorkDirPFM();
        const cTplValGesInit< std::string > & WorkDirPFM()const ;

        cTplValGesInit< Box2di > & BoxTest();
        const cTplValGesInit< Box2di > & BoxTest()const ;

        cSectionGestionChantier & SectionGestionChantier();
        const cSectionGestionChantier & SectionGestionChantier()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cSectionName mSectionName;
        cTplValGesInit< cSectionScoreQualite > mSectionScoreQualite;
        cParamAlgoFusionMNT mParamAlgoFusionMNT;
        cParamGenereStr mGenereRes;
        cParamGenereStrVois mGenereInput;
        cSectionGestionChantier mSectionGestionChantier;
};
cElXMLTree * ToXMLTree(const cParamFusionMNT &);

void  BinaryDumpInFile(ELISE_fp &,const cParamFusionMNT &);

void  BinaryUnDumpFromFile(cParamFusionMNT &,ELISE_fp &);

std::string  Mangling( cParamFusionMNT *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPFNMiseAuPoint
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPFNMiseAuPoint & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt2di > & SzVisu();
        const cTplValGesInit< Pt2di > & SzVisu()const ;

        cTplValGesInit< bool > & TestImageDif();
        const cTplValGesInit< bool > & TestImageDif()const ;

        cTplValGesInit< bool > & VisuGrad();
        const cTplValGesInit< bool > & VisuGrad()const ;

        cTplValGesInit< bool > & VisuLowPts();
        const cTplValGesInit< bool > & VisuLowPts()const ;

        cTplValGesInit< bool > & VisuImageCoh();
        const cTplValGesInit< bool > & VisuImageCoh()const ;

        cTplValGesInit< bool > & VisuSelect();
        const cTplValGesInit< bool > & VisuSelect()const ;

        cTplValGesInit< bool > & VisuEnv();
        const cTplValGesInit< bool > & VisuEnv()const ;

        cTplValGesInit< bool > & VisuElim();
        const cTplValGesInit< bool > & VisuElim()const ;

        cTplValGesInit< std::string > & ImageMiseAuPoint();
        const cTplValGesInit< std::string > & ImageMiseAuPoint()const ;
    private:
        cTplValGesInit< Pt2di > mSzVisu;
        cTplValGesInit< bool > mTestImageDif;
        cTplValGesInit< bool > mVisuGrad;
        cTplValGesInit< bool > mVisuLowPts;
        cTplValGesInit< bool > mVisuImageCoh;
        cTplValGesInit< bool > mVisuSelect;
        cTplValGesInit< bool > mVisuEnv;
        cTplValGesInit< bool > mVisuElim;
        cTplValGesInit< std::string > mImageMiseAuPoint;
};
cElXMLTree * ToXMLTree(const cPFNMiseAuPoint &);

void  BinaryDumpInFile(ELISE_fp &,const cPFNMiseAuPoint &);

void  BinaryUnDumpFromFile(cPFNMiseAuPoint &,ELISE_fp &);

std::string  Mangling( cPFNMiseAuPoint *);

/******************************************************/
/******************************************************/
/******************************************************/
class cGrapheRecouvrt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGrapheRecouvrt & anObj,cElXMLTree * aTree);


        double & TauxRecMin();
        const double & TauxRecMin()const ;

        cTplValGesInit< std::string > & ExtHom();
        const cTplValGesInit< std::string > & ExtHom()const ;

        cTplValGesInit< int > & MinSzFilHom();
        const cTplValGesInit< int > & MinSzFilHom()const ;

        cTplValGesInit< double > & RecSeuilDistProf();
        const cTplValGesInit< double > & RecSeuilDistProf()const ;

        int & NbPtsLowResume();
        const int & NbPtsLowResume()const ;

        cTplValGesInit< double > & CostPerImISOM();
        const cTplValGesInit< double > & CostPerImISOM()const ;
    private:
        double mTauxRecMin;
        cTplValGesInit< std::string > mExtHom;
        cTplValGesInit< int > mMinSzFilHom;
        cTplValGesInit< double > mRecSeuilDistProf;
        int mNbPtsLowResume;
        cTplValGesInit< double > mCostPerImISOM;
};
cElXMLTree * ToXMLTree(const cGrapheRecouvrt &);

void  BinaryDumpInFile(ELISE_fp &,const cGrapheRecouvrt &);

void  BinaryUnDumpFromFile(cGrapheRecouvrt &,ELISE_fp &);

std::string  Mangling( cGrapheRecouvrt *);

/******************************************************/
/******************************************************/
/******************************************************/
class cImageVariations
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImageVariations & anObj,cElXMLTree * aTree);


        bool & V4Vois();
        const bool & V4Vois()const ;

        int & DistVois();
        const int & DistVois()const ;

        double & DynAngul();
        const double & DynAngul()const ;

        double & SeuilStrictVarIma();
        const double & SeuilStrictVarIma()const ;

        cTplValGesInit< double > & PenteRefutInitInPixel();
        const cTplValGesInit< double > & PenteRefutInitInPixel()const ;

        cTplValGesInit< bool > & ComputeIncid();
        const cTplValGesInit< bool > & ComputeIncid()const ;

        cTplValGesInit< int > & DilateBord();
        const cTplValGesInit< int > & DilateBord()const ;
    private:
        bool mV4Vois;
        int mDistVois;
        double mDynAngul;
        double mSeuilStrictVarIma;
        cTplValGesInit< double > mPenteRefutInitInPixel;
        cTplValGesInit< bool > mComputeIncid;
        cTplValGesInit< int > mDilateBord;
};
cElXMLTree * ToXMLTree(const cImageVariations &);

void  BinaryDumpInFile(ELISE_fp &,const cImageVariations &);

void  BinaryUnDumpFromFile(cImageVariations &,ELISE_fp &);

std::string  Mangling( cImageVariations *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPFM_Selection
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPFM_Selection & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & ElimDirectInterior();
        const cTplValGesInit< double > & ElimDirectInterior()const ;

        cTplValGesInit< double > & LowRatioSelectIm();
        const cTplValGesInit< double > & LowRatioSelectIm()const ;

        cTplValGesInit< double > & HighRatioSelectIm();
        const cTplValGesInit< double > & HighRatioSelectIm()const ;
    private:
        cTplValGesInit< double > mElimDirectInterior;
        cTplValGesInit< double > mLowRatioSelectIm;
        cTplValGesInit< double > mHighRatioSelectIm;
};
cElXMLTree * ToXMLTree(const cPFM_Selection &);

void  BinaryDumpInFile(ELISE_fp &,const cPFM_Selection &);

void  BinaryUnDumpFromFile(cPFM_Selection &,ELISE_fp &);

std::string  Mangling( cPFM_Selection *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamFusionNuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamFusionNuage & anObj,cElXMLTree * aTree);


        eTypeMMByP & ModeMerge();
        const eTypeMMByP & ModeMerge()const ;

        cTplValGesInit< Pt2di > & SzVisu();
        const cTplValGesInit< Pt2di > & SzVisu()const ;

        cTplValGesInit< bool > & TestImageDif();
        const cTplValGesInit< bool > & TestImageDif()const ;

        cTplValGesInit< bool > & VisuGrad();
        const cTplValGesInit< bool > & VisuGrad()const ;

        cTplValGesInit< bool > & VisuLowPts();
        const cTplValGesInit< bool > & VisuLowPts()const ;

        cTplValGesInit< bool > & VisuImageCoh();
        const cTplValGesInit< bool > & VisuImageCoh()const ;

        cTplValGesInit< bool > & VisuSelect();
        const cTplValGesInit< bool > & VisuSelect()const ;

        cTplValGesInit< bool > & VisuEnv();
        const cTplValGesInit< bool > & VisuEnv()const ;

        cTplValGesInit< bool > & VisuElim();
        const cTplValGesInit< bool > & VisuElim()const ;

        cTplValGesInit< std::string > & ImageMiseAuPoint();
        const cTplValGesInit< std::string > & ImageMiseAuPoint()const ;

        cPFNMiseAuPoint & PFNMiseAuPoint();
        const cPFNMiseAuPoint & PFNMiseAuPoint()const ;

        double & TauxRecMin();
        const double & TauxRecMin()const ;

        cTplValGesInit< std::string > & ExtHom();
        const cTplValGesInit< std::string > & ExtHom()const ;

        cTplValGesInit< int > & MinSzFilHom();
        const cTplValGesInit< int > & MinSzFilHom()const ;

        cTplValGesInit< double > & RecSeuilDistProf();
        const cTplValGesInit< double > & RecSeuilDistProf()const ;

        int & NbPtsLowResume();
        const int & NbPtsLowResume()const ;

        cTplValGesInit< double > & CostPerImISOM();
        const cTplValGesInit< double > & CostPerImISOM()const ;

        cGrapheRecouvrt & GrapheRecouvrt();
        const cGrapheRecouvrt & GrapheRecouvrt()const ;

        bool & V4Vois();
        const bool & V4Vois()const ;

        int & DistVois();
        const int & DistVois()const ;

        double & DynAngul();
        const double & DynAngul()const ;

        double & SeuilStrictVarIma();
        const double & SeuilStrictVarIma()const ;

        cTplValGesInit< double > & PenteRefutInitInPixel();
        const cTplValGesInit< double > & PenteRefutInitInPixel()const ;

        cTplValGesInit< bool > & ComputeIncid();
        const cTplValGesInit< bool > & ComputeIncid()const ;

        cTplValGesInit< int > & DilateBord();
        const cTplValGesInit< int > & DilateBord()const ;

        cImageVariations & ImageVariations();
        const cImageVariations & ImageVariations()const ;

        cTplValGesInit< double > & ElimDirectInterior();
        const cTplValGesInit< double > & ElimDirectInterior()const ;

        cTplValGesInit< double > & LowRatioSelectIm();
        const cTplValGesInit< double > & LowRatioSelectIm()const ;

        cTplValGesInit< double > & HighRatioSelectIm();
        const cTplValGesInit< double > & HighRatioSelectIm()const ;

        cPFM_Selection & PFM_Selection();
        const cPFM_Selection & PFM_Selection()const ;
    private:
        eTypeMMByP mModeMerge;
        cPFNMiseAuPoint mPFNMiseAuPoint;
        cGrapheRecouvrt mGrapheRecouvrt;
        cImageVariations mImageVariations;
        cPFM_Selection mPFM_Selection;
};
cElXMLTree * ToXMLTree(const cParamFusionNuage &);

void  BinaryDumpInFile(ELISE_fp &,const cParamFusionNuage &);

void  BinaryUnDumpFromFile(cParamFusionNuage &,ELISE_fp &);

std::string  Mangling( cParamFusionNuage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCWWSIVois
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCWWSIVois & anObj,cElXMLTree * aTree);


        std::string & NameVois();
        const std::string & NameVois()const ;
    private:
        std::string mNameVois;
};
cElXMLTree * ToXMLTree(const cCWWSIVois &);

void  BinaryDumpInFile(ELISE_fp &,const cCWWSIVois &);

void  BinaryUnDumpFromFile(cCWWSIVois &,ELISE_fp &);

std::string  Mangling( cCWWSIVois *);

/******************************************************/
/******************************************************/
/******************************************************/
class cCWWSImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCWWSImage & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        std::list< cCWWSIVois > & CWWSIVois();
        const std::list< cCWWSIVois > & CWWSIVois()const ;
    private:
        std::string mNameIm;
        std::list< cCWWSIVois > mCWWSIVois;
};
cElXMLTree * ToXMLTree(const cCWWSImage &);

void  BinaryDumpInFile(ELISE_fp &,const cCWWSImage &);

void  BinaryUnDumpFromFile(cCWWSImage &,ELISE_fp &);

std::string  Mangling( cCWWSImage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cChantierAppliWithSetImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cChantierAppliWithSetImage & anObj,cElXMLTree * aTree);


        std::list< cCWWSImage > & Images();
        const std::list< cCWWSImage > & Images()const ;
    private:
        std::list< cCWWSImage > mImages;
};
cElXMLTree * ToXMLTree(const cChantierAppliWithSetImage &);

void  BinaryDumpInFile(ELISE_fp &,const cChantierAppliWithSetImage &);

void  BinaryUnDumpFromFile(cChantierAppliWithSetImage &,ELISE_fp &);

std::string  Mangling( cChantierAppliWithSetImage *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneZonzATB
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneZonzATB & anObj,cElXMLTree * aTree);


        Box2di & BoxGlob();
        const Box2di & BoxGlob()const ;

        Box2di & BoxMasq();
        const Box2di & BoxMasq()const ;

        Pt2di & GermGlob();
        const Pt2di & GermGlob()const ;

        Pt2di & GermMasq();
        const Pt2di & GermMasq()const ;

        int & NbGlob();
        const int & NbGlob()const ;

        int & NbMasq();
        const int & NbMasq()const ;

        int & Num();
        const int & Num()const ;

        bool & Valide();
        const bool & Valide()const ;
    private:
        Box2di mBoxGlob;
        Box2di mBoxMasq;
        Pt2di mGermGlob;
        Pt2di mGermMasq;
        int mNbGlob;
        int mNbMasq;
        int mNum;
        bool mValide;
};
cElXMLTree * ToXMLTree(const cOneZonzATB &);

void  BinaryDumpInFile(ELISE_fp &,const cOneZonzATB &);

void  BinaryUnDumpFromFile(cOneZonzATB &,ELISE_fp &);

std::string  Mangling( cOneZonzATB *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAnaTopoBascule
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAnaTopoBascule & anObj,cElXMLTree * aTree);


        std::list< cOneZonzATB > & OneZonzATB();
        const std::list< cOneZonzATB > & OneZonzATB()const ;
    private:
        std::list< cOneZonzATB > mOneZonzATB;
};
cElXMLTree * ToXMLTree(const cAnaTopoBascule &);

void  BinaryDumpInFile(ELISE_fp &,const cAnaTopoBascule &);

void  BinaryUnDumpFromFile(cAnaTopoBascule &,ELISE_fp &);

std::string  Mangling( cAnaTopoBascule *);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneZonXmlAMTB
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneZonXmlAMTB & anObj,cElXMLTree * aTree);


        std::string & NameXml();
        const std::string & NameXml()const ;
    private:
        std::string mNameXml;
};
cElXMLTree * ToXMLTree(const cOneZonXmlAMTB &);

void  BinaryDumpInFile(ELISE_fp &,const cOneZonXmlAMTB &);

void  BinaryUnDumpFromFile(cOneZonXmlAMTB &,ELISE_fp &);

std::string  Mangling( cOneZonXmlAMTB *);

/******************************************************/
/******************************************************/
/******************************************************/
class cAnaTopoXmlBascule
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cAnaTopoXmlBascule & anObj,cElXMLTree * aTree);


        bool & ResFromAnaTopo();
        const bool & ResFromAnaTopo()const ;

        std::list< cOneZonXmlAMTB > & OneZonXmlAMTB();
        const std::list< cOneZonXmlAMTB > & OneZonXmlAMTB()const ;
    private:
        bool mResFromAnaTopo;
        std::list< cOneZonXmlAMTB > mOneZonXmlAMTB;
};
cElXMLTree * ToXMLTree(const cAnaTopoXmlBascule &);

void  BinaryDumpInFile(ELISE_fp &,const cAnaTopoXmlBascule &);

void  BinaryUnDumpFromFile(cAnaTopoXmlBascule &,ELISE_fp &);

std::string  Mangling( cAnaTopoXmlBascule *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamFiltreDepthByPrgDyn
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamFiltreDepthByPrgDyn & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & CostNonAff();
        const cTplValGesInit< double > & CostNonAff()const ;

        cTplValGesInit< double > & CostTrans();
        const cTplValGesInit< double > & CostTrans()const ;

        cTplValGesInit< double > & CostRegul();
        const cTplValGesInit< double > & CostRegul()const ;

        double & StepZ();
        const double & StepZ()const ;

        cTplValGesInit< double > & DzMax();
        const cTplValGesInit< double > & DzMax()const ;

        cTplValGesInit< int > & NbDir();
        const cTplValGesInit< int > & NbDir()const ;
    private:
        cTplValGesInit< double > mCostNonAff;
        cTplValGesInit< double > mCostTrans;
        cTplValGesInit< double > mCostRegul;
        double mStepZ;
        cTplValGesInit< double > mDzMax;
        cTplValGesInit< int > mNbDir;
};
cElXMLTree * ToXMLTree(const cParamFiltreDepthByPrgDyn &);

void  BinaryDumpInFile(ELISE_fp &,const cParamFiltreDepthByPrgDyn &);

void  BinaryUnDumpFromFile(cParamFiltreDepthByPrgDyn &,ELISE_fp &);

std::string  Mangling( cParamFiltreDepthByPrgDyn *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlAffinR2ToR
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlAffinR2ToR & anObj,cElXMLTree * aTree);


        double & CoeffX();
        const double & CoeffX()const ;

        double & CoeffY();
        const double & CoeffY()const ;

        double & Coeff1();
        const double & Coeff1()const ;
    private:
        double mCoeffX;
        double mCoeffY;
        double mCoeff1;
};
cElXMLTree * ToXMLTree(const cXmlAffinR2ToR &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlAffinR2ToR &);

void  BinaryUnDumpFromFile(cXmlAffinR2ToR &,ELISE_fp &);

std::string  Mangling( cXmlAffinR2ToR *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlHomogr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlHomogr & anObj,cElXMLTree * aTree);


        cXmlAffinR2ToR & X();
        const cXmlAffinR2ToR & X()const ;

        cXmlAffinR2ToR & Y();
        const cXmlAffinR2ToR & Y()const ;

        cXmlAffinR2ToR & Z();
        const cXmlAffinR2ToR & Z()const ;
    private:
        cXmlAffinR2ToR mX;
        cXmlAffinR2ToR mY;
        cXmlAffinR2ToR mZ;
};
cElXMLTree * ToXMLTree(const cXmlHomogr &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlHomogr &);

void  BinaryUnDumpFromFile(cXmlHomogr &,ELISE_fp &);

std::string  Mangling( cXmlHomogr *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXmlRHHResLnk
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXmlRHHResLnk & anObj,cElXMLTree * aTree);


        cXmlHomogr & Hom12();
        const cXmlHomogr & Hom12()const ;

        bool & Ok();
        const bool & Ok()const ;

        double & Qual();
        const double & Qual()const ;

        int & NbPts();
        const int & NbPts()const ;

        std::vector< Pt3dr > & EchRepP1();
        const std::vector< Pt3dr > & EchRepP1()const ;

        Pt3dr & PRep();
        const Pt3dr & PRep()const ;
    private:
        cXmlHomogr mHom12;
        bool mOk;
        double mQual;
        int mNbPts;
        std::vector< Pt3dr > mEchRepP1;
        Pt3dr mPRep;
};
cElXMLTree * ToXMLTree(const cXmlRHHResLnk &);

void  BinaryDumpInFile(ELISE_fp &,const cXmlRHHResLnk &);

void  BinaryUnDumpFromFile(cXmlRHHResLnk &,ELISE_fp &);

std::string  Mangling( cXmlRHHResLnk *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXMLSaveOriRel2Im
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXMLSaveOriRel2Im & anObj,cElXMLTree * aTree);


        cRotationVect & ParamRotation();
        const cRotationVect & ParamRotation()const ;

        Pt3dr & Centre();
        const Pt3dr & Centre()const ;

        cXmlHomogr & Homogr();
        const cXmlHomogr & Homogr()const ;

        double & BOnHRatio();
        const double & BOnHRatio()const ;

        double & FOVMin();
        const double & FOVMin()const ;

        double & FOVMax();
        const double & FOVMax()const ;
    private:
        cRotationVect mParamRotation;
        Pt3dr mCentre;
        cXmlHomogr mHomogr;
        double mBOnHRatio;
        double mFOVMin;
        double mFOVMax;
};
cElXMLTree * ToXMLTree(const cXMLSaveOriRel2Im &);

void  BinaryDumpInFile(ELISE_fp &,const cXMLSaveOriRel2Im &);

void  BinaryUnDumpFromFile(cXMLSaveOriRel2Im &,ELISE_fp &);

std::string  Mangling( cXMLSaveOriRel2Im *);

/******************************************************/
/******************************************************/
/******************************************************/
class cItem
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cItem & anObj,cElXMLTree * aTree);


        std::vector< Pt3dr > & Pt();
        const std::vector< Pt3dr > & Pt()const ;

        int & Mode();
        const int & Mode()const ;
    private:
        std::vector< Pt3dr > mPt;
        int mMode;
};
cElXMLTree * ToXMLTree(const cItem &);

void  BinaryDumpInFile(ELISE_fp &,const cItem &);

void  BinaryUnDumpFromFile(cItem &,ELISE_fp &);

std::string  Mangling( cItem *);

/******************************************************/
/******************************************************/
/******************************************************/
class cPolyg3D
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPolyg3D & anObj,cElXMLTree * aTree);


        std::vector< cItem > & Item();
        const std::vector< cItem > & Item()const ;
    private:
        std::vector< cItem > mItem;
};
cElXMLTree * ToXMLTree(const cPolyg3D &);

void  BinaryDumpInFile(ELISE_fp &,const cPolyg3D &);

void  BinaryUnDumpFromFile(cPolyg3D &,ELISE_fp &);

std::string  Mangling( cPolyg3D *);

/******************************************************/
/******************************************************/
/******************************************************/
class cXML_TestImportOri
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXML_TestImportOri & anObj,cElXMLTree * aTree);


        int & x();
        const int & x()const ;

        XmlXml & Tree();
        const XmlXml & Tree()const ;
    private:
        int mx;
        XmlXml mTree;
};
cElXMLTree * ToXMLTree(const cXML_TestImportOri &);

void  BinaryDumpInFile(ELISE_fp &,const cXML_TestImportOri &);

void  BinaryUnDumpFromFile(cXML_TestImportOri &,ELISE_fp &);

std::string  Mangling( cXML_TestImportOri *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotSupIm
