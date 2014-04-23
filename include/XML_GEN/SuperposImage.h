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
    private:
        cTplValGesInit< cXmlCylindreRevolution > mCyl;
        cTplValGesInit< cXmlOrthoCyl > mOrthoCyl;
};
cElXMLTree * ToXMLTree(const cXmlDescriptionAnalytique &);

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

class cPN3M_Nuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPN3M_Nuage & anObj,cElXMLTree * aTree);


        cTplValGesInit< cImage_Point3D > & Image_Point3D();
        const cTplValGesInit< cImage_Point3D > & Image_Point3D()const ;

        cTplValGesInit< cImage_Profondeur > & Image_Profondeur();
        const cTplValGesInit< cImage_Profondeur > & Image_Profondeur()const ;
    private:
        cTplValGesInit< cImage_Point3D > mImage_Point3D;
        cTplValGesInit< cImage_Profondeur > mImage_Profondeur;
};
cElXMLTree * ToXMLTree(const cPN3M_Nuage &);

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

        Pt2di & NbPixel();
        const Pt2di & NbPixel()const ;

        cTplValGesInit< cImage_Point3D > & Image_Point3D();
        const cTplValGesInit< cImage_Point3D > & Image_Point3D()const ;

        cTplValGesInit< cImage_Profondeur > & Image_Profondeur();
        const cTplValGesInit< cImage_Profondeur > & Image_Profondeur()const ;

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
        Pt2di mNbPixel;
        cPN3M_Nuage mPN3M_Nuage;
        std::list< cAttributsNuage3D > mAttributsNuage3D;
        cTplValGesInit< cRepereCartesien > mRepereGlob;
        cTplValGesInit< cXmlOneSurfaceAnalytique > mAnam;
        cOrientationConique mOrientation;
        cPM3D_ParamSpecifs mPM3D_ParamSpecifs;
        cTplValGesInit< double > mTolVerifNuage;
        std::list< cVerifNuage > mVerifNuage;
};
cElXMLTree * ToXMLTree(const cXML_ParamNuage3DMaille &);

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

        cTplValGesInit< std::string > & ModeleNuageResult();
        const cTplValGesInit< std::string > & ModeleNuageResult()const ;
    private:
        std::string mKeyNuage;
        std::string mKeyResult;
        cTplValGesInit< std::string > mModeleNuageResult;
};
cElXMLTree * ToXMLTree(const cSectionName &);

/******************************************************/
/******************************************************/
/******************************************************/
class cScoreMM1P
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cScoreMM1P & anObj,cElXMLTree * aTree);


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
        cTplValGesInit< double > mPdsAR;
        cTplValGesInit< double > mPdsDistor;
        cTplValGesInit< double > mAmplImDistor;
        cTplValGesInit< double > mSeuilDist;
        cTplValGesInit< double > mPdsDistBord;
        cTplValGesInit< double > mSeuilDisBord;
};
cElXMLTree * ToXMLTree(const cScoreMM1P &);

class cSectionScoreQualite
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionScoreQualite & anObj,cElXMLTree * aTree);


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

/******************************************************/
/******************************************************/
/******************************************************/
class cFMNtBySort
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFMNtBySort & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & PercFusion();
        const cTplValGesInit< double > & PercFusion()const ;
    private:
        cTplValGesInit< double > mPercFusion;
};
cElXMLTree * ToXMLTree(const cFMNtBySort &);

class cFMNT_GesNoVal
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFMNT_GesNoVal & anObj,cElXMLTree * aTree);


        double & PenteMax();
        const double & PenteMax()const ;

        double & GainNoVal();
        const double & GainNoVal()const ;

        double & Trans();
        const double & Trans()const ;
    private:
        double mPenteMax;
        double mGainNoVal;
        double mTrans;
};
cElXMLTree * ToXMLTree(const cFMNT_GesNoVal &);

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

        double & GainNoVal();
        const double & GainNoVal()const ;

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

class cFMNtByMaxEvid
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFMNtByMaxEvid & anObj,cElXMLTree * aTree);


        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

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

        double & GainNoVal();
        const double & GainNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;
    private:
        double mSigmaPds;
        cTplValGesInit< double > mSigmaZ;
        cTplValGesInit< double > mMaxDif;
        cTplValGesInit< bool > mQuickExp;
        cTplValGesInit< cFMNT_ProgDyn > mFMNT_ProgDyn;
};
cElXMLTree * ToXMLTree(const cFMNtByMaxEvid &);

class cSpecAlgoFMNT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSpecAlgoFMNT & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & PercFusion();
        const cTplValGesInit< double > & PercFusion()const ;

        cTplValGesInit< cFMNtBySort > & FMNtBySort();
        const cTplValGesInit< cFMNtBySort > & FMNtBySort()const ;

        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

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

        double & GainNoVal();
        const double & GainNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;

        cTplValGesInit< cFMNtByMaxEvid > & FMNtByMaxEvid();
        const cTplValGesInit< cFMNtByMaxEvid > & FMNtByMaxEvid()const ;
    private:
        cTplValGesInit< cFMNtBySort > mFMNtBySort;
        cTplValGesInit< cFMNtByMaxEvid > mFMNtByMaxEvid;
};
cElXMLTree * ToXMLTree(const cSpecAlgoFMNT &);

class cParamAlgoFusionMNT
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamAlgoFusionMNT & anObj,cElXMLTree * aTree);


        double & FMNTSeuilCorrel();
        const double & FMNTSeuilCorrel()const ;

        double & FMNTGammaCorrel();
        const double & FMNTGammaCorrel()const ;

        cTplValGesInit< double > & PercFusion();
        const cTplValGesInit< double > & PercFusion()const ;

        cTplValGesInit< cFMNtBySort > & FMNtBySort();
        const cTplValGesInit< cFMNtBySort > & FMNtBySort()const ;

        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

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

        double & GainNoVal();
        const double & GainNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;

        cTplValGesInit< cFMNtByMaxEvid > & FMNtByMaxEvid();
        const cTplValGesInit< cFMNtByMaxEvid > & FMNtByMaxEvid()const ;

        cSpecAlgoFMNT & SpecAlgoFMNT();
        const cSpecAlgoFMNT & SpecAlgoFMNT()const ;
    private:
        double mFMNTSeuilCorrel;
        double mFMNTGammaCorrel;
        cSpecAlgoFMNT mSpecAlgoFMNT;
};
cElXMLTree * ToXMLTree(const cParamAlgoFusionMNT &);

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
        cTplValGesInit< bool > mInterneCalledByProcess;
        cTplValGesInit< std::string > mInterneSingleImage;
        cTplValGesInit< int > mInterneSingleBox;
        cTplValGesInit< std::string > mWorkDirPFM;
        cTplValGesInit< Box2di > mBoxTest;
};
cElXMLTree * ToXMLTree(const cSectionGestionChantier &);

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

        cTplValGesInit< std::string > & ModeleNuageResult();
        const cTplValGesInit< std::string > & ModeleNuageResult()const ;

        cSectionName & SectionName();
        const cSectionName & SectionName()const ;

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

        cTplValGesInit< double > & PercFusion();
        const cTplValGesInit< double > & PercFusion()const ;

        cTplValGesInit< cFMNtBySort > & FMNtBySort();
        const cTplValGesInit< cFMNtBySort > & FMNtBySort()const ;

        double & SigmaPds();
        const double & SigmaPds()const ;

        cTplValGesInit< double > & SigmaZ();
        const cTplValGesInit< double > & SigmaZ()const ;

        cTplValGesInit< double > & MaxDif();
        const cTplValGesInit< double > & MaxDif()const ;

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

        double & GainNoVal();
        const double & GainNoVal()const ;

        double & Trans();
        const double & Trans()const ;

        cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal();
        const cTplValGesInit< cFMNT_GesNoVal > & FMNT_GesNoVal()const ;

        cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn();
        const cTplValGesInit< cFMNT_ProgDyn > & FMNT_ProgDyn()const ;

        cTplValGesInit< cFMNtByMaxEvid > & FMNtByMaxEvid();
        const cTplValGesInit< cFMNtByMaxEvid > & FMNtByMaxEvid()const ;

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
    private:
        cXmlHomogr mHom12;
        bool mOk;
        double mQual;
};
cElXMLTree * ToXMLTree(const cXmlRHHResLnk &);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotSupIm
