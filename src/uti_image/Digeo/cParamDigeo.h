#include "StdAfx.h"
#ifndef Define_NotDigeo
#define Define_NotDigeo
#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;
//namespace NS_ParamDigeo{
typedef enum
{
  eTtpSommet,
  eTtpCuvette,
  eTtpCol,
  eTtpCorner,
  eSiftMaxDog,
  eSiftMinDog
} eTypeTopolPt;
void xml_init(eTypeTopolPt & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypeTopolPt & aVal);

eTypeTopolPt  Str2eTypeTopolPt(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeTopolPt & anObj);

typedef enum
{
  eRDI_121,
  eRDI_010,
  eRDI_11
} eReducDemiImage;
void xml_init(eReducDemiImage & aVal,cElXMLTree * aTree);
std::string  eToString(const eReducDemiImage & aVal);

eReducDemiImage  Str2eReducDemiImage(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eReducDemiImage & anObj);

typedef enum
{
  eTest12345_A,
  eTest12345_B,
  eTest12345_C
} eTest12345;
void xml_init(eTest12345 & aVal,cElXMLTree * aTree);
std::string  eToString(const eTest12345 & aVal);

eTest12345  Str2eTest12345(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTest12345 & anObj);

class cParamExtractCaracIm
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamExtractCaracIm & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & SzMinImDeZoom();
        const cTplValGesInit< int > & SzMinImDeZoom()const ;
    private:
        cTplValGesInit< int > mSzMinImDeZoom;
};
cElXMLTree * ToXMLTree(const cParamExtractCaracIm &);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamVisuCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamVisuCarac & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & DynGray();
        const cTplValGesInit< int > & DynGray()const ;

        std::string & Dir();
        const std::string & Dir()const ;

        cTplValGesInit< int > & Zoom();
        const cTplValGesInit< int > & Zoom()const ;

        double & Dyn();
        const double & Dyn()const ;

        cTplValGesInit< std::string > & Prefix();
        const cTplValGesInit< std::string > & Prefix()const ;

        cTplValGesInit< bool > & ShowCaracEchec();
        const cTplValGesInit< bool > & ShowCaracEchec()const ;
    private:
        cTplValGesInit< int > mDynGray;
        std::string mDir;
        cTplValGesInit< int > mZoom;
        double mDyn;
        cTplValGesInit< std::string > mPrefix;
        cTplValGesInit< bool > mShowCaracEchec;
};
cElXMLTree * ToXMLTree(const cParamVisuCarac &);

/******************************************************/
/******************************************************/
/******************************************************/
class cPredicteurGeom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPredicteurGeom & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Unused();
        const cTplValGesInit< std::string > & Unused()const ;
    private:
        cTplValGesInit< std::string > mUnused;
};
cElXMLTree * ToXMLTree(const cPredicteurGeom &);

class cImageDigeo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImageDigeo & anObj,cElXMLTree * aTree);


        cTplValGesInit< cParamVisuCarac > & VisuCarac();
        const cTplValGesInit< cParamVisuCarac > & VisuCarac()const ;

        std::string & KeyOrPat();
        const std::string & KeyOrPat()const ;

        cTplValGesInit< std::string > & KeyCalcCalib();
        const cTplValGesInit< std::string > & KeyCalcCalib()const ;

        cTplValGesInit< Box2di > & BoxImR1();
        const cTplValGesInit< Box2di > & BoxImR1()const ;

        cTplValGesInit< double > & ResolInit();
        const cTplValGesInit< double > & ResolInit()const ;

        cTplValGesInit< std::string > & Unused();
        const cTplValGesInit< std::string > & Unused()const ;

        cTplValGesInit< cPredicteurGeom > & PredicteurGeom();
        const cTplValGesInit< cPredicteurGeom > & PredicteurGeom()const ;

        cTplValGesInit< double > & NbOctetLimitLoadImageOnce();
        const cTplValGesInit< double > & NbOctetLimitLoadImageOnce()const ;
    private:
        cTplValGesInit< cParamVisuCarac > mVisuCarac;
        std::string mKeyOrPat;
        cTplValGesInit< std::string > mKeyCalcCalib;
        cTplValGesInit< Box2di > mBoxImR1;
        cTplValGesInit< double > mResolInit;
        cTplValGesInit< cPredicteurGeom > mPredicteurGeom;
        cTplValGesInit< double > mNbOctetLimitLoadImageOnce;
};
cElXMLTree * ToXMLTree(const cImageDigeo &);

class cTypeNumeriqueOfNiv
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTypeNumeriqueOfNiv & anObj,cElXMLTree * aTree);


        eTypeNumerique & Type();
        const eTypeNumerique & Type()const ;

        int & Niv();
        const int & Niv()const ;
    private:
        eTypeNumerique mType;
        int mNiv;
};
cElXMLTree * ToXMLTree(const cTypeNumeriqueOfNiv &);

class cPyramideGaussienne
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPyramideGaussienne & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NbByOctave();
        const cTplValGesInit< int > & NbByOctave()const ;

        cTplValGesInit< double > & Sigma0();
        const cTplValGesInit< double > & Sigma0()const ;

        cTplValGesInit< double > & SigmaN();
        const cTplValGesInit< double > & SigmaN()const ;

        cTplValGesInit< int > & NbInLastOctave();
        const cTplValGesInit< int > & NbInLastOctave()const ;

        cTplValGesInit< int > & IndexFreqInFirstOctave();
        const cTplValGesInit< int > & IndexFreqInFirstOctave()const ;

        int & NivOctaveMax();
        const int & NivOctaveMax()const ;

        cTplValGesInit< double > & ConvolFirstImage();
        const cTplValGesInit< double > & ConvolFirstImage()const ;

        cTplValGesInit< double > & EpsilonGauss();
        const cTplValGesInit< double > & EpsilonGauss()const ;

        cTplValGesInit< int > & NbShift();
        const cTplValGesInit< int > & NbShift()const ;

        cTplValGesInit< int > & SurEchIntegralGauss();
        const cTplValGesInit< int > & SurEchIntegralGauss()const ;

        cTplValGesInit< bool > & ConvolIncrem();
        const cTplValGesInit< bool > & ConvolIncrem()const ;
    private:
        cTplValGesInit< int > mNbByOctave;
        cTplValGesInit< double > mSigma0;
        cTplValGesInit< double > mSigmaN;
        cTplValGesInit< int > mNbInLastOctave;
        cTplValGesInit< int > mIndexFreqInFirstOctave;
        int mNivOctaveMax;
        cTplValGesInit< double > mConvolFirstImage;
        cTplValGesInit< double > mEpsilonGauss;
        cTplValGesInit< int > mNbShift;
        cTplValGesInit< int > mSurEchIntegralGauss;
        cTplValGesInit< bool > mConvolIncrem;
};
cElXMLTree * ToXMLTree(const cPyramideGaussienne &);

class cTypePyramide
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTypePyramide & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NivPyramBasique();
        const cTplValGesInit< int > & NivPyramBasique()const ;

        cTplValGesInit< int > & NbByOctave();
        const cTplValGesInit< int > & NbByOctave()const ;

        cTplValGesInit< double > & Sigma0();
        const cTplValGesInit< double > & Sigma0()const ;

        cTplValGesInit< double > & SigmaN();
        const cTplValGesInit< double > & SigmaN()const ;

        cTplValGesInit< int > & NbInLastOctave();
        const cTplValGesInit< int > & NbInLastOctave()const ;

        cTplValGesInit< int > & IndexFreqInFirstOctave();
        const cTplValGesInit< int > & IndexFreqInFirstOctave()const ;

        int & NivOctaveMax();
        const int & NivOctaveMax()const ;

        cTplValGesInit< double > & ConvolFirstImage();
        const cTplValGesInit< double > & ConvolFirstImage()const ;

        cTplValGesInit< double > & EpsilonGauss();
        const cTplValGesInit< double > & EpsilonGauss()const ;

        cTplValGesInit< int > & NbShift();
        const cTplValGesInit< int > & NbShift()const ;

        cTplValGesInit< int > & SurEchIntegralGauss();
        const cTplValGesInit< int > & SurEchIntegralGauss()const ;

        cTplValGesInit< bool > & ConvolIncrem();
        const cTplValGesInit< bool > & ConvolIncrem()const ;

        cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne();
        const cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne()const ;
    private:
        cTplValGesInit< int > mNivPyramBasique;
        cTplValGesInit< cPyramideGaussienne > mPyramideGaussienne;
};
cElXMLTree * ToXMLTree(const cTypePyramide &);

class cPyramideImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPyramideImage & anObj,cElXMLTree * aTree);


        std::list< cTypeNumeriqueOfNiv > & TypeNumeriqueOfNiv();
        const std::list< cTypeNumeriqueOfNiv > & TypeNumeriqueOfNiv()const ;

        cTplValGesInit< bool > & MaximDyn();
        const cTplValGesInit< bool > & MaximDyn()const ;

        cTplValGesInit< double > & ValMaxForDyn();
        const cTplValGesInit< double > & ValMaxForDyn()const ;

        cTplValGesInit< eReducDemiImage > & ReducDemiImage();
        const cTplValGesInit< eReducDemiImage > & ReducDemiImage()const ;

        cTplValGesInit< int > & NivPyramBasique();
        const cTplValGesInit< int > & NivPyramBasique()const ;

        cTplValGesInit< int > & NbByOctave();
        const cTplValGesInit< int > & NbByOctave()const ;

        cTplValGesInit< double > & Sigma0();
        const cTplValGesInit< double > & Sigma0()const ;

        cTplValGesInit< double > & SigmaN();
        const cTplValGesInit< double > & SigmaN()const ;

        cTplValGesInit< int > & NbInLastOctave();
        const cTplValGesInit< int > & NbInLastOctave()const ;

        cTplValGesInit< int > & IndexFreqInFirstOctave();
        const cTplValGesInit< int > & IndexFreqInFirstOctave()const ;

        int & NivOctaveMax();
        const int & NivOctaveMax()const ;

        cTplValGesInit< double > & ConvolFirstImage();
        const cTplValGesInit< double > & ConvolFirstImage()const ;

        cTplValGesInit< double > & EpsilonGauss();
        const cTplValGesInit< double > & EpsilonGauss()const ;

        cTplValGesInit< int > & NbShift();
        const cTplValGesInit< int > & NbShift()const ;

        cTplValGesInit< int > & SurEchIntegralGauss();
        const cTplValGesInit< int > & SurEchIntegralGauss()const ;

        cTplValGesInit< bool > & ConvolIncrem();
        const cTplValGesInit< bool > & ConvolIncrem()const ;

        cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne();
        const cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne()const ;

        cTypePyramide & TypePyramide();
        const cTypePyramide & TypePyramide()const ;
    private:
        std::list< cTypeNumeriqueOfNiv > mTypeNumeriqueOfNiv;
        cTplValGesInit< bool > mMaximDyn;
        cTplValGesInit< double > mValMaxForDyn;
        cTplValGesInit< eReducDemiImage > mReducDemiImage;
        cTypePyramide mTypePyramide;
};
cElXMLTree * ToXMLTree(const cPyramideImage &);

class cSectionImages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionImages & anObj,cElXMLTree * aTree);


        std::list< cImageDigeo > & ImageDigeo();
        const std::list< cImageDigeo > & ImageDigeo()const ;

        std::list< cTypeNumeriqueOfNiv > & TypeNumeriqueOfNiv();
        const std::list< cTypeNumeriqueOfNiv > & TypeNumeriqueOfNiv()const ;

        cTplValGesInit< bool > & MaximDyn();
        const cTplValGesInit< bool > & MaximDyn()const ;

        cTplValGesInit< double > & ValMaxForDyn();
        const cTplValGesInit< double > & ValMaxForDyn()const ;

        cTplValGesInit< eReducDemiImage > & ReducDemiImage();
        const cTplValGesInit< eReducDemiImage > & ReducDemiImage()const ;

        cTplValGesInit< int > & NivPyramBasique();
        const cTplValGesInit< int > & NivPyramBasique()const ;

        cTplValGesInit< int > & NbByOctave();
        const cTplValGesInit< int > & NbByOctave()const ;

        cTplValGesInit< double > & Sigma0();
        const cTplValGesInit< double > & Sigma0()const ;

        cTplValGesInit< double > & SigmaN();
        const cTplValGesInit< double > & SigmaN()const ;

        cTplValGesInit< int > & NbInLastOctave();
        const cTplValGesInit< int > & NbInLastOctave()const ;

        cTplValGesInit< int > & IndexFreqInFirstOctave();
        const cTplValGesInit< int > & IndexFreqInFirstOctave()const ;

        int & NivOctaveMax();
        const int & NivOctaveMax()const ;

        cTplValGesInit< double > & ConvolFirstImage();
        const cTplValGesInit< double > & ConvolFirstImage()const ;

        cTplValGesInit< double > & EpsilonGauss();
        const cTplValGesInit< double > & EpsilonGauss()const ;

        cTplValGesInit< int > & NbShift();
        const cTplValGesInit< int > & NbShift()const ;

        cTplValGesInit< int > & SurEchIntegralGauss();
        const cTplValGesInit< int > & SurEchIntegralGauss()const ;

        cTplValGesInit< bool > & ConvolIncrem();
        const cTplValGesInit< bool > & ConvolIncrem()const ;

        cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne();
        const cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne()const ;

        cTypePyramide & TypePyramide();
        const cTypePyramide & TypePyramide()const ;

        cPyramideImage & PyramideImage();
        const cPyramideImage & PyramideImage()const ;
    private:
        std::list< cImageDigeo > mImageDigeo;
        cPyramideImage mPyramideImage;
};
cElXMLTree * ToXMLTree(const cSectionImages &);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneCarac & anObj,cElXMLTree * aTree);


        eTypeTopolPt & Type();
        const eTypeTopolPt & Type()const ;
    private:
        eTypeTopolPt mType;
};
cElXMLTree * ToXMLTree(const cOneCarac &);

class cCaracTopo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCaracTopo & anObj,cElXMLTree * aTree);


        std::list< cOneCarac > & OneCarac();
        const std::list< cOneCarac > & OneCarac()const ;
    private:
        std::list< cOneCarac > mOneCarac;
};
cElXMLTree * ToXMLTree(const cCaracTopo &);

class cSiftCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSiftCarac & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & DoMax();
        const cTplValGesInit< bool > & DoMax()const ;

        cTplValGesInit< bool > & DoMin();
        const cTplValGesInit< bool > & DoMin()const ;

        cTplValGesInit< int > & NivEstimGradMoy();
        const cTplValGesInit< int > & NivEstimGradMoy()const ;

        cTplValGesInit< double > & RatioAllongMin();
        const cTplValGesInit< double > & RatioAllongMin()const ;

        cTplValGesInit< double > & RatioGrad();
        const cTplValGesInit< double > & RatioGrad()const ;
    private:
        cTplValGesInit< bool > mDoMax;
        cTplValGesInit< bool > mDoMin;
        cTplValGesInit< int > mNivEstimGradMoy;
        cTplValGesInit< double > mRatioAllongMin;
        cTplValGesInit< double > mRatioGrad;
};
cElXMLTree * ToXMLTree(const cSiftCarac &);

class cSectionCaracImages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionCaracImages & anObj,cElXMLTree * aTree);


        bool & ComputeCarac();
        const bool & ComputeCarac()const ;

        std::list< cOneCarac > & OneCarac();
        const std::list< cOneCarac > & OneCarac()const ;

        cTplValGesInit< cCaracTopo > & CaracTopo();
        const cTplValGesInit< cCaracTopo > & CaracTopo()const ;

        cTplValGesInit< bool > & DoMax();
        const cTplValGesInit< bool > & DoMax()const ;

        cTplValGesInit< bool > & DoMin();
        const cTplValGesInit< bool > & DoMin()const ;

        cTplValGesInit< int > & NivEstimGradMoy();
        const cTplValGesInit< int > & NivEstimGradMoy()const ;

        cTplValGesInit< double > & RatioAllongMin();
        const cTplValGesInit< double > & RatioAllongMin()const ;

        cTplValGesInit< double > & RatioGrad();
        const cTplValGesInit< double > & RatioGrad()const ;

        cTplValGesInit< cSiftCarac > & SiftCarac();
        const cTplValGesInit< cSiftCarac > & SiftCarac()const ;
    private:
        bool mComputeCarac;
        cTplValGesInit< cCaracTopo > mCaracTopo;
        cTplValGesInit< cSiftCarac > mSiftCarac;
};
cElXMLTree * ToXMLTree(const cSectionCaracImages &);

/******************************************************/
/******************************************************/
/******************************************************/
class cGenereRandomRect
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenereRandomRect & anObj,cElXMLTree * aTree);


        int & NbRect();
        const int & NbRect()const ;

        int & SzRect();
        const int & SzRect()const ;
    private:
        int mNbRect;
        int mSzRect;
};
cElXMLTree * ToXMLTree(const cGenereRandomRect &);

class cGenereCarroyage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenereCarroyage & anObj,cElXMLTree * aTree);


        int & PerX();
        const int & PerX()const ;

        int & PerY();
        const int & PerY()const ;
    private:
        int mPerX;
        int mPerY;
};
cElXMLTree * ToXMLTree(const cGenereCarroyage &);

class cGenereAllRandom
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenereAllRandom & anObj,cElXMLTree * aTree);


        int & SzFilter();
        const int & SzFilter()const ;
    private:
        int mSzFilter;
};
cElXMLTree * ToXMLTree(const cGenereAllRandom &);

class cSectionTest
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionTest & anObj,cElXMLTree * aTree);


        int & NbRect();
        const int & NbRect()const ;

        int & SzRect();
        const int & SzRect()const ;

        cTplValGesInit< cGenereRandomRect > & GenereRandomRect();
        const cTplValGesInit< cGenereRandomRect > & GenereRandomRect()const ;

        int & PerX();
        const int & PerX()const ;

        int & PerY();
        const int & PerY()const ;

        cTplValGesInit< cGenereCarroyage > & GenereCarroyage();
        const cTplValGesInit< cGenereCarroyage > & GenereCarroyage()const ;

        int & SzFilter();
        const int & SzFilter()const ;

        cTplValGesInit< cGenereAllRandom > & GenereAllRandom();
        const cTplValGesInit< cGenereAllRandom > & GenereAllRandom()const ;

        cTplValGesInit< bool > & VerifExtrema();
        const cTplValGesInit< bool > & VerifExtrema()const ;
    private:
        cTplValGesInit< cGenereRandomRect > mGenereRandomRect;
        cTplValGesInit< cGenereCarroyage > mGenereCarroyage;
        cTplValGesInit< cGenereAllRandom > mGenereAllRandom;
        cTplValGesInit< bool > mVerifExtrema;
};
cElXMLTree * ToXMLTree(const cSectionTest &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSauvPyram
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSauvPyram & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Dir();
        const cTplValGesInit< std::string > & Dir()const ;

        cTplValGesInit< bool > & Glob();
        const cTplValGesInit< bool > & Glob()const ;

        cTplValGesInit< std::string > & Key();
        const cTplValGesInit< std::string > & Key()const ;

        cTplValGesInit< int > & StripTifFile();
        const cTplValGesInit< int > & StripTifFile()const ;

        cTplValGesInit< bool > & Force8B();
        const cTplValGesInit< bool > & Force8B()const ;

        cTplValGesInit< double > & Dyn();
        const cTplValGesInit< double > & Dyn()const ;
    private:
        cTplValGesInit< std::string > mDir;
        cTplValGesInit< bool > mGlob;
        cTplValGesInit< std::string > mKey;
        cTplValGesInit< int > mStripTifFile;
        cTplValGesInit< bool > mForce8B;
        cTplValGesInit< double > mDyn;
};
cElXMLTree * ToXMLTree(const cSauvPyram &);

class cDigeoDecoupageCarac
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDigeoDecoupageCarac & anObj,cElXMLTree * aTree);


        int & SzDalle();
        const int & SzDalle()const ;

        int & Bord();
        const int & Bord()const ;
    private:
        int mSzDalle;
        int mBord;
};
cElXMLTree * ToXMLTree(const cDigeoDecoupageCarac &);

class cModifGCC
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cModifGCC & anObj,cElXMLTree * aTree);


        int & NbByOctave();
        const int & NbByOctave()const ;

        bool & ConvolIncrem();
        const bool & ConvolIncrem()const ;

        eTypeNumerique & TypeNum();
        const eTypeNumerique & TypeNum()const ;
    private:
        int mNbByOctave;
        bool mConvolIncrem;
        eTypeNumerique mTypeNum;
};
cElXMLTree * ToXMLTree(const cModifGCC &);

class cGenereCodeConvol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenereCodeConvol & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Dir();
        const cTplValGesInit< std::string > & Dir()const ;

        cTplValGesInit< std::string > & File();
        const cTplValGesInit< std::string > & File()const ;

        std::vector< cModifGCC > & ModifGCC();
        const std::vector< cModifGCC > & ModifGCC()const ;
    private:
        cTplValGesInit< std::string > mDir;
        cTplValGesInit< std::string > mFile;
        std::vector< cModifGCC > mModifGCC;
};
cElXMLTree * ToXMLTree(const cGenereCodeConvol &);

class cFenVisu
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cFenVisu & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        Pt2di & Sz();
        const Pt2di & Sz()const ;
    private:
        std::string mName;
        Pt2di mSz;
};
cElXMLTree * ToXMLTree(const cFenVisu &);

class cSectionWorkSpace
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionWorkSpace & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & DirectoryChantier();
        const cTplValGesInit< std::string > & DirectoryChantier()const ;

        cTplValGesInit< std::string > & Dir();
        const cTplValGesInit< std::string > & Dir()const ;

        cTplValGesInit< bool > & Glob();
        const cTplValGesInit< bool > & Glob()const ;

        cTplValGesInit< std::string > & Key();
        const cTplValGesInit< std::string > & Key()const ;

        cTplValGesInit< int > & StripTifFile();
        const cTplValGesInit< int > & StripTifFile()const ;

        cTplValGesInit< bool > & Force8B();
        const cTplValGesInit< bool > & Force8B()const ;

        cTplValGesInit< double > & Dyn();
        const cTplValGesInit< double > & Dyn()const ;

        cTplValGesInit< cSauvPyram > & SauvPyram();
        const cTplValGesInit< cSauvPyram > & SauvPyram()const ;

        int & SzDalle();
        const int & SzDalle()const ;

        int & Bord();
        const int & Bord()const ;

        cTplValGesInit< cDigeoDecoupageCarac > & DigeoDecoupageCarac();
        const cTplValGesInit< cDigeoDecoupageCarac > & DigeoDecoupageCarac()const ;

        cTplValGesInit< bool > & ExigeCodeCompile();
        const cTplValGesInit< bool > & ExigeCodeCompile()const ;

        cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol();
        const cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol()const ;

        cTplValGesInit< int > & ShowTimes();
        const cTplValGesInit< int > & ShowTimes()const ;

        std::list< cFenVisu > & FenVisu();
        const std::list< cFenVisu > & FenVisu()const ;

        cTplValGesInit< bool > & ShowConvolSpec();
        const cTplValGesInit< bool > & ShowConvolSpec()const ;

        cTplValGesInit< bool > & Verbose();
        const cTplValGesInit< bool > & Verbose()const ;
    private:
        cTplValGesInit< std::string > mDirectoryChantier;
        cTplValGesInit< cSauvPyram > mSauvPyram;
        cTplValGesInit< cDigeoDecoupageCarac > mDigeoDecoupageCarac;
        cTplValGesInit< bool > mExigeCodeCompile;
        cTplValGesInit< cGenereCodeConvol > mGenereCodeConvol;
        cTplValGesInit< int > mShowTimes;
        std::list< cFenVisu > mFenVisu;
        cTplValGesInit< bool > mShowConvolSpec;
        cTplValGesInit< bool > mVerbose;
};
cElXMLTree * ToXMLTree(const cSectionWorkSpace &);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamDigeo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamDigeo & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        std::list< cImageDigeo > & ImageDigeo();
        const std::list< cImageDigeo > & ImageDigeo()const ;

        std::list< cTypeNumeriqueOfNiv > & TypeNumeriqueOfNiv();
        const std::list< cTypeNumeriqueOfNiv > & TypeNumeriqueOfNiv()const ;

        cTplValGesInit< bool > & MaximDyn();
        const cTplValGesInit< bool > & MaximDyn()const ;

        cTplValGesInit< double > & ValMaxForDyn();
        const cTplValGesInit< double > & ValMaxForDyn()const ;

        cTplValGesInit< eReducDemiImage > & ReducDemiImage();
        const cTplValGesInit< eReducDemiImage > & ReducDemiImage()const ;

        cTplValGesInit< int > & NivPyramBasique();
        const cTplValGesInit< int > & NivPyramBasique()const ;

        cTplValGesInit< int > & NbByOctave();
        const cTplValGesInit< int > & NbByOctave()const ;

        cTplValGesInit< double > & Sigma0();
        const cTplValGesInit< double > & Sigma0()const ;

        cTplValGesInit< double > & SigmaN();
        const cTplValGesInit< double > & SigmaN()const ;

        cTplValGesInit< int > & NbInLastOctave();
        const cTplValGesInit< int > & NbInLastOctave()const ;

        cTplValGesInit< int > & IndexFreqInFirstOctave();
        const cTplValGesInit< int > & IndexFreqInFirstOctave()const ;

        int & NivOctaveMax();
        const int & NivOctaveMax()const ;

        cTplValGesInit< double > & ConvolFirstImage();
        const cTplValGesInit< double > & ConvolFirstImage()const ;

        cTplValGesInit< double > & EpsilonGauss();
        const cTplValGesInit< double > & EpsilonGauss()const ;

        cTplValGesInit< int > & NbShift();
        const cTplValGesInit< int > & NbShift()const ;

        cTplValGesInit< int > & SurEchIntegralGauss();
        const cTplValGesInit< int > & SurEchIntegralGauss()const ;

        cTplValGesInit< bool > & ConvolIncrem();
        const cTplValGesInit< bool > & ConvolIncrem()const ;

        cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne();
        const cTplValGesInit< cPyramideGaussienne > & PyramideGaussienne()const ;

        cTypePyramide & TypePyramide();
        const cTypePyramide & TypePyramide()const ;

        cPyramideImage & PyramideImage();
        const cPyramideImage & PyramideImage()const ;

        cSectionImages & SectionImages();
        const cSectionImages & SectionImages()const ;

        bool & ComputeCarac();
        const bool & ComputeCarac()const ;

        std::list< cOneCarac > & OneCarac();
        const std::list< cOneCarac > & OneCarac()const ;

        cTplValGesInit< cCaracTopo > & CaracTopo();
        const cTplValGesInit< cCaracTopo > & CaracTopo()const ;

        cTplValGesInit< bool > & DoMax();
        const cTplValGesInit< bool > & DoMax()const ;

        cTplValGesInit< bool > & DoMin();
        const cTplValGesInit< bool > & DoMin()const ;

        cTplValGesInit< int > & NivEstimGradMoy();
        const cTplValGesInit< int > & NivEstimGradMoy()const ;

        cTplValGesInit< double > & RatioAllongMin();
        const cTplValGesInit< double > & RatioAllongMin()const ;

        cTplValGesInit< double > & RatioGrad();
        const cTplValGesInit< double > & RatioGrad()const ;

        cTplValGesInit< cSiftCarac > & SiftCarac();
        const cTplValGesInit< cSiftCarac > & SiftCarac()const ;

        cSectionCaracImages & SectionCaracImages();
        const cSectionCaracImages & SectionCaracImages()const ;

        int & NbRect();
        const int & NbRect()const ;

        int & SzRect();
        const int & SzRect()const ;

        cTplValGesInit< cGenereRandomRect > & GenereRandomRect();
        const cTplValGesInit< cGenereRandomRect > & GenereRandomRect()const ;

        int & PerX();
        const int & PerX()const ;

        int & PerY();
        const int & PerY()const ;

        cTplValGesInit< cGenereCarroyage > & GenereCarroyage();
        const cTplValGesInit< cGenereCarroyage > & GenereCarroyage()const ;

        int & SzFilter();
        const int & SzFilter()const ;

        cTplValGesInit< cGenereAllRandom > & GenereAllRandom();
        const cTplValGesInit< cGenereAllRandom > & GenereAllRandom()const ;

        cTplValGesInit< bool > & VerifExtrema();
        const cTplValGesInit< bool > & VerifExtrema()const ;

        cTplValGesInit< cSectionTest > & SectionTest();
        const cTplValGesInit< cSectionTest > & SectionTest()const ;

        cTplValGesInit< std::string > & DirectoryChantier();
        const cTplValGesInit< std::string > & DirectoryChantier()const ;

        cTplValGesInit< std::string > & Dir();
        const cTplValGesInit< std::string > & Dir()const ;

        cTplValGesInit< bool > & Glob();
        const cTplValGesInit< bool > & Glob()const ;

        cTplValGesInit< std::string > & Key();
        const cTplValGesInit< std::string > & Key()const ;

        cTplValGesInit< int > & StripTifFile();
        const cTplValGesInit< int > & StripTifFile()const ;

        cTplValGesInit< bool > & Force8B();
        const cTplValGesInit< bool > & Force8B()const ;

        cTplValGesInit< double > & Dyn();
        const cTplValGesInit< double > & Dyn()const ;

        cTplValGesInit< cSauvPyram > & SauvPyram();
        const cTplValGesInit< cSauvPyram > & SauvPyram()const ;

        int & SzDalle();
        const int & SzDalle()const ;

        int & Bord();
        const int & Bord()const ;

        cTplValGesInit< cDigeoDecoupageCarac > & DigeoDecoupageCarac();
        const cTplValGesInit< cDigeoDecoupageCarac > & DigeoDecoupageCarac()const ;

        cTplValGesInit< bool > & ExigeCodeCompile();
        const cTplValGesInit< bool > & ExigeCodeCompile()const ;

        cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol();
        const cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol()const ;

        cTplValGesInit< int > & ShowTimes();
        const cTplValGesInit< int > & ShowTimes()const ;

        std::list< cFenVisu > & FenVisu();
        const std::list< cFenVisu > & FenVisu()const ;

        cTplValGesInit< bool > & ShowConvolSpec();
        const cTplValGesInit< bool > & ShowConvolSpec()const ;

        cTplValGesInit< bool > & Verbose();
        const cTplValGesInit< bool > & Verbose()const ;

        cSectionWorkSpace & SectionWorkSpace();
        const cSectionWorkSpace & SectionWorkSpace()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cSectionImages mSectionImages;
        cSectionCaracImages mSectionCaracImages;
        cTplValGesInit< cSectionTest > mSectionTest;
        cSectionWorkSpace mSectionWorkSpace;
};
cElXMLTree * ToXMLTree(const cParamDigeo &);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotDigeo
