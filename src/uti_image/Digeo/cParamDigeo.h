#ifndef Define_NotDigeo
	#define Define_NotDigeo
#include "XML_GEN/all.h"
//
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

void  BinaryDumpInFile(ELISE_fp &,const eTypeTopolPt &);

std::string  Mangling( eTypeTopolPt *);

void  BinaryUnDumpFromFile(eTypeTopolPt &,ELISE_fp &);

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

void  BinaryDumpInFile(ELISE_fp &,const eReducDemiImage &);

std::string  Mangling( eReducDemiImage *);

void  BinaryUnDumpFromFile(eReducDemiImage &,ELISE_fp &);

typedef enum
{
  eRefine2D,
  eRefine3D,
  eRefineNone
} ePointRefinement;
void xml_init(ePointRefinement & aVal,cElXMLTree * aTree);
std::string  eToString(const ePointRefinement & aVal);

ePointRefinement  Str2ePointRefinement(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const ePointRefinement & anObj);

void  BinaryDumpInFile(ELISE_fp &,const ePointRefinement &);

std::string  Mangling( ePointRefinement *);

void  BinaryUnDumpFromFile(ePointRefinement &,ELISE_fp &);

class cImageDigeo
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImageDigeo & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & ResolInit();
        const cTplValGesInit< double > & ResolInit()const ;

        cTplValGesInit< double > & NbOctetLimitLoadImageOnce();
        const cTplValGesInit< double > & NbOctetLimitLoadImageOnce()const ;
    private:
        cTplValGesInit< double > mResolInit;
        cTplValGesInit< double > mNbOctetLimitLoadImageOnce;
};
cElXMLTree * ToXMLTree(const cImageDigeo &);

void  BinaryDumpInFile(ELISE_fp &,const cImageDigeo &);

void  BinaryUnDumpFromFile(cImageDigeo &,ELISE_fp &);

std::string  Mangling( cImageDigeo *);

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

void  BinaryDumpInFile(ELISE_fp &,const cTypeNumeriqueOfNiv &);

void  BinaryUnDumpFromFile(cTypeNumeriqueOfNiv &,ELISE_fp &);

std::string  Mangling( cTypeNumeriqueOfNiv *);

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

        cTplValGesInit< bool > & SampledConvolutionKernels();
        const cTplValGesInit< bool > & SampledConvolutionKernels()const ;

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
        cTplValGesInit< bool > mSampledConvolutionKernels;
        cTplValGesInit< double > mConvolFirstImage;
        cTplValGesInit< double > mEpsilonGauss;
        cTplValGesInit< int > mNbShift;
        cTplValGesInit< int > mSurEchIntegralGauss;
        cTplValGesInit< bool > mConvolIncrem;
};
cElXMLTree * ToXMLTree(const cPyramideGaussienne &);

void  BinaryDumpInFile(ELISE_fp &,const cPyramideGaussienne &);

void  BinaryUnDumpFromFile(cPyramideGaussienne &,ELISE_fp &);

std::string  Mangling( cPyramideGaussienne *);

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

        cTplValGesInit< bool > & SampledConvolutionKernels();
        const cTplValGesInit< bool > & SampledConvolutionKernels()const ;

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

void  BinaryDumpInFile(ELISE_fp &,const cTypePyramide &);

void  BinaryUnDumpFromFile(cTypePyramide &,ELISE_fp &);

std::string  Mangling( cTypePyramide *);

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

        cTplValGesInit< bool > & SampledConvolutionKernels();
        const cTplValGesInit< bool > & SampledConvolutionKernels()const ;

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

void  BinaryDumpInFile(ELISE_fp &,const cPyramideImage &);

void  BinaryUnDumpFromFile(cPyramideImage &,ELISE_fp &);

std::string  Mangling( cPyramideImage *);

class cDigeoSectionImages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDigeoSectionImages & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & ResolInit();
        const cTplValGesInit< double > & ResolInit()const ;

        cTplValGesInit< double > & NbOctetLimitLoadImageOnce();
        const cTplValGesInit< double > & NbOctetLimitLoadImageOnce()const ;

        cImageDigeo & ImageDigeo();
        const cImageDigeo & ImageDigeo()const ;

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

        cTplValGesInit< bool > & SampledConvolutionKernels();
        const cTplValGesInit< bool > & SampledConvolutionKernels()const ;

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
        cImageDigeo mImageDigeo;
        cPyramideImage mPyramideImage;
};
cElXMLTree * ToXMLTree(const cDigeoSectionImages &);

void  BinaryDumpInFile(ELISE_fp &,const cDigeoSectionImages &);

void  BinaryUnDumpFromFile(cDigeoSectionImages &,ELISE_fp &);

std::string  Mangling( cDigeoSectionImages *);

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

void  BinaryDumpInFile(ELISE_fp &,const cOneCarac &);

void  BinaryUnDumpFromFile(cOneCarac &,ELISE_fp &);

std::string  Mangling( cOneCarac *);

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

void  BinaryDumpInFile(ELISE_fp &,const cCaracTopo &);

void  BinaryUnDumpFromFile(cCaracTopo &,ELISE_fp &);

std::string  Mangling( cCaracTopo *);

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

        cTplValGesInit< ePointRefinement > & RefinementMethod();
        const cTplValGesInit< ePointRefinement > & RefinementMethod()const ;
    private:
        cTplValGesInit< bool > mDoMax;
        cTplValGesInit< bool > mDoMin;
        cTplValGesInit< int > mNivEstimGradMoy;
        cTplValGesInit< double > mRatioAllongMin;
        cTplValGesInit< double > mRatioGrad;
        cTplValGesInit< ePointRefinement > mRefinementMethod;
};
cElXMLTree * ToXMLTree(const cSiftCarac &);

void  BinaryDumpInFile(ELISE_fp &,const cSiftCarac &);

void  BinaryUnDumpFromFile(cSiftCarac &,ELISE_fp &);

std::string  Mangling( cSiftCarac *);

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

        cTplValGesInit< ePointRefinement > & RefinementMethod();
        const cTplValGesInit< ePointRefinement > & RefinementMethod()const ;

        cTplValGesInit< cSiftCarac > & SiftCarac();
        const cTplValGesInit< cSiftCarac > & SiftCarac()const ;
    private:
        bool mComputeCarac;
        cTplValGesInit< cCaracTopo > mCaracTopo;
        cTplValGesInit< cSiftCarac > mSiftCarac;
};
cElXMLTree * ToXMLTree(const cSectionCaracImages &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionCaracImages &);

void  BinaryUnDumpFromFile(cSectionCaracImages &,ELISE_fp &);

std::string  Mangling( cSectionCaracImages *);

/******************************************************/
/******************************************************/
/******************************************************/
class cDigeoTestOutput
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cDigeoTestOutput & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & OutputGaussians();
        const cTplValGesInit< bool > & OutputGaussians()const ;

        cTplValGesInit< std::string > & OutputGaussiansDirectory();
        const cTplValGesInit< std::string > & OutputGaussiansDirectory()const ;

        cTplValGesInit< bool > & OutputTiles();
        const cTplValGesInit< bool > & OutputTiles()const ;

        cTplValGesInit< std::string > & OutputTilesDirectory();
        const cTplValGesInit< std::string > & OutputTilesDirectory()const ;

        cTplValGesInit< bool > & OutputGradients();
        const cTplValGesInit< bool > & OutputGradients()const ;

        cTplValGesInit< std::string > & OutputGradientsNormDirectory();
        const cTplValGesInit< std::string > & OutputGradientsNormDirectory()const ;

        cTplValGesInit< std::string > & OutputGradientsAngleDirectory();
        const cTplValGesInit< std::string > & OutputGradientsAngleDirectory()const ;

        cTplValGesInit< bool > & MergeTiles();
        const cTplValGesInit< bool > & MergeTiles()const ;

        cTplValGesInit< bool > & SuppressTiles();
        const cTplValGesInit< bool > & SuppressTiles()const ;

        cTplValGesInit< bool > & ForceGradientComputation();
        const cTplValGesInit< bool > & ForceGradientComputation()const ;

        cTplValGesInit< bool > & PlotPointsOnTiles();
        const cTplValGesInit< bool > & PlotPointsOnTiles()const ;

        cTplValGesInit< bool > & RawOutput();
        const cTplValGesInit< bool > & RawOutput()const ;
    private:
        cTplValGesInit< bool > mOutputGaussians;
        cTplValGesInit< std::string > mOutputGaussiansDirectory;
        cTplValGesInit< bool > mOutputTiles;
        cTplValGesInit< std::string > mOutputTilesDirectory;
        cTplValGesInit< bool > mOutputGradients;
        cTplValGesInit< std::string > mOutputGradientsNormDirectory;
        cTplValGesInit< std::string > mOutputGradientsAngleDirectory;
        cTplValGesInit< bool > mMergeTiles;
        cTplValGesInit< bool > mSuppressTiles;
        cTplValGesInit< bool > mForceGradientComputation;
        cTplValGesInit< bool > mPlotPointsOnTiles;
        cTplValGesInit< bool > mRawOutput;
};
cElXMLTree * ToXMLTree(const cDigeoTestOutput &);

void  BinaryDumpInFile(ELISE_fp &,const cDigeoTestOutput &);

void  BinaryUnDumpFromFile(cDigeoTestOutput &,ELISE_fp &);

std::string  Mangling( cDigeoTestOutput *);

class cSectionTest
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionTest & anObj,cElXMLTree * aTree);


        cTplValGesInit< bool > & VerifExtrema();
        const cTplValGesInit< bool > & VerifExtrema()const ;

        cTplValGesInit< bool > & OutputGaussians();
        const cTplValGesInit< bool > & OutputGaussians()const ;

        cTplValGesInit< std::string > & OutputGaussiansDirectory();
        const cTplValGesInit< std::string > & OutputGaussiansDirectory()const ;

        cTplValGesInit< bool > & OutputTiles();
        const cTplValGesInit< bool > & OutputTiles()const ;

        cTplValGesInit< std::string > & OutputTilesDirectory();
        const cTplValGesInit< std::string > & OutputTilesDirectory()const ;

        cTplValGesInit< bool > & OutputGradients();
        const cTplValGesInit< bool > & OutputGradients()const ;

        cTplValGesInit< std::string > & OutputGradientsNormDirectory();
        const cTplValGesInit< std::string > & OutputGradientsNormDirectory()const ;

        cTplValGesInit< std::string > & OutputGradientsAngleDirectory();
        const cTplValGesInit< std::string > & OutputGradientsAngleDirectory()const ;

        cTplValGesInit< bool > & MergeTiles();
        const cTplValGesInit< bool > & MergeTiles()const ;

        cTplValGesInit< bool > & SuppressTiles();
        const cTplValGesInit< bool > & SuppressTiles()const ;

        cTplValGesInit< bool > & ForceGradientComputation();
        const cTplValGesInit< bool > & ForceGradientComputation()const ;

        cTplValGesInit< bool > & PlotPointsOnTiles();
        const cTplValGesInit< bool > & PlotPointsOnTiles()const ;

        cTplValGesInit< bool > & RawOutput();
        const cTplValGesInit< bool > & RawOutput()const ;

        cTplValGesInit< cDigeoTestOutput > & DigeoTestOutput();
        const cTplValGesInit< cDigeoTestOutput > & DigeoTestOutput()const ;
    private:
        cTplValGesInit< bool > mVerifExtrema;
        cTplValGesInit< cDigeoTestOutput > mDigeoTestOutput;
};
cElXMLTree * ToXMLTree(const cSectionTest &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionTest &);

void  BinaryUnDumpFromFile(cSectionTest &,ELISE_fp &);

std::string  Mangling( cSectionTest *);

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

void  BinaryDumpInFile(ELISE_fp &,const cSauvPyram &);

void  BinaryUnDumpFromFile(cSauvPyram &,ELISE_fp &);

std::string  Mangling( cSauvPyram *);

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

void  BinaryDumpInFile(ELISE_fp &,const cDigeoDecoupageCarac &);

void  BinaryUnDumpFromFile(cDigeoDecoupageCarac &,ELISE_fp &);

std::string  Mangling( cDigeoDecoupageCarac *);

class cGenereCodeConvol
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cGenereCodeConvol & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & DirectoryCodeConvol();
        const cTplValGesInit< std::string > & DirectoryCodeConvol()const ;

        cTplValGesInit< std::string > & FileBaseCodeConvol();
        const cTplValGesInit< std::string > & FileBaseCodeConvol()const ;
    private:
        cTplValGesInit< std::string > mDirectoryCodeConvol;
        cTplValGesInit< std::string > mFileBaseCodeConvol;
};
cElXMLTree * ToXMLTree(const cGenereCodeConvol &);

void  BinaryDumpInFile(ELISE_fp &,const cGenereCodeConvol &);

void  BinaryUnDumpFromFile(cGenereCodeConvol &,ELISE_fp &);

std::string  Mangling( cGenereCodeConvol *);

class cSectionWorkSpace
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionWorkSpace & anObj,cElXMLTree * aTree);


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

        cTplValGesInit< std::string > & DirectoryCodeConvol();
        const cTplValGesInit< std::string > & DirectoryCodeConvol()const ;

        cTplValGesInit< std::string > & FileBaseCodeConvol();
        const cTplValGesInit< std::string > & FileBaseCodeConvol()const ;

        cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol();
        const cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol()const ;

        cTplValGesInit< int > & ShowTimes();
        const cTplValGesInit< int > & ShowTimes()const ;

        cTplValGesInit< bool > & ShowConvolSpec();
        const cTplValGesInit< bool > & ShowConvolSpec()const ;

        cTplValGesInit< bool > & Verbose();
        const cTplValGesInit< bool > & Verbose()const ;
    private:
        cTplValGesInit< cSauvPyram > mSauvPyram;
        cTplValGesInit< cDigeoDecoupageCarac > mDigeoDecoupageCarac;
        cTplValGesInit< bool > mExigeCodeCompile;
        cTplValGesInit< cGenereCodeConvol > mGenereCodeConvol;
        cTplValGesInit< int > mShowTimes;
        cTplValGesInit< bool > mShowConvolSpec;
        cTplValGesInit< bool > mVerbose;
};
cElXMLTree * ToXMLTree(const cSectionWorkSpace &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionWorkSpace &);

void  BinaryUnDumpFromFile(cSectionWorkSpace &,ELISE_fp &);

std::string  Mangling( cSectionWorkSpace *);

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

        cTplValGesInit< double > & ResolInit();
        const cTplValGesInit< double > & ResolInit()const ;

        cTplValGesInit< double > & NbOctetLimitLoadImageOnce();
        const cTplValGesInit< double > & NbOctetLimitLoadImageOnce()const ;

        cImageDigeo & ImageDigeo();
        const cImageDigeo & ImageDigeo()const ;

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

        cTplValGesInit< bool > & SampledConvolutionKernels();
        const cTplValGesInit< bool > & SampledConvolutionKernels()const ;

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

        cDigeoSectionImages & DigeoSectionImages();
        const cDigeoSectionImages & DigeoSectionImages()const ;

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

        cTplValGesInit< ePointRefinement > & RefinementMethod();
        const cTplValGesInit< ePointRefinement > & RefinementMethod()const ;

        cTplValGesInit< cSiftCarac > & SiftCarac();
        const cTplValGesInit< cSiftCarac > & SiftCarac()const ;

        cSectionCaracImages & SectionCaracImages();
        const cSectionCaracImages & SectionCaracImages()const ;

        cTplValGesInit< double > & AutoAnnMinDist();
        const cTplValGesInit< double > & AutoAnnMinDist()const ;

        cTplValGesInit< bool > & VerifExtrema();
        const cTplValGesInit< bool > & VerifExtrema()const ;

        cTplValGesInit< bool > & OutputGaussians();
        const cTplValGesInit< bool > & OutputGaussians()const ;

        cTplValGesInit< std::string > & OutputGaussiansDirectory();
        const cTplValGesInit< std::string > & OutputGaussiansDirectory()const ;

        cTplValGesInit< bool > & OutputTiles();
        const cTplValGesInit< bool > & OutputTiles()const ;

        cTplValGesInit< std::string > & OutputTilesDirectory();
        const cTplValGesInit< std::string > & OutputTilesDirectory()const ;

        cTplValGesInit< bool > & OutputGradients();
        const cTplValGesInit< bool > & OutputGradients()const ;

        cTplValGesInit< std::string > & OutputGradientsNormDirectory();
        const cTplValGesInit< std::string > & OutputGradientsNormDirectory()const ;

        cTplValGesInit< std::string > & OutputGradientsAngleDirectory();
        const cTplValGesInit< std::string > & OutputGradientsAngleDirectory()const ;

        cTplValGesInit< bool > & MergeTiles();
        const cTplValGesInit< bool > & MergeTiles()const ;

        cTplValGesInit< bool > & SuppressTiles();
        const cTplValGesInit< bool > & SuppressTiles()const ;

        cTplValGesInit< bool > & ForceGradientComputation();
        const cTplValGesInit< bool > & ForceGradientComputation()const ;

        cTplValGesInit< bool > & PlotPointsOnTiles();
        const cTplValGesInit< bool > & PlotPointsOnTiles()const ;

        cTplValGesInit< bool > & RawOutput();
        const cTplValGesInit< bool > & RawOutput()const ;

        cTplValGesInit< cDigeoTestOutput > & DigeoTestOutput();
        const cTplValGesInit< cDigeoTestOutput > & DigeoTestOutput()const ;

        cTplValGesInit< cSectionTest > & SectionTest();
        const cTplValGesInit< cSectionTest > & SectionTest()const ;

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

        cTplValGesInit< std::string > & DirectoryCodeConvol();
        const cTplValGesInit< std::string > & DirectoryCodeConvol()const ;

        cTplValGesInit< std::string > & FileBaseCodeConvol();
        const cTplValGesInit< std::string > & FileBaseCodeConvol()const ;

        cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol();
        const cTplValGesInit< cGenereCodeConvol > & GenereCodeConvol()const ;

        cTplValGesInit< int > & ShowTimes();
        const cTplValGesInit< int > & ShowTimes()const ;

        cTplValGesInit< bool > & ShowConvolSpec();
        const cTplValGesInit< bool > & ShowConvolSpec()const ;

        cTplValGesInit< bool > & Verbose();
        const cTplValGesInit< bool > & Verbose()const ;

        cSectionWorkSpace & SectionWorkSpace();
        const cSectionWorkSpace & SectionWorkSpace()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cDigeoSectionImages mDigeoSectionImages;
        cSectionCaracImages mSectionCaracImages;
        cTplValGesInit< double > mAutoAnnMinDist;
        cTplValGesInit< cSectionTest > mSectionTest;
        cSectionWorkSpace mSectionWorkSpace;
};
cElXMLTree * ToXMLTree(const cParamDigeo &);

void  BinaryDumpInFile(ELISE_fp &,const cParamDigeo &);

void  BinaryUnDumpFromFile(cParamDigeo &,ELISE_fp &);

std::string  Mangling( cParamDigeo *);

/******************************************************/
/******************************************************/
/******************************************************/
#endif // Define_NotDigeo
