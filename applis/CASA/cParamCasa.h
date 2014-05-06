#ifndef Define_NotCasa
#define Define_NotCasa
#include "StdAfx.h"
// NOMORE ...
class cNuageByImage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cNuageByImage & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & NameMasqSup();
        const cTplValGesInit< std::string > & NameMasqSup()const ;

        std::string & NameXMLNuage();
        const std::string & NameXMLNuage()const ;
    private:
        cTplValGesInit< std::string > mNameMasqSup;
        std::string mNameXMLNuage;
};
cElXMLTree * ToXMLTree(const cNuageByImage &);

void  BinaryDumpInFile(ELISE_fp &,const cNuageByImage &);

void  BinaryUnDumpFromFile(cNuageByImage &,ELISE_fp &);

std::string  Mangling( cNuageByImage *);

class cSectionLoadNuage
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionLoadNuage & anObj,cElXMLTree * aTree);


        std::list< cNuageByImage > & NuageByImage();
        const std::list< cNuageByImage > & NuageByImage()const ;

        double & DistSep();
        const double & DistSep()const ;

        double & DistZone();
        const double & DistZone()const ;

        cTplValGesInit< Pt2di > & SzW();
        const cTplValGesInit< Pt2di > & SzW()const ;
    private:
        std::list< cNuageByImage > mNuageByImage;
        double mDistSep;
        double mDistZone;
        cTplValGesInit< Pt2di > mSzW;
};
cElXMLTree * ToXMLTree(const cSectionLoadNuage &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionLoadNuage &);

void  BinaryUnDumpFromFile(cSectionLoadNuage &,ELISE_fp &);

std::string  Mangling( cSectionLoadNuage *);

class cSectionEstimSurf
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionEstimSurf & anObj,cElXMLTree * aTree);


        eTypeSurfaceAnalytique & TypeSurf();
        const eTypeSurfaceAnalytique & TypeSurf()const ;

        int & NbRansac();
        const int & NbRansac()const ;
    private:
        eTypeSurfaceAnalytique mTypeSurf;
        int mNbRansac;
};
cElXMLTree * ToXMLTree(const cSectionEstimSurf &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionEstimSurf &);

void  BinaryUnDumpFromFile(cSectionEstimSurf &,ELISE_fp &);

std::string  Mangling( cSectionEstimSurf *);

class cSectionInitModele
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionInitModele & anObj,cElXMLTree * aTree);


        std::string & Name();
        const std::string & Name()const ;

        std::list< cNuageByImage > & NuageByImage();
        const std::list< cNuageByImage > & NuageByImage()const ;

        double & DistSep();
        const double & DistSep()const ;

        double & DistZone();
        const double & DistZone()const ;

        cTplValGesInit< Pt2di > & SzW();
        const cTplValGesInit< Pt2di > & SzW()const ;

        cSectionLoadNuage & SectionLoadNuage();
        const cSectionLoadNuage & SectionLoadNuage()const ;

        eTypeSurfaceAnalytique & TypeSurf();
        const eTypeSurfaceAnalytique & TypeSurf()const ;

        int & NbRansac();
        const int & NbRansac()const ;

        cSectionEstimSurf & SectionEstimSurf();
        const cSectionEstimSurf & SectionEstimSurf()const ;
    private:
        std::string mName;
        cSectionLoadNuage mSectionLoadNuage;
        cSectionEstimSurf mSectionEstimSurf;
};
cElXMLTree * ToXMLTree(const cSectionInitModele &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionInitModele &);

void  BinaryUnDumpFromFile(cSectionInitModele &,ELISE_fp &);

std::string  Mangling( cSectionInitModele *);

/******************************************************/
/******************************************************/
/******************************************************/
class cEtapeCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cEtapeCompensation & anObj,cElXMLTree * aTree);


        std::list< double > & Sigma();
        const std::list< double > & Sigma()const ;

        cTplValGesInit< int > & NbIter();
        const cTplValGesInit< int > & NbIter()const ;

        cTplValGesInit< std::string > & Export();
        const cTplValGesInit< std::string > & Export()const ;
    private:
        std::list< double > mSigma;
        cTplValGesInit< int > mNbIter;
        cTplValGesInit< std::string > mExport;
};
cElXMLTree * ToXMLTree(const cEtapeCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cEtapeCompensation &);

void  BinaryUnDumpFromFile(cEtapeCompensation &,ELISE_fp &);

std::string  Mangling( cEtapeCompensation *);

class cSectionCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionCompensation & anObj,cElXMLTree * aTree);


        std::list< cEtapeCompensation > & EtapeCompensation();
        const std::list< cEtapeCompensation > & EtapeCompensation()const ;

        cTplValGesInit< double > & CoherenceOrientation();
        const cTplValGesInit< double > & CoherenceOrientation()const ;
    private:
        std::list< cEtapeCompensation > mEtapeCompensation;
        cTplValGesInit< double > mCoherenceOrientation;
};
cElXMLTree * ToXMLTree(const cSectionCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cSectionCompensation &);

void  BinaryUnDumpFromFile(cSectionCompensation &,ELISE_fp &);

std::string  Mangling( cSectionCompensation *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamCasa
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamCasa & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        std::list< cSectionInitModele > & SectionInitModele();
        const std::list< cSectionInitModele > & SectionInitModele()const ;

        std::list< cEtapeCompensation > & EtapeCompensation();
        const std::list< cEtapeCompensation > & EtapeCompensation()const ;

        cTplValGesInit< double > & CoherenceOrientation();
        const cTplValGesInit< double > & CoherenceOrientation()const ;

        cSectionCompensation & SectionCompensation();
        const cSectionCompensation & SectionCompensation()const ;

        std::string & DirectoryChantier();
        const std::string & DirectoryChantier()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        std::list< cSectionInitModele > mSectionInitModele;
        cSectionCompensation mSectionCompensation;
        std::string mDirectoryChantier;
};
cElXMLTree * ToXMLTree(const cParamCasa &);

void  BinaryDumpInFile(ELISE_fp &,const cParamCasa &);

void  BinaryUnDumpFromFile(cParamCasa &,ELISE_fp &);

std::string  Mangling( cParamCasa *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotCasa
