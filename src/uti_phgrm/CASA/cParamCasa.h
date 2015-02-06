#ifndef Define_NotCasa
#define Define_NotCasa
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

        cTplValGesInit< double > & DistSep();
        const cTplValGesInit< double > & DistSep()const ;

        cTplValGesInit< double > & DistZone();
        const cTplValGesInit< double > & DistZone()const ;

        cTplValGesInit< Pt2di > & SzW();
        const cTplValGesInit< Pt2di > & SzW()const ;
    private:
        std::list< cNuageByImage > mNuageByImage;
        cTplValGesInit< double > mDistSep;
        cTplValGesInit< double > mDistZone;
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

        cTplValGesInit< int > & NbRansac();
        const cTplValGesInit< int > & NbRansac()const ;

        cTplValGesInit< std::string > & OriPts();
        const cTplValGesInit< std::string > & OriPts()const ;

        cTplValGesInit< std::string > & PtsSurf();
        const cTplValGesInit< std::string > & PtsSurf()const ;
    private:
        eTypeSurfaceAnalytique mTypeSurf;
        cTplValGesInit< int > mNbRansac;
        cTplValGesInit< std::string > mOriPts;
        cTplValGesInit< std::string > mPtsSurf;
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

        cTplValGesInit< double > & DistSep();
        const cTplValGesInit< double > & DistSep()const ;

        cTplValGesInit< double > & DistZone();
        const cTplValGesInit< double > & DistZone()const ;

        cTplValGesInit< Pt2di > & SzW();
        const cTplValGesInit< Pt2di > & SzW()const ;

        cSectionLoadNuage & SectionLoadNuage();
        const cSectionLoadNuage & SectionLoadNuage()const ;

        eTypeSurfaceAnalytique & TypeSurf();
        const eTypeSurfaceAnalytique & TypeSurf()const ;

        cTplValGesInit< int > & NbRansac();
        const cTplValGesInit< int > & NbRansac()const ;

        cTplValGesInit< std::string > & OriPts();
        const cTplValGesInit< std::string > & OriPts()const ;

        cTplValGesInit< std::string > & PtsSurf();
        const cTplValGesInit< std::string > & PtsSurf()const ;

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
class cCasaEtapeCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCasaEtapeCompensation & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & NbIter();
        const cTplValGesInit< int > & NbIter()const ;

        cTplValGesInit< std::string > & Export();
        const cTplValGesInit< std::string > & Export()const ;
    private:
        cTplValGesInit< int > mNbIter;
        cTplValGesInit< std::string > mExport;
};
cElXMLTree * ToXMLTree(const cCasaEtapeCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cCasaEtapeCompensation &);

void  BinaryUnDumpFromFile(cCasaEtapeCompensation &,ELISE_fp &);

std::string  Mangling( cCasaEtapeCompensation *);

class cCasaSectionCompensation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCasaSectionCompensation & anObj,cElXMLTree * aTree);


        std::list< cCasaEtapeCompensation > & CasaEtapeCompensation();
        const std::list< cCasaEtapeCompensation > & CasaEtapeCompensation()const ;

        cTplValGesInit< double > & PercCoherenceOrientation();
        const cTplValGesInit< double > & PercCoherenceOrientation()const ;
    private:
        std::list< cCasaEtapeCompensation > mCasaEtapeCompensation;
        cTplValGesInit< double > mPercCoherenceOrientation;
};
cElXMLTree * ToXMLTree(const cCasaSectionCompensation &);

void  BinaryDumpInFile(ELISE_fp &,const cCasaSectionCompensation &);

void  BinaryUnDumpFromFile(cCasaSectionCompensation &,ELISE_fp &);

std::string  Mangling( cCasaSectionCompensation *);

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

        std::list< cCasaEtapeCompensation > & CasaEtapeCompensation();
        const std::list< cCasaEtapeCompensation > & CasaEtapeCompensation()const ;

        cTplValGesInit< double > & PercCoherenceOrientation();
        const cTplValGesInit< double > & PercCoherenceOrientation()const ;

        cCasaSectionCompensation & CasaSectionCompensation();
        const cCasaSectionCompensation & CasaSectionCompensation()const ;

        std::string & DirectoryChantier();
        const std::string & DirectoryChantier()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        std::list< cSectionInitModele > mSectionInitModele;
        cCasaSectionCompensation mCasaSectionCompensation;
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
