class caffichImg
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(caffichImg & anObj,cElXMLTree * aTree);


        int & image();
        const int & image()const ;

        std::string & fichier();
        const std::string & fichier()const ;
    private:
        int mimage;
        std::string mfichier;
};
cElXMLTree * ToXMLTree(const caffichImg &);

void  BinaryDumpInFile(ELISE_fp &,const caffichImg &);

void  BinaryUnDumpFromFile(caffichImg &,ELISE_fp &);

std::string  Mangling( caffichImg *);

/******************************************************/
/******************************************************/
/******************************************************/
class caffichPaire
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(caffichPaire & anObj,cElXMLTree * aTree);


        int & image1();
        const int & image1()const ;

        std::string & fichier1();
        const std::string & fichier1()const ;

        int & image2();
        const int & image2()const ;

        std::string & fichier2();
        const std::string & fichier2()const ;

        int & liste();
        const int & liste()const ;

        cTplValGesInit< bool > & trait();
        const cTplValGesInit< bool > & trait()const ;
    private:
        int mimage1;
        std::string mfichier1;
        int mimage2;
        std::string mfichier2;
        int mliste;
        cTplValGesInit< bool > mtrait;
};
cElXMLTree * ToXMLTree(const caffichPaire &);

void  BinaryDumpInFile(ELISE_fp &,const caffichPaire &);

void  BinaryUnDumpFromFile(caffichPaire &,ELISE_fp &);

std::string  Mangling( caffichPaire *);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamFusionSift
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamFusionSift & anObj,cElXMLTree * aTree);


        string & dossier();
        const string & dossier()const ;

        string & dossierImg();
        const string & dossierImg()const ;

        cTplValGesInit< std::string > & extensionSortie();
        const cTplValGesInit< std::string > & extensionSortie()const ;

        cTplValGesInit< int > & firstfichier();
        const cTplValGesInit< int > & firstfichier()const ;

        cTplValGesInit< int > & lastfichier();
        const cTplValGesInit< int > & lastfichier()const ;

        cTplValGesInit< int > & SzMin();
        const cTplValGesInit< int > & SzMin()const ;

        cTplValGesInit< int > & NbObjMax();
        const cTplValGesInit< int > & NbObjMax()const ;

        cTplValGesInit< Box2dr > & box();
        const cTplValGesInit< Box2dr > & box()const ;

        cTplValGesInit< REAL > & distIsol();
        const cTplValGesInit< REAL > & distIsol()const ;

        cTplValGesInit< int > & ptppi();
        const cTplValGesInit< int > & ptppi()const ;

        cTplValGesInit< double > & mindistalign();
        const cTplValGesInit< double > & mindistalign()const ;

        cTplValGesInit< bool > & filtre1();
        const cTplValGesInit< bool > & filtre1()const ;

        cTplValGesInit< bool > & filtre2();
        const cTplValGesInit< bool > & filtre2()const ;

        cTplValGesInit< bool > & filtre3();
        const cTplValGesInit< bool > & filtre3()const ;

        cTplValGesInit< REAL > & distIsol2();
        const cTplValGesInit< REAL > & distIsol2()const ;

        cTplValGesInit< bool > & rapide();
        const cTplValGesInit< bool > & rapide()const ;

        cTplValGesInit< double > & aDistInitVois();
        const cTplValGesInit< double > & aDistInitVois()const ;

        cTplValGesInit< double > & aFact();
        const cTplValGesInit< double > & aFact()const ;

        cTplValGesInit< int > & aNbMax();
        const cTplValGesInit< int > & aNbMax()const ;

        cTplValGesInit< int > & aNb1();
        const cTplValGesInit< int > & aNb1()const ;

        cTplValGesInit< int > & aNb2();
        const cTplValGesInit< int > & aNb2()const ;

        cTplValGesInit< double > & seuilCoherenceVois();
        const cTplValGesInit< double > & seuilCoherenceVois()const ;

        cTplValGesInit< double > & seuilCoherenceCarre();
        const cTplValGesInit< double > & seuilCoherenceCarre()const ;

        cTplValGesInit< int > & aNb();
        const cTplValGesInit< int > & aNb()const ;

        cTplValGesInit< int > & nbEssais();
        const cTplValGesInit< int > & nbEssais()const ;

        std::list< caffichImg > & affichImg();
        const std::list< caffichImg > & affichImg()const ;

        std::list< caffichPaire > & affichPaire();
        const std::list< caffichPaire > & affichPaire()const ;
    private:
        string mdossier;
        string mdossierImg;
        cTplValGesInit< std::string > mextensionSortie;
        cTplValGesInit< int > mfirstfichier;
        cTplValGesInit< int > mlastfichier;
        cTplValGesInit< int > mSzMin;
        cTplValGesInit< int > mNbObjMax;
        cTplValGesInit< Box2dr > mbox;
        cTplValGesInit< REAL > mdistIsol;
        cTplValGesInit< int > mptppi;
        cTplValGesInit< double > mmindistalign;
        cTplValGesInit< bool > mfiltre1;
        cTplValGesInit< bool > mfiltre2;
        cTplValGesInit< bool > mfiltre3;
        cTplValGesInit< REAL > mdistIsol2;
        cTplValGesInit< bool > mrapide;
        cTplValGesInit< double > maDistInitVois;
        cTplValGesInit< double > maFact;
        cTplValGesInit< int > maNbMax;
        cTplValGesInit< int > maNb1;
        cTplValGesInit< int > maNb2;
        cTplValGesInit< double > mseuilCoherenceVois;
        cTplValGesInit< double > mseuilCoherenceCarre;
        cTplValGesInit< int > maNb;
        cTplValGesInit< int > mnbEssais;
        std::list< caffichImg > maffichImg;
        std::list< caffichPaire > maffichPaire;
};
cElXMLTree * ToXMLTree(const cParamFusionSift &);

void  BinaryDumpInFile(ELISE_fp &,const cParamFusionSift &);

void  BinaryUnDumpFromFile(cParamFusionSift &,ELISE_fp &);

std::string  Mangling( cParamFusionSift *);

/******************************************************/
/******************************************************/
/******************************************************/
