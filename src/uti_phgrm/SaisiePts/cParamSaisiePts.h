#ifndef Define_NotSaisiPts
#define Define_NotSaisiPts
// #include "general/all.h"
// #include "private/all.h"
//
typedef enum
{
  eNSM_GeoCube,
  eNSM_Plaquette,
  eNSM_Pts,
  eNSM_MaxLoc,
  eNSM_MinLoc,
  eNSM_NonValue
} eTypePts;
void xml_init(eTypePts & aVal,cElXMLTree * aTree);
std::string  eToString(const eTypePts & aVal);

eTypePts  Str2eTypePts(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePts & anObj);

typedef enum
{
  eEPI_NonSaisi,
  eEPI_Refute,
  eEPI_Douteux,
  eEPI_Valide,
  eEPI_NonValue,
  eEPI_Disparu,
  eEPI_Highlight
} eEtatPointeImage;
void xml_init(eEtatPointeImage & aVal,cElXMLTree * aTree);
std::string  eToString(const eEtatPointeImage & aVal);

eEtatPointeImage  Str2eEtatPointeImage(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eEtatPointeImage & anObj);

class cContenuPt
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cContenuPt & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & None();
        const cTplValGesInit< std::string > & None()const ;
    private:
        cTplValGesInit< std::string > mNone;
};
cElXMLTree * ToXMLTree(const cContenuPt &);

class cPointGlob
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cPointGlob & anObj,cElXMLTree * aTree);


        eTypePts & Type();
        const eTypePts & Type()const ;

        std::string & Name();
        const std::string & Name()const ;

        cTplValGesInit< Pt3dr > & P3D();
        const cTplValGesInit< Pt3dr > & P3D()const ;

        cTplValGesInit< bool > & Mes3DExportable();
        const cTplValGesInit< bool > & Mes3DExportable()const ;

        cTplValGesInit< Pt3dr > & Incert();
        const cTplValGesInit< Pt3dr > & Incert()const ;

        cTplValGesInit< double > & LargeurFlou();
        const cTplValGesInit< double > & LargeurFlou()const ;

        cTplValGesInit< std::string > & None();
        const cTplValGesInit< std::string > & None()const ;

        cTplValGesInit< cContenuPt > & ContenuPt();
        const cTplValGesInit< cContenuPt > & ContenuPt()const ;

        cTplValGesInit< int > & NumAuto();
        const cTplValGesInit< int > & NumAuto()const ;

        cTplValGesInit< Pt3dr > & PS1();
        const cTplValGesInit< Pt3dr > & PS1()const ;

        cTplValGesInit< Pt3dr > & PS2();
        const cTplValGesInit< Pt3dr > & PS2()const ;

        cTplValGesInit< double > & SzRech();
        const cTplValGesInit< double > & SzRech()const ;

        cTplValGesInit< bool > & Disparu();
        const cTplValGesInit< bool > & Disparu()const ;

        cTplValGesInit< bool > & FromDico();
        const cTplValGesInit< bool > & FromDico()const ;
    private:
        eTypePts mType;
        std::string mName;
        cTplValGesInit< Pt3dr > mP3D;
        cTplValGesInit< bool > mMes3DExportable;
        cTplValGesInit< Pt3dr > mIncert;
        cTplValGesInit< double > mLargeurFlou;
        cTplValGesInit< cContenuPt > mContenuPt;
        cTplValGesInit< int > mNumAuto;
        cTplValGesInit< Pt3dr > mPS1;
        cTplValGesInit< Pt3dr > mPS2;
        cTplValGesInit< double > mSzRech;
        cTplValGesInit< bool > mDisparu;
        cTplValGesInit< bool > mFromDico;
};
cElXMLTree * ToXMLTree(const cPointGlob &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetPointGlob
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetPointGlob & anObj,cElXMLTree * aTree);


        std::list< cPointGlob > & PointGlob();
        const std::list< cPointGlob > & PointGlob()const ;
    private:
        std::list< cPointGlob > mPointGlob;
};
cElXMLTree * ToXMLTree(const cSetPointGlob &);

/******************************************************/
/******************************************************/
/******************************************************/
class cOneSaisie
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cOneSaisie & anObj,cElXMLTree * aTree);


        eEtatPointeImage & Etat();
        const eEtatPointeImage & Etat()const ;

        std::string & NamePt();
        const std::string & NamePt()const ;

        Pt2dr & PtIm();
        const Pt2dr & PtIm()const ;
    private:
        eEtatPointeImage mEtat;
        std::string mNamePt;
        Pt2dr mPtIm;
};
cElXMLTree * ToXMLTree(const cOneSaisie &);

class cSaisiePointeIm
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSaisiePointeIm & anObj,cElXMLTree * aTree);


        std::string & NameIm();
        const std::string & NameIm()const ;

        std::list< cOneSaisie > & OneSaisie();
        const std::list< cOneSaisie > & OneSaisie()const ;
    private:
        std::string mNameIm;
        std::list< cOneSaisie > mOneSaisie;
};
cElXMLTree * ToXMLTree(const cSaisiePointeIm &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSetOfSaisiePointeIm
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSetOfSaisiePointeIm & anObj,cElXMLTree * aTree);


        std::list< cSaisiePointeIm > & SaisiePointeIm();
        const std::list< cSaisiePointeIm > & SaisiePointeIm()const ;
    private:
        std::list< cSaisiePointeIm > mSaisiePointeIm;
};
cElXMLTree * ToXMLTree(const cSetOfSaisiePointeIm &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionWindows
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionWindows & anObj,cElXMLTree * aTree);


        cTplValGesInit< Pt2di > & SzTotIm();
        const cTplValGesInit< Pt2di > & SzTotIm()const ;

        cTplValGesInit< Pt2di > & NbFenIm();
        const cTplValGesInit< Pt2di > & NbFenIm()const ;

        cTplValGesInit< Pt2di > & SzWZ();
        const cTplValGesInit< Pt2di > & SzWZ()const ;

        cTplValGesInit< bool > & ShowDet();
        const cTplValGesInit< bool > & ShowDet()const ;

        cTplValGesInit< bool > & RefInvis();
        const cTplValGesInit< bool > & RefInvis()const ;
    private:
        cTplValGesInit< Pt2di > mSzTotIm;
        cTplValGesInit< Pt2di > mNbFenIm;
        cTplValGesInit< Pt2di > mSzWZ;
        cTplValGesInit< bool > mShowDet;
        cTplValGesInit< bool > mRefInvis;
};
cElXMLTree * ToXMLTree(const cSectionWindows &);

/******************************************************/
/******************************************************/
/******************************************************/
class cImportFromDico
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cImportFromDico & anObj,cElXMLTree * aTree);


        eTypePts & TypePt();
        const eTypePts & TypePt()const ;

        std::string & File();
        const std::string & File()const ;

        cTplValGesInit< double > & LargeurFlou();
        const cTplValGesInit< double > & LargeurFlou()const ;
    private:
        eTypePts mTypePt;
        std::string mFile;
        cTplValGesInit< double > mLargeurFlou;
};
cElXMLTree * ToXMLTree(const cImportFromDico &);

class cSectionInOut
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionInOut & anObj,cElXMLTree * aTree);


        cTplValGesInit< std::string > & Prefix2Add2IdPt();
        const cTplValGesInit< std::string > & Prefix2Add2IdPt()const ;

        std::list< cImportFromDico > & ImportFromDico();
        const std::list< cImportFromDico > & ImportFromDico()const ;

        cTplValGesInit< bool > & FlouGlobEcras();
        const cTplValGesInit< bool > & FlouGlobEcras()const ;

        cTplValGesInit< bool > & TypeGlobEcras();
        const cTplValGesInit< bool > & TypeGlobEcras()const ;

        cTplValGesInit< std::string > & NamePointesImage();
        const cTplValGesInit< std::string > & NamePointesImage()const ;

        cTplValGesInit< std::string > & NamePointsGlobal();
        const cTplValGesInit< std::string > & NamePointsGlobal()const ;

        cTplValGesInit< std::string > & ExportPointeImage();
        const cTplValGesInit< std::string > & ExportPointeImage()const ;

        std::list< std::string > & FixedName();
        const std::list< std::string > & FixedName()const ;

        cTplValGesInit< std::string > & NameAuto();
        const cTplValGesInit< std::string > & NameAuto()const ;

        cTplValGesInit< bool > & EnterName();
        const cTplValGesInit< bool > & EnterName()const ;
    private:
        cTplValGesInit< std::string > mPrefix2Add2IdPt;
        std::list< cImportFromDico > mImportFromDico;
        cTplValGesInit< bool > mFlouGlobEcras;
        cTplValGesInit< bool > mTypeGlobEcras;
        cTplValGesInit< std::string > mNamePointesImage;
        cTplValGesInit< std::string > mNamePointsGlobal;
        cTplValGesInit< std::string > mExportPointeImage;
        std::list< std::string > mFixedName;
        cTplValGesInit< std::string > mNameAuto;
        cTplValGesInit< bool > mEnterName;
};
cElXMLTree * ToXMLTree(const cSectionInOut &);

/******************************************************/
/******************************************************/
/******************************************************/
class cSectionImages
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionImages & anObj,cElXMLTree * aTree);


        std::string & SetOfImages();
        const std::string & SetOfImages()const ;

        cTplValGesInit< bool > & ForceGray();
        const cTplValGesInit< bool > & ForceGray()const ;

        cTplValGesInit< std::string > & KeyAssocOri();
        const cTplValGesInit< std::string > & KeyAssocOri()const ;
    private:
        std::string mSetOfImages;
        cTplValGesInit< bool > mForceGray;
        cTplValGesInit< std::string > mKeyAssocOri;
};
cElXMLTree * ToXMLTree(const cSectionImages &);

/******************************************************/
/******************************************************/
/******************************************************/
class cProfEstimator
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cProfEstimator & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        cTplValGesInit< std::string > & ZMoyenInIma();
        const cTplValGesInit< std::string > & ZMoyenInIma()const ;
    private:
        cTplValGesInit< double > mZMoyen;
        cTplValGesInit< std::string > mZMoyenInIma;
};
cElXMLTree * ToXMLTree(const cProfEstimator &);

class cSectionTerrain
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cSectionTerrain & anObj,cElXMLTree * aTree);


        cTplValGesInit< double > & IntervPercProf();
        const cTplValGesInit< double > & IntervPercProf()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        cTplValGesInit< std::string > & ZMoyenInIma();
        const cTplValGesInit< std::string > & ZMoyenInIma()const ;

        cTplValGesInit< cProfEstimator > & ProfEstimator();
        const cTplValGesInit< cProfEstimator > & ProfEstimator()const ;
    private:
        cTplValGesInit< double > mIntervPercProf;
        cTplValGesInit< cProfEstimator > mProfEstimator;
};
cElXMLTree * ToXMLTree(const cSectionTerrain &);

/******************************************************/
/******************************************************/
/******************************************************/
class cParamSaisiePts
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cParamSaisiePts & anObj,cElXMLTree * aTree);


        cTplValGesInit< cChantierDescripteur > & DicoLoc();
        const cTplValGesInit< cChantierDescripteur > & DicoLoc()const ;

        cTplValGesInit< Pt2di > & SzTotIm();
        const cTplValGesInit< Pt2di > & SzTotIm()const ;

        cTplValGesInit< Pt2di > & NbFenIm();
        const cTplValGesInit< Pt2di > & NbFenIm()const ;

        cTplValGesInit< Pt2di > & SzWZ();
        const cTplValGesInit< Pt2di > & SzWZ()const ;

        cTplValGesInit< bool > & ShowDet();
        const cTplValGesInit< bool > & ShowDet()const ;

        cTplValGesInit< bool > & RefInvis();
        const cTplValGesInit< bool > & RefInvis()const ;

        cSectionWindows & SectionWindows();
        const cSectionWindows & SectionWindows()const ;

        cTplValGesInit< std::string > & Prefix2Add2IdPt();
        const cTplValGesInit< std::string > & Prefix2Add2IdPt()const ;

        std::list< cImportFromDico > & ImportFromDico();
        const std::list< cImportFromDico > & ImportFromDico()const ;

        cTplValGesInit< bool > & FlouGlobEcras();
        const cTplValGesInit< bool > & FlouGlobEcras()const ;

        cTplValGesInit< bool > & TypeGlobEcras();
        const cTplValGesInit< bool > & TypeGlobEcras()const ;

        cTplValGesInit< std::string > & NamePointesImage();
        const cTplValGesInit< std::string > & NamePointesImage()const ;

        cTplValGesInit< std::string > & NamePointsGlobal();
        const cTplValGesInit< std::string > & NamePointsGlobal()const ;

        cTplValGesInit< std::string > & ExportPointeImage();
        const cTplValGesInit< std::string > & ExportPointeImage()const ;

        std::list< std::string > & FixedName();
        const std::list< std::string > & FixedName()const ;

        cTplValGesInit< std::string > & NameAuto();
        const cTplValGesInit< std::string > & NameAuto()const ;

        cTplValGesInit< bool > & EnterName();
        const cTplValGesInit< bool > & EnterName()const ;

        cSectionInOut & SectionInOut();
        const cSectionInOut & SectionInOut()const ;

        std::string & SetOfImages();
        const std::string & SetOfImages()const ;

        cTplValGesInit< bool > & ForceGray();
        const cTplValGesInit< bool > & ForceGray()const ;

        cTplValGesInit< std::string > & KeyAssocOri();
        const cTplValGesInit< std::string > & KeyAssocOri()const ;

        cSectionImages & SectionImages();
        const cSectionImages & SectionImages()const ;

        cTplValGesInit< double > & IntervPercProf();
        const cTplValGesInit< double > & IntervPercProf()const ;

        cTplValGesInit< double > & ZMoyen();
        const cTplValGesInit< double > & ZMoyen()const ;

        cTplValGesInit< std::string > & ZMoyenInIma();
        const cTplValGesInit< std::string > & ZMoyenInIma()const ;

        cTplValGesInit< cProfEstimator > & ProfEstimator();
        const cTplValGesInit< cProfEstimator > & ProfEstimator()const ;

        cSectionTerrain & SectionTerrain();
        const cSectionTerrain & SectionTerrain()const ;

        std::string & DirectoryChantier();
        const std::string & DirectoryChantier()const ;
    private:
        cTplValGesInit< cChantierDescripteur > mDicoLoc;
        cSectionWindows mSectionWindows;
        cSectionInOut mSectionInOut;
        cSectionImages mSectionImages;
        cSectionTerrain mSectionTerrain;
        std::string mDirectoryChantier;
};
cElXMLTree * ToXMLTree(const cParamSaisiePts &);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_NotSaisiPts
