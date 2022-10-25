/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/


//  Modifieur speciaux des XML
//
//      @$#   cles sur la valeur pour envisager qu'il puissent etres speciaux
//
//       ExitOnBrkp="@$#"    -> Modifie le comportement en cas d'erreur
//       Subst="@$#1"        -> Dans le sous arbres, il faut subsituer les symbole
//       NameDecl="@$#1"     -> Dans les attribut qui suivent, les declaration sont
//                              des definitions de symboles
//
//       DirXmlSpec="@$#..."  -> obsolete
//
//
//      ${NikonPatternNumero}  -> a remplacer par la valeur dans le dictionnaire courant de NikonPatternNumero
//
//      Dans une declaration
//      NikonPatternNumero=@KKKK  -> indique qu'il ne faut modifier le dictionnaire que si
//                                 le symbole n'a pas encore de valeur
//
//
//  Ligne de commande :
//
//         aTag=machin  -> modifie le tag aTag, en lui donnant la valeur machin,
//                         assez restrictif sur la non ambiguite qu'il doit y avoir
//                         sur aTag (l'arbre formel doit permettre d'apporter la preuve que
//                         toto est au + unique)
//
//        @aTag=machin  -> se contente de verifier que l'arbre effectif est unique et modifie
//        %anAttrId=machin -> va modifier le tag dont l'attribut Id vaut anAttrId (doit existe et unique)
//        +anAttrId=machin -> va modifier le dictionnaire du  cInterfChantierNameManipulateur
//
//

/*

           cInterfChantierNameManipulateur

                     |                     \
                     |                      \

           cStdChantierMonoManipulateur     cStdChantierMultiManipulateur
                                            {
                                                  cStdChantierMultiManipulateur
                                              std::vector<cInterfChantierNameManipulateur *>  mVM; => Ce sont
                                                    en fait des cStdChantierMonoManipulateur
                                            }

*/

#ifndef _ELISE_XML_GEN_ALL_H
#define _ELISE_XML_GEN_ALL_H

//#include "StdAfx.h"
#include "general/CMake_defines.h"
#include "XML_GEN/ParamChantierPhotogram.h"
#include "XML_GEN/SuperposImage.h"
#include "XML_GEN/cInterfChantierNameManipulateur.h"

void  AddExtensionToSubdir(std::string & aName ,const std::string & aDirExt);

void UseRequirement(const std::string &,const cTplValGesInit<cBatchRequirement> &);


cMicMacConfiguration * MMC();
std::string MMDir();
std::string MMBin();
std::string current_program_fullname();   // mm3d's full name (absolute path + executable name)
std::string current_program_subcommand(); // mm3d's subcommand (Tapioca, Apero, etc.)
int MMNbProc();
bool MPD_MM(); // Est ce que c'est ma machine, afin de ne pas polluer les autres en phase de test !!!!
bool ERupnik_MM();

#if ELISE_QT
    string MMQtLibraryPath();
    void setQtLibraryPath( const string &i_path );
    void initQtLibraryPath();
#endif // ELISE_QT

inline bool isUsingSeparateDirectories();
extern const string temporarySubdirectory; // = "Tmp-MM-Dir/" (see src/photogram/ChantierNameAssoc.cpp)
void setInputDirectory( const std::string &i_directory );
bool isInputDirectorySet();
std::string MMInputDirectory(); // is the directory containing base pictures (must be set by setInputDirectory before use)
std::string MMTemporaryDirectory(); // equals MMUserEnvironment.OutputDirectory (or "./" if not set) + temporarySubdirectory 
std::string MMOutputDirectory(); // equals MMUserEnvironment.OutputDirectory (if set) or MMInputDirectory (if set) or ./
std::string MMLogDirectory(); // equals MMUserEnvironment.LogDirectory or ./ if not set

extern std::string MM3DStr;
extern const  std::string TheStringLastNuageMM;

//   Binaire a "l'ancienne"  MMDir() + std::string("bin" ELISE_STR_DIR  COMMANDE)
std::string MMBinFile(const std::string &);
//   Nouveau par mm3d   MMDir() + std::string("bin" ELISE_STR_DIR "mm3d"  COMMANDE)
std::string MM3dBinFile(const std::string &);
std::string MM3dBinFile_quotes(const std::string &);
//   MMDir() + std::string("include" ELISE_STR_DIR "XML_MicMac" ELISE_STR_DIR "Apero-Cloud.xml ")
std::string XML_MM_File(const std::string &);
std::string Basic_XML_MM_File(const std::string &);
std::string Specif_XML_MM_File(const std::string &);
std::string Basic_XML_User_File(const std::string & aName);
std::string XML_User_Or_MicMac(const std::string & aName);

const cMMUserEnvironment & MMUserEnv();

extern const  std::string BLANK;

class cMMByImNM;





GenIm::type_el Xml2EL(const eTypeNumerique & aType);
Tiff_Im::COMPR_TYPE Xml2EL(const eComprTiff & aType);

class cCompileCAPI;



bool IsActive(const cTplValGesInit<cCmdMappeur> & );
bool IsActive(const std::list<cCmdMappeur> & );



std::string StdNameGBOrient(const std::string & anOri,const std::string & aNameIm,bool AddMinus);
std::string StdNameCSOrient(const std::string & anOri,const std::string & aNameIm,bool AddMinus);


template <class Type> class  cResultSubstAndStdGetFile
{
    public :
        Type * mObj;
        cInterfChantierNameManipulateur * mICNM;
        std::string                       mDC;

cResultSubstAndStdGetFile
(
      int argc,char **argv,
      const std::string & aNameFileObj,
      const std::string & aNameFileSpecif,
      const std::string & aNameTagObj,
      const std::string & aNameTagType,
      const std::string & aNameTagDirectory,   // La directory
      const std::string & aNameTagFDC , // Eventuellement un File Chantier Descripteur,
      const char *  aNameSauv = 0
)
{
   cElXMLTree aTree(aNameFileObj,0,false);



   mDC = "ThisDir";
   cElXMLTree * aTrDC  = aTree.GetOneOrZero(aNameTagDirectory);
   if (aTrDC)
      mDC = aTrDC->Contenu();

   if (mDC=="ThisDir")
   {
        std::string aNF;
        SplitDirAndFile(mDC,aNF,aNameFileObj);
   }

    #if (ELISE_windows)
        replace( mDC.begin(), mDC.end(), '\\', '/' );
    #endif

   {
      std::string aDef;
      std::string aDC = GetValLC(argc,argv,aNameTagDirectory,aDef);
      if (aDC!=aDef)
      {
         mDC = aDC;
      }
   }

   cElXMLTree * aTrFDC = aTree.GetOneOrZero(aNameTagFDC);
   cTplValGesInit<std::string>  aTplFCND;
   if (aTrFDC != 0)
        aTplFCND.SetVal(aTrFDC->Contenu());

// ==========

   mICNM = cInterfChantierNameManipulateur::StdAlloc
                (
                    argc,argv,
                    mDC,
                    aTplFCND,
                    0,
                    true
                );

   mObj  = new Type
               (
                  StdGetObjFromFile_WithLC<Type>
                  (
                     argc,argv,
                     aNameFileObj,
                     aNameFileSpecif,
                     aNameTagObj,
                     aNameTagType,
                     false,
                     &(mICNM->ArgTree()),
                    aNameSauv
                  )
               );
   mICNM->CD_Add(mObj->DicoLoc().PtrVal());

   mICNM->MkDataBase();

#if (ELISE_windows)
   std::string name = mDC;
   name.resize( name.size()-1 );
   ELISE_fp::AssertIsDirectory(name);
#else
   ELISE_fp::AssertIsDirectory(mDC);
#endif
}

};




//    soit A l'aphabet, A* l'ensemble des mots sur A,
//    A*+ l'ensemble des Nuplets (N!=0,  N variable) sur A*
//
//   cInterfNameCalculator est une classe pour representer l'interface des
//   objets "fonction de A*+ dans A*+"
//
//   Direct renvoie l'image d'un  N - uplet de mots; s'il n'a pas d'image
//   (il s'agit d'une fonction) le mot Def est renvoe
//
//   La fonction inverse (pas toujours utile) renvoie par defaut le N-uplet indefini
//
//
//   cInterfChantierNC est une application de A x A*+  dans  A*+  (un parametre
//   supplementaire, la "cle" permet de specifier parmi les associations possible)
//
//
//    cInterfChantierSetNC est  une application de A dans A*+, on peut acceder
//    a des ensemble de mot a partir d'une cle
//
//
//
//   cInterfChantierNameManipulateur est l'interface decrivant les manipulation de noms
//   utiles pour decrire un chantier, il permet d'acceder a des ensemble de mots
//   et a des transformation de mots a partir de cles.
//

class cInterfNameCalculator;
class cNI_AutomNC;
class cInv_AutomNC;
class cMultiNC;
class cInterfChantierNC;
class cDicoChantierNC;

class cInterfNameCalculator
{
    public :
        typedef  std::vector<std::string>   tNuplet;
        virtual ~cInterfNameCalculator();

    static  const tNuplet & NotDef();  // size 0; Valeur renoyee en cas d'echec
    static bool IsDefined(const tNuplet &);

    static cInterfNameCalculator * StdCalcFromXML
                                   (
                                            cInterfChantierNameManipulateur *,
                                            const cAssocNameToName &
                                       );
    static cMultiNC * StdCalcFromXML
                      (
                              const cKeyedNamesAssociations &,
                              cInterfChantierNameManipulateur *,
                              const std::string& aSubDir,
                              bool               aSubDirRec,
                              const std::list<cAssocNameToName> &
                          );

        virtual tNuplet Direct(const tNuplet &) = 0;
        virtual tNuplet Inverse(const tNuplet &);

        virtual cInterfChantierNameManipulateur *    ICNM();
        virtual cKeyedNamesAssociations * KNA();

    protected  :
        cInterfChantierNameManipulateur * mICNM;
        cInterfNameCalculator(cInterfChantierNameManipulateur * aGlob);

    private  :
        static  const tNuplet theNotDef; // genre "Wuyiu5$^%µ"
};

class cInterfChantierNC
{
    public :
        typedef std::string tKey;
        typedef cInterfNameCalculator::tNuplet tNuplet;

        tNuplet  DefinedDirect(const tKey &,const tNuplet&) ;
        tNuplet  DefinedInverse(const tKey &,const tNuplet&);
        virtual tNuplet  Direct(const tKey &,const tNuplet&) = 0;
        virtual tNuplet  Inverse(const tKey &,const tNuplet&) = 0;
    virtual bool AssocHasKey(const tKey & aKey) const = 0;

        cInterfChantierNC();
    virtual ~cInterfChantierNC();
    static cDicoChantierNC *  StdCalcFromXML
               (
                   cInterfChantierNameManipulateur * aICNM,
                   const std::list<cKeyedNamesAssociations> &
               );
    protected :
        void VerifSol(const tNuplet & aSol,const tKey &,const tNuplet&) ;
};


class cElemComFRSE
{
    public :
      cElemComFRSE(int aNb,const std::string & aN);
      int Nb() const;
      const std::string & Name() const;
    private :
      int mNb;
      std::string  mName;
};

class cComputeFiltreRelSsEch
{
    public :
       cComputeFiltreRelSsEch
       (
           cInterfChantierNameManipulateur &aICNM,
           const cFiltreByRelSsEch &
       );
       bool OkCple(const std::string &,const std::string &) const;
   private :
       cInterfChantierNameManipulateur &    mICNM ;
       const cFiltreByRelSsEch &            mFRSE;
       std::map<std::string,std::vector<cElemComFRSE> > mElems;
};


class cComputeFiltreRelOr
{
    public :
        cComputeFiltreRelOr
        (
              const cTplValGesInit<cFiltreDeRelationOrient> & aTplF,
              cInterfChantierNameManipulateur &               anICNM
        ) ;
        ~cComputeFiltreRelOr();

        bool OK_CFOR(const std::string & aNA,const std::string & aNB) const;
    private :
        cComputeFiltreRelOr(const cComputeFiltreRelOr&); // N.I.
        bool OKEquiv(const std::string & aNA,const std::string & aNB) const;
        bool OKEmprise(const std::string & aNA,const std::string & aNB) const;
        bool OKMatrix(const std::string & aNA,const std::string & aNB) const;
        bool OKSsEch(const std::string & aNA,const std::string & aNB) const;
        const cFiltreDeRelationOrient *    mPF; // =  aTplF.PtrVal();
        cInterfChantierNameManipulateur &  mICNM;
        cComputeFiltreRelSsEch *           mFSsEch;
};

class cStdChantierRel
{
    public :
        cStdChantierRel
    (
           cInterfChantierNameManipulateur &,
       const cNameRelDescriptor &
        );


     std::vector<cCpleString> * GetRel();
    void Add(const cCpleString &);
        const cNameRelDescriptor &  NRD() const;
    private :
        bool AddAFile(std::string  aName,bool must_exist);
        void Compute();
        void ComputeFiltrageSpatial();
    void Add(const std::string& aPre,const cCpleString &,const std::string& aPost);

        void AddAllCpleKeySet
             (
                    const std::string & aKEY1,
                    const std::string & aKEY2,
                    cComputeFiltreRelOr &,
                    // const cTplValGesInit<cFiltreDeRelationOrient> &,
                    // cComputeFiltreRelSsEch * & aFSsEch,
                    bool aSym
             );

        void AddAllCpleKeySet
             (
                    //bool IsReflexif,
                    const std::string & aKEY1,
                    const std::string & aKEY2,
                    int aDeltaMin,
                    int aDeltaMax,
                    cComputeFiltreRelOr &,
                    // const cTplValGesInit<cFiltreDeRelationOrient> &,
                    // cComputeFiltreRelSsEch * & aFSsEch,
                    bool aSym,
                    bool IsCirc,
                    int aSampling=1  // <=0  vide ,  1 no sampling, >1 => sampling
             );



        cInterfChantierNameManipulateur & mICNM;
        cNameRelDescriptor  mNRD;
    bool                mIsComp;
    bool                mIsReflexif;

    std::vector<cCpleString>  mIncl;
    std::set<cCpleString>  mExcl;



};

template <class Type>  class  cMapIdName2Val
{
      public :
          void Add(const std::string& anIdBase,const std::string& anIdVal,const Type & aVal);
          Type * Get(const std::string& anIdBase,const std::string& anIdVal);
          const Type * Get(const std::string& anIdBase,const std::string& anIdVal) const;


      private :

          typedef std::map<std::string,Type> tOneBase;
          typedef std::map<std::string,tOneBase> tAllBases;
          tAllBases mDatas;
};

template <class tBase,class tEntriesXML>
        void Add2Base(tBase &,const std::string & aNameBase,const std::list<tEntriesXML>&);



typedef enum
{
    eDefSCMN,
    eMMLCD_SCMN,  // Venant du MicMacLocal ....
    eUnknownSCMN
} eOriSCMN;

class cStdChantierMonoManipulateur : public cInterfChantierNameManipulateur
{
     public :

       // Acces a la base de donnees

         cSetName *  GetSet(const tKey &);


        cTplValGesInit<cResBoxMatr> GetBoxOfMatr(const tKey&,const std::string&);
         const cBatchChantDesc  * BatchDesc(const tKey &) const;
         const cShowChantDesc * ShowChant(const tKey &) const ;
     static cStdChantierMonoManipulateur *
             StdGetFromFile
         (
                     eOriSCMN,
             cInterfChantierNameManipulateur * aGlob,
             const std::string & aDir,
             const std::string & aFileXML,
                     bool  AddCamDB
         );

         cStdChantierMonoManipulateur
     (
             eOriSCMN,
         cInterfChantierNameManipulateur * aGlob,
         const std::string & aDir,
         const cChantierDescripteur &
     );

     cContenuAPrioriImage * GetAPriori(const std::string &,const std::string & aKey,cInterfChantierNameManipulateur * ancetre);
     protected :
         const Pt3dr  * SvpGetPt3dr(const std::string& anIdBase,const std::string& anIdVal) const;
         const double * SvpGetScal(const std::string& anIdBase,const std::string& anIdVal) const;
     private :

         // REF A EXEMPLE APRE MACROADDGETBASE dans  ChantierNameAssoc.cpp
         void AddBase(const cBaseDataCD&);
         const tSet *  Get(const tKey &) ;
         tNuplet  Direct(const tKey &,const tNuplet&);
         tNuplet  Inverse(const tKey &,const tNuplet&);
     const bool  * SetIsIn(const tKey & aKey,const std::string & aName);
     bool AssocHasKey(const tKey & aKey) const;
     bool SetHasKey(const tKey & aKey) const;

     bool  RelHasKey(const tKey &)  ;
         cStrRelEquiv *  GetEquiv(const tKey &) ;
         const tRel * GetRel(const tKey &);

         static std::map<std::string,cStdChantierMonoManipulateur *> theDicAlloc;
         void Compute(cInterfChantierNameManipulateur * ancetre);
         void  AddDicAP
               (
                      const std::string & aName,
                      const std::string & aKey,
                      cContenuAPrioriImage * aCAPI
               );


     cInterfChantierNameManipulateur * mGlob;
         //eOriSCMN                mOrig;
         cInterfChantierSetNC  * mSets;
     cInterfChantierNC  * mAssoc;
     std::map<tKey,cStdChantierRel *> mRels;
     std::map<tKey,cStrRelEquiv *> mEquivs;
     std::map<std::string,cContenuAPrioriImage *>  mDicAP;
         std::vector<cContenuAPrioriImage *>           mVecAP;
         cInterfChantierNameManipulateur *             mAncCompute;
         std::list<cBatchChantDesc>                    mLBatch;
         std::list<cKeyedMatrixStruct>                 mLKMatr;
         std::list<cShowChantDesc>                     mLShow;

         cMapIdName2Val<Pt3dr>                         mBasesPt3dr;
         cMapIdName2Val<double>                        mBasesScal;
};

class cStdChantierMultiManipulateur : public cInterfChantierNameManipulateur
{
     public :
         cSetName *  GetSet(const tKey &);
        cTplValGesInit<cResBoxMatr> GetBoxOfMatr(const tKey&,const std::string&);
         cStdChantierMultiManipulateur(int argc,char ** argv,const std::string & aDir);

        void Add(cInterfChantierNameManipulateur *);
        const cBatchChantDesc * BatchDesc(const tKey &) const;
         const cShowChantDesc * ShowChant(const tKey &) const ;
         const Pt3dr  * SvpGetPt3dr(const std::string& anIdBase,const std::string& anIdVal) const;
         const double * SvpGetScal(const std::string& anIdBase,const std::string& anIdVal) const;
     private :
     cContenuAPrioriImage * GetAPriori(const std::string &,const std::string & aKey,cInterfChantierNameManipulateur * ancetre);
         const tSet *  Get(const tKey &) ;
         tNuplet  Direct(const tKey &,const tNuplet&);
         tNuplet  Inverse(const tKey &,const tNuplet&);
     const bool  * SetIsIn(const tKey & aKey,const std::string & aName);
     bool AssocHasKey(const tKey & aKey) const;
     bool SetHasKey(const tKey & aKey) const;

     std::vector<cInterfChantierNameManipulateur *>  mVM;

     bool  RelHasKey(const tKey &) ;
         const tRel * GetRel(const tKey &);
         cStrRelEquiv *  GetEquiv(const tKey &) ;
         void CD_Add(cChantierDescripteur *);
};







    // Non inversible
class cNI_AutomNC : public cInterfNameCalculator
{
    public :
        tNuplet Direct(const tNuplet &);
    cNI_AutomNC(cInterfChantierNameManipulateur *,const cBasicAssocNameToName &);
    private  :
             cInterfChantierNameManipulateur * mICNM;
             cElRegex       mAutomTransfo;
             cElRegex       mAutomSel;
         tNuplet        mNames2Replace;
         std::string    mSep;
             cTplValGesInit<cNameFilter> mFilter;
             cTplValGesInit<cDataBaseNameTransfo> mDBNT;
};

class cInv_AutomNC : public cNI_AutomNC
{
    public :
      cInv_AutomNC
      (
              cInterfChantierNameManipulateur *,
          const cBasicAssocNameToName & aAutDir,
          const cBasicAssocNameToName & aAutInv,
          cInv_AutomNC * aInv = 0
          );
     private  :
          tNuplet Inverse(const tNuplet &);

          cInv_AutomNC * mInv;
};

// Renvoie le premier defini, en inverse comme en direct
class cMultiNC :  public cInterfNameCalculator
{
    public  :
    cMultiNC
    (
             cInterfChantierNameManipulateur *,
             const cKeyedNamesAssociations & aKNA,
             const std::string& aDir,
             const std::string& aSubDir,
             bool               aSubDirRec,
             const std::vector<cInterfNameCalculator *> &  aVNC
    ) ;
       cInterfChantierNameManipulateur *    ICNM();
       cKeyedNamesAssociations * KNA();
    private :
        tNuplet Direct(const tNuplet &);
        tNuplet Inverse(const tNuplet &);
    void AutoMakeSubDir();
    void AutoMakeSubDirRec(const tNuplet & aNuplet);

        cInterfChantierNameManipulateur *     mICNM;
        cKeyedNamesAssociations               mKNA;
        std::vector<cInterfNameCalculator *>  mVNC;
    std::string                           mDir;
    std::string                           mSubDir;
    bool                                  mSubDirRec;
};


class cDicoChantierNC : public cInterfChantierNC
{
    public :

       // Pour l'instant, conservatrice : erreur si plusieur fois la meme Key
        void Add(cInterfChantierNameManipulateur *,const cKeyedNamesAssociations &);
    private :
        void PrivateAdd(const tKey &,cInterfNameCalculator *);

        // Gere les @ : si la cle n'existe pas  et possede un@, essaye de la construire a
        // a partir de la cle sans @
        std::map<tKey,cInterfNameCalculator *>::iterator  StdFindKey(const tKey & aKey);
    bool AssocHasKey(const tKey & aKey) const;
        tNuplet  Direct(const tKey &,const tNuplet&) ;
        tNuplet  Inverse(const tKey &,const tNuplet&);
    std::map<tKey,cInterfNameCalculator *>  mINC_Dico;
};




cParamChantierPhotogram GetChantierFromFile(const std::string & aNameFile);
cParamChantierPhotogram GetChantierFromFile
                        (
                              const std::string & aNameFile,
                              const std::string & aNameTag
                        );

cChantierPhotogram MakeChantierFromParam(const cParamChantierPhotogram &);
cVolChantierPhotogram MakeVolFromParam(const cParamVolChantierPhotogram &);

std::list<cPDV> ListePDV(const cChantierPhotogram &);

cGraphePdv GrapheOfRecouvrement(const cChantierPhotogram & aCh,double aFact);


double ToRadian(const double &,eUniteAngulaire);
double FromRadian(const double &,eUniteAngulaire);

cNameSpaceEqF::eTypeSysResol ToNS_EqF(eModeSolveurEq);

/*
cParamIFDistRadiale * AllocDRadInc
                      (
                   eConventionsOrientation            aKnownC,
               const cCalibrationInternConique &aCalInit,
               cSetEqFormelles &
               );

*/

cSolBasculeRig         Xml2EL(const cXml_ParamBascRigide &);
cXml_ParamBascRigide   EL2Xml(const cSolBasculeRig &);

cTypeCodageMatr ExportMatr(const ElMatrix<double> & aMat);
ElMatrix<double> ImportMat(const cTypeCodageMatr & aCM);

// Return the cParamOrientSHC of a given name
cParamOrientSHC * POriFromBloc(cStructBlockCam & aBloc,const std::string & aName,bool SVP);
// Return the Rotation that transformate from Cam Coord to Block coordinates (in fact coord of "first" cam)
ElRotation3D  RotCamToBlock(const cParamOrientSHC & aPOS);
// Return the Rotation that transformate from Cam1 Coord to Cam2 Coord
ElRotation3D  RotCam1ToCam2(const cParamOrientSHC & aPOS1,const cParamOrientSHC & aPOS2);




cXml_Rotation El2Xml(const ElRotation3D & aRot);
ElRotation3D Xml2El(const cXml_Rotation & aXml);
ElRotation3D Xml2ElRot(const cXml_O2IRotation & aXml);




ElMatrix<double>   Std_RAff_C2M
                   (
                       const cRotationVect             & aRVect,
                       eConventionsOrientation aConv
                   );


ElRotation3D  Std_RAff_C2M
              (
                  const cOrientationExterneRigide & aCE,
                  eConventionsOrientation aConv
              );

/*
cConvExplicite MakeExplicite(eConventionsOrientation aConv);
ElRotation3D  Std_RAff_C2M
              (
                 const cOrientationExterneRigide & aCE,
                 const cConvExplicite            & aConv
             );
*/


cOrientationExterneRigide From_Std_RAff_C2M
                          (
                               const ElRotation3D &,
                   bool  ModeMatr
                          );


// Fonctionne avec une calib ou une camera orientee
CamStenope * CamOrientGenFromFile(const std::string & aNameFile,cInterfChantierNameManipulateur * anICNM, bool throwAssert = true);

CamStenope * BasicCamOrientGenFromFile(const std::string & aNameFile);
CamStenope * Std_Cal_From_File
             (
                 const std::string & aNameFile,
                 const std::string &  aNameTag = "CalibrationInternConique"
             );

CamStenope * Std_Cal_From_CIC
             (
                   const cCalibrationInternConique & aCIC
             );



cCamStenopeDistRadPol * Std_Cal_DRad_C2M
             (
                const cCalibrationInternConique & aCIC,
            eConventionsOrientation            aKnownC
             );

cCamStenopeModStdPhpgr * Std_Cal_PS_C2M
             (
                const cCalibrationInternConique & aCIC,
            eConventionsOrientation            aKnownC
             );

cCam_Ebner * Std_Cal_Ebner_C2M
             (
                const cCalibrationInternConique & aCIC,
            eConventionsOrientation            aKnownC
         );

cCam_DCBrown * Std_Cal_DCBrown_C2M
             (
                const cCalibrationInternConique & aCIC,
            eConventionsOrientation            aKnownC
         );

cCamera_Param_Unif_Gen *  Std_Cal_Unif
             (
                    const cCalibrationInternConique & aCIC,
                    eConventionsOrientation            aKnownC
             );


cCamStenopeBilin *  Std_Cal_Bilin
             (
                    const cCalibrationInternConique & aCIC,
                    eConventionsOrientation            aKnownC
             );

// cCamStenopeBilin * GlobFromXmlGridStuct(REAL aFoc,Pt2dr aCentre,const cCalibrationInterneGridDef &  aCIG);


bool ConvIsSensVideo(eConventionsOrientation aConv);



Appar23  Xml2EL(const cMesureAppuis &);
cMesureAppuis  El2Xml(const Appar23 &,int aNum=0);

std::list<Appar23>  Xml2EL(const cListeAppuis1Im &);
cListeAppuis1Im  El2Xml(const std::list<Appar23> &,const std::string &NameImage ="NoName");
cListeAppuis1Im  El2Xml(const std::list<Appar23> &, const std::list<int> &     aLInd,const std::string &NameImage ="NoName");



ElRotation3D  CombinatoireOFPA(bool TousDevant,CamStenope & aCam,INT  NbTest,const cListeAppuis1Im &,REAL * Res_Dmin);

cSimilitudePlane El2Xml(const ElSimilitude &);
ElSimilitude Xml2EL(const cSimilitudePlane &);


cAffinitePlane El2Xml(const ElAffin2D &);
ElAffin2D Xml2EL(const cAffinitePlane &);
ElAffin2D Xml2EL(const cTplValGesInit<cAffinitePlane> &);

eTypeProj Xml2EL(const eTypeProjectionCam &);
eTypeProjectionCam El2Xml(const eTypeProj &);

// Encapsulee si + tard on gere qq ch de + complique comme
// des piles de transfo
void AddAffinite(cOrientationConique &,const ElAffin2D &);
ElAffin2D AffCur(const cOrientationConique & anOri);

void AssertOrIntImaIsId(const cOrientationConique &);

ElCamera * Gen_Cam_Gen_From_XML (bool CanUseGr, const cOrientationConique  & anOC,cInterfChantierNameManipulateur * anICNM,const std::string & aDir,const std::string & aNameFile);
ElCamera * Cam_Gen_From_XML ( const cOrientationConique  & anOC,cInterfChantierNameManipulateur * anICNM,const std::string & aNameFile);


ElCamera * Gen_Cam_Gen_From_File
           (
                  bool CanUseGr,
                  const std::string & aNameFile,
                  const std::string &  aNameTag,
                  cInterfChantierNameManipulateur * anICNM
           );

ElCamera * Cam_Gen_From_File
           (
                  const std::string & aNameFile,
                  const std::string &  aNameTag,
                  cInterfChantierNameManipulateur * anICNM
           );

ElCamera * Cam_Gen_From_File
           (
                  const std::string & aNameFile,
                  const std::string &  aNameTag,
                  bool                 Memo,
                  bool                 CanUseGr, // =true
                  cInterfChantierNameManipulateur * anICNM
           );

cFichier_Trajecto * GetTrajFromString(const std::string & aNameFile,bool toMemo);



bool NameFilter(const std::string & aSubD,cInterfChantierNameManipulateur *,const cNameFilter &,const std::string &);
bool NameFilter(const std::string & aSubD,cInterfChantierNameManipulateur *,const cTplValGesInit<cNameFilter> &,const std::string &);

bool NameFilter(cInterfChantierNameManipulateur *,const cNameFilter &,const std::string &);
bool NameFilter(cInterfChantierNameManipulateur *,const cTplValGesInit<cNameFilter> &,const std::string &);





cXML_Date  XML_Date0();
cXML_LinePt3d MM2Matis(const Pt3dr &);

// corientation MM2Matis(const cOrientationConique &);
// cElXMLTree * ToXmlTreeWithAttr(const corientation &);

void DoSimplePastisSsResol(const std::string & aFullName, int aResol, bool forceTMP);


void ModifDAF(cInterfChantierNameManipulateur*,cDicoAppuisFlottant &,const cTplValGesInit<cModifIncPtsFlottant> &);
void ModifDAF(cInterfChantierNameManipulateur*,cDicoAppuisFlottant &,const cModifIncPtsFlottant &);
void ModifDAF(cInterfChantierNameManipulateur*,cDicoAppuisFlottant &,const cOneModifIPF &);

bool SameGeometrie(const cFileOriMnt & aF1,const cFileOriMnt & aF2);


Pt3dr  ToMnt(const cFileOriMnt &,const Pt3dr & aP);
Pt2dr  ToMnt(const cFileOriMnt &,const Pt2dr  &aP);
double ToMnt(const cFileOriMnt &,const double & aZ);
Pt3dr  FromMnt(const cFileOriMnt &,const Pt3dr & aP);
Pt2dr  FromMnt(const cFileOriMnt &,const Pt2dr  &aP);
double FromMnt(const cFileOriMnt &,const double & aZ);

Fonc_Num  AdaptFonc2FileOriMnt
          (
                 const std::string & aContext,
                 const cFileOriMnt & anOriCible,
                 const cFileOriMnt &  aOriInit,
                 Fonc_Num            aFonc,
                 bool                aModifDyn,  // Si il faut adpater en tenant cont des param altimetriques
                 double              aZOffset, // dans le cas modif dyn
                 const Pt2dr &       CropInOriCible
          );


class cSetOfMesureAppuisFlottants;
class cOneMesureAF1I;
// Ecrit les images dans le NamePt !!!
std::vector<cOneMesureAF1I> GetMesureOfPts(const cSetOfMesureAppuisFlottants &,const std::string & aNamePt);
std::vector<cOneMesureAF1I> GetMesureOfPtsIm(const cSetOfMesureAppuisFlottants &,const std::string & aNamePt,const std::string & aNameIm);

class cDicoAppuisFlottant;
class cOneAppuisDAF;
const cOneAppuisDAF * GetApOfName(const cDicoAppuisFlottant &,const std::string & aNamePt);

ElPackHomologue PackFromCplAPF(const cMesureAppuiFlottant1Im & aMes, const cMesureAppuiFlottant1Im & aRef);

const cOneMesureAF1I *  PtsOfName(const cMesureAppuiFlottant1Im &,const std::string & aName);
cMesureAppuiFlottant1Im *  GetMAFOfNameIm(cSetOfMesureAppuisFlottants & aSMAF,const std::string aNameIm,bool CreatIfNone);


class cImSecOfMaster;
const std::list<std::string > * GetBestImSec(const cImSecOfMaster&,int aNb=-1,int aNbMin=-1,int aNbMax=1000,bool OkAndOutWhenNone=false);

cImSecOfMaster StdGetISOM
               (
                    cInterfChantierNameManipulateur * anICNM,
                    const std::string & aNameIm,
                    const std::string & anOri
               );






Im1D_INT4 LutIm(const cLutConvertion & aLut,int aMinVal,int aMaxVal,double aCoeff);
Im1D_INT4 LutGama(int aNbVal,double aGama,double aValRef,int aMaxVal,double aCoeff);
Fonc_Num SafeUseLut(Im1D_INT4 aLut,Fonc_Num aF,double aCoeff);


cGridDirecteEtInverse   ToXMLExp(const cDbleGrid&);

class cCompCNFC
{
     public :
        cCompCNFC(const cCalcNomFromCouple&);
    std::string NameCalc(const std::string&,const std::string&);
     private :
         cCalcNomFromCouple mCNFC;
     cElRegex           mAutom;
};

class cCompCNF1
{
     public :
        cCompCNF1(const cCalcNomFromOne&);
    std::string NameCalc(const std::string&);

     private :
         cCalcNomFromOne   mCNF1;
     cElRegex           mAutom;

};

class cStdMapName2Name
{
     public :
          cStdMapName2Name
          (
                const cMapName2Name &,
                cInterfChantierNameManipulateur  *
          );

          std::string map(const std::string & );
          std::string map_with_def(const std::string & ,const std::string & aDef);

     private :
          cInterfChantierNameManipulateur  *                  mICNM;
          cMapName2Name                                       mMapN2N;
};

cStdMapName2Name * StdAllocMn2n
                   (
                       const cTplValGesInit<cMapName2Name> &  aMapN2N,
                       cInterfChantierNameManipulateur *
                   );



std::vector<std::string> GetStrFromGenStr(cInterfChantierNameManipulateur*,const cParamGenereStr &);
std::vector<std::string> GetStrFromGenStrRel(cInterfChantierNameManipulateur*,const cParamGenereStrVois &,const std::string &);




cEl_GPAO * DoCmdExePar(const cCmdExePar & aCEP,int aNbProcess);

std::list<std::string> GetListFromSetSauvInFile
                       (
                            const std::string & aNameFile,
                            const std::string & aNameTag ="SauvegardeSetString"
                       );


eUniteAngulaire AJStr2UAng(const std::string &);
double AJ2Radian(const cValueAvionJaune &);
Pt3dr  CentreAJ(const cAvionJauneDocument &);

ElMatrix<double> RotationVecAJ(const cAvionJauneDocument &);
ElRotation3D   AJ2R3d(const cAvionJauneDocument &);

cOrientationExterneRigide AJ2Xml(const cAvionJauneDocument &);

std::vector<double>  VecCorrecUnites(const std::vector<double> & aV,const std::vector<eUniteAngulaire> &aVU);



 cCameraEntry *  CamOfName(const std::string & aName);

std::string StdNameGeomCalib(const std::string & aFullName);


void  DC_Add(const cMMCameraDataBase & aDB);


bool RepereIsAnam(const std::string &,bool &IsOrthXCSte,bool & IsAnamXCsteOfCart); // OrthoCyl est un cas 

cConvExplicite GlobMakeExplicite(eConventionsOrientation aConv);
ElRotation3D  GlobStd_RAff_C2M
              (
                 const cOrientationExterneRigide & aCE,
                 const cConvExplicite            & aConv
             );

cConvExplicite GlobMakeExplicite(const cConvOri & aConv);


void AdaptDist2PPaEqPPs(cCalibDistortion & aCD);

void MakeMetaData_XML_GeoI(const std::string & aNameImMasq);
void MakeMetaData_XML_GeoI(const std::string & aNameImMasq,double aResol);


//estd::set<std::string> SetOfCorresp(const std::vector<cCpleString> & aRel,const std::string &);

class cAppliListIm
{
    public :
       cAppliListIm(const std::string & aFullName);
    public :

      std::string                       mFullName;
      std::string                       mDir;
      std::string                       mPat;
      cInterfChantierNameManipulateur * mICNM;
      const cInterfChantierNameManipulateur::tSet * mSetIm;
};

// inline functions

bool isUsingSeparateDirectories(){ return MMUserEnv().UseSeparateDirectories().ValWithDef(false); }


// === Gestionnaire de nom pour les fusions ===============

typedef enum
{
    eTMIN_Depth,
    eTMIN_Min,
    eTMIN_Max,
    eTMIN_Merge
} eTypeMMByImNM;



void MakeListofName(const std::string & aFile,const cInterfChantierNameManipulateur::tSet  *);
void AddistofName(const std::string & aFile,const cInterfChantierNameManipulateur::tSet  *);


class cMMByImNM
{
    public :

        static cMMByImNM * ForGlobMerge(const std::string & aDirGlob,double aDS, const std::string & aNameMatch,bool AddDirLoc=false);
        static cMMByImNM * ForMTDMerge(const std::string & aDirGlob,const std::string & aNameIm,const std::string & aNameMatch);

        static cMMByImNM * FromExistingDirOrMatch(const std::string & aNameDirOriOrMatch,bool Svp,double aDS=1,const std::string & aDir0="./",bool AddDirLoc=false);

        void DoDownScale(const std::string & aNameIm);
        void PatDoDownScale(const std::string & aPat);


        std::string NameFileMasq(eTypeMMByImNM,const std::string aNameIm);
        std::string NameFileProf(eTypeMMByImNM,const std::string aNameIm);
        std::string NameFileXml(eTypeMMByImNM,const std::string aNameIm);
        std::string NameFileEntete(eTypeMMByImNM,const std::string aNameIm);
        std::string NameFileLabel(eTypeMMByImNM,const std::string aNameIm);

        void ModifIp(eTypeMMByImNM,cImage_Profondeur &,const std::string & aNameIm);

        const std::string & FullDir() const;
        const std::string & DirGlob() const;
        const std::string & DirLoc() const;
        const std::string & NameType() const;

        void AddistofName(const cInterfChantierNameManipulateur::tSet  *);
        const std::string & KeyFileLON() const;
        const cEtatPims & Etat() const;
        void  SetOriOfEtat(const std::string &) ;
        const std::string & GetOriOfEtat() const;
          
        static bool  StrIsPImsDIr(const std::string &);
        static std::string StdDirPims(double aDS, const std::string & aNameMatch);

    private  :
        cMMByImNM (double aDS,const std::string & aDirGlob,const std::string & aDirLoc,const std::string & aPrefix,const std::string &  aNameType,bool AddDirLoc=false) ;

        static std::string NameOfType(eTypeMMByImNM);
        std::string NameFileGlob(eTypeMMByImNM,const std::string aNameIm,const std::string aExt);
        std::string NameFileLoc(eTypeMMByImNM,const std::string aNameIm,const std::string aExt);




        double         mDS;
        std::string    mDirGlob;
        std::string    mDirLoc;
        std::string    mPrefix;
        std::string    mFullDir;
        std::string    mNameFileLON;
        std::string    mKeyFileLON;
        std::string    mNameEtat;
        cEtatPims      mEtats;
        std::string    mNameType;

        static const std::string TheNamePimsFile;
        static const std::string TheNamePimsEtat;
};

bool IsMacType(eTypeMMByP aType);

void AutoDetermineTypeTIGB(eTypeImporGenBundle & aType,const std::string & aName);

double UneUniteEnRadian(eUniteAngulaire aUnit);



#endif   // _ELISE_XML_GEN_ALL_H



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
