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

#ifndef _ELISE_XML_GEN_CINTERFCHANTIERNAMEMANIPULATEUR_H
#define _ELISE_XML_GEN_CINTERFCHANTIERNAMEMANIPULATEUR_H

#include "general/CMake_defines.h"

class cDicoSetNC;
class  cSetName;
class cLStrOrRegEx;

class cResulMSO
{
   public :
       cResulMSO();
       ElCamera * &       Cam() ;
       cElNuage3DMaille * & Nuage() ;
       cBasicGeomCap3D * &       Capt3d();
       bool &               IsKeyOri();
   private :
       bool                 mIsKeyOri;
       ElCamera *           mCam;
       cElNuage3DMaille *   mNuage;
       cBasicGeomCap3D *         mCapt3d;

};


class cLStrOrRegEx
{
    public :
       //void Add(const std::string & aName);
       //void Add(cElRegex * anAutom);
       void  AddName(const std::string & aName,cInterfChantierNameManipulateur *anICNM);
       cLStrOrRegEx();
       bool AuMoinsUnMatch(const std::string & aName);
       
    private  :
       std::set<std::string>     mSet;
       std::list<cElRegex *>     mAutom;
};


class cInterfChantierSetNC
{
     public :
         typedef  std::vector<std::string>   tSet;
     typedef  std::string             tKey;

       // renvoie 0 si pas trouve
         virtual const tSet *  Get(const tKey &) = 0;
     virtual const bool  * SetIsIn(const tKey &,const std::string & aName) = 0;
     virtual bool SetHasKey(const tKey & aKey) const=0;
         static cDicoSetNC * StdCalcFromXML
                         (
                        cInterfChantierNameManipulateur *,
                    const std::list<cKeyedSetsOfNames> &
                     );
     virtual ~cInterfChantierSetNC();
         virtual cSetName *  GetSet(const tKey &) = 0;
     private :
};

class cDicoSetNC : public cInterfChantierSetNC
{
     public :
         void Add(const tKey &,cSetName *);
         bool SetHasKey(const tKey & aKey)  const;
         cDicoSetNC();
         void assign( const tKey &,cSetName * );
     private :
         std::map<tKey,cSetName *> mDico;
         const tSet *  Get(const tKey &);
         cSetName *  GetSet(const tKey &);

     const bool  * SetIsIn(const tKey &,const std::string & aName);
};

    //
/*
class  cSetName
{
     public :
          typedef cInterfChantierSetNC::tSet tSet;
          virtual const tSet  * Get() = 0;
      virtual ~cSetName();
      virtual bool IsIn(const std::string & aName) = 0;
};


class  cSetNameByAutom : public cSetName
*/
class  cSetName
{
      public :
          typedef cInterfChantierSetNC::tSet tSet;
          const tSet  * Get();
          cSetName(cInterfChantierNameManipulateur *,const cSetNameDescriptor &);
          bool SetBasicIsIn(const std::string & aName);

          // ce devrait etre SetBasicIsIn, mais par compat avec ce qui marche on ne change pas
          bool IsSetIn(const std::string & aName);

          const cSetNameDescriptor & SND() const;
          cInterfChantierNameManipulateur * ICNM();
          string Dir() const { return mDir; }
          void setDir( const string &i_directory ) { mDir = i_directory; }
      private :
          void CompileDef();
          void AddListName(cLStrOrRegEx & aLorReg,const std::list<std::string> & aLName,cInterfChantierNameManipulateur *anICNM);

          void InternalAddList(const std::list<std::string> &);

          bool mExtIsCalc;
          bool mDefIsCalc;
          cInterfChantierNameManipulateur *          mICNM;
          std::string                mDir;
          cSetNameDescriptor         mSND;
          cInterfChantierSetNC::tSet mRes;
          cLStrOrRegEx               mLA; // Accepteur
          cLStrOrRegEx               mLR; // Refuteur

};

class cResBoxMatr
{
    public :
        Box2dr  mBox;
        Pt2di   mId;
        int     mNumMatr;
        int     mNumGlob;
};



class cCompileCAPI : public cContenuAPrioriImage
{
     public :
        cCompileCAPI();

        cCompileCAPI
        (cInterfChantierNameManipulateur &,
        const cContenuAPrioriImage &,
        const std::string &aDir,
        const std::string & aName,
        const std::string & aName2,
        bool forceTMP);

    Pt2dr Rectif2Init(const Pt2dr &);
    const std::string & NameRectif() const;
     private :
        // Valeur qui ne tient pas compte des bords
    Pt2dr V0Init2Rectif(const Pt2dr &);
    Pt2dr V0Rectif2Init(const Pt2dr &);


    std::string mFullNameFinal;
    double   mScale;
    double   mTeta;
    Box2di   mBox;
    // Tiff_Im  mFileInit;
    Pt2di    mSzIm;
    Pt2dr    mRotI2R;
    ElSimilitude  mRTr_I2R;
    ElSimilitude  mRTr_R2I;
    ElSimilitude  mSim_R2I;
};


// Pour l'instant, on n'a besoin que du minimum (savoir si deux element
// sont dans la meme classe)
class cStrRelEquiv
{
    public :
       bool SameCl(const std::string &,const std::string &) ;
       cStrRelEquiv
       (
            cInterfChantierNameManipulateur &,
            const cClassEquivDescripteur &
       );

       std::string ValEqui(const std::string &) ;
       const std::vector<std::string> * Classe(const std::string &) ;
       void Compile();

    private :
       cStrRelEquiv(const cStrRelEquiv &) ;
       cInterfChantierNameManipulateur & mICNM;
       cClassEquivDescripteur            mKCE;

       const std::vector<std::string> *  mGlobS;
       std::map<std::string,std::vector<std::string>* >   mClasses;
       bool mCompiled;
};

// Seule interface necessaire a l'utilisation
//
//

class cMMDataBase
{
      friend class cInterfChantierNameManipulateur;

    private :

      std::map<std::string,cXmlExivEntry *> mExivs;
};

class cInterfChantierNameManipulateur
{
     public :
    typedef  std::string             tKey;



        std::string   NameAppuiEpip(const std::string & anOri,const std::string & aIm1,const std::string & aIm2) ;
        std::string   NameImEpip(const std::string & anOri,const std::string & aIm1,const std::string & aIm2) ;
        std::string   NameOrientEpipGen(const std::string & anOri,const std::string & aIm1,const std::string & aIm2) ;


        ElPackHomologue StdPackHomol(const std::string & anExt,const std::string & aI1,const std::string &aI2);
        std::string  StdNameHomol(const std::string & anExt,const std::string & aI1,const std::string &aI2);


         std::string  NameOriStenope(const tKey & aKeyOri,const std::string & aNameIm);
         std::string  StdNameCalib(const std::string & anOri,const std::string & aNameIm);  // =>  Ori-XX/AutoCal ...

//  !!!! CONVENTION DIFFERENTES ENTRE StdCamStenOfNames  et les deux AUTRES ,
//  !! NAME puis ORI
         CamStenope *  StdCamStenOfNames(const std::string & aNameIm,const std::string & anOri);  // => Ori-XX/Orientation...
         CamStenope *  StdCamStenOfNamesSVP(const std::string & aNameIm,const std::string & anOri); // => return 0 si Ori non existed

//  !!!  ORI pui NAME
         std::string  StdNameCamGenOfNames(const std::string & anOri,const std::string & aNameIm);
         cBasicGeomCap3D *  StdCamGenerikOfNames(const std::string & anOri,const std::string & aNameIm,bool SVP=false);  // => Ori-XX/Orientation...

         // Ori-XX/Orientation... exist, sinon  Ori-XX/GB-Orientation..
         // cBasicGeomCap3D * StdCamGenOfNames(const std::string & anOri,const std::string & aNameIm,bool SVP=false);
         // return "" si rien trouve


         CamStenope * GlobCalibOfName(const std::string  & aNameIm,const std::string & aPrefOriCal,bool ModeFraser /* Genre un Fraser Basixc ss dist*/ ); // No Dist if aPrefOriCal=""


         std::list<std::string> GetListImByDelta(const cListImByDelta &,const std::string & aN0);

         std::vector<std::string> StdGetVecStr(const std::string &);  // Cas [A,Bn..] , ou toto.txt lit fichie, sinon singleton
         cResulMSO MakeStdOrient(std::string &,bool AccepNone,std::string * aNameIm=0,bool SVP=false);

         cSetName *  KeyOrPatSelector(const std::string &);
         cSetName *  KeyOrPatSelector(const cTplValGesInit<std::string> &);

         const Pt3dr  & GetPt3dr(const std::string& anIdBase,const std::string& anIdVal) ;
         const double  & GetScal(const std::string& anIdBase,const std::string& anIdVal) ;
    // Si le fichier n'existe pas , mais que le sym existe,
    // le cree
        std::string NamePackWithAutoSym
                    (
                        const std::string & aKey,
                        const std::string & aName1,
                        const std::string & aName2,
                        bool  SVP = false  // Si SVP et aucun existe, renvoie le 1er
                    );



        void SetMapCmp(const cCmdMappeur &, int argc,char ** argv);
        void SetMapCmp(const std::list<cCmdMappeur> &, int argc,char ** argv);

        static const char theCharModifDico; // A priori +
        static const char theCharSymbOptGlob; // A priori @

        cInterfChantierNameManipulateur(int argc,char ** argv,const std::string & aDir);
        virtual ~cInterfChantierNameManipulateur();
        typedef  std::vector<std::string>   tNuplet;
        typedef  std::vector<std::string>   tSet;

        virtual  cTplValGesInit<cResBoxMatr> GetBoxOfMatr(const tKey&,const std::string&)=0;

    typedef std::vector<cCpleString>  tRel;


    virtual bool  RelHasKey(const tKey &)  = 0;
        virtual const tRel * GetRel(const tKey &) = 0;

        tSet  GetSetOfRel(const tKey &,const std::string & aStr0,bool Sym=true);



        virtual const tSet *  Get(const tKey &) = 0;
        virtual cSetName *  GetSet(const tKey &) = 0;
        virtual cStrRelEquiv *  GetEquiv(const tKey &) = 0;

    std::string  Assoc1To1(const tKey &,const std::string & aName,bool isDir);
    std::string  Assoc1To1(const std::list<tKey> &,const std::string & aName,bool isDir);

    std::string  Assoc1ToN(const tKey &,const tSet & aVNames,bool isDir);
    std::string  Assoc1To2(const tKey &,const std::string & aName,const std::string & aName2,bool isDir);
    std::string  Assoc1To3(const tKey &,const std::string & aName,const std::string & aName2,const std::string & aName3,bool isDir);

    std::pair<std::string,std::string> Assoc2To1(const tKey &,const std::string & aName,bool isDir);

        virtual tNuplet  Direct(const tKey &,const tNuplet&) = 0 ;
        virtual tNuplet  Inverse(const tKey &,const tNuplet&) =0;
        virtual const bool  * SetIsIn(const tKey & aKey,const std::string & aName) =0;
        virtual bool AssocHasKey(const tKey & aKey) const = 0;
        std::string StdKeyOrient(const tKey &); // Elle meme si existe sinon NKS

        virtual bool SetHasKey(const tKey & aKey) const = 0;
    //  Renvoie true si c'est un fichier et pas une cle
    //  Renvoie false si c'est une cle et pas un fichier
    //  Genere une erreur si c'est aucun ou les deux
    bool  IsFile(const std::string &);

         // Si IsFile(aKeyOrFile), le renvoie aKeyOrFile sinon utilise comme cle
     // pour transformer anEntry
         std::string  StdCorrect
                  (
                  const std::string & aKeyOrFile,
              const std::string & anEntry,
              bool Direct
                      );
         std::string  StdCorrect2
                  (
                  const std::string & aKeyOrFile,
              const std::string & anEntry1,
              const std::string & anEntry2,
              bool Direct
                      );

         // Prof par defaut 2, par compat avec l'existant
         std::list<std::string>  StdGetListOfFile(const std::string & aKeyOrPat, int aProf=2,bool ErrorWhenEmpty=true);
     // Quatre  dictionnaire sont charges :
     //   Priorite 0 :
     //   Priorite 1 : aDir/aName.Val()  (si aName est initialise)
     //   Priorite 2 : aDir/LocalChantierDescripteur.xml (s'il existe)
     //   Priorite 3 : applis/XML-Pattron/DefautChantierDescripteur.xml
     //
    static cInterfChantierNameManipulateur* StdAlloc
                     (
                               int argc,char **argv,
                               const std::string & aDir,
                   const cTplValGesInit<std::string> aName,
                   cChantierDescripteur * aCDisc  = 0,
                               bool                   DoMkDB  = true
                     );

           static cInterfChantierNameManipulateur* BasicAlloc(const std::string & aDir);

      // A cause du facteur d'echelle, l'a priori depend de la paire d'image a
      // apparier
       std::pair<cCompileCAPI,cCompileCAPI>
            APrioriAppar
        (const std::string & aN1,
          const std::string & aN2,
          const std::string & aKEY1,
          const std::string & aKEY2,
          double              aSzMax,  // Si >0 ajuste les echelle pour que
                                         // la plus grande dimension soit aSzMax
          bool forceTMP //forbid using original picture
         );

       cContenuAPrioriImage  APrioriWithDef(const std::string &,const std::string & aKey);

      virtual cContenuAPrioriImage * GetAPriori
                                         (
                                           const std::string &,
                                           const std::string & aKey,
                                           cInterfChantierNameManipulateur * ancetre
                                         ) = 0;

      const std::string   & Dir () const;
      void setDir( const std::string &i_directory );
          cArgCreatXLMTree &  ArgTree();

         // Assez sale, interface pour aller taper dans
         // StdChantierMonoManipulateu
           virtual void CD_Add(cChantierDescripteur *);
           virtual const cBatchChantDesc * BatchDesc(const tKey &) const = 0;
           virtual const cShowChantDesc * ShowChant(const tKey &) const = 0;



           const cPtTrajecto & GetPtTrajecto
                               (
                                    const cFichier_Trajecto & aFT,
                                    const std::string &       aKeyAssoc,
                                    const  std::string &      aNameIm
                               );



         virtual const Pt3dr  * SvpGetPt3dr(const std::string& anIdBase,const std::string& anIdVal) const = 0;
         virtual const double * SvpGetScal(const std::string& anIdBase,const std::string& anIdVal) const = 0;


         std::string  AdaptNameAbs(const std::string &);
         void SetKeySuprAbs2Rel(std::string * aStr);


          // Pour les problemes de noms absolus laisses dans les fichier
          void  StdTransfoNameFile(std::string &);

          // Pour les problemes de noms absolus laisses dans les fichier
          void SetMkDB(const cMakeDataBase &);
          void MkDataBase();
          void  AddMTD2Name(std::string & aName,const std::string & aSep,double aMul);
          std::string  DBNameTransfo(const std::string &,const cTplValGesInit<cDataBaseNameTransfo> &);
          cXmlExivEntry * GetXivEntry(const std::string & aName);


          static cInterfChantierNameManipulateur * Glob();

          bool CorrecNameOrient(std::string & aNameOri,bool SVP=false) ;


     private :
          bool  TestStdOrient (const std::string & Manquant, const std::string & Prf, std::string & anOri,bool AddNKS);


          double BiggestDim(const cContenuAPrioriImage &,const std::string & aNameIm);
          std::string  mDir;




          static const std::string theNameGlob;
          static const std::string theNameLoc;
          static const std::string theNamePrelim;
          cArgCreatXLMTree         mArgTr;
          std::string * mStrKeySuprAbs2Rel;

          std::string NameDataBase(const std::string & anExt);

          void CreateDataBase(const std::string & aDB);

          cMMDataBase   *   mDB;
          cMakeDataBase  *  mMkDB;

          static cInterfChantierNameManipulateur * TheGlob;
          std::map<std::string,CamStenope *>       mMapName2Calib; // Utilise avec GlobCamOfName
};



#endif   // _ELISE_XML_GEN_CINTERFCHANTIERNAMEMANIPULATEUR_H



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
