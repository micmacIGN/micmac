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
Add CVS :

     Grid.tif
     XML_Test/ParamSaisie.xml
A FAIRE  (Par priorite)

  Bugs :
     o report de l'origine repere tangent local
     o pb d'inversion des Y ?

  "pas chers" :
     o  Choix des images en fonction des etapes
     o  Geometries supplementaires (Carto, Faisceaux ...)
 

   "plus chers" 

     o Modeles analytiques : utiles pour PXS

     o Prog Dyn 2D  : utiles pour TS-Autocal

     o Primitives : utile pour bati 3D ? (MNT par routes BD topo)
     o Parties cachees
     o Utilisation des contours

FAIT :
     o  Generation des ori
     o  Masque Binaire 
*/

#ifndef _ELISE_MICMAC_ALL_H_
#define _ELISE_MICMAC_ALL_H_

// Pour la multi correl ponctuelle, le cout de stockage peut etre eleve,
//   donc au depart on a stocke sur des INT1
//   mais ensuite il est apparu que cela pouvait creer des probleme de dynamique
//   on se garde la possibilite d'avoir des INT2
// si l'algo finit par etre valide, on fera peut etre du INT1 avec une dynamique non lineaire

#define USE_INT1_4_MCP false

extern const double MCPMulCorel;

#if (USE_INT1_4_MCP)
typedef U_INT2 tCRVal;
const tCRVal ValUndefCple = 0 ;  // Valeur pour coder une valeur inexistante en correl "a un pixel" multi image
typedef INT1 tMCPVal;
const int TheDynMCP = 127;
const int ValUndefCPONT = -128 ;  // Valeur pour coder une valeur inexistante en correl "a un pixel" multi image
#else
typedef REAL4 tCRVal;
typedef INT2 tMCPVal;
const int TheDynMCP = 10000;
const int ValUndefCPONT = -20000 ;  // Valeur pour coder une valeur inexistante en correl "a un pixel" multi image
// #define TheDynMCP 10000 
// #define ValUndefCPONT  -20000 // Valeur pour coder une valeur inexistante en correl "a un pixel" multi image
const tCRVal ValUndefCple = -1 ;  // Valeur pour coder une valeur inexistante en correl "a un pixel" multi image
#endif

inline int AdaptCostPonct(int aVal)
{
   return ElMax(-TheDynMCP,ElMin(TheDynMCP,aVal));
}



#define BRK_MICMAC_MES(aMes) \
{\
    if (MPD_MM())\
    {\
       std::cout << aMes << " ;BRK = "<< __LINE__ <<" at " << __FILE__ << "\n";\
        getchar();\
    }\
}

#define BRK_MICMAC  BRK_MICMAC_MES("")

#include "cInterfModuleImageLoader.h"

    // ================

#include "GpGpu/GpGpu.h"
//#include "StdAfx.h"

// #include  "cParamMICMAC.h"

#include "uti_MICMAC.h"
template <class T> class cMatrOfSMV;


void MicMacRequiresBinaireAux();



class cGLI_CalibRadiom;



extern cDebugEscalier *  theDE;

typedef enum
{
   eModeNuage3D,
   eModeCarteProfExterne,
   eModeCarteProfInterne //   : si + tard on souhaite embarquer l'image
} eModeExportNuage;




GenIm::type_el TypeIMIL2El(eIFImL_TypeNumerique);

template <class TypeEl,class TypeBase> 
void LoadAllImCorrel
     (
          Im2D<TypeEl,TypeBase> anIm,
          cInterfModuleImageLoader * anIMIL,
          int aDZ,
          Pt2di aP0File
     )
{
   anIMIL->LoadCanalCorrel
   (
        sLowLevelIm<TypeEl>
        (
            anIm.data_lin(),
            anIm.data(),
            Elise2Std(anIm.sz())
        ),
        aDZ,
        cInterfModuleImageLoader::tPInt(0,0),
        Elise2Std(aP0File),
        Elise2Std(anIm.sz())
   );
}

template <class TypeIm>  
TypeIm CreateAllImAndLoadCorrel(cInterfModuleImageLoader * anIMIL,int aDZ)
{
   Pt2di aSz = Std2Elise(anIMIL->Sz(aDZ));
   TypeIm aRes(aSz.x,aSz.y);
   LoadAllImCorrel(aRes,anIMIL,aDZ,Pt2di(0,0));
   return aRes;
}


/*
        Abbreviations et vocabulaire utilises

	N.I.  : non implemented, courrant pour le constructeur
	de copies sur les "gros" objets

        Dans tous les commentaires, noms de variables et de
        fonction, le termes "Paralaxe" est utilise de maniere
        un peu devoye / a son sens reel. Il peut s'agir :
            * d'une vraie paralaxe,
            * d'un couple de coordonnees definissant
              une vraie paralaxe + une paralaxe transverse, 
            * d'un couple de coordonnees jouant des role similaire,
            * du Z terrain
        Le terme designe de maniere generique le systeme de coordonnee
        inconnu (1 ou 2D) qui permet de mettre en correspondance la
        geometrie objet avec les images.
*/




/*
     "Header du Header",  = declaration des classes
 */

// Pour memoire, classes definies dans "cParamMICMAC.h" :

// class cEtapeMEC;
// class cParamMICMAC

// Classe implementant les services de transformations
// geometriques.
   // Correspondance entre l'espace objets "reels" et
   // l'espace discretises
class cGeomDiscR2  ; //                          Dans cGeomXXX
class cGeomDiscFPx ; // : public  cGeomDiscR2    Dans cGeomXXX

   // Correspondance entre espace objet et espaces images
class cGeomImage;  

   // Classe pour representer une image en memoire, essentiellement
   // une classe abstraite , ce sont les derivees templates
   // qui implementent les services reels, typiquement les
   // dervivees contiendront plusieurs images (image, image
   // redressee, moyennee etc....);
class cLoadedImage;

// Classe "maitresse", gere une application
// de mise en correspondance en "coordonnant" les services
// des autres classes
class cAppliMICMAC;  

// Represente une image :  les nom de fichiers
// suivant les resolution, la geometrie
class cPriseDeVue;

// Etape de Mise en correspondance "compilee"
// c'est a dire apres post-traitements
// Role :
//     - verifier en detail les <EtapeMEC> et
//     terminer leur "binding" en memoire
//     - gerer les fichier de paralaxes
//     (calcul des nappes a partir des resolutions
//     inferieures, sauvegarde)
//
class cEtapeMecComp;

// Contient les informations qui sont specifiques a un numero
// de DeZoom donne. Cette classe etant arrivee assez tard, pas
// mal d'info qui devraient s'y trouver sont en fait gerees en
// dehors
class cCaracOfDeZoom;

// Class auxiliaire pour l'implementation de cFilePx
struct cArgOneEtapePx; 

// Gere chaque fichier de paralaxe
class cFilePx;  

// Gere  une "nappe de paralaxe", 
struct cOneNappePx;  

// Gere le chargement en memoire d'une zone de terrain
// (nappes , masque ...)
class cLoadTer;

// Classe abstraite, interface de CoxRoy ou de Prog-Dyn
class cSurfaceOptimiseur;

// class permettant de representer des distribution
// de valeur, d'en calculer la correlation
class cStat1Distrib;
class cStatOneClassEquiv;

class cPxSelector;  // Classe d'interface

// Dim de paralaxe 1 ou 2
static const int theDimPxMax = 2;

class cLineariseProj;

class cModeleAnalytiqueComp;

class cMicMacZbuf;

// Definition des containers
typedef std::vector<cPriseDeVue *>  tContPDV;
typedef tContPDV::const_iterator    tCsteIterPDV;
typedef tContPDV::iterator          tIterPDV;
typedef std::list<cEtapeMecComp *>  tContEMC;


// Definie dans cOptimisationDiff.cpp
class cOptimDiffer;
class cMicMacVisu;

class cStatOneImage
{
   public :

     void   Normalise(double aMoy,double aSigma);
     void   StdNormalise(double aEpsilon = 1e-10);
     double SquareDist(const cStatOneImage & aS2) const;
     double SquareNorm() const;



      std::vector<double> mVals;
      double              mS1;
      double              mS2;

      cStatOneImage();
      void Reset();
      void Add(const double & aV)
      {
         mVals.push_back(aV);
         mS1 += aV;
         mS2 += ElSquare(aV);
      }
};



/*****************************************************/
/*                                                   */
/*               cSurfaceOptimiseur                  */
/*                                                   */
/*****************************************************/


// cCaracOfDeZoom cEquiv1D cEquiv1D mSRA

class cSurfaceOptimiseur
{
    public :
      void SetCout(Pt2di aPTer,int * aPX,REAL aCost,int aLabel=0); 
      virtual ~cSurfaceOptimiseur();
      void SolveOpt() ;
      static cSurfaceOptimiseur * Alloc(cAppliMICMAC &    mAppli,
                                        cLoadTer&         mLT,
                                        const cEquiv1D &        anEqX,
                                        const cEquiv1D &        anEqY
                                        );
      int                     CostR2I(const double &) const;
      double                  CostI2R(const int & aCost) const;



      int  CostI2CorExport(const int & aCost) const;

      double CostTransEnt(int aTrans,int aKPpx);
      bool                    MaskCalcDone();
      Im2D_Bits<1>            MaskCalc();
      virtual void Local_SetCpleRadiom(Pt2di aPTer,int * aPX,tCRVal aR1,tCRVal aR2);  
      virtual void Local_VecMCP(Pt2di aPTer,int * aPX,const  std::vector<tMCPVal> &);

    protected  :
      cSurfaceOptimiseur
      (
             cAppliMICMAC &    anAppli,
             cLoadTer&,
             double aCsteCost,
             const cEquiv1D &        anEqX,
             const cEquiv1D &        anEqY,
             bool                    AcceptEBI,
             bool                    CanFillCorrel
      );
      // Initialisation qui fait appel des fonctions virtuelles

      double                  CostAmpl(const double &) const;
      virtual void Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int aLabel) =0; 
      




      virtual void Local_SolveOpt(Im2D_U_INT1 aImCor) = 0;

      static cSurfaceOptimiseur * AllocCoxRoy
                                  (
                                     cAppliMICMAC &    mAppli,
                                     cLoadTer&         mLT,
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                                  );
      static cSurfaceOptimiseur * AllocCoxMaxOfCorrel
                                  (
                                     cAppliMICMAC &    mAppli,
                                     cLoadTer&         mLT,
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                                  );
      static cSurfaceOptimiseur * AllocPrgDyn
                                  (
                                     cAppliMICMAC &    mAppli,
                                     cLoadTer&         mLT,
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                                  );

      static cSurfaceOptimiseur * AllocNewPrgDyn
                                  (
                                     cAppliMICMAC &    mAppli,
                                     cLoadTer&         mLT,
                                     const cModulationProgDyn & aPrgD,
                                     const cEtapeProgDyn &      anEPG,
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                                  );

      static cSurfaceOptimiseur * AllocAlgoTestGPU
                                  (
                                     cAppliMICMAC &    mAppli,
                                     cLoadTer&         mLT,
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                                  );


      double                  mCsteCost;
      cAppliMICMAC &          mAppli;
      double                  mDefCost;
      bool                    mOneDCAllDC;
      int                     mDimPx;
      const cEtapeMecComp &   mEtape;
      int                     mDeZoom;  // Terrain
      cCaracOfDeZoom *        mCarDZ;
      int                     mAlgoSR;
      cEquiv1D                mEqX;
      cEquiv1D                mEqY;
      const cEtiqBestImage *  mEBI;
      double                  mGlobSR;
      bool                    mWithEQ;
      cLoadTer&              mLTInit;
      Pt2di                   mSzInit;
      // int                     mSRA;
      cLoadTer *              mLTRed;
      const cLoadTer *        mLTCur;
      Pt2di                   mSzCur;
      double                  mCostRegul[theDimPxMax];
      double                  mCostRegul_Quad[theDimPxMax];

      double                  mSeuilAttenZReg[theDimPxMax];
      double                  mCostRegulAttenue[theDimPxMax];
      double                  mCsteCostSeuilAtten[theDimPxMax];




      // double                  mICostRegul[theDimPxMax];
      //Est la meme que mLT si Res!=1
      std::vector<Im2D_INT2>  mImRes;
      std::vector<INT2 **>    mDataImRes;
      std::vector<Im2D_INT2>  mImResInit;
      bool                    mCubeCorrel;
      cMatrOfSMV<U_INT1> *    mMemoCorrel;
      bool                    mCanFillCorrel;
      bool                    mDoFileCorrel;
      cMatrOfSMV<U_INT2> *    mReducCpt;
      cMatrOfSMV<REAL4>  *    mReducCost;

      bool                    mMaskCalcDone;
      Im2D_Bits<1>            mMaskCalc;

      int CostToMemoCorrel(double aCost) const;

      // Remplace aP / mSRA
      inline Pt2di ToSRAlg(const Pt2di & aP)
      {
          return Pt2di(mEqX.NumClasse(aP.x),mEqY.NumClasse(aP.y));
      }

};

/*****************************************************/
/*                                                   */
/*               cStat1Mage                          */
/*                                                   */
/*****************************************************/

class cStat1Distrib  // dans cStatOneClassEquiv
{
    public :
        cStat1Distrib(int aNbV,int * aPtInEc,int aNbVIn,double aRatioIn);

        REAL CoeffCorrel2Dist(const cStat1Distrib & aD2,REAL anEps) const;
        REAL CoeffCorrel3Dist
             (
                   const cStat1Distrib & aD2,
                   const cStat1Distrib & aD3,
                   REAL anEps
             ) const;
        void NormalizeM1M2(REAL anEps);
        double  * Vals()  {return mVals;}

    private :
        cStat1Distrib(const cStat1Distrib &); // N.I.
        std::vector<double> mVData;
        const int           mNbV;
        double  *           mVals;
        int    *            mPtInEc;
        int                 mNbVIn;
        double              mRatioIn;
};

class cStatOneClassEquiv
{
     public :

           friend class cStatGlob;
           cStatOneClassEquiv
           (
                const cAppliMICMAC &,
                INT aNbDistrib,
                const std::vector<int> & aVPtInEc,
                const std::vector<int> & aVIndiceOK
            );
           ~cStatOneClassEquiv();
            void            Clear();
            cStat1Distrib * NextDistrib();



            bool  & SomsMade() {return mSomsMade;}
            REAL EpsCorr() const {return mEpsCorr;}


            REAL Cout() const;
            REAL CorrelToCout(double aCor) const;
     private :

            int NbCurDist() const;

            REAL CoeffCorrelSymetrique() const;
            REAL CoeffCorrelIm1Maitre() const;
            REAL CoeffCorrelMedIm1Maitre() const;
            REAL CoeffCorrelMaxIm1Maitre() const;
            REAL CoeffCorrelMinIm1Maitre() const;
            REAL CoeffInfoMut() const ;
            REAL CoeffCorrelation() const;


           REAL MultiCorrelByCov() const;
           REAL Cov() const;

           const cAppliMICMAC &         mAppli;
           int                          mKCurDist;
           int                          mNbV;
           REAL                         mDefCorr;
           REAL                         mEpsCorr;
           std::vector<cStat1Distrib *> mVDistr;
           std::vector<double *>        mVData;
           double **                    mData; // [Kim][KPix]
           bool                         mSomsMade;
           eModeDynamiqueCorrel         mDynCorr;
           double                       mCorrelMin;
           double                       mGammaCor;
           eModeAggregCorr              mAggregCorr;
           std::vector<int>             mVPtInEc;
           int *                        mPtInEc;
           int                          mNbVIn;

           std::vector<int>             mVIndOK;
           int *                        mIndOK;
           int                          mNbOK;

           double                       mRatioIn;
};

class cStatGlob
{
      public :
          void AddVue(cPriseDeVue &);
          void InitSOCE();  // Initialise mStatClass
          cStatGlob
          (
                const cAppliMICMAC &,
                const std::vector<int> & aVPtInEc,
                const std::vector<int> & aVIndiceOK,
                const std::vector<Pt2di> & aVPtOK
          );
          ~cStatGlob();
          void Clear();
          REAL CorrelToCout(double aCor) const;
          REAL Cout2Correl(double aCor) const;
          REAL CoeffCorrelation() const;
          REAL Cout() const;
          cStat1Distrib * NextDistrib(const cPriseDeVue & aPDV);
          REAL EpsCorr() const {return mEpsCorr;}
          void SetSomsMade(bool);

          const bool & IsFull() const {return mIsFull;}
          const Pt2di * VPtsOK() const {return mPtsOK;}
          const int *   VIndOK() const {return mIndOk;}
          int   NbOK() const {return mNbOK;}
          

      private :
          cStatGlob(const cStatGlob&); // N.I.

          const cAppliMICMAC &      mAppli;
          std::vector<int>          mVPtInEc;
          std::vector<int>          mVIndOK;
          int *                     mIndOk;
          std::vector<Pt2di>        mVPtOK;
          Pt2di *                   mPtsOK;
          int                       mNbOK;



          std::vector<std::string>             mNamesClass;
          std::vector<int>                     mCardOfClass;
          std::vector<cStatOneClassEquiv  *>   mStatClass;
          int                                  mNbClass;
          double                               mDefCorr;
          double                               mEpsCorr;
          eModeDynamiqueCorrel                 mDynCorr;
          double                               mCorrelMin;
          double                               mGammaCor;
          bool                                 mIsFull;

};


/*****************************************************/
/*                                                   */
/*               cPriseDeVue                         */
/*                                                   */
/*****************************************************/
std::string  ExpendPattern
             (
                  const std::string & aPattern,
                  const cPriseDeVue * aIm1,
                  const cPriseDeVue * aIm2
             );

class cModGeomComp
{
     public :
        virtual ~cModGeomComp();
        virtual Pt2dr Modif2GeomInit(const Pt2dr &) const =0;
        virtual Pt2dr Modif2GeomActu(const Pt2dr &) const =0;
        double CoeffDilNonE () const;
     protected :
        cModGeomComp();
	void SetCoeffDilNonE(double);
     private :
        double mCoeffDilNonE;
};

// Gere :
//   * les noms de fichiers associes a une image a resolution donnnee
//   * les fichiers de resolution variees
//   * initialise la geometrie d'une image

class cPriseDeVue
{
        public :
            cPriseDeVue
            (
                       const cAppliMICMAC & anAppli,
                       const std::string &  aName,
                       cInterfModuleImageLoader *,
                       int   aNum,
                       const std::string & aNameGeom,
		       const std::list<cModifieurGeometrie>  &
             );
            ~cPriseDeVue();
	    
			const std::string & Name() const;
        
            // Lazy eval aussi
            cGeomImage & Geom();
            const cGeomImage & Geom() const;
            Pt2di SzIm() const;

            // Charge en RAM une zone de fichier image,
            // la zone Terrain est accedee via mAppli
            // aLT sert a calculer PXmin et PXmax pour convertir
            // l'emprise terrain en emprise image
            bool LoadImageMM (bool ForTest, const cLoadTer& aLT,
							  const Pt2di & aSzMaxGeomTer,
							  bool IsFirstLoaded);
            const cLoadedImage & LoadedIm() const;
            cLoadedImage & LoadedIm();


			cDbleGrid * ReadGridDist() const;
			ElPackHomologue ReadPackHom(const cPriseDeVue * aPDV2) const;

			std::string NamePackHom(const cPriseDeVue * aPDV2) const;

			bool IsMaitre() const;
			void SetMaitre(bool AvecImMaitre);

			cInterfModuleImageLoader * IMIL();
			const cInterfModuleImageLoader * IMIL() const;
			Box2dr BoxIm() const;
			Tiff_Im     FileImMasqOfResol(int aDz) const;
			const std::string & NameGeom() const;

			const std::string  &  NameClassEquiv() const;
			const int & NumEquiv() const;
			int & NumEquiv() ;
			int  Num() const;

			double  DzOverPredic(const Pt3dr &) const;
                        CamStenope *  GetOri() const;
                        std::string NameMasqOfResol(int aMasq) const;

        private :

           



           cGeomImage * StdGeomTerrain();

           std::string OneNameMasq(const std::list<cOneMasqueImage> & aList) const;
           Fonc_Num    FoncMasq(std::string  & aName) const;
  



       /**********************************/
       /*   Partie Data                  */
       /**********************************/

            const cAppliMICMAC &		mAppli;
            cInterfModuleImageLoader *	mIMIL;
			std::string					mName;		// Nom avec extension, sans dir
			std::string					mNameTif;	// Nom avec extension, sans dir, tjs termine par ".tif"
			std::string					mNameGeom; // Nom avec extension, sans dir
            int							mNum;
            cGeomImage *				mGeom;
            cGeomImage *				mGeomTerAssoc;  // Geometrie terrain associee
            mutable Pt2di				mSzIm; // Taille pleine resol
            cLoadedImage *				mLoadIm;
            const cEtapeMecComp*		mCurEtape;
            bool						mIsMaitre;
			std::vector<cModGeomComp *> mVModif;

            std::string                 mNameClassEquiv;
            int                         mNumEquiv;

            cElNuage3DMaille *          mNuagePredict;

            // Im2D_U_INT2                 mImZInGeom;
};

/*****************************************************/
/*****************************************************/
/*****************************************************/
/***                                               ***/
/***                                               ***/
/***                 cGeomXXX                      ***/
/***                                               ***/
/***                                               ***/
/*****************************************************/
/*****************************************************/
/*****************************************************/

/*
    Le champs inconnu recherche par le correlateur est
   une application de R2->R (mode 1-dim) ou R2->R2 (mode 2 dim).
   La technique retenue conduit a discretiser les coordonne en entree
   comme en sortie et a rechercher un champs de Z2-> Z ou
   Z2->Z2.

     Les classes cGeomDiscR2 et cGeomDiscFPx
   permettent de faire la correspondance entre 
   les coordonnée "réelles" et les coordoonnees discretisees
   en fonction des differents parametres de resolution,
   de discretisation ou de clip.

*/



/*****************************************************/
/*                                                   */
/*                  cGeomDiscR2                      */
/*                                                   */
/*****************************************************/

// Classe assurant les services  de discretisation de R2,
// c'est a dire la correspondance entre les coordonnées
// "objets" et leur homologue discret en fonction de
// de different parametres.
// 
// Ces parametres prennent  en compte la resolution finale 
// (par exemple 10 cm  en Terrain), l'emprise finale, 
// le facteur de DeZoom  (lie a l'approche multi-resolution) 
// et un eventuel cliping  (lie a une decoupe pour gestion 
//  memoire "et-ou" mise en // )

class cGeomDiscR2
{
       // defini dans GeomXXX.cpp
     public :
         cGeomDiscR2 (cAppliMICMAC &);

         // Service essentiel offert par la classe
         Pt2dr    DiscToR2(const  Pt2di & aPRas) const;
         Pt2dr    RDiscToR2(const  Pt2dr & aPRas) const;

         Pt2dr   DerX(const  Pt2di & aPRas) const;
         Pt2dr   DerY(const  Pt2di & aPRas) const;

	 // Service inverse pour mise au point
         Pt2di    R2ToDisc(const Pt2dr  & aPRas) const;
         Pt2dr    R2ToRDisc(Pt2dr  aPRas) const;

             // Modificateur
         void SetDeZoom(REAL aDz);
         void SetClip(const  Pt2di & aP0,const  Pt2di  & aP1);

         void SetClipInit();
 
             // Accesseurs

         const Pt2di & SzDz() const;
         const Pt2di & P0Clip() const;
               Pt2di   P1Clip() const;
         const Pt2di & SzClip() const;
	 const Box2dr & BoxEngl() const;
	 const Box2dr & VraiBoxEngl() const;


         const double &  DeZoom() const;

                // DEBUG

         void  Show( const std::string &) const; 
         const Pt2dr & P0() const;
         const Pt2dr & P1() const;
         REAL  ResolDz() const;
         REAL  ResolZ1() const;
     protected :


       /**********************************/
       /*   Partie Data                  */
       /**********************************/

         // Un *, et non un &, pour autoriser le X(const X&)  standard
         cAppliMICMAC *   mAp;
         // Zone Terrain et resolution initiale
         Pt2dr                  mP0;
         Pt2dr                  mP1;
         REAL                   mResol;

          // Boite calculee independamment des valeurs explicites
	  // de l'utilisateur (a des fins de mise au point)
	 Box2dr mBoxEngl;
	 Box2dr mVraiBoxEngl;

         REAL                   mDeZoom;
         // Resolution integrant le  DeZoom donne
         REAL                   mResolDz;
         // Limite de l'espace Discret pour un DeZoom
         Pt2di                  mSzDz;

         // Limite de l'espace Discret une fois clippe
         Pt2di                  mP0DiscClip;  
         Pt2dr                  mRP0DiscClip;  
         Pt2di                  mSzClip;
         Pt2dr                  mDerX;
         Pt2dr                  mDerY;

         // Pour l'instant pas utilise mais devrait a
         // terme permettre de prendre en compte l'inversion
         // des Y du mode carto. C'est eventuellement une
         // transformation affine globale qui pourra etre prise
         // en compte (par exemple pour avoir les coordonnees
         // orientees de la meme maniere que les axes du chantier)
         bool                   mInvY;
};

// Classe assurant les services  discretisation d'une
// "fonction de paralaxe" R2->R ou R2->R2
//
//  Le "Step" est relatif au  DeZoom et a la resolution
//  Par ex, en geom terrain, avec une resolution de 10cm
//  un DeZoom de 4, un Step de 0.75 correspondra a 30 cm
//  terrain.

class cGeomDiscFPx : public  cGeomDiscR2
{
     public :
       // defini dans GeomXXX.cpp

         void SetZIsAbs();

         double  CorrectDerivee() const;

         cGeomDiscFPx(cAppliMICMAC &);

         double RatioResAltiPlani() const;
         double PasPx0() const;
         double PasPxRel0() const;

         void  PxDisc2PxReel(REAL *,const int *) const;
         void  PxDisc2PxReel(REAL *,const double *) const;
         void  PxReel2PxDisc(REAL *,const double *) const;

         int GetEcartInitialPlus(double aPas,int aKPx) const;
         int GetEcartInitialMoins(double aPas,int aKPx) const;
         // void  PxReel2PxDisc(int *,const REAL *) const;  N.I.

         void SetStep(const REAL * aVStep);
         void SetDeZoom(REAL aDz);

         // L'initialisation necessite que cAppliMICMAC soit
         // deja initialise
         void PostInit();

         // Accesseur
         const REAL *  EcPxZone() const;

         const REAL *  V0Px() const;

         void RemplitOri(cFileOriMnt & aFOM,bool DoZAbs) const;

         Pt2di NbPixel() const;

// Devrait etre identiques , mais a cause de surface analytique semi triviale, pour elle : en calcul vaut 0 car le Z Moyen
// est pris dans la surface , mais a l'export il faut le remettre
         double OrigineAlti4Compute() const ;
         double OrigineAlti4Export() const ;



         double ResolutionAlti() const ;

         void SetOriResolPlani(Pt2dr & aOriP,Pt2dr & aResolP) const;


         const double * RatioResAltPlani() const
         {
              return mRatioResAltPlani;
         }
         double RoundCoord(const double & aV);
         Pt2dr  RoundCoord(const Pt2dr  & aP) ;
         Box2dr RoundCoord(const Box2dr  & aP) ;

         bool   TronkExport() const;
         double RatioPasCompUser(int aD) const;


         double  PxMin(int aK) const;
         double  PxMax(int aK) const;




     private :
         int GetEcartInitialGen(double aPas,int aKPx,double anEcart) const;

         void  SetRoundResol(double aRes);
         void  SetUnroundResol(double aRes);
         void  SetResol(double aRes,bool Round);
         // double RoundCoord(const double & aV) const;
         // double RoundCoord(const Pt2dr & aV) const;
         // double RoundCoord(const Box2dr & aV) const;



       /**********************************/
       /*   Partie Data                  */
       /**********************************/

         int    mDimPx;
         REAL   mV0Px[theDimPxMax];


         REAL   mEcPxInit[theDimPxMax];
         REAL   mEcPxInitPlus[theDimPxMax];
         REAL   mEcPxInitMoins[theDimPxMax];


         REAL   mEcPxZone[theDimPxMax];
         REAL   mStepRel[theDimPxMax];
         REAL   mStepAbs[theDimPxMax];
         REAL   mRatioPasCompUser[theDimPxMax];
         REAL   mRatioResAltPlani[theDimPxMax];
         double  mCorrectDerivee;
         cDecimal mRDec;
         bool    mRRIsInit;
         bool    mRCoordIsInit;
         bool    mTronkExport;
};

// Dans cGeomImage 
class cGeometrieImageComp
{
    public :
        cGeometrieImageComp(const cNomsGeometrieImage &,const cAppliMICMAC &);
        bool AcceptAndTransform
             (
                 const std::string & aNameTested,
                 std::string &       aNameResult,
                 int                 aNum
             );
         const cTplValGesInit< cModuleImageLoader > & ModuleImageLoader()const;
	 const std::list< cModifieurGeometrie>  & ModG() const;
    private :
        cNomsGeometrieImage    mGeom;
        const cAppliMICMAC &   mAppli;
        cElRegex *             mAutom;
        cElRegex *             mAutomI1I2;
};


typedef enum
{
   eTagGeomId,
   eTagNoGeom,
   eTagDHD_Px,
   eTagGeom_TerrainOri,
   eTagGeom_IdOri,
   eTagGeomFaisceau,
   eTagGeomFaisZTerMaitre,
   eTagGeomFaisZTerEsclave,
   eTagGeomModule,
   eTagGeomBundleGen
} eTagGeometrie;



// Gere les transformations entre la geometrie terrain
// et une image, compte-tenu d'un eventuel DeZoom et 
// d'une zone de clip (correspondant a la portion d'image
// chargee).
//
// Il s'agit d'une classe d'interface, la partie transformation
// physique entre l'espace terrain et l'image plein a resolution
// 1 ainsi que sa reciproque devront etre definies dans les derivee


struct cGeomImageData
{
         Pt2di mSzImInit;
         int   mDeZoom;
         Pt2di mSzImDz;
         Pt2di mP0Clip;
         Pt2di mP1Clip;
};


// ElDistortion22_Gen : est une class qui permet de representer
// des application du plan dans lui meme (donc des distorsion)
// cGeomImage en herite  surtout pour acceder avec la bonne interface
// au fonction Objet2ImageInit et ImageAndPx2Obj avec une Pax Nulle
//

class cGeomBasculement3D
{
    public :
        virtual Pt3dr Bascule(const Pt3dr &) const = 0;
        virtual Pt3dr BasculeInv(const Pt3dr &) const = 0;
	virtual ~cGeomBasculement3D();
    private :
};


#define PUBLIC_FOR_DEBUG public


class cGeomImage : public cGeomBasculement3D, // Pour pouvoir basculer les MNT en geometrie image
                   PUBLIC_FOR_DEBUG cGeomImageData,
                   public  ElDistortion22_Gen
{
    public :

        virtual ElCamera * GetCamera(const Pt2di & aSz,bool & ToDel,bool & aZUP) const;

    // Pour l'instant seul la geometrie conique accepte les anamorphose,
    // car cela necessite des adaptation pas encore faite pour les autres
        virtual bool AcceptAnamSA() const;
        friend class cAppliMICMAC;
        bool  UseMasqTerAnamSA();
        virtual bool IsInMasqAnamSA(Pt2dr aPTer);
        virtual double IncidTerrain(Pt2dr aPTer);

        virtual bool IsRPC() const;
        virtual Pt2dr RPCGetAltiSolMinMax() const;
        virtual bool RPCIsVisible(const Pt3dr &) const;

        virtual bool MasqImNadirIsDone();
        virtual void DoMasImNadir(TIm2D<REAL4,REAL8> &,cGeomDiscFPx &);
        std::string NameMasqImNadir(int aBK);



        std::string NameMasqAnamSA(const std::string & aPost) const;


         inline double Px1(const REAL * aPx) const {return (mDimPx > 1) ? aPx[1] : 0;}


 // A priori ce pourrait etre la fonction fondamentale, en fait elle
 // a ete rajoutee tardivement et est rarement definie pour l'instant.
 //
 // Si necessaire on peut lui rajouter une valeur par defaut correcte
 //
        virtual ElSeg3D FaisceauPersp(const Pt2dr & )  const;
        //  Les faisceau perspectif sont exprime en geom euclid, si c'est la geometrie
       // Finale (apres anam) qui est interessante, il faut encore composer par cela ...
        virtual Pt3dr GeomFP2GeomFinale(const Pt3dr & )  const;
        virtual Pt3dr GeomFinale2GeomFP(const Pt3dr & )  const;


        Pt3dr Restit2Euclid(const Pt2dr & aP,double * aPax) const;



        Pt3dr  CentreRestit() const;
        virtual bool HasCentre() const;
        virtual  Pt3dr  Centre() const;



        // Pour renseigner eventuellemet le File-Description-Chantier
	virtual bool DirEpipTransv(Pt2dr &) const;

        Pt3dr Bascule(const Pt3dr &) const; // interface pour cGeomBasculement3D, envoie de image vers terrain
        Pt3dr BasculeInv(const Pt3dr &) const; // interface pour cGeomBasculement3D, envoie de image vers terrain
    // Tient compte d'un eventuel triedre euclidien  exprime dans les
    // coordonnees de base
         Pt2dr ImageAndPx2Obj_Euclid(Pt2dr aP,const REAL * aPx) const;
         Pt2dr Objet2ImageInit_Euclid(Pt2dr aP,const REAL * aPx) const;
 
         bool GetPxMoyenne_Euclid(double * aPxMoy,bool MakeInvIfNeeded =false) const;

      // Interface en tant que ElDistortion22_Gen
           Pt2dr Direct(Pt2dr) const;
           bool OwnInverse(Pt2dr &) const ;



     // Utilisee quand on veut faire des modif temporaire
     // sur la geometrie d'une image et les revenir a l'etat
     // anterieur (pas de copie car classe abstraite)

         cGeomImageData SauvData();
         void RestoreData(const cGeomImageData &);

         // Allocateurs voir cGeomImage.cpp pour la signification
         static cGeomImage * GeomId
                             (
                                 const cAppliMICMAC &,
                                 cPriseDeVue &      aPDV,
                                 Pt2di aSzIm,
                                 int   aDimPx
                             );
         static cGeomImage * Geom_DHD_Px
                             (
                                 const cAppliMICMAC &    anAppli,
                                 cPriseDeVue &      aPDV,
                                 Pt2di                   aSzIm,
                                 cDbleGrid *             aPGr1,
                                 cDbleGrid *             aPGr2,
                                 const ElPackHomologue & aPack,
				 int                     aDim // En Gal 2, mais 1 pour des "vraies" epipolaires
                             );

	 static cGeomImage * Geom_Terrain_Ori
		             (
                                  const cAppliMICMAC & anAppli,
                                  cPriseDeVue &      aPDV,
                                  Pt2di aSzIm,
                                  CamStenope *  anOri
                             );

	 static cGeomImage * Geom_Carto_Ori
		             (
                                  const cAppliMICMAC & anAppli,
                                  cPriseDeVue &      aPDV,
                                  Pt2di aSzIm,
                                  CamStenope *  anOri
                             );

         static cGeomImage *GeomFaisZTerMaitre
                            (
                                  const cAppliMICMAC & anAppli,
                                  cPriseDeVue &      aPDV,
                                  Pt2di                aSzIm,
                                  int                  aDimPx,
                                  cGeomImage *         aGeomRef
                            );

         static cGeomImage *GeomFaisZTerEsclave
                            (
                                const cAppliMICMAC & anAppli,
                                cPriseDeVue &      aPDV,
                                Pt2di        aSzIm,
                                int          aDim,
                                cGeomImage*  aGeom,
                                cGeomImage*  aGeomRef
                            );

         


         static cGeomImage * GeomImage_Id_Ori
                             (
                                    const cAppliMICMAC & anAppli,
                                    cPriseDeVue &      aPDV,
                                    Pt2di aSzIm,
                                    int   aDim,
                                    CamStenope *  anOri,
                                    bool          Spherik
                             );
         static cGeomImage * GeomImage_Faisceau
                             (
                                    const cAppliMICMAC & anAppli,
                                    cPriseDeVue &      aPDV,
                                    Pt2di aSzIm,
                                    int   aDim,
                                    CamStenope *  anOri,
                                    CamStenope *  anOriRef,
                                    bool          isSpherik
                             );
         static cGeomImage * GeomImage_Module
                             (
                                    const cAppliMICMAC & anAppli,
                                    cPriseDeVue &      aPDV,
                                    Pt2di aSzIm,
                                    std::string const &nom_ori,
				    std::string const &nom_module,
				    std::string const &nom_geometrie
				    );
        static cGeomImage * GeomImage_Basic3D
                            (
                                    const cAppliMICMAC & anAppli,
                                    cPriseDeVue &      aPDV
                            );
	static cGeomImage * GeomImage_Grille
                             (
                                    const cAppliMICMAC & anAppli,
                                    cPriseDeVue &      aPDV,
                                    Pt2di aSzIm,
                                    std::string const &nom_ori
                                    );
        static cGeomImage * GeomImage_RTO
            (
             const cAppliMICMAC & anAppli,
             cPriseDeVue &      aPDV,
             Pt2di aSzIm,
             std::string const &nom_ori
            );
#ifdef __USE_ORIENTATIONIGN__
	static cGeomImage * GeomImage_CON
            (
             const cAppliMICMAC & anAppli,
             cPriseDeVue &      aPDV,
             Pt2di aSzIm,
             std::string const &nom_ori
            );
#endif



        static cGeomImage * Geom_NoGeomImage
		             (
                                  const cAppliMICMAC & anAppli,
                                  cPriseDeVue &      aPDV,
                                  Pt2di aSzIm
                             );

         // SetDeZoom : reinitialise la zone de clip
         void SetDeZoomIm(int aDeZoom);
         // Tient compte du clip et du zoom
         Pt2dr CurObj2Im(Pt2dr aPTer,const REAL * aPx) const;


         // Ajuste la zone de clipping pour qu'elle
         // contienne (dans la limite de la taille initiale)
         // l'image de la boite objet compte tenu des
         // intervalles  de paralaxe, 
         // et d'un peu de Rab ; ces dernier param sont fixes de
         // maniere 100% empirique,
         // disons que 50 pour le rab geom et 
         // 5 pour le rab interp  sont des valeur pas idiotes ; il doit
         // tenir compte de l'interpolateur et d'une eventuelle
         // "courbure" des fonction de projection
         // RabGeom sera divise par Zoom

         void SetClip
              (
                  Pt2dr aPMinObj,Pt2dr aPMaxObj,
                  const REAL * aMinPx,const REAL * aMaxPx, 
                  double aRabGeom,
                  double aRabInter
              );
         Box2dr BoxImageOfBoxTerrainWithContSpec
                (
                  Pt2dr aPMinObj,Pt2dr aPMaxObj,
                  const REAL * aMinPx,const REAL * aMaxPx,
                  double aRabGeom,
                  double aRabInterp
                ) const;

         // Fait appel au SetClip  du dessus en convertissant
         // la zone de clip de cGeomDiscFPx.
         // aSzVgn est une taille de vignette de correlation
         // a integrer dans la boite
         void  SetClip
               (
                   const cGeomDiscFPx &,
                   Pt2di aSzVgn,
                   const int * aMinPxDisc,
                   const int * aMaxPaxDisc,
                   double aRabGeom,
                   double aRabInter
               );
         
         // Un peu redondant avec le SetClip precedent, indique
         // si la boite terrain trouve un intersection non vide
         // avec l'emprise image
         bool BoxTerHasIntersection
         (
              const cGeomDiscFPx & aFPx,
              const int * aMinPxDisc,const int * aMaxPxDisc,
              Box2dr aBoxTerRas
         ) const;


         virtual std::vector<Pt2dr>  EmpriseImage() const ;
         Box2dr EmpriseTerrain
                (
                     const REAL * aMinPx,const REAL * aMaxPx,
                     double aRab
                )  const;
         // Rapelle celle du dessus en inversant les signes
         virtual ~cGeomImage();

        // Geometrie sous jacent aux geometries faisceaux
         virtual const cGeomImage * GeoTerrainIntrinseque() const;
         cGeomImage * NC_GeoTerrainIntrinseque() ;
        
         // Indique si  l'intersection entre la zone
         // image globale et le clip terrain est vide
         bool ClipIsEmpty() const;
         Box2di BoxClip() const;

         int DimPx() const;
	 double CoeffDilNonE() const;


         // indique si le ratio est rempli
         virtual bool GetRatioResolAltiPlani(double * aRatio) const;
         // Resolution moyenne par defaut de la geometrie
         double GetResolMoyenne_Euclid() const;
         double GetResolNonEuclidBySurfPixel() const;

         // Par defaut ne fair rien
          virtual void RemplitOri(cFileOriMnt & aFOM) const;
         // Par defaut erreur fatale si pas mode Image_Nuage
         virtual void RemplitOriXMLNuage(bool CallFromMere,const cMTD_Nuage_Maille &,const cGeomDiscFPx & aGT,cXML_ParamNuage3DMaille &,eModeExportNuage) const;

         //  Par defaut return 0
           virtual CamStenope *  GetOri() const ;
         // Appelle GetOri, genere une erreure si pas d'Ori
           CamStenope *  GetOriNN() const ;

          // Renvoie la fonction comme une distorsion

          // Partie de l'initialisation qui ne peut etre faite que quand le
          // GeomDFPxInit() de Appli est lui meme initialise (il
          // utilise les images ....)
          void PostInit();

          // Post initialisation "immediate" prend en  compte les appels de methodes
	  // virtuelles qui ne peuvent etre faite dans le constructeur
	  void PostInitVirtual(const std::vector<cModGeomComp *> &);


         const cElPolygone        &  PolygTerPx0() const;
         const Box2dr             &  BoxTerPx0() const;
         const std::vector<Pt2dr> &  ContourIm() const;
         const std::vector<Pt2dr> &  ContourTer() const;
         bool  IntersectEmprTer(const cGeomImage &,Pt2dr & aPMoyEmpr,double * aSurf=0) const;
         double BSurH(const cGeomImage&,const Pt2dr&aPTer,double  aZ) const;
         double BSurH(const cGeomImage&,const Pt2dr&aPTer) const;
 
         virtual bool IsId() const;

         // Si P1 et P2 sont des point hom a px nulle il
         // auront la meme valeur apres CorrigeModeleCple,
         // de plus cela est fait dans une geometrie intermediaires
         // telle que la px residuelle soit le plus proche possible d'une
         // homographie
         virtual Pt2dr CorrigeDist1(const Pt2dr & aP1) const;
         virtual Pt2dr CorrigeDist2(const Pt2dr & aP2) const;
         virtual Pt2dr InvCorrDist1(const Pt2dr & aP1) const;
         virtual Pt2dr InvCorrDist2(const Pt2dr & aP2) const;

         virtual void CorrectModeleAnalytique(cModeleAnalytiqueComp *);

         //
         //  La fonction est telle que 
         //      Objet2ImageInit(P1,P1P2ToPx(P1,P2)) = P2
         //  Par defaut elle est calculee par inversion iterative
         //  mais elle peut si necessaire etre redefinie de maniere
         //  plus efficace
         // 

         virtual Pt2dr P1P2ToPx(Pt2dr aP1,Pt2dr aP2) const;
	 virtual std::string Name() const;
         virtual void InitAnamSA(double aResol,const Box2dr &  aBoxTer);
   protected :

        virtual void InstPostInit();
         bool GetPxMoyenneNulle(double * aPxMoy) const;
         // Fonction generale qui calcule l'image d'une boite
         // englogante, en tenant compte d'un intervalle de para

         Box2dr BoxImageOfVPts
                (
                     const std::vector<Pt2dr> &,
                     REAL aZoom,bool isSensTer2Im,
                     const REAL * aMinPx,const REAL * aMaxPx,
                     double aRab
                )  const;

         Box2dr BoxImageOfBox
                (
                     Pt2dr aPMin,Pt2dr aPMax,
                     REAL aZoom,bool isSensTer2Im,
                     const REAL * aMinPx,const REAL * aMaxPx,
                     double aRab
                )  const;


         cGeomImage
         (
             const cAppliMICMAC &,
             cPriseDeVue &      aPDV,
             eTagGeometrie     aModeGeom,
             Pt2di aSzIm,
             int   aDimPx
         );
         const cAppliMICMAC & mAppli;
         cPriseDeVue &        mPDV;
         cInterfSurfaceAnalytique * mAnamSA;
         const cChCoCart *          mRC;
         const cChCoCart *          mRCI;
         int                        mAnDeZoomM;

         bool                       mAnamSAIsInit;
         bool                       mUseTerMasqAnam;
         bool                       mDoImMasqAnam;


         cParamMasqAnam             mAnamSAPMasq;
// Parametres utilises pour seuiles sur l'incidences
         Im2D_INT1                  mMTA;
         TIm2D<INT1,INT>            mTMTA;

// Parametres utilises pour ordonner sur l'incidence et prendre les K Meilleur Nadir
         double                     mDynIncidTerr;
         TIm2D<INT2,INT>            mTIncidTerr;

         const int            mDimPx;
         eTagGeometrie        mModeGeom;

         cElPolygone          mPolygTerPx0;
         Box2dr               mBoxTerPx0;
         std::vector<Pt2dr>   mContourIm;
         std::vector<Pt2dr>   mContourTer;
	 bool                 mPIV_Done;
	 bool                 mIsIntrinseque;
    // protected :

        // A priori Micmac privilegie le calcul en geometrie terrain,
	// dans ce cas TerrainRest2Euclid renvoie l'identite
	//
	//  Si ce n'est pas le cas , par exemple en geometrie image
	//  alors on cette fonction permet de passer de la restite au
	//  terrain "reel"
	//

    private :

        virtual Pt3dr TerrainRest2Euclid(const Pt2dr & aP,double * aPax) const;



	 std::vector<cModGeomComp *> mVM;
	 double mCoeffDilNonE;

       /**********************************/
       /*   Partie Data                  */
       /**********************************/

    // Ne tient compte d'un eventuel triedre euclidien  exprime dans les
    // coordonnees de base
         // Fonction de localisation "inverse", donne un point
         // "objet" a partir d'un point image et d'une paralaxe,
         //  ne tient compte  ni du DeZoom ni du ClipImage,
         //  utilise uniquement pour definir les limites de l'espace
         //  "objet" a partir des images
         virtual Pt2dr ImageAndPx2Obj_NonEuclid(Pt2dr aP,const REAL * aPx) const=0;

         // Fonction "fondamentale", a definir dans les derivees
         //  ne tient compte  ni du DeZoom ni du ClipImage
// public :
         virtual Pt2dr Objet2ImageInit_NonEuclid(Pt2dr aP,const REAL * aPx) const=0;

         // Indique si la geometrie connait sa Px moyenne (cas altisol
         // des ori, si true rempli le aPxMoy; Defaut renvoie false
         virtual bool GetPxMoyenne_NonEuclid(double * aPxMoy,bool MakeInvIfNeeded) const;
         virtual double GetResolMoyenne_NonEuclid() const = 0;
};


/*****************************************************/
/*                                                   */
/*               cLoadedImage                        */
/*                                                   */
/*****************************************************/

class cMSLoadedIm
{
      public :
        cMSLoadedIm(const cOneParamCMS& ,Im2D_REAL4 *,bool First);
        const Im2D_REAL4 * Im() const;
        Im2D_REAL4 * Im();

      private :
         cOneParamCMS        mImCMS; 
         Im2D_REAL4         mIm;
         TIm2D<REAL4,REAL8> mTIm;
};

class cLoadedImage
{
    public:
      static cLoadedImage * Alloc
                            (
                                const cAppliMICMAC &,
			        const cPriseDeVue &,
                                const cGeomImage & aGeom,
                                Box2di   aSzIm,
                                const Pt2di &   aSzMaxInGeomTer,
                                cInterfModuleImageLoader * aIMIL,
                                int                        aDZ,
                                Fonc_Num                   aFMasq,
				bool          IsFirstLoaded
                            );
      virtual ~cLoadedImage();
      virtual void GetVals(const Pt2dr * aP,double *,int aNb) const = 0;
      inline bool IsOk(const Pt2di & aP)  const
      {
         return (mTMasqIm.get(aP,0) != 0);
      }
      inline bool IsOk(const Pt2dr & aP) const
      {
         return IsOk(Pt2di(aP));
      }

      virtual void LoadImInGeomTerr(Box2di aBox,int * aPx,bool FirstIm) =0;
      virtual void PostLoadImInGeomTerr(Box2di aBox) =0;
      virtual void AddToStat(cStatGlob &,Pt2di aPTer)=0;
      virtual void CalcCorrelRapide (const Box2di &,cLoadedImage &)=0;
      virtual void   NCalcCorrelRapide(const Box2di &,tContPDV&)=0; 
      virtual void   CalcFenSpec(const Box2di &,tContPDV&)=0; 


      //virtual void CalcRapCorrelIm1Maitre
      //     (const Box2di & aBox,tContPDV & aCont)=0;

      virtual bool StatIm1(cLineariseProj &,double &aS1,double & aS11,double * aTabV1) =0;
      virtual bool StatIm2(cLineariseProj &,double * aTabV1,double &aS2,double & aS22,double & aS12) =0;


      Im2D_Bits<1>       MasqImTer();
      const cAppliMICMAC & Appli();
 
      eIFImL_TypeNumerique TypeOfElem() const;
      Pt2di   QuickTer2Cor(const Pt2di & aP)  const
      {
             return aP+mDecalTer2Cor;
      }
      Pt2di   QuickCor2Ter(const Pt2di & aP)  const
      {
             return aP-mDecalTer2Cor;
      }
      const Pt2di & SzPtWMarge() const;
      const Pt2di & SzPtWFixe() const;
      const Pt2di SzIm() const;
      virtual Fonc_Num FIm() = 0;

      void MakePC
           (
               cPriseDeVue &,
               cEtapeMecComp * aCur,
               cEtapeMecComp * aPred,
               const Box2di aBoxTer,
               bool         doPC,
               bool         doMasqAnam

           );
         bool  IsVisible(const Pt2di & aP) const
         {
               return (!mUsePC) || (mTImPC.get(aP,mSeuilPC) <mSeuilPC);
         }
         bool     UsePC() const;
         U_INT1** DataImPC() const;
         int      SeuilPC() const;
         U_INT1** DataMasqIm() const;
         U_INT1** DataMasqImErod() const;
         void DoMasqErod(const Box2di & aBox);

         virtual float *** DataFloatIm()  = 0;
         virtual float ** DataFloatLinIm()  = 0;
       
         virtual  Im2D_REAL4 ** FloatIm()  = 0;

         virtual  Im2D_REAL4 * FirstFloatIm()  = 0;
         const std::vector<cMSLoadedIm>&  MSLI();
         std::vector<Im2D_REAL4>   VNpIm();  // Non pointeur

    protected :
      cLoadedImage
      (
          const cAppliMICMAC &,
          const cPriseDeVue &,
          const cGeomImage &,
          Box2di   aBoxIm,
          const Pt2di &   aSzMaxInGeomTer,
          cInterfModuleImageLoader * aIMIL,
          int                        aDZ,
          Fonc_Num                   aFMasq,
	  bool                       IsFirstLoaded,
	  const eIFImL_TypeNumerique       aType
      );

         void PostInit();

         Pt2di DiscTerAppli2DiscTerCorr(const Pt2di  &aPt);
      // Encapsule le fait qu'il y a une simple translation entre les deux

         const cPriseDeVue &  mPDV;
         const cGeomImage &   mGeomI;
         const cAppliMICMAC & mAppli;
         cLoadTer *     mLT;
         const cEtapeMecComp* mCurEt;
         Pt2di          mSzIm;
         Im2D_Bits<1>   mMasqIm;
         TIm2DBits<1>   mTMasqIm;

         bool           mDoneMasqErod;
         Im2D_Bits<1>   mMasqImErod;
         TIm2DBits<1>   mTMasqImErod;

         // Geometrie de l'image superposable au bloc terrain,
         // a une translation pres, pour une paralaxe donnee
         Pt2di          mSzPtWFixe;
         Pt2di          mSzPtWMarge;
         const cGeomDiscFPx &  mGeomTerAp;
         cGeomDiscFPx   mGeomCorrelImTer;
         Pt2di          mSzCImT;
         
     // Ca c'est un masque qui est calcule a Px donnee
         Im2D_Bits<1>       mMasqImTer;
         TIm2DBits<1>       mTMasqImTer;
         Pt2di              mDecalTer2Cor;

         eIFImL_TypeNumerique     mTypeEl;
         // eTypeNumerique     mTypeEl;

         bool               mUsePC;
         Im2D_U_INT1        mImPC;
         TIm2D<U_INT1,INT>  mTImPC;
         int                mSeuilPC;

         std::vector<cMSLoadedIm>  mMSLI;
         std::vector<Im2D_REAL4 *> mVIm;
         std::vector<Im2D_REAL4>   mVNpIm;  // Non pointeur
         std::vector<float **>     mVDataIm;
         std::vector<float *>      mVDataLin;

    private:

};



/*****************************************************/
/*                                                   */
/*           cArgOneEtapePx                          */
/*           cFilePx                                 */
/*           cEtapeMecComp                           */
/*                                                   */
/*****************************************************/




struct cOneNappePx  // dans cFilePx.cpp
{
     public :

         void TestDebugOPX(const std::string& aMes);

         double  ResultPx(const Pt2dr & aP,bool isOptCont);

         void CalculBornesPax(Box2di aBox,INT & aVmin,INT & aVMax);

         cOneNappePx(int aKPx,Pt2di aSz);
         // Calcul d'une nappe DeZoomee 
         cOneNappePx
         ( 
               TIm2DBits<1> & aMasqDZ,  // Au passage on fait la
              // reduction des masques
               const cOneNappePx &, 
               const TIm2DBits<1> & aMasqInit,
               const cEquiv1D & anEqX,
               const cEquiv1D & anEqY
         );
         void ForceConnexions();
         void ComplWithProj32(const cResProj32 & aRP32);


         int  GetPxMin(const Pt2di & aP)  const
         {
            return mTImPxMin.get(aP);
         }
         int  GetPxMax(const Pt2di & aP) const
         {
            return mTImPxMax.get(aP);
         }
         bool OkPx(const Pt2di & aP,int aPx) const;

         REAL Redressement(const Pt2di & aP) const
         {
             return mRedrPx ?  mTPxRedr.getproj(aP) : 0.0;
         }
         REAL Redressement(const Pt2dr & aP) const
         {
             return mRedrPx ?  mTPxRedr.getprojR(aP) : 0.0;
         }

    //  Data
         Pt2di      mSz;
         Im2D_INT2  mPxRes;  // resultat
         TIm2D<INT2,INT> mTPxRes;
         Im2D_INT2  mPxInit;  // resultat
         TIm2D<INT2,INT> mTPxInit;

         Im2D_INT2  mImPxMin;
         TIm2D<INT2,INT> mTImPxMin;
         Im2D_INT2  mImPxMax;
         TIm2D<INT2,INT> mTImPxMax;

         // On ne tient pas compte de mNPx0
         double  FromDiscPx(double aDisc) const
         {
              return aDisc*mDPx + mNPx0;
              //  return mNPx0+aDisc*mDPx;
         }
         double  ToDiscPx(double aDisc) const
         {
              return   ((aDisc)-mNPx0)/mDPx;
         }
         double     mNPx0;
         double     mDPx;
         int        mIMoyPx;
         double     mRMoyPx;
         Im2D_REAL4         mPxRedr;
         TIm2D<REAL4,REAL8> mTPxRedr;

         int        mVPxMin;
         int        mVPxMax;
  // Quand il y a redressement, les valeurs de nappes sont
  // constantes [-ZDilatAlti,+ZDilatAlti], cependant il est prudent  
  // de  conserver les valeur avant redr pour le calcul d'intervalle
         int        mVPxMinAvRedr;
         int        mVPxMaxAvRedr;

         int        mKPx;
         bool       mRedrPx;

         Video_Win * mW;
         bool        mForDeZ;
};

class cLoadTer // dans cFilePx.cpp
{
    public :
      cLoadTer(int aNbPx,Pt2di aSz,const cEtapeMecComp &);
      cLoadTer(
               const cLoadTer &aLT,
               const cEquiv1D & anEqX,
               const cEquiv1D & anEqY
      );
      cOneNappePx & KthNap(int aK);
      const cOneNappePx & KthNap(int aK) const;
      int NbPx() const;
      const int * PxMin() const;
      const int * PxMax() const;
      int * PxMin() ;
      int * PxMax() ;
      // Masque utilisateur + emprise
      Im2D_Bits<1> ImMasqSsPI() const;
      // Avec prise en compte eventuelle du masque point d'interet pour la
      // correlation "creuse" (type point de liaison)
      Im2D_Bits<1> ImMasqTer() const;
      bool IsInMasq(const Pt2di& aP) const
      {
          return (mTImMasqTer.get(aP) != 0);
      }
      bool IsInMasqOfPx(const Pt2di& aP) const
      {
          return (mTImMasqTerOfPx.get(aP) != 0);
      }
      void CalculBornesPax(Box2di aBox,int * aVMin,int *aVMax);
      void GetBornesPax(const Pt2di & aPt,int * aVMin,int *aVMax) const;
      Pt2di Sz() const;
     
      bool OkPx(const Pt2di & aP,const int * aPx) const;
      void MakeImTerOfPx(Box2di aBox,int * aPx);

      Im2D_REAL8  ImCorrel() const ;
      Im2D_U_INT1  ImCorrelSol () const;
      void Redressement(double *,const Pt2di & aP) const;
      void Redressement(double *,const Pt2dr & aP) const;

      Im2D_Bits<1> & MasqGlob() {return mMasqGlob;}
      Im2D_REAL8 & SomMasqI1I2() {return mSomMasqI1I2;}
      Im2D_REAL8 & Som1() {return mSom1;}
      Im2D_REAL8 & SomI1I2() {return mSomI1I2;}
      Im2D_REAL8 & SomCorrel() {return mSomCorrel;}
      Im2D_REAL8 & SomPdsCorr() {return mSomPdsCorr;}
      void SetDefCorrActif(const Pt2di & aP,int aVal);
      bool  IsDefCorrActif(const Pt2di  &aP) const;
      Im2D_Bits<1>    ImOneDefCorr();

      INT2 ** GPULowLevel_ZMin() const;
      INT2 ** GPULowLevel_ZMax() const;
      U_INT1 ** GPULowLevel_MasqTer() const;
    private :
       std::vector<cOneNappePx> mPxs;
       int             mNbPx;
       Pt2di           mSz;

       Im2D_Bits<1>    mImMasqSsPI;  // Sans les points d'interets

       Im2D_Bits<1>    mImMasqTer;
       TIm2DBits<1>    mTImMasqTer;
       //  Une paralaxe courante aPx etant fixee,
       // contient les points qui sont d'une part dans mImMasqTer et d'autre
       // part pour lesquels aPX est dans les envellope PxMin PxMax
       Im2D_Bits<1>    mImMasqTerOfPx;
       TIm2DBits<1>    mTImMasqTerOfPx;

    // Memorise si un pixel a ete mis a def corr
       Im2D_Bits<1>    mImOneDefCorr;
       TIm2DBits<1>    mTOneDefCorr;


       Im2D_REAL8      mImCorrel;
       TIm2D<REAL8,REAL8> mTImCorrel;
       int             mPxMin[theDimPxMax];
       int             mPxMax[theDimPxMax];

       Im2D_U_INT1     mImCorrelSol;
       // Contient l'intersection de tout les masques image qui ont ete
       // calcules pour une paralaxe donnee, utile pour les fenetres
       // exponentielles qui necessitent que tous le monde ait le meme
       // masque
       Im2D_REAL8     mSomPdsCorr;
       Im2D_REAL8     mSomCorrel;
       Im2D_REAL8     mSomI1I2;
       Im2D_REAL8     mSomMasqI1I2;
       Im2D_REAL8     mSom1;
       Im2D_Bits<1>   mMasqGlob;
};

// Classe support pour faciliter la lecture
// de Regul, Step, DilatAlti, DilatPlani
struct cArgOneEtapePx  
{
       public :

          REAL  mRegul;
          REAL  mRegul_Quad;
          REAL  mUserPas;  // Nom Change pour verifier usage
          REAL  mComputedPas;  // Nom Change pour verifier usage

          int   mDilatAltiPlus;
          int   mDilatAltiMoins;

          int   mDilatPlani;
          bool  mRedrPx;
          bool  mDeqRedrPx;
          double  mDilatPlaniProp;

          int  mIncertPxPlus;
          int  mIncertPxMoins;
         
};

         // cFilePx 

class cFilePx : public cArgOneEtapePx
{
	public :
                void RemplitOri(cFileOriMnt &) const;
		cFilePx
                (
		     const cArgOneEtapePx & anArg,
                     const cAppliMICMAC & anAppli,
		     cEtapeMecComp &        anEt,
                     cFilePx *              aPredCalc,
                     bool                   isPseudoFirst,
                     const std::string &    aPost,
                     int                    aKPx
                );
                // Separe du constructeur  car necessite que le discretiseur de Geometrie de l'etape
                // soit initialise
		void CreateMNTInit();

                void  SetCaracOfZoom(cCaracOfDeZoom &);
		// Renvoie true si un calcul est necessaire, c'est
		// a dire mDilatAlti ou mDilatPlani sont != 0
		bool NappeIsEpaisse() const;
		bool IsFirtOrNappeIsEpaisse() const;
                bool GenFile() const;

                void Show(const std::string  & aMes) const;  // Debug/ MaP

                // ACCESSEURS
                double UserPas() const;
                double ComputedPas() const;

                double Regul() const;
                double Regul_Quad() const;

		int DilatAltiPlus() const;
		int DilatAltiMoins() const;

		int DilatPlani() const;
                bool RedrPx() const;
	        const std::string& NameFile() const;
                // Initialise, sur la boite BoxIn, le contenu
                // des nappes englobantes en :
                //    -  allant lire le fichier de resolution
                //      inferieur (ou egale);
                //     - effectuant le changement d'echelle
                //     - operant les dilatations alti et plani
                //       qui vont bien
                void LoadNappeEstim 
                     (
                          const cEtapeMecComp &,
                          const cResProj32 &,
                          Im2D_Bits<1>,
                          cOneNappePx &,
                          Box2di aBoxIn
                     );
                void SauvResulPx (Fonc_Num,Box2di aBoxOut,Box2di aBoxIn);
                void SauvResulPxRel 
                     (Im2D_INT2,Im2D_INT2 ImInit,Box2di aBoxOut,Box2di aBoxIn);
                void ForceGenFileMA();
               Tiff_Im      FileIm() const;
               Tiff_Im      FileImRel() const;
	       std::string  NameFileSsDir() const;

                void InitComputedPas(double aRatio);


               Tiff_Im      FileIm(bool & IsCreated) const;
	private :
               std::string  NameFileGen(const std::string & aCompl);
	       std::string  NameFileGenSsDir(const std::string & aCompl) const;


	       //  Fichier a partir duquel il faut 
	       //  faire le calcul predictif
               const cAppliMICMAC & mAppli;
               cEtapeMecComp & mEtape;
               cFilePx *     mPredCalc;
               // Entier car DeZoom sont puis de 2
               int           mRatioDzPrec;
               REAL          mRatioStepPrec;
               bool          mIsPseudoFirst;
	       std::string   mNamePost;
	       std::string   mNameFile;
	       std::string   mNameFileRel; // Fichier relatif a la Px Init
	       Pt2di         mSzFile;
	       bool          mGenFile;  // Faut il generer un fichier
	       bool          mGenFileMA;  // Faut il generer un fichier uniquement
                                          // pour le modele analytique (et donc
                                          // a ne pas reutiliser en prediction)
               bool          mGenFileRel;  // Generation de l'ecart / a la prediction
                                           // Debug purpose

                int          mKPx;
};

         //        cCaracOfDeZoom


class cCaracOfDeZoom
{
     public :
          cCaracOfDeZoom
          (
                int aDeZoom,
                cCaracOfDeZoom * mDeZoomInf, 
                cAppliMICMAC &
          );
          int DeZoom() const;
          const cEquiv1D & EqX() const;
          const cEquiv1D & EqY() const;
          bool  HasMasqPtsInt() const;
   // Ratio (donc <= 1) des points etant d'interet
          double RatioPtsInt() const;
          const std::string & NameMasqInt() const;

     private :
          void MakeMasqueInteret();
          void MakeMasqueInteret
               (
                   const cInterv1D<int> & aIntInt,
                   const cInterv1D<int> & aIntOut
               );

          cAppliMICMAC & mAppli;
          int mDeZoom;
          Pt2di mSzIm;
          cCaracOfDeZoom * mDZI;
          bool     mHasMasqPtsInt;
          cEquiv1D mEqX;
          cEquiv1D mEqY;
          cVectTr<int> mLutX;  // Corresp / a l'eventuel mDZI
          cVectTr<int> mLutY;
          std::string mNameMasqInt;
          double  mRatioPtsInt;
};


/*
class  cCompileNuagePredicteur
{
    public :
        cCompileNuagePredicteur 
        (
            const cNuagePredicteur &,
            cAppliMICMAC &,
            cEtapeMecComp &
        );

        
    private :
        cNuagePredicteur  mNP;
        cAppliMICMAC &    mAppli;
        cEtapeMecComp &   mEtape;
};
*/
         //        cEtapeMecComp                

class cEtapeMecComp
{
       public:
          int  MultiplierNbSizeCellule() const;
          std::string NameMasqCarteProf() const;

          void RemplitOri(cFileOriMnt &) const;
          cXML_ParamNuage3DMaille DoRemplitXML_MTD_Nuage() const;
          void DoRemplitXMLNuage() const;
          cXML_ParamNuage3DMaille DoRemplitXMLNuage(const cMMExportNuage &) const;
          void RemplitXMLNuage(const cTplValGesInit<cMTD_Nuage_Maille> &,cXML_ParamNuage3DMaille &,eModeExportNuage) const;

          const std::string &  NameXMLNuage() const;

          ~cEtapeMecComp();
	  cEtapeMecComp
          (
               cAppliMICMAC & anAppli,
               cEtapeMEC &    anEtape,
	       bool                 isLastEtape,
	       const cGeomDiscFPx & aGeomTerrain,
               const tContEMC &     aVEtPrec
          );
          void CreateMNTInit();
      // Accesseur
          Pt2di SzFile() const;
          INT   Num()    const;
          bool  IsLast() const;
          int   DeZoomTer() const;
          int   DeZoomIm() const;
          const cGeomDiscFPx &  GeomTer() const;
  // GeomTerFina tient compte de l'eventuel ExportZAbs qui a ete rajoute a posteriori
          cGeomDiscFPx   GeomTerFinal() const;
          cGeomDiscFPx &  GeomTer() ;

          void Show() const;  // Debug/ MaP
          const cFilePx & KPx(int aK) const;
          // Taille , approximative, prise par l'algo
          // par pixel image (le nombre eventuel de cellule 3D
          // sera estime a partir des Dilatations)
          int MemSizePixelAlgo() const;
          const cEtapeMEC &   EtapeMEC() const;

          void SauvNappes (cLoadTer &,Box2di aBoxOut,Box2di aBoxIn);

          // retourne le nombre de cellules creees
          double LoadNappesAndSetGeom
               (
                   cLoadTer & aVNappes ,
                   Box2di aBoxIn
               );

          Fonc_Num FoncMasqIn(bool ForceReinj=false);



          int SsResAlgo() const;
          const cFilePx & KThPx(int aK) const;
          int   NumSeuleNapEp() const;
          int   NbNappesEp() const;
          bool KthNapIsEpaisse(int aK) const;

          bool SelectImage(cPriseDeVue * ) const;
          int  SzGeomDerivable() const;
          bool UseGeomDerivable() const;

          void ExportModelesAnalytiques();
          
          cModeleAnalytiqueComp * ModeleAnImported() const;
          bool  PxAfterModAnIsNulle() const;
          bool  GenImageCorrel() const;

          const cArgMaskAuto &  ArgMaskAuto() const;
          const cEtiqBestImage *           EBI() const;

          Tiff_Im  FileMaskAuto() const;
          Tiff_Im  FileMask3D2D() const;
          Tiff_Im  FileCorrel() const;
          Tiff_Im  FileRes( GenIm::type_el,const std::string &  aPref,bool NoTile=false) const;
          Tiff_Im  LastFileCorrelOK() const;
          std::string NameFileCorrel() const;
          std::string NameFileRes(const std::string & aPref ) const;
          eAlgoRegul    AlgoRegul() const;

          void SetCaracOfZoom();
          cCaracOfDeZoom &  CaracOfZ();
          bool  IsOptDiffer() const;
          bool  IsOptDequant() const;
          bool  IsOptIdentite() const;
          bool  IsOptimCont() const;
          bool  IsOptimReel() const;
          bool  IsExportZAbs() const;

          // aP0 - aP1 : definit la zone a basculer
	  void AllBasculeMnt(Pt2di aP0,Pt2di aP1,float **,INT2 **,Pt2di aSzData);
	  void OneBasculeMnt(Pt2di aP0,Pt2di aP1,cBasculeRes &,
	                     float **,INT2 **,Pt2di aSzData);

          // Masque utile a cause du OneDefCorIsAllDefCorr
          Tiff_Im FileMasqOfNoDef() ;

	  Pt2dr  ProjectionInImage
	          (const cGeomImage &,cLoadTer &,Pt2dr aP);

          void SauvProjImage
	       (
	           const cGenerateProjectionInImages&,
		   const cPriseDeVue &,
                   cLoadTer & aLT,
                   Box2di      aBoxOut,
                   Box2di      aBoxIn
               );


	  void InitPaxResul(cLoadTer & aLT,const  Pt2dr & aP,double * aPx);  // Px reelles

	  cEtapeMecComp *  PredPC() const; // Derniere etape avec parties cachees
          bool             UsePC() const;
          bool                     IsNewProgDyn() const;
          bool                     HasMaskAuto() const;
          const cModulationProgDyn *  TheModPrgD() const;
          const cEtapeProgDyn *       TheEtapeNewPrgD() const; 
          cInterpolateurIm2D<float> * InterpFloat() const;
          bool        MATP() const;
          bool        UseWAdapt() const;
       private:
	  void VerifInit(bool,const std::string &);
	  static cFilePx * GetPred(const tContEMC &,int anInd);
	  void  InitModeleImported(const tContEMC &);

	  int MemSizeCelluleAlgo() const;
	  int MemSizePixelSsCelluleAlgo() const;

          cAppliMICMAC &    mAppli;
          cEtapeMEC &       mEtape;
          int                     mDeZoomTer;
          int                     mDeZoomIm;
	  bool                    mIsLast;
	  int                     mNum;
	  cGeomDiscFPx            mGeomTer;
          Pt2di                   mSzFile;
	  std::vector<cFilePx *>  mFilesPx;
          int                     mSsResAlgo;
          int                     mNumSeuleNapEp;
          int                     mNbNappesEp;

          std::vector<std::string> mSelectByRel;
          bool                    mPatternModeExcl;
          std::list<cSetName *>   mListAutomDeSel;
          int                     mSzGeomDerivable;
          bool                    mUseGeomDerivable;
          cModeleAnalytiqueComp * mModeleAnToReuse;
          cModeleAnalytiqueComp * mModeleAnImported;
          bool                    mMAImportPredImm;
          std::list<cModeleAnalytiqueComp*> mModelesOut;
          eAlgoRegul              mAlgoRegul;
          cCaracOfDeZoom *        mCaracZ;
          bool                    mIsOptDiffer;
          bool                    mIsOptDequant;
          bool                    mIsExportZAbs;
          bool                    mIsOptIdentite;
          bool                    mIsOtpLeastSQ;
          bool                    mIsOptimCont;
          bool                    mIsOptimReel;
          bool                    mGenImageCorrel;
	  cEtapeMecComp *         mPrec;
	  cEtapeMecComp *         mPredPC; // Derniere etape avec parties cachees
	  cEtapeMecComp *         mPredMaskAuto; // Derniere etape a meme resol avec Mask (peut etre soit meme)
          bool                    mUsePC;

          bool                    mIsNewProgDyn;
          const cArgMaskAuto *       mArgMaskAuto;
          const cEtiqBestImage *           mEBI;
          const cModulationProgDyn * mTheModPrgD;
          const cEtapeProgDyn *      mTheEtapeNewPrgD; 
          mutable cInterpolateurIm2D<float> * mInterpFloat;
          bool                                mMATP;
          bool                                mUseWAdapt;
          mutable std::string                 mNameXMLNuage;
};

/*****************************************************/
/*                                                   */
/*               cLineariseProj                      */
/*                                                   */
/*****************************************************/


// Classe permettant de manipuler la linearisation d'une projection
// Pour l'instant optimisee pour offrir le service du parcour
// d'un rectangle
class cLineariseProj
{
     public :
         void InitLP
              (
        // rectangle sur lequel il faut "optimiser" la linarisation
                   Box2di               aRasBoxTer,  
                   int *                aPxI,
                   const cGeomImage &   aGeomIm,
                   const cGeomDiscFPx & aGeomTer,
                   const cEtapeMecComp& anEtape,
                   const cLoadTer &     aLT,
                   const Pt2di &        aDecalRedr,
                   int   aSurEchW
              );
          const Pt2dr & PCurIm() const {return mCurIm;}
          Pt2di  PCurRasTer() const {return mCurRas + mP0Ras;}
          Pt2dr ProjExacte(const Pt2dr & aP) ;
          Pt2dr ProjExacte(const Pt2di & aP) ;
          bool  Continuer() const;
          void  NexStep();

      private :
          bool   mUseDer;
          int    mDimPx;
          Pt2dr  mDerX;
          Pt2dr  mDerY;
          Pt2dr  mCentre;
          Pt2dr  mDebLigne;
          Pt2di  mSzRas;
          Pt2di  mP0Ras;
          Pt2dr  mCurIm;
          Pt2di  mCurRas;
          double mRPx[theDimPxMax];
          const cGeomImage *   mGeomIm;
          const cGeomDiscFPx * mGeomTer;
          const cLoadTer *     mLT;
          Pt2di                mDecalRedr;
};

/*****************************************************/
/*                                                   */
/*         cModeleAnalytiqueComp                     */
/*                                                   */
/*****************************************************/

// Classe utilisee pour recalculer des coefficient de correlation,
// multi-echelle a partir œ
//


class cMC_PairIm
{
    public :
       cMC_PairIm(int aDZ,const cGeomDiscFPx &,cPriseDeVue &,cPriseDeVue &);

       double Correl(Pt2dr aPTER,int aSzW,double * aVPx);


   private :
       int            mDZ;
       double         mResolZ1;
       cPriseDeVue  & mPDV1;
       cPriseDeVue  & mPDV2;
       Im2D_REAL4 mI1;
       TIm2D<REAL4,REAL8>  mTI1;
       Im2D_REAL4 mI2;
       TIm2D<REAL4,REAL8>  mTI2;
};



class cMatrCorresp  // Dans cModeleAnalytiqueComp.cpp
{
  public :
    cMatrCorresp(cAppliMICMAC &,Pt2di aSz,int aNBXY2);
    Output  StdHisto();
    // La dist aNormpt a pour objet de ramener les coordonnes dans 0-1
    void  Normalize
          (
              const cOneModeleAnalytique &      aModele,
              const cGeomDiscFPx &,
              cPriseDeVue & aPDV1,
              const cGeomImage & aGeom1,
              cPriseDeVue & aPDV2,
              const cGeomImage & aGeom2
          );
 
    bool IsOK(const Pt2di & aPRas) const;
    Pt2dr P1 (const Pt2di & aPRas) const;
    void SetXY2(const Pt2di &,double *) const;
    // Corrige eventuellement de la distorsion
    const ElPackHomologue &   PackHomCorr() const;
    // Brut, utilise pour verifier interiorite image
    const ElPackHomologue &   PackHomInit() const;
    const Pt2di & Sz() const;

    Im2D_REAL4 ImPds();
    Im2D_REAL4 ImAppX();
    Im2D_REAL4 ImAppY();
    Im2D_REAL4 ImAppZ();
  private :
    cMatrCorresp(const cMatrCorresp &); // N I 
    typedef REAL4 ** tR4pp;

    cAppliMICMAC &         mAppli;
    int                      mNBXY2;
    Pt2di                    mSz;
    Im2D_REAL4               mImPds;
    TIm2D<REAL4,REAL8>       mTImPds;
    Im2D_REAL4               mImX1;
    TIm2D<REAL4,REAL8>       mTImX1;
    Im2D_REAL4               mImY1;
    TIm2D<REAL4,REAL8>       mTImY1;

    Im2D_REAL4               mImZ;
    TIm2D<REAL4,REAL8>       mTImZ;

    std::vector<Im2D_REAL4>  mXY2;
    std::vector<tR4pp>       mVYY2;
    tR4pp *                  mDXY2;
    ElPackHomologue          mPackHomCorr;
    ElPackHomologue          mPackHomInit;
    std::map<int,cMC_PairIm *>  mPairs;

     //  En mode nuage
     //    mImX1 ->  x
     //    mImY1 ->  y
};



class cModeleAnalytiqueComp  : public ElDistortion22_Gen
{
     public  :
       friend class cMA_AffineOrient;

       cModeleAnalytiqueComp
       (
            cAppliMICMAC &              anAppli,
            const cOneModeleAnalytique &      aModele,
            cEtapeMecComp &             anEtape
       );
       const bool & Import() const ;
       const bool & Export() const ;
       const bool & Glob() const ;
       void  MakeExport();

       // Filtre les  points homologues qui sont dans les point avec une tolerance de Filtre,
       // renvoie false si tous OK avec la tolerance aTol
       bool  FiltragePointHomologues
             (const ElPackHomologue & aPackInit,ElPackHomologue & aNewPack, double aTol,double aFiltre);
       void  LoadMA();
       void MakeInverseModele();

       Pt2dr CorrecDirecte(const Pt2dr &) const;
       Pt2dr CorrecInverse(const Pt2dr &) const;
       const  cOneModeleAnalytique  &  Modele() const ;

       Pt2dr Direct(Pt2dr aP) const;
       virtual bool OwnInverse(Pt2dr &) const ;
       bool ExportGlob() const;

       cMatrCorresp * GetMatr(int aPas,bool PointUnique);
       cGeomImage &         Geom1();
       cGeomImage &         Geom2();
       void SauvHomologues(const ElPackHomologue & aPack);

       cElRegex_Ptr & AutomExport();
       void TifSauvHomologues(const ElPackHomologue & aPack);


     private  :
        Fonc_Num ImPx(int aK);
        std::string NameFile
                    (
                         bool  DirMec,  // Si pas Dir MEC, alors Dir Geom
                         const std::string & aName,
                         const std::string & aPost
                    );

        void SolveHomographie(const ElPackHomologue &  aPackHom);
        Pt2di                    mSz;
        cAppliMICMAC &         mAppli;
        cOneModeleAnalytique   mModele;
        cEtapeMecComp *        mEtape;
        int                    mNbPx;
	cPriseDeVue &          mPDV1;
        cGeomImage &           mGeom1;
	cPriseDeVue &          mPDV2;
        cGeomImage &           mGeom2;
        cGeomDiscFPx           mGeoTer;
        Pt2di                  mSzGl;
        
        cElHomographie         mHomogr;
        cElHomographie         mHomogrInv;
        Polynome2dReal         mPolX;
        Polynome2dReal         mPolY;
        Box2dr                 mBoxPoly;
        int                    mDegrPolAdd;
        Polynome2dReal         mPolXInv;
        Polynome2dReal         mPolYInv;
        std::string            mNameXML;
        std::string            mNameImX;
        std::string            mNameImY;
        std::string            mNameResX;
        std::string            mNameResY;
        bool                   mExpModeleGlobal;
        cElRegex *             mAutomExport;
};


/*****************************************************/
/*                                                   */
/*               cAppliMICMAC                        */
/*                                                   */
/*****************************************************/

/*
    Remarques generales :

      cParamMICMAC devrait etre un membre de cAppliMICMAC et non
    un heritage (en vertu du principe selon lequel, quand 
    on  a le choix, il vaut toujours mieux un membre). Ici
    les interaction sont tellement nombreuses  que la
    flemme de repeter partout ".mParam"   l'a emporte

*/


class cPxSelector
{
     public :
       // Defaut renvoie tjs true
       virtual bool SelectPx(int *) const; 
       virtual ~cPxSelector();
};

typedef enum
{
     eAllocAM_STD,
     eAllocAM_VisuSup,
     eAllocAM_Saisie,
     eAllocAM_Batch,
     eAllocAM_Surperposition
} eModeAllocAM;


struct cDblePx
{
   Im2D_REAL4        mPx1;
   TIm2D<REAL4,REAL8> mTPx1;
   Im2D_REAL4        mPx2;
   TIm2D<REAL4,REAL8> mTPx2;

   cDblePx(Pt2di aSz);
};

class cGeomTerWithMNT2Image : public ElDistortion22_Gen
{
    public :
       Pt2dr Direct(Pt2dr aPTer) const ;
       Pt2dr ImAndPx2Ter(Pt2dr aPIm,Pt2dr aPax);
       Pt2dr ImAndPx2Px(Pt2dr aPIm,Pt2dr aPax);
       Pt2dr Im2Px(Pt2dr aPIm);

       Pt2dr InverseDirect(const Pt2dr &) const;
       Pt2dr GuessInverseInit(const Pt2dr &) const;
       void  Diff(ElMatrix<REAL> &,Pt2dr) const;

       cGeomTerWithMNT2Image
       (
           const double &             aResol,
	   const cGeomDiscFPx &       aGeomTer,
	   const cGeomImage &         aGeomI,
	   const cDblePx    &         aDblPx
       );
        
    private :
         // Pt2dr InverseDirect(const Pt2dr &) const;
	 double                     mResol;
	 const cGeomDiscFPx &       mGeomTer;
         const cGeomImage &         mGeomI;
	 cDblePx                    mPx;
};


// Pour enregistrer les etats interne de la simulations sans avoir
// a les passer en arg et tout en les isolant du reste, on en fait une
// classe dont on herite

class cStateSimul
{
    public :
      
         double                       mRRGlob;
         double                       mRSrIm;
         const cSectionSimulation  *  mSSim;
         const cSimulRelief  *        mSRel;
         const cProjImPart  *         mPrIm;
         double                       mSimZMin;
         double                       mSimZMax;
         Box2di                       mSimCurBxIn;
         Box2di                       mSimCurBxOut;
         Box2dr                       mSimCurBxTer;
         Box2di                       mSimIntBxTer;
         cFileOriMnt                  mSimFOM;
         double                       mSimRPlani;
         double                       mSimFactPA;
         cGeomDiscFPx *               mGTer;
         Pt2di                        mSimSzMNT;
         Tiff_Im *                    mTifMnt;
         Tiff_Im *                    mTifText;
};

// pour extraire le kieme bit d'un tableau de bits compactes 
// selon les conventions MicMac

inline   bool GET_Val_BIT(const U_INT1 * aData,int anX)
{
    return (aData[anX/8] >> (7-anX %8) ) & 1;
}
template <class Type> static Type QSquare(const Type & aV) {return aV*aV;}




// ============================================================
// Classe pour  representer de maniere minimaliste une image
// (acces a ses valeurs et fonction de projection)
//
// Sert aussi pour stocker les info temporaires relatives a chaque
// image (par ex le vecteur des radiometrie dans le voisinage courant)
// ============================================================

typedef REAL4 tGpuF;
typedef Im2D<tGpuF,REAL8>    tImGpu;
typedef TIm2D<tGpuF,REAL8>   tTImGpu;
typedef std::vector<tGpuF *> tVImGpu;
typedef tGpuF **             tDataGpu;


// tGpuF
// tImGpu
// tTImGpu
// tVImGpu
// tDataGpu


class   cGPU_LoadedImGeom
{
   public :
       void InitCalibRadiom(cGLI_CalibRadiom * aCal);
       double CorrRadiom(double aVal);
       double CorrRadiom(double aVal, const Pt2dr &aP);


       ~cGPU_LoadedImGeom();
       cGPU_LoadedImGeom
       (
             const cAppliMICMAC &,
             cPriseDeVue*,
             const Box2di & aBox,
             const Pt2di & aSzV0,
             const Pt2di & aSzVMax,
             bool Top
        );

        // En cas de geometrie image renvoie l'offset entre le 00 du terrain local et la geom image
        // chargee
        Pt2di OffsetIm();

//  Est-ce que un point terrain est visible (si l'option des parties cachees
//  a ete activee)
       bool  IsVisible(int anX,int anY) const
       {
             return   (!mUsePC) || (mImPC[anY][anX] <mSeuilPC);
       }

// Est ce qu'un point image est dans le domaine de definition de l'image
// (dans le rectangle + dans le masque)

      bool IsOk(double aRX,double aRY)
      {
           int anIX = round_ni(aRX);
           int anIY = round_ni(aRY);

           return     (anIX>=0)
                   && (anIY>=0)
                   && (anIX<mSzX)
                   && (anIY<mSzY)
                   && (GET_Val_BIT(mImMasq[anIY],anIX));

      }

      bool IsOkErod(int anIX,int anIY)
      {
           return     (anIX>=0)
                   && (anIY>=0)
                   && (anIX<mSzX)
                   && (anIY<mSzY)
                   && (GET_Val_BIT(mImMasqErod[anIY],anIX));
      }




      Pt2di getSizeImage()
      {
	  
		  Pt2di size(mSzX,mSzY);
		  
		  return size;  
	  
      }

      cGeomImage * Geom() {return mGeom;}
      void AssertOneImage()
      {
          ELISE_ASSERT(mOneImage,"cGPU_LoadedImGeom::AssertOneImage");
      }
      float ** MyDataIm0()   
      {
          return mMyDataIm0;
      }
      float ** DataIm0()   
      {
           AssertOneImage();
           return mDataIm[0];
      }
      float * LinDIm0()   
      {
           AssertOneImage();
           return mLinDIm[0];
      }
      U_INT1 ** ImPC()      {return mImPC;}
      U_INT1**  ImMasq()    {return mImMasq;}
      U_INT1**  ImMasqErod()    {return mImMasqErod;}
      float *** VDataIm()   {return mDataIm;}
      float **  VLinDIm()   {return mLinDIm;}
      double  PdsMS() const;
      double  CumSomPdsMS() const;
      double  TotSomPdsMS() const;


      double * Vals()     {return &(mVals[0]);}
      void  SetOK(bool aIsOK) {mIsOK = aIsOK;}
      bool  IsOK() const {return mIsOK;}

      double DzOverPredic(const Pt3dr & aP) const
      {
            return mPDV->DzOverPredic(aP);
      }
      bool OkOrtho(int anX,int anY) const
      {
            return ( mDOK_Ortho[anY][anX]!=0 );
      }
      tGpuF ImOrtho(int anX,int anY) const
      {
            return mDOrtho[anY][anX];
      }

       tGpuF **  DataOrtho();
       tGpuF **  DataSomO();
       tGpuF **  DataSomO2();

       U_INT1 ** DataOKOrtho();
       Im2D_U_INT1 ImOK_Ortho();
       tImGpu  ImOrtho();
       tImGpu  ImSomO();
       tImGpu  ImSomO2();
       tImGpu  ImSom12();


       bool InitValNorms(int anX,int anY,int aNbScaleIm);
       double ValNorm(int anX,int anY) const 
       {
            return (mDOrtho[anY][anX]-mMoy) / mSigma;
       }
       double ValOrthoBasik(int anX,int anY) const 
       {
            return mDOrtho[anY][anX] ;
       }
       double SumO(int anX,int anY) const 
       {
            return mDSomO[anY][anX] ;
       }
       double SumO2(int anX,int anY) const 
       {
            return mDSomO2[anY][anX] ;
       }


       double MoyCal() const {return mMoy;}
       double SigmaCalc() const {return mSigma;}
       cPriseDeVue * PDV();

       bool Correl(double & Correl,int anX,int anY,const cGPU_LoadedImGeom & aGeoJ,int aNbIm) const;
       bool CorreCensus(double & Correl,int anX,int anY,const cGPU_LoadedImGeom & aGeoJ,int aNbIm) const;

       // inline double StatIm(int anX,int anY,tGpuF **) const;

       inline double MoyIm(int anX,int anY,int aNbIm) const;
       inline double MoyQuadIm(int anX,int anY,int aNbIm) const;
       inline double CovIm(int anX,int anY,int aNbIm) const;

       REAL GetValOfDisc(int anX,int anY,int aZ);

       Pt2dr ProjOfPDisc(int anX,int anY,int aZ) const;
       void MakeDeriv(int anX,int anY,int aZ);
       Pt2dr ProjByDeriv(int anX,int anY,int aZ) const;
       cStatOneImage * ValueVignettByDeriv(int anX,int anY,int aZ,int aSzV,Pt2di PasVig) ;
       cStatOneImage * VignetteDone();

       
       const std::vector<cGPU_LoadedImGeom *> & MSGLI() const {return mMSGLI;}
       cGPU_LoadedImGeom * KiemeMSGLI(int aK) const {return mMSGLI.at(aK);}


       int  NbScale() const;
       Im2D_REAL4 * FloatIm(int aKScale);
       std::vector<Im2D_REAL4> VIm();

       Pt2di  SzV0() const;

        void InitMCP_AttachePix(const cMCP_AttachePixel * aAP);


   private :
       
       cGPU_LoadedImGeom(const cGPU_LoadedImGeom &); // N.I.
       const cAppliMICMAC & mAppli;
       bool             mTop;
       cPriseDeVue *    mPDV;
       cLoadedImage *   mLI;
       cGeomImage *     mGeom;

       Pt2di              mSzV0;
       Pt2di              mSzVMax;
       Pt2di              mSzOrtho;

       int                mX0Deriv;
       int                mY0Deriv;
       int                mZ0Deriv;
       Pt2dr              mDerivX;
       Pt2dr              mDerivY;
       Pt2dr              mDerivZ;
       Pt2dr              mValueP0D;
       Pt3di              mPOfDeriv;
       cStatOneImage      mBufVignette;

       std::vector<cGPU_LoadedImGeom *> mMSGLI;  // Multi Scale GLI
       cGPU_LoadedImGeom *              mMaster;
       bool                             mOneImage;
       const cOneParamCMS *             mOPCms;
       double                           mPdsMS;
       double                           mCumSomPdsMS;
       double                           mTotSomPdsMS;
       float **                         mMyDataIm0;

// tGpuF
// tImGpu
// tTImGpu
// tVImGpu
// tDataGpu
       tImGpu               mImOrtho;
       tTImGpu              mDImOrtho;
       tVImGpu              mVOrtho;
       tDataGpu             mDOrtho;

       tImGpu               mImSomO;
       tTImGpu              mDImSomO;
       tVImGpu              mVSomO;
       tDataGpu             mDSomO;

       tImGpu               mImSomO2;
       tTImGpu              mDImSomO2;
       tVImGpu              mVSomO2;
       tDataGpu             mDSomO2;

       tImGpu               mImSom12;
       tTImGpu              mDImSom12;
       tVImGpu              mVSom12;
       tDataGpu             mDSom12;




       Im2D_U_INT1           mImOK_Ortho;
       TIm2D<U_INT1,INT>     mDImOK_Ortho;
       std::vector<U_INT1 *> mVImOK_Ortho;
       U_INT1 **             mDOK_Ortho;



       int                  mNbVals;
       double               mMoy;
       double               mSigma;
       
         

    //  tampon pour empiler les valeur de l'image sur un voisinage ("imagette")
        std::vector<double>  mVals;

    //  zone de donnee : "l'image" elle meme en fait

        float ***        mDataIm;
        float **         mLinDIm;
    //  Masque Image (en geometrie image)
        int              mSzX;
        int              mSzY;
        U_INT1**         mImMasq;
        U_INT1**         mImMasqErod;

   // Parties cachee :  masque image (en geom terrain), Seuil  et usage 
       U_INT1 **          mImPC;
       int                mSeuilPC;
       bool               mUsePC;
       bool               mIsOK;
       cGLI_CalibRadiom * mCalR;
};


typedef enum
{
   eModeNoMom,
   eModeMom_2_22,
   eModeMom_12_2_22
}
eModeInitZ;


class cResCorTP;
class cMMTP;

class cAppliMICMAC  : public   cParamMICMAC,
                      private  cStateSimul
{
     public :
        const cMMUseMasq3D *   Masq3DOfEtape(cEtapeMecComp & anEtape);

         //int    MaxPrecision() const;
         void AddPrecisionOfArrondi(const cDecimal &, double aVal);
         void AddPrecisionOfDec(const cDecimal &,double aMul);
         void UpdatePrecision(int aP);

        bool   CMS_ModeEparse() const;
        cAnamorphoseGeometrieMNT * AnaGeomMNT() const;
        cMakeMaskImNadir * MMImNadir() const;


     // Pour borner a priori la taille memoire prise par certains algos, on a besoin de savoir
     // le nombre d'image, avant de charge le terrain, c'est donc approximatif et putot majorant
     int NbApproxVueActive();

      void DoMasqueAutoByTieP(const Box2di& aBox,const cMasqueAutoByTieP & aMATP);
      void CTPAddCell(const cMasqueAutoByTieP & aMATP,int anX,int anY,int aZ,bool Final);
      cResCorTP CorrelMasqTP(const cMasqueAutoByTieP & aMATP,int anX,int anY,int aZ);
      void  OneIterFinaleMATP(const cMasqueAutoByTieP & aMATP,bool Final);


      void MakeDerivAllGLI(int anX,int anY,int aZ);

      double AdaptPas(double) const;
      cStatGlob  * StatGlob();
      bool WM() const;  // Avec Message

      // Charge en RAM un version reduite en resol des champs de px, corrige
      // des zoom et pas:  aResol / a Resol 1
      cDblePx LoadPx(cEtapeMecComp & anEtape,double aResol);


     Pt3dr  GetPtMoyOfOri(ElCamera * anOri) const;
// Ces classes devraient etre des membres ou des parents de 
// cAppliMICMAC, pour alleger le code elles sont deportees
// dans des fichiers a part.
       friend class cOptimDiffer;
       friend class cMicMacVisu;

       void VerifSzFile(Pt2di aSzF) const;

       
        void MicMacErreur
             (
                  eMicMacCodeRetourErreur aCode,
                  const std::string & aMes,
                  const std::string & aDiag = ""
             ) const;
        int CodeMicMax2CodeExt(eMicMacCodeRetourErreur) const;

        friend class cDataGeomTerrain;
        void TestReducIm(int aDZ);

        typedef std::list<cTypePyramImage> tContTPyr;
        typedef tContTPyr::const_iterator   tCsteIterTPyr;

        // Creation par ligne de commande
        static cAppliMICMAC * Alloc(int argc,char ** argv,eModeAllocAM aMode);
	~cAppliMICMAC();


        Tiff_Im FileMasqOfResol(int aDeZoom) ;
        Pt2di   SzOfResol(int aDeZoom);
        // Integre le FileMasqOfResol qui est un carac du chantier
        // et un eventuel masque de point d'interet qui est une 
        // carac de la MEC
           Fonc_Num    FoncSsPIMasqOfResol(int aDz);
           Fonc_Num    FoncMasqOfResol(int aDz);



	// Accesseur 

        // Box2dr          BoxTer();
        const Box2di &  BoxIn()  const ;
        const Box2di &  BoxOut() const ;
        std::string  DirCube() const;
        std::string  NameFileCurCube(const std::string &) const;

        bool  IsOptDiffer() const;
        bool  IsOptimCont() const;
        bool  IsOptimReel() const;
        bool  IsOptDequant() const;
        bool  IsOptIdentite() const;

        const std::string & DirImagesInit() const;
        const std::string & DirMasqueIms() const;
        eModeAllocAM ModeAlloc() const;
        const std::list<cGeometrieImageComp *> & GeoImsComps() const;
        const cGeomDiscFPx &  GeomDFPx() const;
        const cGeomDiscFPx &  GeomDFPxInit() const;
        const cEtapeMecComp * CurEtape() const; 
        const cEtapeMecComp * FirstVraiEtape() const; 
        cCaracOfDeZoom *      GetCurCaracOfDZ() const;
	const std::string & FullDirMEC() const;
	const std::string & FullDirPyr() const;
	const std::string & FullDirGeom() const;
	const std::string & FullDirResult() const;
        std::string NameEtatAvancement();
        cMM_EtatAvancement EtatAvancement();
        void  SauvEtatAvancement(bool AllDone);


        std::string  NamePC( bool  ForPC,
                            const cGenerePartiesCachees &,
                            cEtapeMecComp *,
                            const cPriseDeVue & aPdv) const;


        const cPriseDeVue *           PDV1() const;
        const cPriseDeVue *           PDV2() const;
        cPriseDeVue *                 PDV1() ;
        cPriseDeVue *                 PDV2() ;
        std::vector<cPriseDeVue *>    AllPDV();
        int   DimPx()                        const;
        int   NbPdv()                        const;
        const cLoadTer *           LoadTer() const;
        bool         InversePx() const;  // Remplacer Pax par 1/Pax

        const cPriseDeVue * PDVFromName
                            (
                                const std::string  & aName,
                                const char * MesErreur
                             ) const;
        cPriseDeVue * NC_PDVFromName
                            (
                                const std::string  & aName,
                                const char * MesErreur
                             ) ;




        // Valeur Speciale Not Image
        bool HasVSNI() const;  // Est-ce qu'il y en a une
        int  VSNI() const; // Renvoie 0 si pas de VSNI

	// typedef tContPDV::iterator          tIterFI;
        tCsteIterPDV PdvBegin() const;
        tCsteIterPDV PdvEnd()  const;

        int NbVueAct() const;
        int NumImAct2NumImAbs(int aNum) const;

        double CostCalcCorrelPonctuel(int aNb) const;
        double CostCalcCorrelRectang(Box2di,int aNbPts) const;
        void   CalcCorrelByRect(Box2di,int * aPx);

        void DoCorrelLeastQuare(const Box2di & aBoxOut,const Box2di & aBoxIn,const cCorrel2DLeastSquare &);
	void DoGPU_Correl (const Box2di & aBoxInterne,const cMultiCorrelPonctuel *,double aPdsPix);  
        void DoCensusCorrel(const Box2di & aBox,const cCensusCost &);
        void DoOneCorrelSym(int anX,int anY,int aNbScale);
        void DoOneCorrelIm1Maitre(int anX,int anY,const cMultiCorrelPonctuel *,int aNbIm,bool VireExtr,double aPdsPix);
        void DoOneCorrelMaxMinIm1Maitre(int anX,int anY,bool aModeMax,int aNbIm);
        void DoCostLearnedMMVII(const Box2di & aBox,const cScoreLearnedMMVII &aCPC);


		void DoGPU_Correl_Basik (const Box2di & aBoxInterne); 

#ifdef  CUDA_ENABLED
        ///
        /// \brief Tabul_Projection
        /// \param Z
        /// \param interZ
        /// \param idBuf
        ///
        void Tabul_Projection(short Z,  ushort& interZ, ushort idBuf);
        void setVolumeCost(short interZ0, short interZ1, ushort idBuf);
        void Tabul_Images(int Z, uint &interZ, ushort idBuf);

#endif
		void Correl_MNE_ZPredic (const Box2di & aBoxInterne,const cCorrel_Correl_MNE_ZPredic &);  
		void DoCorrelPonctuelle2ImGeomI(const Box2di&aBoxInterne,const cCorrel_Ponctuel2ImGeomI&);  
        void DoCorrelCroisee2ImGeomI(const Box2di & aBox,const cCorrel_PonctuelleCroisee&aCPC);



        void DoCorrel2ImGeomImGen(const Box2di & aBox,double aRatioI1I2,double aPdsPonct,bool AddCpleRad);


        void DoCorrelMultiFen ( const Box2di & aBox, const cCorrel_MultiFen &);

        void DoCorrelRobusteNonCentree ( const Box2di & aBox, const cCorrel_NC_Robuste & aCNR);




        void StatResultat(const Box2di & aBox, Im2DGen & aPxRes, const cDoStatResult &);

	void DoCorrelAdHoc ( const Box2di & aBoxInterne);  
	void GlobDoCorrelAdHoc (const Box2di & aBoxOut, const Box2di & aBoxInterne);  



        void InitBlocInterne( const Box2di & aBoxInterne);

	void DoOneBlocInterne
             (
                  const cPxSelector &,
                  const Box2di & aBoxInterne
             );  // Dans cAppliMICMAC_MEC.cpp

	 bool  CurWSpecUseMasqGlob() const;
         int   CurNbIterFenSpec() const;
         const Pt2di &  PtSzWMarge() const;
         const Pt2di &  PtSzWFixe() const;
         const Pt2dr &  PtSzWReelle() const;
         int   NbPtsWFixe () const;
         int   SzWFixe    () const;
         int   CurSurEchWCor() const;
         cInterfSurfaceAnalytique * AnamSA() const;
         cXmlOneSurfaceAnalytique * XmlAnamSA() const;
         const cChCoCart *  RC() const;
         const cChCoCart *  RCI() const;
         bool UseAlgoSpecifCorrelRect() const;
         bool DoTheMEC() const;


         Pt2di Px2Point(int * aPx) const
         {
             return Pt2di(aPx[0],(mDimPx>1)?aPx[1]:0);
         }
        
        std::string NameImageMasqOfResol(int aDeZoom) ;
        std::string NameFileSzW(int aDz);
        bool  AucuneGeom() const;
        eModeGeomMEC ModeGeomMEC() const;
        void TestPointsLiaisons(CamStenope *  anOriRef,CamStenope *  anOri,cGeomImage *) const;
        std::string NamePackHom(const std::string &,const std::string &) const;
        std::string NameFilePyr(const std::string &,int) const;
        const std::string & NameChantier() const;

        cModeleAnalytiqueComp &  LastMAnExp();
        void SetLastMAnExp(cModeleAnalytiqueComp *);

        // On centralise les appels aux libraire dynamique
        void * AllocObjFromLibDyn
               (
                   const std::string & aNameLibraire,
                   const std::string & aNameSymb
               ) const;

        void ExeProcessParallelisable (bool AddNameExeMicMac,
                     const  std::list<std::string> & aLProc);
        const double  & RecouvrementMinimal() const;
        cFileOriMnt GetOri(const std::string & aNameOri) const;
        cFileOriMnt OrientFromOneEtape(const cEtapeMecComp &) const;
        Fonc_Num  AdaptFoncFileOriMnt
                  (
                        const cFileOriMnt & anOriCible,
                        Fonc_Num            aFonc,
                        const std::string & aNameOri,
                        bool                aModifDyn,
                        double              anOffset
                 ) const;
        const cExportApero2MM  &  ExpA2Mm() const;
        int FreqPtsInt() const; 
        int DeZoomMax() const;
        int DeZoomMin() const;
        bool FullIm1() const;
        int  NbPDV() const;
        cSurfaceOptimiseur *    SurfOpt();
        const std::string& FileBoxMasqIsBoxTer() const;
        void SetContourSpecIm1(const std::vector<Pt2dr> &);
        bool UseConstSpecIm1() const;
        const std::vector<Pt2dr>  & ContSpecIm1() const;


        cCaracOfDeZoom * GetCaracOfDZ(int aDZ) const;

        const cInterfaceVisualisation * PtrVI() const ;
        void  AnalyseOri(CamStenope *  ) const; // Pour evt corr Orig Tgt Loc
        cInterfModuleImageLoader *  LoaderFiltre
                                    (
                                       cInterfModuleImageLoader * aLoaderBase,
                                       std::string                aName
                                    );

         eTypeWinCorrel CurTypeWC() const;
	 Pt2dr          FactFenetreExp() const;
	 const std::string & NameSpecXML() const;
         const double & DefCost() const;
         cInterfChantierNameManipulateur * ICNM() const;
         int  CurDCAllDC() const;

         std::string NameClassEquiv(const std::string &)const;

         double CurCorrelToCout(double) const;

         const cCorrelAdHoc * CAH() const;
         const cCorrelMultiScale*  CMS() const;
         const cCensusCost *       CC() const;

         double AhDefCost () const {return mAhDefCost;} 
         double AhEpsilon () const {return mAhEpsilon;}

         double DeltaMoy(double aMoy) const
         {
              return mEpsAddMoy + aMoy * mEpsMulMoy;
         }
         const cEtiqBestImage *  EBI() const;
         void SauvFileChantier(Fonc_Num aF,Tiff_Im aFile) const;
         cEl_GPAO *            GPRed2() const;
              
          Pt2dr DequantPlani(int anX,int anY) const {return Pt2dr( mOriPlani.x + mStepPlani.x*anX,mOriPlani.y + mStepPlani.y*anY);}
          double DequantZ(int aZ) const {return  mOrigineZ+ aZ*mStepZ;}

          private :
        static const int theNbImageMaxAlgoRapide = 3; 
	cAppliMICMAC (const cAppliMICMAC  &);     // N.I.

	//  Calcul de la taille memoire a priori d'un process

	int MemSizePixelImage() const;
	double MemSizeProcess(int aTxy) const;
	int GetTXY() const;
	  

        cAppliMICMAC 
        (
             eModeAllocAM        aMode,
             const std::string & aNameExe,
             const std::string & aNameXML,
             cResultSubstAndStdGetFile<cParamMICMAC>,
             /// const cParamMICMAC &,

             char **            aArgAux,
             int                aNbArgAux,
             const std::string  & aNameSpecXML
        );
	void InitDirectories();
	void InitAnamSA();
	void InitImages();
	void PostInitGeom();
        void InitNadirRank();
	void InitMecComp();
        void InitMemPart();
        void SauvMemPart();
        void DoPurgeFile();
        void DoPostProcess();

	std::string ChMpDCraw(const cPriseDeVue *) const;

        void VerifOneEtapes(const cEtapeMEC & anEt) ;
        void VerifEtapesSucc
             (
                  const cEtapeMEC & anEt0,
                  const cEtapeMEC & anEt1
             ) ;
        void VerifEtapes() ;
        void VerifTPyr() const;
        void VerifImages() const;

	// Ajoute une image sous reserve qu'elle n'existe pas encore
	void AddAnImage(const std::string & aName);
	// Ajoute toutes les images de WorkDir filtree par le patter,
	void PatternAddImages(const std::string & aPatternName);
        void  AddImageByDelta(const cListImByDelta &);


        void MakeResultOfEtape(cEtapeMecComp &);
        void MakeDequantSpecial();
        void DoMasq3D(cEtapeMecComp & anEtape,const cMMUseMasq3D &);


        void MakeGenCorPxTransv(cEtapeMecComp &);
	void MakeExportAsModeleDist 
             (  const cGenereModeleRaster2Analytique,
                cEtapeMecComp & anEtape
             );

        void MakeImagePx8Bits
        (
             std::list<std::string> &    mVProcess,
             const cEtapeMecComp &         anEtape,
             int                           aKFile,
             const cTplValGesInit<bool> &  Genere,
             const cTplValGesInit<int>  &  anOffset,
             const cTplValGesInit<double>& aDyn
        );

        cFileOriMnt OrientFromParams(int aDz,REAL aStepZ);



        ///===  GESTION DES PARTIS CACHEES

        // Box -> Disc, Z -> Terrain
        void GetIntervZ
             (
                 const Box2di & aBox,
                 double & aZmin,
                 double & aZMax,
                 double & aZMoy
             );
        void MakePartiesCachees();
        void MakePartiesCachees
             (
                 cPriseDeVue &,
                 const cGenerePartiesCachees &,
                 double aZMin,
                 double aZMax,
                 int aCptModProcess,
                 int aCptDivProcess
             );
       
        void MakeRedrLocAnamSA();
        Pt3dr ToRedr(const cFileOriMnt & aFOMInit,const cFileOriMnt & aFOMCible,const Pt3dr &aPDiscInit);

        ///========================================

        void GenereOrientationMnt();
        std::string   NameOrientationMnt(cEtapeMecComp * itE);

        void GenereOrientationMnt(cEtapeMecComp *);
        void SauvParam();
        void MakeFileFDC();
	void DoAllMEC();  // Dans cAppliMICMAC_MEC.cpp
        // OneEtapeSetCur fait ce qui est necessaire pour la generation de
        // resultats, meme si il n'y a pas de MEC
        void OneEtapeSetCur(cEtapeMecComp & anEtape);
	void DoOneEtapeMEC(cEtapeMecComp &);  // Dans cAppliMICMAC_MEC.cpp
	std::string PrefixGenerikRecalEtapeMicmMac(cEtapeMecComp &);  // Dans cAppliMICMAC_MEC.cpp
	void DoOneBloc( const Box2di & aBoxOut,
			const Box2di & aBoxIn,
                        int aNiv,
			const Box2di & aBoxGlob
                      );  // Dans cAppliMICMAC_MEC.cpp



        REAL CalculScore();
        REAL CalculAccelereScore();
        void PrepareAccelIm1(int *);


        void ShowEtapes() const;
        void MakeFileTA();
        void MakeDefImMasq(int aZoomCible) ;


        void InitCostCalcCorrel();
        void ChoixCalcCorrByQT(Box2di aBoxTer);

        void DoImagesBSurH(const cDoImageBSurH&);

        cInterfModuleImageLoader * GetMIL
                                   (
                                      cGeometrieImageComp *,
                                      const std::string & aName
                                   );

        void OptimisationDifferentielle();
        void OptimisationContinue();
        void Visualisation(Box2di aBoxIn);

        void MakeOrtho( 
                        const Box2di& aBoxGlob,
                        const Box2di& aBoxIn,
                        const Box2di& aBoxOut,
                        int aKBox,
                        double aZMoy,
                        Im2D_Bits<1>,
                        const ElAffin2D & aSPM2PO,
                        const cMakeOrthoParImage &,
                        cPriseDeVue &,
                        cMicMacZbuf *,
                        cMetaDataPartiesCachees & aMDPC
                       );

       /**********************************/
       /*   Partie Data                  */
       /**********************************/

       std::string mNameChantier;
       string mOutputDirectory;
       cInterfChantierNameManipulateur * mICNM;
       bool         mWM;
       bool         mFullIm1;
       bool         mUseConstSpecIm1;
       std::vector<Pt2dr> mContSpecIm1;
       eModeAllocAM mModeAlloc;
       eModeGeomMEC mModeGeomMEC;
       bool         mAucuneGeom;
       bool         mInversePx;  // Remplacer Pax par 1/Pax

       std::string  mNameExe;
       std::string  mNameXML;
       std::string  mNameSpecXML;
       char **      mArgAux;
       int          mNbArgAux;
       std::string  mDirImagesInit;
       std::string  mDirMasqueIms;
       

	//  - -  - - - - - - - - - - - - -
	//  PRISE DE VUE
	//  - -  - - - - - - - - - - - - -
	
        tContPDV                mPrisesDeVue;
        tContPDV                mPDVBoxGlobAct;
        int                     mNbImChCalc;
        tContPDV                mPDVBoxInterneAct;
        int                     mNb_PDVBoxInterne;
        int                     mNbPDV;
                  // En mode image, il est parfois plus simple d'attaquer
                  // directement Im1/Im2
        cPriseDeVue *           mPDV1;
        cPriseDeVue *           mPDV2;

        std::list<cGeometrieImageComp *> mGeoImsComps;
        mutable cElRegex *                       mAutomNomsHoms;
        mutable cElRegex *                       mAutomNomPyr;

	//  - -  - - - - - - - - - - - - -
        // Directories completes pour la mise en corresp,
	// les pyramides
	//  - -  - - - - - - - - - - - - -
	std::string             mFullDirMEC;
	std::string             mFullDirPyr;
	std::string             mFullDirGeom;
	std::string             mFullDirResult;

	//  - -  - - - - - - - - - - - - -
	//   PARALAXES / MEC
	//  - -  - - - - - - - - - - - - -
        int                     mDimPx;
	tContEMC                mEtapesMecComp;
        cEtapeMecComp *         mEtape00;  // "fausse etape", pas executee
                                // permet d'amorcer le processus multi-resol
				// elle contient un fichier de 0, (ou + tard un MNT
				// predictif)

        cEtapeMecComp *         mCurEtape; // Pour eviter de la passer
        cEtapeMecComp *         mPrecEtape; // Pour eviter de la passer
                                // tout le temps en parametre
         const cEtiqBestImage *  mEBI;

         bool                   mIsOptDiffer;
         bool                   mIsOptDequant;
         bool                   mIsOptIdentite;
         bool                   mIsOptimCont;

         cCaracOfDeZoom *       mCurCarDZ;
         cModeleAnalytiqueComp * mCurMAI;
	 // Avant les fenetre exponentielle il y a avait une seule SzW de type int
	 // Maintenant, avec des fentre normales, on a mSzWFixe = mSzWMarge (=ancien mSzW)
	 // Avec des fenetre exponentielles, on a mSzWMarge = 3 * mSwR
	 //  mSzWMarge => mPtSzWMarge
         Pt2dr                  mSzWR;
         eTypeWinCorrel         mCurTypeWC;
	 int                    mCurNbIterFenSpec;
	 bool                   mCurFenSpec;
         int                    mSzWFixe;

	 bool                   mCurForceCorrelByRect;
	 bool                   mCurForceCorrelPontcuelle;


	 Pt2dr                 mFactFenetreExp;


	 bool                   mCurWSpecUseMasqGlob;
         int                    mCurSzWInt; // Fenetre "interieure" de correl
         int                    mCurSurEchWCor;
         bool                   mModeIm1Maitre;
         bool                   mHasOneModeIm1Maitre;
         Pt2di                  mPtSzWMarge;
         Pt2di                  mPtSzWFixe;
         int                    mNbPtsWFixe;
         std::vector<Pt2dr>     mVecPtsW;
         Pt2dr *                mTabPtsW;

        cGeomDiscFPx           *mGeomDFPx;
        cGeomDiscFPx           *mGeomDFPxInit;
        cLoadTer *              mLTer;
        // Box2dr                  mBoxTer;
        Box2di                  mBoxIn;
        Box2di                  mBoxOut;
        Box2di                  mBoxInterne;
          // Variable pour l'exploration de mBoxInterne
        Pt2di                   mCurPterI;
        Box2di                  mCurBoxW;  // Boite lie a la fenetre de correl
        int                     mCurPxI[theDimPxMax];
        cLineariseProj          mLineProj;
        //  cStatOneClassEquiv *         mStatN;
        cStatGlob          *         mStatGlob;
        int                          mNbApproxVueActive;
       

       // Permet de diffuser aux processus fils les resultats
       // du pere (de maniere generale de reutiliser certains calculs
       // "lourds"
       cMemPartMICMAC   mMemPart;
       std::string      mNameFileMemPart;

       // Stream d'affichage des message
       bool             mShowMes;
       std::ostream &   mCout;
       double           mTimeTotCorrel;
       double           mTimeTotOptim;
       int              mNbBoitesToDo;
       int              mKBox;

        double  mNbPointsByRect2;
        double  mNbPointsByRectN;
        double  mNbPointByRectGen;
        double  mNbPointsIsole;

        // Calcul du cout de calcul

        double mCostBilin;  // 1 c'est l'etalon
	double mCostSinCardElem;
        double mCostBicub;  
        double mCostPPV;  
        double mCostInterpol; 
        double mCostTabul; 


        double   mCostEc;  // Cout calcul Moyenn+Ec
        double   mCostCov;  // Cout calcul SXY
        double   mCostCorrel2Im;  // Cout / pixel
        double   mCostCorrel3Im;  // Cout / pixel
        double   mCostCorrel1ImOnNSsEc;  // Cout / pixel / Image , ss Ecart
        double   mCostCorrel1ImOnN;  // Cout / pixel / Image , avec Ecart
        double   mCostGeom;  // Cout des projection Geom
        

        double  mCostPixelPonctuel;
        double  mCostPixelRect;
        double  mCostPixelPtInRect;
   
        double  mRecouvrMin;

        cModeleAnalytiqueComp *  mLastMAnExp;

        int mSzRec;
        int mSzDMin;
        int mSzDMax;



         // Specif DeZoom
        int mDeZoomMax;
        int mDeZoomMin;
        std::vector<cCaracOfDeZoom *> mVCaracZ;
         
        int mFreqPtsInt;  // 1 si pas de Pts d'interet
        // Indique si a la courante etape il y a une acceleration speciale
        // en partique, c'est equivalent a etre en eGeomMECIm1
        bool mCSAccelIm1;

        double mSom0;
        double mSomI1;
        double mSomI11;
        std::vector<double> mVecV1;
        bool                mOkStatIm1;
        double              *mTabV1;
        double mSomI2;
        double mSomI12;
        double mSomI22;
        double mDefCorr;
        double mDefCost;
        double mEpsCorr;
	int    mDeZoomFilesAux;


        //  Variables utilisee dans les correls Ad Hoc

           void DoInitAdHoc(const Box2di & aBox);
           // Si pas FirstZ de la colone et Im1 maitresse, pas la peine
           // de reinitialiser 
           bool InitZ(int aZ,eModeInitZ aMode);

  // Permet entre autre d'optimiser pour geom image maitresse
           bool mFirstZIsInit;

           Pt2di mSzGlobTer;
           int  mX0Ter;
           int  mX1Ter;
           int  mY0Ter;
           int  mY1Ter;

           int  mDilX0Ter;
           int  mDilX1Ter;
           int  mDilY0Ter;
           int  mDilY1Ter;

           int mX0UtiTer;
           int mX1UtiTer;
           int mY0UtiTer;
           int mY1UtiTer;

           int mX0UtiDilTer;
           int mX1UtiDilTer;
           int mY0UtiDilTer;
           int mY1UtiDilTer;

           int mX0UtiLocIm;
           int mX1UtiLocIm;
           int mY0UtiLocIm;
           int mY1UtiLocIm;

           int mX0UtiDilLocIm;
           int mX1UtiDilLocIm;
           int mY0UtiDilLocIm;
           int mY1UtiDilLocIm;


           Pt2di  mCurSzDil;

           int    mZMinGlob;
           int    mZMaxGlob;
           int    mZIntCur;
           double mZTerCur;
           Pt2di  mCurSzV0;
           Pt2di  mCurSzVMax;

           Im2D_U_INT1            mImOkTerCur;
           TIm2D<U_INT1,INT>      mTImOkTerCur;
           std::vector<U_INT1 *>  mVDOkTer;
           U_INT1 **              mDOkTer;

           Im2D_U_INT1            mImSzWCor;
           TIm2D<U_INT1,INT>      mTImSzWCor;

           Im2D_U_INT1            mImOkTerDil;
           TIm2D<U_INT1,INT>      mTImOkTerDil;
           std::vector<U_INT1 *>  mVDOkTerDil;
           U_INT1 **              mDOkTerDil;

           // Besoin comme cas special sur image 1 maitresse pour forcer
           // le calcul sur premiere nappe
           Im2D_U_INT1            mAll1ImOkTerDil;
           TIm2D<U_INT1,INT>      mAll1TImOkTerDil;
           std::vector<U_INT1 *>  mAll1VDOkTerDil;
           U_INT1 **              mAll1DOkTerDil;

           std::vector<Im2D_REAL4>  mBufCensusIm2;
           std::vector<float **> mVBufC;
           float ***             mDataBufC;



           double mAhDefCost;
           double mAhEpsilon;

           int                 mGpuSzD;
           Im2D_REAL4          mGeoX;
           TIm2D<REAL4,REAL8>  mTGeoX;
           Im2D_REAL4          mGeoY;
           TIm2D<REAL4,REAL8>  mTGeoY;

           double mEpsAddMoy;
           double mEpsMulMoy;


           INT2 ** mTabZMin;
           INT2 ** mTabZMax;
           U_INT1 **  mTabMasqTER;

           // Param pour la quantification
           Pt2dr mOriPlani;
           Pt2dr mStepPlani;
           double mOrigineZ;
           double mStepZ;

           std::vector<cGPU_LoadedImGeom *>  mVLI;
           // mVScaIm[aScale][aIm]
           std::vector<std::vector<cGPU_LoadedImGeom *> > mVScaIm;
           int                              mNbIm;
           int                              mNbScale;


           inline bool IsInTer(int anX,int anY) const {return GET_Val_BIT(mTabMasqTER[anY],anX);}



        //  ===================  

        const cInterfaceVisualisation * mPtrIV;
        cMicMacVisu *                   mVisu;

        mutable Pt2dr  mOriPtLoc;
        mutable bool   mOriPtLoc_Read; // ConvertToSameOriPtTgtLoc fait

        cStdMapName2Name *  mMapEquiv;
        // Pour "deformation" de la geometrie de calcul
        cInterfSurfaceAnalytique * mAnamSA;
        cXmlOneSurfaceAnalytique * mXmlAnamSA;
        std::string                mNameAnamSA;
        cChCoCart *  mRepCorrel;
        cChCoCart *  mRepInvCorrel;

       // Utilitaire pour le DoNothingBut
       bool DoNotMemPart() const;
       bool DoNotTA() const;
       bool DoNotMasqChantier() const;
       bool DoNotFDC() const;
       bool DoNotOriMNT() const;
       bool DoMTDNuage() const;
       bool DoNotExtendParam() const;
       bool DoSimul() const;
       cExportApero2MM   mExpA2Mm;
       std::string       mFileBoxMasqIsBoxTer;

       std::vector<Im2DGen *> mOrthoImIn;
       std::vector<cIm2DInter *> mIntOrth;

       cTplCIKTabul<float,double>  mInterpolTabule;
       cSurfaceOptimiseur *    mSurfOpt;
       const cCorrelAdHoc *      mCorrelAdHoc;
       const cCorrelMultiScale*  mCMS;
       const cCensusCost *       mCC;
       bool                      mCMS_ModeEparse;

       bool                  mGIm1IsInPax;
       cEl_GPAO *            mGPRed2;


       void GenerateSimulations();
       void SimulateRelief(bool isNew);
       void SimulatePDV(cPriseDeVue & aPDV);
       void SimulateOneBoxPDV(cPriseDeVue & aPDV,Tiff_Im * aTifIm,Tiff_Im * aFileMNT);


       //  Pour le calcul de masque par tie points
      
      
       
         int                   mTPZoom;
         int                   mTPSzW;
         std::vector<Pt3dr> *  mTP3d;
         bool                  mCurEtUseWAdapt;
         int NbScaleOfPt(int anX,int anY)
         {
              return  mCurEtUseWAdapt  ?
                      ElMin(mNbScale,1+mTImSzWCor.get(Pt2di(anX,anY)))  :              
                      mNbScale;

         }


	// GPGPU
#ifdef CUDA_ENABLED
        GpGpuInterfaceCorrel	IMmGg;
        GpGpu_Interface_Cor_MS    interface_Census_GPU;
#endif	

         cMMTP *  mMMTP;

         bool  mDoTheMEC;
         cAnamorphoseGeometrieMNT * mAnaGeomMNT;
         cMakeMaskImNadir         * mMakeMaskImNadir;

         int     mMaxPrecision;

         std::map<std::string,cGLI_CalibRadiom *> mDicCalRad;
 
     public :

         void ResetCalRad();
         
       // Pour debug MM TieP
         cMasqBin3D *           mGLOBMasq3D;
         cElNuage3DMaille*      mGLOBNuage;
         const cMultiCorrelPonctuel * mMCP;

         std::vector<float>     mStatCNC;
         std::vector<float>     mStat1Pix;
         std::vector<float>     mStatCrois;
         bool                   mDoStatCorrel;

         bool mCorrecAlti4ExportIsInit;
         double mValmCorrecAlti4Export;

          
};

std::string  StdNameFromCple
             (
                   cElRegex_Ptr & anAutom,
                   const std::string & aPatSel,
                   const std::string & aPatRes,
                   const std::string & aSep,
                   const std::string & aName1,
                   const std::string & aName2
             );

bool IsModeIm1Maitre(const eModeAggregCorr &);


/*****************************************************/
/*                                                   */
/*               cLibDynAllocator                    */
/*                                                   */
/*****************************************************/


template <class Type> class cLibDynAllocator
{
       typedef Type * (*tPtrFuncAlloc)                       (
                             const std::string & aName2,
                             const std::string & aName
                      );
    public :

          cLibDynAllocator
          (
               const cAppliMICMAC & anAppli,
               const std::string & aNameLibraire,
               const std::string & aNameSymb
          )  :
             // mAllocateur (reinterpret_cast<tPtrFuncAlloc>(anAppli.AllocObjFromLibDyn(aNameLibraire,aNameSymb)))
             mAllocateur (tPtrFuncAlloc(anAppli.AllocObjFromLibDyn(aNameLibraire,aNameSymb)))
          {
          }
          Type *AllocObj
                (
                     const std::string & aNameRessource,
                     const std::string & aTypeRessource
                )
          {
               return (*mAllocateur)(aNameRessource,aTypeRessource);
          }
    private :
        tPtrFuncAlloc  mAllocateur;
};

bool Apply(ePxApply aPA,int aK);


extern Pt2di PBug;
bool IsPBug(const Pt2di &aP);
extern bool GlobDebugMM;  // Init par DebugMM()
/*
extern Pt2di BoxP0Bug;
extern bool  InBoxBug;
bool BoxContainPBug(Box2di const &);
*/
Fonc_Num  FiltrageImMicMac
                 (
                      const cSpecFitrageImage & aParam,
                      Fonc_Num aFonc,
                      Fonc_Num aFoncMasq,
                      double      aSRA
                 );

Fonc_Num  FiltrageImMicMac
                 (
                      const std::vector<std::string> & aParam,
                      Fonc_Num aFonc,
                      Fonc_Num aFoncMasq
		 );

void DebugPxTrsv(const cAppliMICMAC & anAp);


bool ModeGeomIsIm1InvarPx(const cParamMICMAC & aParam);


template <class TypeEl,class tBase> cInterpolateurIm2D<TypeEl>  * InterpoleOfEtape(const cEtapeMEC & anEt,TypeEl *,tBase *);

void GenTFW(const cFileOriMnt & aFOM,const std::string & aName);


void TestGeomTer(const cFileOriMnt & aFOM,const std::string & aMessage);

// Fonction extern pour appel depuis CPU ou GPU
void CombleTrouPrgDyn (
         const cModulationProgDyn & aPrgD,
         Im2D_Bits<1>  aMaskCalc,
         Im2D_Bits<1>  aMaskTer,
         Im2D_INT2     aImZ
     );

// Fonction du ratio compris entre -1 et 1; 0 qd egaux
double EcartNormalise(double aI1,double aI2);






#endif //  _ELISE_MICMAC_ALL_H_




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3  
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
