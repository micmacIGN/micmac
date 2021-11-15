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



#ifndef _ELISE_NUAGE_3D_MAILLE_  // general
#define _ELISE_NUAGE_3D_MAILLE_

// Classe d'interface, les classes "concretes" derivees sont definies dans
// un ".cpp".  Pour creer un objet il faut passer
// par les allocateurs static


   class cXML_ParamNuage3DMaille;

cXML_ParamNuage3DMaille CropAndSousEch
                        (
                             const cXML_ParamNuage3DMaille & anInit,
                             Pt2dr & aP0,
                             double aSc,
                             Pt2dr & aSz
                        );
cXML_ParamNuage3DMaille CropAndSousEch
                        (
                             const cXML_ParamNuage3DMaille & anInit,
                             Pt2dr & aP0,
                             double aSc
                        );


class cFileOriMnt;
cFileOriMnt ToFOM(const cXML_ParamNuage3DMaille &,bool StdRound);


class  cBasculeNuage;

class cLayerNuage3DM
{
    public :
       cLayerNuage3DM(Im2DGen*, const std::string &);

       void PlyPutHeader(FILE *) const;
       void PlyPutData(FILE *,const Pt2di &,bool aModeBin) const;
       Im2DGen * Im() const;
       const std::string & Name() const;
       bool  Compatible(const cLayerNuage3DM &) const;
       ~cLayerNuage3DM();


    private :
       cLayerNuage3DM(const cLayerNuage3DM &);
       Im2DGen *mIm;
       std::string mName;
};


class cGrpeLayerN3D
{
    public :
       cGrpeLayerN3D(int aK1,int aK2,const std::string & aName);
       std::string mName;
       int         mK1;
       int         mK2;
};


class cArgAuxBasc
{
    public :
       cArgAuxBasc(Pt2di aSz);

       Im2D_U_INT1  mImInd;
       Im2D_Bits<1> mImTriInv;
};

class cArgAuxBasc_Sec : public cArgAuxBasc
{
    public :
       cArgAuxBasc_Sec(Pt2di aSz);

       Im2D_REAL4  mImZ;
};


class cRawNuage
{
    public :
        cRawNuage(Pt2di aSz);
        void SetPt(const  Pt2di & anI,const Pt3dr & aP);
        Pt3dr GetPt(const Pt2di & anI) const;
        Im2D_REAL4 ImX();
        Im2D_REAL4 ImY();
        Im2D_REAL4 ImZ();
/*
*/
    private :
        Im2D_REAL4          mImX;
        TIm2D<REAL4,REAL8>  mTX;

        Im2D_REAL4 mImY;
        TIm2D<REAL4,REAL8>  mTY;

        Im2D_REAL4 mImZ;
        TIm2D<REAL4,REAL8>  mTZ;
};


// Ces parametres permettent de creer des meta donnees pour manipuler le
// nuage a une resol differente de celle a laquelle il a ete cree sans
// modifier les donnees maillees
class cParamModifGeomMTDNuage
{
     public :
        cParamModifGeomMTDNuage
        (
            double aScale,  // Par ex= 2 si nuage cree a ss resol 2 et exploite a resol 2
            Box2dr aBox,    // a la resolution d'exploitation
            bool   aDequant = false
        );

        double mScale;
        Box2dr mBox;
        bool   mDequant;
};

double DynProfInPixel(const cXML_ParamNuage3DMaille &);

class cElNuage3DMaille : public cCapture3D
{
     public :
        // return 0 si pas de pb
        virtual double SeuilDistPbTopo() const;

        Pt2di    SzBasicCapt3D() const;
        bool  CaptHasData(const Pt2dr &) const ;
        Pt2dr    Ter2Capteur   (const Pt3dr & aP) const;
         bool     PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage  * =0) const ;
        ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const;
        bool  HasRoughCapteur2Terrain() const ;
        Pt2dr ImRef2Capteur   (const Pt2dr & aP) const ;
        double ResolImRefFromCapteur() const ;
        bool  HasPreciseCapteur2Terrain() const ;
        Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const ;
        Pt3dr PreciseCapteur2Terrain   (const Pt2dr & aP) const ;
        double ResolSolOfPt(const Pt3dr &) const;
        double ResolSolGlob() const;



        cBasicGeomCap3D *   Cam() const;
        cRawNuage   GetRaw() const;
   // Lecture-Creation  globale

        static cElNuage3DMaille * FromFileIm(const std::string & aFile);
        void Save(const std::string & Name);  // Name+Prof.tif   Name+Masq.tif
        virtual cElNuage3DMaille * Clone() const = 0;
        virtual void ProfBouchePPV() = 0;
        void SetNbPts(int val);
        int GetNbPts();

   // Parcourt par des iterateurs

        typedef Pt2di tIndex2D;
        typedef Pt3di tTri;

        virtual tIndex2D  Begin() const;   // [1]
        virtual tIndex2D  End() const;
        virtual void  IncrIndex(tIndex2D &) const;


   // Acces direct aux index
        Im2D_Bits<1>   ImDef();
        void SetVoisImDef(Im2D_Bits<1>);
        bool IsEmpty();
        bool  IndexHasContenu(const tIndex2D & anI) const
        {
             return (mTImDef.get(anI,0) != 0);
        }

        bool  IndexHasContenuAsNeighboor(const tIndex2D & anI) const
        {
             return (mTVoisImDef.get(anI,0) != 0);
        }
        void SetNormByCenter(int val);
        void SetDistCenter(double val);

        bool  IndexHasContenuForInterpol(const tIndex2D & aP) const
        {
             return (mTImDefInterp.get(aP,0) != 0);
        }
        bool  IndexHasContenuForInterpol(const Pt2dr & aP) const
        {
             return IndexHasContenuForInterpol(round_down(aP));
        }
        const Pt2di & SzGeom() const {return mSzGeom;}
        const Pt2di & SzData() const {return mSzData;}

        const Pt2di & SzUnique() const;//  const {return mSzGeom;}
   //  Acces aux donnees

                 // NUAGE  =>  TERRAIN

        // Si le nuage vient d'un appariement, a combien de pixel de
        // decalage est equivalent 1 offset de profondeur
        double DynProfInPixel() const;

        Pt3dr PtOfIndex(const tIndex2D & aP) const ;
        Pt3dr PtOfIndexInterpol(const Pt2dr & aP) const;  // [2]
        Pt3dr NormaleOfIndex(const tIndex2D&, int, const Pt3dr&) const;


        virtual Pt3dr Loc_PtOfIndex(const tIndex2D & aP) const = 0;
        virtual Pt3dr Loc_PtOfIndexInterpol(const Pt2dr & aP) const;  // [2]


                 //  TERRAIN  => NUAGE
         Pt2dr   Terrain2Index(const Pt3dr &) const;

                 // VIA LES PROFONDEURS
         virtual bool HasProfondeur() const = 0;
         virtual void    SetProfOfIndex(const tIndex2D & aP,double ) ;
         virtual bool    SetProfOfIndexIfSup(const tIndex2D & aP,double ) ;
         virtual double  ProfOfIndex(const tIndex2D & aP) const ;
         virtual double   ProfOfIndexInterpol(const Pt2dr  & aP)const ;
         virtual double   ProfOfIndexInterpolWithDef(const Pt2dr  & aP,double aDef)const =0 ;

    // Si la profondeur n'est vraiment pas euclidienne (genre ZInv), la corrige

         virtual double  ProfEuclidOfIndex(const tIndex2D & aP) const ;
         virtual void    SetProfEuclidOfIndex(const tIndex2D & aP,double ) ;


         // Renvoie en Z avec une dynamique telle que Dz de 1 corresponde a 1 Pixel,
         // tape dans Im2D Gen
         virtual double  ProfEnPixel(const tIndex2D & aP) const ;
         virtual double  ProfInterpEnPixel(const Pt2dr & aP) const ;

                 // Composante geometrique "pure" du capteur (ie independamment des profs)

          ElSeg3D FaisceauFromIndex(const Pt2dr & aP) const;
          virtual Pt3dr Loc_IndexAndProf2Euclid(const   Pt2dr &,const double &) const = 0;
          Pt3dr IndexAndProf2Euclid(const   Pt2dr &,const double &) const;


          virtual Pt3dr Loc_Euclid2ProfAndIndex(const   Pt3dr &) const = 0;
          Pt3dr Euclid2ProfAndIndex(const   Pt3dr &) const;

          virtual Pt3dr Loc_IndexAndProfPixel2Euclid(const   Pt2dr &,const double &) const = 0;
          Pt3dr IndexAndProfPixel2Euclid(const   Pt2dr &,const double &) const;


          virtual Pt3dr Loc_Euclid2ProfPixelAndIndex(const   Pt3dr &) const = 0;
          Pt3dr Euclid2ProfPixelAndIndex(const   Pt3dr &) const ;

         virtual Im2DGen *  ImProf() const;
          Im2D_Bits<1>    ImMask() const;
   //  Modification

        void  SetPtOfIndex(const tIndex2D & aP,const Pt3dr &) ; // Deconseillee car inexacte
        virtual void  V_SetPtOfIndex(const tIndex2D & anI,const Pt3dr & aP3) = 0;
        void  SetNoValue(const tIndex2D & aP) ;

   //  Manipulation plus "elaboree"

             // "Bascule" N2 dans la geometrie de this, par algo de ZBUfer
// NewName pour retracer les appels....
        cElNuage3DMaille * BasculementNewName( const cElNuage3DMaille * N2,
                                        bool SupprTriInv,
                                        double aCoeffEtire = -1
                                      ) const;
        Im2D_U_INT1  ImEtirement();
        // A ete modifier pour debut de dessin  en perspective, le premier
        // cArgAuxBasc permet de memoriser le ZMAX (car ZBuf remet a 0) et le deuxieme
        // permet de memoriser la deuxieme valeur pour eventuelle transparence

        // Nouvelle modif pour permettre de resize au + pres le resultat
        // Si resize, se fait en renvoyant une nouvelle version

        cElNuage3DMaille *   BasculeInThis
               (
                   const cXML_ParamNuage3DMaille * aGeomOutOri,
                   const cElNuage3DMaille * N2,
                   bool SupprTriInv,
                   double aCoeffEtire,
                   cArgAuxBasc *,
                   cArgAuxBasc_Sec *,
                   int aLabel,
                   bool  AutoResize,
                   std::vector<Im2DGen *> *
               );






        double DiffDeSurface(bool& OK,const tIndex2D&,const cElNuage3DMaille &) const;



/*
       [1]  Partie eventuellement re-definissable (les "iterateurs")
        Par ex a redef si l'image est tres creuse

       [2]     Ca peut etre + rapide de faire l'interpol sur image,
              d'ou le virtual
*/


        // A priori l'indexation est independante d'un espace geometrique sous jacent,
        // cependant en general il y a une correlation forte entre les 2. Si c'est le
        // cas, ces trois fonctions remplissent le role;
        //
        // Typiquement les coordonnees plani seront l'espace image (ou terrain) du nuage
        // et la transfo sera le clip-scale donne par l'orientation interne
        //
        virtual bool IndexIsPlani() const ;
        virtual Pt2dr  Index2Plani(const Pt2dr &) const ;
        virtual Pt2dr  Plani2Index(const Pt2dr &) const ;

        //==================

        bool  Compatible(const cElNuage3DMaille &) const;
        void PlyPutFile
             (
                  const std::string & aName,const std::list<std::string>& aComments, bool aModeBin,
                  bool SavePtsCol = true,
                  int aAddNormale=0,
                  const std::list<std::string>& aNormName = {},
                  bool DoublePrec = false,
                  const Pt3dr& anOffset = Pt3dr(0,0,0)
             ) const;

        static void PlyPutFile
               (
                    const std::string & aName,
                    const std::list<std::string> &aComments,
                    const std::vector<const cElNuage3DMaille *> &,
                    const std::vector<Pt3dr> * mPts,
                    const std::vector<Pt3di> * mCouls,
                    bool aModeBin,
                    bool SavePtsCol = true,
                    int aAddNormale = 0,
                    const std::list<std::string>& aNormName = {},
                    bool DoublePrec = false,
                    const Pt3dr& anOffset = Pt3dr(0,0,0)
                ) ;

        void Std_AddAttrFromFile(const std::string & aName,double aDyn=1,double aScale=1,bool ForceRGB=false);


        cElNuage3DMaille * ReScaleAndClip(double aScale);
        cElNuage3DMaille * ReScaleAndClip(Box2dr,double aScale);



         void AddExportMesh();

      // Test les donnees de verification, par defaut (nuage 3D natif) ne fait rien
        virtual void VerifParams() const;

        // Partie commune
        //
        //  Un ElCamera a deja la bonne interface pour definir la
        //  geometrie d'acquisition du nuage. Si c'est un nuage en geometrie
        // camera stenope c'est trivial. Si c'est un nuage en terrain, on
        // le fera avec un camera de projection orthographique

        cElNuage3DMaille
        (
             const std::string & aDir,
             const cXML_ParamNuage3DMaille &,
             Fonc_Num aFDef,
             const std::string & aNameFile,
             bool      WithEmptyData = false
        );
        const std::string & NameFile() const;
        virtual ~cElNuage3DMaille();

        static cElNuage3DMaille * FromFileIm
                                  (
                                      const std::string & aFile,
                                      const std::string &  aTag,
                                      const std::string &  aMasq="",
                                      double ExagZ         =1.0
                                  );
        static cElNuage3DMaille * FromParam
                                  (
                                       const std::string & aNameFile,
                                       const cXML_ParamNuage3DMaille &,
                                       const std::string & aDir,
                                       const std::string & aMasq = "",
                                       double ExagZ         =1.0,
                                       const cParamModifGeomMTDNuage * = 0,
                                       bool  WithEmptyData = false
                                  );
/*

*/


        // int   Tx() const {return mSz.x;}
        // int   Ty() const {return mSz.y;}

        cXML_ParamNuage3DMaille&  Params();
        const cXML_ParamNuage3DMaille&  Params() const;


        // Juste pour fixer une fois pour toute les conventions
        Fonc_Num   ReScaleAndClip(Fonc_Num aF,const Pt2dr & aP0,double aScale);
        const std::vector<cLayerNuage3DM *> &  Attrs() const;

        void NuageXZGCOL(const std::string & aName,bool B64=false);

        bool  IndexInsideGeom(const tIndex2D & aP) const
        {
            return     (aP.x >= 0)
                    && (aP.y >= 0)
                    && (aP.x < mSzGeom.x)
                    && (aP.y < mSzGeom.y);
        }
        bool  IndexInsideData(const tIndex2D & aP) const
        {
            return     (aP.x >= 0)
                    && (aP.y >= 0)
                    && (aP.x < mSzData.x)
                    && (aP.y < mSzData.y);
        }
        bool  IndexIsOK(const tIndex2D & aP) const
        {
             return IndexInsideData(aP) && IndexHasContenu(aP);
        }

        bool IndexIsOKForInterpol(const Pt2dr & aP) const
        {
             Pt2di aQ = round_down(aP);
             return IndexInsideData(aQ) && IndexHasContenuForInterpol(aQ);
        }


     private :

        void   FinishBasculeInThis
               (
                    cBasculeNuage &  aBasc,
                   Im2D_REAL4 aMntBasc,
                   Pt2di anOfOut,
                   bool SupprTriInv,
                   double aCoeffEtire,
                   cArgAuxBasc *,
                   cArgAuxBasc_Sec *,
                   int aLabel
               );


        void AddAttrFromFile ( const std::string &              aName,
                               int                              aFlagChannel,
                               const std::vector<std::string> & aNameProps,
                               double aDyn=1,
                               double aScale=1,
                               bool ForceRGB=false
                             );
        void PlyHeader(FILE *,bool aModeBin) const;
        void PlyPutDataVertex(FILE *,bool aModeBin, int aAddNormale,bool DoublePrec,const Pt3dr & anOffset) const;
        void PlyPutDataFace(FILE *,bool aModeBin,int & anOffset) const;

        virtual cElNuage3DMaille * V_ReScale
                                   (
                                        const Box2dr &Box,
                                        double aScale,
                                        const cXML_ParamNuage3DMaille &,
                                        Im2D_REAL4 anImPds,
                                        std::vector<Im2DGen*> aVNew,
                                        std::vector<Im2DGen*> aVOld
                                   ) = 0;
        virtual void V_Save(const std::string & aNameP) = 0;


        void  AssertInsideData(const tIndex2D & anI) const;
        void  AssertInsideGeom(const tIndex2D & anI) const;
        void AssertCamInit() const;






        void UpdateDefInterp(const Pt2di & aP);
        void UpdateVoisAfterModif(const Pt2di & aP);



        void IncrIndSsFiltre(Pt2di & aP) const;
        void AddGrpeLyaer(int aNb,const std::string & aName);


        Pt3dr  Loc2Glob(const Pt3dr &) const;
        Pt3dr  Glob2Loc(const Pt3dr &) const;

        void GenTri(std::vector<tTri> &,const tIndex2D &,int aOffset) const;
        void AddTri(std::vector<tTri> &,const tIndex2D &,int *K123,int aOffset) const;
        double TriArea(const Pt3dr &,const Pt3dr &, const Pt3dr &) const;

     protected  :
        void AssertNoEmptyData() const;

        void PlyPutDataOneFace(FILE *,const tTri& , bool aModeBin) const;


        int                          mITypeCam; // RPCNuage
        bool                         mEmptyData;
        std::string                  mDir;
        cInterfChantierNameManipulateur * mICNM;
        cXML_ParamNuage3DMaille&     mParams;
        Pt2di                        mSzGeom;
        Pt2di                        mSzData;
        Im2D_Bits<1>                 mImDef;
        int                          mNbPts;
        TIm2DBits<1>                 mTImDef;
        Im2D_Bits<1>                 mImDefInterp;
        TIm2DBits<1>                 mTImDefInterp;
        cBasicGeomCap3D *            mCam;
        Im2D_U_INT1                  mImEtire;


       // En general egale a mImDef, peut etre diff pour nuage en "peau de leopard"
       // Utilise pour les fon de voisinage telle que Normale

        Im2D_Bits<1>                  mVoisImDef;
        TIm2DBits<1>                  mTVoisImDef;
        int                           mNormByCenter;
        double                        mDistCenter;

        cChCoCart *                   m2RepGlob;
        cChCoCart *                   m2RepLoc;
        cInterfSurfaceAnalytique *    mAnam;


        std::vector<cLayerNuage3DM*>     mAttrs;
        std::vector<cGrpeLayerN3D>       mGrpAttr;


        bool                             mGenerMesh;
        Im2D_INT4                        mNumPts;
        TIm2D<INT4,INT>                  mTNumP;
        int                              mNbTri;
        mutable bool                     mResolGlobCalc;
        mutable double                   mResolGlob;
        std::string                      mNameFile;
};



typedef double tElZB;

class cZBuffer
{
    public :
    // A Rajouter si necessaire, les Box de l'ancien ZBuffer quand
    // la zone du MNT ne couvre qu'une partie du resultat (pour savoir
    // rapidement quelles zones ne pas explorer)

        virtual ~cZBuffer();

        // OrigineIn, StepIn, ...  : parametrise la discretisation du terrain
        // Ne sert qu'a definir la grille, en aucune maniere une espace;
        // on peut rajouter a OrigineIn (ou OrigineOut) n'importe quel multiple
        // de StepIn  (vs StepOut) ca NE DEVRAIT RIEN CHANGER (a verifier quand meme
        // car commentaire a posteriori)

        cZBuffer
        (
            Pt2dr OrigineIn,
            Pt2dr StepIn,
            Pt2dr OrigineOut,
            Pt2dr StepOut
        );

        // Fait le basculement "standard" ou l'interpolateur est
        // une triangulation
        Im2D_REAL4 Basculer
                   (
                       Pt2di & aOffset_Out_00,
                       Pt2di aP0In,
                       Pt2di aP1In,
                       float aZDef,  // aZDef doit etre suffisement bas
                       bool  * Ok  = 0
                   );

        Im2D_REAL4 ZCaches
                   (
                       Im2D_REAL4 aMnt,
                       Pt2di aOffset_Out_00,
                       Pt2di aP0In,
                       Pt2di aP1In,
                       float aDef
                   );
        Im2D_REAL4 ZCaches(Pt2di aP0In,Pt2di aP1In,float aDef);

        // Initialise d'abord un basculement "standard", ensuite utilise
        // un schema iteratif, pour calculer le basculement d'un
        // interpolateur specifie par ZInterpofXY
        Im2D_REAL4 BasculerAndInterpoleInverse
                   (
                       Pt2di & aOffset_Out_00,
                       Pt2di aP0In,
                       Pt2di aP1In,
                       float aZDef,  // aZDef doit etre suffisement bas
		       bool * Ok=nullptr
                   );

        Im2D_Bits<1> ImOkTer() const;
        // Image des triangles inverses. Au depart ces triangles ne faisaient pas partie
        // du ZBUF. Certaines applis en ont besoin, d'autres sont perturbees. Au final
        // on memorise l'info, on la met a dispo et chacun se dem...
        Im2D_Bits<1> ImTriInv() const;
        bool OkTer(const Pt2di &) const;

       // Utilise le "MNT" pour trouver le Z, si il y a  un buff,
       // fait une simple lecture
        Pt3dr ProjDisc(const Pt2di  &,double * aPtrValZofXY=0) const;
        Pt3dr ProjReelle(const Pt2dr  &,bool &OK) const;


        // Taille de l'espace d'arrivee apres la p
        Pt2di P0_Out() const;
        Pt2di SzOut() const;

        // Projection discrete au sens ou elle tient compte des pas,
        // mais tout peut etre reel
        Pt3dr ProjDisc(const Pt3dr &) const;
        void SetWithBufXYZ(bool);

        Pt3dr ToCoordInAbs(const Pt3dr & aPInDisc) const;
        Pt3dr ToCoordOutLoc(const Pt3dr & aPOutTer) const;


        void InitDynEtirement(double);
        Im2D_U_INT1    ImEtirement();
        void SetEpsilonInterpoleInverse(double anEps);

        Pt3dr InverseProjDisc(const Pt3dr &) const;
        Pt3dr InverseProjDisc(const Pt2di  &) const;

        void AddImAttr(Im2DGen *);
        std::vector<Im2DGen *> AttrOut();
        void SetRPC(bool IsRPC,double aZMin,double aZMax);

        virtual bool RPCIsBascVisible(const Pt3dr & aP) const;


   private :

        Pt2di ToPtIndexDef(const Pt2di & aPt) const;

        std::vector<Im2DGen *>  mImAttrIn;
        std::vector<Im2DGen *>  mImAttrOut;

        // Projection native, sans tenir compte des pas
        virtual Pt3dr ProjTerrain(const Pt3dr &) const = 0;
        virtual double ZofXY(const Pt2di & aP)   const = 0; // En general le MNT
        // SelectP : espace terrain
        virtual bool SelectP(const Pt2di & aP)   const  ; // def return true
        // SelectPBasc : espace "image", c.a.d apres basculement
        virtual bool SelectPBascul(const Pt2dr & aP)   const  ; // def return true

                 // Par defaut erreur fatale pour ces 2 la, qui ne sont
                 // utilises que pour raffiner
        virtual double ZInterpofXY(const Pt2dr & aP,bool & OK) const;
        virtual  Pt3dr InvProjTerrain(const Pt3dr &) const;


        void BasculerUnTriangle(Pt2di A,Pt2di B,Pt2di C,bool TriBas);

        Pt2dr mOrigineIn;
        Pt2dr mStepIn;
        Pt2dr mOrigineOut;
        Pt2dr mStepOut;
        Pt2di mOffet_Out_00;
        Pt2di mSzRes;


        Im2D_REAL4 mRes;
        float ** mDataRes;
        Im2D_Bits<1> mImOkTer;
        TIm2DBits<1> mTImOkTer;
        Im2D_Bits<1> mImTriInv;
        double       mDynEtire;
        Im2D_U_INT1  mImEtirement;


        bool         mWihBuf;
        bool         mBufDone;

        Im2D<tElZB,REAL8>   mImX3;
        tElZB **     mDX3;
        TIm2D<tElZB,REAL8>  mTX3;
        Im2D<tElZB,REAL8>   mImY3;
        tElZB **     mDY3;
        TIm2D<tElZB,REAL8>  mTY3;
        Im2D<tElZB,REAL8>   mImZ3;
        tElZB **     mDZ3;
        TIm2D<tElZB,REAL8>  mTZ3;
        Pt2di        mP0In;
        Pt2di        mSzIn;

        double       mEpsIntInv;

        //    (A B)  est l'inverse de la matrice qui envoie un pixel (1,0) (0,1) Z=Moyen vers l'espace d'arrivee
        //    (C D)
        TIm2D<REAL8,REAL8>   mTImDef_00;
        TIm2D<REAL8,REAL8>   mTImDef_10;
        TIm2D<REAL8,REAL8>   mTImDef_01;
        TIm2D<REAL8,REAL8>   mTImDef_11;
        bool                 mIsRPC;
        double               mZMinRPC;
        double               mZMaxRPC;
};

class cArgBacule
{
    public :
        cArgBacule(double aSeuilEtir);

        double       mSeuilEtir;
        double       mDynEtir;
        bool         mAutoResize;
        Box2di *     mBoxClipIn;
        Im2D_U_INT1  mResEtir;
};

cElNuage3DMaille *  BasculeNuageAutoReSize
                    (
                       const cXML_ParamNuage3DMaille & aGeomOut,
                       const cXML_ParamNuage3DMaille & aGeomIn,
                       const std::string & aDirIn,
                       const std::string &  aNameRes,
                       cArgBacule &
/*
                       bool  AutoResize,
                       const Box2di  * aBoxClipIn = 0,
                       const cArgBacule &  = cArgBacule::mTheDef
*/
                    );

template <class Type> void WriteType(FILE * aFP,Type f)
{
    size_t  size = sizeof(Type);
    TheIntFuckingReturnValue = (int)fwrite(&f,size,1,aFP);
}


cElNuage3DMaille * NuageWithoutDataWithModel(const std::string & aName,const std::string & aModel);
cElNuage3DMaille * NuageWithoutData(const std::string & aName);
cElNuage3DMaille * NuageWithoutData(const cXML_ParamNuage3DMaille & aParam,const std::string & aName) ;

cXML_ParamNuage3DMaille XML_Nuage(const std::string & aName);
bool GeomCompatForte(cElNuage3DMaille * aN1,cElNuage3DMaille *aN2);

Fonc_Num Pix2Z(const cXML_ParamNuage3DMaille & aCloud,Fonc_Num aF);
Fonc_Num Z2Pix(const cXML_ParamNuage3DMaille & aCloud,Fonc_Num aF);
Fonc_Num Pix2Pix(const cXML_ParamNuage3DMaille &Out,Fonc_Num,const cXML_ParamNuage3DMaille & In);

Fonc_Num Pix2Pix(const cXML_ParamNuage3DMaille &Out,const cXML_ParamNuage3DMaille & In,const std::string& aDir)
;



class cMasqBin3D
{
     public :
        virtual bool IsInMasq(const Pt3dr &) const = 0;
        virtual ~cMasqBin3D();
        static cMasqBin3D * FromSaisieMasq3d(const std::string & aName);

        Im2D_Bits<1>  Mas2DPointInMasq3D(const cElNuage3DMaille &);

     private :
};


#endif // _ELISE_NUAGE_3D_MAILLE_


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
