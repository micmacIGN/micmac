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



#ifndef _ELISE_GENERAL_PHOTOGRAM_H
#define _ELISE_GENERAL_PHOTOGRAM_H

#include <list>


//  La distorsion est-elle codee de maniere privilegiee dans le sens
//  Monde->Cam (M2C,=true) ou Cam->Mond (C2M), jusqu'a present (mars 2008)
//  c'etait dans le
//  sens C2M car c'est le sens utile pour les point de liaison avec une
//  equation de coplanarite. Les convention hors elise sont plutot C2M
//
//  Avec l'introduction des residus terrain (point 3D subsitues), on
//  peut faire l'un ou l'autre. Les avantage de M2C sont :
//
//    - coherence avec le reste du monde,
//    - residu vraiment images
//
//  Les avantages de C2M :
//
//     - on sait a 100% que ca marche, puisque c'est le systeme actuel
//     - on peut continuer a utiliser l'equation de coplanarite
//     (mais, est elle utile avec les points terrain ?)
//
//
//
//
//
#define  ElPrefDist_M2C  true

extern bool NewBug;
extern bool DoCheckResiduPhgrm;
extern bool AcceptFalseRot;

// Definis dans phgr_formel.h
class cSetEqFormelles;
class cParamIntrinsequeFormel;
class cParamIFDistStdPhgr;
class cParamIFDistRadiale;



class ElSeg3D;
class Appar23;
class ElCplePtsHomologues;
class ElPackHomologue;
class StatElPackH;
class ElPhotogram;
class ElProj32;
template <class Type> class  ElProjStenopeGen;
class ElProjStenope;
class ElDistortion22_Gen;
class ElDistortion22_Triviale;
class ElDistRadiale;
class ElDistRadiale_PolynImpair;
class ElDistRadiale_Pol357;
class PolyDegre2XY;
class ElDistPolyDegre2;

class ElDistortionPolynomiale;
class ElCamera;
class CamStenope;
class CameraRPC;
class cRPC;
class cCamStenopeGen;
class CamStenopeIdeale;
class CalcPtsInteret;
class cCamStenopeDistRadPol;

class cDistorBilin;
class cCamStenopeBilin;

class PolynomialEpipolaireCoordinate;
class EpipolaireCoordinate;

class CpleEpipolaireCoord;
class cElHomographie;

class cElemMepRelCoplan;
class cResMepRelCoplan;

class cDbleGrid ;
class cDistCamStenopeGrid;
class cCamStenopeGrid;

class cMirePolygonEtal;


class cCalibrationInternConique;
class cOrientationConique;
class cVerifOrient;
class cCalibDistortion;
class cCalibrationInterneRadiale;
class cCalibrationInternePghrStd;
class cGridDirecteEtInverse;
class cParamForGrid;
class cPreCondGrid;
class cCorrectionRefractionAPosteriori;

class Appar23
{
    public :
        Pt2dr pim;
        Pt3dr pter;
        int   mNum;  // Rajoutes pour gerer les cibles

        Appar23 (Pt2dr PIM,Pt3dr PTER,int aNum=-1) ;

    private  :

};
Appar23  BarryImTer(const std::list<Appar23> &);
void InvY(std::list<Appar23> &,Pt2dr aSzIm,bool InvX=false);


// extern bool BugSolveCstr;
// extern bool BugCstr;
extern bool BugFE; // pour debuguer les pb de non inversibilite des dist fortes
extern bool BugAZL; // pour debuguer les pb d'AZL
extern bool BugGL; // pour debuguer les pb de Guimbal Lock


class cProjCple
{
     public :
          cProjCple(const Pt3dr &,const Pt3dr &,double aPds);
          const Pt3dr & P1() const;
          const Pt3dr & P2() const;

          static cProjCple Spherik(const ElCamera & aCam1,const Pt2dr & ,const ElCamera & aCam2,const Pt2dr &,double aPds);
          static cProjCple Projection(const ElCamera & aCam1,const Pt2dr & ,const ElCamera & aCam2,const Pt2dr &,double aPds);
     private :
           Pt3dr  mP1;
           Pt3dr  mP2;
           double mPds;
};

class cProjListHom
{
     public :
          typedef std::vector<cProjCple>      tCont;
          typedef tCont::iterator           tIter;
          typedef tCont::const_iterator     tCstIter;
          tCstIter & begin() const;
          tCstIter & end() const;

          cProjListHom(  const ElCamera & aCam1,const ElPackHomologue & aPack12,
                         const ElCamera & aCam2,const ElPackHomologue & aPack21,
                         bool Spherik
                      );
     public :
          tCont       mLClpe;
          bool        mSpherik;
};



class cNupletPtsHomologues
{
     public :
    ElCplePtsHomologues & ToCple();
    const ElCplePtsHomologues & ToCple() const;

// Uniquement en dim 2
        const Pt2dr & P1() const ;
        Pt2dr & P1() ;
        const Pt2dr & P2() const ;
        Pt2dr & P2() ;



        const REAL & Pds() const ;
        REAL & Pds() ;

    cNupletPtsHomologues(int aNb,double aPds=1.0);
    int NbPts() const;

    const Pt2dr & PK(int aK) const ;
        Pt2dr & PK(int aK) ;

    void write(class  ELISE_fp & aFile) const;
        static cNupletPtsHomologues read(ELISE_fp & aFile);

        void AddPts(const Pt2dr & aPt);

        bool IsDr(int aK) const;
        void SetDr(int aK);

     private :
        void AssertD2() const;
        std::vector<Pt2dr> mPts;
        REAL  mPds;
  // Gestion super bas-niveau avec des flag de bits pour etre compatible avec la structure physique faite
  // quand on ne contenait que des points ....
        int   mFlagDr;
        void AssertIsValideFlagDr(int aK) const;
        bool IsValideFlagDr(int aK) const;

};

class ElCplePtsHomologues : public cNupletPtsHomologues
{
     public :

        ElCplePtsHomologues (Pt2dr aP1,Pt2dr aP2,REAL aPds=1.0);

        const Pt2dr & P1() const ;
        Pt2dr & P1() ;

        const Pt2dr & P2() const ;
        Pt2dr & P2() ;


        // Box2D
         void SelfSwap(); // Intervertit les  2

      double Profondeur(const ElRotation3D & aR) const;

     private :

};

enum eModeleCamera
{
   eModeleCamIdeale,
   eModeleCamDRad,
   eModeleCamPhgrsStd
};


class cResolvAmbiBase
{
     public :
     // Les orientations sont des orientations tq R.ImAff(0) est le centre optique, Cam->Monde
        cResolvAmbiBase
    (
         const ElRotation3D &  aR0,   // Orientation connue completement
         const ElRotation3D &  aR1   // Orientation connue a un facteur d'echelle pres sur la base
    );

    void AddHom(const ElPackHomologue & aH12,const ElRotation3D & aR2);
        double SolveBase();
    ElRotation3D SolOrient(double & aLambda);


     private :

          Pt3dr mC0;
      Pt3dr mV01;
      ElRotation3D mR1;
          std::vector<double> mLambdas;
};

// txt : format texte,  dat : format binaire (int , double[3] *)
std::vector<Pt3dr> * StdNuage3DFromFile(const std::string &);


// Representation des points homologues comme images, utiles lorsqu'ils
// sont denses et + ou - regulierement espaces avec une image maitresse

class cElImPackHom
{
     public :
        cElImPackHom(const ElPackHomologue &,int mSsResol,Pt2di aSzR);
        cElImPackHom(const std::string &);
        void AddFile(const std::string &);
    void SauvFile(const std::string &);
        int NbIm() const;  // Minimum 2
    ElPackHomologue  ToPackH(int aK);   // aK commence a O
    Pt2di Sz() const;

    Pt2dr P1(Pt2di);
    Pt2dr PN(Pt2di,int aK);
    double PdsN(Pt2di,int aK);
     private :
    void VerifInd(Pt2di aP);
    void VerifInd(Pt2di aP,int aK);

        Pt2di      mSz;
        Im2D_REAL4 mImX1;
        Im2D_REAL4 mImY1;
    std::vector<Im2D_REAL4> mImXn;
    std::vector<Im2D_REAL4> mImYn;
    std::vector<Im2D_REAL4> mImPdsN;

};




class cPackNupletsHom
{
     public :
         typedef std::list<cNupletPtsHomologues>   tCont;
         typedef tCont::iterator                  tIter;
         typedef tCont::const_iterator            tCstIter;
     cPackNupletsHom(int aDim);

     void write(class  ELISE_fp & aFile) const;
         static cPackNupletsHom read(ELISE_fp & aFile);
         typedef tCont::iterator         iterator;
         typedef tCont::const_iterator   const_iterator;

         cNupletPtsHomologues & back();
         const cNupletPtsHomologues & back() const;

         iterator       begin();
         const_iterator begin() const;
         iterator       end();
         const_iterator end() const;
         INT size() const ;
         void clear();

     void AddNuplet(const cNupletPtsHomologues &);

         const cNupletPtsHomologues * Nuple_Nearest(Pt2dr aP,int aK) const;
         void  Nuple_RemoveNearest(Pt2dr aP,int aK) ;

     const ElPackHomologue & ToPckCple() const;

     protected :
         tCont::iterator  NearestIter(Pt2dr aP,int aK);
     private :
         tCont mCont;
     int   mDim;
};


typedef std::pair<Pt2dr,Pt2dr> tPairPt;

class ElPackHomologue : public cPackNupletsHom
{
     private :

         tCont::iterator  NearestIter(Pt2dr aP,bool P1 = true);


        // utilise pour parametrer l'ajustement dans FitDistPolynomiale
        // Par defaut resoud aux moindre L1, l'aspect virtuel permet
        // de definir une classe ayant exactement le meme
        // comportement
          ElMatrix<REAL> SolveSys(const  ElMatrix<REAL> &);


         void  PrivDirEpipolaire(Pt2dr & aDir1,Pt2dr & aDir2,INT aNbDir) const;
     bool  mSolveInL1;

     public :
         Box2dr BoxP1() const;
         ElPackHomologue();
         void SelfSwap(); // Intervertit les  2
         void ApplyHomographies
              (const cElHomographie &H1,const cElHomographie &);

         ElCplePtsHomologues & Cple_Back();
         const ElCplePtsHomologues & Cple_Back() const;


         void Cple_Add(const ElCplePtsHomologues &);

         const ElCplePtsHomologues * Cple_Nearest(Pt2dr aP,bool P1 = true) const;
         void  Cple_RemoveNearest(Pt2dr aP,bool P1 = true) ;
         static ElPackHomologue read(ELISE_fp & aFile);

         Polynome2dReal  FitPolynome
                         (
                 bool aModeL2,
                             INT aDegre,
                             REAL anAmpl,
                             bool aFitX
                          );

         ElDistortionPolynomiale FitDistPolynomiale
                          (
                             bool aL2,
                             INT aDegre,
                             REAL anAmpl,
                             REAL anEpsInv = 1e-7
                          );
         void  DirEpipolaire(Pt2dr & aDir1,Pt2dr & aDir2,INT WantedPts,INT aNbDir,INT aDegre) const;
         CpleEpipolaireCoord *  DirAndCpleEpipolaire
               (Pt2dr & aDir1,Pt2dr & aDir2,INT WantedPts,INT aNbDir,INT aDegreFinal) const;

      ElMatrix<REAL> MatriceEssentielle(bool SysL2);

      REAL MatriceEssentielle
               (
                       class cGenSysSurResol &,
                       double *  Vect,
                       REAL  EcartForPond
               );


            // Optimise la mise en place relative par un algo generique
            // (powel) de descente sur un critere L1
            ElRotation3D OptimiseMEPRel(const ElRotation3D & );

            // Teste la matrice essentielle et le plan et retient la meilleure
            // Par defaut fait une optimisation , pas forcement opportune
            ElRotation3D MepRelGen(REAL LongBase,bool L2,double & aD);
            ElRotation3D MepRelGenSsOpt(REAL LongBase,bool L2,double & aD);
            ElRotation3D MepRelGen(REAL LongBase,bool L2,double & aD,bool Optimize);


            // Comme dab, en entree des couple "photogrammetrique" en sortie
            // la rotation qui envoie de 1 vers 2
            ElMatrix<REAL> MepRelCocentrique(int aNbRansac,int aNbMaxPts) const;


            //   Toutes les mises en place relatives font les hypotheses suivantes
        //
        //      - points "photogrammetriques" (x,y) -> (x,y,1)
        //      - orientation 1 : identite

            // renvoie les rotation qui permet, etant donne
            // un point en coordonnee camera1, d'avoir
            // ses coordonnees en camera 2
             std::list<ElRotation3D> MepRelStd(REAL LongBase,bool L2);

         // Phys : cherche a avoir le max de couples de rayons
         // qui s'intersectent avec des Z positifs
             ElRotation3D MepRelPhysStd(REAL LongBase,bool L2);

         // Nombre de points ayant une intersection positive en Im1 et Im2
         REAL SignInters
                  (
                       const ElRotation3D & aRot1to2,
                       INT &                NbP1,
                       INT &                NbP2
                  ) const;

             tPairPt  PMed() const;

             // Si tous les points sont coplanaires, ou presque,
             //  la mise en place par l'algo standard est degenere,
             // on choisit donc un algo ad hoc
              cResMepRelCoplan   MepRelCoplan (REAL LongBase,bool HomEstL2);
              static cResMepRelCoplan   MepRelCoplan (REAL LongBase,cElHomographie,const tPairPt & Centre);

              // s'adapte a xml, tif , dat
          static ElPackHomologue   FromFile(const std::string &);
          ElPackHomologue   FiltreByFileMasq(const std::string &,double aVMin=0.5) const;
        // Si Post = xml -> XML; si Post = dat -> Bin; sinon erreur
          void StdPutInFile(const std::string &) const;
          void StdAddInFile(const std::string &) const;
              void Add(const ElPackHomologue &) ;

          //  Les mise en place relatives levent l'ambiguite avec un parametre
          //  de distance, souvent il est plus naturel de fixer la profondeur
          //  moyenne, c'est ce que permet de corriger cette fonction
          //
          void SetProfondeur(ElRotation3D & aR,double aProf) const;
          double Profondeur(const ElRotation3D & aR) const;
              void InvY(Pt2dr aSzIm1,Pt2dr aSzIm2);
          void Resize(double aRatioIm1, double aRatioIm2);
              // Dist moy des intersections
          double AngularDistInter(const ElRotation3D & aR) const;

             void  ProfMedCam2
                   (
                        std::vector<double> & VProf,
                        const ElRotation3D & aR
                    ) const;
             // Quasi equiv a la dist inter et (?) bcp + rapide
             double QuickDistInter
                    (
                         const ElRotation3D & aR,
                         const std::vector<double> & VProf
                    ) const;

};

class StatElPackH
{
      public :
            StatElPackH(const ElPackHomologue &);
            Pt2dr Cdg1 () const;
            Pt2dr Cdg2 () const;
            REAL  RMax1 () const;
            REAL  RMax2 () const;
            INT   NbPts() const;
            REAL  SomD1 () const;
            REAL  SomD2 () const;
      private :
            REAL  mSPds;
            INT   mNbPts;
            Pt2dr mCdg1;
            Pt2dr mCdg2;
            REAL  mRMax1;
            REAL  mRMax2;
            REAL  mSomD1;
            REAL  mSomD2;
};



class ElPhotogram  // + ou - NameSpace
{
    public :
       static  void bench_photogram_0();

       static Pt2dr   StdProj(Pt3dr);
       static Pt3dr   PProj(Pt2dr);


        static void ProfChampsFromDist
             (
                 ElSTDNS list<Pt3dr>&  res,  // liste de triplets de prof de champs
                 Pt3dr p1,Pt3dr p2,Pt3dr p3, // points de projection
                 REAL d12, REAL d13, REAL d23
             );
};




// Toutes les projections R3->R2 consideree  dans ElProj32
// sont des projections avec rayon perspectif droits

class ElProj32
{
     public :

     // Methodes de bases a redefinir pour chaque type de projection

        virtual Pt2dr Proj(Pt3dr) const = 0;
        virtual Pt3dr DirRayon(Pt2dr) const = 0;
        virtual void  Diff(ElMatrix<REAL> &,Pt3dr) const = 0;  // differentielle


    virtual void Rayon(Pt2dr,Pt3dr &p0,Pt3dr & p1) const = 0;

     // Interfaces simplifiee
        ElMatrix<REAL> Diff(Pt3dr)const ;

    virtual ~ElProj32() {}
};

class ElProjIdentite : public ElProj32
{
     public :
        Pt2dr Proj(Pt3dr) const ;
        Pt3dr DirRayon(Pt2dr) const ;
        void  Diff(ElMatrix<REAL> &,Pt3dr) const ;  // differentielle
    void Rayon(Pt2dr,Pt3dr &p0,Pt3dr & p1) const ;

       static ElProjIdentite  TheOne;
};

/* Le neologisme AFocal designe les projection qui sont quasi-stenope, cad dont le
   point nodal varie en fonction  de l'angle de visee


*/

#define NbParamAF 2



template <class Type> class  ElProjStenopeGen
{
    public :
        ElProjStenopeGen(Type foc,Type cx,Type cy,const std::vector<Type> & ParamAF);

        Type & focale();
        Type   focale() const;
    Pt2d<Type> PP() const;
    void SetPP(const  Pt2d<Type> &) ;

        void Proj  (Type & x2,Type & y2,Type   x3,Type y3,Type z3) const;
        void DirRayon(Type & x3,Type & y3,Type & z3,Type x2,Type y2) const;

        void  Diff(ElMatrix<Type> &,Type,Type,Type)const ;

        const std::vector<Type>  & ParamAF() const;

        bool   UseAFocal() const;
    protected :
        Type DeltaCProjDirTer(Type x3,Type y3,Type z3) const;
        Type DeltaCProjTer(Type x3,Type y3,Type z3) const;
        Type DeltaCProjIm(Type x2,Type y2) const;

        Type               _focale;
        Type               _cx;
        Type               _cy;
        bool               mUseAFocal;
        std::vector<Type>  mParamAF;
};



typedef  std::vector<double> tParamAFocal;

class ElProjStenope : public ElProj32 ,
                      public ElProjStenopeGen<REAL>
{
     public :

        ElProjStenope(REAL Focale,Pt2dr centre, const tParamAFocal  & AFocalParam);

        Pt2dr Proj(Pt3dr) const;
        Pt3dr DirRayon(Pt2dr) const;
        virtual void Rayon(Pt2dr,Pt3dr &p0,Pt3dr & p1) const;
        void  Diff(ElMatrix<REAL> &,Pt3dr)const ;


        Pt2dr   centre() const;
        void    set_centre(Pt2dr) ;
    virtual ~ElProjStenope() {}
        Pt3dr   CentreProjIm(const Pt2dr &) const;
        Pt3dr   CentreProjTer(const Pt3dr &) const;
     private :
};

class ElDistortion22_Gen
{
     public :
       // Rustine car les camera fraser basic ne savent s'esporter qu'avec leur camera  ToXmlStruct
        ElCamera * CameraOwner();
        void SetCameraOwner(ElCamera*);

        virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;
        void SetName(const char * aName);
        virtual std::string Type() const;
        std::string Name() const;

        static cCalibDistortion  XmlDistNoVal();
        virtual  cPreCondGrid GetAsPreCond() const;
        static ElDistortion22_Gen * AllocPreC
                (const cPreCondGrid&);


        REAL D1(const ElDistortion22_Gen &,Pt2dr P0, Pt2dr P1,INT NbEch) const;
        REAL D2(const ElDistortion22_Gen &,Pt2dr P0, Pt2dr P1,INT NbEch) const;
        REAL DInfini(const ElDistortion22_Gen &,Pt2dr P0, Pt2dr P1,INT NbEch) const;

        virtual Pt2dr Direct(Pt2dr) const = 0 ;    //
        Pt2dr Inverse(Pt2dr) const ; //  Pour inverser :
                                    //    (1) On tente le OwnInverse
                                   //    (2) Sinon on tente le  mPolynInv
                                  //    (3) Sinon on applique le ComputeInvFromDirByDiff


        ElMatrix<REAL> Diff(Pt2dr) const;  // Juste interf a "Diff(ElMatrix<REAL> &,..)"
        virtual ~ElDistortion22_Gen();

        // PolynLeastSquareInverse (REAL aDom,INT aDegre);

        ElDistortionPolynomiale NewPolynLeastSquareInverse
                                (
                                    Box2dr aBox,
                                    INT  aDegre,
                                    INT  aNbPts = -1
                                );
        Polynome2dReal  NewPolynLeastSquareInverse_OneCoord
                        (
                              bool XCoord,
                              Box2dr aBox,
                              INT  aDegre,
                              INT  aNbPts = -1
                        );
    //  aSc * Direct (aP/aSc*)
        // Def renvoit un objet contenant un pointeur sur this
    virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const;
    virtual bool IsId() const; // Def false

    // Fonction "ad hoc" de dynamic cast, par defaut return 0, strict change pour PhgStd qui
    // ne se voit pas alors comme un cas particulier de DRad
    virtual ElDistRadiale_PolynImpair * DRADPol(bool strict = false);

        Box2dr ImageOfBox(Box2dr,INT aNbPtsDisc=8 );
        Box2dr ImageRecOfBox(Box2dr,INT aNbPtsDisc=8  );

       // Par defaut renvoie un objet contenant un pointeur sur this
       // et redirigeant Direct sur inverse et Lycee de Versailles
        virtual ElDistortion22_Gen * CalcInverse() const;
        void SetParamConvInvDiff(INT aNbIter,REAL aEps);

    void SaveAsGrid(const std::string&,const Pt2dr& aP0,const Pt2dr& aP1,const Pt2dr& aStep);

// Def erreur fatale
        virtual Pt2dr  DirectAndDer(Pt2dr aP,Pt2dr & aGradX,Pt2dr & aGradY) const;

 // Est-ce que les distorsion savent se transformer  en F-1 D F ou F est une translation ou
 // une homotetie
        virtual  bool  AcceptScaling() const;
        virtual  bool  AcceptTranslate() const;

//    Soit H (X) == PP + X * F   se transforme en H-1 D H
        void SetScalingTranslate(const double & F,const Pt2dr & aPP);

        double & ScN();
        const double & ScN() const;


       // ULTRA -DANGEREUX, rajoute a contre coeur, comme moyen d'inhiber la parie
       // Tgt-Argt des fishe-eye afin qu'il puisse calcule des pseudo inverse

        void    SetDist22Gen_UsePreConditionner(bool) const;  // Tous sauf const !!
        const bool &   Dist22Gen_UsePreConditionner() const;
    Pt2dr  ComputeInvFromDirByDiff ( Pt2dr aPt, Pt2dr InvEstim0, bool DiffReestim) const;

        void    SetDist22Gen_SupressPreCondInInverse(bool) const;  // Tous sauf const !!
        const bool &   Dist22Gen_SupressPreCondInInverse() const;

    protected :

         void ErrorInvert() const;
         ElDistortion22_Gen();
     void DiffByDiffFinies(ElMatrix<REAL> &,Pt2dr,Pt2dr Eps) const;
     void DiffByDiffFinies(ElMatrix<REAL> &,Pt2dr,REAL Eps) const;

    private :

        virtual  void V_SetScalingTranslate(const double &,const Pt2dr &);

        REAL DistanceObjet(INT tagDist,const ElDistortion22_Gen &,Pt2dr P0, Pt2dr P1,INT NbEch) const;


        ElDistortionPolynomiale * mPolynInv;

        virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  differentielle
        // Def err fatale



        // Defaut 0,0 pas forcement le meilleur choix mais
    // compatibilite anterieure

public :
        virtual Pt2dr GuessInv(const Pt2dr & aP) const ;
private :
        virtual bool OwnInverse(Pt2dr &) const ;    //  return false


protected :
        REAL mEpsInvDiff;
        INT  mNbIterMaxInvDiff;
private :
        double mScN;
        bool   mDist22Gen_UsePreConditionner;
        bool   mDist22Gen_SupressPreCondInInverse;
protected :
        const char * mName;
private :
        ElCamera * mCameraOwner;
};


class cCamAsMap : public cElMap2D
{
    public :
        virtual int Type() const ;
        cCamAsMap(CamStenope * aCam,bool Direct);
        virtual Pt2dr operator () (const Pt2dr & p) const ;
        virtual cElMap2D * Map2DInverse() const;
        virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0

    private :
           CamStenope *           mCam;
           bool                   mDirect;
};


class cXmlAffinR2ToR;

class cElComposHomographie
{
      public :
         REAL operator() (const Pt2dr & aP) const
         {
              return mX*aP.x + mY*aP.y + m1;
         }
         Fonc_Num operator() (Pt2d<Fonc_Num> ) const;


         cElComposHomographie(REAL aX,REAL aY,REAL a1);
         cElComposHomographie(const cXmlAffinR2ToR &);
         cXmlAffinR2ToR ToXml() const;

     cElComposHomographie MulXY(REAL ) const;
     cElComposHomographie MulCste(REAL ) const;

      void write(class  ELISE_fp & aFile) const;
          static cElComposHomographie read(ELISE_fp & aFile);
          friend class cElHomographie;


          REAL & CoeffX();
          REAL & CoeffY();
          REAL & Coeff1();

          REAL  CoeffX() const;
          REAL  CoeffY() const;
          REAL  Coeff1() const;

          void Show(const std::string & aMes);
          bool HasNan() const;

      private  :
          void SetCoHom(REAL *) const;
          void Divide (REAL);
          REAL mX;
          REAL mY;
          REAL m1;
};

class cXmlHomogr;

class cElHomographie  : public cElMap2D
{
     public :
          Pt2dr operator() (const Pt2dr & aP) const;
          virtual int Type() const ;
          virtual  cElMap2D * Map2DInverse() const;
          virtual cElMap2D * Duplicate() ;
          virtual cElMap2D * Identity() ;
          virtual cXml_Map2D    ToXmlGen() ; // Peuvent renvoyer 0
   
          virtual int   NbUnknown() const;
          virtual void  AddEq(Pt2dr & aCste,std::vector<double> & EqX,std::vector<double> & EqY,const Pt2dr & aP1,const Pt2dr & aP2 ) const;
          virtual void  InitFromParams(const std::vector<double> &aSol);
          virtual std::vector<double> Params() const;

          bool HasNan() const;

          Pt2dr Direct  (const Pt2dr & aP) const;
          Pt2d<Fonc_Num> Direct (Pt2d<Fonc_Num> ) const;

          void Show();
          // Renvoie H tel que H(P1) = P2
          // Size = 0 , identite
          // Size = 1 , translation
          // Size = 2 , similitude
          // Size = 3 , affinite
          // Size = 4 ou +, homographie reelle, ajuste par moindre L2  ou  L1

          cElHomographie(const ElPackHomologue &,bool aL2);
          cElHomographie(const cXmlHomogr &);
          cXmlHomogr ToXml() const;

          static cElHomographie RansacInitH(const ElPackHomologue & aPack,int aNbRansac,int aNbMaxPts);

          static cElHomographie Id();
          static cElHomographie Homotie(Pt2dr aP,REAL aSc);  // -> tr + aSc * P
          static cElHomographie FromMatrix(const ElMatrix<REAL> &);

          void ToMatrix(ElMatrix<REAL> &) const;

          cElHomographie
          (
               const cElComposHomographie &,
               const cElComposHomographie &,
               const cElComposHomographie &
          );


          cElHomographie Inverse() const;
          cElHomographie operator * (const cElHomographie &) const;
        //     P ->  aChSacle * Pol(P/aChSacle)
          cElHomographie MapingChScale(REAL aChSacle) const;
      void write(class  ELISE_fp & aFile) const;
          static cElHomographie read(ELISE_fp & aFile);

          cElComposHomographie & HX();
          cElComposHomographie & HY();
          cElComposHomographie & HZ();

          const cElComposHomographie & HX() const;
          const cElComposHomographie & HY() const;
          const cElComposHomographie & HZ() const;

          // Renvoie sa representation matricielle en coordonnees homogenes
          ElMatrix<REAL>  MatCoordHom() const;
          static cElHomographie  RobustInit(double & anEcart,double * aQuality,const ElPackHomologue & aPack,bool & Ok ,int aNbTestEstim, double aPerc,int aNbMaxPts);

          static cElHomographie SomPondHom(const std::vector<cElHomographie> & aVH,const std::vector<double> & aVP);


     private :
          cElComposHomographie mHX;
          cElComposHomographie mHY;
          cElComposHomographie mHZ;

          void AddForInverse(ElPackHomologue & aPack,Pt2dr aP) const;
          void Normalize();
};

class cDistHomographie : public  ElDistortion22_Gen
{
      public :
        cDistHomographie(const ElPackHomologue &,bool aL2);
        cDistHomographie(const cElHomographie &);

        virtual bool OwnInverse(Pt2dr &) const ;    //
        virtual Pt2dr Direct(Pt2dr) const  ;    //
        cDistHomographie MapingChScale(REAL aChSacle) const;
    const cElHomographie & Hom() const;
      private :

    virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const; // Def erreur fatale
        void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  Erreur Fatale
        cElHomographie mHDir;
        cElHomographie mHInv;
};





class ElDistortion22_Triviale : public ElDistortion22_Gen
{
     public :
        void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  // ** differentielle
        Pt2dr Direct(Pt2dr) const  ;     //  **
        static ElDistortion22_Triviale  TheOne;
    virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const; // Def erreur fatale
    virtual bool IsId() const;
        virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;

     private :
        virtual bool OwnInverse(Pt2dr &) const ;    //  return false
};




/*
 *    ** : Methodes a redefinir imperativement si distortion non triviale.
 *
 */


class ElDistRadiale : public ElDistortion22_Gen
{
      public :


        Pt2dr  & Centre();
        const Pt2dr  & Centre() const;
        virtual bool OwnInverse(Pt2dr &) const ;    //  return false
        virtual Pt2dr Direct(Pt2dr) const ;

        virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const;  // differentielle

        // rho -> K0 *rho * (1 + DistDirecte(rho))
        virtual REAL K0() const; // def : return 1
        virtual REAL DistDirecte(REAL R) const = 0;
        virtual REAL DistDirecteR2(REAL R) const = 0;

        // doit renvoyer la derivee de DistDirecte, divisee par rho
        virtual REAL  DerSurRho(REAL R) const = 0; // en delta / a 1

        virtual REAL DistInverse(REAL R)  const;
                     // Par defaut les distortion sont
                     // supposees faibles et la fontion inverse est
                     // - la fonction directe

      protected  :

        ElDistRadiale(Pt2dr Centre);

      private  :
        Pt2dr _centre;
};

class ElDistRadiale_PolynImpair  : public ElDistRadiale // polynome en r de degre impair
{
     public :
      // Pour eviter les comportements sinuguliers
      // si R > RMax on remplace par la differentielle en RMax


        ElDistRadiale_PolynImpair(REAL RMax,Pt2dr centre);
        void ActuRMaxFromDist(Pt2di aSz);
        void ActuRMax();
    void SetRMax(REAL aV);
        virtual REAL DistDirecte(REAL) const;
        REAL DistDirecteR2NoSeuil(REAL R) const ;
        virtual REAL DistDirecteR2(REAL) const;
        virtual REAL DerSurRho(REAL) const;

        void PushCoeff(REAL); // Premiere fois fixe r3 , etc ....
        void PushCoeff(const std::vector<REAL> &); // Premiere fois fixe r3 , etc ....
    REAL & Coeff(INT k);
    REAL  Coeff(INT k) const;
    INT NbCoeff() const;
    INT NbCoeffNN() const;  // Elimine les eventuelles coefficient nul rajoutes
        void VerifCoeff(INT aK) const;
        REAL   CoeffGen(INT aK) const;


        ElDistRadiale_PolynImpair DistRadialeInverse(REAL RhoApp,INT DeltaDeg = 1);


        // aPt -> aChSacle * Direct (aPt / aChSacle)
        ElDistRadiale_PolynImpair MapingChScale(REAL aChSacle) const;

          static ElDistRadiale_PolynImpair DistId(REAL aRMax,Pt2dr aCentre,INT aDeg);

          static ElDistRadiale_PolynImpair read(ELISE_fp & aFile);
          static ElDistRadiale_PolynImpair read(const std::string &);
          void write(ELISE_fp & aFile);

      // DEBUG PURPOSE,
          REAL RMax() const;
          REAL ValRMax() const;
          REAL DiffRMax() const;
      virtual ElDistRadiale_PolynImpair * DRADPol(bool strict = false);

      ElPolynome<REAL> PolynOfR();
      // Rayon max a l'interieur duquel la fonction de
      // distortion est bijective croissante
      REAL RMaxCroissant(REAL BorneInit);

          virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;
          cCalibrationInterneRadiale ToXmlDradStruct() const;

     protected :
        bool  AcceptScaling() const;
        bool  AcceptTranslate() const;
        void V_SetScalingTranslate(const double &,const Pt2dr &);


     private :
          std::vector<REAL> mCoeffs;  // mCoeffs[0] en r3,  mCoeffs[1] en r5 , etc ....

          REAL              mRMax;
          REAL              mRMaxP2N;
          REAL              mValRMax;
          REAL              mDiffRMax;
};

class ElDistRadiale_Pol357  : public ElDistRadiale_PolynImpair // polynome en r de degre 3,5,7
{
      public :
         ElDistRadiale_Pol357(REAL aRMax,Pt2dr centre,REAL  c3,REAL c5,REAL c7);

      private  :
};


// Implemante des deformations du type "D-1 o H o D" avec
// D disrortion radiale polynomiale et H homographie

class cDistHomographieRadiale : public ElDistortion22_Gen
{
      public :
        cDistHomographieRadiale
        (
        const cElHomographie & anHom,
        const ElDistRadiale_PolynImpair & aDRad,
        REAL aRayInv,
        INT  aDeltaDegraInv
    );
        virtual bool OwnInverse(Pt2dr &) const ;    //
        virtual Pt2dr Direct(Pt2dr) const  ;    //
        // aPt -> aChSacle * Direct (aPt / aChSacle)
        cDistHomographieRadiale MapingChScale(REAL aChSacle) const;
      private:
    virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const; // Def erreur fatale
        void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  Erreur Fatale
    cElHomographie            mHom;
    cElHomographie            mHomInv;
    ElDistRadiale_PolynImpair mDist;
    ElDistRadiale_PolynImpair mDistInv;
    REAL                      mRay;
    INT                       mDeg;
};



// Classe de distortion par polynome de degre 2, Pas un interet fou
// (elle sont un peu + rapide que si definies par polynomes generiques)
// mais developpee pour tester rapidement certaine fonctionnalites
// generiques
//

class PolyDegre2XY
{
    public :
            PolyDegre2XY (REAL a,REAL aX,REAL aY,REAL aXX,REAL aXY,REAL aYY);

            REAL Val(Pt2dr aPt) const;
            Pt2dr Grad(Pt2dr aPt) const;

            REAL & Coeff() {return m;}
            REAL & CoeffX() {return mX;}
            REAL & CoeffY() {return mY;}
    private :
            REAL m;
            REAL mX;
            REAL mY;
            REAL mXX;
            REAL mXY;
            REAL mYY;

};


class ElDistPolyDegre2 : public ElDistortion22_Gen
{
     public :

        virtual Pt2dr Direct(Pt2dr) const ;  // **
        ElDistPolyDegre2
        (
            const PolyDegre2XY & aPolX,
            const PolyDegre2XY & aPolY,
            REAL EpsilonInv
        );

               // par defaut appel au fonctions "Quick" (ou Quasi)

        virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const;  // ** differentielle

    private :

        PolyDegre2XY mPolX;
        PolyDegre2XY mPolY;
        //REAL         mEpsilon;
};


class ElDistortionPolynomiale : public ElDistortion22_Gen
{
    // Pour les inverses et autres, on fait l'hypothese
    // que les coeff, hors degres <=1, ont des valeur
    // tres faible

     public :
           static ElDistortionPolynomiale DistId(int aDegre,double anAmpl);
           ElDistortionPolynomiale
           (
               const Polynome2dReal & aDistX,
               const Polynome2dReal & aDistY,
               REAL                   anEpsilonInv = 1e-7
           );
           virtual Pt2dr Direct(Pt2dr) const ;  // **

            const Polynome2dReal & DistX() const ;
            const Polynome2dReal & DistY() const ;
            Fonc_Num FNum() const ;

           ElDistortionPolynomiale (REAL anAmpl,REAL anEpsilonInv = 1e-7) ;

               // par defaut appel au fonctions "Quick" (ou Quasi)

        virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const;  // ** differentielle

        // aPt -> aChSacle * Direct (aPt / aChSacle)
        ElDistortionPolynomiale MapingChScale(REAL aChSacle) const;
    virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const; // Def erreur fatale

     private :
             Polynome2dReal mDistX;
             Polynome2dReal mDistY;
             REAL           mEpsilon;
};

class EpipolaireCoordinate : public ElDistortion22_Gen
{
    public :
         void SaveOrientEpip
              (
                  const std::string &                anOri,
                  cInterfChantierNameManipulateur *  anICNM,
                  const std::string &                aNameIm,
                  const std::string &                aNameOther
               ) const;


        // Lorsque aParal ballaye R, on obtient
        // la courbe epipolaire passant par aP
        Pt2dr  TransOnLineEpip
               (
                Pt2dr aP,
                REAL aParal
               );



                virtual Pt2dr Direct(Pt2dr) const ;
        virtual bool IsEpipId() const;
                // Inverse est heritee  et fait appel a OwnInverse


                Pt2dr DirEpip(Pt2dr,REAL anEpsilon); // Calcul par difference finie !


                Pt2dr P0() const;
                Pt2dr DirX() const;
                Pt2dr TrFin() const;

                virtual const PolynomialEpipolaireCoordinate * CastToPol() const ; // Down cast, Def = Erreur

        //     P ->  aChSacle * Pol(P/aChSacle)
             virtual EpipolaireCoordinate *
             MapingChScale(REAL aChSacle) const = 0;
	     // Def => fatal error
	     virtual void  XFitHom(const ElPackHomologue &,bool aL2,EpipolaireCoordinate *);
	     virtual bool  HasXFitHom() const;
	     virtual std::vector<double>  ParamFitHom() const;


             // Fait heriter les parametre globaux aP0, aDirX, aTrFin
               void HeriteChScale(EpipolaireCoordinate &,REAL aChSacle);

               Box2dr ImageOfBox(Box2dr );
               void   AddTrFinale(Pt2dr);
               void   SetTrFinale(Pt2dr);


           void   SetGridCorrec
              (
                  Fonc_Num DeltaY, // rab de Y epip, exprime en coord image
                  Fonc_Num Pond,   // Binarisee en 0/1 , exprime en coord image
              REAL  aStepGr ,
              Box2dr aBoxIm,
              REAL   aRatioMin = 0.2 // Ratio pour remplir la grille
              );
           virtual ~EpipolaireCoordinate();
        protected :
                EpipolaireCoordinate
                (
                     Pt2dr aP0,
                     Pt2dr aDirX,
                     Pt2dr aTrFin
        );
            RImGrid *  mGridCor; // Pour une eventuelle correction finale avec grille
    private :


        // Pour les "vrai" systemes epipolaire, la transformation
        // epiplaire ne change pas le X, cependant afin de pouvoir
        // utilise dans les correlateur des mappigng quelconques
        // on redefinit ToYEpipol en ToCoordEpipole

         virtual Pt2dr ToCoordEpipol(Pt2dr aPInit) const = 0;
         virtual Pt2dr ToCoordInit(Pt2dr aPEpi) const = 0;


              virtual bool OwnInverse(Pt2dr &) const ;
              virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  => Error Fatale, for now


              Pt2dr      mP0;
              Pt2dr      mDirX;
              Pt2dr      mTrFin;
};

class EpipolaireCoordinateNoDist : public EpipolaireCoordinate
{
    public :
                EpipolaireCoordinateNoDist
                (
                     Pt2dr aP0,
                     Pt2dr aDirX
        );
    private :
         virtual Pt2dr ToCoordEpipol(Pt2dr aPInit) const ;
         virtual Pt2dr ToCoordInit(Pt2dr aPEpi) const ;
             virtual EpipolaireCoordinate *
             MapingChScale(REAL aChSacle) const;
};

class cMappingEpipCoord : public EpipolaireCoordinate
{
      public :
         cMappingEpipCoord(ElDistortion22_Gen *,bool toDel);
         ~cMappingEpipCoord();
      private :
             virtual bool IsEpipId() const;
             EpipolaireCoordinate * MapingChScale(REAL aChSacle) const;
         Pt2dr ToCoordEpipol(Pt2dr aPInit) const ;
         Pt2dr ToCoordInit(Pt2dr aPEpi) const ;

         ElDistortion22_Gen * mDist;
         bool                 mToDel;
};


class PolynomialEpipolaireCoordinate : public EpipolaireCoordinate
{
      public :


              PolynomialEpipolaireCoordinate
              (
                 Pt2dr aP0,
                 Pt2dr aDirX,
                 const Polynome2dReal & aPolY,
                 const Polynome2dReal * aPolInvY,
                 REAL                   anAmpl,
                 INT                    DeltaDegreInv = 2,
                 Pt2dr                  aTrFin = Pt2dr(0,0)
              );


              Polynome2dReal  PolToYEpip();
              Polynome2dReal  PolToYInit();

              virtual  const PolynomialEpipolaireCoordinate * CastToPol() const override;
          void write(class  ELISE_fp & aFile) const;
              static PolynomialEpipolaireCoordinate read(ELISE_fp & aFile);
        //     P ->  aChSacle * Pol(P/aChSacle)
              EpipolaireCoordinate * MapingChScale(REAL aChSacle) const override;
              PolynomialEpipolaireCoordinate * PolMapingChScale(REAL aChSacle) const;

	      // Create new Pol, fixing mC0.., so to mimimize the global paralax
	      void  XFitHom(const ElPackHomologue &,bool aL2,EpipolaireCoordinate *) override;
	      bool  HasXFitHom() const override;
	      std::vector<double>  ParamFitHom() const override;

      private :

          INT DeltaDegre() const;
          REAL AmplInv() const;

          Polynome2dReal  mPolToYEpip;
          Polynome2dReal  mPolToYInit;

          Pt2dr ToCoordEpipol(Pt2dr aPInit) const override;
          Pt2dr ToCoordInit(Pt2dr aPEpi) const override;

	  //  X' = (mNum0 + mNumx X + mNumy Y) / (1 + mDenx X + mDeny Y)
	  //  X  = 
	  double mNum0;
	  double mNumx;
	  double mNumy;
	  double mDenx;
	  double mDeny;
	  bool   mCorCalc;
};


class CpleEpipolaireCoord
{
    public :
         void SaveOrientCpleEpip
              (
                  const std::string &                anOri,
                  cInterfChantierNameManipulateur *  anICNM,
                  const std::string &                aName1,
                  const std::string &                aName2
               ) const;


            static CpleEpipolaireCoord * EpipolaireNoDist
                   (Pt2dr aPHom1,Pt2dr aPHom2,Pt2dr aDir1,Pt2dr aDir2);


            static CpleEpipolaireCoord * PolynomialFromHomologue
                                        (
                                                bool  UseL1,  // Si aSolApprox nulle alors utilise-t-on L1 ?
                                                const ElPackHomologue &,
                                                INT   aDegre,
                                                Pt2dr aDir1,
                                                Pt2dr aDir2,
                                                int   aDeltaDeg=2
                                        );
            static CpleEpipolaireCoord * PolynomialFromHomologue
                                        (
                                                const ElPackHomologue & lHL1,
                                                INT   aDegreL1,
                                                const ElPackHomologue & lHL2,
                                                INT   aDegreL2,
                                                Pt2dr aDir1,
                                                Pt2dr aDir2,
                                                int   aDeltaDeg=2
                                        );

            static CpleEpipolaireCoord * PolynomialFromHomologue
                                        (
                                                bool  UseL1,  // Si aSolApprox nulle alors utilise-t-on L1 ?
                                                CpleEpipolaireCoord  *  aSolApprox, // Solution pour calcul de residu
                                                REAL  aResiduMin,
                                                const ElPackHomologue &,
                                                INT   aDegre,
                                                Pt2dr aDir1,
                                                Pt2dr aDir2,
                                                int   aDeltaDeg=2
                                        );


             static CpleEpipolaireCoord * MappingEpipolaire(ElDistortion22_Gen *,bool ToDel);

            // Il ne s'agit pas d'un vrai systeme epipolaire, l'hoomographie ets
            // utilisee comme un mapping qcq, fait  appel a MappingEpipolaire
             static CpleEpipolaireCoord * MappEpiFromHomographie(cElHomographie);
             static CpleEpipolaireCoord * MappEpiFromHomographieAndDist
                                  (
                                               const cElHomographie &,
                                               const ElDistRadiale_PolynImpair &,
                           REAL aRayInv,
                           INT aDeltaDegreInv
                      );

             static CpleEpipolaireCoord * OriEpipolaire
                                          (
                                             const std::string & aName1, Pt2dr aP1,
                                             const std::string & aName2, Pt2dr aP2,
                                             REAL aZoom
                                          );

             static CpleEpipolaireCoord * CamEpipolaire
                                          (
                                             CamStenope  & aCam1, Pt2dr aP1,
                                             CamStenope  & aCam2, Pt2dr aP2,
                                             REAL aZoom
                                          );

            ~CpleEpipolaireCoord();
            const EpipolaireCoordinate & EPI1() const;
            const EpipolaireCoordinate & EPI2() const;
            EpipolaireCoordinate & EPI1() ;
            EpipolaireCoordinate & EPI2() ;

            Pt2dr Hom12(Pt2dr,Pt2dr aParalaxe); // x=> paralaxe, y variation de colonne
            Pt2dr Hom12(Pt2dr,REAL aParalaxe);
            Pt2dr Hom21(Pt2dr,REAL aParalaxe);
            Pt2dr Hom21(Pt2dr,Pt2dr aParalaxe); // x=> paralaxe, y variation de colonne

        Pt2dr Homol(Pt2dr,Pt2dr aParalaxe,bool Sens12);

            Pt2d<Fonc_Num>  Hom12(Pt2d<Fonc_Num> fXY,Pt2d<Fonc_Num> fParalaxe);


        void write(class  ELISE_fp & aFile) const;
            static CpleEpipolaireCoord * read(ELISE_fp & aFile);
        //     P ->  aChSacle * Pol(P/aChSacle)
            CpleEpipolaireCoord * MapingChScale(REAL aChSacle) const;

            void SelfSwap(); // Intervertit les  2
            CpleEpipolaireCoord * Swap();  // renvoie une nouvelle avec Intervertion
            void AdjustTr2Boxes(Box2dr aBox1,Box2dr aBox2);

            bool IsMappingEpi1() const;

        private:

            EpipolaireCoordinate * mEPI1;
            EpipolaireCoordinate * mEPI2;
            REAL                   mFact;

            CpleEpipolaireCoord
            (
                 EpipolaireCoordinate * mEPI1,
                 EpipolaireCoordinate * mEPI2
            );

            CpleEpipolaireCoord(const CpleEpipolaireCoord &); // Unimplemented
            friend class PourFairePlaisirAGccQuiMemerde;


};


typedef enum
{
   eProjectionStenope,
   eProjectionOrtho
} eTypeProj;


class cCorrRefracAPost
{
     public :
         Pt2dr CorrM2C(const Pt2dr &) const;
         Pt2dr CorrC2M(const Pt2dr &) const;
         cCorrRefracAPost(const cCorrectionRefractionAPosteriori &);

         // Le coefficient est le ratio du coeef de refrac du milieu d'entree sur le milieu de sortie
         Pt3dr CorrectRefrac(const Pt3dr &,double aCoef) const;

          const cCorrectionRefractionAPosteriori & ToXML() const;
     private :

         cCorrectionRefractionAPosteriori * mXML;
         ElCamera *  mCamEstim;
         double      mCoeffRefrac;
         bool       mIntegDist;
};

// La plus basique des classes, normalement tout doit pouvoir etre redefini 
// a partir de ca

class cArgOptionalPIsVisibleInImage
{
    public :
       cArgOptionalPIsVisibleInImage();

       bool   mOkBehind;
       std::string  mWhy;
};

class cBasicGeomCap3D
{
    public :
      typedef ElAffin2D tOrIntIma ;



      virtual ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const =0;
      virtual Pt2dr    Ter2Capteur   (const Pt3dr & aP) const =0;
      virtual Pt2di    SzBasicCapt3D() const = 0;
      virtual Pt2dr    SzPixel() const ; // Defaut SzBasicCapt3D
      virtual double ResolSolOfPt(const Pt3dr &) const = 0;
      virtual bool  CaptHasData(const Pt2dr &) const = 0;
      //  Def return true, mean that the geometry is ok independently of the image data
      virtual bool  CaptHasDataGeom(const Pt2dr &) const ;
      virtual bool     PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage * =0) const =0;

      // is true, facilitate visibility, dont need heurist
      virtual bool     DistBijective   () const;

      // Can be very approximate, using average depth or Z
      virtual Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const =0;

  // Optical center 
      virtual bool     HasOpticalCenterOfPixel() const; // 1 - They are not alway defined
// When they are, they may vary, as with push-broom, Def fatal erreur (=> Ortho cam)
      virtual Pt3dr    OpticalCenterOfPixel(const Pt2dr & aP) const ; 
   // Coincide avec le centre optique pour les camera stenope et RPC, est la position
   // du centre origine pour les camera ortho (utilise pour la geom faisceau)
      virtual Pt3dr    OrigineProf() const ;  // Par defau OpticalCenterOfPixel(Milieu)

      virtual Pt3dr    ImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
      virtual Pt3dr ImEtZ2Terrain(const Pt2dr & aP,double aZ) const;
// Compute the differential, defaut value compute it by finite difference,at step ResolSolOfPt, 
// to accelerate it is note centered en reuse the value PIm
      virtual void Diff(Pt2dr & aDx,Pt2dr & aDy,Pt2dr & aDz,const Pt2dr & aPIm,const Pt3dr & aTer);

      static cBasicGeomCap3D * StdGetFromFile(const std::string &,int & aType, 
                                              const cSystemeCoord * aChSys=0); // !!! aType in fact is eTypeImporGenBundle 

      // Down cast , dirty but usefull ;-)
      virtual CamStenope * DownCastCS() ;

      virtual Pt3dr DirRayonR3(const Pt2dr & aPIm) const;
      virtual double ProfondeurDeChamps(const Pt3dr & aP) const;
      virtual Pt3dr DirVisee() const;

       double  EpipolarEcart(const Pt2dr & aP1,const cBasicGeomCap3D & aCam2,const Pt2dr & aP2,Pt2dr * SauvDir=0) const;


       virtual double ResolutionAngulaire() const;  // OO
       /// This function calls existing virtual function; in many case it will be redundant with
       // them, the goals is to be sure that the Center  and the PTer are exactly
       //  on the line given by Capteur2RayTer; which theoritically may be not the case
       // especialy when centers are computed from multiples intersections
       virtual void  GetCenterAndPTerOnBundle(Pt3dr & aC,Pt3dr & aPTer,const Pt2dr & aPIm) const;

       // Return an "ordre de grandeur'virtual" of the interval of prof in prop; default return 1/600 adapted to
       // pleiade-spot satellites, redefine in Stenope to 0.2;  used for epipolar computation
       virtual double GetVeryRoughInterProf() const;


       virtual  double GetAltiSol() const ;
       virtual  Pt2dr GetAltiSolMinMax() const ;
       virtual   bool AltisSolIsDef() const ;
       virtual   bool AltisSolMinMaxIsDef() const;
       // RPC have some limit in validity, which require some special careness, but don't want to
       // modify existing code that work well, so we need to know if the underline camera is RPC
       virtual   bool IsRPC() const;


       // Save using standard MicMac naming ; !! Not supported for now by Stenope camera; Def :  Fatal Error
       virtual std::string Save2XmlStdMMName(  cInterfChantierNameManipulateur * anICNM,
                                        const std::string & aOriOut,
                                        const std::string & aNameImClip,
                                        const ElAffin2D & anOrIntInit2Cur
                    ) const;
       std::string Save2XmlStdMMName(cInterfChantierNameManipulateur*,const std::string &,const std::string&,const Pt2dr & aP0Clip) const ;
       std::string Save2XmlStdMMName(cInterfChantierNameManipulateur*,const std::string &,const std::string&) const ;

       Pt2dr Mil() const;
       double GlobResol() const;
       Pt3dr  PMoyOfCenter() const;
       virtual bool  HasRoughCapteur2Terrain() const ;
       // virtual   Pt2dr OrGlbImaM2C(const Pt2dr &) const;

       virtual Pt2dr ImRef2Capteur   (const Pt2dr & aP) const;
       virtual double ResolImRefFromCapteur() const;
       virtual bool  HasPreciseCapteur2Terrain() const ;
       virtual Pt3dr PreciseCapteur2Terrain   (const Pt2dr & aP) const ;

  
       void SetScanImaM2C(const tOrIntIma  &);
       void SetScanImaC2M(const tOrIntIma &);
       void SetIntrImaM2C(const tOrIntIma  &);
       void SetIntrImaC2M(const tOrIntIma &);
       Pt2dr OrGlbImaM2C(const Pt2dr &) const;
       Pt2dr OrGlbImaC2M(const Pt2dr &) const;
       Pt2dr OrScanImaM2C(const Pt2dr &) const;
       Pt2dr OrIntrImaC2M(const Pt2dr &) const;
       void ReCalcGlbOrInt();
       virtual ~cBasicGeomCap3D();

        cBasicGeomCap3D();

    protected :

         // Ces deux similitudes permettent d'implanter le crop-scale-rotate
         // peut-etre a remplacer un jour par une ElAffin2D; sans changer
         // l'interface
         //
 
         // A PRIORI INUTILISE DANS cBasicGeomCap3D
         tOrIntIma                      mScanOrImaC2M;
         tOrIntIma                      mIntrOrImaC2M;
         tOrIntIma                      mGlobOrImaC2M;

         tOrIntIma                      mScanOrImaM2C;
         tOrIntIma                      mIntrOrImaM2C;
         tOrIntIma                      mGlobOrImaM2C;
         double               mScaleAfnt;  // Echelle de l'affinite !!
};


class cAffinitePlane;
cBasicGeomCap3D * DeformCameraAffine
                  (
                        const cAffinitePlane & aXmlApInit2Cur,
                        cBasicGeomCap3D * aCam0,
                        const std::string & aName,
                        const std::string &aNameIma
                   );




//  Classe qui permet de manipuler de manière via une interface uniforme une image,
// ou un nuage de point

class cCapture3D : public cBasicGeomCap3D
{
   public :
      // virtual ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const =0;


      virtual double ResolSolGlob() const = 0;

};



class ElCamera : public cCapture3D
{
     public :

         const bool &   IsScanned() const;
         void  SetScanned(bool mIsSC);

         Pt3dr DirRayonR3(const Pt2dr & aPIm) const;
         Pt2di    SzBasicCapt3D() const; 
         double GetVeryRoughInterProf() const;
         bool  CaptHasData(const Pt2dr &) const ;
         Pt2dr    Ter2Capteur   (const Pt3dr & aP) const;
         bool     PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage * =0) const ;
         ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const;
         double ResolImRefFromCapteur() const ;
         bool  HasRoughCapteur2Terrain() const ;
         Pt2dr ImRef2Capteur   (const Pt2dr & aP) const ;
         bool  HasPreciseCapteur2Terrain() const ;
         Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const ;
         Pt3dr PreciseCapteur2Terrain   (const Pt2dr & aP) const ;
         double ResolSolOfPt(const Pt3dr &) const;
         double ResolSolGlob() const;

         double  ScaleAfnt() const;

         Pt3dr Vitesse() const;
         void  SetVitesse(const Pt3dr &);
         bool  VitesseIsInit() const;
         Pt3dr IncCentre() const;
         void  SetIncCentre(const Pt3dr &);

         void TestCam(const std::string & aMes);
         const double & GetTime() const;
         void   SetTime(const double &);
     // ProfIsZ si true, ZProf est l'altisol habituel, sinon c'est une profondeur de champ
         cOrientationConique ExportCalibGlob(Pt2di aSzIm,double AltiSol,double Prof,int AddVerif,bool ModMatr,const char * aNameAux,const Pt3di * aNbVeridDet=0) const;

         cCalibrationInternConique ExportCalibInterne2XmlStruct(Pt2di aSzIm) const;
         // cCalibrationInternConique ExportCalibInterne2XmlStruct(Pt2di aSzIm) const;
         cVerifOrient MakeVerif( int aNbVerif,double aProf,const char *,const Pt3di  * aNbDeterm=0) const;
         cOrientationConique  StdExportCalibGlob(bool Matr) const;
         cOrientationConique  StdExportCalibGlob() const;
         std::string StdExport2File(cInterfChantierNameManipulateur *,const std::string & aDirOri,const std::string & aNameIm,const std::string & aFileInterne = "");  // Test -> Ori-R

      virtual  Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aZ) const = 0;
      virtual  Pt3dr NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const = 0;
          void SetAltiSol(double );
          void SetProfondeur(double );

           // void ChangeSys(const cSysCoord & a1Source,const cSysCoord & a2Cible,const Pt3dr & aP);
         static void ChangeSys
                     (
                            const std::vector<ElCamera *>& ,
                            const cTransfo3D & aTr3D,
                            bool ForceRot,
                            bool AtGroundLevel
                     );

          // Pour compatibilite stricte avec ce qui etait fait avant
         // dans cDistStdFromCam::Diff
          virtual double SzDiffFinie() const = 0;
          Pt3dr DirVisee() const;
          double ProfondeurDeChamps(const Pt3dr & aP) const;

          virtual double ResolutionSol() const = 0;
          virtual double ResolutionSol(const Pt3dr &) const = 0;
          double GetAltiSol() const;
          bool AltisSolIsDef() const;
          void UndefAltisSol() ;


          double GetProfondeur() const;
          virtual double GetRoughProfondeur() const; // Tente Prof puis Alti
          bool   ProfIsDef() const;
          eTypeProj GetTypeProj() const;
          CamStenope * CS();
          const CamStenope * CS() const;


          virtual cCamStenopeBilin * CSBil_SVP();
          cCamStenopeBilin * CSBil();

         double  RatioInterSol(const ElCamera &) const;

         double   EcartAngulaire(Pt2dr aPF2A, const ElCamera & CamB, Pt2dr aPF2B) const;
         double   SomEcartAngulaire(const ElPackHomologue &, const ElCamera & CamB, double & aSomP) const;
         double   EcartAngulaire(const Appar23 &) const;
         double   SomEcartAngulaire(const std::vector<Appar23> & aVApp) const;
    // L'identifiant ou le nom d'un camera, est qq chose d'optionnel , rajoute a posteriori
    // pour la debugage-tracabilite

         const std::string &  IdentCam() const;
         void SetIdentCam(const std::string & aName);

         const std::string &  NameIm() const;
         void SetNameIm(const std::string & aName);

     // ========================

          ElRotation3D & Orient();
          const ElRotation3D & Orient() const;

          void SetOrientation(const ElRotation3D &);
          void AddToCenterOptical(const Pt3dr & aOffsetC);
          void MultiToRotation(const ElMatrix<double> & aOffsetR);

          Pt3dr  PseudoInter(Pt2dr aPF2A,const ElCamera & CamB,Pt2dr aPF2B,double * aD=0) const;
          // Idem PseudoInter mais la precision est celle de reprojection
          Pt3dr  PseudoInterPixPrec(Pt2dr aPF2A,const ElCamera & CamB,Pt2dr aPF2B,double & aD) const;
          Pt3dr  CdgPseudoInter(const ElPackHomologue &,const ElCamera & CamB,double & aD) const;

          REAL EcartProj(Pt2dr aPF2A,const ElCamera & CamB,Pt2dr aPF2B) const;

          REAL EcartProj(Pt2dr aPF2A,Pt3dr aPR3,Pt3dr aDirR3) const;


          double  ScaleCamNorm() const;
          Pt2dr   TrCamNorm() const;

        //   R3 : "reel" coordonnee initiale
        //   L3 : "Locale", apres rotation
        //   C2 :  camera, avant distortion
        //   F2 : finale apres Distortion
        //
        //       Orientation      Projection      Distortion
        //   R3 -------------> L3------------>C2------------->F2

          Pt2dr R3toF2(Pt3dr) const;
          Pt2dr R3toC2(Pt3dr) const;

          virtual Pt3dr R3toL3(Pt3dr) const;
          virtual Pt3dr L3toR3(Pt3dr) const;

          // Direction en terrain de l'axe camera
          Pt3dr  DirK() const; // OO

          // A la orilib
          Pt3dr F2AndZtoR3(const Pt2dr & aPIm,double aZ) const;

      Pt2dr F2toC2(Pt2dr) const;
      void F2toRayonL3(Pt2dr,Pt3dr &aP0,Pt3dr & aP1) const;
      void F2toRayonR3(Pt2dr,Pt3dr &aP0,Pt3dr & aP1) const;

          Pt3dr PtFromPlanAndIm(const cElPlan3D  & aPlan,const Pt2dr& aP) const;


          ElSeg3D F2toRayonR3(Pt2dr) const;
      Pt3dr   F2toDirRayonL3(Pt2dr) const;
      Pt3dr   F2toDirRayonR3(Pt2dr) const;
      Pt3dr   C2toDirRayonR3(Pt2dr) const;
      Pt2dr   F2toPtDirRayonL3(Pt2dr) const;  // Meme chose, enleve la z a 1
      Pt2dr   L3toF2(Pt3dr) const;
      Pt2dr   PtDirRayonL3toF2(Pt2dr) const;

      Pt2dr Pixel2Radian(const Pt2dr & aP) const;
      Pt2dr Radian2Pixel(const Pt2dr & aP) const;

      Pt3dr   C2toDirRayonL3(Pt2dr) const;
      Pt2dr   L3toC2(Pt3dr) const;

          // Transforme en points photogrammetriques
      ElPackHomologue F2toPtDirRayonL3(const ElPackHomologue &,ElCamera * aCam2=0);  // Def = this
      ElCplePtsHomologues F2toPtDirRayonL3(const ElCplePtsHomologues &,ElCamera * aCam2=0); // Def = this

         Appar23   F2toPtDirRayonL3(const Appar23 &);
     std::list<Appar23>  F2toPtDirRayonL3(const std::list<Appar23>&);

          // Renvoie la somme des ecarts entre la projection des points
          // 3D et les points 2D

          bool Devant(const Pt3dr &) const;
          bool TousDevant(const std::list<Pt3dr> &) const;
          REAL EcProj(const ElSTDNS list<Pt3dr> & PR3 ,
                      const ElSTDNS list<Pt2dr> & PF2) const;

          REAL EcProj ( const ElSTDNS list<Appar23> & P23);

          // Differentielle de l'application globale
                // par rapport a un point
          void  DiffR3F2(ElMatrix<REAL> &,Pt3dr) const;
          ElMatrix<REAL>  DiffR3F2(Pt3dr) const;
                // par rapport aux params
          void  DiffR3F2Param(ElMatrix<REAL> &,Pt3dr) const;
          ElMatrix<REAL>  DiffR3F2Param(Pt3dr) const;

      // void SetDistInverse();
      // void SetDistDirecte();

          bool DistIsDirecte() const;
          bool DistIsC2M() const;
      Pt2dr DistDirecte(Pt2dr aP) const;
      Pt2dr DistInverse(Pt2dr aP) const;
      Pt2dr DistDirecteSsComplem(Pt2dr aP) const;
      Pt2dr DistInverseSsComplem(Pt2dr aP) const;


       // Les tailles representent des capteurs avant Clip et Reech
      const  Pt2di & Sz() const;
          Pt2dr  SzPixel() const;
          Pt2dr  SzPixelBasik() const;
          void  SetSzPixel(const Pt2dr &) ;

      void  SetSz(const Pt2di &aSz,bool AcceptInitMult=false);
          bool SzIsInit() const;

         void SetParamGrid(const cParamForGrid &);
              // AVANT REECH etc... , sz soit etre connu
      void  SetRayonUtile(double aRay,int aNbDisc);

        // La Box utile tient compte d'une eventuelle  affinite
        // elle peut tres bien avoir des coord negatives
           Box2dr BoxUtile() const;

          void HeritComplAndSz(const ElCamera &);
          void CamHeritGen(const ElCamera &,bool WithCompl,bool WithOrientInterne=true);

          void AddCorrecRefrac(cCorrRefracAPost *);
      void AddDistCompl(bool isDirect,ElDistortion22_Gen *);
      void AddDistCompl
           (
               const std::vector<bool> &  isDirect,
               const std::vector<ElDistortion22_Gen *> &
           );
      Pt2dr DComplC2M(Pt2dr,bool UseTrScN = true  ) const;
      Pt2dr DComplM2C(Pt2dr,bool UseTrScN = true  ) const;
          Pt2dr NormC2M(Pt2dr aP) const;
          Pt2dr NormM2C(Pt2dr aP) const;

          ElDistortion22_Gen   &  Get_dist()        ;
          const ElDistortion22_Gen   &  Get_dist() const  ;
      const std::vector<ElDistortion22_Gen *> & DistCompl() const;
      const std::vector<bool> & DistComplIsDir() const;


          // Ajoute une transfo finale pour aller vers la
          // camera, typiquement pour un crop/scale





          // const ElSimilitude & SimM2C();
          static const Pt2di   TheSzUndef ;
          const std::vector<Pt2dr> &  ContourUtile() ;
          bool  HasRayonUtile() const;
          bool IsInZoneUtile(const Pt2dr & aP,bool Pixel=false) const;
          bool     GetZoneUtilInPixel() const;

          double  RayonUtile() const;
     // A priori lie a HasRayonUtile, mais eventuellement
     // autre chose
          bool    HasDomaineSpecial() const;

         virtual ElDistortion22_Gen   *  DistPreCond() const ;
         ElDistortion22_Gen   *  StaticDistPreCond() const ;
  // Eventuellement a redef; now : DistPreCond != 0
         bool IsForteDist() const;


         virtual bool IsGrid() const;
         virtual ~ElCamera();
   // Coincide avec le centre optique pour les camera stenope, est la position
   // du centre origine pour les camera ortho (utilise pour la geom faisceau)
         virtual Pt3dr OrigineProf() const;
         virtual bool  HasOrigineProf() const;
         const cElPolygone &  EmpriseSol() const;
         const Box2dr &  BoxSol() const;

         const tOrIntIma & IntrOrImaC2M() const;


         Pt2dr ResiduMond2Cam(const Pt2dr & aRes)const;
         tOrIntIma  InhibeScaneOri();
         void RestoreScaneOri(const tOrIntIma &);
    protected :







  // Translation et scale de Normalisation
         Pt2dr                          mTrN;
         double                         mScN;


     std::vector<ElDistortion22_Gen *> mDistCompl;
     std::vector<bool>                 mDComplIsDirect;
         cCorrRefracAPost *                mCRAP;

         ElCamera(bool isDistC2M,eTypeProj);
         ElRotation3D     _orient;

         virtual       ElProj32 &        Proj()       = 0;
         virtual const ElProj32       &  Proj() const = 0;
     Pt2di    mSz;
         Pt2dr    mSzPixel;



     // Une distorsion de "pre-conditionnement" est une fonction "simple"
     // qui approxime la partie non lineaire de la distorsion, si !=0 elle
     // est exprimee dans le sens M->C , 0 signifie identite
     //
     // Elle est utilisee notamment parce que les distorsions "compliquees"
     // peuvent etre exprimees comme la composition d'une distorsion
     // grille a faible distorsion de la distorsion de "pre-conditionnement"
   protected :
     bool             mDIsDirect;
   public :
         virtual       ElDistortion22_Gen   &  Dist()        = 0;
   protected :
         virtual void InstanceModifParam(cCalibrationInternConique &) const  =0;
         virtual const ElDistortion22_Gen   &  Dist() const  = 0;

         void AssertSolInit() const;


         eTypeProj   mTypeProj;
   protected :
         bool        mAltisSolIsDef;
         double      mAltiSol;
         bool        mProfondeurIsDef;
         double      mProfondeur;

   private :

         std::string  mIdentCam;
         std::string  mNameIm;

         //double      mPrecisionEmpriseSol;
         cElPolygone mEmpriseSol;
         Box2dr      mBoxSol;

         double              mRayonUtile;
         bool                mHasDomaineSpecial;
         bool                mDoneScanContU;
         std::vector<Pt2dr>  mContourUtile;

   protected :
         bool                 mParamGridIsInit;
         Pt2dr                mStepGrid;
         double               mRayonInvGrid;
         double               mTime;
         bool                 mScanned;

   private :
         Pt3dr  mVitesse;
         bool   mVitesseIsInit;
         Pt3dr  mIncCentre;

         mutable ElDistortion22_Gen *mStatDPC;
         mutable bool                mStatDPCDone;
};


// Represente les cameras a projection parallele (focale infinie),
//
//  La focale et le point principal sont indissociables de la translation,
//  ce sont des parametres extrinseques representes dans la translation
//  (via le R3toL3/L3toR3)

class cCameraOrtho : public ElCamera
{
    public :
       // Pour appeler methode virtuelle dans cstrteur ...
       static cCameraOrtho * Alloc(const Pt2di & aSz);
         Pt3dr OrigineProf() const;
         bool  HasOrigineProf() const;
         double ResolutionSol() const ;
         double ResolutionSol(const Pt3dr &) const ;

         virtual bool     HasOpticalCenterOfPixel() const; // 
    private :
         double SzDiffFinie() const;
       cCameraOrtho(const Pt2di & aSz);

         Pt3dr R3toL3(Pt3dr) const;
         Pt3dr L3toR3(Pt3dr) const;

         ElDistortion22_Gen   &  Dist()        ;
         const ElDistortion22_Gen   &  Dist() const  ;
         ElProj32 &        Proj()       ;
         const ElProj32       &  Proj() const ;
         void InstanceModifParam(cCalibrationInternConique &) const ;
     Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
     Pt3dr NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const;

         // La notion d'origine n'a pas reellement de sens pour un projection ortho (au mieux elle
         // situee n'importe ou sur le rayon partant du centre de l'image), pourtant il en faut bien une
         // meme completement arbitraire  pour  des fonctions telle que image et profondeur 2 Terrains
         // quand on correle en faisceau
         Pt3dr mCentre;
};

// Preconditionnement en arc-tangente, adapte a un fish eye conservant
// les angles.
//
class cDistPrecondRadial : public ElDistortion22_Gen
{
     public :
         cDistPrecondRadial(double aFocApriori,const Pt2dr & aCentre);
         cPreCondGrid GetAsPreCond() const;
         Pt2dr  DirectAndDer(Pt2dr aP,Pt2dr & aGradX,Pt2dr & aGradY) const;

     private :

        virtual double  DerMultDirect(const double & ) const = 0;
        virtual double  MultDirect(const double & ) const = 0;
        virtual double  MultInverse(const double & ) const = 0;
        virtual int     Mode() const = 0;

        Pt2dr Direct(Pt2dr) const;    // -> DistDirect() = M2C
        bool OwnInverse(Pt2dr &) const ;

        double  mF;
        Pt2dr   mC;
};


// Modele pour fish eye lineaire

class cDistPrecondAtgt : public cDistPrecondRadial
{
      public :
         cDistPrecondAtgt(double aFocApriori,const Pt2dr & aCentre);
      private :
        double  DerMultDirect(const double & ) const ;
        double  MultDirect(const double & ) const ;
        double  MultInverse(const double & ) const ;
        int     Mode() const ;
};

// Modele pour fish eye equisolid

class cDistPrecond2SinAtgtS2 : public cDistPrecondRadial
{
      public :
         cDistPrecond2SinAtgtS2(double aFocApriori,const Pt2dr & aCentre);
      private :
        double  DerMultDirect(const double & ) const ;
        double  MultDirect(const double & ) const ;
        double  MultInverse(const double & ) const ;
        int     Mode() const ;
};

class cDistPrecondSterographique : public cDistPrecondRadial
{
      public :
         cDistPrecondSterographique(double aFocApriori,const Pt2dr & aCentre);
      private :
        double  DerMultDirect(const double & ) const ;
        double  MultDirect(const double & ) const ;
        double  MultInverse(const double & ) const ;
        int     Mode() const ;
};



class cElDistFromCam : public ElDistortion22_Gen
{
    public :
        cElDistFromCam(const ElCamera &,bool UseRayUtile);
        Pt2dr Direct(Pt2dr) const;    // -> DistDirect() = M2C
        bool OwnInverse(Pt2dr &) const ;
        const ElCamera & mCam;
       void  Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const;

    private :
         bool mUseRay;
         Pt2dr  mSzC;
         Pt2dr  mMil;
         double mRayU;
};


// Donne une mesure en pixel (non distordu) de l'ecart d'appariement
REAL EcartTotalProjection
     (
           const ElCamera & CamA,Pt2dr aPF2A,
           const ElCamera & CamB,Pt2dr aPF2B
      );

/*
Pt3dr IntersectionRayonPerspectif
      (
             const ElCamera & CamA, Pt2dr PF2A,
             const ElCamera & CamB, Pt2dr PF2B
       );
      A FAIRE
*/

// Camera a Stenope


class CamStenope : public ElCamera
{
      public :
         CamStenope * DownCastCS() ;
         virtual std::string Save2XmlStdMMName(  cInterfChantierNameManipulateur * anICNM,
                                        const std::string & aOriOut,
                                        const std::string & aNameImClip,
                                        const ElAffin2D & anOrIntInit2Cur
                    ) const;

         double GetRoughProfondeur() const; // Tente Prof puis Alti
         const tParamAFocal   & ParamAF() const;

         void StdNormalise(bool doScale,bool  doTr);
         void StdNormalise(bool doScale,bool  doTr,double aS,Pt2dr  aTr);
         void UnNormalize();
         // .xml ou .ori
         static CamStenope * StdCamFromFile(bool UseGr,const std::string &,cInterfChantierNameManipulateur * anICNM);

         virtual const cCamStenopeDistRadPol * Debug_CSDRP() const;



          // renvoit la distance de p1 a la projection de la droite
          // Inhibee car non testee

          // La methode a ete definie dans la mere, il n'y a aucun interet
          // apparement a la specialiser

          // REAL EcartProj(Pt2dr aPF2A,const ElCamera & CamB,Pt2dr aPF2B);

         // Helas, le SzIm n'est pas integre dans mes CamStenope ...





         CamStenope(bool isDistC2M,REAL Focale,Pt2dr centre,const std::vector<double>  & AFocalParam);
         CamStenope(const CamStenope &,const ElRotation3D &);

         // Par defaut true, mais peut redefini, par exemple pour
         // un fish-eye
         virtual bool CanExportDistAsGrid() const;

         void OrientFromPtsAppui
              (
                 ElSTDNS list<ElRotation3D> &,
                 Pt3dr R3A, Pt3dr R3B, Pt3dr R3C,
                 Pt2dr F2A, Pt2dr F2B, Pt2dr F2C
              );
         void OrientFromPtsAppui
              (
                 ElSTDNS list<ElRotation3D> & Res,
                 const ElSTDNS list<Pt3dr> & PR3 ,
                 const ElSTDNS list<Pt2dr> & PF2
              );
         void OrientFromPtsAppui
              (
                 ElSTDNS list<ElRotation3D>  & Res,
                 const ElSTDNS list<Appar23> & P32
              );

        // Si  NbSol ==  0 et resultat vide => Erreur
        // Sinon *NbSol Contient  le nombre de solution

         ElRotation3D  OrientFromPtsAppui
              (
                 bool TousDevant,
                 const ElSTDNS list<Pt3dr> & PR3 ,
                 const ElSTDNS list<Pt2dr> & PF2 ,
                 REAL * Ecart = 0,
                 INT  * NbSol    = 0
              );

         ElRotation3D  OrientFromPtsAppui
              (
                                bool TousDevant,
                 const ElSTDNS list<Appar23> & P32 ,
                 REAL * Ecart = 0,
                 INT  * NbSol    = 0
              );
     ElRotation3D  CombinatoireOFPAGen
               (
                                bool TousDevant,
                INT  NbTest,
                const ElSTDNS list<Pt3dr> & PR3 ,
                const ElSTDNS list<Pt2dr> & PF2,
                REAL * Res_Dmin,
                bool   ModeRansac,
                                Pt3dr * aDirApprox = 0
                       );

     ElRotation3D  CombinatoireOFPA
               (
                                bool TousDevant,
                INT  NbTest,
                const ElSTDNS list<Pt3dr> & PR3 ,
                const ElSTDNS list<Pt2dr> & PF2,
                REAL * Res_Dmin,
                                Pt3dr * aDirApprox = 0
               );

     ElRotation3D  RansacOFPA
               (
                                bool TousDevant,
                INT  NbTest,
                const ElSTDNS list<Appar23> & P23 ,
                REAL * Res_Dmin,
                                Pt3dr * aDirApprox = 0
               );



     ElRotation3D  CombinatoireOFPA
               (
                                bool TousDevant,
                INT  NbTest,
                                const ElSTDNS list<Appar23> & P32 ,
                REAL * Res_Dmin,
                                Pt3dr * aDirApprox = 0
               );


         // Orientations avec "GPS", i.e. avec centre fixe

         void Set_GPS_Orientation_From_Appuis
                      (
                           const Pt3dr & aGPS,
                           const std::vector<Appar23> & aVApp,
                           int  aNbRansac
                      );

         // Pour compatibilite temporaire avec la proj carto d'orilib
         virtual Ori3D_Std * CastOliLib();  // OO  Def return 0
         Ori3D_Std * NN_CastOliLib();  //OO   Erreur si 0
         double ResolutionPDVVerticale();  //OO   OriLib::resolution, assume implicitement une
                                           // PDV sub verticale
         double ResolutionAngulaire() const;  // OO
         double ResolutionSol() const ;
         double ResolutionSol(const Pt3dr &) const ;
         // Pour l'instant bovin, passe par le xml
         virtual CamStenope * Dupl() const;   // OO


     REAL Focale() const ;
     Pt2dr PP() const ;
     Pt3dr VraiOpticalCenter() const;
     Pt3dr PseudoOpticalCenter() const;
     Pt3dr    OpticalCenterOfPixel(const Pt2dr & aP) const ; 
     Pt3dr OpticalVarCenterIm(const Pt2dr &) const;
     Pt3dr OpticalVarCenterTer(const Pt3dr &) const;
     Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
     Pt3dr NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
     Pt3dr ImEtZ2Terrain(const Pt2dr & aP,double aZ) const;
     void  Coins(Pt3dr &aP1, Pt3dr &aP2, Pt3dr &aP3, Pt3dr &aP4, double aZ) const;
     void  CoinsProjZ(Pt3dr &aP1, Pt3dr &aP2, Pt3dr &aP3, Pt3dr &aP4, double aZ) const;
     Box2dr BoxTer(double aZ) const;

         Pt3dr  ImEtProfSpherik2Terrain(const Pt2dr & aPIm,const REAL & aProf) const; //OO
         Pt3dr  ImDirEtProf2Terrain(const Pt2dr & aPIm,const REAL & aProf,const Pt3dr & aNormPl) const; //OO
         Pt3dr Im1DirEtProf2_To_Terrain  //OO
               (Pt2dr p1,const CamStenope &  ph2,double prof2,const Pt3dr & aDir) const;
         Pt3dr Im1EtProfSpherik2_To_Terrain (Pt2dr p1,const CamStenope &  ph2,double prof2) const;
    void ExpImp2Bundle(const Pt2di aGridSz, const std::string aName) const;

     double ProfInDir(const Pt3dr & aP,const Pt3dr &) const; // OO


         // Sert pour un clonage, par defaut null
         virtual ElProj32             &  Proj();
         virtual const ElProj32       &  Proj() const;
         virtual ElDistortion22_Gen   &  Dist();
         virtual const ElDistortion22_Gen   &  Dist() const;

// Def  : erreur fatale
         virtual cParamIntrinsequeFormel * AllocParamInc(bool isDC2M,cSetEqFormelles &);


         cCamStenopeDistRadPol *Change2Format_DRP
                            (
                      bool C2M,
                      int  aDegreOut,
                      bool CDistPPLie,
                      double Resol,
                      Pt2dr  Origine
                );




         void InstanceModifParam(cCalibrationInternConique &)  const;
         Pt3dr OrigineProf() const;
         bool  HasOrigineProf() const;
         bool  UseAFocal() const;
      private :
         CamStenope(const CamStenope &); // N.I.

      protected :
         ElProjStenope  _PrSten;
         bool                 mUseAF;
         ElDistortion22_Gen * mDist;

         double SzDiffFinie() const;
};





//
// Soit une camera et l'application D du plan dans le
// plan telle que pour un point P=(U,V) camera,
//  R=(D(U),D(V),1.0) soit la direction issu de P,
//  cDistStdFromCam permet de representer cette application
//  sous forme d'une ElDistortion22_Gen, ce qui peut etre utile
//  par exemple pour generer des grilles;
//  l'Inversion n'est pas tres rapide car elle calcule des
//  derivees par differences finies.

class cDistStdFromCam : public ElDistortion22_Gen
{
    public :
             cDistStdFromCam(ElCamera & Cam);
             Pt2dr Direct(Pt2dr) const ;
             void  Diff(ElMatrix<REAL> &,Pt2dr) const;

    private :
              ElCamera & mCam;
};



// Classe pour presenter une image Orilib comme une
// camera Elise

class cDistorsionOrilib;
class cCamera_Orilib : public CamStenope
{
      public :
          cCamera_Orilib(Data_Ori3D_Std *);
      private :
         Ori3D_Std * CastOliLib();  // Def return 0

         Ori3D_Std                       mOri;
         cDistorsionOrilib             * mDist;
         virtual ElDistortion22_Gen   &  Dist();
         virtual const ElDistortion22_Gen   &  Dist() const;
};


// Camera a Stenope Ideale

class CamStenopeIdeale : public CamStenope
{
      public :
         CamStenopeIdeale (bool isDistC2M,REAL Focale,Pt2dr Centre,const std::vector<double> & ParamAF);
         CamStenopeIdeale(const CamStenopeIdeale &,const ElRotation3D &);
     static CamStenopeIdeale  CameraId(bool isDistC2M,const ElRotation3D &);
         CamStenopeIdeale(const CamStenopeIdeale &);

      private :
         virtual ElDistortion22_Gen   &  Dist();
         virtual const ElDistortion22_Gen   &  Dist() const;
};


//  Permet de dupliquer une camera, sans copier les distorsion
//  sans connaitre son origine

class cCamStenopeGen : public CamStenope
{
    public :
      cCamStenopeGen(CamStenope &);
    private :
};



class cCamStenopeDistRadPol : public CamStenope
{
    public :
           const cCamStenopeDistRadPol * Debug_CSDRP() const;

           cCamStenopeDistRadPol
           (
                bool isDistC2M,
            REAL Focale,
        Pt2dr Centre,
        ElDistRadiale_PolynImpair,
                const std::vector<double> & ParamAF,
        ElDistRadiale_PolynImpair * RefDist  = 0,
                const Pt2di &  aSz = ElCamera::TheSzUndef
           );

        ElDistRadiale_PolynImpair & DRad();
        const ElDistRadiale_PolynImpair & DRad() const;

        void write(class  ELISE_fp & aFile) ;
        void write(const std::string & aName);
            static cCamStenopeDistRadPol * read_new(ELISE_fp & aFile);
            static cCamStenopeDistRadPol * read_new(const std::string &);

            virtual cParamIntrinsequeFormel * AllocParamInc(bool isDC2M,cSetEqFormelles &);
        cParamIFDistRadiale * AllocDRadInc(bool isDC2M,cSetEqFormelles &);
    private :
        ElDistRadiale_PolynImpair & mDist;
        ElDistRadiale_PolynImpair mDistInterne;
        // Non implemente , pb sur la copie de _dist
        // (reference mal initialisee)
        //   Surtout avec mDist != mDistInterne
        cCamStenopeDistRadPol(const cCamStenopeDistRadPol &);

            virtual ElDistortion22_Gen   &  Dist();
            virtual const ElDistortion22_Gen   &  Dist() const;
};




// Classe pour modeliser ma distortion telle que
// decrite dans Fraser, ISPRS 97, Vol 52, 149-159
//



class cDistModStdPhpgr : public ElDistRadiale_PolynImpair
{
       public :
                cDistModStdPhpgr(const ElDistRadiale_PolynImpair &);

                Pt2dr Direct(Pt2dr) const ;
                void  Diff(ElMatrix<REAL> &,Pt2dr) const;
                virtual bool OwnInverse(Pt2dr &) const ;    //  Pour "casser" la valeur radiale
                virtual Pt2dr GuessInv(const Pt2dr & aP) const ;

                REAL & P1();
                REAL & P2();
                REAL & b1();
                REAL & b2();

                const REAL & P1() const;
                const REAL & P2() const;
                const REAL & b1() const;
                const REAL & b2() const;

                ElDistRadiale_PolynImpair & DRad();
                const ElDistRadiale_PolynImpair & DRad() const;
            virtual ElDistRadiale_PolynImpair * DRADPol(bool strict = false);

                virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;
                cCalibrationInternePghrStd ToXmlPhgrStdStruct() const;
       private  :
                 bool  AcceptScaling() const;
                 bool  AcceptTranslate() const;
                 void V_SetScalingTranslate(const double &,const Pt2dr &);
                // ElDistRadiale_PolynImpair mDRad;
                double mP1;
                double mP2;
                double mb1;
                double mb2;

};
//   /*Par defaut fonctionne en inverse (ie Cam -> Monde)

class cCamStenopeModStdPhpgr : public cCamStenopeDistRadPol
{
    public :
           cCamStenopeModStdPhpgr
           (
           bool DistIsC2M, // [1]
           REAL Focale,
           Pt2dr Centre,
           cDistModStdPhpgr,
               const std::vector<double> & ParamAF
       );
       //  true if only linear param are !=0
       bool     DistBijective   () const override;
       cDistModStdPhpgr & DModPhgrStd();
       const cDistModStdPhpgr & DModPhgrStd() const;
        // [1]  DistIsC2M:
        // En point de liaison les equation sont faite C->M, compte
        // tenu de l'absence d'inversion triviale pour le Modele Std,
        // on a interet a toujours raisonner dans ce sens
           virtual ElDistortion22_Gen   &  Dist();
           virtual const ElDistortion22_Gen   &  Dist() const;
            virtual cParamIntrinsequeFormel * AllocParamInc(bool isDC2M,cSetEqFormelles &);
        cParamIFDistStdPhgr * AllocPhgrStdInc(bool isDC2M,cSetEqFormelles &);
    private :
           cCamStenopeModStdPhpgr(const cCamStenopeModStdPhpgr &); // N.I.
           cDistModStdPhpgr mDist;
};






class cCamStenopeDistPolyn : public CamStenope
{
    public :
           cCamStenopeDistPolyn
           (bool isDistC2M,REAL Focale,Pt2dr Centre, const ElDistortionPolynomiale &,const std::vector<double> &);
        const ElDistortionPolynomiale & DistPol() const;
    private :
        ElDistortionPolynomiale mDist;
        cCamStenopeDistPolyn(const cCamStenopeDistPolyn &);
            virtual ElDistortion22_Gen   &  Dist();
            virtual const ElDistortion22_Gen   &  Dist() const;
};

class cCamStenopeDistHomogr : public CamStenope
{
    public :
           cCamStenopeDistHomogr
           (bool isDistC2M,REAL Focale,Pt2dr Centre, const cDistHomographie &,const std::vector<double> &);
       const cElHomographie & Hom() const;
    private :
        cDistHomographie   mDist;
        cCamStenopeDistHomogr(const cCamStenopeDistHomogr &);
            virtual ElDistortion22_Gen   &  Dist();
            virtual const ElDistortion22_Gen   &  Dist() const;
};


class cDistCamStenopeGrid : public ElDistortion22_Gen
{
     public :
       friend class cCamStenopeGrid;
       Pt2dr DirectAndDer(Pt2dr aP,Pt2dr & aGX, Pt2dr & aGY) const;
       Pt2dr Direct(Pt2dr) const;
       virtual bool OwnInverse(Pt2dr &) const ;    //  return true

       // Si RayonInv <=0 pas utilise
       static cDistCamStenopeGrid * Alloc(bool P0P1IsBoxDirect,double aRayInv,const CamStenope &,Pt2dr aStepGr,bool doDir=true,bool doInv=true);

       cDistCamStenopeGrid
       (
             ElDistortion22_Gen *,
             cDbleGrid *
       );

       static void Test(double aRayInv,const CamStenope &,Pt2dr aStepGr);

       virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;

       std::string Type() const;
     private :
       cDistCamStenopeGrid(const cDistCamStenopeGrid &); // N.I.


       ElDistortion22_Gen * mPreC;  // Pre Conditionnement optionnel
       cDbleGrid  *         mGrid;
};


class cCamStenopeGrid :  public CamStenope
{
     public :
       static cCamStenopeGrid * Alloc(double aRayonInv,const CamStenope &,Pt2dr aStepGr,bool doDir=true,bool doInv=true);
       cCamStenopeGrid
       (
             const double & aFoc,
             const Pt2dr &,
             cDistCamStenopeGrid *,
             const Pt2di  & aSz,
             const std::vector<double> & ParamAF
       );

       Pt2dr L2toF2AndDer(Pt2dr aP,Pt2dr & aGX,Pt2dr & aGY);
     private :
       bool IsGrid() const;
       ElDistortion22_Gen   *  DistPreCond() const;

       cCamStenopeGrid(const cCamStenopeGrid &); // N.I.
       cDistCamStenopeGrid * mDGrid;
};



class CalcPtsInteret
{
       public :

            typedef std::list<Pt2dr>  tContainerPtsInt;


            static Fonc_Num CritereFonc(Fonc_Num);  // Typiquement une genre Harris
            static Pt2di GetOnePtsInteret(Flux_Pts,Fonc_Num aFonc); // aFonc -> Avant appli de critere


            // Dans cette version, on specifie la taille des carres
            // de recherche de 1 Pts Interet

            static tContainerPtsInt GetEnsPtsInteret_Size
                                    (
                                         Pt2di aP0,
                                         Pt2di aP1,
                                         Fonc_Num aFonc,
                                         REAL aSize,
                                         REAL aRatio = 0.8
                                    );

            static tContainerPtsInt GetEnsPtsInteret_Nb
                                    (
                                         Pt2di aP0,
                                         Pt2di aP1,
                                         Fonc_Num aFonc,
                                         INT  aNb,  // NbTot = NbX * NbY
                                         REAL aRatio = 0.8
                                     );

            static tContainerPtsInt GetEnsPtsInteret_Size
                                    (
                                         Im2D_U_INT1,
                                         REAL aSize,
                                         REAL aRatio = 0.8
                                    );
            static tContainerPtsInt GetEnsPtsInteret_Nb
                                    (
                                         Im2D_U_INT1,
                                         INT  aNb,
                                         REAL aRatio = 0.8
                                    );

       private :

};


class cElFaisceauDr2D
{
    public :
          // Calcul un point projectif correspond a un point de
          // convergence commun du faisceau de droite,
          // suppose une valeur initiale approche en teta,phi

       void PtsConvergence(REAL  & teta0,REAL & phi0, bool OptimPhi);

         // Itere, s'arrete apres NbStep Etape ou si le
         // de residu < Epsilon, ou si le delta residu < DeltaRes
       void PtsConvergenceItere
                (
                   REAL  & teta0,REAL & phi0,INT NbStep,
                   REAL Epsilon, bool OptimPhi,REAL DeltaResidu =-1
                );
        //  Residu de convergence MOYEN du faisceau vers le point
    //  projectif

       REAL ResiduConvergence(REAL  teta,REAL phi);

       void AddFaisceau(Pt2dr aP0,Pt2dr aDir,REAL aPds);



       // Si tout les faisceau ont approximativement la meme
       // direction renvoie une estimation de cette direction,
       // ne necessite pas de valeur initiale
       REAL TetaDirectionInf();


      void CalibrDistRadiale
           (
               Pt2dr   &            aC0,
               bool                 CentreMobile,
               REAL    &            TetaEpip,
               REAL    &            PhiEpip,
               std::vector<REAL> &  Coeffs
           );


    private :
        enum {IndTeta,IndPhi};
    class cFaisceau : public SegComp
    {
        public :
            cFaisceau(Pt2dr aP0,Pt2dr aDir,REAL aPds);
            REAL Pds() const;
        private :
            REAL mPds;
    };

        typedef cFaisceau        tDr;
        typedef std::list<tDr>   tPckDr;
        typedef tPckDr::iterator tIter;

        tIter Begin() {return mPckDr.begin();}
        tIter End() {return mPckDr.end();}
        INT NbDr() {return (INT) mPckDr.size();}

        tPckDr  mPckDr;
};




class cElemMepRelCoplan
{
        public :
             cElemMepRelCoplan
             (
                const cElHomographie &,
                const ElRotation3D &
             );

             cElemMepRelCoplan ToGivenProf(double aProProf);


             bool PhysOk() const;
             void Show() const;
             REAL AngTot() const;
             REAL Ang1() const;
             REAL Ang2() const;
             const ElRotation3D & Rot() const;

             double TestSol() const;
             void TestPack(const ElPackHomologue &) const;


         // Normale au plan dans le repere cam1
             Pt3dr          Norm() const;
         //  Distance entre le centre optique et l'image "reciproque"
         //  du centre camera 1 sur le plan, permet de normalisee
         //  les bases
         REAL DPlan() const;


             // "Vraie" distance min entre le plan et
         double DistanceEuclid() const;

         // Idem camera2
         REAL DPlan2() ;
                // Point du plan, ayant P1 comme image  par cam1
                // (en coord camera 1)
             Pt3dr ImCam1(Pt2dr aP1);
             // Homographie envoyant un (u,v,1) en (X,Y,0)
             cElHomographie HomCam2Plan(double * aResidu = 0);

         cElPlan3D  Plan() const;
         const Pt3dr & P0() const;
         const Pt3dr & P1() const;
         const Pt3dr & P2() const;

             // Des coordoones Cam2 a Cam1

             Pt3dr ToR1(Pt3dr aP2) const;

        private :

                // Point du plan, ayant P2 comme image  par cam2
                // (en coord camera 1)
             Pt3dr ImCam2(Pt2dr aP2);

             REAL AngleNormale(Pt3dr);


             cElHomographie mHom;
             cElHomographie mHomI;
             ElRotation3D   mRot;

             // tous les points 3D sont en coordonnees Cam1

         // mP0,mP1,mP2  trois point (coord Cam1) du plan
         // mNorm un vecteur unitaire  de la normale au plan
             Pt3dr          mP0;
             Pt3dr          mP1;
             Pt3dr          mP2;
             Pt3dr          mNorm;
             Pt2dr          mCZCC2; // Centre Zone Commune
             Pt2dr          mCZCC1;
             Pt3dr          mCZCMur;
             Pt3dr          mCOptC1;
             Pt3dr          mCOptC2;
             REAL           mProfC1;
             REAL           mProfC2;
             REAL           mAng1;
             REAL           mAng2;
             REAL           mAngTot;
         double         mDEuclidP;
};

class cResMepRelCoplan
{
        public :
           cResMepRelCoplan();
           cElemMepRelCoplan & RefBestSol();
           cElemMepRelCoplan * PtrBestSol();


           void AddSol(const cElemMepRelCoplan &);
           const std::list<ElRotation3D> &  LRot() const;
           const std::vector<cElemMepRelCoplan> & VElOk() const;
           const std::list<cElemMepRelCoplan>    & LElem() const;
        private :
           std::list<cElemMepRelCoplan>    mLElem;
           std::vector<cElemMepRelCoplan>  mVElOk;
           std::list<ElRotation3D>         mLRot;
};

     //  -----------------------------------------------
     //
     //      POLYGONE D'ETALONNAGE
     //
     //  -----------------------------------------------


class cMirePolygonEtal
{
      public :
          bool IsNegatif() const;
          cMirePolygonEtal();
          static const cMirePolygonEtal & IgnMireN6();
          static const cMirePolygonEtal & ENSGMireN6();
          static const cMirePolygonEtal & MtdMire9();
          static const cMirePolygonEtal & IGNMire7();
          static const cMirePolygonEtal & IGNMire5();
          static const cMirePolygonEtal & SofianeMire3();
          static const cMirePolygonEtal & SofianeMire2();
          static const cMirePolygonEtal & SofianeMireR5();
          static const cMirePolygonEtal & MT0();
          static const cMirePolygonEtal & MTClous1();
      static const cMirePolygonEtal & GetFromName(const std::string &);
      INT NbDiam() const;
      REAL KthDiam(INT aK) const;
      const std::string & Name() const;


      private :

          static const double TheIgnN6[6];
          static const double TheENSG6[6];
          static const double TheMTD9[6];
          static const double TheIGNDiams7[7];
          static const double TheIGNDiams5[5];
          static const double TheSofianeDiam3[1];
          static const double TheSofianeDiam2[1];
          static const double TheSofianeDiamR5[5];
          static const double TheMT0Diams[1];
          static const double TheDiamMTClous1[1];

          static cMirePolygonEtal TheNewIGN6;
          static cMirePolygonEtal TheNewENSG6;
          static cMirePolygonEtal TheMTDMire9;
          static cMirePolygonEtal TheIGNMire7;
          static cMirePolygonEtal TheIGNMire5;
          static cMirePolygonEtal TheSofiane3;
          static cMirePolygonEtal TheSofiane2;
          static cMirePolygonEtal TheSofianeR5;
          static cMirePolygonEtal TheMT0;
          static cMirePolygonEtal TheMTClous1;

          cMirePolygonEtal(const std::string & mName,const double *,INT NB);

          std::string  mName;
          const double *     mDiams;
          INT          mNBDiam;
};


// La classe cCibleCalib est la nouvelle "classe" pour representer les
// cibles, elle est plus complete (normale ...) et admet une lecture
// ecriture standard par xml. Pour des raisons de compatibilite on
// conserve cCiblePolygoneEtal qui contient un cCibleCalib *. C'est
// un peu batard mais correspond au moyen le + econome de gerer
//  cette classe qui n'a pas vocation a generer de grand developpement


class cCibleCalib;
class cPolygoneCalib;
class cComplParamEtalPoly;

class cCiblePolygoneEtal
{
      public :
         typedef int tInd;
     typedef enum
     {
             ePerfect = 0,
         eBeurk   = 1
     } tQualCible;


         void SetPos(Pt3dr aP );
         Pt3dr Pos() const;
         tInd Ind() const;
     const cMirePolygonEtal &  Mire() const;
     tQualCible Qual() const;

         cCiblePolygoneEtal
         (
             tInd,Pt3dr,const cMirePolygonEtal &,INT Qual,
             cCibleCalib *,
         int anOrder
         );
     cCiblePolygoneEtal();

     cCibleCalib * CC() const;
     int Order() const;

      private :
         tInd                     mInd;
         Pt3dr                    mPos;
         const cMirePolygonEtal * mMire;
     tQualCible               mQual;
     cCibleCalib *            mCC;
         int                      mOrder;
};

class cPolygoneEtal
{
       public :
           virtual void AddCible(const cCiblePolygoneEtal &) =0;
           virtual const cCiblePolygoneEtal & Cible(cCiblePolygoneEtal::tInd) const = 0;
       virtual ~cPolygoneEtal();
       static cPolygoneEtal * IGN();
       static cPolygoneEtal * FromName
                              (
                      const std::string &,
                      const cComplParamEtalPoly * aParam
                                  );

       typedef std::list<const cCiblePolygoneEtal *>  tContCible;

       const  tContCible  & ListeCible() const;
       cPolygoneCalib * PC() const;
       void SetPC(cPolygoneCalib *);
       protected :
       void LocAddCible(const cCiblePolygoneEtal *);
       cPolygoneEtal();
       void PostProcess();
       private :
           tContCible mListeCible;
       cPolygoneCalib * mPC;
};


class cPointeEtalonage
{
      public :
    cPointeEtalonage(cCiblePolygoneEtal::tInd,Pt2dr,const cPolygoneEtal &);
    Pt2dr PosIm() const;
    Pt3dr PosTer() const;
    void SetPosIm(Pt2dr);
    const cCiblePolygoneEtal  & Cible() const;
    bool  UseIt () const;
    REAL  Pds()    const;
      private :

    Pt2dr                       mPos;
    const cCiblePolygoneEtal *  mCible;
        bool                        mUseIt;
    REAL                        mPds;

};


class cSetNImSetPointes;
class cSetPointes1Im
{
    public :
          friend class cSetNImSetPointes;
          cSetPointes1Im
          (
              const cPolygoneEtal &,
          const std::string &,
          bool  SVP = false  // Si true et fichier inexistant cree set vide
          );
      typedef std::list<cPointeEtalonage> tCont;
      tCont  & Pointes() ;
      cPointeEtalonage & PointeOfId(cCiblePolygoneEtal::tInd);
      cPointeEtalonage * PointeOfIdSvp(cCiblePolygoneEtal::tInd);
      void RemoveCibles(const std::vector<INT> & IndToRemove);
      bool  InitFromFile(const cPolygoneEtal &,ELISE_fp & aFp,bool InPK1);
    private :
      tCont mPointes;
      cSetPointes1Im();
};

class cSetNImSetPointes
{
       public :
          cSetNImSetPointes
          (
              const cPolygoneEtal &,
          const std::string &,
          bool  SVP = false  // Si true et fichier inexistant cree set vide
          );
      typedef std::list<cSetPointes1Im> tCont;
      tCont  & Pointes() ;
      INT NbPointes();
       private :
      tCont mLPointes;

};

class cDbleGrid : public ElDistortion22_Gen
{
     public :

         // Dans le cas ou il s'agit d'une grille photogram
         // le PP est l'image reciproque de (0,0),
         // la Focale est calculee par differnce finie,
         // en X, avec un pas de 1 Pixel
     REAL Focale();
     Pt2dr PP() ;
         const Pt2dr & P0_Dir() const;
         const Pt2dr & P1_Dir() const;
     const Pt2dr  & Step_Dir() const;

     static cDbleGrid *  StdGridPhotogram(const std::string & aName,int aSzDisc=30);



         cDbleGrid
         (
         bool P0P1IsBoxDirect,
         bool AdaptStep,
             Pt2dr aP0,Pt2dr aP1,
             Pt2dr               aStep,
             ElDistortion22_Gen &,
         const std::string & aName = "DbleGrid",
             bool  doDir = true,
             bool  doInv = true
         );
     const std::string & Name() const;

     static cDbleGrid * read(const  std::string &);
     static cDbleGrid * read(ELISE_fp & aFile);

         void write(const  std::string &);
         void write(ELISE_fp & aFile);
     ~cDbleGrid();
         Pt2dr ValueAndDer(Pt2dr aRealP,Pt2dr & aGradX,Pt2dr & aGradY);
        virtual Pt2dr Direct(Pt2dr) const  ;    //
    const PtImGrid & GrDir() const ;
    const PtImGrid & GrInv() const ;
    PtImGrid & GrDir() ;
    PtImGrid & GrInv() ;

        // Applique un chgt d'echelle sur les image direct
        // typiquement si ChScale=Focale() Tr= PP() ; alors
        // on une correction de distorsion assez classique
        void SetTrChScaleDir(REAL aChScale,Pt2dr aTr);
        void SetTrChScaleInv(REAL aChScale,Pt2dr aTr);

        class cXMLMode {
            public :
               cXMLMode(bool toSwapDirInv = false);
               bool    toSwapDirInv;
        };
        cDbleGrid(cXMLMode,const std::string & aDir,const std::string & aXML);


    cDbleGrid(const cGridDirecteEtInverse &);

        void PutXMWithData
             (
                    class cElXMLFileIn &       aFileXML,
                    const std::string &  aNameDir
             );
       bool StepAdapted() const;

       // Nouveau format avec Image incluse
       void SaveXML(const std::string &);


     private :

         void SauvDataGrid
              (
                  const std::string &  aNameDir,
                  Im2D_REAL8 anIm,
                  const std::string & aName
              );

        virtual bool OwnInverse(Pt2dr &) const ;    //  return true
        virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  differentielle
    cDbleGrid(PtImGrid*,PtImGrid*);



         PtImGrid * pGrDir;
         PtImGrid * pGrInv;
     std::string mName;
};


class cEqTrianguApparImage
{
          typedef REAL tV6[6];
     public :
          // Ordre des variables A.x A.y B.x etc ....
          cEqTrianguApparImage(INT aCapa);
          ~cEqTrianguApparImage();
          void Reset();
          void Close();
          void Add
               (
                   REAL aI,
                   REAL aJ, REAL aDxJ, REAL aDyJ,
                   REAL aPdsA, REAL aPdsB, REAL aPdsC
               );

          void SetDer(REAL & aCste,REAL  * aDer,INT aK);

     private :
          INT  mCapa;
          INT  mN;     // Nombre de point
          REAL mSI1;   // Moyenne/Somme de l'image I
          REAL mSI2;   // Moyenne/Somme du carre l'image I
          REAL   mSigmaI;
          REAL mSJ1;   // Moyenne/Somme de l'image J
          REAL mSJ2;   // Moyenne/Somme du carre de l'image J
          REAL   mSigmaJ2;
          REAL   mSigmaJ;
          REAL   mSigmaJ3;


          tV6   mDerSJ1;     // Derivee de SJ1 / a A.x A.y B.x etc ..
          tV6   mDerSJ2;     // Derivee de SJ2 / a A.x A.y B.x etc ..
          tV6   mDerSigmaJ2; // Derivee SigmaJ2 = mSJ2 - Square(mSJ1)
          tV6 * mDerJk;      // mDerJk[k][0] derivee de kieme J / a A.x

          REAL * mVI;
          REAL * mVJ;
};



ElRotation3D RotationCart2RTL(Pt3dr  aP, double aZ);
ElRotation3D RotationCart2RTL(Pt3dr  aP, double aZ,Pt3dr axe_des_x);



class cAnalyseZoneLiaison
{
    public :
        cAnalyseZoneLiaison();
        void AddPt(const Pt2dr &);
        void Reset();

        //  2 - 1 correpond a l'inertie du petit axe
        //  1 -1  correpond a la moyenne des val abs (dans le ptit axe)
        //  2 -0   correpond a l'inertie du petit axe avec une ponderation
        //         normalisee independante du nombre de points
        double Score(double ExposantDist,double ExposantPds);
        const std::vector<Pt2dr> & VPts() const;
    private  :
        cAnalyseZoneLiaison(const cAnalyseZoneLiaison&);
        std::vector<Pt2dr>  mVPts;
        RMat_Inertie        mMat;
};


class   cCS_MapIm2PlanProj : public ElDistortion22_Gen
{
        public :
          cCS_MapIm2PlanProj(CamStenope * pCam) ;
// Directe Image -> Direction de rayon

          Pt2dr Direct(Pt2dr aP) const;
        private :
          bool OwnInverse(Pt2dr & aP) const;
          void  Diff(ElMatrix<REAL> &,Pt2dr) const;

          CamStenope & mCam;
};

std::string LocPxFileMatch(const std::string & aDir,int aNum,int aDeZoom);
std::string LocPx2FileMatch(const std::string & aDir,int aNum,int aDeZoom);
std::string LocMasqFileMatch(const std::string & aDirM,int aNum);
std::string LocCorFileMatch(const std::string & aDir,int aNum);


class cCpleEpip
{
     public :
         cCpleEpip
         (
             const std::string & aDir,
             double aScale,
             const CamStenope & aC1,const std::string & aName1,
             const CamStenope & aC2,const std::string & aName2,
             const std::string & PrefLeft =   "Left_",
             const std::string & PrefRight =  "Right_"
         );
         Pt2dr RatioExp() const;
         double RatioCam() const;
         const bool & Ok() const;
         const int & SzX() const;
         const int & SzY() const;

         double BSurHOfPx(bool Im1,double aPx);
         Fonc_Num BSurHOfPx(bool Im1,Fonc_Num aPx);

         std::string Dir();

         bool IsIm1(const std::string & aNameIm);  // Erreur si ni Im1 ni Im2


         std::string LocDirMatch(const std::string & Im);
         std::string LocNameImEpi(const std::string & Im,int aDeZoom=1,bool Pyram = true);
         std::string LocPxFileMatch(const std::string & Im,int aNum,int aDeZoom);
         std::string LocMasqFileMatch(const std::string & Im,int aNum);


         std::string LocDirMatch(bool Im1);
         std::string LocNameImEpi(bool Im1,int aDeZoom=1,bool Pyram=true);
         std::string LocPxFileMatch(bool Im1,int aNum,int aDeZoom);
         std::string LocMasqFileMatch(bool Im1,int aNum);



         bool IsLeft(bool Im1);
         bool IsLeft(const std::string &);


         void ImEpip(Tiff_Im aFile,const std::string & aNameOriIn,bool Im1,bool InParal=true,bool DoIm=true,const char * NameHom= 0,int aDegPloCor=-1,bool ExpTxt=false);
         void AssertOk() const;

         void LockMess(const std::string & aMes);
         void SetNameLock(const std::string & anExt);
     private :

         Box2dr   BoxCam(const CamStenope & aCam,const CamStenope & aCamOut,bool Show) const;
         inline Pt2dr TransfoEpip(const Pt2dr &,const CamStenope & aCamIn,const CamStenope & aCamOut) const;
         CamStenopeIdeale  CamOut(const CamStenope &,Pt2dr aPP,Pt2di aSz);





         double             mScale;
         std::string        mDir;
         cInterfChantierNameManipulateur  * mICNM;
         const CamStenope & mCInit1;
         std::string        mName1;
         const CamStenope & mCInit2;
         std::string        mName2;
         std::string        mNamePair;
         std::string        mPrefLeft;
         std::string        mPrefRight;
         Pt2di              mSzIn;
         double             mFoc;
         ElMatrix<REAL>     mMatM2C;
         ElMatrix<REAL>     mMatC2M;

         CamStenopeIdeale   mCamOut1;
         CamStenopeIdeale   mCamOut2;
         bool               mOk;
         bool               mFirstIsLeft;
         int                mSzX;
         int                mSzY;
         double             mPxInf;

         std::string        mFileLock;
};
std::string LocDirMec2Im(const std::string & Im1,const std::string & Im2);
std::string StdNameImDeZoom(const std::string & Im1,int aDeZoom);


cCpleEpip * StdCpleEpip
          (
             std::string  aDir,
             std::string  aNameOri,
             std::string  aNameIm1,
             std::string  aNameIm2
          );



// Pour assurer la compatibilite avec les format 2003 ....
CamStenope * CamCompatible_doublegrid(const std::string & aNameFile);



class cTxtCam
{
    public :

       cTxtCam();
       void SetVitesse(const Pt3dr& aV);

       std::string          mNameIm;
       std::string          mNameOri;
       CamStenope *         mCam;
       CamStenope *         mRefCam;// En cas de reference exacte pour faire du reverse engenering
       cOrientationConique  * mOC;
       double               mPrio;
       bool                 mSelC;
       Pt3dr                mC;  // Center
       Pt3dr                mV;  // Vitesse
       Pt3dr                mWPK;  // Angles
       bool                 mVIsCalc;  // Vitesse
       int                  mNum;
       int                  mNumBande;
       double               mTime;
       const cMetaDataPhoto *     mMTD;
};
typedef cTxtCam * cTxtCamPtr;

class cCmpPtrCam
{
    public :
       bool operator() (const cTxtCamPtr & aC1  ,const cTxtCamPtr & aC2);
};


class cCalibrationInterneGridDef;

class cDistorBilin :   public ElDistortion22_Gen
{
     public :
          friend class cPIF_Bilin;

          friend void Test_DBL();

          cDistorBilin(Pt2dr aSz,Pt2dr aP0,Pt2di aNb);
          Pt2dr Direct(Pt2dr) const ;

          Pt2dr & Dist(const Pt2di aP) {return mVDist[aP.x + aP.y*(mNb.x+1)];}
          const Pt2dr & Dist(const Pt2di aP) const {return mVDist[aP.x + aP.y*(mNb.x+1)];}
          const Pt2di & Nb() const {return mNb ;}


          virtual cCalibDistortion ToXmlStruct(const ElCamera *) const;
          cCalibrationInterneGridDef ToXmlGridStruct() const;

          static cDistorBilin FromXmlGridStuct(const cCalibrationInterneGridDef & );


          bool  AcceptScaling() const;
          bool  AcceptTranslate() const;
          void V_SetScalingTranslate(const double &,const Pt2dr &);

     private  :
        //  ==== Tests ============
          Box2dr BoxRab(double aMulStep) const;
          void Randomize(double aFact=0.1);
          void InitAffine(double aF,Pt2dr aPP);

        //  =============
          void  Diff(ElMatrix<REAL> &,Pt2dr) const;
          Pt2dr ToCoordGrid(const Pt2dr &) const;
          Pt2dr FromCoordGrid(const Pt2dr &) const;
          // Renvoie le meilleur interval [X0, X0+1[ contenat aCoordGr, valide qqsoit aCoordGr
          void GetDebInterval(int & aX0,const int & aSzGrd,const double & aCoordGr) const;
          //  tel que aCoordGr soit le barry de (aX0,aX0+1) avec (aPdsX0,1-aPdsX0)  et 0<= aX0 < aSzGr, aX0 entier
          void GetDebIntervalAndPds(int & aX0,double & aPdsX0,const int & aSzGrd,const double & aCoordGr) const;
          //  A partir d'un points en coordonnees grille retourne le coin bas-gauche et le poids
          void GetParamCorner(Pt2di & aCornerBG,Pt2dr & aPdsBG,const Pt2dr & aCoorGr) const;
          void InitEtatFromCorner(const Pt2dr & aCoorGr) const;

          Pt2dr                               mP0;
          Pt2dr                               mP1;
          Pt2dr                               mStep;
          Pt2di                               mNb;
          std::vector<Pt2dr >                 mVDist;

          mutable Pt2di                               mCurCorner;
          mutable double                              mPds[4];
};

class cCamStenopeBilin : public CamStenope
{
    public :
           cCamStenopeBilin
           (
               REAL Focale,
               Pt2dr Centre,
               const  cDistorBilin & aDBL
           );

            const ElDistortion22_Gen & Dist() const;
            ElDistortion22_Gen & Dist() ;
            const cDistorBilin & DBL() const;

            cCamStenopeBilin * CSBil_SVP();
    private :

           cDistorBilin mDBL;
};


/*
   Teste , equivalence de :

       PVExactCostMEP Pt3dr   : 0.376684  => 3/4 fois + rapide
       PVExactCostMEP Pt2drA  : 0.679608
       ExactCostMEP           : 1.27514

  A Faire QuasiExactCostFaiscMEP
*/
       

double QuickD48EProjCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax);
double ProjCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax);
double DistDroiteCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax);
double PVCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax);
double PVCostMEP(const ElRotation3D & aR2to1 ,const Pt3dr & aP1,const Pt3dr & aP2,double aTetaMax);
double LinearCostMEP(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax);
double LinearCostMEP(const ElRotation3D & aR2to1 ,const Pt3dr & aP1,const Pt3dr & aP2,double aTetaMax);


double QuickD48EProjCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax);
double ProjCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax);
double DistDroiteCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax);
double PVCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax);
double LinearCostMEP(const ElPackHomologue & aPack,const ElRotation3D & aRot,double aTetaMax);



Pt3dr MedianNuage(const ElPackHomologue & aPack,const ElRotation3D & aRot);
ElRotation3D RansacMatriceEssentielle(const ElPackHomologue & aPack,const ElPackHomologue & aPackRed,double aFoc);

void InitPackME
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr>  &aVP2,
          std::vector<double>  &aVPds,
          const  ElPackHomologue & aPack
     );



class  cInterfBundle2Image
{
     public :
           cInterfBundle2Image(int mNbCple,double aFoc);
           double ErrInitRobuste(const ElRotation3D &aRot,double aProp = 0.75);
           ElRotation3D   OneIterEq(const  ElRotation3D &aRot,double & anErrStd);
           double         ResiduEq(const  ElRotation3D &aRot,const double & anErrStd);

           static cInterfBundle2Image * LineariseAngle(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCste0);
           static cInterfBundle2Image * LinearDet(const  ElPackHomologue & aPack,double aFoc);
           static cInterfBundle2Image * Bundle(const  ElPackHomologue & aPack,double aFoc,bool UseAccelCoordCste);
           virtual  ~cInterfBundle2Image();

           virtual const std::string & VIB2I_NameType() = 0 ;
           virtual double VIB2I_PondK(const int & aK) const = 0;
           virtual double VIB2I_ErrorK(const ElRotation3D &aRot,const int & aK) const = 0;
           virtual double VIB2I_AddObsK(const int & aK,const double & aPds) =0;
           virtual void   VIB2I_InitNewRot(const ElRotation3D &aRot) = 0;
           virtual ElRotation3D    VIB2I_Solve() = 0;

     protected :

           void   OneIterEqGen(const  ElRotation3D &aRot,double & anErrStd,bool AddEq);
           int    mNbCple;
           double mFoc;
     private :

};

class cResMepCoc
{
     public :

          cResMepCoc(ElMatrix<double> & aMat,double aCostRPur,const ElRotation3D & aR,double aCostVraiRot,Pt3dr aPMed);

          ElMatrix<double>  mMat;
          double            mCostRPure;
          ElRotation3D      mSolRot;
          double            mCostVraiRot;
          Pt3dr             mPMed;
};

cResMepCoc MEPCoCentrik(bool Quick,const ElPackHomologue & aPack,double aFoc,const ElRotation3D * aRef,bool Show);

class L2SysSurResol;
void SysAddEqMatEss(const double & aPds,const Pt2dr & aP1,const Pt2dr & aP2,L2SysSurResol & aSys );
ElMatrix<REAL> ME_Lign2Mat(const double * aSol);
ElRotation3D MatEss2Rot(const  ElMatrix<REAL> & aMEss,const ElPackHomologue & aPack);
ElPackHomologue PackReduit(const ElPackHomologue & aPack,int aNbInit,int aNbFin);

double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH);
double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2);



// Devrait remplacer les anciennes, on y va progressivement
double  NEW_SignInters(const ElPackHomologue & aPack,const ElRotation3D & aR2to1,int & NbP1,int & NbP2);
ElRotation3D  NEW_MatEss2Rot(const  ElMatrix<REAL> & aMEss,const ElPackHomologue & aPack,double * aDistMin = 0);


typedef std::vector<std::vector<Pt2df> *> tMultiplePF;

void TestBundle3Image
     (
          double               aFoc,
          const ElRotation3D & aR12,
          const ElRotation3D & aR13,
          const tMultiplePF  & aH123,
          const tMultiplePF  & aH12,
          const tMultiplePF  & aH13,
          const tMultiplePF  & aH23,
          double aPds3
     );

class cParamCtrlSB3I
{
    public :
         cParamCtrlSB3I(int aNbIter,bool FilterOutlayer=true,double aResStop=-1);

         const int    mNbIter;
         const double mResiduStop;
         const bool   mFilterOutlayer;
         double       mRes3;
         double       mRes2;
};

bool SolveBundle3Image
     (
          double               aFoc,
          ElRotation3D & aR12,
          ElRotation3D & aR13,
          Pt3dr &        aPMed,
          double &       aBOnH,
          const tMultiplePF  & aH123,
          const tMultiplePF  & aH12,
          const tMultiplePF  & aH13,
          const tMultiplePF  & aH23,
          double aPds3,
          cParamCtrlSB3I & aParam
     );



void Merge2Pack
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr> & aVP2,
          int aSeuil,
          const ElPackHomologue & aPack1,
          const ElPackHomologue & aPack2
     );

void Merge3Pack
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr> & aVP2,
          std::vector<Pt2dr> & aVP3,
          int aSeuil,
          const std::vector<Pt2dr> & aV12,
          const std::vector<Pt2dr> & aV21,
          const std::vector<Pt2dr> & aV13,
          const std::vector<Pt2dr> & aV31,
          const std::vector<Pt2dr> & aV23,
          const std::vector<Pt2dr> & aV32
     );

std::vector<ElRotation3D> VRotB3(const ElRotation3D & aR12,const ElRotation3D &aR13);
double QualInterSeg(const std::vector<ElRotation3D> & aVR,const tMultiplePF & aVPMul);
Pt3dr InterSeg(const std::vector<ElRotation3D> & aVR,const std::vector<Pt2dr> & aVP,bool & Ok,double * aResidu);


std::vector<ElRotation3D> OrientTomasiKanade
                          (
                             double &            aPrec,
                             const tMultiplePF & aVPF3,
                             int                 aNbMin,
                             int                 aNbMax,
                             double              aPrecCible,
                             std::vector<ElRotation3D> * aVRotInit
                          );

// Fix a global variable,  dirty !!!!
void SetExtensionIntervZInApero(const double);

#endif // !  _ELISE_GENERAL_PHOTOGRAM_H





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
