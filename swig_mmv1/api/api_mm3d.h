#ifndef API_MM3D_H
#define API_MM3D_H

#include "StdAfx.h"
/**
@file
@brief New methods for python API and existing classes
**/

class CamStenope;

//-------------------- Nouvelles methodes ---------------------

//!internal usage
void mm3d_init();

//! Create CamStenope form a XML file
CamStenope *  CamOrientFromFile(std::string filename);

//! Create XML of an ideal Cam
void createIdealCamXML(double focale, Pt2dr aPP, Pt2di aSz, std::string oriName, std::string imgName, std::string idCam, ElRotation3D &orient, double prof, double rayonUtile);
//void createIdealCamXML(CamStenope * aCam, Pt2dr aPP, Pt2di aSz, std::string oriName, std::string imgName);

//! Convert a python 9-element list into a ElRotation3D
ElRotation3D list2rot(std::vector<double> l);

//! Convert a quaternion into a ElRotation3D
ElRotation3D quaternion2rot(double a, double b, double c, double d);

//! Convert a ElRotation3D into a python 9-element list
std::vector<double> rot2list(ElRotation3D &r);

std::vector<std::string> getFileSet(std::string dir, std::string pattern);

//-------------------- classes MM a exporter ------------------

#ifdef SWIG

#include "general/CMake_defines.h"
#include "general/sys_dep.h"

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
         std::string StdExport2File(cInterfChantierNameManipulateur *,const std::string & aDirOri,const std::string & aNameIm);  // Test -> Ori-R

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
    #ifdef FORSWIG
    cNupletPtsHomologues(){}
    ~cNupletPtsHomologues(){}
    #endif
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
        #ifdef FORSWIG
        ElCplePtsHomologues(){}
        #endif

        const Pt2dr & P1() const ;
        Pt2dr & P1() ;

        const Pt2dr & P2() const ;
        Pt2dr & P2() ;


        // Box2D
         void SelfSwap(); // Intervertit les  2

      double Profondeur(const ElRotation3D & aR) const;

     private :

};

class cPackNupletsHom
{
     public :
         typedef std::list<cNupletPtsHomologues>   tCont;
         typedef tCont::iterator                  tIter;
         typedef tCont::const_iterator            tCstIter;
     cPackNupletsHom(int aDim);
        #ifdef FORSWIG
        tCont &getList(){return mCont;}
        #endif
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
         //Box2dr BoxP1() const;//unimplemented, crashes python
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

             //tPairPt  PMed() const;//unimplemented, crashes python
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


#endif

#endif //API_MM3D_H
