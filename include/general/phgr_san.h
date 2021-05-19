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

#ifndef  _PHGR_SAN_H_
#define  _PHGR_SAN_H_

class cCylindreRevolution;
class cCylindreRevolFormel;
class cInterfSurfaceAnalytique ;
class cProjOrthoCylindrique;
class cProjTore;


  class cXmlToreRevol;
  class cXmlOrthoCyl;
  class cXmlCylindreRevolution;
  class cXmlDescriptionAnalytique;
  class cXmlModeleSurfaceComplexe;
  class cXmlOneSurfaceAnalytique;


// Decrit la facon dont une demi droite coupe une surface:

typedef enum
{
     eSurfPseudoInter, // Fausse intersection, par projection
     eSurfInterTgt,     // Intersection par tangence
     // Vrai Inter, Sotant-Rentrant, on est rentrant si passant en suivant
     // les abscisses positive (de la droite) Z decroit (donc pour une surface
     // avec interieur, comme le cylindre, on rentre dedans
     eSurfVI_Sort,
     eSurfVI_Rent,
} eTypeInterSurDemiDr;

class cInterSurfSegDroite
{
    public :
       double mLamba;
       eTypeInterSurDemiDr   mType;

       cInterSurfSegDroite(double,eTypeInterSurDemiDr);
};


         //============ GLOBAL =====================

const cXmlOneSurfaceAnalytique & SFromId
      (
          const cXmlModeleSurfaceComplexe & aModCompl,
          const std::string & anId
      );

cInterfSurfaceAnalytique * SFromFile
                           (
                                const std::string & aFile,
                                const std::string & anId,  // Nom
                                std::string  aTag="",  // Si defaut valeur xml
                                cXmlOneSurfaceAnalytique * aMemXML = 0
                           );


         //==========================================

class cParamISAPly
{
    public :
       cParamISAPly();

       double mSzRep;
       double mSzSphere;
       double mDensiteSurf;
};

class cPlyCloud
{
     public :
         void PutFile(const std::string & aName);

         typedef Pt3di tCol;
         std::vector<tCol>  mVCol;
         std::vector<Pt3dr>  mVPt;

         void AddSphere(const tCol& ,const Pt3dr & aC,const double & aRay,const int & aNbPerRay);
         void AddSeg(const tCol &,const Pt3dr & aP1,const Pt3dr & aP2,const int & aNbPerRay);
         void AddPt(const tCol &,const Pt3dr & aPt);
         void AddCercle(const tCol &,const Pt3dr & aC,const Pt3dr &aNorm,const double & aRay,const int & aNb);

         void AddCube(const tCol & aColP0,const tCol &aColP,const tCol & aSeg,const Pt3dr & aP1,const Pt3dr &P2,const double & aRay,const int & aNb);

         static const tCol Red;
         static const tCol Green;
         static const tCol Blue;
         static const tCol Yellow;
         static const tCol Cyan;
         static const tCol Magenta;
         static const tCol Black;
         static const tCol White;
         static tCol Gray(const double & aGr); // Entre 0 et 1
		 static tCol RandomColor();

         void PutDigit(char aDigit,Pt3dr aP0,Pt3dr aX,Pt3dr aY,tCol aCoul,double aLargCar,int aNbByCase);
         void PutStringDigit(std::string aDigit,Pt3dr aP0,Pt3dr aX,Pt3dr aY,tCol aCoul,double aLargCar,double aSpace,int aNbByCase);
         void PutString(std::string aDigit,Pt3dr aP0,Pt3dr aX,Pt3dr aY,tCol aCoul,double aLargCar,double aSpace,int aNbByCase,bool OnlyDigit=false);
};

class cInterfSurfaceAnalytique
{
    // UV coordonnee parametrique de la surface , L + ou -
    // troisieme coordonnee (genre faisceau de normal)
     public :


         void MakePly   (const cParamISAPly & , cPlyCloud & ,const std::vector<ElCamera *> &);
         // aProfMoy : Prof /10
         virtual void V_MakePly (const cParamISAPly & , cPlyCloud & ,const std::vector<ElCamera *> &,const Box2dr & aBox,const double aProfMoy);



         virtual double SeuilDistPbTopo() const;

         // renvoie une surface identite, utile pour beneficier
         // de certaine fonction MicMac passant par l'interface
         static cInterfSurfaceAnalytique * Identite(double aZ);
         static cInterfSurfaceAnalytique * FromCCC(const cChCoCart & );

         virtual Pt3dr E2UVL(const Pt3dr & aP) const = 0;
         virtual Pt3dr UVL2E(const Pt3dr & aP) const = 0;
         virtual cXmlDescriptionAnalytique Xml() const=0;
         virtual cXmlModeleSurfaceComplexe SimpleXml(const std::string &Id) const;


        virtual bool HasOrthoLoc() const = 0;  // Apparement identique en pratique a OrthoLocIsXCste ?
                                              // En theorie plus general, indique qu'il doit se desanamorphoser ...
        virtual Pt3dr ToOrLoc(const Pt3dr & aP) const ; // Def Err fatale
        virtual Pt3dr FromOrLoc(const Pt3dr & aP) const ; // Def Err fatale
        virtual bool OrthoLocIsXCste() const ; // Si vrai les ligne F(X,Y,Z0) = F(Y,Z0), la desanamorphose est automatique
        virtual bool IsAnamXCsteOfCart() const ; // Vrai pour Orthocyl faux pour les autres


        // Defaut return 0
         virtual cInterfSurfaceAnalytique * ChangeRepDictPts(const std::map<std::string,Pt3dr> &) const;

         virtual cInterfSurfaceAnalytique * DuplicateWithExter(bool IsExt) ;


         static cInterfSurfaceAnalytique * FromXml(const cXmlOneSurfaceAnalytique &);
         static cInterfSurfaceAnalytique * FromFile(const std::string &);

// Pour gerer d'eventuels pb de topologie, a la faculte de modifier la boite
         virtual void AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const = 0;

       // Renvoie, sous forme de leurs abscisses curvilignes
       //  la liste des intersections de la droite et de la nappe Z=Z0
       // si aucune "vraie" solution renvoie la droite des moindre carres et IsVraiSol = false
       // Peut etre un jour ecrire un valeur par defaut fonctionnant par dichotomie (sinon
       //  mettre virtuelle pure)


         virtual  std::vector<cInterSurfSegDroite>  InterDroite(const ElSeg3D &,double aZ0) const  = 0;

         // Rnvoie rei:q
         cTplValGesInit<Pt3dr> InterDemiDroiteVisible(const ElSeg3D &,double aZ0) const ;
         cTplValGesInit<Pt3dr> PImageToSurf0(const cCapture3D & aCap,const Pt2dr & aPIm) const; // Coord UVL

         // Si SurfExt, on selectionne les rayons rentantrant, coord UVL
         Pt3dr BestInterDemiDroiteVisible(const ElSeg3D &,double aZ0) const ;

         virtual ~cInterfSurfaceAnalytique();
         cInterfSurfaceAnalytique(bool isVueExt);
         bool VueDeLext() const; // Change le nom pour grep / mIsVueExt
         int SignDZSensRayCam()const;

        // Rappiecage pour pouvoir dynamiquement inhiber l'anamorphose verticale sans toucher au reste
         void SetUnusedAnamXCSte();
     protected :
         bool mUnUseAnamXCSte;
     private :
         cTplValGesInit<Pt3dr> InterDemiDroiteVisible(bool Force,const ElSeg3D &,double aZ0) const ;  // En UVL
         bool mIsVueExt;
};


// Dans SAN/cylindre.cpp


// Une cInterfSurfAn_Formelle est a la fois un allocateur
// d'inconnue (comme une rotation, ici les parametres de la surface)
// et une equation d'observation (comme  l'equation d'appuis, ici
// la projection d'un point 3d sur la surface).
//
// Rien n'empeche que d'autres equations soient utilisees sur
// une surface.

class cInterfSurfAn_Formelle : public cElemEqFormelle,
                               public cObjFormel2Destroy
{
     public :
          virtual cMultiContEQF StdContraintes() = 0;
          virtual const cInterfSurfaceAnalytique & CurSurf() const = 0;


          friend class cSetEqFormelles;
          double  AddObservRatt(const Pt3dr & aP, double aPds);

     private :


          void PostInit( bool Code2Gen);
          void PostInitEqRat(bool Code2Gen);
     protected :
          virtual Fonc_Num   EqRat() = 0;


          cInterfSurfAn_Formelle
          (
               cSetEqFormelles & aSet,
               const std::string & aNameSurf
          );

          std::string     mNameSurf;
          cSetEqFormelles & mSet;
          cP3d_Etat_PhgrF mP2Proj;

          std::string           mNameEqRat;
          cIncListInterv        mLIntervEqRat;
          cElCompiledFonc *     mFoncEqRat;
};


//  Cylindre de revolution

class cCylindreRevolution : public cInterfSurfaceAnalytique
{
      public :

        // UVL  = Teta *Ray,   Z   ,   R-R0
         virtual double SeuilDistPbTopo() const;

         friend class cProjTore;
     // aPOnCyl fixe a la fois le rayon et le premier axe
     // du plan Ortho, origine des angles
        cCylindreRevolution
        (
              bool  isVueExt,
              const ElSeg3D & aSeg,
              const Pt3dr & aPOnCyl
        );

         cInterfSurfaceAnalytique * ChangeRepDictPts(const std::map<std::string,Pt3dr> &) const;
         cCylindreRevolution *      CR_ChangeRepDictPts(const std::map<std::string,Pt3dr> &) const;

         cInterfSurfaceAnalytique * DuplicateWithExter(bool IsExt) ;
         cCylindreRevolution * CR_DuplicateWithExter(bool IsExt) ;

        static cCylindreRevolution WithRayFixed
                    (
                          bool  isVueExt,
                          const ElSeg3D & aSeg,
                          double    aRay,
                          const Pt3dr & aPOnCyl
                    );
         static cCylindreRevolution FromXml(
                                         const cXmlOneSurfaceAnalytique&,
                                         const cXmlCylindreRevolution&
                                    );

         bool HasOrthoLoc() const ;
         Pt3dr POnCylInit() const;
         Pt3dr E2UVL(const Pt3dr & aP) const;
         Pt3dr UVL2E(const Pt3dr & aP) const;
         cXmlDescriptionAnalytique Xml() const;
         cXmlCylindreRevolution XmlCyl() const;
         std::vector<cInterSurfSegDroite>  InterDroite(const ElSeg3D &,double aZ0) const ;
         void AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const ;

         const Pt3dr & P0() const;
         const Pt3dr & W() const;
         const Pt3dr & U() const;
         double  Ray() const;
         ElSeg3D Axe() const;

         Pt3dr  PluckerDir();
         Pt3dr  PluckerOrigine();
      private :
         void V_MakePly (const cParamISAPly & , cPlyCloud & ,const std::vector<ElCamera *> &,const Box2dr & aBox,const double aProfMoy);

         Pt3dr mP0; // Point sur l'axe
         double mRay;
         Pt3dr mU;  // vecteur du plan pointant sur P0
         Pt3dr mV;  //
         Pt3dr mW;  //   axe du cylinde
         int   mSign;
};

class cCylindreRevolFormel  : public cInterfSurfAn_Formelle
{
    public :
        friend class cSetEqFormelles;
        const cInterfSurfaceAnalytique & CurSurf() const;
        const cCylindreRevolution & CurCyl() const;
        virtual  void Update_0F2D ();
    private :
         cCylindreRevolFormel
         (
               const std::string & aName,
               cSetEqFormelles & mSet,
               const cCylindreRevolution &
         );

         cMultiContEQF StdContraintes();
         Fonc_Num   EqRat();

     // Coordonnee de Puckler initiale

        cCylindreRevolution  mCyl;
        cCylindreRevolution  mCurCyl;

        Pt3dr  mDirPlk0;
        Pt3dr  mOriPlk0;
        double mRay0;

        Pt3dr  mDirPlkCur;
        Pt3dr  mOriPlkCur;
        double mRayCur;

        int            mIndDir;
        Pt3d<Fonc_Num> mDirPlkF;
        int            mIndOri;
        Pt3d<Fonc_Num> mOriPlkF;
        int            mIndRay;
        Fonc_Num       mRayF;
        cElCompiledFonc *mFcteurNormDir;
        cElCompiledFonc *mFcteurOrthogDirOri;

        double    mTolFctrNorm;
        double    mTolFctrOrtho;
};

class cProjTore : public cInterfSurfaceAnalytique
{
     public :
        cProjTore(const cCylindreRevolution & aCyl,const Pt3dr & aPEuclDiamTor);
        virtual double SeuilDistPbTopo() const;
   //   Euclidien <=> Torique
        Pt3dr E2UVL(const Pt3dr & aP) const;
        Pt3dr UVL2E(const Pt3dr & aP) const;
   // Fonction virtuelle generale , cree un objet multi type
        cXmlDescriptionAnalytique Xml() const;
   // Fonction specifique
        cXmlToreRevol  XmlTore() const;
// En pratique identique a OrthoLocIsXCste
// En theorie plus general, indique qu'il doit se desanamorphoser ...
        bool HasOrthoLoc() const ;
        bool OrthoLocIsXCste() const ; // Si vrai les ligne F(X,Y,Z0) = F(Y,Z0), la desanamorphose est automatique

// Utilise dans la desanamorphose selon les ligne "verticale"  ,
        Pt3dr ToOrLoc(const Pt3dr & aP) const ; // Def Err fatale
        Pt3dr FromOrLoc(const Pt3dr & aP) const ; // Def Err fatale

        // Defaut return 0
         static cProjTore  FromXml(const cXmlOneSurfaceAnalytique &,const cXmlToreRevol &);
// Pour gerer d'eventuels pb de topologie, a la faculte de modifier la boite
         void AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const ;
         std::vector<cInterSurfSegDroite>  InterDroite(const ElSeg3D &,double aZ0) const ;

         cXmlModeleSurfaceComplexe SimpleXml(const std::string &Id) const;
         // X'  ,  Y'*D/(D-Z') , Z'
         // UVL         <----->             X'Y'Z'          <----->   XYZ
         // Torique                         Cylindrique     Abs
         //                                              => mRToE =>
      private :
         inline Pt3dr Cyl2Tore(const Pt3dr &) const;
         inline Pt3dr Tore2Cyl(const Pt3dr &) const;

         cCylindreRevolution    mCyl;
         Pt3dr                  mDiamEucl;
         Pt3dr                  mDiamCyl;
         bool                   mAngulCorr;
};

class cProjOrthoCylindrique : public cInterfSurfaceAnalytique
{
     public :


         // Pour de la generation d'otrtho anOri, anOx, anOy est le plan principal
         // de redressement
         cProjOrthoCylindrique
         (
               const cChCoCart & aL2A,
               const ElSeg3D & aSegAbs,
               bool    aAngulCorr
         );
         // Le parametrage de la droite dans le repere local est
         //  (X' , b Y' , D + c X')

         // Creation a partir des elements "naturels" le plan de projection et l'axe du cylindre;
         // le P0 de la droite projete sur le plan fixe l'origine; si prio au plan la droite est
         // modifiee pour etre // , et lycee de versailles

         Pt3dr E2UVL(const Pt3dr & aP) const;
         Pt3dr UVL2E(const Pt3dr & aP) const;
         cXmlDescriptionAnalytique Xml() const;
         cXmlOrthoCyl XmlOCyl() const;
         void AdaptBox(Pt2dr & aP0,Pt2dr & aP1) const ;
         std::vector<cInterSurfSegDroite>  InterDroite(const ElSeg3D &,double aZ0) const ;

         static cProjOrthoCylindrique FromXml(
                                         const cXmlOneSurfaceAnalytique&,
                                         const cXmlOrthoCyl&
                                    );
        bool OrthoLocIsXCste() const ;
        bool IsAnamXCsteOfCart() const ; // Vrai pour Orthocyl faux pour les autres
        bool HasOrthoLoc() const ;
        Pt3dr ToOrLoc(const Pt3dr & aP) const ; // Def Err fatale
        Pt3dr FromOrLoc(const Pt3dr & aP) const ; // Def Err fatale

     private :
         inline Pt3dr Loc2Abs(const Pt3dr &) const;
         inline Pt3dr Ab2Loc(const Pt3dr &) const;

         inline Pt3dr Cyl2Loc(const Pt3dr &) const;
         inline Pt3dr Loc2Cyl(const Pt3dr &) const;
         // X'  ,  Y'*D/(D-Z') , Z'
         // UVL         <----->             X'Y'Z'          <----->   XYZ
         // Cylindrique                     Local                     Abs
         //                                              => mRToE =>

         cChCoCart  mL2A;
         cChCoCart  mA2L;
         ElSeg3D    mSegAbs;
         double     mDist;
         double     mB;
         double     mC;
         bool       mAngulCorr;
};


#endif  // _PHGR_SAN_H_




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
