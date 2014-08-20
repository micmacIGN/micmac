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

#ifndef  _EXEMPLE_PHGR_FORMEL_H_
#define  _EXEMPLE_PHGR_FORMEL_H_

/*
   Ce fichier contient un (ou plusieurs) exemple d'utilisation
   des mecanismes de calcul formel d'eLiSe

   Le code cpp est dans phgr_cEqObsRotVect.cpp
*/


//  Permet de gerer une variable d'etat :
//     - un nom pour la generation de code
//     - une fonc num pour l'expression formelle
//     - une adresse pour faire le "binding"

class cVarEtat_PhgrF
{
     public :
         cVarEtat_PhgrF(const std::string &);
         Fonc_Num FN() const;
         void InitAdr(cElCompiledFonc & aFoncC);
         void SetEtat(const double &);
         
         void SetEtatSVP(const double &);
         void InitAdrSVP(cElCompiledFonc & aFoncC);

     private :
         std::string mName;
         double *          mAdr;
};


//
//  Permet de gerer les variables d'etat liees a un point 2 D
//  Essentiellement l'agglomeration commode de 2 cVarEtat_PhgrF

class cP2d_Etat_PhgrF
{
     public :
           cP2d_Etat_PhgrF (const std::string & aNamePt);
           Pt2d<Fonc_Num>  PtF() const;

           void InitAdr(cElCompiledFonc & aFoncC);
           void SetEtat(const Pt2dr &);

           void SetEtatSVP(const Pt2dr  &);
           void InitAdrSVP(cElCompiledFonc & aFoncC);
     private :

           cVarEtat_PhgrF   mVarX;
           cVarEtat_PhgrF   mVarY;

           Pt2d<Fonc_Num>      mPtF;

};



//  Permet de gerer les variables d'etat liees a un point 3 D
//  Essentiellement l'agglomeration commode de 3 cVarEtat_PhgrF

class cP3d_Etat_PhgrF
{
     public :
           cP3d_Etat_PhgrF (const std::string & aNamePt);
           Pt3d<Fonc_Num>  PtF() const;
           void InitAdr(cElCompiledFonc & aFoncC);
           void SetEtat(const Pt3dr &);
     private :

           cVarEtat_PhgrF   mVarX;
           cVarEtat_PhgrF   mVarY;
           cVarEtat_PhgrF   mVarZ;

           Pt3d<Fonc_Num>      mPtF;
};



class   cMatr_Etat_PhgrF
{
     public :
         cMatr_Etat_PhgrF(const std::string & aNamePt,int aTx,int aTy);
	 const ElMatrix<Fonc_Num>  & Mat() const;
	 void InitAdr(cElCompiledFonc & aFoncC);
	 void SetEtat(const ElMatrix<double> &);
     private :
          ElMatrix<Fonc_Num>          mMatF;
	  std::vector<cVarEtat_PhgrF> mVF;
};





//      Classe utilisee pour la comparaison de deux calibration
//      J'essaye de la commenter a fond afin que cela puisse servir
//   de "reference" sur l'ajout d'une nouvelle equation d'observation
//   formelle (c'est pour ca que j'ai decide de passer par ce mecanisme
//   general alors, qu'en l'occurrence il aurait ete plus simple de tout faire
//   "a la main").
// 
//   Les inconnues sont : une rotation vectorielle R
//   Les observation sont de la forme 
//             R N1 = N2  , ou N1 et N2 sont des vecteurs de norme 1
//   
//    La classe herite de :
//           cNameSpaceEqF  :      pour l'espace de noms
//           cObjFormel2Destroy  : pour automatiser la destruction
//   
//    Comme toute les classes de ce type, elle a un double role 
//          -  generation du code formel
//          -  utilisation de ce code 
//
//
//          Une fois le code genere, il faut encore
//          ajouter :
//
//             AddEntry("cEqObsRotVect_CodGen",cEqObsRotVect_CodGen::Alloc);
//             (dans phgr_or_code_gen00.cpp) pour creer le lien nom -allocateur
//             qui permettra AllocFromName
//
//
//   Voir aussi cL2EqObsBascult exemple_basculement.h pour l'utilisation
//   de variables et intervalles multiples
//

class cEqObsRotVect : public cNameSpaceEqF,
                      public cObjFormel2Destroy
{
        public :

            cRotationFormelle & RotF();
            virtual ~cEqObsRotVect();
	    // WithD2 = Derivee seconde
            void    AddObservation(Pt3dr aDir1,Pt3dr aDir2,double aPds=1.0,bool WithD2=false);
            // Idem prec, mais les param sont convertis en (x,y,1) pour en faire de Pt3d
            void    AddObservation(Pt2dr aDir1,Pt2dr aDir2,double aPds=1.0,bool WithD2=false);
        private :
            friend class cSetEqFormelles;
            void GenCode();


            cEqObsRotVect
            (
                 cSetEqFormelles &   aSet,
                 cRotationFormelle * aRot, // Si vaut 0, se l'alloue
                 bool                Cod2Gen
            );

            // Rotation utilisee pour le calcul
            cSetEqFormelles &   mSet;
            cRotationFormelle *  mRotCalc;
            cRotationFormelle *  mRot2Destr;
           

//     Dans  l'equation  R N1 = N2  
//           mN1 represente N1, mN2 N2 
//           et mRotCalc->mFMatr le R
//           mResidu represent R N1 - N2

            cP3d_Etat_PhgrF    mN1;
            cP3d_Etat_PhgrF    mN2;
            Pt3d<Fonc_Num>      mResidu;

       // Sert a definir le nom de la classe contenant les code
       // generes pour le calcul des derivees 
            std::string           mNameType;
       // Permet de faire le lien entre les numeros d'inconnues
       // de rotation et les valeur compactees 0,1,2 utilisees
       // en locale
          cIncListInterv        mLInterv;
       // Sert a calculer effectivement les valeurs et derivees
            cElCompiledFonc *     mFoncEqResidu;

};



//  Classe assez voisine de la precedente, mais + generale, utilise pour
//  faire de la "calibration-croisse". Les inconnues sont une 
//  Rotation R et un calibration intrineque. Les observations sont
//  des couple P1 - N2 ou P1 est un point image de la camera a
//  calibration inconnue et N2 est une direction de rayon de la camera
//  connu venant d'un point homologue a P1.L'equation a resoudre  est :
//  
//        C.CamToMonde(P1)  ~  R N2
//

class cEqCalibCroisee : public cNameSpaceEqF,
                        public cObjFormel2Destroy
{
        public :

            cRotationFormelle & RotF();
	    cParamIntrinsequeFormel & PIF();
            virtual ~cEqCalibCroisee();
	    // WithD2 = Derivee seconde, quasi obsolete (plus de derivee seconde calculee
	    // aujourd'hui dans les codes formels
            const std::vector<REAL> &     AddObservation
	         (Pt2dr aPIm1,Pt3dr aDir2,double aPds=1.0,bool WithD2=false);
        private :
            friend class cSetEqFormelles;
            void GenCode();


            cEqCalibCroisee
            (
	    // Le sens C2M est le seul manipule par eLiSe, cela dit cette classe pouvant etre
	    // utilisee pour faire des conversion, notamment vers les autres conventions, on 
	    // conserve la possibilite d'un M2C
	         bool SensC2M,
		 // Pas alloue car classe virtuelle, peut etre radiale, polyn etc...
		 cParamIntrinsequeFormel & aPIF,
                 cRotationFormelle * aRot, // Si vaut 0, se l'alloue
                 bool                Cod2Gen
            );

            // Rotation utilisee pour le calcul
            cSetEqFormelles &   mSet;
	    cParamIntrinsequeFormel & mPIF;
            cRotationFormelle *  mRotCalc;
            cRotationFormelle *  mRot2Destr;

            cP2d_Etat_PhgrF    mP1;
            cP3d_Etat_PhgrF    mN2;
            Pt3d<Fonc_Num>      mResidu;

       // Sert a definir le nom de la classe contenant les code
       // generes pour le calcul des derivees 
            std::string           mNameType;
       // Permet de faire le lien entre les numeros d'inconnues
       // de rotation et les valeur compactees 0,1,2 utilisees
       // en locale
          cIncListInterv        mLInterv;
       // Sert a calculer effectivement les valeurs et derivees
           cElCompiledFonc *     mFoncEqResidu;

};


//  !!!!  WAARRNING  !!!
//  C'etait une idee assez mauvaise de repasser par cParamIntrinsequeFormel pour
//  implemanter cEqDirecteDistorsion
//
//  En fait la formule utilisee n'a pas grand chose a voir avec l'utilisation
//  habituelle de la photogrametrie :
//
//      (aPIF.DistC2M(aP2) ) + (aP2-aPIF.FPP()).mul(aPIF.FFoc())
//
//          au lieu de
//        (aPIF.DistC2M(aP2)-aPIF.FPP()) / aPIF.FFoc()


//
//   Classe utilisee pour calculer directement  les parametres de distortion
//   a partir de relation entre couples de points. Pas tres utile en 
//   photogrametrie, mais permet des convertion a partir de grille,
//   eventuellement utilisable pour d'autre ajustement ?
//
//   Sera utilise pour ajuster un modele sur les appariement de Bayer


class cEqDirecteDistorsion : public cNameSpaceEqF,
                        public cObjFormel2Destroy
{
        public :

         // S'adapte aux differentes equations utilisees
            ElDistortion22_Gen * Dist(Pt2dr aTR0);

	    cParamIntrinsequeFormel & PIF();
            virtual ~cEqDirecteDistorsion();
	    // WithD2 = Derivee seconde, quasi obsolete (plus de derivee seconde calculee
	    // aujourd'hui dans les codes formels
            const std::vector<REAL> &     AddObservation
	         (Pt2dr aPIm1,Pt2dr aDir2,double aPds=1.0,bool WithD2=false);
        private :
            friend class cSetEqFormelles;
            void GenCode();


            cEqDirecteDistorsion
            (
		 // Pas alloue car classe virtuelle, peut etre radiale, polyn etc...
		 cParamIntrinsequeFormel & aPIF,
		 eTypeEqDisDirecre                Usage,
                 bool                Cod2Gen
            );

            // Rotation utilisee pour le calcul
	    eTypeEqDisDirecre   mUsage;
            cSetEqFormelles &   mSet;
	    cParamIntrinsequeFormel & mPIF;

            cP2d_Etat_PhgrF    mP1;
            cP2d_Etat_PhgrF    mP2;
            Pt2d<Fonc_Num>      mResidu;

       // Sert a definir le nom de la classe contenant les code
       // generes pour le calcul des derivees 
            std::string           mNameType;
       // Permet de faire le lien entre les numeros d'inconnues
       // de rotation et les valeur compactees 0,1,2 utilisees
       // en locale
          cIncListInterv        mLInterv;
       // Sert a calculer effectivement les valeurs et derivees
           cElCompiledFonc *     mFoncEqResidu;

};


/**************************************************************/
/*                                                            */
/*    Gestion d'un plan inconnu, pour rattacher un point      */
/*  a un plan                                                 */
/*                                                            */
/**************************************************************/

// Une cSurfInconnueFormelle
// c'est a la fois des inconnues (les parametres) et une equation
// (equation de la surface)
//
class cSurfInconnueFormelle  : public cNameSpaceEqF,
                               public cObjFormel2Destroy
{
     public :
     // La liste de intervalles contenant les parametres definissant la surface
          virtual cIncListInterv & IntervSomInc() = 0;

     // Renvoie +ou- la distance signee a la surface apres avoir
     // ajouter cette contrainte au syst formel avec le poids (si Pds > 0)
     // Pour l'instant on ne passe pas le point car les seuls 
     // utilisations semblent necessite que le point tempo ait deja ete
     // initialise
	  virtual double DoResiduPInc(double aPds) = 0;

      // Probablement on changera avec des surface plus compliquees
	  virtual Pt3dr InterSurfCur(const ElSeg3D &) const = 0;
     protected :
          cSurfInconnueFormelle(cSetEqFormelles & aSet);

	  cSetEqFormelles &  mSet;
	  cEqfP3dIncTmp *    mEqP3I;
     private :
};


class cEqPlanInconnuFormel : public cSurfInconnueFormelle
{
      public :
           friend class cSetEqFormelles;
	   const cElPlan3D  & PlanCur();

      private :

	  cIncListInterv & IntervSomInc();
	  double DoResiduPInc(double aPds);
	  Pt3dr InterSurfCur(const ElSeg3D &)  const;


          cEqPlanInconnuFormel
	  (

	      cTFI_Triangle * aTri,
	      bool            Code2DGen
          );
	  virtual ~cEqPlanInconnuFormel();
          void Update_0F2D();
	  void GenCode(const cMatr_Etat_PhgrF &);
	 
          cTFI_Triangle *    mTri;
	  cElPlan3D          mPlanCur;
          std::string           mNameType;
          cIncListInterv        mLInterv;
          cElCompiledFonc *     mFoncEqResidu;
};


/**************************************************************/
/*                                                            */
/*    Gestion des inconnues temporaires                       */
/*                                                            */
/**************************************************************/

/*
   cEqf1IncTmp : represente une inconnue temporaire (typiquement
    x,y ou z d'un point terrain)

   cEqfBlocIncTmp : represente un bloc d'inconnue teamporaire
   consecutive (typiquement : x,y et z d'un point terrain)

    cEqfP3dIncTmp : specialisation de cEqfBlocIncTmp au cas
    point terrain 3D;

   cSubstitueBlocIncTmp : represente les indexe d'une substitution
   (c.a.d les numero des variables a subsituer + les numeros
   des variable interferant avec elles)

   cBufSubstIncTmp : represente les structure necessaire a effectuer
   une substitution, tout les cSubstitueBlocIncTmp qui ont la
   meme taille (en general ils auront tous la meme taille , ou
   qq taille diff !) partagent le meme cBufSubstIncTmp.

*/

// class 

class cEqf1IncTmp
{
     public :
        friend class cEqfBlocIncTmp;
	void SetVal(const double & aVal);
	Fonc_Num F();
        double Val() const;
     private :
        cEqf1IncTmp
	(
	   cSetEqFormelles & aSet
	);

        cEqf1IncTmp(const cEqf1IncTmp &) ; // N.I.
	double   mVal;
	AllocateurDInconnues & mAlloc;
	int      mCurInc;
	Fonc_Num mF;
};

class cEqfBlocIncTmp  :  public cElemEqFormelle,
                         public cObjFormel2Destroy

{
     public :
	void  CloseEBI();
     protected :
	tContFcteur  FctrRap(const double *);
        cEqfBlocIncTmp
	(
	   cSetEqFormelles & aSet,
	   const std::string & aName,
	   int aNbInc,
           bool Tmp
	);


        std::vector<cEqf1IncTmp *>  mIncTmp;
        std::string                 mName;
};

class cEqfBlocIncNonTmp : public cEqfBlocIncTmp
{
    public :
        cEqfBlocIncNonTmp
	(
	   cSetEqFormelles & aSet,
	   const std::string & aName,
	   int aNbInc
        );
        Fonc_Num F(int aK);
        double   Val(int aK) const;
	void SetVal(int aK,const double & aVal);

/*
    protected  :
        cElCompiledFonc *  FoncRapAffine();
    private :
        cElCompiledFonc *  mFoncRapAffine;
*/
};



class cEqfP3dIncTmp  : public cEqfBlocIncTmp
{
     public :
        friend class cSetEqFormelles;
	Pt3d<Fonc_Num> PF();
	void InitVal(const Pt3dr & aP);

	tContFcteur  FctrRap(const Pt3dr &);

     private :
        cEqfP3dIncTmp ( cSetEqFormelles & aSet, const std::string & aName);
};


class cBufSubstIncTmp
{
      public :

	 double DoSubst
	      (
                  cSetEqFormelles * aSet,
                  const std::vector<cSsBloc> &  aSBlTmp,
                  const std::vector<cSsBloc> &  aSBlNonTmp,
                  const int                     aNbBloc,
                  // const std::vector<int> & aVIndTmp,
                  // const std::vector<int> & aVIndNonTmp,
                  bool  Raz=true // False pour des test de temps, histoire de relancer +sieur fois
	      );

          // Conditionnement du Lambda de l'inversion, on le teste comme un signal
          // d'alarme de pb
          static cBufSubstIncTmp * TheBuf();
          void RazNonTmp(cSetEqFormelles * aSet,const std::vector<cSsBloc> &  aSBlNonTmp);
      private  :

          
          void Resize(cSetEqFormelles * aSet,int aNbX, int aNbY);
          cBufSubstIncTmp(cSetEqFormelles * ,int aNbTmp,int aNbNonTmp);
          cSetEqFormelles *  mSet ;
          int mNbX;  // Tmp
          int mNbY;  // Non Tmp
	  // Les noms des matrices sont ceux utilis\'es dans
	  // la doc micmac
	  ElMatrix<tSysCho>   mA;
	  ElMatrix<tSysCho>   mB;
	  ElMatrix<tSysCho>   mBp;  // B', en general t B  ...
	  ElMatrix<tSysCho>   mBpL;  // mBp * mLambda-1
	  ElMatrix<tSysCho>   mLambda;  // B', en general t B  ...
};


class cSubstitueBlocIncTmp
{
      public :
         cSubstitueBlocIncTmp(cEqfBlocIncTmp &);
	 void AddInc(const cIncListInterv &);
	 void Close();
	 void DoSubst(bool Raz=true);

         void RazNonTmp();

         double  Cond() const;  // Du cBufSubstIncTmp
      private :

         std::vector<cSsBloc>    mVSBlTmp;
         std::vector<cSsBloc>    mSBlNonTmp;
         int                     mNbBloc;

         cEqfBlocIncTmp &   mBlocTmp;
         // std::vector<int>   mVIndTmp;
         // std::vector<int>   mVINonTmp;
         double             mCond;
};


// Doit permettre une manipulation generique, tout en
// facilitant l'usage standard uniquement en pt de liaison
// multiple.
//
//  Etapes initiales :
//       - allouer mP3Inc
//       - initialiser mSubst avec les intervales
//       - "closer" 
//
//  Etapes courantes :
//       - calculer le point initial;
//       - utiliser les fcteur
//       - faire la substitution


struct  cResiduP3Inc
{
    static const  double TheDefBSurH;  // B Sur H pas toujours calcule

    std::vector<Pt2dr> mEcIm;
    double             mEcSurf;
    Pt3dr              mPTer;
    double             mSomPondEr;
    bool               mOKRP3I;
    double             mBSurH;
    std::string        mMesPb;
};



// Classe permettant de parametrer l'equation d'observation standard (projection
// terrain->image) a partir d'un point "projectif" , c.a.d. se trouvant "proche
// de l'infini" dans la direction du faisceau; devrait permettre de resoudre les
// probleme de conditionnemment de matrice lorsque les faisceaux sont quasi paralleles

class cParamPtProj
{
    public :

       cParamPtProj(double SeuilBH,double aLimBsHRefut,bool Debug);

       double mBsH;
       double mEc2;
       bool   mDebug;
       double mSeuilBsH;
       double mSeuilBsHRefut;
       bool   mProjIsInit;

       // Le point terrain d'intersection de faisceau
       Pt3dr mTer;
 
// En projectif , les coordonnees (a,b,c) correspondent au point
// euclidien   mP0 + a mI + b mJ + mK/c 
// la valeur initiale est (0,0,m1sC)

       Pt3dr  mP0;
       Pt3dr  mI;
       Pt3dr  mJ;
       Pt3dr  mK;
       double mBase;
       double mHaut;
       
       bool   wDist;
       Pt2dr  mNDP0;
       Pt2dr  mNDdx;
       Pt2dr  mNDdy;
};

class cRapOnZ
{
    public :
       cRapOnZ(double aZ,double aIncertCompens,double aIncertEstim,const std::string & aLayerIm);

        Pt3dr PZ() const ; // X ET Y ARBRITRAIREMENT A 0
        double Z() const;
        double IncEstim() const;
        double IncComp() const;
        const std::string & LayerIm() const;
    private :
       double mZ;
       double mIC;
       double mIE;
       std::string mLayerIm;
};
//
class cManipPt3TerInc
{
    public :
        cManipPt3TerInc
	(
            cSetEqFormelles &              aSet,
	    cSurfInconnueFormelle *,             // Peut valoir 0 (souvent le cas)
	    std::vector<cCameraFormelle *> aVCamVis,
	    bool                           aClose = true
        );

        std::vector<CamStenope *> VCamCur();


	const cResiduP3Inc & UsePointLiaison
	                     (
                                  double aLimBsHProj,
                                  double aLimBsH,
			          double aPdsPl, // Poids de rattach a l'eventuelle surf
                                  const cNupletPtsHomologues & aNuple,
                                  std::vector<double> & aVPds,
                                  bool   AdEq , // Si false calcule les residu met ne modifie pas le syst
                                  const cRapOnZ *      aRAZ = 0
			     );

	const cResiduP3Inc & UsePointLiaisonWithConstr
	                     (
                                  double aLimBsHProj,
                                  double aLimBsH,
			          double aPdsPl,
                                  const cNupletPtsHomologues & aNuple,
                                  std::vector<double> & aVPds,
                                  bool   AdEq , // Si false calcule les residu met ne modifie pas le syst
				  const Pt3dr  & aPtApuis,
				  const Pt3dr  & anIncertApuis,
				  bool           aUseAppAsInit
			     );

        const std::vector<cCameraFormelle *> &   VCamVis() const;

  // Utilisation "standard", enchaine les 4 prec

         Pt3dr CalcPTerIFC_Robuste
         (
               double                       aDistPdsErr,
               const cNupletPtsHomologues & aNuple,
               const std::vector<double> &  aVPds
         );

         void SetTerrainInit(bool);
         void SetMulPdsGlob(double);



	const cResiduP3Inc & UsePointLiaisonGen
	                     (
                                  double aLimBsHProj,
                                  double aLimBsH,
			          double aPdsPl,
                                  const cNupletPtsHomologues & aNuple,
                                  std::vector<double> & aVPds,
                                  bool   AdEq , // Si false calcule les residu met ne modifie pas le syst
				  const Pt3dr  * aPtApuis,
				  const Pt3dr  * anIncertApuis,
				  bool           aUseAppAsInit,
                                  const cRapOnZ *      aRAZ
			     );
    private :
        cManipPt3TerInc(const  cManipPt3TerInc &); // N.I.

	Pt3dr  CalcPTerInterFaisceauCams
	       (
                   const cRapOnZ *      aRAZ,
                   bool                         CanUseProjectifP,
                   bool &                       OK,
		   const cNupletPtsHomologues & aNuple,
	           const std::vector<double> &,
                   cParamPtProj &            aParam,
                   std::vector<Pt3dr> *      aPAbs,
                   std::string *             mMes = 0
               );


        cSetEqFormelles &                mSet;
        cEqfP3dIncTmp  *                 mP3Inc;
	cSurfInconnueFormelle *          mEqSurf;
        std::vector<cCameraFormelle *>   mVCamVis;
	cResiduP3Inc                     mResidus;
        cSubstitueBlocIncTmp             mSubst;
        bool                             mTerIsInit;
        cParamPtProj                     mPPP;
        double                           mMulGlobPds;
};

Pt3dr CalcPTerIFC_Robuste
      (
           double                       aDistPdsErr,
           std::vector<CamStenope *>    aVCC,
           const cNupletPtsHomologues & aNuple,
           const std::vector<double> &  aVPds
      );


/****************************************************/
/*                                                  */
/*   Paquet de classe utilisees pour effectuer      */
/*   l'amelioration de l'orientation relative.      */
/*                                                  */
/****************************************************/


/*
class cAmeliorOrRel
{
    public :
    private :

       cSetEqFormelles        mSet;
       cRotationFormelle      mRotF;
       cSubstitueBlocIncTmp   mSubst;
};
*/

#endif //   _EXEMPLE_PHGR_FORMEL_H_


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
