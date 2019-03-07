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

#ifndef  _PHGR_DIST_UNIF_H_
#define  _PHGR_DIST_UNIF_H_

//#define __DEBUG_EL_CAMERA

/*
   Ce fichier contient une (tentative de ?) implantation unifiee 
   des calibration internes qui fasse le lien entre : la distortion
   "concrete", la camera "concrete", la distosion "formelle" , le
   parametre intrinseque "formelle";

   L'objectif est que le rajout d'un nouveau modele soit le + simple
   possible, quitte a ce que sa manipulation soit un peu moins naturelle
   (variable pas numero ...)


   Fichier necessitant une intervention pour rajouter un modele :

    include/XML_GEN/ParamChantierPhotogram.xml : rajouter une valeur dans eModelesCalibUnif
    orilib.cpp : pour creer une camera a partir d'un xml
    phgr_ebner_brown_dist.cpp : le code de la distorsion et l'instanciation

*/

void InitVars(double * aVLoc,int aNbLoc,const std::vector<double> * aVInit,const std::string & aName) ;


class  cDist_Param_Unif_Gen;
class  cCamera_Param_Unif_Gen;
class  cPIF_Unif_Gen;

template <class TDistR,class TDistF,const int NbVar,const int NbState>  class  cDist_Param_Unif ;
template <class TDistR,class TDistF,const int NbVar,const int NbState>  class  cCamera_Param_Unif ;
template <class TDistR,class TDistF,const int NbVar,const int NbState>  class  cPIF_Unif ;


typedef enum
{
    eModeContDGPol,
    eModeContDGCDist,
    eModeContDGDRad,
    eModeContDGDCent
} eModeControleVarDGen;




class  cDist_Param_Unif_Gen : public ElDistortion22_Gen
{
    public :

        virtual ~ cDist_Param_Unif_Gen();

	virtual double & KVar(int aK) =0;
	virtual double & KState(int aK)=0;

	virtual const double &  KVar(int aK) const =0;
	virtual const double  & KState(int aK) const=0;

	virtual int   NbV() const =0;
	virtual int   NbS() const =0;
	virtual const std::string & NameType() const = 0;
	const Pt2dr  & SzIm() const;

	virtual int TypeModele() const = 0;  // En fait un enum
        virtual void  Diff(ElMatrix<REAL> &,Pt2dr) const ;


    protected :

	cDist_Param_Unif_Gen(const Pt2dr & aSzIm,CamStenope * aCam);
	Pt2dr   mSzIm;
        ElDistortion22_Gen * mPC;
        CamStenope *         mCam;
        
};



template <class TDistR,class TDistF,const int NbVar,const int NbState>  
    class  cDist_Param_Unif : public cDist_Param_Unif_Gen
{
    public :

        cCalibDistortion ToXmlStruct(const ElCamera *) const;
        bool  AcceptScaling() const;
        bool  AcceptTranslate() const;
        void V_SetScalingTranslate(const double & F,const Pt2dr & aPP);

         Pt2dr Direct(Pt2dr) const;
	 cDist_Param_Unif
	 (
	      Pt2dr aSzIm,
              CamStenope *                ,
	      const std::vector<double> * =0,
              const std::vector<double> * =0
              
	 );
	 virtual ~cDist_Param_Unif();

	 const std::string & NameType() const;
	 int TypeModele() const;  // En fait un enum
	 double & KVar(int aK);
	 double & KState(int aK);
	 const double & KVar(int aK) const;
	 const double & KState(int aK) const;
	 int   NbV() const ;
	 int   NbS() const ;
	 // const Pt2dr  & SzIm() const;
         // TDistR &  TDist();

    private  :
         static const std::string  TheName;
	 static const int          TheType;
         double  mVars[NbVar ? NbVar : 1];        // F..k les compilateur de m...e comme  WS qui ne supportent pas [0]
         double  mStates[NbState ? NbState : 1];
         // TDistR  mTDist;
         Pt2dr GuessInv(const Pt2dr &)  const;
         bool OwnInverse(Pt2dr &) const ;

          
};

class cCamera_Param_Unif_Gen :  public CamStenope
{
    public :
        cCamera_Param_Unif_Gen(bool isDistC2M,REAL Focale, Pt2dr PP,const tParamAFocal  &);
	~cCamera_Param_Unif_Gen();
	virtual cDist_Param_Unif_Gen & DistUnifGen() = 0;
	virtual const cDist_Param_Unif_Gen & DistUnifGen() const = 0;

         virtual bool IsFE() const  = 0;

	// Redef de CamStenope
	cParamIntrinsequeFormel * AllocParamInc(bool isDC2M,cSetEqFormelles &);

        // A definir dans  cCamera_Param_Unif
	virtual cPIF_Unif_Gen * PIF_Gen(bool isDistC2M,cSetEqFormelles &) = 0;
    private :
};


template <class TDistR,class TDistF,const int NbVar,const int NbState>  
    class  cCamera_Param_Unif : public cCamera_Param_Unif_Gen
{
     public :
          typedef cCamera_Param_Unif<TDistR,TDistF,NbVar,NbState>  tCam;
          typedef cDist_Param_Unif<TDistR,TDistF,NbVar,NbState>   tDist;
          typedef cPIF_Unif<TDistR,TDistF,NbVar,NbState>          tPIF;
          virtual bool CanExportDistAsGrid() const;


           bool IsFE()  const;
           cCamera_Param_Unif
	   (
	            bool isDistC2M,
	            REAL Focale, 
		    Pt2dr PP,
		    Pt2dr aSzIm,
                    const tParamAFocal  &,
		    const std::vector<double> * Params=0,
		    const std::vector<double> * State=0
           );  // Sens C2M, Vars et States a 0

		   
           ~cCamera_Param_Unif();

          // Version specifique, pour manipuler les param
          tDist &  DistUnif();
          const tDist &  DistUnif() const;

          // Version semi-specifique, 
	  cDist_Param_Unif_Gen & DistUnifGen() ;
	  const cDist_Param_Unif_Gen & DistUnifGen() const ;

          // Version generique pour manipulation par CamStenope
	   ElDistortion22_Gen   &  Dist();
	   const ElDistortion22_Gen   &  Dist() const;

	    cPIF_Unif_Gen * PIF_Gen(bool isDistC2M,cSetEqFormelles &);
	    tPIF *          PIF(bool isDistC2M,cSetEqFormelles &);
            ElDistortion22_Gen   *  DistPreCond() const ;
     private  :
            tDist   *mDist;

#ifdef __DEBUG_EL_CAMERA
	public:
		cCamera_Param_Unif( const cCamera_Param_Unif<TDistR,TDistF,NbVar,NbState> &i_b );
		cCamera_Param_Unif<TDistR,TDistF,NbVar,NbState> & operator =( const cCamera_Param_Unif<TDistR,TDistF,NbVar,NbState> &i_b );
#endif
};


class cPIF_Unif_Gen : public cParamIntrinsequeFormel
{
    public :
        ~cPIF_Unif_Gen();
        virtual void Inspect() = 0;
        virtual void FigeIfDegreSup(int aDegre,double aTol,eModeControleVarDGen) =0;
// Parfois, equation de type homographique, il peut etre prudent de figer tous les
// degres 1 quoiqu'il arrive par ailleurs:
        virtual void FigeD1_Ou_IfDegreSup(int aDegre,double aTol) =0;
    protected :
        cPIF_Unif_Gen(bool isDistC2M, cCamera_Param_Unif_Gen * aCam, cSetEqFormelles &);
};

template <class TDistR,class TDistF,const int NbVar,const int NbState>
    class  cPIF_Unif : public cPIF_Unif_Gen
{
     public :

         Fonc_Num  NormGradC2M(Pt2d<Fonc_Num> );
         bool UseSz() const;


          typedef cPIF_Unif<TDistR,TDistF,NbVar,NbState>  tPIF;
          typedef cCamera_Param_Unif<TDistR,TDistF,NbVar,NbState>  tCam;
          typedef cDist_Param_Unif<TDistR,TDistF,NbVar,NbState>  tDist;

	  virtual ~cPIF_Unif();
          virtual std::string  NameType() const;

	  virtual cMultiContEQF  StdContraintes();
          virtual  Pt2d<Fonc_Num> VirtualDist(Pt2d<Fonc_Num>,bool UsePC=true,int aKCam=0);


	  
	  virtual CamStenope * CurPIF(); ;
	  virtual CamStenope * DupCurPIF(); ;
	  tCam   CurPIFUnif();
	  tCam * newCurPIFUnif();
	  void    UpdateCurPIF();
	  void    NV_UpdateCurPIF();

	  virtual void InitStateOfFoncteur(cElCompiledFonc *,int aKCam);

	  static cPIF_Unif * Alloc(bool isDistC2M,tCam *,cSetEqFormelles &);

          // Il sont crees tous libre
	  void SetFigeKthParam(int aK,double aTol);
	  void SetFreeKthParam(int aK);

          // fige les degres > a aDegre ; -1 -> Fige ts le monde
          void Inspect() ;
	  void FigeIfDegreSup(int aDegre,double aTol,eModeControleVarDGen);
          void FigeD1_Ou_IfDegreSup(int aDegre,double aTol);

          bool IsDistFiged() const;

     private :

         void VerifIndexVar(int aK);

	 static const int mDegrePolyn[NbVar ? NbVar : 1];
	 static const std::string mNamePolyn[NbVar ? NbVar : 1];
         cPIF_Unif(bool isDistC2M,tCam *,cSetEqFormelles &);

         int     mIndInc0;
         tDist   mDistInit;
         tDist   mDistCur;
         Fonc_Num  mVars[NbVar  ? NbVar :1 ];
	 double    mTolCstr[NbVar ? NbVar :1];
	 bool             mVarIsFree[NbVar ? NbVar :1];

         Fonc_Num  mStates[2][NbState ? NbState : 1];
         std::string  mNameSt[2][NbState ? NbState : 1];

	 tCam *  mCurPIF;
         // TDistR  mTDist;
           // TDistR &  TDist();


	 // double           mContraintePol[NbVar];

};

// Permet de definir des comportement specifique


class cGeneratorElemStd
{
     public :
        static cCalibDistortion ToXmlStruct(const cDist_Param_Unif_Gen & aCam,const ElCamera *) ;
};

class cGeneratorElem
{
     public :
        static bool IsFE() ;
        static bool UseSz();
        static Fonc_Num  NormGradC2M(Pt2d<Fonc_Num> ,Fonc_Num *);
        static int  DegreOfPolyn(const int * aDegGlob,int aK,eModeControleVarDGen); // Par defaut utilise le mDegrePolyn
        static Pt2dr GuessInv(const Pt2dr &,const double * aVar,const double *aState) ;
        static void InitClass();
        static bool CanExportDistAsGrid() ;
        static ElDistortion22_Gen   *  DistPreCond(const double * aVar,const double *aState)  ;
         // static Pt2d<Fonc_Num> GuessInv(const  Pt2d<Fonc_Num> &,const Fonc_Num *,const Fonc_Num *) ;

       // static cDistPrecondRadial * Precond(const double * aVar,const double *aState)  ;
};


class cGeneratorDRadFraser : public cGeneratorElem
{
     public :
        static int  DegreOfPolyn(const int * aDegGlob,int aK,eModeControleVarDGen); // Par defaut utilise le mDegrePolyn
};

template <const int TheNbRad> class cGeneratorFour : public cGeneratorElem
{
     public :
        static int  DegreOfPolyn(const int * aDegGlob,int aK,eModeControleVarDGen); // Par defaut utilise le mDegrePolyn
};

class cGeneratorNoScaleTr
{
    public :
       static bool AcceptScalingTranslate();
       static void SetScalingTranslate(const double & F,const Pt2dr & aPP,double * State,double *Vars);
};

template <class Type> class cEbnersModel_Generator : public cGeneratorElem,
                                                     public cGeneratorElemStd,
                                                     public cGeneratorNoScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
	                
};




template <class Type> class cDCBrownModel_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorNoScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
	                
};

	
class cGeneratorState_FPP_ScaleTr
{
    public :
       static bool AcceptScalingTranslate();
       static void SetScalingTranslate(const double & F,const Pt2dr & aPP,double * State,double *Vars);
};


class cGeneratorState_DRadScaleTr
{
    public :
       // static cCalibDistortion ToXmlStruct(const cDist_Param_Unif_Gen & aCam,ElCamera *) ;
        static cCalibDistortion ToXmlStruct(const cDist_Param_Unif_Gen & aCam,const ElCamera *) ;
       static bool AcceptScalingTranslate();
       static void SetScalingTranslate(const double & F,const Pt2dr & aPP,double * State,double *Vars);
};

class cGeneratorState_FraserScaleTr 
{
    public :
       static cCalibDistortion ToXmlStruct(const cDist_Param_Unif_Gen & aCam,const ElCamera *) ;
       static bool AcceptScalingTranslate();
       static void SetScalingTranslate(const double & F,const Pt2dr & aPP,double * State,double *Vars);
};


// R3 R5 R7 R9 R11
template <class Type> class cDRadModel_Generator : public cGeneratorDRadFraser,
                                                     public cGeneratorState_DRadScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
	                

// R3 R5 R7 R9 R11 +  P1 P2 b1 b2
template <class Type> class cFraserModel_Generator : public cGeneratorDRadFraser,
                                                         public cGeneratorState_FraserScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
	                

template <class Type> class cDistRadFour7x2_Generator : public cGeneratorFour<3>,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistRadFour11x2_Generator : public cGeneratorFour<5>,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistRadFour15x2_Generator : public cGeneratorFour<7>,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};

template <class Type> class cDistRadFour19x2_Generator : public cGeneratorFour<9>,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};

template <class Type> class cDistRadFour19x4_Generator : public cGeneratorFour<9>,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistRadFour19x6_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};



template <class Type> class cDistGen_Deg0_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistGen_Deg1_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};


template <class Type> class cDistGen_Deg2_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};



template <class Type> class cDistGen_Deg3_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};

template <class Type> class cDistGen_Deg4_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistGen_Deg5_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistGen_Deg6_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};
template <class Type> class cDistGen_Deg7_Generator : public cGeneratorElem,
                                                      public cGeneratorElemStd,
                                                      public cGeneratorState_FPP_ScaleTr
{
     public :

         static Pt2d<Type>   DistElem(bool UsePC,const Pt2d<Type> &,const Type * Vars,const Type * States,const Type & aFoc,const Pt2d<Type> & aPP);
};




void TestFishEye();

template  <class Type> class cFE_Precond
{
    public : 
        typedef Type tVal;
};


template <class Type> class cFELinear_Precond : public cFE_Precond<Type>
{
    public :
        static Fonc_Num  NormGradC2M(Pt2d<Fonc_Num> ,Fonc_Num *);
        static Type  C2MRxSRx(const Type & );  // C2M(sqrt(V)) / sqrt(V)
        static Type  M2CRxSRx(const Type & );  // M2C(sqrt(V)) / sqrt(V)
        static Type  SqM2CRx(const Type & );  // M2C(srqt(V)) ^2 
        static ElDistortion22_Gen   *  DistPreCond(const double &   aVar,const Pt2dr & ) ;
};

template <class Type> class cFEEquiSolid_Precond : public cFE_Precond<Type>
{
    public :
        static Fonc_Num  NormGradC2M(Pt2d<Fonc_Num> ,Fonc_Num *);
        static Type  M2CRxSRx(const Type  &);
        static Type  C2MRxSRx(const Type & );  // M2C(sqrt(V)) / sqrt(V)
        static Type  SqM2CRx(const Type & );  // M2C(srqt(V)) ^2 
        static ElDistortion22_Gen   *  DistPreCond(const double &   aVar,const Pt2dr & ) ;
};

template <class Type> class cFEStereoGraphique_Precond : public cFE_Precond<Type>
{
    public :
        static Fonc_Num  NormGradC2M(Pt2d<Fonc_Num> ,Fonc_Num *);
        static Type  M2CRxSRx(const Type  &);
        static Type  C2MRxSRx(const Type & );  // M2C(sqrt(V)) / sqrt(V)
        static Type  SqM2CRx(const Type & );  // M2C(srqt(V)) ^2 
        static ElDistortion22_Gen   *  DistPreCond(const double &   aVar,const Pt2dr & ) ;
};





template <class TPreC,const int NbRad,const int NbDec,const int NbPolyn,const int NBV> 
     class cDistGen_FishEye_Generator  : public cGeneratorNoScaleTr
{
     public :
       static cCalibDistortion ToXmlStruct(const cDist_Param_Unif_Gen & aCam,const ElCamera *) ;
        static bool IsFE() ;
         static bool UseSz();
         static Fonc_Num  NormGradC2M(Pt2d<Fonc_Num> ,Fonc_Num *);

         typedef typename TPreC::tVal  tVal;
         static void InitClass();
         static bool CanExportDistAsGrid() ;

         static int  DegreOfPolyn(const int * ,int aK,eModeControleVarDGen); // utilise son propre degre
         static void IndexCDist(int & aKX,int & aKY);

         cDistGen_FishEye_Generator();

         static Pt2d<tVal>   DistElem(bool UsePC,const Pt2d<tVal> &,const tVal * Vars,const tVal * States,bool Test);
         static Pt2d<tVal>   DistElem(bool UsePC,const Pt2d<tVal> &,const tVal * Vars,const tVal * States,const tVal & aFoc,const Pt2d<tVal> & aPP);


         static Pt2d<tVal> GuessInv(const  Pt2d<tVal> &,const tVal * aVar,const tVal * aState) ;

        static ElDistortion22_Gen   *  DistPreCond(const double * aVar,const double *aState) ;
// Les degres sont recalcule a chaque fois, c'est un perte de temps mineure mais un facilite de coherence
         static int  mDegreRad[NBV ? NBV : 1];
         static int  mDegreDec[NBV ? NBV : 1];
         static int  mDegrePolyn[NBV ? NBV : 1];
         static bool  isInit;
       

         static int IndCX() ;  // 0 
         static int IndCY() ;  // 1
         static int D0Rad();  // 2
         static int D0Dec();  // 2 + mDegreRad
         static int D0Polyn();
};


// CDIST 2
// Coeff 10
// Dec   10
// Polyn 6 + 5 
// Contrainte Rot -3

/*
#define NbV522 



typedef  cDist_Param_Unif<cDistGen_FishEye_Generator<cFELinear_Precond<double>,5,2,2,NbV522>,cDistGen_FishEye_Generator<cFELinear_Precond<Fonc_Num>,5,2,2,NbV522>,NbV522,1> cDistLin_FishEye_5_2_2;
typedef  cCamera_Param_Unif<cDistGen_FishEye_Generator<cFELinear_Precond<double>,5,2,2,NbV522>,cDistGen_FishEye_Generator<cFELinear_Precond<Fonc_Num>,5,2,2,NbV522>,NbV522,1> cCamLin_FishEye_5_2_2;
typedef  cPIF_Unif<cDistGen_FishEye_Generator<cFELinear_Precond<double>,5,2,2,NbV522>,cDistGen_FishEye_Generator<cFELinear_Precond<Fonc_Num>,5,2,2,NbV522>,NbV522,1> cPIFLin_FishEye_5_2_2;
*/

// Nombre de variable calcule par TestFishEye

typedef  cDist_Param_Unif<cDistGen_FishEye_Generator<cFELinear_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFELinear_Precond<Fonc_Num>,10,5,5,50>,50,1> cDistLin_FishEye_10_5_5;
typedef  cCamera_Param_Unif<cDistGen_FishEye_Generator<cFELinear_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFELinear_Precond<Fonc_Num>,10,5,5,50>,50,1> cCamLin_FishEye_10_5_5;
typedef  cPIF_Unif<cDistGen_FishEye_Generator<cFELinear_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFELinear_Precond<Fonc_Num>,10,5,5,50>,50,1> cPIFLin_FishEye_10_5_5;


typedef  cDist_Param_Unif<cDistGen_FishEye_Generator<cFEEquiSolid_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFEEquiSolid_Precond<Fonc_Num>,10,5,5,50>,50,1> cDistEquiSol_FishEye_10_5_5;
typedef  cCamera_Param_Unif<cDistGen_FishEye_Generator<cFEEquiSolid_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFEEquiSolid_Precond<Fonc_Num>,10,5,5,50>,50,1> cCamEquiSol_FishEye_10_5_5;
typedef  cPIF_Unif<cDistGen_FishEye_Generator<cFEEquiSolid_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFEEquiSolid_Precond<Fonc_Num>,10,5,5,50>,50,1> cPIFEquiSol_FishEye_10_5_5;


typedef  cDist_Param_Unif<cDistGen_FishEye_Generator<cFEStereoGraphique_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFEStereoGraphique_Precond<Fonc_Num>,10,5,5,50>,50,1> cDistStereoGraphique_FishEye_10_5_5;
typedef  cCamera_Param_Unif<cDistGen_FishEye_Generator<cFEStereoGraphique_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFEStereoGraphique_Precond<Fonc_Num>,10,5,5,50>,50,1> cCamStereoGraphique_FishEye_10_5_5;
typedef  cPIF_Unif<cDistGen_FishEye_Generator<cFEStereoGraphique_Precond<double>,10,5,5,50>,cDistGen_FishEye_Generator<cFEStereoGraphique_Precond<Fonc_Num>,10,5,5,50>,50,1> cPIFStereoGraphique_FishEye_10_5_5;

// cFEStereoGraphique_Precond



typedef  cDist_Param_Unif<cEbnersModel_Generator<double>,cEbnersModel_Generator<Fonc_Num>,12,1> cDist_Ebner;
typedef  cCamera_Param_Unif<cEbnersModel_Generator<double>,cEbnersModel_Generator<Fonc_Num>,12,1> cCam_Ebner;
typedef  cPIF_Unif<cEbnersModel_Generator<double>,cEbnersModel_Generator<Fonc_Num>,12,1> cPIF_Ebner;


typedef  cDist_Param_Unif<cDCBrownModel_Generator<double>,cDCBrownModel_Generator<Fonc_Num>,14,1> cDist_DCBrown;
typedef  cCamera_Param_Unif<cDCBrownModel_Generator<double>,cDCBrownModel_Generator<Fonc_Num>,14,1> cCam_DCBrown;
typedef  cPIF_Unif<cDCBrownModel_Generator<double>,cDCBrownModel_Generator<Fonc_Num>,14,1> cPIF_DCBrown;


typedef  cDist_Param_Unif<cDRadModel_Generator<double>,cDRadModel_Generator<Fonc_Num>,5,1> cDist_DRad_PPaEqPPs;
typedef  cCamera_Param_Unif<cDRadModel_Generator<double>,cDRadModel_Generator<Fonc_Num>,5,1> cCam_DRad_PPaEqPPs;
typedef  cPIF_Unif<cDRadModel_Generator<double>,cDRadModel_Generator<Fonc_Num>,5,1> cPIF_DRad_PPaEqPPs;


typedef  cDist_Param_Unif<cFraserModel_Generator<double>,cFraserModel_Generator<Fonc_Num>,9,1> cDist_Fraser_PPaEqPPs;
typedef  cCamera_Param_Unif<cFraserModel_Generator<double>,cFraserModel_Generator<Fonc_Num>,9,1> cCam_Fraser_PPaEqPPs;
typedef  cPIF_Unif<cFraserModel_Generator<double>,cFraserModel_Generator<Fonc_Num>,9,1> cPIF_Fraser_PPaEqPPs;
// cDRadModel_Generator  cFraserModel_Generator

typedef  cDist_Param_Unif<cDistGen_Deg0_Generator<double>,cDistGen_Deg0_Generator<Fonc_Num>,0,3> cDist_Polyn0;
typedef  cCamera_Param_Unif<cDistGen_Deg0_Generator<double>,cDistGen_Deg0_Generator<Fonc_Num>,0,3> cCam_Polyn0;
typedef  cPIF_Unif<cDistGen_Deg0_Generator<double>,cDistGen_Deg0_Generator<Fonc_Num>,0,3> cPIF_Polyn0;

typedef  cDist_Param_Unif<cDistGen_Deg1_Generator<double>,cDistGen_Deg1_Generator<Fonc_Num>,2,3> cDist_Polyn1;
typedef  cCamera_Param_Unif<cDistGen_Deg1_Generator<double>,cDistGen_Deg1_Generator<Fonc_Num>,2,3> cCam_Polyn1;
typedef  cPIF_Unif<cDistGen_Deg1_Generator<double>,cDistGen_Deg1_Generator<Fonc_Num>,2,3> cPIF_Polyn1;

typedef  cDist_Param_Unif<cDistGen_Deg2_Generator<double>,cDistGen_Deg2_Generator<Fonc_Num>,6,3> cDist_Polyn2;
typedef  cCamera_Param_Unif<cDistGen_Deg2_Generator<double>,cDistGen_Deg2_Generator<Fonc_Num>,6,3> cCam_Polyn2;
typedef  cPIF_Unif<cDistGen_Deg2_Generator<double>,cDistGen_Deg2_Generator<Fonc_Num>,6,3> cPIF_Polyn2;


typedef  cDist_Param_Unif<cDistGen_Deg3_Generator<double>,cDistGen_Deg3_Generator<Fonc_Num>,14,3> cDist_Polyn3;
typedef  cCamera_Param_Unif<cDistGen_Deg3_Generator<double>,cDistGen_Deg3_Generator<Fonc_Num>,14,3> cCam_Polyn3;
typedef  cPIF_Unif<cDistGen_Deg3_Generator<double>,cDistGen_Deg3_Generator<Fonc_Num>,14,3> cPIF_Polyn3;

typedef  cDist_Param_Unif<cDistGen_Deg4_Generator<double>,cDistGen_Deg4_Generator<Fonc_Num>,24,3> cDist_Polyn4;
typedef  cCamera_Param_Unif<cDistGen_Deg4_Generator<double>,cDistGen_Deg4_Generator<Fonc_Num>,24,3> cCam_Polyn4;
typedef  cPIF_Unif<cDistGen_Deg4_Generator<double>,cDistGen_Deg4_Generator<Fonc_Num>,24,3> cPIF_Polyn4;

typedef  cDist_Param_Unif<cDistGen_Deg5_Generator<double>,cDistGen_Deg5_Generator<Fonc_Num>,36,3> cDist_Polyn5;
typedef  cCamera_Param_Unif<cDistGen_Deg5_Generator<double>,cDistGen_Deg5_Generator<Fonc_Num>,36,3> cCam_Polyn5;
typedef  cPIF_Unif<cDistGen_Deg5_Generator<double>,cDistGen_Deg5_Generator<Fonc_Num>,36,3> cPIF_Polyn5;


typedef  cDist_Param_Unif<cDistGen_Deg6_Generator<double>,cDistGen_Deg6_Generator<Fonc_Num>,50,3> cDist_Polyn6;
typedef  cCamera_Param_Unif<cDistGen_Deg6_Generator<double>,cDistGen_Deg6_Generator<Fonc_Num>,50,3> cCam_Polyn6;
typedef  cPIF_Unif<cDistGen_Deg6_Generator<double>,cDistGen_Deg6_Generator<Fonc_Num>,50,3> cPIF_Polyn6;


typedef  cDist_Param_Unif<cDistGen_Deg7_Generator<double>,cDistGen_Deg7_Generator<Fonc_Num>,66,3> cDist_Polyn7;
typedef  cCamera_Param_Unif<cDistGen_Deg7_Generator<double>,cDistGen_Deg7_Generator<Fonc_Num>,66,3> cCam_Polyn7;
typedef  cPIF_Unif<cDistGen_Deg7_Generator<double>,cDistGen_Deg7_Generator<Fonc_Num>,66,3> cPIF_Polyn7;



// 2 CDid 3 DRad + 6 Deg2
typedef  cDist_Param_Unif<cDistRadFour7x2_Generator<double>,cDistRadFour7x2_Generator<Fonc_Num>,11,3>   cDist_RadFour7x2;
typedef  cCamera_Param_Unif<cDistRadFour7x2_Generator<double>,cDistRadFour7x2_Generator<Fonc_Num>,11,3> cCam_RadFour7x2;
typedef  cPIF_Unif<cDistRadFour7x2_Generator<double>,cDistRadFour7x2_Generator<Fonc_Num>,11,3>          cPIF_RadFour7x2;


typedef  cDist_Param_Unif<cDistRadFour11x2_Generator<double>,cDistRadFour11x2_Generator<Fonc_Num>,13,3>   cDist_RadFour11x2;
typedef  cCamera_Param_Unif<cDistRadFour11x2_Generator<double>,cDistRadFour11x2_Generator<Fonc_Num>,13,3> cCam_RadFour11x2;
typedef  cPIF_Unif<cDistRadFour11x2_Generator<double>,cDistRadFour11x2_Generator<Fonc_Num>,13,3>          cPIF_RadFour11x2;

typedef  cDist_Param_Unif<cDistRadFour15x2_Generator<double>,cDistRadFour15x2_Generator<Fonc_Num>,15,3>   cDist_RadFour15x2;
typedef  cCamera_Param_Unif<cDistRadFour15x2_Generator<double>,cDistRadFour15x2_Generator<Fonc_Num>,15,3> cCam_RadFour15x2;
typedef  cPIF_Unif<cDistRadFour15x2_Generator<double>,cDistRadFour15x2_Generator<Fonc_Num>,15,3>          cPIF_RadFour15x2;

typedef  cDist_Param_Unif<cDistRadFour19x2_Generator<double>,cDistRadFour19x2_Generator<Fonc_Num>,17,3>   cDist_RadFour19x2;
typedef  cCamera_Param_Unif<cDistRadFour19x2_Generator<double>,cDistRadFour19x2_Generator<Fonc_Num>,17,3> cCam_RadFour19x2;
typedef  cPIF_Unif<cDistRadFour19x2_Generator<double>,cDistRadFour19x2_Generator<Fonc_Num>,17,3>          cPIF_RadFour19x2;


// Methodes deplacees dans le header suite a des erreurs de compilation sous MacOS entre gcc4.0 et gcc4.2 (LLVM)
// du type : error: explicit specialization of 'TheType' after instantiation

template <class TDistR,class TDistF,const int NbVar,const int NbState>
int  cDist_Param_Unif<TDistR,TDistF,NbVar,NbState>::TypeModele() const
{
    return TheType;
}

template <class TDistR,class TDistF,const int NbVar,const int NbState>
cDist_Param_Unif<TDistR,TDistF,NbVar,NbState>::cDist_Param_Unif
(
 Pt2dr                        aSzIm,
 CamStenope *                 aCam,
 const std::vector<double> *  aVParam,
 const std::vector<double> *  aVEtats
 ) :
cDist_Param_Unif_Gen  (aSzIm,aCam)
{
    InitVars(mVars,NbVar,aVParam,TheName);
    InitVars(mStates,NbState,aVEtats,TheName);
    
    TDistR::InitClass();
    
    mPC = TDistR::DistPreCond(mVars,mStates);
    if (mPC)
    {
        SetParamConvInvDiff(100,1e-3);
    }
    
}

template <class TDistR,class TDistF,const int NbVar,const int NbState>
const std::string &   cDist_Param_Unif<TDistR,TDistF,NbVar,NbState>:: NameType() const
{
    return TheName;
}

template <class TDistR,class TDistF,const int NbVar,const int NbState>
void  cPIF_Unif<TDistR,TDistF,NbVar,NbState>::SetFigeKthParam(int aK,double aTol)
{
    
    VerifIndexVar(aK);
    double aDiag = euclid(mDistInit.SzIm()) / 2.0;
    
    aTol =  (aTol<=0)                       ?
    cContrainteEQF::theContrStricte :
    aTol / pow(aDiag,mDegrePolyn[aK]);
    mVarIsFree[aK] = false;
    mTolCstr[aK] = aTol;
}
template <class TDistR,class TDistF,const int NbVar,const int NbState>
void  cPIF_Unif<TDistR,TDistF,NbVar,NbState>::SetFreeKthParam(int aK)
{
    VerifIndexVar(aK);
    mVarIsFree[aK] = true;
}


template <class TDistR,class TDistF,const int NbVar,const int NbState>
void  cPIF_Unif<TDistR,TDistF,NbVar,NbState>::FigeIfDegreSup(int aDegre,double aTol,eModeControleVarDGen aModeControl)
{
    for (int aKV=0 ; aKV<NbVar ; aKV++)
    {
        int aDegK = TDistR::DegreOfPolyn(mDegrePolyn,aKV,aModeControl);
        if (aDegK >=0)
        {
            if (aDegK > aDegre)
                SetFigeKthParam(aKV,aTol);
            else
                SetFreeKthParam(aKV);
        }
    }
}

template <class TDistR,class TDistF,const int NbVar,const int NbState>
void  cPIF_Unif<TDistR,TDistF,NbVar,NbState>::Inspect()
{
    for (int aKV=0 ; aKV<NbVar ; aKV++)
    {
        std::cout << " FIDS  " <<  mDistCur.NameType() << " " << mDistCur.KVar(aKV) << " " << mVarIsFree[aKV] << "\n";
    }
}

template <class TDistR,class TDistF,const int NbVar,const int NbState>
void  cPIF_Unif<TDistR,TDistF,NbVar,NbState>::FigeD1_Ou_IfDegreSup(int aDegre,double aTol)
{
    for (int aKV=0 ; aKV<NbVar ; aKV++)
    {
        int aDegK = TDistR::DegreOfPolyn(mDegrePolyn,aKV,eModeContDGPol);
        if (aDegK >=0)
        {
            if (( aDegK> aDegre) ||  (aDegK==1))
                SetFigeKthParam(aKV,aTol);
            else
                SetFreeKthParam(aKV);
        }
    }
}



#endif  // _PHGR_DIST_UNIF_H_

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
