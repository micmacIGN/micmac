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



#ifndef _ELISE_GENERAL_OPTIM_H
#define _ELISE_GENERAL_OPTIM_H

extern bool DEBUG_LSQ;

class cOneEqCalcVarUnkEl
{
     public :
         cOneEqCalcVarUnkEl(double anO,double aPds) :
            mO   (anO*aPds),
            mPds (aPds),
            mRes (0)
         {
         }

         void Add(double anL,int anI)
         {
            mVL.push_back(anL*mPds);
            mVI.push_back(anI);
         }
         void SetResidu(double aRes) {mRes=aRes;}
     // private :

         std::vector<double>  mVL;  //  le Lk tLk de la doc
         std::vector<int>     mVI;  //  les indexes
         double               mO;   // le mOk  de la doc => 
         double               mPds;   // le mOk  de la doc => 
         double               mRes;
};

class cParamCalcVarUnkEl
{
    public :
       void NewEl(double anO,double aPds)
       {
           mVEq.push_back(cOneEqCalcVarUnkEl(anO,aPds));
       }
       void AddVal(double anL,int anI)
       {
           mVEq.back().Add(anL,anI);
       }
       void SetResidu(double aRes)
       {
          mVEq.back().SetResidu(aRes);
       }
       const std::vector<cOneEqCalcVarUnkEl> & VEq() const {return mVEq;}
    private :
       std::vector<cOneEqCalcVarUnkEl> mVEq;
};



class NROptF1vND;
class NROptF1vDer;

// NROptF1vND : Num Recipes Optimisation de Fonction d'1 var, Non Derivable 

class NROptF1vND
{
     public :
          virtual REAL NRF1v(REAL) = 0;
          virtual bool NROptF1vContinue() const;
          virtual ~NROptF1vND();
          NROptF1vND(int aNbIterMax=-1);


          //+++++++++++++++++++++++++++
          // Golden search for mimimum
          //+++++++++++++++++++++++++++

          REAL golden(REAL ax,REAL bx,REAL cx,REAL tol,REAL * xmin);

          //+++++++++++++++++++++++++++
          // Brent search for mimimum
          //+++++++++++++++++++++++++++

               // Appelle une eventuelle redifinition de Brent
               // (cas derivable), fait le bracketing initial
               // sur les valeur 0,1,2 
               Pt2dr  brent (bool ForBench=false);

          //++++++++++++++++++++++++++++++++++++++++++++++++
          // Van Wijngaarden-Deker-Brent search for root
          //++++++++++++++++++++++++++++++++++++++++++++++++
          // REAL zbrent(REAL ax,REAL bx,REAL tol,INT ITMAX=100);


                  // NR interface
          void mnbrack( REAL *ax,REAL *bx,REAL *cx,
                        REAL *fa,REAL * fb,REAL *fc
                      );
     protected :
         int mNbIter;
         int mNbIterMax;
         double  mTolGolden;
         double  TOL;
         double x0,x1,x2,x3;
     private :

          // precondition :
          // f(ax) > f(bx), f(cx) > f(bx),  bx entre ax et cx 
          virtual REAL PrivBrent
                  (
                    REAL ax,REAL bx,REAL cx,
                    REAL tol,
                    REAL * xmin,
                    INT ITMAX=100
               );

};

// NROptF1vDer : Num Recipes Optimisation de Fonction d'1 var, Derivable 

class NROptF1vDer : public NROptF1vND
{
     public :
          virtual REAL DerNRF1v(REAL) = 0;

          //+++++++++++++++++++++++++++
          // Brent search for minimum
          //+++++++++++++++++++++++++++


          //++++++++++++++++++++++++++++++++++++++++++++++++
          // Van Wijngaarden-Deker-Brent recherche de racines
          // en utilisant les derivees
          //++++++++++++++++++++++++++++++++++++++++++++++++
          REAL rtsafe(REAL ax,REAL bx,REAL tol,INT ITMAX=100);

     private :
          REAL PrivBrent  // Retourne la valeur de F au min
               (
                   REAL ax,REAL bx,REAL cx,
                   REAL tol,
                   REAL * xmin,  // retourne xmin
                   INT ITMAX=100
               );
};


template <class Type> class FoncNVarND
{
    public :

       FoncNVarND(INT NbVar);
       virtual ~FoncNVarND();

       virtual REAL ValFNV(const Type *  v) = 0;
       INT NbVar() const;
       INT powel(Type *,REAL ftol,INT ITMAX = 200);

    protected :

       const INT           _NbVar;

       inline REAL NRValFNV(const Type *);  // Just Recall ValFNV with NR convention
       void powel(Type *,REAL ftol,int *iter,REAL * fret,INT ITMAX = 200);
};

template <class Type> class  FoncNVarDer : public FoncNVarND<Type>
{
    public :
       virtual void GradFNV(Type *grad,const Type *   v) = 0;

       FoncNVarDer(INT NbVar);
       INT GradConj(Type *p,REAL ftol,INT ITMAX);


    protected :
       void NRGradFNV(const Type *,Type *);  // Just Recall ValFNV
       void GradConj(Type *p,REAL ftol,INT *iter,REAL *fret,INT ITMAX);

};



class GaussjPrec
{
      public :

          GaussjPrec(INT n,INT m);
          bool init_rec();
          void amelior_sol();
          REAL ecart() const;
          REAL ecart_inv() const;
          void set_size_nm(int n,int m);
          void set_size_m(int);

          ElMatrix<REAL> & M()     {return _M;}
          ElMatrix<REAL> & b()     {return _b;}
          ElMatrix<REAL> & Minv()  {return _Minv;}
          ElMatrix<REAL> & x ()    {return _x;}


          void SelfSetMatrixInverse(ElMatrix<REAL> & aM,INT aNbIter);

      private :

      // resoud _M * _x = _b
          void set_ecart();

          INT            _n;
          INT            _m;

          ElMatrix<REAL> _M;
          ElMatrix<REAL> _Minv;
          ElMatrix<REAL> _b;
          ElMatrix<REAL> _x;
          ElMatrix<REAL> _eps;
          ElMatrix<REAL> _ec;
};                

class AllocateurDInconnues;

class cStateAllocI
{
     public :
           friend class AllocateurDInconnues;
           cStateAllocI(const AllocateurDInconnues&);
           void ShowDiff(const cStateAllocI &) const;
     private  :
        const AllocateurDInconnues & mAlloc;
        std::vector<double>          mVals;
};


std::vector<std::string>   StdVectorOfName(const std::string & aPref,int aNb);



class AllocateurDInconnues
{
      public :
        void AssertUsable(const cStateAllocI &) const;
        void RestoreState(const cStateAllocI &);
        AllocateurDInconnues();
        Fonc_Num        NewF(const std::string & aNameBloc,const std::string & aNameInc,REAL *,bool HasAlwaysInitialValue=false);
        INT             NewInc(const std::string & aNameBloc,const std::string & aNameInc,REAL *);
        Pt3d<Fonc_Num>  NewPt3(const std::string & aNameBloc,REAL *,REAL*,REAL*,bool HasAlwaysInitialValue=false);
        Pt3d<Fonc_Num>            NewPt3(const std::string & aNameBloc,Pt3dr &,bool HasAlwaysInitialValue=false);
        Pt2d<Fonc_Num>            NewPt2(const std::string & aNameBloc,REAL*,REAL*,bool HasAlwaysInitialValue=false);
        Pt2d<Fonc_Num>            NewPt2(const std::string & aNameBloc,Pt2dr &,bool HasAlwaysInitialValue=false);
        Pt2d<Fonc_Num>            NewPt2(const std::string & aNameBloc,REAL*,REAL*,bool HasAlwaysInitialValue,const std::string& aNameX,const std::string & aNameY);

        std::vector<Fonc_Num>            NewVectInc(const std::string & aNameBloc,const std::vector<std::string> & aNameInc,std::vector<double> &);

        TplElRotation3D<Fonc_Num> NewRot(const std::string & aNameBloc,REAL *,REAL*,REAL*,REAL *,REAL*,REAL*);
        INT CurInc() const;

	PtsKD PInits();
	void SetVars(const REAL * aSol);
	double  GetVar(INT aK) const;
	double * GetAdrVar(INT aK);
	void  SetVar(double aVal,INT aK);
	void  SetVarPt(Pt2dr  aVal,INT aK);
	REAL * ValsVar();
        void Reinit(INT aK);

	const std::string &  NamesInc (int aK) const;
	const std::string &  NamesBlocInc (int aK) const;

      private :
	void PushVar(REAL *);
	std::vector<REAL *>  mAdrVar;
	std::vector<REAL  >  mValsVar;
	std::vector<std::string>  mVNamesInc;
	std::vector<std::string>  mVNamesBlocInc;



        INT GetNewInc();
        INT mIdInc;
        AllocateurDInconnues (const AllocateurDInconnues &);
        void operator = (const AllocateurDInconnues &);
};


// Classe pour gerer rapidement des ensemble entier;
// Permet de cumuler les avantages d'un "std::set" et 
// d'un "std::vector<bool> du point de vue ajout, consultation
// par contre, pas de suppression
// 
// On essaye d'avoir le max de compatibilite avec les set de 
// la stl.

class ElGrowingSetInd 
{
     public :
         // Partie typedef

            typedef INT key_type;
            typedef INT value_type;
            typedef std::vector<INT>::const_iterator const_iterator;

         // "Big Three"
            ElGrowingSetInd
            (
                 INT aCapa, 
                 REAL aRatioEncombr = 0.1 // Pour dimensionner mIndexes, Pas Fondamental.
            );
            ~ElGrowingSetInd();
            /// ElGrowingSetInd(const ElGrowingSetInd &); => en private, non implante

         // Pour parcourir un ElGrowingSetInd
            const_iterator begin() const  {return mIndexes.begin();}
            const_iterator end()   const  {return mIndexes.end();}



         // Partie set classique
            void clear();
            void insert(const INT&) ;
            int size() const;


         /*
             >,>=, == , != : peuvent etre fait rapidement
         */

     private :
         inline void AssertValideIndexe(INT anIndexe) const;
         inline bool PrivMember(INT anIndexe) const;
         inline void PrivSet(INT anIndexe,bool) ;
         ElGrowingSetInd(const ElGrowingSetInd &);  // Non implante

         INT         mCapa;
         std::vector<INT>  mIndexes;
         Im2D_Bits<1>      mBuzyIndexes;

         // void erase(const INT&);  a definir dans une classe derivee, "non growing"
};

class ElSignedGrowingSetInd 
{
    public :
    private :
       ElGrowingSetInd mSetPos;
       ElGrowingSetInd mSetNeg;
};

class cIncIntervale;
class cSsBloc;

class cElMatCreuseGen
{
      public :
// aSys.Indexee_EcrireDansMatrWithQuad
// aSys.SoutraitProduc3x3
// Indexee_QuadSet0
// V_GSSR_AddNewEquation_Indexe

        virtual void Test();

        // true si sait inverser non iterativement (cas cholesky),
       // Defaut false
        virtual bool DirectInverse(const tSysCho *,tSysCho *);

      //========= Optimisations possibles =======================

                 // Indique si l'opt est geree
          virtual bool IsOptForEcrireInMatr() const;
          virtual bool IsOptForSousP3x3() const;
          virtual bool IsOptForQuadSet0() const;
          virtual bool IsOptForAddEqIndexee() const;

                 // Optimise , defaut erreur
         virtual void Indexee_EcrireDansMatrWithQuad
	      (   ElMatrix<tSysCho> &aMatr,
                  const std::vector<cSsBloc> &  aVx,
                  const std::vector<cSsBloc> &  aVy
              )   const;

        virtual void SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     );
         virtual void Indexee_QuadSet0 (const std::vector<cSsBloc> & aVIndx,
	                                const std::vector<cSsBloc> & aVIndy);
         
	 virtual void VMAT_GSSR_AddNewEquation_Indexe
		      ( 
                        const std::vector<cSsBloc> * aVSB,
                        double *  FullCoeff,
                        int aNbTot,
			REAL aPds,tSysCho * aDataLin,REAL aB);

         //=====================================

         bool OptSym() const;
         virtual void Verif(const std::string & aMes) ;


	 static cElMatCreuseGen * StdBlocSym
                ( 
                      const  std::vector<cIncIntervale *> &  Blocs,
                      const  std::vector<int> &              I2Bloc
                );

	 static cElMatCreuseGen * StdNewOne(INT aNbCol,INT aNbLign,bool Fixe);
	 virtual ~cElMatCreuseGen();

         virtual void  MulVect(tSysCho * out,const tSysCho * in) const = 0;
         // virtual void  tMulVect(REAL * out,const REAL * in) const = 0;
         Im1D<tSysCho,tSysCho> MulVect(Im1D<tSysCho,tSysCho>) const;
         void  MulVect8(double * out,const double * in) ;

	 void AddElem(INT aX,INT aY,REAL);


	 virtual tSysCho   LowGetElem(INT aX,INT aY) const =0;
	 virtual void    LowSetElem(INT aX,INT aY,const tSysCho &) =0;



	 void LowAddElem(INT aX,INT aY,REAL) ;

	 virtual void Reset()= 0; // Remet tous les elements a 0

	 virtual void AddLineInd
		      (
		          INT aKY,
		          INT aY,
			  REAL aCyP,
			  const std::vector<INT> & aVInd,
			  REAL * aCoeff
		      );
	 virtual void SetOffsets(const std::vector<INT> & aVIndexes);
	 virtual void   EqMatIndexee
                        (
                           const std::vector<INT> & aVInd,
		           REAL aPds,REAL ** aMat
                        );
          virtual void PrecCondQuad(double *); // Def erreur

          virtual void PrepPreCond();
          virtual void  VPCDo(double * out,double * in);
      protected :
         cElMatCreuseGen(bool OptSym,INT aNbCol,INT aNbLign);
      // private :

         bool mOptSym;
         INT  mNbCol;
         INT  mNbLig;
	 Im1D_REAL8         mDiagPreCond;
         double *           mDDPrec;
};


/*
                          ----- cGenSysSurResol ----   ContraintesAssumed -
                         /          |             \
          cFormQuadCreuse     L2SysSurResol   SystLinSurResolu


         cGenSysSurResol   :  ContraintesAssumed, true

        SystLinSurResolu    :  solveur L1
                               ContraintesAssumed, false

        L2SysSurResol    :     solveur L2 , matrice pleine

        cFormQuadCreuse  :      solveur L2, matrice creuse !
*/



std::vector<cSsBloc> SomVSBl(const std::vector<cSsBloc> &,const std::vector<cSsBloc> &);

class cTestPbChol
{
     public :
       cTestPbChol(const std::string & aName);
       std::string mName;
       double  mMinVP;
       double  mMinSomVNeg;
};




class cGenSysSurResol 
{
     public :

           ElMatrix<tSysCho>  MatQuad() const;

          virtual double CoeffNorm() const;

//  FONCTION LIEES AU DEBUG DES  VALEUR <0 DANS CHOLESKY SUR PIAZZABRA
          void VerifGlob(const std::vector<cSsBloc> &,bool doCheck,bool doSVD,bool doV0);
          void BasicVerifMatPos(const std::vector<cSsBloc> &,int );

          void VerifMatPos(ElMatrix<tSysCho>,ElMatrix<tSysCho>  aLambda,cTestPbChol & aTPC,const std::vector<cSsBloc> &);
          void VerifMatPos(const ElMatrix<tSysCho> & ,const ElMatrix<tSysCho> & aLambda,cTestPbChol & aTPC,const std::vector<cSsBloc> &,const std::vector<cSsBloc> &);

 // Mode 0 =  Null ou non   *-
         void ShowGSR(int aMode);

         virtual void VerifGSS(const std::string & aMes) ;

         bool  OptSym() const;
         virtual void AddOneBloc(const cSsBloc &,const cSsBloc &, REAL aPds,REAL * aCoeff);
         virtual void AddOneBlocDiag(const cSsBloc &, REAL aPds,REAL * aCoeff);
         virtual void AddOneBlocCste(const cSsBloc &, REAL aPds,REAL * aCoeff,REAL aB);


        // void toto(const std::vector<cSsBloc>  &);

         virtual ~cGenSysSurResol();
         cGenSysSurResol(bool CstrAssumed,bool OptSym,bool GereNonSym,bool GereBloc);

         Im1D_REAL8  GSSR_Solve(bool * aResOk) ;
         void GSSR_Reset(bool WithCstr) ;

     //           aPds (aCoeff . X = aB) 
         void GSSR_AddNewEquation 
              (
                   REAL aPds,
                   REAL * aCoeff,
                   REAL aB,
                   double * aCoordCur  // Pour les contra univ, peut etre NULL
               );

     //  Pour resoudre, de maniere  simplifiee, une equation 
     //   en Ax et B de la forme
     //       Ax* Xi + B = Yi
     //   Typiquement pour fitter une droite
	void GSSR_Add_EqFitDroite(REAL aXi,REAL aYi,REAL aPds=1.0);
	void GSSR_SolveEqFitDroite(REAL & aAx,REAL &aB,bool * Ok=0);

     //  Pour resoudre, de maniere  simplifiee, une equation 
     //   en Ax et By et C de la forme
     //       Ax* Xi + By * Yi + C = Zi
     //   Typiquement pour fitter un plan
	void GSSR_Add_EqFitPlan(REAL aXi,REAL aYi,REAL aZi,REAL aPds=1.0);
	void GSSR_SolveEqFitPlan(REAL & aAx,REAL &aB,REAL & aC,bool * Ok=0);


     // Pour calculer des pseudo-intersection de droite ou de plan 3D
	 Pt3dr Pt3dSolInter(bool * Ok=0);


	 //  Ajoute une contrainte, sous la forme aC. X = aE 
	 //  sous laquelle sera resolu le systeme
	 //  L'ensemble des contrainte doit forme un systeme libre
         // void GSSR_AddContrainte (REAL * aC,REAL aE);
         void GSSR_AddContrainteIndexee (const std::vector<int> & aVI,REAL * aC,REAL aE);

	 virtual INT NbVar() const = 0;

	 // Renvoie true, deviendra virtuelle
	 virtual bool  	AcceptContrainteNonUniV() const;
         void TraitementContrainteUniVar(const std::vector<int> * aVA2S);


          bool IsCstrUniv(int anX,double & aVal);


// Manipulation  directe des matrices
         // aCste + aVect . X + 1/2 tX aMat X
	 virtual bool GSSR_UseEqMatIndexee();
	 void GSSR_EqMatIndexee
                      (
                           const std::vector<INT> & aVInd,
		           REAL aPds,REAL ** aMat,
			   REAL * aVect,REAL aCste
                      );



	 void GSSR_AddNewEquation_Indexe( const std::vector<cSsBloc> *aVSB,
                                          double *  FullCoeff,
                                          int aNbTot,
                                          const std::vector<INT> & aVInd ,
			                REAL aPds,REAL * aCoeff,REAL aB,
                                        cParamCalcVarUnkEl *);
         // GSSR_AddNewEquation_Indexe fait des pretraitement de prise en compte des contraintes
         // qu'on ne doit pas faire toujours
	 void Basic_GSSR_AddNewEquation_Indexe(
                                        const std::vector<cSsBloc> * aVSB,
                                        double *  FullCoeff,
                                        int aNbTot,
                                        const std::vector<INT> & aVInd ,
			                REAL aPds,REAL * aCoeff,REAL aB,cParamCalcVarUnkEl *);

         // Def = Erreur fatale, n'a pas de sens pout systeme L1
         virtual tSysCho   GetElemQuad(int i,int j) const;
         virtual void  SetElemQuad(int i,int j,const tSysCho& );
         virtual tSysCho  GetElemLin(int i) const;
         virtual void  SetElemLin(int i,const tSysCho& ) ;
         virtual tSysCho SomQuad() const;

         virtual bool    IsTmp(int aK) const;
         virtual void SetTmp(const std::vector<cSsBloc> &  aBlTmp,const std::vector<cSsBloc> &  aBlTNonmp,bool IsTmp);
         virtual int    NumTmp(int aK) const;
         virtual int    NumNonTmp(int aK) const;
         virtual int    InvNumNonTmp(int aK) const;

         virtual bool    IsCalculingVariance () const;
         virtual double    Redundancy () const;
         virtual void Show () const;
         virtual double    R2Pond () const;
         virtual bool    CanCalculVariance() const;
         virtual void SetCalculVariance(bool);
         virtual double  Variance(int aK);
         virtual double *  CoVariance(int aK1,int aK2);
         virtual bool  InverseIsComputedAfterSolve();
         virtual tSysCho   GetElemInverseQuad(int i,int j) const;
         virtual bool  ResiduIsComputedAfterSolve();
         virtual tSysCho   ResiduAfterSol() const;

          
         virtual void LVM_Mul(const tSysCho& aLambda) ;  // Levenberg Marquad modif
         virtual void LVM_Mul(const tSysCho& aLambda,int aK) ;  // Levenberg Marquad modif sur une seule inconnue

	 // Pour ces 4 Fon, Def, utilise GetElemQuad-GetElemLin

/*
         virtual void Indexee_EcrireDansMatrColWithLin
	      (ElMatrix<double> &aMatr,const std::vector<INT> & aVInd) const;
*/
         virtual void Indexee_EcrireDansMatrWithQuad
	      (  ElMatrix<tSysCho> &aMatr,
	         const std::vector<INT> & aVIndx,
		 const std::vector<INT> & aVIndy
              )   const;
         virtual void Indexee_LinSet0  (const std::vector<INT> & aVInd);
         virtual void Indexee_QuadSet0 (const std::vector<INT> & aVIndx,
	                                const std::vector<INT> & aVIndy);


         virtual void Indexee_EcrireDansMatrColWithLin
	      (ElMatrix<tSysCho> &aMatr,const std::vector<cSsBloc> &  aVx) const;
         virtual void Indexee_EcrireDansMatrWithQuad
	      (   ElMatrix<tSysCho> &aMatr,
                  const std::vector<cSsBloc> &  aVx,
                  const std::vector<cSsBloc> &  aVy
              )   const;

         virtual void Indexee_LinSet0  (const std::vector<cSsBloc> & aVInd);
         virtual void Indexee_QuadSet0 (const std::vector<cSsBloc> & aVIndx,
	                                const std::vector<cSsBloc> & aVIndy);




         virtual void Indexee_UpdateLinWithMatrCol
	      (const ElMatrix<tSysCho> &aMatr,const std::vector<INT> & aVInd);
         virtual void Indexee_UpdateQuadWithMatr
	      (  const ElMatrix<tSysCho> &aMatr,
	         const std::vector<INT> & aVIndx,
		 const std::vector<INT> & aVIndy
              )  ;

         virtual void Indexee_SoustraitMatrColInLin
	      (const ElMatrix<tSysCho> &aMatr,const std::vector<cSsBloc> & aVInd);
         virtual void Indexee_SoustraitMatrInQuad
	      (  const ElMatrix<tSysCho> &aMatr,
	         const std::vector<INT> & aVIndx,
		 const std::vector<INT> & aVIndy
              )  ;

        virtual void SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     );
         

         virtual void Indexee_QuadSetId (const std::vector<INT> & aVIndxy);

         void  SetPhaseEquation(const std::vector<int> *);

         virtual double  ResiduOfSol(const double *);
     protected :
	 virtual void V_GSSR_EqMatIndexee
                      (
                           const std::vector<INT> & aVInd,
		           REAL aPds,REAL ** aMat,
			   REAL * aVect,REAL aCste
                      );
	 // Def = erreur fatale
	 virtual void V_GSSR_AddNewEquation_Indexe
		      ( 
                        const std::vector<cSsBloc> * aVSB,
                        double *  FullCoeff,
                        int aNbTot,
                        const std::vector<INT> & aVInd ,
			REAL aPds,REAL * aCoeff,REAL aB, cParamCalcVarUnkEl *);
 
         virtual Im1D_REAL8  V_GSSR_Solve(bool * aResOk) = 0;
         virtual void V_GSSR_Reset() = 0;
         virtual void V_GSSR_AddNewEquation
		      (REAL aPds,REAL * aCoeff,REAL aB) = 0;


         bool  mCstrAssumed;
// Si mOptSym est true , c'est la partie "superieure" des matrice qui est remplie,

         bool  mOptSym;
         bool  mGereNonSym;
         bool  mGereBloc;
	 bool  mPhaseContrainte;
	 bool  mFirstEquation;

         void AssertPhaseContrainte();
         void AssertPhaseEquation();



	 INT   mNbContrainte;
	 INT   mLineCC;
//  Gestion des contraintes, ancienne mode
	 ElMatrix<REAL> mC;
	 ElMatrix<REAL> mE;
	 ElMatrix<REAL> mtL;
	 ElMatrix<REAL> mtLC;
	 ElMatrix<REAL> mtLCE;

	 ElMatrix<REAL> mSol;
	 ElMatrix<REAL> mCSol;

         GaussjPrec     mGP;
// Gestion des contraintes , nouvelle prise en compte specifique des contraintes
// univariees 
         bool        mNewCstrIsInit;
         bool        mNewCstrIsTraitee;
         bool        mUseSpeciCstrUniVar;  // a priori tjs true, false permet de revenir en arriere
         Im1D_REAL8  mValCstr;
         double *    mDValCstr;
         Im1D_U_INT1 mIsCstr;
         U_INT1 *    mDIsCstr;
};


class cVectMatMul
{
    public :
       virtual void VMMDo(Im1D_REAL8 in,Im1D_REAL8 out) = 0;
       virtual ~cVectMatMul();
};
// Meme classe a priori que cVectMatMul, mais comme les matrice doivent en heriter deux fois ....
class cVectPreCond
{
    public :
       virtual void VPCDo(Im1D_REAL8 in,Im1D_REAL8 out) = 0;
       virtual ~cVectPreCond();
};

struct cControleGC
{
     public :
          cControleGC(int aNbIterMax);

          const int mNbIterMax;
};

bool GradConjPrecondSolve
     (
            cVectMatMul&,
            cVectPreCond&,
            Im1D_REAL8  aImB,
            Im1D_REAL8  aImXSol,
            const cControleGC &
     );



class cFormQuadCreuse : public cVectMatMul,
                        public cVectPreCond,
                        public FoncNVarDer<REAL>,
                        public cGenSysSurResol
{
      // mVO + mFLin . X + 1/2 tX mMat X
      public :
          virtual double CoeffNorm() const;
         virtual double  ResiduOfSol(const double *);
         virtual void VerifGSS(const std::string & aMes) ;
         
	  bool  	AcceptContrainteNonUniV() const;
          cFormQuadCreuse(INT aNbVar,cElMatCreuseGen * aMatCr);
	  void AddDiff(Fonc_Num,const ElGrowingSetInd &);
	  void AddDiff(Fonc_Num);
	  virtual ~cFormQuadCreuse();

          virtual void GradFNV(REAL *grad,const REAL *   v) ;
          virtual REAL ValFNV(const REAL *  v) ;
 
	 void SetOffsets(const std::vector<INT> & aVIndexes);
	 bool GSSR_UseEqMatIndexee();
         // aCste + aVect . X + 1/2 tX aMat X
	 void V_GSSR_EqMatIndexee
                      (
                           const std::vector<INT> & aVInd,
		           REAL aPds,REAL ** aMat,
			   REAL * aVect,REAL aCste
                      );
         virtual tSysCho   GetElemQuad(int i,int j) const;
         virtual void  SetElemQuad(int i,int j,const tSysCho& );
         virtual tSysCho  GetElemLin(int i) const;
         virtual void  SetElemLin(int i,const tSysCho& ) ;


         virtual void SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     );
           
         virtual void Indexee_EcrireDansMatrWithQuad
                      (
                             ElMatrix<tSysCho> &aMatr,
                             const std::vector<cSsBloc> &  aVx,
                             const std::vector<cSsBloc> &  aVy
                      )  const;

         virtual void  Indexee_QuadSet0
                       (
                              const std::vector<cSsBloc> & aVx,
                              const std::vector<cSsBloc> & aVy
                       );



         
      private :
           void VMMDo(Im1D_REAL8 in,Im1D_REAL8 out);
           void VPCDo(Im1D_REAL8 in,Im1D_REAL8 out);



           // void SMFGC_Atsub(double *in,double *out,int) ;

	  virtual void V_GSSR_AddNewEquation_Indexe
		      (
                        const std::vector<cSsBloc> * aVSB,
                        double *  FullCoeff,
                        int aNbTot,
                        const std::vector<INT> & aVInd ,
			REAL aPds,REAL * aCoeff,REAL aB, cParamCalcVarUnkEl *);
	 virtual INT NbVar() const ;
         virtual Im1D_REAL8  V_GSSR_Solve(bool * aResOk) ;
         virtual void V_GSSR_Reset() ;
         virtual void V_GSSR_AddNewEquation
		      (REAL aPds,REAL * aCoeff,REAL aB) ;

	  INT                mNbVar;
	  REAL               mV0;
	  Im1D<tSysCho,tSysCho>  mFLin;
	  tSysCho *          mDataLin;
	  Im1D_REAL8         mVGrad;
	  REAL8 *            mDataGrad;
	  cElMatCreuseGen *  mMat;
	  ElGrowingSetInd *  mEGSI;
	  PtsKD           *  mP000;

          bool mMatIsOptForEcrireInMatr;
          bool mMatIsOptForSousP3x3;
          bool mMatIsOptForQuadSet0;
          bool mMatIsOptForAddEqIndexee;
};


/*
    SymBlocMatr :

     void cGenSysSurResol::Indexee_EcrireDansMatrWithQuad
     (
            ElMatrix<double> &aMatr,
            const std::vector<cSsBloc> &  aVx,
            const std::vector<cSsBloc> &  aVy
     )  const;


         virtual void Indexee_EcrireDansMatrColWithLin
	      (ElMatrix<double> &aMatr,const std::vector<cSsBloc> &  aVx) const;

         virtual void Indexee_LinSet0  (const std::vector<cSsBloc> & aVInd);
         virtual void Indexee_QuadSet0 (const std::vector<cSsBloc> & aVIndx,
	                                const std::vector<cSsBloc> & aVIndy);
        virtual void SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<double> &aM1,
                          ElMatrix<double> &aM2,
                          const std::vector<cSsBloc> * aYVSB,
                          const std::vector<INT> & aVIndy
                     );
         
*/


class L2SysSurResol : public cGenSysSurResol
{
     public :
         virtual double    Redundancy () const;
         virtual void Show () const;
         virtual double    R2Pond () const;
         virtual bool    IsCalculingVariance () const;
         virtual bool    CanCalculVariance() const;
         virtual void    SetCalculVariance(bool);
         virtual double  Variance(int aK);
         virtual double  * CoVariance(int aK1,int aK2);
         virtual bool    IsTmp(int aK) const;
         virtual int    NumTmp(int aK) const;
         virtual int    NumNonTmp(int aK) const;
         virtual int    InvNumNonTmp(int aK) const;
         virtual void    SetTmp(const std::vector<cSsBloc> &  aBlTmp,const std::vector<cSsBloc> &  aBlTNonmp,bool IsTmp);

         virtual bool  InverseIsComputedAfterSolve();
         virtual tSysCho   GetElemInverseQuad(int i,int j) const;
         virtual bool  ResiduIsComputedAfterSolve();
         virtual tSysCho   ResiduAfterSol() const;
         virtual double  ResiduOfSol(const double *);
         void  GSSR_Add_EqInterPlan3D(const Pt3dr& aDirOrtho,const Pt3dr& aP0,double aPds=1.0);
         void  GSSR_Add_EqInterDroite3D(const Pt3dr& aDirDroite,const Pt3dr& aP0,double aPds=1.0);

         void   GSSR_AddEquationFitOneVar(int aNumVar,double aVal,double aPds);
         void   GSSR_AddEquationPoint3D(const Pt3dr & aP,const Pt3dr &  anInc);

        virtual void SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<double> &aM1,
                          ElMatrix<double> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     );
         
         void Indexee_EcrireDansMatrWithQuad
	      (  ElMatrix<double> &aMatr,
                 const std::vector<cSsBloc> &  aVx,
                 const std::vector<cSsBloc> &  aVy
              )   const;

         void Indexee_EcrireDansMatrColWithLin
	      (ElMatrix<double> &aMatr,const std::vector<cSsBloc> &  aVx) const;

         void Indexee_LinSet0  (const std::vector<cSsBloc> & aVInd);
         void Indexee_QuadSet0 (const std::vector<cSsBloc> & aVIndx,
	                        const std::vector<cSsBloc> & aVIndy);

   //===================================================

         Im1D_REAL8  V_GSSR_Solve(bool * aResOk);
         void V_GSSR_Reset();
         void V_GSSR_AddNewEquation(REAL aPds,REAL * aCoeff,REAL aB);

         void AddTermLineaire(INT aK,REAL aVal);
         void AddTermQuad(INT aK1,INT aK2,REAL aVal);



        L2SysSurResol (INT aNbVar,bool IsSym=true);
        void SetSize(INT aNbVar);
        void GetMatr(ElMatrix<REAL> & M,ElMatrix<REAL> & tB);

     // Ajoute   :  
     //           aPds (aCoeff . X = aB) 

        void AddEquation(REAL aPds,REAL * aCoeff,REAL aB);
        void Reset();
        Im1D_REAL8  Solve(bool * aResOk);
        Pt3d<double>  Solve3x3Sym(bool * OK);

	INT NbVar() const;


	 virtual bool GSSR_UseEqMatIndexee();
         // aCste + aVect . X + 1/2 tX aMat X
	 virtual void V_GSSR_EqMatIndexee
                      (
                           const std::vector<INT> & aVInd,
		           REAL aPds,REAL ** aMat,
			   REAL * aVect,REAL aCste
                      );
         virtual tSysCho   GetElemQuad(int i,int j) const;
         virtual void  SetElemQuad(int i,int j,const tSysCho& );
         virtual tSysCho  GetElemLin(int i) const;
         virtual void  SetElemLin(int i,const tSysCho& ) ;

         Im2D_REAL8   tLi_Li(); // Sigma des trans(Li) Li
 

     private :
          void SetNum(INT4 * mDataInvNum,INT4 *  mDNumNonTmp,const std::vector<cSsBloc> &  aBlTmp,bool SetNum /* ou UnSet*/);


	  virtual void V_GSSR_AddNewEquation_Indexe
                      (  const std::vector<cSsBloc> * aVSB,
                        double *  FullCoeff,
                        int aNbTot,
		         const std::vector<INT> & aVInd ,
			REAL aPds,REAL * aCoeff,REAL aB,cParamCalcVarUnkEl *);

        INT          mNbVar;
	Im2D_REAL8   mtLi_Li; // Sigma des trans(Li) Li
        REAL8 **     mDatatLi_Li;
	Im2D_REAL8   mInvtLi_Li;    // Inverse Sigma des trans(Li) Li
        REAL8 **     mDataInvtLi_Li;
        Im1D_REAL8   mbi_Li;  // Sigma des bi * Li
        REAL8 *      mDatabi_Li;
        REAL8        mBibi;
        Im1D_REAL8   mSolL2;
        REAL8 *      mDataSolL2;
        INT          mNbEq; // Debug
        INT          mNbIncReel; // Ajoute celle qui sont eliminees
        double       mRedundancy;
        double       mMaxBibi; // Debug
        double       mResiduAfterSol;
        Im1D_INT4    mNumTmp;
        INT4 *       mDNumTmp;
        Im1D_INT4    mNumNonTmp;
        INT4 *       mDNumNonTmp;
        Im1D_INT4    mInvNumNonTmp;
        INT4 *       mInvDNumNonTmp;


        bool         mDoCalculVariance;
        Im2D_REAL8   mCoVariance;
        REAL8 **     mDCoVar;
        double       mVarCurResidu;
        double       mVarCurSomLjAp;
        double       mSomPds;
        double       mSomR2Pds;
};

// Classe Adaptee au contexte bcp d'equations, (relativement) peu de variable
// avec necessite de memoriser ttes les equation (parceque, par exemple)
// resolution L1-barrodale, our resolution par moindre carres ponderes.
//
// En fait, typiquement tout ce qui est estimateur robuste

class  SystLinSurResolu : public cGenSysSurResol  // Herite en tant que Solveur L1
{
	public :
               Im1D_REAL8  V_GSSR_Solve(bool * aResOk);
               void V_GSSR_Reset();
               void V_GSSR_AddNewEquation(REAL aPds,REAL * aCoeff,REAL aB);
	        INT NbVar() const;

               SystLinSurResolu(INT NbVar,INT NbEq);

                  void SetSize(INT NbVar,INT NbEq);
			void SetNbEquation(INT aNbEq);
			void SetNoEquation();


			void PushDifferentialEquation
			     (
                                   Fonc_Num      aFonc,
                                   const PtsKD & aPts,
                                   REAL          aPds = 1.0
                             );
			void PushEquation
			     (
                                   Im1D_REAL8    aFormLin,
				   REAL          aValue,
                                   REAL          aPds = 1.0
                             );
			void PopEquation();
			void PushEquation
			     (
                                   REAL8 *       aFormLin,
				   REAL          aValue,
                                   REAL          aPds = 1.0
                             );

			Im1D_REAL8  L1Solve();
			// Si Ok ==0, matrice sing => erreur fatale
			Im1D_REAL8  L2Solve(bool *Ok); 

			// Non Pondere, signe
			REAL Residu(Im1D_REAL8,INT iEq) const; 
			// Pondere :
			REAL L2SomResiduPond(Im1D_REAL8)const; 
			INT NbEq() const;

			REAL Pds(INT iEq) const;
			REAL CoefLin(INT iVar,INT iEq) const;
			REAL CoefCste(INT iEq) const;

			REAL Residu(const REAL *,INT iEq) const; 
		protected :
		private :

                        void AdjustSizeCapa();
			void BarrodaleSetSize();
			void L2SetSize();
			void AssertIndexEqValide(INT IndEq) const;
			void AssertIndexVarValide(INT IndEq) const;
			void AssertIndexGoodNbVar(INT aNbVar) const;


			INT          mNbVarCur;
			INT          mNbEqCur;
			INT          mNbVarCapa;
			INT          mNbEqCapa;

			Im2D_REAL8   mA;  // mA.data() [IEqu][Ivar]
			REAL8 **     mDataA;
			Im1D_REAL8   mB;
			REAL8 *      mDataB;
			Im1D_REAL8   mPds;
			REAL8 *      mDataPds;


			// variables tempo pour  L1 Barrodale
			Im1D_REAL8   mBarodA;
			REAL8 *      mDataBarodA;
			Im1D_REAL8   mBarodB;
			REAL8 *      mDataBarodB;
			Im1D_REAL8   mBarodSOL;
			REAL8 *      mDataBarodSOL;
			Im1D_REAL8   mBarodRESIDU;
			REAL8 *      mDataBarodRESIDU;
		
			// variables tempo pour  L2-Gaussj
			// Pour resoudre au moindre carre
			// Li X = bi

                         L2SysSurResol mL2;
		
/*
			 Im2D_REAL8   mtLi_Li; // Sigma des trans(Li) Li
                         REAL8 **     mDatatLi_Li;
			 Im1D_REAL8   mbi_Li;  // Sigma des bi * Li
                         REAL8 *      mDatabi_Li;
			 Im1D_REAL8   mSolL2;
                         REAL8 *      mDataSolL2;
*/

};


class cOptimSommeFormelle
{
     public :
	 cOptimSommeFormelle(INT aNbVar);
	 ~cOptimSommeFormelle();
	 void Add(Fonc_Num,bool CasSpecQuad = true);

	 INT GradConjMin(REAL *,REAL ftol,INT ITMAX);
	 INT GradConjMin(PtsKD & ,REAL ftol,INT ITMAX);
	 void Show() ; // Debug purpose
	 INT Dim() const;

         REAL ValFNV(const REAL *  v) ;
         void GradFNV(REAL *grad,const REAL *   v);

     private :
	 cOptimSommeFormelle(const cOptimSommeFormelle &); // Undef 

	 class cMin : public FoncNVarDer<REAL> 
	 {
              public :
                 cMin(cOptimSommeFormelle &);
              private :
                 REAL ValFNV(const REAL *  v) ;
                 void GradFNV(REAL *grad,const REAL *   v);

		 cOptimSommeFormelle & mOSF;
	 };
         friend class cMin;




	 void SetPts(const REAL *);
  
         INT                   mNbVar;
	 std::vector<Fonc_Num> mTabDP;
	 Fonc_Num              mSomme;
	 ElGrowingSetInd       mSetInd;
	 PtsKD *               mPts;

         cElMatCreuseGen*      mMatCr;
	 cFormQuadCreuse       mQuadPart;
};


class Optim_L1FormLin
{
     public :

        // Soit N le nombre de variable et M le nombre de contrainte
        // Flin de taille (N+1,M)


        Optim_L1FormLin (const ElMatrix<REAL> &Flin );

        ElMatrix<REAL> Solve();

        static void bench();
        REAL score(const ElMatrix<REAL> & M); // M : (1,N)


        ElMatrix<REAL> MpdSolve();
        ElMatrix<REAL> BarrodaleSolve();

     private :

       class  AbscD1
       {
           public :
               AbscD1(ElMatrix<REAL> & sc,INT ind);

               REAL _x0;
               REAL _pds;
               INT  _k;
              inline bool operator < (const AbscD1 &) const;
       };

        REAL EcartVar(INT v);
        INT RandF();
        bool get_sol_adm(ElFilo<INT> & SubSet);
        void BenchCombin(REAL val);
        REAL MinCombin();
        void MinCombin
             (
                   ElFilo<INT> & CurSubset, ElFilo<INT> & BestSet,
                   REAL & ScMin,INT NbVarPos,INT CurVarPos
             );

        REAL score(ElFilo<INT> & SubSet);
        REAL Kth_score(const ElMatrix<REAL> & M,INT k); // M : (N,1)

        bool ExploreChVARBov
             (
                ElFilo<INT> & SubSet,
                REAL        & sc_min,
                INT kv
             );
        bool ExploreChVAR
             (
                ElFilo<INT> & SubSet,
                REAL        & sc_min,
                INT kv
             );





        bool Sol(const  ElFilo<INT> & SubSet);

        INT _NbVar;
        INT _NbForm;
		INT _NbStep;

        ElMatrix<REAL> _Flin;

        GaussjPrec     _GP;
        ElMatrix<REAL> & _MGauss;
        ElMatrix<REAL> & _MVGauss;
        ElMatrix<REAL> & _Sol;                           

        ElMatrix<REAL> _SolDirRech;
        ElMatrix<REAL> _Scal1D;

        ElMatrix<REAL>    _BestSol;
        bool              _bench_comb_made;
        ElSTDNS vector<AbscD1>    _vad1;

        static Optim_L1FormLin RandOLF(INT NbVar,INT NbForm,INT Nb0 = 0);
        static void BenchRand(INT NbVar,INT NbForm,INT NbTest,bool Comb);

        static void BenchRandComb(INT NbVar,INT NbForm);
        static void BenchRandComb();



        // Pour le bench "dur" sur les minimum  locaux 

        void SubsetOfFlags(ElFilo<INT> & Subset,INT flag);

        void CombinConjMinLoc
             (
                ElFilo<REAL>&  dic,
                ElFilo<INT> &  Subset,
                ElFilo<INT> &  FlagPos,
                INT            FlagSubset,
                INT            NbVarPos,
                INT            CurVarPos
             );


        void show_flag(INT flag);

        REAL TestNeighConjMinLoc(INT FlagSubset,ElFilo<REAL>&  dic);

        void CombinConjMinLoc
             (
                ElFilo<REAL>&  dic,
                ElFilo<INT> &  Subset,
                ElFilo<INT> &  FlagPos
             );
        static void CombinConjMinLoc
                    (
                         INT N,
                         INT M,
                         ElFilo<REAL>&  dic,
                         ElFilo<INT> &  Subset,
                         ElFilo<INT> &  FlagPos,
						 INT            Nb0 = 0
                    );
        static void CombinConjMinLoc();

		void One_bench_craig();
		static  void bench_craig();
		static  void rand_bench_craig(INT N,INT M);
        
};                      

// Nunerics, roots of polyonme

REAL IRoots(REAL val,INT exp);

ElMatrix<REAL8> MatrFoncMeanSquare
                (
                     Flux_Pts       flux,
                     ElSTDNS list<Fonc_Num> Lfonc,
                     Fonc_Num       Obs,
                     Fonc_Num       Pds
                );

Fonc_Num ApproxFoncMeanSquare
         (
            Flux_Pts       flux,
            ElSTDNS list<Fonc_Num> Lfonc,
            Fonc_Num       Obs,
            Fonc_Num       Pds
         );

	
Fonc_Num SomPondFoncNum
         (
		   ElSTDNS    list<Fonc_Num> Lfonc,
			  ElMatrix<REAL8>
		 );
template <class Type> class cMSymCoffact3x3
{
public:
    Type a;
    Type e;
    Type i;
    Type b;
    Type c;
    Type f;


    Type mA;
    Type mE;
    Type mI;
    Type mB;
    Type mC;
    Type mF;
    Type mDet;

    cMSymCoffact3x3();
    cMSymCoffact3x3(Type ** aMat);
    void CoffSetInv(Type **);
    Pt3d<Type>  CoffVecInv(const Type *) const;
    Pt3d<Type>  CoffMul(const Type *) const;
    void FinishCoFact();
};

class cAMD_Interf
{
    public :
         cAMD_Interf(int aNumberInc);
         void AddArc(int aN1,int aN2,bool VerifDup=false);

     // Renvoie un vecteur qui indique le rang de chaque indexe
     //  si V[0]=3, 0 est le troisiem el (et non, 3 est le premier)
         std::vector<int> DoRank(bool Show=false) ;
    private :
         void VerifN(int aN) const;
         int                           mNb;
         std::vector<std::vector<int> > mV;

};


int amd_demo_1 (void);


/*   
    0 = (p0 x + p1 y + p2 z + p3) - I (p8 x + p9 y + p10 z + p11)
    0 = (p4 x + p5 y + p6 z + p7) - J (p8 x + p9 y + p10 z + p11)
*/


class cEq12Parametre
{
    public :
        cEq12Parametre();
        void AddObs(const Pt3dr & aPGround,const Pt2dr & aPPhgr,const double&  aPds);

        // Cam 2 Monde
        std::pair<ElMatrix<double>,Pt3dr> ComputeNonOrtho();

        // Intrinseques + extrinseques
        std::pair<ElMatrix<double>,ElRotation3D > ComputeOrtho(bool *Ok=0);

        static CamStenope * Camera11Param
                            (
                                const Pt2di&               aSzCam,
                                bool                       isFraserModel,
                                const std::vector<Pt3dr> & aVCPCur,
                                const std::vector<Pt2dr> & aVImCur,
                                double & Alti ,
                                double & Prof
                            );

        static CamStenope * RansacCamera11Param
                            (
                                const Pt2di&               aSzCam,
                                bool                       isFraserModel,
                                const std::vector<Pt3dr> & aVCPCur,
                                const std::vector<Pt2dr> & aVImCur,
                                double & Alti ,
                                double & Prof,
                                int    aNbTest,
                                double  aPropInlier,
                                int     aNbMaxTirage
                            );


    private :
        L2SysSurResol mSys;
        std::vector<Pt3dr>  mVPG;
        std::vector<Pt2dr>  mVPPhgr;
        std::vector<double> mVPds;

        void ComputeOneObs(const Pt3dr & aPGround,const Pt2dr & aPPhgr,const double&  aPds);

        // Indexe et valeur permettant de fixer l'arbitraire
        int    mIndFixArb;
        double mValueFixArb;
};

/*
class cOldBundleIterLin
{
    public :

       void AddObs(const Pt3dr & aQ1,const Pt3dr& aQ2,const double & aPds);
       ElRotation3D CurSol();
       double ErrMoy() const;

       cOldBundleIterLin(const ElRotation3D & aRot,const double & anErrStd);
       ElRotation3D  mRot;
       L2SysSurResol mSysLin5;
       ElMatrix<double> tR0;
       Pt3dr mB0;
       Pt3dr mC,mD;
       std::vector<double> mVRes;
       double              mSomErr;
       double              mSomPds;
       double              mErrStd;
       double              mLastPdsCalc;

};
*/



#endif //  _ELISE_GENERAL_OPTIM_H





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
