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

#ifndef _PHGR_OREL_H_
#define  _PHGR_OREL_H_


class cIncEnsembleCamera;
class cIncSetLiaison;


class cIncParamCamera;
class cIncParamExtrinseque;
class cIncParamIntrinseque;
class cIncCpleCamera;

class cFonctrPond
{
     public :
         cFonctrPond(cElCompiledFonc * aFctr,REAL aPds);
         REAL                 mPds;
         cElCompiledFonc *    mFctr;
};



class cIncParamIntrinseque
{
     public :

        friend class cIncCpleCamera; // Pour GenerateAllCodeGen
        friend class cIncEnsembleCamera; // Pour GenerateAllCodeGen


    // Transforme un point (formel) en pixel (== dist non) , en une direction (formelle), 
    // dans le repere camera

         Pt3d<Fonc_Num>  DirRayon(Pt2d<Fonc_Num> aPCam,INT aNumParam);

         virtual ~cIncParamIntrinseque();

         bool operator == (const cIncParamIntrinseque &) const;  // Fait comparaison physique


         const  cIncIntervale & IntervInc() const;
         cIncIntervale & IntervInc() ;

         std::string   NameType();


         static cIncParamIntrinseque  * NewOneNoDist
         (
             AllocateurDInconnues  &     anAlloc,
             REAL                        aFocale,
             bool                        isFocaleFree,
             Pt2dr                       aPP,
             bool                        isPPFree,
             cIncEnsembleCamera          * = 0,
             ElDistRadiale_PolynImpair * aDist = 0
         );

	 void InitFoncteur(cElCompiledFonc &,INT aNum);
         AllocateurDInconnues & Alloc();

         cIncEnsembleCamera * Ensemble();
	 REAL   CurFocale() const;
	 Pt2dr  CurPP()     const;

         ElDistRadiale_PolynImpair   CurDR() const;
         ElDistRadiale_PolynImpair   DRInit() const;

         std::vector<cElCompiledFonc *> &  FoncteursAuxiliaires ();
         std::vector<cElCompiledFonc *> &  FoncteurRappelCentreDist ();
         std::vector<cElCompiledFonc *> &  FoncteurRappelCoeffDist ();
         cElCompiledFonc *                 FoncteurRappelFocal();

         void SetRappelCrd(Pt2dr aC);
         void CurSetRappelCrd();
         void InitSetRappelCrd();

         void SetRappelFocal(REAL);
         void CurSetRappelFocal();
         void InitSetRappelFocal();

         void SetRappelCoeffDist(INT aK,REAL);
         void CurSetRappelCoeffDist(INT aK);
         void InitSetRappelCoeffDist(INT aK);

     protected :

         virtual Pt2d<Fonc_Num>   DistInv  (Pt2d<Fonc_Num>) ;
         void EndInitIPI();

      private : 

         cIncParamIntrinseque 
         (
             AllocateurDInconnues  &     anAlloc,
             REAL                        aFocale,
             bool                        isFocaleFree,
             Pt2dr                       aPP,
             bool                        isPPFree,
             cIncEnsembleCamera          * = 0,
	     ElDistRadiale_PolynImpair * = 0
         );

          class  cNumVarLoc
          {
               public :
                    void Init(cIncParamIntrinseque & aParam,std::string aNum);

                    std::string            mNum;
                    std::string            mNameFocale;
                    std::string            mNamePPx;
                    std::string            mNamePPy;
                    Fonc_Num               fFocale;
                    Pt2d<Fonc_Num>         fPP;
          };
          friend class cNumVarLoc;

          AllocateurDInconnues & mAlloc;
          cIncIntervale          mIncInterv;
          cIncEnsembleCamera *   mpEns;

          std::string            mNameFocale;
          std::string            mNamePPx;
          std::string            mNamePPy;

          bool                   mIsFFree;
          REAL                   mFocale;
          REAL                   mFocaleInit;
          bool                   mIsPPFree;
          Pt2dr                  mPP;
          INT                    mIndFoc;
          Fonc_Num               fFocale;
          Pt2d<Fonc_Num>         fPP;
          cNumVarLoc             mNVL[2];
	  bool                        mWithDR;
          ElDistRadiale_PolynImpair   mDR;
          ElDistRadiale_PolynImpair   mDRInit;
          INT                         mIndCrd;
          Pt2d<Fonc_Num>              mCentreDR;
	  std::vector<Fonc_Num>       mCoeffDR;

          std::vector<cElCompiledFonc *>   mFoncRCD;
          std::vector<double *>            mFRCDAdr;

          std::vector<cElCompiledFonc *>   mFoncRCoeffD;
          std::vector<double *>            mFRCoeffDAdr;

         cElCompiledFonc *                 mFoncRFoc;
         double *                          mFRadrFoc;

         std::vector<cElCompiledFonc *>   mFoncsAux;
      // Classe permettant de gerer la numerotation des variables locales


      // Prohibited
          void operator = (const cIncParamIntrinseque &);
          cIncParamIntrinseque(const cIncParamIntrinseque &);

         
};




class cIncParamExtrinseque
{
     public :

          typedef enum {ePosFixe,ePosBaseUnite,ePosLibre} tPosition;


       // Omega() * P + Tr() : transforme un point (formel) des coordonnees camera
       //  en un point formel coord monde
         virtual ElMatrix<Fonc_Num> & Omega(INT aNum) =0 ;
         virtual Pt3d<Fonc_Num>     & Tr(INT aNum) =0 ;
	 virtual  void InitFoncteur(cElCompiledFonc &,INT aNum) =0;
 

          AllocateurDInconnues & Alloc();

      // CurRot() : matrice de passage courante   coord camera -> coord monde
          virtual ElRotation3D  CurRot () = 0;
          virtual ~cIncParamExtrinseque();
          std::vector<cElCompiledFonc *> &  
		  FoncteurRappelCentreOptique ();
	  virtual void SetRappelCOInit()=0;

          tPosition TPos() const;
          std::string   NameType();

          const  cIncIntervale & IntervInc() const;
          cIncIntervale & IntervInc() ;
          cIncEnsembleCamera * Ensemble();

          static cIncParamExtrinseque * IPEFixe (AllocateurDInconnues &,ElRotation3D aRInit,cIncEnsembleCamera *);

          static cIncParamExtrinseque * IPEBaseUnite 
		                        (
					   AllocateurDInconnues &,
					   Pt3dr aCentreRot,
					   ElRotation3D aRInit,
					   cIncEnsembleCamera *
					);

          static cIncParamExtrinseque * IPELibre 
		  (AllocateurDInconnues &,ElRotation3D aRInit,cIncEnsembleCamera *);


     protected :


         cIncParamExtrinseque 
         (
             tPosition ,
             AllocateurDInconnues  & anAlloc,
             cIncEnsembleCamera *    
         );
         void EndInitIPE();


         AllocateurDInconnues & mAlloc;
         tPosition              mTPos;
         cIncIntervale          mIncInterv;
         cIncEnsembleCamera *   mpEns;
         std::vector<cElCompiledFonc *> mFCO;  
         std::vector<double *>          mFC0Adr;

     private :


      // Prohibited
          void operator = (const cIncParamExtrinseque &);
          cIncParamExtrinseque(const cIncParamExtrinseque &);

	  friend class cIncCpleCamera;
          static cIncParamExtrinseque *  Alloc
                 (
                     tPosition ,
                     AllocateurDInconnues &,
                     ElRotation3D aRInit,
                     cIncEnsembleCamera *  = 0
                 );

};


class cIncParamCamera
{
      public  :

           cIncParamCamera
           (
                  cIncParamIntrinseque &  anIPI,
                  cIncParamExtrinseque &  anIPE,
                  cIncEnsembleCamera * = 0
           );

	   cIncParamExtrinseque & ParamExtr();
	   cIncParamIntrinseque & ParamIntr();

           cIncParamExtrinseque::tPosition  TPos () const;

           bool SameIntrinseque(const cIncParamCamera &) const;


           Pt3d<Fonc_Num>  DirRayon(Pt2d<Fonc_Num> aPCam,INT aNumParamI,INT aNumParamE);

           Pt3d<Fonc_Num>  VecteurBase(INT Num1,cIncParamCamera &,INT Num2);


           const  cIncIntervale & III() const;
           const  cIncIntervale & IIE() const;
           cIncIntervale & III() ;
           cIncIntervale & IIE() ;

           std::string NameType (bool SameIntr);
	   void InitFoncteur(cElCompiledFonc &,INT aNumI,INT aNumE);
          AllocateurDInconnues & Alloc();
          cIncEnsembleCamera * Ensemble();

      private :

          cIncParamIntrinseque & mIPI;
          cIncParamExtrinseque & mIPE;
          cIncEnsembleCamera *   mpEns;

      // Prohibited
          void operator = (const cIncParamCamera &);
          cIncParamCamera(const cIncParamCamera &);
};




class cIncCpleCamera
{
      public  :

        virtual ~cIncCpleCamera();
        cIncCpleCamera
	(
	     cIncParamCamera & aCam1,
	     cIncParamCamera & aCam2,
             cIncEnsembleCamera * = 0
	);
         static void GenerateAllCode();


	 // retourne le residu, signe, non pondere
	 double ValEqCoPlan(const Pt2dr & aP1,const Pt2dr & aP2);
	 // retourne le residu, signe, non pondere
	 double DevL1AddEqCoPlan
		 (const Pt2dr & aP1,const Pt2dr & aP2,REAL aPds,
                  cGenSysSurResol &);

         double Dev2AddEqCoPlan ( const Pt2dr & aP1, const Pt2dr & aP2,
            REAL aPds, L2SysSurResol & aSys);


	 // Reinitialise mFoncteur a partir des valeurs de Alloc()
	 void InitCoord();
         cIncEnsembleCamera * Ensemble();
         cElCompiledFonc *    Fonc();

      private  :

	 void SetP1P2(const Pt2dr & aP1,const Pt2dr & aP2);
         AllocateurDInconnues & Alloc();
         std::string NameType ();
         void GenerateCode(const std::string  & aDir,const char * Name =0);

         static void GenerateAllCodeSameIntr
                     (
                           const std::string & aDir,
                           cIncParamExtrinseque::tPosition aPos1,
                           bool  aFocFree1,
                           bool  aPPFree1,
			   INT   aDegreDR1,
                           cIncParamExtrinseque::tPosition aPos2,
                           const char * Name = 0
                     );

         static void GenerateAllCodeDiffIntr
                     (
                           const std::string & aDir,
                           cIncParamExtrinseque::tPosition aPos1,
                           bool  aFocFree1,
                           bool  aPPFree1,
                           cIncParamExtrinseque::tPosition aPos2,
                           bool  aFocFree2,
                           bool  aPPFree2
                     );

         static void GenerateAllCodeGen
                     (
                           const std::string & aDir,
                           cIncParamExtrinseque::tPosition aPos1,
                           bool  aFocFree1,
                           bool  aPPFree1,
			   INT   aDegreDR1,
                           cIncParamExtrinseque::tPosition aPos2,
                           bool aSameIntr,
                           bool  aFocFree2,
                           bool  aPPFree2,
			   INT   aDegreDR2,
                           const char * Name = 0
                     );





         bool              mOrdInit;
         cIncParamCamera & mCam1;
         cIncParamCamera & mCam2;
         cIncEnsembleCamera * mpEns;


          std::string     mMemberX1;
          std::string     mMemberX2;
          std::string     mMemberY1;
          std::string     mMemberY2;

          std::string     mParamX1;
          std::string     mParamX2;
          std::string     mParamY1;
          std::string     mParamY2;


          bool            mSameIntr;
          INT             mNumIntr2; // Est-ce que possede ses propre variable locale
          INT             mNumExtr2; // Est-ce que possede ses propre variable locale
          Pt2d<Fonc_Num>  mP1;
          Pt2d<Fonc_Num>  mP2;
          Pt3d<Fonc_Num>  mDRay1;
          Pt3d<Fonc_Num>  mDRay2;

          Fonc_Num        mEqCoplan;
          cIncListInterv  mLInterv;

          cElCompiledFonc * mFoncteur;
	  bool              mWithDynFCT;

	  double *          mAdrX1;
	  double *          mAdrY1;
	  double *          mAdrX2;
	  double *          mAdrY2;


          cElCompiledFonc * mDebugFonc;
	  double *          mDebugAdrX1;
	  double *          mDebugAdrY1;
	  double *          mDebugAdrX2;
	  double *          mDebugAdrY2;

           

      // Prohibited
          void operator = (const cIncCpleCamera &);
          cIncCpleCamera(const cIncCpleCamera &);

         
};

class cIncEnsembleCamera 
{
     public :

         cIncEnsembleCamera(bool DerSec);
         virtual ~cIncEnsembleCamera();

         cIncParamIntrinseque  *  NewParamIntrinseque
                                  (
                                      REAL     aFocale,
                                      bool     isFocaleFree,
                                      Pt2dr    aPP,
                                      bool     isPPFree,
	                              ElDistRadiale_PolynImpair * = 0 // Si on la donne, ell est libre
                                  );


         cIncParamExtrinseque *  NewParamExtrinsequeRigide(ElRotation3D aRInit);
         cIncParamExtrinseque *  NewParamExtrinsequeLibre(ElRotation3D aRInit);
         cIncParamExtrinseque *  NewParamExtrinsequeBaseUnite(Pt3dr Centre,ElRotation3D aRInit);


         cIncParamCamera  * NewParamCamera
                             (
                                    cIncParamIntrinseque &  anIPI,
                                    cIncParamExtrinseque &  anIPE
                             );
         cIncCpleCamera * NewCpleCamera
	                  (
	                       cIncParamCamera & aCam1,
	                       cIncParamCamera & aCam2
	                  );

         void SetL2Opt(bool L2Opt);

         REAL AddEqCoPlan(cIncCpleCamera &,const Pt2dr &,const Pt2dr &,REAL aPds);
         void AddEqRappelCentreDR(cIncParamIntrinseque & aParamI, REAL aPds);

         void ResetEquation();
         void ItereLineaire();  // Ne remplit pas la matrice
         void OneItereDevL1(bool WithFaux);  // Vide et  Remplit la matrice

	 void StdAddEq(cElCompiledFonc *,REAL aP);


         void OptLineaireOnDirL2(std::list<cIncSetLiaison *> *  aListSL,const std::vector<cFonctrPond> &);
         void OptimPowel(std::list<cIncSetLiaison *> * ,const std::vector<cFonctrPond> &,REAL tol,INT ItMax);
         void OptimJacobi(std::list<cIncSetLiaison *> *  aListSL,const std::vector<cFonctrPond> &);

         Im1D_REAL8 CurVals();
         void SetPtCur(const double * aPt);

     private :

        friend class  cIEC_OptimCurDir;
        friend class  cIEC_OptimPowel;

         REAL ScoreLambda(REAL);
         void SetLambda(REAL aLambda);
         void SetImState0();
         void SetCurDir(const double * aDir);
         REAL ScoreCur(bool WithFAux);
         REAL ScoreCurGen(bool WithFAux,bool CumulDeriv);
         INT  NbVal();




	 cIncParamExtrinseque * AddIPE(cIncParamExtrinseque *);
         void VerifFiged();
         void SetOpt();

          typedef std::list<cIncParamIntrinseque *> tContPI;
          typedef std::list<cIncParamExtrinseque *> tContPE;
          typedef std::list<cIncParamCamera *>      tContCam;
          typedef std::list<cIncCpleCamera *>       tContCple;
          typedef std::list<cElCompiledFonc  *>     tContFcteur;

          bool                       mL2Opt;
          bool                       mDerSec;
          AllocateurDInconnues       mAlloc;
          L2SysSurResol *            mSysL2;
          SystLinSurResolu *         mSysL1;
          cGenSysSurResol *          mSys;

          tContPI               mIPIs;
          tContPE               mIPEs;
          tContCam              mCams;
          tContCple             mCples;
          tContFcteur           mLFoncteurs;

      // Prohibited
          void operator = (const cIncEnsembleCamera &);
          cIncEnsembleCamera(const cIncEnsembleCamera &);

          

         std::list<cIncSetLiaison *> *  mListSl;
         std::vector<cFonctrPond>       mFoncsAux;
         Im1D_REAL8                     mImState0;
         Im1D_REAL8                     mImDir;

         ElMatrix<REAL>                 mMatrL2;
         ElMatrix<REAL>                 mMatrtB;
         ElMatrix<REAL>                 mMatrValP;
         ElMatrix<REAL>                 mMatrVecP;
         ElMatrix<REAL>                 mtBVecP;
};



class cIncSetLiaison
{
     public :
        cIncSetLiaison(cIncCpleCamera * aCple);
        void AddCple(Pt2dr aP1,Pt2dr aP2,REAL aPds);
        ElPackHomologue & Pack();
        cIncCpleCamera *  Cple();
     private :
        cIncCpleCamera *    mCple;
        ElPackHomologue     mSetCplH;
};


#endif //  _PHGR_OREL_H_





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
