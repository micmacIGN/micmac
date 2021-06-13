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



#ifndef _ELISE_FONC_NUM_H
#define _ELISE_FONC_NUM_H


/************************************************************************************

      Fonc_Num_Computed :

icste : with this method, a Fonc_Num_Computed indicate id it is an integer constant
     function and, evntually, what is the value of this constant. 
     Most object will ignore
     this fact and will treat constant function like others. Only object that
     can have special optimization with constant (like X windows with not rle mode)
     will decide to use them.

*************************************************************************************/


std::vector<double> MakeVec1(const double & aD);

class Arg_Fonc_Num_Comp;

class Fonc_Num_Computed : public Mcheck
{
      public :
           virtual const Pack_Of_Pts * values(const Pack_Of_Pts *) = 0;

           virtual bool  icste( INT *); // in fact INT[ELise_Std_Max_Dim]

           inline INT  idim_out(void) const {return _dim_out;}
           inline Pack_Of_Pts::type_pack      type_out(void) const {return _type_out;}


           bool integral () const;
          virtual ~Fonc_Num_Computed();
 
      protected :          

          Fonc_Num_Computed(const Arg_Fonc_Num_Comp &,INT dim_out,Pack_Of_Pts::type_pack type_out);
          
          INT                                  _dim_out;
          Pack_Of_Pts::type_pack               _type_out;
          Flux_Pts_Computed *                  _flux_of_comp;

      private :

};

class Arg_Fonc_Num_Comp
{
      public :
         
         Arg_Fonc_Num_Comp(Flux_Pts_Computed *);
         inline Flux_Pts_Computed * flux() const {return _flux;}

     private :
         Flux_Pts_Computed * _flux;
};


class  cECFN_SetString;
class  cDico_SymbFN;
class cDico_Compiled;

class cElCompiledFonc
{
      public :
	  typedef cElCompiledFonc * (* tAllocObj)();
/*
          virtual void Compute(double *) = 0;
          REAL Val()         const {return mVal;}
          REAL Deriv(INT aK) const {return mDer[aK];}
          INT  NbVar()       const {return mNbVar;}
*/


         std::string & NameAlloc();
          static cElCompiledFonc * AllocFromName(const std::string &);
         // Renvoie 0 si pas un des champs de la structure
	 
         virtual double * AdrVarLocFromString(const std::string &) =0;
         double * RequireAdrVarLocFromString(const std::string &);

         void SetMappingCur(const cIncListInterv &,cSetEqFormelles *);
         const cIncListInterv &  MapRef() const;
         void SetCoordCur(const double * aRealCoord);

         void ComputeValAndSetIVC();

// Debug, aucune verif sur init !! Dangereux hors debug
         REAL ValBrute(INT aD)         const;

         REAL Val(INT aD)         const;
         REAL Deriv(INT aD,INT aK) const;
         REAL DerSec(INT aD,INT aK1,INT aK2) const;
	 const std::vector<double> &   Vals() const;
         const std::vector<double> &   CompCoord() const;
	 const std::vector<std::vector<double> > &  CompDer() const;
         const std::vector<double> &   ValSsVerif() const;
         const std::vector<std::vector<double> > &  CompDerSsVerif() const;


        
	 void SVD_And_AddEqSysSurResol
              (
                   bool isCstr,
                   const std::vector<INT> & aVInd,
                   REAL aPds,
                   REAL *       Pts,
                   cGenSysSurResol & aSys,
                   cSetEqFormelles & aSet,
                   bool EnPtsCur,
                   cParamCalcVarUnkEl *
              );
	 void SVD_And_AddEqSysSurResol
              (
                   bool isCstr,
                   const std::vector<INT> & aVInd,
                   const std::vector<double> & aVPds,
                   REAL *       Pts,
                   cGenSysSurResol & aSys,
                   cSetEqFormelles & aSet,
                   bool EnPtsCur,
                   cParamCalcVarUnkEl *
              );



	 void Std_AddEqSysSurResol
              (
                   bool   isCstr,
                   REAL aPds,
                   REAL *       Pts,
                   cGenSysSurResol & aSys,
                   cSetEqFormelles & aSet,
                   bool EnPtsCur,
                   cParamCalcVarUnkEl *
              );
	 void Std_AddEqSysSurResol
              (
                   bool   isCstr,
                   const std::vector<double> & aVPds,
                   REAL *       Pts,
                   cGenSysSurResol & aSys,
                   cSetEqFormelles & aSet,
                   bool EnPtsCur,
                   cParamCalcVarUnkEl *
              );



         void AddDevLimOrd1ToSysSurRes( cGenSysSurResol &,REAL aPds,bool EnPtsCur);
         void AddContrainteToSysSurRes( cGenSysSurResol &,bool EnPtsCur);
         // Ordre 2, tjs en Pts Cur
         void AddDevLimOrd2ToSysSurRes(L2SysSurResol &,REAL aPds);


         virtual ~cElCompiledFonc();

         static cElCompiledFonc * DynamicAlloc(const cIncListInterv &  aListInterv,Fonc_Num);

	 // Pour des foncteur dynamique de type Xk=Cste
         static cElCompiledFonc * FoncSetVar(cSetEqFormelles *,INT Ind,bool GenCode=false);
	 static cElCompiledFonc *FoncSetValsEq
		                 (cSetEqFormelles *,INT Ind1,INT Ind2,bool GenCode=false);

         static const std::string NameFoncSetVar;
         double * FoncSetVarAdr();
	 
	 // Pour des foncteur dynamique de type Sigma(ak,Xk)=Cste
         static const  std::string &  NameKthAffineVar(int aNB);
         static cElCompiledFonc * FoncRappelAffine(cSetEqFormelles *,INT Ind0,INT NbInd);
         double * FoncAffAdrCste();
         double * FoncAffAdrKth(INT aK); // A partir de 0,


         static cElCompiledFonc * FoncFixeNormEucl(cSetEqFormelles *,INT Ind0,INT NbInd,REAL Val,bool GenCode = false);
         static cElCompiledFonc * FoncFixeNormEuclVect(cSetEqFormelles *,INT Ind0,INT Ind1,INT NbInd,REAL Val,bool GenCode = false);
         static cElCompiledFonc * FoncFixedScal(cSetEqFormelles *,INT Ind0,INT Ind1,INT NbInd,REAL Val,bool GenCode = false);


	 void SetNormValFtcrFixedNormEuclid(REAL Val);

	 static cElCompiledFonc * GenFoncVarsInd
		         (cSetEqFormelles *,const std::string &aName,INT aNbVar,
			  std::vector<Fonc_Num> aFonc,bool Code2Gen);
	 static cElCompiledFonc * RegulD1(cSetEqFormelles *,bool Code2Gen);
	 static cElCompiledFonc * RegulD2(cSetEqFormelles *,bool Code2Gen);



	  class cAutoAddEntry 
	  {
		public :
		  cAutoAddEntry(const std::string &,tAllocObj);
	  };
	  static void InitEntries();
          static void AddEntry(const std::string &,tAllocObj);

           void InitBloc(const cSetEqFormelles &);
      protected :


  
         void AddContrainteEqSSR(bool Contr,REAL Pds, cGenSysSurResol &,bool EnPtsCur);
	  friend class cAutoAddEntry;

	  static void AddNewEntryAlloc(const std::string &,tAllocObj);
	  static class cDico_Compiled * mDicoAlloc;
          
          cElCompiledFonc(INT aDimOut);
          void AddIntRef(const cIncIntervale &);
          void Close(bool Dyn);
          void CloseIndexed();
          void AsserNotAlwaysIndexed() const;
          bool AlwaysIndexed() const;

         void SetCoord(double *); // Coordonnee non compactee
          // void SetCoordCur(double *);
         virtual void ComputeVal() = 0;
         virtual void ComputeValDeriv() = 0;
         virtual void ComputeValDerivHessian() = 0;

         virtual  void PostSetCoordCur();

	  INT                     mDimOut;
          bool                    isValComputed;
          bool                    isDerComputed;
          // bool                    isHessComputed;
          bool                    isCoordSet;
          bool                    isCurMappingSet;
	  bool                    intMayOverlap;

          INT                     mNbCompVar;
          // INT                     mNbRealVar;
          // INT                     mVarMaxComp;
          //  INT                     mVarMaxReal;

      // Je pense que le mode mAlwaysIndexed devrait etre le mode
      // systematique, mais par compatibilite ...
          bool                    mAlwaysIndexed;

          std::vector<double>     mCompCoord;
          //  double *                mRealCoord;
          std::vector<INT>        mMapComp2Real;
	  std::vector<std::vector<double> >    mCompDer;
	  // std::vector<std::vector<std::vector<double> > >   mCompHessian;

	  // Utile pour AddSys
	  // std::vector<std::vector<double> >    mRealDer;
          // std::vector<INT>        mMapReal2Comp;

          // std::vector<INT>       mListIndComp;
          // std::vector<INT>       mListIndReal;

          // Pour assurer la suppression progressive de mListIndComp & co
          inline int LIC(const int &) const;

	  std::vector<double>    mVal;
         
      private :
         void SetNoInit();

         static cElCompiledFonc * FoncFixedNormScal
                   (cSetEqFormelles * aSet,INT Ind0,INT Ind1,INT NbInd,REAL Val,bool Code2Gen,
                    cAllocNameFromInt & aNameAlloc,bool ModeNorm);

          cIncListInterv                   mMapRef;
          std::vector<cSsBloc>     mBlocs;
          std::string              mNameAlloc;
          // std::vector<double>     mBufLin;
           
};

class cElCompileFN
{
	public :

	     cElCompileFN &  operator << (const std::string &);
	     cElCompileFN &  operator << (const char *);
	     cElCompileFN &  operator << (const INT &);
	     cElCompileFN &  operator << (const double &);
	     cElCompileFN &  operator << (Fonc_Num &);

	     void  PutVarNum(INT aK);
	     void  PutVarLoc(cVarSpec);

             static void DoEverything
                         (
                            const std::string           &   aDir,
                            const std::string           &   aNameCl,
                            Fonc_Num                        aVar,
                            const cIncListInterv &          aList

                         );
             static void DoEverything
                         (
                            const std::string           &   aDir,
                            const std::string           &   aNameCl,
			    std::vector<Fonc_Num>           aVar,
                            const cIncListInterv &          aList,
                           // si true il y a d'abord les fonction pour la valeur ensuite
                           // celle pour les derivee
                            bool  SpecFnumCoorUseCsteVal = false

                         );

             friend class cNexisteQuePourFairePlaisiraGcc;


            static cElCompiledFonc * DynamicAlloc
                                     (
                                          const cIncListInterv &  aListInterv,
                                          Fonc_Num
                                     );

	    void AddToDict(Fonc_Num);
	    bool SqueezComp(Fonc_Num);
	private :

             ~cElCompileFN();
             cElCompileFN
             (
                 const std::string &              aNamDir,
                 const std::string &              aNameCl,
                 const cIncListInterv &           aListInterv
             );
             

             void SetFile(const std::string & aPostFixe,const char * incl);
             void CloseFile();
	     std::string  NameVarLoc(const std::string &);
             void MakeFileCpp(std::vector<Fonc_Num>,bool  SpecFnumCoorUseCsteVal = false  );
             void MakeFonc(std::vector<Fonc_Num> f,INT DegDeriv,bool  SpecFnumCoorUseCsteVal = false);
             void MakeFileH(bool  SpecFnumCoorUseCsteVal = false);

             cElCompileFN(const cElCompileFN &);      // Unimplemanted
             void operator = (const cElCompileFN &);  // Unimplemanted

 
             FILE * mFile;
             cECFN_SetString * mNamesLoc;
	     cDico_SymbFN   *  mDicSymb;
             INT               mNVMax;
	     std::string       mNameVarNum;
             std::string       mPrefVarLoc;

             std::string       mNameDir;
             std::string       mNameClass;
             
             const cIncListInterv &          mListInterv;

             std::string       mNameFile;
             std::string       mNameTagInclude;
};

class Fonc_Num_Not_Comp : public RC_Object
{
      public :
         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &) = 0;
         virtual bool integral_fonc(bool integral_flux) const = 0;
         virtual INT dimf_out() const = 0;

         virtual bool  is0() const;
         virtual void   inspect() const;
         virtual bool  is1() const;
         virtual bool  IsCsteRealDim1(REAL &) const;

	 virtual void compile (cElCompileFN &);
         virtual Fonc_Num deriv(INT k) const ;
         virtual void  show(std::ostream &) const ;
         virtual REAL  ValFonc(const  PtsKD &  pts) const ;
         virtual REAL  ValDeriv(const  PtsKD &  pts,INT k) const ;
	 virtual INT  NumCoord() const;
         virtual void VarDerNN(ElGrowingSetInd &) const = 0;
         virtual INT DegrePoly() const;

         virtual   Fonc_Num Simplify() ;



         virtual Fonc_Num::tKindOfExpr  KindOfExpr();
         virtual INT CmpFormelIfSameKind(Fonc_Num_Not_Comp *);
	 Fonc_Num_Not_Comp();
	 ~Fonc_Num_Not_Comp();

         std::string NameCpp();
         bool        HasNameCpp();
         void        SetNameCpp(const std::string &);

   private :
	 std::string * mNameCPP;
        // void VarDerNN(ElGrowingSetInd &)const {ELISE_ASSERT(false,"No VarDerNN");}
};


class Op_Bin_Not_Comp : public Fonc_Num_Not_Comp
{
      public :

         Fonc_Num Simplify() ;

         
         typedef double   (* TyVal)  (double,double);
         typedef Fonc_Num (* TyDeriv)(Fonc_Num,Fonc_Num,INT k);
         typedef double   (* TyValDeriv)(Fonc_Num,Fonc_Num,const  PtsKD &,INT k);

         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);
         virtual  Fonc_Num_Computed * op_bin_comp
                                      (const Arg_Fonc_Num_Comp &,
                                       Fonc_Num_Computed       * f1,
                                       Fonc_Num_Computed       * f2
                                      ) = 0;
         Op_Bin_Not_Comp
         (
                Fonc_Num,
                Fonc_Num, 
		bool  isInfixe,
                const char *,
                TyVal,
                TyDeriv,
		TyValDeriv
         );
         virtual INT dimf_out() const;
	 virtual void compile (cElCompileFN &);
         virtual Fonc_Num::tKindOfExpr  KindOfExpr();
         virtual INT CmpFormelIfSameKind(Fonc_Num_Not_Comp *);
	 void  PutFoncPar
	       (
			Fonc_Num f,
			cElCompileFN & anEnv,
			const char * mSimpl
	       );



      protected :
         Fonc_Num       _f0; 
         Fonc_Num       _f1; 
	 bool           mIsInfixe;
         const char *   _name;
         TyVal          _OpBinVal;
         TyDeriv        _OpBinDeriv;
		 TyValDeriv      mOpBinValDeriv;

         virtual void VarDerNN(ElGrowingSetInd &) const;
         virtual Fonc_Num deriv(INT k) const ;
		 virtual void  show(std::ostream &) const ;
         virtual REAL  ValFonc(const  PtsKD &  pts) const;
		 REAL ValDeriv(const  PtsKD &  pts,INT k) const;
};


class Op_Un_Not_Comp : public Fonc_Num_Not_Comp
{
      public :
         Fonc_Num Simplify() ;
         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);
         virtual  Fonc_Num_Computed * op_un_comp
                                      (const Arg_Fonc_Num_Comp &,
                                       Fonc_Num_Computed       * f
                                      ) = 0;

          typedef double   (* TyVal)  (double);
          typedef Fonc_Num (* TyDeriv)(Fonc_Num,INT k);
          typedef REAL (* TyValDeriv)(Fonc_Num,const  PtsKD &,INT k);


         Op_Un_Not_Comp(Fonc_Num,const char *,TyVal,TyDeriv,TyValDeriv);

      protected :
         Fonc_Num           _f; 
         const char *       _name;
         TyVal              _OpUnVal;
         TyDeriv            _OpUnDeriv;
		 TyValDeriv         mOpUnValDeriv;

         virtual INT        dimf_out() const;
         virtual void VarDerNN(ElGrowingSetInd &) const;
         virtual Fonc_Num deriv(INT k) const ;
		 virtual void  show(std::ostream &) const ;
         virtual REAL  ValFonc(const  PtsKD &  pts) const;
		 REAL ValDeriv(const  PtsKD &  pts,INT k) const;
	 virtual void compile (cElCompileFN &);
         virtual Fonc_Num::tKindOfExpr  KindOfExpr();
         virtual INT CmpFormelIfSameKind(Fonc_Num_Not_Comp *);
};


        /**************************************************/
        /*                                                */
        /*         Utilitaries                            */
        /*                                                */
        /**************************************************/

      
      // convertion 

extern  Fonc_Num_Computed * convert_fonc_num
        (       const Arg_Fonc_Num_Comp & arg,
                Fonc_Num_Computed * f,
                Flux_Pts_Computed * flx,
                Pack_Of_Pts::type_pack type_wished
        );


/*
    If one of the tf is REAL, all will be converted to REAL.
*/


extern Pack_Of_Pts::type_pack  convert_fonc_num_to_com_type
       (
          const Arg_Fonc_Num_Comp & arg,
          Fonc_Num_Computed * * tf,
          Flux_Pts_Computed * flx,
          INT nb
       );


extern  Fonc_Num_Computed * clip_fonc_num_def_val
        (       const Arg_Fonc_Num_Comp & arg,
                Fonc_Num_Computed * f,
                Flux_Pts_Computed * flux,
                const INT * _p0,
                const INT * _p1,
                REAL        def_val,
                REAL        rab_p0 = 0.0,
                REAL        rab_p1 = 0.0,
                bool        flush_flx  = false
        );


Fonc_Num r2d_adapt_filtr_lin(Fonc_Num f,const char *);


Fonc_Num nflag_open_sym_id(Fonc_Num);
class cAllocNameFromInt
{
      public :
          const  std::string &  NameKth(int aNB);

          cAllocNameFromInt(const std::string & aRac);

      private :
          std::vector<std::string> mNAMES;
          std::string              mRac;
};


#endif  /* _ELISE_FONC_NUM_H */

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
