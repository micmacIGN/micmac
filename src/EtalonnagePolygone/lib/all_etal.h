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
    bin/EPExePointeInitPolyg  param.txt NumCible  : pour saisir

Script : 

    EPExeRechCibleInit  param.txt   :  recherche des cibles avec modeles a priori 
                                      (sur les images ListeImagesInit)

    EPExeCalibInit      param.txt   :  modele radiale initial

    EPExeRechCibleDRad param.txt   : recherche des cibles sur toutes les
                                     images avec un premier modele radiale
 
    EPExeCalibFinale  param.txt :  modele radial final.


    bin/EPtest param.txt aKindIn aKindOut  [XML=]

             aKindIn = NoGrid |  NonDeGrille.xml
             aKindOut = Hom | NoDist | Pol3 | ... | Pol7 | DRad | PhgrStd
    
*/


/*
    Salut le LOEMI

    A Faire :

     - ecriture de Sz dans param_calcule
     - Le Type "lazy Tiff"


     :
*/




/*
*/

#pragma once

#include "StdAfx.h"



typedef const std::string * tCStrPtr;

class cCoordNormalizer;
class cCibleRechImage;
class cCamIncEtalonage;
class cCpleCamEtal;
class cEtalonnage;
class cHypDetectCible;
class cSetHypDetectCible;
class cParamEtal;
// Relient entre eux un ensemble de cameras rigides
class cBlockEtal;


class cHypDetectCible
{
      public :

	bool OkForDetectInitiale(bool RequirePerf) const;

	cHypDetectCible(const cSetHypDetectCible *,
			const cCiblePolygoneEtal &,
			const cCamIncEtalonage &,
			Pt3dr DirU,
			Pt3dr DirV
		       );
        Pt2dr Terrain2ImageGlob(Pt3dr) const;

	REAL DistCentre(const cHypDetectCible & aH2) const;
	REAL DistForme(const cHypDetectCible & aH2) const;

	REAL A0() const;
	REAL B0() const;
	REAL C0() const;
	Pt2dr Centr0() const;
	Pt2dr CentreFinal() const;
	const cCamIncEtalonage & Cam() const;

	void SetConfusionPossible(bool);
	bool ConfPos() const;
	bool InsideImage() const;
	const cCiblePolygoneEtal & Cible();
	REAL  GdAxe() const ;
        REAL  Largeur() const;

	void SetResult
	     (
	          Pt2dr aCentre,
		  REAL  aLarg,
	          bool Ok,
		  REAL Correl,REAL DistC,REAL DistShape
	     );

	const cSetHypDetectCible & Set() const;
	bool OkDetec() const;
	bool  PosFromPointe() const;


      private :
	const cSetHypDetectCible * pSet;
	bool                       mOkDetec;
	bool                       mConfPos;
	const cCiblePolygoneEtal * mCible;
	const cCamIncEtalonage *   mCam;
	REAL                       mRay;
	Pt2dr                      mCentr0;
	REAL                       mA0;
	REAL                       mB0;
	REAL                       mC0;
	bool                       mInsideImage;
	REAL                       mGdAxe;
	REAL                       mDirGA;
	REAL                       mPtAxe;

	Pt2dr                      mCentreFinal;
	REAL                       mLargeur;
	REAL                       mCorrel;
	REAL                       mDistCentreInit;
	REAL                       mDistShapeInit;
	bool                       mPosFromPointe;
};


class cSetHypDetectCible
{
      public :
	      void AddHyp
		   (
		        const cCiblePolygoneEtal &,
			const cCamIncEtalonage &,
			Pt3dr DirU,
			Pt3dr DirV
	           );
	      const std::list<cHypDetectCible *>  & Hyps() const;

	      cSetHypDetectCible
	      (
		         const cParamEtal &,
                  const cPolygoneEtal &,
	          const cCamIncEtalonage &,
	          REAL aDistConf,
		  REAL aFactConfEll,
                  Pt3dr DirU,
                  Pt3dr DirV
	      );

	      REAL DistConfusionCentre() const;
	      REAL DistConfusionShape()  const;
	      void SauvFile
	           (
		          const cEtalonnage &,
		          const cSetPointes1Im & aPointes,
                          const cParamRechCible & ,
			  const std::string & aName,
			  bool Complet
                   );
      private :
	      std::list<cHypDetectCible *>  mHyps;
	      REAL                          mDistConf;
	      REAL                          mFactConfEll;

};


class cCibleRechImage
{
	public :
		cCibleRechImage(cEtalonnage &,INT aSz,INT Zoom);  // Zoom <=0 -> Pas de Visu
                // void  Recherche();
	
		void RechercheImage
		     (
		        cHypDetectCible &
		     );
		const cParamEtal & ParamEtal() const;
        private :
		Video_Win * AllocW();
		void ShowCible(Video_Win *,INT aCoul);


		// Retourne la correlation entre image et synthese
		REAL OneItereRaffinement
                     (REAL aFactPts,bool LargLibre,bool ABCLibre);

		cEtalonnage &              mEtal;
		const cCamIncEtalonage *   pCam;
		Pt2di       mSzIm;
		INT         mZoom;
                Im2D_INT4   mIm;
		TIm2D<INT,INT> mTIm;
                Im2D_U_INT1 mImSynth;
		TIm2D<U_INT1,INT> mTImSynth;

                Im2D_U_INT1 mImInside;

		Pt2dr       mCentreImSynt;
                Im2D_U_INT1 mImPds;
		std::vector<Pt2di> mPtsCible;
                Video_Win*  pWIm;
                Video_Win*  pWSynth;
                Video_Win*  pWFFT;
                Video_Win*  pWGlob;
		Pt2di       mDecIm;

                cSetEqFormelles   mSetM7;
                cSetEqFormelles   mSetM6;
                cSetEqFormelles   mSetM5;

                cSetEqFormelles   mSetS3;
                cSetEqFormelles   mSetS2;
                cSetEqFormelles   mSetSR5;
                cSetEqFormelles   mSetMT0;
                cSetEqFormelles   mSetMN6;
                cSetEqFormelles   mSetME6;


                cSetEqFormelles*  pSet;
                REAL              mDefLarg;

                cEqEllipseImage * pEqElImM7;
                cEqEllipseImage * pEqElImM6;
                cEqEllipseImage * pEqElImM5;
                cEqEllipseImage * pEqElImS3;
                cEqEllipseImage * pEqElImS2;
                cEqEllipseImage * pEqElImSR5;
                cEqEllipseImage * pEqElImMT0;
                cEqEllipseImage * pEqElImN6;
                cEqEllipseImage * pEqElImE6;
                cEqEllipseImage * pEqElIm;

};

class cCoordNormalizer
{
	public :
		virtual Pt2dr ToCoordNorm(Pt2dr ) const = 0;
		virtual Pt2dr ToCoordIm(Pt2dr ) const = 0;
		virtual ~ cCoordNormalizer();

		static cCoordNormalizer * NormCamId(REAL Foc,Pt2dr pp);
		static cCoordNormalizer * NormCamDRad
			(bool C2M,REAL Foc,Pt2dr pp,const ElDistRadiale_PolynImpair &);
		static cCoordNormalizer * NormalizerGrid(const std::string &);
	protected:
		cCoordNormalizer();
	private :
};

class cCamIncEtalonage
{
	public :
		cCamIncEtalonage
                (
		     INT                 aNumCam,
		     cEtalonnage &       anEtal,
		     const std::string & aNameTiff,
                     const std::string & aShortName,
		     bool                PointeCanBeVide,
                     const std::string & aNamePointes
                );

		void TestOrient();
		void TestOrient(ElRotation3D);
		void InitOrient(cParamIntrinsequeFormel *,cCamIncEtalonage * CamRat);
		ElRotation3D RotationInitiale();

		cSetPointes1Im & SetPointes();
		cCameraFormelle & CF();

		const CamStenopeIdeale & Cam() const; 
		Tiff_Im             Tiff() const;
		Pt2dr Terrain2ImageGlob(Pt3dr) const;
		const std::string & Name() const;

                std::list<Appar23> StdAppuis(bool doPImNorm);

		void SauvEOR();
		cCamStenopeDistRadPol * CurCam() const;
		CamStenope           * CurCamGen() const;

		const cSetPointes1Im & Pointes()  const;
		const std::string & NamePointes() const;

                const std::list<Appar23> & AppuisInit() const;
		const std::list<int> &     IndAppuisInit() const;

		bool   IsLastEtape() const;
		cSetPointes1Im * PointeInitial() const;
		bool UseDirectPointeManuel() const;

		void ExportAsDico(const cExportAppuisAsDico&);
	private:
		INT                 mNumCam;
		cEtalonnage &       mEtal;
		cSetPointes1Im      mPointes;
		std::string         mNamePointes;
		std::string         mName;
		std::string         mNameTiff;
		cCameraFormelle *   pCF;
		CamStenopeIdeale    mCam;
		cLazyTiffFile       mTiff;
                std::list<Appar23>  mAppuisInit;
		// En paralele a mAppuisInit
		std::list<int>      mIndAppuisInit;
		mutable cSetPointes1Im *    mPointeInitial;

		
		typedef cSetPointes1Im::tCont tLPointes;
		bool mUseDirectPointeManuel;
};

class cCpleCamEtal
{
      public :
          cCpleCamEtal(bool isC2M,cSetEqFormelles &,cCamIncEtalonage *,cCamIncEtalonage *);
	  void AddLiaisons(cEtalonnage &);

	  void Show();
	  REAL DCopt() const;
	  cCamIncEtalonage * Cam1();
	  cCamIncEtalonage * Cam2();
      private :
          cCamIncEtalonage * pCam1;
          cCamIncEtalonage * pCam2;
	  cCpleCamFormelle * pCpleRes1;
	  cCpleCamFormelle * pCpleRes2;
	  ElPackHomologue    mPack;
};

struct cCiblePointeScore
{
           cCiblePointeScore(cCamIncEtalonage *,REAL Score,INT Id);
	   REAL mScore;
           INT  mId;
	   cCamIncEtalonage * mCam;
};

extern bool AllowUnsortedVarIn_SetMappingCur;

class cEtalonnage
{
	public :

		friend class cBlockEtal;
		static INT CodeSuccess();
		cCpleCamEtal * CpleMaxDist() const;
                void XmlSauvPointe();
		static void RechercheCiblesInit(int argc,char ** argv);
		static void CalculModeleRadialeInit(int argc,char ** argv);
		static void RechercheCiblesDRad(int argc,char ** argv);
		static void CalculModeleRadialeFinal(int argc,char ** argv);

	        static void TestHomOnGrid(int argc,char ** argv);
	        static void TestHomOnDRad(int argc,char ** argv);
	        static void TestPolOnDRad(int argc,char ** argv);

	        static void TestGrid(int argc,char ** argv);
		void SauvGrid(REAL aStep,const std::string &, bool XML);
		void SauvDataGrid(Im2D_REAL8,const std::string & aName);
	        static void TestLiaison(int argc,char ** argv);

                static void DoCompensation(int argc,char **argv);
                void DoCompensation(const std::string & aParamComp,const std::string & aPrefix,bool OptionFigeC);


		void ExportAsDico(const cExportAppuisAsDico&);

		void AddErreur(REAL anEcart,REAL aPds);
                REAL ECT() const;
		Pt2dr  SzIm();
		const cParamEtal & Param() const;
		const cPolygoneEtal &  Polygone() const;
		Pt2dr FromPN(Pt2dr aP) const ; 
		Pt2dr ToPN(Pt2dr aP) const;  // Transforme en un point photogrametrique
		REAL FocAPriori() const;


		cEtalonnage
	        (
		     bool   isLastEtape,
		     const cParamEtal &,
		     cBlockEtal * aBlock = 0,
		     const std::string & ModeDist = ""
		);
		~cEtalonnage();
		std::string NamePointeInit(const std::string &);  // 4 point
		void InitNormCmaIdeale();
		void InitNormIdentite();
                ElRotation3D RotationFromAppui(std::string &,cSetPointes1Im * SetPrecis =0);
                std::list<Appar23> StdAppuis(bool doPImNorm,std::string &,cSetPointes1Im *  SetPrecis= 0);
		cEtalonnage &  EtalRatt();
		cEtalonnage *  EtalRattSvp();
		// Toute les rotation sont sauvees dans le sens Monde->Cam
		ElRotation3D GetBestRotEstim(const std::string & aNameCam);

		static void Verif (bool IsLast,int,char**);

		void WriteCamDRad(const std::string &);

		ElDistRadiale_PolynImpair CurDist() const; // Remise en coordonnes images
		// Remise en coordonnes images
		ElDistRadiale_PolynImpair CurDistInv(INT aDelta) const; 

		REAL CurFoc() const;
		Pt2dr CurPP() const;

		REAL  NormFoc0() const;
		Pt2dr NormPP0() const;

                void  TestVisuFTM();
		std::string NamePointeResult(const std::string &,bool Interm,bool Complet); 

                void SauvAppuisFlottant();
	private:
		void TestEpip
	             (Video_Win aW,Pt2dr aP,Pt2dr P2,const std::string &,CpleEpipolaireCoord *,bool Im1,bool show);

		cSetEqFormelles & Set();
		static void Verif (bool IsLast,const std::string &,std::list<std::string> &);
		void  Verif(std::list<std::string> &);
		void VerifTifFile(const std::string &,bool UseRat);
		void VerifTifFile(const std::vector<std::string> &,bool UseRat);


		std::string NameRot(const std::string & Etape,const std::string & Cam);

                cCamIncEtalonage * CamFromShrtName(const std::string &,bool SVP);
		cCamIncEtalonage*  AddCam
		     (
				  const std::string & aNameTiff,
                                  const std::string & aShortName,
				  bool  PointeCanBeVide,
                                  const std::string & aNamePointesFull
	             );
		
                void SauvPointes(const std::string & aName,cDbleGrid * = 0);
		void InitNormGrid(const std::string &);
		void InitNormDrad(const std::string &);

		void ResetErreur();
                void GenImageSaisie();


		void CalculLiaison(bool ParamFixe);

		Video_Win *              WGlob();

		    // Avec Rech sans Mod rad

		std::string NameTiffIm(const std::string &);
                void Do8Bits(const std::vector<string> & mVName);
		void RechercheCibles
		     (
                          int argc,char ** argv,
                          const std::vector<string> & mVNames,
                          const cParamRechCible & aPRC
		     );
		void CalculModeleRadiale
	             (
		         bool sauv,
		         const std::string &NameRot,
		         bool  forTest,
			 bool  FreeRad,
		         const std::vector<string> & mVNames,
			 bool PtsInterm
	             );

            // Param compl
               void UseParamCompl();
               void MakeCibleAdd();
               void MakeCibleAddOnPointe(const cCibleACalcByLiaisons &);


		void SauvDRad(const std::string & aName,const std::string & aNPhG);
		void SauvXML(const std::string & aName);
		void SauvDRad(cCamStenopeDistRadPol & aCDR);
	         void  SauvLeica(CamStenope & aCamElise);

		static const std::string  TheNameDradInterm;
		static const std::string  TheNameDradFinale;
		static const std::string  TheNamePhgrStdInterm;
		static const std::string  TheNamePhgrStdFinale;

		static const  std::string TheNameRotInit;  // Apres DRAD Init
		static const std::string TheNameRotFinale;  // Apres DRAD Glob
		static const std::string TheNameRotCouplee; // Apres Couplages Inter Calib

		static const INT NbRotEstim = 3;
		static tCStrPtr	TheRotPoss[NbRotEstim];

		typedef std::list<cCamIncEtalonage *> tContCam;
		typedef std::list<cCpleCamEtal *> tContCpleCam;
		typedef cSetPointes1Im::tCont tLPointes;

		void AddEqAllCam(bool Sauv,REAL aSeuilC);
		void AddEqCam(cCamIncEtalonage &,REAL aSeuilC);
		void NormalisePIM(cCamIncEtalonage &);

		cParamIFDistRadiale *    PIFDR() const;


		cParamEtal       mParam;
		bool             mIsLastEtape;
		bool             mDoNormPt;
		std::string      mDir;
		Pt2dr            mSzIm;
		REAL             mFocAPriori;
		Pt2dr            mMil;
		cCoordNormalizer * mNorm;
		cCoordNormalizer * mNormId;
		REAL             mMaxRay;

		REAL             mNormFoc0;
		Pt2dr            mNormPP0;

		REAL             mFactNorm;
		Pt2dr            mDecNorm;


		cPolygoneEtal &  mPol;

		cBlockEtal *              pBlock;
                bool                      mBloquee;
		cParamIFDistRadiale *     mParamIFDR;
		cParamIFHomogr *          mParamIFHom;
                cParamIFDistStdPhgr *     mParamPhgrStd;
		INT                       mDegPolXY;
		cParamIFDistPolynXY *     mParamIFPol;
		cParamIntrinsequeFormel * mParamIFGen;
	        tContCam                  mCams;
		tContCpleCam              mCples;
		REAL                      mSomErr;
		REAL                      mSomE2;
		REAL                     mSomPds;
		cElStatErreur            mStats;
		Video_Win *              mWGlob;
		// Si pEtalPartageInc existe, il sert aussi de pEtalRattachement
		cEtalonnage *            pEtalRattachement;
		cCamStenopeDistRadPol *          pCamDRad;
                std::string                      mModeDist;
		std::vector<cCiblePointeScore>   mVCPS;
                cInterfChantierNameManipulateur  * mICNM;
};

class cBlockEtal
{
	public :
		cSetEqFormelles & Set();
		cBlockEtal(bool isLastEtape,const cParamEtal &,bool AddAll);

		static void TheTestMultiRot(bool isLastEtape,int,char **);
	private :
		void AddEtal(const cParamEtal & aParam);
		void  TestMultiRot();
		void  TestMultiRot(INT K1,INT K2);

                bool                      mIsLastEtape;
		cSetEqFormelles           mSet;
		cParamEtal                mParam;
		cEtalonnage *             pE0;
		std::vector<cEtalonnage *>  mEtals;

};

class cParamEllipse
{
   public :
      cParamEllipse(int aNbDisc,Pt2dr aC,Pt2dr aDirH,Pt2dr aDirV,Pt2dr aProjN);
      int NbDisc() const;
      Pt2dr KiemV(int aK) const;  // Vecteur, ne tient pas compte de mC
      const Pt2dr &  DirH() const;
      const Pt2dr &  DirV() const;
      const Pt2dr &  ProjN() const;
           // Ne tient pas compte du centre
      const Pt2dr & PLarg()  const;

      void    Compute();
      bool VecInEllipe(const Pt2dr & aP) const;

      Pt2dr   Centre() const;
      Pt2dr   DirGAxe() const;
      Pt2dr   DirPAxe() const;
      double  GdAxe() const;
      double  PtAxe() const;
      void    SetCentre(Pt2dr aP) ;
      double  SurfInterRect(Pt2dr aP0,Pt2dr aP1);

      Pt2d<Fonc_Num>  StdTransfo(Pt2d<Fonc_Num>);
      Pt2dr           StdTransfo(Pt2dr);

   private :
      bool    mComputed;
      //  Representation par equation
      //  (AX+BY)2 + (BX+CY)2 = 1
      double  mA;
      double  mB;
      double  mC;

      //  Representation par parametre physique


      double mGdAxe;
      double mPtAxe;
      double mTetaGA;





      //  Representation initiale
      double  mTeta0;
      INT     mNbDisc;
      Pt2dr   mCentre;
      Pt2dr   mDirH;
      Pt2dr   mDirV; 
      Pt2dr   mProjN;
      Pt2dr   mPLarg;
};


class cRechercheCDD : Optim2DParam
{
   public :

       cRechercheCDD(Im2D_REAL4 anImGlob,Video_Win * mW);
       void RechercheInit(cParamEllipse & anEl,double aRab);
       void RechercheCorrel(cParamEllipse & anEl);
       void Show(cParamEllipse & anEl);

   private :

       REAL Op2DParam_ComputeScore(REAL,REAL);

       Pt2di               mDec;
       Pt2di               mSz;
       Im2D_U_INT1         mImCible;
       TIm2D<U_INT1,INT>   mTImC;
       Im2D_REAL4          mImGlob;
       TIm2D<REAL4,REAL8>  mTImG;
       Video_Win *         mW;
};


extern void TestEtal(int,char **);





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
