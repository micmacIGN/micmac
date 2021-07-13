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

#ifndef _APERO_H_
#define _APERO_H_
#include "StdAfx.h"
#include "cParamApero.h"
#include "BundleGen.h"

double DistanceMatr(const ElRotation3D & aR1,const ElRotation3D & aR2);

class cImplemBlockCam;


extern bool ResidualStepByStep ;


void AjustNormalSortante(bool Sortante,Pt3dr & aNorm, const ElCamera * aCS1,const Pt2dr &aPIm);


double  GuimbalAnalyse(const ElRotation3D & aR,bool show);


// typedef cSetIntMultiple<4>  tFixedSetInt;
typedef cVarSetIntMultiple  tFixedSetInt;

//extern Pt2dr BugIM;
//extern Pt3dr BugTER;
//
extern bool BugBestCam;

class cCalibCam;
class cAppliApero;
class cGenPoseCam;
class cPoseCam;
class cObservLiaison_1Cple;
class cPackObsLiaison;

class cPonderateur;
class cSurfParam;
class cOnePtsMult;
class cFctrPtsOfPMul;
class cOneCombinMult;
class cOneVisuPMul;
class  cBdAppuisFlottant;

typedef std::map<std::string,cCalibCam *> tDiCal;
typedef std::map<std::string,const cCalibrationCameraInc *> tDiArgCab; // Pour gerer les calib/pose qui necessitent
typedef std::map<std::string,cPoseCam *>  tDiPo;
typedef std::map<std::string,cGenPoseCam *>  tDiPoGen;
typedef std::map<std::string, cPackObsLiaison *> tDiLia;


// Pour les rapports
class cRes1OnsAppui;

class cArgGetPtsTerrain;
class cArgVerifAero;

typedef ElQT<cOnePtsMult *,Pt2dr,cFctrPtsOfPMul> tIndPMul;


class cStatObs;
class cLayerImage;
class cOneImageOfLayer;


class cClassEquivPose;
class cRelEquivPose;


class cPoseCdtImSec;

std::vector<cGenPoseCam *> ToVecGP(const std::vector<cPoseCam *> &);
std::vector<cPoseCam *> ToVecDownCastPoseCamNN(const std::vector<cGenPoseCam *> &);

// Compilation des blocs, avant init
class cPreCB1Pose;
class cPreCompBloc;

/************************************************************/
/*                                                          */
/*              EQUIVALENCE                                 */
/*                                                          */
/************************************************************/

class cObsCentre
{
    public :
       Pt3dr                 mCentre;
       Pt3dr                 mIncertOnC;
       bool                  mHasObsC;
       bool                  mVitFiable;
       cTplValGesInit<Pt3dr> mVitesse;
};

// Utilisation dans AddObservationsRigidGrp 

class cClassEquivPose
{
    public :
        cClassEquivPose(const std::string & anId);
        void AddAPose(cGenPoseCam *);
        const std::vector<cGenPoseCam *> &   Grp() const;
        const std::string & Id() const;
    private :
        cClassEquivPose(const cClassEquivPose &); // N.I.

        std::string               mId;
        std::vector<cGenPoseCam *>   mGrp;
};

class cRelEquivPose
{
      public :
          //cRelEquivPose(int aNum);
          cRelEquivPose();
          cClassEquivPose * AddAPose(cPoseCam *,const std::string & aName);

          const std::map<std::string,cClassEquivPose *> & Map() const;
          void Show();
          cClassEquivPose &  ClassOfPose(const cGenPoseCam &);

          bool SameClass(const cGenPoseCam &,const cGenPoseCam &);
      private :
          cRelEquivPose(const cRelEquivPose &); // N.I. 

          // int                                     mNum;
          std::map<std::string,cClassEquivPose *> mMap; // Map   NomDeClasse -> Classe
          std::map<std::string,cClassEquivPose *> mPos2C; // Map   Nom de pose -> Classe
};


/************************************************************/
/*                                                          */
/*              INCONNUES                                   */
/*                                                          */
/************************************************************/

void CompleteSurfParam();


class cAperoOffsetGPS
{
     public :
          cAperoOffsetGPS(const cGpsOffset &,cAppliApero &);
          const cGpsOffset & ParamCreate() const;
          cBaseGPS *         BaseUnk();
     private :
          cAppliApero & mAppli;
          cGpsOffset    mParam;
          cBaseGPS *    mBaseUnk;
};

class cCalibCam
{
     public :

        virtual void Inspect();
        static cCalibCam *  Alloc
	                    (
                                 const std::string & aKeyId,
			         cAppliApero &,
				 const cCalibrationCameraInc &,
                                 cPoseCam *
                            );   
	CamStenope & CamInit();
	void SetContrainte(const cContraintesCamerasInc &);
	       // Resultat indique si la contrainte a ete traitee
	virtual bool InstSetContrainte
	             (
		           double aTol,
			   const eTypeContrainteCalibCamera &
	             )= 0;

	cParamIntrinsequeFormel & PIF();
	Pt2di  SzIm() const;

	void ActiveContrainte(bool Stricte);

	const cCalibrationCameraInc &   CCI();
        // double    RMaxU() const;
        void SetRMaxU(double ,bool IsRel,bool OnlyFE);
        bool IsInZoneU(const Pt2dr & ) const;
        const std::string & KeyId();

        bool HasRayonMax() const;
        double RayonMax() const;
        void AddViscosite(const std::vector<double> & aTol);
        void InitAvantCompens();
        void AddPds(const Pt2dr & aPt,const double & aPds);
        void Export(const std::string & aNameXml);
        void PostFinCompens();


     protected :
        virtual ~cCalibCam();

        cCalibCam  
	(
            const cCalibrationInternConique &,
            bool                isFE,
            const std::string & aKeyId,
            cAppliApero &,
            const cCalibrationCameraInc &, 
            cParamIntrinsequeFormel &,
	    CamStenope &            aCamInit,
	    Pt2di                   aSzIm
	);

        bool                            mIsFE;
	cAppliApero &                   mAppli;
        std::string                     mKeyId;
	const cCalibrationCameraInc &   mCCI;
	cParamIntrinsequeFormel &       mPIF;
	CamStenope &                    mCamInit;
	Pt2di                           mSzIm;
	Pt2dr                           mMil;
        double                          mRMaxU2;
        bool                            mFiged;
        double                          mPropDiagU;
        double                          mRay2Max;

        double                          mReducPReg;
        Pt2di                           mSzPReg;
        Im2D_REAL4                      mImReg;
        TIm2D<REAL4,REAL>               mTImReg;
        double                          mSomNbReg;
        double                          mSomPdsReg;
};

/*
template <class T1,class T2> 
void AssertEntreeDicoVide(T1 &  aCont,const T2 & aVal,const std::string & aMessage)
{
    if (aCont.find(aVal) != aCont.end())
    {
         std::cout << "  NAME= "<< aVal << "\n";
	 std::string aM = "Non unique name for "+aMessage;
         ELISE_ASSERT(false,aM.c_str());
    }

}

template <class TDic> 
typename TDic::mapped_type 
GetEntreeNonVide(TDic & aDic,const std::string& aName,const std::string& aMes)
{
    typename TDic::mapped_type  aV=aDic[aName];
    if (aV==0)
    {
        std::cout << "Entree = " << aName << "  ;; Contexte = " << aMes << "\n";
	ELISE_ASSERT(false,"Pas d'entree trouvee dans le dictionnaire\n");
    }
    return aV;
}
*/

// Ce peut etre (tjs le cas aujourd'hui) un simple plan
//
//
class cSurfParam
{
     public :
         static cSurfParam *  NewSurfPlane
	 (
            cAppliApero & anAppli,
            const std::string &,
            const Pt3dr &,
            const Pt3dr &,
            const Pt3dr & 
	 );

	 void MakeInconnu(const cSurfParamInc & aParamP);
	 cSurfInconnueFormelle * EqSurfInc();

	 void AssertUsed() const;
	 void SetUsed();
     private :
         cSurfParam
	 (
            cAppliApero & anAppli,
            const std::string &,
            const Pt3dr &,
            const Pt3dr &,
            const Pt3dr & 
        );
         cAppliApero &          mAppli;
         const std::string      mName;
         Pt3dr                  mP0;
         Pt3dr                  mP1;
         Pt3dr                  mP2;
	 cSurfInconnueFormelle * mEqInc;
         bool                   mIsUsed;
};


int PROF_UNDEF();

class cAttrArcPose
{
    public :
        cAttrArcPose();
        double & Pds();
        int    & Nb();
        int   Nb() const;
        double Pds() const;
    private :
        double mPds;
        int    mNb;
};

typedef ElGraphe<cGenPoseCam*,cAttrArcPose> tGrApero;

class cCompileAOI
{
   public :
     cCompileAOI(const cOptimizeAfterInit &);
     cOptimizationPowel mParam;
     std::vector<cElRegex *> mPats;
     std::vector<eTypeContraintePoseCamera> mCstr;
};


class cGenPoseCam
{
    public :
	  const std::string & Name() const;
          cPoseCam * DownCastPoseCamNN();
          const cPoseCam * DownCastPoseCamNN() const;
          virtual cPoseCam * DownCastPoseCamSVP();
          virtual const cPoseCam * DownCastPoseCamSVP() const;

          virtual cGenPDVFormelle *  PDVF() = 0;
          virtual const cGenPDVFormelle *  PDVF() const  = 0;
          virtual cCalibCam *  CalibCam() const ;
          cCalibCam *  CalibCamNN() const ;
          int   & NumTmp(); // Entre autre dans bloc bascule

          virtual const cBasicGeomCap3D * GenCurCam () const ;
          virtual cBasicGeomCap3D * GenCurCam () ;
          cPoseCdtImSec *  & CdtImSec();

          virtual bool IsInZoneU(const Pt2dr & ) const;
          virtual Pt3dr CurCentreOfPt(const Pt2dr & ) const;
          virtual void Trace() const;
          bool RotIsInit() const;
          bool PreInit() const;
          void ResetStatR();
          void AddStatR(double aPds,double aRes);
          void GetStatR(double & aSomP,double & aSomPR,double & aSom1) const;
          const ElAffin2D &  OrIntM2C() const;
          const ElAffin2D &  OrIntC2M() const;

          void ResetPtsVu();
          void AddPtsVu(const Pt3dr &);
          const std::vector<Pt3dr> & PtsVu() const;
          bool HasMasqHom() const;
	  void    AddPMoy(const Pt2dr &aPIm,const Pt3dr & aP,double aBSurH,int aKPoseThis=-1,const std::vector<double> * =0,const std::vector<cGenPoseCam*>* =0);
	  virtual void    VirtualAddPMoy(const Pt2dr &aPIm,const Pt3dr & aP,int aKPoseThis=-1,const std::vector<double> * =0,const std::vector<cGenPoseCam*>* =0);
	  void    InitAvantCompens();
	  virtual void    VirtualInitAvantCompens();
	  bool    PMoyIsInit() const;
	  Pt3dr   GetPMoy() const;
	  double  SomPM() const;
	  double   ProfMoyHarmonik() const;
          int   NbPtsMulNN() const ;
          void  SetNbPtsMulNN(int) ;
          int                  &  NbPLiaisCur();
          cAnalyseZoneLiaison  &  AZL();
          double               &  QualAZL();
          bool CanBeUSedForInit(bool OnInit) const;
          void SetCurLayer(cLayerImage *);
          cOneImageOfLayer * GetCurLayer();
          void C2MCompenseMesureOrInt(Pt2dr &);
          virtual bool AcceptPoint(const Pt2dr &) const;
         void SetSom(tGrApero::TSom &);
         tGrApero::TSom * Som();
         virtual Pt2di SzCalib() const = 0;
    protected :
          virtual void UseRappelOnPose() const;
          cGenPoseCam(cAppliApero & anAppli,const std::string & aName);
   
          cAppliApero & mAppli;
	  std::string   mName;
          int           mNumTmp; // Entre autre dans bloc bascule
          cPoseCdtImSec *              mCdtImSec;
          bool                         mRotIsInit;
          bool                         mPreInit;
          double                       mStatRSomP;
          double                       mStatRSomPR;
          double                       mStatRSom1;
          ElAffin2D                    mOrIntM2C;
          ElAffin2D                    mOrIntC2M;
          std::vector<Pt3dr>           mPtsVu;
          Im2D_Bits<1> *               mMasqH;
          TIm2DBits<1> *               mTMasqH;
	  Pt3dr                        mPMoy;
	  double                       mMoyInvProf;
	  double                       mSomPM;
          bool                         mLastEstimProfIsInit;
          double                       mLasEstimtProf;
          int                          mNbPtsMulNN;
          int                          mNbPLiaisCur;
          cAnalyseZoneLiaison          mAZL;
          double                       mQualAZL;
          cOneImageOfLayer *           mCurLayer;
          tGrApero::TSom *             mSom;
};


class cPosePolynGenCam : public  cGenPoseCam
{
     public  :
         cPosePolynGenCam(cAppliApero &,const std::string & aNameIma,const std::string & aDirOri);
         virtual cGenPDVFormelle *  PDVF() ;
         virtual const cGenPDVFormelle *  PDVF() const;
         cPolynBGC3M2D_Formelle *   PolyF() ;
         Pt2di SzCalib() const ;
     private :
         cPosePolynGenCam(const cPosePolynGenCam &); // N.I. 

         std::string             mNameOri;
         cPolynomial_BGC3M2D *   mCam;
         cPolynBGC3M2D_Formelle  mCamF;
};

class cStructRigidInit
{
    public :
        cStructRigidInit(cPoseCam * RigidMere,const ElRotation3D &);

        cPoseCam *   mCMere;    
        ElRotation3D mR0m1L0 ;  //  R0-1 L0 = matrice de passage selon notaion cImplemBlockCam.cpp
};

class cPreCB1Pose
{
   public :
       cPreCB1Pose(const ElRotation3D &);
       const ElRotation3D mRot;
};
class cPreCompBloc
{
    public :
        cPreCompBloc(const std::string  & aNameBloc);
        std::vector<cPoseCam*> mGrp;
        const std::string            mNameBloc;
};



class cPoseCam : public cGenPoseCam
{
     public :

	 virtual void    VirtualAddPMoy(const Pt2dr & aPIm,const Pt3dr & aP,int aKPoseThis=-1,const std::vector<double> * =0,const std::vector<cGenPoseCam*> * =0);
	 virtual void    VirtualInitAvantCompens();
         virtual cPoseCam * DownCastPoseCamSVP();
         virtual const cPoseCam * DownCastPoseCamSVP() const;
         virtual cCalibCam *  CalibCam() const ;
         virtual cGenPDVFormelle *  PDVF() ;
         virtual const cGenPDVFormelle *  PDVF() const;
         Pt2di SzCalib() const ;

       // Fonction relative a une camera eventuellement non ortho,
       // si active alors toute evolution est bloquee
         void  SetCamNonOrtho(CamStenope *);
         CamStenope *  GetCamNonOrtho() const;  // Erreur si != 0
         bool HasCamNonOrtho() const;
         void AssertHasCamNonOrtho() const;
         void AssertHasNotCamNonOrtho() const;



         void AddMajick(cMajickChek &) const;

         bool IsId(const ElAffin2D & anAff) const;
         bool FidExist() const;
         bool AcceptPoint(const Pt2dr &) const;

/*
         double & MMNbPts();
         double & MMGainAng();
         Pt3dr  & MMDir();
	 Pt2dr  & MMDir2D();
	 std::vector<double> & MMGainTeta();

         bool &  MMSelected();
         double & MMGain();
         double & MMAngle();
*/



         void SetOrInt(const cTplValGesInit<cSetOrientationInterne> &);
         

         double Time() const;
         void Trace() const;
         static cPoseCam * Alloc
	                   (
			         cAppliApero &,
				 const cPoseCameraInc &,
				 const std::string & aNamePose,
				 const std::string & aNameCalib,
                                 cCompileAOI *
                           );

         const CamStenope * CurCam() const;
         CamStenope * NC_CurCam();
         CamStenope * DupCurCam() const;
         const cBasicGeomCap3D * GenCurCam () const ;
         cBasicGeomCap3D * GenCurCam () ;
         void DoInitIfNow();

          cCalibCam * Calib() const;
	  void SetContrainte(const cContraintesPoses &);
          void SetFigee();
          void SetDeFigee();
	  cCameraFormelle * CamF();
	  void ActiveContrainte(bool Stricte);
	  double  AltiSol() const;
	  double  Profondeur() const;
          double  GetProfDyn(int & Ok) const;



          void  ShowRel(const cTraceCpleCam &,const cPoseCam & aCam2) const;

         // Si true requiert une initialisation complete
          bool ProfIsInit() const;
          int Prof2Init() const;
          void Set0Prof2Init();
          void UpdateHeriteProf2Init(const cPoseCam &) ;
          void   InitRot();
          void   InitCpt();
          const std::string &  NameCalib() const;

          cPoseCam *  PoseInitMST1();
          cPoseCam *  PoseInitMST2();
          void SetPoseInitMST1(cPoseCam * aPoseInitMST1);
          void SetPoseInitMST2(cPoseCam * aPoseInitMST2);

          double & PdsTmpMST();
          ElRotation3D   CurRot() const;
          Pt3dr CurCentre() const;
          Pt3dr CurCentreOfPt(const Pt2dr & ) const;
          void PCSetCurRot(const ElRotation3D & aRot);
          void  SetBascRig(const cSolBasculeRig & aSBR);



          void InitIm();
          bool PtForIm(const Pt3dr & aPTer,const Pt2di & aRab,bool Add);
          void CloseAndLoadIm(const Pt2di & aRab);
          bool ImageLoaded() const;
          void AssertImL() const;
          const Box2di & BoxIm();
          Im2D_U_INT2  Im();

          bool HasObsOnCentre() const;
          bool LastItereHasUsedObsOnCentre() const;
          
          bool HasObsOnVitesse() const;
          const Pt3dr  & ObsCentre() const;
          Pt3dr   Vitesse() const;
	  cRotationFormelle &     RF();

          bool DoAddObsCentre(const cObsCentrePDV & anObs);

          Pt3dr AddObsCentre
               (
                     const cObsCentrePDV &,
                     const cPonderateur &  aPondPlani,
                     const cPonderateur &  aPondAlti,
                     cStatObs &
               );


          void BeforeCompens();
          int  NumInit() const;

          std::string  CalNameFromL(const cLiaisonsInit & aLI);

          void SetNameCalib(const std::string &);
          void  SetLink(cPoseCam * aPrec,bool OKLink);

          int  NumCreate() const;

          bool IsInZoneU(const Pt2dr & ) const;
          int   NbPosOfInit(int aValDef);
          void  SetNbPosOfInit(int);


          cEqOffsetGPS *   EqOffsetGPS();
         // Relatif a un couple dans un bloc
          void SetSRI(cStructRigidInit * aSRI);
          cStructRigidInit*  GetSRI(bool SVP) const;

          // Dand cImplemBlockCam.cpp 
          cPreCompBloc * GetPreCompBloc(bool SVP) const; // SVP => can be 0
          void   SetPreCompBloc(cPreCompBloc *);
          cPreCB1Pose *  GetPreCB1Pose(bool SVP) const; // SVP => can be 0
          void  SetPreCB1Pose(cPreCB1Pose *);
          void UseRappelOnPose() const override;
          int DifBlocInf1(const cPoseCam &) const; // Return "Many" if not initialized
          void SetNumTimeBloc(int aNum);

          void AddObsPlaneOneCentre(const cXml_ObsPlaneOnPose & ,const double & aWeight);
     private  :

          void AssertHasObsCentre() const;
          void AssertHasObsVitesse() const;
          void   AffineRot();
	  void SetRattach(const std::string &);
          cPoseCam
	  (
	        cAppliApero &,
		const cPoseCameraInc &,
                const std::string & aNamePose,
                const std::string & aNameCalib,
	        cPoseCam *             aPRat,
                cCompileAOI            *
          );


	  void TenteInitAltiProf(int aPrio,double anAlti, double aProf);

	  static int                theCpt;

          // cAppliApero &             mAppli;
	  std::string               mNameCalib;
	  // std::string               mName;
	  int                       mCpt;  // Compteur de date de creation
          // 0 si direct (appuis,BDD, ) sinon 1 + Prof de la pose de base
          int                       mProf2Init;
          double                    mPdsTmpMST;
	  const cPoseCameraInc *    mPCI;
	  cCalibCam *               mCalib;
	  cPoseCam *                mPoseRat;
          cPoseCam *                mPoseInitMST1;
          cPoseCam *                mPoseInitMST2;

	  cCameraFormelle *         mCamRF;
	  cCameraFormelle *         mCF;
	  cRotationFormelle *       mRF;
	  // Distance aux zone plane
	  std::map<std::string,double> mDZP;
	  double                       mAltiSol;
	  double                       mProfondeur;
          double                       mTime;
	  int                          mPrioSetAlPr;
          const cContraintesPoses *    mLastCP;
          cCompileAOI *                mCompAOI;

          bool                         mFirstBoxImSet;
          bool                         mImageLoaded;
          Box2di                       mBoxIm;
          Im2D_U_INT2                  mIm;
          TIm2D<U_INT2,INT>            mTIm;

          cObsCentre                   mObsCentre;
          bool                         mHasObsOnCentre;
          bool                         mHasObsOnVitesse;
          bool                         mLastItereHasUsedObsOnCentre;

          // Pour qualifier les Pack Pts Mul

          // Ensemble des poses a centre comun 
          int                          mNumInit;
          int                          mNumBande;
          cPoseCam *                   mPrec;  
          cPoseCam *                   mNext;

          int                          mNumCreate; // Pour conserver l'ordre de creation
          // Ensemble des poses liees par des equations de liaison
          int                          mNbPosOfInit;
   // Parametres lies aux export pour MicMac (en fait de maniere + generale
   // a la gestion du canevas)

/*
          bool                         mMMSelected;
          double                       mMMGain;
          double                       mMMAngle;
          Pt2dr                        mMMDir2D;
          Pt3dr                        mMMDir;
          double                       mMMNbPts;
          double                       mMMGainAng;
          std::vector<double>          mMMGainTeta;
*/


          bool                         mFidExist;

          CamStenope *                 mCamNonOrtho;
          cEqOffsetGPS *               mEqOffsetGPS;
          cStructRigidInit *           mSRI;
          cPreCompBloc *               mBlocCam;
          int                          mNumTimeBloc;
          cPreCB1Pose *                mPoseInBlocCam;
          bool                         mUseRappelPose;  // Do we use a "rappel" to a given value
          ElRotation3D                 mRotURP;  // Rotation use Rappel Pose
};


/************************************************************/
/*                                                          */
/*              OBSERVATIONS (Mesures)                      */
/*                                                          */
/************************************************************/


// N.B. les classes cObserv1Im et cPackObserv1Im ne sont pas tres bien faites,
// Elles possedent en commun d'etre (+ou-) en bijection avec les Poses inconnues .  Apres
// avoir ecrit le code de 


/*
    Pour utiliser ce mecanisme :

      - Definir tObj  :  le type de la donnee basique qui sera stockee dans  la
       valeur mVals
*/


template <class TypeEngl> class cObserv1Im;
class  cTypeEnglob_Appuis
{
     public :
        typedef std::list<Appar23>  tObj;
	typedef cBDD_PtsAppuis      tArg;
	static tObj CreateFromXML( cAppliApero &,const std::string &,const tArg &,cObserv1Im<cTypeEnglob_Appuis> &);

        struct tSuppplem
	{
	     Pt3dr   mBarryTer;
	};
};

class  cTypeEnglob_Orient
{
     public :
        typedef ElRotation3D  tObj;
	typedef cBDD_Orient   tArg;
	static tObj CreateFromXML( cAppliApero &,const std::string &,const tArg &,cObserv1Im<cTypeEnglob_Orient> &);

        struct tSuppplem
	{
	     double  mAltiSol;
	     double  mProfondeur;
	     double  mTime;
             ElAffin2D  mOrIntC2M;
	};
};


class cTypeEnglob_Centre
{
     public :
        typedef cObsCentre  tObj;
	typedef cBDD_Centre   tArg;
	static tObj CreateFromXML( cAppliApero &,const std::string &,const tArg &,cObserv1Im<cTypeEnglob_Centre> &);

        struct tSuppplem
	{
	};
};


template <class TypeEngl> class cObserv1Im : public TypeEngl::tSuppplem
{
     public :
         cObserv1Im
         (
                 cAppliApero & anAppli,
                 const std::string& aNameXML,
                 const std::string& aNameIm,
		 const typename TypeEngl::tArg &
         );
		 ~cObserv1Im();


         cObserv1Im(cAppliApero & anAppli,typename TypeEngl::tObj aVals,const std::string& aNameIm);

         const typename TypeEngl::tObj  & Vals() const;
         typename TypeEngl::tObj  & Vals() ;
         void Compile(cAppliApero &);
	 const std::string   &   Im() const;
         cPoseCam *              PC() const;
     public :
         cAppliApero &              mAppli;
	 std::string                mIm;
         cPoseCam *                 mPose;
	 cCameraFormelle *          mCF;
	 typename TypeEngl::tObj *mVals;
};


class cPackGlobVide
{
};

// Dervrait remplacer les "appuis flottants"
class cOneAppuiMul
{
     public :
      
         cOneAppuiMul(const Pt3dr & aPTer,int aNum);
         void AddPt(cGenPoseCam *,const Pt2dr & aPIm);

         const Pt3dr & PTer() const;
         Pt3dr  PInter() const;
         int   NbInter() const;
     private :
         Pt3dr   mPTer;
         //int     mNum;
         std::vector<double>      mVPds;
         std::vector<cGenPoseCam *>  mPoses;
         cNupletPtsHomologues     mPt;
};

template <class TypeEngl,class cPackGlob> class cPackObserv1Im;

class cPackGlobAppuis
{
   public :
       cPackGlobAppuis();
       void AddObsPack(cPackObserv1Im<cTypeEnglob_Appuis,cPackGlobAppuis> &,const cBDD_PtsAppuis &);
       std::map<int,cOneAppuiMul *> *  Apps();

   private :

        int GetNum(const Appar23 &,const cBddApp_AutoNum &);
        void AddOsb1Im(cObserv1Im<cTypeEnglob_Appuis> &, const cBDD_PtsAppuis &);
        std::map<int,cOneAppuiMul *> * mDicoApps;

        int                            mNumCur;
};





template <class TypeEngl,class cPackGlob> class cPackObserv1Im
{
      public :
             cPackObserv1Im
	     (
                  cAppliApero &,
		  const typename TypeEngl::tArg & anArg
	     );
             cPackObserv1Im
	     (
                  cAppliApero &,
		  const std::string & anId
	     );




             const  typename  TypeEngl::tObj & Vals (const std::string & aName) ;
	     cObserv1Im<TypeEngl>  & Obs(const std::string & aName);
	     cObserv1Im<TypeEngl>  * PtrObs(const std::string & aName);
	     void Compile();
	     std::list<cObserv1Im<TypeEngl> *> & LObs();
	     const std::list<cObserv1Im<TypeEngl> *> & LObs() const;
             void Add (cObserv1Im<TypeEngl> * anObs);
             cPackGlob & Glob();

             typename  TypeEngl::tArg & Arg();

      private :
          cAppliApero &                               mAppli;      
	  std::string                                 mId;
          std::list<cObserv1Im<TypeEngl> *>             mLObs;
	  std::map<std::string,cObserv1Im<TypeEngl> *>  mDicObs;
          cPackGlob                                     mGlob;
          typename TypeEngl::tArg *                     mArg;
};

typedef cPackObserv1Im<cTypeEnglob_Appuis,cPackGlobAppuis> tPackAppuis;
typedef cPackObserv1Im<cTypeEnglob_Orient,cPackGlobVide> tPackOrient;
typedef cPackObserv1Im<cTypeEnglob_Centre,cPackGlobVide> tPackCentre;



class cResul_RL;
class cAgglomRRL;

class cObservLiaison_1Cple  // Dans cPackObsLiaison
{
      public :
         cObservLiaison_1Cple
	 (
	     const cBDD_PtsLiaisons &,
	     const std::string& aNamePack,
	     const std::string& aNameIm1,
	     const std::string& aNameIm2
	 );

	 void  ImageResidu(cAgglomRRL & anAgl);

	 const std::string& NameIm1();
	 const std::string& NameIm2();
         double EcMax() const; 

	 int NbH() const;

         cPoseCam * Pose1() const;
         cPoseCam * Pose2() const;

      private :
         friend class cPackObsLiaison;

	 const ElPackHomologue & Pack() const;
         void Compile(cSurfParam *,cAppliApero &);
	 double AddObs( const cPonderationPackMesure & aPond,
		        const cPonderationPackMesure * aPondSurf
		      );


         ElPackHomologue  mPack;
	 std::string      mIm1;
	 std::string      mIm2;
         cPoseCam *       mPose1;
         cPoseCam *       mPose2;

	 // cCpleCamFormelle * mCpleR1;  // Resisu en Im1 ou Im2
	 // cCpleCamFormelle * mCpleR2;
	 cManipPt3TerInc  * mPLiaisTer;
	 cSurfParam *                               mSurf;
	 cSurfInconnueFormelle *                    mEqS;

          double                                     mEcMax; 
	                            // Ecart Max, pour repere
                                    // les erreurs d'appar
	  double                                     mSomPds;
	  double                                     mNbPts;
	  // Multiplicateur a apporter aux poids,
	  double                                     mMultPds;
};


// Pb eventuel du couts memmoire d'une combinaison specifique par cameras:
//
//   1- Pour l'instant scenario 0, on ne fait rien
//   2- Premier niveau d'intervention, au chargement on peut mettre des
//      poids 0 pour diminuer le nombre de camera
//   3- Au niveau de la compile, on peut aussi "unifier" (supprimer les combine trop
//   rare)


/*
    cOneCombinMult : contient , un ensemble de pose + le cManipPt3TerInc (qui permet 
    de manipuler le pt 3D inconnu pour un ensemmble de camera donnee).
        Cette structure est partagee par tout les cOnePtsMult correspondant a la meme
     camera.


    cOnePtsMult : contient un cNupletPtsHomologues (simplement un vecteur de points 2D) +
     un cOneCombinMult *
*/



class cOneCombinMult
{
    public :
        cOneCombinMult
	(
	       cSurfInconnueFormelle  *        anEqS,
	       const std::vector<cGenPoseCam *> & aVP,
	       const std::vector<cGenPDVFormelle *>  & aVCF ,
	       const tFixedSetInt &                    aFlag
        );
	cManipPt3TerInc * LiaisTer();
	const std::vector<int> & NumCams();
	const std::vector<cGenPoseCam *> &  GenVP();
        //
        // Renvoie -1 si pas trouve
        int IndOfPose(cGenPoseCam *) const;
        cGenPoseCam *  GenPose0() const;
        cGenPoseCam *  GenPoseK(int aK) const;
        void AddLink(cAppliApero &);
	void InitRapOnZ(const cRapOnZ *,cAppliApero & anAppli);
	bool RappelOnZApply() const;
    private :
	 cManipPt3TerInc  * mPLiaisTer;
	 std::vector<int>   mNumCams;
	 std::vector<cGenPoseCam *>  mGenVP;
	 // Pour savoir le rappel on Z calcule 
	 const cRapOnZ *   mRapOnZ;
	 bool        mRappelOnZApply;
};

Pt3dr TestInterFaisceaux
      (
           const std::vector<cGenPoseCam *> & aVC,
           const cNupletPtsHomologues  &   aNPt,
           double                          aSigma,
           bool                            Show
      );


Pt3dr InterFaisceaux
      (
           const std::vector<double> & aVPds,
           const std::vector<cGenPoseCam *> & aVC,
           const cNupletPtsHomologues  &   aNPt
      );

// Pour pouvoir optionnellement reconstituer les TieP originaux
// Par exemple dans l'export

class cSingleTieP
{
    public :
      int mK1;
      int mK2;
      Pt2dr mP1;
      Pt2dr mP2;
};

class cOnePtsMult
{
    public :
        // Si aInitRequired = false , compte le nombre de Pre-Init
        // Si aInitRequired = true,   compte le nombre de Init-Full
        int NbPoseOK(bool aFullInitRequired,bool UseZU) const;
     // Si il n'y a qu'une seule pose initialisee sur ce PM, renvoie
     // la droite image correspondant a  la projection du faisceau issu
     // de ce point dans la premiere camera
        ElSeg3D  GetUniqueDroiteInit(bool UseZU);


        const Pt2dr& P0()  const;
        const Pt2dr& PK(int ) const ;
        void AddPt2PMul(int aNum,const Pt2dr & aP,bool IsFirstSet,double aPds);
        cOnePtsMult();
        const tFixedSetInt & Flag() const; 
        void SetCombin(cOneCombinMult *);
        const double & Pds() const;
        cOneCombinMult * OCM();
        const  cNupletPtsHomologues & NPts() const;

        //  NbRealRotIsInit Nombre de rot reellement init, peut etre > si pts elim because ZU

        int  InitPdsPMul(double aPds,std::vector<double> &,int * NbRealRotIsInit=0) const;

        const cResiduP3Inc * ComputeInter
                     (
                         double aPds,
                         std::vector<double> & aVpds
                     ) const;

         Pt3dr QuickInter(std::vector<double> & aVPds) const;



        // Renvoie -1 si pas trouve
        int IndOfPose(cGenPoseCam *) const;
        cGenPoseCam *  GenPose0() const;
        cGenPoseCam *  GenPoseK(int aK) const;

         double & MemPds() ;
         Pt3dr  & MemPt() ;
         bool    MemPtOk() const;
         void    SetMemPtOk(bool) ;


         bool OnPRaz() const;
         void SetOnPRaz(bool);
         void AddSTP(const cSingleTieP & aSTP);
         const std::list<cSingleTieP> & LSTP();

    private :
        double              mMemPds;
        Pt3dr               mMemPt;
        // std::vector<Pt2dr>  mPts;
        cNupletPtsHomologues mNPts;
        tFixedSetInt         mFlagI;
        cOneCombinMult *     mOCM;
  // Rajouter a posteriori, donc valeur def par compatibilite, c.a.d si pas specifiee,
  // tous les points appartiennent au plan de rappel
        U_INT1                 mOnPlaneRapOnz;
        U_INT1                 mMemPtOk;
        std::list<cSingleTieP> mLSTP;
};




class cFctrPtsOfPMul
{
    public :
         Pt2dr operator()(cOnePtsMult * const  & aPts) const;
    private :
};


class cAppar1Im;


class cOneElemLiaisonMultiple
{
     public :
        cOneElemLiaisonMultiple(const std::string & aNameCam);
	const std::string & NameCam();
	cGenPoseCam * GenPose();
	void  Compile(cAppliApero &);

     private :
         std::string     mNameCam;
	 cGenPoseCam *   mGenPose;
};

class cStatErB
{
   public :
      cStatErB();

      void AddLab(eTypeResulPtsBundle,double aPds=1 );
      void AddLab(const cStatErB & aS2);
      void ShowStatErB();
   private : 
      double mStatRes[(int)eTRPB_NbVals] ;
      double mNbTot;
};


class cStatObs
{
    public :
         cStatObs(bool AdEq);
         void AddSEP(double aSEP);
         double SomErPond() const;
         bool   AddEq() const;

         void AddEvol(const double & aPds,const double & anEvol,const double & aMaxEvol);
         double PdsEvol() const;
         double MaxEvol() const;
         double MoyEvol() const;
         cStatErB & StatErB();
    private :
         void AssertPdsEvolNN() const;

         double mSomErPond;
         double mAddEq;
         double mMaxEvol;
         double mPdsEvol;
         double mSomEvol;
         cStatErB mStatErB;
};

/*
struct cArgObsLM
{
    public :
      cArgObsLM();
      void SetUseZ(double aZ,double aPds);
    private :
       cArgObsLM(const cArgObsLM&);
       bool  mUseP;
       Pt3dr mPTer;
       Pt3dr mPInc;
       
};
*/

class cObsLiaisonMultiple
{
      public :


//   Ces deux fonctions correspondent a la version modifiee du choix des poses
//
//   QualityZoneAlgoCV :  permet de choisir la prochaine pose, sachant
//   que le critere est calcule sur tous les points qui voient une meme zone.
//   A priori utilise pour mettre en place la deuxieme image. En l'absence
//   de point multiples pour les appuis, ce ne peut etre que celle la
//   qui fonctionne. Se content de rappeler BestPoseInitStd qui fait tres
//   bien l'essentiel du boulot
//
//
//   QualityZonePMul : permet de choisir la prochaine pose, sachant que
//   le critere que l'on priviligie est a priori le critere d'existence de 
//   points multiples, considere comme plus robuste.
//
//
//
        double StdQualityZone(bool UseZU,bool OnInit,int aNbMinPts,double aExpDist,double aExpNb,bool & GotPMul);
        double QualityZonePMul(bool UseZU,bool OnInit,int aNbPtsMin,double aExpDist,double aExpNb,bool & GotPMul);
        double QualityZoneAlgoCV(bool UseZU,bool OnInit,int aNbMinPts,double aExpDist,double aExpNb,int NbPoseMin);

          std::vector<cPoseCam *>  BestPoseInitStd
                                    (
                                      bool UseZU,
                                      bool OnInit,
                                      std::vector<double> & aVCost,
                                      int  aNbMinMul,
                                      double aExpDist,
                                      double aExpNb
                                    );

          void ClearAggregImage();


           Pt3dr CentreNuage(const cMasqBin3D * ,int * aNb) const;



         void AddLink();


          double CostOrientationPMult(const ElRotation3D & aR,const cAppar1Im &) const;
          double CostOrientationPMult(const ElRotation3D & aR,const Appar23 &) const;

          double CostOrientationPMult
                 (
                      const ElRotation3D & aR,
                      const std::vector<Appar23> &,
                      const std::vector<cAppar1Im> &
                 ) const;
     

          void TestMEPAppuis
               (
                   bool UseZU,
                   ElRotation3D & aR,
                   int aNbRansac,
                   const cLiaisonsInit &
               );

           void TestMEPCentreInit
               (
                   ElRotation3D & aR,
                   const Pt3dr & aP,
                   const cLiaisonsInit &
               );


          cObsLiaisonMultiple
          (
                   cAppliApero &       anAppli,
                   const std::string & aNamePack,
                   const std::string & aName1,
                   const std::string & aName2,
                   bool isFirstSet,
                   bool packMustBeSwap=false // file aNamePack contains the couples in inverse order (the first point belongs to aName2 and the second to aName1)
          );
          void AddLiaison(const std::string & aNamePack,const std::string & aName2,bool isFirstSet, bool packMustBeSwapped=false);

	  bool InitPack(ElPackHomologue & aPack, const std::string& aN2);
          void Compile(cSurfParam *);

	  double AddObsLM(const cPonderationPackMesure & aPond,
		        const cPonderationPackMesure * aPondSurf,
                        cArgGetPtsTerrain *,
                        cArgVerifAero * ,
                        cStatObs &,
                        const cRapOnZ *
		 );
	  double BasicAddObsLM(const cPonderationPackMesure & aPond,cStatObs &,const cRapOnZ *);
          void OLMCheckInit();






	  const std::vector<cOneElemLiaisonMultiple *> &  VPoses() const;
          const std::vector<cOnePtsMult *> & VPMul();

          cOnePtsMult& PMulLPP(Pt2dr aPt);

          cOnePtsMult * CreateNewPM
                        (
                            const std::vector<double> &       aVPds,
                            const std::vector<cPoseCam*>  &   aVPC,
                            const cNupletPtsHomologues    &   aNuple
                        );

          int IndOfCam(const cGenPoseCam *) const;
          void CompilePose();
          cGenPoseCam *  Pose1() const;

          int  NbRotPreInit() const;

      private :



          void InitRapOnZ(const cRapOnZ *      aRAZGlob);
          cOneCombinMult * AddAFlag(const cOnePtsMult & aPM);
	  int GetIndexeOfName(const std::string &); // Def -1
          void    AddPose(const std::string &,bool IsFirstSet);

           void AddPack
                (
                      const std::string & aNamePack,
                      const std::string & aName1,
                      const std::string & aName2,
                      bool isFirstSet,
                      bool packMustBeSwapped=false // file aNamePack contains the couples in inverse order (the first point belongs to aName2 and the second to aName1)
                );

           void AddCple(int anI1,int anI2,const ElCplePtsHomologues&,bool IsFirstSet,double aPds);


          cAppliApero &                              mAppli;
          //cElImPackHom                               mPack;
	  // Pt2di                                      mSz;
	  std::vector<cOneElemLiaisonMultiple *>     mVPoses;



	  std::map<tFixedSetInt, cOneCombinMult*>             mDicoMP3TI;


	  cSurfParam *                               mSurf;
	  cSurfInconnueFormelle *                    mEqS;
	  double                                     mSomPds;
	  double                                     mNbPts;
	  // Multiplicateur a apporter aux poids,
	  double                                     mMultPds;
          tIndPMul *                                 mIndPMul;

          std::vector<cOnePtsMult *>                 mVPMul;
          bool                                       mCompilePoseDone;
          cGenPoseCam *                              mPose1;
// Pour l'instant inutile, mais qd les points multiples seront partages
// entre les images, il conviendra de gerer un numero variable;
// c'est plutot un tag pour a partir de maintenant reperer les references
          int mKIm;
          const cRapOnZ *  mRazGlob;
          bool             mLayerImDone;
          Box2dr           mBox;
};

typedef enum
{
  eModeAGPNone,
  eModeAGPIm,
  eModeAGPHypso,
  eModeAGPNormale,
  eModeAGPNormaleByC,//optical center
  eModeAGPNoAttr,
  eModeAGPNoPoint
} eModeAGP;

typedef enum
{
   eTGC_CS,
   eTGC_Gen
} eTypeGenCam;

extern const  double mAGPFactN;
class cArgGetPtsTerrain
{
     public :
          cArgGetPtsTerrain(double aResolMAsq,double aLimBsH);
          void SetMasq(Im2D_Bits<1> *);
          void AddAGP
               (
                   Pt2dr aPIm, 
                   const Pt3dr &,
                   double aPds,
                   bool aUseImRed,
                   const std::vector<double> * aVPds = 0,
                   const std::vector<cGenPoseCam *> * aVPose=0

                    
               );
          // void Add(Pt3dr aPts,double aPds);

          const std::vector<Pt3dr>  &  Pts() const ;
          const std::vector<Pt3di>  &  Cols() const ;
          const std::vector<double> &  Pds() const ;
          void InitFileColor(const std::string &,double aStepIm,const std::string & aRef,int NbChan);
          void AddSeg(Pt3dr aP1,Pt3dr aP2,double aStep,Pt3di aCoul);

          void AddPts(Pt3dr aP1,Pt3di aCoul);

          void InitColiorageDirec(Pt3dr,double);
          void InitModeNormale();
          void InitModeNoAttr();
          double LimBsH() const;
          void SetByIm(bool DoByIm,bool Sym);


      private :
          cArgGetPtsTerrain (const cArgGetPtsTerrain &); // N.I.

          std::vector<Pt3dr>    mPts;
          std::vector<Pt3di>    mCouls;
          std::vector<double>   mPds;
          double                mResol;
          Im2D_Bits<1> *        mMasq;

          std::vector<Im2DGen *>   mVIms;
          double                   mStepImRed;
          std::vector<Im2DGen *>   mVImRed;

          int                      mKR;
          int                      mKG;
          int                      mKB;

          // bool                    mModeIm;
          eModeAGP                mMode;
          Pt3dr                   mDirCol;
          double                  mLimBsH;
          bool                    mDoByIm;
          bool                    mSymDoByIm;
};


struct cAVA_PtHS // Points Hors Seuil
{
    public :
        cAVA_PtHS(Pt2dr aPIM,double aDZ,const std::vector<double> &  aPds,const cOnePtsMult & aPM);

        Pt2di                 mPIM;
        double                mDZ;
        std::vector<double>   mPDS;
        const cOnePtsMult *   mPM;
};


struct cAVA_Residu
{
    public :
        cAVA_Residu(const Pt2dr &,const Pt2dr &);
        Pt2dr mP;
        Pt2dr mRes;
};


class cArgVerifAero
{
     public :
         void AddPImDZ(Pt2dr aPIM,double aDZ,const std::vector<double> &, const cOnePtsMult & );

         void AddResidu(const Pt2dr & aP1,const  Pt2dr & aRes);

         cArgVerifAero
         (
               cAppliApero & anAppli,
               Pt2di aSz,
               const cVerifAero  &,
               const std::string & aPref,
               const std::string & aPost
         );
         ~cArgVerifAero();
         const cVerifAero & VA() const;
     private :

         cAppliApero & mAppli;
         cVerifAero   mVA;
         std::string  mName;
         std::string  mNameS;
         std::string  mNameR;
         std::string  mNameB;
         std::string  mNameT;
         double       mResol;
         Pt2di        mSz;
         Bitm_Win     mW;
         Im2D_REAL4   mImZ;
         Im2D_REAL4   mImPds;
         double       mPasR;
         double       mPasB;
         std::vector<cAVA_PtHS>  mPHS;
         std::vector<cAVA_Residu> mRes;
};


class cPackObsLiaison
{
	public :
		cPackObsLiaison
		(
			cAppliApero &,
			const cBDD_PtsLiaisons & aBDL,
			int   aCpt
		);

		std::list<cPoseCam *> ListCamInitAvecPtCom(cPoseCam *);

		// Resultat indique si swaped 
		bool InitPack
		(
			ElPackHomologue &, 
			const std::string& aNameIm1, 
			const std::string& aNameIm2
		);

		void  GetPtsTerrain
		(
			const cParamEstimPlan & aPEP,
			cSetName &                    aSelectorEstim,
			cArgGetPtsTerrain &,
			const char * Attr // Nom en + pour calculer le masque
		);


		void Compile();
		void AddLink();
		double AddObs
		(
			const cPonderationPackMesure & aPondIm,
			const cPonderationPackMesure * aPondSurf,
			cStatObs & aSO,
			const cRapOnZ *
		);

		void OneExportRL(const cExportImResiduLiaison & anEIL) const;

		void AddContrainteSurfParam
		(
			cSurfParam *,
			cElRegex *  aPatI1,
			cElRegex *  aPatI2
		);

		cObsLiaisonMultiple * ObsMulOfName(const std::string &);

		std::map<std::string,cObsLiaisonMultiple *> & DicoMul();
	  
		private :
			void addFileToObservation( 
										const std::string &i_poseName1, const std::string &i_poseName2,
										const std::string &i_packFilename,
										const cBDD_PtsLiaisons &i_bd_liaison,
										int i_iPackObs, // index of the current pack in cAppliApero->mDicoLiaisons
										bool i_isFirstKeySet,
										bool i_isReverseFile // couples inside i_packFilename are to be reversed before use
									 );

			cAppliApero &                    mAppli;      
			cBDD_PtsLiaisons                 mBDL;
			std::string                      mId;
			bool                             mIsMult;

			std::vector<cObservLiaison_1Cple *>  mLObs;
			std::map<std::string,std::map<std::string,cObservLiaison_1Cple *> > mDicObs;

			std::map<std::string,cObsLiaisonMultiple *> mDicoMul;
			std::vector<cSurfParam *>           mVSurfCstr;
			std::vector<cElRegex *>             mPatI1EqPl;
			std::vector<cElRegex *>             mPatI2EqPl;
			int                                 mFlagArc;
};



/************************************************************/
/*                                                          */
/*              Les appuis flottants                        */
/*                                                          */
/************************************************************/

class cOneAppuisFlottant
{
    public :
       cOneAppuisFlottant
       (
          cAppliApero &,
	  const std::string & aName,
          bool  HasGround,
          const Pt3dr & aPt,
	  const Pt3dr & anInc,
          cBdAppuisFlottant &
       );

       void AddLiaison(const std::string & aNameIm,const cOneMesureAF1I &,const Pt2dr & anOffset,bool aModeDr,double anEcart);
       void Compile();
       double AddObs(const cObsAppuisFlottant &,cStatObs & aSO,std::string & aCamMaxErr);

       const Pt3dr &  PtRes() const;
       const Pt3dr &  PtInit() const;
       const Pt3dr &  PInc() const;
       const std::string & Name() const;
       bool   HasGround() const;

       Pt3dr PInter() const;

    //  EN fait ne fait rien car  mMP3TI est une interface sur PtTmp ..
       void DoAMD(cAMD_Interf *);
       int NbMesures() const;

       void Add2Appar32(std::list<Appar23> &,const std::string & aNameCam,int & aNum);

    private :
       cAppliApero &           mAppli;
       std::string             mName;
       cNupletPtsHomologues *  mNupl;
       cManipPt3TerInc *       mMP3TI;
       std::vector<Pt2dr>      mPts;
       std::vector<double>     mPdsIm;
       std::vector<double>     mEcartIm;
       std::vector<cGenPoseCam *> mCams;
       std::vector<bool>       mIsDroite;
       bool mHasGround;
       Pt3dr mPt;
       Pt3dr mInc;
       Pt3dr mPtRes;
       //cBdAppuisFlottant & mBDF;

};



class cBdAppuisFlottant
{	  
    public :
       void ShowError();
       cBdAppuisFlottant(cAppliApero &);
       void AddAFLiaison(const std::string & aNameIm,const cOneMesureAF1I &,const Pt2dr & anOffset,bool OkNoGr,bool ModeDr,double anEcart);
       void AddAFDico(const cDicoAppuisFlottant &);

       void Compile();
       void AddObs(const cObsAppuisFlottant &,cStatObs & aSO);
       void  ExportFlottant(const cExportPtsFlottant & anEPF);
       void DoAMD(cAMD_Interf *);
       const std::map<std::string,cOneAppuisFlottant *> & Apps() const;

       std::list<Appar23> Appuis32FromCam(const std::string & aName);
    private :

       cBdAppuisFlottant(const cBdAppuisFlottant &) ; // N.I.

       std::map<std::string,cOneAppuisFlottant *> mApps;
       cAppliApero & mAppli;

};

/************************************************************/
/*                                                          */
/*                 VISUALISATION                            */
/*                                                          */
/************************************************************/


class cRes1OnsAppui
{
    public :
      int        mNum;
      cPoseCam * mPC;
      Pt2dr      mPIm;
      Pt2dr      mErIm;

      cRes1OnsAppui(int aNum,cPoseCam * aPC,Pt2dr aPIm, Pt2dr aErIm);
      
};


class cAuxVisuPMul
{
    public :
         cAuxVisuPMul
         (
              const cVisuPtsMult & aVPM,
              Video_Win * aWRef,
              Video_Win::ePosRel
         );
         Video_Win * W();
         void clear();
         void SetChc(Pt2dr aTr,double aSc);
    private :
         int       mSz;
         Video_Win mW;
};



class cOneVisuPMul
{
    public :
         friend class cOVP_Inter;
         cOneVisuPMul(const cVisuPtsMult &,cAppliApero &);
         void InterAction();
         void DoOnePMul(cOnePtsMult & aPM );

         void ShowVect();

         cInterpolateurIm2D<U_INT2>* Interp() const;
         const Pt3dr&  PTer00()   const;
         const Pt3dr&  X_VecTer() const;
         const Pt3dr&  Y_VecTer() const;

    private :
         void ShowCenterOnePMul(cOnePtsMult &,int aCoul);

         cAppliApero &             mAppli;
         const cVisuPtsMult &      mVMP;
         bool                      mModeVisu;
         cPoseCam *                mCam;
         std::string               mNameCam;
         cObsLiaisonMultiple *     mOLM;
         std::string               mNameF;
         Tiff_Im                   mFileIm;
         Pt2di                     mSzF;
         double                    mRatio;
         Pt2di                     mSzW;
         Video_Win *               mWPrinc;
         VideoWin_Visu_ElImScr *   mVVE;
         ElPyramScroller *         mPyr;
         EliseStdImageInteractor * mESII;
         std::vector<cAuxVisuPMul *>  mVWaux;

     // Lies a la recherche des pts multiples
         cInterpolateurIm2D<U_INT2>* mInterp;
         Pt3dr                       mPTer00;
         Pt3dr                       mX_VecTer;
         Pt3dr                       mY_VecTer;
        
};


/************************************************************/
/*                                                          */
/*              La "re-correlation"                         */
/*                                                          */
/************************************************************/

class cRecorrel : private Optim2DParam
{
     public :
         cRecorrel
         (
              const cOneVisuPMul &,
              cPoseCam *          ,
              double aSzV,  // 1 Pour 3x3
              double aStep  // 1.0 "habituellement", mais pq p 0.5 ou 0.33
         );
 
         double TestAndUpdateOneCorrelOfDec(Pt2dr aDec,cRecorrel &);
         double TestAndUpdateOneCorrelOfDec
                (Pt2dr aDec,const std::vector<cRecorrel*> &aVRC);


         void  ExploreVois
               (int aNb,double aStep,const std::vector<cRecorrel*> &aVRC);
         void  ExploreVois (int aNb,double aStep,cRecorrel &aVRC);
         

         double OneCorrelOfDec(Pt2dr aDec,cRecorrel &);
         double BestCorrel() const;
         Pt2dr BestPImAbs() const;

         void DescAs_Optim2DParam(const std::vector<cRecorrel*> &aVRC);
         void DescAs_Optim2DParam(cRecorrel& aVRC);

         void TransfertState(const cRecorrel &);
     private :
          

          REAL Op2DParam_ComputeScore(REAL,REAL);

          void Udpate(const Pt2dr & aDec,double aCorrel);

          bool SetValDec(Pt2dr aDec);
          cInterpolateurIm2D<U_INT2>  * mInterp;
          Im2D_U_INT2           mIm;
          TIm2D<U_INT2,INT>     mTIm;
          Pt2di                 mSzIm;
          Pt2dr                 mBestDec;
          double                mBestCorrel;
          Box2dr                mBox;

          std::vector<Pt2dr>   mPInit;
          std::vector<Pt2dr>   mPDec;
          std::vector<double>  mValues;
          std::vector<double>  mBestValues;
          int                  mNbPts;
          double               mS1;
          double               mS2;
          Pt2dr                mPImInitAbs;
          Pt2dr                mPImInitLoc;
          Pt2dr                mDx;
          Pt2dr                mDy;
          bool                 mIsInit;


          const std::vector<cRecorrel*> * mVRC;
};

/************************************************************/
/*                                                          */
/*              L'Application                               */
/*                                                          */
/************************************************************/


class cPonderateur
{
     public :
         cPonderateur
	 (
	      const cPonderationPackMesure &,
	      double             aNbMesure
	 );

	 double PdsOfError(double) const;
         const cPonderationPackMesure & PPM() const;
         void SetPondOfNb(double aPdsNb);

     private :

         cPonderationPackMesure    mPPM;
         double                    mPdsStd;
         double                    mPdsStd0;
	 double                    mEcMax;
	 double                    mExpoLK;
	 double                    mSigmaP;
	 eModePonderationRobuste   mMode;

};

/*
  1-cAppliApero::cAppliApero()

    | 1.1 PreCompile()
    |
    |      | CompileInitPoseGen(true) -> Cree la liste des noms de poses
    |      | PreCompileAppuisFlottants(true) -> Cree la liste points flottants, pas
    |      |        encore relies aux mesures
    |
    | 1.2 InitBDDObservations() -> charge toutes les observations, (depend de 1.1
    |                           -> pour filtrer les noms reellement utilises, qui
    |                           -> peut etre < au match des Regex)
    |      |  InitBDDLiaisons()
    |      |  InitBDDAppuis()
    |      |  InitBDDOrient()
    |      |  InitBDDCentre()
    |
    |
    | 1.3 InitInconnues() -> Cree pour chaque inconnue, la structure qui la relie au
    |                     ->  systeme formel global, depend de 1.2 pour initialisation
    |
    |      |  InitCalibCam()
    |      |  InitOffsGps()
    |      |  InitPoses()
    |      |  InitPlans()
    |
    |  1.4  CompileObsersvations() -> relie les observations aux inconnues
    |                              -> creees en 1.3
    |      |  CompileLiaisons()
    |      |  CompileNewPMul()
    |      |  CompileAppuis()
    |      |  CompileOsbOr()
    |      |  CompileObsCentre()
    |      |  InitAndCompileBDDObsFlottant();
    |      |  InitAndCompileBDDObsDr();


-----------------------------------------------

  2- cAppliApero::DoCompensation()

 Map Lambda(UneEtape) :->  DoOneEtapeCompensation()
    
    |  2.1  MAJContraintes(UneEtape) -> Modifie pour chaque inconnue l'etat des ses
    |                                -> contraintes internes
    |   
    |       MAP   MAJContrainteCamera  UneEtape.ContraintesCamerasInc()
    |       MAP   MAJContraintePose    UneEtape.ContraintesPoses()
    |   
    |  2.2 Map  Lambda(aKIter)   : OneIterationCompensation
    |
    |     2.2.1  ActiveContraintes() -> Ajoute les contraintes actives au systeme
    |
    |        MAP    ActiveContrainte() mDicoCalib
    |        MAP    ActiveContrainte() mDicoPos
    |
    |
    |     2.2.2   AddObservations(UneEtape.Observations());
    |
    |  [0  ...  UneEtape.NbIteration]
    |
    |  2.3  Export(UneEtape)

 EtapeCompensation()

 */

class cMTActive
{
   public :
      cMTActive(int aNbPer);
      void SetKCur(int aKCur);
      bool SelectVal(const Pt2dr &) const;
   private :
      bool SelectVal(const double &) const;
      int mNbPer;
      int mKCur;
};

class cMTResult
{
     public :
         void AddResisu(double);
         void NewSerie();
         bool IsActif();
         void SetActif();
         void SetInactif();
         cMTResult();
         void Show(FILE*);
         void AddCam(CamStenope * aCam);
     private :
          std::vector<double> mVRes;
          std::vector<double> mVNb;
          std::vector<std::vector<CamStenope *> > mVCams;
          bool                mIsActif;
};


class cCompFilterProj3D
{
     public :
           cCompFilterProj3D(cAppliApero &,const cFilterProj3D &);
           void AddPose(cPoseCam *);
           bool  InFiltre(const Pt3dr &) const;
     private :
           cAppliApero &                 mAppli;
           cFilterProj3D                 mFilter;
           std::vector<cPoseCam *>       mCams; 
           std::vector<Im2D_Bits<1> *>   mVMasq;
           std::vector<TIm2DBits<1> *>   mVTMasq;
};


class cOneImageOfLayer
{
    public :
         cOneImageOfLayer
         (
                cAppliApero &,
                const  cLayerImageToPose &,
                const std::string & aNameIm,
                cOneImageOfLayer *  aLayerTer
         );

         INT LayerOfPt(const Pt2dr & aP);
         static const int  mTheNoLayer;


         void SplitLayer(cOneImageOfLayer&,const std::string & aNameH,const cSplitLayer & aSL);
    private  :

         bool LoadFileRed(const std::string & aNameRed);
         Im2D_U_INT1 MakeImagePrio(Im2D_U_INT1 aImIn,int aDeZoom,int aSzBox);


        
         cAppliApero &         mAppli;
         Im2D_U_INT1           mIm;
         TIm2D<U_INT1,INT>     mTIm;
         std::vector<int>      mVPrio;
         std::vector<int>      mVLabOfPrio;
         int                   mDeZoom;
         int                   mLabMin;
         int                   mLabMax;
         cOneImageOfLayer *    mLayerTer;
         ElCamera *            mCam;
         cGeoRefRasterFile *   mGRRF;
         cSysCoord *           mSysCam;
         double                mZMoy;
};


class cLayerImage
{
    public :
         cLayerImage(cAppliApero &,const cLayerImageToPose &);
         void AddLayer(cPoseCam &);
         cOneImageOfLayer & GetLayer(const std::string & aNameIm);
         void SplitHomFromImageLayer(const std::string &, const cSplitLayer &,const std::string &,const std::string &);
         bool IsTerrain() const;
         cOneImageOfLayer * NamePose2Layer(const std::string &);
    private  :
         std::string  NamePose2NameLayer(const std::string &);

         cAppliApero &                              mAppli;
         cLayerImageToPose                          mLI2P;
         std::map<std::string,cOneImageOfLayer *>   mIms;
         cLayerTerrain *                            mParamLT;
         cOneImageOfLayer *                         mImTerrain;
};

class cParamBascBloc
{
     public :
         cParamBascBloc();

         double             mSomInvNb;
         std::vector<double> mBsH;
         std::vector<Pt3dr> mP1;
         std::vector<Pt3dr> mP2;
};


class cCompiledObsRelGPS
{
    public :
        cCompiledObsRelGPS(
               cAppliApero &,
               cDeclareObsRelGPS
        );
        const cDeclareObsRelGPS & XML() const;
        const std::vector<cPoseCam *> &       VOrderedPose() const;
        const std::vector<cEqRelativeGPS *> & VObs() const;

    private :
        std::vector<cPoseCam *>        mVOrderedPose;
        std::vector<cEqRelativeGPS *>  mVObs;
        cDeclareObsRelGPS mXML;
        cAppliApero *     mAppli;
};

class cSetPMul1ConfigTPM;
class cSetTiePMul;
class cCompile_BDD_NewPtMul;
class cStatResPM;
class cInfoAccumRes;
class cAccumResidu;


class cAppliApero : public NROptF1vND
{
    public :
       void ExportImageResidu() ;
       void ExportImageResidu(const std::string & aName,const cAccumResidu &) ;


        void AddInfoImageResidu
             (
                 const Pt3dr &                 aPt,
                 const  cNupletPtsHomologues & aNupl,
                 const std::vector<cGenPoseCam *> aVP,
                 const std::vector<double> &  aVpds
             );
        void AddOneInfoImageResidu
             (
                 const cInfoAccumRes & anInfo,
                 const std::string &   aName,
                 Pt2di                 aSz,
                 double                aFactRed,
                 bool                  OnlySign,
                 int                   aDeg
             );


        std::string GetNewIdCalib(const std::string & aLongName);
        std::string GetNewIdIma(const std::string & aLongName);
        std::string IdOfCalib(const int &) const;
        std::string IdOfIma(const int &) const;

        void AddEcPtsFlot(const Pt3dr &) ;

        void AddStatCam(cGenPoseCam *,double aRes,double aPerc);
        void DebugPbConvAppui();
        cXmlSauvExportAperoOneIter & CurXmlE(bool SVP=false);

        int  NumSauvAuto() const {return  mNumSauvAuto;}
        bool NumIterDebug() const;
        int   CptIterCompens() const {return mCptIterCompens;}
        FILE * FileDebug();
        std::string MagickStr();
        void   MessageDebug(const std::string &);
        void AddMajick(double aVal);
  
        FILE *  FpRT();  // File Rapport Txt
        cMesureAppuiFlottant1Im StdGetOneMAF(const std::string & aName);

        ElPackHomologue  StdPackCorrected(const std::string& aNamePack,const std::string&aNameIm1,const std::string&aNameIm2);

        cCompFilterProj3D * FilterOfId(const std::string&);
        bool PIsActif(const Pt2dr& aP) const;
        cAppliApero( cResultSubstAndStdGetFile<cParamApero> aParam);
        ~cAppliApero();


        // Accesseurs basiques 
	cSetEqFormelles & SetEq();
	const cParamApero & Param() const;

        // Acces aux inconnues

	bool  CalibExist(const std::string &);
	cCalibCam * CalibFromName(const std::string &,cPoseCam * );

	cPoseCam *  PoseFromName  (const std::string &);
	cPoseCam *  PoseFromNameSVP  (const std::string &);
	cGenPoseCam *  PoseGenFromName  (const std::string &);
	cGenPoseCam *  PoseGenFromNameSVP  (const std::string &);
	// cPoseCam *  PoseFromNameGen  (const std::string &,bool SVP);


        // :0 = > Stenope , 2 Gen ,
	cPoseCam *     PoseCSFromNameGen  (const std::string &,bool SVP);
	cGenPoseCam *  PoseGenFromNameGen  (const std::string &,bool SVP);

	bool   PoseExist  (const std::string &);
        //  cSurfParam * SurfIncFromName(const std::string &);

        // Acces aux observations

       std::list<cPoseCam *> ListCamInitAvecPtCom
                             (
                                 const std::string& anId,
                                 cPoseCam *         aCam1
                             );

       cPackObsLiaison * PackOfInd(const std::string& anId);
       cObsLiaisonMultiple * PackMulOfIndAndNale
                             (
                                  const std::string& anId,
                                  const std::string& aName
                             );
       bool InitPack
            ( 
                const std::string& anId,
	        ElPackHomologue & aPack,
                const std::string& aN1,
	        const std::string& aN2
            );
       bool InitPackPhgrm
            ( 
                const std::string& anId,
	        ElPackHomologue & aPack,
                const std::string& aN1,CamStenope * aCam1,
	        const std::string& aN2,CamStenope * aCam2
	     );
        std::list<Appar23>  GetAppuisDyn( const std::string& anId,const std::string & aName) ;
        // const std::list<Appar23> & OldAppuisStat( const std::string& anId,const std::string & aName) ;
        std::list<Appar23>  AppuisPghrm(const std::string& anId,const std::string & aName,cCalibCam *);
        const ElRotation3D & Orient(const std::string& anId,const std::string & aName);
	cObserv1Im<cTypeEnglob_Orient> & ObsOrient(const std::string& anId,const std::string & aName);
        cObserv1Im<cTypeEnglob_Appuis> & ObsAppuis(const std::string& anId,const std::string & aName);
        cPackGlobAppuis & PackGlobApp(const std::string& anId);
        cPackGlobAppuis * PtrPackGlobApp(const std::string& anId,bool SVP);

        cObserv1Im<cTypeEnglob_Centre> & ObsCentre(const std::string& anId,const std::string & aName);
        bool HasObsCentre(const std::string& anId,const std::string & aName);
	cPackObsLiaison * GetPackOfName(const std::string&);

        cDataObsPlane *  GetDOPOfName(const std::string& Id);
        void AddObservationsPlane(const cDataObsPlane &);


        // Dans le cas "particulier" ou on ajoute toutes les images connexes a une images
        // donnees, il est necessaire d'acceder a une liste de points homolgues avant 
        // de connaitre l'ensemble des images

        const cBDD_PtsLiaisons & GetBDPtsLiaisonOfId(const std::string &);

	void DoCompensation();

	// Verifie qu'il n'existe pas et la rajoute
	void NewSymb(const std::string &);

	bool NamePoseGenIsKnown(const std::string &) const;
	bool NamePoseCSIsKnown(const std::string &) const;

        const std::string &  DC() const;
        const std::string &  OutputDirectory() const;
        bool  HasEqDr() const;

        const std::string & NameCalOfPose(const std::string &);
	cInterfChantierNameManipulateur * ICNM();

	cSurfParam * AddPlan
	        (
                    const std::string & aName,
                    const Pt3dr &   aP0,
                    const Pt3dr &   aP1,
                    const Pt3dr &   aP2,
                    bool  CanExist   // Est-ce que le plan peut deja exister
		);

	cBdAppuisFlottant * BAF_FromName(const std::string & aName, bool CanCreate,bool SVP=false);
        tGrApero &  Gr();

        double PdsOfPackForInit(const ElPackHomologue & aPack,int & aNbHom);

        void PowelOptimize
             (
                   const cOptimizationPowel & anOpt,
                   const std::vector<cPoseCam *>&  mCams,
                   const std::vector<eTypeContraintePoseCamera>&  mLib
             );

        void AddPoseInit(int,cPoseCam *);

       void   LoadImageForPtsMul
              (
                    Pt2di aRabIncl,
                    Pt2di aRabFinal,
                   const std::list<cOnePtsMult *> & aLMul
              );


        
        const std::vector<cPoseCam*> & VecLoadedPose();
        const std::vector<cPoseCam*> & VecAllPose();

        // Si vecteur non vide, donne garantie que 
        //  1- Chaque pose contient la projection de aPM avec le rab qui va bien
        //  2- Il y a au - deux pose
        //  3- La premiere est la pose0
        //
        std::vector<cPoseCam*>  VecLoadedPose(const cOnePtsMult & aPM,int aSz);

        Im2D_Bits<1> * MasqHom(const std::string & aName);

        void AddRotInit();
        void AddRotPreInit();
        int  NbRotInit() const;
        int  NbRotPreInit() const;
        const std::string & SymbPack0() const;
        bool  ModeMaping() const;
	void  DoMaping(int argc,char ** argv);
        bool  ShowMes() const;
        ostream &             COUT();

        bool AcceptCible(int aNum) const;

        bool TracePose(const std::string &) const;
        bool TracePose(const cPoseCam &) const;

        void  NormaliseScTr(CamStenope &);
        std::vector<cPoseCam *> ListPoseOfPattern(const std::string & aPat);

        void AddLinkCamCam(cGenPoseCam *,cGenPoseCam *);

        void SplitHomFromImageLayer(const std::string &, const cSplitLayer &,const std::string &,const std::string &);

        void AddResiducentre(const Pt3dr &);
        void AddRetard(double aT);

        void InitPosesTimeLink(const cTimeLinkage &);

       cLayerImage * LayersOfName(const std::string & aName);
       cRelEquivPose * RelFromId(const std::string & anId);

       bool ZuUseInInit() const;

       void TestInteractif(const cTplValGesInit<cTestInteractif> & aTTI,bool Avant);
       void TestF2C2();

       bool SameClass(const std::string&,const cGenPoseCam & aPC1,const cGenPoseCam & aPC2);



       void CheckInit(const cLiaisonsInit * ,cPoseCam *);
       bool SqueezeDOCOAC() const;  
       cAperoOffsetGPS *  OffsetNNOfName(const std::string &);
       const cXmlSLM_RappelOnPt *  XmlSMLRop();
       cArg_UPL                    ArgUPL();

       bool   UsePdsCalib();
       const cXmlPondRegDist * CurXmlPondRegDist();

       int      NbIterDone() const;
       int      NbIterTot()  const;
       double   PdsAvIter()  const;
       double   MoyGeomPdsIter(const double & aPds0, const double &  aPds1) const;
       double   MoyGeomPdsIter(const double & aPds0, const cTplValGesInit<double> &  aPds1) const;
       double   RBW_PdsTr(const cRigidBlockWeighting  &) const;
       double   RBW_PdsRot(const cRigidBlockWeighting &) const;

       void InitNewBDL();
       cCompile_BDD_NewPtMul * CDNP_FromName(const std::string & aName);
       bool CDNP_InavlideUse_StdLiaison(const std::string & anId);
       void CDNP_Compense(const std::string & anId,const cObsLiaisons &);

        void CDNP_Compense(std::vector<cStatResPM> & ,cSetPMul1ConfigTPM*,cSetTiePMul*,const cObsLiaisons &);
        bool IsLastEtapeOfLastIter() const;

        // Ne genere pas d'erreur mais met aNameTime a "" si pb (aucune ou multiple)
        ElRotation3D  SVPGetRotationBloc(const std::string & aNameBloc,const std::string& aNameCam,std::string & aNameTime);
        //  Genere erreur si moindre probleme
        ElRotation3D  GetUnikRotationBloc(const std::string & aNameBloc,const std::string& aNameCam);

        void PreInitBloc(const std::string & aNameBloc);
        int  LevStaB() const;
        bool MemoSingleTieP() const;
        bool ExportTiePEliminated() const;
        bool DebugEliminateNumTieP(int aNum) const
        {
            if (! mUseVDETp)
               return false;
            return CalcDebugEliminateNumTieP(aNum);
        }

        const cRappelPose * PtrRP() const;


    private :
       bool CalcDebugEliminateNumTieP(int aNum) const;

       void SetPdsRegDist(const cXmlPondRegDist *);

       void InitNewBDL(const cBDD_NewPtMul &);


       // Active uniquement si  mFileDebug != 0
       void AddAllMajick(int aLine,const std::string & aFile,const std::string & aMes);
       void PosesAddMajick();
       void MajAddCoeffMatrix();  

       void SetSqueezeDOCOAC();  
       void ClearAllCamPtsVu();


        void  BasculeBloc(const cBlocBascule &);
        std::vector<cPoseCam *> PoseOfPattern(const std::string & aKeyPat);
        void  BasculeBloc
              (
                       const std::vector<cPoseCam *> & aVP1,
                       const std::vector<cPoseCam *> & aVP2,
                       const std::string &             anInd
              );

        void           PrepareBlocBascule
                        (
                              cParamBascBloc &,
                              const std::vector<cPoseCam *> &, 
                              const std::vector<cPoseCam *> &,
                              const std::string & anIndPts
                        );
        void            BlocBasculeOneWay
                        (
                                    cParamBascBloc &     aBlc,
                                    const std::vector<cPoseCam *> &, 
                                    std:: vector<Pt3dr> &          ,
                                    int                        aNum1,
                                    const std::vector<cPoseCam *> &,
                                    std:: vector<Pt3dr> &          ,
                                    int                        aNum2,
                                    const std::string & anIndPts
                        );



        void InitRapportDetaille(const cTxtRapDetaille & aTRD);

        void  InitLVM
              (
                     const cSectionLevenbergMarkard*&,
                     const cTplValGesInit<cSectionLevenbergMarkard> &,
                     double & aMult,
                     const cTplValGesInit<double> &
              );
        void ShowResiduCentre();
        void ShowRetard();
         

        void AddMTResisu(const double &) ;


        void  ConstructMST
              (
                  const std::list<std::string> &,
                  const cPoseCameraInc &   aPCI
              );

        void ExportOneSimule(const cExportSimulation & anES);
	void InitPointsTerrain(std::list<Pt3dr>  &, const std::string &);
	void InitPointsTerrain(std::list<Pt3dr>  &, const cGPtsTer_By_ImProf &);

	void SimuleOneLiaison
	     (
                 const cGenerateLiaisons &,
		 const std::list<Pt3dr>  &,
                 cPoseCam & aCam1,
                 cPoseCam & aCam2
             );

        void InitLayers();
        void PreCompile();
        void PreCompilePose();
	void CompileInitPoseGen(bool isPreComp);
        void PreCompileAppuisFlottants();

        void InitClassEquiv();

	void InitBDDObservations();
	void InitBDDLiaisons();
	void InitBDDAppuis();
	void InitBDDOrient();
	void InitBDDCentre();

	void InitInconnues();
	void InitCalibCam();
	void InitCalibConseq();
	void InitOffsGps();
	void InitObsRelGPS ();
	void InitPoses();
	void InitSurf();
        void InitGenPoses();
        void InitGenPoses(const cCamGenInc&);


         void CompileObsersvations();
         void CompileLiaisons();
         void CompileNewPMul();
         void CompileAppuis();
         void CompileOsbOr();
         void CompileObsCentre();
	 void InitAndCompileBDDObsFlottant();
	 void InitHasEqDr();

         void DoAMD();
         void AMD_AddBlockCam();

          void VerifAero(const cVerifAero & aVA);
          void VerifAero(const cVerifAero & aVA,cPoseCam *,cObsLiaisonMultiple  &);

          void InitBlockCameras();
          void EstimateOIBC(const cXml_EstimateOrientationInitBlockCamera &);
          cImplemBlockCam * GetBlockCam(const std::string & anId);

          void InitFilters();

          void Verifs();
          void VerifSurf();

	template <class Type,class TGlob> void CompileObs1Im
	         (
		      std::map<std::string,cPackObserv1Im<Type,TGlob> *> &
		 );
	template <class Type,class TGlob> void InitBDDPose
	         (
		      std::map<std::string,cPackObserv1Im<Type,TGlob> *> &,
                      const std::list<typename Type::tArg> &
	         );

         void InitOneSetObsFlot(cBdAppuisFlottant * ,const cSetOfMesureAppuisFlottants &,const Pt2dr &,cSetName *,bool OkNoGr);
         void InitOneSetOnsDr(cBdAppuisFlottant *,const cSetOfMesureSegDr &,const Pt2dr &,cSetName *,bool       OkNoGr);


	                    

	void InitOneSurfParam(const cSurfParamInc &);
        void DoOneEtapeCompensation(const cEtapeCompensation &,bool LastEtape);


        void DoOneContraintesAndCompens
             (
                    const cEtapeCompensation & anEC,
                    const cIterationsCompensation &  anIter,
                    bool IsLastIter
             );
        void DoContraintesAndCompens
             (
                    const cEtapeCompensation & anEC,
                    const cIterationsCompensation &  anIter,
                    bool IsLastIter,
                    bool IsLastEtape
             );


        void DoShowPtsMult(const std::list<cVisuPtsMult> &);

        void InspectCalibs();

        void MAJContraintes(const cSectionContraintes & aSC);
	  void MAJContrainteCamera(const cContraintesCamerasInc &);
	  void MAJContraintePose(const cContraintesPoses &);

        void  OneIterationCompensation(const cIterationsCompensation & ,const cEtapeCompensation &,bool IsLast);
        double ScoreLambda(double aLambda);  
        double NRF1v(REAL); // = ScoreLambda
        bool   NROptF1vContinue() const;


	void ActiveContraintes(bool Stricte);
        void BlocContraintes(bool Stricte);
	// void ActiveContraintesCalib();
	// void ActiveContraintesPose();

        void AddObservations
             (
                 const cSectionObservations &,
                 bool IsLastIter,
                 cStatObs & aSO
             );
        void AddOneLevenbergMarkard(const cSectionLevenbergMarkard *,double aMult,cStatObs & aSO);
        void AddLevenbergMarkard(cStatObs & aSO);
        void AddRappelOnAngle(const cRappelOnAngles & aRAO,double aMult,cStatObs & aSO);
        void AddRappelOnCentre(const cRappelOnCentres & aRAC,double aMult,cStatObs & aSO);
        void AddRappelOnIntrinseque(const cRappelOnIntrinseque & aROI,double aMult,cStatObs & aSO);


        void AddObservationsAppuis(const std::list<cObsAppuis> &,bool IsLastIter,cStatObs & aSO);
        void DoRapportAppuis
             (
                  const cObsAppuis &,
                  const cRapportObsAppui&,
                  std::vector<cRes1OnsAppui> &
             );
        void AddObservationsLiaisons(const std::list<cObsLiaisons> &,bool IsLastIter,cStatObs & aSO);
        void AddObservationsAppuisFlottants(const std::list<cObsAppuisFlottant> &,bool IsLastIter,cStatObs & aSO);
        void AddObservationsCentres(const std::list<cObsCentrePDV> &,bool IsLastIter,cStatObs & aSO);

        void AddObservationsCamConseq(const cContrCamConseq &  aCCC);


        void AddOneObservationsRelGPS(const cObsRelGPS &);
        void AddObservationsRelGPS(const std::list<cObsRelGPS> & aLO);
        void AddObservationsBaseGpsInit();

        void AddObservationsRigidGrp(const std::list<cObsRigidGrpImage> &,bool IsLastIter,cStatObs & aSO);
        void AddObservationsRigidGrp(const cObsRigidGrpImage &,bool IsLastIter,cStatObs & aSO);

        void AddObservationsRigidBlockCam(const cObsBlockCamRig &,bool IsLastIter,cStatObs & aSO);
        void AddObservationsRigidBlockCam(const std::list<cObsBlockCamRig> &,bool IsLastIter,cStatObs & aSO);

        void AddObservationsContrCamGenInc(const std::list<cContrCamGenInc> &,bool IsLastIter,cStatObs & aSO);
        void AddObservationsContrCamGenInc(const cContrCamGenInc &,bool IsLastIter,cStatObs & aSO);

        double AddAppuisOnePose
              (
                 const cObsAppuis &,cObserv1Im<cTypeEnglob_Appuis> *,
                 std::vector<cRes1OnsAppui> *, cStatObs & aSO,
                 double & aGlobSomErPds, double & aGlobSomPoids
              );


        void  Export(const cSectionExport &);
        void  ExportAttrPose(const cExportAttrPose &);
        void  ExportOrthoCyl(const cExportRepereLoc & anERL,const cExportOrthoCyl & anEOC,const cRepereCartesien & aCRC);
        void  ExportRepereLoc(const cExportRepereLoc &);
        void  ExportPose(const cExportPose &,const std::string & aPref="");
        void  ExportSauvAutom();
        void  ExportCalib(const cExportCalib &);
        void  ExportVisuConfigPose(const cExportVisuConfigGrpPose & anEVCGP);

        void ExportImMM(const cChoixImMM &);
        bool ExportImSecMM(const cChoixImMM &,cPoseCam *,const cMasqBin3D * aMasq3D);

         void ExportMesuresFromCarteProf(const cExportMesuresFromCarteProf&);
         void ExportMesuresFromCarteProf
              (
                 const cExportMesuresFromCarteProf&,const cCartes2Export &
              );
        cSetOfMesureAppuisFlottants StdGetMAF(const std::string &);
        cDicoAppuisFlottant StdGetDAF(const std::string &);

        cAperoPointeMono CreatePointeMono(const cSetOfMesureAppuisFlottants &,const std::string & aNamePt,const cAperoPointeMono * aDef=0);
        cAperoPointeStereo CreatePointeStereo(const cSetOfMesureAppuisFlottants &,const std::string & aNamePt);

        Pt3dr CreatePtFromPointeMonoOrStereo
              (
                    const cSetOfMesureAppuisFlottants & aMAF,
                    const std::string & aNamePt,
                    const cElPlan3D  * aPlan,
                    const std::string & aNameSec="",
                    const Pt3dr * aPDef  = 0
              );

        ElSeg3D   PointeMono2Seg(const cAperoPointeMono &) ;
        Pt3dr     PointeMonoAndPlan2Pt(const cAperoPointeMono &,const cElPlan3D &);
        // Pt3dr     PointeStereo2Pt(const cAperoPointeStereo &);  => CpleIm2PTer


        void ExportMesuresFromCarteProf
             (
                const cExportMesuresFromCarteProf & anEM,
                const cCartes2Export &              aC,
                cElNuage3DMaille *                  aNuage,
                const ElPackHomologue &             aPackH,
                cGenPoseCam *                          aPose,
                const std::string &                 aName
             );

       void ExportRedressement(const cExportRedressement & );
       void ExportRedressement(const cExportRedressement &,cPoseCam & aPC );

       
       void ExportNuage(const cExportNuage &);
       void ExportBlockCam(const cExportBlockCamera &);



        void  SauvDataGrid(Im2D_REAL8 anIm,const std::string & aName);

	void  ExportFlottant(const cExportPtsFlottant & anEPF);

        cSolBasculeRig BasculePoints (const cBasculeOnPoints &,
                            cSetName &            aSelectorEstim,
                            cElRegex &            aSelectorApply
             );
        void BasculePlan (const cBasculeLiaisonOnPlan &,
                            cSetName &            aSelectorEstim,
                            cElRegex &            aSelectorApply
             );
         cElPlan3D EstimPlan
         (
                const cParamEstimPlan &       aPEP,
                cSetName &                    aSelectorEstim,
                const char *                  anAttr
         );

       void AddCamsToMTR();

       void FixeEchelle(const cFixeEchelle &);
       double StereoGetDistFE(const cStereoFE &);
       

       void FixeOrientPlane(const cFixeOrientPlane &);
       void BasicFixeOrientPlane(const std::string &);
       Pt2dr GetVecHor(const cHorFOP &);

        // Utilitaire de basculement
        Pt3dr CpleIm2PTer(const cAperoPointeStereo &);
        Pt3dr PImetZ2PTer(const cAperoPointeMono &,double aZ);


        void ResetNumTmp(int aNumInit,int aNumNonI);
        void ResetNumTmp(const std::vector<cPoseCam *> &,int aNumI,int aNumNonI);

        void Bascule(const cBasculeOrientation &,bool CalleAfter);

         void UpdateMul(double & aMult,double aNew,bool aModeMin);


        cParamApero     mParam;
        std::string     mDC;
        std::string     mOutputDirectory;
	cInterfChantierNameManipulateur * mICNM;


        cSetEqFormelles  mSetEq;
        cAMD_Interf *    mAMD;
	tDiCal mDicoCalib;
        tDiArgCab mDicoArgCalib; // Pour gerer les calib/pose qui necessitent
                                                                     // une initialisation differeee
        std::vector<cGenPoseCam*> mVecGenPose;

        std::vector<cPosePolynGenCam*> mVecPolynPose;
        std::vector<cPoseCam*> mVecPose;
        std::vector<cPoseCam*> mTimeVP; // Triee selon le temps

        std::map<std::string,cCompiledObsRelGPS *> mMCORelGps;

    // Utilise pour connaitre les poses pour lesquels des images te chargees
    // (lorsque l'on recherche  a affiner les pts mul par re-correl)
        std::vector<cPoseCam*> mVecLoadedPose;

	tDiPo    mDicoPose;
	tDiPoGen mDicoGenPose;
	tDiPo    mDPByNum;

        tDiLia  mDicoLiaisons;
        std::map<std::string,tPackAppuis *> mDicoAppuis;
        std::map<std::string,tPackOrient *> mDicoOrient;
        std::map<std::string,tPackCentre *> mDicoCentre;
        std::set<std::string>               mSetFileDist;



	std::map<std::string,cSurfParam *>  mDicoSurfParam;


	std::set<std::string>  mLSymbs;

	std::map<std::string,cBdAppuisFlottant *>  mDicPF;

        tGrApero  mGr;

        std::list<cOneVisuPMul *>   mVisuPM;

        cStdMapName2Name   *  mMapMaskHom;

        int                   mNbRotInit;
        int                   mNbRotPreInit;
        std::string           mSymbPack0;
        int                   mProfInit;
        int                   mProfMax;

        bool                  mModeMaping;
        bool                  mShowMes;
        std::string           mLogName;

        bool                  mHasLogF;
        ofstream               mLogFile;

        cParamEtal *          mParamEtal;

        cMTActive *          mMTAct;
        cMTResult  *         mMTRes;

        cElRegex *           mAutomTracePose;

        std::map<std::string,cCompFilterProj3D *> mMapFilters;

        std::set<std::pair<cGenPoseCam *,cGenPoseCam *> > mSetLinkedCamCam;

        const cEtapeCompensation * mCurEC;
        bool                       mIsLastIter;
        bool                       mIsLastEtape;
        bool                       mIsLastEtapeOfLastIter;
        double                     mScoreLambda0;
        double                     mScoreLambda1;

        std::map<std::string,cLayerImage *> mMapLayers;

       
        cShowPbLiaison *                    mCurPbLiaison;
        int                                 mNbEtape;
        int                                 mNbIterDone;
        int                                 mNbIterTot;

        std::vector<Pt3dr>                  mResiduCentre;
        std::vector<double>                 mRetardGpsC;

 
        const cSectionLevenbergMarkard *    mCurSLMGlob;
        double                              mMulSLMGlob;
        const cSectionLevenbergMarkard *    mCurSLMEtape;
        double                              mMulSLMEtape;
        const cSectionLevenbergMarkard *    mCurSLMIter;
        double                              mMulSLMIter;

        std::map<std::string,cRelEquivPose *>   mRels;
        std::map<std::string,cImplemBlockCam *> mBlockCams;
        bool                                    mHasBlockCams;
        std::map<std::string,cAperoOffsetGPS *> mDicoOffGPS;
        int                                     mNumSauvAuto;

         
        FILE *                                 mFpRT;  // File Rapport Txt
        FILE *                                 mFileDebug;  // File Rapport Txt
        cMajickChek                            mMajChck;
        int                                    mCptIterCompens;
        bool                                   mHasEqDr;
        cStatObs                               mStatLastIter;
             // flag utilise lorque l'on a utilise ori non ortho
        int                                    mSqueezeDOCOAC;  
        cXmlSauvExportAperoGlob                mXMLExport;
        const cXmlSLM_RappelOnPt *             mXmlSMLRop;
        const cXmlPondRegDist *                mCurXmlPondRegDist;
        const cExportSensibParamAero *         mESPA;

        
    public :
         double                                mWorstRes;
         double                                mWorstPerc;
         cGenPoseCam *                         mPoseWorstRes;
         cGenPoseCam *                         mPoseWorstPerc;

         // Stat sur les Pts flottants 
         double mNbPtsFlot;
         double mMaxDistFlot;
         double mSomDistFlot;
         double mSomDistXYFlot;
         Pt3dr  mSomEcPtsFlot;
         Pt3dr  mSomAbsEcPtsFlot;
         Pt3dr  mSomRmsEcPtsFlot;
         Pt3dr  mMaxAbsEcPtsFlot;

         // Utilise pour genere les identifiant permettant d'interpreter les incertitudes
         int   mNumCalib;
         std::vector<std::string> mNamesIdCalib;
         int   mNumImage;
         std::vector<std::string> mNamesIdIm;


         std::map<std::string,cCompile_BDD_NewPtMul *>  mDicoNewBDL;
         std::vector<cCompile_BDD_NewPtMul *>           mVectNewBDL;
         // std::vector<cGenPoseCam*>                      mVCamNewB;
         // cManipPt3TerInc *                              mGlobManiP3TI;  pas la bonne voie
         std::map<std::string,cAccumResidu *> mMapAR;

         std::string mDirExportImRes;
         FILE *      mFileExpImRes;
         int         mLevStaB;

         // DebugVecElimTieP
         bool  mUseVDETp;
         std::vector<int>  mNumsVDETp;
         int               mDebugNumPts;
         const cRappelPose * mRappelPose;
};

#define ADDALLMAJ(aMes) AddAllMajick(__LINE__,__FILE__,aMes)

#endif //  _APERO_H_


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
