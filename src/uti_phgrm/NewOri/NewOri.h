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

#ifndef _ELISE_NEW_ORI_H
#define _ELISE_NEW_ORI_H

#include "StdAfx.h"
#include "Extern_NewO.h"

#define NbCamTest 6


//================ SEUILS ==============

// Nombre de point pour echantillonner le recouvrt / homogr
#define NbRecHom 40
// Nombre de point minimum pour etudier un couple
#define NbMinPts2Im 20
#define NbMinPts2Im_AllSel 10


//  Sur les triplets

#define  TNbCaseP1  6  // Nombre de case sur lesquelle on discretise
#define  TQuant     30 // Valeur de quantification
#define  TQuantBsH  100 // Valeur de quantification
#define  TBSurHLim  0.15  // Valeur d'attenuation du gain en B/H
//  #define  TNbMinPMul 8  // Nombre de point triple minimal pour un triplet
#define  TAttenDens 3.0

//#define TNbMinTriplet 8    // Nombre de point triple minimal pour un triplet // er: added to cCommonMartiniAppli class
//#define TStdNbMaxTriplet 20   // Nombre maximal de triplet calcule  // er:  added to cCommonMartiniAppli class
//#define TQuickNbMaxTriplet 3   // Nombre maximal de triplet calcule // er:  added to cCommonMartiniAppli class
#define TGainSeuil    5e-3

#define NbMaxATT 100000
#define FactAttCohMed 3.0

//=====================================

class cNewO_OneIm;
class cNewO_OrInit2Im;
class cNewO_NameManager;
class cNewO_Appli;


typedef cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> > tMergeLPackH;
typedef cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal>                     tMergeCplePt;



typedef std::list<tMergeCplePt *>  tLMCplP;
ElPackHomologue ToStdPack(const tMergeLPackH *,bool PondInvNorm,double PdsSingle=0.1);

ElPackHomologue PackReduit(const ElPackHomologue & aPack,int aNbInit,int aNbFin);
ElPackHomologue PackReduit(const ElPackHomologue & aPack,int aNbFin);


class cCommonMartiniAppli
{
    public :


       std::string    mNameOriCalib;
       std::string    mPrefHom;
       std::string    mExtName;
       bool           mExpTxt;
       std::string    mInOri;
       std::string    mOriOut;
       std::string    mOriGPS;
       std::string    mOriCheck;
       bool           mDebug;
       //  std::string    mBlinis;
       bool           mAcceptUnSym;
       bool           mQuick;
       bool           mShow;
       // const std::string &   NameNOMode();
       int mTStdNbMaxTriplet;
       int mTQuickNbMaxTriplet;
       int mTNbMinTriplet;

       eTypeModeNO    ModeNO() const;
       cNewO_NameManager * NM(const std::string & aDir) const;
       LArgMain &     ArgCMA();
       std::string    ComParam();
       cCommonMartiniAppli();
       
       bool GpsIsInit();
       bool CheckIsInit();
       Pt3dr GpsVal(cNewO_OneIm *);
       CamStenope * CamCheck(cNewO_OneIm *);
      
    private :
       LArgMain * mArg;
       mutable bool       mPostInit;
       mutable cNewO_NameManager * mNM;
       std::string    mNameNOMode;
       mutable eTypeModeNO   mModeNO;

       void PostInit() const;
       cCommonMartiniAppli(const cCommonMartiniAppli &) ; // N.I.
};


class cNewO_OneIm
{
    public :
            cNewO_OneIm
            (
                 cNewO_NameManager & aNM,
                 const std::string  & aName,
                 bool  WithOri = true
            );

            CamStenope * CS();
            const std::string & Name() const;
            const cNewO_NameManager&  NM() const;
            cNewO_NameManager&  NM() ;
    private :
            cNewO_NameManager*  mNM;
            CamStenope *        mCS;
            std::string         mName;
};

class cNOCompPair
{
    public :
       cNOCompPair(const Pt2dr & aP1,const Pt2dr & aP2,const double & aPds);

       Pt2dr mP1;
       Pt2dr mP2;
       double mPds;
       double mLastPdsOfErr;
       Pt3dr  mQ1;
       Pt3dr  mQ2;
       Pt3dr  mQ2R;
       Pt3dr  mU1vQ2R;
};



double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2);


class cNewO_OrInit2Im
{
    public :
          cNewO_OrInit2Im
          (
                bool GenereOri,
                bool aQuick,
                cNewO_OneIm * aI1,
                cNewO_OneIm * aI2,
                tMergeLPackH *      aMergeTieP,
                ElRotation3D *      aTestSol,
                ElRotation3D *      aInOri,
                bool                Show,
                bool                aHPP,
                bool                aSelAllIm,
                cCommonMartiniAppli &
          );

          double ExactCost(const ElRotation3D & aRot,double aTetaMax) const;
          double PixExactCost(const ElRotation3D & aRot,double aTetaMax) const;
          const cXml_Ori2Im &  XmlRes() const;
          void DoExpMM();
          void DoExpMM(cNewO_OneIm *,const ElRotation3D &,const Pt3dr & aPMed);

    private :


           void TestNewSel(const ElPackHomologue & aPack);
          
       //======== Amniguity ====
            void CalcAmbig();
            void CalcSegAmbig();
            ElRotation3D  SolOfAmbiguity(double aTeta);

            Pt3dr CalcBaseOfRot(ElMatrix<double> aMat,Pt3dr aTr0);
            Pt3dr OneIterCalcBaseOfRot(ElMatrix<double> aMat,Pt3dr aTr0,double & anErMoy);
            Pt2dr ToW(const Pt2dr & aP) const;
            void ShowPack(const ElPackHomologue & aPack,int aCoul,double aRay);
            void ClikIn();
            double RecouvrtHom(const cElHomographie & aHom);


       //===================
          void  AddNewInit(const ElRotation3D & aR);
          // double DistRot(const ElRotation3D & aR1,const ElRotation3D & aR2) const;


          double CostLinear(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const;
          double CostLinear(const ElRotation3D & aRot,const Pt3dr & aP1,const Pt3dr & aP2,double aTetaMax) const;

          void TestCostLinExact(const ElRotation3D & aRot);
          void AmelioreSolLinear(ElRotation3D  aRot,const std::string & aMes);
          double ExactCost (const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const;

          double FocMoy() const;



          double            mPdsSingle;
          bool              mQuick;
          cNewO_OneIm *     mI1;
          cNewO_OneIm *     mI2;
          tMergeLPackH *    mMergePH;
          ElRotation3D *    mTestC2toC1;
          ElPackHomologue   mPackPDist;
          ElPackHomologue   mPackPStd;
          Pt2dr             mPInfI1;
          Pt2dr             mPSupI1;
          ElPackHomologue   mPackStdRed;
          ElPackHomologue   mPack150;
          ElPackHomologue   mPack30;
          

     // Resolution lineraire
          int                      mNbCP;
          double                   mErStd;
          std::vector<cNOCompPair> mStCPairs;
          std::vector<cNOCompPair> mRedCPairs;
          L2SysSurResol            mSysLin5;
          L2SysSurResol            mSysLin2;
          L2SysSurResol            mSysLin3;
          cInterfBundle2Image *    mLinDetIBI;
          cInterfBundle2Image *    mBundleIBI;
          cInterfBundle2Image *    mBundleIBI150;
          cInterfBundle2Image *    mRedPvIBI;
          cInterfBundle2Image *    mFullPvIBI;

          bool                     mShow;

          double mCurLamndaLVM;
       

          ElRotation3D  mBestSol;
          double        mCostBestSol;
          bool          mBestSolIsInit;
          double        mBestErrStd;
          std::vector<double> mResidBest;
          std::vector<double> mCurResidu;

     // Ambiguite
          Pt3dr         mDirAmbig;
          ElSeg3D       mSegAmbig;
          Pt3dr         mIA;  // Intersetion
     // ===============================
          Video_Win *   mW;
          Pt2dr         mP0W;
          double        mScaleW;
          cXml_Ori2Im   mXml;
          bool          mSelAllIm;
};



class cNewO_NameManager : public cVirtInterf_NewO_NameManager
{
     public :
           cNewO_NameManager
           (
               const std::string  & anExt, // => mis en premier pour forcer la re-compile
               const std::string  & aPrefHom, // => mis en premier pour forcer la re-compile
               bool  Quick,
               const std::string  & aDir,
               const std::string  & anOri,
               const std::string  & PostTxt,
               const std::string  & anOriOut=""  // Def => Martini / MartiniGin
           );
           CamStenope * CamOfName(const std::string & aName) const;
           ElPackHomologue PackOfName(const std::string & aN1,const std::string & aN2) const;
           std::string NameOriOut(const std::string & aNameIm) const;

           std::string KeySetCpleOri() const ;
           std::string KeyAssocCpleOri() const ;


           std::string NameXmlOri2Im(const std::string & aN1,const std::string & aN2,bool Bin) const;
           std::string NameXmlOri2Im(cNewO_OneIm* aI1,cNewO_OneIm* aI2,bool Bin) const;

           cXml_Ori2Im GetOri2Im(const std::string & aN1,const std::string & aN2) const;

           std::string  NameTimingOri2Im() const;
           const std::string & Dir() const;

           // 
           CamStenope * CamOriOfName(const std::string & aName,const std::string & anOri);
           CamStenope * CamOriOfNameSVP(const std::string & aName,const std::string & anOri);
           const std::string &  OriCal() const;
           const std::string &  OriOut() const;
           cInterfChantierNameManipulateur *  ICNM();


           // Dand cNewO_PointsTriples.cpp , a cote de cAppli_GenPTripleOneImage::GenerateHomFloat
           std::string Dir3P(bool WithMakeDir=false) const;
           std::string Dir3POneImage(cNewO_OneIm *,bool WithMakeDir=false) const;
           std::string Dir3POneImage(const std::string & aName,bool WithMakeDir=false) const;


           // Liste des image tels que  N3-N1 et N3-N2 soient oriente
           std::list<std::string > ListeCompleteTripletTousOri(const std::string & aN1,const std::string & aN2) const;


           std::string NameTripletsOfCple(cNewO_OneIm *,cNewO_OneIm *,bool Bin);
           std::string Dir3PDeuxImage(cNewO_OneIm *,cNewO_OneIm *,bool WithMakeDir=false);
           std::string Dir3PDeuxImage(const std::string&,const std::string&,bool WithMakeDir=false);
           std::string NameHomFloat(cNewO_OneIm * ,cNewO_OneIm * );
           std::string NameHomFloat(const std::string&,const std::string&);

           std::string NameListeImOrientedWith(const std::string &,bool Bin) const;
           std::string RecNameListeImOrientedWith(const std::string &,bool Bin) const;
           std::list<std::string>  ListeImOrientedWith(const std::string & aName) const;
           std::list<std::string>  Liste2SensImOrientedWith(const std::string & aName) const; // Ajoute Rec

           CamStenope * OutPutCamera(const std::string & aName) const;
           CamStenope * CalibrationCamera(const std::string  & aName) const;
           std::pair<CamStenope*,CamStenope*> CamOriRel(const std::string & aN1,const std::string & aN2) const;

           cResVINM  ResVINM(const std::string &,const std::string &) const;
           // L'orientation Cam2Monde de 2 sur 1
           ElRotation3D OriCam2On1(const std::string & aN1,const std::string & aN2,bool &OK) const;
           std::string NameListeCpleOriented(bool Bin) const;
           std::string NameListeCpleConnected(bool Bin) const;
           std::string NameRatafiaSom(const std::string & aName,bool Bin) const;

           // Orientation d'un triplets a partir d'un Ori existante, convention Martini
           std::pair<ElRotation3D,ElRotation3D> OriRelTripletFromExisting
                                                (
                                                    const std::string & Ori,
                                                    const std::string & aNameI1,
                                                    const std::string & aNameI2,
                                                    const std::string & aNameI3,
                                                    bool & Ok
                                                );

           void LoadHomFloats(std::string,std::string,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,bool SVP=false);
           void LoadHomFloats(cNewO_OneIm * ,cNewO_OneIm *,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2);
           void GenLoadHomFloats(const std::string &  aNameH,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,bool SVP);

           std::string NameHomTriplet(cNewO_OneIm *,cNewO_OneIm *,cNewO_OneIm *,bool WithMakeDir=false);
           std::string NameHomTriplet(const std::string&,const std::string&,const std::string&,bool WithMakeDir=false);

           std::string NameOriInitTriplet(bool ModeBin,cNewO_OneIm *,cNewO_OneIm *,cNewO_OneIm *,bool WithMakeDir=false);
           std::string NameOriOptimTriplet(bool ModeBin,cNewO_OneIm *,cNewO_OneIm *,cNewO_OneIm *,bool WithMakeDir=false);
           std::string NameOriOptimTriplet(bool ModeBin,const std::string&,const std::string&,const std::string&,bool WithMakeDir=false);

           std::string NameOriGenTriplet(bool Quick,bool ModeBin,cNewO_OneIm *,cNewO_OneIm *,cNewO_OneIm *);

           std::string NameTopoTriplet(bool ModeBin);
           std::string NameCpleOfTopoTriplet(bool ModeBin);


           bool LoadTriplet(cNewO_OneIm * ,cNewO_OneIm *,cNewO_OneIm *,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,std::vector<Pt2df> * aVP3);
           bool LoadTriplet(const std::string &,const std::string &,const std::string &,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,std::vector<Pt2df> * aVP3);
           

           void WriteTriplet(const std::string & aNameFile,tCVP2f &,tCVP2f &,tCVP2f &,tCVUI1 &);
           void WriteCouple(const std::string & aNameFile,tCVP2f &,tCVP2f &,tCVUI1 &);

     private :

           void WriteTriplet(const std::string & aNameFile,tCVP2f &,tCVP2f &,tCVP2f *,tCVUI1 &);




           std::string NameAttribTriplet(const std::string & aPrefix,const std::string & aPost,cNewO_OneIm *,cNewO_OneIm *,cNewO_OneIm *,bool WithMakeDir=false);
           std::string NameAttribTriplet(const std::string & aPrefix,const std::string & aPost,const std::string & aN1,const std::string & aN2,const std::string & aN3,bool WithMakeDir=false);


           cInterfChantierNameManipulateur *  mICNM;
           std::string                        mDir;
           std::string                        mPrefOriCal;
           std::string                        mPostHom;
           std::string                        mPrefHom;
           std::string                        mExtName;
           // std::map<std::string,CamStenope *> mDicoCam;
           static const std::string           PrefixDirTmp;
           std::string                        mDirTmp;
           std::string                        mPostfixDir;
           bool                               mQuick;
           std::string                        mOriOut;
};



template <const int TheNb> void NOMerge_AddPackHom
                           (
                                cStructMergeTieP< cFixedSizeMergeTieP<TheNb,Pt2dr,cCMT_NoVal> > & aMap,
                                const ElPackHomologue & aPack,
                                const ElCamera & aCam1,int aK1,
                                const ElCamera & aCam2,int aK2
                           );

template <const int TheNb> void NOMerge_AddAllCams
                           (
                                cStructMergeTieP< cFixedSizeMergeTieP<TheNb,Pt2dr,cCMT_NoVal> >  & aMap,
                                std::vector<cNewO_OneIm *> aVI
                           );







extern Pt3dr MedianNuage(const ElPackHomologue & aPack,const ElRotation3D & aRot);
ElMatrix<double> TestMEPCoCentrik(const ElPackHomologue & aPack,double aFoc,const ElRotation3D * aRef,double & anEcart);

void AddSegOfRot(std::vector<Pt3dr> & aV1,std::vector<Pt3dr> & aV2,const ElRotation3D & aR,const Pt2df &  aP);
double Residu(cNewO_OneIm  * anIm , const ElRotation3D & aR,const Pt3dr & aPTer,const Pt2df & aP);

class  cResIPR
{
    public :
         std::vector<int> mVSel;
         double           mMoyDistNN;
};

cResIPR cResIPRIdent(int aNb);

cResIPR  IndPackReduit(const std::vector<Pt2df> & aV,int aNbMaxInit,int aNbFin);
cResIPR  IndPackReduit(const std::vector<Pt2df> & aV,int aNbMaxInit,int aNbFin,const cResIPR & aResExist,const std::vector<Pt2df> & aVPtsExist);



extern const std::string TheStdModeNewOri;
eTypeModeNO ToTypeNO(const std::string &);

class cExeParalByPaquets
{
    public :
          cExeParalByPaquets(const std::string & aMes,int anEstimNbCom);
          void AddCom(const std::string & aCom);
          ~cExeParalByPaquets();

    private :
          void    ExeCom();
          ElTimer mChrono;
          std::list<std::string> mLCom;
          std::string            mMes;
          int                    mEstimNbCom;
          int                    mCpt;
          int                    mNbInOnePaquet;
};

CamStenope * DefaultCamera(const std::string & aName);

void TestEllips_3D();


#endif // _ELISE_NEW_ORI_H

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
