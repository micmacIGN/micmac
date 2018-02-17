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

#ifndef _ELISE_MERGE_CLOUD
#define _ELISE_MERGE_CLOUD

#include "StdAfx.h"


inline double pAramSurBookingIm() {return 5.0;}
/*
inline double pAramElimDirectInterior() {return 10.0;}
inline double pAramLowRatioSelectIm() {return 0.001;}
inline double pAramHighRatioSelectIm() {return 0.05;}
*/
//
//
//

typedef enum
{
  eLFNoAff,
  eLFMaster,
  eLFMasked,
  eLFBorder
  // Warn si on change le nb de label : stocke pour l'instant sur des 2 bits
  //  eLFTmp
} eLabelFinaux;



//================== HEADER du HEADER ====================

class cAppliMergeCloud;
class cASAMG; // Attribut Sommet cAppliMergeCloud
class cResumNuage; // Classe pour representer un nuage rapidement

class c3AMG; // Attribut ARC cAppliMergeCloud
class c3AMGS; // c3AMG sym

typedef ElSom<cASAMG*,c3AMG*>  tMCSom;
typedef ElArc<cASAMG*,c3AMG*>  tMCArc;
typedef ElGraphe<cASAMG*,c3AMG*>  tMCGr;
typedef ElSubGraphe<cASAMG*,c3AMG*>  tMCSubGr;
typedef cSubGrFlagArc<tMCSubGr>  tMCSubGrFA;
typedef std::pair<tMCSom *,tMCSom *> tMCPairS;
typedef ElArcIterator<cASAMG*,c3AMG*> tArcIter;

class cResumNuage
{
    public :
        void Reset(int aReserve);
        int mNbSom;
        std::vector<INT2> mVX;
        std::vector<INT2> mVY;
        std::vector<INT2> mVNb;
 
        Pt2di PK(const int & aK) const {return Pt2di(mVX[aK],mVY[aK]);}
        
};


class c3AMGS
{
   public :
};
class c3AMG
{
   public :
      c3AMG(c3AMGS *,double aRec);
      const double & Rec() const;
   private :
      c3AMGS * mSym;
      double   mRec;
};

class cASAMG
{
   public :
      cASAMG(cAppliMergeCloud *,cImaMM *);
      void MakeVoisinInit();

      double LowRecouvrt(const cASAMG &) const;
      double Recouvrt(const cASAMG &,const cResumNuage &) const;
      void TestDifProf(const cASAMG & aNE) const;

      cImaMM *     IMM(); 
      const cOneSolImageSec *  SolOfCostPerIm(double aCost);
      const cImSecOfMaster &  ISOM() const;

      void AddCloseVois(cASAMG *);

      void TestImCoher();

      void InspectQual(bool WithClik);
      void InspectEnv();
      INT   MaxNivH() const;
      double QualOfNiv() const;
      int    NbOfNiv() const;
      int    NbTot() const;


      void InitNewStep(int aNiv);
      void FinishNewStep(int aNiv);
      bool IsCurSelectable() const;
      bool IsSelected() const;
      int  NivSelected() const;

      void SetSelected(int aNivSel,int aNivElim,tMCSom * aSom);
      void InitGlobHisto();
      inline void SuppressPix(const Pt2di &, const int & aLab);


      // Valeur >0 si dedans,  <0 dehors, quantifie l'interiorite
      double  InterioriteEnvlop(const Pt2di & aP,double aProfTest,double & aDeltaProf) const;

      std::string ExportMiseAuPoint();
 
      bool  IsImageMAP() const;

      Video_Win *   TheWinIm() const;
      Pt2di         Sz() const;
      double Resol() const;

   private :
     void MakeVec3D(std::vector<Pt3dr> & aVPts,const cResumNuage &) const;
     double Recouvrt(const cASAMG &,const cResumNuage &,const std::vector<Pt3dr> & aVPts) const;

     double QualityProjOnMe(const std::vector<Pt3dr> &,const cResumNuage &) const;
     double QualityProjOnOther(const cASAMG &,const Pt3dr &) const;
     double QualityProjOnMe(const Pt3dr &) const;
     double SignedDifProf(const Pt3dr &) const;
     double DifProf2Gain(double aDif) const;

     void ProjectElim(cASAMG *,Im2D_Bits<1> aMasq,const cRawNuage & aNuage);
     void DoOneTri(const Pt2di & aP0,const Pt2di & aP1,const Pt2di & P2,const cRawNuage &aNuage,const cRawNuage & aMasterNuage);




     inline double DynAng() const ;
     inline bool   CCV4()   const ;
     inline int    CCDist() const ;
     inline int    SeuimNbPtsCCDist() const ;

     void ComputeIncid();
     void ComputeIncidAngle3D();
     void ComputeIncidGradProf();
     void ComputeIncidKLip(Fonc_Num fMasq,double aPenteInPix,int aNumQual);

     void ComputeSubset(int aNbPts,cResumNuage &);
     

     cAppliMergeCloud *   mAppli;
     const cParamFusionNuage & mPrm;
     cImaMM *             mIma;
     std::string          mNameIm;
     cElNuage3DMaille *   mStdN;
     double               mResol;
     Im2D_Bits<1>         mMasqN;
     TIm2DBits<1>         mTMasqN;
     Im2DGen *            mImProf;

     Im2D_U_INT1          mImCptr;
     TIm2D<U_INT1,INT>    mTCptr;
     Pt2di                mSz;
     Im2D_U_INT1          mImIncid;
     TIm2D<U_INT1,INT>    mTIncid;

     Im2D_Bits<4>         mImQuality;
     TIm2DBits<4>         mTQual;

     Im2D_Bits<2>         mImLabFin;
     TIm2DBits<2>         mTLabFin;

     Im2D_Bits<4>         mImEnvSup;
     TIm2DBits<4>         mTEnvSup;
     Im2D_Bits<4>         mImEnvInf;
     TIm2DBits<4>         mTEnvInf;
     Im1D_INT4            mLutDecEnv;
     INT4 *               mDLDE;



     // Im2D_U_INT1          mImQuality;
     // TIm2D<U_INT1,INT>    mTQual;

     // Im2D_U_INT1          mImQuality;
     // TIm2D<U_INT1,INT>    mTQual;



     Im1D<INT4,INT>       mHisto;
     INT *                mDH;
     INT                  mMaxNivH;
     INT                  mNbNivH;
     double               mQualOfNiv;
     int                  mNbOfNiv;
     int                  mNbTot;

     double               mSSIma;

     std::vector<cASAMG *>  mCloseNeigh;

     cResumNuage            mLowRN;  // Basse resolution pour la topologie
     cImSecOfMaster         mISOM;
     int                    mNivSelected;
     bool                   mIsMAP;

     static const int theNbValCompr;
};


//==================================================

class cStatNiv
{
    public :
        cStatNiv();

        double  mGofQ;
        double  mRecTot;
        double  mCumRecT;
        double  mRecMoy;
        double  mNbImPourRec;
        int     mNbIm;
        int     mCumNbIm;
};

class cAppliMergeCloud : public cAppliWithSetImage
{
    public :

/*
       std::string NameFileInput(bool DownScale,const std::string & aNameIm,const std::string & aPost,const std::string & aPref="");
       std::string NameFileInput(bool DownScale,cImaMM *,const std::string & aPost,const std::string & aPref="");
*/



       cAppliMergeCloud
       (
            int argc,
            char ** argv
       );
       const cParamFusionNuage & Param() {return mParam;}
       Video_Win *   TheWinIm(const cASAMG *);

       static int MaxValQualTheo() {return eQC_Coh3;}
       REAL8    GainQual(int aNiv) const;

       tMCSubGr &  SubGrAll();

       bool IsInImageMAP(cASAMG*);
       bool DoPlyCoul() const;
       int  SzNormale() const;
       bool NormaleByCenter() const;
       cMMByImNM *       MMIN();
       
       bool          HasOffsetPly() ;
       const Pt3dr & OffsetPly() ;
       const bool    Export64BPly();
    private :
       tMCArc * TestAddNewarc(tMCSom * aS1,tMCSom *aS2);
       tMCSom * SomOfName(const std::string & aName);
       void AddVoisVois(std::vector<tMCArc *> & aVArc,tMCSom&,tMCSom&);
       void CreateGrapheConx();
       void OneStepSelection();



       std::string mFileParam;
       cParamFusionNuage mParam;
       Video_Win *       mTheWinIm;
       double            mRatioW;

       std::vector<cASAMG *>           mVAttr;
       std::vector<tMCSom *>           mVSoms;
       std::map<std::string,tMCSom *>  mDicSom;
       tMCGr                           mGr;
       int                             mFlagCloseN;
       tMCSubGr                        mSubGrAll;
       tMCSubGrFA                      mSubGrCloseN;
       std::set<tMCPairS>              mTestedPairs;
       INT                             mGlobMaxNivH;
       INT                             mCurNivSelSom;
       INT                             mCurNivElim;
       std::vector<cStatNiv>           mVStatNivs;
/*
       Im1D_REAL8                      mImGainOfQual;
       REAL8 *                         mDataGainOfQual;
       Im1D_INT4                       mNbImOfNiv;
       INT *                           mDataNION;
       Im1D_INT4                       mCumNbImOfNiv;
       INT *                           mDataCNION;
*/
       double                          mRecouvrTot;
       double                          mRecMoy;
       double                          mNbImMoy;
       int                             mNbImSelected;
       cElRegex *                      mPatMAP;
       bool                            mDoPly;
       bool                            mDoPlyCoul;
       int                             mSzNormale;
       bool                            mNormaleByCenter;
       eTypeMMByP                      mModeMerge;
       cMMByImNM *                     mMMIN;
       double                          mDS;
       Pt3dr                           mOffsetPly;

       std::string                     mSH;
       bool                            mDoublePrec;
};

   //==============================================================================

double cASAMG::DynAng() const {return mAppli->Param().ImageVariations().DynAngul();}
bool   cASAMG::CCV4()   const {return mAppli->Param().ImageVariations().V4Vois();}
int    cASAMG::CCDist() const {return mAppli->Param().ImageVariations().DistVois();}
int    cASAMG::SeuimNbPtsCCDist() const  {return 2 * (1+2*CCDist());}



// inline bool pAramComputeIncid() {return false;}

#endif // _ELISE_MERGE_CLOUD

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
