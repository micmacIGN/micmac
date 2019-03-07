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


#ifndef _NewRechMATCH_IMAGE_H_
#define _NewRechMATCH_IMAGE_H_


// AFM :  Appli Fits Match

ElSimilitude SimilRobustInit(const ElPackHomologue & aPackFull,double aPropRan,int aNbTir);

class cIndexCodeBinaire; // Permet de retrouver rapidement les element ayant peu de bits differents avec un pt car donne
class cCdtCplHom; // Stocke une hypothese de deux pts car apparie le Master(M) et le secondaire
class cSetOPC; // Un ensemble pt car, incluant un index binaire
class cAFM_Im; // Class maitre  des image , master ou secondaire
class cAFM_Im_Master; // Specialisation master
class cAFM_Im_Sec ;   // Specialisation secondaire
class cAppli_FitsMatch1Im; // class application


//============================================================

class cIndexCodeBinaire
{
    public :
         cIndexCodeBinaire (const cCompCB &);
         const std::vector<cCompileOPC *> & VectVois(const cCompileOPC & aPC);
         void Add(cSetOPC &,const cFitsOneLabel &);
    private :
         void Add(cCompileOPC * anOpc);
         int                      mNBBTot;
         int                      mNBBVois;
         const std::vector<int> * mFlagV;
         std::vector<std::vector<cCompileOPC *> > mVTabIndex;
};

// Stocke une hypothese de deux pts car apparie le Master(M) et le secondaire
class cCdtCplHom
{
    public :
       cCdtCplHom(cCompileOPC * aPM,cCompileOPC * aPS,double aCorr,int aShift) :
           mPM    (aPM),
           mPS    (aPS),
           mCorrel  (aCorr),
           mShift (aShift),
           mOk    (true),
           mDistS (0)
       {
       }

       Pt2dr PM() const {return mPM->mOPC.Pt();}
       Pt2dr PS() const {return mPS->mOPC.Pt();}
       cCompileOPC * mPM;
       cCompileOPC * mPS;
       double        mCorrel;
       int           mShift;
       bool          mOk;
       double        mDistS; // Dist / a la predic simil
};

// Foncteur pour mettre le cCdtCplHom dans un quod tree
class cPtFromPCC
{
   public :
       Pt2dr operator() (cCdtCplHom * aCC) { return aCC->PM(); }
};

typedef ElQT<cCdtCplHom*,Pt2dr,cPtFromPCC> tQtCC ;


// 
class cSetOPC
{
    public :
       // Initialise avec les 
       void InitLabel(const cFitsOneLabel &,const cSeuilFitsParam&,bool DoIndex);
       cIndexCodeBinaire & Ind();
       const cFitsOneLabel &  FOL() const;
       const cSeuilFitsParam &  Seuil() const;
       cSetOPC();
       ~cSetOPC();
       const std::vector<cCompileOPC*> &  VOpc() const;
       std::vector<cCompileOPC*> &  VOpc() ;
       void Add(cCompileOPC*);
       cCompileOPC& At(int aK);
        
       void ResetMatch();

    private :
       std::vector<cCompileOPC*>  mVOpc;
       cIndexCodeBinaire  *   mIndexCB;
       const cFitsOneLabel *  mFOL;
       const cSeuilFitsParam *  mSeuil;
};

class cPrediCoord
{
     public :
        cPrediCoord(Pt2di aSzGlob,int  aNbPix);
        void  Init(double aMulDist,cElMap2D * aMap,const std::vector<cCdtCplHom> aVC);
        Pt2dr Predic(const Pt2dr &) const;
        double Inc(const Pt2dr &) const;
     private :
         Pt2di                mSzGlob;
         double               mFacRed;
         cElMap2D *           mMap;
         Pt2di                mSzRed;
         Im2D_REAL8           mImX;
         TIm2D<REAL8,REAL8>   mTImX;
         Im2D_REAL8           mImY;
         TIm2D<REAL8,REAL8>   mTImY;
         Im2D_REAL8           mImPds;
         TIm2D<REAL8,REAL8>   mTImPds;

         Im2D_REAL8           mImInc;
         TIm2D<REAL8,REAL8>   mTImInc;
};


class cAFM_Im
{
     public :
         friend class cAFM_Im_Master;
         friend class cAFM_Im_Sec;
         friend class cAppli_FitsMatch1Im;

         cAFM_Im (const std::string  &,cAppli_FitsMatch1Im &);
         ~cAFM_Im ();
         void LoadLab(bool DoIndex,bool Glob,eTypePtRemark aLab,bool MaintainIfExist);
         const std::string & NameIm() const;

         void ResetMatch();

     protected :
         cAppli_FitsMatch1Im & mAppli;
         std::string mNameIm;
         cMetaDataPhoto mMTD;
         Pt2di       mSzIm;
         // cSetPCarac                             mSetPC;
         std::vector<cSetOPC*>                  mVSetCC;
         std::vector<cSetOPC*>                  mSetInd0; // decision rapide sur l'overlap
         // std::vector<cCompileOPC> &             mVIndex;

};


class cAFM_Im_Master : public  cAFM_Im
{
     public :
         cAFM_Im_Master (const std::string  &,cAppli_FitsMatch1Im &);
         void MatchOne(bool OverLap,cAFM_Im_Sec & , cSetOPC & ,cSetOPC & ,std::vector<cCdtCplHom> & ,int aNbMin);
         bool MatchLow(cAFM_Im_Sec & anISec,std::vector<cCdtCplHom> & aVCpl);


         bool             MatchGlob(cAFM_Im_Sec &);

         void FiltrageSpatialGlob(std::vector<cCdtCplHom> & aVCpl,int aNbMin);


         // std::vector<std::vector<cCompileOPC *> > mVTabIndex;
         void FilterVoisCplCt(std::vector<cCdtCplHom> & aV);
         void RemoveCpleQdt(cCdtCplHom &);

         ElSimilitude   RobusteSimilitude(std::vector<cCdtCplHom> & aV0,double aDistSeuilNbV);

         cPtFromPCC  mArgQt;
         tQtCC   mQt;

         cPrediCoord        mPredicGeom;
};


class cAFM_Im_Sec : public  cAFM_Im
{
     public :
         cAFM_Im_Sec (const std::string  &,cAppli_FitsMatch1Im &);
         void LoadLabsLow(bool AllLabs);
         bool mAllLoaded;
};


class cAppli_FitsMatch1Im
{
     public :
          cAppli_FitsMatch1Im(int argc,char ** argv);
          const std::string &   ExtNewH () const {return    mExtNewH;}
          const cFitsParam & FitsPm() const {return mFitsPm;}
          std::string NameCple(const std::string & aN1,const std::string & aN2) const;
          int NbBIndex() const;
          int ThreshBIndex() const;
          Pt2di  NbMaxS0() const;
          bool  ShowDet() const;
          eTypePtRemark    LabInit() const;
          bool DoFiltrageSpatial() const;
          double   SeuilCorrelRatio12() const;
          double   SeuilGradRatio12() const;
          double   SeuilDistGrad() const;
          double   ExposantPdsDistGrad() const;
// Dist "a la sift"
          double DistHistoGrad(cCompileOPC & aMast,int aShift,cCompileOPC & aSec);

          bool LabInInit(eTypePtRemark) const;

          void SetCurMapping(cElMap2D * aMap);
          cElMap2D & CurMapping();

          bool HasFileCple() const;
          bool InSetCple(const std::string & aStr) const;
     private :
          cFitsParam         mFitsPm;
          std::string        mNameMaster;
          std::string        mPatIm;
          cElemAppliSetFile  mEASF;
          cAFM_Im_Master *   mImMast;
          cAFM_Im_Sec *      mCurImSec;
          std::string        mNameXmlFits;
          std::string        mExtNewH;
          std::string        mSH;
          std::string        mPostHom;
          bool               mExpTxt;
          // int                mNbBIndex;
          // int                mThreshBIndex;
          bool               mOneWay;
          bool               mSelf;
          bool               mShowDet;
          bool               mCallBack;
          Pt2di              mNbMaxS0;  // Nb max en presel x=> pour overlap en point a analyser, y=> pour modele 3D, y en point voulu
          eTypePtRemark      mLabInit;
          bool               mDoFiltrageSpatial;
          int                mFlagLabsInit;
          cElMap2D *         mCurMap;
          std::string        mFileCple;
          bool               mHasFileCple;
          std::set<std::string>   mSetCple;
          std::vector<double>     mVSeuils;
};


// transforme les couples apparies en des points homologues classiques
ElPackHomologue PackFromVCC(const  std::vector<cCdtCplHom> &aVCpl);

// compare les couples, pour trier par mDistS decroissante
bool CmpCC(const cCdtCplHom & aC1,const cCdtCplHom & aC2) ;


// Filtre les couples  selon un critere de direction globale
//  1- Evalue cette direction par calcul de l'histogramme (convolue)
//  2- calcule la direction principale
//  3- filtre les points loin de cette direction

void FiltrageDirectionnel(std::vector<cCdtCplHom> & aVCpl);







#endif //  _NewRechMATCH_IMAGE_H_


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
aooter-MicMac-eLiSe-25/06/2007*/
