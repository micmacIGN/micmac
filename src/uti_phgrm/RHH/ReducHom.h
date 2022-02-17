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

#if (1)  // Pour l'instant tout dans un namespace
#define  NS_RHH_BEGIN namespace RHH{
#define  NS_RHH_END };
#define  NS_RHH_USE using namespace RHH;

#else
#define  NS_RHH_BEGIN
#define  NS_RHH_END
#define  NS_RHH_USE
#endif



NS_RHH_BEGIN

#ifndef _ELISE_REDUC_HHH_
#define _ELISE_REDUC_HHH_

/*
   Pour chaque paire  P1,I1  P2,I2

   A-  Si I1 (P1) et I2(P2) n'existent pas , on cree un nouveau point mult
       vu par P1 et P2 (rajout symetrique);

   B- Si I1(P1) exist en M1 et pas I2(P2) on rajoute  P2,I2 au point multiple,
      sauf si M1(I2) existe car alors incoherence


   C- Si I1(P1) et I2(P2) existent et c'est le meme point multiple,
      on incremente le compteur  d'arc du point multiple et c'est tout;

   D- Si I1(P1) et I2(P2) existent sur des point multiples differents  M1
      et M2:

       D-1 : Si    Image(M1 ) Inter Image(M2) = Vide alors
            on fusionne M1 et M2

       D-2 : Sinon une incoherence est detectee, pour l'instant on ne
             fait rien, il sont tout les deux marques comme incoherent


*/



#define TEST 1

#if TEST
#define NB_RANSAC_H 20
#else
#define NB_RANSAC_H 200
#endif

void WarnTest();

class cPtHom;    // Point final multiple
class cLink2Img; // "arc" du graphe de visibilite
class cImagH;    // Une image
class cPhIndexed;     // Pt Hom en cours de traitement
class cIndexImag;    // Structure temporaire pour analyser les point d'une nouvelle image
class cAppliReduc;    // Structure temporaire pour analyser les point d'une nouvelle image

typedef std::map<cImagH *,cLink2Img *> tMapName2Link;

  //======================================
  //           cPtHom
  //======================================


/*
    cPtHom  : point multiple,  composante connexe resultante du graphe des points de
              liaisons,   pour chaque image il contient  un vecteur
              des points 2D ou il est mesure (unique qd pas d'incoherence)
*/

class cPtHom
{
    public :
         cPtHom();

         //  Free the objet, and put it in the list of free objtc "mReserve" for future allocation
         void  Recycle();

         // static constructor, create an object from a single tie point
         static cPtHom * NewGerm(cImagH * aI1,const Pt2dr & aP1,cImagH* aI2,const Pt2dr & aP2);

         // Used for udate mCptArc, which apparently is only used for tuning in ShowAll
         void IncrCptArc();
         // Update the Pt Hom to take into account that in I2, it was observed at P2
         // Detect an incoherence as soons as several mesures are done in I2
         // Memorize all the measurment ddOnePtUnique
         void AddMesureInImage(cImagH * aI2,const Pt2dr & aP2);

         // Number of image where the PtHom is observed
         int NbIm() const;

         // Merge this and H2, Recycle H2, memorize if the merge is coherent
         bool OkAbsorb(cPtHom * H2);

         static void ShowAll();
    private :

         // Allocate by searcchinh in mReserve or new
         static cPtHom * Alloc();


         cPtHom(const cPtHom &); // N.I.
         void  Clear();

         static std::list<cPtHom  *> mReserve;
         static std::list<cPtHom  *> mAllExist;  // Used for tuning (Show All)

         std::map<cImagH*,std::vector<Pt2dr> >  mMesures;
         bool                     mCoherent;
         int                      mCptArc;
};


/*
    HOMOGRAPHY :

     The objective of the homography computation is to compute for each image,
   a homography from the image to a common ground. It is named :

         *  mHi2t  (i2t = Image to Terrain, Terrain=ground in french)


      This homography for each images has to be computed from the
   homograohy between pair of images :

      Let HA2T et HB2T  be the homgraphy from images A,B to Ground
      Let  HA2B be the homgraphy from A to B (computed from Tie point, the mHom12 from
      cLink2Img),
  We have :

           HA2T (P)  =  HB2T (HA2B(P))   or     HA2T = HB2T o  HA2B  (1)

      Equation (1) has to be solved globally, there is N unknown and N (N-1)/2 equation.

      However, it obviouly undertermined up to a global homography

*/



  //======================================
  //           cLink2Img
  //           cImagH
  //======================================

/*
  Represent the link between two images :
    - homologous point
    - homographie (posssibly)
    - distribution of point

*/

class cLink2Img  // dans cImagH.cpp
{
    public :
         cLink2Img(cImagH * aSrce,cImagH * aDest,const std::string & aNameH);
         double CoherenceH() ;

         cImagH * Srce() const;
         cImagH * Dest() const;
         const std::string & NameH() const;
         cElHomographie CalcSrceFromDest();

         // int   & NbPtsAttr();  Je sais plus a quoi ce devait servir


         // Dependant de homogra
         const double &         QualHom() const;
         const bool &           OkHom() const;
         const cElHomographie & Hom12() const;
         cElHomographie & Hom12() ;

         // Dependant de PackHom
         const int   & NbPts() const;
         const ElPackHomologue & Pack() const;
         ElPackHomologue & Pack() ;
             // Obtained by GetDistribRepresentative from util/pt2di.cpp
             // simply by averaging points on a regular grid
             // list of   Pt2dr+weight , represent the distribution of the points
         const std::vector<Pt3dr> & EchantP1() const;

         cEqHomogFormelle * &  EqHF();

         std::string NameComHomogr() const;
         void LoadComHomogr();
         double PdsEchant() const;
    private :
    
       void LoadPtsHom();
       void LoadStatPts(bool ExigOk); // Hom, Ech, Cdg, Nb ....
       void LoadXmlHom(const cXmlRHHResLnk & aXml);

       // Gestion des noms
       std::string NameHomol() const;
       std::string NameXmlHomogr() const;

       // 2 imposteurs sur les const
       void LoadPtsHom() const;
       void LoadStatPts(bool ExigOk) const;

        cLink2Img(const cLink2Img &) ; // N.I.
        int      mNbPts;
        int      mNbPtsAttr;
        cImagH * mSrce;
        cImagH * mDest;
        cAppliReduc &    mAppli;
        std::string mNameH;
        double      mQualHom;
        //
        cElHomographie mHom12;
        bool            mPckLoaded;
        bool            mHomLoaded;
        bool            mOkHom;
        ElPackHomologue mPack;

        std::vector<Pt3dr> mEchantP1;
        Pt3dr              mPRep1;
        cEqHomogFormelle * mEqHF;
};

class cTestPlIm
{
    public :
        cTestPlIm(cLink2Img * aLnk,cElemMepRelCoplan * aRMCP,bool Show,double aEps) ;

        cLink2Img *               mLnk;
        cElemMepRelCoplan*        mRMCP;
        double                    mResiduH; // must be initialized before mHomI2T, see cTestPlIm::cTestPlIm
        cElHomographie            mHomI2T;
        bool                      mOk;
    private :
        // cTestPlIm(const cTestPlIm&);  // N.I.
};


class cImagH
{
     public :
// PRE REQUIS POUR LE MERGING
//=====================

        void TestEstimPlDirect();
        void Close();

        cLink2Img * GetLinkOfImage(cImagH*);


         cImagH(const std::string & aName,cAppliReduc &, int aNum);
         void AddLink(cImagH *,const std::string & aNameH);
         const std::string & Name() const;
         const std::string & NameCalib() const;
         const std::string & NameVerif() const;

         void ComputePts();

         void SetPHom(const Pt2dr & aP,cPtHom *);
         void ComputeLnkHom();
         std::string EstimatePlan();

         void SetMarqued(int);
         void SetUnMarqued(int);
         bool Marqued(int) const;
         // std::vector<cImagH *> AdjRefl();  // Image adj + lui meme


         void AddComCompHomogr(std::list<std::string> & aLCom);
         void LoadComHomogr();


        static void VoisinsNonMarques(const std::vector<cImagH*> & aIn,std::vector<cImagH*> & aV,int aFlagN,int FlagT );
        void   VoisinsMarques(std::vector<cLink2Img*> & aVois,int aFlagN);

         cElHomographie &     Hi2t() ;  // 
         cElHomographie &     HTmp() ;  // 
         cElHomographie &     H2ImC() ;  // 

         cAppliReduc &    Appli();
         int & NumTmp();
         cHomogFormelle *  & HF();
         const tMapName2Link & Lnks() const;
         CamStenope *  CamC();
         std::string NameOriHomPlane() const;
         const std::vector<cLink2Img*> &  VLink() const;
         cEqOneHomogFormelle * &  EqOneHF();
         bool  &                    C2CI();  // Connected to Center Image
         void AddViscositty(double aPds);
         double PdsEchant() const;
         double & GainLoc();
         bool & InitLoc();

     private :


         void TestCplePlan(int aK1,int aK2);


         cLink2Img * GetLnkKbrd(int & aK);  // Saisir un lien au clavier


         bool ComputeLnkHom(cLink2Img & aLnK);
         void AddOnePtToExistingH(cPtHom *,const Pt2dr & aP1,cImagH *aI2,const Pt2dr & aP2);
         void AddOnePair(const Pt2dr & aP1,cImagH *,const Pt2dr & aP2);
         void FusionneIn(cPtHom *aH1,const Pt2dr & aP1,cImagH *aI2,cPtHom *aH2,const Pt2dr & aP2);

         cImagH(const cImagH &); // N.I.
         void ComputePtsLink(cLink2Img & aLnk);
         void AssertLnkUnclosed() const;
         void AssertLnkClosed() const;

         cAppliReduc &              mAppli;
         std::map<Pt2dr,cPtHom *>   mMapH;  // Liste des Hom deja trouves via les prec
         tMapName2Link                  mLnks;
         std::vector<cLink2Img*>    mVLnkInterneSorted;  // Sort by name, valide une  fois closed
         std::string                mName;
         std::string                mNameCalib;
         CamStenope *               mCamC;
         std::string                mNameVerif;
         int                        mNum;
         int                        mNumTmp;
         double                     mSomQual;
         double                     mSomNbPts;
         ElTabFlag                  mMarques;

         cElHomographie             mH2ImC;  // stocke le resultat de l'H vers l'image central, qd elle existe
         cElHomographie             mHi2t;  // Envoie terrain ver im
         cElHomographie             mHTmp;  // Envoie terrain ver im
         cHomogFormelle *           mHF;
         cMetaDataPhoto             mMDP;
         bool                       mPlanEst;
         bool                       mLnkClosed;


    // Variable temporaire pour l'estimation des plans

         std::vector<cElemMepRelCoplan>   mVercp;
         std::vector<cTestPlIm>           mVTPlIm;
         cEqOneHomogFormelle *            mEqOneHF;
         bool                       mC2CI;  // Connected to Center Image

         double    mGainLoc;
         bool      mInitLoc;
};



  //======================================
  //           cIndexImag
  //======================================


// Pour indexer les Pt Hom dans un QTree
class cPhIndexed
{
     public :
         const Pt2dr & Pt() const;
         cPhIndexed();
         bool operator == (const cPhIndexed&) const;
         cPhIndexed(const Pt2dr&,cPtHom*);
     private :
          Pt2dr      mPt;
          cPtHom *   mPH;
};

typedef Pt2dr (*tPtOfPhi)(const cPhIndexed &);

typedef enum
{
   eShowNone,
   eShowGlob,
   eShowDetail,
   eShowAll
} eNivShow;

class cAppliReduc
{
     public :
         cAppliReduc(int argc,char ** argv);
         void DoAll();

         void AddPtsHIndexed(const Pt2dr & aP,cPtHom *);
         const std::string & Dir() const;
         int    MinNbPtH() const;
         double SeuilQual () const;
         bool Show(eNivShow aLev) const;
         double RatioQualMoy () const;
         double SeuilDistNorm () const;
         int    KernConnec() const;
         int    KernSize() const;
         cSetEqFormelles & SetEq();

         void FigeAllNotFlaged(int aFlag);


         void  QuadrReestimFromVois(std::vector<cImagH*> & aVI,int aFlag);

         // the two version of TestMerge() found during the merge of binaries
         void TestMerge_SimulMerge();
         void TestMerge_CalcHcImage();

         void ComputePts();
         void ComputeHom();
         std::string NameCalib(const std::string & aNameIm) const;
         std::string NameVerif(const std::string & aNameIm) const;
         bool  H1On2() const;
         cInterfChantierNameManipulateur *ICNM() const;
         std::string  NameFileHomogr(const cLink2Img &) const;
         std::string  NameFileHomolH(const cLink2Img &) const;
         bool SkipHomDone() const;
         bool SkipPlanDone() const;
         double AltiCible() const;
         bool             HasImFocusPlan () const;
         std::string      ImFocusPlan () const;

         void AmelioHomLocal(cImagH & anIm);
         double  ErrorSolLoc();


     private :

         void CreateIndex();
         void ClearIndex();
         std::string KeyHIn(const std::string & aKeyGen) const;

         bool         mHomByParal;
         std::string  mName;
         std::string  mDir;
         std::string  mFullName;
         std::string  mOri;
         std::string  mOriVerif;
         std::string  mKeyOri;
         std::string  mKeyVerif;

         bool         mImportTxt;
         bool         mExportTxt;
         std::string  mExtHomol;
         int          mMinNbPtH;
         double       mSeuilQual;
         double       mRatioQualMoy;
         double       mSeuilDistNorm;
         int          mKernConnec;
         int          mKernSize;

         cInterfChantierNameManipulateur *mICNM;
         const cInterfChantierNameManipulateur::tSet * mSetNameIm;
         const cInterfChantierNameManipulateur::tSet * mSetNameHom;
         std::string  mKeySetHomol;
         std::string  mKeyInitIm2Homol;


         std::vector<cImagH *>  mIms;
         std::map<std::string,cImagH *>  mDicoIm;
         ElQT<cPhIndexed,Pt2dr,tPtOfPhi>  * mQT;

        cSetEqFormelles                     mSetEq;
        eNivShow                            mNivShow;
        bool                                mH1On2;
        bool                                mHFD; // HomogrFormatDump
        std::string                         mKeyHomogr;
        std::string                         mKeyHomolH;
        bool                                mSkipHomDone;
        bool                                mSkipPlanDone;
        bool                                mSkipAllDone;
        double                              mAltiCible;
        bool                                mHasImFocusPlan;
        std::string                         mImFocusPlan;

        bool                                mHasImCAmel;
        std::string                         mNameICA;
        cImagH  *                           mImCAmel;

        bool                                mDoCompensLoc;
};

class cAttrLnkIm{};

typedef cMergingNode<cImagH,cAttrLnkIm> tNodIm;

class cParamMerge
{
    public :
        cParamMerge( cAppliReduc &);

       void Vois(cImagH* anIm,std::vector<cImagH *> &);

       double  Gain // <0, veut dire on valide pas le noeud
               (
                     tNodIm * aN1,tNodIm * aN2,
                     const std::vector<cImagH*>&,
                     const std::vector<cImagH*>&,
                     const std::list<std::pair<cImagH*,cImagH*> >&,
                     int aNewNum
               );

       // Typiquement pour creer les Attibuts
       void OnNewLeaf(tNodIm * aSingle);
       void OnNewCandidate(tNodIm * aN1);
       void OnNewMerge(tNodIm * aN1);

       void Show(cImagH * anI){std::cout << anI->Name();}


   private :
        cAppliReduc & mAppli;
};

std::string NameNode(tNodIm * aN);




#endif //  _ELISE_REDUC_HHH_

NS_RHH_END



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
