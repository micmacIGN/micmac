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


#ifndef _ELISE_APERO_ALL_H_
#define _ELISE_APERO_ALL_H_

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

typedef std::map<cImagH *,cLink2Img *> tSetLinks;

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
         int   & NbPts();
         int   & NbPtsAttr();
         double &         Qual();
         cElHomographie & Hom12();

         cElHomographie CalcSrceFromDest();
         const ElPackHomologue & Pack() const;

        // Obtained by GetDistribRepresentative from util/pt2di.cpp
        // simply by averaging points on a regular grid
        // list of   Pt2dr+weight , represent the distribution of the points
         const std::vector<Pt3dr> & EchantP1() const;

        cEqHomogFormelle * &  EqHF();
    private :
       void LoadPack();
        
        cLink2Img(const cLink2Img &) ; // N.I.
        int      mNbPts;
        int      mNbPtsAttr;
        cImagH * mSrce;
        cImagH * mDest;
        std::string mNameH;
        double      mQual;
        //
        cElHomographie mHom12;
        bool            mPckLoaded;
        ElPackHomologue mPack;
        
        std::vector<Pt3dr> mEchantP1; 
        Pt3dr              mCdg1;
        cEqHomogFormelle * mEqHF;
};

class cImagH
{
     public :
// PRE REQUIS POUR LE MERGING
//=====================

        cLink2Img * GetLinkOfImage(cImagH*);


         cImagH(const std::string & aName,cAppliReduc &, int aNum);
         void AddLink(cImagH *,const std::string & aNameH);
         const std::string & Name() const;

         void ComputePts();

         void SetPHom(const Pt2dr & aP,cPtHom *);
         void ComputeLnkHom();

         void SetMarqued(int);
         void SetUnMarqued(int);
         bool Marqued(int) const;
         // std::vector<cImagH *> AdjRefl();  // Image adj + lui meme




        static void VoisinsNonMarques(const std::vector<cImagH*> & aIn,std::vector<cImagH*> & aV,int aFlagN,int FlagT );
        void   VoisinsMarques(std::vector<cLink2Img*> & aVois,int aFlagN);

         cElHomographie &     Hi2t() ;  // terrain ver I
         cElHomographie &     HTmp() ;  // terrain ver I

         cAppliReduc &    Appli();
         int & NumTmp();
         cHomogFormelle *  & HF();
         const tSetLinks & Lnks() const;
     private :


         bool ComputeLnkHom(cLink2Img & aLnK);
         void AddOnePtToExistingH(cPtHom *,const Pt2dr & aP1,cImagH *aI2,const Pt2dr & aP2);
         void AddOnePair(const Pt2dr & aP1,cImagH *,const Pt2dr & aP2);
         void FusionneIn(cPtHom *aH1,const Pt2dr & aP1,cImagH *aI2,cPtHom *aH2,const Pt2dr & aP2);

         cImagH(const cImagH &); // N.I.
         void ComputePtsLink(cLink2Img & aLnk);

         std::map<Pt2dr,cPtHom *>   mMapH;  // Liste des Hom deja trouves via les prec
         tSetLinks                  mLnks;
         std::string                mName;
         cAppliReduc &              mAppli;
         int                        mNum;
         int                        mNumTmp;
         double                     mSomQual;
         double                     mSomNbPts;
         ElTabFlag                  mMarques;

         cElHomographie             mHi2t;  // Envoie terrain ver im
         cElHomographie             mHTmp;  // Envoie terrain ver im
         cHomogFormelle *           mHF;
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


class cAppliReduc
{
     public :
         cAppliReduc(int argc,char ** argv);
         void DoAll();

         void AddPtsHIndexed(const Pt2dr & aP,cPtHom *);
         const std::string & Dir() const;
         int    MinNbPtH() const;
         double SeuilQual () const;
         double RatioQualMoy () const;
         int    KernConnec() const;
         int    KernSize() const;
         cSetEqFormelles & SetEq();

         void FigeAllNotFlaged(int aFlag);


         void  QuadrReestimFromVois(std::vector<cImagH*> & aVI,int aFlag);

		 // the two version of TestMerge() found during the merge of binaries
         void TestMerge_SimulMerge();
         void TestMerge_CalcHcImage();

     private :

         void CreateIndex();
         void ClearIndex();
         std::string KeyHIn(const std::string & aKeyGen) const;
         void ComputePts();

         std::string  mName;
         std::string  mDir;
         std::string  mFullName;
         bool         mImportTxt;
         bool         mExportTxt;
         std::string  mExtHomol;
         int          mMinNbPtH;
         double       mSeuilQual;
         double       mRatioQualMoy;
         int          mKernConnec;
         int          mKernSize;

         cInterfChantierNameManipulateur *mICNM;
         const cInterfChantierNameManipulateur::tSet * mSetNameIm;
         const cInterfChantierNameManipulateur::tSet * mSetNameHom;
         std::string  mKeyHomol;
         std::string  mKeyH2I;
         

         std::vector<cImagH *>  mIms;
         std::map<std::string,cImagH *>  mDicoIm;
         ElQT<cPhIndexed,Pt2dr,tPtOfPhi>  * mQT;

        cSetEqFormelles                     mSetEq;

};

class cAttrLnkIm{};

typedef cMergingNode<cImagH,cAttrLnkIm> tNodIm;

class cParamMerge
{
    public :
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

};

std::string NameNode(tNodIm * aN);




#endif //  _ELISE_APERO_ALL_H_




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
