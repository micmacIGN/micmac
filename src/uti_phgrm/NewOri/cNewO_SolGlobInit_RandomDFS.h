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


#include "NewOri.h"
//#include "general/CMake_defines.h"
#include "graphes/cNewO_BuildOptions.h"

#ifdef GRAPHVIZ_ENABLED
	#include  <graphviz/gvc.h>
#endif



namespace SolGlobInit_DFS {


#define MIN_LNK_SEED 4
#define MAX_SAMPLE_SEED 50
		
class cNOSolIn_AttrSom;
class cNOSolIn_AttrASym;
class cNOSolIn_AttrArc;
class cNOSolIn_Triplet;
class cLinkTripl;
class cSolGlobInit_RandomDFS;
class cNO_HeapIndTri_NSI;
class cNO_CmpTriByCost;

typedef  ElSom<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tSomNSI;
typedef  ElArc<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tArcNSI;
typedef  ElSomIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItSNSI;
typedef  ElArcIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItANSI;
typedef  ElGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>      tGrNSI;
typedef  ElSubGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>   tSubGrNSI;

typedef ElHeap<cLinkTripl*,cNO_CmpTriByCost,cNO_HeapIndTri_NSI> tHeapTriNSI;


class cNOSolIn_AttrSom
{
     public :
         cNOSolIn_AttrSom() :
             mCurRot (ElRotation3D::Id),
             mTestRot (ElRotation3D::Id) {}
         cNOSolIn_AttrSom(const std::string & aName,cSolGlobInit_RandomDFS & anAppli);


         void AddTriplet(cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
         cNewO_OneIm * Im() {return mIm;}
		 ElRotation3D & CurRot() {return mCurRot;}
         ElRotation3D & TestRot() {return mTestRot;}

         std::vector<cLinkTripl> & Lnk3() {return mLnk3;}



    private:
         std::string                      mName;
         cSolGlobInit_RandomDFS *         mAppli;
         int                              mHeapIndex;
         cNewO_OneIm *                    mIm;
         std::vector<cLinkTripl>          mLnk3;
		 ElRotation3D                     mCurRot;
         ElRotation3D                     mTestRot;

};



class cNOSolIn_AttrArc
{
     public :
           cNOSolIn_AttrArc(cNOSolIn_AttrASym *,bool OrASym);
           cNOSolIn_AttrASym * ASym() {return mASym;}
           bool          IsOrASym() const {return mOrASym;}

     private :
           cNOSolIn_AttrASym * mASym;
           bool                mOrASym;
};


class cNOSolIn_Triplet
{
      public :
          cNOSolIn_Triplet(cSolGlobInit_RandomDFS *,tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit &);
          void SetArc(int aK,tArcNSI *);
          tSomNSI * KSom(int aK) const {return mSoms[aK];}
          tArcNSI * KArc(int aK) const {return mArcs[aK];}
          double CoherTest() const;



          int  Nb3() const {return mNb3;}
          ElTabFlag & Flag() {return   mTabFlag;}
          int & NumCC() {return mNumCC;}

		  const ElRotation3D & RotOfSom(tSomNSI * aS) const
          {
                if (aS==mSoms[0]) return ElRotation3D::Id;
                if (aS==mSoms[1]) return mR2on1;
                if (aS==mSoms[2]) return mR3on1;
                ELISE_ASSERT(false," RotOfSom");
                return ElRotation3D::Id;
          }
          const ElRotation3D & RotOfK(int aK) const
          {
                switch (aK)
                {
                      case 0 : return ElRotation3D::Id;
                      case 1 : return mR2on1;
                      case 2 : return mR3on1;
                }
                ELISE_ASSERT(false," RotOfSom");
                return ElRotation3D::Id;
          }

		  float   CostArc() const {return mCostArc;}
		  float&  CostArc() {return mCostArc;}
		  float   CostArcMed() const {return mCostArcMed;}
		  float&  CostArcMed() {return mCostArcMed;}
		  std::vector<float>& CostArcPerSample() {return mCostArcPerSample;};

          int  & HeapIndex() {return mHeapIndex;}


      private :
          cNOSolIn_Triplet(const cNOSolIn_Triplet &); // N.I.
          cSolGlobInit_RandomDFS * mAppli;
          tSomNSI *           mSoms[3];
          tArcNSI *           mArcs[3];
		  float               mCostArc;
          float               mCostArcMed;
		  std::vector<float>  mCostArcPerSample;

          int           mNb3;
          ElTabFlag     mTabFlag;
          int           mNumCC;
		  ElRotation3D  mR2on1;
          ElRotation3D  mR3on1;
          float         mBOnH;
		
		  int           mHeapIndex;
};

inline bool ValFlag(cNOSolIn_Triplet & aTrip,int aFlagSom)
{
   return aTrip.Flag().kth(aFlagSom);
}
inline void  SetFlag(cNOSolIn_Triplet & aTrip,int aFlag,bool aVal)
{
    aTrip.Flag().set_kth(aFlag,aVal);
}

class cLinkTripl
{
     public :
         cLinkTripl(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3) :
            m3   (aTrip),
            mK1  (aK1),
            mK2  (aK2),
            mK3  (aK3)
         {
         }

         int  & HeapIndex() {return mHeapIndex;}

         cNOSolIn_Triplet  *  m3;
         U_INT1               mK1;
         U_INT1               mK2;
         U_INT1               mK3;
         tSomNSI *            S1() const;
         tSomNSI *            S2() const;
         tSomNSI *            S3() const;

	 private:
		 int mHeapIndex; //Heap index pour tirer le meilleur triplets

};

class cNO_CC_TripSom
{
    public :
        std::vector<cNOSolIn_Triplet *> mTri;
        std::vector<tSomNSI *>          mSoms;
        int                             mNumCC;
};


class  cNO_HeapIndTri_NSI
{
    public :
           static void SetIndex(cLinkTripl* aV,int i) {   aV->HeapIndex()=i;}
           static int  Index(cLinkTripl * aV)
           {
                return aV->HeapIndex();
           }

};

class cNO_CmpTriByCost
{
    public:
        bool operator()(cLinkTripl * aL1,cLinkTripl * aL2)
        {
            return (aL1->m3)->CostArc() < (aL2->m3)->CostArc();
        }
};

//typedef ElHeap<cLinkTripl*,cNO_CmpTriByCost,cNO_HeapIndTri_NSI> tHeapTriNSI;

class cNOSolIn_AttrASym
{
     public :
         cNOSolIn_AttrASym();

         void AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3);
         std::vector<cLinkTripl> & Lnk3() {return mLnk3;}
         std::vector<cLinkTripl*> & Lnk3Ptr() {return mLnk3Ptr;}

         cLinkTripl *       GetBestTri();
         tHeapTriNSI        mHeapTri;

     private :
         std::vector<cLinkTripl> mLnk3; // Liste des triplets partageant cet arc
		 std::vector<cLinkTripl*> mLnk3Ptr; //Dirty trick pour faire marcher heap

};

class  cNO_HeapIndTriSol_NSI
{
    public :
           static void SetIndex(cNOSolIn_Triplet * aV,int i) {   aV->HeapIndex()=i;}
           static int  Index(cNOSolIn_Triplet * aV)
           {
                return aV->HeapIndex();
           }

};

class cNO_CmpTriSolByCost
{
    public:
        bool operator()(cNOSolIn_Triplet * aL1,cNOSolIn_Triplet * aL2)
        {
            return aL1->CostArc() < aL2->CostArc();
        }
};

typedef ElHeap<cNOSolIn_Triplet*,cNO_CmpTriSolByCost,cNO_HeapIndTriSol_NSI> tHeapTriSolNSI;

class cSolGlobInit_RandomDFS : public cCommonMartiniAppli
{
    public:
        cSolGlobInit_RandomDFS(int argc,char ** argv);
		cNewO_NameManager & NM() {return *mNM;}

		void DoOneRandomDFS(bool UseCoherence=false);
		void DoRandomDFS();

    private:
        void CreateArc(tSomNSI *,tSomNSI *,cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
		void CoherTriplets();
		void CoherTripletsAllSamples();
		void HeapPerEdge();
		void HeapPerSol();
		void ShowTripletCost();

		cNOSolIn_Triplet * GetBestTri();

		void Save(std::string& OriOut);

        std::string mFullPat;
        cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;

        tGrNSI               mGr;
        tSubGrNSI            mSubAll;
        std::map<std::string,tSomNSI *> mMapS;
		std::map<std::string,tSomNSI *> mVS; //variable to keep the visited sommets


        std::vector<cNOSolIn_Triplet*> mV3;
		std::vector<cNO_CC_TripSom *> mVCC;

        int             		mNbSom;
        int             		mNbArc;
        int             		mNbTrip;
		ElFlagAllocator         mAllocFlag3;
		int 					mFlag3CC;
        bool            		mDebug;
        int             		mNbSamples;
        ElTimer         		mChrono;
		int                     mIterCur;


#ifdef GRAPHVIZ_ENABLED
		GVC_t *GRAPHVIZ_GVCInit(const std::string& aGName);
		std::pair<graph_t *,GVC_t *> GRAPHVIZ_GraphInit(const std::string& aGName);
		graph_t *GRAPHIZ_GraphRead(std::string& aFName);
		void GRAPHIZ_GraphSerialize(std::string& aFName,graph_t *g);
		void GRAPHIZ_GraphKill(graph_t *g,GVC_t *gvc,std::string aWriteName="");
		void GRAPHIZ_NodeInit(graph_t *,
                      const std::string& ,
                      const std::string& ,
                      const std::string& );
		void GRAPHVIZ_NodeAdd(graph_t* ,
                      const std::string& ,
                      const std::string& ,
                      const std::string& );
		void GRAPHVIZ_NodeChgColor(graph_t*,const std::string& );
		void GRAPHVIZ_EdgeChgColor(graph_t*,
                                   const std::string& ,
                                   const std::string& ,
					   			   std::string aColor="red");
		void WriteGraphToFile();

		graph_t *mG;
		GVC_t   *mGVC;

#endif
		std::string mGraphName;
		tHeapTriSolNSI mHeapTriAll;

};


} //SolGlobInit_DFS

class cAppliGenOptTriplets : public cCommonMartiniAppli
{
	public:
		cAppliGenOptTriplets(int argc,char ** argv);
		

	private:
		ElMatrix<double> RandPeturbR();

		std::string mFullPat; 
		std::string InOri;


		int    mNbTri;
		double mSigma;//bruit
		double mRatioOutlier;//outlier ratio, if 0.3 => 30% of outliers will be present among the triplets

		cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;
		
};

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

