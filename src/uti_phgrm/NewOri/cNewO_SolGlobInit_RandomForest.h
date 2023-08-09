#pragma once
/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de
Correlation eLiSe  : ELements of an Image Software Environnement

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
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <array>

#include "general/photogram.h"
#include "general/ptxd.h"
#include "graphes/cNewO_BuildOptions.h"
#define TREEDIST_WITH_MMVII false
#include "../../../MMVII/include/TreeDist.h"

#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/mman.h>


namespace SolGlobInit {

namespace RandomForest {

constexpr double MIN_WEIGHT = 0.5;
constexpr double MAX_WEIGHT = 10.0;
constexpr double IFLAG = -1.0;
constexpr double MIN_LNK_SEED = 4;
constexpr double MAX_SAMPLE_SEED = 50;

class cNOSolIn_AttrSom;
class cNOSolIn_AttrASym;
class cNOSolIn_AttrArc;
class cNOSolIn_Triplet;
class cLinkTripl;
class RandomForest;
struct cNO_HeapIndTri_NSI;
struct cNO_CmpTriByCost;

using tSomNSI   = ElSom<cNOSolIn_AttrSom, cNOSolIn_AttrArc>;
using tArcNSI   = ElArc<cNOSolIn_AttrSom, cNOSolIn_AttrArc>;
using tItSNSI   = ElSomIterator<cNOSolIn_AttrSom, cNOSolIn_AttrArc>;
using tItANSI   = ElArcIterator<cNOSolIn_AttrSom, cNOSolIn_AttrArc>;
using tGrNSI    = ElGraphe<cNOSolIn_AttrSom, cNOSolIn_AttrArc>;
using tSubGrNSI = ElSubGraphe<cNOSolIn_AttrSom, cNOSolIn_AttrArc>;

using tHeapTriNSI = ElHeap<cLinkTripl*, cNO_CmpTriByCost, cNO_HeapIndTri_NSI>;

using tElM = cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal>;
using tMapM = cStructMergeTieP<tElM>;
using tListM = std::list<tElM *>;


using tTriPoint = std::array<Pt2dr, 3>;
using tTriPointList = std::vector<tTriPoint>;

class cNOSolIn_AttrSom {
   public:
    cNOSolIn_AttrSom()
        :           prev(nullptr),
        oriented(false),
        mIm(),
          mCurRot(ElRotation3D::Id),
          mTestRot(ElRotation3D::Id),
          mNumCC(IFLAG),
          mNumId(IFLAG)
    {}
    cNOSolIn_AttrSom(const std::string& aName, RandomForest* anAppli);

    //~cNOSolIn_AttrSom();

    void AddTriplet(cNOSolIn_Triplet*, int aK0, int aK1, int aK2);
    cNewO_OneIm* Im() { return mIm; }
    ElRotation3D& CurRot() { return mCurRot; }
    const ElRotation3D& CurRot() const { return mCurRot; }
    ElRotation3D& TestRot() { return mTestRot; }

    std::vector<cLinkTripl>& Lnk3() { return mLnk3; }
    int& NumCC() { return mNumCC; }
    int& NumId() { return mNumId; }

    cLinkTripl* prev;
    std::vector<cLinkTripl*> next1;
    std::vector<cLinkTripl*> next2;
    bool oriented;
    double residue;

    double treeCost;
    int& HeapIndex() { return mHeapIndex; }

   private:
    std::string mName;
    RandomForest* mAppli;
    int mHeapIndex = 0;
    cNewO_OneIm* mIm;
    std::vector<cLinkTripl> mLnk3;
    ElRotation3D mCurRot;
    ElRotation3D mTestRot;

    // unique Id, corresponds to the distance of the triplet
    // which built/included this node in the solution;
    // mNumCC is used in the graph-based incoherence computation
    int mNumCC;
    int mNumId;
};

class cNOSolIn_AttrArc {
   public:
    cNOSolIn_AttrArc(cNOSolIn_AttrASym*, bool OrASym);
    cNOSolIn_AttrASym* ASym() { return mASym; }
    bool IsOrASym() const { return mOrASym; }

    int& HeapIndex() { return mHeapIndex; }

    double treecost;

   private:
    int mHeapIndex = 0;

    cNOSolIn_AttrASym* mASym;
    bool mOrASym;
};

template <class Container>
double mean(const Container& v){
    double sum = 0.;
    for (auto a : v) {
        sum += a;
    }
    return sum / v.size();
}

template <class Container>
double sd(const Container& v) {
    double sum = 0.0, mean, standardDeviation = 0.0;
    size_t i;

    for (i = 0; i < v.size(); ++i) {
        sum += v[i];
    }

    mean = sum / v.size();

    for (i = 0; i < v.size(); ++i) {
        standardDeviation += pow(v[i] - mean, 2);
    }

    return sqrt(standardDeviation / v.size());
}

class cNOSolIn_Triplet {
   public:
    cNOSolIn_Triplet(RandomForest*, tSomNSI* aS1, tSomNSI* aS2, tSomNSI* aS3,
                     const cXml_Ori3ImInit&);
    void SetArc(int aK, tArcNSI*);
    tSomNSI* KSom(int aK) const { return mSoms[aK]; }
    tArcNSI* KArc(int aK) const { return mArcs[aK]; }
    double CoherTest() const;
    double ProjTest() const;

    void AddHomolPts(tTriPointList& pts) { mHomolPts = pts; }
    void AddHomolAllPts(tTriPointList& pts) { mHomolAllPts = pts; }

    tTriPointList& getHomolPts() { return mHomolPts; }
    const tTriPointList& getHomolPts() const { return mHomolPts; }
    tTriPointList& getHomolAllPts() { return mHomolAllPts; }
    const tTriPointList& getHomolAllPts() const { return mHomolAllPts; }

    int Nb3() const { return mNb3; }
    ElTabFlag& Flag() { return mTabFlag; }
    //Number of connected component
    int& NumCC() { return mNumCC; }

    //Unique id of the triplet
    int& NumId() { return mNumId; }

    // The number order in the tree
    int& NumTT() { return mNumTT; }

    double Cost() const {
        return Sum()[indexSum];
    }

    const ElRotation3D& RotOfSom(const tSomNSI* aS) const {
        if (aS == mSoms[0]) return ElRotation3D::Id;
        if (aS == mSoms[1]) return mR2on1;
        if (aS == mSoms[2]) return mR3on1;
        ELISE_ASSERT(false, " RotOfSom");
        return ElRotation3D::Id;
    }

    const ElRotation3D& RotOfK(int aK) const {
        switch (aK) {
            case 0:
                return ElRotation3D::Id;
            case 1:
                return mR2on1;
            case 2:
                return mR3on1;
        }
        ELISE_ASSERT(false, " RotOfSom");
        return ElRotation3D::Id;
    }

    double treecost;

    /*double Residue() const { return mResidue; }
    double& Residue() { return mResidue; }

    double Dist() const { return mDistance; }
    double& Dist() { return mDistance; }*/

    /*
     * 0: Residue
     *
     * 1: Distance
     *
     * 2: Score
     */
    std::array<std::vector<double>, 3>& Data() { return mData; };
    static constexpr int indexSum = 3;
    /*
     * 0: Mean
     *
     * 1: Median
     *
     * 2: Mean Score
     *
     * 3: Median Score
     */
    std::array<double, 5>& Sum() { return mSum; };
    const std::array<double, 5>& Sum() const { return mSum; };

    double CalcDistArc();

    int& HeapIndex() { return mHeapIndex; }

    int category = -1;

    double confiance;

    double pondSum;
    double pondN;

    std::string print() {
        return mSoms[0]->attr().Im()->Name() + "/" +
               mSoms[1]->attr().Im()->Name() + "/" +
               mSoms[2]->attr().Im()->Name();
    }

   private:
    cNOSolIn_Triplet(const cNOSolIn_Triplet&);  // N.I.
    RandomForest* mAppli;
    tSomNSI* mSoms[3];
    // Arc between
    tArcNSI* mArcs[3];

    tTriPointList mHomolPts;
    tTriPointList mHomolAllPts;
    //Stats
    double mResidue;
    double mDistance;

    //End Stats

    std::array<std::vector<double>, 3> mData;
    /*
     * sum of Pds for the computation of the weighted mean
     * sum of cost times pds for the computation of the
     * weighted mean
     */
    std::array<double, 5> mSum;

    int mNb3;
    ElTabFlag mTabFlag;
    int mNumCC;  // id of its connected component
    int mNumId;  // unique Id throughout all iters
    int mNumTT;  // unique Id equiv of triplet order; each iter
    ElRotation3D mR2on1;
    ElRotation3D mR3on1;
    float mBOnH;

    int mHeapIndex = 0;
};

inline bool ValFlag(cNOSolIn_Triplet& aTrip, int aFlagSom) {
    return aTrip.Flag().kth(aFlagSom);
}
inline void SetFlag(cNOSolIn_Triplet& aTrip, int aFlag, bool aVal) {
    aTrip.Flag().set_kth(aFlag, aVal);
}

class cLinkTripl {
   public:
    cLinkTripl(cNOSolIn_Triplet* aTrip, int aK1, int aK2, int aK3)
        : m3(aTrip), mK1(aK1), mK2(aK2), mK3(aK3) {}

    int& HeapIndex() { return mHeapIndex; }
    double& Pds() { return aPds; }

    bool operator<(cLinkTripl& other) const {
        return m3->NumId() < other.m3->NumId();
    }
    bool operator==(cLinkTripl& other) const {
        std::cout << "###\n";
        return m3->NumId() == other.m3->NumId();
    }

    cLinkTripl* prev;
    bool viewed = false;

    cNOSolIn_Triplet* m3;

    U_INT1 mK1;
    U_INT1 mK2;
    U_INT1 mK3;

    tSomNSI* S1() const;
    tSomNSI* S2() const;
    tSomNSI* S3() const;

   private:
    int mHeapIndex = 0;  // Heap index pour tirer le meilleur triplets
    double aPds;  // Score de ponderation pour le tirage au sort de l'arbre
};

/*
 * Connected component containing a list of triplets cNOSolIn_Triplet and
 * nodes/sommets tSomNSI
 *
 */
class cNO_CC_TripSom {
   public:
    std::vector<cNOSolIn_Triplet*> mTri;
    std::vector<tSomNSI*> mSoms;

    // Index of the connected component in with this sommet is
    int mNumCC;
};

struct cNO_HeapIndTri_NSI {
    static void SetIndex(cLinkTripl* aV, int i) { aV->HeapIndex() = i; }
    static int Index(cLinkTripl* aV) { return aV->HeapIndex(); }
};

struct cNO_CmpTriByCost {
    bool operator()(cLinkTripl* aL1, cLinkTripl* aL2) {
        //return (aL1->m3)->Sum()[(aL1->m3)->indexSum] < (aL2->m3)->Sum()[(aL2->m3)->indexSum];
        return (aL1->m3)->Sum()[(aL1->m3)->indexSum] < (aL2->m3)->Sum()[(aL2->m3)->indexSum];
        //return (aL1->m3)->CostArc() < (aL2->m3)->CostArc();
    }
};
struct cNO_CmpTriByCostR {
    bool operator()(cLinkTripl& aL1, cLinkTripl& aL2) {
        //return (aL1->m3)->Sum()[(aL1->m3)->indexSum] < (aL2->m3)->Sum()[(aL2->m3)->indexSum];
        return (aL1.m3)->Sum()[(aL1.m3)->indexSum] < (aL2.m3)->Sum()[(aL2.m3)->indexSum];
        //return (aL1->m3)->CostArc() < (aL2->m3)->CostArc();
    }
};

// typedef ElHeap<cLinkTripl*,cNO_CmpTriByCost,cNO_HeapIndTri_NSI> tHeapTriNSI;

class cNOSolIn_AttrASym {
   public:
    cNOSolIn_AttrASym();

    void AddTriplet(cNOSolIn_Triplet* aTrip, int aK1, int aK2, int aK3);
    std::vector<cLinkTripl>& Lnk3() { return mLnk3; }
    //std::vector<cLinkTripl*>& Lnk3Ptr() { return mLnk3Ptr; }

    cLinkTripl* GetBestTri();
    tHeapTriNSI mHeapTri;

    int& NumArc() { return mNumArc; }

   private:
    std::vector<cLinkTripl> mLnk3;      // Liste des triplets partageant cet arc
    //std::vector<cLinkTripl*> mLnk3Ptr;  // Dirty trick pour faire marcher heap

    int mNumArc;
};

class cNO_HeapIndTriSol_NSI {
   public:
    static void SetIndex(cNOSolIn_Triplet* aV, int i) { aV->HeapIndex() = i; }
    static int Index(cNOSolIn_Triplet* aV) { return aV->HeapIndex(); }
};

class cNO_HeapIndSom_NSI {
   public:
    static void SetIndex(tSomNSI* aV, int i) { aV->attr().HeapIndex() = i; }
    static int Index(tSomNSI* aV) { return aV->attr().HeapIndex(); }
};

class cNO_HeapIndArc_NSI {
   public:
    static void SetIndex(tArcNSI* aV, int i) { aV->attr().HeapIndex() = i; }
    static int Index(tArcNSI* aV) { return aV->attr().HeapIndex(); }
};


class cNO_CmpTriSolByCost {
   public:
    bool operator()(cNOSolIn_Triplet* aL1, cNOSolIn_Triplet* aL2) {
        //return aL1->Sum()[aL1->indexSum] < aL2->Sum()[aL2->indexSum];
        size_t n1 = 0;
        size_t n2 = 0;
        for (int i = 0; i < 3; i++) {
            n1 += aL1->KSom(i)->attr().Lnk3().size();
            n2 += aL2->KSom(i)->attr().Lnk3().size();
        }
        return n1 > n2 || aL1->Sum()[aL1->indexSum] < aL2->Sum()[aL2->indexSum];
        //return aL1->Sum()[] < aL2->Sum()[aL2->indexSum];
        //return (aL1->pondSum / aL1->pondN) < (aL2->pondSum / aL2->pondN);
        //return (aL1->Pds()) < (T2->Pds());
    }
};

class cNO_CmpTriSolByDist {
   public:
    cNO_CmpTriSolByDist(const std::map<cNOSolIn_Triplet*, double>& dist)
        : dist(dist) {}
    bool operator()(cNOSolIn_Triplet* aL1, cNOSolIn_Triplet* aL2) {
        return dist.at(aL1) < dist.at(aL2);
    }
    const std::map<cNOSolIn_Triplet*, double>& dist;
};

class cNO_CmpSomByDist {
   public:
    cNO_CmpSomByDist (const std::map<tSomNSI*, double>& dist)
        : dist(dist) {}
    bool operator()(tSomNSI* aL1, tSomNSI* aL2) {
        return dist.at(aL1) < dist.at(aL2);
    }
    const std::map<tSomNSI*, double>& dist;
};

class cNO_CmpArcByDist {
   public:
    cNO_CmpArcByDist (const std::map<tSomNSI*, double>& dist)
        : dist(dist) {}
    bool operator()(tArcNSI* aL1, tArcNSI* aL2) {
        return dist.at(&aL1->s2()) < dist.at(&aL2->s2());
    }
    const std::map<tSomNSI*, double>& dist;
};



using tHeapTriSolNSI =
    ElHeap<cNOSolIn_Triplet*, cNO_CmpTriSolByCost, cNO_HeapIndTriSol_NSI>;

struct CmpLnk {
    bool operator()(cLinkTripl* T1, cLinkTripl* T2) const {
        //return (T1->m3->NumId()) < (T2->m3->NumId());
        //std::cout << "Pds" << T1->Pds() << std::endl;
        return (T1->m3->pondSum / T1->m3->pondN) < (T2->m3->pondSum / T2->m3->pondN);
    }
};

class Dataset {
   public:
    Dataset()
        : mFlag3CC(mAllocFlag3.flag_alloc()),
          mFlagS(mGr.alloc_flag_som()),
          mNbSom(0),
          mNbTrip(0) {}

    ~Dataset() = default;

    void CreateArc(tSomNSI* aS1, tSomNSI* aS2, cNOSolIn_Triplet* aTripl,
                   int aK1, int aK2, int aK3);
    /*
     * Graph Structure of images. Nodes:
     * tSomNSI
     */
    tGrNSI mGr;

    ElFlagAllocator mAllocFlag3;
    // Flag to indicate if tripplet was visited
    int mFlag3CC;
    // Flag to indicate if sommit was visited
    int mFlagS;

    /*
     * Sub graph of the images
     */
    tSubGrNSI mSubAll;

    // Mapping of all nodes tSomNSI with image name
    std::map<std::string, tSomNSI*> mMapS;

    // Complete list of loaded triplets
    std::vector<cNOSolIn_Triplet*> mV3;
    std::vector<cNO_CC_TripSom*> mVCC;
    std::unordered_map<std::string, CamStenope*> cam;

    // Stats variables

    // Number of nodes/sommet in the graph
    int mNbSom;
    // Number of arc / edge in the graph
    int mNbArc;
    // Number of Triplets in the graph
    int mNbTrip;
};

/*
 * Object storing a current travel of the dataset
 */
class DataTravel {
   public:
    DataTravel(Dataset& data) : data(data), gen(rd()) {
        auto r = rd();
        std::seed_seq seed{r};
        gen.seed(seed);
        std::cout << "DataTravel Seed: " << r << std::endl;
    }

    // variable to keep the visited
    std::map<std::string, tSomNSI*> mVS;
    // Holding on the current dataset we are traveling
    Dataset& data;

    // CC vars
    // dynamic list of currently adjacent triplets
    std::set<cLinkTripl*, CmpLnk> mSCur3Adj;
    void AddArcOrCur(cNOSolIn_AttrASym*);
    void AddArcOrCur(cNOSolIn_AttrASym* anArc, int flagSommet);

    void FreeSomNumCCFlag();
    void FreeSomNumCCFlag(std::vector<tSomNSI*>);
    void FreeTriNumTTFlag(std::vector<cNOSolIn_Triplet*>&);
    void FreeSCur3Adj(tSomNSI*);

    void resetFlags(cNO_CC_TripSom* aCC);

    cLinkTripl* GetRandTri(bool Pond = false);
    /*
     * Get the next triplet from tree previously generated.
     * Undefined behavior if tree not generated before
     *
     * Flag is the flag used to mark sommit as explored
     *
     */
    cLinkTripl* GetNextTri(int flag);
    std::random_device rd;
    //std::mt19937 gen;
    std::knuth_b gen;

    // if particles decay once per second on average,
    // how much time, in seconds, until the next one?
    //std::exponential_distribution<> d{0.3};
    //std::geometric_distribution<int> d{0.3};
};

struct ffinalTree {
    std::map<cNOSolIn_Triplet*, cLinkTripl*> links;
    std::map<cNOSolIn_Triplet*, std::set<cNOSolIn_Triplet*>> childs;
    std::map<tSomNSI*, std::set<tSomNSI*>> g;
    std::map<tSomNSI*, tSomNSI*> pred;
    std::map<tSomNSI*, std::set<tSomNSI*>> next;
    std::map<tSomNSI*, cLinkTripl*> ori;

    std::map<tSomNSI*, std::set<cNOSolIn_Triplet*>> triplets;
//    std::map<tSomNSI*, cNOSolIn_Triplet*> ori;

    cNOSolIn_Triplet* troot;
    tSomNSI* root;
};

struct finalScene {
    //std::set<cNOSolIn_Triplet*> ts;
    std::set<tSomNSI*> ss;

    void merge(const finalScene& other) {
        //for (auto e : other.ts) ts.insert(e);
        for (auto e : other.ss) ss.insert(e);
    }
};

class RandomForest : public cCommonMartiniAppli {
   public:
    RandomForest(int argc, char** argv);

    cNewO_NameManager& NM() { return *mNM; }

    // Load the triplets
    void loadDataset(Dataset& data);
    //void loadHomol(cNOSolIn_Triplet* aTriplet, tTriPointList& aLst);
    void loadHomol(cNOSolIn_Triplet* aTriplet, tTriPointList& aLst, tTriPointList& aLstAll);

    std::set<tSomNSI*> updateViewFrom(std::string name, std::set<tSomNSI*> views);

    // Entry point
    void DoNRandomSol(Dataset& data);

    void hierarchique(Dataset& data, size_t cc, ffinalTree& tree);
    finalScene bfs(Dataset& data, ffinalTree& tree, tSomNSI* node);
    std::set<tSomNSI*> bfs3(Dataset& data, ffinalTree& tree, tSomNSI* node);
    void callbackNode(Dataset& data,
              const std::unordered_map<tSomNSI*, std::set<tSomNSI*>>& ss,
              std::set<tSomNSI*>& processed,
              tSomNSI* node);

    std::set<tSomNSI*> bfs2(Dataset& data, ffinalTree& tree, tSomNSI* node);

    finalScene dfs(Dataset& data, ffinalTree& tree, tSomNSI* node);

    finalScene processNode(Dataset& data, const ffinalTree& tree, const std::map<tSomNSI*, finalScene>& r, tSomNSI* node);
    void processNode2(Dataset& data, const ffinalTree& tree, const std::unordered_map<tSomNSI*, std::set<tSomNSI*>>& ss, tSomNSI* node);
    void postorderNode(size_t depth, Dataset& data,std::unordered_map<tSomNSI*, std::set<tSomNSI*>>& ss, const ffinalTree& tree, tSomNSI* node);

    void RandomSolAllCC(Dataset& data, double* output, size_t n);
    void RandomSolOneCC(Dataset& data, cNO_CC_TripSom*, double* output, size_t n);

    void RandomSolAllCC(Dataset& data, cNO_CC_TripSom* aCC);
    void RandomSolOneCC(Dataset& data, cNOSolIn_Triplet* seed, int NbSomCC);

    void BestSolAllCC(Dataset& data);

    void BestSolOneCC(Dataset& data, cNO_CC_TripSom*, ffinalTree& );
    void BestSolOneCCDjikstra(Dataset& data, cNO_CC_TripSom*, ffinalTree& );
    void BestSolOneCCFloydWarshall(Dataset& data, cNO_CC_TripSom*, ffinalTree& );

   private:
    void NumeroteCC(Dataset& data);

    void AddTriOnHeap(Dataset& data, cLinkTripl*, bool oriented = true);
    void RemoveTriOnHeap(Dataset& data, cLinkTripl*);
    //void EstimRt(cLinkTripl*);

    void PreComputeTriplets(Dataset& data);
    void CoherTriplets(Dataset& data);
    void CoherTriplets(std::vector<cNOSolIn_Triplet*>& aV3);
    void CoherTripletsGraphBasedV2(
        Dataset& data, std::vector<cNOSolIn_Triplet*>& aV3, int, int,
        double* output);

    void CoherTripletsAllSamples(Dataset& data);
    void CoherTripletsAllSamplesMesPond(Dataset& data);
    void HeapPerEdge(Dataset& data);
    void HeapPerSol(Dataset& data);

    void ShowTripletCost(Dataset& data);
    void ShowTripletCostPerSample(Dataset& data);

    cNOSolIn_Triplet* GetBestTri();
    cNOSolIn_Triplet* GetBestTri(Dataset& data);
    cLinkTripl* GetBestTriDyn();

    void logTotalGraph(Dataset& data, std::string dir, std::string filename);

    void Save(Dataset& data, const std::string& OriOut, bool SaveListOfName = false);

    void SaveTriplet(Dataset& data, const std::string& OriOut);


    // MicMac managements variables
    std::string mFullPat;
    cElemAppliSetFile mEASF;
    cNewO_NameManager* mNM;
    std::string mOutName;

    bool mDebug;
    int mNbSamples;
    ElTimer mChrono;
    int mIterCur;
    bool mGraphCoher;

    // contains all triplets
    tHeapTriSolNSI mHeapTriAll;
    tHeapTriNSI mHeapTriDyn;

    double mDistThresh;
    double mResidu;
    bool mApplyCostPds;
    double mAlphaProb;

    double mR0;

    bool aModeBin = true;
    bool aPond = false;
};

template<typename... T>
class DataLog {
   public:
    DataLog() {
        hasHeader = false;
    }
    DataLog(std::array<std::string, sizeof...(T)> i) {
        header = i;
        hasHeader = true;
    }

    void add(std::tuple<T...> e) {
        data.push_back(e);
    }

    void write(std::string directory, std::string filename)
    {
        ELISE_fp::MkDirRec(directory);
        std::ofstream f(directory + "/" + filename, std::ios::out);
        if (hasHeader) {
            for (auto a : header) {
                f << a << ",";
            }
            f << std::endl;
        }
        for (auto& e : data) {
            print(f, e);
            f << std::endl;
        }
    }

   private:
    bool hasHeader;
    std::array<std::string, sizeof...(T)> header;
    std::vector<std::tuple<T...>> data;

    template<typename S, std::size_t I = 0, typename... Tp>
        inline typename std::enable_if<I == sizeof...(Tp), void>::type
        print(S& s, std::tuple<Tp...>& t)
        { }

    template<typename S, std::size_t I = 0, typename... Tp>
        inline typename std::enable_if<I < sizeof...(Tp), void>::type
        print(S& s,std::tuple<Tp...>& t)
        {
            s << to_string(std::get<I>(t)) << ",";
            print<S, I + 1, Tp...>(s, t);
        }

    template<typename A>
    std::string to_string(A& t) {
        return std::to_string(t);
    }
    std::string to_string(std::string& t) {
        return t;
    }

};

#ifdef GRAPHVIZ_ENABLED

#include  <graphviz/gvc.h>

class GraphViz {
   public:
    Pt3dr offset;
    GraphViz() {
        g = agopen((char*)"g", Agundirected, 0);
        tripG = agsubg(g, (char*)"triplets", 1);
        nG = agsubg(g, (char*)"views", 1);
        tnG = agsubg(g, (char*)"triplets_views", 1);
        treeG = agsubg(g, (char*)"triplets_tree", 1);
        agsafeset(g, (char*)"component", (char*)"true", "");
        //agattr(g,AGNODE,(char*)"component", (char*)"True");

        //offset.z = 10;
    }
    ~GraphViz() { agclose(g); }

    void write(const std::string dir, const std::string aFName) {
        ELISE_fp::MkDirRec(dir);
        std::string path = dir + "/" + aFName;
        FILE* fp = fopen((char*)path.c_str(), "w");
        agwrite(g, fp);
        fclose(fp);
    }

    void addTriplet(cNOSolIn_Triplet& node) {
        node_t* n[3];
        std::string names[] = {node.KSom(0)->attr().Im()->Name(),
                             node.KSom(1)->attr().Im()->Name(),
                             node.KSom(2)->attr().Im()->Name()};
        std::string triplet_name = std::to_string(node.NumId());

        node_t* triplet = agnode(tripG, (char*)triplet_name.c_str(), 0);
        if (!triplet) { //Triplet Id don't exist
            triplet = agnode(tripG, (char*)triplet_name.c_str(), 1);
        }
        Pt3dr center;
        for (int i = 0; i < 3; i++) {
            n[i] = agnode(nG, (char*)names[i].c_str(), 0);
            auto tr = node.KSom(i)->attr().CurRot().tr();
            if (!n[i]) { //Node don't exist so create and make it position
                n[i] =
                    agnode(nG, (char*)names[i].c_str(), 1);
            }
            std::string pos = "" +
                std::to_string(tr.x) + "," +
                std::to_string(tr.y) + "," +
                std::to_string(tr.z);
            agsafeset(n[i], (char*)"pos", (char*)pos.c_str(), "");
            //agsafeset(n[i], (char*)"label", (char*)std::to_string(node.KSom(i)->attr().NumId()).c_str(), "");
            agsafeset(n[i], (char*)"label", (char*)std::to_string(node.NumTT()).c_str(), "");
            center = center + tr;
        }
        center = center / 3;
        center = center + offset;
        std::string pos = "" + std::to_string(center.x) + "," +
                          std::to_string(center.y) + "," +
                          std::to_string(center.z);
        agsafeset(triplet, (char*)"pos", (char*)pos.c_str(), "");
        agsafeset(triplet, (char*)"label", (char*)triplet_name.c_str(), "");

        //Edge triplet to triplet
        for (uint8_t i = 0; i < 3; i++) {
            auto ltrip = node.KSom(i)->attr().Lnk3();
            for (auto lt : ltrip) {
                std::string triplet_name2 = std::to_string(lt.m3->NumId());
                node_t* t = agnode(tripG, (char*)triplet_name2.c_str(), 0);
                if (t)
                    agedge(tripG, triplet, t, 0, 1);
            }
        }

        //Edge view to view
        edge_t* e[3] = {
            agedge(nG, n[0], n[1], 0, 1),
            agedge(nG, n[2], n[1], 0, 1),
            agedge(nG, n[2], n[0], 0, 1)
        };

        //Edge triplet to view
        edge_t* te[3] = {
            agedge(tnG, triplet, n[0], 0, 1),
            agedge(tnG, triplet, n[1], 0, 1),
            agedge(tnG, triplet, n[2], 0, 1),
        };
        (void)e;
        (void)te;

        for (int i = 0; i < 3; i++) {
            agsafeset(e[i], (char*)"weight", (char*)std::to_string(node.Sum()[node.indexSum]).c_str(), "");
        }
    }

    void linkTree(cNOSolIn_Triplet* t1, cNOSolIn_Triplet* t2) {
        std::string triplet_name = std::to_string(t1->NumId());
        node_t* triplet = agnode(tripG, (char*)triplet_name.c_str(), 0);
        if (!triplet) {  // Triplet Id don't exist
            std::cout << "triplet dont exist GraphViz" << std::endl;
            return;
        }
        std::string cname = std::to_string(t2->NumId());
        node_t* ct = agnode(tripG, (char*)cname.c_str(), 0);
        if (!ct) {  // Triplet Id don't exist
            std::cout << "triplet dont exist GraphViz next" << t2->NumId()
                      << std::endl;
            return;
        }
        agedge(treeG, triplet, ct, 0, 1);
    }

    void linkTree(tSomNSI* t1, tSomNSI* t2, bool create = false) {
        std::string name1 = t1->attr().Im()->Name();
        node_t* i1 = agnode(nG, (char*)name1.c_str(), create);
        if (!i1) {  // Triplet Id don't exist
            std::cout << "triplet dont exist GraphViz" << std::endl;
            return;
        }
        std::string name2 = t2->attr().Im()->Name();
        node_t* i2 = agnode(nG, (char*)name2.c_str(), create);
        if (!i2) {  // Triplet Id don't exist
            std::cout << "image dont exist GraphViz next" << t2->attr().Im()->Name()
                      << std::endl;
            return;
        }
        agedge(treeG, i1, i2, 0, 1);
    }

    void travelGraph(Dataset& data, cNO_CC_TripSom& aCC,
                     cNOSolIn_Triplet* aSeed) {
        auto travelFlagTri = data.mAllocFlag3.flag_alloc();
        auto travelFlagSom = data.mGr.alloc_flag_som();

        DataTravel travel(data);

        addTriplet(*aSeed);

        for (int aK = 0; aK < 3; aK++) {
            // Mark as explored
            aSeed->KSom(aK)->flag_set_kth_true(travelFlagSom);
        }

        for (int aK = 0; aK < 3; aK++) {
            // Add the seed adjacent to the set of not visited triplets
            travel.AddArcOrCur(aSeed->KArc(aK)->attr().ASym(), travelFlagSom);
        }

        size_t Cpt = 0;
        cLinkTripl* aTri = 0;
        while ((aTri = travel.GetNextTri(travelFlagSom)) &&
               ((Cpt + 3) < aCC.mTri.size())) {
            //std::cout << "GRAPH Size: "
             //         << std::to_string(travel.mSCur3Adj.size()) << std::endl;
            // Flag as visted
            aTri->m3->Flag().set_kth_true(travelFlagTri);
            // Mark sommit as vistied
            aTri->S3()->flag_set_kth_true(travelFlagSom);
            addTriplet(*aTri->m3);

            // Free mSCur3Adj from all triplets connected to S3
            travel.FreeSCur3Adj(aTri->S3());

            // Add two new edges and their respective adjacent triplets
            travel.AddArcOrCur(aTri->m3->KArc(aTri->mK1)->attr().ASym(),
                               travelFlagSom);
            travel.AddArcOrCur(aTri->m3->KArc(aTri->mK2)->attr().ASym(),
                               travelFlagSom);

            Cpt++;
        }

        // Unflag all triplets do pursue with DFS
        for (unsigned aK3 = 0; aK3 < data.mV3.size(); aK3++) {
            data.mV3[aK3]->Flag().set_kth_false(travelFlagTri);
        }
        FreeAllFlag(aCC.mSoms, travelFlagSom);

        std::cout << "\nIn this CC, nb of connected nodes " << Cpt + 3 << "\n";

        data.mGr.free_flag_som(travelFlagSom);
        data.mAllocFlag3.flag_free(travelFlagTri);
    }

    void loadTotalGraph(Dataset& data) {
        DataTravel travel(data);
        // Clear the sommets
        travel.mVS.clear();
        auto flag = data.mAllocFlag3.flag_alloc();

        cNOSolIn_Triplet* aTri0 = data.mV3[0];

        // initialise the starting node
        addTriplet(*aTri0);
        std::vector<cNOSolIn_Triplet*> aCC3;
        // Add first triplet
        aCC3.push_back(aTri0);
        aTri0->Flag().set_kth_true(flag);  // explored

        unsigned aKCur = 0;
        // Visit all sommets (not all triplets)
        while (aKCur != aCC3.size()) {
            cNOSolIn_Triplet* aTri1 = aCC3[aKCur];
            // For each edge of the current triplet
            for (int aKA = 0; aKA < 3; aKA++) {
                // Get triplet adjacent to this edge and parse them
                std::vector<cLinkTripl>& aLnk =
                    aTri1->KArc(aKA)->attr().ASym()->Lnk3();

                for (unsigned aKL = 0; aKL < aLnk.size(); aKL++) {
                    // If not marked, mark it and push it in aCC3, return it was
                    // added
                    if (SetFlagAdd(aCC3, aLnk[aKL].m3, flag)) {
                        // travel.mVS[aLnk[aKL].S3()->attr().Im()->Name()] =
                        //    aLnk[aKL].S3();

                        /*std::cout << aCC3.size() << "=["
                                  << aLnk[aKL].S1()->attr().Im()->Name();
                        std::cout << "," << aLnk[aKL].S2()->attr().Im()->Name();
                        std::cout << "," << aLnk[aKL].S3()->attr().Im()->Name()
                                  << "=====\n";*/

                        addTriplet(*aLnk[aKL].m3);
                    }
                }
            }
            aKCur++;
        }

        // Unflag all triplets do pursue with DFS
        for (unsigned aK3 = 0; aK3 < data.mV3.size(); aK3++) {
            data.mV3[aK3]->Flag().set_kth_false(flag);
        }

        data.mAllocFlag3.flag_free(flag);
    }

   private:
    graph_t* g;
    graph_t* tripG;
    graph_t* nG;
    graph_t* tnG;
    graph_t* treeG;
};
#else
class GraphViz {
   public:
    GraphViz() {}
    ~GraphViz() {}

    void write(const std::string dir, const std::string aFName) {}

    void addTriplet(cNOSolIn_Triplet& node) {}

    void travelGraph(Dataset& data, cNO_CC_TripSom& aCC,
                     cNOSolIn_Triplet* aSeed) {}

    void loadTotalGraph(Dataset& data) {}

   private:
};
#endif


}  // namespace RandomForest

}  // namespace SolGlobInit

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
