#include "MMVII_GraphTriplets.h"

#include "MMVII_nums.h"
#ifndef  _MMVII_ARBOTRIPLETS_H_
#define  _MMVII_ARBOTRIPLETS_H_

#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_EnumCycles.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Sensor.h"
#include "MMVII_PoseTriplet.h"
// #include "MMVII_Tpl_GraphAlgo_Group.h"
#include "MMVII_Tpl_Images.h"

namespace MMVII
{

class cNodeArborTriplets;  // the hierarchical decomposition in tree of triplet
class cMakeArboTriplet;    // class for computing everything
class  cSolLocNode;        // class for storing the pose of one image inside a cNodeArborTriplets
class cOneTripletMerge;    // class for storing the topological/links between 2 Nodes before merge

typedef std::pair<int,int>  tPairI;

///   Store the pose and an ident (int) to the image
class  cSolLocNode
{
public :
    cSolLocNode(tPoseR aPose,int aNumPose) : mPose (aPose), mNumPose (aNumPose) {}
    tPoseR mPose;
    int    mNumPose;
};

/// Temporory struct, store the common images, triplet, edges ...  that allow to compute transfer similitude
class cOneTripletMerge
{
public :

    std::vector<tPairI>  mVLinkPose;  /// num inside mLocSols of poses linked by a triplet
    std::vector<tPairI>  mVLinkInTri; ///< num inside the triplet (in [0,1,2])  of the link
    std::vector<tPairI>  mVCommon;    ///< num inside mLocSols of poses common to 2 children
    int                  mNumTri;     ///< num of the triplet in the graph of triplet
};

/// store the hierarchical decomposition
class  cNodeArborTriplets : public cMemCheck
{
public :
    typedef cNodeArborTriplets * tNodePtr;
    typedef   std::vector<cCplIV<tREAL8>> tVIV;
    typedef   cSparseVect<tREAL8>         tSV;
    //typedef  std::pair<tPoseR,tPoseR>  tPP;

    /// constructor, recursively split the tree, dont parallelize !
    cNodeArborTriplets(cMakeArboTriplet &,const t3G3_Tree &,int aLevel);
    /// destructor, recursively free the children
    ~cNodeArborTriplets();

    /// compute the pose solution, do it recurively
    void ComputeResursiveSolution();
    ///  Test with GT
    void CmpWithGT();
    /// Save global solution
    void SaveGlobSol(const cPhotogrammetricProject&) const;


private :
    void AddEqLink(cLinearOverCstrSys<tREAL8> *,cSetIORSNL_SameTmp<tREAL8> * aSubst,tREAL8 aWeight, int aKEq,
                   const cPt3dr &aC0_In_W0, const cPt3dr &aC1_In_W0,
                   const cPt3dr &aCTri0_In_W0, const cPt3dr &aCTri1_In_W0
                   );


    ///  compute the relative pose associated to pair of local index, eventually adpat to ordering

    ///  compute the relative pose associated to pair of local index, eventually adpat to ordering
    tPoseR  PoseRelEdge(int aI0Loc,int aI1Loc) const;

    /// return the information on the link between the 2 children corresponding to a vertex of the graph of triplet
    cOneTripletMerge   ComputeTripletLinking(const t3G3_Vertex & aVTri);

    /** estimate the rotation of the similitude transfer from information on link , note tha Common and link2
            do not come from cOneTripletMerge because come merge has been done
        */
    tSim3dR EstimateSimTransfert
        (
            const std::vector<tPairI>& aVPairCommon,
            const std::vector<tPairI>& aVPairLink2,
            const std::vector<cOneTripletMerge> &  aVLink3
            );

    /** idem, but estimate the similitude (after computing the rotation) */
    tRotR EstimateRotTransfert
        (
            std::vector<tREAL8> &  aWeightRot,
            const std::vector<tPairI>& aVPairCommon,
            const std::vector<tPairI>& aVPairLink2,
            const std::vector<cOneTripletMerge> &  aVLink3
            );
    /// compute the mTabGlob2LocInd
    void MakeIndexGlob2Loc();

    /// print the  the prefix + level + the poses
    void ShowPose(const std::string & aPrefix) ;
    /// make the merge for non terminal nodes
    void MergeChildrenSol();
    /// free temporary data, non longer used after having been mergerd
    void FreeIndexSol();
    /// extract the num of poses of the tree
    void GetPoses(std::vector<int> &);

    /// solution of global index , null if dont exist
    cSolLocNode *  SolOfGlobalIndex(int aNumPose) ;

    int                       mDepth;     ///< level in the hierarchy, used for pretty printing
    t3G3_Tree                 mTree;      ///< tree of triplet
    std::array<tNodePtr,2>    mChildren;  ///< sub-nodes (if any ...)
    cMakeArboTriplet*         mPMAT;      ///< access to the global structure
    std::vector<cSolLocNode>  mLocSols;   ///< store the "local" solution
    std::vector<cSolLocNode>  mRotateLS;   ///< store the sol of N1 turned of rotation N1->N0
    std::vector<int>          mTabGlob2LocInd;  ///< index global -> index local (for acces to mLocSols) , -1 if no local homologous
};




class cMakeArboTriplet
{
public :

    cMakeArboTriplet(cTripletSet & aSet3,bool doCheck,tREAL8 aWBalance,cMMVII_Appli &);
    ~cMakeArboTriplet();

    /// Print some info on dimensions of data
    void ShowStat();

    // make the graph on pose, using triplet as 3 edges
    void MakeGraphPose(const cPhotogrammetricProject&);

    /// compute the reference pose
    void DoPoseRef();
    /// Compute the weight each triplet
    void MakeWeightingGraphTriplet();
    /// Compute the connectivity on triangles
    void MakeCnxTriplet();

    /// For each edge, compute the lowest cost triplet it belongs
    void ComputeArbor();

    /// Activate the simulation mode
    void SetRand(const std::vector<tREAL8> &);

    /// Save the global orientation
    void SaveGlobSol(const cPhotogrammetricProject&) const;


    tREAL8 WBalance() const {return mWBalance;}  ///< Accessor
    tREAL8 & CostMergeTree() {return mCostMergeTree;}  ///< Accessor

    t3G3_Graph &       GO3()  {return mGTriC;}  ///< Accessor
    t3GOP &            GOP()  {return mGPoses;}  ///< Accessor
    cTimerSegm  &   TimeSegm() {return mTimeSegm;}

    bool  PerfectData() const {return mPerfectData;}  ///< Accessor
    bool & PerfectData() {return mPerfectData;}
    bool  PerfectOri() const {return mPerfectOri;}
    bool & PerfectOri() {return mPerfectOri;}
    bool  DoRand() const {return mDoRand;}  ///< Accessor
    std::vector<tREAL8> & WeigthEdge3() {return mWeigthEdge3;}
    std::string & MapI2Str(const int aNum)  {return *mMapStrI.I2Obj(aNum);}  ///< Accessor


private :

    cMMVII_Appli &          mAppli;
    cTimerSegm  &           mTimeSegm;
    cTripletSet  &          mSet3;          ///< Initial triplet structure
    bool                    mDoCheck;       ///< do checking ....
    t2MapStrInt             mMapStrI;       ///< Maping name of pose / int used to import triplet
    t3GOP                   mGPoses;       ///< Graph of pose
    t3G3_Graph              mGTriC;         ///<  Graph of triplet

    bool                    mDoRand;         ///< Do we generate random values
    std::vector<tREAL8>     mLevelRand;      ///< Parameters of random values [RandOnTr,RandOnRot]
    std::vector<tREAL8>     mWeigthEdge3;    ///< Parameters of random values [RandOnTr,RandOnRot]
    bool                    mPerfectData;    ///< Are the triplet perfect with simulated pose
    bool                    mPerfectOri;       ///< Ground truth input orientation
    cNodeArborTriplets *    mArbor;          ///< Tree  for hierarchical  split
    tREAL8                  mWBalance;       ///<  Weighting for balance the tree
    tREAL8                  mWeightTr;       ///<  Relative weight Tranlastion vs rotation for pose distance
    tREAL8                  mCostMergeTree;  ///<  ????
    int                     mNbEdgeP;        ///<  Number of edge for poses
    int                     mNbHypP;         ///<  Number of hyp pose for pose (?=3 NbTriplet)
    int                     mNbEdgeTri;      ///<  Number of edge for triplet
    int                     mNbTriAnchor;    ///< Number of triplet that are anchor points
    int                     mNbTreeGlob;     ///< Number of triplet  in tri (NbTriplet - 1 for connected graph)
    int                     mNbTree2Split;   ///< Number of triplet after pruning
};

class cAppli_ArboTriplets : public cMMVII_Appli
{
public :

    cAppli_ArboTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
private :
    cPhotogrammetricProject   mPhProj;
    int                       mNbMaxClust;
    tREAL8                    mDistClust;
    std::vector<tREAL8>       mLevelRand;
    std::vector<tREAL8>       mWeigthEdge3;
    bool                      mDoCheck;
    tREAL8                    mWBalance;
    bool                      mPerfectData;
};

}; // namespace MMVII

#endif // _MMVII_ARBOTRIPLETS_H_
