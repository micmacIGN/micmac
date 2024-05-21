#ifndef  _MMVII_SFMINIT_H_
#define  _MMVII_SFMINIT_H_

#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PoseTriplet.h"
#include "MMVII_PoseRel.h"
//#include "MMVII_TplHeap.h"   //conflict with mmv1
#include <thread>

//#include "mt-kahypar/macros.h"
//#include "mt-kahypar/definitions.h"

//#undef ASSERT
//#undef HEAVY_ASSERT0
#include "libmtkahypar.h"

//#include "mt-kahypar/datastructures/connectivity_set.h"
//#include "mt-kahypar/io/partitioning_output.h"
//#undef ASSERT
//#undef HEAVY_ASSERT0


namespace MMVII
{
//extern const int HEAP_NO_INDEX;

class cVertex;
class cEdge;
class cHyperEdge;
class cHyperGraph;

/* ********************************************************** */
/*                                                            */
/*                         cVertex                            */
/*                                                            */
/* ********************************************************** */

class cVertex : public cMemCheck
{
 public:
     cVertex(int,std::string);// : mId(aId), mPose(cView()), _ISORIENTED(false) {}

     const int Id() const {return mId;}

     const cView& Pose() const {return mPose;}
     cView& Pose() {return mPose;}

     void SetPose(cView& aP) {mPose=aP;}
     bool& FlagOriented() {return FLAG_ISORIENTED;}

     void Show();

 private:
     int   mId;
     cView mPose;


     bool FLAG_ISORIENTED;
};

/* ********************************************************** */
/*                                                            */
/*                         cEdge                              */
/*                                                            */
/* ********************************************************** */

class cEdge : public cMemCheck
{
    public:
         cEdge(cVertex* aStart,cVertex* aEnd) : mStart(aStart), mEnd(aEnd) {}

         const cVertex * StartVertex() const {return mStart;}
         cVertex * StartVertex()  {return mStart;}
         const cVertex * EndVertex() const {return mEnd;}
         cVertex * EndVertex()  {return mEnd;}

         bool operator==(const cEdge& compare) const {
                 return (
                             (mStart == compare.StartVertex() && mEnd == compare.EndVertex()) ||
                             (mStart == compare.EndVertex() && mEnd == compare.StartVertex())
                        );
         }

         struct Hash {
                 size_t operator()(const cEdge& aE) const {
                     size_t hashStart = std::hash<const cVertex*>{}(aE.StartVertex());
                     size_t hashEnd = std::hash<const cVertex*>{}(aE.EndVertex());
                     return hashStart ^ (hashEnd << 1);
                 }
             };

    private:
         cVertex * mStart;
         cVertex * mEnd;
};

/* ********************************************************** */
/*                                                            */
/*                        cHyperEdge                          */
/*                                                            */
/* ********************************************************** */

class cHyperEdge : public cMemCheck
{
 public:
     cHyperEdge(std::vector<cVertex*>, cTriplet&, tU_INT4);
     ~cHyperEdge();

     std::vector<cVertex*>& Vertices() {return mVVertices;}
     const std::vector<cVertex*>& Vertices() const {return mVVertices;}

     std::vector<cEdge*>& Edges() {return mVEdges;}
     const std::vector<cEdge*>& Edges() const {return mVEdges;}

     double BsurH() {return mBsurH;}
     const double BsurH() const {return mBsurH;}

     double& Quality() {return mQual;}
     const double Quality() const {return mQual;}

     std::vector<double>& QualityVec() {return mQualVec;}
     const std::vector<double>& QualityVec() const {return mQualVec;}

     tU_INT4& IndexHeap() {return mIndexHeap;}
     const tU_INT4& IndexHeap() const {return mIndexHeap;}
     const tU_INT4& Index() const {return mIndex;}

     void  SetRelPoses(cTriplet& aT) {mRelPoses=aT;}
     const cView& RelPose(int aK) const {return mRelPoses.PVec()[aK];}

     const bool& FlagOriented() const {return FLAG_IS_ORIENTED;}
     bool& FlagOriented() {return FLAG_IS_ORIENTED;}


     void Show();

 private:
     std::vector<cVertex*> mVVertices;
     std::vector<cEdge*>   mVEdges;

     double                mBsurH;
     double                mQual;
     std::vector<double>   mQualVec; //< stores the quality estimate in each random spanning tree
     tU_INT4               mIndex;
     tU_INT4               mIndexHeap;

     cTriplet              mRelPoses; // see if make it pointer

     bool                  FLAG_IS_ORIENTED; //< flag to exclude from coherency score
};


/* ********************************************************** */
/*                                                            */
/*                        cHyperGraph                         */
/*                                                            */
/* ********************************************************** */

class cHyperGraph : public cMemCheck
{
 public:
      cHyperGraph() : IS_INIT(false) {};
      ~cHyperGraph();

      void InitFromTriSet(const cTripletSet*);

      void                      AddHyperedge(cHyperEdge*);
      std::vector<cHyperEdge*>& GetAdjacentHyperedges(cEdge*);
      bool                      HasAdjacentHyperedges(cEdge*);

      cHyperEdge*               GetHyperEdge(int aK) {return mVHEdges[aK];}
      const cHyperEdge*         GetHyperEdge(int aK) const {return mVHEdges[aK];}

      cHyperEdge*               CurrentBestHyperedge();

      const std::unordered_map<std::string,cVertex*>& GetMapVertices() const {return mMapVertices;}

      cVertex*                  GetVertex(std::string& aN) {return mMapVertices[aN];}

      tU_INT4                   NbHEdges() {return mVHEdges.size();}
      tU_INT4                   NbVertices() {return mMapVertices.size();}
      tU_INT4                   NbPins();

      void SetVertices(std::unordered_map<std::string,cVertex*>&);

      void RandomizeQualOfHyperE();
      void UpdateQualFromVec(double);

      void CoherencyOfHyperEdges();

      void ClearFlags();
      void UpdateIndHeap();

      void SaveDotFile(std::string&);
      void SaveHMetisFile(std::string&);

      void Show();
      void ShowFlags(bool);
      void ShowHyperEdgeVQual();

 private:
     std::unordered_map<std::string,cVertex*>    mMapVertices;
     std::vector<cHyperEdge*>                    mVHEdges;
     std::unordered_map<cEdge, std::vector<cHyperEdge*>, typename cEdge::Hash> mAdjMap;

     bool                                        IS_INIT;

};

/* ********************************************************** */
/*                                                            */
/*                 cAppli_SfmInitWithPartition                */
/*                                                            */
/* ********************************************************** */

class cAppli_SfmInitWithPartition: public cMMVII_Appli
{
  public :
     typedef cIsometry3D<tREAL8>  tPose;

     cAppli_SfmInitWithPartition(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     int Exe() override;
     cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
     cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


 private :
     cPhotogrammetricProject   mPhProj;
     int                 mNbParts;
     double              mImbalance;
     //std::string         mKPInitFile;
     std::string         mHMetisFile;
     std::string         mPartOutFile;

};

};
#endif // _MMVII_SFMINIT_H_

