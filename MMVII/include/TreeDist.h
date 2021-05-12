/** \file MMVII_TreeDist.h
    \brief  classes for fast computation  of distance in tree

     Computation of distance inside a tree,   also it is an easy operation,
   in some context, like  pattern recoginition, we have to do it millions of
   times on thousands of tree, it is then necessary to have specialized efficient method.

     Let N be the size of the graphe, the computation time of implementation described is :

         * N log(N) for precomputation

         * log(N) in worst case for each tested pair, but probably must better in average
           on all pair  (empiricall 1.5 N^2 for all N^2 pair) ; also this constant
           time maybe optimistic for praticall case as with this algoritm we get 
           longer time for shortest dist, and pratically we may have more short dist
           (with the simulation Avg=1.5 on all pair, but Avg=2.7 with pair corresponding
           to Dist<=3)

    All the library is in the namespace NS_MMVII_FastTreeDist.
*/


// Set false for external use, set true inside MMMVII to benefit from functionality
// with additionnal correctness checking

#ifndef TREEDIST_WITH_MMVII
#define TREEDIST_WITH_MMVII true
#endif


#if (TREEDIST_WITH_MMVII)
#include "include/MMVII_all.h"
#define TREEDIST_cMemCheck MMVII::cMemCheck
#else             //========================================================== WITH_MMVI
class TREEDIST_cMemCheck
{
};
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>
#endif   //========================================================== WITH_MMVI

namespace NS_MMVII_FastTreeDist
{
/* ======================================= */
/* ======================================= */
/* =                                     = */ 
/* = Pre-Declaration of all used classes = */
/* =                                     = */ 
/* ======================================= */
/* ======================================= */

// class offering the services of fast distance in tree, only required interface
class cFastTreeDist;  

// class for basic representation of adjacency graph (could be reused), cFastTreeDist
// inherits of it, but user don't need to know this class for the servive
class cAdjGraph;    

// auxilary, a "cFastTreeDist" contains several "cOneLevFTD"
class cOneLevFTD;   

// class to test the validity of the implementation (definition at end of file)
class cOneBenchFastTreeDist;

/* ======================================= */
/* ======================================= */
/* =                                     = */ 
/* =   Declaration of all used classes   = */
/* =                                     = */ 
/* ======================================= */
/* ======================================= */

/**
    A class for compact representation of graph as adjacency list (ie for each summit
   offer direct access to the set of its neighboor).

     Not especially flexible, rather targeted for specialized algoritm. Could easily 
   evolve to a weighted version.
*/

/*  Example of graph representation by adjajency in this implementation
  
Call to InitAdj  :
        (
              {0,2,3,3,5,7,9},
              {1,0,4,0,6,6,10}
        );
Corresponding to graph :
    1
     \
      0-3-4   5-6-7 8  9-10
     /
    2

We get for  mBeginNeigh mEndNeigh; mBufNeigh; 

mBufNeigh:    | 1 | 2 | 3 | 0 | 0 | 0 | 4 | 3 | 6 | 5 | 7 | 6  ...
              ^           ^   ^   ^       ^   ^ 
mBeginNeigh   0           1   2   3       4   5
mEndNeigh                 0   1   2       3   4

Note that we cannot use  mEndNeigh[X] = mBeginNeigh[X+1] in the implemantation,
to econmize on vector.  Because this property , true a creation,
disappear after suppression ,at least with the fast
suppression we use, and we need to supress edge

For example, if we suppres 1, we get :

mBufNeigh:    | 2 | 3 |###|###| 0 | 0 | 4 | 3 | 6 | 5 | 7 | 6
              ^           ^   ^   ^       ^   ^ 
mBeginNeigh   0           1   2   3       4   5
mEndNeigh              0  1       2       3   4

*/

class cAdjGraph  : public TREEDIST_cMemCheck
{
    public :
       /// We dont want unvolontar copy
       cAdjGraph(const cAdjGraph &) = delete;

       /// create with number of summit, reserve the memory that can be allocat
       inline cAdjGraph(const int & aNbSom);

       /** Creat edge for each pair (aVS1[aK],aVS2[aK]), put it 2 way ,
           remove potential exsiting edge. */
       inline void InitAdj(const std::vector<int> & aVS1,const std::vector<int> & aVS2); 

       inline void SupressAdjSom(const int aS); ///< Supress all the neigboor of S in two way
       inline int  NbNeigh(int) const;  ///<  Number of neigboor
       inline void Show() const; ///< Print a representation
       
       /// Check that the connected component has no cycle
       inline void  CheckIsTree(const std::vector<int> & aCC) const; 
       /// Compute distance with basic algorithm; used here to check the fast implementation
       inline int RawDist(int aS0,int aS1) const;

       /// Compute connected component
       inline void CalcCC(std::vector<int> & aRes,int aS0,std::vector<int> & aMarq,int aMIn,int aMOut);
    protected :
        void AssertOk(int aNumS) const; ///< Check num is valide range

        int mNbSom;                        ///< Number of summits
        std::vector<int *>    mBeginNeigh; ///< Pointer to begin of neighoor
        std::vector<int *>    mEndNeigh;   ///< Pointer to end   of neighboor
        std::vector<int>      mBufNeigh;   ///< Buffer to store the neigh (Begin/end point in it)
};

/** A cFastTreeDist will make several recursive split and store
   the computation of the corresponing level in cOneLevFTD
*/
class cOneLevFTD : public TREEDIST_cMemCheck
{
    public :
        friend class cFastTreeDist;
        
        inline cOneLevFTD(int aNbSom); ///< create alloocating data
    private :
        std::vector<int>      mDistTop; ///< distance to subtree top
        std::vector<int>      mLabels;  ///< Label of subtree
         
};

/**  class offering all necessary services for fast computation of dist in a tree.
     Interface is limited to 3 method

       1- cFastTreeDist => constructor, allocate memory with a number N of summit
       2- MakeDist  => make the precomputation on give set of edges, time "N log(N)"
       3-  Dist => compute the dist between two summit, time log(N) in worst case 
       
*/

class cFastTreeDist : private cAdjGraph 
{
    public :
        //    =====  Constructor / Destructor =====
            /// Copy has no added value, so forbid it to avoid unwanted use
        cFastTreeDist(const cFastTreeDist &) = delete;
            /// just need to know the number of summit to allocate
        inline cFastTreeDist(const int & aNbSom);
        
        /**  Make all the computation necessary to compute dist, if the graph is
        not a forest, generate an error */
        
        inline void MakeDist(const std::vector<int> & aVS1,const std::vector<int> & aVS2); 

        /// compute the distance between two summit; return -1 if not connected
        inline int Dist(int aI1,int aI2) const;

        /// Not 4 computation. Usefull only to make empiricall stat on the number of step
        inline int TimeDist(int aI1,int aI2) const;

    private :
        /// Compute, rescursively, the Quality of S a Origin of a sub tree
        inline void ComputeQualityPivot(int aS);

        /** Explorate, recursively, the connected component :
            - S a summit the "seed" of the CC
            - aLev = the   Level in the recursive call
            - aNumCC = num of the CC, used a level 1 to memorize if 2 summit are connected
        */
        inline void Explorate(int aS,int aLev,int aNumCC);

        bool mShow;         ///< Print message ?
        int  mNbLevel;      ///< Number max of level,
        std::vector<int>    mMarq;      ///< Marker used for connected component (CC)
        std::vector<int>    mNumInsideCC;   ///< reach order in CC, used for orienting graph
        std::vector<int>    mNbDesc;    ///< Number of descend in the oriented tree
        std::vector<int>    mPivotQual;  ///< Quality  to select  as Origin
        std::vector<int>    mNumCC;     ///< Num of initial CC, special case for dist

        std::vector<cOneLevFTD> mLevels;  ///< Stack of levels in recursive split
};

/* ======================================= */
/* ======================================= */
/* =                                     = */ 
/* =   Definition of all used classes    = */
/* =                                     = */ 
/* ======================================= */
/* ======================================= */

/* ---------------------------------- */
/* |         cAdjGraph              | */
/* ---------------------------------- */

cAdjGraph::cAdjGraph(const int & aNbSom) :
    mNbSom    (aNbSom),
    mBeginNeigh   (mNbSom,nullptr),
    mEndNeigh   (mNbSom,nullptr)
{
}

/*  For example


*/

void cAdjGraph::InitAdj(const std::vector<int> & aVS1,const std::vector<int> & aVS2)
{
  bool IsSym = true;
  // 0  vector counting number of succ
  std::vector<int> aVNbSucc(mNbSom,0);

   // 1  count the number of succ
   assert(aVS1.size()==aVS2.size());
   for (int aKS=0 ; aKS<int(aVS1.size()) ; aKS++)
   {
       // chekc they are valide
       AssertOk(aVS1[aKS]);
       AssertOk(aVS2[aKS]);
       // update number of succ
       aVNbSucc[aVS1[aKS]]++;
       if (IsSym)
          aVNbSucc[aVS2[aKS]]++;
   }

   // 2  set the buffer size
   mBufNeigh.resize((IsSym ? 2 : 1)*aVS1.size());

   // 3 initialize the adjacency list by pointing inside the mBufNeigh
   int aSumNb = 0;
   for (int aKS=0 ; aKS<mNbSom ; aKS++)
   {
       mBeginNeigh[aKS] = mBufNeigh.data() + aSumNb;
       mEndNeigh[aKS] =  mBeginNeigh[aKS];
       aSumNb += aVNbSucc[aKS];
   }


  // 5  Finally create adjajency lists
   for (int aKS=0 ; aKS<int(aVS1.size()) ; aKS++)
   {
       // Memorise summit
       int aS1 = aVS1[aKS];
       int aS2 = aVS2[aKS];
       // Add new neigh
  
       *(mEndNeigh[aS1]++) = aS2;
       if (IsSym)
          *(mEndNeigh[aS2]++) = aS1;
   }
}

void cAdjGraph::AssertOk(int aNumS) const
{
    assert(aNumS>=0);
    assert(aNumS<int(mNbSom));
}


void cAdjGraph::Show() const
{
    std::cout << "------------------------------\n";
    for (int aKS1=0 ; aKS1<mNbSom ; aKS1++)
    {
        std::cout << " " << aKS1 << ":"; 
        for (auto aPS2 = mBeginNeigh[aKS1] ; aPS2 <mEndNeigh[aKS1] ; aPS2++)
            std::cout << " " << *aPS2 ;
        std::cout << "\n";
    }
}

void cAdjGraph::SupressAdjSom(const int aS1)
{
    AssertOk(aS1);
    // 1- Parse the neighoor S2 of S1, and supress S1 in the neighboor of S2
    for (auto aS2 = mBeginNeigh[aS1] ; aS2 <mEndNeigh[aS1] ; aS2++)
    {
        int * aBS2In = mBeginNeigh[*aS2]; 
        int * aBS2Out = aBS2In;
        // Copy in aBS2Out the element of aBS2In that are not equal to S1
        while (aBS2In!= mEndNeigh[*aS2])
        {
            if (*aBS2In != aS1)
               *(aBS2Out++) = *aBS2In;
             aBS2In++;
        }
        // Less succ to *aS2
        mEndNeigh[*aS2] = aBS2Out;
    }
    //2- Set  No succ to S1
    mEndNeigh[aS1] = mBeginNeigh[aS1];
}

int  cAdjGraph::NbNeigh(int aS) const
{
    return mEndNeigh[aS] - mBeginNeigh[aS];
}

void cAdjGraph::CalcCC(std::vector<int> & aRes,int aS0,std::vector<int> & aMarq,int aMIn,int aMOut)
{
    assert(aMarq[aS0]==aMIn);
    aMarq[aS0] = aMOut;      // It's explored
    aRes.push_back(aS0);     //  Add it  to component
    int aKS = 0;             ///< index of next som

    while (aKS != int(aRes.size()))
    {
        int aS = aRes[aKS];
        // Parse neigh of current som
        for (auto aNeigh = mBeginNeigh[aS] ; aNeigh <mEndNeigh[aS] ; aNeigh++)
        {
            // If not explored marq it and add it to Res
            if (aMarq[*aNeigh] == aMIn)
            {
                aMarq[*aNeigh] = aMOut;
                aRes.push_back(*aNeigh);
            }
	}
        aKS ++;
    }
}
int cAdjGraph::RawDist(int aS0,int aS1) const
{
    //if (aS0==aS1) 
       //return 0;
    std::vector<int> aVMarq(mNbSom,0);
    std::vector<int> aVSom;   ///< Summits of component

    aVMarq[aS0] = 1;
    aVSom.push_back(aS0);     //  Add it  to component
    int aK0 = 0;             ///< index of next som
    int aK1 = 1;             ///< index of next som
    int aDist = 0;

    while (aK0!=aK1)
    {
        // Here we explore by "generation" to keep track of distance
        for (int aK=aK0; aK<aK1; aK++)
        {
           int aS = aVSom[aK];
           if (aS==aS1) 
              return aDist;
           for (auto aNeigh = mBeginNeigh[aS] ; aNeigh <mEndNeigh[aS] ; aNeigh++)
           {
               if (aVMarq[*aNeigh] == 0)
               {
                   aVMarq[*aNeigh] = 1;
                   aVSom.push_back(*aNeigh);
               }
	   }
        }
        aK0 = aK1;
        aK1 = aVSom.size();
        aDist++;
    }
    return -1;
}


void   cAdjGraph::CheckIsTree(const std::vector<int> & aCC) const
{
   //  Check there is no cycle by consistency nb summit/ nb edges
   int aNbE=0;
   for (const auto & aS : aCC)
       aNbE +=   NbNeigh(aS);

   if (aNbE!= 2*int(aCC.size()-1))
   {
      std::cout << "Not a forest, NbE=" << aNbE << " NbS=" << aCC.size()  << "\n";
      assert(false);
   }
}

/* ---------------------------------- */
/* |         cOneLevFTD             | */
/* ---------------------------------- */

cOneLevFTD::cOneLevFTD(int aNbSom) :
   mDistTop (aNbSom,-1), 
   mLabels  (aNbSom,-1)
{
}

/* ---------------------------------- */
/* |       cFastTreeDist            | */
/* ---------------------------------- */


cFastTreeDist::cFastTreeDist(const int & aNbSom) :
    cAdjGraph    (aNbSom),
    mShow        (false),
    mNbLevel     (1+ceil(log(1+aNbSom)/log(2.0))),
    mMarq        (aNbSom,-1),
    mNumInsideCC (aNbSom),
    mNbDesc      (aNbSom),
    mPivotQual    (aNbSom),
    mNumCC       (aNbSom,-1),
    mLevels      (mNbLevel,cOneLevFTD(aNbSom))
{
}


void cFastTreeDist::ComputeQualityPivot(int aS1)
{
    // Compute the indicator itself
    mPivotQual[aS1] = 0;
    for (auto aPS2 = mBeginNeigh[aS1] ; aPS2 <mEndNeigh[aS1] ; aPS2++)
        mPivotQual[aS1] = std::max(mPivotQual[aS1],mNbDesc[*aPS2]);

    // Now recursive formula
    for (auto aPS2 = mBeginNeigh[aS1] ; aPS2 <mEndNeigh[aS1] ; aPS2++)
    {
       // to go in oriented graph
       if (mNumInsideCC[*aPS2]>mNumInsideCC[aS1])
       {
           // Save current value of  Number of desc
           int aNbDesc1 = mNbDesc[aS1];
           int aNbDesc2 = mNbDesc[*aPS2];

           // Modify  the nuber of desc, reflecting that we change the head of the tree
           mNbDesc[aS1] -= aNbDesc2;
           mNbDesc[*aPS2] += mNbDesc[aS1];
           // recursive call 
           ComputeQualityPivot(*aPS2);

           // Restore previous value
           mNbDesc[aS1] = aNbDesc1;
           mNbDesc[*aPS2] = aNbDesc2;
       }
    }
}

void cFastTreeDist::Explorate(int aS0,int aLev,int aNumCC)
{
    // Probably not necessary, but I have no formall proof that pre computation is enough
    while (int(mLevels.size())<=aLev)
    {
        mLevels.push_back(cOneLevFTD(mNbSom));
    }

    //-1-  Explore connected component, compute mNumInsideCC 
    std::vector<int> aCC;
    mMarq[aS0] = aLev;      // It's explored
    aCC.push_back(aS0);     //  Add it  to component
    {
       int aKS = 0;             ///< index of next som already explore
       while (aKS != int(aCC.size()))  // it is not at the top
       {
           int aS = aCC[aKS];  // get the value and parse its neigboors
           mNumInsideCC[aS] = aKS; // memorize order of reaching
           mNbDesc[aS] = 1;    // Number of desc before propagation
           if (aLev==0)   // Memo num CC to know of two som are disconnected
              mNumCC[aS] = aNumCC;
           for (auto aNeigh = mBeginNeigh[aS] ; aNeigh <mEndNeigh[aS] ; aNeigh++)
           {
               if (mMarq[*aNeigh] == (aLev-1))  // if not explored
               {
                   mMarq[*aNeigh] = aLev;   // marq it exlpored
                   aCC.push_back(*aNeigh);  // and store it
               }
	   }
           aKS ++;
       }
   }
   //2- Check users gave a forest (i.e. this CC is a tree)
   if (aLev==0)
   {
      CheckIsTree(aCC);
   }

   //3- if size =1, we are done
   if (aCC.size()==1)
      return;
 
   // Taking a orientate graph
   //  compute mNbDesc = recursive som of each son
   for (int aK=(aCC.size()-1) ; aK>=0 ; aK--)
   {
        int aS1 = aCC[aK];
        for (auto aPS2 = mBeginNeigh[aS1] ; aPS2 <mEndNeigh[aS1] ; aPS2++)
        {
            if (mNumInsideCC[*aPS2]<mNumInsideCC[aS1])
               mNbDesc[*aPS2] += mNbDesc[aS1];
        }
   }
   //4- Make the computation of quality as origin on all the component
   ComputeQualityPivot(aS0);
 
   //5- The kern is the summit minimizing the kern indicator
   int aBestOrig    = aCC[0];
   int aQualOrig =  mPivotQual[aBestOrig];
   for (int aK=1 ; aK<int(aCC.size()) ; aK++)
   {
       int aS = aCC[aK];
       int aCrit = mPivotQual[aS];
       if (aCrit<aQualOrig)
       {
           aBestOrig  = aS;
           aQualOrig = aCrit;
       }
   }

   // 6 - Compute distance to origin and labels of subtree
   //     6.1  compute value of origins and its neighboor
   std::vector<int> &  aDistTop =  mLevels.at(aLev).mDistTop;
   std::vector<int> &  aLabels  =  mLevels.at(aLev).mLabels;
   aDistTop[aBestOrig] = 0;
   aLabels[aBestOrig]  = 0;
   int aLab=1;
   for (auto aNeigh = mBeginNeigh[aBestOrig] ; aNeigh <mEndNeigh[aBestOrig] ; aNeigh++)
   {
        // aDistTop[*aNeigh] = 1;
        aLabels[*aNeigh] = aLab;
        aLab++;
   }
   // 6.2 Now a connected component again starting from origin 
   // to propagate distance & labels
   aCC.clear();
   aCC.push_back(aBestOrig);     //  Add it  to component
   {
       int aKS = 0;             ///< index of next som already explore
       while (aKS != int(aCC.size()))  // it is not at the top
       {
           int aS = aCC[aKS];  // get the value and parse its neigboors
           for (auto aNeigh = mBeginNeigh[aS] ; aNeigh <mEndNeigh[aS] ; aNeigh++)
           {
               if (aDistTop[*aNeigh] == -1)
               {
                   aCC.push_back(*aNeigh);  // and store it
                   aDistTop[*aNeigh] = 1+aDistTop[aS];
                   if (aDistTop[aS]>0)
                       aLabels[*aNeigh] =aLabels[aS];
               }
	   }
           aKS ++;
       }
    }
   
    if (mShow)
    {
       std::cout << "L=" << aLev << " [S,NbD,Qual,Dist,Lab] " ;
       for (auto aS : aCC)
       {
           std::cout << " [" 
                     << aS            << "," 
                     << mNbDesc[aS]   << "," 
                     << mPivotQual[aS] << ","
                     << aDistTop[aS] << ","
                     << aLabels[aS] 
                     << "]";
           if (aS==aBestOrig)
              std::cout << "*";
        }
        std::cout << "\n";
    }

    // 7 Recursive call on trees after supression of origin
    SupressAdjSom(aBestOrig);
    for (const auto  & aS : aCC)
    {
        if ((aS!=aBestOrig) && (mMarq[aS]==aLev))
        {
            Explorate(aS,aLev+1,-1);  // -1 for NumCC as it is unused after lev 0
        }
    }
}

void cFastTreeDist::MakeDist(const std::vector<int> & aVS1,const std::vector<int> & aVS2)
{
    InitAdj(aVS1,aVS2);

    // Reset all state
    for (auto & aM : mMarq)
        aM = -1;
    for (auto & aL : mLevels)
        for (auto & aD : aL.mDistTop)
            aD=-1;

    int aNumCC=0;
    for (int aS=0 ; aS<mNbSom ; aS++)
    {
         if (mMarq[aS] == -1)
         {
            Explorate(aS,0,aNumCC);
            aNumCC++;
         }
    }
}

int cFastTreeDist::TimeDist(int aI1,int aI2) const
{
   if (aI1==aI2) return 1; // Case not well handled else
   if (mNumCC[aI1]!=mNumCC[aI2]) return 1;

   for (int aK=0 ; aK<int(mLevels.size()) ; aK++)
       if (mLevels[aK].mLabels[aI1]!=mLevels[aK].mLabels[aI2])
          return aK+1;

   assert(false);
   return 0;
}

int cFastTreeDist::Dist(int aI1,int aI2) const
{
   if (aI1==aI2) return 0; // Case not well handled else

   // Case not connected; return conventionnal value
   if (mNumCC[aI1]!=mNumCC[aI2]) return -1;

   for (const auto & aLev : mLevels)
       if (aLev.mLabels[aI1]!=aLev.mLabels[aI2])
          return aLev.mDistTop[aI1]+aLev.mDistTop[aI2];

   assert(false);
   return 0;
}

/* ---------------------------------- */
/* |   cOneBenchFastTreeDist        | */
/* ---------------------------------- */

// Some basic rand function defined in MMVII, used to generated random tree
inline int TREEDIST_RandUnif_N(int aNb) { return rand() % aNb; }
#define TREEDIST_NB_RAND_UNIF 1000000
inline float TREEDIST_RandUnif_0_1() { return TREEDIST_RandUnif_N(TREEDIST_NB_RAND_UNIF) / float(TREEDIST_NB_RAND_UNIF); }


/** Class to check correctness of implemantion. Basically, the serice offered in
    the main method are :

      - create random forest to test many configuration
      - compute fast dist  on any pair of summit
      - compute distance with basic method
      - check that the two computation give the same result
*/

class cOneBenchFastTreeDist
{
    public :
      inline cOneBenchFastTreeDist(int aNbSom,int aNbCC);
      inline void MakeOneTest(bool Show,bool ChekDist);

      double  AvgT() {return mSomAvgT / mNbTest;}
      double  SomAvgLowT() {return mSomAvgLowT / mNbTest;}

    private :
      int                mNbSom; ///< Number of summit
      int                mNbCC;  ///< Number of connected components
      cFastTreeDist      mFTD;   ///< Fast tree to compute fast distance
      cAdjGraph          mAdjG;  ///< Adjence Graph to check validity of fast distance

      int                mNbTest;
      double             mSomAvgT;
      double             mSomAvgLowT;
};

cOneBenchFastTreeDist::cOneBenchFastTreeDist(int aNbSom,int aNbCC) :
   mNbSom  (aNbSom),
   mNbCC   (std::min(aNbCC,aNbSom)),
   mFTD    (mNbSom),
   mAdjG   (mNbSom),
   mNbTest (0),
   mSomAvgT(0.0),
   mSomAvgLowT(0.0)
   // mGr     (nullptr)
{
}

void cOneBenchFastTreeDist::MakeOneTest(bool Show,bool CheckDist)
{ 
    mNbTest ++;
    //========= 1 Generate a random forest with mNbCC components =====================

          //  ----  1.1 generate the label of the CC -------------
    std::vector<std::vector<int>> aVCC(mNbCC); // set of connected components
    for (int aK=0 ; aK<mNbSom ; aK++)
    {
        int aNum = TREEDIST_RandUnif_N(mNbCC); 
        aVCC.at(aNum).push_back(aK);  // Add a summit inside the CC aNum
    }

          //  ----  1.2 generate the graph -------------
    std::vector<int> aV1;
    std::vector<int> aV2;
    for (auto & aCC : aVCC) // For each CC
    {
        // order randomly the CC
        std::vector<double> aVR;
        for (int aK=0 ; aK<int(aCC.size()) ; aK++)
        {
            aVR.push_back(TREEDIST_RandUnif_0_1());
        }
        std::sort
        (
            aCC.begin(),aCC.end(),
            [aVR](int i1,int i2) {return aVR[i1]<aVR[i1];}
        );
        for (int aK1=1 ; aK1<int(aCC.size()) ; aK1++)
        {
             // We add CC[aK] to  the already created tree we select
             // randomly a summit inside this tree
             double aPds = sqrt(TREEDIST_RandUnif_0_1());  // Bias to have longer chain
             int aK2 = floor(aPds*aK1);  // we rattach K1 to K2
             aK2 = std::max(0,std::min(aK2,aK1-1));  // to be sure that index is correct
             aV1.push_back(aCC[aK1]);
             aV2.push_back(aCC[aK2]);
        }
    }
    mFTD.MakeDist(aV1,aV2);  // Create Dist with this graph
    mAdjG.InitAdj(aV1,aV2);  // Make a copy of the same graph for checking

    if (CheckDist)
    {
       for (int aS1=0 ; aS1<mNbSom ; aS1++)
       {
           for (int aS2=0 ; aS2<mNbSom ; aS2++)
           {
               int aD= mFTD.Dist(aS1,aS2);  // Fast distance
               int aD2= mAdjG.RawDist(aS1,aS2);  // Easy algorihtm to check
               assert(aD==aD2);
           }
       }
    }

    // Is show and 1 Connected component make stat on the time for distance
    if ( (mNbCC==1) && (mNbSom>1))
    {
       int aNbTest =0;  // Number of test
       int aSomT  =0;   // Sum of time (=number of step)
       int aMaxT  =0;   // Max of time

       const int aThresholdD = 3;  // Threshold for "low" distances
       int aSomTLowD  = 0;   // Sum of time on low distances
       int aNbLowD   = 0;   // number of test on low distance

       for (int aS1=0 ; aS1<mNbSom ; aS1++)
       {
           for (int aS2=0 ; aS2<aS1 ; aS2++)
           {
               aNbTest++;
               int aD = mFTD.Dist(aS1,aS2);
               int aT = mFTD.TimeDist(aS1,aS2);
               aSomT += aT;
               aMaxT = std::max(aMaxT,aT);
               if (aD<=aThresholdD)
               {
                  aSomTLowD += aT;
                  aNbLowD++;
               }
           }
       }
       mSomAvgT += aSomT/double(aNbTest);
       mSomAvgLowT += aSomTLowD/double(aNbLowD);
       if (Show)
       {
          std::cout << "Nb=" << mNbSom  
                    << " AvgT=" << aSomT/double(aNbTest) 
                    << " MaxT=" << aMaxT
                    << " AvgTLow=" << aSomTLowD/double(aNbLowD) 
                    << "\n";
       }
    }
}

/* ---------------------------------- */
/* |   Global function              | */
/* ---------------------------------- */

inline void OneBenchFastTreeDist(int aNbSom,int aNbCC,bool Show)
{
    cOneBenchFastTreeDist aBFTD(aNbSom,aNbCC);

    // We call 4 MakeOneTest to check that several call to MakeDist on the same cFastTreeDist is fine
    for (int aK=0 ; aK<4 ; aK++)
    {
        aBFTD.MakeOneTest(Show,true);
    }
    if (Show)
       std::cout << "DONNE  "<<  aNbSom << " " << aNbCC << "\n";
}

inline void AllBenchFastTreeDist(bool Show,int aNbTest)
{
    for (int aKTest=1 ; aKTest<=aNbTest ; aKTest++)
    {
        // We want to test various size , including high size, but to go relatively faste
        // test only square 1 4 9 ... 225
        int aNb2 = aKTest*aKTest;
        // We test with various number of connected components
        OneBenchFastTreeDist(aNb2,1,Show);
        OneBenchFastTreeDist(aNb2+1,2,Show);
        OneBenchFastTreeDist(aNb2+2,3,Show);
    }
}

inline void StatTimeBenchFastTreeDist()
{
    for (int aNb=2 ; aNb<=30 ; aNb++)
    {
        int aNb2 = aNb*aNb;
        cOneBenchFastTreeDist aBFTD(aNb2,1);
        for (int aK=0 ; aK<20 ; aK++)
        {
            aBFTD.MakeOneTest(false,false);
        }
        std::cout << "Nb " << aNb2 
                  << " " << aBFTD.AvgT()  
                  << " " << aBFTD.SomAvgLowT()
                  << " " << aBFTD.SomAvgLowT() / log(aNb2)
                  << "\n";
    }
}

};



