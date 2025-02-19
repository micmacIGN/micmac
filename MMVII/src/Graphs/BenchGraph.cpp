#include "MMVII_util_tpl.h"
#include "MMVII_Bench.h"
#include "cMMVII_Appli.h"
#include "MMVII_Images.h"

#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"


namespace MMVII
{

/**  \file  : BenchGraph.cpp

     \brief : containt test&tutorial on graphs, based on a grid-graph (vertices on regular grid)


     This file contains test and tuto on basic graph manipulation. The vertices of the graph are put on a regular
   grid inside a given rectangle,  the attribute of each vertex V contain the pixel "P=(i,j)".  For commodity,
   the vertices are indexed and the function "VertOfPt(P)" return  the vertex associated to a pixel.

         The connectivity is done between all pair of "adjacent pixels" , i.e   (i1,j1) and  (i2,j2) such that :
         
                        max(|i1-i2|,|j1-j2|) <= 1

      Also, to each vertex we associate in the attribute a "curvilinear asbsisca" illustrated bellow.

      Part of the test is done using the fact that property in the graph can be computed from property on the grid. For
   example :

           - shortest path with value of edge to 1,  we check that  d(V1,V2) = Norm1(P1,P2)
           - shortest path with restriction to 4-neighboor, we check that  d(V1,2) = NormInf(P1,P2)
           - shortest path with weithing (A1-A2)^2,  we check that  d(V1,V2) = |A1-A2|


         Once the graph constructed , we make 4 kind of test :

            - test on  elementary access to vertex, edge ... in  "Bench_FoncElem"

            - test on  connected components in "Bench_ConnectedComponent", this test is made using a sub-graph
              such that the edges betwen V1 and V2 is in the graph iff  y1=y2

            - test on  shortest path algoritm with the 3 distances mentionned above in  Bench_ShortestPath

            - test on  minimum spaning tree and forest in "Bench_MinSpanTree",  we check the fundamental
              property of such tree :
                      * for any edge E0, is we supress E0, we have 2 connected components C1 and C2
                      * for any edge E12 joining a vertex of C1 to a vertec of C2, we must have 
  
                                     Cost(E0) <= Cost(E12)
         
                       * else, by supressing E0 and adding E12, we would improve the cost of the tree
*/


/*  @FIG:CURV:ABS

    Value of the curvilinear absisca computed on a regular grid  (4 x 5)  we  print  [x,y]A


        [0,4],16 --> [1,4],17--> [2,4],18 -> [3,4] 19
         / \ 
          |
          |
        [0,3],15 <-- [1,3],14 <- [2,3],13 <- [3,3] 12
                                              / \ 
                                               |
                                               |
        [0,2],8  --> [1,2],9 --> [2,2],10 -> [3,2] 11
         / \ 
          |
          |
        [0,1],7  <-- [1,1],6 <-- [2,1],5 <-- [3,1] 4
                                              / \ 
                                               |
                                               |
        [0,0],0  --> [1,0],1 --> [2,0],2 --> [3,0] 3
*/


/* *********************************************************** */

/* The 3 type of attributes used to instantiate the graph */

class cBGG_AttrVert;   // Class Bench Graph Grid  Attribute of Vertex 
class cBGG_AttrEdgOr;  // Class Bench Graph Grid  Attribute of Oriented Edge
class cBGG_AttrEdgSym; // Class Bench Graph Grid  Attribute of Symetric Edge


/*  we will create a graph  of type cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym>  */

/**  Atribute of vertices */
class cBGG_AttrVert
{
     public :
           cBGG_AttrVert(cPt2di  aPt,tREAL8 aZ) :
               mPt       (aPt),
               mAbsCurv  (aZ)
           {
           }
           cPt2di mPt;       // "pixel" of the vertex
           tREAL8 mAbsCurv;  // curvilinear absisca as described above
};

/**  Oriented attribute of edges  */
class cBGG_AttrEdgOr
{
     public :
           cBGG_AttrEdgOr(tREAL8 aDeltaAC) : 
		   mDeltaAC (aDeltaAC) 
	   {}
           tREAL8 mDeltaAC;  // signed difference between curvilinear absisca 
     private :
};

/**  Symetric attribute of edges  */
class cBGG_AttrEdgSym
{
     public :
           cBGG_AttrEdgSym(tREAL8 aDist) : 
                mDist     (aDist) ,
                mRanCost  (RandUnif_C()),
                mIsOk     (true)
           {}
           tREAL8 mDist;      // euclidean distance, not really use 4 now
           tREAL8 mRanCost;   // use to modify the weighting  
           bool   mIsOk;      // use to creat some specific subgraph
     private :
};

/**  Class for "grid" graph */

class  cBGG_Graph : public cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym>
{
     public :
          typedef cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym> tGraph;
          typedef typename tGraph::tVertex                              tVertex;
          typedef  tVertex*                                             tPtrV;
          typedef  cAlgoSP<cBGG_Graph>                                  tAlgoSP;
          typedef  typename tAlgoSP::tSetPairVE                         tSetPairVE;
          typedef  cAlgoCC<cBGG_Graph>                                  tAlgoCC;
          typedef  tAlgoSP::tForest                                     tForest;
          typedef  cAlgo_ParamVG<cBGG_Graph>                            tParamA;


          cBGG_Graph (cPt2di aSzGrid); /// constructor , take the size of the grid

          /// Vertex of a given pixel, return reference to a pointer so that it can be used also in init
          tPtrV & VertOfPt(const cPt2di & aPix) 
          {
               return mGridVertices.at(aPix.y()).at(aPix.x());
          }

          /// test creation , acces to neighbours, attributes
          void Bench_FoncElem();  
          ///  Test the algorithms
          void BenchAlgos();
     private :
          /// test connected component, for a single pixel or a region
          void Bench_ConnectedComponent(tVertex * aV0,tVertex * aV1);
          /// Test computation of shortest path between 2 vertices, mode-> correspond to different distance/conexion
          void Bench_ShortestPath(tVertex * aV0,tVertex * aV1,int aMode);
          
          /// Test minimum spaning tree & forest
          void Bench_MinSpanTree(tVertex * aSeed);
          /// Check that computed a tree on a subgraph is a minimum spaning
          void Check_MinSpanTree(const tSetPairVE& aSet,const tParamA& );

          cPt2di                             mSzGrid;       ///< sz of the grid
          cRect2                             mBox;          ///< Box associated to the  
          std::vector<std::vector<tPtrV>>    mGridVertices; ///< such the mGridVertices[y][x] -> tVertex *
          void AddEdge(const cPt2di&aP0,const cPt2di & aP1); ///< Add and edge bewteen vertices corresponding to 2 Pix
};


cBGG_Graph::cBGG_Graph (cPt2di aSzGrid) :
      cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym>(),  
      mSzGrid        (aSzGrid),
      mBox           (mSzGrid),
      // Allocate the vector of vector
      mGridVertices  (mSzGrid.y(),std::vector<tVertex*>(mSzGrid.x(),nullptr))  
{
      //  ----------- [0]  Create the vertices ------------------------
      for (const auto & aPix : mBox)  // parse all pixel
      {
         // Formula to create curvilinear absisca corresponding to  @FIG:CURV:ABS
          cPt2di aPix4Num 
                 (
                   (aPix.y()%2) ? (mSzGrid.x()-aPix.x()-1) : aPix.x(),
                   aPix.y()
                 );
          tREAL8 anAbsicsa = mBox.IndexeLinear(aPix4Num);

          // create a vertex and put it on the grid
          VertOfPt(aPix) = NewSom(cBGG_AttrVert(aPix,anAbsicsa));
      }

      //  ----------- [1]  Create the edges ------------------------
      for (const auto & aPix : mBox)
      {
           // consider the 8-neighbour, but do it only in one way as the edges
           // are stored two  ways
           AddEdge(aPix,aPix+cPt2di(0,1));
           AddEdge(aPix,aPix+cPt2di(1,0));
           AddEdge(aPix,aPix+cPt2di(1,1));
           AddEdge(aPix,aPix+cPt2di(-1,1));
      }
}

void cBGG_Graph::AddEdge(const cPt2di&aP1,const cPt2di & aP2)
{
    // Check both pix are in box
    if ( (!mBox.Inside(aP1)) || (!mBox.Inside(aP2)) )
       return;

    tVertex * aV1 = VertOfPt(aP1);  // extract Vertex 1
    tVertex * aV2 = VertOfPt(aP2);  // extract Vertex 2

    tREAL8 aDist = Norm2(aV1->Attr().mPt-aV2->Attr().mPt);   // euclidean distance
    tREAL8 aDeltaAbs12 = aV2->Attr().mAbsCurv-aV1->Attr().mAbsCurv; // difference of asbisca

    // just for the "fun" create different attribute with orientation
    cBGG_AttrEdgOr aA12(aDeltaAbs12);
    cBGG_AttrEdgOr aA21(-aDeltaAbs12);

    cBGG_AttrEdgSym  aASym(aDist);

    // Add an edge with the 2 edges and 3 attributes
    tGraph::AddEdge(*aV1,*aV2,aA12,aA21,aASym);
}

void cBGG_Graph::Bench_FoncElem()
{
      for (size_t aKV1=0 ; aKV1<NbVertex() ; aKV1++) // parse all vertices
      {
          tVertex &   aV1 = VertexOfNum(aKV1);
          cPt2di aP1 = aV1.Attr().mPt;        // extract pixelof vertex
          size_t aNbSucc=0;
          // parse the 8 neighbourhoud, 
          for (const auto aDPt : cRect2::BoxWindow(2))
          {
              cPt2di aP2 = aP1+aDPt;
              bool OkInside =  mBox.Inside(aP2) ;  // is P2 in the grid ?
              bool OkSucc =  OkInside &&  (NormInf(aDPt)==1);  // should there exit an edge
              aNbSucc += OkSucc;
              if (OkSucc)
              {
                 // is there must be an edge, check that we can extract it and get its attribute
                 tVertex &   aV2 = *VertOfPt(aP2);
                 const tEdge * anE12=  aV1.EdgeOfSucc(aV2);
                 MMVII_INTERNAL_ASSERT_bench(std::abs(anE12->AttrSym().mDist-Norm2(aP1-aP2))<1e-10,"Dist in cBGG_Graph");
                 MMVII_INTERNAL_ASSERT_bench(aV2.Attr().mAbsCurv-aV1.Attr().mAbsCurv==anE12->AttrOriented().mDeltaAC,"Dz in cBGG_Graph");
                 MMVII_INTERNAL_ASSERT_bench( (&anE12->Succ() ==  &aV2) ," Adrr in cBGG_Graph");

		 MMVII_INTERNAL_ASSERT_bench(anE12->IsDirInit()!=anE12->EdgeInv()->IsDirInit(),"DirInit /DirInv");
		 MMVII_INTERNAL_ASSERT_bench(anE12->EdgeInitOr()->IsDirInit(),"DirInitOriented");
              }
              else if (OkInside)
              {
                 // is there must be no edge, check that in this case EdgeOfSuc return nullptr
                 tVertex &   aV2 = *VertOfPt(aP2);
                 MMVII_INTERNAL_ASSERT_bench(aV1.EdgeOfSucc(aV2,SVP::Yes)==nullptr,"EdgeOfSucc in cBGG_Graph");
              }
          }
          MMVII_INTERNAL_ASSERT_bench(aNbSucc==aV1.EdgesSucc().size(),"NbSucc in cBGG_Graph");
      }
}

void cBGG_Graph::Bench_ShortestPath(tVertex * aV0,tVertex * aV1,int aMode)
{
    /*   We  compute the shortest path between V0 and V1 with 3 variation possible  depending on aMode :

        - [0]  we use a parameter of class cAlgo_ParamVG<cBGG_Graph>, all weight are equal to 1, and all the connexion
          are valid,  in this case the distance DG in the graph  must be equal to :
  
                  *  DG= max(|x1-x2|,|y1-y2|) = NormInf(P1-P2) 

         
        - [1]  we use the parameter of class cNeigh_4_Connex, weight are still equals to 1, but  validate only the edges 
           corresponding to 4-neigbooroud, in this case the the distance in the graph  must be equal to :

                  * DG = |x1-x2|+|y1-y2| = Norm1(aP1,aPyy2)

        - [2]  we use the parameter of class cWeighSqDeltaAbs, all the existing connexion ar still valide, but the weighting 
               is the square difference between curvilinear absisca :

                      W(E(P1,P2)) = (Abs1-Abs2) ^ 2

              in this case, it can be proved mathematically that the path that minimize the cost is such that |Abs1-Abs2|=1
              for all consecutive neighbours, and consequently for the path P1 to P2 :

                     DG = |Abs(P1)-Abs(aP2)|  

              The shortest path with this distance follow the "snake" of figure  @FIG:CURV:ABS

    */

    class cNeigh_4_Connex : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
         // this formula validate the edge iff  |x1-x2|+|y1-y2| <= 1
         bool   InsideEdge(const    tEdge & anE) const override 
		{ 
                     return Norm1(anE.VertexInit().Attr().mPt-anE.Succ().Attr().mPt)<=1; 
                }
    };
    class cWeighSqDeltaAbs : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
             // weighting is square difference of curvilinear absisca
             tREAL8 WeightEdge(const tVertex &,const    tEdge & anE) const override
             { 
                     return Square(anE.AttrOriented().mDeltaAC);
             }
    };



    cAlgo_ParamVG<cBGG_Graph> aParam0;
    cNeigh_4_Connex           aParam1;
    cWeighSqDeltaAbs          aParam2;

    std::vector<cAlgo_ParamVG<cBGG_Graph>*>  aVParam{&aParam0,&aParam1,&aParam2};

    cAlgo_ParamVG<cBGG_Graph> *aParam = aVParam.at(aMode);

    tVertex *  aTarget =  tAlgoSP::ShortestPath_A2B(*this,*aV0,*aV1,*aParam);

    //  compute the theoreticall distance according to previous discussion
    int aDTh = NormInf(aV0->Attr().mPt -aV1->Attr().mPt);
    if (aMode==1) aDTh = Norm1(aV0->Attr().mPt -aV1->Attr().mPt);
    if (aMode==2) aDTh = std::abs(aV0->Attr().mAbsCurv -aV1->Attr().mAbsCurv);


     // check that cost in V1 is equal to the theoreticall distance in the considered graph
     MMVII_INTERNAL_ASSERT_bench(aDTh==aV1->AlgoCost(),"Cost in shortest path"); 
     // check that, with a single reachable vertex, result of shortest path is the singleton goal
     MMVII_INTERNAL_ASSERT_bench(aTarget==aV1,"Target in shortest path");

     /*  Now check the function "BackTrackFathersPath" that compute by "back-tracking" the path going from one target
         to the seed.  
     */
     std::vector<tVertex*> aPath = aTarget->BackTrackFathersPath();
     // first check, the seed being {aV0}, it is the end of the back-tracked path
     MMVII_INTERNAL_ASSERT_bench(aV0==aPath.back(),"seed in BackTrackFathersPath");

     //  in all 3 case the edge used by shortest path have a lenght of 1, this is obvious for case 0 and 1
     // because all edge have  lenght 1, for case 2 this is  a consequence of convexity of cost function
     // so we check the global cost 
     MMVII_INTERNAL_ASSERT_bench(aTarget->AlgoCost()+1==aPath.size(),"size in BackTrackFathersPath");

     // we also check that consecutive vertices are connected according to the sub-graph
     for (size_t aKV=1 ; aKV<aPath.size() ; aKV++)
     {
          int aD = NormInf(aPath.at(aKV-1)->Attr().mPt-aPath.at(aKV)->Attr().mPt);
          if (aMode==1)  aD = Norm1(aPath.at(aKV-1)->Attr().mPt-aPath.at(aKV)->Attr().mPt);
          MMVII_INTERNAL_ASSERT_bench(aD==1,"Succ in BackTrackFathersPath");
     }
}

/*  Test on connected component extraction,  doing it with the full graph would be "boring", we would
    get the full grid each time.  So we consider the sub-graph defined by :

        (P1 <=> P2)   iff  (x1==x2)

     Visually speaking, the connected component are "columns" of the grid. Given 2 point/vertex P0 and P1, we make
     the following check on:

          *  the connected component of P0

          * all multiple  connected components in rectangle [P0,P1]

          * all the connected components of the graph

*/

void cBGG_Graph::Bench_ConnectedComponent(tVertex * aSeed,tVertex * aV1)
{
    // Class for defining the subgraph  P1<=>P2  iff  x1=x2
    class cGrVert : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
         bool   InsideEdge(const    tEdge & anE) const override 
		{ 
                     return anE.VertexInit().Attr().mPt.x()==anE.Succ().Attr().mPt.x(); 
                }
    };

    cGrVert aParamVert;

    /* -----------[0] check on connected component of aSeed --------------------------------*/
    std::vector<tVertex*>  aCC = tAlgoCC::ConnectedComponent(*this,*aSeed,aParamVert);

    //  [0.1]   its number of point must be equal to the height of the rectangle
    MMVII_INTERNAL_ASSERT_bench(aCC.size()==(size_t)mSzGrid.y(),"Size in Bench_ConnectedComponent");
    //  [0.2]   check that all vertices have the "x" equal to aSeed.x
    for (auto aVertex : aCC)
        MMVII_INTERNAL_ASSERT_bench(aSeed->Attr().mPt.x()==aVertex->Attr().mPt.x(),"x-diff in Bench_ConnectedComponent");


    /* -----------[1] check Multiple_ConnectedComponent  --------------------------------*/

          // [1.0]  create a vector containing all vertices in rectangle [P0,P1[ 
    std::vector<tVertex *> aVecSeed; // vector to store all vertices in [P0,P1[
    cPt2di aP0 = aSeed->Attr().mPt;  // extract P0
    cPt2di aP1 = aV1->Attr().mPt;    // extract P1
    MakeBox(aP0,aP1);                // assure that x0<=x1 and y0<=y1

           // push  vertices of rectangle in aVecSeed
    for (int aX=aP0.x() ; aX<aP1.x() ; aX++)
        for (int aY=aP0.y() ; aY<aP1.y() ; aY++)
            aVecSeed.push_back(VertOfPt(cPt2di(aX,aY)));
 
          // [1.1]  extract the multiple connected component
    std::list< std::vector<tVertex*>> aVecCC =  tAlgoCC::Multiple_ConnectedComponent(*this,aVecSeed,aParamVert);

          // [1.2.0]  compute the theoreticall number of components = witdh in x, except if height in y =0
    int aThNbCC = (aP1.y()==aP0.y()) ? 0 :(aP1.x()-aP0.x());
          // [1.2.1]  check that computed number of CC is equal to theoreticall value
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == aThNbCC,"NbCC in Multiple_ConnectedComponent");

          // [1.3]  check each connected component
    for (const auto & aCC : aVecCC)
    {
        // [1.3.0]  check the size of each CC is the height
        MMVII_INTERNAL_ASSERT_bench(aCC.size() == (size_t)mSzGrid.y() ,"sz of CC in Multiple_ConnectedComponent");
        // [1.3.1]  check that all vertex of CC have the same x
        for (const auto & aV : aCC)
        {
            MMVII_INTERNAL_ASSERT_bench(aV->Attr().mPt.x()== aCC.at(0)->Attr().mPt.x(),"XX in Multiple_ConnectedComponent");
        }
    }

    /* -----------[2] check  All_ConnectedComponent --------------------------------*/

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,aParamVert);

    // check the number of CC, and the size of each CC
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == mSzGrid.x(),"NbCC in All_ConnectedComponent");
    for (const auto & aCC : aVecCC)
        MMVII_INTERNAL_ASSERT_bench((int) aCC.size()==mSzGrid.y(),"Size in All_ConnectedComponent");
}

/*    For a given set of pair and a sub graph defined by "aParam", defining a tree "T", check that it is the minimum 
    spaning tree in sub-graph a aParam.  Method used :

          - for all pair E=(V1,V2)  of "T"
          - consider the sub-tree obtained by supressing E, it gives two connected component C1 and C2
          - check that any edge connected C1 to C2 have a cost >= to E

       For this we use the field "mIsOk" of  symetric attribute :
   
           - initially mIsOk is true only for the edges of "T"
           - for each edge, we set mIsOk to false
           - we consider a sub-graph defined by  "edge belong to G" :  iff  "mIsOk is true"
*/


void cBGG_Graph::Check_MinSpanTree(const tSetPairVE& aSetPair,const tParamA& aParam )
{
     // class of subgr containing edge such that mIsOk is true
     class cASymOk : public  cAlgo_ParamVG<cBGG_Graph>
     {
       public :
         bool   InsideEdge(const    tEdge & anE) const override {return anE.AttrSym().mIsOk;}
     };
 
      // ---------------  [0]  set mIsOk iff it belong to tree "aSetPair" ------------
     
         // [0.1] set all edge to false
     for (const auto & aPtrAttrSym : this->AllAttrSym())
        aPtrAttrSym-> mIsOk = false;
         // [0.1] set edges of tree to true
     for (const auto & [aV1,anE] : aSetPair )
         anE->AttrSym().mIsOk = true;


     //  ------------ [1] parse all edge and supress it alternatively ------------------
     for (const auto & [aV1,anETree] : aSetPair )
     {
         // [1.0]  supress the edge in subgraph
         anETree->AttrSym().mIsOk = false;

         // [1.1] extract the 2 connected components
         std::vector<tVertex*>  aCC1 = tAlgoCC::ConnectedComponent(*this,*aV1,cASymOk());
         std::vector<tVertex*>  aCC2 = tAlgoCC::ConnectedComponent(*this,anETree->Succ(),cASymOk());

         // [1.2]   check the sum of size of CC that must be equal to total number of vertices of the tree
         MMVII_INTERNAL_ASSERT_bench(aCC1.size() + aCC2.size()==aSetPair.size()+1,"Sizes in Check_MinSpanTree");

         tREAL8 aCostCut =anETree->AttrSym().mRanCost;  // store the cost of the "cut"
         // [1.3]  parse all pairs of  "C1 X C2"
         for (const auto & aV1 : aCC1)
         {
              for (const auto & aV2 : aCC2)
              {
                  const tEdge * anE12 = aV1->EdgeOfSucc(*aV2,SVP::Yes);
                  // if it is an Edge AND it is in the subgraph
                  if (anE12 &&  aParam.InsideEdge(*anE12))
                  {
                      // [1.3.1] then it cost cannot be better than the cut
                      MMVII_INTERNAL_ASSERT_bench(anE12->AttrSym().mRanCost>= aCostCut,"Sizes in Check_MinSpanTree");
                  }
              }
         }
         

         // [1.4]  restore the edg
         anETree->AttrSym().mIsOk = true;
     }

     // [2]  just in case, restore mIsOk
     for (const auto & aPtrAttrSym : this->AllAttrSym())
        aPtrAttrSym->mIsOk = true;
}

/*  Make the test on spaning tree.  :

    First test :

        - define a weighted graph using the attribute RanCost
        - compute the global spaning tree with this cost
        - check that it is the global minimal spaning tree

    Second test :

         - define a graph G(Thr)  with previous weigthing and a selection of edges such that :
                RanCost > Threshold

         - compute the minimal  spaning forest "F" of G(Thr)
         - check  that each tree of "F" complies with the properties of "Check_MinSpanTree"
*/

void cBGG_Graph::Bench_MinSpanTree(tVertex * aSeed)
{
    // class for defining a graph weighted by RanCost
    class cWRanCost : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
             tREAL8 WeightEdge(const tVertex &,const    tEdge & anE) const override
             { 
                 return anE.AttrSym().mRanCost;
             }
    };

    // class for defining a sub-graph with selection of edge  such that "mRanCost>mThreshold"
    class cRC_Thr : public  cWRanCost
    {
       public :
             bool InsideEdge(const    tEdge & anE) const override
             { 
                  return anE.AttrSym().mRanCost > mThreshold;
             }
             cRC_Thr(const tREAL8 aThrs) : mThreshold (aThrs) {}
             tREAL8  mThreshold;
    };

    // [0]  make a random initialization of costs
    for (const auto & aPtrAttrSym : this->AllAttrSym())
        aPtrAttrSym-> mRanCost = RandUnif_C();

    // [1]  compute global minimal spaning tree
    tSetPairVE aSetPair = tAlgoSP::MinimumSpanninTree(*this,*aSeed,cWRanCost()).second;
    MMVII_INTERNAL_ASSERT_bench((int)aSetPair.size()==(mBox.NbElem()-1),"Size in All_ConnectedComponent");
    //  [1.0]  check that it is effectively the minimal spaning tree
    Check_MinSpanTree(aSetPair,cAlgo_ParamVG<cBGG_Graph>());

    // [2]  compute a minimal spaning forest
    cRC_Thr  aRC(1.0 - 2*std::pow(RandUnif_0_1(),2.0));  // define a sub-graph
    tForest aForest;
    tAlgoSP::MinimumSpanningForest(aForest,*this,this->AllVertices(), aRC);  // extract the forest

    size_t aNbEdge = 0;
    // check that each tree of the forest complies with Check_MinSpanTree
    for (const auto & [aVert,aVPair] : aForest)
    {
        Check_MinSpanTree(aVPair,aRC);
        aNbEdge += aVPair.size();
    }
    // check on number of edge, +- Eulers relation for a graph w/o cycle
    MMVII_INTERNAL_ASSERT_bench(aForest.size()+aNbEdge== (size_t)mBox.NbElem(),"Edge and CC in MinimumSpanningForest");
}

void cBGG_Graph::BenchAlgos()
{
     for (int aKTime=0 ; aKTime<2*mBox.NbElem()  ; aKTime++)
     {
	 tVertex * aV0 =  VertOfPt(mBox.GeneratePointInside());
	 tVertex * aV1 =  VertOfPt(mBox.GeneratePointInside());

	 Bench_ConnectedComponent(aV0,aV1);

         Bench_ShortestPath(aV0,aV1,0);
         Bench_ShortestPath(aV0,aV1,1);
         Bench_ShortestPath(aV0,aV1,2);

         Bench_MinSpanTree(aV0);
      }
}


void BenchValuatedGraph(cParamExeBench & aParam)
{
    if (! aParam.NewBench("ValuatedGraph")) return;
    //StdOut() << "BenchValuatedGraphBenchValuatedGraph\n";

    cBGG_Graph   aGr(cPt2di(8,13));
    aGr.Bench_FoncElem();
    aGr.BenchAlgos();

    aParam.EndBench();
}


/************************************************************/
/*                                                          */
/*                                                          */
/*                                                          */
/************************************************************/

template <class TGraph>   class cAlgoEnumCycle
{
     public :
          typedef  cAlgo_ParamVG<TGraph>          tParamA;
          typedef  cAlgo_SubGr<TGraph>            tSubGr;
          typedef typename TGraph::tVertex        tVertex;
          typedef typename TGraph::tEdge          tEdge;
          typedef std::vector<tVertex*>           tVectVertex;

	  cAlgoEnumCycle(TGraph & ,const tSubGr &);
	  ~cAlgoEnumCycle();

          void DoExplorate(size_t);

     private :
          void ExplorateOneEdge(tEdge & anE,size_t);
          void RecursiveExplorateOneEdge(tEdge & anE,size_t);

          bool  OkEdge(const tEdge & anE) const;
          
	  TGraph &          mGraph;
	  const tSubGr &    mSubGr;
	  size_t            mBitEgdeExplored; 
};

template <class TGraph>  
     cAlgoEnumCycle<TGraph>::cAlgoEnumCycle
     (
           TGraph &          aGraph,
           const tSubGr &    aSubGr
     ) :
         mGraph           (aGraph),
         mSubGr           (aSubGr),
         mBitEgdeExplored (mGraph.Edge_AllocBitTemp())
{
}

template <class TGraph>  cAlgoEnumCycle<TGraph>::~cAlgoEnumCycle()
{
   mGraph.Edge_FreeBitTemp(mBitEgdeExplored);
}

template <class TGraph> bool  cAlgoEnumCycle<TGraph>::OkEdge(const tEdge & anE) const
{
    return mSubGr.InsideV1AndEdgeAndSucc(anE) && anE.SymBitTo1(mBitEgdeExplored);
}


template <class TGraph> void  cAlgoEnumCycle<TGraph>::ExplorateOneEdge(tEdge & anE,size_t aSz)
{
     cAlgoSP<TGraph>::MakeShortestPathGen
     (
           mGraph,
           true,
           {&anE.VertexInit()},
           cAlgo_ParamVG<TGraph>(),
           cAlgo_SubGrCostOver<TGraph>(aSz)
     );

     anE.SymSetBit1(mBitEgdeExplored);
}


template <class TGraph> void  cAlgoEnumCycle<TGraph>::DoExplorate(size_t aSz)
{
    for (auto & aVPTr : mGraph.AllVertices())
    {
          if (mSubGr.InsideVertex(*aVPTr))
          {
               for (auto & anEdgePtr :  aVPTr->EdgesSucc())
               {
                   if (OkEdge(*anEdgePtr))
                   {
                         ExplorateOneEdge(*anEdgePtr,aSz);
                   }
               }
          }
    }

    for (auto & aVPTr : mGraph.AllVertices())
    {
        for (auto & anEdgePtr :  aVPTr->EdgesSucc())
           anEdgePtr->SymSetBit0(mBitEgdeExplored);
    }
}


typedef cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym> tBGG;
template class  cAlgoEnumCycle<tBGG>;


};

