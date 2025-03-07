#include "MMVII_util_tpl.h"
#include "MMVII_Bench.h"
#if (1)
#include "cMMVII_Appli.h"
#include "MMVII_Images.h"

#include "MMVII_Tpl_GraphStruct.h"
#include "MMVII_Tpl_GraphAlgo_SPCC.h"
#include "MMVII_2Include_Serial_Tpl.h"  // for hash
#include "MMVII_Tpl_GraphAlgo_EnumCycles.h"


namespace MMVII
{

template <class TGraph>   
     void cVG_Tree<TGraph>::Split(t2Tree& a2T ,tEdge *aSplitingEdge)
{
    aSplitingEdge = aSplitingEdge->EdgeInitOr();

    // computed a vector of all edges except aSplitingEdge
    std::vector<tEdge*> aVEdgesMaintained;
    for (const auto & anE : mEdges)
        if (anE->EdgeInitOr() != aSplitingEdge)
           aVEdgesMaintained.push_back(anE);
    MMVII_INTERNAL_ASSERT_tiny(mEdges.size()==(aVEdgesMaintained.size()+1)," Split: edge not in tree");

    cSubGraphOfEdges_Only<TGraph> aSG_EdM(*mGraph,aVEdgesMaintained); // Sub-graph ~ aVEdgesMaintained

    // recover vertices  associated to edges, use to accelerate Multiple_ConnectedComponent
    std::vector<tVertex*>  allV = Vertices();
    std::list<std::vector<tVertex *>>  aListCC = cAlgoCC<TGraph>::Multiple_ConnectedComponent(*mGraph,allV,aSG_EdM);

    MMVII_INTERNAL_ASSERT_tiny(aListCC.size()==2," Split: Bad CC");

    // parse the 2 CC 
    int aKTree=0;
    for (const auto & aVerticesCC : aListCC)
    {
        std::vector<tEdge*>  aEdgesCC;
        cVG_OpBool<TGraph>::EdgesInterVertices(aEdgesCC,mEdges,aVerticesCC);
        MMVII_INTERNAL_ASSERT_tiny(aVerticesCC.size()==(aEdgesCC.size()+1)," Split: edge not in tree");
        if (aEdgesCC.empty())
        {
           tVertex * aV0 =  aVerticesCC.at(0);
           a2T.at(aKTree) = tTree(aV0->Graph(),aV0);
        }
        else
        {
           a2T.at(aKTree) = tTree(aEdgesCC);
        }
        aKTree++;
    }
/*
    StdOut() << "\n";
*/

    // std::vector<tVertex*>  aCC1 = tAlgoCC::ConnectedComponent(*mGraph,anETree->VertexInit(),aSubV):
//StdOut() << "SplitSplit " << aListCC.size() << " ES=" << mEdges.size()  <<  " " << aCC1.size() << " " << aCC2.size() << "\n";


    // return   std::pair<cVG_Tree<TGraph>,cVG_Tree<TGraph>> (*this,*this);
}
/*
*/



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
                mIsOk     (true),
                mCptPath0 (0),
                mCptPath1 (0),
                mHCode0   (0),
                mHCode1   (0)
           {}
           tREAL8 mDist;      // euclidean distance, not really use 4 now
           tREAL8 mRanCost;   // use to modify the weighting  
           bool   mIsOk;      // use to creat some specific subgraph
           int    mCptPath0;  // count the number of path
           int    mCptPath1;  // use to "back up" path counter for check
           size_t mHCode0;    // h-code of pathes going throuh it
           size_t mHCode1;    // h-code of pathes going throuh it
     private :
};

/**  Class for "grid" graph */

class  cBGG_Graph : public cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym>
{
     public :
          // ------------------- typedef part -------------------------------------------------------
          typedef cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym> tGraph;
          typedef typename tGraph::tVertex                              tVertex;
          typedef  tVertex*                                             tPtrV;
          typedef  cAlgoSP<cBGG_Graph>                                  tAlgoSP;
          typedef  typename cVG_Tree<tGraph>::tSetEdges             tSetTreeEdges;
          typedef  cAlgoCC<cBGG_Graph>                                  tAlgoCC;
          typedef  tAlgoSP::tForest                                     tForest;
          typedef  cAlgo_ParamVG<cBGG_Graph>                            tParamA;
          typedef  cAlgo_ParamVG<cBGG_Graph>                             tSubGr;
          typedef std::pair<size_t,size_t> tPSzH; // used in pair count/hcode


          cBGG_Graph (cPt2di aSzGrid); /// constructor , take the size of the grid

          /// Vertex of a given pixel, return reference to a pointer so that it can be used also in init
          tPtrV & VertOfPt(const cPt2di & aPix) 
          {
               return mGridVertices.at(aPix.y()).at(aPix.x());
          }
          tEdge&  EdgeOfPts(const cPt2di & aP1,const cPt2di & aP2)
          {
                tVertex & aV1 = *VertOfPt(aP1);
                tVertex & aV2 = *VertOfPt(aP2);
                return *aV1.EdgeOfSucc(aV2)->EdgeInitOr();
          }

          tPtrV & VertOfAbsCurv(int aAbsCurv)
	  {
		  return FindIf(this->AllVertices(),[aAbsCurv](const auto & aV) {return aV->Attr().mAbsCurv==aAbsCurv;});
	  }

	  int NumQuad(const cPt2di & aPt,int aNb)
	  {
              cPt2dr aSzR = ToR(mSzGrid) / tREAL8(aNb-0.5);
              return   round_ni(aPt.x()/aSzR.x())  + round_ni((aPt.y()/aSzR.y())) * aNb;
	  }
          

          /// test creation , acces to neighbours, attributes
          void Bench_FoncElem();  
          ///  Test the algorithms
          void BenchAlgos();

          /// use to compute a unique id up to a circular permutation
          size_t PathHashCode(const std::vector<tEdge *>&) const;
     private :
          /// Test minimum spaning tree & forest
          void Bench_Pruning();

	  /// global test algorithms on cycle enumeration
          void Bench_EnumCycle(bool Is4Cnx);
	  /// test algorithms on cycle enumeration going through  a single edge
          tPSzH EnumCycle_1Edge(const cPt2di & aP1,const cPt2di & aP2,size_t aSzCycle,int aNbTh, tSubGr&);

          /// test connected component, for a single pixel or a region
          void Bench_ConnectedComponent(tVertex * aV0,tVertex * aV1);
          /// Test computation of shortest path between 2 vertices, mode-> correspond to different distance/conexion
          void Bench_ShortestPath(tVertex * aV0,tVertex * aV1,int aMode);
          
          /// Test minimum spaning tree & forest
          void Bench_MinSpanTree(tVertex * aSeed);
          /// Check that computed a tree on a subgraph is a minimum spaning
          void Check_MinSpanTree(const tSetTreeEdges& aSet,const tParamA& );

          cPt2di                             mSzGrid;       ///< sz of the grid
          cPt2di                             mMid;          ///< sz of the grid
          cRect2                             mBox;          ///< Box associated to the  
          std::vector<std::vector<tPtrV>>    mGridVertices; ///< such the mGridVertices[y][x] -> tVertex *
          void AddEdge(const cPt2di&aP0,const cPt2di & aP1); ///< Add and edge bewteen vertices corresponding to 2 Pix
          tAlgoSP  mAlgoSP;

         class cNeigh_4_Connex : public  cAlgo_ParamVG<cBGG_Graph>
         {
            public :
              // this formula validate the edge iff  |x1-x2|+|y1-y2| <= 1
                   bool   InsideEdge(const    tEdge & anE) const override 
		   {   
                     return Norm1(anE.VertexInit().Attr().mPt-anE.Succ().Attr().mPt)<=1; 
                   }
         };
};
typedef cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym> tBGG;


cBGG_Graph::cBGG_Graph (cPt2di aSzGrid) :
      cVG_Graph<cBGG_AttrVert,cBGG_AttrEdgOr,cBGG_AttrEdgSym>(),  
      mSzGrid        (aSzGrid),
      mMid           (mSzGrid/2),
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
          VertOfPt(aPix) = NewVertex(cBGG_AttrVert(aPix,anAbsicsa));
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

    /* ========================================================== */
    /* ========================================================== */
    /* ===                 Elementary functions               === */
    /* ========================================================== */
    /* ========================================================== */

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

    /* ========================================================== */
    /* ========================================================== */
    /* ===                Shortest Pathes                     === */
    /* ========================================================== */
    /* ========================================================== */


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

        - [3]  we use the same distance then [2] but implement it by a lambda sub-graph 

    */

    class cWeighSqDeltaAbs : public  cAlgo_ParamVG<cBGG_Graph>
    {
       public :
             // weighting is square difference of curvilinear absisca
             tREAL8 WeightEdge(const    tEdge & anE) const override
             { 
                     return Square(anE.AttrOriented().mDeltaAC);
             }
    };


    auto aLamb_SQDA =  [](const auto & aE){return Square(aE.VertexInit().Attr().mAbsCurv-aE.Succ().Attr().mAbsCurv);};
    auto aParam3  = Tpl_InsideAndWSubGr (this,this->V_True(), this->E_True(), aLamb_SQDA);

    cAlgo_ParamVG<cBGG_Graph> aParam0;
    cNeigh_4_Connex           aParam1;
    cWeighSqDeltaAbs          aParam2;

    std::vector<cAlgo_ParamVG<cBGG_Graph>*>  aVParam{&aParam0,&aParam1,&aParam2,&aParam3};

    cAlgo_ParamVG<cBGG_Graph> *aParam = aVParam.at(aMode);

    tVertex *  aTarget =  mAlgoSP.ShortestPath_A2B(*this,*aV0,*aV1,*aParam);

    //  compute the theoreticall distance according to previous discussion
    int aDTh = NormInf(aV0->Attr().mPt -aV1->Attr().mPt);
    if (aMode==1) aDTh = Norm1(aV0->Attr().mPt -aV1->Attr().mPt);
    if ((aMode==2)|| (aMode==3)) aDTh = std::abs(aV0->Attr().mAbsCurv -aV1->Attr().mAbsCurv);


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


    /* ========================================================== */
    /* ========================================================== */
    /* ===           connected components                     === */
    /* ========================================================== */
    /* ========================================================== */


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

    /* -----------[3] check "Lamda subgraph" (with   All_ConnectedComponent ) --------------------------------*/

       // sub graph of vertices equal, %3, to (1 or 2) 
    auto aX_12_3 =  [](const auto & aV){return (aV.Attr().mPt.x()%3)!=0;};
    auto aSubGr_X12_3 = Tpl_InsideAndWSubGr (this, aX_12_3, this->E_True(), this->E_W1() );

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,aSubGr_X12_3);
    int aNbThCC_12_3 = (mSzGrid.x()+1)/3; // theoritcall number of connected component
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == aNbThCC_12_3,"NbCC with Lambda / Hori");

       // now same sub-gr but with additionnal constraint of only considering horizontal edges 
    auto aEdge_Hori =  [](const auto & anE){return anE.VertexInit().Attr().mPt.y() == anE.Succ().Attr().mPt.y();};
    auto aHor_SubGr_X12_3 = Tpl_InsideAndWSubGr (this, aX_12_3, aEdge_Hori, this->E_W1() );

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,aHor_SubGr_X12_3);
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == aNbThCC_12_3 *  mSzGrid.y(),"NbCC with Lambda / Hori");


    /* -----------[4] check sub-graph of extensive set of edges / vertices--------------------------------*/

    std::vector<tVertex*> aV_12_3;  // point with X%3 = 1,2
    std::vector<tVertex*> aV_12_3_Y; // idem + Y%5 != Sz_Grid.y/2

    std::vector<tEdge*> aV_E4;     // edge that link inside each quarter 
    std::vector<tEdge*> aV_E4Sym;  // idem but only symetric edges
    int aNbSector = 3;
    for (const auto & aV : this->AllVertices())
    {
        cPt2di aP1 = aV->Attr().mPt;

        if (aP1.x()%3)
	{
	   aV_12_3.push_back(aV);
	   if (aP1.y() != mSzGrid.y()/2)
              aV_12_3_Y.push_back(aV);
	}
	for (const auto & anE : aV->EdgesSucc())
	{
            cPt2di aP2 = anE->Succ().Attr().mPt;
	    if (NumQuad(aP1,aNbSector) == NumQuad(aP2,aNbSector) )
	    {
                aV_E4.push_back(anE);
	        if (anE->IsDirInit())
                   aV_E4Sym.push_back(anE);
	    }
	}
    }

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,cSubGraphOfVertices<cBGG_Graph>(*this,aV_12_3));
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == aNbThCC_12_3,"NbCC with subgr-vertices ");

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,cSubGraphOfVertices<cBGG_Graph>(*this,aV_12_3_Y));
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == aNbThCC_12_3*2,"NbCC with subgr-vertices ");

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,cSubGraphOfEdges<cBGG_Graph>(*this,aV_E4));
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == Square(aNbSector),"NbCC with subgr-edges ");

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,cSubGraphOfEdges<cBGG_Graph>(*this,aV_E4Sym));
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == Square(aNbSector),"NbCC with subgr-edges ");

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,cSubGraphOfEdges_Only<cBGG_Graph>(*this,aV_E4));
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == Square(aNbSector),"NbCC with subgr-edges ");

    aVecCC = tAlgoCC::All_ConnectedComponent(*this,cSubGraphOfEdges_Only<cBGG_Graph>(*this,aV_E4Sym));
    MMVII_INTERNAL_ASSERT_bench((int)aVecCC.size() == Square(aNbSector),"NbCC with subgr-edges ");

    {
        cSubGraphOfVertices<cBGG_Graph>     aSG1(*this,aV_12_3);
        cSubGraphOfNotVertices<cBGG_Graph>  aSG2(*this,aV_12_3);
        for (const auto & aV : AllVertices())
            MMVII_INTERNAL_ASSERT_bench(aSG1.InsideVertex(*aV)==!aSG2.InsideVertex(*aV),"Sub Gr neg");

        std::vector<tEdge *> aVF;
        cVG_OpBool<tGraph>::EdgesMinusVertices(aVF,AllEdges_DirInit(),aV_12_3);

        for (const auto &anE : aVF)
        {
               MMVII_INTERNAL_ASSERT_bench(anE->VertexInit().Attr().mPt.x()%3==0,"EdgesMinusVertices");
               MMVII_INTERNAL_ASSERT_bench(anE->Succ().Attr().mPt.x()%3==0,"EdgesMinusVertices");
        }
        int aNbTh = (mSzGrid.y()-1) * ((mSzGrid.x()+2) /3 );
        MMVII_INTERNAL_ASSERT_bench((int)aVF.size()==aNbTh,"EdgesMinusVertices::Nb");
     }

}

    /* ========================================================== */
    /* ========================================================== */
    /* ===            spaning trees                           === */
    /* ========================================================== */
    /* ========================================================== */


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


void cBGG_Graph::Check_MinSpanTree(const tSetTreeEdges& aSetPair,const tParamA& aParam )
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
     for (const auto & anE : aSetPair )
         anE->AttrSym().mIsOk = true;


     //  ------------ [1] parse all edge and supress it alternatively ------------------
     for (const auto & anETree : aSetPair )
     {
         // [1.0]  supress the edge in subgraph
         anETree->AttrSym().mIsOk = false;

         // [1.1] extract the 2 connected components
         std::vector<tVertex*>  aCC1 = tAlgoCC::ConnectedComponent(*this,anETree->VertexInit(),cASymOk());
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
             tREAL8 WeightEdge(const    tEdge & anE) const override
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
    tSetTreeEdges aSetPair = mAlgoSP.MinimumSpanninTree(*this,*aSeed,cWRanCost()).Edges();

    {
        cVG_Tree<tGraph> aTree(aSetPair);
std::array<cVG_Tree<tGraph>,2> aPairT;
aTree.Split(aPairT,aSetPair.at(0));
        auto aVV = tEdge::VerticesOfEdges(aSetPair);
        MMVII_INTERNAL_ASSERT_bench(aSetPair.size()+1==aVV.size(),"VerticesOfEdges");
    }

    
    MMVII_INTERNAL_ASSERT_bench((int)aSetPair.size()==(mBox.NbElem()-1),"Size in All_ConnectedComponent");
    //  [1.0]  check that it is effectively the minimal spaning tree
    Check_MinSpanTree(aSetPair,cAlgo_ParamVG<cBGG_Graph>());

    // [2]  compute a minimal spaning forest
    cRC_Thr  aRC(1.0 - 2*std::pow(RandUnif_0_1(),2.0));  // define a sub-graph
    tForest aForest(*this);
    mAlgoSP.MinimumSpanningForest(aForest,*this,this->AllVertices(), aRC);  // extract the forest

    size_t aNbEdge = 0;
    // check that each tree of the forest complies with Check_MinSpanTree
    for (const auto & aTree : aForest.VTrees())
    {
        Check_MinSpanTree(aTree.Edges(),aRC);
        aNbEdge += aTree.Edges().size();
    }
    // check on number of edge, +- Eulers relation for a graph w/o cycle
    MMVII_INTERNAL_ASSERT_bench(aForest.VTrees().size()+aNbEdge== (size_t)mBox.NbElem(),"Edge and CC in MinimumSpanningForest");
}

    /* ========================================================== */
    /* ========================================================== */
    /* ===               cycles enumeration                   === */
    /* ========================================================== */
    /* ========================================================== */


/*  Make the test for correctness of cycle enumeration.  Also it's difficult to make an extensive test like
we do with previous one, we do the following test :

     - all computed path are effectively closed loops
     - all computed path are different
     - for a select number of configuration (where we can count them "by hand"), we get the expected answer
     - when computing global paths, we get the same results that individual pathes (count and hash code)
*/


/*     A hash code is computed on cycles, it is used to checks that set of cycles are identic.
      We compute a hash code that compare loop up to a shift and the orientation, ie :

            (A B C D E) = (E A B C D) = (E D B  C A)

     Because when comparing loops computed globally or locally we dont fix the origin nor the orientation.
*/

size_t cBGG_Graph::PathHashCode(const std::vector<tEdge *>& aPath) const
{
    // [1]  Transformate vecto of edge in vector of index of points, and extract the min index
    std::vector<int>  aVInd;
    cWhichMin<int,int> aWMin;
    {
        for (auto & anE : aPath)
        {
            aVInd.push_back(mBox.IndexeLinear(anE->Succ().Attr().mPt));  // compute vector of index
            aWMin.Add(aVInd.size()-1,aVInd.back()); // update to know index min
        }
    }
    size_t aK0 = aWMin.IndexExtre();  
    // to fix the orientation , select a sens of orientation
    int aDelta = (ValCirc(aVInd,aK0+1)>ValCirc(aVInd,aK0-1)) ? 1 : -1;

    // [2]  compute a hash-code 
    size_t aHasKey=0;
    for (size_t aK=0 ; aK<aVInd.size() ; aK++)
    {
        int aV = ValCirc(aVInd,aK0 + aK*aDelta);  // begin with K0, follow the direction select
        hash_combine(aHasKey,aV);
    }
    return aHasKey;
}

    /* ========================================================== */
    /* ========================================================== */
    /* ===               Pruning                              === */
    /* ========================================================== */
    /* ========================================================== */

void cBGG_Graph::Bench_Pruning()
{
    auto aLamb_Dabs1 =  [](const auto & aE) -> bool {return std::abs(aE.VertexInit().Attr().mAbsCurv-aE.Succ().Attr().mAbsCurv) == 1;}  ;
    auto aSubGr_DA1  = Tpl_InsideAndWSubGr (this,this->V_True(),  aLamb_Dabs1,this->E_W1());

    {
        cAlgoPruningExtre<cBGG_Graph>  aAlgoP(*this,aSubGr_DA1,cAlgo_SubGrNone<cBGG_Graph>());
        // StdOut() << "NBE " <<  aAlgoP.Extrem().size() << "\n";
        MMVII_INTERNAL_ASSERT_bench(aAlgoP.Extrem().size()== NbVertex() ,"Nb vertices of pruning, all graph");
    }

    {
        std::vector<std::vector<int>> aVVA{{1,10},{44},{88,22,45}};

	for (const auto & aVA : aVVA)
	{
            int aMinA = 1000;
            int aMaxA = 0;
            std::vector<tVertex*>  aVV;
	    for (const auto anA : aVA)
	    {
                UpdateMin(aMinA,anA);
                UpdateMax(aMaxA,anA);
		aVV.push_back(VertOfAbsCurv(anA));
	    }
            cAlgoPruningExtre<cBGG_Graph>  aAlgoP(*this,aSubGr_DA1,cSubGraphOfVertices<cBGG_Graph>(*this,aVV));
            int aNbNotSupr = aVA.empty() ? 0 : (aMaxA-aMinA+1) ;
            int aCheck = (int)  aAlgoP.Extrem().size() + aNbNotSupr - (int) NbVertex() ;
            MMVII_INTERNAL_ASSERT_bench(aCheck==0 ,"Nb vertices of pruning, S-graph");
        }
    }

    {
        auto   Comb = [] (const tEdge& anE) 
                         {
                             cPt2di aP1 = anE.VertexInit().Attr().mPt;
                             cPt2di aP2 = anE.Succ().Attr().mPt;
                             if (aP1.x()== aP2.x()) return true;
                             if ((aP1.y()>=1)&&(aP1.y()<=2) && (aP2.y()>=1)&&(aP2.y()<=2) ) return true;
                             return false;
                         } ;

        auto aSubGr_Comb  = Tpl_InsideAndWSubGr (this,this->V_True(),  Comb,this->E_W1());

        cAlgoPruningExtre<cBGG_Graph>  aAlgoP0(*this,aSubGr_Comb,cAlgo_SubGrNone<cBGG_Graph>());
        size_t aNbTh0 = mSzGrid.x() * (mSzGrid.y()-2);
        MMVII_INTERNAL_ASSERT_bench(aNbTh0== aAlgoP0.Extrem().size() ,"Nb vertices of pruning, Comb-graph");


        cPt2di aP1(1,5); cPt2di aP2(5,9);
        std::vector<tVertex*>  aVV {VertOfPt(aP1),VertOfPt(aP2)};
        cAlgoPruningExtre<cBGG_Graph>  aAlgoP1(*this,aSubGr_Comb,cSubGraphOfVertices<cBGG_Graph>(*this,aVV));
        int  aNbTh1 = mSzGrid.x() * (mSzGrid.y()-2)  -(aP1.y()-2) - (aP2.y()-2);
        MMVII_INTERNAL_ASSERT_bench(aNbTh1==(int)aAlgoP1.Extrem().size()," Nb vertices of pruning, Comb Graph");
    }

}


          /* ----------------------------------------- */
          /*                                           */
          /*            cCheckCycle                    */
          /*                                           */
          /* ----------------------------------------- */

/**  Class to check the cycle enumeration.  The  mMode is used to have 3 variation on use :

          - in mode 0, we modify   mCptPath0 and mHCode0
          - in mode 1, we modify   mCptPath1 and mHCode1
          - in mode 2, we reset    mCptPath1 and mHCode1

    This is necessary because when we compare the global computation  (ExplorateAllCycles()) and the
  local (ExplorateCyclesOneEdge()), we must save the computation at two different place. Also after
  each local computation we must re start the computation at each step from a "virgin state". So

      - mode 0 is used for global computation
      - while local computation do mode 1 then mode 2 
*/

class cCheckCycle : public cActionOnCycle<cBGG_Graph>
{
      public :
          typedef typename tBGG::tVertex        tVertex;
          typedef cAlgoEnumCycle<cBGG_Graph>          tAlgoEnum;
	  cCheckCycle(cBGG_Graph &,int aMode);

          void OnCycle(const tAlgoEnum&)  override;

	  cBGG_Graph &       mBGG;  ///< the graph it is working on²
          std::set<size_t>   mSetH; ///< set of h-code computed 
          int                mMode; ///< mode of action
};

cCheckCycle::cCheckCycle(cBGG_Graph & aBGG,int aMode) :
   mBGG  (aBGG),
   mMode (aMode)
{
}


void cCheckCycle::OnCycle(const tAlgoEnum& anAlgo)
{
    const auto & aPath = anAlgo.CurPath();      // extract the current path
    size_t aHasKey = mBGG.PathHashCode(aPath);  // comput a hash code

   // check it is a loop (i.e begin = end)
    MMVII_INTERNAL_ASSERT_bench(&aPath.front()->VertexInit()==&aPath.back()->Succ(),"OnCycle begin!=end");
    // check it is a path i.e. all consecutive edges are connected
    for (size_t aK=1 ; aK<aPath.size() ; aK++)
    {
        MMVII_INTERNAL_ASSERT_bench(&aPath.at(aK-1)->Succ()==&aPath.at(aK)->VertexInit(),"OnCycle not a path");
    }

    size_t aBitMark = mBGG.AllocBitTemp();  // bit allocated to check unicity of result
    for (auto & anE : aPath)
    {
        // check that marking with aBitMark is done once
        MMVII_INTERNAL_ASSERT_bench(!anE->SymBitTo1(aBitMark),"OnCycle mulltiple edge");
        anE->SymSetBit1(aBitMark);
    
        switch (mMode)  // execute action corresponding to the mode
        {
           case 0 :  // used in global mode, udpdate "mCptPath0/mHCode0"
              anE->AttrSym().mCptPath0++;
              anE->AttrSym().mHCode0 ^=  aHasKey;
           break;
           case 1 : // used in local mode, udpdate "mCptPath1/mHCode1"
              anE->AttrSym().mCptPath1++;
              anE->AttrSym().mHCode1 ^=  aHasKey;
           break;
           case 2 :  // used in local mode, reset "mCptPath1/mHCode1"
              anE->AttrSym().mCptPath1=0;
              anE->AttrSym().mHCode1 = 0;
           break;
           default :
               MMVII_INTERNAL_ASSERT_bench(false,"Bas mode in cCheckCycle::OnCycle");
        }
    }
    // unmark the aBitMark  and free it
    for (auto & anE : aPath)
        anE->SymSetBit0(aBitMark);
    mBGG.FreeBitTemp(aBitMark);

    //  check that each loop it visited only once
    MMVII_INTERNAL_ASSERT_bench(!BoolFind(mSetH,aHasKey),"Path present multiple time");
    mSetH.insert(aHasKey);
}

          /* ----------------------------------------- */
          /*                                           */
          /*            cBGG_Graph                     */
          /*                                           */
          /* ----------------------------------------- */

cBGG_Graph::tPSzH
    cBGG_Graph::EnumCycle_1Edge
    (
         const cPt2di & aP1,   // first point of edge 
         const cPt2di & aP2,   // second point of edge,
         size_t aMaxSzCycle,   // Maximal size of cycle explored
         int aNbTh,            // number of cycle we expect (-1 if we have no request)
         tSubGr  &aSubGr       // sub-graph for computing the cycle
    )
{
     tPSzH aRes (1234568,9999);  // result, init urbish
     tEdge&  aE = EdgeOfPts(aP1,aP2);  // extract the edge of graph

     // parse 2 mode :  1=>compute values  , 2=> clean values
     for (int aMode =1 ; aMode<=2 ; aMode++)
     {
         cCheckCycle  aCheckCycle(*this,aMode);  // object for call back
         cAlgoEnumCycle<cBGG_Graph>  aAlgoEnum(*this,aCheckCycle,aSubGr,aMaxSzCycle);  // class for enumerating cycles
         aAlgoEnum.ExplorateCyclesOneEdge(aE); // explorate the cycles going through an edge

         if (aMode==1)  
            aRes = tPSzH(aE.AttrSym().mCptPath1,aE.AttrSym().mHCode1);  // mode 1, result computed, store it
     }

     // is the expected number was specified, check tha value equals was what expected
     if (aNbTh!=-1)
     {
         MMVII_INTERNAL_ASSERT_bench
         (
              aNbTh==(int)aRes.first,
              "EnumCycle_1Edge : number unexpected Exp="+ToStr(aNbTh)  + ", Got=" +ToStr(aRes.first)
         );
     }

     return aRes;
}

void cBGG_Graph::Bench_EnumCycle(bool Is4Cnx)
{
    cNeigh_4_Connex aSubGr4;  // sub-graph for all connexion
    tSubGr  aSubGrAll;        // sub-graph for all edges
    tSubGr & aSubGr = Is4Cnx ? (aSubGr4): aSubGrAll;  // used sub-graphe

    // set expected values, for lenght 3 & 4, for edge horizontal & diag
    int aCptHori_3 = Is4Cnx ? 0 : 4; // number of lenght 3 with edge horizontal
    int aCptHori_4 = Is4Cnx ? 2 : 12; // number of lenght 4 with edge horizontal
    int aCptDiag_3 = Is4Cnx ? 0 : 2; // number of lenght 3 with edge diagonal
    int aCptDiag_4 = Is4Cnx ? 0 : 12; // number of lenght 3 with edge diagonal

    // for simple configurations, check with the expected number of cycles, name l(n) the number equal to n and
    // L(n) the number <= to n,  as we have l(0)=l(1)=l(2). We have :
    //    - L(3) = l(3)
    //    - L(4) = l(3) + l(4) 
    EnumCycle_1Edge(cPt2di(3,3),cPt2di(4,3),3, aCptHori_3               ,aSubGr);
    EnumCycle_1Edge(cPt2di(3,3),cPt2di(4,3),4, aCptHori_3 + aCptHori_4  ,aSubGr);
    EnumCycle_1Edge(cPt2di(3,3),cPt2di(4,4),3, aCptDiag_3               ,aSubGr);
    EnumCycle_1Edge(cPt2di(3,4),cPt2di(4,3),3, aCptDiag_3               ,aSubGr);
    EnumCycle_1Edge(cPt2di(3,3),cPt2di(4,4),4, aCptDiag_3 + aCptDiag_4  ,aSubGr);

    // For 4 connexion we compute also L(6) 
    if (Is4Cnx)
    {
        size_t aCptHori_6 = 6;
        EnumCycle_1Edge(cPt2di(3,3),cPt2di(4,3),6, aCptHori_4 + aCptHori_6   ,aSubGr);
    }

    size_t  aSizeC =  Is4Cnx ? 6 : 4 ; // number of cycle we do the global computation

    cCheckCycle  aCheckCycle(*this,0);  //  object for call back, mode 0 => modif  mCptPath0 / mHCode0
    cAlgoEnumCycle<cBGG_Graph>  aAlgoEnum(*this,aCheckCycle,aSubGr,aSizeC); // class for enumerating cycles
    aAlgoEnum.ExplorateAllCycles();  // exlporate all the cycles of length <= aSizeC

    // parses all edges to compare local & global computation
    for (const auto & aPtrE : this->AllEdges_DirInit())
    {
        // compute values for a single edge
         cPt2di aP1 = aPtrE->VertexInit().Attr().mPt;
         cPt2di aP2 = aPtrE->Succ().Attr().mPt;
         auto [aSz1,aH1]  =  EnumCycle_1Edge(aP1,aP2,aSizeC,-1,aSubGr);

        // read values from global computation
         size_t aSz0 =  aPtrE->AttrSym().mCptPath0;
         size_t aH0 =  aPtrE->AttrSym().mHCode0;

         // check tey are equal
         MMVII_INTERNAL_ASSERT_bench(aSz0==aSz1,"EnumCycle_1Edge : number unexpected");
         MMVII_INTERNAL_ASSERT_bench(aH0==aH1,"EnumCycle_1Edge : number unexpected");
    }

    // clean valued mCptPath0/mHCode0 for possible re-use
    for (const auto & aPtrE : this->AllEdges_DirInit())
    {
        aPtrE->AttrSym().mCptPath0 =0;
        aPtrE->AttrSym().mHCode0   =0;
    }
}

void cBGG_Graph::BenchAlgos()
{

     for (int aKTime=0 ; aKTime<2*mBox.NbElem()  ; aKTime++)
     {
         Bench_Pruning();
	
	 tVertex * aV0 =  VertOfPt(mBox.GeneratePointInside());
	 tVertex * aV1 =  VertOfPt(mBox.GeneratePointInside());

	 Bench_ConnectedComponent(aV0,aV1);

         Bench_ShortestPath(aV0,aV1,0);
         Bench_ShortestPath(aV0,aV1,1);
         Bench_ShortestPath(aV0,aV1,2);

         Bench_ShortestPath(aV0,aV1,3);

         Bench_MinSpanTree(aV0);
      }

      for (int aKTime=0 ; aKTime<20  ; aKTime++)
      {
         Bench_EnumCycle(true);
         Bench_EnumCycle(false);
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


//template class  cAlgoEnumCycle<tBGG>;

};
#else
namespace MMVII
{
void BenchValuatedGraph(cParamExeBench & aParam)
{
    MMVII_INTERNAL_ASSERT_bench(false,"NO BenchValuatedGraph");
}
};
#endif

