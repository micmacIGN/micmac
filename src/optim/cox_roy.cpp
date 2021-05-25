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

//    INITIAL COPRYRIGHT OF S. Roy/NEC

/***
Sofware: stereomf
Author : Sebastien Roy (sebastien@research.nj.nec.com)

               Copyright (c) 1999, NEC Research Institute Inc.
                            All Rights Reserved.

Permission to use, copy, modify, and distribute this software and its
associated documentation for non-commercial purposes is hereby granted,
provided that the above copyright notice appears in all copies, derivative
works or modified versions of the software and any portions thereof, and
that both the copyright notice and this permission notice appear in the
documentation.  NEC Research Institute Inc. shall be given a copy of any
such derivative work or modified version of the software and NEC Research
Institute Inc. and its affiliated companies (collectively referred to as
NECI) shall be granted permission to use, copy, modify and distribute the
software for internal use and research.  The name of NEC Research Institute
Inc. and its affiliated companies shall not be used in advertising or
publicity related to the distribution of the software, without the prior
written consent of NECI.  All copies, derivative works or modified versions
of the software shall be exported or reexported in accordance with
applicable laws and regulations relating to export control.  This software
is experimental.  NECI does not make any representations regarding the
suitability of this software for any purpose and NECI will not support the
software.
THE SOFTWARE IS PROVIDED AS IS.  NECI DOES NOT MAKE ANY
WARRANTIES EITHER EXPRESS OR IMPLIED WITH REGARD TO THE SOFTWARE.  NECI
ALSO DISCLAIMS ANY WARRANTY THAT THE SOFTWARE IS FREE OF INFRINGEMENT OF
ANY INTELLECTUAL PROPERTY RIGHTS OF OTHERS.  NO OTHER LICENSE EXPRESS OR
IMPLIED IS HEREBY GRANTED. NECI SHALL NOT BE LIABLE FOR ANY DAMAGES,
INCLUDING GENERAL, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, ARISING
OUT OF THE USE OR INABILITY TO USE THE SOFTWARE.
***/

/**

               Copyright (c) 2002, Insitut Geographique National
                            All Rights Reserved.

Modification : Marc PIERROT DESEILLIGNY
               marc.pierrot-deseilligny@ign.fr
               11/01/02
**/
#include "StdAfx.h"
#include "api/cox_roy.h"






// Exemple minimaliste d'utilisation de l'interface

void  ExempleUtilisationCoxRoy 
      (
          signed short ** aZRes,  // Tableau de resultat
          int aSzX, int aSzY,     // Taille en X et en Y du rectangle
          signed short ** aZmin,  // Borne Inf de la nappe
          signed short ** aZmax,  // Borne Sup de la nappe
          int   (* aPtrFuncCost) (int,int,int), // Fonction indiquant le cout d'un pt x,y,z
          double aCoefLin,double aCoefCst       // Coefficients de regularisation
      )
{
   // [1] On cree l'objet
    cInterfaceCoxRoyAlgo *  aCRAlgo = cInterfaceCoxRoyAlgo::NewOne
                                      (
                                          aSzX,aSzY,
                                          aZmin,aZmax,
                                          false,          // on choisit la  4 Conne-xite
                                          true            // Stocke sur UCHAR
                                      );

    // [2] On fixe les couts des arcs verticaux (couts d'attache aux donnees)
    for (int anX=aCRAlgo->X0(); anX<aCRAlgo->X1() ; anX++)
        for (int anY=aCRAlgo->Y0(); anY<aCRAlgo->Y1() ; anY++)
             for (int aZ = aCRAlgo->ZMin(anX,anY); aZ< aCRAlgo->ZMax(anX,anY) ; aZ++)
             {
                 aCRAlgo->SetCostVert(anX,anY,aZ ,aPtrFuncCost(anX,anY,aZ));
             }

     // [3] On en induit les couts de regularisation

     aCRAlgo->SetStdCostRegul(aCoefLin,aCoefCst,1);
 

     // [4] Au travail

      aCRAlgo->TopMaxFlowStd(aZRes);


     // [5] Liberation de la memoire temporaire


     delete aCRAlgo;
     
}







#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cassert>
#include <deque>
#include <vector>
#include <list>
#include <set>


/**** module implementing a simple FIFO queue ****/

#define NbEdgesMax 10

const int         InverseEdgeTable[NbEdgesMax]     =  { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8};
const signed int xdelta[NbEdgesMax]               =  {-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
const signed int ydelta[NbEdgesMax]               =  { 0, 0,-1, 1, 0, 0, 1,-1,-1, 1};
const signed int zdelta[NbEdgesMax]               =  { 0, 0, 0, 0,-1, 1, 0, 0, 0, 0};
const bool        tabCRIsVertical[NbEdgesMax]      =  { false, false, false, false, true, true, false, false, false, false};
const bool        tabCRIsArcV8[NbEdgesMax]         =  { false, false, false, false, false, false, true, true, true, true};



class cRoyPt
{
     public :
        typedef signed short tCoord;

        inline cRoyPt(tCoord anX,tCoord anY,tCoord aZ) : 
             mX(anX),mY(anY),mZ(aZ) 
        {}

        inline cRoyPt (const cRoyPt & aP,int  aDir)  : // Construction of the Neighboord in dir aDir
           mX (aP.mX + xdelta[aDir]),
           mY (aP.mY + ydelta[aDir]),
           mZ (aP.mZ + zdelta[aDir])
        {
        }


        tCoord mX,mY,mZ;

        inline cRoyPt():mX(0),mY(0),mZ(0){}

     private :
};



class cCRQueue
{
     public :
       inline ~cCRQueue() {QueueReset();}
       inline cCRQueue(int ,int,int)  
       {
           QueueReset();
       }

       inline void QueueReset() {mQ.clear();}

       inline void  QueueAdd(const cRoyPt & E)
       {
            mQ.push_back(E);
         
            // To test influence of queuing on Global perf  (negligeable)
            //   mQ.push_back(E); mQ.pop_back();
            //   mQ.push_front(E); mQ.pop_front();
            //   
         
       }


       inline bool empty() const {return mQ.empty();}

       inline void QueueRemove(cRoyPt & E)
       {
           E=mQ.front();
           mQ.pop_front();
       }

private :
	std::deque<cRoyPt>  mQ;
};


/*****************************************************************************************/
/*****************************************************************************************/
/*****************************************************************************************/
/*****************************************************************************************/
/*****************************************************************************************/

/*********************/
/********   LINK *****/
/*********************/


template <class Type,const int NbEl> 
class cCRHeapList
{
public :

	struct tTab { Type  mTab[NbEl];};
	typedef std::list<tTab>   tContainer;

	inline cCRHeapList() : mNB0 (0) {}

	void push_front(const Type & aV,tContainer & aReserv) 
	{
		mNB0--; 
		if (mNB0 <0)
		{
			if (aReserv.empty())
			{
				tTab aT0; 
				for (int k=0; k<100; k++)
					aReserv.push_front(aT0);
			}
			mNB0 = NbEl-1;
			mCont.splice(mCont.begin(),aReserv,aReserv.begin());
                    // mCont.push_front(tTab());
		}
		mCont.front().mTab[mNB0] = aV;
	}

	inline const Type & front() const {return   mCont.front().mTab[mNB0];}
	void pop_front(tContainer & aReserv)
	{
		mNB0++;
		if (mNB0==NbEl)
		{
			mNB0 = 0;
			aReserv.splice(aReserv.begin(),mCont,mCont.begin());
			// mCont.pop_front();
		}
	}

	inline bool empty() const {return mCont.empty();}
	inline void clear(tContainer & aReserv) 
	{
		aReserv.splice(aReserv.begin(),mCont);
		mNB0=0;
	}
          
private :
	tContainer   mCont;
	int          mNB0;      
};



class cCRHeap
{
    public :

       enum {eNbPack = 10};


       ~cCRHeap(){}
       cCRHeap(int NbKey,int /*Size*/,int /*Inc*/,int /*IncKey*/)
       {
              mQueue.reserve(NbKey);
              mQueue.clear();
              mMaxKey = -1;
       }

//       void ShowLink(){}
       void  Set2NonEmptyKey(int &  Level) const
       {
	     if (Level <0) return;

            if (! mQueue[Level].empty()) 
                return;

            tSetI::const_iterator anIt = mSetIndNV.upper_bound(Level);

            if (anIt == mSetIndNV.end())
              Level = -1;
            else 
              Level = *anIt;
       }



       void LinkQReset()
       {
            mSetIndNV.clear();
            mMaxKey = -1;
            for (tQueue::iterator  itQ= mQueue.begin() ; itQ!= mQueue.end() ; itQ++)
                itQ->clear(mReserv);
       }


       int MaxKey() const {return mMaxKey;}

       int LinkQInsert(const cRoyPt & aRoyPt,int aPrio)
       {
             assert(aPrio>=0);
             while((int)mQueue.size() <= aPrio) 
                   mQueue.push_back(mL0);

             if (mQueue[aPrio].empty())
                mSetIndNV.insert(aPrio);

             mQueue[aPrio].push_front(aRoyPt,mReserv);
             if (aPrio > mMaxKey)
                mMaxKey = aPrio;
             return 0;
       }

       cRoyPt LinkQRemove(int aPrio)
       {
           assert((aPrio>=0) && (aPrio < (int)mQueue.size()));
           tList & aLP = mQueue[aPrio];

           assert(!aLP.empty());
           cRoyPt aRes= aLP.front();

           aLP.pop_front(mReserv);
           if (aLP.empty())
               mSetIndNV.erase(aPrio);
           // if (aPrio == mMaxKey)
	   // if ( (mMaxKey>=0) && (mQueue[mMaxKey].empty())
           {
		   Set2NonEmptyKey(mMaxKey);
		   /*
               while ((mMaxKey>=0) && ( mQueue[mMaxKey].empty()))
                     mMaxKey --;
		     */
           }

           return aRes;
       }

    private :
       
       int  mMaxKey;
       typedef cCRHeapList<cRoyPt,eNbPack>   tList;
       typedef std::vector<tList >           tQueue;
       typedef std::set<int,std::greater<int> >                         tSetI;

       tList::tContainer            mReserv;
       tList                        mL0;
       tQueue                       mQueue;
       tSetI                        mSetIndNV;
};





/*****************************************************************************************/
/*****************************************************************************************/
/*****************************************************************************************/
/*****************************************************************************************/
/*****************************************************************************************/

/*********************/
/********   FLOW *****/
/*********************/



/****** Maximum-flow computation in (x,y,d) graph module ******/
/*** Based on preflow-push relabel method from Andrew Goldberg ***/





/**** most stuff is done with global variables holding the data ****/


class cBaseCRNode
{
public :
	inline cBaseCRNode()  : mCurrentEdge (0) {}

    // Current Edge manipulator
	inline int  CurrentEdge() const {	return mCurrentEdge & 0x0F;	}
	inline void SetCurrentEdge(int aCurentEdge) {	mCurrentEdge = aCurentEdge|(mCurrentEdge & 0xF0);	}

    // Sink Source connexion
	inline void SetSinkConnected() {mCurrentEdge |= 0x80;}
	inline bool SinkConnected() const {return (mCurrentEdge&0x80) != 0;}
	inline void SetSourceConnected() {mCurrentEdge |= 0x40;}
	inline bool SourceConnected() const {return (mCurrentEdge&0X40) != 0;}

    // Excess
	inline void SetExcess(int anExcess) { mExcess = anExcess;}
	inline void AddExcess(int aDeltaExcess) { mExcess += aDeltaExcess;}
	inline int  Excess() const {return mExcess;}

    // Height
	inline void SetHeight(int aHeight)  { mHeight = aHeight;}
	inline int  Height() const {return mHeight;}
	inline bool Over1(const cBaseCRNode & aNode) {return mHeight == aNode.mHeight+1;}
	inline bool Over(const cBaseCRNode & aNode) {return mHeight > aNode.mHeight;}

private :

	int mExcess;	/* excess flow */
	int mHeight;	/* height */
	unsigned char mCurrentEdge;	/* 
                                            bits [0 4[  val 0 to 5 : current edge, 6=nil, 
                                            bits 7 : sink connected
                                            bits 6 : source connected
                                         */

};


template <class tElem,const int NbEdges> class cTplCRNode : public  cBaseCRNode
{
public :
	enum {eNbEdges = NbEdges};
             
	cTplCRNode() :  mFlagEdge (0) {}
    // residual Flow
	inline void SetResidualFlow(int ed,int aRF) 
	{ 
		if (aRF <0) aRF = 0;
		else if (aRF >MaxVal) aRF = MaxVal;
		mRF[ed] = aRF;
	}
	inline bool UnSatured(int ed) const {return mRF[ed] > 0;}
	inline void AddResidualFlow(int ed,int aDeltaRF) 	{		mRF[ed] += aDeltaRF;	}
	inline int  ResidualFlow(int ed) const {return mRF[ed];}

	//   Edge Flags
	inline bool EdgeIsValide(int ed) const {return (mFlagEdge &(1<<ed)) != 0;}
	inline void SetEdgeValide(int ed)   { mFlagEdge |= (1<<ed);}
	inline void SetEdgeNotValide(int ed)   { mFlagEdge &= ~(1<<ed);}
	inline void SetAllEdgeValide() {mFlagEdge = (1<<NbEdges)-1;}

	static const int MaxVal;
private :
	unsigned short         mFlagEdge;    /* bit 0..5 -> 1=edge exist, 0=no edge */
	tElem                  mRF[NbEdges]; /* residual flow (=capacity-flow) */
};

template <>  const int cTplCRNode<unsigned char,6>::MaxVal = 100;
template <>  const int cTplCRNode<unsigned char,10>::MaxVal = 100;
template <>  const int cTplCRNode<unsigned short,6>::MaxVal = 20000;
template <>  const int cTplCRNode<unsigned short,10>::MaxVal = 20000;

template <>  const int cTplCRNode<unsigned int,6>::MaxVal = 1000000;
template <>  const int cTplCRNode<unsigned int,10>::MaxVal = 1000000;


template <class cCRNode> class cTplCoxRoyAlgo : public cInterfaceCoxRoyAlgo
{
     public :


           virtual ~cTplCoxRoyAlgo();
           cTplCoxRoyAlgo
           (
                   int xsz,int ysz,
                   signed short ** aDataZmin,
                   signed short ** aDataZmax
           );

           int TopMaxFlowStd(short **Sol);


           inline int NbEdges() const {return eNbEdges;} // 6 ou 10 suivant la conne-xite
           inline bool Cnx8() const {return NbEdges() == 10;}



           void SetCostVert(int anX,int anY,int aZ,int aCost);
           virtual void SetStdCostRegul(double aCoeff,double aCste,int aVMin);



           
	   inline int ZLength(int anX,int anY)const {return ZMax(anX,anY)-ZMin(anX,anY);}
	   inline int ZMaxGlob() const {return mZMaxGlob;}
	   inline int ZMinGlob() const {return mZMinGlob;}
           void SetCost(int anX,int anY,int aZ,int mNumEdge,int aCost);

     private :
           enum {eNbEdges = cCRNode::eNbEdges};


           void AssertPtIsValide(const cRoyPt&) const;
           void EndMaxFlowStd(int aX0,int aY0,int aX1,int aY1);
           int RecursifMaxFlowStd(int aX0,int aY0,int aX1,int aY1);


          inline bool UnLabelled(const cCRNode & aNode) const {return aNode.Height() == mNoLabel;}


          cCRNode *** AllocCNodes();
          inline cCRNode & NodeOfP(const cRoyPt & aP) {return mCNodes[aP.mY][aP.mX][aP.mZ];}
          inline cCRNode * ColumnOfP(int anX,int anY) {return mCNodes[anY][anX];}






          int mX0Loc,mX1Loc,mY0Loc,mY1Loc;  // Limit of current rectangle

          int          mZMaxGlob;
          int          mZMinGlob;
                       
          int  init_sz();
          int sz;    

          int largef;	/* large flow value for initial flooding */
          cCRQueue mCRQ;
          cCRHeap  mCRH; /** contain list of node with excess flow, sorted by level **/

          cCRNode *mLNodes; /* [xs*ys*ds], access : [(y*xs+x)*ds+d] */
          cCRNode *** mCNodes;

          const int mNoLabel;
          const int mNoLabelBorder;
          const int mSourceHeight;
          const int mSinkHeight; 

          int mSinkFlow; /* total flow to the sink */

          bool ResizeDetected;
          int mDirZPlus;
          int mDirZMoins;

          int mNbDischarge;
          int mNbLift;
          int mNbRelabel;
          bool mNLA;  // No Lift Allowed
          /** the whole graph, without source/sink **/

          int (*capacity)(cRoyPt aP1,cRoyPt aP2);

          void findcut(short **Sol);
          void relabel(void);
          void Discharge(const cRoyPt &);
          bool Lift(const cRoyPt &);
          void Push(const cRoyPt &, int dirU2V);
          void SetFlagEdges(int aX0,int aX1,int aY0,int aY1,bool UpdateSinkSource);
};



/*************** support ***************/




template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::SetFlagEdges
                              (int aX0,int aY0,int aX1,int aY1,bool UpdateSinkSource)
{
	for (int anX =aX0; anX<aX1 ; anX++)
	{
	    for (int anY =aY0; anY<aY1 ; anY++)
	    {
                 cCRNode *  aCol = ColumnOfP(anX,anY);
		 int  aZ0 = ZMin(anX,anY);
		 int  aZ1 = ZMax(anX,anY);
		 for (int aZ = aZ0; aZ<aZ1 ; aZ++)
	             aCol[aZ].SetAllEdgeValide();

                 aCol[aZ0].SetEdgeNotValide(mDirZMoins);
                 aCol[aZ1-1].SetEdgeNotValide(mDirZPlus);

                 if (UpdateSinkSource)
                 {
                     aCol[aZ0].SetSourceConnected();
                     aCol[aZ1-1].SetSinkConnected();
                 }

		 for (int ed=0 ; ed<eNbEdges ; ed++)
                 {
                     if (zdelta[ed]==0)
                     {
                         int vX = anX+ xdelta[ed];
                         int vY = anY+ ydelta[ed];
                         bool isInside = (vX>=aX0)&&(vX<aX1)&&(vY>=aY0)&&(vY<aY1);
                         
                         int zMinUnval = aZ0;
                         int zMaxUnval = aZ0;
// Modif MPD-GM du 26/06/09  suite a Plantage lorsque les colonnes ne sont pas connectees
/*
                         if (isInside)
                         {
                             zMinUnval = ZMin(vX,vY);
                             zMaxUnval = ZMax(vX,vY);
                         }
*/
                         if (isInside)
                         {
                             zMinUnval = ElMin(aZ1,ZMin(vX,vY));
                             zMaxUnval = ElMax(aZ0,ZMax(vX,vY));
                         }
                         for (int aZ = aZ0; aZ<zMinUnval ; aZ++)
                              aCol[aZ].SetEdgeNotValide(ed);
						 {
                         for (int aZ = zMaxUnval; aZ<aZ1 ; aZ++)
                              aCol[aZ].SetEdgeNotValide(ed);
						 }

                         if (UpdateSinkSource && isInside)
                         {
                             for (int aZ = aZ0; aZ<zMinUnval ; aZ++)
                                 aCol[aZ].SetSourceConnected();
							 {
                             for (int aZ = zMaxUnval; aZ<aZ1 ; aZ++)
                                 aCol[aZ].SetSinkConnected();
							 }
                         }
                     }
                 }
	    }
	}
}





template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::Push(const cRoyPt & aPU, int eU2V)
{
        cCRNode  & aNodU = NodeOfP(aPU);


	/** u is overflowing? **/
	if( aNodU.Excess()==0 ) return;

	/** some residual flow? **/
	if( ! aNodU.UnSatured(eU2V) ) return;

	/** height difference is 1? **/
        cRoyPt aPV(aPU,eU2V);
        cCRNode  & aNodV = NodeOfP(aPV);
	// if( aNodU.h!=aNodV.h+1 ) return;
	 if(! aNodU.Over1(aNodV) ) return;

	/** ok, you can push **/

	int k=aNodU.Excess();			    /* get rid of excess */
	if( k>aNodU.ResidualFlow(eU2V) ) k=aNodU.ResidualFlow(eU2V);  /* but not more than capacity */

	aNodU.AddResidualFlow(eU2V,-k);  /* less residual flow to v */
	aNodV.AddResidualFlow(InverseEdgeTable[eU2V],k); /* more residual flow to u */
	aNodU.AddExcess(-k);

	assert( aNodV.Excess() + k >= aNodV.Excess() ); 
	// printf("OVERFLOW! %d + %d = %d\n",aNodV.Excess(),k,aNodV.Excess()+k);


	aNodV.AddExcess(k);

	/** add to heap **/
	/** add only if excess is exactly k, implying that **/
	/** node v was NOT in the queue **/
	if( aNodV.Excess()==k ) {
		if( mCRH.LinkQInsert(aPV,aNodV.Height()) ) ResizeDetected=true;
	}
}



template <class cCRNode> bool cTplCoxRoyAlgo<cCRNode>::Lift(const cRoyPt & aPU)
{
	mNbLift++; /* stats */
	cCRNode & aNodU = NodeOfP(aPU);

	/** is u overflowing? **/
	if( aNodU.Excess()==0 ) return true;

	/** for each outgoing edge with cf>0, check h[u]<=h[v] **/
	int hmin=mSourceHeight;
	for(int ed=0;ed<eNbEdges;ed++) 
	{
		if (aNodU.EdgeIsValide(ed) && ( aNodU.UnSatured(ed)))
                {
		    cCRNode & aNodV = NodeOfP(cRoyPt(aPU,ed));

		    if( aNodU.Over(aNodV) ) 
		    {
                        if (!mNLA)
                        {
                            printf("   -- ?!#!! Bizarre Assertion in Cox-Roy  %d %d %d \n",aPU.mX,aPU.mY,aPU.mZ);
                        }
                        // FirstHere = false;
		        // assert(false);
			/*
			printf("u=(%d,%d,%d) v=(%d,%d,%d) cf=%d with u.h=%d v.h=%d!\n",
			aPU.mX,aPU.mY,aPU.mZ,
			aPV.mX,aPV.mY,aPV.mZ,
			aNodU.rf[ed],aNodU.Height(),aNodV.Height());
			*/

                        mNLA = true;
			return false; /* no lift allowed */
		    }

		    if( aNodV.Height() < hmin ) hmin= aNodV.Height();
                }
	}

	aNodU.SetHeight(hmin+1);
        return true;
}



template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::Discharge(const cRoyPt & aPU)
{
	mNbDischarge++; /* stats */

	cCRNode & aNodU = NodeOfP(aPU);

	while( aNodU.Excess() > 0 ) 
        {
		/* special case : push back to source */
		if( aNodU.SourceConnected() && aNodU.Height()==mSourceHeight+1 ) {
			/* push back into source */
			//printf("*** returned %d to source from (%d,%d,%d) ***\n",mLNodes[u].e,x,y,d);
			aNodU.SetExcess(0);
			break;
		}

		/* special case : push to sink */
		if ( aNodU.SinkConnected() && aNodU.Height()==mSinkHeight+1 ) {
			/* push into sink */
			/**
			printf("*** added %d to sink from (%d,%d,%d) ***\n",mLNodes[u].e,x,y,d);
			**/
			mSinkFlow+=aNodU.Excess();
			aNodU.SetExcess(0);
			break;
		}

		int ed = aNodU.CurrentEdge();
		if( ( aNodU.EdgeIsValide(ed) )
		 && ( aNodU.Over1(NodeOfP(cRoyPt(aPU,ed))))
		 && ( aNodU.UnSatured(ed)) 
		 ) 
		{
			Push(aPU,ed);
			continue;
		}

		aNodU.SetCurrentEdge(ed+1);

		if( aNodU.CurrentEdge()==eNbEdges ) 
                {
			bool OkLift = Lift(aPU);
			aNodU.SetCurrentEdge(0);
			/** special case **/
			/** if node at or above source, remove excess **/
			if( aNodU.Height() > mSourceHeight ) 
                        {
				aNodU.SetExcess(0);
				break;
			}
                        if (!OkLift) return;
		}
	}
}





/** use breadth first from sink to find cut **/
/** relabel at the same time **/
template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::relabel(void)
{
	mNbRelabel++; /* stats */

	// printf("-- relabel --\n");

	mCRQ.QueueReset();

	/** first, flush the heap **/
	mCRH.LinkQReset();

	/*BackToSource=0;*/

	/** all node are unlabelled **/
	for (int anX =mX0Loc; anX<mX1Loc ; ++anX)
	{
		for (int anY =mY0Loc; anY<mY1Loc ; ++anY)
		{
			cCRNode *  aCol = ColumnOfP(anX,anY);
			int  aZ0 = ZMin(anX,anY);
			int  aZ1 = ZMax(anX,anY);
			for (int aZ = aZ0; aZ<aZ1 ; ++aZ)
			{
				aCol[aZ].SetCurrentEdge(0);
				/** start by queueing all nodes adjacent to sink **/
				if ( aCol[aZ].SinkConnected())
				{
					aCol[aZ].SetHeight(mSinkHeight+1);
					mCRQ.QueueAdd(cRoyPt(anX,anY,aZ));
				}
				else
					aCol[aZ].SetHeight(mNoLabel);
			}
		}
	}


	cRoyPt aPU(-1111,1111,6669); // BIDON
	while(!mCRQ.empty() ) 
	{
		mCRQ.QueueRemove(aPU);
		cCRNode & aNodU = NodeOfP(aPU);

		/** add this node to list of node to process **/
		if( aNodU.Excess()>0 ) mCRH.LinkQInsert(aPU,aNodU.Height());

		/** check each edge for residual flow **/
		for(int ed=0;ed<eNbEdges; ++ed) 
			if ( aNodU.EdgeIsValide(ed))
			{
				cRoyPt aPV (aPU,ed);
				cCRNode & aNodV = NodeOfP(aPV);
				if (UnLabelled(aNodV) && (aNodV.UnSatured(InverseEdgeTable[ed])))
				{
					aNodV.SetHeight(aNodU.Height()+1);
					mCRQ.QueueAdd(aPV);
				}
			}
	}
}


/** allocate Sol (if non null) to (xs x ys) solution array **/
template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::findcut(short **Sol)
{

	// printf("-- find cut --\n");

	for(int anY=mY0Glob ; anY<mY1Glob ; anY++)
        {
	   for(int anX=mX0Glob ; anX<mX1Glob ; anX++)
           {
                Sol[anY][anX]= ZMinGlob() -1;

		int  aZ0 = ZMin(anX,anY);
		int  aZ1 = ZMax(anX,anY);
                Sol[anY][anX]=aZ0;  //Bug qui fait que parfois le Z n'est pas trouve
		for (int aZ = aZ0; aZ<aZ1 ; aZ++)
                {
                     cRoyPt aPU(anX,anY,aZ);
                     cCRNode & aNodU = NodeOfP(aPU);

		     if(UnLabelled(aNodU))
                     {

		         for(int ed=0;ed<eNbEdges;ed++) 
                         {
			     if ( aNodU.EdgeIsValide(ed))
                             {
                                 cRoyPt aPV(aPU,ed);
                                 cCRNode & aNodV = NodeOfP(aPV);

			         if(aNodV.Height()!=mNoLabel&&aNodV.Height()!=mNoLabelBorder)
                                 {
			            aNodU.SetHeight(mNoLabelBorder);
			            Sol[anY][anX]=aZ;
                                 }
                             }
		         }
		     }
	        }

           }
        }
}







/*************** main access point ****************/




template <class cCRNode> int cTplCoxRoyAlgo<cCRNode>::init_sz()
{
   int res =0;

   mZMinGlob =  ZMin(mX0Glob,mY0Glob);
   mZMaxGlob =  ZMax(mX0Glob,mY0Glob);
   
   for(int anX=mX0Glob ; anX<mX1Glob ; anX++)
   {
	for(int anY=mY0Glob ; anY<mY1Glob ; anY++)
        {
              int aZ0 = ZMin(anX,anY);
              int aZ1 = ZMax(anX,anY);
              assert(aZ0<aZ1);

              res += aZ1-aZ0;
              if (aZ0 < mZMinGlob) mZMinGlob = aZ0;
              if (aZ1 > mZMaxGlob) mZMaxGlob = aZ1;
        }
   }

   return res;
}

template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::AssertPtIsValide(const cRoyPt& aP) const
{
    assert(IsInside(aP.mX ,aP.mY,aP.mZ));
}

template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::SetCost(int anX,int anY,int aZ,int aNumEdge,int aCost)
{
    cRoyPt p1(anX,anY,aZ);
    AssertPtIsValide(p1);
    NodeOfP(p1).SetResidualFlow(aNumEdge,aCost);
}

template <class cCRNode> void cTplCoxRoyAlgo<cCRNode>::SetCostVert(int anX,int anY,int aZ,int aCost)
{
    cRoyPt p1(anX,anY,aZ);

    AssertPtIsValide(p1);
    cCRNode & aS1 = NodeOfP(p1);

    aS1.SetResidualFlow(mDirZPlus,aCost);
    if (aS1.EdgeIsValide(mDirZPlus))
    {
        cRoyPt p2(p1,mDirZPlus);
        cCRNode & aS2 = NodeOfP(p2);
        aS2.SetResidualFlow(mDirZMoins,aCost);
    }
}

template <class cCRNode> void  cTplCoxRoyAlgo<cCRNode>::SetStdCostRegul(double aCoeff,double aCste,int aVmin)
{
    for (int anX=X0(); anX<X1() ; anX++)
        for (int anY=Y0(); anY<Y1() ; anY++)
             for (int aZ = ZMin(anX,anY); aZ< ZMax(anX,anY) ; aZ++)
             {
                 cRoyPt    aP1 (anX,anY,aZ);
                 cCRNode & aS1 = NodeOfP(aP1);
                 int  aC1 = aS1.ResidualFlow(mDirZPlus);

                 for (int anEdg=0; anEdg<NbEdges() ; anEdg++)
                 {
                     if (aS1.EdgeIsValide(anEdg) && (!tabCRIsVertical[anEdg]))
                     {
                           cRoyPt aP2(aP1,anEdg);
                           cCRNode & aS2 = NodeOfP(aP2);
                           int aC2 = aS2.ResidualFlow(mDirZPlus);

                           double aCost = aCste + aCoeff*(aC1+aC2)/2.0;
                           if (Cnx8())
                               aCost *= tabCRIsArcV8[anEdg] ? 0.2928 : 0.4142 ;
                           int iCost = int(aCost+0.5);
                           if (iCost<aVmin)
                              iCost = aVmin;
                           aS1.SetResidualFlow(anEdg,iCost);
                     }
                 }
             }
}


template <class cCRNode> cCRNode *** cTplCoxRoyAlgo<cCRNode>::AllocCNodes()
{  typedef cCRNode ** cCRNodeHdl;
   typedef cCRNode * cCRNodePtr;

//   cCRNode *** aRes = (new(cCRNode **)) [mY1Glob-mY0Glob] -mY0Glob;
   cCRNodeHdl * aRes = new cCRNodeHdl[mY1Glob-mY0Glob] - mY0Glob;
   cCRNode * aLN = mLNodes;

   for (int anY = mY0Glob; anY<mY1Glob ; anY++)
   {
        aRes[anY] = new cCRNodePtr[mX1Glob-mX0Glob] -mX0Glob;
        for (int anX =mX0Glob; anX<mX1Glob ; anX++)
        {
            aRes[anY][anX] = aLN - ZMin(anX,anY); 
            aLN += ZLength(anX,anY);
        }
   }

   return aRes;
}

template <class cCRNode> cTplCoxRoyAlgo<cCRNode>::~cTplCoxRoyAlgo()
{
   for (int anY = mY0Glob; anY<mY1Glob ; anY++)
   {
        delete [] (mCNodes[anY]+mX0Glob);
   }

   delete [] ( mCNodes+mY0Glob);
   delete [] (mLNodes);
}



template <class cCRNode> cTplCoxRoyAlgo<cCRNode>::cTplCoxRoyAlgo
    (
        int xsz,int ysz,
        signed short ** aZMin, signed short ** aZMax
     )  :
     cInterfaceCoxRoyAlgo(xsz,ysz,aZMin,aZMax,eNbEdges),
     mX0Loc   (0),
     mX1Loc   (xsz),
     mY0Loc   (0),
     mY1Loc   (ysz),

     sz      (init_sz()),

     largef  (100*cCRNode::MaxVal),
     mCRQ    (3*xsz*ysz,sizeof(int),2*xsz*ysz),
     mCRH    (xsz*ysz/4,(xsz*ysz)*2,4*(xsz+ysz),10*(xsz+ysz)),
     mLNodes  (new cCRNode [sz]), 
     mCNodes (AllocCNodes()),
     mNoLabel       (sz+2),
     mNoLabelBorder (sz+9),
     mSourceHeight  (mNoLabel),
     mSinkHeight    (0),
     mDirZPlus      (-1),
     mDirZMoins     (-1),
     mNbDischarge   (0),
     mNbLift        (0),
     mNbRelabel     (0),
     mNLA           (false)
{
	for (int ed=0 ; ed<eNbEdges ; ed++)
        {
            if ((xdelta[ed]==0)&&(ydelta[ed]==0)&&(zdelta[ed]==1))
               mDirZPlus = ed;
            if ((xdelta[ed]==0)&&(ydelta[ed]==0)&&(zdelta[ed]==-1))
                mDirZMoins = ed;
        }
        assert((mDirZPlus!=-1) && (mDirZMoins!=-1));


	/** global share crap **/

	// printf("--- max flow (%d,%d,%d) s=%d ---\n",mX1Glob,mY1Glob,mZMaxGlob-mZMinGlob,sz);

	// printf("--- sizeof(cCRNode) = %d ---\n",sizeof(cCRNode));
	// printf("--- total mem required = %dk ---\n",sz*sizeof(cCRNode)/1000);



	/** for relabel **/

	// printf("--- init ---\n");

	if( mLNodes==NULL ) 
        { 
              printf("Out of Mem! n\n");
              ElEXIT(-1,"Cox Roy Out of Memory"); 
        }


	mSinkFlow=0;

	// printf("--- Go! ---\n");

        // Begin Init()

	cRoyPt aPU(0,0,0);
	for ( aPU.mX=mX0Glob; aPU.mX<mX1Glob ; aPU.mX++)
        {
	    for (aPU.mY=mY0Glob; aPU.mY<mY1Glob ; aPU.mY++)
	    {
		 int  aZ0 = ZMin(aPU.mX,aPU.mY);
		 int  aZ1 = ZMax(aPU.mX,aPU.mY);
		 for (aPU.mZ=aZ0 ; aPU.mZ<aZ1 ; aPU.mZ++)
		 {
                     cCRNode & aNodU = NodeOfP(aPU);
		     aNodU.SetExcess(0);
		     aNodU.SetHeight(ZMaxGlob()-aPU.mZ);
		     aNodU.SetCurrentEdge(0);
                     for (int ed=0 ; ed<eNbEdges ; ed++)
                     {    
                          aNodU.SetResidualFlow(ed,cCRNode::MaxVal);
                     }

		 }
	    }
        }
        SetFlagEdges(mX0Glob,mY0Glob,mX1Glob,mY1Glob,true);
}


cInterfaceCoxRoyAlgo::~cInterfaceCoxRoyAlgo() {}

cInterfaceCoxRoyAlgo::cInterfaceCoxRoyAlgo
(
        int xsz,int ysz,
        signed short ** aZMin, signed short ** aZMax,
        int aNbVois
)  :
    mX0Glob  (0),
    mX1Glob  (xsz),
    mY0Glob  (0),
    mY1Glob  (ysz),
    mZMin    (aZMin),
    mZMax    (aZMax),
    mNbVois  (aNbVois)
{
}

int cInterfaceCoxRoyAlgo::XOfNumVois(int aNum) const {return xdelta[aNum];}
int cInterfaceCoxRoyAlgo::YOfNumVois(int aNum) const {return ydelta[aNum];}
int cInterfaceCoxRoyAlgo::ZOfNumVois(int aNum) const {return zdelta[aNum];}



template <class cCRNode> 
void cTplCoxRoyAlgo<cCRNode>::EndMaxFlowStd(int aX0,int aY0,int aX1,int aY1)
{
        mX0Loc = aX0;
        mY0Loc = aY0;
        mX1Loc = aX1;
        mY1Loc = aY1;

       SetFlagEdges(mX0Loc,mY0Loc,mX1Loc,mY1Loc,false);


	// Init();
	/** flood the first nodes **/
	for(int anY=mY0Loc ; anY<mY1Loc ; anY++)
        {
	    for(int anX=mX0Loc ; anX<mX1Loc ; anX++) 
            {
                 cCRNode *  aCol = ColumnOfP(anX,anY);
		 int  aZ0 = ZMin(anX,anY);
		 int  aZ1 = ZMax(anX,anY);
		 for (int aZ = aZ0; (aZ<aZ1) && (aCol[aZ].SourceConnected()) ; aZ++)
                 {

                     aCol[aZ].SetExcess(largef);
		     mCRH.LinkQInsert(cRoyPt(anX,anY,aZ),aCol[aZ].Height());
                 }
	    }
        }
	ResizeDetected=false;

        // End Init()



	int Level=mCRH.MaxKey();
        int Count =0;

	// printf("Starting at Key=%d\n",Level);

	/** generic preflow-push **/
	for(;;) 
        {
		/** This is the schedule for relabelling **/
		/** it can be changed for more/less relabel steps **/
		if( ((Count+1)%(2*sz/2)==0 && mSinkFlow!=0) || ResizeDetected ) 
		  {
			relabel();
			Level=mCRH.MaxKey();
			ResizeDetected=false;
		}

		/** update level to a non empty key **/
        mCRH.Set2NonEmptyKey(Level);

		/* if reached bottom, reset to top */
		if( Level<0 ) Level=mCRH.MaxKey(); /* highest level */

		/* if Level still <0 -> list MUST be empty! */
		if( Level<0 ) break; /* no more nodes! */

		/** get overflowing node with largest level **/
		cRoyPt aPU = mCRH.LinkQRemove(Level);

//		if( Count%100000==0 ) 
//		{
//        	mCRH.ShowLink();
			// printf("%6dk... SinkFlow=%8d  \n",Count/1000,mSinkFlow);
//		}

		Discharge(aPU);

		Count++;
	}

	// printf("SinkFlow=%d\n",mSinkFlow);
}


template <class cCRNode>  int cTplCoxRoyAlgo<cCRNode>::RecursifMaxFlowStd(int aX0,int aY0,int aX1,int aY1) 
{
    int  Xmil = (aX0+aX1)/2;


    EndMaxFlowStd(aX0,aY0,Xmil,aY1);

    EndMaxFlowStd(Xmil,aY0,aX1,aY1);

    EndMaxFlowStd(aX0,aY0,aX1,aY1);

    return(0);
}

template <class cCRNode> int cTplCoxRoyAlgo<cCRNode>::TopMaxFlowStd(short **Sol)
{
        EndMaxFlowStd(mX0Glob,mY0Glob,mX1Glob,mY1Glob);
	relabel();
	findcut(Sol);


	return mSinkFlow;
}

typedef cTplCoxRoyAlgo<cTplCRNode<unsigned char,6> >  cV4_UC_CoxRoyAlgo;
typedef cTplCoxRoyAlgo<cTplCRNode<unsigned char,10> > cV8_UC_CoxRoyAlgo;
typedef cTplCoxRoyAlgo<cTplCRNode<unsigned short,6> > cV4_US_CoxRoyAlgo;
typedef cTplCoxRoyAlgo<cTplCRNode<unsigned short,10> >cV8_US_CoxRoyAlgo;

typedef cTplCoxRoyAlgo<cTplCRNode<unsigned int,6> > cV4_UI_CoxRoyAlgo;
typedef cTplCoxRoyAlgo<cTplCRNode<unsigned int,10> >cV8_UI_CoxRoyAlgo;

cInterfaceCoxRoyAlgo * cInterfaceCoxRoyAlgo::NewOne
                       (
                           int xsz,int ysz,
                           signed short ** aDataZmin,
                           signed short ** aDataZmax,
                           bool  Cx8, 
                           bool  OnUChar
                       )
{

   bool VerifCxn = false;
   if (VerifCxn)
   {
       int aNbV = Cx8 ? 8  : 4;
       Pt2di  * aTabV =  Cx8 ? TAB_8_NEIGH : TAB_4_NEIGH;
       int aMinInterv = 100000;

       for (int aX1=0 ; aX1<xsz ; aX1++)
       {
           for (int aY1=0 ; aY1<ysz ; aY1++)
           {
                for (int aKV = 0 ; aKV < aNbV ; aKV++)
                {
                     int aX2 = aX1 + aTabV[aKV].x;
                     int aY2 = aY1 + aTabV[aKV].y;
                     if ((aX2>=0) && (aX2<xsz) && (aY2>=0) && (aY2<ysz))
                     {
                        int aZMin1 = aDataZmin[aY1][aX1];
                        int aZMin2 = aDataZmin[aY2][aX2];
                        int aZMinInter = ElMax(aZMin1,aZMin2);

                        int aZMax1 = aDataZmax[aY1][aX1];
                        int aZMax2 = aDataZmax[aY2][aX2];
                        int aZMaxInter = ElMin(aZMax1,aZMax2);

                        int aSzInterv = aZMaxInter-aZMinInter;

                        ElSetMin(aMinInterv,aSzInterv);

                        if (aSzInterv<=0)
                        {
                             std::cout << aZMinInter << " " << aZMaxInter << "\n";
                             ELISE_ASSERT(false," Bad Conx in Cox Roy");
                        }
                     }
                }
           }
       }
       std::cout << "Verif CX done " << aMinInterv << "\n";
   }


   if (OnUChar)
   {
       if (Cx8)
          return new cV8_UC_CoxRoyAlgo(xsz,ysz,aDataZmin,aDataZmax);
       else
          return new cV4_UC_CoxRoyAlgo(xsz,ysz,aDataZmin,aDataZmax);
   } 
   else
   {
       if (Cx8)
          return new cV8_US_CoxRoyAlgo(xsz,ysz,aDataZmin,aDataZmax);
       else
          return new cV4_US_CoxRoyAlgo(xsz,ysz,aDataZmin,aDataZmax);
   } 
}












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
