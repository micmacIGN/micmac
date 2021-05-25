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



#include "StdAfx.h"
#include "ext_stl/intheap.h"



/*********************************************/
/*                                           */
/*               cCoxAlgoGen                 */
/*                                           */
/*********************************************/


cCoxAlgoGen::cCoxAlgoGen
(
     INT aCostRegul,
     INT aCostV8,
     Pt2di aSz
)  :
   mCostV4  (aCostRegul),
   mCostV8  (aCostV8),
   mOrder   (-1)
{
}


static void InitInterv
            (
	      INT &aDelta,INT &aV0,INT &aV1,
	      INT aVMin,INT aVMax,bool isAsc
	    )
{
   if (isAsc)
   {
       aDelta = 1;
       aV0= aVMin;
       aV1 = aVMax;
   }
   else
   {
       aDelta = -1;
       aV0 = aVMax-1;
       aV1=  aVMin-1;
   }
}

void cCoxAlgoGen::NewOrder(Pt2di aP0,Pt2di aP1)
{
   mOrder++;
   InitInterv(mDx,mX0,mX1,aP0.x,aP1.x,(mOrder & 1)==0);
   InitInterv(mDy,mY0,mY1,aP0.y,aP1.y,(mOrder & 2)==0);
}

// Ordre : Vois Vert, puis 4-Voisin, puis 8-Voisin
// D'abord les coordonnees positives

/*
static cCoxAlgoGenRelVois aV0(Pt3di( 0, 0, 1),true ,true ,0       ,false  ,1);
static cCoxAlgoGenRelVois aV1
static cCoxAlgoGenRelVois aV2
static cCoxAlgoGenRelVois aV3
static cCoxAlgoGenRelVois aV4
static cCoxAlgoGenRelVois aV5
static cCoxAlgoGenRelVois aV6
static cCoxAlgoGenRelVois aV



const cCoxAlgoGenRelVois  cCoxAlgoGen::TheVois[10] ={aV0,aV0,aV0,aV0,aV0,aV0,aV0,aV0,aV0,aV0};
*/


cCoxAlgoGenRelVois::cCoxAlgoGenRelVois(Pt3di aP,bool aVert,bool aDirect,INT aNF,bool aV4,INT aSym) :
    mPt        (aP),
    mVert      (aVert),
    mDirect    (aDirect),
    mNumFlow   (aNF),
    mV4        (aV4),
    mSym       (aSym)
{
}
                 

const cCoxAlgoGenRelVois  cCoxAlgoGen::TheVois[10] =
{
   //                  Vert   Direct   NumFlow   V4
    cCoxAlgoGenRelVois(Pt3di( 0, 0, 1)  ,true  ,true     ,0       ,false  ,1),   // 0
    cCoxAlgoGenRelVois(Pt3di( 0, 0,-1)  ,true  ,false    ,0       ,false  ,0),   // 1
    cCoxAlgoGenRelVois(Pt3di( 1, 0, 0)  ,false ,true     ,1       ,true   ,4),   // 2
    cCoxAlgoGenRelVois(Pt3di( 0, 1, 0)  ,false ,true     ,2       ,true   ,5),   // 3
    cCoxAlgoGenRelVois(Pt3di(-1, 0, 0)  ,false ,false    ,1       ,true   ,2),   // 4
    cCoxAlgoGenRelVois(Pt3di( 0,-1, 0)  ,false ,false    ,2       ,true   ,3),   // 5
    cCoxAlgoGenRelVois(Pt3di( 1, 1, 0)  ,false ,true     ,3       ,false  ,8),
    cCoxAlgoGenRelVois(Pt3di(-1, 1, 0)  ,false ,true     ,4       ,false  ,9),
    cCoxAlgoGenRelVois(Pt3di(-1,-1, 0)  ,false ,false    ,3       ,false  ,6),
    cCoxAlgoGenRelVois(Pt3di( 1,-1, 0)  ,false ,false    ,4       ,false  ,7)
};


/*********************************************/
/*                                           */
/*               cCoxAlgo<NbV,TypeCost>      */
/*                                           */
/*********************************************/

static const INT ThePckL = 10;
static const INT TheMinPrio = -500;
static const INT TheMaxPrio = 1000;

static INT GlobV4;
static INT GlobV8;

template <const INT NbV,class TC,class TF>
          class cCoxAlgo :      public cInterfaceCoxAlgo,
                                public cCoxAlgoGen
{
	public :

    enum
           {
        // Numeros Symboliques
                   NoPere     = 255,
                   NumSource  = 254,

        // Constante Entiere
                   NVoisNOr   = NbV+1,
                   NVoisOr    = 2*(NbV+1)
           };
    private :
 
           typedef TC   tCost;
           typedef TF   tFlag;


       
           class cSom
           {
                   public :
                      void Reinit();
                      cSom();

                      void SetPere(U_INT1 aNumPere,cSom * aPere)
                      {
                            mNumPere = aNumPere;
                      }


                      bool HasNoPere() const {return mNumPere ==NoPere;}

                      void SetNoPere(){ mNumPere = NoPere;}
                      void SetNoFamille() {SetNoPere();}

                      INT Pere() const {return mNumPere;}
                      tCost & Cost() {return mCost;}


                      INT CapaResid(cSom * s2,INT k) 
		      {return Capa(s2,k)-ValueFlow(s2,k);}


		      INT CapaResidStd(cSom * s2,INT k,bool aStd)
		      {
                         return aStd ?
                                CapaResid(s2,k) :
                                s2->CapaResid(this,TheVois[k].mSym);
		      }


                      void  AddToFlow(INT aVal,cSom * s2,INT k)
                      {
                         INT aSign;
                         tCost * aCost = FlowCur(s2,k,aSign);
                         *aCost += aVal * aSign;
                      }

                      void SetKV (INT aKv) { mFlagV |= (1<<aKv);}
                      void SetNoKV () { mFlagV =0;}
                      bool  KV (INT aKv)   { return (mFlagV &(1<<aKv)) != 0;}

                      bool Impasse() {return (mImpasse != 0);}
                      void SetImpasse() {mImpasse = true;}
                      void SetNoImpasse() {mImpasse = false;}


                  private :
 

                      tCost * FlowCur (cSom *s2,INT k,INT & Sign)
                      {
                           INT kF = TheVois[k].mNumFlow;
        
                           if ( TheVois[k].mDirect )
                           {
                               Sign = 1;
                               return (mFlowCur+kF);
                           }
                           else
                           {
                               Sign = -1;
                               return s2->mFlowCur+kF;
                           }
                      }

                      INT Capa(cSom * s2,INT k) const
                      {

                        if (TheVois[k].mVert)
                            return TheVois[k].mDirect ? mCost : s2->mCost;
                        else
                        {
                            
                              return  1+ (mCost+s2->mCost+50)  / 100 ;
                              // return  (mCost+s2->mCost)  / (TheVois[k].mV4 ? 20 : 28) ;
                             //  return TheVois[k].mV4 ? GlobV4 : GlobV8 ;
                        }
                      }

                      INT   ValueFlow(cSom * s2,INT k)
                      {
                           INT aSign;
                           INT aVal = *FlowCur(s2,k,aSign);
                           return aSign  * aVal;
                      }



                      tCost mFlowCur[NbV+1];
                      tCost mCost;
                      tFlag mFlagV;
                      U_INT1 mImpasse;
                      U_INT1   mNumPere; //Numero depuis lequel le voit son pere


           };

           void ShowStateSom(Pt3di);
// Correction pour compilation windows
		   public:
           class cSomInStack
           {
              public :
                 cSomInStack(Pt2di aSrc,cSom * ,Pt3di,INT aPrio);
                 cSomInStack();  // Pour ElFifo

		 Pt2di  mSrc;
                 cSom * mPSom;
                 Pt3di  mPt;
		 INT    mPrio;
           };


    public :

	 ~cCoxAlgo();
         cCoxAlgo
         (
             Pt2di aSz,
             Fonc_Num aZMin,
             Fonc_Num aZMax,
             INT aCostRegul,
             REAL aRatioRegul8 = 0.32 // (2-sqrt(2))/ (2 sqrt(2)-1)
         );

	  INT  NbChem();
         void Reinit();

       // bool Inside(Pt3di aPt) const

         INT tx()     const {return mSoms.tx();}
         INT ty()     const {return mSoms.ty();}
         INT NbObj () const {return  mSoms.NbObj();}
         void Show();

         void SetCost(Pt3di aP, INT aCost);
         void SetCostBin
              (
	           Pt3di aP1,
		   cCoxAlgo<NbV,tCost,tFlag> & anAlgo
	      );
         // INT Cost(Pt3di aP);

         INT OneStepPCCNoRec(Pt2di aP0,Pt2di aP1);
	 INT  UltimePCCNoRec(Pt2di aP0,Pt2di aP1);
	 INT  PCCRec(Pt2di aP0,Pt2di aP1);
         INT  PccMaxFlow();

         Im2D_INT2  Sol(INT aDef);
         Im2D_INT2  SolMinCol(INT aDef);

         static inline tCost AdjustVal(INT aVal);
         static inline tCost CostMax();

    private :

           bool mModeHeap;
           class cSomZCmp
           {
                 public :
                   bool operator () (const cSomInStack & s1,const cSomInStack & s2)
                   {
                        return s1.mPrio < s2.mPrio;
                   }
           };


           INT AugmenChem(Pt3di);
//           void Reroutage(cSom *,Pt3di);
// Modif DB : pas implemente...
//           void MarquerImpasseDesendance(cSom *,Pt3di);
// Modif DB : pas implemente...
//           void CherchNouvChem(cSom *,Pt3di);
// Modif DB : pas implemente...

           void AddInStack(Pt2di aSrc,cSom *,Pt3di aP,INT aNumPere,cSom * aPere,INT aPrio);


           cTplNape3D<cSom> mSoms;
           ElFifo<cSomInStack> * mStack;
           ElBornedIntegerHeap<cSomInStack,ThePckL> * mHST;
           INT                          mStepOnePcc;
           Video_Win mW;
           void DrawRect(Pt2di aP0,Pt2di aP1,INT aCoul)
           {
               mW.fill_rect(Pt2dr(aP0),Pt2dr(aP1),mW.pdisc()(aCoul));
           }

	   Im2D_U_INT2        mImZMaxSource;
	   U_INT2 **          mDataZMaxSource;
           INT ZMinSource(Pt2di aP)  const {return mSoms.ZMin(aP);}
	   U_INT2 &           ZMaxSource(Pt2di aP) {return mDataZMaxSource[aP.y][aP.x];}

	   Im2D_U_INT2        mImZMinPuis;
	   U_INT2 **          mDataZMinPuis;
           INT ZMaxPuis(Pt2di aP)  const {return mSoms.ZMax(aP);}
	   U_INT2 &           ZMinPuis(Pt2di aP) {return mDataZMinPuis[aP.y][aP.x];}

	   Im2D_INT4        mCpt;
	   INT4 **          mDataCpt;
	   bool             mSP_Std;
	   INT & Cpt(Pt2di aP) {return mDataCpt[aP.y][aP.x];}

          void Push(const cSomInStack & aSom)
          {
              if (mModeHeap)
                mHST->push(aSom,aSom.mPrio);
              else
                mStack->push_back(aSom);
          }
          bool Empty()
          {
              if (mModeHeap)
                 return mHST->empty();
              else
                 return mStack->empty();
          }
          void Pop(cSomInStack & aCS)
          {
             if (mModeHeap)
	     {
		 INT aIndex;
                 mHST->pop(aCS,aIndex);
	     }
             else
                aCS=  mStack->popfirst(); 
          }

};



/*********************************************/
/*                                           */
/*     cCoxAlgo<NbV,TypeCost>::cSom          */
/*                                           */
/*********************************************/
template <const INT NbV,class TC,class TF>  
         void cCoxAlgo<NbV,TC,TF>::cSom::Reinit()
{

        SetNoFamille();

	for (INT aK=0 ; aK<NVoisNOr ; aK++)
           mFlowCur[aK] = 0;
        mCost = CostMax();
}

template <const INT NbV,class TC,class TF>  
         cCoxAlgo<NbV,TC,TF>::cSom::cSom() :
           mFlagV(0)
{
       mFlagV = 0;
}

/*********************************************/
/*                                           */
/*     cCoxAlgo<NbV,TypeCost>::cSomInStack   */
/*                                           */
/*********************************************/

template <const INT NbV,class TC,class TF>  
     cCoxAlgo<NbV,TC,TF>::cSomInStack::cSomInStack() :
     mPSom (0),
     mPt   (0,0,0),
     mPrio (0)
{
}

template <const INT NbV,class TC,class TF>  
     cCoxAlgo<NbV,TC,TF>::cSomInStack::cSomInStack(Pt2di aSrc,cSom * aPSom,Pt3di aPt,INT aPrio) :
     mSrc  (aSrc),
     mPSom (aPSom),
     mPt   (aPt),
     mPrio (aPrio)
{
}


/*********************************************/
/*                                           */
/*               cCoxAlgo<NbV,TypeCost>      */
/*                                           */
/*********************************************/



// 71339



template <const INT NbV,class TC,class TF>
   TC  cCoxAlgo<NbV,TC,TF>::CostMax()
{
   return (1<<(sizeof(tCost)*8-1))-1;
}

template <const INT NbV,class TC,class TF>
   TC  cCoxAlgo<NbV,TC,TF>::AdjustVal(INT aVal)
{
  return ElMax(0,ElMin(aVal,(INT)CostMax()));
}





template <const INT NbV,class TC,class TF>
         void cCoxAlgo<NbV,TC,TF>::SetCost(Pt3di aP, INT aCost)
{
   ELISE_ASSERT(mSoms.Inside(aP),"Pts out in cCoxAlgo");
   mSoms.El(aP).Cost() = AdjustVal(aCost);
}

template <const INT NbV,class TC,class TF>
         void cCoxAlgo<NbV,TC,TF>::SetCostBin
              (
	           Pt3di aP1,
		   cCoxAlgo<NbV,TC,TF> & anAlgo
	      )
{
   static bool First = false;
   if (First)
   {
	    cout << "NO DebugCot Chem in cCoxAlgo \n";
	    getchar();
	    First = true;
   }
}

/*
static Im2D_INT2 Init_ZMinMax(Im2D_INT2 aIZMin,Fonc_Num aFZ1,Fonc_Num aFZ2)
{
     Im2D_INT2  aIZMax(aIZMin.tx(),aIZMin.ty());

    Symb_FNum aS1(aFZ1),aS2(aFZ2);
    Symb_FNum aZmin(Min(aS1,aS2)),aZmax(Max(aS1,aS2));

    ELISE_COPY
    (
       aIZMin.all_pts(),
       Virgule(aZmin,aZmax),
       Virgule(aIZMin.out(),aIZMax.out())
    );


     return aIZMax;
}
*/


template <const INT NbV,class TC,class TF>  
         cCoxAlgo<NbV,TC,TF>::cCoxAlgo
	 (
               Pt2di    aSz,
               Fonc_Num aZMin,
               Fonc_Num aZMax,
               INT      aCostRegul,
               REAL     aRatioRegul8
	 ) :
	   cCoxAlgoGen (
                             AdjustVal(aCostRegul),
                             AdjustVal(round_ni(aCostRegul*aRatioRegul8)),
                             aSz
                       ),
	   mSoms       (aSz,aZMin,aZMax),
	   mStack      (new  ElFifo<cSomInStack> (2*tx()*ty())),
	   mHST        (new ElBornedIntegerHeap<cSomInStack,ThePckL>(TheMaxPrio-TheMinPrio)),
           mStepOnePcc (0),

           mW              (Video_Win::WStd(aSz,1.0)),
	   mImZMaxSource   (tx(),ty()),
	   mDataZMaxSource (mImZMaxSource.data()),
	   mImZMinPuis     (tx(),ty()),
	   mDataZMinPuis   (mImZMinPuis.data()),
	   mCpt            (tx(),ty()),
	   mDataCpt        (mCpt.data())

{
    Reinit();

    for (INT x=0; x<tx() ; x++)
    {
        for (INT y=0; y<ty() ; y++)
	{
             Pt2di aP(x,y);

             INT aZmin =  mSoms.ZMin(aP);
             INT aZmax =  mSoms.ZMax(aP);

             {
                 for (INT aZ=aZmin ; aZ<aZmax ; aZ++)
                 {
                     Pt3di aP(x,y,aZ);
                     cSom & aS = mSoms.El(aP);
                     for (INT aK = 0 ; aK<NVoisOr ; aK++)
                     {
                         Pt3di aQ = aP+TheVois[aK].mPt;
                         if (mSoms.Inside(aQ))
                             aS.SetKV(aK);
                     }
                 }
             }

	     INT aZmaxSource = aZmin+1;
	     INT aZminPuis =   aZmax-1;
	     Pt2di * mV2 = (NbV == 2) ? TAB_4_NEIGH :  TAB_8_NEIGH;
             for (INT aKV = 0; aKV<(NbV *2) ; aKV++)
	     {
		  Pt2di aPV = aP + mV2[aKV];
		  if (mSoms.Inside(aPV))
		  {
		      ElSetMax(aZmaxSource,mSoms.ZMin(aPV));
		      ElSetMin(aZminPuis, mSoms.ZMax(aPV));
		  }
	     }

	     ZMaxSource(aP) = ElMin(aZmaxSource,aZmax);
	     ZMinPuis(aP)   = ElMax(aZminPuis,aZmin);
        }
    }
}


template <const INT NbV,class TC,class TF>  cCoxAlgo<NbV,TC,TF>::~cCoxAlgo()
{
   delete mStack;
   delete mHST;
}



template <const INT NbV,class TC,class TF>  
         void cCoxAlgo<NbV,TC,TF>::Reinit()
{
    for (INT aK=0 ; aK<NbObj() ; aK++)
    {
	    mSoms.El(aK).Reinit();
    }
}


	     


template <const INT NbV,class TC,class TF>
         void cCoxAlgo<NbV,TC,TF>:: AddInStack
              (
	           Pt2di  aSrc,
	           cSom * aPSom,
		   Pt3di aP3,
		   INT   aNumPere,
	           cSom * aPere,
		   INT   aPrio
              )
{
    aPrio = ElMax(TheMinPrio,ElMin(TheMaxPrio,aPrio));
    aPSom->SetPere(aNumPere,aPere);
    aPSom->SetNoImpasse();
    Cpt(aSrc)++;
    Push(cSomInStack(aSrc,aPSom,aP3,aPrio));
}




template <const INT NbV,class TC,class TF>
         INT cCoxAlgo<NbV,TC,TF>::AugmenChem(Pt3di aP0)
{

   cSom * apS0 = &(mSoms.El(aP0));

   if (apS0->HasNoPere())
      return 0;

   if (apS0->Impasse())
      return 0;

   Pt3di aP3Fils = aP0; 
   cSom * aSFils = 0;
   INT aCapaMin = 1000000;

   if (mSP_Std)
   {
      aP3Fils = aP0; 
      aSFils = apS0;
      aCapaMin = aSFils->CapaResid(0,0);
   }

   // UVGcc cSom * aSReRoot = aSFils;
   Pt3di aPReRoot = aP3Fils;

   while ((aCapaMin>0) && (aSFils->Pere() < NVoisOr))
   {
       INT aKP = aSFils->Pere();
       Pt3di aP3Pere = aP3Fils-TheVois[aKP].mPt;
       cSom * aSPere  = &(mSoms.El(aP3Pere));

       INT aCapRes = aSPere->CapaResidStd(aSFils,aKP,mSP_Std);
       if (aCapRes <= aCapaMin)
       {
            aCapaMin = aCapRes;
            // UVGcc  aSReRoot = aSFils;
            aPReRoot = aP3Fils;
       }
       aP3Fils = aP3Pere;
       aSFils = aSPere;
   }


   if (aCapaMin >0)
   {
        ELISE_ASSERT(aSFils->Pere()==NumSource,"Incoherence in cCoxAlgo");

        aP3Fils = aP0; 
        aSFils = apS0;
        aSFils->AddToFlow(aCapaMin,0,0);
        while (aSFils->Pere() < NVoisOr)
        {
           INT aKP = aSFils->Pere();
           Pt3di aP3Pere = aP3Fils-TheVois[aKP].mPt;
           cSom * aSPere  = &(mSoms.El(aP3Pere));
        
           aSPere->AddToFlow(aCapaMin,aSFils,aKP);
           aP3Fils = aP3Pere;
           aSFils = aSPere;
        }

   }

   return aCapaMin;
}



template <const INT NbV,class TC,class TF>
         Im2D_INT2  cCoxAlgo<NbV,TC,TF>::SolMinCol(INT aDef)
{
    Im2D_INT2 aSol(tx(),ty());
    INT2 ** aData = aSol.data();

    for (INT x=0; x<tx() ; x++)
    {
        for (INT y=0; y<ty() ; y++)
	{
             Pt2di aP(x,y);
	     aData[y][x] = aDef;
	     INT costBest = 10000;

             INT aZmin =  mSoms.ZMin(aP);
             INT aZmax =  mSoms.ZMax(aP);
             cSom * aS  = &mSoms.El(Pt3di(x,y,aZmin));
             for (INT aZ = aZmin; aZ<aZmax ; aZ++)
	     {
                 INT aCost =aS->CapaResid(0,0);
		 if (aCost <costBest)
		 {
                      costBest = aCost;
		     aData[y][x] = aZ;
		 }
		 aS++;
	     }
	}
    }

    return aSol;
}

template <const INT NbV,class TC,class TF> 
   INT  cCoxAlgo<NbV,TC,TF>::NbChem()
{
	return 0;
}



template <const INT NbV,class TC,class TF>
         Im2D_INT2  cCoxAlgo<NbV,TC,TF>::Sol(INT aDef)
{
    Im2D_INT2 aSol(tx(),ty());
    INT2 ** aData = aSol.data();

    for (INT x=0; x<tx() ; x++)
    {
        for (INT y=0; y<ty() ; y++)
	{
             Pt2di aP(x,y);
	     if (mSoms.NbInCol(aP))
	     {
                INT aZ    =  mSoms.ZMin(aP);
                INT aZmax =  mSoms.ZMax(aP);
		cSom * aS  = &mSoms.El(Pt3di(x,y,aZ));

		bool cont = true;
		while (cont)
		{
		   if (aS->HasNoPere())
		   {
                       cont = false;
                       aZ--;
		   }
		   else if (aZ == aZmax-1)
		   {
                       cont = false;
		   }
		   else
		   {
                        aZ++;
                        aS  = &mSoms.El(Pt3di(x,y,aZ));
		   }
		}
                aData [y][x] = aZ;
	     }
	     else
	     {
                 aData[y][x] = aDef;
	     }
	}
    }

    return aSol;
}




#include <algorithm> 
bool CmpY(const Pt2di & aP1,const Pt2di & aP2)
{
	return aP1.y < aP2.y;
}
void OrderRand(Pt2di * aPts,INT aNb)
{
    for (INT k=0; k<aNb ; k++)
        aPts[k] = Pt2di(k,INT(1000*NRrandom3()));

    std::sort(aPts,aPts+aNb,CmpY);
}


template <const INT NbV,class TC,class TF>
         void cCoxAlgo<NbV,TC,TF>::ShowStateSom(Pt3di aP)
{
   cSom & aS = mSoms.El(aP);
   cout << "   Som : " << aP << " Pere : " << aS.Pere() << "\n";
}

static REAL aTotForW  = 0.0;
static REAL aTotBackW = 0.0;
static REAL aTotMarq = 0.0;

template <const INT NbV,class TC,class TF>
         INT cCoxAlgo<NbV,TC,TF>::OneStepPCCNoRec
             (
		 Pt2di aP0,
		 Pt2di aP1
             )
{
// cout << "One "  << aP0 << " " << aP1 << "\n";

     mStepOnePcc ++;
     mSP_Std =  true;

    mModeHeap = false;//(mStepOnePcc <=1);

    NewOrder(aP0,aP1);

 // Marque tous les sommets comme libres
	{

    for (INT anX=mX0 ; anX !=mX1 ; anX += mDx)
    {
        for (INT anY=mY0 ; anY !=mY1 ; anY += mDy)
	{
            Pt2di aP2(anX,anY);
	    INT aNbObj = mSoms.NbInCol(aP2);
	    cSom * pS0 = &  mSoms.El0(aP2);
	    for (INT kObj=0 ; kObj<aNbObj ; kObj++)
            {
               pS0->SetNoFamille();
               pS0->SetNoImpasse();
               pS0++;
            }
            Cpt(aP2) = 0; 
	}
    }
	}

    GlobV4 = mCostV4;
    GlobV8 = mCostV8;

// Met dans la file d'attente tous les sommets qui
// sont connectes a "Source" (les sommets de Z minimal)

// cout << "One , end Init"  << aP0 << " " << aP1 << "\n";


INT aNbSt =  0;
ElTimer aTF;

{
    for (INT anX=mX0 ; anX !=mX1 ; anX += mDx)
    {
        for (INT anY=mY0 ; anY !=mY1 ; anY += mDy)
        {
	    Pt2di aP(anX,anY);
	    INT aZ1 = (mSP_Std ?  ZMinSource(aP) :ZMinPuis(aP));
	    INT aZ2 = (mSP_Std ?  ZMaxSource(aP) :ZMaxPuis(aP));
	    for (INT aZ = aZ1; aZ<aZ2; aZ++)
	    {
	       Pt3di aP3(anX,anY,aZ);
               AddInStack
	       (
		   Pt2di(anX,anY),
		   &(mSoms.El(aP3)),
		   aP3,
		   NumSource,
                   (cSom *)0,
		   aZ
	       );
	       aNbSt++;
	    }
        }
    }

}
// cout << "One , end Stack 0 "  << aP0 << " " << aP1 << "\n";


    // Algo PCC de base, pique le premier sommet
    // de la pile et remet ses voisins non explores
    // en attente a la fin de la pile

    INT aCpt =0;
    Pt2di anOrder[NVoisOr];
    OrderRand(anOrder,NVoisOr);


    bool BUG = false;

   while (!Empty())
    {
	aCpt ++;
	// cout << "cpt " << aCpt << "\n";
	// BUG = (aCpt>=12917);
	//if ((aCpt %11==0) && (NRrandom3() <0.5)) OrderRand(anOrder,NVoisOr);
        // OrderRand(anOrder,NVoisOr);

	if (BUG) cout << "AA\n";

        cSomInStack aCS; Pop(aCS);
	cSom * pS1 = aCS.mPSom;

	for (INT aKord=0;  aKord<NVoisOr ; aKord++)
	{
	if (BUG) cout << "BB\n";
		INT aKV =  aKord; // anOrder[aKord].x;
                if (pS1->KV(aKV))
                {
        if (BUG) cout << "CCC\n";
                   Pt3di aP = aCS.mPt + TheVois[aKV].mPt;
if (false)
{
   cout << "Cpt " << aCpt << "\n";
   cout <<  aCS.mPt << " => " << aP << "\n";
   cout << "aDir = " << aKV << "\n";
   getchar();

}
        if (BUG) cout << aP <<  aCS.mPt << aKV << "\n";
                   {
                       cSom * pS2 = &(mSoms.El(aP));
		       if (pS2->HasNoPere())
		       {
        if (BUG) cout << "DDDD\n";
			  INT aCapaRes = 
				  pS1->CapaResidStd(pS2,aKV,mSP_Std);
			  if (aCapaRes > 0)
			  {
        if (BUG) cout << "EEEE\n";
		              INT aPrio = (mStepOnePcc == 1) ? 
				          aP.z : 
					  (500+Cpt(aCS.mSrc));

                              AddInStack(aCS.mSrc,pS2,aP,aKV,pS1,aPrio);
	                      aNbSt++;
        if (BUG) cout << "FFFFF\n";
			  }
        if (BUG) cout << "GGGGG\n";
		       }
        if (BUG) cout << "HHH\n";
                   }
        if (BUG) cout << "III\n";
                }
        if (BUG) cout << "JJJ\n";
	}
        if (BUG) cout << "KKK\n";
    }
aTotForW += aTF.uval();

ElTimer aTB;

    INT  aSomAugm = 0;
    INT aCptAug = 0;
    INT aCptStag = 0;


    INT aNbPuits = 0;
    bool First = true;
// cout << "One , end Stack 1 "  << aP0 << " " << aP1 << "\n";

    
    while (true)
    {
        for (INT anX=mX0 ; anX !=mX1 ; anX += mDx)
        {
            for (INT anY=mY0 ; anY !=mY1 ; anY += mDy)
            {
	        Pt2di aP(anX,anY);
	        INT aZ1 = ((!mSP_Std) ?  ZMinSource(aP) :ZMinPuis(aP));
	        INT aZ2 = ((!mSP_Std) ?  ZMaxSource(aP) :ZMaxPuis(aP));
	        for (INT aZ = aZ1; aZ<aZ2; aZ++)
	        {
		    if (First) aNbPuits++;
                    INT anAugm =  AugmenChem(Pt3di(anX,anY,aZ));
                    aSomAugm +=  anAugm;
	            if (anAugm)
                    {
                       aCptAug++;
		       aCptStag = 0;
                    }
	            else
		       aCptStag++;

	        }
//cout << First << " " << aCptStag << " " << aNbPuits << "\n";
                if ((!First) && (aCptStag >= aNbPuits)) goto EndBoucle;
            }
        }
	First = false;
    }

EndBoucle :

    aTotBackW += aTB.uval();


    return aSomAugm;
}

template <const INT NbV,class TC,class TF>
         INT cCoxAlgo<NbV,TC,TF>::UltimePCCNoRec
             (
		 Pt2di aP0,
		 Pt2di aP1
             )
{
//cout << "Ultime "  << aP0 << " " << aP1 << "\n";
	aP0.SetSup(Pt2di(0,0));
	aP1.SetInf(mSoms.Sz());
INT NbSom = 0;

ElTimer aTM;

     mStepOnePcc = 0;
     for (INT anX=aP0.x; anX<aP1.x; anX++)
     {
         for (INT anY=aP0.y; anY<aP1.y; anY++)
	 {
             Pt2di aP(anX,anY);

             INT aZmin =  mSoms.ZMin(aP);
             INT aZmax =  mSoms.ZMax(aP);
	     NbSom += (aZmax-aZmin);

             {
                 for (INT aZ=aZmin ; aZ<aZmax ; aZ++)
                 {
                     Pt3di aP(anX,anY,aZ);
                     cSom & aS = mSoms.El(aP);
                     aS.SetNoKV();
                     for (INT aK = 0 ; aK<NVoisOr ; aK++)
                     {
                         Pt3di aQ = aP+TheVois[aK].mPt;
                         if (
                                 (aQ.x>=aP0.x)&& (aQ.x<aP1.x)
                               &&(aQ.y>=aP0.y)&& (aQ.y<aP1.y)
                               &&(mSoms.ZInside(aQ))
			    )
                            aS.SetKV(aK);
                     }
                 }
             }
	 }
     }
     aTotMarq += aTM.uval();

     INT aNbTime =0;
     INT aRes =0;
     while (1)
     {
        INT aDelta = OneStepPCCNoRec(aP0,aP1);
	aRes += aDelta;
	if ((aDelta==0) || (aNbTime > 1000000))
           return aRes;
         aNbTime++;
     }
}


template <const INT NbV,class TC,class TF>
         INT cCoxAlgo<NbV,TC,TF>::PCCRec
             (
		 Pt2di aP0,
		 Pt2di aP1
             )
{

    INT aSzX = aP1.x-aP0.x;
    INT aSzY = aP1.y-aP0.y;

    if  ((aSzX >5) && (aSzY >5))
    {
       cout << "Pcc Rec " << aP0 <<  aP1 << "\n";
       cout << "  FW : " << aTotForW 
	    << "  BW : " << aTotBackW  
	    << "  Marq : " << aTotMarq  
	    << "\n ";
    }

    if ((aSzX<=5) && (aSzY<=5))
    {
       return UltimePCCNoRec (aP0,aP1);
    }



    if  (aSzX > aSzY)
    {
       INT aXMil = (aP0.x+aP1.x)/2;
       INT aRes  = PCCRec(aP0,Pt2di(aXMil,aP1.y));
       aRes += PCCRec(Pt2di(aXMil,aP0.y),aP1);

       DrawRect(aP0,aP1,P8COL::blue);
       
       INT dX=1;
       while (1)
       {
	  INT aX0 = ElMax(aP0.x,aXMil-dX);
	  INT aX1 = ElMin(aP1.x,aXMil+dX);

/*
          for (INT dY = 4; dY< aSzY/4 ; dY++)
          {
             for (INT y= aP0.y+dY/2; y<aP1.y; y+=dY)
             {
                   INT aY0 = ElMax(aP0.y,y-dY/2);
                   INT aY1 = ElMin(aP1.y,y+(3*dY)/2);
                   aRes += UltimePCCNoRec(Pt2di(aX0,aY0),Pt2di(aX1,aY1));
             }
          }
*/

          Pt2di aQ0(aX0,aP0.y);
          Pt2di aQ1(aX1,aP1.y);



          aRes += UltimePCCNoRec(aQ0,aQ1);
          DrawRect(aQ0,aQ1,P8COL::green);
          dX *=2;
	  if ((aX0<=aP0.x)&& (aX1>=aP1.x))
              return aRes;
       }
    }
    else
    {
       INT aYMil = (aP0.y+aP1.y)/2;
       INT aRes  = PCCRec(aP0,Pt2di(aP1.x,aYMil));
       aRes += PCCRec(Pt2di(aP0.x,aYMil),aP1);

       DrawRect(aP0,aP1,P8COL::blue);
       
       INT dY=1;
       while (1)
       {
	  INT aY0 = ElMax(aP0.y,aYMil-dY);
	  INT aY1 = ElMin(aP1.y,aYMil+dY);

/*
          for (INT dX = 4; dX< aSzX/4 ; dX++)
          {
             for (INT x= aP0.x+dX/2; x<aP1.x; x+=dX)
             {
                   INT aX0 = ElMax(aP0.x,x-dX/2);
                   INT aX1 = ElMin(aP1.x,x+(3*dX)/2);
                   aRes += UltimePCCNoRec(Pt2di(aX0,aY0),Pt2di(aX1,aY1));
             }
          }
*/

          Pt2di aQ0(aP0.x,aY0);
          Pt2di aQ1(aP1.x,aY1);

          aRes += UltimePCCNoRec(aQ0,aQ1);
          DrawRect(aQ0,aQ1,P8COL::green);
          dY *=2;
	  if ((aY0<=aP0.y)&& (aY1>=aP1.y))
              return aRes;
       }
    }
}

template <const INT NbV,class TC,class TF>
         INT cCoxAlgo<NbV,TC,TF>::PccMaxFlow()
{
    aTotForW = 0;
    aTotBackW = 0;

    INT aRes = PCCRec(Pt2di(0,0),mSoms.Sz());
    cout << "END  FW : " << aTotForW 
         << "  BW : " << aTotBackW  
	 << "  Marq : " << aTotMarq  
	 << "\n ";

     return aRes;
}




template <const INT NbV,class TC,class TF>
         void cCoxAlgo<NbV,TC,TF>::Show()
{
/*
    cout << "------------COX-------------\n";
    for (INT x=0 ; x<mSoms.tx() ; x++)
    {
        for (INT y=0 ; y<mSoms.ty() ; y++)
        {
	   Pt2di aP(x,y);
	   INT z0 = mSoms.ZMin(aP);
	   INT z1 = mSoms.ZMax(aP);
           for (INT z=z0; z<z1 ; z++)
	   {
		Pt3di aP(x,y,z);
		cSom * aS1 =  &mSoms.El(aP);
		cout << "  " << aP << " ";
	        for (INT aKV=0;  aKV<NVoisOr ; aKV++)
	        {
		     cout << aKV << " ";
                     Pt3di aQ = aP + TheVois[aKV].mPt;
		     if (mSoms.Inside(aQ) || (aKV == 0))
		     {
		        cSom* aS2= mSoms.Inside(aQ)?&mSoms.El(aQ):0;
			cout << "[" << "FC= " <<(INT) aS1->mFlowCur[aKV] << " " << "; CR=" << aS1->CapaResid(aS2,aKV) << "]";
		     }
		     else
			     cout << "- ";
	        }
		cout << "\n";
	   }
	}
    }
    cout << "-----------------------\n";
*/
}

 
template class cCoxAlgo<2,INT1,U_INT1>;
template class cCoxAlgo<4,INT1,U_INT1>;

/**************************************************/
/*                                                */
/*            cInterfaceCoxAlgo                   */
/*                                                */
/**************************************************/

cInterfaceCoxAlgo::~cInterfaceCoxAlgo() {}

cInterfaceCoxAlgo * cInterfaceCoxAlgo::StdNewOne        
                 (
                          Pt2di    aSz,
                          Fonc_Num aZMin,
                          Fonc_Num aZMax,
                          INT      aCostRegul,
                          bool     V8
                    ) 
{
   if (V8)
      return new  cCoxAlgo<4,INT1,U_INT1>(aSz,aZMin,aZMax,aCostRegul,0.7);
   else
      return new  cCoxAlgo<2,INT1,U_INT1>(aSz,aZMin,aZMax,aCostRegul,0.7);
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
