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



#ifndef _ELISE_EXT_STL_CLUSTER
#define _ELISE_EXT_STL_CLUSTER

#include "ext_stl/heap.h"

/*
   Classe pour 
*/

#ifndef DEBUG
	#define DEBUG true
#endif

template <class Type,class TAttrLnk> class cMergingNode;
template <class Type,class TAttrLnk> class IndexMergingNode;
template  <class Type,class  TAttrLnk> class cCmpMNode;
template <class Type,class TAttrLnk,class TParam> class cAlgoMergingRec;


class cNoAttr
{
};


template <class Type,class TAttrLnk> class cMergingNode
{
     public :
         static const int TheNbFils = 2;
         typedef cMergingNode<Type,TAttrLnk>   tNode;
         typedef tNode * tPtrMN;
         typedef std::set<tNode *>  tSetN;
         typedef std::list<tNode *>  tListN;

         // ============== constructeur===================

         cMergingNode(Type * aVal,int aNum) :
             mNum   (aNum),
             mGain  (0.0),
             mVal   (aVal),
             mDepth (0)
         {
              mFils[0] = mFils[1] = 0;
         }
         cMergingNode(tPtrMN aF0,tPtrMN aF1,double aGain,int aNum) :
             mNum    (aNum),
             mGain   (aGain),
             mVal    (0),
             mDepth  (1 + ElMax(aF0->mDepth,aF1->mDepth))
         {
              mFils[0] = aF0;
              mFils[1] = aF1;
              for (int aKF=0 ; aKF<NbFils() ; aKF++)
              {
                 mFils[aKF]->mFathers.insert(this);
              }
         }


         // ============ Parentee =====================

         tPtrMN  UnikBro(tPtrMN  aBro)
         {
             ELISE_ASSERT(NbFils()==2,"UnikBro Not 2");
             if (mFils[0] == aBro) return mFils[1];
             if (mFils[1] == aBro) return mFils[0];

             std::cout << "UNIK BRO " << Num() << " " << aBro->Num() << "\n";
             ELISE_ASSERT(false,"UnikBro");
             return 0;
         }


         void EraseFathers()
         {
            mFathers.clear();
         }

         void EraseFromFathers(tNode * aPere)
         {
            int aNbSupr = (int)mFathers.erase(aPere);
            ELISE_ASSERT(aNbSupr==1,"Chek failed in mRoots.erase/UpdateAdoption");
         }


         int NbFils() const {return TheNbFils;}
         tPtrMN FilsK(int aKF) const {return mFils[aKF];}
         const tSetN & Fathers () const {return mFathers;} 

         // ============ Autres, attributs ... =====================

         double & Gain() {return mGain;}
         int &  HeapIndex()  {return mHeapIndex;}
         TAttrLnk & Attr() {return mAttr;}
         Type *  Val() {return mVal;}
         int   Depth() const  {return mDepth;}
         int   Num() const  {return mNum;}

          void GetVals(std::vector<Type *> & aVV) const
          {
               if (mVal)
                  aVV.push_back(mVal);
               for (int aK=0 ; aK< NbFils() ; aK++)
               {
                   if (mFils[aK])
                      mFils[aK]->GetVals(aVV);
               }
          }

          // A appeler par  cAlgoMergingRec

     private :
            int               mNum;
            tPtrMN            mFils[TheNbFils];
            tSetN             mFathers; // N'est construit que pour les cluster existant et tant qu'ils sont "top"

            double            mGain;
            Type *            mVal;
            TAttrLnk          mAttr;
            int               mDepth;
            int               mHeapIndex;
};

template <class Type,class TAttrLnk> class IndexMergingNode
{
     public :
        typedef cMergingNode<Type,TAttrLnk>   tNode;
        static void SetIndex(tNode * & aNode,int i) 
        {
                aNode->HeapIndex() = i;
        }
        static int  Index(tNode * & aNode)
        {
             return aNode->HeapIndex();
        }
};


template  <class Type,class  TAttrLnk> class cCmpMNode
{
    public :
        typedef cMergingNode<Type,TAttrLnk>   tNode;
        bool operator () (tNode * aN1,tNode * aN2)
        {
            return aN1->Gain() > aN2->Gain();
        }
};


template <class Type,class TAttrLnk,class TParam> class cAlgoMergingRec
{
     public :
            typedef cMergingNode<Type,TAttrLnk>      tNode;
            typedef cCmpMNode<Type,TAttrLnk>         tCmp;
            typedef IndexMergingNode<Type,TAttrLnk>  tInd;
            typedef std::set<tNode*>                 tSetN;
            typedef std::list<tNode*>                tListN;

            typedef std::vector<Type*>               tVecV;
            typedef std::pair<Type*,Type*>           tPairV;
            typedef std::set<tPairV>                 tSetPairV;
            typedef std::list<tPairV>                tListPairV;
            
            cAlgoMergingRec(const std::vector<Type *> &aVals,TParam & aParam,int aNivShow) :
                mNumNode (0),
                mParam   (aParam),
                mHeap    (mCmp),
                mVals    (aVals),
                mNivShow (aNivShow)
             {
                 Init();
                 DoMerge();
             }

             const std::set<tNode *> & Roots () {return mRoots;}

             void Show(tNode * aNode)
             {
                 Show(aNode,0);
             }
     private :
             void Blank (int aLevel)
             {
                for (int aK=0 ; aK<aLevel; aK++)
                    std::cout << "=====";
             }
             void Show(tNode * aNode,int aLevel)
             {
                if (aNode->Val())
                {
                    Blank(aLevel);
                    mParam.Show(aNode->Val());
                    std::cout << "\n";
                }
                else
                {
                     Show(aNode->FilsK(0),aLevel+1);
                     Blank(aLevel);
                     std::cout  << "[" << aNode->Num()<< "]\n";
                     Show(aNode->FilsK(1),aLevel+1);
                }
             }

           void Init();
           void DoMerge();
           void UpdateAdoption(tNode * aNode,tNode * aSelectedF,tSetN & aNewVois);
           int CreateNewCandidate(tNode *aN1,tNode * aN2);
           void AddToListPair(const tVecV & aV1,const tVecV & aV2,tListPairV &);

           int                         mNumNode;
           TParam                      mParam;
           tCmp                        mCmp;
           ElHeap<tNode *,tCmp,tInd >  mHeap;

           std::vector<Type*>  mVals;
           std::set<tNode *>   mRoots;
           int                 mNivShow;
};


template <class Type,class TAttrLnk,class TParam> 
         void cAlgoMergingRec<Type,TAttrLnk,TParam>::UpdateAdoption
              (
                   tNode * anAdopted,
                   tNode * aSelectedF,
                   tSetN & aNewVois
              )
{
   int aNbSupr = (int)mRoots.erase(anAdopted);
   ELISE_ASSERT(aNbSupr==1,"Chek failed in mRoots.erase/UpdateAdoption");

   int aNbPereGot = 0;  // Pour checker
   const tSetN & aLF = anAdopted->Fathers () ;
   for (typename tSetN::const_iterator itF=aLF.begin(); itF!=aLF.end() ; itF++)
   {
       tNode * anOldF = *itF;
       if (anOldF==aSelectedF)
       {
           aNbPereGot++;
       }
       else
       {
            tNode * aBro = anOldF->UnikBro(anAdopted);
            aNewVois.insert(aBro);
            aBro->EraseFromFathers(anOldF);
            mHeap.Sortir(anOldF);


             if (! DEBUG)
               delete anOldF;
       }
   }
   ELISE_ASSERT(aNbPereGot==1,"Chek failed aNbPereGot/UpdateAdoption");

   anAdopted->EraseFathers(); // Les peres deviennent inutile, donc recupere la place
}


template <class Type,class TAttrLnk,class TParam> 
         void cAlgoMergingRec<Type,TAttrLnk,TParam>::DoMerge()
{
// double aTimeA = 0;
// double aTimeB = 0;
// double aTimeC = 0;
// double aTimeD = 0;
    for (;;)
    {
        // Victoire tout est fusionne
        if (mRoots.size() == 1)
           return;

        tNode * aNew=0;
        if (! mHeap.pop(aNew))
           return;  // Plus de merge possible, on sort avec une foret

       // On a trouve un nouveau noeud

            // Mise a jour des fils


        tSetN  aNewVois;
        for (int aKF=0 ; aKF<aNew->NbFils() ; aKF++)
        {
           UpdateAdoption(aNew->FilsK(aKF),aNew,aNewVois);
        }
        mRoots.insert(aNew);

            // Mise

        int aNbPair = 0;
        for (typename tSetN::const_iterator itN=aNewVois.begin(); itN!=aNewVois.end() ; itN++)
        {
             aNbPair += CreateNewCandidate(*itN,aNew);
        }

        mParam.OnNewMerge(aNew);
        if (mNivShow >=2)
        {
            std::cout << "Root " << mRoots.size()  
                     << " Heap " <<  mHeap.nb() 
                     << " NbV " << aNewVois.size() 
                     << " NbP " <<  aNbPair
                     << " Moy-NbP " <<  aNbPair/double(aNewVois.size())
                      << "\n";
        }
    }
}


template <class Type,class TAttrLnk,class TParam> 
         void cAlgoMergingRec<Type,TAttrLnk,TParam>::AddToListPair
              (
                    const tVecV & aV1Glob,
                    const tVecV & aV2Glob,
                    tListPairV & aSetPair
              )
{
   std::set<Type *> aS2(aV2Glob.begin(),aV2Glob.end());

   for (int aK1=0 ; aK1<int(aV1Glob.size()) ; aK1++)
   {
       Type * aV1 = aV1Glob[aK1];
       std::vector<Type*>  aV2Loc;
       mParam.Vois(aV1,aV2Loc);
       for (int aK2=0 ; aK2<int(aV2Loc.size()) ; aK2++)
       {
            Type * aV2 = aV2Loc[aK2];
            if (aS2.find(aV2)!=aS2.end())
            {
               aSetPair.push_back(tPairV(aV1,aV2));
            }
       }
   }
}


template <class Type,class TAttrLnk,class TParam> 
         int  cAlgoMergingRec<Type,TAttrLnk,TParam>::CreateNewCandidate
              (
                  tNode * aN1,
                  tNode * aN2
              )
{
    if ((aN1==0) || (aN2==0) || (aN1==aN2))
    {
        std::cout << aN1 << " " << aN2 << "\n";
        ELISE_ASSERT
        (
             false,
             "Arcs incohe in cAlgoMergingRec<>::init"
        );
    }
    std::vector<Type*>  aVV1;
    std::vector<Type*>  aVV2;
 
    aN1->GetVals(aVV1);
    aN2->GetVals(aVV2);

    tListPairV aSetPair;

    AddToListPair(aVV1,aVV2,aSetPair);
    AddToListPair(aVV2,aVV1,aSetPair);

    if (aSetPair.empty())
       return 0;

    double aGain = mParam.Gain(aN1,aN2,aVV1,aVV2,aSetPair,mNumNode);
    if (aGain  <0)
       return 0;


    cMergingNode<Type,TAttrLnk> * aMerge = new cMergingNode<Type,TAttrLnk>(aN1,aN2,aGain,mNumNode++);
    mHeap.push(aMerge);
    if (mNivShow >=3)
       std::cout << " Pair " << aSetPair.size() << " Sets " <<  aVV1.size() << " " << aVV2.size() << "\n";
    // mParam.OnNewCandidate(aMerge);
    return (int)aSetPair.size();
}


template <class Type,class TAttrLnk,class TParam> void cAlgoMergingRec<Type,TAttrLnk,TParam>::Init()
{
    // On cree les cluster singletons
    std::map<Type *,tNode*> aDicV2N;
    for (int aK=0 ; aK <int(mVals.size()) ; aK++)
    {
        Type * aVal = mVals[aK];
        tNode  * aSingl = new tNode (aVal,mNumNode++);
        aDicV2N[aVal] = aSingl;
        mRoots.insert(aSingl);
        mParam.OnNewLeaf(aSingl);
    }

    // On cree les cluster potentiels 
    
    tSetPairV aSetPair;
    for (int aK1=0 ; aK1 <int(mVals.size()) ; aK1++)
    {
        Type * aV1 = mVals[aK1];
        std::vector<Type*> aVois;
        mParam.Vois(aV1,aVois);
        for (int aK2=0 ; aK2 <int(aVois.size()) ; aK2++)
        {
            Type * aV2 = aVois[aK2];
            Type * aV1Loc = aV1;
            if (aV1Loc > aV2)
               ElSwap(aV1Loc,aV2);
            aSetPair.insert(std::pair<Type*,Type*>(aV1Loc,aV2));
        }
    }
    for 
    (
         typename tSetPairV::const_iterator itP=aSetPair.begin();
         itP!=aSetPair.end();
         itP++
    )
    {
        tNode* aN1 = aDicV2N[itP->first];
        tNode* aN2 = aDicV2N[itP->second];


        CreateNewCandidate(aN1,aN2);
    }
}


/*************************************************************/
/* SOME UTILS ON TAB                                         */
/*************************************************************/





#endif  //  _ELISE_EXT_STL_CLUSTER


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
