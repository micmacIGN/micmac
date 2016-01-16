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


class cComMergeTieP
{
    public  :
        bool IsOk() const {return mOk;}
        void SetNoOk() {mOk=false;}
        void SetOkForDelete() {mOk=true;}  // A n'utiliser que dans cFixedMergeStruct::delete
        int  NbArc() const {return mNbArc;}
        void IncrArc() { mNbArc++;}
    protected :
        cComMergeTieP();
        bool  mOk;
        int   mNbArc;

};


template <class Type>  class cVarSizeMergeTieP : public cComMergeTieP
{
     public :
       typedef Type                    tVal;
       typedef cVarSizeMergeTieP<Type> tMerge;
       //  typedef std::map<Type,tMerge *>     tMapMerge;
       typedef  DefcTpl_GT<Type,tMerge> tMapMerge;

       cVarSizeMergeTieP() ;
       void FusionneInThis(cVarSizeMergeTieP<Type> & anEl2,std::vector<tMapMerge> &  Tabs); 
       void AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2);

        bool IsInit(int aK) const ;
        const Type & GetVal(int aK) const ;
        int  NbSom() const ;
        void AddSom(const Type & aV,int aK);
     private :

        std::vector<int>   mVecInd;
        std::vector<Type>  mVecV;
};

template <const int TheNbPts,class Type>  class cFixedSizeMergeTieP : public cComMergeTieP
{
     public :
       typedef Type                    tVal;
       typedef cFixedSizeMergeTieP<TheNbPts,Type> tMerge;
       //  typedef std::map<Type,tMerge *>     tMapMerge;
       typedef  DefcTpl_GT<Type,tMerge> tMapMerge;

       cFixedSizeMergeTieP() ;
       void FusionneInThis(cFixedSizeMergeTieP<TheNbPts,Type> & anEl2,std::vector<tMapMerge> &  Tabs);
       void AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2);

        bool IsInit(int aK) const;
        const Type & GetVal(int aK) const;
        int  NbSom() const ;
        void AddSom(const Type & aV,int aK);
     private :

        Type mVals[TheNbPts];
        bool  mTabIsInit[TheNbPts];
};


// cFixedMergeStruct
template <class Type> class cStructMergeTieP
{
     public :
        typedef Type        tMerge;
        typedef typename Type::tVal  tVal;

        typedef  DefcTpl_GT<tVal,tMerge> tMapMerge;
        typedef typename tMapMerge::GT_tIter         tItMM;

        // Pas de delete implicite dans le ~X(),  car exporte l'allocation dans
        void Delete();
        void DoExport();
        const std::list<tMerge *> & ListMerged() const;


        void AddArc(const tVal & aV1,int aK1,const tVal & aV2,int aK2);
        cStructMergeTieP(int aNbVal);

        const tVal & ValInf(int aK) const {return mEnvInf[aK];}
        const tVal & ValSup(int aK) const {return mEnvSup[aK];}


     private :
        cStructMergeTieP(const cStructMergeTieP<Type> &); // N.I.
        void AssertExported() const;
        void AssertUnExported() const;
        void AssertUnDeleted() const;

        int                                 mTheNb;
        std::vector<tMapMerge>              mTheMapMerges;
        std::vector<tVal>                   mEnvInf;
        std::vector<tVal>                   mEnvSup;
        std::vector<int>                    mNbSomOfIm;
        std::vector<int>                    mStatArc;
        bool                                mExportDone;
        bool                                mDeleted;
        std::list<tMerge *>                 mLM;
};

/*
*/



/**************************************************************************/
/*                                                                        */
/*             cComMergeTieP                                              */
/*                                                                        */
/**************************************************************************/


cComMergeTieP::cComMergeTieP() :
  mOk (true),
  mNbArc (0)
{
}

/**************************************************************************/
/*                                                                        */
/*        cVarSizeMergeTieP  / cFixedSizeMergeTieP                        */
/*                                                                        */
/**************************************************************************/

   // ======================= Constructeurs =========================

template <class Type> 
cVarSizeMergeTieP<Type>::cVarSizeMergeTieP() :
     cComMergeTieP()
{
}

template <const int TheNbPts,class Type>
cFixedSizeMergeTieP<TheNbPts,Type>::cFixedSizeMergeTieP() :
     cComMergeTieP()
{
    for (int aK=0 ; aK<TheNbPts; aK++)
    {
        mTabIsInit[aK] = false;
    }

}

   // ======================= Addsom =========================


template <class Type> 
void  cVarSizeMergeTieP<Type>::AddSom(const Type & aV,int anInd)
{
     for (int aKI=0 ; aKI< int(mVecInd.size()) ; aKI++)
     {
         if ( mVecInd[aKI] == anInd)
         {
             if (mVecV[aKI] != aV) mOk = false;
             return;
         }
     }

     mVecV.push_back(aV);
     mVecInd.push_back(anInd);
}

template <const int TheNbPts,class Type>
   void  cFixedSizeMergeTieP<TheNbPts,Type>::AddSom(const Type & aV,int aK)
{
     if (mTabIsInit[aK])
     {
        if (mVals[aK] != aV)
        {
           mOk = false;
        }
     }
     else
     {
        mVals[aK] = aV;
        mTabIsInit[aK] = true;
     }
}


   // ======================= NbSom  =========================

template <class Type> 
int  cVarSizeMergeTieP<Type>::NbSom() const
{
   return mVecV.size();
}

template <const int TheNbPts,class Type>   
int cFixedSizeMergeTieP<TheNbPts,Type>::NbSom() const
{
   int aRes=0; 
   for (int aK=0 ; aK<TheNbPts ; aK++)
   {
       if (mTabIsInit[aK])
       {
           aRes++;
       }
   }
   return aRes;
}

   // ======================= AddArc =========================

template <class TTieP,class Type> void AddArcTieP(TTieP & aTieP,const Type & aV1,int aK1,const Type & aV2,int aK2)
{
    aTieP.AddSom(aV1,aK1);
    aTieP.AddSom(aV2,aK2);
    aTieP.IncrArc();
}

 
template <class Type> 
void  cVarSizeMergeTieP<Type>::AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2)
{
    AddArcTieP(*this,aV1,aK1,aV2,aK2);
}
template <const int TheNbPts,class Type>
   void  cFixedSizeMergeTieP<TheNbPts,Type>::AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2)
{
    AddArcTieP(*this,aV1,aK1,aV2,aK2);
}


   // ======================= IsInit =========================

template <class Type> 
bool  cVarSizeMergeTieP<Type>::IsInit(int anInd) const
{
     for (int aKI=0 ; aKI< int(mVecInd.size()) ; aKI++)
     {
         if ( mVecInd[aKI] == anInd)
            return true;
     }
     return false;
}



template <const int TheNbPts,class Type>
   bool  cFixedSizeMergeTieP<TheNbPts,Type>::IsInit(int aK) const
{
   return mTabIsInit[aK];
}

   // ======================= GetVal =========================

template <class Type> 
const Type &    cVarSizeMergeTieP<Type>::GetVal(int anInd) const
{
     for (int aKI=0 ; aKI< int(mVecInd.size()) ; aKI++)
     {
         if ( mVecInd[aKI] == anInd)
            return mVecV[aKI];
     }
     ELISE_ASSERT(false,":::GetVal");
     return mVecV[0];
}

template <const int TheNbPts,class Type>
const Type & cFixedSizeMergeTieP<TheNbPts,Type>::GetVal(int aK) const
{
   return mVals[aK];
}


   // ======================= FusionneInThis =========================
       // void FusionneInThis(cFixedSizeMergeTieP<TheNbPts,Type> & anEl2,std::vector<tMapMerge> &  Tabs);

template <class Type>   
void cVarSizeMergeTieP<Type>::FusionneInThis(cVarSizeMergeTieP<Type> & anEl2,std::vector<tMapMerge> &  Tabs)
{
     if ((!mOk) || (! anEl2.mOk))
     {
         mOk = anEl2.mOk = false;
         return;
     }
     mNbArc += anEl2.mNbArc;
     for (int aK2=0 ; aK2<int(anEl2.mVecV.size()); aK2++)
     {
          int anInd2 = anEl2.mVecInd[aK2];
          const Type  & aV2 = anEl2.mVecV[aK2];
          for (int aK1=0 ; aK1<int(mVecInd.size()) ; aK1++)
          {
              if (mVecInd[aK1] == anInd2)
              {
                  // Ce cas ne devrait pas se produire, il doivent avoir ete fusionnes
                  ELISE_ASSERT(mVecV[aK1]!= aV2,"cFixedMergeTieP");
                  mOk = anEl2.mOk = false;
                  return;
              }
          }
          mVecInd.push_back(anInd2);
          mVecV.push_back(aV2);
          Tabs[anInd2].GT_SetVal(aV2,this);
     }
}


template <const int TheNbPts,class Type>   
void cFixedSizeMergeTieP<TheNbPts,Type>::FusionneInThis(cFixedSizeMergeTieP<TheNbPts,Type> & anEl2,std::vector<tMapMerge> &  Tabs)
{

     if ((!mOk) || (! anEl2.mOk))
     {
         mOk = anEl2.mOk = false;
         return;
     }
     mNbArc += anEl2.mNbArc;
     for (int aK=0 ; aK<TheNbPts; aK++)
     {
         if ( mTabIsInit[aK] && anEl2.mTabIsInit[aK] )
         {
            // Ce cas ne devrait pas se produire, il doivent avoir ete fusionnes
            ELISE_ASSERT(mVals[aK]!= anEl2.mVals[aK],"cFixedMergeTieP");
            mOk = anEl2.mOk = false;
            return;
         }
         else if ( (!mTabIsInit[aK]) && anEl2.mTabIsInit[aK] )
         {
            mVals[aK] = anEl2.mVals[aK] ;
            mTabIsInit[aK] = true;
            // GT::  Tabs[aK][mVals[aK]] = this;
            Tabs[aK].GT_SetVal(mVals[aK],this);
         }
     }
}

/**************************************************************************/
/*                                                                        */
/*                           cStructMergeTieP                             */
/*                                                                        */
/**************************************************************************/

template <class Type> cStructMergeTieP<Type>::cStructMergeTieP(int aNb) :
    mTheNb      (aNb),
    mTheMapMerges    (aNb),
    mEnvInf     (aNb),
    mEnvSup     (aNb),
    mNbSomOfIm  (aNb,0),
    mStatArc    (),
    mExportDone (false),
    mDeleted    (false)
{
}



template <class Type>
  void cStructMergeTieP<Type>::AddArc(const tVal & aV1,int aK1,const tVal & aV2,int aK2)
{

            ELISE_ASSERT((aK1!=aK2) && (aK1>=0) && (aK1<mTheNb) && (aK2>=0) && (aK2<mTheNb),"cStructMergeTieP::AddArc Index illicit");

            AssertUnExported();
            tMapMerge & aMap1 = mTheMapMerges[aK1];
            tMerge * aM1 = aMap1.GT_GetVal(aV1);

            tMapMerge & aMap2 = mTheMapMerges[aK2];
            tMerge * aM2 = aMap2.GT_GetVal(aV2);
            tMerge * aMerge = 0;
             


             if ((aM1==0) && (aM2==0))
             {
                 aMerge = new tMerge;
                 aMap1.GT_SetVal(aV1,aMerge);
                 aMap2.GT_SetVal(aV2,aMerge);
                 //  GT::aMap1[aV1] = aMerge;
                 //  GT::aMap2[aV2] = aMerge;
             }
             else if ((aM1!=0) && (aM2!=0))
             {
                  if (aM1==aM2) 
                  {   
                     aM1->IncrArc();
                     return;
                  }
                  aM1->FusionneInThis(*aM2,mTheMapMerges);
                  if (aM1->IsOk() && aM2->IsOk())
                  {
                     delete aM2;
                     aMerge = aM1;
                  }
                  else
                     return;
             }
             else if ((aM1==0) && (aM2!=0))
             {
                 // GT:: aMerge = mTheMaps[aK1][aV1] = aM2;
                 aMerge = aM2;
                 mTheMapMerges[aK1].GT_SetVal(aV1,aM2);
             }
             else
             {
                 // GT :: aMerge =  mTheMaps[aK2][aV2] = aM1;
                 aMerge = aM1;
                 mTheMapMerges[aK2].GT_SetVal(aV2,aM1);
             }
             aMerge->AddArc(aV1,aK1,aV2,aK2);
}


template <class Type> void cStructMergeTieP<Type>::Delete()
{
    for (int aK=0 ; aK<mTheNb ; aK++)
    {
        tMapMerge & aMap = mTheMapMerges[aK];
        for (tItMM anIt = aMap.GT_Begin() ; anIt != aMap.GT_End() ; anIt++)
        {
            tMerge * aM = tMapMerge::GT_GetValOfIt(anIt);
            aM->SetOkForDelete();
        }
    }
    std::vector<tMerge *> aV2Del;
    for (int aK=0 ; aK<mTheNb ; aK++)
    {
        tMapMerge & aMap = mTheMapMerges[aK];
        for (tItMM anIt = aMap.GT_Begin() ; anIt != aMap.GT_End() ; anIt++)
        {
            tMerge * aM = tMapMerge::GT_GetValOfIt(anIt);
            if (aM->IsOk())
            {
               aV2Del.push_back(aM);
               aM->SetNoOk();
            }
        }
    }


    for (int aK=0 ; aK<int(aV2Del.size()) ; aK++)
        delete aV2Del[aK];


    mDeleted = true;
}


template <class Type>   void cStructMergeTieP<Type>::DoExport()
{
    AssertUnExported();
    mExportDone = true;

    for (int aK=0 ; aK<mTheNb ; aK++)
    {
        tMapMerge & aMap = mTheMapMerges[aK];

        for (tItMM anIt = aMap.GT_Begin() ; anIt != aMap.GT_End() ; anIt++)
        {
            tMerge * aM = tMapMerge::GT_GetValOfIt(anIt);
            if (aM->IsOk())
            {
               mLM.push_back(aM);
               aM->SetNoOk();
            }
        }
    }

    for (int aK=0 ; aK<mTheNb ; aK++)
    {
       mNbSomOfIm[aK] = 0;
    }
    
    for (typename std::list<tMerge *>::const_iterator itM=mLM.begin() ; itM!=mLM.end() ; itM++)
    {
        int aNbA = (*itM)->NbArc();
        while (int(mStatArc.size()) <= aNbA)
        {
           mStatArc.push_back(0);
        }
        mStatArc[aNbA] ++;
        for (int aKS=0 ; aKS<mTheNb ; aKS++)
        {
           if ((*itM)->IsInit(aKS))
           {
               const tVal &  aVal = (*itM)->GetVal(aKS);
               if(mNbSomOfIm[aKS] == 0)
               {
                   mEnvSup[aKS] = mEnvInf[aKS] = aVal;
               }
               mNbSomOfIm[aKS] ++;
               mEnvInf[aKS] = Inf(mEnvInf[aKS],aVal);
               mEnvSup[aKS] = Sup(mEnvSup[aKS],aVal);
           }
        }
    }
}


template <class Type>  const  std::list<Type *> & cStructMergeTieP<Type>::ListMerged() const
{
   AssertExported();
   return mLM;
}


template <class Type>  void cStructMergeTieP<Type>::AssertExported() const
{
   AssertUnDeleted();
   ELISE_ASSERT(mExportDone,"cStructMergeTieP<Type>::AssertExported");
}
template <class Type>  void cStructMergeTieP<Type>::AssertUnExported() const
{
   AssertUnDeleted();
   ELISE_ASSERT(!mExportDone,"cStructMergeTieP<Type>::AssertUnExported");
}
template <class Type>  void cStructMergeTieP<Type>::AssertUnDeleted() const
{
   ELISE_ASSERT(!mDeleted,"cStructMergeTieP<Type>::AssertUnExported");
}


   // ======================= Instanciation =========================
   // ======================= Instanciation =========================
   // ======================= Instanciation =========================


template  class cVarSizeMergeTieP<Pt2df>;
template  class cFixedSizeMergeTieP<3,Pt2df>;
template  class cFixedSizeMergeTieP<2,Pt2df>;



template  class cStructMergeTieP<cFixedSizeMergeTieP<3,Pt2df> >;
template  class cStructMergeTieP<cFixedSizeMergeTieP<2,Pt2df> >;
template  class cStructMergeTieP<cVarSizeMergeTieP<Pt2df> >;


template <class Type> double ChkMerge(const Type & aM)
{
    double aRes =  1/ (2.3+ aM.NbArc());
    for (int aK=0 ; aK<NbCamTest ; aK++)
    {
         if (aM.IsInit(aK))
         {
            Pt2df aP = aM.GetVal(aK);
            aRes = aRes + cos(aP.x) + aP.y / (10.0 + ElAbs(aP.y));
         }
         else
         {
             aRes += 1/(8.65 + aK);
         }
    }
    return aRes;
}

template <class Type> double ChkSomMerge(const std::list<Type  *> & aList,int & aNbArc)
{
    aNbArc=0;
    double aRes = 0;
    for (typename std::list<Type  *>::const_iterator iT=aList.begin(); iT!=aList.end() ; iT++)
    {
        aRes += ChkMerge(**iT);
        aNbArc += (*iT)->NbArc();
    }
    return aRes;
}


typedef  ElSom<Pt2df,int>       tSomTestTieP;
typedef  ElArc<Pt2df,int>       tArcTestTieP;
typedef  ElGraphe<Pt2df,int>    tGraphTestTieP;
typedef  ElSubGraphe<Pt2df,int> tSubGraphTestTieP;
typedef  ElArcIterator<Pt2df,int> tArtIterTestTieP;

/*
template <class AttrSom,class AttrArc>
         class ElSubGrapheFlag  : public ElSubGraphe<AttrSom,AttrArc>
{
   public :
        ElSubGrapheFlag(int aFlagS,int aFlagA) :  // -1 => No Flag
             mFlagS (aFlagS),
             mFlagA (aFlagA)
        {
        }
        bool   inS(ElSom<AttrSom,AttrArc>  & aSom )  {return (mFlagS<0) ||  aSom.flag_kth(mFlagS);}
        bool   inA(ElArc<AttrSom,AttrArc>  & anArc)  {return (mFlagA<0) || anArc.flag_kth(mFlagA);}
   private :
        int mFlagS;
        int mFlagA;

};
*/
typedef  ElSubGrapheFlag<Pt2df,int> tSubGraphTestTiepOr;

void OneTestNewMerge()
{
    static int aCpt=0;
    for (int aNb = 2 ; aNb < 500 ; aNb += 3)
    {
         aCpt++; 
         double aProbaPInt = NRrandom3();
         double aProbaArcGlob = NRrandom3();
         cFixedMergeStruct<NbCamTest,Pt2df> aFMS;
         cStructMergeTieP<cVarSizeMergeTieP<Pt2df> > aVSMT(NbCamTest);
         cStructMergeTieP<cFixedSizeMergeTieP<NbCamTest,Pt2df> > aFSMT(NbCamTest);

         tGraphTestTieP aGr;
         int aFlagOr = aGr.alloc_flag_arc();
         tSubGraphTestTiepOr aSubOr(-1,aFlagOr);
         std::map<Pt2df,tSomTestTieP *> aMapS;
         

         for (int aKP=0 ; aKP<aNb ;  aKP++)
         {
             double aProbaArc = aProbaArcGlob * NRrandom3();
             std::vector<Pt2df> aVP;
             std::vector<tSomTestTieP *> aVS;
             for (int aKC = 0 ; aKC<NbCamTest ; aKC++)
             {
                 Pt2df aP;
                 if (NRrandom3() < aProbaPInt)
                 {
                     aP = Pt2df(aKC,NRrandom3(aNb*NbCamTest));
                 }
                 else
                 {
                     aP = Pt2df(aKC,aKP) ; // Chaque point est different
                 }
                 aVP.push_back(aP);

                 if (aMapS[aP] == 0)
                    aMapS[aP] = &aGr.new_som(aP);
                 aVS.push_back(aMapS[aP]);
                 
             }
             for (int aKC1=0 ; aKC1<NbCamTest ; aKC1++)
             {
                 for (int aKC2=0 ; aKC2<NbCamTest ; aKC2++)
                 {
                      if ((NRrandom3() < aProbaArc) && (aKC1!=aKC2))
                      {
                          aVSMT.AddArc(aVP[aKC1],aKC1,aVP[aKC2],aKC2);
                          aFSMT.AddArc(aVP[aKC1],aKC1,aVP[aKC2],aKC2);
                          aFMS.AddArc(aVP[aKC1],aKC1,aVP[aKC2],aKC2);
                          
                          tSomTestTieP * aS1 = aVS[aKC1];
                          tSomTestTieP * aS2 = aVS[aKC2];
                          tArcTestTieP * anArc =  aGr.arc_s1s2(*aS1,*aS2);
                          if (anArc==0)
                             anArc = &aGr.add_arc(*aS1,*aS2,0);
                          anArc->flag_set_kth_true(aFlagOr);
                          anArc->attr()++;

                      }
                 }
             }
         }
         tSubGraphTestTieP aSub;
         ElPartition<tSomTestTieP * >  aPart;
         PartitionCC(aPart,aGr,aSub);
         int aNbCCOk=0;
         int aNbArcOK=0;
         for (int aKSet=0 ; aKSet <aPart.nb()  ; aKSet++)
         {
             ElSubFilo<tSomTestTieP *> aSet = aPart[aKSet];
             if (aSet.size() != 1)
             {
                  bool Ok=true;
                  std::vector<int> aCpt(NbCamTest,0);
                  int aNbArc = 0;
                  for (int aKSom=0 ; aKSom<aSet.size() ; aKSom++)
                  {
                       Pt2df aP = aSet[aKSom]->attr();
                       int aKC = round_ni(aP.x);
                       if (aCpt[aKC]) 
                           Ok= false;
                       aCpt[aKC]++;
                       for (tArtIterTestTieP itA= aSet[aKSom]->begin(aSub) ; itA.go_on() ; itA++)
                          aNbArc += (*itA).attr();
                  }
                  if (Ok) 
                  {
                     aNbArcOK += aNbArc;
                     aNbCCOk++;
                  }
             }
         }

         aVSMT.DoExport();
         aFSMT.DoExport();
         aFMS.DoExport();

         const std::list<cVarSizeMergeTieP<Pt2df>  *> &  aLVT = aVSMT.ListMerged();
         const std::list<cFixedSizeMergeTieP<NbCamTest,Pt2df> *> &  aLFT = aFSMT.ListMerged();
         const std::list<cFixedMergeTieP<NbCamTest,Pt2df> *> &  aLF = aFMS.ListMerged();

         ELISE_ASSERT(int(aLVT.size())==aNbCCOk,"Tie Point CC Check");

         int aNbA1,aNbA2,aNbA3;
         double aChk1 = ChkSomMerge(aLVT,aNbA1);
         double aChk2 = ChkSomMerge(aLFT,aNbA2);
         double aChk3 = ChkSomMerge( aLF,aNbA3);

         std::cout << aCpt << " ============== " << aLVT.size() << " " << aNbCCOk << " " << aNbArcOK << " " << " " << aNbA1 << "\n";
         if ( (ElAbs(aChk1-aChk2)> 1E-5) || (ElAbs(aChk1-aChk3)> 1E-5) )
         {
               std::cout << "HHHHH " <<   aChk1  << " " << aChk2  << " " << aChk3 << " " << "\n";
               ELISE_ASSERT(false,"Chk TieP");
         }

         if (aNbArcOK != aNbA1)
         {
             std::cout << " NB ARC : " << aNbArcOK << " " << aNbA1 << "\n";
             ELISE_ASSERT(aNbArcOK==aNbA1,"Tie Point CC Check");
         }


         // std::cout << "HHHHH " <<   ChkSomMerge(aLVT)  << " " << ChkSomMerge(aLFT)  << " " << ChkSomMerge(aLF)  << " " << "\n";
         aVSMT.Delete();
         aFSMT.Delete();
         aFMS.Delete();
    }
}




int TestNewMergeTieP_main(int argc,char ** argv)
{
    for (int aKT=0 ; aKT<1000000000; aKT++)
    {
        OneTestNewMerge();
        std::cout << "TestNewMergeTieP_main " << aKT << "\n";
        // getchar();
    }
    return EXIT_SUCCESS;
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
