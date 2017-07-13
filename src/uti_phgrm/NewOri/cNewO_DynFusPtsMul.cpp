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


template <class Type,class TypeArc> void CheckCnx(const  cVarSizeMergeTieP<Type,TypeArc> & aVM,const std::string & aMes,int aK1 , int aK2)
{
static int aCpt=0; aCpt++;
std::cout << "CCx " << aMes << " " << aCpt << " K1K2 " << aK1 << " " << aK2 << " Adr=" << &aVM  << " KS=" <<aVM.VecInd() << "\n";
   const std::vector<Pt2di> &  aVE = aVM.Edges();
   for (int aKCple=0 ; aKCple<int(aVE.size()) ; aKCple++)
   {
std::cout << aVE[aKCple] << "\n";
       int aKCam1 = aVE[aKCple].x;
       int aKCam2 = aVE[aKCple].y;
       aVM.GetVal(aKCam1);
       aVM.GetVal(aKCam2);
   }
}
template <const int TheNbPts,class Type,class TypeArc> void CheckCnx(const  cFixedSizeMergeTieP<TheNbPts,Type,TypeArc> & aVM,const std::string & aMes,int aK1 , int aK2)
{
}

/**************************************************************************/
/*                                                                        */
/*             cCMT_NoVal / cCMT_U_INT1                                   */
/*                                                                        */
/**************************************************************************/
cCMT_NoVal::cCMT_NoVal() 
{
}


void cCMT_NoVal::Fusione(const cCMT_NoVal &)
{
}


   //------------------------------------
   
cCMT_U_INT1::cCMT_U_INT1() :
   mVal (0)
{
}
cCMT_U_INT1::cCMT_U_INT1(U_INT1 aVal) :
   mVal (aVal)
{
}

void cCMT_U_INT1::Fusione(const  cCMT_U_INT1 & aV2)
{
    mVal = ElMin(mVal,aV2.mVal);
}


/**************************************************************************/
/*                                                                        */
/*             cComMergeTieP                                              */
/*                                                                        */
/**************************************************************************/


template <class TypeArc> cComMergeTieP<TypeArc>::cComMergeTieP() :
  mOk (true),
  mNbArc (0)
{
}

template <class TypeArc> void cComMergeTieP<TypeArc>::MemoCnx(int aK1,int aK2,const TypeArc & aValArc)
{
   Pt2di aNewP(aK1,aK2);
   for (int aKE=0 ; aKE<int(mEdges.size()) ; aKE++)
   {
       if (mEdges[aKE] == aNewP)
       {
           mVecValArc[aKE].Fusione(aValArc);
           return;
       }
   }

   mEdges.push_back(Pt2di(aK1,aK2));
   mVecValArc.push_back(aValArc);
}

template <class TypeArc> const std::vector<Pt2di> &  cComMergeTieP<TypeArc>::Edges() const
{
   return mEdges;
}


template <class TypeArc> std::vector<Pt2di> &  cComMergeTieP<TypeArc>::NC_Edges() 
{
   return mEdges;
}



template <class TypeArc> const std::vector<TypeArc> &  cComMergeTieP<TypeArc>::ValArc() const
{
   return mVecValArc;
}

template <class TypeArc> std::vector<TypeArc> &  cComMergeTieP<TypeArc>::NC_ValArc() 
{
   return mVecValArc;
}



template <class TypeArc> void cComMergeTieP<TypeArc>::FusionneCnxInThis(const cComMergeTieP<TypeArc> & aC2)
{
    for (int aK=0 ; aK<int(aC2.mEdges.size()) ; aK++)
    {
        mEdges.push_back(aC2.mEdges[aK]);
        mVecValArc.push_back(aC2.mVecValArc[aK]);
    }
}


/**************************************************************************/
/*                                                                        */
/*        cVarSizeMergeTieP  / cFixedSizeMergeTieP                        */
/*                                                                        */
/**************************************************************************/

   // =================== Specif cVarSizeMergeTieP =================

template <class Type,class TypeArc> const std::vector<cPairIntType<Type> >  & cVarSizeMergeTieP<Type,TypeArc>::VecIT() const 
{
   return mVecIT;
}





   // ======================= Constructeurs =========================

template <class Type,class TypeArc> 
cVarSizeMergeTieP<Type,TypeArc>::cVarSizeMergeTieP() :
     cComMergeTieP<TypeArc>()
{
}

template <const int TheNbPts,class Type,class TypeArc>
cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::cFixedSizeMergeTieP() :
     cComMergeTieP<TypeArc>()
{
    for (int aK=0 ; aK<TheNbPts; aK++)
    {
        mTabIsInit[aK] = false;
    }

}
   // ======================= FixedSize =========================

template <class Type,class TypeArc>                    int  cVarSizeMergeTieP<Type,TypeArc>::FixedSize() {return -1;}
template <const int TheNbPts,class Type,class TypeArc> int cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::FixedSize() {return TheNbPts;}

   // ======================= Addsom =========================


template <class Type,class TypeArc> 
void  cVarSizeMergeTieP<Type,TypeArc>::AddSom(const Type & aV,int anInd)
{
     for (int aKI=0 ; aKI< int(mVecIT.size()) ; aKI++)
     {
         if ( mVecIT[aKI].mNum == anInd)
         {
             if (mVecIT[aKI].mVal != aV) this->mOk = false;
             return;
         }
     }

     mVecIT.push_back(tPairIT(anInd,aV));
     // mVecInd.push_back(anInd);
}

template <const int TheNbPts,class Type,class TypeArc>
   void  cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::AddSom(const Type & aV,int aK)
{
     if (mTabIsInit[aK])
     {
        if (mVals[aK] != aV)
        {
           this->mOk = false;
        }
     }
     else
     {
        mVals[aK] = aV;
        mTabIsInit[aK] = true;
     }
}


   // ======================= NbSom  =========================

template <class Type,class TypeArc> 
int  cVarSizeMergeTieP<Type,TypeArc>::NbSom() const
{
   return mVecIT.size();
}

template <const int TheNbPts,class Type,class TypeArc>   
int cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::NbSom() const
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

template <class TTieP,class Type,class TypeArc> void AddArcTieP(TTieP & aTieP,const Type & aV1,int aK1,const Type & aV2,int aK2,bool MemoEdge,const TypeArc & aValArc)
{
    aTieP.AddSom(aV1,aK1);
    aTieP.AddSom(aV2,aK2);
    aTieP.IncrArc();
    if (MemoEdge) 
    {
       aTieP.MemoCnx(aK1,aK2,aValArc);
       // CheckCnx(aTieP,"ADDARC",aK1,aK2);
    }
}

 
template <class Type,class TypeArc> 
void  cVarSizeMergeTieP<Type,TypeArc>::AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2,bool MemoEdge,const TypeArc & aValArc)
{
    AddArcTieP(*this,aV1,aK1,aV2,aK2,MemoEdge,aValArc);
}
template <const int TheNbPts,class Type,class TypeArc>
   void  cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2,bool MemoEdge,const TypeArc & aValArc)
{
    AddArcTieP(*this,aV1,aK1,aV2,aK2,MemoEdge,aValArc);
}


   // ======================= IsInit =========================

template <class Type,class TypeArc> 
bool  cVarSizeMergeTieP<Type,TypeArc>::IsInit(int anInd) const
{
     for (int aKI=0 ; aKI< int(mVecIT.size()) ; aKI++)
     {
         if ( mVecIT[aKI].mNum == anInd)
            return true;
     }
     return false;
}



template <const int TheNbPts,class Type,class TypeArc>
   bool  cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::IsInit(int aK) const
{
   return mTabIsInit[aK];
}

   // ======================= GetVal =========================

template <class Type,class TypeArc> 
const Type &    cVarSizeMergeTieP<Type,TypeArc>::GetVal(int anInd) const
{
     for (int aKI=0 ; aKI< int(mVecIT.size()) ; aKI++)
     {
         if ( mVecIT[aKI].mNum == anInd)
            return mVecIT[aKI].mVal;
     }
     std::cout << "VECIND " << mVecIT  << " Ind=" << anInd << "\n";
     ELISE_ASSERT(false,":::GetVal");
     return mVecIT[0].mVal;
}

template <const int TheNbPts,class Type,class TypeArc>
const Type & cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::GetVal(int aK) const
{
   return mVals[aK];
}


   // ======================= FusionneInThis =========================
       // void FusionneInThis(cFixedSizeMergeTieP<TheNbPts,Type> & anEl2,std::vector<tMapMerge> &  Tabs);

template <class Type,class TypeArc>   
void cVarSizeMergeTieP<Type,TypeArc>::FusionneInThis(cVarSizeMergeTieP<Type,TypeArc> & anEl2,std::vector<tMapMerge> &  Tabs)
{
     if ((!this->mOk) || (! anEl2.mOk))
     {
         this->mOk = anEl2.mOk = false;
         return;
     }
     this->mNbArc += anEl2.mNbArc;
     for (int aK2=0 ; aK2<int(anEl2.mVecIT.size()); aK2++)
     {
          int anInd2 = anEl2.mVecIT[aK2].mNum;
          const Type  & aV2 = anEl2.mVecIT[aK2].mVal;
          for (int aK1=0 ; aK1<int(mVecIT.size()) ; aK1++)
          {
              if (mVecIT[aK1].mNum == anInd2)
              {
                  // Ce cas ne devrait pas se produire, il doivent avoir ete fusionnes
                  ELISE_ASSERT(mVecIT[aK1].mVal != aV2,"cVarSizeMergeTieP");
                  this->mOk = anEl2.mOk = false;
                  return;
              }
          }
          mVecIT.push_back(tPairIT(anInd2,aV2));
          // mVecV.push_back(aV2);
          Tabs[anInd2].GT_SetVal(aV2,this);
     }
}


template <const int TheNbPts,class Type,class TypeArc>   
void cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::FusionneInThis(cFixedSizeMergeTieP<TheNbPts,Type,TypeArc> & anEl2,std::vector<tMapMerge> &  Tabs)
{

     if ((!this->mOk) || (! anEl2.mOk))
     {
         this->mOk = anEl2.mOk = false;
         return;
     }
     this->mNbArc += anEl2.mNbArc;
     for (int aK=0 ; aK<TheNbPts; aK++)
     {
         if ( mTabIsInit[aK] && anEl2.mTabIsInit[aK] )
         {
            // Ce cas ne devrait pas se produire, il doivent avoir ete fusionnes
            ELISE_ASSERT(mVals[aK]!= anEl2.mVals[aK],"cVarSizeMergeTieP");
            this->mOk = anEl2.mOk = false;
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

   // =================== CompileForExport =================

// template <class Type> I//
template <class Type,class TypeArc>  void   cVarSizeMergeTieP<Type,TypeArc>::CompileForExport() 
{
   std::sort(mVecIT.begin(),mVecIT.end());
}

template <const int TheNbPts,class Type,class TypeArc> void cFixedSizeMergeTieP<TheNbPts,Type,TypeArc>::CompileForExport() 
{
}

/**************************************************************************/
/*                                                                        */
/*                           cStructMergeTieP                             */
/*                                                                        */
/**************************************************************************/

template <class Type> cStructMergeTieP<Type>::cStructMergeTieP(int aNb,bool WithMemoEdges) :
    mTheNb      (aNb),
    mTheMapMerges    (aNb),
    mEnvInf     (aNb),
    mEnvSup     (aNb),
    mNbSomOfIm  (aNb,0),
    mStatArc    (),
    mExportDone (false),
    mDeleted    (false),
    mWithMemoEdges (WithMemoEdges)
{
     int aNb2 = Type::FixedSize();
     if (aNb2>=0)
     {
          ELISE_ASSERT(aNb2==aNb,"Incoh in StructMergeTieP<Type>::cStructMergeTie");
     }
}



template <class Type>
  void cStructMergeTieP<Type>::AddArc(const tVal & aV1,int aK1,const tVal & aV2,int aK2,const tArc & aValArc)
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
                     aM1->AddArc(aV1,aK1,aV2,aK2,mWithMemoEdges,aValArc);
/*
                     aM1->IncrArc();
                     if (mWithMemoEdges)
                     {
                        aM1->MemoCnx(aK1,aK2);
// CheckCnx(*aM1,"M1=m2",aK1,aK2);
// aM1->GetVal(aK1);
// aM1->GetVal(aK2);
                     }
*/
                     return;
                  }
                  aM1->FusionneInThis(*aM2,mTheMapMerges);
                  if (aM1->IsOk() && aM2->IsOk())
                  {
                     if (mWithMemoEdges) 
                        aM1->FusionneCnxInThis(*aM2);
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
             aMerge->AddArc(aV1,aK1,aV2,aK2,mWithMemoEdges,aValArc);
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
            aM->CompileForExport();
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


template  class cFixedSizeMergeTieP<3,Pt2df,cCMT_NoVal>;
template  class cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal>;
template  class cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal>;
template  class cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal>;
template  class cVarSizeMergeTieP<Pt2df,cCMT_NoVal>;
template  class cVarSizeMergeTieP<Pt2df,cCMT_U_INT1>;

template void NOMerge_AddAllCams<2>(cStructMergeTieP<cFixedSizeMergeTieP<2, Pt2dr, cCMT_NoVal> >& aMap, std::vector<cNewO_OneIm*> aVI);

template  class cStructMergeTieP<cFixedSizeMergeTieP<3,Pt2df,cCMT_NoVal> >;
template  class cStructMergeTieP<cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> >;
template  class cStructMergeTieP<cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> >;
template  class cStructMergeTieP<cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> >;
template  class cStructMergeTieP<cVarSizeMergeTieP<Pt2df,cCMT_NoVal> >;
template  class cStructMergeTieP<cVarSizeMergeTieP<Pt2df,cCMT_U_INT1> >;

template class cComMergeTieP<cCMT_NoVal>;
template class cComMergeTieP<cCMT_U_INT1>;

/**************************************************************************/
/*                                                                        */
/*                           Util                                         */
/*                                                                        */
/**************************************************************************/
template <const int TheNb,class Type,class TypeArc> void NOMerge_AddVect
                                           (
                                                cStructMergeTieP< cFixedSizeMergeTieP<TheNb,Type,TypeArc> > & aMap,
                                                const std::vector<Type> & aV1, int aK1,
                                                const std::vector<Type> & aV2, int aK2
                                           )
{
    TypeArc aValArc;
    ELISE_ASSERT(aV1.size()==aV2.size(),"NOMerge_AddVect");
    for ( int aKV=0 ; aKV<int(aV1.size()) ; aKV++)
    {
         aMap.AddArc(aV1[aKV],aK1,aV2[aKV],aK2,aValArc);
    }
}

void Merge3Pack
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr> & aVP2,
          std::vector<Pt2dr> & aVP3,
          int aSeuil,
          const std::vector<Pt2dr> & aV12,
          const std::vector<Pt2dr> & aV21,
          const std::vector<Pt2dr> & aV13,
          const std::vector<Pt2dr> & aV31,
          const std::vector<Pt2dr> & aV23,
          const std::vector<Pt2dr> & aV32
     )
{
    cStructMergeTieP< cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> > aMergeStr(3,false);

    NOMerge_AddVect(aMergeStr,aV12,0,aV21,1);
    NOMerge_AddVect(aMergeStr,aV13,0,aV31,2);
    NOMerge_AddVect(aMergeStr,aV23,1,aV32,2);

    aMergeStr.DoExport();

    const std::list<cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> *> & aLM = aMergeStr.ListMerged();
    for (std::list<cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> *>::const_iterator itM=aLM.begin(); itM!=aLM.end() ; itM++)
    {
        if (((*itM)->NbSom()==3) && ((*itM)->NbArc()>=aSeuil))
        {
            aVP1.push_back((*itM)->GetVal(0));
            aVP2.push_back((*itM)->GetVal(1));
            aVP3.push_back((*itM)->GetVal(2));
        }
    }
}





template <const int TheNb> void NOMerge_AddPackHom
                           (
                                cStructMergeTieP< cFixedSizeMergeTieP<TheNb,Pt2dr,cCMT_NoVal> > & aMap,
                                const ElPackHomologue & aPack,
                                const ElCamera * aCam1,int aK1,
                                const ElCamera * aCam2,int aK2
                           )
{
    cCMT_NoVal aValArc;
    for
    (
          ElPackHomologue::tCstIter itH=aPack.begin();
          itH !=aPack.end();
          itH++
    )
    {
         ElCplePtsHomologues aCple = itH->ToCple();
         Pt2dr aP1 =  aCple.P1();
         Pt2dr aP2 =  aCple.P2();
         if (aCam1)
         {
             aP1 =  ProjStenope(aCam1->F2toDirRayonL3(aP1));
         }
         if (aCam2)
         {
            aP2 =  ProjStenope(aCam2->F2toDirRayonL3(aP2));
         }
         aMap.AddArc(aP1,aK1,aP2,aK2,aValArc);
    }
}

template <const int TheNb> void NOMerge_AddPackHom
                           (
                                cStructMergeTieP< cFixedSizeMergeTieP<TheNb,Pt2dr,cCMT_NoVal> > & aMap,
                                const ElPackHomologue & aPack,
                                const ElCamera & aCam1,int aK1,
                                const ElCamera & aCam2,int aK2
                           )
{
    NOMerge_AddPackHom(aMap,aPack,&aCam1,aK1,&aCam2,aK2);
}

void Merge2Pack
     (
          std::vector<Pt2dr> & aVP1,
          std::vector<Pt2dr> & aVP2,
          int aSeuil,
          const ElPackHomologue & aPack1,
          const ElPackHomologue & aPack2
     )
{
    cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> > aMergeStr(2,false);
    const ElCamera  * aPtrCam = (const ElCamera *)NULL;
// NOMerge_AddPackHom
    NOMerge_AddPackHom(aMergeStr,aPack1,aPtrCam,0,aPtrCam,1);
    NOMerge_AddPackHom(aMergeStr,aPack2,aPtrCam,1,aPtrCam,0);

    aMergeStr.DoExport();

    const std::list<cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> *> & aLM = aMergeStr.ListMerged();
    for (std::list<cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> *>::const_iterator itM=aLM.begin(); itM!=aLM.end() ; itM++)
    {
        if (((*itM)->NbSom()==2) && ((*itM)->NbArc()>=aSeuil))
        {
            aVP1.push_back((*itM)->GetVal(0));
            aVP2.push_back((*itM)->GetVal(1));
        }
    }

}


template <const int TheNb> void NOMerge_AddAllCams
                           (
                                cStructMergeTieP< cFixedSizeMergeTieP<TheNb,Pt2dr,cCMT_NoVal> >  & aMap,
                                std::vector<cNewO_OneIm *> aVI
                           )
{
    ELISE_ASSERT(TheNb==int(aVI.size()),"MeregTieP All Cams");

    for (int aK1=0 ; aK1<TheNb ; aK1++)
    {
        for (int aK2=0 ; aK2<TheNb ; aK2++)
        {
            ElPackHomologue aLH12 = aVI[aK1]->NM().PackOfName(aVI[aK1]->Name(),aVI[aK2]->Name());
            NOMerge_AddPackHom(aMap,aLH12,*(aVI[aK1]->CS()),aK1,*(aVI[aK2]->CS()),aK2);
        }
    }
}

void  NewOri_Info1Cple
(
      const ElCamera & aCam1,
      const ElPackHomologue & aPack12,
      const ElCamera & aCam2,const ElPackHomologue & aPack21
)
{
    cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> > aMap2(2,false);

    NOMerge_AddPackHom(aMap2,aPack12,aCam1,0,aCam2,1);
    NOMerge_AddPackHom(aMap2,aPack21,aCam2,1,aCam1,0);
    aMap2.DoExport();
    std::list<cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> *>  aRes = aMap2.ListMerged();
    int aNb1=0;
    int aNb2=0;
    for (std::list<cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> *>::const_iterator  itR=aRes.begin() ; itR!=aRes.end() ; itR++)
    {
        if ((*itR)->NbArc() ==1)
        {
           aNb1++;
        }
        else if ((*itR)->NbArc() ==2)
        {
           aNb2++;
        }
        else
        {
           ELISE_ASSERT(false,"NO_MergeTO_Test2_0");
        }
    }
    std::cout << "INPUT " << aPack12.size() << " " << aPack21.size() << " Exp " << aNb1 << " " << aNb2 << "\n";
}

double NormPt2Ray(const Pt2dr & aP)
{
   return euclid(Pt3dr(aP.x,aP.y,1.0));
}


ElPackHomologue ToStdPack(const tMergeLPackH * aMPack,bool PondInvNorm,double aPdsSingle)
{
    ElPackHomologue aRes;

    const tLMCplP & aLM = aMPack->ListMerged();

    for ( tLMCplP::const_iterator itC=aLM.begin() ; itC!=aLM.end() ; itC++)
    {
        const Pt2dr & aP0 = (*itC)->GetVal(0);
        const Pt2dr & aP1 = (*itC)->GetVal(1);

        double aPds = ((*itC)->NbArc() == 1) ? aPdsSingle : 1.0;

        if (aPds > 0)
        {
           if (PondInvNorm)
           {
                aPds /= NormPt2Ray(aP0) * NormPt2Ray(aP1);
           }

           ElCplePtsHomologues aCple(aP0,aP1,aPds);
           aRes.Cple_Add(aCple);
       }
    }

    return aRes;
}






/**************************************************************************/
/*                                                                        */
/*                           Check                                        */
/*                                                                        */
/**************************************************************************/

class cChkMerge
{
     public :

         cChkMerge() : mRes(0.0) {}
         void AddNbArc(int aNbArc) {mRes +=  1/(2.319811 + aNbArc);}
         void AddSomAbs(int aK)    {mRes +=  1/(8.651786 + aK);}
         void AddSomIn(int aK,Pt2df aP) {mRes += cos(aP.x+aK) + aP.y / (10.0 + ElAbs(aP.y)) ; }

         double mRes;
};

template <class Type> double ChkMerge(const Type & aM,bool CheckEdges)
{
    cChkMerge aChk;
    if (CheckEdges)
    {
       ELISE_ASSERT(aM.NbArc()==int(aM.Edges().size()),"ChkMerge");
    }


    aChk.AddNbArc(aM.NbArc());
    for (int aK=0 ; aK<NbCamTest ; aK++)
    {
         if (aM.IsInit(aK))
           aChk.AddSomIn(aK,aM.GetVal(aK));
         else
           aChk.AddSomAbs(aK);
    }
    return aChk.mRes;

}

template <class Type> double ChkSomMerge(const std::list<Type  *> & aList,int & aNbArc,bool CheckEdges)
{
    aNbArc=0;
    double aRes = 0;
    for (typename std::list<Type  *>::const_iterator iT=aList.begin(); iT!=aList.end() ; iT++)
    {
        aRes += ChkMerge(**iT,CheckEdges);
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
         // std::cout << "================ " << aNb << " =============================\n";
         bool MemoArc = true;
         aCpt++; 
         double aProbaPInt = NRrandom3();
         double aProbaArcGlob = NRrandom3();
         cStructMergeTieP<cVarSizeMergeTieP<Pt2df,cCMT_NoVal> > aVSMT(NbCamTest,MemoArc);
         cStructMergeTieP<cFixedSizeMergeTieP<NbCamTest,Pt2df,cCMT_NoVal> > aFSMT(NbCamTest,MemoArc);

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
                          aVSMT.AddArc(aVP[aKC1],aKC1,aVP[aKC2],aKC2,cCMT_NoVal());
                          aFSMT.AddArc(aVP[aKC1],aKC1,aVP[aKC2],aKC2,cCMT_NoVal());
                          
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
         cChkMerge  aChkGr;

         for (int aKSet=0 ; aKSet <aPart.nb()  ; aKSet++)
         {
             ElSubFilo<tSomTestTieP *> aSet = aPart[aKSet];
             if (aSet.size() != 1)
             {
                  bool Ok=true;
                  std::vector<int> aCpt(NbCamTest,0);
                  std::vector<Pt2df> aVPts(NbCamTest,Pt2df(0,0));
                  int aNbArc = 0;
                  for (int aKSom=0 ; aKSom<aSet.size() ; aKSom++)
                  {
                       Pt2df aP = aSet[aKSom]->attr();
                       int aKC = round_ni(aP.x);
                       aVPts[aKC] = aP;
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
                     aChkGr.AddNbArc(aNbArc);
                     for (int aKC=0 ; aKC<NbCamTest ; aKC++)
                     {
                         if (aCpt[aKC])
                            aChkGr.AddSomIn(aKC,aVPts[aKC]);
                         else
                            aChkGr.AddSomAbs(aKC);
                     }
                  }
             }
         }

         aVSMT.DoExport();
         aFSMT.DoExport();

         const std::list<cVarSizeMergeTieP<Pt2df,cCMT_NoVal>  *> &  aLVT = aVSMT.ListMerged();
         const std::list<cFixedSizeMergeTieP<NbCamTest,Pt2df,cCMT_NoVal> *> &  aLFT = aFSMT.ListMerged();

         ELISE_ASSERT(int(aLVT.size())==aNbCCOk,"Tie Point CC Check");

         int aNbA1,aNbA2;
         double aChk1 = ChkSomMerge(aLVT,aNbA1,MemoArc);
         double aChk2 = ChkSomMerge(aLFT,aNbA2,MemoArc);

         // std::cout << aCpt << " ============== " << aLVT.size() << " " << aNbCCOk << " " << aNbArcOK << " " << " " << aNbA1 << "\n";
         if  (ElAbs(aChk1-aChk2)> 1E-5)  
         {
               std::cout << "HHHHH " <<   aChk1  << " " << aChk2  << " "  << "\n";
               ELISE_ASSERT(false,"Chk TieP");
         }
         if (ElAbs(aChk1-aChkGr.mRes)> 1e-5)
         {
               std::cout << "UUuuUUUi " <<   aChk1 << " " << aChkGr.mRes  << " " << "\n";
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
