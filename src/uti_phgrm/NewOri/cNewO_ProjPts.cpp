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



/**************************************************************/
/*                                                            */
/*                       ::                                   */
/*                                                            */
/**************************************************************/



template <const int TheNb> void NOMerge_AddPackHom
                           (
                                cFixedMergeStruct<TheNb,Pt2dr> & aMap,
                                const ElPackHomologue & aPack,
                                const ElCamera & aCam1,int aK1,
                                const ElCamera & aCam2,int aK2
                           )
{
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
         aP1 =  ProjStenope(aCam1.F2toDirRayonL3(aP1));
         aP2 =  ProjStenope(aCam2.F2toDirRayonL3(aP2));
         aMap.AddArc(aP1,aK1,aP2,aK2);
    }
}



template <const int TheNb> void NOMerge_AddAllCams
                           (
                                cFixedMergeStruct<TheNb,Pt2dr> & aMap,
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


/**********************************************************************/
/*                                                                    */
/*                         cFixedMergeTieP                            */
/*                                                                    */
/**********************************************************************/

template <const int TheNbPts,class Type>   cFixedMergeTieP<TheNbPts,Type>::cFixedMergeTieP() :
           mOk     (true),
           mNbArc  (0)
{
    for (int aK=0 ; aK<TheNbPts; aK++)
    {
        mTabIsInit[aK] = false;
    }
}

template <const int TheNbPts,class Type>   
void cFixedMergeTieP<TheNbPts,Type>::FusionneInThis(cFixedMergeTieP<TheNbPts,Type> & anEl2,tMapMerge * Tabs)
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
            Tabs[aK][mVals[aK]] = this;
         }
     }
}

template <const int TheNbPts,class Type> 
   void cFixedMergeTieP<TheNbPts,Type>::AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2)
{
    AddSom(aV1,aK1);
    AddSom(aV2,aK2);
    mNbArc ++;
}

template <const int TheNbPts,class Type>
   void  cFixedMergeTieP<TheNbPts,Type>::AddSom(const Type & aV,int aK)
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
template <const int TheNbPts,class Type>   
int cFixedMergeTieP<TheNbPts,Type>::NbSom() const
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

template class  cFixedMergeTieP<2,Pt2dr>;
template class  cFixedMergeTieP<3,Pt2dr>;
template class  cFixedMergeTieP<2,Pt2df>;
template class  cFixedMergeTieP<3,Pt2df>;

/**********************************************************************/
/*                                                                    */
/*                         cFixedMergeStruct                            */
/*                                                                    */
/**********************************************************************/

template <const int TheNb,class Type> cFixedMergeStruct<TheNb,Type>::cFixedMergeStruct() :
    mExportDone (false),
    mDeleted    (false)
{
}

template <const int TheNb,class Type> void cFixedMergeStruct<TheNb,Type>::Delete()
{
    for (int aK=0 ; aK<TheNb ; aK++)
    {
        tMapMerge & aMap = mTheMaps[aK];
        for (tItMM anIt = aMap.begin() ; anIt != aMap.end() ; anIt++)
        {
            tMerge * aM = anIt->second;
            aM->SetOkForDelete();
        }
    }
    std::vector<tMerge *> aV2Del;
    for (int aK=0 ; aK<TheNb ; aK++)
    {
        tMapMerge & aMap = mTheMaps[aK];
        for (tItMM anIt = aMap.begin() ; anIt != aMap.end() ; anIt++)
        {
            tMerge * aM = anIt->second;
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


template <const int TheNb,class Type>   void cFixedMergeStruct<TheNb,Type>::DoExport()
{
    AssertUnExported();
    mExportDone = true;

    for (int aK=0 ; aK<TheNb ; aK++)
    {
        tMapMerge & aMap = mTheMaps[aK];
        for (tItMM anIt = aMap.begin() ; anIt != aMap.end() ; anIt++)
        {
            tMerge * aM = anIt->second;
            if (aM->IsOk())
            {
               mLM.push_back(aM);
               aM->SetNoOk();
            }
        }
    }

    for (int aK=0 ; aK<TheNb ; aK++)
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
        for (int aKS=0 ; aKS<TheNb ; aKS++)
        {
           if ((*itM)->IsInit(aKS))
           {
               const Type &  aVal = (*itM)->GetVal(aKS);
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



template <const int TheNb,class Type>
        void cFixedMergeStruct<TheNb,Type>::AddArc(const Type & aV1,int aK1,const Type & aV2,int aK2)
{

             ELISE_ASSERT((aK1!=aK2) && (aK1>=0) && (aK1<TheNb) && (aK2>=0) && (aK2<TheNb),"cFixedMergeStruct::AddArc Index illicit");

             AssertUnExported();
             tMapMerge & aMap1 = mTheMaps[aK1];
             tItMM anIt1  = aMap1.find(aV1);
             tMerge * aM1 = (anIt1 != aMap1.end()) ? anIt1->second : 0;

             tMapMerge & aMap2 = mTheMaps[aK2];
             tItMM anIt2  = aMap2.find(aV2);
             tMerge * aM2 =  (anIt2 != aMap2.end()) ? anIt2->second : 0;
             tMerge * aMerge = 0;


             if ((aM1==0) && (aM2==0))
             {
                 aMerge = new tMerge;
                 aMap1[aV1] = aMerge;
                 aMap2[aV2] = aMerge;
             }
             else if ((aM1!=0) && (aM2!=0))
             {
                  if (aM1==aM2) 
                  {   
                     aM1->IncrArc();
                     return;
                  }
                  aM1->FusionneInThis(*aM2,mTheMaps);
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
                 aMerge = mTheMaps[aK1][aV1] = aM2;
             }
             else
             {
                 aMerge =  mTheMaps[aK2][aV2] = aM1;
             }
             aMerge->AddArc(aV1,aK1,aV2,aK2);
}

template <const int TheNb,class Type>  const  std::list<cFixedMergeTieP<TheNb,Type> *> & cFixedMergeStruct<TheNb,Type>::ListMerged() const
{
   AssertExported();
   return mLM;
}



template <const int TheNb,class Type>  void cFixedMergeStruct<TheNb,Type>::AssertExported() const
{
   AssertUnDeleted();
   ELISE_ASSERT(mExportDone,"cFixedMergeStruct<TheNb,Type>::AssertExported");
}

template <const int TheNb,class Type>  void cFixedMergeStruct<TheNb,Type>::AssertUnExported() const
{
   AssertUnDeleted();
   ELISE_ASSERT(!mExportDone,"cFixedMergeStruct<TheNb,Type>::AssertUnExported");
}


template <const int TheNb,class Type>  void cFixedMergeStruct<TheNb,Type>::AssertUnDeleted() const
{
   ELISE_ASSERT(!mDeleted,"cFixedMergeStruct<TheNb,Type>::AssertUnExported");
}
template class  cFixedMergeStruct<2,Pt2df>;
template class  cFixedMergeStruct<3,Pt2df>;
template class  cFixedMergeStruct<2,Pt2dr>;
template class  cFixedMergeStruct<3,Pt2dr>;


/**********************************************************************************************/
/*                                                                                            */
/*                           BENCHS                                                           */
/*                                                                                            */
/**********************************************************************************************/

std::vector<int> CptArc(const std::list<cFixedMergeTieP<2,Pt2dr> *> aL,int & aTot)
{
    std::vector<int> aRes(100);
    aTot = 0;

    for (std::list<cFixedMergeTieP<2,Pt2dr> *>::const_iterator  itR=aL.begin() ; itR!=aL.end() ; itR++)
    {
        int aNb = (*itR)->NbArc();
        aTot += aNb;
        aRes[aNb] ++;
    }
     return aRes;
}

void AssertCptArc(cFixedMergeStruct<2,Pt2dr> & aMap ,int aNb0,int aNb1,int aNb2)
{
   aMap.DoExport();
   std::list<cFixedMergeTieP<2,Pt2dr> *>  aRes = aMap.ListMerged();
   int aTot;
   std::vector<int> aCpt = CptArc(aRes,aTot);

   if ((aNb0 != aCpt[0]) || (aNb1 != aCpt[1])   || (aNb2 != aCpt[2]) )
   {
      std::cout << "Nb0 " << aCpt[0] << " ; Nb1=" << aCpt[1] << " ; Nb2=" << aCpt[2] << "\n";
      ELISE_ASSERT(false,"AssertCptArc");
   }
}


void  NO_MergeTO_Test2_Basic(int aK)
{
    cFixedMergeStruct<2,Pt2dr> aMap;
    
    aMap.AddArc(Pt2dr(0,0),0,Pt2dr(1,1),1);

    if (aK==0)
    {
       AssertCptArc(aMap,0,1,0);  // 1 ARc
       return ;
    }

    aMap.AddArc(Pt2dr(1,1),1,Pt2dr(0,0),0);  
    if (aK==1)
    {
       AssertCptArc(aMap,0,0,1); // Arc Sym
       return ;
    }

    aMap.AddArc(Pt2dr(2,2),0,Pt2dr(3,3),1);
    if (aK==2)
    {
       AssertCptArc(aMap,0,1,1);  // 1 Sym + 1 ASym
       return ;
    }

    aMap.AddArc(Pt2dr(2,2),0,Pt2dr(4,4),1);  // Incoherent
    if (aK==3)
    {
       AssertCptArc(aMap,0,0,1);  //  1 Sym, 1 Inco
       return ;
    }
    return ;
}




void  NewOri_Info1Cple
(  
      const ElCamera & aCam1,
      const ElPackHomologue & aPack12,
      const ElCamera & aCam2,const ElPackHomologue & aPack21
)  
{
    cFixedMergeStruct<2,Pt2dr> aMap2;
    NOMerge_AddPackHom(aMap2,aPack12,aCam1,0,aCam2,1);
    NOMerge_AddPackHom(aMap2,aPack21,aCam2,1,aCam1,0);
    aMap2.DoExport();
    std::list<cFixedMergeTieP<2,Pt2dr> *>  aRes = aMap2.ListMerged();
    int aNb1=0;
    int aNb2=0;
    for (std::list<cFixedMergeTieP<2,Pt2dr> *>::const_iterator  itR=aRes.begin() ; itR!=aRes.end() ; itR++)
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



void  NO_MergeTO_Test4_Basic(int aK)
{
    cFixedMergeStruct<4,Pt2dr> aMap;

    if (aK==0)
    {
        aMap.DoExport();
        std::list<cFixedMergeTieP<4,Pt2dr> *>  aRes = aMap.ListMerged();
        ELISE_ASSERT(aRes.size()==0,"NO_MergeTO_Test4_Basic");
        return;
    }
    
    aMap.AddArc(Pt2dr(0,0),0,Pt2dr(1,1),1);
    aMap.AddArc(Pt2dr(2,2),2,Pt2dr(3,3),3);
    if (aK==1)
    {
        aMap.DoExport();
        std::list<cFixedMergeTieP<4,Pt2dr> *>  aRes = aMap.ListMerged();
        ELISE_ASSERT(aRes.size()==2,"NO_MergeTO_Test4_Basic");
        for 
        (
             std::list<cFixedMergeTieP<4,Pt2dr> *>::const_iterator itM=aRes.begin();
             itM!= aRes.end();
             itM++
        )
        {
            ELISE_ASSERT((*itM)->NbSom() == 2,"NO_MergeTO_Test4_Basic");
            ELISE_ASSERT((*itM)->NbArc() == 1,"NO_MergeTO_Test4_Basic");
        }
        return;
    }


    aMap.AddArc(Pt2dr(2,2),2,Pt2dr(1,1),1);

    aMap.DoExport();
    std::list<cFixedMergeTieP<4,Pt2dr> *>  aRes = aMap.ListMerged();

    ELISE_ASSERT(aRes.size()==1,"NO_MergeTO_Test4_Basic");
    ELISE_ASSERT((*aRes.begin())->NbArc()==3,"NO_MergeTO_Test4_Basic");
    ELISE_ASSERT((*aRes.begin())->NbSom()==4,"NO_MergeTO_Test4_Basic");
}


template<const int TheNb> void Bench_RandNewOri(int aNbTir,int aMaxVal)
{
    
    cFixedMergeStruct<TheNb,int> aMap;

    for (int aK=0 ; aK< aNbTir ; aK++)
    {
        int aInd1 = NRrandom3(TheNb);
        int aInd2 = NRrandom3(TheNb);
        if (aInd1 != aInd2)
        {
           aMap.AddArc(NRrandom3(aMaxVal),aInd1,NRrandom3(aMaxVal),aInd2);
        }
    }
}

void Bench_NewOri()
{

   for (int aK=0 ; aK<10000000 ; aK++)
   {
       std::cout << " Bench_NewOri " << aK << "\n";
       Bench_RandNewOri<3>(100,6);
       Bench_RandNewOri<4>(100,7);
   }
   std::cout << "All Fine New Ori\n";

   for (int aK=0 ; aK< 10 ; aK++)
   {
     NO_MergeTO_Test2_Basic(aK);
     NO_MergeTO_Test4_Basic(aK);
   }
}


void ForceInstanceNOMerge_AddAllCams()
{
    std::vector<cNewO_OneIm*> aVI;
    cFixedMergeStruct<2,Pt2dr> aMap;
    NOMerge_AddAllCams(aMap,aVI);
}


double NormPt2Ray(const Pt2dr & aP)
{
   return euclid(Pt3dr(aP.x,aP.y,1.0));
}


/***********************************************/
/*           FONCTIONS GLOBALES                */
/***********************************************/



ElPackHomologue ToStdPack(const tMergeLPackH * aMPack,bool PondInvNorm,double aPdsSingle)
{
    ElPackHomologue aRes;

    const tLMCplP & aLM = aMPack->ListMerged();
    for ( tLMCplP::const_iterator itC=aLM.begin() ; itC!=aLM.end() ; itC++)
    {
        const Pt2dr & aP0 = (*itC)->GetVal(0);
        const Pt2dr & aP1 = (*itC)->GetVal(1);

        double aPds = ((*itC)->NbArc() == 1) ? aPdsSingle : 1.0;

        if (PondInvNorm)
        {
             aPds /= NormPt2Ray(aP0) * NormPt2Ray(aP1);
        }

        ElCplePtsHomologues aCple(aP0,aP1,aPds);
        aRes.Cple_Add(aCple);
    }

    return aRes;
}


/*
struct cCdtPckR
{
      public :
        cCdtPckR(const Pt2dr& aP1,const Pt2dr & aP2,const double & aPds) :
            mP1       (aP1),
            mP2       (aP2),
            mPds0     (aPds),
            mPdsOccup (0.0),
            mDMin     (1e10),
            mTaken    (false)
         {
         }
         Pt2dr  mP1;
         Pt2dr  mP2;
         double mPds0;
         double mPdsOccup;
         double mDMin;
         bool   mTaken;
};



ElPackHomologue PackReduit(const ElPackHomologue & aPackIn,int aNbMaxInit,int aNbFin)
{

    //------------------------------------------------------------------------
    // A- 1- Preselrection purement aleatoire d'un nombre raisonnable depoints
    //------------------------------------------------------------------------

    std::vector<cCdtPckR> aVPres;

    RMat_Inertie aMat;

    {
       cRandNParmiQ aSelec(aNbMaxInit,aPackIn.size());
       for (ElPackHomologue::const_iterator itP=aPackIn.begin() ; itP!=aPackIn.end() ; itP++)
       {
            if (aSelec.GetNext())
            {
               cCdtPckR aPair(itP->P1(),itP->P2(),itP->Pds());
               aVPres.push_back(aPair);
               Pt2dr aP1 = aPair.mP1;
               aMat.add_pt_en_place(aP1.x,aP1.y);
            }
       }
    }
    aMat = aMat.normalize();
    int aNbSomTot = int(aVPres.size());

    double aSurfType  =  sqrt (aMat.s11()* aMat.s22() - ElSquare(aMat.s12()));
    double aDistType = sqrt(aSurfType/aNbSomTot);



    //------------------------------------------------------------------------
    // A-2   Calcul d'une fonction de deponderation  
    //------------------------------------------------------------------------

    for (int aKS1 = 0 ; aKS1 <aNbSomTot ; aKS1++)
    {
        for (int aKS2 = aKS1 ; aKS2 <aNbSomTot ; aKS2++)
        {
           // sqrt pour attenuer la ponderation
           double aDist = sqrt(dist48_euclid( aVPres[aKS1].mP1-aVPres[aKS2].mP1) );
           // double aDist =  dist48_euclid( aVPres[aKS1].mP1-aVPres[aKS2].mP1) ;
           // double  aDist=1;
           double aPds = 1 / (aDistType+aDist);
           aVPres[aKS1].mPdsOccup += aPds;
           aVPres[aKS2].mPdsOccup += aPds;
        }
    }
    for (int aKSom = 0 ; aKSom <aNbSomTot ; aKSom++)
    {
       aVPres[aKSom].mPdsOccup *= aVPres[aKSom].mPds0;
    }


    int aNbSomSel = ElMin(aNbSomTot,aNbFin);

    //------------------------------------------------------------------------
    // A-3  Calcul de aNbSomSel points biens repartis
    //------------------------------------------------------------------------

    ElTimer aChrono;
    ElPackHomologue aRes;
    for (int aKSel=0 ; aKSel<aNbSomSel ; aKSel++)
    {
         // Recherche du cdt le plus loin
         double aMaxDMin = 0;
         cCdtPckR * aBest = 0;
         for (int aKSom = 0 ; aKSom <aNbSomTot ; aKSom++)
         {
             cCdtPckR & aCdt = aVPres[aKSom];
             double aDist = aCdt.mDMin *  aCdt.mPdsOccup;
             if ((!aCdt.mTaken) &&  (aDist > aMaxDMin))
             {
                 aMaxDMin = aDist;
                 aBest = & aCdt;
             }
         }
         ELISE_ASSERT(aBest!=0,"cNewO_CombineCple");
         aBest->mTaken = true;
         for (int aKSom = 0 ; aKSom <aNbSomTot ; aKSom++)
         {
             cCdtPckR & aCdt = aVPres[aKSom];
             aCdt.mDMin = ElMin(aCdt.mDMin,dist48(aCdt.mP1-aBest->mP1));
         }
         ElCplePtsHomologues aCple(aBest->mP1,aBest->mP2,aBest->mPds0);
         aRes.Cple_Add(aCple);

         // mVCdtSel.push_back(aBest);
    }

    return aRes;
}
*/


class cCdtSelect
{
     public :
        cCdtSelect(int aNum,const double & aPds) :
            mNum      (aNum),
            mPds0     (aPds),
            mPdsOccup (0.0),
            mDMin     (1e10),
            mTaken    (false)
         {
         }
        int    mNum;
        double mPds0;
        double mPdsOccup;
        double mDMin;
        bool   mTaken;
};


template<class TypePt> class   cTplSelecPt
{
    public :
        cTplSelecPt(const std::vector<TypePt> & aVPres,const std::vector<double> * = 0);
        void InitPresel(int aNbPres);
        void CalcPond();
        const TypePt & PSel(const int & aK) const
        {
              return (*mVPts)[mVPresel[aK].mNum];
        }
        const TypePt & PSel(const cCdtSelect & aCdt) const
        {
              return (*mVPts)[aCdt.mNum];
        }

        void UpdateDistPtsAdd(const TypePt & aNewP);
        void SelectN(int aN,double aDistArret = -1);
        const std::vector<int>  & VSel() const {return mVSel;}

        double DistMinSelMoy() const;

    private :
        double Pds(int aK) {return mVPds ? (*mVPds)[aK] : 1.0;}
 
        double mDistType;
        const std::vector<TypePt> * mVPts;
        const std::vector<double> * mVPds;
        std::vector<cCdtSelect>     mVPresel;
        int                         mNbPts;
        int                         mNbPres;
        cResIPR                     mRes;
        std::vector<int> &          mVSel;
        
};

template<class TypePt> cTplSelecPt<TypePt>::cTplSelecPt(const std::vector<TypePt> & aVPts,const std::vector<double> * aVPds) :
     mVPts  (&aVPts),
     mVPds  (aVPds),
     mNbPts (mVPts->size()),
     mVSel  (mRes.mVSel)
{
}


template<class TypePt> void cTplSelecPt<TypePt>::InitPresel(int aNbPres)
{
    RMat_Inertie aMat;
    {
       cRandNParmiQ aSelec(aNbPres,mNbPts);
       for (int aK=0 ; aK<mNbPts ; aK++)
       {
            if (aSelec.GetNext())
            {
               const  TypePt & aPt = (*mVPts)[aK];
               aMat.add_pt_en_place(aPt.x,aPt.y);
               mVPresel.push_back(cCdtSelect(aK,Pds(aK)));
            }
       }
    }
    aMat = aMat.normalize();
    mNbPres = int(mVPresel.size());

    double aSurfType  =  sqrt (aMat.s11()* aMat.s22() - ElSquare(aMat.s12()));
    mDistType = sqrt(aSurfType/mNbPres);
}


template<class TypePt> void cTplSelecPt<TypePt>::CalcPond()
{
    for (int aKS1 = 0 ; aKS1 <mNbPres ; aKS1++)
    {
        const TypePt & aP1 = PSel(aKS1);
        for (int aKS2 = aKS1 ; aKS2 <mNbPres ; aKS2++)
        {
           double aDist = euclid( aP1-PSel(aKS2));
           // sqrt pour attenuer la ponderation
           double aPds = sqrt(1 / (mDistType+aDist));
           mVPresel[aKS1].mPdsOccup += aPds;
           mVPresel[aKS2].mPdsOccup += aPds;
        }
    }
    for (int aKSom = 0 ; aKSom<mNbPres ; aKSom++)
    {
       mVPresel[aKSom].mPdsOccup *= mVPresel[aKSom].mPds0;
    }
}



template<class TypePt> void cTplSelecPt<TypePt>::UpdateDistPtsAdd(const TypePt & aNewP)
{
   for (int aKSom = 0 ; aKSom <mNbPres ; aKSom++)
   {
       cCdtSelect & aCdt = mVPresel[aKSom];
       aCdt.mDMin = ElMin(aCdt.mDMin,euclid(PSel(aCdt)-aNewP));
   }
}


template<class TypePt> void cTplSelecPt<TypePt>::SelectN(int aTargetNbSel,double aDistArret)
{
    int aNbSomSel = ElMin(mNbPres,aTargetNbSel);
    mVSel.clear();
    bool Cont = true;

    for (int aKSel=0 ; (aKSel<aNbSomSel) && Cont ; aKSel++)
    {
         // Recherche du cdt le plus loin
         double aMaxDMin = 0;
         cCdtSelect * aBest = 0;
         for (int aKSom = 0 ; aKSom <mNbPres ; aKSom++)
         {
             cCdtSelect & aCdt = mVPresel[aKSom];
             double aDist = aCdt.mDMin *  aCdt.mPdsOccup;
             if ((!aCdt.mTaken) &&  (aDist > aMaxDMin))
             {
                 aMaxDMin = aDist;
                 aBest = & aCdt;
             }
         }

         ELISE_ASSERT(aBest!=0,"cNewO_CombineCple");
         aBest->mTaken = true;

         UpdateDistPtsAdd(PSel(*aBest));
         mVSel.push_back(aBest->mNum);

         //    aKSom>50 => pour que la dist puisse etre fiable;  aKSom%10 pour gagner du temps
         if ( (aDistArret>0) && (aKSel>50)  && ((aKSel%10)==0) )
         {
             Cont = (DistMinSelMoy() > aDistArret);
         }
    }
}

template<class TypePt> double cTplSelecPt<TypePt>::DistMinSelMoy() const
{
    double aSom = 0.0;
    for (int aKS=0 ; aKS<int(mVSel.size()) ; aKS++)
    {
          aSom += mVPresel[aKS].mDMin;
    }
    return aSom / mVSel.size();
}


template<class TypePt> std::vector<int>  TplIndPackReduit(const std::vector<TypePt> & aVPts,int aNbMaxInit,int aNbFin,const std::vector<double> * aVPds = 0)
{
    cTplSelecPt<TypePt> aSel(aVPts,aVPds);
    aSel.InitPresel(aNbMaxInit);
    aSel.CalcPond();
    aSel.SelectN(aNbFin);


    return aSel.VSel();
}


std::vector<int>  IndPackReduit(const std::vector<Pt2dr> & aV,int aNbMaxInit,int aNbFin)
{
   return TplIndPackReduit(aV,aNbMaxInit,aNbFin);
}

std::vector<int>  IndPackReduit(const std::vector<Pt2df> & aV,int aNbMaxInit,int aNbFin)
{
   return TplIndPackReduit(aV,aNbMaxInit,aNbFin);
}


void cNewO_OrInit2Im::TestNewSel(const ElPackHomologue & aPack)
{
     std::vector<Pt2dr> aVPts;
     for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
     {
         aVPts.push_back(itP->P1());
     }

     for (int aK=0 ; aK<int(aVPts.size()) ; aK++)
        mW->draw_circle_abs(ToW(aVPts[aK]),2.0,mW->pdisc()(P8COL::green));

     std::vector<int>  aVSel = TplIndPackReduit(aVPts,1500,500);
     for (int aK=0 ; aK<int(aVSel.size()) ; aK++)
     {
         Pt2dr aP = ToW(aVPts[aVSel[aK]]);
         mW->draw_circle_abs(aP,4.0,mW->pdisc()(P8COL::red));
         mW->draw_circle_abs(aP,6.0,mW->pdisc()(P8COL::red));
     }
     mW->clik_in();
}
   
ElPackHomologue PackReduit(const ElPackHomologue & aPackIn,int aNbMaxInit,int aNbFin)
{
   std::vector<Pt2dr> aVP1;
   std::vector<Pt2dr> aVP2;
   for (ElPackHomologue::const_iterator itP=aPackIn.begin() ; itP!=aPackIn.end() ; itP++)
   {
       aVP1.push_back(itP->P1());
       aVP2.push_back(itP->P2());
   }

   ElPackHomologue aRes;
   std::vector<int>  aVSel = TplIndPackReduit(aVP1,aNbMaxInit,aNbFin);
   for (int aK=0 ; aK<int(aVSel.size()) ; aK++)
   {
         ElCplePtsHomologues aCple(aVP1[aVSel[aK]],aVP2[aVSel[aK]]);
         aRes.Cple_Add(aCple);
   }

   return aRes;
}




ElPackHomologue PackReduit(const ElPackHomologue & aPackIn,int aNbFin)
{
    return PackReduit(aPackIn,aPackIn.size(),aNbFin);
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
