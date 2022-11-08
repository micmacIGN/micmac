#include "MMVII_SetITpl.h"
#include "MMVII_2Include_Serial_Tpl.h"
/*
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <functional>
*/



/** \file uti_set_sel.cpp
    \brief Implementation of selector & sets

   This file contains implementation of a set of class that allow to
  construct selectors, elemenatary one, or by boolean construction

*/

namespace MMVII
{
/* ======================================================= */
/*                                                         */
/*             cSetIntDyn:                                 */
/*                                                         */
/* ======================================================= */

cSetIntDyn::cSetIntDyn(size_t aNb) :
      mOccupied(aNb,false)
{
}

void cSetIntDyn::SortInd()
{
    std::sort(mVIndOcc.begin(),mVIndOcc.end());
}


void cSetIntDyn::MakeInvertIndex()
{
    SortInd();
    mVInvertInd.resize(mOccupied.size(),-1);

   for (size_t anInd=0 ; anInd<mVIndOcc.size() ; anInd++)
       mVInvertInd[mVIndOcc[anInd]] = anInd;
}

cSetIntDyn::cSetIntDyn(size_t aNb,const std::vector<size_t> & aVInd) :
     cSetIntDyn (aNb)
{
    for (const auto & anInd : aVInd)
        AddInd(anInd);
}


void cSetIntDyn::Clear()
{
   for (const auto & aInd : mVIndOcc)
       mOccupied.at(aInd) = false;
   mVIndOcc.clear();
}


void cSetIntDyn::AddInd(size_t aK)
{
      while (mOccupied.size() <= aK)
            mOccupied.push_back(false);

      AddIndFixe(aK);
}

/* ======================================================= */
/*                                                         */
/*             cRandKAmongN                                */
/*                                                         */
/* ======================================================= */

cRandKAmongN::cRandKAmongN(int aK,int aN) :
   mK (aK),
   mN (aN)
{
}

bool cRandKAmongN::GetNext()
{
  MMVII_INTERNAL_ASSERT_tiny(mN!=0,"cRandNParmiQ::GetNext");
  bool aRes =(RandUnif_0_1() * mN) <= mK;
  mN--;
  if (aRes)
      mK--;

   return aRes;

}


/* ======================================================= */
/*                                                         */
/*             cSetIExtension                              */
/*                                                         */
/* ======================================================= */

cSetIExtension::cSetIExtension()
{
}
cSetIExtension::cSetIExtension(const std::vector<size_t> & aVElems) :
   mElems (aVElems)
{
}

cSetIExtension   cSetIExtension::EmptySet() {return cSetIExtension();}

void cSetIExtension::AddElem(size_t anElem)
{
    mElems.push_back(anElem);
}


 
/* ======================================================= */
/*                                                         */
/*             GenRanQsubCardKAmongN                       */
/*                                                         */
/* ======================================================= */

static void MakeRandomSet(std::vector<size_t> & aRes,cSetIntDyn & aSet,int aK)
{
   while (aK!=0)
   {
        int aV = RandUnif_N(aSet.mOccupied.size());
        if (! aSet.mOccupied.at(aV))
        {
             aSet.AddIndFixe(aV);
             aK--;
        }
   }
   aRes = aSet.mVIndOcc;
   std::sort(aRes.begin(),aRes.end());
   aSet.Clear();
}

typedef std::pair<size_t,std::vector<size_t> > tHashedVI;


void GenRanQsubCardKAmongN(std::vector<cSetIExtension> & aRes,int aQ,int aK,int aN)
{


   MMVII_INTERNAL_ASSERT_tiny(aK<=aN,"GenRanQsubCardKAmongN K>N");
   tREAL8 aSzMax = rBinomialCoeff(aK,aN);


   // Curent case, we require more subset that possible in max case, just return all possible subset
   // Typicall  K=2 , N=10 , Q=1000
   if (aSzMax<=aQ)
   {
  // StdOut()  << "LL " << __LINE__ << " \n";
        aRes = SubKAmongN<cSetIExtension>(aK,aN);
        return ;
   }

   double aFact = 5.0;

   // intermediary case generate all subset and select a random subset of it
   if (aSzMax<= aFact* aQ)
   {
         // Generate all the set, that may be "slightly" too big
         std::vector<cSetIExtension> aSet = SubKAmongN<cSetIExtension>(aK,aN);
         aRes.clear();
         
         // and extract randomly Q among them
         cRandKAmongN aSel(aQ,aSet.size());
         for (int aK=0 ; aK<int(aSet.size()) ; aK++)
             if (aSel.GetNext())
                aRes.push_back(aSet[aK]);
        return ;
   }

   /*   More problematic case, all the subset would be too much,
        for example
          Typicall  K=8 , N=1000  , Q=10000  ,  C(8,N) = several billion, cannot generate it

        Algo :
           generate a "bit" too many subset,
           supresse duplicata
           finally  supress exceding number
   */
   double aProp = double(aQ) / double(aSzMax);
   // Empirically, this formula for the r
   double aRab = 1/(1- 0.6*aProp) -1;
   int aDelta = 20;

   while (1)
   {
        int aNbTest = std::max(aQ+aDelta,round_ni(aQ*(1+aRab)));
        cSetIntDyn aGlobSet(aN);
        std::vector<tHashedVI>  aVHVI;
        for (int aKTest=0 ; aKTest< aNbTest ; aKTest++)
        {
             std::vector<size_t> aSet;
             MakeRandomSet(aSet,aGlobSet,aK);
             size_t aHCode=  HashValue(aSet,true);
             
             aVHVI.push_back(tHashedVI(aHCode,aSet));
        }
        std::sort
        (
            aVHVI.begin(),
            aVHVI.end(),
            [](const auto & aP1,const auto & aP2) {return aP1.first < aP2.first;}
        );
        int aNbDiff = 0;
        for (int aKSet=0 ; aKSet< int(aVHVI.size()-1); aKSet++)
        {
            if (aVHVI[aKSet].first != aVHVI[aKSet+1].first)
            {
               aNbDiff++;
            }
        }

        cRandKAmongN aSel(aQ,aNbDiff);
        aRes.clear();
        for (int aKSet=0 ; aKSet< int(aVHVI.size()-1); aKSet++)
        {
            if (aVHVI[aKSet].first != aVHVI[aKSet+1].first)
            {
               if (aSel.GetNext())
                  aRes.push_back(cSetIExtension(aVHVI[aKSet].second));
            }
        }

        if (int(aRes.size())==aQ) return;
        aRab *= 1.5;
        aDelta += 20;
   }
}

void BenchRansSubset(int aQ,int aK,int aN)
{
     std::vector<cSetIExtension> aRes;
     GenRanQsubCardKAmongN(aRes,aQ,aK,aN);

     {  // Test number of subset, can get more than existing subset or required subset
        tU_INT4 aTheorCard = std::min(tU_INT4(aQ),iBinomialCoeff(aK,aN));
        //StdOut() << "Q=" << aQ  << " Max=" << BinomialCoeff(aK,aN) 
        //         << " K=" << aK  << " N=" << aN << " R=" << aRes.size() << "\n";
        MMVII_INTERNAL_ASSERT_bench(aTheorCard==(aRes.size()),"Bad number of subset in GenRanQsubCardKAmongN");
     }

     for (auto & aSubs : aRes)
     {
          auto & aVEl = aSubs.mElems;
          std::sort(aVEl.begin(),aVEl.end());
          MMVII_INTERNAL_ASSERT_bench(aK==int(aVEl.size()),"Bad subset size for BenchRansSubset");
          for (int aK=1 ; aK<int(aVEl.size()) ; aK++)
          {
              MMVII_INTERNAL_ASSERT_bench(aVEl.at(aK)   == aVEl.at(aK),"Not a subset of BenchRansSubset");
              MMVII_INTERNAL_ASSERT_bench(aVEl.at(aK-1) != aVEl.at(aK),"Not a subset of BenchRansSubset");
          }
     }
     std::sort
     (
         aRes.begin(),
         aRes.end(),
         [](const auto & aSet1,const auto & aSet2) {return aSet1.mElems < aSet2.mElems;}
     );
     for (int aK=1 ; aK<int(aRes.size()) ; aK++)
     {
         MMVII_INTERNAL_ASSERT_bench(aRes.at(aK).mElems == aRes.at(aK).mElems,"Not a subset of BenchRansSubset");
         MMVII_INTERNAL_ASSERT_bench(aRes.at(aK-1).mElems != aRes.at(aK).mElems,"Not a subset of BenchRansSubset");
     }
}


void BenchRansSubset(cParamExeBench & aParam)
{
    if (! aParam.NewBench("RandSubsets")) return;

    for (int aTime=0 ; aTime <2000 ; aTime++)
    {
        int aN = 1+ RandUnif_N(20);
        int aK = std::min(aN,int(RandUnif_N(4)));
        int aSzMax = iBinomialCoeff(aK,aN);

        int aQ =  aSzMax * (RandUnif_0_1()*2);
        BenchRansSubset(aQ,aK,aN);


    }
    aParam.EndBench();
    // MMVII_INTERNAL_ASSERT_bench(false,"xxxxxxxxx Unfinished bench for BenchRansSubset");

}

};//  namespace MMVII


