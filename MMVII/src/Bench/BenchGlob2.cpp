#include "include/MMVII_all.h"
#include "include/MMVII_TplHeap.h"

namespace MMVII
{


class  cCmpPt2drOnX
{
     public :
         bool operator ()(const cPt2dr & aP1,const cPt2dr & aP2) const  {return aP1.x() < aP2.x();}
};


class  cCmpPtrPt2drOnX
{
     public :
         bool operator ()(const cPt2dr * aP1,const cPt2dr * aP2)  const {return aP1->x() < aP2->x();}
};

class cIndexPtrPt2drOnY
{
     public :
        static void SetIndex(cPt2dr * aPts,tINT4 i) { aPts->y() = i;}
        static int  GetIndex(const cPt2dr * aPts) { return round_ni(aPts->y()); }
};



void Bench_Heap(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Heap")) return;

     for (int aSz=10 ; aSz<50 ; aSz+=6)
     {
          // Generate a vector (i,i) in random order
          std::vector<cPt2dr>  aVPts;
          for (int aK=0 ; aK<aSz ; aK++)
              aVPts.push_back(cPt2dr(aK,aK));
           RandomOrder(aVPts);

           cCmpPt2drOnX aCmp;

          // 1  ===========   Check "basic" heap, without indexation  ===============
          {
              //   -- 1.1  push the points in the heap
              cIndexedHeap<cPt2dr,cCmpPt2drOnX>  aHeap(aCmp);
              for (const auto & aPt : aVPts)
                  aHeap.Push(aPt);

              //   -- 1.2   pop the point and check they are in good order
              int aIndex=0;
              while (! aHeap.IsEmpty())
              {
                  cPt2dr aPt;
                  bool  isOk = aHeap.Pop(aPt);
                  MMVII_INTERNAL_ASSERT_bench(isOk,"Heap pop");
                  MMVII_INTERNAL_ASSERT_bench(aPt==cPt2dr(aIndex,aIndex),"Heap bad value");
                  aIndex ++;
              }
              MMVII_INTERNAL_ASSERT_bench(aIndex==aSz,"Heap bad numbers");
          }

          // 2  ===========   Now Check heap, with indexation, object can be modified  ===============

          //   -- 2.1  push the points in the heap
          {
              cCmpPtrPt2drOnX aCmpPtr;
              cIndexedHeap<cPt2dr*,cCmpPtrPt2drOnX,cIndexPtrPt2drOnY>  aHeapPtr(aCmpPtr);
              for (auto & aPt : aVPts)
                  aHeapPtr.Push(&aPt);

              //   -- 2.2 Now modify the points and inform the heap of this modification; take out even x
              for (auto & aPt : aVPts)
              {
                    aPt.x() *= -1;  // modify the point
                    aHeapPtr.UpDate(&aPt);  // send the message to heap that structure is to be recovered
                    MMVII_INTERNAL_ASSERT_bench(aHeapPtr.IsInHeap(&aPt),"Heap is in");
                    if ( (int (aPt.x()))%2 !=0)  // take out even
                    {
                       aHeapPtr.TakeOut(&aPt);
                       MMVII_INTERNAL_ASSERT_bench(!aHeapPtr.IsInHeap(&aPt),"Heap is out");
                    }
              }

              //   -- 2.3   pop the point and check they are in good order
              int aIndex= aSz;
              while (! aHeapPtr.IsEmpty())
              {
                  aIndex -=2;
                  cPt2dr * aPt=nullptr;
                  const cPt2dr * aPt2 = *(aHeapPtr.Lowest()) ;
                  bool  isOk = aHeapPtr.Pop(aPt);
                  MMVII_INTERNAL_ASSERT_bench(aPt==aPt2,"Heap pop/lowest");
                  MMVII_INTERNAL_ASSERT_bench(isOk,"Heap pop");
                  MMVII_INTERNAL_ASSERT_bench(aPt->x()==-aIndex,"Heap pop");
              }
              MMVII_INTERNAL_ASSERT_bench(aHeapPtr.Lowest()==0,"Bad heap lowest");
              MMVII_INTERNAL_ASSERT_bench(aIndex==0,"Bad size check in heap");
          }
          // 3  ===========   check K Best Set  ===============
          {
             int aNbMax = 2+RandUnif_N(aSz/2);
             cKBestValue<cPt2dr,cCmpPt2drOnX>  aKBV(aCmp,aNbMax);

             // select the K Best elem
             for (const auto & aPt : aVPts)
                 aKBV.Push(aPt);

              MMVII_INTERNAL_ASSERT_bench(aKBV.Sz()==aNbMax,"Bad sz KBV");
              int aK= aNbMax;
              while (aKBV.Sz())
              {
                  aK--;
                  cPt2dr aP (10000,999);
                  aKBV.Heap().Pop(aP);
                  MMVII_INTERNAL_ASSERT_bench((aP.x()+aK)==0,"Bad KBV");  // !! => value are negative due to change sign above ...
              }
              MMVII_INTERNAL_ASSERT_bench(aK==0,"Bad KBV");

          }
     }
     aParam.EndBench();
}

#if (0)




template <class TyVal,class TyPrio> class cTplPrioByOther
{
     public :
          cTplPrioByOther(const TyVal & aVal,const TyPrio & aPrio) :
               mVal (aVal),
               mPrio (aPrio)
          {
          }
          TyVal mVal;
          TyPrio mPrio;
};

template <class TyVal,class TyPrio>   class  cCmpSupPBO
{
    public :
       bool operator () (const cTplPrioByOther<TyVal,TyPrio> & aS1, const cTplPrioByOther<TyVal,TyPrio>  & aS2)
       {
           return aS1.mPrio > aS2.mPrio;
       }

};
#endif


};





