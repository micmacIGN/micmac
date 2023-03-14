#ifndef _MMVII_HEAP_H_
#define _MMVII_HEAP_H_

#include "MMVII_Error.h"
#include "MMVII_nums.h"

namespace MMVII
{

template <class Type,class Compare, class TParam> class cIndexedHeap ; //  indexed heap
template <class Type,class Compare> class cKBestValue;  // efficient memorization of K best values, use heap
template <class Type> class DefaultParamHeap; // usable as 3rd template param of cIndexedHeap when only non idexed heap is needed


// #define HEAP_NO_INDEX -1
static constexpr int HEAP_NO_INDEX = -1;

/**  Class to use for non indexable heap */
template <class Type> class DefaultParamHeap
{
     public :
        static void SetIndex(const Type &,tINT4 i) {}  ///< fixing indexe just do nothing
        static int  GetIndex(const Type &)  ///< we dont know what to do here => error
        {
             MMVII_INTERNAL_ERROR("No DefaultParamHeap::Index");
             return HEAP_NO_INDEX;
        }
};



/**  Class  for indexed heap; it's also a standard heap, implemented by a vector
      and respecting the heap constraint

           H(i) <= H(2i+1)
           H(i) <= H(2i+2)

     The indexation allow to modify at any moment the object contained in the heap while
     maintening the heap structure.  The "price" to pay is :

           *  the object must store an index, that contain it's place in the heap, it's indicated
              by the "SetIndex/GetIndex" method of TParam

           *  the user must warn the heap by a call to "UpDate" if the object has been modified in 
               way that may change the order

           * the heap will contain not object but access to object (typically pointers)

           * there is a slight over cost in run-time (but operations are still in log2(N))

     With default parametr for TParam, the heap is used not indexed.
*/

template <class Type,class Compare, class TParam=DefaultParamHeap<Type> > class cIndexedHeap 
{
    public :

        // =============== for any heap  ==========================

                /// constructor 
        cIndexedHeap(Compare inferior,tINT4 aCapa = 2) :
           mIsInferior    (inferior)
        {
		mElements.reserve(aCapa);
        }
            // ==== Accesors =============
        const std::vector<Type> & Elements() const {return mElements;}  ///< Vector of elements
        const Compare & Inferior() const {return mIsInferior;}  /// Comparator
        bool IsInferior(const Type & aV1,const Type & aV2) const {return mIsInferior(aV1,aV2);}


              /// Add a value in the heap
        void  Push(const Type & v)
        {
             mElements.push_back(v);   // push it at the end of the heap
             ResetIndex(Sz()-1);       // initialize its index
             HeapDown(Sz()-1);        // make a down movement to respect heap constraint
        }

             /// if empty return false; else  , pop the lowest in v if !=0  ,and return true
        bool PopPtr(Type * v) 
        {
             if (Sz() <= 0) 
                return false;
             if (v) 
             {
                  *v = mElements[0];  
             } 
             SetNoIndex(mElements[0]);
             mElements[0] = mElements[Sz()-1];  
             ResetIndex(0);
	     mElements.pop_back();
             HeapUp(0);
             return true;
        }

             /// call popPtr with no null value, most frequent use
        bool Pop(Type & v) 
        {
             return PopPtr(&v);
        }
             /// pop, don't get the values
        bool Pop() 
        {
            return PopPtr(0);
        }
             /// directly get the poped value, but require not empty ...
        Type  PopVal(const Type & aDef) 
        {
             Type aV = aDef;
             bool Ok= PopPtr(&aV);

             MMVII_INTERNAL_ASSERT_tiny(Ok,"Popval in enty heap");
             return aV;
        }

             /// return ptr to lowest elem if not empty, 0 else
        const Type * Lowest() const
        {
           if (IsEmpty()) return nullptr;
           return & mElements[0];
        }
             /// return ptr to lowest elem if not empty, 0 else
        Type * Lowest() 
        {
           if (IsEmpty()) return nullptr;
           return & mElements[0];
        }

             /// number of element
        int Sz() const {return  mElements.size();}
             /// Is the heap empty
        bool IsEmpty() const {return mElements.empty();}

             /// Is the heap empty
        void Clear() 
        { 
            for (auto & anElem : mElements)
             SetNoIndex(anElem);
            mElements.clear();
        }
        // =============== Specific to indexed heap ==========================

             /// Direct acces to heap belongness , use indexe
        bool IsInHeap(const Type & aV) const
        {
            return (TParam::GetIndex(aV)!=HEAP_NO_INDEX);
        }
             /// Update the position in the heap because the object may have been modified
        void UpDate(const Type & aV)
        {
           MMVII_INTERNAL_ASSERT_tiny(IsInHeap(aV),"update : Object not in heap");
           RestoreHeapStruct(TParam::GetIndex(aV));
        }

             /// Update if it is in the  heap, add if it is not
        void UpdateOrAdd(const Type & aV)
        {
            if (IsInHeap(aV))
               UpDate(aV);
            else
               Push(aV);
        }

        /// Take out an element that is in the heap (error if it is not)

        void TakeOut(const Type & aV)
        {
            MMVII_INTERNAL_ASSERT_tiny(IsInHeap(aV),"update : Object not in heap");

            int i = TParam::GetIndex(aV);

            mElements[i] =  mElements[Sz()-1] ;   // put last element at the free place
            ResetIndex(i);  //  update index to this new positio,
            SetNoIndex(aV);  // udate index of V as it is now out 
            mElements.pop_back();  // supress the element in  the queue
            RestoreHeapStruct(i);  // at the position i, we must rebuild the heap struct
        }



   private :
        cIndexedHeap(const cIndexedHeap<Type,Compare> &) = delete;

        // ===   Tree hierarchy, index take into account C-convention with first at 0   0->(1,2)  1->(3,4) 2->(5,6) ....

        static tINT4  Son(tINT4 i) { return   (i+1)/2-1;}   ///< (1,2) =>0  , (3,4) =>1 ,  (5,6) => 2 ....
        static tINT4  Father1(tINT4 i){ return   i*2+1;}    ///<  0 ->1   1-> 3    2 -> 5 ....
        static tINT4  Father2(tINT4 i){ return   i*2+2;}    ///<  0->2    1->4     2->6   ....

        ///  readjust index so that it contain it address in the heap
        void ResetIndex(tINT4 aK)
        {
              TParam::SetIndex(mElements[aK],aK);
        }
        ///  readjust index so that it contain the conventional "NO INDEX"
        void SetNoIndex(const Type & v)
        {
            TParam::SetIndex(v,HEAP_NO_INDEX);
        }

        /// swap values, udpate indexes, set ind1 to ind2
        void SwapAndSet(tINT4 & ind1,tINT4 ind2)
        {
             std::swap(mElements[ind1],mElements[ind2]);
             ResetIndex(ind1);  // restore index
             ResetIndex(ind2);  // restore index
             ind1 = ind2;
        }


       /// restore the heap structure if it has been potentially broken between the sumit and one of his ancester
        void HeapUp(tINT4 anIndex)
        {
            for(;;)
            {
                tINT4 indSmallestFather = anIndex;

                // compute the smallest of this and its two "father"
                {
                    tINT4 p1 = Father1(anIndex);
                    if ((p1 < Sz()) && mIsInferior(mElements[p1],mElements[indSmallestFather]))
                        indSmallestFather = p1;

                    tINT4 p2 = Father2(anIndex);
                    if ((p2 < Sz()) && mIsInferior(mElements[p2],mElements[indSmallestFather]))
                        indSmallestFather = p2;
                }

                // if this is the smallest the heap constraint is respected, no more things to do
                if (indSmallestFather == anIndex)
                   return;
                else // else we rebuild locally the heap structure
                {
                   // exchange this with the smallet and set to smallest for next iter
                    SwapAndSet(anIndex,indSmallestFather);
                }
            }
        }

       /// restore  heap structure potentially broken between the sumit and its "son", indicate if change was done
        bool HeapDown(tINT4 anIndex)
        {
            bool SthDone = false;
            // as long as it is inferior to its son
            while ((anIndex>0)&& mIsInferior(mElements[anIndex],mElements[Son(anIndex)]))
            {
                  // exchange with its son , and set to son for next iter
                  SwapAndSet(anIndex,Son(anIndex));
                  SthDone = true;
            }
            return SthDone;
        }

        /// restore the heap structure that may have been broken up or down "anIndex" 
        void RestoreHeapStruct(tINT4 anIndex)
        {
            if (anIndex< Sz())
            {
                if (!HeapDown(anIndex)) 
                   HeapUp(anIndex);
            }
        }
        //  ============  Data ===================
        Compare              mIsInferior;  ///< comparison function
	std::vector<Type>    mElements;         ///< vector of elements of the heap
};

/**  Class for memorizing the K Best value in a given subset , use heap to be efficient,
     suppose   Compare(V1,V2) is "V1<V2", store the K Biggest value.
*/

template <class Type,class Compare> class cKBestValue
{

    public :
      typedef cIndexedHeap<Type,Compare> tHeap ;

      /// Constructor , NbMax is the number of k best value
      cKBestValue(Compare isInferior,int aNbMax) :
          mHeap     (isInferior,aNbMax),
          mNbMax    (aNbMax)
      {
          Check();
      }

      /// number of element currently stored
      int Sz() const {return  mHeap.Sz();}
      /// vector  of element currently stored
      const std::vector<Type> & Elements() {return mHeap.Elements();}
      /// Heap used for implementation
      tHeap & Heap() {return  mHeap ;}

      /// Add a value, eventually supress id required an old value
      void Push(const Type& aV)
      {
          // if not enough element, just add new val in heap
          if (mHeap.Sz() < mNbMax)
          {
               mHeap.Push(aV);
          }
          // else, is > to lowest : add it and supresse lowest
          else if (mHeap.IsInferior(*mHeap.Lowest(),aV))
          {
              mHeap.Pop();
              mHeap.Push(aV);
          }
          // else do nothing, its not in the NbMax best Val
      }
      /// supress all value
      void clear() { mHeap.clear();}

      /// Change the number of K best val, eventually supress excedent values
      void SetNbMax(int aNbMax)
      {
          while (mHeap.Sz() > aNbMax) mHeap.pop();
          mNbMax = aNbMax;
          Check();
      }
      ///  reset and change capacity
      void ClearAndSetNbMax(int aNbMax)
      {
           mHeap.clear();
           mNbMax = aNbMax;
           Check();
      }

    private :
      /// Implementation would fail with no value
      void Check() const
      {
           MMVII_INTERNAL_ASSERT_tiny((mNbMax>=1),"Bad NbMax in Heap");
      }
      tHeap                mHeap ;   ///< Heap contaning the partially ordered value
      int                  mNbMax;   ///< Number of K best values we want, at most, to memorize
};



};

#endif //  _MMVII_HEAP_H_




