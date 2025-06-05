#ifndef  _MMVII_MEMORY_H_
#define  _MMVII_MEMORY_H_

#include "MMVII_Error.h"

namespace MMVII
{

/** \file MMVII_memory.h
    \brief Memory checking functionnality

    Contain support class for tracking the two main source of error :
        - bad desallocation
        - acces out of range

    For now the access out or range is done only for writing at desallocation
   time, maybe will add some access check for "tableau".

*/


void mem_raz(void * adr,int64_t nb);
#define MEM_RAZ(x,nb) mem_raz((void *)(x),(nb)*sizeof(*(x)))
template <class Type> void  MemCopy(Type * aDest,const Type * aSrce,size_t aNum) {memcpy(aDest,aSrce,sizeof(Type)*aNum);}

template <class T1,class T2>  void ConvertData(T1 * aD1,const T2 * aD2,size_t aNb)
{
    for (size_t aK=0 ; aK<aNb ; aK++)
        aD1[aK] = aD2[aK];
}
template <class T1>  void ConvertData(T1 * aD1,const T1 * aD2,size_t aNb)
{
        MemCopy(aD1,aD2,aNb);
}



/*
class  cMemState; // Memory state
class  cMemManager; // Allocator/desallocator tracking memory state
class  cMemCheck;   // Class calling cMemManager for allocation
*/


/// Class to register current state of memory

/**
   cMemState  memorize a summary of memory state container the number of
   object, the size allocacted and some majic number; it allow to check 
   if between 2 times, all what was allocated was desallocated

   Typical use
   cMemState aState = cMemManager::CurState();
   {
       .. do some stuff
   }
   Assert(aState==cMemManager::CurState());
   or
   cMemManager::CheckRestoration(aState);
*/
class cMemState
{
    public :
        friend class cMemManager;
        bool operator == (const cMemState &) const;
        int NbObjCreated() const;
        void SetCheckAtDestroy();
        ~cMemState(); ///< may call a check
    private :
        cMemState();
        int64_t  mCheckNb;
        int64_t  mCheckSize;
        int64_t  mCheckPtr;
        int64_t  mNbObjCreated;   ///< Number of allocation/desalloc
        bool     mDoCheckAtDestroy; ///< Sometime we need to do the check at the very end of the existence, this is 4 last object
};

/**
   This class, contains only  static funtion/object
   It group the functionnality that are usefull for
   checking allocation/desallocation.
   This service are activated only wijt debug over The_MMVII_DebugLevel_InternalError_medium
*/


class cMemManager
{
    public :
        ///   (1) Allocate and (2) modify("add") memory state and (3) write "majic" numbers at frontiers of allocatez zone  (of course allocate a bit more) 
        static void * Calloc(size_t nmemb, size_t size);
        /// (1) Free and (2) modify("substract") memory state and (3) check majic number are unmodified
        static void   Free(void * ptr);

        ///  Memorize the current memory state
        static const cMemState  CurState() ;
        ///  Return if the given state is equal to the current memory state
        static bool  IsOkCheckRestoration(const cMemState &) ;
        ///  Assert that the given state is equal to the current memory state
        static void  CheckRestoration(const cMemState &) ;

        // Par ex :    short * aPtr = cMemManager::Alloc<short>(aNb);
        template <class Type> static inline Type * Alloc(size_t nmemb)
        {
           return  static_cast<Type *> (cMemManager::Calloc(nmemb,sizeof(Type)));
        }

        template <class Type> static inline Type * Alloc(int aX0,int aX1)
        {
           return  Alloc<Type>(size_t(aX1-aX0)) - aX0;
	}

        template <class Type> static inline Type ** AllocMat(int aTx,int aTy)
        {
           Type ** aRes = Alloc<Type*>(aTy);
	   for (int aY=0 ; aY<aTy ; aY++)
               aRes[aY] = Alloc<Type>(aTx);
	   return aRes;
        }

        template <class Type> static void  FreeMat(Type ** aMat,int aTy)
	{
            for (int aY=0 ; aY<aTy ; aY++)
                Free(aMat[aY]);
            Free(aMat);
	}

        template <class Type> static bool Resize (Type *& aPtr,int aX0Prec,int & aSzMax,int aX0New,int aSzNew)
        {
           if (aSzNew > aSzMax)
           {
                Free(aPtr+aX0Prec); 
                aPtr = Alloc<Type> (aSzNew)-aX0New;
                aSzMax = aSzNew;
                return true;
           }
           else
               aPtr = aPtr + aX0Prec-aX0New;
           return false;
        }
        static void SetActiveMemoryCount(bool aVal);
        static bool IsActiveMemoryCount();
    private :

        static cMemState mState;
        static bool TheActiveMemoryCount;
};


/** cMemCountable to check that all object allocated are destroyed */

class cMemCountable
{
     public :
#if (The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_tiny)
         cMemCountable()  
         {
             mCpt = TheCptObj;
	     TheCptObj++;
#if (The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_micro)
             MMVII_INTERNAL_ASSERT_tiny((size_t)mCpt==TheVectFreeed.size(),"Internal check in cMemCountable::TheVectFreeed");
             MMVII_INTERNAL_ASSERT_tiny((size_t)mCpt==TheVectObjects.size(),"Internal check in cMemCountable::TheVectObjects");
             TheVectFreeed.push_back(false);
             TheVectObjects.push_back(this);
	     if (mCpt==TheNumObjTagged)
	     {
                 MMVII_INTERNAL_ERROR("Creation of object tagged for memory leak tracking");
	     }
#endif
              mActiveNbObj=   cMemManager::IsActiveMemoryCount();
              if (mActiveNbObj)
              {
                 TheNbObjLive++;
              }
         }
         cMemCountable(const cMemCountable &)  : cMemCountable () {}
#if (The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_micro)
         virtual ~cMemCountable()
#else
         ~cMemCountable()
#endif
         {
#if (The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_micro)
            TheVectFreeed.at(mCpt) = true;
            TheVectObjects.at(mCpt) = nullptr;
#endif
            if (mActiveNbObj)
            {
                 TheNbObjLive--;
            }
         }
         bool mActiveNbObj; 
	 int  mCpt;
#endif
         static int    NbObjLive();
	 static const std::vector<bool> *                  AdrTheVectFreeed();
	 static const std::vector<const cMemCountable*> *  AdrTheVectObjects();
	 static void  SetTaggedObjectAtCreation(int aNum);
      private :
         static int    TheNbObjLive;
         static int    TheCptObj;
	 static int    TheNumObjTagged;

#if (The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_micro)
	 static std::vector<bool>                  TheVectFreeed;
	 static std::vector<const cMemCountable*>  TheVectObjects;
         /// This method will allow dynamic typing
         virtual void Method4DynamicTyping() const {}
#endif
};


/**
    This classe redefine operators new and delate to check alloc / desalloc and (some) bad access.
    Allocation and desallocation is delegated to  cMemManager
*/

class  cMemCheck : public cMemCountable
{
      public :
         void * operator new    (size_t sz);
         void operator delete   (void * ptr) ;
         cMemCheck () : cMemCountable() {}
         cMemCheck(const cMemCheck &)  : cMemCheck () {}
       // to avoid use 
};

/**  some object have to live untill the end of the process, just before the verif is done in main
 *   appli destructor.
 */
class cObj2DelAtEnd  : public cMemCountable
{
        public :
                virtual ~cObj2DelAtEnd();
};



template<class Type> class cGestObjetEmpruntable
{
     public :
         cGestObjetEmpruntable() :
               mNbEmprunte (0)
         {
         }
         ~cGestObjetEmpruntable()
         {
             MMVII_INTERNAL_ASSERT_medium(mNbEmprunte==0,"Bad handling of cGestObjetEmpruntable");
             DeleteAllAndClear(mReserve);
         }

         Type * EmprunterOne()
         {
             mNbEmprunte++;
             if (!mReserve.empty())
             {
                  Type * aRes = mReserve.back();
                  mReserve.pop_back();
                  return aRes;
             }
             return new Type;
         }
         void RendreOne(Type * aVal)
         {
             mNbEmprunte--;
             mReserve.push_back(aVal);
         }
     private :
         int                 mNbEmprunte;
         std::vector<Type *> mReserve;
};


};

#endif  //  _MMVII_MEMORY_H_
