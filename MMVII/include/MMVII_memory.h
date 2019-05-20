#ifndef  _MMVII_MEMORY_H_
#define  _MMVII_MEMORY_H_

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
        int64_t  mNbObjCreated;
        bool     mDoCheckAtDestroy; ///< Sometime we need to do the check at the very end of the existence
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
        // Pour des raison inexpliquee Alloc<char *> ne semble pas fonctionner => contournement
/*
        template <class Type> static inline Type ** AllocP(size_t nmemb)
        {
           return  static_cast<Type **> (cMemManager::Calloc(nmemb,sizeof(Type *)));
        }
*/
    private :

        static cMemState mState;
};


/**
    This classe redefine l'operetor new and delate to checker alloc / desalloc and (some) bad access.
    Allocation and desallocation is delegated to  cMemManager
*/

class  cMemCheck
{
      public :
         void * operator new    (size_t sz);
         void operator delete   (void * ptr) ;
      private :

       // to avoid use 
         void * operator new []  (size_t sz);
         void operator delete [] (void * ptr) ;
};



///  Usefull, delete all object of the container
template <class Type> inline void DeleteAllAndClear(Type & aVal)
{
    for (auto it=aVal.begin() ; it!=aVal.end() ; it++)
        delete *it;
    aVal.clear();
}

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
