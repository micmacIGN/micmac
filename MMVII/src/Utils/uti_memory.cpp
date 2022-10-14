#include "include/MMVII_all.h"

namespace MMVII
{
//================= :: ==================

void mem_raz(void * adr,int64_t nb)
{
    memset(adr,0,nb);
}

void DoNothingWithIt(void *)
{
   // As expected ;-)
}

//================ cObj2DelAtEnd =======

cObj2DelAtEnd::~cObj2DelAtEnd()
{
        // StdOut() << "DELO2DEAE " << this << "\n";
}


//================ cMemState =======

cMemState::cMemState() :
   mCheckNb          (0),
   mCheckSize        (0),
   mCheckPtr         (0),
   mNbObjCreated     (0),
   mDoCheckAtDestroy (false)
{
}
      

void cMemState::SetCheckAtDestroy()
{
    mDoCheckAtDestroy = true;
}



cMemState::~cMemState()
{
   if (mDoCheckAtDestroy)
   {
      cMemManager::CheckRestoration(*this);
   }
}

bool cMemState::operator == (const cMemState & aSt2) const
{
    return       (mCheckNb    ==  aSt2.mCheckNb  )
           &&    (mCheckSize  ==  aSt2.mCheckSize)
           &&    (mCheckPtr   ==  aSt2.mCheckPtr )   
    ;
}

int cMemState::NbObjCreated() const
{
    return mNbObjCreated;
}


//================ cMemManager =======

cMemState  cMemManager::mState;


const cMemState  cMemManager::CurState() 
{
    return mState;
}
bool  cMemManager::IsOkCheckRestoration(const cMemState & aSt)  {return aSt==mState;}


void cMemManager::CheckRestoration(const cMemState & aState) 
{
   MMVII_INTERNAL_ASSERT_always
   (
        IsOkCheckRestoration(aState),
        "Allocates memory were not correctly freed"
   );
}


#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_medium)

// void CheckRestoration(const cMemState &) ;


/*
static const unsigned int  maj_32A = 0xF98A523F;
static const unsigned int  maj_32B = 0xC158E6B1;
static const unsigned int  maj_32C = 0xA57EF39D;
static const unsigned int  maj_32D = 0x7089AE99;
static const unsigned int  rubbish  = 0xFEDCBAEF;
int32_t
*/

// Two majic value at the same place to pass a message=> Use or not counting
static const int32_t  maj_32A1 = 0xF98A523F;  // Count Active
static const int32_t  maj_32A0 = 0xF98A524F;  // Count UnActivated

static const int32_t  maj_32B = 0xC158E6B1;
static const int32_t  maj_32C = 0xA57EF39D;
static const int32_t  maj_32D = 0x7089AE99;
static const int32_t  rubbish  = 0xFEDCBAEF;

static const unsigned char maj_octet = 0xE7;

// Because some object are never destroyed
bool cMemManager::TheActiveMemoryCount = true;
void cMemManager::SetActiveMemoryCount(bool aVal)
{
    TheActiveMemoryCount = aVal;
}
bool cMemManager::IsActiveMemoryCount()
{
    return TheActiveMemoryCount;
}

void * cMemManager::Calloc(size_t nmemb, size_t size)
{
     // !!! MPD !!! :  I dont know where the int conversion comes from, but surely it creates bugs with big allocation
     // size_t aNbOct = (int) (nmemb * size);
     size_t aNbOct =  (nmemb * size);
     // size_t aNb8 = (int) ((nmemb * size + 7)/8); // Assume 8 is ok as universal allignment constant
     size_t aNb8 =  ((nmemb * size + 7)/8); // Assume 8 is ok as universal allignment constant
     size_t aNb4 = aNb8 *2;

     void * v =  calloc(aNb8+3,8);
     MMVII_INTERNAL_ASSERT_always((v!=0),"Cannot allocate object");

     int64_t * aRes64 = static_cast<int64_t *>(v) + 2;
     int32_t * aRes32 = static_cast<int32_t *>(v) + 4;

     // Assure que par hasard, l'allocation ne soit pas faite de 0 qui
     // masquerait une non initialisation
     for (size_t i =0; i<aNb4; i++) aRes32[i] = 
         rubbish;

     aRes64 [-2]  = aNbOct;     // Taille de la zone en  octet
     aRes32[-2]   = TheActiveMemoryCount ? maj_32A1 : maj_32A0 ;   // majic nunmber
     aRes32[-1]   = maj_32B;   // majic nunmber
     aRes32[aNb4]  = maj_32C;   // majic number
     aRes32[aNb4+1] = maj_32D;
     {
          unsigned char * aRes1 = static_cast<unsigned char *> (v) + 16;
          // unsigned char * res1 = static_cast<unsigned char *> (v) + 16;
          // int1_t char * res1 = static_cast<unsigned char *> (v) + 16;
          size_t aNbOctAllign8 = 8 * aNb8;
          for (size_t i =   aNbOct ; i < aNbOctAllign8; i++)
          {
              aRes1[i] = maj_octet;
          }
     }

     if (TheActiveMemoryCount)
     {
         mState.mCheckNb ++;
         mState.mCheckSize +=  aNbOct;
         mState.mCheckPtr  ^=  reinterpret_cast<int64_t> (aRes64);
     }
     mState.mNbObjCreated ++;

     return aRes64;
}

void   cMemManager::Free(void * aPtr)
{
     int64_t * aPtr64 = static_cast<int64_t *>(aPtr) ;
     int32_t * aPtr32 = static_cast<int32_t *>(aPtr) ;

     size_t aNbOct =  aPtr64[-2];

     const char * aMesDebord = "cMemManager::Free write out memory detected";
     bool CountActivated =  (aPtr32[-2] == maj_32A1);
     if (!CountActivated)
        MMVII_INTERNAL_ASSERT_always((aPtr32[-2] == maj_32A0),aMesDebord);
     MMVII_INTERNAL_ASSERT_always((aPtr32[-1] == maj_32B),aMesDebord);

     size_t aNb8 = (int) ((aNbOct + 7)/8); // Assume 8 is ok as universal allignment constant
     size_t aNb4 = aNb8 *2;

     MMVII_INTERNAL_ASSERT_always((aPtr32[aNb4]   == maj_32C),aMesDebord);
     MMVII_INTERNAL_ASSERT_always((aPtr32[aNb4+1] == maj_32D),aMesDebord);

     {
          unsigned char * aRes1 = static_cast<unsigned char *> (aPtr) ;
          size_t aNbOctAllign8 = 8 * aNb8;
          for (size_t i =   aNbOct ; i < aNbOctAllign8; i++)
          {
              MMVII_INTERNAL_ASSERT_always(aRes1[i]==maj_octet,aMesDebord);
          }
     }

     if (CountActivated)
     {
        mState.mCheckNb --;
        mState.mCheckSize -=  aNbOct;
        mState.mCheckPtr  ^=  reinterpret_cast<int64_t> (aPtr64);
     }

     free(aPtr64-2);
}
#else  //  (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_medium)

void * cMemManager::Calloc(size_t nmemb, size_t size)
{
   return calloc(nmemb,size);
}

void   cMemManager::Free(void * aPtr)
{
   free(aPtr);
}
void cMemManager::SetActiveMemoryCount(bool aVal)
{
    // TheActiveMemoryCount = aVal;
}
#endif // (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_medium)


//================ cMemCheck =======

void * cMemCheck::operator new    (size_t aSz)
{
   return cMemManager::Calloc(1,aSz);
}

void cMemCheck::operator delete   (void * aPtr) 
{
    cMemManager::Free(aPtr);
}

int     cMemCheck::NbObjLive() {return TheNbObjLive;}

int     cMemCheck::TheNbObjLive=0;



};

