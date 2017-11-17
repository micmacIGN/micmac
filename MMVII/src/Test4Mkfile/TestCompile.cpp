#include "include/MMVII_all.h"


/*
class cTestUnikPtr;
extern cTestUnikPtr * AllocUnikPtr();

void DeleteUnikPtr(cTestUnikPtr *);
typedef void (*tDeleteUnikPtr)(cTestUnikPtr *);

void  TestCompile()
{
    // std::unique_ptr<cTestUnikPtr,tDeleteUnikPtr> aP (AllocUnikPtr(),DeleteUnikPtr);
    std::unique_ptr<cTestUnikPtr, void (*)(cTestUnikPtr *)> aP (AllocUnikPtr(),DeleteUnikPtr);
}
*/

/*
class cTestUnikPtr
{
    public :
      virtual ~cTestUnikPtr() ;
}      virtual void F() = 0;
*/




