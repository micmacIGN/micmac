#ifndef  _MMVII_MMError_H_
#define  _MMVII_MMError_H_

#include "MMVII_enums.h"

namespace MMVII
{


/** \file MMVII_Error.h
    \brief Error handling, basic for now

   Probably, sooner or later, I will have to change completely the error
  handling. However, waiting for that, we need to do something, and keep
  track in the file of the location of error detection.

*/


// It generates error that, to my best knowledge, should not. Waiting for better time where
// I will understand and solve them, they are tagged unresolved
#define  The_MMVII_DebugLevel_Unresoved     6  
// Ce ne sont pas de petite erreurs, mais des erruer couteuse a checker
#define  The_MMVII_DebugLevel_InternalError_tiny     5  
#define  The_MMVII_DebugLevel_InternalError_medium   4
// Ce ne sont pas de petite erreurs, mais des erruer couteuse a checker
#define  The_MMVII_DebugLevel_InternalError_strong   3
#define  The_MMVII_DebugLevel_UserError              2
#define  The_MMVII_DebugLevel_BenchError             1
#define  The_MMVII_DebugLevel_NoError                0


// extern int  The_MMVII_DebugLevel = The_MMVII_DebugLevel_InternalError_medium;
#define The_MMVII_DebugLevel The_MMVII_DebugLevel_InternalError_tiny
// #define The_MMVII_DebugLevel The_MMVII_DebugLevel_UserError

/**  The error handler can be change , so its a function Ptr of type PtrMMVII_Error_Handler,
     This can be used for example when in bench, we want to test the error handling
*/

typedef void (* PtrMMVII_Error_Handler) (const std::string & aType,const std::string &  aMes,const char * aFile,int aLine);


/**  The current handler */
extern PtrMMVII_Error_Handler MMVVI_Error;
/** Set a new handler */
void MMVII_SetErrorHandler(PtrMMVII_Error_Handler);
/** Restore default error handler */
void MMVII_RestoreDefaultHandle();


#define MMVII_INTERNAL_ERROR(aMes)\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}


// void MMVVI_Error(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine);

#define MMVII_INTERNAL_ASSERT_Unresolved(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_Unresoved ) && (!(aTest)))\
{MMVII_INTERNAL_ERROR(aMes);}

// Version not friendly but sure at 100% that has no impact on performance
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny )
#define MMVII_INTERNAL_ASSERT_tiny(aTest,aMes)\
 if (!(aTest)) {MMVII_INTERNAL_ERROR(aMes);}
#define AT_VECT(aVect,aK)  (aVect.at(aK))
#else
#define MMVII_INTERNAL_ASSERT_tiny(aTest,aMes) {}
#define AT_VECT(aVect,aK)  aVect[aK]
#endif


#define MMVII_INTERNAL_ASSERT_NotNul(aVal)  MMVII_INTERNAL_ASSERT_tiny((aVal!=0),"Unexpected null value")


#define MMVII_INTERNAL_ASSERT_medium(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_medium ) && (!(aTest)))\
{MMVII_INTERNAL_ERROR(aMes);}

#define MMVII_INTERNAL_ASSERT_strong(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_strong ) && (!(aTest)))\
{MMVII_INTERNAL_ERROR(aMes);}


#define MMVII_INTERNAL_ASSERT_bench(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_BenchError ) && (!(aTest)))\
{MMVII_INTERNAL_ERROR(aMes);}

void MMVII_UsersErrror(const eTyUEr &,const std::string & aMes);
#define MMVII_INTERNAL_ASSERT_User(aTest,aRef,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_UserError) && (!(aTest))) \
{  MMVII_UsersErrror(aRef,aMes);}
void MMVII_UnclasseUsEr(const std::string & aMes);


#define MMVII_INTERNAL_ASSERT_always(aTest,aMes)\
 if  (!(aTest))\
{MMVII_INTERNAL_ERROR(aMes);}



////====================  SPECIFIC ASSERTION ON NUMERICAL VALUE ================

#define  MMVII_ASSERT_INVERTIBLE_VALUE(VALUE)\
MMVII_INTERNAL_ASSERT_tiny(ValidInvertibleFloatValue(VALUE),"Non invertible value")

#define  MMVII_ASSERT_STRICT_POS_VALUE(VALUE)\
MMVII_INTERNAL_ASSERT_tiny(ValidStrictPosFloatValue(VALUE),"Non strict positive value")

#define  MMVII_ASSERT_POS_VALUE(VALUE)\
MMVII_INTERNAL_ASSERT_tiny(ValidPosFloatValue(VALUE),"Non positive value")

template<class T> void IgnoreUnused( const T& ) { }; /// To avoid some warning on TEMPORARILY unused variable 
void DoNothingWithIt(void *);  /// Used to avoid compiler optimization, make believe it can be used

#define BREAK_POINT(MSG)  {StdOut() << MSG << "; BREAK POINT at " << __LINE__ << " of " << __FILE__ << "\n";getchar();}

/* handling of eigen test on sucess of decomposition, by default it generate an error, btw this
can be temporarily supress if the use know what he does. For example see BenchLsqDegenerate
*/
bool EigenDoTestSuccess();  ///< do we test result of eigen operation
void OnEigenNoSucc(const  char * aMesg,int aLine,const char * aFile); ///< func to execute if not sucess
#define ON_EIGEN_NO_SUCC(MESG)  OnEigenNoSucc(MESG,__LINE__,__FILE__);

void PushErrorEigenErrorLevel(eLevelCheck aLevel);  ///< change behaving : no test, warning, error
void PopErrorEigenErrorLevel(); ///< restore previous behaving


class cMMVII_Warning
{
    int           mCpt;
    std::string   mMes;
    int           mLine;
    std::string   mFile;
  public :
    cMMVII_Warning(const std::string & aMes,int aLine,const std::string &  aFile);
    ~cMMVII_Warning();
    void Activate();
};

#define MMVII_WARGNING(MES) {static MMVII::cMMVII_Warning aWarn(MES,__LINE__,__FILE__); aWarn.Activate();}


/// A fake function, to stop momentarilly warnings about unused variable ...
template <class Type> void FakeUseIt(const Type &) {}
/** A function returning always false, use when we dont want to execute something want to compile it
 even with too "clever" compiler who would skip if (0) */
bool NeverHappens();


};

#endif  //  _MMVII_MMError_H_
