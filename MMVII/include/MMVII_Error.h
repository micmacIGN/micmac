#ifndef  _MMVII_MMError_H_
#define  _MMVII_MMError_H_

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

/**  The error handler can be change , so its a function Ptr of type PtrMMVII_Error_Handler
*/

typedef void (* PtrMMVII_Error_Handler) (const std::string & aType,const std::string &  aMes,const char * aFile,int aLine);


/**  The current handler */
extern PtrMMVII_Error_Handler MMVVI_Error;
/** Set a new handler */
void MMVII_SetErrorHandler(PtrMMVII_Error_Handler);
/** Restore default error handler */
void MMVII_RestoreDefaultHandle();




// void MMVVI_Error(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine);

#define MMVII_INTERNAL_ASSERT_Unresolved(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_Unresoved ) && (!(aTest)))\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}

#define MMVII_INTERNAL_ASSERT_tiny(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny ) && (!(aTest)))\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}

#define MMVII_INTERNAL_ASSERT_medium(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_medium ) && (!(aTest)))\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}

#define MMVII_INTERNAL_ASSERT_strong(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_strong ) && (!(aTest)))\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}


#define MMVII_INTERNAL_ASSERT_bench(aTest,aMes)\
 if ((The_MMVII_DebugLevel>=The_MMVII_DebugLevel_BenchError ) && (!(aTest)))\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}

void MMVII_UsersErrror(const eTyUEr &,const std::string & aMes);
#define MMVII_INTERNAL_ASSERT_user(aRef,aMes)\
 if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_UserError ) \
{  MMVII_UsersErrror(aRef,aMes);}



#define MMVII_INTERNAL_ASSERT_always(aTest,aMes)\
 if  (!(aTest))\
{ MMVVI_Error("Internal Error",aMes,__FILE__,__LINE__);}


template<class T> void IgnoreUnused( const T& ) { };

};

#endif  //  _MMVII_MMError_H_
