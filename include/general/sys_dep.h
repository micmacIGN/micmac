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



/*
    This file contains the macro and type definitions necessary
    for portability.
*/

#ifndef _ELISE_SYS_DEP_H
#define _ELISE_SYS_DEP_H

#include "general/CMake_defines.h"

#ifndef ELISE_unix
    #ifdef _WIN32
        #define USE_NOYAU 0
        #define ELISE_unix 0
        #define ELISE_windows 1
        #define ELISE_MacOs 0
        #define ELISE_POSIX 0
        #ifdef __MINGW__
            #define ELISE_MinGW 1
        #else
            #define ELISE_MinGW 0
        #endif
    #elif __APPLE__
        #define USE_NOYAU 0
        #define ELISE_unix 0
        #define ELISE_MacOs 1
        #define ELISE_windows 0
        #define ELISE_MinGW 0
        #define ELISE_POSIX 1
    #else
        #define USE_NOYAU 0
        #define ELISE_unix 1
        #define ELISE_MacOs 0
        #define ELISE_windows 0
        #define ELISE_MinGW 0
        #define ELISE_POSIX 1
    #endif
#endif

#include "GpGpu/GpGpu_BuildOptions.h"

// Only for g++ 2.7.2.1 on alpha
#define BUG_CPP_Fclose 0
#define ElBugHomeMPD 1


#define Compiler_Turbo_4_5     0
#define Compiler_Visual_5_0    0
#define Compiler_Visual_6_0    0
#define Compiler_Gpp2_7_2      0
#define GPP3etPlus             1
#define Compiler_Visual_7_0    0

// A cause de  qq pb sur bleriot
#define MACHINE_BLERIOT        0



#define SUN_WS5 0
#define SUN_WS6 0



// Pour l'instant : Unix=>X11, Win => pas de visu

#ifndef NO_X11
    #ifndef ELISE_X11
        #define ELISE_X11  (ELISE_unix | ELISE_MacOs)
    #endif
#endif
/*
*/

#define ELISE_NO_VIDEO  (! ELISE_X11)



#define ELISE_VW_W95NT_API 0
#define ELISE_WXW      0


#define ElUseNameSpace      1

//===================================================

#if ELISE_unix
    #define SYS_MV "mv"
    #define SYS_RM "\\rm"   // MODID MPD CAR rm ne fonctionne pas si il a ete redefini par alias !!
    #define SYS_CP "cp"
    #define SYS_CAT "cat"
    #define ELISE_CAR_DIR  '/'
    #define ELISE_Current_DIR  "./"
    #define ELISE_STR_DIR "/"
    // the character separating directories in PATH environment variable
    #define ELISE_CAR_ENV ':'
#endif

#if ELISE_MacOs
    #define SYS_MV "mv"
    #define SYS_RM "rm"
    #define SYS_CP "cp"
    #define SYS_CAT "cat"
    #define ELISE_CAR_DIR  '/'
    #define ELISE_Current_DIR  "./"
    #define ELISE_STR_DIR "/"
    #define ELISE_CAR_ENV ':'
#endif

#if ELISE_windows
	#define _MSC_VER_2015 1900
	#define _MSC_VER_2013 1800
	#define _MSC_VER_2012 1700
	#define _MSC_VER_2010 1600

    #define SYS_MV "move"
    #define SYS_RM "del"
    #define SYS_CP "copy"
    #define SYS_CAT "type"
    #define ELISE_CAR_DIR  '/'
    #define ELISE_Current_DIR  "./"

    #define ELISE_STR_DIR "/"
    // the character separating directories in PATH environment variable
    #define ELISE_CAR_ENV ';'
    #if !ELISE_MinGW
		#include <float.h>
		#define std_isnan _isnan
		#define std_isinf isinf
		#define isinf(x) (!_finite(x))
		typedef unsigned int uint;
	#else
		#include <cmath>
		#define std_isnan std::isnan
		#define std_isinf std::isinf
	#endif

	#if _MSC_VER<_MSC_VER_2013 && !ELISE_MinGW
		double round( double aX );
	#endif
#else
    #include <cmath>
    #define std_isnan std::isnan
    #define std_isinf std::isinf
#endif

template <class Type> bool BadNumber(const Type & aVal) {return (std_isnan(aVal)||std_isinf(aVal));}

#if __cplusplus > 199711L | (_MSC_VER == 1800 & CPP11THREAD_NOBOOSTTHREAD == 1)
    #define std_unique_ptr std::unique_ptr
    #define NULLPTR nullptr
    #define CPPX11
    #ifndef     __CUDACC__
        #define    NOCUDA_X11
    #endif
#else // under c++11
    #define std_unique_ptr std::auto_ptr
    #define NULLPTR NULL
#endif

#if Compiler_Gpp2_7_2   // =========
    #define ElTyName typename
#elif(GPP3etPlus || Compiler_Visual_7_0)      // =========
    #define ElTyName typename
#else                  // =========
    #define ElTyName
#endif

#define ElTemplateInstantiation 1

// Apparemment MSW est assez restrictif sur l'emploi du typename
#if ( ELISE_windows & !ELISE_MinGW )
    #define ElTypeName_NotMSW
    #define  ClassFriend
#else
    #define ElTypeName_NotMSW typename
    #define  ClassFriend class
#endif



#if ElUseNameSpace
    #define STDSORT std::sort
    #define STDUNIQUE std::unique
    #define STDOSTREAM std::ostream
    #define STDLIST std::list
    #define USING_STD_NAME_SPACE using namespace std;
    #define NS_BEGIN_eLiSe  namespace eLiSe{
    #define NS_END_eLiSe }
    #define NS_USING_eLiSe using namespace eLiSe;
#else
    #define STDSORT sort
    #define STDUNIQUE unique
    #define STDOSTREAM ostream
    #define STDLIST list
    #define USING_STD_NAME_SPACE
    #define BEGIN_ELISE_NAME_SPACE
    #define END_ELISE_NAME_SPACE
#endif

#define ElSTDNS  std::
#define ElTmplSpecNull template <>


#if (SUN_WS5)
    #define ElMemberTpl 0
#else
    #define ElMemberTpl 1
#endif

// Directory where Elise is installed


//===================================================
//  You should not need to redefine this section
//===================================================

#if (Compiler_Visual_5_0 || Compiler_Visual_6_0)
    #define PRE_CLASS
#else
    #define PRE_CLASS class
#endif

#define SUN_WS (SUN_WS5 || SUN_WS6)


/*
 *   STRICT_ANSI_FRIEND_TPL vaut vrai si on suit le 14.5.3 de la norme
 *   ISO/IEC 14881:1998 (E) (autrement dit la norme ANSI/C++).
 *
 */
#if (ELISE_windows & !ELISE_MinGW)
    #define STRICT_ANSI_FRIEND_TPL 0
#else
    #define STRICT_ANSI_FRIEND_TPL 1
#endif
/******************************************************************/
/******************************************************************/
/******************************************************************/



  /*********************************************/

/*

typedef enum
{
     ELISE_VW_X11,

}
TY_ELISE_VIDEO_WIN;

typedef enum
{
     ELISE_I486DX4,
     ELISE_DEC3000
}
TY_ELISE_PROCESSOR;



typedef enum
{
     ELISE_LINUX,
     ELISE_OSF,
     ELISE_WNT
}
TY_OPERATING_SYST;



#define _ELISE_IGN 1
#define _ELISE_HOME_X11 0
#define _ELISE_HOME_WNT 0



#if (_ELISE_IGN)
const  TY_ELISE_PROCESSOR  ELISE_PROCESSOR =  ELISE_DEC3000;
const  TY_OPERATING_SYST   ELISE_OS        =  ELISE_OSF;
#endif
#if (_ELISE_HOME_X11)
const  TY_ELISE_PROCESSOR  ELISE_PROCESSOR =  ELISE_I486DX4;
const  TY_OPERATING_SYST   ELISE_OS        =  ELISE_LINUX;
#endif
#if (_ELISE_HOME_WNT)
const  TY_ELISE_PROCESSOR  ELISE_PROCESSOR =  ELISE_I486DX4;
const  TY_OPERATING_SYST   ELISE_OS        =  ELISE_WNT;
#endif
*/


/*******************************************/
/*  [1]                                    */
/*  Definition of portable numerical types */
/*                                         */
/*******************************************/

#include <stdint.h>
#define U_INT4 unsigned  int
#define INT4   int
#define U_INT8 unsigned long long int
#define _INT8   long long int // INT8 is already defined by windows.h and means "char"
#define U_INT2 unsigned short
#define INT2   signed short
#define U_INT1 unsigned char
#define INT1   signed char

#define REAL4   float
#define REAL8   double
#define REAL16  long double

#define INTByte8  long long int


// INT and REAL are the type used for storing intermediate
//  computation. On almost every hardware we will have
//  INT = int and REAL = double.

#ifndef INT
#define INT    INT4
#endif

#ifndef U_INT
#define U_INT  unsigned int
#endif

#ifndef REAL
#define REAL   REAL8
#endif

extern int TheIntFuckingReturnValue;
extern char * TheCharPtrFuckingReturnValue;
int trace_system( const char *cmd );		 // print cmd and execute ::system (helps with debugging)
extern int (*system_call)( const char*cmd ); // equals ::system unless __TRACE_SYSTEM__ is defined (see all.cpp)
#if (!ELISE_windows)
	#include <stdio.h>
	// same thing as system but with popen
	FILE * trace_popen( const char *cmd, const char *acces );
	extern FILE * (*popen_call)( const char *cmd, const char *acces );
#endif

#define  VoidFscanf TheIntFuckingReturnValue=::fscanf
#define  VoidScanf TheIntFuckingReturnValue=::scanf
#define  VoidSystem TheIntFuckingReturnValue=::system_call
#define  VoidFgets TheCharPtrFuckingReturnValue=::fgets


/************* SPECIAL BUGS AND PROBLEMS ******************/


   // bugs, with g++ 2.7.2, on linux,
   // see  "bugs_cpp/stat_const_class.C"

#if (ELISE_OS==ELISE_LINUX)
#define CONST_STAT_TPL
#else
#define CONST_STAT_TPL const
#endif


typedef const INT * const *  Const_INT_PP ;
typedef const REAL * const *  Const_REAL_PP ;


#if  (GPP3etPlus)
#if (MACHINE_BLERIOT)
#define STD_INPUT_STRING_STREAM   istrstream
#else
#define STD_INPUT_STRING_STREAM   istringstream
#endif
#else
#define STD_INPUT_STRING_STREAM   istrstream
#endif

           /****************  PRAGMA ************/

#if (Compiler_Turbo_4_5)
typedef enum  {false,true} bool;
#endif


extern bool MSBF_PROCESSOR();


#if (ELISE_windows)
#define ElIsBlank ISBLANK
#else
#define ElIsBlank isblank
#endif


#define Chol16Byte 0

#if (Chol16Byte)
typedef REAL16  tSysCho ;
#else
typedef REAL8  tSysCho ;
#endif


// Version int de __HG_REV__
// is doesn't work with the new Git Version
//int NumHgRev();

#if ELISE_PTR_SIZE==4
    #define ELISE_PTR_U_INT U_INT4
    #define ELISE_PTR_FORMAT "%l"
#elif ELISE_PTR_SIZE==8
    #define ELISE_PTR_U_INT U_INT8
    #define ELISE_PTR_FORMAT "%ll"
#else
    unhandled size of pointer
#endif


#endif /* ! _ELISE_SYS_DEP_H */



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
