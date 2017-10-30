#include "include/MMVII_all.h"
#include <cmath>

namespace MMVII
{

#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
void Bench_0000_SysDepString()
{
    std::string aPath0 = "MMVII";
    MMVII_INTERNAL_ASSERT_bench(DirOfPath (aPath0,false)=="./","Dir Bench_0000_SysDepString");
    MMVII_INTERNAL_ASSERT_bench(FileOfPath(aPath0,false)=="MMVII","File Bench_0000_SysDepString");


    std::string aPath1 = "af.tif";
    MMVII_INTERNAL_ASSERT_bench(DirOfPath (aPath1,false)=="./","Dir Bench_0000_SysDepString");
    MMVII_INTERNAL_ASSERT_bench(FileOfPath(aPath1,false)=="af.tif","File Bench_0000_SysDepString");

    std::string aPath2 = "./toto.txt";
    MMVII_INTERNAL_ASSERT_bench(DirOfPath (aPath2,false)=="./","Dir Bench_0000_SysDepString");
    MMVII_INTERNAL_ASSERT_bench(FileOfPath(aPath2,false)=="toto.txt","File Bench_0000_SysDepString");

    std::string aPath3 = "/a/bb/cc/";
    MMVII_INTERNAL_ASSERT_bench(DirOfPath (aPath3,false)==aPath3,"Dir Bench_0000_SysDepString");
    MMVII_INTERNAL_ASSERT_bench(FileOfPath(aPath3,false)=="","File Bench_0000_SysDepString");

    std::string aPath4 = "/a/bb/cc/tutu";
    MMVII_INTERNAL_ASSERT_bench(DirOfPath (aPath4,false)==aPath3,"Dir Bench_0000_SysDepString");
    MMVII_INTERNAL_ASSERT_bench(FileOfPath(aPath4,false)=="tutu","File Bench_0000_SysDepString");
}
#else
void Bench_0000_SysDepString()
{
}
#endif

void Bench_0000_Memory()
{
    cMemState  aSt =   cMemManager::CurState();
    int aNb=5;
    // short * aPtr = static_cast<short *> (cMemManager::Calloc(sizeof(short),aNb));
    ///  short * aPtr = MemManagerAlloc<short>(aNb);
    short * aPtr = cMemManager::Alloc<short>(aNb);
    for (int aK=0; aK<aNb ; aK++)
        aPtr[aK] = 10 + 234 * aK;

    // Plein de test d'alteration 
    if (0)  aPtr[-1] =9;
    if (0)  aPtr[aNb+6] =9;
    if (0)  aPtr[aNb] =9;
    if (0)  aPtr[aNb+1] =9;
    // Plante si on teste avant liberation
    if (0)  cMemManager::CheckRestoration(aSt);
    cMemManager::Free(aPtr);
    // std::cout << "cMemManager::Free " << cMemManager::IsOkCheckRestoration(aSt) << "\n";
    cMemManager::CheckRestoration(aSt);
}



/*************************************************************/
/*                                                           */
/*            cAppli_MMVII_Bench                             */
/*                                                           */
/*************************************************************/

class cAppli_MMVII_Bench : public cMMVII_Appli
{
     public :
        cAppli_MMVII_Bench(int,char**);
        void Bench_0000_String();
        int Exe();
};


void cAppli_MMVII_Bench::Bench_0000_String()
{
    // Bench elem sur la fonction SplitString
    std::vector<std::string> aSplit;
    SplitString(aSplit,"@  @AA  BB@CC DD   @  "," @");
    MMVII_INTERNAL_ASSERT_bench(aSplit.size()==4,"Size in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[0]=="AA","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[1]=="BB","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[2]=="CC","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[3]=="DD","SplitString in Bench_0000_String");

    MMVII_INTERNAL_ASSERT_bench(Prefix("AA.tif")=="AA",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("AA.tif")=="tif","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("AA.tif",'t')=="if","Postfix in Bench_0000_String");

    MMVII_INTERNAL_ASSERT_bench(Prefix("a.b.c",'.',true,true)=="a.b",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Prefix("a.b.c",'.',true,false)=="a",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("a.b.c",'.',true,true)=="c",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("a.b.c",'.',true,false)=="b.c",  "Prefix in Bench_0000_String");
}


   // std::string & aBefore,std::string & aAfter,const std::string & aStr,char aSep,bool SVP=false,bool PrivPref=true);

cAppli_MMVII_Bench::cAppli_MMVII_Bench (int argc,char **argv) :
    cMMVII_Appli
    (
        argc,
        argv,
        DirCur(),
        cArgMMVII_Appli
        (
        )
    )
{
   MMVII_INTERNAL_ASSERT_always
   (
        The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_tiny,
        "MMVII Bench requires highest level of debug"
   );
   // The_MMVII_DebugLevel = The_MMVII_DebugLevel_InternalError_weak;
}


int  cAppli_MMVII_Bench::Exe()
{
   // cMemManager::Alloc<short>(4);
   //  On teste les macro d'assertion
   MMVII_INTERNAL_ASSERT_bench((1+1)==2,"Theoreme fondamental de l'arithmetique");
   // MMVII_INTERNAL_ASSERT_all((1+1)==3,"Theoreme  pas tres fondamental de l'arithmetique");

   // 
   Bench_0000_SysDepString();

   Bench_0000_String();
   Bench_0000_Memory();
   Bench_0000_String();
   Bench_0000_Ptxd();


   Bench_0000_SysDepString();

   std::cout << "BenchGlobBenchGlob \n";

   // std::cout << " 1/0=" << 1/0  << "\n";
   std::cout <<  " 1.0/0.0" << 1.0/0.0  << "\n";
   std::cout << " sqrt(-1)=" << sqrt(-1)  << "\n";
   std::cout << " asin(2)=" << asin(2.0) << "\n";





   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_MMVII_Bench(int argc,char ** argv)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_Bench(argc,argv));
}
 

cSpecMMVII_Appli  TheSpecBench
(
     "Bench",
      Alloc_MMVII_Bench,
      "This command execute (many) self verification on MicMac-V2 behaviour",
      "Test",
      "None",
      "Console"
);



};

