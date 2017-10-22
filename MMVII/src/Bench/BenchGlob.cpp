#include "include/MMVII_all.h"

namespace MMVII
{

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
    MMVII_INTERNAL_ASSERT_bench(aSplit[0]=="AA","V0 in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[1]=="BB","V0 in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[2]=="CC","V0 in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit[3]=="DD","V0 in Bench_0000_String");
}



cAppli_MMVII_Bench::cAppli_MMVII_Bench (int argc,char **argv) :
    cMMVII_Appli
    (
        argc,
        argv,
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

    Bench_0000_Memory();
    Bench_0000_String();
    Bench_0000_Ptxd();

    std::cout << "BenchGlobBenchGlob \n";

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

