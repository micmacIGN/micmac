#include "include/MMVII_all.h"
#include <cmath>

/** \file BenchGlob.cpp
    \brief Main bench


    For now bench are relatively simples and executed in the same process,
  it's than probable that when MMVII grow, more complex bench will be done
  by sub process specialized.

*/


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

void   Bench_0000_Param()
{
   int a,b;
   cCollecSpecArg2007 aCol;
   aCol << Arg2007(a,"UnA") << AOpt2007(b,"b","UnB") ;
   aCol[0]->InitParam("111");
   aCol[1]->InitParam("222");
   std::cout << "GGGGGGGG " << a << " " << b << "\n";

   MMVII_INTERNAL_ASSERT_bench(a==111,"Bench_0000_Param");
   MMVII_INTERNAL_ASSERT_bench(b==222,"Bench_0000_Param");
}


/*************************************************************/
/*                                                           */
/*            cAppli_MMVII_Bench                             */
/*                                                           */
/*************************************************************/

class cAppli_MMVII_Bench : public cMMVII_Appli
{
     public :

        cAppli_MMVII_Bench(int,char**,const cSpecMMVII_Appli & aSpec);
        void Bench_0000_String();
        void BenchFiles(); ///< A Bench on creation/deletion/existence of files
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override {return anArgObl;}
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override {return anArgOpt;}
};


void cAppli_MMVII_Bench::Bench_0000_String()
{
    // Bench elem sur la fonction SplitString
    std::vector<std::string> aSplit;
    SplitString(aSplit,"@  @AA  BB@CC DD   @  "," @");
    MMVII_INTERNAL_ASSERT_bench(aSplit.size()==6,"Size in Bench_0000_String");

    MMVII_INTERNAL_ASSERT_bench(aSplit.at(0)=="","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit.at(1)=="AA","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit.at(2)=="BB","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit.at(3)=="CC","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit.at(4)=="DD","SplitString in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(aSplit.at(5)=="","SplitString in Bench_0000_String");

    MMVII_INTERNAL_ASSERT_bench(Prefix("AA.tif")=="AA",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("AA.tif")=="tif","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("AA.tif",'t')=="if","Postfix in Bench_0000_String");

    MMVII_INTERNAL_ASSERT_bench(Prefix("a.b.c",'.',true,true)=="a.b",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Prefix("a.b.c",'.',true,false)=="a",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("a.b.c",'.',true,true)=="c",  "Prefix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("a.b.c",'.',true,false)=="b.c",  "Prefix in Bench_0000_String");

    MMVII_INTERNAL_ASSERT_bench(Postfix("AA.",'.')=="","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench( Prefix("AA.",'.')=="AA","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix(".AA",'.')=="AA","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench( Prefix(".AA",'.')=="","Postfix in Bench_0000_String");
}


   // std::string & aBefore,std::string & aAfter,const std::string & aStr,char aSep,bool SVP=false,bool PrivPref=true);

cAppli_MMVII_Bench::cAppli_MMVII_Bench (int argc,char **argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec)
{
  MMVII_INTERNAL_ASSERT_always
  (
        The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_tiny,
        "MMVII Bench requires highest level of debug"
  );
  // The_MMVII_DebugLevel = The_MMVII_DebugLevel_InternalError_weak;
}

static void CreateFile(const std::string & aNameFile)
{
   cMMVII_Ofs aFile(aNameFile);
   int anI=44;
   aFile.Write(anI);
}

void cAppli_MMVII_Bench::BenchFiles()
{
   const std::string & aTDir = TmpDirTestMMVII();
   std::string aNameFile = aTDir+"toto.txt";
   
   // Dir should be empty
   MMVII_INTERNAL_ASSERT_always(!ExistFile(aNameFile),"BenchFiles");
   CreateFile(aNameFile);
   // File should now exist
   MMVII_INTERNAL_ASSERT_always(ExistFile(aNameFile),"BenchFiles");

   // CreateFile(aTDir+"a/b/c/toto.txt"); Do not work directly
   CreateDirectories(aTDir+"a/b/c/",false);
   CreateFile(aTDir+"a/b/c/toto.txt"); // Now it works


   RemoveRecurs(TmpDirTestMMVII(),true,false);
}


int  cAppli_MMVII_Bench::Exe()
{
   // Begin with purging directory
   CreateDirectories(TmpDirTestMMVII(),true );
   RemoveRecurs(TmpDirTestMMVII(),true,false);


   //  On teste les macro d'assertion
   MMVII_INTERNAL_ASSERT_bench((1+1)==2,"Theoreme fondamental de l'arithmetique");
   // La on a verifie que ca marchait pas
   // MMVII_INTERNAL_ASSERT_all((1+1)==3,"Theoreme  pas tres fondamental de l'arithmetique");

   BenchFiles();
   Bench_0000_SysDepString();
   Bench_0000_String();
   Bench_0000_Memory();
   Bench_0000_Ptxd();
   Bench_0000_Param();

   BenchSerialization(mDirTestMMVII+"Tmp/",mDirTestMMVII+"Input/");

   // std::cout << "BenchGlobBenchGlob \n";

   // std::cout << " 1/0=" << 1/0  << "\n";
   // std::cout <<  " 1.0/0.0" << 1.0/0.0  << "\n";
   // std::cout << " sqrt(-1)=" << sqrt(-1)  << "\n";
   // std::cout << " asin(2)=" << asin(2.0) << "\n";

   BenchSet(mDirTestMMVII);
   BenchSelector(mDirTestMMVII);
   BenchEditSet();


   // We clean the temporary files created
   RemoveRecurs(TmpDirTestMMVII(),true,false);

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_MMVII_Bench(int argc,char ** argv,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_Bench(argc,argv,aSpec));
}
 

cSpecMMVII_Appli  TheSpecBench
(
     "Bench",
      Alloc_MMVII_Bench,
      "This command execute (many) self verification on MicMac-V2 behaviour",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console}
);


/* ========================================================= */
/*                                                           */
/*            cAppli_MPDTest                                 */
/*                                                           */
/* ========================================================= */

class cAppli_MPDTest : public cMMVII_Appli
{
     public :
        cAppli_MPDTest(int argc,char** argv,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override {return anArgObl;}
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override {return anArgOpt;}
};

cAppli_MPDTest:: cAppli_MPDTest(int argc,char** argv,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (argc,argv,aSpec)
{
}



void TestArg0(const std::vector<int> & aV0)
{
   for (auto I : aV0){I++; std::cout << "I=" << I << "\n"; }
}


std::string BUD(const std::string & aDir);
void TestBooostIter();

class cTestShared
{
    public :
        cTestShared() {std::cout  << "CREATE cTestShared "<< this << "\n";;}
        ~cTestShared() {std::cout << "XXXXXX cTestShared "<< this << "\n";;}
        static void Test()
        {
            cTestShared anOb;
            // std::shared_ptr<cTestShared> aT(&anOb);
            std::shared_ptr<cTestShared> aT(new cTestShared);
        }
};




// #include <limits>
int cAppli_MPDTest::Exe()
{
   
  cTestShared::Test();
  std::cout << "CHAR LIMS " << (int) std::numeric_limits<char>::min() << " " << (int) std::numeric_limits<char>::max() << "\n";

  std::cout << "DIRBIN2007:" << DirBin2007 << "\n";

  tNameSet aSet;
  aSet.Add("toto");
  std::cout << "SIZZZ " << aSet.size() << "\n";
  SaveInFile(aSet,"toto.xml");
  std::vector<std::string> aV;
  aV.push_back("toto");
  SaveInFile(aV,"totoV.xml");

  tNameSet aS2;
  ReadFromFile(aS2,"toto.xml");
  std::cout << "SIZZZ " << aS2.size() << "\n";
/*
   TestBooostIter();
   BUD(".");
   BUD("/a/b/c");
   BUD("a/b/c");
   BUD("a");
   TestArg0({1,3,9});
   TestArg1({});
   TestArg1({eTypeArg::MDirOri});
   TestArg1({eTypeArg::MPatIm,eTypeArg::MDirOri,{eTypeArg::MDirOri,"Un"}});
  CreateDirectories("a/b/c/d",false);
*/

   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_MPDTest(int argc,char ** argv,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MPDTest(argc,argv,aSpec));
}

cSpecMMVII_Appli  TheSpecMPDTest
(
     "MPDTest",
      Alloc_MPDTest,
      "This used a an entry point to all quick and dirty test by MPD ...",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console}
);


};

