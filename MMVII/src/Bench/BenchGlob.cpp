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

    std::string aPath5 = "NONE";
    MMVII_INTERNAL_ASSERT_bench(DirOfPath (aPath5,false)=="./","Dir Bench_0000_SysDepString");
    MMVII_INTERNAL_ASSERT_bench(FileOfPath(aPath5,false)=="NONE","File Bench_0000_SysDepString");
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
    cMemManager::CheckRestoration(aSt);
}

void   Bench_0000_Param()
{
   int a,b;
   cCollecSpecArg2007 aCol;
   aCol << Arg2007(a,"UnA") << AOpt2007(b,"b","UnB") ;
   aCol[0]->InitParam("111");
   aCol[1]->InitParam("222");

   MMVII_INTERNAL_ASSERT_bench(a==111,"Bench_0000_Param");
   MMVII_INTERNAL_ASSERT_bench(b==222,"Bench_0000_Param");
}


/*************************************************************/
/*                                                           */
/*            cAppli_MMVII_Bench                             */
/*                                                           */
/*************************************************************/

/// entry point for all unary test

/** This class contain all the unary test to check the validaty of
    command / classes / function relatively to their specs.

    For now its essentially a serie of function that are called linearly.
   When the test become long to execute, it may evolve with option allowing
   to do only some specific bench.
*/

class cAppli_MMVII_Bench : public cMMVII_Appli
{
     public :

        cAppli_MMVII_Bench(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        void Bench_0000_String();
        void BenchFiles(); ///< A Bench on creation/deletion/existence of files
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::string mTest; ///< Do remember why I added that !!
        int         mNumBugRecall; ///< Used if we want to force bug generation in recall process
};

cCollecSpecArg2007 & cAppli_MMVII_Bench::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              << Arg2007(mTest,"Unused, to check reentrance" )
    ;
}

cCollecSpecArg2007 & cAppli_MMVII_Bench::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
  return
      anArgOpt
         << AOpt2007(mNumBugRecall,"NBR","Num to Generate a Bug in Recall,(4 manuel inspection of log file)")
  ;
}



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

    MMVII_INTERNAL_ASSERT_bench(Postfix("AA",'.',true,true)=="AA","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Prefix("AA",'.',true,true)=="","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Postfix("AA",'.',true,false)=="","Postfix in Bench_0000_String");
    MMVII_INTERNAL_ASSERT_bench(Prefix("AA",'.',true,false)=="AA","Postfix in Bench_0000_String");
}


   // std::string & aBefore,std::string & aAfter,const std::string & aStr,char aSep,bool SVP=false,bool PrivPref=true);

cAppli_MMVII_Bench::cAppli_MMVII_Bench (const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli    (aVArgs,aSpec),
  mNumBugRecall   (-1)
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
   cMMVII_Ofs aFile(aNameFile,false);
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


void BenchFilterLinear();


int  cAppli_MMVII_Bench::Exe()
{
   // Begin with purging directory
   CreateDirectories(TmpDirTestMMVII(),true );
   RemoveRecurs(TmpDirTestMMVII(),true,false);


   //  On teste les macro d'assertion
   MMVII_INTERNAL_ASSERT_bench((1+1)==2,"Theoreme fondamental de l'arithmetique");
   // La on a verifie que ca marchait pas
   // MMVII_INTERNAL_ASSERT_all((1+1)==3,"Theoreme  pas tres fondamental de l'arithmetique");

   BenchRecall(mNumBugRecall);

   BenchDenseMatrix0();
   BenchGlobImage();

   Bench_Nums();

   BenchFiles();
   Bench_0000_SysDepString();
   Bench_0000_String();
   Bench_0000_Memory();
   Bench_0000_Ptxd();
   Bench_0000_Param();

   BenchSerialization(mDirTestMMVII+"Tmp/",mDirTestMMVII+"Input/");


   BenchSet(mDirTestMMVII);
   BenchSelector(mDirTestMMVII);
   BenchEditSet();
   BenchEditRel();

   BenchEnum();
   Bench_Random();
   Bench_Duration();

   BenchStrIO();


   BenchFilterImage1();
   BenchFilterLinear();

   // We clean the temporary files created
   RemoveRecurs(TmpDirTestMMVII(),true,false);

   //  TestTimeV1V2(); => Valide ratio ~=  1

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_MMVII_Bench(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_Bench(aVArgs,aSpec));
}
 

cSpecMMVII_Appli  TheSpecBench
(
     "Bench",
      Alloc_MMVII_Bench,
      "This command execute (many) self verification on MicMac-V2 behaviour",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console},
      __FILE__
);

/* ========================================================= */
/*                                                           */
/*            cAppli_MMRecall                                */
/*                                                           */
/* ========================================================= */

/// A class to test mecanism of MMVII recall itself

/** This class make some rather stupid computation to
    generate and multiple recall of MMVII by itself

    Each appli has an Id number Num, if the level of  recursion
    is not reached, it generate two subprocess 2*Num and 2*Num +1

    Each process generate a file, as marker of it was really executed

    At the end we check that all the marker exist (and no more), and we clean

*/

class cAppli_MMRecall : public cMMVII_Appli
{
     public :
        static const int NbMaxArg=2;

        cAppli_MMRecall(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        int          mNum;  ///< kind of identifiant of the call
        int          mLev0; ///< to have the right depth we must know level of
        std::string  mAM[NbMaxArg];  ///<  to get the mandatory Args
        std::string  mAO[NbMaxArg];  ///<  to get the optionall Args  
        bool         mRecalInSameProcess; ///< The recall mecanism can be tested by subprocess or inside same
        int          mNumBug;  ///< Generate bug 4 this num
};


cAppli_MMRecall::cAppli_MMRecall(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli         (aVArgs,aSpec),
  mLev0                (0),
  mRecalInSameProcess  (true),
  mNumBug              (-1)
{
}

int cAppli_MMRecall::Exe() 
{
    std::string aDirT =  TmpDirTestMMVII() ;
    // Purge TMP
    if (mLevelCall == mLev0)
    {
        RemovePatternFile(aDirT+".*",true);
    }

    // to create a file with Num in name
    { 
       std::string aNameF = aDirT + ToStr(mNum) + ".txt";
       cMMVII_Ofs (aNameF,false);
    }
    MMVII_INTERNAL_ASSERT_always(mNum!=mNumBug,"Bug generate by user in cAppli_MMRecall");

    // Break recursion
    if (mLevelCall-mLev0 >= NbMaxArg)
       return EXIT_SUCCESS;

    // Recursive call to two son  N-> 2N, 2N+1, it's the standard binary tree like in heap, this make it bijective
    {
        std::vector<std::string> aLVal;
        aLVal.push_back(ToStr(2*mNum));
        aLVal.push_back(ToStr(2*mNum+1));

        cColStrAOpt  aSub;
        eTyModeRecall aMRec = mRecalInSameProcess ? eTyModeRecall::eTMR_Inside : eTyModeRecall::eTMR_Serial;

        ExeMultiAutoRecallMMVII("0",aLVal ,aSub,aMRec);
    }

    // Test that we have exactly the expected file (e.g. 1,2, ... 31 )  and purge
    if (mLevelCall == mLev0)
    {
        tNameSet  aSet1 = SetNameFromPat(aDirT+".*");
        MMVII_INTERNAL_ASSERT_bench(aSet1.size()== ((2<<NbMaxArg) -1),"Sz set in  cAppli_MMRecall");

        tNameSet  aSet2 ;
        for (int aK=1 ; aK<(2<<NbMaxArg) ; aK++)
            aSet2.Add( ToStr(aK) + ".txt");

        MMVII_INTERNAL_ASSERT_bench(aSet1.Equal(aSet2),"Sz set in  cAppli_MMRecall");
        RemovePatternFile(aDirT+".*",true);
    }

    return EXIT_SUCCESS;
}


cCollecSpecArg2007 & cAppli_MMRecall::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return anArgObl
          << Arg2007(mNum,"Num" )
          << Arg2007(mAM[0],"Mandatory arg0" )
          << Arg2007(mAM[1],"Mandatory arg1" )
          // << Arg2007(mAM[2],"Mandatory arg2" )
          // << Arg2007(mAM[3],"Mandatory arg3" )
   ;

}

cCollecSpecArg2007 & cAppli_MMRecall::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
  return
      anArgOpt
         << AOpt2007(mAO[0],"A0","Optional Arg 0")
         << AOpt2007(mAO[1],"A1","Optional Arg 1")
         // << AOpt2007(mAO[2],"A2","Optional Arg 2")
         // << AOpt2007(mAO[3],"A3","Optional Arg 3")
         << AOpt2007(mLev0,"Lev0","Level of first call")
         << AOpt2007(mRecalInSameProcess,"RISP","Recall in same process")
         << AOpt2007(mNumBug,"NumBug","Num 4 generating purpose scratch")
   ;
}

tMMVII_UnikPApli Alloc_TestRecall(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMRecall(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecTestRecall
(
     "TestRecall",
      Alloc_TestRecall,
      "Use in Bench to Test Recall of MMVII by itself ",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console},
      __FILE__
);

void OneBenchRecall(bool InSameP,int aNumBug)
{
    cMMVII_Appli &  anAp = cMMVII_Appli::CurrentAppli();

    anAp.StrObl() << "1";

    for (int aK=0 ; aK< cAppli_MMRecall::NbMaxArg; aK++)
        anAp.StrObl() << ToStr(10*aK);

    anAp.ExeCallMMVII
    (
        TheSpecTestRecall,
        anAp.StrObl() ,
        anAp.StrOpt() << t2S("Lev0",ToStr(1+anAp.LevelCall()))
                      << t2S("RISP",ToStr(InSameP))
                      << t2S("NumBug",ToStr(aNumBug))
    );
}

void BenchRecall(int aNum)
{
     OneBenchRecall(true,-1);
     OneBenchRecall(false,aNum);
}


/* ========================================================= */
/*                                                           */
/*            cAppli_MPDTest                                 */
/*                                                           */
/* ========================================================= */


/// A class to make quick and dirty test

/** The code in this class will evolve
  quickly, it has no perenity, if a test become
  important it must be put in bench
*/

class cAppli_MPDTest : public cMMVII_Appli
{
     public :
        cAppli_MPDTest(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override {return anArgObl;}
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override {return anArgOpt;}
};

cAppli_MPDTest:: cAppli_MPDTest(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec)
{
}



void TestArg0(const std::vector<int> & aV0)
{
   for (auto I : aV0){I++; StdOut() << "I=" << I << "\n"; }
}


std::string BUD(const std::string & aDir);
void TestBooostIter();

class cTestShared
{
    public :
        cTestShared()  {StdOut()  << "CREATE cTestShared "<< this << "\n";;}
        ~cTestShared() {StdOut() << "XXXXXX cTestShared "<< this << "\n";;}
        static void Test()
        {
            cTestShared anOb;
            // std::shared_ptr<cTestShared> aT(&anOb);
            std::shared_ptr<cTestShared> aT(new cTestShared);
        }
};


/*
class cMultipleOfs  : public  std::ostream
{
    public :
        cMultipleOfs(std::ostream & aos1,std::ostream & aos2) :
           mOs1 (aos1),
           mOs2 (aos2)
        {
        }
        
        std::ostream & mOs1;
        std::ostream & mOs2;

        template <class Type> cMultipleOfs & operator << (const Type & aVal)
        {
             mOs1 << aVal;
             mOs2 << aVal;
             return *this;
        }
};
class cMultipleOfs  : public  std::ostream
{
    public :
        cMultipleOfs()
        {
        }
        void Add(std::ostream & aOfs) {mVOfs.push_back(&aOfs);}
        
        std::vector<std::ostream *> mVOfs;

        template <class Type> cMultipleOfs & operator << (const Type & aVal)
        {
             for (const auto & Ofs :  mVOfs) 
                 *Ofs << aVal;
             return *this;
        }
};
*/

void TestVectBool()
{
    StdOut() << "BEGIN TBOOL \n"; getchar();

    for (int aK=0 ; aK<5000 ; aK++)
    {
        std::vector<bool> * aV = new std::vector<bool>;
        for (int aK=0 ; aK<1000000 ; aK++)
           aV->push_back(true);
    }
    StdOut() << "END TBOOL \n"; getchar();
    for (int aK=0 ; aK<5000 ; aK++)
    {
        std::vector<tU_INT1> * aV = new std::vector<tU_INT1>;
        for (int aK=0 ; aK<1000000 ; aK++)
           aV->push_back(1);
    }
    StdOut() << "END TBYTE \n"; getchar();
}

bool F(const std::string & aMes) {std::cout <<"FFFFF=" << aMes << "\n"; return true;}
#define UN 1
#define DEUX 2 

// #include <limits>
int cAppli_MPDTest::Exe()
{
    if ((UN>DEUX) && F("aaaa"))
    {
       F("bbbb");
    }
    F("ccccc");
   
/*
   cSparseVect<float>  aSV;
   for (const auto & aP : aSV)
   {
        std::cout << aP.mI << "\n";
   }
*/

/*
   cIm2D<tU_INT1> aIm(cPt2di(3,3));
   aIm.DIm().SetV(cPt2di(0,0),13);
   // aIm.DIm().SetV(cPt2di(0,0),1000);
   // aIm.DIm().SetV(cPt2di(-1,0),1);
   // new cIm2D<tU_INT1>(cPt2di(3,3));
   cDataIm2D<tU_INT1> & aDIm = aIm.DIm();
   tU_INT1*  aPtr = aDIm.RawDataLin();
   StdOut() << "aIm=" << int(aPtr[0]) <<  "\n";
   aPtr[0] = 14;
   StdOut() << "aIm=" << (int)aDIm.GetV(cPt2di(0,0)) <<  "\n";
   // aPtr[-1] = 0;
*/

/*
    TestVectBool();
   cMMVII_Ofs aOs1("toto1.txt");
   cMMVII_Ofs aOs2("toto2.txt");

    
   cMultipleOfs amOs; // (aOs1.Ofs(),aOs2.Ofs());
   amOs.Add(aOs1.Ofs());
   amOs.Add(aOs2.Ofs());
   amOs << "1+1=" << 1+1 << "\n";
   cMMVII_Ofs aFile("toto.txt");
   std::ostream & anOFS =  aFile.Ofs();
   anOFS << "TEST OFFFSSSSSSSSSSSS\n";
*/

   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_MPDTest(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MPDTest(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMPDTest
(
     "MPDTest",
      Alloc_MPDTest,
      "This used a an entry point to all quick and dirty test by MPD ...",
      {eApF::Test},
      {eApDT::None},
      {eApDT::Console},
      __FILE__
);


};

