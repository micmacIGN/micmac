#include "SymbDer/SymbDer_Common.h"
#include "MMVII_DeclareCste.h"
#include "cMMVII_Appli.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Radiom.h"
#include "MMVII_Random.h"
#include "MMVII_Tpl_Images.h"
#include <cmath>
#include <functional>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif // _WIN32

using namespace std;


// Cross-platform sleep function
// or use
// #include <chrono>
// #include <thread>
// std::this_thread::sleep_for(std::chrono::milliseconds(x));

void sleepcp(int milliseconds)
{
    #ifdef _WIN32
        Sleep(milliseconds);
    #else
        usleep(milliseconds * 1000);
    #endif // _WIN32
}

#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"

using namespace NS_SymbolicDerivative ;
/** \file BenchGlob.cpp
    \brief Main bench


    For now bench are relatively simples and executed in the same process,
  it's than probable that when MMVII grow, more complex bench will be done
  by sub process specialized.

*/

// #include "include/TreeDist.h"


namespace MMVII
{

/* ================================================ */
/*                                                  */
/*            cAppliBenchAnswer                     */
/*                                                  */
/* ================================================ */

cAppliBenchAnswer::cAppliBenchAnswer(bool HasBench,double aTime) :
   mHasBench (HasBench),
   mTime     (aTime)
{
}

/* ================================================ */
/*                                                  */
/*              cParamExeBench                      */
/*                                                  */
/* ================================================ */

cParamExeBench::cParamExeBench(const std::string & aPattern,const std::string &aBugKey,int aLevInit,bool Show) :
   mInsideFunc  (false),
   mLevInit     (aLevInit),
   mCurLev      (mLevInit),
   mShow        (Show),
   mNbExe       (0),
   mName        (aPattern),
   mPattern     (AllocRegex(aPattern)),
   mBugKey      (aBugKey)
{
}

bool  cParamExeBench::NewBench(const std::string & aName,bool ExactMatch)
{
   if (mCurLev==mLevInit) 
   {
      mVallBench.push_back(aName);
      mVExactMatch.push_back(ExactMatch);
      mVAllBugKeys.push_back(std::vector<std::string> ());
   }
   MMVII_INTERNAL_ASSERT_always(!mInsideFunc,"Bad NewBench/EndBench handling");
   if (ExactMatch ? (mName==aName)  : mPattern.Match(aName))
   {
       mNbExe++;
       mInsideFunc = true;
       StdOut() << "  Bench : " << aName << std::endl;
       cRandGenerator::TheOne()->setSeed(mCurLev);
   }
   return  mInsideFunc;
}

bool  cParamExeBench::GenerateBug(const std::string & aKey)
{
   if (mCurLev==mLevInit)
   {
      MMVII_INTERNAL_ASSERT_always(! mVAllBugKeys.empty() ,"Bad NewBench/EndBench handling");
      mVAllBugKeys.back().push_back(aKey);
   }
   return (mBugKey==aKey);
}


void cParamExeBench::ShowIdBench() const
{
   StdOut() << "=====  POSSIBLE ID FOR BENCH ==============,  # require exact match" << std::endl;
   for (int aK=0 ; aK<int(mVallBench.size()) ; aK++)
   {
       StdOut() << "    " << (mVExactMatch[aK] ? "#" : "-")   << " " << mVallBench[aK]  << " ";
       if (!mVAllBugKeys[aK].empty())
       {
          StdOut() << ":: "  ;
          for (const auto & aNameBug :mVAllBugKeys[aK])
             StdOut()  << aNameBug << " ";
       }
       StdOut()  << std::endl;
   }
}

void  cParamExeBench::EndBench()
{
   MMVII_INTERNAL_ASSERT_always(mInsideFunc,"Bad NewBench/EndBench handling");
   mInsideFunc = false;
}

void  cParamExeBench::IncrLevel()
{
   mCurLev++;
}

bool  cParamExeBench::Show() const  { return mShow; }
int   cParamExeBench::Level() const { return mCurLev; }
int   cParamExeBench::NbExe() const { return mNbExe; }




/* ================================================ */
/*                                                  */
/*            ::MMVII                               */
/*                                                  */
/* ================================================ */

#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
void Bench_0000_SysDepString(cParamExeBench & aParam)
{
    if (! aParam.NewBench("SysDepString")) return;

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

    aParam.EndBench();
}
#else
void Bench_0000_SysDepString(cParamExeBench & aParam)
{
}
#endif

void TestDir(const std::string & aDir);

void Bench_0000_String(cParamExeBench & aParam)
{
    int aNb=0;
    for (int aK=10 ; aK>0 ; aK--)
        aNb++;
    MMVII_INTERNAL_ASSERT_bench(aNb==10,"Test for (int aK=10 ; aK>0 ; aK--)");



    if (! aParam.NewBench("StringOperation")) return;
    // Bench elem sur la fonction SplitString
    // std::vector<std::string> aSplit;
    // SplitString(aSplit,"@  @AA  BB@CC DD   @  "," @");

    std::vector<std::string> aSplit= SplitString("@  @AA  BB@CC DD   @  "," @");
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


    MMVII_INTERNAL_ASSERT_bench(starts_with("ABC","")==true,"starts_with");
    MMVII_INTERNAL_ASSERT_bench(starts_with("ABC","AB")==true,"starts_with");
    MMVII_INTERNAL_ASSERT_bench(starts_with("ABC","ABC")==true,"starts_with");
    MMVII_INTERNAL_ASSERT_bench(starts_with("","")==true,"starts_with");

    MMVII_INTERNAL_ASSERT_bench(starts_with("ABC","ABCD")==false,"starts_with");
    MMVII_INTERNAL_ASSERT_bench(starts_with("","A")==false,"starts_with");
    MMVII_INTERNAL_ASSERT_bench(starts_with("ABC","Ab")==false,"starts_with");
    MMVII_INTERNAL_ASSERT_bench(starts_with("ABC","a")==false,"starts_with");


    MMVII_INTERNAL_ASSERT_bench(ToStandardStringIdent("+A 9")==std::string("A_9"),"ToStandardStringIdent");
    MMVII_INTERNAL_ASSERT_bench(ToStandardStringIdent("&?AbZ!aBz 012 9")==std::string("AbZaBz_012_9"),"ToStandardStringIdent");

    aParam.EndBench();
}


void Bench_0000_Memory(cParamExeBench & aParam)
{
    if (! aParam.NewBench("MemoryOperation")) return;

    cMemState  aSt =   cMemManager::CurState();
    int aNb=5;
    // short * aPtr = static_cast<short *> (cMemManager::Calloc(sizeof(short),aNb));
    ///  short * aPtr = MemManagerAlloc<short>(aNb);
    short * aPtr = cMemManager::Alloc<short>(aNb);
    for (int aK=0; aK<aNb ; aK++)
        aPtr[aK] = 10 + 234 * aK;

    // Plein de test d'alteration 
    if (aParam.GenerateBug("Debord_M1"))  aPtr[-1] =9;
    if (aParam.GenerateBug("Debord_P6"))  aPtr[aNb+6] =9;
    if (aParam.GenerateBug("Debord_P0"))  aPtr[aNb] =9;
    if (aParam.GenerateBug("Debord_P1"))  aPtr[aNb+1] =9;
    // Plante si on teste avant liberation
    if (aParam.GenerateBug("Restore"))  cMemManager::CheckRestoration(aSt);
    cMemManager::Free(aPtr);
    cMemManager::CheckRestoration(aSt);

    aParam.EndBench();
}

void   Bench_0000_Param(cParamExeBench & aParam)
{
   if (! aParam.NewBench("ParamArg2007")) return;

   int a,b;
   cCollecSpecArg2007 aCol;
   aCol << Arg2007(a,"UnA") << AOpt2007(b,"b","UnB") ;
   aCol[0]->InitParam("111");
   aCol[1]->InitParam("222");

   MMVII_INTERNAL_ASSERT_bench(a==111,"Bench_0000_Param");
   MMVII_INTERNAL_ASSERT_bench(b==222,"Bench_0000_Param");

   aParam.EndBench();
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
        void BenchFiles(cParamExeBench & aParam); ///< A Bench on creation/deletion/existence of files


        int  ExecuteBench(cParamExeBench &) override; /// Execute the "unitary" bench
        cAppliBenchAnswer BenchAnswer()    const override; ///< Yes this command has a bench
        int Exe() override;  ///< Call all the bench of commands that have one



        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        int         mLevelMax;  // Max level of bench
        int         mLevMin;   // Min level of bench
        int         mShow;    // Do the bench show details 
        std::string mPat;    // Pattern for selected bench
        std::string mKeyBug;    // Pattern for selected bench
        int         mNumBugRecall; ///< Used if we want to force bug generation in recall process
        bool        mDoBUSD;       ///< Do we do  BenchUnbiasedStdDev
};

cCollecSpecArg2007 & cAppli_MMVII_Bench::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              << Arg2007(mLevelMax,"Level of Bench, Higher more test" )
    ;
}

cCollecSpecArg2007 & cAppli_MMVII_Bench::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
  return
      anArgOpt
         << AOpt2007(mLevMin,"LevMin","Min level of bench",{{eTA2007::HDV}})
         << AOpt2007(mPat,"PatBench","Pattern filtering exec bench, use XXX to get existing ones",{{eTA2007::HDV}})
         << AOpt2007(mKeyBug,"KeyBug","Key for forcing bug")
         << AOpt2007(mShow,"Show","Show mesg, Def=true if PatBench init")
         << AOpt2007(mNumBugRecall,"NBR","Num to Generate a Bug in Recall,(4 manuel inspection of log file)")
         << AOpt2007(mDoBUSD,"DoBUSD","Do BenchUnbiasedStdDev (which currently dont work) ? ",{{eTA2007::HDV}})
  ;
}

cAppli_MMVII_Bench::cAppli_MMVII_Bench (const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli    (aVArgs,aSpec),
  mLevMin         (0),
  mShow           (false),
  mPat            (".*"),
  mNumBugRecall   (-1),
  mDoBUSD         (false)
{
}


void BenchFilterLinear();

cAppliBenchAnswer cAppli_MMVII_Bench::BenchAnswer() const 
{
   return cAppliBenchAnswer(true,0.0);  // Time difficult to define later
}



int  cAppli_MMVII_Bench::Exe()
{
    if (The_MMVII_DebugLevel < The_MMVII_DebugLevel_InternalError_tiny)
    {
        StdOut() << "WARNN  MMVII Bench requires highest level of debug " << std::endl ;
    }
    else
    {
        /*
      MMVII_INTERNAL_ASSERT_always
      (
            The_MMVII_DebugLevel >= The_MMVII_DebugLevel_InternalError_tiny,
            "MMVII Bench requires highest level of debug"
      );
 */
    }

    if (!IsInit(&mShow))
      mShow =  IsInit(&mPat); // Becoz, if mPat init, few bench => we can display msg

   cParamExeBench aParam(mPat,mKeyBug,mLevMin,mShow);

   for (int aLev=mLevMin ; aLev<mLevelMax ; aLev++)
   {
        StdOut() << "=====  RUN BENCH AT LEVEL " << aLev << "========" << std::endl;
        ExecuteBench(aParam);

        // No bench where executed
        if ((aLev==mLevMin) && ((aParam.NbExe()==0) || (IsInit(&mKeyBug))))
        {
             aParam.ShowIdBench();
             return EXIT_SUCCESS;
        }

        aParam.IncrLevel();
   }
   return EXIT_SUCCESS;
}

// Test handling, the test consist to confirm that ErrHanldOneAndOne was
// executed, this is done by checking that StrTestErrHandling was set to MesUneAndUne
static std::string MesUneAndUne="1+1!=3";
static std::string  StrTestErrHandling;
static void ErrHanldOneAndOne(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
   StrTestErrHandling = aMes;
}




int  cAppli_MMVII_Bench::ExecuteBench(cParamExeBench & aParam)
{
   //  On teste les macro d'assertion
   MMVII_INTERNAL_ASSERT_bench((1+1)==2,"Theoreme fondamental de l'arithmetique");
   // La on a verifie que des assertion fausses génère une erreur
   {
       MMVII_SetErrorHandler(ErrHanldOneAndOne);
       MMVII_INTERNAL_ASSERT_bench((1+1)==3,MesUneAndUne); // Theoreme  pas tres fondamental de l'arithmetique
       MMVII_RestoreDefaultHandle();
       MMVII_INTERNAL_ASSERT_bench(StrTestErrHandling==MesUneAndUne,"Error handling");
   }

   // Begin with purging directory
   CreateDirectories(TmpDirTestMMVII(),true );
   RemoveRecurs(TmpDirTestMMVII(),true,false);


   {
        //==== Bench_0000 bench on very basic support functionnalities

        Bench_Random(aParam);  // Bench random generator, check they are acceptably unbiased

        // Test on split Dir/File, string op,
        Bench_0000_SysDepString(aParam);
        Bench_0000_String(aParam);
        Bench_0000_Memory(aParam);
        Bench_0000_Param(aParam);
        Bench_0000_Ptxd(aParam);

        Bench_Duration(aParam);
        BenchStrIO(aParam);

        //==== Bench on general support services

	BenchcNewReadFilesStruct(aParam);
        // Make bench on STL, this can be thing I want to be 100% sure on STL behaviour
        // or test some extension I added in STL like spirit
        BenchSTL_Support(aParam);

        BenchEnum(aParam); // Read/Write of enum for which it exist
        this->BenchFiles(aParam); // Creation deletion of file
        Bench_Nums(aParam); // Basic numericall services
        BenchKTHVal(aParam);
        BenchCurveDigit(aParam);
        BenchHamming(aParam);
        BenchPolynome(aParam);
        BenchInterpol(aParam);
        BenchPoseEstim(aParam);
        BenchRansSubset(aParam);
        BenchRecall(aParam,mNumBugRecall); // Force MMVII to generate call to itself
        BenchSet(aParam,DirTestMMVII());  // Set (in extension)
        BenchSelector(aParam,DirTestMMVII());  // Set (in comprehension)

        Bench_Heap(aParam); // Basic numericall services

	Bench_SetI(aParam); // Bench manip on set of integers

           // Check read/write of object usign serialization
        BenchSerialization(aParam,DirTestMMVII()+"Tmp/",DirTestMMVII()+"Input/");
        //====  MORE CONSISTENT BENCH

        BenchPly(aParam);
	BenchMeshDev(aParam);
        BenchTri2D(aParam);
        BenchDelaunay(aParam);
        // Test Fast Tree Dist
        BenchFastTreeDist(aParam);

        // Test derivation with Jets
        BenchMyJets(aParam);
        BenchJetsCam(aParam);

        // Test extremum computation on images, or 3 images (case of multi scale),
        // seems easy  but rigourous handling of equality
        BenchExtre(aParam);

        // Test some matrix op : QR, EigenSym ....
        BenchDenseMatrix0(aParam);

        // Test SysCo
        BenchSysCo(aParam);

        // Test topo compensation
        BenchTopoComp(aParam);

        // Call several test on images : File, RectObj, Im1D, Im2D, BaseImage
        BenchGlobImage(aParam);
        
        BenchFilterImage1(aParam);
        BenchFilterLinear(aParam);

        // Test in fact that the exp filter have the expected sigma
        BenchStat(aParam);
        // As symb deriv must be able to compute without MMVII, pass the parameters
        if (aParam.NewBench("FormalDerivative"))
        {
           BenchFormalDer(aParam.Level(),aParam.Show());
           aParam.EndBench();
        }

        //  ====== NOW  funcion that do no bench but do some test an print
        //  ====== message that may be informative

        // Inspect cube is funcion that do no bench, maybe usefull for inspect symbol der
        // with a simple cubic fonction
        if (aParam.NewBench("InspectCube",true))
        {
           InspectCube();
           aParam.EndBench();
        }

        // Test vector operation efficiency (using 4 eigen and others)
        BenchCmpOpVect(aParam);

        // Test geometric basic
        BenchGeom(aParam);

        // Test mapping Buf/NotBuf  Jacob  Inverse ...
        BenchMapping(aParam);

        // Apparently this bench do not succeed; to see later ?
        if (mDoBUSD)
        {
           BenchUnbiasedStdDev();
        }

        BenchSSRNL(aParam);
        BenchDeformIm(aParam);

        BenchClino(aParam);

	BenchCentralePerspective(aParam);
	cImageRadiomData::Bench(aParam);

	BenchL1Solver(aParam);
	Bench_MatEss(aParam);
	Bench_SpatialIndex(aParam);
	Bench_ToHomMult(aParam);
        BenchLinearConstr(aParam);
    }

    // Now call the bench of all application that define their own bench
    for (const auto & aSpec : cSpecMMVII_Appli::VecAll())
    {
       // Avoid recursive call
       if (aSpec->Name() != mSpecs.Name())
       {
          // Not really necessary to init, but no bad ...
          std::vector<std::string> aVArgs = {FullBin()," "+ aSpec->Name()};
          tMMVII_UnikPApli anAppli = aSpec->Alloc()(aVArgs,*aSpec);
          anAppli->SetNot4Exe();

          if (anAppli->BenchAnswer().HasBench())
          {
              if (aParam.NewBench(aSpec->Name()))
              {
                  anAppli->ExecuteBench(aParam);
              //  StdOut()  << aSpec->Name() << " => " << aSpec->Comment() << std::endl;
                  aParam.EndBench();
              }
          }
       }
    }



        // We clean the temporary files created
   RemoveRecurs(TmpDirTestMMVII(),true,false);


   ResetToFileIfFirstime<cPerspCamIntrCalib>();




   //NS_MMVII_FastTreeDist::AllBenchFastTreeDist(true);
/*




   //  TestTimeV1V2(); => Valide ratio ~=  1

*/


   return EXIT_SUCCESS;
}


static void CreateFile(const std::string & aNameFile)
{
   cMMVII_Ofs aFile(aNameFile, eFileModeOut::CreateBinary);
   int anI=44;
   aFile.Write(anI);
}

void cAppli_MMVII_Bench::BenchFiles(cParamExeBench & aParam)
{
   if (! aParam.NewBench("Files")) return;

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

   aParam.EndBench();
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
        // HERE
        RemovePatternFile(aDirT+".*",true);
    }

    // to create a file with Num in name
    { 
       std::string aNameF = aDirT + ToStr(mNum) + ".txt";
       cMMVII_Ofs (aNameF, eFileModeOut::CreateText);
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
      {eApF::Test,eApF::NoGui},
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

void BenchRecall(cParamExeBench & aParam,int aNum)
{
     if (! aParam.NewBench("Recall")) return;

     OneBenchRecall(true,-1);
     OneBenchRecall(false,aNum);

     aParam.EndBench();
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
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        std::string mMsg;
        bool   mMMV1_GenCodeTestCam;
        cPt3di mDegDistTest;
};

cCollecSpecArg2007 & cAppli_MPDTest::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return      anArgObl
	      <<  Arg2007(mMsg ,"Message (for test)")
      ;

}
cCollecSpecArg2007 & cAppli_MPDTest::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
  return
      anArgOpt
         << AOpt2007(mMMV1_GenCodeTestCam,"V1_GCTC","Generate code for Test Cam")
         << AOpt2007(mDegDistTest,"DDT","Degree Distorion Test")
  ;
}

cAppli_MPDTest:: cAppli_MPDTest(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli     (aVArgs,aSpec),
  mMMV1_GenCodeTestCam  (false)
{
}



void TestArg0(const std::vector<int> & aV0)
{
   for (auto I : aV0){I++; StdOut() << "I=" << I << std::endl; }
}


std::string BUD(const std::string & aDir);
void TestBooostIter();

class cTestShared
{
    public :
        cTestShared()  {StdOut()  << "CREATE cTestShared "<< this << std::endl;}
        ~cTestShared() {StdOut() << "XXXXXX cTestShared "<< this << std::endl;}
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
    StdOut() << "BEGIN TBOOL " << std::endl; getchar();

    for (int aK=0 ; aK<5000 ; aK++)
    {
        std::vector<bool> * aV = new std::vector<bool>;
        for (int aK=0 ; aK<1000000 ; aK++)
           aV->push_back(true);
    }
    StdOut() << "END TBOOL " << std::endl; getchar();
    for (int aK=0 ; aK<5000 ; aK++)
    {
        std::vector<tU_INT1> * aV = new std::vector<tU_INT1>;
        for (int aK=0 ; aK<1000000 ; aK++)
           aV->push_back(1);
    }
    StdOut() << "END TBYTE " << std::endl; getchar();
}

bool PrintAndTrue(const std::string & aMes) 
{
    return true;
}

void ShowAdr(double & anAdr)
{
       StdOut () <<  "ADDDDDr " << &(anAdr) << std::endl;
}
void TTT();
void TestDNA();

/** A test to experientally check a hypothesis on "quaternion" , maybe obvious, but not at ease with it, and
better safe than sorry ...  Let RI,RJ and RK be the 3 rotation about mains axes, and Id the identity matrix,
we check that for any rotation matrix R, there  exist a,b,c,d such than :

    R = a Id + b RI + c RJ + d RK

*/

void TestQuat()
{
    std::vector<tRotR> aBQ{tRotR::Identity()} ;
    for (int aK=0 ; aK< 3; aK++)
    {
        aBQ.push_back(tRotR::RotArroundKthAxe(aK));

       aBQ.back().Mat().Show();
       StdOut() <<  "=======================\n";
    }

    for (int aKT=0 ; aKT<100 ; aKT++)
    {
         cDenseMatrix<tREAL8>  aSum1(3,eModeInitImage::eMIA_Null);
         for (const auto & aK : {0,1,2,3})
         {
             aSum1 = aSum1 +  (aBQ[aK].Mat() * RandUnif_C());
         }

         cDenseMatrix<tREAL8> aStS = aSum1 * aSum1.Transpose();

         aStS.Show();
         StdOut() <<  "=======================\n";
    }
    StdOut() <<  "TestQuatTestQuat\n"; getchar();
}

// #include <limits>
int cAppli_MPDTest::Exe()
{
   if (0)
   {
	   TestQuat();
	   return EXIT_SUCCESS;
   }
   if (1)
   {
       TestDNA();
   }
   if (1)
   {
       StdOut()  <<   " ================  TEST INIT & SPEC ===================" << std::endl;
       int aVExt;
       StdOut() <<  "-VExt ## InSpec: " << IsInSpec(&aVExt)        << " SpecObl: " << IsInSpecObl(&aVExt)  
	       << " SpecFac: " << IsInSpecFac(&aVExt) << " IsInit: " << IsInit(&aVExt) << "\n";
       StdOut() <<  "- MSG ## InSpec: " << IsInSpec(&mMsg)        << " SpecObl: " << IsInSpecObl(&mMsg) 
	       << " SpecFac: " << IsInSpecFac(&mMsg) << " IsInit: " << IsInit(&mMsg) << "\n";
       StdOut() <<  "- Deg ## InSpec: " << IsInSpec(&mDegDistTest)<< " SpecObl: " << IsInSpecObl(&mDegDistTest) 
	       << " SpecFac: " << IsInSpecFac(&mDegDistTest) << " IsInit: " << IsInit(&mDegDistTest) << "\n";

       getchar();
   }
   if (IsInit(&mDegDistTest))
   {
      for (auto isFraserMode : {true,false})
      {
          std::vector<cDescOneFuncDist>  aVD =  DescDist(mDegDistTest,isFraserMode);

	  StdOut() << "============== FraserMode : " << isFraserMode << " ==================\n";
          for (const auto & aDesc : aVD)
	      StdOut() << " "  << aDesc.mName <<  " " << aDesc.mLongName << std::endl;

      }
      return EXIT_SUCCESS;
   }
   TTT ();
   {
     StdOut() << "T0:" << cName2Calc<double>::CalcFromName("toto",10,true) << std::endl;
     StdOut() << "T1:" << cName2Calc<double>::CalcFromName("EqDist_Dist_Rad3_Dec1_XY1",10) << std::endl;
      return EXIT_SUCCESS;
   }
}

tMMVII_UnikPApli Alloc_MPDTest(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MPDTest(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMPDTest
(
     "TestMPD",
      Alloc_MPDTest,
      "This used a an entry point to all quick and dirty test by MPD ...",
      {eApF::Test,eApF::NoGui},
      {eApDT::None},
      {eApDT::Console},
      __FILE__
);


};

