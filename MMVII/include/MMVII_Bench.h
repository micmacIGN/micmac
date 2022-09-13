#ifndef  _MMVII_Bench_H_
#define  _MMVII_Bench_H_

namespace MMVII
{


class cAppliBenchAnswer;  // Appli describe their bench, if any
class cParamExeBench;     // Parameters of bench passed to functions

/** \file MMVII_Bench.h
    \brief Declare function that will be called for bench

    When MMVII will grow, most bench function will be done in file/folde
   close to the functionality. So it will be necessary to declare these function.

*/

/// With this class, an Appli indicate how it deal with Bench
class cAppliBenchAnswer
{
    public :
         cAppliBenchAnswer(bool HasBench,double Time);
         bool   HasBench() const {return mHasBench;}
         double Time()     const {return mTime; }
  
    private :
         bool   mHasBench;  ///< Has the apply  a Bench method ?
         double mTime;      ///< Time to execute, just an order of magnitude
};

/// With this class, an Appli indicate how it deal with Bench
class cParamExeBench
{
    public :
         cParamExeBench(const std::string & aPattern,const std::string & aBugKey,int aLevInit,bool Show);

         bool  NewBench(const std::string & aName,bool ExactMatch=false); ///< Memo the name, Indicate if the bench is executed, 
         bool  GenerateBug(const std::string & aKey);
         void  EndBench(); ///< Application must signal end of the bench
         bool  Show() const;   ///< Show intermediar msg, 
         int   Level() const;  ///< Bench are piped with increasing levels, higher level/ more test
         int   NbExe() const;  ///< Number of execution made
         void  Messg(const std::string &);  ///< Add messg to log file
         void  IncrLevel();  ///< Add messg to log file

         void ShowIdBench() const; ///< Print all id of bench

    private :
         std::vector<std::string>  mVallBench;  ///< Memo all name, used for users info
         std::vector<std::vector<std::string> >  mVAllBugKeys; ///< For a given bench, store the bug key for display
         std::vector<bool>         mVExactMatch;  ///< Memo exact match attribute, for print
         bool                      mInsideFunc; ///< Used to check correc use of NewBench/EndBench
         int                       mLevInit;     ///< Current level of test
         int                       mCurLev;     ///< Current level of test
         bool                      mShow;       ///< Do the function print msg on console
         int                       mNbExe;
         std::string               mName;    ///< Exact Name for exact select
         tNameSelector             mPattern;    ///< Pattern for select bench
         std::string               mBugKey;
};


void BenchFastTreeDist(cParamExeBench & aParam); ///< Test method for fast computation of dist in tree
void BenchMyJets(cParamExeBench & aParam);  ///< Test on Jets, correctness and efficience
void BenchFormalDer(int aLevel,bool show); ///< Dont import MicMac header, so dif interface
void BenchJetsCam(cParamExeBench & aParam);
void BenchSTL_Support(cParamExeBench & aParam);  ///< Test STL+support function that could/should exist in standard libs
void InspectCube();
void BenchCmpOpVect(cParamExeBench & aParam);

void BenchRecall(cParamExeBench & aParam,int NumGenerateBugRecall);       ///< Mecanism for MMVII calling itself
void Bench_0000_SysDepString(cParamExeBench & aParam); ///< String split (dir, path, file ...)
void Bench_0000_Memory(cParamExeBench & aParam); ///< Bench on memory integrity
void Bench_0000_Param(cParamExeBench & aParam);  ///< Bench on param line processing (elementary)
void Bench_0000_Ptxd(cParamExeBench & aParam);  ///< Basic Ptxd
void BenchEnum(cParamExeBench & aParam); ///< Bench on Str2E / E2Str
void BenchStrIO(cParamExeBench & aParam); ///< Test str/obj conv, specially for vectors

void Bench_Nums(cParamExeBench & aParam); ///< Bench on rounding, modulo ... basic numeric service
void BenchSet(cParamExeBench & aParam,const std::string & aDir); ///< Bench on cExtSet (set "en extension")
void BenchSelector(cParamExeBench & aParam,const std::string & aDir); ///< Bench on selecto, (set "en comprehension")
void Bench_Heap(cParamExeBench & aParam); ///< Bench on rounding, modulo ... basic numeric service

// Check conversion time/string; not sure essential, but it exist => no harm ...
void Bench_Duration(cParamExeBench & aParam);

// To inspect in detail, apparenly some bench dont work completely, after many iter=> numerical problem
void BenchDenseMatrix0(cParamExeBench & aParam); ///< Basic Vector 

// void cAppli_MMVII_Bench::Bench_0000_String(); => Bench on string-split
void BenchSerialization(cParamExeBench & aParam,const std::string & aDirOut,const std::string & aDirIn); ///< Bench on seriaization function

// Test generation of random subet for ransac
void BenchRansSubset(cParamExeBench & aParam);


//  void BenchGlob();      ///< All Bench


void Bench_Random(cParamExeBench & aParam); ///< Bench on random generator
void Bench_SetI(cParamExeBench & aParam); ///< Bench on set of int

void BenchExtre(cParamExeBench & aParam);  ///< Test Extremum computations, refinement ....
void BenchStat(cParamExeBench & aParam);

void BenchGlobImage(cParamExeBench & aParam); ///< Global bench on image
void BenchFilterImage1(cParamExeBench & aParam);
void BenchFilterLinear(cParamExeBench & aParam);
void BenchGeom(cParamExeBench & aParam);

void BenchMapping(cParamExeBench & aParam);
void BenchInvertMapping(cParamExeBench & aParam);
void BenchSymDerMap(cParamExeBench & aParam);
void BenchLeastSqMap(cParamExeBench & aParam);

void BenchDelaunay(cParamExeBench & aParam);
void BenchTri2D(cParamExeBench & aParam);
void BenchPly(cParamExeBench & aParam);
void BenchHamming(cParamExeBench & aParam);

void BenchSSRNL(cParamExeBench & aParam);  // Syst Sur Resol Non Linear
void BenchDeformIm(cParamExeBench & aParam); // using image in non-linear least square system



/* Called by BenchGlobImage */
void BenchRectObj(); ///< Global bench on image
void BenchBaseImage(); ///< Global bench on image
void BenchImNDim();
void BenchIm3D(); ///<  Bench on fulll 3D Images +  "Layer" images

void BenchGlobImage2d(); ///< Global bench on image
void BenchFileImage(); ///< Global bench on image


void TestTimeV1V2(); ///< Not a formal Bench, require visual inspection

void BenchUnbiasedStdDev();  ///< Test one specific function currently not correct, by default test not activated


void BenchJetsCam();  ///< Test specifique to camera projection
void MMV1_GenerateCodeTestCam(); ///< To generate code of derivative MMV1-like (for comparing with jets)





};

#endif  //  _MMVII_Bench_H_
