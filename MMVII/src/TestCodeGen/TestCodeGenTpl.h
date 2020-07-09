#ifndef TESTCODEGENTPL_H
#define TESTCODEGENTPL_H

#ifdef _OPENMP
#include "omp.h"
#endif

#include "ChronoBench.h"
#include "SymbDer/SymbolicDerivatives.h"

#include <ceres/jet.h>

using ceres::Jet;
namespace  SD = NS_SymbolicDerivative;

// ========== Define on Jets two optimization as we did on formal

template <typename T, int N> inline Jet<T, N> square(const Jet<T, N>& f)
{
  return Jet<T, N>(SD::square(f.a), (2.0*f.a) * f.v);
}

template <typename T, int N> inline Jet<T, N> cube(const Jet<T, N>& f)
{
  T a2 = SD::square(f.a);
  return Jet<T, N>(f.a*a2, (3.0*a2) * f.v);
}


template <typename T, int N> inline Jet<T, N> powI(const Jet<T, N>& aJ,const int & aExp)
{
   // In this case avoid compute 1/x and multiply by x
   if (aExp==0) return Jet<T,N>(1.0);

   // make a single computation of pow
   T aPm1 = SD::powI(aJ.a,aExp-1);
   return Jet<T,N>(aJ.a*aPm1,(aExp*aPm1)*aJ.v);
}

template <class T,const int N> Jet<T, N>  CreateCste(const T & aV,const Jet<T, N>&)
{
    return Jet<T, N>(aV);
}



typedef ChronoBench<3> Bench;
static std::ostream &operator<<(std::ostream &os, const Bench::Times &t)
{
    os << "[" ;
    for (size_t i=0; i<t.size(); i++)
        os << t[i] << (i<t.size()-1 ? "," : "");
    os << "]" ;
    return os;
}


template<class EQDEF, class EQNADDR, class EQFORM>
class cCodeGenTest
{
public:
    cCodeGenTest(unsigned nbTest, const std::string& aName="") : mVUk(TheNbUk,0.0),mVObs (TheNbObs,0.0),mNbTestTarget(nbTest),mNbTest(nbTest),mSizeBuf(1),filePrefix(aName)
 {
        if (filePrefix.size())
            filePrefix += "_";
//    static_assert(EQDEF::NbUk() == EQNADDR::NbUk(),"Test codegen: incompatible interpreted and N-Addr compiled formula");
//    static_assert(EQDEF::NbUk() == EQFORM::NbUk(),"Test codegen: incompatible interpreted and compiled formula");
//    static_assert(EQDEF::NbObs() == EQNADDR::NbObs(),"Test codegen: incompatible interpreted and N-Addr compiled formula");
//    static_assert(EQDEF::NbObs() == EQFORM::NbObs(),"Test codegen: incompatible interpreted and compiled formula");
 }
    void setSizeBuf(size_t aSizeBuf) { mSizeBuf = aSizeBuf ; mNbTest = (mNbTestTarget / mSizeBuf) * mSizeBuf; }
    void oneShot(int numThreads, int sizeBuf);
    void benchMark(void);
    void checkAll();

    std::vector<double> mVUk;
    std::vector<double> mVObs;

protected:
    static const auto TheNbUk = EQDEF::NbUk();
    static const auto TheNbObs = EQDEF::NbObs();

    enum Tests {Jets, Dyn, NAddr, Devel};
    static constexpr std::initializer_list<Tests> allTests = {Jets,Dyn,NAddr,Devel};

    typedef typename EQNADDR::tOneRes tOneRes;

    void checkVsJet(Tests test, const tOneRes &values);

    void TestSD(Tests test, Bench &bench);
    void TestJets(Bench &bench);

    static inline bool almostEqual(const double & aV1,const double & aV2,const double & aEps)
    {
       return std::abs(aV1-aV2) <= aEps*(std::abs(aV1)+std::abs(aV2));
    }

    unsigned mNbTestTarget;
    unsigned mNbTest;
    size_t mSizeBuf;
    std::string filePrefix;

    static const std::array<const std::string,allTests.size()> testNames ;
};

template<class EQDEF, class EQNADDR, class EQFORM>
const std::array<const std::string,cCodeGenTest<EQDEF,EQNADDR,EQFORM>::allTests.size()>
cCodeGenTest<EQDEF,EQNADDR,EQFORM>::testNames = {"Jet","Buf","NAd","Fml"};

template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::checkVsJet(Tests test, const tOneRes &val)
{
    auto name = testNames[test];
    // Verif resultats
    typedef Jet<double,TheNbUk> tJets;
    std::vector<tJets> aVJetUk;
    for (int aK=0 ; aK<TheNbUk ; aK++)
        aVJetUk.push_back(tJets(mVUk[aK],aK));
    auto aJetRes = EQDEF::formula(aVJetUk,mVObs);
    size_t step = TheNbUk + 1;

    for (int aKVal=0 ; aKVal<int(aJetRes.size()) ; aKVal++)
    {
        if (! almostEqual(aJetRes[aKVal].a,val[aKVal*step],1e-5)) {
            std::cerr << name  << ": Error for value " << aKVal;
            std::cerr << "  (Jet:" <<  aJetRes[aKVal].a << ", Diff:" << aJetRes[aKVal].a - val[aKVal*step] << ")\n" ;
        }
        for (int aKDer=0;  aKDer< TheNbUk ; aKDer++)
        {
            if (! almostEqual(aJetRes[aKVal].v[aKDer],val[aKVal*step+aKDer+1],1e-5)) {
                std::cerr << name  << ": Error for derivative #" << aKDer << " and value " << aKVal;
                std::cerr << "  (Jet:" <<  aJetRes[aKVal].v[aKDer] << ", Diff:" << aJetRes[aKVal].v[aKDer] - val[aKVal*step+aKDer+1] << ")\n" ;
            }
        }
    }
}


template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::checkAll()
{
    const size_t BUF_SIZE=256;

    std::cout << "Checking all results ..." << "\n";
    // Check EPS
    tOneRes val;
    SD::cCalculator<double> *calculator=0;


    for (auto &test : allTests) {
        switch (test) {
        case Jets: continue;
        case Dyn: {
            auto mCFD = new SD::cCoordinatorF<double>(EQDEF::FormulaName(),BUF_SIZE,EQDEF::VNamesUnknowns(),EQDEF::VNamesObs());
            auto aVFormula = EQDEF::formula(mCFD->VUk(),mCFD->VObs());
            mCFD->SetCurFormulasWithDerivative(aVFormula);
            calculator = mCFD;
            break;
        }
        case NAddr: calculator = new EQNADDR(BUF_SIZE); break;
        case Devel: calculator = new EQFORM(BUF_SIZE); break;
        }
        for (size_t i=0; i<calculator->SzBuf(); i++)
            calculator->PushNewEvals(mVUk,mVObs);
        auto res = calculator->EvalAndClear();
        checkVsJet(test,*res[0]);
        for (size_t i=0; i< res.size(); i++) {
            if (*res[i] != *res[0])
                std::cerr << this->testNames[test] << ": Error buffer: values[" << i << "] != values[0]\n";
        }
        delete calculator;
    }

}


template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::TestJets(Bench &bench)
{
    typedef Jet<double,TheNbUk> tJets;
    std::vector<tJets> aRes;

    bench.start();
    bench.start(1);
    std::vector<tJets>  aVUk;
    bench.stopStart(1,2);
    for (int aK=0 ; aK<TheNbUk ; aK++)
        aVUk.push_back(tJets(mVUk[aK],aK));
    bench.stopStart(2,3);
    for (size_t aK=0 ; aK<mNbTest ; aK++)  {
        aRes = EQDEF::formula(aVUk,mVObs);
    }
    bench.stop(3);
    bench.stop();
}

template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::TestSD(Tests test,Bench &bench)
{
    SD::cCalculator<double> *calculator=0;
    bench.start();
    bench.start(1);
    switch (test) {
    case Jets: return;
    case Dyn: {
        auto mCFD = new SD::cCoordinatorF<double>(EQDEF::FormulaName(),mSizeBuf,EQDEF::VNamesUnknowns(),EQDEF::VNamesObs());
        auto aVFormula = EQDEF::formula(mCFD->VUk(),mCFD->VObs());
        mCFD->SetCurFormulasWithDerivative(aVFormula);
        calculator = mCFD;
        break;
    }
    case NAddr: calculator = new EQNADDR(mSizeBuf); break;
    case Devel: calculator = new EQFORM(mSizeBuf); break;
    }
    bench.stop(1);
    for (unsigned aK=0 ; aK<mNbTest / mSizeBuf ; aK++)
    {
        // Fill the buffers with data
        bench.start(2);
        for (size_t aKi=0 ; aKi<mSizeBuf ; aKi++)
            calculator->PushNewEvals(mVUk,mVObs);
        bench.stopStart(2,3);
        calculator->EvalAndClear();
        bench.stop(3);
    }
    bench.stop();
    delete calculator;
}


template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::oneShot(int numThreads, int sizeBuf)
{
    Bench bench;

#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#endif
    setSizeBuf(sizeBuf);
    checkAll();

    for (auto &test : allTests) {
        if (test == Jets ) {
            TestJets(bench);
        } else {
            TestSD(test,bench);
        }
        std::cout << testNames[test] << ": " <<bench.currents() << "\n";
        bench.reset();
    }
}



template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::benchMark(void)
{
//    static const size_t NbTest=20;
    static const size_t NbTest=10;

    for (auto &test : allTests) {
        std::vector<size_t> nb_buffer;
        std::vector<size_t> nb_thread;

        std::cout << "** " << EQDEF::FormulaName() << "  " << this->testNames[test] << "\n";
        if (test == 0) {
            nb_buffer={1};
            nb_thread={1};
        } else {
//            nb_buffer={1,10,32,64,128,200,500,800,1000,2000};
            nb_buffer={200,500,800,1000,2000};
#ifdef _OPENMP
            nb_thread={1,2,3,4,5,6,7,8};
#else
            nb_thread={1};
#endif
        }

        std::ofstream os(filePrefix +  EQDEF::FormulaName() + "_"  + this->testNames[test] + ".txt");
        os << "Buf ";
        for (auto &t : nb_thread) os << t << " ";
        for (auto &t : nb_thread) os << t << " ";
        for (auto &t : nb_thread) os << t << " ";
        os << "\n";
        for (auto &bufs : nb_buffer) {
            std::vector<Bench::Times> allTimes;
            setSizeBuf(bufs);
            for (auto &threads : nb_thread) {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#endif
                Bench bench;
                std::string testName = this->testNames[test];
                if (test!=0)
                    testName += " B:" + std::to_string(bufs) + " T:" + std::to_string(threads);
                testName += " ";

                while(1) {
                    for (size_t n=0; n<NbTest; n++)  {
                        if (test == Jets)
                            TestJets(bench);
                        else
                            TestSD(test,bench);
                        std::cout << testName << bench.currents() << "\n";
                        bench.next();
                    }
                    auto filter=bench.filter();
                    std::cout << testName << "Main Mean:" << filter.mean << " sigma:" << filter.stddev << "\n";
                    if (filter.stddev < filter.mean * 0.15)
                        break;
                    std::cout << testName << "Redoing this test ...\n";
                }
                std::cout << testName << "Means: " << bench.fltMeans() << "\n\n";
                allTimes.push_back(bench.fltMeans());
            }
            os << bufs << " ";
            for (auto &t : allTimes) os << t[0] << " ";
            for (auto &t : allTimes) os << t[3] << " ";
            for (auto &t : allTimes) os << t[2] << " ";
            os << "\n";
            os.flush();
        }
    }
}

#endif // TESTCODEGENTPL_H
