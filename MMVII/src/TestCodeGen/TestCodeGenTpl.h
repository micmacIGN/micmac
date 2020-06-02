#ifndef TESTCODEGENTPL_H
#define TESTCODEGENTPL_H

#ifdef _OPENMP
#include "omp.h"
#endif

#include "ChronoBench.h"
#include "include/SymbDer/SymbolicDerivatives.h"

#include <ceres/jet.h>

using ceres::Jet;
namespace  FD = NS_MMVII_FormalDerivative;

// ========== Define on Jets two optimization as we did on formal

template <typename T, int N> inline Jet<T, N> square(const Jet<T, N>& f)
{
  return Jet<T, N>(FD::square(f.a), (2.0*f.a) * f.v);
}

template <typename T, int N> inline Jet<T, N> cube(const Jet<T, N>& f)
{
  T a2 = FD::square(f.a);
  return Jet<T, N>(f.a*a2, (3.0*a2) * f.v);
}


template <typename T, int N> inline Jet<T, N> powI(const Jet<T, N>& aJ,const int & aExp)
{
   // In this case avoid compute 1/x and multiply by x
   if (aExp==0) return Jet<T,N>(1.0);

   // make a single computation of pow
   T aPm1 = FD::powI(aJ.a,aExp-1);
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
    cCodeGenTest(unsigned nbTest) : mVUk(TheNbUk,0.0),mVObs (TheNbObs,0.0),mNbTestTarget(nbTest),mNbTest(nbTest),mSizeBuf(1)
 {
    static_assert(EQDEF::NbUk() == EQNADDR::NbUk(),"Test codegen: incompatible interpreted and N-Addr compiled formula");
    static_assert(EQDEF::NbUk() == EQFORM::NbUk(),"Test codegen: incompatible interpreted and compiled formula");
    static_assert(EQDEF::NbObs() == EQNADDR::NbObs(),"Test codegen: incompatible interpreted and N-Addr compiled formula");
    static_assert(EQDEF::NbObs() == EQFORM::NbObs(),"Test codegen: incompatible interpreted and compiled formula");
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

    typedef typename EQNADDR::ResType ResType;

    void checkVsJet(const std::string& name, const ResType &values);

    template<class FORMULA>
    void codeGenTestCGen(Bench &bench,const std::string &name);

    void codeGenTestJets(Bench &bench);
    void codeGenTestDyn(Bench &bench);
    void codeGenTestNAddr(Bench &bench);
    void codeGenTestForm(Bench &bench);

    static inline bool almostEqual(const double & aV1,const double & aV2,const double & aEps)
    {
       return std::abs(aV1-aV2) <= aEps*(std::abs(aV1)+std::abs(aV2));
    }


    template<typename T>
    ResType CFDtoArray(const FD::cCoordinatorF<T>& cfd, size_t n) {
        ResType val;
        size_t step = TheNbUk + 1;
        size_t nVal = val.size() / step;

        for (size_t i=0; i< nVal; i++) {
            val[i*step] = cfd.ValComp(n,i);
            for (size_t j=0; j<TheNbUk; j++)
                val[i*step + j + 1] = cfd.DerComp(n,i,j);
        }
        return val;
    }

    unsigned mNbTestTarget;
    unsigned mNbTest;
    size_t mSizeBuf;

    static const std::vector<std::string> testNames;
};

template<class EQDEF, class EQNADDR, class EQFORM>
const std::vector<std::string>
cCodeGenTest<EQDEF,EQNADDR,EQFORM>::testNames = {"Jet","Buf","NAd","Fml"};


template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::checkVsJet(const std::string& name, const ResType &val)
{
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
    ResType val;

    // Check dynamic evaluation
    FD::cCoordinatorF<double>  mCFD(BUF_SIZE,EQDEF::VNamesUnknowns(),EQDEF::VNamesObs());
    auto aVFormula = EQDEF::formula(mCFD.VUk(),mCFD.VObs());
    mCFD.SetCurFormulasWithDerivative(aVFormula);
    for (size_t i=0; i<mCFD.SzBuf(); i++)
        mCFD.PushNewEvals(mVUk,mVObs);
    mCFD.EvalAndClear();

    val=CFDtoArray(mCFD,0);
    checkVsJet(testNames[1],val);
    for (size_t i=0; i< mCFD.SzBuf(); i++) {
        if (CFDtoArray(mCFD,i) != val)
            std::cerr << testNames[1] << ": Error buffer: values[" << i << "] != values[0]\n";
    }

    // Check codegen N-Addr
    EQNADDR formNAddr(BUF_SIZE);
    for (size_t i=0; i<formNAddr.bufferSize(); i++)
        formNAddr.pushNewEvals(mVUk,mVObs);
    formNAddr.evalAndClear();
    checkVsJet(testNames[2],formNAddr.result()[0]);
    auto nAddrRes = formNAddr.result();
    for (size_t i=0; i< nAddrRes.size(); i++) {
        if (nAddrRes[i] != nAddrRes[0])
            std::cerr << testNames[2] << ": Error buffer: values[" << i << "] != values[0]\n";
    }

    // Check codegen formula
    EQFORM formForm(BUF_SIZE);
    for (size_t i=0; i<formForm.bufferSize(); i++)
        formForm.pushNewEvals(mVUk,mVObs);
    formForm.evalAndClear();
    checkVsJet(testNames[3],formForm.result()[0]);
    auto formRes = formForm.result();
    for (size_t i=0; i< formRes.size(); i++) {
        if (formRes[i] != formRes[0])
            std::cerr << testNames[3] << ": Error buffer: values[" << i << "] != values[0]\n";
    }
}


template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::codeGenTestJets(Bench &bench)
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
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::codeGenTestDyn(Bench &bench)
{
    FD::cCoordinatorF<double>  mCFD(mSizeBuf,EQDEF::VNamesUnknowns(),EQDEF::VNamesObs());

    bench.start();
    bench.start(1);
    auto aVFormula = EQDEF::formula(mCFD.VUk(),mCFD.VObs());
    mCFD.SetCurFormulasWithDerivative(aVFormula);
    bench.stop(1);
    for (unsigned aK=0 ; aK<mNbTest / mSizeBuf ; aK++)
    {
        // Fill the buffers with data
        bench.start(2);
        for (size_t aKInBuf=0 ; aKInBuf<mSizeBuf ; aKInBuf++)
            mCFD.PushNewEvals(mVUk,mVObs);
        bench.stopStart(2,3);
        // Evaluate the derivate once buffer is full
        mCFD.EvalAndClear();
        bench.stop(3);
    }
    bench.stop();
}


template<class EQDEF, class EQNADDR, class EQFORM>
template<class FORMULA>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::codeGenTestCGen(Bench &bench, const std::string& name)
{
    FORMULA formula(0);
    typename FORMULA::UkType aVUk;
    typename FORMULA::ObsType aVObs;
    for (size_t i=0; i<aVUk.size(); i++)
        aVUk[i] = mVUk[i];
    for (size_t i=0; i<aVObs.size(); i++)
        aVObs[i] = mVObs[i];

    bench.start();
    bench.start(1);
    formula = FORMULA(mSizeBuf);
    bench.stop(1);
    for (unsigned aK=0 ; aK<mNbTest / mSizeBuf ; aK++)
    {
        // Fill the buffers with data
        bench.start(2);
        for (size_t aKi=0 ; aKi<mSizeBuf ; aKi++)
            formula.pushNewEvals(aVUk,aVObs);
        bench.stopStart(2,3);
        formula.evalAndClear();
        bench.stop(3);
    }
    bench.stop();
}

template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::codeGenTestNAddr(Bench &bench)
{
    this->codeGenTestCGen<EQNADDR>(bench,"NAddr");
}


template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::codeGenTestForm(Bench &bench)
{
    this->codeGenTestCGen<EQFORM>(bench,"Form");
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

    for (size_t i=0; i<testNames.size(); i++) {
        switch(i) {
        case 0: codeGenTestJets(bench); break;
        case 1: codeGenTestDyn(bench); break;
        case 2: codeGenTestNAddr(bench); break;
        case 3: codeGenTestForm(bench); break;
        }
        std::cout << testNames[i] << ": " <<bench.currents() << "\n";
        bench.reset();
    }
}



template<class EQDEF, class EQNADDR, class EQFORM>
void cCodeGenTest<EQDEF,EQNADDR,EQFORM>::benchMark(void)
{
//    static const size_t NbTest=20;
    static const size_t NbTest=10;

    for (size_t test=0; test <testNames.size(); test++) {
        std::vector<size_t> nb_buffer;
        std::vector<size_t> nb_thread;

        std::cout << "** " << EQDEF::FormulaName() << "  " << testNames[test] << "\n";
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

        std::ofstream os(EQDEF::FormulaName() + "_"  + testNames[test] + ".txt");
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
                std::string testName = testNames[test];
                if (test!=0)
                    testName += " B:" + std::to_string(bufs) + " T:" + std::to_string(threads);
                testName += " ";

                while(1) {
                    for (size_t n=0; n<NbTest; n++)  {
                        switch(test) {
                        case 0: codeGenTestJets(bench); break;
                        case 1: codeGenTestDyn(bench); break;
                        case 2: codeGenTestNAddr(bench); break;
                        case 3: codeGenTestForm(bench); break;
                        }
                        std::cout << testName << bench.currents() << "\n";
                        bench.next();
                    }
                    auto filter=bench.filter();
                    std::cout << testName << "Main Mean:" << filter.mean << " sigma:" << filter.stddev << "\n";
                    if (filter.stddev < filter.mean / 10)
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
