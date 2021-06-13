#include "TestCodeGenTpl.h"
#include "CodeGen_IncludeAll.h"

#include "Formula_Fraser_Test.h"
#include "Formula_Primitives_Test.h"
#include "Formula_Ratkowskyresidual.h"
#include "Formula_Eqcollinearity.h"


namespace SD = NS_SymbolicDerivative;


class  TestPrimitives : public cCodeGenTest<cPrimitivesTest,SD::cPrimitivesTest_ValAndDer,SD::cPrimitivesTest_ValAndDerLongExpr>
{
public:
    TestPrimitives(size_t nbTest) : cCodeGenTest<cPrimitivesTest,SD::cPrimitivesTest_ValAndDer,SD::cPrimitivesTest_ValAndDerLongExpr>(nbTest)
    {
        mVUk[0] = 1;
        mVUk[1] = 2;
    }
};




static int doTest(int sizeBuf, int nbThreads, const std::string& name)
{
//    TestFraser test(1000000);
//   TestPrimitives test(1000000);
//    TestRatkowsky test(1000000);

    std::vector<double>  uk(cEqCoLinearity<cTplPolDist<7>>::NbUk(),0);
    std::vector<double> obs(cEqCoLinearity<cTplPolDist<7>>::NbObs(),0);
    uk[11] = 1.0;
    // In obs, we set the current matrix to Id
    obs[0] = obs[4] = obs[8] = 1;

    // Here we initiate with "peferct" projection, to check something
    // Fix X,Y,Z
    uk[0] = 1.0;
    uk[1] = 2.0;
    uk[2] = 10.0;
    // Fix I,J (pixel projection)
    obs[ 9] = 0.101;
    obs[10] = 0.2;

//     cCodeGenTest<cFraserCamColinear,SD::cFraser,SD::cFraserLongExpr> test(1000000);
    cCodeGenTest<cEqCoLinearity<cTplPolDist<7>>,SD::cEqColLinearityXYPol_Deg7_ValAndDer,SD::cEqColLinearityXYPol_Deg7_ValAndDerLongExpr> test(100000,name);
//     cCodeGenTest<cEqCoLinearity<cTplFraserDist>,SD::cEqColLinearityFraser,SD::cEqColLinearityFraserLongExpr> test(1000000);
     test.mVUk = uk;
     test.mVObs =  obs;
     test.checkAll();
    if (sizeBuf == 0 || nbThreads==0) {
        test.benchMark();
    } else {
        test.oneShot(nbThreads,sizeBuf);
    }
   return EXIT_SUCCESS;
}


int main(int argc, char *argv[])
{
    int opt;
    int sizeBuf=0;
    int nbThreads=0;
    std::string name;

    while ((opt = getopt(argc, argv, "b:t:h")) != -1) {
          switch (opt) {
          case 'b':
              sizeBuf = atoi(optarg);
              break;
          case 't':
              nbThreads = atoi(optarg);
              break;
          default: /* '?' */
              std::cerr << "Usage: " << argv[0] <<  " [-b SizeBuf -t NbThreads] [nom test]\n";
              exit(EXIT_FAILURE);
          }
    }
    if (sizeBuf == 0 && nbThreads == 0) {
        if (optind < argc) {
            name = argv[optind];
            optind++;
        }
    }
    if (((sizeBuf==0) != (nbThreads==0)) || optind < argc) {
        std::cerr << "Usage: " << argv[0] <<  " [-b SizeBuf -t NbThreads] [nom test]\n";
        exit(EXIT_FAILURE);
    }
    return doTest(sizeBuf,nbThreads,name);
}
