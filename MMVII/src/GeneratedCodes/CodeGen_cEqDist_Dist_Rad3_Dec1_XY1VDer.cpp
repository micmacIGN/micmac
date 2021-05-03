#include "CodeGen_cEqDist_Dist_Rad3_Dec1_XY1VDer.h"

namespace NS_SymbolicDerivative {

void cEqDist_Dist_Rad3_Dec1_XY1VDer::DoEval()
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t aK=0; aK < this->mNbInBuf; aK++) {
// Declare local vars in loop to make them per thread
    double &xPi = this->mVUk[aK][0];
    double &yPi = this->mVUk[aK][1];
    double &K1 = this->mVObs[aK][0];
    double &K2 = this->mVObs[aK][1];
    double &K3 = this->mVObs[aK][2];
    double &p1 = this->mVObs[aK][3];
    double &p2 = this->mVObs[aK][4];
    double &b2 = this->mVObs[aK][5];
    double &b1 = this->mVObs[aK][6];
    double F66 = (yPi + yPi);
    double F35 = (b1 * xPi);
    double F34 = (b2 * yPi);
    double F44 = (xPi + xPi);
    double F58 = (2 * yPi);
    double F78 = (2 * xPi);
    double F14 = (xPi * yPi);
    double F13 = (yPi * yPi);
    double F12 = (xPi * xPi);
    double F95 = (F78 * p1);
    double F70 = (F66 * K1);
    double F60 = (F44 * 2);
    double F80 = (F66 * p1);
    double F59 = (F58 * p2);
    double F79 = (F78 * p2);
    double F30 = (F13 * 2);
    double F48 = (F44 * K1);
    double F85 = (F44 * p2);
    double F36 = (F34 + F35);
    double F86 = (F58 * p1);
    double F92 = (F66 * 2);
    double F23 = (F12 * 2);
    double F15 = (F12 + F13);
    double F26 = (F14 * 2);
    double F28 = (F26 * p2);
    double F61 = (F44 + F60);
    double F18 = (F15 * K1);
    double F16 = (F15 * F15);
    double F67 = (F15 * F66);
    double F24 = (F23 + F15);
    double F81 = (F80 + F79);
    double F93 = (F66 + F92);
    double F27 = (F26 * p1);
    double F45 = (F15 * F44);
    double F31 = (F15 + F30);
    double F87 = (F85 + F86);
    double F17 = (F15 * F16);
    double F73 = (F16 * F66);
    double F25 = (F24 * p1);
    double F68 = (F67 + F67);
    double F32 = (F31 * p2);
    double F94 = (F93 * p2);
    double F51 = (F16 * F44);
    double F62 = (F61 * p1);
    double F19 = (F16 * K2);
    double F46 = (F45 + F45);
    double F69 = (F68 * K2);
    double F29 = (F25 + F28);
    double F33 = (F27 + F32);
    double F50 = (F15 * F46);
    double F72 = (F15 * F68);
    double F47 = (F46 * K2);
    double F20 = (F18 + F19);
    double F21 = (F17 * K3);
    double F96 = (F95 + F94);
    double F63 = (F59 + F62);
    double F74 = (F72 + F73);
    double F49 = (F48 + F47);
    double F52 = (F50 + F51);
    double F71 = (F70 + F69);
    double F22 = (F21 + F20);
    double F53 = (F52 * K3);
    double F75 = (F74 * K3);
    double F41 = (F22 * yPi);
    double F37 = (F22 * xPi);
    double F42 = (F41 + yPi);
    double F38 = (F37 + xPi);
    double F76 = (F71 + F75);
    double F54 = (F49 + F53);
    double F84 = (F54 * yPi);
    double F89 = (F76 * yPi);
    double F43 = (F33 + F42);
    double F77 = (F76 * xPi);
    double F55 = (F54 * xPi);
    double F39 = (F29 + F38);
    double F90 = (F22 + F89);
    double F40 = (F39 + F36);
    double F88 = (F84 + F87);
    double F82 = (F77 + F81);
    double F56 = (F22 + F55);
    double F57 = (F56 + 1);
    double F91 = (F90 + 1);
    double F83 = (F82 + b2);
    double F64 = (F57 + F63);
    double F97 = (F91 + F96);
    double F65 = (F64 + b1);
    this->mBufLineRes[aK][0] = F40;
    this->mBufLineRes[aK][1] = F65;
    this->mBufLineRes[aK][2] = F83;
    this->mBufLineRes[aK][3] = F43;
    this->mBufLineRes[aK][4] = F88;
    this->mBufLineRes[aK][5] = F97;
  }
}

cCompiledCalculator<double> * Alloc_EqDist_Dist_Rad3_Dec1_XY1VDer(int aSzBuf)
{
   return new cEqDist_Dist_Rad3_Dec1_XY1VDer(aSzBuf);
}

cName2Calc<double> TheNameAlloc_EqDist_Dist_Rad3_Dec1_XY1VDer("EqDist_Dist_Rad3_Dec1_XY1VDer",Alloc_EqDist_Dist_Rad3_Dec1_XY1VDer);

} // namespace NS_SymbolicDerivative
