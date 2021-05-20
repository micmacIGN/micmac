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
    double F50 = (xPi + xPi);
    double F40 = (b2 * yPi);
    double F41 = (b1 * xPi);
    double F12 = (xPi * xPi);
    double F84 = (2 * xPi);
    double F64 = (2 * yPi);
    double F72 = (yPi + yPi);
    double F13 = (yPi * yPi);
    double F14 = (xPi * yPi);
    double F76 = (F72 * K1);
    double F66 = (F50 * 2);
    double F34 = (F13 * 2);
    double F65 = (F64 * p2);
    double F54 = (F50 * K1);
    double F42 = (F40 + F41);
    double F31 = (F14 * 2);
    double F85 = (F84 * p2);
    double F91 = (F50 * p2);
    double F92 = (F64 * p1);
    double F101 = (F84 * p1);
    double F98 = (F72 * 2);
    double F86 = (F72 * p1);
    double F15 = (F12 + F13);
    double F29 = (F12 * 2);
    double F99 = (F72 + F98);
    double F30 = (F29 + F15);
    double F87 = (F86 + F85);
    double F33 = (F31 * p1);
    double F35 = (F15 + F34);
    double F67 = (F50 + F66);
    double F51 = (F15 * F50);
    double F36 = (F31 * p2);
    double F93 = (F91 + F92);
    double F73 = (F15 * F72);
    double F16 = (F15 * F15);
    double F18 = (F15 * K1);
    double F68 = (F67 * p1);
    double F79 = (F16 * F72);
    double F74 = (F73 + F73);
    double F57 = (F16 * F50);
    double F38 = (F35 * p2);
    double F52 = (F51 + F51);
    double F32 = (F30 * p1);
    double F100 = (F99 * p2);
    double F21 = (F16 * K2);
    double F17 = (F15 * F16);
    double F39 = (F33 + F38);
    double F78 = (F15 * F74);
    double F75 = (F74 * K2);
    double F102 = (F101 + F100);
    double F37 = (F32 + F36);
    double F25 = (F17 * K3);
    double F69 = (F65 + F68);
    double F22 = (F18 + F21);
    double F53 = (F52 * K2);
    double F56 = (F15 * F52);
    double F77 = (F76 + F75);
    double F26 = (F25 + F22);
    double F55 = (F54 + F53);
    double F58 = (F56 + F57);
    double F80 = (F78 + F79);
    double F59 = (F58 * K3);
    double F81 = (F80 * K3);
    double F43 = (F26 * xPi);
    double F47 = (F26 * yPi);
    double F44 = (F43 + xPi);
    double F82 = (F77 + F81);
    double F48 = (F47 + yPi);
    double F60 = (F55 + F59);
    double F49 = (F39 + F48);
    double F45 = (F37 + F44);
    double F90 = (F60 * yPi);
    double F83 = (F82 * xPi);
    double F61 = (F60 * xPi);
    double F95 = (F82 * yPi);
    double F46 = (F45 + F42);
    double F62 = (F26 + F61);
    double F94 = (F90 + F93);
    double F88 = (F83 + F87);
    double F96 = (F26 + F95);
    double F97 = (F96 + 1);
    double F63 = (F62 + 1);
    double F89 = (F88 + b2);
    double F70 = (F63 + F69);
    double F103 = (F102 + F97);
    double F71 = (F70 + b1);
    this->mBufLineRes[aK][0] = F46;
    this->mBufLineRes[aK][1] = F71;
    this->mBufLineRes[aK][2] = F89;
    this->mBufLineRes[aK][3] = F49;
    this->mBufLineRes[aK][4] = F94;
    this->mBufLineRes[aK][5] = F103;
  }
}

cCompiledCalculator<double> * Alloc_EqDist_Dist_Rad3_Dec1_XY1VDer(int aSzBuf)
{
   return new cEqDist_Dist_Rad3_Dec1_XY1VDer(aSzBuf);
}

cName2Calc<double> TheNameAlloc_EqDist_Dist_Rad3_Dec1_XY1VDer("EqDist_Dist_Rad3_Dec1_XY1VDer",Alloc_EqDist_Dist_Rad3_Dec1_XY1VDer);

} // namespace NS_SymbolicDerivative
