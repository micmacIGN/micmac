#include "CodeGen_cEqDist_Dist_Rad3_Dec1_XY1Val.h"

namespace NS_SymbolicDerivative {

void cEqDist_Dist_Rad3_Dec1_XY1Val::DoEval()
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
    double F12 = (xPi * xPi);
    double F41 = (b1 * xPi);
    double F40 = (b2 * yPi);
    double F13 = (yPi * yPi);
    double F14 = (xPi * yPi);
    double F34 = (F13 * 2);
    double F15 = (F12 + F13);
    double F29 = (F12 * 2);
    double F42 = (F40 + F41);
    double F31 = (F14 * 2);
    double F33 = (F31 * p1);
    double F18 = (F15 * K1);
    double F35 = (F15 + F34);
    double F36 = (F31 * p2);
    double F30 = (F29 + F15);
    double F16 = (F15 * F15);
    double F32 = (F30 * p1);
    double F21 = (F16 * K2);
    double F17 = (F15 * F16);
    double F38 = (F35 * p2);
    double F37 = (F32 + F36);
    double F39 = (F33 + F38);
    double F22 = (F18 + F21);
    double F25 = (F17 * K3);
    double F26 = (F25 + F22);
    double F47 = (F26 * yPi);
    double F43 = (F26 * xPi);
    double F48 = (F47 + yPi);
    double F44 = (F43 + xPi);
    double F49 = (F39 + F48);
    double F45 = (F37 + F44);
    double F46 = (F45 + F42);
    this->mBufLineRes[aK][0] = F46;
    this->mBufLineRes[aK][1] = F49;
  }
}

cCompiledCalculator<double> * Alloc_EqDist_Dist_Rad3_Dec1_XY1Val(int aSzBuf)
{
   return new cEqDist_Dist_Rad3_Dec1_XY1Val(aSzBuf);
}

cName2Calc<double> TheNameAlloc_EqDist_Dist_Rad3_Dec1_XY1Val("EqDist_Dist_Rad3_Dec1_XY1Val",Alloc_EqDist_Dist_Rad3_Dec1_XY1Val);

} // namespace NS_SymbolicDerivative
