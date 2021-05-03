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
    double F35 = (b1 * xPi);
    double F34 = (b2 * yPi);
    double F13 = (yPi * yPi);
    double F14 = (xPi * yPi);
    double F30 = (F13 * 2);
    double F15 = (F12 + F13);
    double F23 = (F12 * 2);
    double F36 = (F34 + F35);
    double F26 = (F14 * 2);
    double F27 = (F26 * p1);
    double F18 = (F15 * K1);
    double F31 = (F15 + F30);
    double F28 = (F26 * p2);
    double F24 = (F23 + F15);
    double F16 = (F15 * F15);
    double F25 = (F24 * p1);
    double F19 = (F16 * K2);
    double F17 = (F15 * F16);
    double F32 = (F31 * p2);
    double F29 = (F25 + F28);
    double F33 = (F27 + F32);
    double F20 = (F18 + F19);
    double F21 = (F17 * K3);
    double F22 = (F21 + F20);
    double F41 = (F22 * yPi);
    double F37 = (F22 * xPi);
    double F42 = (F41 + yPi);
    double F38 = (F37 + xPi);
    double F43 = (F33 + F42);
    double F39 = (F29 + F38);
    double F40 = (F39 + F36);
    this->mBufLineRes[aK][0] = F40;
    this->mBufLineRes[aK][1] = F43;
  }
}

cCompiledCalculator<double> * Alloc_EqDist_Dist_Rad3_Dec1_XY1Val(int aSzBuf)
{
   return new cEqDist_Dist_Rad3_Dec1_XY1Val(aSzBuf);
}

cName2Calc<double> TheNameAlloc_EqDist_Dist_Rad3_Dec1_XY1Val("EqDist_Dist_Rad3_Dec1_XY1Val",Alloc_EqDist_Dist_Rad3_Dec1_XY1Val);

} // namespace NS_SymbolicDerivative
