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
    double &PPx = this->mVObs[aK][0];
    double &PPy = this->mVObs[aK][1];
    double &PPz = this->mVObs[aK][2];
    double &K1 = this->mVObs[aK][3];
    double &K2 = this->mVObs[aK][4];
    double &K3 = this->mVObs[aK][5];
    double &p1 = this->mVObs[aK][6];
    double &p2 = this->mVObs[aK][7];
    double &b2 = this->mVObs[aK][8];
    double &b1 = this->mVObs[aK][9];
    double F15 = (xPi * xPi);
    double F38 = (b1 * xPi);
    double F37 = (b2 * yPi);
    double F16 = (yPi * yPi);
    double F17 = (xPi * yPi);
    double F33 = (F16 * 2);
    double F26 = (F15 * 2);
    double F39 = (F37 + F38);
    double F18 = (F15 + F16);
    double F29 = (F17 * 2);
    double F19 = (F18 * F18);
    double F30 = (F29 * p1);
    double F34 = (F18 + F33);
    double F27 = (F26 + F18);
    double F31 = (F29 * p2);
    double F21 = (F18 * K1);
    double F35 = (F34 * p2);
    double F28 = (F27 * p1);
    double F22 = (F19 * K2);
    double F20 = (F18 * F19);
    double F36 = (F30 + F35);
    double F23 = (F21 + F22);
    double F32 = (F28 + F31);
    double F24 = (F20 * K3);
    double F25 = (F24 + F23);
    double F44 = (F25 * yPi);
    double F40 = (F25 * xPi);
    double F41 = (F40 + xPi);
    double F45 = (F44 + yPi);
    double F42 = (F32 + F41);
    double F46 = (F36 + F45);
    double F43 = (F42 + F39);
    double F49 = (F46 * PPz);
    double F47 = (F43 * PPz);
    double F50 = (F49 + PPy);
    double F48 = (F47 + PPx);
    this->mBufLineRes[aK][0] = F48;
    this->mBufLineRes[aK][1] = F50;
  }
}

cCalculator<double> * Alloc_EqDist_Dist_Rad3_Dec1_XY1Val(int aSzBuf)
{
   return new cEqDist_Dist_Rad3_Dec1_XY1Val(aSzBuf);
}

cName2Calc<double> TheNameAlloc_EqDist_Dist_Rad3_Dec1_XY1Val("EqDist_Dist_Rad3_Dec1_XY1Val",Alloc_EqDist_Dist_Rad3_Dec1_XY1Val);

} // namespace NS_SymbolicDerivative
