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
    double F86 = (2 * xPi);
    double F37 = (b2 * yPi);
    double F74 = (yPi + yPi);
    double F51 = (xPi + xPi);
    double F17 = (xPi * yPi);
    double F65 = (2 * yPi);
    double F16 = (yPi * yPi);
    double F67 = (F51 * 2);
    double F105 = (F86 * p1);
    double F55 = (F51 * K1);
    double F102 = (F74 * 2);
    double F95 = (F65 * p1);
    double F78 = (F74 * K1);
    double F66 = (F65 * p2);
    double F88 = (F74 * p1);
    double F87 = (F86 * p2);
    double F33 = (F16 * 2);
    double F94 = (F51 * p2);
    double F29 = (F17 * 2);
    double F39 = (F37 + F38);
    double F18 = (F15 + F16);
    double F26 = (F15 * 2);
    double F21 = (F18 * K1);
    double F34 = (F18 + F33);
    double F31 = (F29 * p2);
    double F30 = (F29 * p1);
    double F89 = (F88 + F87);
    double F19 = (F18 * F18);
    double F75 = (F18 * F74);
    double F103 = (F74 + F102);
    double F68 = (F51 + F67);
    double F27 = (F26 + F18);
    double F96 = (F94 + F95);
    double F52 = (F18 * F51);
    double F81 = (F19 * F74);
    double F35 = (F34 * p2);
    double F28 = (F27 * p1);
    double F53 = (F52 + F52);
    double F20 = (F18 * F19);
    double F76 = (F75 + F75);
    double F104 = (F103 * p2);
    double F58 = (F19 * F51);
    double F69 = (F68 * p1);
    double F22 = (F19 * K2);
    double F70 = (F66 + F69);
    double F57 = (F18 * F53);
    double F32 = (F28 + F31);
    double F24 = (F20 * K3);
    double F54 = (F53 * K2);
    double F77 = (F76 * K2);
    double F80 = (F18 * F76);
    double F106 = (F104 + F105);
    double F23 = (F21 + F22);
    double F36 = (F30 + F35);
    double F59 = (F57 + F58);
    double F25 = (F24 + F23);
    double F82 = (F80 + F81);
    double F79 = (F78 + F77);
    double F56 = (F55 + F54);
    double F44 = (F25 * yPi);
    double F40 = (F25 * xPi);
    double F60 = (F59 * K3);
    double F83 = (F82 * K3);
    double F45 = (F44 + yPi);
    double F84 = (F79 + F83);
    double F41 = (F40 + xPi);
    double F61 = (F56 + F60);
    double F46 = (F36 + F45);
    double F62 = (F61 * xPi);
    double F42 = (F32 + F41);
    double F99 = (F84 * yPi);
    double F93 = (F61 * yPi);
    double F85 = (F84 * xPi);
    double F97 = (F93 + F96);
    double F49 = (F46 * PPz);
    double F100 = (F25 + F99);
    double F90 = (F85 + F89);
    double F63 = (F25 + F62);
    double F43 = (F42 + F39);
    double F101 = (F100 + 1);
    double F47 = (F43 * PPz);
    double F98 = (F97 * PPz);
    double F50 = (F49 + PPy);
    double F91 = (F90 + b2);
    double F64 = (F63 + 1);
    double F92 = (F91 * PPz);
    double F71 = (F64 + F70);
    double F48 = (F47 + PPx);
    double F107 = (F101 + F106);
    double F72 = (F71 + b1);
    double F108 = (F107 * PPz);
    double F73 = (F72 * PPz);
    this->mBufLineRes[aK][0] = F48;
    this->mBufLineRes[aK][1] = F73;
    this->mBufLineRes[aK][2] = F92;
    this->mBufLineRes[aK][3] = F50;
    this->mBufLineRes[aK][4] = F98;
    this->mBufLineRes[aK][5] = F108;
  }
}

cCalculator<double> * Alloc_EqDist_Dist_Rad3_Dec1_XY1VDer(int aSzBuf)
{
   return new cEqDist_Dist_Rad3_Dec1_XY1VDer(aSzBuf);
}

cName2Calc<double> TheNameAlloc_EqDist_Dist_Rad3_Dec1_XY1VDer("EqDist_Dist_Rad3_Dec1_XY1VDer",Alloc_EqDist_Dist_Rad3_Dec1_XY1VDer);

} // namespace NS_SymbolicDerivative
