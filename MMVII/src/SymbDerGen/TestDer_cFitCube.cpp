#include "TestDer_cFitCube.h"



namespace NS_SymbolicDerivative {

void cFitCube::DoEval()
{




#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t aK=0; aK < this->mNbInBuf; aK++) {
// Declare local vars in loop to make them per thread
    double &x = this->mVUk[aK][0];
    double &y = this->mVUk[aK][1];
    double &a = this->mVObs[aK][0];
    double &b = this->mVObs[aK][1];
    double F7_ = (b * x);
    double F13_ = (2 * b);
    double F8_ = (F7_ + a);            // ax+b
    double F9_ = square(F8_);          // (ax+b)^2
    double F14_ = (F13_ * F8_);        // 2b(ax+b)
    double F10_ = std::cos(F9_);       // cos((ax+b)^2)
    double F15_ = -(F14_);             // -2b(ax+b)
    double F12_ = std::sin(F9_);       // sin((ax+b)^2)
    double F11_ = (F10_ - y);          // cos((ax+b)^2) -y 
    double F16_ = (F15_ * F12_);       // -2b(ax+b) sin((ax+b)^2)
				      
    this->mBufLineRes[aK][0] = F11_;   // F =  cos((ax+b)^2) -y 
    this->mBufLineRes[aK][1] = F16_;   // dF/dx =  -2b(ax+b) sin((ax+b)^2)
    this->mBufLineRes[aK][2] = -1;     // dF/dy =  -1
  }
}

} // namespace NS_SymbolicDerivative
