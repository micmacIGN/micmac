#ifdef _OPENMP
#include <omp.h>
#endif
#include "include/SymbDer/SymbDer_Common.h"

namespace NS_SymbolicDerivative {

class cEqDist_Dist_Rad3_Dec1_XY1Val : public cCalculator<double>
{
public:
    typedef cCalculator<double> Super;
    cEqDist_Dist_Rad3_Dec1_XY1Val(size_t aSzBuf) : 
      Super("EqDist_Dist_Rad3_Dec1_XY1Val", aSzBuf,2,10,0,1),
      mVUk(aSzBuf),mVObs(aSzBuf)
    {
      this->mNbElem = 2;
      for (auto& line : this->mBufLineRes)
        line.resize(2);
      for (auto& aUk : this->mVUk)
        aUk.resize(this->NbUk());
      for (auto& aObs : this->mVObs)
        aObs.resize(this->NbObs());
    }
    static std::string FormulaName() { return "EqDist_Dist_Rad3_Dec1_XY1Val";}
protected:
    virtual void SetNewUks(const std::vector<double> & aVUks) override
    {
      for (size_t i=0; i<this->NbUk(); i++)
        this->mVUk[this->mNbInBuf][i] = aVUks[i];
    }
    virtual void SetNewObs(const std::vector<double> & aVObs) override
    {
      for (size_t i=0; i<this->NbObs(); i++)
        this->mVObs[this->mNbInBuf][i] = aVObs[i];
    }
    virtual void DoEval() override;
    std::vector<std::vector<double>> mVUk;
    std::vector<std::vector<double>> mVObs;
};

} // namespace NS_SymbolicDerivative
