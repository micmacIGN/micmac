#ifdef _OPENMP
#include <omp.h>
#endif
#include "include/SymbDer/SymbDer_Common.h"

namespace NS_SymbolicDerivative {

class cEqDist_Dist_Rad3_Dec1_XY1Val : public cCompiledCalculator<double>
{
public:
    typedef cCompiledCalculator<double> Super;
    cEqDist_Dist_Rad3_Dec1_XY1Val(size_t aSzBuf) : 
      Super(
          "EqDist_Dist_Rad3_Dec1_XY1Val",
           aSzBuf,//SzBuf
          2,//NbElement
          2,//SzOfLine
          {"xPi","yPi"},// Name Unknowns
          {"K1","K2","K3","p1","p2","b2","b1"},// Name Observations
          0,//With derivative ?
          1//Size of interv
      )
    {
    }
    static std::string FormulaName() { return "EqDist_Dist_Rad3_Dec1_XY1Val";}
protected:
    virtual void DoEval() override;
};

} // namespace NS_SymbolicDerivative
