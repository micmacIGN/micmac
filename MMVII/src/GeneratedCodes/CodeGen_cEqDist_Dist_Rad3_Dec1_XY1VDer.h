#ifdef _OPENMP
#include <omp.h>
#endif
#include "include/SymbDer/SymbDer_Common.h"

namespace NS_SymbolicDerivative {

class cEqDist_Dist_Rad3_Dec1_XY1VDer : public cCompiledCalculator<double>
{
public:
    typedef cCompiledCalculator<double> Super;
    cEqDist_Dist_Rad3_Dec1_XY1VDer(size_t aSzBuf) : 
      Super(
          "EqDist_Dist_Rad3_Dec1_XY1VDer",
           aSzBuf,//SzBuf
          2,//NbElement
          6,//SzOfLine
          {"xPi","yPi"},// Name Unknowns
          {"K1","K2","K3","p1","p2","b2","b1"},// Name Observations
          1,//With derivative ?
          3//Size of interv
      )
    {
    }
    static std::string FormulaName() { return "EqDist_Dist_Rad3_Dec1_XY1VDer";}
protected:
    virtual void DoEval() override;
};

} // namespace NS_SymbolicDerivative
