#ifdef _OPENMP
#include <omp.h>
#endif
#include "SymbDer/SymbDer_Common.h"

namespace NS_SymbolicDerivative {

class cFitCube : public cCompiledCalculator<double>
{
public:
    typedef cCompiledCalculator<double> Super;
    cFitCube(size_t aSzBuf) : 
      Super(
          "FitCube",
           aSzBuf,//SzBuf
          1,//NbElement
          3,//SzOfLine
          {"x","y"},// Name Unknowns
          {"a","b"},// Name Observations
          1,//With derivative ?
          3//Size of interv
      )
    {
    }
    static std::string FormulaName() { return "FitCube";}
protected:
    virtual void DoEval() override;
private:
    static Super* Alloc(int aSzBuf) {return new cFitCube(aSzBuf); }
    friend void cName2CalcRegisterAll(void);
    static void Register() {cName2Calc<TypeElem>::Register(FormulaName(),cFitCube::Alloc);}
};

} // namespace NS_SymbolicDerivative
