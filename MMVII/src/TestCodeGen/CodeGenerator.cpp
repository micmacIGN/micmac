#include <vector>
#include <string>
#include <fstream>

#include "SymbDer/SymbolicDerivatives.h"

#include "Formula_Fraser_Test.h"
#include "Formula_Primitives_Test.h"
#include "Formula_Ratkowskyresidual.h"
#include "Formula_Eqcollinearity.h"


static std::vector<std::string> includesNames;

template<typename FORMULA>
void GenerateCode()
{
    NS_SymbolicDerivative::cCoordinatorF<double>
            mCFD1(FORMULA::FormulaName(),0,FORMULA::VNamesUnknowns(),FORMULA::VNamesObs());

    auto aVFormula = FORMULA::formula(mCFD1.VUk(),mCFD1.VObs());
    mCFD1.SetCurFormulasWithDerivative(aVFormula);
    auto name = mCFD1.GenerateCode(FORMULA::FormulaName());
    includesNames.push_back(name + ".h");
    name = mCFD1.GenCodeDevel(FORMULA::FormulaName());
    includesNames.push_back(name + ".h");
}


int main(int , char **)
{
    GenerateCode<cFraserCamColinear>();
    GenerateCode<cPrimitivesTest>();
    GenerateCode<cRatkowskyResidual>();
    GenerateCode<cEqCoLinearity<cTplFraserDist>>();
    GenerateCode<cEqCoLinearity<cTplPolDist<7>>>();
    GenerateCode<cEqCoLinearity<cTplPolDist<2>>>();

    std::ofstream os(std::string("CodeGen_IncludeAll.h"));
    for (auto &include : includesNames )
        os << "#include \"" << include << "\"\n";

    return EXIT_SUCCESS;
}
