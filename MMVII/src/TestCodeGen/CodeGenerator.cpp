#include <vector>
#include <string>
#include <fstream>

#include "include/MMVII_FormalDerivatives.h"

#include "Formula_Fraser_Test.h"
#include "Formula_Primitives_Test.h"
#include "Formula_Ratkowskyresidual.h"
#include "Formula_Eqcollinearity.h"


static std::vector<std::string> includesNames;

template<typename FORMULA>
void GenerateCode()
{
    NS_MMVII_FormalDerivative::cCoordinatorF<double>
            mCFD1(0,FORMULA::VNamesUnknowns(),FORMULA::VNamesObs());

    auto aVFormula = FORMULA::formula(mCFD1.VUk(),mCFD1.VObs());
    mCFD1.SetCurFormulasWithDerivative(aVFormula);
    auto names = mCFD1.GenerateCode(FORMULA::FormulaName());
    includesNames.insert(includesNames.end(),names.begin(),names.end());
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
