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
void GenerateCode(bool WithDerivative)
{
    std::string aPost = WithDerivative ? "_ValAndDer" : "_Val";

    NS_SymbolicDerivative::cCoordinatorF<double>
            mCFD1(FORMULA::FormulaName()+aPost,0,FORMULA::VNamesUnknowns(),FORMULA::VNamesObs());

    auto aVFormula = FORMULA::formula(mCFD1.VUk(),mCFD1.VObs());
    if (WithDerivative) 
       mCFD1.SetCurFormulasWithDerivative(aVFormula);
    else
       mCFD1.SetCurFormulas(aVFormula);
    auto [name,file] = mCFD1.GenerateCode();
    includesNames.push_back(file);
    auto [nameLE,fileLE] = mCFD1.GenCodeLonExpr();
    includesNames.push_back(fileLE);
}


int main(int , char **)
{
    for (int aTime=0 ; aTime<2 ; aTime++)
    {
         bool WithDer = (aTime==0);
         // GenerateCode<cEqCoLinearity<cTplFraserDist<3>>>(WithDer);
         GenerateCode<cEqDist<cTplFraserDist<3>>>(WithDer);
         GenerateCode<cFraserCamColinear>(WithDer);
         GenerateCode<cPrimitivesTest>(WithDer);
         GenerateCode<cRatkowskyResidual>(WithDer);
         GenerateCode<cEqCoLinearity<cTplFraserDist<4>>>(WithDer);
         GenerateCode<cEqCoLinearity<cTplPolDist<7>>>(WithDer);
         GenerateCode<cEqCoLinearity<cTplPolDist<2>>>(WithDer);
    }

    std::ofstream os(std::string("CodeGen_IncludeAll.h"));
    for (auto &include : includesNames )
        os << "#include \"" << include << ".h\"\n";

    os = std::ofstream(std::string("CodeGen_MakeFile"));
    for (auto &file : includesNames )
        os << file << ".o: " << file << ".cpp " << file << ".h" << "\n";

    return EXIT_SUCCESS;
}
