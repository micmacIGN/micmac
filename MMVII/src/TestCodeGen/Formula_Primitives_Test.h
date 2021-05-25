#ifndef FORMULA_PRIMITIVES_TEST_H
#define FORMULA_PRIMITIVES_TEST_H
#include <vector>
#include <string>
#include <stdexcept>

class cPrimitivesTest
{
public :
    static const int TheNbUk  = 2;
    static const int TheNbObs = 0;
    static const  std::vector<std::string> TheVNamesUnknowns;
    static const  std::vector<std::string> TheVNamesObs;

    static constexpr int NbUk() {return TheNbUk;}
    static constexpr int NbObs() {return TheNbObs;}
    static const std::vector<std::string>& VNamesUnknowns() { return TheVNamesUnknowns;}
    static const std::vector<std::string>& VNamesObs() { return TheVNamesObs;}
    static std::string FormulaName() { return "PrimitivesTest";}

    template <class TypeUk,class TypeObs>
    static std::vector<TypeUk> formula (const std::vector<TypeUk> & aVUk, const std::vector<TypeObs> & aVObs)
    {
        // TODO : Exception ?
        if (aVUk.size() != TheNbUk)
            throw std::range_error("FormulaTestAll: Bad Unk size");

        if (aVObs.size() != TheNbObs)
            throw std::range_error("FormulaTestAll: Bad Obs size");

        // 0 - Ground Coordinates of projected point
        const auto & X = aVUk[0];
        const auto & Y = aVUk[1];

        return {    X,
                    Y,
                    X+Y,
                    X-Y,
                    X*Y,
                    X/Y,
                    pow(X,Y),
                    -X,
                    square(X),
                    cube(X),
                    exp(X),
                    log(X),
                    pow(X,10.0)
        };
    }
};

const std::vector<std::string>
  cPrimitivesTest::TheVNamesUnknowns {"X","Y"};

const std::vector<std::string>
cPrimitivesTest::TheVNamesObs {};

#endif // FORMULA_PRIMITIVES_TEST_H
