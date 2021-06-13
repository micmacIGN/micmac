#ifndef FORMULA_RATKOWSKYRESIDUAL_H
#define FORMULA_RATKOWSKYRESIDUAL_H

class cRatkowskyResidual {
public:
    static constexpr int NbUk() {return 4;}
    static constexpr int NbObs() {return 2;}
    static const std::vector<std::string>& VNamesUnknowns() {
        static std::vector<std::string> uks = {"b1","b2","b3","b4"};
        return uks;
    }
    static const std::vector<std::string>& VNamesObs() {
        static std::vector<std::string> obs = {"y","x"}; // Warn the data I got were in order y,x ..
        return obs;
    }
    static std::string FormulaName() { return "RatkowskyResidual";}


    template <class TypeUk,class TypeObs>
    static std::vector<TypeUk> formula
                  (
                      const std::vector<TypeUk> & aVUk,
                      const std::vector<TypeObs> & aVObs
                  )
    {
        auto & b1 = aVUk[0];
        auto & b2 = aVUk[1];
        auto & b3 = aVUk[2];
        auto & b4 = aVUk[3];

        auto & x  = aVObs[1];  // Warn the data I got were in order y,x ..
        auto & y  = aVObs[0];

        // Model :  y = b1 / (1+exp(b2-b3*x)) ^ 1/b4 + Error()  [Ratko]
        return { b1 / pow(1.0+exp(b2-b3*x),1.0/b4) - y } ;
    }

};

#endif // FORMULA_RATKOWSKYRESIDUAL_H
