#ifndef _FORMULA_RPC_H_
#define _FORMULA_RPC_H_

// PUSHB

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_PhgrDist.h"

#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

/** "Helper" class for evaluating the ground to image projection RPC projection function
*/

//TO-DO
class cRPC_Formulas
{
    public:

        //returns a vectors of unique identifiers
        //  for 20 coefficients of a polynomial
        static std::vector<std::string> NameCoeff(const std::string& aPref)
        {
            std::vector<std::string> aRes;

            for(int aK=0; aK<20; aK++)
                aRes.push_back(aPref+ "_" + ToStr(aK));

            return aRes;
        }

        //Polyn() : multiplication of the 3rd degree polynomial with a 3D point


        //RatioPolyn() : the computation of the ratio of two polynomials


        //NormIn() : normalise the input 3d coordinates


        //NormOut() : de-normalise the output 3d coordinates


};

//TO-DO
// Implement the ground to image projection function

/*class cFormula_RPC_RatioPolyn
{
public:
        // class empty constructor


        // VNamesUnknowns() : function returning names of unknowns


        // VNamesObs() : function returning unique names of observations :
        //   - offset_in, scale_in
        //   - offset_out, scale_out
        //   - RPC coeffs (NumI,DenI,NumJ,DenJ)


        // FormulaName() : function returning the name of the equation

        // formula() : your projection equation



};*/

};//  namespace MMVII

#endif // _FORMULA_RPC_H_
