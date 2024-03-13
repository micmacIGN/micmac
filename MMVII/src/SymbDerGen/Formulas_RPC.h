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

//TODO-RPCProj
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
        template <typename tUk,typename tObs>
        static tUk Polyn(const cPtxd<tUk,3> & aP,
                         const std::vector<tObs> & aVCoeffs,
                         int aK0)
        {
            return
                aVCoeffs[aK0+0] +
                aVCoeffs[aK0+1] * aP.y() +
                aVCoeffs[aK0+2] * aP.x() +
                aVCoeffs[aK0+3] * aP.z() +
                aVCoeffs[aK0+4] * aP.x() * aP.y() +
                aVCoeffs[aK0+5] * aP.y() * aP.z() +
                aVCoeffs[aK0+6] * aP.x() * aP.z() +
                aVCoeffs[aK0+7] * aP.y() * aP.y() +
                aVCoeffs[aK0+8] * aP.x() * aP.x() +
                aVCoeffs[aK0+9] * aP.z() * aP.z() +
                aVCoeffs[aK0+10] * aP.x() * aP.y() * aP.z() +
                aVCoeffs[aK0+11] * aP.y() * aP.y() * aP.y() +
                aVCoeffs[aK0+12] * aP.y() * aP.x() * aP.x() +
                aVCoeffs[aK0+13] * aP.y() * aP.z() * aP.z() +
                aVCoeffs[aK0+14] * aP.y() * aP.y() * aP.x() +
                aVCoeffs[aK0+15] * aP.x() * aP.x() * aP.x() +
                aVCoeffs[aK0+16] * aP.x() * aP.z() * aP.z() +
                aVCoeffs[aK0+17] * aP.y() * aP.y() * aP.z() +
                aVCoeffs[aK0+18] * aP.x() * aP.x() * aP.z() +
                aVCoeffs[aK0+19] * aP.z() * aP.z() * aP.z();
        }

        //RatioPolyn() : the computation of the ratio of two polynomials
        template <typename tUk,typename tObs>
        static tUk RatioPolyn(const cPtxd<tUk,3> & aP,
                              const std::vector<tObs> & aVCoeffs,
                              int aK0Num,int aK0Den)
        {
            return Polyn(aP,aVCoeffs,aK0Num) / Polyn(aP,aVCoeffs,aK0Den);
        }


        //NormIn() : normalise the input 3d coordinates
        template <typename tUk,typename tObs>
        static cPtxd<tUk,3> NormIn(const cPtxd<tUk,3> & aP,
                                   const cPtxd<tObs,3> & aOffsetIn,
                                   const cPtxd<tObs,3> & aScaleIn)
        {
            return DivCByC(aP - aOffsetIn,aScaleIn);
        }


        //NormOut() : de-normalise the output 3d coordinates
        template <typename tUk,typename tObs>
        static cPtxd<tUk,3> NormOut(const cPtxd<tUk,3> & aP,
                                    const cPtxd<tObs,3> & aOffsetOut,
                                    const cPtxd<tObs,3> & aScaleOut)
        {
            return (MulCByC(aP,aScaleOut) + aOffsetOut);
        }

};

//TODO-RPCProj
// Implement the ground to image projection function

class cFormula_RPC_RatioPolyn : cRPC_Formulas
{
public:
        // class empty constructor
    cFormula_RPC_RatioPolyn() {}


        // VNamesUnknowns() : function returning names of unknowns
    static std::vector<std::string> VNamesUnknowns()
    {
        return {"X","Y","Z"};
    }

        // VNamesObs() : function returning unique names of observations :
        //   - offset_in, scale_in
        //   - offset_out, scale_out
        //   - RPC coeffs (NumI,DenI,NumJ,DenJ)
    static std::vector<std::string> VNamesObs()
    {
        return Append(
            {"OffSetIn_x","OffSetIn_y","OffSetIn_z","ScaleIn_x","ScaleIn_y","ScaleIn_z"},
            {"OffSetOut_x","OffSetOut_y","OffSetOut_z","ScaleOut_x","ScaleOut_y","ScaleOut_z"},
            Append(
                Append( NameCoeff("INum"),NameCoeff("IDen") ),
                Append( NameCoeff("JNum"),NameCoeff("JDen") )
                )
            );
    }


        // FormulaName() : function returning the name of the equation
    static std::string FormulaName(){ return "RPC_Transformation"; }


        // formula() : your projection equation
    template <typename tUk,typename tObs>
    static std::vector<tUk> formula(const std::vector<tUk> & aVUk,
                                    const std::vector<tObs> & aVObs)
    {

        //extract observations
        cPtxd<tObs,3> aOffsetIn = VtoP3(aVObs,0);
        cPtxd<tObs,3> aScaleIn = VtoP3(aVObs,3);
        cPtxd<tObs,3> aOffsetOut = VtoP3(aVObs,6);
        cPtxd<tObs,3> aScaleOut = VtoP3(aVObs,9);

        // extract unknowns
        cPtxd<tUk,3> aPIn = VtoP3(aVUk,0);

        // normalise input
        aPIn = NormIn(aPIn,aOffsetIn,aScaleIn);
        SymbComment(aPIn.x(),"Input point x norm");
        SymbComment(aPIn.y(),"Input point y norm");
        SymbComment(aPIn.z(),"Input point z norm");


        // apply rational polynomial
        tUk aI = RatioPolyn(aPIn,aVObs,12,32);
        tUk aJ = RatioPolyn(aPIn,aVObs,52,72);


        cPtxd<tUk,3> aPtOut(aI,aJ,aPIn.z());

        // normalise output
        aPtOut = NormOut(aPtOut,aOffsetOut,aScaleOut);
        SymbCommentDer(aPtOut.x(),0,"dI/dX");
        SymbCommentDer(aPtOut.x(),1,"dI/dY");
        SymbCommentDer(aPtOut.x(),2,"dI/dZ");

        SymbPrintDer(aPtOut.x(),0,"dI/dX");
        SymbPrintDer(aPtOut.x(),1,"dI/dY");
        SymbPrintDer(aPtOut.x(),2,"dI/dZ");

        SymbPrintDer(aPtOut.y(),0,"dJ/dX");
        SymbPrintDer(aPtOut.y(),1,"dJ/dY");
        SymbPrintDer(aPtOut.y(),2,"dJ/dZ");


        //return vector with output point + z
        return ToVect(aPtOut);
    }



};

};//  namespace MMVII

#endif // _FORMULA_RPC_H_
