#ifndef _FORMULA_CLINOROT_H_
#define _FORMULA_CLINOROT_H_



#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

    class cFormulaClinoRot
    {
        public:
            std::string FormulaName() const { return "ClinoRot";}

            // This formula prevents solution from diverging, even if mathematically, B2 * B.T should not be always equal to R0.

            std::vector<std::string>  VNamesUnknowns()  const
            {
                // We have two Boresight matrix as unknown
                // The unknowns are the axiator
                return  Append(NamesP3("W1"), NamesP3("W2"));
            }

            std::vector<std::string>    VNamesObs() const
            {
                // we have :
                // - the first boresight matrix (current solution : 9 observations) mB1,
                // - the second boresight matrix (current solution : 9 observations) mB2,
                // - the initial relative orientation between the two matrix (rotation matrix : 9 observations) mR0
                std::vector<std::string> aVec = Append(NamesMatr("mB1",cPt2di(3,3)), NamesMatr("mB2",cPt2di(3,3)), NamesMatr("mR0",cPt2di(3,3)));
                return  aVec;
            }
        

            template <typename tUk> std::vector<tUk> formula
                (
                    const std::vector<tUk> & aVUk,
                    const std::vector<tUk> & aVObs
                ) const
            {

                // get current solution for B1
                cMatF<tUk> aB1 = cMatF<tUk>(3,3,aVObs, 0);
                
                // get current solution for B2
                cMatF<tUk> aB2 = cMatF<tUk>(3,3,aVObs,9);
                
                // get initial relative orientation between the two matrix
                cMatF<tUk> aR0 = cMatF<tUk>(3,3,aVObs,18);
                
                // get first rotation matrix from axiator
                cMatF<tUk> aB1Ax = aB1 * cMatF<tUk>::MatAxiator(-VtoP3(aVUk,0));

                // get second rotation matrix from axiator
                cMatF<tUk> aB2Ax = aB2 * cMatF<tUk>::MatAxiator(-VtoP3(aVUk,3));
               
                // Theorically, aB2Ax * aB1Ax.Transpose() = aR0
                cMatF<tUk> aM = aB2Ax * aB1Ax.Transpose() - aR0;

                std::vector<tUk> aVec;

                aM.PushInVect(aVec);

                return aVec;
	        }
    };

}

#endif 