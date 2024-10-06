#ifndef _FORMULA_CLINOBLOC_H_
#define _FORMULA_CLINOBLOC_H_



#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "ComonHeaderSymb.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

    class cFormulaClinoBloc
    {
        public:
            std::string FormulaName() const { return "ClinoBloc";}

            std::vector<std::string>  VNamesUnknowns()  const
            {
                // We have one Boresight matrix  and the camera pose as unknown
                // The unknowns are the axiator
                return  Append(NamesP3("W1"), NamesPose("CA","WA"));
            }

            std::vector<std::string>    VNamesObs() const
            {
                // we have :
                // - the boresight matrix (current solution : 9 observations) mB,
                // - the vertical (3 observations) mVert,
                // - the current orientation of the camera (rotation matrix : 9 observations) mR
                // - value of clinometer
                std::vector<std::string> aVec = Append(NamesMatr("mB",cPt2di(3,3)), NamesMatr("mVert",cPt2di(1,3)), NamesMatr("mR",cPt2di(3,3)));
                aVec.push_back("C");
                return  aVec;
            }
        

            template <typename tUk> std::vector<tUk> formula
                (
                    const std::vector<tUk> & aVUk,
                    const std::vector<tUk> & aVObs
                ) const
            {

                // get current solution for B
                cMatF<tUk> aB = cMatF<tUk>(3,3,aVObs, 0);
                
                // get vertical
                cMatF<tUk> aVertical = cMatF<tUk>(1,3,aVObs,9);
                
                // get current solution for camera orientation
                cMatF<tUk> aR = cMatF<tUk>(3,3,aVObs,12);
                // get camera orientation from axiator
                cMatF<tUk> aRAx = aR * cMatF<tUk>::MatAxiator(-VtoP3(aVUk,6));
                
                // get clinometer measure 
                tUk aTheta = aVObs[21];

                // get rotation matrix from axiator
                cMatF<tUk> aBAx = aB * cMatF<tUk>::MatAxiator(-VtoP3(aVUk,0));
               
                // aBAx * aRAx * aVertical : vertical in the clinometer system
                cMatF<tUk> aV1 = aBAx * aRAx.Transpose() * aVertical;

                // Projection on plane (i, j)
                tUk aNorm = sqrt(aV1(0,0)*aV1(0,0) + aV1(0,1)*aV1(0,1));
                tUk aDelta1 = aV1(0,0)/aNorm - cos(aTheta);
                tUk aDelta2 = aV1(0,1)/aNorm - sin(aTheta);

                std::vector<tUk> aVec;
                aVec.push_back(aDelta1);
                aVec.push_back(aDelta2);
                return aVec;
	        }
    };

}

#endif 