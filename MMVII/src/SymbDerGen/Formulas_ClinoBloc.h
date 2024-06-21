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
                // We have the two Boresight matrix as unknown
                // The unknowns are the the axiator
                return  Append(NamesP3("W1"), NamesP3("W2"));
            }

            std::vector<std::string>    VNamesObs() const
            {
                // we have :
                // - the first boresight matrix (initial solution) mB1,
                // - the second boresight matrix (initial solution) mB,
                // - the vertical (3 observations)
                // - the orientation of the image (rotation matrix : 9 observations) mR
                // - value of clinometer 1
                // - value of clinometer 2
                std::vector<std::string> aVec = Append(NamesMatr("mB1",cPt2di(3,3)), NamesMatr("mB2",cPt2di(3,3)), NamesMatr("mVert",cPt2di(1,3)), NamesMatr("mR",cPt2di(3,3)));
                aVec.push_back("C1");
                aVec.push_back("C2");
                return  aVec;
            }

            template <typename tUk> cMatF<tUk> RotKappa(const tUk aKappa) const
            {
                cMatF<tUk> aRot(3, 3, CreateCste(0.0, aKappa));
                aRot(0,0) = cos(aKappa);
                aRot(1,0) = -sin(aKappa);
                aRot(0,1) = sin(aKappa);
                aRot(1,1) = cos(aKappa);
                aRot(2,2) = CreateCste(1.0, aKappa);
                return aRot;
            }

            

            template <typename tUk> std::vector<tUk> formula
                (
                    const std::vector<tUk> & aVUk,
                    const std::vector<tUk> & aVObs
                ) const
            {

                // get initial solution for B1
                cMatF<tUk> aB1 = cMatF<tUk>(3,3,aVObs, 0);
                // get initial solution for B2
                cMatF<tUk> aB2 = cMatF<tUk>(3,3,aVObs, 9);
                // get vertical
                cMatF<tUk> aVertical = cMatF<tUk>(1,3,aVObs,18);
                // get camera orientation
                cMatF<tUk> aR = cMatF<tUk>(3,3,aVObs,21).Transpose();
                
                // get clinometer measure 
                tUk aTheta1 = -aVObs[30];
                tUk aTheta2 = -aVObs[31];

                cMatF<tUk> aTheta1Rot = RotKappa(aTheta1);
                cMatF<tUk> aTheta2Rot = RotKappa(aTheta2);

                // get rotation matrix from axiator 1
                cMatF<tUk> aBAx1 = aB1 * cMatF<tUk>::MatAxiator(-VtoP3(aVUk,0));
                // get rotation matrix from axiator 2
                cMatF<tUk> aBAx2 = aB2 * cMatF<tUk>::MatAxiator(-VtoP3(aVUk,3));

                // Rotation from clino 1 system to clino 2 system
                cMatF<tUk> aRc1c2 = aBAx2 * aBAx1.Transpose();

                // aBAx1 * aR * aVertical : vertical in first clinometer system
                // aTheta1Rot * aBAx1 * aR * aVertical : vertical in first clinometer system corrected by first clinometer measure
                // aRc1c2 * aTheta1Rot * aBAx1 * aR * aVertical : vertical corrected by first clinometer measure in second clinometer system 
                // aTheta2Rot * aRc1c2 * aTheta1Rot * aBAx1 * aR * aVertical : vertical corrected by the two clinometers measures in second clinometer system
                // Theorically, aV1 = (1, 0, 0)
                cMatF<tUk> aV1 = aTheta2Rot * aRc1c2 * aTheta1Rot * aBAx1 * aR * aVertical;

                cMatF<tUk> aV2 = aTheta1Rot * aRc1c2.Transpose() * aTheta2Rot * aBAx2 * aR * aVertical;


                tUk aDelta1 = aV1(0,0) - 1.0;
                tUk aDelta2 = aV2(0,0) - 1.0;


                std::vector<tUk> aVec;
                aVec.push_back(aDelta1);
                aVec.push_back(aV1(0,1));
                aVec.push_back(aV1(0,2));
                aVec.push_back(aDelta2);
                aVec.push_back(aV2(0,1));
                aVec.push_back(aV2(0,2));
                return aVec;

	        }
    };

}

#endif 