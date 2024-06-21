#include "MMVII_Bench.h"
#include "MMVII_BundleAdj.h"
//#include "GeneratedCodeGen_cClinoBlocVal.h"
#include "MMVII_PhgrDist.h"
#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_MACRO.h"
#include "../BundleAdjustment/BundleAdjustment.h"

namespace MMVII
{

    void BenchClinoFormula()
    {
        //Check Formulas_ClinoBloc
        
        cCalculator<double> * aEqClinoBloc;
        aEqClinoBloc = EqClinoBloc(false, 1, true);
        
        // Value of axiator
        std::vector<tREAL8> aVUK = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<tREAL8> aVObs;

        
        // First Boresight matrix
        cRotation3D<tREAL8> aRotation = cRotation3D<tREAL8>::RotFromCanonicalAxes("j-k-i");
        cDenseMatrix<tREAL8> aBoresight = aRotation.Mat();
        
        // Vertical
        tPt3dr aVertical (0.0, 0.0, -1.0);
        
        // Image orientation
        cDenseMatrix<tREAL8> aImageOrientation = cDenseMatrix<tREAL8>::Identity(3);
        // values of the two clinometers
        tREAL8 aClinoValue1 = 0;
        tREAL8 aClinoValue2 = 0;


        // Second Boresight matrix
        std::string aNameRel12 = "i-kj";
        tRotR aRel12 = tRotR::RotFromCanonicalAxes(aNameRel12);
        cDenseMatrix<tREAL8> aBoresight2 = aRel12.Mat() * aBoresight;

        
        // Push observations
        aBoresight.PushByLine(aVObs);
        aBoresight2.PushByLine(aVObs);
        aVertical.PushInStdVector(aVObs);
        aImageOrientation.PushByLine(aVObs);
        aVObs.push_back(aClinoValue1);
        aVObs.push_back(aClinoValue2);

        
        
        std::vector<tREAL8> aVResult = aEqClinoBloc->DoOneEval(aVUK, aVObs);
        for (auto & aResult : aVResult)
        {
            MMVII_INTERNAL_ASSERT_bench(std::abs(aResult-0)<1e-5,"ClinoFormulaBench failed");
        }
        

    }

    void BenchClinoBa()
    {
        
        // Test for Bundle Adjustment of clinometer
        // Three cameras. The orientation of the first one is the identity. For the two other, there are small rotations around x and y axes
        // The orientation of the first clinometer relative to the camera is -zx-y
        // The orientation of the second clinometer relative to the first clinometer is i-kj
        // The final boresight matrixes are expected to be the same than initial matrixes
        
        // Create clinos
        cOneCalibClino firstCalibClino("C1");
        cOneCalibClino secondCalibClino("C2");
        std::vector<cOneCalibClino>aVCalibClino({firstCalibClino, secondCalibClino});
        
        // Create camera
        cDenseMatrix<tREAL8> aCameraOrientation = cDenseMatrix<tREAL8>::Identity(3);
        std::string aCameraName = "001";

        // Create aCalibSetClino
        cCalibSetClino* aCalibSetClino = new cCalibSetClino(aCameraName, aVCalibClino);
        cBA_Clino aBAClino(nullptr, aCalibSetClino);
        
        cRotation3D<tREAL8> aRotation(aCameraOrientation, false);
        cPtxd<tREAL8,3> aTr = {0.0, 0.0, 0.0};
        cIsometry3D<tREAL8> aIsometry(aTr, aRotation);
        cSensorCamPC aSensorCamPC(aCameraName, aIsometry, nullptr);
        
        // Create clino measures
        std::vector<std::string>aVClinoName({firstCalibClino.NameClino(), secondCalibClino.NameClino()});
        std::vector<tREAL8>aVAngles({0.0, 0.00001, 0.0, 0.00001});
        cClinoMes1Cam aClinoMes1Cam(&aSensorCamPC, aVClinoName, aVAngles);
        aBAClino.addClinoMes1Cam(aClinoMes1Cam);


        // Create a second camera with a rotation on y axe
        tREAL8 aDeltaPhi = 0.05;
        std::string aCameraName2 = "002";
        cDenseMatrix<tREAL8> aCameraOrientation2 = cRotation3D<tREAL8>::RotPhi(aDeltaPhi);
        
        cRotation3D<tREAL8> aRotation2(aCameraOrientation2, false);
        cIsometry3D<tREAL8> aIsometry2(aTr, aRotation2);
        cSensorCamPC aSensorCamPC2(aCameraName2, aIsometry2, nullptr);
        std::vector<tREAL8>aVAngles2({aDeltaPhi, 0.00001, 0.0, 0.00001});
        cClinoMes1Cam aSecondClinoMes1Cam(&aSensorCamPC2, aVClinoName, aVAngles2);
        aBAClino.addClinoMes1Cam(aSecondClinoMes1Cam);

        // Create a third camera with a rotation on x axe
        tREAL8 aDeltaOmega = 0.05;
        std::string aCameraName3 = "003";
        cDenseMatrix<tREAL8> aCameraOrientation3 = cRotation3D<tREAL8>::RotOmega(aDeltaOmega);
        cRotation3D<tREAL8> aRotation3(aCameraOrientation3, false);
        cIsometry3D<tREAL8> aIsometry3(aTr, aRotation3);
        cSensorCamPC aSensorCamPC3(aCameraName3, aIsometry3, nullptr);
        std::vector<tREAL8>aVAngles3({0.0, 0.00001, aDeltaOmega, 0.00001});
        cClinoMes1Cam aThirdClinoMes1Cam(&aSensorCamPC3, aVClinoName, aVAngles3);
        aBAClino.addClinoMes1Cam(aThirdClinoMes1Cam);
        
    
        // Create initial boresight matrix for the first clinometer
        cRotation3D<tREAL8> aRotationB1 = cRotation3D<tREAL8>::RotFromCanonicalAxes("j-k-i");

        aBAClino.addClinoWithUK("C1", aRotationB1);

        // Create initial boresight matrix for the second clinometer
        std::string aNameRel12 = "i-kj";
        tRotR aRel12 = tRotR::RotFromCanonicalAxes(aNameRel12);
        cDenseMatrix<tREAL8> aBoresight2 = aRel12.Mat() * aRotationB1.Mat();
        cRotation3D<tREAL8> aRotationB2(aBoresight2, false);

        aBAClino.addClinoWithUK("C2", aRotationB2);

        
        // Beginning of Bundle Adjustment
        cSetInterUK_MultipeObj<tREAL8> aSetIntervMultObj;
        tREAL8 aLVM = 0.;

        aBAClino.AddToSys(aSetIntervMultObj);
        cDenseVect<tREAL8> aVUk = aSetIntervMultObj.GetVUnKnowns();
        cResolSysNonLinear<tREAL8>  aSys = cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aVUk);

        for (int iter=0; iter<6; ++iter)
        {
            aBAClino.addEquations(aSys);
            const auto & aVectSol = aSys.R_SolveUpdateReset(aLVM);
            aSetIntervMultObj.SetVUnKnowns(aVectSol);
        }

        // Check that final first Boresight matrix is the same than the initial one
        std::vector<tRotR> aVClinosWithUKRot = aBAClino.ClinosWithUKRot();
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; i < 3; i++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aVClinosWithUKRot[0].Mat()(i, j) - aRotationB1.Mat()(i, j))<1e-8,"ClinoBABench failed");
            }
        }

        // Check that final second Boresight matrix is the same than the initial one
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; i < 3; i++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aVClinosWithUKRot[1].Mat()(i, j) - aBoresight2(i, j))<1e-8,"ClinoBABench failed");
            }
        }
    }


    void BenchClino(cParamExeBench & aParam)
    {
        if (! aParam.NewBench("BAClino")) return;

        BenchClinoFormula();
        BenchClinoBa();

        aParam.EndBench();
        return;
    }

}