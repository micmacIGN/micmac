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
        std::vector<tREAL8> aVUK = {0.0, 0.0, 0.0};
        std::vector<tREAL8> aVObs;

        
        // Boresight matrix
        cRotation3D<tREAL8> aRotation = cRotation3D<tREAL8>::RotFromCanonicalAxes("j-k-i");
        cDenseMatrix<tREAL8> aBoresight = aRotation.Mat();
        
        // Vertical
        tPt3dr aVertical (0.0, 0.0, -1.0);
        
        // Image orientation
        cDenseMatrix<tREAL8> aImageOrientation = cDenseMatrix<tREAL8>::Identity(3);
        // values of the two clinometers
        tREAL8 aClinoValue = 0;
        
        // Push observations
        aBoresight.PushByLine(aVObs);

        aVertical.PushInStdVector(aVObs);
        aImageOrientation.PushByLine(aVObs);
        aVObs.push_back(aClinoValue);
        
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

        aBAClino.setVNamesClino({"C1", "C2"});
        
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
        std::vector<tREAL8>aVAngles2({-aDeltaPhi, 0.00001, 0.0, 0.00001});
        cClinoMes1Cam aSecondClinoMes1Cam(&aSensorCamPC2, aVClinoName, aVAngles2);
        aBAClino.addClinoMes1Cam(aSecondClinoMes1Cam);

        // Create a third camera with a rotation on x axe
        tREAL8 aDeltaOmega = 0.05;
        std::string aCameraName3 = "003";
        cDenseMatrix<tREAL8> aCameraOrientation3 = cRotation3D<tREAL8>::RotOmega(aDeltaOmega);
        cRotation3D<tREAL8> aRotation3(aCameraOrientation3, false);
        cIsometry3D<tREAL8> aIsometry3(aTr, aRotation3);
        cSensorCamPC aSensorCamPC3(aCameraName3, aIsometry3, nullptr);
        std::vector<tREAL8>aVAngles3({0.0, 0.00001, -aDeltaOmega, 0.00001});
        cClinoMes1Cam aThirdClinoMes1Cam(&aSensorCamPC3, aVClinoName, aVAngles3);
        aBAClino.addClinoMes1Cam(aThirdClinoMes1Cam);
        
    
        // Create initial boresight matrix for the first clinometer
        cRotation3D<tREAL8> aRotationB1 = cRotation3D<tREAL8>::RotFromCanonicalAxes("j-k-i");

        aBAClino.addClinoWithUK("C1", aRotationB1);
        aBAClino.addInitRotClino("C1", aRotationB1);

        // Create initial boresight matrix for the second clinometer
        std::string aNameRel12 = "i-kj";
        tRotR aRel12 = tRotR::RotFromCanonicalAxes(aNameRel12);
        cDenseMatrix<tREAL8> aBoresight2 = aRel12.Mat() * aRotationB1.Mat();
        cRotation3D<tREAL8> aRotationB2(aBoresight2, false);

        aBAClino.addClinoWithUK("C2", aRotationB2);
        aBAClino.addInitRotClino("C2", aRotationB2);

        
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
            for (size_t j = 0; j < 3; j++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aVClinosWithUKRot[0].Mat()(i, j) - aRotationB1.Mat()(i, j))<1e-8,"ClinoBABench failed");
            }
        }

        // Check that final second Boresight matrix is the same than the initial one
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aVClinosWithUKRot[1].Mat()(i, j) - aBoresight2(i, j))<1e-8,"ClinoBABench failed");
            }
        }
    }


    void Bench1Clino2PointsBa(){

        // Three points in ground geometry
        std::string aP1Name = "aP1";
        std::string aP2Name = "aP2";
        std::string aP3Name = "aP3";
        cPt3dr aP1 = {0.0, 0.0, 0.0};
        cPt3dr aP2 = {5.0, 20.0, 5.0};
        cPt3dr aP3 = {10.0, 10.0, 15.0};

        // Camera
        cPtxd<tREAL8,3> aCameraTr = {5.0, 5.0, 100.0};
        cRotation3D<tREAL8> aCameraRot = cRotation3D<tREAL8>::RotFromCanonicalAxes("-ij-k")*cRotation3D<tREAL8>::RandomRot(0.01);
        cIsometry3D<tREAL8> aIsometry(aCameraTr, aCameraRot);
        eProjPC aTypeProj = eProjPC::eStenope;
        cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::RandomCalib(aTypeProj, 0);
        std::string aCameraName = "aCamera";
        cSensorCamPC aSensorCamPC(aCameraName, aIsometry, aCalib);

        // Clinometer
        cRotation3D<tREAL8> aClinoOrientation = cRotation3D<tREAL8>::RotFromCanonicalAxes("j-k-i")*cRotation3D<tREAL8>::RandomRot(0.1);
        cRotation3D<tREAL8> aBoresight = cRotation3D<tREAL8>(aClinoOrientation.Mat()*aCameraRot.Mat().Transpose(), false);
        
        // Compute needle measure
        cPtxd<tREAL8,3> aV = {0.0, 0.0, -1.0};
        cPtxd<tREAL8,3> aVClino =  aBoresight.Mat() * aCameraRot.Mat() * aV;
        tREAL8 aNorm = sqrt(aVClino.x()*aVClino.x() + aVClino.y()*aVClino.y());
        tREAL8 aCos = aVClino.x() / aNorm;
        tREAL8 aSin = aVClino.y() / aNorm;
        tREAL8 aR = sqrt(aCos*aCos+aSin*aSin);
        tREAL8 aNeedleMeasure =  2*atan(aSin / (aR+aCos));


        // Points in image geometry
        cPt2dr aP1Image = aSensorCamPC.Ground2Image(aP1);
        cPt2dr aP2Image = aSensorCamPC.Ground2Image(aP2);
        cPt2dr aP3Image = aSensorCamPC.Ground2Image(aP3);

        // Apply small translation and rotation to the camera
        // It will be initial conditions of bundle adjustment
        // The aim is to retrieve previous translation and rotation
        cPtxd<tREAL8,3> aDeltaTr = {0.005, 0.005, 0.005};
        cPtxd<tREAL8,3> aNewCameraTr = aCameraTr+aDeltaTr;
        cRotation3D<tREAL8> aNewCameraRot = aCameraRot*cRotation3D<tREAL8>::RandomRot(0.0001);
        cIsometry3D<tREAL8> aNewIsometry(aNewCameraTr, aNewCameraRot);
        std::string aNewCameraName = "aNewCamera";
        cSensorCamPC aNewSensorCamPC(aNewCameraName, aNewIsometry, aCalib);

        // Create BAClino object
        std::string aClinoName = "aClino";
        cOneCalibClino aCalibClino(aClinoName);
        std::vector<cOneCalibClino>aVCalibClino({aCalibClino});
        cCalibSetClino* aCalibSetClino = new cCalibSetClino(aCameraName, aVCalibClino);
        cBA_Clino aBAClino(nullptr, aCalibSetClino);
        
        // Create clino measures
        std::vector<std::string>aVClinoName({aCalibClino.NameClino()});
        std::vector<tREAL8>aVAngles({aNeedleMeasure, 0.01});
        cClinoMes1Cam aClinoMes1Cam(&aNewSensorCamPC, aVClinoName, aVAngles);
        aBAClino.addClinoMes1Cam(aClinoMes1Cam);
        aBAClino.addClinoWithUK(aClinoName, aBoresight);
        aBAClino.addInitRotClino(aClinoName, aBoresight);
        aBAClino.setVNamesClino({aClinoName});
        
        
        // Create cMMVII_BundleAdj object
        cMMVII_BundleAdj aBundleAdj(nullptr);
        aBundleAdj.AddClinoBloc(&aBAClino);// add clino
        aBundleAdj.AddCamPC(&aNewSensorCamPC);//add camera
        aBundleAdj.AddBenchSensor(&aNewSensorCamPC);
        aBundleAdj.SetFrozenClinos(".*");//freeze boresight matrix
        aBundleAdj.SetParamFrozenCalib(".*");//freeze camera calibration
        // Now, only orientation and position of camera are not frozen

        // Add GCPs
        cSetMesImGCP aSetMesImGCP = cSetMesImGCP();
        // Add 3D measures 
        cSetMesGCP aSetMesGCP = cSetMesGCP();
        aSetMesGCP.AddMeasure(cMes1GCP(aP1, aP1Name, 0.0));
        aSetMesGCP.AddMeasure(cMes1GCP(aP2, aP2Name, 0.0));
        aSetMesGCP.AddMeasure(cMes1GCP(aP3, aP3Name, 0.0));
        aSetMesImGCP.AddMes3D(aSetMesGCP);
        // Add 2D measures
        cSetMesPtOf1Im aSetMesPtOf1Im = cSetMesPtOf1Im();
        tREAL8 aSigma = 0.01;
        aSetMesPtOf1Im.AddMeasure(cMesIm1Pt(aP1Image, aP1Name, aSigma));
        aSetMesPtOf1Im.AddMeasure(cMesIm1Pt(aP2Image, aP2Name, aSigma));
        aSetMesPtOf1Im.AddMeasure(cMesIm1Pt(aP3Image, aP3Name, aSigma));
        aSetMesImGCP.AddMes2D(aSetMesPtOf1Im, &aNewSensorCamPC);
        cStdWeighterResidual aStdWeighterResidual = cStdWeighterResidual();
        aBundleAdj.AddGCP("aGCP", 0.0, aStdWeighterResidual, &aSetMesImGCP);

        for (int aKIter=0 ; aKIter<20 ; aKIter++)
        {
            aBundleAdj.OneIteration();
        }

        
        MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Tr().x()-aSensorCamPC.Pose().Tr().x())<1e-10,"Bench1Clino2PointsBa failed");
        MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Tr().y()-aSensorCamPC.Pose().Tr().y())<1e-10,"Bench1Clino2PointsBa failed");
        MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Tr().z()-aSensorCamPC.Pose().Tr().z())<1e-10,"Bench1Clino2PointsBa failed");

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Rot().Mat()(i, j) - aSensorCamPC.Pose().Rot().Mat()(i, j))<1e-8,"ClinoBABench failed");
            }
        }
    }


    void BenchClino(cParamExeBench & aParam)
    {
        if (! aParam.NewBench("BAClino")) return;

        BenchClinoFormula();
        BenchClinoBa();
        Bench1Clino2PointsBa();

        aParam.EndBench();
        return;
    }

}