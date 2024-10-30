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
        std::vector<tREAL8> aVUK = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
        std::vector<tREAL8>aVAngles({0.0, 0.0});
        std::vector<tREAL8>aVWeights({0.00001, 0.00001});
        cClinoMes1Cam aClinoMes1Cam(&aSensorCamPC, aVClinoName, aVAngles, aVWeights);
        aBAClino.addClinoMes1Cam(aClinoMes1Cam);


        // Create a second camera with a rotation on y axe
        tREAL8 aDeltaPhi = 0.05;
        std::string aCameraName2 = "002";
        cDenseMatrix<tREAL8> aCameraOrientation2 = cRotation3D<tREAL8>::RotPhi(aDeltaPhi);
        
        cRotation3D<tREAL8> aRotation2(aCameraOrientation2, false);
        cIsometry3D<tREAL8> aIsometry2(aTr, aRotation2);
        cSensorCamPC aSensorCamPC2(aCameraName2, aIsometry2, nullptr);
        std::vector<tREAL8>aVAngles2({aDeltaPhi, 0.0});
        std::vector<tREAL8>aVWeights2({0.00001, 0.00001});
        cClinoMes1Cam aSecondClinoMes1Cam(&aSensorCamPC2, aVClinoName, aVAngles2, aVWeights2);
        aBAClino.addClinoMes1Cam(aSecondClinoMes1Cam);

        // Create a third camera with a rotation on x axe
        tREAL8 aDeltaOmega = 0.05;
        std::string aCameraName3 = "003";
        cDenseMatrix<tREAL8> aCameraOrientation3 = cRotation3D<tREAL8>::RotOmega(aDeltaOmega);
        cRotation3D<tREAL8> aRotation3(aCameraOrientation3, false);
        cIsometry3D<tREAL8> aIsometry3(aTr, aRotation3);
        cSensorCamPC aSensorCamPC3(aCameraName3, aIsometry3, nullptr);
        std::vector<tREAL8>aVAngles3({0.0, aDeltaOmega});
        std::vector<tREAL8>aVWeights3({0.00001, 0.00001});
        cClinoMes1Cam aThirdClinoMes1Cam(&aSensorCamPC3, aVClinoName, aVAngles3, aVWeights3);
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

        // Add unknowns objects to the system 
        aBAClino.AddToSys(aSetIntervMultObj);
        aSetIntervMultObj.AddOneObj(&aSensorCamPC); 
        aSetIntervMultObj.AddOneObj(&aSensorCamPC2); 
        aSetIntervMultObj.AddOneObj(&aSensorCamPC3);
        cDenseVect<tREAL8> aVUk = aSetIntervMultObj.GetVUnKnowns();
        cResolSysNonLinear<tREAL8>  aSys = cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aVUk);

        // Freeze Camera unknowns : without GCP, resolution cannot converge
        aSys.SetFrozenAllCurrentValues(aSensorCamPC);
        aSys.SetFrozenAllCurrentValues(aSensorCamPC2);
        aSys.SetFrozenAllCurrentValues(aSensorCamPC3);

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

    void AddGCP(const std::string& aPName, const cPt3dr& aCoords, const cSensorCamPC& aSensorCamPC, cSetMesGCP& aSetMesGCP, cSetMesPtOf1Im& aSetMesPtOf1Im, const tREAL8& aSigma){
        cPt2dr aP1Image = aSensorCamPC.Ground2Image(aCoords);
        aSetMesGCP.AddMeasure(cMes1GCP(aCoords, aPName, 0.0));
        aSetMesPtOf1Im.AddMeasure(cMesIm1Pt(aP1Image, aPName, aSigma));
    }

    tREAL8 ComputeNeedleMeasure(const cRotation3D<tREAL8>& aBoresight, const cRotation3D<tREAL8>& aCameraRot){
        cPtxd<tREAL8,3> aV = {0.0, 0.0, -1.0};
        cPtxd<tREAL8,3> aVClino =  aBoresight.Mat() * aCameraRot.Mat().Transpose() * aV;
        tREAL8 aNorm = sqrt(aVClino.x()*aVClino.x() + aVClino.y()*aVClino.y());
        tREAL8 aCos = aVClino.x() / aNorm;
        tREAL8 aSin = aVClino.y() / aNorm;
        tREAL8 aNeedleMeasure =  2*atan(aSin / (1+aCos));
        return aNeedleMeasure;
    }

    cRotation3D<tREAL8> CreateClino(const std::string& aCanonicalAxe, const cRotation3D<tREAL8>& aCameraRot){
        cRotation3D<tREAL8> aClinoOrientation = cRotation3D<tREAL8>::RotFromCanonicalAxes(aCanonicalAxe)*cRotation3D<tREAL8>::RandomRot(0.01);
        cRotation3D<tREAL8> aBoresight = cRotation3D<tREAL8>(aClinoOrientation.Mat()*aCameraRot.Mat(), false);
        return aBoresight;
    }

    void Bench1Clino1Point2Ba(){

        /*
        There is one camera, one clino and one GCP
        All informations are known ( GCP coordinates, camera pose, boresight matrix between camera and clino...)

        A small rotation is applied to the camera.
        The aim is to retrieve the initial orientation
        */

        // GCP in ground geometry
        std::map<std::string, cPt3dr>     mVPoints;
        mVPoints["aP1"] = cPt3dr({0.0, 0.0, 0.0});
        
        // Sigma on clino measure and GCP
        tREAL8 aClinoSigma = 1e-5;
        tREAL8 aGCPSigma = 1e-2;

        // Variations on camera position and orientation
        tREAL8 aAmplTr = 10;
        tREAL8 aAmpl = 1e-4;

        // Create camera : look at the ground with a small random rotation
        // Camera is in the 10 meters (distance L1) of the point (15, 15, 100)
        // Then camera will be never exactly above the GCP
        cPtxd<tREAL8,3> aCameraTr = {15.0, 15.0, 100.0};
        cPtxd<tREAL8,3> aRandomTr = cPtxd<tREAL8,3>::PRandC()*aAmplTr;
        cPtxd<tREAL8,3> aInitTr = aCameraTr + aRandomTr;

        // Camera must be oriented toward the GCP
        // Else, if the GCP is not visible by the camera, then the least square will not converge 
        cPtxd<tREAL8,3> aP0 = VUnit(-aInitTr);
        cPtxd<tREAL8,3> aP1 = VUnit(VOrthog(aP0));
        cPtxd<tREAL8,3> aP2 = aP0 ^ aP1;
        cRotation3D<tREAL8> aCameraRot = cRotation3D<tREAL8>(MatFromCols(aP2,aP1,aP0),false);


        cIsometry3D<tREAL8> aIsometry(aInitTr, aCameraRot);
        eProjPC aTypeProj = eProjPC::eStenope;
        cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::RandomCalib(aTypeProj, 0);
        std::string aCameraName = "aCamera";
        cSensorCamPC aSensorCamPC(aCameraName, aIsometry, aCalib);

        // Clinometer
        cRotation3D<tREAL8> aBoresight = CreateClino("j-k-i", aCameraRot);
        // Compute needle measure
        tREAL8 aNeedleMeasure =  ComputeNeedleMeasure(aBoresight, aCameraRot);

        // Apply small rotation to the camera
        // It will be initial conditions of bundle adjustment
        // The aim is to retrieve previous rotation
        cRotation3D<tREAL8> aNewCameraRot = aCameraRot*cRotation3D<tREAL8>::RandomRot(aAmpl);
        cIsometry3D<tREAL8> aNewIsometry(aInitTr, aNewCameraRot);
        std::string aNewCameraName = "aNewCamera";
        cSensorCamPC aNewSensorCamPC(aNewCameraName, aNewIsometry, aCalib);

        // Create BAClino object
        std::string aClinoName = "aClino1";
        cOneCalibClino aCalibClino(aClinoName);
        std::vector<cOneCalibClino>aVCalibClino({aCalibClino});
        cCalibSetClino* aCalibSetClino = new cCalibSetClino(aCameraName, aVCalibClino);
        cBA_Clino* aBAClino = new cBA_Clino(nullptr, aCalibSetClino);
        
        // Create clino measures
        std::vector<std::string>aVClinoName({aCalibClino.NameClino()});
        std::vector<tREAL8>aVAngles({aNeedleMeasure});
        std::vector<tREAL8>aVWeights({aClinoSigma});
        cClinoMes1Cam aClinoMes1Cam(&aNewSensorCamPC, aVClinoName, aVAngles, aVWeights);
        aBAClino->addClinoMes1Cam(aClinoMes1Cam);

        aBAClino->addClinoWithUK(aClinoName, aBoresight);
        aBAClino->addInitRotClino(aClinoName, aBoresight);
        aBAClino->setVNamesClino({aClinoName});
        
        
        // Create cMMVII_BundleAdj object
        cMMVII_BundleAdj* aBundleAdj = new cMMVII_BundleAdj(nullptr);
        aBundleAdj->AddClinoBloc(aBAClino);// add clino
        aBundleAdj->AddCamPC(&aNewSensorCamPC);//add camera
        aBundleAdj->AddBenchSensor(&aNewSensorCamPC);
        aBundleAdj->SetFrozenClinos(".*");//freeze boresight matrix
        aBundleAdj->SetFrozenCenters(".*");//freeze center of camera
        aBundleAdj->SetParamFrozenCalib(".*");//freeze camera calibration
        aBundleAdj->setVerbose(false);// Not print residuals
        // Now, only orientation of camera is not frozen

        // Add GCPs
        cSetMesGCP aSetMesGCP = cSetMesGCP();
        cSetMesPtOf1Im aSetMesPtOf1Im = cSetMesPtOf1Im();

        for (auto & [aPName, aCoords] : mVPoints)
        {
            AddGCP(aPName, aCoords, aSensorCamPC, aSetMesGCP, aSetMesPtOf1Im, aGCPSigma);
        }
        
        cSetMesImGCP* aSetMesImGCP = new cSetMesImGCP();
        aSetMesImGCP->AddMes3D(aSetMesGCP);
        aSetMesImGCP->AddMes2D(aSetMesPtOf1Im, &aNewSensorCamPC);
        
        // Solve least squares
        cStdWeighterResidual aStdWeighterResidual = cStdWeighterResidual();
        aBundleAdj->AddGCP("aGCP", 0.0, aStdWeighterResidual, aSetMesImGCP);

        for (int aKIter=0 ; aKIter<20 ; aKIter++)
        {
            aBundleAdj->OneIteration();
        }

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Rot().Mat()(i, j) - aSensorCamPC.Pose().Rot().Mat()(i, j))<1e-8,"Bench1Clinos1Point2Ba failed");
            }
        }

        delete aBundleAdj;
        delete aCalib;
    }

    void Bench2Clinos3Point2Ba(){

        /*
        There is one camera, two clino and three aligned GCP
        All informations are known ( GCP coordinates, camera pose, boresight matrixes between camera and clinos...)

        A small rotation and a small translation are applied to the camera.
        The aim is to retrieve the initial pose
        */

        // GCP in ground geometry
        std::map<std::string, cPt3dr>     mVPoints;
        mVPoints["aP1"] = cPt3dr({0.0, 0.0, 0.0});
        mVPoints["aP2"] = cPt3dr({5.0, 5.0, 5.0});
        mVPoints["aP3"] = cPt3dr({10.0, 10.0, 10.0});
        
        // Sigma on clino measure and GCPs
        tREAL8 aClinoSigma = 1e-5;
        tREAL8 aGCPSigma = 1e-2;

        // Variations on camera position and orientation
        tREAL8 aAmplTr = 10; // Determine initial camera position
        tREAL8 aAmpl = 1e-4;
        tREAL8 aDeltaTrAmpl = 1e-2; // Determine the small translation to the initial camera position

        // Create camera : look at the ground with a small random rotation
        cPtxd<tREAL8,3> aCameraTr = {5.0, 5.0, 100.0};
        cPtxd<tREAL8,3> aRandomTr = cPtxd<tREAL8,3>::PRandC()*aAmplTr;
        cRotation3D<tREAL8> aCameraRot = cRotation3D<tREAL8>::RotFromCanonicalAxes("-ij-k")*cRotation3D<tREAL8>::RandomRot(0.01);
        cIsometry3D<tREAL8> aIsometry(aCameraTr+aRandomTr, aCameraRot);
        eProjPC aTypeProj = eProjPC::eStenope;
        cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::RandomCalib(aTypeProj, 0);
        std::string aCameraName = "aCamera";
        cSensorCamPC aSensorCamPC(aCameraName, aIsometry, aCalib);

        // First Clinometer
        cRotation3D<tREAL8> aBoresight = CreateClino("j-k-i", aCameraRot);
        // Compute needle measure
        tREAL8 aNeedleMeasure =  ComputeNeedleMeasure(aBoresight, aCameraRot);

        // Second Clinometer
        cRotation3D<tREAL8> aBoresight2 = CreateClino("-k-j-i", aCameraRot);
        // Compute needle measure
        tREAL8 aNeedleMeasure2 =  ComputeNeedleMeasure(aBoresight2, aCameraRot);

        // Apply small translation and rotation to the camera
        // It will be initial conditions of bundle adjustment
        // The aim is to retrieve previous translation and rotation
        
        cPtxd<tREAL8,3> aRandomDeltaTr = cPtxd<tREAL8,3>::PRandC()*aDeltaTrAmpl;
        cPtxd<tREAL8,3> aNewCameraTr = aCameraTr+aRandomTr+aRandomDeltaTr;
        cRotation3D<tREAL8> aNewCameraRot = aCameraRot*cRotation3D<tREAL8>::RandomRot(aAmpl);
        cIsometry3D<tREAL8> aNewIsometry(aNewCameraTr, aNewCameraRot);
        std::string aNewCameraName = "aNewCamera";
        cSensorCamPC aNewSensorCamPC(aNewCameraName, aNewIsometry, aCalib);

        // Create BAClino object
        std::string aClinoName = "aClino1";
        cOneCalibClino aCalibClino(aClinoName);
        std::string aClinoName2 = "aClino2";
        cOneCalibClino aCalibClino2(aClinoName2);
        std::vector<cOneCalibClino>aVCalibClino({aCalibClino, aCalibClino2});
        cCalibSetClino* aCalibSetClino = new cCalibSetClino(aCameraName, aVCalibClino);
        cBA_Clino* aBAClino = new cBA_Clino(nullptr, aCalibSetClino);
        
        // Create clino measures
        std::vector<std::string>aVClinoName({aCalibClino.NameClino(), aCalibClino2.NameClino()});
        std::vector<tREAL8>aVAngles({aNeedleMeasure, aNeedleMeasure2});
        std::vector<tREAL8>aVWeights({aClinoSigma, aClinoSigma});
        cClinoMes1Cam aClinoMes1Cam(&aNewSensorCamPC, aVClinoName, aVAngles, aVWeights);
        aBAClino->addClinoMes1Cam(aClinoMes1Cam);

        aBAClino->addClinoWithUK(aClinoName, aBoresight);
        aBAClino->addInitRotClino(aClinoName, aBoresight);
        aBAClino->addClinoWithUK(aClinoName2, aBoresight2);
        aBAClino->addInitRotClino(aClinoName2, aBoresight2);
        aBAClino->setVNamesClino({aClinoName, aClinoName2});
        
        
        // Create cMMVII_BundleAdj object
        cMMVII_BundleAdj* aBundleAdj = new cMMVII_BundleAdj(nullptr);
        aBundleAdj->AddClinoBloc(aBAClino);// add clino
        aBundleAdj->AddCamPC(&aNewSensorCamPC);//add camera
        aBundleAdj->AddBenchSensor(&aNewSensorCamPC);
        aBundleAdj->SetFrozenClinos(".*");//freeze boresight matrix
        aBundleAdj->SetParamFrozenCalib(".*");//freeze camera calibration
        aBundleAdj->setVerbose(false);// Not print residuals
        // Now, only orientation and position of camera are not frozen

        // Add GCPs
        cSetMesGCP aSetMesGCP = cSetMesGCP();
        cSetMesPtOf1Im aSetMesPtOf1Im = cSetMesPtOf1Im();

        for (auto & [aPName, aCoords] : mVPoints)
        {
            AddGCP(aPName, aCoords, aSensorCamPC, aSetMesGCP, aSetMesPtOf1Im, aGCPSigma);
        }
        
        cSetMesImGCP* aSetMesImGCP = new cSetMesImGCP();
        aSetMesImGCP->AddMes3D(aSetMesGCP);
        aSetMesImGCP->AddMes2D(aSetMesPtOf1Im, &aNewSensorCamPC);
        
        // Solve least squares
        cStdWeighterResidual aStdWeighterResidual = cStdWeighterResidual();
        aBundleAdj->AddGCP("aGCP", 0.0, aStdWeighterResidual, aSetMesImGCP);

        for (int aKIter=0 ; aKIter<20 ; aKIter++)
        {
            aBundleAdj->OneIteration();
        }

        
        MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Tr().x()-aSensorCamPC.Pose().Tr().x())<1e-10,"Bench2Clinos3Point2Ba failed");
        MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Tr().y()-aSensorCamPC.Pose().Tr().y())<1e-10,"Bench2Clinos3Point2Ba failed");
        MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Tr().z()-aSensorCamPC.Pose().Tr().z())<1e-10,"Bench2Clinos3Point2Ba failed");

        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(aNewSensorCamPC.Pose().Rot().Mat()(i, j) - aSensorCamPC.Pose().Rot().Mat()(i, j))<1e-8,"Bench2Clinos3Point2Ba failed");
            }
        }

        delete aBundleAdj;
        delete aCalib;
    }

    void BenchClino(cParamExeBench & aParam)
    {
        if (! aParam.NewBench("BAClino")) return;

        BenchClinoFormula();
        BenchClinoBa();
        Bench1Clino1Point2Ba();
        Bench2Clinos3Point2Ba();

        aParam.EndBench();
        return;
    }

}