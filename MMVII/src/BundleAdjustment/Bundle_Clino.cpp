#include "BundleAdjustment.h"

namespace MMVII
{

    cClinoMes1Cam::cClinoMes1Cam(const cSensorCamPC * aCam, const std::vector<std::string> &aVClinoName, const std::vector<tREAL8> & aVAngles, const std::vector<tREAL8> & aVWeights, const cPt3dr & aVertAbs) :
        mCam       (aCam),
        mVertInLoc (mCam->Vec_W2L(cPt3dr(0,0,-1)))
    {
        MMVII_INTERNAL_ASSERT_bench(aVAngles.size()==aVWeights.size(),"Number of clino measures is different of number of weights measures");
        
        // For each clino, add angle measure and weight 
        for (size_t aK = 0; aK < aVAngles.size(); aK++)
        {
            mVDir[aVClinoName[aK]] = aVAngles[aK];
            mWeights[aVClinoName[aK]] = aVWeights[aK];
        }
    }
    

    void cClinoMes1Cam::pushClinoObs(std::vector<double> & aVObs, const std::string aClinoName)
    {
        // Push camera orientation
        mCam->Pose().Rot().Mat().PushByLine(aVObs);
        
        // Push the clino measure
        aVObs.push_back(mVDir[aClinoName]);
    }

    void cClinoMes1Cam::pushIndex(std::vector<int> & aVInd)
    {
        // Push camera unknowns into the index
        mCam->PushIndexes(aVInd);
    }

    void cClinoMes1Cam::pushClinoWeights(std::vector<double> & aVWeights, const std::string aClinoName)
    {
        
        // For each weight, add it 2 times
        for (size_t i = 0; i < 2; i++)
        {
            aVWeights.push_back(mWeights[aClinoName]);
        }
    }

    cClinoWithUK::cClinoWithUK(tRotR aRot, const std::string & aNameClino):
        mNameClino  (aNameClino),
        mRot  (aRot),
        mOmega (0.0,0.0,0.0)
    {}

    cClinoWithUK::cClinoWithUK():
        mNameClino  (""),
        mRot  (tRotR(cDenseMatrix<tREAL8>(3,3,eModeInitImage::eMIA_Null),false)),
        mOmega (0.0,0.0,0.0)
    {}

    void cClinoWithUK::OnUpdate()
    {
        // Modify rotation by axiator
        mRot = mRot * cRotation3D<tREAL8>::RotFromAxiator(-mOmega);
        // Initialize axiator
        mOmega = cPt3dr(0.0,0.0,0.0);
    }

    void cClinoWithUK::PutUknowsInSetInterval() 
    {
        mSetInterv->AddOneInterv(mOmega);
    }

    void  cClinoWithUK::GetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
    {
        aGAIP.TestParam(this, &( mOmega.x())    ,"Wx");
        aGAIP.TestParam(this, &( mOmega.y())    ,"Wy");
        aGAIP.TestParam(this, &( mOmega.z())    ,"Wz");
    }

    void cClinoWithUK::pushIndex(std::vector<int> & aVInd) const
    {
        // Push index of unknowns in aVInd
        for (int i = IndUk0(); i < IndUk1(); i++)
        {
            aVInd.push_back(i);
        } 
    }



    cBA_Clino::cBA_Clino
    (
        const cPhotogrammetricProject *aPhProj,
        const std::string & aNameClino,
        const std::string & aFormat,
        const std::vector<std::string> & aPrePost
    ):
        mPhProj  (aPhProj),
        mNameClino   (aNameClino),
        mFormat   (aFormat),
        mPrePost  (aPrePost),
        mEqBlUK  (EqClinoBloc(true,1,true)),
        mEqBlUKRot  (EqClinoRot(true,1,true))     
    {
        // Read initial values of clinometers computed by ClinoInit 
        readMeasures();
    }

    cBA_Clino::cBA_Clino
    (
        const cPhotogrammetricProject *aPhProj
    ):
        mPhProj  (aPhProj),
        mEqBlUK  (EqClinoBloc(true,1,true)),
        mEqBlUKRot  (EqClinoRot(true,1,true))
    {}


    void cBA_Clino::readMeasures()
    {
        // Read clino observations file
        cReadFilesStruct aRFS(mNameClino,mFormat,0,-1,'#');
        aRFS.Read();

        // Get clino names
        mVNamesClino = aRFS.VStrings().at(0);

        // Initialize cameara calibration (to get after its orientation)
        cPerspCamIntrCalib * aCalib = nullptr;

        // For each measure
        size_t aNbMeasures = aRFS.NbRead();
        for (size_t aKLine=0 ; aKLine<aNbMeasures ; aKLine++)
        {
            // get image name
            std::string aNameIm =  aRFS.VNameIm().at(aKLine);

            // If prepost is defined, add prepost before and after aNameIm
            if (mPrePost.size()==2)
            {
                aNameIm = mPrePost[0] +  aNameIm + mPrePost[1];
            }
            
            // read camera orientation if files
            cSensorCamPC * aCam = mPhProj->ReadCamPC(aNameIm,true,true);
            
            if (aCam != nullptr)
            {
                // Get camera calibration
                aCalib = aCam->InternalCalib();
                // Get clino names for this measure
                std::vector<std::string> aVString = aRFS.VStrings().at(aKLine);
                // Get clino measures for this measure
                std::vector<tREAL8> aVNum = aRFS.VNums().at(aKLine);

                // Divide aVNum in clino measures and clino weights                
                std::vector<tREAL8> aVClino;
                std::vector<tREAL8> aVWeights;
                for (size_t aK = 0; aK < aVString.size(); aK++)
                {
                    aVClino.push_back(aVNum[2*aK]);
                    aVWeights.push_back(aVNum[2*aK+1]);
                }

                // Add measure
                cClinoMes1Cam aClinoMes1Cam(aCam, aVString, aVClino, aVWeights);
                mVMeasures.push_back(aClinoMes1Cam);
            }
            else
            {
                StdOut() << "Image " << aNameIm << " not found" << std::endl;
            }
            
        }

        // If no measures, return an error
        if (mVMeasures.size() == 0)
        {
            MMVII_INTERNAL_ERROR("Not enough measures");
        }

        mCameraName = aCalib->Name();

        // Create cClinoWithUK objects and add initial rotation in a map
        for(auto & aClinoName:mVNamesClino)
        {
            cOneCalibClino* aCalibClino = mPhProj->GetClino(*aCalib, aClinoName);
            mClinosWithUK.emplace(std::piecewise_construct, std::make_tuple( aCalibClino->NameClino()), std::make_tuple(aCalibClino->Rot(), aCalibClino->NameClino()));
            mInitRotClino[aCalibClino->NameClino()] = aCalibClino->Rot();
            delete aCalibClino;
        }
    }


    void cBA_Clino::pushClinoObs(std::vector<double> & aVObs, const cPt3dr & aCamTr, const std::string aClinoName)
    {
        // Push initial boresight matrixes
        mClinosWithUK[aClinoName].Rot().Mat().PushByLine(aVObs);

        //Push the vertical
        tPt3dr aVertical = {0.0,0.0,-1.0};
        // If mPhProj is null (for ClinoBench), vertical is (0,0,-1)
        // Else, vertical is defined by the system
        // If no vertical is defined by system (local system for example), return an error
        if (mPhProj)
        {
            tRotR aRotVertical = mPhProj->CurSysCoOri()->getRot2Vertical(aCamTr);
            aVertical = aRotVertical.Value(aVertical);
        }

        aVObs.push_back(aVertical.x());
        aVObs.push_back(aVertical.y());
        aVObs.push_back(aVertical.z());
    }

    void cBA_Clino::pushRotObs(std::vector<double> & aVObs, const std::string aClino1, const std::string aClino2)
    {
        // Push boresight matrixes
        mClinosWithUK[aClino1].Rot().Mat().PushByLine(aVObs);
        mClinosWithUK[aClino2].Rot().Mat().PushByLine(aVObs);

        // Push initial relative orientation between the two matrix
        tRotR aClinoRot1 = mInitRotClino[aClino1];
        tRotR aClinoRot2 = mInitRotClino[aClino2];

        cDenseMatrix<tREAL8> aInitRelativeRot = aClinoRot2.Mat() * aClinoRot1.Mat().Transpose();
        aInitRelativeRot.PushByLine(aVObs);
    }

    void cBA_Clino::pushClinoIndex(std::vector<int> & aVInd, const std::string aClinoName)
    {
        // Push index of clino unknowns
        mClinosWithUK[aClinoName].pushIndex(aVInd);
    }

    void cBA_Clino::pushRotIndex(std::vector<int> & aVInd, const std::string aClino1, const std::string aClino2)
    {
        // Push index of clino unknowns
        mClinosWithUK[aClino1].pushIndex(aVInd);
        mClinosWithUK[aClino2].pushIndex(aVInd);
    }

    void cBA_Clino::pushRotWeights(std::vector<double> & aVWeights)
    {
        for (size_t i = 0; i < 9; i++)
        {
            aVWeights.push_back(1);
        }
    }


    cPt2dr cBA_Clino::addOneClinoEquation(cResolSysNonLinear<tREAL8> & aSys, cClinoMes1Cam & aMeasure, const std::string aClinoName)
    {
        // Vector with observations
        std::vector<double> aVObs;

        // Vector with weights
        std::vector<double> aVWeights;

        // Vector with index of unknowns
        std::vector<int> aVInd;

        // Push initial value of boresight matrix and vertical
        pushClinoObs(aVObs, aMeasure.Cam()->Pose().Tr(), aClinoName);
        
        // Push orientation of camera
        aMeasure.pushClinoObs(aVObs, aClinoName);

        // Push index of unknowns (Boresight matrix)
        pushClinoIndex(aVInd, aClinoName);

        //Push index of unknowns (Camera orientation)
        aMeasure.pushIndex(aVInd);

        // Push weights
        aMeasure.pushClinoWeights(aVWeights, aClinoName);   
        
        // Compute solution for the unknowns defined in aVInd
        aSys.R_CalcAndAddObs(
            mEqBlUK,
            aVInd,
            aVObs,
            cResidualWeighterExplicit<tREAL8>(false, aVWeights)
        );

        
        // Compute residuals
        cPt2dr aRes(0,1);

        // For the two equations defined in cFormulaClinoBloc, add the residuals
        for (size_t aKU = 0; aKU < 2 ; aKU++)
        {
            aRes[0] += Square(mEqBlUK->ValComp(0, aKU));
        }
        
        // Return the mean of residuals
        return cPt2dr(aRes.x()/2.0, 1.0);
    }

    cPt2dr cBA_Clino::addOneRotEquation(cResolSysNonLinear<tREAL8> & aSys, const std::string aClino1, const std::string aClino2)
    {
        // Vector with observations
        std::vector<double> aVObs;

        // Vector with weights
        std::vector<double> aVWeights;

        // Vector with index of unknowns
        std::vector<int> aVInd;

        // Push values of the two boresight matrix and initial relative orientation between these two matrix
        pushRotObs(aVObs, aClino1, aClino2);

        // Push index of unknowns
        pushRotIndex(aVInd, aClino1, aClino2);

        // Push weights
        pushRotWeights(aVWeights);

        // Compute solution for the unknowns defined in aVInd
        aSys.R_CalcAndAddObs(
            mEqBlUKRot,
            aVInd,
            aVObs,
            cResidualWeighterExplicit<tREAL8>(false, aVWeights)
        );

        // Compute residuals
        cPt2dr aRes(0,1);

        // For the nine equations defined in cFormulaClinoRot, add the residuals
        for (size_t aKU = 0; aKU < 9 ; aKU++)
        {
            aRes[0] += Square(mEqBlUKRot->ValComp(0, aKU));
        }
        
        // Return the mean of residuals
        return cPt2dr(aRes.x()/9.0, 1.0);

    }


    void cBA_Clino::addEquations(cResolSysNonLinear<tREAL8> & aSys)
    {
        // Initialize residuals
        cPt2dr aClinoRes(0,0);
        cPt2dr aRotRes(0,0);
        
        // For each measure and each clinometer, solve least squares and add residual to residuals
        for (auto aMeasure : mVMeasures)
        {
            for (auto aClinoName : mVNamesClino)
            {
                aClinoRes += addOneClinoEquation(aSys, aMeasure, aClinoName);
            }
        }


        for (size_t aK1 = 0; aK1 < mVNamesClino.size(); aK1++)
        {
            std::string aClino1 = mVNamesClino[aK1];
            for (size_t aK2 = aK1+1; aK2 < mVNamesClino.size(); aK2++)
            {
                std::string aClino2 = mVNamesClino[aK2];
                aRotRes += addOneRotEquation(aSys, aClino1, aClino2);
            }
            
        }
        

        // Return mean residual
        if (aClinoRes.y()!=0)
        {
            mClinoRes = aClinoRes/aClinoRes.y();
        }

        if (aRotRes.y()!=0)
        {
            mRotRes = aRotRes/aRotRes.y();
        }
    }

    void cBA_Clino::printRes() const
    {
        StdOut() << "Residual for clino formula : " << sqrt(mClinoRes.x()) << std::endl ;
        StdOut() << "Residual for rot formula : " << sqrt(mRotRes.x()) << std::endl ;
    }
    
    
    void cBA_Clino::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
    {
        // Add each clino to the system
        for (auto & [aClinoName, aClinoWithUK] : mClinosWithUK)
        {
            aSet.AddOneObj(&aClinoWithUK); 
        }
    }


    void cBA_Clino::SetFrozenVar(cResolSysNonLinear<tREAL8> & aSys, std::string aPatFrozenClino)  
    {
        // Froze boresight matrix of clinos described by aPatFrozenClino
        tNameSelector aSel = AllocRegex(aPatFrozenClino);   

        for (auto & [aClinoName, aClinoWithUK] : mClinosWithUK)
        {
            if (aSel.Match(aClinoName))
            {
                for (auto i = aClinoWithUK.IndUk0(); i < aClinoWithUK.IndUk1(); i++)
                {
                    aSys.SetFrozenVarCurVal(i);
                }
            }
        }
    }


    void cBA_Clino::Save()
    {
        // Save relative orientations between clino and reference camera
        
        std::vector<cOneCalibClino> aVCalibClino;
        for (auto & [aClinoName, aClinoWithUK] : mClinosWithUK)
        {
            cOneCalibClino aOneCalibClino = cOneCalibClino(aClinoName);
            aOneCalibClino.mRot = aClinoWithUK.Rot();
            aOneCalibClino.mCameraName = mCameraName;
            aVCalibClino.push_back(aOneCalibClino);
        }
        
        cCalibSetClino aCalibSetClino = cCalibSetClino(mCameraName, aVCalibClino);
        mPhProj->SaveClino(aCalibSetClino);
    }

    void cBA_Clino::addClinoMes1Cam(const cClinoMes1Cam & aClinoMes1Cam)
    {
        mVMeasures.push_back(aClinoMes1Cam);
    }

    void cBA_Clino::addClinoWithUK(const std::string & aClinoName, tRotR & aRot)
    {
        mClinosWithUK.emplace(std::piecewise_construct, std::make_tuple( aClinoName), std::make_tuple(aRot, aClinoName));
    }

    std::vector<tRotR>  cBA_Clino::ClinosWithUKRot() const
    {
        std::vector<tRotR> aVRotR;
        
        for (const auto & [aK, aV] : mClinosWithUK)
        {
            aVRotR.push_back(aV.Rot());
        }
        return aVRotR;
    }

}