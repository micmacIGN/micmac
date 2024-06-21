#include "BundleAdjustment.h"

namespace MMVII
{

    cClinoMes1Cam::cClinoMes1Cam(const cSensorCamPC * aCam, const std::vector<std::string> &aVClinoName, const std::vector<tREAL8> & aVAngles,const cPt3dr & aVertAbs) :
        mCam       (aCam),
        mVertInLoc (mCam->Vec_W2L(cPt3dr(0,0,-1)))
    {
        
        for (size_t aK = 0; aK < aVClinoName.size(); aK++)
        {
            mVDir[aVClinoName[aK]] = aVAngles[2*aK];
            mWeights[aVClinoName[aK]] = aVAngles[2*aK+1];
        }
    }
    

    void cClinoMes1Cam::pushObs(std::vector<double> & aVObs) const
    {
        // Push camera orientation
        mCam->Pose().Rot().Mat().PushByLine(aVObs);
        // Push the clino measures
        for (const auto & [aK, aV] : mVDir)
        {
            aVObs.push_back(aV);
        }    
    }

    void cClinoMes1Cam::pushWeights(std::vector<double> & aVWeights) const
    {
        
        for (const auto & [aK, aV] : mWeights)
        {
            for (size_t i = 0; i < 3; i++)
            {
                aVWeights.push_back(aV);
            }
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
        mOmega = cPt3dr(0,0,0);
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
        mEqBlUK  (EqClinoBloc(true,1,true))       
    {
        // Read initial values of clinometers computed by ClinoInit 
        readMeasures();
    }

    cBA_Clino::cBA_Clino
    (
        const cPhotogrammetricProject *aPhProj,
        cCalibSetClino *aCalibSetClino
    ):
        mPhProj  (aPhProj),
        mEqBlUK  (EqClinoBloc(true,1,true)),
        mCalibSetClino (aCalibSetClino)     
    {}

    cBA_Clino::~cBA_Clino()
    {
        delete mCalibSetClino;
    }


    void cBA_Clino::readMeasures()
    {
        cReadFilesStruct aRFS(mNameClino,mFormat,0,-1,'#');
        aRFS.Read();

        mVNamesClino = aRFS.VStrings().at(0);

        cPerspCamIntrCalib * aCalib = nullptr;

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
            
            cSensorCamPC * aCam = mPhProj->ReadCamPC(aNameIm,true,true);
            
            if (aCam != nullptr)
            {
                aCalib = aCam->InternalCalib();
                std::vector<std::string> aVString = aRFS.VStrings().at(aKLine);
                std::vector<tREAL8> aVNum = aRFS.VNums().at(aKLine);
                cClinoMes1Cam aClinoMes1Cam(aCam, aVString, aVNum);
                mVMeasures.push_back(aClinoMes1Cam);
            }
            else
            {
                StdOut() << "Image " << aNameIm << " not found" << std::endl;
            }
            
        }

        if (mVMeasures.size() == 0)
        {
            MMVII_INTERNAL_ERROR("Not enough measures");
        }
        
        // Read initial value
        if (aCalib)
        {
            mCalibSetClino = mPhProj->GetClino(*aCalib);
        }
        
        for(auto & aClinoCal:mCalibSetClino->ClinosCal())
        {
            mClinosWithUK.emplace(std::piecewise_construct, std::make_tuple( aClinoCal.NameClino()), std::make_tuple(aClinoCal.Rot(), aClinoCal.NameClino()));
        }
    }


    void cBA_Clino::pushObs(std::vector<double> & aVObs, const cPt3dr & aCamTr) const
    {
        // Push initial boresight matrixes
        for (const auto & [aK, aV] : mClinosWithUK)
        {
            aV.Rot().Mat().PushByLine(aVObs);
        }

        //Add the vertical
        tPt3dr aVertical = {0.0,0.0,-1.0};
        // If mPhProj is null (for ClinoBench), vertical is (0,0,-1)
        // Else, vertical is defined by the system
        // If no vertical is defined by system (local system for example), return an error
        if (mPhProj)
        {
            tRotR aRotVertical = mPhProj->CurSysCoOri()->getVertical(aCamTr);
            aVertical = aRotVertical.Value(aVertical);
        }

        aVObs.push_back(aVertical.x());
        aVObs.push_back(aVertical.y());
        aVObs.push_back(aVertical.z());
    }

    void cBA_Clino::pushIndex(std::vector<int> & aVInd) const
    {
        for (const auto & [aK, aV] : mClinosWithUK)
        {
            aV.pushIndex(aVInd);
        }
    }


    cPt2dr cBA_Clino::addOneEquation(cResolSysNonLinear<tREAL8> & aSys, cClinoMes1Cam & aMeasure)
    {
        std::vector<double> aVObs;
        std::vector<double> aVWeights;
        std::vector<int> aVInd;

        // Push initial value of boresight matrix and vertical
        pushObs(aVObs, aMeasure.Cam()->Pose().Tr());
        // Push orientation of camera
        aMeasure.pushObs(aVObs);

        pushIndex(aVInd);

        aMeasure.pushWeights(aVWeights);   
        
        aSys.R_CalcAndAddObs(
            mEqBlUK,
            aVInd,
            aVObs,
            cResidualWeighterExplicit<tREAL8>(false, aVWeights)
        );

        cPt2dr aRes(0,1);

        for (size_t aKU = 0; aKU < 6 ; aKU++)
        {
            aRes[0] += Square(mEqBlUK->ValComp(0, aKU));
        }
        
        return cPt2dr(aRes.x()/6.0, 1.0);
    }


    void cBA_Clino::addEquations(cResolSysNonLinear<tREAL8> & aSys)
    {
        cPt2dr aRes(0,0);
        // For each camera
        for (auto aMeasure : mVMeasures)
        {
            aRes += addOneEquation(aSys, aMeasure);
        }

        mRes = aRes/aRes.y();
    }

    void cBA_Clino::printRes() const
    {
        StdOut() << "Residual for clino bloc : " << mRes.x() << std::endl ;
    }
    
    
    void cBA_Clino::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
    {
        std::map<std::string, cClinoWithUK>::iterator itr;
        for (itr = mClinosWithUK.begin(); itr != mClinosWithUK.end(); ++itr)
        {
            aSet.AddOneObj(&itr->second); 
        }
    }


    void cBA_Clino::Save() const
    {
        
        std::vector<cOneCalibClino> aVOneCalibClino = mCalibSetClino->ClinosCal();
        for (auto & aOneCalibClino : aVOneCalibClino)
        {
            auto aClinoWithUK = mClinosWithUK.find(aOneCalibClino.mNameClino);
            aOneCalibClino.mRot = aClinoWithUK->second.Rot();
        }
        
        mPhProj->SaveClino(*mCalibSetClino);
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


    void cBA_Clino::setCalibSetClino(cCalibSetClino* aCalibSetClino)
    {
        mCalibSetClino = aCalibSetClino;
    }

}