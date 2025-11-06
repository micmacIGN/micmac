#include "MMVII_StaticLidar.h"
#include "MMVII_Sensor.h"
#include "MMVII_Geom3D.h"

/**
   \file importStaticScan.cpp

   \brief import static scan into instrument geometry
*/


namespace MMVII
{
/* ********************************************************** */
/*                                                            */
/*                 cAppli_ImportStaticScan                    */
/*                                                            */
/* ********************************************************** */

class cAppli_ImportStaticScan : public cMMVII_Appli
{
public :
    cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    std::vector<std::string>  Samples() const override;

    void getAnglesMinMax();
    void estimatePhiStep();
    void computeLineCol();
    tREAL8 doVerticalize(); // returns correction angle applied
    void testLineColError();
    void computeAngStartStep();
    void exportThetas(const std::string & aFileName, int aNbThetas, bool aCompareToCol);
private :
    cPhotogrammetricProject  mPhProj;

    // Mandatory Arg
    std::string              mNameFile;

    // Optional Arg
    std::string              mTransfoIJK;
    tREAL8 mPhiStepApprox;

    tREAL8 mThetaStepApprox;

    cStaticLidar mSL_data;
};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mPhProj         (*this),
    mTransfoIJK     ("ijk"),
    mPhiStepApprox  (NAN),
    mSL_data        (mNameFile, cIsometry3D<tREAL8>({}, cRotation3D<tREAL8>::Identity()), nullptr)
{
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
           <<  mPhProj.DPStaticLidar().ArgDirOutMand()
           <<  Arg2007(mSL_data.mStationName ,"Station name",{eTA2007::Topo}) // TODO: change type to future station
           <<  Arg2007(mSL_data.mScanName ,"Scan name",{eTA2007::Topo}) // TODO: change type to future scan
        ;
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mTransfoIJK,"Transfo","Transfo to have primariy rotation axis as Z and X as theta origin",{{eTA2007::HDV}})
        ;
}


void cAppli_ImportStaticScan::estimatePhiStep()
{
    StdOut() << "estimatePhiStep\n";
    // find phi min diff
    // successive phi diff is useful, but only if we are in the same scanline, and we are not too close to the pole with vertical error
    tREAL8 previousTheta = NAN;
    tREAL8 previousPhi = NAN;
    mPhiStepApprox = INFINITY; // signed value that is min in abs. For successive points on one column
    tREAL8 angularPrecisionInSteps = 0.01; // we suppose that theta changes slower than phi... prevents scanline change and pole error
    for (const auto & aPtAng: mSL_data.mSL_importer.mVectPtsTPD)
    {
        if (aPtAng.z()<mSL_data.mSL_importer.DistMinToExist()) continue;
        auto aDiffPhi = aPtAng.y()-previousPhi;
        auto aDiffTheta = aPtAng.x()-previousTheta;
        if (fabs(aDiffTheta)<fabs(mPhiStepApprox)*angularPrecisionInSteps) // we are on the same scanline
        {
            if (fabs(aDiffPhi)<fabs(mPhiStepApprox))
            {
                //std::cout<<"with prev "<<previousTheta<< " "<< previousPhi<< "  curr "<<aPtAng<<":\n";
                //std::cout<<"up: "<<minDiffPhi <<" " <<aDiffPhi<<"\n";
                mPhiStepApprox = aDiffPhi;
            }
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "phiStep " << mPhiStepApprox << "\n";

    // estimate mThetaStepApprox
    // search for nb points in one col
    tREAL8 aMinAngToZenith = 0.1;
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsTPD.size()-1; ++i)
    {
        auto aPtAng1 = mSL_data.mSL_importer.mVectPtsTPD[i];
        if (aPtAng1.z()<mSL_data.mSL_importer.DistMinToExist()) continue;
        if (fabs(fabs(aPtAng1.y())-M_PI/2)<aMinAngToZenith) continue;
        for (size_t j=i+1; j<mSL_data.mSL_importer.mVectPtsTPD.size(); ++j)
        {
            auto aPtAng2 = mSL_data.mSL_importer.mVectPtsTPD[j];
            if (aPtAng2.z()<mSL_data.mSL_importer.DistMinToExist()) continue;
            if (fabs(fabs(aPtAng2.y())-M_PI/2)<aMinAngToZenith) continue;
            if ((aPtAng2.y()-aPtAng1.y())/mPhiStepApprox < angularPrecisionInSteps)
            {
                mThetaStepApprox = aPtAng2.x()-aPtAng1.x();
                StdOut() << "ThetaStep " << mThetaStepApprox << "\n";
                return;
            }
        }
    }
}


void cAppli_ImportStaticScan::computeLineCol()
{
    computeAngStartStep();
    if (mSL_data.mSL_importer.HasRowCol())
        return; // nothing to do
    tREAL8 aColChangeDetectorInPhistep = 20;

    // compute line and col for each point
    mSL_data.mSL_importer.mVectPtsLine.resize(mSL_data.mSL_importer.mVectPtsXYZ.size());
    mSL_data.mSL_importer.mVectPtsCol.resize(mSL_data.mSL_importer.mVectPtsXYZ.size());
    mSL_data.mMaxCol = 0;
    tREAL8 previousPhi = NAN;
    //tREAL8 previousTheta = NAN;
    mSL_data.mMaxLine = 0;
    int aCurrLine = 0;
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_data.mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_data.mSL_importer.DistMinToExist())
        {
            mSL_data.mSL_importer.mVectPtsLine[i] = 0;
            mSL_data.mSL_importer.mVectPtsCol[i] = 0;
            aCurrLine++;
            continue;
        }
        if (mSL_data.mSL_importer.NoMiss())
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-mSL_data.mPhiStart)/fabs(mPhiStepApprox);

        if (aCurrLine>mSL_data.mMaxLine)
            mSL_data.mMaxLine = aCurrLine;
        //StdOut() << aPtAng.y() << " " << previousPhi << " " << -(aPtAng.y()-previousPhi)/mPhiStepApprox
        //         << " " << aCurrLine << " " << mSL_data.mMaxLine << "\n";
        if (-(aPtAng.y()-previousPhi)/mPhiStepApprox > aColChangeDetectorInPhistep)
        {
            mSL_data.mMaxCol++;
            aCurrLine=0;
        }
        mSL_data.mSL_importer.mVectPtsLine[i] = aCurrLine;
        mSL_data.mSL_importer.mVectPtsCol[i] = mSL_data.mMaxCol;
        //previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "Max col found: "<<mSL_data.mMaxCol<<"\n";
    StdOut() << "Max line found: "<<mSL_data.mMaxLine<<"\n";

    StdOut() << "Image size: "<<cPt2di(mSL_data.mMaxCol+1, mSL_data.mMaxLine+1)<<"\n";

    MMVII_INTERNAL_ASSERT_tiny((mSL_data.mMaxCol>0) && (mSL_data.mMaxLine>0),
                               "Image size found incorrect")

    int mThetaDir = -1; // TODO: compute!
    // precise estimation of mThetaStep and mPhiStep;
    int aCurrCol = 0;
    tREAL8 aAvgTheta = 0.;
    long aNbTheta = 0;
    tREAL8 aLoopCorrection = 0.;
    std::vector<tREAL8> allThetaAvg(mSL_data.mMaxCol+1);
    int aLineLimitDown = mSL_data.mMaxLine * 0.2;
    int aLineLimitUp = mSL_data.mMaxLine * 0.8;
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_data.mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_data.mSL_importer.DistMinToExist())
            continue;
        if ((mSL_data.mSL_importer.mVectPtsLine[i]>aLineLimitUp) || (mSL_data.mSL_importer.mVectPtsLine[i]<aLineLimitDown))
            continue; // avoid points where theta is not well defined
        if (aCurrCol!=mSL_data.mSL_importer.mVectPtsCol[i])
        {
            aAvgTheta = aAvgTheta/aNbTheta + aLoopCorrection;
            if ((aCurrCol>0) && (fabs(aAvgTheta-allThetaAvg[aCurrCol-1])>M_PI))
            {
                aAvgTheta -= aLoopCorrection;
                aLoopCorrection += 2*M_PI * mThetaDir;
                aAvgTheta += aLoopCorrection;
            }
            allThetaAvg[aCurrCol] = aAvgTheta;
            aCurrCol = mSL_data.mSL_importer.mVectPtsCol[i];
            aAvgTheta = 0;
            aNbTheta = 0;
        }
        aAvgTheta += aPtAng.x();
        ++aNbTheta;
    }
    aAvgTheta = aAvgTheta/aNbTheta + aLoopCorrection;
    if ((aCurrCol>0) && (fabs(aAvgTheta-allThetaAvg[aCurrCol-1])>M_PI))
    {
        aAvgTheta -= aLoopCorrection;
        aLoopCorrection += 2*M_PI * mThetaDir;
        aAvgTheta += aLoopCorrection;
    }
    allThetaAvg[aCurrCol] = aAvgTheta;

    std::vector<tREAL8> allThetaDiff(mSL_data.mMaxCol);
    for (unsigned int i=0;i<allThetaDiff.size();++i)
        allThetaDiff[i] = allThetaAvg[i+1]-allThetaAvg[i];
    tREAL8 aThetaStep = NonConstMediane(allThetaDiff);
    StdOut() << "ThetaStep: " << aThetaStep << "\n";
}

void cAppli_ImportStaticScan::computeAngStartStep()
{
    tREAL8 aMinAngToZenith = 0.1;
    if (mSL_data.mSL_importer.HasRowCol())
    {
        // search for 2 points with diff line/col to estimate steps
        long a1stPti = -1;
        for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsTPD.size(); ++i)
        {
            auto & aPtAng = mSL_data.mSL_importer.mVectPtsTPD[i];
            if ((aPtAng.z()>mSL_data.mSL_importer.DistMinToExist()) && (fabs(fabs(aPtAng.y())-M_PI/2)>aMinAngToZenith))
            {
                if (a1stPti<0)
                    a1stPti = i;
                else {
                    if ((mSL_data.mSL_importer.mVectPtsCol[i] != mSL_data.mSL_importer.mVectPtsCol[a1stPti]) && (mSL_data.mSL_importer.mVectPtsLine[i] != mSL_data.mSL_importer.mVectPtsLine[a1stPti]))
                    {
                        auto & a1stPtAng = mSL_data.mSL_importer.mVectPtsTPD[a1stPti];
                        mSL_data.mPhiStep = (aPtAng.y()-a1stPtAng.y())/(mSL_data.mSL_importer.mVectPtsLine[i]-mSL_data.mSL_importer.mVectPtsLine[a1stPti]);
                        mSL_data.mThetaStep = (aPtAng.x()-a1stPtAng.x())/(mSL_data.mSL_importer.mVectPtsCol[i]-mSL_data.mSL_importer.mVectPtsCol[a1stPti]);
                        mSL_data.mPhiStart = a1stPtAng.y() - mSL_data.mPhiStep * mSL_data.mSL_importer.mVectPtsLine[a1stPti];
                        mSL_data.mThetaStart = a1stPtAng.x() - mSL_data.mThetaStep * mSL_data.mSL_importer.mVectPtsCol[a1stPti];
                        break;
                    }
                }
            }
        }
        // make a better approx using a second point near the end
        for (long i=(long)mSL_data.mSL_importer.mVectPtsTPD.size()-1; i>a1stPti; --i)
        {
            if ((mSL_data.mSL_importer.mVectPtsTPD[i].z()>mSL_data.mSL_importer.DistMinToExist()) && (fabs(fabs(mSL_data.mSL_importer.mVectPtsTPD[i].y())-M_PI/2)>aMinAngToZenith))
            {
                if ((mSL_data.mSL_importer.mVectPtsCol[i] != mSL_data.mSL_importer.mVectPtsCol[a1stPti]) && (mSL_data.mSL_importer.mVectPtsLine[i] != mSL_data.mSL_importer.mVectPtsLine[a1stPti]))
                {
                    auto & a1stPtAng = mSL_data.mSL_importer.mVectPtsTPD[a1stPti];
                    auto a2ndPtAng = mSL_data.mSL_importer.mVectPtsTPD[i]; // copy to unroll
                    tREAL8 aTheta = mSL_data.mSL_importer.mVectPtsCol[i]*mSL_data.mThetaStep + mSL_data.mThetaStart;
                    a2ndPtAng.x() = toMinusPiPlusPi(a2ndPtAng.x(), aTheta);
                    mSL_data.mPhiStep = (a2ndPtAng.y()-a1stPtAng.y())/(mSL_data.mSL_importer.mVectPtsLine[i]-mSL_data.mSL_importer.mVectPtsLine[a1stPti]);
                    mSL_data.mThetaStep = (a2ndPtAng.x()-a1stPtAng.x())/(mSL_data.mSL_importer.mVectPtsCol[i]-mSL_data.mSL_importer.mVectPtsCol[a1stPti]);
                    mSL_data.mPhiStart = a1stPtAng.y() - mSL_data.mPhiStep * mSL_data.mSL_importer.mVectPtsLine[a1stPti];
                    mSL_data.mThetaStart = a1stPtAng.x() - mSL_data.mThetaStep * mSL_data.mSL_importer.mVectPtsCol[a1stPti];
                    StdOut() << "i1 i2: " << a1stPti << " " << i << ", "
                             << mSL_data.mSL_importer.mVectPtsCol[a1stPti] << " " << mSL_data.mSL_importer.mVectPtsLine[a1stPti] << " "
                             << mSL_data.mSL_importer.mVectPtsCol[i] << " " << mSL_data.mSL_importer.mVectPtsLine[i] << "\n";
                    StdOut() << a1stPtAng.x() << " " << a1stPtAng.y() << " "
                             << a2ndPtAng.x() << " " << a2ndPtAng.y() << "\n";
                    StdOut() << "PhiStart: " << mSL_data.mPhiStart << ", "
                             << "PhiStep: " << mSL_data.mPhiStep << ", "
                             << "ThetaStart: " << mSL_data.mThetaStart << ", "
                             << "ThetaStep: " << mSL_data.mThetaStep << "\n";
                    StdOut() << "PhiEnd: " << mSL_data.mPhiStart+mSL_data.mMaxLine*mSL_data.mPhiStep << ", "
                             << "ThetaEnd: " << mSL_data.mThetaStep+mSL_data.mMaxCol*mSL_data.mThetaStep << "\n";
                    return;
                }
            }
        }
    } else {
        if (mSL_data.mSL_importer.NoMiss())
        {
            MMVII_INTERNAL_ASSERT_tiny(false, "No computeAngStartEnd() without linecol for now")
        } else {
            MMVII_INTERNAL_ASSERT_tiny(false, "No computeAngStartEnd() for sparse cloud for now")
        }
    }
}

void cAppli_ImportStaticScan::testLineColError()
{
    tREAL8 aMaxThetaError = -1.;
    tREAL8 aMaxPhiError = -1.;
    tREAL8 aMinAngToZenith = 0.1;
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_data.mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_data.mSL_importer.DistMinToExist())
            continue;
        if (fabs(fabs(aPtAng.y())-M_PI/2)<aMinAngToZenith)
            continue; // no stats on points too close to undefined theta
        tREAL8 aTheta = mSL_data.mSL_importer.mVectPtsCol[i]*mSL_data.mThetaStep + mSL_data.mThetaStart;
        tREAL8 aPhi = mSL_data.mSL_importer.mVectPtsLine[i]*mSL_data.mPhiStep + mSL_data.mPhiStart;
        aTheta = toMinusPiPlusPi(aTheta, aPtAng.x());
        if (fabs(aTheta-aPtAng.x())>aMaxThetaError)
        {
            aMaxThetaError = fabs(aTheta-aPtAng.x());
            //StdOut() << i << " " << aPtAng <<" => " << mSL_data.mSL_importer.mVectPtsCol[i] << " " << mSL_data.mSL_importer.mVectPtsLine[i] << " " << aTheta << " " << aPhi << " => " << aMaxThetaError << " " << aMaxPhiError << "\n";
        }
        if (fabs(aPhi-aPtAng.y())>aMaxPhiError)
        {
            aMaxPhiError = fabs(aPhi-aPtAng.y());
            //StdOut() << i << " " << aPtAng <<" => " << mSL_data.mSL_importer.mVectPtsCol[i] << " " << mSL_data.mSL_importer.mVectPtsLine[i] << " " << aTheta << " " << aPhi << " => " << aMaxThetaError << " " << aMaxPhiError << "\n";
        }
    }
    StdOut() << "Max ang errors: " << aMaxThetaError << " " << aMaxPhiError <<"\n";
}


tREAL8 cAppli_ImportStaticScan::doVerticalize()
{
    StdOut() << "Verticalizing..." << std::endl;
    // estimate verticalization correction if scanner with compensator
    int aColChangeDetectorInPhistep = 100;
    int aNbPlanes = 20; // try to get several planes for instrument primariy axis estimation
    float aCorrectPlanePhiRange = 40*M_PI/180; // try to get points with this phi diff in a scanline
    int aColPlaneStep = mSL_data.mMaxCol / aNbPlanes;
    int aLineGoodRange = aCorrectPlanePhiRange/fabs(mPhiStepApprox);
    if (aLineGoodRange > mSL_data.mMaxLine*0.5)
        aLineGoodRange = mSL_data.mMaxLine*0.5; // for small scans, use full height

    int aTargetCol = 0; // the next we search for
    int aTargetLine = 0;
    float previousPhi = NAN;
    //tREAL8 previousTheta = NAN;
    int aCurrCol = 0;
    int aCurrLine = 0;
    std::vector<std::tuple<cPt3dr, cPt3dr, cPt3dr>> aVPtsPlanes; // list of triplets to find vertical planes
    cPt3dr * aPtBottom = nullptr;
    cPt3dr * aPtTop = nullptr;
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsTPD.size(); ++i)
    {
        // TODO: factorize xyz points list to linecol!
        auto aPtAng = mSL_data.mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_data.mSL_importer.DistMinToExist())
        {
            aCurrLine++;
            continue;
        }
        if (mSL_data.mSL_importer.NoMiss())
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-mSL_data.mPhiStart)/fabs(mPhiStepApprox);

        if (-(aPtAng.y()-previousPhi)/mPhiStepApprox > aColChangeDetectorInPhistep)
        {
            aCurrCol++;
            aCurrLine=0;
        }
        //previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
        if (aCurrCol == aTargetCol)
        {
            if (!aPtBottom)
            {
                aPtBottom = &mSL_data.mSL_importer.mVectPtsXYZ[i];
                aTargetLine = aCurrLine + aLineGoodRange;
            }
            else if ((aCurrLine>aTargetLine)&&(!aPtTop))
            {
                aPtTop = &mSL_data.mSL_importer.mVectPtsXYZ[i];
                aVPtsPlanes.push_back( {cPt3dr(0.,0.,0.),(*aPtBottom)/Norm2(*aPtBottom), (*aPtTop)/Norm2(*aPtTop)} );
                aPtBottom = nullptr;
                aPtTop = nullptr;
                aTargetCol = aCurrCol + aColPlaneStep;
            }
        }
    }

    std::vector<cPlane3D> aVPlanes;
    for (auto & [aP0,aP1,aP2] :aVPtsPlanes)
    {
        aVPlanes.push_back(cPlane3D::From3Point(aP0,aP1,aP2));
    }
    tSeg3dr aSegVert = cPlane3D::InterPlane(aVPlanes, 3);
    if (aSegVert.V12().z()<0)
        aSegVert.Swap(); // make sure to have a vector going up
    StdOut() << "Vert: " << aSegVert.V12() << "\n";

    mSL_data.mVertRot = cRotation3D<tREAL8>::CompleteRON(aSegVert.V12(),2);

    // update xyz and tpd coordinates
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        mSL_data.mSL_importer.mVectPtsXYZ[i] = mSL_data.mVertRot.Inverse(mSL_data.mSL_importer.mVectPtsXYZ[i]);
    }
    mSL_data.mSL_importer.convertToThetaPhiDist();
    // update line col
    computeLineCol();

    return mSL_data.mVertRot.Angle();
}

void cAppli_ImportStaticScan::exportThetas(const std::string & aFileName, int aNbThetas, bool aCompareToCol)
{
    StdOut() << "Export thetas\n";
    // export thetas on several cols
    std::fstream file_thetas;
    file_thetas.open(aFileName, std::ios_base::out);
    long aTargetCol = 0; // the next we search for
    bool isFirstofCol = true;
    tREAL8 aThetaCol = 0.; // theta of first point
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        if ( mSL_data.mSL_importer.mVectPtsTPD[i].z()<mSL_data.mSL_importer.DistMinToExist())
            continue;
        if (mSL_data.mSL_importer.mVectPtsCol[i]==aTargetCol)
        {
            auto &aPtTPD = mSL_data.mSL_importer.mVectPtsTPD[i];
            if (isFirstofCol)
            {
                if (aCompareToCol)
                {
                    aThetaCol = mSL_data.mThetaStart + mSL_data.mThetaStep * aTargetCol;
                    aThetaCol = toMinusPiPlusPi(aThetaCol);
                    StdOut() <<  aPtTPD.x() << " " << aThetaCol << " " << aPtTPD.x() - aThetaCol << "\n";
                }
                else
                    aThetaCol = aPtTPD.x();
                isFirstofCol = false;
            }
            tREAL8 aError = aPtTPD.x() - aThetaCol;
            aError= toMinusPiPlusPi(aError);
            //file_thetas << mSL_data.mSL_importer.mVectPtsLine[i] << " " <<  <<"\n";
            file_thetas << aError <<" ";
        }
        if (mSL_data.mSL_importer.mVectPtsCol[i] > aTargetCol)
        {
            file_thetas << "\n";
            aTargetCol = mSL_data.mSL_importer.mVectPtsCol[i] + mSL_data.mMaxCol / aNbThetas;
            isFirstofCol = true;
            if (aTargetCol>=mSL_data.mMaxCol)
                aTargetCol=mSL_data.mMaxCol;
        }
    }
    file_thetas << "\n";
    file_thetas.close();
}

int cAppli_ImportStaticScan::Exe()
{
    mPhProj.FinishInit();
    mSL_data.mSL_importer.read(mNameFile);

    mSL_data.mMaxCol = mSL_data.mSL_importer.MaxCol();
    mSL_data.mMaxLine = mSL_data.mSL_importer.MaxLine();

    StdOut() << "Read data: ";
    if (mSL_data.mSL_importer.HasCartesian())
        StdOut() << mSL_data.mSL_importer.mVectPtsXYZ.size() << " cartesian points";
    if (mSL_data.mSL_importer.HasSpherical())
        StdOut() << mSL_data.mSL_importer.mVectPtsTPD.size() << " spherical points";
    if (mSL_data.mSL_importer.HasIntensity())
        StdOut() << " with intensity";
    if (mSL_data.mSL_importer.HasRowCol())
        StdOut() << " with row-col";
    StdOut() << "\n";

    if (mSL_data.mSL_importer.HasCartesian() && !mSL_data.mSL_importer.HasSpherical())
    {
        cRotation3D<tREAL8>  aRotFrame = cRotation3D<tREAL8>::RotFromCanonicalAxes(mTransfoIJK);
        // apply rotframe to original points
        for (auto & aPtXYZ : mSL_data.mSL_importer.mVectPtsXYZ)
        {
            aPtXYZ = aRotFrame.Value(aPtXYZ);
        }
        mSL_data.mSL_importer.convertToThetaPhiDist();
    } else if (!mSL_data.mSL_importer.HasCartesian() && mSL_data.mSL_importer.HasSpherical()) // mTransfoIJK not used if spherical
    {
        mSL_data.mSL_importer.convertToXYZ();
    }

    MMVII_INTERNAL_ASSERT_tiny(!mSL_data.mSL_importer.mVectPtsXYZ.empty(),"Error reading "+mNameFile);
    if (mSL_data.mSL_importer.HasIntensity())
    {
        MMVII_INTERNAL_ASSERT_tiny(mSL_data.mSL_importer.mVectPtsXYZ.size()==mSL_data.mSL_importer.mVectPtsIntens.size(),"Error reading "+mNameFile);
    }

    StdOut() << "Cartesian sample:\n";
    for (size_t i=0; (i<10)&&(i<mSL_data.mSL_importer.mVectPtsXYZ.size()); ++i)
    {
        StdOut() << mSL_data.mSL_importer.mVectPtsXYZ.at(i);
        if (mSL_data.mSL_importer.HasIntensity())
            StdOut() << " " << mSL_data.mSL_importer.mVectPtsIntens.at(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    // check theta-phi :
    StdOut() << "Spherical sample:\n";
    for (size_t i=0; (i<10)&&(i<mSL_data.mSL_importer.mVectPtsTPD.size()); ++i)
    {
        StdOut() << mSL_data.mSL_importer.mVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "..." << std::endl;

    estimatePhiStep();
    computeLineCol();

    exportThetas("thetas_before.txt", 20, false);

    tREAL8 aVertCorrection = doVerticalize();
    StdOut() << "VerticalCorrection: " << aVertCorrection << "\n";

    StdOut() << "Sample after verticalization:\n";
    for (size_t i=0; (i<10)&&(i<mSL_data.mSL_importer.mVectPtsXYZ.size()); ++i)
    {
        StdOut() << mSL_data.mSL_importer.mVectPtsXYZ.at(i);
        if (mSL_data.mSL_importer.HasIntensity())
            StdOut() << " " << mSL_data.mSL_importer.mVectPtsIntens.at(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    StdOut() << "Spherical sample after verticalization:\n";
    for (size_t i=0; (i<10)&&(i<mSL_data.mSL_importer.mVectPtsTPD.size()); ++i)
    {
        StdOut() << mSL_data.mSL_importer.mVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    // export clouds for debug
    #include <fstream>
    std::fstream file1;
    file1.open("cloud.xyz", std::ios_base::out);
    std::fstream file2;
    file2.open("cloud_norm.xyz", std::ios_base::out);
    int aNbThetas = 10;
    int aTargetCol = 0; // the next we search for
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        if (mSL_data.mSL_importer.mVectPtsCol[i]==aTargetCol)
        {
            int r = 127 + 127 * sin(i/1000. + 0*M_PI/3);
            int g = 127 + 127 * sin(i/1000. + 1*M_PI/3);
            int b = 127 + 127 * sin(i/1000. + 2*M_PI/3);

            auto &aPt = mSL_data.mSL_importer.mVectPtsXYZ[i];
            auto norm = Norm2(aPt);
            file1 << aPt.x() << " " << aPt.y() << " " << aPt.z() << " " << r << " " << g << " " << b << "\n"; //i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
            file2 << aPt.x()/norm << " " << aPt.y()/norm << " " << aPt.z()/norm << " " << r << " " << g << " " << b << "\n"; //<< i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
        }
        if (mSL_data.mSL_importer.mVectPtsCol[i] > aTargetCol)
            aTargetCol = mSL_data.mSL_importer.mVectPtsCol[i] + mSL_data.mMaxCol / aNbThetas;
    }
    file2.close();
    file1.close();

    exportThetas("thetas_after.txt", 20, true);


    // export line/col stats
    file1.open("stats.txt", std::ios_base::out);
    long prev_col = -1;
    long nb_pts_in_col = 0;
    for (size_t i=0; i<mSL_data.mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        if (mSL_data.mSL_importer.mVectPtsCol[i]!=prev_col)
        {
            if (prev_col>=0)
                file1 << prev_col << " " << nb_pts_in_col <<"\n";
            prev_col = mSL_data.mSL_importer.mVectPtsCol[i];
            nb_pts_in_col = 0;
        }
        ++nb_pts_in_col;
    }
    file1.close();


    testLineColError();

    mSL_data.fillRasters(mPhProj.DPStaticLidar().FullDirOut(), true);

    mSL_data.FilterIntensity(0.01,0.99);
    mSL_data.FilterDistance(1., 10);
    mSL_data.FilterIncidence(M_PI/2-0.05);
    mSL_data.MaskBuffer(mSL_data.mPhiStep*10, mPhProj.DPStaticLidar().FullDirOut());
    mSL_data.SelectPatchCenters2(200);

    SaveInFile(mSL_data, mPhProj.DPStaticLidar().FullDirOut() + "Scan-" + mSL_data.mStationName + "-" + mSL_data.mScanName + ".xml");

    mSL_data.ToPly("Out_filtered.ply", true);
    return EXIT_SUCCESS;

}

std::vector<std::string>  cAppli_ImportStaticScan::Samples() const
{
    return
        {

        };
}


tMMVII_UnikPApli Alloc_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_ImportStaticScan(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportStaticScan
    (
        "ImportStaticScan",
        Alloc_ImportStaticScan,
        "Import static scan cloud point into instrument raster geometry",
        {eApF::Cloud},
        {eApDT::Ply},
        {eApDT::MMVIICloud},
        __FILE__
        );

}; // MMVII

