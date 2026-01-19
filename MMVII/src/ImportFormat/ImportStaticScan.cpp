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
    tREAL8 doVerticalize(); //< returns correction angle applied
    void testLineColError();
    void fixLineColRasterDirections(); //< flip ligne or col to be coherent with raster geometry
    void computeAngStartStep();
    void exportThetas(const std::string & aFileName, int aNbThetas, bool aCompareToCol);
    void poseFromXYZ();
private :
    cPhotogrammetricProject  mPhProj;

    // Mandatory Arg
    std::string              mNameFile;
    std::string              mStationName;
    std::string              mScanName;

    // Optional Arg
    std::string              mStrInput2TSL;
    bool                     mForceStructured;
    bool                     mDoVerticalize;
    cPt2dr                   mIntensityMinMax;
    cPt2dr                   mDistanceMinMax;
    tREAL8                   mIncidenceMin;
    tREAL8                   mMaskBufferSteps;
    int                      mNbPatches;
    std::string              mPoseXYZFilename;

    // data
    tPoseR                   mForcedPose;
    tREAL8 mPhiStepApprox;
    tREAL8 mThetaStepApprox;
    cStaticLidarImporter mSL_importer;
};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mPhProj         (*this),
    mStrInput2TSL     ("ijk"),
    mForceStructured(false), // skip all checks, suppose all the points are present and ordered by col
    mDoVerticalize  (false),
    mIntensityMinMax({0.01,0.99}),
    mDistanceMinMax ({0.,100.}),
    mIncidenceMin   (0.05),
    mMaskBufferSteps(2.),
    mNbPatches      (1000),
    mForcedPose     (tPoseR::Identity()),
    mPhiStepApprox  (NAN)
{
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileCloud})
           <<  mPhProj.DPStaticLidar().ArgDirOutMand()
           <<  Arg2007(mStationName ,"Station name",{eTA2007::Topo}) // TODO: change type to future station
           <<  Arg2007(mScanName ,"Scan name",{eTA2007::Topo}) // TODO: change type to future scan
        ;
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mStrInput2TSL,"Transfo","Transfo to have primariy rotation axis as Z and X as theta origin",{{eTA2007::HDV}})
           << AOpt2007(mForceStructured,"Structured","Suppose the scan is structured, skip all checks",{{eTA2007::HDV}})
           << AOpt2007(mDoVerticalize,"Vert","Try to verticalize scan columns",{{eTA2007::HDV}})
           << AOpt2007(mIntensityMinMax,"FilterIntensity","Filter on min and max intensity",{{eTA2007::HDV}})
           << AOpt2007(mDistanceMinMax,"FilterDistance","Filter on min and max distance",{{eTA2007::HDV}})
           << AOpt2007(mIncidenceMin,"FilterIncidence","Filter on min incidence (rad)",{{eTA2007::HDV}})
           << AOpt2007(mMaskBufferSteps,"MaskBuffer","Final mask buffer in hz scan steps",{{eTA2007::HDV}})
           << AOpt2007(mNbPatches,"NbPatches","Approx nb patches to make",{{eTA2007::HDV}})
           << AOpt2007(mPoseXYZFilename,"PoseXYZ","Set initial pose from a Comp3D .xyz file",{{eTA2007::HDV, eTA2007::FileAny}})
        ;
}


void cAppli_ImportStaticScan::estimatePhiStep()
{
    StdOut() << "estimatePhiStep\n";
    // find phi step
    // as median of phi differences for consecutive points, with no missing points between them (median disregards column changes)
    tREAL8 previousPhi = NAN;
    mPhiStepApprox = INFINITY; // signed value that is min in abs. For successive points on one column
    tREAL8 angularPrecisionInSteps = 0.01; // we suppose that theta changes slower than phi... prevents scanline change and pole error
    bool aPreviousWasMissing = true; // do not check for steps with 2 points if there are missing points between them

    const int aNbSamples = 1000;
    const int aNbFollowingSamples = 10;
    std::vector<tREAL8> allPhiDiff;
    allPhiDiff.reserve(aNbSamples);

    for (size_t i=0; i< mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_importer.DistMinToExist())
        {
            aPreviousWasMissing = true;
            continue;
        }
        if (!aPreviousWasMissing)
        {
            allPhiDiff.push_back(aPtAng.y()-previousPhi);
            if (allPhiDiff.size()==aNbSamples)
                break;
            if (allPhiDiff.size()%aNbFollowingSamples==aNbFollowingSamples-1) // every now and then
            {
                // jump to an other part of the scan
                i += 0.6*mSL_importer.mVectPtsTPD.size()/aNbSamples/aNbFollowingSamples - allPhiDiff.size();
                aPreviousWasMissing = true; // after the jump we missed points
                continue;
            }
        }
        previousPhi = aPtAng.y();
        aPreviousWasMissing = false;
    }
    mPhiStepApprox = NonConstMediane(allPhiDiff);
    StdOut() << "phiStep " << mPhiStepApprox << "\n";

    // estimate mThetaStepApprox
    // search for nb points in one col
    tREAL8 aMinAngToZenith = 0.1;
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size()-1; ++i)
    {
        auto aPtAng1 = mSL_importer.mVectPtsTPD[i];
        if (aPtAng1.z()<mSL_importer.DistMinToExist()) continue;
        if (fabs(fabs(aPtAng1.y())-M_PI/2)<aMinAngToZenith) continue;
        for (size_t j=i+1; j<mSL_importer.mVectPtsTPD.size(); ++j)
        {
            auto aPtAng2 = mSL_importer.mVectPtsTPD[j];
            if (aPtAng2.z()<mSL_importer.DistMinToExist()) continue;
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
    if (mSL_importer.HasRowCol())
    {
        mSL_importer.checkLineCol();
        return; // nothing to do
    }
    tREAL8 aColChangeDetectorInPhistep = 8;

    // compute line and col for each point
    mSL_importer.mVectPtsLine.resize(mSL_importer.mVectPtsXYZ.size());
    mSL_importer.mVectPtsCol.resize(mSL_importer.mVectPtsXYZ.size());
    mSL_importer.mNbCol = 0;
    tREAL8 previousPhi = NAN;
    //tREAL8 previousTheta = NAN;
    mSL_importer.mNbLine = 0;
    int aCurrLine = -1;
    int aCurrCol = 0;
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_importer.DistMinToExist())
        {
            mSL_importer.mVectPtsLine[i] = 0;
            mSL_importer.mVectPtsCol[i] = 0;
            aCurrLine++;
            continue;
        }
        if (mSL_importer.NoMiss())
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-mSL_importer.mPhiStart)/fabs(mPhiStepApprox);

        if (-(aPtAng.y()-previousPhi)/mPhiStepApprox > aColChangeDetectorInPhistep)
        {
            aCurrCol++;
            aCurrLine=0;
        }

        if (aCurrLine>=mSL_importer.mNbLine)
            mSL_importer.mNbLine = aCurrLine+1;

        mSL_importer.mVectPtsLine[i] = aCurrLine;
        mSL_importer.mVectPtsCol[i] = aCurrCol;
        //previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    mSL_importer.mNbCol = aCurrCol + 1;
    StdOut() << "Max col found: "<<mSL_importer.mNbCol<<"\n";
    StdOut() << "Max line found: "<<mSL_importer.mNbLine<<"\n";

    StdOut() << "Image size: "<<cPt2di(mSL_importer.mNbCol, mSL_importer.mNbLine)<<"\n";

    MMVII_INTERNAL_ASSERT_tiny((mSL_importer.mNbCol>0) && (mSL_importer.mNbLine>0),
                               "Image size found incorrect")

    mSL_importer.mHasRowCol = true;

    if (mSL_importer.IsStructured())
    {
        int aIndexMidFirstCol = mSL_importer.mNbLine/2;
        int aIndexMidSecondCol = mSL_importer.mNbLine*3./2;
        mSL_importer.mThetaStart = mSL_importer.mVectPtsTPD.at(aIndexMidFirstCol).x();
        mSL_importer.mThetaStep = mSL_importer.mVectPtsTPD.at(aIndexMidSecondCol).x() - mSL_importer.mVectPtsTPD.at(aIndexMidFirstCol).x();
    }

    int mThetaDir = -1; // TODO: compute!
    // precise estimation of mThetaStep and mPhiStep;
    aCurrCol = 0;
    tREAL8 aAvgTheta = 0.;
    long aNbTheta = 0;
    tREAL8 aLoopCorrection = 0.;
    std::vector<tREAL8> allThetaAvg(mSL_importer.mNbCol+1);
    int aLineLimitDown = mSL_importer.mNbLine * 0.2;
    int aLineLimitUp = mSL_importer.mNbLine * 0.8;
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_importer.DistMinToExist())
            continue;
        if ((mSL_importer.mVectPtsLine[i]>aLineLimitUp) || (mSL_importer.mVectPtsLine[i]<aLineLimitDown))
            continue; // avoid points where theta is not well defined
        if (aCurrCol!=mSL_importer.mVectPtsCol[i])
        {
            aAvgTheta = aAvgTheta/aNbTheta + aLoopCorrection;
            if ((aCurrCol>0) && (fabs(aAvgTheta-allThetaAvg[aCurrCol-1])>M_PI))
            {
                aAvgTheta -= aLoopCorrection;
                aLoopCorrection += 2*M_PI * mThetaDir;
                aAvgTheta += aLoopCorrection;
            }
            allThetaAvg[aCurrCol] = aAvgTheta;
            aCurrCol = mSL_importer.mVectPtsCol[i];
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

    std::vector<tREAL8> allThetaDiff(mSL_importer.mNbCol);
    for (unsigned int i=0;i<allThetaDiff.size();++i)
        allThetaDiff[i] = allThetaAvg[i+1]-allThetaAvg[i];
    tREAL8 aThetaStep = NonConstMediane(allThetaDiff);
    StdOut() << "ThetaStep: " << aThetaStep << "\n";
    mSL_importer.checkLineCol();
}

void cAppli_ImportStaticScan::computeAngStartStep()
{
    tREAL8 aMinAngToZenith = 0.1;
    if (mSL_importer.HasRowCol())
    {
        // search for 2 points with diff line/col to estimate steps
        long a1stPti = -1;
        for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
        {
            auto & aPtAng = mSL_importer.mVectPtsTPD[i];
            if ((aPtAng.z()>mSL_importer.DistMinToExist()) && (fabs(fabs(aPtAng.y())-M_PI/2)>aMinAngToZenith))
            {
                if (a1stPti<0)
                    a1stPti = i;
                else {
                    if ((mSL_importer.mVectPtsCol[i] != mSL_importer.mVectPtsCol[a1stPti]) && (mSL_importer.mVectPtsLine[i] != mSL_importer.mVectPtsLine[a1stPti]))
                    {
                        auto & a1stPtAng = mSL_importer.mVectPtsTPD[a1stPti];
                        mSL_importer.mPhiStep = (aPtAng.y()-a1stPtAng.y())/(mSL_importer.mVectPtsLine[i]-mSL_importer.mVectPtsLine[a1stPti]);
                        mSL_importer.mThetaStep = (aPtAng.x()-a1stPtAng.x())/(mSL_importer.mVectPtsCol[i]-mSL_importer.mVectPtsCol[a1stPti]);
                        mSL_importer.mPhiStart = a1stPtAng.y() - mSL_importer.mPhiStep * mSL_importer.mVectPtsLine[a1stPti];
                        mSL_importer.mThetaStart = a1stPtAng.x() - mSL_importer.mThetaStep * mSL_importer.mVectPtsCol[a1stPti];
                        break;
                    }
                }
            }
        }
        // make a better approx using a second point near the end
        for (long i=(long)mSL_importer.mVectPtsTPD.size()-1; i>a1stPti; --i)
        {
            if ((mSL_importer.mVectPtsTPD[i].z()>mSL_importer.DistMinToExist()) && (fabs(fabs(mSL_importer.mVectPtsTPD[i].y())-M_PI/2)>aMinAngToZenith))
            {
                if ((mSL_importer.mVectPtsCol[i] != mSL_importer.mVectPtsCol[a1stPti]) && (mSL_importer.mVectPtsLine[i] != mSL_importer.mVectPtsLine[a1stPti]))
                {
                    auto & a1stPtAng = mSL_importer.mVectPtsTPD[a1stPti];
                    auto a2ndPtAng = mSL_importer.mVectPtsTPD[i]; // copy to unroll
                    tREAL8 aTheta = mSL_importer.mVectPtsCol[i]*mSL_importer.mThetaStep + mSL_importer.mThetaStart;
                    a2ndPtAng.x() = toMinusPiPlusPi(a2ndPtAng.x(), aTheta);
                    mSL_importer.mPhiStep = (a2ndPtAng.y()-a1stPtAng.y())/(mSL_importer.mVectPtsLine[i]-mSL_importer.mVectPtsLine[a1stPti]);
                    mSL_importer.mThetaStep = (a2ndPtAng.x()-a1stPtAng.x())/(mSL_importer.mVectPtsCol[i]-mSL_importer.mVectPtsCol[a1stPti]);
                    mSL_importer.mPhiStart = a1stPtAng.y() - mSL_importer.mPhiStep * mSL_importer.mVectPtsLine[a1stPti];
                    mSL_importer.mThetaStart = a1stPtAng.x() - mSL_importer.mThetaStep * mSL_importer.mVectPtsCol[a1stPti];
                    StdOut() << "i1 i2: " << a1stPti << " " << i << ", "
                             << mSL_importer.mVectPtsCol[a1stPti] << " " << mSL_importer.mVectPtsLine[a1stPti] << " "
                             << mSL_importer.mVectPtsCol[i] << " " << mSL_importer.mVectPtsLine[i] << "\n";
                    StdOut() << a1stPtAng.x() << " " << a1stPtAng.y() << " "
                             << a2ndPtAng.x() << " " << a2ndPtAng.y() << "\n";
                    break;
                }
            }
        }
    } else {
        if (mSL_importer.IsStructured())
        {
            mSL_importer.mPhiStart = mSL_importer.mVectPtsTPD.at(0).y();
            mSL_importer.mThetaStart = NAN; // do it when line/col computed
            mSL_importer.mPhiStep = mSL_importer.mVectPtsTPD.at(1).y() - mSL_importer.mVectPtsTPD.at(0).y();
            mSL_importer.mThetaStep = NAN;
        } else {
            MMVII_INTERNAL_ASSERT_tiny(false, "No computeAngStartEnd() for sparse cloud for now")
        }
    }

    StdOut() << "PhiStart: " << mSL_importer.mPhiStart << ", "
             << "PhiStep: " << mSL_importer.mPhiStep << ", "
             << "ThetaStart: " << mSL_importer.mThetaStart << ", "
             << "ThetaStep: " << mSL_importer.mThetaStep << "\n";
    StdOut() << "PhiEnd: " << mSL_importer.mPhiStart+(mSL_importer.mNbLine-1)*mSL_importer.mPhiStep << ", "
             << "ThetaEnd: " << mSL_importer.mThetaStep+(mSL_importer.mNbCol-1)*mSL_importer.mThetaStep << "\n";
}

void cAppli_ImportStaticScan::testLineColError()
{
    tREAL8 aMaxThetaError = -1.;
    tREAL8 aMaxPhiError = -1.;
    tREAL8 aMinAngToZenith = 0.1;
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        auto & aPtAng = mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_importer.DistMinToExist())
            continue;
        if (fabs(fabs(aPtAng.y())-M_PI/2)<aMinAngToZenith)
            continue; // no stats on points too close to undefined theta
        tREAL8 aTheta = mSL_importer.mVectPtsCol[i]*mSL_importer.mThetaStep + mSL_importer.mThetaStart;
        tREAL8 aPhi = mSL_importer.mVectPtsLine[i]*mSL_importer.mPhiStep + mSL_importer.mPhiStart;
        aTheta = toMinusPiPlusPi(aTheta, aPtAng.x());
        if (fabs(aTheta-aPtAng.x())>aMaxThetaError)
        {
            aMaxThetaError = fabs(aTheta-aPtAng.x());
            //StdOut() << i << " " << aPtAng <<" => " << mSL_importer.mVectPtsCol[i] << " " << mSL_importer.mVectPtsLine[i] << " " << aTheta << " " << aPhi << " => " << aMaxThetaError << " " << aMaxPhiError << "\n";
        }
        if (fabs(aPhi-aPtAng.y())>aMaxPhiError)
        {
            aMaxPhiError = fabs(aPhi-aPtAng.y());
            //StdOut() << i << " " << aPtAng <<" => " << mSL_importer.mVectPtsCol[i] << " " << mSL_importer.mVectPtsLine[i] << " " << aTheta << " " << aPhi << " => " << aMaxThetaError << " " << aMaxPhiError << "\n";
        }
    }
    StdOut() << "Max ang errors: " << aMaxThetaError << " " << aMaxPhiError <<"\n";
}


void cAppli_ImportStaticScan::fixLineColRasterDirections()
{
    // raster geometry is col to the right and lines to the bottom
    // raster lines correspond to scan lines if phi step is < 0
    // raster cols correspond to scan cols if theta step is < 0

    if (mSL_importer.mPhiStep > 0)
    {
        for (auto & aLine: mSL_importer.mVectPtsLine)
            aLine = mSL_importer.NbLine() -1 - aLine;
        // invert start and end
        mSL_importer.mPhiStart = mSL_importer.mPhiStart+(mSL_importer.mNbLine-1)*mSL_importer.mPhiStep;
        mSL_importer.mPhiStep = -mSL_importer.mPhiStep;
    }

    if (mSL_importer.mThetaStep > 0)
    {
        for (auto & aCol: mSL_importer.mVectPtsCol)
            aCol = mSL_importer.NbCol() -1 - aCol;
        // invert start and end
        mSL_importer.mThetaStart = mSL_importer.mThetaStep+(mSL_importer.mNbCol-1)*mSL_importer.mThetaStep;
        mSL_importer.mThetaStep = -mSL_importer.mThetaStep;
    }

}

tREAL8 cAppli_ImportStaticScan::doVerticalize()
{
    StdOut() << "Verticalizing..." << std::endl;
    // estimate verticalization correction if scanner with compensator
    int aColChangeDetectorInPhistep = 100;
    int aNbPlanes = 20; // try to get several planes for instrument primariy axis estimation
    float aCorrectPlanePhiRange = 40*M_PI/180; // try to get points with this phi diff in a scanline
    int aColPlaneStep = mSL_importer.mNbCol / aNbPlanes;
    int aLineGoodRange = aCorrectPlanePhiRange/fabs(mPhiStepApprox);
    if (aLineGoodRange > mSL_importer.mNbLine*0.5)
        aLineGoodRange = mSL_importer.mNbLine*0.5; // for small scans, use full height

    int aTargetCol = 0; // the next we search for
    int aTargetLine = 0;
    float previousPhi = NAN;
    //tREAL8 previousTheta = NAN;
    int aCurrCol = 0;
    int aCurrLine = 0;
    std::vector<std::tuple<cPt3dr, cPt3dr, cPt3dr>> aVPtsPlanes; // list of triplets to find vertical planes
    cPt3dr * aPtBottom = nullptr;
    cPt3dr * aPtTop = nullptr;
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        // TODO: factorize xyz points list to linecol!
        auto aPtAng = mSL_importer.mVectPtsTPD[i];
        if (aPtAng.z()<mSL_importer.DistMinToExist())
        {
            aCurrLine++;
            continue;
        }
        if (mSL_importer.NoMiss())
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-mSL_importer.mPhiStart)/fabs(mPhiStepApprox);

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
                aPtBottom = &mSL_importer.mVectPtsXYZ[i];
                aTargetLine = aCurrLine + aLineGoodRange;
            }
            else if ((aCurrLine>aTargetLine)&&(!aPtTop))
            {
                aPtTop = &mSL_importer.mVectPtsXYZ[i];
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

    mSL_importer.mVertRot = cRotation3D<tREAL8>::CompleteRON(aSegVert.V12(),2);

    // update xyz and tpd coordinates
    for (size_t i=0; i<mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        mSL_importer.mVectPtsXYZ[i] = mSL_importer.mVertRot.Inverse(mSL_importer.mVectPtsXYZ[i]);
    }
    mSL_importer.convertToThetaPhiDist();
    // update line col
    computeLineCol();

    return mSL_importer.mVertRot.Angle();
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
    for (size_t i=0; i<mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        if ( mSL_importer.mVectPtsTPD[i].z()<mSL_importer.DistMinToExist())
            continue;
        if (mSL_importer.mVectPtsCol[i]==aTargetCol)
        {
            auto &aPtTPD = mSL_importer.mVectPtsTPD[i];
            if (isFirstofCol)
            {
                if (aCompareToCol)
                {
                    aThetaCol = mSL_importer.mThetaStart + mSL_importer.mThetaStep * aTargetCol;
                    aThetaCol = toMinusPiPlusPi(aThetaCol);
                    StdOut() <<  aPtTPD.x() << " " << aThetaCol << " " << aPtTPD.x() - aThetaCol << "\n";
                }
                else
                    aThetaCol = aPtTPD.x();
                isFirstofCol = false;
            }
            tREAL8 aError = aPtTPD.x() - aThetaCol;
            aError= toMinusPiPlusPi(aError);
            //file_thetas << mSL_importer.mVectPtsLine[i] << " " <<  <<"\n";
            file_thetas << aError <<" ";
        }
        if (mSL_importer.mVectPtsCol[i] > aTargetCol)
        {
            file_thetas << "\n";
            aTargetCol = mSL_importer.mVectPtsCol[i] + mSL_importer.mNbCol / aNbThetas;
            isFirstofCol = true;
            if (aTargetCol>mSL_importer.mNbCol)
                aTargetCol=mSL_importer.mNbCol;
        }
    }
    file_thetas << "\n";
    file_thetas.close();
}

void cAppli_ImportStaticScan::poseFromXYZ()
{
    /*The rotation is given from ground to TSL frame. Has to be converted to MM camera frame
     *
     * Comp3D .XYZ file format :

CT195	-8.901	-24.577	2.187	0.005
[...]
CT197	3.580	-3.306	5.238	0.001
**!  Station    : L01
*  S =             104.7634 102.3907 97.8046
*            -0.8624946      -0.5060662       0.0000281
*  R =        0.5060662      -0.8624946       0.0000021
*             0.0000232       0.0000161       1.0000000
****  instr = R.(global-S) <=> global = R'.instr +S  ****
****  here global frame is in cartesian  ****
[...]

     */

    std::ifstream aXYZfile(mPoseXYZFilename);
    MMVII_INTERNAL_ASSERT_tiny(aXYZfile.is_open(),"Error opening "+mPoseXYZFilename);
    std::string aLine;
    while (std::getline(aXYZfile, aLine)) {
        if (aLine.find("**!") != std::string::npos) {
            break;
        }
    }
    tREAL8 x,y,z;
    std::string tmp;
    cPt3dr aT;
    cPt3dr aR1, aR2, aR3;
    {
        std::getline(aXYZfile, aLine);
        MMVII_INTERNAL_ASSERT_tiny(aLine.find("*  S =") != std::string::npos,"Error reading "+mPoseXYZFilename);
        std::istringstream iss(aLine);
        iss >> tmp >> tmp >> tmp >> x >> y >> z;
        aT = {x, y, z};
    }
    {
        std::getline(aXYZfile, aLine);
        std::istringstream iss(aLine);
        iss >> tmp >> x >> y >> z;
        aR1 = {x, y, z};
    }
    {
        std::getline(aXYZfile, aLine);
        std::istringstream iss(aLine);
        iss >> tmp >> tmp >> tmp >> x >> y >> z;
        aR2 = {x, y, z};
    }
    {
        std::getline(aXYZfile, aLine);
        std::istringstream iss(aLine);
        iss >> tmp >> x >> y >> z;
        aR3 = {x, y, z};
    }
    MMVII_INTERNAL_ASSERT_tiny(aXYZfile.good(),"Error reading "+mPoseXYZFilename);

    cRotation3D<tREAL8> aRotTSL2MM = cRotation3D<tREAL8>::RotFromCanonicalAxes("k-i-j");

    mForcedPose.Tr() = aT;
    mForcedPose.Rot() =
                        (aRotTSL2MM *
                         cRotation3D<tREAL8>({aR1.x(), aR1.y(), aR1.z()},
                                             {aR2.x(), aR2.y(), aR2.z()},
                                             {aR3.x(), aR3.y(), aR3.z()}, true)
                        ).MapInverse();
    }

int cAppli_ImportStaticScan::Exe()
{
    mPhProj.FinishInit();

    mSL_importer.read(mNameFile, false, mForceStructured, mStrInput2TSL);

    MMVII_INTERNAL_ASSERT_tiny(!mSL_importer.mVectPtsXYZ.empty(),"Error reading "+mNameFile);
    if (mSL_importer.HasIntensity())
    {
        MMVII_INTERNAL_ASSERT_tiny(mSL_importer.mVectPtsXYZ.size()==mSL_importer.mVectPtsIntens.size(),"Error reading "+mNameFile);
    }

    StdOut() << "Cartesian sample:\n";
    for (size_t i=0; (i<10)&&(i<mSL_importer.mVectPtsXYZ.size()); ++i)
    {
        StdOut() << mSL_importer.mVectPtsXYZ.at(i);
        if (mSL_importer.HasIntensity())
            StdOut() << " " << mSL_importer.mVectPtsIntens.at(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    // check theta-phi :
    StdOut() << "Spherical sample:\n";
    for (size_t i=0; (i<10)&&(i<mSL_importer.mVectPtsTPD.size()); ++i)
    {
        StdOut() << mSL_importer.mVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "..." << std::endl;

    estimatePhiStep();
    computeLineCol();
    computeAngStartStep(); // best results when having line col

    exportThetas("thetas_before.txt", 20, false);

    if (mDoVerticalize)
    {
        tREAL8 aVertCorrection = doVerticalize();
        StdOut() << "VerticalCorrection: " << aVertCorrection << "\n";

        StdOut() << "Sample after verticalization:\n";
        for (size_t i=0; (i<1000)&&(i<mSL_importer.mVectPtsXYZ.size()); ++i)
        {
            StdOut() << mSL_importer.mVectPtsXYZ.at(i);
            if (mSL_importer.HasIntensity())
                StdOut() << " " << mSL_importer.mVectPtsIntens.at(i);
            StdOut() << "\n";
        }
        StdOut() << "...\n";

        StdOut() << "Spherical sample after verticalization:\n";
        for (size_t i=0; (i<1000)&&(i<mSL_importer.mVectPtsTPD.size()); ++i)
        {
            StdOut() << mSL_importer.mVectPtsTPD[i];
            StdOut() << "\n";
        }
        StdOut() << "...\n";
    }

    // export clouds for debug
    #include <fstream>
    std::fstream file1;
    file1.open("cloud.xyz", std::ios_base::out);
    std::fstream file2;
    file2.open("cloud_norm.xyz", std::ios_base::out);
    int aNbThetas = 10;
    int aTargetCol = 0; // the next we search for
    for (size_t i=0; i<mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        if (mSL_importer.mVectPtsCol[i]==aTargetCol)
        {
            int r = 127 + 127 * sin(i/1000. + 0*M_PI/3);
            int g = 127 + 127 * sin(i/1000. + 1*M_PI/3);
            int b = 127 + 127 * sin(i/1000. + 2*M_PI/3);

            auto &aPt = mSL_importer.mVectPtsXYZ[i];
            auto norm = Norm2(aPt);
            file1 << aPt.x() << " " << aPt.y() << " " << aPt.z() << " " << r << " " << g << " " << b << "\n"; //i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
            file2 << aPt.x()/norm << " " << aPt.y()/norm << " " << aPt.z()/norm << " " << r << " " << g << " " << b << "\n"; //<< i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
        }
        if (mSL_importer.mVectPtsCol[i] > aTargetCol)
            aTargetCol = mSL_importer.mVectPtsCol[i] + mSL_importer.mNbCol / aNbThetas;
    }
    file2.close();
    file1.close();

    exportThetas("thetas_after.txt", 20, true);


    // export line/col stats
    file1.open("stats.txt", std::ios_base::out);
    long prev_col = -1;
    long nb_pts_in_col = 0;
    for (size_t i=0; i<mSL_importer.mVectPtsXYZ.size(); ++i)
    {
        if (mSL_importer.mVectPtsCol[i]!=prev_col)
        {
            if (prev_col>=0)
                file1 << prev_col << " " << nb_pts_in_col <<"\n";
            prev_col = mSL_importer.mVectPtsCol[i];
            nb_pts_in_col = 0;
        }
        ++nb_pts_in_col;
    }
    file1.close();


    testLineColError();

    fixLineColRasterDirections();

    // compute transfo from scan instrument frame to sensor frame
    mSL_importer.ComputeRotInput2Raster(mStrInput2TSL);
    mSL_importer.ComputeAgregatedAngles();

    // create sensor from imported data
    std::string aScanName = cStaticLidar::ScanPrefixName() + mStationName + "-" + mScanName;
    // find PP: image of the (Oz) axis
    cPt3dr aOzAxisInput = mSL_importer.RotInput2TSL().Inverse({1.,0.,0.});  // axis 1,0,0 in TSL frame, just to get equator vertical angle
    cPt2dr aEquatorAngles = mSL_importer.Input3DtoRasterAngle(aOzAxisInput);
    std::cout<< aEquatorAngles.y()<<" approx "
             << mSL_importer.LocalPhiToLineApprox(aEquatorAngles.y())
              <<" precise "
              << mSL_importer.LocalPhiToLinePrecise(aEquatorAngles.y()) <<"\n";

    cPt2dr aPP((mSL_importer.NbCol()-1)/2., mSL_importer.LocalPhiToLinePrecise(aEquatorAngles.y()));
    //find F: scale from angle to pixels
    tREAL8 aFy = 1./fabs(mSL_importer.mPhiStep); //TODO: add polynomial disto for different angular steps

    MMVII_INTERNAL_ASSERT_tiny(fabs((fabs(mSL_importer.mPhiStep)-fabs(mSL_importer.mThetaStep))/mSL_importer.mPhiStep)<1e-2,
                               "Error: different steps in theta and phi are not supported yet!");

    cPerspCamIntrCalib* aCalib =
        cPerspCamIntrCalib::SimpleCalib(cStaticLidar::CalibPrefixName() + aScanName, eProjPC::eEquiRect,
                                        cPt2di(mSL_importer.NbCol(), mSL_importer.NbLine()),
                                        cPt3dr(aPP.x(),aPP.y(),aFy), cPt3di(0,0,0));
    aCalib->ToFile(mPhProj.DPStaticLidar().FullDirOut() + aCalib->Name() + ".xml");

    cStaticLidar aSL_data(mNameFile, mStationName, mScanName,
                          cIsometry3D<tREAL8>({}, cRotation3D<tREAL8>::Identity()),
                          aCalib, mSL_importer.RotInput2Raster());

    if (IsInit(&mPoseXYZFilename))
    {
        StdOut() << "Read XYZ pose file: " << mPoseXYZFilename << std::endl;
        poseFromXYZ();
        aSL_data.SetPose(mForcedPose);
    } else {
        aSL_data.SetPose(mSL_importer.ReadPose());
    }

    aSL_data.fillRasters(mSL_importer, mPhProj.DPStaticLidar().FullDirOut(), true);

    aSL_data.FilterIntensity(mSL_importer, mIntensityMinMax[0], mIntensityMinMax[1]);
    aSL_data.FilterDistance(mDistanceMinMax[0], mDistanceMinMax[1]);
    aSL_data.FilterIncidence(mSL_importer, M_PI/2-mIncidenceMin);
    aSL_data.MaskBuffer(mSL_importer, mSL_importer.mPhiStep*mMaskBufferSteps, mPhProj.DPStaticLidar().FullDirOut());
    aSL_data.SelectPatchCenters2(mSL_importer, mNbPatches);

    SaveInFile(aSL_data, mPhProj.DPStaticLidar().FullDirOut() + aScanName + ".xml");

    aSL_data.ToPly("Out_filtered.ply", true);
    delete aCalib;
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

