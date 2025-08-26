#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ReadFileStruct.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"
#include "../Mesh/happly.h"
#include <functional>

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


    void readPlyPoints(std::string aPlyFileName);
    void convertToThetaPhiDist();
    void getAnglesMinMax();
    void estimatePhiStep();
    void computeLineCol();
    tREAL8 doVerticalize(); // returns correction angle applied
    template <typename TYPE> void fillRaster(const std::string& aFileName, std::function<TYPE (int)> func );
private :

    // Mandatory Arg
    std::string              mNameFile;
    std::string              mStationName;
    std::string              mScanName;

    // Optional Arg
    std::string              mTransfoIJK;
    bool                     mNoMiss; // are every point present in cloud, even when no response?


    // data
    tREAL8 mDistMinToExist; // boundary to check if no response points are present
    std::vector<cPt3dr> mVectPtsXYZ;
    std::vector<tREAL8> mVectPtsIntens;
    std::vector<cPt3dr> mVectPtsTPD;
    tREAL8 mThetaMin, mThetaMax, mPhiMin, mPhiMax;
    tREAL8 mPhiStep;
    int mMaxCol, mMaxLine;

    // line and col for each point
    std::vector<int> mVectPtsLine;
    std::vector<int> mVectPtsCol;
};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mTransfoIJK     ("ijk"),
    mNoMiss         (false),
    mDistMinToExist (1e-6),
    mThetaMin       (NAN),
    mThetaMax       (NAN),
    mPhiMin         (NAN),
    mPhiMax         (NAN),
    mPhiStep        (NAN)
{
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
           <<  Arg2007(mStationName ,"Station name",{eTA2007::Topo}) // TODO: change type to future station
           <<  Arg2007(mScanName ,"Scan name",{eTA2007::Topo}) // TODO: change type to future scan
        ;
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mTransfoIJK,"Transfo","Transfo to have primariy rotation axis as Z and X as theta origin",{{eTA2007::HDV}})
        ;
}

cPt3dr cart2spher(const cPt3dr & aPtCart)
{
    tREAL8 dist = Norm2(aPtCart);
    tREAL8 theta =  atan2(aPtCart.y(),aPtCart.x());
    tREAL8 distxy = sqrt(aPtCart.BigX2()+aPtCart.BigY2());
    tREAL8 phi =  atan2(aPtCart.z(),distxy);
    return {theta, phi, dist};
}

using  namespace happly;

void cAppli_ImportStaticScan::readPlyPoints(std::string aPlyFileName)
{
    mVectPtsXYZ.clear();
    mVectPtsIntens.clear();
    try
    {
        PLYData  aPlyF(aPlyFileName,false);
        auto aElementsNames = aPlyF.getElementNames();
        // Read points
        {
            std::vector<std::array<double, 3>> aVecPts = aPlyF.getVertexPositions();
            mVectPtsXYZ.resize(aVecPts.size());
            for (size_t i=0; i<aVecPts.size(); ++i)
            {
                mVectPtsXYZ.at(i) = cPt3dr(aVecPts[i][0],aVecPts[i][1],aVecPts[i][2]);
            }
        }

        // try to fill points attribute with any "i", "I" or "*intens*" property
        auto aVertProps = aPlyF.getElement("vertex").getPropertyNames();
        auto aPropIntensityName = std::find_if(aVertProps.begin(), aVertProps.end(), [](const std::string &s){
            return (ToLower(s)=="i") || (ToLower(s).find("intens") != std::string::npos);
        });
        if (aPropIntensityName!= aVertProps.end())
        {
            mVectPtsIntens = aPlyF.getElement("vertex").getProperty<tREAL8>(*aPropIntensityName);
        }

    }
    catch (const std::runtime_error &e)
    {
        MMVII_UserError(eTyUEr::eReadFile, std::string("Error reading PLY file \"") + aPlyFileName + "\": " + e.what());
    }
}

void cAppli_ImportStaticScan::convertToThetaPhiDist()
{
    mNoMiss = false;
    mVectPtsTPD.resize(mVectPtsXYZ.size()); // all points in theta-phi-dist
    size_t aNbPtsNul = 0;
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        mVectPtsTPD[i] = cart2spher(mVectPtsXYZ[i]);
        if (mVectPtsTPD[i].z()<mDistMinToExist)
            aNbPtsNul++;
    }
    StdOut() << aNbPtsNul << " null points\n";
    mNoMiss = aNbPtsNul>0;

    // get min max theta phi
    cWhichMinMax<int, tREAL8> aMinMaxTheta;
    cWhichMinMax<int, tREAL8> aMinMaxPhi;
    for (const auto & aPtAng: mVectPtsTPD)
    {
        aMinMaxTheta.Add(0,aPtAng.x());
        aMinMaxPhi.Add(0,aPtAng.y());
    }
    mThetaMin = aMinMaxTheta.Min().ValExtre();
    mThetaMax = aMinMaxTheta.Max().ValExtre();
    mPhiMin = aMinMaxPhi.Min().ValExtre();
    mPhiMax = aMinMaxPhi.Max().ValExtre();
    StdOut() << "Box:  theta " << mThetaMin << ", " << mThetaMax << "   phi "
             << mPhiMin << ", " << mPhiMax << "\n";
}

void cAppli_ImportStaticScan::estimatePhiStep()
{
    // find phi min diff
    // successive phi diff is useful, but only if we are in the same scanline, and we are not too close to the pole with vertical error
    tREAL8 previousTheta = NAN;
    tREAL8 previousPhi = NAN;
    mPhiStep = INFINITY; // signed value that is min in abs. For successive points on one column
    tREAL8 angularPrecisionInSteps = 0.01; // we suppose that theta changes slower than phi... prevents scanline change and pole error
    for (const auto & aPtAng: mVectPtsTPD)
    {
        if (aPtAng.z()<mDistMinToExist) continue;
        auto aDiffPhi = aPtAng.y()-previousPhi;
        auto aDiffTheta = aPtAng.x()-previousTheta;
        if (fabs(aDiffTheta)<fabs(mPhiStep)*angularPrecisionInSteps) // we are on the same scanline
        {
            if (fabs(aDiffPhi)<fabs(mPhiStep))
            {
                //std::cout<<"with prev "<<previousTheta<< " "<< previousPhi<< "  curr "<<aPtAng<<":\n";
                //std::cout<<"up: "<<minDiffPhi <<" " <<aDiffPhi<<"\n";
                mPhiStep = aDiffPhi;
            }
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "phiStep " << mPhiStep << ",  " << (mPhiMax-mPhiMin)/fabs(mPhiStep) << " steps\n";

}

void cAppli_ImportStaticScan::computeLineCol()
{
    tREAL8 aColChangeDetectorInPhistep = 100;

    // compute line and col for each point
    mVectPtsLine.resize(mVectPtsXYZ.size());
    mVectPtsCol.resize(mVectPtsXYZ.size());
    mMaxCol = 0;
    tREAL8 previousPhi = NAN;
    //tREAL8 previousTheta = NAN;
    mMaxLine = 0;
    int aCurrLine = 0;
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        auto aPtAng = mVectPtsTPD[i];
        if (aPtAng.z()<mDistMinToExist)
        {
            mVectPtsLine[i] = 0;
            mVectPtsCol[i] = 0;
            aCurrLine++;
            continue;
        }
        if (mNoMiss)
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-mThetaMin)/fabs(mPhiStep);

        if (aCurrLine>mMaxLine) mMaxLine = aCurrLine;

        if (-(aPtAng.y()-previousPhi)/mPhiStep > aColChangeDetectorInPhistep)
        {
            mMaxCol++;
            aCurrLine=0;
        }
        mVectPtsLine[i] = aCurrLine;
        mVectPtsCol[i] = mMaxCol;
        //previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "Max col found: "<<mMaxCol<<"\n";
    StdOut() << "Max line found: "<<mMaxLine<<"\n";

    StdOut() << "Image size: "<<cPt2di(mMaxCol+1, mMaxLine+1)<<"\n";
}

template <typename TYPE> void cAppli_ImportStaticScan::fillRaster(const std::string& aFileName, std::function<TYPE (int)> func )
{
    MMVII_INTERNAL_ASSERT_tiny(mVectPtsCol.size()==mVectPtsXYZ.size(), "Error: Compute line/col numbers before fill raster")
    cIm2D<TYPE> aRaster(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterData = aRaster.DIm();
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        cPt2di aPcl = {mVectPtsCol[i], mMaxLine-mVectPtsLine[i]};
        aRasterData.SetV(aPcl, func(i));
    }
    aRasterData.ToFile(aFileName);
}

tREAL8 cAppli_ImportStaticScan::doVerticalize()
{
    // estimate verticalization correction if scanner with compensator
    int aColChangeDetectorInPhistep = 100;
    int aNbPlanes = 10; // try to get several planes for instrument primariy axis estimation
    float aCorrectPlanePhiRange = 80*M_PI/180; // try to get points with this phi diff in a scanline
    int aColPlaneStep = mMaxCol / aNbPlanes;
    int aLineGoodRange = aCorrectPlanePhiRange/fabs(mPhiStep);
    if (aLineGoodRange > mMaxLine - 2)
        aLineGoodRange = mMaxLine - 2; // for small scans, use full height

    int aTargetCol = 0; // the next we search for
    int aTargetLine = 0;
    float previousPhi = NAN;
    //tREAL8 previousTheta = NAN;
    int aCurrCol = 0;
    int aCurrLine = 0;
    std::vector<std::tuple<cPt3dr, cPt3dr, cPt3dr>> aVPtsPlanes; // list of triplets to find vertical planes
    cPt3dr * aPtBottom = nullptr;
    cPt3dr * aPtTop = nullptr;
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        // TODO: factorize xyz points list to linecol!
        auto aPtAng = mVectPtsTPD[i];
        if (aPtAng.z()<mDistMinToExist)
        {
            aCurrLine++;
            continue;
        }
        if (mNoMiss)
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-mThetaMin)/fabs(mPhiStep);

        if (-(aPtAng.y()-previousPhi)/mPhiStep > aColChangeDetectorInPhistep)
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
                aPtBottom = &mVectPtsXYZ[i];
                aTargetLine = aCurrLine + aLineGoodRange;
            }
            else if ((aCurrLine>aTargetLine)&&(!aPtTop))
            {
                aPtTop = &mVectPtsXYZ[i];
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
    tSeg3dr aSegVert = cPlane3D::InterPlane(aVPlanes, aNbPlanes/2);
    StdOut() << "Vert: " << aSegVert.V12() << "\n";

    cRotation3D<tREAL8> aVertRot = cRotation3D<tREAL8>::CompleteRON(aSegVert.V12(),2);

    // update xyz and tpd coordinates
    for (size_t i=0; i<mVectPtsXYZ.size(); ++i)
    {
        mVectPtsXYZ[i] = aVertRot.Inverse(mVectPtsXYZ[i]);
    }
    convertToThetaPhiDist();
    // update line col
    computeLineCol();

    return aVertRot.Angle();
}

int cAppli_ImportStaticScan::Exe()
{
    readPlyPoints(mNameFile);

    StdOut() << "Got " << mVectPtsXYZ.size() << " points.\n";
    MMVII_INTERNAL_ASSERT_tiny(!mVectPtsXYZ.empty(),"Error reading "+mNameFile);
    if (!mVectPtsIntens.empty())
    {
        StdOut() << "Intensity found.\n";
        MMVII_INTERNAL_ASSERT_tiny(mVectPtsXYZ.size()==mVectPtsIntens.size(),"Error reading "+mNameFile);
    }

    StdOut() << "Sample:\n";
    for (size_t i=0; (i<10)&&(i<mVectPtsXYZ.size()); ++i)
    {
        StdOut() << mVectPtsXYZ.at(i);
        if (!mVectPtsIntens.empty())
            StdOut() << " " << mVectPtsIntens.at(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    cRotation3D<tREAL8>  aRotFrame = cRotation3D<tREAL8>::RotFromCanonicalAxes(mTransfoIJK);
    // apply rotframe to original points
    for (auto & aPtXYZ : mVectPtsXYZ)
    {
        aPtXYZ = aRotFrame.Value(aPtXYZ);
    }

    convertToThetaPhiDist();

    // check theta-phi :
    StdOut() << "Spherical sample:\n";
    for (size_t i=0; (i<10)&&(i<mVectPtsTPD.size()); ++i)
    {
        StdOut() << mVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    estimatePhiStep();

    computeLineCol();

    //fill rasters
    fillRaster<tU_INT1>("totoMask.png", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.z()<mDistMinToExist;} );
    fillRaster<tU_INT1>("totoIntens.png", [this](int i){return mVectPtsIntens[i]*255;} );
    fillRaster<tU_INT4>("totoIndex.png", [](int i){return i;} );
    fillRaster<float>("totoDist.tif", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.z();} );
    fillRaster<float>("totoTheta.tif", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.x();} );
    fillRaster<float>("totoPhi.tif", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.y();} );

    tREAL8 aVertCorrection = doVerticalize();
    StdOut() << "VerticalCorrection: " << aVertCorrection << "\n";

    // export clouds for debug
    #include <fstream>
    std::fstream file1;
    file1.open("cloud.xyz", std::ios_base::out);
    std::fstream file2;
    file2.open("cloud_norm.xyz", std::ios_base::out);
    int aNbThetas = 10;
    int aTargetCol = 0; // the next we search for
    for (size_t i=0; i<mVectPtsXYZ.size(); ++i)
    {
        if (mVectPtsCol[i]==aTargetCol)
        {
            int r = 127 + 127 * sin(i/1000. + 0*M_PI/3);
            int g = 127 + 127 * sin(i/1000. + 1*M_PI/3);
            int b = 127 + 127 * sin(i/1000. + 2*M_PI/3);

            auto aPt = mVectPtsXYZ[i];
            auto norm = Norm2(aPt);
            file1 << aPt.x() << " " << aPt.y() << " " << aPt.z() << " " << r << " " << g << " " << b << "\n"; //i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
            file2 << aPt.x()/norm << " " << aPt.y()/norm << " " << aPt.z()/norm << " " << r << " " << g << " " << b << "\n"; //<< i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
        }
        if (mVectPtsCol[i] > aTargetCol)
            aTargetCol = mVectPtsCol[i] + mMaxCol / aNbThetas;
    }
    file2.close();
    file1.close();

    fillRaster<tU_INT1>("titiMask.png", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.z()<mDistMinToExist;} );
    fillRaster<tU_INT1>("titiIntens.png", [this](int i){return mVectPtsIntens[i]*255;} );
    fillRaster<tU_INT4>("titiIndex.png", [](int i){return i;} );
    fillRaster<float>("titiDist.tif", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.z();} );
    fillRaster<float>("titiTheta.tif", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.x();} );
    fillRaster<float>("titiPhi.tif", [this](int i){auto aPtAng = mVectPtsTPD[i];return aPtAng.y();} );


    return EXIT_SUCCESS;


    /*
    // make statistics on theta phi, using raster geometry
    StdOut() << "Compute steps from inital raster geometry\n";
    //std::vector<tREAL8> aVDiffTheta;
    std::fstream file_theta;
    file_theta.open("thetas.txt", std::ios_base::out);
    std::fstream file_theta_abs;
    file_theta_abs.open("thetas_abs.txt", std::ios_base::out);
    tREAL8 aThetaStep2 = 0.;
    int nbThetaStep = 0;
    for (int c=1+mMaxCol/3; c<2*mMaxCol/3+1; ++c)
    {
        tU_INT4 i = aRasterIndexData.GetV(cPt2di(c,mMaxLine/2));
        tU_INT4 ic = aRasterIndexData.GetV(cPt2di(c-1,mMaxLine/2));
        file_theta_abs<<i<<" "<<mVectPtsTPD[i].x()<< " "<<ic<<" "<<mVectPtsTPD[ic].x()<<"\n";
        if ((i!=0)&&(ic!=0))
        {
            if ((mVectPtsTPD[i].z()<mDistMinToExist)
                ||(mVectPtsTPD[ic].z()<mDistMinToExist))
            {
                std::cout<<"EEEERRRROROORORORO\n";
                std::cout<<i<<" "<<mVectPtsTPD[i]<< " "<<ic<<" "<<mVectPtsTPD[ic]<<"\n";
            }
            auto aDiffTheta = mVectPtsTPD[i].x()-mVectPtsTPD[ic].x();
            if (aDiffTheta>M_PI)
                aDiffTheta -= 2*M_PI;
            if (aDiffTheta<-M_PI)
                aDiffTheta += 2*M_PI;
            file_theta<<aDiffTheta<<"\n";
            aThetaStep2+=aDiffTheta;
            nbThetaStep++;
        }
    }
    file_theta.close();
    file_theta_abs.close();
    aThetaStep2/=nbThetaStep;
    StdOut() << " New Theta Step: " << aThetaStep2<<"\n";

    std::fstream file_phi;
    file_phi.open("phis.txt", std::ios_base::out);
    std::fstream file_phi_abs;
    file_phi_abs.open("phis_abs.txt", std::ios_base::out);
    tREAL8 aPhiStep2 = 0.;
    int nbPhiStep = 0;
    for (int l=1+mMaxLine/3; l<2*mMaxLine/3+1; ++l)
    {
        tU_INT4 i = aRasterIndexData.GetV(cPt2di(mMaxCol/2,l));
        tU_INT4 il = aRasterIndexData.GetV(cPt2di(mMaxCol/2,l-1));
        file_phi_abs<<i<<" "<<mVectPtsTPD[i].y()<< " "<<il<<" "<<mVectPtsTPD[il].y()<<"\n";
        if ((i!=0)&&(il!=0))
        {
            if ((mVectPtsTPD[i].z()<mDistMinToExist)
                ||(mVectPtsTPD[il].z()<mDistMinToExist))
            {
                std::cout<<"EEEERRRROROORORORO\n";
                std::cout<<i<<" "<<mVectPtsTPD[i]<< " "<<il<<" "<<mVectPtsTPD[il]<<"\n";
            }
            file_phi<<(mVectPtsTPD[i].y()-mVectPtsTPD[il].y())<<"\n";
            aPhiStep2+=(mVectPtsTPD[i].y()-mVectPtsTPD[il].y());
            nbPhiStep++;
        }
    }
    file_phi.close();
    file_phi_abs.close();
    aPhiStep2/=nbPhiStep;
    StdOut() << " New Phi Step: " << aPhiStep2<<"\n";
*/

    /*
    float minDiffPhi = INFINITY;
    float minDiffTheta = INFINITY;
    int signDiffPhi = 0;
    int signDiffTheta = 0;
    for (int l=1; l<aMaxLine+1; ++l)
    {
        for (int c=1; c<aMaxCol+1; ++c)
        {
            tU_INT4 i = aRasterIndexData.GetV(cPt2di(c,l));
            tU_INT4 ic = aRasterIndexData.GetV(cPt2di(c-1,l));
            tU_INT4 il = aRasterIndexData.GetV(cPt2di(c,l-1));
            if ((i!=0)&&(ic!=0)&&(il!=0))
            {
                if ((mVectPtsTPD[i].z()<aDistMinToExist)
                    ||(mVectPtsTPD[ic].z()<aDistMinToExist)
                    ||(mVectPtsTPD[il].z()<aDistMinToExist))
                {
                    std::cout<<"EEEERRRROROORORORO\n";
                    std::cout<<i<<" "<<mVectPtsTPD[i]<< " "<<ic<<" "<<mVectPtsTPD[ic]<<" "<<il<<" "<<mVectPtsTPD[il]<<"\n";
                }
                float aDiffTheta = mVectPtsTPD[i].x()-mVectPtsTPD[ic].x();
                if (minDiffTheta>fabs(aDiffTheta))
                {
                    std::cout<<"up th:"<<aDiffTheta<<"\n";
                    minDiffTheta = fabs(aDiffTheta);
                    signDiffTheta = fabs(aDiffTheta)/aDiffTheta;
                }
                float aDiffPhi = mVectPtsTPD[i].y()-mVectPtsTPD[il].y();
                if (minDiffPhi>fabs(aDiffPhi))
                {
                    std::cout<<"up ph:"<<aDiffTheta<<"\n";
                    minDiffPhi = fabs(aDiffPhi);
                    signDiffPhi = fabs(aDiffPhi)/aDiffPhi;
                }
                //aVDiffTheta.push_back(mVectPtsTPD[i2].x()-mVectPtsTPD[i1].x());
            }
        }
    }
    StdOut() << "Phi step found: "<<signDiffPhi*minDiffPhi<<"\n";
    StdOut() << "Theta step found: "<<signDiffTheta*minDiffTheta<<"\n";
    */

    /*
    //fill rasters
    cIm2D<tU_INT1> aRasterIntens2(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterIntens2Data = aRasterIntens2.DIm();
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        auto aPtAng = mVectPtsTPD[i];
        cPt2di aPcl = {mVectPtsCol[i], mVectPtsLine[i]};
        if (aPtAng.z()<mDistMinToExist)
        {
            continue;
        }
        aRasterIntens2Data.SetV(aPcl, mVectPtsIntens[i]*255);
    }
    aRasterIntens2Data.ToFile("titiIntens_.png");


    // search for min/max line and col with step
    float aMinLinef = INFINITY;
    float aMinColf = INFINITY;
    float aMaxLinef = -INFINITY;
    float aMaxColf = -INFINITY;
    float aCurrLineFloat, aCurrColFloat;
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        auto aPtAng = mVectPtsTPD[i];
        if (aPtAng.z()<mDistMinToExist)
        {
            continue;
        }
        if (aPhiStep2>0)
            aCurrLineFloat = (aPtAng.y()-mThetaMin)/aPhiStep2;
        else
            aCurrLineFloat = (aPtAng.y()-mThetaMax)/aPhiStep2;
        if (aThetaStep2>0)
            aCurrColFloat = (aPtAng.x()-mPhiMin)/aThetaStep2;
        else
            aCurrColFloat = (aPtAng.x()-mPhiMax)/aThetaStep2;
        if (aMinLinef > aCurrLineFloat)
            aMinLinef = aCurrLineFloat;
        if (aMinColf > aCurrColFloat)
            aMinColf = aCurrColFloat;
        if (aMaxLinef < aCurrLineFloat)
            aMaxLinef = aCurrLineFloat;
        if (aMaxColf < aCurrColFloat)
            aMaxColf = aCurrColFloat;
    }
    StdOut() << " " << aMinColf << "  " << aMaxColf << " " << aMinLinef << "  " << aMaxLinef << std::endl;
    cIm2D<float> aRasterDenity(cPt2di(aMaxColf-aMinColf+3, aMaxLinef-aMinLinef+3), 0, eModeInitImage::eMIA_Null); // +3 pixels for safety
    auto & aRasterDensityData = aRasterDenity.DIm();
    StdOut() << "aRasterDensityData.Sz(): " << aRasterDensityData.Sz() <<"\n";
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        auto aPtAng = mVectPtsTPD[i];
        if (aPtAng.z()<mDistMinToExist)
        {
            continue;
        }
        if (aPhiStep2>0)
            aCurrLineFloat = (aPtAng.y()-mThetaMin)/aPhiStep2;
        else
            aCurrLineFloat = (aPtAng.y()-mThetaMax)/aPhiStep2;
        if (aThetaStep2>0)
            aCurrColFloat = (aPtAng.x()-mPhiMin)/aThetaStep2;
        else
            aCurrColFloat = (aPtAng.x()-mPhiMax)/aThetaStep2;
        //StdOut()<<aCurrColFloat+2 << " " << aCurrLineFloat+2 <<"\n";
        auto aP2d = cPt2dr(aCurrColFloat+aMinColf+1., aCurrLineFloat+aMinLinef+1.);
        if (!aRasterDensityData.InsideBL(aP2d))
        {
            StdOut()<<"pb: " << aP2d <<"\n";
        }
        aRasterDensityData.AddVBL(aP2d, 1.);
    }
    aRasterDensityData.ToFile("titiDensity.tif");
    */

/*cIm2D<float> aRasterOrder(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterOrderData = aRasterOrder.DIm();
    for (size_t i=0; (i<10000) && (i<mVectPtsTPD.size()); ++i)
    {
        auto aPtAng = mVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist) continue;
        aRasterOrderData.SetV(cPt2di(aMaxCol-mVectPtsCol[i], aMaxLine-mVectPtsLine[i]), i);
    }
    aRasterOrderData.ToFile("order.tif");

    return EXIT_SUCCESS;
*/
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

