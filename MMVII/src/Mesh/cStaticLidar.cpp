#include "MMVII_StaticLidar.h"

#include <functional>
#include <fstream>

#include "../Mesh/happly.h"
#include "E57SimpleReader.h"
#include "MMVII_TplGradImFilter.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_2Include_CSV_Serial_Tpl.h"


namespace MMVII
{


cPt3dr cart2spher(const cPt3dr & aPtCart)
{
    tREAL8 dist = Norm2(aPtCart);
    tREAL8 theta =  atan2(aPtCart.y(),aPtCart.x());
    tREAL8 distxy = sqrt(aPtCart.BigX2()+aPtCart.BigY2());
    tREAL8 phi =  atan2(aPtCart.z(),distxy);
    return {theta, phi, dist};
}

cPt3dr spher2cart(const cPt3dr & aPtspher)
{
    tREAL8 dhz = aPtspher.z()*cos(aPtspher.y());
    tREAL8 x = dhz* cos(aPtspher.x());
    tREAL8 y = dhz* sin(aPtspher.x());
    tREAL8 z = aPtspher.z()*sin(aPtspher.y());
    return {x, y, z};
}

tREAL8 toMinusPiPlusPi(tREAL8 aAng, tREAL8 aOffset)
{
    if (!std::isfinite(aAng))
        return aAng;
    while (aAng-aOffset<-M_PI)  aAng += 2*M_PI;
    while (aAng-aOffset>M_PI)  aAng -= 2*M_PI;
    return aAng;
}

cStaticLidarImporter::cStaticLidarImporter() :
    mNoMiss(false), mIsStrucured(false),
    mReadPose(tPoseR::Identity()), mDistMinToExist(1e-5)
{

}

using  namespace happly;

void cStaticLidarImporter::readPlyPoints(std::string aPlyFileName)
{
    StdOut() << "Read ply file " << aPlyFileName << "..." << std::endl;
    mVectPtsXYZ.clear();
    mVectPtsTPD.clear();
    mVectPtsIntens.clear();
    mVectPtsCol.clear();
    mVectPtsLine.clear();
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

        mHasCartesian = true;
        mHasIntensity = false;
        mHasSpherical = false;
        mHasRowCol = false;

        // try to fill points attribute with any "i", "I" or "*intens*" property
        auto aVertProps = aPlyF.getElement("vertex").getPropertyNames();
        auto aPropIntensityName = std::find_if(aVertProps.begin(), aVertProps.end(), [](const std::string &s){
            return (ToLower(s)=="i") || (ToLower(s).find("intens") != std::string::npos);
        });
        if (aPropIntensityName!= aVertProps.end())
        {
            mHasIntensity = true;
            mVectPtsIntens = aPlyF.getElement("vertex").getProperty<tREAL8>(*aPropIntensityName);
        }

    }
    catch (const std::runtime_error &e)
    {
        MMVII_UserError(eTyUEr::eReadFile, std::string("Error reading PLY file \"") + aPlyFileName + "\": " + e.what());
    }
}

void cStaticLidarImporter::readE57Points(std::string aE57FileName)
{
    StdOut() << "Read e57 file " << aE57FileName << "..." << std::endl;
    mVectPtsXYZ.clear();
    mVectPtsTPD.clear();
    mVectPtsIntens.clear();
    mVectPtsCol.clear();
    mVectPtsLine.clear();
    try
    {
        e57::Reader reader( aE57FileName, {});
        MMVII_INTERNAL_ASSERT_tiny(reader.IsOpen(), "Error: unable to open file " + aE57FileName)
        StdOut() << "Image2DCount: " << reader.GetImage2DCount() << "\n";
        StdOut() << "Data3DCount: " << reader.GetData3DCount() << "\n";
        MMVII_INTERNAL_ASSERT_tiny(reader.GetData3DCount()==1, "Error: File should have exactly 1 scan for now")
        e57::E57Root fileHeader;
        reader.GetE57Root( fileHeader );
        /*StdOut() << fileHeader.formatName << " =? " << "ASTM E57 3D Imaging Data File" << std::endl;
        StdOut() << fileHeader.versionMajor << " =? " << 1 << std::endl;
        StdOut() << fileHeader.versionMinor << " =? " << 0 << std::endl;
        StdOut() << fileHeader.guid << " =? " << "Zero Points GUID" << std::endl;*/
        e57::Data3D data3DHeader;
        reader.ReadData3D( 0, data3DHeader );
        // data3DHeader.indexBounds is not correct
        const uint64_t cNumPoints = data3DHeader.pointCount;
        e57::Data3DPointsFloat pointsData( data3DHeader );
        auto vectorReader = reader.SetUpData3DPointsData( 0, cNumPoints, pointsData );
        const uint64_t cNumRead = vectorReader.read();
        MMVII_INTERNAL_ASSERT_tiny(cNumPoints==cNumRead, "Error: cNumPoints!=cNumRead")

        mHasCartesian = pointsData.cartesianX && pointsData.cartesianY && pointsData.cartesianZ;
        mHasIntensity = pointsData.intensity;
        mHasSpherical = pointsData.sphericalAzimuth && pointsData.sphericalElevation && pointsData.sphericalRange;
        mHasRowCol = pointsData.columnIndex && pointsData.rowIndex;

        if (mHasCartesian){
            mVectPtsXYZ.resize(cNumRead);
            for (uint64_t i=0;i<cNumRead;++i)
                mVectPtsXYZ[i] = {pointsData.cartesianX[i], pointsData.cartesianY[i], pointsData.cartesianZ[i]};
        }
        if (mHasSpherical){
            mVectPtsTPD.resize(cNumRead);
            for (uint64_t i=0;i<cNumRead;++i)
                mVectPtsTPD[i] = {pointsData.sphericalAzimuth[i], pointsData.sphericalElevation[i], pointsData.sphericalRange[i]};
        }
        if (mHasIntensity){
            mVectPtsIntens.resize(cNumRead);
            for (uint64_t i=0;i<cNumRead;++i)
                mVectPtsIntens[i] = pointsData.intensity[i];
        }
        if (mHasRowCol){
            mVectPtsLine.resize(cNumRead);
            mVectPtsCol.resize(cNumRead);
            mMaxLine = 0;
            for (uint64_t i=0;i<cNumRead;++i)
            {
                mVectPtsLine[i] = pointsData.rowIndex[i];
                if (pointsData.rowIndex[i]>mMaxLine)
                    mMaxLine = pointsData.rowIndex[i];
            }
            mMaxCol = 0;
            for (uint64_t i=0;i<cNumRead;++i)
            {
                mVectPtsCol[i] = pointsData.columnIndex[i];
                if (pointsData.columnIndex[i]>mMaxCol)
                    mMaxCol = pointsData.columnIndex[i];
            }
        }

        vectorReader.close();

    }
    catch (const std::runtime_error &e)
    {
        MMVII_UserError(eTyUEr::eReadFile, std::string("Error reading E57 file \"") + aE57FileName + "\": " + e.what());
    }
}


void cStaticLidarImporter::readPtxPoints(std::string aPtxFileName)
{
    StdOut() << "Read PTX file " << aPtxFileName << "..." << std::endl;
    mVectPtsXYZ.clear();
    mVectPtsTPD.clear();
    mVectPtsIntens.clear();
    mVectPtsCol.clear();
    mVectPtsLine.clear();
    try
    {
        std::ifstream  aPtxFile(aPtxFileName);
        mHasCartesian = true;
        mHasIntensity = true; // to check
        mHasSpherical = false;
        mHasRowCol = true; // directly computed
        mIsStrucured = true;
        mNoMiss = false;

        aPtxFile >> mMaxCol;
        aPtxFile >> mMaxLine;
        tREAL8 aTx, aTy, aTz;
        aPtxFile >> aTx >> aTy >> aTz;
        tREAL8 aR11, aR12, aR13;
        aPtxFile >> aR11 >> aR12 >> aR13;
        tREAL8 aR21, aR22, aR23;
        aPtxFile >> aR21 >> aR22 >> aR23;
        tREAL8 aR31, aR32, aR33;
        aPtxFile >> aR31 >> aR32 >> aR33;
        mReadPose.Tr() = {aTx, aTy, aTz};
        mReadPose.Rot() = cRotation3D<tREAL8>({aR11, aR12, aR13}, {aR21, aR22, aR23}, {aR31, aR32, aR33}, false);
        char tmp[200];
        aPtxFile.getline(tmp, 200); // for now just skip transformation matrix
        aPtxFile.getline(tmp, 200);
        aPtxFile.getline(tmp, 200);
        aPtxFile.getline(tmp, 200);

        mVectPtsXYZ.resize(mMaxCol*mMaxLine);
        mVectPtsCol.resize(mMaxCol*mMaxLine);
        mVectPtsLine.resize(mMaxCol*mMaxLine);

        long i =0 ;
        tREAL8 aX, aY, aZ, aI;
        // for 1st line we test if there is intensity
        aPtxFile.getline(tmp, 200);
        std::istringstream iss(tmp);
        iss >> aX >> aY >> aZ >> aI;
        mVectPtsXYZ[i] = {aX, aY, aZ};
        if (iss.bad())
        {
            mHasIntensity = false;
        } else {
            mVectPtsIntens.resize(mMaxCol*mMaxLine);
            mVectPtsIntens[i] = aI;
        }

        for (long aCol = 0; aCol < mMaxCol; aCol++)
        {
            for (long aRow = 0; aRow < mMaxLine; aRow++)
            {
                ++i;
                aPtxFile.getline(tmp, 200);
                std::istringstream iss(tmp);
                iss >> aX >> aY >> aZ;
                mVectPtsXYZ[i] = {aX, aY, aZ};
                mVectPtsCol[i] = aCol;
                mVectPtsLine[i] = aRow;
                if (mHasIntensity)
                {
                    iss >> aI;
                    mVectPtsIntens[i] = aI;
                }
            }
        }


    }
    catch (const std::runtime_error &e)
    {
        MMVII_UserError(eTyUEr::eReadFile, std::string("Error reading PTX file \"") + aPtxFileName + "\": " + e.what());
    }
}

bool cStaticLidarImporter::read(const std::string & aName, bool OkNone, bool aForceStructured)
{
    std::string aPost = LastPostfix(aName);
    if (UCaseEqual(aPost,"ply"))
       readPlyPoints(aName);
    else if (UCaseEqual(aPost,"e57"))
       readE57Points(aName);
    else if (UCaseEqual(aPost,"ptx"))
        readPtxPoints(aName);
    else
    {
        if (! OkNone)
        {
           MMVII_UnclasseUsEr("Cannot read cloud for " + aName);
        }
        return false;
    }
    if (aForceStructured)
        mIsStrucured = true;

    if (!mHasIntensity)
    {
        // fake intensity
        mVectPtsIntens.resize(std::max(mVectPtsXYZ.size(),mVectPtsTPD.size()), 0.5);
        mHasIntensity = true;
    }

    return true;
}



void cStaticLidarImporter::convertToThetaPhiDist()
{
    StdOut() << "convertToThetaPhiDist\n";
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
    mNoMiss = mIsStrucured || (aNbPtsNul>0);
}


void cStaticLidarImporter::convertToXYZ()
{
    StdOut() << "convertToXYZ\n";
    mNoMiss = false;
    mVectPtsXYZ.resize(mVectPtsTPD.size());
    size_t aNbPtsNul = 0;
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        mVectPtsXYZ[i] = spher2cart(mVectPtsTPD[i]);
        if (mVectPtsTPD[i].z()<mDistMinToExist)
            aNbPtsNul++;
    }
    StdOut() << aNbPtsNul << " null points\n";
    mNoMiss = mIsStrucured || (aNbPtsNul>0);
}


cStaticLidar::cStaticLidar(const std::string & aNameFile, const tPose & aPose, cPerspCamIntrCalib * aCalib) :
    cSensorCamPC(aNameFile, aPose, aCalib),
    mThetaStart       (NAN),
    mThetaStep         (NAN),
    mPhiStart         (NAN),
    mPhiStep           (NAN),
    mMaxCol            (0),
    mMaxLine           (0),
    mVertRot           (cRotation3D<tREAL8>::Identity())
{
}

cStaticLidar * cStaticLidar::FromFile(const std::string & aNameFile, const std::string &aNameRastersDir)
{
    cStaticLidar * aRes = new cStaticLidar("NONE",tPoseR::Identity(),nullptr);
    ReadFromFile(*aRes, aNameFile);
    aRes->mRasterDistance = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterDistancePath));
    aRes->mRasterIntensity = std::make_unique<cIm2D<tU_INT1>>(cIm2D<tU_INT1>::FromFile(aNameRastersDir+"/"+aRes->mRasterIntensityPath));
    aRes->mRasterMask = std::make_unique<cIm2D<tU_INT1>>(cIm2D<tU_INT1>::FromFile(aNameRastersDir+"/"+aRes->mRasterMaskPath));
    aRes->mRasterX = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterXPath));
    aRes->mRasterY = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterYPath));
    aRes->mRasterZ = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterZPath));
    return aRes;
}

long cStaticLidar::NbPts() const
{
    return mMaxCol * mMaxLine;
}

float cStaticLidar::ColToLocalThetaApprox(float aCol) const
{
    return mThetaStart + aCol * mThetaStep;
}

float cStaticLidar::LineToLocalPhiApprox(float aLine) const
{
    return mPhiStart + aLine * mPhiStep;
}

cPt3dr cStaticLidar::to3D(cPt2di aRasterPx) const
{
    auto & aRasterXData = mRasterX->DIm();
    auto & aRasterYData = mRasterY->DIm();
    auto & aRasterZData = mRasterZ->DIm();
    return cPt3dr{
        aRasterXData.GetV(aRasterPx),
        aRasterYData.GetV(aRasterPx),
        aRasterZData.GetV(aRasterPx),
    };
}

void cStaticLidar::ToPly(const std::string & aName,bool useMask) const
{
    std::vector<cPt4dr> aSelectionXYZI;
    aSelectionXYZI.reserve(mMaxCol*mMaxLine);
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    auto & aRasterIntensityData = mRasterIntensity->DIm();
    auto & aRasterXData = mRasterX->DIm();
    auto & aRasterYData = mRasterY->DIm();
    auto & aRasterZData = mRasterZ->DIm();
    for (int l = 0 ; l <= mMaxLine; ++l)
    {
        for (int c = 0 ; c <= mMaxCol; ++c)
        {
            cPt2di aPt(c, l);
            if (aMaskImData.GetV(aPt))
                aSelectionXYZI.push_back(cPt4dr(aRasterXData.GetV(aPt),aRasterYData.GetV(aPt),
                                                aRasterZData.GetV(aPt),aRasterIntensityData.GetV(aPt)));
        }
    }

    cMMVII_Ofs anOfs(aName,eFileModeOut::CreateText);

    bool  aMode8 =  false;

    std::string aSpecCoord = aMode8 ? "float64" : "float32";
    anOfs.Ofs() <<  "ply\n";
    anOfs.Ofs() <<  "format ascii 1.0\n";
    anOfs.Ofs() <<  "comment Generated by MMVVI\n";
    anOfs.Ofs() <<  "element vertex " << aSelectionXYZI.size() << "\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" x\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" y\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" z\n";
    anOfs.Ofs() <<  "property uchar intensity\n";
    anOfs.Ofs() <<  "element face 0\n";
    anOfs.Ofs() <<  "end_header\n";


    for (auto& aPt : aSelectionXYZI)
    {
        anOfs.Ofs() << aPt.x() << " " << aPt.y() << " " << aPt.z();
        anOfs.Ofs() << " " << aPt.t();
        anOfs.Ofs() << "\n";
    }
}

template <typename TYPE> void cStaticLidar::fillRaster(const std::string& aPhProjDirOut, const std::string& aFileName,
                              std::function<TYPE (int)> func, std::unique_ptr<cIm2D<TYPE> > & aIm, bool saveRaster)
{
    MMVII_INTERNAL_ASSERT_tiny(mSL_importer.mVectPtsCol.size()==mSL_importer.mVectPtsXYZ.size(), "Error: Compute line/col numbers before fill raster");

    aIm.reset(new cIm2D<TYPE>(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null));
    auto & aRasterData = aIm->DIm();
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        cPt2di aPcl = {mSL_importer.mVectPtsCol[i], mSL_importer.mVectPtsLine[i]};
        aRasterData.SetV(aPcl, func(i));
    }
    if (saveRaster)
        aRasterData.ToFile(aPhProjDirOut + aFileName);
}

template <typename TYPE> void cStaticLidar::fillRaster(const std::string& aPhProjDirOut, const std::string& aFileName,
                              std::function<TYPE (int)> func, bool saveRaster)
{
    std::unique_ptr<cIm2D<TYPE>> aIm; // temporary image
    fillRaster(aPhProjDirOut, aFileName, func, aIm, saveRaster);
}

void cStaticLidar::fillRasters(const std::string& aPhProjDirOut, bool saveRasters)
{
    mRasterDistancePath = mStationName + "_" + mScanName + "_distance.tif";
    mRasterIntensityPath = mStationName + "_" + mScanName + "_intensity.tif";
    mRasterMaskPath = mStationName + "_" + mScanName + "_mask.tif";
    mRasterXPath = mStationName + "_" + mScanName + "_X.tif";
    mRasterYPath = mStationName + "_" + mScanName + "_Y.tif";
    mRasterZPath = mStationName + "_" + mScanName + "_Z.tif";

    mRasterThetaPath = mStationName + "_" + mScanName + "_Theta.tif";
    mRasterPhiPath = mStationName + "_" + mScanName + "_Phi.tif";
    mRasterThetaErrPath = mStationName + "_" + mScanName + "_ThetaErr.tif";
    mRasterPhiErrPath = mStationName + "_" + mScanName + "_PhiErr.tif";

    fillRaster<tU_INT1>(aPhProjDirOut, mRasterMaskPath, [this](int i)
                        {
                            auto aPtAng = mSL_importer.mVectPtsTPD[i];
                            return (aPtAng.z()<mSL_importer.DistMinToExist())?0:255;
                        }, mRasterMask, saveRasters);
    fillRaster<tU_INT1>(aPhProjDirOut, mRasterIntensityPath, [this](int i){return mSL_importer.mVectPtsIntens[i]*255;}, mRasterIntensity, saveRasters );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterDistancePath,
                      [this](int i){auto aPtAng = mSL_importer.mVectPtsTPD[i];return aPtAng.z();},
                       mRasterDistance, saveRasters);

    fillRaster<tREAL4>(aPhProjDirOut, mRasterXPath, [this](int i){auto aPtXYZ = mSL_importer.mVectPtsXYZ[i];return aPtXYZ.x();}, mRasterX, saveRasters );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterYPath, [this](int i){auto aPtXYZ = mSL_importer.mVectPtsXYZ[i];return aPtXYZ.y();}, mRasterY, saveRasters );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterZPath, [this](int i){auto aPtXYZ = mSL_importer.mVectPtsXYZ[i];return aPtXYZ.z();}, mRasterZ, saveRasters );

    fillRaster<tREAL4>(aPhProjDirOut, mRasterThetaPath, [this](int i){auto aPtAng = mSL_importer.mVectPtsTPD[i];return aPtAng.x();}, saveRasters );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterPhiPath, [this](int i){auto aPtAng = mSL_importer.mVectPtsTPD[i];return aPtAng.y();}, saveRasters );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterThetaErrPath, [this](int i)
                      {
                          auto aPtAng = mSL_importer.mVectPtsTPD[i];
                          tREAL8 aThetaCol = mThetaStart + mThetaStep * mSL_importer.mVectPtsCol[i];
                          aThetaCol = toMinusPiPlusPi(aThetaCol);
                          return aPtAng.x()-aThetaCol;
                      }, saveRasters );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterPhiErrPath, [this](int i)
                      {
                          auto aPtAng = mSL_importer.mVectPtsTPD[i];
                          tREAL8 aPhiLine = mPhiStart + mPhiStep * mSL_importer.mVectPtsLine[i];
                          aPhiLine = toMinusPiPlusPi(aPhiLine);
                          return aPtAng.y()-aPhiLine;
                      }, saveRasters );

    mRasterScore.reset(new cIm2D<tREAL4>(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null));
}

void cStaticLidar::FilterIntensity(tREAL8 aLowest, tREAL8 aHighest)
{
    if (!mSL_importer.HasIntensity())
        return;
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    auto & aRasterScoreData = mRasterScore->DIm();
    tREAL8 aMiddle = (aLowest + aHighest) / 2.;
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        cPt2di aPcl = {mSL_importer.mVectPtsCol[i], mSL_importer.mVectPtsLine[i]};
        if ((mSL_importer.mVectPtsIntens[i]<aLowest) || (mSL_importer.mVectPtsIntens[i]>aHighest))
            aMaskImData.SetV(aPcl, 0);
        aRasterScoreData.SetV(aPcl, aRasterScoreData.GetV(aPcl) + fabs(mSL_importer.mVectPtsIntens[i]-aMiddle));
    }
    aMaskImData.ToFile("MaskIntens.png");
}

void cStaticLidar::FilterIncidence(tREAL8 aAngMax)
{
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    auto & aRasterScoreData = mRasterScore->DIm();

    // TODO: use im.InitCste()
    cIm2D<tREAL4> aImDistGrX(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aImDistGrXData = aImDistGrX.DIm();
    cIm2D<tREAL4> aImDistGrY(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aImDistGrYData = aImDistGrY.DIm();

    tREAL4 aTanAngMax = tan(aAngMax);

    // gaussian blur of masked image: blur image and mask, for valid pixels, result = blured_im/blured_mask
    auto aRasterDistGauss = mRasterDistance->Dup();
    auto & aRasterDistGaussData = aRasterDistGauss.DIm();
    ExpFilterOfStdDev(aRasterDistGaussData, 2, 3.);

    mRasterMask->DIm().ToFile("Mask.tif");
    auto aRasterMaskGauss = Convert((float*)nullptr, mRasterMask->DIm()) * (1./255.);
    auto & aRasterMaskGaussData = aRasterMaskGauss.DIm();
    ExpFilterOfStdDev(aRasterMaskGaussData, 2, 3.);

    aRasterDistGaussData.ToFile("DistGaussData.tif");
    aRasterMaskGaussData.ToFile("MaskGaussData.tif");

    cImGrad<tREAL4> aDistGradIm(aRasterDistGauss);
    ComputeSobel<tREAL4,tREAL4>(*aDistGradIm.mDGx, *aDistGradIm.mDGy, aRasterDistGaussData);
    for (int l = 0 ; l <= mMaxLine; ++l)
    {
        tREAL4 phi = lToPhiApprox(l);
        tREAL4 aStepThetaFix = mThetaStep*cos(phi);
        for (int c = 0 ; c <= mMaxCol; ++c)
        {
            cPt2di aPt(c, l);
            tREAL4 aDist = mRasterDistance->DIm().GetV(aPt);
            tREAL4 aValDistGradX = aDistGradIm.Gx(aPt);
            tREAL4 aValDistGradY = aDistGradIm.Gy(aPt);
            tREAL4 aValGaussMask = aRasterMaskGaussData.GetV(aPt);
            tREAL4 aValMask = mRasterMask->DIm().GetV(aPt);
            if (! aValMask)
                continue;
            tREAL4 aTanIncidX = aValDistGradX / (aStepThetaFix * aDist) / aValGaussMask;
            aImDistGrXData.SetV(aPt, aTanIncidX);
            tREAL4 aTanIncidY = aValDistGradY / (mPhiStep * aDist) / aValGaussMask;
            aImDistGrYData.SetV(aPt, aTanIncidY);
            if (fabs(aTanIncidX*aTanIncidX+aTanIncidY*aTanIncidY)>aTanAngMax*aTanAngMax)
                aMaskImData.SetV(aPt, 0);
            aRasterScoreData.SetV(aPt, aRasterScoreData.GetV(aPt) + 10.*fabs(aTanIncidX*aTanIncidX+aTanIncidY*aTanIncidY));
        }
    }
    aImDistGrXData.ToFile("DistGrXData.tif");
    aImDistGrYData.ToFile("DistGrYData.tif");
    aMaskImData.ToFile("MaskIncidence.png");
}

void cStaticLidar::FilterDistance(tREAL8 aDistMin, tREAL8 aDistMax)
{
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    //auto & aRasterScoreData = mRasterScore->DIm(); // add something to mRasterScore?
    auto & aRasterDistData = mRasterDistance->DIm();
    for (int l = 0 ; l <= mMaxLine; ++l)
    {
        for (int c = 0 ; c <= mMaxCol; ++c)
        {
            cPt2di aPt(c, l);
            tREAL4 aDist = aRasterDistData.GetV(aPt);
            if ((aDist<aDistMin)||(aDist>aDistMax))
            {
                aMaskImData.SetV(aPt, 0);
            }
        }
    }
}

void cStaticLidar::MaskBuffer(tREAL8 aAngBuffer, const std::string &aPhProjDirOut)
{
    StdOut() << "Computing Mask buffer..."<<std::endl;
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    mRasterMaskBuffer.reset( new cIm2D<tU_INT1>(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_NoInit));
    auto & aMaskBufImData = mRasterMaskBuffer->DIm();

    auto & aRasterScoreData = mRasterScore->DIm();

    bool aHzLoop = false;
    if (fabs(fabs(mThetaStep) * (mMaxCol+1) - 2 * M_PI) < 2 * fabs(mThetaStep))
        aHzLoop = true;

    tREAL8 aRadPx = aAngBuffer/mPhiStep;
    aMaskBufImData.InitCste(255);

    std::vector<bool> aLinesFull(mMaxLine+1, false); // record lignes completely masked to pass them next time
    // int c = 100;
    // for (int l = 100; l < 2700; l += 500)
    for (int l = 0 ; l <= mMaxLine; ++l)
    {
        for (int c = 0 ; c <= mMaxCol; ++c)
        {
            auto aMaskVal = aMaskImData.GetV(cPt2di(c, l));
            if (aMaskVal==0)
            {
                for (int il = l - aRadPx; il <= l + aRadPx; ++il)
                {
                    if ((il<0) || (il>mMaxLine)) continue;
                    if (aLinesFull[il]) continue;
                    tREAL8 phi = lToPhiApprox(il);
                    tREAL8 w = fabs(sqrt(aRadPx*aRadPx - (il-l)*(il-l))/cos(phi));
                    if (w>mMaxCol)
                    {
                        w=mMaxCol;
                        aLinesFull[il] = true;
                        // TODO: fill line and continue
                    }
                    for (int ic = c - w; ic <= c + w; ++ic)
                    {
                        int icc = ic; // working copy
                        if (aHzLoop)
                        {
                            if (icc<0)
                                icc += (mMaxCol+1);
                            if (icc>mMaxCol)
                                icc -= (mMaxCol+1);
                        }
                        if ((icc<0)||(icc>mMaxCol))
                            continue;
                        aMaskBufImData.SetV(cPt2di(icc, il), 0);
                    }
                }
            }
        }
    }
    for (int l = 0 ; l <= mMaxLine; ++l)
        for (int c = 0 ; c <= mMaxCol; ++c)
        {
            if (aMaskBufImData.GetV(cPt2di(c, l))==0)
                aRasterScoreData.SetV(cPt2di(c, l), 1000.);
        }
    aMaskBufImData.ToFile("MaskBuff.png");
    //record as new mask
    aMaskBufImData.DupIn(aMaskImData);
    aMaskImData.ToFile(aPhProjDirOut + mRasterMaskPath);
}

void cStaticLidar::SelectPatchCenters1(int aNbPatches)
{
    mPatchCenters.clear();
    float aNbPatchesFactor = 2.; // a priori search for aNbPatches * aNbPatchesFactor
    auto & aRasterScoreData = mRasterScore->DIm();
    cResultExtremum aRes;
    double aRadius = sqrt(aRasterScoreData.SzX()*aRasterScoreData.SzY()/(M_PI*aNbPatches))/aNbPatchesFactor;
    ExtractExtremum1(aRasterScoreData, aRes, aRadius);
    mPatchCenters = aRes.mPtsMin;
    StdOut() << "Nb pathes: " << mPatchCenters.size() <<"\n";
    std::fstream file1;
    file1.open("centers.txt", std::ios_base::out);
    for (auto & aCenter : mPatchCenters)
    {
        file1 << aCenter.x() << " " << -aCenter.y() <<"\n";
    }
    aRasterScoreData.ToFile("Score.tif");
}

void cStaticLidar::SelectPatchCenters2(int aNbPatches)
{
    mPatchCenters.clear();
    auto & aRasterMaskData = mRasterMask->DIm();
    /*cResultExtremum aRes;
    double aRadius = sqrt(aRasterMaskBufferData.SzX()*aRasterMaskBufferData.SzY()/(M_PI*nbPatches));
    ExtractExtremum1(aRasterMaskBufferData, aRes, aRadius);
    mPatchCenters = aRes.mPtsMax;*/

    // regular grid
    float aAvgDist = 3.;
    auto & aRasterDistData = mRasterDistance->DIm();
    float aXYratio=((float)aRasterMaskData.SzX())/aRasterMaskData.SzY();
    int aNbPatchesX = sqrt((double)aNbPatches)*sqrt(aXYratio)+1;
    int aNbPatchesY = sqrt((double)aNbPatches)/sqrt(aXYratio)+1;
    float aNbPatchesFactor = 1.5; // a priori search for aNbPatches * aNbPatchesFactor
    float aX;
    float aY = float(aRasterMaskData.SzY()) / aNbPatchesY / 2.;
    float aXStep;
    float aYStep = float(aRasterMaskData.SzY()) / aNbPatchesY / aNbPatchesFactor;
    if (aYStep<1.)
        aYStep = 1.;
    int aLineCounter = 0;
    while (aY<aRasterMaskData.SzY())
    {
        aX = float(aRasterMaskData.SzX()) / aNbPatchesX * ((aLineCounter%2)?1./3.:2./3.);
        while (aX<aRasterMaskData.SzX())
        {
            // take lat/long proj into account
            aXStep = fabs(((float)aRasterMaskData.SzX()) / aNbPatchesX / aNbPatchesFactor / cos(LineToLocalPhiApprox(aY)));
            auto aPt = cPt2di(aX, aY);
            if (aRasterMaskData.GetV(aPt))
            {
                mPatchCenters.push_back(aPt);
                aXStep *= aAvgDist/aRasterDistData.GetV(aPt); // take depth into account
            } else
                aXStep /= 3.;
            if (aXStep<1.)
                aXStep = 1.;
            aX += aXStep;

        }
        aY += aYStep;
        aLineCounter++;
    }

    StdOut() << "Nb pathes: " << mPatchCenters.size() <<"\n";
    std::fstream file1;
    file1.open("centers.txt", std::ios_base::out);
    for (auto & aCenter : mPatchCenters)
    {
        file1 << aCenter.x() << " " << -aCenter.y() <<"\n";
    }
}


void cStaticLidar::MakePatches
    (
        std::list<std::vector<cPt2di> > & aLPatches,
        tREAL8 aGndPixelSize,
        int    aNbPointByPatch,
        int    aSzMin
        ) const
{
    auto & aRasterDistData = mRasterDistance->DIm();
    auto & aRasterMaskData = mRasterMask->DIm();

    // shortcut if only 1 point needed: just get the centers
    if (aNbPointByPatch==1)
    {
        for (auto & aCenter: mPatchCenters)
        {
            aLPatches.push_back( {aCenter} );
        }
        return;
    }

    // parse center points
    for (auto & aCenter: mPatchCenters)
    {
        // compute raster step to get aNbPointByPatch separated by aGndPixelSize
        tREAL4 aMeanDepth = aRasterDistData.GetV(aCenter);
        tREAL4 aProjColFactor = 1/cos(LineToLocalPhiApprox(aCenter.y()));
        tREAL4 aNbStepRadius = sqrt(aNbPointByPatch/M_PI) + 1;
        tREAL4 aRasterPxGndW = mThetaStep * aMeanDepth * aProjColFactor;
        tREAL4 aRasterPxGndH = mThetaStep * aMeanDepth;
        tREAL4 aRasterStepPixelsY = aGndPixelSize / aRasterPxGndH;
        tREAL4 aRasterStepPixelsX = aGndPixelSize / aRasterPxGndW;

        std::vector<cPt2di> aPatch;
        for (int aJ = -aNbStepRadius; aJ<=aNbStepRadius; ++aJ)
            for (int aI = -aNbStepRadius; aI<=aNbStepRadius; ++aI)
            {
                cPt2di aPt = aCenter + cPt2di(aI*aRasterStepPixelsX,aJ*aRasterStepPixelsY);
                if (aRasterMaskData.Inside(aPt) && aRasterMaskData.GetV(aPt))
                    aPatch.push_back(aPt);
            }

        // some requirement on minimal size
        if ((int)aPatch.size() > aSzMin)
        {
            aLPatches.push_back(aPatch);
        }
    }
}


void cStaticLidar::AddData(const  cAuxAr2007 & anAux)
{
    cSensorCamPC::AddData(anAux);
    MMVII::AddData(cAuxAr2007("StationName",anAux),mStationName);
    MMVII::AddData(cAuxAr2007("ScanName",anAux),mScanName);
    MMVII::AddData(cAuxAr2007("ThetaStart",anAux),mThetaStart);
    MMVII::AddData(cAuxAr2007("ThetaStep",anAux),mThetaStep);
    MMVII::AddData(cAuxAr2007("PhiStart",anAux),mPhiStart);
    MMVII::AddData(cAuxAr2007("Phistep",anAux),mPhiStep);
    MMVII::AddData(cAuxAr2007("MaxCol",anAux),mMaxCol);
    MMVII::AddData(cAuxAr2007("MaxLine",anAux),mMaxLine);
    MMVII::AddData(cAuxAr2007("VertRot",anAux),mVertRot);

    MMVII::AddData(cAuxAr2007("RasterDistance",anAux),mRasterDistancePath);
    MMVII::AddData(cAuxAr2007("RasterIntensity",anAux),mRasterIntensityPath);
    MMVII::AddData(cAuxAr2007("RasterMask",anAux),mRasterMaskPath);
    MMVII::AddData(cAuxAr2007("RasterX",anAux),mRasterXPath);
    MMVII::AddData(cAuxAr2007("RasterY",anAux),mRasterYPath);
    MMVII::AddData(cAuxAr2007("RasterZ",anAux),mRasterZPath);

    MMVII::AddData(cAuxAr2007("PatchCenters",anAux),mPatchCenters);

    MMVII::AddData(cAuxAr2007("RasterTheta",anAux),mRasterThetaPath);
    MMVII::AddData(cAuxAr2007("RasterPhi",anAux),mRasterPhiPath);
    MMVII::AddData(cAuxAr2007("RasterThetaErr",anAux),mRasterThetaErrPath);
    MMVII::AddData(cAuxAr2007("RasterPhiErr",anAux),mRasterPhiErrPath);
}

void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL)
{
   aSL.AddData(anAux);
}

};

