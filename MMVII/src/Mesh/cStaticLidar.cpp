#include "MMVII_StaticLidar.h"

#include <functional>
#include <fstream>

#include "../Mesh/happly.h"
#include "E57SimpleReader.h"
#include "MMVII_TplGradImFilter.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_2Include_CSV_Serial_Tpl.h"
#include "../SymbDerGen/Formulas_CentralProj.h"


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
    mReadPose(tPoseR::Identity()), mDistMinToExist(1e-5),
    mNbCol            (0),
    mNbLine           (0),
    mThetaStart        (NAN),
    mThetaStep         (NAN),
    mPhiStart          (NAN),
    mPhiStep           (NAN),
    mRotInput2TSL      (cRotation3D<tREAL8>::Identity()),
    mVertRot           (cRotation3D<tREAL8>::Identity())
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
            mNbLine = 0;
            for (uint64_t i=0;i<cNumRead;++i)
            {
                mVectPtsLine[i] = pointsData.rowIndex[i];
                if (pointsData.rowIndex[i] >= mNbLine)
                    mNbLine = pointsData.rowIndex[i] + 1;
            }
            mNbCol = 0;
            for (uint64_t i=0;i<cNumRead;++i)
            {
                mVectPtsCol[i] = pointsData.columnIndex[i];
                if (pointsData.columnIndex[i] >= mNbCol)
                    mNbCol = pointsData.columnIndex[i] + 1;
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

        aPtxFile >> mNbCol;
        aPtxFile >> mNbLine;
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

        mVectPtsXYZ.resize(mNbCol*mNbLine);
        mVectPtsCol.resize(mNbCol*mNbLine);
        mVectPtsLine.resize(mNbCol*mNbLine);

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
            mVectPtsIntens.resize(mNbCol*mNbLine);
            mVectPtsIntens[i] = aI;
        }

        for (long aCol = 0; aCol < mNbCol; aCol++)
        {
            for (long aRow = 0; aRow < mNbLine; aRow++)
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

bool cStaticLidarImporter::read(const std::string & aName, bool OkNone, bool aForceStructured, std::string aStrInput2TSL)
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


    StdOut() << "Read data: ";
    if (HasCartesian())
        StdOut() << mVectPtsXYZ.size() << " cartesian points";
    if (HasSpherical())
        StdOut() << mVectPtsTPD.size() << " spherical points";
    if (HasIntensity())
        StdOut() << " with intensity";
    if (HasRowCol())
        StdOut() << " with row-col";
    StdOut() << "\n";

    if (HasCartesian() && !HasSpherical())
    {
        mRotInput2TSL = cRotation3D<tREAL8>::RotFromCanonicalAxes(aStrInput2TSL);
        // apply rotframe to original points
        for (auto & aPtXYZ : mVectPtsXYZ)
        {
            aPtXYZ = mRotInput2TSL.Value(aPtXYZ);
        }
        convertToThetaPhiDist();
        // go back to original xyz
        for (auto & aPtXYZ : mVectPtsXYZ)
        {
            aPtXYZ = mRotInput2TSL.Inverse(aPtXYZ);
        }
    } else if (!HasCartesian() && HasSpherical()) // mTransfoIJK not used if spherical
    {
        convertToXYZ();
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

void cStaticLidarImporter::ComputeRotInput2Raster(std::string aTransfoIJK)
{
    MMVII_INTERNAL_ASSERT_tiny(HasRowCol(),"Error: ComputeRotInput2Raster needs row/col");

    cRotation3D<tREAL8> aRotInput2TSL = cRotation3D<tREAL8>::RotFromCanonicalAxes(aTransfoIJK);
    size_t aIndexMidColStart = mNbLine*mNbCol/2;
    tREAL8 aThetaMid = NAN;
    for (size_t i=aIndexMidColStart; i<aIndexMidColStart+mNbLine; ++i)
    {
        auto & aPtAng = mVectPtsTPD[i];
        if (aPtAng.z()<DistMinToExist())
            continue;
        else
        {
            aThetaMid = aPtAng.x();
            break;
        }
    }
    cRotation3D<tREAL8> aRotHz(cRotation3D<tREAL8>::RotKappa(aThetaMid), false);
    //  scanner     to      raster
    //  z|   y
    //   | /
    //   + --- x            + --- z
    //                      | `x
    //                     y|
    cRotation3D<tREAL8> aRotZvert2view = cRotation3D<tREAL8>::RotFromCanonicalAxes("k-i-j");

    std::cout<<"aRotZvert2view:\n"<<aRotZvert2view.AxeI()<<"\n"
              <<aRotZvert2view.AxeJ()<<"\n"<<aRotZvert2view.AxeK()<<std::endl;
    std::cout<<"aRotHz:\n"<<aRotHz.AxeI()<<"\n"
              <<aRotHz.AxeJ()<<"\n"<<aRotHz.AxeK()<<std::endl;
    std::cout<<"aRotFrame:\n"<<aRotInput2TSL.AxeI()<<"\n"
              <<aRotInput2TSL.AxeJ()<<"\n"<<aRotInput2TSL.AxeK()<<std::endl;

    mRotInput2Raster = aRotZvert2view * aRotHz * aRotInput2TSL; // TODO: use aRotFrame
    std::cout<<"mRotInput2Raster:\n"<<mRotInput2Raster.AxeI()<<"\n"
              <<mRotInput2Raster.AxeJ()<<"\n"<<mRotInput2Raster.AxeK()<<std::endl;
}


bool cStaticLidarImporter::checkLineCol()
{
    auto [aColMinIt, aColMaxIt] = std::minmax_element(mVectPtsCol.begin(), mVectPtsCol.end());
    auto [aLineMinIt, aLineMaxIt] = std::minmax_element(mVectPtsLine.begin(), mVectPtsLine.end());
    bool isOk = true;
    if ((*aColMinIt<0) || (*aColMaxIt>mNbCol-1))
        isOk = false;
    if ((*aLineMinIt<0) || (*aLineMaxIt>mNbLine-1))
        isOk = false;
    MMVII_INTERNAL_ASSERT_tiny(isOk,"Error: checkLineCol");
    mNbCol = *aColMaxIt + 1;
    mNbLine = *aLineMaxIt + 1;
    return isOk;
}


float cStaticLidarImporter::ColToLocalThetaApprox(float aCol) const
{
    return mThetaStart + aCol * mThetaStep;
}

float cStaticLidarImporter::LineToLocalPhiApprox(float aLine) const
{
    return mPhiStart + aLine * mPhiStep;
}

float cStaticLidarImporter::LocalThetaToColApprox(float aTheta) const
{
    tREAL8 aNbCol2pi = 2 * M_PI / fabs(mThetaStep);
    tREAL8 aCol = (aTheta - mThetaStart) / mThetaStep;
    // try to return to [-aNbCol2pi/2 : aNbCol2pi/2]
    if (aCol<-aNbCol2pi/2)
        return aCol+aNbCol2pi;
    if (aCol>aNbCol2pi*3/2)
        return aCol-aNbCol2pi;
    return aCol;
}

float cStaticLidarImporter::LocalPhiToLineApprox(float aPhi) const
{
    return (aPhi - mPhiStart) / mPhiStep;
}

void cStaticLidarImporter::ComputeAgregatedAngles()
{
    mVectPhisCol.resize(mNbLine, 0.);
    mVectThetasLine.resize(mNbCol, 0.);
    std::vector<int> aNbMesPhisCol(mNbLine,0);
    std::vector<int> aNbMesThetasLine(mNbCol,0);
    for (size_t i=0; i<mVectPtsTPD.size(); ++i)
    {
        if (mVectPtsTPD[i].z()<mDistMinToExist)
            continue;
        mVectPhisCol.at(mVectPtsLine[i]) += mVectPtsTPD[i].y();
        aNbMesPhisCol.at(mVectPtsLine[i])++;
        mVectThetasLine.at(mVectPtsCol[i]) += mVectPtsTPD[i].x();
        aNbMesThetasLine.at(mVectPtsCol[i])++;
    }
    // compute average (NAN for no data)
    for (int i=0; i<mNbLine; ++i)
    {
        mVectPhisCol[i] /= aNbMesPhisCol[i];
    }
    for (int i=0; i<mNbCol; ++i)
    {
        mVectThetasLine[i] /= aNbMesThetasLine[i];
    }
    // TODO: check that phis are constant among first and last lines
}

float cStaticLidarImporter::LocalPhiToLinePrecise(float aPhi) const
{
    MMVII_INTERNAL_ASSERT_tiny(!mVectPhisCol.empty(),"Error: run ComputeAgregatedAngles() before LocalPhiToLinePrecise()");

    float aLineApprox = LocalPhiToLineApprox(aPhi);
    if ((aLineApprox<0) || (aLineApprox>=mNbLine))
        return aLineApprox;
    for (int i=0; i<5;++i)
    {
        int aLineBefore = (int)aLineApprox;
        int aLineAfter = (int)aLineApprox + 1;
        if ((aLineBefore<0) || (aLineBefore>=mNbLine))
            return aLineApprox;
        if ((aLineAfter<0) || (aLineAfter>=mNbLine))
            return aLineApprox;
        float aPhiBefore = mVectPhisCol[aLineBefore];
        float aPhiAfter = mVectPhisCol[aLineAfter];
        std::cout<<"iter "<<aLineApprox<<"\n";
        aLineApprox = aLineBefore + (aPhi-aPhiBefore)/(aPhiAfter-aPhiBefore)*(aLineAfter-aLineBefore);
    }
    return aLineApprox;
}

float cStaticLidarImporter::LocalThetaToColPrecise(float aTheta) const
{
    MMVII_INTERNAL_ASSERT_tiny(!mVectThetasLine.empty(),"Error: run ComputeAgregatedAngles() before LocalThetaToColPrecise()");

    float aColApprox = LocalThetaToColApprox(aTheta);
    if ((aColApprox<0) || (aColApprox>=mNbCol))
        return aColApprox;
    for (int i=0; i<5;++i)
    {
        int aColBefore = (int)aColApprox;
        int aColAfter = (int)aColApprox + 1;
        if ((aColBefore<0) || (aColBefore>=mNbCol))
            return aColApprox;
        if ((aColAfter<0) || (aColAfter>=mNbCol))
            return aColApprox;
        float aThetaBefore = mVectThetasLine[aColBefore];
        float aThetaAfter = mVectThetasLine[aColAfter];
        aColApprox = aColBefore + (aTheta-aThetaBefore)/(aThetaAfter-aThetaBefore)*(aColAfter-aColBefore);
    }
    return aColApprox;
}


cPt2dr cStaticLidarImporter::Input3DtoRasterAngle(const cPt3dr &aPt3DInput) const
{
    std::cout<<"Input3DtoRasterAngle: "<<aPt3DInput<<"\n";
    cPt3dr aPt3DInputNorm = aPt3DInput/Norm2(aPt3DInput);
    cPt3dr aPt3DRaster = RotInput2Raster().Value(aPt3DInputNorm);
    cProj_EquiRect aProjEquiRect(M_PI);
    auto aThetaPhi = cPt2dr::FromStdVector(aProjEquiRect.Proj(aPt3DRaster.ToStdVector()));
    return aThetaPhi;
}
/*    cPt2dr aP2d_approx;
    aP2d_approx.x() = LocalThetaToColApprox(aThetaPhi.x());
    aP2d_approx.y() = LocalPhiToLineApprox(aThetaPhi.y());
    //TODO: iterative search around aP2d_approx in X/Z and Y/Z rasters
    cPt2dr aP2d = aP2d_approx;
    std::cout<<"  => "<<aThetaPhi<<"  => "<<aP2d<<"\n";
    return aP2d;
}*/


cStaticLidar::cStaticLidar(const std::string & aNameFile, const std::string & aStationName,
                           const std::string & aScanName, const tPose & aPose, cPerspCamIntrCalib * aCalib,
                           cRotation3D<tREAL8> aRotInput2Raster) :
    cSensorCamPC(aNameFile, aPose, aCalib),
    mStationName(aStationName),
    mScanName(aScanName),
    mRotInput2Raster(aRotInput2Raster)
{
}

cStaticLidar * cStaticLidar::FromFile(const std::string & aNameCalibFile, const std::string & aNameScanFile, const std::string &aNameRastersDir)
{
    cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::FromFile(aNameCalibFile);
    cStaticLidar * aRes = new cStaticLidar("NONE","?","?",tPoseR::Identity(),aCalib,tRotR::Identity());
    ReadFromFile(*aRes, aNameScanFile);
    aRes->mRasterDistance = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterDistancePath));
    aRes->mRasterIntensity = std::make_unique<cIm2D<tU_INT1>>(cIm2D<tU_INT1>::FromFile(aNameRastersDir+"/"+aRes->mRasterIntensityPath));
    aRes->mRasterMask = std::make_unique<cIm2D<tU_INT1>>(cIm2D<tU_INT1>::FromFile(aNameRastersDir+"/"+aRes->mRasterMaskPath));
    aRes->mRasterX = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterXPath));
    aRes->mRasterY = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterYPath));
    aRes->mRasterZ = std::make_unique<cIm2D<tREAL4>>(cIm2D<tREAL4>::FromFile(aNameRastersDir+"/"+aRes->mRasterZPath));
    return aRes;
}

cPt3dr cStaticLidar::Image2InputXYZ(const cPt2di & aRasterPx) const
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

cPt3dr cStaticLidar::Image2InputXYZ(const cPt2dr & aRasterPx) const
{
    auto & aRasterXData = mRasterX->DIm();
    auto & aRasterYData = mRasterY->DIm();
    auto & aRasterZData = mRasterZ->DIm();
    return cPt3dr{
        aRasterXData.GetVBL(aRasterPx),
        aRasterYData.GetVBL(aRasterPx),
        aRasterZData.GetVBL(aRasterPx),
    };
}

cPt3dr cStaticLidar::Image2Ground(const cPt2di & aRasterPx) const
{
    cPt3dr aCam3DPt = Image2Camera3D(aRasterPx);
    return Pose().Value(aCam3DPt);
}

cPt3dr cStaticLidar::Image2Ground(const cPt2dr & aRasterPx) const
{
    cPt3dr aCam3DPt = Image2Camera3D(aRasterPx);
    return Pose().Value(aCam3DPt);
}

cPt2dr cStaticLidar::Ground2ImagePrecise(const cPt3dr & aGroundPt) const
{
    //std::cout<<"  Ground2ImagePrecise for point "<<aGroundPt<<"\n";
    cPt2dr aDirCam3DTheoretical = InternalCalib()->Dir_Proj()->Value(Pose().Inverse(aGroundPt));
    //std::cout<<"  UV th: "<<aDirCam3DTheoretical<<"\n";
    cPt3dr aPtRasterApprox = this->Ground2ImageAndDepth(aGroundPt);
    cPt2dr aPtRaster = {aPtRasterApprox.x(), aPtRasterApprox.y()};

    // test if int value
    cPt2di aPtRasterRounded(round(aPtRaster.x()),round(aPtRaster.y()));
    if (Norm2(aPtRaster - cPt2dr(aPtRasterRounded.x(),aPtRasterRounded.y()))< 1e-5)
    {
        // in this case Image2Camera3D will not use GetVBL => works on first and last columns
        cPt2dr aDirTest = InternalCalib()->Dir_Proj()->Value(Image2Camera3D(aPtRasterRounded));
        if (Norm2(aDirTest - aDirCam3DTheoretical)< 1e-5)
        {
            //std::cout<<"  skip iter\n";
            return aPtRaster;
        }
    }

    // if approx is sufficient, no need for iter
    cPt2dr aDirTest = InternalCalib()->Dir_Proj()->Value(Image2Camera3D(aPtRaster));
    if (Norm2(aDirTest - aDirCam3DTheoretical)< 1e-5)
    {
        //std::cout<<"  skip iter\n";
        return aPtRaster;
    }

    for (int i = 0; i<3; ++i)
    {
        //std::cout<<"   raster: "<<aPtRaster<<"\n";
        cPt2di aPtRasterUL((int)aPtRaster.x(), (int)aPtRaster.y());
        cPt2di aPtRasterLR((int)aPtRaster.x()+1, (int)aPtRaster.y()+1);
        cPt2dr aDirUL = InternalCalib()->Dir_Proj()->Value(Image2Camera3D(aPtRasterUL));
        cPt2dr aDirLR = InternalCalib()->Dir_Proj()->Value(Image2Camera3D(aPtRasterLR));
        //std::cout<<"   Dirs: "<<aDirUL<<" "<<aDirLR<<"\n";
        float aBetterX = aPtRasterUL.x() + (aDirCam3DTheoretical.x()-aDirUL.x())/(aDirLR.x()-aDirUL.x())
                                               *(aPtRasterLR.x()-aPtRasterUL.x());
        float aBetterY = aPtRasterUL.y() + (aDirCam3DTheoretical.y()-aDirUL.y())/(aDirLR.y()-aDirUL.y())
                                               *(aPtRasterLR.y()-aPtRasterUL.y());
        aPtRaster = {aBetterX, aBetterY};
    }

    return aPtRaster;
}


void cStaticLidar::ToPly(const std::string & aName,bool useMask) const
{
    std::vector<cPt4dr> aSelectionXYZI;
    aSelectionXYZI.reserve(SzPix().x()*SzPix().y());
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    auto & aRasterIntensityData = mRasterIntensity->DIm();
    auto & aRasterXData = mRasterX->DIm();
    auto & aRasterYData = mRasterY->DIm();
    auto & aRasterZData = mRasterZ->DIm();
    for (int l = 0 ; l < SzPix().y(); ++l)
    {
        for (int c = 0 ; c < SzPix().x(); ++c)
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

template <typename TYPE> void cStaticLidar::fillRaster(const cStaticLidarImporter & aSL_importer,
                              const std::string& aPhProjDirOut, const std::string& aFileName,
                              std::function<TYPE (int)> func, std::unique_ptr<cIm2D<TYPE> > & aIm, bool saveRaster)
{
    MMVII_INTERNAL_ASSERT_tiny(aSL_importer.mVectPtsCol.size()==aSL_importer.mVectPtsXYZ.size(), "Error: Compute line/col numbers before fill raster");

    aIm.reset(new cIm2D<TYPE>(cPt2di(aSL_importer.NbCol(), aSL_importer.NbLine()), 0, eModeInitImage::eMIA_Null));
    auto & aRasterData = aIm->DIm();
    for (size_t i=0; i<aSL_importer.mVectPtsTPD.size(); ++i)
    {
        cPt2di aPcl = {aSL_importer.mVectPtsCol[i], aSL_importer.mVectPtsLine[i]};
        aRasterData.SetV(aPcl, func(i));
    }
    if (saveRaster)
        aRasterData.ToFile(aPhProjDirOut + aFileName);
}

template <typename TYPE> void cStaticLidar::fillRaster(const cStaticLidarImporter & aSL_importer,
                              const std::string& aPhProjDirOut, const std::string& aFileName,
                              std::function<TYPE (int)> func, bool saveRaster)
{
    std::unique_ptr<cIm2D<TYPE>> aIm; // temporary image
    fillRaster(aSL_importer, aPhProjDirOut, aFileName, func, aIm, saveRaster);
}

void cStaticLidar::fillRasters(const cStaticLidarImporter & aSL_importer, const std::string& aPhProjDirOut, bool saveRasters)
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

    fillRaster<tU_INT1>(aSL_importer,aPhProjDirOut, mRasterMaskPath, [&aSL_importer](int i)
                        {
                            auto aPtAng = aSL_importer.mVectPtsTPD[i];
                            return (aPtAng.z()<aSL_importer.DistMinToExist())?0:255;
                        }, mRasterMask, saveRasters);
    fillRaster<tU_INT1>(aSL_importer, aPhProjDirOut, mRasterIntensityPath, [&aSL_importer](int i){return aSL_importer.mVectPtsIntens[i]*255;}, mRasterIntensity, saveRasters );
    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterDistancePath,
                      [&aSL_importer](int i){auto aPtAng = aSL_importer.mVectPtsTPD[i];return aPtAng.z();},
                       mRasterDistance, saveRasters);

    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterXPath, [&aSL_importer](int i){auto aPtXYZ = aSL_importer.mVectPtsXYZ[i];return aPtXYZ.x();}, mRasterX, saveRasters );
    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterYPath, [&aSL_importer](int i){auto aPtXYZ = aSL_importer.mVectPtsXYZ[i];return aPtXYZ.y();}, mRasterY, saveRasters );
    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterZPath, [&aSL_importer](int i){auto aPtXYZ = aSL_importer.mVectPtsXYZ[i];return aPtXYZ.z();}, mRasterZ, saveRasters );

    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterThetaPath, [&aSL_importer](int i){auto aPtAng = aSL_importer.mVectPtsTPD[i];return aPtAng.x();}, saveRasters );
    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterPhiPath, [&aSL_importer](int i){auto aPtAng = aSL_importer.mVectPtsTPD[i];return aPtAng.y();}, saveRasters );
    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterThetaErrPath, [&aSL_importer](int i)
                      {
                          auto aPtAng = aSL_importer.mVectPtsTPD[i];
                          tREAL8 aThetaCol = aSL_importer.ThetaStart() + aSL_importer.ThetaStep() * aSL_importer.mVectPtsCol[i];
                          aThetaCol = toMinusPiPlusPi(aThetaCol);
                          return aPtAng.x()-aThetaCol;
                      }, saveRasters );
    fillRaster<tREAL4>(aSL_importer, aPhProjDirOut, mRasterPhiErrPath, [&aSL_importer](int i)
                      {
                          auto aPtAng = aSL_importer.mVectPtsTPD[i];
                          tREAL8 aPhiLine = aSL_importer.PhiStart() + aSL_importer.PhiStep() * aSL_importer.mVectPtsLine[i];
                          aPhiLine = toMinusPiPlusPi(aPhiLine);
                          return aPtAng.y()-aPhiLine;
                      }, saveRasters );

    mRasterScore.reset(new cIm2D<tREAL4>(cPt2di(aSL_importer.NbCol()+1, aSL_importer.NbLine()+1), 0, eModeInitImage::eMIA_Null));
}

void cStaticLidar::FilterIntensity(const cStaticLidarImporter &aSL_importer, tREAL8 aLowest, tREAL8 aHighest)
{
    if (!aSL_importer.HasIntensity())
        return;
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    auto & aRasterScoreData = mRasterScore->DIm();
    tREAL8 aMiddle = (aLowest + aHighest) / 2.;
    for (size_t i=0; i<aSL_importer.mVectPtsTPD.size(); ++i)
    {
        cPt2di aPcl = {aSL_importer.mVectPtsCol[i], aSL_importer.mVectPtsLine[i]};
        if ((aSL_importer.mVectPtsIntens[i]<aLowest) || (aSL_importer.mVectPtsIntens[i]>aHighest))
            aMaskImData.SetV(aPcl, 0);
        aRasterScoreData.SetV(aPcl, aRasterScoreData.GetV(aPcl) + fabs(aSL_importer.mVectPtsIntens[i]-aMiddle));
    }
    aMaskImData.ToFile("MaskIntens.png");
}

void cStaticLidar::FilterIncidence(const cStaticLidarImporter &aSL_importer, tREAL8 aAngMax)
{
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    auto & aRasterScoreData = mRasterScore->DIm();

    // TODO: use im.InitCste()
    cIm2D<tREAL4> aImDistGrX(cPt2di(aSL_importer.NbCol()+1, aSL_importer.NbLine()+1), 0, eModeInitImage::eMIA_Null);
    auto & aImDistGrXData = aImDistGrX.DIm();
    cIm2D<tREAL4> aImDistGrY(cPt2di(aSL_importer.NbCol()+1, aSL_importer.NbLine()+1), 0, eModeInitImage::eMIA_Null);
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
    for (int l = 0 ; l < aSL_importer.NbLine(); ++l)
    {
        //tREAL4 phi = lToPhiApprox(l, aSL_importer.PhiStart(), aSL_importer.PhiStep());
        tREAL4 phi = InternalCalib()->DirBundle({0.,(double)l}).y();
        tREAL4 aStepThetaFix = aSL_importer.ThetaStep()*cos(phi);
        for (int c = 0 ; c < aSL_importer.NbCol(); ++c)
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
            tREAL4 aTanIncidY = aValDistGradY / (aSL_importer.PhiStep() * aDist) / aValGaussMask;
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
    for (int l = 0 ; l < SzPix().y(); ++l)
    {
        for (int c = 0 ; c < SzPix().x(); ++c)
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

void cStaticLidar::MaskBuffer(const cStaticLidarImporter &aSL_importer, tREAL8 aAngBuffer, const std::string &aPhProjDirOut)
{
    StdOut() << "Computing Mask buffer..."<<std::endl;
    MMVII_INTERNAL_ASSERT_tiny(mRasterMask, "Error: mRasterMask must be computed first");
    auto & aMaskImData = mRasterMask->DIm();
    mRasterMaskBuffer.reset( new cIm2D<tU_INT1>(cPt2di(aSL_importer.NbCol(), aSL_importer.NbLine()), 0, eModeInitImage::eMIA_NoInit));
    auto & aMaskBufImData = mRasterMaskBuffer->DIm();

    auto & aRasterScoreData = mRasterScore->DIm();

    bool aHzLoop = false;
    if (fabs(fabs(aSL_importer.ThetaStep()) * (aSL_importer.NbCol()+1) - 2 * M_PI) < 2 * fabs(aSL_importer.ThetaStep()))
        aHzLoop = true;

    tREAL8 aRadPx = aAngBuffer/aSL_importer.PhiStep();
    aMaskBufImData.InitCste(255);

    std::vector<bool> aLinesFull(aSL_importer.NbLine()+1, false); // record lignes completely masked to pass them next time
    // int c = 100;
    // for (int l = 100; l < 2700; l += 500)
    for (int l = 0 ; l < aSL_importer.NbLine(); ++l)
    {
        for (int c = 0 ; c < aSL_importer.NbCol(); ++c)
        {
            auto aMaskVal = aMaskImData.GetV(cPt2di(c, l));
            if (aMaskVal==0)
            {
                for (int il = l - aRadPx; il <= l + aRadPx; ++il)
                {
                    if ((il<0) || (il>aSL_importer.NbLine())) continue;
                    if (aLinesFull[il]) continue;
                    //tREAL8 phi = lToPhiApprox(il, aSL_importer.PhiStart(), aSL_importer.PhiStep());
                    tREAL8 phi = InternalCalib()->DirBundle({0.,(double)l}).y();
                    tREAL8 w = fabs(sqrt(aRadPx*aRadPx - (il-l)*(il-l))/cos(phi));
                    if (w>aSL_importer.NbCol())
                    {
                        w=aSL_importer.NbCol();
                        aLinesFull[il] = true;
                        // TODO: fill line and continue
                    }
                    for (int ic = c - w; ic <= c + w; ++ic)
                    {
                        int icc = ic; // working copy
                        if (aHzLoop)
                        {
                            if (icc<0)
                                icc += (aSL_importer.NbCol()+1);
                            if (icc>aSL_importer.NbCol())
                                icc -= (aSL_importer.NbCol()+1);
                        }
                        if ((icc<0)||(icc>aSL_importer.NbCol()))
                            continue;
                        aMaskBufImData.SetV(cPt2di(icc, il), 0);
                    }
                }
            }
        }
    }
    for (int l = 0 ; l < aSL_importer.NbLine(); ++l)
        for (int c = 0 ; c < aSL_importer.NbCol(); ++c)
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
    StdOut() << "Nb patches: " << mPatchCenters.size() <<"\n";
    std::fstream file1;
    file1.open("centers.txt", std::ios_base::out);
    for (auto & aCenter : mPatchCenters)
    {
        file1 << aCenter.x() << " " << -aCenter.y() <<"\n";
    }
    aRasterScoreData.ToFile("Score.tif");
}

void cStaticLidar::SelectPatchCenters2(const cStaticLidarImporter &aSL_importer, int aNbPatches)
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
            aXStep = fabs(((float)aRasterMaskData.SzX()) / aNbPatchesX / aNbPatchesFactor / cos(aSL_importer.LineToLocalPhiApprox(aY)));
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

    StdOut() << "Nb patches: " << mPatchCenters.size() <<"\n";
    std::fstream file1;
    file1.open("centers.txt", std::ios_base::out);
    for (auto & aCenter : mPatchCenters)
    {
        file1 << aCenter.x() << " " << -aCenter.y() <<"\n";
    }
}


void cStaticLidar::MakePatches
    (std::list<std::set<cPt2di> > &aLPatches,
     std::vector<cSensorCamPC *> & aVCam,
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

    std::vector<tREAL8> aVectGndPixelSize;
    aVectGndPixelSize.resize(aVCam.size());
    // parse center points
    for (auto & aCenter: mPatchCenters)
    {
        //search for average GndPixelSize
        aVectGndPixelSize.clear();
        cPt3dr aGndCenter = Image2Ground(aCenter);
        int aNumCamVisib = 0;
        for (const auto & aCam: aVCam)
        {
            if (aCam->IsVisible(aGndCenter))
            {
                ++aNumCamVisib;
                tREAL8 aDist = Norm2(aCam->Center()-aGndCenter);
                cPt2dr aImCenter = aCam->Ground2Image(aGndCenter);

                aVectGndPixelSize.push_back(Norm2(aCam->ImageAndDepth2Ground(cPt3dr{aImCenter.x(), aImCenter.y(), aDist})
                                                  - aCam->ImageAndDepth2Ground(cPt3dr{aImCenter.x()+1., aImCenter.y(), aDist})));

            }
        }
        if (aNumCamVisib<2) continue;
        tREAL8 aGndPixelSize = NonConstMediane(aVectGndPixelSize);
        //StdOut() << "GndPixelSize: " << aGndPixelSize << "\n";

        // compute raster step to get aNbPointByPatch separated by aGndPixelSize
        tREAL4 aMeanDepth = aRasterDistData.GetV(aCenter);

        cPt3dr aCenterThetaPhiDist = Image2ThetaPhiDist(aCenter);
        tREAL4 aThetaStep = aCenterThetaPhiDist.x() - Image2ThetaPhiDist(aCenter+cPt2di(1, 0)).x();

        tREAL4 aProjColFactor = 1/cos(aCenterThetaPhiDist.y());
        tREAL4 aNbStepRadius = sqrt(aNbPointByPatch/M_PI) + 1;
        tREAL4 aRasterPxGndW = fabs(aThetaStep) * aMeanDepth * aProjColFactor;
        tREAL4 aRasterPxGndH = fabs(aThetaStep) * aMeanDepth;
        tREAL4 aRasterStepPixelsY = aGndPixelSize / aRasterPxGndH;
        tREAL4 aRasterStepPixelsX = aGndPixelSize / aRasterPxGndW;
        //StdOut() << "RasterStepPixels: " << aRasterStepPixelsX << " " << aRasterStepPixelsY << "\n";

        // have a least one scan step of difference between patch points
        if (aRasterStepPixelsX < 1.)
            aRasterStepPixelsX = 1.;
        if (aRasterStepPixelsY < 1.)
            aRasterStepPixelsY = 1.;

        std::set<cPt2di> aPatch;
        for (int aJ = -aNbStepRadius; aJ<=aNbStepRadius; ++aJ)
            for (int aI = -aNbStepRadius; aI<=aNbStepRadius; ++aI)
            {
                cPt2di aPt = aCenter + cPt2di(aI*aRasterStepPixelsX,aJ*aRasterStepPixelsY);
                if (aRasterMaskData.Inside(aPt) && aRasterMaskData.GetV(aPt))
                    aPatch.insert(aPt);
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

    MMVII::AddData(cAuxAr2007("RasterDistance",anAux),mRasterDistancePath);
    MMVII::AddData(cAuxAr2007("RasterIntensity",anAux),mRasterIntensityPath);
    MMVII::AddData(cAuxAr2007("RasterMask",anAux),mRasterMaskPath);
    MMVII::AddData(cAuxAr2007("RasterX",anAux),mRasterXPath);
    MMVII::AddData(cAuxAr2007("RasterY",anAux),mRasterYPath);
    MMVII::AddData(cAuxAr2007("RasterZ",anAux),mRasterZPath);

    MMVII::AddData(cAuxAr2007("RasterTheta",anAux),mRasterThetaPath);
    MMVII::AddData(cAuxAr2007("RasterPhi",anAux),mRasterPhiPath);
    MMVII::AddData(cAuxAr2007("RasterThetaErr",anAux),mRasterThetaErrPath);
    MMVII::AddData(cAuxAr2007("RasterPhiErr",anAux),mRasterPhiErrPath);

    MMVII::AddData(cAuxAr2007("RotInput2Raster",anAux),mRotInput2Raster);

    MMVII::AddData(cAuxAr2007("PatchCenters",anAux),mPatchCenters);
}

void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL)
{
   aSL.AddData(anAux);
}


template<typename TYPE>
void TestRaster2Gnd2Raster(const std::vector<TYPE> &aVectPtsTest, cStaticLidar * aScan)
{
    float aPrecision = 1e-2;
    if constexpr (std::is_same_v<TYPE, cPt2di>)
        aPrecision = 1e-3;
    long i=0;
    for (auto & aPIm: aVectPtsTest)
    {
        std::cout<<"Test " << i << ": "<<aPIm<<"\n";
        auto aPgnd = aScan->Image2Ground(aPIm);
        auto aPImtest = aScan->Ground2ImagePrecise(aPgnd);
        std::cout<<"Result: "<<aPIm<<" -> "<<aPgnd<<" -> "<<aPImtest<<"\n";
        ++i;
        MMVII_INTERNAL_ASSERT_bench(Norm2(cPt2dr(aPIm.x(), aPIm.y())-aPImtest)<aPrecision ,"TestRaster2Gnd2Raster: " + std::to_string(i));
    }
}

/// tests the scans of a cube, where summit is {0,0,8.66} in ground coords
void TestPose(const std::string & aInPath, const std::string & aCalibName, const std::string & aScanName, const cPt2dr& aSummitPx)
{
    cStaticLidar * aScan =  cStaticLidar::FromFile(aInPath + aCalibName, aInPath + aScanName, aInPath);
    MMVII_INTERNAL_ASSERT_bench(Norm2(aScan->Ground2ImagePrecise({0,0,8.66})-aSummitPx)<1e-3 ,"TestPose " + aScanName);
    delete aScan;
}

void BenchTSL(cParamExeBench & aParam)
{
    if (! aParam.NewBench("TSL")) return;

    const std::string & aInPath = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "/TSL/Scan1/";

    // test with scan pose = Id
    cStaticLidar * aScan =  cStaticLidar::FromFile(aInPath + "Calib-Scan-St1-Sc1.xml",
                                                   aInPath + "Scan-St1-Sc1.xml", aInPath);

    aScan->ToPly(cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() + "/TSL.ply");

    auto & pp = aScan->InternalCalib()->PP();
    cPt2di ppInt = cPt2di(round(pp.x()), round(pp.y()));
    auto & sz = aScan->InternalCalib()->SzPix();
    std::vector<cPt2di> aVectPtsTest1 = {ppInt, {0, ppInt.y()}, {sz.x()-1, ppInt.y()}, {0, 0}, {sz.x()-1, sz.y()-1}};
    TestRaster2Gnd2Raster(aVectPtsTest1, aScan);

    std::vector<cPt2dr> aVectPtsTest2;
    for (int i = 0; i<10; ++i)
        aVectPtsTest2.push_back( pp + cPt2dr(pp.x()/10. * i * cos(2*M_PI*i/10),
                                            pp.y()/10. * i * sin(2*M_PI*i/10)));
    TestRaster2Gnd2Raster(aVectPtsTest2, aScan);
    delete aScan;

    // tests with scan translation
    TestPose(aInPath, "Calib-Scan-St2-Sc1.xml", "Scan-St2-Sc1.xml", {35.5508,50});

    // tests with scan translation + rotation
    TestPose(aInPath, "Calib-Scan-St3-Sc1.xml", "Scan-St3-Sc1.xml", {40.7816,64.542});

    // just rot x
    TestPose(aInPath, "Calib-Scan-St4-Sc1.xml", "Scan-St4-Sc1.xml", {61.279,60.9406});

    // just rot xyz
    TestPose(aInPath, "Calib-Scan-St5-Sc1.xml", "Scan-St5-Sc1.xml", {67.6836,71.4344});

    //std::cout<<"Bench TSL finished."<<std::endl;
    aParam.EndBench();
    return;
}


};

