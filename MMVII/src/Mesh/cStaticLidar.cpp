#include "MMVII_StaticLidar.h"

#include "../Mesh/happly.h"
#include "E57SimpleReader.h"

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
    mNoMiss(false), mDistMinToExist(1e-5)
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
    mNoMiss = aNbPtsNul>0;
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
    mNoMiss = aNbPtsNul>0;
}









cStaticLidar::cStaticLidar(const std::string & aNameImage,const tPose & aPose,cPerspCamIntrCalib * aCalib) :
    cSensorCamPC(aNameImage, aPose, aCalib),
    mThetaStart       (NAN),
    mThetaStep         (NAN),
    mPhiStart         (NAN),
    mPhiStep           (NAN)
{
}

long cStaticLidar::NbPts() const
{
    return mMaxCol * mMaxLine;
}

void cStaticLidar::ToPly(const std::string & aName,bool WithOffset) const
{
/*    cMMVII_Ofs anOfs(aName,eFileModeOut::CreateText);

    size_t aNbP = NbPts();
    size_t aNbC = mColors.size(); 
    bool  WithVis = (mMulDegVis>0);
    if (WithVis)
    {
        MMVII_INTERNAL_ASSERT_always(aNbC==0,"Colors & DegVis ...");
        aNbC=1;
    }
    // with use 8 byte if initially 8 byte, or if we use the offset that creat big coord
    bool  aMode8 =  mMode8 || WithOffset;

    std::string aSpecCoord = aMode8 ? "float64" : "float32";
    anOfs.Ofs() <<  "ply\n";
    anOfs.Ofs() <<  "format ascii 1.0\n";
    anOfs.Ofs() <<  "comment Generated by MMVVI\n";
    anOfs.Ofs() <<  "element vertex " << aNbP << "\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" x\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" y\n";
    anOfs.Ofs() <<  "property " <<  aSpecCoord  <<" z\n";
    if (aNbC) 
    {
        anOfs.Ofs() <<  "property uchar red\n"; 
        anOfs.Ofs() <<  "property uchar green\n"; 
        anOfs.Ofs() <<  "property uchar blue\n"; 
    }
    anOfs.Ofs() <<  "end_header\n";


    for (size_t aKPt=0 ; aKPt<aNbP ; aKPt++)
    {
        if (aMode8)
        {
            cPt3dr aPt = WithOffset ? KthPt(aKPt) :  KthPtWoOffs(aKPt);
            anOfs.Ofs() <<  aPt.x() << " " << aPt.y() << " " << aPt.z();
        }
        else
        {
            const cPt3df&  aPt = mPtsF.at(aKPt);
            anOfs.Ofs() <<  aPt.x() << " " << aPt.y() << " " << aPt.z();
        }
        if (aNbC)
        {
           if (aNbC==1)
           {
              size_t aC =  WithVis ? round_ni(GetDegVis(aKPt) *255)  : mColors.at(0).at(aKPt);
              anOfs.Ofs() << " " << aC << " " << aC << " " << aC;
           }
           else if (aNbC==3)
           {
               for (size_t aKC=0 ; aKC<aNbC ; aKC++)
                  anOfs.Ofs() << " " << (size_t)  mColors.at(aKC).at(aKPt);
           }
           else 
           {
               MMVII_INTERNAL_ERROR("Bad number of channel in ply generate : " + ToStr(aNbC));
           }
        }
        anOfs.Ofs() << "\n";
    }*/
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

    MMVII::AddData(cAuxAr2007("RasterDistance",anAux),mRasterDistance);
    MMVII::AddData(cAuxAr2007("RasterIntensity",anAux),mRasterIntensity);
    MMVII::AddData(cAuxAr2007("RasterMask",anAux),mRasterMask);
    MMVII::AddData(cAuxAr2007("RasterX",anAux),mRasterX);
    MMVII::AddData(cAuxAr2007("RasterY",anAux),mRasterY);
    MMVII::AddData(cAuxAr2007("RasterZ",anAux),mRasterZ);

    MMVII::AddData(cAuxAr2007("RasterTheta",anAux),mRasterTheta);
    MMVII::AddData(cAuxAr2007("RasterPhi",anAux),mRasterPhi);
    MMVII::AddData(cAuxAr2007("RasterThetaErr",anAux),mRasterThetaErr);
    MMVII::AddData(cAuxAr2007("RasterPhiErr",anAux),mRasterPhiErr);
}

void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL)
{
   aSL.AddData(anAux);
}

};

