#include "MMVII_StaticLidar.h"

#include <functional>

#include "../Mesh/happly.h"
#include "E57SimpleReader.h"
#include "MMVII_TplGradImFilter.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_ImageInfoExtract.h"

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

bool cStaticLidarImporter::read(const std::string & aName,bool OkNone)
{
    std::string aPost = LastPostfix(aName);
    if (UCaseEqual(aPost,"ply"))
       readPlyPoints(aName);
    else if (UCaseEqual(aPost,"e57"))
       readE57Points(aName);
    else
    {
        if (! OkNone)
        {
           MMVII_UnclasseUsEr("Cannot read cloud for " + aName);
        }
        return false;
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

float cStaticLidar::ColToLocalThetaApprox(float aCol) const
{
    return mThetaStart + aCol * mThetaStep;
}

float cStaticLidar::LineToLocalPhiApprox(float aLine) const
{
    return mPhiStart + aLine * mPhiStep;
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

template <typename TYPE> void cStaticLidar::fillRaster(const std::string& aPhProjDirOut, const std::string& aFileName,
                              std::function<TYPE (int)> func, std::unique_ptr<cIm2D<TYPE> > & aIm)
{
    MMVII_INTERNAL_ASSERT_tiny(mSL_importer.mVectPtsCol.size()==mSL_importer.mVectPtsXYZ.size(), "Error: Compute line/col numbers before fill raster");

    aIm.reset(new cIm2D<TYPE>(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null));
    auto & aRasterData = aIm->DIm();
    for (size_t i=0; i<mSL_importer.mVectPtsTPD.size(); ++i)
    {
        cPt2di aPcl = {mSL_importer.mVectPtsCol[i], mSL_importer.mVectPtsLine[i]};
        aRasterData.SetV(aPcl, func(i));
    }
    aRasterData.ToFile(aPhProjDirOut + aFileName);
}

template <typename TYPE> void cStaticLidar::fillRaster(const std::string& aPhProjDirOut, const std::string& aFileName,
                              std::function<TYPE (int)> func)
{
    std::unique_ptr<cIm2D<TYPE>> aIm; // temporary image
    fillRaster(aPhProjDirOut, aFileName, func, aIm);
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
                        }, mRasterMask);
    if (mSL_importer.HasIntensity())
        fillRaster<tU_INT1>(aPhProjDirOut, mRasterIntensityPath, [this](int i){return mSL_importer.mVectPtsIntens[i]*255;} );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterDistancePath,
                      [this](int i){auto aPtAng = mSL_importer.mVectPtsTPD[i];return aPtAng.z();},
                       mRasterDistance );

    fillRaster<tREAL4>(aPhProjDirOut, mRasterXPath, [this](int i){auto aPtXYZ = mSL_importer.mVectPtsXYZ[i];return aPtXYZ.x();} );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterYPath, [this](int i){auto aPtXYZ = mSL_importer.mVectPtsXYZ[i];return aPtXYZ.y();} );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterZPath, [this](int i){auto aPtXYZ = mSL_importer.mVectPtsXYZ[i];return aPtXYZ.z();} );

    fillRaster<tREAL4>(aPhProjDirOut, mRasterThetaPath, [this](int i){auto aPtAng = mSL_importer.mVectPtsTPD[i];return aPtAng.x();} );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterPhiPath, [this](int i){auto aPtAng = mSL_importer.mVectPtsTPD[i];return aPtAng.y();} );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterThetaErrPath, [this](int i)
                      {
                          auto aPtAng = mSL_importer.mVectPtsTPD[i];
                          tREAL8 aThetaCol = mThetaStart + mThetaStep * mSL_importer.mVectPtsCol[i];
                          aThetaCol = toMinusPiPlusPi(aThetaCol);
                          return aPtAng.x()-aThetaCol;
                      } );
    fillRaster<tREAL4>(aPhProjDirOut, mRasterPhiErrPath, [this](int i)
                      {
                          auto aPtAng = mSL_importer.mVectPtsTPD[i];
                          tREAL8 aPhiLine = mPhiStart + mPhiStep * mSL_importer.mVectPtsLine[i];
                          aPhiLine = toMinusPiPlusPi(aPhiLine);
                          return aPtAng.y()-aPhiLine;
                      } );

    mRasterScore.reset(new cIm2D<tREAL4>(cPt2di(mMaxCol+1, mMaxLine+1), 0, eModeInitImage::eMIA_Null));
}

void cStaticLidar::FilterIntensity(tREAL8 aLowest, tREAL8 aHighest)
{
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
            aImDistGrXData.SetV(cPt2di(c, l), aTanIncidX);
            tREAL4 aTanIncidY = aValDistGradY / (mPhiStep * aDist) / aValGaussMask;
            aImDistGrYData.SetV(cPt2di(c, l), aTanIncidY);
            if (fabs(aTanIncidX*aTanIncidX+aTanIncidY*aTanIncidY)>aTanAngMax*aTanAngMax)
                aMaskImData.SetV(cPt2di(c, l), 0);
            aRasterScoreData.SetV(cPt2di(c, l), aRasterScoreData.GetV(cPt2di(c, l)) + 10.*fabs(aTanIncidX*aTanIncidX+aTanIncidY*aTanIncidY));
        }
    }
    aImDistGrXData.ToFile("DistGrXData.tif");
    aImDistGrYData.ToFile("DistGrYData.tif");
    aMaskImData.ToFile("MaskIncidence.png");
}

void cStaticLidar::MaskBuffer(tREAL8 aAngBuffer)
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
    auto & aRasterMaskBufferData = mRasterMaskBuffer->DIm();
    /*cResultExtremum aRes;
    double aRadius = sqrt(aRasterMaskBufferData.SzX()*aRasterMaskBufferData.SzY()/(M_PI*nbPatches));
    ExtractExtremum1(aRasterMaskBufferData, aRes, aRadius);
    mPatchCenters = aRes.mPtsMax;*/

    // regular grid
    float aAvgDist = 3.;
    auto & aRasterDistData = mRasterDistance->DIm();
    float aXYratio=((float)aRasterMaskBufferData.SzX())/aRasterMaskBufferData.SzY();
    int aNbPatchesX = sqrt((double)aNbPatches)*sqrt(aXYratio)+1;
    int aNbPatchesY = sqrt((double)aNbPatches)/sqrt(aXYratio)+1;

    float aNbPatchesFactor = 1.5; // a priori search for aNbPatches * aNbPatchesFactor
    float aX;
    float aY = float(aRasterMaskBufferData.SzY()) / aNbPatchesY / 2.;
    float aXStep;
    float aYStep = aRasterMaskBufferData.SzY() / aNbPatchesY / aNbPatchesFactor;

    int aLineCounter = 0;
    while (aY<aRasterMaskBufferData.SzY())
    {
        aX = float(aRasterMaskBufferData.SzX()) / aNbPatchesX * ((aLineCounter%2)?1./3.:2./3.);
        while (aX<aRasterMaskBufferData.SzX())
        {
            // take lat/long proj into account
            aXStep = ((float)aRasterMaskBufferData.SzX()) / aNbPatchesX / aNbPatchesFactor / cos(LineToLocalPhiApprox(aY));
            auto aPt = cPt2di(aX, aY);
            if (aRasterMaskBufferData.GetV(aPt))
            {
                mPatchCenters.push_back(aPt);
                aXStep *= aAvgDist/aRasterDistData.GetV(aPt); // take depth into account
            } else
                aXStep /= 3.;
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

