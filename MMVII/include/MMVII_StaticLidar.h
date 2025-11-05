#ifndef  _MMVII_STATICLIDAR_H_
#define  _MMVII_STATICLIDAR_H_

#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"

namespace MMVII
{

/** \file MMVII_StaticLidar.h
    \brief Static lidar internal representation



*/


cPt3dr cart2spher(const cPt3dr & aPtCart);
cPt3dr spher2cart(const cPt3dr & aPtspher);
tREAL8 toMinusPiPlusPi(tREAL8 aAng, tREAL8 aOffset = 0.);


class cStaticLidarImporter
{
public:
    cStaticLidarImporter();
    void readPlyPoints(std::string aPlyFileName);
    void readE57Points(std::string aE57FileName);
    bool read(const std::string & aName,bool OkNone=false); //< Addapt to adequate function from postfix, return is some read suceeded

    void convertToThetaPhiDist();
    void convertToXYZ();

    bool HasCartesian(){ return mHasCartesian;}
    bool HasIntensity(){ return mHasIntensity;}
    bool HasSpherical(){ return mHasSpherical;}
    bool HasRowCol(){ return mHasRowCol;}
    bool NoMiss(){ return mNoMiss;}
    int MaxCol(){ return mMaxCol;}
    int MaxLine(){ return mMaxLine;}
    tREAL8 DistMinToExist(){ return mDistMinToExist;}

    // line and col for each point
    std::vector<int> mVectPtsLine;
    std::vector<int> mVectPtsCol;
    // points
    std::vector<cPt3dr> mVectPtsXYZ;
    std::vector<tREAL8> mVectPtsIntens;
    std::vector<cPt3dr> mVectPtsTPD;
protected:
    // data
    bool mHasCartesian; // in original read data
    bool mHasIntensity; // in original read data
    bool mHasSpherical; // in original read data
    bool mHasRowCol;    // in original read data
    int mMaxCol, mMaxLine;

    bool mNoMiss; // seems to be full
    tREAL8 mDistMinToExist;
};

class cStaticLidar: public cSensorCamPC
{
    friend class cAppli_ImportStaticScan;
public :

    cStaticLidar(const std::string &aNameFile, const tPose &aPose, cPerspCamIntrCalib *aCalib);

    static cStaticLidar *FromFile(const std::string & aNameFile, const std::string & aNameRastersDir);

    long NbPts() const;

    void ToPly(const std::string & aName, bool WithOffset=false) const;
    void AddData(const  cAuxAr2007 & anAux) ;

    void fillRasters(const std::string &aPhProjDirOut, bool saveRasters);

    inline tREAL8 lToPhiApprox(int l) const { return mPhiStart + l * mPhiStep; }
    inline tREAL8 cToThetaApprox(int c) const { return mThetaStart + c * mThetaStep; }

    void FilterIntensity(tREAL8 aLowest, tREAL8 aHighest); // add to mRasterMask
    void FilterIncidence(tREAL8 aAngMax);
    void FilterDistance(tREAL8 aDistMin, tREAL8 aDistMax);
    void MaskBuffer(tREAL8 aAngBuffer, const std::string &aPhProjDirOut);
    void SelectPatchCenters1(int aNbPatches);
    void SelectPatchCenters2(int aNbPatches);
    void MakePatches(std::list<std::vector<cPt2di> > & aLPatches,
                     tREAL8 aGndPixelSize, int aNbPointByPatch, int aSzMin) const;

    float ColToLocalThetaApprox(float aCol) const;
    float LineToLocalPhiApprox(float aLine) const;

    cPt3dr to3D(cPt2di aRasterPx) const;


    cStaticLidarImporter mSL_importer;
private :
    template <typename TYPE> void fillRaster(const std::string& aPhProjDirOut, const std::string& aFileName,
                    std::function<TYPE (int)> func, bool saveRaster); // do not keep image in memory

    template <typename TYPE> void fillRaster(const std::string& aPhProjDirOut, const std::string& aFileName,
                    std::function<TYPE (int)> func, std::unique_ptr<cIm2D<TYPE>> & aIm, bool saveRaster); // keep image in memory

    std::string mStationName;
    std::string mScanName;
    std::string mRasterDistancePath;
    std::unique_ptr<cIm2D<tREAL4>> mRasterDistance;
    std::string mRasterIntensityPath;
    std::string mRasterMaskPath;
    std::unique_ptr<cIm2D<tU_INT1>> mRasterMask;
    std::string mRasterXPath;
    std::unique_ptr<cIm2D<tREAL4>> mRasterX;
    std::string mRasterYPath;
    std::unique_ptr<cIm2D<tREAL4>> mRasterY;
    std::string mRasterZPath;
    std::unique_ptr<cIm2D<tREAL4>> mRasterZ;
    std::string mRasterThetaPath;
    std::string mRasterPhiPath;
    std::string mRasterThetaErrPath;
    std::string mRasterPhiErrPath;


    tREAL8 mThetaStart, mThetaStep;
    tREAL8 mPhiStart, mPhiStep;
    int mMaxCol, mMaxLine;
    cRotation3D<tREAL8> mVertRot;
    std::vector<cPt2di> mPatchCenters;

    // rasters for filtering
    std::unique_ptr<cIm2D<tU_INT1>> mRasterMaskBuffer;
    std::unique_ptr<cIm2D<tREAL4>> mRasterScore; // updated on each filter, used to find patch centers. High=bad
};

void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL);

}

#endif  //  _MMVII_STATICLIDAR_H_
