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

    cStaticLidar(const std::string &aNameImage, const tPose &aPose, cPerspCamIntrCalib *aCalib);
    long NbPts() const;

    void ToPly(const std::string & aName,bool WithOffset=false) const;
    void AddData(const  cAuxAr2007 & anAux) ;


private :
    std::string mStationName;
    std::string mScanName;
    std::string mRasterDistance;
    std::string mRasterIntensity;
    std::string mRasterMask;
    std::string mRasterX;
    std::string mRasterY;
    std::string mRasterZ;
    std::string mRasterTheta;
    std::string mRasterPhi;
    std::string mRasterThetaErr;
    std::string mRasterPhiErr;



    tREAL8 mThetaStart, mThetaStep;
    tREAL8 mPhiStart, mPhiStep;
    int mMaxCol, mMaxLine;
    cRotation3D<tREAL8> mVertRot;
};

void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL);

}

#endif  //  _MMVII_STATICLIDAR_H_
