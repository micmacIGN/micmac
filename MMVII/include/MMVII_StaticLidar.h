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
