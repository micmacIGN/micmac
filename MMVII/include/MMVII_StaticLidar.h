#ifndef  _MMVII_STATICLIDAR_H_
#define  _MMVII_STATICLIDAR_H_

#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include <unordered_set>

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
    friend class cAppli_ImportStaticScan;
public:
    cStaticLidarImporter();
    void readPlyPoints(std::string aPlyFileName);
    void readE57Points(std::string aE57FileName);
    void readPtxPoints(std::string aPtxFileName);
    bool read(const std::string & aName, bool OkNone=false, bool aForceStructured=false); //< Adapt to adequate function from postfix, return if some read suceeded

    void convertToThetaPhiDist();
    void convertToXYZ();
    void ComputeRotInstr2Raster(std::string aTransfoIJK); //< give frame used to return to primary rotation axis = z

    bool HasCartesian() const {return mHasCartesian;}
    bool HasIntensity() const {return mHasIntensity;}
    bool HasSpherical() const {return mHasSpherical;}
    bool HasRowCol() const {return mHasRowCol;}
    bool NoMiss() const {return mNoMiss;}
    bool IsStructured() const {return mIsStrucured;}
    int MaxCol() const {return mMaxCol;}
    int MaxLine() const {return mMaxLine;}
    tREAL8 ThetaStart() const {return mThetaStart;}
    tREAL8 ThetaStep() const {return mThetaStep;}
    tREAL8 PhiStart() const {return mPhiStart;}
    tREAL8 PhiStep() const {return mPhiStep;}
    tREAL8 DistMinToExist() const {return mDistMinToExist;}
    tPoseR ReadPose() const { return mReadPose;}
    const cRotation3D<tREAL8> & RotInstr2Raster() const { return mRotInstr2Raster; }

    float ColToLocalThetaApprox(float aCol) const;
    float LineToLocalPhiApprox(float aLine) const;
    float LocalThetaToColApprox(float aTheta) const;
    float LocalPhiToLineApprox(float aPhi) const;
    cPt2dr Instr3DtoRaster(const cPt3dr & aPt3DInstr) const;

    // line and col for each point
    std::vector<int> mVectPtsLine;
    std::vector<int> mVectPtsCol;
    // points
    std::vector<cPt3dr> mVectPtsXYZ;
    std::vector<tREAL8> mVectPtsIntens;
    std::vector<cPt3dr> mVectPtsTPD;
protected:
    // data
    bool mHasCartesian; //< in original read data
    bool mHasIntensity; //< in original read data
    bool mHasSpherical; //< in original read data
    bool mHasRowCol;    //< in original read data

    bool mNoMiss; // seems to be full
    bool mIsStrucured;
    tPoseR mReadPose;
    tREAL8 mDistMinToExist;

    int mMaxCol, mMaxLine;
    tREAL8 mThetaStart, mThetaStep;
    tREAL8 mPhiStart, mPhiStep;
    cRotation3D<tREAL8> mVertRot; //< verticalizarion rotation in cloud frame
    cRotation3D<tREAL8> mRotInstr2Raster; //< to go from z vertical to z view direction of PP, and make PPx in center
};

class cStaticLidar: public cSensorCamPC
{
    friend class cAppli_ImportStaticScan;
public :

    cStaticLidar(const std::string &aNameFile, const std::string & aStationName,
                 const std::string & aScanName, const tPose &aPose, cPerspCamIntrCalib *aCalib);

    static cStaticLidar *FromFile(const std::string & aNameCalibFile, const std::string &aNameScanFile, const std::string & aNameRastersDir);

    void ToPly(const std::string & aName, bool useMask=false) const;
    void AddData(const  cAuxAr2007 & anAux) ;

    void fillRasters(const cStaticLidarImporter & aSL_importer, const std::string &aPhProjDirOut, bool saveRasters);

    //inline tREAL8 lToPhiApprox(int l, double aPhiStart, double aPhiStep) const { return aPhiStart + l * aPhiStep; }
    //inline tREAL8 cToThetaApprox(int c, double aThetaStart, double aThetaStep) const { return aThetaStart + c * aThetaStep; }

    void FilterIntensity(const cStaticLidarImporter & aSL_importer, tREAL8 aLowest, tREAL8 aHighest); // add to mRasterMask
    void FilterIncidence(const cStaticLidarImporter &aSL_importer, tREAL8 aAngMax);
    void FilterDistance(tREAL8 aDistMin, tREAL8 aDistMax);
    void MaskBuffer(const cStaticLidarImporter &aSL_importer, tREAL8 aAngBuffer, const std::string &aPhProjDirOut);
    void SelectPatchCenters1(int aNbPatches);
    void SelectPatchCenters2(const cStaticLidarImporter &aSL_importer, int aNbPatches);
    void MakePatches(const cStaticLidarImporter &aSL_importer,   //TODO: remove aSL_importer, use calib!
                     std::list<std::set<cPt2di> > &aLPatches,
                     std::vector<cSensorCamPC *> &aVCam, int aNbPointByPatch, int aSzMin) const;

    cPt3dr Image2Instr3D(const cPt2di & aRasterPx) const;
    cPt3dr Image2Instr3D(const cPt2dr & aRasterPx) const;
    cPt3dr Image2Ground(const cPt2di & aRasterPx) const;
    cPt3dr Image2Ground(const cPt2dr & aRasterPx) const;

private :
    template <typename TYPE> void fillRaster(const cStaticLidarImporter & aSL_importer, const std::string& aPhProjDirOut, const std::string& aFileName,
                    std::function<TYPE (int)> func, bool saveRaster); // do not keep image in memory

    template <typename TYPE> void fillRaster(const cStaticLidarImporter & aSL_importer, const std::string& aPhProjDirOut, const std::string& aFileName,
                    std::function<TYPE (int)> func, std::unique_ptr<cIm2D<TYPE>> & aIm, bool saveRaster); // keep image in memory

    std::string mStationName;
    std::string mScanName;
    std::string mRasterDistancePath;
    std::unique_ptr<cIm2D<tREAL4>> mRasterDistance;
    std::string mRasterIntensityPath;
    std::unique_ptr<cIm2D<tU_INT1>> mRasterIntensity;
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

    std::vector<cPt2di> mPatchCenters;

    // rasters for filtering
    std::unique_ptr<cIm2D<tU_INT1>> mRasterMaskBuffer;
    std::unique_ptr<cIm2D<tREAL4>> mRasterScore; // updated on each filter, used to find patch centers. High=bad
};

void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL);

}

#endif  //  _MMVII_STATICLIDAR_H_
