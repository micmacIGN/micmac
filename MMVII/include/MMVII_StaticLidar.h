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
    bool read(const std::string & aName, bool OkNone=false, bool aForceStructured=false, std::string aStrInput2TSL="ijk"); //< Adapt to adequate function from postfix, return if some read suceeded

    void convertToThetaPhiDist();
    void convertToXYZ();
    void ComputeRotInput2Raster(std::string aTransfoIJK); //< give frame used to return to primary rotation axis = z

    bool HasCartesian() const {return mHasCartesian;}
    bool HasIntensity() const {return mHasIntensity;}
    bool HasSpherical() const {return mHasSpherical;}
    bool HasRowCol() const {return mHasRowCol;}
    bool NoMiss() const {return mNoMiss;}
    bool IsStructured() const {return mIsStrucured;}
    int NbCol() const {return mNbCol;}
    int NbLine() const {return mNbLine;}
    tREAL8 ThetaStart() const {return mThetaStart;}
    tREAL8 ThetaStep() const {return mThetaStep;}
    tREAL8 PhiStart() const {return mPhiStart;}
    tREAL8 PhiStep() const {return mPhiStep;}
    tREAL8 DistMinToExist() const {return mDistMinToExist;}
    tPoseR ReadPose() const { return mReadPose;}
    bool checkLineCol(); // verify that mMaxCol/mMaxLine ar compatible with mVectPtsLine/mVectPtsCol
    const cRotation3D<tREAL8> & RotInput2TSL() const { return mRotInput2TSL; }
    const cRotation3D<tREAL8> & RotInput2Raster() const { return mRotInput2Raster; }

    float ColToLocalThetaApprox(float aCol) const;
    float LineToLocalPhiApprox(float aLine) const;
    float LocalThetaToColApprox(float aTheta) const;
    float LocalPhiToLineApprox(float aPhi) const;
    void ComputeAgregatedAngles();
    float LocalPhiToLinePrecise(float aPhi) const;
    float LocalThetaToColPrecise(float aTheta) const;
    cPt2dr Input3DtoRasterAngle(const cPt3dr & aPt3DInput) const;

    // line and col for each point
    std::vector<int> mVectPtsLine;
    std::vector<int> mVectPtsCol;
    // points
    std::vector<cPt3dr> mVectPtsXYZ;
    std::vector<tREAL8> mVectPtsIntens;
    std::vector<cPt3dr> mVectPtsTPD;

    // agregated angles per col/line
    std::vector<tREAL8> mVectPhisCol;
    std::vector<tREAL8> mVectThetasLine;

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

    int mNbCol, mNbLine;
    tREAL8 mThetaStart, mThetaStep;
    tREAL8 mPhiStart, mPhiStep;
    cRotation3D<tREAL8> mRotInput2TSL; // from xyz input file to classical TSL (rot around z)
    cRotation3D<tREAL8> mVertRot; //< verticalizarion rotation in cloud frame
    cRotation3D<tREAL8> mRotInput2Raster; //< to go from z vertical to z view direction of PP, and make PPx in center
};

class cStaticLidar: public cSensorCamPC
{
    friend class cAppli_ImportStaticScan;
    friend class cStaticLidarImporter;
public :

    cStaticLidar(const std::string &aNameFile, const std::string & aStationName,
                 const std::string & aScanName, const tPose &aPose, cPerspCamIntrCalib *aCalib,
                 cRotation3D<tREAL8> aRotInput2Raster);

    static cStaticLidar *FromFile(const std::string & aNameScanFile, const std::string & aNameRastersDir="");

    bool AreRastersReady() const { return mAreRastersReady;}
    void ToPly(const std::string & aName, bool useMask=false) const;
    void AddData(const  cAuxAr2007 & anAux) ;
    virtual void ToFile(const std::string &) const override;
    void fillRasters(const cStaticLidarImporter & aSL_importer, const std::string &aPhProjDirOut, bool saveRasters);

    //inline tREAL8 lToPhiApprox(int l, double aPhiStart, double aPhiStep) const { return aPhiStart + l * aPhiStep; }
    //inline tREAL8 cToThetaApprox(int c, double aThetaStart, double aThetaStep) const { return aThetaStart + c * aThetaStep; }

    void FilterIntensity(const cStaticLidarImporter & aSL_importer, tREAL8 aLowest, tREAL8 aHighest); // add to mRasterMask
    void FilterIncidence(const cStaticLidarImporter &aSL_importer, tREAL8 aAngMax);
    void FilterDistance(tREAL8 aDistMin, tREAL8 aDistMax);
    void MaskBuffer(const cStaticLidarImporter &aSL_importer, tREAL8 aAngBuffer, const std::string &aPhProjDirOut);
    void SelectPatchCenters1(int aNbPatches);
    void SelectPatchCenters2(const cStaticLidarImporter &aSL_importer, int aNbPatches);
    void MakePatches(std::list<std::set<cPt2di> > &aLPatches,
                     const std::vector<cSensorCamPC *> &aVCam, int aNbPointByPatch, int aSzMin) const;

    cPt3dr Image2InputXYZ(const cPt2di & aRasterPx) const; // in input frame
    cPt3dr Image2InputXYZ(const cPt2dr & aRasterPx) const;

    template <typename TYPE>
    cPt3dr Image2Camera3D(const TYPE & aRasterPx) const; // in sensor frame (Z forward)

    template <typename TYPE>
        cPt3dr Image2ThetaPhiDist(const TYPE & aRasterPx) const;

    cPt3dr Image2Ground(const cPt2di & aRasterPx) const;
    cPt3dr Image2Ground(const cPt2dr & aRasterPx) const;

    cPt2dr Ground2ImagePrecise(const cPt3dr & aGroundPt) const;

    static std::string  PrefixName() ;
    std::string  V_PrefixName() const override;
    static std::string Pat2Sup(const std::string & aPatSelect);

    cDataIm2D<tREAL4> &getRasterDistance() const;
    bool IsValidPoint(const cPt2dr &aRasterPx) const;


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

    bool mAreRastersReady;

    std::vector<cPt2di> mPatchCenters;

    // rasters for filtering
    std::unique_ptr<cIm2D<tU_INT1>> mRasterMaskBuffer;
    std::unique_ptr<cIm2D<tREAL4>> mRasterScore; // updated on each filter, used to find patch centers. High=bad

    cRotation3D<tREAL8> mRotInput2Raster; //< to go from z vertical to z view direction of PP, and make PPx in center
};

template <typename TYPE>
cPt3dr cStaticLidar::Image2Camera3D(const TYPE & aRasterPx) const
{
    cPt3dr aPtInput3D = Image2InputXYZ(aRasterPx);
    cPt3dr aPtCam3D = mRotInput2Raster.Value(aPtInput3D);
    //std::cout<<"   Image > TLS > Camera3D: " << aRasterPx << " => "<< aPtTLS3D <<" => "<< aPtCam3D <<"\n";
    return aPtCam3D;
}

template <typename TYPE>
    cPt3dr cStaticLidar::Image2ThetaPhiDist(const TYPE & aRasterPx) const
{
    cPt3dr aPtCam3D = Image2Camera3D(aRasterPx);
    tREAL8 aDist = Norm2(aPtCam3D);
    cPt2dr aDir = InternalCalib()->Value(aPtCam3D);
    return {aDir.x(), aDir.y(), aDist};
}
void AddData(const  cAuxAr2007 & anAux,cStaticLidar & aSL);

}

#endif  //  _MMVII_STATICLIDAR_H_
