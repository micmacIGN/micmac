#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"

/**
   \file  cPhotogrammetricProjectMemory.cpp

   \brief implementation of cPhotogrammetricProjectMemory: an in-memory
          implementation of cIPhProj that stores calibrations, sensor poses
          and homologous/multiple tie-points in std::map containers instead
          of reading them from disk.
*/

namespace MMVII
{

/* **************************************** */
/*                                          */
/*     cPhotogrammetricProjectMemory        */
/*                                          */
/* **************************************** */

cPhotogrammetricProjectMemory::cPhotogrammetricProjectMemory()
{
}

cPhotogrammetricProjectMemory::~cPhotogrammetricProjectMemory()
{
}

// ==================  Population methods  ==================

void  cPhotogrammetricProjectMemory::AddCalib(const std::string & aNameIm, cPerspCamIntrCalib * aCalib)
{
    mCalibMap[aNameIm] = aCalib;
}

void  cPhotogrammetricProjectMemory::AddSensor(const std::string & aNameIm, cSensorCamPC * aSensor)
{
    mSensorMap[aNameIm] = aSensor;
}

void  cPhotogrammetricProjectMemory::AddHomol(const std::string & aNameIm1,
                                              const std::string & aNameIm2,
                                              const cSetHomogCpleIm & aSet)
{
    mHomolMap[{aNameIm1, aNameIm2}] = aSet;
}

void  cPhotogrammetricProjectMemory::AddMulTieP(const std::string & aNameIm,
                                                const cVecTiePMul & aVec)
{
    mMulTiePMap[aNameIm] = aVec;
}

// ==================  cIPhProj interface  ==================

cPerspCamIntrCalib *  cPhotogrammetricProjectMemory::InternalCalibFromStdName(const std::string aNameIm,
                                                                               bool /*isRemanent*/) const
{
    auto aIt = mCalibMap.find(aNameIm);
    if (aIt == mCalibMap.end())
        MMVII_UserError(eTyUEr::eUnClassedError, "cPhotogrammetricProjectMemory: no calib for image " + aNameIm);
    return aIt->second;
}

cSensorCamPC *  cPhotogrammetricProjectMemory::ReadCamPC(const std::string & aNameIm,
                                                          bool /*ToDeleteAutom*/,
                                                          bool SVP) const
{
    auto aIt = mSensorMap.find(aNameIm);
    if (aIt == mSensorMap.end())
    {
        if (SVP) return nullptr;
        MMVII_UserError(eTyUEr::eUnClassedError, "cPhotogrammetricProjectMemory: no sensor for image " + aNameIm);
    }
    return aIt->second;
}

cSensorImage *  cPhotogrammetricProjectMemory::ReadSensor(const std::string & aNameIm,
                                                           bool ToDeleteAutom,
                                                           bool SVP) const
{
    return ReadCamPC(aNameIm, ToDeleteAutom, SVP);
}

tPoseR  cPhotogrammetricProjectMemory::ReadPoseCamPC(const std::string & aNameIm, bool * IsOk) const
{
    cSensorCamPC * aCamPC = ReadCamPC(aNameIm, false, IsOk != nullptr);
    if (IsOk)
    {
        *IsOk = (aCamPC != nullptr);
        if (!*IsOk)
            return tPoseR::RandomIsom3D(10);
    }
    return aCamPC->Pose();
}

void  cPhotogrammetricProjectMemory::SaveSensor(const cSensorImage &) const
{
    MMVII_UserError(eTyUEr::eUnClassedError, "cPhotogrammetricProjectMemory::SaveSensor: not supported in memory mode");
}

void  cPhotogrammetricProjectMemory::SaveCamPC(const cSensorCamPC &) const
{
    MMVII_UserError(eTyUEr::eUnClassedError, "cPhotogrammetricProjectMemory::SaveCamPC: not supported in memory mode");
}

void  cPhotogrammetricProjectMemory::SaveCalibPC(const cPerspCamIntrCalib &) const
{
    MMVII_UserError(eTyUEr::eUnClassedError, "cPhotogrammetricProjectMemory::SaveCalibPC: not supported in memory mode");
}

void  cPhotogrammetricProjectMemory::ReadHomol(cSetHomogCpleIm & aSet,
                                               const std::string & aNameIm1,
                                               const std::string & aNameIm2,
                                               const std::string & /*aDir*/) const
{
    auto aIt = mHomolMap.find({aNameIm1, aNameIm2});
    if (aIt == mHomolMap.end())
        MMVII_UserError(eTyUEr::eUnClassedError,
                        "cPhotogrammetricProjectMemory: no homol for pair " + aNameIm1 + "/" + aNameIm2);
    aSet = aIt->second;
}

void  cPhotogrammetricProjectMemory::SaveHomol(const cSetHomogCpleIm &,
                                               const std::string &,
                                               const std::string &,
                                               const std::string &) const
{
    MMVII_UserError(eTyUEr::eUnClassedError, "cPhotogrammetricProjectMemory::SaveHomol: not supported in memory mode");
}

void  cPhotogrammetricProjectMemory::ReadMultipleTiePFromFolder(const std::string & /*aFolder*/,
                                                                 cVecTiePMul & aVPm,
                                                                 const std::string & aNameIm,
                                                                 bool SVP) const
{
    auto aIt = mMulTiePMap.find(aNameIm);
    if (aIt == mMulTiePMap.end())
    {
        if (SVP) return;
        MMVII_UserError(eTyUEr::eUnClassedError,
                        "cPhotogrammetricProjectMemory: no multiple tie-points for image " + aNameIm);
    }
    aVPm = aIt->second;
}


std::string  cPhotogrammetricProjectMemory::MulTiePDirIn() const
{
    return "";  // no disk folder; folder argument is ignored in ReadMultipleTiePFromFolder
}

};  //  namespace MMVII
