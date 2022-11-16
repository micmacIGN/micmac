#include "MMVII_nums.h"
#include "MMVII_Error.h"
#include "MMVII_Sensor.h"

/**
   \file cMetaDataImages.cpp

   \brief file for handling meta data
*/


namespace MMVII
{


tREAL8  cMedaDataImage::Aperture() const
{
   MMVII_INTERNAL_ASSERT_User(mAperture>0,eTyUEr::eNoAperture,"Aperture is not init for " + mNameImage);
   return mAperture;
}

cMedaDataImage::cMedaDataImage(const std::string & aNameIm) :
   cMedaDataImage()
{
     mNameImage    = aNameIm;
     mAperture = 11.0;  
     MMVII_WARGNING("cMedaDataImage : quick and (VERY) dirty implementation, most probably wrong");
}

cMedaDataImage::cMedaDataImage() :
    mCameraName       (""),
    mAperture         (-1),
    mFocalMM          (-1),
    mFocalMMEqui35    (-1)
{
}


}; // MMVII

