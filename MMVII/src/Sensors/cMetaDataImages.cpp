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

tREAL8  cMedaDataImage::FocalMM() const
{
   MMVII_INTERNAL_ASSERT_User(mFocalMM>0,eTyUEr::eNoFocale,"Focale is not init for " + mNameImage);
   return mFocalMM;
}

tREAL8  cMedaDataImage::FocalMMEqui35() const
{
    MMVII_INTERNAL_ASSERT_User(mFocalMMEqui35>0,eTyUEr::eNoFocaleEqui35,"FocaleEqui35 is not init for " + mNameImage);
   return mFocalMMEqui35;
}


cMedaDataImage::cMedaDataImage(const std::string & aNameIm) :
   cMedaDataImage()
{
     mNameImage    = aNameIm;
     if (starts_with(aNameIm,"_DSC"))
         mAperture = 11.0;  
     else if (starts_with(aNameIm,"Img"))
         mAperture = 11.0;  
     else 
     {
         mAperture = 11.0;  
         // MMVII_INTERNAL_ERROR("cMedaDataImage to implemant");
     }


     MMVII_DEV_WARNING("cMedaDataImage : quick and (VERY) dirty implementation, most probably wrong");
}

cMedaDataImage::cMedaDataImage() :
    mCameraName       (""),
    mAperture         (-1),
    mFocalMM          (-1),
    mFocalMMEqui35    (-1)
{
}


}; // MMVII

