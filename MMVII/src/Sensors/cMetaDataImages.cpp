#include "MMVII_nums.h"
#include "MMVII_Error.h"
#include "MMVII_Sensor.h"

/**
   \file cMetaDataImages.cpp

   \brief file for handling meta data
*/


namespace MMVII
{

enum class eMTDIm
           {
              Focalmm,   
              Aperture,   
              ModeleCam,
              eNbVals    
           };


class cOneTryCAI
{
     public :
        cOneTryCAI();


        cOneTryCAI(const std::string & aPat,const std::string & aValue);

        std::string   mPat;
	tNameSelector mSel;
        std::string   mValue;
};

/* ******************************************* */
/*                                             */
/*                cOneTryCAI                   */
/*                                             */
/* ******************************************* */

cOneTryCAI::cOneTryCAI(const std::string & aPat,const std::string & aValue) :
     mPat     (aPat),
     mSel     (AllocRegex(aPat)),
     mValue   (aValue)
{
}

cOneTryCAI::cOneTryCAI() :
	cOneTryCAI("","")
{
}

void AddData(const cAuxAr2007 & anAux,cOneTryCAI & aTry)
{
     AddData(cAuxAr2007("Pat",anAux),aTry.mPat);

     if (anAux.Input())
     {
         aTry.mSel = AllocRegex(aTry.mPat);
     }
     AddData(cAuxAr2007("Val",anAux),aTry.mValue);
}

/* ******************************************* */
/*                                             */
/*                cOneCalAttrIm                */
/*                                             */
/* ******************************************* */
class cOneCalAttrIm
{
     public :
	  cOneCalAttrIm();

          std::string Translate(const std::string & aName);

	  eMTDIm                   mIm;
          std::vector<cOneTryCAI>  mVTry;
	  std::string              mDefault;
};

std::string cOneCalAttrIm::Translate(const std::string & aName)
{
    for (const auto & aTry : mVTry)
    {
        if (aTry.mSel.Match(aName))
	{
            std::string aTransfo = ReplacePattern(aTry.mPat,aTry.mValue,aName);
	    if (aTransfo != mDefault)
               return aTransfo;
	}
    }
    return mDefault;
}



class cCalculMetaData
{
     public :
};




tREAL8  cMetaDataImage::Aperture() const
{
   MMVII_INTERNAL_ASSERT_User(mAperture>0,eTyUEr::eNoAperture,"Aperture is not init for " + mNameImage);
   return mAperture;
}

tREAL8  cMetaDataImage::FocalMM() const
{
   MMVII_INTERNAL_ASSERT_User(mFocalMM>0,eTyUEr::eNoFocale,"Focale is not init for " + mNameImage);
   return mFocalMM;
}

tREAL8  cMetaDataImage::FocalMMEqui35() const
{
    MMVII_INTERNAL_ASSERT_User(mFocalMMEqui35>0,eTyUEr::eNoFocaleEqui35,"FocaleEqui35 is not init for " + mNameImage);
   return mFocalMMEqui35;
}


cMetaDataImage::cMetaDataImage(const std::string & aNameIm) :
   cMetaDataImage()
{
     mNameImage    = aNameIm;
     if (starts_with(aNameIm,"_DSC"))
         mAperture = 11.0;  
     else if (starts_with(aNameIm,"Img"))
         mAperture = 11.0;  
     else 
     {
         mAperture = 11.0;  
         // MMVII_INTERNAL_ERROR("cMetaDataImage to implemant");
     }


     MMVII_DEV_WARNING("cMetaDataImage : quick and (VERY) dirty implementation, most probably wrong");
}

cMetaDataImage::cMetaDataImage() :
    mCameraName       (""),
    mAperture         (-1),
    mFocalMM          (-1),
    mFocalMMEqui35    (-1)
{
}


}; // MMVII

