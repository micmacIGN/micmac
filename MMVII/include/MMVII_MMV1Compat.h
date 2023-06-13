#ifndef  _MMVII_MMV1_Compat_H_
#define  _MMVII_MMV1_Compat_H_

#include "MMVII_Sensor.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

/** \file MMVII_MMV1Compat.h
    \brief Function/class to make communication between MMVII and MMv1

*/

tNameSet  MMV1InitSet(const std::string & aName);
tNameRel  MMV1InitRel(const std::string & aName);


/** class for exporting internal calibration resulting of MMV1process, mainly exported as
 */
struct  cExportV1StenopeCalInterne
{
       public :
             cExportV1StenopeCalInterne(bool isForCalib,const std::string& aFile,int aNbPointPerDim=30,int aNbLayer=2);

	     cIsometry3D<tREAL8>   mPose;
	     std::string           mNameCalib;
             bool                  mIsForCalib;
	     eProjPC               eProj;
	     cPt2di                mSzCam;
	     tREAL8                mFoc;
	     cPt2dr                mPP;
	     cSet2D3D              mCorresp;
};

/**  Interface class  for importing homolog point MMV1 to MMVII
*/

class cInterfImportHom : public cMemCheck
{
      public :
          virtual void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const = 0;
          virtual bool HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const = 0;

	  static cInterfImportHom *  CreateImportV1(const std::string & aDir,const std::string & aSubDir,const std::string & aExt="dat") ;

      private :

          // std::string  mDirGlob;
          // std::string  mExtIn;
          // std::string  mExt;
          // std::string  mKHIn;
          // cElemAppliSetFile                 mEASF;
          // cInterfChantierNameManipulateur * mICNM ;
};




//  Defined in MMVII_Stringifier.h for Serialization
// template<class Type> void  MMv1_SaveInFile(const Type & aVal,const std::string & aName)
// template<> void  MMv1_SaveInFile(const tNameSet & aVal,const std::string & aName);




};

#endif  //  _MMVII_MMV1_Compat_H_
