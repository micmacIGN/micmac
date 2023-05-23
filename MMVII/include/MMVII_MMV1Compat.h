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

/**  Class for converting homolog point MMV1 to MMVII
*/

class cConvertHomV1
{
      public :
          cConvertHomV1(const std::string & aDir,const std::string & aSubDir,const std::string & anExt);


          std::string NameHom(const std::string & aNameIm1,const std::string & aNameIm2) const;
      private :

          std::string  mDir;
          std::string  mSubDir;
          std::string  mExt;
          std::string  mKHIn;
          // cElemAppliSetFile                 mEASF;
          // cInterfChantierNameManipulateur * mICNM ;
};




//  Defined in MMVII_Stringifier.h for Serialization
// template<class Type> void  MMv1_SaveInFile(const Type & aVal,const std::string & aName)
// template<> void  MMv1_SaveInFile(const tNameSet & aVal,const std::string & aName);




};

#endif  //  _MMVII_MMV1_Compat_H_
