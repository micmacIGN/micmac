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
             cExportV1StenopeCalInterne
             (
                 bool isForCalib,
                 const std::string& aFile,
                 int aNbPointPerDim=30,
                 int aNbLayer=2,
                 tREAL8 aDownScale=1.0,
                 const std::string& aFileInterneCalib=""
             );

	     cIsometry3D<tREAL8>   mPose;
	     std::string           mNameCalib;
             bool                  mIsForCalib;
	     eProjPC               eProj;
	     cPt2di                mSzCam;
	     tREAL8                mFoc;
	     cPt2dr                mPP;
	     cSet2D3D              mCorresp;
};

//  Defined in MMVII_Stringifier.h for Serialization
// template<class Type> void  MMv1_SaveInFile(const Type & aVal,const std::string & aName)
// template<> void  MMv1_SaveInFile(const tNameSet & aVal,const std::string & aName);

///  Import a file of measures image in V1 format in a V2 structure
void ImportMesImV1(std::list<cSetMesPtOf1Im>  &,const std::string & aNameFileMesImV1);

///  Import a file of measures ground-control-points in V1 format in a V2 structure
cSetMesGnd3D ImportMesGCPV1(const std::string & aNameFileMesGCPV1,const std::string & aNameSet);



};

#endif  //  _MMVII_MMV1_Compat_H_
