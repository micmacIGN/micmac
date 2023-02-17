#ifndef CPP_XIFGPS2XML_H
#define CPP_XIFGPS2XML_H

#include "StdAfx.h"
// #include "XML_GEN/all_tpl.h"

#include "../TpMMPD/ConvertRtk.h"

class cAppli_XifGps2Xml ;
class cAppli_XifDate;
class cAppli_VByDate; // Calculate Im Velocity with Exif Date data
class cIm_XifGp;
class cIm_XifDate;



class cIm_XifDate // get exif time info for one image
{
    public:
        cIm_XifDate(const string &aName);
        cIm_XifDate(const std::string & aName,cElHour & aBeginTime);

        std::string      mName;
        cMetaDataPhoto   mMDP;
        cElDate          mDate;
        double           mDiffSecond; // time lap w.r.t the first image in (s)
};


class cAppli_XifDate : public cAppliListIm // get exif time info for a set of cIm_XifDate
{
    public:
        cAppli_XifDate(const std::string & aFullName);
        void Export(const std::string & aOut);
        std::vector<double> GetVDiffSecond();

        std::vector<cIm_XifDate> mVIm;
        cElHour                  mBegin;
};


class cIm_XifGp
{
     public :
         cIm_XifGp(const std::string & aName,cAppli_XifGps2Xml &);

         cAppli_XifGps2Xml * mAppli;
         std::string         mName;
         cMetaDataPhoto      mMDP;
         bool                mHasPT;
         Pt3dr               mPGeoC;
};


class cAppli_XifGps2Xml : public cAppliListIm
{
    public :
       cAppli_XifGps2Xml(const std::string & aFullName,double aDefZ);
       void ExportSys(cSysCoord *,const std::string & anOri);
       void ExportCoordTxtFile(std::string aOut, std::string aOutFormat);

    public :
      std::vector<cIm_XifGp>  mVIm;
      double                  mDefZ;
      double                  mNbOk;
      Pt3dr                   mGeoCOriRTL;
      Pt3dr                   mWGS84DegreeRTL;
      cSystemeCoord           mSysRTL;
      cOrientationConique     mOC0;
};




class cAppli_VByDate : public cAppli_XifDate // Calculate velocity from cam position and cam time
{
    public:
        cAppli_VByDate(const std::string & aFullName, std::string & aOri);
        void CalcV(const std::string & aOut, const bool aHeader);

        std::vector<CamStenope*>          mVCam;
};

#endif // CPP_XIFGPS2XML_H

