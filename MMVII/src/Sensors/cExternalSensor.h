#include "StdAfx.h"
#include "V1VII.h"
#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "../Serial/Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{
class  cDataExternalSensor
{
     public :
          cDataExternalSensor(const std::string& aNameFile="");
          
          std::string    mNameFile;
          eTypeSensor    mType;
          eFormatSensor  mFormat;
};

void AddData(const  cAuxAr2007 & anAux,cDataExternalSensor & aDES);

struct cAnalyseTSOF
{
     cAnalyseTSOF(const std::string& aNameFile,bool SVP=false);

     void FreeAnalyse();

     /*
     std::string    mNameFile;
     eFormatSensor  mFormat;
     eTypeSensor    mType;
     */
     cDataExternalSensor   mData;
     cSerialTree *         mSTree;
};
cSensorImage *  AllocRPCDimap(const cAnalyseTSOF &,const std::string & aNameImage);

};
