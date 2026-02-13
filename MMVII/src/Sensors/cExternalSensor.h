#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "../Serial/Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{

cSensorImage * SensorTryReasChSys(const cPhotogrammetricProject & aPhProj,const std::string & aNameImage);
cSensorImage * SensorTryReadImported(const cPhotogrammetricProject & aPhProj,const std::string & aNameImage);
cSensorImage * SensorTryReadSensM2D(const cPhotogrammetricProject & aPhProj,const std::string & aNameImage);


class  cDataImportSensor
{
     public :
          cDataImportSensor();
          
          std::string    mNameFileInit;  // if it is an imported external sensor
          std::string    mNameImage;
          eTypeSensor    mType;
          eFormatSensor  mFormat;
};
// void AddData(const  cAuxAr2007 & anAux,cDataEmbededSensor & aDES);


struct cAnalyseTSOF
{
     explicit cAnalyseTSOF(const std::string& aNameFile,bool SVP=false);

     void FreeAnalyse();
     cDataImportSensor   mData;
     cSerialTree *         mSTree;
};
cSensorImage *  AllocRPCDimap(const cAnalyseTSOF &,const std::string & aNameImage);
cSensorImage *  ReadExternalSensor(const cAnalyseTSOF & anAnalyse ,const std::string & aNameImage,bool SVP=false);
cSensorImage *  ReadExternalSensor(const std::string& aNameFile ,const std::string & aNameImage,bool SVP=false);

template <class TypeSens>
   TypeSens * SimpleTplTryRead(const cPhotogrammetricProject & aPhProj, const std::string & aNameImage,bool & AlreadyExist)
{
   AlreadyExist = false;
   std::string  aDirSens = aPhProj.DPOrient().FullDirIn();
   std::string aNameFile = aDirSens + cSensorImage::NameOri_From_PrefixAndImage(TypeSens::StaticPrefixName(),aNameImage);

   if (!ExistFile(aNameFile))
      return nullptr;

   // -1-   Create the object
   TypeSens * aResult = SimpleRemanentNewObjectFromFile<TypeSens>(aNameFile,&AlreadyExist);
   // -2-   Reset (for name image)
   aResult->SetNameImage(aNameImage);

   return aResult;
}

};
