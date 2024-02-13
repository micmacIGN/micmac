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

class cExternalSensor : public cSensorImage
{
      public :
         cExternalSensor(const cDataExternalSensor & aData,const std::string& aNameImage,cSensorImage * aSI);
         virtual ~cExternalSensor();
         void AddData(const  cAuxAr2007 & anAux);
         static std::string  StaticPrefixName();

         void SetSensorInit(cSensorImage *);
         const cDataExternalSensor &  Data() const;

      protected :
         // ====  Methods overiiding for being a cSensorImage =====
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;
        /// Indicate how much a point belongs to sensor visibilty domain
         double DegreeVisibility(const cPt3dr &) const  override;
         bool  HasIntervalZ()  const override;
         cPt2dr GetIntervalZ() const override;
         cPt3dr  PseudoCenterOfProj() const override;

         const cPixelDomain & PixelDomain() const ;
         std::string  V_PrefixName() const  override;
         void ToFile(const std::string &) const override;

         cDataExternalSensor     mData;
         cSensorImage *          mSensorInit;
};



};
