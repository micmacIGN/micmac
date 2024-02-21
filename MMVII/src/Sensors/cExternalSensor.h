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
class  cDataEmbededSensor
{
     public :
          cDataEmbededSensor(const std::string& aNameFileOri="",const std::string& aNameImage="");
          
          std::string    mNameFileInit;
          std::string    mNameImage;
          eTypeSensor    mType;
          eFormatSensor  mFormat;
	  std::string    mSysCoOri;
	  std::string    mSysCoTarget;
};

void AddData(const  cAuxAr2007 & anAux,cDataEmbededSensor & aDES);

struct cAnalyseTSOF
{
     explicit cAnalyseTSOF(const std::string& aNameFile,bool SVP=false);

     void FreeAnalyse();
     cDataEmbededSensor   mData;
     cSerialTree *         mSTree;
};
cSensorImage *  AllocRPCDimap(const cAnalyseTSOF &,const std::string & aNameImage);


class cExternalSensor : public cSensorImage
{
      public :
         cExternalSensor(const cDataEmbededSensor & aData,const std::string& aNameImage,cSensorImage * aSI=nullptr);

	 static cExternalSensor * TryRead(const std::string&aDirInit,const std::string&aDirSens,const std::string&aNameImage);

         virtual ~cExternalSensor();
         void AddData(const  cAuxAr2007 & anAux);
         static std::string  StaticPrefixName();

         void SetSensorInit(cSensorImage *);
         cSensorImage * SensorInit();
         const cDataEmbededSensor &  Data() const;
      protected :
         // ====  Methods overiiding for being a cSensorImage =====
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;
        /// Indicate how much a point belongs to sensor visibilty domain
         double DegreeVisibility(const cPt3dr &) const  override;

         ///  Return center of footprint (used for example in RTL)
         const cPt3dr *  CenterOfFootPrint() const override;

         // ============   Differenciation =========================

          cPt3dr  EpsDiffGround2Im(const cPt3dr &) const override;
          tProjImAndGrad  DiffGround2Im(const cPt3dr &) const override;

         bool  HasIntervalZ()  const override;
         cPt2dr GetIntervalZ() const override;
         cPt3dr  PseudoCenterOfProj() const override;

         const cPixelDomain & PixelDomain() const ;
         std::string  V_PrefixName() const  override;
         void ToFile(const std::string &) const override;

         cDataEmbededSensor     mData;
         cSensorImage *          mSensorInit;
};
void AddData(const  cAuxAr2007 & anAux,cExternalSensor& aDES);
cSensorImage *  CreateAutoExternalSensor(const cAnalyseTSOF & anAnalyse ,const std::string & aNameImage,bool SVP=false);
cSensorImage *  CreateAutoExternalSensor(const std::string& aNameFile ,const std::string & aNameImage,bool SVP=false);



class cExternalSensorModif2D : public cExternalSensor
{
     public :
         static std::string  StaticPrefixName();
         cExternalSensorModif2D
         (
                const cDataEmbededSensor & aData,
                const std::string& aNameImage,
                cSensorImage * aSI = nullptr,
                int aDegree = -1,
                const std::string & aTargetSysCo = ""
         ) ;
         void AddData(const  cAuxAr2007 & anAux);

         void PerturbateRandom(tREAL8 anAmpl,bool Show);
         ~cExternalSensorModif2D();

         void Finish(const cPhotogrammetricProject &);
         static cExternalSensorModif2D * TryRead(const cPhotogrammetricProject &,const std::string&aNameImage);

         double DegreeVisibility(const cPt3dr & aPGround) const override;

     private :
         cPt3dr  EpsDiffGround2Im(const cPt3dr &) const override;
         tProjImAndGrad  DiffGround2Im(const cPt3dr & aP) const override;
         void InitPol2D();

         // ====  Methods overiiding for being a cSensorImage =====
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;

         void ToFile(const std::string &) const override;
         std::string  V_PrefixName() const  override;

              // ------------------- Bundles Adjustment -----------------------
         cCalculator<double> * CreateEqColinearity(bool WithDerives,int aSzBuf,bool ReUse) override;
         void PutUknowsInSetInterval() override;
         void PushOwnObsColinearity( std::vector<double> &,const cPt3dr &)  override;
	  
	         // ====  Method to override in derived classes  =====

         /// For "initial" coordinat (pixel) to "final"
         cPt2dr  InitPix2Correc (const cPt2dr & aP0) const ;
         /// For "final" to "initial" coordinat (pixel)
         cPt2dr  Correc2InitPix (const cPt2dr & aP0) const ;

         int                     mDegree;
         bool                    mActiveDistG2I;  // for differenciation we need to inhibate temporarily
         std::string             mTargetSysCo;
         std::vector<tREAL8>     mVParams;
         cCalculator<double> *   mEqIma2End;  // Functor that gives the distorstion
         cChangSysCoordV2        mSysI2F;  // Chang cood Init 2 Final
         cPt3dr                  mPtEpsDeriv;
         bool                    mIdChSys;

};








};
