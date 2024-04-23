#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_PhgrDist.h"
#include "../Serial/Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "cExternalSensor.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cExternalSensor : public cSensorImage
{
      public :
         cExternalSensor(const cDataImportSensor & aData,const std::string& aNameImage,cSensorImage * aSI,const std::string & aSysCo);
	 cExternalSensor();

         static cExternalSensor * TryRead(const cPhotogrammetricProject &,const std::string&aNameImage);

         virtual ~cExternalSensor();
         void AddData(const  cAuxAr2007 & anAux);
         static std::string  StaticPrefixName();

         void SetSensorInit(cSensorImage *);
         cSensorImage * SensorInit();
         const cDataImportSensor &  Data() const;

         /// Used to set image after read
         void ResetRead();
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

         const cPixelDomain & PixelDomain() const override;
         std::string  V_PrefixName() const  override;
         void ToFile(const std::string &) const override;

         cDataImportSensor     mData;
         cSensorImage *          mSensorInit;
};

/* =============================================== */
/*                                                 */
/*                 cDataImportSensor              */
/*                                                 */
/* =============================================== */

cDataImportSensor::cDataImportSensor() :
   mNameFileInit   (""),
   mNameImage      (""),
   mType           (eTypeSensor::eNbVals),
   mFormat         (eFormatSensor::eNbVals)
{
}


/* =============================================== */
/*                                                 */
/*                 cAnalyseTSOF                    */
/*                                                 */
/* =============================================== */

bool TreeIsDimap(const cSerialTree & aTree)
{
    return  aTree.HasValAtUniqueTag("METADATA_FORMAT","DIMAP") ;
}


cAnalyseTSOF::cAnalyseTSOF(const std::string& aNameFile,bool SVP) :
   mData     (),
   mSTree    (nullptr)
{
    mData.mNameFileInit = aNameFile;
    std::string aPost = LastPostfix(aNameFile);
    eTypeSerial aTypeS = Str2E<eTypeSerial>(ToLower(aPost),true);
    
    if (aTypeS != eTypeSerial::eNbVals)
    {
        cSerialFileParser * aSFP = cSerialFileParser::Alloc(aNameFile,aTypeS);
	mSTree = new cSerialTree(*aSFP);
        delete aSFP;
        // Is it a dimap tree
	// METADATA_FORMAT DIMAP
        if (TreeIsDimap(*mSTree))
        {
           mData.mFormat =  eFormatSensor::eDimap_RPC;
           mData.mType   =  eTypeSensor::eRPC;
	   return ;
        }
    }
    if (! SVP)
       MMVII_UnclasseUsEr("AnalyseFileSensor dont recognize : " + aNameFile);
    return ;
}

cSensorImage *  ReadExternalSensor(const cAnalyseTSOF & anAnalyse ,const std::string & aNameImage,bool SVP)
{
    if (anAnalyse.mData.mFormat == eFormatSensor::eDimap_RPC)
    {
       return   AllocRPCDimap(anAnalyse,aNameImage);
    }

    if (!SVP)
    {
        MMVII_INTERNAL_ERROR("ReadExternalSensor dont handle for file :" + anAnalyse.mData.mNameFileInit);
    }
    return nullptr;
}

cSensorImage *  ReadExternalSensor(const std::string& aNameFile,const std::string & aNameImage,bool SVP)
{
    cAnalyseTSOF anAnalyse(aNameFile,SVP);
    cSensorImage * aSI = ReadExternalSensor(anAnalyse,aNameImage);
    anAnalyse.FreeAnalyse();
    return aSI;
}

void cAnalyseTSOF::FreeAnalyse()
{
     delete mSTree;
     mSTree = nullptr;
}



/* =============================================== */
/*                                                 */
/*                 cExternalSensor                 */
/*                                                 */
/* =============================================== */

#if (0)  
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


	 // ============   Differenciation =========================

	  cPt3dr  EpsDiffGround2Im(const cPt3dr &) const override;
          tProjImAndGrad  DiffGround2Im(const cPt3dr &) const override;

         bool  HasIntervalZ()  const override;
         cPt2dr GetIntervalZ() const override;
         cPt3dr  PseudoCenterOfProj() const override;

         const cPixelDomain & PixelDomain() const override;
         std::string  V_PrefixName() const  override;
         void ToFile(const std::string &) const override;

         cDataExternalSensor     mData;
         cSensorImage *          mSensorInit;
};


#endif
   // ================  Constructor/Destructor ====================

cExternalSensor::cExternalSensor(const cDataImportSensor & aData,const std::string& aNameImage,cSensorImage * aSI,const std::string & aSysCo) :
     cSensorImage  (aNameImage),
     mData         (aData),
     mSensorInit   (nullptr)
{
  // full name was usefull for analyse, but for write/re-read we need only the name without folder
   mData.mNameFileInit = FileOfPath(mData.mNameFileInit,false);
   SetCoordinateSystem(aSysCo);
   SetSensorInit(aSI);
}

cExternalSensor::cExternalSensor() :
     cExternalSensor(cDataImportSensor (),MMVII_NONE,nullptr,MMVII_NONE)
{
}

cExternalSensor::~cExternalSensor() 
{
    delete mSensorInit;
}

void cExternalSensor::AddData(const  cAuxAr2007 & anAux0)
{
    cAuxAr2007 anAux("ExternalSensor",anAux0); // Embed all in this tag

    MMVII::AddData(cAuxAr2007("NameFileInit",anAux),mData.mNameFileInit);
    MMVII::AddData(cAuxAr2007("NameImage",anAux),mData.mNameImage);
    MMVII::EnumAddData(anAux,mData.mType,"TypeSensor");
    MMVII::EnumAddData(anAux,mData.mFormat,"FileFormat");
    MMVII::AddOptData(anAux,TagCoordSys,OptCoordinateSystem());
}

void AddData(const  cAuxAr2007 & anAux,cExternalSensor& aDES)
{
     aDES.AddData(anAux);
}

cExternalSensor * cExternalSensor::TryRead
                  (
		        const cPhotogrammetricProject & aPhProj,
                        const std::string & aNameImage
                  )
{
   bool AlreadyExist = false;
   // cExternalSensor  *  aResult = TplTryRead<cExternalSensor,cDataImportSensor>(aPhProj,aNameImage,AlreadyExist);
   cExternalSensor  *  aResult = SimpleTplTryRead<cExternalSensor>(aPhProj,aNameImage,AlreadyExist);
   if ((aResult==nullptr) || AlreadyExist) return aResult;

   // -3-   Read the initial sensor
   std::string  aDirInit = aPhProj.DirImportInitOri();
   std::string aNameInit  = aDirInit + aResult->Data().mNameFileInit;
   cSensorImage *  aSI =  ReadExternalSensor(aNameInit,aNameImage,false);
   aResult->SetSensorInit(aSI);

   return aResult;
}

cSensorImage * SensorTryReadImported(const cPhotogrammetricProject & aPhProj,const std::string & aNameImage)
{
     return cExternalSensor::TryRead(aPhProj,aNameImage);
}


void cExternalSensor::SetSensorInit(cSensorImage * aSI)
{
    MMVII_INTERNAL_ASSERT_strong
    (
         (mSensorInit==nullptr) || (aSI==nullptr),
        "Multiple Init for  cExternalSensor::SetSensorInit"
    );
    mSensorInit = aSI;
    /*
    if (aSI)
       TransferateCoordSys(*aSI);
       */
}

cSensorImage * cExternalSensor::SensorInit()
{
	return mSensorInit;
}

   
     // ==============   READ/WRITE/SERIAL ================

std::string  cExternalSensor::StaticPrefixName() { return  "ExternalSensor"  ; }
std::string  cExternalSensor::V_PrefixName() const { return  StaticPrefixName(); }


void cExternalSensor::ToFile(const std::string & aNameFile) const 
{
     SaveInFile(*this,aNameFile);
}

const cDataImportSensor &  cExternalSensor::Data() const {return mData;}

void cExternalSensor::ResetRead()
{
    SetNameImage(mData.mNameImage);
}

     // =============   METHOD FOR BEING a cSensorImage =====================

tSeg3dr  cExternalSensor::Image2Bundle(const cPt2dr & aP) const 
{
	return mSensorInit->Image2Bundle(aP);
}

cPt2dr cExternalSensor::Ground2Image(const cPt3dr & aPGround) const 
{
	return mSensorInit->Ground2Image(aPGround);
}

cPt3dr cExternalSensor::ImageAndZ2Ground(const cPt3dr & aPE) const 
{
    return mSensorInit->ImageAndZ2Ground(aPE);
}

double cExternalSensor::DegreeVisibility(const cPt3dr & aPGround) const
{
	return mSensorInit->DegreeVisibility(aPGround);
}

const cPt3dr *  cExternalSensor::CenterOfFootPrint() const 
{
	return mSensorInit->CenterOfFootPrint();
}

cPt3dr  cExternalSensor::EpsDiffGround2Im(const cPt3dr & aPt) const
{
     return mSensorInit->EpsDiffGround2Im(aPt);
}

tProjImAndGrad  cExternalSensor::DiffGround2Im(const cPt3dr & aPt) const
{
     return mSensorInit->DiffGround2Im(aPt);
}

bool  cExternalSensor::HasIntervalZ()  const {return mSensorInit->HasIntervalZ();}
cPt2dr cExternalSensor::GetIntervalZ() const {return mSensorInit->GetIntervalZ();}

//  for small deformation , the pixel domain is the same than the init sensor
const cPixelDomain & cExternalSensor::PixelDomain() const 
{
	return mSensorInit->PixelDomain();
}

cPt3dr  cExternalSensor::PseudoCenterOfProj() const {return mSensorInit->PseudoCenterOfProj();}


/* =============================================== */
/*                                                 */
/*           cAppliImportInitialExternSensor       */
/*                                                 */
/* =============================================== */

class cAppliImportInitialExternSensor : public cMMVII_Appli
{
     public :

        cAppliImportInitialExternSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;

	void ImportOneImage(const std::string &);

        cPhotogrammetricProject  mPhProj;

        // --- Mandatory ----
        std::string                 mNameImagesIn;
	std::vector<std::string>    mPatChgName;
	std::string                 mSysCoordOri;

        // --- Optionnal ----

     // --- Internal ----
};

cAppliImportInitialExternSensor::cAppliImportInitialExternSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this)
{
}


cCollecSpecArg2007 & cAppliImportInitialExternSensor::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return anArgObl
      <<   Arg2007(mNameImagesIn,"Name of input sensor gile", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      <<   Arg2007(mPatChgName,"[PatNameIm,NameSens]", {{eTA2007::ISizeV,"[2,2]"}})
      <<   mPhProj.DPOrient().ArgDirOutMand()
   ;
}

cCollecSpecArg2007 & cAppliImportInitialExternSensor::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mSysCoordOri,"InitialSysCoord","Specify Coordinate System of  initial coordinate, in case default dont work")
   ;
}

std::vector<std::string>  cAppliImportInitialExternSensor::Samples() const
{
   return {
              "MMVII ImportPushbroom AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init",
              "MMVII ImportPushbroom AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init"
	};
}

void  cAppliImportInitialExternSensor::ImportOneImage(const std::string & aNameIm)
{
    // Compute the name of the sensor from the name of image using pat-subst
    std::string aFullNameSensor = ReplacePattern(mPatChgName.at(0),mPatChgName.at(1),aNameIm);
    // supress the name of folder that may exist
    std::string aNameSensor = FileOfPath(aFullNameSensor,false);

    //  Make a local copy of the initial sensor (that maybe located anyway and may disapear later)
    CopyFile(aFullNameSensor,mPhProj.DirImportInitOri()+aNameSensor);

    // Make analyse to recognize automatically the kind of file
    //   ######## cAnalyseTSOF  anAnalyse (aNameSensor);
    cAnalyseTSOF  anAnalyse (aFullNameSensor);

    anAnalyse.mData.mNameImage = aNameIm;
    // Now create the initial sensor
    cSensorImage *  aSensorInit =  ReadExternalSensor(anAnalyse ,aNameIm);

    // Set Ori Sys coord and check  is valid
    if (aSensorInit->HasCoordinateSystem())
    {
       SetIfNotInit(mSysCoordOri,aSensorInit->GetCoordinateSystem());
    }

    mPhProj.SaveCurSysCoOri(mPhProj.ReadSysCo(mSysCoordOri));

    // Encapsulate this initial sensor with eventually coefficient
    cSensorImage * aSensorEnd =  new cExternalSensor(anAnalyse.mData,aNameIm,aSensorInit,mSysCoordOri);

    // free the memory of Analyse (is not automatic, because can be copied) 
    anAnalyse.FreeAnalyse();
    // Save the result
    mPhProj.SaveSensor(*aSensorEnd);

    StdOut() << "NAMES    Ima: " << aNameIm << " SensInit : " << aNameSensor  << " SensSave : " << aSensorEnd->NameOriStd() << "\n";

    //  Free the sensor (was not allocated by PhProj, will not be automatically deleted)
    delete aSensorEnd;
}


int cAppliImportInitialExternSensor::Exe()
{
    mPhProj.FinishInit();

    for (const auto & aNameIm :  VectMainSet(0))
    {
         ImportOneImage(aNameIm);
    }

    return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_ImportExtSens(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliImportInitialExternSensor(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecImportExtSens
(
     "ImportInitExtSens",
      Alloc_ImportExtSens,
      "Import an  Initial External Sensor",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);





};
