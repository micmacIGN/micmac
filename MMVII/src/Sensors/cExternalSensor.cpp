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
/* =============================================== */
/*                                                 */
/*                 cDataEmbededSensor              */
/*                                                 */
/* =============================================== */

cDataEmbededSensor::cDataEmbededSensor(const std::string& aNameFile,const std::string & aNameImage) :
   mNameFileInit (aNameFile),
   mNameImage    (aNameFile),
   mType         (eTypeSensor::eNbVals),
   mFormat       (eFormatSensor::eNbVals),
   mSysCoOri     (MMVII_NONE),
   mSysCoTarget  (MMVII_NONE)
{
}

/* =============================================== */
/*                                                 */
/*                 cAnalyseTSOF                    */
/*                                                 */
/* =============================================== */

void AddData(const  cAuxAr2007 & anAux,cDataEmbededSensor & aDES)
{
    AddData(cAuxAr2007("NameFileInit",anAux),aDES.mNameFileInit);
    AddData(cAuxAr2007("NameImage",anAux),aDES.mNameImage);
    EnumAddData(anAux,aDES.mType,"TypeSensor");
    EnumAddData(anAux,aDES.mFormat,"FileFormat");
    AddData(cAuxAr2007("InitialCoordSys",anAux),aDES.mSysCoOri);
    AddData(cAuxAr2007("TargetCoordSys",anAux),aDES.mSysCoTarget);
}


cAnalyseTSOF::cAnalyseTSOF(const std::string& aNameFile,bool SVP) :
   mData     (aNameFile),
   mSTree    (nullptr)
{
    std::string aPost = LastPostfix(aNameFile);
    eTypeSerial aTypeS = Str2E<eTypeSerial>(ToLower(aPost),true);
    
    if (aTypeS != eTypeSerial::eNbVals)
    {
        cSerialFileParser * aSFP = cSerialFileParser::Alloc(aNameFile,aTypeS);
	mSTree = new cSerialTree(*aSFP);
        delete aSFP;
        // Is it a dimap tree
        if (!mSTree->GetAllDescFromName("Dimap_Document").empty())
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

cSensorImage *  CreateAutoExternalSensor(const cAnalyseTSOF & anAnalyse ,const std::string & aNameImage,bool SVP)
{
    if (anAnalyse.mData.mFormat == eFormatSensor::eDimap_RPC)
    {
       return   AllocRPCDimap(anAnalyse,aNameImage);
    }

    if (!SVP)
    {
        MMVII_INTERNAL_ERROR("CreateAutoExternalSensor dont handle for file :" + anAnalyse.mData.mNameFileInit);
    }
    return nullptr;
}

cSensorImage *  CreateAutoExternalSensor(const std::string& aNameFile,const std::string & aNameImage,bool SVP)
{
    cAnalyseTSOF anAnalyse(aNameFile,SVP);
    cSensorImage * aSI = CreateAutoExternalSensor(anAnalyse,aNameImage);
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

   // ================  Constructor/Destructor ====================

cExternalSensor::cExternalSensor(const cDataEmbededSensor & aData,const std::string& aNameImage,cSensorImage * aSI) :
     cSensorImage  (aNameImage),
     mData         (aData),
     mSensorInit   (aSI)
{
}

cExternalSensor::~cExternalSensor() 
{
    delete mSensorInit;
}


cExternalSensor * cExternalSensor::TryRead
                  (
                        const std::string & aDirInit,
                        const std::string & aDirSens,
                        const std::string & aNameImage
                  )
{

   std::string aNameFile = aDirSens + cSensorImage::NameOri_From_PrefixAndImage(cExternalSensor::StaticPrefixName(),aNameImage);

   if (!ExistFile(aNameFile))
      return nullptr;

   // -1-   Create the object
   cExternalSensor *  aResult = new cExternalSensor (cDataEmbededSensor(),aNameImage);

   // -2-   Read the data contained in the file
   ReadFromFile(*aResult,aNameFile);
//-------------- TEMPO ---------------------------------
MMVII_INTERNAL_ASSERT_strong(aResult->Data().mSysCoOri==aResult->Data().mSysCoTarget,"For Now dont handle diff coord sys");
   // -3-   Read the initial sensor
   std::string aNameInit  = aDirInit + aResult->Data().mNameFileInit;
   cSensorImage *  aSI =  CreateAutoExternalSensor(aNameInit,aNameImage,false);
   aResult->SetSensorInit(aSI);

   return aResult;
}


void cExternalSensor::SetSensorInit(cSensorImage * aSI)
{
    MMVII_INTERNAL_ASSERT_strong((mSensorInit==nullptr) || (aSI==nullptr),"Multiple Init for  cExternalSensor::SetSensorInit");
    mSensorInit = aSI;
}

cSensorImage * cExternalSensor::SensorInit()
{
	return mSensorInit;
}

   
     // ==============   READ/WRITE/SERIAL ================

std::string  cExternalSensor::StaticPrefixName() { return  "ExternalSensor"  ; }
std::string  cExternalSensor::V_PrefixName() const { return  StaticPrefixName(); }

void cExternalSensor::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("General",anAux),mData);
}

void AddData(const  cAuxAr2007 & anAux,cExternalSensor& aDES)
{
     aDES.AddData(anAux);
}

void cExternalSensor::ToFile(const std::string & aNameFile) const 
{
     SaveInFile(const_cast<cExternalSensor &>(*this),aNameFile);
}

const cDataEmbededSensor &  cExternalSensor::Data() const {return mData;}

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
	std::string                 mSysCoordTarget;

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
           << AOpt2007(mSysCoordTarget,"TargetSysCoord","Specify Target Coordinate System if != initial")
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
    CopyFile(aNameSensor,mPhProj.DirImportInitOri()+aNameSensor);

    // Make analyse to recognize automatically the kind of file
    cAnalyseTSOF  anAnalyse (aNameSensor);

    // Now create the initial sensor
    cSensorImage *  aSensorInit =  CreateAutoExternalSensor(anAnalyse ,aNameIm);

    // Set Ori Sys coord and check  is valid
    SetIfNotInit(mSysCoordOri,aSensorInit->CoordinateSystem());
    mPhProj.ReadSysCo(mSysCoordOri);
    anAnalyse.mData.mSysCoOri = mSysCoordOri;

    // Set Target Sys coord and check  is valid
    SetIfNotInit(mSysCoordTarget,mSysCoordOri);
    mPhProj.ReadSysCo(mSysCoordTarget);
    anAnalyse.mData.mSysCoTarget = mSysCoordTarget;

    // Encapsulate this initial sensor with eventually coefficient
    cSensorImage * aSensorEnd =  new cExternalSensor(anAnalyse.mData,aNameIm,aSensorInit);

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
