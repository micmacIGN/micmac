#include "StdAfx.h"
#include "V1VII.h"
#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "../Serial/Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "cExternalSensor.h"


namespace MMVII
{
/* =============================================== */
/*                                                 */
/*                 cDataExternalSensor             */
/*                                                 */
/* =============================================== */

cDataExternalSensor::cDataExternalSensor(const std::string& aNameFile) :
   mNameFile (aNameFile),
   mType     (eTypeSensor::eNbVals),
   mFormat   (eFormatSensor::eNbVals)
{
}

/* =============================================== */
/*                                                 */
/*                 cAnalyseTSOF                    */
/*                                                 */
/* =============================================== */

void AddData(const  cAuxAr2007 & anAux,cDataExternalSensor & aDES)
{
    AddData(cAuxAr2007("NameFileInit",anAux),aDES.mNameFile);
    EnumAddData(anAux,aDES.mType,"TypeSensor");
    EnumAddData(anAux,aDES.mFormat,"FileFormat");
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

cSensorImage *  AllocAutoSensorFromFile(const cAnalyseTSOF & anAnalyse ,const std::string & aNameImage,bool SVP=false)
{
    if (anAnalyse.mData.mFormat == eFormatSensor::eDimap_RPC)
    {
       return   AllocRPCDimap(anAnalyse,aNameImage);
    }

    if (!SVP)
    {
        MMVII_INTERNAL_ERROR("AllocAutoSensorFromFile dont handle for file :" + anAnalyse.mData.mNameFile);
    }
    return nullptr;
}

cSensorImage *  AllocAutoSensorFromFile(const std::string& aNameFile,const std::string & aNameImage,bool SVP=false)
{
    cAnalyseTSOF anAnalyse(aNameFile,SVP);
    cSensorImage * aSI = AllocAutoSensorFromFile(anAnalyse,aNameImage);
    anAnalyse.FreeAnalyse();
    return aSI;
}




void cAnalyseTSOF::FreeAnalyse()
{
     delete mSTree;
}



/* =============================================== */
/*                                                 */
/*                 cExternalSensor                 */
/*                                                 */
/* =============================================== */

class cExternalSensor : public cSensorImage
{
      public :
         cExternalSensor(const cDataExternalSensor & aData,const std::string& aNameImage,cSensorImage * aSI);
         virtual ~cExternalSensor();
         void AddData(const  cAuxAr2007 & anAux);
         static std::string  StaticPrefixName();

         void SetSensorInit(cSensorImage *);
         const cDataExternalSensor &  Data() const;

      private :

         // void AddData(const  cAuxAr2007 & anAux,cDataExternalSensor & aDES)

         
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

         cDataExternalSensor     mData;
	 cSensorImage *          mSensorInit;

	 void ToFile(const std::string &) const override;

};

/*
class cExternalSensorModif2D : public cExternalSensor
{
     public :
         std::string  V_PrefixName() const  override;

	 // ====  Method to override in derived classes  ===== 
	 virtual cPt2dr  Init2End (const cPt2dr & aP0) const ;
	 virtual cPt2dr  End2Init (const cPt2dr & aP0) const ;
	 virtual  std::string NameModif2D() const;
         virtual void AddDataComplem(const  cAuxAr2007 & anAux);
};
     AddDataComplem(cAuxAr2007("Model2D",anAux));

std::string  cExternalSensorModif2D::V_PrefixName() const
{
	return  "ExternalSensor_Polyn2D" +  NameModif2D() ;
}
*/



   // ================  Constructor/Destructor ====================

cExternalSensor::cExternalSensor(const cDataExternalSensor & aData,const std::string& aNameImage,cSensorImage * aSI) :
     cSensorImage  (aNameImage),
     mData         (aData),
     mSensorInit   (aSI)
{
}

cExternalSensor::~cExternalSensor() 
{
    delete mSensorInit;
}

void cExternalSensor::SetSensorInit(cSensorImage * aSI)
{
    MMVII_INTERNAL_ASSERT_strong(mSensorInit==nullptr,"Multiple Init for  cExternalSensor::SetSensorInit");
    mSensorInit = aSI;
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

const cDataExternalSensor &  cExternalSensor::Data() const {return mData;}

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

bool  cExternalSensor::HasIntervalZ()  const {return mSensorInit->HasIntervalZ();}
cPt2dr cExternalSensor::GetIntervalZ() const {return mSensorInit->GetIntervalZ();}

//  for small deformation , the pixel domain is the same than the init sensor
const cPixelDomain & cExternalSensor::PixelDomain() const 
{
	return mSensorInit->PixelDomain();
}

cPt3dr  cExternalSensor::PseudoCenterOfProj() const {return mSensorInit->PseudoCenterOfProj();}


template <class TypeSens>  cSensorImage * GenAllocExternalSensor
                                          (
                                               const std::string & aDirInit,
                                               const std::string & aDirSens,
                                               const std::string aNameImage
                                          )
{
   std::string aNameFile = aDirSens + cSensorImage::NameOri_From_PrefixAndImage(TypeSens::StaticPrefixName(),aNameImage);

   if (ExistFile(aNameFile))
   {
        // -1-   Create the object
        cExternalSensor *  aResult = new TypeSens (cDataExternalSensor(),aNameImage,nullptr);
        // -2-   Read the data contained in the file
        ReadFromFile(*aResult,aNameFile);
        // -3-   Read the initial sensor
        std::string aNameInit  = aDirInit + aResult->Data().mNameFile;

        cSensorImage *  aSI =  AllocAutoSensorFromFile(aNameInit,aNameImage,false);
        aResult->SetSensorInit(aSI);
        return aResult;
   }

    return nullptr;
}

cSensorImage * cSensorImage::AllocExternalSensor(const std::string & aDirInit,const std::string & aDirSens,const std::string aNameImage)
{
    cSensorImage * aRes = nullptr;

    if (aRes==nullptr) aRes =  GenAllocExternalSensor<cExternalSensor>(aDirInit,aDirSens,aNameImage);

    return aRes;
}




/* =============================================== */
/*                                                 */
/*                 cAppliImportPushbroom           */
/*                                                 */
/* =============================================== */

class cAppliImportPushbroom : public cMMVII_Appli
{
     public :

        cAppliImportPushbroom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

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

        // --- Optionnal ----
        std::string mNameSensorOut;

     // --- Internal ----
};

cAppliImportPushbroom::cAppliImportPushbroom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this)
{
}


cCollecSpecArg2007 & cAppliImportPushbroom::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return anArgObl
      <<   Arg2007(mNameImagesIn,"Name of input sensor gile", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      <<   Arg2007(mPatChgName,"[PatNameIm,NameSens]", {{eTA2007::ISizeV,"[2,2]"}})
      <<   mPhProj.DPOrient().ArgDirOutMand()
   ;
}

cCollecSpecArg2007 & cAppliImportPushbroom::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameSensorOut,CurOP_Out,"Name of output file if correction are done")
   ;
}

std::vector<std::string>  cAppliImportPushbroom::Samples() const
{
   return {
              "MMVII ImportPushbroom 'SPOT_1.*tif' '[SPOT_(.*).tif,RPC_$1.xml]'"
	};
}

void  cAppliImportPushbroom::ImportOneImage(const std::string & aNameIm)
{
    std::string aFullNameSensor = ReplacePattern(mPatChgName.at(0),mPatChgName.at(1),aNameIm);
    std::string aNameSensor = FileOfPath(aFullNameSensor,false);

    
    CopyFile(aNameSensor,mPhProj.DirImportInitOri()+aNameSensor);

    StdOut() << "NameSensor=" << aNameIm << " => " << aNameSensor << "\n";

    cAnalyseTSOF  anAnalyse (aNameSensor);
    cSensorImage *  aSensorInit =  AllocAutoSensorFromFile(anAnalyse ,aNameIm);
    cSensorImage * aSensorEnd = new cExternalSensor(anAnalyse.mData,aNameIm,aSensorInit);
    anAnalyse.FreeAnalyse();

    StdOut() << "NAMEORI=[" << aSensorEnd->NameOriStd()  << "]\n";

    mPhProj.SaveSensor(*aSensorEnd);

    aSensorEnd->ToFile("toto_"+aSensorEnd->NameOriStd());

    delete aSensorEnd;
}


int cAppliImportPushbroom::Exe()
{
    mPhProj.FinishInit();

    for (const auto & aNameIm :  VectMainSet(0))
    {
         ImportOneImage(aNameIm);
    }

    // TestRPCProjections(mNameSensorIn);

    return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_ImportPushbroom(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliImportPushbroom(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecImportPushbroom
(
     "ImportPushbroom",
      Alloc_ImportPushbroom,
      "Import a pushbroom sensor",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

};
