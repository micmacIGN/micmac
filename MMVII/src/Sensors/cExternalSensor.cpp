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

/* =============================================== */
/*                                                 */
/*                 cExternalSensorModif2D          */
/*                                                 */
/* =============================================== */


class cExternalSensorModif2D : public cExternalSensor
{
     public :
	 static std::string  StaticPrefixName();
         cExternalSensorModif2D(const cDataExternalSensor & aData,const std::string& aNameImage,cSensorImage * aSI,int aDegree) ;
         void AddData(const  cAuxAr2007 & anAux);

	 void PerturbateRandom(tREAL8 anAmpl,bool Show);

	 // NS_SymbolicDerivative::cCalculator<double> * EqDistPol2D(int  aDeg,bool WithDerive,int aSzBuf,bool ReUse); // PUSHB

     private :
        void InitPol2D();

	 // ====  Methods overiiding for being a cSensorImage =====
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
         ///    Method specialized, more efficent than using bundles
         cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;

         void ToFile(const std::string &) const override;
         std::string  V_PrefixName() const  override;

	 // ====  Method to override in derived classes  ===== 

	 /// For "initial" coordinat (pixel) to "final"
	 cPt2dr  Init2End (const cPt2dr & aP0) const ;
	 /// For "final" to "initial" coordinat (pixel)
	 cPt2dr  End2Init (const cPt2dr & aP0) const ;

	 int                     mDegree;
	 std::vector<tREAL8>     mVParams;
	 cCalculator<double> *   mEqIma2End;  // Functor that gives the distorstion
};

    //------------------------ Create/Read/Write   ----------------------------------

cExternalSensorModif2D::cExternalSensorModif2D
(
      const cDataExternalSensor & aData,
      const std::string& aNameImage,
      cSensorImage * aSI,
      int            aDegree=-1
)  :
   cExternalSensor(aData,aNameImage,aSI),
   mDegree  (aDegree),
   mEqIma2End   (nullptr)
{
    if (mDegree>=0)
       InitPol2D();
}

void cExternalSensorModif2D::InitPol2D()
{
    mEqIma2End   =  EqDistPol2D(mDegree,false/*W/O derive*/,1,true/*Recycling mode*/) ;
    std::vector<cDescOneFuncDist>  aVDesc =  Polyn2DDescDist(mDegree);
    mVParams.resize(aVDesc.size(),0.0);
}




void cExternalSensorModif2D::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("General",anAux),mData);
     MMVII::AddData(cAuxAr2007("Degree",anAux),mDegree);
     
     if (anAux.Ar().Input())
     {
         InitPol2D();
     }

     {
         cAuxAr2007  anAuxCoeff("Coeffs",anAux);
         std::vector<cDescOneFuncDist>  aVDesc =  Polyn2DDescDist(mDegree);
         for (size_t aK=0 ; aK<mVParams.size() ; aK++)
         {
             MMVII::AddData(cAuxAr2007(aVDesc[aK].mName,anAuxCoeff),mVParams[aK]);
         }
     }
}
void AddData(const  cAuxAr2007 & anAux,cExternalSensorModif2D & aExtSM2d)
{
     aExtSM2d.AddData(anAux);
}

void cExternalSensorModif2D::ToFile(const std::string & aNameFile) const 
{
     SaveInFile(const_cast<cExternalSensorModif2D &>(*this),aNameFile);
}

std::string  cExternalSensorModif2D::StaticPrefixName() { return  "ExtSensModifPol2D";}
std::string  cExternalSensorModif2D::V_PrefixName() const  {return StaticPrefixName();}


      //------------------------  Fundemantal methods : 3D/2D correspondance ---------------------------

tSeg3dr  cExternalSensorModif2D::Image2Bundle(const cPt2dr & aPixIm) const 
{
	return   mSensorInit->Image2Bundle(Init2End(aPixIm));
}
cPt2dr cExternalSensorModif2D::Ground2Image(const cPt3dr & aPGround) const
{
	return  End2Init(mSensorInit->Ground2Image(aPGround));
}

cPt3dr cExternalSensorModif2D::ImageAndZ2Ground(const cPt3dr & aPxyz) const 
{
     cPt2dr aPCor = Init2End(cPt2dr(aPxyz.x(),aPxyz.y()));

     return mSensorInit->ImageAndZ2Ground(cPt3dr(aPCor.x(),aPCor.y(),aPxyz.z()));
}


      //------------------------ 2D deformation -----------------------------------------

cPt2dr  cExternalSensorModif2D::End2Init (const cPt2dr & aP0) const
{
     std::vector<tREAL8>  aVXY = aP0.ToStdVector();
     std::vector<tREAL8>  aDistXY =  mEqIma2End->DoOneEval(aVXY,mVParams);
     return cPt2dr::FromStdVector(aDistXY);
}

cPt2dr  cExternalSensorModif2D::Init2End (const cPt2dr & aPInit) const
{
    // For now implement a very basic "fix point" method for inversion
     tREAL8 aThresh = 1e-3;
     int aNbIterMax = 10;

     cPt2dr aPEnd = aPInit;
     bool GoOn = true;
     int aNbIter = 0;
     while (GoOn)
     {
	  cPt2dr aNewPInit  = End2Init(aPEnd);
	  // We make the taylor expansion assume End2Init ~Identity
	  // End2Init (aPEnd + aPInit - aNewPInit) ~ End2Init (aPEnd) + aPInit - aNewPInit = aNewPInit +aPInit - aNewPInit
	  aPEnd += aPInit - aNewPInit;
          aNbIter++;
	  GoOn = (aNbIter<aNbIterMax) && (Norm2(aNewPInit-aPInit)>aThresh);
     }
     // StdOut() << "NBITER=" << aNbIter << "\n";
     return aPEnd;
}

void cExternalSensorModif2D::PerturbateRandom(tREAL8 anAmpl,bool Show)
{
    anAmpl /= mVParams.size();
    tREAL8 aNorm = Norm2(Sz());

    std::vector<cDescOneFuncDist>  aVDesc =  Polyn2DDescDist(mDegree);
    for (size_t aK=0 ; aK<mVParams.size() ; aK++)
    {
        mVParams[aK] = (RandUnif_C() * anAmpl ) / std::pow(aNorm,aVDesc[aK].mDegTot-1);
    }

    if (Show)
    {
       for (int aK=0 ; aK<20 ; aK++)
       {
          cPt2dr aP0 = MulCByC(ToR(Sz()) ,cPt2dr::PRand());

          cPt2dr aP1 = End2Init(aP0);
          cPt2dr aP2 = Init2End(aP1);

          StdOut() << aP0 << aP1-aP0  << aP2 -aP0 << "\n";
       }
    }
}

/* =============================================== */
/*                                                 */
/*                 cSensorImage                    */
/*                                                 */
/* =============================================== */


template <class TypeSens>  TypeSens * GenAllocExternalSensor
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
        TypeSens *  aResult = new TypeSens (cDataExternalSensor(),aNameImage,nullptr);
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

    // Try the existence  of different  

    if (aRes==nullptr) aRes =  GenAllocExternalSensor<cExternalSensor>(aDirInit,aDirSens,aNameImage);
    if (aRes==nullptr) aRes =  GenAllocExternalSensor<cExternalSensorModif2D>(aDirInit,aDirSens,aNameImage);


    return aRes;
}



/* =============================================== */
/*                                                 */
/*                 cAppliImportExternSensor        */
/*                                                 */
/* =============================================== */

class cAppliImportExternSensor : public cMMVII_Appli
{
     public :

        cAppliImportExternSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

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
	int         mDegreeCorr;
	tREAL8      mRanPert;

     // --- Internal ----
};

cAppliImportExternSensor::cAppliImportExternSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this)
{
}


cCollecSpecArg2007 & cAppliImportExternSensor::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return anArgObl
      <<   Arg2007(mNameImagesIn,"Name of input sensor gile", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      <<   Arg2007(mPatChgName,"[PatNameIm,NameSens]", {{eTA2007::ISizeV,"[2,2]"}})
      <<   mPhProj.DPOrient().ArgDirOutMand()
   ;
}

cCollecSpecArg2007 & cAppliImportExternSensor::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mDegreeCorr,"DegCor","Degree of correction for sensors")
           << AOpt2007(mRanPert,"RandomPerturb","Random perturbation of initial 2D-Poly",{eTA2007::Tuning})
   ;
}

std::vector<std::string>  cAppliImportExternSensor::Samples() const
{
   return {
              "MMVII ImportPushbroom AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init",
              "MMVII ImportPushbroom AllIm.xml '[SPOT_(.*).tif,RPC_$1.xml]' SPOT_Init"
	};
}

void  cAppliImportExternSensor::ImportOneImage(const std::string & aNameIm)
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
    cSensorImage *  aSensorInit =  AllocAutoSensorFromFile(anAnalyse ,aNameIm);

    // Encapsulate this initial sensor with eventually coefficient
    cSensorImage * aSensorEnd = nullptr;
    if  (IsInit(&mDegreeCorr))
    {
        cExternalSensorModif2D * aSensor2D = new cExternalSensorModif2D(anAnalyse.mData,aNameIm,aSensorInit,mDegreeCorr);
	if (IsInit(&mRanPert))
	   aSensor2D->PerturbateRandom(mRanPert,false);
        aSensorEnd = aSensor2D;
    }
    else
    {
        aSensorEnd = new cExternalSensor(anAnalyse.mData,aNameIm,aSensorInit);
    }
    // free the memory of Analyse (is not automatic, because can be copied) 
    anAnalyse.FreeAnalyse();
    // Save the result
    mPhProj.SaveSensor(*aSensorEnd);

    StdOut() << "NAMES    Ima: " << aNameIm << " SensInit : " << aNameSensor  << " SensSave : " << aSensorEnd->NameOriStd() << "\n";

    //  Free the sensor (was not allocated by PhProj, will not be automatically deleted)
    delete aSensorEnd;
}


int cAppliImportExternSensor::Exe()
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

tMMVII_UnikPApli Alloc_ImportExtSens(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliImportExternSensor(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecImportExtSens
(
     "ImportExtSens",
      Alloc_ImportExtSens,
      "Import an External Sensor",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

};
