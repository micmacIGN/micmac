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

bool  DEBUG_I2B=false;


    //------------------------ Create/Read/Write   ----------------------------------

cExternalSensorModif2D::cExternalSensorModif2D
(
      const cDataEmbededSensor & aData,
      const std::string& aNameImage,
      cSensorImage * aSI,
      int            aDegree,
      const std::string & aTargetSysCo
)  :
   cExternalSensor  (aData,aNameImage,aSI),
   mDegree          (aDegree),
   mActiveDistG2I   (true),
   mTargetSysCo     (aTargetSysCo),
   mEqIma2End       (nullptr),
   mSysI2F          {},
   mPtEpsDeriv      (1,1,1),
   mIdChSys         (true)
{
    if (mDegree>=0)
       InitPol2D();
}

/*
template <class Type> Type * GetSetRemanentObj(const std::string & aName,Type * aVal=nullptr)
{
    static std::map<std::string > TheMap;
    if (aVal!=nullptr) return aVal;

    Type * & aRes = TheMap[aName];
    if (aRes==nullptr)
    {
    }
}
*/


cExternalSensorModif2D * cExternalSensorModif2D::TryRead
                  (
                        const cPhotogrammetricProject & aPhProj,
                        const std::string & aNameImage
                  )
{
   std::string  aDirSens = aPhProj.DPOrient().FullDirIn();
   std::string aNameFile = aDirSens + cSensorImage::NameOri_From_PrefixAndImage(cExternalSensorModif2D::StaticPrefixName(),aNameImage);

   if (!ExistFile(aNameFile))
      return nullptr;

   std::string  aDirInit = aPhProj.DirImportInitOri();
   // -1-   Create the object
   cExternalSensorModif2D *  aResult = new cExternalSensorModif2D (cDataEmbededSensor(),aNameImage);
   // -2-   Read the data contained in the file
   ReadFromFile(*aResult,aNameFile);
   // -3-   Read the initial sensor
   std::string aNameInit  = aDirInit + aResult->Data().mNameFileInit;
   cSensorImage *  aSI =  CreateAutoExternalSensor(aNameInit,aNameImage,false);
   aResult->SetSensorInit(aSI);
   aResult->Finish(aPhProj);

   return aResult;
}

cExternalSensorModif2D::~cExternalSensorModif2D()
{
}

void cExternalSensorModif2D::InitPol2D()
{
    mEqIma2End   =  EqDistPol2D(mDegree,false/*W/O derive*/,1,true/*Recycling mode*/) ;
    std::vector<cDescOneFuncDist>  aVDesc =  Polyn2DDescDist(mDegree);
    mVParams.resize(aVDesc.size(),0.0);
}


void cExternalSensorModif2D::Finish(const cPhotogrammetricProject & aPhP)
{
     mSysI2F = aPhP.ChangSys(mData.mSysCoOri,mTargetSysCo);
     mIdChSys = mSysI2F.IsIdent();

     tPtrSysCo aSysTarget = aPhP.ReadSysCo(mTargetSysCo);
     mPtEpsDeriv =  aSysTarget->mPtEpsDeriv;
}

cPt3dr  cExternalSensorModif2D::EpsDiffGround2Im(const cPt3dr &) const {return mPtEpsDeriv;}


tProjImAndGrad  cExternalSensorModif2D::DiffGround2Im(const cPt3dr & aP) const
{
     // MMVII_DEV_WARNING("ExternalSensorModif2D::DiffGround2Im");
     // return cExternalSensor::DiffGround2Im(aP);

	/*
     if (mIdChSys)  
        return cExternalSensor::DiffGround2Im(aP);
	*/

     return DiffG2IByFiniteDiff(aP);
}



void cExternalSensorModif2D::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("InitialSensor",anAux),mData);
     MMVII::AddData(cAuxAr2007("FinalCoordSys",anAux),mTargetSysCo);
     MMVII::AddData(cAuxAr2007("DegreeCorrection",anAux),mDegree);
     
     if (anAux.Ar().Input())
     {
         InitPol2D();
     }

     {
         cAuxAr2007  anAuxCoeff("CoeffsCorrection",anAux);
         std::vector<cDescOneFuncDist>  aVDesc =  Polyn2DDescDist(mDegree);
         for (size_t aK=0 ; aK<mVParams.size() ; aK++)
         {
             MMVII::AddData(cAuxAr2007(aVDesc[aK].mName,anAuxCoeff),mVParams[aK]);
         }
StdOut() << "AdddDatatat " << mVParams <<  " XXXX=" << this << "\n";
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


      //------------------------  bundles ---------------------------
cCalculator<double> * cExternalSensorModif2D::CreateEqColinearity(bool WithDerive,int aSzBuf,bool ReUse) 
{
    return EqColinearityCamGen(mDegree,WithDerive,aSzBuf,ReUse);
}

void cExternalSensorModif2D::PutUknowsInSetInterval() 
{
     mSetInterv->AddOneInterv(mVParams);
}

void cExternalSensorModif2D::PushOwnObsColinearity(std::vector<double> & aVParam,const cPt3dr & aPGround)
{
   // "IObs","JObs",  "P0x","P0y","P0z"    "IP0","JP0",    "dIdX","dIDY","dIdZ",   "dJdX","dJDY","dJdZ"

// cPt2dr aP000 = cPt2dr::FromStdVector(aVParam);
// StdOut() << "VPPPP " << aP000 << aPGround << "\n"; 


    aPGround.PushInStdVector(aVParam);
    //  inibate dist correction to compute diff
    mActiveDistG2I=false;
    tProjImAndGrad aProj = DiffGround2Im(aPGround);
    mActiveDistG2I=true;
    aProj.mPIJ.PushInStdVector(aVParam);
    aProj.mGradI.PushInStdVector(aVParam);
    aProj.mGradJ.PushInStdVector(aVParam);


// StdOut() << "DIFFFF " << aProj.mPIJ  << aProj.mGradI << aProj.mGradJ << "\n";
// getchar();
}
         // cChangSysCoordV2  ChangSys(const std::vector<std::string> &,tREAL8 aEpsDif=0.1);


      //------------------------  Fundemantal methods : 3D/2D correspondance ---------------------------

tSeg3dr  cExternalSensorModif2D::Image2Bundle(const cPt2dr & aPixIm) const 
{
if (DEBUG_I2B)
{
	StdOut() << "I2BBB CORREC " << aPixIm - InitPix2Correc(aPixIm) << " P="  << mVParams << " XXXX=" << this << "\n";
}
	tSeg3dr aSeg =   mSensorInit->Image2Bundle(InitPix2Correc(aPixIm));

	return tSeg3dr(mSysI2F.Value(aSeg.P1()), mSysI2F.Value(aSeg.P2()));
}
cPt2dr cExternalSensorModif2D::Ground2Image(const cPt3dr & aPGround) const
{
      //StdOut()  << "GGGG " << aPGround << " " <<  mSysI2F.Value(aPGround) << mSysI2F.Inverse(aPGround) << "\n";
      cPt2dr aP_WO_Corr = mSensorInit->Ground2Image(mSysI2F.Inverse(aPGround));

      if ( mActiveDistG2I)
         return  Correc2InitPix(aP_WO_Corr);

      return aP_WO_Corr;
}

cPt3dr cExternalSensorModif2D::ImageAndZ2Ground(const cPt3dr & aPxyz) const 
{
     return cSensorImage::ImageAndZ2Ground(aPxyz);
     if (0)
     {
	  cPt2dr aP0 (aPxyz.x(),aPxyz.y());
	  cPt2dr aP1 = InitPix2Correc(aP0);
	  cPt2dr aP2 = Correc2InitPix(aP1);

          StdOut() << "cExternalSensorModif2D::ImageAndZ2Ground " << aP0-aP1 << " " << aP0 - aP2 << "\n";
     }

     cPt2dr aPCor = InitPix2Correc(cPt2dr(aPxyz.x(),aPxyz.y()));

     return cSensorImage::ImageAndZ2Ground(cPt3dr(aPCor.x(),aPCor.y(),aPxyz.z()));

     //  return  mSysI2F.Value(mSensorInit->ImageAndZ2Ground(cPt3dr(aPCor.x(),aPCor.y(),aPxyz.z())));
}

double cExternalSensorModif2D::DegreeVisibility(const cPt3dr & aPGround) const
{
     return mSensorInit->DegreeVisibility(mSysI2F.Inverse(aPGround));
}



      //------------------------ 2D deformation -----------------------------------------

cPt2dr  cExternalSensorModif2D::Correc2InitPix (const cPt2dr & aP0) const
{
     std::vector<tREAL8>  aVXY = aP0.ToStdVector();
     std::vector<tREAL8>  aDistXY =  mEqIma2End->DoOneEval(aVXY,mVParams);
     return cPt2dr::FromStdVector(aDistXY);
}

cPt2dr  cExternalSensorModif2D::InitPix2Correc (const cPt2dr & aPInit) const
{
    // For now implement a very basic "fix point" method for inversion
     tREAL8 aThresh = 1e-3;
     int aNbIterMax = 10;

     cPt2dr aPEnd = aPInit;
     bool GoOn = true;
     int aNbIter = 0;
     while (GoOn)
     {
	  cPt2dr aNewPInit  = Correc2InitPix(aPEnd);
	  // We make the taylor expansion assume Correc2InitPix ~Identity
	  // Correc2InitPix (aPEnd + aPInit - aNewPInit) ~ Correc2InitPix (aPEnd) + aPInit - aNewPInit = aNewPInit +aPInit - aNewPInit
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

          cPt2dr aP1 = Correc2InitPix(aP0);
          cPt2dr aP2 = InitPix2Correc(aP1);

          StdOut() << aP0 << aP1-aP0  << aP2 -aP0 << "\n";
       }
    }
}

/* =============================================== */
/*                                                 */
/*                 cSensorImage                    */
/*                                                 */
/* =============================================== */



cSensorImage * cPhotogrammetricProject::AllocExternalSensor
               (
	            const std::string & aDirInit,
		    const std::string & aDirSens,
		    const std::string aNameImage
		    )
{
    cSensorImage * aRes = nullptr;

    if (aRes==nullptr) aRes =  cExternalSensor::TryRead(aDirInit,aDirSens,aNameImage);
    if (aRes==nullptr) aRes =  cExternalSensorModif2D::TryRead(*this,aNameImage);

    return aRes;
}

cSensorImage * cPhotogrammetricProject::AllocExternalSensor(const std::string aNameImage)
{
     return AllocExternalSensor(DirImportInitOri(),mDPOrient.FullDirIn(),aNameImage);
}


/* =============================================== */
/*                                                 */
/*                 cAppliImportExternSensor        */
/*                                                 */
/* =============================================== */

class cAppliParametrizeSensor : public cMMVII_Appli
{
     public :

        cAppliParametrizeSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

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
	int          mDegreeCorr;
	tREAL8       mRanPert;
	std::string  mTargetSysCo;

     // --- Internal ----
};

cAppliParametrizeSensor::cAppliParametrizeSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this)
{
}


cCollecSpecArg2007 & cAppliParametrizeSensor::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return anArgObl
      <<   Arg2007(mNameImagesIn,"Name of input sensor gile", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      <<   mPhProj.DPOrient().ArgDirInMand()
      <<   mPhProj.DPOrient().ArgDirOutMand()
      <<   Arg2007(mDegreeCorr,"Degree of correction for sensors")
   ;
}

cCollecSpecArg2007 & cAppliParametrizeSensor::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mRanPert,"RandomPerturb","Random perturbation of initial 2D-Poly",{eTA2007::Tuning})
           << AOpt2007(mTargetSysCo,"TargetSysCo","Targeted system of coord when != init")
   ;
}

std::vector<std::string>  cAppliParametrizeSensor::Samples() const
{
   return {
              "MMVII OriParametrizeSensor AllIm.xml ..."
	};
}

void  cAppliParametrizeSensor::ImportOneImage(const std::string & aNameIm)
{
    // read the external sensor
    cExternalSensor * anExtS =  cExternalSensor::TryRead(mPhProj.DirImportInitOri(),mPhProj.DPOrient().FullDirIn(),aNameIm);
    if (anExtS==nullptr)
    {
	    MMVII_UnclasseUsEr("Sensor was not imported for " + aNameIm);
    }
    cDataEmbededSensor aData = anExtS->Data();

    // If a coordinate system was specified 
    SetIfNotInit(mTargetSysCo,aData.mSysCoOri);
    // Test that coordinate system is valid
    mPhProj.ReadSysCo(mTargetSysCo);

    cExternalSensorModif2D * aSensor2D = new cExternalSensorModif2D(aData,aNameIm,anExtS->SensorInit(),mDegreeCorr,mTargetSysCo);
    aSensor2D->Finish(mPhProj);
    cMMVII_Appli::AddObj2DelAtEnd(aSensor2D); // dont destroy now, maybe it will be used in next versions


    // Eventually add noise for simulation
    if (IsInit(&mRanPert))
       aSensor2D->PerturbateRandom(mRanPert,false);

    // Save the result
    mPhProj.SaveSensor(*aSensor2D);
    // StdOut() << "NAMES    Ima: " << aNameIm << " SensInit : " << aNameSensor  << " SensSave : " << aSensorEnd->NameOriStd() << "\n";
   
    // A bit touchy but the SensorInit() is now owned by aSensor2D, so we supress the ref in anExtS before deleting it
    anExtS->SetSensorInit(nullptr);
    delete anExtS;
}


int cAppliParametrizeSensor::Exe()
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

tMMVII_UnikPApli Alloc_ParametrizeSensor(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliParametrizeSensor(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecParametrizeSensor
(
     "OriParametrizeSensor",
      Alloc_ParametrizeSensor,
      "Import an External Sensor",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

#if (0)
#endif

};
