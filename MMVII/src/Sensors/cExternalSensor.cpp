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
/*                 cAnalyseTSOF                    */
/*                                                 */
/* =============================================== */

void AddData(const  cAuxAr2007 & anAux,cDataExternalSensor & aDES)
{
    AddData(cAuxAr2007("NameFileInit",anAux),aDES.mNameFile);
    EnumAddData(anAux,aDES.mType,"TypeSensor");
    EnumAddData(anAux,aDES.mFormat,"FileFormat");
}


cDataExternalSensor::cDataExternalSensor(const std::string& aNameFile) :
   mNameFile (aNameFile),
   mType     (eTypeSensor::eNbVals),
   mFormat   (eFormatSensor::eNbVals)
{
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
/*                 cAppliTestSensor                */
/*                                                 */
/* =============================================== */

/**  A basic application for  */

class cAppliTestSensor : public cMMVII_Appli
{
     public :

        cAppliTestSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;

	///  Test that the accuracy of ground truth, i.e Proj(P3) = P2
        void TestGroundTruth(const  cSensorImage & aSI) const;
	///  Test coherence of Direct/Inverse model, i.e Id = Dir o Inv = Inv o Dir
        void TestCoherenceDirInv(const  cSensorImage & aSI) const;

        cPhotogrammetricProject  mPhProj;
        std::string              mNameImage;
        std::string              mNameRPC;
        bool                     mShowDetail;

};

std::vector<std::string>  cAppliTestSensor::Samples() const
{
   return {
              "MMVII TestSensor SPOT_1B.tif RPC_1B.xml InPointsMeasure=XingB"
	};
}

cAppliTestSensor::cAppliTestSensor(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mShowDetail  (false)
{
}

cCollecSpecArg2007 & cAppliTestSensor::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             << Arg2007(mNameImage,"Name of input Image", {eTA2007::FileDirProj})
             << Arg2007(mNameRPC,"Name of input RPC", {eTA2007::Orient})
      ;
}

cCollecSpecArg2007 & cAppliTestSensor::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << mPhProj.DPPointsMeasures().ArgDirInOpt()
               << AOpt2007(mShowDetail,"ShowD","Show detail",{eTA2007::HDV})
            ;
}

void cAppliTestSensor::TestGroundTruth(const  cSensorImage & aSI) const
{
    // Load mesure from standard MMVII project
    cSetMesImGCP aSetMes;
    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,mNameImage);
    cSet2D3D aSetM23;
    aSetMes.ExtractMes1Im(aSetM23,mNameImage);

    cStdStatRes  aStCheckIm;  //  Statistic of reproj errorr

    for (const auto & aPair : aSetM23.Pairs()) // parse all pair to accumulate stat of errors
    {
         cPt3dr  aPGr = aPair.mP3;
         cPt2dr  aPIm = aSI.Ground2Image(aPGr);
	 tREAL8 aDifIm = Norm2(aPIm-aPair.mP2);
	 aStCheckIm.Add(aDifIm);

         if (mShowDetail) 
         {
             StdOut()  << "ImGT=" <<  aDifIm << std::endl;

         }
    }
    StdOut() << "  ==============  Accuracy / Ground trurh =============== " << std::endl;
    StdOut()  << "    Avg=" <<  aStCheckIm.Avg() << ",  Worst=" << aStCheckIm.Max() << "\n";

}

void cAppliTestSensor::TestCoherenceDirInv(const  cSensorImage & aSI) const
{
     bool  InDepth = ! aSI.HasIntervalZ();  // do we use Im&Depth or Image&Z

     cPt2dr aIntZD = cPt2dr(1,2);
     if (InDepth)
     {  // if depth probably doent matter which one is used
     }
     else
     {
        aIntZD = aSI.GetIntervalZ(); // at least with RPC, need to get validity interval
     }

     int mNbByDim = 10;
     int mNbDepth = 5;
     cSet2D3D  aS32 = aSI.SyntheticsCorresp3D2D(mNbByDim,mNbDepth,aIntZD.x(),aIntZD.y(),InDepth);

     cStdStatRes  aStConsistIm;  // stat for image consit  Proj( Proj-1(PIm)) ?= PIm
     cStdStatRes  aStConsistGr;  // stat for ground consist  Proj-1 (Proj(Ground)) ?= Ground

     for (const auto & aPair : aS32.Pairs())
     {
         cPt3dr  aPGr = aPair.mP3;
         cPt3dr  aPIm (aPair.mP2.x(),aPair.mP2.y(),aPGr.z());

         cPt3dr  aPIm2 ;
         cPt3dr  aPGr2 ;
	
	 if (InDepth)
	 {
	    aPIm2 = aSI.Ground2ImageAndDepth(aSI.ImageAndDepth2Ground(aPIm));
	    aPGr2 = aSI.ImageAndDepth2Ground(aSI.Ground2ImageAndDepth(aPGr));
	 }
	 else
	 {
	    aPIm2 = aSI.Ground2ImageAndZ(aSI.ImageAndZ2Ground(aPIm));
	    aPGr2 = aSI.ImageAndZ2Ground(aSI.Ground2ImageAndZ(aPGr));
	 }
	 tREAL8 aDifIm = Norm2(aPIm-aPIm2);
	 aStConsistIm.Add(aDifIm);

	 tREAL8 aDifGr = Norm2(aPGr-aPGr2);
	 aStConsistGr.Add(aDifGr);
	
     }

     StdOut() << "  ==============  Consistencies Direct/Inverse =============== " << std::endl;
     StdOut() << "     * Image :  Avg=" <<   aStConsistIm.Avg() 
	                 <<  ", Worst=" << aStConsistIm.Max()  
	                 <<  ", Med=" << aStConsistIm.ErrAtProp(0.5)  
                         << std::endl;

     StdOut() << "     * Ground:  Avg=" <<   aStConsistGr.Avg() 
	                 <<  ", Worst=" << aStConsistGr.Max()  
	                 <<  ", Med=" << aStConsistGr.ErrAtProp(0.5)  
			 << std::endl;

}



int cAppliTestSensor::Exe()
{
    mPhProj.FinishInit();
    cSensorImage *  aSI =  AllocAutoSensorFromFile(mNameRPC,mNameImage);

    if (mPhProj.DPPointsMeasures().DirInIsInit())
       TestGroundTruth(*aSI);

    TestCoherenceDirInv(*aSI);

    StdOut() << "NAMEORI=[" << aSI->NameOriStd()  << "]\n";

    delete aSI;

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestImportSensors(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliTestSensor(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecTestImportSensors
(
     "TestSensor",
      Alloc_TestImportSensors,
      "Test orientation functions : coherence Direct/Inverse, ground truth 2D/3D correspondance",
      {eApF::Ori},
      {eApDT::Ori,eApDT::GCP},
      {eApDT::Console},
      __FILE__
);


/* =============================================== */
/*                                                 */
/*                 cExternalSensorMod2D            */
/*                                                 */
/* =============================================== */

class cExternalSensorMod2D : public cSensorImage
{
      public :
         cExternalSensorMod2D(const cDataExternalSensor & aData,const std::string& aNameImage,cSensorImage * aSI);
         virtual ~cExternalSensorMod2D();
         void AddData(const  cAuxAr2007 & anAux);

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

	 // ====  Method to override in derived classes  ===== 
	 virtual cPt2dr  Init2End (const cPt2dr & aP0) const ;
	 virtual cPt2dr  End2Init (const cPt2dr & aP0) const ;
	 virtual  std::string NameModif2D() const;

         virtual void AddDataComplem(const  cAuxAr2007 & anAux);
};

   // ================  Constructor/Destructor ====================

cExternalSensorMod2D::cExternalSensorMod2D(const cDataExternalSensor & aData,const std::string& aNameImage,cSensorImage * aSI) :
     cSensorImage  (aNameImage),
     mData         (aData),
     mSensorInit   (aSI)
{
}

cExternalSensorMod2D::~cExternalSensorMod2D() 
{
    delete mSensorInit;
}
   
     // ==============   READ/WRITE/SERIAL ================
   
std::string  cExternalSensorMod2D::V_PrefixName() const
{
	return mSensorInit->V_PrefixName() + "_Modif2D_" +  NameModif2D() ;
}

void cExternalSensorMod2D::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("General",anAux),mData);
     AddDataComplem(cAuxAr2007("Model2D",anAux));
}

void AddData(const  cAuxAr2007 & anAux,cExternalSensorMod2D & aDES)
{
     aDES.AddData(anAux);
}

void cExternalSensorMod2D::AddDataComplem(const  cAuxAr2007 & anAux)
{
     std::string anIdent("Identity");
     MMVII::AddData(anAux,anIdent);
}

void cExternalSensorMod2D::ToFile(const std::string & aNameFile) const 
{
     SaveInFile(const_cast<cExternalSensorMod2D &>(*this),aNameFile);
}


     // ==============  Default method to override  ================


cPt2dr  cExternalSensorMod2D::Init2End (const cPt2dr & aP0) const {return aP0; }

/// Maybe to change with a basic fix point method
cPt2dr  cExternalSensorMod2D::End2Init (const cPt2dr & aP0) const 
{
    return aP0; 
}
std::string cExternalSensorMod2D::NameModif2D() const {return "Ident";}


     // =============   METHOD FOR BEING a cSensorImage =====================

tSeg3dr  cExternalSensorMod2D::Image2Bundle(const cPt2dr & aP) const 
{
	return mSensorInit->Image2Bundle(End2Init(aP));
}

cPt2dr cExternalSensorMod2D::Ground2Image(const cPt3dr & aPGround) const 
{
	return Init2End(mSensorInit->Ground2Image(aPGround));
}

cPt3dr cExternalSensorMod2D::ImageAndZ2Ground(const cPt3dr & aPE) const 
{
    cPt2dr aPI = End2Init(cPt2dr(aPE.x(),aPE.y()));

    return mSensorInit->ImageAndZ2Ground(cPt3dr(aPI.x(),aPI.y(),aPE.z()));
}

double cExternalSensorMod2D::DegreeVisibility(const cPt3dr & aPGround) const
{
	return mSensorInit->DegreeVisibility(aPGround);
}

bool  cExternalSensorMod2D::HasIntervalZ()  const {return mSensorInit->HasIntervalZ();}
cPt2dr cExternalSensorMod2D::GetIntervalZ() const {return mSensorInit->GetIntervalZ();}

//  for small deformation , the pixel domain is the same than the init sensor
const cPixelDomain & cExternalSensorMod2D::PixelDomain() const 
{
	return mSensorInit->PixelDomain();
}

cPt3dr  cExternalSensorMod2D::PseudoCenterOfProj() const {return mSensorInit->PseudoCenterOfProj();}




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
    cSensorImage * aSensorEnd = new cExternalSensorMod2D(anAnalyse.mData,aNameIm,aSensorInit);
    anAnalyse.FreeAnalyse();

    StdOut() << "NAMEORI=[" << aSensorEnd->NameOriStd()  << "]\n";

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
