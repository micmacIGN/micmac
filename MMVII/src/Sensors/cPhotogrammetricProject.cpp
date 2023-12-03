#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_BlocRig.h"


/**
   \file  cPhotogrammetricProject.cpp

   \brief file for handling names/upload/download of photogram data (pose,calib, ...)

   test Git
*/


namespace MMVII
{

std::string SuppressDirFromNameFile(const std::string & aDir,const std::string & aName,bool ByDir)
{
    // mOriIn.starts_with(aDir);  -> C++20
    // to see if StringDirSeparator() is not a meta carac on window ?

     std::string aPat =  "(.*" + aDir+")?" + "([A-Za-z0-9_-]+)";
     if (ByDir)
         aPat = aPat + "[\\/]?";
     else
	 aPat = aPat + "\\." +  GlobTaggedNameDefSerial()  ;
     if (! MatchRegex(aName,aPat))
     {
         MMVII_UsersErrror
         (
             eTyUEr::eUnClassedError,
             "SuppressDirFromNameFile:No match for subdir, with name=" + aName + " Dir=" + aDir
         );
     }
     std::string aRes =  ReplacePattern(aPat,"$2",aName);
	     
     return aRes;
}

   /* ********************************************************** */
   /*                                                            */
   /*                       cDirsPhProj                          */
   /*                                                            */
   /* ********************************************************** */

        //   ======================  creation =======================================

cDirsPhProj::cDirsPhProj(eTA2007 aMode,cPhotogrammetricProject & aPhp):
   mMode           (aMode),
   mPhp            (aPhp),
   mAppli          (mPhp.Appli()),
   mPrefix         (E2Str(mMode)),
   mDirLocOfMode   (MMVII_DirPhp + mPrefix + StringDirSeparator()),
   mPurgeOut       (false)
{
}

const std::string &  cDirsPhProj::DirLocOfMode() const { return mDirLocOfMode; }

void cDirsPhProj::Finish()
{
    //  Allow user to specify indiferrently short name of full name, will extract short name
    // for ex :   "MMVII-PhgrProj/Orient/Test/" ,  "Test/",  "Test" ...  =>   "Test"
    //
    if (mAppli.IsInit(&mDirIn))  // dont do it if mDirIn not used ...
        mDirIn  = SuppressDirFromNameFile(mDirLocOfMode,mDirIn,true);   

    mFullDirIn  = mAppli.DirProject() + mDirLocOfMode + mDirIn + StringDirSeparator();

    // To see if this rule applies always, 4 now dont see inconvenient
    if (mAppli.IsInSpec(&mDirOut) &&  (! mAppli.IsInit(&mDirOut)))
    {
       mDirOut = mDirIn;
    }

    if (mAppli.IsInit(&mDirOut))  // dont do it if mDirIn not used ...
        mDirOut  = SuppressDirFromNameFile(mDirLocOfMode,mDirOut,true);   
    mFullDirOut = mAppli.DirProject() + mDirLocOfMode + mDirOut + StringDirSeparator();

    // Create output directory if needed
    if ((mAppli.IsInSpec(&mDirOut)) || (mAppli.IsInit(&mDirOut)))
    {
        CreateDirectories(mFullDirOut,true);
	if (mPurgeOut)
           RemoveRecurs(mFullDirOut,true,true);
    }
}

        //   ======================  Arg for command =======================================

tPtrArg2007    cDirsPhProj::ArgDirInMand(const std::string & aMesg) 
{ 
    return  Arg2007 (mDirIn ,StrWDef(aMesg,"Input " +mPrefix) ,{mMode,eTA2007::Input }); 
}

tPtrArg2007    cDirsPhProj::ArgDirInOpt(const std::string & aNameVar,const std::string & aMsg,bool WithHDV)  
{ 
    std::vector<tSemA2007>   aVOpt{mMode,eTA2007::Input};
    if (WithHDV) aVOpt.push_back(eTA2007::HDV);
    return  AOpt2007
	    (
               mDirIn,
               StrWDef(aNameVar,"In"+mPrefix) ,
               StrWDef(aMsg,"Input "  + mPrefix),
               aVOpt
            ); 
}

tPtrArg2007    cDirsPhProj::ArgDirInputOptWithDef(const std::string & aDef,const std::string & aNameVar,const std::string & aMsg)
{ 
    mDirIn = aDef;
    mAppli.SetVarInit(&mDirIn);
    return ArgDirInOpt(aNameVar,aMsg,true);
}





tPtrArg2007    cDirsPhProj::ArgDirOutMand(const std::string & aMesg)
{ 
     return  Arg2007(mDirOut,StrWDef(aMesg,"Output " + mPrefix),{mMode,eTA2007::Output}); 
}

tPtrArg2007    cDirsPhProj::ArgDirOutOpt(const std::string & aNameVar,const std::string & aMsg,bool WithDV)
{ 
    std::vector<tSemA2007>   aVOpt{mMode,eTA2007::Output};
    if (WithDV) aVOpt.push_back(eTA2007::HDV);
    return  AOpt2007
            (
                mDirOut,
                StrWDef(aNameVar,"Out"+mPrefix),
                StrWDef(aMsg,"Output " + mPrefix),
                aVOpt
            ); 
}

tPtrArg2007    cDirsPhProj::ArgDirOutOptWithDef(const std::string & aDef,const std::string & aNameVar,const std::string & aMsg)
{ 
    mDirOut = aDef;
    mAppli.SetVarInit(&mDirOut);
    return ArgDirOutOpt(aNameVar,aMsg,true);
}


        //   ======================  Initialization =======================================

void cDirsPhProj::AssertDirInIsInit()    const
{
     MMVII_INTERNAL_ASSERT_User(mAppli.IsInit(&mDirIn),eTyUEr::eUnClassedError,"Input-Dir " + mPrefix  +" required non init");
}
void cDirsPhProj::AssertDirOutIsInit()    const
{
     MMVII_INTERNAL_ASSERT_User(mAppli.IsInit(&mDirOut),eTyUEr::eUnClassedError,"Output-Dir " + mPrefix  +" required non init");
}

bool cDirsPhProj::DirInIsInit() const   
{
    return mAppli.IsInit(&mDirIn);
}
bool cDirsPhProj::DirOutIsInit() const  
{
    return mAppli.IsInit(&mDirOut);
}

        //   ======================  Accessor/Modifier =======================================

const std::string & cDirsPhProj::DirIn() const      
{
   AssertDirInIsInit();
   return mDirIn;
}
const std::string & cDirsPhProj::DirOut() const     
{
   AssertDirOutIsInit();
   return mDirOut;
}
const std::string & cDirsPhProj::FullDirIn() const  
{
   AssertDirInIsInit();
   return mFullDirIn;
}

const std::string & cDirsPhProj::FullDirOut() const 
{
   AssertDirOutIsInit();
   return mFullDirOut;
}

const std::string & cDirsPhProj::FullDirInOut(bool isIn) const
{
   return isIn ? FullDirIn() : FullDirOut();
}

void cDirsPhProj::SetDirIn(const std::string & aDirIn)
{
     mDirIn = aDirIn;
     mAppli.SetVarInit(&mDirIn); // required becaus of AssertOriInIsInit
}

void cDirsPhProj::SetDirInIfNoInit(const std::string & aDirIn)
{
    if (! DirInIsInit())
       SetDirIn(aDirIn);
}

void cDirsPhProj::SetDirOut(const std::string & aDirOut)
{
     mDirOut = aDirOut;
     mAppli.SetVarInit(&mDirOut); // required becaus of AssertOriInIsInit
}

void cDirsPhProj::SetDirOutInIfNotInit()
{
    if (! DirOutIsInit())
    {
        SetDirOut(DirIn());
    }
}


   /* ********************************************************** */
   /*                                                            */
   /*                 cPhotogrammetricProject                    */
   /*                                                            */
   /* ********************************************************** */

        //  =============  Construction & destuction =================

cPhotogrammetricProject::cPhotogrammetricProject(cMMVII_Appli & anAppli) :
    mAppli            (anAppli),
    mDPOrient         (eTA2007::Orient,*this),
    mDPRadiomData     (eTA2007::RadiomData,*this),
    mDPRadiomModel    (eTA2007::RadiomModel,*this),
    mDPMeshDev        (eTA2007::MeshDev,*this),
    mDPMask           (eTA2007::Mask,*this),
    mDPPointsMeasures (eTA2007::PointsMeasure,*this),
    mDPTieP           (eTA2007::TieP,*this),
    mDPMulTieP        (eTA2007::MulTieP,*this),
    mDPMetaData       (eTA2007::MetaData,*this),
    mDPRigBloc        (eTA2007::RigBlock,*this),  // RIGIDBLOC
    mGlobCalcMTD      (nullptr)
{
}


void cPhotogrammetricProject::FinishInit() 
{
    mFolderProject = mAppli.DirProject() ;

    mDirPhp   = mFolderProject + MMVII_DirPhp + StringDirSeparator();
    mDirVisu  = mDirPhp + "VISU" + StringDirSeparator();
    mDirSysCo = mDirPhp + E2Str(eTA2007::SysCo) + StringDirSeparator();

    if (mAppli.LevelCall()==0)
    {
        CreateDirectories(mDirVisu,false);
        CreateDirectories(mDirSysCo,false);

	 // cPt3dr  aZeroNDP(652215.52,6861681.77,35.6);
	 // SaveSysCo(CreateSysCoRTL(aZeroNDP,"Lambert93"),"RTL_NotreDame");
	// maintain it, who knows, but now replaced by 
	// SaveSysCo(cSysCoordV2::Lambert93(),E2Str(eSysCoGeo::eLambert93),true);  
	// SaveSysCo(cSysCoordV2::GeoC()     ,E2Str(eSysCoGeo::eGeoC)     ,true);
    }


    mDPOrient.Finish();
    mDPRadiomData.Finish();
    mDPRadiomModel.Finish();
    mDPMeshDev.Finish();
    mDPMask.Finish();
    mDPPointsMeasures.Finish();
    mDPTieP.Finish();
    mDPMulTieP.Finish();
    mDPMetaData.Finish();
    mDPRigBloc.Finish() ; // RIGIDBLOC

    // Force the creation of directory for metadata spec, make 
    if (! mDPMetaData.DirOutIsInit())
    {
        mDPMetaData.ArgDirOutOptWithDef("Std","","");
    }
    //  Make Std as default value for input 
    if (! mDPMetaData.DirInIsInit())
	mDPMetaData.SetDirIn("Std");

    mDPMetaData.Finish();
    // Create an example file  if none exist
    GenerateSampleCalcMTD();

    // read the data base of existing cameras
    MakeCamDataBase();
}

cDirsPhProj * cPhotogrammetricProject::NewDPIn(eTA2007 aType,const std::string & aDirIn)
{
    cDirsPhProj * aDP = new cDirsPhProj(aType,*this);
    aDP->SetDirIn(aDirIn);
    aDP->Finish();
    mDirAdded.push_back(aDP);

    return aDP;
}


cPhotogrammetricProject::~cPhotogrammetricProject() 
{
    DeleteMTD();
    DeleteAllAndClear(mDirAdded);
}


cMMVII_Appli &  cPhotogrammetricProject::Appli()    {return mAppli;}

const std::string & cPhotogrammetricProject::TaggedNameDefSerial() const {return mAppli.TaggedNameDefSerial();}
const std::string & cPhotogrammetricProject::VectNameDefSerial() const {return mAppli.VectNameDefSerial();}

cDirsPhProj &   cPhotogrammetricProject::DPOrient() {return mDPOrient;}
cDirsPhProj &   cPhotogrammetricProject::DPRadiomData() {return mDPRadiomData;}
cDirsPhProj &   cPhotogrammetricProject::DPRadiomModel() {return mDPRadiomModel;}
cDirsPhProj &   cPhotogrammetricProject::DPMeshDev() {return mDPMeshDev;}
cDirsPhProj &   cPhotogrammetricProject::DPMask() {return mDPMask;}
cDirsPhProj &   cPhotogrammetricProject::DPPointsMeasures() {return mDPPointsMeasures;}
cDirsPhProj &   cPhotogrammetricProject::DPMetaData() {return mDPMetaData;}
cDirsPhProj &   cPhotogrammetricProject::DPTieP() {return mDPTieP;}
cDirsPhProj &   cPhotogrammetricProject::DPMulTieP() {return mDPMulTieP;}
cDirsPhProj &   cPhotogrammetricProject::DPRigBloc() {return mDPRigBloc;} // RIGIDBLOC

const cDirsPhProj &   cPhotogrammetricProject::DPOrient() const {return mDPOrient;}
const cDirsPhProj &   cPhotogrammetricProject::DPRadiomData() const {return mDPRadiomData;}
const cDirsPhProj &   cPhotogrammetricProject::DPRadiomModel() const {return mDPRadiomModel;}
const cDirsPhProj &   cPhotogrammetricProject::DPMeshDev() const {return mDPMeshDev;}
const cDirsPhProj &   cPhotogrammetricProject::DPMask() const {return mDPMask;}
const cDirsPhProj &   cPhotogrammetricProject::DPPointsMeasures() const {return mDPPointsMeasures;}
const cDirsPhProj &   cPhotogrammetricProject::DPMetaData() const {return mDPMetaData;}
const cDirsPhProj &   cPhotogrammetricProject::DPTieP() const {return mDPTieP;}
const cDirsPhProj &   cPhotogrammetricProject::DPMulTieP() const {return mDPMulTieP;}
const cDirsPhProj &   cPhotogrammetricProject::DPRigBloc() const {return mDPRigBloc;} // RIGIDBLOC


const std::string &   cPhotogrammetricProject::DirPhp() const   {return mDirPhp;}
const std::string &   cPhotogrammetricProject::DirVisu() const  {return mDirVisu;}





        //  =============  Radiometric Data =================

cImageRadiomData * cPhotogrammetricProject::ReadRadiomData(const std::string & aNameIm) const
{
    mDPRadiomData.AssertDirInIsInit();

    std::string aFullName  = mDPRadiomData.FullDirIn() + cImageRadiomData::NameFileOfImage(aNameIm);
    return cImageRadiomData::FromFile(aFullName);
}

void cPhotogrammetricProject::SaveRadiomData(const cImageRadiomData & anIRD) const
{
    anIRD.ToFile(mDPRadiomData.FullDirOut()+anIRD.NameFile());
}

        //  =============  Radiometric Calibration =================

cCalibRadiomIma * cPhotogrammetricProject::ReadCalibRadiomIma(const std::string & aNameIm) const
{
/* With only the name of images and the folder, cannot determinate the model used, so the methods
 * test the possible model by testing existence of files.
 */	
    std::string aNameFile = mDPRadiomModel.FullDirIn() + PrefixCalRadRad + aNameIm + "." + TaggedNameDefSerial();
    if (ExistFile(aNameFile))
       return cCalRadIm_Pol::FromFile(aNameFile);

   MMVII_UsersErrror(eTyUEr::eUnClassedError,"Cannot determine Image RadiomCalib  for :" + aNameIm + " in " + mDPRadiomModel.DirIn());
   return nullptr;
}

void cPhotogrammetricProject::SaveCalibRad(const cCalibRadiomIma & aCalRad) const
{
     aCalRad.ToFile(mDPRadiomModel.FullDirOut() + PrefixCalRadRad + aCalRad.NameIm()+ "." + TaggedNameDefSerial());
}

std::string cPhotogrammetricProject::NameCalibRadiomSensor(const cPerspCamIntrCalib & aCam,const cMetaDataImage & aMTD) const
{
    return  PrefixCalRadRad  + "Sensor-" + aCam.Name() + "-Aperture_" + ToStr(aMTD.Aperture());
}

std::string cPhotogrammetricProject::NameCalibRSOfImage(const std::string & aNameIm) const
{
     cMetaDataImage aMetaData =  GetMetaData(aNameIm);
     cPerspCamIntrCalib* aCalib = InternalCalibFromImage(aNameIm);

     return NameCalibRadiomSensor(*aCalib,aMetaData);
}

cRadialCRS * cPhotogrammetricProject::CreateNewRadialCRS(size_t aDegree,const std::string& aNameIm,bool WithCste,int aDegPol)
{
      static std::map<std::string,cRadialCRS *> TheDico;
      std::string aNameCal = NameCalibRSOfImage(aNameIm);

      cRadialCRS * &  aRes = TheDico[aNameCal];

      if (aRes != nullptr)  return aRes;

      cPerspCamIntrCalib* aCalib = InternalCalibFromImage(aNameIm);

      aRes = new cRadialCRS(aCalib->PP(),aDegree,aCalib->SzPix(),aNameCal,WithCste,aDegPol);

      mAppli.AddObj2DelAtEnd(aRes);

      return aRes;
}



        //  =============  Orientation =================

void cPhotogrammetricProject::SaveCamPC(const cSensorCamPC & aCamPC) const
{
    // aCamPC.ToFile(mDPOrient.FullDirOut() + aCamPC.NameOriStd());
    SaveSensor(aCamPC);
}

void cPhotogrammetricProject::SaveSensor(const cSensorImage & aSens) const
{
    aSens.ToFile(mDPOrient.FullDirOut() + aSens.NameOriStd());
}



void cPhotogrammetricProject::SaveCalibPC(const  cPerspCamIntrCalib & aCalib) const
{
    std::string aNameCalib = mDPOrient.FullDirOut() + aCalib.Name() + "." + TaggedNameDefSerial();
    aCalib.ToFileIfFirstime(aNameCalib);
}


cSensorCamPC * cPhotogrammetricProject::ReadCamPC(const cDirsPhProj & aDP,const std::string & aNameIm,bool ToDeleteAutom,bool SVP) const
{
    aDP.AssertDirInIsInit();

    std::string aNameCam  =  aDP.FullDirIn() + cSensorCamPC::NameOri_From_Image(aNameIm);
    // if kindly asked and dont exist return
    if ( SVP && (!ExistFile(aNameCam)) )
    {
       return nullptr;
    }
    // Modif MPD : if we want to delete it ourself (ToDeleteAuto=false) it must not be a remanent object
    // cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam,!ToDelete);
    cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam,ToDeleteAutom);

    if (ToDeleteAutom)
       cMMVII_Appli::AddObj2DelAtEnd(aCamPC);
      

    return aCamPC;
}

cSensorCamPC * cPhotogrammetricProject::ReadCamPC(const std::string & aNameIm,bool ToDeleteAutom,bool SVP) const
{
    return ReadCamPC(mDPOrient,aNameIm,ToDeleteAutom,SVP);
}

cSensorImage* cPhotogrammetricProject::LoadSensor(const std::string  &aNameIm,bool SVP)
{
     cSensorImage*   aSI;
     cSensorCamPC *  aSPC;

     LoadSensor(aNameIm,aSI,aSPC,SVP);

     return aSI;
}

void cPhotogrammetricProject::LoadSensor(const std::string  &aNameIm,cSensorImage* & aSI,cSensorCamPC * & aSPC,bool SVP)
{
     aSI = nullptr;
     aSPC =nullptr;

     aSPC = ReadCamPC(aNameIm,true,true);
     if (aSPC !=nullptr)
     {
        aSI = aSPC;
        return;
     }

     if (!SVP)
     {
         MMVII_UsersErrror
         (
             eTyUEr::eUnClassedError,
             "Cannot get sensor for image " + aNameIm
         );
     }
}

cPerspCamIntrCalib *  cPhotogrammetricProject::InternalCalibFromImage(const std::string & aNameIm) const
{
    // 4 now, pretty basic allox sensor, extract internal, destroy
    // later will have to handle :
    //    * case where calib exist but not pose
    //    * case where nor calib nor pose exist, and must be created from xif 
    mDPOrient.AssertDirInIsInit();
    cSensorCamPC *  aPC = ReadCamPC(aNameIm,false);
    cPerspCamIntrCalib * aCalib = aPC->InternalCalib();
    delete aPC;

    return aCalib;
}
        //  =============  Calibration =================

std::string  cPhotogrammetricProject::StdNameCalibOfImage(const std::string aNameIm) const
{
     cMetaDataImage aMTD = GetMetaData(mFolderProject+FileOfPath(aNameIm,false));
// StdOut() << "StdNameCalibOfImageStdNameCalibOfImage " <<  aMTD.InternalCalibGeomIdent()  << "\n";
     return aMTD.InternalCalibGeomIdent();
}

std::string  cPhotogrammetricProject::FullDirCalibIn() const
{
   mDPOrient.AssertDirInIsInit();
   return mDPOrient.FullDirIn();
}
std::string  cPhotogrammetricProject::FullDirCalibOut() const
{
   return mDPOrient.FullDirOut();
}

cPerspCamIntrCalib *   cPhotogrammetricProject::InternalCalibFromStdName(const std::string aNameIm) const
{
    std::string aNameCalib = FullDirCalibIn() + StdNameCalibOfImage(aNameIm) + "." + TaggedNameDefSerial();
    cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::FromFile(aNameCalib);

    return aCalib;
}

        //  =============  Masks =================

std::string cPhotogrammetricProject::NameMaskOfImage(const std::string & aNameImage) const
{
    return mDPMask.FullDirIn() + aNameImage + ".tif";
}

bool  cPhotogrammetricProject::ImageHasMask(const std::string & aNameImage) const
{
   return    mDPMask.DirInIsInit()
          && ExistFile(NameMaskOfImage(aNameImage)) ;
}

cIm2D<tU_INT1>  cPhotogrammetricProject::MaskWithDef(const std::string & aNameImage,const cBox2di & aBox,bool DefVal) const
{
    if (ImageHasMask( aNameImage))
    {
        return cIm2D<tU_INT1>::FromFile(NameMaskOfImage(aNameImage),aBox);
    }

    return cIm2D<tU_INT1> (aBox.Sz(),nullptr,  (DefVal ? eModeInitImage::eMIA_V1 : eModeInitImage::eMIA_Null)) ;
}


        //  =============  PointsMeasures =================

void cPhotogrammetricProject::SaveMeasureIm(const cSetMesPtOf1Im &  aSetM) const
{
     aSetM.ToFile(mDPPointsMeasures.FullDirOut() +aSetM.StdNameFile());
}

std::string cPhotogrammetricProject::NameMeasureGCPIm(const std::string & aNameIm,bool isIn) const
{
    return  mDPPointsMeasures.FullDirInOut(isIn) + cSetMesPtOf1Im::StdNameFileOfIm(FileOfPath(aNameIm,false)) ;
}

cSetMesPtOf1Im cPhotogrammetricProject::LoadMeasureIm(const std::string & aNameIm,bool isIn) const
{
   //  std::string aDir = mDPPointsMeasures.FullDirInOut(isIn);
   //  return cSetMesPtOf1Im::FromFile(aDir+cSetMesPtOf1Im::StdNameFileOfIm(aNameIm));

   return cSetMesPtOf1Im::FromFile(NameMeasureGCPIm(aNameIm,isIn));
}

void cPhotogrammetricProject::SaveGCP(const cSetMesGCP & aMGCP)
{
     aMGCP.ToFile(mDPPointsMeasures.FullDirOut() + aMGCP.StdNameFile());
     // aMGCP.ToFile(mDPPointsMeasures.FullDirOut() + cSetMesGCP::ThePrefixFiles + aMGCP.Name() + "." + TaggedNameDefSerial());
}

std::string cPhotogrammetricProject::GCPPattern(const std::string & aArgPatFiltr) const
{
    return (aArgPatFiltr=="") ? (cSetMesGCP::ThePrefixFiles + ".*." +TaggedNameDefSerial())  : aArgPatFiltr;
}

std::vector<std::string>  cPhotogrammetricProject::ListFileGCP(const std::string & aArgPatFiltr) const
{
   std::string aPatFiltr = GCPPattern(aArgPatFiltr);
   std::string aDir = mDPPointsMeasures.FullDirIn();
   std::vector<std::string> aRes;

   GetFilesFromDir(aRes,aDir,AllocRegex(aPatFiltr));

   for (auto & aName : aRes)
      aName = aDir + aName;

   return aRes;
}

void cPhotogrammetricProject::LoadGCP(cSetMesImGCP& aSetMes,const std::string & aArgPatFiltr,const std::string & aFiltrNameGCP) const
{
   std::vector<std::string> aListFileGCP = ListFileGCP(aArgPatFiltr);
   MMVII_INTERNAL_ASSERT_User(!aListFileGCP.empty(),eTyUEr::eUnClassedError,"No file found in LoadGCP");

   for (const auto  & aNameFile : aListFileGCP)
   {
       cSetMesGCP aMesGCP = cSetMesGCP::FromFile(aNameFile);
       if (aFiltrNameGCP!="")
          aMesGCP = aMesGCP.Filter(aFiltrNameGCP);
       aSetMes.AddMes3D(aMesGCP);
   }
}

void cPhotogrammetricProject::CpGCPPattern(const std::string & aDirIn,const std::string & aDirOut,const std::string & aArgPatFiltr) const
{
   CopyPatternFile(aDirIn,GCPPattern(aArgPatFiltr),aDirOut);
}

void cPhotogrammetricProject::CpGCP() const
{
	CpGCPPattern(mDPPointsMeasures.FullDirIn(),mDPPointsMeasures.FullDirOut());
}



void cPhotogrammetricProject::LoadIm(cSetMesImGCP& aSetMes,const std::string & aNameIm,cSensorImage * aSIm,bool SVP) const
{
//    std::string aDir = mDPPointsMeasures.FullDirIn();
   //cSetMesPtOf1Im  aSetIm = cSetMesPtOf1Im::FromFile(aDir+cSetMesPtOf1Im::StdNameFileOfIm(aNameIm));
   if (SVP && (! ExistFile(NameMeasureGCPIm(aNameIm,true))))
      return;
   cSetMesPtOf1Im  aSetIm = LoadMeasureIm(aNameIm);
   aSetMes.AddMes2D(aSetIm,aSIm);
}

void cPhotogrammetricProject::LoadIm(cSetMesImGCP& aSetMes,cSensorImage & aSIm) const
{
     LoadIm(aSetMes,aSIm.NameImage(),&aSIm);
}

cSet2D3D  cPhotogrammetricProject::LoadSet32(const std::string & aNameIm) const
{
    cSetMesImGCP aSetMes;

    LoadGCP(aSetMes);
    LoadIm(aSetMes,aNameIm);

    cSet2D3D aSet23;
    aSetMes.ExtractMes1Im(aSet23,aNameIm);

    return aSet23;
}


void cPhotogrammetricProject::SaveAndFilterAttrEll(const cSetMesPtOf1Im &  aSetM,const std::list<std::string> & ToRem) const
{

     std::string  aNameIn = cSaveExtrEllipe::NameFile(*this,aSetM,true);
     if (!ExistFile(aNameIn))
        return;

     std::vector<cSaveExtrEllipe> aVSEEIn;
     ReadFromFile(aVSEEIn,aNameIn);

     std::vector<cSaveExtrEllipe> aVSEEOut;
     for (const auto & aSEE : aVSEEIn)
         if (! BoolFind(ToRem,aSEE.mNameCode))
            aVSEEOut.push_back(aSEE);
     SaveInFile(aVSEEOut,cSaveExtrEllipe::NameFile(*this,aSetM,false));
}

        //  =============  Multiple Tie Points =================

std::string cPhotogrammetricProject::NameMultipleTieP(const std::string & aNameIm) const
{
   return "PMUL-"+ aNameIm + "." + VectNameDefSerial();
}

void  cPhotogrammetricProject::SaveMultipleTieP(const cVecTiePMul& aVPm,const std::string & aNameIm) const
{
   PushPrecTxtSerial(3);
   SaveInFile(aVPm.mVecTPM,mDPMulTieP.FullDirOut()+NameMultipleTieP(aNameIm));
   PopPrecTxtSerial();
}

void  cPhotogrammetricProject::ReadMultipleTieP(cVecTiePMul& aVPm,const std::string & aNameIm) const
{
   ReadFromFile(aVPm.mVecTPM,mDPMulTieP.FullDirIn()+NameMultipleTieP(aNameIm));
   aVPm.mNameIm = aNameIm;
}


std::string cPhotogrammetricProject::NameConfigMTP(const std::string &  anExt)
{
    return  "MTP-Config." + anExt;
}

std::string cPhotogrammetricProject::NameConfigMTPIn() const
{
    return mDPMulTieP.FullDirIn() + NameConfigMTP();
}

std::string cPhotogrammetricProject::NameConfigMTPOut(const std::string &  anExt) const
{
    return mDPMulTieP.FullDirOut() + NameConfigMTP(anExt);
}



        //  =============  Homologous point =================

void  cPhotogrammetricProject::SaveHomol
      (
           const cSetHomogCpleIm & aSetHCI,
           const std::string & aNameIm1 ,
	   const std::string & aNameIm2,
	   const std::string & aDirIn
      ) const
{
	std::string aDir = (aDirIn=="") ? mDPTieP.FullDirOut() : aDirIn;

	aDir = aDir + aNameIm1 + StringDirSeparator();
	CreateDirectories(aDir,true);

	std::string  aName = aDir+aNameIm2 + "." +  VectNameDefSerial();
	aSetHCI.ToFile(aName);
}

std::string cPhotogrammetricProject::NameTiePIn(const std::string & aNameIm1,const std::string & aNameIm2,const std::string & aDirIn) const
{
    std::string aDir = (aDirIn=="") ? mDPTieP.FullDirIn() : aDirIn;
    return  aDir+aNameIm1+StringDirSeparator()+aNameIm2+"."+VectNameDefSerial();
}


void  cPhotogrammetricProject::ReadHomol
      (
           cSetHomogCpleIm & aSetHCI,
           const std::string & aNameIm1 ,
           const std::string & aNameIm2,
	   const std::string & aDirIn
      ) const
{
    std::string aName = NameTiePIn(aNameIm1,aNameIm2,aDirIn); 
    ReadFromFile(aSetHCI.SetH(),aName);
}
        //  =============  coord system  =================

void cPhotogrammetricProject::SaveSysCo(tPtrSysCo aSys,const std::string& aName,bool OnlyIfNew) const
{
     std::string aFullName = mDirSysCo + aName + "."+  GlobTaggedNameDefSerial();

     if (OnlyIfNew && ExistFile(aFullName))
        return;
     aSys->ToFile(aFullName);
}

std::string  cPhotogrammetricProject::FullNameSysCo(const std::string &aName,bool SVP) const
{
     //  try to get the file in local folder
     std::string aNameGlob = mDirSysCo + aName + "." + GlobTaggedNameDefSerial();
     if (ExistFile(aNameGlob))
        return aNameGlob;

     //  try the name 
     aNameGlob = cMMVII_Appli::DirRessourcesMMVII() + "SysCo/" + aName + "." + GlobTaggedNameDefSerial();
     if (ExistFile(aNameGlob))
        return aNameGlob;

     if (! SVP)
        MMVII_UnclasseUsEr("Cannot find coord sys for " + aName);

     return "";
}

tPtrSysCo cPhotogrammetricProject::ReadSysCo(const std::string &aName,bool SVP) const
{
     // compute name
     std::string aNameGlob = FullNameSysCo(aName,SVP);
     if (aNameGlob=="") // if we are here, SVP=true and we can return nullptr
         return tPtrSysCo(nullptr);
     return  cSysCoordV2::FromFile(aNameGlob);
}

tPtrSysCo cPhotogrammetricProject::CreateSysCoRTL(const cPt3dr & aOrig,const std::string &aName,bool SVP) const
{
    std::string  aNameFull = FullNameSysCo(aName,SVP);
    if (aNameFull=="")
       return tPtrSysCo(nullptr);

    return cSysCoordV2::RTL(aOrig,aNameFull);
}

cChangSysCoordV2  cPhotogrammetricProject::ChangSys(const std::vector<std::string> & aVec,tREAL8 aEpsDif) 
{
    if (! mAppli.IsInit(&aVec))  return cChangSysCoordV2{};

    if (aVec.size() == 1)
       return cChangSysCoordV2(ReadSysCo(aVec.at(0)));

    return cChangSysCoordV2(   ReadSysCo(aVec.at(0))  ,  ReadSysCo(aVec.at(1))  );
}

             //==================   SysCo saved in standard folder ============

std::string  cPhotogrammetricProject::NameCurSysCo(const cDirsPhProj & aDP,bool IsIn) const
{
   return aDP.FullDirInOut(IsIn)   +  "CurSysCo." +  GlobTaggedNameDefSerial();

}
tPtrSysCo  cPhotogrammetricProject::CurSysCo(const cDirsPhProj & aDP,bool SVP) const
{
    std::string aName = NameCurSysCo(aDP,true);
    if (! ExistFile(aName))
    {
       if (! SVP)
           MMVII_UnclasseUsEr("CurSysCo dont exist : " + aName);
       return tPtrSysCo(nullptr);
    }

    return cSysCoordV2::FromFile(NameCurSysCo(aDP,true));
}

tPtrSysCo  cPhotogrammetricProject::CurSysCoOri(bool SVP) const {return CurSysCo(mDPOrient,SVP);}
tPtrSysCo  cPhotogrammetricProject::CurSysCoGCP(bool SVP) const {return CurSysCo(mDPPointsMeasures,SVP);}

void cPhotogrammetricProject::SaveCurSysCo(const cDirsPhProj & aDP,tPtrSysCo aSysCo) const 
{
    aSysCo->ToFile(NameCurSysCo(aDP,false));
}

void cPhotogrammetricProject::SaveCurSysCoOri(tPtrSysCo aSysCo) const { SaveCurSysCo(mDPOrient,aSysCo); }
void cPhotogrammetricProject::SaveCurSysCoGCP(tPtrSysCo aSysCo) const { SaveCurSysCo(mDPPointsMeasures,aSysCo); }


void cPhotogrammetricProject::CpSysIn2Out(bool  OriIn,bool OriOut) const
{
   tPtrSysCo aSysIn = OriIn ?  CurSysCoOri(true) : CurSysCoGCP(true);

   if (aSysIn.get() == nullptr)
      return;

   if (OriOut)
      SaveCurSysCoOri(aSysIn);
   else
      SaveCurSysCoGCP(aSysIn);
}




//  cMMVII_Appli DirRessourcesMMVII

        //  =============  Rigid bloc  =================

	                   // RIGIDBLOC
static std::string PrefixRigidBloc = "RigidBloc_";

void   cPhotogrammetricProject::SaveBlocCamera(const cBlocOfCamera & aBloc) const
{
     std::string  aName = mDPRigBloc.FullDirOut()   + PrefixRigidBloc + aBloc.Name() + "." + TaggedNameDefSerial();
     aBloc.ToFile(aName);
}

	                   // RIGIDBLOC
std::list<cBlocOfCamera *> cPhotogrammetricProject::ReadBlocCams() const
{
    std::list<cBlocOfCamera *> aRes;

    std::vector<std::string>  aVNames =   GetFilesFromDir(mDPRigBloc.FullDirIn(),AllocRegex(PrefixRigidBloc+".*"));
    for (const auto & aName : aVNames)
        aRes.push_back(cBlocOfCamera::FromFile(mDPRigBloc.FullDirIn()+aName));

    return aRes;
}

        //  =============  Meta Data =================

//  see cMetaDataImages.cpp


}; // MMVII

