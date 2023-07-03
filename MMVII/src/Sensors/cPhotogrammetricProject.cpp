#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"

/**
   \file  cPhotogrammetricProject.cpp

   \brief file for handling names/upload/download of photogram data (pose,calib, ...)

   test Git
*/


namespace MMVII
{




std::string SuppressDirFromNameFile(const std::string & aDir,const std::string & aName)
{
    // mOriIn.starts_with(aDir);  -> C++20
    // to see if StringDirSeparator() is not a meta carac on window ?

     if (TheSYS == eSYS::Windows)
     {
          MMVII_DEV_WARNING("SuppressDirFromNameFile check regular expression on Window");
     }
     
     std::string aPat =  "(.*" + aDir+")?" + "([A-Za-z0-9_-]+)" + StringDirSeparator() + "?";
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

void cDirsPhProj::Finish()
{
    //  Allow user to specify indiferrently short name of full name, will extract short name
    // for ex :   "MMVII-PhgrProj/Orient/Test/" ,  "Test/",  "Test" ...  =>   "Test"
    //
    if (mAppli.IsInit(&mDirIn))  // dont do it if mDirIn not used ...
        mDirIn  = SuppressDirFromNameFile(mDirLocOfMode,mDirIn);   

    mFullDirIn  = mAppli.DirProject() + mDirLocOfMode + mDirIn + StringDirSeparator();

    // To see if this rule applies always, 4 now dont see inconvenient
    if (mAppli.IsInSpec(&mDirOut) &&  (! mAppli.IsInit(&mDirOut)))
    {
       mDirOut = mDirIn;
    }

    if (mAppli.IsInit(&mDirOut))  // dont do it if mDirIn not used ...
        mDirOut  = SuppressDirFromNameFile(mDirLocOfMode,mDirOut);   
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

void cDirsPhProj::SetDirIn(const std::string & aDirIn)
{
     mDirIn = aDirIn;
     mAppli.SetVarInit(&mDirIn); // required becaus of AssertOriInIsInit
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
    mDPTieP          (eTA2007::TieP,*this),
    mDPMetaData       (eTA2007::MetaData,*this),
    mGlobCalcMTD      (nullptr)
{
}


void cPhotogrammetricProject::FinishInit() 
{
    mFolderProject = mAppli.DirProject() ;

    mDPOrient.Finish();
    mDPRadiomData.Finish();
    mDPRadiomModel.Finish();
    mDPMeshDev.Finish();
    mDPMask.Finish();
    mDPPointsMeasures.Finish();
    mDPTieP.Finish();
    mDPMetaData.Finish();

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

    // StdOut() << "MTD=" <<   mDPMetaData.FullDirOut() << "\n"; 
    // StdOut() << "MTD=" <<   mDPMetaData.FullDirOut() << "\n"; getchar();
}

cPhotogrammetricProject::~cPhotogrammetricProject() 
{
    DeleteMTD();
    DeleteAllAndClear(mLCam2Del);
}


cMMVII_Appli &  cPhotogrammetricProject::Appli()    {return mAppli;}

cDirsPhProj &   cPhotogrammetricProject::DPOrient() {return mDPOrient;}
cDirsPhProj &   cPhotogrammetricProject::DPRadiomData() {return mDPRadiomData;}
cDirsPhProj &   cPhotogrammetricProject::DPRadiomModel() {return mDPRadiomModel;}
cDirsPhProj &   cPhotogrammetricProject::DPMeshDev() {return mDPMeshDev;}
cDirsPhProj &   cPhotogrammetricProject::DPMask() {return mDPMask;}
cDirsPhProj &   cPhotogrammetricProject::DPPointsMeasures() {return mDPPointsMeasures;}
cDirsPhProj &   cPhotogrammetricProject::DPMetaData() {return mDPMetaData;}
cDirsPhProj &   cPhotogrammetricProject::DPTieP() {return mDPTieP;}

const cDirsPhProj &   cPhotogrammetricProject::DPOrient() const {return mDPOrient;}
const cDirsPhProj &   cPhotogrammetricProject::DPRadiomData() const {return mDPRadiomData;}
const cDirsPhProj &   cPhotogrammetricProject::DPRadiomModel() const {return mDPRadiomModel;}
const cDirsPhProj &   cPhotogrammetricProject::DPMeshDev() const {return mDPMeshDev;}
const cDirsPhProj &   cPhotogrammetricProject::DPMask() const {return mDPMask;}
const cDirsPhProj &   cPhotogrammetricProject::DPPointsMeasures() const {return mDPPointsMeasures;}
const cDirsPhProj &   cPhotogrammetricProject::DPMetaData() const {return mDPMetaData;}
const cDirsPhProj &   cPhotogrammetricProject::DPTieP() const {return mDPTieP;}




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
    std::string aNameFile = mDPRadiomModel.FullDirIn() + PrefixCalRadRad + aNameIm + "." + PostF_XmlFiles;
    if (ExistFile(aNameFile))
       return cCalRadIm_Pol::FromFile(aNameFile);

   MMVII_UsersErrror(eTyUEr::eUnClassedError,"Cannot determine Image RadiomCalib  for :" + aNameIm + " in " + mDPRadiomModel.DirIn());
   return nullptr;
}

void cPhotogrammetricProject::SaveCalibRad(const cCalibRadiomIma & aCalRad) const
{
     aCalRad.ToFile(mDPRadiomModel.FullDirOut() + PrefixCalRadRad + aCalRad.NameIm()+ "." + PostF_XmlFiles);
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
    aCamPC.ToFile(mDPOrient.FullDirOut() + aCamPC.NameOriStd());
}


void cPhotogrammetricProject::SaveCalibPC(const  cPerspCamIntrCalib & aCalib) const
{
    std::string aNameCalib = mDPOrient.FullDirOut() + aCalib.Name() + ".xml";
    aCalib.ToFileIfFirstime(aNameCalib);
}


cSensorCamPC * cPhotogrammetricProject::ReadCamPC(const std::string & aNameIm,bool ToDelete,bool SVP) const
{
    mDPOrient.AssertDirInIsInit();

    std::string aNameCam  =  mDPOrient.FullDirIn() + cSensorCamPC::NameOri_From_Image(aNameIm);
    // if kindly asked and dont exist return
    if ( SVP && (!ExistFile(aNameCam)) )
    {
       return nullptr;
    }
    cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam);

    if (ToDelete)
       mLCam2Del.push_back(aCamPC);

    return aCamPC;
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
    std::string aNameCalib = FullDirCalibIn() + StdNameCalibOfImage(aNameIm) + ".xml";
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

cSetMesPtOf1Im cPhotogrammetricProject::LoadMeasureIm(const std::string & aNameIm) const
{
   std::string aDir = mDPPointsMeasures.FullDirIn();
   return cSetMesPtOf1Im::FromFile(aDir+cSetMesPtOf1Im::StdNameFileOfIm(aNameIm));
}

void cPhotogrammetricProject::SaveGCP(const cSetMesImGCP& aSetMes,const std::string & aExt)
{
     cSetMesGCP  aMGCP = aSetMes.ExtractSetGCP(aExt);
     aMGCP.ToFile(mDPPointsMeasures.FullDirOut() + cSetMesGCP::ThePrefixFiles +aExt + ".xml");
}

void cPhotogrammetricProject::LoadGCP(cSetMesImGCP& aSetMes,const std::string & aArgPatFiltr) const
{
   std::string aPatFiltr = (aArgPatFiltr=="") ? (cSetMesGCP::ThePrefixFiles + ".*.xml")  : aArgPatFiltr;

   std::string aDir = mDPPointsMeasures.FullDirIn();
   std::vector<std::string> aListFileGCP =  GetFilesFromDir(aDir,AllocRegex(aPatFiltr));
   MMVII_INTERNAL_ASSERT_User(!aListFileGCP.empty(),eTyUEr::eUnClassedError,"No file found in LoadGCP");

   for (const auto  & aNameFile : aListFileGCP)
   {
       cSetMesGCP aMesGGP = cSetMesGCP::FromFile(aDir+aNameFile);
       aSetMes.AddMes3D(aMesGGP);
   }
}

void cPhotogrammetricProject::LoadIm(cSetMesImGCP& aSetMes,const std::string & aNameIm,cSensorImage * aSIm) const
{
//    std::string aDir = mDPPointsMeasures.FullDirIn();
   //cSetMesPtOf1Im  aSetIm = cSetMesPtOf1Im::FromFile(aDir+cSetMesPtOf1Im::StdNameFileOfIm(aNameIm));
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

        //  =============  Homologous point =================

void  cPhotogrammetricProject::SaveHomol
      (
           const cSetHomogCpleIm & aSetHCI,
           const std::string & aNameIm1 ,
	   const std::string & aNameIm2,
	   const std::string & anExt
      ) const
{
	std::string aDir = mDPTieP.FullDirOut();

	aDir = aDir + aNameIm1 + StringDirSeparator();
	CreateDirectories(aDir,true);

	std::string  aName = aDir+aNameIm2 + "." +anExt;
	aSetHCI.ToFile(aName);

	// SaveInFile(aSetHCI,aName);
}


        //  =============  Meta Data =================

//  see cMetaDataImages.cpp


}; // MMVII

