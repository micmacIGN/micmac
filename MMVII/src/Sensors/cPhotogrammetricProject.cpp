#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_BlocRig.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Clino.h"
#include "cExternalSensor.h"
#include "MMVII_Topo.h"
#include "MMVII_PoseTriplet.h"
#include "MMVII_InstrumentalBlock.h"

/**
   \file  cPhotogrammetricProject.cpp

   \brief file for handling names/upload/download of photogram data (pose,calib, ...)

   test Git
*/


namespace MMVII
{
/** "Facility" class for function like "LoadMeasureImFromFolder" where we want to change temporarilly
 * the input directory of cDirsPhProj, use the destructor to automatically restor initial context */
class cAutoChgRestoreDefFolder
{
public :
    cAutoChgRestoreDefFolder(const std::string & aFolder,const cDirsPhProj & aDP, bool aIsIn);
    ~cAutoChgRestoreDefFolder();
private :
    cDirsPhProj & mDP;
    bool mIsIn;
    std::string   mCurDir; // if empty: nothing to restore, dir was not init beforehand
};

cAutoChgRestoreDefFolder::cAutoChgRestoreDefFolder(const std::string & aFolder,const cDirsPhProj & aDP, bool aIsIn) :
    mDP          (const_cast<cDirsPhProj&> (aDP)),
    mIsIn        (aIsIn),
    mCurDir      (mIsIn? (mDP.DirInIsInit()?mDP.DirIn():""):(mDP.DirOutIsInit()?mDP.DirOut():""))
{
    if (mIsIn)
        mDP.SetDirIn(aFolder);
    else
        mDP.SetDirOut(aFolder);
}
cAutoChgRestoreDefFolder::~cAutoChgRestoreDefFolder() 
{
    if (!mCurDir.empty())
    {
        if (mIsIn)
            mDP.SetDirIn(mCurDir);
        else
            mDP.SetDirOut(mCurDir);
    }
}

    // =============================================================================
   
std::string SuppressDirFromNameFile(const std::string & aDir,const std::string & aName,bool ByDir)
{
    // mOriIn.starts_with(aDir);  -> C++20
    // to see if StringDirSeparator() is not a meta carac on window ?

     std::string aPat =  "(.*" + aDir+")?" + "([A-Za-z0-9_.-]+)";
     if (ByDir)
         aPat = aPat + "[\\/]?";
     else
	 aPat = aPat + "\\." +  GlobTaggedNameDefSerial()  ;
     if (! MatchRegex(aName,aPat))
     {
         MMVII_UserError
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
    if ( ((mAppli.IsInSpec(&mDirOut)) || (mAppli.IsInit(&mDirOut)))  && (mDirOut!=MMVII_NONE))
    {
        CreateDirectories(mFullDirOut,true);
	if (mPurgeOut)
           RemoveRecurs(mFullDirOut,true,true);
    }
}

        //   ======================  Arg for command =======================================

tPtrArg2007    cDirsPhProj::ArgDirInMand(const std::string & aMesg,std::string * aDest) 
{ 
    return  Arg2007 ((aDest ? *aDest : mDirIn) ,StrWDef(aMesg,"Input " +mPrefix) ,{mMode,eTA2007::Input }); 
}
tPtrArg2007    cDirsPhProj::ArgDirInMand(const std::string & aMesg) 
{
	return ArgDirInMand(aMesg,nullptr);
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

bool cDirsPhProj::CheckDirExists(bool In, bool DoError) const
{
    std::string aPath = In?mFullDirIn:mFullDirOut;
    bool aExists = IsDirectory(aPath);
    if (DoError)
    {
        MMVII_INTERNAL_ASSERT_User(aExists, eTyUEr::eOpenFile, aPath+" is not a directory!");
    }
    return aExists;
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
    return (mDirIn!= MMVII_NONE)  && mAppli.IsInit(&mDirIn);
}
bool cDirsPhProj::DirInIsNONE() const   
{
    return  mAppli.IsInit(&mDirIn) && (mDirIn== MMVII_NONE);
}



bool cDirsPhProj::DirOutIsInit() const  
{
    return (mDirOut!= MMVII_NONE) && mAppli.IsInit(&mDirOut);
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
    //  StdOut() << "cDirsPhProj::SetDirI In cDirsPhProj::SetDirIn\n";
    // MPD : may be dangerous, but seems required, dont understand why it was not made before
    Finish();
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
     Finish();
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
    mCurSysCo         (nullptr),
    mChSysCo          (),
    mDPOrient         (eTA2007::Orient,*this),
    mDPOriTriplets    (eTA2007::OriTriplet,*this),
    mDPRadiomData     (eTA2007::RadiomData,*this),
    mDPRadiomModel    (eTA2007::RadiomModel,*this),
    mDPMeshDev        (eTA2007::MeshDev,*this),
    mDPMask           (eTA2007::Mask,*this),
    mDPGndPt3D        (eTA2007::ObjCoordWorld,*this),
    mDPGndPt2D        (eTA2007::ObjMesInstr,*this),
    mDPTieP           (eTA2007::TieP,*this),
    mDPMulTieP        (eTA2007::MulTieP,*this),
    mDPMetaData       (eTA2007::MetaData,*this),
    mDPBlockInstr     (eTA2007::InstrBlock,*this),  
    mDPRigBloc        (eTA2007::RigBlock,*this),  // RIGIDBLOC
    mDPClinoMeters    (eTA2007::Clino,*this),  
    mDPMeasuresClino  (eTA2007::MeasureClino,*this),
    mDPTopoMes        (eTA2007::Topo,*this),  // Topo
    mDPStaticLidar    (eTA2007::StaticLidar,*this),  // StaticLidar
    mGlobCalcMTD      (nullptr)
{
}


void cPhotogrammetricProject::FinishInit() 
{
    mFolderProject = mAppli.DirProject() ;

    mDirPhp   = mFolderProject + MMVII_DirPhp + StringDirSeparator();
    mDirVisu  = mDirPhp + "VISU" + StringDirSeparator();
    mDirVisuAppli  = mDirVisu + mAppli.Specs().Name()  + StringDirSeparator();
    mDirSysCo = mDirPhp + E2Str(eTA2007::SysCo) + StringDirSeparator();
    mDirImportInitOri =  mDirPhp + "InitialOrientations" + StringDirSeparator();

    if (mAppli.LevelCall()==0)
    {
        CreateDirectories(mDirVisu,false);
        CreateDirectories(mDirVisuAppli,false);
        CreateDirectories(mDirSysCo,false);
        CreateDirectories(mDirImportInitOri,false);

    }


    mDPOrient.Finish();
    mDPOriTriplets.Finish();
    mDPRadiomData.Finish();
    mDPRadiomModel.Finish();
    mDPMeshDev.Finish();
    mDPMask.Finish();
    mDPGndPt3D.Finish();
    mDPGndPt2D.Finish();
    mDPTieP.Finish();
    mDPMulTieP.Finish();
    mDPMetaData.Finish();
    mDPBlockInstr.Finish() ; 
    mDPRigBloc.Finish() ; // RIGIDBLOC
    mDPClinoMeters.Finish() ; 
    mDPMeasuresClino.Finish() ; 
    mDPTopoMes.Finish() ; // TOPO
    mDPStaticLidar.Finish() ;

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

    if (mAppli.IsInit(&mNameChSysCo))
    {
       mChSysCo = ChangSysCo(mNameChSysCo);
    }

    if (mAppli.IsInit(&mNameCurSysCo))
    {
       mCurSysCo = ReadSysCo(mNameCurSysCo);
    }
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
cDirsPhProj &   cPhotogrammetricProject::DPOriTriplets() {return mDPOriTriplets;}
cDirsPhProj &   cPhotogrammetricProject::DPRadiomData() {return mDPRadiomData;}
cDirsPhProj &   cPhotogrammetricProject::DPRadiomModel() {return mDPRadiomModel;}
cDirsPhProj &   cPhotogrammetricProject::DPMeshDev() {return mDPMeshDev;}
cDirsPhProj &   cPhotogrammetricProject::DPMask() {return mDPMask;}
cDirsPhProj &   cPhotogrammetricProject::DPGndPt3D() {return mDPGndPt3D;}
cDirsPhProj &   cPhotogrammetricProject::DPGndPt2D() {return mDPGndPt2D;}
cDirsPhProj &   cPhotogrammetricProject::DPMetaData() {return mDPMetaData;}
cDirsPhProj &   cPhotogrammetricProject::DPTieP() {return mDPTieP;}
cDirsPhProj &   cPhotogrammetricProject::DPMulTieP() {return mDPMulTieP;}
cDirsPhProj &   cPhotogrammetricProject::DPBlockInstr() {return mDPBlockInstr;} 
cDirsPhProj &   cPhotogrammetricProject::DPRigBloc() {return mDPRigBloc;} // RIGIDBLOC
cDirsPhProj &   cPhotogrammetricProject::DPClinoMeters() {return mDPClinoMeters;} 
cDirsPhProj &   cPhotogrammetricProject::DPMeasuresClino() {return mDPMeasuresClino;}
cDirsPhProj &   cPhotogrammetricProject::DPTopoMes() {return mDPTopoMes;} // TOPO
cDirsPhProj &   cPhotogrammetricProject::DPStaticLidar() {return mDPStaticLidar;}

const cDirsPhProj &   cPhotogrammetricProject::DPOrient() const {return mDPOrient;}
const cDirsPhProj &   cPhotogrammetricProject::DPOriTriplets() const {return mDPOriTriplets;}
const cDirsPhProj &   cPhotogrammetricProject::DPRadiomData() const {return mDPRadiomData;}
const cDirsPhProj &   cPhotogrammetricProject::DPRadiomModel() const {return mDPRadiomModel;}
const cDirsPhProj &   cPhotogrammetricProject::DPMeshDev() const {return mDPMeshDev;}
const cDirsPhProj &   cPhotogrammetricProject::DPMask() const {return mDPMask;}
const cDirsPhProj &   cPhotogrammetricProject::DPGndPt3D() const {return mDPGndPt3D;}
const cDirsPhProj &   cPhotogrammetricProject::DPGndPt2D() const {return mDPGndPt2D;}
const cDirsPhProj &   cPhotogrammetricProject::DPMetaData() const {return mDPMetaData;}
const cDirsPhProj &   cPhotogrammetricProject::DPTieP() const {return mDPTieP;}
const cDirsPhProj &   cPhotogrammetricProject::DPMulTieP() const {return mDPMulTieP;}
const cDirsPhProj &   cPhotogrammetricProject::DPBlockInstr() const {return mDPBlockInstr;} 
const cDirsPhProj &   cPhotogrammetricProject::DPRigBloc() const {return mDPRigBloc;} // RIGIDBLOC
const cDirsPhProj &   cPhotogrammetricProject::DPClinoMeters() const {return mDPClinoMeters;} // RIGIDBLOC
const cDirsPhProj &   cPhotogrammetricProject::DPMeasuresClino() const {return mDPMeasuresClino;} // RIGIDBLOC
const cDirsPhProj &   cPhotogrammetricProject::DPTopoMes() const {return mDPTopoMes;} // Topo
const cDirsPhProj &   cPhotogrammetricProject::DPStaticLidar() const {return mDPStaticLidar;}


const std::string &   cPhotogrammetricProject::DirPhp() const   {return mDirPhp;}
const std::string &   cPhotogrammetricProject::DirVisu() const  {return mDirVisu;}
const std::string &   cPhotogrammetricProject::DirVisuAppli() const  {return mDirVisuAppli;}





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

   MMVII_UserError(eTyUEr::eUnClassedError,"Cannot determine Image RadiomCalib  for :" + aNameIm + " in " + mDPRadiomModel.DirIn());
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
      thread_local static std::map<std::string,cRadialCRS *> TheDico;
      std::string aNameCal = NameCalibRSOfImage(aNameIm);

      cRadialCRS * &  aRes = TheDico[aNameCal];

      if (aRes != nullptr)  return aRes;

      cPerspCamIntrCalib* aCalib = InternalCalibFromImage(aNameIm);

      aRes = new cRadialCRS(aCalib->PP(),aDegree,aCalib->SzPix(),aNameCal,WithCste,aDegPol);

      mAppli.AddObj2DelAtEnd(aRes);

      return aRes;
}

         
        //  ============================================
        //                   Orientation 
        //  ============================================

const std::string &   cPhotogrammetricProject::DirImportInitOri() const { return mDirImportInitOri; }


bool cPhotogrammetricProject::IsOriInDirInit() const
{
    return mDPOrient.DirInIsInit();
}

         //  =============  Central Perspective camera =======================

void cPhotogrammetricProject::SaveCamPC(const cSensorCamPC & aCamPC) const
{
    // aCamPC.ToFile(mDPOrient.FullDirOut() + aCamPC.NameOriStd());
    SaveSensor(aCamPC);
}

void cPhotogrammetricProject::SaveSensor(const cSensorImage & aSens) const
{
     if ( mDPOrient.DirOut() == MMVII_NONE)
        return;


    /*  Supression by global pattern can be very slow with big data
     *  So we creat the first time a map that contain for an image all the files corresponding to
     *  a sensor in the standard out folder.
     *
     *  This is done by (1) computing all the file (2) use regular expression to recover the
     *  name of image from the file.  This works because the MMVII prefix dont contain any "-" .
     */
    thread_local static std::map<std::string,std::vector<std::string>> TheMapIm2Sensors;
    thread_local static bool First = true;
    if (First)
    {
         First = false;
         std::string aPat2Sup =  "Ori-[A-Za-z0-9]*-(.*)." + GlobTaggedNameDefSerial()  ;
         std::string aFullPat2Sup = mDPOrient.FullDirOut() + aPat2Sup;
	 tNameSet aSet = SetNameFromPat(aFullPat2Sup);

	 std::vector<std::string> aVect = ToVect(aSet);
	 for (const auto & aNameSens : aVect)
	 {
            std::string aNameIm = PatternKthSubExpr(aPat2Sup,1,aNameSens);

	    TheMapIm2Sensors[aNameIm].push_back(aNameSens);
	 }
    }


    // We dont want to have different variant of the same image in a given folder
    // so supress potentiel existing orientation of the same image
    // CM: Should be ...Image() + "\\." + Glob..., but '\' is a directory separator on Windows
    //     and SplitDirAndFile() called by RemovePatternFile() will do bad things in this case ...
    //
    //

    if (0)
    {
        //     can be very slow with big data file  ...
        std::string aPat2Sup = mDPOrient.FullDirOut() + "Ori-.*-" + aSens.NameImage() + "." + GlobTaggedNameDefSerial()  ;
        RemovePatternFile(aPat2Sup,false);
    }
    else
    {
         for (const  auto & aName : TheMapIm2Sensors[aSens.NameImage()])
	 {
             RemoveFile(mDPOrient.FullDirOut() + aName,false);
	 }
    }

    aSens.ToFile(mDPOrient.FullDirOut() + aSens.NameOriStd());

    // if (UserIsMPD())
    {
        if (aSens.HasCoordinateSystem())
        {
            SaveCurSysCoOri(ReadSysCo(aSens.GetCoordinateSystem()));
        }
    }
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

tPoseR cPhotogrammetricProject::ReadPoseCamPC(const std::string & aNameIm,bool * IsOk) const
{
    cSensorCamPC * aCamPC = ReadCamPC(aNameIm,DelAuto::Yes,SVP::Yes);

    if (IsOk)
    {
       *IsOk = aCamPC!=nullptr;
       if (!*IsOk)
          return tPoseR::RandomIsom3D(10);
     }
    else 
    {
        MMVII_INTERNAL_ASSERT_strong(aCamPC!=nullptr,"Cannot ReadPoseCamPC");
    }
    
    return aCamPC->Pose();
}


cSensorImage* cPhotogrammetricProject::ReadSensor(const std::string  &aNameIm,bool ToDeleteAutom,bool SVP) const
{
     cSensorImage*   aSI;
     cSensorCamPC *  aSPC;

     ReadSensor(aNameIm,aSI,aSPC,ToDeleteAutom,SVP);

     return aSI;
}

void cPhotogrammetricProject::ReadSensor(const std::string  &aNameIm,cSensorImage* & aSI,cSensorCamPC * & aSPC,bool ToDeleteAutom,bool SVP) const
{
     aSI = nullptr;
     aSPC =nullptr;

     // Try a stenope camera which has interesting properties
     aSPC = ReadCamPC(aNameIm,ToDeleteAutom,true);
     if (aSPC !=nullptr)
     {
        aSI = aSPC;
        return;
     }

     // Else try an external sensor
     if (aSI==nullptr) aSI =  SensorTryReadImported(*this,aNameIm);
     if (aSI==nullptr) aSI =  SensorTryReasChSys(*this,aNameIm);
     if (aSI==nullptr) aSI =  SensorTryReadSensM2D(*this,aNameIm);

     if (aSI!=nullptr)
     {
        if (ToDeleteAutom)
           cMMVII_Appli::AddObj2DelAtEnd(aSI);

        return;
     }


     if (!SVP)
     {
         std::string aErrorMessage = "Cannot get sensor for image " + aNameIm;
         if (mDPOrient.DirInIsInit())
         {
             aErrorMessage += " in Ori " + mDPOrient.DirIn();
         }
         MMVII_UserError
         (
             eTyUEr::eUnClassedError,
             aErrorMessage
         );
     }
}

cSensorImage* cPhotogrammetricProject::ReadSensorFromFolder(const std::string  & aFolder,const std::string  &aNameIm,bool ToDeleteAutom,bool SVP) const
{
     cAutoChgRestoreDefFolder  aCRDF(aFolder,DPOrient(),true); // Chg Folder and restore at destruction
     cSensorImage* aSensor = ReadSensor(aNameIm,true/*ToDelAutom*/);
     return aSensor;
}


cPerspCamIntrCalib *  cPhotogrammetricProject::InternalCalibFromImage(const std::string & aNameIm) const
{
    //  alloc sensor and if exist, extract internal, destroy
    //  else try to extract calib from standard name
    //    * case where nor calib nor pose exist, and must be created from xif still to implemant
    mDPOrient.AssertDirInIsInit();

    cSensorCamPC *  aPC = ReadCamPC(aNameIm,false,SVP::Yes);
    if (aPC==nullptr)
    {
        return InternalCalibFromStdName(aNameIm);
    }

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

cPerspCamIntrCalib *   cPhotogrammetricProject::InternalCalibFromStdName(const std::string aNameIm,bool isRemanent) const
{
    if (mDPOrient.DirInIsNONE())
       return nullptr;

    std::string aNameCalib = FullDirCalibIn() + StdNameCalibOfImage(aNameIm) + "." + TaggedNameDefSerial();
    cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::FromFile(aNameCalib,isRemanent);

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

cIm2D<tU_INT1>  cPhotogrammetricProject::MaskWithDef(const std::string & aNameImage,const cBox2di & aBox,bool DefVal,bool OkNoMasq) const
{
    if (ImageHasMask( aNameImage))
    {
        return cIm2D<tU_INT1>::FromFile(NameMaskOfImage(aNameImage),aBox);
    }

     MMVII_INTERNAL_ASSERT_always(OkNoMasq,"Masq dont exist for image : " + aNameImage);

    return cIm2D<tU_INT1> (aBox.Sz(),nullptr,  (DefVal ? eModeInitImage::eMIA_V1 : eModeInitImage::eMIA_Null)) ;
}

cIm2D<tU_INT1>  cPhotogrammetricProject::MaskOfImage(const std::string & aNameImage,const cBox2di & aBox) const
{
	return MaskWithDef(aNameImage,aBox,false,false);
}

        //  =============  PointsMeasures =================

void cPhotogrammetricProject::SaveMeasureIm(const cSetMesPtOf1Im &  aSetM) const
{
     aSetM.ToFile(mDPGndPt2D.FullDirOut() +aSetM.StdNameFile());
}

std::string cPhotogrammetricProject::NameMeasureGCPIm(const std::string & aNameIm,bool isIn) const
{
    return  mDPGndPt2D.FullDirInOut(isIn) + cSetMesPtOf1Im::StdNameFileOfIm(FileOfPath(aNameIm,false)) ;
}


bool cPhotogrammetricProject::HasMeasureIm(const std::string & aNameIm,bool InDir) const
{
   return ExistFile(NameMeasureGCPIm(aNameIm,InDir));
}

bool cPhotogrammetricProject::HasMeasureImFolder(const std::string & aFolder,const std::string & aNameIm) const
{
     cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt2D(), true); // Chg Folder and restore at destruction
     return HasMeasureIm(aNameIm,true);
}


cSetMesPtOf1Im cPhotogrammetricProject::LoadMeasureIm(const std::string & aNameIm,bool isIn) const
{
   //  std::string aDir = mDPPointsMeasures.FullDirInOut(isIn);
   //  return cSetMesPtOf1Im::FromFile(aDir+cSetMesPtOf1Im::StdNameFileOfIm(aNameIm));

   return cSetMesPtOf1Im::FromFile(NameMeasureGCPIm(aNameIm,isIn));
}

void cPhotogrammetricProject::SaveGCP3D(const cSetMesGnd3D & aMGCP3D, const std::string &aDefaultOutName, bool aDoAddCurSysCo) const
{
    std::map<std::string, MMVII::cSetMesGnd3D> aSplittedGCP3D = aMGCP3D.SplitPerOutDir(aDefaultOutName);
    for (const auto& [aDirName, aSetMesGnd3D] : aSplittedGCP3D)
    {
        if (!aDirName.empty()) // outname="" means do not export
        {
            cAutoChgRestoreDefFolder  aCRDF(aDirName,DPGndPt3D(),false); // Chg output Folder and restore at destruction
            aSetMesGnd3D.ToFile(mDPGndPt3D.FullDirOut() + aMGCP3D.StdNameFile());
            if (aDoAddCurSysCo)
                SaveCurSysCoGCP(CurSysCo(DPGndPt3D(),true));
        }
    }
}

std::string cPhotogrammetricProject::GCPPattern(const std::string & aArgPatFiltr) const
{
    return (aArgPatFiltr=="") ? (cSetMesGnd3D::ThePrefixFiles + ".*." +TaggedNameDefSerial())  : aArgPatFiltr;
}

std::vector<std::string>  cPhotogrammetricProject::ListFileGCP(const std::string & aArgPatFiltr) const
{
   std::string aPatFiltr = GCPPattern(aArgPatFiltr);
   std::string aDir = mDPGndPt3D.FullDirIn();
   std::vector<std::string> aRes;

   GetFilesFromDir(aRes,aDir,AllocRegex(aPatFiltr));

   for (auto & aName : aRes)
      aName = aDir + aName;

   return aRes;
}

void cPhotogrammetricProject::LoadGCP3D(cSetMesGndPt& aSetMes,cMes3DDirInfo * aMesDirInfo, const std::string & aArgPatFiltr,const std::string & aFiltrNameGCP,
                                      const std::string & aFiltrAdditionalInfoGCP) const
{
   std::vector<std::string> aListFileGCP = ListFileGCP(aArgPatFiltr);
   MMVII_INTERNAL_ASSERT_User(!aListFileGCP.empty(),eTyUEr::eUnClassedError,"No file found in LoadGCP");

   for (const auto  & aNameFile : aListFileGCP)
   {
       cSetMesGnd3D aMesGCP3D = cSetMesGnd3D::FromFile(aNameFile);
       if ( (!aFiltrNameGCP.empty()) || (!aFiltrAdditionalInfoGCP.empty()) )
          aMesGCP3D = aMesGCP3D.Filter(aFiltrNameGCP, aFiltrAdditionalInfoGCP);
       aSetMes.AddMes3D(aMesGCP3D, aMesDirInfo);
   }
}


cSetMesGnd3D cPhotogrammetricProject::LoadGCP3D() const
{
    cSetMesGndPt  aSetMesIm;
    LoadGCP3D(aSetMesIm);
    return aSetMesIm.AllMesGCP();
}

cSetMesGnd3D cPhotogrammetricProject::LoadGCP3DFromFolder(const std::string & aFolder) const
{
     cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt3D(),true); // Chg Folder and restore at destruction
     return  LoadGCP3D();
}



cSetMesPtOf1Im cPhotogrammetricProject::LoadMeasureImFromFolder(const std::string & aFolder,const std::string & aNameIm) const
{
     cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt2D(),true); // Chg Folder and restore at destruction
     return  LoadMeasureIm(aNameIm);
     
     // auto aRes  = LoadMeasureIm(aNameIm);
     // FakeUseIt(
     // return aRes;
     /*
     cDirsPhProj& aDPPM = const_cast<cPhotogrammetricProject *>(this)->DPPointsMeasures();
     // Save current orientation and fix new
     std::string aDirInit = aDPPM.DirIn();
     aDPPM.SetDirIn(aFolder);

     cSetMesPtOf1Im aRes = LoadMeasureIm(aNameIm);
     // Restore initial current orientation
     aDPPM.SetDirIn(aDirInit);

     return aRes;
     */

}


void cPhotogrammetricProject::LoadGCP3DFromFolder
     (const std::string & aFolder,
          cSetMesGndPt& aSetMes,
          MMVII::cMes3DDirInfo *aMesDirInfo,
          const std::string & aArgPatFiltr,
          const std::string & aFiltrNameGCP,
          const std::string & aFiltrAdditionalInfoGCP) const
{
     cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt3D(), true); // Chg Folder and restore at destruction
     LoadGCP3D(aSetMes,aMesDirInfo,aArgPatFiltr,aFiltrNameGCP,aFiltrAdditionalInfoGCP);
}

void cPhotogrammetricProject::CpGCPPattern(const std::string & aDirIn,const std::string & aDirOut,const std::string & aArgPatFiltr) const
{
   CopyPatternFile(aDirIn,GCPPattern(aArgPatFiltr),aDirOut);
   CopyPatternFile(aDirIn,"CurSysCo.xml",aDirOut);
}

void cPhotogrammetricProject::CpGCP() const
{
	CpGCPPattern(mDPGndPt3D.FullDirIn(),mDPGndPt3D.FullDirOut());
}

void cPhotogrammetricProject::CpMeasureIm() const
{
    CopyPatternFile
    (
        mDPGndPt2D.FullDirIn(),
	cSetMesPtOf1Im::ThePrefixFiles+ ".*"+ TaggedNameDefSerial(),
        mDPGndPt2D.FullDirOut()
    );
}




void cPhotogrammetricProject::LoadIm(cSetMesGndPt& aSetMes, const std::string & aNameIm, MMVII::cMes2DDirInfo *aMesDirInfo, cSensorImage * aSIm, bool SVP) const
{
//    std::string aDir = mDPPointsMeasures.FullDirIn();
   //cSetMesPtOf1Im  aSetIm = cSetMesPtOf1Im::FromFile(aDir+cSetMesPtOf1Im::StdNameFileOfIm(aNameIm));
   if (SVP && (! ExistFile(NameMeasureGCPIm(aNameIm,true))))
   {
      // StdOut() << "LoadImLoadIm " << aNameIm << "\n";
      return;
   }
      //  StdOut() << "LoadImLoadIm " << aNameIm << "\n";
   cSetMesPtOf1Im  aSetIm = LoadMeasureIm(aNameIm);
   aSetMes.AddMes2D(aSetIm,aMesDirInfo,aSIm);
}

void cPhotogrammetricProject::LoadImFromFolder
     (
           const std::string & aFolder,
           cSetMesGndPt& aSetMes,
           cMes2DDirInfo * aMesDirInfo,
           const std::string & aNameIm,
           cSensorImage * aSIm,bool SVP
     ) const
{
    cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt2D(), true); // Chg Folder and restore at destruction
    DPGndPt2D().CheckDirExists(true, true);
    LoadIm(aSetMes,aNameIm,aMesDirInfo,aSIm,SVP);
}

void cPhotogrammetricProject::LoadIm(cSetMesGndPt& aSetMes,MMVII::cMes2DDirInfo *aMesDirInfo, cSensorImage & aSIm) const
{
     LoadIm(aSetMes,aSIm.NameImage(),aMesDirInfo,&aSIm);
}

cSet2D3D  cPhotogrammetricProject::LoadSet32(const std::string & aNameIm) const
{
    cSetMesGndPt aSetMes;

    LoadGCP3D(aSetMes);
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
     // ============================   LINES ==============================================

std::string  cPhotogrammetricProject::NameFileLines(const std::string & aNameIm,bool isIn) const
{
    return DPGndPt2D().FullDirInOut(isIn) + "SegsAntiParal-"+ aNameIm + "."+ GlobTaggedNameDefSerial();
}

bool   cPhotogrammetricProject::HasFileLines(const std::string & aNameIm)  const
{
    return ExistFile(NameFileLines(aNameIm,IO::In));
}

bool   cPhotogrammetricProject::HasFileLinesFolder(const std::string & aFolder,const std::string & aNameIm)  const
{
    cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt2D(),true); // Chg Folder and restore at destruction
    return HasFileLines(aNameIm);
}




void  cPhotogrammetricProject::SaveLines(const cLinesAntiParal1Im &aLAP1I) const
{
    SaveInFile(aLAP1I,NameFileLines(aLAP1I.mNameIm,IO::Out));
}

cLinesAntiParal1Im  cPhotogrammetricProject::ReadLines(const std::string & aNameIm) const
{
    cLinesAntiParal1Im aRes;
    ReadFromFile(aRes,NameFileLines(aNameIm,IO::In));
    return aRes;
}

cLinesAntiParal1Im  cPhotogrammetricProject::ReadLinesFolder(const std::string & aFolder,const std::string & aNameIm) const
{
    cAutoChgRestoreDefFolder  aCRDF(aFolder,DPGndPt2D(),true); // Chg Folder and restore at destruction
    return ReadLines(aNameIm);
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

void  cPhotogrammetricProject::ReadMultipleTieP(cVecTiePMul& aVPm,const std::string & aNameIm,bool SVP) const
{
   std::string aNameFile = mDPMulTieP.FullDirIn()+NameMultipleTieP(aNameIm);
   if (! ExistFile(aNameFile))
   {
     MMVII_INTERNAL_ASSERT_User(SVP,eTyUEr::eUnClassedError,"Cannot find Multi Tie Points for " + aNameIm);
   }
   else
       ReadFromFile(aVPm.mVecTPM,mDPMulTieP.FullDirIn()+NameMultipleTieP(aNameIm));
   aVPm.mNameIm = aNameIm;
}

void  cPhotogrammetricProject::ReadMultipleTiePFromFolder(const std::string &  aFolder,cVecTiePMul& aVPm,const std::string & aNameIm,bool SVP) const
{
     cAutoChgRestoreDefFolder  aCRDF(aFolder,DPMulTieP(),true); // Chg Folder and restore at destruction
     ReadMultipleTieP(aVPm,aNameIm,SVP);
}




bool cPhotogrammetricProject::HasNbMinMultiTiePoints(const std::string & aNameIm,size_t aNbMinTieP,bool AcceptNoDirIn ) const
{
    if (!DPMulTieP().DirInIsInit())
    {
        MMVII_INTERNAL_ASSERT_strong(AcceptNoDirIn,"No DirInIsInit in HasNbMinMultiTiePoints");
        return true;
    }

    cVecTiePMul aVPM(aNameIm);
    ReadMultipleTieP(aVPM,aNameIm,true);
    return aVPM.mVecTPM.size() >= aNbMinTieP;
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
        //  =============  Clino meters  =================

std::string cPhotogrammetricProject::NameFileClino(const std::string &aNameCam,bool Input, const std::string aClinoName) const
{
    static const std::string TheClinoPrefix = "ClinoCalib-";
    return mDPClinoMeters.FullDirInOut(Input) + TheClinoPrefix + aClinoName + "-" + aNameCam + "."+ GlobTaggedNameDefSerial();
}

void cPhotogrammetricProject::SaveClino(const cCalibSetClino & aCalib) const
{
    std::vector<cOneCalibClino> aOneCalibClinoVector = aCalib.ClinosCal();
    std::string aCameraName = aCalib.NameCam();
    for (auto aOneCalibClino : aOneCalibClinoVector)
    {
        std::string aClinoName = aOneCalibClino.NameClino();
        SaveInFile(aOneCalibClino,NameFileClino(aCameraName,false, aClinoName));
    }
}

bool cPhotogrammetricProject::HasClinoCalib(const cPerspCamIntrCalib & aCalib, const std::string aClinoName) const
{
    return ExistFile(NameFileClino(aCalib.Name(),true, aClinoName));
}


void  cPhotogrammetricProject::ReadGetClino
      (
            cOneCalibClino& aCalClino,
            const cPerspCamIntrCalib & aCalibCam, 
            const std::string aClinoName
      ) const
{
    std::string aFileName = NameFileClino(aCalibCam.Name(),true, aClinoName);
    if (!ExistFile(aFileName))
    {
        MMVII_UserError(eTyUEr::eOpenFile, "Clino filename not found : " + aFileName);
    }
    ReadFromFile(aCalClino,aFileName);
}

cOneCalibClino * cPhotogrammetricProject::GetClino(const cPerspCamIntrCalib & aCalib, const std::string aClinoName) const
{
    cOneCalibClino * aResult = new cOneCalibClino;
    ReadGetClino(*aResult,aCalib,aClinoName);
    return aResult;
}

cCalibSetClino  cPhotogrammetricProject::ReadSetClino
                (  
                    const cPerspCamIntrCalib &        aCalib,   
                    const std::vector<std::string> &  aVecClinoName
                 ) const
{
   std::vector<cOneCalibClino> aVCC(aVecClinoName.size());
   for (size_t aK=0 ; aK<aVecClinoName.size() ; aK++)
       ReadGetClino(aVCC.at(aK),aCalib,aVecClinoName.at(aK));

   return cCalibSetClino(aCalib.Name(),aVCC);
}



            //  ================  Measures clino ===================

static const  std::string TheNameDefMeasureClino = "ClinoMeasures";
std::string cPhotogrammetricProject::NameFileMeasuresClino(bool Input,const std::string & aN0) const
{
     std::string  aNameFile = (aN0=="") ? (TheNameDefMeasureClino + "." +   GlobTaggedNameDefSerial() ) : aN0;

     return mDPMeasuresClino.FullDirInOut(Input) + aNameFile;
}

void cPhotogrammetricProject::SaveMeasureClino(const cSetMeasureClino & aSetM) const
{
     SaveInFile(const_cast<cSetMeasureClino&>(aSetM),NameFileMeasuresClino(false));
}

void cPhotogrammetricProject::ReadMeasureClino(cSetMeasureClino & aSet,const std::string * aPat) const
{
   ReadFromFile(aSet,NameFileMeasuresClino(true));
   if (aPat!=nullptr)
   {
      aSet.FilterByPatIdent(*aPat);
   }
}

cSetMeasureClino  cPhotogrammetricProject::ReadMeasureClino(const std::string * aPat) const
{
    cSetMeasureClino aRes;
    ReadMeasureClino(aRes,aPat);

    return aRes;
}





        //  =============  Rigid bloc  =================

	                   // RIGIDBLOC
static const std::string PrefixRigidBloc = "RigidBloc_";

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

cBlocOfCamera * cPhotogrammetricProject::ReadUnikBlocCam() const
{
    std::list<cBlocOfCamera *>   aListBloc = ReadBlocCams();
    MMVII_INTERNAL_ASSERT_tiny(aListBloc.size()==1,"Number of bloc ="+ ToStr(aListBloc.size()));
    return *(aListBloc.begin());
}

//  =============  Static Lidar  =================

cStaticLidar * cPhotogrammetricProject::ReadStaticLidar(const cDirsPhProj& aDP, const std::string &aScanName, bool ToDeleteAutom) const
{
    aDP.AssertDirInIsInit();
    std::string aScanFileName  =  aDP.FullDirIn() + aScanName;
    cStaticLidar * aScan =  cStaticLidar::FromFile(aScanFileName, aDP.FullDirIn());

    if (ToDeleteAutom)
       cMMVII_Appli::AddObj2DelAtEnd(aScan);
    return aScan;
}


//  =============  Topo Mes  =================

               // TOPO


void   cPhotogrammetricProject::SaveTopoMes(const cBA_Topo & aBATopo) const
{
    std::string  aName = mDPTopoMes.FullDirOut() + "TopoOut." + TaggedNameDefSerial();
    aBATopo.ToFile(aName);
}

std::vector<std::string> cPhotogrammetricProject::ReadTopoMes() const
{
    return GetFilesFromDir(mDPTopoMes.FullDirIn(),AllocRegex(std::string(".*")));
}


        //  =============  Meta Data =================

//  see cMetaDataImages.cpp

static const std::string PrefixTripletSet = "TripletSet_";

void cPhotogrammetricProject::SaveTriplets(const cTripletSet &aSet,bool  useXmlraterThanDmp) const
{
    std::string anExt = useXmlraterThanDmp ? PostF_DumpFiles  : PostF_DumpFiles;
    std::string aName =  mDPOriTriplets.FullDirOut() + PrefixTripletSet + aSet.Name() + "." + anExt;
    StdOut() << "aName: " << aName << std::endl;
    aSet.ToFile(aName);
}

cTripletSet * cPhotogrammetricProject::ReadTriplets() const
{
    std::vector<std::string> aVNames = GetFilesFromDir(mDPOriTriplets.FullDirIn(),AllocRegex(PrefixTripletSet+".*"));

    return cTripletSet::FromFile(mDPOriTriplets.FullDirIn()+aVNames[0]);

}

        //  =============  Instrument bloc =================

static const std::string  PREFIX_RIG_BL = "FileRB_";

std::string   cPhotogrammetricProject::NameRigBoI(const std::string & aName,bool isIn) const
{
    return DPBlockInstr().FullDirInOut(isIn) + PREFIX_RIG_BL + aName + "." + GlobTaggedNameDefSerial();
}

cIrbCal_Block *  cPhotogrammetricProject::ReadRigBoI(const std::string & aName,bool SVP) const
{
    std::string aFullName  = NameRigBoI(aName,IO::In);
    cIrbCal_Block * aRes = new cIrbCal_Block(aName);

    if (! ExistFile(aFullName))  // if it doesnt exist and we are OK, it return a new empty bloc
    {
        MMVII_INTERNAL_ASSERT_User_UndefE(SVP,"cIrbCal_Block file dont exist");
    }
    else
    {
        ReadFromFile(*aRes,aFullName);
    }

    return aRes;
}

void   cPhotogrammetricProject::SaveRigBoI(const cIrbCal_Block & aBloc) const
{
      SaveInFile(aBloc,NameRigBoI(aBloc.NameBloc(),IO::Out));
}

std::vector<std::string>  cPhotogrammetricProject::ListBlockExisting() const
{
    std::vector<std::string> aRes;

    std::vector<std::string>  aVec =  GetFilesFromDir
                                    (
                                        DPBlockInstr().FullDirIn(),
                                        AllocRegex(PREFIX_RIG_BL + ".*" + "." + GlobTaggedNameDefSerial())
                                     );

    for (const auto & aName : aVec)
    {
        aRes.push_back(LastPrefix(aName).substr(PREFIX_RIG_BL.length()));
    }
    return aRes;
}



}; // MMVII

