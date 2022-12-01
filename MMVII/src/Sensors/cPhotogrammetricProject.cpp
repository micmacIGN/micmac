#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"

/**
   \file  cPhotogrammetricProject.cpp

   \brief file for handling names/upload/download of photogram data (pose,calib, ...)
*/


namespace MMVII
{

std::string SuppresDir(const std::string & aDir,const std::string & aName)
{
    // mOriIn.starts_with(aDir);  -> C++20
    // to see if StringDirSeparator() is not a meta carac on window ?

     if (TheSYS == eSYS::Windows)
     {
          MMVII_DEV_WARNING("SuppresDir check regular expression on Window");
     }
     
     std::string aPat =  "(" + aDir+")?" + "([A-Za-z0-9_-]+)" + StringDirSeparator() + "?";
     if (! MatchRegex(aName,aPat))
     {
	     MMVII_UsersErrror(eTyUEr::eUnClassedError,"No match for subdir, with name=" + aName + " Dir=" + aDir);
     }
     std::string aRes =  ReplacePattern(aPat,"$2",aName);
	     
     return aRes;
}




   /* ********************************************************** */
   /*                                                            */
   /*                 cPhotogrammetricProject                    */
   /*                                                            */
   /* ********************************************************** */

        //  =============  Construction & destuction =================

cPhotogrammetricProject::cPhotogrammetricProject(cMMVII_Appli & anAppli) :
    mAppli  (anAppli)
{
}

void cPhotogrammetricProject::AssertOriInIsInit()    const
{
     MMVII_INTERNAL_ASSERT_tiny(mAppli.IsInit(&mOriIn),"Orientation required non init");
}

void cPhotogrammetricProject::AssertRadiomInIsInit() const
{
     MMVII_INTERNAL_ASSERT_tiny(mAppli.IsInit(&mRadiomIn),"Radiometry required non init");
}


void cPhotogrammetricProject::FinishInit() 
{
    // the user can give the full directory, which may be usefull with car completion
    if (mAppli.IsInit(&mOriIn))  // dont do it if OriIn not used ...
        mOriIn  = SuppresDir(MMVIIDirOrient,mOriIn);

    mFullOriIn   = mAppli.DirProject() + MMVIIDirOrient + mOriIn  + StringDirSeparator();
    mFullOriOut  = mAppli.DirProject() + MMVIIDirOrient + mOriOut + StringDirSeparator();
    if (mAppli.IsInit(&mOriOut))
    {
        CreateDirectories(mFullOriOut,true);
    }


    if (mAppli.IsInit(&mRadiomIn))  // dont do it if mRadiomIn not used ...
        mRadiomIn  = SuppresDir(MMVIIDirRadiom,mRadiomIn);

    mFullRadiomIn  =   mAppli.DirProject() +  MMVIIDirRadiom + mRadiomIn  + StringDirSeparator();
    if (mAppli.IsInSpec(&mRadiomOut) &&  (! mAppli.IsInit(&mRadiomOut)))
    {
       mRadiomOut = mRadiomIn;
    }
    mFullRadiomOut =   mAppli.DirProject() +  MMVIIDirRadiom + mRadiomOut + StringDirSeparator();
    if (mAppli.IsInSpec(&mRadiomOut) )
    {
        CreateDirectories(mFullRadiomOut,true);
    }
}

cPhotogrammetricProject::~cPhotogrammetricProject() 
{
    DeleteAllAndClear(mLCam2Del);
}

        //  =============  Arg processing =================
tPtrArg2007 cPhotogrammetricProject::CalibInMand(){return Arg2007(mOriIn ,"Input Calibration",{eTA2007::Orient,eTA2007::Input });}

tPtrArg2007 cPhotogrammetricProject::OriInMand() {return  Arg2007(mOriIn ,"Input Orientation",{eTA2007::Orient,eTA2007::Input });}
tPtrArg2007 cPhotogrammetricProject::OriOutMand() {return Arg2007(mOriOut,"Output Orientation",{eTA2007::Orient,eTA2007::Output});}
tPtrArg2007 cPhotogrammetricProject::OriInOpt(){return AOpt2007(mOriIn,"InOri","Input Orientation",{eTA2007::Orient,eTA2007::Input});}

tPtrArg2007  cPhotogrammetricProject::RadiomOptOut() 
  {return AOpt2007(mRadiomOut,"OutRad","Output Radiometry ",{eTA2007::Radiom,eTA2007::Output});}

tPtrArg2007  cPhotogrammetricProject::RadiomOptIn() {return AOpt2007(mRadiomIn,"InRad","Output Radiometry ",{eTA2007::Radiom,eTA2007::Input});}

tPtrArg2007 cPhotogrammetricProject::RadiomInMand() {return Arg2007(mRadiomIn,"Input Radiometry",{eTA2007::Radiom,eTA2007::Input});}

bool  cPhotogrammetricProject::RadiomOptOutIsInit() const {return mAppli.IsInit(&mRadiomOut);}
bool  cPhotogrammetricProject::RadiomOptInIsInit() const {return mAppli.IsInit(&mRadiomIn);}


        //  =============  Saving object =================

void cPhotogrammetricProject::SaveCamPC(const cSensorCamPC & aCamPC) const
{
    aCamPC.ToFile(mFullOriOut + aCamPC.NameOriStd());
}

void cPhotogrammetricProject::SaveRadiomData(const cImageRadiomData & anIRD) const
{
    anIRD.ToFile(mFullRadiomOut+anIRD.NameFile());
}

void cPhotogrammetricProject::SaveCalibRad(const cCalibRadiomIma & aCalRad) const
{
     aCalRad.ToFile(mFullRadiomOut + PrefixCalRadRad + aCalRad.NameIm()+ "." + PostF_XmlFiles);
}

cCalibRadiomIma * cPhotogrammetricProject::AllocCalibRadiomIma(const std::string & aNameIm) const
{
/* With only the name of images and the folder, cannot determinate the model used, so the methods
 * test the possible model by testing existence of files.
 */	
    std::string aNameFile = mFullRadiomIn + PrefixCalRadRad + aNameIm + "." + PostF_XmlFiles;
    if (ExistFile(aNameFile))
       return cCalRadIm_Cst::FromFile(aNameFile);

   MMVII_UsersErrror(eTyUEr::eUnClassedError,"Cannot determine Image RadiomCalib  for :" + aNameIm + " in " + mFullRadiomIn);
   return nullptr;
}



        //  =============  Creating object =================

cSensorCamPC * cPhotogrammetricProject::AllocCamPC(const std::string & aNameIm,bool ToDelete)
{
    AssertOriInIsInit();

    std::string aNameCam  = mFullOriIn + cSensorCamPC::NameOri_From_Image(aNameIm);
    cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam);

    if (ToDelete)
       mLCam2Del.push_back(aCamPC);

    return aCamPC;
}

cImageRadiomData * cPhotogrammetricProject::AllocRadiomData(const std::string & aNameIm) const
{
    AssertRadiomInIsInit();

    std::string aFullName  = mFullRadiomIn + cImageRadiomData::NameFileOfImage(aNameIm);
    return cImageRadiomData::FromFile(aFullName);
}

cPerspCamIntrCalib *  cPhotogrammetricProject::AllocCalib(const std::string & aNameIm)
{
    // 4 now, pretty basic allox sensor, extract internal, destroy
    // later will have to handle :
    //    * case where calib exist but not pose
    //    * case where nor calib nor pose exist, and must be created from xif 
    AssertOriInIsInit();

    cSensorCamPC *  aPC = AllocCamPC(aNameIm,false);
    cPerspCamIntrCalib * aCalib = aPC->InternalCalib();
    delete aPC;

    return aCalib;
}

cMedaDataImage cPhotogrammetricProject::GetMetaData(const std::string & aNameIm) const
{
   static std::map<std::string,cMedaDataImage> aMap;
   auto  anIt = aMap.find(aNameIm);
   if (anIt== aMap.end())
   {
        aMap[aNameIm] = cMedaDataImage(aNameIm);
   }

   return aMap[aNameIm];
}

std::string cPhotogrammetricProject::NameCalibRadiomSensor(const cPerspCamIntrCalib & aCam,const cMedaDataImage & aMTD) const
{
    return  PrefixCalRadRad  + "Sensor-" + aCam.Name() + "-Aperture_" + ToStr(aMTD.Aperture());
}



        //  =============  Accessor/Modiier to dir =================

const std::string & cPhotogrammetricProject::GetOriIn() const {return mOriIn;}
void cPhotogrammetricProject::SetOriIn(const std::string & aNameOri)
{
     mOriIn = aNameOri;
     mAppli.SetVarInit(&mOriIn); // required becaus of AssertOriInIsInit
}



}; // MMVII

