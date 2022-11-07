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
          MMVII_WARGNING("SuppresDir check regular expression on Window");
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

void cPhotogrammetricProject::FinishInit() 
{
    // the user can give the full directory, which may be usefull with car completion
    if (mAppli.IsInit(&mOriIn))  // dont do it if OriIn not used ...
        mOriIn  = SuppresDir(MMVIIDirOrient,mOriIn);

    mFullOriOut  = mAppli.DirProject() + MMVIIDirOrient + mOriOut + StringDirSeparator();
    mFullOriIn   = mAppli.DirProject() + MMVIIDirOrient + mOriIn  + StringDirSeparator();
    if (mAppli.IsInit(&mOriOut))
    {
        CreateDirectories(mFullOriOut,true);
    }


    if (mAppli.IsInit(&mRadiomIn))  // dont do it if mRadiomIn not used ...
        mRadiomIn  = SuppresDir(MMVIIDirRadiom,mRadiomIn);

    mFullRadiomIn  =   mAppli.DirProject() +  MMVIIDirRadiom + mRadiomIn  + StringDirSeparator();
    mFullRadiomOut =   mAppli.DirProject() +  MMVIIDirRadiom + mRadiomOut + StringDirSeparator();
    if (mAppli.IsInit(&mRadiomOut))
    {
        CreateDirectories(mFullRadiomOut,true);
    }
}

cPhotogrammetricProject::~cPhotogrammetricProject() 
{
    DeleteAllAndClear(mLCam2Del);
}

        //  =============  Arg processing =================

tPtrArg2007 cPhotogrammetricProject::OriInMand() {return  Arg2007(mOriIn ,"Input Orientation",{eTA2007::Orient,eTA2007::Input });}
tPtrArg2007 cPhotogrammetricProject::OriOutMand() {return Arg2007(mOriOut,"Output Orientation",{eTA2007::Orient,eTA2007::Output});}
tPtrArg2007 cPhotogrammetricProject::OriInOpt(){return AOpt2007(mOriIn,"InOri","Input Orientation",{eTA2007::Orient,eTA2007::Input});}
tPtrArg2007  cPhotogrammetricProject::RadiomOptOut() 
  {return AOpt2007(mRadiomOut,"OutRad","Output Radiometry ",{eTA2007::Radiom,eTA2007::Output});}
tPtrArg2007 cPhotogrammetricProject::RadiomInMand() {return Arg2007(mRadiomIn,"Input Radiometry",{eTA2007::Radiom,eTA2007::Input});}

bool  cPhotogrammetricProject::RadiomOptOutIsInit() const {return mAppli.IsInit(&mRadiomOut);}


        //  =============  Saving object =================

void cPhotogrammetricProject::SaveCamPC(const cSensorCamPC & aCamPC) const
{
    aCamPC.ToFile(mFullOriOut + aCamPC.NameOriStd());
}

void cPhotogrammetricProject::SaveRadiomData(const cImageRadiomData & anIRD) const
{
    anIRD.ToFile(mFullRadiomOut+anIRD.NameFile());
}

        //  =============  Creating object =================

cSensorCamPC * cPhotogrammetricProject::AllocCamPC(const std::string & aNameIm,bool ToDelete)
{
    std::string aNameCam  = mFullOriIn + cSensorCamPC::NameOri_From_Image(aNameIm);
    cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam);

    if (ToDelete)
       mLCam2Del.push_back(aCamPC);

    return aCamPC;
}

cImageRadiomData * cPhotogrammetricProject::AllocRadiom(const std::string & aNameIm)
{
    std::string aFullName  = mFullRadiomIn + cImageRadiomData::NameFileOfImage(aNameIm);
    return cImageRadiomData::FromFile(aFullName);
}





        //  =============  Accessor/Modiier to dir =================

const std::string & cPhotogrammetricProject::GetOriIn() const {return mOriIn;}
void cPhotogrammetricProject::SetOriIn(const std::string & aNameOri)
{
	mOriIn = aNameOri;
}



}; // MMVII

