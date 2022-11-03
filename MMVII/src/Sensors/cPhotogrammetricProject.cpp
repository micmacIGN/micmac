#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Sys.h"

/**
   \file  cPhotogrammetricProject.cpp

   \brief file for handling names/upload/download of photogram data (pose,calib, ...)
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cPhotogrammetricProject                    */
   /*                                                            */
   /* ********************************************************** */


cPhotogrammetricProject::cPhotogrammetricProject(cMMVII_Appli & anAppli) :
    mAppli  (anAppli)
{
}

cPhotogrammetricProject::~cPhotogrammetricProject() 
{
    DeleteAllAndClear(mLCam2Del);
}

tPtrArg2007 cPhotogrammetricProject::OriInMand() {return  Arg2007(mOriIn ,"Input Orientation",{eTA2007::Orient,eTA2007::Input });}
tPtrArg2007 cPhotogrammetricProject::OriOutMand() {return Arg2007(mOriOut,"Outot Orientation",{eTA2007::Orient,eTA2007::Output});}
tPtrArg2007 cPhotogrammetricProject::OriInOpt(){return AOpt2007(mOriIn,"InOri","Input Orientation",{eTA2007::Orient,eTA2007::Input});}


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
}

void cPhotogrammetricProject::SaveCamPC(const cSensorCamPC & aCamPC) const
{
    aCamPC.ToFile(mFullOriOut + aCamPC.NameOriStd());
}

cSensorCamPC * cPhotogrammetricProject::AllocCamPC(const std::string & aNameIm,bool ToDelete)
{
    std::string aNameCam  = mFullOriIn + cSensorCamPC::NameOri_From_Image(aNameIm);
    cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam);

    if (ToDelete)
       mLCam2Del.push_back(aCamPC);

    return aCamPC;
}

const std::string & cPhotogrammetricProject::GetOriIn() const {return mOriIn;}
void cPhotogrammetricProject::SetOriIn(const std::string & aNameOri)
{
	mOriIn = aNameOri;
}



}; // MMVII

