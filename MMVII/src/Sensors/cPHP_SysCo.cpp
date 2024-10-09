#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_BlocRig.h"
#include "MMVII_DeclareCste.h"
#include "cExternalSensor.h"



/**
   \file  cPhotogrammetricProject.cpp

   \brief file for handling names/upload/download of photogram data (pose,calib, ...)

   test Git
*/


namespace MMVII
{

        //  =============  coord system  =================

               // ----------------   ChSys ---------------------------

const cChangeSysCo & cPhotogrammetricProject::ChSysCo() const
{
   AssertChSysCoIsInit();
   return mChSysCo;
}

cChangeSysCo & cPhotogrammetricProject::ChSysCo()
{
   AssertChSysCoIsInit();
   return mChSysCo;
}

bool  cPhotogrammetricProject::ChSysCoIsInit() const { return mAppli.IsInit(&mNameChSysCo); }
void  cPhotogrammetricProject::AssertChSysCoIsInit() const
{
     MMVII_INTERNAL_ASSERT_strong(ChSysCoIsInit(),"Chang coord system is not init");
}



tPtrArg2007 cPhotogrammetricProject::ArgChSys(bool DefaultUndefined)
{
    std::vector<tSemA2007>   aVOpt{{eTA2007::ISizeV,"[1,2]"}};

    if (DefaultUndefined)
    {
        aVOpt.push_back(eTA2007::HDV);
        mAppli.SetVarInit(&mNameChSysCo);
        mNameChSysCo = {"Local"+MMVII_NONE};
    }
    return AOpt2007(mNameChSysCo,"ChSys","Change coordinate system, if 1 Sys In=Out",aVOpt);
}


void  cPhotogrammetricProject::AssertSysCoIsInit() const { MMVII_INTERNAL_ASSERT_strong(SysCoIsInit(),"Chang coord system is not init"); }
bool cPhotogrammetricProject::SysCoIsInit () const { return IsInit(&mNameCurSysCo); }

cSysCo & cPhotogrammetricProject::SysCo()
{
	AssertSysCoIsInit();
	return *mCurSysCo;
}
const cSysCo & cPhotogrammetricProject::SysCo()  const
{
	AssertSysCoIsInit();
	return *mCurSysCo;
}

tPtrArg2007  cPhotogrammetricProject::ArgSysCo()
{
    return AOpt2007(mNameCurSysCo,"SysCo","Name of coordinate system");
}

/*
         const cSysCoordV2 & SysCo() const ;
         bool  SysCoIsInit() const;
         void  AssertSysCoIsInit() const;
	 */


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

     // seems to be just a definition
     return aName;
}

void  cPhotogrammetricProject::SaveSysCo(tPtrSysCo aSys,const std::string& aName,bool OnlyIfNew) const
{
    std::string aFullName = mDirSysCo + aName + "."+  GlobTaggedNameDefSerial();

    if (OnlyIfNew && ExistFile(aFullName))
       return;
    SaveInFile(aSys->toSysCoData(), aFullName);
}

tPtrSysCo cPhotogrammetricProject::ReadSysCo(const std::string &aName, bool aDebug) const
{
    // compute name
    std::string aNameGlob = FullNameSysCo(aName,true);
    if (!ExistFile(aNameGlob))
    {
        return cSysCo::MakeSysCo(aName, aDebug);
    }
    return  cSysCo::FromFile(aNameGlob, aDebug);
}

tPtrSysCo cPhotogrammetricProject::CreateSysCoRTL(const cPt3dr & aOrig,const std::string &aNameRef,bool SVP) const
{
    std::string  aNameFull = FullNameSysCo(aNameRef,SVP);
    //if (aNameFull=="")
    //   return tPtrSysCo(nullptr);

    return cSysCo::makeRTL(aOrig,aNameFull);
}



cChangeSysCo cPhotogrammetricProject::ChangSysCo(const std::string aS1,const std::string aS2) const
{
    if (aS1==aS2)
       return  cChangeSysCo{};
    return cChangeSysCo(ReadSysCo(aS1),ReadSysCo(aS2));
}

cChangeSysCo cPhotogrammetricProject::ChangSysCo(const std::vector<std::string> & aVec,tREAL8 aEpsDif)
{
    if (! mAppli.IsInit(&aVec))  return cChangeSysCo{};

    if (aVec.size() == 1)
       return cChangeSysCo(ReadSysCo(aVec.at(0)),ReadSysCo(aVec.at(0)));

    return cChangeSysCo(   ReadSysCo(aVec.at(0))  ,  ReadSysCo(aVec.at(1))  );
}

             //==================   SysCo saved in standard folder ============

std::string  cPhotogrammetricProject::NameCurSysCo(const cDirsPhProj & aDP,bool IsIn) const
{
   return aDP.FullDirInOut(IsIn)   +  "CurSysCo." +  GlobTaggedNameDefSerial();

}

tPtrSysCo  cPhotogrammetricProject::CurSysCo(const cDirsPhProj & aDP,bool SVP) const
{
    std::string aPath = NameCurSysCo(aDP,true);
    if (! ExistFile(aPath))
    {
       if (! SVP)
           MMVII_UnclasseUsEr("CurSysCo dont exist : " + aPath);
       return tPtrSysCo(nullptr);
    }

    return  cSysCo::FromFile(aPath);
}


tPtrSysCo  cPhotogrammetricProject::CurSysCoOri(bool SVP) const {return CurSysCo(mDPOrient,SVP);}
tPtrSysCo  cPhotogrammetricProject::CurSysCoGCP(bool SVP) const {return CurSysCo(mDPPointsMeasures,SVP);}

void cPhotogrammetricProject::SaveCurSysCo(const cDirsPhProj & aDP,tPtrSysCo aSysCo) const
{
    SaveInFile(aSysCo->toSysCoData(),NameCurSysCo(aDP,false));
}


void cPhotogrammetricProject::SaveCurSysCoOri(tPtrSysCo aSysCo) const { SaveCurSysCo(mDPOrient,aSysCo); }
void cPhotogrammetricProject::SaveCurSysCoGCP(tPtrSysCo aSysCo) const { SaveCurSysCo(mDPPointsMeasures,aSysCo); }

void cPhotogrammetricProject::SaveStdCurSysCo(bool IsOri) const
{
     AssertSysCoIsInit();
     SaveCurSysCo((IsOri ? mDPOrient : mDPPointsMeasures),mCurSysCo);
}


void cPhotogrammetricProject::CpSysIn2Out(bool  OriIn,bool OriOut) const
{
   StdOut() << "ENTER_CpSysIn2Out\n";
   tPtrSysCo aSysIn = OriIn ?  CurSysCoOri(true) : CurSysCoGCP(true);

   StdOut() << "CpSysIn2OutCpSysIn2Out " << OriIn << " " << OriOut << " PTR=" << aSysIn.get() << "\n";

   if (aSysIn.get() == nullptr)
      return;

   if (OriOut)
      SaveCurSysCoOri(aSysIn);
   else
      SaveCurSysCoGCP(aSysIn);
}


void cPhotogrammetricProject::InitSysCoRTLIfNotReady(const cPt3dr & aCenter) 
{
    if (  (ChSysCo().SysTarget()->getType()==eSysCo::eRTL)  &&  (!ChSysCo().SysTarget()->isReady())  )
    {
        std::string aRTLName = ChSysCo().SysTarget()->Def();
        ChSysCo().setTargetsysCo(CreateSysCoRTL(aCenter,ChSysCo().SysOrigin()->Def()));
        SaveInFile(ChSysCo().SysTarget()->toSysCoData(), getDirSysCo() + aRTLName + "." + GlobTaggedNameDefSerial());
    }
}



}; // MMVII

