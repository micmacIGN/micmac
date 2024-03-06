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

const cChangSysCoordV2 & cPhotogrammetricProject::ChSys() const 
{
   AssertChSysIsInit();
   return mChSys;
}
cChangSysCoordV2 & cPhotogrammetricProject::ChSys() 
{
   AssertChSysIsInit();
   return mChSys;
}

bool  cPhotogrammetricProject::ChSysIsInit() const { return mAppli.IsInit(&mNameChSys); }
void  cPhotogrammetricProject::AssertChSysIsInit() const
{
     MMVII_INTERNAL_ASSERT_strong(ChSysIsInit(),"Chang coord system is not init");
}


tPtrArg2007 cPhotogrammetricProject::ArgChSys(bool DefaultUndefined)
{
    std::vector<tSemA2007>   aVOpt{{eTA2007::ISizeV,"[1,2]"}};

    if (DefaultUndefined)
    {
        aVOpt.push_back(eTA2007::HDV);
        mAppli.SetVarInit(&mNameChSys);
        mNameChSys = {"Local"+MMVII_NONE};
    }
    return AOpt2007(mNameChSys,"ChSys","Change coordinate system, if 1 Sys In=Out",aVOpt);
}

               // ----------------   ChSys ---------------------------
	       
struct cRefSysCo
{
   public :
	cRefSysCo() : mName (MMVII_NONE) {}
	cRefSysCo(const cSysCoordV2 & aSys) : mName (aSys.Name()) {}
	std::string mName;
};

void AddData (const cAuxAr2007 & anAux0,cRefSysCo & aRef)
{
     cAuxAr2007 anAux("Reference",anAux0);
     AddData(cAuxAr2007("Name",anAux),aRef.mName);
}


void  cPhotogrammetricProject::AssertSysCoIsInit() const { MMVII_INTERNAL_ASSERT_strong(SysCoIsInit(),"Chang coord system is not init"); }
bool cPhotogrammetricProject::SysCoIsInit () const { return IsInit(&mNameCurSysCo); }

cSysCoordV2 & cPhotogrammetricProject::SysCo() 
{
	AssertSysCoIsInit();
	return *mCurSysCo;
}
const cSysCoordV2 & cPhotogrammetricProject::SysCo()  const
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
     if (starts_with(aName,MMVII_LocalSys))
     {
         std::string aSubstr = aName.substr(MMVII_LocalSys.size(),std::string::npos);
         tPtrSysCo aRes = cSysCoordV2::LocalSystem(aSubstr);
         //  see if it already exist 
         tPtrSysCo aReRead = ReadSysCo(aSubstr,SVP::Yes);
         //  if not create it
         if (aReRead.get() == nullptr)
            SaveSysCo(aRes,aSubstr);

         return aRes;
     }


     // compute name
     std::string aNameGlob = FullNameSysCo(aName,SVP);
     if (aNameGlob=="") // if we are here, SVP=true and we can return nullptr
         return tPtrSysCo(nullptr);
     return  cSysCoordV2::FromFile(aNameGlob);
}

tPtrSysCo cPhotogrammetricProject::CreateSysCoRTL(const std::string& aNameResult,const cPt3dr & aOrig,const std::string &aNameRef,bool SVP) const
{
    std::string  aNameFull = FullNameSysCo(aNameRef,SVP);
    if (aNameFull=="")
       return tPtrSysCo(nullptr);

    return cSysCoordV2::RTL(aNameResult,aOrig,aNameFull);
}

cChangSysCoordV2  cPhotogrammetricProject::ChangSys(const std::string aS1,const std::string aS2) const
{
   if (aS1==aS2)
      return  cChangSysCoordV2{};	   

    return cChangSysCoordV2(ReadSysCo(aS1),ReadSysCo(aS2));
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
    cRefSysCo aRef;
    ReadFromFile(aRef,aName);

    return  ReadSysCo(aRef.mName,false);
}


tPtrSysCo  cPhotogrammetricProject::CurSysCoOri(bool SVP) const {return CurSysCo(mDPOrient,SVP);}
tPtrSysCo  cPhotogrammetricProject::CurSysCoGCP(bool SVP) const {return CurSysCo(mDPPointsMeasures,SVP);}

void cPhotogrammetricProject::SaveCurSysCo(const cDirsPhProj & aDP,tPtrSysCo aSysCo) const 
{
    SaveInFile(cRefSysCo(*aSysCo),NameCurSysCo(aDP,false));
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



}; // MMVII

