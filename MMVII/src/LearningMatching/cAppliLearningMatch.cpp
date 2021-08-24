#include "include/MMVII_all.h"
#include "LearnDM.h"

namespace MMVII
{

cAppliLearningMatch::cAppliLearningMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli(aVArgs,aSpec)
{
    SetNamesProject("","");
}

std::string  cAppliLearningMatch::NameReport() const
{
   return DirResult() + "Report_" + mSpecs.Name()  + "_" + mNameOutput + "_" + StrIdTime() + ".txt";
}

void cAppliLearningMatch::SetNamesProject (const std::string & aNameInput,const std::string & aNameOutput)
{
   mNameInput  = aNameInput;
   mNameOutput = aNameOutput;
   CreateDirectories(DirVisu(),true);
   CreateDirectories(DirResult(),true);
   if (mNameInput!="") CreateDirectories(SubDirResult(true),true);
   if (mNameOutput!="") CreateDirectories(SubDirResult(false),true);
}

std::string cAppliLearningMatch::Prefix(bool isIn) const {return isIn ? mNameInput : mNameOutput;}
std::string cAppliLearningMatch::Post(bool isXml) {return StdPostF_ArMMVII(isXml) ; }
std::string cAppliLearningMatch:: DirVisu()  const  {return "DirVisu" +  StringDirSeparator() +mNameOutput +  StringDirSeparator() ;}
std::string cAppliLearningMatch:: DirResult()  const  { return std::string("Result") + StringDirSeparator(); }
std::string cAppliLearningMatch:: SubDirResult(bool isIn)  const  
{ 
    return DirResult() + Prefix(isIn)  + StringDirSeparator(); 
}

std::string cAppliLearningMatch:: FileHisto1Carac(bool isIn,bool isXml)   const 
{
   return SubDirResult(isIn) + "Histo1Carac" + "."+ Post(isXml);
}

std::string cAppliLearningMatch::FileHistoNDIm(const std::string & aName,bool isIn)  const
{
   return SubDirResult(isIn) + "HND_"  +  aName + "." + PostF_DumpFiles;
}


std::string  cAppliLearningMatch::PrefixAll()   {return "DMTrain_";}
std::string  cAppliLearningMatch::Im1()   {return "Im1";}
std::string  cAppliLearningMatch::Im2()   {return "Im2";}
std::string  cAppliLearningMatch::Px1()   {return "Pax1";}
std::string  cAppliLearningMatch::Px2()   {return "Pax2";}
std::string  cAppliLearningMatch::Masq1() {return "Masq1";}
std::string  cAppliLearningMatch::Masq2() {return "Masq2";}

std::string cAppliLearningMatch::MakeName(const std::string & aName,const std::string & aPref)
{
    return PrefixAll() + aName + "_" + aPref + ".tif";
}

void cAppliLearningMatch::GenConvertIm(const std::string & aInput, const std::string & aOutput)
{
    std::string aCom =   "convert -colorspace Gray -compress none " + aInput + " " + aOutput;
    GlobSysCall(aCom);
}

std::string cAppliLearningMatch::NameIm1(const std::string & aName) {return MakeName(aName,Im1());}
std::string cAppliLearningMatch::NameIm2(const std::string & aName) {return MakeName(aName,Im2());}
std::string cAppliLearningMatch::NamePx1(const std::string & aName) {return MakeName(aName,Px1());}
std::string cAppliLearningMatch::NamePx2(const std::string & aName) {return MakeName(aName,Px2());}
std::string cAppliLearningMatch::NameMasq1(const std::string & aName) {return MakeName(aName,Masq1());}
std::string cAppliLearningMatch::NameMasq2(const std::string & aName) {return MakeName(aName,Masq2());}

std::string cAppliLearningMatch::NameRedrIm1(const std::string & aName) {return MakeName(aName,"REDRIn_"+Im1());}
std::string cAppliLearningMatch::NameRedrIm2(const std::string & aName) {return MakeName(aName,"REDRIn_"+Im2());}

void cAppliLearningMatch::ConvertIm1(const std::string & aInput,const std::string & aName) {GenConvertIm(aInput,NameIm1(aName));}
void cAppliLearningMatch::ConvertIm2(const std::string & aInput,const std::string & aName) {GenConvertIm(aInput,NameIm2(aName));}

std::string cAppliLearningMatch::Im2FromIm1(const std::string & aIm1)
{
  return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif","_"+Im2()+".tif");
}
std::string cAppliLearningMatch::Px1FromIm1(const std::string & aIm1)
{
   return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif","_"+Px1()+".tif");
}
std::string cAppliLearningMatch::Masq1FromIm1(const std::string & aIm1)
{
   return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif","_"+Masq1()+".tif");
}


std::string  cAppliLearningMatch::PrefixHom()    {return "LDHAime";}  // Learn Dense Home
std::string  cAppliLearningMatch::Hom(int aNum)  {return PrefixHom() +ToStr(aNum);}
std::string  cAppliLearningMatch::Index(int aNum)  {return "Box" +ToStr(aNum);}
std::string cAppliLearningMatch::HomFromIm1(const std::string & aIm1,int aNumHom,std::string anExt,bool isXml)
{
    std::string aPost =  "_" + anExt+ "_" + Hom(aNumHom) + "." + Post(isXml);
    return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif",aPost);
}
std::string cAppliLearningMatch::HomFromHom0(const std::string & aName,int aNumHom)
{
    return replaceFirstOccurrence(aName,Hom(0),Hom(aNumHom));
}



};
