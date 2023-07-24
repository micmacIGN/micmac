#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"


namespace MMVII
{

std::string  cMMVII_Appli::DirCSV()
{
     std::string aRes =  DirProject() + MMVII_DirPhp + "CSV-Reports" + StringDirSeparator();
     mCSVSep = ',';
     return aRes;
}

std::string  cMMVII_Appli::DirSubPCSV(const std::string &anId)
{
      return DirCSV() + "Tmp-" + anId + StringDirSeparator();
}


void  cMMVII_Appli::InitDirCSV(const std::string &anId,eModeCall aMode)
{
    mDoMergeCVS = aMode==eModeCall::eMulTop;
    if (LevelCall()==0)
    {
         CreateDirectories(DirCSV(),true);
	 mMapIdFilesCSV[anId] = DirCSV()+anId + ".csv";
	 if (aMode==eModeCall::eMulTop)
            CreateDirectories(DirSubPCSV(anId),true);
    }
    else if (LevelCall()==1)
    {
        mMapIdFilesCSV[anId]  = DirSubPCSV(anId) + FileOfPath(UniqueStr(0))  +".csv";
    }

    cMMVII_Ofs(mMapIdFilesCSV[anId],false);
}

void  cMMVII_Appli::AddOneMesCSV(const std::string &anId,const std::vector<std::string> & aVecMsg)
{
     std::string  aName = mMapIdFilesCSV[anId];
     MMVII_INTERNAL_ASSERT_tiny(aName!="","No file in AddOneMesCSV");
     MMVII_INTERNAL_ASSERT_tiny(!aVecMsg.empty(),"No file in AddOneMesCSV");


    cMMVII_Ofs aFile(mMapIdFilesCSV[anId],false);

    aFile.Ofs() << aVecMsg.at(0);
    for (size_t aK=1 ; aK<aVecMsg.size() ; aK++)
    {
        aFile.Ofs() << mCSVSep;
        aFile.Ofs() << aVecMsg.at(aK);
    }
    aFile.Ofs() << "\n";
}

void  cMMVII_Appli::DoMergeCSV()
{
     if (! mDoMergeCVS)
        return;

     for (const auto & anIt : mMapIdFilesCSV)
     {
FakeUseIt(anIt);
     }
}





	/*
void  cMMVII_Appli::StdAddOneMesCSV(const std::string &anId,const std::vector<std::string> & VecMsg)
{
}


void  cMMVII_Appli::StdMergeMesCSV(const std::string &anId,const std::vector<std::string> & VecMsg)
{
}
void  TopP_StdAddOneMesCSV(const std::string &anId,const std::vector<std::string> & VecMsg);
void  SubP_StdAddOneMesCSV(const std::string &anId,const std::vector<std::string> & VecMsg);
void  TopP_InitDirTmpCSV(const std::string &anId);
void  TopP_StdMergeMesCSV(const std::string &anId,bool PurgeTmp);

MMVII_DirPhp

*/


};
