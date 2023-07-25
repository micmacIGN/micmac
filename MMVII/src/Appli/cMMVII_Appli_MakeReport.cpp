#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"


namespace MMVII
{

std::string  cMMVII_Appli::DirReport()
{
     std::string aRes =  DirProject() + MMVII_DirPhp + "Reports" + StringDirSeparator();
     mCSVSep = ',';
     return aRes;
}

std::string  cMMVII_Appli::DirSubPReport(const std::string &anId)
{
      return DirReport() + "Tmp-" + anId + StringDirSeparator();
}


void  cMMVII_Appli::InitReport(const std::string &anId,const std::string &aPost,eModeCall aMode)
{
    mDoMergeReport = aMode==eModeCall::eMulTop;
    if (LevelCall()==0)
    {
         CreateDirectories(DirReport(),true);
	 mMapIdFilesReport[anId] = DirReport()+anId + "." + aPost;
	 if (aMode==eModeCall::eMulTop)
            CreateDirectories(DirSubPReport(anId),true);
    }
    else if (LevelCall()==1)
    {
        mMapIdFilesReport[anId]  = DirSubPReport(anId) + FileOfPath(UniqueStr(0))  +"." + aPost;
    }
    mMapIdPostReport[anId] = aPost;

    cMMVII_Ofs(mMapIdFilesReport[anId],false);
}

void  cMMVII_Appli::AddOneReport(const std::string &anId,const std::string & aMsg)
{
    std::string  aName = mMapIdFilesReport[anId];
    MMVII_INTERNAL_ASSERT_tiny(aName!="","No file in AddOneMesCSV");
    cMMVII_Ofs aFile(aName,true);

    aFile.Ofs() << aMsg;
}

void  cMMVII_Appli::AddOneReportCSV(const std::string &anId,const std::vector<std::string> & aVecMsg)
{
    MMVII_INTERNAL_ASSERT_tiny(!aVecMsg.empty(),"No file in AddOneMesCSV");
    std::string aLine = aVecMsg.at(0);
    for (size_t aK=1 ; aK<aVecMsg.size() ; aK++)
    {
        aLine += mCSVSep;
        aLine +=  aVecMsg.at(aK);
    }
    aLine += "\n";
    AddOneReport(anId,aLine);
}



void  cMMVII_Appli::DoMergeReport()
{
     if (! mDoMergeReport)
        return;

     for (const auto & anIt : mMapIdFilesReport)
     {
         cMMVII_Ofs aFileGlob(anIt.second,true);
         const std::string & anId = anIt.first;
	 for (const auto & aNameIm : VectMainSet(0))
	 {
             std::string aNameIn = DirSubPReport(anId) + FileOfPath(aNameIm) + "." + mMapIdPostReport[anId];
	     aFileGlob.Ofs() << aNameIn << "\n";
	 }
     }
}


};
