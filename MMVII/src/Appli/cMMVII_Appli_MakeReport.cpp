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


void  cMMVII_Appli::InitReport(const std::string &anId,const std::string &aPost,bool IsMul)
{
    mDoMergeReport = (IsMul && (LevelCall()==0));
    if (LevelCall()==0)
    {
         CreateDirectories(DirReport(),true);
	 mMapIdFilesReport[anId] = DirReport()+anId + "." + aPost;
	 if (IsMul)
            CreateDirectories(DirSubPReport(anId),true);
    }
    else if (LevelCall()==1)
    {
        mMapIdFilesReport[anId]  = DirSubPReport(anId) + FileOfPath(UniqueStr(0))  +"." + aPost;
    }
    else
        return;
    mMapIdPostReport[anId] = aPost;

    cMMVII_Ofs(mMapIdFilesReport[anId], eFileModeOut::CreateText);
}

void  cMMVII_Appli::AddTopReport(const std::string &anId,const std::string & aMsg)
{
    if (LevelCall()!=0)
       return;
    
    AddOneReport(anId,aMsg);
}

void  cMMVII_Appli::AddOneReport(const std::string &anId,const std::string & aMsg)
{
    std::string  aName = mMapIdFilesReport[anId];
    MMVII_INTERNAL_ASSERT_tiny(aName!="","No file in AddOneMesCSV");
    cMMVII_Ofs aFile(aName, eFileModeOut::AppendText);

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
         cMMVII_Ofs aFileGlob(anIt.second, eFileModeOut::AppendText);
         const std::string & anId = anIt.first;
StdOut() << "DoMergeReportDoMergeReport " << __LINE__ << "\n";
	 for (const auto & aNameIm : VectMainSet(0))
	 {
             std::string aNameIn = DirSubPReport(anId) + FileOfPath(aNameIm) + "." + mMapIdPostReport[anId];
	     cMMVII_Ifs aIn(aNameIn, eFileModeIn::Text);

	     std::string aLine;
	     while (std::getline(aIn.Ifs(), aLine))
	     {
	          aFileGlob.Ofs() << aLine<< "\n";
	     }

	 }
StdOut() << "DoMergeReportDoMergeReport " << __LINE__ << "\n";
         RemoveRecurs(DirSubPReport(anId),false,false);
StdOut() << "DoMergeReportDoMergeReport " << __LINE__ << "\n";
     }
}


};
