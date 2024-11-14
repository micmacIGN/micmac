#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Matrix.h"


namespace MMVII
{

std::string  cMMVII_Appli::DirReport()
{
     std::string aRes =    DirProject() + MMVII_DirPhp 
	                + "Reports" + StringDirSeparator()
	                + mSpecs.Name() + StringDirSeparator()
                  ;

     if (mReportSubDir!="")
        aRes = aRes + mReportSubDir + StringDirSeparator();

     mCSVSep = ',';
     return aRes;
}

void cMMVII_Appli::SetReportSubDir(const std::string & aSubDir)
{
   mReportSubDir = aSubDir;
}

void  cMMVII_Appli::SetReportRedir(const std::string &anId,const std::string & aNewDir)
{
    CreateDirectories(aNewDir,false);
    mMapIdRedirect[anId] = aNewDir;
}


std::string  cMMVII_Appli::DirSubPReport(const std::string &anId)
{
      return DirReport() + "Tmp-" + anId + StringDirSeparator();
}


void  cMMVII_Appli::InitReport(const std::string &anId,const std::string &aPost,bool IsMul,const std::vector<std::string> & aHeader)
{
    if (IsMul && (LevelCall()==0))
    {
       mReport2Merge.insert(anId);
    }

    if (LevelCall()==0)
    {
         CreateDirectories(DirReport(),true);
	 mMapIdFilesReport[anId] = DirReport()+anId + "." + aPost;
	 if (IsMul)
            CreateDirectories(DirSubPReport(anId),true);
    }
    else if (LevelCall()==1)
    {
        mMapIdFilesReport[anId]  = DirSubPReport(anId) + FileOfPath(UniqueStr(0),false)  +"." + aPost;
    }
    else
        return;
    mMapIdPostReport[anId] = aPost;

    cMMVII_Ofs(mMapIdFilesReport[anId], eFileModeOut::CreateText);

    if (! aHeader.empty())
       AddHeaderReportCSV(anId,aHeader);
}

const std::string& cMMVII_Appli::NameFileCSVReport(const std::string & anId) const
{
     auto anIt = mMapIdFilesReport.find(anId);

     MMVII_INTERNAL_ASSERT_tiny(anIt!=mMapIdPostReport.end(),"NameFileCSVReport for Id=" + anId);

     return anIt->second;
}

/*
void  cMMVII_Appli::AddTopReport(const std::string &anId,const std::string & aMsg)
{
    if (LevelCall()!=0)
       return;
    
    AddOneReport(anId,aMsg);
}
*/

void  cMMVII_Appli::AddOneReport(const std::string &anId,const std::string & aMsg)
{
    std::string  aName = mMapIdFilesReport[anId];
    MMVII_INTERNAL_ASSERT_tiny(aName!="","No file in AddOneMesCSV");
    cMMVII_Ofs aFile(aName, eFileModeOut::AppendText);

    aFile.Ofs() << aMsg;
}

void  cMMVII_Appli::AddHeaderReportCSV(const std::string &anId,const std::vector<std::string> & aVecMsg)
{
    // Add header line : do handle single or multiple process , do it only if at top-level
    if (mLevelCall==0)
       AddOneReportCSV(anId,aVecMsg);
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


void cMMVII_Appli::AddStdHeaderStatCSV(const std::string &anId,const std::string & aNameCol1,const std::vector<int> aVPerc,const std::vector<std::string>  & Additional)
{
    std::vector<std::string> aVStd {aNameCol1,"NbMes","Avg","StdDev","Avg2"};
    for (const auto & aPerc : aVPerc)
        aVStd.push_back("P"+ToStr(aPerc));

   AddOneReportCSV(anId,Append(aVStd,Additional));
}

void  cMMVII_Appli:: AddStdStatCSV(const std::string &anId,const std::string & aCol1,const cStdStatRes & aStat,const std::vector<int> aVPerc,const std::vector<std::string>  & Additional)
{
    if (aStat.NbMeasures()==0)
    {
       std::vector<std::string> aVStd {aCol1,"0","XXX","XXX"};
       for (size_t aK=0 ; aK< aVPerc.size() ; aK++)
           aVStd.push_back("XXX");
       AddOneReportCSV(anId,Append(aVStd,Additional));
       return;
    }
    std::vector<std::string> aVStd 
                             {
				     aCol1,
                                     ToStr(aStat.NbMeasures()),
				     ToStr(aStat.Avg()),
				     ((aStat.NbMeasures()>1) ? ToStr(aStat.DevStd()) : "XXX"),
				     ToStr(aStat.QuadAvg())
			     };
    for (const auto & aPerc : aVPerc)
        aVStd.push_back(ToStr(aStat.ErrAtProp(aPerc/100.0)));
   AddOneReportCSV(anId,Append(aVStd,Additional));
}



void  cMMVII_Appli::DoMergeReport()
{
     for (const auto & anIt : mMapIdFilesReport)
     {
        if (BoolFind(mReport2Merge,anIt.first))
	{
	     int aNbLines = 0;
             // Put aFileGlob in {} to create destruction before OnCloseReport that may generat error
             {
                 cMMVII_Ofs aFileGlob(anIt.second, eFileModeOut::AppendText);
                 const std::string & anId = anIt.first;

	         if (mRMSWasUsed)
	         {
	            for (const auto & aNameIm : VectMainSet(0))
	            {
                        std::string aNameIn = DirSubPReport(anId) + FileOfPath(aNameIm,false) + "." + mMapIdPostReport[anId];
	                cMMVII_Ifs aIn(aNameIn, eFileModeIn::Text);

	                std::string aLine;
	                while (std::getline(aIn.Ifs(), aLine))
	                {
	                     aFileGlob.Ofs() << aLine<< "\n";
			     aNbLines++;
	                }
	             }
	         }
                 RemoveRecurs(DirSubPReport(anId),false,false);

             }
	     OnCloseReport(aNbLines,anIt.first,anIt.second);
	}
        if (MapBoolFind(mMapIdRedirect,anIt.first) && (LevelCall()==0))
        {
            std::string aNewFile = mMapIdRedirect[anIt.first] + FileOfPath(anIt.second);
            RenameFiles(anIt.second,aNewFile);
        }
     }
}

// By default nothing to do
void  cMMVII_Appli::OnCloseReport(int aNbLine,const std::string & anIdent,const std::string & aNameFile) const
{
}


};




