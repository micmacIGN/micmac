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
    mMapAttrReport[anId].mDirRedirect  = aNewDir;
}


std::string  cMMVII_Appli::DirSubPReport(const std::string &anId)
{
      return DirReport() + "Tmp-" + anId + StringDirSeparator();
}

std::string  cMMVII_Appli::NameParamPostFixReport() { return "ReportPostF"; }
std::string  cMMVII_Appli::CommentParamPostFixReport() { return "Postfix for folder in report generation"; }


void  cMMVII_Appli::InitReportCSV(const std::string &anId,const std::string &aPost,bool IsMul,const std::vector<std::string> & aHeader)
{
    cAttrReport & anAttr = mMapAttrReport[anId];
    anAttr.mIsMul = IsMul;
    anAttr.m2Merge = IsMul && (LevelCall()==0);
    anAttr.mPost = aPost;

    if ((LevelCall()==0) || (!IsMul))
    {
         CreateDirectories(DirReport(),true);
         anAttr.mFile = DirReport()+anId + "." + aPost;
	 if (IsMul)
            CreateDirectories(DirSubPReport(anId),true);
    }
    else if (LevelCall()==1)
    {
        anAttr.mFile = DirSubPReport(anId) + FileOfPath(UniqueStr(0),false)  +"." + aPost;
    }
    else
        return;

    cMMVII_Ofs(anAttr.mFile, eFileModeOut::CreateText);

    if (! aHeader.empty())
       AddHeaderReportCSV(anId,aHeader);
}

const std::string& cMMVII_Appli::NameFileCSVReport(const std::string & anId) const
{
     auto anIt = mMapAttrReport.find(anId);

     MMVII_INTERNAL_ASSERT_tiny(anIt!=mMapAttrReport.end(),"NameFileCSVReport for Id=" + anId);

     return anIt->second.mFile;
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
    std::string  aName = NameFileCSVReport(anId);
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

std::vector<std::string> VInt2VStrPerc(const std::vector<int> &aVPerc)
{
    std::vector<std::string> aRes;
    for (const auto & aPerc : aVPerc)
        aRes.push_back("P"+ToStr(aPerc));
    return aRes;
}

std::vector<std::string> VInt2VStrPerc(const std::vector<int> & aVPerc,const cStdStatRes aStat)
{
    std::vector<std::string> aRes;
    for (const auto & aPerc : aVPerc)
        aRes.push_back(ToStr(aStat.ErrAtProp(aPerc/100.0)));
    return aRes;
}

void cMMVII_Appli::AddStdHeaderStatCSV
     (
         const std::string &anId,
         const std::string & aNameCol1,
         const std::vector<int> aVPerc,
         const std::vector<std::string>  & Additional
     )
{
    const std::vector<std::string> aVStd {aNameCol1,"NbMes","Avg","UbStdDev","Avg2","Min","Max"};
    AddOneReportCSV(anId,Append(aVStd,VInt2VStrPerc(aVPerc),Additional));
}

void  cMMVII_Appli::AddStdStatCSV
      (
           const std::string &anId,
           const std::string & aCol1,
           const cStdStatRes & aStat,
           const std::vector<int> aVPerc,
           const std::vector<std::string>  & Additional
      )
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
				     ((aStat.NbMeasures()>1) ? ToStr(aStat.UBDevStd(-1)) : "XXX"),
				     ToStr(aStat.QuadAvg()),
                                     ToStr(aStat.Min()),
                                     ToStr(aStat.Max())
			     };
   AddOneReportCSV(anId,Append(aVStd,VInt2VStrPerc(aVPerc,aStat),Additional));
}


void  cMMVII_Appli::DoMergeReport()
{
     for (const auto & [anId,anAttr] : mMapAttrReport)
     {
        if (anAttr.m2Merge)
	{
	     int aNbLines = 0;
             // Put aFileGlob in {} to create destruction before OnCloseReport that may generat error
             {
                 cMMVII_Ofs aFileGlob(anAttr.mFile, eFileModeOut::AppendText);

	         if (mRMSWasUsed)
	         {
	            for (const auto & aNameIm : VectMainSet(0))
	            {
                        std::string aNameIn = DirSubPReport(anId) + FileOfPath(aNameIm,false) + "." + anAttr.mPost;
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
	     OnCloseReport(aNbLines,anId,anAttr.mFile);
	}
/*
if (MapBoolFind(mMapIdRedirect,anIt.first))
{
StdOut() <<  "WWftR78::: " <<   anIt.second << " => " <<   mMapIdRedirect[anIt.first] + FileOfPath(anIt.second) << "\n";
}
*/
        if ( (anAttr.mDirRedirect!="")  && ( (LevelCall()==0) || (! anAttr.mIsMul)))
        {
            std::string aNewFile = anAttr.mDirRedirect + FileOfPath(anAttr.mFile);
            RenameFiles(anAttr.mFile,aNewFile);
        }
     }
}


// By default nothing to do
void  cMMVII_Appli::OnCloseReport(int aNbLine,const std::string & anIdent,const std::string & aNameFile) const
{
}


};




