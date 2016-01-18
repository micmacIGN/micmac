/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include <vector>
//Cmt Luc : unistd.h does not exist in Windows VS. Not including it still compiles, io.h is, I've heard, the W equivalent (to be confirmed)
#if (ELISE_unix || ELISE_Cygwin || ELISE_MacOs)
#include <unistd.h>
#endif
//#if (ELISE_windows)
//#include <io.h>
//#endif

class cAppli_TNR_Main  : public  cElErrorHandlor
{
    public :
	cAppli_TNR_Main(int argc,char ** argv);
    private :

	//Declaration des fonctions
	void OnError();
	void DoOneGlobTNR(const std::string &,const std::string & mOutXML,const std::string & mOutErrorsXML);
	vector<cXmlTNR_OneTestReport> TestOneCom(const cXmlTNR_OneTest & aOneT);
	cXmlTNR_TestFileReport TestExistFile(const cXmlTNR_TestExistFile & aFile);
	cXmlTNR_TestDirReport TestExistDir(const cXmlTNR_TestExistDir & aDirPath);
	cXmlTNR_CalibReport TestCalib(const cXmlTNR_TestDiffCalib & aCalib);
	cXmlTNR_OriReport TestOri(const cXmlTNR_TestDiffOri & aOri);
	cXmlTNR_ImgReport TestImg(const cXmlTNR_TestDiffImg & aImg);
	vector<string> DirArbo(const string & DirPath) const;
	int DirSize(const std::string & DirPath);
	void  Error(const std::string &);

	void createInitialDirectory() const;

	//Declaration des variables
	cElemAppliSetFile mEASF;
	std::string mPattern;
	std::string mCurDirRef;
	std::string mCurDirExe;
	std::list<std::string>mLFileExe;
	std::list<std::string>mLFileRef;
	std::string mFullName;
	std::string mOutReportXML;
	std::string mOutErrorsXML;
	std::string mInRefDir;
	std::string mInExeDir;
	bool mPurge;
	cInterfChantierNameManipulateur*mICNMExe;
	cInterfChantierNameManipulateur*mICNMRef;
	cInterfChantierNameManipulateur*DirFile;
    cXmlTNR_GlobTest mXML_CurGT;
	std::string mCurCom;
};

//Return message when command fails
void cAppli_TNR_Main::OnError()
{
    std::cout << "ERROR WHEN EXE " << mCurCom << "\n";
    exit(EXIT_SUCCESS);
} 

//Error fonction
void cAppli_TNR_Main::Error(const std::string & aMes)
{
    std::cout << aMes << "\n";
}

class cTestElParseDir : public ElActionParseDir
{
    public :
        void act(const ElResParseDir & aRPD)
	{
	    FileList.push_back(aRPD.name());
	};
	vector<std::string> FileList;
};

//Recursive function to get complete arborescence of one directory
vector<string> cAppli_TNR_Main::DirArbo(const string & DirPath) const
{
    cTestElParseDir aTPD;
    ElParseDir(DirPath.c_str(),aTPD,1000);
    return aTPD.FileList;
}

//Get directory size
int cAppli_TNR_Main::DirSize(const std::string & DirPath)
{
    std::vector<std::string> FileList = DirArbo(DirPath);
    int Sz = 0;
    for(unsigned int i=0;i<FileList.size();i++)
    {
		if(ELISE_fp::IsDirectory(FileList[i])==0)
		{
			Sz += ELISE_fp::file_length(FileList[i]);
		}
    }
    return Sz;
}

//Function to test if a file exist and compare it between Ref and Exe (size)
cXmlTNR_TestFileReport cAppli_TNR_Main::TestExistFile(const cXmlTNR_TestExistFile & aFile)
{
    //Global test report in Xml
    cXmlTNR_TestFileReport aFileReport;
    aFileReport.FileName() = aFile.NameFile(); //Get FileName
    
    std::cout<<"Check for file in Ref and Exe directory "<<aFile.NameFile()<<endl;
    int IsExeFile = ELISE_fp::exist_file(mCurDirExe+aFile.NameFile());//See if the file exist in Exe
    int IsRefFile = ELISE_fp::exist_file(mCurDirRef+aFile.NameFile());//See if the file exist in Ref
    
    //File exist in the Exe directory
    if(IsExeFile==1){aFileReport.TestExeFile() = true;}
    else{aFileReport.TestExeFile() = false;}
    
    //File exist in the Ref directory
    if(IsRefFile==1){aFileReport.TestRefFile() = true;}
    else{aFileReport.TestRefFile() = false;}
    
    int ExeFileSize = ELISE_fp::file_length(mCurDirExe+aFile.NameFile());
	int RefFileSize = ELISE_fp::file_length(mCurDirRef+aFile.NameFile());
    
    if((IsExeFile==1)&&(IsRefFile==1)&&(ExeFileSize==RefFileSize))
    {
		//Else display the size of both
		aFileReport.TestFileDiff() = true;
		std::cout<<"Files are the SAME : file size "<<ELISE_fp::file_length(mCurDirExe+aFile.NameFile())<<"o"<<endl;
    }
    if(ExeFileSize!=RefFileSize)
    {
		//If file are different display size of Reference and New
		std::cout<<"WARNING : Files are DIFFERENT : Exe file size "<<ELISE_fp::file_length(mCurDirExe+aFile.NameFile())<<"o Ref file size "<<ELISE_fp::file_length(mCurDirRef+aFile.NameFile())<<"o"<<endl;
		aFileReport.TestFileDiff() = false;
	}
	if((IsExeFile==0)&&(IsRefFile==0))
	{
		//If file doesn't exist display message
		aFileReport.TestFileDiff() = false;//False to FileTest
		std::cout<<"ERROR : file "<<mCurDirExe+aFile.NameFile()<<" DOESN'T EXIST"<<endl;
    }
	aFileReport.ExeFileSize() = ExeFileSize; //Stock Exe file size
	aFileReport.RefFileSize() = RefFileSize; //Stock Ref file size
    return aFileReport;
}

//Function to test if a directory exist and compare it between Ref and Exe (Files and size)
cXmlTNR_TestDirReport cAppli_TNR_Main::TestExistDir(const cXmlTNR_TestExistDir & aDir)
{
    std::vector<std::string> FileListRef = DirArbo(mCurDirRef+aDir.NameDir());
    std::vector<std::string> FileListExe = DirArbo(mCurDirExe+aDir.NameDir());
    
    //Global test report in Xml
    cXmlTNR_TestDirReport aDirReport;
    aDirReport.DirName() = aDir.NameDir();//Get Dir name

    int DirExeSz=0;
    int DirRefSz=0;
    int DirSz=0;
    
    int diff=0;
    for(unsigned int i=0;i<FileListRef.size();i++)
    {
		std::string tmp = FileListRef[i];
		std::string tmpstr = tmp.substr(mCurDirRef.size(),tmp.size());
		std::size_t found1 = tmpstr.find("Shade");
		std::size_t found2 = tmpstr.find("MakefileParallelisation");
		std::size_t found3 = tmpstr.find(".ply");
		std::size_t found4 = tmpstr.find("8Bits");
	
		int IsDir = ELISE_fp::IsDirectory(mCurDirExe+tmpstr);
		if(IsDir!=1&&found1>1000&&found2>1000&&found3>1000&&found4>1000)
		{
			std::cout<<"Check for file "<<tmpstr<<endl;
			DirRefSz += ELISE_fp::file_length(mCurDirRef+tmpstr);
			DirExeSz += ELISE_fp::file_length(mCurDirExe+tmpstr);
			if(ELISE_fp::exist_file(mCurDirExe+tmpstr)==1)
			{
				int CmpFile = ELISE_fp::CmpFiles(mCurDirExe+tmpstr,mCurDirRef+tmpstr);
				if((CmpFile==1)|(CmpFile==-1))
				{
					//If file are different display size of Reference and New
					std::cout<<"WARNING : Files are DIFFERENT : Exe file size "<<ELISE_fp::file_length(mCurDirExe+tmpstr)<<"o Ref file size "<<ELISE_fp::file_length(mCurDirRef+tmpstr)<<"o"<<"\n\n";
					cFileDiff tmpFile;
					tmpFile.Name() = tmpstr;
					tmpFile.DiffSize() = ELISE_fp::file_length(mCurDirRef+tmpstr) - ELISE_fp::file_length(mCurDirExe+tmpstr);
					aDirReport.FileDiff().push_back(tmpFile);
					//~ if(DirRefSz!=DirExeSz)
					//~ {
						//~ diff += 1;
					//~ }
				}
				if(CmpFile==0)
				{
					//Else display the size of both
					std::cout<<"Files are the SAME : Exe file size "<<ELISE_fp::file_length(mCurDirExe+tmpstr)<<"o Ref file size "<<ELISE_fp::file_length(mCurDirRef+tmpstr)<<"o"<<"\n\n";
					DirSz += ELISE_fp::file_length(mCurDirExe+tmpstr);
				}
			}
			if(ELISE_fp::exist_file(mCurDirExe+tmpstr)!=1&&tmpstr!=aDir.NameDir())
			{
				std::cout<<"WARNING : Ref file "<<tmpstr<<" NOT FOUND in Exe directory"<<endl;
				aDirReport.MissingExeFile().push_back(tmpstr);
				diff += 1;
			}
		}
    }
    
    for(unsigned int i=0;i<FileListExe.size();i++)
    {
		std::string tmp = FileListExe[i];
		std::string tmpstr = tmp.substr(mCurDirExe.size(),tmp.size());
		//DirExeSz += ELISE_fp::file_length(mCurDirExe+tmpstr);
		std::size_t found1 = tmpstr.find("Shade");
		std::size_t found2 = tmpstr.find("MakefileParallelisation");
		std::size_t found3 = tmpstr.find(".ply");
		std::size_t found4 = tmpstr.find("8Bits");
		if(found1>1000&&found2>1000&&found3>1000&&found4>1000&&ELISE_fp::exist_file(mCurDirRef+tmpstr)!=1&&tmpstr!=aDir.NameDir()&&ELISE_fp::IsDirectory(mCurDirRef+tmpstr)!=1)
		{
			std::cout<<"WARNING : Exe file "<<tmpstr<<" NOT FOUND in Ref directory"<<endl;
			aDirReport.MissingRefFile().push_back(tmpstr);
			diff += 1;
		}
    }
    
    aDirReport.ExeDirSize() = DirExeSz;
    aDirReport.RefDirSize() = DirRefSz;
    if(diff==0)
    {
		aDirReport.TestDirDiff()=true;
    }
    else
    {
		aDirReport.TestDirDiff()=false;
    }
    
    if(ELISE_fp::IsDirectory(mCurDirExe+aDir.NameDir())!=1)
    {
		//Else display message
		aDirReport.TestExeDir() = false;
		std::cout<<"ERROR : Exe directory "<<mCurDirExe+aDir.NameDir()<<" DOESN'T EXIST"<<endl;
    }
    else
    {
		aDirReport.TestExeDir() = true;
    }
    
    if(ELISE_fp::IsDirectory(mCurDirRef+aDir.NameDir())!=1)
    {
		//Else display message
		aDirReport.TestRefDir() = false;
		std::cout<<"ERROR : Ref directory "<<mCurDirRef+aDir.NameDir()<<" DOESN'T EXIST"<<endl;
    }
    else
    {
		aDirReport.TestRefDir() = true;
    }
    return aDirReport;
}

cXmlTNR_CalibReport cAppli_TNR_Main::TestCalib(const cXmlTNR_TestDiffCalib & aCalib)
{	
    //mm3d CmpCalib to compare two AutoCal files (in Ori directories)
    std::string CmdCalib = "mm3d CmpCalib " + (mCurDirExe + aCalib.NameTestCalib()) + " " + (mCurDirRef + aCalib.NameTestCalib())+" DispW=false Out=ecarts_calib.txt XmlG=Calib.xml";
    std::cout << CmdCalib << "\n";
    System(CmdCalib,true);//execute system cmd
    cXmlTNR_TestCalibReport mXML_TestCalib = StdGetFromSI("Calib.xml",XmlTNR_TestCalibReport);//Get the file and parse it
    
    cXmlTNR_CalibReport mCalib;
    mCalib.CalibName () =  mXML_TestCalib.CalibName();
    mCalib.TestCalibDiff() = mXML_TestCalib.TestCalibDiff();
    if(!mCalib.TestCalibDiff())
    {
		for (std::list<Pt2dr>::const_iterator i= mXML_TestCalib.EcartsRadiaux().begin() ; i!= mXML_TestCalib.EcartsRadiaux().end() ;  i++)
		{
			mCalib.EcartsRadiaux().push_back(*i);
		}
	
		for (std::list<cEcartsPlani>::iterator i= mXML_TestCalib.EcartsPlani().begin() ; i!= mXML_TestCalib.EcartsPlani().end() ;  i++)
		{
			crEcartsPlani mEP;
			mEP.CoordPx()=(*i).CoordPx();
			mEP.UxUyE()=(*i).UxUyE();
			mCalib.rEcartsPlani().push_back(mEP);
		}
    }
	
    //mCalib.EcartsRadiaux() = mXML_TestCalib.EcartsRadiaux();
    //mCalib.DistMatrix() = mXML_TestCalib.DistMatrix();
    return mCalib;//return the same structure for Global Test Report
}

cXmlTNR_OriReport cAppli_TNR_Main::TestOri(const cXmlTNR_TestDiffOri & aOri)
{
    //mm3d CmpOri to compare two Orientation
    std::string CmdOri = "mm3d CmpOri " +(mCurDirExe+aOri.PatternTestOri())+" "+aOri.NameTestOri()+" "+aOri.NameTestOri()+" DirOri2="+mCurDirRef +" XmlG=Ori.xml";
    std::cout << CmdOri << "\n";
    System(CmdOri,true);//execute system cmd
    cXmlTNR_TestOriReport mXML_TestOri = StdGetFromSI("Ori.xml",XmlTNR_TestOriReport);//Get the file and parse it
    
    cXmlTNR_OriReport mOri;
    mOri.OriName() =  mXML_TestOri.OriName();
    mOri.TestOriDiff() = mXML_TestOri.TestOriDiff();
    if(!mOri.TestOriDiff())
    {
		mOri.DistCenter() = mXML_TestOri.DistCenter();
		mOri.DistMatrix() = mXML_TestOri.DistMatrix();
    }
    return mOri;//return the same structure for Global Test Report
}

cXmlTNR_ImgReport cAppli_TNR_Main::TestImg(const cXmlTNR_TestDiffImg & aImg)
{
    //mm3d CmpIm to compare images px per px
    std::string CmdImg = "mm3d CmpIm " +(mCurDirExe + aImg.NameTestImg()) + " " + (mCurDirRef + aImg.NameTestImg())+" XmlG=CmpImg.xml";
    std::cout << CmdImg << "\n";
    System(CmdImg,true);//execute system cmd
    cXmlTNR_TestImgReport mXML_TestImg = StdGetFromSI("CmpImg.xml",XmlTNR_TestImgReport);//Get the file and parse it
    
    cXmlTNR_ImgReport mImg;
    mImg.ImgName() =  mXML_TestImg.ImgName();
    mImg.TestImgDiff() = mXML_TestImg.TestImgDiff();
    mImg.NbPxDiff() = mXML_TestImg.NbPxDiff();
    mImg.SumDiff() = mXML_TestImg.SumDiff();
    mImg.MoyDiff() = mXML_TestImg.MoyDiff();
    mImg.DiffMaxi() = mXML_TestImg.DiffMaxi();
    return mImg;
}

vector<cXmlTNR_OneTestReport> cAppli_TNR_Main::TestOneCom(const cXmlTNR_OneTest & aOneT)
{      
    
    cXmlTNR_OneTestReport aOTR;//initialize One Test Report
    cXmlTNR_OneTestReport aFTR;//initialize One Test Report
    
    mCurCom = aOneT.Cmd();
    cXmlTNR_TestCmdReport aCmd;
    aCmd.CmdName() = aOneT.Cmd();
    if(mCurCom!="")
    {
		std::cout<<"**************************************************"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**                TESTING COMMAND               **"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**************************************************"<<endl;
		
		ctPath path = getWorkingDirectory();
		setWorkingDirectory(mCurDirExe);	
		int aVal = System(mCurCom,true);//execute commande of the test
		setWorkingDirectory(path);
		
		//Check if you get de cmd execution status
		if ((aVal != EXIT_SUCCESS)  && (aOneT.TestReturnValue().Val()))
		{
			aFTR.XmlTNR_TestCmdReport().push_back(aCmd);
			Error("Cannot exe "+ mCurCom);
			aCmd.TestCmd() = false;
		}
		else{aCmd.TestCmd() = true;}
		aOTR.XmlTNR_TestCmdReport().push_back(aCmd);
    }
    
    for (std::list<cXmlTNR_FileCopy>::const_iterator it7T= aOneT.FileCopy().begin() ; it7T!= aOneT.FileCopy().end() ;  it7T++)
	{
		std::cout<<"**                COPY FILE                     **"<<endl;
		ELISE_fp::CpFile((mCurDirRef + (*it7T).FilePath()),(mCurDirExe + (*it7T).FilePath()));        
    }
    
    for (std::list<cXmlTNR_DirCopy>::const_iterator it8T= aOneT.DirCopy().begin() ; it8T!= aOneT.DirCopy().end() ;  it8T++)
	{
		std::cout<<"**                COPY DIR                      **"<<endl;
		std::vector<std::string> FileList = DirArbo(mCurDirRef + (*it8T).DirPath());
		ELISE_fp::MkDir(mCurDirExe + (*it8T).DirPath());
		for(unsigned int i=0;i<FileList.size();i++)
		{
			if(ELISE_fp::IsDirectory(mCurDirRef + (*it8T).DirPath() + FileList[i])!=1)
			{
				ELISE_fp::CpFile((mCurDirRef + (*it8T).DirPath() + FileList[i]),mCurDirExe + (*it8T).DirPath());
			}
		}         
    }
    
    //For each file to check
    int ct = 0;
    int c = 0;
    for (std::list<cXmlTNR_TestExistFile>::const_iterator it2T= aOneT.TestFiles().begin() ; it2T!= aOneT.TestFiles().end() ;  it2T++)
    {
		c += 1;
		std::cout<<"**************************************************"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**                TESTING FILE                  **"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**************************************************"<<endl;
		cXmlTNR_TestFileReport aFile = TestExistFile(*it2T);
		if(aFile.TestFileDiff())
		{
			ct+=1;
		}
		else
		{
			aFTR.XmlTNR_TestFileReport().push_back(aFile);  
		}
		aOTR.XmlTNR_TestFileReport().push_back(aFile);           
    }
    
    
    
    //For each directory to check
    for (std::list<cXmlTNR_TestExistDir>::const_iterator it3T= aOneT.TestDir().begin() ; it3T!= aOneT.TestDir().end() ;  it3T++)
    {
		c += 1;
		std::cout<<"**************************************************"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**                TESTING DIR                   **"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**************************************************"<<endl;
		cXmlTNR_TestDirReport aDir = TestExistDir(*it3T);
		if(aDir.TestDirDiff())
		{
			ct+=1;
		}
		else
		{
			aFTR.XmlTNR_TestDirReport().push_back(aDir);  
		}
		aOTR.XmlTNR_TestDirReport().push_back(aDir);        
	}
		
		//For each calib to check
	for (std::list<cXmlTNR_TestDiffCalib>::const_iterator it4T= aOneT.TestCalib().begin() ; it4T!= aOneT.TestCalib().end() ;  it4T++)
	{
		c += 1;
		std::cout<<"**************************************************"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**                TESTING CALIB                 **"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**************************************************"<<endl;          
		cXmlTNR_CalibReport aCalib = TestCalib(*it4T);
		if(aCalib.TestCalibDiff())
		{
			ct+=1;
		}
		else
		{
			aFTR.XmlTNR_CalibReport().push_back(aCalib);  
		}
		aOTR.XmlTNR_CalibReport().push_back(aCalib);          
	}
    
    //For each Ori to check
    for (std::list<cXmlTNR_TestDiffOri>::const_iterator it5T= aOneT.TestOri().begin() ; it5T!= aOneT.TestOri().end() ;  it5T++)
    {
		c += 1;
		std::cout<<"**************************************************"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**                TESTING ORIENTATION           **"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**************************************************"<<endl;          
		cXmlTNR_OriReport aOri = TestOri(*it5T);
		if(aOri.TestOriDiff())
		{
			ct+=1;
		}
		else
		{
			aFTR.XmlTNR_OriReport().push_back(aOri);  
		}
		aOTR.XmlTNR_OriReport().push_back(aOri);          
	}
		
	//For each Img to check
	for (std::list<cXmlTNR_TestDiffImg>::const_iterator it6T= aOneT.TestImg().begin() ; it6T!= aOneT.TestImg().end() ;  it6T++)
	{
		c += 1;
		std::cout<<"**************************************************"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**                TESTING IMAGE                 **"<<endl;
		std::cout<<"**                                              **"<<endl;
		std::cout<<"**************************************************"<<endl;
		cXmlTNR_ImgReport aImg = TestImg(*it6T);
		if(aImg.TestImgDiff())
		{
			ct+=1;
		}
		else
		{
			aFTR.XmlTNR_ImgReport().push_back(aImg); 
		}
		aOTR.XmlTNR_ImgReport().push_back(aImg);           
    }
    
    if(ct==c&&aCmd.TestCmd())
    {
		aOTR.TestOK() = true;
    }
    else{aOTR.TestOK() = false;}
    vector<cXmlTNR_OneTestReport> Test;
    Test.push_back(aOTR);
    Test.push_back(aFTR);
    return Test;
}

void cAppli_TNR_Main::createInitialDirectory() const
{
	cout << "--- creating initial Exe directory [" << mCurDirExe << ']' << endl;
	ELISE_fp::MkDir(mCurDirExe);
	if ( !cElFilename(mCurDirExe).copyRights(cElFilename(mCurDirExe)))
	{
		ELISE_DEBUG_ERROR(true, "cAppli_TNR_Main::createInitialDirectory", "failed to copy rights from [" << mCurDirRef << "] to [" << mCurDirExe << "]");
		return;
	}

	for (list<string>::const_iterator itN = mXML_CurGT.PatFileInit().begin(); itN != mXML_CurGT.PatFileInit().end(); itN++)
	{
		cout << '\t' << "copy files of pattern [" << mCurDirRef + *itN << ']' << endl;

		cElPathRegex pathRegex(mCurDirRef + *itN);
		pathRegex.copy(mCurDirExe);

		//~ ELISE_fp::CpFile((mCurDirRef + *itN), mCurDirExe);
	}

	for (list<string>::const_iterator itN = mXML_CurGT.DirInit().begin(); itN != mXML_CurGT.DirInit().end(); itN++)
	{
		cout << '\t' << "copy directory [" << mCurDirRef + *itN << ']' << endl;

		ctPath(mCurDirRef + *itN).copy(ctPath(mCurDirExe + *itN));

		//~ vector<string> FileList = DirArbo(mCurDirRef + *itN);
		//~ ELISE_fp::MkDir(mCurDirExe + *itN);
		//~ for(size_t i = 0; i < FileList.size(); i++)
			//~ if(ELISE_fp::IsDirectory(mCurDirRef + *itN + FileList[i])) ELISE_fp::CpFile((mCurDirRef + *itN + FileList[i]), mCurDirExe + *itN);
	}
}

void cAppli_TNR_Main::DoOneGlobTNR(const std::string & aNameFile,const std::string & mOutXML, const std::string & mOutErrorsXML)
{
    cXmlTNR_GlobTestReport aGTR;//initialize Global Test Report
    cXmlTNR_GlobTestReport aGFT;//initialize Global Test Report
    mXML_CurGT = StdGetFromSI(mEASF.mDir+aNameFile,XmlTNR_GlobTest);
    if(mInRefDir!=""){mCurDirRef = mEASF.mDir+mInRefDir+"/";}
    else{
		mCurDirRef = mEASF.mDir+  "TNR-Ref-" + mXML_CurGT.Name() + "/";
    }
    if(mInExeDir!=""){mCurDirExe = mEASF.mDir+mInExeDir+"/";}
    else
    {
		mCurDirExe = mEASF.mDir+  "TNR-Exe-" + mXML_CurGT.Name() + "/";
    }
    
	// On fait une directory TNR vide
	if (ELISE_fp::IsDirectory(mCurDirExe) && mPurge) ELISE_fp::PurgeDirRecursif(mCurDirExe);
	createInitialDirectory();

    int c = 0;
    int ct = 0;
    for (std::list<cXmlTNR_OneTest>::const_iterator it1T= mXML_CurGT.Tests().begin() ; it1T!= mXML_CurGT.Tests().end() ;  it1T++)
    {
		c += 1;   
		vector<cXmlTNR_OneTestReport> aOT = TestOneCom(*it1T);
		if(aOT[0].TestOK())
		{
			ct += 1;
		}
		else
		{
			aOT[1].TestOK()=false;
			aGFT.XmlTNR_OneTestReport().push_back(aOT[1]);
		}	
		aGTR.XmlTNR_OneTestReport().push_back(aOT[0]);
	
    }
    aGTR.NbTest() = c;
    aGTR.NbTestOk() = ct;
    
    aGFT.NbTest() = c;
    aGFT.NbTestOk() = ct;
    
    if(c==ct)
    {
		aGTR.Bilan()=true;
		aGFT.Bilan()=true;
    }
    else
    {
		aGTR.Bilan()=false;
		aGFT.Bilan()=false;
    }
    
    aGTR.Name() = mXML_CurGT.Name();
    aGFT.Name() = mXML_CurGT.Name();
    if(mOutReportXML!=""){MakeFileXML(aGTR, mOutReportXML);}
    else{MakeFileXML(aGTR, "GlobTest.xml");}
    if(mOutErrorsXML!=""){MakeFileXML(aGFT, mOutErrorsXML);}
    else{MakeFileXML(aGFT, "ErrorsReport.xml");}		      
}

cAppli_TNR_Main::cAppli_TNR_Main(int argc,char ** argv)  
{
    TheCurElErrorHandlor = this;
    bool aShowArgs = false;
    mOutReportXML="";
    mOutErrorsXML="";
    mInRefDir="";
    mInExeDir="";
    mPurge = false;
    ElInitArgMain
    (
		argc,argv,
		LArgMain()  << EAMC(mPattern, "Xml file with test specification",  eSAM_IsPatFile),
		LArgMain()  << EAM(mOutReportXML,"OutReportXML", true,"XML report file (Def=GlobTest.xml)")
					<< EAM(mOutErrorsXML,"OutErrorsXML", true,"XML errors file (Def=ErrorsReport.xml)")
					<< EAM(mInRefDir,"InRefDir", true,"Input Reference Directory (Def=TNR-Ref-Name)")
					<< EAM(mPurge,"Purge", true,"Purge Exe directory (Def=false)")
					//<< EAM(mInExeDir,"InExeDir", true,"Input Execution Directory")
    );
    
    mEASF.Init(mPattern);
    const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();
    SplitDirAndFile(mCurDirRef, mPattern, mFullName);
	
    for (int aK=0 ; aK<int(aSetIm->size()) ; aK++)
    {
		DoOneGlobTNR((*aSetIm)[aK],mOutReportXML,mOutErrorsXML);
    }
    
    if (aShowArgs) ShowArgs();
}


//----------------------------------------------------------------------------

int TNR_main(int argc,char ** argv)
{
    cAppli_TNR_Main anAppli(argc,argv);
    //ELISE_fp::PurgeDirRecursif("Tmp-MM-Dir");
    //~ ELISE_fp::RmFileIfExist("Calib.xml");
    //~ ELISE_fp::RmFileIfExist("CmpImg.xml");
    //~ ELISE_fp::RmFileIfExist("ecarts_calib.txt");
    //~ ELISE_fp::RmFileIfExist("Ori.xml");
    //~ ELISE_fp::RmFileIfExist("SauvApero.xml");
    return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

   Ce logiciel est un programme informatique servant à la mise en
   correspondances d'images pour la reconstruction du relief.

   Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
   respectant les principes de diffusion des logiciels libres. Vous pouvez
   utiliser, modifier et/ou redistribuer ce programme sous les conditions
   de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
   sur le site "http://www.cecill.info".

   En contrepartie de l'accessibilité au code source et des droits de copie,
   de modification et de redistribution accordés par cette licence, il n'est
   offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
   seule une responsabilité restreinte pèse sur l'auteur du programme,  le
   titulaire des droits patrimoniaux et les concédants successifs.

   A cet égard  l'attention de l'utilisateur est attirée sur les risques
   associés au chargement,  à l'utilisation,  à la modification et/ou au
   développement et à la reproduction du logiciel par l'utilisateur étant
   donné sa spécificité de logiciel libre, qui peut le rendre complexe à
   manipuler et qui le réserve donc à des développeurs et des professionnels
   avertis possédant  des  connaissances  informatiques approfondies.  Les
   utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
   logiciel à leurs besoins dans des conditions permettant d'assurer la
   sécurité de leurs systèmes et ou de leurs données et, plus généralement,
   à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

   Le fait que vous puissiez accéder à cet en-tête signifie que vous avez
   pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
   termes.
   Footer-MicMac-eLiSe-25/06/2007/*/
