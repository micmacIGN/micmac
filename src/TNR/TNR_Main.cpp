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


class cAppli_TNR_Main  : public  cElErrorHandlor
{
    public :
        cAppli_TNR_Main(int argc,char ** argv);
    private :

        void OnError();

        

        void DoOneGlobTNR(const std::string &,const std::string & mOutXML);
        cXmlTNR_OneTestReport TestOneCom(const cXmlTNR_OneTest & aOneT);
        cXmlTNR_TestFileReport TestExistFile(const cXmlTNR_TestExistFile & aFile);
        cXmlTNR_TestDirReport TestExistDir(const cXmlTNR_TestExistDir & aDirPath);
        cXmlTNR_CalibReport TestCalib(const cXmlTNR_TestDiffCalib & aCalib);
        cXmlTNR_OriReport TestOri(const cXmlTNR_TestDiffOri & aOri);
        cXmlTNR_ImgReport TestImg(const cXmlTNR_TestDiffImg & aImg);
        cElemAppliSetFile mEASF;
        std::string mPattern;
        std::string mCurDirRef;
        std::string mCurDirExe;
        std::list<std::string>mLFileExe;
        std::list<std::string>mLFileRef;
        std::string mFullName;
        std::string mOutXML;
        std::string mInDir;
		cInterfChantierNameManipulateur*mICNMExe;
		cInterfChantierNameManipulateur*mICNMRef;

        cXmlTNR_GlobTest mXML_CurGT;

        void  Error(const std::string &);
        std::string mCurCom;
};

void cAppli_TNR_Main::OnError()
{
    std::cout << "ERROR WHEN EXE " << mCurCom << "\n";
    exit(EXIT_SUCCESS);
} 

void cAppli_TNR_Main::Error(const std::string & aMes)
{
    std::cout << aMes << "\n";
}

cXmlTNR_TestFileReport cAppli_TNR_Main::TestExistFile(const cXmlTNR_TestExistFile & aFile)
{
	//Global test report in Xml
	cXmlTNR_TestFileReport aFileReport;
	aFileReport.FileName() = aFile.NameFile(); //Get FileName
	
	std::cout<<"Check for file in Ref and Exe directory "<<mCurDirExe+aFile.NameFile()<<endl;
	int IsExeFile = ELISE_fp::exist_file(mCurDirExe+aFile.NameFile());//See if the file exist in Exe
	int IsRefFile = ELISE_fp::exist_file(mCurDirRef+aFile.NameFile());//See if the file exist in Ref
	
	//File exist in the Exe directory
	if(IsExeFile==1){aFileReport.TestExeFile() = true;}
	else{aFileReport.TestExeFile() = false;}
	
	//File exist in the Ref directory
	if(IsRefFile==1){aFileReport.TestRefFile() = true;}
	else{aFileReport.TestRefFile() = false;}
	
	if(IsExeFile==1&&IsRefFile==1)
	{
		//If file exist, check for file size		
		std::cout<<"Exe file "<<mCurDirExe+aFile.NameFile()<<" EXIST"<<endl;
		int CmpFile = ELISE_fp::CmpFiles(mCurDirExe+aFile.NameFile(),mCurDirRef+aFile.NameFile()); //Binary comparison beetween Ref and Exe file
		if(CmpFile==1)
		{
			//If file are different display size of Reference and New
			std::cout<<"WARNING : Files are DIFFERENT : Exe file size "<<ELISE_fp::file_length(mCurDirExe+aFile.NameFile())<<"o Ref file size "<<ELISE_fp::file_length(mCurDirRef+aFile.NameFile())<<"o"<<endl;
			aFileReport.ExeFileSize() = ELISE_fp::file_length(mCurDirExe+aFile.NameFile());//Stock Exe file size
			aFileReport.RefFileSize() = ELISE_fp::file_length(mCurDirExe+aFile.NameFile());//Stock Ref file size
			aFileReport.TestFileDiff() = false;
		}
		else
		{
			//Else display the size of both
			aFileReport.TestFileDiff() = true;
			std::cout<<"Files are the SAME : file size "<<ELISE_fp::file_length(mCurDirExe+aFile.NameFile())<<"o"<<endl;
			aFileReport.ExeFileSize() = ELISE_fp::file_length(mCurDirExe+aFile.NameFile());//Stock Exe file size
			aFileReport.RefFileSize() = ELISE_fp::file_length(mCurDirRef+aFile.NameFile());//Stock Ref file size
		}
	}
	else
	{
		//If file doesn't exist display message
		aFileReport.TestFileDiff() = false;//False to FileTest
		std::cout<<"ERROR : file "<<mCurDirExe+aFile.NameFile()<<" DOESN'T EXIST"<<endl;
	}
	return aFileReport;
}

cXmlTNR_TestDirReport cAppli_TNR_Main::TestExistDir(const cXmlTNR_TestExistDir & aDir)
{
	//Global test report in Xml
	cXmlTNR_TestDirReport aDirReport;
	aDirReport.DirName() = aDir.NameDir();//Get Dir name
	
	//Check for dir in Ref and Exe
	std::cout<<"Check for Exe directory "<<mCurDirExe+aDir.NameDir()<<endl;
	int IsExeDir = ELISE_fp::IsDirectory(mCurDirExe+aDir.NameDir());
	int IsRefDir = ELISE_fp::IsDirectory(mCurDirRef+aDir.NameDir());
	
	if(IsExeDir==1){aDirReport.TestExeDir() = true;}
	else{aDirReport.TestExeDir() = false;}
	
	if(IsRefDir==1){aDirReport.TestRefDir() = true;}
	else{aDirReport.TestRefDir() = false;}
	
	if(IsExeDir==1&&IsRefDir==1)
	{
		std::cout<<"Exe directory "<<mCurDirExe+aDir.NameDir()<<" EXIST"<<endl;
		
		//Get files in reference directory
		mICNMRef = cInterfChantierNameManipulateur::BasicAlloc(mCurDirRef+aDir.NameDir());
		mLFileRef = mICNMRef->StdGetListOfFile(".*");
		
		//Get files in exe directory
		mICNMExe = cInterfChantierNameManipulateur::BasicAlloc(mCurDirExe+aDir.NameDir());
		mLFileExe = mICNMExe->StdGetListOfFile(".*");
		
		std::cout<<"Check for content of Exe directory "<<mCurDirRef+aDir.NameDir()<<"\n\n";
		
		vector<string> RefNotFound;
		vector<string> ExeNotFound;
		
		int DirExeSz=0;
		int DirRefSz=0;
		
		//Get file of Ref which doesn't exist in Exe
		for (std::list<std::string>::iterator j=mLFileRef.begin() ; j!=mLFileRef.end() ; j++)
		{
			if(ELISE_fp::exist_file(mCurDirExe+aDir.NameDir()+*j)==0)
			{
				RefNotFound.push_back (*j);
			}
			DirRefSz += ELISE_fp::file_length(mCurDirRef+aDir.NameDir()+*j);//Get total size of Ref directory
		}
		
		//Get file of Exe which doesn't exist in Ref
		for (std::list<std::string>::iterator i=mLFileExe.begin() ; i!=mLFileExe.end() ; i++)
		{				
			if(ELISE_fp::exist_file(mCurDirRef+aDir.NameDir()+*i)==0)
			{
				ExeNotFound.push_back (*i);
			}
			DirExeSz += ELISE_fp::file_length(mCurDirExe+aDir.NameDir()+*i);//Get total size of Exe directory
		}
		
		aDirReport.ExeDirSize() = DirExeSz;//Stock size in xml file
		aDirReport.RefDirSize() = DirRefSz;
		
		int DirSz=0;
		
		//Comparison of all files in Ref and Exe directory
		for (std::list<std::string>::iterator j=mLFileRef.begin() ; j!=mLFileRef.end() ; j++)
		{
			for (std::list<std::string>::iterator i=mLFileExe.begin() ; i!=mLFileExe.end() ; i++)
			{
				if(*i==*j)
				{
					//file exist, check for file size
					std::cout<<"Ref file "<<mCurDirRef+aDir.NameDir()+*j<<" FOUND in Exe directory "<<mCurDirExe+aDir.NameDir()<<endl;
					std::cout<<"Compare content between Ref file and Exe file"<<endl;
					int CmpFile = ELISE_fp::CmpFiles(mCurDirExe+aDir.NameDir()+*i,mCurDirRef+aDir.NameDir()+*j); //Comparison with the reference file
					if(CmpFile==1)
					{
						//If file are different display size of Reference and New
						std::cout<<"WARNING : Files are DIFFERENT : Exe file size "<<ELISE_fp::file_length(mCurDirExe+aDir.NameDir()+*i)<<"o Ref file size "<<ELISE_fp::file_length(mCurDirRef+aDir.NameDir()+*j)<<"o"<<endl;
					}
					else
					{
						//Else display the size of both
						std::cout<<"Files are the SAME : file size "<<ELISE_fp::file_length(mCurDirExe+aDir.NameDir()+*i)<<"o"<<"\n\n";
						DirSz += ELISE_fp::file_length(mCurDirExe+aDir.NameDir()+*i);
					}
				}
			}
		}
		
		//The test is true only if size is the same for the two directory
		if((DirSz==DirExeSz)&&(DirSz==DirRefSz)){aDirReport.TestDirDiff()=true;}
		else{aDirReport.TestDirDiff()=false;}
		
		for(unsigned int aRNF=0;aRNF<RefNotFound.size();aRNF++)
		{
			std::cout<<"WARNING : Ref file "<<RefNotFound[aRNF]<<" NOT FOUND in Exe directory"<<endl;
		}
		
		for(unsigned int aENF=0;aENF<ExeNotFound.size();aENF++)
		{
			std::cout<<"WARNING : Exe file "<<ExeNotFound[aENF]<<" NOT FOUND in Ref directory"<<endl;
		}
		
	}
	else
	{
		//Else display message
		std::cout<<"ERROR : Exe directory "<<mCurDirExe+aDir.NameDir()<<" DOESN'T EXIST"<<endl;
	}
	return aDirReport;
}

cXmlTNR_CalibReport cAppli_TNR_Main::TestCalib(const cXmlTNR_TestDiffCalib & aCalib)
{	
	//mm3d CmpCalib to compare two AutoCal files (in Ori directories)
	std::string aComCp = "mm3d CmpCalib " + (mCurDirExe + aCalib.NameTestCalib()) + " " + (mCurDirRef + aCalib.NameTestCalib())+" DispW=false Out=ecarts_calib.txt XmlG=Calib.xml";
    std::cout << aComCp << "\n";
    System(aComCp);//execute system cmd
    cXmlTNR_TestCalibReport mXML_TestCalib = StdGetFromSI("Calib.xml",XmlTNR_TestCalibReport);//Get the file and parse it
    
    cXmlTNR_CalibReport mCalib;
    mCalib.CalibName () =  mXML_TestCalib.CalibName();
	mCalib.TestCalibDiff() = mXML_TestCalib.TestCalibDiff();
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
	
	//mCalib.EcartsRadiaux() = mXML_TestCalib.EcartsRadiaux();
	//mCalib.DistMatrix() = mXML_TestCalib.DistMatrix();
    return mCalib;//return the same structure for Global Test Report
}

cXmlTNR_OriReport cAppli_TNR_Main::TestOri(const cXmlTNR_TestDiffOri & aOri)
{
	//mm3d CmpOri to compare two Orientation
	std::string aComCp = "mm3d CmpOri " +(mCurDirExe+aOri.PatternTestOri())+" "+aOri.NameTestOri()+" "+aOri.NameTestOri()+" DirOri2="+mCurDirExe +" XmlG=Ori.xml";
    std::cout << aComCp << "\n";
    System(aComCp);//execute system cmd
    cXmlTNR_TestOriReport mXML_TestOri = StdGetFromSI("Ori.xml",XmlTNR_TestOriReport);//Get the file and parse it
    
    cXmlTNR_OriReport mOri;
    mOri.OriName() =  mXML_TestOri.OriName();
	mOri.TestOriDiff() = mXML_TestOri.TestOriDiff();
	mOri.DistCenter() = mXML_TestOri.DistCenter();
	mOri.DistMatrix() = mXML_TestOri.DistMatrix();
    return mOri;//return the same structure for Global Test Report
}

cXmlTNR_ImgReport cAppli_TNR_Main::TestImg(const cXmlTNR_TestDiffImg & aImg)
{
	//mm3d CmpIm to compare images px per px
	std::string aComCp = "mm3d CmpIm " + (mCurDirExe + aImg.NameTestImg()) + " " + (mCurDirRef + aImg.NameTestImg())+" XmlG=CmpImg.xml";
    std::cout << aComCp << "\n";
    System(aComCp);//execute system cmd
    cXmlTNR_TestImgReport mXML_TestImg = StdGetFromSI("CmpImg.xml",XmlTNR_TestImgReport);//Get the file and parse it
    
    cXmlTNR_ImgReport mImg;
    mImg.TestImgDiff() = mXML_TestImg.TestImgDiff();
    if(mImg.TestImgDiff())
    {
		mImg.NbPxDiff() = mXML_TestImg.NbPxDiff();
        mImg.SumDiff() = mXML_TestImg.SumDiff();
        mImg.MoyDiff() = mXML_TestImg.MoyDiff();
		mImg.DiffMaxi() = mXML_TestImg.DiffMaxi();
	}
	return mImg;
}

cXmlTNR_OneTestReport cAppli_TNR_Main::TestOneCom(const cXmlTNR_OneTest & aOneT)
{      
      cXmlTNR_OneTestReport aOTR;//initialize Global Test Report
      
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
	  }
      int aVal = System(mCurCom+" @ExitOnBrkp",true);//execute commande of the test

      //Check if you get de cmd execution status
      if ((aVal != EXIT_SUCCESS)  && (aOneT.TestReturnValue().Val()))
      {
             Error("Cannot exe "+ mCurCom);
             if(mCurCom!=""){aCmd.TestCmd() = false;}
             else{aCmd.TestCmd() = true;}
      }
      else{aCmd.TestCmd() = true;}
      aOTR.XmlTNR_TestCmdReport().push_back(aCmd);
      
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
           aOTR.XmlTNR_OriReport().push_back(aOri);          
      }
      
      //For each Ori to check
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
           aOTR.XmlTNR_ImgReport().push_back(aImg);           
      }
      if(ct==c&&aCmd.TestCmd())
      {
		aOTR.TestOK() = true;
	  }
	  else{aOTR.TestOK() = false;}
	return aOTR;
}

void cAppli_TNR_Main::DoOneGlobTNR(const std::string & aNameFile,const std::string & mOutXML)
{
      cXmlTNR_GlobTestReport aGTR;//initialize Global Test Report
      mXML_CurGT = StdGetFromSI(mEASF.mDir+aNameFile,XmlTNR_GlobTest);
	if(mInDir!=""){mCurDirRef = mEASF.mDir+mInDir+"/";}
	else{
      mCurDirRef = mEASF.mDir+  "TNR-Ref-" + mXML_CurGT.Name() + "/";
     }
      mCurDirExe = mEASF.mDir+  "TNR-Exe-" + mXML_CurGT.Name() + "/";

      // On fait une directory TNR vide
      if (mXML_CurGT.PurgeExe().Val())
      {
          ELISE_fp::PurgeDir(mCurDirExe);
          ELISE_fp::MkDirSvp(mCurDirExe);


          for (std::list<std::string>::const_iterator itN=mXML_CurGT.PatFileInit().begin() ; itN!=mXML_CurGT.PatFileInit().end() ; itN++)
          {
              //std::cout << *itN <<endl;
              std::string aComCp = "cp " + (mCurDirRef + *itN) + " " + mCurDirExe;
              std::cout << aComCp << "\n";
              System(aComCp);
          }
      }

	  int c = 0;
	  int ct = 0;
      for (std::list<cXmlTNR_OneTest>::const_iterator it1T= mXML_CurGT.Tests().begin() ; it1T!= mXML_CurGT.Tests().end() ;  it1T++)
      {
           c += 1;
           cXmlTNR_OneTestReport aOT = TestOneCom(*it1T);
           if(aOT.TestOK())
           {
			   ct += 1;
		   }
           aGTR.XmlTNR_OneTestReport().push_back(aOT);
      }
    aGTR.NbTest() = c;
    aGTR.NbTestOk() = ct;
    
	if(c==ct){aGTR.Bilan()=true;}
	else{aGTR.Bilan()=false;}
	if(mOutXML!=""){MakeFileXML(aGTR, mOutXML);}
	else{MakeFileXML(aGTR, "GlobTest.xml");}
      
}

cAppli_TNR_Main::cAppli_TNR_Main(int argc,char ** argv)  
{
    TheCurElErrorHandlor = this;
    bool aShowArgs = false;
    std::string mOutXML="";
    std::string mInDir="";
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mPattern, "Pattern of Xml test specification",  eSAM_IsPatFile),
         LArgMain()  << EAM(mOutXML,"OutXML", true,"XML out file")
					 << EAM(mInDir,"InDir", true,"Input Reference Directory")
    );

    mEASF.Init(mPattern);
    const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();
	SplitDirAndFile(mCurDirRef, mPattern, mFullName);
	
    for (int aK=0 ; aK<int(aSetIm->size()) ; aK++)
    {
         DoOneGlobTNR((*aSetIm)[aK],mOutXML);
    }
    if (aShowArgs) ShowArgs();
}


//----------------------------------------------------------------------------

int TNR_main(int argc,char ** argv)
{
  cAppli_TNR_Main anAppli(argc,argv);
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
