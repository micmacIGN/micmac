#include "include/MMVII_2Include_Serial_Tpl.h"
#include<map>

/** \file cMMVII_CatVideo.cpp
    \brief Command for concat video

   This command is basic interface to ffmpeg functionnality
of concatenating video (in fact media mp4, mp3 ...).

*/

// mm3d MapCmd  S=1  ffmpeg -i "P=(.*)\.mp4" -vf scale=1280:720 "c=Reduced-\$1.mp4"


namespace MMVII
{


double  TestDNA_1Sz1Event(size_t aSz)
{
    std::vector<double>  aV;
    for (size_t aK=0 ; aK< aSz ; aK++)
	    aV.push_back(RandUnif_0_1());
    std::sort(aV.begin(),aV.end());
    aV.push_back(1+aV[0]);  // make circ

    double aMax = 0;


    for (size_t aK=0 ; aK< aSz ; aK++)
	UpdateMax(aMax,aV[aK+1]-aV[aK]);

    return aMax;
}

cPt2dr  TestDNA_1SzNEvent(size_t aSz,size_t aNbEvent)
{
    std::vector<double>  aV;
    double               aSom=0;
    for (size_t aKEv =0; aKEv<aNbEvent ; aKEv++)
    {
        double aVal = TestDNA_1Sz1Event(aSz);
	aSom += aVal;
        aV.push_back(aVal);
    }
    
    return cPt2dr(NonConstMediane(aV),aSom/aNbEvent);

}

template <typename tFunc> 
        void TestDNA_OneFunc(const std::vector<double> &aVSz ,const std::vector<cPt2dr> &aVMeasTime ,tFunc aFunc,const std::string & aMes)
{
     std::vector<double>  aVFitTime;
     std::vector<double>  aVRatioMed;
     std::vector<double>  aVRatioAvg;
     for (size_t aK=0 ; aK<aVSz.size() ; aK++)
     {
         double aFitTime = aFunc(aVSz[aK]);
	 aVFitTime.push_back(aFitTime);
	 aVRatioMed.push_back(aVMeasTime[aK].x()/aFitTime);
	 aVRatioAvg.push_back(aVMeasTime[aK].y()/aFitTime);
     }

     double aRatioMed = ConstMediane(aVRatioMed);
     double aRatioAvg = ConstMediane(aVRatioAvg);

     StdOut()  <<  "===============  TEST FOR " << aMes << " ================ \n\n";
     for (size_t aK=0 ; aK<aVSz.size() ; aK++)
     {
         double aFitTime = aVFitTime[aK];
         StdOut()   << " Sz=" << aVSz[aK]  
		    << " Med=" << (aVMeasTime[aK].x()/aFitTime)/aRatioMed  
		    << " Avg=" << (aVMeasTime[aK].y()/aFitTime)/aRatioAvg  
		    << "\n";
     } 
}

void  TestDNA()
{
    size_t aNbEv = 10000;

    std::vector<double> aVSz ;
    std::vector<cPt2dr> aVTime ;
    for (int  aK=0 ; aK<=25; aK++)
    {
         size_t aSz = round_ni(5.0 * pow(2.0,aK/4.0));
         cPt2dr  aTime = TestDNA_1SzNEvent(aSz,aNbEv);
	 aVSz.push_back(aSz);
	 aVTime.push_back(aTime);
    }

    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return 1/aN;},"1/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return 1/std::sqrt(aN);},"1/sqrt(N)");

    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN)) /aN;},"Log(N)/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN) +0.5) /aN;},"(Log(N)+0.5)/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN*2)) /aN;},"(Log(2*N))/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN)) /(aN-1);},"(Log(N))/(N-1)");

    //  TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN) +std::log(std::log(aN))) /aN;},"(Log(N)+LogLog(N))/N");
    /*
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN+0.5)) /aN;},"(Log(N)+0.5)/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN-0.5)) /(aN-0.5);},"(Log(N)+0.5)/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN+0.5)) /(aN+0.5);},"(Log(N)+0.5)/N");
    TestDNA_OneFunc(aVSz,aVTime,[](const double & aN){return (std::log(aN)+0.5) /(aN+0.5);},"(Log(N)+0.5)/N");
    */
      
    getchar();
}


/* ==================================================== */
/*                                                      */
/*          cAppli_ReduceVideo                          */
/*                                                      */
/* ==================================================== */


/** Application for concatenating videos */

class cAppli_ReduceVideo : public cMMVII_Appli
{
     public :
        cAppli_ReduceVideo(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
         std::string               mPat;         ///< Pattern of input file

         std::string               mDir;         ///< Pattern of input file
         bool                      mExec;        ///< Execute cat and remove file (else just create file)
	 cPt2di                    mSzReduc;
         std::string               mPrefixRed;
         std::string               mPrefixRedDone;
         int                       mLevMin;
         int                       mLevMax;

         std::vector<std::string>  mAllFiles;
};



cCollecSpecArg2007 & cAppli_ReduceVideo::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
         << Arg2007(mPat,"Pattern for input files")

   ;
}

cCollecSpecArg2007 & cAppli_ReduceVideo::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mExec,"Exec","Execute reduction, else only print",{eTA2007::HDV})
         << AOpt2007(mSzReduc,"SzRed","Reduction size, x=-1 if no reduction",{eTA2007::HDV})
   ;
}


cAppli_ReduceVideo::cAppli_ReduceVideo
(
      const std::vector<std::string> &  aVArgs,
      const cSpecMMVII_Appli & aSpec
) :
  cMMVII_Appli (aVArgs,aSpec),
  mDir            (DirCur()),
  mExec           (false),
  mSzReduc        (1280,720),  // 854x480  640x360
  mPrefixRed      ("Reduced-"),
  mPrefixRedDone  (mPrefixRed + "DONE-"),
  mLevMin         (0),
  mLevMax         (10)
{
}

int cAppli_ReduceVideo::Exe()
{
    mAllFiles = RecGetFilesFromDir(mDir,AllocRegex(mPat),mLevMin,mLevMax);

    std::set<std::string>  aDirsWithWhite;// store  folder with ' '
    std::list<std::string>  aFileFails;// store  folder with ' '

    for (const auto & aFullN0 : mAllFiles)
    {
         std::string aCurDir = DirOfPath(aFullN0);
	 if (aCurDir.find(' ') !=  std::string::npos)  // Dont want to process them
	 {
             aDirsWithWhite.insert(aCurDir);
	 }
	 else
	 {
             std::string aNameInit = FileOfPath(aFullN0);
	     if (! starts_with(aNameInit,mPrefixRed))
	     {
		     StdOut() << "NN=[" << aNameInit << "] RR=[" << mPrefixRed << "]\n";
	         // if name of file has ' ' , replace with '_'
	         if (false) // ( (aNameInit.find(' ') !=  std::string::npos) || (aNameInit.find('&') !=  std::string::npos))
	         {
                      std::string aNameCor;
		      for (const auto & aCar : aNameInit)
		      {
                           char aNew = aCar;
			   if (aNew==' ' ) aNew = '_';
			   if (aNew=='&' ) aNew = 'a';
                           aNameCor.push_back(aNew);
		      }

		      RenameFiles(aCurDir+aNameInit,aCurDir+aNameCor);
		      aNameInit = aNameCor;
	         }
	         std::string aNameTmp   =  mPrefixRed + LastPrefix(aNameInit) + "-TmpRed" + ".mp4";

                 std::string aStrSz;

		 if (mSzReduc.x()>0)  //  x=-1 => convention for conserving size
		 {
                    aStrSz = " -vf scale=" + ToStr(mSzReduc.x()) + ":" + ToStr(mSzReduc.y())  ;
		 }
	         std::string aCom =    "ffmpeg -i " + Quote(aCurDir+aNameInit)
		                     + aStrSz
				     + " " +  Quote(aCurDir+aNameTmp);

		 if (mExec)
		 {
		    if ( GlobSysCall(aCom,true) == EXIT_SUCCESS)
		    {
	                 std::string aNameReduc =  mPrefixRed + LastPrefix(aNameInit) + ".mp4";
	                 std::string aNameDone  =     mPrefixRed + LastPrefix(aNameInit) + "-DONE."+ LastPostfix(aNameInit) ;

			 // if reduction didnt work, maintain names
			 if (SizeFile(aFullN0)<SizeFile(aCurDir+aNameTmp))
			 {
	                    aNameDone  =     mPrefixRed + LastPrefix(aNameInit) + "-DONE-00."+ LastPostfix(aNameInit) ;
                            std::swap(aNameReduc,aNameReduc);
			 }

		         RenameFiles(aCurDir+aNameTmp,aCurDir+aNameReduc);
		         RenameFiles(aFullN0         ,aCurDir+aNameDone);

		    }
		    else
		    {
                         aFileFails.push_back(aFullN0);
		    }
		 }
	         StdOut() << "COM=[" << aCom <<"]\n"; 
	     }
	 }
    }

    if (! aDirsWithWhite.empty())
    {
       StdOut() << "===============  FOLDER WITH WHITE ===============\n";
       for (const auto & aDir : aDirsWithWhite)
       {
           StdOut() << "    DIR=[" << aDir << "]\n";
       }
    }
    if (! aFileFails.empty())
    {
       StdOut() << "===============  FILE WITH FAILS ===============\n";
       for (const auto & aName : aFileFails)
       {
           StdOut() << "    Name=[" << aName << "]\n";
       }
    }

    // std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,tNameSelector  aNS,int aLevMin, int aLevMax);
// void RenameFiles(const std::string & anOldName, const std::string & aNewName); ///< Move/Rename

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_ReduceVideo(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ReduceVideo(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecReduceVideo
(
     "MediaReduceVideo",
      Alloc_ReduceVideo,
      "This command is used for reducing the size of video files",
      {eApF::Perso},
      {eApDT::Media},
      {eApDT::Media},
      __FILE__
);

};

