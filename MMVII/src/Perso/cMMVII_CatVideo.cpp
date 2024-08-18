
#include "MMVII_2Include_Serial_Tpl.h"
#include<map>

/** \file cMMVII_CatVideo.cpp
    \brief Command for concat video

   This command is basic interface to ffmpeg functionnality
of concatenating video (in fact media mp4, mp3 ...).

*/


namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_CatVideo                             */
/*                                                      */
/* ==================================================== */


/** Application for concatenating videos */

class cAppli_CatVideo : public cMMVII_Appli
{
     public :
        cAppli_CatVideo(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
         std::string mPat;         ///< Pattern of input file
         bool        mExec;        ///< Execute cat and remove file (else just create file)
         bool        mAppend;      ///< Do we append to file
         std::string mNameFoF;     ///< name of file of files
         std::string mNameResult;  ///< name of Resulting media
         bool        mVideoMode;       ///< is it video, change def options
         std::vector<std::string> mOptions;     ///< is it video, change options
};



cCollecSpecArg2007 & cAppli_CatVideo::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
         << Arg2007(mPat,"Pattern for input files",{{eTA2007::MPatFile,"0"},eTA2007::FileDirProj})

   ;
}

cCollecSpecArg2007 & cAppli_CatVideo::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mExec,"Exec","Execute concat (vs only create file)",{eTA2007::HDV})
         << AOpt2007(mNameResult,CurOP_Out,"Resulting video")
         << AOpt2007(mVideoMode,"Video","Is it video, if not set computed from post if possible")
         << AOpt2007(mOptions,"Options","Options to add to ffmpeg")
   ;
}


cAppli_CatVideo::cAppli_CatVideo
(
      const std::vector<std::string> &  aVArgs,
      const cSpecMMVII_Appli & aSpec
) :
  cMMVII_Appli (aVArgs,aSpec),
  mExec        (true),
  mAppend      (false),
  mNameFoF     ("FileCatVideo.txt"),
  mNameResult  ("CatVideo")
{
}

int cAppli_CatVideo::Exe()
{
   tNameSet aSetPostfix;
   // Sometime computing postfix fails, so dont do it when you dont need it
   bool ComputePost = (! IsInit(&mNameResult));
  
   cMMVII_Ofs aFileOfFile(mNameFoF,mAppend ? eFileModeOut::AppendText : eFileModeOut::CreateText);
   for (const auto & aStr : ToVect(MainSet0()))
   {
       aFileOfFile.Ofs() << "file '" << aStr << "'" << std::endl;
       if (ComputePost)
       {
          aSetPostfix.Add(LastPostfix(aStr));
       }
       StdOut() << " STR=" << aStr << std::endl;
   }
   aFileOfFile.Ofs().close();

   if (ComputePost)
   {
       std::vector<std::string> aVP = ToVect(aSetPostfix);
       MMVII_INTERNAL_ASSERT_User
       (
           aVP.size()==1,
           eTyUEr::eMultiplePostifx,
           "Unspecified out and non unique (or empty) postfix"
       );
       mNameResult = mNameResult + "." + aVP[0];
   }
   std::string aPost = Postfix(mNameResult);
   if (!IsInit(&mVideoMode))
   {
       if (UCaseEqual(aPost,"mp3"))
         mVideoMode = false;
       else if (UCaseEqual(aPost,"mp4")||UCaseEqual(aPost,"avi"))
         mVideoMode = true;
       else
       {
          MMVII_UnclasseUsEr("Uncataloged media extension, cannot detemine video mode");
       }
   }

/*
   JOE => modif MPD ,   else generate :
   ffmpeg -safe 0 -f concat -i FileCatVideo.txt "-vcodec mpeg4 -b 15000k" toto.mp4
   and the ffmpeg refuse to have  as a single string "-vcodec mpeg4 -b 15000k" 

   if (! IsInit(&mOptions))
   {
      if (mVideoMode)
         mOptions = "-vcodec mpeg4 -b 15000k";
   }

   cParamCallSys aCom("ffmpeg","-safe","0","-f","concat","-i",mNameFoF,mOptions,mNameResult);
*/

   cParamCallSys aCom("ffmpeg","-safe","0","-f","concat","-i",mNameFoF);
   if (! IsInit(&mOptions))
   {
      if (mVideoMode)
         mOptions = {"-vcodec","mpeg4","-b","15000k"};
   }


   for (const auto & anOpt : mOptions)
       aCom.AddArgs(anOpt);
   aCom.AddArgs(mNameResult);
   // End modif


   int aRes = EXIT_SUCCESS ;
   if (mExec) 
   {
      aRes = ExtSysCall(aCom,false);
   }
   StdOut() << "Com=[" << aCom.Com() << "]" << std::endl;

   return aRes;
}


tMMVII_UnikPApli Alloc_CatVideo(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CatVideo(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCatVideo
(
     "MediaCat",
      Alloc_CatVideo,
      "This command is used for concatening medias (interface to ffmpeg)",
      {eApF::Perso,eApF::NoGui},
      {eApDT::Media},
      {eApDT::Media},
      __FILE__
);

};

