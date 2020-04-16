#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include<map>

/** \file cMMVII_DaisyFormat.cpp
    \brief Command for generating an audio book, in daisy format, from a list of mp3

    I needed this program and it was the occasion to add some functionnality
  and tests in MMVII

*/


namespace MMVII
{



//========================================================

class cDaisyFOneMp3;
class cDaisyFOneChap;
class cDaisyFOneBook;
class cAppli_Daisy;

// ====   cDaisyFOneMp3  =======================


/**
    Class for storing info on one mp3 file of the daisy format
*/

class cDaisyFOneMp3
{
    public :
       cDaisyFOneMp3(const std::string & aFile,const double & aTime) :
           mFile (aFile),
           mTime (aTime)
       {
       }
       cDaisyFOneMp3() : cDaisyFOneMp3("",0.0) {}

       std::string mFile;  ///< Name mp3 file
       double      mTime;  ///< time in second
      // Calculated parted, not saved
       int         mNum;   ///< nul for generate smil file
       std::string mPrefSmil;  ///< Name "smil" file
       std::string mNameFileSmil;  ///< Name "smil" file
};



void AddData(const  cAuxAr2007 & anAux,cDaisyFOneMp3 & aFMp3)
{
    AddData(cAuxAr2007("File",anAux),aFMp3.mFile);
    AddData(cAuxAr2007("Time",anAux),aFMp3.mTime);
}


// ====   cDaisyFOneChap  =======================

/**
    Class for storing info on chapter of daisy format book.
   Also chapter is not used for now because problem with my understanding,
   better to use it when it will be necessary.
*/

class cDaisyFOneChap
{
    public :
       cDaisyFOneChap(const std::string & aNameChap,const std::list<cDaisyFOneMp3> & aLFiles) :
           mNameChap (aNameChap),
           mFiles    (aLFiles)
       {
       }
       cDaisyFOneChap() : cDaisyFOneChap("",{}) {}
       std::string  mNameChap; ///< Name chapter
       std::list<cDaisyFOneMp3>  mFiles;  ///< list of files

       double Time() const
       {
          double aSom=0.0;
          for (const  auto & aFile : mFiles)
             aSom += aFile.mTime;
          return aSom;
       }
};


void AddData(const  cAuxAr2007 & anAux,cDaisyFOneChap & aChap)
{
    AddData(cAuxAr2007("NameChap",anAux),aChap.mNameChap);
    AddData(cAuxAr2007("Files",anAux),aChap.mFiles);
}

// ====   cDaisyFOneBook  =======================


/**
    Class for storing info on  a daisy book.
*/

class cDaisyFOneBook
{
    public :
       cDaisyFOneBook(const std::string & aNameBook,const std::list<cDaisyFOneChap> & aLChaps) :
           mNameBook (aNameBook),
           mChaps    (aLChaps)
       {
       }
       cDaisyFOneBook() : cDaisyFOneBook("",{}) {}
       double Time() const
       {
          double aSom=0.0;
          for (const  auto & aChap : mChaps)
             aSom += aChap.Time();
          return aSom;
       }

       std::string  mNameBook;  ///< Name of the book
       std::string  mAuthor;  ///< Name of the book
       std::string  mNarrator;  ///< Name of the book
       std::list<cDaisyFOneChap>  mChaps;  ///< Liste of chapter of the book
};

void AddData(const  cAuxAr2007 & anAux,cDaisyFOneBook & aBook)
{
    AddData(cAuxAr2007("NameBook",anAux),aBook.mNameBook);
    AddData(cAuxAr2007("Author",anAux),aBook.mAuthor);
    AddData(cAuxAr2007("Narrator",anAux),aBook.mNarrator);
    AddData(cAuxAr2007("Chaps",anAux),aBook.mChaps);

    StdOut() << "Time " << aBook.Time() << "\n";
}


// ====   cOneEntrySaveWalkman =======================



/* ==================================================== */
/*                                                      */
/*           cAppli_Daisy                               */
/*                                                      */
/* ==================================================== */

///  Class for converting mp3 files to the daisy format (audio book)
/**
    Application for daisy file
    This class has been made for answering to on precise personnal application.
  Not sure it will be usefull to others. BTW, it was the opportunity to check MMVII facilities.
*/


class cAppli_Daisy : public cMMVII_Appli
{
     public :
        cAppli_Daisy(const std::vector<std::string> &  ,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
         std::string StrDuration(const double & aT,bool Full,std::string * aFormat=0) const;
         std::string StrTimeInSec(const double & aT) const;
         void GenerateOneFile(cDaisyFOneMp3 & aFile,const double &aElapsTime);
         void PutDc(const std::string &aField,const std::string & aValue,const std::string & aStrAdd="");
         void PutDcUnk(const std::string &aField);
         void PutNcc(const std::string &aField,const std::string & aValue,const std::string & aStrAdd="");

         std::string  mInput;      ///< xml  file containing a cDaisyFOneBook
         std::string  mPrefOut;    ///< Prefix for Outpt file, def=ncc
         std::string  mOutFile;    ///< Name used for output main file
         bool mTest;               ///< Not used for now
         double mTimeSlot;         ///< Duration to cut mp3 file , def =100.0 sec
         std::ofstream * mPtrOfs;  ///< 
};


cCollecSpecArg2007 & cAppli_Daisy::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
          <<   Arg2007(mInput,"xml file specifying the book",{})
   ;
}

cCollecSpecArg2007 & cAppli_Daisy::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mTest,"Test","Test, if true generate a file to see format",{})
   ;
}


std::string cAppli_Daisy::StrTimeInSec(const double & aT) const
{
   return FixDigToStr(aT,3);
}




std::string  cAppli_Daisy::StrDuration(const double & aT,bool Full,std::string * aFormat) const
{
    return cMMVII_Duration::FromSecond(aT).ToDaisyStr(aFormat,Full);
}


void cAppli_Daisy::GenerateOneFile(cDaisyFOneMp3 & aFile,const double &aElapsTime)
{
   double aTime = aFile.mTime;
   int aNb = std::max(1,round_ni(aTime/mTimeSlot));
   // std::string aStrTimeTot = "00:10:32";
   aFile.mPrefSmil = "icth"+ ToStr(aFile.mNum,4) ;
   aFile.mNameFileSmil = aFile.mPrefSmil +".smil" ;
   std::ofstream aOfstr(aFile.mNameFileSmil);
   aOfstr << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
   aOfstr << "<!DOCTYPE smil PUBLIC \"-//W3C//DTD SMIL 1.0//EN\" \"SMIL10.dtd\">\n";
   aOfstr << "<smil>\n";
   aOfstr << "   <head>\n";
   aOfstr << "      <meta name=\"dc:format\" content=\"Daisy 2.02\"/>\n";
   aOfstr << "      <meta name=\""<<mPrefOut << ":timeInThisSmil\" content=\""   <<  StrDuration(aTime,false)  << "\"/>\n";
   aOfstr << "      <meta name=\""<<mPrefOut << ":totalElapsedTime\" content=\"" << StrDuration(aElapsTime,false)    << "\"/>\n";
   aOfstr << "      <layout> <region id=\"txt-view\"/> </layout> \n";
   aOfstr << "      <meta name=\""+mPrefOut+":generator\" content=\"EasePublisher 2.13 Build 163 # 044FS2212172434\"/>\n";
   aOfstr << "   </head>\n";
   aOfstr << "   <body>\n";
   aOfstr << "      <seq dur=\"" +  StrTimeInSec(aTime) + "s\">\n";
   aOfstr << "         <par endsync=\"last\"> \n";
   aOfstr << "            <text src=\"" +  mOutFile + "#" + aFile.mPrefSmil + "\" id=\""+ aFile.mPrefSmil + "\"/>\n";
   aOfstr << "            <seq>\n";
   for (int aK=0 ; aK<aNb ; aK++)
   {
   double aT0 = (aTime * aK) / aNb;
   double aT1 = (aTime * (aK+1)) / aNb;
   // double aT1 = (aTime * aK) / aNb;
   aOfstr << "                <audio src=\""  <<  aFile.mFile << "\"";
   aOfstr << " clip-begin=\"npt=" << StrTimeInSec(aT0) << "s\" " ;
   aOfstr << " clip-end=\"npt=" << StrTimeInSec(aT1) << "s\" " ;
   aOfstr << " id=\"audio_" << ToStr(aK+1,4)  << "\"" ;
   aOfstr <<   + "/>\n"; 
                       // 12_18th_.mp3" clip-begin="npt=0.000s" clip-end="npt=4.284s" id="audio_0001"/>
   }
   aOfstr << "            </seq>\n";
   aOfstr << "         </par>\n";
   aOfstr << "      </seq>\n";
   aOfstr << "   </body>\n";
   aOfstr << "</smil>\n";
   aOfstr.close();
}

cAppli_Daisy::cAppli_Daisy(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mPrefOut     ("ncc"),
  mTest        (false),
  mTimeSlot    (100.0)
{
}


void cAppli_Daisy::PutDc(const std::string & aField,const std::string & aValue,const std::string & aStrAdd)
{
   *mPtrOfs << "      <meta name=\"dc:" << aField  << "\" content=\"" << aValue << "\" " <<  aStrAdd << "/>\n";
}

void cAppli_Daisy::PutDcUnk(const std::string & aField)
{
    PutDc(aField,"Unknown");
}

void cAppli_Daisy::PutNcc(const std::string & aField,const std::string & aValue,const std::string & aStrAdd)
{
   *mPtrOfs << "      <meta name=\""<<mPrefOut<<":" << aField  << "\" content=\"" << aValue << "\" " <<  aStrAdd << "/>\n";
}



int cAppli_Daisy::Exe()
{
   if (mTest)
   {
      cDaisyFOneMp3 aF1("F1.mp3",10.0);
      cDaisyFOneMp3 aF2("F2.mp3",20.0);
      cDaisyFOneChap aChap1("Chap1",{aF1,aF2});


      cDaisyFOneMp3 aF3("F3.mp3",20.0);
      cDaisyFOneChap aChap2("Chap2",{aF3});

      cDaisyFOneBook aBook("TheBook",{aChap1,aChap2});

      SaveInFile(aBook,"Book.xml");

      return EXIT_SUCCESS;
   }

   mOutFile = mPrefOut + ".html";

   cDaisyFOneBook aXmlSpec ;
   ReadFromFile(aXmlSpec,mInput);

   int aNum=0;
   double aElapsTime = 0.0;

   double aSizeTot=0;
   int    aNbFile=0;
   for (auto & aChap : aXmlSpec.mChaps)
   {
      for (auto & aFile : aChap.mFiles)
      {
          aFile.mNum = aNum++;
          GenerateOneFile(aFile,aElapsTime);
          aElapsTime += aFile.mTime;
          aSizeTot += SizeFile(aFile.mFile);
          aNbFile++;
      }
   }

   // double aTotTime = aXmlSpec.Time();
   std::string aStrTimeTot,aStrFormatTT;
   aStrTimeTot = StrDuration(aElapsTime,false,&aStrFormatTT);

   std::ofstream aOfstr(mOutFile);
   mPtrOfs = & aOfstr;
   aOfstr<< "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
   aOfstr<< "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"xhtml1-transitional.dtd\">\n";
   aOfstr<< "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n";
   aOfstr<< "   <head>\n";
   aOfstr<< "      <title> " << aXmlSpec.mNameBook << " </title>\n";
   PutDc("format","Daisy 2.02");
   PutDc("title",aXmlSpec.mNameBook);
   PutDcUnk("identifier");
   PutDcUnk("publisher");
   PutDc("date","2000-01-01","scheme=\"yyyy-mm-dd\"/");
   PutDc("language","fr","scheme=\"ISO 639\"/");
   PutDc("creator",aXmlSpec.mAuthor);
   PutDcUnk("subject");
   PutDcUnk("source");

   PutNcc("charset","utf-8");
   // PutNcc("tocItems","43"); // ???
   PutNcc("generator","MicMacV2-project 2007-MMVII");
   PutNcc("totalTime",aStrTimeTot,"scheme=\"" + aStrFormatTT + "\"");
/*
   PutNcc("pageFront","0");
   PutNcc("pageNormal","0");
   PutNcc("pageSpecial","0");
   PutNcc("prodNotes","0");
   PutNcc("sidebars","0");
   PutNcc("footnotes","0");
*/
   PutNcc("kByteSize",ToS(round_ni(aSizeTot/1000.)));
   PutNcc("files",ToS(1+2*aNbFile));
   // PutNcc("setInfo","1 of 1");
   // PutNcc("maxPageNormal","26");
   // PutNcc("depth","1");
   // <meta name="ncc:sourceDate" content="2009-07-18" scheme="yyyy-mm-dd"/>
   // <meta name="ncc:multimediaType" content="audioNcc"/>
   // <meta name="ncc:sourceEdition" content="1."/>
   // <meta name="ncc:sourcePublisher" content="The galaxy vol. 8 no. 1 (July 1869),   pages 49-68"/>
   // <meta name="ncc:sourceRights" content="Gutenberg project"/>
   PutNcc("narrator",aXmlSpec.mNarrator);
   PutNcc("sourceTitle",aXmlSpec.mNameBook);
/*
		<meta name="prod:ep_update" content=""/>
		<meta name="prod:last_used_id" content=""/>
		<meta http-equiv="Content-type" content="text/html; charset=utf-8"/>
*/
   aOfstr<< "   </head>\n";

   aOfstr<< "   <body>\n";
/*
   for (auto & aChap : aXmlSpec.mChaps)
   {
      for (auto & aFile : aChap.mFiles)
      {
          aOfstr<< "      <h1 class="\"title>" << << "</h1>\n";
          aFile.mNum = aNum++;
          GenerateOneFile(aFile,aElapsTime);
          aElapsTime += aFile.mTime;
          aSizeTot += SizeFile(aFile.mFile);
          aNbFile++;
      }
   }
*/

   aOfstr<< "   </body>\n";

/*
                <h1 class="title" id="icth0001"><a href="icth0001.smil#icth0001">La chine 1</a></h1>
                <h1 class="title" id="icth0002"><a href="icth0002.smil#icth0002">La chine 2</a></h1>
*/


   aOfstr<< "</html>\n";
   aOfstr.close();

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_Daisy(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_Daisy(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecDaisy
(
     "MediaDaisy",
      Alloc_Daisy,
      "This command is used to generate audio book to daisy format from mp3 files",
      {eApF::Perso},
      {eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);
/*
*/

};

