#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include<map>

/** \file cMMVII_Walkman.cpp
    \brief Command for my music random selection ;-)

    I needed this program and it was the occasion to add some functionnality
  and tests in MMVII

*/


namespace MMVII
{

class cOneEntrySaveWalkman;
class cOneEntryWalkMan;
class cSaveWalkman;
class cAppli_Walkman ;

// ====   cOneEntrySaveWalkman =======================

class cOneEntrySaveWalkman
{
    public :
        cOneEntrySaveWalkman(const std::string & aName,int aNbList=0) :
           mName        (aName),
           mNbListened  (aNbList)
        {
        }
        cOneEntrySaveWalkman() {}

        std::string mName;
        int         mNbListened;
};
void AddData(const  cAuxAr2007 & anAux,cOneEntrySaveWalkman & anE)
{
    AddData(cAuxAr2007("Name",anAux),anE.mName);
    AddData(cAuxAr2007("NLis",anAux),anE.mNbListened);
}

// ====   cSaveWalkman  =======================

class cSaveWalkman
{
   public :
       cSaveWalkman() :
           mNbTot (0)
       {
       }
       std::vector<cOneEntrySaveWalkman> mVE;
       int                               mNbTot; /// Number of music file from the beging, used for id
};

void AddData(const  cAuxAr2007 & anAux,cSaveWalkman & aSW)
{
    AddData(cAuxAr2007("NbTot",anAux),aSW.mNbTot); /// != sz of list
    AddData(anAux,aSW.mVE);
}


// ====   cOneEntryWalkMan  =======================

class cOneEntryWalkMan : public cOneEntrySaveWalkman
{
    public :
        cOneEntryWalkMan(const std::string & aName,double aFileSz) :
           cOneEntrySaveWalkman(aName),
           mFileSize     (aFileSz),
           mPrio         (1.0),
           mPrioRand     (1.0)
        {
        }
 
        cOneEntryWalkMan() 
        {
        }

        double      mFileSize;
        double      mPrio;
        double      mPrioRand;
};


bool CmpPtrPEW(const cOneEntryWalkMan * anE1,const cOneEntryWalkMan * anE2)
{
   return  anE1->mPrioRand > anE2->mPrioRand;
}

/* ==================================================== */
/*                                                      */
/*          cAppli_Walkman                              */
/*                                                      */
/* ==================================================== */

/// An application for random selection of music
/**
*/


class cAppli_Walkman : public cMMVII_Appli
{
     public :
        cAppli_Walkman(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int DefSeedRand() override { return -1;}
     private :
         std::string mDest;      ///< Destination folder
         double      mTargSize;  ///< Target size for total
         std::string mPat;       ///< Pattern of file
         std::map<std::string,cOneEntryWalkMan>  mMapE;
         std::string  mNameSauv;
};


cCollecSpecArg2007 & cAppli_Walkman::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
          <<   Arg2007(mDest,"Destination folder",{})
          <<   Arg2007(mTargSize,"Sz of selected music in MO",{})
   ;
}

cCollecSpecArg2007 & cAppli_Walkman::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
      anArgOpt
         << AOpt2007(mPat,"Pat","Pattern (regex), def=.*\\.mp3",{eTA2007::HDV})
   ;
}





cAppli_Walkman::cAppli_Walkman(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mPat         (".*\\.mp3"),
  mNameSauv    ("Wakman.xml")
{
}





int cAppli_Walkman::Exe()
{
   cSaveWalkman aSW;
   ReadFromFileWithDef(aSW,mDirProject+mNameSauv);
   // std::vector<std::string> aVN = RecGetFilesFromDir(mDirProject,AllocRegex(mPat),0,10);

   // Create mMapE from file read on folder
   for (const auto & aName :  RecGetFilesFromDir(mDirProject,AllocRegex(mPat),0,10))
   {
       cOneEntryWalkMan anEntry(aName,SizeFile(aName));
       mMapE[FileOfPath(aName)] = anEntry;
   }

   // Put the memorized value of Nb Listened
   for (const auto & aSE :  aSW.mVE)
   {
       auto it = mMapE.find(aSE.mName);
       if (it !=  mMapE.end())
       {
            it->second.mNbListened = aSE.mNbListened;
       }
   }

   // Randomize the priority and put in vect
   std::vector<cOneEntryWalkMan *> mVE;
   for (auto & aPair : mMapE)
   {
       cOneEntryWalkMan & anE  = aPair.second;
       anE.mPrioRand =  (anE.mPrio * RandUnif_0_1()) / (0.5 + anE.mNbListened);
       mVE.push_back(&anE);
   }
   std::sort(mVE.begin(),mVE.end(),CmpPtrPEW);

   // Select the music up tp given size
   int aKSel=0;
   double aSzSel = 0;
   while ((aSzSel<mTargSize) && (aKSel<int(mVE.size())))
   { 
      cOneEntryWalkMan & anE  = *(mVE.at(aKSel));
      aSzSel +=  anE.mFileSize / 1e6;
      StdOut() << "NAME=" << anE.mName <<  " SZF=" << anE.mFileSize << "\n";
      anE.mNbListened++;

      int aNum = aSW.mNbTot;
/*
      std::string aStrNum = ToStr(aNum);
      while (aStrNum.size() < 6)
           aStrNum = "0" + aStrNum;
*/
      std::string aNew = mDest + "Num_"+  ToStr(aNum,6) + "_" + FileOfPath(anE.mName);
      CopyFile(anE.mName, aNew);
      aKSel ++;
      aSW.mNbTot++;
   }
   aSW.mVE.clear();
   for (auto & aPair : mMapE)
   {
      cOneEntrySaveWalkman anE(aPair.first,aPair.second.mNbListened);
      aSW.mVE.push_back(anE);
   }

   StdOut() << "Nb Files " << aKSel << " Sz=" << aSzSel << "\n";
   SaveInFile(aSW,mDirProject+mNameSauv);

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_Walkman(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_Walkman(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecWalkman
(
     "MediaWalkman",
      Alloc_Walkman,
      "This command is used to make a random selection of music",
      {eApF::Perso},
      {eApDT::FileSys,eApDT::Media},
      {eApDT::Xml,eApDT::Media},
      __FILE__
);

};

