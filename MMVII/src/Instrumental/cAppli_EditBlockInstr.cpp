#include "MMVII_InstrumentalBlock.h"

#include "cMMVII_Appli.h"
#include "MMVII_2Include_Serial_Tpl.h"




/**
  \file cInstrumentalBloc.cpp


  \brief This file contains the core implemantation of Block of rigid instrument
 
*/

namespace MMVII
{


/* *************************************************************** */
/*                                                                 */
/*                        cAppli_EditBlockInstr                    */
/*                                                                 */
/* *************************************************************** */

class cAppli_EditBlockInstr : public cMMVII_Appli
{
     public :

        cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject   mPhProj;       //< As usual ....
        std::string               mNameBloc;     //< Name of the block edited (generally default MMVII)
        std::vector<std::string>  mVPatsIm4Cam;  //< Patterns for cam structure : [PatSelOnDisk,PatTimeStamp?,PatSelInBlock?]
        bool                      mFromScratch;  //< If exist file : Reset of Modify ?
        std::vector<std::vector<std::string>>  mCstrOrthog;  //<  Vector for relative orientations
        std::vector<int>                       mNumPoseInstr;
};

cAppli_EditBlockInstr::cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this),
    mNameBloc     (cIrbCal_Block::theDefaultName),
    mFromScratch  (false)
{
}


std::vector<std::string>  cAppli_EditBlockInstr::Samples() const 
{
   return 
   {
       "MMVII EditBlockInstr Bl0  PatsIm4Cam='[.*_(.*).tif]' InMeasureClino=MesClin_043",
       "MMVII EditBlockInstr Bl0  PatsIm4Cam='[.*tif,.*_(.*).tif,Fils-100.xml]' InMeasureClino=MesClin_043"
   };
}


cCollecSpecArg2007 & cAppli_EditBlockInstr::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
     ;

}

cCollecSpecArg2007 & cAppli_EditBlockInstr::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
            << AOpt2007(mVPatsIm4Cam,"PatsIm4Cam","Pattern images [PatSelOnDisk,PatTimeStamp?,PatSelInBlock?]",{{eTA2007::ISizeV,"[1,3]"}})
            << AOpt2007(mFromScratch,"FromScratch","Do we start from a new file, even if already exist",{{eTA2007::HDV}})
            << mPhProj.DPBlockInstr().ArgDirOutOpt()
            << mPhProj.DPMeasuresClino().ArgDirInOpt()
            << AOpt2007(mCstrOrthog,"CstrOrthog","Constraint for vectors orthogonality [[Instr1,Instr2,Sigma]*...] ")
            << AOpt2007(mNumPoseInstr,"NPI","Num of cams used  for estimate pose of intsrument")
        ;
}


int cAppli_EditBlockInstr::Exe() 
{
    mPhProj.DPBlockInstr().SetDirOutInIfNotInit();
    mPhProj.FinishInit();

    cIrbCal_Block *  aBlock =    mFromScratch                           ?
                                 new cIrbCal_Block                      :
                                 mPhProj.ReadRigBoI(mNameBloc,SVP::Yes) ;


    // if we add structure for camera
    if (IsInit(&mVPatsIm4Cam))
    {
        std::string aPatSelOnDisk = mVPatsIm4Cam.at(0);
        std::string aPatTimeStamp = GetDef(mVPatsIm4Cam,1,aPatSelOnDisk);
        std::string aPatSelIm = GetDef(mVPatsIm4Cam,2,aPatTimeStamp);

        auto aVNameIm = ToVect(SetNameFromString(aPatSelOnDisk,true));
        std::vector<std::string> aSetNameCal;
        for (const auto & aNameIm : aVNameIm)
        {
            std::string aNameCal = mPhProj.StdNameCalibOfImage(aNameIm);
            if (! BoolFind(aSetNameCal,aNameCal))
            {
                aSetNameCal.push_back(aNameCal); // OK
            }
        }
        std::sort(aSetNameCal.begin(),aSetNameCal.end());

        for (const auto & aNameCal : aSetNameCal)
            aBlock->SetCams().AddCam(aNameCal,aPatTimeStamp,aPatSelIm,SVP::Yes); // Not OK:
    }

    if (IsInit(&mNumPoseInstr))
       aBlock->SetCams().SetNumPoseInstr(mNumPoseInstr);




    // if we add the structure for clinometers
    if (mPhProj.DPMeasuresClino().DirInIsInit())
    {
         cSetMeasureClino aMesClin =  mPhProj.ReadMeasureClino();
         const std::vector<std::string> &  aVNames = aMesClin.NamesClino();
         size_t aNbC = aVNames.size();
         std::vector<cWeightAv<tREAL8,tREAL8>> aVW (aNbC);

         for (const auto & aMes: aMesClin.SetMeasures())
         {
            const auto & aSigm = aMes.VSigma();
            if (aSigm.has_value())
            {
                for (size_t aK=0 ; aK<aNbC ; aK++ )
                    aVW.at(aK).Add(1.0,aSigm.value().at(aK));
            }
  //          StdOut() << "SIGINIT= " << aSigm.has_value() << " " << aMes.Ident()<< "\n";
         }
         for(size_t aK=0 ; aK<aNbC ; aK++ )
         {
             tREAL8 aSigma =  aVW.at(aK).Average(-1.0);
         //    StdOut() << "  SIGMA " << aSigma << "\n";
             aBlock->SetClinos().AddClino(aVNames.at(aK),aSigma,SVP::Yes);
         }
    }

    if (IsInit(&mCstrOrthog))
    {
        for (const auto & aCstr : mCstrOrthog)
        {
            MMVII_INTERNAL_ASSERT_User_UndefE(aCstr.size()==3,"Bad size for cstr ori rel");
            aBlock->AddCstrRelOrthog
            (
                aCstr.at(0),
                aCstr.at(1),
                cStrIO<double>::FromStr(aCstr.at(2))
            );
        }
    }

    // save the result on disk
    mPhProj.SaveRigBoI(*aBlock);

    delete aBlock;
    return EXIT_SUCCESS;
}


    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_EditBlockInstr(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EditBlockInstr(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_EditBlockInstr
(
     "BlockInstrEdit",
      Alloc_EditBlockInstr,
      "Create/Edit a block of instruments",
      {eApF::BlockInstr},
      {eApDT::BlockInstr},
      {eApDT::BlockInstr},
      __FILE__
);


};

