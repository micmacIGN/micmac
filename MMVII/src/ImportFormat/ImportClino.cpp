#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ReadFileStruct.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Clino.h"


/**
   \file ImportClino.cpp

   \brief file for conversion from raw calibration file to  MMVII format

*/


namespace MMVII
{

/* **************************************************************** */
/*                                                                  */
/*                          cSetMeasureClino                        */
/*                                                                  */
/* **************************************************************** */

cSetMeasureClino::cSetMeasureClino (const std::vector<std::string> & aVNames) :
// (const std::string &aPatMatch)
//,const std::string & aPatReplace,const std::vector<std::string> & aVNames) :
      mNamesClino  (aVNames)
//,      mPatMatch    (aPatMatch),
//      mPatReplace  (aPatReplace)
{
}
       
cSetMeasureClino::cSetMeasureClino() :
    cSetMeasureClino( std::vector<std::string>{})
{
}

const std::vector<std::string> &     cSetMeasureClino::NamesClino() const  {return mNamesClino;}
const std::vector<cOneMesureClino>&  cSetMeasureClino::SetMeasures() const {return mSetMeasures;}



void cSetMeasureClino::AddData(const  cAuxAr2007 & anAux)
{
  //  MMVII::AddData(cAuxAr2007("PatMatch",anAux),mPatMatch);
  //  MMVII::AddData(cAuxAr2007("PatReplace",anAux),mPatReplace);
    MMVII::AddData(cAuxAr2007("NamesClino",anAux),mNamesClino);
    MMVII::AddData(cAuxAr2007("SetMeasures",anAux),mSetMeasures);
}

void AddData(const  cAuxAr2007 & anAux, cSetMeasureClino & aSet)
{
	aSet.AddData(anAux);
}

void cSetMeasureClino::Add1Mesures(const cOneMesureClino & aMes)
{
   // First time we check size-coherence Names/Angle
   if (mSetMeasures.empty())
   {
       if (mNamesClino.empty())  // if names were not set, set them with default
       {
            for (size_t aK=0 ; aK<aMes.Angles().size() ; aK++)
               mNamesClino.push_back("Clino_" + ToStr(aK));
       }
       else if (mNamesClino.size() !=  aMes.Angles().size()) // else check coherence
       {
           MMVII_UnclasseUsEr("Incoherence between number of angles and name");
       }
   }
   else  // else, check size coherence between different clino values
   {
       if (mSetMeasures.back().Angles().size() != aMes.Angles().size())
           MMVII_UnclasseUsEr("Variable size in name clino");
   }
   mSetMeasures.push_back(aMes);
}

void cSetMeasureClino::SetNames(const  std::vector<std::string> & aVNames)
{
    mNamesClino = aVNames;
}

/*
std::string cSetMeasureClino::NameOfIm(const cOneMesureClino &  aMes) const
{
	return ReplacePattern(mPatMatch,mPatReplace,aMes.Ident());
}
*/

std::string cSetMeasureClino::ClinoDeprecatedNameOfImage(const cOneMesureClino&) const
{
    MMVII_INTERNAL_ERROR("ClinoDeprecatedNameOfImage");
    return "";
}

const  cOneMesureClino * cSetMeasureClino::ClinoDeprecatedMeasureOfImage(const std::string & aNameIm) const
{
    MMVII_INTERNAL_ERROR("ClinoDeprecatedNameOfImage");
    return nullptr;
}



void cSetMeasureClino::FilterByPatIdent(const std::string & aPat)
{
    erase_if(mSetMeasures,[aPat](const cOneMesureClino &aMes){return ! MatchRegex(aMes.Ident(),aPat);});
}

const  cOneMesureClino *  cSetMeasureClino::MeasureOfId(const std::string & anId,bool SVP) const
{

   for (const auto & aMes : mSetMeasures)
   {
        if (aMes.Ident() == anId)
           return & aMes;
   }

   MMVII_INTERNAL_ASSERT_strong(SVP,"Could not get measure clino for Id="+anId);
   return nullptr;
}

void cSetMeasureClino::Merge(const cSetMeasureClino& anOld)
{
    if (mNamesClino != anOld.mNamesClino)
    {
         MMVII_UnclasseUsEr ("Different names in cSetMeasureClino::Merge");
    }
    for (const auto & anOldM : anOld.SetMeasures())
    {
        const  cOneMesureClino *  aNewM =  MeasureOfId(anOldM.Ident(),true);
        if (aNewM == nullptr)
        {
            Add1Mesures(anOldM);
        }
    }
}


/* **************************************************************** */
/*                                                                  */
/*                          cOneMesureClino                         */
/*                                                                  */
/* **************************************************************** */

cOneMesureClino::cOneMesureClino() :
    mIdent (MMVII_NONE)
{
}

const std::string &  cOneMesureClino::Ident() const {return mIdent; }
const std::vector<tREAL8> & cOneMesureClino::Angles() const {return mAngles;}
const std::optional<std::vector<tREAL8>> & cOneMesureClino::VSigma() const {return mVSigma;}

cOneMesureClino::cOneMesureClino
(
     const std::string & anId,
     const  std::vector<tREAL8> & aVAngles,
     const std::optional<std::vector<tREAL8>>&  aVSigma
) :
	mIdent  (anId),
	mAngles (aVAngles),
	mVSigma (aVSigma)
{
    if (mVSigma.has_value())
    {
        MMVII_INTERNAL_ASSERT_strong(mAngles.size() == mVSigma->size(),"Incoherence in clino sigma/angles");
    }
}

void cOneMesureClino::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Ident",anAux),mIdent);
    MMVII::AddData(cAuxAr2007("Angles",anAux),mAngles);
    MMVII::AddOptData(anAux,"Sigmas",mVSigma);
}

void AddData(const  cAuxAr2007 & anAux, cOneMesureClino & aMesClino)
{
    aMesClino.AddData(anAux);
}

/* **************************************************************** */
/*                                                                  */
/*                          cSetMeasureClino                        */
/*                                                                  */
/* **************************************************************** */


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportClino                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportClino : public cMMVII_Appli
{
     public :
        cAppli_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;
    // std::vector<std::string> mPrePost;

	// Optionall Arg
	cNRFS_ParamRead            mParamRead;
    //std::vector<cPt2di>        mIndCosCorrec;
    // std::vector<tREAL8>        mFactMult;
	std::vector<std::string>   mChgId;
    int                        mOffsetIdent;    ///< Offset to add to ident
    eModeFusionData            mModeFusionData; // Mode Fusion if data already exist


	//   Format specif
	std::string              mNameFieldIm;
	std::string              mNameFieldAngle;
	std::string              mNameFieldSigma;
	std::string              mNameFieldNClino;
	std::string              mSpecFormatMand;
	std::string              mSpecFormatTot;

    size_t                   mNbDig; ///< Number minimal of digits
};

cAppli_ImportClino::cAppli_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mPhProj         (*this),
   mOffsetIdent    (0),
   mModeFusionData (eModeFusionData::eMerge),
   mNameFieldIm    ("Im"),
   mNameFieldAngle ("A"),
   mNameFieldSigma ("S"),
   mNameFieldNClino ("N"),
   mSpecFormatMand (mNameFieldIm+mNameFieldAngle+"*" + mNameFieldSigma+"*" + mNameFieldNClino + "*"),
   mSpecFormatTot  (cNewReadFilesStruct::MakeSpecTot(mSpecFormatMand,"")),
   mNbDig          (0)
{
	// std::map<std::string,int>  aMap{{"2",2}};
}

cCollecSpecArg2007 & cAppli_ImportClino::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
              <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
              <<  Arg2007(mFormat   ,cNewReadFilesStruct::MsgFormat(mSpecFormatTot))
              <<  mPhProj.DPMeasuresClino().ArgDirOutMand()
        ;
}

cCollecSpecArg2007 & cAppli_ImportClino::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    mParamRead.AddArgOpt(anArgOpt);

    return      anArgOpt
              << AOpt2007(mOffsetIdent,"OffsetId","Offset to add to the ide")
              << AOpt2007(mNbDig,"NbDig","Fix the number of digit for identifier")
              << AOpt2007(mModeFusionData,"MF","Mode of Fusion if already exist",{AC_ListVal<eModeFusionData>()})
              << AOpt2007(mChgId,"ChgId","Change Id [Pat,Name], for ex \"[0(.*),1\\$1]\"  0017->1017",{{eTA2007::ISizeV,"[2,2]"}})
    ;
}

std::string ToFixSize(const std::string & aName,size_t aSize,char aPref = '0')
{
   MMVII_INTERNAL_ASSERT_User_UndefE(aName.size()<=aSize,"String too big to fix size");
   if (aName.size() == aSize) return aName;
   return std::string(aSize-aName.size(),aPref) + aName;
}

std::string StrAddInt(const std::string & aS0,int aAdd)
{
   int aI0 =  cStrIO<int>::FromStr(aS0);

   return cStrIO<int>::ToStr(aI0+aAdd);
}


int cAppli_ImportClino::Exe()
{
    mPhProj.FinishInit();

    // Read file
    cNewReadFilesStruct aNRFS;
    // Add specific rules for reading INT/FLOAT/STRING , here "A" is a name of angles
    aNRFS.SetPatternAddType({"^$",mNameFieldAngle,"^$"});
    aNRFS.SetFormat(mFormat,mSpecFormatMand,mSpecFormatTot);
    aNRFS.ReadFile(mNameFile,mParamRead);

    // Analyse the number of angle & sigma
    size_t aNbAngle = aNRFS.ArrityField(mNameFieldAngle);
    size_t aNbSigma = aNRFS.ArrityField(mNameFieldSigma);
    size_t aNbNameC = aNRFS.ArrityField(mNameFieldNClino);

    if ((aNbSigma!=0) && (aNbSigma!=aNbAngle))
    {
        MMVII_UnclasseUsEr("Nb sigma must equal nb angle or 0");
    }
    if ((aNbNameC!=0) && (aNbNameC!=aNbAngle))
    {
        MMVII_UnclasseUsEr("Nb name clino must equal nb angle or 0");
    }

    // Create set, 
    cSetMeasureClino aSetM;

    std::vector<std::string> aVNames;
    for (size_t aKL=0 ; aKL<aNRFS.NbLineRead() ; aKL++)  // parse all lines
    {
        std::string aNameIdent =  aNRFS.GetValue<std::string>(mNameFieldIm,aKL);

        if (IsInit(&mOffsetIdent))
        {
           aNameIdent =  StrAddInt(aNameIdent,mOffsetIdent);
        }
        // for case where name is an int on fix digit but 0 have been omited (pb with CERN data)
        if (IsInit(&mNbDig))
           aNameIdent = ToFixSize(aNameIdent,mNbDig) ;

        ChgName(mChgId,aNameIdent);

        if (aNbNameC!=0)
        {
            for (size_t aKN=0 ; aKN<aNbNameC ; aKN++)
            {
                std::string aNameC = aNRFS.GetKthValue<std::string>(mNameFieldNClino,aKL,aKN);

                if (aKL==0)
                {
                   StdOut() << "NAME CLINO=" << aNameC << "\n";
                   aVNames.push_back(aNameC);
                }
                else
                {
                    MMVII_INTERNAL_ASSERT_User_UndefE(aNameC==aVNames.at(aKN),"Variable names of clino");
                }
	     }

	     if (aKL==0)
            aSetM.SetNames(aVNames);
	 }

	 // extract angles & sigma
	 std::vector<tREAL8> aVAngles;
	 std::optional<std::vector<tREAL8>>  aVSigma;
	 if (aNbSigma!=0) 
        aVSigma = std::vector<tREAL8>();
	 for (size_t aKA=0 ; aKA<aNbAngle ; aKA++)
	 {
         aVAngles.push_back(aNRFS.GetKthValue<tREAL8>(mNameFieldAngle,aKL,aKA));
	     if (aNbSigma!=0)
             aVSigma->push_back(aNRFS.GetKthValue<tREAL8>(mNameFieldSigma,aKL,aKA));
	 }

	 // create the measure
         cOneMesureClino aMesClino(aNameIdent,aVAngles,aVSigma);

	 // add the measure to set
	 aSetM.Add1Mesures(aMesClino);
    }

    std::string aNameOut = mPhProj.NameFileMeasuresClino(false);
    if (ExistFile(aNameOut))
    {
       cSetMeasureClino aSetOld;
       ReadFromFile(aSetOld,aNameOut);
       switch (mModeFusionData)
       {
            case eModeFusionData::eDoNothing :
                 aSetM = aSetOld;
                 break;

            case eModeFusionData::eOverWrite :
                 break;

             case eModeFusionData::eMerge :
                  aSetM.Merge(aSetOld);
                  break;

            default :
                  MMVII_INTERNAL_ERROR("Measure already exist");
            ;
       }
    }
    else
    {

    }
    //if (mPhProj.NameFileMeasuresClino(false)

    // save result in standard file
    mPhProj.SaveMeasureClino(aSetM);

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportClino::Samples() const
{
   return 
   {
       "MMVII ImportClino Data-Input/ClinoValue.txt \"Im AS AS AS AS\" CernInit   ICC=[[0,3],[1,2]]",
       "MMVII ImportMeasuresClino Data-Input/ClinoMeasures.txt \"Im NAS NAS NAS NAS\" Measure0   "
   };
}


tMMVII_UnikPApli Alloc_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportClino(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportClino
(
     "ImportMeasuresClino",
      Alloc_ImportClino,
      "Import/Convert file of clinometers from raw to MMVII format",
      {eApF::Lines},
      {eApDT::Lines},
      {eApDT::Lines},
      __FILE__
);

}; // MMVII

