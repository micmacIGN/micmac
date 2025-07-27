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

cSetMeasureClino::cSetMeasureClino(const std::string &aPatMatch,const std::string & aPatReplace,const std::vector<std::string> & aVNames) :
      mNamesClino  (aVNames),
      mPatMatch    (aPatMatch),
      mPatReplace  (aPatReplace)
{
}
       
cSetMeasureClino::cSetMeasureClino()
{
}

const std::vector<std::string> &     cSetMeasureClino::NamesClino() const  {return mNamesClino;}
const std::vector<cOneMesureClino>&  cSetMeasureClino::SetMeasures() const {return mSetMeasures;}



void cSetMeasureClino::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("PatMatch",anAux),mPatMatch);
    MMVII::AddData(cAuxAr2007("PatReplace",anAux),mPatReplace);
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

std::string cSetMeasureClino::NameOfIm(const cOneMesureClino &  aMes) const
{
	return ReplacePattern(mPatMatch,mPatReplace,aMes.Ident());
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

const  cOneMesureClino *  cSetMeasureClino::MeasureOfImage(const std::string & aNameImage,bool SVP) const
{

   for (const auto & aMes : mSetMeasures)
   {
        if (NameOfIm(aMes) == aNameImage)
           return & aMes;
   }

   MMVII_INTERNAL_ASSERT_strong(SVP,"Could not get measure clino for Id="+aNameImage);
   return nullptr;
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
	std::vector<std::string> mPrePost;

	// Optionall Arg
	cNRFS_ParamRead            mParamRead;
	std::vector<cPt2di>        mIndCosCorrec;
	std::vector<tREAL8>        mFactMult;


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
	      // <<  Arg2007(mFormat   ,"Format of file as for in spec :  \"" + mSpecFormatTot + "\"")
	      <<  Arg2007(mFormat   ,cNewReadFilesStruct::MsgFormat(mSpecFormatTot))
              <<  mPhProj.DPMeasuresClino().ArgDirOutMand()
	      <<  Arg2007(mPrePost  ,"Pre/Post to add to image",{{eTA2007::ISizeV,"[2,2]"}} )
           ;
}

cCollecSpecArg2007 & cAppli_ImportClino::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    mParamRead.AddArgOpt(anArgOpt);

    return      anArgOpt
	     << AOpt2007(mIndCosCorrec,"ICC","Indexes for cosinus corrections")
	     << AOpt2007(mNbDig,"NbDig","Fix the number of digit for identifier")
	     << AOpt2007(mFactMult,"FactMult","Multiplier of measures M -> M * (1.0 + Value)")
    ;
}

std::string ToFixSize(const std::string & aName,size_t aSize,char aPref = '0')
{
   MMVII_INTERNAL_ASSERT_User_UndefE(aName.size()<=aSize,"String too big to fix size");
   if (aName.size() == aSize) return aName;
   return std::string(aSize-aName.size(),aPref) + aName;
}

int cAppli_ImportClino::Exe()
{
    mPhProj.FinishInit();

    // Read file
    cNewReadFilesStruct aNRFS;
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
    cSetMeasureClino aSetM("^(.*)$", mPrePost.at(0)+"$&"+mPrePost.at(1) );

    std::vector<std::string> aVNames;
    for (size_t aKL=0 ; aKL<aNRFS.NbLineRead() ; aKL++)  // parse all lines
    {
         std::string aNameIdent =  aNRFS.GetValue<std::string>(mNameFieldIm,aKL);

         // for case where name is an int on fix digit but 0 have been omited (pb with CERN data)
         if (IsInit(&mNbDig))
            aNameIdent = ToFixSize(aNameIdent,mNbDig) ;

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

	 // eventually do a cosinus correction
	 if (IsInit(&mIndCosCorrec))
         {
             tREAL8 aExp = -1.0;
             for (const auto &  aP : mIndCosCorrec)
	     {
                 tREAL8  aA1 = aVAngles.at(aP.x());
                 tREAL8  aA2 = aVAngles.at(aP.y());
		 aVAngles.at(aP.x()) = aA1 * std::pow(std::cos(aA2),aExp);
		 aVAngles.at(aP.y()) = aA2 * std::pow(std::cos(aA1),aExp);
	     }
         }

	 if (IsInit(&mFactMult))
         {
             MMVII_INTERNAL_ASSERT_User_UndefE(mFactMult.size()==aVAngles.size(),"Size of multiplier != number of clino");
             for (size_t aK=0 ; aK<aVAngles.size() ; aK++)
                 aVAngles.at(aK) *= (1.0+mFactMult.at(aK));
         }

	 // create the measure
         cOneMesureClino aMesClino(aNameIdent,aVAngles,aVSigma);

	 // check if ident->name generate an existing file, if not generate a warning
	 std::string aNameFile = aSetM.NameOfIm(aMesClino);
	 if (! ExistFile(aNameFile))
	 {
              MMVII_USER_WARNING("Some file dont exist, (for ex : "+ aNameFile + ")");
	 }

	 // add the measure to set
	 aSetM.Add1Mesures(aMesClino);

    }

    // save result in standard file
    mPhProj.SaveMeasureClino(aSetM);

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportClino::Samples() const
{
   return 
   {
       "MMVII ImportClino Data-Input/ClinoValue.txt \"Im AS AS AS AS\" CernInit \"[043_,.tif]\"  ICC=[[0,3],[1,2]]",
       "MMVII ImportMeasuresClino Data-Input/ClinoMeasures.txt \"Im NAS NAS NAS NAS\" Measure0    [,,]"
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

