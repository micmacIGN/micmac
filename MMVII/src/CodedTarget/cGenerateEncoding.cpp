#include "cMMVII_Appli.h"
#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "CodedTarget.h"
#include "MMVII_Stringifier.h"
#include "MMVII_MeasuresIm.h"
#include "MMVII_Sensor.h"


namespace MMVII
{

class cAppliGenerateEncoding;

class cPrioCC;


/* ************************************************* */
/*                                                   */
/*              cSpecBitEncoding                     */
/*                                                   */
/* ************************************************* */

cSpecBitEncoding::cSpecBitEncoding() :
     mType           (eTyCodeTarget::eIGNIndoor), // Fake init, 4 serialization
     mNbBits         (1<<30),  ///< Absurd val -> must be initialized
     mFreqCircEq     (0),
     mMinHammingD    (1), ///< No constraint
     mUseHammingCode (false),
     mMaxRunL        (1000,1000), ///< No constraint
     mParity         (3), ///< No constraint
     mMaxNb          (1000),
     mBase4Name      (10),
     mNbDigit        (0),
     mPrefix         ("XXXX"),
     mTargetNamePrefix(""),
     mMaxNum         (0),
     mMaxLowCode     (0),
     mMaxCodeEqui    (0)
{
}

void cSpecBitEncoding::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::EnumAddData(anAux, mType,"Type");
    MMVII::AddData(cAuxAr2007("NbBits",anAux),mNbBits);
    MMVII::AddData(cAuxAr2007("FreqCircEq",anAux),mFreqCircEq);
    MMVII::AddData(cAuxAr2007("MinHammingD",anAux),mMinHammingD);
    MMVII::AddData(cAuxAr2007("UseHammingCode",anAux),mUseHammingCode);
    MMVII::AddData(cAuxAr2007("MaxRunL",anAux),mMaxRunL);
    MMVII::AddData(cAuxAr2007("Parity",anAux),mParity);

    MMVII::AddData(cAuxAr2007("MaxNb",anAux),mMaxNb);
    MMVII::AddData(cAuxAr2007("Base4N",anAux),mBase4Name);
    {
       cAuxAr2007 aV("Computed",anAux);
       {
          MMVII::AddData(cAuxAr2007("Prefix",anAux),mPrefix);
          MMVII::AddData(cAuxAr2007("NbDigit",anAux),mNbDigit);
          MMVII::AddData(anAux,"TargetNamePrefix",mTargetNamePrefix, std::string());
          MMVII::AddData(cAuxAr2007("MaxNum",anAux),mMaxNum);
          MMVII::AddData(cAuxAr2007("MaxLowCode",anAux),mMaxLowCode);
          MMVII::AddData(cAuxAr2007("MaxCodeEqui",anAux),mMaxCodeEqui);
       }
    }
}

void AddData(const  cAuxAr2007 & anAux,cSpecBitEncoding & aSpec)
{
     aSpec.AddData(anAux);
}
/* ************************************************* */
/*                                                   */
/*              cOneEncoding                         */
/*                                                   */
/* ************************************************* */

cOneEncoding::cOneEncoding(size_t aNum,size_t aCode,const std::string & aName) 
{
	mName = aName;

	mNC[0] = aNum;
	mNC[1] = aCode;
	mNC[2] = 0;
}
cOneEncoding::cOneEncoding() : cOneEncoding(0,0) {}

void cOneEncoding::SetName(const std::string & aName)
{
    mName = aName;
}

void cOneEncoding::AddData(const  cAuxAr2007 & anAux)
{
   // AddTabData(anAux,mNC,2);
   AddTabData(cAuxAr2007("NumCode",anAux),mNC,2);
   MMVII::AddData(cAuxAr2007("Name",anAux),mName);
   if ((! anAux.Input()) && (mNC[2]!=0))
       anAux.Ar().AddComment(StrOfBitFlag(Code(),size_t(1)<<mNC[2]));
}

size_t cOneEncoding::Num()  const {return mNC[0];}
size_t cOneEncoding::Code() const {return mNC[1];}
const std::string & cOneEncoding::Name() const {return mName;}

void   cOneEncoding::SetNBB (size_t aNbB)
{
    mNC[2] = aNbB;
}

void AddData(const  cAuxAr2007 & anAux,cOneEncoding & anEC)
{
     anEC.AddData(anAux);
}

/* ************************************************* */
/*                                                   */
/*              cBitEncoding                         */
/*                                                   */
/* ************************************************* */

void cBitEncoding::AddOneEncoding(size_t aNum,size_t aCode)
{
     mEncodings.push_back(cOneEncoding(aNum,aCode));
}

void cBitEncoding::AddData(const  cAuxAr2007 & anAuxParam)
{
    cAuxAr2007 anAux (cBitEncoding::TheMainTag,anAuxParam);
    mSpecs.AddData(cAuxAr2007("Specifs",anAux));

    // Used to add comments with string 01
    for (auto & aEncoding : mEncodings )
        aEncoding.SetNBB(mSpecs.mNbBits);

    StdContAddData(cAuxAr2007("Encoding",anAux),mEncodings);
}

void AddData(const  cAuxAr2007 & anAux,cBitEncoding & aBE)
{
    aBE.AddData(anAux);
}

void cBitEncoding::SetSpec(const cSpecBitEncoding& aSpecs)
{
     mSpecs = aSpecs;
}

const cSpecBitEncoding & cBitEncoding::Specs() const {return mSpecs;}
const std::vector<cOneEncoding> &  cBitEncoding::Encodings() const {return mEncodings;}
std::vector<cOneEncoding> &  cBitEncoding::Encodings() {return mEncodings;}

const std::string cBitEncoding::TheMainTag = "BitEncoding";


/*  *********************************************************** */
/*                                                              */
/*             cPrioCC                                          */
/*                                                              */
/*  *********************************************************** */

/**  Class for processing the selection of cells, contains the cell itsel, an "a priori" score,
 *   and a hamming distance (updated)
 */
class cPrioCC
{
     public :
         cPrioCC(cCelCC * aCel,tREAL8 aScoreIntr) ; 

	 tREAL8 Score() const;  ///< "Magic" formula, privilagiate Haming , and use intrinc score when equals
         cCelCC * Cel() const; ///<  Accessor
	 size_t HammingMinD() const;   ///< Accessor

	 void UpdateHammingD(const cPrioCC &);  ///< update distance taking a new selected

     private:
         cCelCC * mCel;
	 tREAL8   mScoreIntr;
	 size_t      mHammingMinD;
};

cPrioCC::cPrioCC(cCelCC * aCel,tREAL8 aScoreIntr) :
     mCel          (aCel),
     mScoreIntr    (aScoreIntr),
     mHammingMinD  (1000)  // Many 
{
}

tREAL8   cPrioCC::Score()       const {return mHammingMinD + mScoreIntr * 1e-5;}
size_t   cPrioCC::HammingMinD() const {return mHammingMinD;}
cCelCC * cPrioCC::Cel()         const {return mCel;}

void cPrioCC::UpdateHammingD(const cPrioCC & aPC2)
{
	UpdateMin(mHammingMinD,mCel->HammingDist(*aPC2.mCel));
}


/*  *********************************************************** */
/*                                                              */
/*             cAppliGenerateEncoding                           */
/*                                                              */
/*  *********************************************************** */


class cAppliGenerateEncoding : public cMMVII_Appli
{
     public :

        cAppliGenerateEncoding(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
	void Show() ;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	cPrioCC * GetBest();
        cPhotogrammetricProject     mPhProj;


	// tREAL8  ScoreOfCodeAndDist(,int aHamingDist);

        cSpecBitEncoding      mSpec;
	int                   mP2;
	size_t                mPerCircPerm;
	bool                  mMiror;
	bool                  mUseAiconCode;
	cCompEquiCodes  *     mCEC;
	std::vector<cCelCC*>  mVOC;
	std::vector<cPrioCC>  mPrioCC;
	std::string           mNameOut;
	std::string           mPostfixOut;
};

cPrioCC * cAppliGenerateEncoding::GetBest()
{
   return WhitchMaxVect(mPrioCC,[](const auto & aPC){return aPC.Score();});
}


cAppliGenerateEncoding::cAppliGenerateEncoding
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli   (aVArgs,aSpec),
   mPhProj        (*this),
   mMiror         (false),
   mCEC           (nullptr)
{
}

cCollecSpecArg2007 & cAppliGenerateEncoding::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<   Arg2007(mSpec.mType  ,"Type among enumerated values",{AC_ListVal<eTyCodeTarget>()})
           <<   Arg2007(mSpec.mNbBits,"Number of bits for the code")
   ;
}


cCollecSpecArg2007 & cAppliGenerateEncoding::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
                  anArgOpt
               << AOpt2007(mSpec.mMinHammingD,"MinHamD","Minimal Hamming Dist (def depend of type)")
               << AOpt2007(mSpec.mMaxRunL,"MaxRunL","Maximal Run Length (def depend of type)")
               << AOpt2007(mSpec.mFreqCircEq,"FreqCircEq","Freq for generating circular permuts (conventionnaly 0->highest) (def depend of type)")
               << AOpt2007(mSpec.mParity,"Parity","Parity check , 1 odd, 2 even, 3 all (def depend of type)")
               << AOpt2007(mSpec.mMaxNb,"MaxNb","Max number of codes",{eTA2007::HDV})
               << AOpt2007(mSpec.mTargetNamePrefix,"TargetNamePrefix","Prefix for targets names",{eTA2007::HDV})
               << AOpt2007(mSpec.mBase4Name,"Base4N","Base for name",{eTA2007::HDV})
               << AOpt2007(mSpec.mNbDigit,"NbDig","Number of digit for name (default depend of max num & base)")
               << AOpt2007(mSpec.mUseHammingCode,"UHC","Use Hamming code")
               << AOpt2007(mSpec.mPrefix,"Prefix","Prefix for output files")
               << AOpt2007(mMiror,"Mir","Unify mirro codes")
               << AOpt2007(mNameOut,"Out","Name for output file")
               << AOpt2007(mPostfixOut,"Postfix","Postfix for output file (def->default tagged extension")
               <<   mPhProj.DPGndPt3D().ArgDirInOpt("GCPNames","Dir GCP for code selection on names")
          ;
}

void cAppliGenerateEncoding::Show()
{
     for (const auto & aPC : mVOC)
         StdOut() << StrOfBitFlag(aPC->mLowCode,mP2) << std::endl;
}

/*
void  Read
NameCERNLookUpTable

void  Read FilesNum(const std::string & aFormat,std::vector<std::vector<double>> & aVRes,const std::string & aNameFile)

void MakeFile3DCern3DTargt(size_t aNBB,size_t aNbD)
{
    std::string aNameFileIn = cCompEquiCodes::NameCERNPannel(aNBB);
    if (!ExistFile(aNameFileIn))
       return;

    cSetMesGCP aSetM("AICON-CERN-Pannel");
    std::string aNameFileOut  = DirOfPath(aNameFileIn) + aSetM.StdNameFile();

    if (ExistFile(aNameFileOut))
       return;

    std::vector<std::vector<double>> aVV;

    ReadFilesNum("FFFFSSSFF",aVV,aNameFileIn);


    for (const auto & aV : aVV)
    {
         cPt3dr aPt(aV.at(1),aV.at(2),aV.at(3));
	 std::string aName = ToStr(round_ni(aV.at(0)),aNbD);
	 aSetM.AddMeasure(cMes1GCP(aPt,aName,1.0));
	 // StdOut() << "N=" << aName << " P=" << aPt << std::endl;
    }

    aSetM.ToFile(aNameFileOut);

    // StdOut()  << "AICON => " << aVV[0] << "......." << aVV.back() << std::endl;
}
*/


int  cAppliGenerateEncoding::Exe()
{
   mPhProj.FinishInit();
   int Num000 = 0;
   //  [0]  ========  Finish initialization and checking ==================
   
   // By convention Freq=0 mean highest frequence 
   if (mSpec.mFreqCircEq==0) 
      mSpec.mFreqCircEq  = mSpec.mNbBits;

   bool  mCompactNum = true;

   // make all default init that are type-dependant
   if (mSpec.mType==eTyCodeTarget::eIGNIndoor)
   {
        SetIfNotInit(mSpec.mFreqCircEq,size_t(2));
        SetIfNotInit(mSpec.mMinHammingD,size_t(3));
        SetIfNotInit(mSpec.mMaxRunL,cPt2di(2,3));
   }
   else if (mSpec.mType==eTyCodeTarget::eIGNDroneSym)
   {
        SetIfNotInit(mSpec.mFreqCircEq,size_t(2));
        SetIfNotInit(mSpec.mMinHammingD,size_t(3));
        SetIfNotInit(mSpec.mMaxRunL,cPt2di(2,3));
   }
   else if (mSpec.mType==eTyCodeTarget::eIGNDroneTop)
   {
        SetIfNotInit(mSpec.mFreqCircEq,size_t(1));
        SetIfNotInit(mSpec.mUseHammingCode,true);
   }
   else if (mSpec.mType==eTyCodeTarget::eCERN)
   {
        mUseAiconCode = true;
        mCompactNum   = false;
        SetIfNotInit(mSpec.mParity,size_t(2));
	Num000 = 1;
   }

   cHamingCoder aHC(1);
   if (mSpec.mUseHammingCode) // if we use hamming code, not all numbers of bits are possible
   {
      aHC = cHamingCoder::HCOfBitTot(mSpec.mNbBits,true); // true : at the end we  want even number of bit (?)
      mSpec.mNbBits = aHC.NbBitsOut();
   }


   // for comodity, user specify a frequency, we need to convert it in a period
   MMVII_INTERNAL_ASSERT_strong((mSpec.mNbBits%mSpec.mFreqCircEq)==0,"NbBits should be a multiple of frequency");
   mPerCircPerm = mSpec.mNbBits / mSpec.mFreqCircEq;

   // check base is valide
   MMVII_INTERNAL_ASSERT_User
   (
        (mSpec.mBase4Name>=2)&&(mSpec.mBase4Name<=36),
	eTyUEr::eUnClassedError,
	"Base shoulde be in [2 36]"
   );

   //  Set the prefix usigng complicated defaut rule
   if (! IsInit(&mSpec.mPrefix))
   {
      mSpec.mPrefix =    E2Str(mSpec.mType) 
                       + "_Nbb"  + ToStr(mSpec.mNbBits)
                       + "_Freq" + ToStr(mSpec.mFreqCircEq)
                       + "_Hamm" + ToStr(mSpec.mMinHammingD)
                       + "_Run" + ToStr(mSpec.mMaxRunL.x()) + "_" + ToStr(mSpec.mMaxRunL.y());
   }
   if (! IsInit(&mPostfixOut))
      mPostfixOut = TaggedNameDefSerial();

   if (! IsInit(&mNameOut))
      mNameOut  =   mSpec.mPrefix + "_SpecEncoding." + mPostfixOut;

   // calls method in cMMVII_Appli, to show current value of params, as many transformation have been made
   ShowAllParams();


   mP2 = (1<<mSpec.mNbBits);

   //  [1] =============   read initial value of cells
   mCEC = cCompEquiCodes::Alloc(mSpec.mNbBits,mPerCircPerm,mMiror);
   mVOC = mCEC->VecOfCells();
   StdOut() <<  "Size Cells init " << mVOC.size() << std::endl;

   //  [2]  ========  filter : if there exist an external file of codes, use it to filter ==========

   if (mUseAiconCode)
   {
       // Read the file in ressources MMVII
       std::vector<cPt2di>  aVCode ;
       ReadCodesTarget(aVCode,cCompEquiCodes::NameCERNLookUpTable(mSpec.mNbBits));

       // In this case, by default, take all the code that were specified
       SetIfNotInit(mSpec.mMaxNb,aVCode.size());
       std::list<cCompEquiCodes::tAmbigPair>  aLamb = mCEC->AmbiguousCode(aVCode);

       if (!aLamb.empty())
       {
           MMVII_DEV_WARNING("Use of ambiguous filter code");
       }
       mVOC = mCEC->VecOfUsedCode(aVCode,true);
       StdOut() <<  "Size after file filter " << mVOC.size() << std::endl;


       if (1)
       {
          for (size_t aK=1 ; aK<aVCode.size(); aK++)
	  {
              // StdOut() << "KKK " << aVCode[aK] << std::endl;
              MMVII_INTERNAL_ASSERT_bench(aVCode[aK-1].x() < aVCode[aK].x(),"Not growing order for num in 3D-AICON");
              MMVII_INTERNAL_ASSERT_bench(aVCode[aK-1].y() < aVCode[aK].y(),"Not growing order for bitflag in 3D-AICON");
	  }

          for (size_t aK=0 ; aK<aVCode.size(); aK++)
	  {
		 cCelCC * aCel = mCEC->CellOfCode(aVCode[aK].y());
                 aCel->mNum = aVCode[aK].x();
                 MMVII_INTERNAL_ASSERT_bench(aCel!=0,"CellOfCode in3D AICON");
                 MMVII_INTERNAL_ASSERT_bench(aVCode[aK].y()==(int)aCel->mLowCode,"CellOfCode in3D AICON");
	  }


       }
   }

   if (mPhProj.DPGndPt3D().DirInIsInit())
   {
      cSetMesGnd3D aSetGCP =  mPhProj.LoadGCP3D();
      std::set<int>   aLInt;
      for (const auto & aGCP : aSetGCP.Measures())
          aLInt.insert(cStrIO<int>::FromStr(aGCP.mNamePt));
      //  StdOut() << "VOOOCSIZE= " << mVOC.size()  << " "  << aSetGCP.Measures().size() << "\n";
      erase_if
      (
             mVOC,
             [aLInt] (const auto & aPtr) {return  ! MapBoolFind(aLInt,aPtr->mNum);}
      );
      //  StdOut() << "VOOOCSIZE= " << mVOC.size()  << " "  << aSetGCP.Measures().size() << "\n";
      //  getchar();
      // std::vector<cCelCC*>  mVOC;

      //  StdOut() << "VOOOCSIZE= " << mVOC.size()  << "\n";
      //  StdOut() <<  "  * N0=" <<  mVOC.at(0)->mNum << "\n";
      //  StdOut() <<  "  * N1=" <<  mVOC.at(1)->mNum << "\n";
   }
  
   // [3.0]  if we use hamming code, not all numbers are possible
   if (mSpec.mUseHammingCode) 
   {
       VecFilter
       (
	     mVOC,
             [&aHC](auto aPC) 
             {
                return aHC.UnCodeWhenCorrect(aPC->mLowCode) <0;
             } 
       );
       StdOut() <<  "Size after hamming code " << mVOC.size()  << std::endl;
   }
   //  [3]  ========  filter : if there is a parity check  ====================

   if (mSpec.mParity !=3)
   {
       VecFilter
       (
	     mVOC,
             [&](auto aPC) 
             {
	       bool toSupress =   ((NbBits(aPC->mLowCode) %2==1)  ^  (mSpec.mParity == 1)) ;
                return  toSupress;
             } 
       );
       StdOut() <<  "Size after parity filter " << mVOC.size()  <<  " PARITY=" << mSpec.mParity << std::endl;
   }

   //  [4]  ========  filter : if there is constraint on run lenght  ====================

   if (IsInit(&mSpec.mMaxRunL))
   {
       VecFilter
       (
	     mVOC,
             [&](auto aPC) 
             {
		     cPt2di aRL = MaxRunLength(aPC->mLowCode,mP2);
		     return (aRL.x() > mSpec.mMaxRunL.x()) || (aRL.y() > mSpec.mMaxRunL.y());
             } 
       );
       StdOut() <<  "Size after max run lenght filter " << mVOC.size() << std::endl;
   }
     
   //  [5]  ========  make a selection of "maxmimal" subset respecting hamming criteria  ====================

        // 5.1   initialize :   priority queue in mPrioCC
   for (auto aCC : mVOC)
   {
       // CM: MaxRun2Length return an unsigned size_t. We must convert it to double before applying unary '-' operator !
       tREAL8 aScore = - double(MaxRun2Length(aCC->mLowCode,mP2));
        mPrioCC.push_back(cPrioCC(aCC,aScore));
   }

   bool GoOn = ! mVOC.empty(); // Precaution, else core dump when get null ptr

   std::vector<cCelCC*>  aNewVOC;

   cTimeSequencer aTSeq(0.5); // to make use patientate
			     
  
   bool  mDoFilter = (mSpec.mMinHammingD!=1) || (mSpec.mMaxNb<mVOC.size()  );
   StdOut() <<  " MaxNb=" << mSpec.mMaxNb << "\n";
   StdOut() <<  " MinH=" << mSpec.mMinHammingD << "\n";
         //  5.2 Now iteratively select one and update others
   if (mDoFilter)
   {
     while (GoOn)
     {
       // Extract best solution
       cPrioCC * aNextP = GetBest();

       // if best one is under threshold end
       if (aNextP->HammingMinD() < mSpec.mMinHammingD)
       {
           GoOn = false;
       }
       else
       {
           aNewVOC.push_back(aNextP->Cel());  // add new one
           for (auto & aPC : mPrioCC) // update remaining
	       aPC.UpdateHammingD(*aNextP);

	   if (aNewVOC.size() >= mSpec.mMaxNb)  // if enoug stop
              GoOn = false;
       }
       if (aTSeq.ItsTime2Execute())  // make user patient
       {
	   StdOut() << "Hamming filter, still to do " << mSpec.mMaxNb-aNewVOC.size() << std::endl;
       }
     }
  }
  else
  {
      aNewVOC = mVOC;
  }
     
   StdOut() << "SSSSS " << mVOC.size() << " => " << aNewVOC.size() << "\n";
   mVOC = aNewVOC;
   StdOut() <<  "Size after hamming  distance selection" << mVOC.size() << std::endl;

   //  [6] ==================== Finalization : 
   //        * Compute range (max val)  of code, code-equiv and num
   //        * put the selected encoding in a cBitEncoding, 
   //        * compute nb of digit,  names ..
   //        * save in a file

   // restore order on low code that may have been supress by hamming, useful also to 
   // restor AICON num
   SortOnCriteria(mVOC,[](auto aPCel){return aPCel->mLowCode;});
   {
       cBitEncoding aBE;
       for (size_t aK1=0 ; aK1<mVOC.size(); aK1++)  
       {
           size_t aNum = aK1 + Num000;
// StdOut() << "NNNNNnnN= " << aNum  << " " << mVOC[aK1]->mNum << "\n";
           MMVII_INTERNAL_ASSERT_strong(mVOC[aK1]->mNum>=0,"Num was not correctly set in cCelCC");
           // For AICON-like, with external spec, with maintain the numerotation, for internal MMVII system, we
           // prefer to "compactify" the numbering
           if (! mCompactNum)
           {
              aNum = mVOC[aK1]->mNum;
           }
	   size_t aCode = mVOC[aK1]->mLowCode;
           aBE.AddOneEncoding(aNum,aCode);  // add a new encoding

	   // Update all ranges
	   UpdateMax(mSpec.mMaxNum,aNum);
	   UpdateMax(mSpec.mMaxLowCode,aCode);
	   for (const auto & aCodeEqui : mCEC->CellOfCodeOK(aCode).mEquivCode )
	        UpdateMax(mSpec.mMaxCodeEqui,aCodeEqui);
       }

       size_t aNbD= GetNDigit_OfBase(mSpec.mMaxNum, mSpec.mBase4Name);
       UpdateMax(mSpec.mNbDigit,aNbD);

       StdOut() <<  "NMax=" << mSpec.mMaxNum << " NDig=" <<  mSpec.mNbDigit 
	        << " "  << NameOfNum_InBase(mSpec.mMaxNum, mSpec.mBase4Name,mSpec.mNbDigit) 
	        << " "  << NameOfNum_InBase(9, mSpec.mBase4Name,mSpec.mNbDigit) 
		<< "\n";

       for (auto & anEncode : aBE.Encodings())
       {
           anEncode.SetName(mSpec.mTargetNamePrefix + NameOfNum_InBase(anEncode.Num(),mSpec.mBase4Name,mSpec.mNbDigit));
       }

       aBE.SetSpec(mSpec);
       SaveInFile(aBE,mNameOut);
   }

   /*
   if (mUseAiconCode)
   {
       MakeFile3DCern3DTargt(mSpec.mNbBits,mSpec.mNbDigit);
   }		  
   */

   delete mCEC;


   return EXIT_SUCCESS;
}

void GenSpec_BitEncoding(const std::string & aDir)
{
    SpecificationSaveInFile<cBitEncoding>(aDir +"BitEncoding.xml");
    SpecificationSaveInFile<cBitEncoding>(aDir +"BitEncoding.json");

}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_GenerateEncoding(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenerateEncoding(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecGenerateEncoding
(
     "CodedTargetGenerateEncoding",
      Alloc_GenerateEncoding,
      "Generate en encoding for coded target, according to some specification",
      {eApF::CodedTarget},
      {eApDT::None},
      {eApDT::Xml},
      __FILE__
);


};

