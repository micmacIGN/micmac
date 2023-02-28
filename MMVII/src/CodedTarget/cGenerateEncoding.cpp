#include "cMMVII_Appli.h"
#include "MMVII_nums.h"
#include "MMVII_util_tpl.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "CodedTarget.h"
#include "MMVII_Stringifier.h"

namespace MMVII
{
using namespace cNS_CodedTarget;
namespace  cNS_CodedTarget
{
/* ************************************************* */
/*                                                   */
/*              cSpecBitEncoding                     */
/*                                                   */
/* ************************************************* */

cSpecBitEncoding::cSpecBitEncoding() :
     mNbBits        (1<<30),  ///< Absurd val -> must be initialized
     mFreqCircEq    (0),
     mMinHammingD   (1), ///< No constraint
     mMaxRunL       (1000,1000), ///< No constraint
     mParity        (3), ///< No constraint
     mMaxNb         (1000)
{
}

void cSpecBitEncoding::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::EnumAddData(anAux, mType,"Type");
    MMVII::AddData(cAuxAr2007("NbBits",anAux),mNbBits);
    MMVII::AddData(cAuxAr2007("FreqCircEq",anAux),mFreqCircEq);
    MMVII::AddData(cAuxAr2007("MinHammingD",anAux),mMinHammingD);
    MMVII::AddData(cAuxAr2007("MaxRunL",anAux),mMaxRunL);
    MMVII::AddData(cAuxAr2007("Parity",anAux),mParity);
    MMVII::AddData(cAuxAr2007("MaxNb",anAux),mMaxNb);
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

cOneEncoding::cOneEncoding(size_t aNum,size_t aCode) 
{
	mNC[0] = aNum;
	mNC[1] = aCode;
}
cOneEncoding::cOneEncoding() : cOneEncoding(0,0) {}

void cOneEncoding::AddData(const  cAuxAr2007 & anAux)
{
   if (! anAux.Input())
       AddComment(anAux.Ar(),StrOfBitFlag(Code(),1<<mNC[2]));
   AddTabData(anAux,mNC,2);
}
           // void   SetNBB (size_t ) ; ///< used to vehicle info 4 AddComm

size_t cOneEncoding::Num()  const {return mNC[0];}
size_t cOneEncoding::Code() const {return mNC[1];}

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

void cBitEncoding::AddData(const  cAuxAr2007 & anAux)
{
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



/*  *********************************************************** */
/*                                                              */
/*             cAppliGenerateEncoding                           */
/*                                                              */
/*  *********************************************************** */

class cPrioCC
{
     public :
         cPrioCC(cCelCC * aCel,tREAL8 aScoreIntr) ; 

	 tREAL8 Score() const;
	 size_t HammingMinD() const;
         cCelCC * Cel() const;

	 void UpdateHammingD(const cPrioCC &);

     private:
         cCelCC * mCel;
	 tREAL8   mScoreIntr;
	 size_t      mHammingMinD;
};

cPrioCC::cPrioCC(cCelCC * aCel,tREAL8 aScoreIntr) :
     mCel          (aCel),
     mScoreIntr    (aScoreIntr),
     mHammingMinD  (100)
{
}

tREAL8   cPrioCC::Score()       const {return mHammingMinD + mScoreIntr * 1e-5;}
size_t   cPrioCC::HammingMinD() const {return mHammingMinD;}
cCelCC * cPrioCC::Cel()         const {return mCel;}

void cPrioCC::UpdateHammingD(const cPrioCC & aPC2)
{
	UpdateMin(mHammingMinD,mCel->HammingDist(*aPC2.mCel));
}

};


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

	// tREAL8  ScoreOfCodeAndDist(,int aHamingDist);

        cSpecBitEncoding      mSpec;
	int                   mP2;
	size_t                mPerCircPerm;
	bool                  mMiror;
	bool                  mUseAiconCode;
	cCompEquiCodes  *     mCEC;
	std::vector<cCelCC*>  mVOC;
	std::vector<cPrioCC>  mPrioCC;
	bool                  mShow;
};

cAppliGenerateEncoding::cAppliGenerateEncoding
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli   (aVArgs,aSpec),
   mMiror         (false),
   mCEC           (nullptr),
   mShow          (false)
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
               << AOpt2007(mShow,"Show","Show Res at end",{eTA2007::HDV})
          ;
}

void cAppliGenerateEncoding::Show()
{
     for (const auto & aPC : mVOC)
         StdOut() << StrOfBitFlag(aPC->mLowCode,mP2) << "\n";
}

int  cAppliGenerateEncoding::Exe()
{
   if (mSpec.mFreqCircEq==0) 
      mSpec.mFreqCircEq  = mSpec.mNbBits;


   if (mSpec.mType==eTyCodeTarget::eIGNIndoor)
   {
        SetIfNotInit(mSpec.mFreqCircEq,size_t(2));
        SetIfNotInit(mSpec.mMinHammingD,size_t(3));
        SetIfNotInit(mSpec.mMaxRunL,cPt2di(3,2));
   }
   else if (mSpec.mType==eTyCodeTarget::eIGNDrone)
   {
        SetIfNotInit(mSpec.mFreqCircEq,size_t(2));
        SetIfNotInit(mSpec.mMinHammingD,size_t(3));
        SetIfNotInit(mSpec.mMaxRunL,cPt2di(3,2));
   }
   else if (mSpec.mType==eTyCodeTarget::eCERN)
   {
        mUseAiconCode = true;
        SetIfNotInit(mSpec.mParity,size_t(2));
   }

   MMVII_INTERNAL_ASSERT_strong((mSpec.mNbBits%mSpec.mFreqCircEq)==0,"NbBits should be a multiple of Nb Bits");
   mPerCircPerm = mSpec.mNbBits / mSpec.mFreqCircEq;

   StdOut() <<  " Freq=" <<   mSpec.mFreqCircEq
	    <<  " HamD=" <<   mSpec.mMinHammingD
	    <<  " MaxR=" <<   mSpec.mMaxRunL
	    <<  " Parity=" << mSpec.mParity
	    << "\n";


   mP2 = (1<<mSpec.mNbBits);
   //  read initial value of cells
   mCEC = cCompEquiCodes::Alloc(mSpec.mNbBits,mPerCircPerm,mMiror);
   mVOC = mCEC->VecOfCells();
   StdOut() <<  "Size Cells init " << mVOC.size() << "\n";

   //  if there exist an external file of codes, use it to filter
   if (mUseAiconCode)
   {
       std::vector<cPt2di>  aVCode;
       ReadCodesTarget(aVCode,cCompEquiCodes::NameCERNLookUpTable(mSpec.mNbBits));

       std::list<cCompEquiCodes::tAmbigPair>  aLamb = mCEC->AmbiguousCode(aVCode);

       if (!aLamb.empty())
       {
           MMVII_DEV_WARNING("Use of ambiguous filter code");
       }
       mVOC = mCEC->VecOfUsedCode(aVCode,true);
       StdOut() <<  "Size after file filter " << mVOC.size() << "\n";
   }
   // Id there is a parity check
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
       StdOut() <<  "Size after parity filter " << mVOC.size()  <<  " PARITY=" << mSpec.mParity << "\n";
   }

   // Id there is a mMaxRunL
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
       StdOut() <<  "Size after max run lenght filter " << mVOC.size() << "\n";
   }

   for (auto aCC : mVOC)
   {
        tREAL8 aScore = - MaxRun2Length(aCC->mLowCode,mP2);
        mPrioCC.push_back(cPrioCC(aCC,aScore));
   }

   bool GoOn = ! mVOC.empty();

   std::vector<cCelCC*>  aNewVOC;

   cTimeSequencer aTSeq(0.5);
   while (GoOn)
   {
       cPrioCC * aNextP = WhitchMaxVect(mPrioCC,[](const auto & aPC){return aPC.Score();});

       if (aNextP->HammingMinD() < mSpec.mMinHammingD)
       {
           GoOn = false;
       }
       else
       {
           aNewVOC.push_back(aNextP->Cel());
           for (auto & aPC : mPrioCC)
	       aPC.UpdateHammingD(*aNextP);

	   if (aNewVOC.size() >= mSpec.mMaxNb)
              GoOn = false;
       }
       if (aTSeq.ItsTime2Execute())
       {
	   StdOut() << "Hamming filter, still to do " << mSpec.mMaxNb-aNewVOC.size() << "\n";
       }
   }
   mVOC = aNewVOC;
   StdOut() <<  "Size after after hamming " << mVOC.size() << "\n";

   {
       cBitEncoding aBE;
       aBE.SetSpec(mSpec);
       for (size_t aK=0 ; aK<mVOC.size(); aK++)  
       {
           aBE.AddOneEncoding(aK+1,mVOC[aK]->mLowCode);
       }
       SaveInFile(aBE,"SpecEncoding.xml");
   }

   if (mShow)
      Show();

   delete mCEC;


   return EXIT_SUCCESS;
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

