#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
#include "include/MMVII_util_tpl.h"

//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{


class cHistoCarNDim  : public cMemCheck
{
    public :
       
       typedef tINT4                        tDataNd;
       typedef cDataGenDimTypedIm<tDataNd>  tHistND;
       typedef cHistoCumul<tINT4,tREAL8>    tHisto1;
       typedef cDenseVect<tINT4>            tIndex;
        
       cHistoCarNDim(int aSzH,const tVecCar &,const cStatAllVecCarac &);
       void  ComputePts(const cVecCaracMatch &);
       void  Add(const cVecCaracMatch &,bool isH0);
       void  Show() const;
       double CarSep() const;
    private :
       cHistoCarNDim(const cHistoCarNDim &) = delete;

       int                       mDim;
       tIndex                    mSz;
       tVecCar                   mVCar;
       tIndex                    mPts;
       
       std::vector<const tHisto1*>     mHd1_0;
       tHistND                   mHist0;  // Homolog
       tHistND                   mHist2;  //  Non (or close) Homolog
       std::string               mName;
};

cHistoCarNDim::cHistoCarNDim(int aSzH,const tVecCar & aVCar,const cStatAllVecCarac & aStat) :
   mDim   (aVCar.size()),
   mSz    (tIndex::Cste(mDim,aSzH)),
   mVCar  (aVCar),
   mPts   (mDim),
   mHd1_0 (),
   mHist0 (mSz),
   mHist2 (mSz),
   mName  (NameVecCar(aVCar))
{
   for (const auto & aLabel : aVCar)
      mHd1_0.push_back(&(aStat.OneStat(aLabel).Hist(0)));

}

double cHistoCarNDim::CarSep() const
{
   cComputeSeparDist aCSD;
   tDataNd * aDH0 = mHist0.RawDataLin();
   tDataNd * aDH2 = mHist2.RawDataLin();
   for (int aK=0 ; aK<mHist0.NbElem() ; aK++)
       aCSD.AddPops(aDH0[aK],aDH2[aK]);

   return aCSD.Sep();
}

/*
void TEST(const tIndex & aSz,const tVecCar & aVCar,const cStatAllVecCarac & aStat)
{
StdOut() << "aaaaa  " << __LINE__ << "\n";
mSz.Sz();
StdOut() << "aaaaa  " << __LINE__ << "\n";
aSz.Dup();
}
*/

void  cHistoCarNDim::Show() const
{
   StdOut() << "Name = " << mName  << " " << CarSep() << "\n";
}


void cHistoCarNDim::ComputePts(const cVecCaracMatch & aVCM)
{
     aVCM.FillVect(mPts,mVCar);

     for (int aK=0 ; aK<mDim ; aK++)
     {
         int aSzK = mSz(aK);
         int  aV = round_down(mHd1_0[aK]->PropCumul(mPts(aK)) * aSzK);

         mPts(aK) = std::min(aV,aSzK-1);
     }
}

void cHistoCarNDim::Add(const cVecCaracMatch & aVCM,bool isH0)
{
    ComputePts(aVCM);
    tHistND & aHist = (isH0  ? mHist0 : mHist2);
    aHist.AddV(mPts,1);
}



class cAppliCalcHistoNDim : public cMMVII_Appli,
                            public cNameFormatTDEDM
{
     public :
        cAppliCalcHistoNDim(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);


     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
        void AddHistoOneFile(const std::string &);


        // -- Mandatory args ----
        std::string              mPatHom0;
        std::string              mNameInput;
        std::string              mNameOutput;
        std::vector<std::string> mPatsCar;

         // -- Optionnal args ----
         int                     mSzH1;
         bool                    mAddPrefixSeq;
             //  std::string       mPatShowSep;
             //  bool              mWithCr;

         // -- Internal variables ----
        int                          mDim;
        int                          mMaxSzH;
        cStatAllVecCarac             mStats;
        std::vector<cHistoCarNDim*>  mVHistN;
};


cAppliCalcHistoNDim::cAppliCalcHistoNDim(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mAddPrefixSeq (false),
   mMaxSzH       (50),
   mStats        (false)
{
}


cCollecSpecArg2007 & cAppliCalcHistoNDim::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mPatHom0,"Name of input(s) file(s)",{{eTA2007::MPatFile,"0"}})
          <<   Arg2007(mNameInput,"Name used for input ")
          <<   Arg2007(mNameOutput,"Name used for output ")
          <<   Arg2007(mPatsCar,"vector of pattern of carac")
   ;
}

cCollecSpecArg2007 & cAppliCalcHistoNDim::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mSzH1, "SzH","Size of histogramm in each dim")
          << AOpt2007(mAddPrefixSeq, "APS","Add Prefix subseq of carac(i.e abc=>a,ab,abc)",{eTA2007::HDV})
   ;
}

void cAppliCalcHistoNDim::AddHistoOneFile(const std::string & aStr0)
{
    for (const int & aNumH : {0,2})
    {
        std::string aStr = HomFromHom0(aStr0,aNumH);
        StdOut() << "      ****** "   << aStr  << "   *******\n";
        cFileVecCaracMatch aFCV(aStr);
        for (const auto & aPtrH :  mVHistN)
        {
            for (const auto & aVCM : aFCV.VVCM())
            {
                aPtrH->Add(aVCM,aNumH==0);
            }
        }
    }
    for (const auto & aPtrH :  mVHistN)
    {
        aPtrH->Show();
    }
    StdOut() << "-------------------------------------------------\n";

}


int  cAppliCalcHistoNDim::Exe()
{
   cNameFormatTDEDM::SetNamesProject(mNameInput,mNameOutput);
   mDim = mPatsCar.size();

   ReadFromFile(mStats,FileHisto1Carac(true));



   //  ------------------------------------------------------------------------------
   //   1 ----------  Compute the sequence of caracteristique matched by pattern ----
   //  ------------------------------------------------------------------------------

          //  --- 1.1  compute rough cartersian product
   std::vector<tVecCar> aVPatExt;
   for (const auto & aPat: mPatsCar)
   {
        aVPatExt.push_back(SubOfPat<eModeCaracMatch>(aPat,false));
   }
   
   std::vector<tVecCar>  aVSeqCarInit = ProdCart(aVPatExt);

   if (mAddPrefixSeq)
   {
       std::vector<tVecCar> aVSeqSub;
       for (const auto & aSeq : aVSeqCarInit)
       {
           for (int aK=1 ; aK<=int(aSeq.size()) ; aK++)
           {
                aVSeqSub.push_back(tVecCar(aSeq.begin(),aSeq.begin()+aK));
           }
       }
       aVSeqCarInit  = aVSeqSub;
   }
   std::map<tVecCar,tVecCar>  aMapSeq;  // Used to select unique sequence on set criteria but maintain initial order

          //  --- 1.2  order is not important in the sequence
   for (const auto & aSeq : aVSeqCarInit)
   {
       tVecCar aSeqId = aSeq;
       std::sort ( aSeqId.begin(), aSeqId.end());
       // Supress the possible duplicate carac
       aSeqId.erase( unique( aSeqId.begin(), aSeqId.end() ), aSeqId.end() );
       // For now, supress those where there were duplicates
       if (aSeqId.size() == aSeq.size())
       {
          aMapSeq[aSeqId] = aSeq;
       }
   }


   //  ------------------------------------------------------------------------------
   //   2 ----------  Initialise the N Dimensionnal histogramm  ---------------------
   //  ------------------------------------------------------------------------------

   if (! IsInit(&mSzH1))
   {
       double aNbCell = 5e8 / aMapSeq.size();
       mSzH1 = round_down(pow(aNbCell,1/double(mDim)));

       mSzH1 = std::min(mSzH1,mMaxSzH);
   }

   for (const auto & aPair : aMapSeq)
   {
       mVHistN.push_back(new cHistoCarNDim(mSzH1,aPair.second,mStats));
   }

   if (1)
   {
      for (const auto & aPair : aMapSeq)
      {
          StdOut() << "* " << NameVecCar(aPair.second) << " <= Id:" << NameVecCar(aPair.first) << "\n";
      }
      StdOut() << "NB SEQ=" << aMapSeq.size() << " NBLAB=" << (int) eModeCaracMatch::eNbVals << "\n";
      StdOut() << "SzH=" << mSzH1 <<  " NbSeq= " <<mVHistN.size() << "\n";

      getchar();
   }

   //  ------------------------------------------------------------------------------
   //   4 ----------  Parse the files of veccar      --------------------------------
   //  ------------------------------------------------------------------------------

   for (const auto & aStr : ToVect(MainSet0()))
   {
         AddHistoOneFile(aStr);
   }

   //  ------------------------------------------------------------------------------
   //   5 ----------   Save the results     --------------------------------
   //  ------------------------------------------------------------------------------

   //  ------------------------------------------------------------------------------
   //   ZZZZ ----------------  Free and quit ----------------------------------------
   //  ------------------------------------------------------------------------------
   DeleteAllAndClear(mVHistN);
   return EXIT_SUCCESS;
}




/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CalcHistoNDim(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCalcHistoNDim(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCalcHistoNDim
(
     "DM3CalcHistoNDim",
      Alloc_CalcHistoNDim,
      "Compute and save histogramm on multiple caracteristics",
      {eApF::Match},
      {eApDT::FileSys},
      {eApDT::FileSys},
      __FILE__
);
/*
*/



};
