#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
#include "include/MMVII_util_tpl.h"

//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

void AddData(const cAuxAr2007 & anAux,eModeCaracMatch & aMCM)  {EnumAddData(anAux,aMCM,"ModeCar");}


/* ************************************************ */
/*                                                  */
/*             cHistoCarNDim                        */
/*                                                  */
/* ************************************************ */

class cHistoCarNDim  : public cMemCheck
{
    public :
       
       typedef tINT4                        tDataNd;
       typedef cDataGenDimTypedIm<tDataNd>  tHistND;
       typedef cHistoCumul<tINT4,tREAL8>    tHisto1;
       typedef cDenseVect<tINT4>            tIndex;
        
       cHistoCarNDim(int aSzH,const tVecCar &,const cStatAllVecCarac &);
       cHistoCarNDim();  // Used for AddData requiring default cstrc
       cHistoCarNDim(const std::string&);  // Used for AddData requiring default cstrc
       ~cHistoCarNDim();  // Used for AddData requiring default cstrc
       void  Add(const cVecCaracMatch &,bool isH0);
       void  Show(cMultipleOfs &,bool WithCr) const;
       double CarSep() const;
       void AddData(const cAuxAr2007 &);
       const std::string & Name() const;

       double ScoreCr(const cVecCaracMatch &) const;
       void   UpDateCr(const cVecCaracMatch & aHom,const cVecCaracMatch & aNotHom);
    private :
       void  ComputePts(const cVecCaracMatch &) const;
       cHistoCarNDim(const cHistoCarNDim &) = delete;

       bool                      mIP;
       int                       mDim;
       tIndex                    mSz;
       tVecCar                   mVCar;
       mutable tIndex                    mPts;
       
       std::vector<const tHisto1*>     mHd1_0;
       tHistND                   mHist0;  // Homolog
       tHistND                   mHist2;  //  Non (or close) Homolog
       std::string               mName;
       double                    mNbOk;
       double                    mNbNotOk;
};


void cHistoCarNDim::AddData(const cAuxAr2007 & anAux)
{
     // eModeCaracMatch aMC;
     // F2(aMC);
     MMVII::AddData(cAuxAr2007("Car",anAux),mVCar);
     MMVII::AddData(cAuxAr2007("Sz",anAux),mSz);
     MMVII::AddData(cAuxAr2007("Hd1",anAux),mHd1_0);
     MMVII::AddData(cAuxAr2007("HN0",anAux),mHist0);
     MMVII::AddData(cAuxAr2007("HN0",anAux),mHist2);
     if (anAux.Input())
     {
         mName = NameVecCar(mVCar);
         mSz = mHist0.Sz().Dup();
         mDim = mSz.Sz();
         mPts = tIndex(mDim);
     }
}

void AddData(const cAuxAr2007 & anAux,cHistoCarNDim & aHND)
{
   aHND.AddData(anAux);
}

cHistoCarNDim::cHistoCarNDim()  :
   mIP      (false),
   mDim     (1),
   mSz      (tIndex::Cste(mDim,1)),
   mVCar    (),
   mPts     (1),
   mHd1_0   (),
   mHist0   (mSz),
   mHist2   (mSz),
   mName    (""),
   mNbOk    (0.0),
   mNbNotOk (0.0)
{
}

cHistoCarNDim::cHistoCarNDim(const std::string & aNameFile) :
   cHistoCarNDim ()
{
   ReadFromFile(*this,aNameFile);
}

cHistoCarNDim::~cHistoCarNDim()  
{
   if (! mIP)
       DeleteAllAndClear(mHd1_0);
}


cHistoCarNDim::cHistoCarNDim(int aSzH,const tVecCar & aVCar,const cStatAllVecCarac & aStat) :
   mIP      (true),
   mDim     (aVCar.size()),
   mSz      (tIndex::Cste(mDim,aSzH)),
   mVCar    (aVCar),
   mPts     (mDim),
   mHd1_0   (),
   mHist0   (mSz),
   mHist2   (mSz),
   mName    (NameVecCar(aVCar)),
   mNbOk    (0.0),
   mNbNotOk (0.0)
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


void  cHistoCarNDim::Show(cMultipleOfs & anOfs,bool WithCr) const
{
    anOfs << "Name = " << mName  << " Sep: " << CarSep() ;
    if (WithCr)
         anOfs  << " Cr: " <<  mNbOk / (mNbOk+mNbNotOk);
    anOfs << "\n";
}


void cHistoCarNDim::ComputePts(const cVecCaracMatch & aVCM) const
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

double  cHistoCarNDim::ScoreCr(const cVecCaracMatch & aVCM) const
{
    ComputePts(aVCM);
    tDataNd aV0 = mHist0.GetV(mPts);
    tDataNd aV2 = mHist2.GetV(mPts);

    if ((aV0==0) && (aV2==0)) 
       return 0.5;

    return aV0 / double(aV0+aV2);
}

void   cHistoCarNDim::UpDateCr(const cVecCaracMatch & aHom,const cVecCaracMatch & aNotHom)
{
    double aScH  = ScoreCr(aHom);
    double aScNH = ScoreCr(aNotHom);
    if (aScH<aScNH)
       mNbOk++;
    else if (aScH>aScNH)
       mNbNotOk++;
    else
    {
       mNbOk += 0.5;
       mNbNotOk += 0.5;
    }
}

const std::string & cHistoCarNDim::Name() const
{
    return mName;
}

/* ************************************************ */
/*                                                  */
/*          cAppliCalcHistoNDim                     */
/*                                                  */
/* ************************************************ */

class cAppliCalcHistoNDim : public cAppliLearningMatch
{
     public :
        cAppliCalcHistoNDim(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);


     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
        void AddHistoOneFile(const std::string &,int aKFile,int aNbFile);
        void  ShowCarSep(cMultipleOfs &) const;

        std::string NameSaveH(const cHistoCarNDim & aHND,bool isInput)
        {
           return FileHistoNDIm(aHND.Name(),isInput);
        }


        // -- Mandatory args ----
        std::string              mPatHom0;
        std::string              mNameInput;
        std::string              mNameOutput;
        std::vector<std::string> mPatsCar;

         // -- Optionnal args ----
         int                     mSzH1;
         bool                    mAddPrefixSeq;
         bool                    mInitialProcess; // Post Processing
         bool                    mCloseH;
             //  std::string       mPatShowSep;
             //  bool              mWithCr;

         // -- Internal variables ----
        int                          mDim;
        int                          mMaxSzH;
        cStatAllVecCarac             mStats;
        std::vector<cHistoCarNDim*>  mVHistN;
};


cAppliCalcHistoNDim::cAppliCalcHistoNDim(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mAddPrefixSeq        (false),
   mInitialProcess      (true),
   mCloseH              (false),
   mMaxSzH              (50),
   mStats               (false)
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
          << AOpt2007(mInitialProcess, "IP","Initial processing. If not , read previous histo, more stat, post proc",{eTA2007::HDV})
          << AOpt2007(mCloseH, "CloseH","Use close homologs, instead of random",{eTA2007::HDV})
   ;
}

void cAppliCalcHistoNDim::AddHistoOneFile(const std::string & aStr0,int aKFile,int aNbFile)
{
    StdOut() << "      ####### "   << aStr0  << " : " << aKFile << "/" << aNbFile << "   ####### \n";
    int aMulH = (mCloseH ? 1 : 2);
    if (mInitialProcess)
    {
       for (const int & aKNumH : {0,1})
       {
           std::string aStr = HomFromHom0(aStr0,aKNumH*aMulH);
           cFileVecCaracMatch aFCV(aStr);
           for (const auto & aPtrH :  mVHistN)
           {
               for (const auto & aVCM : aFCV.VVCM())
               {
                   aPtrH->Add(aVCM,aKNumH==0);
               }
           }
       }
    }
    else
    {
      
       cFileVecCaracMatch aFCV0(HomFromHom0(aStr0,0));
       cFileVecCaracMatch aFCV2(HomFromHom0(aStr0,aMulH));
       const std::vector<cVecCaracMatch> &  aV0 = aFCV0.VVCM();
       const std::vector<cVecCaracMatch> &  aV2 = aFCV2.VVCM();
       MMVII_INTERNAL_ASSERT_strong(aV0.size()==aV2.size(),"Diff size in Hom/NonHom");
       for (auto & aPtrH :  mVHistN)
       {
           for (int aK=0 ; aK<int(aV0.size()) ; aK++)
           {
              aPtrH->UpDateCr(aV0[aK],aV2[aK]);
           }
       }
    }
}
void cAppliCalcHistoNDim::ShowCarSep(cMultipleOfs & anOfs) const
{
    for (const auto & aPtrH :  mVHistN)
    {
        aPtrH->Show(anOfs,!mInitialProcess);
    }
    anOfs << "-------------------------------------------------\n";
}



int  cAppliCalcHistoNDim::Exe()
{
   SetNamesProject(mNameInput,mNameOutput);
   mDim = mPatsCar.size();

   if ( mInitialProcess)
   {
      ReadFromFile(mStats,FileHisto1Carac(true));
   }



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

   if (mInitialProcess)
   {
       if (! IsInit(&mSzH1))
       {
           double aNbCell = 5e8 / aMapSeq.size();
           mSzH1 = round_down(pow(aNbCell,1/double(mDim)));

           mSzH1 = std::min(mSzH1,mMaxSzH);
       }
   }
   else
   {
      mSzH1 = 1; // put low value it will be reallocated after read
   }

   for (const auto & aPair : aMapSeq)
   {
       if ( mInitialProcess)
       {
          mVHistN.push_back(new cHistoCarNDim(mSzH1,aPair.second,mStats));
       }
       else
       { 
           std::string aNameFile = FileHistoNDIm(NameVecCar(aPair.second),true);
           mVHistN.push_back(new cHistoCarNDim(aNameFile));
       }
   }

   if (1)
   {
      cMultipleOfs  aMulOfs(NameReport(),false);
      aMulOfs << "COM=[" << Command() << "]\n\n";
      for (bool InReport : {false,true})
      {
          cMultipleOfs &  aOfs = InReport ? aMulOfs : StdOut() ;
          for (const auto & aPair : aMapSeq)
          {
              aOfs << "* " << NameVecCar(aPair.second) << " <= Id:" << NameVecCar(aPair.first) << "\n";
          }
          aOfs << "NB SEQ=" << aMapSeq.size() << " NBLAB=" << (int) eModeCaracMatch::eNbVals << "\n";
          aOfs << "SzH=" << mSzH1 <<  " NbSeq= " << mVHistN.size() << "\n";
      }

      // getchar();
   }

   //  ------------------------------------------------------------------------------
   //   4 ----------  Parse the files of veccar      --------------------------------
   //  ------------------------------------------------------------------------------

   int aKFile =0 ;
   int aNbFile = MainSet0().size();
   for (const auto & aStr : ToVect(MainSet0()))
   {
         AddHistoOneFile(aStr,aKFile,aNbFile);
         ShowCarSep(StdOut());
         aKFile++;
   }

   //  ------------------------------------------------------------------------------
   //   5 ----------   Save the results     --------------------------------
   //  ------------------------------------------------------------------------------

   cMultipleOfs  aMulOfs(NameReport(),true);
   aMulOfs << "\n========================================\n\n";
   ShowCarSep(aMulOfs);

   if (mInitialProcess)
   {
       for (const auto & aPtrH :  mVHistN)
       {
           // std::string aNameSave = FileHistoNDIm(aPtrH->Name(),false);
           SaveInFile(*aPtrH,NameSaveH(*aPtrH,false));
       }
   }

   

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
