
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
#include "include/MMVII_util_tpl.h"
#include "include/MMVII_Tpl_Images.h"

//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

void AddData(const cAuxAr2007 & anAux,eModeCaracMatch & aMCM)  {EnumAddData(anAux,aMCM,"ModeCar");}


/* ************************************************ */
/*                                                  */
/*             cHistoCarNDim                        */
/*                                                  */
/* ************************************************ */



void cHistoCarNDim::AddData(const cAuxAr2007 & anAux)
{
     // eModeCaracMatch aMC;
     // F2(aMC);
     MMVII::AddData(cAuxAr2007("Car",anAux),mVCar);
     MMVII::AddData(cAuxAr2007("Sz",anAux),mSz);
     MMVII::AddData(cAuxAr2007("Hd1",anAux),mHd1_0);
     MMVII::AddData(cAuxAr2007("HN0",anAux),mHist0);
     MMVII::AddData(cAuxAr2007("HN2",anAux),mHist2);
     if (anAux.Input())
     {
         mName = NameVecCar(mVCar);
         mSz = mHist0.Sz().Dup();
         mDim = mSz.Sz();
         mPts = tIndex(mDim);
         mPtsInit = tIndex(mDim);
         mRPts = tRIndex(mDim);
     }
}

void AddData(const cAuxAr2007 & anAux,cHistoCarNDim & aHND)
{
   aHND.AddData(anAux);
}

cHistoCarNDim::cHistoCarNDim()  :
   mIP        (false),
   mDim       (1),
   mSz        (tIndex::Cste(mDim,1)),
   mVCar      (),
   mPts       (1),
   mPtsInit   (1),
   mRPts      (1),
   mHd1_0     (),
   mHist0     (mSz),
   mHist2     (mSz),
   mName      (""),
   mNbOk      (0.0),
   mNbNotOk   (0.0),
   mGV2I      (false),
   mHistoI0   (cPt2di(1,1)),
   mHistoI2   (cPt2di(1,1))
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


cHistoCarNDim::cHistoCarNDim(int aSzH,const tVecCar & aVCar,const cStatAllVecCarac & aStat,bool genVis2DI) :
   mIP      (true),
   mDim     (aVCar.size()),
   mSz      (tIndex::Cste(mDim,aSzH)),
   mVCar    (aVCar),
   mPts     (mDim),
   mPtsInit (mDim),
   mRPts    (mDim),
   mHd1_0   (),
   mHist0   (mSz),
   mHist2   (mSz),
   mName    (NameVecCar(aVCar)),
   mNbOk    (0.0),
   mNbNotOk (0.0),
   mGV2I    (genVis2DI && (mDim==2)),
   mHistoI0 (cPt2di(1,1) * (mGV2I  ? cVecCaracMatch::TheDyn4Visu : 1),nullptr,eModeInitImage::eMIA_Null),
   mHistoI2 (cPt2di(1,1) * (mGV2I  ? cVecCaracMatch::TheDyn4Visu : 1),nullptr,eModeInitImage::eMIA_Null)
{
    for (const auto & aLabel : aVCar)
        mHd1_0.push_back(&(aStat.OneStat(aLabel).HistSom(7)));
        //mHd1_0.push_back(&(aStat.OneStat(aLabel).Hist(0)));
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

double cHistoCarNDim::Correctness() const
{
    return mNbOk / (mNbOk+mNbNotOk);
}


void  cHistoCarNDim::Show(cMultipleOfs & anOfs,bool WithCr) const
{
    anOfs << "Name = " << mName  << " Sep: " << CarSep() ;
    if (WithCr)
         anOfs  << " UnCorec: " <<  1-Correctness();
    anOfs << "\n";
}


void cHistoCarNDim::ComputePts(const cVecCaracMatch & aVCM) const
{
     aVCM.FillVect(mPts,mVCar);
     if (mGV2I)
        mPtsInit = mPts.Dup();

     for (int aK=0 ; aK<mDim ; aK++)
     {
         int aSzK = mSz(aK);
         double  aV = mHd1_0[aK]->PropCumul(mPts(aK)) * aSzK;
         mPts(aK) = std::min(round_down(aV),aSzK-1);
         mRPts(aK) = std::min(aV,aSzK-1.0001);
	 /*
         int  aV = round_down(mHd1_0[aK]->PropCumul(mPts(aK)) * aSzK);
         mPts(aK) = std::min(aV,aSzK-1);
	 */
     }

}


void cHistoCarNDim::Add(const cVecCaracMatch & aVCM,bool isH0)
{
    ComputePts(aVCM);
    tHistND & aHist = (isH0  ? mHist0 : mHist2);
    aHist.AddV(mPts,1);

    if (mGV2I)
    {
       cPt2di aPt = cVecCaracMatch::ToVisu(cPt2di::FromVect(mPtsInit));
       if (isH0)
           mHistoI0.DIm().AddVal(aPt,1);
       else
           mHistoI2.DIm().AddVal(aPt,1);
    }
}

template <class Type>  double ScoreHnH(const Type & aV0,const Type & aV2)
{
    if ((aV0==0) && (aV2==0)) 
       return 0.5;
    return aV0 / double(aV0+aV2);
}

//  tREAL4   GetNLinearVal(const tRIndex&) const; // Get value by N-Linear interpolation
//  void     AddNLinearVal(const tRIndex&,const double & aVal) ; // Get value by N-Linear interpolation

double  cHistoCarNDim::HomologyLikelihood(const cVecCaracMatch & aVCM,bool  Interpol) const
{
    ComputePts(aVCM);
    double aV0 = Interpol  ?  mHist0.GetNLinearVal(mRPts) : mHist0.GetV(mPts);
    double aV2 = Interpol  ?  mHist2.GetNLinearVal(mRPts) : mHist2.GetV(mPts);
    return ScoreHnH(aV0,aV2);
    // return ScoreHnH(mHist0.GetV(mPts),mHist2.GetV(mPts));
}




void   cHistoCarNDim::UpDateCorrectness(const cVecCaracMatch & aHom,const cVecCaracMatch & aNotHom)
{
    double aScH  = HomologyLikelihood(aHom,false);
    double aScNH = HomologyLikelihood(aNotHom,false);
    if (aScH<aScNH)
       mNbNotOk++;
    else if (aScH>aScNH)
       mNbOk++;
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

void  cHistoCarNDim::GenerateVisuOneIm(const std::string & aDir,const std::string & aPrefix,const tHistND & aHist)
{
     // make a float image, of the int image, for Vino ...
     cIm2D<float> anIm = Convert((float*)nullptr,aHist.ToIm2D().DIm());
     
     std::string aName = aDir + "Histo-" + aPrefix +  mName + ".tif";
     anIm.DIm().ToFile(aName);
}

void  cHistoCarNDim::GenerateVisu(const std::string & aDir)
{
  if (mDim==2)
  {
      GenerateVisuOneIm(aDir,"Hom",mHist0);
      GenerateVisuOneIm(aDir,"NonH",mHist2);

      cIm2D<tDataNd> aI0 = mHist0.ToIm2D();
      cIm2D<tDataNd> aI2 = mHist2.ToIm2D();
      cIm2D<tREAL4>  aImSc(aI0.DIm().Sz());
      cIm2D<tREAL4>  aImPop(aI0.DIm().Sz());
      for (const auto & aP : aImSc.DIm())
      {
          tINT4 aV0 = aI0.DIm().GetV(aP);
          tINT4 aV2 = aI2.DIm().GetV(aP);
          aImSc.DIm().SetV(aP,ScoreHnH(aV0,aV2));
          aImPop.DIm().SetV(aP,aV0+aV2);
      }
      std::string aName = aDir + "HistoScore-"+  mName + ".tif";
      aImSc.DIm().ToFile(aName);
      aName = aDir + "HistoPop-"+  mName + ".tif";
      aImPop.DIm().ToFile(aName);
  }
}

void  cHistoCarNDim::GenerateVis2DInitOneInit(const std::string & aDir,const std::string & aPrefix,cIm2D<double> aH,const tHistND& aHN)
{
     std::string aName = aDir + "HistoInit-" + aPrefix +  mName + ".tif";
     aH.DIm().ToFile(aName);

/*
     int aTx = mSz(0);
     int aTy = mSz(1);
     cIm1D<double> aH1D(aTy,nullptr,eModeInitImage::eMIA_Null);
     double aSomV=0.0;
     for (int aX=0 ; aX<aTx ;aX++)
     {
         for (int aY=0 ; aY<aTy ;aY++)
         {
             cPt2di aP(aX,aY);
             double aV = aHN.GetV(aP.ToVect());
             aH1D.DIm().AddV(aY,aV);
             aSomV += aV;
         }
     }
     for (int aY=0 ; aY<aTy ;aY++)
     {
         StdOut() << "Y=" << aY << " V=" << aH1D.DIm().GetV(aY) * (aTy/aSomV) << "\n";
     }
     getchar();
*/
}

void  cHistoCarNDim::GenerateVis2DInit(const std::string & aDir)
{
     if (mGV2I)
     {
         GenerateVis2DInitOneInit(aDir,"Hom" ,mHistoI0,mHist0);
         GenerateVis2DInitOneInit(aDir,"NonH",mHistoI2,mHist2);
     }
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
	std::vector<std::string>  Samples() const  override;

        void AddHistoOneFile(const std::string &,int aKFile,int aNbFile);
	// print performance : Separibility + (If computed) Correctness 
        void  ShowPerf(cMultipleOfs &) const;



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
	 std::string             mSerialSeparator;
         bool                    mAddPrefixSeq;
         bool                    mInitialProcess; // Post Processing
         bool                    mCloseH;
         bool                    mGenerateVisu; // Do we generate visual (Image for 2d, later ply for 3d)
         bool                    mGenVis2DInit; // Do we generate visual init (w/o equal)
	 bool                    mOkDuplicata;  // Do accept duplicata in sequence,
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
   mGenerateVisu        (false),
   mGenVis2DInit        (false),
   mOkDuplicata         (false),
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
          << AOpt2007(mSerialSeparator, "SerSep","Serial Separator, if no pattern, e.q [a@b,c@d@e] with \"@ \"")
          << AOpt2007(mAddPrefixSeq, "APS","Add Prefix subseq of carac(i.e abc=>a,ab,abc)",{eTA2007::HDV})
          << AOpt2007(mInitialProcess, "IP","Initial processing. If not , read previous histo, more stat, post proc",{eTA2007::HDV})
          << AOpt2007(mCloseH, "CloseH","Use close homologs, instead of random",{eTA2007::HDV})
          << AOpt2007(mGenerateVisu, "GeneVisu","Generate visualisation",{eTA2007::HDV})
          << AOpt2007(mGenVis2DInit, "GV2DI","Generate vis 2D w/o equalization",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mOkDuplicata, "OkDupl","Accept duplicatas in sequences",{eTA2007::HDV,eTA2007::Tuning})
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
              aPtrH->UpDateCorrectness(aV0[aK],aV2[aK]);
           }
       }
    }
}


void cAppliCalcHistoNDim::ShowPerf(cMultipleOfs & anOfs) const
{
    std::vector<cHistoCarNDim*>  aVHSort = mVHistN;

    if (mInitialProcess)
        SortOnCriteria(aVHSort,[](const auto & aPtr){return aPtr->CarSep();});
    else
        SortOnCriteria(aVHSort,[](const auto & aPtr){return 1-aPtr->Correctness();});

    for (const auto & aPtrH :  aVHSort)
    {
        aPtrH->Show(anOfs,!mInitialProcess);
    }
    anOfs << "-------------------------------------------------\n";
}

std::vector<std::string>  cAppliCalcHistoNDim::Samples() const
{
    return std::vector<std::string>
           (
               {
                   "MMVII DM3CalcHistoNDim \".*\"  Test Test [xx] # Generate enum values",
                   "MMVII DM3CalcHistoNDim DMTrain.*LDHAime0.dmp  AllMDLB2014 AllMDLB2014 [.*]"
               }
          );

}



int  cAppliCalcHistoNDim::Exe()
{
   if (mGenerateVisu)
   {
      if (IsInit(&mInitialProcess)) { MMVII_INTERNAL_ASSERT_strong(!mInitialProcess,"IP Specified with Visu ..."); }
      else { mInitialProcess = false; }
   }

   if (mGenVis2DInit)
   {
      if (IsInit(&mInitialProcess)) { MMVII_INTERNAL_ASSERT_strong(mInitialProcess,"IP false with Vis2DI"); }
      else { mInitialProcess = true; }
   }


   SetNamesProject(mNameInput,mNameOutput);
   mDim = mPatsCar.size();

   if ( mInitialProcess)
   {
      ReadFromFile(mStats,FileHisto1Carac(true));
   }



   //  ------------------------------------------------------------------------------
   //   1 ----------  Compute the sequence of caracteristique matched by pattern ----
   //  ------------------------------------------------------------------------------
   std::map<tVecCar,tVecCar>  aMapSeq;  // Used to select unique sequence on set criteria but maintain initial order
   if (! IsInit(&mSerialSeparator))
   {
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

          //  --- 1.2  order is not important in the sequence
      for (const auto & aSeq : aVSeqCarInit)
      {
          tVecCar aSeqId = aSeq;
          std::sort ( aSeqId.begin(), aSeqId.end());
       // Supress the possible duplicate carac, can be maintained for mNoDuplicata
          if (! mOkDuplicata)
          {
              aSeqId.erase( unique( aSeqId.begin(), aSeqId.end() ), aSeqId.end() );
          }
       // For now, supress those where there were duplicates
          if (aSeqId.size() == aSeq.size())
          {
             aMapSeq[aSeqId] = aSeq;
          }
      }
   }
   else
   {
      for (const auto & aPat: mPatsCar)
      {
          tVecCar aVCar;
	  for (const auto & aStr : SplitString(aPat,mSerialSeparator))
	  {
              aVCar.push_back(Str2E<eModeCaracMatch>(aStr));
              StdOut() << aStr << "///";
	  }
	  StdOut() << "\n";
          aMapSeq[aVCar] = aVCar;
      }
      // BREAK_POINT("Test Separe");
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
          mVHistN.push_back(new cHistoCarNDim(mSzH1,aPair.second,mStats,mGenVis2DInit));
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
      aMulOfs << "COM=[" << CommandOfMain() << "]\n\n";
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

   if (mGenerateVisu)
   {
      for (const auto & aPtrH :  mVHistN)
      {
            aPtrH->GenerateVisu(DirVisu());
      }
   }
   else
   {
       int aKFile =0 ;
       int aNbFile = MainSet0().size();
       for (const auto & aStr : ToVect(MainSet0()))
       {
             AddHistoOneFile(aStr,aKFile,aNbFile);
             ShowPerf(StdOut());
             aKFile++;
       }
   }

   //  ------------------------------------------------------------------------------
   //   5 ----------   Save the results     --------------------------------
   //  ------------------------------------------------------------------------------

   if (! mGenerateVisu)
   {
       cMultipleOfs  aMulOfs(NameReport(),true);
       aMulOfs << "\n========================================\n\n";
       ShowPerf(aMulOfs);

       if (mInitialProcess)
       {
           for (const auto & aPtrH :  mVHistN)
           {
               // std::string aNameSave = FileHistoNDIm(aPtrH->Name(),false);
               SaveInFile(*aPtrH,NameSaveH(*aPtrH,false));
               aPtrH->GenerateVis2DInit(DirVisu());
           }
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
