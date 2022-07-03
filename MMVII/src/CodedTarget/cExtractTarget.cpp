#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


// Test git branch

namespace MMVII
{

namespace  cNS_CodedTarget
{

/*  *********************************************************** */
/*                                                              */
/*                       cDCT                                   */
/*                                                              */
/*  *********************************************************** */


cDCT::cDCT(const cPt2di aPt,cAffineExtremum<tREAL4> & anAffEx) :
   mGT    (nullptr),
   mPix0  (aPt),
   mPt    (anAffEx.StdIter(ToR(aPt),1e-2,3)),
   mState (eResDCT::Ok),
   mSym   (-1),
   mBin   (-1),
   mRad   (-1)
{
    if ( (anAffEx.Im().Interiority(Pix())<20) || (Norm2(mPt-ToR(aPt))>2.0)  )  
       mState = eResDCT::Divg;
}


/*  *********************************************************** */
/*                                                              */
/*             cAppliExtractCodeTarget                          */
/*                                                              */
/*  *********************************************************** */

class cAppliExtractCodeTarget : public cMMVII_Appli,
	                        public cAppliParseBoxIm<tREAL4>
{
     public :
        cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	///  Create the matching between GT and extracted
        void MatchOnGT(cGeomSimDCT & aGSD);
	/// compute direction of ellipses
        void ExtractDir(cDCT & aDCT);
	/// Print statistique initial 
        void ShowStatGTInit();

	int ExeOnParsedBox() override;

	void TestFilters();
	void DoExtract();
        void ShowStats(const std::string & aMes) ;
        void MarkDCT() ;
        void SelectOnFilter(cFilterDCT<tREAL4> * aFilter,bool MinCrown,double aThrS,eResDCT aModeSup);

	std::string mNameTarget;

	cParamCodedTarget        mPCT;
	cPt2dr                   mRaysTF;
        std::vector<eDCTFilters> mTestedFilters;    

        cImGrad<tREAL4>  mImGrad;  ///< Result of gradient
        double   mR0Sym;           ///< R min for first very quick selection on symetry
        double   mR1Sym;           ///< R max for first very quick selection on symetry
        double   mRExtreSym;       ///< R to compute indice of local maximal of symetry
        double   mTHRS_Sym;        ///< Threshold for symetricity


        double mTHRS_Bin;

        std::vector<cDCT*>  mVDCT; ///< vector of detected target
	cResSimul           mRSim; ///< result of simulation when exist


        cRGBImage      mImVisu;
        std::string    mPatF;
        bool           mWithGT;
        double         mDMaxMatch;
         
};

/*
void cAppliExtractCodeTarget::ExtractDir(cDCT & aDCT)
{
     // tDataIm &  aDIm = APBI_DIm();
     int aNbDir = 100;
     double aStep = 0.25;
     double aRay =  8.0;
     double  aR2Max = Square(aRay);
     double  aR2Min = Square(2.0);

     int iRay = round_up(aRay);
     for (int aKx=-iRay ; aKx<=iRay ; aKx++)
     {
         for (int aKy=-iRay ; aKy<=iRay ; aKy++)
         {
		 cPt2di  aVois(aKx,aKy);
         }
     }


     int aNb = aRay/aStep;

     for (int aKx=-aNb ; aKx<=aNb ; aKx++)
     {
          for (int aKy=-aNb ; aKy<=aNb ; aKy++)
	  {
	       cPt2dr aVois(aKx*aStep,aKy*aStep);
	       double aR2 = SqN2(aVois);
	       if ((aR2>aR2Min) && (aR2<aR2Max))
	       {
	       }
	  }
     }
}
*/


/* *************************************************** */
/*                                                     */
/*              cAppliExtractCodeTarget                */
/*                                                     */
/* *************************************************** */

cAppliExtractCodeTarget::cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(5000,5000),cPt2di(300,300),false), // static_cast<cMMVII_Appli & >(*this))
   mRaysTF        ({4,8}),
   mImGrad        (cPt2di(1,1)),
   mR0Sym         (3.0),
   mR1Sym         (8.0),
   mRExtreSym     (9.0),
   mTHRS_Sym      (0.7),
   mTHRS_Bin      (0.5),
   mImVisu        (cPt2di(1,1)),
   mPatF          ("XXX"),
   mWithGT        (false),
   mDMaxMatch     (2.0)
{
}

cCollecSpecArg2007 & cAppliExtractCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   // Standard use, we put args of  cAppliParseBoxIm first
   return
         APBI_ArgObl(anArgObl)
             <<   Arg2007(mNameTarget,"Name of target file")
   ;
}
/* But we could also put them at the end
   return
         APBI_ArgObl(anArgObl <<   Arg2007(mNameTarget,"Name of target file"))
   ;
*/

cCollecSpecArg2007 & cAppliExtractCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return APBI_ArgOpt
	  (
	        anArgOpt
                    << AOpt2007(mRaysTF, "RayTF","Rays Min/Max for testing filter",{eTA2007::HDV,eTA2007::Tuning})
                    << AOpt2007(mPatF, "PatF","Pattern filters" ,{AC_ListVal<eDCTFilters>()})
	  );
   ;
}

void cAppliExtractCodeTarget::ShowStats(const std::string & aMes) 
{
   int aNbOk=0;
   for (const auto & aR : mVDCT)
   {
      if (aR->mState == eResDCT::Ok)
         aNbOk++;
   }
   StdOut() <<  aMes << " NB DCT = " << aNbOk << " Prop " << (double) aNbOk / (double) APBI_DIm().NbElem() << "\n";
}

void cAppliExtractCodeTarget::MarkDCT() 
{
     for (auto & aDCT : mVDCT)
     {
          cPt3di aCoul (-1,-1,-1);

          if (aDCT->mState == eResDCT::Ok)      aCoul =  cRGBImage::Green;
	  /*
          if (aDCT.mState == eResDCT::Divg)    aCoul =  cRGBImage::Red;
          if (aDCT.mState == eResDCT::LowSym)  aCoul =  cRGBImage::Yellow;
          if (aDCT.mState == eResDCT::LowBin)  aCoul =  cRGBImage::Blue;
          if (aDCT.mState == eResDCT::LowRad)  aCoul =  cRGBImage::Cyan;
	  */
          if (aDCT->mState == eResDCT::LowSymMin)  aCoul =  cRGBImage::Red;


          if (aCoul.x() >=0)
             mImVisu.SetRGBrectWithAlpha(aDCT->Pix0(),2,aCoul,0.5);
     }
}

void cAppliExtractCodeTarget::SelectOnFilter(cFilterDCT<tREAL4> * aFilter,bool MinCrown,double aThrS,eResDCT aModeSup)
{
  for (auto & aDCT : mVDCT)
  {
      if (aDCT->mState == eResDCT::Ok)
      {
         double aSc =  MinCrown ?   aFilter->ComputeValMaxCrown(aDCT->mPt,aThrS)   : aFilter->ComputeVal(aDCT->mPt);
         if (aSc>aThrS)
            aDCT->mState = aModeSup;
      }
  }
  ShowStats(E2Str(aFilter->ModeF()) + " Min=" +ToStr(MinCrown));
  delete aFilter;
}

void cAppliExtractCodeTarget::MatchOnGT(cGeomSimDCT & aGSD)
{

     cWhitchMin<cDCT*,double>  aWMin(nullptr,1e10);

     for (auto aPtrDCT : mVDCT)
         aWMin.Add(aPtrDCT,SqN2(aPtrDCT->mPt-aGSD.mC));

     if (aWMin.ValExtre() < Square(mDMaxMatch))
     {
	aGSD.mResExtr = aWMin.IndexExtre();
	aGSD.mResExtr->mGT =& aGSD;
        StdOut()<< "ddddd " << Norm2(aGSD.mC- aGSD.mResExtr->mPt)  << " " <<  Norm2(aGSD.mC- ToR(aGSD.mResExtr->mPix0))  << "\n";
     }
     else
     {
        StdOut()<< "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn\n";
     }
}

void  cAppliExtractCodeTarget::DoExtract()
{
     tDataIm &  aDIm = APBI_DIm();
     tIm        aIm = APBI_Im();
     mImVisu =   RGBImFromGray(aDIm);
     // mNbPtsIm = aDIm.Sz().x() * aDIm.Sz().y();

     // [1]   Extract point that are extremum of symetricity
     
         //    [1.1]   extract integer pixel
     cIm2D<tREAL4>  aImSym = ImSymetricity(false,aIm,mR0Sym,mR1Sym,0);  // compute fast symetry
     cResultExtremum aRExtre(true,false);              //structire for result of extremun , compute min not max
     ExtractExtremum1(aImSym.DIm(),aRExtre,mRExtreSym);  // do the extraction

         // [1.2]  afine to real coordinate by fiting quadratic models
     cAffineExtremum<tREAL4> anAffEx(aImSym.DIm(),2.0);
     for (const auto & aPix : aRExtre.mPtsMin)
     {
          mVDCT.push_back(new cDCT(aPix,anAffEx));
     }

     if (mWithGT)
     {
        for (auto & aGSD : mRSim.mVG)
             MatchOnGT(aGSD);
     }
         //
     ShowStats("Init ");

     //   ====   Symetry filters ====
     for (auto & aDCT : mVDCT)
     {
        if (aDCT->mState == eResDCT::Ok)
        {
           aDCT->mSym = aImSym.DIm().GetV(aDCT->Pix());
           if (aDCT->mSym>mTHRS_Sym)
              aDCT->mState = eResDCT::LowSym;  
        }
     }
     ShowStats("LowSym");

     //   ====   Binarity filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocBin(aIm,6,8),false,mTHRS_Bin,eResDCT::LowBin);


     //   ====   Radial filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocRad(mImGrad,3.5,5.5,1.0),false,0.5,eResDCT::LowRad);

     // Min of symetry
     SelectOnFilter(cFilterDCT<tREAL4>::AllocSym(aIm,4,8,1),true,0.8,eResDCT::LowSym);

     // Min of bin 
     SelectOnFilter(cFilterDCT<tREAL4>::AllocBin(aIm,4,8),true,mTHRS_Bin,eResDCT::LowBin);

     //   ====   MinOf Symetry ====
     //   ====   MinOf Symetry ====

     MarkDCT() ;
     mImVisu.ToFile("VisuCodeTarget.tif");
}


void  cAppliExtractCodeTarget::TestFilters()
{
     tDataIm &  aDIm = APBI_DIm();
     tIm        aIm = APBI_Im();

     StdOut() << "SZ "  <<  aDIm.Sz() << " Im=" << APBI_NameIm() << "\n";


     for (const auto & anEF :  mTestedFilters)
     {
          StdOut()  << " F=" << E2Str(anEF) << "\n";
          cFilterDCT<tREAL4> * aFilter = nullptr;

	  if (anEF==eDCTFilters::eSym)
	     aFilter =  cFilterDCT<tREAL4>::AllocSym(aIm,mRaysTF.x(),mRaysTF.y(),1);

	  if (anEF==eDCTFilters::eBin)
	     aFilter =  cFilterDCT<tREAL4>::AllocBin(aIm,mRaysTF.x(),mRaysTF.y());

	  if (anEF==eDCTFilters::eRad)
	     aFilter =  cFilterDCT<tREAL4>::AllocRad(mImGrad,mRaysTF.x(),mRaysTF.y(),1);

	  if (aFilter)
	  {
              cIm2D<tREAL4>  aImF = aFilter->ComputeIm();
	      std::string aName = "TestDCT_" +  E2Str(anEF)  + "_" + Prefix(mNameIm) + ".tif";
	      aImF.DIm().ToFile(aName);
	  }

	  delete aFilter;
     }

}

int cAppliExtractCodeTarget::ExeOnParsedBox()
{
   mImGrad    =  Deriche(APBI_DIm(),2.0);
   TestFilters();
   DoExtract();

   return EXIT_SUCCESS;
}

int  cAppliExtractCodeTarget::Exe()
{
   std::string aNameGT = LastPrefix(APBI_NameIm()) + std::string("_GroundTruth.xml");
   if (ExistFile(aNameGT))
   {
      mWithGT = true;
      mRSim   = cResSimul::FromFile(aNameGT);
   }
 
   mTestedFilters = SubOfPat<eDCTFilters>(mPatF,true);
   StdOut()  << " IIIIm=" << APBI_NameIm()   << " " << aNameGT << "\n";

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
      return ResultMultiSet();

   mPCT.InitFromFile(mNameTarget);
   APBI_ExecAll();  // run the parse file  SIMPL


   return EXIT_SUCCESS;
}
};


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_CodedTarget;

tMMVII_UnikPApli Alloc_ExtractCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliExtractCodeTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCodedTarget
(
     "CodedTargetExtract",
      Alloc_ExtractCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);


};
