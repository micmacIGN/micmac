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
/*             cAppliExtractCodeTarget                          */
/*                                                              */
/*  *********************************************************** */

enum class eResDCT // Result Detect Code Target
{
     Ok,
     Divg,
     LowSym,
     LowSymMin,
     LowBin,
     LowRad
};

class  cDCT
{
     public  :
         cDCT(const cPt2di aPt,cAffineExtremum<tREAL4> & anAffEx) :
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

         cPt2di  Pix()  const {return ToI(mPt);}
         cPt2di  Pix0() const {return mPix0;}

         cPt2di  mPix0;
         cPt2dr  mPt;
         eResDCT mState;

         double  mSym;
         double  mBin;
         double  mRad;
};


class cAppliExtractCodeTarget : public cMMVII_Appli,
	                        public cAppliParseBoxIm<tREAL4>
{
     public :
        cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	int ExeOnParsedBox() override;

	void TestFilters();
	void DoExtract();
        void ShowStats(const std::string & aMes) ;
        void MarkDCT() ;

	std::string mNameTarget;

	cParamCodedTarget        mPCT;
	cPt2dr                   mRaysTF;
        std::vector<eDCTFilters> mTestedFilters;    

        double   mR0Sym;     ///< R min for first very quick selection on symetry
        double   mR1Sym;     ///< R max for first very quick selection on symetry
        double   mRExtreSym; ///< R to compute indice of local maximal of symetry
        double   mTHRS_Sym; ///< Threshold for symetricity


        double mTHRS_Bin;

        std::vector<cDCT>   mVDCT; ///< vector of detected target


        cRGBImage  mImVisu;
        std::string mPatF;
         
};


/* *************************************************** */
/*                                                     */
/*              cAppliExtractCodeTarget                   */
/*                                                     */
/* *************************************************** */

cAppliExtractCodeTarget::cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(5000,5000),cPt2di(300,300),false), // static_cast<cMMVII_Appli & >(*this))
   mRaysTF        ({4,8}),
   mR0Sym         (3.0),
   mR1Sym         (8.0),
   mRExtreSym     (7.0),
   mTHRS_Sym      (0.8),
   mTHRS_Bin      (0.5),
   mImVisu        (cPt2di(1,1)),
   mPatF          ("XXX")
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
      if (aR.mState == eResDCT::Ok)
         aNbOk++;
   }
   StdOut() <<  aMes << " NB DCT = " << aNbOk << " Prop " << (double) aNbOk / (double) APBI_DIm().NbElem() << "\n";
}

void cAppliExtractCodeTarget::MarkDCT() 
{
     for (auto & aDCT : mVDCT)
     {
          cPt3di aCoul (-1,-1,-1);

          if (aDCT.mState == eResDCT::Ok)      aCoul =  cRGBImage::Green;
	  /*
          if (aDCT.mState == eResDCT::Divg)    aCoul =  cRGBImage::Red;
          if (aDCT.mState == eResDCT::LowSym)  aCoul =  cRGBImage::Yellow;
          if (aDCT.mState == eResDCT::LowBin)  aCoul =  cRGBImage::Blue;
          if (aDCT.mState == eResDCT::LowRad)  aCoul =  cRGBImage::Cyan;
	  */
          if (aDCT.mState == eResDCT::LowSymMin)  aCoul =  cRGBImage::Red;


          if (aCoul.x() >=0)
             mImVisu.SetRGBrectWithAlpha(aDCT.Pix0(),2,aCoul,0.5);
     }
}

void  cAppliExtractCodeTarget::DoExtract()
{
     tDataIm &  aDIm = APBI_DIm();
     tIm        aIm = APBI_Im();
     mImVisu =   RGBImFromGray(aDIm);
     // mNbPtsIm = aDIm.Sz().x() * aDIm.Sz().y();

     // Extract point that are extremum of symetricity
     cIm2D<tREAL4>  aImSym = ImSymetricity(false,aIm,mR0Sym,mR1Sym,0);
     cResultExtremum aRExtre(true,false);
     ExtractExtremum1(aImSym.DIm(),aRExtre,mRExtreSym);

     cAffineExtremum<tREAL4> anAffEx(aImSym.DIm(),2.0);


     for (const auto & aPix : aRExtre.mPtsMin)
     {
          mVDCT.push_back(cDCT(aPix,anAffEx));
     }
     ShowStats("Init ");

     //   ====   Symetry filters ====
     for (auto & aDCT : mVDCT)
     {
        if (aDCT.mState == eResDCT::Ok)
        {
           aDCT.mSym = aImSym.DIm().GetV(aDCT.Pix());
           if (aDCT.mSym>mTHRS_Sym)
              aDCT.mState = eResDCT::LowSym;  
        }
     }
     ShowStats("LowSym ");

     //   ====   Binarity filters ====
     {
         std::vector<cPt2di>  aVectVois =  VectOfRadius(6,8,false);

         for (auto & aDCT : mVDCT)
         {
             if (aDCT.mState == eResDCT::Ok)
             {
                 aDCT.mBin = IndBinarity(aDIm,aDCT.Pix(),aVectVois);
                 if (aDCT.mBin>mTHRS_Bin)
                    aDCT.mState = eResDCT::LowBin;  
             }
        }
     }
     ShowStats("Binary ");

     //   ====   Radial filters ====
     {
         cImGrad<tREAL4>  aImG = Deriche(aDIm,1.0);
         // std::vector<cPt2di>  aVectVois =  VectOfRadius(3.5,5.5,false);
         std::vector<cPt2di>  aVectVois =  VectOfRadius(2,4,false);
         std::vector<cPt2dr>  aVDir = VecDir(aVectVois);


         for (auto & aDCT : mVDCT)
         {
             if (aDCT.mState == eResDCT::Ok)
             {
                 aDCT.mRad =  Starity (aImG,aDCT.mPt,aVectVois,aVDir,1.0);

                 if (aDCT.mRad>0.5)
                    aDCT.mState = eResDCT::LowRad;  
             }
        }
     }
     ShowStats("Starity ");

     //   ====   MinOf Symetry ====
     {
         // std::vector<cPt2di>  aVectVois =  VectOfRadius(3.5,5.5,false);
         std::vector<tREAL8>  aVRadius ={3.0,4.0,5.0,6.0};
         tREAL8 aThickN = 1.5;
         
         std::vector<std::vector<cPt2di> >  aVVOis ;
         for (const auto & aRadius : aVRadius)
             aVVOis.push_back(VectOfRadius(aRadius,aRadius+aThickN,true));
/*

         for (auto & aDCT : mVDCT)
         {
             if (aDCT.mState == eResDCT::Ok)
             {
                 double aMaxSym = 0.0;

                 aDCT.mRad =  Starity (aImG,aDCT.mPt,aVectVois,aVDir,1.0);

                 if (aDCT.mRad>0.5)
                    aDCT.mState = eResDCT::LowRad;  
             }
        }
*/
     }
     ShowStats("MaxSym ");



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

            cIm2D<tREAL4> aImF(cPt2di(1,1));

            if (anEF==eDCTFilters::eBin)
                aImF = ImBinarity(aDIm,mRaysTF.x(),mRaysTF.y(),1.0);

            if (anEF==eDCTFilters::eSym)
                aImF = ImSymetricity(true,aIm,mRaysTF.x(),mRaysTF.y(),1.0);

            if (anEF==eDCTFilters::eRad)
            {
                cImGrad<tREAL4>  aImG = Deriche(aDIm,1.0);
                aImF = ImStarity(aImG,mRaysTF.x(),mRaysTF.y(),1.0);
            }


            if (aImF.DIm().Sz().x() > 1)
            {
	       std::string aName = "TestDCT_" +  E2Str(anEF)  + "_" + Prefix(mNameIm) + ".tif";
	       aImF.DIm().ToFile(aName);
            }

/*
  
          cIm2D<tREAL4>  aImBin = ImBinarity(aDIm,aDist/1.5,aDist,1.0);
	  std::string aName = "TestBin_" + ToStr(aDist) + "_" + Prefix(APBI_NameIm()) + ".tif";
	  aImBin.DIm().ToFile(aName);
	  StdOut() << "Done Bin\n";
          cIm2D<tREAL4>  aImSym = ImSymetricity(aDIm,aDist/1.5,aDist,1.0);
	  std::string aName = "TestSym_" + ToStr(aDist) + "_" + Prefix(APBI_NameIm()) + ".tif";
	  aImSym.DIm().ToFile(aName);
	  StdOut() << "Done Sym\n";
*/

	  /*
          cIm2D<tREAL4>  aImStar = ImStarity(aImG,aDist/1.5,aDist,1.0);
	  aName = "TestStar_" + ToStr(aDist) + "_" + Prefix(mNameIm) + ".tif";
	  aImStar.DIm().ToFile(aName);
	  StdOut() << "Done Star\n";

          cIm2D<tREAL4>  aImMixte =   aImSym + aImStar * 2.0;
	  aName = "TestMixte_" + ToStr(aDist) + "_" + Prefix(mNameIm) + ".tif";
	  aImMixte.DIm().ToFile(aName);
	  */
     }

}

int cAppliExtractCodeTarget::ExeOnParsedBox()
{
/*
   if (APBI_TestMode())
   {
   }
   else
   {
   }
*/
   TestFilters();
   DoExtract();

   return EXIT_SUCCESS;
}

int  cAppliExtractCodeTarget::Exe()
{
   mTestedFilters = SubOfPat<eDCTFilters>(mPatF,true);
   StdOut()  << " IIIIm=" << APBI_NameIm()   << "\n";

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
