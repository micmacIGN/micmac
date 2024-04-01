#include "MMVII_PCSens.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_ExtractLines.h"


namespace MMVII
{

cHoughPS::cHoughPS(const cHoughTransform * aHT,const cPt2dr & aTR,tREAL8 aCumul,const cPt2dr & aP1,const cPt2dr & aP2) :
    mHT       (aHT),
    mTetaRho  (aTR),
    mCumul    (aCumul),
    mSegE     (aP1,aP2),
    mCode     (eCodeHPS::Ok)
{
    InitMatch();
}




tREAL8 cHoughPS::DistAnglAntiPar(const cHoughPS& aPS2) const
{
     return diff_circ(Teta()+M_PI,aPS2.Teta(),2*M_PI);
}

tREAL8 cHoughPS::DY(const cHoughPS & aHPS) const
{
   return mSegE.ToCoordLoc(aHPS.mSegE.PMil()) .y() ;
}

const cSegment2DCompiled<tREAL8> & cHoughPS::Seg() const {return mSegE;}
const cPt2dr & cHoughPS::TetaRho() const {return mTetaRho;}
const tREAL8 & cHoughPS::Teta()    const {return mTetaRho.x();}
const tREAL8 & cHoughPS::Rho()     const {return mTetaRho.y();}
const tREAL8 & cHoughPS::Cumul()     const {return mCumul;}
cHoughPS * cHoughPS::Matched() const {return mMatched;}
eCodeHPS  cHoughPS::Code() const  {return mCode;}
void cHoughPS::SetCode(eCodeHPS aCode) {  mCode = aCode;}

cPt2dr  cHoughPS::IndTetaRho() const
{
	return cPt2dr(mHT->Teta2RInd(Teta()),mHT->Rho2RInd(Rho()));
}

tREAL8 cHoughPS::Dist(const cHoughPS& aPS2,const tREAL8 &aFactTeta) const
{
     return   std::abs(Rho()-aPS2.Rho())
	    + DistAnglAntiPar(aPS2) * (aFactTeta * mHT->RhoMax())
     ;
}

void cHoughPS::Test(const cHoughPS &  aHPS) const
{
	StdOut() << "LARG " << mSegE.ToCoordLoc(aHPS.mSegE.PMil()) .y()
		 << " RMDistAng " << DistAnglAntiPar(aHPS) * mHT->RhoMax()
		 << " DistAng " << DistAnglAntiPar(aHPS) 
		 // << " T1=" << Teta()<< " T2=" aHPS.Teta()  << " "<< diff_circ(Teta(),aHPS.Teta(),2*M_PI)
		 << "\n";
}

bool cHoughPS::Match(const cHoughPS & aPS2,bool IsLight,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax) const
{
   tREAL8 aDY1 =      DY(aPS2 );
   tREAL8 aDY2 = aPS2.DY(*this);

   // Test the coherence of left/right position 
   if ( ((aDY1>0) ==IsLight) || ((aDY2>0) ==IsLight))  return false;

   if ( DistAnglAntiPar(aPS2) > aMaxTeta)  return false;

   tREAL8 aDYAbs1 = std::abs(aDY1);
   tREAL8 aDYAbs2 = std::abs(aDY2);

   return (aDYAbs1>aDMin) && (aDYAbs1<aDMax) && (aDYAbs2>aDMin) && (aDYAbs2<aDMax);
}


void cHoughPS::InitMatch()
{
     mMatched = nullptr;
     mDistM   = 1e10;
}
void cHoughPS::UpdateMatch(cHoughPS * aNewM,tREAL8 aDist)
{
    if (aDist<mDistM)
    {
       mMatched = aNewM;
       mDistM   = aDist;
    }
}

void cHoughPS::SetMatch(std::vector<cHoughPS*> & aVPS,bool IsLight,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax)
{
     //  Reset matches
     for (auto & aPtr : aVPS)
	 aPtr->InitMatch();

     //  compute best match
     for (size_t aK1=0 ; aK1<aVPS.size() ; aK1++)
     {
          for (size_t aK2=aK1+1 ; aK2<aVPS.size() ; aK2++)
	  {
               if (aVPS[aK1]->Match(*aVPS[aK2],IsLight,aMaxTeta,aDMin,aDMax))
               {
                  tREAL8 aD12 = aVPS[aK1]->Dist(*aVPS[aK2],2.0);
                  aVPS[aK1]->UpdateMatch(aVPS[aK2],aD12);
                  aVPS[aK2]->UpdateMatch(aVPS[aK1],aD12);
               }
	  }
     }

     //  Test if reciproc match
     for (auto & aPtr : aVPS)
     {
          if (aPtr->mMatched && (aPtr->mMatched->mMatched!=aPtr))
	  {
             aPtr->mMatched->mMatched =nullptr;
             aPtr->mMatched =nullptr;
	  }
     }
}

/* =============================================== */
/*                                                 */
/*                 cAppliExtractLine               */
/*                                                 */
/* =============================================== */


/**  An application for  computing line extraction. Created for CERN
 *  porject, and parametrization is more or less optimized for this purpose.
 */

class cAppliExtractLine : public cMMVII_Appli
{
     public :
        cAppliExtractLine(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     private :
        typedef tREAL4 tIm;

	cPt2dr Redist(const cPt2dr &) const;
	cPt2dr Undist(const cPt2dr &) const;

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	std::vector<std::string>  Samples() const override;
        virtual ~cAppliExtractLine();

        void  DoOneImage(const std::string & aNameIm) ;

	void MakeVisu(const std::string & aNameIm);

        cPhotogrammetricProject  mPhProj;
        std::string              mPatImage;
	bool                     mLineIsWhite;
        bool                     mShowSteps;
	std::vector<tREAL8>      mVParams;
        cPerspCamIntrCalib *     mCalib;
	bool                     mAffineMax;
	std::vector<double>      mThreshCpt;
        tREAL8                   mTransparencyCont;
        cExtractLines<tIm>*      mExtrL;
	int                      mZoomImL;
        std::vector<cHoughPS*>   mVPS;
        std::vector<cHoughPS*>   mMatchedVPS;
	std::string              mNameReportByLine;
	std::string              mNameReportByIm;
	tREAL8                   mRelThrsCumulLow;
	tREAL8                   mRelThrsCumulHigh;

	bool                     mWithGT;  ///< Is there a ground truth of "handcrafted" segment
	bool                     mGTEmpty; ///<  Does the GT "says" that here is no valid segment
	std::vector<cPt2dr>      mVPtsGT;
};


cAppliExtractLine::cAppliExtractLine(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mShowSteps        (true),
    mVParams          { 4e-4,2.0,5.0},
    mCalib            (nullptr),
    mAffineMax        (true),
    mThreshCpt        {100,200,400,600},
    mTransparencyCont (0.5),
    mExtrL            (nullptr),
    mZoomImL          (1),
    mNameReportByLine ("LineMulExtract"),
    mNameReportByIm   ("LineByIm"),
    mRelThrsCumulLow    (0.15),
    mRelThrsCumulHigh   (0.30),
    mWithGT             (false),
    mGTEmpty            (true)
{
}

cAppliExtractLine::~cAppliExtractLine()
{
     delete mExtrL;
     DeleteAllAndClear(mVPS);
}

cCollecSpecArg2007 & cAppliExtractLine::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             <<  Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
	     <<  Arg2007(mLineIsWhite," True : its a light line , false dark ")
      ;
}

cCollecSpecArg2007 & cAppliExtractLine::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << mPhProj.DPOrient().ArgDirInOpt("","Folder for calibration to integrate distorsion")
	       << AOpt2007(mAffineMax,"AffineMax","Affinate the local maxima",{eTA2007::HDV})
	       << AOpt2007(mShowSteps,"ShowSteps","Show detail of computation steps by steps",{eTA2007::HDV})
	       << AOpt2007(mZoomImL,"ZoomImL","Zoom for images of line",{eTA2007::HDV})
	       << AOpt2007(mRelThrsCumulLow,"ThrCumLow","Low Thresold relative for cumul in histo",{eTA2007::HDV})
	       << AOpt2007(mRelThrsCumulHigh,"ThrCumHigh","Low Thresold relative for cumul in histo",{eTA2007::HDV})
               << mPhProj.DPPointsMeasures().ArgDirInOpt("","Folder for ground truth measure")
            ;
}

std::vector<std::string>  cAppliExtractLine::Samples() const
{
   return {
              "MMVII ExtractLine 'DSC_.*.JPG' ShowSteps=1 InOri=FB"
	};
}

cPt2dr cAppliExtractLine::Redist(const cPt2dr & aP) const
{
     return mCalib ? mCalib->Redist(aP) : aP;
}

cPt2dr cAppliExtractLine::Undist(const cPt2dr & aP) const
{
     return mCalib ? mCalib->Undist(aP) : aP;
}

void  cAppliExtractLine::DoOneImage(const std::string & aNameIm)
{

    tREAL8 aMulTeta = 1.0/M_PI;
// aMulTeta = 1.0;
    bool mShow = true;
    mCalib = nullptr;
    if (mPhProj.DPOrient().DirInIsInit())
       mCalib = mPhProj.InternalCalibFromImage(aNameIm);


   if (mPhProj.DPPointsMeasures().DirInIsInit()  && mPhProj.HasMeasureIm(aNameIm))
   {
      mWithGT = true;
      cSetMesPtOf1Im  aSetMes = mPhProj.LoadMeasureIm(aNameIm);

      if (aSetMes.NameHasMeasure("Line1") && aSetMes.NameHasMeasure("Line2"))
      {
          mGTEmpty = false;
          mVPtsGT.push_back(Undist(aSetMes.MeasuresOfName("Line1").mPt));
          mVPtsGT.push_back(Undist(aSetMes.MeasuresOfName("Line2").mPt));
      }
   }



    cIm2D<tIm> anIm = cIm2D<tIm>::FromFile(aNameIm);
    tREAL8  aTrhsCumulLow  = mRelThrsCumulLow   * Norm2(anIm.DIm().Sz());
    tREAL8  aTrhsCumulHigh = mRelThrsCumulHigh * Norm2(anIm.DIm().Sz());
    mExtrL = new cExtractLines<tIm> (anIm);

    mExtrL->SetDericheGradAndMasq(2.0,10.0,2,mShow); // aAlphaDerich,aRayMaxLoc,aBorder
    mExtrL->SetHough(cPt2dr(aMulTeta,1.0),0.1,mCalib,mAffineMax,mShow);

    // Extract Local Maxima
    std::vector<cPt3dr> aVMaxLoc = mExtrL->Hough().ExtractLocalMax(10,4.0,10.0,0.1);

    //  Select Maxima with Cum > aTrhsCumulLow + labelize the quality of seg
    int aRank=0;
    for (const auto & aPMax : aVMaxLoc)
    {
        cHoughPS * aPS = mExtrL->Hough().PtToLine(aPMax);
	if (aPS->Cumul() > aTrhsCumulLow)
	{
           if (aPS->Cumul() < aTrhsCumulHigh)
               aPS->SetCode(eCodeHPS::LowCumul);
	   else if (aRank !=0)
               aPS->SetCode(eCodeHPS::NotFirst);
           mVPS.push_back(aPS);
	}
	else
           delete aPS;
	aRank++;
    }

    cHoughPS::SetMatch(mVPS,mLineIsWhite,mVParams.at(0),mVParams.at(1),mVParams.at(2));

    for (auto & aHS1 : mVPS)
    {
        cHoughPS *aHS2 = aHS1->Matched();
        if ((aHS2!=nullptr) && (! BoolFind(mMatchedVPS,aHS2)))
        {
            mMatchedVPS.push_back(aHS1);
	}
    }


    // Compute the quality and save it in report
    {
        std::string aStringQual = "OK";

        if (mMatchedVPS.empty()) 
           aStringQual = "Pb_Empty";
        else if ((mMatchedVPS.size()>=2) && (mMatchedVPS.at(1)->Code() == eCodeHPS::Ok))
           aStringQual = "Pb_AmbNOK";
        else if ((mMatchedVPS.size()>=2) && ( (mMatchedVPS.at(1)->Cumul()/mMatchedVPS.at(0)->Cumul() ) > 0.5))
           aStringQual = "Pb_AmbRatio12";
        else if (mMatchedVPS.at(0)->Code() != eCodeHPS::Ok)
           aStringQual = "Pb_LowCumul";


	if (mWithGT)
	{
            if (mGTEmpty)
	    {
               aStringQual  = mMatchedVPS.empty() ? "Ok_Empty"  :  "Pb_NotEmpty";
	    }
	    else
	    {
                 if (mMatchedVPS.empty())
		 {
		 }
		 else
		 {
                     bool  OkGT= true;
                     for (const auto & aPt : mVPtsGT)
		     {
                         for (bool isSeg2 : {false,true})
			 {
                              cHoughPS * aSeg = mMatchedVPS.at(0);
			      if (isSeg2)
                                 aSeg = aSeg->Matched();
			      cPt2dr aPLoc = aSeg->Seg().ToCoordLoc(aPt);
			      if (aPLoc.y() > 2.0)
				      OkGT=false;
			 }
		     }
		     aStringQual = OkGT ? "OK_ByGT" : "PB_ByGT";
		 }
	    }
	}

        AddOneReportCSV(mNameReportByIm,{aNameIm,aStringQual});
    }
   



    for (size_t aKH=0 ; aKH<mVPS.size() ; aKH++)
    {
        cHoughPS *aHS1 = mVPS[aKH];
        cHoughPS *aHS2 = aHS1->Matched();
        // Compute colour, depend of ranking
        bool isOk = (aHS2 != nullptr);
        // if (aKH>=2)  aCoul = cRGBImage::Green;
        // if (aKH>=4)  aCoul = cRGBImage::Blue;
        if (isOk && (aHS1> aHS2))
        {
            tREAL8 aDAng = aHS1->DistAnglAntiPar(*aHS2) * mExtrL->Hough().RhoMax() ;
            tREAL8 aLarg = aHS1->DY(*aHS2) ;
            tREAL8 aCumul = (aHS1->Cumul()+aHS2->Cumul())/2.0;
            AddOneReportCSV(mNameReportByLine,{aNameIm,ToStr(aDAng),ToStr(aLarg),ToStr(aCumul)});
        }
    }

    if (mShowSteps)
       MakeVisu(aNameIm);
}

void cAppliExtractLine::MakeVisu(const std::string & aNameIm)
{
    const  cDataIm2D<tREAL4>& aDAccum =mExtrL->Hough().Accum().DIm();
    tREAL8 aVMax=0.0;

    // [0]  Print some  statistic to compared the compatcness of histogramm
    {
        std::vector<int>  aVCpt(mThreshCpt.size(),0);  // count the number of point over a threshold
        for (const auto & aPix : aDAccum)
        {
            tREAL8 aV = aDAccum.GetV(aPix);
	    UpdateMax(aVMax,aV);
            for (size_t aK=0 ; aK<mThreshCpt.size() ; aK++)
            {
                if (aV>mThreshCpt.at(aK))
                   aVCpt.at(aK)++;
            }
        }

	// print max value
	StdOut() << "VMAX=" << aVMax << std::endl;
        for (size_t aK=0 ; aK<mThreshCpt.size() ; aK++)
            StdOut() << " Cpt=" << aVCpt.at(aK) << " for threshold " << mThreshCpt.at(aK) << std::endl;
    }


    std::string aNameTif = LastPrefix(aNameIm) + ".tif";

    //  [1]  Visu selected max in  direction of gradient
    {
         cRGBImage aImV= mExtrL->MakeImageMaxLoc(mTransparencyCont);
         aImV.ToJpgFileDeZoom(mPhProj.DirVisu() + "DetectL_"+ aNameTif,1);
    }

    //  [2]  Visu module of gradient
    {
         std::string aNameGrad = mPhProj.DirVisu()+"Grad_" + aNameTif;
         mExtrL->Grad().NormG().DIm().ToFile(aNameGrad);
         // Convert_JPG(aNameGrad,true,90,"jpg");
    }

       
    // [3] Visu  the accum + local maximal
    {
         cRGBImage  aVisAccum =  RGBImFromGray(aDAccum,255.0/aVMax);
	 int aK=0;
         for (const auto & aPS : mVPS)
         {
             //aVisAccum.SetRGBrectWithAlpha(ToI(aPS->IndTetaRho())  ,15,cRGBImage::Red,0.5);
	     cPt2dr aC=aPS->IndTetaRho();
	     cPt2dr aSz(7,7);
             aVisAccum.FillRectangle(cRGBImage::Red,ToI(aC-aSz),ToI(aC+aSz),cPt3dr(0.5,1.0,1.0));
	     aK++;
         }
         aVisAccum.ToJpgFileDeZoom(mPhProj.DirVisu() + "Accum_" + aNameTif,1);
         aDAccum.ToFile(mPhProj.DirVisu() + "RawAccum_" + aNameTif);
    }
    // [4]  Visu of Image +  Lines
    {
         cRGBImage  aVisIm =  cRGBImage::FromFile(aNameIm,mZoomImL); // Initialize with image
	 const auto & aDIm = aVisIm.ImR().DIm();
         for (size_t aKH=0 ; aKH<mVPS.size() ; aKH++)
	 {
             cHoughPS *aHS1 = mVPS[aKH];
             cHoughPS *aHS2 = aHS1->Matched();
             // Compute colour, depend of ranking
	     bool isOk = (aHS2 != nullptr);
             cPt3di aCoul = isOk ? cRGBImage::Red : cRGBImage::Blue ;

	     // Compute Hough-Point -> line
             cSegment<tREAL8,2> aSeg =  mVPS[aKH]->Seg();

	     tREAL8 aRay=5;
	     cPt2dr aC = Redist(aSeg.PMil() +VUnit(aSeg.V12()*cPt2dr(0,-1))* aRay);
             aVisIm.DrawCircle(aCoul,aC ,aRay);

	     //  write point by point, in two direction
             for (tREAL8 aSign : {-1.0,1.0})
             {
                 cPt2dr aPt = aSeg.PMil();
                 cPt2dr aTgt = VUnit(aSeg.V12()) * aSign; // direction
		 cPt2dr  aQ = Redist(aPt) ; // eventulay make invert distorsion correcion
                 while (aDIm.InsideBL(aQ))
                 {
                     aVisIm.SetRGBPoint(aQ,aCoul);  // print point
                     aPt = aPt+aTgt; // increment point
		     aQ = Redist(aPt);  // eventulay make invert distorsion correcion
                 }
             }
	 }
	 std::string aNameLine = mPhProj.DirVisu() + "Lines_" + aNameTif;
	 // convert dont handle well big file, so generate jpg only if zoom=1
	 if (mZoomImL==1)
	    aVisIm.ToJpgFileDeZoom(mPhProj.DirVisu() + "Lines_" + aNameTif,1);
	 else
	    aVisIm.ToFile(aNameLine);

    }
}



int cAppliExtractLine::Exe()
{
    mPhProj.FinishInit();
    InitReport(mNameReportByLine,"csv",true,{"NameIm","Paral","Larg","Cumul"});
    InitReport(mNameReportByIm,"csv",true,{"NameIm","CodeResult"});

    // AddHeaderReportCSV(mNameReportByLine,{"NameIm","Paral","Larg","Cumul"});
    // AddHeaderReportCSV(mNameReportByIm,{"NameIm","CodeResult"});
    if (RunMultiSet(0,0))
    {
       return ResultMultiSet();
    }
    DoOneImage(UniqueStr(0));
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_AppliExtractLine(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliExtractLine(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecAppliExtractLine
(
     "ExtractLine",
      Alloc_AppliExtractLine,
      "Extraction of lines",
      {eApF::Ori},
      {eApDT::Ori,eApDT::GCP},
      {eApDT::Console},
      __FILE__
);

};
