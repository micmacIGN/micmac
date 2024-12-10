#include "MMVII_PCSens.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_ExtractLines.h"
#include "MMVII_2Include_CSV_Serial_Tpl.h"



/*    ========  HOUGH ==========================

Best HOUGH:
   * 0002.2713 : Hough

Houh w/o teta =>  * 0000.2490 (relase) => 0.16 debug,
This mean almost all the code in in the main loop of accumulator, no much thing to optimize 
(maybe full integer computation ??)
*/

/*
    Gradient 
  * 0006.6604 : DericheAndMasq  
     # with 5+Der :   0012.7374 
     # with 5+Norm    0008.5546 

    Deriche ~ 1.2
    Norm    ~ 0.4
    10 Sobel -> 0.4 s !!


With tabul-releas
    Norm * 1  : DericheAndMasq   0003.7359 0003.8297
    Norm * 21  : DericheAndMasq   0005.2141  0005.0927   5.4152
    Norm -> 0.075 s

*/

/*

****************  Non Accurate *******
DEBUG :
 * 0000.0357 :  OTHERS
 * 0006.2783 : DericheAndMasq
 * 0003.2337 : Hough
 * 0000.0049 : Initialisation
 * 0000.1839 : MaxLocHough
 * 0000.0773 : ReadImage
 * 0006.2598 : Report
 * 0000.0307 : Time-w/o-tabul
 * 0000.0005 : Time-with-tabul

RELEASE :
 * 0000.0075 :  OTHERS
 * 0003.9107 : DericheAndMasq
 * 0002.2713 : Hough
 * 0000.0051 : Initialisation
 * 0000.0898 : MaxLocHough
 * 0000.0814 : ReadImage
 * 0004.4496 : Report
 * 0000.0238 : Time-w/o-tabul
 * 0000.0003 : Time-with-tabul


****************  Accurate *******
DEBUG :
 * 0000.0030 :  OTHERS
 * 0006.4666 : DericheAndMasq
 * 0010.7677 : Hough
 * 0000.0013 : Initialisation
 * 0000.1895 : MaxLocHough
 * 0000.0587 : ReadImage
 * 0006.4582 : Report
 * 0000.0088 : Time-w/o-tabul
 * 0000.0002 : Time-with-tabul


RELEASE :
 * 0000.0501 :  OTHERS
 * 0004.0072 : DericheAndMasq
 * 0008.4389 : Hough
 * 0000.0052 : Initialisation
 * 0000.0936 : MaxLocHough
 * 0000.0812 : ReadImage
 * 0004.5516 : Report
 * 0000.0258 : Time-w/o-tabul
 * 0000.0003 : Time-with-tabul

*/

namespace MMVII
{

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


        void InitCalibration();
        void  DoOneImage(const std::string & aNameIm) ;

	void MakeVisu(const std::string & aNameIm);

        cPhotogrammetricProject  mPhProj;
        std::string              mPatImage;
        std::string              mNameCurIm;
	bool                     mLineIsWhite;
        int                      mGenVisu;
	std::vector<tREAL8>      mParamMatch;
        cPerspCamIntrCalib *     mCalib;      ///< Calibration 4 correc dist
        // cTabuMapInv<2>*          mCalDUD;     ///< Tabul of dist/undist for acceleratio,
        int                      mNbTabCalib; ///< Number of grid in calib tabul
	bool                     mAccurateHough;
	std::vector<double>      mThreshCpt;
        tREAL8                   mTransparencyCont;
        cExtractLines<tIm>*      mExtrL;
	int                      mZoomImL;
        std::vector<cHoughPS>    mVPS;
        std::vector<cParalLine>  mParalLines;
	std::string              mNameReportByLine;
	std::string              mNameReportByIm;
	tREAL8                   mRelThrsCumulLow;
	tREAL8                   mHoughSeuilAng;

	bool                     mWithGT;  ///< Is there a ground truth of "handcrafted" segment
	bool                     mGTHasSeg; ///<  Does the GT "says" that here is no valid segment
	std::optional<tSeg2dr>   mSegGT;
	cTimerSegm               mTimeSeg;

        std::string              mIdExportCSV;  ///<  used of export lines in CSV files
}; 


cAppliExtractLine::cAppliExtractLine(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mPhProj           (*this),
    mGenVisu          (0),
    mParamMatch       {6e-4,2.0,7.0},
    mCalib            (nullptr),
    // mCalDUD           (nullptr),
    mNbTabCalib       (100),
    mAccurateHough    (false),
    mThreshCpt        {100,200,400,600},
    mTransparencyCont (0.5),
    mExtrL            (nullptr),
    mZoomImL          (1),
    mNameReportByLine ("LineMulExtract"),
    mNameReportByIm   ("LineByIm"),
    mRelThrsCumulLow    (0.05),
    mHoughSeuilAng      (0.20),
    mWithGT             (false),
    mGTHasSeg           (false),
    mTimeSeg            (this),
    mIdExportCSV        ("Lines")
{
}

cAppliExtractLine::~cAppliExtractLine()
{
     delete mExtrL;
}

cCollecSpecArg2007 & cAppliExtractLine::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             <<  Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
	     <<  Arg2007(mLineIsWhite," True : its a light line , false dark ")
             << mPhProj.DPOrient().ArgDirInMand()
             << mPhProj.DPPointsMeasures().ArgDirOutMand()
      ;
}

cCollecSpecArg2007 & cAppliExtractLine::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
	       << AOpt2007(mAccurateHough,"AccurateHough","Accurate/Quick hough",{eTA2007::HDV})
	       << AOpt2007(mGenVisu,"GenVisu","Generate Visu 0 none, 1 déroulé, 2 image wire, 3",{eTA2007::HDV})
	       << AOpt2007(mZoomImL,"ZoomImL","Zoom for images of line",{eTA2007::HDV})
	       << AOpt2007(mParamMatch,"MatchParam","[Angl,DMin,DMax]",{eTA2007::HDV,{eTA2007::ISizeV,"[3,3]"}})
	       << AOpt2007(mRelThrsCumulLow,"ThrCumLow","Low Thresold relative for cumul in histo",{eTA2007::HDV})
	       << AOpt2007(mHoughSeuilAng,"HoughThrAng","Angular threshold for hough acummulator",{eTA2007::HDV})
               << mPhProj.DPPointsMeasures().ArgDirInOpt("","Folder for ground truth measure")
            ;
}

std::vector<std::string>  cAppliExtractLine::Samples() const
{
   return {
              "MMVII ExtractLine AllImFil.xml true BA_311_C Fils"
	};
}

cPt2dr cAppliExtractLine::Redist(const cPt2dr & aP) const { return mCalib ? mCalib->Redist(aP) : aP; }
cPt2dr cAppliExtractLine::Undist(const cPt2dr & aP) const { return mCalib ? mCalib->Undist(aP) : aP; }

void  cAppliExtractLine::DoOneImage(const std::string & aNameIm)
{
    mNameCurIm = aNameIm;
    cAutoTimerSegm  anATSInit (mTimeSeg,"Initialisation");
    if (! IsInit(&mHoughSeuilAng))
       mHoughSeuilAng = mAccurateHough ? 0.2 : 0.1;

    tREAL8 aMulTeta = 1.0/M_PI;
    bool mShow = true;

   // [1]  Eventually init calibration for correction distorsion
    mCalib = nullptr;
    if (mPhProj.DPOrient().DirInIsInit())
    {
       mCalib = mPhProj.InternalCalibFromImage(mNameCurIm);

       // Init the tabulation, also make some timing test to check the efficiency
       //
       //  0000.3150 : Time-w/o-tabul
       //  0000.0096 : Time-with-tabul
       //  A factor 30 : efficient !!

       int aNbTest= 1;  //  make no test for now ;
       for (int aK=0; aK<2 ; aK++)
       {
            cAutoTimerSegm  anATSTab (mTimeSeg,(aK==0) ? "Time-w/o-tabul" : "Time-with-tabul");
            std::vector<cPt2dr>  aVPt = mCalib->PtsSampledOnSensor(10,true);

            for (int aKT1=0 ; aKT1< (aNbTest/100) ; aKT1++)
                for (const auto & aPt : aVPt)
                    mCalib->Undist(aPt);
            if (aK==0)
              mCalib->SetTabulDUD(mNbTabCalib);
       }
    }
    else
    {
    }


   // [2]  Eventually init ground truth  2D-points
   if (mPhProj.DPPointsMeasures().DirInIsInit()  && mPhProj.HasMeasureIm(mNameCurIm))
   {
      mWithGT = true;
      cSetMesPtOf1Im  aSetMes = mPhProj.LoadMeasureIm(mNameCurIm);

      if (aSetMes.NameHasMeasure("Line1") && aSetMes.NameHasMeasure("Line2"))
      {
          mGTHasSeg = true;
          cPt2dr aP1 = Undist(aSetMes.MeasuresOfName("Line1").mPt);
          cPt2dr aP2 = Undist(aSetMes.MeasuresOfName("Line2").mPt);
	  mSegGT = tSeg2dr(aP1,aP2);
	  /*
          mVPtsGT.push_back(Undist(aSetMes.MeasuresOfName("Line1").mPt));
          mVPtsGT.push_back(Undist(aSetMes.MeasuresOfName("Line2").mPt));


	  for (bool IsDir : {false,true})
	  {
	      tSeg2dr aSeg(mVPtsGT.at(IsDir?0:1), mVPtsGT.at(IsDir?1:0));
              mVSegsGT.push_back(aSeg);
	  }
	  */
      }
   }



    cAutoTimerSegm  anATSReadIm (mTimeSeg,"ReadImage");
    cIm2D<tIm> anIm = cIm2D<tIm>::FromFile(mNameCurIm);
    tREAL8  aTrhsCumulLow  = mRelThrsCumulLow   * Norm2(anIm.DIm().Sz());
//   [[maybe_unused]] tREAL8  aTrhsCumulHigh = mRelThrsCumulHigh * Norm2(anIm.DIm().Sz());
    mExtrL = new cExtractLines<tIm> (anIm);

    // Compute Gradient and extract max-loc in gradient direction
    cAutoTimerSegm  anATSDerAndMasq (mTimeSeg,"DericheAndMasq");
    mExtrL->SetSobelAndMasq(eIsWhite::Yes,10.0,12,mShow);  // aRayMaxLoc,aBorder
    // mExtrL->SetDericheGradAndMasq(2.0,10.0,12,mShow); // aAlphaDerich,aRayMaxLoc,aBorder
						     
    // Compute Hough-Transform
    cAutoTimerSegm  anATSHough (mTimeSeg,"Hough");
    mExtrL->SetHough(cPt2dr(aMulTeta,1.0),mHoughSeuilAng,mCalib,mAccurateHough,mShow);


    // Check that  Seg -> Houhg -> Seg  recover initial seg 
    if (mSegGT.has_value())
    {
        cSegment2DCompiled<tREAL8> aSeg0(mSegGT.value());
        cPt2dr aPtH0 = mExtrL->Hough().Line2PtPixel(aSeg0);
        cHoughPS  aSH = mExtrL->Hough().PtToLine(TP3z0(aPtH0));
	cSegment2DCompiled<tREAL8> aSeg1 = aSH.Seg();

	MMVII_INTERNAL_ASSERT_bench(aSeg1.Dist(aSeg0.P1())<1e-5,"Hough check seg");
	MMVII_INTERNAL_ASSERT_bench(aSeg1.Dist(aSeg0.P2())<1e-5,"Hough check seg");
        MMVII_INTERNAL_ASSERT_bench(Norm2(aSeg0.Tgt() - aSeg1.Tgt()) <1e-8,"Hough check seg");

	StdOut() << "HOUGH-GT= " << aPtH0 << "\n";
    }

    // Extract Local Maxima in hough space
    cAutoTimerSegm  anATSMaxLoc (mTimeSeg,"MaxLocHough");
    std::vector<cPt3dr> aVMaxLoc = mExtrL->Hough().ExtractLocalMax(10,5.0,10.0,0.1);
    StdOut() << " # VMAXLoc " << aVMaxLoc.size() << "\n";

    //  Refine the position in euclidean space
    //  Select Maxima with Cum > aTrhsCumulLow + labelize the quality of seg
    {
        mExtrL->MarqBorderMasq();
        for (const auto & aPMax : aVMaxLoc)
        {
            cHoughPS  aPS = mExtrL->Hough().PtToLine(aPMax);
            mExtrL->RefineLineInSpace(aPS);
	    if (aPS.Cumul() > aTrhsCumulLow)
	    {
               mVPS.push_back(aPS);
	    }
        }
        mExtrL->UnMarqBorderMasq();
    }
    StdOut() << " # VMAX>CumMin " << mVPS.size() << "\n";


    {
        cAutoTimerSegm  anATSMatchHough (mTimeSeg,"MatchLocHough");
	std::vector<std::pair<int,int>>  aVMatches = cHoughPS::GetMatches(mVPS,mLineIsWhite,mParamMatch.at(0),mParamMatch.at(1),mParamMatch.at(2));

	for (const auto & [aK1,aK2] : aVMatches)
	{
		mParalLines.push_back(cParalLine(mVPS.at(aK1),mVPS.at(aK2)));
	}
        StdOut() << " # NBMatched " << aVMatches.size() << "\n";
	SortOnCriteria
        (
              mParalLines,
	      [](const auto & aPtrHP) {return -aPtrHP.ScoreMatch();}
	);
		
        for (size_t aK=0 ; aK<mParalLines.size() ; aK++)
        {
            mParalLines[aK].SetRankMatch(aK);
	    std::string aNamVisuRH = (mGenVisu >=2) ? (mPhProj.DirVisuAppli() + LastPrefix(mNameCurIm)) : "";
            mParalLines[aK].ComputeRadiomHomog(anIm.DIm(),mCalib,aNamVisuRH);
        }

        std::vector<cParalLine>  aNewParL;
        for (const auto & aParL : mParalLines)
        {
            if (! aParL.RejectByComparison(mParalLines.at(0)))
               aNewParL.push_back(aParL);
        }
	mParalLines = aNewParL;

    }

    // StdOut() <<  "NbMatchHHHHHH " << mMatchedVPS.size() << "\n";

    cAutoTimerSegm  anATSReport (mTimeSeg,"Report");
    // Compute the quality and save it in report
    {
        // Intrinsic evaluation
        std::string aStringQual = "OK";

        if (mParalLines.empty()) 
           aStringQual = "Pb_Empty";
        else if (mParalLines.size()>=2) 
           aStringQual = "Pb_AmbNOK";

// StdOut() << "GGGGGTTTTT " << mWithGT
        // modification if there is a ground truth
	if (mWithGT)
	{
            if (mSegGT.has_value())
	    {
                 if (mParalLines.empty()) { } // Pb_Empty as it should be
		 else
		 {
                     tREAL8 aDist = mParalLines.at(0).DistGt(mSegGT.value());

		     if (aDist<2)
		     {
                        aStringQual = (mParalLines.size()==1) ? "OK_GT_1" : "OK_GT_Multiple_Match";
		     }
		     else
                        aStringQual = "PB_ByGT";
		 }
	    }
	    else
	    {
               aStringQual  = mParalLines.empty() ? "Ok_Empty"  :  "Pb_NotEmpty";
	    }
	}
 
        // register  the result in a report
        AddOneReportCSV(mNameReportByIm,{mNameCurIm,aStringQual});
    }
   

    {
        // generate the export
        cLinesAntiParal1Im  aExAllLines;
        aExAllLines.mNameIm =  mNameCurIm;
        if (mPhProj.DPOrient().DirInIsInit())
           aExAllLines.mDirCalib = mPhProj.DPOrient().DirIn();

	for (const auto & aParL :  mParalLines)
	{
            aExAllLines.mLines.push_back(aParL.GetLAP(*mCalib));
	    /*
            StdOut()  <<  " =========  SCM=" << aParL.ScoreMatch() << "\n";
	    for (int aKSeg=0 ; aKSeg<2 ; aKSeg++)
	    {
                 cHoughPS aHS = aParL.VHS().at(aKSeg);
		 StdOut() << "   PH=" << aHS.TetaRho() << "\n";

	    }
	    */
	}
	mPhProj.SaveLines(aExAllLines);

        for  (const auto & aLine : aExAllLines.mLines)
        {
             Tpl_AddOneObjReportCSV(*this,mIdExportCSV,aLine);
        }
    }

#if (0)
    for (cHoughPS * aHS1 : mMatchedVPS)
    {
        cHoughPS *aHS2 = aHS1->Matched();
        cOneLineAntiParal aEx1L;
        aEx1L.mAng    = aHS1->DistAnglAntiPar(*aHS2) * mExtrL->Hough().RhoMax() ;
        aEx1L.mWidth  = -( aHS1->DY(*aHS2) + aHS2->DY(*aHS1) ) / 2.0;
        aEx1L.mCumul  = (aHS1->Cumul()+aHS2->Cumul())/2.0;
        aEx1L.mSeg    = aHS1->MidlSeg();
        AddOneReportCSV(mNameReportByLine,{mNameCurIm,ToStr(aEx1L.mAng),ToStr(aEx1L.mWidth),ToStr(aEx1L.mCumul),ToStr(aHS1->RadHom())});

        aExAllLines.mLines.push_back(aEx1L);
    }
    mPhProj.SaveLines(aExAllLines);
    // SaveInFile(aExAllLines,mPhProj.DPPointsMeasures().FullDirOut() + "Segs-"+mNameCurIm + "."+ GlobTaggedNameDefSerial());
#endif

    if (mGenVisu)
       MakeVisu(mNameCurIm);

    // mTimeSeg.Show();

    // delete mCalDUD;
}

void cAppliExtractLine::MakeVisu(const std::string & aNameIm)
{
    std::string aNameTif = LastPrefix(mNameCurIm) + ".tif";
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


    //  [1]  Visu selected max in  direction of gradient
    if (mGenVisu>=2)
    {
         cRGBImage aImV= mExtrL->MakeImageMaxLoc(mTransparencyCont);
         aImV.ToJpgFileDeZoom(mPhProj.DirVisuAppli() + "DetectL_"+ aNameTif,1);
    }

    //  [2]  Visu module of gradient
    if (mGenVisu>=4)
    {
         std::string aNameGrad = mPhProj.DirVisuAppli()+"Grad_" + aNameTif;
         mExtrL->Grad().NormG().DIm().ToFile(aNameGrad);
         // Convert_JPG(aNameGrad,true,90,"jpg");
    }

       
    // [3] Visu  the accum + local maximal
    if (mGenVisu>=3)
    {
         cRGBImage  aVisAccum =  RGBImFromGray(aDAccum,255.0/aVMax);
         for (const auto & aPS : mVPS)
         {
             //aVisAccum.SetRGBrectWithAlpha(ToI(aPS->IndTetaRho())  ,15,cRGBImage::Red,0.5);
	     cPt2dr aC=aPS.IndTetaRho();
	     cPt2dr aSz(7,7);
             aVisAccum.FillRectangle(cRGBImage::Red,ToI(aC-aSz),ToI(aC+aSz),cPt3dr(0.5,1.0,1.0));
         }
         aVisAccum.ToJpgFileDeZoom(mPhProj.DirVisuAppli() + "Accum_" + aNameTif,1);
         aDAccum.ToFile(mPhProj.DirVisuAppli() + "RawAccum_" + aNameTif);
    }
    // [4]  Visu of Image +  Lines
    if (mGenVisu>=1)
    {
         cRGBImage  aVisIm =  cRGBImage::FromFile(mNameCurIm,mZoomImL); // Initialize with image
	 // const auto & aDIm = aVisIm.ImR().DIm();
         for (const auto & aLine : mParalLines)
	 {
             cPt3di aCoul = (aLine.RankMatch()==0) ? cRGBImage::Red : cRGBImage::Green ;
             MMVII_INTERNAL_ASSERT_tiny(mCalib!=nullptr,"Calib mandatory for now in line detect");
	     cOneLineAntiParal aOLAP = aLine.GetLAP(*mCalib);
	     aVisIm.DrawCircle(cRGBImage::Orange,aOLAP.mSeg.P1(),5.0);
	     aVisIm.DrawCircle(cRGBImage::Orange,aOLAP.mSeg.P2(),5.0);

	     for (int aKSeg=0 ; aKSeg<2 ; aKSeg++)
	     {
                 cHoughPS aHS = aLine.VHS().at(aKSeg);

                 cSegment<tREAL8,2> aSeg =  aHS.Seg();

	         tREAL8 aRay=5;
	         cPt2dr aC = Redist(aSeg.PMil() +VUnit(aSeg.V12()*cPt2dr(0,-1))* aRay);
                 aVisIm.DrawCircle(aCoul,aC ,aRay);

                 if (mCalib)
	         {

	             cSegment2DCompiled<tREAL8> aSegC (mCalib->ExtenSegUndistIncluded(false,aSeg));
	             for (tREAL8 aC=0 ; aC< aSegC.N2() ; aC+= 1.0)
	             {
		         cPt2dr  aQ = Redist(aSegC.FromCoordLoc(cPt2dr(aC,0.0))); ; // eventulay make invert distorsion correcion
                         aVisIm.SetRGBPoint(aQ,aCoul);  // print point
                     }
	         }
	     }
	 }
	 std::string aNameLine = mPhProj.DirVisuAppli() + "Lines_" + aNameTif;
	 // convert dont handle well big file, so generate jpg only if zoom=1
	 if (mZoomImL==1)
	    aVisIm.ToJpgFileDeZoom(mPhProj.DirVisuAppli() + "Lines_" + aNameTif,1);
	 else
	    aVisIm.ToFile(aNameLine);
    }
}



int cAppliExtractLine::Exe()
{
    mPhProj.FinishInit();
    InitReportCSV(mNameReportByLine,"csv",true,{"NameIm","Paral","Larg","Score","RadHom"});
    InitReportCSV(mNameReportByIm,"csv",true,{"NameIm","CodeResult"});

    //  Create a report with header computed from type
    Tpl_AddHeaderReportCSV<cOneLineAntiParal>(*this,mIdExportCSV,true);
    // Redirect the reports on folder of result
    SetReportRedir(mIdExportCSV,mPhProj.DPPointsMeasures().FullDirOut());

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
