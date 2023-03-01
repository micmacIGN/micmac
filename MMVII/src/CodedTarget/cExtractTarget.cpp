#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"
#include "src/Matrix/MMVII_EigenWrap.h"
#include <random>
#include <bitset>
#include <time.h>
#include <typeinfo>
#include <iostream>
#include <fstream>


#include "include/MMVII_PhgrDist.h"
using namespace NS_SymbolicDerivative;

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */
#define PI 3.14159265

// Test git branch

namespace MMVII
{
void TestParamTarg();



namespace  cNS_CodedTarget
{

/*  *********************************************************** */
/*                                                              */
/*                       cDCT                                   */
/*                                                              */
/*  *********************************************************** */


cDCT::cDCT(const cPt2dr aPtR,eResDCT aState) :
   mGT          (nullptr),
   mPt          (aPtR),
   mState       (aState),
   mScRadDir    (1e5),
   mSym         (1e5),
   mBin         (1e5),
   mRad         (1e5),
   mRecomputed  (false)

{
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

	void DoAllMatchOnGT();
	/// compute direction of ellipses
        void ExtractDir(cDCT & aDCT);
	/// Print statistique initial

        int ExeOnParsedBox() override;

        void DoExtract();
        void ShowStats(const std::string & aMes) ;
        void MarkDCT() ;
        void SelectOnFilter(cFilterDCT<tREAL4> * aFilter,bool MinCrown,double aThrS,eResDCT aModeSup);
        bool analyzeDCT(cDCT*, const cDataIm2D<float> &);                        ///< Analyze a potential target
        int decodeTarget(tDataImT &, double, double, std::string&, bool);        ///< Decode a potential target
        bool markImage(tDataImT &, cPt2di, int, int);                            ///< Plot mark on gray level image
        std::vector<cPt2dr> solveIntersections(cDCT*, double*);
        std::vector<cPt2dr> extractButterflyEdge(const cDataIm2D<float> &, cDCT*);
        std::vector<cPt2dr> extractButterflyEdgeOld(const cDataIm2D<float> &, cDCT*);
        void exportInXml(std::vector<cDCT*>);
        void plotDebugImage(cDCT*, const cDataIm2D<float>&);
        tImTarget generateRectifiedImage(cDCT*, const cDataIm2D<float>&);
        double ellipseResidual(std::vector<cPt2dr>, std::vector<double>);        ///< Computes pixel residual of ellipse fit



        // ---------------------------------------------------------------------------------
        // Fonctions à déplacer dans un autre fichier
        // ---------------------------------------------------------------------------------
        void printMatrix(MatrixXd);
        bool plotSafeRectangle(cRGBImage, cPt2di, double, cPt3di, int, int, double);

        int cartesianToNaturalEllipse(double*, double*);                         ///< Convert (A,B,C,D,E,F) ellipse parameters to (x0,y0,a,b,theta)
        cPt2dr generatePointOnEllipse(double*, double, double);                  ///< Generate point on ellipse from natural parameters
        std::vector<cPt2dr> generatePointsOnEllipse(double*, unsigned, double);  ///< Generate points on ellipse from natural parameters
        void translateEllipse(double*, cPt2dr);                                  ///< Ellipse translation in cartesian coordinates
        void benchEllipse();                                                     ///< Bench test for ellipse fit
        void benchEllipseR();                                                    ///< Bench test for ellipse fit
        void benchAffinity();                                                    ///< Bench text for affinity fit
        void plotCaseR(std::vector<cPt2dr>, double*);                            ///< Plot ellipse fit in R for debugging
        bool printDebug(std::string, bool);                                      ///< Print debug for focus on a center
        bool printDebug(std::string, double, double);                            ///< Print debug for focus on a center

        int fitEllipse (std::vector<cPt2dr>, cPt2dr, bool, double*);             ///< Least squares estimation of an ellipse from 2D points
        int fitFreeEllipse(std::vector<cPt2dr>, double*);                        ///< Least squares estimation of a floating ellipse from 2D points
        int fitConstrainedEllipse(std::vector<cPt2dr>, cPt2dr, double*);         ///< Least squares estimation of a constrained ellipse from 2D points

        std::vector<double> estimateRectification(std::vector<cPt2dr>, double);
        std::vector<double> estimateAffinity(std::vector<cPt2dr>, std::vector<cPt2dr>);
        bool isValidAffinity(std::vector<double>);
        cPt2dr applyAffinity(cPt2dr, std::vector<double>);
        cPt2dr applyAffinityInv(cPt2dr, std::vector<double>);


	std::string mNameTarget;

	cParamCodedTarget        mPCT;
    double                   mDiamMinD;
    bool                     mConstrainCenter;
	cPt2dr                   mRaysTF;
	cPt2di                   mTestCenter;

        std::vector<eDCTFilters> mTestedFilters;

        cImGrad<tREAL4>  mImGrad;  ///< Result of gradient
        double   mRayMinCB;        ///< Ray Min CheckBoard
        double   mR0Sym;           ///< R min for first very quick selection on symetry
        double   mR1Sym;           ///< R max for first very quick selection on symetry
        double   mRExtreSym;       ///< R to compute indice of local maximal of symetry
        double   mTHRS_Sym;        ///< Threshold for symetricity


        double mTHRS_Bin;
        void FilterDCTOk();

        std::vector<cDCT*>  mVDCT; ///< vector of detected target
        std::vector<cDCT*>  mVDCTOk; ///< sub vector of detected target Ok,
        cResSimul           mGTResSim; ///< result of simulation when exist


        cRGBImage      mImVisu;
        std::string    mPatExportF;
        bool           mWithGT;
        double         mDMaxMatch;

        bool mTest;               ///< Test option for debugging program
        bool mRecompute;          ///< Recompute affinity for size adaptation
        std::string mXml;         ///< Print xml output in file
        int mDebugPlot;           ///< Debug plot code (binary)
        double mMaxEcc;           ///< Max. eccentricity of detected ellispes
        bool mSaddle;             ///< Prefiletring with Saddle test
        double mMargin;           ///< Percent margin of butterfly edge used for fit



        std::vector<double>  mParamBin;
        cParamCodedTarget spec;
        std::string mOutput_folder;
        int mTargetCounter;
        std::vector<double> mTransfo;

        std::vector<std::string> mToRestrict;
        bool mFailure;            ///< Plot also failures on rectified folder
        double mTolerance;        ///< Tolerance for decoding bits (in %)

        std::vector<cPt2dr> mPoints;
        std::bitset<10> mBitsPlotDebug;
        std::string mFlagDebug;
        double mLineWidthDebug;
        int mLetter;

        double mErrAvgGT;
        double mErrMaxGT;
        double mCompGT;
        std::string mGroundTruthFile;
        double mStepButterfly;
        double mGradButterfly;
};



/* *************************************************** */
/*                                                     */
/*              cAppliExtractCodeTarget                */
/*                                                     */
/* *************************************************** */

cAppliExtractCodeTarget::cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(10000,10000),cPt2di(300,300),false), // static_cast<cMMVII_Appli & >(*this))
   mDiamMinD        (40.0),
   mConstrainCenter (false),
   mRaysTF          ({4,8}),
   mTestCenter      (cPt2di(-1,-1)),
   mImGrad          (cPt2di(1,1)),
   mR0Sym           (3.0),
   mR1Sym           (8.0),
   mRExtreSym       (9.0),
   mTHRS_Sym        (0.7),
   mTHRS_Bin        (0.6),
   mImVisu          (cPt2di(1,1)),
   mPatExportF      ("XXXXXXXXX"),
   mWithGT          (false),
   mDMaxMatch       (2.0),
   mTest            (false),
   mRecompute       (false),
   mXml             (""),
   mMaxEcc          (1e9),
   mSaddle          (false),
   mMargin          (0.20),
   mToRestrict      ({}),
   mFailure         (false),
   mTolerance       (0.0),
   mFlagDebug       (""),
   mLineWidthDebug  (1.0),
   mLetter          (1),
   mErrAvgGT        (0.0),
   mErrMaxGT        (0.0),
   mCompGT          (0.0),
   mGroundTruthFile (""),
   mStepButterfly   (0.05),
   mGradButterfly   (0)
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
                    << AOpt2007(mDiamMinD, "DMD","Diam min for detect",{eTA2007::HDV})
                    << AOpt2007(mConstrainCenter, "CC","Constrain centers for ellipse fit",{eTA2007::HDV})
                    << AOpt2007(mRaysTF, "RayTF","Rays Min/Max for testing filter",{eTA2007::HDV,eTA2007::Tuning})
                    << AOpt2007(mTestCenter, "TestCenter","Test program only on a center +/- 2 px",{eTA2007::HDV,eTA2007::Tuning})
                    << AOpt2007(mPatExportF, "PatExpF","Pattern export filters" ,{AC_ListVal<eDCTFilters>(),eTA2007::HDV})
                    << AOpt2007(mTest, "Test", "Test for Ellipse Fit", {eTA2007::HDV})
                    << AOpt2007(mParamBin, "BinF", "Param for binary filter", {eTA2007::HDV})
                    << AOpt2007(mRecompute, "Adjust", "Recompute directions with adjusted size", {eTA2007::HDV})
                    << AOpt2007(mXml, "Xml", "Print xml outout in file", {eTA2007::HDV})
                    << AOpt2007(mDebugPlot, "Debug", "Plot debug image with options", {eTA2007::HDV})
                    << AOpt2007(mMaxEcc, "MaxEcc", "Max. eccentricity of targets", {eTA2007::HDV})
                    << AOpt2007(mSaddle, "Saddle", "Prefiltering with saddle test", {eTA2007::HDV})
                    << AOpt2007(mMargin, "Margin", "Margin on butterfly edge for fit", {eTA2007::HDV})
                    << AOpt2007(mToRestrict, "Restrict", "List of codes to restrict on", {eTA2007::HDV})
                    << AOpt2007(mFailure, "Failure", "Plot also failures in RectifTargets", {eTA2007::HDV})
                    << AOpt2007(mTolerance, "Tolerance", "Tolerance (in %) for reading bits", {eTA2007::HDV})
                    << AOpt2007(mLineWidthDebug, "Line", "Size of lines in debug plot", {eTA2007::HDV})
                    << AOpt2007(mLetter, "Letter", "Size of letters in debug plot", {eTA2007::HDV})
                    << AOpt2007(mGroundTruthFile, "GT", "Ground truth file (if any)", {eTA2007::HDV})
	  );
   ;
}

void cAppliExtractCodeTarget::ShowStats(const std::string & aMes)
{
   int aNbOk=0;
   int aNbGTOk=0;
   double aSomDist=0;
   std::vector<double> aVDistGT;
   for (const auto & aR : mVDCT)
   {
      if (aR->mState == eResDCT::Ok)
      {
         aNbOk++;
	 if (aR->mGT)
	 {
            aVDistGT.push_back(Norm2(aR->mPt-aR->mGT->mC));
            aSomDist += aVDistGT.back();
            aNbGTOk++;
	 }
      }
   }
   StdOut() <<  aMes << " NB DCT = " << aNbOk << " PropAll " << (double) aNbOk / (double) APBI_DIm().NbElem() ;

   if (aNbGTOk)
   {
       size_t aNbGtAll = mGTResSim.mVG.size();
       StdOut()  <<  " PropGT=" << double(aNbGTOk)/  aNbGtAll
	       << " AvgDist=" << aSomDist/ aNbGTOk
	       // << "     ** D50,75=" << NC_KthVal(aVDistGT,0.5) << " " << NC_KthVal(aVDistGT,0.75)
       ;
   }

   StdOut()   << "\n";
}


void cAppliExtractCodeTarget::MarkDCT()
{

     for (auto & aDCT : mVDCT)
     {
          cPt3di aCoul (-1,-1,-1);

          if (aDCT->mState == eResDCT::Ok)      aCoul =  cRGBImage::Green;
          if (aDCT->mState == eResDCT::Divg)    aCoul =  cRGBImage::Red;
          if (aDCT->mState == eResDCT::LowSym)  aCoul =  cRGBImage::Yellow;               // High symmetry
          if (aDCT->mState == eResDCT::LowBin)  aCoul =  cRGBImage::Blue;                 // High binarity
          if (aDCT->mState == eResDCT::LowRad)  aCoul =  cRGBImage::Cyan;                 // High radiality
          if (aDCT->mState == eResDCT::LowSymMin)  aCoul =  cRGBImage::Magenta;           // High symmetry

          if (aCoul.x() >=0){
             mImVisu.SetRGBrectWithAlpha(aDCT->Pix(), 1, aCoul, 0.0);
          }

          /*

          if (aDCT->mState == eResDCT::Ok) StdOut()        << "OK "        << aDCT->mPt.x() << " " << aDCT->mPt.y() << "\n";
          if (aDCT->mState == eResDCT::Divg) StdOut()      << "DIVG "      << aDCT->mPt.x() << " " << aDCT->mPt.y() << "\n";
          if (aDCT->mState == eResDCT::LowSym) StdOut()    << "LOWSYM "    << aDCT->mPt.x() << " " << aDCT->mPt.y() << "\n";
          if (aDCT->mState == eResDCT::LowBin) StdOut()    << "LOWBIN "    << aDCT->mPt.x() << " " << aDCT->mPt.y() << "\n";
          if (aDCT->mState == eResDCT::LowRad) StdOut()    << "LOWRAD "    << aDCT->mPt.x() << " " << aDCT->mPt.y() << "\n";
          if (aDCT->mState == eResDCT::LowSymMin) StdOut() << "LOWSYMMIN " << aDCT->mPt.x() << " " << aDCT->mPt.y() << "\n";

          */

     }
}

void cAppliExtractCodeTarget::SelectOnFilter(cFilterDCT<tREAL4> * aFilter,bool MinCrown,double aThrS,eResDCT aModeSup)
{
  mVDCTOk.clear();
  for (auto & aDCT : mVDCT)
  {
      if (aDCT->mState == eResDCT::Ok)
      {
         double aSc =  MinCrown ?   aFilter->ComputeValMaxCrown(aDCT->mPt,aThrS)   : aFilter->ComputeVal(aDCT->mPt);
         aFilter->UpdateSelected(*aDCT);
         if (aSc>aThrS)
            aDCT->mState = aModeSup;
         else
            mVDCTOk.push_back(aDCT);
      }
  }
  ShowStats(E2Str(aFilter->ModeF()) + " Min=" +ToStr(MinCrown));

  if (std::find(mTestedFilters.begin(),mTestedFilters.end(),aFilter->ModeF()) != mTestedFilters.end())
  {
      cIm2D<tREAL4>  aImF = aFilter->ComputeIm();
      std::string aName = "TestDCT_" +  E2Str(aFilter->ModeF())  + "_" + Prefix(mNameIm) + ".tif";
      aImF.DIm().ToFile(aName);
  }
  delete aFilter;
}

void cAppliExtractCodeTarget::MatchOnGT(cGeomSimDCT & aGSD)
{
     // strtucture for extracting min
     cWhichMin<cDCT*,double>  aWMin(nullptr,1e10);

     for (auto aPtrDCT : mVDCT)
         aWMin.Add(aPtrDCT,SqN2(aPtrDCT->mPt-aGSD.mC));

     if (aWMin.ValExtre() < Square(mDMaxMatch))
     {
     	aGSD.mResExtr = aWMin.IndexExtre(); // the simul memorize its detected
        aGSD.mResExtr->mGT =& aGSD;         // the detected memorize its ground truth
     }
     else
     {
     }
}

void cAppliExtractCodeTarget::DoAllMatchOnGT()
{
     if (mWithGT)
     {
        int aNbGTMatched = 0;
        for (auto & aGSD : mGTResSim.mVG)
	{
             MatchOnGT(aGSD);
	     if (aGSD.mResExtr )
		aNbGTMatched++;
	     else
	     {
                 StdOut() << " UNMATCH000 at " << aGSD.mC << "\n";
	     }
	}

	StdOut()  << "GT-MATCHED : %:" << (100.0*aNbGTMatched) /mGTResSim.mVG.size() << " on " << mGTResSim.mVG.size() << " total-GT\n";
     }
}

void  cAppliExtractCodeTarget::DoExtract(){

    if (mToRestrict.size() > 0){
        StdOut() << "List of target codes to keep: \n";
        for (unsigned i=0; i<mToRestrict.size(); i++){
            StdOut() << mToRestrict.at(i) << " ";
        }
        StdOut() << "\n";
    }
    StdOut() << "------------------------------------------------------------------\n";

    spec.InitFromFile(mNameTarget);

    // --------------------------------------------------------------------------------------------------------
    // Get debug plot options
    // --------------------------------------------------------------------------------------------------------
    // [0000] 0000000000 default: no plot
    // [0001] 0000000001 plot only candidates after filtering operations (magenta pixels)
    // [0002] 0000000010 plot only center of detected targets (green pixels)
    // [0004] 0000000100 plot only transitions on circles around candidate targets (yellow pixels)
    // [0008] 0000001000 plot only axis lines of dected chessboard patterns (green lines)
    // [0016] 0000010000 plot only data point for ellipse fit (cyan pixels)
    // [0032] 0000100000 plot only fitted ellipse (red lines)
    // [0064] 0001000000 plot only intersections between ellipse and axes (blue pixels)
    // [0128] 0010000000 plot only detected target frames (blue lines)
    // [0256] 0100000000 plot detected target code name (cyan characters)
    // [0512] 1000000000 plot rectified images with detected codes (RectifTargets directory)
    // [1023] 1111111111 plot all features above in debug images
    // --------------------------------------------------------------------------------------------------------
    mBitsPlotDebug = std::bitset<10>(mDebugPlot);
    if (mDebugPlot){
        StdOut() << "\n------------------------------------------------------------------\n";
        StdOut() << "DEBUG PLOT:\n";
        StdOut() << "------------------------------------------------------------------\n";
        if (mBitsPlotDebug[0]) StdOut() << "* CANDIDATES AFTER FILTERING\n";
        if (mBitsPlotDebug[1]) StdOut() << "* CENTER OF DETECTED TARGETS\n";
        if (mBitsPlotDebug[2]) StdOut() << "* TRANSITIONS ON CIRCLES AROUND TARGETS\n";
        if (mBitsPlotDebug[3]) StdOut() << "* AXIS LINES ON CHESSBOARDS\n";
        if (mBitsPlotDebug[4]) StdOut() << "* DATA POINTS TO FIT ELLIPSE\n";
        if (mBitsPlotDebug[5]) StdOut() << "* FITTED ELLIPSE ON CHESSBOARD\n";
        if (mBitsPlotDebug[6]) StdOut() << "* INTERSECTIONS BETWEEN ELLIPSE AND AXES\n";
        if (mBitsPlotDebug[7]) StdOut() << "* DETECTED TARGET FRAMES\n";
        if (mBitsPlotDebug[8]) StdOut() << "* DETECTED TARGET CODE NAMES\n";
        if (mBitsPlotDebug[9]) StdOut() << "* RECTIFIED IMAGES OF DETECTED TARGETS\n";
        StdOut() << "------------------------------------------------------------------\n";
    }


    // --------------------------------------------------------------------------------------------------------

     tDataIm &  aDIm = APBI_DIm();
     tIm        aIm = APBI_Im();
     mImVisu =   RGBImFromGray(aDIm);
     // mNbPtsIm = aDIm.Sz().x() * aDIm.Sz().y();



     if (!mSaddle) //  Case prefiltring by symetry
     {
          // [1]   Extract point that are extremum of symetricity
         //    [1.1]   extract integer pixel
         cIm2D<tREAL4>  aImSym = ImSymetricity(false,aIm,mRayMinCB*0.4,mRayMinCB*0.8,0);  // compute fast symetry

         if (1)
         {
            aImSym.DIm().ToFile("TestDCT_SYMINIT_SimulTarget_test.tif");
         }

         cResultExtremum aRExtre(true,false);              //structire for result of extremun , compute min not max
         ExtractExtremum1(aImSym.DIm(),aRExtre,mRExtreSym);  // do the extraction

         // [1.2]  afine to real coordinate by fiting quadratic models
         cAffineExtremum<tREAL4> anAffEx(aImSym.DIm(),2.0);
         for (const auto & aPix : aRExtre.mPtsMin)
         {
             eResDCT aState = eResDCT::Ok;
             cPt2dr aPtR =  anAffEx.StdIter(ToR(aPix),1e-2,3);
             if ( (anAffEx.Im().Interiority(aPix)<20) || (Norm2(aPtR-ToR(aPix))>2.0)  )
                aState = eResDCT::Divg;

              mVDCT.push_back(new cDCT(aPtR,aState));
         }
         DoAllMatchOnGT();
         ShowStats("Init ");
     }
     else  // Case prefiltering by Saddle
     {
         // Set for interior Maybe to adapt  DRONE
         double aRaySaddle = mDiamMinD / 4.0;
         double aDistSaddleExtre = mDiamMinD / 4.0;  //   Div 3 generate one false neg
         double aThrSadCPT = 0.45;  //   Div 3 generate one false neg

	 // 1.1 compute saddle images
         StdOut()  <<  "------- BEGIN SADDLE-------------\n";
         auto [aImDif,aImCpt] =  FastComputeSaddleCriterion(APBI_Im(),aRaySaddle);
         StdOut()  <<  "------- END SADDLE-------------\n";

	 if (0) /// Save image sadles for visu
	 {
	    aImDif.DIm().ToFile("SadleDif.tif");
	    aImCpt.DIm().ToFile("SadleCpt.tif");
	 }

	 // 1.2  select point that are extrema of saddle-dif
         cResultExtremum aRExtre(false,true);  //structire for result of extremun , compute max and not min
         ExtractExtremum1(aImDif.DIm(),aRExtre,aDistSaddleExtre);  // do the extraction

         for (const auto & aPix : aRExtre.mPtsMax)
         {
             eResDCT aState = eResDCT::Ok;
	     cPt2dr aPtR = ToR(aPix);
             if  (APBI_DIm().Interiority(aPix)<20)  // remove point closde to border of image
                aState = eResDCT::Divg;
	     else
                mVDCT.push_back(new cDCT(aPtR,aState));
         }
         DoAllMatchOnGT();  // Match on GT
         ShowStats("SadleDiffRel ");

	 std::vector<double>  aVCPT_GT;  // cpt-sadle for ground-truh to check values
	 std::vector<double>  aVCPT_Glob;  // cpt-sadle for all points

	 // 1.3  refine position by /
	 // WARN  RAY=5 generate many divg, see if can already estimate the ray at this Step ??
	 cCalcSaddle  aCSad(3.0,0.5);

	 for (auto & aPtrDCT :  mVDCT )
	 {
             tREAL8 aCpt = aImCpt.DIm().GetV(aPtrDCT->Pix());
	     if (aPtrDCT->mGT)
	     {
                 aVCPT_GT.push_back(aCpt);
	     }
	     aVCPT_Glob.push_back(aCpt);

	     if (aCpt<aThrSadCPT)
                aPtrDCT->mState = eResDCT::LowSadleRel;
	     if (aPtrDCT->mState ==eResDCT::Ok)
	     {
                  aCSad.RefineSadlePointFromIm(APBI_Im(),*aPtrDCT);
	     }
	 }
	 for (double aVal : {0.4,0.45,0.5})
	 {
	     StdOut() << "   --- SadCPT : " << aVal  << " PropGT " <<  Rank(aVCPT_GT,aVal) << " PropStd " <<  Rank(aVCPT_Glob,aVal)<< "\n";
	 }
         ShowStats("SadleCpt ");
     }

     //   ====   Symetry filters ====
     /*   Not sure this very usefull as symetry is done
           *   optionnaly at the begining for prefiltering  (it symetry or saddle)
	   *   always as a post fiter
     for (auto & aDCT : mVDCT){
        if (aDCT->mState == eResDCT::Ok){
           aDCT->mSym = aImSym.DIm().GetV(aDCT->Pix());
           if (aDCT->mSym > mTHRS_Sym){
              aDCT->mState = eResDCT::LowSym;
           }
        }
     }
     ShowStats("LowSym");
     */


     cParamAllFilterDCT aGlobParam;


     //   ==== Binarity filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocBin(aIm,aGlobParam),false,mTHRS_Bin,eResDCT::LowBin);


     //   ==== Radial filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocRad(mImGrad,aGlobParam),false,0.9,eResDCT::LowRad);


     //   ==== Min of symetry ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocSym(aIm,aGlobParam),true,0.8,eResDCT::LowSym);


     mVDCTOk.clear();


    for (auto aPtrDCT : mVDCT){

        // -------------------------------------------
        // TEST CENTRAGE SUR UNE CIBLE
        // -------------------------------------------
        if (mTestCenter.x() != -1){
            if (abs(aPtrDCT->Pix().x() - mTestCenter.x()) > 2) continue;
            if (abs(aPtrDCT->Pix().y() - mTestCenter.y()) > 2) continue;
        }

        if (aPtrDCT->mState == eResDCT::Ok){
            if (!TestDirDCT(*aPtrDCT,APBI_Im(), 0.4*mRayMinCB, 0.8*mRayMinCB, aPtrDCT->mDetectedVectors)){
                aPtrDCT->mState = eResDCT::BadDir;
            }else{
                mVDCTOk.push_back(aPtrDCT);
            }
        }
    }


     ShowStats("ExtractDir");
     StdOut()  << "MAINTAINED " << mVDCTOk.size() << "\n";


     // ----------------------------------------------------------------------------------------------
     // [512] 1000000000 plot rectified images with detected codes (RectifTargets directory)
     // ----------------------------------------------------------------------------------------------
     mOutput_folder = "RectifTargets";
     if (mBitsPlotDebug[9]) CreateDirectories(mOutput_folder, true);


    mTargetCounter = 0;


    for (auto aDCT : mVDCTOk){
        if ((analyzeDCT(aDCT, aDIm)) || (mTestCenter.x() != -1)){
            plotDebugImage(aDCT, aDIm);
        }
    }

    // ------------------------------------------------
    // Control with ground truth (if any)
    // ------------------------------------------------
    int NGT = mGTResSim.mVG.size();
    if ((mGroundTruthFile != "") && (NGT != 0)){
        double rmse = sqrt(mErrAvgGT/NGT);
        double comp = floor(mCompGT/NGT*1e5)/1e3;
        StdOut() << "GROUND TRUTH COMPARISON:\n";
        StdOut() << "   *PROP: " << comp << " %\n";
        StdOut() << "   *RMSE: " << rmse << " px\n";
        StdOut() << "   *MAX.: " << mErrMaxGT << " px\n";
    }

    // ------------------------------------------------
    // Xml output (if needed)
    // ------------------------------------------------
    if (mXml != "") exportInXml(mVDCTOk);

    // ------------------------------------------------
    // Plot debug if needed
    // ------------------------------------------------
    if (mDebugPlot) mImVisu.ToFile("VisuCodeTarget.tif");

}


// ---------------------------------------------------------------------------
// Function to export result in xml
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::exportInXml(std::vector<cDCT*> mVDCTOk){
    std::string xml_output  = "    <MesureAppuiFlottant1Im>\n";
    xml_output += std::string("        <NameIm>");
        xml_output += std::string(mNameIm);
        xml_output += std::string("</NameIm>\n");
        for (auto aDCT : mVDCTOk){
            if ((aDCT->mDecodedName != "") && (aDCT->mDecodedName.substr(0,2) != "NA")){
                xml_output += "        <OneMesureAF1I>\n";
                xml_output += "            <NamePt>" +  aDCT->mDecodedName + "</NamePt>\n";
                xml_output += "            <PtIm>" + std::to_string(aDCT->mRefinedCenter.x());
                xml_output += " " + std::to_string(aDCT->mRefinedCenter.y()) +"</PtIm>\n";
                xml_output += "        </OneMesureAF1I>\n";
            }
        }
        xml_output += "     </MesureAppuiFlottant1Im>\n";
        std::ofstream xml_file;
        xml_file.open (mXml, std::ios_base::app);
        xml_file << xml_output;
        xml_file.close();

}



// ---------------------------------------------------------------------------
// Function to test if a predetected DCT candidate is valid
// ---------------------------------------------------------------------------
bool cAppliExtractCodeTarget::analyzeDCT(cDCT* aDCT, const cDataIm2D<float> & aDIm){

    aDCT->mFinalState = false;
    if ((mTestCenter.x() != -1) &&  (mTestCenter.y() != -1)) mFlagDebug = "   [DEBUG TEST CENTER] ";

    // ---------------------------------------------------------------------------
    // Functions parameters (to tune)
    // ---------------------------------------------------------------------------
    int px_binarity = 25;      // Threshold on binarity
    int limit_border = 30;     // Target not considered if that close to border
    // ---------------------------------------------------------------------------

    // --------------------------------------------------------------------------------------------------------
    // Build list of codes from specification file
    // --------------------------------------------------------------------------------------------------------

    if (mFlagDebug != "") StdOut() << mFlagDebug << "Focus on center: " << mTestCenter << "\n";

    cPt2di center = aDCT->Pix();

    // -------------------------------------------
    // Test on binarity of target
    // -------------------------------------------
    double binarity = aDCT->mVWhite - aDCT->mVBlack;
    if (!printDebug("Binarity test", binarity, px_binarity)) return false;

    // ----------------------------------------------------------------------------------------------
    // [001] 0000000001 plot only candidates after filtering operations (magenta pixels)
    // ----------------------------------------------------------------------------------------------
    if(mBitsPlotDebug[0]) mImVisu.SetRGBrectWithAlpha(center, 1, cRGBImage::Magenta, 0.0);

    // -----------------------------------------------------------------
    // Testing borders
    // -----------------------------------------------------------------
    bool lim =  (center.x() > limit_border) && (center.y() > limit_border);
    lim = lim && (aDIm.Sz().x()-center.x() > limit_border) && (aDIm.Sz().y()-center.y() > limit_border);
    if (!printDebug("Border limit", lim)) return false;


    // Butterfly edge extraction
    mPoints = extractButterflyEdge(aDIm, aDCT);
    double min_imposed = 0.8*2*(1-2*mMargin)/mStepButterfly;
    if (!printDebug("Butterfly edge extraction size", mPoints.size(), min_imposed)) return false;

    // -----------------------------------------------------------------
    // Ellipse fit
    // -----------------------------------------------------------------
    double param[6]; double ellipse[5];
    bool ellipseOk = (fitEllipse(mPoints, aDCT->mPt, mConstrainCenter, param) >= 0);

    if (!printDebug("Ellipse fit", ellipseOk)) return false;
    cartesianToNaturalEllipse(param, ellipse);

    // Invalid ellipse fit
    if (!printDebug("Ellipse cartesian parameterization", (ellipse[0] == ellipse[0])))  return false;
    if (!printDebug("Ellipse eccentricity", mMaxEcc, ellipse[2]/ellipse[3]))            return false;

    // -----------------------------------------------------------------

    // Generate ellipse for plot
    for (double t=0; t<2*PI; t+=0.01){
        cPt2dr aPoint = generatePointOnEllipse(ellipse, t, 0.0);
        cPt2di pt = cPt2di(aPoint.x(), aPoint.y());
        aDCT->mDetectedEllipse.push_back(pt);
    }

    // Solve intersections
    aDCT->mDetectedCorners = solveIntersections(aDCT, param);
    if (!printDebug("Ellipse-cross intersections", (aDCT->mDetectedCorners.size() == 4))) return false;

    // Affinity first estimation
    mTransfo = estimateRectification(aDCT->mDetectedCorners, spec.mChessboardAng);
    aDCT->mSizeTargetEllipse = std::min(ellipse[2], ellipse[3]);

    // ======================================================================
    // Recomputing directions and intersections if needed
    // ======================================================================

    if ((aDCT->mSizeTargetEllipse > mRayMinCB) && (mRecompute)){

        // Recomputing directions if needed
        double min_ray_adjust = 0.4*mRayMinCB;
        double max_ray_adjust = 0.8*aDCT->mSizeTargetEllipse;

        TestDirDCT(*aDCT, APBI_Im(), min_ray_adjust, max_ray_adjust, aDCT->mDetectedVectors);

        // Recomputing intersections if needed
        aDCT->mDetectedCorners = solveIntersections(aDCT, param);

        // Recomputing affinity if needed (and if possible)
        if (aDCT->mDetectedCorners.size() != 4) return false;
        mTransfo = estimateRectification(aDCT->mDetectedCorners, spec.mChessboardAng);
        aDCT->mRecomputed = true;
    }

    // Affinity estimation test
    bool validAff = isValidAffinity(mTransfo);
    if (!printDebug("Affinity estimation", validAff)) return false;

    // Ellipse fit residual test (requires affinity estimation);
    double rmse_px = ellipseResidual(mPoints, mTransfo)/600*mDiamMinD;
    if (!printDebug("Ellipse fit residual", 1, rmse_px)) return false;

    // Control on center position
    double x_centre_moy = (ellipse[0] + aDCT->mPt.x())/2.0;
    double y_centre_moy = (ellipse[1] + aDCT->mPt.y())/2.0;
    double dx = std::round(100*std::abs(ellipse[0] - aDCT->mPt.x())/2.0)/100;
    double dy = std::round(100*std::abs(ellipse[1] - aDCT->mPt.y())/2.0)/100;

    double dxy = sqrt(dx*dx + dy*dy);
    if (!printDebug("Target center coincidence", 5.0, dxy)) return false;

    // ======================================================================
    // Image generation
    // ======================================================================

    tImTarget aImT = generateRectifiedImage(aDCT, aDIm);

    // ======================================================================
    // Decoding
    // ======================================================================

    std::string code_binary;
    int code = decodeTarget(aImT.DIm(), aDCT->mVWhite, aDCT->mVBlack, code_binary, spec.mModeFlight);

    std::string chaine = "NA";

    if (code != -1){
        chaine = spec.NameOfBinCode(code);
        if (mToRestrict.size() > 0){
            if (chaine != "NA"){
                if (std::find(mToRestrict.begin(), mToRestrict.end(), chaine) == mToRestrict.end()){
                    return false;
                }
            }
        }
    }

    std::string name_file = "target_" + chaine + ".tif";
    aDCT->mDecodedName = chaine;

    aDCT->mRefinedCenter = cPt2dr(x_centre_moy, y_centre_moy);
    if (chaine == "NA"){
        aDCT->mDecodedName = "NA_"+std::to_string(mTargetCounter);
        name_file = "failure_" + std::to_string(mTargetCounter) + ".tif";
        printDebug("Target decoding", false);
        if (!mFailure)  return false;
    }

    // --------------------------------------------------------------------------------
    // Begin print console
    // --------------------------------------------------------------------------------
    StdOut() << " [" << mTargetCounter << "]" << " Centre: [";
    StdOut() << x_centre_moy << " +/- " << dx  << ", " << y_centre_moy << " +/- " << dy << "]  -  ";
    StdOut() << code_binary;
    StdOut() << "  ->  " << name_file;
    mTargetCounter ++;
    // --------------------------------------------------------------------------------
    // End print console
    // --------------------------------------------------------------------------------


    // ----------------------------------------------------------------------------------------------
    // [512] 1000000000 plot rectified images with detected codes (RectifTargets directory)
    // ----------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[9]) aImT.DIm().ToFile(mOutput_folder+ "/" + mNameIm + "_" + name_file);


    // ----------------------------------------------------------------------------------------------
    // Comparison with ground truth (if any)
    // ----------------------------------------------------------------------------------------------
    for (auto & aGSD : mGTResSim.mVG){
        if (aGSD.name == chaine){
            double error = sqrt(pow(aGSD.mC.x()-x_centre_moy, 2) + pow(aGSD.mC.x()-x_centre_moy, 2));
            if (error < 10){
                mErrAvgGT += error*error;
                mErrMaxGT = std::max(mErrMaxGT, error);
                mCompGT += 1;
            }
        }
    }

    StdOut() << "\n";

    aDCT->mFinalState = true;
    return true;

}


// ---------------------------------------------------------------------------
// Functions to print debug
// ---------------------------------------------------------------------------
bool cAppliExtractCodeTarget::printDebug(std::string name, bool pass){
    std::string pass_status = pass? "ok ":"failed ";
    if (mFlagDebug != "") StdOut() << mFlagDebug << name << ": " << pass_status << "\n";
    return pass;
}
bool cAppliExtractCodeTarget::printDebug(std::string name, double value, double threshold){
    std::string pass_status = value > threshold? "ok ":"failed ";
    if (mFlagDebug != ""){
        StdOut() << mFlagDebug << name << ": " << pass_status << " [";
        StdOut() << value << " >= " << threshold << "]\n";
    }
    return (pass_status == "ok ");
}


// ---------------------------------------------------------------------------
// Function to generate image of rectified targets
// ---------------------------------------------------------------------------
tImTarget cAppliExtractCodeTarget::generateRectifiedImage(cDCT* aDCT, const cDataIm2D<float>& aDIm){

    double irel, jrel;
    int Ni = spec.mModeFlight?640:600;
    int Nj = spec.mModeFlight?1280:600;
    tImTarget aImT(cPt2di(Ni, Nj));
    tDataImT & aDImT = aImT.DIm();

    if (!spec.mModeFlight){

        // -------------------------------------------------------------
        // Standard case
        // -------------------------------------------------------------

        for (int i=0; i<Ni; i++){
            for (int j=0; j<Nj; j++){
                irel = +6*((double)i-Ni/2.0)/Ni;
                jrel = -6*((double)j-Nj/2.0)/Nj;
                cPt2dr p = applyAffinity(cPt2dr(irel, jrel), mTransfo);
                if ((p.x() < 0) || (p.y() < 0) || (p.x() >= aDIm.Sz().x()-1) || (p.y() >= aDIm.Sz().y()-1)){
                    continue;
                }
                aDImT.SetV(cPt2di(i,j), aDIm.GetVBL(p));
                if ((i == 0) || (j == 0) || (i == Ni-1) || (j == Nj-1)){
                    aDCT->mDetectedFrame.push_back(cPt2di(p.x(), p.y()));
                }
            }
        }

        // Corners and center
        markImage(aDImT, cPt2di(300, 300), 3, 0);
        markImage(aDImT, cPt2di(300, 300), 2, 255);

    }else{

        // -------------------------------------------------------------
        // Aerial case
        // -------------------------------------------------------------

        for (int i=0; i<Ni; i++){
            for (int j=0; j<Nj; j++){
                irel =  1.00*3.2*((double)i-Ni/2.0)/Ni;              //  !!!!!!!!! WARNING: temp scale factor for Patricio * 1.35
                jrel = -6.3*((double)j-Nj/2.0)/Nj;
                cPt2dr p = applyAffinity(cPt2dr(irel, jrel), mTransfo);
                if ((p.x() < 0) || (p.y() < 0) || (p.x() >= aDIm.Sz().x()-1) || (p.y() >= aDIm.Sz().y()-1)){
                    continue;
                }
                aDImT.SetV(cPt2di(i,j), aDIm.GetVBL(p));
                if ((i == 0) || (j == 0) || (i == Ni-1) || (j == Nj-1)){
                    aDCT->mDetectedFrame.push_back(cPt2di(p.x(), p.y()));
                }
            }
        }
    }

    return aImT;

}

// ---------------------------------------------------------------------------
// Function to apply affinity
// ---------------------------------------------------------------------------
cPt2dr cAppliExtractCodeTarget::applyAffinity(cPt2dr p, std::vector<double> param){
    double a11 = param[0]; double a12 = param[1]; double bx  = param[2];
    double a21 = param[3]; double a22 = param[4]; double by  = param[5];
    return cPt2dr(a11*p.x() + a12*p.y() + bx, a21*p.x() + a22*p.y() + by);
}

cPt2dr cAppliExtractCodeTarget::applyAffinityInv(cPt2dr p, std::vector<double> param){
    double a11 = param[0]; double a12 = param[1]; double bx  = param[2];
    double a21 = param[3]; double a22 = param[4]; double by  = param[5];
    double det = a11*a22 - a21-a12;
    double tx = p.x()-bx; double ty = p.y()-by;
    return cPt2dr((a22*tx - a12*ty)/det, (-a21*tx+a11*ty)/det);
}

// ---------------------------------------------------------------------------
// Function to test if affinity is valid
// ---------------------------------------------------------------------------
bool cAppliExtractCodeTarget::isValidAffinity(std::vector<double> param){
    for (unsigned i=0; i<6; i++){
        if (isnan(param[0])){
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Function to plot debug image
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::plotDebugImage(cDCT* target, const cDataIm2D<float>& aDIm){

    // ----------------------------------------------------------------------------------------------------------
    // [008] 0000001000 plot only axis lines of detected chessboard patterns (green lines)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[3]){
        for (int sign=-1; sign<=1; sign+=2){
            for (int i=1; i<=target->mSizeTargetEllipse; i++){
                cPt2di center = target->Pix();
                cPt2di p1 = cPt2di(center.x()+sign*target->mDirC1.x()*i, center.y()+sign*target->mDirC1.y()*i);
                cPt2di p2 = cPt2di(center.x()+sign*target->mDirC2.x()*i, center.y()+sign*target->mDirC2.y()*i);
                plotSafeRectangle(mImVisu, p1, (mLineWidthDebug-1)/2, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
                plotSafeRectangle(mImVisu, p2, (mLineWidthDebug-1)/2, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
            }
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [002] 0000000010 plot only center of detected targets (white pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[1]){
        plotSafeRectangle(mImVisu, target->Pix(), 0, cRGBImage::White, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
    }

    // ----------------------------------------------------------------------------------------------------------
    // [004] 0000000100 plot only transitions on circles around candidate targets (yellow pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[2]){
        cPt3di color = (target->mRecomputed?cRGBImage::Yellow:cRGBImage::Orange);
        for (unsigned i=0; i<target->mDetectedVectors.size(); i++){
            plotSafeRectangle(mImVisu, target->mDetectedVectors.at(i), 0, color, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [032] 0000100000 plot only fitted ellipse (red lines)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[5]){
        for (unsigned i=0; i<target->mDetectedEllipse.size(); i++){
            plotSafeRectangle(mImVisu, target->mDetectedEllipse.at(i), (mLineWidthDebug-1)/2, cRGBImage::Red, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [016] 0000010000 plot only data point for ellipse fit (cyan pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[4]){
        for (unsigned i=0; i<mPoints.size(); i++){
            plotSafeRectangle(mImVisu, cPt2di(mPoints.at(i).x(), mPoints.at(i).y()), 0.0, cRGBImage::Cyan, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [064] 0001000000 plot only intersections between ellipse and axes (yellow pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[6]){
        for (unsigned i=0; i<target->mDetectedCorners.size(); i++){
            cPt2di p = cPt2di((int)(target->mDetectedCorners.at(i).x()), (int)(target->mDetectedCorners.at(i).y()));
            plotSafeRectangle(mImVisu, p, 0, cRGBImage::Yellow, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [128] 0010000000 plot only detected target frames (blue lines)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[7]){
        for (unsigned i=0; i<target->mDetectedFrame.size(); i++){
            plotSafeRectangle(mImVisu, target->mDetectedFrame.at(i), mLineWidthDebug, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [256] 0100000000 plot detected target code name (cyan characters)
    // ----------------------------------------------------------------------------------------------------------
    if (mBitsPlotDebug[8]){
        cPt2dr p = applyAffinity(cPt2dr(4.0,4.0), mTransfo);
        double it = p.x(); double jt = p.y();
        for (unsigned lettre=0; lettre<target->mDecodedName.size(); lettre++){
            std::string aStr; aStr.push_back(target->mDecodedName[lettre]);
            cIm2D<tU_INT1> aImStr = ImageOfString_10x8(aStr,1); cDataIm2D<tU_INT1>&  aDataImStr = aImStr.DIm();
            for (int i=0; i<11*mLetter; i++){
                for (int j=0; j<11*mLetter; j++){
                    if (aDataImStr.DefGetV(cPt2di(i/mLetter,j/mLetter),0)){
                        plotSafeRectangle(mImVisu, cPt2di(it + i + lettre*10*mLetter, jt + j), 0.0, cRGBImage::Cyan, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Function to plot "safe" rectangle in image
// ---------------------------------------------------------------------------
bool cAppliExtractCodeTarget::plotSafeRectangle(cRGBImage image, cPt2di point, double sz, cPt3di color, int imax, int jmax, double transparency){
    if ((point.x()+sz < imax) && (point.y()+sz < jmax)){
        if ((point.x()-sz >= 0) && (point.y()-sz >= 0)){
            image.SetRGBrectWithAlpha(point, sz, color, transparency);
            return true;
        }
    }
    return false;
}


// ---------------------------------------------------------------------------
// Function to plot "safe" rectangle in gray level image
// ---------------------------------------------------------------------------
bool cAppliExtractCodeTarget::markImage(tDataImT & aDImT, cPt2di p, int sz, int level){

    bool success = true;

    for (int i=-sz; i<=sz; i++){
        for (int j=-sz; j<=sz; j++){
            cPt2di pt = cPt2di(p.x()+i, p.y()+j);
            if ((pt.x() < aDImT.Sz().x()) && (pt.y() < aDImT.Sz().y()) && (pt.x() >= 0) && (pt.y() >= 0)){
                aDImT.SetV(pt, level);
            }else{
                success = false;
            }
        }
    }
    return success;
}


// ---------------------------------------------------------------------------
// Function to inspect a matrix in R
// ---------------------------------------------------------------------------

// EIGEN
void cAppliExtractCodeTarget::printMatrix(MatrixXd M){
    StdOut() << "================================================================\n";
    StdOut() << "M = matrix(c(\n";
    for (int i=0; i<M.rows(); i++){
        for (int j=0; j<M.cols(); j++){
            StdOut() << M(i,j);
            if ((i != M.rows()-1) || (j != M.cols()-1)) StdOut() << ",";
        }
        StdOut() << "\n";
    }
    StdOut() << " ), ncol=" << M.cols() <<", nrow=" << M.rows() << ", byrow=TRUE)\n";
    StdOut() << "image(M, col = gray.colors(255))\n";
    StdOut() << "================================================================\n";
}

// ---------------------------------------------------------------------------
// Function to extract edge of butterfly pattern for ellipse fit data points
// Inputs: the full image
// Outputs: a set of boundary points (cPt2dr)
// ---------------------------------------------------------------------------
std::vector<cPt2dr> cAppliExtractCodeTarget::extractButterflyEdge(const cDataIm2D<float>& aDIm, cDCT* aDCT){
    std::vector<cPt2dr> POINTS;
    double threshold = (aDCT->mVBlack + aDCT->mVWhite)/2.0;
    cPt2di center = aDCT->Pix();
    double x, y, vx, vy, z_prec, z_curr, w1, w2;
    double vx1 = aDCT->mDirC1.x(); double vy1 = aDCT->mDirC1.y();
    double vx2 = aDCT->mDirC2.x(); double vy2 = aDCT->mDirC2.y();
    for (double t=mMargin; t<1-mMargin; t+=mStepButterfly){
        vx = t*vx1 + (1-t)*vx2;
        vy = t*vy1 + (1-t)*vy2;
        for (int sign=-1; sign<=1; sign+=2){
            z_prec = 0; cPt2dr pf_prec = cPt2dr(0,0);
            z_curr = 0; cPt2dr pf_curr = cPt2dr(0,0);
            for (int i=mDiamMinD/10; i<=300; i++){
                x = center.x()+sign*vx*i;
                y = center.y()+sign*vy*i;
                pf_prec = pf_curr;
                pf_curr = cPt2dr(x, y);
                if ((x < 0) || (y < 0) || (x >= aDIm.Sz().x()-1) || (y >= aDIm.Sz().y()-1)) continue;
                z_prec = z_curr; z_curr = aDIm.GetVBL(pf_curr);
               // plotSafeRectangle(mImVisu, cPt2di(pf_curr.x(), pf_curr.y()), 0.0, cRGBImage::White, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                if ((z_curr > threshold) && (z_curr-z_prec > mGradButterfly) && (pf_prec.x()*pf_prec.y() > 0)){
                    w1 = +(z_prec-threshold)/(z_prec-z_curr);
                    w2 = -(z_curr-threshold)/(z_prec-z_curr);
                    cPt2dr pf = cPt2dr(w1*pf_curr.x() + w2*pf_prec.x(), w1*pf_curr.y() + w2*pf_prec.y());
                    POINTS.push_back(pf);
                   // plotSafeRectangle(mImVisu, cPt2di(pf.x(), pf.y()), 0.0, cRGBImage::Cyan, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                    break;
                }
            }
        }
    }
    //plotDebugImage(aDCT, aDIm);
    return POINTS;
}


// ---------------------------------------------------------------------------
// Function to extract edge of butterfly pattern for ellipse fit data points
// Inputs: the full image
// Outputs: a set of boundary points (cPt2dr)
// ---------------------------------------------------------------------------
std::vector<cPt2dr> cAppliExtractCodeTarget::extractButterflyEdgeOld(const cDataIm2D<float>& aDIm, cDCT* aDCT){
    std::vector<cPt2dr> POINTS;
    double threshold = (aDCT->mVBlack + aDCT->mVWhite)/2.0;
    cPt2di center = aDCT->Pix();
    double vx1 = aDCT->mDirC1.x(); double vy1 = aDCT->mDirC1.y();
    double vx2 = aDCT->mDirC2.x(); double vy2 = aDCT->mDirC2.y();
    double lim_inf_apriori = 5;
    for (double angle=0; angle<0.5-mMargin; angle+=mStepButterfly){
        for (double side=-1; side<=+1; side+=2){
            double t = 0.5 + side*angle;
            for (int sign=-1; sign<=1; sign+=2){
                double z_prec = 0; cPt2dr pf_prec = cPt2dr(0,0);
                double z_curr = 0; cPt2dr pf_curr = cPt2dr(0,0);
                for (int i=lim_inf_apriori; i<=300; i++){
                    double vx = t*vx1 + (1-t)*vx2;
                    double vy = t*vy1 + (1-t)*vy2;
                    double x = center.x()+sign*vx*i;
                    double y = center.y()+sign*vy*i;
                    pf_prec = pf_curr;
                    pf_curr = cPt2dr(x, y);
                    if ((x < 0) || (y < 0) || (x >= aDIm.Sz().x()-1) || (y >= aDIm.Sz().y()-1)) continue;
                    z_prec = z_curr;
                    z_curr = aDIm.GetVBL(pf_curr);
                    if (z_curr > threshold){
                        double w1 = +(z_prec-threshold)/(z_prec-z_curr);
                        double w2 = -(z_curr-threshold)/(z_prec-z_curr);
                        cPt2dr pf = cPt2dr(w1*pf_curr.x() + w2*pf_prec.x(), w1*pf_curr.y() + w2*pf_prec.y());
                        POINTS.push_back(pf);
                        lim_inf_apriori = 0.5*i;  // A tuner (un peu)
                        break;
                    }
                }
            }
        }
    }
    return POINTS;
}


// ---------------------------------------------------------------------------
// Function to fit an ellipse on a set of 2D points
// Inputs:
//      - a vector of 2D floating points (cPt2dr)
//      - a floating point for optional constrain on center (cPt2dr)
//      - a boolean to activate constrain on center (bool)
//      - a pointer to array of parameters  (A, B, C, D, E, F) of equation:
// Ax2 + Bxy + Cy^2 + Dx+ Ey + F. Adapted to handle center-constrained fit
// of ellipse. All solutions are guaranteed to be proper ellipse: B^2-4AC < 0
// ---------------------------------------------------------------------------
// NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES
// (Halir and Flusser, 1998)
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::fitEllipse(std::vector<cPt2dr> points, cPt2dr center, bool constrained, double* output){

	const unsigned N = points.size();
	const unsigned M = (!constrained)*2 + 1;
    cDenseMatrix<double>  D1Wrap(3,N);
    cDenseMatrix<double>  D2Wrap(M,N);
    cDenseMatrix<double>  MWrap(3,3);


    double xmin = center.x();
    double ymin = center.y();

    if (!constrained){
        xmin = 1e300;
        ymin = 1e300;
        for (unsigned i=0; i<N; i++){
            if (points.at(i).x() < xmin) xmin = points.at(i).x();
            if (points.at(i).y() < ymin) ymin = points.at(i).y();
        }
    }

    for (unsigned i=0; i<N; i++){
        points.at(i).x() -= xmin;
        points.at(i).y() -= ymin;
    }

    for (unsigned i=0; i<N; i++){
        double x = points.at(i).x(); double y = points.at(i).y();
        D1Wrap.SetElem(0,i,x*x);
        D1Wrap.SetElem(1,i,x*y);
        D1Wrap.SetElem(2,i,y*y);
        if (!constrained){
            D2Wrap.SetElem(0,i,x);
            D2Wrap.SetElem(1,i,y);
            D2Wrap.SetElem(2,i,1);
        }else{
             D2Wrap.SetElem(0,i,1);
        }
    }

    cDenseMatrix<double> S1Wrap = D1Wrap.Transpose() * D1Wrap;
    cDenseMatrix<double> S2Wrap = D1Wrap.Transpose() * D2Wrap;
    cDenseMatrix<double> S3Wrap = D2Wrap.Transpose() * D2Wrap;
    cDenseMatrix<double> TWrap  = (-1)*S3Wrap.Inverse() * S2Wrap.Transpose();
    cDenseMatrix<double> M1Wrap = S1Wrap + S2Wrap*TWrap;

	for (unsigned i=0; i<3; i++){
		MWrap.SetElem(i,0,+M1Wrap(i,2)/2.0);
		MWrap.SetElem(i,1,-M1Wrap(i,1)    );
		MWrap.SetElem(i,2,+M1Wrap(i,0)/2.0);
	}

    cResulEigenDecomp<double> eigensolverWrap  = MWrap.Eigen_Decomposition();

    cDenseMatrix<double>  PWrap = eigensolverWrap.mEigenVec_R;

	double v12 = PWrap.GetElem(1,0); double v13 = PWrap.GetElem(2,0);
	double v22 = PWrap.GetElem(1,1); double v23 = PWrap.GetElem(2,1);
	double v32 = PWrap.GetElem(1,2); double v33 = PWrap.GetElem(2,2);

	bool cond2 = 4*v12*v32-v22*v22 > 0;
	bool cond3 = 4*v13*v33-v23*v23 > 0;
	int index = cond2*1 + cond3*2;
	if (index > 2){
        return -1;
	}

	double a1 = PWrap.GetElem(index, 0);
	double a2 = PWrap.GetElem(index, 1);
	double a3 = PWrap.GetElem(index, 2);

	output[0] = a1;
	output[1] = a2;
	output[2] = a3;
	if (!constrained){
        output[3] = TWrap.GetElem(0,0)*a1 + TWrap.GetElem(1,0)*a2 + TWrap.GetElem(2,0)*a3;
        output[4] = TWrap.GetElem(0,1)*a1 + TWrap.GetElem(1,1)*a2 + TWrap.GetElem(2,1)*a3;
        output[5] = TWrap.GetElem(0,2)*a1 + TWrap.GetElem(1,2)*a2 + TWrap.GetElem(2,2)*a3;
	}else{
        output[3] = 0.0;
        output[4] = 0.0;
        output[5] = TWrap.GetElem(0,0)*a1 + TWrap.GetElem(1,0)*a2 + TWrap.GetElem(2,0)*a3;
	}

    // Recenter ellipse
    translateEllipse(output, cPt2dr(xmin, ymin));

    return 0;

}



// ---------------------------------------------------------------------------
// Function to fit an ellipse on a set of 2D points
// Short cut to fitEllipse without constraint on center
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::fitFreeEllipse(std::vector<cPt2dr> points, double* output){
    return fitEllipse(points, cPt2dr(0, 0), false, output);
}

// ---------------------------------------------------------------------------
// Function to fit an ellipse on a set of 2D points
// Short cut to fitEllipse with constraint on center
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::fitConstrainedEllipse(std::vector<cPt2dr> points, cPt2dr center, double* output){
    return fitEllipse(points, center, true, output);
}

// ---------------------------------------------------------------------------
// Function to test ellipse fit pixel residuals
// Requires estimated affinity paramer to compute RMSE in pixel space
// ---------------------------------------------------------------------------
double cAppliExtractCodeTarget::ellipseResidual(std::vector<cPt2dr> points, std::vector<double> transfo){
    double m1r = 0;
    double m2r = 0;
    for (unsigned i=0; i<points.size(); i++){
        cPt2dr p = applyAffinityInv(points.at(i), transfo);
        double value = sqrt(p.x()*p.x() + p.y()*p.y());
        m1r += value;
        m2r += value*value;
    }
    m1r /= points.size();
    m2r /= points.size();
    return sqrt(m2r - m1r*m1r);
}


// ---------------------------------------------------------------------------
// Function to convert cartesian parameters (A,B,C,D,E,F) to natural
// parameters (x0, y0, a, b, theta).
// Inputs: an array of 6 floating point parameters
// Output: an array of 5 floating point parameters
// ---------------------------------------------------------------------------
// Ellipse algebraic equation: e(x,y) = Ax2 + Bxy + Cy2 + Dx + Ey + F = 0 with
// B2 - 4AC < 0.
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::cartesianToNaturalEllipse(double* parameters, double* output){

    double A, B, C, D, F, G;
    double x0, y0, a, b, theta;
    double delta, num, fac, temp = 0;

    // Conversions for applying e'(x,y) = ax^2 + 2bxy + cy^2 + 2dx + 2fy + g
    A = parameters[0];
    B = parameters[1]/2;
    C = parameters[2];
    D = parameters[3]/2;
    F = parameters[4]/2;
    G = parameters[5];


    delta = B*B - A*C;

    if (delta > 0){
        //StdOut() << "Error: bad coefficients for ellipse algebraic equation \n";
        return 1;
    }

    // Center of ellipse
    x0 = (C*D-B*F)/delta;
    y0 = (A*F-B*D)/delta;

    num = 2*(A*F*F+C*D*D+G*B*B-2*B*D*F-A*C*G);
    fac = sqrt(pow(A-C, 2) + 4*B*B);

    // Semi-major and semi-minor axes
    a = sqrt(num/delta/(fac-A-C));
    b = sqrt(num/delta/(-fac-A-C));
    if (b > a){
        temp = a; a = b; b = temp;
    }

    // Ellipse orientation
    if (b == 0){
        theta = (A < C)? 0:PI/2;
    }else{
        theta = atan(2.0*B/(A-C))/2.0 + ((A>C)?PI/2:0);
    }
    theta += temp? PI/2 : 0;

    theta = std::fmod(theta,PI);

    output[0] = x0; output[1] = y0;
    output[2] = a ; output[3] = b ;
    output[4] = theta;

    return 0;

}

// ---------------------------------------------------------------------------
// Function to solve intersection between cross segments and ellipse
// ---------------------------------------------------------------------------
// Inputs:
//    - a pointer to target
//    - param double vector (A, B, C, D, E, F)
// Ouput:
//    - vector of 4 cPt2dr intersections
// ---------------------------------------------------------------------------
std::vector<cPt2dr> cAppliExtractCodeTarget::solveIntersections(cDCT* target, double* param){

    std::vector<cPt2dr> INTERSECTIONS;

    double cx = target->Pix().x();
    double cy = target->Pix().y();

    double vx1 = target->mDirC1.x(); double vy1 = target->mDirC1.y();
    double vx2 = target->mDirC2.x(); double vy2 = target->mDirC2.y();

    double A = param[0]; double B = param[1]; double C = param[2];
    double D = param[3]; double E = param[4]; double F = param[5];
    double a1 = A*vx1*vx1 + B*vx1*vy1 + C*vy1*vy1;
    double b1 = 2*A*cx*vx1 + B*(cx*vy1 + cy*vx1) + 2*C*cy*vy1 + D*vx1 + E*vy1;
    double c1 = A*cx*cx + B*cx*cy + C*cy*cy + D*cx + E*cy + F;
    double sqrt_del1 = b1*b1-4*a1*c1;

    if (sqrt_del1 < 0) return INTERSECTIONS;

    sqrt_del1 = sqrt(sqrt_del1);
    double t11 = (-b1-sqrt_del1)/(2*a1);
    double t12 = (-b1+sqrt_del1)/(2*a1);

    double a2 = A*vx2*vx2 + B*vx2*vy2 + C*vy2*vy2;
    double b2 = 2*A*cx*vx2 + B*(cx*vy2 + cy*vx2) + 2*C*cy*vy2 + D*vx2 + E*vy2;
    double c2 = A*cx*cx + B*cx*cy + C*cy*cy + D*cx + E*cy + F;
    double sqrt_del2 = b2*b2-4*a2*c2;

    if (sqrt_del2 < 0) return INTERSECTIONS;

    sqrt_del2 = sqrt(sqrt_del2);
    double t21 = (-b2-sqrt_del2)/(2*a2);
    double t22 = (-b2+sqrt_del2)/(2*a2);

    INTERSECTIONS.push_back(cPt2dr(cx + t22*vx2, cy + t22*vy2));
    INTERSECTIONS.push_back(cPt2dr(cx + t11*vx1, cy + t11*vy1));
    INTERSECTIONS.push_back(cPt2dr(cx + t21*vx2, cy + t21*vy2));
    INTERSECTIONS.push_back(cPt2dr(cx + t12*vx1, cy + t12*vy1));

    return INTERSECTIONS;
}



// ---------------------------------------------------------------------------
// Function to estimate affinity on n points
// ---------------------------------------------------------------------------
// Inputs:
//  - Coordinates X of points in input (rectified) image
//  - Coordinates Y of points in output image
// Outputs:
//  - vector of parameters a11, a12, bx, a21, a22, by + RMSE
// ---------------------------------------------------------------------------
std::vector<double> cAppliExtractCodeTarget::estimateAffinity(std::vector<cPt2dr> X, std::vector<cPt2dr> Y){

    std::vector<double> transfo = {0, 0, 0, 0, 0, 0, 0};

    const unsigned N = X.size();
    cDenseMatrix<double>  A(6,2*N); A = 0*A;
    cDenseMatrix<double>  B(1,2*N); B = 0*B;

    for (unsigned i=0; i<N; i++){

        B.SetElem(0,2*i,Y.at(i).x());
        A.SetElem(0,2*i,X.at(i).x());
        A.SetElem(1,2*i,X.at(i).y());
        A.SetElem(2,2*i,1          );

        B.SetElem(0,2*i+1,Y.at(i).y());
        A.SetElem(3,2*i+1,X.at(i).x());
        A.SetElem(4,2*i+1,X.at(i).y());
        A.SetElem(5,2*i+1,1          );

    }

    cDenseMatrix<double> x = (A.Transpose()*A).Inverse()*A.Transpose()*B;

    for (unsigned i=0; i<6; i++) transfo[i] = x.GetElem(0,i);
    cDenseMatrix<double> V = A*x-B;

    transfo[6] = sqrt((V.Transpose()*V).GetElem(0,0)/(2*N-6)) ;

    return transfo;

}


// ---------------------------------------------------------------------------
// Function to estimate affinity on 4 points for image rectifying
// ---------------------------------------------------------------------------
// Inputs:
//  - Coordinates X of points in output image
//  - Chessboard rotation angle theta
// Outputs:
//  - vector of parameters a11, a12, bx, a21, a22, by + RMSE
// ---------------------------------------------------------------------------
std::vector<double> cAppliExtractCodeTarget::estimateRectification(std::vector<cPt2dr>Y, double theta){

    theta = theta - PI/4.0;

    cPt2dr c1 = cPt2dr(cos(theta)*(+1) - sin(theta)*(+1), sin(theta)*(+1) + cos(theta)*(+1));
    cPt2dr c2 = cPt2dr(cos(theta)*(+1) - sin(theta)*(-1), sin(theta)*(+1) + cos(theta)*(-1));
    cPt2dr c3 = cPt2dr(cos(theta)*(-1) - sin(theta)*(-1), sin(theta)*(-1) + cos(theta)*(-1));
    cPt2dr c4 = cPt2dr(cos(theta)*(-1) - sin(theta)*(+1), sin(theta)*(-1) + cos(theta)*(+1));

    std::vector<cPt2dr> X = {c1, c2, c3, c4};

    return estimateAffinity(X, Y);

}


// ---------------------------------------------------------------------------
// Function to generate a point at coordinate (double) t from natural
// parameters of an ellipse
// Inputs: an array of 5 floating point parameters, a coordinate double t and
// a noise level (standard deviation).
// Output: a cPt2dr object
// ---------------------------------------------------------------------------
cPt2dr cAppliExtractCodeTarget::generatePointOnEllipse(double* parameters, double t, double noise = 0.0){
    static std::default_random_engine generator;
    double x, y;
    double x0 = parameters[0]; double y0 = parameters[1];
    double a = parameters[2]; double b = parameters[3]; double theta = parameters[4];
    std::normal_distribution<double> distribution(0.0, noise);
    x = x0 + a*cos(t)*cos(theta) - b*sin(t)*sin(theta) + distribution(generator);
    y = y0 + a*cos(t)*sin(theta) + b*sin(t)*cos(theta) + distribution(generator);
    return cPt2dr(x,y);
}



// ---------------------------------------------------------------------------
// Function to generate a set of points on an ellipse
// Inputs: an array of 5 floating point parameters, an integer N giving the
// number of points to generate and a noise level (std)
// Output: a vector of cPt2dr object
// ---------------------------------------------------------------------------
std::vector<cPt2dr> cAppliExtractCodeTarget::generatePointsOnEllipse(double* parameters, unsigned N, double noise = 0.0){
    std::vector<cPt2dr> POINTS;
    for (unsigned i=0; i<N; i++){
        POINTS.push_back(generatePointOnEllipse(parameters, RandInInterval(0,2*PI), noise));
    }
    return POINTS;
}

// ---------------------------------------------------------------------------
// Function to translate an ellipse directly in cartesian coordinates
// ---------------------------------------------------------------------------
// Inputs:
//      - A [6 x 1] array [A, B, C, D, E, F]
//      - a translate (dx, dy) given as cPt2dr
// Output: the array describing the ellipse translated by (dx, dy).
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::translateEllipse(double* parameters, cPt2dr translation){

	double u = translation.x();
	double v = translation.y();

	double A =  parameters[0];
	double B =  parameters[1];
	double C =  parameters[2];
	double D =  parameters[3];
	double E =  parameters[4];
	double F =  parameters[5];

	// Translation
    parameters[0] = A;
    parameters[1] = B;
    parameters[2] = C;
    parameters[3] = D - B*v - 2*A*u;
    parameters[4] = E - 2*C*v - B*u;
    parameters[5] = F + A*u*u  + C*v*v + B*u*v  - D*u - E*v;

}



// ---------------------------------------------------------------------------
// Function to decode a potential target on image
// Inputs:
//     - a rectified image after affinity computation
//     - threshold for white pixel values area
//     - threshold for black pixel values area
//     - a string to get read bits
//     - a boolean for mode (true for mode flight)
// Output: decoded id of target (-1 if failed)
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::decodeTarget(tDataImT & aDImT, double thw, double thb, std::string& debug, bool modeFilght){

    double R1 = 140.1*(1+0.35/2);
    double R2 = 140.1*(1+3*0.35/2);


    double cx = 300;
    double cy = 300;

    double threshold = (thw + thb)/2.0;
    double th_std_white_circle = 35;
    //double int_circle_th = 0.25*thb + 0.75*thw;

    if (modeFilght){
        R1 = 300;
        cx = 640/2.0;
        cy = 1260/2.0;
    }


    // -------------------------------------------
    // Internal white strip circle control
    // -------------------------------------------
    double M = 0;
    double S = 0;
    int nb_pts = 0;

    for (double t=0; t<2*PI; t+=0.01){

        double xc_int = cx + R1*cos(t);
        double yc_int = cy + R1*sin(t);

        double value = aDImT.GetVBL(cPt2dr(xc_int, yc_int));

        if (t == 0){
            M = value;
            continue;
        }

		double Mt = M;
		M += (value-M)/(nb_pts+1);
		S += (value-Mt)*(value-M);
		nb_pts ++;

        // Image modification (at the end!)
        aDImT.SetV(cPt2di(xc_int ,yc_int), 0.0);

    }

    S = sqrt(S/(nb_pts-1));

    printDebug("Internal strip circle avg", M, 0.8*thw);
    printDebug("Internal strip circle std", th_std_white_circle, S);

    if ((S > th_std_white_circle) || (M < 0.8*thw)){
        return -1;
    }


    // ====================================================================
    // Code central symetry control and decoding
    // ====================================================================

    int bits = 0; bool bit, bit1, bit2; double max_error = 0; debug = ""; bool consistent = true;

    // -------------------------------------------------------------
    // Standard case
    // -------------------------------------------------------------

    if (!modeFilght){
        for (int t=0; t<9; t++){

            double theta = (1.0/18.0 + t/9.0)*PI;

            double xc_code = 300 + R2*cos(theta);    double xc_code_tol = 300 + (1+mTolerance/100.0)*R2*cos(theta);
            double yc_code = 300 + R2*sin(theta);    double yc_code_tol = 300 + (1+mTolerance/100.0)*R2*sin(theta);

            double xc_code_opp = 300 - R2*cos(theta);   double xc_code_opp_tol = 300 - (1+mTolerance/100.0)*R2*cos(theta);
            double yc_code_opp = 300 - R2*sin(theta);   double yc_code_opp_tol = 300 - (1+mTolerance/100.0)*R2*sin(theta);


            // Code reading
            double value1 = std::min(aDImT.GetVBL(cPt2dr(xc_code    , yc_code    )), aDImT.GetVBL(cPt2dr(xc_code_tol    , yc_code_tol    )));
            double value2 = std::min(aDImT.GetVBL(cPt2dr(xc_code_opp, yc_code_opp)), aDImT.GetVBL(cPt2dr(xc_code_opp_tol, yc_code_opp_tol)));
            double avgval = (value1+value2)/2;

            bit1 = value1 > threshold;
            bit2 = value2 > threshold;
            bit = avgval > threshold;

            consistent = consistent && (bit1 == bit2);

            max_error = std::max(max_error, abs(value1-value2));

            bits += (bit ? 0:1)*pow(2,t);
            debug += (bit ? std::string("0 "):std::string("1 "));

            // Image modification (at the end!)
            if (bit1 == bit2){
                markImage(aDImT, cPt2di(xc_code        , yc_code        ), 5, bit1?0:255);
                markImage(aDImT, cPt2di(xc_code_opp    , yc_code_opp    ), 5, bit2?0:255);
                markImage(aDImT, cPt2di(xc_code_tol    , yc_code_tol    ), 5, bit1?0:255);
                markImage(aDImT, cPt2di(xc_code_opp_tol, yc_code_opp_tol), 5, bit2?0:255);
            } else {
                markImage(aDImT, cPt2di(xc_code        , yc_code        ), 5, 128);
                markImage(aDImT, cPt2di(xc_code_opp    , yc_code_opp    ), 5, 128);
                markImage(aDImT, cPt2di(xc_code_tol    , yc_code_tol    ), 5, 128);
                markImage(aDImT, cPt2di(xc_code_opp_tol, yc_code_opp_tol), 5, 128);
            }
        }
    }else {

    // -------------------------------------------------------------
    // Aerial case
    // -------------------------------------------------------------
        bits = 0;


        // Prepare hamming decoder
        cHamingCoder aHC(spec.mNbBit-1);

        aDImT.SetV(cPt2di(300,300), 255.0);


        int NbCols = ceil(((double)aHC.NbBitsOut())/2.0);

        int sq_vt = 90;
        int sq_sz = 480/NbCols;
        int idl, idc;
        int px, py;
        double val1, val2;

        double mx1 = 98 ; double my1 = 636;
        double mx2 = 550; double my2 = 636;

        val1 = aDImT.GetVBL(cPt2dr(mx1, my1));
        val2 = aDImT.GetVBL(cPt2dr(mx2, my2));
        bool hypo1 = val1 > threshold;
        bool hypo2 = val2 > threshold;
        markImage(aDImT, cPt2di(mx1, my1), 10, hypo1?0:255);
        markImage(aDImT, cPt2di(mx2, my2), 10, hypo2?0:255);

        hypo1 = val1 > val2;

        StdOut() << "------------------------------------------------------------------------------------\n";

        for (int k=0; k<aHC.NbBitsOut(); k++){
            idc = k % NbCols;
            idl = (k>=NbCols)*1;

            // ---------------------------------------------------
            // Upper part hypothesis
            // ---------------------------------------------------

            if (!hypo1){
                px = 125+idc*sq_sz;
                py = 150 + idl*sq_vt;
                val1 = aDImT.GetVBL(cPt2dr(px, py));
                bit = val1 > threshold;
                markImage(aDImT, cPt2di(px, py), 10, bit?0:255);
                bits += (bit ? 0:1)*pow(2,aHC.NbBitsOut()-1-k);
                debug = (bit ? std::string("0 "):std::string("1 ")) + debug;
            }else{
            // ---------------------------------------------------
            // Lower part hypothesis
            // ---------------------------------------------------
                px = 125+idc*sq_sz;
                py = 1035 + idl*sq_vt;
                val1 = aDImT.GetVBL(cPt2dr(px, py));
                bit = val1 > threshold;
                markImage(aDImT, cPt2di(px, py), 10, bit?0:255);
                bits += (bit ? 0:1)*pow(2,k);
                debug = debug + (bit ? std::string("0 "):std::string("1 "));
            }

        }

        StdOut() << "Hamming code: " << bits;
        bits = aHC.UnCodeWhenCorrect(bits);
        StdOut() << " / Uncoded: " << bits << " / Recoded: " << aHC.Coding(bits) << "\n";

    }

    // !!!!!!!!!!!!!!!!!!! PROVISOIRE !!!!!!!!!!!!!!!!!!!
    printDebug("Code symetric consistency", (max_error < 60) && consistent);
    if ((bits == -1) || (max_error > 100) || (!consistent)) return bits = 0;

    return bits;
}


// ---------------------------------------------------------------------------
// Function to generate plot in R script for ellipse fit debugging
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::benchEllipseR(){

    // -------------------------------------------------
    // Bench test for elliptical fit
    // -------------------------------------------------

    // Generating random ellipse

    double parameter[5];
    parameter[0] = RandInInterval(-1e4,+1e4);
    parameter[1] = RandInInterval(-1e4,+1e4);
    parameter[2] = RandInInterval(1e-3,1e3);
    parameter[3] = RandInInterval(1e-1*parameter[2],parameter[2]);
    parameter[4] = RandInInterval(0,PI);

    double output[6];


    std::vector<cPt2dr> POINTS = generatePointsOnEllipse(parameter, 100, 10.0);

    // -------------------------------------------------
    // Bench test for elliptical fit
    // -------------------------------------------------
    fitEllipse(POINTS, cPt2dr(parameter[0], parameter[1]), true, output);


    double solution[5];
    cartesianToNaturalEllipse(output, solution);


    StdOut() << "pdf('test.pdf')\n";

    // Data points generation
    StdOut() << "A = matrix(c(\n";
    for (unsigned i=0; i<POINTS.size(); i++){
        StdOut() << POINTS.at(i).x() << "," << POINTS.at(i).y();
        if (i < POINTS.size()-1)  StdOut() << ",";
        StdOut() << "\n";
    }
    StdOut() << "), ncol=2, byrow=TRUE)\n";


    POINTS = generatePointsOnEllipse(solution, 1000, 0.0);


    StdOut() << "B = matrix(c(\n";
    for (unsigned i=0; i<1000; i++){
        StdOut() << POINTS.at(i).x() << "," << POINTS.at(i).y();
        if (i < 999)  StdOut() << ",";
        StdOut() << "\n";
    }
    StdOut() << "), ncol=2, byrow=TRUE)\n";
    StdOut() << "plot(A[,1], A[,2], col='blue', pch=4)\n";
    StdOut() << "points(B[,1], B[,2], col='red', cex=.1)\n";
    StdOut() << "points(" << parameter[0] << "," << parameter[1] << ", pch=4)\n";
    StdOut() << "points(" << solution[0]  << "," << solution[1]  << ", pch=5)\n";
    StdOut() << "dev.off()\n";
    StdOut() << "cat('-----------------------------------------\n')\n";
    for (unsigned i=0; i<5; i++){
        StdOut() << "cat('" << parameter[i] << " " << solution[i] << " " << parameter[i]-solution[i] <<  "\n')\n";
    }
    StdOut() << "cat('-----------------------------------------\n')\n";

}


// ---------------------------------------------------------------------------
// Function to generate plot in R script for ellipse fit debugging
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::plotCaseR(std::vector<cPt2dr> POINTS, double* solution){

    StdOut() << "A = matrix(c(\n";
    for (unsigned i=0; i<POINTS.size(); i++){
        StdOut() << POINTS.at(i).x() << "," << POINTS.at(i).y();
        if (i < POINTS.size()-1)  StdOut() << ",";
        StdOut() << "\n";
    }
    StdOut() << "), ncol=2, byrow=TRUE)\n";

    std::vector<cPt2dr> ELLIPSE = generatePointsOnEllipse(solution, 1000, 0.0);

    StdOut() << "B = matrix(c(\n";
    for (unsigned i=0; i<1000; i++){
        StdOut() << ELLIPSE.at(i).x() << "," << ELLIPSE.at(i).y();
        if (i < 999)  StdOut() << ",";
        StdOut() << "\n";
    }
    StdOut() << "), ncol=2, byrow=TRUE)\n";
    StdOut() << "plot(A[,1], A[,2], col='blue', pch=4)\n";
    StdOut() << "points(B[,1], B[,2], col='red', cex=.1)\n";
    StdOut() << "points(" << solution[0]  << "," << solution[1]  << ", pch=5)\n";

}




// ---------------------------------------------------------------------------
// Function to generate bench tests for ellipse fit functions
// Generates test for:
//      - multiple values of ellipse parameters
//      - constrained and unconstrained on center
//      - different amplitude of noise (including perfect zero-noise data)
//      - different data sample sizes
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::benchEllipse(){

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    const unsigned N = 10000;      // Number of tests to perform
    const unsigned m = 100;        // Min number of points in data
    const unsigned M = 1000;       // Max number of points in data
    const double cx_max = 1e4;     // Max absolute value for center x coords
    const double cy_max = 1e4;     // Max absolute value for center y coords
    const double a_min = 1e-3;     // Min value for a semi-minor-axis
    const double a_max = 1e+3;     // Max value for a semi-major-axis
    const double ecc_max = 1e-1;   // Min eccentricity of ellipse
    const double noise_max = 5e-2; // Max level noise (w.r.t semi-minor axis)
    // -----------------------------------------------------------------------

    double param[5];

    for (unsigned i=0; i<N; i++){

        // Generate random ellipse
        param[0] = RandInInterval(-cx_max, +cx_max);
        param[1] = RandInInterval(-cy_max, +cy_max);
        param[2] = RandInInterval(a_min, a_max);
        param[3] = RandInInterval(param[2]*ecc_max, param[2]);
        param[4] = RandInInterval(0,PI);

        // Generate data
        int nb = (int)(RandInInterval(m, M)); double flat = (param[2]-param[3])/param[2];
        double noise = RandInInterval(0, noise_max)*param[3]*(RandInInterval(0,1) > 1e-1);
        std::vector<cPt2dr> POINTS = generatePointsOnEllipse(param, nb, noise);
        StdOut() << "Generating " << nb << " pts:  FLAT. = " << flat;
        StdOut() << " ANG = " << param[4]*180/PI << " NOISE = " << noise << " ";

        // Estimate ellipse
        double fit[6]; double solution[5];
        fitFreeEllipse(POINTS, fit);
        cartesianToNaturalEllipse(fit, solution);

        if (RandInInterval(0,1)<0.5){
            StdOut() << " -  constrained fit:   ";
            fitConstrainedEllipse(POINTS, cPt2dr(param[0], param[1]), fit);
        }else{
            StdOut() << " -  unconstrained fit:   ";
            fitFreeEllipse(POINTS, fit);
            cartesianToNaturalEllipse(fit, solution);
        }

        bool ok = true;
        double dcx = std::abs(param[0]-solution[0]); ok = ok && (dcx < std::max(2*noise, 1e-6));
        double dcy = std::abs(param[1]-solution[1]); ok = ok && (dcy < std::max(2*noise, 1e-6));
        double da  = std::abs(param[2]-solution[2]); ok = ok && (da  < 2*noise+1e-6);
        double db  = std::abs(param[3]-solution[3]); ok = ok && (db  < 2*noise+1e-6);

        if ((flat > 0.25) && (nb > 100)){   // Skip this test if ellipse is too 'circular'
            double dth1 = std::abs(param[4]-solution[4]);
            double dth2 = std::abs(param[4]-solution[4] - PI);
            ok = ok && ((dth1 < 0.05)||(dth2 < 0.05));
        }

        StdOut() << "[" << (ok?"PASSED":"FAILED") << "]";

        if (!ok){
            StdOut() << "\n--------------------------------------------\n";
            StdOut() << "TEST FAILURE DETAILS: \n";
            StdOut() << "--------------------------------------------\n";
            for (int j=0; j<5; j++){
                StdOut() << param[j] << " " << solution[j] << " " << param[j] - solution[j] << "\n";
            }
            StdOut() << "--------------------------------------------\n";

        }

        StdOut() << "\n";

    }

}


// ---------------------------------------------------------------------------
// Function to generate bench tests for affinity fit function
// ---------------------------------------------------------------------------
void cAppliExtractCodeTarget::benchAffinity(){

    int N = 10;
    double noise = 0.1;

    double a11 = +5.1;   double a12 = +7.2;  double bx = -2.3;
    double a21 = -1.2;   double a22 = +4.5;  double by = +3.7;

    std::vector<cPt2dr> X;
    std::vector<cPt2dr> Y;

    for (int i=0; i<N; i++){
        double x = RandInInterval(-100,100);
        double y = RandInInterval(-100,100);
        double xi = a11*x + a12*y + bx + noise*RandInInterval(-1,1);
        double yi = a21*x + a22*y + by + noise*RandInInterval(-1,1);
        X.push_back(cPt2dr(x , y ));
        Y.push_back(cPt2dr(xi, yi));
    }

    std::vector<double> param = estimateAffinity(X, Y);

    StdOut() << param[0]-a11 << " ";
    StdOut() << param[1]-a12 << " ";
    StdOut() << param[2]-bx  << " ";
    StdOut() << param[3]-a21 << " ";
    StdOut() << param[4]-a22 << " ";
    StdOut() << param[5]-by  << " ";
    StdOut() << param[6] << "\n";

}




int cAppliExtractCodeTarget::ExeOnParsedBox()
{
   // TestSadl(APBI_Im());
  //  std::pair<cIm2D<tREAL4>,cIm2D<tREAL4>>>

   mImGrad    =  Deriche(APBI_DIm(),2.0);
   DoExtract();

   return EXIT_SUCCESS;
}


int  cAppliExtractCodeTarget::Exe(){


    if (mTest){



        double a11 = 2;
        double a12 = 2;
        double a21 = 1;
        double a22 = 3;
        double tx = 10;
        double ty = 12;
        double a_col = 1.5;
        double b_col = 38;



        cDenseVect<double> aVInit(8);
        aVInit(0) = a11 + RandUnif_C();
        aVInit(1) = a12 + RandUnif_C();
        aVInit(2) = a21 + RandUnif_C();
        aVInit(3) = a22 + RandUnif_C();
        aVInit(4) = tx + 3*RandUnif_C();
        aVInit(5) = ty + 3*RandUnif_C();
        aVInit(6) = a_col + RandUnif_C();
        aVInit(7) = b_col + 3*RandUnif_C();


        cIm2D<tREAL8> mIm = cIm2D<tREAL8>(cPt2di(100,100));
        cDataIm2D<tREAL8>& mDIm =mIm.DIm();


        std::vector<cPt2dr> mVPtsMod;
        std::vector<double> mValueMod;
        for (const auto & aPixIm : mDIm){
            if (mDIm.Interiority(aPixIm)>3){
                mVPtsMod.push_back(cPt2dr(aPixIm.x(), aPixIm.y()));
                double value = (double)(128+128*RandUnif_C());
                mValueMod.push_back(value);
                StdOut() << cPt2dr(aPixIm.x(), aPixIm.y()) << " " << value << "\n";
            }
        }


        /*
        for (unsigned i=0; i<mVPtsMod.size(); i++){
            StdOut() << mVPtsMod.at(i) << " " << mValueMod.at(i) << "\n";
        }
        */

        mDIm.ToFile("testImage.tif");

        cResolSysNonLinear<tREAL8>* mSys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense,aVInit);
        cCalculator<double> *  mEqHomIm = EqDeformImHomotethy(true,1);


        cDenseVect<double> aVCur = mSys->CurGlobSol();


        StdOut() << "-----------------------\n";
        StdOut() << "Valeurs initiales\n";
        StdOut() << "-----------------------\n";
        for (int i=0; i<aVCur.Sz(); i++) {
            StdOut() << aVCur(i) << "\n";
        }


        FakeUseIt(mSys);
        FakeUseIt(mEqHomIm);


        // Free allocated memory
        delete mSys;
        delete mEqHomIm;


        return 0;
    }








    StdOut() << "============================================================\n";
    StdOut() << "CODED TARGET AUTOMATIC EXTRACTION \n";
    StdOut() << "============================================================\n";

    // -------------------------------------------------------------------------------------
    // Loading ground truth (if any)
    // -------------------------------------------------------------------------------------
    if ((mGroundTruthFile != "") && (ExistFile(mGroundTruthFile))){
        StdOut() << "Reading GT file: " << mGroundTruthFile << "... ";
        mWithGT = true;
        mGTResSim   = cResSimul::FromFile(mGroundTruthFile);
        for (auto & aGSD : mGTResSim.mVG){      //  Take into account offset of reading
         //   StdOut() << aGSD << "\n";
            aGSD.Translate(-ToR(mBoxTest.P0()));
        }
        StdOut() << "ok (" << mGTResSim.mVG.size() << " targets loaded)\n";
    }
    // -------------------------------------------------------------------------------------



   if (IsInit(&mPatExportF))
       mTestedFilters = SubOfPat<eDCTFilters>(mPatExportF,true);

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
      return ResultMultiSet();

   mPCT.InitFromFile(mNameTarget);
   mRayMinCB = (mDiamMinD/2.0) * (mPCT.mRho_0_EndCCB/mPCT.mRho_4_EndCar);

// StdOut() << "mRayMinCB " << mRayMinCB << "\n"; getchar();





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

