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
   mGT       (nullptr),
   mPt       (aPtR),
   mState    (aState),
   mScRadDir (1e5),
   mSym      (1e5),
   mBin      (1e5),
   mRad      (1e5)

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
        bool analyzeDCT(cDCT*, const cDataIm2D<float> &, int);                   ///< Analyze a potential target
        int decodeTarget(tDataImT &, double, double, std::string&, bool);        ///< Decode a potential target
        bool markImage(tDataImT &, cPt2di, int, int);                            ///< Plot mark on gray level image
        std::vector<cPt2dr> solveIntersections(cDCT*, double*);
        std::vector<cPt2dr> extractButterflyEdge(const cDataIm2D<float> &, cDCT*);
        void exportInXml(std::vector<cDCT*>);




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

        int fitEllipse (std::vector<cPt2dr>, cPt2dr, bool, double*);             ///< Least squares estimation of an ellipse from 2D points
        int fitFreeEllipse(std::vector<cPt2dr>, double*);                        ///< Least squares estimation of a floating ellipse from 2D points
        int fitConstrainedEllipse(std::vector<cPt2dr>, cPt2dr, double*);         ///< Least squares estimation of a constrained ellipse from 2D points

        std::vector<double> estimateRectification(std::vector<cPt2dr>, double);
        std::vector<double> estimateAffinity(std::vector<cPt2dr>, std::vector<cPt2dr>);


	std::string mNameTarget;

	cParamCodedTarget        mPCT;
    double                   mDiamMinD;
    bool                     mConstrainCenter;
	cPt2dr                   mRaysTF;
	cPt2di                   mTestCenter;

        std::vector<eDCTFilters> mTestedFilters;

        cImGrad<tREAL4>  mImGrad;  ///< Result of gradient
        double   mRayMinCB;        ///<  Ray Min CheckBoard
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

        std::vector<cPt2di> vec2plot;

        std::vector<double>  mParamBin;
        cParamCodedTarget spec;
        std::string mOutput_folder;
        std::vector<std::string> mToRestrict;

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
   mToRestrict      ({})
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
                    << AOpt2007(mRecompute, "Recompute", "Recompute affinity if needed", {eTA2007::HDV})
                    << AOpt2007(mXml, "Xml", "Print xml outout in file", {eTA2007::HDV})
                    << AOpt2007(mDebugPlot, "Debug", "Plot debug image with options", {eTA2007::HDV})
                    << AOpt2007(mMaxEcc, "MaxEcc", "Max. eccentricity of targets", {eTA2007::HDV})
                    << AOpt2007(mSaddle, "Saddle", "Prefiltering with saddle test", {eTA2007::HDV})
                    << AOpt2007(mMargin, "Margin", "Margin on butterfly edge for fit", {eTA2007::HDV})
                    << AOpt2007(mToRestrict, "Restrict", "List of codes to restrict on", {eTA2007::HDV})
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
            StdOut() << mToRestrict.at(i) << "\n";
        }
    }

    spec.InitFromFile("Target_Spec.xml");

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
    std::bitset<10> bitsPlotDebug = std::bitset<10>(mDebugPlot);
    if (mDebugPlot){
    StdOut() << "\n------------------------------------------------------------------\n";
    StdOut() << "DEBUG PLOT:\n";
    StdOut() << "------------------------------------------------------------------\n";
    if (bitsPlotDebug[0]) StdOut() << "* CANDIDATES AFTER FILTERING\n";
    if (bitsPlotDebug[1]) StdOut() << "* CENTER OF DETECTED TARGETS\n";
    if (bitsPlotDebug[2]) StdOut() << "* TRANSITIONS ON CIRCLES AROUND TARGETS\n";
    if (bitsPlotDebug[3]) StdOut() << "* AXIS LINES ON CHESSBOARDS\n";
    if (bitsPlotDebug[4]) StdOut() << "* DATA POINTS TO FIT ELLIPSE\n";
    if (bitsPlotDebug[5]) StdOut() << "* FITTED ELLIPSE ON CHESSBOARD\n";
    if (bitsPlotDebug[6]) StdOut() << "* INTERSECTIONS BETWEEN ELLIPSE AND AXES\n";
    if (bitsPlotDebug[7]) StdOut() << "* DETECTED TARGET FRAMES\n";
    if (bitsPlotDebug[8]) StdOut() << "* DETECTED TARGET CODE NAMES\n";
    if (bitsPlotDebug[9]) StdOut() << "* RECTIFIED IMAGES OF DETECTED TARGETS\n";
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
            if (!TestDirDCT(*aPtrDCT,APBI_Im(), mRayMinCB, 1.0, vec2plot)){
                aPtrDCT->mState = eResDCT::BadDir;
            }else{
                mVDCTOk.push_back(aPtrDCT);

                // ----------------------------------------------------------------------------------------------------------
                // [004] 0000000100 plot only transitions on circles around candidate targets (yellow pixels)
                // ----------------------------------------------------------------------------------------------------------
                if (bitsPlotDebug[2]){
                    for (unsigned i=0; i<vec2plot.size(); i++){
                        plotSafeRectangle(mImVisu, vec2plot.at(i), 0, cRGBImage::Yellow, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                    }
                }
            }
        }
        vec2plot.clear();
    }


     ShowStats("ExtractDir");
     StdOut()  << "MAINTAINED " << mVDCTOk.size() << "\n";


     // ----------------------------------------------------------------------------------------------
     // [512] 1000000000 plot rectified images with detected codes (RectifTargets directory)
     // ----------------------------------------------------------------------------------------------
     mOutput_folder = "RectifTargets";
     if (bitsPlotDebug[9]) CreateDirectories(mOutput_folder, true);


    int counter = 0;

    // String for xml output in console



    for (auto aDCT : mVDCTOk){
        analyzeDCT(aDCT, aDIm, counter);
        counter++;
    }


    // ------------------------------------------------
    // Xml output (if needed)
    // ------------------------------------------------
    if (mXml != "") exportInXml(mVDCTOk);

    if (0)
       MarkDCT() ;

    // Plot debug if needed
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
            if (aDCT->mDecodedName != ""){
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
bool cAppliExtractCodeTarget::analyzeDCT(cDCT* aDCT, const cDataIm2D<float> & aDIm, int counter){

    aDCT->mFinalState = false;

    // -----------------------------------------------------------------------
    // Functions parameters (to tune)
    // -----------------------------------------------------------------------
    int px_binarity = 25;      // Threshold on binarity
    int limit_border = 30;     // Target not consider if that close to border
    // -----------------------------------------------------------------------

    std::bitset<10> bitsPlotDebug = std::bitset<10>(mDebugPlot);


    // --------------------------------------------------------------------------------------------------------
    // Build list of codes from specification file
    // --------------------------------------------------------------------------------------------------------

    std::string CODES[2*spec.NbCodeAvalaible()+1];
    for (int i=0; i<2*spec.NbCodeAvalaible()+1; i++){
        CODES[i] = "NA";
    }

    for (int aNum=0 ; aNum<spec.NbCodeAvalaible(); aNum++){
        std::vector<int> code = spec.CodesOfNum(aNum).CodeOfNumC(0).ToVect();
        std::vector<int> binary_code;
        int sum = 0;
        binary_code.push_back(0); binary_code.push_back(0); binary_code.push_back(0); binary_code.push_back(0);
        binary_code.push_back(0); binary_code.push_back(0); binary_code.push_back(0); binary_code.push_back(0);
        binary_code.push_back(0);
        for (unsigned i=0; i<code.size(); i++){
            binary_code.at(code.at(i)) = 1;
            sum += pow(2, code.at(i));
        }

        if (spec.mModeFlight){
            CODES[aNum] = spec.NameOfNum(aNum);
        }else{
             CODES[sum] = spec.NameOfNum(aNum);   // Attention : problème entre ces deux lignes !!!!!
        }
    }



    cPt2di center = aDCT->Pix();

    // -------------------------------------------
    // Test on binarity of target
    // -------------------------------------------
    if (aDCT->mVWhite - aDCT->mVBlack < px_binarity) return false;

    // ----------------------------------------------------------------------------------------------
    // [001] 0000000001 plot only candidates after filtering operations (magenta pixels)
    // ----------------------------------------------------------------------------------------------
    if(bitsPlotDebug[0]) mImVisu.SetRGBrectWithAlpha(center, 1, cRGBImage::Magenta, 0.0);


    // -----------------------------------------------------------------
    // Testing borders
    // -----------------------------------------------------------------
    if (center.x() < limit_border) return false;
    if (center.y() < limit_border) return false;
    if (aDIm.Sz().x()-center.x() < limit_border) return false;
    if (aDIm.Sz().y()-center.y() < limit_border) return false;

    std::vector<cPt2dr> POINTS = extractButterflyEdge(aDIm, aDCT);

    if (POINTS.size() < 10) return false;


    // -----------------------------------------------------------------
    // Ellipse fit
    // -----------------------------------------------------------------
    double param[6];
    if (fitEllipse(POINTS, aDCT->mPt, mConstrainCenter, param) < 0) return false;

    double ellipse[5];
    cartesianToNaturalEllipse(param, ellipse);

    // -----------------------------------------------------------------


    // Invalid ellipse fit
    if (ellipse[0] != ellipse[0])  return false;
    if (ellipse[2]/ellipse[3] > mMaxEcc) return false;


    std::vector<cPt2di> ELLIPSE_TO_PLOT;
    for (double t=0; t<2*PI; t+=0.01){
        cPt2dr aPoint = generatePointOnEllipse(ellipse, t, 0.0);
        cPt2di pt = cPt2di(aPoint.x(), aPoint.y());
        ELLIPSE_TO_PLOT.push_back(pt);
    }


    // Solve intersections
    std::vector<cPt2dr> Y = solveIntersections(aDCT, param);
    if (Y.size() == 0) return false;


    // Affinity estimation
    std::vector<double> transfo = estimateRectification(Y, spec.mChessboardAng);
    double a11 = transfo[0]; double a12 = transfo[1]; double bx  = transfo[2];
    double a21 = transfo[3]; double a22 = transfo[4]; double by  = transfo[5];

    // StdOut() << "AFFINITY RMSE: " << transfo[6] << " PX\n";

    if ((isnan(a11)) || (isnan(a12)) || (isnan(a21)) || (isnan(a22)) || (isnan(bx)) || (isnan(by))){
        return false;
    }

    // ---------------------------------------------------
    // Recomputing directions and intersections if needed
    // ---------------------------------------------------
    double size_target_ellipse = sqrt(ellipse[2]* ellipse[2] + ellipse[3]*ellipse[3]);



    if ((size_target_ellipse > 30) && (mRecompute)){

        // Recomputing directions if needed
        double correction_factor = std::min(size_target_ellipse/15.0, 10.0);
        StdOut() << "\nSIZE OF TARGET: " << size_target_ellipse << " - RECOMPUTING DIRECTIONS WITH FACTOR " << correction_factor << "\n";

        TestDirDCT(*aDCT, APBI_Im(), mRayMinCB, correction_factor, vec2plot);

        // Recomputing intersections if needed
        Y = solveIntersections(aDCT, param);

        // Recomputing affinity if needed
        std::vector<double> transfo = estimateRectification(Y, spec.mChessboardAng);
        double a11 = transfo[0];
        double a12 = transfo[1];
        double a21 = transfo[2];
        double a22 = transfo[3];
        double bx  = transfo[4];
        double by  = transfo[5];

        if ((isnan(a11)) || (isnan(a12)) || (isnan(a21)) || (isnan(a22)) || (isnan(bx)) || (isnan(by))){
            return false;
        }

    }



    // ======================================================================
    // Image generation
    // ======================================================================

    std::vector<cPt2di> FRAME_TO_PLOT;

    int Ni = spec.mModeFlight?640:600;
    int Nj = spec.mModeFlight?1280:600;
    double irel, jrel, it, jt;
    tImTarget aImT(cPt2di(Ni, Nj));
    tDataImT  & aDImT = aImT.DIm();



    if (!spec.mModeFlight){

        // -------------------------------------------------------------
        // Standard case
        // -------------------------------------------------------------

        for (int i=0; i<Ni; i++){
            for (int j=0; j<Nj; j++){
                irel = +6*((double)i-Ni/2.0)/Ni;
                jrel = -6*((double)j-Nj/2.0)/Nj;
                it = a11*irel + a12*jrel + bx;
                jt = a21*irel + a22*jrel + by;
                if ((it < 0) || (jt < 0) || (it >= aDIm.Sz().x()-1) || (jt >= aDIm.Sz().y()-1)){
                    continue;
                }
                aDImT.SetV(cPt2di(i,j), aDIm.GetVBL(cPt2dr(it, jt)));
                if ((i == 0) || (j == 0) || (i == Ni-1) || (j == Nj-1)){
                    FRAME_TO_PLOT.push_back(cPt2di(it, jt));
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
                irel = +3.2*((double)i-Ni/2.0)/Ni;
                jrel = -6.3*((double)j-Nj/2.0)/Nj;
                it = a11*irel + a12*jrel + bx;
                jt = a21*irel + a22*jrel + by;
                if ((it < 0) || (jt < 0) || (it >= aDIm.Sz().x()-1) || (jt >= aDIm.Sz().y()-1)){
                    continue;
                }
                aDImT.SetV(cPt2di(i,j), aDIm.GetVBL(cPt2dr(it, jt)));
                if ((i == 0) || (j == 0) || (i == Ni-1) || (j == Nj-1)){
                    FRAME_TO_PLOT.push_back(cPt2di(it, jt));
                }
            }
        }
    }

    std::string code_binary;
    int code = decodeTarget(aDImT, aDCT->mVWhite, aDCT->mVBlack, code_binary, spec.mModeFlight);

    std::string chaine = "NA";
    if (code != -1){
        chaine = CODES[code];
        if (mToRestrict.size() > 0){
            if (std::find(mToRestrict.begin(), mToRestrict.end(), chaine) == mToRestrict.end()){
                return false;
            }
        }
    }else{
        return false;
    }

    // Control on center position
    if (abs(ellipse[0] - aDCT->mPt.x()) > 2.0) return false;
    if (abs(ellipse[1] - aDCT->mPt.y()) > 2.0) return false;

    // --------------------------------------------------------------------------------
    // Begin print console
    // --------------------------------------------------------------------------------
    double x_centre_moy = (ellipse[0] + aDCT->mPt.x())/2.0;
    double y_centre_moy = (ellipse[1] + aDCT->mPt.y())/2.0;

    StdOut() << " [" << counter << "]" << " Centre: [";
    StdOut() << x_centre_moy << "," << y_centre_moy << "]  -  ";
    StdOut() << code_binary;




    // --------------------------------------------------------------------------------
    // End print console
    // --------------------------------------------------------------------------------
    std::string name_file = "target_" + chaine + ".tif";
    if (chaine == "NA"){
    //  continue; // To avoid printing failures
        name_file = "failure_" + std::to_string(counter) + ".tif";
    }else{
        aDCT->mDecodedName = chaine;
        aDCT->mRefinedCenter = cPt2dr(x_centre_moy, y_centre_moy);
    }

    StdOut() << "  ->  " << name_file << "\n";
    // --------------------------------------------------------------------------------


    // ==========================================================================================================
    // Begin plot debug images
    // ==========================================================================================================

    // ----------------------------------------------------------------------------------------------------------
    // [002] 0000000010 plot only center of detected targets (green pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[1]){
        plotSafeRectangle(mImVisu, center, 1, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
    }

    // ----------------------------------------------------------------------------------------------------------
    // [008] 0000001000 plot only axis lines of dected chessboard patterns (green lines)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[3]){
        for (int sign=-1; sign<=1; sign+=2){
            for (int i=1; i<=size_target_ellipse; i++){
                cPt2di center = aDCT->Pix();
                cPt2di p1 = cPt2di(center.x()+sign*aDCT->mDirC1.x()*i, center.y()+sign*aDCT->mDirC1.y()*i);
                cPt2di p2 = cPt2di(center.x()+sign*aDCT->mDirC2.x()*i, center.y()+sign*aDCT->mDirC2.y()*i);
                plotSafeRectangle(mImVisu, p1, 0, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
                plotSafeRectangle(mImVisu, p2, 0, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
            }
        }
    }


    // ----------------------------------------------------------------------------------------------------------
    // [032] 0000100000 plot only fitted ellipse (red lines)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[5]){
        for (unsigned i=0; i<ELLIPSE_TO_PLOT.size(); i++){
            plotSafeRectangle(mImVisu, ELLIPSE_TO_PLOT.at(i), 0, cRGBImage::Red, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [016] 0000010000 plot only data point for ellipse fit (cyan pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[4]){
        for (unsigned i=0; i<POINTS.size(); i++){
            plotSafeRectangle(mImVisu, cPt2di(POINTS.at(i).x(), POINTS.at(i).y()), 0.0, cRGBImage::Cyan, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [064] 0001000000 plot only intersections between ellipse and axes (blue pixels)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[6]){
        for (unsigned i=0; i<Y.size(); i++){
            cPt2di p = cPt2di((int)(Y.at(i).x()), (int)(Y.at(i).y()));
            plotSafeRectangle(mImVisu, p, 0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [128] 0010000000 plot only detected target frames (blue lines)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[7]){
        for (unsigned i=0; i<FRAME_TO_PLOT.size(); i++){
            plotSafeRectangle(mImVisu, FRAME_TO_PLOT.at(i), 0.0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
        }
    }

    // ----------------------------------------------------------------------------------------------------------
    // [256] 0100000000 plot detected target code name (cyan characters)
    // ----------------------------------------------------------------------------------------------------------
    if (bitsPlotDebug[8]){
        it = 4*(a11 + a12) + bx; jt = 4*(a21 + a22) + by;
        for (int lettre=0; lettre<2; lettre++){
            std::string aStr; aStr.push_back(chaine[lettre]);
            cIm2D<tU_INT1> aImStr = ImageOfString_10x8(aStr,1); cDataIm2D<tU_INT1>&  aDataImStr = aImStr.DIm();
            for (int i=0; i<11; i++){
                for (int j=0; j<11; j++){
                    if (aDataImStr.DefGetV(cPt2di(i,j),0)){
                        plotSafeRectangle(mImVisu, cPt2di(it + i + lettre*10, jt + j), 0.0, cRGBImage::Cyan, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------------
    // [512] 1000000000 plot rectified images with detected codes (RectifTargets directory)
    // ----------------------------------------------------------------------------------------------
    if (bitsPlotDebug[9]) aImT.DIm().ToFile(mOutput_folder+ "/" + name_file);

    // --------------------------------------------------------------------------------
    // End plot image
    // --------------------------------------------------------------------------------

    aDCT->mFinalState = true;
    return true;

}



// ---------------------------------------------------------------------------
// Function to plot "safe" rectangle in image
// ---------------------------------------------------------------------------
bool cAppliExtractCodeTarget::plotSafeRectangle(cRGBImage image, cPt2di point, double sz, cPt3di color, int imax, int jmax, double transparency){
    if ((point.x() < imax) && (point.y() < jmax)){
        if ((point.x() >= 0) && (point.y() >= 0)){
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
// Outputs:
// ---------------------------------------------------------------------------
std::vector<cPt2dr> cAppliExtractCodeTarget::extractButterflyEdge(const cDataIm2D<float>& aDIm, cDCT* aDCT){
    std::vector<cPt2dr> POINTS;
    double threshold = (aDCT->mVBlack + aDCT->mVWhite)/2.0;
    cPt2di center = aDCT->Pix();
    double vx1 = aDCT->mDirC1.x(); double vy1 = aDCT->mDirC1.y();
    double vx2 = aDCT->mDirC2.x(); double vy2 = aDCT->mDirC2.y();
     for (double t=mMargin; t<1-mMargin; t+=0.01){
        for (int sign=-1; sign<=1; sign+=2){
            double z_prec = 0; cPt2dr pf_prec = cPt2dr(0,0);
            double z_curr = 0; cPt2dr pf_curr = cPt2dr(0,0);
            for (int i=5; i<=300; i++){
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
                    break;
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

   // double int_circle_th = 0.25*thb + 0.75*thw;

    double R1 = 137.5*(1+0.35/2);
    double R2 = 137.5*(1+3*0.35/2);

    double cx = 300;
    double cy = 300;

    double threshold = (thw + thb)/2.0;

    double th_std_white_circle = 35;

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

    //if ((M < int_circle_th) | (S > th_std_white_circle)){
    if (S > th_std_white_circle){
        return -1;
    }

    // ====================================================================
    // Code central symetry control and decoding
    // ====================================================================

    int bits = 0; bool bit; double max_error = 0; debug = "";

    // -------------------------------------------------------------
    // Standard case
    // -------------------------------------------------------------

    if (!modeFilght){
        for (int t=0; t<9; t++){

            double theta = (1.0/18.0 + t/9.0)*PI;
            double xc_code = 300 + R2*cos(theta);
            double yc_code = 300 + R2*sin(theta);

            double xc_code_opp = 300 - R2*cos(theta);
            double yc_code_opp = 300 - R2*sin(theta);

            // Code reading
            double value1 = aDImT.GetVBL(cPt2dr(xc_code, yc_code));
            double value2 = aDImT.GetVBL(cPt2dr(xc_code_opp, yc_code_opp));

            bit = (value1 + value2)/2.0 > threshold;
            max_error = std::max(max_error, abs(value1-value2));

            bits += (bit ? 0:1)*pow(2,t);
            debug += (bit ? std::string("0 "):std::string("1 "));

            // Image modification (at the end!)
            markImage(aDImT, cPt2di(xc_code    , yc_code)    , 5, bit?0:255);
            markImage(aDImT, cPt2di(xc_code_opp, yc_code_opp), 5, bit?0:255);

        }
    }else {

    // -------------------------------------------------------------
    // Aerial case
    // -------------------------------------------------------------
        bits = 0;


        // Prepare hamming decoder
        cHamingCoder aHC(9-1);

        aDImT.SetV(cPt2di(300,300), 255.0);


        int NbCols = ceil(((double)aHC.NbBitsOut())/2.0);

        int sq_vt = 90;
        int sq_sz = 480/NbCols;
        int idl, idc;
        int px, py;
        double val;

        double mx = 98;
        double my = 636;
        val = aDImT.GetVBL(cPt2dr(mx, my));
        bool hypo = val > threshold;
        markImage(aDImT, cPt2di(mx, my), 10, hypo?0:255);

        StdOut() << "------------------------------------------------------------------------------------\n";

        for (int k=0; k<aHC.NbBitsOut(); k++){
            idc = k % NbCols;
            idl = (k>=NbCols)*1;

            // ---------------------------------------------------
            // Upper part hypothesis
            // ---------------------------------------------------

            if (!hypo){
                px = 125+idc*sq_sz;
                py = 150 + idl*sq_vt;
                val = aDImT.GetVBL(cPt2dr(px, py));
                bit = val > threshold;
                markImage(aDImT, cPt2di(px, py), 10, bit?0:255);
                bits += (bit ? 0:1)*pow(2,aHC.NbBitsOut()-1-k);
                debug = (bit ? std::string("0 "):std::string("1 ")) + debug;
            }else{
            // ---------------------------------------------------
            // Lower part hypothesis
            // ---------------------------------------------------
                px = 125+idc*sq_sz;
                py = 1035 + idl*sq_vt;
                val = aDImT.GetVBL(cPt2dr(px, py));
                bit = val > threshold;
                markImage(aDImT, cPt2di(px, py), 10, bit?0:255);
                bits += (bit ? 0:1)*pow(2,k);
                debug = debug + (bit ? std::string("0 "):std::string("1 "));
            }

        }

        StdOut() << "Hamming code: " << bits;
        bits = aHC.UnCodeWhenCorrect(bits);
        StdOut() << " / Uncoded: " << bits << " / Recoded: " << aHC.Coding(bits) << "\n";

    }

    if (bits == -1) return bits = 0;      // !!!!!!!!!!!!!!!!!!! PROVISOIRE !!!!!!!!!!!!!!!!!!!

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
        benchAffinity();
        //benchEllipse();
        return 0;
    }


   std::string aNameGT = LastPrefix(APBI_NameIm()) + std::string("_GroundTruth.xml");
   if (ExistFile(aNameGT))
   {
      mWithGT = true;
      mGTResSim   = cResSimul::FromFile(aNameGT);

      //  Take into account offset of reading
      for (auto & aGSD : mGTResSim.mVG)
      {
	      aGSD.Translate(-ToR(mBoxTest.P0()));
	      // FakeUseIt(aGSD);
	      // StdOut() << "Bbbb=" << mBoxTest << "\n";
      }
   }

   if (IsInit(&mPatExportF))
       mTestedFilters = SubOfPat<eDCTFilters>(mPatExportF,true);
   StdOut()  << " IIIIm=" << APBI_NameIm()   << " " << aNameGT << "\n";

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

