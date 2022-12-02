#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"
#include "src/Matrix/MMVII_EigenWrap.h"
#include <random>
#include <bitset>
#include <time.h>
#include <typeinfo>


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


cDCT::cDCT(const cPt2di aPt,cAffineExtremum<tREAL4> & anAffEx) :
   mGT       (nullptr),
   mPix0     (aPt),
   mPt       (anAffEx.StdIter(ToR(aPt),1e-2,3)),
   mState    (eResDCT::Ok),
   mScRadDir (1e5),
   mSym      (1e5),
   mBin      (1e5),
   mRad      (1e5)

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

        int ExeOnParsedBox() override;

        void DoExtract();
        void ShowStats(const std::string & aMes) ;
        void MarkDCT() ;
        void SelectOnFilter(cFilterDCT<tREAL4> * aFilter,bool MinCrown,double aThrS,eResDCT aModeSup);

        int fitEllipse(std::vector<cPt2dr>, double*);                       ///< Least squares estimation of an ellipse from 2D points
        int cartesianToNaturalEllipse(double*, double*);                    ///< Convert (A,B,C,D,E,F) ellipse parameters to (x0,y0,a,b,theta)
        cPt2dr generatePointOnEllipse(double*, double, double);             ///< Generate point on ellipse from natural parameters
        int decodeTarget(tDataImT &, double, double, std::string&, bool);   ///< Decode a potential target
        bool markImage(tDataImT &, cPt2di, int, int);                       ///< Plot mark on gray level image
        std::vector<cPt2dr> solveIntersections(cDCT*, double*);
        std::vector<double> estimateAffinity(double, double, double, double, double, double, double, double, double);
        void printMatrix(MatrixXd);
        bool plotSafeRectangle(cRGBImage, cPt2di, double, cPt3di, int, int, double);


	std::string mNameTarget;

	cParamCodedTarget        mPCT;
    double                   mDiamMinD;
	cPt2dr                   mRaysTF;

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
        bool mXml;                ///< Print xml output in console
        int mDebugPlot;           ///< Debug plot code (binary)


	std::vector<double>  mParamBin;


};



/* *************************************************** */
/*                                                     */
/*              cAppliExtractCodeTarget                */
/*                                                     */
/* *************************************************** */

cAppliExtractCodeTarget::cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(5000,5000),cPt2di(300,300),false), // static_cast<cMMVII_Appli & >(*this))
   mDiamMinD      (40.0),
   mRaysTF        ({4,8}),
   mImGrad        (cPt2di(1,1)),
   mR0Sym         (3.0),
   mR1Sym         (8.0),
   mRExtreSym     (9.0),
   mTHRS_Sym      (0.7),
   mTHRS_Bin      (0.6),
   mImVisu        (cPt2di(1,1)),
   mPatExportF    ("XXXXXXXXX"),
   mWithGT        (false),
   mDMaxMatch     (2.0),
   mTest          (false),
   mRecompute     (false),
   mXml           (false)
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
                    << AOpt2007(mRaysTF, "RayTF","Rays Min/Max for testing filter",{eTA2007::HDV,eTA2007::Tuning})
                    << AOpt2007(mPatExportF, "PatExpF","Pattern export filters" ,{AC_ListVal<eDCTFilters>(),eTA2007::HDV})
                    << AOpt2007(mTest, "Test", "Test for Ellipse Fit", {eTA2007::HDV})
                    << AOpt2007(mParamBin, "BinF", "Param for binary filter", {eTA2007::HDV})
                    << AOpt2007(mRecompute, "Recompute", "Recompute affinity if needed", {eTA2007::HDV})
                    << AOpt2007(mXml, "Xml", "Print xml outout in console", {eTA2007::HDV})
                    << AOpt2007(mDebugPlot, "Debug", "Plot debug image with options", {eTA2007::HDV})
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
   StdOut() <<  aMes << " NB DCT = " << aNbOk << " Prop " << (double) aNbOk / (double) APBI_DIm().NbElem() ;

/*
  if (mWithGT && )
  {
        //for (auto & aGSD : mGTResSim.mVG)
  }
*/

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
             mImVisu.SetRGBrectWithAlpha(aDCT->Pix0(), 1, aCoul, 0.0);
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
     cWhitchMin<cDCT*,double>  aWMin(nullptr,1e10);

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

void  cAppliExtractCodeTarget::DoExtract(){


    cParamCodedTarget spec;
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


    // --------------------------------------------------------------------------------------------------------

     tDataIm &  aDIm = APBI_DIm();
     tIm        aIm = APBI_Im();
     mImVisu =   RGBImFromGray(aDIm);
     // mNbPtsIm = aDIm.Sz().x() * aDIm.Sz().y();

     // [1]   Extract point that are extremum of symetricity


     // ------------------------------------------------------------------------------------------------------------------
     // New version
     // ------------------------------------------------------------------------------------------------------------------


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
          mVDCT.push_back(new cDCT(aPix,anAffEx));
     }

     if (mWithGT)
     {
        for (auto & aGSD : mGTResSim.mVG)
             MatchOnGT(aGSD);
     }
         //
     ShowStats("Init ");


     //   ====   Symetry filters ====
     for (auto & aDCT : mVDCT){
        if (aDCT->mState == eResDCT::Ok){
           aDCT->mSym = aImSym.DIm().GetV(aDCT->Pix());
           if (aDCT->mSym > mTHRS_Sym){
              aDCT->mState = eResDCT::LowSym;
           }
        }
     }
     ShowStats("LowSym");


     cParamAllFilterDCT aGlobParam;


     //   ==== Binarity filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocBin(aIm,aGlobParam),false,mTHRS_Bin,eResDCT::LowBin);


     //   ==== Radial filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocRad(mImGrad,aGlobParam),false,0.9,eResDCT::LowRad);


     //   ==== Min of symetry ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocSym(aIm,aGlobParam),true,0.8,eResDCT::LowSym);


     mVDCTOk.clear();

    std::vector<cPt2di> vec2plot;



    for (auto aPtrDCT : mVDCT){

        // -------------------------------------------
        // TEST CENTRAGE SUR UNE CIBLE
        // -------------------------------------------
        // if (abs(aPtrDCT->Pix0().x() - 1748) > 2) continue;
        // if (abs(aPtrDCT->Pix0().y() - 3407) > 2) continue;
        // -------------------------------------------

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
     std::string output_folder = "RectifTargets";
     if (bitsPlotDebug[9]) CreateDirectories(output_folder, true);


    int counter = 0;

    // String for xml output in console
    std::string xml_output  = "    <MesureAppuiFlottant1Im>\n";
    xml_output += std::string("        <NameIm>");
    xml_output += std::string("Image_name.tif");
    xml_output += std::string("</NameIm>\n");

    for (auto aDCT : mVDCTOk){

        cPt2di center = aDCT->Pix0();

        // -------------------------------------------
        // Test on binarity of target
        // -------------------------------------------
        if (aDCT->mVWhite - aDCT->mVBlack < 25) continue;

        // ----------------------------------------------------------------------------------------------
        // [001] 0000000001 plot only candidates after filtering operations (magenta pixels)
        // ----------------------------------------------------------------------------------------------
        if(bitsPlotDebug[0]) mImVisu.SetRGBrectWithAlpha(center, 1, cRGBImage::Magenta, 0.0);

        // -----------------------------------------------------------------
        // Testing borders
        // -----------------------------------------------------------------
        if (center.x() < 30) continue;
        if (center.y() < 30) continue;
        if (aDIm.Sz().x()-center.x() < 30) continue;
        if (aDIm.Sz().y()-center.y() < 30) continue;

        double vx1 = aDCT->mDirC1.x(); double vy1 = aDCT->mDirC1.y();
        double vx2 = aDCT->mDirC2.x(); double vy2 = aDCT->mDirC2.y();
        double threshold = (aDCT->mVBlack + aDCT->mVWhite)/2.0;

        std::vector<cPt2dr> POINTS;

        double z_prec = 0; cPt2dr pf_prec = cPt2dr(0,0);
        double z_curr = 0; cPt2dr pf_curr = cPt2dr(0,0);

        for (double t=0.20; t<0.8; t+=0.01){
            for (int sign=-1; sign<=1; sign+=2){
                for (int i=5; i<=300; i++){
                    double vx = t*vx1 + (1-t)*vx2;
                    double vy = t*vy1 + (1-t)*vy2;
                    double x = center.x()+sign*vx*i;
                    double y = center.y()+sign*vy*i;

                    pf_prec = pf_curr;
                    pf_curr = cPt2dr(x, y);

                    if ((x < 0) || (y < 0) || (x >= aDIm.Sz().x()-1) || (y >= aDIm.Sz().y()-1)){
                        continue;
                    }

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

        if (POINTS.size() < 10) continue;


        // -----------------------------------------------------------------
        // Ellipse fit
        // -----------------------------------------------------------------
        double param[6];
        if (fitEllipse(POINTS, param) < 0) continue;

        double ellipse[5];
        cartesianToNaturalEllipse(param, ellipse);
        // -----------------------------------------------------------------

        // Invalid ellipse fit
        if (ellipse[0] != ellipse[0])  continue;


        std::vector<cPt2di> ELLIPSE_TO_PLOT;
        for (double t=0; t<2*PI; t+=0.01){
            cPt2dr aPoint = generatePointOnEllipse(ellipse, t, 0.0);
            cPt2di pt = cPt2di(aPoint.x(), aPoint.y());
            ELLIPSE_TO_PLOT.push_back(pt);
        }


         // Solve intersections
        std::vector<cPt2dr> INTERSECTIONS = solveIntersections(aDCT, param);
        if (INTERSECTIONS.size() == 0) continue;
        double x1 = INTERSECTIONS.at(0).x(); double y1 = INTERSECTIONS.at(0).y(); cPt2di p1 = cPt2di(x1, y1);
        double x2 = INTERSECTIONS.at(1).x(); double y2 = INTERSECTIONS.at(1).y(); cPt2di p2 = cPt2di(x2, y2);
        double x3 = INTERSECTIONS.at(2).x(); double y3 = INTERSECTIONS.at(2).y(); cPt2di p3 = cPt2di(x3, y3);
        double x4 = INTERSECTIONS.at(3).x(); double y4 = INTERSECTIONS.at(3).y(); cPt2di p4 = cPt2di(x4, y4);


        // Affinity estimation
        double theta = PI/4.0 - spec.mChessboardAng;
        std::vector<double> transfo = estimateAffinity(x1, y1, x2, y2, x3, y3, x4, y4, theta);
        double a11 = transfo[0];
        double a12 = transfo[1];
        double a21 = transfo[2];
        double a22 = transfo[3];
        double bx  = transfo[4];
        double by  = transfo[5];

        if ((isnan(a11)) || (isnan(a12)) || (isnan(a21)) || (isnan(a22)) || (isnan(bx)) || (isnan(by))){
            continue;
        }

        // ---------------------------------------------------
        // Recomputing directions and intersections if needed
        // ---------------------------------------------------
        double size_target_ellipse = sqrt(ellipse[2]* ellipse[2] + ellipse[3]*ellipse[3]);
        if ((size_target_ellipse > 30) && (mRecompute)){

            // Recomputing directions if needed
            double correction_factor = std::min(size_target_ellipse/15.0, 10.0);
            //  StdOut() << "\nSIZE OF TARGET: " << size_target_ellipse << " - RECOMPUTING DIRECTIONS WITH FACTOR " << correction_factor << "\n";

            TestDirDCT(*aDCT, APBI_Im(), mRayMinCB, correction_factor, vec2plot);
            vx1 = aDCT->mDirC1.x(); vy1 = aDCT->mDirC1.y();
            vx2 = aDCT->mDirC2.x(); vy2 = aDCT->mDirC2.y();

            // Recomputing intersections if needed
            INTERSECTIONS = solveIntersections(aDCT, param);
            x1 = INTERSECTIONS.at(0).x(); y1 = INTERSECTIONS.at(0).y(); p1 = cPt2di(x1, y1);
            x2 = INTERSECTIONS.at(1).x(); y2 = INTERSECTIONS.at(1).y(); p2 = cPt2di(x2, y2);
            x3 = INTERSECTIONS.at(2).x(); y3 = INTERSECTIONS.at(2).y(); p3 = cPt2di(x3, y3);
            x4 = INTERSECTIONS.at(3).x(); y4 = INTERSECTIONS.at(3).y(); p4 = cPt2di(x4, y4);

            // Recomputing affinity if needed
            transfo = estimateAffinity(x1, y1, x2, y2, x3, y3, x4, y4, theta);
            double a11 = transfo[0];
            double a12 = transfo[1];
            double a21 = transfo[2];
            double a22 = transfo[3];
            double bx  = transfo[4];
            double by  = transfo[5];

            if ((isnan(a11)) || (isnan(a12)) || (isnan(a21)) || (isnan(a22)) || (isnan(bx)) || (isnan(by))){
                continue;
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
        }else{
           // continue;
        }

        // Control on center position
        if (abs(ellipse[0] - aDCT->mPt.x()) > 2.0) continue;
        if (abs(ellipse[1] - aDCT->mPt.y()) > 2.0) continue;

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
            if (mXml){
                xml_output += "        <OneMesureAF1I>\n";
                xml_output += "            <NamePt>" + chaine + "</NamePt>\n";
                xml_output += "            <PtIm>" + std::to_string(x_centre_moy);
                xml_output += " " + std::to_string(y_centre_moy) +"</PtIm>\n";
                xml_output += "        </OneMesureAF1I>\n";
            }
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
                    plotSafeRectangle(mImVisu, cPt2di(center.x()+sign*vx1*i, center.y()+sign*vy1*i), 0, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
                    plotSafeRectangle(mImVisu, cPt2di(center.x()+sign*vx2*i, center.y()+sign*vy2*i), 0, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
                }
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
        // [032] 0000100000 plot only fitted ellipse (red lines)
        // ----------------------------------------------------------------------------------------------------------
        if (bitsPlotDebug[5]){
            for (unsigned i=0; i<ELLIPSE_TO_PLOT.size(); i++){
                plotSafeRectangle(mImVisu, ELLIPSE_TO_PLOT.at(i), 0, cRGBImage::Red, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            }
        }

        // ----------------------------------------------------------------------------------------------------------
        // [064] 0001000000 plot only intersections between ellipse and axes (blue pixels)
        // ----------------------------------------------------------------------------------------------------------
        if (bitsPlotDebug[6]){
            plotSafeRectangle(mImVisu, p1, 0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            plotSafeRectangle(mImVisu, p2, 0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            plotSafeRectangle(mImVisu, p3, 0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            plotSafeRectangle(mImVisu, p4, 0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
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
        if (bitsPlotDebug[9]) aImT.DIm().ToFile(output_folder+ "/" + name_file);

        // --------------------------------------------------------------------------------
        // End plot image
        // --------------------------------------------------------------------------------

        counter++;

    }


    if (0)
       MarkDCT() ;
    // APBI_DIm().ToFile("VisuWEIGHT.tif");

    // Plot debug if needed
    if (mDebugPlot) mImVisu.ToFile("VisuCodeTarget.tif");

    // Print xml if needed
    if (mXml) StdOut() << xml_output;

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
// Function to fit an ellipse on a set of 2D points
// Inputs: a vector of 2D points
// Output: a vector of parameters (A, B, C, D, E, F) of equation:
// Ax2 + Bxy + Cy^2 + Dx+ Ey + F
// ---------------------------------------------------------------------------
// NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES
// (Halir and Flusser, 1998)
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::fitEllipse(std::vector<cPt2dr> points, double* output){

	const unsigned N = points.size();
    cDenseMatrix<double>  D1Wrap(3,N);
    cDenseMatrix<double>  D2Wrap(3,N);
    cDenseMatrix<double>  MWrap(3,3);

    double xmin = 1e300;
    double ymin = 1e300;

    for (unsigned i=0; i<N; i++){
        if (points.at(i).x() < xmin) xmin = points.at(i).x();
        if (points.at(i).y() < ymin) ymin = points.at(i).y();

    }

    for (unsigned i=0; i<N; i++){
        points.at(i).x() -= xmin;
        points.at(i).y() -= ymin;
    }

    for (unsigned i=0; i<N; i++){
        double x = points.at(i).x(); double y = points.at(i).y();
        D1Wrap.SetElem(0,i,x*x);  D1Wrap.SetElem(1,i,x*y);  D1Wrap.SetElem(2,i,y*y);
        D2Wrap.SetElem(0,i,x)  ;  D2Wrap.SetElem(1,i,y  );  D2Wrap.SetElem(2,i,1  );
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

	double A, B, C, D, E, F;
	A = a1;
	B = a2;
	C = a3;
	D = TWrap.GetElem(0,0)*a1 + TWrap.GetElem(1,0)*a2 + TWrap.GetElem(2,0)*a3;
	E = TWrap.GetElem(0,1)*a1 + TWrap.GetElem(1,1)*a2 + TWrap.GetElem(2,1)*a3;
	F = TWrap.GetElem(0,2)*a1 + TWrap.GetElem(1,2)*a2 + TWrap.GetElem(2,2)*a3;

	double u = xmin;
	double v = ymin;

	// Translation
    output[0] = A;
    output[1] = B;
    output[2] = C;
    output[3] = D - B*v - 2*A*u;
    output[4] = E - 2*C*v - B*u;
    output[5] = F + A*u*u  + C*v*v + B*u*v  - D*u - E*v;

    return 0;

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

    double cx = target->Pix0().x();
    double cy = target->Pix0().y();

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
// Function to estimate affinity on 4 points
// ---------------------------------------------------------------------------
// Inputs:
//  - Corner coordinates in pixels x1, y1, x2, y2, x3, y3, x4, y4
//  - Chessboard rotation angle theta
// Outputs:
//  - vector of parameters a11, a12, a21, a22,
// ---------------------------------------------------------------------------
std::vector<double> cAppliExtractCodeTarget::estimateAffinity(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4, double theta){

    std::vector<double> transfo = {0, 0, 0, 0, 0, 0};

    double a11 = (x1 + x2 - x3 - x4)/4.0;
    double a12 = (x1 - x2 - x3 + x4)/4.0;
    double a21 = (y1 + y2 - y3 - y4)/4.0;
    double a22 = (y1 - y2 - y3 + y4)/4.0;
    transfo[4]  = (x1 + x2 + x3 + x4)/4.0;
    transfo[5]  = (y1 + y2 + y3 + y4)/4.0;

    // Chessboard rotation
    transfo[0] =  a11*cos(theta) + a12*sin(theta);
    transfo[1] = -a11*sin(theta) + a12*cos(theta);
    transfo[2] =  a21*cos(theta) + a22*sin(theta);
    transfo[3] = -a21*sin(theta) + a22*cos(theta);


    /*
    double rmse = 0;
    rmse += pow(a11*(+1) + a12*(+1) + transfo[4] - x1, 2);
    rmse += pow(a21*(+1) + a22*(+1) + transfo[5] - y1, 2);
    rmse += pow(a11*(+1) + a12*(-1) + transfo[4] - x2, 2);
    rmse += pow(a21*(+1) + a22*(-1) + transfo[5] - y2, 2);
    rmse += pow(a11*(-1) + a12*(-1) + transfo[4] - x3, 2);
    rmse += pow(a21*(-1) + a22*(-1) + transfo[5] - y3, 2);
    rmse += pow(a11*(-1) + a12*(+1) + transfo[4] - x4, 2);
    rmse += pow(a21*(-1) + a22*(+1) + transfo[5] - y4, 2);

    rmse = sqrt(rmse/8.0);

    StdOut() << "AFFINITY RMSE: " << rmse << " PX\n";
    */


    return transfo;
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



int cAppliExtractCodeTarget::ExeOnParsedBox()
{
   mImGrad    =  Deriche(APBI_DIm(),2.0);
   DoExtract();

   return EXIT_SUCCESS;
}


int  cAppliExtractCodeTarget::Exe()
{


if (mTest){

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


        // Data points generation
         std::vector<cPt2dr> POINTS;
        for (unsigned i=0; i<100; i++){
            POINTS.push_back(generatePointOnEllipse(parameter, RandInInterval(0,2*PI), 0.0));
        }

        double output[6];
        fitEllipse(POINTS, output);

        double solution[5];
        cartesianToNaturalEllipse(output, solution);

        StdOut() <<  ((std::abs(parameter[0] - solution[0]) < std::abs(1e-6*parameter[0]))? "OK" : "NOT OK") << "\n";
        StdOut() <<  ((std::abs(parameter[1] - solution[1]) < std::abs(1e-6*parameter[1]))? "OK" : "NOT OK") << "\n";
        StdOut() <<  ((std::abs(parameter[2] - solution[2]) < std::abs(1e-6*parameter[2]))? "OK" : "NOT OK") << "\n";
        StdOut() <<  ((std::abs(parameter[3] - solution[3]) < std::abs(1e-6*parameter[3]))? "OK" : "NOT OK") << "\n";
        StdOut() <<  ((std::abs(parameter[4] - solution[4]) < std::abs(1e-6*parameter[4]))? "OK" : "NOT OK") << "\n";


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

