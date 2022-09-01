#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"
#include "src/Matrix/MMVII_EigenWrap.h"
#include <random>
#include <time.h>
#include <typeinfo>

#define PI 3.14159265

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

        void TestFilters();
        void DoExtract();
        void ShowStats(const std::string & aMes) ;
        void MarkDCT() ;
        void SelectOnFilter(cFilterDCT<tREAL4> * aFilter,bool MinCrown,double aThrS,eResDCT aModeSup);

        int fitEllipse(std::vector<cPt2dr>, double*);                 ///< Least squares estimation of an ellipse from 2D points
        int cartesianToNaturalEllipse(double*, double*);              ///< Convert (A,B,C,D,E,F) ellipse parameters to (x0,y0,a,b,theta)
        cPt2dr generatePointOnEllipse(double*, double, double);       ///< Generate point on ellipse from natural parameters
        int decodeTarget(tDataImT &, double, double);                 ///< Decode a potential target
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
        std::string    mPatF;
        bool           mWithGT;
        double         mDMaxMatch;

        bool mTest;


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
                    << AOpt2007(mDiamMinD, "DMD","Diam min for detect",{eTA2007::HDV})
                    << AOpt2007(mRaysTF, "RayTF","Rays Min/Max for testing filter",{eTA2007::HDV,eTA2007::Tuning})
                    << AOpt2007(mPatF, "PatF","Pattern filters" ,{AC_ListVal<eDCTFilters>()})
                    << AOpt2007(mTest, "Test", "Test for Ellipse Fit", {eTA2007::HDV})
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

	  /*
          if (aDCT.mState == eResDCT::Divg)    aCoul =  cRGBImage::Red;
          if (aDCT.mState == eResDCT::LowSym)  aCoul =  cRGBImage::Yellow;
          if (aDCT.mState == eResDCT::LowBin)  aCoul =  cRGBImage::Blue;
          if (aDCT.mState == eResDCT::LowRad)  aCoul =  cRGBImage::Cyan;
	  */
          if (aDCT->mState == eResDCT::LowSymMin)  aCoul =  cRGBImage::Red;


          if (aCoul.x() >=0){
             mImVisu.SetRGBrectWithAlpha(aDCT->Pix0(),0,aCoul,0.0);
          }

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
        CODES[sum] = spec.NameOfNum(aNum);
        //StdOut() << spec.NameOfNum(aNum) << " " << binary_code << "\n";
    }

    // --------------------------------------------------------------------------------------------------------

     tDataIm &  aDIm = APBI_DIm();
     tIm        aIm = APBI_Im();
     mImVisu =   RGBImFromGray(aDIm);
     // mNbPtsIm = aDIm.Sz().x() * aDIm.Sz().y();

     // [1]   Extract point that are extremum of symetricity

         //    [1.1]   extract integer pixel
     cIm2D<tREAL4>  aImSym = ImSymetricity(false,aIm,mRayMinCB*0.4,mRayMinCB*0.8,0);  // compute fast symetry
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
     SelectOnFilter(cFilterDCT<tREAL4>::AllocBin(aIm,mRayMinCB*0.4,mRayMinCB*0.8),false,mTHRS_Bin,eResDCT::LowBin);


     //   ====   Radial filters ====
     SelectOnFilter(cFilterDCT<tREAL4>::AllocRad(mImGrad,3.5,5.5,1.0),false,0.5,eResDCT::LowRad);


     // Min of symetry
     SelectOnFilter(cFilterDCT<tREAL4>::AllocSym(aIm,mRayMinCB*0.4,mRayMinCB*0.8,1),true,0.8,eResDCT::LowSym);

     // Min of bin
     SelectOnFilter(cFilterDCT<tREAL4>::AllocBin(aIm,mRayMinCB*0.4,mRayMinCB*0.8),true,mTHRS_Bin,eResDCT::LowBin);



     mVDCTOk.clear();

     for (auto aPtrDCT : mVDCT)
     {
          // if (aPtrDCT->mGT)
          if (aPtrDCT->mState == eResDCT::Ok)
          {
             if (!TestDirDCT(*aPtrDCT,APBI_Im(),mRayMinCB))
                aPtrDCT->mState = eResDCT::BadDir ;
             else
                mVDCTOk.push_back(aPtrDCT);
          }



     }
     ShowStats("ExtractDir");
     StdOut()  << "MAINTAINED " << mVDCTOk.size() << "\n";

     //   ====   MinOf Symetry ====
     //   ====   MinOf Symetry ====


     std::string output_folder = "RectifTargets";

     CreateDirectories(output_folder, true);

    int counter = 0;
    for (auto & aDCT : mVDCT){
        if (aDCT->mState == eResDCT::Ok){
            cPt2di center = aDCT->Pix0();
            double vx1 = aDCT->mDirC1.x(); double vy1 = aDCT->mDirC1.y();
            double vx2 = aDCT->mDirC2.x(); double vy2 = aDCT->mDirC2.y();
            double threshold = (aDCT->mVBlack + aDCT->mVWhite)/2.0;


            std::vector<cPt2dr> POINTS;


            for (int sign=-1; sign<=1; sign+=2){
                for (int i=1; i<=50; i++){
                    plotSafeRectangle(mImVisu, cPt2di(center.x()+sign*vx1*i, center.y()+sign*vy1*i), 0, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
                    plotSafeRectangle(mImVisu, cPt2di(center.x()+sign*vx2*i, center.y()+sign*vy2*i), 0, cRGBImage::Green, aDIm.Sz().x(), aDIm.Sz().y(), 0.5);
                }
            }


            for (double t=0.1; t<0.9; t+=0.01){
                for (int sign=-1; sign<=1; sign+=2){
                    for (int i=5; i<=100; i++){
                        double vx = t*vx1 + (1-t)*vx2;
                        double vy = t*vy1 + (1-t)*vy2;
                        double x = center.x()+sign*vx*i;
                        double y = center.y()+sign*vy*i;
                        cPt2dr pf = cPt2dr(x, y);
                        if ((x < 0) || (y < 0) || (x >= aDIm.Sz().x()-1) || (y >= aDIm.Sz().y()-1)){
                            continue;
                        }
                        double z = aDIm.GetVBL(pf);
                        if (z > threshold){
                            POINTS.push_back(pf); break;
                        }

                    }
                }
            }

            double param[6];
            fitEllipse(POINTS, param);


            double ellipse[5];
            cartesianToNaturalEllipse(param, ellipse);


            if (ellipse[0] != ellipse[0])  continue;


            for (double t=0; t<2*PI; t+=0.01){
                cPt2dr aPoint = generatePointOnEllipse(ellipse, t, 0.0);
                cPt2di pt = cPt2di(aPoint.x(), aPoint.y());
                plotSafeRectangle(mImVisu, pt, 0, cRGBImage::Red, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            }

             // Solve intersections
            double cx = center.x();
            double cy = center.y();
            double A = param[0]; double B = param[1]; double C = param[2];
            double D = param[3]; double E = param[4]; double F = param[5];
            double a1 = A*vx1*vx1 + B*vx1*vy1 + C*vy1*vy1;
            double b1 = 2*A*cx*vx1 + B*(cx*vy1 + cy*vx1) + 2*C*cy*vy1 + D*vx1 + E*vy1;
            double c1 = A*cx*cx + B*cx*cy + C*cy*cy + D*cx + E*cy + F;
            double sqrt_del1 = sqrt(b1*b1-4*a1*c1);
            double t11 = (-b1-sqrt_del1)/(2*a1);
            double t12 = (-b1+sqrt_del1)/(2*a1);

            double a2 = A*vx2*vx2 + B*vx2*vy2 + C*vy2*vy2;
            double b2 = 2*A*cx*vx2 + B*(cx*vy2 + cy*vx2) + 2*C*cy*vy2 + D*vx2 + E*vy2;
            double c2 = A*cx*cx + B*cx*cy + C*cy*cy + D*cx + E*cy + F;
            double sqrt_del2 = sqrt(b2*b2-4*a2*c2);
            double t21 = (-b2-sqrt_del2)/(2*a2);
            double t22 = (-b2+sqrt_del2)/(2*a2);


            double x1 = center.x() + t22*vx2; double y1 = center.y() + t22*vy2; cPt2di p1 = cPt2di(x1, y1);
            double x2 = center.x() + t11*vx1; double y2 = center.y() + t11*vy1; cPt2di p2 = cPt2di(x2, y2);
            double x3 = center.x() + t21*vx2; double y3 = center.y() + t21*vy2; cPt2di p3 = cPt2di(x3, y3);
            double x4 = center.x() + t12*vx1; double y4 = center.y() + t12*vy1; cPt2di p4 = cPt2di(x4, y4);

            plotSafeRectangle(mImVisu, p1, 1, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            plotSafeRectangle(mImVisu, p2, 1, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            plotSafeRectangle(mImVisu, p3, 1, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
            plotSafeRectangle(mImVisu, p4, 1, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);

            // Affinity estimation
            double a11 = (x1 + x2 - x3 - x4)/4.0;
            double a12 = (x1 - x2 - x3 + x4)/4.0;
            double a21 = (y1 + y2 - y3 - y4)/4.0;
            double a22 = (y1 - y2 - y3 + y4)/4.0;
            double bx  = (x1 + x2 + x3 + x4)/4.0;
            double by  = (y1 + y2 + y3 + y4)/4.0;


            double theta = PI/4.0 - spec.mChessboardAng;

            // Chessboard rotation
            double a11t =  a11*cos(theta) + a12*sin(theta);
            double a12t = -a11*sin(theta) + a12*cos(theta);
            double a21t =  a21*cos(theta) + a22*sin(theta);
            double a22t = -a21*sin(theta) + a22*cos(theta);
            a11 = a11t;
            a12 = a12t;
            a21 = a21t;
            a22 = a22t;

            if ((isnan(a11)) || (isnan(a12)) || (isnan(a21)) || (isnan(a22)) || (isnan(bx)) || (isnan(by))){
                continue;
            }

            // Image generation
            int Nout = 600;
            double irel, jrel, it, jt;
            tImTarget aImT(cPt2di(Nout, Nout));
            tDataImT  & aDImT = aImT.DIm();
            for (int i=0; i<Nout; i++){
                for (int j=0; j<Nout; j++){
                    irel = +6*((double)i-Nout/2.0)/Nout;
                    jrel = -6*((double)j-Nout/2.0)/Nout;
                    it = a11*irel + a12*jrel + bx;
                    jt = a21*irel + a22*jrel + by;
                    if ((it < 0) || (jt < 0) || (it >= aDIm.Sz().x()-1) || (jt >= aDIm.Sz().y()-1)){
                        continue;
                    }

                    aDImT.SetV(cPt2di(i,j), aDIm.GetVBL(cPt2dr(it, jt)));

                    if ((i == 0) || (j == 0) || (i == Nout-1) || (j == Nout-1)){
                        plotSafeRectangle(mImVisu, cPt2di(it, jt), 0.0, cRGBImage::Blue, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                    }
                }
            }


            // --------------------------------------------------------------------------------
            // Begin print console
            // --------------------------------------------------------------------------------
            double x_centre_moy = (ellipse[0] + aDCT->mPt.x())/2.0;
            double y_centre_moy = (ellipse[1] + aDCT->mPt.y())/2.0;

            StdOut() << " [" << counter << "]" << " Centre: [";
            StdOut() << x_centre_moy << "," << y_centre_moy << "]  -  ";
            // -------------------------------------------------------------------------------

            int code = decodeTarget(aDImT, aDCT->mVWhite, aDCT->mVBlack);

            if (code == -1){
                continue;
            }

            std::string chaine = CODES[code];
            it = 4*(a11 + a12) + bx;
            jt = 4*(a21 + a22) + by;
            for (int lettre=0; lettre<2; lettre++){

                std::string aStr;
                aStr.push_back(chaine[lettre]);

                cIm2D<tU_INT1> aImStr = ImageOfString_10x8(aStr,1);
                cDataIm2D<tU_INT1>&  aDataImStr = aImStr.DIm();

                for (int i=0; i<11; i++){
                    for (int j=0; j<11; j++){
                        if (aDataImStr.DefGetV(cPt2di(i,j),0)){
                            cPt2di pt = cPt2di(it + i + lettre*10, jt + j);
                            plotSafeRectangle(mImVisu, pt, 0.0, cRGBImage::Cyan, aDIm.Sz().x(), aDIm.Sz().y(), 0.0);
                        }
                    }
                }
            }


            // --------------------------------------------------------------------------------
            // End print console
            // --------------------------------------------------------------------------------
            std::string name_file = "target_" + chaine + ".tif";
            StdOut() << "  ->  " << name_file << "\n";
            // --------------------------------------------------------------------------------

            aImT.DIm().ToFile(output_folder+ "/" + name_file);

        counter++;

        }
    }


     MarkDCT() ;

     mImVisu.ToFile("VisuCodeTarget.tif");

     // APBI_DIm().ToFile("VisuWEIGHT.tif");
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



// ---------------------------------------------------------------------------
// Function to plot "safe" rectanngle in image
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
// Function to inspect a matrix in R
// ---------------------------------------------------------------------------
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
    MatrixXd D1(N,3);
    MatrixXd D2(N,3);
    Eigen::Matrix<double, 3, 3> M;

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
        D1(i,0) = x*x;  D1(i,1) = x*y;  D1(i,2) = y*y;
        D2(i,0) = x  ;  D2(i,1) = y  ;  D2(i,2) = 1  ;
    }

	MatrixXd S1 = D1.transpose() * D1;
    MatrixXd S2 = D1.transpose() * D2;
    MatrixXd S3 = D2.transpose() * D2;
    MatrixXd T  = (-1)*S3.inverse() * S2.transpose();
    MatrixXd M1 = S1 + S2*T;

	for (unsigned i=0; i<3; i++){
		M(0,i) = +M1(2,i)/2.0;
		M(1,i) = -M1(1,i);
		M(2,i) = +M1(0,i)/2.0;
	}

	Eigen::EigenSolver<Eigen::Matrix<double, 3,3>> eigensolver(M);

	auto P = eigensolver.eigenvectors();

	double v12 = P(0,1).real(); double v13 = P(0,2).real();
	double v22 = P(1,1).real(); double v23 = P(1,2).real();
	double v32 = P(2,1).real(); double v33 = P(2,2).real();


	bool cond2 = 4*v12*v32-v22*v22 > 0;
	bool cond3 = 4*v13*v33-v23*v23 > 0;
	int index = cond2*1 + cond3*2;

	double a1 = P(0,index).real();
	double a2 = P(1,index).real();
	double a3 = P(2,index).real();

	double A, B, C, D, E, F;
	A = a1;
	B = a2;
	C = a3;
	D = T(0,0)*a1 + T(0,1)*a2 + T(0,2)*a3;
	E = T(1,0)*a1 + T(1,1)*a2 + T(1,2)*a3;
	F = T(2,0)*a1 + T(2,1)*a2 + T(2,2)*a3;


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
        StdOut() << "Error: bad coefficients for ellipse algebraic equation \n";
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
// Output: decoded id of target (-1 if failed)
// ---------------------------------------------------------------------------
int cAppliExtractCodeTarget::decodeTarget(tDataImT & aDImT, double thw, double thb){

    double int_circle_th = 0.25*thb + 0.75*thw;

    double R1 = 137.5*(1+0.35/2);
    double R2 = 137.5*(1+3*0.35/2);

    double threshold = (thw + thb)/2.0;


    // -------------------------------------------
    // Internal white strip circle control
    // -------------------------------------------
    double M = 0;
    double S = 0;
    int nb_pts = 0;

    for (double t=0; t<2*PI; t+=0.01){

        double xc_int = 300 + R1*cos(t);
        double yc_int = 300 + R1*sin(t);

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

    if ((M < int_circle_th) | (S > 30)){
        return -1;
    }

    // -------------------------------------------
    // Code central symetry control and decoding
    // -------------------------------------------
    int bits = 0;
    for (int t=0; t<9; t++){
        double theta = (1.0/18.0 + t/9.0)*PI;
        double xc_code = 300 + R2*cos(theta);
        double yc_code = 300 + R2*sin(theta);

        double xc_code_opp = 300 - R2*cos(theta);
        double yc_code_opp = 300 - R2*sin(theta);

        // Code reading
        double value1 = aDImT.GetVBL(cPt2dr(xc_code, yc_code));
        double value2 = aDImT.GetVBL(cPt2dr(xc_code_opp, yc_code_opp));

        bits += ((value1 + value2)/2.0 > threshold ? 0:1)*pow(2,t);

        StdOut() << ((value1 + value2)/2.0 > threshold ? 0:1) << " ";

        // Image modification (at the end!)
        aDImT.SetV(cPt2di(xc_code     ,yc_code    ), 255.0);
        aDImT.SetV(cPt2di(xc_code_opp ,yc_code_opp), 255.0);

    }
    return bits;
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


if (mTest){

        std::vector<cPt2dr> POINTS;

        double params[6] = {-0.49610428,   0.67946411,  -0.54056366,   6.55701404,  -6.59996909, -16.53083264};
        double nat_par[5];
        cartesianToNaturalEllipse(params, nat_par);

        nat_par[0] = 0;
        nat_par[1] = 0;
        nat_par[2] = 3;
        nat_par[3] = 1;
		nat_par[4] = 0.76;

        for (double t=0; t<2*PI; t+=0.01){
            cPt2dr aPoint = generatePointOnEllipse(nat_par, t, 0.1);
            POINTS.push_back(aPoint);
        }

        double PARAM[5];
        fitEllipse(POINTS, PARAM);


        double nat_par2[5];
        cartesianToNaturalEllipse(PARAM, nat_par2);


        return 0;

    }


   std::string aNameGT = LastPrefix(APBI_NameIm()) + std::string("_GroundTruth.xml");
   if (ExistFile(aNameGT))
   {
      mWithGT = true;
      mGTResSim   = cResSimul::FromFile(aNameGT);
   }

   mTestedFilters = SubOfPat<eDCTFilters>(mPatF,true);
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
