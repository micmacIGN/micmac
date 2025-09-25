#include "cMMVII_Appli.h"
#include "MMVII_StaticLidar.h"
/*
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PointCloud.h"
*/


namespace MMVII
{

/* =============================================== */
/*                                                 */
/*                 cAppliCloudClip                 */
/*                                                 */
/* =============================================== */

/**  A basic application for making "quick& dirty" tests on
 * lidar geometry recover
*/


struct cLine_SLRE
{
    public :
        int mInd0;
        int mInd1;
};

struct cPt_SLRE
{
    public :
       cPt3dr mPtS;      //< Pt on the sphere
       tREAL8 mDepth;    //<  mPtS * mDepth -> original point
       tREAL8 mDPhiPrec; //<  Diff teta / Prec
};

class cAppliTestStaticLidarRevEng : public cMMVII_Appli
{
     public :

        cAppliTestStaticLidarRevEng(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


	void CreatePointSphere();
	void ComputeStepPhi();
	void ComputeLines();

	void AddLine(int aI0,int aI1);
	void AddPt(cPt3dr aPtS,tREAL8 aDepth);

     // --- Mandatory ----
	std::string mNameInputScan;
     // --- Optional ----
        bool                  mShow;          //< do we show
	tREAL8                mThsRelStepPhi; //< relative toreshold for phi
	std::vector<int>      mTestLines;   // Line for which we do test

	cStaticLidarImporter  mSLI;
	tREAL8                mStepPhi;       //< estimation of delta phi


	std::vector<cPt_SLRE>    mPtS;
	std::vector<cLine_SLRE>  mVLines;

        std::vector<int>     mIndBegline;     //< indexes of line begining
        std::vector<int>     mIndNearestLeft; //< indexes of line begining
        std::vector<int>     mIndNearestRight; //< indexes of line begining
};

cAppliTestStaticLidarRevEng::cAppliTestStaticLidarRevEng(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mShow            (true),
   mThsRelStepPhi   (4.0),
   mTestLines       {5,4}     
{
}


cCollecSpecArg2007 & cAppliTestStaticLidarRevEng::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameInputScan,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileCloud})
   ;
}

cCollecSpecArg2007 & cAppliTestStaticLidarRevEng::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
	   /*
           << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
           << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file if correction are done")
           << AOpt2007(mDo2D,"Do2DC","check also as a 2D-triangulation (orientation)",{eTA2007::HDV})
           << AOpt2007(mDoCorrect,"Correct","Do correction, Defaut: Do It Out specified")
	   */
   ;
}

void cAppliTestStaticLidarRevEng::AddLine(int aI0,int aI1)
{
    cLine_SLRE aLine;
    aLine.mInd0 = aI0;
    aLine.mInd1 = aI1;
    mVLines.push_back(aLine);
}

void cAppliTestStaticLidarRevEng::AddPt(cPt3dr aPtS,tREAL8 aDepth)
{
    cPt_SLRE aNewP;
    aNewP.mPtS = aPtS;
    aNewP.mDepth = aDepth;
    aNewP.mDPhiPrec  = -1.0;

    mPtS.push_back(aNewP);
}


void cAppliTestStaticLidarRevEng::CreatePointSphere()
{
   if (mSLI.mVectPtsTPD.size())  // case we have a Teta-Phi-Dept
   {
      mPtS.reserve(mSLI.mVectPtsTPD.size());
      for (const auto & aPt : mSLI.mVectPtsTPD)
      {
          tREAL8 aD = aPt.z() ? 1.0 : 0;
	  AddPt(spher2cart(cPt3dr(aPt.x(),aPt.y(),aD)),aPt.z());
      }
   }
   else if (mSLI.mVectPtsXYZ.size()) // case we have cartesian
   {
      mPtS.reserve(mSLI.mVectPtsXYZ.size());
      for (const auto & aPt : mSLI.mVectPtsXYZ)
      {
          tREAL8 aN = Norm2(aPt);
          if (aN)
	     AddPt(aPt/aN,aN);
	  else
	     AddPt(cPt3dr(0,0,0),0);
      }
   }
   else
   {
      MMVII_UnclasseUsEr("No data found in " + mNameInputScan);
   }
   StdOut() << " done sphere \n";
}

void cAppliTestStaticLidarRevEng::ComputeStepPhi()
{
   // ---------------------- compute the value of delta teta --------------------------
   cStdStatRes aStatDPhi;
   for (size_t aK=1 ; aK< mPtS.size() ; aK++)
   {
       auto & aPPrec = mPtS.at(aK-1);
       auto & aPCur  = mPtS.at(aK);

       if ( aPPrec.mDepth && aPCur.mDepth )
       {
          tREAL8 aDif = Norm2(aPPrec.mPtS-aPCur.mPtS);
          aPCur.mDPhiPrec = aDif;
          aStatDPhi.Add(aDif);
       }
   }

   mStepPhi =  aStatDPhi.ErrAtProp(0.5);

   if (mShow)
   {
      StdOut()  << "=========== Stat  Delta Phi ============ \n";
      for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99})
          StdOut() << "  * P= " << aProp << " Dt=" << aStatDPhi.ErrAtProp(aProp) << "\n";
   }
}

void cAppliTestStaticLidarRevEng::ComputeLines()
{	
   tREAL8 aThrshDT =  mThsRelStepPhi * mStepPhi;


   // ---------------------- compute the lines, "jump" in mDPhiPrec  --------------------------
   int aKPrec=0;
   for (size_t aK=1 ; aK< mPtS.size() ; aK++)
   {
       const auto & aPPrec = mPtS.at(aK-1);
       const auto & aPCur  = mPtS.at(aK);

       if ( aPPrec.mDepth && aPCur.mDepth)
       {
	   if (aPCur.mDPhiPrec >aThrshDT)
	   {
	      AddLine(aKPrec,aK-1);
              aKPrec = aK;
	   }
       }
   }
   AddLine(aKPrec,mPtS.size());

   if (mShow)
   {
       cStdStatRes aStatLongLine;
       for (const auto & aLine  : mVLines)
           aStatLongLine.Add(aLine.mInd1-aLine.mInd0);
       StdOut()  << "=========== Stat  Long line , NbL=" <<  mIndBegline.size() << "============ \n";
       for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99})
           StdOut() << "  * P= " << aProp << " Dt=" << aStatLongLine.ErrAtProp(aProp) << "\n";

       int aNbL   = mTestLines.at(0);
       int aNbByL = mTestLines.at(1);

       StdOut()  <<  " ===============  Test Cste Phi glob ==========\n";
       for (int aKP=1 ; aKP<aNbByL ; aKP++)
       {
           std::vector<tREAL8> aVTeta;
	   tREAL8 aSumTeta = 0.0;
           for (int aKL=0 ; aKL < aNbL ; aKL++)
           {
               int aNumL = round_ni( ((aKL+0.5)/aNbL) * mVLines.size());
	       const auto & aLine = mVLines.at(aNumL);
	       int aNbInL = aLine.mInd1-aLine.mInd0;
	       int aNumP0 = round_ni( ((0+0.5)/aNbByL) * aNbInL);
	       int aNumP1 = round_ni( ((aKP+0.5)/aNbByL) * aNbInL);

	       cPt3dr aP0 = mPtS.at(aLine.mInd0+aNumP0).mPtS;
	       cPt3dr aP1 = mPtS.at(aLine.mInd0+aNumP1).mPtS;

	       tREAL8 aTeta =  AbsAngle(aP0,aP1);
	       aVTeta.push_back(aTeta);
	       aSumTeta += aTeta;
           }
	   aSumTeta /= aNbL;

	   StdOut() << " KP= " << aKP << " DIFFS=" ;
	   for (const auto & aTeta : aVTeta)
	       StdOut() << " " << (aTeta-aSumTeta)/mStepPhi ;
	   StdOut() << "\n";
       }
   }
}

int cAppliTestStaticLidarRevEng::Exe() 
{
   StdOut() << "BEGIN cAppliTestStaticLidarRevEng \n";

   mSLI.read(mNameInputScan);

   StdOut() << " done read \n";

   StdOut() << "NBPTS,  XYZ=" << mSLI.mVectPtsXYZ.size() << " TPD=" << mSLI.mVectPtsTPD.size()<< "\n";

   //  --------------------  Compute the value of point put on the sphere -------
   CreatePointSphere();
  
   //  --------------------  Estimate the step in phi ---------------------------
   ComputeStepPhi();

   //  --------------------  Split in lines ----------------
   ComputeLines();


#if (0)
   tREAL8 aThrshDT =  mDeltatT * 2.0;


   // ---------------------- compute the value of delta teta --------------------------
   cStdStatRes aStatLongLine;
   mIndBegline.push_back(0);
   for (size_t aK=1 ; aK< mVShere.size() ; aK++)
   {
       const cPt3dr & aPPrec = mVShere.at(aK-1);
       const cPt3dr & aPCur  = mVShere.at(aK);

       if ((!IsNull(aPPrec)) && (!IsNull(aPCur)))
       {
           tREAL8 aDif = Norm2(aPPrec-aPCur);
	   if (aDif >aThrshDT)
	   {
              aStatLongLine.Add(aK-mIndBegline.back());
              mIndBegline.push_back(aK);
	   }
       }
   }
   mIndBegline.push_back(mVShere.size());

   StdOut()  << "=========== Stat  Long line , NbL=" <<  mIndBegline.size() << "============ \n";
   for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99})
       StdOut() << "  * P= " << aProp << " Dt=" << aStatLongLine.ErrAtProp(aProp) << "\n";


   if (0)
   {
       cStdStatRes aStatDistB;
       for (size_t aK=2 ; aK< mVShere.size() ; aK++)
       {
           const cPt3dr & aP0 = mVShere.at(aK-2);
           const cPt3dr & aP1 = mVShere.at(aK-1);
           const cPt3dr & aP2 = mVShere.at(aK-0);
           if ( (!IsNull(aP0)) && (!IsNull(aP1)) && (!IsNull(aP2)) )
	   {
	       cSegmentCompiled<tREAL8,3> aSeg01(aP0,aP1);
	       tREAL8 aDist = aSeg01.Dist(aP2);
	       aStatDistB.Add(aDist);
	   }
       }
       StdOut()  << "=========== Stat  dist Bundle \n";
       for (const auto & aProp : {0.5,0.9,0.99,0.999,0.9995,0.99975} )
            StdOut() << "  * P= " << aProp << " Dt=" << aStatDistB.ErrAtProp(aProp) / mDeltatT << "\n";
   }
#endif

   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_TestSaticLidarRevEng(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliTestStaticLidarRevEng(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecTestLidarRevEng
(
     "TestSLRE",
      Alloc_TestSaticLidarRevEng,
      "Make some quik&dirty test on Static Lidar Reverse Engeneering",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);


};
