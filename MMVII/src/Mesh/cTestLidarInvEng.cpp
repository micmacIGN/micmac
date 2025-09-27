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
        cPt3dr mNormal;
        bool   mNormComp;
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


	void AddPt(cPt3dr aPtS,tREAL8 aDepth);
	void CreatePointSphere();

	void ComputeStepPhi();

	void AddLine(int aI0,int aI1);
	void ComputeLines();

        void EstimateNormal_of_1Line(cLine_SLRE &,bool Show=false);
        void EstimateNormal_Glob();

        void MakeImages();


     // --- Mandatory ----
	std::string mNameInputScan;
     // --- Optional ----
        bool                  mShow;          //< do we show msg
	tREAL8                mThsRelStepPhi; //< relative toreshold for phi
        std::vector<int>      mIndParamN;     //< Param for computing normal
        std::vector<int>      mTestLines;     //< Line for which we do test

	cStaticLidarImporter  mSLI;
	tREAL8                mStepPhi;       //< estimation of delta phi
        int                   mStepNormal;
        int                   mK0ComputeN;

	std::vector<cPt_SLRE>    mVPtS;
	std::vector<cPt_SLRE>    mVPtSInit;
	std::vector<cLine_SLRE>  mVLines;
        cPt3dr                   mVert;

        size_t                   mNbPhi;
        size_t                   mNbTeta;

        // std::vector<int>     mIndBegline;     //< indexes of line begining
        // std::vector<int>     mIndNearestLeft; //< indexes of line begining
        //  std::vector<int>     mIndNearestRight; //< indexes of line begining
};

cAppliTestStaticLidarRevEng::cAppliTestStaticLidarRevEng(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mShow            (true),
   mThsRelStepPhi   (4.0),
   mIndParamN       {4,0},
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
           << AOpt2007(mIndParamN,"IndParamN","Indexes or normal computation",{eTA2007::HDV,{eTA2007::ISizeV,"[2,2]"}})
/*
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
    aLine.mNormal = cPt3dr(0,0,0);
    aLine.mNormComp = false;
    mVLines.push_back(aLine);
}

void cAppliTestStaticLidarRevEng::EstimateNormal_of_1Line(cLine_SLRE & aLine,bool Show)
{
    std::vector<cPt3dr> aVNorm;

    for (int aKPt= aLine.mInd0+mK0ComputeN; aKPt + mStepNormal <  aLine.mInd1 ; aKPt+= mStepNormal)
    {
             const auto & aPPrec = mVPtS.at(aKPt);
             const auto & aPCur  = mVPtS.at(aKPt+mStepNormal);

             if (aPPrec.mDepth && aPCur.mDepth)
             {
                 aVNorm.push_back(VUnit(aPPrec.mPtS ^ aPCur.mPtS));
             }

    }

    if (Show && NeverHappens())
    {
       for (int aKPt=0 ; aKPt< 10 ; aKPt++)
       {
           const auto & aP0 = mVPtS.at(aLine.mInd0+aKPt).mPtS;
           const auto & aP1  = mVPtS.at(aLine.mInd0+aKPt+1).mPtS;
           const auto & aP2  = mVPtS.at(aLine.mInd0+aKPt+2).mPtS;

       StdOut()  << "D2=" << VUnit(aP0-aP2)  <<  " V2=" << VUnit(aP0-aP2)  
                 <<  "   ### " 
                 << "D1=" << VUnit(aP0-aP1)  <<  " V1=" << VUnit(aP0-aP1) << "\n";
       }
       StdOut() << "--------------------------------\n"; getchar();
    }


    cPt3dr aNorm = VUnit(cComputeCentroids< std::vector<cPt3dr> >::MedianCentroids(aVNorm));
    for (int aKStep=0 ; aKStep<2 ; aKStep++)
    {
         tREAL8   aSigmaAvg = cComputeCentroids<std::vector<cPt3dr>>::SigmaDist(aVNorm,aNorm,0.666);
         tREAL8   aSigmaElim = cComputeCentroids<std::vector<cPt3dr>>::SigmaDist(aVNorm,aNorm,0.95);

         aNorm = VUnit(cComputeCentroids<std::vector<cPt3dr>>::LinearWeigtedCentroids(aVNorm,aNorm,aSigmaAvg,1.0,aSigmaElim));
    }

    if (Show)
    {
             std::vector<tREAL8> aVRes;
             for (const auto & aPt : aVNorm)
                 aVRes.push_back(Norm2(aNorm-aPt));
             std::sort(aVRes.begin(),aVRes.end());
             StdOut() << " --- NORM --- ";
             for (const auto aKR : {1,2,3,4,10,50,100,(int)aVRes.size()/2} )
                 StdOut() << " [K=" << aKR << " " << aVRes.at(aVRes.size() - aKR) << "]";
             StdOut() << "\n";
    }

    aLine.mNormComp=true;
    aLine.mNormal = aNorm;
}

void cAppliTestStaticLidarRevEng::EstimateNormal_Glob()
{
   for (size_t aKL=0 ; aKL<mVLines.size(); aKL++)
   {
        EstimateNormal_of_1Line(mVLines.at(aKL));
        if (mShow && SelectQAmongN(aKL,10,mVLines.size()))
           StdOut() << "Lines to do " << mVLines.size()-aKL << "\n";
   }

   std::vector<cPt3dr> aVNorm;
   for (size_t aKL1=0 ; aKL1<mVLines.size(); aKL1++)
   {
       //  +-  at 90 degree
       size_t aKL2 =  (aKL1 + mVLines.size() /4) % mVLines.size();
       aVNorm.push_back( VUnit(mVLines.at(aKL1).mNormal ^ mVLines.at(aKL2).mNormal ));
   }
   mVert = VUnit(cComputeCentroids<std::vector<cPt3dr>>::StdRobustCentroid(aVNorm,0.8,2));
   if (mVert.z() < 0)
      mVert = -mVert;

   StdOut() << " VERTICAL=" << mVert << "\n";

   if (mShow)
   {
      cStdStatRes  aStatScal;
      for (const auto& aLine : mVLines)
          aStatScal.Add(std::abs(Scal(mVert,aLine.mNormal)));
      StdOut()  << "=========== Stat  Scal  Vert/Norm ============ \n";
      for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99})
          StdOut() << "  * Scal= " << aProp << " Dt=" << aStatScal.ErrAtProp(aProp) << "\n";
   }


   mVPtSInit = mVPtS;
   cPt3dr aI =  VUnit(cPt3dr(0,1,0)^mVert);
   cPt3dr aJ =  VUnit(mVert^aI);
   tRotR  mRot = tRotR(aI,aJ,mVert,false);

   for (auto & aPts : mVPtS)
   {
        aPts.mPtS = mRot.Inverse(aPts.mPtS);
   }
}

void cAppliTestStaticLidarRevEng::AddPt(cPt3dr aPtS,tREAL8 aDepth)
{
    cPt_SLRE aNewP;
    aNewP.mPtS = aPtS;
    aNewP.mDepth = aDepth;
    aNewP.mDPhiPrec  = -1.0;

    mVPtS.push_back(aNewP);
}


void cAppliTestStaticLidarRevEng::CreatePointSphere()
{
   if (mSLI.mVectPtsTPD.size())  // case we have a Teta-Phi-Dept
   {
      mVPtS.reserve(mSLI.mVectPtsTPD.size());
      for (const auto & aPt : mSLI.mVectPtsTPD)
      {
          tREAL8 aD = aPt.z() ? 1.0 : 0;
          AddPt(spher2cart(cPt3dr(aPt.x(),aPt.y(),aD)),aPt.z());
      }
   }
   else if (mSLI.mVectPtsXYZ.size()) // case we have cartesian
   {
      mVPtS.reserve(mSLI.mVectPtsXYZ.size());
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
   for (size_t aK=1 ; aK< mVPtS.size() ; aK++)
   {
       auto & aPPrec = mVPtS.at(aK-1);
       auto & aPCur  = mVPtS.at(aK);

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

void cAppliTestStaticLidarRevEng::MakeImages()
{
    cPt2di aSz(mNbTeta,mNbPhi);
    cIm2D<tREAL4>  aImTeta(aSz,nullptr,eModeInitImage::eMIA_Null);
    cIm2D<tREAL4>  aImPhi (aSz,nullptr,eModeInitImage::eMIA_Null);


    for (size_t aKT=0 ; aKT<mNbTeta ; aKT++)
    {
         const auto & aLine = mVLines.at(aKT);
         size_t aNbP = std::min(mNbPhi,size_t(aLine.mInd1-aLine.mInd0));

         for (size_t aKP=0 ; aKP<aNbP ; aKP++)
         {
             const auto & aPS = mVPtS.at(aLine.mInd0+aKP);
             cPt2di aP_tp(aKT,aKP);
/*
if (aP_tp==cPt2di(5000,2000))
{
   StdOut() << " C2S : " << aPS.mPtS << " => " << aPS.mDepth << "\n";
}
*/
             if (aPS.mDepth)
             {
                 cPt3dr  aCoordSpher = cart2spher(aPS.mPtS);
                 aImTeta.DIm().SetV(aP_tp,aCoordSpher.x());
                 aImPhi.DIm().SetV(aP_tp,aCoordSpher.y());
             }
         }
    }

    aImTeta.DIm().ToFile("RevIng/" + mNameInputScan + "_teta.tif");
    aImPhi.DIm().ToFile("RevIng/" + mNameInputScan + "_phi.tif");
    // aImTeta.ToFile("RevIng/" + mNameInputScan + "_teta.tif");

    // cIm2D<tREAL4>  aImPhi;
}



/*
int  KthSelectQAmonN(int aKTh,int aQ,int aN,tREAL8 aPhase=0.5)
{
    int aRes = round_ni( ((aKTh+ aPhase) /aQ) * aN);

    return std::max(0,std::min(aRes,aN-1));
}
*/

void cAppliTestStaticLidarRevEng::ComputeLines()
{	
   tREAL8 aThrshDT =  mThsRelStepPhi * mStepPhi;



   // ---------------------- compute the lines, "jump" in mDPhiPrec  --------------------------
   int aKPrec=0;
   for (size_t aK=1 ; aK< mVPtS.size() ; aK++)
   {
       const auto & aPPrec = mVPtS.at(aK-1);
       const auto & aPCur  = mVPtS.at(aK);

       if ( aPPrec.mDepth && aPCur.mDepth)
       {
	   if (aPCur.mDPhiPrec >aThrshDT)
	   {
	      AddLine(aKPrec,aK-1);
              aKPrec = aK;
	   }
       }
   }
   AddLine(aKPrec,mVPtS.size());

   mNbTeta =  mVLines.size();
   cStdStatRes aStatLongLine;
   for (const auto & aLine  : mVLines)
       aStatLongLine.Add(aLine.mInd1-aLine.mInd0);
   mNbPhi = aStatLongLine.ErrAtProp(0.5);

   if (mShow)
   {
       StdOut()  << "=========== Stat  Long line , NbL=" <<  mVLines.size() << "============ \n";
       for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99})
           StdOut() << "  * P= " << aProp << " Dt=" << aStatLongLine.ErrAtProp(aProp) << "\n";

       int aNbL   = mTestLines.at(0);
       int aNbPtsByL = mTestLines.at(1);

       StdOut()  <<  " ===============  Test Cste Phi glob ==========\n";
       for (int aKP=1 ; aKP<aNbPtsByL ; aKP++)
       {
           cStdStatRes aStatRes;

           for (int aKL=0 ; aKL < aNbL ; aKL++)
           {
               // int aNumL = round_ni( ((aKL+0.5)/aNbL) * mVLines.size());
               int aNumL = KthSelectQAmonN(aKL,aNbL, mVLines.size());
               const auto & aLine = mVLines.at(aNumL);
               int aNbInThisL = aLine.mInd1-aLine.mInd0;
               int aNumP0 = KthSelectQAmonN(0,aNbPtsByL,aNbInThisL);
               int aNumP1 = KthSelectQAmonN(aKP,aNbPtsByL,aNbInThisL);
               //int aNumP0 = round_ni( ((0+0.5)/aNbByL) * aNbInL);
               //int aNumP1 = round_ni( ((aKP+0.5)/aNbByL) * aNbInL);

               cPt3dr aP0 = mVPtS.at(aLine.mInd0+aNumP0).mPtS;
               cPt3dr aP1 = mVPtS.at(aLine.mInd0+aNumP1).mPtS;
               tREAL8 aTeta =  AbsAngle(aP0,aP1);
               aStatRes.Add(aTeta);

           }

           StdOut() << " KP= " << aKP
                    << " StdDev=" << aStatRes.UBDevStd(-1.0) / mStepPhi
                    << " DIFFS=" ;
           for (const auto & aTeta : aStatRes.VRes())
               StdOut() << " " << (aTeta-aStatRes.Avg())/mStepPhi ;
           StdOut() << "\n";
       }

       for (int aKL=0 ; aKL < aNbL ; aKL++)
       {
           int aNumL = KthSelectQAmonN(aKL,aNbL, mVLines.size());
           EstimateNormal_of_1Line( mVLines.at(aNumL),true);
       }
   }

}

int cAppliTestStaticLidarRevEng::Exe() 
{
   StdOut() << "BEGIN cAppliTestStaticLidarRevEng \n";

   mStepNormal = mIndParamN.at(0);
   mK0ComputeN = mIndParamN.at(1);;

   mSLI.read(mNameInputScan);

   StdOut() << " done read \n";

   StdOut() << "NBPTS,  XYZ=" << mSLI.mVectPtsXYZ.size() << " TPD=" << mSLI.mVectPtsTPD.size()<< "\n";

   //  --------------------  Compute the value of point put on the sphere -------
   CreatePointSphere();
  
   //  --------------------  Estimate the step in phi ---------------------------
   ComputeStepPhi();

   //  --------------------  Split in lines ----------------
   ComputeLines();

   //  --------------------   ----------------
   EstimateNormal_Glob();
   // ComputeStepPhi();
   //EstimateNormal_Glob();


    MakeImages();


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
