#include "cMMVII_Appli.h"
#include "MMVII_StaticLidar.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_ImageInfoExtract.h"

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
       int    mIndTeta;
       int    mIndPhi;
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
        void SetNewAxeZ(const cPt3dr & aPt);
        void EstimateNormal_Glob_ByProdVect();

        void EstimateNormal_ByImageRad(bool ComputeNewNormal,bool Export);

        void MakeIm_TetaPhi(const std::string & aPrefix);
        void MakeIm_Star();


     // --- Mandatory ----
	std::string mNameInputScan;
     // --- Optional ----
        bool                  mShow;          //< do we show msg
	tREAL8                mThsRelStepPhi; //< relative toreshold for phi
        std::vector<int>      mIndParamN;     //< Param for computing normal
        std::vector<int>      mTestLines;     //< Line for which we do test

	cStaticLidarImporter  mSLI;
	tREAL8                mStepPhi;       //< estimation of delta phi
        tRotR                 mRotInit2New;
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
   mTestLines       {5,4},
   mRotInit2New     (tRotR::Identity())
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


   /*  ------------------------------------------- */
   /*   Method for points construction             */
   /*  ------------------------------------------- */


void cAppliTestStaticLidarRevEng::AddPt(cPt3dr aPtS,tREAL8 aDepth)
{
    cPt_SLRE aNewP;
    aNewP.mPtS = aPtS;
    aNewP.mDepth = aDepth;
    aNewP.mDPhiPrec  = -1.0;
    aNewP.mIndTeta   = -1;
    aNewP.mIndPhi    = -1;

    mVPtS.push_back(aNewP);
}


void cAppliTestStaticLidarRevEng::CreatePointSphere()
{
   //  compute the gemetry : point on the sphere + depth
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

   // the segmentation in line using data, does not work,so 4 now, let be very conservative
   MMVII_INTERNAL_ASSERT_User_UndefE(mVPtS.size()==mSLI.mVectPtsLine.size(),"Bad size for mVectPtsLine");
   MMVII_INTERNAL_ASSERT_User_UndefE(mVPtS.size()==mSLI.mVectPtsCol.size(),"Bad size for mVectPtsLine");

   //  now store the index of line & cols
   for (size_t aKPt=0 ; aKPt < mVPtS.size() ; aKPt++)
   {
       mVPtS.at(aKPt).mIndTeta = mSLI.mVectPtsCol.at(aKPt);
       mVPtS.at(aKPt).mIndPhi = mSLI.mVectPtsCol.at(aKPt);
   }

   mVPtSInit = mVPtS;
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
      StdOut()  << " Nb-Pts=" << mVPtS.size()  << " Nb-OK=" << aStatDPhi.NbMeasures() << "\n";
      StdOut()  << "=========== Stat  Delta Phi ============ \n";
      for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99})
          StdOut() << "  * P= " << aProp << " Dt=" << aStatDPhi.ErrAtProp(aProp) << "\n";
   }
}

void cAppliTestStaticLidarRevEng::AddLine(int aI0,int aI1)
{
//  StdOut() << "-----LINE " << aI0 << " " << aI1 << "\n";
    cLine_SLRE aLine;
    aLine.mInd0 = aI0;
    aLine.mInd1 = aI1;
    aLine.mNormal = cPt3dr(0,0,0);
    aLine.mNormComp = false;
    mVLines.push_back(aLine);
}

void cAppliTestStaticLidarRevEng::ComputeLines()
{	
   // ---------------- compute the lines,  corresponding to variation of col/IndTeta  --------------------
   int aKPrec=0;
   for (size_t aK=1 ; aK< mVPtS.size() ; aK++)
   {
       const auto & aPPrec = mVPtS.at(aK-1);
       const auto & aPCur  = mVPtS.at(aK);

       if (aPPrec.mIndTeta != aPCur.mIndTeta)
       {
	      AddLine(aKPrec,aK);
              aKPrec = aK;
       }

/*
   // tREAL8 aThrshDT =  mThsRelStepPhi * mStepPhi;
       if ( aPPrec.mDepth && aPCur.mDepth)
       {
	   if (aPCur.mDPhiPrec >aThrshDT)
	   {
	      AddLine(aKPrec,aK-1);
              aKPrec = aK;
	   }
       }
*/
   }
   AddLine(aKPrec,mVPtS.size());

   {
       mNbTeta =  mVLines.size();
       cStdStatRes aStatLongLine;
       for (const auto & aLine  : mVLines)
           aStatLongLine.Add(aLine.mInd1-aLine.mInd0);
       mNbPhi = aStatLongLine.ErrAtProp(0.5);

       //  for now make very conservative assumption 
   /*
       StdOut()    << " COOl==" << mSLI.MaxLine() << " " << mNbPhi << " " << aStatLongLine.Min()<< " " << aStatLongLine.Max() << "\n";
       StdOut()    << " LiiG==" << mSLI.MaxCol() << " " << mNbTeta << "\n";
       if (mShow)
       {
           StdOut()  << "=========== Stat  Long line , NbL=" <<  mVLines.size() << "============ \n";
           for (const auto & aProp : {0.5,0.1,0.9,0.01,0.99,0.0,1.0})
               StdOut() << "  * P= " << aProp << " Dt=" << aStatLongLine.ErrAtProp(aProp) << "\n";
       }
   */

       MMVII_INTERNAL_ASSERT_User_UndefE(mSLI.MaxLine()+1==(int)mNbPhi,"Incoherent MaxLine/NbPhi");
       MMVII_INTERNAL_ASSERT_User_UndefE(mNbPhi==aStatLongLine.Min(),"Incoherent Min Stat Line");
       MMVII_INTERNAL_ASSERT_User_UndefE(mNbPhi==aStatLongLine.Max(),"Incoherent Max Stat Line");
       MMVII_INTERNAL_ASSERT_User_UndefE(mSLI.MaxCol()+1==(int)mNbTeta,"Incoherent Max Col/NbTeta");
   }

   if (mShow)
   {
       //  Test that globally the phi angles are constant 
       //  For some sample lines, we check that distance to a reference line are  the same on some sampled colum
       int aNbL   = mTestLines.at(0);
       int aNbPtsByL = mTestLines.at(1);

       StdOut()  <<  " ===============  Test Cste Phi glob ==========\n";
       for (int aKP=1 ; aKP<aNbPtsByL ; aKP++)  // parse the lines
       {
           cStdStatRes aStatRes;  //  static on distance to first line

           for (int aKL=0 ; aKL < aNbL ; aKL++)
           {
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

void cAppliTestStaticLidarRevEng::EstimateNormal_of_1Line(cLine_SLRE & aLine,bool Show)
{
    // the normalto the plane must be orthogonal to all consecutive vector,  we consider
    // difference of mStepNormal and not 1 :
    //     *  because of the problem as in "bureauJM.e57" described bellox
    //     * it go faster
    //     * it it possibly less noisy 
    //     * by the way mStepNormal must be small enough because of outlayers
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

    // this code is to activate to illustrate the problem we have we some data,  teta is ~ constant for 
    // odd/even number but there is high variation if we take two consective values
    if (Show  && NeverHappens())
    {
       for (int aKPt=0 ; aKPt< 10 ; aKPt++)
       {
           const auto & aP0 = mVPtS.at(aLine.mInd0+aKPt).mPtS;
           const auto & aP1  = mVPtS.at(aLine.mInd0+aKPt+1).mPtS;
           const auto & aP2  = mVPtS.at(aLine.mInd0+aKPt+2).mPtS;

           // for exemple on data-set "bureauJM.e57" we see that D2 is ~ constant while D1 highly variates

           StdOut()  << "D2=" << VUnit(aP0-aP2)  <<  "   ### " << "D1=" << VUnit(aP0-aP1)   << "\n";
       }
       StdOut() << "--------------------------------\n"; getchar();
    }

    // compute the robust centroid 

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

void cAppliTestStaticLidarRevEng::EstimateNormal_Glob_ByProdVect()
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

   SetNewAxeZ(mVert);
}

void cAppliTestStaticLidarRevEng::SetNewAxeZ(const cPt3dr & aNewZ)
{
   cPt3dr aNewX =  VUnit(cPt3dr(0,1,0)^aNewZ);
   cPt3dr aNewY =  VUnit(aNewZ^aNewX);

   mRotInit2New =   tRotR(aNewX,aNewY,aNewZ,false).MapInverse() *    mRotInit2New;
   for (size_t aKPt =0 ; aKPt<mVPtS.size() ; aKPt++)
   {
         mVPtS.at(aKPt).mPtS = mRotInit2New.Value(mVPtSInit.at(aKPt).mPtS);
   }
}


void cAppliTestStaticLidarRevEng::EstimateNormal_ByImageRad(bool ComputeNewNormal,bool MakeImage)
{
    int aSzRad = 2000;
    tREAL8 aNbGauss = 5.0;
    tREAL8 aFactExag = 10.0;


    tREAL8  aAmplRad = aFactExag / mStepPhi;
    cPt2dr aPMil = cPt2dr(aSzRad,aSzRad) / 2.0;
    cIm2D<tREAL8>  aImRad(cPt2di(aSzRad,aSzRad),nullptr,eModeInitImage::eMIA_Null);
    cIm2D<tREAL8>  aImFiltered = aImRad.Dup();

    for (const auto & aPS : mVPtS)
    {
        if (aPS.mDepth)
        {
             cPt2dr  aPt = Proj(aPS.mPtS);
             aPt =  aPMil + aAmplRad * aPt;
             if (aImRad.DIm().InsideBL(aPt))
                aImRad.DIm().AddVBL(aPt,1.0);
        }
    }
    ExpFilterOfStdDev(aImFiltered.DIm(),aImRad.DIm(),10,aNbGauss*aFactExag);
    if (MakeImage)
    {
        aImRad.DIm().ToFile("RevIng/" + mNameInputScan + "_RadInit.tif");
        aImFiltered.DIm().ToFile("RevIng/" + mNameInputScan + "_RadFiltr.tif");
    }
    if (ComputeNewNormal)
    {
        cPt2dr aPtMax =  ToR(WhichMax(aImFiltered.DIm()));
        StdOut()  <<  "MMMMMMMMMMMMMMMMMMaaxx= " <<  aPtMax << "\n";
        cAffineExtremum<tREAL8>  aAffineEx(aImFiltered.DIm(),aFactExag*2);
        for (int aK=0 ;aK<2 ; aK++)
            aPtMax  = aAffineEx.OneIter(aPtMax);
        StdOut()  <<  "MM Refine MMMMMMMMMaaxx= " <<  aPtMax << "\n";

        aPtMax =  (aPtMax-aPMil) / aAmplRad;



        SetNewAxeZ(VUnit(cPt3dr(aPtMax.x(),aPtMax.y(),1.0)));
    }
}

void cAppliTestStaticLidarRevEng::MakeIm_Star()
{
    // int aNbLine = mVLines.size();

    cPt2di aSz(2*mNbPhi,2*mNbPhi);
    cPt2dr aPMil =  ToR(aSz) / 2.0;
    cIm2D<tREAL4>  aImStar(aSz,nullptr,eModeInitImage::eMIA_Null);
    int aNbLine = 20;
    tREAL8  aFactExag = 1.0;

    for (int aKthLine = 0 ; aKthLine<aNbLine ; aKthLine++)
    {
        int aNumL = KthSelectQAmonN(aKthLine,aNbLine,mVLines.size());
        const auto & aLine = mVLines.at(aNumL);
        std::vector<cPt2dr> aV2d;
        for (int aKPt = aLine.mInd0 ; aKPt<aLine.mInd1 ; aKPt++)
        {
            const auto & aPS =  mVPtS.at(aKPt);
            if (aPS.mDepth)
            {
                cPt3dr aPt = cart2spher (aPS.mPtS);
                tREAL8 aTeta = aPt.x();
                tREAL8 aRho = (M_PI/2.0-aPt.y()) ;
                aV2d.push_back(FromPolar(aRho/mStepPhi,aTeta)+aPMil);
            }
        }
        cSegment2DCompiled<tREAL8> aSeg(aPMil,aV2d.back());
        for (const auto &aPt1 : aV2d)
        {
            cPt2dr aPLoc = aSeg.ToCoordLoc(aPt1);
            cPt2dr  aPMod = aSeg.FromCoordLoc(cPt2dr(aPLoc.x(),aPLoc.y() * aFactExag));
            if (aImStar.DIm().InsideBL(aPMod))
               aImStar.DIm().AddVBL(aPMod,1.0);
        }
    }
    aImStar.DIm().ToFile("RevIng/" + mNameInputScan + "_star.tif" );
}



void cAppliTestStaticLidarRevEng::MakeIm_TetaPhi(const std::string & aPrefix)
{
    cPt2di aSz(mNbTeta,mNbPhi);
    cIm2D<tREAL4>  aImTeta(aSz,nullptr,eModeInitImage::eMIA_Null);
    cIm2D<tREAL4>  aImPhi (aSz,nullptr,eModeInitImage::eMIA_Null);
    cIm2D<tU_INT1> aImMask(aSz,nullptr,eModeInitImage::eMIA_Null);

    cIm2D<tREAL4>  aDifMoyTeta(aSz,nullptr,eModeInitImage::eMIA_Null);

    for (size_t aKT=0 ; aKT<mNbTeta ; aKT++)
    {
         const auto & aLine = mVLines.at(aKT);
         size_t aNbP = std::min(mNbPhi,size_t(aLine.mInd1-aLine.mInd0));

         cWeightAv<tREAL8,tREAL8> aAvgTeta;

         for (size_t aKP=0 ; aKP<aNbP ; aKP++)
         {
             const auto & aPS = mVPtS.at(aLine.mInd0+aKP);
             cPt2di aP_tp(aKT,aKP);
             if (aPS.mDepth)
             {
                 cPt3dr  aCoordSpher = cart2spher(aPS.mPtS);
                 aImTeta.DIm().SetV(aP_tp,aCoordSpher.x());
                 aImPhi.DIm().SetV(aP_tp,aCoordSpher.y());
                 aImMask.DIm().SetV(aP_tp,1);
                 aAvgTeta.Add(1.0,aCoordSpher.x());
             }
         }
         for (size_t aKP=0 ; aKP<aNbP ; aKP++)
         {
             const auto & aPS = mVPtS.at(aLine.mInd0+aKP);
             cPt2di aP_tp(aKT,aKP);
             if (aPS.mDepth)
                 aDifMoyTeta.DIm().SetV(aP_tp,100.0 * (aImTeta.DIm().GetV(aP_tp)-aAvgTeta.Average()));
         }
    }

    aImTeta.DIm().ToFile("RevIng/" + mNameInputScan + aPrefix+ "_teta.tif");
    aImPhi.DIm().ToFile("RevIng/"  + mNameInputScan + aPrefix+ "_phi.tif");
    aDifMoyTeta.DIm().ToFile("RevIng/"  + mNameInputScan + aPrefix+ "_DifTeta.tif");
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


int cAppliTestStaticLidarRevEng::Exe() 
{
   StdOut() << "BEGIN cAppliTestStaticLidarRevEng \n";


   mStepNormal = mIndParamN.at(0);
   mK0ComputeN = mIndParamN.at(1);;

   mSLI.read(mNameInputScan);

   StdOut() << " done read  " << mSLI.mVectPtsLine.size() << " " << mSLI.mVectPtsCol.size() << "\n";

   StdOut() << "NBPTS,  XYZ=" << mSLI.mVectPtsXYZ.size() << " TPD=" << mSLI.mVectPtsTPD.size()<< "\n";

   //  --------------------  Compute the value of point put on the sphere -------
   CreatePointSphere();
  
   //  --------------------  Estimate the step in phi ---------------------------
   ComputeStepPhi();

   //  --------------------  Split in lines ----------------
   ComputeLines();

   //  --------------------   ----------------
   EstimateNormal_Glob_ByProdVect();
   // ComputeStepPhi();
   //EstimateNormal_Glob();

    MakeIm_TetaPhi("Before");
 
    EstimateNormal_ByImageRad(true,true);

    MakeIm_TetaPhi("After");
    MakeIm_Star();

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
