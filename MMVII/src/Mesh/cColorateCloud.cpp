#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PointCloud.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Interpolators.h"


namespace MMVII
{
/*
    To do mark :
*/

typedef cDataMapping<tREAL8,3,3> tProjPC;


class cCameraOrtho
{
    public :
};



class cOrthoProj  : public  tProjPC
{
    public :
       typedef std::vector<cPt3dr> tVecP3;

       cOrthoProj (const cPt3dr & aDir);
        
    private  :
       const  tVecP3 &  Values(tVecP3 &,const tVecP3 & ) const override;
       cPt3dr mDir;
};

cOrthoProj::cOrthoProj (const cPt3dr & aDir) :
   mDir (VUnit(aDir)) // (aDir/aDir.z())
{
}

const  std::vector<cPt3dr> &  cOrthoProj::Values(tVecP3 & aVOut,const tVecP3 & aVIn ) const 
{
   aVOut.clear();
   for (size_t aK=0 ; aK<aVIn.size() ; aK++)
   {
       const cPt3dr & aPIn = aVIn.at(aK);

       aVOut.push_back(cPt3dr(aPIn.x()-aPIn.z()*mDir.x(),aPIn.y()-aPIn.z()*mDir.y(),aPIn.z()));
   }
   return aVOut;
}

///  Class for computing projection of a point cloud

class cProjPointCloud
{
     public :
         /// constructor : memoriez PC, inialize accum, allocate mem
         cProjPointCloud(cPointCloud& aPC,tREAL8 aSurResol,tREAL8 aWeightInit );

	 /// Process on projection for  OR  (1) modify colorization of points (2) 
         void ProcessOneProj(const tProjPC &,tREAL8 aW,bool ModeImage);
         
	 // export the average of radiomeries (in mSumRad) as a field of mPC
         void ColorizePC(); 
     private :
	 // --------- Processed at initialization ----------------
         cPointCloud&           mPC;       ///< memorize cloud point
	 const int              mNbPts;    ///< store number of points
         std::vector<cPt3dr>    mVPtsInit; ///< initial point cloud (stores once  for all in 64-byte, for efficienciency)
         const tREAL8           mSurResol;
	 const tREAL8           mAvgD;       ///< Avg 2D-Distance between points in 3D Cloud
         const tREAL8           mStepProf;  ///< Step for computing depth-images
	 // --------- Updated  with  "ProcessOneProj"  ----------------
         tREAL8                 mSumW;      ///< accumulate sum of weight on radiometries
         std::vector<tREAL4>    mSumRad;    ///< accumulate sum of radiometry
	 // --------- Computed at each run of "ProcessOneProj"  ------------------
         std::vector<cPt3dr>    mVPtsProj;  ///< memorize projections 
         std::vector<cPt2di>    mVPtImages; ///< Projection in image of given 3D Point
         cTplBoxOfPts<int,2>    mBoxInd;    ///< Compute for of mVPtImages
};

cProjPointCloud::cProjPointCloud(cPointCloud& aPC,tREAL8 aSurResol,tREAL8 aWeightInit) :
   mPC       (aPC),
   mNbPts    (aPC.NbPts()),
   mSurResol (aSurResol),
   mAvgD     (std::sqrt(1.0/mPC.Density())),
   mStepProf (mAvgD / mSurResol),

   mSumW     (aWeightInit),
   mSumRad   (mNbPts,0.0)
{
   // reserve size for Pts
   mVPtsInit.reserve(mNbPts);
   mVPtsProj.reserve(mNbPts);

   //  Init mVPtsInit && SumRad
   for (size_t aKPt=0 ; aKPt<aPC.NbPts() ; aKPt++)
   {
       mVPtsInit.push_back(aPC.KthPt(aKPt));
       if (mPC.DegVisIsInit())
          mSumRad.at(aKPt) = aPC.GetDegVis(aKPt) * aWeightInit;
   }
}

void cProjPointCloud::ColorizePC()
{
   // Put mSumRad as an attribute of PC for memorization
   for (size_t aK=0 ; aK<mVPtsProj.size() ; aK++)
   {
       mPC.SetDegVis(aK,mSumRad.at(aK)  / mSumW);
   }
}

void cProjPointCloud::ProcessOneProj(const tProjPC & aProj,tREAL8 aW,bool isModeImage)
{
     mSumW += aW;               // accumlate weight
     tREAL8 aMinInfty = -1e10;  // minus infinity, any value lower than anr real one
     tREAL8 aPlusInfty = - aMinInfty;

     // ========================================================================
     // == [0] ==================  Init proj, indexes, images  =================
     // ========================================================================
     
     //    [0.1] ---  Compute 3D proj+ its 2d-box ----
     aProj.Values(mVPtsProj,mVPtsInit); 
     cTplBoxOfPts<tREAL8,2> aBOP;

     for (const auto & aPt : mVPtsProj)
     {
         aBOP.Add(Proj(aPt));
     }
     cBox2dr aBox = aBOP.CurBox();


     //    [0.2]  ---------- compute the images indexes of points + its box  & sz ---
     mBoxInd= cTplBoxOfPts<int,2> (); //
     mVPtImages.clear();
     for (const auto & aPt : mVPtsProj)
     {
         cPt2di anInd = ToI((Proj(aPt)-aBox.P0()) / mStepProf);  // compute image index
         mBoxInd.Add(anInd); // memo in box
         mVPtImages.push_back(anInd); 
     }

     //    [0.3]  ---------- Alloc images --------------------
     //    [0.3.1]   image of depth
     cPt2di aSzImProf = mBoxInd.CurBox().Sz() + cPt2di(1,1);
     cIm2D<tREAL4> aImDepth(aSzImProf);
     cDataIm2D<tREAL4> & aDImDepth = aImDepth.DIm();
     aDImDepth.InitCste(aMinInfty);


     //    [0.3.2]   image of radiometry
     cPt2di aSzImRad = isModeImage ? aSzImProf : cPt2di(1,1);
     cIm2D<tREAL4> aImRad(aSzImRad);
     cDataIm2D<tREAL4>& aDImRad = aImRad.DIm();
     aDImRad.InitCste(0.0);

     //    [0.3.2]   image of masq
     cIm2D<tREAL4> aImWeigth(aSzImRad);
     cDataIm2D<tREAL4>& aDImWeight = aImWeigth.DIm();
     aDImWeight.InitCste(0.0);


     //    [0.4]  ---------- Alloc vector SzLeaf -> neighboor in image coordinate (time efficiency) ----------------
     std::vector<std::vector<cPt2di>> aVVdisk(256);  // as size if store 8-byte, its sufficient
     for (int aK=0 ; aK<=255 ; aK++)
     {
         tREAL8 aSzL = mPC.ConvertInt2SzLeave(aK);
         aVVdisk.at(aK) = VectOfRadius(-1,aSzL/mStepProf);
     }

     // ==================================================================================================================
     // == [1] ==================   compute the depth image : accumulate for each pixel the maximal depth ================
     // ==================================================================================================================

     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt); // extract index
         // extract depth, supress tiny value, so next time  Depth>stored value even with rounding in store
         tREAL8   aDepth      = mVPtsProj.at(aKPt).z() - mAvgD/100.0; 
         // update depth for all point of the "leaf"
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh : aVDisk)
         {
             cPt2di aPt = aCenter + aNeigh;
             if (aDImDepth.Inside(aPt))
             {
                 aDImDepth.SetMax(aPt,aDepth);
             } 
         }
     }

     // ===========================================================================================================================
     // == [2] ===   for each point use depth image and if it is visible
     //         * in mode std  accumulate its visibility 
     //         * in mode image, project its radiometry
     // ===========================================================================================================================
     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt);
         tREAL8   aDepth      = mVPtsProj.at(aKPt).z();
         int aNbVis = 0;
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh :aVDisk) // parse all point of leaf
         {
             cPt2di aPt = aCenter + aNeigh;
             if (aDImDepth.DefGetV(aPt,aPlusInfty) <= aDepth)  // if the point is visible
             {
                if (isModeImage)  // in mode image udpate radiometry & image
                {
                   aDImWeight.SetV(aPt,1.0);
                   aDImRad.SetV(aPt,mPC.GetDegVis(aKPt)*255);
                }
                else  // in mode standard uptdate visib count
                {
                   aNbVis++;
                }
             } 
         }
         if (!isModeImage)  // in mode std we know the visibility 
         {
            mSumRad.at(aKPt) +=  (aW * aNbVis) / aVDisk.size();
         }
     }
     

     // =====================================================================================
     // == [3] ==================   compute the images (radiom, weight, depth) ==============
     // =====================================================================================

     if (isModeImage)
     {
         tREAL8 aResolImRel = 0.5;
         tREAL8 aStepImAbs =  mAvgD / aResolImRel;
         tREAL8 aResolImaRel = aStepImAbs / mStepProf;
         tREAL8 aSigmaImaFinal = 1.0;
         tREAL8 aSigmaImaInit = aSigmaImaFinal * aResolImaRel;
         int    aNbIter = 5;

         MulImageInPlace(aDImDepth,aDImWeight);
         

         ExpFilterOfStdDev( aDImRad,aNbIter,aSigmaImaInit);
         ExpFilterOfStdDev(aDImWeight,aNbIter,aSigmaImaInit);
         ExpFilterOfStdDev( aDImDepth,aNbIter,aSigmaImaInit);

         for (const auto & aPix : aDImWeight)
         {
            tREAL8 aW =   aDImWeight.GetV(aPix);
            tREAL8 aD =   aDImDepth.GetV(aPix);
            tREAL8 aR =   aDImRad.GetV(aPix);
            aDImRad.SetV(aPix,aW ?  aR/aW : 0.0);
            aDImDepth.SetV(aPix,aW ?  aD/aW : 0.0);
         }
       
        static int aCpt=0; aCpt++;
         
        cPt2di  aSzImFinal = ToI(ToR(aSzImRad)/aResolImaRel);
        cIm2D<tU_INT1>      aIm8B(aSzImFinal);
        cDataIm2D<tU_INT1>& aDIm8B = aIm8B.DIm();
        std::unique_ptr<cDiffInterpolator1D> aInterp (cDiffInterpolator1D::TabulSinC(5));

        for (const auto & aPixI : aDIm8B)
        {
            cPt2dr aPixR = ToR(aPixI) * aResolImaRel;
            aDIm8B.SetVTrunc(aPixI,aDImRad.ClipedGetValueInterpol(*aInterp,aPixR,0));
        }
        aDIm8B.ToFile("IIP_RadOut"+ToStr(aCpt) + ".tif");

 aDImDepth.ToFile("IIP_RadOut_Depth"+ToStr(aCpt) + ".tif");

        StdOut() << "RESOL ;  IMA-REL=" << aResolImaRel << " Ground=" << aStepImAbs << "\n";
     }
}

/* =============================================== */
/*                                                 */
/*             cAppli_MMVII_CloudImProj            */
/*                                                 */
/* =============================================== */

class cAppli_MMVII_CloudImProj : public cMMVII_Appli
{
     public :

        cAppli_MMVII_CloudImProj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // --- Mandatory ----
	std::string   mNameCloudIn;
        // --- Optionnal ----
        std::string   mNameImageOut;

        tREAL8  mSurResolSun;
        cPt3dr      mSun;
        std::string mNameSavePCSun;
};

cAppli_MMVII_CloudImProj::cAppli_MMVII_CloudImProj
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec),
     mSurResolSun      (2.0)
{
}

cCollecSpecArg2007 & cAppli_MMVII_CloudImProj::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
   ;
}

cCollecSpecArg2007 & cAppli_MMVII_CloudImProj::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mNameImageOut,CurOP_Out,"Name of image  file, def= Ima+Input")
          << AOpt2007(mSun,"Sun","Sun : Dir3D=(x,y,1)  ,  Z=WEIGHT !! ")
          << AOpt2007(mNameSavePCSun,"CloudSun","Name of cloud with sun, if sun was added")
   ;
}

int  cAppli_MMVII_CloudImProj::Exe()
{
   if (!IsInit(&mNameImageOut))
      mNameImageOut =  "ImProj_" + LastPrefix(mNameCloudIn) + ".tif";


   cPointCloud   aPC_In ;
   ReadFromFile(aPC_In,mNameCloudIn);

   if  (IsInit(&mSun))
   {
       cProjPointCloud  aPPC(aPC_In,mSurResolSun,1.0);
       aPPC.ProcessOneProj( cOrthoProj  (cPt3dr(mSun.x(),mSun.y(),1.0)), mSun.z(),false);

       aPPC.ColorizePC();
       if (IsInit(&mNameSavePCSun))
           SaveInFile(aPC_In,mNameSavePCSun);
   }

   cProjPointCloud  aPPC(aPC_In,mSurResolSun,1.0);
   for (int aK=-5 ; aK<=5 ; aK++)
       aPPC.ProcessOneProj(cOrthoProj(cPt3dr(aK*0.2,0,1.0)), 0.0,true);

   StdOut() << "NbLeaves "<< aPC_In.LeavesIsInit () << "\n";

   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       MMVII                     */
     /* =============================================== */

tMMVII_UnikPApli Alloc_MMVII_CloudImProj(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_CloudImProj(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MMVII_CloudImProj
(
     "CloudMMVIIImProj",
      Alloc_MMVII_CloudImProj,
      "Generate image projections of coloured point cloud",
      {eApF::Cloud,eApF::Simul},
      {eApDT::MMVIICloud},
      {eApDT::Image},
      __FILE__
);
/*
*/


/* =============================================== */
/*                                                 */
/*                 cAppli_MMVII_CloudColorate      */
/*                                                 */
/* =============================================== */


class cAppli_MMVII_CloudColorate : public cMMVII_Appli
{
     public :

        cAppli_MMVII_CloudColorate(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // --- Mandatory ----
	std::string   mNameCloudIn;
        // --- Optionnal ----
        std::string mNameCloudOut;

        tREAL8   mPropRayLeaf;
        tREAL8   mSurResol;
        int      mNbSampS;
        cPt3dr   mSun;
};

cAppli_MMVII_CloudColorate::cAppli_MMVII_CloudColorate
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec),
     mPropRayLeaf      (1.1),
     mSurResol         (2.0),
     mNbSampS          (5)
{
}

cCollecSpecArg2007 & cAppli_MMVII_CloudColorate::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
   ;
}


cCollecSpecArg2007 & cAppli_MMVII_CloudColorate::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file, def=Colorate_+InPut")
          << AOpt2007(mPropRayLeaf,"RayLeaves","Ray of leaves (/ avg dist)",{eTA2007::HDV})
          << AOpt2007(mSurResol,"SurResol","Sur resol in computation (/ avg dist)",{eTA2007::HDV})
          << AOpt2007(mNbSampS,"NbSampS","Number of sample/face for sphere discretization",{eTA2007::HDV})
          << AOpt2007(mSun,"Sun","Sun : Dir3D=(x,y,1)  ,  Z=WEIGHT !! ")
   ;
}

int  cAppli_MMVII_CloudColorate::Exe()
{
   if (! IsInit(&mNameCloudOut))
      mNameCloudOut = "Colorate_"+ mNameCloudIn;

  
   cAutoTimerSegm aTSRead(TimeSegm(),"Read");
   cPointCloud   aPC_In ;
   ReadFromFile(aPC_In,mNameCloudIn);

   // generate the sz of leaves
   if (! aPC_In.LeavesIsInit())
   {
       aPC_In.SetLeavesUnit(0.05,SVP::Yes);  // fix unit step,
       tREAL8  aRayLeaf  = mPropRayLeaf  / std::sqrt(aPC_In.Density());
       for (size_t aKPt=0 ; aKPt<aPC_In.NbPts() ; aKPt++)
       {
           aPC_In.SetSzLeaves(aKPt,aRayLeaf);
       }
   }

   cAutoTimerSegm aTSInit(TimeSegm(),"Init");
   tREAL8 aWeightInit = (mNbSampS==0);
   cProjPointCloud  aPPC(aPC_In,mSurResol,aWeightInit);  // Weight Init 0 if NbS ,  

    
   cAutoTimerSegm aTSProj(TimeSegm(),"1Proj");

   int aNbStd=0;
   if (mNbSampS)
   {
       aPC_In.SetMulDegVis(1e4);
       cSampleSphere3D aSampS(mNbSampS);
       for (int aK=0 ; aK< aSampS.NbSamples() ; aK++)
       {
           cPt3dr aDir = VUnit(aSampS.KthPt(aK));
           if (aDir.z() >= 0.2)
           {
               aPPC.ProcessOneProj( cOrthoProj(aDir),1.0,false);
               aNbStd++;
               StdOut() << "Still " << aSampS.NbSamples() - aK << "\n";
           }
       }
    }

   if (IsInit(&mSun))
   {
       tREAL8 aW0  = mNbSampS ? aNbStd : 1.0;
       aPPC.ProcessOneProj( cOrthoProj  (cPt3dr(mSun.x(),mSun.y(),1.0)),aW0 * mSun.z(),false);
   }

   aPPC.ColorizePC();
   SaveInFile(aPC_In,mNameCloudOut);


   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_MMVII_CloudColorate(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_CloudColorate(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MMVII_CloudColorate
(
     "CloudMMVIIColorate",
      Alloc_MMVII_CloudColorate,
      "Generate a colorate version of  MMVII-Cloud",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);
#if (0)
#endif


};
