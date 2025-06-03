#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PointCloud.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Interpolators.h"
#include "MMVII_PCSens.h"


namespace MMVII
{
/*
    To do mark :
*/


/* *********************************** */
/*                                     */
/*              cOrthoProj             */
/*                                     */
/* *********************************** */

/**  The orthographic projection as a mapping.  Used as element of cCamOrthoC.  */

class cOrthoProj  :  public  tIMap_R3
{
    public :
       typedef std::vector<cPt3dr> tVecP3;

       cOrthoProj (const tRotR & aRot ,const cPt3dr& aC,const cPt2dr& aPP ,tREAL8 aResol) ;
       cOrthoProj (const cPt3dr & aDir,const cPt3dr& aC =cPt3dr(0,0,0),const cPt2dr& aPP= cPt2dr(0,0) ,tREAL8 aResol=1.0) ;
       tSeg3dr  BundleInverse(const cPt2dr &) const ;

       cOrthoProj(const cOrthoProj&);

       void SetProfIsZIN() {mProfIsZIN=true;}
        
    private  :
       const  tVecP3 &  Values(tVecP3 &,const tVecP3 & ) const override;

       tRotR  mRL2W;
       cPt3dr mC;
       cPt2dr mPP;
       tREAL8 mResol;
       bool   mProfIsZIN;  ///
};

cOrthoProj::cOrthoProj (const tRotR & aRot ,const cPt3dr & aC,const cPt2dr & aPP ,tREAL8 aResol)  :
    mRL2W         (aRot),
    mC            (aC),
    mPP           (aPP),
    mResol        (aResol),
    mProfIsZIN    (false)
{
}

cOrthoProj:: cOrthoProj(const cOrthoProj& anOP) :
    cOrthoProj(anOP.mRL2W,anOP.mC,anOP.mPP,anOP.mResol)
{
}

cOrthoProj::cOrthoProj (const cPt3dr & aDir ,const cPt3dr & aC,const cPt2dr& aPP ,tREAL8 aResol)  :
   cOrthoProj(tRotR::CompleteRON(aDir,2),aC,aPP,aResol)
{
}
/*
*/


const  std::vector<cPt3dr> &  cOrthoProj::Values(tVecP3 & aVOut,const tVecP3 & aVIn ) const 
{
   aVOut.clear();
   for (size_t aK=0 ; aK<aVIn.size() ; aK++)
   {
       const cPt3dr & aPIn = aVIn.at(aK);
       cPt3dr  aPLoc = mRL2W.Inverse(aPIn-mC);
       cPt2dr  aPProj = Proj(aPLoc);
       aPProj = mPP+ aPProj/mResol;

       // aVOut.push_back(TP3z(aPProj,aPIn.z()*0.01));
       tREAL8 aZ = mProfIsZIN ? aPIn.z() : aPLoc.z() ;
       aVOut.push_back(TP3z(aPProj,aZ));

       // aVOut.push_back(TP3z(aPProj,aPLoc.z()));
   }
   return aVOut;
}

tSeg3dr   cOrthoProj::BundleInverse(const cPt2dr & aPIm0) const 
{
    cPt2dr aPIm = (aPIm0-mPP) * mResol;
    cPt3dr aP0 = TP3z(aPIm,-1.0);
    cPt3dr aP1 = TP3z(aPIm, 1.0);

    return tSeg3dr(mRL2W.Value(aP0),mRL2W.Value(aP1));
}


/* *********************************** */
/*                                     */
/*              cCamOrthoC             */
/*                                     */
/* *********************************** */

class cCamOrthoC  :  public  cSensorImage
{
    public :
       cCamOrthoC(const std::string &aName,const cOrthoProj & aProj,const cPt2di & aSz);

 //       cPt2dr Ground2Image(const cPt3dr &) const override;
       const cPixelDomain & PixelDomain() const override;

    private :
       cOrthoProj         mProj;
       cDataPixelDomain   mDataPixDom;
       cPixelDomain       mPixelDomain;
};

cCamOrthoC::cCamOrthoC(const std::string &aNameImage,const cOrthoProj & aProj,const cPt2di & aSz) :
     cSensorImage (aNameImage),
     mProj        (aProj),
     mDataPixDom  (aSz),
     mPixelDomain (&mDataPixDom)
{
}
      
const cPixelDomain & cCamOrthoC::PixelDomain() const { return mPixelDomain; }

//cPt2dr cCamOrthoC::Ground2Image(const cPt3dr &) const { }


/*
       tSeg3dr  Image2Bundle(const cPt2dr &) const override;
       double DegreeVisibility(const cPt3dr &) const override;
       std::string  V_PrefixName() const   override;
       cPt3dr  PseudoCenterOfProj() const ;
*/
/*
*/

/*
cOrthoProj::cOrthoProj (const cPt3dr & aDir,cPt2dr aPP,tREAL8 aResol) :
   mDir   (VUnit(aDir)), // (aDir/aDir.z())
   mPP    (aPP),
   mResol (aResol)
{
}



std::string  cOrthoProj::V_PrefixName() const 
{
    return "CamOrtho";
}
*/

/* ********************************************* */
/*                                               */
/*                cParamProjCloud                */
/*                                               */
/* ********************************************* */

/** Class for parametrization of cProjPointCloud::ProcessOneProj */

class cParamProjCloud
{
   public :
       cParamProjCloud(const tIMap_R3 * aProj) :
           mProj   (aProj),
           mSetOK  (nullptr)
       {
       }
       const tIMap_R3 * mProj;
       tSet_R3 *        mSetOK;
};

///  Class for computing projection of a point cloud

class cProjPointCloud
{
     public :
         /// constructor : memoriez PC, inialize accum, allocate mem
         cProjPointCloud(cPointCloud & aParam,tREAL8 aSurResol,tREAL8 aWeightInit );

	 /// Process on projection for  OR  (1) modify colorization of points (2) 
         void ProcessOneProj(const cParamProjCloud &,tREAL8 aW,bool ModeImage);
         
	 // export the average of radiomeries (in mSumRad) as a field of mPC
         void ColorizePC(); 
     private :
	 // --------- Processed at initialization ----------------
         cPointCloud&           mPC;       ///< memorize cloud point
	 const int              mNbPtsGlob;    ///< store number of points
	 // int                    mNbPts;    ///<  Dynamic, change with SetOk
         std::vector<cPt3dr>    mGlobPtsInit; ///< initial point cloud (stores once  for all in 64-byte, for efficienciency)
         std::vector<cPt3dr> *  mVPtsInit;     /// Dynamic, change with SetOk
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
   mPC        (aPC),
   mNbPtsGlob (aPC.NbPts()),
   mSurResol  (aSurResol),
   mAvgD      (std::sqrt(1.0/mPC.Density())),
   mStepProf  (mAvgD / mSurResol),

   mSumW      (aWeightInit),
   mSumRad    (mNbPtsGlob,0.0)
{
   // reserve size for Pts
   mGlobPtsInit.reserve(mNbPtsGlob);
   mVPtsProj.reserve(mNbPtsGlob);

   //  Init mGlobPtsInit && SumRad
   for (size_t aKPt=0 ; aKPt<aPC.NbPts() ; aKPt++)
   {
       mGlobPtsInit.push_back(aPC.KthPt(aKPt));
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

void cProjPointCloud::ProcessOneProj(const cParamProjCloud & aParam,tREAL8 aW,bool isModeImage)
{
static int aCpt=0 ; aCpt++;
size_t aKBug = 2;
bool aBugCpt = (aCpt==7);
cPt2di aNBug(2,0);
cPt2di aPtBug(852,3371);
// aBugCpt=-1;

     const tIMap_R3 & aProj  = *(aParam.mProj);

     mSumW += aW;               // accumlate weight
     tREAL8 aMinInfty = -1e10;  // minus infinity, any value lower than anr real one
     tREAL8 aPlusInfty = - aMinInfty;

     // ========================================================================
     // == [0] ==================  Init proj, indexes, images  =================
     // ========================================================================


     //    [0.0] ---  Compute eventually the selection of point ------
     mVPtsInit = & mGlobPtsInit;  // Default case , take all the point
     std::vector<cPt3dr>  aVPtsSel;  // will contain the selection if required, must be at the same scope
     if (aParam.mSetOK)  // if we have a selection
     {
         for (const auto & aPt : mGlobPtsInit)
             if (aParam.mSetOK->Inside(aPt))
                aVPtsSel.push_back(aPt);
         mVPtsInit  = & aVPtsSel;
     }
     
     //    [0.1] ---  Compute 3D proj+ its 2d-box ----
     aProj.Values(mVPtsProj,*mVPtsInit); 
tREAL8 aDepthBug = mVPtsProj.at(aKBug).z();
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
     if (aBugCpt)
     {
        StdOut() << "INDIma=" <<  mVPtImages.at(aKBug) 
                 << " PIn="   << mVPtsInit->at(aKBug)
                 << " PProj="   << mVPtsProj.at(aKBug)
                 << "\n";
     }

     //    [0.3]  ---------- Alloc images --------------------
     //    [0.3.1]   image of depth
     cPt2di aSzImProf = mBoxInd.CurBox().Sz() + cPt2di(1,1);
     cIm2D<tREAL8> aImDepth(aSzImProf);
     cDataIm2D<tREAL8> & aDImDepth = aImDepth.DIm();
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
         //  tREAL8   aDepth      = mVPtsProj.at(aKPt).z() - mAvgD/1e8;
         // tREAL8 aZ = mVPtsProj.at(aKPt).z();
         // tREAL8   aDepth  = aZ * (1.0 +  1e-6* ((aZ>0) ? -1 : 1));

         tREAL8   aDepth  = mVPtsProj.at(aKPt).z();

         // update depth for all point of the "leaf"
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh : aVDisk)
         {
             cPt2di aPt = aCenter + aNeigh;
             if (aDImDepth.Inside(aPt))
             {
                 aDImDepth.SetMax(aPt,aDepth);
                 if (aBugCpt && (aPt==aPtBug))
                 {
                    StdOut() << "MAJ KPT=" << aKPt << " Diff=" << aDepthBug - aDepth << "\n";
                 }
             } 
         }
     }

     // ===========================================================================================================================
     // == [2] ===   for each point use depth image and if it is visible
     //         * in mode std  accumulate its visibility 
     //         * in mode image, project its radiometry
     // ===========================================================================================================================
int aNbEqualBug=0;
     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt);
         tREAL8   aDepth      = mVPtsProj.at(aKPt).z();
         int aNbVis = 0;
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh :aVDisk) // parse all point of leaf
         {
             cPt2di aPt = aCenter + aNeigh;
             bool IsVisible = (aDImDepth.DefGetV(aPt,aPlusInfty) <= aDepth);
aNbEqualBug += (aDImDepth.DefGetV(aPt,aPlusInfty) == aDepth);
             if (aBugCpt && (aKPt==aKBug) && (aNeigh==aNBug))
             {
                 StdOut() << " Neigh=" << aNeigh << " Vis=" << IsVisible << " Pt=" << aPt << "\n";
                 StdOut() << " [DEPTH] " 
                          << " Im=" << aDImDepth.DefGetV(aPt,aPlusInfty) 
                          << " Pt=" <<  aDepth 
                          << " Diff=" << aDepth-aDImDepth.DefGetV(aPt,aPlusInfty) 
                          << "\n";
             }
             if (IsVisible)  // if the point is visible
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
            tREAL8 aGray = (aW * aNbVis) / aVDisk.size();
            mSumRad.at(aKPt) +=  aGray;
            if (aBugCpt && (aKPt==aKBug))
               StdOut() << "Cpt=" << aCpt << " K=" << aKPt << " NBV=" << aNbVis << " SR="  <<  mSumRad.at(aKPt) << " Gr=" << aGray << "\n";
         }
     }

if (aBugCpt) 
{
   int aNbEqTh = 0;
   for (const auto & aP : aDImDepth)
       if ( aDImDepth.GetV(aP) > (aMinInfty/2))
          aNbEqTh++;
   StdOut() << "NBEQ=" << aNbEqualBug << " " << aNbEqualBug - aNbEqTh << "\n";
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
        cIm2D<tU_INT1>      aIm8BReduc(aSzImFinal);  // radiometric image
        cDataIm2D<tU_INT1>& aDIm8BReduc = aIm8BReduc.DIm();
        cIm2D<tREAL4>       aImDepReduc(aSzImFinal);  // Z/depth  image
        cDataIm2D<tREAL4>&  aDImDepReduc = aImDepReduc.DIm();

        cIm2D<tU_INT1>      aImWeightReduc(aSzImFinal);  // radiometric image
        cDataIm2D<tU_INT1>& aDImWeightReduc = aImWeightReduc.DIm();


        std::unique_ptr<cDiffInterpolator1D> aInterp (cDiffInterpolator1D::TabulSinC(5));

        for (const auto & aPixI : aDIm8BReduc)
        {
            cPt2dr aPixR = ToR(aPixI) * aResolImaRel;
            aDIm8BReduc.SetVTrunc(aPixI,aDImRad.ClipedGetValueInterpol(*aInterp,aPixR,0));
            aDImDepReduc.SetV(aPixI,aDImDepth.ClipedGetValueInterpol(*aInterp,aPixR,0));

            aDImWeightReduc.SetVTrunc(aPixI,round_ni(256*aDImWeight.ClipedGetValueInterpol(*aInterp,aPixR,0)));
        }
        aDIm8BReduc.ToFile("IIP_Radiom_"+ToStr(aCpt) + ".tif");
        aDImDepReduc.ToFile("IIP_Depth_"+ToStr(aCpt) + ".tif");
        aDImWeightReduc.ToFile("IIP_Weight_"+ToStr(aCpt) + ".tif");

        StdOut() << "RESOL ;  IMA-REL=" << aResolImaRel << " Ground=" << aStepImAbs << "\n";
     }
/*
*/
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
        tREAL8  mSurResolSun;
        std::string   mNameImageOut;

        cPt2di        mSzIm;
        tREAL8        mFOV;
        cPt2di        mNbBande;
        cPt2dr        mBSurH;
        
        tREAL8        mFocal;
        cPerspCamIntrCalib * mCalib;

        cPt3dr        mSun;
        std::string   mNameSavePCSun;
};

cAppli_MMVII_CloudImProj::cAppli_MMVII_CloudImProj
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec),
     mSurResolSun      (2.0),
     mSzIm             (3000,2000),
     mFOV              (0.4),
     mNbBande          (5,1),
     mBSurH            (0.1,0.2),
     mFocal            (-1),
     mCalib            (nullptr)
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

   mFocal = Norm2(mSzIm) / mFOV ;
   mCalib = cPerspCamIntrCalib::SimpleCalib("MeshSim",eProjPC::eStenope,mSzIm,cPt3dr(mSzIm.x()/2.0,mSzIm.y()/2.0,mFocal),cPt3di(0,0,0));


   cPointCloud   aPC_In ;
   ReadFromFile(aPC_In,mNameCloudIn);

   if  (IsInit(&mSun))
   {
       cProjPointCloud  aPPC(aPC_In,mSurResolSun,1.0);
       cOrthoProj  aProj(cPt3dr(mSun.x(),mSun.y(),1.0));
       aPPC.ProcessOneProj(cParamProjCloud(&aProj), mSun.z(),false);

       aPPC.ColorizePC();
       if (IsInit(&mNameSavePCSun))
           SaveInFile(aPC_In,mNameSavePCSun);
   }

   cProjPointCloud  aPPC(aPC_In,mSurResolSun,1.0);
   if (false)
   {
   }
   else
   {
       for (int aK=-5 ; aK<=5 ; aK++)
       {
           cOrthoProj aProj(cPt3dr(aK*0.2,0,1.0));
           aPPC.ProcessOneProj(cParamProjCloud(&aProj),0.0,true);
       }
   }

   StdOut() << "NbLeaves "<< aPC_In.LeavesIsInit () << "\n";

   delete mCalib;
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
	bool     mProfIsZIn;
};

cAppli_MMVII_CloudColorate::cAppli_MMVII_CloudColorate
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli    (aVArgs,aSpec),
     mPropRayLeaf    (1.1),
     mSurResol       (2.0),
     mNbSampS        (5),
     mProfIsZIn      (false)
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
          << AOpt2007(mProfIsZIn,"ProfIsZIn","Set prof ZIn/Loc",{eTA2007::HDV,eTA2007::Tuning})
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
   if (mNbSampS>0)
   {
       aPC_In.SetMulDegVis(1e4);
       cSampleSphere3D aSampS(mNbSampS);
       for (int aK=0 ; aK< aSampS.NbSamples() ; aK++)
       {
           cPt3dr aDir = VUnit(aSampS.KthPt(aK));
           if (aDir.z() >= 0.2)
           {
               cOrthoProj aProj(aDir);
	       if (mProfIsZIn) 
                  aProj.SetProfIsZIN();
               aPPC.ProcessOneProj(cParamProjCloud(&aProj),1.0,false);
               aNbStd++;
               // StdOut() << "Still " << aSampS.NbSamples() - aK << "\n";
           }
       }
    }

   if (IsInit(&mSun))
   {
       tREAL8 aW0  = mNbSampS ? aNbStd : 1.0;
       cOrthoProj  aProj(cPt3dr(mSun.x(),mSun.y(),1.0));
       aPPC.ProcessOneProj(cParamProjCloud(&aProj),aW0 * mSun.z(),false);
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
