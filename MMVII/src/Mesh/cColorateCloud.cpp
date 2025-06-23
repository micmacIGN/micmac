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

class cCamOrthoC;

class cOrthoProj  :  public  tIMap_R3
{
    public :
       friend cCamOrthoC;
       typedef std::vector<cPt3dr> tVecP3;

       cOrthoProj (const tRotR & aRot ,const cPt3dr& aC,const cPt2dr& aPP ,tREAL8 aResol) ;
       cOrthoProj (const cPt3dr & aDir,const cPt3dr& aC =cPt3dr(0,0,0),const cPt2dr& aPP= cPt2dr(0,0) ,tREAL8 aResol=1.0) ;
       tSeg3dr  BundleInverse(const cPt2dr &) const ;

       cOrthoProj(const cOrthoProj&);

        
    private  :
       const  tVecP3 &  Values   (tVecP3 &,const tVecP3 & ) const override;
       const  tVecP3 &  Inverses (tVecP3 &,const tVecP3 & ) const override;

       tRotR  mRL2W;
       cPt3dr mC;
       cPt2dr mPP;
       tREAL8 mResol;
};

cOrthoProj::cOrthoProj (const tRotR & aRot ,const cPt3dr & aC,const cPt2dr & aPP ,tREAL8 aResol)  :
    mRL2W         (aRot),
    mC            (aC),
    mPP           (aPP),
    mResol        (aResol)
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


const  std::vector<cPt3dr> &  cOrthoProj::Values(tVecP3 & aVOut,const tVecP3 & aVIn ) const 
{
   aVOut.clear();
   for (size_t aK=0 ; aK<aVIn.size() ; aK++)
   {
       const cPt3dr & aPIn = aVIn.at(aK);
       cPt3dr  aPLoc = mRL2W.Inverse(aPIn-mC);
       cPt2dr  aPProj = Proj(aPLoc);
       aPProj = mPP+ aPProj/mResol;

       tREAL8 aZ =  aPLoc.z() ;
       aVOut.push_back(TP3z(aPProj,aZ));
   }

   return aVOut;
}

const  std::vector<cPt3dr> &  cOrthoProj::Inverses(tVecP3 & aVOut,const tVecP3 & aVIn ) const 
{
   aVOut.clear();
   for (size_t aK=0 ; aK<aVIn.size() ; aK++)
   {
       const cPt3dr & aPIn = aVIn.at(aK);
       tREAL8 aZ = aPIn.z();
       cPt2dr aPProj = (Proj(aPIn)-mPP) * mResol;
       cPt3dr aPGlob = mC + mRL2W.Value(TP3z(aPProj,aZ));
       aVOut.push_back(aPGlob);

       /*

       cPt3dr  aPLoc = mRL2W.Inverse(aPIn-mC);
       cPt2dr  aPProj = Proj(aPLoc);
       aPProj = mPP+ aPProj/mResol;

       // aVOut.push_back(TP3z(aPProj,aPIn.z()*0.01));
       aVOut.push_back(TP3z(aPProj,aZ));
       */

       // aVOut.push_back(TP3z(aPProj,aPLoc.z()));
   }


   return aVOut;
}

tSeg3dr   cOrthoProj::BundleInverse(const cPt2dr & aPIm0) const 
{
    cPt2dr aPIm = (aPIm0-mPP) * mResol;
    cPt3dr aP0 = TP3z(aPIm,-1.0)+mC;
    cPt3dr aP1 = TP3z(aPIm, 1.0)+mC;

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

       cPt2dr Ground2Image(const cPt3dr &) const override;
       const cPixelDomain & PixelDomain() const override;
       tSeg3dr  Image2Bundle(const cPt2dr &) const override;
       std::string  V_PrefixName() const   override;
       cPt3dr  PseudoCenterOfProj() const override;
       double DegreeVisibility(const cPt3dr &) const override;

       bool  HasImageAndDepth() const override;
       cPt3dr Ground2ImageAndDepth(const cPt3dr &) const override;
       cPt3dr ImageAndDepth2Ground(const cPt3dr &) const override;

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

cPt2dr cCamOrthoC::Ground2Image(const cPt3dr & aPGround) const
{
	return Proj(mProj.Value(aPGround));
}
cPt3dr cCamOrthoC::Ground2ImageAndDepth(const cPt3dr & aPGround) const 
{
    return mProj.Value(aPGround);
}

cPt3dr cCamOrthoC::ImageAndDepth2Ground(const cPt3dr & aPImAndD) const 
{
    return mProj.Inverse(aPImAndD);
}


tSeg3dr  cCamOrthoC::Image2Bundle(const cPt2dr & aPIm) const
{
	return mProj.BundleInverse(aPIm);
}

std::string  cCamOrthoC::V_PrefixName() const { return "CamOrthoC"; }

cPt3dr  cCamOrthoC::PseudoCenterOfProj() const 
{
    cSegmentCompiled<tREAL8,3>  aSeg = Image2Bundle(ToR(Sz())/2.0);
    return aSeg.Proj(mProj.mC);
}

double cCamOrthoC::DegreeVisibility(const cPt3dr & aPGround) const 
{
    cPt2dr aPIm = Proj(mProj.Value(aPGround));

    return   mPixelDomain.DegreeVisibility(aPIm)>0;
}

bool  cCamOrthoC::HasImageAndDepth() const { return true; }

void BenchCamOrtho(const cOrthoProj &anOP,const cPt3dr & aDir)
{
     for (int aKPt=0 ; aKPt<20 ; aKPt++)
     {
          cPt3dr aPt1 = cPt3dr::PRandC() * 100.0;
          cPt3dr aPt2 =  anOP.Value( anOP.Inverse(aPt1));

	  //  StdOut() << "NNNN " << Norm2(aPt1-aPt2) << "\n";
          MMVII_INTERNAL_ASSERT_bench(Norm2(aPt1-aPt2)<1e-5,"BenchCamOrtho Value/Inverse");

     }
}


void BenchCamOrtho()
{
   for (int aK=0 ; aK<20 ; aK++)
   {
       cPt3dr aDir = cPt3dr::PRandC();
       cOrthoProj  anOP1(aDir,cPt3dr::PRand()*10.0);

       BenchCamOrtho(anOP1,aDir);
   }
}


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
           mCam    (nullptr)
       {
       }
       const tIMap_R3 * mProj;
       cSensorImage *   mCam;
};

///  Class for computing projection of a point cloud

class cProjPointCloud
{
     public :
         /// constructor : memoriez PC, inialize accum, allocate mem
         cProjPointCloud(cPointCloud & aParam,tREAL8 aWeightInit );

	 /// Process on projection for  OR  (1) modify colorization of points (2) 
         void ProcessOneProj(tREAL8 aSurResol,const cSensorImage &,tREAL8 aW,bool ModeImage);
         
         void ProcessImage(tREAL8 aSurResol,const cSensorImage &,const std::string & aPost);

	 // export the average of radiomeries (in mSumRad) as a field of mPC
         void ColorizePC(); 
	 cCamOrthoC * CamOrtho(const cPt3dr & aDir,const cPt2di & aSzIm=cPt2di(0,0));


     private :
	 typedef tREAL8 tImageDepth;
	 // --------- Processed at initialization ----------------
         cPointCloud&           mPC;       ///< memorize cloud point
	 const int              mNbPtsGlob;    ///< store number of points
	 // int                    mNbPts;    ///<  Dynamic, change with SetOk
         std::vector<cPt3dr>    mGlobPtsInit; ///< initial point cloud (stores once  for all in 64-byte, for efficienciency)
         std::vector<cPt3dr> *  mVPtsInit;     /// Dynamic, change with SetOk
         // const tREAL8           mSurResol;
	 const tREAL8           mAvgD;       ///< Avg 2D-Distance between points in 3D Cloud
         //const tREAL8           mStepProf;  ///< Step for computing depth-images
	 // --------- Updated  with  "ProcessOneProj"  ----------------
         tREAL8                 mSumW;      ///< accumulate sum of weight on radiometries
         std::vector<tREAL4>    mSumRad;    ///< accumulate sum of radiometry
	 // --------- Computed at each run of "ProcessOneProj"  ------------------
         std::vector<cPt3dr>    mVPtsProj;  ///< memorize projections 
         std::vector<cPt2di>    mVPtImages; ///< Projection in image of given 3D Point
         cTplBoxOfPts<int,2>    mBoxInd;    ///< Compute for of mVPtImages
					    
	 cPt2di                  mSzIm;
	 cIm2D<tImageDepth>      mImDepth;
         cDataIm2D<tImageDepth>* mDImDepth;
         cIm2D<tREAL4>            mImRad;
         cDataIm2D<tREAL4>*       mDImRad;
         cIm2D<tREAL4>            mImWeigth;
         cDataIm2D<tREAL4>*       mDImWeigth;
};

cCamOrthoC * cProjPointCloud::CamOrtho(const cPt3dr & aDir,const cPt2di & aSzIm)
{
   cOrthoProj aProj(aDir,mPC.Centroid(),ToR(aSzIm)/2.0,mPC.GroundSampling());
   cCamOrthoC*  aCam = new cCamOrthoC ("ColMesh",aProj,PtSupEq(aSzIm,cPt2di(1,1)));

   return aCam;
}


cProjPointCloud::cProjPointCloud(cPointCloud& aPC,tREAL8 aWeightInit) :
   mPC        (aPC),
   mNbPtsGlob (aPC.NbPts()),
   // mSurResol  (aSurResol),
   mAvgD      (std::sqrt(1.0/mPC.CurStdDensity())),
   //mStepProf  (mAvgD / mSurResol),
   mSumW      (aWeightInit),
   mSumRad    (mNbPtsGlob,0.0),
   mImDepth   (cPt2di(1,1)),
   mDImDepth  (nullptr),
   mImRad     (cPt2di(1,1)),
   mDImRad    (nullptr),
   mImWeigth  (cPt2di(1,1)),
   mDImWeigth (nullptr)
{
//  StdOut() << "SSSSS  " << mStepProf << " AAA=" <<mAvgD << " SRrr=" << mSurResol << "\n";
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

void cProjPointCloud::ProcessOneProj(tREAL8 aSurResol,const cSensorImage & aSensor,tREAL8 aW,bool isModeImage)
{

     mSumW += aW;               // accumlate weight
     tREAL8 aMinInfty = -1e10;  // minus infinity, any value lower than anr real one
     tREAL8 aPlusInfty = - aMinInfty;
FakeUseIt(aPlusInfty);

     // ========================================================================
     // == [0] ==================  Init proj, indexes, images  =================
     // ========================================================================


     //    [0.0] ---  Compute eventually the selection of point ------
     mVPtsInit = & mGlobPtsInit;  // Default case , take all the point
     std::vector<cPt3dr>  aVPtsSel;  // will contain the selection if required, must be at the same scope
     if (isModeImage)  
     {
         // In mode image we select only the  point visible in the camera
         for (const auto & aPt : mGlobPtsInit)
	 {
             if (aSensor.DegreeVisibility(aPt)>0)
	     {
                aVPtsSel.push_back(aPt);
	     }
         }
         StdOut()  << "SELLL=" << mVPtsInit->size() << " " << aVPtsSel.size() << "\n";
         mVPtsInit  = & aVPtsSel;
	 // aCenter = Centroid(aVPtsSel);
     }
     
     //    [0.1] ---  Compute 3D proj+ its 2d-box ----
     mVPtsProj.clear();
     for (const auto & aPt : *mVPtsInit)
          mVPtsProj.push_back(aSensor.Ground2ImageAndDepth(aPt));

     cPt2dr aPMin(0.0,0.0);
     if (! isModeImage)
     {
        for (const auto & aPt : mVPtsProj)
        {
            SetInfEq(aPMin,Proj(aPt));
        }
     }


     //    [0.2]  ---------- compute the images indexes of points + its box  & sz ---
     mBoxInd= cTplBoxOfPts<int,2> (); //
     mVPtImages.clear();

     for (const auto & aPt : mVPtsProj)
     {
         cPt2di anInd = ToI(  (Proj(aPt)-aPMin)*aSurResol   );  // compute image index
         mBoxInd.Add(anInd); // memo in box
         mVPtImages.push_back(anInd); 
     }

     /*
     if (0)
     {
          cWeightAv<tREAL8,cPt3dr> aWPts0;
          cWeightAv<tREAL8,cPt3dr> aWProj;
          for (size_t aKPt=0 ; aKPt<mVPtsInit->size() ; aKPt++)
	  {
              cPt3dr aPt =  mVPtsInit->at(aKPt);
              cPt3dr aPProj =  aProj.Value(aPt);

              aWPts0.Add(1.0,aPt);
              aWProj.Add(1.0,aPProj);
	  }
	  StdOut()  << "AVG ;;  P0=" << aWPts0.Average()  << " PROJ=" << aWProj.Average() << " Step=" << mStepProf << "\n";
     }
     */

     //    [0.3]  ---------- Alloc images --------------------
     //    [0.3.1]   image of depth
     mSzIm = mBoxInd.CurBox().Sz() + cPt2di(1,1);



     mDImDepth = & (mImDepth.DIm());
     mDImDepth->Resize(mSzIm);
     mDImDepth->InitCste(aMinInfty);

     /*
	 cPt2di                  mSzIm;
	 cIm2D<tImageDepth>      mImDepth;
         cDataIm2D<tImageDepth>* mDImDepth;
	 */

     //    [0.3.2]   image of radiometry
     /*
     cIm2D<tREAL4> aImRad(mSzIm);
     cDataIm2D<tREAL4>& aDImRad = aImRad.DIm();
     */
     if (isModeImage) 
     {
         mDImRad = & (mImRad.DIm());
         mDImRad->Resize(mSzIm);
         mDImRad->InitCste(0.0);

         mDImWeigth = & (mImWeigth.DIm());
         mDImWeigth->Resize(mSzIm);
         mDImWeigth->InitCste(0.0);
     }



     //    [0.4]  ---------- Alloc vector SzLeaf -> neighboor in image coordinate (time efficiency) ----------------
     std::vector<std::vector<cPt2di>> aVVdisk(256);  // as size if store 8-byte, its sufficient
     {
         cPt3dr aCenter = mPC.Centroid();
         tREAL8 aGS = aSensor.Gen_GroundSamplingDistance(aCenter);
         for (int aK=0 ; aK<=255 ; aK++)
         {
             tREAL8 aSzL = mPC.ConvertInt2SzLeave(aK) / aGS;
             aVVdisk.at(aK) = VectOfRadius(-1,aSurResol*aSzL);
         }
      }


     // ==================================================================================================================
     // == [1] ==================   compute the depth image : accumulate for each pixel the maximal depth ================
     // ==================================================================================================================

     int aNbPtsCover = 0;
     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt); // extract index
         tImageDepth   aDepth  = mVPtsProj.at(aKPt).z();

         // update depth for all point of the "leaf"
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh : aVDisk)
         {
             cPt2di aPt = aCenter + aNeigh;
             if (mDImDepth->Inside(aPt))
             {
                 mDImDepth->SetMax(aPt,aDepth);
                 aNbPtsCover++;
             } 
         }
     }
/*
     StdOut() << "SZIII = " << mSzIm  
              << " PropPtIn=" <<  mVPtsProj.size() / (tREAL8) (mSzIm.x() * mSzIm.y()) 
              << " PropPtCov=" <<  aNbPtsCover / (tREAL8) (mSzIm.x() * mSzIm.y()) 
              << "\n";
      getchar();
*/


     // ===========================================================================================================================
     // == [2] ===   for each point use depth image and if it is visible
     //         * in mode std  accumulate its visibility 
     //         * in mode image, project its radiometry
     // ===========================================================================================================================
 
//int aNbModif=0;
     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt);
         tImageDepth   aDepth      = mVPtsProj.at(aKPt).z();
         int aNbVis = 0;
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh :aVDisk) // parse all point of leaf
         {
             cPt2di aPt = aCenter + aNeigh;
             bool IsVisible = (mDImDepth->DefGetV(aPt,aPlusInfty) <= aDepth);

             if (IsVisible)  // if the point is visible
             {
// aNbModif++;
                if (isModeImage)  // in mode image udpate radiometry & image
                {
                   mDImWeigth->SetV(aPt,1.0);
                   mDImRad->SetV(aPt,mPC.GetDegVis(aKPt)*255);
                }
                else  // in mode standard uptdate visib count
                {
                   aNbVis++;
                }
             } 
         }
         if (!isModeImage)  // in mode std we know the visibility 
         {
            tREAL8 aGray =  aNbVis / tREAL8(aVDisk.size());
            mSumRad.at(aKPt) +=  aGray * aW;
         }
     }
// StdOut() << "NBMMM=" << aNbModif << "\n";
}

void cProjPointCloud::ProcessImage(tREAL8 aSurResol,const cSensorImage & aSensor,const std::string & aPrefix)
{
     // =====================================================================================
     // == [3] ==================   compute the images (radiom, weight, depth) ==============
     // =====================================================================================

     tREAL8 aSigmaImaFinal = 1.0;
     tREAL8 aSigmaImaInit = aSigmaImaFinal * aSurResol;
     int    aNbIter = 5;

     //  DImDepth has def value to -Infty, we need to set to 0 the pixel non initialized    
     //  not needed for mDImRad & mDImDepth
     MulImageInPlace(*mDImDepth,*mDImWeigth);

     //  make some gaussian averaging for Rad/Depth/Weigth
     ExpFilterOfStdDev( *mDImRad,aNbIter,aSigmaImaInit);
     ExpFilterOfStdDev(*mDImWeigth,aNbIter,aSigmaImaInit);
     ExpFilterOfStdDev( *mDImDepth,aNbIter,aSigmaImaInit);

     //  make Depth /= Weith    Rad /= Weitgh
     for (const auto & aPix : *mDImWeigth)
     {
         tREAL8 aW =   mDImWeigth->GetV(aPix);
         tREAL8 aD =   mDImDepth->GetV(aPix);
         tREAL8 aR =   mDImRad->GetV(aPix);
         mDImRad->SetV(aPix,aW ?  aR/aW : 0.0);
         mDImDepth->SetV(aPix,aW ?  aD/aW : 0.0);
     }
       
     static int aCpt=0; aCpt++;
         
     cPt2di  aSzImFinal = aSensor.Sz();
     cIm2D<tU_INT1>      aIm8BReduc(aSzImFinal);  // radiometric image
     cDataIm2D<tU_INT1>& aDIm8BReduc = aIm8BReduc.DIm();
     cIm2D<tREAL4>       aImDepReduc(aSzImFinal);  // Z/depth  image
     cDataIm2D<tREAL4>&  aDImDepReduc = aImDepReduc.DIm();

     cIm2D<tU_INT1>      aImWeightReduc(aSzImFinal);  // radiometric image
     cDataIm2D<tU_INT1>& aDImWeightReduc = aImWeightReduc.DIm();


     std::unique_ptr<cDiffInterpolator1D> aInterp (cDiffInterpolator1D::TabulSinC(5));

     for (const auto & aPixI : aDIm8BReduc)
     {
         cPt2dr aPixR = ToR(aPixI) * aSurResol;
         aDIm8BReduc.SetVTrunc(aPixI,mDImRad->ClipedGetValueInterpol(*aInterp,aPixR,0));
         aDImDepReduc.SetV(aPixI,mDImDepth->ClipedGetValueInterpol(*aInterp,aPixR,0));

         aDImWeightReduc.SetVTrunc(aPixI,round_ni(256*mDImWeigth->ClipedGetValueInterpol(*aInterp,aPixR,0)));
     }
     aDIm8BReduc.ToFile    (aPrefix+"_Radiom_"+ToStr(aCpt) + ".tif");
     aDImDepReduc.ToFile   (aPrefix+"_Depth_"+ToStr(aCpt) + ".tif");
     aDImWeightReduc.ToFile(aPrefix+"_Weight_"+ToStr(aCpt) + ".tif");
}


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

        cPt2dr   mPropRayLeaf;
        tREAL8   mSurResol;
        int      mNbSampS;
        cPt3dr   mSun;
};

cAppli_MMVII_CloudColorate::cAppli_MMVII_CloudColorate
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli    (aVArgs,aSpec),
     mPropRayLeaf    (1.8,2.2),
     mSurResol       (2.0),
     mNbSampS        (5)
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
       for (size_t aKPt=0 ; aKPt<aPC_In.NbPts() ; aKPt++)
       {
           tREAL8  aRayLeaf  = RandInInterval(mPropRayLeaf)  * aPC_In.GroundSampling();
           aPC_In.SetSzLeaves(aKPt,aRayLeaf);
       }
   }

   cAutoTimerSegm aTSInit(TimeSegm(),"Init");
   tREAL8 aWeightInit = (mNbSampS==0);
   cProjPointCloud  aPPC(aPC_In,aWeightInit);  // Weight Init 0 if NbS ,  
// cProjPointCloud::cProjPointCloud(cPointCloud& aPC,tREAL8 aWeightInit) :

    
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
               std::unique_ptr<cCamOrthoC> aCam (aPPC.CamOrtho(aDir));
               aPPC.ProcessOneProj(mSurResol,*aCam,1.0,false);
               aNbStd++;
               StdOut() << "Still " << aSampS.NbSamples() - aK << "\n";
           }
       }
    }

   if (IsInit(&mSun))
   {
       tREAL8 aW0  = mNbSampS ? aNbStd : 1.0;
       std::unique_ptr<cCamOrthoC> aCam (aPPC.CamOrtho(cPt3dr(mSun.x(),mSun.y(),1.0)));
       aPPC.ProcessOneProj(mSurResol,*aCam,aW0 * mSun.z(),false);
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

	tREAL8        mResolOrthoC;
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
     mResolOrthoC      (0.2),
     mSzIm             (5000,5000),
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
          << AOpt2007(mSzIm,"SzIm","Size of resulting image",{eTA2007::HDV})
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

   cProjPointCloud  aPPC(aPC_In,1.0);

   if  (IsInit(&mSun))
   {
       std::unique_ptr<cCamOrthoC> aCam (aPPC.CamOrtho(cPt3dr(mSun.x(),mSun.y(),1.0)));
       aPPC.ProcessOneProj(mSurResolSun,*aCam,mSun.z(),false);

       aPPC.ColorizePC();

       if (IsInit(&mNameSavePCSun))
           SaveInFile(aPC_In,mNameSavePCSun);
   }
#if (0)

   if (false)
   {
   }
   else
   {
       for (int aK=-0 ; aK<=0 ; aK++)
       {
            cOrthoProj aProj(cPt3dr(aK*0.2,0,1.0),aPC_In.Centroid(),ToR(mSzIm)/2.0);
// cOrthoProj  aProj(cPt3dr(aK*0.2,0,1.0),aPC_In.Centroid());
	   cCamOrthoC aCam("ORTHO-"+ToStr(aK),aProj,mSzIm);

	   cParamProjCloud aParam(&aProj);
	   aParam.mCam = & aCam;

           aPPC.ProcessOneProj(aParam,0.0,true);
	   aPPC.ProcessImage("IIP");
       }
   }

   StdOut() << "NbLeaves "<< aPC_In.LeavesIsInit () << "\n";
#endif

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

};
