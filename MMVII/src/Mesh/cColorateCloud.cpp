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

#include "cColorateCloud.h"

namespace MMVII
{
/*
    To do mark :
*/

class cCamOrthoC;
class cOrthoProj;


/* *********************************** */
/*                                     */
/*              cOrthoProj             */
/*                                     */
/* *********************************** */

cOrthoProj::cOrthoProj (const tRotR & aRot ,const cPt3dr & aC,const cPt2dr & aPP ,tREAL8 aResol,bool profIsZ0)  :
    mRL2W         (aRot),
    mC            (aC),
    mPP           (aPP),
    mResol        (aResol),
    mProfIsZ0     (profIsZ0)
{
}

cOrthoProj:: cOrthoProj(const cOrthoProj& anOP) :
    cOrthoProj(anOP.mRL2W,anOP.mC,anOP.mPP,anOP.mResol,anOP.mProfIsZ0)
{
}

cOrthoProj::cOrthoProj (const cPt3dr & aDir ,const cPt3dr & aC,const cPt2dr& aPP ,tREAL8 aResol,bool profIsZ0)  :
   cOrthoProj(tRotR::CompleteRON(aDir,2),aC,aPP,aResol,profIsZ0)
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

       tREAL8 aZ =   mProfIsZ0 ? aPIn.z() : aPLoc.z() ;
       aVOut.push_back(TP3z(aPProj,aZ));
   }

   return aVOut;
}

tSeg3dr   cOrthoProj::BundleInverse(const cPt2dr & aPIm0) const 
{
    cPt2dr aPIm = (aPIm0-mPP) * mResol;
    cPt3dr aP0 = TP3z(aPIm,-1.0);
    cPt3dr aP1 = TP3z( aPIm, 1.0);

    return tSeg3dr(mRL2W.Value(aP0)+mC,mRL2W.Value(aP1)+mC);
}


const  std::vector<cPt3dr> &  cOrthoProj::Inverses(tVecP3 & aVOut,const tVecP3 & aVIn ) const 
{
   aVOut.clear();
   for (size_t aK=0 ; aK<aVIn.size() ; aK++)
   {
       const cPt3dr & aPIn = aVIn.at(aK);
       tREAL8 aZ = aPIn.z();
       if (mProfIsZ0)
       {
          tSeg3dr   aSeg =  BundleInverse(Proj(aPIn));

          //  aSeg.P0 + L aSeg. V12()  -> Z
          tREAL8 aLambda = SafeDiv (aZ - aSeg.P1().z(),   aSeg.V12().z());
          cPt3dr aPt = aSeg.P1() + aSeg.V12() *aLambda;
          aVOut.push_back(aPt);
       }
       else
       {
           cPt2dr aPProj = (Proj(aPIn)-mPP) * mResol;
           cPt3dr aPGlob = mC + mRL2W.Value(TP3z(aPProj,aZ));
           aVOut.push_back(aPGlob);
       }
   }


   return aVOut;
}



/* *********************************** */
/*                                     */
/*              cCamOrthoC             */
/*                                     */
/* *********************************** */

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
   // StdOut() << "BenchCamOrthoBenchCamOrthoBenchCamOrthoBenchCamOrthoBenchCamOrthoBenchCamOrthoBenchCamOrtho\n";
   for (int aK=0 ; aK<20 ; aK++)
   {
       bool ProfIsZ0 = (aK%2)==1;
       cPt3dr aDir = cPt3dr::PRandC();
       if (ProfIsZ0 && (std::abs(aDir.z())< 0.1))
          aDir.z() = 0.1;

       cOrthoProj  anOP1(aDir,cPt3dr::PRand()*10.0,cPt2dr::PRand()*10.0,RandInInterval(0.1,10),ProfIsZ0);

       BenchCamOrtho(anOP1,aDir);
   }
}

/* ********************************************* */
/*                                               */
/*                cProjPointCloud                */
/*                                               */
/* ********************************************* */

cResImagesPPC::cResImagesPPC(const cPt2di & aSz) :
   mImRadiom  (aSz),
   mImWeight  (aSz),
   mImDepth   (aSz)
{
}

/* ********************************************* */
/*                                               */
/*                cProjPointCloud                */
/*                                               */
/* ********************************************* */

///  Class for computing projection of a point cloud


cCamOrthoC * cProjPointCloud::PPC_CamOrtho(int aK,bool  ProfIsZ0,const tRotR & aRot,tREAL8 aMulResol,tREAL8 aMulSz)
{
   cBox3dr   aBox3 = mPC.Box3d();
   cBox2dr   aBox2 = mPC.Box2d();
   tREAL8 aResol = mPC.GroundSampling() * aMulResol;
   cPt2di aSzIm = ToI(aBox2.Sz() * (aMulSz / aResol));
   
   cOrthoProj aProj(aRot,aBox3.Middle(),ToR(aSzIm)/2.0,aResol,ProfIsZ0);
   cCamOrthoC*  aCam = new cCamOrthoC ("C"+ToStr(aK),aProj,aSzIm);

   return aCam;
}

cCamOrthoC * cProjPointCloud::PPC_CamOrtho(int aK,bool  ProfIsZ0,const cPt3dr & aDir,tREAL8 aMulResol,tREAL8 aMulSz)
{
   return PPC_CamOrtho(aK,ProfIsZ0,tRotR::CompleteRON(aDir,2),aMulResol,aMulSz);
}


cProjPointCloud::cProjPointCloud(cPointCloud& aPC,tREAL8 aWeightInit) :
   mPC        (aPC),
   mNbPtsGlob (aPC.NbPts()),
   mComputeProfMax (true),
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
   mDImWeigth (nullptr),
   mImIndex   (cPt2di(1,1)),
   mDImIndex  (nullptr)
   
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

void cProjPointCloud::SetComputeProfMax(bool aComputeProfMax)
{
   mComputeProfMax = aComputeProfMax;
}

void cProjPointCloud::ColorizePC()
{
   // Put mSumRad as an attribute of PC for memorization
   for (size_t aK=0 ; aK<mVPtsProj.size() ; aK++)
   {
       mPC.SetDegVis(aK,mSumRad.at(aK)  / mSumW);
   }
}

void cProjPointCloud::ProcessOneProj
     (
             tREAL8 aSurResol,
             const cSensorImage & aSensor,
             tREAL8 aWeight,
             bool isModeImage,
             const std::string & aMsg,
             bool  ShowMsg,
             bool  ExportIm
     )
{

   // std::vector<cPt2di> aVPixBugImage{{310,50},{310,150}};
   // std::vector<cPt2di> aVPixBugInit{{825,971},{826,971},{826,915},{826,916}};
    //cPt2di aPixBug(166,128);  // Blanc, devrait etre noir


     mSumW += aWeight;               // accumlate weight
     tREAL8 aMinInfty = -1e10;  // minus infinity, any value lower than anr real one
     tREAL8 aPlusInfty = - aMinInfty;

     // ========================================================================
     // == [0] ==================  Init proj, indexes, images  =================
     // ========================================================================


     //    [0.0] ---  Compute eventually the selection of point ------
     mVPtsInit = & mGlobPtsInit;  // Default case , take all the point
     std::vector<cPt3dr>  aVPtsSel;  // will contain the selection if required, must be at the same scope

     // this index of selected is required as in mode image the index of mGlobPtsInit in point cloud are different of mVPtsInit
     std::vector<int> aVIndeSel;
     if (isModeImage)  
     {
         // In mode image we select only the  point visible in the camera
         for (size_t aKPt=0 ; aKPt<mGlobPtsInit.size() ; aKPt++)
         {
             const auto & aPt = mGlobPtsInit.at(aKPt);
             if (aSensor.DegreeVisibility(aPt)>0)
             {
                aVPtsSel.push_back(aPt);
                aVIndeSel.push_back(aKPt);
             }
         }
         mVPtsInit  = & aVPtsSel;
     }
     else
     {
            for (size_t aKPt=0 ; aKPt<mGlobPtsInit.size() ; aKPt++)
                aVIndeSel.push_back(aKPt);
     }

   //  StdOut() << " Ratio Sel " << aVPtsSel.size() / double(mGlobPtsInit.size()) << "\n";
     
     //    [0.1] ---  Compute 3D proj+ its 2d-box ----
     mVPtsProj.clear();
     for (const auto & aPt : *mVPtsInit)
     {
         /*
         if (mVPtsProj.empty())
         {
             StdOut() << " xxxxxxxPt=" << aPt
                      << " G2ID=" << aSensor.Ground2ImageAndDepth(aPt)
                      << "\n";
         }*/

          mVPtsProj.push_back(aSensor.Ground2ImageAndDepth(aPt));

     }

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

     mSzIm = ( isModeImage ?    mBoxInd.CurBox().P1() : mBoxInd.CurBox().Sz())   + cPt2di(1,1);


     if (false && isModeImage)
     {
        auto aBox = cBox3dr::FromVect(mVPtsProj);
        StdOut() <<  " Box3D= " << aBox 
                 <<  " BoxInd= "  << mBoxInd.CurBox()
                 << " SzSens=" << aSensor.Sz() 
                 << " SzIm=" << mSzIm
                 << " SR="  << aSurResol 
                 << "\n";
     }


     //    [0.3]  ---------- Alloc images --------------------
     //    [0.3.1]   image of depth

     mDImIndex  = & (mImIndex.DIm());
     mDImIndex->Resize(mSzIm);
   //   StdOut() << "xxxSZ_IM=" << mSzIm << "\n";
     mDImIndex->InitCste(NoIndex);


     mDImDepth = & (mImDepth.DIm());
     mDImDepth->Resize(mSzIm);
     mDImDepth->InitCste(aMinInfty);


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

     //bool
     int aNbPtsCover = 0;
     cWeightAv<tREAL8,tREAL8> aAvgNb;
     FakeUseIt(aNbPtsCover);
     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt); // extract index
         tImageDepth   aDepth  = mVPtsProj.at(aKPt).z();

         // update depth for all point of the "leaf"
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         aAvgNb.Add(1.0,aVDisk.size());
         for (const auto & aNeigh : aVDisk)
         {
             cPt2di aPt = aCenter + aNeigh;
             if ( mDImIndex->Inside(aPt))
             {
                 int aIndex = mDImIndex->GetV(aPt);
                 if ((aIndex==NoIndex) || ((aDepth>mVPtsProj.at(aIndex).z())==mComputeProfMax) )
                 {
                    mDImIndex->SetV(aPt,aKPt);
                    aNbPtsCover++;
                 }

                /* for ( auto & aPixBug : aVPixBugImage)
                 {
                     if (aPt==aPixBug)
                     {
                         StdOut() << " Pix=" << aPt
                                  << " Kpt=" << aKPt
                                  << " Index=" << aIndex
                                  << " Proj=" << mVPtsProj.at(aKPt)
                                  << " PInit=" << mVPtsInit->at(aKPt)

                                  << "\n";
                       //  aPixBug = cPt2di(-100,-100);
                     }
                 }*/
             }
         }
     }

     //StdOut() << " AvgNb=" << aAvgNb.Average() << "\n";
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
 
     cWeightAv<tREAL8,tREAL8>  aLumPt;
     cWeightAv<tREAL8,tREAL8>  aLumVis;
     int aNbVisTot = 0;


     tImageDepth aVMinInit =  aPlusInfty;
     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++) // parse all points
     {
         const cPt2di  & aCenter = mVPtImages.at(aKPt);
         tImageDepth   aDepth      = mVPtsProj.at(aKPt).z();
         UpdateMin(aVMinInit,aDepth);
         int aNbVis = 0;
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         tREAL8 aDegVis = mPC.GetDegVis(aVIndeSel.at(aKPt));

         aLumPt.Add(1.0,aDegVis);

         for (const auto & aNeigh :aVDisk) // parse all point of leaf
         {
             cPt2di aPt = aCenter + aNeigh;
             bool IsVisible = (mDImIndex->DefGetV(aPt,NoIndex) == (int)aKPt);

             aNbVisTot += IsVisible;
             if (! isModeImage)
                 aLumVis.Add(1.0, IsVisible);

             if (IsVisible)  // if the point is visible
             {
                 if (isModeImage)
                    aLumVis.Add(1.0, aDegVis);
// aNbModif++;
                if (isModeImage)  // in mode image udpate radiometry & image
                {
                   mDImWeigth->SetV(aPt,1.0);
                   mDImRad->SetV(aPt,aDegVis*255);
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
            mSumRad.at(aKPt) +=  aGray * aWeight;
         }
     }

     if (ShowMsg)
     {
         StdOut() << " MSG=["  << aMsg << "]"
             << "LumStd=" << aLumPt.Average() 
             << " LumVis=" << aLumVis.Average() 
             << " NbVis/Im=" << tREAL8(aNbVisTot) / (mSzIm.x() * mSzIm.y())
             << "\n";
    }

    // Now put z in image depth

    for (const auto & aPix : *mDImIndex)
    {
        int aIndex =  mDImIndex->GetV(aPix);
        tImageDepth aDepth= (aIndex==NoIndex) ?  (aVMinInit - 100.0) : mVPtsInit->at(aIndex).z();

        mDImDepth->SetV(aPix,aDepth);
    }

    if (ExportIm)
    {
       std::string aPrefix = (isModeImage ? "IIIP-" : "Colorate-") + aMsg;
       if (mDImRad)
          mDImRad->ToFile(aPrefix+"-RAD.tif");
       if (mDImWeigth)
          mDImWeigth->ToFile(aPrefix+"-WEIGHT.tif");
       if (mDImDepth)
       {
          mDImDepth->ToFile(aPrefix+"-DEPTH.tif");
       }
    }
}

cResImagesPPC cProjPointCloud::ProcessImage(tREAL8 aSurResol,const cSensorImage & aSensor)
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

     // mDImRad->ToFile("IIII-RAD0-FILTR.tif");
       
     static int aCpt=0; aCpt++;
         
     cPt2di  aSzImFinal = aSensor.Sz();
     cResImagesPPC aRes(aSzImFinal);

    // cIm2D<tU_INT1>      aIm8BReduc(aSzImFinal);  // radiometric image
     cDataIm2D<tU_INT1>& aDIm8BReduc =  aRes.mImRadiom.DIm();
    //  cIm2D<tREAL4>       aImDepReduc(aSzImFinal);  // Z/depth  image
     cDataIm2D<tREAL4>&  aDImDepReduc = aRes.mImDepth.DIm();

     // cIm2D<tU_INT1>      aImWeightReduc(aSzImFinal);  // radiometric image
     cDataIm2D<tU_INT1>& aDImWeightReduc = aRes.mImWeight.DIm();


     std::unique_ptr<cDiffInterpolator1D> aInterp (cDiffInterpolator1D::TabulSinC(5));

     for (const auto & aPixI : aDIm8BReduc)
     {
         cPt2dr aPixR = ToR(aPixI) * aSurResol;
         bool Ok;

         aDIm8BReduc.SetVTrunc(aPixI,mDImRad->ClipedGetValueInterpol(*aInterp,aPixR,0,&Ok));
         aDImDepReduc.SetV(aPixI,mDImDepth->ClipedGetValueInterpol(*aInterp,aPixR,0,&Ok));
         aDImWeightReduc.SetVTrunc(aPixI,round_ni(256*mDImWeigth->ClipedGetValueInterpol(*aInterp,aPixR,0,&Ok)));
     }

     return aRes;
     /*
     aDIm8BReduc.ToFile    (aPrefix+"_Radiom_"+ToStr(aCpt) + ".tif");
     aDImDepReduc.ToFile   (aPrefix+"_Depth_"+ToStr(aCpt) + ".tif");
     aDImWeightReduc.ToFile(aPrefix+"_Weight_"+ToStr(aCpt) + ".tif");
     */
}
/*
cIm2D<tImageDepth>      ImDepth()  const;
cIm2D<tREAL4>           ImRadiom() const;
cIm2D<tREAL4>           ImWeight() const;
*/

};
