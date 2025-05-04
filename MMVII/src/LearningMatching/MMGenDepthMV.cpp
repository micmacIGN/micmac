#include <StdAfx.h>
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include <fstream>
#include <iostream>
#include "MMVII_ZBuffer.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include "MMVII_2Include_Tiling.h"
#include "LearnDM.h"


//static int NODATA=-9999;

namespace  MMVII {

namespace  cNS_MMGenDepthMV
{

  class cWorldCoordinates
  {

  public:
    cWorldCoordinates(std::string tfw_file)
    {
      ifstream input_tfw(tfw_file);
      string aline;
      std::vector< string > acontent;
      while(std::getline(input_tfw,aline))
        {
          acontent.push_back(aline);
        }
      input_tfw.close();

      gsd_x=stof(acontent.at(0));
      gsd_y=stof(acontent.at(3));
      x_ul=stof(acontent.at(4));
      y_ul=stof(acontent.at(5));

      acontent.clear();
    };

    void to_world_coordinates(const cPt2dr & aPx, cPt2dr & aWPx);
    void to_pixel_coordinates(cPt2dr & aWPx, cPt2dr & aPx);

    tREAL4 gsd_x=0;
    tREAL4 gsd_y=0;
    tREAL4 x_ul=0;
    tREAL4 y_ul=0;
  };


 void cWorldCoordinates::to_world_coordinates(const cPt2dr & aPx, cPt2dr & aWPx)
 {
  aWPx.x()=x_ul+aPx.x()*gsd_x;
  aWPx.y()=y_ul+aPx.y()*gsd_y;
 }

 void cWorldCoordinates::to_pixel_coordinates(cPt2dr & aWPx, cPt2dr & aPx)
 {
  aPx.x()=(aWPx.x()-x_ul)/gsd_x;
  aPx.y()=(aWPx.y()-y_ul)/gsd_y;
 }


 bool compdepth(cPt3dr a, cPt3dr b) {
     return (a.z() < b.z()) && (a.x() != b.x()) && (a.y()!=b.y());
 }

  class cAppliMMGenDepthMV;

  class cAppliMMGenDepthMV : public cMMVII_Appli,
                             public cAppliParseBoxIm<tREAL4>
  {

   public :
    typedef tU_INT1               tElemMasq;
    typedef tU_INT2               tElemImage;
    typedef tREAL4                tElemDepth;
    typedef cIm2D<tElemMasq>      tImMasq;
    typedef cIm2D<tElemImage>     tImImage;
    typedef cIm2D<tElemDepth>     tImDepth;
    typedef cDataIm2D<tElemMasq>  tDImMasq;
    typedef cDataIm2D<tElemImage> tDImImage;
    typedef cDataIm2D<tElemDepth> tDImDepth;
    typedef std::vector<tImImage>  tVecIms;
    typedef std::vector<tImMasq>   tVecMasq;
    typedef tREAL8  tCoordDensify;
    typedef cTriangle2DCompiled<tCoordDensify>  tTriangle2DCompiled;

    cAppliMMGenDepthMV(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
    int Exe() override;
    int ExeOnParsedBox() override;
    bool MakeDecision(std::vector<cPt3dr> & aVecPoints);
    void Generate_sparse_depth(std::string aNameImage,
                                std::vector<std::string> aVecLidar,
                                size_t aSzMin,
                                tREAL8 aThresholdVisbility);
    void MakeOneTri(const  tTriangle2DCompiled & aTri);

  // --- constructed ---
   cPhotogrammetricProject   mPhProj;
   cTriangulation3D<tREAL8>* mTri3D;
   cTriangulation3D<tREAL8>* mTri3DReproj;
   std::string mSpecImIn;
   std::string mPatternLidar;
   std::string mNameOutDepth="Depth_";
   std::string mNameOutMasq ="Masq_";
   size_t                    mNbF;
   size_t                    mNbP;
   cSensorCamPC *            mCamPC;
   tImDepth mDepthImage;
   tImMasq mMasqImage;
   tDImDepth * mDDepthImage;
   tDImMasq * mDMasqImage;

   tImDepth mImInterp;
   tImMasq mIMasqOut;
   tDImDepth * mDImInterp;
   tDImMasq * mDIMasqOut;

   tREAL8 mThreshGrad;
   tREAL8 mNoisePx;
   int mMasq2Tri;
   int mNbPointByPatch;
  };

  /* *************************************************** */
  /*                                                     */
  /*              cAppliMMGenDepthMV                     */
  /*                                                     */
  /* *************************************************** */


  // OBJECTIVE:
  // FOR EACH IMAGES --> DEPTH MAP GIVEN LIDAR
  //                 --> LIST OF IMAGES AND BOUNDING BOXES OVERLAPPING WITH IT

  cAppliMMGenDepthMV::cAppliMMGenDepthMV(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
     cMMVII_Appli  (aVArgs,aSpec),
     cAppliParseBoxIm<tREAL4>  (*this,eForceGray::Yes,cPt2di(2000,2000),cPt2di(50,50),false),
     mPhProj          (*this),
     mTri3D           (nullptr),
     mTri3DReproj     (nullptr),
     mCamPC           (nullptr),
     mDepthImage    (cPt2di(1,1)),
     mMasqImage     (cPt2di(1,1)),
     mDDepthImage       (nullptr),
     mDMasqImage       (nullptr),
     mImInterp    (cPt2di(1,1)),
     mIMasqOut     (cPt2di(1,1)),
     mDImInterp       (nullptr),
     mDIMasqOut       (nullptr),
     mThreshGrad                (0.3),
     mNoisePx                   (1.0),
     mMasq2Tri                  (0),
    mNbPointByPatch (32)
  {
  }


  cCollecSpecArg2007 & cAppliMMGenDepthMV::ArgObl(cCollecSpecArg2007 & anArgObl)
  {
      return anArgObl
             << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
             << Arg2007(mPatternLidar,"Pattern of input clouds",{{eTA2007::MPatFile,"1"}})
             << mPhProj.DPOrient().ArgDirInMand()
          ;
  }

  cCollecSpecArg2007 & cAppliMMGenDepthMV::ArgOpt(cCollecSpecArg2007 & anArgOpt)
  {
     return anArgOpt
            << AOpt2007(mNameOutDepth,"DepthMap" ,"Depth Map image Name")
             << AOpt2007(mNbPointByPatch,"NbNeighbors", "number of neighbors")
     ;
  }



  bool cAppliMMGenDepthMV::MakeDecision(std::vector<cPt3dr> & aVecPoints)
  {

      // implement the z_buffer algo to select visible points fot a certain point of view
      // center point is the first one
      cPt3dr PatchCenter=aVecPoints[0];
      auto It=aVecPoints.begin();
      aVecPoints.erase(It);
      //1. approximate plane with the lowest depths and check if we are higher or not
      std::sort(aVecPoints.begin(),aVecPoints.end(),compdepth);
      std::cout<<"aPlane "<<aVecPoints.size()<<std::endl;
      std::cout<<"points "<<aVecPoints[0]<<"  "<<aVecPoints[1]<<"  "<<aVecPoints[2]<<"  "<<
          aVecPoints[3]<<"  "<<aVecPoints[4]<<"  "<<aVecPoints[5]<<"  "<<PatchCenter.z()<<std::endl;

      cPlane3D aLowerPointsPlane= cPlane3D::From3Point(aVecPoints[0],
                                                        aVecPoints[1],
                                                        aVecPoints[2]);
      std::cout<<"aPlane 2"<<std::endl;
      //2. check if center depth is above or under plane
      tREAL8 aScal=Scal(aLowerPointsPlane.AxeK(),PatchCenter-aLowerPointsPlane.P0());
      std::cout<<"ASCALAR "<<aScal<<std::endl;

      if (aScal<0) // lower point
          return true;
      else
        return false;
  }

  void cAppliMMGenDepthMV::Generate_sparse_depth(std::string aNameImage,
                                                 std::vector<std::string> aVecLidar,
                                                 size_t aSzMin,
                                                 tREAL8 aThresholdVisbility)
  {
      mCamPC=mPhProj.ReadCamPC(aNameImage,true);

      // Depth
      mDepthImage=cIm2D<tElemDepth>(mCamPC->Sz(),nullptr,eModeInitImage::eMIA_Null);
      mDDepthImage=&(mDepthImage.DIm());

      //Masq

      mMasqImage=cIm2D<tElemMasq>(mCamPC->Sz(),nullptr,eModeInitImage::eMIA_Null);
      mDMasqImage=&(mMasqImage.DIm());

      tREAL8 aDistReject;
      // project cloud into image geometry x,y and depth
      std::vector <cPt3dr> aVPts;
      std::vector <cPt3di> aFaces;
      {
          for (const std::string & aLidarName: aVecLidar)
          {
              mTri3D= new cTriangulation3D<tREAL8>(aLidarName);

              for (size_t aKP=0; aKP<mTri3D->NbPts();aKP++)
              {
                  cPt3dr aP3D=ToR(mTri3D->KthPts(aKP));
                  if (mCamPC->IsVisible(aP3D))
                  {
                      cPt3dr aPt(mCamPC->Ground2Image(aP3D).x(),
                                 mCamPC->Ground2Image(aP3D).y(),
                                 mCamPC->Pose().Inverse(aP3D).z());

                      aVPts.push_back(aPt);
                  }

              }
          }
      }

      mTri3D=nullptr;

      if (aVPts.empty())
          return;

      mTri3DReproj =new cTriangulation3D<tREAL8>(aVPts,aFaces);

      //1. Nearest Neighbor search in 2D
      cBox2dr aBox = mTri3DReproj->Box2D();
      std::cout<<"BOX  "<<aBox<<std::endl;
      aDistReject=1.0*std::sqrt(mNbPointByPatch *aBox.NbElem()/ (mTri3DReproj->NbPts()*M_PI));
      // indexation of all points
      std::cout<<"aDist Reject "<<aDistReject<<std::endl;
      cTiling<cTil2DTri3D<tREAL8> >  aTileAll(aBox,true,mTri3DReproj->NbPts()/20,mTri3DReproj);
      for (size_t aKP=0 ; aKP<mTri3DReproj->NbPts() ; aKP++)
      {
          aTileAll.Add(cTil2DTri3D<tREAL8>(aKP));
      }
      #pragma omp parallel
      {
        #pragma omp for
          for (size_t aKPt=0; aKPt<mTri3DReproj->NbPts(); aKPt++)
          {
              cPt2dr aPt= ToR(Proj(mTri3DReproj->KthPts(aKPt)));
              auto aLIptr = aTileAll.GetObjAtDist(aPt,aDistReject);
              std::vector<int> aPatch; // the patch itself = index of points
              aPatch.push_back(aKPt);  // add the center at begining
              for (const auto aPtrI : aLIptr)
              {
                  if (aPtrI->Ind() !=aKPt) // dont add the center twice
                  {
                      aPatch.push_back(aPtrI->Ind());
                  }
              }

              std::vector<cPt3dr> aVP;
              for (const auto anInd : aPatch)
                  aVP.push_back(ToR(mTri3DReproj->KthPts(anInd)));

              if(aVP.size()<aSzMin)
                  continue;

              std::vector<tREAL8> aVDepths;

              for (const auto & aPt: aVP)
              {
                aVDepths.push_back(aPt.z());
              }


              // add another criterion

              tREAL8 aDepthMin = *min_element(aVDepths.begin(),aVDepths.end());
              tREAL8 aDepthMax = *max_element(aVDepths.begin(),aVDepths.end());
              // Compute visibility criterion
              tREAL8 aVisibility=exp(-pow((aVDepths.at(0)-aDepthMin),2)/pow((aDepthMax-aDepthMin+1e-8),2));

              if (aVisibility<aThresholdVisbility)
                  continue;
              /*if( ! MakeDecision(aVP))
                  continue;*/
              cPt2di aP2DCam=Pt_round_ni(cPt2dr(aVP[0].x(),aVP[0].y()));
              if (mDDepthImage->Inside(aP2DCam))
              {
                  if (mDMasqImage->GetV(aP2DCam))
                    {
                      if (mDDepthImage->GetV(aP2DCam)>aVP[0].z())
                      {
                          mDDepthImage->SetV(aP2DCam,aVP[0].z());
                      }
                    }
                  else
                    {
                      mDDepthImage->SetV(aP2DCam,aVP[0].z());
                      mDMasqImage->SetV(aP2DCam,1);
                    }
              }
          }
      }
      mTri3DReproj= nullptr;

      // save Depth image
      mDDepthImage->ToFile(mNameOutDepth+mCamPC->NameImage());
      mDMasqImage->ToFile(mNameOutMasq+mCamPC->NameImage());

      mCamPC=nullptr;
      mDDepthImage=nullptr;
      mDMasqImage=nullptr;

      mDepthImage=cPt2di(1,1);
      mMasqImage=cPt2di(1,1);
  }

  void cAppliMMGenDepthMV::MakeOneTri(const  tTriangle2DCompiled & aTri)
  {
      bool   isHGrowPx=false;

      // Compute 3 value in a point
      cPt3dr aPPx;
      for (int aKp=0 ; aKp<3 ; aKp++)
      {
          aPPx[aKp] = mDDepthImage->GetV(ToI(aTri.Pt(aKp)));
      }
      //  Tricky for WMM, but if used aWMM() => generate warning
      cWhichMinMax<int,double>  aWMM(0,aPPx[0]);
      for (int aKp=1 ; aKp<3 ; aKp++)
      {
          aWMM.Add(aKp,aPPx[aKp]);
      }

      // Compute Min,Max,Med
      int aKMin = aWMM.Min().IndexExtre();
      int aKMax = aWMM.Max().IndexExtre();
      int aKMed = 3-aKMin-aKMax;   // KMed is remaining index : 0,1,2 => sum equal 3

      double aPxMax = aPPx[aKMax];
      double aPxMin = aPPx[aKMin];
      double aPxMed = aPPx[aKMed];

      // Compute attenuation to take into account noise in gradident estimate ,
      double aMul = 1;
      double anEc = aPxMax - aPxMin;
      if (anEc!=0)
      {
          aMul = std::max(0.0,anEc-mNoisePx)/anEc;
      }

      // Compute occlusion on gradient threshold
      cPt2dr aG  = aTri.GradientVI(aPPx)*aMul;
      double aNG = Norm2(aG);
      bool isOcclusion = (aNG>mThreshGrad);
      int aValMasq = isOcclusion ? 0 : 255;

      int aKLow = isHGrowPx ? aKMin : aKMax;
      double  aValOcl = aPPx[aKLow];
      bool isTri2Low =  isOcclusion && (std::abs(aPxMed-aValOcl)<anEc/2.0);

      double aValTri;
      cPt2dr aVecTri;
      if (isTri2Low)  // Case where two vertices of the triangle are low
      {
          aValMasq = mMasq2Tri;
          cSegment aSeg(aTri.Pt(aKLow),aTri.Pt(aKMed));
          aSeg.CompileFoncLinear(aValTri,aVecTri,aPPx[aKLow],aPPx[aKMed]);
      }

      //  Now compute all the pixel and set the value

      static std::vector<cPt2di> aVPixTri;
      aTri.PixelsInside(aVPixTri);
      for (const auto & aPix : aVPixTri)
      {
          if (isOcclusion)
          {
              if (isTri2Low)  // 2 point low, interpol along segment
                  mDImInterp->SetV(aPix,aValTri+Scal(aVecTri,ToR(aPix)));
              else
                  mDImInterp->SetV(aPix,aValOcl);  // One point low, used lowest value
          }
          else   // Not occluded, use linear interpol
          {
              mDImInterp->SetV(aPix,aTri.ValueInterpol(ToR(aPix),aPPx));
          }
          mDIMasqOut->SetV(aPix,aValMasq);
      }
  }



  int cAppliMMGenDepthMV::ExeOnParsedBox()
  {
      // densify depth map using delaunay triangulation
      mDepthImage = APBI_ReadIm<tElemDepth>(mNameOutDepth+mCamPC->NameImage());
      mDDepthImage = &(mDepthImage.DIm());

      mMasqImage = APBI_ReadIm<tElemMasq>(mNameOutMasq+mCamPC->NameImage());
      mDMasqImage= &(mMasqImage.DIm());


      mImInterp = cIm2D<tElemDepth>(mDDepthImage->Sz(),nullptr,eModeInitImage::eMIA_Null);
      mDImInterp = &(mImInterp.DIm());
      mIMasqOut  = cIm2D<tElemMasq>(mDMasqImage->Sz(),nullptr,eModeInitImage::eMIA_Null);
      mDIMasqOut = &mIMasqOut.DIm();

      std::vector<cPt2dr> aVPts;
      for (const auto & aPix : *mDMasqImage)
      {
          if (mDMasqImage->GetV(aPix))
              aVPts.push_back(ToR(aPix));
      }

      if (aVPts.size()>3)
      {
          cTriangulation2D<tCoordDensify> aTriangul(aVPts);
          aTriangul.MakeDelaunay();
          StdOut() << "NbFace= " <<  aTriangul.NbFace() << "\n";


          // Initiate image of interpolated value
          for (size_t aKTri=0 ; aKTri<aTriangul.NbFace() ; aKTri++)
          {
              MakeOneTri(cTriangle2DCompiled(aTriangul.KthTri(aKTri)));
          }
      }
      else
      {}

      APBI_WriteIm("DensifyPx_"+LastPrefix(APBI_NameIm()) + ".tif",mImInterp);
      //mDImInterp->ToFile("DensifyPx_"+LastPrefix(APBI_NameIm()) + ".tif");
      APBI_WriteIm("DensifyMasq_"+LastPrefix(APBI_NameIm()) + ".tif",mIMasqOut);

      return EXIT_SUCCESS;
  }


  int cAppliMMGenDepthMV::Exe()
  {
      mPhProj.FinishInit();

    // image names pattern
      std::vector<std::string> aVecIms= VectMainSet(0);

    // Pattern of Lidar data files
      std::vector<std::string> aVecLidar= VectMainSet(1);
      size_t aSzMin=5;
      tREAL8 aThresholdVisbility=0.8;

      for (const auto & anImageName: aVecIms)
      {
          // generate depth

          Generate_sparse_depth(anImageName,
                                aVecLidar,
                                aSzMin,
                                aThresholdVisbility);

          std::cout<<"Generated depth map : sparse "<<anImageName<<std::endl;

          this->mNameIm=anImageName;
          mCamPC=mPhProj.ReadCamPC(anImageName,true);
          // densify
          if (RunMultiSet(0,0))
              return ResultMultiSet();
          APBI_ExecAll();
      }

     // Densify Depth Map using Delaunay Triangulation


      return EXIT_SUCCESS;
  }

};



using namespace  cNS_MMGenDepthMV;


tMMVII_UnikPApli Alloc_MMGenDepthMV(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliMMGenDepthMV(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpecMMGenDepthMV
(
     "DepthMapForImage",
      Alloc_MMGenDepthMV,
      "Generate a depth map for each image given orientation and Lidar point clouds",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
