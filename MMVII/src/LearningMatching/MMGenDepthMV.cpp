#include <StdAfx.h>
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_DeclareCste.h"
#include <fstream>
#include <iostream>
#include "MMVII_ZBuffer.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include "MMVII_2Include_Tiling.h"
#include "LearnDM.h"
#include "MMVII_AllClassDeclare.h"
#include "MMVII_Interpolators.h"


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
    int ExeScale();
    int Exe() override;
    int ExeOnParsedBox() override;
    bool MakeDecision(std::vector<cPt3dr> & aVecPoints);

    std::string NameImOri(std::string NameIM,std::string OriFolder, string SuffOri);

    void ReadLidarTile (std::string aLidarTileName,std::vector<cPt3dr>&  aVPts_private);
    void ReadLidarTiles (std::vector<cTriangulation3D<tREAL8>*>& allTris3D,
                        std::vector<std::vector<cPt3dr>>&  aVPtsAll);
    void Generate_sparse_depth(std::string aNameImage,
                                size_t aSzMin,
                                tREAL8 aThresholdVisbility,
                                std::vector<cPt3dr>& aVPts,
                                std::vector<cPt3di>& aFaces);
    void OneVisibilitySimple(size_t & aSzMin,
                             tREAL8 & aThresholdCriteria,
                             int nbIter,
                             bool isVisibility=true);

    void OneVisibility(size_t & aSzMin,
                       tREAL8 & aThresholdCriteria,
                      /* std::vector<tREAL8> & aSurfVec,*/
                       int nbIter,
                       bool isVisibility=true);
    void filter_on_visibility(std::vector<tREAL8> & aSurfVec, tREAL8 aThresh);
    void MakeOneTri(const  tTriangle2DCompiled & aTri);

  // --- constructed ---
   cPhotogrammetricProject   mPhProj;
   cTriangulation3D<tREAL8>* mTri3D;
   cTriangulation3D<tREAL8>* mTri3DReproj;
   std::string mSpecImIn;
   std::string mPatternLidar;
   std::string mOriFolder;
   std::string mNameOutDepth="Depth_";
   std::string mNameOutMasq ="Masq_2_";
   size_t                    mNbF;
   size_t                    mNbP;
   cSensorCamPC *            mCamPC;
   CamStenope * mCamPCV1;
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
   std::vector<int> mFirstVisIndices;
   std::vector<int> mSelectedVisIndices;
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
     mCamPCV1         (nullptr),
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
     mNbPointByPatch (15),
     mFirstVisIndices (std::vector<int>()),
     mSelectedVisIndices (std::vector<int>())
  {
  }


  cCollecSpecArg2007 & cAppliMMGenDepthMV::ArgObl(cCollecSpecArg2007 & anArgObl)
  {
      return anArgObl
             << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"}})
             << Arg2007(mOriFolder,"Pattern of input clouds",{{eTA2007::FolderAny,"2"}})
             //<< mPhProj.DPOrient().ArgDirInMand()
             //<< mPhProj.DPMeshDev().ArgDirOutMand()
          ;
  }

  cCollecSpecArg2007 & cAppliMMGenDepthMV::ArgOpt(cCollecSpecArg2007 & anArgOpt)
  {
     return anArgOpt
             << AOpt2007(mPatternLidar,"PatLidar","Pattern (regex), def=Semis_.*laz",{eTA2007::HDV})
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

      // measure concavity index using eigenvalues
  }


  void cAppliMMGenDepthMV::filter_on_visibility(std::vector<tREAL8> & aSurfVec,
                                                tREAL8 aThresh)
  {
      std::vector<cPt3dr> aVSPts;
      std::vector <cPt3di> aFaces;

      MMVII_INTERNAL_ASSERT_strong(aSurfVec.size()==mSelectedVisIndices.size(),"Problem with size of tri and visibility stored !");

      for (size_t iKP=0; iKP<mSelectedVisIndices.size();iKP++)
      {
          cPt3dr aP3Vis(mTri3DReproj->KthPts(iKP).x(),
                     mTri3DReproj->KthPts(iKP).y(),
                     mTri3DReproj->KthPts(iKP).z());
          aVSPts.push_back(aP3Vis);
      }

    // nearest neighbor analysis based on visibility values
      cTriangulation3D<tREAL8>* aTriVisibility= new cTriangulation3D<tREAL8>(aVSPts,aFaces);

      mSelectedVisIndices.clear();
      aSurfVec.clear();

      tREAL8 aDistReject;
      //1. Nearest Neighbor search in 2D
      cBox2dr aBox = aTriVisibility->Box2D();
      aDistReject=1.0*std::sqrt(mNbPointByPatch *aBox.NbElem()/ (aTriVisibility->NbPts()*M_PI));
      // indexation of all points
      std::cout<<"aDist Reject "<<aDistReject<<std::endl;
      cTiling<cTil2DTri3D<tREAL8> >  aTileAll(aBox,true,aTriVisibility->NbPts()/20,aTriVisibility);
      for (size_t aKP=0 ; aKP<aTriVisibility->NbPts() ; aKP++)
      {
          aTileAll.Add(cTil2DTri3D<tREAL8>(aKP));
      }
      //#pragma omp parallel
      {
          //#pragma omp for
          for (size_t aKPt=0; aKPt<aTriVisibility->NbPts(); aKPt++)
          {
              cPt2dr aPt= ToR(Proj(aTriVisibility->KthPts(aKPt)));
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
                  aVP.push_back(ToR(aTriVisibility->KthPts(anInd)));

              /*
              std::vector<tREAL8> aVVisibs;
              for (size_t iKP=0; iKP<aVP.size();iKP++)
              {
                  aVVisibs.push_back(aVP[iKP].z());
              }

              //compare with mean or median visibility
              tREAL8 aVisibility= aVVisibs[0];
              tREAL8 aMed= NC_KthVal(aVVisibs,0.5) ;
              */


              std::vector<tREAL8> aVDepths;
              for (size_t iKP=0; iKP<aVP.size();iKP++)
              {
                  aVDepths.push_back(aVP[iKP].z());
              }

              //compare with mean or median visibility
              tREAL8 aDepth0= aVDepths[0];
              tREAL8 aMed= NC_KthVal(aVDepths,0.5) ;


              //StdOut()<<" Visibility : "<<aVisibility<<"  "<<" aMed : "<<aMed<<std::endl;
              if ((aDepth0 < aMed))
              {
                  //StdOut()<<" Visibility : "<<aVisibility<<"  "<<" aMed : "<<aMed<<std::endl;
                  aSurfVec.push_back(aDepth0);
                  mSelectedVisIndices.push_back(aKPt);
              }
          }
      }
      delete aTriVisibility ;
  }


  void cAppliMMGenDepthMV::ReadLidarTiles (std::vector<cTriangulation3D<tREAL8>*>& allTris3D,
                                         std::vector<std::vector<cPt3dr>> &  aVPtsAll)
  {
      //cTriangulation3D<tREAL8> aTri3D=  allTris3D[0];

      for (int i=0; i<(int)allTris3D.size();i++)
      {
          #pragma omp parallel
          {
              std::vector <cPt3dr> aVPts_pp;
              #pragma omp for
              for (size_t aKP=0; aKP<allTris3D[i]->NbPts();aKP++)
              {
                  cPt3dr aP3D=ToR(allTris3D[i]->KthPts(aKP));
                  Pt3dr aP3DMMV1= Pt3dr(aP3D.x(),aP3D.y(),aP3D.z());
                  if (mCamPCV1->PIsVisibleInImage(aP3DMMV1))
                  {
                      Pt2dr aP2DCapteur = mCamPCV1->Ter2Capteur(aP3DMMV1);

                      /*
                              cPt3dr aPt(mCamPC->Ground2Image(aP3D).x(),
                                         mCamPC->Ground2Image(aP3D).y(),
                                         mCamPC->Pose().Inverse(aP3D).z());
                              */
                      cPt3dr aPt (aP2DCapteur.x,
                                 aP2DCapteur.y,
                                 mCamPCV1->ProfondeurDeChamps(aP3DMMV1));
                      aVPts_pp.push_back(aPt);
                  }
              }

              #pragma omp critical
              aVPtsAll[i].insert(aVPtsAll[i].end(),
                                   aVPts_pp.begin(),
                                   aVPts_pp.end());
          }
      }
  }



void cAppliMMGenDepthMV::ReadLidarTile (std::string aLidarTileName,
                                            std::vector<cPt3dr>&  aVPts_private)
  {

      //StdOut()<<"test "<<std::endl;
      std::vector<cTriangulation3D<tREAL8>*> allTris3D;
      #pragma  omp parallel for
      for (int i=0; i<2; i++)
      {
          cTriangulation3D<tREAL8> * aTri3D= new cTriangulation3D<tREAL8>(aLidarTileName);
          allTris3D.push_back(aTri3D);
      }
      //cTriangulation3D<tREAL8> aTri3D=  allTris3D[0];

     #pragma omp parallel
      {
          std::vector <cPt3dr> aVPts_pp;
          #pragma omp for
          for (size_t aKP=0; aKP<allTris3D[0]->NbPts();aKP++)
          {
              cPt3dr aP3D=ToR(allTris3D[0]->KthPts(aKP));
              Pt3dr aP3DMMV1= Pt3dr(aP3D.x(),aP3D.y(),aP3D.z());
              if (mCamPCV1->PIsVisibleInImage(aP3DMMV1))
              {
                  Pt2dr aP2DCapteur = mCamPCV1->Ter2Capteur(aP3DMMV1);

                  /*
                          cPt3dr aPt(mCamPC->Ground2Image(aP3D).x(),
                                     mCamPC->Ground2Image(aP3D).y(),
                                     mCamPC->Pose().Inverse(aP3D).z());
                          */
                  cPt3dr aPt (aP2DCapteur.x,
                             aP2DCapteur.y,
                             mCamPCV1->ProfondeurDeChamps(aP3DMMV1));
                  aVPts_pp.push_back(aPt);
              }
          }

          #pragma omp critical
          aVPts_private.insert(aVPts_private.end(),
                                     aVPts_pp.begin(),
                                     aVPts_pp.end());
        }
  }

  void cAppliMMGenDepthMV::Generate_sparse_depth(std::string aNameImage,
                                                 size_t aSzMin,
                                                 tREAL8 aThresholdVisbility,
                                                 std::vector<cPt3dr>& aVPts,
                                                 std::vector<cPt3di>& aFaces)
  {
      //mCamPC=mPhProj.ReadCamPC(aNameImage,true);

      // Depth
      cPt2di aSz =cPt2di(mCamPCV1->Sz().x,mCamPCV1->Sz().y);
      mDepthImage=cIm2D<tElemDepth>(aSz,nullptr,eModeInitImage::eMIA_Null);
      mDDepthImage=&(mDepthImage.DIm());

      //Masq

      mMasqImage=cIm2D<tElemMasq>(aSz,nullptr,eModeInitImage::eMIA_Null);
      mDMasqImage=&(mMasqImage.DIm());


      mTri3DReproj =new cTriangulation3D<tREAL8>(aVPts,aFaces);

      if (0)
      {
          OneVisibilitySimple(aSzMin,aThresholdVisbility,0);
      }

      if (1)
      {
          mSelectedVisIndices.clear();

          OneVisibility(aSzMin,aThresholdVisbility,0);

         std::vector<cPt3dr> aVSPts;

          for ( const auto & aKPt : mSelectedVisIndices)
          {
              aVSPts.push_back(mTri3DReproj->KthPts(aKPt));
          }

          mTri3DReproj= nullptr;

          mTri3DReproj= new cTriangulation3D<tREAL8>(aVSPts,aFaces);

          mSelectedVisIndices.clear();

          OneVisibility(aSzMin,aThresholdVisbility,1);

         for ( const auto & aKPt : mSelectedVisIndices)
              {
                  cPt3dr aPtS= mTri3DReproj->KthPts(aKPt);

                  cPt2di aP2DCam=Pt_round_ni(cPt2dr(aPtS.x(),aPtS.y()));
                  if (mDDepthImage->Inside(aP2DCam))
                  {
                      if (mDMasqImage->GetV(aP2DCam))
                      {
                          if (mDDepthImage->GetV(aP2DCam)>aPtS.z())
                          {
                              mDDepthImage->SetV(aP2DCam,aPtS.z());
                          }
                      }
                      else
                      {
                          mDDepthImage->SetV(aP2DCam,aPtS.z());
                          mDMasqImage->SetV(aP2DCam,1);
                      }
                  }
              }
          StdOut()<<" Selected VISIBLE Points "<<mSelectedVisIndices.size()<<std::endl;
      }

      mTri3DReproj= nullptr;

      // save Depth image
      mDDepthImage->ToFile(mNameOutDepth+/*mCamPC->NameImage()*/ aNameImage);
      mDMasqImage->ToFile(mNameOutMasq+/*mCamPC->NameImage()*/ aNameImage);

      //mCamPC=nullptr;
      mDDepthImage=nullptr;
      mDMasqImage=nullptr;

      mDepthImage=cPt2di(1,1);
      mMasqImage=cPt2di(1,1);
  }

  void cAppliMMGenDepthMV::OneVisibility(size_t & aSzMin,
                                         tREAL8 & aThresholdCriteria,
                                         int nbIter,
                                         bool isVisibility)
  {
      tREAL8 aDistReject;
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
          std::vector <int> aVPts_private;
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

              cWhichMinMax<int, tREAL8> aWMM(0,aVP[0].z());
              for (size_t iKP=1; iKP<aVP.size();iKP++)
              {
                  aWMM.Add(iKP,aVP[iKP].z());
              }
              // add another criterion

              if (isVisibility)
              {
                  tREAL8 aDMax= aVP[aWMM.Max().IndexExtre()].z();
                  tREAL8 aDMin= aVP[aWMM.Min().IndexExtre()].z();
                  tREAL8 aVisibility = exp (-pow(aVP[0].z()-aDMin,2)/pow(aDMax-aDMin,2));

                if (aVisibility<aThresholdCriteria)
                    continue;

                  /*tREAL8 aSurfaceVariation;
                  if (nbIter>0)
                  {

                      aSurfaceVariation= Surface_Variation(aVP);
                      if(aSurfaceVariation>0.01)
                      {
                          aVisibility= exp (-pow(aVP[0].z()-aDMin,0.5)/pow(aDMax-aDMin,0.5));
                        // check if surface variation is important
                          if (aVisibility<aThresholdCriteria)
                            continue;
                      }
                  }*/
              }

              aVPts_private.push_back(aKPt);
          }
          //StdOut()<<" slices "<<aVPts_private.size()<<std::endl;
        // fill global aVPts
        #pragma omp critical
            mSelectedVisIndices.insert(mSelectedVisIndices.end(),
                                     aVPts_private.begin(),
                                     aVPts_private.end());
      }
  }


  void cAppliMMGenDepthMV::OneVisibilitySimple(size_t & aSzMin,
                                         tREAL8 & aThresholdCriteria,
                                         int nbIter,
                                         bool isVisibility)
  {
      tREAL8 aDistReject;
      //1. Nearest Neighbor search in 2D
      cBox2dr aBox = mTri3DReproj->Box2D();
      std::cout<<"BOX  "<<aBox<<std::endl;
      aDistReject=1.0*std::sqrt(mNbPointByPatch *aBox.NbElem()/ (mTri3DReproj->NbPts()*M_PI));
      // indexation of all points
      std::cout<<"aDist Reject "<<aDistReject<<std::endl;
      cTiling<cTil2DTri3D<tREAL8> >  aTileAll(aBox,true,mTri3DReproj->NbPts()/20,mTri3DReproj);
      {
          for (size_t aKP=0 ; aKP<mTri3DReproj->NbPts() ; aKP++)
          {
              aTileAll.Add(cTil2DTri3D<tREAL8>(aKP));
          }
      }

      #pragma omp parallel
      {
          #pragma omp for
          for (size_t aKPt=0; aKPt<mTri3DReproj->NbPts(); aKPt++)
          {
              cPt3dr aP3D=mTri3DReproj->KthPts(aKPt);
              cPt2dr aPt= ToR(Proj(aP3D));
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

              cWhichMinMax<int, tREAL8> aWMM(0,aVP[0].z());
              for (size_t iKP=1; iKP<aVP.size();iKP++)
              {
                  aWMM.Add(iKP,aVP[iKP].z());
              }

              tREAL8 aDMax= aVP[aWMM.Max().IndexExtre()].z();
              tREAL8 aDMin= aVP[aWMM.Min().IndexExtre()].z();
              tREAL8 aVisibility = exp (-pow(aVP[0].z()-aDMin,2)/pow(aDMax-aDMin,2));

              if (aVisibility<aThresholdCriteria)
                  continue;

              // fill in image
              cPt2di aP2DCam=Pt_round_ni(aPt);
              if (mDDepthImage->Inside(aP2DCam))
              {
                  if (mDMasqImage->GetV(aP2DCam))
                  {
                      if (mDDepthImage->GetV(aP2DCam)>aP3D.z())
                      {
                          mDDepthImage->SetV(aP2DCam,aP3D.z());
                      }
                  }
                  else
                  {
                      mDDepthImage->SetV(aP2DCam,aP3D.z());
                      mDMasqImage->SetV(aP2DCam,1);
                  }
              }
          }
      }
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

  std::string cAppliMMGenDepthMV::NameImOri(std::string NameIM,std::string OriFolder, string SuffOri)
  {
      return OriFolder+"/"+SuffOri+NameIM+".xml";
  }


  int cAppliMMGenDepthMV::ExeOnParsedBox()
  {
      // densify depth map using delaunay triangulation
      mDepthImage = APBI_ReadIm<tElemDepth>(mNameOutDepth+mNameIm);
      mDDepthImage = &(mDepthImage.DIm());

      mMasqImage = APBI_ReadIm<tElemMasq>(mNameOutMasq+mNameIm);
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


      if (IsInit(&mPatternLidar))
      {
          std::vector<string> aVecLidar;

          for (const auto & aName :  RecGetFilesFromDir(mDirProject,AllocRegex(mPatternLidar),0,1))
          {
              aVecLidar.push_back(aName);
          }

          StdOut()<<"aVEC LIDAR SOZE "<<aVecLidar.size()<<std::endl;

          // Read all tiles lidar

          std::vector<cTriangulation3D<tREAL8>*> allTris3D;
          //#pragma omp parallel for
          for (int i=0; i<(int)aVecLidar.size();i++)
          {
              cTriangulation3D<tREAL8> * aTri3D= new cTriangulation3D<tREAL8>(aVecLidar[i]);
              allTris3D.push_back(aTri3D);
          }



          // compute orifolder
          cInterfChantierNameManipulateur * aICNMOris=cInterfChantierNameManipulateur::BasicAlloc(mOriFolder);

          size_t aSzMin=5;
          tREAL8 aThresholdVisbility=0.68;

          for (const auto & anImageName : aVecIms)

          {
              // generate depth
              std::string aNameImOri = NameImOri(anImageName,mOriFolder,"Orientation-");

              mCamPCV1 = CamOrientGenFromFile(aNameImOri,aICNMOris);

              // project cloud into image geometry x,y and depth
              std::vector <cPt3dr> aVPts;
              std::vector <cPt3di> aFaces;


              std::vector<std::vector<cPt3dr>> aVAll{aVecLidar.size()};
              ReadLidarTiles(allTris3D,aVAll);
              StdOut()<<"READ TILES "<<std::endl;
              for (auto aV: aVAll)
                  aVPts.insert(aVPts.end(),aV.begin(),aV.end());

              StdOut()<<"size "<<aVPts.size()<<std::endl;


              if (aVPts.empty())
                  return EXIT_SUCCESS;

              Generate_sparse_depth(anImageName,
                                    aSzMin,
                                    aThresholdVisbility,
                                    aVPts,
                                    aFaces);
              std::cout<<"Generated depth map : sparse "<<anImageName<<std::endl;

              this->mNameIm=anImageName;
              //mCamPC=mPhProj.ReadCamPC(anImageName,true);

               // densify
              /*if (RunMultiSet(0,0))
                  return ResultMultiSet();*/
              APBI_ExecAll();
          }

    }



      return EXIT_SUCCESS;
  }


 // Test MulScaledInterpolator


  int cAppliMMGenDepthMV::ExeScale()
  {
      mPhProj.FinishInit();

      // image names pattern
      std::vector<std::string> aVecIms= VectMainSet(0);

      std::string aNameImage= aVecIms[0];

      mCamPC=mPhProj.ReadCamPC(aNameImage,true);

      cIm2D<tREAL4> anIm = cIm2D<tREAL4>::FromFile(mCamPC->NameImage());

      // Test MulScaleInterpolator

      std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};

     // cDiffInterpolator1D *  mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);

      //std::unique_ptr<cTabulatedDiffInterpolator>  aTabInt ( cScaledInterpolator::AllocTab(*mInterp,2.0,1000));
      tREAL4 aScale=3.0;
      tREAL4 aDilatedFactor =2.0 ;
      cDiffInterpolator1D *  mInterp = cScaledInterpolator::AllocTab(cCubicInterpolator(-0.5),aScale*aDilatedFactor,1000);

      cIm2D<tREAL4>  aImSc  = anIm.Scale(*mInterp,aScale);


      // save

      aImSc.DIm().ToFile(mCamPC->NameImage()+"_scaled_3_2_dilated_.tif");

      delete mInterp;

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
