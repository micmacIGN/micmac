#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Triangles.h"
#include "MMVII_Image2D.h"
#include "MMVII_ZBuffer.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include <fstream>
#include <iostream>
#include <StdAfx.h>



namespace MMVII {

namespace cNs_Sample3DpointsFromCloudIntoImage
{

class cAppliSample3DpointsFromCloudIntoImage;
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

/* =================================================================== */
/*                                                                     */
/*              cAppliSample3DpointsFromCloudIntoImage                 */
/*                                                                     */
/* =================================================================== */
typedef cPtxd<tREAL8,3> tPtD;
typedef cPtxd<tREAL4,3> tPtF;

class cAppliSample3DpointsFromCloudIntoImage: public cMMVII_Appli
{
    public:
        cAppliSample3DpointsFromCloudIntoImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private:
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


        // Mandatory arguments

        std::string mNamePattern2DIm;  ///< Pattern of images
        std::string mNamePattern3DCld; ///< Pattern of laz files from which to sample 3D points

        //optional


        //constructed

        cPhotogrammetricProject mPhProj;
        cTriangulation3DLas<tREAL8>* mTri3DLas;
        cSensorCamPC * mCamPC;

        std::vector<tPtD>  mV3DCld;
        std::vector<std::vector<cPt2dr>> mProj2Im;
        std::vector<std::vector<bool>> mVis2Im;
};


cAppliSample3DpointsFromCloudIntoImage::cAppliSample3DpointsFromCloudIntoImage(const std::vector<std::string> & aVArgs,
                                                                               const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
    // internal vars
   mPhProj          (*this),
   mTri3DLas        (nullptr),
   mCamPC           (nullptr)
{
}


cCollecSpecArg2007 & cAppliSample3DpointsFromCloudIntoImage::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<   Arg2007(mNamePattern2DIm,"Pattern of images", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
           <<   Arg2007(mNamePattern3DCld,"Pattern of input clouds",{{eTA2007::MPatFile,"1"}})
           <<   mPhProj.DPOrient().ArgDirInMand()
           <<   mPhProj.DPOrient().ArgDirOutMand()
           //<<   mPhProj.DPRadiomData().ArgDirInMand()
    ;
}


cCollecSpecArg2007 & cAppliSample3DpointsFromCloudIntoImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOptBench()
   ;
}


int cAppliSample3DpointsFromCloudIntoImage::Exe()
{
    mV3DCld.clear();
    mPhProj.FinishInit();
    //image pattern
    std::vector<std::string> aVecIm   = VectMainSet(0); //interface to MainSet
    std::cout<<"Names images "<<aVecIm.at(0)<<std::endl;

    //cloud pattern
    std::vector<std::string> aVecClds = VectMainSet(1); //interface to MainSet

    // read clouds and filter out and save targetted points
    for (const std::string & aCldName : aVecClds)
        {


            mTri3DLas= new cTriangulation3DLas<tREAL8>(aCldName);

            std::cout<<"CLd SIZE >>>>  "<<mTri3DLas->NbPts()<<std::endl;

            // Sample points on ground and buildings' rooftops
            mTri3DLas->SamplePts(true,1.0);
            for (auto & anId: mTri3DLas->SelectedIds())
              {
                mV3DCld.push_back(mTri3DLas->VPts()[anId]);
              }
            mTri3DLas=nullptr;
        }
    // parse all selected  3D cloud points and find their reprojection if they exist in sensors

   return EXIT_SUCCESS;
}

};

using namespace cNs_Sample3DpointsFromCloudIntoImage;
/* =============================================== */
/*                       ::                        */
/* =============================================== */

tMMVII_UnikPApli Alloc_Sample3DpointsFromCloudIntoImage(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
return tMMVII_UnikPApli(new cAppliSample3DpointsFromCloudIntoImage(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecSample3DpointsFromCloudIntoImage
(
"Sample3DpointsFromCloudIntoImage",
 Alloc_Sample3DpointsFromCloudIntoImage,
 "Compute Lidar based 3D points",
 {eApF::Ori},
 {eApDT::Orient},
 {eApDT::Orient},
 __FILE__
);

};


