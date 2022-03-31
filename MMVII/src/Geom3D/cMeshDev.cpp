#include "include/MMVII_all.h"


namespace MMVII
{


/* ******************************************************* */
/*                                                         */
/*                    cDevTriangu3d                        */
/*                                                         */
/* ******************************************************* */

/** Class that effectively compute the "optimal" devlopment of a surface
 * Separate from cAppliMeshDev to be eventually reusable
 */

class cDevTriangu3d
{
      public :
          typedef typename cTriangulation<3>::tFace tFace;
          cDevTriangu3d(const  cTriangulation3D &);
      private :
	  cDevTriangu3d(const cDevTriangu3d &) = delete;
	  void AddFace(int aKFace);

	  int               mNumStep;
	  const cTriangulation3D & mTri;
	  std::vector<int>  mSomStepSel;  ///< indicate if a submit is selected
	  //std::vector<int>   mVSomSel;   ///< vector of selected soms
	  // size_t                mLastNbSel;  
	  std::vector<int>  mFaceStepSel;  ///< indicate if a 
	  // std::vector<int>   mVFaceSel;   ///< vector of selected faces
};

void cDevTriangu3d::AddFace(int aFace)
{
}

cDevTriangu3d::cDevTriangu3d(const cTriangulation3D & aTri) :
     mNumStep      (0),
     mTri          (aTri),
     mSomStepSel   (mTri.NbPts(),-1),
     mFaceStepSel  (mTri.NbFace(),-1)
{
    AddFace(mTri.IndexCenterFace());
    // const tFace & aFace0 = mTri.CenterFace();

    // FakeUseIt(aFace0);
    // tPt PAvg() const
}
/* ******************************************************* */
/*                                                         */
/*                    cAppliMeshDev                        */
/*                                                         */
/* ******************************************************* */


/**  A basic application for clipping 3d data ,  almost all the job is done in
 * libraries so it essentially interface to command line */

class cAppliMeshDev : public cMMVII_Appli
{
     public :

        cAppliMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

           // --- Mandatory ----
	      std::string mNameCloudIn;
           // --- Optionnal ----
	      std::string mNameCloudOut;
	      bool        mBinOut;
           // --- Internal ----

};

cAppliMeshDev::cAppliMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliMeshDev::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileCloud})
   ;
}

cCollecSpecArg2007 & cAppliMeshDev::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file")
           << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
   ;
}


int  cAppliMeshDev::Exe()
{
   InitOutFromIn(mNameCloudOut,"Clip_"+mNameCloudIn);

   cTriangulation3D  aTri(mNameCloudIn);
   aTri.WriteFile(DirProject()+mNameCloudOut,mBinOut);

   return EXIT_SUCCESS;
}



/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_MeshDev(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliMeshDev(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMeshDev
(
     "MeshDev",
      Alloc_MeshDev,
      "Clip a point cloud/mesh  using a region",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);



};
