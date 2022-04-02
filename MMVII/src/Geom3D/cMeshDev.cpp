#include "include/MMVII_all.h"


namespace MMVII
{

typedef tREAL8  tCoordDevTri;
typedef  cTriangulation3D<tCoordDevTri> tTriangulation3D;

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
          typedef typename cTriangulation<tCoordDevTri,3>::tFace tFace;

          static constexpr int NO_STEP = -1;

          cDevTriangu3d(const  tTriangulation3D &);

      private :
	  cDevTriangu3d(const cDevTriangu3d &) = delete;
	  void AddOneFace(int aKFace); // Mark the face and its sums as reached when not
	  int  NbUnreached(int aKFace) const; // Number of the 3 vertices not reached

	  int MakeNewFace();

	  int               mNumCurStep;
          int               mNbFaceReached;
	  const tTriangulation3D & mTri;
	  std::vector<int>  mStepReach_S;  ///< indicate if a submit is selected and at which step
	  // size_t                mLastNbSel;  
	  std::vector<int>  mStepReach_F;  ///< indicate if a face and at which step
};

void cDevTriangu3d::AddOneFace(int aKFace)
{
     if (mStepReach_F.at(aKFace) != NO_STEP) return; // if face already reached nothing to do

     mNbFaceReached++;
     mStepReach_F.at(aKFace) = mNumCurStep; // mark it reached at current step

     // Mark all som of face that are not marked
     const tFace & aFace = mTri.KthFace(aKFace);
     for (int aNumV=0 ; aNumV<3 ; aNumV++)
     {
         int aKS= aFace[aNumV];
         if (mStepReach_S.at(aKS) == NO_STEP)
            mStepReach_S.at(aKS) = mNumCurStep;
     }
}

int  cDevTriangu3d::NbUnreached(int aKFace) const
{
     const tFace & aFace = mTri.KthFace(aKFace);

     int aRes=0;
     for (int aNumV=0 ; aNumV<3 ; aNumV++)
     {
         int aKS= aFace[aNumV];
         if (mStepReach_S.at(aKS) == NO_STEP)
            aRes++;
     }
     return aRes;
}

int cDevTriangu3d::MakeNewFace()
{
     // A Face is adjacent to reached iff it contains exactly 2 reached 
     std::vector<int> aVFNeigh;  // put first in vect to avoir recursive add
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         if (NbUnreached(aKF)==1)
            aVFNeigh.push_back(aKF);
     }
     // Now mark the faces 
     for (const auto & aKF : aVFNeigh)
         AddOneFace(aKF);

     // Mark face that containt 3 reaches vertices (may have been created
     // by previous step)
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         if (NbUnreached(aKF)==0)
            AddOneFace(aKF);
     }

     return aVFNeigh.size();
}

cDevTriangu3d::cDevTriangu3d(const tTriangulation3D & aTri) :
     mNumCurStep  (0),
     mNbFaceReached (0),
     mTri         (aTri),
     mStepReach_S (mTri.NbPts() ,NO_STEP),
     mStepReach_F (mTri.NbFace(),NO_STEP)
{
    AddOneFace(mTri.IndexCenterFace());

    while (int aNbF=MakeNewFace())
    {
        StdOut() << "NnFF " << aNbF << "\n";
    }


    StdOut() << " FRR " << mNbFaceReached << " " << mTri.NbFace() << "\n";
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

   tTriangulation3D  aTri(mNameCloudIn);
   // aTri.WriteFile(DirProject()+mNameCloudOut,mBinOut);

   cDevTriangu3d aDev(aTri);

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

#if (0)
#endif


};
