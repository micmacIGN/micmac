#include "include/MMVII_all.h"


namespace MMVII
{

   /* ======  header of header  ====== */
typedef tREAL8  tCoordDevTri;
typedef  cTriangulation3D<tCoordDevTri> tTriangulation3D;
typedef  cTriangle<tCoordDevTri,3> tTri3D;
typedef  cIsometry3D<tCoordDevTri> tIsom;
typedef  cPtxd<tCoordDevTri,3>     tPtTri3d;
typedef cTriangle<int,2> tTriPix;

class cGenerateSurfDevOri;
class cDevTriangu3d;

   /* ======================= */
   /* ======  header   ====== */
   /* ======================= */

class cGenerateSurfDevOri
{
     public :
         cGenerateSurfDevOri(const cPt2di & aNb = cPt2di(15,5));

	 // tTriangulation3D cTriangulation(const tVPt& =tVPt(),const tVFace & =tVFace());
	 //
	 std::vector<tPtTri3d> VPts() const;
	 std::vector<cPt3di>   VFaces() const;

	 tPtTri3d PCenter() const;

     private :
	 int  NumOfPix(const cPt2di & aKpt) const; ///< Basic numerotation of points using video sens
	 tPtTri3d  Pt3OfPix(const cPt2dr & aKpt) const; ///< Basic numerotation of points using video sens

	 cPt2di    mNb;
	 bool      mPlaneCart; ///< if true generate a plane cartesian mesh

};

/** Class that effectively compute the "optimal" devlopment of a surface
 * Separate from cAppliMeshDev to be eventually reusable
 */

class cDevTriangu3d
{
      public :
          typedef typename cTriangulation<tCoordDevTri,3>::tFace tFace;

          static constexpr int NO_STEP = -1;

          cDevTriangu3d(const  tTriangulation3D &);
          void DoDevlpt ();
	  void SetFaceC(int aNumF);

      private :
	  cDevTriangu3d(const cDevTriangu3d &) = delete;

	  void AddOneFace(int aKFace); // Mark the face and its sums as reached when not
	  std::vector<int>  Unreached(int aKFace) const; // subset of the 3 vertices not reached

	  // tPtTri3d

	  int MakeNewFace();

	  int               mNumCurStep;
          int               mNbFaceReached;
	  const tTriangulation3D & mTri;
	  std::vector<int>  mStepReach_S;  ///< indicate if a submit is selected and at which step
	  std::vector<cPt2dr>  mVPtsDev; ///< Vector of devloped 2D points
	  // size_t                mLastNbSel;  
	  std::vector<int>  mStepReach_F;  ///< indicate if a face and at which step
          int               mIndexFC; ///< Index of centerface
	  cPt3di            mFaceC;
};



/* ******************************************************* */
/*                                                         */
/*               cGenerateSurfDevOri                       */
/*                                                         */
/* ******************************************************* */

int  cGenerateSurfDevOri::NumOfPix(const cPt2di & aKpt)  const
{
     return aKpt.x() + aKpt.y() *  (mNb.x());
}


tPtTri3d  cGenerateSurfDevOri::Pt3OfPix(const cPt2dr & aKpt) const
{
     if (mPlaneCart)
     {
        return tPtTri3d(aKpt.x(),aKpt.y(),0.0);
     }
     double aNbTour =  1.5;
     double aRatiobByTour = 1.5;

     double aAmpleTeta =  aNbTour * 2 * M_PI;

     double aTeta =  aAmpleTeta * ((double(aKpt.x())  / (mNb.x()-1) -0.5));
     cPt2dr  aPPlan = FromPolar(pow(aRatiobByTour,aTeta/(2*M_PI)),aTeta);
     double  aZCyl = (aKpt.y() * aAmpleTeta) / (mNb.x()-1);

     tPtTri3d  aPCyl(aPPlan.x(),aPPlan.y(),aZCyl);


     return tPtTri3d(aPCyl.y(),aPCyl.z(),aPCyl.x());

}

std::vector<tPtTri3d> cGenerateSurfDevOri::VPts() const
{
    std::vector<tPtTri3d> aRes(mNb.x()*mNb.y());

    for (const auto & aPix : cRect2(cPt2di(0,0),mNb))
    {
         aRes.at(NumOfPix(aPix)) = Pt3OfPix(ToR(aPix));
    }

    return aRes;
}

std::vector<cPt3di> cGenerateSurfDevOri::VFaces() const
{
    std::vector<cPt3di> aRes;
    // parse rectangle into each pixel
    for (const auto & aPix00 : cRect2(cPt2di(0,0),mNb-cPt2di(1,1)))
    {
         // split the pixel in two tri
          // const std::vector<cTriangle<int,2> > &   aVTri = SplitPixIn2<int>(HeadOrTail());
	  for (const auto & aTri : SplitPixIn2<int>(HeadOrTail()))
	  {
              cPt3di aFace;
              for (int aK=0 ; aK<3 ; aK++)
              {
                   cPt2di aPix = aPix00 + aTri.Pt(aK);
		   aFace[aK] = NumOfPix(aPix);
              }
	      aRes.push_back(aFace);
	  }
    }
    return aRes;
}

tPtTri3d  cGenerateSurfDevOri::PCenter() const
{
     return Pt3OfPix(ToR(mNb)/2.0);
}

cGenerateSurfDevOri::cGenerateSurfDevOri(const cPt2di & aNb) :
     mNb         (aNb),
     mPlaneCart  (false)
{
}

/* ******************************************************* */
/*                                                         */
/*                 cAppliGenMeshDev                        */
/*                                                         */
/* ******************************************************* */

class cAppliGenMeshDev : public cMMVII_Appli
{
     public :

        cAppliGenMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

           // --- Mandatory ----
	      std::string mNameCloudOut;
           // --- Optionnal ----
	      bool        mBinOut;
           // --- Internal ----

};

cAppliGenMeshDev::cAppliGenMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliGenMeshDev::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudOut,"Name of output cloud/mesh", {eTA2007::FileDirProj})
   ;
}

cCollecSpecArg2007 & cAppliGenMeshDev::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           // << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file")
           // << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
   ;
}


int  cAppliGenMeshDev::Exe()
{
   // generate synthetic mesh
   cGenerateSurfDevOri aGenSD;
   tTriangulation3D  aTri(aGenSD.VPts(),aGenSD.VFaces());
   aTri.WriteFile(mNameCloudOut,mBinOut);
   aTri.MakeTopo();

   //  devlop it
   cDevTriangu3d aDev(aTri);
   aDev.SetFaceC(aTri.IndexClosestFace(aGenSD.PCenter()));
   aDev.DoDevlpt();

   return EXIT_SUCCESS;
}




/* ******************************************************* */
/*                                                         */
/*                    cDevTriangu3d                        */
/*                                                         */
/* ******************************************************* */

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

std::vector<int>  cDevTriangu3d::Unreached(int aKFace) const
{
     const tFace & aFace = mTri.KthFace(aKFace);

     std::vector<int>  aRes;
     for (int aNumV=0 ; aNumV<3 ; aNumV++)
     {
         int aKS= aFace[aNumV];
         if (mStepReach_S.at(aKS) == NO_STEP)
	 {
            aRes.push_back(aNumV);
	 }
     }
     return aRes;
}

int cDevTriangu3d::MakeNewFace()
{
     mNumCurStep++;
     // A Face is adjacent to reached iff it contains exactly 2 reached 
     std::vector<int> aVFNeigh;  // put first in vect to avoir recursive add
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         std::vector<int> aVU=Unreached(aKF);
         if (aVU.size()==1)
	 {
            aVFNeigh.push_back(aKF);
	 }
     }
     // Now mark the faces 
     for (const auto & aKF : aVFNeigh)
         AddOneFace(aKF);

     // Mark face that containt 3 reaches vertices (may have been created
     // by previous step)
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         if (Unreached(aKF).size()==0)
            AddOneFace(aKF);
     }

     return aVFNeigh.size();
}

cDevTriangu3d::cDevTriangu3d(const tTriangulation3D & aTri) :
     mNumCurStep  (0),
     mNbFaceReached (0),
     mTri         (aTri),
     mStepReach_S (mTri.NbPts() ,NO_STEP),
     mVPtsDev     (mTri.NbPts()),
     mStepReach_F (mTri.NbFace(),NO_STEP),
     mIndexFC     (-1)
{
}

void cDevTriangu3d::SetFaceC(int aNumF)
{
    mIndexFC = aNumF;
    mFaceC   = mTri.KthFace(mIndexFC);
}

void  cDevTriangu3d::DoDevlpt ()
{
    // =======  1   DEVELOP FIRST FACE ===========
    
    //  1.1 get it : if initial face was not set, init it with IndexCenter (good for Z=F(x,y))
    if (mIndexFC==-1)
        SetFaceC(mTri.IndexCenterFace());

    // 1.2  Make face and soms marked
    AddOneFace(mIndexFC); 

    // 1.3 computes its geometry
    {
       tTri3D  aTriC = mTri.KthTri(mIndexFC);  // 3D triangle
       tIsom  anIsom =  tIsom::FromTriOut(0,aTriC).MapInverse();  // Isometry plane TriC-> plane Z=0

       for (int aK=0 ; aK<3 ;aK++)
       {
            mVPtsDev.at(mFaceC[aK]) =  Proj(anIsom.Value(aTriC.Pt(aK)));
	    // StdOut() << " Pt= " << mFaceC[aK] << " " << mTri.NbPts() << "\n";
	    // StdOut() << " Pt= " << anIsom.Value(aTriC.Pt(aK)) << "\n";
       }
       // getchar();
    }

    while (int aNbF=MakeNewFace())
    {
        StdOut() << "NnFF " << aNbF << " Step="  << mNumCurStep<< "\n";
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
   aDev.DoDevlpt();

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

tMMVII_UnikPApli Alloc_GenMeshDev(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenMeshDev(aVArgs,aSpec));
}
cSpecMMVII_Appli  TheSpecGenMeshDev
(
     "MeshDevGen",
      Alloc_GenMeshDev,
      "Generate artificial(synthetic) devlopable surface",
      {eApF::Cloud},
      {eApDT::Console},
      {eApDT::Ply},
      __FILE__
);


};
