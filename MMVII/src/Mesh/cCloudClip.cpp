#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"

namespace MMVII
{

/* =============================================== */
/*                                                 */
/*                 cAppliCloudClip                 */
/*                                                 */
/* =============================================== */

/**  A basic application for clipping 3d data ,  almost all the job is done in
 * libraries so it essentially interface to command line */

class cAppliCheckMesh : public cMMVII_Appli
{
     public :

        cAppliCheckMesh(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     // --- Mandatory ----
	std::string mNameCloudIn;

     // --- Optionnal ----
	std::string mNameCloudOut;
	bool        mBinOut;
	bool        mDoCorrect;
	bool        mDo2D;
     // --- Internal ----
};

cAppliCheckMesh::cAppliCheckMesh(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mBinOut          (false),
   mDoCorrect       (false),
   mDo2D            (false)
{
}


cCollecSpecArg2007 & cAppliCheckMesh::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileCloud})
   ;
}

cCollecSpecArg2007 & cAppliCheckMesh::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
           << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file if correction are done")
           << AOpt2007(mDo2D,"Do2DC","check also as a 2D-triangulation (orientation)",{eTA2007::HDV})
           << AOpt2007(mDoCorrect,"Correct","Do correction, Defaut: Do It Out specified")
   ;
}

int cAppliCheckMesh::Exe() 
{
   if (! IsInit(&mDoCorrect))
       mDoCorrect  = IsInit(&mNameCloudOut);

   cTriangulation3D<tREAL8>  aTri(mNameCloudIn);
   aTri.CheckAndCorrect(mDoCorrect);

   aTri.CheckOri3D();
   if (mDo2D)
      aTri.CheckOri2D();

   if (IsInit(&mNameCloudOut))
   {
       aTri.WriteFile(DirProject()+mNameCloudOut,mBinOut);
   }



   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_CheckMesh(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCheckMesh(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMeshCheck
(
     "MeshCheck",
      Alloc_CheckMesh,
      "Make some checking on a mesh, eventually correct it 4 easy defaults",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);

/* =============================================== */
/*                                                 */
/*                       cAppliCloudClip           */
/*                                                 */
/* =============================================== */

/**  A basic application for clipping 3d data ,  almost all the job is done in
 * libraries so it essentially interface to command line */

class cAppliCloudClip : public cMMVII_Appli
{
     public :

        cAppliCloudClip(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

           // --- Mandatory ----
	      std::string mNameCloudIn;
	      std::string mNameMasq;
           // --- Optionnal ----
	      std::string mNameCloudOut;
	      bool        mBinOut;
           // --- Internal ----

};

cAppliCloudClip::cAppliCloudClip(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mBinOut          (false)
{
}


cCollecSpecArg2007 & cAppliCloudClip::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileCloud})
	  <<   Arg2007(mNameMasq,"Name of 3D masq", {eTA2007::File3DRegion})

   ;
}

cCollecSpecArg2007 & cAppliCloudClip::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file")
           << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
   ;
}


int  cAppliCloudClip::Exe()
{
   InitOutFromIn(mNameCloudOut,"Clip_"+mNameCloudIn);

   cTriangulation3D<tREAL8>  aTri(mNameCloudIn);
   cDataBoundedSet<tREAL8,3> * aMasq=  MMV1_Masq(aTri.BoxEngl(),DirProject()+mNameMasq);
   aTri.Filter(*aMasq);
   aTri.WriteFile(DirProject()+mNameCloudOut,mBinOut);

   delete aMasq;
   return EXIT_SUCCESS;
}


     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_CloudClip(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCloudClip(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCloudClip
(
     "MeshCloudClip",
      Alloc_CloudClip,
      "Clip a point mesh/cloud  using a region",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);



};
