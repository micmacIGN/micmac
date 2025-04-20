#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PointCloud.h"


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
/*                 cAppliCloudClip                 */
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
	      int         mNbMinVertexByTri;
           // --- Internal ----

};

cAppliCloudClip::cAppliCloudClip(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli      (aVArgs,aSpec),
   mBinOut           (false),
   mNbMinVertexByTri (3)
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
           << AOpt2007(mNbMinVertexByTri,"NbMinV","Number minimal of vertex to maintain a triangle",{eTA2007::HDV})
   ;
}


int  cAppliCloudClip::Exe()
{
   InitOutFromIn(mNameCloudOut,"Clip_"+mNameCloudIn);

   cTriangulation3D<tREAL8>  aTri(mNameCloudIn);
   cDataBoundedSet<tREAL8,3> * aMasq=  MMV1_Masq(aTri.BoxEngl(),DirProject()+mNameMasq);
   aTri.Filter(*aMasq,mNbMinVertexByTri);
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

/* =============================================== */
/*                                                 */
/*                 cAppli_MMVII_CloudClip          */
/*                                                 */
/* =============================================== */

/**  A basic application for clipping 3d data ,  almost all the job is done in
 * libraries so it essentially interface to command line */

class cAppli_MMVII_CloudClip : public cMMVII_Appli
{
     public :

        cAppli_MMVII_CloudClip(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // --- Mandatory ----
	std::string   mNameCloudIn;
	cBox2dr       mBoxRel;
        // --- Optionnal ----
        std::string mNameCloudOut;

};

cAppli_MMVII_CloudClip::cAppli_MMVII_CloudClip
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec),
     mBoxRel (cPt2dr(1.0,1.0))
{
}

cCollecSpecArg2007 & cAppli_MMVII_CloudClip::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
	  <<   Arg2007(mBoxRel,"Box relative of clip")
   ;
}


cCollecSpecArg2007 & cAppli_MMVII_CloudClip::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file, def=Clip_+InPut")
   ;
}

int  cAppli_MMVII_CloudClip::Exe()
{
   cPointCloud   mPC_In;
   ReadFromFile(mPC_In,mNameCloudIn);

   if (! IsInit(&mNameCloudOut))
      mNameCloudOut = "Clip_" + mNameCloudIn;


//    cBox3dr  aBox3Glob = mPC_In.Box();
   cBox2dr  aBox2Glob = mPC_In.Box2d();

   cPt2dr aP0 = aBox2Glob.FromNormaliseCoord(mBoxRel.P0());
   cPt2dr aP1 = aBox2Glob.FromNormaliseCoord(mBoxRel.P1());


   cBox2dr aBoxClip(aP0,aP1);

   cPointCloud   mPC_Out;
   mPC_In.Clip(mPC_Out,aBoxClip);
   SaveInFile(mPC_Out,mNameCloudOut);


   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_MMVII_CloudClip(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_CloudClip(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MMVII_CloudClip
(
     "CloudMMVIIClip",
      Alloc_MMVII_CloudClip,
      "Clip a MMVII-Cloud format  using a box",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);

/* =============================================== */
/*                                                 */
/*                 cAppli_MMVII_CloudClip          */
/*                                                 */
/* =============================================== */

/**  A basic application for clipping 3d data ,  almost all the job is done in
 * libraries so it essentially interface to command line */

class cAppli_MMVII_Cloud2Ply : public cMMVII_Appli
{
     public :

        cAppli_MMVII_Cloud2Ply(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // --- Mandatory ----
	std::string   mNameCloudIn;
        // --- Optionnal ----
        std::string mNameCloudOut;

};

cAppli_MMVII_Cloud2Ply::cAppli_MMVII_Cloud2Ply
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppli_MMVII_Cloud2Ply::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
   ;
}


cCollecSpecArg2007 & cAppli_MMVII_Cloud2Ply::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file, def=Clip_+InPut")
   ;
}

int  cAppli_MMVII_Cloud2Ply::Exe()
{
   if (! IsInit(&mNameCloudOut))
      mNameCloudOut = LastPrefix(mNameCloudIn) + ".ply";

   cPointCloud   mPC_In;
   ReadFromFile(mPC_In,mNameCloudIn);

   mPC_In.ToPly(mNameCloudOut,false);


   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_MMVII_Cloud2Ply(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_Cloud2Ply(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MMVII_Cloud2Ply
(
     "CloudMMVII2Ply",
      Alloc_MMVII_Cloud2Ply,
      "Generate a ply version of  MMVII-Cloud",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);
#if (0)
#endif


};
