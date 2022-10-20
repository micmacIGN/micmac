#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"

namespace MMVII
{

/**   This abstract class is used to decribe an object containing many triangles.
 * In Z-Buffer, we can use an explicit mesh, but also an implicit one if we parse an image
 * where each pixel is made from two triangle. This implicit class allow to maipulate the two
 * object in the same interface (an avoid converting the pixel in million of triangles ...)
 */

	/*
class  cTri3DIterator
{
     public :
        virtual bool GetNextTri(cTri3dR &) = 0;
        virtual bool GetNextPoint(cPt3dr &) = 0;
        virtual void Reset()  = 0;
     private :
};

class  cZBuffer
{
      public :
          typedef cDataInvertibleMapping<tREAL8,3>  tMap;

          cZBuffer(cTri3DIterator & aMesh,const tMap & aMap,const cBox2dr * aBoxOutClip);
      private :
          cZBuffer(const cZBuffer & ) = delete;



	  cTri3DIterator & mMesh;
	  const tMap &     mMapI2O;
          const cBox2dr*   mBoxOutClip;
          cBox3dr          mBoxIn;  ///< Box in input space, not sure usefull, but ....
          cBox3dr          mBoxOut; ///< Box in output space, usefull for xy, not sure for z , but ...
};

cZBuffer::cZBuffer(cTri3DIterator & aMesh,const tMap & aMapI2O,const cBox2dr * aBoxOutClip) :
    mMesh     (aMesh),
    mMapI2O   (aMapI2O),

    mBoxIn    (cBox3dr::Empty()),
    mBoxOut   (cBox3dr::Empty())
{
    cTplBoxOfPts<tReal8,3> aBoxIn;
    cTplBoxOfPts<tReal8,3> aBoxOut;

    //  compute the box in put and output space
    cPt3dr aPIn;
    while (mMesh.GetNextPoint(aPIn))
    {
        cPt3dr aPOut = mMapI2O.Value(aPIn);

	if (1)
	{
           aBoxIn.Add(aPIn);
           aBoxOut.Add(aPut);
	}
    }

}
*/



/* =============================================== */
/*                                                 */
/*                 cAppliCloudClip                 */
/*                                                 */
/* =============================================== */

/** Application for projecting a mesh on image, for each triangle,
 * indicate if it is visible and what is its quality=resolution in lowest dim
*/

class cAppliProMeshImage : public cMMVII_Appli
{
     public :

        cAppliProMeshImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     // --- Mandatory ----
	std::string mNameCloud3DIn;
	std::string mNameIm;
	std::string mNameOri;


     // --- Optionnal ----
	std::string mNameCloud2DIn;

     // --- constructed ---
        cPhotogrammetricProject   mPhProj;
        // cTriangulation3D<tREAL8>  mTri3D;
        cSensorCamPC *            mCamPC;
};

cCollecSpecArg2007 & cAppliProMeshImage::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return anArgObl
	  <<   Arg2007(mNameCloud3DIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileCloud,eTA2007::Input})
	  <<   Arg2007(mNameIm,"Name of image", {eTA2007::FileImage,eTA2007::OptionalExist})
	  <<   mPhProj.OriInMand()

   ;
}


cAppliProMeshImage::cAppliProMeshImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mCamPC           (nullptr)
{
}


cCollecSpecArg2007 & cAppliProMeshImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameCloud2DIn,"M2","Mesh 2D, dev of cloud 3D, to generate a visu of hiden part ", {eTA2007::FileCloud,eTA2007::Input})
   ;

}


int cAppliProMeshImage::Exe() 
{
   mPhProj.FinishInit();

   mCamPC = mPhProj.AllocCamPC(mNameIm,true);

   StdOut() << "FOCALE "  << mCamPC->InternalCalib()->F() << "\n";


   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_ProMeshImage(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliProMeshImage(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecProMeshImage
(
     "0_MeshProjImage",
      Alloc_ProMeshImage,
      "(internal) Project a mes on an image",
      {eApF::Cloud},
      {eApDT::Ply,eApDT::Orient},
      {eApDT::FileSys},
      __FILE__
);

#if (0)
#endif

}
