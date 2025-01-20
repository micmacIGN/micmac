#include "MMVII_Radiom.h"

#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Triangles.h"
#include "MMVII_Image2D.h"
#include "MeshDev.h"
#include "MMVII_Sys.h"
#include "MMVII_Triangles.h"


namespace MMVII
{


class cAppliMeshImageDevlp : public cMMVII_Appli
{
     public :

        cAppliMeshImageDevlp(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


	void DoAnIm(size_t aKIm);

        // cAppliBenchAnswer BenchAnswer() const override;  ///< this command is used in the bench
	// int  ExecuteBench(cParamExeBench &) override ;   ///< indicate what is done in the bench


        std::string NameFileMeshDev(const std::string& aNameFile)
	{
            return  mPhProj.DPMeshDev().FullDirIn() + aNameFile;
	}

     // --- Mandatory ----
	std::string mNameCloud2DIn;

     // --- Optionnal ----
	bool                           mMiror; 
	bool                           mWGrayLab; 
	bool                           mWRGBLab; 
	/*
	double      mResolZBuf;
	int         mNbPixImRedr;
        bool        mDoImages;
        bool        mSKE;
        bool        mNameBenchMode;
	*/

     // --- constructed ---
        cPhotogrammetricProject     mPhProj;
	cTriangulation2D<tREAL8>*   mTriDev;
	cTriangulation3D<tREAL8>*   mTri3;
	cBox2dr                     mBoxDev;

        cHomot2D<tREAL8>               mHDev2Pix ;
	cPt2di                         mSzPix;

        cMeshDev_BestIm                mMDBI;
	size_t                         mNbIm;
	size_t                         mNbF;
	size_t                         mNbPts;
	std::vector<std::list<size_t>> mVListIndTri;  

	std::vector<cPt2dr>            mPtsCurIm; ///< projection of mesh vertex in image
	std::vector<cPt2dr>            mPtsGlob;
	cSetIntDyn                     mSetPCurIm;
	cRGBImage                      mGlobIm;     ///<  Devloped  Image in RAM, has full size
	cIm2D<tINT2>                   mGrLabIm;
	cRGBImage                      mRGBLabIm;

	cIm2D<tINT1>                   mNormX;  ///< Image of X-normal, stored on 8-byte 
	cIm2D<tINT1>                   mNormY;  ///< Image of X-normal, stored on 8-byte
	tREAL8                         mNormMult; ///< Multiplier of normal					
        bool                           mWithNorm; ///< Do we create an image of normal

	std::vector<cPt3di>            mLutLabel;
	std::string                    mNameDev;

};


cAppliMeshImageDevlp::cAppliMeshImageDevlp(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
    // opt args
   mMiror           (true),
   mWGrayLab        (false),
   mWRGBLab         (false),
    // internal vars
   mPhProj          (*this),
   mTriDev          (nullptr),
   mBoxDev          (cBox2dr::Empty()),
   mSetPCurIm       (0),
   mGlobIm          (cPt2di(1,1)),  ///< init with minmal size
   mGrLabIm         (cPt2di(1,1)),
   mRGBLabIm        (cPt2di(1,1)),
   mNormX           (cPt2di(1,1)),
   mNormY           (cPt2di(1,1)),
   mNormMult        (100.0),  // defautlt value fit [-1,1] in [-128,127]
   mWithNorm        (false),
   mNameDev         ("DevIm.tif")
{
}



cCollecSpecArg2007 & cAppliMeshImageDevlp::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mWGrayLab,"WGL","With Gray Label 2-byte image generation",{eTA2007::HDV})
           << AOpt2007(mWRGBLab,"WRGBL","With RGB Label  1 chanel-label/2 label contrast",{eTA2007::HDV})
	   << mPhProj.DPRadiomModel().ArgDirInOpt()
           << AOpt2007(mWithNorm,"WithNormal","Do we want to create images of  normal ?",{eTA2007::HDV})
   ;

}

cCollecSpecArg2007 & cAppliMeshImageDevlp::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return anArgObl
	  <<   Arg2007(mNameCloud2DIn,"Name of 2d devloped mesh",{eTA2007::FileDirProj} )
	  <<   mPhProj.DPMeshDev().ArgDirInMand()

   ;
}

void cAppliMeshImageDevlp::DoAnIm(size_t aKIm)
{
   cCalibRadiomIma * aRadIma = nullptr;
   if (mVListIndTri.at(aKIm).empty())
   {
      return;
   }

   std::string aNameIm = mMDBI.mNames[aKIm];
   if (mPhProj.DPRadiomModel().DirInIsInit())
   {
       aRadIma = mPhProj.ReadCalibRadiomIma(aNameIm);
   }

   cDataFileIm2D aDFIIm = cDataFileIm2D::Create(aNameIm,eForceGray::No);

   cSensorCamPC * aCamPC = mPhProj.ReadCamPC(aNameIm,false);

   // compute Ind of pts used in this image
   for (const auto & aIndF :   mVListIndTri.at(aKIm))
   {
       const cPt3di  & aFace =  mTriDev->KthFace(aIndF);
       for (int aK3=0 ; aK3<3 ; aK3++)
           mSetPCurIm.AddIndFixe(aFace[aK3]);
   }
   // conpute  coord of pts in ortho and in cur Im
   cTplBoxOfPts<tREAL8,2> aBoxDev;  /// store bounding box of devlopped point
   cTplBoxOfPts<tREAL8,2> aBoxIm;  /// store box of vertex in image
   for (const auto & anIndPts : mSetPCurIm.mVIndOcc)
   {
       cPt2dr aPGlob = mPtsGlob.at(anIndPts);
       aBoxDev.Add(aPGlob);

       cPt2dr aPIm = aCamPC->Ground2Image(mTri3->KthPts(anIndPts));
       aBoxIm.Add(aPIm);    
       mPtsCurIm.at(anIndPts) = aPIm; // memorize coordinate of vertex in image
   }

   // Create box of pix from box of Tri Proj : dilate (for margin),  cast to int, intersect with box of image
   cBox2di  aBoxPixIm =  aBoxIm.CurBox().Dilate(4.0).ToI();
   aBoxPixIm = aBoxPixIm.Inter(aDFIIm);
   cPt2dr aP0Im = ToR(aBoxPixIm.P0()); // used for radial correc


   cRGBImage  aImCur = cRGBImage::FromFile(aNameIm,aBoxPixIm);

   StdOut() << " KKIm" << aKIm
	    << " " << aNameIm
	    << " " << aCamPC->InternalCalib()->F()
	    << " " << aBoxDev.CurBox().Sz()
	    << " " << aBoxIm.CurBox().Sz()
	    << " " << aBoxPixIm
	    << "\n"; 

   // === cPt3di aRGBLab(aKIm,aKIm,aKIm);

   for (const auto & aIndF :   mVListIndTri.at(aKIm))
   {
	 auto aTriangle3D = mTri3->KthTri(aIndF);
         cPt3dr aNormal = NormalUnit(aTriangle3D);
        const cPt3di  & aFace =  mTriDev->KthFace(aIndF);

        cTriangle2DCompiled<tREAL8> aTriGlob  = TriFromFace(mPtsGlob,aFace);
        tTri2dr aTriCurIm = TriFromFace(mPtsCurIm,aFace);

        cAffin2D<tREAL8> aAffG2I = cAffin2D<tREAL8>::Tri2Tri(aTriGlob,aTriCurIm);
	aAffG2I = cAffin2D<tREAL8>::Translation(-ToR(aBoxPixIm.P0())) * aAffG2I;

//StdOut() << "FFF " << aAffG2I.Tr() << aAffG2I.VX() << aAffG2I.VY() << std::endl;

	std::vector<cPt2di>  aVPixGlob;
	aTriGlob.PixelsInside(aVPixGlob,1e-8);

	for (const auto & aPixG : aVPixGlob)
	{
            cPt2dr aPtIm = aAffG2I.Value(ToR(aPixG));
	    cPt3di aRGB = aImCur.GetRGBPixBL(aPtIm);

	    if (aRadIma)
	    {
                cPt2dr aPtGlobIm = aPtIm+ aP0Im;
		aRGB = ToI( aRadIma->ImageCorrec(ToR(aRGB),aPtGlobIm)  );
	    }

            mGlobIm.SetRGBPix(aPixG,aRGB);
	    if (mWGrayLab)
               mGrLabIm.DIm().SetV(aPixG,aKIm);
	    if (mWRGBLab)
               mRGBLabIm.SetRGBPix(aPixG,mLutLabel.at(aKIm));

	    if (mWithNorm)
	    {
               mNormX.DIm().SetV(aPixG,round_ni(aNormal.x() * mNormMult));
               mNormY.DIm().SetV(aPixG,round_ni(aNormal.y() * mNormMult));
	    }
	}
   }

   delete aCamPC;
   delete aRadIma;
   mSetPCurIm.Clear();
}


int cAppliMeshImageDevlp::Exe() 
{
     //  Init mMDBI & mPhProj
     mPhProj.FinishInit();
     ReadFromFile(mMDBI,mPhProj.DPMeshDev().FullDirIn()+MeshDev_NameTriResol);
     mPhProj.DPOrient().SetDirIn(mMDBI.mNameOri);
     mPhProj.FinishInit();

     if (mPhProj.DPRadiomModel().DirInIsInit())
     {
         mNameDev = Prefix(mNameDev) + "_" +  mPhProj.DPRadiomModel().DirIn() + ".tif";
     }

     StdOut() << "JJJJJ " << NameFileMeshDev(mNameDev)  << std::endl;


     //  Read triangulations
     mTriDev = new cTriangulation2D<tREAL8>(DirProject() + FileOfPath(mNameCloud2DIn ));
     if (mMiror)
     {
         for (size_t aKP=0 ; aKP<mTriDev->NbPts() ; aKP++)
              mTriDev->KthPts(aKP).y() *= -1;
     }
     mNbF = mTriDev->NbFace();
     mNbPts = mTriDev->NbPts();
     mBoxDev = mTriDev->BoxEngl();
     mTri3 = new cTriangulation3D<tREAL8>(DirProject() +  mMDBI.mNamePly);

     MMVII_INTERNAL_ASSERT_tiny(mTri3->NbFace()==mNbF,"Incoherent NbFace 2d/3d");
     MMVII_INTERNAL_ASSERT_tiny(mTri3->NbPts()==mNbPts,"Incoherent NbPts 2d/3d");

     //  Initialize mVListIndTri
     mNbIm = mMDBI.mNames.size();
     mLutLabel = cRGBImage::LutVisuLabRand(mNbIm+1);
     mVListIndTri.resize(mNbIm);
     for (size_t aKTri=0 ; aKTri<mMDBI.mNumBestIm.size() ; aKTri++)
     {
         int aIndBI = mMDBI.mNumBestIm.at(aKTri);
         if (aIndBI>=0)
            mVListIndTri.at(aIndBI).push_back(aKTri);
     }

     // Initialize Mapping Tri2 / Pixel
     tREAL8 aScale = mMDBI.mAvgResol;
     cPt2dr aBrd(2,2);
     mHDev2Pix = cHomot2D<tREAL8>(aBrd - mBoxDev.P0()*aScale, aScale);

     mPtsGlob.resize(mNbPts);
     mPtsCurIm.resize(mNbPts);
     for (size_t aKp=0 ; aKp<mNbPts ; aKp++)
     {
         mPtsGlob.at(aKp) = mHDev2Pix.Value(mTriDev->KthPts(aKp));
     }
     mSetPCurIm = cSetIntDyn(mNbPts);

     // Create file if required
     mSzPix = Pt_round_up(aBrd+mHDev2Pix.Value(mBoxDev.P1()));

     mGlobIm = cRGBImage(mSzPix);
     if (mWGrayLab)
     {
        mGrLabIm.DIm().Resize(mSzPix);
        mGrLabIm.DIm().InitCste(-1);
     }
     if (mWRGBLab)
        mRGBLabIm = cRGBImage(mSzPix,mLutLabel.at(mNbIm)) ;

     if (mWithNorm)  // if need them, give adequate size (else minimal size required)
     {
         mNormX.DIm().Resize(mSzPix);
         mNormY.DIm().Resize(mSzPix);
     }

     for (size_t aKIm = 0 ; aKIm < mNbIm ; aKIm++)
     {
          DoAnIm(aKIm);
     }

     mGlobIm.ToFile(NameFileMeshDev(mNameDev));
     if (mWGrayLab)
        mGrLabIm.DIm().ToFile(NameFileMeshDev("LabGRIm.tif"));
     if (mWRGBLab)
        mRGBLabIm.ToFile(NameFileMeshDev("LabRGBIm.tif"));

     if (mWGrayLab || mWRGBLab)
        SaveInFile(mMDBI.mNames,NameFileMeshDev("LabNames.xml"));

     if (mWithNorm)
     {
         mNormX.DIm().ToFile(NameFileMeshDev("XNorm.tif"));
         mNormY.DIm().ToFile(NameFileMeshDev("YNorm.tif"));
     }

     delete mTriDev;
     delete mTri3;
     return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */


tMMVII_UnikPApli Alloc_MeshImageDevlp(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliMeshImageDevlp(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecMeshImageDevlp
(
     "MeshImageDevlp",
      Alloc_MeshImageDevlp,
      " Compute devlopped images from 3d-mesh, 2d-dev-mesh and ori",
      {eApF::ImProc},
      {eApDT::Ply,eApDT::Orient,eApDT::Image},
      {eApDT::Image},
      __FILE__
);

}
