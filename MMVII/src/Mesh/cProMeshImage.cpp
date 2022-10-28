#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Triangles.h"
#include "MMVII_Image2D.h"
#include "MMVII_ZBuffer.h"
#include "MeshDev.h"

namespace MMVII
{

/**  This class make a conversion between pixel space and real space using a R^2->R^2 map,
 * frequently used wih homothety when we do sampling*/
	
	/*
template <class TypeMap>  class  cMapPixelization
{
        public :
            typedef typename TypeMap::tTypeElem   tTypeElem;
            typedef cPtxd<tTypeElem,2>            tPtR;
            typedef cPtxd<int,2>                  tPixel;  // use typedef, maybe later in 3D 

	    cMapPixelization(const TypeMap & aMap) : mMap (aMap) {}

            inline tPixel  ToPix(const tPtR & aPtR) const   {return  ToI(mMap.Value(aPtR));  }

        private :
	    TypeMap  mMap;
};
*/



/* =============================================== */
/* =============================================== */
/* =============================================== */




/* =============================================== */
/*                                                 */
/*              cZBuffer                           */
/*                                                 */
/* =============================================== */



/* =============================================== */
/*                                                 */
/*                 cMeshDev_BestIm                 */
/*                                                 */
/* =============================================== */

const std::string  MeshDev_NameTriResol = "TabBestResol.xml";


void AddData(const cAuxAr2007 & anAux,cMeshDev_BestIm& aRMS)
{
    AddData(cAuxAr2007("Name",anAux),aRMS.mNames);
    AddData(cAuxAr2007("NumsIm",anAux),aRMS.mNumBestIm);
    AddData(cAuxAr2007("Resol",anAux),aRMS.mBestResol);
}


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
        std::string  NameResult(const std::string & aNameIm) const;

	void MakeDevlptIm(cZBuffer &  aZB);
        void ProcessNoPix(cZBuffer &  aZB);
        void MergeResults();

     // --- Mandatory ----
	std::string mNameCloud3DIn;
	std::string mNameIm;
	std::string mNameOri;


     // --- Optionnal ----
	std::string mNameCloud2DIn;
	double      mResolZBuf;
	int         mNbPixImRedr;

     // --- constructed ---
        cPhotogrammetricProject   mPhProj;
        cTriangulation3D<tREAL8>* mTri3D;
        cSensorCamPC *            mCamPC;
	std::string               mDirMeshDev;
	std::string               mNameResult;
        bool                      mDoImages;
        bool                      mSKE;
};

cCollecSpecArg2007 & cAppliProMeshImage::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return anArgObl
	  <<   Arg2007(mNameIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
	  <<   Arg2007(mNameCloud3DIn,"Name of input cloud/mesh", {eTA2007::FileCloud,eTA2007::Input})
	  <<   mPhProj.OriInMand()

   ;
}

cAppliProMeshImage::cAppliProMeshImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mResolZBuf       (3.0),
   mNbPixImRedr     (2000),
   mPhProj          (*this),
   mTri3D           (nullptr),
   mCamPC           (nullptr),
   mDoImages        (false),
   mSKE             (true)
{
}


cCollecSpecArg2007 & cAppliProMeshImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameCloud2DIn,"M2","Mesh 2D, dev of cloud 3D,to generate a visu of hiden part ",{eTA2007::FileCloud,eTA2007::Input})
           << AOpt2007(mResolZBuf,"ResZBuf","Resolution of ZBuffer", {eTA2007::HDV})
	  <<  AOpt2007(mDoImages,"DoIm","Do images", {eTA2007::HDV})
           << AOpt2007(mNbPixImRedr,"NbPixIR","Resolution of ZBuffer", {eTA2007::HDV})
           << AOpt2007(mSKE,CurOP_SkipWhenExist,"Skip command when result exist")
   ;

}

void cAppliProMeshImage::MakeDevlptIm(cZBuffer &  aZB )
{
   
   cTriangulation2D<tREAL8> aTri2D (cTriangulation3D<tREAL8>(DirProject()+mNameCloud2DIn));
   // mTri2D = new cTriangulation3D<tREAL8>(DirProject()+mNameCloud2DIn);

   MMVII_INTERNAL_ASSERT_tiny(aTri2D.NbFace()==mTri3D->NbFace(),"Incompat tri 2/3");
   MMVII_INTERNAL_ASSERT_tiny(aTri2D.NbPts ()==mTri3D->NbPts (),"Incompat tri 2/3");


   cBox2dr   aBox2 = aTri2D.BoxEngl(1e-3);
   // cBox2dr   aBox2(Proj(aBox3.P0()),Proj(aBox3.P1()));

   double aScale = mNbPixImRedr / double(NormInf(aBox2.Sz()));

   cHomot2D<tREAL8> mHTri2Pix = cHomot2D<tREAL8>(cPt2dr(2,2) - aBox2.P0()*aScale, aScale);

   cPt2di aSz = Pt_round_up(mHTri2Pix.Value(aBox2.P1())) + cPt2di(0,0);



   cRGBImage  aIm(aSz,cRGBImage::Yellow);

   for (size_t aKF=0 ; aKF<aTri2D.NbFace() ; aKF++)
   {
       const cResModeSurfD&   aRD = aZB.ResSurfD(aKF) ;
       cPt3di aCoul(0,0,0);

       if (aRD.mResult == eZBufRes::BadOriented)    aCoul = cRGBImage::Green;
       if (aRD.mResult == eZBufRes::Hidden)         aCoul = cRGBImage::Red;
       if (aRD.mResult == eZBufRes::OutIn)          aCoul = cRGBImage::Cyan;
       if (aRD.mResult == eZBufRes::NoPix)          aCoul = cRGBImage::Blue;
       if (aRD.mResult == eZBufRes::LikelyVisible)  aCoul = cRGBImage::Magenta;

       if ((aRD.mResult == eZBufRes::Visible) || (aRD.mResult == eZBufRes::LikelyVisible))
       {
           int aGray = round_ni(255 *  aRD.mResol/ aZB.MaxRSD());
	   // aGray = 128;
           aCoul = cPt3di(aGray,aGray,aGray);

       }
       cTri2dR  aTriPix = aTri2D.KthTri(aKF);
       cTriangle2DCompiled<tREAL8>  aTriComp(ImageOfTri(aTriPix,mHTri2Pix));

       std::vector<cPt2di> aVPix;
       aTriComp.PixelsInside(aVPix,1e-8);

       for (const auto aPix : aVPix)
           aIm.SetRGBPix(aPix,aCoul);
   }

   aIm.ToFile(mDirMeshDev+"Dev-"+LastPrefix(mNameIm)+".tif");
}

void cAppliProMeshImage::ProcessNoPix(cZBuffer &  aZB)
{
    mTri3D->MakeTopo();
    const cGraphDual &  aGrD = mTri3D->DualGr() ;

    bool  GoOn = true;
    while (GoOn)
    {
        GoOn = false;
        for (size_t aKF1 = 0 ; aKF1<mTri3D->NbFace() ; aKF1++)
        {
            if (aZB.ResSurfD(aKF1).mResult == eZBufRes::NoPix)
            {
                std::vector<int> aVF2;
                aGrD.GetFacesNeighOfFace(aVF2,aKF1);
                for (const auto &  aKF2 : aVF2)
		{
                     if (     (aZB.ResSurfD(aKF2).mResult == eZBufRes::Visible)
                          ||  (aZB.ResSurfD(aKF2).mResult == eZBufRes::LikelyVisible)
			)
		     {
                        aZB.ResSurfD(aKF1).mResult =  eZBufRes::LikelyVisible;
                        GoOn = true;
		     }
		}
            }
        }
    }
}

std::string  cAppliProMeshImage::NameResult(const std::string & aNameIm) const
{
   return  mDirMeshDev +  "TabResolTri-" + aNameIm  + ".dmp";
}

void cAppliProMeshImage:: MergeResults()
{
    mTri3D = new cTriangulation3D<tREAL8>(DirProject()+mNameCloud3DIn);
    size_t aNbF = mTri3D->NbFace();
    cMeshDev_BestIm aMBI;
    aMBI.mNames = VectMainSet(0);
    aMBI.mNumBestIm.resize(aNbF,-1);
    aMBI.mBestResol.resize(aNbF,-1);

    for (size_t aKIm=0 ; aKIm<aMBI.mNames.size(); aKIm++)
    {
        std::string aName = aMBI.mNames.at(aKIm);

        std::vector<cResModeSurfD> aVRMS;
	ReadFromFile(aVRMS,NameResult(aName));
        MMVII_INTERNAL_ASSERT_tiny(aVRMS.size()==aNbF,"Incompat tri 3 , TabResolTri");

	for (size_t aKF=0 ; aKF<aNbF ; aKF++)
	{
           const cResModeSurfD & aRMS = aVRMS.at(aKF);
           eZBufRes aRes = aRMS.mResult;

           if (      ((aRes == eZBufRes::Visible) || (aRes == eZBufRes::LikelyVisible))
                 &&  (aRMS.mResol > aMBI.mBestResol.at(aKF))
	      )
	   {
              aMBI.mBestResol.at(aKF) = aRMS.mResol;
              aMBI.mNumBestIm.at(aKF) = aKIm;
	   }
	}
    }
    delete mTri3D;

    SaveInFile(aMBI,mDirMeshDev+MeshDev_NameTriResol);
}

int cAppliProMeshImage::Exe() 
{

   mDirMeshDev = DirProject()+ MMVIIDirMeshDev;
   if (LevelCall()==0)
   {
      CreateDirectories(mDirMeshDev,true);
   } 
   mPhProj.FinishInit();

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
      int aResult =  ResultMultiSet();

      MergeResults();
	

      return aResult;
   }

   // By default  SKE true with multiple file (i.e. we are recalled) but false with single file (why else run it)
   SetIfNotInit(mSKE,LevelCall()!=0);
   mNameResult = NameResult(mNameIm);

   if (mSKE && ExistFile(mNameResult))
      return EXIT_SUCCESS;

   mTri3D = new cTriangulation3D<tREAL8>(DirProject()+mNameCloud3DIn);

   cMeshTri3DIterator  aTriIt(mTri3D);

   mCamPC = mPhProj.AllocCamPC(mNameIm,true);
   cSIMap_Ground2ImageAndProf aMapCamDepth(mCamPC);

   cSetVisibility aSetVis(mCamPC);

   double Infty =1e20;
   cPt2di aSzPix = mCamPC->SzPix();
   cBox3dr  aBox(cPt3dr(0,0,-Infty),cPt3dr(aSzPix.x(),aSzPix.y(),Infty));
   cDataBoundedSet<tREAL8,3>  aSetCam(aBox);


   cZBuffer aZBuf(aTriIt,aSetVis,aMapCamDepth,aSetCam,mResolZBuf);
   aZBuf.MakeZBuf(eZBufModeIter::ProjInit);
   aZBuf.MakeZBuf(eZBufModeIter::SurfDevlpt);

   ProcessNoPix(aZBuf);


   // if (mDoImage)
      //aZBuf.ZBufIm().DIm().ToFile(mDirMeshDev+ "ZBuf-"+ mNameIm);
   // aIm.ToFile(mDirMeshDev+"Dev-"+LastPrefix(mNameIm)+".tif");
   if (mDoImages)
      aZBuf.ZBufIm().DIm().ToFile(mDirMeshDev+"ZBuf-"+LastPrefix(mNameIm)+".tif");


   if (IsInit(&mNameCloud2DIn))
   {
      MakeDevlptIm(aZBuf);
   }

   SaveInFile(aZBuf.VecResSurfD(),mNameResult);

   delete mTri3D;
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
