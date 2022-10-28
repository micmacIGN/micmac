#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Triangles.h"
#include "MMVII_Image2D.h"
#include "MMVII_ZBuffer.h"
#include "MeshDev.h"
#include "MMVII_Sys.h"

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
    AddData(cAuxAr2007("AvgResol",anAux),aRMS.mAvgResol);
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

        cAppliBenchAnswer BenchAnswer() const override;  ///< this command is used in the bench
	int  ExecuteBench(cParamExeBench &) override ;   ///< indicate what is done in the bench


	/// name to save the result of an image
        std::string  NameResult(const std::string & aNameIm) const;

	/** Process the triangle that correspond to no pixel => make them almost visble if close to 
	    a triangle which is visible */
        void ProcessNoPix(cZBuffer &  aZB);
	/** After all images have been processed , merge their result to extract best image*/
        void MergeResults();

	/** Make a  developped  and visualisation to check labelisartion*/
	void MakeDevlptIm(cZBuffer &  aZB);

     // --- Mandatory ----
	std::string mNamePatternIm;
	std::string mNameSingleIm;
	std::string mNameCloud3DIn;


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
	  <<   Arg2007(mNamePatternIm,"Name of image", {{eTA2007::MPatFile,"0"},eTA2007::FileDirProj} )
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
	   << AOptBench()
   ;

}

cAppliBenchAnswer cAppliProMeshImage::BenchAnswer() const
{
   return cAppliBenchAnswer(true,0.1);
}



int  cAppliProMeshImage::ExecuteBench(cParamExeBench & aParam) 
{
   // no randomization, so once is enough
   if (aParam.Level() != 0)
      return EXIT_SUCCESS;

   std::string aCom =
	              Bin2007 + BLANK
		  +   mSpecs.Name() + BLANK
		  +  mInputDirTestMMVII + "Ply/FileTestMesh.xml Clip_C3DC_QuickMac_poisson_depth5.ply TestProjMesh BenchMode=1";

   // MMVII  0_MeshProjImage ../MMVII-TestDir/Input/Ply/FileTestMesh.xml Clip_C3DC_QuickMac_poisson_depth5.ply TestProjMesh BenchMode=1
   GlobSysCall(aCom);

    return EXIT_SUCCESS;
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

   aIm.ToFile(mDirMeshDev+"Dev-"+LastPrefix(mNameSingleIm)+".tif");
}

void cAppliProMeshImage::ProcessNoPix(cZBuffer &  aZB)
{
    mTri3D->MakeTopo();
    const cGraphDual &  aGrD = mTri3D->DualGr() ;

    bool  GoOn = true;
    while (GoOn)  // continue as long as we made a modification
    {
        GoOn = false;
        for (size_t aKF1 = 0 ; aKF1<mTri3D->NbFace() ; aKF1++)
        {
            // check if a NoPix face has a neigboor visible or likely
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
   std::string aName =  "TabResolTri-" + aNameIm  + ".dmp";
   aName = (mIsInBenchMode ? MMVII_PrefRefBench  : mPhProj.OriIn())  + aName;

   //std::cout << "NNNNNNNNNNN " << aName  << " [" << <<"]\n";

   return  mDirMeshDev   +  aName;
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


    // double aSumSurf = 0.0;
    // double aSomWResol = 0.0;
    cWeightAv<tREAL8>  aWAvg;
    for (size_t aKF=0 ; aKF<aNbF ; aKF++)
    {
        aWAvg.Add(mTri3D->KthTri(aKF).Area(),aMBI.mBestResol.at(aKF) );
    }
    aMBI.mAvgResol= aWAvg.Average();

    delete mTri3D;

    if (mIsInBenchMode)
    {
       cMeshDev_BestIm aRef;
       ReadFromFile(aRef,mDirMeshDev+MMVII_PrefRefBench+MeshDev_NameTriResol);

       MMVII_INTERNAL_ASSERT_bench(aRef.mNumBestIm.size()==aMBI.mNumBestIm.size(),"size dif in bench projmesh");

       for (size_t aK=0 ; aK<aMBI.mNumBestIm.size() ; aK++)
       {
           MMVII_INTERNAL_ASSERT_bench(aRef.mNumBestIm.at(aK)==aMBI.mNumBestIm.at(aK),"best-im dif in bench projmesh");
       }
       MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aRef.mAvgResol,aMBI.mAvgResol)<1e-5,"relative avg in bench projmesh");
    }
    else
    {
        SaveInFile(aMBI,mDirMeshDev+mPhProj.OriIn()+MeshDev_NameTriResol);
    }
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

   // if there is a single file in a xml set of file, the subst has not been made ...
   mNameSingleIm = FileOfPath(VectMainSet(0).at(0),false);
   // By default  SKE true with multiple file (i.e. we are recalled) but false with single file (why else run it)
   SetIfNotInit(mSKE,LevelCall()!=0);
   mNameResult = NameResult(mNameSingleIm);

   if (  (!mIsInBenchMode) && (mSKE && ExistFile(mNameResult)) )
      return EXIT_SUCCESS;

   mTri3D = new cTriangulation3D<tREAL8>(DirProject()+mNameCloud3DIn);

   cMeshTri3DIterator  aTriIt(mTri3D);

   mCamPC = mPhProj.AllocCamPC(mNameSingleIm,true);
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


   if (mDoImages)
      aZBuf.ZBufIm().DIm().ToFile(mDirMeshDev+"ZBuf-"+LastPrefix(mNameSingleIm)+".tif");


   if (IsInit(&mNameCloud2DIn))
   {
      MakeDevlptIm(aZBuf);
   }

   if (mIsInBenchMode)
   {
      std::vector<cResModeSurfD> aRef;
      ReadFromFile(aRef,mNameResult);
      const std::vector<cResModeSurfD> & aVRSD = aZBuf.VecResSurfD();

      MMVII_INTERNAL_ASSERT_bench(aRef.size()==aVRSD.size(),"size dif in bench projmesh");

      for (size_t aK=0 ; aK<aVRSD.size() ; aK++)
      {
         MMVII_INTERNAL_ASSERT_bench(aRef.at(aK).mResult ==aVRSD.at(aK).mResult,"result dif in bench projmesh");
         MMVII_INTERNAL_ASSERT_bench(RelativeDifference(aRef.at(aK).mResol,aVRSD.at(aK).mResol)<1e-5,"resol dif in bench projmesh");
      }
   }
   else
   {
       SaveInFile(aZBuf.VecResSurfD(),mNameResult);
   }

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
      "(internal) Project a mes on an image to prepare devlopment",
      {eApF::Cloud},
      {eApDT::Ply,eApDT::Orient},
      {eApDT::FileSys},
      __FILE__
);

#if (0)
#endif

}
