#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
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

/**   This abstract class is used to decribe an object containing many triangles.
 *
 * In Z-Buffer, we can use an explicit mesh, but also an implicit one if we parse an image
 * where each pixel is made from two triangle. This implicit class allow to maipulate the two
 * object in the same interface (an avoid converting the pixels in hundred million of triangles ...)
 */

class cCountTri3DIterator ;
class  cTri3DIterator
{
     public :
        virtual bool GetNextTri(cTri3dR &) = 0;
        virtual bool GetNextPoint(cPt3dr &) = 0;
        virtual void ResetTri()  = 0;
        virtual void ResetPts()  = 0;

        virtual cCountTri3DIterator * CastCount();

        void ResetAll() ;
     private :
};

/** in many case, the implementation can be done by counters */

class cCountTri3DIterator : public cTri3DIterator
{
     public :
        cCountTri3DIterator(size_t aNbP,size_t aNbF);

	virtual cPt3dr  KthP(int aKP) const = 0;
	virtual cTri3dR KthF(int aKF) const = 0;

        bool GetNextTri(cTri3dR &) override;
        bool GetNextPoint(cPt3dr &) override;
        void ResetTri()  override;
        void ResetPts()  override;
        cCountTri3DIterator * CastCount() override;

     private :
	size_t  mNbP;
	size_t  mNbF;
	size_t  mIndexF;
	size_t  mIndexP;
};

class cMeshTri3DIterator : public cCountTri3DIterator
{
     public :
        cMeshTri3DIterator(cTriangulation3D<tREAL8> *);

	cPt3dr  KthP(int aKP) const override;
	cTri3dR KthF(int aKF) const override;
     private :
	cTriangulation3D<tREAL8> *  mTri;
};



/* =============================================== */
/*                                                 */
/*                 cTri3DIterator                  */
/*                                                 */
/* =============================================== */

void cTri3DIterator::ResetAll()
{
    ResetTri();
    ResetPts();
}

cCountTri3DIterator * cTri3DIterator::CastCount() {return nullptr;}

/* =============================================== */
/*                                                 */
/*            cCountTri3DIterator                  */
/*                                                 */
/* =============================================== */

cCountTri3DIterator::cCountTri3DIterator(size_t aNbP,size_t aNbF) :
    mNbP  (aNbP),
    mNbF  (aNbF)
{
   ResetPts();
   ResetTri();
}

void cCountTri3DIterator::ResetTri() { mIndexF=0;}
void cCountTri3DIterator::ResetPts() { mIndexP=0;}

bool cCountTri3DIterator::GetNextPoint(cPt3dr & aP )
{
    if (mIndexP>=mNbP) return false;
    aP = KthP(mIndexP);
    mIndexP++;
    return true;
}

bool cCountTri3DIterator::GetNextTri(cTri3dR & aTri)
{
    if (mIndexF>=mNbF) return false;
    aTri = KthF(mIndexF);
    mIndexF++;
    return true;
}

cCountTri3DIterator * cCountTri3DIterator::CastCount() {return this;}




/* =============================================== */
/*                                                 */
/*              cMeshTri3DIterator                 */
/*                                                 */
/* =============================================== */

cMeshTri3DIterator::cMeshTri3DIterator(cTriangulation3D<tREAL8> * aTri) :
    cCountTri3DIterator(aTri->NbPts(),aTri->NbFace()),
    mTri (aTri)
{
}

cPt3dr  cMeshTri3DIterator::KthP(int aKP) const {return mTri->KthPts(aKP);}
cTri3dR cMeshTri3DIterator::KthF(int aKF) const {return mTri->KthTri(aKF);}


/* =============================================== */
/* =============================================== */
/* =============================================== */


enum class eZBufRes
           {
              Undefined,      ///< to have some value to return when nothing is computed
              UnRegIn,        ///< Un-Regular since input
              OutIn,          ///< Out domain  since input (current ?)
              UnRegOut,       ///< Un-Regular since output
              OutOut,         ///< Out domain  since output (never )
              BadOriented,    ///< Badly oriented
              Hidden,         ///< Hidden by other
              NoPix,          ///< When there is no pixel in triangle, decision har to do
              LikelyVisible,  ///< Probably visible -> No Pixel but connected to visible
              Visible         ///< Visible 
           };

enum class eZBufModeIter
           {
               ProjInit,
               SurfDevlpt
           };

struct cResModeSurfD
{
    public :
	eZBufRes mResult;
	double   mResol   ;
};


void  AddData(const cAuxAr2007  &anAux,cResModeSurfD& aRMS )
{
     int aResult = int(aRMS.mResult);
     AddData(cAuxAr2007("Result",anAux),aResult);
     if (anAux.Input())
        aRMS.mResult = eZBufRes(aResult);
     AddData(cAuxAr2007("Resol",anAux),aRMS.mResol);
}



class  cZBuffer
{
      public :

          typedef tREAL4                            tElem;
          typedef cDataIm2D<tElem>                  tDIm;
          typedef cIm2D<tElem>                      tIm;
	  typedef cIm2D<tU_INT1>                    tImSign;
	  typedef cDataIm2D<tU_INT1>                tDImSign;

          typedef cDataInvertibleMapping<tREAL8,3>  tMap;
          typedef cDataBoundedSet<tREAL8,3>         tSet;

	  static constexpr tElem mInfty =  -1e10;

          cZBuffer(cTri3DIterator & aMesh,const tSet & aSetIn,const tMap & aMap,const tSet & aSetOut,double aResolOut);

          const cPt2di  SzPix() ; ///< Accessor
	  tIm   ZBufIm() const; ///< Accessor
          eZBufRes MakeOneTri(const cTri3dR & aTriIn,const cTri3dR & aTriOut,eZBufModeIter aMode);


	  void MakeZBuf(eZBufModeIter aMode);
          double ComputeResol(const cTri3dR & aTriIn ,const cTri3dR & aTriOut) const;

	  cResModeSurfD&  ResSurfD(size_t) ;
	  std::vector<cResModeSurfD> & VecResSurfD() ;
	  double  MaxRSD() const;

      private :
          cZBuffer(const cZBuffer & ) = delete;


	  cPt2dr  ToPix(const cPt3dr&) const;

	  bool                  mZF_SameOri; ///< Axe of Z (in out coord) and oriented surface have same orientation
          int                   mMultZ;
	  cTri3DIterator &      mMesh;
          cCountTri3DIterator * mCountMesh;
	  const tMap &          mMapI2O;
          const tSet &          mSetIn;
          const tSet &          mSetOut;
	  double                mResolOut;

          cBox3dr          mBoxIn;     ///< Box in input space, not sure usefull, but ....
          cBox3dr          mBoxOut;    ///< Box in output space, usefull for xy, not sure for z , but ...
	  cHomot2D<tREAL8> mROut2Pix;  ///<  Mapping Out Coord -> Pix Coord
	  tIm              mZBufIm;
	  tImSign          mImSign;   ///< sign of normal  1 or -1 , 0 if uninit
          cPt2di           mSzPix;

	  double           mLastResSurfDev;
	  double           mMaxRSD;

	  std::vector<cResModeSurfD>  mResSurfD;
};

cZBuffer::cZBuffer(cTri3DIterator & aMesh,const tSet &  aSetIn,const tMap & aMapI2O,const tSet &  aSetOut,double aResolOut) :
    mZF_SameOri (false),
    mMultZ      (mZF_SameOri ? 1 : -1),
    mMesh       (aMesh),
    mCountMesh  (mMesh.CastCount()),
    mMapI2O     (aMapI2O),
    mSetIn      (aSetIn),
    mSetOut     (aSetOut),
    mResolOut   (aResolOut),

    mBoxIn      (cBox3dr::Empty()),
    mBoxOut     (cBox3dr::Empty()),
    mROut2Pix   (),
    mZBufIm     (cPt2di(1,1)),
    mImSign     (cPt2di(1,1))
{
    cTplBoxOfPts<tREAL8,3> aBoxOfPtsIn;
    cTplBoxOfPts<tREAL8,3> aBoxOfPtsOut;

    //  compute the box in put and output space
    cPt3dr aPIn;

    mMesh.ResetAll();
    int aCptTot=0;
    int aCptIn=0;
    while (mMesh.GetNextPoint(aPIn))
    {
        aCptTot++;
	if (mSetIn.InsideWithBox(aPIn))
	{
            cPt3dr aPOut = mMapI2O.Value(aPIn);

	    if (mSetOut.InsideWithBox(aPOut))
	    {
               aCptIn++;
               aBoxOfPtsIn.Add(aPIn);
               aBoxOfPtsOut.Add(aPOut);
	    }
	}
    }
    // StdOut() << " cCCCCCCCCCCC " << aCptIn  << " " << aCptTot << "\n";
    mMesh.ResetPts();

    mBoxIn = aBoxOfPtsIn.CurBox();
    mBoxOut = aBoxOfPtsOut.CurBox();

    cPt2di aBrd(2,2);
    //   aP0/aResout + aTr -> 1,1
    cPt2dr aTr = ToR(aBrd) - Proj(mBoxOut.P0()) * (1.0/mResolOut);
    mROut2Pix = cHomot2D<tREAL8>(aTr,1.0/mResolOut);

    mSzPix =  Pt_round_up(ToPix(mBoxOut.P1())) + aBrd;


    mZBufIm = tIm(mSzPix);
    mZBufIm.DIm().InitCste(mInfty);
    mImSign = tImSign(mSzPix,nullptr,eModeInitImage::eMIA_Null);
}

cPt2dr  cZBuffer::ToPix(const cPt3dr & aPt) const {return mROut2Pix.Value(Proj(aPt));}
cZBuffer::tIm  cZBuffer::ZBufIm() const {return mZBufIm;}
cResModeSurfD&  cZBuffer::ResSurfD(size_t aK)  {return mResSurfD.at(aK);}
double  cZBuffer::MaxRSD() const {return mMaxRSD;}

std::vector<cResModeSurfD> & cZBuffer::VecResSurfD() {return mResSurfD;}

void cZBuffer::MakeZBuf(eZBufModeIter aMode)
{
    if (aMode==eZBufModeIter::SurfDevlpt)
    {
	mResSurfD.clear();
        mMaxRSD = 0.0;
    }

    cTri3dR  aTriIn = cTri3dR::Tri000();
    while (mMesh.GetNextTri(aTriIn))
    {
        mLastResSurfDev = -1;
        eZBufRes aRes = eZBufRes::Undefined;
        //  not sure this us to test that, or the user to assure it give clean data ...
        if (aTriIn.Regularity() <=0)  
           aRes = eZBufRes::UnRegIn;
	else if (! mSetIn.InsideWithBox(aTriIn))
           aRes = eZBufRes::OutIn;
	else 
        {
            cTri3dR aTriOut = mMapI2O.TriValue(aTriIn);
	     
            if (aTriOut.Regularity() <=0) 
               aRes = eZBufRes::UnRegOut;
	    else if (! mSetOut.InsideWithBox(aTriOut))
               aRes = eZBufRes::OutOut;
	    else
	    {
               aRes = MakeOneTri(aTriIn,aTriOut,aMode);
	    }
        }

	if (aMode==eZBufModeIter::SurfDevlpt)
	{
           cResModeSurfD aRMS;
	   aRMS.mResult = aRes;
	   aRMS.mResol  = mLastResSurfDev;
	   mResSurfD.push_back(aRMS);
	}
    }
    mMesh.ResetTri();
}

double cZBuffer::ComputeResol(const cTri3dR & aTri3In ,const cTri3dR & aTri3Out) const
{
	// input triangle, developped isometrically on the plane
	cTri2dR aTri2In  = cIsometry3D<tREAL8>::ToPlaneZ0(0,aTri3In,true);
	// output triangle, projected on the plane
        cTri2dR aTri2Out = Proj(aTri3Out);
	// Affinity  Input-Dev -> Output proj
	cAffin2D<tREAL8> aAffI2O =  cAffin2D<tREAL8>::Tri2Tri(aTri2In,aTri2Out);

	return aAffI2O.MinResolution();
}

eZBufRes cZBuffer::MakeOneTri(const cTri3dR & aTriIn,const cTri3dR &aTri3,eZBufModeIter  aMode)
{
    eZBufRes aRes = eZBufRes::Undefined;

    //  cTriangle2DCompiled<tREAL8>  aTri2(ToPix(aTri3.Pt(0)) , ToPix(aTri3.Pt(1)) ,ToPix(aTri3.Pt(2)));
    cTriangle2DCompiled<tREAL8>  aTri2 = ImageOfTri(Proj(aTri3),mROut2Pix);

    cPt3dr aPtZ(aTri3.Pt(0).z(),aTri3.Pt(1).z(),aTri3.Pt(2).z());

    std::vector<cPt2di> aVPix;
    std::vector<cPt3dr> aVW;


    cPt3dr aNorm = Normal(aTri3);

    int aSign = (aNorm.z() > 0) ? 1 : - 1;
     ///  the axe K of camera is in direction of view, the normal is in direction of visibility => they are opposite
    bool WellOriented =  mZF_SameOri ?  (aSign>0)  :(aSign<0);

    aTri2.PixelsInside(aVPix,1e-8,&aVW);
    tDIm & aDZImB = mZBufIm.DIm();
    int aNbVis = 0;
    for (size_t aK=0 ; aK<aVPix.size() ; aK++)
    {
       const cPt2di  & aPix = aVPix[aK];
       tElem aNewZ = mMultZ * Scal(aPtZ,aVW[aK]);
       tElem aZCur = aDZImB.GetV(aPix);
       if (aMode==eZBufModeIter::ProjInit)
       {
           if (aNewZ> aZCur)
           {
               aDZImB.SetV(aPix,aNewZ);
           }
       }
       else 
       {
           if (aNewZ==aZCur)
              aNbVis++;
       }
    }

    if (aMode==eZBufModeIter::SurfDevlpt)
    {
       if (! WellOriented) 
          aRes =  eZBufRes::BadOriented;
       else
       {
           bool IsVis = ((aNbVis*2)  > int(aVPix.size()));
           aRes = IsVis ? eZBufRes::Visible : eZBufRes::Hidden;
           mLastResSurfDev = ComputeResol(aTriIn,aTri3);
	   if (IsVis)
	   {
	       UpdateMax(mMaxRSD,mLastResSurfDev);
	   }

	   if ((aVPix.size()<=0) && (aNbVis==0))
              aRes = eZBufRes::NoPix;
       }
    }

    return aRes;
}

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
