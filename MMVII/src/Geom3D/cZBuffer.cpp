#include "MMVII_Ptxd.h"
#include "MMVII_Image2D.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Triangles.h"
#include "MMVII_Mappings.h"

#include "MMVII_ZBuffer.h"


namespace MMVII
{

bool  ZBufLabIsOk(eZBufRes aLab)
{
     return (aLab==eZBufRes::Visible) || (aLab==eZBufRes::LikelyVisible) ;
}


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

bool cCountTri3DIterator::GetNextTri(tTri3dr & aTri)
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
tTri3dr cMeshTri3DIterator::KthF(int aKF) const {return mTri->KthTri(aKF);}

/* =============================================== */
/*                                                 */
/*              cResModeSurfD                      */
/*                                                 */
/* =============================================== */

void  AddData(const cAuxAr2007  &anAux,cResModeSurfD& aRMS )
{
     int aResult = int(aRMS.mResult);
     AddData(cAuxAr2007("Result",anAux),aResult);
     if (anAux.Input())
        aRMS.mResult = eZBufRes(aResult);
     AddData(cAuxAr2007("Resol",anAux),aRMS.mResol);
}

/* =============================================== */
/*                                                 */
/*              cZBuffer                           */
/*                                                 */
/* =============================================== */


cZBuffer::cZBuffer(cTri3DIterator & aMesh,const tSet &  aSetIn,const tMap & aMapI2O,const tSet &  aSetOut,double aResolOut) :
    mIsOk       (true),
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
            aCptIn++;
            cPt3dr aPOut = mMapI2O.Value(aPIn);

            if (mSetOut.InsideWithBox(aPOut))
            {
               aBoxOfPtsIn.Add(aPIn);
               aBoxOfPtsOut.Add(aPOut);
            }
        }
    }
    mMesh.ResetPts();

    if ((aBoxOfPtsIn.NbPts()<3) || (aBoxOfPtsOut.NbPts()<3))
    {
        mIsOk = false;
        return;
    }

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

void cZBuffer::AssertIsOk() const
{
   MMVII_INTERNAL_ASSERT_tiny(mIsOk,"Non ok Buffer");
}

bool cZBuffer::IsOk() const {return mIsOk;}

void cZBuffer::MakeZBuf(eZBufModeIter aMode)
{

    if (aMode==eZBufModeIter::SurfDevlpt)
    {
        mResSurfD.clear();
        mMaxRSD = 0.0;
    }

    tTri3dr  aTriIn = tTri3dr::Tri000();
    int aNbTriVis = 0;
    while (mMesh.GetNextTri(aTriIn))
    {
        mLastResSurfDev = -1;
        eZBufRes aRes = eZBufRes::Undefined;
        if (!mIsOk)
        {
        }
        //  not sure this is us to test that, or the user to assure it give clean data ...
        else if (aTriIn.Regularity() <=0)
           aRes = eZBufRes::UnRegIn;
        else if (! mSetIn.InsideWithBox(aTriIn))
           aRes = eZBufRes::OutIn;
        else
        {
            tTri3dr aTriOut = mMapI2O.TriValue(aTriIn);

            if (aTriOut.Regularity() <=0)
               aRes = eZBufRes::UnRegOut;
            else if (! mSetOut.InsideWithBox(aTriOut))
               aRes = eZBufRes::OutOut;
            else
            {
               aNbTriVis++;
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
    if (aNbTriVis==0) 
       mIsOk=false;

    mMesh.ResetTri();
}


double cZBuffer::ComputeResol(const tTri3dr & aTri3In ,const tTri3dr & aTri3Out) const
{
        // input triangle, developped isometrically on the plane
        tTri2dr aTri2In  = cIsometry3D<tREAL8>::ToPlaneZ0(0,aTri3In,true);
        // output triangle, projected on the plane
        tTri2dr aTri2Out = Proj(aTri3Out);
        // Affinity  Input-Dev -> Output proj
        cAffin2D<tREAL8> aAffI2O =  cAffin2D<tREAL8>::Tri2Tri(aTri2In,aTri2Out);

        return aAffI2O.MinResolution();
}


eZBufRes cZBuffer::MakeOneTri(const tTri3dr & aTriIn,const tTri3dr &aTri3,eZBufModeIter  aMode)
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








};
