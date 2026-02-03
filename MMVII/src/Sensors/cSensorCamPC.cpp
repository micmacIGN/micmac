#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"


/**
   \file cSensorCamPC.cpp  

   \brief file for implementing class sensor for perspective central
*/


namespace MMVII
{

/* ******************************************************* */
/*                                                         */
/*                   cP3dNormWithUK                        */
/*                                                         */
/* ******************************************************* */

cP3dNormWithUK::cP3dNormWithUK(const cPt3dr & aPt, const std::string& aNameType,const std::string & aNameGrp) :
    mPNorm (aPt)
{
    Init();
    SetNameType(aNameType);
    SetNameIdObj(aNameGrp);
}

cP3dNormWithUK::~cP3dNormWithUK()
{
    OUK_Reset();  // copy what we have on  cPoseWithUK
}


void cP3dNormWithUK::Init()
{
    mPNorm = VUnit(mPNorm);
    tRotR aRot = tRotR::CompleteRON(mPNorm);
    mU = aRot.AxeJ();
    mV = aRot.AxeK();

    mDuDv = cPt2dr(0,0);
}

void cP3dNormWithUK::OnUpdate()
{
    mPNorm = GetPNorm();
    Init();
}

void cP3dNormWithUK::PutUknowsInSetInterval()
{
   mSetInterv->AddOneInterv(mDuDv);
}

cPt2dr & cP3dNormWithUK::DuDv(){return mDuDv;}

void cP3dNormWithUK::AddIdexesAndObs(std::vector<int> & aVecIndexe, std::vector<double>& aVecObs)
{
    PushIndexes(aVecIndexe);
    AppendIn(aVecObs,mPNorm.ToStdVector());
    AppendIn(aVecObs,mU.ToStdVector());
    AppendIn(aVecObs,mV.ToStdVector());
}

void  cP3dNormWithUK::FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
   aGAIP.TestParam(this, &( mDuDv.x()),"Dx");
   aGAIP.TestParam(this, &( mDuDv.y()),"Dy");
}

cPt3dr cP3dNormWithUK::GetPNorm() const
{
    return  VUnit(mPNorm + mU*mDuDv.x() + mV*mDuDv.y());
}

void cP3dNormWithUK::SetPNorm(const cPt3dr & aTr)
{
    mPNorm = aTr;
    Init();
}


void AddData(const  cAuxAr2007 & anAux,cP3dNormWithUK & aPUK)
{
    cPt3dr aPt = aPUK.GetPNorm();
    MMVII::AddData(anAux,aPt);
    if (anAux.Input())
        aPUK.SetPNorm(aPt);
    else
    {
         MMVII_INTERNAL_ASSERT_tiny(IsNull(aPUK.DuDv()),"Write unknown pt with DuDv!=0");
    }
}


/* ******************************************************* */
/*                                                         */
/*                   cRotWithUK                            */
/*                                                         */
/* ******************************************************* */
//      cRotWithUK(const tRotR & aPose);


cRotWithUK::cRotWithUK(const tRotR & aRot) :
    mRot (aRot),
    mOmega (0.0,0.0,0.0)
{

}

//       cRotWithUK();

cRotWithUK::cRotWithUK() :
    cRotWithUK (tRotR())
{
}


cRotWithUK::~cRotWithUK()
{
    OUK_Reset();
}

const tRotR &   cRotWithUK::Rot()   const {return mRot;}
void cRotWithUK::SetRot( const tRotR & aRot)
{
    mOmega = cPt3dr(0.0,0.0,0.0);
    mRot= aRot;
}

cPt3dr cRotWithUK::AxeI()   const {return mRot.AxeI();}
cPt3dr cRotWithUK::AxeJ()   const {return mRot.AxeJ();}
cPt3dr cRotWithUK::AxeK()   const {return mRot.AxeK();}


const cPt3dr &  cRotWithUK::Omega() const {return mOmega;}
cPt3dr &  cRotWithUK::GetRefOmega()  {return mOmega;}
void  cRotWithUK::SetOmega(const cPt3dr & anOmega) {mOmega = anOmega;}

void cRotWithUK::PushObs(std::vector<double> & aVObs,bool TransposeMatr)
{
    if (TransposeMatr)
       mRot.Mat().PushByCol(aVObs);
    else
       mRot.Mat().PushByLine(aVObs);
}


void cRotWithUK::OnUpdate()
{
    //  used above formula to modify  rotation
   mRot =  mRot * cRotation3D<tREAL8>::RotFromAxiator(-mOmega);
   // now this have modify rotation, the "delta" is void :
   mOmega = cPt3dr(0,0,0);
}

void  cRotWithUK::FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
    aGAIP.TestParam(this, &( mOmega.x())    ,"Wx");
    aGAIP.TestParam(this, &( mOmega.y())    ,"Wy");
    aGAIP.TestParam(this, &( mOmega.z())    ,"Wz");

    SetNameTypeId(aGAIP);
}


void cRotWithUK::AddIdexesAndObs(std::vector<int> & aVIndexes, std::vector<double>& aVObs,bool TranspMatr)
{
  //   StdOut() << "IIIIIIInnd " << IndUk0() << " " << IndUk1() << "\n";
   PushIndexes(aVIndexes);
   PushObs(aVObs,TranspMatr);
}
/*
PushIndexes(aVecIndexe);
AppendIn(aVecObs,mPNorm.ToStdVector());
AppendIn(aVecObs,mU.ToStdVector());
AppendIn(aVecObs,mV.ToStdVector());
*/


void cRotWithUK::PutUknowsInSpecifiedSetInterval(cSetInterUK_MultipeObj<tREAL8> * aSetInterv)
{
 //StdOut() << "cRotWithUK::PutUknowsInSpecifiedSetIntervacRotWithUK::PutUknowsInSpecifiedSetIntervacRotWithUK::PutUknowsInSpecifiedSetInte\n";
    aSetInterv->AddOneInterv(mOmega);

 // StdOut() << "RRrrrr_IIIIIIInnd " << IndUk0() << " " << IndUk1() << "\n";
}
void cRotWithUK::PutUknowsInSetInterval()
{
    PutUknowsInSpecifiedSetInterval(mSetInterv);
}


cPt3dr cRotWithUK::ValAxiatorFixRot(const tRotR & aRotFix) const
{
     cDenseMatrix<tREAL8>  aM = mRot.Mat().Transpose() * aRotFix.Mat();
     tREAL8 aZ = ( aM(1,0) - aM(0,1)) / 2.0;
     tREAL8 aY = (-aM(2,0) + aM(0,2)) / 2.0;
     tREAL8 aX = ( aM(2,1) - aM(1,2)) / 2.0;

     return cPt3dr(aX,aY,aZ);
}

/* ******************************************************* */
/*                                                         */
/*                   cPoseWithUK                           */
/*                                                         */
/* ******************************************************* */

cPoseWithUK::cPoseWithUK(const tPoseR & aPose)  :
      mTr   (aPose.Tr()),
      mRUK (aPose.Rot())
{
}


cPoseWithUK::cPoseWithUK() :
	cPoseWithUK(tPoseR())
{
}



cPoseWithUK::~cPoseWithUK()
{
    OUK_Reset();
}

void cPoseWithUK::SetPose(const tPoseR & aPose)
{
    mRUK.SetRot(aPose.Rot());
    mTr = aPose.Tr();
}


tPoseR   cPoseWithUK::Pose()   const
{
    return tPoseR(mTr,mRUK.Rot());
}

const tRotR & cPoseWithUK::Rot() const  {return mRUK.Rot();}

cRotWithUK & cPoseWithUK::RUK() {return mRUK;}


const cPt3dr &   cPoseWithUK::Tr() const {return mTr;}
cPt3dr &   cPoseWithUK::GetRefTr() {return mTr;}

cPt3dr  cPoseWithUK::AxeI()   const {return mRUK.AxeI();}
cPt3dr  cPoseWithUK::AxeJ()   const {return mRUK.AxeJ();}
cPt3dr  cPoseWithUK::AxeK()   const {return mRUK.AxeK();}



const cPt3dr &  cPoseWithUK::Omega() const {return mRUK.Omega();}
cPt3dr &  cPoseWithUK::GetRefOmega() {return  mRUK.GetRefOmega();}

void cPoseWithUK::SetOmega(const cPt3dr & anOmega) {mRUK.SetOmega(anOmega);}


/*   Let R be the rotation of pose  P=(C,R) = : Cam-> Word, what is optimized in colinearity for a ground point G
 *   is Word->Cam  :
 *
 *          tR(G-C)
 *
 * So the optimal rotation R' with get satisfy the equation :
 *
 *          (1+^W) tR0 =tR'   
 *
 * The we have the formula :
 *
 *          --->   R'=R0 t(1+^W)
 *
 *  And note that for axiators :
 *
 *       t Axiator(W) = Axiator(-W)
 *
 */


/*  aRF  =  R0 * Ax(-W)   
*   Ax(-W) =  tR0  * aRF
*  cPtxd<T,3>(  0    , -W.z() ,  W.y() ),
*  cPtxd<T,3>( W.z() ,   0    , -W.x() ),
*  cPtxd<T,3>(-W.y() ,  W.x() ,   0    )
*
*  aMat(1,0) = -aW.z(); aMat(2,0) =  aW.y(); aMat(2,1) = -aW.x(); 
*/  


cPt3dr cPoseWithUK::ValAxiatorFixRot(const tRotR & aRotFix) const
{
    return mRUK.ValAxiatorFixRot(aRotFix);
    /*
     cDenseMatrix<tREAL8>  aM = mPose.Rot().Mat().Transpose() * aRotFix.Mat();
     tREAL8 aZ = ( aM(1,0) - aM(0,1)) / 2.0;
     tREAL8 aY = (-aM(2,0) + aM(0,2)) / 2.0;
     tREAL8 aX = ( aM(2,1) - aM(1,2)) / 2.0;

     return cPt3dr(aX,aY,aZ);
     */
}
#if (0)

#endif

void cPoseWithUK::OnUpdate()
{
    mRUK.OnUpdate();
}

void  cPoseWithUK::FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
   aGAIP.TestParam(this, &( mTr.x()),"Cx");
   aGAIP.TestParam(this, &( mTr.y()),"Cy");
   aGAIP.TestParam(this, &( mTr.z()),"Cz");

   mRUK.FillGetAdrInfoParam(aGAIP);
   SetNameTypeId(aGAIP);
}

void cPoseWithUK::PutUknowsInSpecifiedSetInterval(cSetInterUK_MultipeObj<tREAL8> * aSetInterv)
{
    aSetInterv->AddOneInterv(mTr);
    mRUK.PutUknowsInSpecifiedSetInterval(aSetInterv);

   // StdOut() << "PPPPrrrr_IIIIIIInnd " << IndUk0() << " " << IndUk1() << "\n";

}

void cPoseWithUK::PutUknowsInSetInterval()
{
//StdOut() << "PoseWithUK::PutUknowsInSetInterval " << __LINE__ << "\n";
    PutUknowsInSpecifiedSetInterval(mSetInterv);
//StdOut() << "PoseWithUK::PutUknowsInSetInterval " << __LINE__ << "\n";
 //   mRUK.PutUknowsInSetInterval();
//StdOut() << "PoseWithUK::PutUknowsInSetInterval " << __LINE__ << "\n";
}

void AddData(const  cAuxAr2007 & anAux,cPoseWithUK & aPUK)
{
   // StdOut() << "vv765vvVVvv void AddData(const  cAuxAr2007 & anAux,cPoseWithUK & aPUK)\n";

    tPoseR aPose = aPUK.Pose();
    MMVII::AddData(anAux,aPose);
    if (anAux.Input())
        aPUK.SetPose(aPose);

   // MMVII::AddData(anAux,aPUK.Pose());

    if (anAux.Input())
    {
       aPUK.SetOmega(cPt3dr(0,0,0));
    }
    else
    {
         MMVII_INTERNAL_ASSERT_tiny(IsNull(aPUK.Omega()),"Write unknown rot with omega!=0");
    }
}

void cPoseWithUK::PushObs(std::vector<double> & aVObs,bool TransposeMatr)
{
      mRUK.PushObs(aVObs,TransposeMatr);
      /*
     if (TransposeMatr) 
        mPose.Rot().Mat().PushByCol(aVObs);
     else
        mPose.Rot().Mat().PushByLine(aVObs);
        */
}

void cPoseWithUK::AddIdexesAndObs(std::vector<int> & aVIndexes, std::vector<double>& aVObs,bool TranspMatr)
{
    PushIndexes(aVIndexes);
    PushObs(aVObs,TranspMatr);
}

//   StdOut() << "IIIIIIInnd " << IndUk0() << " " << IndUk1() << "\n";

/* ******************************************************* */
/*                                                         */
/*                   cSensorCamPC                          */
/*                                                         */
/* ******************************************************* */

cSensorCamPC::cSensorCamPC(const std::string & aNameImage,const tPose & aPose,cPerspCamIntrCalib * aCalib) :
   cSensorImage     (aNameImage),
   mPose_WU         (aPose),
   mInternalCalib   (aCalib)
{
}

void cSensorCamPC::SetPose(const tPose & aPose)
{
   mPose_WU.SetPose(aPose);
}

         // void SetCenter(const cPt3dr & aC);

void cSensorCamPC::SetOrient(const tRotR & anOrient)
{
     SetPose(tPose(Center(),anOrient));
}

void cSensorCamPC::SetCenter(const cPt3dr & aC)
{
     SetPose(tPose(aC,Orient()));
}


#if (1)
std::vector<cObjWithUnkowns<tREAL8> *>  cSensorCamPC::GetAllUK() 
{
    // Dont work because unknown are added twice
    // return std::vector<cObjWithUnkowns<tREAL8> *> {this,mInternalCalib,&mPose_WU};
    
    if (mInternalCalib)
       return std::vector<cObjWithUnkowns<tREAL8> *> {this,mInternalCalib};
   return std::vector<cObjWithUnkowns<tREAL8> *> {this};
}
void cSensorCamPC::PutUknowsInSetInterval()
{
    mPose_WU.PutUknowsInSpecifiedSetInterval(mSetInterv);
}
#else
std::vector<cObjWithUnkowns<tREAL8> *>  cSensorCamPC::GetAllUK() 
{
    mInternalCalib ?
    return std::vector<cObjWithUnkowns<tREAL8> *> {this,mInternalCalib,&mPose_WU};
}
void cSensorCamPC::PutUknowsInSetInterval()
{
    // Dont work because IndOfVal cannot be found
    // mPose_WU.PutUknowsInSetInterval(mSetInterv);
}
#endif



cPt3dr  cSensorCamPC::EpsDiffGround2Im(const cPt3dr & aPt) const 
{
    tREAL8 aNbPixel = 5.0;

    tREAL8 aEps = (Norm2(aPt-Center()) / mInternalCalib->F()) * aNbPixel;
    return cPt3dr::PCste(aEps);
}


cPt2dr cSensorCamPC::Ground2Image(const cPt3dr & aP) const
{
     return mInternalCalib->Value(Pt_W2L(aP));
}


cPlane3D  cSensorCamPC::SegImage2Ground(const tSeg2dr & aSeg,tREAL8 aDepth) const
{
     cPt3dr aPG1 = cSensorImage::ImageAndDepth2Ground(aSeg.P1(),aDepth);
     cPt3dr aPG2 = cSensorImage::ImageAndDepth2Ground(aSeg.P2(),aDepth);
     cPt3dr aPG0 = Center();

     return cPlane3D::From3Point(aPG0,aPG1,aPG2);
}

tREAL8  cSensorCamPC::GroundDistBundleSeg(const cPt2dr & aPIm,const cSegmentCompiled<tREAL8,3>  & aSeg3) const
{
   cSegmentCompiled<tREAL8,3> aBundIm = Image2Bundle(aPIm);
   cPt3dr  aPWire =  BundleInters(aSeg3,aBundIm,1.0);

   return aBundIm.Dist(aPWire);
}

tREAL8  cSensorCamPC::PixDistBundleSeg(const cPt2dr & aPIm,const cSegmentCompiled<tREAL8,3>  & aSeg3) const
{
   cSegmentCompiled<tREAL8,3> aBundIm = Image2Bundle(aPIm);
   cPt3dr  aPWire =  BundleInters(aSeg3,aBundIm,1.0);
   cPt3dr  aPBund =  BundleInters(aSeg3,aBundIm,0.0);

   tREAL8 aEps = Norm2(aPBund-Center()) / mInternalCalib->F(); // +or- the epsilon 3D correspond to 1 pixel

   cPt2dr aPIm1 = Ground2Image(aPBund + aSeg3.Tgt() * aEps);
   cPt2dr aPIm2 = Ground2Image(aPBund - aSeg3.Tgt() * aEps);
   cPt2dr aDirIm = aPIm2-aPIm1;

   cSegment2DCompiled aSeg(aPIm,aPIm+aDirIm);

   return aSeg.DistLine(Ground2Image(aPWire));
}



        //  Local(0,0,0) = Center, then mPose Cam->Word, then we use Inverse, BTW Inverse is as efficient as direct
cPt3dr cSensorCamPC::Pt_W2L(const cPt3dr & aP) const { return       Pose().Inverse(aP); }
cPt3dr cSensorCamPC::Vec_W2L(const cPt3dr & aP) const { return Pose().Rot().Inverse(aP); }

cPt3dr cSensorCamPC::Pt_L2W(const cPt3dr & aP) const { return       Pose().Value(aP); }
cPt3dr cSensorCamPC::Vec_L2W(const cPt3dr & aP) const { return Pose().Rot().Value(aP); }


double cSensorCamPC::DegreeVisibility(const cPt3dr & aP) const
{
     return mInternalCalib->DegreeVisibility(Pose().Inverse(aP));
}

double cSensorCamPC::DegreeVisibilityOnImFrame(const cPt2dr & aP) const
{
     return mInternalCalib->DegreeVisibilityOnImFrame(aP);
}

bool   cSensorCamPC::HasImageAndDepth() const {return true;}

const cPt2di & cSensorCamPC::SzPix() const {return  mInternalCalib->SzPix();}


     // ---------------   Ground2ImageAndDepth ----------------------------

cPt3dr cSensorCamPC::PlaneSweep_Ground2ImageAndDepth(const cPt3dr & aP) const
{
    cPt3dr aPCam = Pose().Inverse(aP);  // P in camera coordinate
    cPt2dr aPIm = mInternalCalib->Value(aPCam);

    return cPt3dr(aPIm.x(),aPIm.y(),aPCam.z());
}
cPt3dr cSensorCamPC::Dist_Ground2ImageAndDepth(const cPt3dr & aP) const
{
    cPt3dr aPCam = Pose().Inverse(aP);  // P in camera coordinate
    cPt2dr aPIm = mInternalCalib->Value(aPCam);

    return cPt3dr(aPIm.x(),aPIm.y(),Norm2(aPCam));
}
cPt3dr cSensorCamPC::Ground2ImageAndDepth(const cPt3dr & aP) const
{
    return mInternalCalib->TypeProj() == eProjPC::eStenope  ?
                       PlaneSweep_Ground2ImageAndDepth(aP)  :
                           Dist_Ground2ImageAndDepth(aP)    ;
}


     // ---------------   ImageAndDepth2Ground ----------------------------

cPt3dr cSensorCamPC::PlaneSweep_ImageAndDepth2Ground(const cPt3dr & aPImAndD) const
{
    cPt2dr aPIm = Proj(aPImAndD);
    cPt3dr aPCam = mInternalCalib->DirBundle(aPIm);
    cPt3dr aRes =  Pose().Value(aPCam * (aPImAndD.z() / aPCam.z()));
    return aRes;
}

cPt3dr cSensorCamPC::Dist_ImageAndDepth2Ground(const cPt3dr & aPImAndD) const
{
    cPt2dr aPIm = Proj(aPImAndD);
    cPt3dr aPCam = mInternalCalib->DirBundle(aPIm);
    cPt3dr aRes =  Pose().Value( VUnit(aPCam)*aPImAndD.z() );
    return aRes;
}


cPt3dr cSensorCamPC::ImageAndDepth2Ground(const cPt3dr & aPImAndD) const
{
    return mInternalCalib->TypeProj() == eProjPC::eStenope  ?
           PlaneSweep_ImageAndDepth2Ground(aPImAndD)        :
                 Dist_ImageAndDepth2Ground(aPImAndD)        ;
}


     // -------------------------------------------


tSeg3dr  cSensorCamPC::Image2Bundle(const cPt2dr & aPIm) const 
{
   return  tSeg3dr(Center(),static_cast<const cSensorImage*>(this)->ImageAndDepth2Ground(aPIm,1.0));
}


const cPt3dr * cSensorCamPC::CenterOfPC() const { return  & Center(); }
         /// Return the calculator, adapted to the type, for computing colinearity equation
cCalculator<double> * cSensorCamPC::CreateEqColinearity(bool WithDerives,int aSzBuf,bool ReUse) 
{
   if (mInternalCalib==nullptr) 
      return nullptr;
   return mInternalCalib->EqColinearity(WithDerives,aSzBuf,ReUse);
}

cCalculator<double> * cSensorCamPC::EqProjSeg()
{
   MMVII_INTERNAL_ASSERT_always(mInternalCalib,"cSensorCamPC::EqProjSeg");
   return  mInternalCalib->SetAndGet_EqProjSeg();
}

void cSensorCamPC::PushOwnObsColinearity(std::vector<double> & aVObs,const cPt3dr &)
{
     mPose_WU.PushObs(aVObs,true);
}

const cPixelDomain & cSensorCamPC::PixelDomain() const 
{
	return mInternalCalib->PixelDomain();
}



cPerspCamIntrCalib * cSensorCamPC::InternalCalib() const {return mInternalCalib;}

const cPt3dr & cSensorCamPC::Center()  const {return mPose_WU.Tr();}
const tRotR &   cSensorCamPC::Orient() const {return mPose_WU.Rot();}

const cPt3dr & cSensorCamPC::Omega()  const {return mPose_WU.Omega();}
cPt3dr & cSensorCamPC::Center() {return mPose_WU.GetRefTr();}
cPt3dr & cSensorCamPC::Omega()  {return mPose_WU.GetRefOmega();}
cPt3dr  cSensorCamPC::PseudoCenterOfProj() const {return Center();}

cPt3dr cSensorCamPC::AxeI()   const {return mPose_WU.AxeI();}
cPt3dr cSensorCamPC::AxeJ()   const {return mPose_WU.AxeJ();}
cPt3dr cSensorCamPC::AxeK()   const {return mPose_WU.AxeK();}
tPoseR cSensorCamPC::Pose() const {return mPose_WU.Pose();}

cPoseWithUK & cSensorCamPC::Pose_WU() {return mPose_WU;}


cIsometry3D<tREAL8>  cSensorCamPC::RelativePose(const cSensorCamPC& aCam2) const
{
    //  (A->W)-1 (B->W)(B) = (A->W)-1 (W) = A
    //  (A->W) -1 (B->W)  = (B->A) 
    return Pose().MapInverse()*aCam2.Pose();
}


void cSensorCamPC::OnUpdate()
{
     mPose_WU.OnUpdate();
}




cSensorCamPC * cSensorCamPC::PCChangSys(cDataInvertibleMapping<tREAL8,3> & aMap) const 
{
    cDataNxNMapping<tREAL8,3>::tResJac aCJac = aMap.Jacobian(Center());

    cPt3dr aNewC = aCJac.first;
    cDenseMatrix<tREAL8>  aJac  = aCJac.second ;

    if (1)
    {
        cPt3dr  aIJac ; GetCol(aIJac,aJac,0);
        cPt3dr  aJJac ; GetCol(aJJac,aJac,1);
        cPt3dr  aKJac ; GetCol(aKJac,aJac,2);
        aJac = M3x3FromCol(VUnit(aIJac),VUnit(aJJac),VUnit(aKJac));
    }
    cDenseMatrix<tREAL8>  aNewMatNonR  = aJac *  Pose().Rot().Mat();
    cDenseMatrix<tREAL8>  aNewMatR     =  aNewMatNonR.ClosestOrthog();

    tPoseR aNewPose(aNewC,cRotation3D<tREAL8>(aNewMatR,false));

    // return new cSensorCamPC
    //  M(aC+I)-M(C) = GradM * I
    //


    if (0)
    {

        cPt3dr aC = Center(); 
        tREAL8 aEps = 0.2;
        cPt3dr  aDx(aEps/2,0,0);
        cPt3dr  aDy(0,aEps/2,0);
        cPt3dr  aDz(0,0,aEps/2);

        cPt3dr aI = (aMap.Value(aC+aDx) - aMap.Value(aC-aDx)) / aEps;
        cPt3dr aJ = (aMap.Value(aC+aDy) - aMap.Value(aC-aDy)) / aEps;
        cPt3dr aK = (aMap.Value(aC+aDz) - aMap.Value(aC-aDz)) / aEps;


        StdOut()  << " CIJ "  << Cos(aI,aJ) << " CIK="  << Cos(aI,aK)  << " CJK=" << Cos(aJ,aK)  << "\n";
        if (std::abs( Cos(aI,aJ)) > 1e-2)
        {
               for (tREAL8 aEpsK : {0.001,0.01,0.1,1.0,10.0})
               {
                   cPt3dr  aDxK(aEpsK/2,0,0);
                   cPt3dr  aDyK(0,aEpsK/2,0);

                   cPt3dr aIK = (aMap.Value(aC+aDxK) - aMap.Value(aC-aDxK)) / aEpsK;
                   cPt3dr aJK = (aMap.Value(aC+aDyK) - aMap.Value(aC-aDyK)) / aEpsK;
                   StdOut()  << aIK << aJK << "\n";
                   StdOut()  << " CIJ "  << Cos(aIK,aJK) << "\n";
               }

              getchar();
        }
/*
        StdOut()  << " NI "  << Norm2(aI) << " NJ="  << Norm2(aJ)  << " NK=" << Norm2(aK)  << "\n";
        StdOut()  << " NI-NJ "  << std::abs(Norm2(aI) - Norm2(aJ) )   << "\n";

        cPt3dr  aIJac ; GetCol(aIJac,aJac,0);
        cPt3dr  aJJac ; GetCol(aJJac,aJac,1);
        cPt3dr  aKJac ; GetCol(aKJac,aJac,2);
        StdOut() << aI  - aIJac  << "\n";
        StdOut() << aJ  - aJJac  << "\n";
        StdOut() << aK  - aKJac  << "\n";

        StdOut()  <<  "CCC  " << aNewC <<  " DRot=" << aNewMatR.L2Dist(aNewMatNonR) <<  std::endl;

	cPt3dr aI,aJ;
        GetCol(aI,aNewMatNonR,0);
        GetCol(aJ,aNewMatNonR,1);

        cDenseMatrix<tREAL8>  aM0 =  Pose().Rot().Mat();

	StdOut()  <<   "UUU  M0=" <<  Pose().Rot().Mat().Unitarity() << "\n";
	StdOut()  <<  " Glob   Dij  "  << Norm2(aI)- Norm2(aJ) << " D1="  << Norm2(aI)-1.0 << " Cos=" << Cos(aI,aJ) << "\n";


        GetCol(aI,aJac.second.Transpose(),0);
        GetCol(aJ,aJac.second.Transpose(),1);
	StdOut()  <<  " TTT Jac   Dij  "  << Norm2(aI)- Norm2(aJ) << " D1="  << Norm2(aI)-1.0 << " Cos=" << Cos(aI,aJ) << "\n";

        GetCol(aI,aJac.second,0);
        GetCol(aJ,aJac.second,1);
	StdOut()  <<  " Jac   Dij  "  << Norm2(aI)- Norm2(aJ) << " D1="  << Norm2(aI)-1.0 << " Cos=" << Cos(aI,aJ) << "\n";

        aI = aMap.Value(Center()+cPt3dr(0.5,0,0)) - aMap.Value(Center()+cPt3dr(-0.5,0,0));
        aJ = aMap.Value(Center()+cPt3dr(0,0.5,0)) - aMap.Value(Center()+cPt3dr(0,-0.5,0));


	StdOut()  <<  "  MMM  "  << Norm2(aI)- Norm2(aJ) << " D1="  << Norm2(aI)-1.0 << " Cos=" << Cos(aI,aJ) << "\n";
	StdOut()  <<  "  IJ  "  << aI << " " << aJ << "\n";

	for (int aK=0 ; aK<3 ; aK++)
	{
		cPt3dr aP1,aP2;
		GetCol(aP1,aNewMatNonR,aK);
		GetCol(aP2,aNewMatR,aK);
                StdOut()  <<  "    DP  " << aP1 - aP2 <<  std::endl;
	}
*/
    }
    return new cSensorCamPC(NameImage(),aNewPose,InternalCalib());
}

cSensorImage * cSensorCamPC::SensorChangSys(const std::string &, cChangeSysCo &aMap) const
{
	return PCChangSys(aMap);
}




//
tREAL8 cSensorCamPC::AngularProjResiudal(const cPair2D3D& aPair,bool InPix) const
{
    cPt2dr aPIm =  aPair.mP2;
    cPt3dr aDirBundleIm = ImageAndDepth2Ground(cPt3dr(aPIm.x(),aPIm.y(),1.0)) - Center();
    cPt3dr aDirProj =  aPair.mP3 - Center();              // direction of projection

    tREAL8 aRes = Norm2(VUnit(aDirBundleIm)-VUnit(aDirProj));  // equivalent to angular distance
    if (InPix)
        aRes *= mInternalCalib->F();

    return aRes;
}

std::vector<tREAL8> cSensorCamPC::ListAngularProjResiudal(const cSet2D3D& aSet,bool InPix) const
{
    std::vector<tREAL8> aRes;
    for (const auto & aPair : aSet.Pairs())
    {
        aRes.push_back(AngularProjResiudal(aPair,InPix));
    }

    return aRes;
}



tREAL8  cSensorCamPC::AvgAngularProjResiudal(const cSet2D3D& aSet,bool InPix) const
{
   cWeightAv<tREAL8> aWA;

   for (const auto & aPair : aSet.Pairs())
   {
       aWA.Add(1.0,AngularProjResiudal(aPair,InPix));
   }

   return aWA.Average();

}


     // =================  READ/WRITE on files ===================

void cSensorCamPC::AddData(const cAuxAr2007 & anAux0)
{
    cAuxAr2007 anAux("CameraPose",anAux0);
    std::string aNameImage = NameImage();
    cPtxd<tREAL8,4>  aQuat =  MatrRot2Quat(Pose().Rot().Mat());
    std::string      aNameCalib = (anAux.Input() ? "" : (mInternalCalib?  mInternalCalib->Name() : MMVII_NONE ));


    MMVII::AddData(cAuxAr2007("NameImage",anAux),aNameImage);
    MMVII::AddData(cAuxAr2007("NameInternalCalib",anAux),aNameCalib);

    MMVII::AddData(anAux,mPose_WU);
    if (anAux.Input())
    {
         SetNameImage(aNameImage);
	 mTmpNameCalib = aNameCalib;
    }

    MMVII::AddData(cAuxAr2007("EQ",anAux),aQuat);
    anAux.Ar().AddComment("EigenQuaternion, for information");

    cPt3dr aWPK = Pose().Rot().ToWPK() *  (180.0/M_PI);
    MMVII::AddData(cAuxAr2007("WPK",anAux),aWPK);
    anAux.Ar().AddComment("Omega Phi Kapa in degree, for information");


    cPt3dr aYPR = Pose().Rot().ToYPR() *  (180.0/M_PI);
    MMVII::AddData(cAuxAr2007("YPR",anAux),aYPR);
    anAux.Ar().AddComment("Yaw Pitch Roll in degree, for information");

}

void AddData(const cAuxAr2007 & anAux,cSensorCamPC & aPC)
{
    aPC.AddData(anAux);
}

void cSensorCamPC::ToFile(const std::string & aNameFile) const
{
    SaveInFile(const_cast<cSensorCamPC &>(*this),aNameFile);
    if (mInternalCalib)
    {
        std::string aNameCalib = DirOfPath(aNameFile) + mInternalCalib->Name() + "." + GlobTaggedNameDefSerial();
        mInternalCalib->ToFileIfFirstime(aNameCalib);
    }
}

cSensorCamPC * cSensorCamPC::FromFile(const std::string & aFile,bool Remanent)
{
   ASSERT_NO_MUTI_THREAD();

   // Cannot use RemanentObjectFromFile because construction is not standard
   static std::map<std::string,cSensorCamPC*> TheMapRes;
   cSensorCamPC * & anExistingCam = TheMapRes[aFile];

   if (Remanent && (anExistingCam!= nullptr))
      return anExistingCam;


   cSensorCamPC * aPC = new cSensorCamPC("NONE",tPose::Identity(),nullptr);
   ReadFromFile(*aPC,aFile);

   if (aPC->mTmpNameCalib != MMVII_NONE)
       aPC->mInternalCalib =  cPerspCamIntrCalib::FromFile(DirOfPath(aFile) + aPC->mTmpNameCalib + "." + GlobTaggedNameDefSerial());
   else 
       aPC->mInternalCalib = nullptr;
   aPC->mTmpNameCalib = "";

   anExistingCam = aPC;
   return aPC;
}

std::string  cSensorCamPC::NameOri_From_Image(const std::string & aNameImage)
{
   return cSensorImage::NameOri_From_PrefixAndImage(PrefixName(),aNameImage);
}

std::vector<cPt2dr>  cSensorCamPC::PtsSampledOnSensor(int aNbByDim,tREAL8 aEps) const 
{
// StdOut()<< "PtsSampledOnSensorPtsSampledOnSensor " << aEps << "\n";  => PB EPS NOT USED
     return  mInternalCalib->PtsSampledOnSensor(aNbByDim,true);
}


std::string  cSensorCamPC::V_PrefixName() const { return PrefixName() ; }
std::string  cSensorCamPC::PrefixName()  { return "PerspCentral";}

void  cSensorCamPC::FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
   mPose_WU.FillGetAdrInfoParam(aGAIP);
   aGAIP.SetNameType("PoseCamPC");
   aGAIP.SetIdObj(NameImage());
}
     // =================  becnh ===================

void cSensorCamPC::Bench()
{
static int aCptCam=0; aCptCam++; 
  //  (int aNbByDim,int aNbDepts,double aD0,double aD1,bool IsDepthOrZ,tREAL8 aEpsMarginRel)
   cSet2D3D  aSet32 =  SyntheticsCorresp3D2D(20,3,1.0,10.0,true) ;
   tREAL8 aRes = AvgAngularProjResiudal(aSet32);

   for (const auto & aPair : aSet32.Pairs())
   {
static int aCptPt=0; aCptPt++; bool aBugPt = (aCptPt==49918); 
       const cPt3dr & aP3G1 = aPair.mP3;

       {
           cPt3dr aPt1 = Dist_Ground2ImageAndDepth(aP3G1);
           cPt3dr aPt2 = Dist_ImageAndDepth2Ground(aPt1);
           tREAL8 aDist =  Norm2(aPt2-aP3G1);
           MMVII_INTERNAL_ASSERT_bench(aDist<1e-6,"SensorCamPC::Bench  DG");

           cPt3dr aPt3 = Dist_Ground2ImageAndDepth(aPt2);
           aDist =  Norm2(aPt1-aPt3);
           // StdOut() << "NNN=" << aDist << "\n";
           MMVII_INTERNAL_ASSERT_bench(aDist<1e-4,"SensorCamPC::Bench  DG");
       }

       cPt3dr  aPIm1 =  PlaneSweep_Ground2ImageAndDepth(aP3G1);
       cPt3dr  aP3G2 =  PlaneSweep_ImageAndDepth2Ground(aPIm1);
       cPt3dr  aPIm2 =  PlaneSweep_Ground2ImageAndDepth(aP3G2);
       tREAL8 aDG = Norm2(aP3G1-aP3G2);
       tREAL8 aDI = Norm2(aPIm1-aPIm2);

       cPt3dr aPLoc = Pt_W2L(aP3G1);
       tREAL8 aRatio = std::abs(aPLoc.z()) / Norm2(aPLoc);
       if (0)
       {
           FakeUseIt(aBugPt);
           StdOut() << "HHHHH  " << aDG << " " << aDI << " C-PT=" << aCptPt << " C-Cam=" << aCptCam << "\n";//  getchar();

           StdOut() << " Proj=" << E2Str(mInternalCalib->TypeProj()) 
                    << " PLOC=" << aPLoc 
                    << " SzzzIm=" << Sz()
                    << " PIm=" << aPair.mP2 
                    << " PG1=" << aP3G1  
                    << " C=" <<  Center() 
                    << " Ratio=" << aRatio
                    << "\n";

       }
       if (aRatio>=1e-2)
       {
           tREAL8 aThrsG = 1e-4 + 1e-5/aRatio;
           MMVII_INTERNAL_ASSERT_bench(aDG<aThrsG,"SensorCamPC::Bench  DG");
       }
       MMVII_INTERNAL_ASSERT_bench(aDI<1e-4,"SensorCamPC::Bench  DI");
   }

   MMVII_INTERNAL_ASSERT_bench(aRes<1e-8,"Avg res ang");
}

void cSensorCamPC::BenchOneCalib(cPerspCamIntrCalib * aCalib)
{
    cIsometry3D<tREAL8> aPose = cIsometry3D<tREAL8>::RandomIsom3D(10.0);

    cSensorCamPC aCam("BenchCam",aPose,aCalib);
    aCam.Bench();
}

     // =================  Cast ===================

bool  cSensorCamPC::IsSensorCamPC() const  { return true; }
const cSensorCamPC * cSensorCamPC::GetSensorCamPC() const { return this; }
cSensorCamPC * cSensorCamPC::GetSensorCamPC() { return this; }



}; // MMVII


