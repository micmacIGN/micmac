#include "BundleAdjustment.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Manifolds.h"

namespace MMVII
{





/** Store data one line 2d->3d, essentially data */

class cOneData_L23 : public cMemCheck
{
    public :
        ///  constructor 
        cOneData_L23(cSensorCamPC * ,const tSeg2dr & aSeg,int aKIm);
        /// desctructor, free mutable object
        ~cOneData_L23();
        ///  create, if dont exist,  manifold for distorted line
        cLineDist_Manifold* SetAndGet_LineM() const;
        ///  create, if dont exist,  calculator for line-adjustment
        cCalculator<tREAL8> * SetAndGet_CalcEqSeg() const;


        cSensorCamPC *           mCam;   //< camera seeing the line
        const tSeg2dr            mSeg;   //< line seen in a camera
        const cPlane3D           mPlane; //< 3d plane projecting on the line
        const int                mKIm;   //< index in a vector (unused 4 now)
    private :
        mutable cLineDist_Manifold*   mLineM;
        mutable cCalculator<tREAL8> * mCalcEqSeg;
};



/// class to handle computation
class cCam2_Line_2Dto3D : public cMemCheck
{
    public :
       cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject *);

       const tSegComp3dr & Seg3d () const;               //< Accessor
       const std::string & NameLine() const;             //< Accessor
       const std::vector<cOneData_L23> &    Datas_L23() const; //< Accessor
       cPt3dr PtOfWeight(const tREAL8 aWeight);

    private :
       cCam2_Line_2Dto3D(const cCam2_Line_2Dto3D&) = delete;
       void AssertSeg3dIsInit() const;

       tSegComp3dr                  mSeg3d;
       bool                         mSeg3dIsInit;
       std::string                  mNameLine;

       std::vector<cOneData_L23>    mDatas_L23;
};


/// class handling a 3D unknown line for bundle adjusment
class cUK_Line3D_4BA :   public cObjWithUnkowns<tREAL8>
{
    public :
	 friend cMMVII_BundleAdj;
         //<  constructor,
         cUK_Line3D_4BA(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject *,cMMVII_BundleAdj *,tREAL8 aSigmaIm,int aNbPts);
         ~cUK_Line3D_4BA();

         void AddEquation();
    private :


	 cUK_Line3D_4BA(const cUK_Line3D_4BA &) = delete;
         void AddOneEquation(tREAL8 aLambda, tREAL8 aWeight,const cOneData_L23&);
         void InitNormals();

         /// "reaction" after linear update
         void OnUpdate() override;
         /// method called when the object must indicate its unknowns
         void PutUknowsInSetInterval() override;
         void  FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> &) override;

         cMMVII_BundleAdj*      mBA;
         cCam2_Line_2Dto3D    *  mLineInit;
         tSegComp3dr mSeg;
         cPt3dr      mNorm_x;     //< the first vector normal
         cPt3dr      mNorm_y;     //< the second vector normal
         cPt2dr      mUkN1;       //< unknown displacement at Seg.P1, coded as "Uk1.x Nx+ Uk1.y Ny"
         cPt2dr      mUkN2;       //<  unknown displacement at Seg.P2
         tREAL8      mSigmaIm;
         int         mNbPtSampling;
         //cUK_Line3D_4BA_Data    mData;
};


/* *********************************************************** */
/*                                                             */
/*                 cOneData_L23                                */
/*                                                             */
/* *********************************************************** */


cOneData_L23::cOneData_L23(cSensorCamPC * aCam,const tSeg2dr & aSeg,int aKIm) :
    mCam   (aCam),
    mSeg   (aSeg),
    mPlane (mCam->SegImage2Ground(mSeg)),
    mKIm   (aKIm),
    mLineM (nullptr),
    mCalcEqSeg (nullptr)
{
}




cOneData_L23::~cOneData_L23()
{
    delete mLineM;
}

cLineDist_Manifold* cOneData_L23::SetAndGet_LineM() const
{
    if (mLineM==nullptr)
        mLineM = new cLineDist_Manifold(mSeg,mCam->InternalCalib());

    return mLineM;
}

cCalculator<tREAL8> * cOneData_L23::SetAndGet_CalcEqSeg() const
{
   if (mCalcEqSeg==nullptr)
       mCalcEqSeg = mCam->InternalCalib()->SetAndGet_EqProjSeg();

   return mCalcEqSeg;
}

/* *********************************************************** */
/*                                                             */
/*                 cCam2_Line_2Dto3D                           */
/*                                                             */
/* *********************************************************** */

cCam2_Line_2Dto3D::cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject * aPhProj) :
     mSeg3d        (cPt3dr(0,0,0),cPt3dr(1,1,1)),
     mSeg3dIsInit (false)
{
    std::vector<cPlane3D> aVPlaneOk;

    for (size_t aKCam=0 ; aKCam<aVCam.size() ; aKCam++)
    {
         const auto & aCam = aVCam.at(aKCam);
	 if (aCam != nullptr)
	 {
            const std::string & aNameIm = aCam->NameImage();
            if (aPhProj->HasFileLines(aNameIm))
            {
                cLinesAntiParal1Im   aSetL  = aPhProj->ReadLines(aNameIm);
                const std::vector<cOneLineAntiParal> & aVL  = 	aSetL.mLines;

                // At this step we dont handle multiple lines
                if (aVL.size()==1)
                {
                    mDatas_L23.push_back(cOneData_L23(aCam,aVL.at(0).mSeg,aKCam));
                    aVPlaneOk.push_back(mDatas_L23.back().mPlane);
                }
            }
         }
    }

    if (aVPlaneOk.size()>=2)
    {
        mSeg3dIsInit = true;
        mSeg3d =  tSegComp3dr (cPlane3D::InterPlane(aVPlaneOk));
        cBoundVals<tREAL8>           aIntervAbsc;
        mNameLine = aPhProj-> DPGndPt2D().DirIn();

        for (auto & aData : mDatas_L23)
        {
            for (const auto & aP2 : aData.mSeg.VPts())
            {
                tSeg3dr aBundle =  aData.mCam->Image2Bundle(aP2);
                cPt3dr aABC;
                BundleInters(aABC,mSeg3d,aBundle,1.0);
                tREAL8 aAbsc = aABC.x();
                aIntervAbsc.Add(aAbsc);
              //  aData.mIntAbsc.Add(aAbsc);
            }
        }

        cPt3dr aP1 = mSeg3d.PtOfAbscissa(aIntervAbsc.VMin());
        cPt3dr aP2 = mSeg3d.PtOfAbscissa(aIntervAbsc.VMax());
        mSeg3d = tSegComp3dr(aP1,aP2);

    }

}

void cCam2_Line_2Dto3D::AssertSeg3dIsInit() const
{
    MMVII_INTERNAL_ASSERT_always(mSeg3dIsInit,"cCam2_Line_2Dto3D::AssertSeg3dIsInit");
}


const tSegComp3dr & cCam2_Line_2Dto3D::Seg3d () const
{
   AssertSeg3dIsInit();
   return mSeg3d;
}

const std::string & cCam2_Line_2Dto3D::NameLine() const
{
    AssertSeg3dIsInit();
    return mNameLine;
}


const std::vector<cOneData_L23> &  cCam2_Line_2Dto3D::Datas_L23() const
{
   return mDatas_L23;
}

/* *********************************************************** */
/*                                                             */
/*                 cUK_Line3D_4BA                              */
/*                                                             */
/* *********************************************************** */

    //  mUkN1      (cPt2dr(0,0),std::string("Line3d_Uk1") + mLineInit->NameLine()),
cUK_Line3D_4BA::cUK_Line3D_4BA
(
      const std::vector<cSensorCamPC *> & aVCam,
      cPhotogrammetricProject * aPhProj,
      cMMVII_BundleAdj *  aBA,
      tREAL8 aSigmaIm,
      int aNbPts
) :
    mBA           (aBA),
    mLineInit     (new cCam2_Line_2Dto3D (aVCam,aPhProj)),
    mSeg          (mLineInit->Seg3d()),
    mUkN1         (0.0,0.0),
    mUkN2         (0.0,0.0) ,
    mSigmaIm      (aSigmaIm),
    mNbPtSampling (aNbPts)
{
	//  ddOneObj()

    // aBA->SetIntervUK().AddOneObj(&mUkN1);
    // aBA->SetIntervUK().AddOneObj(&mUkN2);

    InitNormals();
    for (auto aCam : aVCam)
        aCam->InternalCalib()->SetAndGet_EqProjSeg();
}

cUK_Line3D_4BA::~cUK_Line3D_4BA()
{
    delete mLineInit;
}

void cUK_Line3D_4BA::InitNormals()
{
    tRotR  aRot = tRotR::CompleteRON(mSeg.Tgt());
    mNorm_x = aRot.AxeJ();
    mNorm_y = aRot.AxeK();
}

void cUK_Line3D_4BA::PutUknowsInSetInterval()
{
   mSetInterv->AddOneInterv(mUkN1);
   mSetInterv->AddOneInterv(mUkN2);
}

void cUK_Line3D_4BA::FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP) 
{
   aGAIP.TestParam(this, &(mUkN1.x()),"x1");
   aGAIP.TestParam(this, &(mUkN1.y()),"y1");
   aGAIP.TestParam(this, &(mUkN2.x()),"x2");
   aGAIP.TestParam(this, &(mUkN2.y()),"y2");

   aGAIP.SetNameType("Line3D");
   aGAIP.SetIdObj(mLineInit->NameLine());

}


void cUK_Line3D_4BA::OnUpdate() 
{
    cPt3dr aNewP1 = mSeg.P1() + mNorm_x * mUkN1.x()  +  mNorm_y * mUkN1.y();
    cPt3dr aNewP2 = mSeg.P2() + mNorm_x * mUkN2.x()  +  mNorm_y * mUkN2.y();

    mSeg = tSegComp3dr(aNewP1,aNewP2);

    mUkN1 = cPt2dr(0,0);
    mUkN2 = cPt2dr(0,0);

    InitNormals();
}


void cUK_Line3D_4BA::AddOneEquation(tREAL8 aLambda,tREAL8 aWeight,const cOneData_L23& aData)
{
    std::vector<double> aVObs ;
    std::vector<int> aVIndexes;

    //                         aPGround = aQ1 * (aC1-aLambda)  +  aQ2 * aLambda ;
    // template <cs T,const int Dim> cPtxd<T,Dim> Centroid(T aW0,const cPtxd<T,Dim> & aP0,const cPtxd<T,Dim> & aP1);
    cPt3dr aPGround = Centroid(1-aLambda,mSeg.P1(),aLambda,mSeg.P2());

   if (! aData.mCam->IsVisible(aPGround))
       return;

   {
      cPt2dr aPImPG = aData.mCam->Ground2Image(aPGround);
      cLineDist_Manifold* aLDM = aData.SetAndGet_LineM();

      cPt2dr aPImOnL2 = aLDM->Proj(aPImPG);
      cPt2dr aTgtL2 = aLDM->TgSpace(aPImOnL2).at(0);
      cPt2dr aNormL2 = Rot90(aTgtL2);

//StdOut() << " PIMG=" << aPImPG << " PL=" <<  aPImOnL2 << "\n";

    //     std::vector<std::string>  aVecLIne2D = Append(NamesP2("Line2D_Pt"),NamesP2("Line2D_Norm"));
       aPImOnL2.PushInStdVector(aVObs);
       aNormL2.PushInStdVector(aVObs);
   }
   {
       //   std::vector<std::string> aVPtsLine     =  Append(NamesP3("Line3d_Pt1"),NamesP3("Line3d_Pt2"));
       //   std::vector<std::string> aVPtsNormLine =  Append(NamesP3("Line3d_Norm_x"),NamesP3("Line3d_Norm_y"));
      // mData.PushObs(aVObs);
       mSeg.P1().PushInStdVector(aVObs);
       mSeg.P2().PushInStdVector(aVObs);
       mNorm_x.PushInStdVector(aVObs);
       mNorm_y.PushInStdVector(aVObs);

       // std::vector<std::string> aVLambdaLine  {"Line3d_Lambda"};
       // std::vector<std::string>  aVecLIne3D =  Append(aVPtsLine,aVPtsNormLine,aVLambdaLine);
        aVObs.push_back(aLambda);

        //    return Append(aVecLIne2D,aVecLIne3D,NamesMatr("M",cPt2di(3,3)));
         aData.mCam->PushOwnObsColinearity(aVObs,aPGround);
    }


    {

       this->PushIndexes(aVIndexes);
       // mUkN1.PushIndexes(aVIndexes);
       // mUkN2.PushIndexes(aVIndexes);

       for (auto & anObj : aData.mCam->GetAllUK())
       {
           anObj->PushIndexes(aVIndexes);
       }
   }


    mBA->Sys()->R_CalcAndAddObs(aData.SetAndGet_CalcEqSeg(),aVIndexes,aVObs,aWeight);
}

void cUK_Line3D_4BA::AddEquation()
{
    for (int aKPt=0 ; aKPt<mNbPtSampling ; aKPt++)
    {
        tREAL8 aLambda = (aKPt+0.5) / mNbPtSampling;
        tREAL8 aWeight = 1.0 /  (Square(mSigmaIm) * mNbPtSampling);

        for (const auto & aData_L23 : mLineInit->Datas_L23())
             AddOneEquation(aLambda,aWeight,aData_L23);
    }
}

/* *********************************************************** */
/*                                                             */
/*                 cMMVII_BundleAdj                            */
/*                                                             */
/* *********************************************************** */

void cMMVII_BundleAdj::AddLineAdjust(const std::vector<std::string> & aVParam)
{
   tREAL8 aSigmaIm =  cStrIO<double>::FromStr(aVParam.at(0));
   int    aNbPts   =  cStrIO<int>::FromStr(aVParam.at(1));

   mLineAdjust = new cUK_Line3D_4BA(mVSCPC,mPhProj,this,aSigmaIm,aNbPts);
   mSetIntervUK.AddOneObj(mLineAdjust);
}

void cMMVII_BundleAdj::IterAdjustOnLine()
{
    if (mLineAdjust)
        mLineAdjust->AddEquation();
}


// cUK_Line3D_4BA::cUK_Line3D_4BA(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject * aPhProj,cMMVII_BundleAdj *  aBA) :

void cMMVII_BundleAdj::DeleteLineAdjust()
{
	delete mLineAdjust;
}


};
