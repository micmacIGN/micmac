#include "BundleAdjustment.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Manifolds.h"

namespace MMVII
{

class cOneData_L23
{
    public :
        cOneData_L23(cSensorCamPC * ,const tSeg2dr & aSeg,int aKIm);
        ~cOneData_L23();
        cLineDist_Manifold* SetAndGet_LineM() const;
        cCalculator<tREAL8> * SetAndGet_CalcEqSeg() const;


        cSensorCamPC *           mCam;
        const tSeg2dr            mSeg;
        const cPlane3D           mPlane;
        const int                mKIm;
        // cBoundVals<tREAL8>       mIntAbsc;
    private :
        mutable cLineDist_Manifold*   mLineM;
        mutable cCalculator<tREAL8> * mCalcEqSeg;
};

cOneData_L23::cOneData_L23(cSensorCamPC * aCam,const tSeg2dr & aSeg,int aKIm) :
    mCam   (aCam),
    mSeg   (aSeg),
    mPlane (mCam->SegImage2Ground(mSeg)),
    mKIm   (aKIm),
    mLineM (nullptr)
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


/// class to handle computation
class cCam2_Line_2Dto3D
{
    public :
       cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject *);

       const tSegComp3dr & Seg3d () const;
       const std::string & NameLine() const;
       std::vector<cOneData_L23>    Datas_L23();
       cPt3dr PtOfWeight(const tREAL8 aWeight);

    private :
       void AssertSeg3dIsInit() const;

       tSegComp3dr                  mSeg3d;
       bool                         mSeg3dIsInit;
       std::string                  mNameLine;

       std::vector<cOneData_L23>    mDatas_L23;
};

cCam2_Line_2Dto3D::cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject * aPhProj) :
     mSeg3d        (cPt3dr(0,0,0),cPt3dr(1,1,1)),
     mSeg3dIsInit (false)
{
    std::vector<cPlane3D> aVPlaneOk;

    for (size_t aKCam=0 ; aKCam<aVCam.size() ; aKCam++)
    {
         const auto & aCam = aVCam.at(aKCam);
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

/*
cPt3dr cCam2_Line_2Dto3D::PtOfWeight(const tREAL8 aWeight)
{
    tREAL8 aAbsc = mIntervAbsc.VMin()*(1-aWeight) + mIntervAbsc.VMax()*aWeight;

    return mSeg3d.PtOfAbscissa(aAbsc);
}
*/


/** in cUK_Line3D_4BA with put data in a specific class to allow copy (in "OnUpdate"),
 *  which would be forbiden due to inheritance */


/// class handling a 3D unknown line for bundle adjusment
class cUK_Line3D_4BA :   public cObjWithUnkowns<tREAL8>
{
    public :
         //<  constructor,
         cUK_Line3D_4BA(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject *,cREAL8_RSNL *);
         //< called to fill the "obs" in an equation
         void PushObs(std::vector<double>&);
         // cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject *);

         void AddEquation(tREAL8 aSigmaLine,int aNbPts);
    private :
         void AddOneEquation(tREAL8 aLambda, tREAL8 aWeight,const cOneData_L23&);
         void InitNormals();

         /// "reaction" after linear update
         void OnUpdate() override;                 
         /// method called when the object must indicate its unknowns
         void PutUknowsInSetInterval() override;

         cREAL8_RSNL *          mSys;
         cCam2_Line_2Dto3D      mLineInit;
         tSegComp3dr mSeg;
         cPt3dr      mNorm_x;     //< the first vector normal
         cPt3dr      mNorm_y;     //< the second vector normal
         cPt2dr_UK   mUkN1;       //< unknown displacement at Seg.P1, coded as "Uk1.x Nx+ Uk1.y Ny"
         cPt2dr_UK   mUkN2;       //<  unknown displacement at Seg.P2
         //cUK_Line3D_4BA_Data    mData;
};




/* *********************************************************** */
/*                                                             */
/*                 cUK_Line3D_4BA                              */
/*                                                             */
/* *********************************************************** */

cUK_Line3D_4BA::cUK_Line3D_4BA(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject * aPhProj,  cREAL8_RSNL *  aSys) :
    mSys       (aSys),
    mLineInit  (aVCam,aPhProj),
    mSeg       (mLineInit.Seg3d()),
    mUkN1      (cPt2dr(0,0),std::string("Line3d_Uk1") + mLineInit.NameLine()),
    mUkN2      (cPt2dr(0,0),std::string("Line3d_Uk2") + mLineInit.NameLine())
{
    InitNormals();
    for (auto aCam : aVCam)
        aCam->InternalCalib()->SetAndGet_EqProjSeg();
}

void cUK_Line3D_4BA::InitNormals()
{
    tRotR  aRot = tRotR::CompleteRON(mSeg.Tgt());
    mNorm_x = aRot.AxeJ();
    mNorm_y = aRot.AxeK();
}

void cUK_Line3D_4BA::PutUknowsInSetInterval()
{
   mSetInterv->AddOneInterv(mUkN1.Pt());
   mSetInterv->AddOneInterv(mUkN2.Pt());
}


void cUK_Line3D_4BA::OnUpdate() 
{
    cPt3dr aNewP1 = mSeg.P1() + mNorm_x * mUkN1.Pt().x()  +  mNorm_y * mUkN1.Pt().y();
    cPt3dr aNewP2 = mSeg.P2() + mNorm_x * mUkN2.Pt().x()  +  mNorm_y * mUkN2.Pt().y();

    mSeg = tSegComp3dr(aNewP1,aNewP2);

    mUkN1.Pt() = cPt2dr(0,0);
    mUkN2.Pt() = cPt2dr(0,0);

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
       mUkN1.PushIndexes(aVIndexes);
       mUkN2.PushIndexes(aVIndexes);

       for (auto & anObj : aData.mCam->GetAllUK())
       {
           anObj->PushIndexes(aVIndexes);
       }
   }

    mSys->R_CalcAndAddObs(aData.SetAndGet_CalcEqSeg(),aVIndexes,aVObs,aWeight);
}

void cUK_Line3D_4BA::AddEquation(tREAL8 aSigmaLine,int aNbPts)
{
    for (int aKPt=0 ; aKPt<aNbPts ; aKPt++)
    {
        tREAL8 aLambda = (aKPt+0.5) / aNbPts;
        tREAL8 aWeight = 1.0 /  (Square(aSigmaLine) * aNbPts);

        for (const auto & aData_L23 : mLineInit.Datas_L23())
             AddOneEquation(aLambda,aWeight,aData_L23);
    }
}


};
