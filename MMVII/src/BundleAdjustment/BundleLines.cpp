#include "BundleAdjustment.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

class cOneData_L23
{
    public :
        cOneData_L23();

        cSensorCamPC *     mCam;
        size_t             mKIm;
        tSeg2dr            mSeg;
        cBoundVals<tREAL8> mIntAbsc;
};


/// class to handle computation
class cCam2_Line_2Dto3D
{
    public :
       cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject *);

       const tSegComp3dr & Seg3d () const;

    private :
       void AssertSeg3dIsInit() const;

       tSegComp3dr                  mSeg3d;
       bool                         mSeg3dIsInit;
       cBoundVals<tREAL8>           mIntervAbsc;

       std::vector<cOneData_L23>    mDatas;

       std::vector<cPlane3D>        mVPlane;
       std::vector<cSensorCamPC *>  mVCamOk;
       std::vector<tSeg2dr>         mVSegOk;

};

cCam2_Line_2Dto3D::cCam2_Line_2Dto3D(const std::vector<cSensorCamPC *> & aVCam,cPhotogrammetricProject * aPhProj) :
     mSeg3d        (cPt3dr(0,0,0),cPt3dr(1,1,1)),
     mSeg3dIsInit (false)
{
    for (const auto & aCam : aVCam)
    {
         const std::string & aNameIm = aCam->NameImage();
         if (aPhProj->HasFileLines(aNameIm))
         {
             cLinesAntiParal1Im   aSetL  = aPhProj->ReadLines(aNameIm);
             const std::vector<cOneLineAntiParal> & aVL  = 	aSetL.mLines;

             // At this step we dont handle multiple lines
             if (aVL.size()==1)
             {
                 //cOneData_L23 aData;

                // comput seg, and correct it if we are at this step
                tSeg2dr aSeg = aVL.at(0).mSeg;
              //  aSeg =  CorrMesSeg(aCam,aSeg);

                //  memorize plane, seg and cam
                mVPlane.push_back(aCam->SegImage2Ground(aSeg));
                mVCamOk.push_back(aCam);
                mVSegOk.push_back(aSeg);
             }
         }
    }

#if (0)
    if (mVPlane.size()>=2)
    {
        mSeg3dIsInit = true;
        mSeg3d =  tSegComp3dr (cPlane3D::InterPlane(mVPlane));

        for (size_t aKS=0 ; aKS<mVSegOk.size() ; aKS++)
        {
            //std::vector<cPt2dr> aVP2{mVSegOk.at(aKS).P1(),mVSegOk.at(aKS).P2()};
            for (const auto & aP2 : {mVSegOk.at(aKS).P1(),mVSegOk.at(aKS).P2()})
            {
                tSeg3dr aBundle =  mVCamOk.at(aKS)->Image2Bundle(aP2);
                cPt3dr aABC;
                BundleInters(aABC,mSeg3d,aBundle,1.0);
                tREAL8 aAbsc = aABC.x();
                mIntervAbsc.Add(aAbsc);
 // eUseIt(aPInter);
            }
        }
    }
#endif

}

void cCam2_Line_2Dto3D::AssertSeg3dIsInit() const
{
    MMVII_INTERNAL_ASSERT_always(mSeg3dIsInit,"cCam2_Line_2Dto3D::AssertSeg3dIsInit");
}
const tSegComp3dr & cCam2_Line_2Dto3D::Seg3d () const
{
   return mSeg3d;
}
/** in cUK_Line3D_4BA with put data in a specific class to allow copy (in "OnUpdate"),
 *  which would be forbiden due to inheritance */

struct cUK_Line3D_4BA_Data
{
    cUK_Line3D_4BA_Data (const cPt3dr & aP1,const cPt3dr & aP2);
    void Update() ;
    void  PushObs(std::vector<double>&);

    tSegComp3dr mSeg;        //< the segement itself
    cPt3dr      mNorm_x;     //< the first vector normal
    cPt3dr      mNorm_y;     //< the second vector normal
    cPt2dr      mUkN1;       //< unknown displacement at Seg.P1, coded as "Uk1.x Nx+ Uk1.y Ny"
    cPt2dr      mUkN2;       //<  unknown displacement at Seg.P2
};

/// class handling a 3D unknown line for bundle adjusment
class cUK_Line3D_4BA :   public cObjWithUnkowns<tREAL8>
{
    public :
         //<  constructor,
         cUK_Line3D_4BA(const cPt3dr & aP1,const cPt3dr & aP2);
         //< called to fill the "obs" in an equation
         void PushObs(std::vector<double>&);

    private :
 

         /// "reaction" after linear update
         void OnUpdate() override;                 
         /// method called when the object must indicate its unknowns
         void PutUknowsInSetInterval() override;

         cUK_Line3D_4BA_Data* mData;
};


/* *********************************************************** */
/*                                                             */
/*                 cUK_Line3D_4BA_Data                         */
/*                                                             */
/* *********************************************************** */

cUK_Line3D_4BA_Data::cUK_Line3D_4BA_Data (const cPt3dr & aP1,const cPt3dr & aP2) :
    mSeg (aP1,aP2),
    mUkN1 (0.0,0.0),
    mUkN2 (0.0,0.0)
{
    tRotR  aRot = tRotR::CompleteRON(mSeg.Tgt());
    mNorm_x = aRot.AxeJ();
    mNorm_y = aRot.AxeK();
}

void cUK_Line3D_4BA_Data::Update() 
{
    cPt3dr aNewP1 = mSeg.P1() + mNorm_x * mUkN1.x()  +  mNorm_y * mUkN1.y();
    cPt3dr aNewP2 = mSeg.P2() + mNorm_x * mUkN2.x()  +  mNorm_y * mUkN2.y();

    *this = cUK_Line3D_4BA_Data(aNewP1,aNewP2);

}
void  cUK_Line3D_4BA_Data::PushObs(std::vector<double>& aVObs)
{
    mSeg.P1().PushInStdVector(aVObs);
    mSeg.P2().PushInStdVector(aVObs);
    mNorm_x.PushInStdVector(aVObs);
    mNorm_y.PushInStdVector(aVObs);

}

/* *********************************************************** */
/*                                                             */
/*                 cUK_Line3D_4BA                              */
/*                                                             */
/* *********************************************************** */

cUK_Line3D_4BA::cUK_Line3D_4BA(const cPt3dr & aP1,const cPt3dr & aP2) :
    mData (nullptr)
{
}

void cUK_Line3D_4BA::PutUknowsInSetInterval()
{
   mSetInterv->AddOneInterv(mData->mUkN1);
   mSetInterv->AddOneInterv(mData->mUkN2);
}

void cUK_Line3D_4BA::OnUpdate() 
{
    mData->Update();
}

void cUK_Line3D_4BA::PushObs(std::vector<double>& aVObs)
{
    mData->PushObs(aVObs);
}


};
