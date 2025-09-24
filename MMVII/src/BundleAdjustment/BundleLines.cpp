#include "BundleAdjustment.h"

namespace MMVII
{

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
