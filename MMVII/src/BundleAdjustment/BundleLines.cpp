#include "BundleAdjustment.h"

namespace MMVII
{

struct cUK_Line3D_4BA_Data
{
    cUK_Line3D_4BA_Data (const cPt3dr & aP1,const cPt3dr & aP2);
    void Update() ;

    tSegComp3dr mSeg;
    cPt3dr      mNorm_x;
    cPt3dr      mNorm_y;
    cPt2dr      mUkN1;
    cPt2dr      mUkN2;
};

/// class handling a 3D unknown line for bundle adjusment
class cUK_Line3D_4BA :   public cObjWithUnkowns<tREAL8>
{
    public :
         cUK_Line3D_4BA(const cPt3dr & aP1,const cPt3dr & aP2);
    private :
 

         /// "reaction" after linear update
         void OnUpdate() override;                 
         void PutUknowsInSetInterval() override;

         cUK_Line3D_4BA_Data mData;
};


/* *********************************************************** */
/*                                                             */
/*                 cUK_Line3D_4BA                              */
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

cUK_Line3D_4BA::cUK_Line3D_4BA(const cPt3dr & aP1,const cPt3dr & aP2) :
    mData (aP1,aP2)
{
}


void cUK_Line3D_4BA::OnUpdate() 
{
    mData.Update();
}



};
