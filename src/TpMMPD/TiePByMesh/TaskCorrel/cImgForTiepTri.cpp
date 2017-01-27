#include "TaskCorrel.h"

//  ============================== cImgForTiepTri ==========================


cImgForTiepTri::cImgForTiepTri(cAppliTaskCorrel * anAppli, string aNameIm, int aNum, bool aNoTif):
    mNum    (aNum),
    mAppli  (anAppli),
    mTif    (aNoTif ? Tiff_Im::StdConv(mAppli->Dir() + "Tmp-MM-Dir/" + aNameIm + "_Ch1.tif"):Tiff_Im::StdConv(mAppli->Dir() + aNameIm)),
    mCam    (anAppli->ICNM()->StdCamOfNames(aNameIm, anAppli->Ori())),
    mSz     (mCam->Sz()),
    mName   (aNameIm)
{
    mTask.NameMaster() = aNameIm;
}


bool cImgForTiepTri::inside(Pt2dr aPt, double aRab)
{
    return     (aPt.x - aRab >= 0)
            && (aPt.y - aRab >= 0)
            && (aPt.x + aRab < mSz.x)
            && (aPt.y + aRab < mSz.y);
}


