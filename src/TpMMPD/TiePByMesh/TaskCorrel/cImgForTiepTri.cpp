#include "TaskCorrel.h"

//  ============================== cImgForTiepTri ==========================


cImgForTiepTri::cImgForTiepTri(cAppliTaskCorrel * anAppli, string aNameIm, int aNum):
    mNum    (aNum),
    mAppli  (anAppli),
    mCam    (anAppli->ICNM()->StdCamOfNames(aNameIm, anAppli->Ori())),
    mTif    (Tiff_Im::StdConv(mAppli->Dir() + aNameIm)),
    mSz     (mTif.sz()),
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

