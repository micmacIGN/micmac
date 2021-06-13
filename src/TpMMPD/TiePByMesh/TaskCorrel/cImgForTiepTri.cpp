#include "TaskCorrel.h"

//  ============================== cImgForTiepTri ==========================


cImgForTiepTri::cImgForTiepTri(cAppliTaskCorrel * anAppli, string aNameIm, int aNum, bool aNoTif):
    mNum    (aNum),
    mAppli  (anAppli),
    mCamGen (anAppli->ICNM()->StdCamGenerikOfNames(anAppli->Ori(),aNameIm)),
    mCamSten (mCamGen->DownCastCS()),
    mTif    (Tiff_Im::UnivConvStd(mAppli->Dir() + aNameIm)),
    mSz     (round_ni(mCamGen->SzPixel())),
    mName   (aNameIm),
    mImgWithGCP (false)
{
    mTask.NameMaster() = aNameIm;
    mTaskWithGCP.NameMaster() = aNameIm;
}


bool cImgForTiepTri::inside(Pt2dr aPt, double aRab)
{
    return     (aPt.x - aRab >= 0)
            && (aPt.y - aRab >= 0)
            && (aPt.x + aRab < mSz.x)
            && (aPt.y + aRab < mSz.y);
}


