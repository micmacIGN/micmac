#include "EsSimilitude.h"

cImgEsSim::cImgEsSim(string & aName, cAppliEsSim * aAppli):
    mAppli (aAppli),
    mName (aName),
    mTif (Tiff_Im::UnivConvStd(mAppli->Param()->mDir + aName)),
    mImgDep (1,1),
    mDecal  (0,0)
{}

bool cImgEsSim::IsInside(Pt2dr & aPt)
{
    Pt2di aSzImg = mTif.sz();
    if ((aPt.x > aSzImg.x) || (aPt.y > aSzImg.y) || (aPt.x < 0) || (aPt.y < 0))
        return false;
    else
        return true;
}

bool cImgEsSim::getVgt (Pt2dr & aPtCtr, int & aSzw)
{
    Pt2dr aP0(aPtCtr.x-aSzw,aPtCtr.y-aSzw);
    Pt2dr aP1(aPtCtr.x+aSzw,aPtCtr.y+aSzw);
    mDecal = Pt2di(aP0);
    if (IsInside(aP0) && IsInside(aP1))
    {
        mImgDep.Resize(Pt2di(2*aSzw+1,2*aSzw+1));
        ELISE_COPY(mImgDep.all_pts(),trans(mTif.in(0),Pt2di(mDecal)),mImgDep.out());
        return true;
    }
    return false;
}
