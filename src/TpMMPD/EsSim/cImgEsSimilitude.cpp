#include "EsSimilitude.h"

cImgEsSim::cImgEsSim(string & aName, cAppliEsSim * aAppli):
    mAppli (aAppli),
    mName (aName),
    mTif (Tiff_Im::UnivConvStd(mAppli->Param()->mDir + aName)),
    mImgDep (mTif.sz().x, mTif.sz().y),
    mDecal  (0,0)
{
    ELISE_COPY(mImgDep.all_pts(), mTif.in(), mImgDep.out());
}

bool cImgEsSim::IsInside(Pt2dr & aPt)
{
    Pt2di aSzImg = mTif.sz();
    if ((aPt.x > aSzImg.x) || (aPt.y > aSzImg.y) || (aPt.x < 0) || (aPt.y < 0))
        return false;
    else
        return true;
}

bool cImgEsSim::getVgt (tImgEsSim & aVigReturn, Pt2dr & aPtCtr, int & aSzw)
{
    Pt2dr aP0(aPtCtr.x-aSzw,aPtCtr.y-aSzw);
    Pt2dr aP1(aPtCtr.x+aSzw,aPtCtr.y+aSzw);
    mDecal = Pt2di(aP0);
    if (IsInside(aP0) && IsInside(aP1))
    {
        aVigReturn.Resize(Pt2di(2*aSzw+1,2*aSzw+1));
        ELISE_COPY(aVigReturn.all_pts(),trans(mTif.in(0),Pt2di(mDecal)),aVigReturn.out());
        return true;
    }
    return false;
}

void cImgEsSim::normalize(tImgEsSim & aImSource,tImgEsSim & aImDest, double rangeMin, double rangeMax)
{
    double minVal; //min value of ImSource
    double maxVal; //Max value of ImSource

    //find the Min and Max of ImSource
    ELISE_COPY(aImSource.all_pts(),aImSource.in(),VMax(maxVal)|VMin(minVal));
    cout<<"Avant Min/Max : "<<minVal<<"/"<<maxVal<<endl;

    //check if the sizes of ImSource and ImDest are coherent
    ELISE_ASSERT((aImSource.sz().x == aImDest.sz().x && aImSource.sz().y == aImDest.sz().y), "Size not coherent in normalize image");

    //normalize ImSource
    double factor = (rangeMax-rangeMin)/(maxVal-minVal);
    ELISE_COPY(aImSource.all_pts(),(aImSource.in()-minVal)*factor,aImDest.out());

    //find the Min and Max of ImDest
    ELISE_COPY(aImDest.all_pts(),aImDest.in(),VMax(maxVal)|VMin(minVal));
    cout<<"Apres Min/Max : "<<minVal<<"/"<<maxVal<<endl;

}

