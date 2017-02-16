#include "StdAfx.h"

typedef Im2D<double,double> tImgEsSim;

/*====================== cParamEsSim ========================*/
class cParamEsSim
{
    public:
        cParamEsSim(string & aDir, string & aImgX, string & aImgY, Pt2dr & aPtCtr, int & aSzW, Pt3di & aDispParam, int & nInt);
        string mDir;
        string mImgX;
        string mImgY;
        Pt2dr mPtCtr;
        int mSzW;
        Pt2di mDispSz;
        int mZoom;
        int mInt;
};

cParamEsSim::cParamEsSim(string & aDir, string & aImgX, string & aImgY, Pt2dr & aPtCtr, int & aSzW, Pt3di & aDispParam, int & nInt):
    mDir (aDir),
    mImgX (aImgX),
    mImgY (aImgY),
    mPtCtr (aPtCtr),
    mSzW (aSzW),
    mDispSz (Pt2di(aDispParam.x, aDispParam.y)),
    mZoom  (aDispParam.z),
    mInt   (nInt)
{}

/*====================== cAppliEsSim ========================*/
class cAppliEsSim
{
public:
    cAppliEsSim(cParamEsSim * aParam);
    cParamEsSim * Param() {return mParam;}
private:
    cParamEsSim * mParam;
};

/*====================== cImgEsSim ========================*/
class cImgEsSim
{
public:
    cImgEsSim(string & aName, cAppliEsSim * aAppli);
    bool IsInside(Pt2dr & aPt);
    bool getVgt (Pt2dr & aPtCtr, int & aSzw);
private:
    cAppliEsSim * mAppli;
    string mName;
    Tiff_Im mTif;
    tImgEsSim mImgDep;
    Pt2di mDecal;
};


