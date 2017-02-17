#include "StdAfx.h"

typedef Im2D<double,double> tImgEsSim;
class cAppliEsSim;
class cImgEsSim;

/*====================== cParamEsSim ========================*/
class cParamEsSim
{
    public:
        cParamEsSim(string & aDir, string & aImgX, string & aImgY, Pt2dr & aPtCtr, int & aSzW, Pt3di & aDispParam, int & nInt, Pt2di & mNbGrill, double & aSclDepl);
        string mDir;
        string mImgX;
        string mImgY;
        Pt2dr mPtCtr;
        int mSzW;
        Pt2di mDispSz;
        int mZoom;
        int mInt;
        Pt2di mNbGrill;
        double mSclDepl;
};

/*====================== cAppliEsSim ========================*/
class cAppliEsSim
{
public:
    cAppliEsSim(cParamEsSim * aParam);
    cParamEsSim * Param() {return mParam;}
    ElPackHomologue & HomolDep() {return mHomolDep;}
    cImgEsSim * ImgX() {return mImgX;}
    cImgEsSim * ImgY() {return mImgY;}
    void creatHomol(cImgEsSim * aImgX, cImgEsSim * aImgY);
    void writeHomol(ElPackHomologue & aPack);
    bool getHomolInVgt (ElPackHomologue & aPack, Pt2dr & aPtCtr, int & aSzw);
    bool EsSimFromHomolPack (ElPackHomologue & aPack, Pt2dr & rotCosSin, Pt2dr & transXY);
    bool EsSimAndDisp (Pt2dr & aPtCtr, int & aSzw, Pt2dr & rotCosSin, Pt2dr & transXY);
    vector<Pt2di> VaP0Grill() {return mVaP0Grill;}
    bool EsSimEnGrill(vector<Pt2di> aVPtCtrVig, int & aSzw, Pt2dr & rotCosSin, Pt2dr & transXY);
private:
    cParamEsSim * mParam;
    cImgEsSim * mImgX;
    cImgEsSim * mImgY;
    ElPackHomologue mHomolDep;
    Video_Win * mWX;
    Video_Win * mWY;
    vector<Pt2di> mVaP0Grill;
};

/*====================== cImgEsSim ========================*/
class cImgEsSim
{
public:
    cImgEsSim(string & aName, cAppliEsSim * aAppli);
    bool IsInside(Pt2dr & aPt);
    bool getVgt (tImgEsSim & aVigReturn, Pt2dr & aPtCtr, int & aSzw);
    Tiff_Im & Tif() {return mTif;}
    tImgEsSim & ImgDep() {return mImgDep;}
    string & Name() {return mName;}
    Pt2di & Decal() {return mDecal;}
    void normalize(tImgEsSim & aImSource,tImgEsSim & aImDest, double rangeMin, double rangeMax);

private:
    cAppliEsSim * mAppli;
    string mName;
    Tiff_Im mTif;
    tImgEsSim mImgDep;
    Pt2di mDecal;
};




