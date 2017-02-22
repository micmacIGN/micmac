#include "StdAfx.h"

typedef Im2D<double,double> tImgEsSim;
class cAppliEsSim;
class cImgEsSim;

/*====================== cParamEsSim ========================*/
class cParamEsSim
{
    /* string [aDir]: repertoire
     * string [aImgX]: name of the deplacement image of X-axis
     * string [aImgY]: name of the deplacement image of Y-axis
     * Pt2dr [aPtCtr]: coordinate of the center point of the vignette
     * int [aSzW]: size of the vignette
     * Pt3di [aDispParam]: size of the affiche window (Pt2di) and the zoom factor (int)
     * int [nInt]: interaction
     * Pt2di [aNbGrill]: Nb of Grill
     * double [aSclDepl]: Scale when affiche the deplacment vector */
    public:
        cParamEsSim(string & aDir, string & aImgX, string & aImgY, Pt2dr & aPtCtr, int & aSzW, Pt3di & aDispParam, int & nInt, Pt2di & aNbGrill, double & aSclDepl, bool & aSaveImg);
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
        bool mSaveImg;
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
    void creatHomol(cImgEsSim * aImgX, cImgEsSim * aImgY); //create pairs of tie points for all the pixels
    void writeHomol(ElPackHomologue & aPack); //write pairs of tiep oints in a pack
    bool getHomolInVgt (ElPackHomologue & aPack, Pt2dr & aPtCtr, int & aSzw); //get tie points of the vignette and indicate if the vignette is inside the original image
    bool EsSimFromHomolPack (ElPackHomologue & aPack, Pt2dr & rotCosSin, Pt2dr & transXY);//estimate the rotation and translation from deplacement
    bool EsSimAndDisp (Pt2dr & aPtCtr, int & aSzw, Pt2dr & rotCosSin, Pt2dr & transXY);
    vector<Pt2di> VaP0Grill() {return mVaP0Grill;}
    bool EsSimEnGrill(vector<Pt2di> aVPtCtrVig, int & aSzw, Pt2dr & rotCosSinAll, Pt2dr & transXYAll);
private:
    cParamEsSim * mParam;
    cImgEsSim * mImgX;
    cImgEsSim * mImgY;
    ElPackHomologue mHomolDep;
    Video_Win * mWX;
    Video_Win * mWY;
    vector<Pt2di> mVaP0Grill;
    double* aData1;
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




