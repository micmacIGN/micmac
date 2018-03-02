#ifndef TIEPTRIFAR_H
#define TIEPTRIFAR_H

#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../TaskCorrel/TaskCorrel.h"
//#include "../ZBufferRaster/ZBufferRaster.h"
#include <stack>
#include <iostream>

extern bool convexHull(vector<Pt2dr> points, stack<Pt2dr> &S);

class cParamTiepTriFar;
class cAppliTiepTriFar;
class cImgTieTriFar;
class ExtremePoint;
class cResultMatch;
class cParamMatch;

class cResultMatch
{
    public:
        cResultMatch();
        bool IsInit;        // if this instant is updated ?
        Pt2dr aPtMatched;   // Pt found on img 2 as a match with aPtToMatch
        Pt2dr aPtToMatch;   // Pt on img 1 (origin pt to be matched)
        double aScore;      // Last correlation score
        int aNbUpdate;  // Nb of times this instant updated
        int update(Pt2dr & aPtOrg, Pt2dr & aPtMatched, double & aScore);

};

class cParamMatch
{
    public:
        cParamMatch();
        cParamMatch(int & aSzWin, double & aThres, double & aStep);
        int aSzWin; // correlation win size (1/2)
        double aThres; // reject correlation score
        double aStep; // pixel sample step
};


class cParamTiepTriFar
{
    public:
        cParamTiepTriFar();
        bool aDisp;
        double  aZoom;
        Pt2di aSzW;
        bool aDispVertices;
        double aRad;
        string aDirZBuf;
        string aNameMesh;
};

class cAppliTiepTriFar
{
    public:
        cAppliTiepTriFar (cParamTiepTriFar & aParam,
                          cInterfChantierNameManipulateur * aICNM,
                          vector<string> & vNameImg,
                          string & aDir,
                          string & aOri
                         );

        cParamTiepTriFar & Param() {return mParam;}
        void LoadMesh(string & aMeshName);
        string & Dir() {return mDir;}
        string & Ori() {return mOri;}
        cInterfChantierNameManipulateur * ICNM() {return mICNM;}
        vector <string> & VNameImg() {return mVNameImg;}
        vector<cImgTieTriFar*>  & VImg() {return mvImg;}

        // FROM 3D MESH TO 2D MASK
        void loadMask2D();

        bool FilterContrast();

        cImgTieTriFar * ImgLeastPts() {return mImgLeastPts;}

        vector<cIntTieTriInterest*> & PtToCorrel() {return mPtToCorrel;}

        int Matching();

    private:
        cParamTiepTriFar & mParam;
        vector <string> mVNameImg;
        vector<cImgTieTriFar*>  mvImg;
        string & mDir;
        string & mOri;

        vector<cTri3D> mVTri3D;
        cInterfChantierNameManipulateur * mICNM;

        cImgTieTriFar * mImgLeastPts;

        vector<cIntTieTriInterest*> mPtToCorrel;

};

class cImgTieTriFar
{
  public :
        cImgTieTriFar(cAppliTiepTriFar & aAppli, string & aName);

        string NameIm() {return mNameIm;}

        Tiff_Im   Tif() {return mTif;}

        Pt2di & SzIm() {return mSzIm;}


        cBasicGeomCap3D * CamGen() {return mCamGen;}
        CamStenope *      CamSten() {return mCamSten;}

        vector<Pt2dr> & SetVertices() {return mSetVertices;}

        Video_Win * & VW() {return mVW;}

        Im2D<double, double>  &   ImInit() {return mImInit;}

        TIm2D<double, double> &   TImInit() {return mTImInit;}

        template <typename T> bool IsInside(Pt2d<T> p, Pt2d<T> aRab=Pt2d<T>(0,0));

        Im2D_Bits<1> & MasqIm() {return mMasqIm;}
        TIm2DBits<1> & TMasqIm() {return mTMasqIm;}

        int DetectInterestPts();

        vector<Pt2dr> & InterestPt() {return mInterestPt;}

        vector<cIntTieTriInterest> & InterestPt_v2() {return mInterestPt_v2;}

        Tiff_Im &  TifZBuf() {return mTifZBuf;}

        tImZBuf & ImZBuf() {return mImZBuf;}
        tTImZBuf & TImZBuf() {return mTImZBuf;}

  private :
        cAppliTiepTriFar & mAppli;

        string    mNameIm;

        Tiff_Im   mTif;


        Pt2di     mSzIm;

        Im2D_Bits<1> mMasqIm;
        TIm2DBits<1> mTMasqIm;

        cBasicGeomCap3D * mCamGen;
        CamStenope *      mCamSten;

        vector<Pt2dr> mSetVertices;

        Video_Win * mVW;

        Im2D<double, double>     mImInit;

        TIm2D<double, double>     mTImInit;

        vector<Pt2dr> mInterestPt;

        vector<cIntTieTriInterest> mInterestPt_v2;

        Tiff_Im   mTifZBuf;

        Im2D<double, double>     mImPtch;   //store image region or correlation
        TIm2D<double, double>    mTImPtch;

        tImZBuf mImZBuf;
        tTImZBuf mTImZBuf;
};







#endif // TIEPTRIFAR_H

