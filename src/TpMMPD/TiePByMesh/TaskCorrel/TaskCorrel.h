#ifndef TASKCORREL_H
#define TASKCORREL_H

#include <stdio.h>
#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../InitOutil.h"

const double TT_SEUIL_SURF_TRIANGLE = 100;   //min surface du triangle projecte en img
const double TT_SEUIL_RESOLUTION = DBL_MIN;  //min resolution du triangle reprojecte on img

class cAppliTaskCorrel;
class cImgForTiepTri;
class cTriForTiepTri;

//  ============================= cAppliTaskCorrel ==========================
class cAppliTaskCorrel
{
public:
    cAppliTaskCorrel (cInterfChantierNameManipulateur *,
                       const std::string & aDir,
                       const std::string & anOri,
                       const std::string & aPatImg
                     );
    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    vector<cImgForTiepTri*> VImgs() {return mVImgs;}
    void lireMesh(std::string & aNameMesh);
    PlyFile * Ply() {return mPly;}
    const std::string Ori() {return mOri;}
    const std::string Dir() {return mDir;}
    vector<triangle*> & VTri() {return mVTri;}
    cImgForTiepTri* DoOneTri(triangle * aTri);
    void DoAllTri();
private:
    cInterfChantierNameManipulateur * mICNM;
    const string mDir;
    const string mOri;
    vector<cImgForTiepTri*> mVImgs;
    vector<triangle*> mVTri;
    PlyFile * mPly;
};

//  ============================== cImgForTiepTri ==========================
class cImgForTiepTri
{
public:
        cImgForTiepTri(cAppliTaskCorrel & , string & aNameIm, int aNum);
        CamStenope * Cam() {return mCam;}
        bool inside(Pt2dr aPt, double aRab = 0);
private:
        int mNum;
        cAppliTaskCorrel mAppli;
        CamStenope * mCam;
        Tiff_Im mTif;
        Pt2di mSz;
};

//  ============================== cTriForTiepTri ==========================
class cTriForTiepTri
{
public:
        cTriForTiepTri(cAppliTaskCorrel & , triangle * aTri3d);
        bool reprj(int aNumImg);
        bool rprjOK() {return mrprjOK;}
        double valElipse();
private:
        int mNumImg;
        Pt2dr mPt1;
        Pt2dr mPt2;
        Pt2dr mPt3;
        cAppliTaskCorrel mAppli;
        triangle * mTri3D;
        bool mrprjOK;
};
#endif





