#ifndef TASKCORREL_H
#define TASKCORREL_H

#include <stdio.h>
#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../InitOutil.h"
#include "../DrawOnMesh.h"


const double TT_SEUIL_SURF_TRIANGLE = 100;   //min surface du triangle projecte en img
const double TT_SEUIL_RESOLUTION = DBL_MIN;  //min resolution du triangle reprojecte on img

class cAppliTaskCorrel;
class cAppliTaskCorrelByXML;
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
    void lireMesh(std::string & aNameMesh, vector<triangle*> &, vector<cTriForTiepTri*> &);
    void updateVTriFWithNewAppli(vector<triangle*> & tri);
    PlyFile * Ply() {return mPly;}
    const std::string Ori() {return mOri;}
    const std::string Dir() {return mDir;}
    vector<triangle*> & VTri() {return mVTri;}
    vector<cTriForTiepTri*> & VTriF() {return mVTriF;}
    cImgForTiepTri* DoOneTri(int aNumT);
    void DoAllTri();
    int  NInter()    {return mNInter;}
    void SetNInter(int & aNInter, double &aZoomF);
    vector<Video_Win*> & VVW() {return mVVW;}
    vector<cXml_TriAngulationImMaster> & VTask() {return mVTask;}
    vector<int> & Cur_Img2nd() {return mCur_Img2nd;}
    void ExportXML(Pt3dr clIni = Pt3dr(255,255,255));

private:
    cInterfChantierNameManipulateur * mICNM;
    const string mDir;
    const string mOri;
    int  mNInter;
    double  mZoomF;
    vector<cImgForTiepTri*> mVImgs;
    vector<triangle*> mVTri;
    vector<cTriForTiepTri*> mVTriF;
    PlyFile * mPly;
    vector<Video_Win*> mVVW;
    vector<cXml_TriAngulationImMaster> mVTask;
    vector<int> mCur_Img2nd;
    string mDirXML;
    int cptDel;

};
//  ============================= cAppliTaskCorrelByXML==========================
class cAppliTaskCorrelByXML
{
public:
    cAppliTaskCorrelByXML(const std::string & xmlFile,
                          cInterfChantierNameManipulateur * aICNM,
                          const std::string & aDir,
                          const std::string & anOri,
                          const std::string & aPatImg,
                          const std::string & aPathMesh);
    vector<cImgForTiepTri*> VImgs() {return mVImgs;}
    vector<triangle*> & VTri() {return mVTri;}
    vector<cTriForTiepTri*> & VTriF() {return mVTriF;}
    vector<cXml_TriAngulationImMaster> Cur_VTask() {return Cur_mVTask;}
    vector<cXml_TriAngulationImMaster> & VTask() {return mVTask;}
    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    vector<string> VNImgs() {return mVNImgs;}
    void importXML(string XmlFile);
    void DoAllCpl();
    vector<cXml_TriAngulationImMaster> DoACpl(CplString aCpl);
    void filterCplProcess(vector<CplString> & mVCplImg, vector<string> & mVNImgs);
private:
    cInterfChantierNameManipulateur * mICNM;
    vector<cImgForTiepTri*> mVImgs;
    vector<triangle*> mVTri;
    vector<cTriForTiepTri*> mVTriF;
    vector<cXml_TriAngulationImMaster> Cur_mVTask;
    vector<cXml_TriAngulationImMaster> mVTask;
    vector<string> mVNImgs;
    vector<CplString> mVCplImg;
    string mXmlFile;
    string mDir;
    string mOri;
    string mPathMesh;
};


//  ============================== cImgForTiepTri ==========================
class cImgForTiepTri
{
public:
        cImgForTiepTri(cAppliTaskCorrel* , string aNameIm, int aNum);
        CamStenope * Cam() {return mCam;}
        cAppliTaskCorrel * Appli() {return mAppli;}
        bool inside(Pt2dr aPt, double aRab = 0);
        Tiff_Im Tif() {return mTif;}
        Pt2di Sz() {return mSz;}
        cXml_TriAngulationImMaster & Task() {return mTask;}
        string & Name() {return mName;}
        int Num() {return mNum;}


private:
        int mNum;
        cAppliTaskCorrel * mAppli;
        CamStenope * mCam;
        Tiff_Im mTif;
        Pt2di mSz;
        string mName;
        cXml_TriAngulationImMaster mTask;
};

//  ============================== cTriForTiepTri ==========================
class cTriForTiepTri
{
public:
        cTriForTiepTri(cAppliTaskCorrel* , triangle * aTri3d);
        bool reprj(int aNumImg);
        bool rprjOK() {return mrprjOK;}
        Pt2dr Pt1() {return mPt1;}
        Pt2dr Pt2() {return mPt2;}
        Pt2dr Pt3() {return mPt3;}
        double valElipse(int & aNInter);
private:
        int mNumImg;
        Pt2dr mPt1;
        Pt2dr mPt2;
        Pt2dr mPt3;
        cAppliTaskCorrel * mAppli;
        triangle * mTri3D;
        bool mrprjOK;
};
#endif





