#ifndef TASKCORREL_H
#define TASKCORREL_H

#include <stdio.h>
#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/TiepTri.h"
#include "../InitOutil.h"
#include "../DrawOnMesh.h"
#include "../ZBufferRaster/ZBufferRaster.h"


const double TT_SEUIL_SURF_TRIANGLE = 100;   //min surface du triangle projecte en img
// => Devenir MD_SEUIL_SURF_TRIANGLE, passer comme parametre d'entree

const double TT_SEUIL_RESOLUTION = DBL_MIN;  //min resolution du triangle reprojecte on img
const int TT_DEF_SCALE_ZBUF = 2;        //default deZoom pour calculer ZBuffer


class cAppliTaskCorrel;
class cAppliTaskCorrelByXML;
class cImgForTiepTri;
class cTriForTiepTri;

struct cMesureStruct
{
    string aNameIm;
    int aIndImg;
    vector<Pt2dr> aVPts;
    vector<string> aNamePt;
    vector<int> aVIndTri;
};
//  ============================= cAppliTaskCorrel ==========================
class cParamAppliTaskCorrel
{
    public:
    cParamAppliTaskCorrel(cInterfChantierNameManipulateur * aICNM,
                                                 const std::string & aDir,
                                                 const std::string & aOri,
                                                 const std::string & aPatImg,
                                                 bool & aNoTif,
                                                 string aMesureXML,
                                                 bool aInverseOrder
                            );
        cInterfChantierNameManipulateur * pICNM;
        string pDir;
        string pOri;
        string pPatImg;
        bool pNoTif;
        string pMesureXML;
        bool aInverseOrder;
};

// ==========================================================================


//  ============================= cAppliTaskCorrel ==========================
class cAppliTaskCorrel
{
public:
    cAppliTaskCorrel (cInterfChantierNameManipulateur * aICNM,
                                         const std::string & aDir,
                                         const std::string & aOri,
                                         const std::string & aPatImg,
                                         bool & aNoTif
                                       );
    cAppliTaskCorrel (cInterfChantierNameManipulateur *,
                       const std::string & aDir,
                       const std::string & anOri,
                       const std::string & aPatImg,
                       bool &aNoTif,
                       cParamAppliTaskCorrel * aParam
                      );
    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    vector<cImgForTiepTri*> VImgs() {return mVImgs;}
    void lireMesh(std::string & aNameMesh);
    void updateVTriFWithNewAppli(vector<cTri3D> &tri);
    PlyFile * Ply() {return mPly;}
    const std::string Ori() {return mOri;}
    const std::string Dir() {return mDir;}
    string & NameMesh() {return mNameMesh;}
    vector<triangle*> & VTri() {return mVTri;}
    vector<cTriForTiepTri*> & VTriF() {return mVTriF;}
    cImgForTiepTri* DoOneTri(cTriForTiepTri * aTri2D);
    void DoAllTri();
    int  NInter()    {return mNInter;}
    void SetNInter(int & aNInter, double &aZoomF);
    vector<Video_Win*> & VVW() {return mVVW;}
    vector<cXml_TriAngulationImMaster> & VTask() {return mVTask;}
    vector<cXml_TriAngulationImMaster_WithPts> & VTaskWithGCP() {return mVTaskWithGCP;}

    vector<int> & Cur_Img2nd() {return mCur_Img2nd;}
    void ExportXML(string aDirXML, Pt3dr clIni = Pt3dr(255,255,255));

    vector<cTri3D> & VcTri3D() {return mVcTri3D;}
    double & DistMax() {return mDistMax;}
    double & Rech() {return mRech;}

    void ZBuffer();
    bool & KeepAll2nd(){return mKeepAll2nd;}

    double & SEUIL_SURF_TRIANGLE() {return MD_SEUIL_SURF_TRIANGLE;}
    int & MethodZBuf() {return mMethodZBuf;}

    void ReadXMLMesurePts(string aGCPMesureXML, vector<cImgForTiepTri *> &mVImgs);
    bool & WithGCP() {return mWithGCP;}

    void GetIndTriHasGCP();

    bool & SafeZBuf() {return mSafeZBuf;}

    bool & ZBuf_InverseOrder() {return mZBuf_InverseOrder;}

private:
    cInterfChantierNameManipulateur * mICNM;
    const string mDir;
    const string mOri;
    string mNameMesh;
    int  mNInter;
    double  mZoomF;
    vector<cImgForTiepTri*> mVImgs;
    vector<string> mVName;
    vector<triangle*> mVTri;
    vector<cTriForTiepTri*> mVTriF;
    PlyFile * mPly;
    vector<Video_Win*> mVVW;
    vector<cXml_TriAngulationImMaster> mVTask;
    vector<int> mCur_Img2nd;
    string mDirXML;
    int cptDel;
    vector<cTri3D> mVcTri3D;
    double mDistMax;
    double mRech;
    bool   mNoTif;

    vector< vector<bool> > mVTriValid;
    vector< vector<int> > mVIndTriValid;
    cSauvegardeNamedRel  mRelIm;

    bool   mKeepAll2nd;
    double MD_SEUIL_SURF_TRIANGLE;
    int    mMethodZBuf;
    bool   mWithGCP;
    vector<cXml_TriAngulationImMaster_WithPts> mVTaskWithGCP;

    bool mSafeZBuf;
    bool mZBuf_InverseOrder;


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
                          const std::string & aPathMesh,
                          bool mNoTif
                          );
    vector<cImgForTiepTri*> VImgs() {return mVImgs;}
    vector<triangle*> & VTri() {return mVTri;}
    vector<cTriForTiepTri*> & VTriF() {return mVTriF;}
    vector<cXml_TriAngulationImMaster> Cur_VTask() {return Cur_mVTask;}
    vector<cXml_TriAngulationImMaster> & VTask() {return mVTask;}
    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    vector<string> VNImgs() {return mVNImgs;}
    vector<Pt2di> & CplValidIndex() {return mCplValidIndex;}
    void importXML(string XmlFile);
    void ExportXML(string & aXMLOut);
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
    vector<Pt2di> mCplValidIndex;
    string mXmlFile;
    string mDir;
    string mOri;
    string mPathMesh;
    bool mNoTif;
};


//  ============================== cImgForTiepTri ==========================
class cImgForTiepTri
{
public:
        cImgForTiepTri(cAppliTaskCorrel* , string aNameIm, int aNum, bool aNoTif);

        CamStenope * CamSten()     {ELISE_ASSERT(mCamSten!=0,"cImgForTiepTri::CamSten") return mCamSten;}
        CamStenope * CamStenSVP()  {return mCamSten;}
        cBasicGeomCap3D * CamGen() {return mCamGen;}


        cAppliTaskCorrel * Appli() {return mAppli;}
        bool inside(Pt2dr aPt, double aRab = 0);
        Tiff_Im Tif() {return mTif;}
        Pt2di Sz() {return mSz;}
        cXml_TriAngulationImMaster & Task() {return mTask;}
        cXml_TriAngulationImMaster_WithPts & TaskWithGCP() {return mTaskWithGCP;}

        string & Name() {return mName;}
        int Num() {return mNum;}

        vector<bool> &  TriValid() {return mTriValid;}
        vector<int>  &  IndTriValid() {return mIndTriValid;}
        cMesureStruct & Mesure() {return aMesure;}

        bool & ImgWithGCP() {return mImgWithGCP;}

private:
        int mNum;
        cAppliTaskCorrel * mAppli;

        cBasicGeomCap3D * mCamGen;
        CamStenope *      mCamSten;


        Tiff_Im mTif;
        Pt2di mSz;
        string mName;
        cXml_TriAngulationImMaster mTask;

        vector<bool>   mTriValid;
        vector<int>    mIndTriValid;
        cMesureStruct aMesure;
        cXml_TriAngulationImMaster_WithPts mTaskWithGCP;
        bool mImgWithGCP;
};

//  ============================== cTriForTiepTri ==========================
class cTriForTiepTri
{
public:

        //cTriForTiepTri(cAppliTaskCorrel* , triangle * aTri3d, double & ind);
        cTriForTiepTri(cAppliTaskCorrel* , cTri3D aTri3d, int & ind);
        bool reprj(cImgForTiepTri *aImg);
        bool reprj_pure(cImgForTiepTri * aImg, Pt2dr & P1, Pt2dr & P2, Pt2dr & P3);
        bool rprjOK() {return mrprjOK;}
        Pt2dr Pt1() {return mPt1;}
        Pt2dr Pt2() {return mPt2;}
        Pt2dr Pt3() {return mPt3;}
        int & Ind() {return mInd;}
        double valElipse(int & aNInter);
private:
        int mNumImg;
        Pt2dr mPt1;
        Pt2dr mPt2;
        Pt2dr mPt3;
        cAppliTaskCorrel * mAppli;
        triangle * mTri3D;
        cTri3D mTri3D_;
        bool mrprjOK;
        int mInd;
};
#endif





