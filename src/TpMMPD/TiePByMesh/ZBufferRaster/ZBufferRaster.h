#include "../InitOutil.h"
#include "StdAfx.h"

const double TT_DEFAULT_PROF_NOVISIBLE  = -1.0;
const double TT_SEUIL_SURF = 100;
// => Devenir MD_SEUIL_SURF_TRIANGLE, passer comme parametre d'entree
const double TT_SCALE_1 = 1.0;
const double TT_DISTMAX_NOLIMIT = -1.0;

typedef double                    tElZBuf;
typedef Im2D<tElZBuf,tElZBuf>     tImZBuf;
typedef TIm2D<tElZBuf,tElZBuf>    tTImZBuf;

class cImgForTiepTri;
class cImgZBuffer;
class cTri3D;
class cTri2D;

template <typename T> bool comparatorPt2dY (Pt2d<T> const &l, Pt2d<T> const &r);
template <typename T> void sortDescendPt2dY(vector<Pt2d<T>> & input);

class cParamZbufferRaster
{
    public :
        cParamZbufferRaster();
        bool        mFarScene;
        string      mPatFIm, mMesh, mOri;
        int         mInt;
        Pt2di       mSzW;
        int         mrech;
        double      mDistMax;
        bool        mWithLbl;
        bool        mNoTif;
        int         mMethod;
        double      MD_SEUIL_SURF_TRIANGLE;
        double      mPercentVisible;
        bool        mSafe;
        bool        mInverseOrder;
};


class cAppliZBufferRaster
{
public:
    cAppliZBufferRaster(
                        cInterfChantierNameManipulateur *,
                        const std::string & aDir,
                        const std::string & anOri,
                        vector<cTri3D> & aVTri,
                        vector<string> & aVImg,
                        bool aNoTif,
                        cParamZbufferRaster aParam
                       );

    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    const std::string &               Ori() const {return mOri;}
    const std::string &               Dir() const {return mDir;}
    void                              SetNameMesh(string & aNameMesh);
    vector<cTri3D> &                  VTri() {return mVTri;}
    vector<string> &                  VImg() {return mVImg;}
    int  &                            NInt() {return mNInt;}
    Pt2di &                           SzW() {return mSzW;}
    double &                          Reech() {return mReech;}
    double &                          DistMax() {return mDistMax;}
    vector< vector<bool> >            TriValid() {return mTriValid;}
    vector< vector<double> >          IndTriValid() {return mIndTriValid;}
    bool &                            WithImgLabel(){return mWithImgLabel;}
    bool &                            IsTmpZBufExist() {return mIsTmpZBufExist;}
    void                              DoAllIm();
    void                              DoAllIm(vector<vector<bool> > &aVTriValid);
    void                              DoAllIm(vector<cImgForTiepTri*> & aVImgTiepTri); //reserve for TaskCorrel

    int & Method() {return mMethod;}
    double & SEUIL_SURF_TRIANGLE() {return MD_SEUIL_SURF_TRIANGLE;}

    vector<Pt2di> &                    AccNbImgVisible(){return mAccNbImgVisible;}
    cParamZbufferRaster &              Param() {return mParam;}

    vector<bool>       &               vImgVisibleFarScene() {return mvImgVisibleFarScene;}

private:
    cInterfChantierNameManipulateur * mICNM;
    std::string                       mDir;
    std::string                       mOri;
    std::string                       mNameMesh;
    vector<cTri3D>                    mVTri;
    vector<string>                    mVImg;
    int                               mNInt;
    Video_Win *                       mW;
    Video_Win *                       mWLbl;

    Pt2di                             mSzW;
    double                            mReech;
    double                            mDistMax;
    bool                              mWithImgLabel;
    bool                              mIsTmpZBufExist;
    vector< vector<bool> >            mTriValid;
    vector< vector<double> >          mIndTriValid;
    bool                              mNoTif;

    int                               mMethod;
    double                            MD_SEUIL_SURF_TRIANGLE;

    vector<Pt2di>                     mAccNbImgVisible; // couple (ind, acc)
    vector<bool>                      mvImgVisibleFarScene; // des image voir "far scene"
    cParamZbufferRaster               mParam;
};

class cTri3D
{
public:
    cTri3D(Pt3dr P1, Pt3dr P2, Pt3dr P3);
    cTri3D(Pt3dr P1, Pt3dr P2, Pt3dr P3, int ind);
    bool IsLoaded() {return mIsLoaded;}
    const Pt3dr & P1() const {return mP1;}
    const Pt3dr & P2() const {return mP2;}
    const Pt3dr & P3() const {return mP3;}
    Pt3dr & Vec_21() {return mVec_21;}
    Pt3dr & Vec_31() {return mVec_31;}
    bool  & HaveBasis() {return mHaveBasis;}
    double   & Ind() {return mInd;}

    void calVBasis();
    cTri2D reprj(cBasicGeomCap3D * aCam);
    cTri2D reprj(cBasicGeomCap3D * aCam, bool & OK);
    double dist2Cam(cBasicGeomCap3D * aCam);


private:
    Pt3dr mP1;
    Pt3dr mP2;
    Pt3dr mP3;
    Pt3dr mCtr;
    bool  mIsLoaded;
    Pt3dr mVec_21;
    Pt3dr mVec_31;
    bool  mHaveBasis;

    double   mInd;
};

class cTri2D
{
public:
    cTri2D(Pt2dr P1, Pt2dr P2, Pt2dr P3);
    cTri2D();
    bool & IsInCam() {return mIsInCam;}
    const Pt2dr & P1() const {return mP1;}
    const Pt2dr & P2() const {return mP2;}
    const Pt2dr & P3() const {return mP3;}
    static cTri2D Default();
    bool & HaveBasis() {return mHaveBasis;}
    bool & InverseOrder() {return mInverseOrder;}


    void SetReech(double & scale);

    void calVBasis();
    Pt3dr pt3DFromVBasis(Pt2dr & ptInTri2D, cTri3D & aTri3D);
    double profOfPixelInTri(Pt2dr & ptInTri2D, cTri3D & aTri3D, cBasicGeomCap3D * aCam, bool aSafe = true);

    bool orientToCam(cBasicGeomCap3D * aCam);
    double surf();

private:
    Pt2dr mP1;
    Pt2dr mP2;
    Pt2dr mP3;
    Pt2dr mVec_21;
    Pt2dr mVec_31;
    bool  mIsInCam;
    double mReech;
    bool mHaveBasis;
    bool mInverseOrder; // we don't know order of triangle's vertice in mesh
};

class cImgZBuffer
{
public:
    cImgZBuffer(cAppliZBufferRaster *anAppli , const std::string& aNameIm, bool & aNoTif, int aInd = -1);

    cAppliZBufferRaster * Appli() {return mAppli;}
    const string & NameIm() {return mNameIm;}
    cBasicGeomCap3D * CamGen() {return mCamGen;}
    tImZBuf & ImZ() {return mImZ;}
    tTImZBuf & TImZ() {return mTImZ;}
    tImZBuf & ImInd() {return mImInd;}
    tTImZBuf & TImInd() {return mTImInd;}


    int & CntTriValab() {return mCntTriValab;}
    int & CntTriTraite() {return mCntTriTraite;}

    Tiff_Im &  Tif() {return mTif;}

    vector<bool> &   TriValid() {return mTriValid;}
    vector<double> & IndTriValid() {return mIndTriValid;}

    void LoadTri(cTri3D);
    void updateZ(tImZBuf & , Pt2dr & , double & prof_val, double & ind_val);
    void normalizeIm(tImZBuf & aIm, double valMin, double valMax);
    void ImportResult(string & fileTriLbl, string & fileImgZBuf);

    int & Ind(){return mInd;}
private:
    cAppliZBufferRaster * mAppli;
    std::string    mNameIm;
    int            mInd;
    Tiff_Im        mTif;
    Pt2di          mSzIm;
    cBasicGeomCap3D *   mCamGen;

    tImZBuf        mImZ;
    tTImZBuf       mTImZ;

    tImZBuf         mImInd;
    tTImZBuf        mTImInd;

    Im2D_Bits<1>   mMasqTri;
    TIm2DBits<1>   mTMasqTri;
    Im2D_Bits<1>   mMasqIm;
    TIm2DBits<1>   mTMasqIm;


    Video_Win *    mW;

    int            mCntTri;
    int            mCntTriValab;
    int            mCntTriTraite;


    vector<bool>   mTriValid;
    vector<double> mIndTriValid;

};



