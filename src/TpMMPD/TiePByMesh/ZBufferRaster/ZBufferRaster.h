#include "../InitOutil.h"
#include "StdAfx.h"

const double TT_DEFAULT_PROF_NOVISIBLE  = -1.0;
const double TT_SEUIL_SURF = 100;
const double TT_SCALE_1 = 1.0;
const double TT_DISTMAX_NOLIMIT = -1.0;


typedef double                    tElZBuf;
typedef Im2D<tElZBuf,tElZBuf>     tImZBuf;
typedef TIm2D<tElZBuf,tElZBuf>    tTImZBuf;

class cImgForTiepTri;
class cImgZBuffer;
class cTri3D;
class cTri2D;

class cAppliZBufferRaster
{
public:
    cAppliZBufferRaster(
                        cInterfChantierNameManipulateur *,
                        const std::string & aDir,
                        const std::string & anOri,
                        vector<cTri3D> & aVTri,
                        vector<string> & aVImg

                       );

    cInterfChantierNameManipulateur * ICNM() {return mICNM;}
    const std::string &               Ori() const {return mOri;}
    const std::string &               Dir() const {return mDir;}
    vector<cTri3D> &                  VTri() {return mVTri;}
    vector<string> &                  VImg() {return mVImg;}
    int  &                            NInt() {return mNInt;}
    Pt2di &                           SzW() {return mSzW;}
    double &                          Reech() {return mReech;}
    double &                          DistMax() {return mDistMax;}
    vector< vector<bool> >            TriValid() {return mTriValid;}
    vector< vector<double> >          IndTriValid() {return mIndTriValid;}
    bool &                            WithImgLabel(){return mWithImgLabel;}

    void                              DoAllIm();
    void                              DoAllIm(vector<vector<bool> > &aVTriValid);
    void                              DoAllIm(vector<cImgForTiepTri*> & aVImgTiepTri); //reserve for TaskCorrel


private:
    cInterfChantierNameManipulateur * mICNM;
    std::string                       mDir;
    std::string                       mOri;
    vector<cTri3D>                    mVTri;
    vector<string>                    mVImg;
    int                               mNInt;
    Video_Win *                       mW;
    Video_Win *                       mWLbl;

    Pt2di                             mSzW;
    double                            mReech;
    double                            mDistMax;
    bool                              mWithImgLabel;
    vector< vector<bool> >            mTriValid;
    vector< vector<double> >          mIndTriValid;
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
    cTri2D reprj(CamStenope * aCam);
    double dist2Cam(CamStenope * aCam);


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

    void SetReech(double & scale);

    void calVBasis();
    Pt3dr pt3DFromVBasis(Pt2dr & ptInTri2D, cTri3D & aTri3D);
    double profOfPixelInTri(Pt2dr & ptInTri2D, cTri3D & aTri3D, CamStenope * aCam);

    bool orientToCam(CamStenope * aCam);
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
};

class cImgZBuffer
{
public:
    cImgZBuffer(cAppliZBufferRaster *anAppli , const std::string& aNameIm);

    cAppliZBufferRaster * Appli() {return mAppli;}
    const string & NameIm() {return mNameIm;}
    CamStenope * Cam() {return mCam;}
    tImZBuf & ImZ() {return mImZ;}
    tTImZBuf & TImZ() {return mTImZ;}
    tImZBuf & ImInd() {return mImInd;}
    int & CntTriValab() {return mCntTriValab;}
    int & CntTriTraite() {return mCntTriTraite;}

    Tiff_Im &  Tif() {return mTif;}

    vector<bool> &   TriValid() {return mTriValid;}
    vector<double> & IndTriValid() {return mIndTriValid;}

    void LoadTri(cTri3D);
    void updateZ(tImZBuf & , Pt2dr & , double & prof_val, double & ind_val);
    void normalizeIm(tImZBuf & aIm, double valMin, double valMax);


private:
    cAppliZBufferRaster * mAppli;
    std::string    mNameIm;
    Tiff_Im        mTif;
    Pt2di          mSzIm;
    CamStenope *   mCam;

    tImZBuf        mImZ;
    tTImZBuf       mTImZ;

    tImZBuf        mImInd;

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



