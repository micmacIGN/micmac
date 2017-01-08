#include "../InitOutil.h"

const double TT_DEFAULT_PROF_NOVISIBLE  = -1.0;
const double TT_SEUIL_SURF = 100;


typedef double                    tElZBuf;
typedef Im2D<tElZBuf,tElZBuf>     tImZBuf;
typedef TIm2D<tElZBuf,tElZBuf>    tTImZBuf;

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
    void                              DoAllIm();
private:
    cInterfChantierNameManipulateur * mICNM;
    std::string                       mDir;
    std::string                       mOri;
    vector<cTri3D>                    mVTri;
    vector<string>                    mVImg;
    int                               mNInt;
    Video_Win *                       mW;

};

class cTri3D
{
public:
    cTri3D(Pt3dr P1, Pt3dr P2, Pt3dr P3);
    bool IsLoaded() {return mIsLoaded;}
    const Pt3dr & P1() const {return mP1;}
    const Pt3dr & P2() const {return mP2;}
    const Pt3dr & P3() const {return mP3;}

    cTri2D reprj(CamStenope * aCam);

private:
    Pt3dr mP1;
    Pt3dr mP2;
    Pt3dr mP3;
    bool  mIsLoaded;
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

    Pt3dr pt3DFromVBasis(Pt2dr & ptInTri2D, cTri3D & aTri3D);
    double profOfPixelInTri(Pt2dr & ptInTri2D, cTri3D & aTri3D, CamStenope * aCam);

    bool orientToCam(CamStenope * aCam);
    double surf();

private:
    Pt2dr mP1;
    Pt2dr mP2;
    Pt2dr mP3;
    bool  mIsInCam;
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
    int CntTriValab() {return mCntTriValab;}

    void LoadTri(cTri3D);
    bool updateZ(tImZBuf & ImZ, Pt2dr &pxl, double &prof_val);
    void normalizeIm(tImZBuf & aIm, double valMin, double valMax);

private:
    cAppliZBufferRaster * mAppli;
    std::string    mNameIm;
    Tiff_Im        mTif;
    CamStenope *   mCam;
    Pt2di          mSzIm;

    tImZBuf        mImZ;
    tTImZBuf       mTImZ;

    Im2D_Bits<1>   mMasqTri;
    TIm2DBits<1>   mTMasqTri;
    Im2D_Bits<1>   mMasqIm;
    TIm2DBits<1>   mTMasqIm;


    Video_Win *    mW;
    int            mCntTri;
    int            mCntTriValab;
};



