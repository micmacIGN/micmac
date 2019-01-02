#include "StdAfx.h"
#include "../../uti_phgrm/TiepTri/MultTieP.h"


#define BEGIN_IMAGE 2
#define A_POINT 3
#define END_IMAGE 1
#define UNKNOW_TYPE -1
#define _TP_PIX4D 1
#define _TP_BINGO 4
#define _TP_ORIMA 6


class cPtMul
{
    public :
        cPtMul(int aId_bingo);
        vector<int> & VIm(){return mVIm;}
        vector<Pt2dr> & VPt(){return mVPt;}
    private :
        int mID_bingo;
        vector<int> mVIm;
        vector<Pt2dr> mVPt;
};


class ConvTiePointPix4DMM
{
  public :
    ConvTiePointPix4DMM();
    bool ImportTiePointFile(string aFile, int & file_type);
    bool ReadBingoFile(string aFile);
    bool ReadPix4DFile(string aFile);
    Pt2dr PtTranslation(Pt2dr aPt); // translate origin of Bingo point; need image size
    cSetTiePMul * SetTiepMul(){return mSetTiep;}
    bool IsPointRegisted(int aIdPoint);
    bool IsImageRegisted(string aNameIm);
    int parseLine(string & aL, vector<string> & aVWord);
    void exportToMMNewFH();
    void exportToMMClassicFH(string aDir, string aOut, bool aBin, bool aIs2Way = false);
    string & SuffixOut(){return mSuffixOut;}
    bool & BinOut(){return mBinOut;}
    double & SzPhotosite(){return mSzPhotosite;}
    int & file_type(){return mfile_type;}
private :
    cSetTiePMul * mSetTiep;
    std::map<int, cPtMul*> mMap_Id_PtMul;
    std::map<string, int> mMap_IdImPmul_NameIm;
    double mSzPhotosite;
    string mCurIm;
    Pt2dr mCurSzIm_Bingo;
    int mCurImId;
    string mSuffixOut;
    bool mBinOut;
    string mFileName;
    string mImgFormat;
    string mPMulFileName;
    int mfile_type;
};
