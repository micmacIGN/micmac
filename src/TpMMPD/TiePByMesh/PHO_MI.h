#ifndef PHO_MI_H
#define PHO_MI_H
extern void StdCorrecNameHomol_G(std::string & aNameH,const std::string & aDir);
struct AbreHomol
{
    string ImgRacine;
    vector<string> ImgBranch;
    vector< vector<string> > Img3eme;
    vector<double> NbPointHomo;
    vector<double> NbPtFiltre;
};

class VectorSurface
{
    public:
    VectorSurface(Pt2dr dirX, Pt2dr dirY);
    VectorSurface();
    Pt2dr dirX;
    Pt2dr dirY;
};

class RepereImagette
{
    public:
    RepereImagette(Pt2dr centre, Pt2dr dirX, Pt2dr dirY);
    RepereImagette();
    Pt2dr centre;
    Pt2dr dirX;
    Pt2dr dirY;
    Pt2dr uv2img(Pt2dr coorOrg);
};

class CplImg : public cCorrelImage
{
    public :
          CplImg(string aNameImg1, string aNameImg2, string aNameHomol, string aOri, string aHomolOutput,
                 string aFullPatternImages, bool ExpTxt, double aPropDiag, double aCorel,
                 double asizeVignette, bool adisplayVignette, bool aFiltreBy1Img, double aTauxGood, double aSizeSearchAutour);
          vector<double> nul;
          vector<string> mCollection3emeImg;
          string mNameImg1;
          string mNameImg2;
          Im2D<U_INT1,INT4> mImg1;
          Im2D<U_INT1,INT4> mImg2;
          CamStenope * mCam1;
          CamStenope * mCam2;
          string mDirImages;
          string mPatImages;
          string mNameHomol;
          string mOri;
          string mHomolOutput;
          double mPropDiag;
          double mCorel;
          double msizeVignette;
          bool mdisplayVignette;
          vector<string> mSetImages;
          string mKHIn, mKHOutDat, mKHOut;
          cInterfChantierNameManipulateur * mICNM;
          bool mExpTxt;
          bool mFiltreBy1Img;
          double mTauxGood;
          double mSizeSearchAutour;
          VectorSurface mSurfImg1;
          VectorSurface mSurfImg2;

          void SupposeVecSruf1er(Pt2dr dirX, Pt2dr dirY);
          void ValPtsLia(vector<double> NorSur);

          vector<bool> CalVectorSurface(string mImg3eme, string ModeSurf);
          //bool IsInside(Pt2dr checkPoint, double w, double h);
          Video_Win *mW, *mW1, *mW2;
          bool NotTif_flag;    
};

class UneImage
{
    public :
          UneImage (string aNameImg);
          string mName;
          string mDir;
          string mPatOrg;
          string mOri;
          CamStenope * aCam;
};

class VerifParRepr
{
    public :
          VerifParRepr(vector<string> mListImg, vector<string> mListImg_NoTif, string mDirImages, string mPatImages, string mNameHomol, string mOri , string aHomolOutput, double aDistHom, double aDistRepr );
          vector<AbreHomol> creatAbre();
          vector<string> displayAbreHomol(vector<AbreHomol> aAbre, bool disp);
          vector<bool> FiltreDe3img(string aNameImg1, string aNameImg2, string aNameImg3);
          void creatHomolFromPair(string aNameImg1, string aNameImg2, vector<bool> decision);
          void FiltragePtsHomo();

          vector<AbreHomol> mAbre;
          vector<string> mtempArbeRacine;
          vector<string> mListImg, mListImg_NoTif;
          string mDirImages;
          string mPatImages;
          string mNameHomol;
          string mOri;
          bool mExpTxt;
          Pt2dr mcentre_img;
          double mdiag;
          double mDistRepr;
          double mDistHom;
          string mHomolOutput;
          vector<bool> mdecision;
};



struct PtTrip
{
    Pt2dr P1;
    Pt2dr P2;
    Pt2dr P3;
};

struct PtDoub
{
    Pt2dr P1;
    Pt2dr P2;
};

struct PairHomol
{
    string ImgA;
    string ImgB;
    ElPackHomologue HomoA_B;
};

#endif
