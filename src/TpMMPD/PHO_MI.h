#ifndef PHO_MI_H
#define PHO_MI_H
struct AbreHomol
{
    string ImgRacine;
    vector<string> ImgBranch;
    vector< vector<string> > Img3eme;
    vector<double> NbPointHomo;
};

class VectorSurface
{
    public:
    VectorSurface(double dirX[2], double dirY[2]);
    double dirX[2];
    double dirY[2];
};

class CplImg
{
    public :
          CplImg(string aNameImg1, string aNameImg2);

          vector<string> mCollection3emeImg;
          string mNameImg1;
          string mNameImg2;
          Im2D<U_INT1,INT4> mImg1;
          Im2D<U_INT1,INT4> mImg2;
          CamStenope * mCam1;
          CamStenope * mCam2;
          //VectorSurface mSurfImg1;
          //VectorSurface mSurfImg2;

          void SupposeVecSruf1er(double dirX[2], double dirY[2]);
          void ValPtsLia(vector<double> NorSur);
          void CalVectorSurface(string mImg3eme);
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
          VerifParRepr(vector<string> mListImg, string mDirImages, string mPatImages, string mNameHomol, string mOri , string aHomolOutput, bool ExpTxt, double aDistHom, double aDistRepr );
          vector<AbreHomol> creatAbre();
          vector<string> displayAbreHomol(vector<AbreHomol> aAbre, bool disp);
          vector<bool> FiltreDe3img(string aNameImg1, string aNameImg2, string aNameImg3);
          void creatHomolFromPair(string aNameImg1, string aNameImg2, vector<bool> decision);
          void FiltragePtsHomo();

          vector<AbreHomol> mAbre;
          vector<string> mtempArbeRacine;
          vector<string> mListImg;
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
