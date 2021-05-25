#ifndef KUGELHUPF_H
#define KUGELHUPF_H


class cOneHypMarkFid
{
    public :

        cOneHypMarkFid (const cOneMesureAF1I &);

        cOneMesureAF1I   mMes;
        Pt2di            mTargetSize;
        
};

class cQuickCorrelPackIm
{
    public :
          cQuickCorrelPackIm(Pt2di aSzBufIm,Pt2di aSzMarqq,double aResol,bool Debug);
          void InitByReduce(const cQuickCorrelPackIm & aPack,double aDil);
          void FinishLoad();
          Pt2di               mSzIm;  // Taille de la ss res en fft, interet a ce que ce soit une puis de 2
          Pt2di               mSzMarq;  // Taille de la ss res en fft, interet a ce que ce soit une puis de 2
          double              mResol;

          TIm2D<REAL4,REAL8>  mTIm;   // Buffer pour l'image
          TIm2D<REAL4,REAL8>  mTRef;  // Buffer pour la marque fid
          TIm2D<REAL4,REAL8>  mTMasq;   // Buffer pour le masque
          double              mSomPds;
          Video_Win * mW;

          std::list<Pt3dr>    DecFFT();

          double Correl(const Pt2dr & aP,eModeInterpolation  aMode);
          Pt2di OptimizeInt(const Pt2di aP0,int aSzW);
          Pt2dr OptimizeReel(const Pt2dr aP0,double aStep,int aSzW,double & aMaxCor);
          bool  mDebug;
};


class cOneSol_QuickCor
{
    public :
        Pt2dr               mPOut;
        std::list<Pt3dr>    mLSols;
};


class cQuickCorrelOneFidMark
{
     public :
           cQuickCorrelOneFidMark
           (
              Fonc_Num            aFoncIm,
              Fonc_Num            aFoncRef,
              Fonc_Num            aFoncMasq,
              Box2di              aBoxRef,
              Pt2di               aIncLoc,
              int                 aSzFFT,
              bool                Debug
           );

           cOneSol_QuickCor TestCorrel(const Pt2dr & aMes);
     private :
          void MakeRed( TIm2D<REAL4,REAL8>  ImIn, TIm2D<REAL4,REAL8> ImOut);
          void LoadIm();

          Fonc_Num    mFoncFileIm;
          Fonc_Num    mFoncRef;
          Fonc_Num    mFoncMasqRef;
          bool        mNoMasq;

          Box2di mBoxRef;   //  Box autour marque fid
          Pt2di  mSzRef;    // Taille Marque Fid
          Pt2di  mIncLoc;   //  Incert Taile Marques
          Pt2di  mSzBuf;    // Taille des zones memoires a charger
          int    mSzSsResFFT;  // Taille de la ss res en fft, interet a ce que ce soit une puis de 2
          double mSsRes;    // Ss resolution
          int    mNbNiv;    // MbNiveau dans la pyramide
          double mSsResByNiv;  // Dif resol entre 2 niveau de pyramide

          std::vector<cQuickCorrelPackIm> mPyram;

          Pt2di  mCurDecIm;
          Pt2di  mCurDecRef;
          bool   mDebug;

};


class cOneSol_FFTKu
{
    public :
        cOneMesureAF1I mIn;
        cOneSol_QuickCor  mOut;
};
class cAppli_FFTKugelhupf_main :  public cAppliWithSetImage
{
    public :
        cAppli_FFTKugelhupf_main(int argc,char ** argv);
        void DoResearch();
    private :
        cOneSol_FFTKu Research1(const cOneMesureAF1I &);

        void  TestOneSolCombine(int aK1,int aK2,int aK3);


        std::string mFullPattern;
        std::string mFiducPtsFileName;

        Pt2di         mTargetHalfSzPx;
        int           mSearchIncertitudePx;
        int           mSzFFT;
        std::string   mExtMasq;

        cMesureAppuiFlottant1Im mDico;


        std::string                 mNameIm2Parse;
        std::string                 mNameImRef;
        std::string                 mNameFileMasq;
        cQuickCorrelOneFidMark *    mQCor;
        bool                        mWithMasq;
        std::vector<cOneSol_FFTKu>  mVSols;
        ElPackHomologue             mPackH;
        bool                        mDebug;

        static const std::string    TheKeyOI;

        std::list<cOneHypMarkFid>   mListHMF;

        bool                        mValSim;
        double                      mPdsCorr;
        double                      mBestCostComb;
        std::vector<Pt2dr>          mBestSolComb;
};


static const double TheDefCorrel = -2.0;

//Image for correlation class
//all images for correlation have the same size
class cCorrelImage
{
  public :
    cCorrelImage();
    Im2D<U_INT1,INT4> * getIm(){return &mIm;}
    TIm2D<U_INT1,INT4> * getImT(){return &mTIm;}
    double CrossCorrelation(const cCorrelImage & aIm2);
    double Covariance(const cCorrelImage & aIm2);
    int getSzW();
    Pt2di getmSz();
    void getFromIm(Im2D<U_INT1,INT4> * anIm,double aCenterX,double aCenterY);
    void getWholeIm(Im2D<U_INT1,INT4> * anIm);
    static int mSzW;//window size for the correlation
    static void setSzW(int aSzW);

  protected:
    void prepare();//prepare for correlation (when mTifIm is set)

    Pt2di mSz;
    TIm2D<U_INT1,INT4> mTIm; //the picture
    Im2D<U_INT1,INT4>  mIm;
    TIm2D<REAL4,REAL8> mTImS1; //the sum picture
    Im2D<REAL4,REAL8>  mImS1;
    TIm2D<REAL4,REAL8> mTImS2; //the sum² picture
    Im2D<REAL4,REAL8>  mImS2;
};

class cScannedImage
{
  public:
    cScannedImage
      (
       std::string aNameScannedImage,
       cInterfChantierNameManipulateur * aICNM,
       std::string aXmlDir
      );
    void load();
    Pt2di getSize(){return mImgSz;}
    TIm2D<U_INT1,INT4> * getImT(){if (!mIsLoaded) load();return & mImT;}
    Im2D<U_INT1,INT4> * getIm(){if (!mIsLoaded) load();return & mIm;}
    cMesureAppuiFlottant1Im & getAllFP(){return mAllFP;}//all fiducial points
    std::string getName(){return mName;}
    std::string getXmlFileName(){return mXmlFileName;}
    bool isExistingXmlFile(){return ELISE_fp::exist_file(mXmlFileName);}



  protected:
    std::string        mName;
    std::string        mNameImageTif;
    cMesureAppuiFlottant1Im mAllFP;//all fiducial points
    std::string mXmlFileName;
    Tiff_Im            mTiffIm;
    Pt2di              mImgSz;
    TIm2D<U_INT1,INT4> mImT;
    Im2D<U_INT1,INT4>  mIm;
    bool mIsLoaded;
};





#endif // KUGELHUPF_H
