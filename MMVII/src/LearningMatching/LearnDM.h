#include "MMVII_enums.h"
#include "MMVII_AimeTieP.h"
#include "cMMVII_Appli.h"
#include "MMVII_Matrix.h"


/*
    Caracteristiques envisagees :

      * L1  sur tout pixel
      * L1  pondere par 1/Scale sur tout pixel
      * L1 sur le + petit rayon
      * census (pondere ?)
      * correlation  (pondere ? 1 pixel? 2 pixel ? ...)
      * gradient de L1  sur paralaxe (mesure ambiguite residuelle)
      * mesure sur gradient (rho ? theta ? ...)
*/

namespace MMVII
{
bool TESTPT(const cPt2dr & aPt,int aLine,const std::string& aFile);
#define TPT(AP) TESTPT(AP,__LINE__,__FILE__)


void AddData(const cAuxAr2007 & anAux,eModeCaracMatch & aMCM); 


typedef std::vector<eModeCaracMatch> tVecCar;
std::string  NameVecCar(const tVecCar &);

class cComputeSeparDist
{
    public :
         cComputeSeparDist();
         void AddPops(double aPopA,double aPopB);
         double Sep() const; // Something like mSomSep / mSomP
    private :
         double  mSomSep;  ///<  S(AB/(A+B))
         double  mSomP;    ///<  S(A+B)
};
template <class Type> double ComputeSep(const Type * aD1,const Type * aD2,int aNb);
template <class Type,int Dim> double ComputeSep(const cDataTypedIm<Type,Dim> &,const cDataTypedIm<Type,Dim> &);

extern bool DEBUG_LM;

class cAppliLearningMatch : public cMMVII_Appli
{
    public :
	const int &  NbOct() const;
	const int &  NbLevByOct() const;
	const int &  NbOverLapByO() const;
        static const int SzMaxStdNeigh() {return 8;}
    protected :
        cAppliLearningMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

        void SetNamesProject (const std::string & aNameInput,const std::string & aNameOutput) ;
        std::string Prefix(bool isIn) const;
        static std::string Post(bool isXml)  ;
        std::string  DirVisu()  const;
        std::string  DirResult()  const;
        std::string  SubDirResult(bool isIn)  const;
        std::string  FileHisto1Carac(bool isIn,bool isXml=false)  const ;
        std::string  NameReport() const;
        std::string  FileHistoNDIm(const std::string &,bool IsIn) const;


        static std::string  PrefixAll();
        static std::string  Im1();
        static std::string  Im2();
        static std::string  Px1();
        static std::string  Px2();
        static std::string  Masq1();
        static std::string  Masq2();

	static bool  IsFromType(const std::string & aName,const std::string & aPost);
	static bool  IsIm1(const std::string & aName);
	static bool  IsIm2(const std::string & aName);
	static bool  Im1OrIm2(const std::string & aName); // Generate an error if none




        static std::string MakeName(const std::string & aName,const std::string & aPref) ;
        static void GenConvertIm(const std::string & aInput, const std::string & aOutput);

        static std::string NameIm1(const std::string & aName); 
        static std::string NameIm2(const std::string & aName);
        static std::string NamePx1(const std::string & aName);
        static std::string NamePx2(const std::string & aName);
        static std::string NameMasq1(const std::string & aName);
        static std::string NameMasq2(const std::string & aName);

        static std::string NameRedrIm1(const std::string & aName); 
        static std::string NameRedrIm2(const std::string & aName);

        static void ConvertIm1(const std::string & aInput,const std::string & aName);
        static void ConvertIm2(const std::string & aInput,const std::string & aName);


        static std::string Im2FromIm1(const std::string & aIm1);
        static std::string Px1FromIm1(const std::string & aIm1);
        static std::string Masq1FromIm1(const std::string & aIm1);
        static std::string Px2FromIm2(const std::string & aIm2);
        static std::string Masq2FromIm2(const std::string & aIm2);

        static std::string PxFromIm(const std::string & aIm12);
        static std::string MasqFromIm(const std::string & aIm12);

        // static std::string  Ext(bool isXml);
        static std::string  PrefixHom();
        static std::string  Hom(int aNum);
        static std::string  Index(int aNum);
        static std::string HomFromIm1(const std::string & aIm1,int aNumHom,std::string anExt,bool isXml=false);
        static std::string HomFromHom0(const std::string & aName,int aNumHom);
       
    private :
        std::string mNameInput;
        std::string mNameOutput;

	int   mNbOct;       // 3 octave for window , maybe add 2 learning multiscale
	int   mNbLevByOct;  // more or less minimalist
	int   mNbOverLapByO; // 1 overlap is required for junction at decimation
};

class cPyr1ImLearnMatch : public cMemCheck
{
      public :
          typedef cGaussianPyramid<tREAL4>   tPyr;
          typedef std::shared_ptr<tPyr>      tSP_Pyr;
          typedef cIm2D<tREAL4>              tImFiltred;
          typedef cDataIm2D<tREAL4>          tDataImF;

          cPyr1ImLearnMatch
          (
                const cBox2di & aBox,
                const cBox2di & aBoxOut, // Required by pyram but apparently not used
                const std::string & aName,
                const cAppliLearningMatch &,
                const cFilterPCar&,
                bool  initRand
          );
          cPyr1ImLearnMatch(const cPyr1ImLearnMatch &) = delete;
          void SaveImFiltered() const;
          bool  CalculAimeDesc(const cPt2dr & aPt);
	  double  MulScale() const;
	  const tDataImF &  ImInit() const;
	  const tDataImF &  ImFiltered() const;
	  cAimePCar   DupLPIm() const;
          // ~cPyr1ImLearnMatch();
      private :
          cBox2di                     mBox;
          std::string                 mNameIm;
          const cAppliLearningMatch & mAppli;
          cGP_Params                  mGP;
          tSP_Pyr                     mPyr;
          tImFiltred                  mImF;
	  cAimePCar                   mPC;
};


class cVecCaracMatch : public cMemCheck
{
     public :
        typedef cDataIm2D<tREAL4>   tDataIm;
        static constexpr int TheDyn4Save = 20000;
        static constexpr int TheDyn4Visu = 1000;
        static int ToVisu(int aVal) {return std::min(TheDyn4Visu-1,(aVal*TheDyn4Visu)/TheDyn4Save);}
        static int FromVisu(int aVal) {return std::min(TheDyn4Save-1,(aVal*TheDyn4Save)/TheDyn4Visu);}
        static cPt2di ToVisu(const cPt2di & aPt) {return cPt2di(ToVisu(aPt.x()),ToVisu(aPt.y()));}

        static constexpr int TheUnDefVal = TheDyn4Save +1;
        static constexpr int TheNbVals = int (eModeCaracMatch::eNbVals);
        typedef tU_INT2 tSaveValues;

        void SetValue(eModeCaracMatch aCarac,const float & aVal);
        const tSaveValues & Value(eModeCaracMatch aCarac) const ;

        cVecCaracMatch
        (
             float aScaleRho,
             const tDataIm & aImInit1,const tDataIm & aImInit2,
             const tDataIm & aImNorm1,const tDataIm & aImNorm2,
             const cAimePCar &,const cAimePCar &
        );
        cVecCaracMatch
        (
             const cPyr1ImLearnMatch  & aPyr1,const cPyr1ImLearnMatch  & aPyr2,
             const cAimePCar &,const cAimePCar &
        );
/*
        cVecCaracMatch
        (
             float aScaleRho,float aGrayLev1,float aGrayLev2,
             const cAimePCar &,const cAimePCar &
        );
*/
        cVecCaracMatch() ;
        void AddData(const cAuxAr2007 & anAux);
        void Show(tNameSelector);

        void FillVect(cDenseVect<tINT4> &,const tVecCar &) const;
     private :
        
        tSaveValues   mVecCarac[TheNbVals];
};


void AddData(const cAuxAr2007 & anAux, cVecCaracMatch &    aVCM);

class cFileVecCaracMatch : public cMemCheck
{
     public :
        cFileVecCaracMatch(const cFilterPCar &,int aNb);
        cFileVecCaracMatch(const std::string &); ///< From file
        void AddCarac(const cVecCaracMatch &);
        void AssertCompatible(const cFileVecCaracMatch &);
        void AddData(const cAuxAr2007 & anAux);
        const std::vector<cVecCaracMatch> & VVCM() const;
     private  :
        int                         mNbVal;
        cFilterPCar                 mFPC;
        std::vector<cVecCaracMatch> mVVCM;
        std::string                 mCheckRW;  // to check read/write works

};

void AddData(const cAuxAr2007 & anAux, cFileVecCaracMatch &    aVCM);


class cStatOneVecCarac : public cMemCheck
{
    public :
       typedef cHistoCumul<tINT4,tREAL8>  tHisto;
       static constexpr int TheDyn4Save = cVecCaracMatch::TheDyn4Save;
       static constexpr int TheDyn4Visu = cVecCaracMatch::TheDyn4Visu;
       static constexpr int TheNbH = 3;
       cStatOneVecCarac(const cPt2di & aSzCr = cPt2di(1,1));
       void Add(int aNum,int aVal)
       {
            Hist(aNum).AddV(aVal,1);
       }
       double  Separ(int aN1,int aN2) const; // Compute separability betwen Hist[N1] and Hist[N2]
       void Inspect(const cStatOneVecCarac &);
       cDataIm2D<tINT4> & ImCr(bool Close);
       const cDataIm2D<tINT4> & ImCr(bool Close) const;
       double  FiabCr(bool Close) const;

       void SaveCr(int aDeZoom,bool isClose,const std::string &);
       void SaveHisto(int aSz,const std::string &);
       // Reduce size of mImCr01, wich are note usefull for saving
       void PackForSave();
       void MakeCumul();
       void AddData(const cAuxAr2007 & anAux);

       tHisto  & Hist(int aNum);
       const tHisto  & Hist(int aNum) const;
       const tHisto  & HistSom(int aFlag) const;
    private :
       tHisto  mHist[TheNbH];
       mutable tHisto  mHistSom;
       cIm2D<tINT4>                   mImCr01;  // Contain stat of Hom/CloseHom
       cIm2D<tINT4>                   mImCr02;  // Contain stat of Hom/NonHom
};
void AddData(const cAuxAr2007 & anAux,cStatOneVecCarac&);


class cStatAllVecCarac : public cMemCheck
{
     public :
        static constexpr int TheNbVals = int (eModeCaracMatch::eNbVals);
        static constexpr int TheDyn4Save = cVecCaracMatch::TheDyn4Save;
        static constexpr int TheDyn4Visu = cVecCaracMatch::TheDyn4Visu;

        cStatAllVecCarac(bool WithCrois);
        void AddOneFile(int aNum,const cFileVecCaracMatch &);
        void AddCr(const cFileVecCaracMatch &,const cFileVecCaracMatch &,bool isClose);
        void ShowSepar(const std::string & aPat,cMultipleOfs &);
        void Inspect();
        void SaveCr(int aDeZoom,const std::string &aDir);
        void SaveHisto(int aSz,const std::string &aDir);
        void PackForSave(); // Supress Cr to reduce size
        void AddData(const cAuxAr2007 & anAux);
        void MakeCumul();
        const cStatOneVecCarac & OneStat(eModeCaracMatch) const;
     private :
        bool                           mWithCr;
        cPt2di                         mSzCr;
        std::vector<cStatOneVecCarac>  mStats;
};
void AddData(const cAuxAr2007 & anAux,cStatAllVecCarac&);


// Class to represent statistic of conbination of criterion,
// memorize the histogramm of Hom & Nom Hom, for each dimension the value
// are resampled according to some histogramm equalization 
//
class cHistoCarNDim  : public cMemCheck
{
    public :

       typedef tINT4                        tDataNd;
       typedef cDataGenDimTypedIm<tDataNd>  tHistND;
       typedef cHistoCumul<tINT4,tREAL8>    tHisto1;
       typedef cDenseVect<tINT4>            tIndex;
       typedef cDenseVect<tREAL4>           tRIndex;

       cHistoCarNDim(int aSzH,const tVecCar &,const cStatAllVecCarac &,bool genVis2DI);
       cHistoCarNDim();  // Used for AddData requiring default cstrc
       cHistoCarNDim(const std::string&);  // Used for AddData requiring default cstrc
       ~cHistoCarNDim();  // Used for AddData requiring default cstrc
       void  Add(const cVecCaracMatch &,bool isH0);
       void  Show(cMultipleOfs &,bool WithCr) const;
       void AddData(const cAuxAr2007 &);
       const std::string & Name() const;

       /* Score Cr/CarSep  
	    CarSep : separation between the 2 dist using cComputeSeparDist
            HomologyLikelihood(V) : Likelihood that a given vector is homolog
	    UpDateCorrectness(Hom,NHom) : update the the proba that  HomologyLikelihood(Hom) > HomologyLikelihood(NonHom)
	    Correctness() : global proba of HomologyLikelihood(Hom) > HomologyLikelihood(NonHom)
	*/
       double CarSep() const;
       double Correctness() const;
       double HomologyLikelihood(const cVecCaracMatch &,bool Interpol) const;
       void   UpDateCorrectness(const cVecCaracMatch & aHom,const cVecCaracMatch & aNotHom);
       // Generate 4 Visu :
       //     Hom, Non Hom, Score, Pop
       void  GenerateVisu(const std::string & aDir);
       // Generate 2 Visu in initial dynamic : hom & non hom
       void  GenerateVis2DInit(const std::string & aDir);

    private :
       //  Generarte visu of one Histogramme
       void  GenerateVisuOneIm(const std::string & aDir,const std::string & aPrefix,const tHistND &);
       //  
       void  GenerateVis2DInitOneInit(const std::string & aDir,const std::string & aPrefix,cIm2D<double>,const tHistND&);
       void  ComputePts(const cVecCaracMatch &) const;
       cHistoCarNDim(const cHistoCarNDim &) = delete;

       bool                      mIP;
       int                       mDim;
       tIndex                    mSz;
       tVecCar                   mVCar;
       mutable tIndex                    mPts;
       mutable tIndex                    mPtsInit; // Memorize Pts before Histo Equal, for visualization
       mutable tRIndex                   mRPts;    // Real Pts

       std::vector<const tHisto1*>     mHd1_0;
       tHistND                   mHist0;  // Homolog
       tHistND                   mHist2;  //  Non (or close) Homolog
       std::string               mName;
       double                    mNbOk;
       double                    mNbNotOk;
       bool                      mGV2I;
       cIm2D<double>             mHistoI0;
       cIm2D<double>             mHistoI2;
};
void AddData(const cAuxAr2007 & anAux,cHistoCarNDim & aHND);





};

