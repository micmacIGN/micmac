#include "include/MMVII_all.h"

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

void AddData(const cAuxAr2007 & anAux,eModeCaracMatch & aMCM); 


typedef std::vector<eModeCaracMatch> tVecCar;
std::string  NameVecCar(const tVecCar &);

class cComputeSeparDist
{
    public :
         cComputeSeparDist();
         void AddPops(double aPopA,double aPopB);
         double Sep() const;
    private :
         double  mSomSep;  ///<  S(AB/(A+B))
         double  mSomP;    ///<  S(A+B)
};
template <class Type> double ComputeSep(const Type * aD1,const Type * aD2,int aNb);
template <class Type,int Dim> double ComputeSep(const cDataTypedIm<Type,Dim> &,const cDataTypedIm<Type,Dim> &);

class cAppliLearningMatch : public cMMVII_Appli
{
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


        // static std::string  Ext(bool isXml);
        static std::string  PrefixHom();
        static std::string  Hom(int aNum);
        static std::string  Index(int aNum);
        static std::string HomFromIm1(const std::string & aIm1,int aNumHom,std::string anExt,bool isXml=false);
        static std::string HomFromHom0(const std::string & aName,int aNumHom);
       
    private :
        std::string mNameInput;
        std::string mNameOutput;
};

class cVecCaracMatch : public cMemCheck
{
     public :
        static constexpr int TheDynSave = 1000;
        static constexpr int TheUnDefVal = TheDynSave +1;
        static constexpr int TheNbVals = int (eModeCaracMatch::eNbVals);
        typedef tU_INT2 tSaveValues;

        void SetValue(eModeCaracMatch aCarac,const float & aVal);
        const tSaveValues & Value(eModeCaracMatch aCarac) const ;

        cVecCaracMatch
        (
             float aScaleRho,float aGrayLev1,float aGrayLev2,
             const cAimePCar &,const cAimePCar &
       );
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
       static constexpr int TheDynSave = cVecCaracMatch::TheDynSave;
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
    private :
       tHisto  mHist[TheNbH];
       cIm2D<tINT4>                   mImCr01;  // Contain stat of Hom/CloseHom
       cIm2D<tINT4>                   mImCr02;  // Contain stat of Hom/NonHom
};
void AddData(const cAuxAr2007 & anAux,cStatOneVecCarac&);


class cStatAllVecCarac : public cMemCheck
{
     public :
        static constexpr int TheNbVals = int (eModeCaracMatch::eNbVals);
        static constexpr int TheDynSave = cVecCaracMatch::TheDynSave;

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

};

