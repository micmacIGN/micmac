/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.


Header-MicMac-eLiSe-25/06/2007*/

#include "StdAfx.h"
#include <algorithm>

class cSolChemOptImV;
class cOneImageVideo;
class cAppliDevideo;

// -r rate
// -i input
// -ss  debu
// -t   duree
//  
// avconv  
// ffmpeg -i MVI_0001.MOV  -ss 30 -t 20 Im%5d_Ok.png
// ffmpeg -i MVI_0247.MOV Im_0247_%5d_Ok.png

// Im*_Ok => OK
// Im*_Nl => Image Nulle (eliminee)

// POUR L'INSTANT LES IMAGES DOIVENT AVOIR UN NOM FIGE AVEC Im_0000_

// ffmpeg -i MVI_0101.MOV Im_0000_%5d_Ok.png

// Adjudant 50 59

//=================================================

/*
cMTDImCalc GetMTDImCalc(const std::string & aNameIm);
const cMIC_IndicAutoCorrel * GetIndicAutoCorrel(const cMTDImCalc & aMTD,int aSzW);
std::string NameMTDImCalc(const std::string & aFullName);
*/

double  GlobAutoCorrel(const std::string &aName,int aWSz)
{
    Tiff_Im aTF = Tiff_Im::StdConvGen(aName,1,true);

    Pt2di aSz = aTF.sz();
    Im2D_REAL4 aI0(aSz.x,aSz.y);
    ELISE_COPY( aTF.all_pts(),aTF.in(),aI0.out());


    TIm2D<REAL4,REAL8> aTIm(aI0);

     double aSomGlob=0.0;
     double aNbGlob=0.0;

     for (int aKdx=-aWSz ; aKdx<=aWSz ; aKdx+=aWSz)
     {
         for (int aKdy=-aWSz ; aKdy<=aWSz ; aKdy+=aWSz)
         {
             int aDx = aKdx;
             int aDy = aKdy;
             Pt2di aDep(aDx,aDy);
             if (dist8(aDep) == aWSz)
             {
                Pt2di aP;
                RMat_Inertie aMat;
                for (aP.x = aWSz ; aP.x<aSz.x-aWSz ; aP.x++)
                {
                    for (aP.y=aWSz ; aP.y<aSz.y-aWSz ; aP.y++)
                    {
                      aMat.add_pt_en_place(aTIm.get(aP),aTIm.get(aP+aDep));
                    }
                }
                double aC = aMat.correlation();
                aC = 1-aC;
                aSomGlob += aC;
                aNbGlob ++;
             }
         }
     }
     return  aSomGlob / aNbGlob;
}




int  CalcAutoCorrel_main(int argc,char ** argv)
{
    int          aSzW=2;
    std::string  aNameIm;
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aNameIm,"Name Im", eSAM_IsPatFile) ,
           LArgMain() << EAM(aSzW,"SzW", true,"SzW Auto Correl")
    );

    cMTDImCalc aMTD = GetMTDImCalc(aNameIm);
    const cMIC_IndicAutoCorrel *  aIAC = GetIndicAutoCorrel(aMTD,aSzW);

    if (aIAC==0)
    {
        cMIC_IndicAutoCorrel aNewIAC;
        aNewIAC.AutoC() = GlobAutoCorrel(aNameIm,aSzW);
        aNewIAC.SzCalc() = aSzW;
        aMTD.MIC_IndicAutoCorrel().push_back(aNewIAC);
        for (int aKBin=0 ; aKBin<=1 ; aKBin++)
        {
            std::string aNameXml = NameMTDImCalc(aNameIm, aKBin != 0);
            MakeFileXML(aMTD,aNameXml);
        }

    }

    return EXIT_SUCCESS;
}



//=================================================


static const int  TheSzDecoup = 300; // Taille de decoupe pour limiter taille et temps de Fenetre de temps
static const double ThePropRechPiv = 0.1;



int DevOneImPtsCarVideo_main(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}


class cSolChemOptImV
{
     public :
          cSolChemOptImV(double aCost);
          double mCost;
          int    mIndPred;
};

class cOneImageVideo
{
    public :
        cOneImageVideo(const std::string & aNameIm,cAppliDevideo *,int aK);
        void CalcScoreAutoCorrel(const  std::vector<cOneImageVideo*> &,int aSzW);
        void CalcRec(cOneImageVideo *);

        const std::string & NameOk()    const {return mNameOk;}
        void  Show();
        bool  Pressel() const;
        void  InitOk() ;
        bool  IsMaxLoc() const;
        double   AutoCorrel() const;
        double   PdsAutoCor() const;
        int   TimeNum() const;
        int   DifTime(const cOneImageVideo &) const;
        cAppliDevideo *  Appli();

        void SetLongPred(int aK);
        int  PresselNum() const;
        void InitChemOpt();
        void ClearSol();
        void UpdateCheminOpt(int aS0,int aS1,const std::vector<cOneImageVideo*> &,double aStdJ);
        const cSolChemOptImV & SolOfLength(int aNbL);
        void LoadAutoCorrel();
        double  CumNonRec() const {return mCumNonRec;}
        double DistCum(const cOneImageVideo & anIm) const {return ElAbs(mCumNonRec-anIm.mCumNonRec);}
        double  NormAC() const {return  mNormAC;}
    private :
        void  TestInitSift();

        cAppliDevideo *  mAppli;
        std::string      mNameInit;
        std::string      mNameOk;
        std::string      mNameNl;
        std::string      mNamePtsSift;
        bool             mOkFin;            // => selection finale
        int              mLongPred;    // => longueur du chemin
        int              mTimeNum;
        double           mAutoCor;
        double           mNormAC;
        double           mPdsAutoCor;
        std::vector<cSolChemOptImV>  mSols;
        double            mNonRecPrec;
        double            mCumNonRec;
        const cMetaDataPhoto &mMDT;
        Pt2di                 mSz;
        int                   mNbHom;
};



class cAppliDevideo
{
    public :
        cAppliDevideo(int argc,char ** argv);
        const std::string & Dir() {return mEASF.mDir;}
        std::string CalcName(const std::string & aName,const std::string aNew);

        std::string  NamePtsSift( cOneImageVideo * anOIV);
        cInterfChantierNameManipulateur * ICNM() {return mEASF.mICNM;}
        double NacModifExtre(cOneImageVideo * anI,cOneImageVideo * anI0);
    private :
        int DecndreGivenLenght(int aSInit,double aLength,int aSLimit,bool Depasse);
        std::string NamePat(const std::string & aPref) const;
        int PivotAPriori(int aKI) { return round_ni((aKI*(mNbIm-1))/mNbInterv);}
        int   ModifPivot(int aIndP0,int aSign);
        void ComputeChemOpt(int aS0,int aS1);

        std::string mFullNameVideo;
        double      mF35;
        double      mFoc;
        std::string mCam;
        bool        mDoVideo2Im;


        cElemAppliSetFile mEASF;
        std::string mPrefix;
        std::string mPostfix;
        std::string mMMPatImDev;
        cElRegex *  mAutoMM;
        std::vector<std::string>   mVName;
        std::vector<cOneImageVideo*>      mVIms;
        int                               mNbIm;
        double      mTargetRate;
        double      mRateVideoInit;
        double      mStdJump;
        int         mParamSzSift;
        std::string mStrSzS;
        int         mNbInterv;
        std::string mPatNumber;
        Pt2di       mMinMax;
        int         mNbDigit;
        double      mPercImInit;
        bool        mTuning;
        std::vector<int> mVPivots;
        int              mSzDecoup;
        double           mOverLap;
};



// =============  cSolChemOptImV ===================================
cSolChemOptImV::cSolChemOptImV(double aCost) :
   mCost (aCost),
   mIndPred (-1)
{
}


// =============  cOneImageVideo ===================================

double GetAutoCorrel(const cMTDImCalc & aMTD,int aSzW);


cOneImageVideo::cOneImageVideo(const std::string & aNameIm,cAppliDevideo * anAppli,int anTimeNum) :
   mAppli       (anAppli),
   mNameInit    (aNameIm),
   mNameOk      (mAppli->CalcName(aNameIm,"Ok")),
   mNameNl      (mAppli->CalcName(aNameIm,"Nl")),
   mOkFin       (true),
   mLongPred    (-1),
   mTimeNum     (anTimeNum),
   mAutoCor     (-1),
   mNonRecPrec  (0),
   mCumNonRec   (0),
   mMDT         (cMetaDataPhoto::CreateExiv2(aNameIm)),
   mSz          (mMDT.SzImTifOrXif())
{
    // std::cout << mNameInit   << "  " << mNameOk << "\n";
    if (mNameInit!= mNameOk)
       ELISE_fp::MvFile(anAppli->Dir()+mNameInit,anAppli->Dir()+mNameOk);
}

void cOneImageVideo::CalcRec(cOneImageVideo * aPrec)
{
   std::string aNameH = mAppli->Dir() 
                      + mAppli->ICNM()->Assoc1To2("NKS-Assoc-CplIm2Hom@@dat",mNameInit,aPrec->mNameInit,true);
   ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);
   mNbHom = aPack.size();

   double aDist,aQual;
   bool Ok;
   cElHomographie aHom = cElHomographie::RobustInit   
                         (
                            aDist,
                            &aQual,
                            aPack,
                            Ok,
                            20,
                            80.0,
                            100
                         );



   Box2dr aBox(Pt2dr(0,0),Pt2dr(mSz));
   Pt2dr aC4[4];
   aBox.Corners(aC4);
   std::vector<Pt2dr> aCont1,aCont2;
   for (int aK=0 ; aK<4 ; aK++)
   {
       aCont1.push_back(aC4[aK]);
       aCont2.push_back(aHom.Direct(aC4[aK]));
   }
   cElPolygone  aPol1,aPol2;
   aPol1.AddContour(aCont1,false);
   aPol2.AddContour(aCont2,false);
   cElPolygone aPolInter = aPol1 * aPol2;
   cElPolygone aPolUnion = aPol1 + aPol2;

   mNonRecPrec = ElMax(0.0, (1.0-aPolInter.Surf()/aPolUnion.Surf())/1.0 );
  
   mCumNonRec = aPrec->mCumNonRec + mNonRecPrec;

   std::cout << "cOneImageVideo::CalcRec " << mNameInit  
             << " S=" << mNbHom
             << " R=" << mNonRecPrec
             << " C=" << mCumNonRec
             << "\n";
          



}

void cOneImageVideo::LoadAutoCorrel()
{
    cMTDImCalc  aMDTI = GetMTDImCalc(mAppli->Dir()+mNameOk);
    mAutoCor = GetAutoCorrel(aMDTI,2);

}


   // mNamePtsSift =mAppli->NamePtsSift(this);

class cCmpIVPtrOnAC
{
    public :
        bool operator () (const cOneImageVideo * aV1,const cOneImageVideo *aV2)
        {
           return aV1->AutoCorrel() < aV2->AutoCorrel();
        }
};

class cGetACOnVPtr
{
    public :
       double operator()(const cOneImageVideo * aV) const {return aV->AutoCorrel();}
       typedef double tValue;
};
class cGetPdsOnVPtr
{
    public :
       double operator()(const cOneImageVideo * aV) const {return aV->PdsAutoCor();}
       typedef double tValue;
};



void cOneImageVideo::CalcScoreAutoCorrel(const  std::vector<cOneImageVideo*> & aVOIV,int aSzW)
{
      int aK0 = ElMax(0,mTimeNum-aSzW);
      int aK1 = ElMin(mTimeNum+1+aSzW,int(aVOIV.size()));

      std::vector<cOneImageVideo*> aSV;
      double aSomPds = 0;
      for (int aK=aK0 ; aK<aK1 ; aK++)
      {
          double aPds = ElAbs(aSzW+1 -ElAbs(mTimeNum-aK)) / double(aSzW+1);
          aVOIV[aK]->mPdsAutoCor = aPds;
          aSomPds += aPds;
          aSV.push_back(aVOIV[aK]);
      }
      cCmpIVPtrOnAC aCmp;
      std::sort(aSV.begin(),aSV.end(),aCmp);

      cGetACOnVPtr aGetVal;
      cGetPdsOnVPtr aGetPds;
      double aVMed = GenValPdsPercentile(aSV,50.0,aGetVal,aGetPds,aSomPds);

      double aEcartMoy = 0.0;
      for (int aK=aK0 ; aK<aK1 ; aK++)
      {
           aEcartMoy += ElAbs(aVOIV[aK]->mAutoCor-aVMed) * aVOIV[aK]->mPdsAutoCor;
      }
      aEcartMoy /= aSomPds;

      mNormAC = (mAutoCor-aVMed)/aEcartMoy;

      std::cout << mNameOk << " " << mAutoCor << " Med " << aVMed  << " NAC " << mNormAC << "\n";
}





void  cOneImageVideo::Show()
{
    std::cout << (mOkFin? "###" :  "---") << mNameOk << " Time:" << mTimeNum << " CUM:" << mCumNonRec << " C:" << mAutoCor  <<  " N:" << mNormAC << "\n";
}


int cOneImageVideo::TimeNum() const {return mTimeNum;}
cAppliDevideo *cOneImageVideo::Appli() {return mAppli;}
double cOneImageVideo::AutoCorrel() const {return mAutoCor;}
double cOneImageVideo::PdsAutoCor() const {return mPdsAutoCor;}



void cOneImageVideo::SetLongPred(int aL)
{
    mLongPred = aL;
}

void cOneImageVideo::InitOk()
{
   if (mLongPred<0)
   {
      ELISE_fp::MvFile(mAppli->Dir()+mNameOk,mAppli->Dir()+mNameNl);
      mOkFin=false;
   }
}


int cOneImageVideo::DifTime(const cOneImageVideo & anOIV) const
{
    return ElAbs(mTimeNum-anOIV.mTimeNum);
}


     // Lie au calcul du chemin opt



void cOneImageVideo::InitChemOpt()
{
   mSols.push_back(cSolChemOptImV(0.0));
}

void cOneImageVideo::ClearSol()
{
    mSols.clear();
}

void cOneImageVideo::UpdateCheminOpt(int aS0,int aS1,const std::vector<cOneImageVideo*> & aVIV,double aStdJ)
{
    for (int aS=aS0 ; aS< aS1 ; aS++)
    {
        const cOneImageVideo & aPred = *(aVIV[aS]);
        // double aDeltaL = ElAbs(ElAbs(aS-mTimeNum)-aStdJ) / aStdJ;
        double aDeltaL = DistCum(*(aVIV[aS])) /aStdJ;
        double aEps = 0.2;
        if (aDeltaL<1) aDeltaL = (1+aEps)/(aDeltaL +aEps);
        aDeltaL = ElAbs(aDeltaL);


        double aGainArc = mNormAC +  aPred.mNormAC - (aDeltaL + 0*ElSquare(aDeltaL)) * 1.0;
        for (int aKSolP=0 ; aKSolP<int(aPred.mSols.size())  ; aKSolP++)
        {
            while (int(mSols.size()) <= (aKSolP+1))
            {
                mSols.push_back(cSolChemOptImV(-1e30));
            }
            double aNewCost = aGainArc + aPred.mSols[aKSolP].mCost;
            cSolChemOptImV & aCurSol = mSols[aKSolP+1];
            if (aNewCost > aCurSol.mCost)
            {
                     aCurSol.mCost = aNewCost;
                     aCurSol.mIndPred =  aS;
            }
        }
    }
}

const cSolChemOptImV &  cOneImageVideo::SolOfLength(int aNbL)
{
    if ( (aNbL<0) || (aNbL>=int(mSols.size())) || (mSols[aNbL].mIndPred<0))
    {
        int aKBest = -1;
        double aDMin = 1e9;
        for (int aK=0 ; aK<int(mSols.size()); aK++)
        {
             if ( mSols[aK].mIndPred>=0)
             {
                 double  aDist = ElAbs(aK-aNbL);
                 if (aDist<aDMin)
                 {
                     aDMin = aDist;
                     aKBest = aK;
                 }
             }
        }
        ELISE_ASSERT(aKBest!=-1,"cOneImageVideo::SolOfLength");
        aNbL = aKBest;
    }

    return  mSols[aNbL];
}


// =============  cAppliDevideo ===================================

#define TEST true

// To get output of ffmpeg's FPS request

std::string getCmdOutput(const char* cmd) {
    char buffer[128];
    std::string result = "";
    #if (ELISE_windows)
        FILE* pipe = _popen(cmd, "r");
        if (!pipe) throw std::runtime_error("_popen() failed!");
        try {
            while (fgets(buffer, sizeof buffer, pipe) != NULL) {
                result += buffer;
            }
        } catch (...) {
            _pclose(pipe);
            throw;
        }
        _pclose(pipe);
    #else
        FILE* pipe = popen(cmd, "r");
        if (!pipe) throw std::runtime_error("popen() failed!");
        try {
            while (fgets(buffer, sizeof buffer, pipe) != NULL) {
                result += buffer;
            }
        } catch (...) {
            pclose(pipe);
            throw;
        }
        pclose(pipe);
    #endif
    return result;
}

cAppliDevideo::cAppliDevideo(int argc,char ** argv) :
     mCam            ("NONE"),
     mPostfix        ("png"),
     mTargetRate     (4.0),
     mRateVideoInit  (24.0),
     mParamSzSift    (-1),
     mMinMax         (0,100000000),
     mNbDigit        (5),
     mPercImInit     (100.0),
     mTuning         (false),
     mSzDecoup       (TheSzDecoup),
     mOverLap        (0.1)
{

    ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mFullNameVideo,"Full name of video", eSAM_IsPatFile) 
                      << EAMC(mF35,"Focal equiv 35, set -1 if no xif wanted"),
           LArgMain() << EAM(mTargetRate,"Rate",true,"Rate final Def=4")
                      << EAM(mParamSzSift,"ParamSzSift",true,"Parameter used for sift development, def=-1 (Highest)")
                      << EAM(mPatNumber,"PatNumber",true,"Pat number (reduce number for test)")
                      << EAM(mDoVideo2Im,"DoV2I",true,"Do the development of video 2 images, def true if no image to corresp pattern")
                      << EAM(mNbDigit,"NDig",true,"Nb digit for numbering out images (Def=5)")
                      << EAM(mFoc,"Foc",true,"Set focale in xif, def=F35")
                      << EAM(mCam,"Cam",true,"Set Cam in xif")
                      << EAM(mPercImInit,"PercImInit",true,"Pecentage of images a priori sele")
                      << EAM(mMinMax,"MinMax",true,"Min Max numbers (reduce number for test)")
                      << EAM(mTuning,"Tuning",true,"as it says ... ")
                      << EAM(mSzDecoup,"TheSzDecoup",true,"Sz of a priori split, \"expert\" level , Def=300 ")
                      << EAM(mOverLap,"OverLap",true,"Target overlap between images ")
                      << EAM(mRateVideoInit,"RateVideo",true,"Rate image per seconde (FPS), Def=fps value from ffmpeg ")
    );

// avconv -i adjudant.MOV Im_0247_%5d_Ok.png  

    if (! EAMIsInit(&mFoc)) 
       mFoc = mF35;

    if (MMVisualMode) return;

    std::cout << "BEGIN Devideo \n";

    mEASF.Init(mFullNameVideo);
    mPrefix  = "DIV_" + StdPrefix(mEASF.mPat) +"_";
    mPatNumber =  "[0-9]{"+ ToString(mNbDigit)  +"}";
    mMMPatImDev = NamePat("Ok|Nl");
    mAutoMM =  new cElRegex(mMMPatImDev,10);

    std::vector<std::string>  aSetImExisting = *(mEASF.mICNM->Get(mMMPatImDev));
    if (!EAMIsInit(&mDoVideo2Im))
    {
        mDoVideo2Im = (aSetImExisting.size() == 0);
    }

    // On remet en Ok toute les images Nl
    {
        for (int aK=0 ; aK<int(aSetImExisting.size()) ; aK++)
        {
            std::string aNInit = (aSetImExisting)[aK];
            std::string aNOk = CalcName(aNInit,"Ok");
            if (aNInit!= aNOk)
            {
               ELISE_fp::MvFile(Dir()+aNInit,Dir()+aNOk);
               aSetImExisting[aK] = aNOk;
            }
        }
    }

    // Extraction des images fixes +  remplit les xif
    if (mDoVideo2Im)
    {
        std::string aComDev =       "ffmpeg -i "
                              +  mFullNameVideo + " "
                              +  mEASF.mDir  + mPrefix + "%" + ToString(mNbDigit)  + "d_Ok." + mPostfix;

        System(aComDev);
        // std::cout << aComDev<< "\n";

        std::string aComFPS =       "ffmpeg -i "
                              +  mFullNameVideo + " 2>&1 | sed -n \"s/.*, \\(.*\\) fp.*/\\1/p\"";

        std::string aFps = getCmdOutput(aComFPS.c_str());
        std::stringstream ss;
        ss << aFps;
        ss >> mRateVideoInit;
        std::cout << "FPS video : " << mRateVideoInit << "\n";

        if (mF35>0)
        {
               std::string aComXif =        MM3dBinFile("SetExif ") 
                                        +   QUOTE(mMMPatImDev) 
                                        + " F=" + ToString(mFoc)
                                        + " F35=" + ToString(mF35)
                                        + " Cam='" + mCam + "'"
                                    ;

               System(aComXif);
               // std::cout << aComXif<< "\n";
        }
    }

    // Preselection d'un pourcentage a priori sur le numero des images
    {
        // std::cout << "AAAAAAAAAAAAA\n"; getchar();
        int aNbSel = round_up((aSetImExisting.size()* mPercImInit) /100.0);
        for (int aK=0 ; aK<int(aSetImExisting.size()) ; aK++)
        {
            int aIndGrpP = ((aK-1)  * aNbSel) / (int)(aSetImExisting.size() - 1);
            int aIndGrp = (aK  * aNbSel) / (int)(aSetImExisting.size() - 1);
            std::string aNInit = aSetImExisting[aK];
            std::string aNTarget = CalcName(aNInit,(aIndGrp==aIndGrpP) ?"Nl" : "Ok");

            if (aNInit!=aNTarget)
            {
                ELISE_fp::MvFile(Dir()+aNInit,Dir()+aNTarget);
                aSetImExisting[aK] = aNTarget;
            }
        }
    }

    // if (mDoVideo2Im)


    // std::cout << " KKK " <<  mMMPatImDev << " SZ " << aSetImExisting->size()  << "\n";


//========================================================
//========================================================
//========================================================



    mStrSzS = " " + ToString(mParamSzSift) + " ";
    mStdJump = mRateVideoInit/mTargetRate;

    ELISE_fp::MkDir(Dir()+"Tmp-MM-Dir/");



    {
        std::string aMMPatImOk = NamePat("Ok");
        if (!mTuning)
           MakeXmlXifInfo(aMMPatImOk,mEASF.mICNM,true);
        std::cout << "   Devideo :: Done XmlXif \n";

        Pt2dr aSomSz;
        // Calcul des noms dans l'intervalle
        {
            const std::vector<std::string> * aVN = mEASF.mICNM->Get(aMMPatImOk);
            int aK0 = ElMax(0,mMinMax.x);
            int aK1 = ElMin(int(aVN->size()),mMinMax.y);
            for (int aK=aK0; aK<aK1 ;aK++)
            {
                mVName.push_back((*aVN)[aK]);
                const cMetaDataPhoto & aMTD = cMetaDataPhoto::CreateExiv2(mVName.back());
                aSomSz = aSomSz + Pt2dr(aMTD.SzImTifOrXif());
            }
        }
        aSomSz = aSomSz / double(mVName.size());
        double aLarg = dist8(aSomSz);

        std::string aComTieP =      MM3dBinFile("Tapioca ") 
                                +   std::string("Line ")
                                +   QUOTE(aMMPatImOk) + " "
                                +   ToString(round_ni(aLarg/2.0))
                                +   std::string(" 1");

        if (!mTuning)
           System(aComTieP);
    }


    // developpement des images

    for (int aK=0 ; aK<int(mVName.size()) ; aK++)
    {
        mVIms.push_back(new cOneImageVideo(mVName[aK],this,aK));
    }
    mNbIm = (int)mVIms.size();

    if (!mTuning)
       Paral_Tiff_Dev(Dir(),mVName,1,true);
    std::cout << "   Devideo :: Done Dev \n";


    // calcul des parametres d'autocorrelation
    {
        std::list<std::string> aLComAC;
        for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
        {
             std::string aCom = MM3dBinFile("TestLib") + "CalcAutoCorrel " + mVIms[aK]->NameOk();
             aLComAC.push_back(aCom);
             // std::cout << aCom << "\n";
             // getchar();
        }
        if (!mTuning)
           cEl_GPAO::DoComInParal(aLComAC);
    }
    std::cout << "   Devideo :: Done AutoCorr \n";


    // lecture de ces parametres
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK]->LoadAutoCorrel();
    }

    // lecture de ces parametres
    for (int aK=1 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK]->CalcRec(mVIms[aK-1]);
    }
    std::cout << "   Devideo :: Done Cover \n";  


    // Calcul d'un score relatif (pour quil ait une influence equilibree sur tout le chemin)
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
         mVIms[aK]->CalcScoreAutoCorrel(mVIms,round_up(5*mStdJump));
    }

    // ============= Decoupage en intervalle ===
    mNbInterv = round_up((mNbIm-1) / double(mSzDecoup));
    mVPivots.push_back(PivotAPriori(0));
    for (int aKI=1 ; aKI< mNbInterv ; aKI++)
    {
        int aPivPrec = PivotAPriori(aKI-1);
        int aPiv  = PivotAPriori(aKI);
        int aPivNext = PivotAPriori(aKI+1);

        int aPiv0 =  round_ni(barry(ThePropRechPiv,aPivPrec,aPiv));
        int aPiv1 =  ElMax(aPiv+1,round_ni(barry(ThePropRechPiv,aPivNext,aPiv)));
        int aPivMax = -1;
        double aScoreMax=-1;
        for (int aPiv=aPiv0  ; aPiv<=aPiv1 ; aPiv++)
        {
             double aScore =  mVIms[aPiv]->AutoCorrel();
             if (aScore>aScoreMax)
             {
                aScoreMax = aScore;
                aPivMax = aPiv;
             }
        }
        // std::cout << " PvMax " << aPivMax << " Smx=" << aScoreMax << " in [" << aPiv0 << "," << aPiv1 << "]\n";
        ELISE_ASSERT(aScoreMax>=0,"Incohen in find Piv (DIV command)");
        mVPivots.push_back(aPivMax);

        std::cout << "PIV=" << aPivMax  << " In [" << aPiv0 << " - " << aPiv1 << "]" << "\n";

    }
    mVPivots.push_back(PivotAPriori(mNbInterv));
    
    int aPiv0 = ModifPivot(0,1);
    int aPivEnd = ModifPivot(mNbInterv,-1);
    mVPivots[0] = aPiv0;
    mVPivots[mNbInterv] = aPivEnd;


    //  ============= On modifie les pivots extremes


    // ============= optimisation dans chaque intervalle ===

    mVIms[aPiv0]->SetLongPred(0);
    for (int aK=1 ; aK<= mNbInterv ; aK++)
    {
       ComputeChemOpt(mVPivots[aK-1],mVPivots[aK]);
       std::cout << "Interv " << mVPivots[aK-1] << " " << mVPivots[aK] << "\n";
    }

    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
       mVIms[aK]->InitOk();
       mVIms[aK]->Show();
    }
}

double cAppliDevideo::NacModifExtre(cOneImageVideo * anI,cOneImageVideo * anI0)
{
    double aDist =  anI->DistCum(*anI0);
    double aScore = anI->NormAC();

    return aScore * ((mOverLap-aDist)/mOverLap);
}

int cAppliDevideo::ModifPivot(int aIndP0,int aSign)
{
    int  aP0 = mVPivots[aIndP0];
    cOneImageVideo * anI0 = mVIms[aP0];
    int  aP1 = (mVPivots[aIndP0+aSign] + aP0) / 2;

    int aBestP = aP0;
    double aBestAC =  NacModifExtre(anI0,anI0);

    bool Cont = true;
    int aCurP = aP0;
    // double 
    while (Cont)
    {
        if (aCurP==aP1)
        {
           Cont = false;
        }
        else
        {
            cOneImageVideo * aCurIm = mVIms[aCurP];
            if (aCurIm->DistCum(*anI0) > (mOverLap/2.0))
            {
               Cont = false;
            }
            else
            {
               double aCorr = NacModifExtre(aCurIm,anI0);
               if (aCorr>aBestAC)
               {
                  aBestAC = aCorr;
                  aBestP = aCurP;
               }

               aCurP += aSign;
            }
        }
    }


    return aBestP;
}
/*
*/
#if (0)
#endif


std::string  cAppliDevideo::NamePtsSift( cOneImageVideo * anOIV)
{
   std::string aNameOk = anOIV->NameOk();
   if (mParamSzSift!=-1)
   {
        std::string aTest;
        getPastisGrayscaleFilename(Dir(),aNameOk,mParamSzSift,aTest);
        aNameOk = NameWithoutDir(aTest);
   }

   return Dir()+  "Pastis/LBPp"+ aNameOk  + ".dat";
}
/*
*/

int cAppliDevideo::DecndreGivenLenght(int aSInit,double aLength,int aSLimit,bool Depasse)
{
     int aCurS = aSInit;
     while (1)
     {
         if (aCurS==aSLimit) 
            return aCurS;
         double aD = mVIms[aSInit]->DistCum(*(mVIms[aCurS]));
         if (aD >  aLength)
         {
             if ((!Depasse) && (aCurS+1 != aSInit)) aCurS++;
             return aCurS;
         }
         aCurS--;
     }
     ELISE_ASSERT(false,"cAppliDevideo:DecndreGivenLenght");
     return 0;
}

void cAppliDevideo::ComputeChemOpt(int aS0,int aS1)
{
    mVIms[aS0]->InitChemOpt();

    for (int aS=aS0+1 ; aS<=aS1 ; aS++)
    {
        int aSEnd = DecndreGivenLenght(aS,mOverLap/2.0,aS0,false);
        int aSDeb = DecndreGivenLenght(aS,mOverLap*2.0,aS0,true);
        if ((aSDeb==aSEnd) && (aSDeb > aS0))
           aSDeb --;
        if ((aSDeb==aSEnd) && (aSEnd < aS))
           aSEnd ++;

        ELISE_ASSERT(aSDeb<aSEnd,"cAppliDevideo::ComputeChemOpt");

        mVIms[aS]->UpdateCheminOpt(aSDeb,aSEnd,mVIms,mOverLap);
    }


    int aSom = aS1;
    int aNbL =  round_up(mVIms[aS0]->DistCum(*(mVIms[aS1])) / mOverLap);
    while (aSom!=aS0)
    {
         const cSolChemOptImV & aSol = mVIms[aSom]->SolOfLength(aNbL);

         mVIms[aSom]->SetLongPred(aSom-aSol.mIndPred);
         std::cout << "CCCCC " << aSom  << " Leng " << aSom-aSol.mIndPred << "\n";
         aSom = aSol.mIndPred;
         aNbL--;
    }

    std::cout << " ComputeChemOpt " << aS0 << " " << aS1 << "\n";


    for (int aS=aS0 ; aS<=aS1 ; aS++)
    {
        mVIms[aS]->ClearSol();
    }
}

std::string  cAppliDevideo::NamePat(const std::string & aPref) const
{
   // aPatNumber = "0(0[0-9]|12[0-9]{2}";
   return  "("+ mPrefix + mPatNumber + "_)(" + aPref   + ")(\\." + mPostfix+")";
}

std::string cAppliDevideo::CalcName(const std::string & aName,const std::string aNew)
{
   return MatchAndReplace(*mAutoMM,aName,"$1" + aNew + "$3");
/*
   std::cout << mMMPatImDev << " " << aName << "\n";
   bool aOk =  mAutoMM->Replace("$1" + aNew + "$3");
   std::cout << "RRRR="  <<  mAutoMM->LastReplaced() << "\n";
   ELISE_ASSERT(aOk,"cAppliDevideo::CalcName");
   return  mAutoMM->LastReplaced();
*/
}




//========================== :: ===========================

int Devideo_main(int argc,char ** argv)
{
    cAppliDevideo anAppli(argc,argv);
    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
