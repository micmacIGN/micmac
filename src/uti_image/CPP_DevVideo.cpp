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

// ffmpeg -i MVI_0001.MOV  -ss 30 -t 20 Im%5d_Ok.png

// Im*_Ok => OK 
// Im*_Nl => Image Nulle (eliminee)



//=================================================

/*
cMTDImCalc GetMTDImCalc(const std::string & aNameIm);
const cMIC_IndicAutoCorrel * GetIndicAutoCorrel(const cMTDImCalc & aMTD,int aSzW);
std::string NameMTDImCalc(const std::string & aFullName);
*/

double  AutoCorrel(const std::string &aName,int aWSz)
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
        aNewIAC.AutoC() = AutoCorrel(aNameIm,aSzW);
        aNewIAC.SzCalc() = aSzW;
        aMTD.MIC_IndicAutoCorrel().push_back(aNewIAC);
        std::string aNameXml = NameMTDImCalc(aNameIm);
        MakeFileXML(aMTD,aNameXml);

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
};



class cAppliDevideo
{
    public :
        cAppliDevideo(int argc,char ** argv);
        const std::string & Dir() {return mEASF.mDir;}
        std::string CalcName(const std::string & aName,const std::string aNew);

        std::string  NamePtsSift( cOneImageVideo * anOIV);
    private :
        std::string NamePat(const std::string & aPref) const;
        int PivotAPriori(int aKI) { return round_ni((aKI*(mNbIm-1))/mNbInterv);}
        void ComputeChemOpt(int aS0,int aS1);
        std::string mFullName;
        cElemAppliSetFile mEASF;
        std::string mPrefix;
        std::string mPostfix;
        std::string mMMPatImDev;
        std::string mMMPatImOk;
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
        std::string mNumVid;
        Pt2di       mMinMax;
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
   mAutoCor     (-1)
{
    // std::cout << mNameInit   << "  " << mNameOk << "\n";
    if (mNameInit!= mNameOk)
       ELISE_fp::MvFile(anAppli->Dir()+mNameInit,anAppli->Dir()+mNameOk);
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
    std::cout << (mOkFin? "###" :  "---") << mNameOk << " Time:" << mTimeNum << " C:" << mAutoCor  <<  " N:" << mNormAC << "\n";
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
        double aDeltaL = ElAbs(ElAbs(aS-mTimeNum)-aStdJ) / aStdJ;

        double aGainArc = mNormAC +  aPred.mNormAC - (aDeltaL + 2*ElSquare(aDeltaL)) * 1.0;
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

cAppliDevideo::cAppliDevideo(int argc,char ** argv) :
     mPrefix         ("Im_"),
     mPostfix        ("png"),
     mTargetRate     (4.0),
     mRateVideoInit  (24.0),
     mParamSzSift    (-1),
     mMinMax         (0,100000000)
{


    std::cout << "BEGIN Devideo \n";
    ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mFullName,"Full name (Dir+Pat)", eSAM_IsPatFile) ,
           LArgMain() << EAM(mTargetRate,"Rate",true,"Rate final Def=4")
                      << EAM(mNumVid,"NumVid",true,"Num of video, def = 00 ")
                      << EAM(mParamSzSift,"ParamSzSift",true,"Parameter used for sift devlopment, def=-1 (Highest)")
                      << EAM(mPatNumber,"PatNumber",true,"Pat number (reduce number for test)")
                      << EAM(mMinMax,"MinMax",true,"Min Max numbers (reduce number for test)")
    );

   
    mEASF.Init(mFullName);
    if (!EAMIsInit(&mNumVid)) 
    {
        mNumVid = ExtractDigit(StdPrefix(mEASF.mPat),"0000");
    }

    if (!EAMIsInit(&mPatNumber)) 
    {
        mPatNumber = mNumVid +"_" + "[0-9]{5}";
    }

    mStrSzS = " " + ToString(mParamSzSift) + " ";
    mStdJump = mRateVideoInit/mTargetRate;

    ELISE_fp::MkDir(Dir()+"Tmp-MM-Dir/");

    
    mMMPatImDev = NamePat("Ok|Nl");
    mAutoMM =  new cElRegex(mMMPatImDev,10);
    mMMPatImOk = NamePat("Ok");


//    *** std::cout << mMMPatImDev << " " << mMMPatImOk << "\n"; getchar();
    {
        const std::vector<std::string> * aVN = mEASF.mICNM->Get(mMMPatImDev);
        for (int aK=0 ; aK<int(aVN->size()) ; aK++)
        {
            std::string aNInit = (*aVN)[aK];
            std::string aNOk = CalcName(aNInit,"Ok");
            if (aNInit!= aNOk)
            {
               ELISE_fp::MvFile(Dir()+aNInit,Dir()+aNOk);
            }
        }
    }


    MakeXmlXifInfo(mMMPatImOk,mEASF.mICNM);
    std::cout << "   Devideo :: Done XmlXif \n";


    // Calcul des noms dans l'intervalle
    {
        const std::vector<std::string> * aVN = mEASF.mICNM->Get(mMMPatImOk);
        int aK0 = ElMax(0,mMinMax.x);
        int aK1 = ElMin(int(aVN->size()),mMinMax.y);
        for (int aK=aK0; aK<aK1 ;aK++)
        {
            mVName.push_back((*aVN)[aK]);
        }
    }
    
    
    // developpement des images

    for (int aK=0 ; aK<int(mVName.size()) ; aK++)
    {
        mVIms.push_back(new cOneImageVideo(mVName[aK],this,aK));
    }
    mNbIm = mVIms.size();

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
        cEl_GPAO::DoComInParal(aLComAC);
    }
    std::cout << "   Devideo :: Done AutoCorr \n";


    // lecture de ces parametres
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK]->LoadAutoCorrel();
    }


    // Calcul d'un score relatif (pour quil ait une influence equilibree sur tout le chemin)
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
         mVIms[aK]->CalcScoreAutoCorrel(mVIms,round_up(5*mStdJump));
    }


    // ============= Decoupage en intervalle ===
    std::vector<int> vPivot;
    mNbInterv = round_up((mNbIm-1) / double(TheSzDecoup));
    vPivot.push_back(PivotAPriori(0));
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
        vPivot.push_back(aPivMax);

        std::cout << "PIV=" << aPivMax  << " In [" << aPiv0 << " - " << aPiv1 << "]" << "\n";

    }
    vPivot.push_back(PivotAPriori(mNbInterv));


    // ============= optimisation dans chaque intervalle === 

    mVIms[0]->SetLongPred(0);
    for (int aK=1 ; aK<= mNbInterv ; aK++)
    {
       ComputeChemOpt(vPivot[aK-1],vPivot[aK]);
       std::cout << "Interv " << vPivot[aK-1] << " " << vPivot[aK] << "\n";
    }

    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
       mVIms[aK]->InitOk();
       mVIms[aK]->Show();
    }
}


extern void getPastisGrayscaleFilename(const std::string & aParamDir, const string &i_baseName, int i_resolution, string &o_grayscaleFilename );

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

void cAppliDevideo::ComputeChemOpt(int aS0,int aS1)
{
    mVIms[aS0]->InitChemOpt();

    int aJumpMax = round_up(2.0*mStdJump);
    int aJumpMin = round_down(0.5*mStdJump);
    for (int aS=aS0+1 ; aS<=aS1 ; aS++)
    {
        int aSDeb = ElMax(aS0,aS-aJumpMax);
        int aSEnd = ElMax(aSDeb+1,aS-aJumpMin);
        mVIms[aS]->UpdateCheminOpt(aSDeb,aSEnd,mVIms,mStdJump);
    }

    int aSom = aS1;
    int aNbL = round_up((mVIms[aS1]->DifTime(*mVIms[aS0])) / mStdJump) ;
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
