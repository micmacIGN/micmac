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

class cLinkImV;
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
static const double ThePropRechPiv = 0.08; 



int DevOneImPtsCarVideo_main(int argc,char ** argv)
{
    return EXIT_SUCCESS;
}

class cLinkImV
{
    public :
        cLinkImV(cOneImageVideo* aSom,cOneImageVideo * aPred);
        void InitSzSift();

        cOneImageVideo * mPred;
        std::string      mNameSift;
        int              mSzSift;

};

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
        cLinkImV AddPredd(cOneImageVideo *);

        const std::string & NameOk()    const {return mNameOk;}
        void LoadPts();
        void  UpDateMaxLoc(cOneImageVideo & anOIV);
        void  Show();
        bool  Pressel() const;
        void  SetNotOk() ;
        bool  IsMaxLoc() const;
        int   SzSift() const;
        int   TimeNum() const;
        int   DifTime(const cOneImageVideo &) const;
        bool DescripteurSiftDone();
        cAppliDevideo *  Appli();
        void InitSzLinks();

        void SetPresselNum(int aK);
        int  PresselNum() const;
        void InitChemOpt();
        void ClearSol();
        void UpdateCheminOpt(int aS0);
        const cSolChemOptImV & SolOfLength(int aNbL);
        void ShowSols();
    private :
        void  TestInitSift();

        cAppliDevideo *  mAppli;
        std::string      mNameInit;
        std::string      mNameOk;
        std::string      mNameNl;
        std::string      mNamePtsSift;
        bool             mIsMaxLoc;
        bool             mPressel;
        int              mTimeNum;
        int              mPresselNum;
        int              mSzSift;
        std::vector<cLinkImV>     mPreds;
        std::vector<cSolChemOptImV>  mSols;
};

class cCmpOIV_SzSift
{
    public :
       bool operator() (cOneImageVideo * anOIV1 ,cOneImageVideo * anOIV2)
       {
           return anOIV1->SzSift() < anOIV2->SzSift();
       }
 
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
        int PivotAPriori(int aKI) { return round_ni((aKI*(mNbPres-1))/mNbInterv);}
        void ComputeChemOpt(int aS0,int aS1);
        std::string mFullName;
        cElemAppliSetFile mEASF;
        std::string mPrefix;
        std::string mPostfix;
        std::string mMMPatImDev;
        std::string mMMPatImOk;
        cElRegex *  mAutoMM;
        const std::vector<std::string> * mVName;
        std::vector<cOneImageVideo*>      mVIms;
        std::vector<cOneImageVideo*>      mVImsPressel;
        int                               mNbPres;
        double      mTargetRate;
        double      mRateVideoInit;
        double      mStdJump;
        int         mParamSzSift;
        std::string mStrSzS;
        int         mNbInterv;
};



// =============  cSolChemOptImV ===================================
cSolChemOptImV::cSolChemOptImV(double aCost) :
   mCost (aCost),
   mIndPred (-1)
{
}


// =============  cLinkImV ===================================

cLinkImV::cLinkImV(cOneImageVideo * aSom,cOneImageVideo *  aPred) :
    mPred      (aPred),
    mNameSift  (aPred->Appli()->Dir() + "Homol/Pastis" + aSom->NameOk() + "/"+aPred->NameOk() + ".dat"),
    mSzSift    (sizeofile(mNameSift.c_str()))
{
   // std::cout << mSzSift << " => " << mNameSift << "\n";
}


void cLinkImV::InitSzSift()
{
   if (mSzSift<=0)
   {
        mSzSift = sizeofile(mNameSift.c_str());
        // std::cout << mNameSift << " => " << mSzSift << "\n";
   }
}

/*
void cLinkImV::TestLoad(cOneImageVideo * aSom)
{
    if 
}
*/

// =============  cOneImageVideo ===================================


cOneImageVideo::cOneImageVideo(const std::string & aNameIm,cAppliDevideo * anAppli,int anTimeNum) :
   mAppli       (anAppli),
   mNameInit    (aNameIm),
   mNameOk      (mAppli->CalcName(aNameIm,"Ok")),
   mNameNl      (mAppli->CalcName(aNameIm,"Nl")),
   // mNamePtsSift (mAppli->Dir()+  "Pastis/LBPp"+ mNameOk  + ".dat"),
   mIsMaxLoc    (true),
   mPressel     (true),
   mTimeNum     (anTimeNum),
   mSzSift      (0)
{
    // std::cout << mNameInit   << "  " << mNameOk << "\n";
    if (mNameInit!= mNameOk)
       ELISE_fp::MvFile(anAppli->Dir()+mNameInit,anAppli->Dir()+mNameOk);
   mNamePtsSift =mAppli->NamePtsSift(this);
}

void cOneImageVideo::InitSzLinks()
{
    for (int aKL=0 ; aKL<int(mPreds.size()) ; aKL++)
    {
        mPreds[aKL].InitSzSift();
    }
}

bool cOneImageVideo::DescripteurSiftDone()
{
    TestInitSift();
    return mSzSift>0;
}

void cOneImageVideo::TestInitSift()
{
   if (mSzSift<=0) mSzSift =  sizeofile(mNamePtsSift.c_str());
}

void cOneImageVideo::LoadPts()
{
    TestInitSift();
    ELISE_ASSERT(mSzSift>0,"cOneImageVideo::LoadPts");
}

void  cOneImageVideo::Show()
{
    std::cout << (mIsMaxLoc? "###" : (mPressel ? "ooo" : "---")) << mNameOk << " Time:" << mTimeNum << " SzS:" << mSzSift << "\n";
}

void  cOneImageVideo::UpDateMaxLoc(cOneImageVideo & anOIV)
{
    if (mSzSift < anOIV.mSzSift)  mIsMaxLoc = false;
    if (mSzSift > anOIV.mSzSift)  anOIV.mIsMaxLoc = false;
}

bool cOneImageVideo::Pressel() const {return mPressel;}
bool cOneImageVideo::IsMaxLoc() const {return mIsMaxLoc;}
int cOneImageVideo::SzSift() const {return mSzSift;}
int cOneImageVideo::TimeNum() const {return mTimeNum;}
cAppliDevideo *cOneImageVideo::Appli() {return mAppli;}


void cOneImageVideo::SetPresselNum(int aK)
{
   mPresselNum = aK;
}

int  cOneImageVideo::PresselNum() const
{
   return mPresselNum;
}


void cOneImageVideo::SetNotOk() 
{
   ELISE_fp::MvFile(mAppli->Dir()+mNameOk,mAppli->Dir()+mNameNl);
   mPressel=false;
}

int cOneImageVideo::DifTime(const cOneImageVideo & anOIV) const
{
    return ElAbs(mTimeNum-anOIV.mTimeNum);
}


     // Lie au calcul du chemin opt

cLinkImV  cOneImageVideo::AddPredd(cOneImageVideo * aPred)
{
     mPreds.push_back(cLinkImV(this,aPred));
     return mPreds.back();
}


void cOneImageVideo::InitChemOpt()
{
   mSols.push_back(cSolChemOptImV(0.0));
}

void cOneImageVideo::ClearSol()
{
    mSols.clear();
}

void cOneImageVideo::UpdateCheminOpt(int aS0)
{
    for (int aKL=0 ; aKL<int(mPreds.size()) ; aKL++)
    {
        const cLinkImV & aLnk = mPreds[aKL];
        double aCostLnk = 1e4 / (1.0+aLnk.mSzSift);
        const cOneImageVideo * aPred = aLnk.mPred;
        if (aPred->PresselNum() >= aS0)
        {
            for (int aKSolP=0 ; aKSolP<int(aPred->mSols.size())  ; aKSolP++)
            {
                const cSolChemOptImV & aSolPred = aPred->mSols[aKSolP];
                while (int(mSols.size()) <= (aKSolP+1)) 
                {
                    mSols.push_back(cSolChemOptImV(1e30));
                }
                cSolChemOptImV & aCurSol = mSols[aKSolP+1];
                double aNewCost  = aCostLnk + aSolPred.mCost;
                if (aNewCost < aCurSol.mCost)
                {
                     aCurSol.mCost = aNewCost;
                     aCurSol.mIndPred =  aPred->PresselNum() ;
                }
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

void cOneImageVideo::ShowSols()
{
    for (int aK=0 ; aK<int(mSols.size()) ; aK++)
    {
        std::cout << mSols[aK].mIndPred << "\n";
    }
}

// =============  cAppliDevideo ===================================

cAppliDevideo::cAppliDevideo(int argc,char ** argv) :
     mPrefix         ("Im"),
     mPostfix        ("png"),
     mTargetRate     (4.0),
     mRateVideoInit  (24.0),
     mParamSzSift    (-1)
{


    std::cout << "BEGIN Devideo \n";
    ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mFullName,"Full name (Dir+Pat)", eSAM_IsPatFile) ,
           LArgMain() << EAM(mTargetRate,"Rate",true,"Rate final Def=4")
                      << EAM(mParamSzSift,"ParamSzSift",true,"Parameter used for sift devlopment, def=-1 (Highest)")
    );

    mStrSzS = " " + ToString(mParamSzSift) + " ";
    mStdJump = mRateVideoInit/mTargetRate;

    mEASF.Init(mFullName);
    ELISE_fp::MkDir(Dir()+"Tmp-MM-Dir/");

    
    mMMPatImDev = NamePat("Ok|Nl");
    MakeXmlXifInfo(mMMPatImDev,mEASF.mICNM);
    std::cout << "   Devideo :: Done XmlXif \n";
    mMMPatImOk = NamePat("Ok");

    mAutoMM =  new cElRegex(mMMPatImDev,10);
    mVName = mEASF.mICNM->Get(mMMPatImDev);
    // ELISE_fp::MkDir(Dir()+"Pastis/");


    Paral_Tiff_Dev(Dir(),*mVName,1,true);
    
    std::cout << "   Devideo :: Done Dev \n";
    {
        std::list<std::string> aLComAC;
        for (int aKN=0 ; aKN<int(mVName->size()) ; aKN++)
        {
             std::string aCom = MM3dBinFile("TestLib") + "CalcAutoCorrel " + (*mVName)[aKN];
             aLComAC.push_back(aCom);
             // std::cout << aCom << "\n";
             // getchar();
        }
        cEl_GPAO::DoComInParal(aLComAC);
    }
    std::cout << "   Devideo :: Done AutoCorr \n";
return;


    bool AllSiftDone = true;
    for (int aK=0 ; aK<int(mVName->size()) ; aK++)
    {
        mVIms.push_back(new cOneImageVideo((*mVName)[aK],this,aK));
        if (! mVIms.back()->DescripteurSiftDone())
           AllSiftDone = false;
        // MisOneSift = MisOneSift || mVIms.back()->MisSift();
        // std::string aNamePts = mVIms.back()->NameDigeo();
    }
    std::cout << "    Devideo  :: images loaded , AllSif = " << AllSiftDone << "\n";

    std::string aComTapioca = MM3dBinFile("Tapioca") + " Line " + QUOTE(mMMPatImOk) +  mStrSzS + " 1 ";
    if (! AllSiftDone)
       System(aComTapioca);


    // Calcule sz sift et init max loc
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK]->LoadPts();
    }
    for (int aK=1 ; aK<int(mVIms.size()) ; aK++)
    {
        mVIms[aK-1]->UpDateMaxLoc(*(mVIms[aK]));
         
    }

    std::cout << "    Devideo  :: sift desc size done \n";

    //  Supression des pts en commencant par le  - de pts sift

    std::vector<cOneImageVideo*>  aVSort =  mVIms;
    cCmpOIV_SzSift aCmp;
    std::sort(aVSort.begin(),aVSort.end(),aCmp);

    for (int aKS=0 ; aKS<int(aVSort.size()) ; aKS++)
    {
         int aK= aVSort[aKS]->TimeNum();
         int aKP = aK+1;
         while ((aKP<int(mVIms.size()) && (!mVIms[aKP]->Pressel()))) aKP++;
         int aKM = aK-1;
         while ((aKM>=0) && (!mVIms[aKM]->Pressel())) aKM--;
         if ((aKP-aKM) <= mStdJump/2.0) 
         {
            aVSort[aKS]->SetNotOk();
         }
    }

    mVImsPressel.clear();
    for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
    {
         if (mVIms[aK]->Pressel())
         {
            mVIms[aK]->SetPresselNum(mVImsPressel.size());
            mVImsPressel.push_back(mVIms[aK]);
         }
    }
    mNbPres = mVImsPressel.size();
    std::cout << "    Devideo  :: pre selected images done \n";
    // === Calcul du graphe ====

    int aJumpMax = round_up(mStdJump * 2);
    cSauvegardeNamedRel aSNR;
    for (int aK1=0 ; aK1<mNbPres ; aK1++)
    {
         // Interval ]aK1 , aK2 [
         int aK2 = aK1+1;
         while (    (aK2<int(mVImsPressel.size()))  
                 && (  
                            (aK2<=aK1+1)
                       ||   (mVImsPressel[aK1]->DifTime(*(mVImsPressel[aK2])) < aJumpMax)
                    )
                )
         {
               cLinkImV aLnk = mVImsPressel[aK2]->AddPredd(mVImsPressel[aK1]);
               if (aLnk.mSzSift <=0)
               {
                  cCpleString aCple(mVImsPressel[aK2]->NameOk(),mVImsPressel[aK1]->NameOk());
                  aSNR.Cple().push_back(aCple);
               }
               aK2++;
         }
         // Pour etre sur interval non vide
         // std::cout << aK1 << " => " << aK2 << " ;" ;   
         //  printf("%3d => %3d ",aK1,aK2); mVImsPressel[aK1]->Show();
    }
    // calcul des points sifts sur le graphe
    if (! aSNR.Cple().empty())
    {
         std::string aNameGraph = Dir()+"GrapDIV.xml";
         MakeFileXML(aSNR,aNameGraph);
         std::string aComTapioca = MM3dBinFile("Tapioca") + " File " + aNameGraph +  mStrSzS;
         System(aComTapioca);
         for (int aK=0 ; aK<mNbPres ; aK++)
         {
             mVImsPressel[aK]->InitSzLinks();
         }
         ELISE_fp::RmFile(aNameGraph);
    }
    else
    {
    }

    std::cout << "    Devideo  :: sz sift link done \n";

    // ============= Decoupage en intervalle ===
    std::vector<int> vPivot;
    mNbInterv = round_up((mNbPres-1) / double(TheSzDecoup));
    vPivot.push_back(PivotAPriori(0));
    for (int aKI=1 ; aKI< mNbInterv ; aKI++)
    {
        int aPivPrec = PivotAPriori(aKI-1);
        int aPiv  = PivotAPriori(aKI);
        int aPivNext = PivotAPriori(aKI+1);
        
        int aPiv0 =  round_ni(barry(ThePropRechPiv,aPivPrec,aPiv));
        int aPiv1 =  ElMax(aPiv+1,round_ni(barry(ThePropRechPiv,aPivNext,aPiv)));
        int aPivMax = -1;
        int aScoreMax=-1;
        for (int aPiv=aPiv0  ; aPiv<=aPiv1 ; aPiv++)
        {
             int aScore =  mVImsPressel[aPiv]->SzSift();
             if (aScore>aScoreMax)
             {
                aScoreMax = aScore;
                aPivMax = aPiv;
             }
        }
        // std::cout << " PvMax " << aPivMax << " Smx=" << aScoreMax << " in [" << aPiv0 << "," << aPiv1 << "]\n";
        ELISE_ASSERT(aScoreMax>=0,"Incohen in find Piv (DIV command)");
        vPivot.push_back(aPivMax);

    }
    vPivot.push_back(PivotAPriori(mNbInterv));

    // ============= optimisation dans chaque intervalle === 

    for (int aK=1 ; aK<= mNbInterv ; aK++)
    {
       ComputeChemOpt(vPivot[aK-1],vPivot[aK]);
       std::cout << "Interv " << vPivot[aK-1] << " " << vPivot[aK] << "\n";
    }


    std::cout << "NB Im = " << mVName->size() << " Presel " << mVImsPressel.size() << " JMP=" << mStdJump << "\n";
    std::cout << "NBOK=" << mVImsPressel.size() << "\n";

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
    mVImsPressel[aS0]->InitChemOpt();

    for (int aS=aS0+1 ; aS<=aS1 ; aS++)
    {
        mVImsPressel[aS]->UpdateCheminOpt(aS0);
    }

    int aSom = aS1;
    int aNbL = round_up((mVImsPressel[aS1]->DifTime(*mVImsPressel[aS0])) / mStdJump) ;
    while (aSom!=aS0)
    {
         std::cout << "CCCCC " << aSom  << " nbl " << aNbL << "\n";
         const cSolChemOptImV & aSol = mVImsPressel[aSom]->SolOfLength(aNbL);
         aSom = aSol.mIndPred;
         aNbL--;
    }

    std::cout << " ComputeChemOpt " << aS0 << " " << aS1 << "\n";


    for (int aS=aS0 ; aS<=aS1 ; aS++)
    {
        mVImsPressel[aS]->ClearSol();
    }
}

std::string  cAppliDevideo::NamePat(const std::string & aPref) const
{
   std::string aPatNumber = "[0-9]{5}";
   // aPatNumber = "0(0[0-9]|12[0-9]{2}";
   return  "("+ mPrefix + aPatNumber + "_)(" + aPref   + ")(\\." + mPostfix+")";
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
