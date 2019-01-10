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


/*
    Acceleration :
      - Presel sur point les plus stables
      - calcul de distance de stabilite ? => Uniquement si pas Invar Ech !!!!
      - Apres pre-sel, a simil (ou autre) :
             * selection des point dans  regions homologues
             * indexation

    Plus de points :
        SIFT Criteres ?
*/

#include "NewRechPH.h"
#include "Match_Image.h"




//================================


//================================

// cIndexCodeBinaire::cIndexCodeBinaire(const cCompCB &)
// NKS-Assoc-CplIm2Hom



/*************************************************/
/*                                               */
/*             ::                                */
/*                                               */
/*************************************************/

ElPackHomologue PackFromVCC(const  std::vector<cCdtCplHom> &aVCpl)
{
   ElPackHomologue aPack;
   for (const auto  & aCpl : aVCpl)
   {
       // aNB1++;
       aPack.Cple_Add(ElCplePtsHomologues(aCpl.mPM->mOPC.Pt(),aCpl.mPS->mOPC.Pt()));
   }
 
   return aPack;
}

bool CmpCC(const cCdtCplHom & aC1,const cCdtCplHom & aC2)
{
    return aC1.mDistS > aC2.mDistS;
}

class cStatHist
{
   public :
      cStatHist() :
         mSomOk (0),
         mNbOk (0),
         mSomNotOk (0),
         mNbNotOk (0)
      {
      }
      void Show(const std::string & aName)
      {
           std::cout  << " NB " << mNbOk << " " << mNbNotOk
                      << "========== " << aName <<  " " << (mSomOk/mNbOk) << " " << (mSomNotOk/mNbNotOk) << "\n";
      }
      void Add(bool Ok,double aCor)
      {
         (Ok ? mNbOk  : mNbNotOk)  ++;
         (Ok ? mSomOk : mSomNotOk) += aCor;
      }
      double mSomOk;
      double mNbOk;
      double mSomNotOk;
      double mNbNotOk;
};


void FiltrageDirectionnel(std::vector<cCdtCplHom> & aVCpl,cAppli_FitsMatch1Im & anAppli)
{
   if (aVCpl.empty()) 
      return;
   int aNbDir = aVCpl[0].mPM->mSzIm.y;

   double aPropConv = 0.1;  // Taille normalise du filtre de convolution
   double  aPropDir = 0.07; // Seuil prorportionnel de l'écart en direction

   int   aMul = 100;
       

   // Seuil de distance directionnelle
   int aSeuilDir = ElMax(1,round_up(aPropDir*aNbDir));
   // Poids du filtre de convolution
   int aNbConv = round_up(aNbDir*aPropConv);


   std::vector<int> aHConv;
   for (int aKC=0 ; aKC<= aNbConv ; aKC++)
   {
       double aVal = aNbDir*aPropConv-aKC;
       if (aVal>0) 
          aHConv.push_back(ElMax(0,round_up(aMul*aVal)));
   }
   aNbConv = aHConv.size() -1;


   // Histo non filtre
   std::vector<int> aHDir(aNbDir,0.0);
   for (const auto  & aCpl : aVCpl)
   {
       // std::cout << "DIR=" << aCpl.mShift << "\n";
       ELISE_ASSERT(aCpl.mShift<aNbDir,"Dir over 64");
       aHDir[aCpl.mShift]++;
   }
   // convolution
   std::vector<int> aHConvDir(aNbDir,0);
   for (int aKD=0 ; aKD<aNbDir ; aKD++)
   {
      int aVDir = aHDir[aKD];
      if (aVDir)
      {
         for (int aKC=-aNbConv ; aKC<aNbConv ; aKC++)
         {
            aHConvDir.at(mod(aKD+aKC,aNbDir)) += aHConv.at(ElAbs(aKC)) * aVDir;
         }
      }
   }

   // calcul du pt max
   int aHMax=-1;
   int aKMax = -1;
   for (int aKD=0 ; aKD<aNbDir ; aKD++)
   {
       if (aHConvDir[aKD]>aHMax)
       {
          aKMax = aKD;
          aHMax = aHConvDir[aKD];
       }
   }

   cStatHist aSH;
   std::vector<cCdtCplHom> aNewV;
   for (const auto  & aCpl : aVCpl)
   {
       int aDif = mod(aCpl.mShift-aKMax,aNbDir);
       aDif = ElMin(aDif,aNbDir-aDif);
       bool Ok  = (aDif < aSeuilDir);

       double aDHG =  anAppli.DistHistoGrad(*(aCpl.mPM),aCpl.mShift,*(aCpl.mPS));
       // std::cout << "FILTR DIP " << aDHG << " OK=" << Ok  << "\n";
       aSH.Add(Ok, aDHG);
       if (Ok)
          aNewV.push_back(aCpl);

   }
   aSH.Show("Filtrage Directionnel");
   aVCpl = aNewV;
}


/*************************************************/
/*                                               */
/*             ::                                */
/*                                               */
/*************************************************/

cFitsOneLabel * FOLOfLab(cFitsParam * aFP,eTypePtRemark aLab,bool SVP)
{
   for (auto & aFOL : aFP->GenLabs())
       if (aFOL.KindOf() == aLab)
          return &aFOL;

   if (!SVP) 
   {
       std::cout << "For lab = " << eToString(aLab) << "\n";
       ELISE_ASSERT(false,"FOLOfLab");
   }
   return 0;
}
const cFitsOneLabel * FOLOfLab(const cFitsParam * aFP,eTypePtRemark aLab,bool SVP)
{
   for (const auto & aFOL : aFP->GenLabs())
       if (aFOL.KindOf() == aLab)
          return &aFOL;

   if (!SVP) 
   {
       std::cout << "For lab = " << eToString(aLab) << "\n";
       ELISE_ASSERT(false,"FOLOfLab");
   }
   return 0;
}



void InitOneLabelFitsPm(cFitsOneBin & aFB,const std::string & aDir,eTypePtRemark aLab)
{
    std::string aName = aDir + aFB.PrefName() +  eToString(aLab) +  aFB.PostName().Val();


    cCompCB aCB = StdGetFromNRPH(aName,CompCB);
    aFB.CCB().SetVal(aCB);

    // std::cout <<   aCB.CompCBOneBit().size()  << "  "  << aName << "\n";
}

void InitOneFitsPm(cFitsOneLabel & aFOL,const std::string & aDir)
{
    eTypePtRemark aLab = aFOL.KindOf();
    InitOneLabelFitsPm(aFOL.BinIndexed(),aDir,aLab);
    ELISE_ASSERT(aFOL.BinIndexed().CCB().Val().CompCBOneBit().size()<=16,"InitOneFitsPm");
    InitOneLabelFitsPm(aFOL.BinDecisionShort(),aDir,aLab);
    InitOneLabelFitsPm(aFOL.BinDecisionLong(),aDir,aLab);
}

const  std::string TheDirXmlFits=    string("include")    + ELISE_CAR_DIR
                                      + string("XML_MicMac") + ELISE_CAR_DIR 
                                      + string("Fits")       + ELISE_CAR_DIR;
const  std::string DefNameFitParam =  "FitsParam.xml";

void InitFitsPm(cFitsParam & aFP,const std::string & aDir, const std::string & aName)
{
    aFP = StdGetFromNRPH(aDir+aName,FitsParam);
    InitOneFitsPm(aFP.DefInit(),aDir);

    for (int aKL=0 ; aKL<eTPR_NoLabel ; aKL++)
    {
        eTypePtRemark aLab = eTypePtRemark(aKL);
        const cFitsOneLabel * aFOL = FOLOfLab(&aFP,aLab,true);
        if (aFOL==0)
        {
            aFP.GenLabs().push_back(aFP.DefInit());
            aFP.GenLabs().back().KindOf() = aLab;
        }
    }

    for (auto & aFOL : aFP.GenLabs())
        InitOneFitsPm(aFOL,aDir);
}

void StdInitFitsPm(cFitsParam & aFP)
{
    InitFitsPm(aFP,MMDir() + TheDirXmlFits,DefNameFitParam);
}


/*************************************************/
/*                                               */
/*              cIndexCodeBinaire                */
/*                                               */
/*************************************************/

cIndexCodeBinaire::cIndexCodeBinaire(const cCompCB & aCCB) :
   mNBBTot    (aCCB.CompCBOneBit().size()),
   mNBBVois   (aCCB.BitThresh()),
   mFlagV     (FlagOfNbb(mNBBTot,mNBBVois)),
   mVTabIndex (1<<mNBBTot)
{
}

void cIndexCodeBinaire::Add(cCompileOPC * aPC)
{
   int aFlagPC =   aPC->mIndexFlag ;
   for (const auto & aFlagVois : *mFlagV)
   {
       mVTabIndex.at(aFlagPC^aFlagVois).push_back(aPC);
   }
}

const std::vector<cCompileOPC *> & cIndexCodeBinaire::VectVois(const cCompileOPC & aPC)
{
     return mVTabIndex.at(aPC.mIndexFlag);
}

void cIndexCodeBinaire::Add(cSetOPC & aSet,const cFitsOneLabel & aFOL)
{
    for (auto & aCel : mVTabIndex)
       aCel.clear();

    for (auto & aPC : aSet.VOpc())
    {
        Add(aPC);
    }
}


/*************************************************/
/*                                               */
/*                cSetOPC                        */
/*                                               */
/*************************************************/

cSetOPC::cSetOPC() :
   mIndexCB (0),
   mFOL     (0),
   mSeuil   (0)
{
}

void cSetOPC::InitLabel(const cFitsOneLabel & aFOL,const cSeuilFitsParam & aSeuil,bool DoIndex)
{
    if (DoIndex)
    {
       mIndexCB = new cIndexCodeBinaire(aFOL.BinIndexed().CCB().Val());
       mIndexCB->Add(*this,aFOL);
    }
    mFOL = & aFOL;
    mSeuil = & aSeuil;
}

void  cSetOPC::Add(cCompileOPC* anOPC)
{
   mVOpc.push_back(anOPC);
}

const std::vector<cCompileOPC*> &  cSetOPC::VOpc() const { return mVOpc; }
std::vector<cCompileOPC*> &  cSetOPC::VOpc() { return mVOpc; }


cIndexCodeBinaire & cSetOPC::Ind()
{
   ELISE_ASSERT(mIndexCB!=0,"cSetOPC::Ind");
   return *mIndexCB;
}

const cFitsOneLabel & cSetOPC::FOL() const
{
   ELISE_ASSERT(mFOL!=0,"cSetOPC::FOL");
   return *mFOL;
}
const cSeuilFitsParam & cSetOPC::Seuil() const
{
   ELISE_ASSERT(mSeuil!=0,"cSetOPC::Seuil");
   return *mSeuil;
}

cCompileOPC& cSetOPC::At(int aK) {return *(mVOpc.at(aK));}

cSetOPC::~cSetOPC()
{
    DeleteAndClear(mVOpc);
}
 
void cSetOPC::ResetMatch()
{
    for (auto & aPC : mVOpc)
        aPC->ResetMatch();
}



/*************************************************/
/*                                               */
/*           cAFM_Im                             */
/*                                               */
/*************************************************/

bool CmpCPOC(const cCompileOPC &aP1,const  cCompileOPC &aP2)
{
   return aP1.mOPC.ScaleStab() > aP2.mOPC.ScaleStab();
}



cAFM_Im::cAFM_Im (const std::string  & aNameIm,cAppli_FitsMatch1Im & anAppli) :
   mAppli  (anAppli),
   mNameIm (aNameIm),
   mMTD    (cMetaDataPhoto::CreateExiv2(mNameIm)),
   mSzIm   (mMTD.SzImTifOrXif())
{
   for (int aKL=0 ; aKL<int(eTPR_NoLabel) ; aKL++)
   {
      mVSetCC.push_back(nullptr);
      mSetInd0.push_back(nullptr);
   }
}

typedef cSetOPC * tPtrSO;

void cAFM_Im::LoadLab(bool DoIndex,bool aGlob, eTypePtRemark aLab,bool MaintainIfExist)
{

   std::vector<cSetOPC*> &  aV = aGlob ? mVSetCC : mSetInd0;
   tPtrSO  & aSet = aV[int(aLab)];
   if ((aSet != nullptr) && MaintainIfExist)
   {
      return;
   }
   delete aSet;
   aSet = new cSetOPC;

   const cFitsOneLabel * aFOL =  FOLOfLab(&(mAppli.FitsPm()),aLab,false);
   std::string aExt = mAppli.ExtNewH();
   if (! aGlob)
      aExt = "_HighS" + aExt;
   cSetPCarac* aSetPC = LoadStdSetCarac(aLab,mNameIm,aExt);

    for (const auto & aPC : aSetPC->OnePCarac())
    {
        cCompileOPC  * aCPC = new cCompileOPC(aPC);
        aCPC->SetFlag(*aFOL);
        aSet->Add(aCPC);
    }

    // On rajoute dans Glob ceux d'indexe 0 
    if (aGlob)
    {
       for (const auto & aPC : mSetInd0[int(aLab)]->VOpc())
       {
           aSet->Add(aPC);
       }
    }

    const cSeuilFitsParam & aSeuil =  aGlob ?  mAppli.FitsPm().SeuilGen()   : mAppli.FitsPm().SeuilOL();
    // cSeuilFitsParam * aSeuil =  new cSeuilFitsParam(aGlob ?  mAppli.FitsPm().SeuilGen()   : mAppli.FitsPm().SeuilOL());
    aSet->InitLabel(*aFOL,aSeuil,DoIndex);


    delete aSetPC;
}

cAFM_Im::~cAFM_Im()
{
    DeleteAndClear(mVSetCC);
    DeleteAndClear(mSetInd0);
}

const std::string & cAFM_Im::NameIm() const {return mNameIm;}

void cAFM_Im::ResetMatch()
{
   for (auto & aV : mVSetCC)
   {
       if (aV)
          aV->ResetMatch();
   }
   for (auto & aV : mSetInd0)
   {
       if (aV)
          aV->ResetMatch();
   }
}

/*************************************************/
/*                                               */
/*           cAFM_Im_Master                      */
/*                                               */
/*************************************************/

cAFM_Im_Master::cAFM_Im_Master(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im     (aName,anApli),
    mQt         (mArgQt,Box2dr(Pt2dr(-10,-10),Pt2dr(10,10)+Pt2dr(mSzIm)),5,euclid(mSzIm)/20.0),
    mPredicGeom (mSzIm,100)
{
   for (int aKL = 0; aKL<eTPR_NoLabel ; aKL++)
   {
       LoadLab( true, false,eTypePtRemark(aKL),false);
       LoadLab(false,true,eTypePtRemark(aKL),false);
   }
}


ElSimilitude   cAFM_Im_Master::RobusteSimilitude(std::vector<cCdtCplHom> & aV0,double aDistSeuilNbV)
{
    mQt.clear(); 
    for (auto  & aCpl : aV0)
    {
        mQt.insert(&aCpl);
    }

    ElSimilitude aRes;
    double aScoreMin = 1e20;

    for (auto  & aCpl1 : aV0)
    {
        std::list<cCdtCplHom *> aLVois = mQt.KPPVois(aCpl1.PM(),3,aDistSeuilNbV);
        for (auto  & aCpl2 : aLVois)
        {
            if (&aCpl1 != aCpl2)
            {
                //  S  = aTr + aFact * aM
                Pt2dr aFact = (aCpl1.PS() - aCpl2->PS()) /  (aCpl1.PM() - aCpl2->PM());
                Pt2dr aTr =  aCpl1.PS() - aCpl1.PM()*aFact ;
                ElSimilitude aSim(aTr,aFact);

                // std::cout << "hhhhh  " << aSim(aCpl1.PM()) -  aCpl1.PS() << aSim(aCpl2->PM()) -  aCpl2->PS()  << "\n";

                Pt2dr aBarryS = (aCpl1.PS() + aCpl2->PS()) / 2.0;

                double aScore = 0;
                for (auto  & aCpl3 : aV0)
                {
                   double aD = euclid(aCpl3.PS()-aBarryS);
                   double anEcart = euclid(aSim(aCpl3.PM()) -  aCpl3.PS());
                   double anAnErr = anEcart/ElMax(aDistSeuilNbV,aD) + 0.3* anEcart/aDistSeuilNbV;
                   aScore += anAnErr;
                }
                if (aScore < aScoreMin)
                {
                    aScoreMin = aScore;
                    aRes = aSim;
                }
            }
        }
    }
    mQt.clear(); 
    return aRes;
}

void cAFM_Im_Master::FilterVoisCplCt(std::vector<cCdtCplHom> & aV0)
{       
    double aDiag = euclid(mSzIm);
    double aSeuilOk = aDiag / 100.0;
    double aSeuilPb = aDiag / 5.0;
    double aSeuilCoh = 0.3;
    int aNbVois = 10;
    double aSurfPP = (double(mSzIm.x) * mSzIm.y) / aV0.size() ; // Surf per point
    double aDistSeuilNbV = sqrt(aSurfPP*aNbVois) * 2;

    ElSimilitude aS0 = RobusteSimilitude(aV0,aDistSeuilNbV);

    mQt.clear(); 
    ElPackHomologue aPack = PackFromVCC(aV0); 

    ElSimilitude  aSim = SimilRobustInit(aPack,0.666,100);

    aSim = aS0;
    mAppli.SetCurMapping(new ElSimilitude(aSim));

    for (auto  & aCpl : aV0)
    {
        aCpl.mDistS = euclid(aSim(aCpl.PM()) - aCpl.PS());
    }
    std::sort(aV0.begin(),aV0.end(),CmpCC);
    // On refait une deuxieme passe car adresse pas conservee dans std::sort ...
    for (auto  & aCpl : aV0)
    {
        mQt.insert(&aCpl);
    }


    
    for (auto  & aCpl : aV0)
    {
        if (aCpl.mDistS >aSeuilPb)
        {
            RemoveCpleQdt(aCpl);
        }
        else if (aCpl.mDistS<aSeuilOk)
        {
        }
        else
        {
            double aSomD=0;
            int    aNbD =0;
            std::list<cCdtCplHom *> aLVois = mQt.KPPVois(aCpl.PM(),aNbVois+1,aDistSeuilNbV);
            if (int(aLVois.size()) < (aNbVois/2))
            {
               RemoveCpleQdt(aCpl);
            }
            else
            {
                for (auto  & aCpV : aLVois)
                {
                    if (&aCpl!=aCpV)
                    {
                        Pt2dr aV1 = aSim(aCpl.PM()) - aSim(aCpV->PM());
                        Pt2dr aV2 = aCpl.PS() - aCpV->PS();
                        double aD = euclid(aV1-aV2) / (1+euclid(aV1)+euclid(aV2)); // +1 modelise erreur pixel
                        aSomD += aD;
                        aNbD ++;
                    }
                    else
                    {
                     
                    }
                }
            }
            aSomD /= aNbD;

            if (aSomD> aSeuilCoh)
               RemoveCpleQdt(aCpl);
 
            // std::cout << "SOMDDD =  " << aSomD << " " <<  aCpl.mDistS / aDiag  <<  "\n";
        }
    }

    cStatHist aSH;

    std::vector<cCdtCplHom> aRes;
    for (auto  & aCpl : aV0)
    {
        aSH.Add(aCpl.mOk,aCpl.mPM->m2BGrad.mScore1);
        if (aCpl.mOk)
           aRes.push_back(aCpl);
    }
    aSH.Show("Filtrage spatial");
    aV0 = aRes;
}


void cAFM_Im_Master::RemoveCpleQdt(cCdtCplHom & aCpl)
{
    mQt.remove(&aCpl);
    aCpl.mOk = false;
}



int IScal(double aS) {return round_ni(5 * log(aS)/log(2));}





void cAFM_Im_Master::FiltrageSpatialGlob(std::vector<cCdtCplHom> & aVCpl,int aNbMin)
{
  std::cout << "Avant  filt dir " << aVCpl.size() << "\n";
   FiltrageDirectionnel(aVCpl,mAppli);
   // if (mAppli.ShowDet())
      std::cout << "After  filt dir " << aVCpl.size() << "\n";

   if (int(aVCpl.size()) <=  aNbMin)
   {
      return ;
   }

   FilterVoisCplCt(aVCpl);
  std::cout << "At end   " << aVCpl.size() << "\n";
   if (int(aVCpl.size()) <=  aNbMin)
      return ;
}


void cAFM_Im_Master::MatchOne
     (
          bool OverLap,
          cAFM_Im_Sec & anISec, 
          cSetOPC & aSetM,
          cSetOPC & aSetS,
          std::vector<cCdtCplHom> & aOld,
          int aNbMin
     )
{

   cTimeMatch aTimeMatch;
   cTimeMatch * aPtrTM = mAppli.ShowDet() ? &aTimeMatch : nullptr;

   const cFitsParam &  aFPM = mAppli.FitsPm();
   // eTypePtRemark aT0 = aFPM.KindOl();
   // cSetOPC & aSetM = mSetInd0;
   // cSetOPC & aSetS = anISec.mSetInd0;

   //int aNB1=0;
   // int aNBSup=0;

   if (mAppli.ShowDet())
   {
      cFHistoInt aFH;
      for (const auto & aPC : aSetS.VOpc())
      {
          if(aPC->mOPC.ScaleStab()>0)
             aFH.Add(IScal(aPC->mOPC.ScaleStab()),1,__LINE__);
      }
      std::cout << "======= HISTO SCALE BEFORE ===========\n";
      aFH.Show();
   }

   cFHistoInt aHLF;
   int First= true;

   int aNbCpleIndex =0;
   int aNbCpleSel =0;
   int aNbCpleTot =     aSetS.VOpc().size() * aSetM.VOpc().size() ;
   for (int aKs=0 ; aKs<(int)aSetS.VOpc().size() ; aKs++)
   {
      cCompileOPC & aPCS = aSetS.At(aKs);
      // aPCS.S etFlag(aSetM.FOL());
      // int aFlagS = aPCS.mShortFlag;
      // std::vector<cCompileOPC *> & aVSel = mVTabIndex.at(aFlagS);

      const std::vector<cCompileOPC *> & aVSel = aSetM.Ind().VectVois(aPCS);

      aNbCpleIndex += aVSel.size();


      // std::cout << "SSSIND " << aVSel.size() << " on " << aSetM.VOpc().size() << "\n";
      
      for (int aKSel=0 ; aKSel<(int)aVSel.size() ; aKSel++)
      {
          int aLevFail;
          int aShift;
          cCompileOPC * aPCM = aVSel[aKSel];


          double aD =  aPCM->Match(aPCS,aSetM.FOL(),aSetM.Seuil(),aShift,aLevFail,aPtrTM);
          aHLF.Add(aLevFail,1,__LINE__);
          if (aD > 0)
          {
             double aScoreGrad = mAppli.DistHistoGrad(*aPCM,aShift,aPCS);

             if (aScoreGrad < mAppli.SeuilDistGrad())
             {
                 double aScoreCor = ElMax(0.0,1-aD);
                 aPCS.SetMatch(aPCM,aScoreCor,aScoreGrad,-aShift);
                 aPCM->SetMatch(&aPCS,aScoreCor,aScoreGrad,aShift);
                 aNbCpleSel++;
                 // std::cout << "DddddddddD= " << aD << "\n";
                 aPCS.mTmpNbHom++;
                 aPCM->mTmpNbHom++;
             }
          }
          if (First && mAppli.ShowDet())
          {
             std::vector<double>  aVT =  aPCM->Time(aPCS,aFPM);
             std::cout << "================ TIMING THEO ================\n";
             for (int aK=0 ; aK<int(aVT.size()) ; aK++)
                 std::cout << "  T[" << aK << "]=" << aVT[aK] << "\n";
          }
          First = false;
      }
   }



   if (mAppli.ShowDet())
   {
      std::cout << "======= HISTO LEV FAIL ===========\n";
      aHLF.Show();
      std::cout << "NbSom " << aSetS.VOpc().size()  << " " <<  aSetM.VOpc().size()  << "\n";
      std::cout << "NbCouple " << aSetS.VOpc().size() * aSetM.VOpc().size() 
                               << " PropI=" << aNbCpleIndex/double(aNbCpleTot)
                               << " PropS=" << aNbCpleSel/double(aNbCpleTot)
                               << "\n";
      std::cout << "======= TIME EFFECTIF  ===========\n";
      aPtrTM->Show();
   }
   for (auto & aPCM : aSetM.VOpc())
   {
         if (aPCM->OkCpleBest(mAppli.SeuilCorrelRatio12(),mAppli.SeuilGradRatio12()))
         {
            cCompileOPC * aPCS = aPCM->m2BCor.mBest;
/*
*/
            aOld.push_back(cCdtCplHom(aPCM,aPCS,aPCM->m2BCor.mScore1,aPCM->CorrShiftBest()));
         }
   }
   return ;
}

bool  cAFM_Im_Master::MatchLow(cAFM_Im_Sec & anISec,std::vector<cCdtCplHom> & aVCpl)
{
    if (mAppli.HasFileCple() &&  (! mAppli.InSetCple(anISec.NameIm())))
       return false;
    // int aNbBeforeDir=0;
    int aNbMin0 = mAppli.HasFileCple() ? 6 : 3;

    // eTypePtRemark aLab = mAppli.LabInit();
    // int aKL = int(aLab);
    
    // Premier calcul sur nb de points reduit
    // for (const auto & aKL : aVLab)
    for (int aKL=0 ; aKL< int(eTPR_NoLabel) ; aKL++)
    {
       cSetOPC * aSetM = (mSetInd0[aKL]);
       cSetOPC * aSetS = (anISec.mSetInd0[aKL]);
       if (aSetS && aSetM)
       {
          MatchOne(true,anISec,*aSetM,*aSetS,aVCpl,aNbMin0);
       }
    }

    if (mAppli.ShowDet())
    {
       std::cout << "After-match_one " << aVCpl.size() << "\n";
       // getchar();
    }
    if (int(aVCpl.size()) <= aNbMin0) 
    {
       return false;
    }

    // aNbBeforeDir = aVCpl.size();

    if (mAppli.DoFiltrageSpatial())
       FiltrageSpatialGlob(aVCpl,aNbMin0);
    if (mAppli.ShowDet())
       std::cout << "After Filtrage sparial  " << aVCpl.size() << "\n";
    if (int(aVCpl.size()) <=  aNbMin0)
       return false ;

    return true;
}

bool  cAFM_Im_Master::MatchGlob(cAFM_Im_Sec & anISec)
{
    std::vector<cCdtCplHom> aVCpl;

    if ( ! MatchLow(anISec,aVCpl))
       return false;

    if ( ! anISec.mAllLoaded)
    {
       ResetMatch();
       anISec.ResetMatch();
       aVCpl.clear();
       anISec.LoadLabsLow(true);
       if ( ! MatchLow(anISec,aVCpl))
          return false;
    }
    
    ElPackHomologue aPack = PackFromVCC(aVCpl);
    aPack.StdPutInFile(mAppli.NameCple(mNameIm,anISec.mNameIm));

    if (mAppli.DoFiltrageSpatial())
    {
        mPredicGeom.Init(2.0,&(mAppli.CurMapping()),aVCpl);
    }

    return true ;
}

/*
bool  cAFM_Im_Master::MatchGlob(cAFM_Im_Sec & anISec)
{
    // int aNbBeforeDir=0;
    int aNbMin0 = 6;
    std::vector<cCdtCplHom> aVCpl;

    

    // eTypePtRemark aLab = mAppli.LabInit();
    // int aKL = int(aLab);
    
    // Premier calcul sur nb de points reduit
    // for (const auto & aKL : aVLab)
    for (int aKL=0 ; aKL< int(eTPR_NoLabel) ; aKL++)
    {
       cSetOPC * aSetM = (mSetInd0[aKL]);
       cSetOPC * aSetS = (anISec.mSetInd0[aKL]);
       if (aSetS && aSetM)
       {
          MatchOne(true,anISec,*aSetM,*aSetS,aVCpl,aNbMin0);
       }
    }

    if (mAppli.ShowDet())
       std::cout << "After match one " << aVCpl.size() << "\n";
    if (int(aVCpl.size()) <= aNbMin0) 
    {
       return false;
    }

    // aNbBeforeDir = aVCpl.size();

    if (mAppli.DoFiltrageSpatial())
       FiltrageSpatialGlob(aVCpl,aNbMin0);
    if (mAppli.ShowDet())
       std::cout << "After Filtrage sparial  " << aVCpl.size() << "\n";
    if (int(aVCpl.size()) <=  aNbMin0)
       return false ;
    ElPackHomologue aPack = PackFromVCC(aVCpl);
    aPack.StdPutInFile(mAppli.NameCple(mNameIm,anISec.mNameIm));

    {
         mPredicGeom.Init(2.0,&(mAppli.CurMapping()),aVCpl);
    }

    return true ;
}
*/

/*
bool  cAFM_Im_Master::MatchGlob(cAFM_Im_Sec & anISec)
{
}
*/

/*************************************************/
/*                                               */
/*           cAFM_Im_Sec                         */
/*                                               */
/*************************************************/

cAFM_Im_Sec::cAFM_Im_Sec(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im(aName,anApli)
{
    LoadLabsLow(false);
}

void cAFM_Im_Sec::LoadLabsLow(bool AllLabs)
{
    mAllLoaded = true;
    for (int aKL=0 ; aKL<int(eTPR_NoLabel) ; aKL++)
    { 
       eTypePtRemark aLab =  eTypePtRemark(aKL);
       if (AllLabs || mAppli.LabInInit(aLab) )
       {
          LoadLab(false,false,aLab,true);
       }
       else
       {
          mAllLoaded = false;
       }
    }
}

/*************************************************/
/*                                               */
/*           cAppli_FitsMatch1Im                 */
/*                                               */
/*************************************************/

   // return  mEASF.mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH +"@" + mPostHom,aN1,aN2,true);

cAppli_FitsMatch1Im::cAppli_FitsMatch1Im(int argc,char ** argv) :
   mImMast      (nullptr),
   mCurImSec    (nullptr),
   mNameXmlFits (DefNameFitParam),
   mExtNewH     ("Std"),
   mSH          (""),
   mPostHom     ("dat"),
   mExpTxt      (false),
   mOneWay      (true),
   mSelf        (false),
   mShowDet     (false),
   mCallBack    (false),
   mNbMaxS0     (1000,200),
   mDoFiltrageSpatial  (true),
   mFlagLabsInit (1 << int(eTPR_GrayMax)),
   mCurMap       (0),
   mHasFileCple  (false)
{
   MemoArg(argc,argv);
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mNameMaster, "First Image")
                     << EAMC(mPatIm, "Name Image2"),
         LArgMain()  << EAM(mNameXmlFits,"XmlFits",true,"Name of xml file for Fits parameters")
                     <<  EAM(mExtNewH,"ExtPC",true,"Extension for P cararc to NewPH... ")
                     <<  EAM(mOneWay,"1W",true,"Do computation one way (def = true) ")
                     <<  EAM(mExpTxt,"ExpTxt",true,"Export in texte format")
                     <<  EAM(mCallBack,"CallBack",true,"Internal")
                     <<  EAM(mNbMaxS0,"NbMaxPreSel",true,"Number of most significant point in presel mode(x before,y after)")
                     <<  EAM(mShowDet,"ShowDet",true,"Show Details, def=true if 1 pair")
                     <<  EAM(mDoFiltrageSpatial,"DoFS",true,"Do spatial filtering")
                     <<  EAM(mSelf,"Self",true,"Accept self match (tuning)")
                     <<  EAM(mFlagLabsInit,"FLI",true,"Flag Labs Init, def=>GrayMax, -1=> all")
                     <<  EAM(mFileCple,"FileCple",true,"Flag for cple")
                     <<  EAM(mVSeuils,"Seuils",true,"Seuils [CorrelLogPol,CorrelInvRad]")
   );

   if ((! EAMIsInit(&mFlagLabsInit)) && EAMIsInit(&mFileCple))
   {
     mFlagLabsInit = -1;
   }

   if (mFlagLabsInit<0)
     mFlagLabsInit = ((1<<int(eTPR_NoLabel)) - 1);

   if (mExpTxt)
      mPostHom = "txt";


   {
       cElemAppliSetFile  aPatMast(mNameMaster);
       const std::vector<std::string>* aVMast = aPatMast.SetIm();
       if (aVMast->size() > 1)
       {
           std::list<std::string> aLCom;
           for (const auto & aNM : *aVMast)
               aLCom.push_back(SubstArgcArvGlob(2,aNM, true) + " CallBack=true");

           cEl_GPAO::DoComInParal(aLCom);
           
           exit(EXIT_SUCCESS);
       }
   }
   mHasFileCple = EAMIsInit(&mFileCple);
   if (mHasFileCple)
   {
       cSauvegardeNamedRel aLCple =  StdGetFromPCP(mFileCple,SauvegardeNamedRel);
       for (auto & aCpl : aLCple.Cple())
       {
           // Verifie hypothese
           std::string aN1 = aCpl.N1();
           std::string aN2 = aCpl.N2();
           set_min_max(aN1,aN2);
           ELISE_ASSERT(aN1<aN2,"Non orderes cpl");
           if ((aN1== mNameMaster) && (aN1<aN2))
           {
                mSetCple.insert(aN2);
           }
       }
   }

   if (!EAMIsInit(&mSH) )
   {
        mSH = "HFits"+ mExtNewH;
   }
   
   InitFitsPm(mFitsPm,MMDir()+TheDirXmlFits,mNameXmlFits);
   mLabInit = mFitsPm.DefInit().KindOf();
/*
eTypePtRemark  cAppli_FitsMatch1Im::LabInit() const {return eTPR_GrayMax;}
   mLabOL = mFitsPm.OverLap().KindOf();
    // const cFitsParam & aFitsPM = anAppli.FitsPm();
   const cFitsOneLabel & aFOL = mFitsPm.OverLap();
   mNbBIndex =  aFOL.BinIndexed().CCB().Val().CompCBOneBit().size();
   mThreshBIndex =  aFOL.BinIndexed().CCB().Val().BitThresh();
*/

   if (EAMIsInit(&mVSeuils))
   {
      if (mVSeuils.size() >=1)
      {
          mFitsPm.SeuilGen().SeuilCorrLP() = mVSeuils[0];
          mFitsPm.SeuilOL().SeuilCorrLP()  = mVSeuils[0];
      }
      if (mVSeuils.size() >=2)
      {
          mFitsPm.SeuilGen().SeuilCorrDR() = mVSeuils[1];
          mFitsPm.SeuilOL().SeuilCorrDR()  = mVSeuils[1];
      }
   }

   mEASF.Init(mPatIm);
   if (! EAMIsInit(&mShowDet))
      mShowDet = (!mCallBack) && (mEASF.SetIm()->size()==1);


   mImMast = new cAFM_Im_Master(mNameMaster,*this);


   std::string   aNameFileTest = "TEST-FitsMatch-" + mNameMaster + ".txt";
   ELISE_fp aFileTest(aNameFileTest.c_str(),ELISE_fp::WRITE);
   fprintf(aFileTest.FP(),"%s\n",GlobArcArgv.c_str());
   aFileTest.close();

   int aNbFailConseq=0;
   for (const auto &  aName : *(mEASF.SetIm()))
   {
       if ((aName != mNameMaster) || (mSelf))
       {
           if ((! mOneWay) || (aName >= mNameMaster))
           {
              int aTime= 1;
              for (int aKT=0 ; aKT<aTime; aKT++)
              {
                  mCurImSec = new  cAFM_Im_Sec(aName,*this);

                   bool OkMatch = mImMast->MatchGlob(*mCurImSec);
                   if (OkMatch)
                   {
                       aNbFailConseq=0;
                   }
                   else
                   {
                       aNbFailConseq++;
                   }

                   if ((aNbFailConseq+1)%10==0)
                   {
                        std::cout << "None for " << mNameMaster << " " << aName << "\n";
                   }

                   std::cout << "HHHH  " << OkMatch << " " <<  aName << "\n";
                  
               // aPack.StdPutInFile(mAppli.NameCple(mNameIm,anISec.mNameIm));
              // cSetOPC & aSetM = mSetInd0;
              // cSetOPC & aSetS = anISec.mSetInd0;

                  delete mCurImSec;
                  mCurImSec = nullptr;
                  mImMast->ResetMatch();
              }
          }
       }
   }
   ELISE_fp::RmFileIfExist(aNameFileTest);
}

bool cAppli_FitsMatch1Im::HasFileCple() const
{
   return mHasFileCple;
}


bool cAppli_FitsMatch1Im::InSetCple(const std::string & aStr) const
{
   ELISE_ASSERT(mHasFileCple,"cAppli_FitsMatch1Im::InSetCple");
   return BoolFind(mSetCple,aStr);
}


bool cAppli_FitsMatch1Im::LabInInit(eTypePtRemark aLab) const
{
   return (mFlagLabsInit & (1<<int(aLab))) != 0;
}


Pt2di cAppli_FitsMatch1Im::NbMaxS0() const {return mNbMaxS0;}
bool cAppli_FitsMatch1Im::ShowDet() const { return mShowDet; }
//   int cAppli_FitsMatch1Im::NbBIndex() const { return mNbBIndex; }
//   int cAppli_FitsMatch1Im::ThreshBIndex() const { return mThreshBIndex; }
eTypePtRemark  cAppli_FitsMatch1Im::LabInit() const {return mLabInit;}
bool   cAppli_FitsMatch1Im::DoFiltrageSpatial() const {return mDoFiltrageSpatial;}

std::string cAppli_FitsMatch1Im::NameCple(const std::string & aN1,const std::string & aN2) const
{
      return  mEASF.mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH +"@" + mPostHom,aN1,aN2,true);
}

double   cAppli_FitsMatch1Im::SeuilCorrelRatio12() const
{
   return mFitsPm.SeuilGen().SeuilCorrelRatio12().Val();
}
double   cAppli_FitsMatch1Im::SeuilGradRatio12() const
{
   return mFitsPm.SeuilGen().SeuilGradRatio12().Val();
}
double   cAppli_FitsMatch1Im::SeuilDistGrad() const
{
   return mFitsPm.SeuilGen().SeuilDistGrad().Val();
}
double   cAppli_FitsMatch1Im::ExposantPdsDistGrad() const
{
   return mFitsPm.SeuilGen().ExposantPdsDistGrad().Val();
}

void cAppli_FitsMatch1Im::SetCurMapping(cElMap2D * aMap)
{
    delete mCurMap;
    mCurMap = aMap;
}

cElMap2D &  cAppli_FitsMatch1Im::CurMapping()
{
   ELISE_ASSERT(mCurMap!=0,"cAppli_FitsMatch1Im::CurMapping");
   return *mCurMap;
}



#if (0)
#endif
int CPP_FitsMatch1Im(int argc,char ** argv)
{
   cAppli_FitsMatch1Im anAppli(argc,argv);
   return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
aooter-MicMac-eLiSe-25/06/2007*/
