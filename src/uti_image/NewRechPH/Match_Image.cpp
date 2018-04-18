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



void FiltrageDirectionnel(std::vector<cCdtCplHom> & aVCpl)
{
   if (aVCpl.empty()) 
      return;
   int aNbDir = aVCpl[0].mPM->mSzIm.y;

   double aPropConv = 0.1;
   double  aPropDir = 0.07;

   int   aMul = 100;
       

   int aSeuilDir = ElMax(1,round_ni(aPropDir*aNbDir));
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

   std::vector<cCdtCplHom> aNewV;
   for (const auto  & aCpl : aVCpl)
   {
       int aDif = mod(aCpl.mShift-aKMax,aNbDir);
       aDif = ElMin(aDif,aNbDir-aDif);
       if (aDif < aSeuilDir)
          aNewV.push_back(aCpl);

   }
   aVCpl = aNewV;
}


/*************************************************/
/*                                               */
/*             ::                                */
/*                                               */
/*************************************************/

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
    InitOneLabelFitsPm(aFOL.BinDecision(),aDir,aLab);
}

const  std::string TheDirXmlFits=    string("include")    + ELISE_CAR_DIR
                                      + string("XML_MicMac") + ELISE_CAR_DIR 
                                      + string("Fits")       + ELISE_CAR_DIR;
const  std::string DefNameFitParam =  "FitsParam.xml";

void InitFitsPm(cFitsParam & aFP,const std::string & aDir, const std::string & aName)
{
    aFP = StdGetFromNRPH(aDir+aName,FitsParam);
    InitOneFitsPm(aFP.OverLap(),aDir);
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

cIndexCodeBinaire::cIndexCodeBinaire(const cCompCB & aCCB,bool Overlap) :
   mNBBTot    (aCCB.CompCBOneBit().size()),
   mNBBVois   (aCCB.BitThresh()),
   mFlagV     (FlagOfNbb(mNBBTot,mNBBVois)),
   mVTabIndex (1<<mNBBTot),
   mOverlap   (Overlap)
{
}

void cIndexCodeBinaire::Add(cCompileOPC & aPC)
{
   int aFlag =  mOverlap ? aPC.mOL_ShortFlag : aPC.mDec_ShortFlag;
   for (const auto & aFlagV : *mFlagV)
   {
       mVTabIndex.at(aFlag^aFlagV).push_back(&aPC);
   }
}

const std::vector<cCompileOPC *> & cIndexCodeBinaire::VectVois(const cCompileOPC & aPC)
{
     return mVTabIndex.at(mOverlap ? aPC.mOL_ShortFlag :  aPC.mDec_ShortFlag);
}

void cIndexCodeBinaire::Add(cSetOPC & aSet,const cFitsOneLabel & aFOL)
{
std::cout << "cIndexCodeBinaire::Add " << aSet.VOpc().size() << "\n";
    for (auto & aCel : mVTabIndex)
       aCel.clear();

    for (auto & aPC : aSet.VOpc())
    {
        aPC.SetFlag(aFOL,mOverlap);
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

void cSetOPC::FiltrageFromHighestScale(const cSetOPC & aBigSet,int aNb,bool aShow)
{
    std::vector<double> aVStab;
    for (const auto & aPC : aBigSet.mVOpc)
    {
        aVStab.push_back(aPC.mOPC.ScaleStab());
    }

    // On selectionne ceux qui sont au dessus de l'echelle limite
    double aProp = 1-aNb/(double) aBigSet.mVOpc.size();
    double aScaleLim = KthValProp(aVStab,aProp);
    if (aShow)
         std::cout << "SCALE LIMITE ========== " << aScaleLim  << " " << aProp << "\n"; // getchar();
    mVOpc.clear();
    for (const auto & aPC : aBigSet.mVOpc)
    {
       if (aPC.mOPC.ScaleStab() >= aScaleLim)
           mVOpc.push_back(aPC);
    }
   
}
void cSetOPC::InitLabel(const cFitsOneLabel & aFOL,const cSeuilFitsParam & aSeuil,bool DoIndex,bool Overlap) 
{
std::cout << " cSetOPC::InitLabe " << DoIndex << " " << Overlap << "\n";
    if (DoIndex)
    {
       mIndexCB = new cIndexCodeBinaire(aFOL.BinIndexed().CCB().Val(),Overlap);
       mIndexCB->Add(*this,aFOL);
    }
    mFOL = & aFOL;
    mSeuil = & aSeuil;
}

void  cSetOPC::Add(const cCompileOPC& anOPC)
{
   mVOpc.push_back(anOPC);
}

const std::vector<cCompileOPC> &  cSetOPC::VOpc() const { return mVOpc; }
std::vector<cCompileOPC> &  cSetOPC::VOpc() { return mVOpc; }


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

cCompileOPC& cSetOPC::At(int aK) {return mVOpc.at(aK);}



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
   mSzIm   (mMTD.SzImTifOrXif()),
   mVSetCC (int(eTIR_NoLabel))
{
    std::string aNamePC = NameFileNewPCarac(mNameIm,true,anAppli.ExtNewH());

    mSetPC = StdGetFromNRPH(aNamePC,SetPCarac);
    
    for (const auto & aPC : mSetPC.OnePCarac())
    {
        // mVSetCC.at(int(aPC.Kind())).mVOpc.push_back(cCompileOPC(aPC));
        mVSetCC.at(int(aPC.Kind())).Add(cCompileOPC(aPC));
    }

    mSetInd0.FiltrageFromHighestScale(mVSetCC.at(mAppli.LabOL()),anAppli.NbMaxS0().x,anAppli.ShowDet());
}



cAFM_Im::~cAFM_Im()
{
//   static cSetPCarac TheSetPC;
//   mSetPC = TheSetPC;
}

void cAFM_Im::SetFlagVSetCC(bool DoIndex)
{
    for (const auto &  aFOL : mAppli.FitsPm().GenLabs() )
    {
         mVSetCC.at(int(aFOL.KindOf())).InitLabel(aFOL,mAppli.FitsPm().SeuilGen(),DoIndex,false);
    }
}


/*************************************************/
/*                                               */
/*           cAFM_Im_Master                      */
/*                                               */
/*************************************************/

cAFM_Im_Master::cAFM_Im_Master(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im     (aName,anApli),
    mQt         (mArgQt,Box2dr(Pt2dr(-10,-10),Pt2dr(10,10)+Pt2dr(mSzIm)),5,euclid(mSzIm)/20.0)
{
    mSetInd0.InitLabel(mAppli.FitsPm().OverLap(),mAppli.FitsPm().SeuilOL(),true,true);

    SetFlagVSetCC(true);
/*
    mIndexCB.Add(mSetInd0,mAppli.FitsPm().OverLap());
    for (auto & aPC : mSetInd0.mVOpc)
    {
        aPC.SetFlag(mAppli.FitsPm().OverLap());
        mIndexCB.Add(aPC);
   }
*/
}


void cAFM_Im_Master::FilterVoisCplCt(std::vector<cCdtCplHom> & aV0)
{
    mQt.clear(); 
    ElPackHomologue aPack = PackFromVCC(aV0); 

    ElSimilitude  aSim = SimilRobustInit(aPack,0.666);

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

    double aDiag = euclid(mSzIm);
    double aSeuilOk = aDiag / 100.0;
    double aSeuilPb = aDiag / 5.0;
    double aSeuilCoh = 0.3;

    int aNbVois = 10;

    double aSurfPP = (double(mSzIm.x) * mSzIm.y) / aV0.size() ; // Surf per point
    double aDistSeuilNbV = sqrt(aSurfPP*aNbVois) * 2;

    
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


    std::vector<cCdtCplHom> aRes;
    for (auto  & aCpl : aV0)
    {
        if (aCpl.mOk)
           aRes.push_back(aCpl);
    }
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
   FiltrageDirectionnel(aVCpl);
   if (int(aVCpl.size()) <=  aNbMin)
      return ;

   FilterVoisCplCt(aVCpl);
   if (int(aVCpl.size()) <=  aNbMin)
      return ;

}


void cAFM_Im_Master::MatchOne(bool OverLap,cAFM_Im_Sec & anISec, cSetOPC & aSetM,cSetOPC & aSetS,std::vector<cCdtCplHom> & aOld,int aNbMin)
{
   std::vector<cCdtCplHom> aVCpl;
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
          if(aPC.mOPC.ScaleStab()>0)
             aFH.Add(IScal(aPC.mOPC.ScaleStab()),1,__LINE__);
      }
      std::cout << "======= HISTO SCALE BEFORE ===========\n";
      aFH.Show();
   }

   cFHistoInt aHLF;
   int First= true;
   for (int aKs=0 ; aKs<(int)aSetS.VOpc().size() ; aKs++)
   {
      cCompileOPC & aPCS = aSetS.At(aKs);
      aPCS.SetFlag(aSetM.FOL(),OverLap);
      // int aFlagS = aPCS.mShortFlag;
      // std::vector<cCompileOPC *> & aVSel = mVTabIndex.at(aFlagS);

      const std::vector<cCompileOPC *> & aVSel = aSetM.Ind().VectVois(aPCS);
      
      for (int aKSel=0 ; aKSel<(int)aVSel.size() ; aKSel++)
      {
          int aLevFail;
          int aShift;
          cCompileOPC * aPCM = aVSel[aKSel];
          double aD =  aPCM->Match(OverLap,aPCS,aSetM.FOL(),aSetM.Seuil(),aShift,aLevFail,aPtrTM);
          aHLF.Add(aLevFail,1,__LINE__);
          if (aD > 0)
          {
             aPCS.mTmpNbHom++;
             aPCM->mTmpNbHom++;

             aVCpl.push_back(cCdtCplHom(aPCM,&aPCS,aD,aShift));
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
      std::cout << "NbCouple " << aSetS.VOpc().size() * aSetM.VOpc().size() << "\n";
      std::cout << "======= TIME EFFECTIF  ===========\n";
      aPtrTM->Show();
   }

   {
       for (const auto  & aCpl : aVCpl)
       {
          if ((aCpl.mPM->mTmpNbHom==1) && (aCpl.mPS->mTmpNbHom==1))
             aOld.push_back(aCpl);
       }
       for (const auto  & aCpl : aVCpl)
       {
          aCpl.mPM->mTmpNbHom=0; 
          aCpl.mPS->mTmpNbHom=0; 
       }
   }

/*
   int aNbBeforeDir = aVCpl.size();

   FiltrageSpatialGlob(aVCpl,aNbMin);
   if (int(aVCpl.size()) <=  aNbMin)
      return ;

   if (mAppli.ShowDet())
   {
      cFHistoInt aFHDif;
      cFHistoInt aFHScale;
      for (const auto & aCpl : aVCpl)
      {
          aFHDif.Add(ElAbs(IScal(aCpl.mPM->mOPC.ScaleStab()) -  IScal(aCpl.mPS->mOPC.ScaleStab())));
          aFHScale.Add(IScal(aCpl.mPS->mOPC.ScaleStab()));
      }
      std::cout << "======= HISTO SCALE AFTER  ===========\n";
      aFHScale.Show();
      std::cout << "======= HISTO DIFF SCALE  ===========\n";
      aFHDif.Show();
   }

   std::cout <<"(Im " << mNameIm << " " << anISec.mNameIm << ") "
             << "(NbSel " << aVCpl.size()  << " from " << aSetS.mVOpc.size() << ")"
             << " PropF " << aVCpl.size() /double(aSetS.mVOpc.size())  
             << " PropD " << aVCpl.size() /double(aNbBeforeDir)  
             << "\n";

*/

   return ;
}

bool  cAFM_Im_Master::MatchGlob(cAFM_Im_Sec & anISec)
{
    int aNbBeforeDir=0;
    int aNbMin0 = 6;
    std::vector<cCdtCplHom> aVCpl;

    // Premier calcul sur nb de points reduit
    {
       MatchOne(true,anISec,mSetInd0, anISec.mSetInd0,aVCpl,aNbMin0);
       if (int(aVCpl.size()) <= aNbMin0) 
          return false;

       aNbBeforeDir = aVCpl.size();

       FiltrageSpatialGlob(aVCpl,aNbMin0);
       if (int(aVCpl.size()) <=  aNbMin0)
          return false ;
    }

    if (mAppli.ShowDet())
    {
       cFHistoInt aFHDif;
       cFHistoInt aFHScale;
       for (const auto & aCpl : aVCpl)
       {
          aFHDif.Add(ElAbs(IScal(aCpl.mPM->mOPC.ScaleStab()) -  IScal(aCpl.mPS->mOPC.ScaleStab())),1,__LINE__);
          aFHScale.Add(IScal(aCpl.mPS->mOPC.ScaleStab()),1,__LINE__);
       }
       std::cout << "======= HISTO SCALE AFTER  ===========\n";
       aFHScale.Show();
       std::cout << "======= HISTO DIFF SCALE  ===========\n";
       aFHDif.Show();
    }

    int aNbInit =  anISec.mSetInd0.VOpc().size();
    std::cout <<"(Im " << mNameIm << " " << anISec.mNameIm << ") "
              << "(NbSel " << aVCpl.size()  << " from " << aNbInit << ")"
              << " PropF " << aVCpl.size() /double(aNbInit)  
              << " PropD " << aVCpl.size() /double(aNbBeforeDir)  
              << "\n";

    anISec.SetFlagVSetCC(false);
    aVCpl.clear();
    for (auto & aFOL : mAppli.FitsPm().GenLabs())
    {
       int aKL = int(aFOL.KindOf());
       MatchOne(false,anISec,mVSetCC.at(aKL), anISec.mVSetCC.at(aKL),aVCpl,aNbMin0);

       std::cout << "HHHHHHHHHHHHhh " << mVSetCC.at(aKL).VOpc().size() << " => " << aVCpl.size() << "\n";
    }
    FiltrageSpatialGlob(aVCpl,aNbMin0);

    // :std::vector<cCdtCplHom> aVCpl.clear();

    ElPackHomologue aPack = PackFromVCC(aVCpl);
    aPack.StdPutInFile(mAppli.NameCple(mNameIm,anISec.mNameIm));
    return true ;
}

/*************************************************/
/*                                               */
/*           cAFM_Im_Sec                         */
/*                                               */
/*************************************************/

cAFM_Im_Sec::cAFM_Im_Sec(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im(aName,anApli)
{
    // mSetInd0.InitLabel(mAppli.FitsPm().OverLap(),false,true);
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
   mExtNewH     (""),
   mSH          (""),
   mPostHom     ("dat"),
   mExpTxt      (false),
   mOneWay      (true),
   mShowDet     (false),
   mCallBack    (false),
   mNbMaxS0     (1000,200)
{

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
   );


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

   if (!EAMIsInit(&mSH) )
   {
        mSH = "HFits"+ mExtNewH;
   }
   
   InitFitsPm(mFitsPm,MMDir()+TheDirXmlFits,mNameXmlFits);
   mLabOL = mFitsPm.OverLap().KindOf();
    // const cFitsParam & aFitsPM = anAppli.FitsPm();
   const cFitsOneLabel & aFOL = mFitsPm.OverLap();
   mNbBIndex =  aFOL.BinIndexed().CCB().Val().CompCBOneBit().size();
   mThreshBIndex =  aFOL.BinIndexed().CCB().Val().BitThresh();

   mEASF.Init(mPatIm);
   if (! EAMIsInit(&mShowDet))
      mShowDet = (!mCallBack) && (mEASF.SetIm()->size()==1);


   mImMast = new cAFM_Im_Master(mNameMaster,*this);


   int aNbFailConseq=0;
   for (const auto &  aName : *(mEASF.SetIm()))
   {
       if (aName != mNameMaster)
       {
           if ((! mOneWay) || (aName > mNameMaster))
           {
              mCurImSec = new  cAFM_Im_Sec(aName,*this);

               if ( mImMast->MatchGlob(*mCurImSec))
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
                  
               // aPack.StdPutInFile(mAppli.NameCple(mNameIm,anISec.mNameIm));
              // cSetOPC & aSetM = mSetInd0;
              // cSetOPC & aSetS = anISec.mSetInd0;

              delete mCurImSec;
              mCurImSec = nullptr;
          }
       }
   }
}

Pt2di cAppli_FitsMatch1Im::NbMaxS0() const {return mNbMaxS0;}
bool cAppli_FitsMatch1Im::ShowDet() const { return mShowDet; }
int cAppli_FitsMatch1Im::NbBIndex() const { return mNbBIndex; }
int cAppli_FitsMatch1Im::ThreshBIndex() const { return mThreshBIndex; }
eTypePtRemark  cAppli_FitsMatch1Im::LabOL() const {return mLabOL;}

std::string cAppli_FitsMatch1Im::NameCple(const std::string & aN1,const std::string & aN2) const
{
   return  mEASF.mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH +"@" + mPostHom,aN1,aN2,true);
}

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
