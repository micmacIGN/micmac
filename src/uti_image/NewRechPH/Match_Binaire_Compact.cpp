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

#include "NewRechPH.h"

class cCompileOPC;    // Un TieP avec acces rapide aux data
class cPairOPC;       // Paire de TieP , memorise le score de nombre de bits egaux
class cSetPairOPC;    // Un ensemble de paire, peut etre le Truth ou Random
class cCalcOB;        // Un pds de calc binaire + la memo des resultats dont deux vecteur de bit
class cCombCalcOB;    // Une combinaison de cCalcOB
class cAppli_NH_ApprentBinaire;    // la classe mere


class cPairOPC
{
      public :
          cPairOPC(const cOnePCarac & aP1,const cOnePCarac & aP2) ;
          bool CompAndIsEq(const cCompCBOneBit &) const;
          int         mNbNonEq;
          const cCompileOPC & P1() const;
          const cCompileOPC & P2() const;
          const double & V1() const;
          const double & V2() const;
          void AddFlag(const cCompCB & aCOB,Im1D_REAL8 aImH) const;
      private :
          cCompileOPC mP1;
          cCompileOPC mP2;
          mutable double      mV1;
          mutable double      mV2;
};



class cScoreOPC
{
    public :
          cScoreOPC(double aScore,double aHTrue,double aHRand,int aBit);
           
          double mScore;
          double mHTrue;
          double mHRand;
          int    mBit;
          std::vector<double> mHCumT;
          std::vector<double> mHCumR;
          double mInH;
};

class cSetPairOPC
{
      public :
          cSetPairOPC();
          void Compile(int aNbBitsMax);
          void Add(const TIm2DBits<1> &,int aSign);
          void Reset();
          void Finish();
          cScoreOPC Score(const cSetPairOPC &,double aPdsWrong,int aBitMax) const; // Score, Truth, Rand
         
          std::vector<cPairOPC>  mVP; 
          void AddFlag(const cCompCB & aCOB,Im1D_REAL8 aImH) const;
      private :
          int                  mPairNbBitsMax;
          int                  mNbPair;
          std::vector<int>     mHisto;
          std::vector<double>  mHistoCum;
};

class cCalcOB
{
    public :
       cCalcOB(const  cCompCBOneBit &,const cSetPairOPC & aVTruth,const cSetPairOPC & aVRand);
       const cCompCBOneBit &  COB() const;

       const TIm2DBits<1> &  TIT() const;
       const TIm2DBits<1> &  TIR() const;
       const double & Inhomog() const;
    private :
       void Init(TIm2DBits<1> & aTB,const cSetPairOPC &,int & aNbEq,double & aProp,double & aPropPlus);
       // cCompCBOneBit
       cCompCBOneBit  mCOB;
       Im2D_Bits<1>   mIT;
       TIm2DBits<1>   mTIT;
       int            mNbEqT;
       Im2D_Bits<1>   mIR;
       TIm2DBits<1>   mTIR;
       int            mNbEqR;
       double         mScoreIndiv;
       double         mPropPlus;
       double         mInhomog;
};

class cCombCalcOB
{
    public :
       void  Save(const std::string &,int aBitThreshold) const;
       cCompCB CCOB() const;
       const std::vector<cCalcOB> & VC() const;
       std::vector<cCalcOB> & VC() ;
       void Add(cCalcOB);
       cCombCalcOB(const std::vector<cCalcOB> & = std::vector<cCalcOB>() );
       cCombCalcOB Modif(std::vector<int> & ,const std::vector<cCompCBOneBit> &,const cSetPairOPC & aVTruth,const cSetPairOPC & aVRand) const;
    private :
       std::vector<cCalcOB> mVC;
};



/*********************************************/
/*                                           */
/*               ::                          */
/*                                           */
/*********************************************/


void AddTo(cSetRefPCarac &aRes,const cSetRefPCarac & ToAdd)
{
   std::copy(ToAdd.SRPC_Truth().begin(),ToAdd.SRPC_Truth().end(),std::back_inserter(aRes.SRPC_Truth()));
   std::copy(ToAdd.SRPC_Rand().begin(),ToAdd.SRPC_Rand().end(),std::back_inserter(aRes.SRPC_Rand()));
}

void MakeSetRefPCarac(cSetRefPCarac &aRes,const std::vector<std::string> & aVName)
{
   aRes = cSetRefPCarac();
   for (int aK=0 ; aK<int(aVName.size()) ; aK++)
   {
      AddTo(aRes,StdGetFromNRPH(aVName.at(aK),SetRefPCarac));
   }
}

/**************************************************/
/*                                                */
/*             cCompileOPC                        */
/*                                                */
/**************************************************/


cCompileOPC::cCompileOPC(const cOnePCarac & aOPC) :
   mOPC   (aOPC),
   mDR    (mOPC.InvR().ImRad().data())
{
}

double   cCompileOPC::ValCB(const cCompCBOneBit & aCCOB) const
{
   double aRes = 0.0;
   const int * aDX    = aCCOB.IndX().data();
   const int * aDY    = aCCOB.IndY().data();
   const double * aDC = aCCOB.Coeff().data();
   for (int aK=0 ; aK<int(aCCOB.IndX().size()) ; aK++)
   {
       aRes += mDR[aDY[aK]][aDX[aK]] * aDC[aK];
   }
   return aRes;
}

int cCompileOPC::ShortFlag(const cCompCB & aCOB,int aK0,int aK1) const
{
   int aFlag = 0;
   for (int aK=aK0 ; aK<aK1 ; aK++)
   {
      if (ValCB(aCOB.CompCBOneBit()[aK]) > 0)
         aFlag |= 1<<(aK-aK0);
   }

   return aFlag;
}

int cCompileOPC::ShortFlag(const cCompCB & aCOB) const
{
   return ShortFlag(aCOB,0,aCOB.CompCBOneBit().size());
}

tCodBin cCompileOPC::LongFlag(const cCompCB & aCOB) const
{
   int aSz = aCOB.CompCBOneBit().size();
   int aNbByte = (aSz+15) /16;

   tCodBin aRes(1,aNbByte);
   U_INT2 * aData = aRes.data()[0];
   for (int aByte =0 ; aByte <aNbByte ; aByte++)
      aData[aByte] = ShortFlag(aCOB,aByte*16,ElMin((aByte+1)*16,aSz));
   return aRes;
}




void cCompileOPC::AddFlag(const cCompCB & aCOB,Im1D_REAL8 aImH) const
{
   aImH.data()[ShortFlag(aCOB)]++;
}

/**************************************************/
/*                                                */
/*             cScoreOPC                          */
/*                                                */
/**************************************************/

cScoreOPC::cScoreOPC(double aScore,double aHTrue,double aHRand,int aBit) :
   mScore (aScore),
   mHTrue (aHTrue),
   mHRand (aHRand),
   mBit   (aBit),
   mInH   (0.0)
{
}
           
/**************************************************/
/*                                                */
/*             cPairOPC                           */
/*                                                */
/**************************************************/

cPairOPC::cPairOPC(const cOnePCarac & aP1,const cOnePCarac & aP2) :
   mNbNonEq (0),
   mP1   (aP1),
   mP2   (aP2)
{
}

bool cPairOPC::CompAndIsEq(const cCompCBOneBit & aCCOB) const
{
   mV1 = mP1.ValCB(aCCOB);
   mV2 = mP2.ValCB(aCCOB);
   return  ((mV1>0) == (mV2>0)) ;
}

const cCompileOPC & cPairOPC::P1() const { return mP1; }
const cCompileOPC & cPairOPC::P2() const { return mP2; }
const double & cPairOPC::V1() const { return mV1; }
const double & cPairOPC::V2() const { return mV2; }

void cPairOPC::AddFlag(const cCompCB & aCOB,Im1D_REAL8 aImH) const
{
    mP1.AddFlag(aCOB,aImH);
    mP2.AddFlag(aCOB,aImH);
}


/**************************************************/
/*                                                */
/*             cCalcOB                            */
/*                                                */
/**************************************************/

cCalcOB::cCalcOB(const  cCompCBOneBit & aCOB,const cSetPairOPC & aVTruth,const cSetPairOPC & aVRand) :
   mCOB    (aCOB),
   mIT     (aVTruth.mVP.size(),1,0),
   mTIT    (mIT),
   mNbEqT  (0),
   mIR     (aVRand.mVP.size(),1,0),
   mTIR    (mIR),
   mNbEqR  (0)
{
    double aPropT,aPropR,aPropPlusT,aPropPlusR;
    Init(mTIT,aVTruth,mNbEqT,aPropT,aPropPlusT);
    Init(mTIR,aVRand,mNbEqR,aPropR,aPropPlusR);
    
    mScoreIndiv = aPropT-aPropR;

    mInhomog = (ElAbs(aPropPlusT-0.5)+ElAbs(aPropPlusR-0.5))/2.0;
}
const cCompCBOneBit &  cCalcOB::COB() const 
{
   return mCOB;
}

const double & cCalcOB::Inhomog() const
{
   return mInhomog;
}



void cCalcOB::Init(TIm2DBits<1> & aTB,const cSetPairOPC & aVOP,int &aNbEq,double & aProp,double & aPropPlus)
{
    int aNbPlus = 0;
    aNbEq = 0;
    int aNbP = aVOP.mVP.size();
    for (int aK=0 ; aK<aNbP ; aK++)
    {
         bool isEq = aVOP.mVP[aK].CompAndIsEq(mCOB);
         aNbPlus += aVOP.mVP[aK].V1() >0;
         aNbPlus += aVOP.mVP[aK].V2() >0;
         aTB.oset(Pt2di(aK,0),isEq ? 1 : 0);
         if (isEq) 
            aNbEq++;
    }
    aProp = double(aNbEq) / double(aNbP);
    aPropPlus = aNbPlus / double(2*aNbP);
}

const TIm2DBits<1> &  cCalcOB::TIT() const {return mTIT;}
const TIm2DBits<1> &  cCalcOB::TIR() const {return mTIR;}

/**************************************************/
/*                                                */
/*             cCombCalcOB                        */
/*                                                */
/**************************************************/

cCombCalcOB::cCombCalcOB(const std::vector<cCalcOB> & aVC) :
  mVC (aVC)
{
}

const std::vector<cCalcOB> & cCombCalcOB::VC() const { return mVC; }
std::vector<cCalcOB> & cCombCalcOB::VC() { return mVC; }

void cCombCalcOB::Add(cCalcOB aCalc)
{
   mVC.push_back(aCalc);
}

cCompCB cCombCalcOB::CCOB() const
{
   cCompCB aCC;
   for (const auto & aCOB : mVC)
       aCC.CompCBOneBit().push_back(aCOB.COB());
   return aCC;
}

void  cCombCalcOB::Save(const std::string & aNameSave,int aBitThreshold) const
{
/*
   cCompCB aCC;
   for (const auto & aCOB : mVC)
       aCC.CompCBOneBit().push_back(aCOB.COB());
*/
   cCompCB aCCOB = CCOB();
   aCCOB.BitThresh() = aBitThreshold;

   MakeFileXML(aCCOB,aNameSave);
}

cCombCalcOB cCombCalcOB::Modif
            (
                std::vector<int> & aVI,
                const std::vector<cCompCBOneBit> & aVOB,
                const cSetPairOPC & aVTruth,
                const cSetPairOPC & aVRand
            ) const
{
    std::vector<cCalcOB> aVC = mVC;
    for (int aK=0 ; aK<int(aVI.size()) ; aK++)
    {
       aVC[aVI[aK]] = cCalcOB(aVOB[aK],aVTruth,aVRand);
    }
    return  cCombCalcOB (aVC);
}

/**************************************************/
/*                                                */
/*                cSetPairOPC                     */
/*                                                */
/**************************************************/

cSetPairOPC::cSetPairOPC() 
{
}
void cSetPairOPC::Compile(int aNbBitsMax) 
{
    mPairNbBitsMax = aNbBitsMax;
    mNbPair    = mVP.size();
    mHisto     = std::vector<int>(aNbBitsMax+1,0);
    mHistoCum  = std::vector<double>(aNbBitsMax+1,0);
}

void cSetPairOPC::Add(const TIm2DBits<1> & aTB,int aSign)
{
    for (int aK=0 ; aK<int(mVP.size()) ; aK++)
    {
       if (! aTB.get(Pt2di(aK,0)))
       {
           mVP[aK].mNbNonEq+= aSign;
       }
    }
}

void cSetPairOPC::Reset()
{
    for (auto & aP : mVP)
        aP.mNbNonEq = 0;
}

void cSetPairOPC::Finish()
{
  // Calcul de l'histo
  for (auto & aH : mHisto)
      aH = 0;
  for (auto & aP : mVP)
     mHisto.at(aP.mNbNonEq)++;
  // Cumul
  mHistoCum[0] = mHisto[0];
  for (int aK=1 ; aK<int(mHisto.size()) ; aK++)
  {
      mHistoCum[aK] = mHistoCum[aK-1] + mHisto[aK];
  }
  // Normalization
  double aSom = mHistoCum.back();
  for (int aK=0 ; aK<int(mHisto.size()) ; aK++)
  {
      mHistoCum[aK]   /= aSom;
  }
}

cScoreOPC cSetPairOPC::Score(const cSetPairOPC & aVOP,double aPdsWrong,int aBitMaxTheo) const
{
    cScoreOPC aRes(-1e10,0,0,0);
    int aBitMax = ElMin(aBitMaxTheo+1,int(mHistoCum.size()));
    for (int aK=0 ; aK<aBitMax ; aK++)
    {
       double aScore = mHistoCum[aK]-aVOP.mHistoCum[aK]*aPdsWrong;
       if (aScore > aRes.mScore)
       {
           aRes = cScoreOPC(aScore,mHistoCum[aK],aVOP.mHistoCum[aK],aK);
       }
    }
   
    aRes.mHCumT = mHistoCum;
    aRes.mHCumR = aVOP.mHistoCum;
    return aRes;
}

void cSetPairOPC::AddFlag(const cCompCB & aCOB,Im1D_REAL8 aImH) const
{
  for (auto & aP : mVP)
      aP.AddFlag(aCOB,aImH);
}

/**************************************************/
/*                                                */
/*             cAppli_NH_ApprentBinaire                       */
/*                                                */
/**************************************************/

class cAppli_NH_ApprentBinaire
{
      typedef std::pair<cCompileOPC,cCompileOPC> tLearnPOPC;
      public :
            cAppli_NH_ApprentBinaire(int argc, char ** argv);
            // cAppli_NH_ApprentBinaire(cSetRefPCarac  & aSRPC,int aNbT,int aNbRand,int aNBBitMax);
            cScoreOPC Score(const cCombCalcOB &);
            double ComputeInH(const cCombCalcOB &);
            cScoreOPC ScoreModif(std::vector<int> & ,const std::vector<cCompCBOneBit> &);
            void OptimCombin(const std::string &);
            void OptimLocal(const std::string & aIn,const std::string & aOut);
            cCalcOB COB(const  cCompCBOneBit &);
      private :
            // Si aNumInv<0 => Random
            // aModeRand 0 => rand X, 1=> rand Y , 2 => all
            cCompCBOneBit RandomCOB_OneInv(int aModeRand,int aNumInv,int aNbCoef);
            // aProp => Proportion de la perturbation / aux coeffs
            cCompCBOneBit RandomCOB_PerturbCoeff(const cCompCBOneBit &,double aProp);
            cScoreOPC FinishScore();
            int NbOfMode(int aMode) const;

            void Subst(cSetPairOPC &,int ,const cCompCBOneBit& );

            void DoOne(eTypePtRemark aType);
            std::vector<cCalcOB> SelectGermes();

            cInterfChantierNameManipulateur * mICNM;
            std::string   mDir;
            std::string   mDirPC;
            std::string   mPatType;
            cElRegex *    mAutomType;
            // cSetRefPCarac mCurSet;

            cSetPairOPC     mVTruth;
            cSetPairOPC     mVRand;
            int             mNbInvR;
            int             mNbQuantIR;

            int             mNbBitsMax;
            int             mNbT;
            int             mNbRand;
            double          mPdsWrong;
            int             mBitMax;
            int             mNumStep;
            std::string     mPrefName;
            std::string     mNameCombine;
            std::string     mNameInput;
            std::string     mNameLocale;
            cCombCalcOB     mLastCompute;
            bool            mDoSave;
            std::string     mExt;
            double          mSomInH;
            double          mPdhInH;
};

// cAppli_NH_ApprentBinaire::cAppli_NH_ApprentBinaire(cSetRefPCarac  & aSRPC,int aNbT,int aNbRand,int aNBBitMax) :
cAppli_NH_ApprentBinaire::cAppli_NH_ApprentBinaire(int argc, char ** argv) :
    mDir        ("./"),
    mDirPC      ("./PC-Ref-NH/"),
    mNbBitsMax  (16),
    mNbT        (400000),
    mNbRand     (400000),
    mPdsWrong   (1.0),
    mBitMax     (1000000),
    mNumStep    (1),
    mDoSave     (true),
    mExt        (""),
    mPdhInH     (0)
{
   Pt2di aNbTR;
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatType, "Patten of type in eTypePtRemark")
                    ,
         LArgMain()  << EAM(mPdsWrong,"PdsW",true,"Weighting for wrong vs true, def=1.0")
                     << EAM(mBitMax,"BitM",true,"Max bit avalaible")
                     << EAM(mNbBitsMax,"NBB",true,"Number of bits")
                     << EAM(mNumStep,"Step",true,"0:  combine, 1: combine->local , 2 local->local")
                     << EAM(aNbTR,"NbTR",true,"Number True-Random (in k 1->1000)")
                     << EAM(mNameInput,"Input",true,"Name input if != def")
                     << EAM(mExt,"Ext",true,"Extension to add to name")
                     << EAM(mPdhInH,"PdsInH",true,"Pds for In hoomogeneite")
   );
   if (EAMIsInit(&aNbTR))
   {
       mNbT =  aNbTR.x*1000;
       mNbRand =  aNbTR.y*1000;
   }

   mPrefName =      "PC-Ref-NH/Save_CompCB" 
                  + (EAMIsInit(&mBitMax) ? std::string("_BM"+ToString(mBitMax)) : "")
                  + ( "_NBB"+ToString(mNbBitsMax)) 
                  + ((mExt!="") ? std::string("_"+mExt) : "")
                  + ((mPdhInH==0.0) ? "" : std::string("_PdsInH" +ToString(round_ni(1000*mPdhInH))))
                  + ("_PdsW" +ToString(round_ni(1000*mPdsWrong)))
               ;


   mPatType = ".*(" + mPatType + ").*";
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mAutomType = new cElRegex(mPatType,10);

   for (int aKL=0 ; aKL<int(eTPR_NoLabel) ; aKL++)
   {
       DoOne(eTypePtRemark(aKL));
   }
}

void cAppli_NH_ApprentBinaire::DoOne(eTypePtRemark aType)
{
   std::string aNameType = eToString(aType);
   mNameCombine = mPrefName + "_" + aNameType +"_Comb.xml";
   mNameLocale = mPrefName  + "_" + aNameType +"_Local.xml";
   if (!mAutomType->Match(aNameType))
      return;

   const std::vector<std::string> * aSetName =   mICNM->Get("NKS-Set-NHPtRef@"+ aNameType);

   // Proportion of random pair in a given imahe
   double aPropRandIsoI = 0.5;
   int aNbSet = aSetName->size();

   int aNbTotTruth = 0;
   int aNbTotRand = 0;
   for (int aKS=0 ; aKS<aNbSet ; aKS++)
   {
       cSetRefPCarac aSetRP = StdGetFromNRPH(aSetName->at(aKS),SetRefPCarac);
       aNbTotTruth += aSetRP.SRPC_Truth().size();
       aNbTotRand  += aSetRP.SRPC_Rand().size();
   }
   mNbT = ElMin(aNbTotTruth,mNbT);
   mNbRand = ElMin(aNbTotRand,mNbRand);
   double aPropTruh = mNbT/double(aNbTotTruth);
   double aPropRand = mNbRand/double(aNbTotRand);


   for (int aKS=0 ; aKS<aNbSet ; aKS++)
   {
       cSetRefPCarac aSetRP = StdGetFromNRPH(aSetName->at(aKS),SetRefPCarac);

       // Truth pair
       int aNbTLoc = round_up( aSetRP.SRPC_Truth().size()*aPropTruh);
       cRandNParmiQ aRand(aNbTLoc,aSetRP.SRPC_Truth().size());
       for (auto & aPT : aSetRP.SRPC_Truth())
       {
           if (aRand.GetNext())
           {
              mVTruth.mVP.push_back(cPairOPC(aPT.P1(),aPT.P2()));
           }
       }

       // Random pair inside same set
       int aNbRLoc =  round_up(aSetRP.SRPC_Rand().size() * aPropRandIsoI *aPropRand);
       const std::vector<cOnePCarac> & aVR = aSetRP.SRPC_Rand();
       int aNR = aVR.size();
       for (int aKp=0 ; aKp<aNbRLoc ; aKp++)
       {
           int aK1 = NRrandom3(aNR);
           int aK2 = NRrandom3(aNR);
           mVRand.mVP.push_back(cPairOPC(aVR.at(aK1),aVR.at(aK2)));
       }
   }

   int aNbRLoc =  round_up(mNbRand*(1-aPropRandIsoI));
   for (int aK=0 ; aK<aNbRLoc ; aK++)
   {
      int aK1 = NRrandom3(mVTruth.mVP.size());
      int aK2 = NRrandom3(mVTruth.mVP.size());
      mVRand.mVP.push_back(cPairOPC(mVTruth.mVP.at(aK1).P1().mOPC,mVTruth.mVP.at(aK2).P2().mOPC));
   }

    mVTruth.Compile(mNbBitsMax);
    mVRand.Compile(mNbBitsMax);

    const cOnePCarac & anOPC = mVRand.mVP.at(0).P1().mOPC;
    mNbQuantIR = anOPC.InvR().ImRad().tx();
    mNbInvR    = anOPC.InvR().ImRad().ty();


    if (mNumStep==0)
       OptimCombin(mNameCombine);
    else if (mNumStep==1)
       OptimLocal(mNameCombine,mNameLocale);
    else if (mNumStep==2)
    {
       OptimLocal(mNameLocale,mNameLocale);
    }
}

void SetMoyNulle(cCompCBOneBit & aCOB)
{
   double aMoy = 0.0;
   for (auto & aV  : aCOB.Coeff())
   {
       aMoy  += aV;
   }
   aMoy /= aCOB.Coeff().size();
   
}

cCompCBOneBit  cAppli_NH_ApprentBinaire::RandomCOB_PerturbCoeff(const cCompCBOneBit & aInput,double aProp)
{
   cCompCBOneBit aRes = aInput;

   double aSomA=0;
   for (const auto & aV  : aRes.Coeff())
   {
       aSomA += ElAbs(aV);
   }
   aSomA *=  aProp / aRes.Coeff().size();

   for (auto & aV  : aRes.Coeff())
   {
      aV += aSomA * NRrandC();
   }
   
   SetMoyNulle(aRes);

   return aRes;

}

cCompCBOneBit cAppli_NH_ApprentBinaire::RandomCOB_OneInv(int aModeRand,int aNumInv,int aNbCoef)
{
    cCompCBOneBit aRes;
    aRes.IndBit() = -1;
    std::vector<int>  aPermQ = RandPermut(mNbQuantIR);
    std::vector<int>  aPermI = RandPermut(mNbInvR);

    double aSom = 0;
    for (int aK=0 ; aK<aNbCoef ; aK++)
    {
        double aVal = (aK==(aNbCoef-1)) ? -aSom : NRrandC();
        aRes.Coeff().push_back(aVal);
            // aModeRand 0 => rand X, 1=> rand Y , 2 => all
        aRes.IndX().push_back((aModeRand==1) ? aNumInv : aPermQ[aK]);
        aRes.IndY().push_back((aModeRand==0) ? aNumInv : aPermI[aK]);
        aSom += aVal;
    }

    return aRes;
}

cScoreOPC cAppli_NH_ApprentBinaire::FinishScore()
{
   mVTruth.Finish();
   mVRand.Finish();
   
   return mVTruth.Score(mVRand,mPdsWrong,mBitMax);
}


double cAppli_NH_ApprentBinaire::ComputeInH(const cCombCalcOB & aComb)
{
   if (aComb.VC().size()==1)
   {
      return mSomInH ;
   }
   int aNBB =  aComb.VC().size();
   int aSzH = 1<<aNBB;
   Im1D_REAL8 aH(aSzH,0.0);
   cCompCB aCOB =  aComb.CCOB() ;
   mVTruth.AddFlag(aCOB,aH);
   mVRand.AddFlag(aCOB,aH);

   FilterHistoFlag(aH,2,0.5,false);

   double aS0,aS1,aS2;

   ELISE_COPY
   (
        aH.all_pts(),
        Virgule(1.0,aH.in(),Square(aH.in())),
        Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
   );

   aS1 /= aS0;
   aS2 /= aS0;

   double aRes = aS2 / ElSquare(aS1);

   // std::cout << "INHOM= " << aRes << "\n";

   return  aRes;
}

cScoreOPC cAppli_NH_ApprentBinaire::Score(const cCombCalcOB & aComb)
{
   mVTruth.Reset();
   mVRand.Reset();
   mSomInH=0;
   for (const  auto & aCalc : aComb.VC())
   {
      mVTruth.Add(aCalc.TIT(),1);
      mVRand.Add(aCalc.TIR(),1);
      mSomInH += aCalc.Inhomog();
   }
   mSomInH /= aComb.VC().size();
   mLastCompute = aComb;
   cScoreOPC aRes = FinishScore();

   if (mPdhInH !=0)
   {
      aRes.mInH = ComputeInH(aComb);
      aRes.mScore -= aRes.mInH * mPdhInH ;
   }
   return aRes;
}


        // Subst(mVTruth,int ,const cCalcOB> )
cScoreOPC cAppli_NH_ApprentBinaire::ScoreModif(std::vector<int> & aVI,const std::vector<cCompCBOneBit> & aVC)
{
   for (int aK=0 ; aK<int(aVI.size()) ; aK++)
   {
       cCalcOB & aCOB = mLastCompute.VC()[aVI[aK]];
       mVTruth.Add(aCOB.TIT(),-1);
       mVRand.Add(aCOB.TIR(),-1);

       aCOB = cCalcOB(aVC[aK], mVTruth,mVRand);
       // mLastCompute
       mVTruth.Add(aCOB.TIT(),1);
       mVRand.Add(aCOB.TIR(),1);
   }
   return FinishScore();
}
    
int cAppli_NH_ApprentBinaire::NbOfMode(int aMode) const
{
  if (aMode==0) 
     return mNbInvR;
  if (aMode==1) 
     return mNbQuantIR;
  return  mNbQuantIR+mNbInvR;
}


cCalcOB cAppli_NH_ApprentBinaire::COB(const  cCompCBOneBit & aCOB) {return cCalcOB(aCOB,mVTruth,mVRand);}

// class cCombCalcOB
std::vector<cCalcOB> cAppli_NH_ApprentBinaire::SelectGermes()
{
    int aNbCoef= 3;
    int aNbInOneLine= (mPdhInH !=0)  ? 1 : 3;
    int aNbTests = (mPdhInH !=0)  ? 10 : 100;

    std::vector<cCalcOB> aVGlobC;
    for (int aMode=0 ; aMode<3 ; aMode++)
    {
       int aNbIR = NbOfMode(aMode);
      
       for (int aKIR=0 ; aKIR<aNbIR ; aKIR++)
       {
           cCombCalcOB aVC;
           for (int aKNb=0 ; aKNb<aNbInOneLine; aKNb++)
           {
               cScoreOPC aScMax(-1e10,0,0,0);
               cCalcOB aCOBMax =  COB(RandomCOB_OneInv(aMode,aKIR,aNbCoef));
               for (int aNbT=0 ; aNbT<100 ; aNbT++)
               {
                   cCalcOB aCOB =  COB(RandomCOB_OneInv(aMode,aKIR,aNbCoef));
                   cCombCalcOB aVCur = aVC;
                   aVCur.Add(aCOB);
                   cScoreOPC aS = Score(aVCur);
                   if (aS.mScore> aScMax.mScore)
                   {
                      aCOBMax = aCOB;
                      aScMax = aS;
                   }
               }

               for (int aKT=0 ; aKT<aNbTests ; aKT++)
               {
                   cCalcOB aCOB =  COB(RandomCOB_PerturbCoeff(aCOBMax.COB(),1/(1+aKT*0.3)));
                   cCombCalcOB aVCur = aVC;
                   aVCur.Add(aCOB);
                   cScoreOPC aS = Score(aVCur);
                   if (aS.mScore> aScMax.mScore)
                   {
                      aCOBMax = aCOB;
                      aScMax = aS;
                   }
               }

             
               aVC.Add(aCOBMax);
               aVGlobC.push_back(aCOBMax);
               std::cout << "SSSSSSS glob : " << aScMax.mHTrue 
                         << " "<< aScMax.mHRand << " " << aScMax.mBit 
                         << " InH " << aScMax.mInH 
                         << " Mode=" << aMode << "  KIReste=" << aNbIR - aKIR
                         << "\n";
           }
           std::cout << "=============================\n";
       }
    }
    return aVGlobC;
}

void cAppli_NH_ApprentBinaire::OptimLocal(const std::string & aIn,const std::string & aOut)
{
   std::string aInput = EAMIsInit(&mNameInput) ? mNameInput : aIn;
   cCompCB aCOB = StdGetFromNRPH(aInput,CompCB);
   cCombCalcOB aCMax;
   for (const auto & aOCB : aCOB.CompCBOneBit())
   {
       cCalcOB aCalc(aOCB,mVTruth,mVRand);
       aCMax.Add(aCalc);
   }

   cScoreOPC aSMax = Score(aCMax);
   for (int aK=0 ; aK< (int)aSMax.mHCumT.size() ; aK++)
   {
       std::cout << aK << "T: " << aSMax.mHCumT[aK] << " R: " << aSMax.mHCumR[aK] << "\n";
   }
   std::cout << "SCORE INIT " << aSMax.mScore << " PRESS ENTER\n";
   getchar();
   int aCpt=0;
   while (1)
   {
// std::cout << "WWWWWWWWWWWWWWWW\n";
        int aNbModif = round_down(1 + 2 * pow(NRrandom3(),4));
        aNbModif = ElMax(1,ElMin(mNbBitsMax,aNbModif));
        aNbModif = 1;

// std::cout << "NNN " << aNbModif << "\n";

 ; // NRrandom3(mNbBitsMax)
        std::vector<int>  aRP = RandPermut(mNbBitsMax);

        std::vector<int> aVI;
        std::vector<cCompCBOneBit>  aVCOB;
        for (int aK=0 ; aK<aNbModif ; aK++)
        {
            int aIndK= aRP[aK];
            aVI.push_back(aIndK);
            if (NRrandom3() > 0.25)
            {
                cCompCBOneBit aCOB =  RandomCOB_PerturbCoeff(aCMax.VC()[aIndK].COB(),pow(NRrandom3(),4));
                aVCOB.push_back(aCOB);
            }
            else
            {
                int aMode = ElMin(3,round_down(3*pow(NRrandom3(),2)));
                int MaxInd = NbOfMode(aMode) ;
                aVCOB.push_back(RandomCOB_OneInv(aMode,NRrandom3(MaxInd),3));
            }
        }
        if (1) // Voie lente mais "safe" ?
        {
             cCombCalcOB aNew = aCMax.Modif(aVI,aVCOB,mVTruth,mVRand);
             cScoreOPC aSNew = Score(aNew);
             if (aSNew.mScore > aSMax.mScore)
             {
                aSMax = aSNew;
                aCMax = aNew;
                std::cout << " Cpt=" << aCpt
                     << " Score : " << aSMax.mScore 
                     << " True : " << aSMax.mHTrue 
                     << " Rand : "<< aSMax.mHRand 
                     << " MB " << aSMax.mBit  
                     << " InH " << aSMax.mInH  
                     << " NBBB " << mNbBitsMax << "\n";
                if (mDoSave)
                   aCMax.Save(aOut,aSMax.mBit);
             }
        }
/*
        // cScoreOPC aS1 = ScoreModif(aVI,aVCOB);
        if (aS1.mScore > aSMax)
        {
        }


        cScoreOPC aS0 = Score(aCMax);

        // cScoreOPC aS2 = Score(mLastCompute);

        if (aS1.mScore > aS0.mScore)
        {
           mLastCompute
        std::cout << aS0.mScore << S1.mScore << " " << aS2.mScore << "\n";
           getchar();
        }
*/
      aCpt++;
   }
   
}

void cAppli_NH_ApprentBinaire::OptimCombin(const std::string & aNameSave)
{
    std::vector<cCalcOB>  aVGlobC = SelectGermes();

    cScoreOPC aScMax(-1e10,0,0,0);

    int aCpt=0;
    while (1)
    {
        std::vector<int>  aRP = RandPermut(aVGlobC.size());
        cCombCalcOB aVCur ;
        for (int aK=0 ; aK<mNbBitsMax ; aK++)
        {
            aVCur.Add(aVGlobC[aRP[aK]]);
        }
        cScoreOPC aS = Score(aVCur);
        if (aS.mScore>aScMax.mScore)
        {
           aScMax = aS;
           std::cout << " Cpt=" << aCpt
                     << " Score : " << aScMax.mScore 
                     << " True : " << aScMax.mHTrue 
                     << " Rand : "<< aScMax.mHRand 
                     << " InH " << aScMax.mInH  
                     << " MB " << aScMax.mBit  
                     << " NBBB " << mNbBitsMax << "\n";
           if (mDoSave)
              aVCur.Save(aNameSave,aScMax.mBit);
        }
        aCpt++;
    }

/*
    for (int aK1=0 ; aK1<int(aVGlobC.size()) ; aK1++)
    {
        for (int aK2=aK1 ; aK2<int(aVGlobC.size()) ; aK2++)
        {
            std::vector<cCalcOB> aVCur ;
            aVCur.push_back(aVGlobC[aK1]);
            aVCur.push_back(aVGlobC[aK2]);
            double aS = Score(aVCur);
            if (aS>aCostMax)
            {
               aCostMax = aS;
               std::cout << "CostMax " << aCostMax << "\n";
            }
        }
    }
*/
}

int  CPP_PHom_ApprentBinaire(int argc,char ** argv)
{
   cAppli_NH_ApprentBinaire anAppli(argc,argv);
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
