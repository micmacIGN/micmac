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
#include "Apero.h"


// bool DebugVisb=false;

//  Lorsque l'on veut ponderer des observation ponctuelle dans le plan, 
//  si elles tombe toute au meme endroit, chacune doit avoir un poid proportionnel
// a l'inverse du nombre d'observation;
//  Ici cela est gere avec une certaine incertitude, les observation etant suposee
// etre localisees selon une gaussienne


/*
*/

class cCombinPosCam;  // Un sous ensemble d'image secondaire
class cSetCombPosCam;  // Contient toute les combinaison avec des moyens d'y acceder (via flag ou autre)
class cCaseOcupIm;
class cPCICentr;
class cPoseCdtImSec ;

/***************************************************/
/*                                                 */
/*              ::                                 */
/*                                                 */
/***************************************************/

extern std::string ExtractDigit(const std::string & aName,const std::string &  aDef);


bool ShowACISec = false;
#define TetaMaxOccup 1.5
typedef REAL4 tImOccup;
typedef REAL8 tBaseImOccup;

int NbOccup = 50;

// A Angle  , O Optimum
//  La formule est faite pour que
//   *  on ait un gain proportionnel a l'angle en 0
//   *  un maximum pour Opt avec une derivee nulle
//   *


double GainAngle(double A,double Opt)
{
   A /= Opt;
   if (A <= 1)
      return pow(ElAbs(2*A - A*A),1.0);

   return  1 / (1+ pow(ElAbs(A-1),2));
}


/***************************************************/
/*                                                 */
/*        cPCICentr //   cPoseCdtImSec             */
/*                                                 */
/***************************************************/

class cPCICentr
{
     public :
        cPCICentr(cPoseCam * aPC0,Pt3dr aPt0);


        cPoseCam * mPC0;
        double     mProf;
        Pt3dr      mPt0;
        Pt3dr      mDir;
        ElMatrix<double> mMat;
};

cPCICentr::cPCICentr(cPoseCam * aPC0,Pt3dr aPt0) :
   mPC0    (aPC0),
   mProf   (mPC0->CurCam()->ProfondeurDeChamps(aPt0)),
   mPt0    (aPt0),
   mDir    (mPC0->CurCam()->PseudoOpticalCenter()  - mPt0),
   mMat    (1,1)
{
    Pt3dr aDirN = mDir;
    Pt3dr aU,aV;
    MakeRONWith1Vect(aDirN,aU,aV);
    ElMatrix<double>  aMatRep = MakeMatON(aU,aV);
    mMat  = gaussj(aMatRep);
}

        //  cPoseCdtImSec cPoseCdtImSec cPoseCdtImSec cPoseCdtImSec



class cPoseCdtImSec 
{
    public :
         cPoseCdtImSec(int aKP,cPoseCam *,cPCICentr aPCIC,const cChoixImMM & aCIM);
         bool Ok(const cChoixImMM & aCIM);
         double RatioStereoVert() const;
         double GainRatio() const;
         void Show() const;
         void MakeImageOccup();
         double  Recouvrt(const cPoseCdtImSec &) const;
         double PropOccup(const Pt2di & aP) {return mTImOccup.get(aP)/mSomPds;}

         int                    mKP;
         cPoseCam  *            mPC;
         Pt3dr                  mDir;
         double                 mRatioD;
         double                 mBsH;
         double                 mGain;
         Pt2dr                  mDir2;
         int                    mNbPts;
         // std::vector<double>    mRecouvrt;
         TIm2D<tImOccup,tBaseImOccup>      mTImOccup;
         double                 mSomPds;
         double                 mRatioVisib;

};


cPoseCdtImSec::cPoseCdtImSec(int aKP,cPoseCam * aPC,cPCICentr aPCIC,const cChoixImMM & aCIM) :
   mKP    (aKP),
   mPC    (aPC),
   mDir   (mPC->CurCam()->PseudoOpticalCenter()  - aPCIC.mPt0),
   mNbPts (0),
   mTImOccup   (Pt2di(1,1)),
   mRatioVisib (0)
{
    double aRatioDist = euclid(mDir) / euclid(aPCIC.mDir);
    mRatioD = ElMin(aRatioDist,1/aRatioDist);
    double aScal = scal(vunit(mDir),vunit(aPCIC.mDir));
    mBsH = acos(ElMax(-1.0,ElMin(1.0,aScal)));


    double aPenalD = 1 / (1 + pow(4*ElAbs(1-mRatioD),2));

    mGain = GainAngle(mBsH,aCIM.TetaOpt().Val()) * aPenalD;

    Pt3dr aDirLoc  =   aPCIC.mMat * vunit(mDir);
    mDir2  = Pt2dr (aDirLoc.x,aDirLoc.y);


}

double cPoseCdtImSec::RatioStereoVert() const
{
    return (1-mRatioD) /ElMax(1e-30,mBsH);
}

bool cPoseCdtImSec::Ok(const cChoixImMM & aCIM)
{
   return    (mRatioD >= aCIM.RatioDistMin().Val())
          && (mBsH >= aCIM.TetaMinPreSel().Val())
          && (mBsH <= aCIM.TetaMaxPreSel().Val())
          && (RatioStereoVert() <= aCIM.RatioStereoVertMax().Val())
          && (mNbPts>= aCIM.NbMinPtsHom().Val());
}

void cPoseCdtImSec::Show() const
{
    std::cout << mPC->Name() <<  " B/H=" << mBsH <<  " D2=" << mDir2 <<   " " << mGain << "\n";
}

double cPoseCdtImSec::GainRatio() const
{
    return mRatioVisib * mGain;
}


void cPoseCdtImSec::MakeImageOccup()
{
   mTImOccup.Resize(Pt2di(NbOccup,NbOccup));

   Pt2dr aDirU = vunit(mDir2);
   double aRho = euclid(mDir2);

   double aSigmY = 0.15;
   double aSigmX = 0.3;
   mSomPds = 0;

   Pt2di aPi;
   for (aPi.x=0 ; aPi.x <NbOccup ; aPi.x++)
   {
       for (aPi.y=0 ; aPi.y <NbOccup ; aPi.y++)
       {
           Pt2dr aPr = (Pt2dr(aPi)/double(NbOccup*0.5) -Pt2dr(1.0,1.0)) * TetaMaxOccup;
           aPr = aPr/ aDirU;
           // On privilegie les point qui sont de l'autre cote de l'image
           double aDX = (aPr.x >0) ? (ElAbs(aPr.x-aRho)) : (aRho -3*aPr.x);

           double aRho2 = ElSquare(aDX/aSigmX) +ElSquare(aPr.y/aSigmY);

           double aPds =   exp(-aRho2)  +  exp(-4*aRho2)*4 +   exp(-9*aRho2)*9 ; // ??  + exp(-aRho2/4)/4;
           // double aPds =   exp(-aRho2)  +  exp(-4*aRho2)*4 +  exp(-aRho2/4)/16;




           mSomPds += aPds;
           mTImOccup.oset(aPi,aPds);
       }
   }
   // Tiff_Im::Create8BFromFonc("Pds"+mPC->Name()+".tif",mTImOccup.sz(),mTImOccup._the_im.in()*(255.0/(1+4+9+1/4.0)));
}

double  cPoseCdtImSec::Recouvrt(const cPoseCdtImSec & aPCIS2) const
{
   if (this==&aPCIS2) return 1.0;
   Pt2di aP;
   double aSomDif=0;
   for (aP.x=0 ; aP.x <NbOccup ; aP.x++)
   {
       for (aP.y=0 ; aP.y <NbOccup ; aP.y++)
       {
           aSomDif += ElAbs(mTImOccup.get(aP)-aPCIS2.mTImOccup.get(aP));
       }
   }
    return (1-aSomDif/(mSomPds+aPCIS2.mSomPds));
}
/*
*/

class cCmpImOnGainHom
{
    public :

       bool operator()(cPoseCam* aPC1,cPoseCam * aPC2)
       {
             return aPC1->CdtImSec()->GainRatio()  > aPC2->CdtImSec()->GainRatio();
       }
};


     //=============      cSetCdtCIMS ================

class cSetCdtCIMS
{
    public :
         cSetCdtCIMS(const  int aKS,double aGain) :
              mKS   (aKS),
              mGain (aGain)
          {
          }

          bool operator < (const cSetCdtCIMS & aCdt2) const
          {
               return mGain > aCdt2.mGain;
          }

          int    mKS;
          double mGain;
};


    //  Manip Flags

int FlagOfVI(const std::vector<int> & aSub )
{
   int aRes =0 ;
   for (int aK=0 ; aK<int(aSub.size()) ; aK++)
      aRes |= 1 << aSub[aK];

   return aRes;
}

std::vector<int> VIOfFlag(int aFlag)
{
   std::vector<int> aRes;


   for (int aP=1, aLog=0 ; aP<= aFlag ; aP*=2, aLog++)
       if (aP&aFlag)
          aRes.push_back(aLog);
       
   
   return aRes;
}

     //=============      cCaseOcupIm ================

class cCaseOcupIm
{
    public :
        
        cCaseOcupIm(const Pt2dr& aCentre,double aProf) :
             mSomPds (0),
             mSomPts (0,0)
        {
            AddPts(aCentre,aProf,1e-2);
        }
        void AddPts(const Pt2dr & aP,double aProf,double aPds)
        {
             mSomPds += aPds;
             mSomPts = mSomPts + aP*aPds;
             aVProf.push_back(aProf);
        }

        void Finish(cPoseCam* aPC)
        {
             double aProf = MedianeSup(aVProf);
             mPIm = mSomPts / mSomPds;
             if (aPC->CurCam()->IsInZoneUtile(mPIm))
             {
                 mPTer  = aPC->CurCam()->ImEtProf2Terrain(mPIm,aProf);
             }
             else
             {
                  mSomPds = 0.0;
             }
             aVProf.clear();
        }

        bool PoseMeVoit(cPoseCam* aPC) const
        {
             return aPC->CurCam()->PIsVisibleInImage(mPTer);
        }
        double SomPds() const {return mSomPds;}
        double & SomPds() {return mSomPds;}
        Pt3dr PTer() const
        {
            ELISE_ASSERT(mSomPds!=0,"Nul Pds for cCaseOcupIm::PTer");
            return mPTer;
        }
    private  :

        Pt2dr mPIm;
        Pt3dr mPTer;
        std::vector<double> aVProf;
        double mSomPds;
        Pt2dr  mSomPts;
};

double SomPds(const std::vector<cCaseOcupIm>  & aVC)
{
    double aRes = 0;
    for (int aKC=0 ; aKC<int(aVC.size()) ; aKC++)
        aRes += aVC[aKC].SomPds();
    return aRes;
}
double SomPds(const std::vector<std::vector<cCaseOcupIm> > & aVVC)
{
    double aRes = 0;
    for (int aKV=0 ; aKV<int(aVVC.size()) ; aKV++)
       aRes += SomPds(aVVC[aKV]);
    return aRes;
}

double SomPdsVisible(const std::vector<cCaseOcupIm>  & aVC,cPoseCam * aPC)
{
    double aRes = 0;
    for (int aKC=0 ; aKC<int(aVC.size()) ; aKC++)
        if (aVC[aKC].PoseMeVoit(aPC))
           aRes += aVC[aKC].SomPds();
    return aRes;
}
double SomPdsVisible(const std::vector<std::vector<cCaseOcupIm> > & aVVC,cPoseCam * aPC)
{
    double aRes = 0;
    for (int aKV=0 ; aKV<int(aVVC.size()) ; aKV++)
       aRes += SomPdsVisible(aVVC[aKV],aPC);
    return aRes;
}



     //=============      cCombinPosCam ================


     //=============      cCombinPosCam ================


class cCombinPosCam
{
    public :
       cCombinPosCam(const std::vector<int> & aSub ,const std::vector<cPoseCam*>&  aVPres);
       cCombinPosCam() {}
       double GainOfCase(const cCaseOcupIm &,cSetCombPosCam &);
       double GainOfCase(const std::vector<cCaseOcupIm> &,cSetCombPosCam &);
       double GainOfCase(const std::vector<std::vector<cCaseOcupIm> >&,cSetCombPosCam &);

       int mFlag;
       int mNbIm;
       std::vector<cPoseCam*> mVCam;
       double mGainDir;   // Gain en direction sans tenir compte du fait que image ne se recouvrent pas
       double mGainGlob;  // Gain pondere en tenant compte du fait que image ne se recouvrent pas

       cOneSolImageSec MakeSol(double aPenalCard);
};

cCombinPosCam::cCombinPosCam(const std::vector<int> & aSub ,const std::vector<cPoseCam*> & aVPres) :
   mFlag  (FlagOfVI(aSub)),
   mNbIm  ((int)aSub.size()),
   mGainDir  (0),
   mGainGlob (0)
{
    for (int aKs=0 ; aKs<mNbIm ; aKs++)
    {
         int anI = aSub[aKs];
         mVCam.push_back(aVPres[anI]);
    }
    Pt2di aPi;
    for (aPi.x=0 ; aPi.x <NbOccup ; aPi.x++)
    {
         for (aPi.y=0 ; aPi.y <NbOccup ; aPi.y++)
         {
              // int aKBestIm = -1;
              double aMaxOc = 0.0;
              for (int aKIm=0 ; aKIm <mNbIm ; aKIm++)
              {
                   cPoseCdtImSec *aPCIS = mVCam[aKIm]->CdtImSec();
                   double aPOc = aPCIS->PropOccup(aPi) *aPCIS->mGain;
                   if (aPOc>aMaxOc)
                   {
                        aMaxOc = aPOc;
                   }
              }
              mGainDir +=  aMaxOc;
         }
    }
}

cOneSolImageSec cCombinPosCam::MakeSol(double aPenalCard)
{
   cOneSolImageSec aSol;
   for (int aKIm=0 ; aKIm<mNbIm ; aKIm++)
   {
       aSol.Images().push_back(mVCam[aKIm]->Name());
   }
   aSol.Coverage() =  mGainGlob;
   aSol.Score() =  mGainGlob - aPenalCard * mNbIm ;
   return aSol;
}

     //=============      cSetCombPosCam ================


class cSetCombPosCam
{
     public :
          cSetCombPosCam(const std::vector<cPoseCam*> &  aVPres);
          cCombinPosCam & GetComb(const std::vector<int> & aSub);
          cCombinPosCam & GetComb(int aFlag);
          void SetNoAMBC();
          cOneSolImageSec MakeGainGlob(const std::vector<std::vector<cCaseOcupIm> >&,cImSecOfMaster & ,double);
     private :

         bool                         mAddMBC;  // Add MapByCard
         std::vector<cPoseCam*>       mVPres;
         std::map<int,cCombinPosCam>  mMapCPC;
         std::map<int,std::list<cCombinPosCam*> >  mMapByCard;
};

void cSetCombPosCam::SetNoAMBC()
{
   mAddMBC = false;
}

cSetCombPosCam::cSetCombPosCam(const std::vector<cPoseCam*> &  aVPres) :
    mAddMBC (true),
    mVPres (aVPres)
{
}

cCombinPosCam & cSetCombPosCam::GetComb(int aFlag)
{
    std::map<int,cCombinPosCam>::iterator it = mMapCPC.find(aFlag);
    if (it != mMapCPC.end()) return it->second;

    std::vector<int>  aSub = VIOfFlag(aFlag);
    mMapCPC[aFlag] = cCombinPosCam(aSub,mVPres);
    cCombinPosCam & aRes = mMapCPC[aFlag];
    if (mAddMBC)
        mMapByCard[(int)aSub.size()].push_back(&aRes);
    return aRes;
}

cCombinPosCam & cSetCombPosCam::GetComb(const std::vector<int> & aSub)
{
   return GetComb(FlagOfVI(aSub));
}


cOneSolImageSec cSetCombPosCam::MakeGainGlob
                (
                     const std::vector<std::vector<cCaseOcupIm> >& aVV,
                     cImSecOfMaster &  aISM,
                     double aPenalCard
                )
{
      cOneSolImageSec  aRes;
      double aBestScore=0;
      SetNoAMBC();
      for (std::map<int,std::list<cCombinPosCam*> >::iterator ItM= mMapByCard.begin(); ItM!=mMapByCard.end() ; ItM++)
      {
          std::list<cCombinPosCam*> & aL = ItM->second;
          double aBestGain=0;
          cCombinPosCam * aBestCombine = 0;
          for (std::list<cCombinPosCam*>::iterator itL=aL.begin() ; itL!=aL.end() ; itL++)
          {
             cCombinPosCam * aCC = *itL;
             aCC->mGainGlob  = aCC->GainOfCase(aVV,*this);
             if (aCC->mGainGlob > aBestGain)
             {
                  aBestGain = aCC->mGainGlob;
                  aBestCombine = aCC;
             }
          }
          if (aBestCombine)
          {
              cOneSolImageSec aSol = aBestCombine->MakeSol(aPenalCard);
              aISM.Sols().push_back(aSol);
              if (aSol.Score() > aBestScore)
              {
                  aBestScore = aSol.Score();
                  aRes = aSol;
              }
          }
      }
      return aRes;
}

    //   ==========   cCombinPosCam ================

double cCombinPosCam::GainOfCase(const cCaseOcupIm & aCase,cSetCombPosCam & aSet)
{
   int aFlagRes = 0;
   int aFlagGlob = 0;
   for (int aPuis2=1, aLog=0 ; aPuis2<= mFlag ; aPuis2*=2)
   {
       if (aPuis2&mFlag)
       {
           const CamStenope * aCS = mVCam[aLog]->CurCam();
           if ((aCase.SomPds()) && (aCS->PIsVisibleInImage(aCase.PTer())))
           {
               aFlagRes |= aPuis2;
           }
           aLog++;
           aFlagGlob |= aPuis2;
       }
   }
   ELISE_ASSERT(aFlagGlob==mFlag,"Check flags in cCombinPosCam::Gain");
   
   return aSet.GetComb(aFlagRes).mGainDir * aCase.SomPds();
}

double cCombinPosCam::GainOfCase(const std::vector<cCaseOcupIm> & aVC,cSetCombPosCam & aSet)
{
    double aRes = 0;
    for (int aKC=0 ; aKC<int(aVC.size()) ; aKC++)
       aRes += GainOfCase(aVC[aKC],aSet);

    return aRes;
}

double cCombinPosCam::GainOfCase(const std::vector<std::vector<cCaseOcupIm> > & aVVC,cSetCombPosCam & aSet)
{
    double aRes = 0;
    for (int aKV=0 ; aKV<int(aVVC.size()) ; aKV++)
       aRes += GainOfCase(aVVC[aKV],aSet);

    return aRes;
}


/***************************************************/
/*                                                 */
/*                  cAppliApero                    */
/*                                                 */
/***************************************************/

bool DebugPVII = false;

bool  cAppliApero::ExportImSecMM(const cChoixImMM & aCIM,cPoseCam* aPC0,const cMasqBin3D * aMasq3D)
{

   if (aCIM.KeyExistingFile().IsInit())
   {
        std::string aNameFile =   mDC+ mICNM->Assoc1To1(aCIM.KeyExistingFile().Val(),aPC0->Name(),true);
        if (! ELISE_fp::exist_file(aNameFile))
        {
           return false;
        }
   }
   bool Test = (aPC0->Name()==std::string ("IMGP3450.PEF"));
   cPoseCam* aP44=0;


   int NbTestSetPrecis = aCIM.NbTestPrecis().Val();
   NbOccup = aCIM.NbCellOccAng().Val();
   int NbDigIm = aCIM.NbCaseIm().Val();

   double aPenal = aCIM.PenalNbIm().Val();
   cImSecOfMaster aISM;
   aISM.UsedPenal().SetVal(aPenal);
   aISM.Master() = aPC0->Name();
   aISM.ISOM_AllVois().SetVal(cISOM_AllVois());
   cISOM_AllVois &  aILV = aISM.ISOM_AllVois().Val();

   if (ShowACISec) 
       std::cout << " ************ " << aPC0->Name() << " ***********\n";
   int aNbPose = (int)mVecPose.size();
   cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale (aCIM.IdBdl(),aPC0->Name());

   int aNbPtsInNu;
   cPCICentr aPCIC(aPC0,anOLM->CentreNuage(aMasq3D,&aNbPtsInNu));

   if (aNbPtsInNu < 10)
   {
       return false;
   }

   // Initialisation a partir du centre nuage
   for(int aKP=0 ; aKP<aNbPose ;aKP++)
   {
      mVecPose[aKP]->CdtImSec() = new cPoseCdtImSec(aKP,mVecPose[aKP],aPCIC,aCIM);
      if ( mVecPose[aKP]->Name() == "IMGP3444.PEF")
         aP44 =  mVecPose[aKP];
   }
   if (Test) ELISE_ASSERT(aP44!=0,"Cannot find P44");

   // On cree un tableau de cases vides

   const CamStenope * aCS0 = aPC0->CurCam();
   Pt2di aSz = aCS0->Sz();
   double aSzCaseMoy = sqrt((aSz.x*aSz.y) / double(NbDigIm*NbDigIm));
   int aNbCaseX = round_up(aSz.x/aSzCaseMoy);
   int aNbCaseY = round_up(aSz.y/aSzCaseMoy);
   double aSzCaseX = aSz.x / double(aNbCaseX);
   double aSzCaseY = aSz.y / double(aNbCaseY);
   // std::cout << "NBCASE X " << aNbCaseX << " " << aSzCaseX << " Y "<< aSzCaseY << " " << aNbCaseY << "\n";
   std::vector<std::vector<cCaseOcupIm> > aVCase(aNbCaseY);
   for (int aKY = 0 ; aKY < aNbCaseY ; aKY++)
   {
       for (int aKX = 0 ; aKX < aNbCaseX ; aKX++)
       {
           Pt2dr aP((aKX+0.5)*aSzCaseX,(aKY+0.5)*aSzCaseY);
           aVCase.at(aKY).push_back(cCaseOcupIm(aP,aPCIC.mProf));
       }
   }

   // On compte le nombre de points de liaisons par case et par image
   //      aVCase[aKY][aKX].AddPts(aPIm,aProf,1.0);
   //      aVP[aKPos]->CdtImSec()->mNbPts++
   
   const std::vector<cOnePtsMult *> &  aVPM = anOLM->VPMul();
   for (int aKPt=0 ; aKPt<int(aVPM.size()) ;aKPt++)
   {
        cOnePtsMult & aPMul = *(aVPM[aKPt]);
        if (aPMul.MemPds() >0)
        {
           cOneCombinMult * anOCM = aPMul.OCM();
           const std::vector<cGenPoseCam *> & aVP = anOCM->GenVP();
           bool Ok = true;
           std::vector<double> aVPds;
           Pt3dr aPI = aPMul.QuickInter(aVPds);
           if (aMasq3D)
           {
              Ok =  aMasq3D->IsInMasq(aPI);
           }

           if (Ok)
           {
              Pt2dr aPIm = aCS0->R3toF2(aPI);
              int aKX = round_down(aPIm.x/aSzCaseX);
              int aKY = round_down(aPIm.y/aSzCaseY);
              if ((aKX>=0) && (aKX<aNbCaseX) && (aKY>=0) && (aKY<aNbCaseY))
              {
                   double aProf = aCS0->ProfondeurDeChamps(aPI);
                   aVCase[aKY][aKX].AddPts(aPIm,aProf,1.0);
              }
              for (int aKPos=1 ; aKPos<int(aVP.size()) ;aKPos++)
              {
                  aVP[aKPos]->CdtImSec()->mNbPts++;
              }
           }
        }
   }



   // On finalise les cases : calcul 3D et  modulation de la fonction de poids
   {
       double aSomPds = SomPds(aVCase);
       double aSomMoy = aSomPds / (aNbCaseY * aNbCaseX);

       // Limitation des pois fort
       for (int aKY = 0 ; aKY < aNbCaseY ; aKY++)
       {
           for (int aKX = 0 ; aKX < aNbCaseX ; aKX++)
           {
               cCaseOcupIm & aCase  =  aVCase.at(aKY).at(aKX);
               aCase.Finish(aPC0);
               double  & aSomP = aCase.SomPds();
               aSomP  = aSomP /aSomMoy;
               if (aSomP > 1) aSomP =  2.0 - 1 / aSomP;
           }
       }

       // Mis a somme de 1 le poids sur les cases
       aSomPds = SomPds(aVCase);
       for (int aKY = 0 ; aKY < aNbCaseY ; aKY++)
       {
           for (int aKX = 0 ; aKX < aNbCaseX ; aKX++)
           {
               cCaseOcupIm & aCase  =  aVCase.at(aKY).at(aKX);
               aCase.SomPds()  /= aSomPds;
           }
       }
   }


    // Pre selection  (cChoixImMM : aCIM)
    // On 
    //   - selectionne les images   sur un criteres purement geometrique (surtout pour limiter la combinatoire)
    //   -  calcule (pondere par les cases) de la visibilite de  l'image principale dans chaque image secondair

    std::vector<cPoseCam*> aVPPres;
    for(int aKP=0 ; aKP<aNbPose ;aKP++)
    {
       cPoseCam * aPC = mVecPose[aKP];
       aPC->CdtImSec()->mRatioVisib = SomPdsVisible(aVCase,aPC);
       bool Ok = aPC->CdtImSec()->Ok(aCIM);
       if (Ok)
       {
           aVPPres.push_back(aPC);

           cISOM_Vois aV;
           aV.Name() = aPC->Name();
           aV.Nb() = aPC->CdtImSec()->mNbPts;
           aV.Angle() =  aPC->CdtImSec()->mBsH;
           aV.RatioVis().SetVal(aPC->CdtImSec()->mRatioVisib);
           aILV.ISOM_Vois().push_back(aV);
       }
    }

   // Limitation du nombre si necessaire , on selectionne les K premiere images sur le critere
   // Gain * Ratio  (Gain purement geom, Ratio recouvrement)

    cCmpImOnGainHom aCmp;
    std::sort(aVPPres.begin(),aVPPres.end(),aCmp);
    while (int(aVPPres.size()) >aCIM.NbMaxPresel().Val()) aVPPres.pop_back();
    int aNbImAct = (int)aVPPres.size();


    // Calcul de l'image d'occupation et de la matrice de recouvrement
    for (int aKP=0 ; aKP<aNbImAct ;aKP++)
    {
        aVPPres[aKP]->CdtImSec()->MakeImageOccup();
    }
    ElMatrix<double> aMatRec(aNbImAct,aNbImAct);
    for (int aKP1=0 ; aKP1<aNbImAct ;aKP1++)
    {
        for (int aKP2=aKP1 ; aKP2<aNbImAct ;aKP2++)
        {
             double aRec = aVPPres[aKP1]->CdtImSec()->Recouvrt(*(aVPPres[aKP2]->CdtImSec()));
             aMatRec(aKP1,aKP2) = aRec;
             aMatRec(aKP2,aKP1) = aRec;
        }
     }


    //double aBestScoreGlob = -10;
    //cOneSolImageSec aBestSol;
    
    cSetCombPosCam aSetComb(aVPPres);
    // ON TESTE LES SUBSET 
    int aMaxCard = ElMin(aNbImAct,aCIM.CardMaxSub().Val()); 
    for (int aCard=1 ; aCard<=aMaxCard  ; aCard++)
    {
         // On selectionne les subset qui ont le bon cardinal
         std::vector<std::vector<int> > aSubSub;
         GetSubset(aSubSub,aCard,aNbImAct);
         std::vector<cSetCdtCIMS> aVSetPond;

         // On calcule un cout avec une formule approx Sigm(gain/Sigma(Rec))
         for (int aKS = 0 ; aKS<int(aSubSub.size()) ; aKS++)
         {
              const std::vector<int> & aSet = aSubSub[aKS];
              double aSomGain = 0;
              double aSomVis = 0;
              for (int aKIm1=0 ; aKIm1 <aCard ; aKIm1++)
              {
                   double aSomRec = 0;
                   for (int aKIm2=0 ; aKIm2 <aCard ; aKIm2++)
                   {
                        aSomRec += aMatRec(aSet[aKIm1],aSet[aKIm2]);
                   }
                   aSomGain += aVPPres[aSet[aKIm1]]->CdtImSec()->mGain  / aSomRec;
                   aSomVis +=  aVPPres[aSet[aKIm1]]->CdtImSec()->mRatioVisib;
              }
              aSomVis /= aCard;
              aVSetPond.push_back(cSetCdtCIMS(aKS,aSomGain*aCard));
         }

         // On trie pour avoir les K meilleurs sur ce cout approx
         std::sort(aVSetPond.begin(),aVSetPond.end());
         int aNbTest = ElMin(NbTestSetPrecis,int(aVSetPond.size()));

         for (int aKTest=0; aKTest<aNbTest ; aKTest++)
         {
              const std::vector<int>  & aTestSet = aSubSub[aVSetPond[aKTest].mKS];
              aSetComb.GetComb(aTestSet);  // Insere la combinaison
         }
    }

    // On calcul les cout exact
    aSetComb.SetNoAMBC();

    ElTimer aChrono;
    cOneSolImageSec aBestSol= aSetComb.MakeGainGlob( aVCase,aISM,aPenal);


    if (ShowACISec) 
    {
       for (int aKP=0 ; aKP<int(aVPPres.size()) ;aKP++)
       {
           aVPPres[aKP]->CdtImSec()->Show();
       }
       for (int aKP1=0 ; aKP1<aNbImAct ;aKP1++)
       {
           for (int aKP2=0 ; aKP2<aNbImAct ;aKP2++)
           {
                printf("%5f ",aMatRec(aKP1,aKP2));
           }
                printf("\n");
       }
       
    }


   // calcul de l'occupation de la sphere des direction



   // Fin, on libere la mémoire tempo sur les cPoseCdtImSec
   for(int aKP=0 ; aKP<aNbPose ;aKP++)
   {
       delete mVecPose[aKP]->CdtImSec() ;
   }

   if (aCIM.KeyAssoc().IsInit())
   {
      std::string aName = mDC + mICNM->Assoc1To1(aCIM.KeyAssoc().Val(),aPC0->Name(),true);
      MakeFileXML(aISM,aName);
   }
   std::cout << "Chx : " << aPC0->Name()  << " Nb:" <<  aBestSol.Images().size() ;
   int aCpt=0;
   for (std::list<std::string>::iterator itS=aBestSol.Images().begin(); itS!=aBestSol.Images().end() ; itS++)
   {
       std::cout << ((aCpt==0) ? " [" : "|") <<  ExtractDigit(StdPrefixGen(*itS),"XXXX") ;
       aCpt++;
   }
   std::cout << "] Cov:" << aBestSol.Coverage()  << "\n";

   return true;
}





void cAppliApero::ExportImMM(const cChoixImMM & aCIM)
{ 


    cListOfName aLON;
    cSetName *  aSelector = mICNM->KeyOrPatSelector(aCIM.PatternSel());
    cMasqBin3D * aMasq3D = 0;
    if (aCIM.Masq3D().IsInit())
       aMasq3D= cMasqBin3D::FromSaisieMasq3d(DC()+aCIM.Masq3D().Val());

    for(int aKP=0 ; aKP<int(mVecPose.size()) ;aKP++)
    {
       cPoseCam* aPC = mVecPose[aKP];
       if (aSelector->IsSetIn(aPC->Name()))
       {
           bool Ok = ExportImSecMM(aCIM,aPC,aMasq3D);
           if (Ok)
           {
              aLON.Name().push_back(aPC->Name());
           }
       }
    }
    if (aCIM.FileImSel().IsInit())
    {
          MakeFileXML(aLON,DC()+aCIM.FileImSel().Val());
    }


    // Chek Flag
    if (0)
    {
       for (int aK=0 ; aK< 2000 ; aK++)
       {
           std::cout << aK << " " << FlagOfVI(VIOfFlag(aK)) << "\n";
           ELISE_ASSERT(aK==FlagOfVI(VIOfFlag(aK)) ,"Flag Incohe");
       }
    }
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
Footer-MicMac-eLiSe-25/06/2007*/
