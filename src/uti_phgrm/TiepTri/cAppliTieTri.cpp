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


#include "TiepTri.h"

/*
class cCmpPt2diOnEuclid
{
   public :
       bool operator () (const Pt2di & aP1, const Pt2di & aP2)
       {
                   return euclid(aP1) < euclid(aP2) ;
       }
};

std::vector<Pt2di> VoisinDisk(double aDistMin,double aDistMax)
{
   std::vector<Pt2di> aResult;
   int aDE = round_up(aDistMax);
   Pt2di aP;
   for (aP.x=-aDE ; aP.x <= aDE ; aP.x++)
   {
       for (aP.y=-aDE ; aP.y <= aDE ; aP.y++)
       {
            double aD = euclid(aP);
            if ((aD <= aDistMax) && (aD>aDistMin))
               aResult.push_back(aP);
       }
   }
   return aResult;
}
*/


cAppliTieTri::cAppliTieTri
(
   const cParamAppliTieTri & aParam,
   cInterfChantierNameManipulateur * anICNM,
   const std::string & aDir,
   const std::string & anOri,
   const cXml_TriAngulationImMaster & aTriang
)  :
     cParamAppliTieTri (aParam),
     mICNM          (anICNM),
     mDir           (aDir),
     mOri           (anOri),
     mWithW         (false),
     mDisExtrema    (TT_DIST_EXTREMA),
     mDistRechHom   (TT_DIST_RECH_HOM),
     mCurPlan       (Pt3dr(0,0,0),Pt3dr(1,0,0),Pt3dr(0,1,0)),
     mNbTriLoaded   (0),
     mNbPts         (0),
     mTimeCorInit   (0.0),
     mTimeCorDense  (0.0),
     mHasPtSelecTri (false),
     mHasNumSelectImage (false),
     mKeyMasqIm         ("NKS-Assoc-STD-Masq"),
     mMoyDifAffHomo (Pt2dr(0,0)),
     mCountDiff (0),
     mHistoErrAffHomoX (vector<int> (round_up(mMaxErr/0.01))),
     mHistoErrAffHomoY (vector<int> (round_up(mMaxErr/0.01)))

{
   mMasIm = new cImMasterTieTri(*this,aTriang.NameMaster());

   for (int aK=0 ; aK<int(aTriang.NameSec().size()) ; aK++)
   {
      mImSec.push_back(new cImSecTieTri(*this,aTriang.NameSec()[aK],aK));
   }

   mVoisExtr = SortedVoisinDisk(0.5,mDisExtrema,true);
   mVoisHom = SortedVoisinDisk(-1,mDistRechHom,false);

   cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
   mInterpolSinC = new cTabIM2D_FromIm2D<tElTiepTri>(aSinC,1000,false);


   mInterpolBilin = new cInterpolBilineaire<tElTiepTri>;

   cCubicInterpKernel * aBic = new cCubicInterpKernel(-0.5);
   // mInterpolBicub = new cTplCIKTabul<tElTiepTri,tElTiepTri>(10,8,-0.5);
   mInterpolBicub = new cTabIM2D_FromIm2D<tElTiepTri>(aBic,1000,false);

   if (mUseHomo)
   {
       if (ELISE_fp::exist_file("ErrLogAffHomo.txt"))
           mErrLog.open("ErrLogAffHomo.txt", std::ios_base::app);
       else
           mErrLog.open("ErrLogAffHomo.txt");
   }
}


void StatCorrel(const  std::vector<cResulMultiImRechCorrel*> &  aVec, const std::string & aMes)
{
    //============= Statistic sur un vector result de correlation =============
    double aSomC = 0;
    int aNbC = 0;
    std::vector<double> aVCor;

    for (int aKR = 0 ; aKR<int(aVec.size()) ; aKR++)
    {
        cResulMultiImRechCorrel * aRMIRC =  aVec[aKR];  // prendre result un pt Master
        std::vector<cResulRechCorrel > &  aVRRC = aRMIRC->VRRC() ;
        for (int aKIndIm=0 ; aKIndIm<int(aVRRC.size()) ; aKIndIm++)
        {
            aVCor.push_back(aVRRC[aKIndIm].mCorrel);    // prendre score correl avec chaque pt 2nd
            aSomC += aVRRC[aKIndIm].mCorrel ;           // accumuler les valeurs de scores de correls
            aNbC ++ ;                                   // nombre de couple Master-2nd
        }
    }
    if (aVCor.size() > 0)
    {
        std::cout << "StatC:" << aMes
                  << " Moy=" << aSomC/aNbC                  // score correl moyen
                  << " Med=" << KthValProp(aVCor,0.5)       // score median
                  << " 20%=" << KthValProp(aVCor,0.2)       // score à 20% en premier
                  << " Nb=" << aNbC
                  << "\n";
    }
}


void print_progress_bar(int percentage)
{
  string progress = ToString(percentage) + "% " +"[" + string(percentage, '*') + string(100 - percentage, ' ') + "]";
 // cout << progress << "\r\033[F\033[F\033[F" << flush;
  //cout<<"."<<flush;
  cout << "\r\033[F" << progress << flush;
}


void cAppliTieTri::DoAllTri(const cXml_TriAngulationImMaster & aTriang)
{
    // ==== Parcour des triangles =============
    cout<<"Im: " <<aTriang.NameMaster()<<endl<<endl;
    int aNbTri = aTriang.Tri().size();

    for (int aK=0 ; aK<int(aTriang.Tri().size()) ; aK++)
    {
        DoOneTri(aTriang.Tri()[aK],aK);
        /*
        if (aK<aTriang.Tri().size()-2)
            print_progress_bar(aK*100/aTriang.Tri().size());
        else
            print_progress_bar(100);
        cout << endl;
        */


        if ( ( (aK%20)==0) && (! mWithW))
        {
            std::cout << "Av = "  << (aNbTri-aK) * (100.0/aNbTri) << "% "
                      << " NbP/Tri " << double(mNbPts) / mNbTriLoaded
                      << "\n";
        }


    }
    if (mErrLog.is_open())
        mErrLog.close();

    std::cout << "NB TRI LOADED = " << mNbTriLoaded << "\n";


    if (CurEtapeInFlagFiltre())
    {
       StatCorrel(mGlobMRIRC,"Avant");

       mGlobMRIRC = FiltrageSpatial(mGlobMRIRC,mDistFiltr,TT_FSDeltaCorrel);
    }
    StatCorrel(mGlobMRIRC,"Apres");


    // ==== Prepare la structure de points multiples =============
    vector<string> * aVIm = new vector<string>();
    //cout<<"  ++ ImMaster :"<<Master()->NameIm()<<endl;
    for (int aKIm=0 ; aKIm<int(mImSec.size()) ; aKIm++)
    {
        cImSecTieTri* aImSec = mImSec[aKIm];
        //cout<<"  ++ Im2nd : "<<aImSec->Num()<<" - "<<aImSec->NameIm()<<endl;
        aVIm->push_back(aImSec->NameIm());
    }
    aVIm->push_back(Master()->NameIm());


    cSetTiePMul * aMulHomol = new cSetTiePMul(0, aVIm); // [Im2nd...ImMaster]
    vector< vector<int> > VNumIms;
    vector< vector<Pt2dr> > VPtsIms;

    //====== Parcour les  PtsMul et rempli Hom classique + nouvelle structure==========//
    {
        for (int aKP=0 ; aKP<int(mGlobMRIRC.size()) ; aKP++)
        {
            //=====================//
            vector<int>  aNumIms;
            vector<Pt2dr>  aPtsIms;
            //======================//

             cResulMultiImRechCorrel & aRMIRC =  *(mGlobMRIRC[aKP]);
             Pt2dr aPMaster (aRMIRC.PtMast());
             const std::vector<int> &   aVInd = aRMIRC.VIndex();
             int aNbIm = aVInd.size();
             ELISE_ASSERT(aNbIm==int(aRMIRC.VRRC().size()),"Incoh size in cAppliTieTri::DoAllTri");
             for (int aKI=0 ; aKI<aNbIm ; aKI++)
             {
                 const cResulRechCorrel & aRRC = aRMIRC.VRRC()[aKI];

                 //std::cout << "Corr " << aRRC.mCorrel << " " << aRMIRC.IsInit() << " KT=" << aTMIRC.KT() << "\n";
                 if (aRRC.IsInit())
                 {
                    cImSecTieTri * anIm = mImSec[aVInd[aKI]];
                    anIm->PackH().Cple_Add(ElCplePtsHomologues(aPMaster,aRRC.mPt)) ;
                    //=================//
                    //aNumIms.push_back(aKI);           // -> bug sur index of image when export new FH ?
                    aNumIms.push_back(anIm->Num());
                    aPtsIms.push_back(aRRC.mPt);
                    //==================//

                 }
                 else
                 {
                      // Ce cas peut sans doute se produire legitimment si la valeur entiere etait trop proche
                      // de la frontiere et que toute les  tentative etaient en dehors
                      //ELISE_ASSERT(false,"Incoh init in cAppliTieTri::DoAllTri");
                      // getchar();
                 }
             }
             //===============//
             //add 1 config
             if (aNumIms.size() > 0 && aPtsIms.size() > 0)
             {
                aNumIms.push_back(mImSec.size());
                aPtsIms.push_back(aPMaster);
                vector<float> vAttr;
                ELISE_ASSERT (aNumIms.size() == aPtsIms.size(), "Nb Imgs & Nb Pts not coherent, new Format Homol");
                VNumIms.push_back(aNumIms);
                VPtsIms.push_back(aPtsIms);
                //aHomol->AddPts(aNumIms,aPtsIms,vAttr);
             }
             //===============//
        }
    }

    // Sauve au nouveau format
    cout<<"Write pts homo to disk:..."<<endl;
    cout<<" ++ ImMaster :"<<Master()->NameIm()<<endl;
    for (uint aKHomol=0; aKHomol<VNumIms.size(); aKHomol++)
    {
        vector<float> vAttr;
        aMulHomol->AddPts(VNumIms[aKHomol], VPtsIms[aKHomol],vAttr);
    }
    std::string aHomolOut = mHomolOut;
    string aPmulHomolName = "Homol" + aHomolOut + "/PMul_" + this->Master()->NameIm() + ".txt";
    aMulHomol->Save(aPmulHomolName);


    // Sauve a l'ancien format
    for (int aKIm=0 ; aKIm<int(mImSec.size()) ; aKIm++)
    {
        cImSecTieTri* aImSec = mImSec[aKIm];
        std::string pic1 = Master()->NameIm();
        std::string pic2 = aImSec->NameIm();
        // La classe cHomolPackTiepTri semble n'apporter aucun service par rapport a sauver directement ...
        cHomolPackTiepTri aPack(pic1, pic2, aKIm, mICNM, true); //true = skipPackVide
        aPack.Pack() = aImSec->PackH();
        if (aPack.Pack().size() > 0)
        {
            cout<<"  ++ Im2nd : "<<aImSec->NameIm();
            cout<<" - Nb Pts= "<<aImSec->PackH().size()<<endl;
        }
        aPack.writeToDisk(aHomolOut);
    }
    if (mUseHomo)
    {
        cout<<"+ Diff Homo Stat: "<<mMoyDifAffHomo/mCountDiff<<" Nb Acc = "<<mCountDiff<<" Max = "<<mMaxDifAffHomo<<endl;
        for (uint aKHisto=0; aKHisto < HistoErrAffHomoX().size(); aKHisto++)
        {
            cout<<"    + Bin "<<aKHisto<<" : "<<HistoErrAffHomoX()[aKHisto]<<endl;
        }
    }
}

/*
void cAppliTieTri::RechHomPtsDense(cResulMultiImRechCorrel & aRMIRC)
{
     std::vector<cResulRechCorrel > & aVRRC = aRMIRC.VRRC();


     for (int aKNumIm = 0 ; aKNumIm <int(aVRRC.size())  ; aKNumIm++)
     {
         cResulRechCorrel & aRRC = aVRRC[aKNumIm];
         int aKIm = aRMIRC.VIndex()[aKNumIm];

         // aRRC = mImSecLoaded[aKIm]->RechHomPtsDense(aRMIRC.PMaster().mPt,aRRC);
         aRRC = mImSec[aKIm]->RechHomPtsDense(false,aRMIRC.PtMast(),aRRC);
     }
}
*/

void cAppliTieTri::PutInGlobCoord(cResulMultiImRechCorrel & aRMIRC,bool WithDecal,bool WithRedr)
{

     aRMIRC.PIMaster().mPt = aRMIRC.PtMast() + mMasIm->Decal();
     std::vector<cResulRechCorrel> & aVRRC = aRMIRC.VRRC();


     for (int aKNumIm=0 ; aKNumIm<int(aVRRC.size()) ; aKNumIm++)
     {
         cResulRechCorrel & aRRC = aVRRC[aKNumIm];
         int aKIm = aRMIRC.VIndex()[aKNumIm];
         if (WithRedr)
         {
             if (mUseHomo)
             {
                 aRRC.mPt = mImSec[aKIm]->Mas2Sec_Hom(aRRC.mPt);
             }
             else
             {
                aRRC.mPt = mImSec[aKIm]->Mas2Sec(aRRC.mPt);
             }
         }

         if (WithDecal)
            aRRC.mPt = aRRC.mPt + Pt2dr(mImSec[aKIm]->Decal());
     }
}

bool cAppliTieTri::CurEtapeInFlagFiltre() const
{
   return  (mFlagFS&(1<<mCurEtape)) !=0;
}


void cAppliTieTri::DoOneTri(const cXml_Triangle3DForTieP & aTri,int aKT )
{
    mPIsInImRedr = true;

    // ================  Chargement des images ======================
    //   Cela inclut le calcul des points d'interet pour toute les images
    //   ainsi qu'un filtrage spatial sur l'image Master, selon le critere Fast

    if (!  mMasIm->LoadTri(aTri)) return;

    mNbTriLoaded++;

    mCurPlan = cElPlan3D(aTri.P1(),aTri.P2(),aTri.P3());
    mImSecLoaded.clear();
    for (int aKNumIm=0 ; aKNumIm<int(aTri.NumImSec().size()) ; aKNumIm++)
    {
        int aKIm = aTri.NumImSec()[aKNumIm];
        if ( mImSec[aKIm]->LoadTri(aTri))
        {
            mImSecLoaded.push_back(mImSec[aKIm]);
        }
    }

    if (mImSecLoaded.size() == 0)
       return;


    // ================ Calcul des correlations entieres ======================

    mCurEtape = ETAPE_CORREL_ENT;
    if (mWithW && (mEtapeInteract==0))
    {
         while (1)
         {
              cIntTieTriInterest aPI= mMasIm->GetPtsInteret();
              for (int aKIm=0 ; aKIm<int(mImSecLoaded.size()) ; aKIm++)
              {
                  mImSecLoaded[aKIm]->RechHomPtsInteretEntier(true,aPI);  //1pxl/2 -> pxl entier   //  sub pxl
              }
         }
    }

    {
         const std::list<cIntTieTriInterest> & aLIP =  mMasIm->LIP();
         ElTimer aChrono;
         for (std::list<cIntTieTriInterest>::const_iterator itI=aLIP.begin(); itI!=aLIP.end() ; itI++)
         {
              cResulMultiImRechCorrel * aRMIRC = new cResulMultiImRechCorrel(*itI);
              for (int aKIm=0 ; (aKIm<int(mImSecLoaded.size()))  ; aKIm++)
              {
                  cResulRechCorrel aRes = mImSecLoaded[aKIm]->RechHomPtsInteretEntier(false,*itI);
                  if (aRes.IsInit())
                  {
                     aRMIRC->AddResul(aRes,mImSecLoaded[aKIm]->Num());
                     if (mWithW && (mEtapeInteract != 0))
                     {
                         cout<<"  + Correl INT : "<<mImSecLoaded[aKIm]->NameIm()<<" - "<<aRes.mCorrel<<" - "<<aRes.mPt<<endl;
                     }
                  }
              }
              if (aRMIRC->IsInit())
              {
                  mVCurMIRMC.push_back(aRMIRC);
              }
              else
              {
                  delete aRMIRC;
              }
         }
         mTimeCorInit += aChrono.uval();
    }
    if (CurEtapeInFlagFiltre())    // Flag = 1 => Filter Spatial in each triangle
        mVCurMIRMC = FiltrageSpatial(mVCurMIRMC,mDistFiltr/TT_RatioCorrEntFiltrSpatial,TT_FSDeltaCorrel);

    // ================ Calcul des correlations sous pixellaire ======================


    {
       for (mCurEtape=ETAPE_CORREL_BILIN ; mCurEtape<=mLastEtape ; mCurEtape++)    // mLastEtape = LastStep in command's parameter. default = 2
       {
           mPIsInImRedr = (mCurEtape <ETAPE_CORREL_DENSE); // mCurEtape = 1 => Point correl est pts redress ; = 2 => non
           bool ModeInteractif = mWithW && (mEtapeInteract==mCurEtape);
           for (int aKp=0 ; aKp<int(mVCurMIRMC.size()) ; /* aKp++ SURTOUT PAS INCREMENTER FAIT EN FIN DE BOUCLE !! */ )  // => aKp incremente seulement quand !ModeInteractif
           {
if (ModeInteractif) std::cout << "AAAAAA\n";
               cResulMultiImRechCorrel * aRMIRC = ModeInteractif ?  mMasIm->GetRMIRC(mVCurMIRMC) : mVCurMIRMC[aKp];
if (ModeInteractif) std::cout << "BBBBBB\n";
               //cResulMultiImRechCorrel * aRMIRC = mVCurMIRMC[aKp];

               const std::vector<int> &   aVI =  aRMIRC->VIndex() ;
               std::vector<cResulRechCorrel > &  aVRRC = aRMIRC->VRRC() ;
               for (int aKIndIm=0 ; aKIndIm<int(aVI.size()) ; aKIndIm++)
               {
                   int aKIm =  aVI[aKIndIm];
                   cResulRechCorrel  aRRC = 
                                          (mCurEtape==ETAPE_CORREL_BILIN)                                                         ?
                                          // Etape 1 (redress) => RechHomPtsInteretBilin
                                          mImSec[aKIm]->RechHomPtsInteretBilin(ModeInteractif,*aRMIRC,aKIndIm) :    
                                          // Etape 2 (original) => RechHomPtsDense
                                          mImSec[aKIm]->RechHomPtsDense(ModeInteractif,*aRMIRC,aKIndIm)        ;    
                   if (! ModeInteractif)
                      aVRRC[aKIndIm] = aRRC;
               }
               if (! ModeInteractif) aKp++;
           }

            // 1=> en geometrie redressee, 2 en geometrie initiale
 
            cResulMultiImRechCorrel::SuprUnSelect(mVCurMIRMC);
            double aRatio = (mCurEtape==ETAPE_CORREL_BILIN) ? TT_RatioCorrSupPix :   TT_RatioCorrLSQ;
            // mFlagFS = 2 => Filtrage Spa en Etape 1; mFlagFS = 4 => Filtrage Spa en Etape 2
            if (CurEtapeInFlagFiltre())
                mVCurMIRMC = FiltrageSpatial(mVCurMIRMC,mDistFiltr/aRatio,TT_FSDeltaCorrel);
       }
    }


    if (mMasIm->W())
    {
        mMasIm->W()->disp().clik();
    }

    for (int aKp=0 ; aKp<int(mVCurMIRMC.size()) ; aKp++)
    {
        //  PutInGlobCoord( .. ,bool WithDecal,bool WithRedr)
        PutInGlobCoord(*mVCurMIRMC[aKp],true,(mLastEtape<=1));  // (mLastEtape<=1) => to know in which geometry we are
        // PutInGlobCoord(*mVCurMIRMC[aKp],true,false);
    }

//   std::cout << "NBPPSS " << mVCurMIRMC.size() << "\n";
    mNbPts += mVCurMIRMC.size();
    // mVGlobMIRMC.push_back(cOneTriMultiImRechCorrel(aKT,mVCurMIRMC));
    // std::copy(mSetIm->begin(),mSetIm->end(),back_inserter(aLN));
    std::copy(mVCurMIRMC.begin(),mVCurMIRMC.end(),std::back_inserter(mGlobMRIRC));
    mVCurMIRMC.clear();
    mCurEtape = ETAPE_FINALE;
}

class cCmpPtrRMIRC
{
    public :
          bool operator() (cResulMultiImRechCorrel * aRMIRC1, cResulMultiImRechCorrel * aRMIRC2)
          {
               return aRMIRC1->Score() > aRMIRC2->Score();
          }
};



void  cAppliTieTri::SetSzW(Pt2di aSzW, int aZoom)
{
    mSzW = aSzW;
    mZoomW = aZoom;
    mWithW = true;
}


void cAppliTieTri::SetPtsSelect(const Pt2dr & aP)
{
    mHasPtSelecTri = true;
    mPtsSelectTri = aP;
}

bool cAppliTieTri::HasPtSelecTri() const {return mHasPtSelecTri;}
const Pt2dr & cAppliTieTri::PtsSelectTri() const {return mPtsSelectTri;}

bool cAppliTieTri::NumImageIsSelect(const int aNum) const
{
   if (!mHasNumSelectImage) return true;
   return BoolFind(mNumSelectImage,aNum);
}


void cAppliTieTri::SetNumSelectImage(const std::vector<int> & aVNum)
{
     mHasNumSelectImage = true;
     mNumSelectImage = aVNum;
}


const std::string &  cAppliTieTri::KeyMasqIm() const
{
    return mKeyMasqIm;
}
void cAppliTieTri::SetMasqIm(const  std::string  & aKeyMasqIm)
{
    mKeyMasqIm = aKeyMasqIm;
}





cInterfChantierNameManipulateur * cAppliTieTri::ICNM()      {return mICNM;}
const std::string &               cAppliTieTri::Ori() const {return mOri;}
const std::string &               cAppliTieTri::Dir() const {return mDir;}

Pt2di cAppliTieTri::SzW() const {return mSzW;}
int   cAppliTieTri::ZoomW() const {return mZoomW;}
bool  cAppliTieTri::WithW() const {return mWithW;}


cImMasterTieTri * cAppliTieTri::Master() {return mMasIm;}

const std::vector<Pt2di> &   cAppliTieTri::VoisExtr() const { return mVoisExtr; }
const std::vector<Pt2di> &   cAppliTieTri::VoisHom() const { return mVoisHom; }


bool &   cAppliTieTri::Debug() {return mDebug;}
const double &   cAppliTieTri::DistRechHom() const {return mDistRechHom;}

const cElPlan3D & cAppliTieTri::CurPlan() const {return mCurPlan;}

tInterpolTiepTri * cAppliTieTri::Interpol() 
{
   if (mNumInterpolDense==0) return mInterpolBilin;
   if (mNumInterpolDense==1) return mInterpolBicub;
   if (mNumInterpolDense==2) return mInterpolSinC;

   ELISE_ASSERT(false,"AppliTieTri::Interp");
   return 0;
}




/***************************************************************************/

cIntTieTriInterest::cIntTieTriInterest(const Pt2di & aP,eTypeTieTri aType,const double & aQualFast) :
   mPt       (aP),
   mType     (aType),
   mFastQual (aQualFast),
   mSelected (true)
{
}

cIntTieTriInterest::cIntTieTriInterest(const cIntTieTriInterest & aP)
{
    mPt        = aP.mPt;
    mType      = aP.mType;
    mFastQual  = aP.mFastQual;
    mSelected  = aP.mSelected;
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
