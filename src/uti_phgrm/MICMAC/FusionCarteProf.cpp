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



// template <class Type> cFil


/*********************************************************************/
/*                                                                   */
/*                       cFusionCarteProf                            */
/*                                                                   */
/*********************************************************************/

// 5200.71341656884579

//  Orientation-IMGP7043.JPG
//                 <Centre>2.10219430796389961 -1.59390571262596681 -6.83295583978635435</Centre>
//  Orientation-IMGP7042.JPG
//                <Centre>2.20813991647144192 -0.622937523946268445 -7.29642287586240545</Centre>
//  Orientation-IMGP7041.JPG
//               <Centre>2.23797875030617188 0.257920861375460941 -7.46202393391516239</Centre>
// Dist ->2
// 9.78319836420305178e-05  -> 1E-4

// mResolPlaniReel



//mResolPlaniReel 2 Equi 0.000195664
//mResolPlaniReel 2 Equi 0.000195664



// static bool  LocBug=false;

// Type est le type de stockage des cartes : apparemment sur des float par defaut

double DynCptrFusDepthMap=10;

template <class Type> class  cLoadedCP;        // Represente une carte de profondeur chargee
template <class Type> class cFusionCarteProf;  // Represente l'appli globale

class cElPilePrgD;  // Element minimaliste sauvgarder lors de la prog dyn

// Au depart on se sait pas quelle sera la taille de la nappe (nb de cluster/ x,y)
// , on sauvegarde toute l'information 
// necessaire dans une liste de vecteur de cElTmp0Pile; ensuite one reparcourir la liste dans le
// meme ordre pour remplir la structure de programmation dynamique
class cElTmp0Pile;  

class cTmpPile; // Utilise pour le premier filtrage (gaussien etc) avant reduction du nombre

    //=================================


class cElPilePrgD
{
    public :
        cElPilePrgD (double aZ) :
           mZ  (aZ)
        {
        }
        cElPilePrgD () :
           mZ    (0)
        {
        }

        void InitTmp(const cTplCelNapPrgDyn<cElPilePrgD> & aCel)
        {
            *this = aCel.ArgAux();
        }


        const float & Z() const {return mZ;}
    private :
        float mZ;   // C'est le Z "absolu" du nuage, corrige de l'offset et du pas, c'est a la sauvegarde qu'on le remet eventuellement au pas
};

class cElTmp0Pile : public cElPilePrgD
{
    public :
        cElTmp0Pile (double aZ,double aPds,double aCptr= -1,const cLoadedCP<float> * aLCP = 0) :
           cElPilePrgD(aZ),
           mPdsPile  (aPds),
           mCPtr (aCptr),
           mLCP  (aLCP)
        {
        }

        cElTmp0Pile() :
           cElPilePrgD(),
           mPdsPile(-1),
           mCPtr (0),
           mLCP(0)
        {
        }

        void SetPdsPile(float aPds) {mPdsPile = aPds;}
        const float & P() const {return mPdsPile;}
        const float & CPtr() const {return mCPtr;}
        const cLoadedCP<float> *  LCP() const {return mLCP; }
    private :
        float mPdsPile;  // Poids
        float mCPtr;   // Compteur
        const cLoadedCP<float> * mLCP;
};


class cTmpPile
{
    public :
        cTmpPile(int aK,float aZ,float aPds,const cLoadedCP<float> * aLCP);
        // Caclul entre deux cellule successive le poids exponentiel 
       // qui sera utilise pour le filtrage recursif
        void SetPPrec(cTmpPile &,float aExpFact);
        // double Pds() {return mPds0 / mNb0;}
        double ZMoy() {return mPds0 ? (mZP0/mPds0) : 0 ;}

    // private :
         int    mK;
         double mCpteur;
         double mZInit;
         double mPInit;
         double mZP0;   // Z Pondere par le poids
         double mPds0;  // Poids 
         double mNb0;   // si le PdsInit=1, alors mNb0==mPds0, compte le nombre de pt de chaque cluste
                        // (a une constante globale pres)
         const cLoadedCP<float> * mLCP;

        // Variale pour le filtrage recursif "plus"
         double mNewZPp;
         double mNewPdsp;
         double mNewNbp;

        // Variale pour le filtrage recursif "moins"
         double mNewZPm;
         double mNewPdsm;
         double mNewNbm;

         double mPPrec;
         double mPNext;
         bool   mSelected;
};



template <class Type> class  cLoadedCP
{
    public :

        typedef  Type  tNum;
        typedef  typename El_CTypeTraits<tNum>::tBase  tNBase;

        //typedef  float  tNum;
        // typedef  double  tNBase;

        cLoadedCP(cFusionCarteProf<Type> &, const std::string & anId,const std::string & aFus,int aNum);
        const cXML_ParamNuage3DMaille & Nuage() {return mNuage;}

        cElTmp0Pile  CreatePile(const Pt2dr &) const;
        double   PdsLinear(const Pt2dr &) const;

        void  SetSz(const Pt2di & aSz);
        bool  ReLoad(const Box2dr & aBoxTer);
        const  cImage_Profondeur & IP() {return mIP;}
        const std::string & NameNuage() {return mNameNuage;}
        Im2D_U_INT1   ImCorrel() {return  mImCorrel;}

        std::string NameMM1P(const std::string aPref) const;
        std::string NameIm() const;
        Tiff_Im FileMM1P(const std::string aPref);
        double ZOfP(const Pt2di & aP) const {return mTImCP.get(aP);}

    private :
        double ToZAbs(double aZ) const
        {
           aZ = mIP.OrigineAlti() +  aZ * mIP.ResolutionAlti();
           return aZ;
        }

        int                       mNum;
        cFusionCarteProf<Type>  & mFCP;
        const cParamFusionMNT & mParam;
        const cParamAlgoFusionMNT & mPAlg;
        double                      mSeuilC;
        cInterfChantierNameManipulateur * mICNM;

        std::string   mFus;
        std::string   mNameIm;
        std::string   mNameNuage;
        cXML_ParamNuage3DMaille  mNuage;
        cImage_Profondeur        mIP;
        ElAffin2D                mAfM2CGlob;
        ElAffin2D                mAfC2MGlob;
        ElAffin2D                mAfM2CCur;
        ElAffin2D                mAfC2MCur;
        std::string              mDirNuage;

        std::string        mNameCP;
        Tiff_Im            mTifCP;
        Pt2di              mSzGlob;
        Box2di             mBoxImGlob;
        Pt2di              mSzCur;
        Box2di             mBoxImCur;
        Im2D<tNum,tNBase>  mImCP;
        TIm2D<tNum,tNBase> mTImCP;

        std::string        mNameMasq;
        Tiff_Im            mTifMasq;
        Im2D_Bits<1>       mImMasq;
        TIm2DBits<1>       mTImMasq;

        bool               mHasCorrel;
        std::string        mNameCorrel;
        Im2D_U_INT1        mImCorrel;
        TIm2D<U_INT1,INT>  mTImCorrel;
        bool               mZIsInv;
        bool               mQualImSaved;
        double             mPdsIm;

};

static const double MulCost = 1e3;
static bool DebugActif = false;

template <class Type> class cFusionCarteProf
{
     public :
        bool IsPBug(const Pt2di & aP)
        {
            //  return (aP==Pt2di(398,445)) ||  (aP==Pt2di(399,446));
            return ((mDecal+aP)==Pt2di(404,460));
        }
        int ToICost(double aCost) {return round_ni(MulCost * aCost);}
        double ToRCost(int    aCost) {return aCost/ MulCost;}

        typedef  Type  tNum;
        typedef  typename El_CTypeTraits<tNum>::tBase  tNBase;

      //=================================================================
      // Interface pour utiliser la prog dyn
      //=================================================================
         //-------- Pre-requis
            typedef  cElPilePrgD tArgCelTmp;
            typedef  cElPilePrgD tArgNappe;

         //-------- Pas pre-requis mais aide a la declaration
            typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
            typedef  cTplCelOptProgDyn<tArgCelTmp>  tCelOpt;

         //-------- Pre-requis
           void DoConnexion
           (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*Input,int aInZMin,int aInZMax,
                  tCelOpt*Ouput,int aOutZMin,int aOutZMax
           );
           void GlobInitDir(cProg2DOptimiser<cFusionCarteProf> &);

          // -- NOT a requirement, just here, an help for implementation
          // of DoConnexion
                void DoConexTrans
                     (
                                  tCelOpt & aCelIn,
                                  tCelOpt & aCelOut,
                                  ePrgSens aSens
                     );

      //=================================================================

      // typedef  float  tNum;
      // typedef  double  tNBase;

          cFusionCarteProf(const cResultSubstAndStdGetFile<cParamFusionMNT>  & aP,const std::string & aCom);
          const cParamFusionMNT & Param() {return mParam;}
          cInterfChantierNameManipulateur *ICNM() {return mICNM;}


         cLoadedCP<Type> *  VCLOfName(const std::string &) ;
         bool InParal() const;
     private :

           double ToZSauv(double aZ) const;

          void DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut);
          void DoOneFusion(const std::string &);
          void DoCalc();

          cParamFusionMNT mParam;
          std::string     mCom;
          const cSpecAlgoFMNT & mSpecA;
          const cFMNT_ProgDyn  * mFPrgD;
          const cFMNT_GesNoVal * mFNoVal;
          //double                 mSigmaP;
          //double                 mSigmaZ;
          cInterfChantierNameManipulateur * mICNM;
          std::vector<std::string>          mGenRes;
          // std::string                             mNameNuageIn;
          bool                                    mCalledBySubP;
          bool                                    mInParal;
          bool                                    mInSerialP;
          bool                                    mBySubPr;
          bool                                    mThrowSubPr;
          std::vector<cLoadedCP<Type> *>          mVC;
          std::vector<cLoadedCP<Type> *>          mVCL;
          cXML_ParamNuage3DMaille                 mNuage;
          cImage_Profondeur *                     mIP;
          ElAffin2D                               mAfM2CGlob;
          ElAffin2D                               mAfC2MGlob;
          ElAffin2D                               mAfM2CCur;
          ElAffin2D                               mAfC2MCur;
          Pt2di                                   mSzGlob;
          Pt2di                                   mSzCur;
          std::string                             mNameTif;
          std::string                             mNameMasq;
          std::string                             mNameCorrel;
          std::string                             mNameCptr;
          bool                                    mZIsInv;
          std::list<std::string>                  mListCom;
          double                                  mResolPlaniReel; // La vrai resol
          double                                  mResolPlaniEquiAlt; // Celle qui est equivalente a l'alti
          Pt2di                                   mDecal;
};



     /**********************************************************************************/
     /*                                                                                */
     /*                       cTmpPile, cElPile                                        */
     /*                                                                                */
     /**********************************************************************************/


bool operator < (const cElTmp0Pile & aP1, const cElTmp0Pile & aP2) {return aP1.Z() < aP2.Z();}
class cCmpPdsPile
{
   public :
     bool operator()(const cTmpPile & aP1,const cTmpPile & aP2)
     {
          return aP1.mPds0 > aP2.mPds0;
     }
};


/*
mK ;  // indexe initial
mCpteur  ; //  rempli le cpt de cElTmp0Pile; no de voisins avec une fonctionne de ponder au dessous de seuil 
mPInit ; // poids initial
mPds0;

*/

cTmpPile::cTmpPile(int aK,float aZ,float aPds,const cLoadedCP<float> * aLCP) :
    mK     (aK),
    mCpteur(0),
    mZInit (aZ),
    mPInit (aPds),
    mZP0   (aZ * aPds),
    mPds0  (aPds),
    mNb0   (1.0),
    mLCP   (aLCP),
    mPPrec (-1),
    mPNext (-1),
    mSelected (false)
{
}


void cTmpPile::SetPPrec(cTmpPile & aPrec,float aExpFact)
{
    ELISE_ASSERT(mZInit>=aPrec.mZInit,"Ordre coherence in cTmpPile::SetPPrec");


    double aPds = (aPrec.mZInit-mZInit)/aExpFact;
    aPds = exp(aPds);
    mPPrec  = aPds;
    aPrec.mPNext = aPds;
}

void FiltrageAllerEtRetour(std::vector<cTmpPile> & aVTmp)
{
     int aNb = (int)aVTmp.size();

   // Propagation de la gauche vers la droite qui seront stockes dans mNewPdp etc..
         // Initialistion de la gauche
     aVTmp[0].mNewPdsp = aVTmp[0].mPds0;
     aVTmp[0].mNewZPp  = aVTmp[0].mZP0;
     aVTmp[0].mNewNbp  = aVTmp[0].mNb0;

          // Propagattion
     for (int aK=1 ; aK<int(aVTmp.size()) ; aK++)
     {
            //=>Pds0_current_cell + (NewPds_prev_cell * PPrec_current_cell)
            //=>Zpond_current_cell + (Zpond_prev_cell * PPrec_current_cell)
                //PPrec = exp(Z_prev_cell - Z_curr_cell)/( Sigma0 * Resol_Plani * sqrt(2) ) <=> exp(|az|)

          aVTmp[aK].mNewPdsp = aVTmp[aK].mPds0 + aVTmp[aK-1].mNewPdsp * aVTmp[aK].mPPrec;
          aVTmp[aK].mNewZPp  = aVTmp[aK].mZP0  + aVTmp[aK-1].mNewZPp  * aVTmp[aK].mPPrec;
          aVTmp[aK].mNewNbp  = aVTmp[aK].mNb0  + aVTmp[aK-1].mNewNbp  * aVTmp[aK].mPPrec;
     }


   // Propagation de droite  a gauche da,s Pdsm etc ...
     aVTmp[aNb-1].mNewPdsm = aVTmp[aNb-1].mPds0;
     aVTmp[aNb-1].mNewZPm = aVTmp[aNb-1].mZP0;
     aVTmp[aNb-1].mNewNbm = aVTmp[aNb-1].mNb0;
     for (int aK=(int)(aVTmp.size() - 2); aK>=0 ; aK--)
     {
          aVTmp[aK].mNewPdsm = aVTmp[aK].mPds0 + aVTmp[aK+1].mNewPdsm * aVTmp[aK].mPNext;
          aVTmp[aK].mNewZPm  = aVTmp[aK].mZP0  + aVTmp[aK+1].mNewZPm  * aVTmp[aK].mPNext;
          aVTmp[aK].mNewNbm  = aVTmp[aK].mNb0  + aVTmp[aK+1].mNewNbm  * aVTmp[aK].mPNext;
     }

     // Memorisation dans mZP0 etc.. du resultat (droite + gauche - VCentrale) , VCentrale a ete compte deux fois
     for (int aK=0 ; aK<int(aVTmp.size()) ; aK++)
     {
          aVTmp[aK].mZP0  = (aVTmp[aK].mNewZPp  + aVTmp[aK].mNewZPm  - aVTmp[aK].mZP0) / aVTmp.size();
          aVTmp[aK].mPds0 = (aVTmp[aK].mNewPdsp + aVTmp[aK].mNewPdsm - aVTmp[aK].mPds0) / aVTmp.size();
          aVTmp[aK].mNb0  = (aVTmp[aK].mNewNbp  + aVTmp[aK].mNewNbm  - aVTmp[aK].mNb0) / aVTmp.size();
     }
}

Pt3dr VerifExp(const std::vector<cElTmp0Pile> & aVPile,cElTmp0Pile aP0,float aPixFact)
{
    Pt3dr aRes (0,0,0);
    for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
    {
       double aPds  = exp(-ElAbs(aP0.Z()-aVPile[aK].Z())/aPixFact);
       aRes = aRes + Pt3dr(aVPile[aK].Z()*aVPile[aK].P(),aVPile[aK].P(),1.0) * aPds;
    }
    return aRes / aVPile.size();
}

Pt3dr  PdsSol(const std::vector<cElTmp0Pile> & aVPile,const cElTmp0Pile & aP0,float aPixFact)
{
    Pt3dr aRes (0,0,0);
    for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
    {
       double aPds  = exp(-0.5*ElSquare((aP0.Z()-aVPile[aK].Z())/aPixFact)   );
       Pt3dr  aQ
              (
                     aVPile[aK].Z()*aVPile[aK].P(),
                     aVPile[aK].P(),
                     1.0
              );
       aRes = aRes  + aQ * aPds;
    }
    return aRes / aVPile.size();

}

bool IsMaxLoc(const std::vector<cTmpPile> & aVPile,int aK,float anEc,int aStep,int aLim)
{
   cTmpPile aP0 = aVPile[aK];
   for (int aK1=aK+aStep ; (aK1!=aLim) ; aK1+=aStep)
   {
       cTmpPile aP1 = aVPile[aK1];
       if (ElAbs(aP0.mZInit-aP1.mZInit) > anEc) return true;
       if (aP0.mPds0 < aP1.mPds0) return false;
   }
   return true;
}

bool IsMaxLoc(const std::vector<cTmpPile> & aVPile,int aK,float anEc)
{
    return    IsMaxLoc(aVPile,aK,anEc,-1,-1)
           && IsMaxLoc(aVPile, aK, anEc, 1, (int)aVPile.size());
}

double PdsCptr(double aDZ,double aSeuil)
{
    double aRes = aDZ/aSeuil;
    if (aRes<0.5) return 1;
    if (aRes>1  ) return 0;
    return  2* (1-aRes);

    // return aDZ < aSeuil;
    // return ElMax(0.0,1-ElSquare(aDZ/aSeuil));
}

void IncreCptr(const std::vector<cTmpPile> & aVPile,cTmpPile & aPile,int aK0,float aSeuil,int aStep,int aLim)
{
    while (1)
    {
        if (aK0 == aLim) return;
        double aP = PdsCptr(ElAbs(aVPile[aK0].mZInit-aPile.mZInit),aSeuil);
        if (aP<=0) return;
        aPile.mCpteur  += aP;
        aK0 += aStep;
    }
}

std::vector<cElTmp0Pile>  ComputeExpEv(const std::vector<cElTmp0Pile> & aVPile,double aResolPlani,const cSpecAlgoFMNT & aFEv)
{

  static int aCpt=0; aCpt++;
//   Bug= (aCpt ==3025);


   float aZFact = (aFEv.SigmaPds()*aResolPlani) * sqrt(2.0);
   double aSzML = aFEv.SeuilMaxLoc() * aResolPlani;
   double aSCpt = aFEv.SeuilCptOk() * aResolPlani;


   std::vector<cTmpPile> aTmp;

   for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
   {
        aTmp.push_back(cTmpPile(aK,aVPile[aK].Z(),aVPile[aK].P(),aVPile[aK].LCP()));
        if (aK>0)
        {
            aTmp[aK].SetPPrec(aTmp[aK-1],aZFact);
        }

   }

// On fait deux filtrage exponentiel ce qui fait + ou - un filtrage gaussien
   FiltrageAllerEtRetour(aTmp);

   if (false) // (LocBug)
   {
       for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
       {
            Pt3dr aPTest = VerifExp(aVPile,aVPile[aK],aZFact);
            Pt3dr aDif = aPTest - Pt3dr(aTmp[aK].mZP0,aTmp[aK].mPds0,aTmp[aK].mNb0 );
            // if (LocBug) std::cout << euclid(aDif) << " " << aPTest  << "\n";
            if (euclid (aDif)> 1e-3)
            {
                 std::cout << euclid (aDif) << aPTest << " " <<  aTmp[aK].mZInit << " " << aTmp[aK].mZP0 << " " << aTmp[aK].mPds0 ;
                 if (aVPile[aK].LCP()) std::cout << " N=" << (aVPile[aK].LCP()) ;
                 std::cout  << "\n";
                 ELISE_ASSERT(false,"Coher in ComputeExpEv");
            }
       }
       // getchar();
   }
   FiltrageAllerEtRetour(aTmp);
   for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
   {
        if (false) // (LocBug)
        {
                // Pt3dr aPSOL =  PdsSol(aVPile,aVPile[aK],aZFact);
                std::cout << " OUT " <<  aTmp[aK].mZInit << " P0 " << aTmp[aK].mPds0  << " PI " << aTmp[aK].mPInit
                          << " PIm " << (aTmp[aK].mPds0/aTmp[aK].mNb0)
                          << " MaxLoc " << IsMaxLoc(aTmp,aK,aSzML)  << " SML " << aSzML;
                 if (aVPile[aK].LCP()) std::cout << " N=" << (aVPile[aK].LCP()) ;
                 std::cout << "\n";
                              //  << " " << aTmp[aK].mNb0 << " SOL " <<  (aPSOL.x/aPSOL.y) << " " << aPSOL.y  << " " << aPSOL.z << "\n";
        }
   }

   // On selectionne les elements qui sont maxima local du Pds0
   std::vector<cTmpPile> aResTmp;
   for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
   {
       if (IsMaxLoc(aTmp,aK,aSzML))
       {
           aResTmp.push_back (aTmp[aK]);
       }
   }

   // Si necessaire on ne garde que les NBMaxMaxLoc meilleurs element (selon Pds0)
   cCmpPdsPile aCmp;
   std::sort(aResTmp.begin(),aResTmp.end(),aCmp);
   while (int(aResTmp.size()) > aFEv.NBMaxMaxLoc().Val())  aResTmp.pop_back();


   // Compte pour les element selectionne les voisin avec un fonction de ponderation
   // qui vaut 1 jusqu'a Seuil/2,  0 au de la Seuil, et raccord continue entre les deux
   for (int aI=0 ; aI<int(aResTmp.size()) ; aI++)
   {
         IncreCptr(aTmp,aResTmp[aI],aResTmp[aI].mK  ,aSCpt,-1,            -1);
         // IncreCptr(aTmp,aResTmp[aI],aResTmp[aI].mK+1,aSCpt, 1,aResTmp.size());
         IncreCptr(aTmp, aResTmp[aI], aResTmp[aI].mK + 1, aSCpt, 1, (int)aTmp.size());
   }



  // Export sous forme d'un pile minimaliste
   std::vector<cElTmp0Pile> aRes;
   for (int aK=0 ; aK<int(aResTmp.size()) ; aK++)
   {
       cElTmp0Pile  aPil (aResTmp[aK].ZMoy(),aResTmp[aK].mPds0,aResTmp[aK].mCpteur,aResTmp[aK].mLCP);
       aRes.push_back (aPil);
   }


   return aRes;
}







/**********************************************************************/
/*                                                                    */
/*                      cLoadedCP                                     */
/*                                                                    */
/**********************************************************************/

//template <class Type> Show(const Type &




template <class Type>  cLoadedCP<Type>::cLoadedCP(cFusionCarteProf<Type> & aFCP, const std::string & anId,const std::string & aFus,int aNum) :
  mNum     (aNum),
  mFCP     ((aFCP)),
  mParam   ((aFCP.Param())),
  mPAlg    ((mParam.ParamAlgoFusionMNT())),
  mSeuilC  ((mPAlg.FMNTSeuilCorrel())),
  mICNM    ((aFCP.ICNM())),

  // mFus         ((std::cout << "ZZZZZZyyyy " << anId << " " << aFus << "\n", aFus)),
  mFus            ( aFus),
  // mNameIm      ((StdPrefix(mFus).substr(6,std::string::npos)),
  // mNameIm      (StdPrefix(mFus)),
  mNameIm      (mICNM->Assoc1To1(mParam.KeyNuage2Im().Val(),mFus,true)),
  mNameNuage  (mICNM->Dir()+mICNM->Assoc1To2(mParam.KeyNuage(),anId,aFus,true)),
  mNuage      (StdGetObjFromFile<cXML_ParamNuage3DMaille>
                 (
                     mNameNuage,
                     StdGetFileXMLSpec("SuperposImage.xml"),
                     "XML_ParamNuage3DMaille",
                     "XML_ParamNuage3DMaille"
                 )
              ),
  mIP         (mNuage.PN3M_Nuage().Image_Profondeur().Val()),
  mAfM2CGlob  (Xml2EL(mNuage.Orientation().OrIntImaM2C())), // RPCNuage
  mAfC2MGlob  (mAfM2CGlob.inv()),
  mDirNuage   (DirOfFile(mNameNuage)),

  mNameCP  (mDirNuage+mIP.Image()),
  mTifCP   (Tiff_Im::StdConv(mNameCP)),
  mSzGlob    (mTifCP.sz()),
  mBoxImGlob (Pt2di(0,0),mSzGlob),
  mImCP    (1,1),
  mTImCP   (mImCP),

  mNameMasq (mDirNuage+mIP.Masq()),
  mTifMasq  (Tiff_Im::StdConv(mNameMasq)),
  mImMasq   (1,1),
  mTImMasq   (mImMasq),
  mHasCorrel (mIP.Correl().IsInit()),
  mNameCorrel (mHasCorrel ? mDirNuage+mIP.Correl().Val() : ""),
  mImCorrel   (1,1),
  mTImCorrel  (mImCorrel),
  mZIsInv     (false),
  mQualImSaved (false),
  mPdsIm       (1.0)
{


  if (mParam.KeyPdsNuage().IsInit())
  {
      std::string aNamePdsIm = mICNM->Assoc1To1(mParam.KeyPdsNuage().Val(),aFus,true);
      FromString(mPdsIm,aNamePdsIm);
  }

  // std::cout << "PDSssIM " << mPdsIm << " for " << anId << "\n";

   if (mNuage.ModeFaisceauxImage().IsInit())
      mZIsInv = mNuage.ModeFaisceauxImage().Val().ZIsInverse();
}

template <class Type> void  cLoadedCP<Type>::SetSz(const Pt2di & aSz)
{
   mSzCur = aSz;

   mImCP.Resize(mSzCur);
   mTImCP = TIm2D<tNum,tNBase>(mImCP);

   mImMasq = Im2D_Bits<1>(mSzCur.x,mSzCur.y);
   mTImMasq = TIm2DBits<1>(mImMasq);

   if (mHasCorrel)
   {
       mImCorrel.Resize(mSzCur);
       mTImCorrel = Im2D<U_INT1,INT>(mImCorrel);
   }

}
template <class Type> std::string cLoadedCP<Type>::NameMM1P(const std::string aPref) const
{
    return  mParam.WorkDirPFM().Val() + aPref + "-" +  mNameIm + ".tif";
}
template <class Type> std::string cLoadedCP<Type>::NameIm() const { return    mNameIm ; }

template <class Type> Tiff_Im cLoadedCP<Type>::FileMM1P(const std::string aPref)
{
    return Tiff_Im::StdConv(NameMM1P(aPref));
}

template <class Type> bool  cLoadedCP<Type>::ReLoad(const Box2dr & aBoxTer)
{
   mBoxImCur =  R2I(aBoxTer.BoxImage(mAfM2CGlob));
   if (InterVide(mBoxImCur,mBoxImGlob))
   {
       SetSz(Pt2di(1,1));
       return false;
   }

   mBoxImCur = Inf(mBoxImCur,mBoxImGlob);
   SetSz(mBoxImCur.sz());

   mAfM2CCur =  ElAffin2D::trans(-Pt2dr(mBoxImCur._p0)) * mAfM2CGlob;
   mAfC2MCur = mAfM2CCur.inv();

   ELISE_COPY(mImCP.all_pts(),trans(mTifCP.in(),mBoxImCur._p0),mImCP.out());
   ELISE_COPY(mImMasq.all_pts(),trans(mTifMasq.in()!=0,mBoxImCur._p0),mImMasq.out());
   if (mHasCorrel)
   {
       const cSectionScoreQualite  * aSSQ  = mParam.SectionScoreQualite().PtrVal();
       if (aSSQ)
       {
           if (aSSQ->ScoreMM1P().IsInit())
           {
               const cScoreMM1P & aSM1P = aSSQ->ScoreMM1P().Val();
               // std::string aName = StdPrefix(mFus).substr(6,std::string::npos);

               Fonc_Num aF1 =  FileMM1P("Score-AR").in_proj();
               double aP1 = aSSQ->PdsAR().Val();

               Fonc_Num aF2 = FileMM1P("Dist").in_proj();
               aF2 = aF2 /  aSSQ->AmplImDistor().Val();
               aF2  = 255 / (1.0 + aF2 /aSSQ->SeuilDist().Val());
               double aP2 = aSSQ->PdsDistor().Val();

               Fonc_Num aF3 = FileMM1P("Mask").in_proj();
               double aD = aSSQ->SeuilDisBord().Val();
               aF3 = extinc_32(aF3,aD);
               aF3 = 255 * ( Min(1.0,aF3/aD));
               double aP3 = aSSQ->PdsDistBord().Val();

               Fonc_Num aFCor = (aF1 * aP1 + aF2*aP2 + aF3*aP3) / (aP1 + aP2 + aP3) ;

               ELISE_COPY(mImCorrel.all_pts(),trans(aFCor,mBoxImCur._p0),mImCorrel.out());


               if ((!mQualImSaved) && aSM1P.MakeFileResult().Val() && (!mFCP.InParal()))
               {
                      mQualImSaved = true;

                      Tiff_Im::Create8BFromFonc
                      (
                          NameMM1P("Quality"),
                          FileMM1P("Mask").sz(),
                          aFCor
                      );
               }
 // getchar();
               // Tiff
           }
       }
       else
       {
           Tiff_Im aTifCorrel = Tiff_Im::StdConv(mNameCorrel);
           ELISE_COPY(mImCorrel.all_pts(),trans(aTifCorrel.in(),mBoxImCur._p0),mImCorrel.out());
       }
   }
   return true;
}

template <class Type> double  cLoadedCP<Type>::PdsLinear(const Pt2dr & aPTer) const
{
   Pt2dr aPIm = mAfM2CCur(aPTer);
   double aPds = mTImMasq.get(round_ni(aPIm),0) * mPdsIm ;
   if (aPds > 0)
   {
       if (mHasCorrel)
       {
           aPds = mTImCorrel.getprojR(aPIm);
           aPds = aPds / 256.0;
/*
           aPds = (aPds -128.0) / 128.0;
           aPds = (aPds-mSeuilC)/(1.0-mSeuilC);
*/
       }
       return ElMin(1.0,ElMax(0.0,aPds));
   }
   return 0;
}


template <class Type> cElTmp0Pile  cLoadedCP<Type>::CreatePile(const Pt2dr & aPTer) const
{
   Pt2dr aPIm = mAfM2CCur(aPTer);
   double aPds = PdsLinear(aPTer);
   //double aPds = mTImMasq.get(round_ni(aPIm),0);
   double aZ = 0;

   if (aPds > 0)
   {
      aPds = pow(aPds,mPAlg.FMNTGammaCorrel());
   }
   if (aPds > 0)
   {
       aZ = ToZAbs(mTImCP.getprojR(aPIm));
   }
   return cElTmp0Pile(aZ,aPds,-1,this);
}



/**********************************************************************/
/*                                                                    */
/*                      cFusionCarteProf                              */
/*                                                                    */
/**********************************************************************/


template <class Type> cLoadedCP<Type> *  cFusionCarteProf<Type>::VCLOfName(const std::string & aNameNuage)
{
    for (int aK=0 ; aK<int(mVCL.size()) ; aK++)
    {
/*
        std::cout << mVCL[aK]->NameNuage()  << "\n";
*/
        if (mVCL[aK]->NameNuage() == aNameNuage)
           return mVCL[aK];
    }

   ELISE_ASSERT(false,"::VCLOfName");
   return 0;
}


template <class Type> void cFusionCarteProf<Type>::DoOneFusion(const std::string & anId)
{
    std::string aNameNuage =  mICNM->Assoc1To1(mParam.KeyResult(),anId,true);
    if (mParam.KeyResultIsLoc().Val())
       aNameNuage = mICNM->Dir() + aNameNuage;

    mNameTif = StdPrefix(aNameNuage)+ "_Prof.tif";
    // std::cout << anId  << "=> " << mNameTif<< "\n";
    mNameMasq = StdPrefix(aNameNuage)+ "_Masq.tif";
    mNameCorrel = StdPrefix(aNameNuage)+ "_Correl.tif";
    mNameCptr = StdPrefix(aNameNuage)+ "_Cptr.tif";


    std::vector<std::string> aStrFus = GetStrFromGenStrRel(mICNM,mParam.GenereInput(),anId);

    if (aStrFus.size() == 0)
    {
        std::cout << "FOR ID = " << anId  << "\n";
        ELISE_ASSERT(false,"No data in DoOneFusion");
    }

	//load all depth maps
    for (int aK=0 ; aK<int(aStrFus.size()) ; aK++)
    {
          mVC.push_back(new cLoadedCP<Type>(*this,anId,aStrFus[aK],aK));
    }


    if (mParam.ModeleNuageResult().IsInit())
    {
       std::string aNameNuageIn =   mICNM->Assoc1To1(mParam.ModeleNuageResult().Val(),anId,true);


       if ( ELISE_fp::exist_file(mICNM->Dir() +aNameNuageIn))
       // if ( ! ELISE_fp::exist_file(aNameNuageIn))
          aNameNuageIn =  mICNM->Dir() + aNameNuageIn;


       mNuage = StdGetObjFromFile<cXML_ParamNuage3DMaille>
                (
                     //mParam.ModeleNuageResult().Val(),
                     aNameNuageIn,
                     StdGetFileXMLSpec("SuperposImage.xml"),
                     "XML_ParamNuage3DMaille",
                     "XML_ParamNuage3DMaille"
                );
    }
    else
    {
         mNuage = mVC[0]->Nuage();
         double aSomResolAlti = 0;
         double aSomOriAlti = 0;

         std::vector<double> aVR;
         Pt2dr aP0(1e30,1e30);
         Pt2dr aP1(-1e30,-1e30);
         for (int aK=0 ; aK<int(mVC.size()) ; aK++)
         {
              double aResol = mVC[aK]->IP().ResolutionAlti();
              double anOri = mVC[aK]->IP().OrigineAlti();

              aSomResolAlti += aResol;
              aSomOriAlti += anOri;

              aVR.push_back( ResolOfNu(mVC[aK]->Nuage()));
              Box2dr aBox = BoxTerOfNu(mVC[aK]->Nuage());
              aP0.SetInf(aBox._p0);
              aP1.SetSup(aBox._p1);
         }

         double aResol = MedianeSup(aVR);
         Pt2dr aSz = aP1 - aP0;
         Pt2di aNbPix = round_up(aSz/aResol);

         //    (aP0.x, aP1.y)  +  (aP.x/R  - aP.y/R)
         ElAffin2D aAfC2M(Pt2dr(aP0.x,aP1.y),Pt2dr(aResol,0),Pt2dr(0,-aResol));
         mNuage.Orientation().OrIntImaM2C().SetVal(El2Xml(aAfC2M.inv())); // RPCNuage
         mNuage.NbPixel() = aNbPix;

         aSomResolAlti /=  mVC.size();
         aSomOriAlti /=  mVC.size();
         // std::cout << "MOYYYY " << aSomOriAlti  << " " << aSomResolAlti << "\n";
         // std::cout << "AAAAAAAAAkkKKKKK " <<  mVC[0]->NameNuage()  << "\n"; getchar();

         mNuage.Image_Profondeur().Val().ResolutionAlti() = aSomResolAlti;
         mNuage.Image_Profondeur().Val().OrigineAlti() = aSomOriAlti;

        // Creation du TFW
        {
            ElAffin2D aAfM2C = Xml2EL(mNuage.Orientation().OrIntImaM2C()); // RPCNuage
            GenTFW(aAfM2C.inv(),StdPrefix(mNameTif) + ".tfw");
        }
    }



   mZIsInv = false;
   if (mNuage.ModeFaisceauxImage().IsInit())
      mZIsInv = mNuage.ModeFaisceauxImage().Val().ZIsInverse();

    mIP = &(mNuage.Image_Profondeur().Val());
    mIP->Image() = NameWithoutDir(mNameTif) ;
    mIP->Masq() =  NameWithoutDir(mNameMasq);
    mIP->Correl() =  NameWithoutDir(mNameCorrel);

    mAfM2CGlob  = Xml2EL(mNuage.Orientation().OrIntImaM2C()); // RPCNuage
    mAfC2MGlob = mAfM2CGlob.inv();
    mSzGlob = mNuage.NbPixel();

    if (! mCalledBySubP)
    {
       mNuage.Image_Profondeur().Val().Masq() = NameWithoutDir(mNameMasq);
       mNuage.Image_Profondeur().Val().Image() = NameWithoutDir(mNameTif);
       mNuage.Image_Profondeur().Val().Correl() = NameWithoutDir(mNameCorrel);


// std::cout << "AAAAAAAAAAAA " << mNameNuageIn << "\n";
       // MakeFileXML(mNuage,mNameNuageIn);

       bool IsModified;
       Im2D<tNum,tNBase> aITest(1,1);
       Tiff_Im::CreateIfNeeded
       (
              IsModified,
              mNameTif,
              mSzGlob,
              aITest.TypeEl(),
              Tiff_Im::No_Compr,
              Tiff_Im::BlackIsZero
       );

       Tiff_Im::CreateIfNeeded
       (
               IsModified,
               mNameCptr,
               mSzGlob,
               GenIm::u_int1,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero
       );


       Tiff_Im::CreateIfNeeded
       (
               IsModified,
               mNameCorrel,
               mSzGlob,
               GenIm::u_int1,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero
       );

       Tiff_Im::CreateIfNeeded
       (
               IsModified,
               mNameMasq,
               mSzGlob,
               GenIm::bits1_msbf,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero
       );
       MakeFileXML(mNuage,aNameNuage);
    }


    if (!mParam.BoxTest().IsInit())
    {
          cDecoupageInterv2D aDecoup = cDecoupageInterv2D::SimpleDec
                                (
                                     mSzGlob,
                                     mParam.SzDalles().Val(),
                                     mParam.RecouvrtDalles().Val()
                                );

         for (int aKI=0 ; aKI<aDecoup.NbInterv() ; aKI++)
         {

             if (mBySubPr && (!mCalledBySubP))
             {
                  std::string aNewCom =   mCom
                                  + std::string(" InterneCalledByProcess=true")
                                  + std::string(" InterneSingleImage=") +  anId
                                  + std::string(" InterneSingleBox=") + ToString(aKI);
                  mListCom.push_back(aNewCom);

                  if (mParam.ShowCom().Val()) 
                  {
                      std::cout << aNewCom << "\n";
                  }
             }
             else
             {
                 if ((!mCalledBySubP) || (mParam.InterneSingleBox().Val()==aKI))
                 {
                    DoOneBloc
                    (
                        aDecoup.NbInterv()-aKI,
                        aDecoup.KthIntervIn(aKI),
                        aDecoup.KthIntervOut(aKI)
                    );
                 }
             }
         }
   }
   else
   {
        Box2di aBox = mParam.BoxTest().Val();
        DoOneBloc(0,aBox,aBox);
   }

   DeleteAndClear(mVC);
}

template <class Type> bool cFusionCarteProf<Type>::InParal() const {return mInParal;}

template <class Type> double cFusionCarteProf<Type>::ToZSauv(double aZ) const
{
   // if (mZIsInv) aZ = 1/aZ;
   return  (aZ -mIP->OrigineAlti()) / mIP->ResolutionAlti();
}
// aZ = mIP.OrigineAlti() +  aZ * mIP.ResolutionAlti();


template <class Type> void cFusionCarteProf<Type>::DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut)
{
   mDecal = aBoxIn._p0;
   ElTimer aChrono;
   bool ShowTime = false;
   std::cout << "RESTE " << aKB <<   " BLOCS \n";
   mAfM2CCur =  ElAffin2D::trans(-Pt2dr(aBoxIn._p0)) * mAfM2CGlob ;
   mAfC2MCur = mAfM2CCur.inv();


   mResolPlaniReel = (euclid(mAfC2MCur.I10()) + euclid(mAfC2MCur.I01()))/2.0;

//  std::cout << "mResolPlaniReelmResolPlaniReel " << mResolPlaniReel << "\n";
   mResolPlaniEquiAlt = mResolPlaniReel * mNuage.RatioResolAltiPlani().Val();
/*
   mRatioPlaniAlti = mResolPlani;
   if (mNuage.ModeFaisceauxImage().IsInit())
   {
       if (mZIsInv)
       {
       }
   }
*/


   // std::cout << "  SCALE " << mResolPlani << "\n";

   mSzCur = aBoxIn.sz();

   Im2D<tNum,tNBase>  aImFus(mSzCur.x,mSzCur.y);
   TIm2D<tNum,tNBase> aTImFus(aImFus);

   Im2D_Bits<1>       aImMasq(mSzCur.x,mSzCur.y);
   TIm2DBits<1>       aTImMasq(aImMasq);


   Im2D<U_INT1,INT>  aImCorrel(mSzCur.x,mSzCur.y);
   TIm2D<U_INT1,INT> aTImCorrel(aImCorrel);

   Im2D<U_INT1,INT>  aImCptr(mSzCur.x,mSzCur.y);
   TIm2D<U_INT1,INT> aTImCptr(aImCptr);


   Box2di aBoxInLoc(Pt2di(0,0),mSzCur);
   Box2dr aBoxTer = aBoxInLoc.BoxImage(mAfC2MCur);


   mVCL.clear();

   for (int aK=0 ; aK<int(mVC.size()) ; aK++)
   {
       bool  aReload = mVC[aK]->ReLoad(aBoxTer);
       // std::cout << "RELOAD " <<  mVC[aK]->NameNuage() << " " << aReload << "\n";
       if (aReload)
          mVCL.push_back(mVC[aK]);
   }
   if (ShowTime)
   {
      // std::cout << "  " << mIP->OrigineAlti()
   }

   TIm2D<INT2,INT>  aTIm0(mSzCur);
   TIm2D<INT2,INT>  aTImNb(mSzCur);
   std::list<std::vector<cElTmp0Pile> > aLVP;


/*
   if (mFPrgD)
   {
      aTIm0.Resize(mSzCur);
      aTImNb.Resize(mSzCur);

      Pt2di aQ0;
      for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
      {
           for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
           {
               int aNbOk =0;
               for (int aKI=0 ; aKI<int(mVCL.size()); aKI++)
               {
                   Pt2dr aT0 = mAfC2MCur(Pt2dr(aQ0));
                   double aPds =  mVCL[aKI]->PdsLinear(aT0);
                   if (aPds >0)
                   {
                      aNbOk ++;
                   }
               }
               aTIm0.oset(aQ0,-1); // Bug dans prog dyn si nappes vides
               aTImNb.oset(aQ0,aNbOk);
           }
      }
      aPrgD = new cProg2DOptimiser<cFusionCarteProf>(*this,aTIm0._the_im,aTImNb._the_im,0,1);
  }
*/

   if (ShowTime)
   {
   }


   // Creation de la structure de pile


   Pt2di aQ0;
   for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
   {
        std::vector<cElTmp0Pile> aPCel;
        for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
        {

            Pt2dr aT0 = mAfC2MCur(Pt2dr(aQ0));
            aPCel.clear();
            double aSomP=0;
            for (int aKI=0 ; aKI<int(mVCL.size()); aKI++)
            {
                cElTmp0Pile anEl = mVCL[aKI]->CreatePile(aT0);
                if (anEl.P()>0)
                {
                   aPCel.push_back(anEl);
                   aSomP += anEl.P();
                }
            }

            aTIm0.oset(aQ0,-1);
            aTImNb.oset(aQ0,0);
            bool isDebug = DebugActif && IsPBug(aQ0);
            if (aSomP>0)
            {
                for (int aKP=0 ; aKP < int(aPCel.size()) ; aKP++)
                {
                    // On veut pour des question de normalisation que la moyenne des poids sur
                    // x,y donne soit egale a 1
                    aPCel[aKP].SetPdsPile(aPCel[aKP].P() * (aPCel.size()/aSomP));
                }

                std::sort(aPCel.begin(),aPCel.end());
                std::vector<cElTmp0Pile> aVPile = ComputeExpEv(aPCel,mResolPlaniEquiAlt,mSpecA);

                aLVP.push_back(aVPile);
                aTImNb.oset(aQ0, (int)aVPile.size());

                if (isDebug)
                {
                   std::cout << "PBUG " << aQ0   <<  " In:" << aPCel.size()  << " Out:" << aVPile.size() << "\n";
                   for (int aK=0 ; aK<int( aPCel.size() ) ; aK++)
                   {
                        std::cout << "IN :: Corr " << aPCel[aK].P() << " " << ToZSauv( aPCel[aK].Z()) << "\n";
                   }
                   for (int aK=0 ; aK<int( aVPile.size() ) ; aK++)
                   {
                        std::cout << "Out :: Corr " << aVPile[aK].P() << " " << ToZSauv( aVPile[aK].Z()) << "\n";
                   }
                }
            }
        }
   }


   if (ShowTime)
      std::cout << " Init Cost time= " << aChrono.uval() << "\n";

   // Cas ou on fait de programmation dynamique
   if (1)
   {
       // 1- Remplir la nappe avec les cellules
       double aDefPds =   mFNoVal ?  mFNoVal->CostNoVal() :0.5  ;
       cElPilePrgD aPDef(0);
       cProg2DOptimiser<cFusionCarteProf>  * aPrgD = new cProg2DOptimiser<cFusionCarteProf>(*this,aTIm0._the_im,aTImNb._the_im,0,1); // 0,1 => Rab et Mul
       {
           cDynTplNappe3D<cTplCelNapPrgDyn<cElPilePrgD> > & aNap = aPrgD->Nappe();
           std::list<std::vector<cElTmp0Pile> >::const_iterator anIt =  aLVP.begin();

           for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
           {
                for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
                {
                     int aNb = aTImNb.get(aQ0);//the no. of depths/Z for the current cell
                     cTplCelNapPrgDyn<cElPilePrgD> * aTabP = aNap.Data()[aQ0.y][aQ0.x];
                     aTabP[-1].ArgAux()= aPDef;
                     aTabP[-1].SetOwnCost(ToICost(aDefPds));
                     bool isDebug = DebugActif && IsPBug(aQ0);
                     if (aNb)
                     {
                         ELISE_ASSERT(anIt!=aLVP.end(),"(1)Incoh in cFusionCarteProf");
                         const std::vector<cElTmp0Pile> & aPil = (*anIt);
                         for (int aKz=0 ; aKz<aNb ; aKz++)
                         {
                             const cElTmp0Pile& aPk = aPil[aKz];
                             aTabP[aKz].ArgAux() = cElPilePrgD(aPk.Z());
                             aTabP[aKz].SetOwnCost(ToICost(ElMax(0.0,ElMin(1.0,1.0-aPk.P()))));


if (0)
{
static double MaxP=-1;
if (aPk.P()>MaxP)
{
   MaxP=aPk.P();
   std::cout << "==== MaxP " << MaxP << "\n";
}
}

                             if (isDebug)
                             {
                                 std::cout << "Fill Pill, Kz " << aKz << " , Z "
                                           << aTabP[aKz].ArgAux().Z()
                                           <<  " CostInit " << ToICost(ElMax(0.0,ElMin(1.0,1.0-aPk.P())))
                                           <<  " PrgdCost=" << ToRCost(aTabP[aKz].OwnCost()) << "\n";
                             }
                         }
                         anIt++;
                     }
                }
           }
           ELISE_ASSERT(anIt==aLVP.end(),"(2) Incoh in cFusionCarteProf");
           // aLVP.clear();
       }

       aPrgD->DoOptim(mFPrgD->NbDir());
       std::cout << " Prg Dyn time= " << aChrono.uval()  << " Nb Dir " << mFPrgD->NbDir() << "\n";
       Im2D_INT2 aSol(mSzCur.x,mSzCur.y);
       INT2 ** aDSol = aSol.data();
       aPrgD->TranfereSol(aDSol);
       std::list<std::vector<cElTmp0Pile> >::const_iterator anIt =  aLVP.begin();
       for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
       {
            for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
            {
                int aZ = aDSol[aQ0.y][aQ0.x];
                tCelNap & aCol =  aPrgD->Nappe().Data()[aQ0.y][aQ0.x][aZ];

                bool isDebug = DebugActif && IsPBug(aQ0);
                if (isDebug)
                {
                    std::cout << "Sortie Kz " << aZ << " Z " <<  aCol.ArgAux().Z() << "\n";
                }

                if (aZ>=0)
                {

                   aTImFus.oset(aQ0,ToZSauv(aCol.ArgAux().Z()));
                   aTImMasq.oset(aQ0,1);
                   const cElTmp0Pile & aPz = (*anIt)[aZ];
                   aTImCorrel.oset(aQ0,ElMax(0,ElMin(255,(round_ni(aPz.P()*255)))));
                   aTImCptr.oset(aQ0,ElMax(0,ElMin(255,(round_ni(aPz.CPtr()*DynCptrFusDepthMap )))));

                }
                else
                {
                     aTImMasq.oset(aQ0,0);
                     aTImFus.oset(aQ0,0);
                     aTImCorrel.oset(aQ0,0);
                     aTImCptr.oset(aQ0,0);
                }

                int aNb = aTImNb.get(aQ0);
                if (aNb)
                   anIt++;
            }
       }
   }

   Im2D_Bits<1> aImMasq0 = aImMasq;
   if (mParam.ParamRegProf().IsInit())
   {
       aImMasq = FiltreDetecRegulProf(aImFus,aImMasq,mParam.ParamRegProf().Val());
   }

   if (ShowTime)
      std::cout << " Dow Opt time= " << aChrono.uval() << "\n";

   if (1)
   {
        Im2D_Bits<1>       aIm1(mSzCur.x,mSzCur.y,1);
        TIm2DBits<1>       aTIm1(aIm1);
        ComplKLipsParLBas (aImMasq, aIm1,aImFus,1.0);
   }
   // Une fois que l'on a detecte les zone a pb potentiel et leur a affecte une valeur par extrapolation
   // on les remet en NDG STD afin de boucher les trous d'orthos
   aImMasq = aImMasq0;



   ELISE_COPY
   (
       rectangle(aBoxOut._p0,aBoxOut._p1),
       trans(aImFus.in(),-aBoxIn._p0),
       Tiff_Im(mNameTif.c_str()).out()
   );

   ELISE_COPY
   (
       rectangle(aBoxOut._p0,aBoxOut._p1),
       trans(aImCorrel.in(),-aBoxIn._p0),
       Tiff_Im(mNameCorrel.c_str()).out()
   );

   ELISE_COPY
   (
       rectangle(aBoxOut._p0,aBoxOut._p1),
       trans(aImCptr.in(),-aBoxIn._p0),
       Tiff_Im(mNameCptr.c_str()).out()
   );




   ELISE_COPY
   (
       rectangle(aBoxOut._p0,aBoxOut._p1),
       trans(aImMasq.in(),-aBoxIn._p0),
       Tiff_Im(mNameMasq.c_str()).out()
   );

   // std::cout << "ENnnndd \n"; getchar();
}

template <class Type>   void cFusionCarteProf<Type>::DoConexTrans
                             (
                                  tCelOpt & aCelIn,
                                  tCelOpt & aCelOut,
                                  ePrgSens aSens
                             )
{
    aCelOut.UpdateCostOneArc(aCelIn,aSens,ToICost(mFNoVal?mFNoVal->Trans():0));
}

template <class Type>   void cFusionCarteProf<Type>::DoConnexion
                             (
                                    const Pt2di & aPIn, const Pt2di & aPOut,
                                    ePrgSens aSens,int aRab,int aMul,
                                    tCelOpt*aTabInput,int aInZMin,int aInZMax,
                                    tCelOpt*aTabOuput,int aOutZMin,int aOutZMax
                             )
{
    double aSig0 = mFPrgD->Sigma0();
    for (int aZIn=0 ; aZIn<aInZMax ; aZIn++)
    {
        tCelOpt & anInp = aTabInput[aZIn];
        const  cElPilePrgD & aPIn = anInp.ArgAux();
        for (int aZOut=0 ; aZOut<aOutZMax ; aZOut++)
        {
            tCelOpt & anOut = aTabOuput[aZOut];
            const  cElPilePrgD & aPOut = anOut.ArgAux();
            double aDZ = ElAbs(aPIn.Z()-aPOut.Z())/mResolPlaniEquiAlt;
            if ((mFNoVal==0) || (aDZ < mFNoVal->PenteMax()))
            {
            // Fonction concave, nulle et de derivee 1 en 0
                 double aCost = (sqrt(1+aDZ/aSig0)-1) * 2*aSig0 * mFPrgD->Regul();
                 anOut.UpdateCostOneArc(anInp,aSens,ToICost(aCost));
            }
        }
    }

    aTabOuput[-1].UpdateCostOneArc(aTabInput[-1],aSens,0);
    for (int aZIn=0 ; aZIn<aInZMax ; aZIn++)
    {
        DoConexTrans(aTabInput[aZIn],aTabOuput[-1],aSens);
    }
    for (int aZOut=0 ; aZOut<aOutZMax ; aZOut++)
    {
        DoConexTrans(aTabInput[-1],aTabOuput[aZOut],aSens);
    }
}


template <class Type> void cFusionCarteProf<Type>::GlobInitDir(cProg2DOptimiser<cFusionCarteProf> &)
{
    // std::cout << "===========DO ONE DIR \n";
}



template <class Type> void cFusionCarteProf<Type>::DoCalc()
{
   for (int aKS=0 ; aKS<int(mGenRes.size()) ; aKS++)
   {
       if ((!mCalledBySubP) || (mGenRes[aKS]==mParam.InterneSingleImage().Val()))
       {
          DoOneFusion(mGenRes[aKS]);
       }
   }

}



template <class Type> cFusionCarteProf<Type>::cFusionCarteProf
(
       const cResultSubstAndStdGetFile<cParamFusionMNT>  & aParam,
       const std::string &                                 aCom
)  :
     mParam        (*(aParam.mObj)),
     mCom          (aCom),
     mSpecA        (mParam.SpecAlgoFMNT()),
     mFPrgD        (mSpecA.FMNT_ProgDyn().PtrVal() ),
     mFNoVal       (mFPrgD ? mFPrgD->FMNT_GesNoVal().PtrVal() : 0),
     mICNM         (aParam.mICNM),
     mGenRes       (GetStrFromGenStr(mICNM,mParam.GenereRes())),
     mCalledBySubP  (mParam.InterneCalledByProcess().Val()),
     mInParal      (mParam.ParalMkF().IsInit()),
     mInSerialP    (mParam.ByProcess().Val()),
     mBySubPr      (mInParal || mInSerialP),
     mThrowSubPr   ((!mCalledBySubP) && mBySubPr)
{
/*
    if (mFByEv)
    {
          mSigmaP = mFByEv->SigmaPds();
          mSigmaZ =  mFByEv->SigmaZ().ValWithDef(mSigmaP);
    }
*/

    DoCalc();

    if (mThrowSubPr)
    {
        if (mInSerialP)
           cEl_GPAO::DoComInSerie(mListCom);
        else if (mInParal)
           cEl_GPAO::DoComInParal(mListCom,mParam.ParalMkF().Val());
    }
}




/*
*/


int FusionCarteProf_main(int argc,char ** argv)
{
   /*if ((argc>=2)  && (std::string(argv[1])==std::string("-help")))
   {
       cout << "Mandatory unnamed args : \n";
       cout << "   * string :: {XML file - see include/XML_MicMac/Fusion-MMByP-*.xml} \n";
       cout << "Named args : \n";
       cout << "  [Name=WorkDirPFM] string ::{}\n"
   }*/

   ELISE_ASSERT(argc>=2,"Not Enough args to FusionCarteProf.cpp");
   MMD_InitArgcArgv(argc,argv);

   Tiff_Im::SetDefTileFile(50000);

   std::string aCom0 = MMBin() + "mm3d "+ MakeStrFromArgcARgv(argc, argv, true); // true = aProtect
   // std::cout << aCom0 << "\n"; getchar();


   cResultSubstAndStdGetFile<cParamFusionMNT> aP2
                                           (
                                              argc-2,argv+2,
                                              argv[1],
                                              StdGetFileXMLSpec("SuperposImage.xml"),
                                              "ParamFusionMNT",
                                              "ParamFusionMNT",
                                              "WorkDirPFM",
                                              "FileChantierNameDescripteur"
                                           );

   cFusionCarteProf<float>  aFCP(aP2,aCom0);
   return 0;
}

/***************************************************************************/

class cSimpleFusionCarte
{
     public :
         cSimpleFusionCarte(int argc,char ** argv);

     private :
          std::string         mFullName;
          cElemAppliSetFile   mEASF;
          const std::vector<std::string> * mVNames;
          std::string         mNameTarget;
          std::string         mNameOut;

};


cSimpleFusionCarte::cSimpleFusionCarte(int argc,char ** argv) :
     mNameOut ("Fusion.xml")
{
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(mFullName,"Full name (Dir+Pat)", eSAM_IsPatFile),
           LArgMain() << EAM(mNameTarget,"TargetGeom",true,"Name of Targeted geometry, Def computed by agregation of input")
                      << EAM(mNameOut,"Out",true,"Result, Def=Fusion.xml")

    );

    if (MMVisualMode) return;

    mEASF.Init(mFullName);

    std::string aCom =       MM3dBinFile("MergeDepthMap")
                          +  XML_MM_File("Fusion-Basic.xml ")
                          +  " WorkDirPFM="    +  mEASF.mDir
                          +  " +PatternInput=" +  QUOTE(mEASF.mPat)
                          +  " +NameOutput="   +  mNameOut
                       ;


    if (EAMIsInit(&mNameTarget))
    {
        aCom =     aCom +   " +WithTarget=true +NameTarget=" + mNameTarget ;
    }

    std::cout << "COM= " << aCom << "\n";
    System(aCom);
}





int SimpleFusionCarte_main(int argc,char ** argv)
{
    cSimpleFusionCarte anAppli(argc,argv);

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
