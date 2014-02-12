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

using namespace NS_ParamMICMAC;
using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;


// template <class Type> cFil

/*********************************************************************/
/*                                                                   */
/*                       cFusionCarteProf                            */
/*                                                                   */
/*********************************************************************/




static Pt2di LocPBug(234,55);
static bool  LocBug=false;

template <class Type> class  cLoadedCP;        // Represente une carte de profondeur chargee
template <class Type> class cFusionCarteProf;  // Represente l'appli globale

class cElPile;

class fPPile;  // Functeur : renvoie le poids
class fZPile;  // Functer : renvoie le Z
    //=================================

class cElPile
{
    public :
        cElPile (double aZ,double aPds,double aPOwn= -1,const std::string * aName =  0) :
           mZ  (aZ),
           mP  (aPds),
           mPOwn (aPOwn),
           mName (aName)
        {
        }

        void InitTmp(const cTplCelNapPrgDyn<cElPile> & aCel)
        {
            *this = aCel.ArgAux();
        }
        cElPile() :
           mZ(0), 
           mP(-1) 
        {
        }

        const float & Z() const {return mZ;}
        const float & P() const {return mP;}
        const float & POwn() const {return mPOwn;}
        const std::string *  Name() const {return mName;}
    private :
        float mZ;   // C'est le Z "absolu" du nuage, corrige de l'offset et du pas, c'est a la sauvegarde qu'on le remet eventuellement au pas
        float mP;
        float mPOwn;
        const std::string * mName;
};


bool operator < (const cElPile & aP1, const cElPile & aP2) {return aP1.Z() < aP2.Z();}

class cCmpPdsPile
{
   public :
     bool operator()(const cElPile & aP1,const cElPile & aP2)
     {
          return aP1.P() > aP2.P();
     }
};


class cTmpPile
{
    public :
        cTmpPile(float aZ,float aPds);
        void SetPPrec(cTmpPile &,float aExpFact);
        // double Pds() {return mPds0 / mNb0;}
        double ZMoy() {return mPds0 ? (mZP0/mPds0) : 0 ;}

    // private :
         double mZInit;
         double mPInit;
         double mZP0;
         double mPds0;
         double mNb0;

         double mNewZPp;
         double mNewPdsp;
         double mNewNbp;

         double mNewZPm;
         double mNewPdsm;
         double mNewNbm;

         double mPPrec;
         double mPNext;
         bool   mSelected;
};

cTmpPile::cTmpPile(float aZ,float aPds) :
    mZInit (aZ),
    mPInit (aPds),
    mZP0   (aZ * aPds),
    mPds0  (aPds),
    mNb0   (1.0),
    mPPrec (-1),
    mPNext (-1),
    mSelected (false)
{
}

void cTmpPile::SetPPrec(cTmpPile & aPrec,float aExpFact)
{
    ELISE_ASSERT(mZInit>=aPrec.mZInit,"Ordre coherence in cTmpPile::SetPPrec");
    double aPds = exp((aPrec.mZInit-mZInit)/aExpFact);
    mPPrec  = aPds;
    aPrec.mPNext = aPds;
}

void OneSensMoyTmpPile(std::vector<cTmpPile> & aVTmp)
{
     int aNb = aVTmp.size();
     aVTmp[0].mNewPdsp = aVTmp[0].mPds0;
     aVTmp[0].mNewZPp  = aVTmp[0].mZP0;
     aVTmp[0].mNewNbp  = aVTmp[0].mNb0;

     for (int aK=1 ; aK<int(aVTmp.size()) ; aK++)
     {
          aVTmp[aK].mNewPdsp = aVTmp[aK].mPds0 + aVTmp[aK-1].mNewPdsp * aVTmp[aK].mPPrec;
          aVTmp[aK].mNewZPp  = aVTmp[aK].mZP0  + aVTmp[aK-1].mNewZPp  * aVTmp[aK].mPPrec;
          aVTmp[aK].mNewNbp  = aVTmp[aK].mNb0  + aVTmp[aK-1].mNewNbp  * aVTmp[aK].mPPrec;
     }

     aVTmp[aNb-1].mNewPdsm = aVTmp[aNb-1].mPds0;
     aVTmp[aNb-1].mNewZPm = aVTmp[aNb-1].mZP0;
     aVTmp[aNb-1].mNewNbm = aVTmp[aNb-1].mNb0;
     for (int aK=(aVTmp.size()) -2 ; aK>=0 ; aK--)
     {
          aVTmp[aK].mNewPdsm = aVTmp[aK].mPds0 + aVTmp[aK+1].mNewPdsm * aVTmp[aK].mPNext;
          aVTmp[aK].mNewZPm  = aVTmp[aK].mZP0  + aVTmp[aK+1].mNewZPm  * aVTmp[aK].mPNext;
          aVTmp[aK].mNewNbm  = aVTmp[aK].mNb0  + aVTmp[aK+1].mNewNbm  * aVTmp[aK].mPNext;
     }

     for (int aK=0 ; aK<int(aVTmp.size()) ; aK++)
     {
          aVTmp[aK].mZP0  = (aVTmp[aK].mNewZPp  + aVTmp[aK].mNewZPm  - aVTmp[aK].mZP0) / aVTmp.size();
          aVTmp[aK].mPds0 = (aVTmp[aK].mNewPdsp + aVTmp[aK].mNewPdsm - aVTmp[aK].mPds0) / aVTmp.size();
          aVTmp[aK].mNb0  = (aVTmp[aK].mNewNbp  + aVTmp[aK].mNewNbm  - aVTmp[aK].mNb0) / aVTmp.size();
     }
}

Pt3dr VerifExp(const std::vector<cElPile> & aVPile,cElPile aP0,float aPixFact)
{
    Pt3dr aRes (0,0,0);
    for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
    {
       double aPds  = exp(-ElAbs(aP0.Z()-aVPile[aK].Z())/aPixFact);
       aRes = aRes + Pt3dr(aVPile[aK].Z()*aVPile[aK].P(),aVPile[aK].P(),1.0) * aPds;
    }
    return aRes / aVPile.size();
} 

Pt3dr  PdsSol(const std::vector<cElPile> & aVPile,const cElPile & aP0,float aPixFact)
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
       if (ElAbs(aP0.mZInit-aP0.mZInit) > anEc) return true;
       if (aP0.mPds0 < aP1.mPds0) return false;
   }
   return true;
}

bool IsMaxLoc(const std::vector<cTmpPile> & aVPile,int aK,float anEc)
{
    return    IsMaxLoc(aVPile,aK,anEc,-1,-1)
           && IsMaxLoc(aVPile,aK,anEc, 1,aVPile.size());
}


std::vector<cElPile>  ComputeExpEv(const std::vector<cElPile> & aVPile,double aResolPlani,float aPixFact)
{
   std::vector<cTmpPile> aTmp;
   float aZFact = (aPixFact*aResolPlani) * sqrt(2.0);
   for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
   {
        aTmp.push_back(cTmpPile(aVPile[aK].Z(),aVPile[aK].P()));
        if (aK>0)
        {
            aTmp[aK].SetPPrec(aTmp[aK-1],aZFact);
        }
        
   }
   OneSensMoyTmpPile(aTmp);

   if (LocBug)
   {
       for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
       {
            Pt3dr aPTest = VerifExp(aVPile,aVPile[aK],aZFact);
            Pt3dr aDif = aPTest - Pt3dr(aTmp[aK].mZP0,aTmp[aK].mPds0,aTmp[aK].mNb0 );
            // if (LocBug) std::cout << euclid(aDif) << " " << aPTest  << "\n";
            if (euclid (aDif)> 1e-3)
            {
                 std::cout << euclid (aDif) << aPTest << " " <<  aTmp[aK].mZInit << " " << aTmp[aK].mZP0 << " " << aTmp[aK].mPds0 ;
                 if (aVPile[aK].Name()) std::cout << " N=" << *(aVPile[aK].Name()) ;
                 std::cout  << "\n";
                 ELISE_ASSERT(false,"Coher in ComputeExpEv");
            }
       }
       // getchar();
   }
   OneSensMoyTmpPile(aTmp);
   for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
   {
        if (LocBug)
        {
                // Pt3dr aPSOL =  PdsSol(aVPile,aVPile[aK],aZFact);
                std::cout << " OUT " <<  aTmp[aK].mZInit << " P0 " << aTmp[aK].mPds0  << " PI " << aTmp[aK].mPInit 
                          << " PIm " << (aTmp[aK].mPds0/aTmp[aK].mNb0)
                          << " MaxLoc " << IsMaxLoc(aTmp,aK,aPixFact) ;
                 if (aVPile[aK].Name()) std::cout << " N=" << *(aVPile[aK].Name()) ;
                 std::cout << "\n";
                              //  << " " << aTmp[aK].mNb0 << " SOL " <<  (aPSOL.x/aPSOL.y) << " " << aPSOL.y  << " " << aPSOL.z << "\n";
        }
   }

   std::vector<cElPile> aRes;
   for (int aK=0 ; aK<int(aVPile.size()) ; aK++)
   {
       if (IsMaxLoc(aTmp,aK,aPixFact))
       {
           aTmp[aK].mSelected = true;
           cElPile  aPil (aTmp[aK].ZMoy(),aTmp[aK].mPds0,aTmp[aK].mPds0/aTmp[aK].mNb0);
           aRes.push_back (aPil);
       }
   }
   
   cCmpPdsPile aCmp;
   std::sort(aRes.begin(),aRes.end(),aCmp);


   while (aRes.size() > 5)  aRes.pop_back();


   if (LocBug)
   {
        std::cout << "ZFCAT " << aZFact << "\n";
   }

   return aRes;
}







template <class Type> class  cLoadedCP
{
    public :

        typedef  Type  tNum;
        typedef  typename El_CTypeTraits<tNum>::tBase  tNBase;

        //typedef  float  tNum;
        // typedef  double  tNBase;

        cLoadedCP(cFusionCarteProf<Type> &, const std::string & anId,const std::string & aFus);
        const cXML_ParamNuage3DMaille & Nuage() {return mNuage;}

        cElPile  CreatePile(const Pt2dr &) const;
        double   PdsLinear(const Pt2dr &) const;

        void  SetSz(const Pt2di & aSz);
        bool  ReLoad(const Box2dr & aBoxTer) ;
        const  cImage_Profondeur & IP() {return mIP;}
        const std::string & NameNuage() {return mNameNuage;}
        Im2D_U_INT1   ImCorrel() {return  mImCorrel;}

        std::string NameMM1P(const std::string aPref);
        Tiff_Im FileMM1P(const std::string aPref);

    private :

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


};



static const double MulCost = 1e3;

template <class Type> class cFusionCarteProf
{
     public :
        int ToICost(double aCost) {return round_ni(MulCost * aCost);}

        typedef  Type  tNum;
        typedef  typename El_CTypeTraits<tNum>::tBase  tNBase;

      //=================================================================
      // Interface pour utiliser la prog dyn
      //=================================================================
         //-------- Pre-requis 
            typedef  cElPile tArgCelTmp;
            typedef  cElPile tArgNappe;

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

          // -- Comlement
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
     private :

          std::vector<cElPile> ComputeEvidence(const std::vector<cElPile> & aPile,double aResolPlani);
          cElPile ComputeOneEvidence(const std::vector<cElPile> & aPile,const cElPile aP0,double aResolPlani);
          const cElPile * BestElem(const std::vector<cElPile> & aPile);

           double ToZSauv(double aZ) const;

          void DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut);
          void DoOneFusion(const std::string &);
          void DoCalc();

          cParamFusionMNT mParam;
          std::string     mCom;
          const cSpecAlgoFMNT & mSpecA;
          const cFMNtBySort *    mFBySort;
          const cFMNtByMaxEvid * mFByEv;
          const cFMNT_ProgDyn  * mFPrgD;
          const cFMNT_GesNoVal * mFNoVal;
          double                 mSigmaP;
          double                 mSigmaZ;
          cInterfChantierNameManipulateur * mICNM;
          std::vector<std::string>          mGenRes;
          bool                                    mCalledByMkf;
          bool                                    mDoByMkF;
          bool                                    mGenereMkF;
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
          bool                                    mZIsInv;
          std::list<std::string>                  mListCom;
          double                                  mResolPlani;
};


/**********************************************************************/
/*                                                                    */
/*                      cLoadedCP                                     */
/*                                                                    */
/**********************************************************************/

template <class Type>  cLoadedCP<Type>::cLoadedCP(cFusionCarteProf<Type> & aFCP, const std::string & anId,const std::string & aFus) :
  mFCP     (aFCP),
  mParam   (aFCP.Param()),
  mPAlg    (mParam.ParamAlgoFusionMNT()),
  mSeuilC  (mPAlg.FMNTSeuilCorrel()),
  mICNM    (aFCP.ICNM()),

  mFus         (aFus),
  mNameIm      (StdPrefix(mFus).substr(6,std::string::npos)),
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
  mAfM2CGlob  (Xml2EL(mNuage.Orientation().OrIntImaM2C())),
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
  mZIsInv     (false)

{


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
template <class Type> std::string cLoadedCP<Type>::NameMM1P(const std::string aPref)
{
    return  mParam.WorkDirPFM().Val() + aPref + "-" +  mNameIm + ".tif";
}

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
   ELISE_COPY(mImMasq.all_pts(),trans(mTifMasq.in(),mBoxImCur._p0),mImMasq.out());
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
               std::cout << "HHHHH " << aSM1P.PdsAR().Val()  << " " << mNameIm << "\n";
               std::cout << FileMM1P("Depth").sz() << "\n";

               Tiff_Im::Create8BFromFonc
               (
                   NameMM1P("Quality"),
                   FileMM1P("Mask").sz(),
                   aFCor
               );
 getchar();
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
   double aPds = mTImMasq.get(round_ni(aPIm),0);
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


template <class Type> cElPile  cLoadedCP<Type>::CreatePile(const Pt2dr & aPTer) const
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
       aZ = mIP.OrigineAlti() +  mTImCP.getprojR(aPIm) * mIP.ResolutionAlti();
       if (mZIsInv)
          aZ= 1.0/aZ;
   }
   return cElPile(aZ,aPds,-1,&mNameCP);
}
/*
*/



/**********************************************************************/
/*                                                                    */
/*                      cFusionCarteProf                              */
/*                                                                    */
/**********************************************************************/

template <class Type> cElPile  cFusionCarteProf<Type>::ComputeOneEvidence(const std::vector<cElPile> & aPile,const cElPile aP0,double aResolPlani)
{
    double aSomPp = 0;
    double aSomPz = 0;
    double aSomZ = 0;
    const float & aZ0 = aP0.Z();

    for (int aKp=0 ; aKp<int(aPile.size()) ; aKp++)
    {
        const cElPile & aPilK = aPile[aKp];
        const float & aPk = aPilK.P();
        if (aPk)
        {
            const float & aZk = aPilK.Z();
            double aDz = ElAbs(aZ0-aZk) / aResolPlani;  // Le DZ est relatif a la resolution alti pour le seuillage
            if (aDz<mFByEv->MaxDif().Val())
            {
                 double aPp = exp(-ElSquare(aDz/mSigmaP)) * aPk;
                 double aPz = exp(-ElSquare(aDz/mSigmaZ)) * aPk;

                 aSomPp += aPp;
                 aSomPz += aPz;
                 aSomZ += aPz * aZk;
            }
        }
    }

    if (aSomPz>0) 
       aSomZ /= aSomPz;
    else
       aSomPp =0;

    return cElPile(aSomZ,aSomPp);
}

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

template <class Type> std::vector<cElPile>  cFusionCarteProf<Type>::ComputeEvidence(const std::vector<cElPile> & aPile,double aResolPlani)
{
    std::vector<cElPile> aRes;
    for (int aKp=0 ; aKp<int(aPile.size()) ; aKp++)
    {
         aRes.push_back(ComputeOneEvidence(aPile,aPile[aKp],aResolPlani));
    }
    return aRes;
}


template <class Type> const cElPile * cFusionCarteProf<Type>::BestElem(const std::vector<cElPile> & aPile)
{
    int aBesK=0 ;
    float aBestP=-1;

    for (int aKp=0 ; aKp<int(aPile.size()) ; aKp++)
    {
        if (aPile[aKp].P() > aBestP)
        {
           aBesK = aKp;
           aBestP= aPile[aKp].P();
        }
        
    }
    return &(aPile[aBesK]);
}
 

template <class Type> void cFusionCarteProf<Type>::DoOneFusion(const std::string & anId)
{
    std::string aNameNuage = mICNM->Dir() + mICNM->Assoc1To1(mParam.KeyResult(),anId,true);

    mNameTif = StdPrefix(aNameNuage)+ ".tif";
    std::cout << anId  << "=> " << mNameTif<< "\n";
    mNameMasq = StdPrefix(aNameNuage)+ "_Masq.tif";
    mNameCorrel = StdPrefix(aNameNuage)+ "_Correl.tif";


    std::vector<std::string> aStrFus = GetStrFromGenStrRel(mICNM,mParam.GenereInput(),anId);

    if (aStrFus.size() == 0)
    {
        std::cout << "FOR ID = " << anId  << "\n";
        ELISE_ASSERT(false,"No data in DoOneFusion");
    }

 
    for (int aK=0 ; aK<int(aStrFus.size()) ; aK++)
    {
          mVC.push_back(new cLoadedCP<Type>(*this,anId,aStrFus[aK]));
    }

    if (mParam.ModeleNuageResult().IsInit())
    {
       mNuage = StdGetObjFromFile<cXML_ParamNuage3DMaille>
                (
                     //mParam.ModeleNuageResult().Val(),
                      mICNM->Dir() + mICNM->Assoc1To1(mParam.ModeleNuageResult().Val(),anId,true),
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
         for (int aK=0 ; aK<int(mVC.size()) ; aK++)
         {
              double aResol = mVC[aK]->IP().ResolutionAlti();
              double anOri = mVC[aK]->IP().OrigineAlti();

              //  std::cout << "VVVa " <<  aK << " " << anOri  << " " << aResol << "\n";
              aSomResolAlti += aResol;
              aSomOriAlti += anOri;
         }

         aSomResolAlti /=  mVC.size();
         aSomOriAlti /=  mVC.size();
         //  std::cout << "MOYYYY " << aSomOriAlti  << " " << aSomResolAlti << "\n";

         mNuage.Image_Profondeur().Val().ResolutionAlti() = aSomResolAlti;
         mNuage.Image_Profondeur().Val().OrigineAlti() = aSomOriAlti;
        
        // Creation du TFW
        {
            std::string aNameTFW = StdPrefix(mNameTif) + ".tfw";
            std::ofstream aFtfw(aNameTFW.c_str());
            aFtfw.precision(10);
            
            ElAffin2D aAfM2C = Xml2EL(mNuage.Orientation().OrIntImaM2C());
            
            
            double resolutionX = 1./aAfM2C.I10().x;
            double resolutionY = 1./aAfM2C.I01().y;
            double origineX = -aAfM2C.I00().x * resolutionX;
            double origineY = -aAfM2C.I00().y * resolutionY;
            aFtfw << resolutionX << "\n" << 0 << "\n";
            aFtfw << 0 << "\n" << resolutionY << "\n";
            aFtfw << origineX << "\n" << origineY << "\n";
            
            //aFtfw << aFOM.ResolutionPlani().x << "\n" << 0 << "\n";
            //aFtfw << 0 << "\n" << aFOM.ResolutionPlani().y << "\n";
            //aFtfw << aFOM.OriginePlani().x << "\n" << aFOM.OriginePlani().y << "\n";
            aFtfw.close();
        }
        
    }

   mZIsInv = false;
   if (mNuage.ModeFaisceauxImage().IsInit())
      mZIsInv = mNuage.ModeFaisceauxImage().Val().ZIsInverse();

    mIP = &(mNuage.Image_Profondeur().Val());
    mIP->Image() = NameWithoutDir(mNameTif) ;
    mIP->Masq() =  NameWithoutDir(mNameMasq);
    mIP->Correl() =  NameWithoutDir(mNameCorrel);
    
    mAfM2CGlob  = Xml2EL(mNuage.Orientation().OrIntImaM2C());
    mAfC2MGlob = mAfM2CGlob.inv();
    mSzGlob = mNuage.NbPixel();

    if (! mCalledByMkf)
    {
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

             if (mGenereMkF)
             {
                  std::string aNewCom =   mCom 
                                  + std::string(" InterneCalledByProcess=true")
                                  + std::string(" InterneSingleImage=") +  anId
                                  + std::string(" InterneSingleBox=") + ToString(aKI);
                  mListCom.push_back(aNewCom);
             }
             else
             {
                 if ((!mCalledByMkf) || (mParam.InterneSingleBox().Val()==aKI))
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


template <class Type> double cFusionCarteProf<Type>::ToZSauv(double aZ) const
{
   if (mZIsInv) aZ = 1/aZ;
   return  (aZ -mIP->OrigineAlti()) / mIP->ResolutionAlti();
}

template <class Type> void cFusionCarteProf<Type>::DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut)
{
   ElTimer aChrono;
   bool ShowTime = false;
   std::cout << "RESTE " << aKB <<   " BLOCS \n";
   mAfM2CCur =  ElAffin2D::trans(-Pt2dr(aBoxIn._p0)) * mAfM2CGlob ;
   mAfC2MCur = mAfM2CCur.inv();


   mResolPlani = (euclid(mAfC2MCur.I10()) + euclid(mAfC2MCur.I01()))/2.0;
   // std::cout << "  SCALE " << mResolPlani << "\n";

   mSzCur = aBoxIn.sz();

   Im2D<tNum,tNBase>  aImFus(mSzCur.x,mSzCur.y);
   TIm2D<tNum,tNBase> aTImFus(aImFus);

   Im2D_Bits<1>       aImMasq(mSzCur.x,mSzCur.y);
   TIm2DBits<1>       aTImMasq(aImMasq);


   Im2D<tNum,tNBase>  aImCorrel(mSzCur.x,mSzCur.y);
   TIm2D<tNum,tNBase> aTImCorrel(aImCorrel);


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
      std::cout << "RRELOAD " << mVCL.size() << " on " << mVC.size() << " time= " << aChrono.uval() << "\n";
   }

   cProg2DOptimiser<cFusionCarteProf>  * aPrgD = 0;
   TIm2D<INT2,INT>  aTIm0(Pt2di(1,1));
   TIm2D<INT2,INT>  aTImNb(Pt2di(1,1));
   

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
   
   if (ShowTime)
   {
      std::cout << " Init PrgD time= " << aChrono.uval() << "\n";
   }


   cInterfaceCoxRoyAlgo  * aCox = 0;
   Im2D_INT2 aIZMin(1,1);
   Im2D_INT2 aIZMax(1,1);

   if (true)
   {
        aIZMin = Im2D_INT2(mSzCur.x,mSzCur.y,0);
        aIZMax = Im2D_INT2(mSzCur.x,mSzCur.y,3);
        aCox = cInterfaceCoxRoyAlgo::NewOne(mSzCur.x,mSzCur.y,aIZMin.data(),aIZMax.data(),true,false);
   }



   double aMul = 100.0;
   double aGainDef = 0.15;
   double aRegul = 0.5;

   std::vector<cElPile> aPCel;
   Pt2di aQ0;
   for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
   {
        for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
        {
if (false)
{
static int aCpt =-1; aCpt++;
static       Video_Win aW = Video_Win::WStd(mSzCur,1.0);

cLoadedCP<Type> * aLCP = VCLOfName("/media/data2/Munich/MTD-Nuage/Basculed-40_0314_PAN.tif-40_0315_PAN.tif.xml");
// static       Tiff_Im aTF = Tiff_Im::StdConv("FusionZ1_NuageImProf_LeChantier_Etape_1_Correl.tif");

ELISE_COPY(aLCP->ImCorrel().all_pts(),aLCP->ImCorrel().in(),aW.ogray());
if (aCpt==-1) 
{
   aQ0 = Pt2di(573,213);
}
else
{
   Clik aClk = aW.clik_in();
   aQ0 = Pt2di(aClk._pt);
}
LocBug=true;
}


            Pt2dr aT0 = mAfC2MCur(Pt2dr(aQ0));
            aPCel.clear();
            for (int aKI=0 ; aKI<int(mVCL.size()); aKI++)
            {
                cElPile anEl = mVCL[aKI]->CreatePile(aT0);
                if (anEl.P()>0)
                {
                   aPCel.push_back(anEl);
                }
            }
            int Ok= 1;
            double aZ=0;
            double aP=0;
            if (aPCel.size() >0)
            {
                std::sort(aPCel.begin(),aPCel.end());
                std::vector<cElPile> aVPile = ComputeExpEv(aPCel,mResolPlani,mSigmaP);
                aZ = aVPile[0].Z();
                aP = aVPile[0].POwn();

                aTImFus.oset(aQ0,(tNBase)aZ);
                aTImCorrel.oset(aQ0,ElMax(0,ElMin(255,(round_ni(aP*255)))));
                if (aCox)
                {
                   aCox->SetCostVert(aQ0.x,aQ0.y,0,round_ni(aMul*(1-aGainDef)));
                   aCox->SetCostVert(aQ0.x,aQ0.y,1,round_ni(aMul*(1-aP)));
                   aCox->SetCostVert(aQ0.x,aQ0.y,2, round_ni(aMul*2));
                }
            }
            aTImMasq.oset(aQ0,Ok);
if (LocBug)
{
   std::cout << "Q00= " << aQ0 << "\n";
}
        }
   }

   if (aCox)
   {
      ElTimer aT0;
      std::cout << "Begin Cox-Roy\n";
      aCox->SetStdCostRegul(0,aMul*aRegul,0);

      Im2D_INT2 aISol(mSzCur.x,mSzCur.y);
      aCox->TopMaxFlowStd(aISol.data());
      std::cout << "End Cox-Roy " << aT0.uval() << "\n";
      ELISE_COPY(aISol.all_pts(),aISol.in()!=0,aImMasq.out());
   }

   if (ShowTime)
      std::cout << " Init Cost time= " << aChrono.uval() << "\n";

   if (aPrgD)
   {
       aPrgD->DoOptim(mFPrgD->NbDir());
       std::cout << " Prg Dyn time= " << aChrono.uval()  << " Nb Dir " << mFPrgD->NbDir() << "\n";
       Im2D_INT2 aSol(mSzCur.x,mSzCur.y);
       INT2 ** aDSol = aSol.data();
       aPrgD->TranfereSol(aDSol);
       for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
       {
            for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
            {
                int aZ = aDSol[aQ0.y][aQ0.x];
                tCelNap & aCol =  aPrgD->Nappe().Data()[aQ0.y][aQ0.x][aZ];
                if (aZ>=0)
                {
                   aTImFus.oset(aQ0,ToZSauv(aCol.ArgAux().Z()));
                }
                else
                {
                     aTImMasq.oset(aQ0,0);
                }
            }
       }
   }

   if (ShowTime)
      std::cout << " Dow Opt time= " << aChrono.uval() << "\n";

   if (1)
   {
        Im2D_Bits<1>       aIm1(mSzCur.x,mSzCur.y,1);
        TIm2DBits<1>       aTIm1(aIm1);
        ComplKLipsParLBas (aImMasq, aIm1,aImFus,1.0);
   }



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
    aCelOut.UpdateCostOneArc(aCelIn,aSens,(mFNoVal?mFNoVal->Trans():0));
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
        const  cElPile & aPIn = anInp.ArgAux();
        for (int aZOut=0 ; aZOut<aOutZMax ; aZOut++)
        {
            tCelOpt & anOut = aTabOuput[aZOut];
            const  cElPile & aPOut = anOut.ArgAux();
            double aDZ = ElAbs(aPIn.Z()-aPOut.Z())/mResolPlani;
            if ((mFNoVal==0) || (aDZ < mFNoVal->PenteMax()))
            {
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
       if ((!mCalledByMkf) || (mGenRes[aKS]==mParam.InterneSingleImage().Val()))
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
     mFBySort      (mSpecA.FMNtBySort().PtrVal()),
     mFByEv        (mSpecA.FMNtByMaxEvid().PtrVal()),
     mFPrgD        (mFByEv ? mFByEv->FMNT_ProgDyn().PtrVal() : 0),
     mFNoVal       (mFPrgD ? mFPrgD->FMNT_GesNoVal().PtrVal() : 0),
     mICNM         (aParam.mICNM),
     mGenRes       (GetStrFromGenStr(mICNM,mParam.GenereRes())),
     mCalledByMkf  (mParam.InterneCalledByProcess().Val()),
     mDoByMkF      (mParam.ParalMkF().IsInit()),
     mGenereMkF    ((!mCalledByMkf) && mDoByMkF)
{
// std::cout << "mCalledByMkf " << mCalledByMkf << "\n"; getchar();
    if (mFByEv)
    {
          mSigmaP = mFByEv->SigmaPds();
          mSigmaZ =  mFByEv->SigmaZ().ValWithDef(mSigmaP);
    }

    DoCalc();

    if (mGenereMkF)
    {
        cEl_GPAO::DoComInParal(mListCom,mParam.ParalMkF().Val());
    }
/*
    if (mParam.ParalMkF().IsInit()  && (! mCalledByMkf))
    {
        DoMake();
    }
    else
    {
        DoCalc();
    }
*/
}




/*
*/

int FusionCarteProf_main(int argc,char ** argv)
{
  ELISE_ASSERT(argc>=2,"Not Enough args to FusionMNT.cpp");
  MMD_InitArgcArgv(argc,argv);

  Tiff_Im::SetDefTileFile(50000);

  std::string aCom0 = MMBin() + "mm3d "+ MakeStrFromArgcARgv(argc,argv);
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
