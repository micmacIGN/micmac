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


/*********************************************************************/
/*                                                                   */
/*                       cFusionCarteProf                            */
/*                                                                   */
/*********************************************************************/

template <class Type> class  cLoadedCP;        // Represente une carte de profondeur chargee
template <class Type> class cFusionCarteProf;  // Represente l'appli globale

class cElPile;

class fPPile;  // Functeur : renvoie le poids
class fZPile;  // Functer : renvoie le Z
    //=================================

class cElPile
{
    public :
        cElPile (double aZ,double aPds) :
           mZ  (aZ),
           mP  (aPds)
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
    private :
        float mZ;
        float mP;
};

bool operator < (const cElPile & aP1, const cElPile & aP2) {return aP1.Z() < aP2.Z();}

class fPPile
{
    public :
       double operator ()(const cElPile & aPile) const {return aPile.P();}
};
static fPPile ThePPile;
class fZPile
{
    public :
       typedef double tValue;
       double operator ()(const cElPile & aPile) const {return aPile.Z();}
};
static fZPile TheZPile;





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

    private :

        cFusionCarteProf<Type>  & mFCP;
        const cParamFusionMNT & mParam;
        const cParamAlgoFusionMNT & mPAlg;
        double                      mSeuilC;
        cInterfChantierNameManipulateur * mICNM;

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
     private :

          std::vector<cElPile> ComputeEvidence(const std::vector<cElPile> & aPile);
          cElPile ComputeOneEvidence(const std::vector<cElPile> & aPile,const cElPile aP0);
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
          bool                                    mZIsInv;
          std::list<std::string>                  mListCom;
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
       Tiff_Im aTifCorrel = Tiff_Im::StdConv(mNameCorrel);
       ELISE_COPY(mImCorrel.all_pts(),trans(aTifCorrel.in(),mBoxImCur._p0),mImCorrel.out());
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
           aPds = (aPds -128.0) / 128.0;
           aPds = (aPds-mSeuilC)/(1.0-mSeuilC);
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
   return cElPile(aZ,aPds);
}
/*
*/



/**********************************************************************/
/*                                                                    */
/*                      cFusionCarteProf                              */
/*                                                                    */
/**********************************************************************/

template <class Type> cElPile  cFusionCarteProf<Type>::ComputeOneEvidence(const std::vector<cElPile> & aPile,const cElPile aP0)
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
            double aDz = ElAbs(aZ0-aZk);
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

template <class Type> std::vector<cElPile>  cFusionCarteProf<Type>::ComputeEvidence(const std::vector<cElPile> & aPile)
{
    std::vector<cElPile> aRes;
    for (int aKp=0 ; aKp<int(aPile.size()) ; aKp++)
    {
         aRes.push_back(ComputeOneEvidence(aPile,aPile[aKp]));
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
                     mICNM->Assoc1To1(mParam.ModeleNuageResult().Val(),anId,true),
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
    }
   mZIsInv = false;
   if (mNuage.ModeFaisceauxImage().IsInit())
      mZIsInv = mNuage.ModeFaisceauxImage().Val().ZIsInverse();

    mIP = &(mNuage.Image_Profondeur().Val());
    mIP->Image() = NameWithoutDir(mNameTif) ;
    mIP->Masq() =  NameWithoutDir(mNameMasq);
    mIP->Correl().SetNoInit();
    
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

       Tiff_Im  aTifMasq = Tiff_Im::CreateIfNeeded
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

   DeleteAndClear(mVC);
}


template <class Type> double cFusionCarteProf<Type>::ToZSauv(double aZ) const
{
   if (mZIsInv) aZ = 1/aZ;
   return  (aZ -mIP->OrigineAlti()) / mIP->ResolutionAlti();
}

template <class Type> void cFusionCarteProf<Type>::DoOneBloc(int aKB,const Box2di & aBoxIn,const Box2di & aBoxOut)
{
   std::cout << "RESTE " << aKB <<   " BLOCS \n";
   mAfM2CCur =  ElAffin2D::trans(-Pt2dr(aBoxIn._p0)) * mAfM2CGlob ;
   mAfC2MCur = mAfM2CCur.inv();

   mSzCur = aBoxIn.sz();

   Im2D<tNum,tNBase>  aImFus(mSzCur.x,mSzCur.y);
   TIm2D<tNum,tNBase> aTImFus(aImFus);


   Im2D_Bits<1>       aImMasq(mSzCur.x,mSzCur.y);
   TIm2DBits<1>       aTImMasq(aImMasq);

   Box2di aBoxInLoc(Pt2di(0,0),mSzCur);
   Box2dr aBoxTer = aBoxInLoc.BoxImage(mAfC2MCur);


   mVCL.clear();
   for (int aK=0 ; aK<int(mVC.size()) ; aK++)
   {
       if (mVC[aK]->ReLoad(aBoxTer))
          mVCL.push_back(mVC[aK]);
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
   



   std::vector<cElPile> aPCel;
   Pt2di aQ0;
   for (aQ0.y = 0 ; aQ0.y < mSzCur.y; aQ0.y++)
   {
        for (aQ0.x = 0 ; aQ0.x < mSzCur.x; aQ0.x++)
        {
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
            int Ok= 0;
            double aZ=0;
            if (aPCel.size() >0)
            {
                std::sort(aPCel.begin(),aPCel.end());
                std::vector<cElPile> aNewV;
                const cElPile * aBestP = 0;

                if (mFBySort)
                {
                   double aSomP = SomPerc(aPCel,ThePPile);
                   aZ = GenValPdsPercentile(aPCel,mFBySort->PercFusion().Val(),TheZPile,ThePPile,aSomP);
                }
                else if (mFByEv)
                {
                    aNewV =   ComputeEvidence(aPCel);
                    aBestP = BestElem(aNewV);
                    aZ = aBestP->Z();
                }
                if (aPrgD)
                {
                    //typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
                    // cTplCelNapPrgDyn
                    tCelNap * aCol =  aPrgD->Nappe().Data()[aQ0.y][aQ0.x];
                    ELISE_ASSERT(int(aNewV.size()) ==aTImNb.get(aQ0),"Incoh aPCel.size() ==aTImNb.get(aQ0)");

                    for (int aK=0 ; aK< int(aPCel.size()) ; aK++)
                    {
                         // aCol[aK].SetOwnCost(ToICost(0));
                         aCol[aK].SetOwnCost(ToICost(aBestP->P()-aNewV[aK].P()));
                         // aCol[aK].SetOwnCost(ToICost(ElAbs(aQ0.x-aK)%3));
                         aCol[aK].ArgAux() = aNewV[aK];
                    }
//std::cout << "Bestt Ppp " << aBestP->P() << " " << aNewV.size() << "\n";
                    double aCostNV = mFNoVal  ?
                                     (aBestP->P()* mFNoVal->GainNoVal()) :
                                     (10+aNewV.size()+aBestP->P() * 2);
                    aCol[-1].SetOwnCost(ToICost(aCostNV));
                }
                else
                {
                   aZ = ToZSauv(aZ);
                   //   if (:mZIsInv) aZ = 1/aZ;
                   //   aZ = (aZ -mIP->OrigineAlti()) / mIP->ResolutionAlti();
                }
                Ok=1;
            }
            if (! aPrgD)
            {
               aTImFus.oset(aQ0,(tNBase)aZ);
            }
            aTImMasq.oset(aQ0,(tNBase)Ok);
        }
   }


   if (aPrgD)
   {
       aPrgD->DoOptim(mFPrgD->NbDir());
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
            double aDZ = ElAbs(aPIn.Z()-aPOut.Z());
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
                                              "WorkDir",
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
