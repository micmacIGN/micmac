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

#ifndef _CPTOFCORREL_H_
#define _CPTOFCORREL_H_

// Classe pour calculer des points "favorables a la correlation 2D",
//  c'est a dire des point qui n'auto-correlent pas avec leur
// translate et qui ont du contraste

class cQAC_XYV;
class cQAC_Stat_XYV;
class cPtOfCorrel
{
    public :
        typedef cQAC_XYV tElem;
        typedef cQAC_Stat_XYV tCumul;


        cPtOfCorrel(Pt2di aSz,Fonc_Num aF,int aVCor);

        void NoOp() {}
        double AutoCorTeta(Pt2di aP,double aTeta,double anEpsilon);
        double BrutaleAutoCor(Pt2di aP,int aNbTeta,double anEpsilon);


        void QuickAutoCor_Gen(Pt2di aP,double & aLambda,double & aSvv);

        double QuickAutoCor(Pt2di aP);
        double QuickAutoCorWithNoise(Pt2di aP,double aBr);

        void MakeScoreAndMasq
             (
                   Im2D_REAL4 & aISc,
                   double  aEstBr,
                   Im2D_Bits<1> & aMasq,
                   double aSeuilVp,
                   double aSeuilEcart
             );


        void Init(const std::complex<int> & aP,cQAC_XYV &);
        void UseAggreg(const std::complex<int> & aP,const cQAC_Stat_XYV &);
        void OnNewLine(int);
   private :
       double  PdsCorrel(double anX,double anY);

       Pt2di                   mSz;
       int                     mVCor;
       Im2D_REAL4              mImIn;
       TIm2D<REAL4,REAL8>      mTIn;
       Im2D_REAL4              mGrX;
       TIm2D<REAL4,REAL8>      mTGrX;
       Im2D_REAL4              mGrY;
       TIm2D<REAL4,REAL8>      mTGrY;

       TIm2D<REAL4,REAL8> *   mISc;
       TIm2DBits<1> *         mMasq;
       double                 mEstBr;
       double                 mSeuilVP;
       double                 mSeuilEcart;
};

//  Classe pour reparti les points d'interet en respectant les criteres
//  suivants :
//       - pas de points dans le masque
//       - un point par cellule


class cRepartPtInteret
{
    public :
        cRepartPtInteret
        (
            Im2D_REAL4        aImSc,
            Im2D_Bits<1>      aMasq,
            const cEquiv1D &  anEqX,
            const cEquiv1D &  anEqY,
            double            aDistRejet,
            double            aDistEvitement
        );


        Im2D_Bits<1> ItereAndResult();

    private :
        void OnePasse(double aProp);

       void UpdatePond (Pt2di aP, Im2D_REAL4 aPond, bool Ajout);

       Pt2di    mNbC;
       Im2D_REAL4          mImSc;
       TIm2D<REAL4,REAL8>  mTSc;
       Im2D_Bits<1>        mMasq;
       TIm2DBits<1>        mTMasq;
       Pt2di               mSzGlob;
       Im2D_REAL4          mPondGlob;
       TIm2D<REAL4,REAL8>  mTPondGlob;

       cEquiv1D mEqX;
       cEquiv1D mEqY;
       Im2D_INT4 mPX;
       TIm2D<INT4,INT4> mTPx;
       Im2D_INT4 mPY;
       TIm2D<INT4,INT4> mTPy;
       double    mValRejet;
       double    mDRejMax;
       //double    mDRejCur;
       //double    mDEvCur;
       double    mDEvMax;
       Im2D_REAL4  mLastImPond;
};

//=================================== Critere type fast pour selectionner les points favorables à la correl

class cFastCriterCompute
{
     public :
        cFastCriterCompute(Flux_Pts aFlux):
             mNbPts (SortedAngleFlux2StdCont(mVPt,aFlux).size()),
             mMaxSz ((1+mNbPts)/2),
             mFRA   (OpMax,0,mNbPts,mMaxSz-mNbPts,mMaxSz)
        {
            cCmpPtOnAngle<Pt2di>  aCmp;
            std::sort(mVPt.begin(),mVPt.end(),aCmp);
        }


        static cFastCriterCompute * Circle(double aRay)
        {
            return new cFastCriterCompute(circle(Pt2dr(0,0),aRay));
        }
        const  std::vector<Pt2di> & VPt() const {return mVPt;}
        cFastReducAssoc<double> & FRA() {return mFRA;}
     private :
         cFastCriterCompute(const cFastCriterCompute&);
         std::vector<Pt2di>       mVPt;
         int                      mNbPts;
         int                      mMaxSz;
         cFastReducAssoc<double>  mFRA;
  
};

template <class TIm> Pt2dr  FastQuality(TIm anIm,Pt2di aP,cFastCriterCompute & aCrit,bool IsMax,Pt2dr aProp)
{
   std::vector<double> aVVals;
   const  std::vector<Pt2di> & aVPt = aCrit.VPt();
   int aNbPts = aVPt.size();    //les voisins
   typename TIm::tValueElem aDef =   IsMax ?
                                     El_CTypeTraits<typename TIm::tValueElem>::MaxValue():
                                     El_CTypeTraits<typename TIm::tValueElem>::MinValue();
   double aSign = IsMax ? 1.0 : -1.0;
   for (int aK=0 ; aK<aNbPts ; aK++)
   {
       typename TIm::tValueElem aVal = anIm.get(aP+aVPt[aK],aDef) * aSign;
       aVVals.push_back(aVal); //valeur des voisins
       aCrit.FRA().In(aK) = aVal;
   }
   typename TIm::tValueElem aV0 = anIm.getproj(aP)*aSign;

   // Definition standard de Fast
   std::vector<double> aFOut(aNbPts);
   double aPropStd = aProp.x;   //TT_PropFastStd (0.75)
   double aVPerc = KthValProp(aVVals,aPropStd);
   double aResStd = (aV0 -aVPerc) ; // > TT_SeuilFastStd = 5

   // Definition contignu
   double aPropC = aProp.y;
   int aNbC = round_up(aPropC * aNbPts);
   int aNbMin = aNbC / 2;
   aCrit.FRA().Compute(-aNbMin,aNbC-aNbMin,eCFR_Per);
   typename TIm::tValueElem aVMin =   aCrit.FRA().Out(0);
   for (int aK=1 ; aK<aNbPts ; aK++)
   {
      typename TIm::tValueElem aVk = aCrit.FRA().Out(aK);
     // std::cout << "HHHH " << aCrit.FRA().In(aK) << " " << aCrit.FRA().Out(aK) << "\n";
       aVMin = ElMin(aVMin, aVk);
       // aVMin = ElMin(aVMin, aCrit.FRA().Out(aK)); // Pb de cast
   }
   double  aResC = (aV0-aVMin);

   return Pt2dr(aResStd,aResC);
}

/*************************************************************/
/*                                                           */
/*           Auto Correlation directionnelle                 */
/*                                                           */
/*************************************************************/

template <class TypeIm> class  cAutoCorrelDir
{
    public :
        typedef typename TypeIm::tValueElem tElem;
        typedef typename TypeIm::OutputFonc tBase;
        cAutoCorrelDir(TypeIm anIm,const Pt2di & aP0,double aRho,int aSzW) :
           mTIm   (anIm),
           mP0    (aP0),
           mRho   (aRho),
           mSzW   (aSzW)
        {
        }

        Pt2dr DoIt()
        {
           double aStep0 = 1/mRho;
           int aNb = round_up(PI/aStep0);
           Pt2dr aRes0 = DoItOneStep(0.0,aNb,aStep0);
           Pt2dr aRes1 = DoItOneStep(aRes0.x,3,aStep0/4.0);
           Pt2dr aRes2 = DoItOneStep(aRes1.x,2,aStep0/10.0);
           return aRes2;
        }

        void ResetIm(const TypeIm & aTIm) {mTIm=aTIm;}
    protected :
        Pt2dr  DoItOneStep(double aTeta0,int aNb,double aStep)
        {
           double aScMax = -1e10;
           double aTetaMax = 0;
           for (int aK=-aNb; aK<aNb ; aK++)
           {
               double aTeta =  aTeta0 + aK * aStep;
               double aVal =  CorrelTeta(aTeta) ;
               if (aVal >aScMax)
               {
                  aScMax = aVal;
                  aTetaMax = aTeta;
               }
           }
           return Pt2dr(aTetaMax,aScMax);
        }


        double  RCorrelOneOffset(const Pt2di & aP0,const Pt2dr & anOffset,int aSzW)
        {
            tBase aDef =   El_CTypeTraits<tBase>::MaxValue();
            RMat_Inertie aMat;
            for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
            {
                for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
                {
                    Pt2di aP1 = aP0 + Pt2di(aDx,aDy);
                    tBase aV1 = mTIm.get(aP1,aDef);
                    if (aV1==aDef) return -1;
                    tBase aV2 = mTIm.getr(Pt2dr(aP1)+anOffset,aDef);
                    if (aV2==aDef) return -1;
                    aMat.add_pt_en_place(aV1,aV2);
                }
            }
            return aMat.correlation();
        }

        double  ICorrelOneOffset(const Pt2di & aP0,const Pt2di & anOffset,int aSzW)
        {
            tBase aDef =   El_CTypeTraits<tBase>::MaxValue();
            RMat_Inertie aMat;
            for (int aDx=-aSzW ; aDx<=aSzW ; aDx++)
            {
                for (int aDy=-aSzW ; aDy<=aSzW ; aDy++)
                {
                    Pt2di aP1 = aP0 + Pt2di(aDx,aDy);
                    tBase aV1 = mTIm.get(aP1,aDef);
                    if (aV1==aDef) return -1;
                    tBase aV2 = mTIm.get(aP1+anOffset,aDef);
                    if (aV2==aDef) return -1;
                    aMat.add_pt_en_place(aV1,aV2);
                }
            }
            return aMat.correlation();
        }

        double  CorrelTeta(double aTeta)
        {
            return RCorrelOneOffset(mP0,Pt2dr::FromPolar(mRho,aTeta),mSzW);
        }


        TypeIm  mTIm;
        Pt2di   mP0;
        double  mRho;
        int     mSzW;
};

template <class TypeIm> class cCutAutoCorrelDir : public cAutoCorrelDir<TypeIm>
{
    public :
         int mNumOut;
         double  mCorOut;

         cCutAutoCorrelDir(TypeIm anIm,const Pt2di & aP0,double aRho,int aSzW ) :
             cAutoCorrelDir<TypeIm> (anIm,aP0,aRho,aSzW),
             mNbPts                 (SortedAngleFlux2StdCont(mVPt,circle(Pt2dr(0,0),aRho)).size())
         {
         }
         void ResetIm(const TypeIm & anIm) { cAutoCorrelDir<TypeIm>::ResetIm(anIm); }
        bool  AutoCorrel(const Pt2di & aP0,double aRejetInt,double aRejetReel,double aSeuilAccept,Pt2dr * aPtrRes=0)
         {
               this->mP0 = aP0;
               double aCorrMax = -2;
               int    aKMax = -1;
               for (int aK=0 ; aK<mNbPts ; aK++)
               {
                    double aCor = this->ICorrelOneOffset(this->mP0,mVPt[aK],this->mSzW);
// if (BugAC) std::cout << "CCcccI " << aCor << " " << this->mTIm.sz() << "\n";
                    if (aCor > aSeuilAccept) 
                    {
                         mCorOut = aCor;
                         mNumOut = 0;
                         return true;
                    }
                    if (aCor > aCorrMax)
                    {
                        aCorrMax = aCor;
                        aKMax = aK;
                    }
               }
               ELISE_ASSERT(aKMax!=-1,"AutoCorrel no K");
               if (aCorrMax < aRejetInt) 
               {
                    mCorOut = aCorrMax;
                    mNumOut = 1;
                    return false;
               }

               Pt2dr aRhoTeta = Pt2dr::polar(Pt2dr(mVPt[aKMax]),0.0);

               double aStep0 = 1/this->mRho;
               //  Pt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y,aStep0*0.5,2);  BUG CORRIGE VERIF AVEC GIANG
               Pt2dr aRes1 =  this->DoItOneStep(aRhoTeta.y,2,aStep0*0.5);

               if (aRes1.y>aSeuilAccept)   
               {
                  mNumOut = 2;
                  mCorOut = aRes1.y;
                  return true;
               }
               if (aRes1.y<aRejetReel) 
               {
                  mNumOut = 3;
                  mCorOut = aRes1.y;
                  return false;
               }

               // Pt2dr aRes2 =  this->DoItOneStep(aRes1.x,aStep0*0.2,2); BUG CORRIGE VERIF AVEC GIANG
               Pt2dr aRes2 =  this->DoItOneStep(aRes1.x,2,aStep0*0.2);

               if (aPtrRes) 
                  *aPtrRes = aRes2;

               mNumOut = 4;
               mCorOut = aRes2.y;
               return aRes2.y > aSeuilAccept;
         }

    private :
         std::vector<Pt2di> mVPt;
         int mNbPts;
};








#endif//  _CPTOFCORREL_H_

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
