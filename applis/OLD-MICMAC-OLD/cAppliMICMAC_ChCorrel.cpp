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
#include "general/all.h"
#include "MICMAC.h"
#include "ext_stl/tab2D_dyn.h"


/*
   Ce fichier contient les fonctionnalites qui doivent permettre de
  choisir entre les differentes methodes de calcul de la correlation.
*/

namespace NS_ParamMICMAC
{

/****************************************************/
/*                                                  */
/*                 cPxSelector                      */
/*                                                  */
/****************************************************/

bool cPxSelector::SelectPx(int *) const
{
   return true;
}

cPxSelector::~cPxSelector()
{
}

/****************************************************/
/*                                                  */
/*              cStatPointsOfPx                     */
/*                                                  */
/****************************************************/

// Cette classe va memoriser pour une  paralaxe donnee
// des information sur l'ensemble des pixels ayant cette
// parallaxe

typedef enum
{
   eToCalcByPoint,
   eToCalcByRect,
   eToCalcBySplit,
   eCalculated
} eEtatCalc;

class cStatPointsOfPx
{
     public :
        cStatPointsOfPx()  :
             mBox(Pt2di(-1,-1),Pt2di(-1,-1)),
             mIsInit (false),
             mNbPts      (0),
             mCostCalc   (0.0)
        {
        }

        void AddPt(const Pt2di & aP)
        {
            mNbPts++;
            if (mIsInit)
            {
                mBox._p0.SetInf(aP);
                mBox._p1.SetSup(aP);
            }
            else
            {
                mIsInit = true;
                mBox._p0 = aP;
                mBox._p1 = aP;
            }
        }
        void MakeStdBox()
        {
             mBox._p1 += Pt2di(1,1);
        }
        void AddStat(const cStatPointsOfPx & aStat,bool WithCostCalc)
        {
            mNbPts += aStat.mNbPts;
            if (mIsInit)
            {
                mBox._p0.SetInf(aStat.mBox._p0);
                mBox._p1.SetSup(aStat.mBox._p1);
            }
            else
            {
                mIsInit = true;
                mBox = aStat.mBox;
            }
            if (WithCostCalc)
            {
               mCostCalc += aStat.mCostCalc;      
            }
        }

        INT NbPts() const {return mNbPts;}
        const Box2di & Box() {return mBox;}
        double & Cost() {return mCostCalc;}
        eEtatCalc &  StatCalc() {return  mStatCalc;}

     private :

        Box2di    mBox;
        bool      mIsInit;
        eEtatCalc mStatCalc;
        INT       mNbPts;
        double    mCostCalc;
};
typedef cElBoxTab2DResizeableCreux<cStatPointsOfPx> tTabStat;

/****************************************************/
/*                                                  */
/*              cFusionStatFils                     */
/*              cDownSetCostCorrel                  */
/*              cSetCostCorrelHerited               */
/*                                                  */
/****************************************************/

//   classes utilisees pour le parcour de cNodeAnalysePx


/*
       cFusionStatFils

   Permet d'initialiser le "pere" en
   parcourant les "fils".
*/

class cFusionStatFils
{
   public :
      cFusionStatFils(tTabStat & aPere) :
         mPere (aPere)
      {
      }

      void operator()(cStatPointsOfPx & aNFils,Pt2di aP)
      {
          mPere.Get(aP).AddStat(aNFils,true);
      }
  private :
    tTabStat & mPere;
};

/*
       cDownSetCostCorrel

   Fixe les cout des fils "terminaux".
*/

class cDownSetCostCorrel
{
   public :
      cDownSetCostCorrel(const cAppliMICMAC &  anAppli) :
         mAppli (anAppli)
      {
      }

      void operator()(cStatPointsOfPx & aSt,Pt2di)
      {
          aSt.MakeStdBox();
          REAL aCostByPt = mAppli.CostCalcCorrelPonctuel(aSt.NbPts());
          REAL aCostByR  = mAppli.CostCalcCorrelRectang(aSt.Box(),aSt.NbPts());

          if (aCostByPt<aCostByR)
          {
              aSt.Cost() = aCostByPt;
              aSt.StatCalc() = eToCalcByPoint;
          }
          else
          {
              aSt.Cost() = aCostByR;
              aSt.StatCalc() = eToCalcByRect;
          }
      }
  private :
    const cAppliMICMAC &  mAppli;
};

/*
       cSetCostCorrelHerited

   Fixe les cout des fils non-"terminaux".
*/

class cSetCostCorrelHerited
{
   public :
      cSetCostCorrelHerited
      (
            const cAppliMICMAC &  anAppli
      ) :
         mAppli (anAppli)
      {
      }
      void operator()(cStatPointsOfPx & aSt,Pt2di)
      {
          REAL aCostByR  = mAppli.CostCalcCorrelRectang(aSt.Box(),aSt.NbPts());
          if (aCostByR < aSt.Cost())
          {
              aSt.Cost() = aCostByR;
              aSt.StatCalc() = eToCalcByRect;
          }
          else
          {
              aSt.StatCalc() = eToCalcBySplit;
          }
      }

   private :
      const cAppliMICMAC &    mAppli;
};

inline void InitPxFromPt(int * aVPx,const Pt2di & aPtPx)
{
   aVPx[0] = aPtPx.x;
   aVPx[1] = aPtPx.y;
}

class cNodeMakeCalcCorrel
{
   public :
      cNodeMakeCalcCorrel
      (
            cAppliMICMAC &  anAppli
      ) :
         mAppli (anAppli)
      {
      }
      void operator()(cStatPointsOfPx & aSt,Pt2di aPtPx)
      {
          switch (aSt.StatCalc())
          {
              case eToCalcByPoint :
                   // Ne fait  rien sera traite dans une passe
                   // finale de parcour / par le terrain
              break;

              case eToCalcByRect :
                   int aVPx[theDimPxMax];
                   InitPxFromPt(aVPx,aPtPx);
                   mAppli.CalcCorrelByRect(aSt.Box(),aVPx);
                   aSt.StatCalc() = eCalculated;
              break;

              case eToCalcBySplit :
                   // Ne fait  rien sera traite dans la descendance
              break;

              case eCalculated :
                   // Ne fait  rien , a deja ete fait
              break;

          }
      }
   private :
      cAppliMICMAC &  mAppli;
};

class cHeritCalculated
{
   public :
      cHeritCalculated(tTabStat & aPere) :
         mPere (aPere)
      {
      }

      void operator()(cStatPointsOfPx & aNFils,Pt2di aP)
      {
          if (mPere.Get(aP).StatCalc()==eCalculated)
             aNFils.StatCalc() = eCalculated;
      }
  private :
    tTabStat & mPere;
};



/****************************************************/
/*                                                  */
/*              cNodeAnalysePx                      */
/*                                                  */
/****************************************************/

class cNodeAnalysePx : public cPxSelector
{
     public :
          friend class cSetCostCorrelFromFils;

          cNodeAnalysePx
          (
               Box2di aBoxTer,
               cLoadTer & aLT,
               const  cAppliMICMAC &
           );
          ~cNodeAnalysePx();

         void MakeCalc(cLoadTer &,cAppliMICMAC &);
         bool SelectPx(int *) const;

     private :
          typedef cNodeAnalysePx * tPtrThis;

          bool       mSplited;
          Box2di     mBoxTer;
          tTabStat * mTabStat;
          tPtrThis   mFils[4];

      
          
};

bool cNodeAnalysePx::SelectPx(int * aPx) const
{
    Pt2di aPtPx(aPx[0],aPx[1]);
    return mTabStat->Get(aPtPx).StatCalc() == eToCalcByPoint;
}
  
cNodeAnalysePx::~cNodeAnalysePx()
{
   for (int aK=0 ; aK<4 ; aK++)
       delete mFils[aK];
   delete mTabStat;
}




cNodeAnalysePx::cNodeAnalysePx
(
    Box2di                aBoxTer,
    cLoadTer &            aLT,
    const cAppliMICMAC &  anAppli
) :
   mSplited (
                   (! anAppli.GetCurCaracOfDZ()->HasMasqPtsInt())
                && (anAppli.CurSurEchWCor() == 1)
                && ElSquare(anAppli.SzMinDecomposCalc().Val())<aBoxTer.surf()
                && (aBoxTer.hauteur() > 1)
                && (aBoxTer.largeur() > 1)
             ),
   mBoxTer  (aBoxTer),
   mTabStat (0)
{
   if (mSplited)
   {
       Box2di::QBox aTBox;
       aBoxTer.QSplit(aTBox);

       // Initialise les fils et en induit la taille de
       // la boite
       Box2di aBoxPx;
       for (int aK=0 ; aK<4 ; aK++)
       {
           mFils[aK]= new cNodeAnalysePx(aTBox[aK],aLT,anAppli);
           Box2di aKBoxPx = mFils[aK]->mTabStat->Box();
           if (aK==0)
              aBoxPx = aKBoxPx;
           else
              aBoxPx = Sup(aBoxPx,aKBoxPx);
       }
       mTabStat = new tTabStat(aBoxPx,false);


      // Aggrege les donnees du fils , a l'issue
      // le noeud a "aggrege", le nb de point,
      // le rect englob, le cout de calcul de la correl
       for (int aK=0 ; aK<4 ; aK++)
       {
           cFusionStatFils aFus(*mTabStat);
           ParseElT2dRC(*(mFils[aK]->mTabStat),aFus);
       }
       // Decide si le calcul se fait par rect ou par descente
       // suppose que le cout d'heritage est initialise (fait
       // dans "cFusionStatFils")
       cSetCostCorrelHerited aSCCH(anAppli);
       ParseElT2dRC(*mTabStat,aSCCH);
   }
   else
   {
       for (int aK=0 ; aK<4 ; aK++)
          mFils[aK] =0;

       // Initialisation du tableau indexe par
       // les paralaxes

       int aPxMin[theDimPxMax] ={0,0};
       int aPxMax[theDimPxMax] ={1,1};
       aLT.CalculBornesPax(aBoxTer,aPxMin,aPxMax);
       Pt2di aPMin (aPxMin[0],aPxMin[1]);
       Pt2di aPMax (aPxMax[0],aPxMax[1]);
       mTabStat = new tTabStat(Box2di(aPMin,aPMax),false);

       // Parcourt les paralaxes utilisees pour initialiser
       // les stat / paralax
       Pt2di aPTer;
       TIm2DBits<1> aMasq (aLT.ImMasqTer());
       for (aPTer.y=mBoxTer._p0.y ; aPTer.y<mBoxTer._p1.y ; aPTer.y++)
       {
           for (aPTer.x=mBoxTer._p0.x ; aPTer.x<mBoxTer._p1.x ; aPTer.x++)
           {
               if(aMasq.get(aPTer))
               {
                   aLT.GetBornesPax(aPTer,aPxMin,aPxMax);
                   Pt2di aPtPx;
                   for (aPtPx.y = aPxMin[1] ; aPtPx.y<aPxMax[1] ; aPtPx.y++)
                   {
                       for (aPtPx.x=aPxMin[0] ; aPtPx.x<aPxMax[0] ; aPtPx.x++)
                       {
                          mTabStat->Get(aPtPx).AddPt(aPTer); 
                       }
                   }
               }
           }
       }
       // Initialise le cout de calcul et son mode
       cDownSetCostCorrel aDSCC(anAppli);
       ParseElT2dRC(*mTabStat,aDSCC);
   }
}



void cNodeAnalysePx::MakeCalc
     (
         cLoadTer &            aLT,
         cAppliMICMAC & anAppli
     )
{
     // Parcourt les paralaxes pour eventuellement
     // faire le calcul des zones rectangulaires
     cNodeMakeCalcCorrel aNMC(anAppli);
     ParseElT2dRC(*mTabStat,aNMC);


     if (mSplited)
     {
        for (int aK=0 ; aK<4 ; aK++)
        {
            // Propage l'attribut Calculated au fils
             cHeritCalculated aHC(*mTabStat);
             ParseElT2dRC(*(mFils[aK]->mTabStat),aHC);

             // Appel recursif
             mFils[aK]->MakeCalc(aLT,anAppli);
        }

     }
     else
     {
        // this est passe comme parametre de selection des
        // paralaxes de DoOneBlocInterne
         anAppli.DoOneBlocInterne(*this,mBoxTer);
     }
}

/****************************************************/
/*                                                  */
/*              cAppliMICMAC                        */
/*                                                  */
/****************************************************/

void cAppliMICMAC::InitCostCalcCorrel()
{
  // Pifometre
  mCostBilin =  1.0;
  mCostPPV    = 0.5 * mCostBilin;
  mCostBicub  = 3.0 * mCostBilin;
  mCostSinCardElem =  mCostBilin;
  mCostGeom = (mAnam ? 15.0 : 2) * mCostBilin;
  mCostTabul = mCostBilin;
  //

  const cEtapeMEC & anEM = mCurEtape->EtapeMEC();

  switch(anEM.ModeInterpolation().Val())
  {
      case eInterpolPPV :
          mCostInterpol = mCostPPV;
      break;

      case eInterpolBiLin :
          mCostInterpol = mCostBilin;
      break;

      case eInterpolBiCub :
          mCostInterpol = mCostBicub;
      break;

      case eOldInterpolSinCard :
      case eInterpolSinCard :
          mCostInterpol = mCostSinCardElem * ElSquare(anEM.SzSinCard().ValWithDef(5));
      break;

      case eInterpolMPD :
      case eInterpolBicubOpt :
          mCostInterpol = mCostTabul;
      break;
  }

  mCostEc = mCostBilin * 0.5; // Pifometrique
  mCostCov = mCostEc *0.5;    // Approximatif
  mCostCorrel2Im = 2* mCostEc  + mCostCov;  // Assez precis
  mCostCorrel3Im = 3* mCostEc  + 3* mCostCov;  // Assez precis
       // Approx : 2 Passes : normal + Ec entre images
  mCostCorrel1ImOnNSsEc =  2 * mCostEc; 
  mCostCorrel1ImOnN = mCostCorrel1ImOnNSsEc + mCostEc; // Assez precis


  mCostPixelPonctuel =  mCostInterpol;
  if (mNbImChCalc==2)
     mCostPixelPonctuel += mCostCorrel2Im;
  else if (mNbImChCalc==3)
     mCostPixelPonctuel += mCostCorrel3Im;
  else
    mCostPixelPonctuel += mNbImChCalc * mCostCorrel1ImOnN;
  mCostPixelPonctuel *= mNbPtsWFixe;
  mCostPixelPonctuel += 4 * mCostGeom * mNbImChCalc;


  double aSurCostAlgoRap = 4.0;
  mCostPixelRect = mCostInterpol;
  mCostPixelPtInRect = 0.0;

  mCostPixelRect += mCostGeom * mNbImChCalc / ElSquare(1+2*mCurEtape->SzGeomDerivable());

  if (mNbImChCalc==2)
     mCostPixelRect += aSurCostAlgoRap*mCostCorrel2Im;
  else if (mNbImChCalc==3)
     mCostPixelRect += aSurCostAlgoRap*mCostCorrel3Im;
  else
  {
      mCostPixelRect +=  mNbImChCalc * mCostEc * aSurCostAlgoRap;
      mCostPixelPtInRect += mNbImChCalc * mCostCorrel1ImOnNSsEc * mNbPtsWFixe;
   }

/*
std::cout <<  "mNbPtsWFixe " << mNbPtsWFixe << "\n";
std::cout <<  "mCostCorrel1ImOnNSsEc  " << mCostCorrel1ImOnNSsEc << "\n";
std::cout <<  "mNbImChCalc  " << mNbImChCalc << "\n";

std::cout << "C Pontc " << mCostPixelPonctuel << "\n";
std::cout << "C R1    " << mCostPixelRect << "\n";
std::cout << "C R2    " << mCostPixelPtInRect << "\n";
*/
}

double cAppliMICMAC::CostCalcCorrelPonctuel(int aNb) const
{
   /*
   */
   if (mCurForceCorrelByRect)
      return 1e20;
   return mCostPixelPonctuel * aNb;
}

double cAppliMICMAC::CostCalcCorrelRectang(Box2di aBox,int aNbPts) const
{
   if (mCurForceCorrelPontcuelle)
       return 1e20;
   // if (mCurForceCorrelByRect) return 0;
   return   mCostPixelPtInRect * aNbPts
          +   mCostPixelRect 
            * (aBox._p1.x-aBox._p0.x + 2.0*mPtSzWMarge.x)
            * (aBox._p1.y-aBox._p0.y + 2.0*mPtSzWMarge.y);

}


void cAppliMICMAC::ChoixCalcCorrByQT(Box2di aBoxTer)
{
     ElTimer aChrono;

     cNodeAnalysePx * aNode = new cNodeAnalysePx(aBoxTer,*mLTer,*this);
     aNode->MakeCalc(*mLTer,*this);
     delete aNode;
// cout << "CC " << aChrono.uval() << "\n";
}

};

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
