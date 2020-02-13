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

namespace NS_ParamMICMAC
{

static double Cor2Cost(eModeDynamiqueCorrel aDyn,double aCorrel) 
{
   aCorrel = ElMax(-1.0,ElMin(1.0,aCorrel));

   switch (aDyn)
   {
       case eCoeffCorrelStd :
            return 1-aCorrel;
       break;

       case eCoeffAngle :
            return acos(aCorrel) / (PI/2);
       break;
   }
   ELISE_ASSERT(false,"cStatOneClassEquiv::Cout");
   return 0;
}

static double Cost2Cor(eModeDynamiqueCorrel aDyn,double aCost)
{
   switch (aDyn)
   {
       case eCoeffCorrelStd :
            return 1-aCost;
       break;

       case eCoeffAngle :
            return cos((PI/2)* aCost);
       break;
   }
   ELISE_ASSERT(false,"cStatOneClassEquiv::Cout");
   return 0;
}


/********************************************/
/*                                          */
/*        cStat1Distrib                     */
/*                                          */
/********************************************/

cStat1Distrib::cStat1Distrib(int aNbV,int * aPtInEc,int aNbVIn,double aRatioIn) :
   mVData   (aNbV,-1.0),
   mNbV     (aNbV),
   mVals    (&mVData[0]),
   mPtInEc  (aPtInEc),
   mNbVIn   (aNbVIn),
   mRatioIn (aRatioIn)
{
}

REAL cStat1Distrib::CoeffCorrel2Dist(const cStat1Distrib & aD2,REAL anEps) const
{
  RMat_Inertie aMat;

  double * V2 =aD2.mVals;
  for (int aK=0; aK<mNbV ; aK++)
  {
      aMat.add_pt_en_place(mVals[aK],V2[aK]);
  }
  return aMat.correlation(anEps);
}

// Pour, eventuellement gagner du temps, verifier
// les equivalences entre les formules de correlation
// et (surtout) leur implementation

REAL cStat1Distrib::CoeffCorrel3Dist
     (
           const cStat1Distrib & aD2,
           const cStat1Distrib & aD3,
           REAL anEps
     ) const
{

  double * aVec1 =mVals;
  double * aVec2 =aD2.mVals;
  double * aVec3 =aD3.mVals;

  double aS1=0,aS2=0,aS3=0;
  double aS11=0,aS22=0,aS33=0;
  double aS12=0,aS13=0,aS23=0;

  for (int aK=0; aK<mNbV ; aK++)
  {
      double aV1 = aVec1[aK];
      double aV2 = aVec2[aK];
      double aV3 = aVec3[aK];

      aS1 += aV1;
      aS2 += aV2;
      aS3 += aV3;

      aS11 += ElSquare(aV1);
      aS22 += ElSquare(aV2);
      aS33 += ElSquare(aV3);

      aS12 += aV1 * aV2;
      aS13 += aV1 * aV3;
      aS23 += aV2 * aV3;
  }

  aS1 /= mNbV;
  aS2 /= mNbV;
  aS3 /= mNbV;

  aS11 = aS11/mNbV-ElSquare(aS1);
  aS22 = aS22/mNbV-ElSquare(aS2);
  aS33 = aS33/mNbV-ElSquare(aS3);

  aS12 = aS12/mNbV-aS1*aS2;
  aS13 = aS13/mNbV-aS1*aS3;
  aS23 = aS23/mNbV-aS2*aS3;


  aS12 /= sqrt(ElMax(anEps,aS11*aS22));
  aS13 /= sqrt(ElMax(anEps,aS11*aS33));
  aS23 /= sqrt(ElMax(anEps,aS22*aS33));

  return (aS12+aS13+aS23) / 3.0;
}



void cStat1Distrib::NormalizeM1M2(REAL anEps)
{
    double aS1=0.0;
    double aS2=0.0;
    for (int aK=0; aK<mNbV ; aK++)
    {
        aS1 += mVals[aK];
        aS2 += ElSquare(mVals[aK]);
    }
    aS2 -= ElSquare(aS1)/mNbV;
    aS1 /= mNbV;
    aS2 = sqrt(ElMax(aS2,anEps));
    for (int aK=0; aK<mNbV ; aK++)
        mVals[aK] = (mVals[aK]-aS1)/aS2;
}


/********************************************/
/*                                          */
/*        cStatOneClassEquiv                */
/*                                          */
/********************************************/

cStatOneClassEquiv::cStatOneClassEquiv
(
    const cAppliMICMAC & anAppli,
    INT aNbDistrib,
    const std::vector<int> & aVPtInEc,
    const std::vector<int> & aVIndiceOK
) :
   mAppli      (anAppli),
   mNbV        ((int) aVPtInEc.size()),
   mDefCorr    (mAppli.DefCorrelation().Val()),
   mEpsCorr    (mAppli.EpsilonCorrelation().Val()),
   mDynCorr    (mAppli.CurEtape()->EtapeMEC().DynamiqueCorrel().Val()),
   mAggregCorr (mAppli.CurEtape()->EtapeMEC().AggregCorr().Val()),
   mVPtInEc    (aVPtInEc),
   mPtInEc     (&(mVPtInEc[0])),

   mVIndOK      (aVIndiceOK),
   mIndOK       (&(mVIndOK[0])),
   mNbOK        (mVIndOK.size())

{
   mNbVIn =0;
   for (int aK=0 ; aK<mNbV ; aK++)
       mNbVIn += (mPtInEc[aK] !=0);
   mRatioIn =  mNbVIn/ double(mNbV);
   // JE COMPREND PAS TROP POURQUOI ... MAIS APPAREMMENT ...
   mRatioIn = pow(mRatioIn,1.5);

   // std::cout << "RATIO IN " << mRatioIn << "\n";

   // Adaptation 100% empirique pour essayer de tenir compte d'effet bizare de
   // sur la dynamique des coeff de correl , a voir 
   if (anAppli.AdapteDynCov().IsInit())
   {
     const cAdapteDynCov & anADC = anAppli.AdapteDynCov().Val();

	 double aSzW  = sqrt((double)mNbV);
     double aCovTheo = anADC.CovLim().Val() + anADC.TermeDecr().Val() / aSzW;

     mRatioIn *= aCovTheo / anADC.ValRef().Val();

/*
     static bool isFirst = true;
     if (isFirst)
     {
         std::cout << "!!!!   Empirik sur mRatioIn !!! \n";
         isFirst = false;
     }
     double aNb = round_ni(sqrt(1/mRatioIn));
     double aCoeff = 10*aNb/ 35+aNb;
      mRatioIn /= aCoeff;
*/
     
       // std::cout << "ST = " << aST/aNbS << " " <<  ( aST/aNbS) * (10*aNb/(10+aNb)) << "\n";
   }


   for (int aK=0 ; aK<aNbDistrib ; aK++)
   {
       mVDistr.push_back(new cStat1Distrib(mNbV,mPtInEc,mNbVIn,mRatioIn));
       mVData.push_back(mVDistr.back()->Vals());
   }
   mData = &mVData[0];
}

cStatOneClassEquiv::~cStatOneClassEquiv()
{
    DeleteAndClear(mVDistr);
}

void cStatOneClassEquiv::Clear()
{
    mKCurDist =0;
}

cStat1Distrib * cStatOneClassEquiv::NextDistrib()
{
   return mVDistr[mKCurDist++];
}

REAL cStatOneClassEquiv::Cov() const
{
  REAL aSigmaTot = 0.0;

  for (int aIndPix=0; aIndPix<mNbOK ; aIndPix++)
  {
      int aKPix = mIndOK[aIndPix];
      {
         REAL aS1=0.0;
         REAL aS2=0.0;
         for (INT aKDist=0; aKDist<mKCurDist ; aKDist++)
         {
            double aV =  mData[aKDist][aKPix];
            aS1 += aV;
            aS2 += ElSquare(aV);
         }
         aSigmaTot += aS2-ElSquare(aS1)/mKCurDist;
      }
  }
/*
  for (int aKPix=0; aKPix<mNbV ; aKPix++)
  {
      if (mPtInEc[aKPix])
      {
         REAL aS1=0.0;
         REAL aS2=0.0;
         for (INT aKDist=0; aKDist<mKCurDist ; aKDist++)
         {
            double aV =  mData[aKDist][aKPix];
            aS1 += aV;
            aS2 += ElSquare(aV);
         }
         aSigmaTot += aS2-ElSquare(aS1)/mKCurDist;
     }
  }
*/

// Ne pas supprimer, a activer pour stat
if (0)
{
    static double aST=0,aNbS=0;

   aNbS++;
   aST += aSigmaTot /  ((mKCurDist-1) *mRatioIn) ;

    if ((round_ni(aNbS)%20000)==0)
    {
     // double aNb = round_ni(sqrt(1/mRatioIn));
     
       std::cout << "ST = " << aST/aNbS <<  "\n";

    }
}

  // std::cout << aSigmaTot << " " << mKCurDist << " " << mRatioIn << "\n";
      
  return aSigmaTot /  ((mKCurDist-1) *mRatioIn);
}

REAL cStatOneClassEquiv::MultiCorrelByCov() const
{
  return 1-Cov() ;
}


int cStatOneClassEquiv::NbCurDist() const
{
    return mKCurDist;
}

//REAL cStat1Distrib::CoeffCorrel2Dist(const cStat1Distrib & aD2,REAL anEps) const


REAL cStatOneClassEquiv::CoeffInfoMut() const
{
  // cout<<"cStatOneClassEquiv::CoeffInfoMut() \n" ;
  // trouver le nb de bin pertinent : pour l'instant en dur = 9 * 9 
  //(val pour 2 images 8bits)

  int nb_bins=9;
  int nb_niv=256;

  std::vector<std::vector<int> > histo2d(nb_bins,std::vector<int>(nb_bins,0));

  for(int aKPix=0; aKPix<mNbV ; aKPix++)
    {
      if (mPtInEc[aKPix])
	{
	  //  cout<<"mData[0][aKPix] "<<mData[0][aKPix]<<" mData[1][aKPix] "<< mData[1][aKPix]; 
	  int i = int (mData[0][aKPix]/(double(nb_niv)/double(nb_bins)));
	  int j = int (mData[1][aKPix]/(double(nb_niv)/double(nb_bins)));
	  // cout<< " i "<<i<<" j "<<j<<"\n";
	  histo2d[i][j]++;
	}
    }

//   for(int i=0; i<nb_bins; i++)
//     for(int j=0;j<nb_bins; j++)
//       cout<<"histo2d["<<i<<"]["<<j<<"] "<<histo2d[i][j]<<"\n";

  // calculer histo_2d : matrice 9 * 9 dont la somme des coeff est égale
  // au nb de pixels dans la fenêtre : (2N+1)*(2N+1) (N = taille de la 1/2 fenetre)
  
  // matrice de double hist2d_norm = histo_2d/(2N+1)*(2N+1)
  std::vector<std::vector<double> > histo2d_norm(nb_bins,std::vector<double>(nb_bins,0.));
  for(int i=0; i<9; i++)
    for(int j=0; j<9; j++)
      histo2d_norm[i][j]=histo2d[i][j]/double(mNbV);

  std::vector<double> pim1(nb_bins,0.),pim2(nb_bins,0.);
  // vecteur de double pim1 et pim2 de taille n_bins
  // calculer les densites de proba marginales
  for(int i=0; i<nb_bins; i++) 
    for(int j=0 ; j<nb_bins; j++)
      {
	pim1[i]+=histo2d_norm[i][j];
	pim2[j]+=histo2d_norm[i][j];
      }
  // calculer la formule finale
  double num=0., denom=0.;
  for(int i=0; i<nb_bins; i++)
    for(int j=0; j<nb_bins; j++)
      {
	denom+=histo2d_norm[i][j]*log(histo2d_norm[i][j]);
	num+=histo2d_norm[i][j]*log(pim1[i]*pim2[j]);
      }
  return num/denom;
}


REAL cStatOneClassEquiv::CoeffCorrelIm1Maitre() const
{
   if (mKCurDist<2)
      return mDefCorr;
   double aRes=0.0;
   for (int aK=1 ;aK<mKCurDist ; aK++)
   {
       aRes+= mVDistr[0]->CoeffCorrel2Dist(*mVDistr[aK],mEpsCorr);
   }
   return aRes / (mKCurDist-1);
}



REAL cStatOneClassEquiv::CoeffCorrelMaxIm1Maitre() const
{
   double aRes=mDefCorr;
   for (int aK=1 ;aK<mKCurDist ; aK++)
   {
       ElSetMax(aRes,mVDistr[0]->CoeffCorrel2Dist(*mVDistr[aK],mEpsCorr));
   }
   return aRes ;
}


REAL cStatOneClassEquiv::CoeffCorrelSymetrique() const
{
   if (mKCurDist<2)
      return mDefCorr;

   if ( (mNbV==mNbVIn)   && (! mSomsMade))
   {
      if (mKCurDist==2) 
      {
          ELISE_ASSERT
          (
              ! mSomsMade,
              "Correl a 2 image, n'utilise pas mSomsMade"
          );
          return mVDistr[0]->CoeffCorrel2Dist(*mVDistr[1],mEpsCorr);
      }

      if (mKCurDist==3)
      {
          ELISE_ASSERT
          (
              ! mSomsMade,
              "Correl a 3 image, n'utilise pas mSomsMade"
          );
          // return  mVDistr[0]->CoeffCorrel3Dist(*mVDistr[1],*mVDistr[2],mEpsCorr);
          return   mVDistr[0]->CoeffCorrel3Dist(*mVDistr[1],*mVDistr[2],mEpsCorr);
      }
   }

   if (! mSomsMade)
   {
      for (int aK=0 ; aK<mKCurDist ; aK++)
      {
        mVDistr[aK]->NormalizeM1M2(mEpsCorr);
      }
   }
   return MultiCorrelByCov();
}

REAL cStatOneClassEquiv::CoeffCorrelation() const
{
   switch (mAggregCorr)
   {
      case eAggregSymetrique :
           return CoeffCorrelSymetrique();
      break;

      case eAggregIm1Maitre :
           return CoeffCorrelIm1Maitre();
      break;

      case eAggregMaxIm1Maitre :
           return CoeffCorrelMaxIm1Maitre();
      break;


      case eAggregInfoMut : 
      {
           // cout<<"Choix Info Mut dans CoeffCorrelation()\n";
           return CoeffInfoMut();
      }
      break;
   }
   ELISE_ASSERT(false,"cStatOneClassEquiv::CoeffCorrelation");
   return 0;
}

REAL cStatOneClassEquiv::Cout() const
{
   return CorrelToCout(CoeffCorrelation());
}


double cStatOneClassEquiv::CorrelToCout(double aCorrel) const
{
   return Cor2Cost(mDynCorr,aCorrel);
}



/********************************************/
/*                                          */
/*           cStatGlob                      */
/*                                          */
/********************************************/

cStatGlob::cStatGlob
     (
          const cAppliMICMAC & anAppli,
          const std::vector<int> & aVPtInEc,
          const std::vector<int> & aVIndiceOK,
          const std::vector<Pt2di> & aVPtOK

     ) :
       mAppli      (anAppli),
       mVPtInEc    (aVPtInEc),
       mVIndOK     (aVIndiceOK),
       mIndOk      (&(mVIndOK[0])),
       mVPtOK      (aVPtOK),
       mPtsOK      (&(mVPtOK[0])),
       mNbOK       (mVPtOK.size()),
       mNbClass    (0),
       mDefCorr    (mAppli.DefCorrelation().Val()),
       mEpsCorr    (mAppli.EpsilonCorrelation().Val()),
       mDynCorr    (mAppli.CurEtape()->EtapeMEC().DynamiqueCorrel().Val()),
       mIsFull     (mVIndOK.size()==mVPtInEc.size())
{
}


cStatGlob::~cStatGlob ()
{
   DeleteAndClear(mStatClass);
}

void cStatGlob::InitSOCE()
{
    for (int aK=0 ; aK<mNbClass ; aK++)
    {
        mStatClass.push_back(new cStatOneClassEquiv(mAppli,mCardOfClass[aK],mVPtInEc,mVIndOK));
    }
}

void cStatGlob::AddVue(cPriseDeVue & aPDV) 
{
    int anInd = IndFind(mNamesClass,aPDV.NameClassEquiv());

    if (anInd==-1)
    {
       anInd = mNbClass;
       mNbClass++;
       mNamesClass.push_back(aPDV.NameClassEquiv());
       mCardOfClass.push_back(0);
       // mStatClass.push_back(new);
    }   

/*
    std::cout << anInd << " " 
              << aPDV.NameClassEquiv() << " "  
              << aPDV.Name() <<  "\n";
*/
    aPDV.NumEquiv() = anInd;
    mCardOfClass[anInd]++;
}


void  cStatGlob::Clear()
{
    for (int aK=0 ; aK<mNbClass ; aK++)
        mStatClass[aK]->Clear();
}


double cStatGlob::CorrelToCout(double aCorrel) const
{
   return Cor2Cost(mDynCorr,aCorrel);
}

double cStatGlob::Cout2Correl(double aCost) const
{
   return Cost2Cor(mDynCorr,aCost);
}


REAL cStatGlob::Cout() const
{
   return CorrelToCout(CoeffCorrelation());
}


REAL cStatGlob::CoeffCorrelation() const
{
    double aSP=0,aSC=0;

    for (int aK=0 ; aK<mNbClass ; aK++)
    {
         double aC = mStatClass[aK]->CoeffCorrelation();
         double aP = (mStatClass[aK]->NbCurDist() - 1);
         if (aP > 0)
         {
            aSP += aP;
            aSC += aC * aP;
         }
    }
    if (aSP  > 0)
    {
       return aSC / aSP;
    }
    return mDefCorr;
}


cStat1Distrib * cStatGlob::NextDistrib(const cPriseDeVue & aPDV)
{
   return mStatClass[aPDV.NumEquiv()]->NextDistrib();
}

void cStatGlob::SetSomsMade(bool aSSM)
{
    for (int aK=0 ; aK<mNbClass ; aK++)
        mStatClass[aK]->SomsMade() = aSSM;
}

};

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a  l'utilisation,  a  la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
