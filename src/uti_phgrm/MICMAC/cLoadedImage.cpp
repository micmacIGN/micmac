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

#include "../src/uti_phgrm/MICMAC/MICMAC.h"
// Pour des tests
#include "cOrientationRTO.h"

double aSeuilComSupInf = 0.99;
double aSeuilGlobSupInf = 0.5;

double aSeuilComSupRect  = 1-1e-5;
double aSeuilGlobSupRect = 1-1e-5;


template <class TypeEl>
class cTplLoadedImage : public cLoadedImage
{
   public:
      bool StatIm1(cLineariseProj &,double &aS1,double & aS11,double * aTabV1) ;
      bool StatIm2(cLineariseProj &,double * aTabV1,double &aS2,double & aS22,double & aS12);

      typedef typename El_CTypeTraits<TypeEl>::tBase tBase;
      typedef Im2D<TypeEl,tBase>            tIm;
      typedef TIm2D<TypeEl,tBase>            tTIm;

      cTplLoadedImage
      (
         const cAppliMICMAC & anAppli,
         const cPriseDeVue & aPDV,
         const cGeomImage & aGeom,
         Box2di aBox,
         Pt2di  aSzMaxGT,
         cInterfModuleImageLoader * aIMIL,
         int                        aDZ,
         Fonc_Num                   aFMasq,
	 bool                       IsFirstLoaded
      );
      ~cTplLoadedImage();
      virtual void GetVals(const Pt2dr * aP,double *,int aNb) const;
      void LoadImInGeomTerr(Box2di aBox,int * aPx,bool IsFirstLoaded);
      void PostLoadImInGeomTerr(Box2di aBox);

      tIm ImTer() {return mImTer;}
      Im2D<tBase,tBase>  ImSomITer()  {return mImSomITer;}
      Im2D_REAL8 ImSomI2Ter()         {return mImSomI2Ter;}

      void AddToStat(cStatGlob &,Pt2di aPTer);

      void Instance_CalcCorrelRapide 
                 (const Box2di &,cTplLoadedImage<TypeEl> &,bool aModeAdd);
      void CalcCorrelRapide (const Box2di &,cLoadedImage &);

      void  NCalcCorrelRapide(const Box2di &,tContPDV&);
      void  CalcFenSpec(const Box2di & aBoxTer,tContPDV&);
      void  OneCorrelSpec(const Box2di & aBoxCorr,cTplLoadedImage<TypeEl>&);

      // void CalcRapCorrelIm1Maitre(const Box2di & aBox,tContPDV & aCont);
      typedef typename std::vector<cTplLoadedImage<TypeEl> *>  tVecTPtr;
      tVecTPtr ToCurType(tContPDV&);

   private:
      float *** DataFloatIm();
      float **  DataFloatLinIm();
      Im2D_REAL4 ** FloatIm();
      Im2D_REAL4 * FirstFloatIm();

      void PrecalcRect(const Box2di & aBoxCorrTer);
      void FiltrageWRect(const Box2di & aBoxCorrTer,Im2D_REAL8 anIm);
      void FiltrageWExp(const Box2di & aBoxCorrTer,Im2D_REAL8 anIm);
      void FiltrageW(const Box2di & aBoxCorrTer,Im2D_REAL8 anIm);
      void Filtrage1DRect(double * aLine,int aInd0,int aInd1,double aSzW,double * aBuf);

      eTypeWinCorrel     mTypeWC;
      tIm   mIm;
      TypeEl ** mDataIm;
      cInterpolateurIm2D<TypeEl> * mInterpol;

      tIm   mImTer;
      tTIm  mTImTer;
      Im2D_REAL8          mImSomITer;
      TIm2D<REAL8,REAL8>  mTImSomITer;
      Im2D_REAL8         mImSomI2Ter;
      TIm2D<REAL8,REAL8> mTImSomI2Ter;

      Pt2di              mSzImSpec;
      Im2D_REAL8         mImSomMasq;
      TIm2D<REAL8,REAL8> mTImSomMasq;


      int                mNbPtsW;
      Im1D<REAL8,REAL8>  mBufL;
      Im1D<REAL8,REAL8>  mBufL2;

      double             mEpsC;

      Pt2di mP0TerCorMarge;
      Pt2di mP1TerCorMarge;
      Box2di mBoxTerCorMarge;

      Pt2di mP0TerCor;
      Pt2di mP1TerCor;
      Box2di mBoxTerCor;
      Video_Win * mWPC;

      bool  mIsFirstLIGT;

      Fonc_Num FIm() 
      {
          return mIm.in_proj();
      }  

      static const eIFImL_TypeNumerique theTypeEl;

};

template<> const  eIFImL_TypeNumerique cTplLoadedImage<REAL4>::theTypeEl=eFloat;

template <class TypeEl>
cTplLoadedImage<TypeEl>::~cTplLoadedImage()
{
   delete mInterpol;
}


template <class TypeEl> float *** cTplLoadedImage<TypeEl>::DataFloatIm() 
{
   // return (float **)  mIm.data();
   return & (mVDataIm[0]);
}

template <class TypeEl> float ** cTplLoadedImage<TypeEl>::DataFloatLinIm() 
{
   // return (float *)  mIm.data_lin();
   return & (mVDataLin[0]);
}

template <class TypeEl> Im2D_REAL4 ** cTplLoadedImage<TypeEl>::FloatIm()
{
/*
  cTplLoadedImage<TypeEl> * aNI = const_cast<cTplLoadedImage<TypeEl> *>(this);
  return (Im2D_REAL4 *)  &(aNI->mIm);
  return (Im2D_REAL4 *)  &(mIm);
*/
   return  & (mVIm[0]);
}

template <class TypeEl> Im2D_REAL4 * cTplLoadedImage<TypeEl>::FirstFloatIm()
{
  ELISE_ASSERT(mIm.TypeEl()==GenIm::real4,":DataFloatIm");
/*
  cTplLoadedImage<TypeEl> * aNI = const_cast<cTplLoadedImage<TypeEl> *>(this);
  return (Im2D_REAL4 *)  &(aNI->mIm);
  return  & (mVIm[0]);
*/
  return (Im2D_REAL4 *)  &(mIm);
}





template <class TypeEl,class tBase> cInterpolateurIm2D<TypeEl>  * InterpoleOfEtape(const cEtapeMEC & anEt,TypeEl *,tBase *)
// template <class TypeEl,class tBase> cInterpolateurIm2D<TypeEl>  InterpoleOfEtape(const cEtapeMEC & anEt)
{
   double aCoef3 = anEt.CoefInterpolationBicubique().ValWithDef(-0.5);
   switch(anEt.ModeInterpolation().Val())
   {
       case eInterpolPPV :
         return new cInterpolPPV<TypeEl>;
       break;

       case eInterpolBiLin :
// std::cout << "BIIIIIIIIIIIIILLLINNNNN \n";
         return  new cInterpolBilineaire<TypeEl>;
       break;
       case eInterpolBiCub :
         return new cInterpolBicubique<TypeEl>(aCoef3);
       break;

       case eInterpolBicubOpt :
         return new cTplCIKTabul<TypeEl,tBase>(10,8,aCoef3);
       break;

       case eInterpolMPD :
// std::cout << "MPDDDDDDDDDDDDDDD \n";
         return  new cTplCIKTabul<TypeEl,tBase>(10,8,0.0,eTabulMPD_EcartMoyen);
       break;

       case eInterpolSinCard :
       {
           double aSzK = anEt.SzSinCard().ValWithDef(5.0);
           double aSzA = anEt.SzAppodSinCard().ValWithDef(5.0);
           int aNbD = anEt.NdDiscKerInterp().ValWithDef(1000);

          cSinCardApodInterpol1D aKer(cSinCardApodInterpol1D::eTukeyApod,aSzK,aSzA,1e-4,false);

          return  new cTabIM2D_FromIm2D<TypeEl>(&aKer,aNbD,false);
          break;
      }
      case eOldInterpolSinCard :
      {
         return  new cInterpolSinusCardinal<TypeEl>(anEt.TailleFenetreSinusCardinal().Val(), anEt.ApodisationSinusCardinal().Val());
         break;
      }

   }
   ELISE_ASSERT(false,"Incoh in InterpoleOfEtape");
   return 0;

}


template <class TypeEl>
cTplLoadedImage<TypeEl>::cTplLoadedImage
(
    const cAppliMICMAC & anAppli,
    const cPriseDeVue & aPDV,
    const cGeomImage & aGeomI,
    Box2di aBox,
    Pt2di  aSzMaxGT,
    cInterfModuleImageLoader * aIMIL,
    int                        aDZ,
    Fonc_Num                   aFMasq,
    bool                       IsFirstLoaded
) :
     cLoadedImage  (anAppli,aPDV,aGeomI,aBox,aSzMaxGT,aIMIL,aDZ,aFMasq,IsFirstLoaded,theTypeEl),
     mTypeWC       (anAppli.CurTypeWC()),
     mIm           (mSzIm.x,mSzIm.y),
     mDataIm       (mIm.data()),
     mInterpol     (0),
     mImTer        (mSzCImT.x,mSzCImT.y),
     mTImTer       (mImTer),
     mImSomITer    (mSzCImT.x,mSzCImT.y),
     mTImSomITer   (mImSomITer),
     mImSomI2Ter   (mSzCImT.x,mSzCImT.y),
     mTImSomI2Ter  (mImSomI2Ter),
     mSzImSpec     ((mTypeWC==eWInCorrelFixe) ? Pt2di(1,1) : mSzCImT),
     mImSomMasq    (mSzImSpec.x,mSzImSpec.y),
     mTImSomMasq   (mImSomMasq),
     mNbPtsW       (anAppli.NbPtsWFixe()),
     mBufL         (ElMax(mSzCImT.x,mSzCImT.y)),
     mBufL2        (ElMax(mSzCImT.x,mSzCImT.y)),
     mEpsC         (anAppli.EpsilonCorrelation().Val()),
     mWPC          (0)
{


   LoadAllImCorrel(mIm,aIMIL,aDZ,aBox._p0);

   const cEtapeMEC & anEt = anAppli.CurEtape()->EtapeMEC();
   if (! anEt.FiltreImageLoc().empty())
   {
       Fonc_Num aFonc = mIm.in_proj();
       for
       (
            std::list<cSpecFitrageImage>::const_iterator itF = anEt.FiltreImageLoc().begin();
            itF != anEt.FiltreImageLoc().end();
            itF++
       )
       {
              if (
	            itF->UseIt().Val()
	            && 
		    (
                          (! itF->PatternSelFiltre().IsInit())
                       || (itF->PatternSelFiltre().Val()->Match(mPDV.Name()))
		    )
              )
              {
		    aFonc = FiltrageImMicMac(*itF,aFonc, mMasqIm.in_proj(),1.0);
              }
        }
	ELISE_COPY(mIm.all_pts(),aFonc,mIm.out());
   }


   mInterpol= InterpoleOfEtape(anEt,(TypeEl *)0,(tBase *)0);
/*
   mInterpol= InterpoleOfEtape<TypeEl,tBase>(anEt);
   double aCoef3 = anEt.CoefInterpolationBicubique().ValWithDef(-0.5);
   switch(anEt.ModeInterpolation().Val())
   {
       case eInterpolPPV :
         mInterpol= new cInterpolPPV<TypeEl>;
       break;
       case eInterpolBiLin :
// std::cout << "BIIIIIIIIIIIIILLLINNNNN \n";
         mInterpol= new cInterpolBilineaire<TypeEl>;
       break;
       case eInterpolBiCub :
         mInterpol= new cInterpolBicubique<TypeEl>(aCoef3);
       break;

       case eInterpolBicubOpt :
         mInterpol= new cTplCIKTabul<TypeEl,tBase>(10,8,aCoef3);
       break;

       case eInterpolMPD :
// std::cout << "MPDDDDDDDDDDDDDDD \n";
         mInterpol= new cTplCIKTabul<TypeEl,tBase>(10,8,0.0,eTabulMPD_EcartMoyen);
       break;

       case eInterpolSinCard :
       {
           double aSzK = anEt.SzSinCard().ValWithDef(5.0);
           double aSzA = anEt.SzAppodSinCard().ValWithDef(5.0);
           int aNbD = anEt.NdDiscKerInterp().ValWithDef(1000);

          cSinCardApodInterpol1D aKer(cSinCardApodInterpol1D::eTukeyApod,aSzK,aSzA,1e-4,false);

          mInterpol= new cTabIM2D_FromIm2D<TypeEl>(&aKer,aNbD,false);
          break;
      }
      case eOldInterpolSinCard :
      {
         mInterpol= new cInterpolSinusCardinal<TypeEl>(anEt.TailleFenetreSinusCardinal().Val(), anEt.ApodisationSinusCardinal().Val());
      }

       break;
   }
*/

   // Gestion du masque 

   // Transfert d'une eventuelle valeur conventionnelle "d'exclusion"

   // Dilatatation des valeurs exclues en fonction de la taille de
   // noyau d'interpolation
   ELISE_COPY
   (
       mMasqIm.all_pts(),
       erod_d8(mMasqIm.in(0),mInterpol->SzKernel()),
       mMasqIm.out()
   );

}

template <class TypeEl>
void cTplLoadedImage<TypeEl>::GetVals(const Pt2dr * aPts,double *aVals,int aNb) const
{
   return mInterpol->GetVals(mDataIm,aPts,aVals,aNb);
}

template <class TypeEl>
bool cTplLoadedImage<TypeEl>::StatIm2
     (
           cLineariseProj & aLineProj,
           double * aTabV1,
           double &aS2,
           double & aS22,
           double & aS12
     ) 
{
   aS2 = 0;
   aS22 = 0;
   aS12 = 0;

   while(aLineProj.Continuer())
   {
      Pt2dr aP = aLineProj.PCurIm();
      if(IsOk(aP))
      {
          double aV2 = mInterpol->GetVal(mDataIm,aP);
//  std::cout << "nnnnnnnnnnnnnnn "<< aP << " :" << aV2 << "\n";
          aS2 += aV2;
          aS22 += ElSquare(aV2);
          aS12 += *(aTabV1++) * aV2;
      }
      else
      {
//  std::cout << "nnnnnnnnnnnnnnn "<< aP << " :" << " ???????????????" << "\n";
          return false;
      }
      aLineProj.NexStep();
   }
   return true;
}



template <class TypeEl>
bool cTplLoadedImage<TypeEl>::StatIm1
     (
           cLineariseProj & aLineProj,
           double &aS1,
           double & aS11,
           double * aTabV1
     ) 
{
   aS1 = 0;
   aS11 = 0;

   while(aLineProj.Continuer())
   {
      Pt2dr aP = aLineProj.PCurIm();
      if(IsOk(aP))
      {
          double aV1 = mInterpol->GetVal(mDataIm,aP);
          aS1 += aV1;
          aS11 += ElSquare(aV1);
          *(aTabV1++) = aV1;
      }
      else
      {
          return false;
      }
      aLineProj.NexStep();
   }
   return true;
}

        //
        //    Reechantillonage en geometrie terrain  
        // + eventuel pre-calcul rapide des stats sur 
        // 1 image                                  
        //

template <class TypeEl>
void cTplLoadedImage<TypeEl>::LoadImInGeomTerr
     (
             Box2di aBox,
             int * aPx,
	     bool IsFirstLoaded
     )
{
   mIsFirstLIGT = IsFirstLoaded;
   mGeomCorrelImTer = mGeomTerAp;
   mGeomCorrelImTer.SetClip
   (
        mGeomTerAp.P0Clip() + aBox._p0-mSzPtWMarge,
        mGeomTerAp.P0Clip() + aBox._p1+mSzPtWMarge
   );
 
  TIm2D<REAL8,REAL8> aTSom1(mLT->Som1());
  TIm2DBits<1>  aTMasqGlob(mLT->MasqGlob());
   // ===========

   mDecalTer2Cor = DiscTerAppli2DiscTerCorr(Pt2di(0,0));
   mP0TerCorMarge = DiscTerAppli2DiscTerCorr(aBox._p0-mSzPtWMarge);
   mP1TerCorMarge = DiscTerAppli2DiscTerCorr(aBox._p1+mSzPtWMarge);
   mBoxTerCorMarge= Box2di(mP0TerCorMarge,mP1TerCorMarge);

   mP0TerCor  = DiscTerAppli2DiscTerCorr(aBox._p0); 
   mP1TerCor  = DiscTerAppli2DiscTerCorr(aBox._p1); 
   mBoxTerCor = Box2di(mP0TerCor,mP1TerCor);

   int aSzBloc =1+2*mCurEt->SzGeomDerivable();


//  std::cout << "SZ BLOC " << aSzBloc << "\n"; getchar();

   ELISE_ASSERT
   (
      mAppli.CurSurEchWCor() == 1,
      "Pas de gestion de CurSurEchWCor dans LoadImInGeomTerr "
   );
   cLineariseProj aLP;


   for (int anY0 = mP0TerCorMarge.y; anY0<mP1TerCorMarge.y ; anY0+=aSzBloc)
   {
       int anY1 = ElMin(mP1TerCorMarge.y,anY0+aSzBloc);
       for (int anX0 = mP0TerCorMarge.x; anX0<mP1TerCorMarge.x ; anX0+=aSzBloc)
       {
           int anX1 = ElMin(mP1TerCorMarge.x,anX0+aSzBloc);
           aLP.InitLP
           (
               Box2di(Pt2di(anX0,anY0),Pt2di(anX1-1,anY1-1)),
               aPx,
               mGeomI,
               mGeomCorrelImTer,
               *mCurEt,
               *mLT,
               mDecalTer2Cor,
               1
           );
           while (aLP.Continuer())
           {
              Pt2di  aPRasT = aLP.PCurRasTer();


              const Pt2dr & aPIm   = aLP.PCurIm();
// std::cout << aPRasT << aPIm << "\n";
              bool  isInMask  = (mTMasqIm.get(round_ni(aPIm),0) != 0);


              mTMasqImTer.oset ( aPRasT,isInMask);

              double aV = isInMask ?  mInterpol->GetVal(mDataIm,aPIm) : 0 ;
	      //std::cout << "Position : "<<aPRasT<<" Valeur : "<<aV<<std::endl; 
              mTImTer.oset(aPRasT,/*TypeEl(aV)*/aV);
	      if (IsFirstLoaded)
	         aTMasqGlob.oset(aPRasT,isInMask);
	      else
	         aTMasqGlob.oset(aPRasT,aTMasqGlob.get(aPRasT)&&isInMask);


	      if (mTypeWC != eWInCorrelFixe)
	      {
	         if ( IsFirstLoaded)
                    aTSom1.oset(aPRasT,1.0);
                 mTImSomMasq.oset(aPRasT,isInMask);
                 mTImSomITer.oset(aPRasT,aV);
                 mTImSomI2Ter.oset(aPRasT,aV*aV);
              }

              aLP.NexStep();
           }
       }
   }

    // AAAAAAAAAAAAAAAAAAA
 /*
   bool aBug = aBox.inside(Pt2di(100,100));
   if (aBug )
   {
      std::cout << "BOX " <<  aBox._p0 << aBox._p1 << "\n";
      std::cout << "IM TER " << ImTer().sz() << "\n";
      std::cout << "Masq Im " << mMasqIm.sz() << "\n";

   }
   if (aBug && (!IsFirstLoaded))
   {
std::cout <<  " BBBBBBBBBBBBBBBB \n";
Video_Win aW = Video_Win::WStd(Pt2di(500,500),1);
ELISE_COPY(ImTer().all_pts(),ImTer().in(),aW.ogray());
   getchar();
ELISE_COPY(mLT->MasqGlob().all_pts(),mLT->MasqGlob().in(),aW.odisc());
   getchar();
ELISE_COPY(aW.all_pts(),P8COL::red,aW.odisc());
ELISE_COPY(mMasqIm.all_pts(),mMasqIm.in(),aW.odisc());
   getchar();
   }
*/

}

template <class TypeEl>
void cTplLoadedImage<TypeEl>::PostLoadImInGeomTerr
     (
             Box2di aBox
     )
{
   if (mAppli.CurWSpecUseMasqGlob())
   {
        TIm2DBits<1>  aTMasqGlob(mLT->MasqGlob());
        TIm2D<REAL8,REAL8> aTSomM(mImSomMasq);
        TIm2D<REAL8,REAL8> aTSomI(mImSomITer);
        TIm2D<REAL8,REAL8> aTSomI2(mImSomI2Ter);

        Pt2di aP;
        for (aP.x = mP0TerCorMarge.x; aP.x<mP1TerCorMarge.x ; aP.x++)
        {
            for (aP.y = mP0TerCorMarge.y; aP.y<mP1TerCorMarge.y ; aP.y++)
            {
                if (aTMasqGlob.get(aP)==0)
                {
                    aTSomM.oset(aP,0.0);
                    aTSomI.oset(aP,0.0);
                    aTSomI2.oset(aP,0.0);
                }
            }
        }
   }

   // Si algo de base : pre-calcul des moyennes et moy quad
   if (!mAppli.UseAlgoSpecifCorrelRect())
   {
      PrecalcRect(Box2di(QuickTer2Cor(aBox._p0),QuickTer2Cor(aBox._p1)));
   }
   else if (mTypeWC != eWInCorrelFixe)
   {
       if (mIsFirstLIGT)
           FiltrageW(mBoxTerCorMarge,mLT->Som1());
       FiltrageW(mBoxTerCorMarge,mImSomMasq);
       FiltrageW(mBoxTerCorMarge,mImSomITer);
       FiltrageW(mBoxTerCorMarge,mImSomI2Ter);
   }
}
       /**********************************************/
       /*     Algo pour les fenetres "speciales"     */
       /**********************************************/

template <class TypeEl>
      void cTplLoadedImage<TypeEl>::Filtrage1DRect
      (
          double * aLine,
	  int aInd0,
	  int aInd1,
	  double aSzWR,
	  double * aRes
     )
{
  int aSzI = round_down(aSzWR);

  double aTot = 0.0;
  for (int aDx=-aSzI ; aDx < aSzI ; aDx++)
      aTot += aLine[aInd0+aDx];

  for (int anI = aInd0; anI <aInd1 ; anI++)
  {
      aTot += aLine[anI+aSzI];
      aRes[anI] = aTot;
      aTot -=  aLine[anI-aSzI];
  }

  double aFrac = aSzWR-aSzI;
  if (aFrac != 0)
  {
      aSzI++;
      for (int anI = aInd0; anI <aInd1 ; anI++)
          aRes[anI] += aFrac * ( aLine[anI-aSzI]+aLine[anI+aSzI]);
  }

  memcpy(aLine+aInd0,aRes+aInd0,(aInd1-aInd0)*sizeof(*aLine));
}

template <class TypeEl>
void cTplLoadedImage<TypeEl>::FiltrageWRect(const Box2di & aBoxCorrTer,Im2D_REAL8 anIm)
{
    double **  aD = anIm.data();
    double  * mBL = mBufL.data();
    double  * mBL2 = mBufL2.data();

    int anX0 = aBoxCorrTer._p0.x;
    int anX1 = aBoxCorrTer._p1.x;
    int anY0 = aBoxCorrTer._p0.y;
    int anY1 = aBoxCorrTer._p1.y;

    for (int anY=anY0 ; anY<anY1 ; anY++)
    {
        Filtrage1DRect
	(
	    aD[anY],
	    mP0TerCor.x,mP1TerCor.x,
	    mAppli.PtSzWReelle().x,
	    mBL
	);
    }

    for (int anX=anX0 ; anX<anX1 ; anX++)
    {
       for (int anY=anY0 ; anY<anY1 ; anY++)
           mBL2[anY] = aD[anY][anX];
        Filtrage1DRect
	(
	    mBL2,
	    mP0TerCor.y,mP1TerCor.y,
	    mAppli.PtSzWReelle().y,
	    mBL
	);
       for (int anY=anY0 ; anY<anY1 ; anY++)
           aD[anY][anX] =  mBL2[anY];
    }
}

// A REMPLACER PAR  "FilterExp" dans "im_tpl/algo_filter_exp.cpp"

template <class TypeEl>
void cTplLoadedImage<TypeEl>::FiltrageWExp(const Box2di & aBoxCorrTer,Im2D_REAL8 anIm)
{
    double **  aD = anIm.data();
    double  * mBL = mBufL.data();

    int anX0 = aBoxCorrTer._p0.x;
    int anX1 = aBoxCorrTer._p1.x;
    int anY0 = aBoxCorrTer._p0.y;
    int anY1 = aBoxCorrTer._p1.y;

    double aFx = mAppli.FactFenetreExp().x;
    double aFy = mAppli.FactFenetreExp().y;

    for (int anY=anY0 ; anY<anY1 ; anY++)
    {
        double * aLine = aD[anY];
	mBL[anX0] = 0;
	for (int anX = anX0+1; anX<anX1 ; anX++)
	{
	   mBL[anX] =  aFx *(mBL[anX-1] + aLine[anX-1]);
        }
	
	for (int anX = anX1-2; anX>=anX0 ; anX--)
	{
	   aLine[anX]   += aFx * aLine[anX+1];
	   aLine[anX+1] +=  mBL[anX+1];
	}
    }

    for (int anX=anX0 ; anX<anX1 ; anX++)
    {
	mBL[anY0] = 0;
	for (int anY = anY0+1; anY<anY1 ; anY++)
	{
	   mBL[anY] =  aFy *(mBL[anY-1] + aD[anY-1][anX]);
        }
	for (int anY = anY1-2; anY>=anY0 ; anY--)
	{
	   aD[anY][anX]   += aFy * aD[anY+1][anX];
	   aD[anY+1][anX] +=  mBL[anY+1];
	}
    }

}


template <class TypeEl>
void cTplLoadedImage<TypeEl>::FiltrageW(const Box2di & aBoxCorrTer,Im2D_REAL8 anIm)
{
    for (int aK=0 ; aK<mAppli.CurNbIterFenSpec() ; aK++)
    {
        switch(mTypeWC)
        {
            case eWInCorrelExp :
	         FiltrageWExp(aBoxCorrTer,anIm);
            break ;
	
            case eWInCorrelRectSpec :
	         // FiltrageWExp(aBoxCorrTer,anIm);
	         FiltrageWRect(aBoxCorrTer,anIm);
            break ;
	    default :
                 ELISE_ASSERT(false,"Incoherence in FiltrageW");
            break ;
        }
    }
}


template <class TypeEl>
    void  cTplLoadedImage<TypeEl>::OneCorrelSpec
          (
                const Box2di & aBoxCorr,
                cTplLoadedImage<TypeEl>& aIm2
          )
{
   bool aSupInf = (mTypeWC != eWInCorrelRectSpec);

   double aSeuilCom =  aSupInf ? aSeuilComSupInf : aSeuilComSupRect;
   double aSeuilGlob =  aSupInf ? aSeuilComSupInf : aSeuilComSupRect;


   Box2di aBoxFiltrage(aBoxCorr._p0-mSzPtWMarge,aBoxCorr._p1+mSzPtWMarge);

    
   TIm2D<REAL8,REAL8> aTSomI1I2(mLT->SomI1I2());
   TIm2D<REAL8,REAL8> aTSomMasqI1I2(mLT->SomMasqI1I2());

   Pt2di aP;
   int aFX0 = aBoxFiltrage._p0.x;
   int aFX1 = aBoxFiltrage._p1.x;
   int aFY0 = aBoxFiltrage._p0.y;
   int aFY1 = aBoxFiltrage._p1.y;
   for (aP.x=aFX0; aP.x<aFX1 ; aP.x++)
   {
       for (aP.y=aFY0; aP.y<aFY1 ; aP.y++)
       {
           aTSomI1I2.oset
           (
                aP,
                mTImTer.get(aP)*aIm2.mTImTer.get(aP)
           ); 
           aTSomMasqI1I2.oset
           (
                aP,
                mTMasqImTer.get(aP)*aIm2.mTMasqImTer.get(aP)
           ); 
       }
   }

   FiltrageW(aBoxFiltrage,mLT->SomI1I2());
   FiltrageW(aBoxFiltrage,mLT->SomMasqI1I2());

   TIm2D<REAL8,REAL8> aTSom1(mLT->Som1());
   TIm2D<REAL8,REAL8> aTSomCorrel(mLT->SomCorrel());
   TIm2D<REAL8,REAL8> aTSomPdsCorr(mLT->SomPdsCorr());

   int anX0 = aBoxCorr._p0.x;
   int anX1 = aBoxCorr._p1.x;
   int anY0 = aBoxCorr._p0.y;
   int anY1  = aBoxCorr._p1.y;



   for (aP.x=anX0; aP.x<anX1 ; aP.x++)
   {
       for (aP.y=anY0; aP.y<anY1 ; aP.y++)
       {
           double aNbA = mTImSomMasq.get(aP);
           double aNbB = aIm2.mTImSomMasq.get(aP);
           double aNb1 = aTSom1.get(aP);
	   double aNbAB = aTSomMasqI1I2.get(aP);
	   if (
	             (aNbAB > aSeuilCom * aNbA)
	         &&  (aNbAB > aSeuilCom * aNbB)
		 &&  (aNbAB > aSeuilGlob  * aNb1)
	      )
	   {
              double aSA =  mTImSomITer.get(aP) /aNbA;
              double aSAA = mTImSomI2Ter.get(aP) /aNbA -ElSquare(aSA);
              double aSB =  aIm2.mTImSomITer.get(aP) /aNbB;
              double aSBB = aIm2.mTImSomI2Ter.get(aP) /aNbB -ElSquare(aSB);

	      double aSAB = aTSomI1I2.get(aP) / sqrt(aNbA*aNbB) - aSA*aSB;
	      double anEct = ElMax(mEpsC,aSAA*aSBB);
	      double aCor = aSAB / sqrt(anEct);


	      aTSomCorrel.add(aP,aCor);
	      aTSomPdsCorr.add(aP,1.0);
	   }
       }
   }
}


template <class TypeEl>
void   cTplLoadedImage<TypeEl>::CalcFenSpec(const Box2di & aBox,tContPDV& aCont)
{

   tVecTPtr aVTpl = ToCurType(aCont);

   Pt2di aP0 = QuickTer2Cor(aBox._p0);
   Pt2di aP1 = QuickTer2Cor(aBox._p1);

   TIm2D<REAL8,REAL8> aTSomCorrel(mLT->SomCorrel());
   TIm2D<REAL8,REAL8> aTSomPdsCorr(mLT->SomPdsCorr());

   Pt2di aP;
   for (aP.x=aP0.x ; aP.x<aP1.x ; aP.x++)
   {
       for (aP.y=aP0.y ; aP.y<aP1.y ; aP.y++)
       {
            aTSomCorrel.oset(aP,0.0);
            aTSomPdsCorr.oset(aP,0.0);
       }
   }

  
   switch (mAppli.CurEtape()->EtapeMEC().AggregCorr().Val())
   {
       case eAggregSymetrique :
            for (int aK1=0 ; aK1<int(aVTpl.size()) ; aK1++)
            {
                for (int aK2=aK1+1 ; aK2<int(aVTpl.size()) ; aK2++)
                {
                    aVTpl[aK1]->OneCorrelSpec(Box2di(aP0,aP1),*(aVTpl[aK2]));
                }
            }
       break;
       case eAggregIm1Maitre :
       case eAggregMaxIm1Maitre :
       case eAggregMinIm1Maitre :
            for (int aK2=1 ; aK2<int(aVTpl.size()) ; aK2++)
            {
                aVTpl[0]->OneCorrelSpec(Box2di(aP0,aP1),*(aVTpl[aK2]));
            }
       break;

       default :
           ELISE_ASSERT(false,"Mode Aggregation incompatible avec FenSpec");
       break;
   }

   double aDefCor =  mAppli.DefCorrelation().Val();

   TIm2D<REAL8,REAL8> aTCorrel(mLT->ImCorrel());
   for (aP.x=aP0.x ; aP.x<aP1.x ; aP.x++)
   {
       for (aP.y=aP0.y ; aP.y<aP1.y ; aP.y++)
       {
           double aS1 = aTSomPdsCorr.get(aP); 
           double aSCor = aTSomCorrel.get(aP); 
	   double aMoyCor = (aS1>0) ? (aSCor/aS1) : aDefCor;
           aTCorrel.oset(QuickCor2Ter(aP),aMoyCor);
       }
   }
}

       /***************************************/
       /* Algo rapide pour stat a une image   */
       /***************************************/

// Contient une couple de valeur Masque + Image
template <class Type> class cMasqIm
{
    public :
       int    mMasq;
       Type   mIm;
};

// Contient le cumul du masque, de la somme de l'image
// et de la somme des carres de l'image

template <class Type> class cSomMII2
{
    public :
       cSomMII2 () :
            mSMasq (0),
            mSIm (0),
            mSI2 (0.0)
       {
       }
       void AddElem(int aSigne,const cMasqIm<Type> & aMI)
       {
           mSMasq += aSigne * aMI.mMasq;
           mSIm   += aSigne * aMI.mIm;
           mSI2   += aSigne * ElSquare(double(aMI.mIm));
       }

       void AddCumul (int aSigne,const cSomMII2<Type> & aS)
       {
           mSMasq += aSigne * aS.mSMasq;
           mSIm   += aSigne * aS.mSIm  ;
           mSI2   += aSigne * aS.mSI2  ;
       }
       
       int     mSMasq;
       Type    mSIm;
       double  mSI2;
};

template <class TypeEl> class  cSomRapideMII2
{
    public :
       typedef typename El_CTypeTraits<TypeEl>::tBase tBase;
       typedef cMasqIm<tBase>  tElem;
       typedef cSomMII2<tBase> tCumul;
       cSomRapideMII2(cTplLoadedImage<TypeEl> & aLI) :
         mSzW         (aLI.Appli().SzWFixe()),
         mNbPtsW      (aLI.Appli().NbPtsWFixe()),
         mTImTer      (aLI.ImTer()),
         mTMasqImTer  (aLI.MasqImTer()),
         mTImSomITer  (aLI.ImSomITer()),
         mTImSomI2Ter (aLI.ImSomI2Ter())
       {
       }

       void Init(const std::complex<int> & aPC,tElem & anEl)
       {
          Pt2di aP(aPC.real(),aPC.imag());
          anEl.mMasq = mTMasqImTer.get(aP);
          anEl.mIm   = mTImTer.get(aP);
       }

       void UseAggreg(const std::complex<int> & aPC,const tCumul & aCum)
       {
          Pt2di aP(aPC.real(),aPC.imag());
          mTMasqImTer.oset(aP,aCum.mSMasq==mNbPtsW);
          mTImSomITer.oset(aP,aCum.mSIm);
          mTImSomI2Ter.oset(aP,aCum.mSI2);
       }
       void  OnNewLine(int anY) {}

       
    private :
         int                  mSzW;
         int                  mNbPtsW;
         TIm2D<TypeEl,tBase>  mTImTer;
         TIm2DBits<1>         mTMasqImTer;
         TIm2D<tBase,tBase>   mTImSomITer;
         TIm2D<REAL8,REAL8>   mTImSomI2Ter;
};

template <class TypeEl>
void cTplLoadedImage<TypeEl>::PrecalcRect(const Box2di & aBox)
{
   cSomRapideMII2<TypeEl> aSR(*this);

   cTplOpbBufImage< cSomRapideMII2<TypeEl> > aOpBuf
                  (
                      aSR,
                      Elise2Std(aBox._p0),
                      Elise2Std(aBox._p1),
                      Elise2Std(-mSzPtWFixe),
                      Elise2Std(mSzPtWFixe)
                  );
   aOpBuf.DoIt();
}



template <class TypeEl>
void cTplLoadedImage<TypeEl>::AddToStat
     (
        cStatGlob & aStatGlob,
        Pt2di aPTer
     )
{

    Pt2di aPTerInit = aPTer;
    aPTer = QuickTer2Cor(aPTer);


    if (! mTMasqImTer.get(aPTer))
       return;


    if (!IsVisible(aPTerInit))
       return;

    cStat1Distrib *aDist = aStatGlob.NextDistrib(mPDV);
    double * aVD = aDist->Vals();
    double aS1 = mTImSomITer.get(aPTer);
    double aS2 = mTImSomI2Ter.get(aPTer)-ElSquare(aS1)/double(mNbPtsW);
    aS2 = sqrt(ElMax(aS2,aStatGlob.EpsCorr()));
    aS1 /= mNbPtsW;


    //if (true)
    if (aStatGlob.IsFull())
    {
       Pt2di aP0 = aPTer-mSzPtWFixe;
       Pt2di aP1 = aPTer+mSzPtWFixe;
       Pt2di aP;
       for (aP.y = aP0.y ; aP.y<=aP1.y ; aP.y++)
       {
           for (aP.x = aP0.x ; aP.x<=aP1.x ; aP.x++)
           {
               *(aVD++) = (mTImTer.get(aP)-aS1)/aS2;
           }
       }
    }
    else
    {

          for (int aK=0 ; aK<aStatGlob.NbOK() ; aK++)
          {
                aVD[aStatGlob.VIndOK()[aK]] =  (mTImTer.get(aPTer+aStatGlob.VPtsOK()[aK])-aS1)/aS2;
          }
    }
}


       /*******************************************/
       /* Algo "rapide" pour Correl a N images    */
       /* Je ne suis pas completement sur que ce  */
       /* soit beaucoup plus rapide que l'algo    */
       /* standard, c'est plutot pour tester.     */
       /*                                         */
       /* Par contre ca pourrait etre une base    */
       /* interessante pour faire de la multi     */
       /* correlation "selective" (par ex ne      */
       /* retenant  que les couples correspondant */
       /* a des points de vues proches)           */
       /*******************************************/

template <class Type,const int aNb> class cNMasqIm
{
    public :
       int    mMsq[aNb];
       Type   mIm[aNb];
};


template <class Type,const int aNb> class cNVCov
{
    public :
       static const int aNCple = (aNb*(aNb-1)) / 2;
       cNVCov () 
       {
           for (int aK=0 ; aK<aNb ; aK++)
           {
               mSM[aK]=0;
               mSI[aK]=0;
               mSII[aK]=0;
           }
           for (int aK=0 ; aK<aNCple ; aK++)
              mSIJ[aK]=0;
       }
       void AddElem(int aSigne,const cNMasqIm<Type,aNb> & aMI)
       {
           for (int aK=0 ; aK<aNb ; aK++)
           {
               mSM[aK]  += aSigne * aMI.mMsq[aK];
               mSI[aK]  += aSigne * aMI.mIm[aK];
               mSII[aK] += aSigne * ElSquare(double(aMI.mIm[aK]));
           }
           for (int aI=0,aK=0 ; aI<aNb ; aI++)
           {
               for (int aJ=0 ; aJ<aI ; aJ++)
               {
                   mSIJ[aK++] += aSigne * aMI.mIm[aI] * double(aMI.mIm[aJ]);
               }
           }
       }

       void AddCumul (int aSigne,const cNVCov<Type,aNb> & aS)
       {
           for (int aK=0 ; aK<aNb ; aK++)
           {
               mSM[aK]  += aSigne * aS.mSM[aK];
               mSI[aK]  += aSigne * aS.mSI[aK];
               mSII[aK] += aSigne * aS.mSII[aK];
           }
           for (int aK=0 ; aK<aNCple ; aK++)
           {
              mSIJ[aK] += aSigne * aS.mSIJ[aK];
           }
       }
       
       int     mSM[aNb];
       Type    mSI[aNb];
       double  mSII[aNb];
       double  mSIJ[aNCple];
};


template <class TypeEl,const int aNb> class  cNVCovRapide
{
    public :
       typedef typename El_CTypeTraits<TypeEl>::tBase tBase;
       typedef cNMasqIm<tBase,aNb>  tElem;
       typedef cNVCov<tBase,aNb> tCumul;

       cNVCovRapide
       (
            std::vector<cTplLoadedImage<TypeEl> *> & aVLIm
       ) :
         mLIa         (*aVLIm[0]),
         mLT          (*(mLIa.Appli().LoadTer())),
         mNbPtsW      (mLIa.Appli().NbPtsWFixe()),
         mRNbPtsW     (mNbPtsW),
         mEpsC        (mLIa.Appli().EpsilonCorrelation().Val()),
         mDefC        (mLIa.Appli().DefCorrelation().Val()),
         mTCorrel     (mLT.ImCorrel())
       {
           for (int aK=0; aK<aNb ; aK++)
           {
                mVTIm.push_back(TIm2D<TypeEl,tBase>(aVLIm[aK]->ImTer()));
                mVTMasq.push_back(TIm2DBits<1>(aVLIm[aK]->MasqImTer()));
           }
       }

       void Init(const std::complex<int> & aPC,tElem & anEl)
       {
          Pt2di aP(aPC.real(),aPC.imag());

          for (int aK=0; aK<aNb ; aK++)
          {
             anEl.mMsq[aK]= mVTMasq[aK].get(aP);
             anEl.mIm[aK]= mVTIm[aK].get(aP);
          }
       }

       void UseAggreg(const std::complex<int> & aPC,const tCumul & aCum)
       {
          Pt2di aP(aPC.real(),aPC.imag());
          if  (! mLT.IsInMasqOfPx (mLIa.QuickCor2Ter(aP)))
          {
             mTCorrel.oset(mLIa.QuickCor2Ter(aP),mDefC);
             return;
          }
          for (int aK=0; aK<aNb ; aK++)
          {
               mOK[aK] = (aCum.mSM[aK] == mNbPtsW);
               if (mOK[aK])
               {
                  mSN[aK] = aCum.mSI[aK]/mRNbPtsW;
                  mS2N[aK] = aCum.mSII[aK]/mRNbPtsW-ElSquare(mSN[aK]);
                  mS2N[aK] = sqrt(ElMax(sqrt(mEpsC),mS2N[aK]));
               }
          }
          double aSCor=0;
          double aNbOk=0;
          for (int aI=0,aK=0 ; aI<aNb ; aI++)
          {
               for (int aJ=0 ; aJ<aI ; aJ++)
               {
                   if (mOK[aI] && mOK[aJ])
                   {
                      double aCIJ =  aCum.mSIJ[aK]/mRNbPtsW
                                      -mSN[aI]*mSN[aJ];
                      aSCor += aCIJ / (mS2N[aI]*mS2N[aJ]);
                      aNbOk++;
                   }
                   aK++;
               }
          }
          if (aNbOk)
             aSCor /= aNbOk;
          else
             aSCor = mDefC;
          mTCorrel.oset(mLIa.QuickCor2Ter(aP),aSCor);
       }
       void  OnNewLine(int anY) {}

       
    private :
         
         cTplLoadedImage<TypeEl> & mLIa;
         const cLoadTer &     mLT;
         int                  mNbPtsW;
         double               mRNbPtsW;
         double               mEpsC;
         double               mDefC;
         double               mSN[aNb];
         double               mS2N[aNb];
         bool                 mOK[aNb];

         std::vector<TIm2D<TypeEl,tBase> >  mVTIm;
         std::vector<TIm2DBits<1> >         mVTMasq;

         TIm2D<REAL8,REAL8>   mTCorrel;

};


template <class TypeEl,const int aNb>
void GlobCalcNCorrelRapide
           (
                const Box2di & aBox,
                std::vector<cTplLoadedImage<TypeEl> *> & aVLIm
           )
{
   cNVCovRapide<TypeEl,aNb> aNVR(aVLIm);
   cTplLoadedImage<TypeEl> & aI0 = *(aVLIm[0]);

   cTplOpbBufImage< cNVCovRapide<TypeEl,aNb> > aOpBuf
                  (
                      aNVR,
                      Elise2Std(aI0.QuickTer2Cor(aBox._p0)),
                      Elise2Std(aI0.QuickTer2Cor(aBox._p1)),
                      Elise2Std(-aI0.SzPtWFixe()),
                      Elise2Std( aI0.SzPtWFixe())
                  );
   aOpBuf.DoIt();

}

template <class TypeEl> 
     std::vector<cTplLoadedImage<TypeEl> *> 
     cTplLoadedImage<TypeEl>::ToCurType(tContPDV& aCont)
{
   std::vector<cTplLoadedImage<TypeEl> *>  aVTpl;

   for (int aK=0 ; aK<int(aCont.size()) ; aK++)
   {
      aVTpl.push_back
      (
           static_cast<cTplLoadedImage<TypeEl> *>
           (
                &(aCont[aK]->LoadedIm())
           )
      );
   }

   return aVTpl;
}



template <class TypeEl>
void  cTplLoadedImage<TypeEl>::NCalcCorrelRapide
           (
                const Box2di & aBox,
                tContPDV & aCont
           )
{
   tVecTPtr aVTpl = ToCurType(aCont);
 
   switch(aVTpl.size())
   {
      case 2 :
         GlobCalcNCorrelRapide<TypeEl,2>(aBox,aVTpl);
      break;
      case 3 :
         GlobCalcNCorrelRapide<TypeEl,3>(aBox,aVTpl);
      break;

      default :
         ELISE_ASSERT(false,"NCalcCorrelRapide");
      break;
   }
}


       /*******************************************/
       /* Algo rapide pour Correl a deux images   */
       /*******************************************/

template <class Type> class cMasqImAB
{
    public :
       int    mMa;
       int    mMb;
       Type   mIa;
       Type   mIb;
};

// Contient le cumul du masque, 
// et des variances et covariances 

template <class Type> class cVCovIaIb
{
    public :
       cVCovIaIb () :
            mSMa  (0),
            mSMb  (0),
            mSIa  (0),
            mSIb  (0),
            mSIaa (0),
            mSIab (0),
            mSIbb (0)
       {
       }
       void AddElem(int aSigne,const cMasqImAB<Type> & aMI)
       {
           mSMa  += aSigne * aMI.mMa;
           mSMb  += aSigne * aMI.mMb;
           mSIa  += aSigne * aMI.mIa;
           mSIb  += aSigne * aMI.mIb;
           mSIaa += aSigne * ElSquare(double(aMI.mIa));
           mSIab += aSigne * aMI.mIa * double(aMI.mIb);
           mSIbb += aSigne * ElSquare(double(aMI.mIb));
       }

       void AddCumul (int aSigne,const cVCovIaIb<Type> & aS)
       {
           mSMa  += aSigne * aS.mSMa;
           mSMb  += aSigne * aS.mSMb;
           mSIa  += aSigne * aS.mSIa;
           mSIb  += aSigne * aS.mSIb;
           mSIaa += aSigne * aS.mSIaa;
           mSIab += aSigne * aS.mSIab;
           mSIbb += aSigne * aS.mSIbb;
       }
       
       int     mSMa;
       int     mSMb;
       Type    mSIa;
       Type    mSIb;
       double  mSIaa;
       double  mSIab;
       double  mSIbb;
};



template <class TypeEl> class  cVCovABRapide
{
    public :
       typedef typename El_CTypeTraits<TypeEl>::tBase tBase;
       typedef cMasqImAB<tBase>  tElem;
       typedef cVCovIaIb<tBase> tCumul;

       cVCovABRapide
       (
            cTplLoadedImage<TypeEl> & aLIa,
            cTplLoadedImage<TypeEl> & aLIb,
            bool                      aModeAdd
       ) :
         mLIa         (aLIa),
         mLT          (*(aLIa.Appli().LoadTer())),
         mNbPtsW      (aLIa.Appli().NbPtsWFixe()),
         mRNbPtsW     (mNbPtsW),
         mEpsC        (aLIa.Appli().EpsilonCorrelation().Val()),
         mDefC        (aLIa.Appli().DefCorrelation().Val()),
         mTIa         (aLIa.ImTer()),
         mTMa         (aLIa.MasqImTer()),
         mTIb         (aLIb.ImTer()),
         mTMb         (aLIb.MasqImTer()),
         mTCorrel     (mLT.ImCorrel()),
         mModeAdd     (aModeAdd)
       {
       }

       void Init(const std::complex<int> & aPC,tElem & anEl)
       {
          Pt2di aP(aPC.real(),aPC.imag());
          anEl.mMa = mTMa.get(aP);
          anEl.mMb = mTMb.get(aP);
          anEl.mIa = mTIa.get(aP);
          anEl.mIb = mTIb.get(aP);
	}

       void UseAggreg(const std::complex<int> & aPC,const tCumul & aCum)
       {
          Pt2di aP(aPC.real(),aPC.imag());
          double aCor = mDefC;
          if 
          (
                  (aCum.mSMa==mNbPtsW) 
               && (aCum.mSMb==mNbPtsW)
               && ( mLT.IsInMasqOfPx (mLIa.QuickCor2Ter(aP)))
           )
          {
             double aSa  = aCum.mSIa  / mRNbPtsW;
             double aSb  = aCum.mSIb  / mRNbPtsW;
             double aSaa = aCum.mSIaa / mRNbPtsW -ElSquare(aSa);
             double aSbb = aCum.mSIbb / mRNbPtsW -ElSquare(aSb);
             double aSab = aCum.mSIab / mRNbPtsW -aSa*aSb;

             aCor = aSab / sqrt(ElMax(mEpsC,aSaa*aSbb));
          }
          if (mModeAdd)
	  {
		  mTCorrel.add(mLIa.QuickCor2Ter(aP),aCor);
	  }
          else
	  {
             mTCorrel.oset(mLIa.QuickCor2Ter(aP),aCor);
	  }
       }
       void  OnNewLine(int anY) {}

       
    private :
         
         const cTplLoadedImage<TypeEl> & mLIa;
         const cLoadTer &     mLT;
         int                  mNbPtsW;
         double               mRNbPtsW;
	 double               mEpsC;
	 double               mDefC;
         TIm2D<TypeEl,tBase>  mTIa;
	 TIm2DBits<1>         mTMa;
	 TIm2D<TypeEl,tBase>  mTIb;
	 TIm2DBits<1>         mTMb;
         TIm2D<REAL8,REAL8>   mTCorrel;
         bool                 mModeAdd;

};


template <class TypeEl>
void  cTplLoadedImage<TypeEl>::Instance_CalcCorrelRapide
(
 const Box2di & aBox,
 cTplLoadedImage<TypeEl> & anI2,
 bool                      aModeAdd
 )
{
	// Correlation Rapide entre deux images (*this) et anI2
	// le resultat est stocke dans mTCorrel

   cVCovABRapide<TypeEl> aVR(*this,anI2,aModeAdd);

/*
   bool aBug = aBox.inside(Pt2di(100,100));
   if (aBug)
   {
// BBBBBBBBBBBBBBBB
Video_Win aW = Video_Win::WStd(Pt2di(500,500),1);
ELISE_COPY(anI2.ImTer().all_pts(),anI2.ImTer().in(),aW.ogray());
   }
*/


   cTplOpbBufImage< cVCovABRapide<TypeEl> > aOpBuf
                  (
                      aVR,
                      Elise2Std(QuickTer2Cor(aBox._p0)),
                      Elise2Std(QuickTer2Cor(aBox._p1)),
                      Elise2Std(-mSzPtWFixe),
                      Elise2Std(mSzPtWFixe)
                  );
   aOpBuf.DoIt();

/*
   if (aBug)
   {
      std::cout << "eeeeeeeee " << mLT->ImCorrel().data()[100][100] << "\n";
      getchar();
   }
*/
}

template <class TypeEl>
void cTplLoadedImage<TypeEl>::CalcCorrelRapide 
           (
                const Box2di & aBox,
                cLoadedImage & anI
           )
{
   ELISE_ASSERT
   (
      TypeOfElem() ==anI.TypeOfElem(),
      "Cast Up in cTplLoadedImage<TypeEl>::CalcCorrelRapide"
   );
   Instance_CalcCorrelRapide
   (
       aBox,
       static_cast<cTplLoadedImage<TypeEl> &>(anI),
       false
   );

   
}

/*
template <class TypeEl>
void cTplLoadedImage<TypeEl>::CalcRapCorrelIm1Maitre 
           (
                const Box2di & aBox,
                tContPDV & aCont
           )
{
   Pt2di aP;

   TIm2D<REAL8,REAL8> mTCorrel (mAppli.LoadTer()->ImCorrel());

   for (aP.x=aBox._p0.x; aP.x<aBox._p1.x; aP.x++)
       for (aP.y=aBox._p0.y; aP.y<aBox._p1.y; aP.y++)
       {
             mTCorrel.oset(QuickCor2Ter(aP),0.0);
       }
   
   for (int aK=1 ; aK<int(aCont.size()) ; aK++)
   {
       Instance_CalcCorrelRapide
       (
           aBox,
           static_cast<cTplLoadedImage<TypeEl> &>(aCont[aK]->LoadedIm()),
           true
       );
   }

   double aNbIm = aCont.size()-1.0;
   for (aP.x=aBox._p0.x; aP.x<aBox._p1.x; aP.x++)
       for (aP.y=aBox._p0.y; aP.y<aBox._p1.y; aP.y++)
       {
             mTCorrel.div(QuickCor2Ter(aP),aNbIm);
       }
}
*/

/*****************************************/
/*                                       */
/*             cMSLoadedIm               */
/*                                       */
/*****************************************/

cMSLoadedIm::cMSLoadedIm(const cOneParamCMS& aParam ,Im2D_REAL4 * anI,bool First) :
   mImCMS (aParam),
   mIm    (First ? *anI : Im2D_REAL4(anI->sz().x,anI->sz().y)),
   mTIm   (mIm)
{
   if (! First)
   {
       mIm.dup(*anI);
   }

   if (aParam.Sigma() > 0)
   {
       if (aParam.SquareW().Val())
       {
           int aNb = round_ni(aParam.Sigma());
           if (aNb>0)
              SelfQMoyenne(mIm,Pt2di(aNb,aNb));
       }
       else
       {
          FilterGauss(mIm,aParam.Sigma(),2);
       }
   }
}

Im2D_REAL4 * cMSLoadedIm::Im() { return &mIm; }
const Im2D_REAL4 * cMSLoadedIm::Im() const { return &mIm; }

/*
*/


/*****************************************/
/*                                       */
/*            cLoadedImage               */
/*                                       */
/*****************************************/

cGeomDiscFPx AdaptSzW(const cGeomDiscFPx & aGeomInit,Pt2di aPtSzW)
{
   cGeomDiscFPx aRes = aGeomInit;
   aRes.SetClip
   (
       aRes.P0Clip() - aPtSzW,
       aRes.P1Clip() + aPtSzW
   );
   return aRes;
}


cLoadedImage::cLoadedImage
(
    const cAppliMICMAC & anAppli,
    const cPriseDeVue &  aPDV,
    const cGeomImage &   aGeomI,
    Box2di               aBoxIm,
    const Pt2di &        aSzMaxGT,
    cInterfModuleImageLoader * aIMIL,
    int                        aDZ,
    Fonc_Num                   aFMasq,
    bool                       IsFirstLoaded,
    eIFImL_TypeNumerique             aTypeNum
) :
   mPDV             (aPDV),
   mGeomI           (aGeomI),
   mAppli           (anAppli),
   mLT              (const_cast<cLoadTer *>(mAppli.LoadTer())),
   mCurEt           (mAppli.CurEtape()),
   mSzIm            (aBoxIm.sz()),
   mMasqIm          (mSzIm.x,mSzIm.y,1),
   mTMasqIm         (mMasqIm),

   mDoneMasqErod    (false),
   mMasqImErod      (1,1,0),
   mTMasqImErod     (mMasqImErod),

   mSzPtWFixe       ( mAppli.PtSzWFixe()),
   mSzPtWMarge      ( mAppli.PtSzWMarge()),
   mGeomTerAp       (anAppli.GeomDFPx()),
   mGeomCorrelImTer (AdaptSzW(mGeomTerAp,mSzPtWMarge)),

   //mSzCImT          (mGeomCorrelImTer.SzClip()),
   // mSzCImT          (aSzMaxGT + (mSzPtWMarge*2) + Pt2di(3,3)),
   mSzCImT         ( anAppli.CAH()                             ?
                     Pt2di(1,1)                                :
                     (aSzMaxGT + (mSzPtWMarge*2) + Pt2di(3,3)) 
                   ) ,

   mMasqImTer       (mSzCImT.x,mSzCImT.y,1),
   mTMasqImTer      (mMasqImTer),
   mTypeEl          (aTypeNum),
   mUsePC           (false),
   mImPC            (1,1),
   mTImPC           (mImPC)
{


   ELISE_COPY
   (
       mMasqIm.all_pts(),
       trans(aFMasq,aBoxIm._p0),
       mMasqIm.out()
   );


   const cEtapeMEC & anEt = anAppli.CurEtape()->EtapeMEC();
   for 
   (
       std::list<cMasqOfEtape>::const_iterator itM=anEt.MasqOfEtape().begin();
       itM!=anEt.MasqOfEtape().end();
       itM++
   )
   {

      if (itM->PatternApply()->Match(mPDV.Name()))
      {
          if (itM->RectInclus().IsInit())
          {
               Box2dr aRect = itM->RectInclus().Val();
               double aZoom = anAppli.CurEtape()->DeZoomIm();
               Pt2di aP0 = round_ni(aRect._p0/aZoom - Pt2dr(aBoxIm._p0));
               Pt2di aP1 = round_ni(aRect._p1/aZoom - Pt2dr(aBoxIm._p0));
               ELISE_COPY
               (
                   select
                   (
                       mMasqIm.all_pts(),
                       (FX<aP0.x) || (FX>aP1.x) || (FY<aP0.y) || (FY>aP1.y) 
                   ),
                   0,
                   mMasqIm.out()
               );
          }
      }
   }

   if (IsFirstLoaded)
   {
      mLT->SomMasqI1I2() =  Im2D_REAL8(mSzCImT.x,mSzCImT.y);
      mLT->Som1()        =  Im2D_REAL8(mSzCImT.x,mSzCImT.y);
      mLT->SomI1I2()     =  Im2D_REAL8(mSzCImT.x,mSzCImT.y);
      mLT->SomCorrel()   =  Im2D_REAL8(mSzCImT.x,mSzCImT.y);
      mLT->SomPdsCorr()  =  Im2D_REAL8(mSzCImT.x,mSzCImT.y);
      mLT->MasqGlob ()  =  Im2D_Bits<1>(mSzCImT.x,mSzCImT.y);
   }
   else
   {
       ELISE_ASSERT
       (
            mLT->SomMasqI1I2().sz() == mSzCImT,
	    "Erreur Interne MicMac: Pb coherence in cLoadedImage"
       );
   }
}



void cLoadedImage::PostInit()
{
    const cCorrelMultiScale*  aCMS = mAppli.CMS();

    if ( aCMS) 
    {
       for (int aK=0 ; aK<int(aCMS->OneParamCMS().size()) ; aK++)
       {
           cMSLoadedIm  aMLI(aCMS->OneParamCMS()[aK],FirstFloatIm() , aK==0);
           mMSLI.push_back(aMLI);
       }
    }
    else
    {
         cOneParamCMS aParam;
         aParam.SzW() = mAppli.PtSzWFixe();
         aParam.Sigma() = 0;
         aParam.Pds() = 1;

         cMSLoadedIm  aMLI(aParam,FirstFloatIm() , true);
         mMSLI.push_back(aMLI);
    }

    for (int aK=0 ; aK<int(mMSLI.size()) ; aK++)
    {
         Im2D_REAL4 * anIm = mMSLI[aK].Im();

         mVIm.push_back(anIm);
         mVNpIm.push_back(*anIm);
         mVDataIm.push_back(anIm->data());
         mVDataLin.push_back(anIm->data_lin());
    }

}

const std::vector<cMSLoadedIm>&  cLoadedImage::MSLI()
{
    return mMSLI;
}

std::vector<Im2D_REAL4>  cLoadedImage::VNpIm()
{
   return mVNpIm;
}


cLoadedImage * cLoadedImage::Alloc
               (
                   const cAppliMICMAC & anAppli,
                    const cPriseDeVue & aPDV,
                   const cGeomImage & aGeomI,
                   Box2di   aBox,
                   const Pt2di &  aSzMaxGT,
                   cInterfModuleImageLoader * aIMIL,
                   int                        aDZ,
                   Fonc_Num                   aFMasq,
		   bool IsFirstLoaded
                )
{
  /*
      Sur des images a tres peu de niveaux, a interet a avoir des images
      reechantillonnees en flottant
  */


   cLoadedImage * aRes = new cTplLoadedImage<REAL4>(anAppli,aPDV,aGeomI,aBox,aSzMaxGT,aIMIL,aDZ,aFMasq,IsFirstLoaded);

   aRes->PostInit();
   return aRes;
/*
   switch (aIMIL->PreferedTypeOfResol(aDZ))
   {
      case  eUnsignedChar :
            return new cTplLoadedImage<U_INT1>(anAppli,aGeomI,aBox,aSzMaxGT,aIMIL,aDZ,aFMasq);
      break;

      case  eUnsignedShort :
            return new cTplLoadedImage<U_INT2>(anAppli,aGeomI,aBox,aSzMaxGT,aIMIL,aDZ,aFMasq);
      break;

      case  eFloat :
            return new cTplLoadedImage<REAL4>(anAppli,aGeomI,aBox,aSzMaxGT,aIMIL,aDZ,aFMasq);
      break;

      default :
            ELISE_ASSERT(false,"cLoadedImage::Alloc");
      break;
   }
   return 0;
*/
}

cLoadedImage::~cLoadedImage()
{
}

void cLoadedImage::MakePC
     (
               cPriseDeVue & aPdv,
               cEtapeMecComp * aCur,
               cEtapeMecComp * aPred,
               const Box2di aBoxTer,
               bool         doPC,
               bool         doMasqAnam
     )
{
   mUsePC = true;
   Pt2di aSzCur = aBoxTer.sz();
   mImPC = Im2D_U_INT1(aSzCur.x,aSzCur.y,0);
   mTImPC = TIm2D<U_INT1,INT>(mImPC);

   mSeuilPC = 1;



  if (doPC)
  {

      std::string aName = mAppli.NamePC(true,aPred->EtapeMEC().GenerePartiesCachees().Val(),aPred,aPdv);
      cMetaDataPartiesCachees aMetaData = 
                   StdGetObjFromFile<cMetaDataPartiesCachees>
                   (
                       aName + ".xml",
                       StdGetFileXMLSpec("SuperposImage.xml"),
                       // mAppli.NameSpecXML(),
                       "MetaDataPartiesCachees",
                       "MetaDataPartiesCachees"
                   );
      if (! aMetaData.Done())
      {
          return;
      }

      Tiff_Im  aTif = Tiff_Im::BasicConvStd(aName+".tif");

      double aRealRDZ = aPred->DeZoomTer() / aCur->DeZoomTer();
      int aRDZ = round_ni(aRealRDZ);
      ELISE_ASSERT(aRDZ==aRealRDZ,"cLoadedImage::MakePC");


      // Pt2di aOfs = aMetaData.Offset() * aRDZ;
      mSeuilPC = aMetaData.SeuilUse();


      Box2di aBoxPrec 
             (
                round_down(Pt2dr(aBoxTer._p0)/aRealRDZ),
                round_up(Pt2dr(aBoxTer._p1)/aRealRDZ)
             );
      Pt2di aSzPrec = aBoxPrec.sz();

      Im2D_U_INT1 aImPC_Prec(aSzPrec.x,aSzPrec.y);
       TIm2D<U_INT1,INT> aTIPP (aImPC_Prec);

      ELISE_COPY
      (
         aImPC_Prec.all_pts(),
         trans(aTif.in(0),aBoxPrec._p0-aMetaData.Offset()),
         aImPC_Prec.out()
      );



      Pt2di aP ;
      for (aP.x=0 ; aP.x <aSzCur.x ; aP.x++)
      {
         for (aP.y=0 ; aP.y <aSzCur.y ; aP.y++)
         {
              Pt2dr aPP = Pt2dr(aP+aBoxTer._p0)/aRealRDZ - Pt2dr(aBoxPrec._p0);
              mTImPC.oset(aP,aTIPP.get(round_ni(aPP),255));
         }
      }

      cTplValGesInit<std::string> aNameVisu = aCur->EtapeMEC().NameVisuTestPC();
      if (aNameVisu.IsInit())
      {
         Tiff_Im aTF = Tiff_Im::UnivConvStd(mAppli.WorkDir()+aNameVisu.Val());
         static Video_Win aW = Video_Win::WStd(Pt2di(1000,800),1.0);

         ELISE_COPY(aW.all_pts(),P8COL::yellow,aW.odisc());
         ELISE_COPY
         (
             mImPC.all_pts(),
             trans(aTF.in(0),aBoxTer._p0),
             aW.ogray()
         );

         ELISE_COPY
         (
             select(mImPC.all_pts(),mImPC.in()>3),
             P8COL::red,
             aW.odisc()
         );
         getchar();

      }
   }

  if (doMasqAnam)
  {
      cGeomImage & aGeomI = aPdv.Geom();
      const cGeomDiscFPx & aGeoT = aCur->GeomTer();
      Pt2di aP ;
      for (aP.y=0 ; aP.y< aSzCur.y ; aP.y++)
      {
         for (aP.x=0 ; aP.x< aSzCur.x ; aP.x++)
         {
// std::cout << aP << aGeoT.DiscToR2(aP) << "\n";
             if (! aGeomI.IsInMasqAnamSA(aGeoT.DiscToR2(aP)))
                mTImPC.oset(aP,mSeuilPC);
         }
      }
/*
std::cout << aBoxTer._p0 << aBoxTer._p1 << "\n";
Pt2dr aM = aBoxTer.milieu() ;
std::cout << aM <<  aGeoT.DiscToR2(aM) << "\n";
std::cout <<  aGeoT.DiscToR2(Pt2dr(0,0)) <<  << "\n";


      static Video_Win aW = Video_Win::WStd(Pt2di(1000,800),1.0);
      ELISE_COPY(aW.all_pts(),P8COL::red,aW.odisc());
      ELISE_COPY(mImPC.all_pts(),mImPC.in(),aW.odisc());
      getchar();
*/
/*
*/
  }
}


Pt2di cLoadedImage::DiscTerAppli2DiscTerCorr(const Pt2di  &aPt)
{
   Pt2dr aResR =   mGeomCorrelImTer.R2ToRDisc(mGeomTerAp.DiscToR2(aPt));
   Pt2di aResI = round_ni(aResR);
   double anEr = ElAbs(aResR.x-aResI.x) +  ElAbs(aResR.y-aResI.y);
   // Apparement, ca marche parce que le resultat est implicitement
   // entier.
   ELISE_ASSERT(anEr<1e-5,"CcLoadedImage::DiscTerAppli2DiscTerCorr");
   return  aResI;
}


Im2D_Bits<1> cLoadedImage::MasqImTer()     {return mMasqImTer;}
const cAppliMICMAC & cLoadedImage::Appli() {return mAppli;}

eIFImL_TypeNumerique cLoadedImage::TypeOfElem() const {return mTypeEl;}

const Pt2di & cLoadedImage::SzPtWMarge() const {return mSzPtWMarge;}
const Pt2di & cLoadedImage::SzPtWFixe() const {return mSzPtWFixe;}

const Pt2di cLoadedImage::SzIm() const {return mSzIm;}

bool     cLoadedImage::UsePC() const {return mUsePC;}
U_INT1** cLoadedImage::DataImPC() const  {return mImPC.data();}
int      cLoadedImage::SeuilPC() const {return mSeuilPC;}

U_INT1** cLoadedImage::DataMasqIm() const  {return mMasqIm.data();}
U_INT1** cLoadedImage::DataMasqImErod() const  
{
   return  mDoneMasqErod ? mMasqImErod.data() : 0 ;
}

void cLoadedImage::DoMasqErod(const Box2di & aBox)
{
    ElTimer aChrono;
    ELISE_ASSERT(!mDoneMasqErod,"Mulriple void cLoadedImage::DoMasqErod");
    mDoneMasqErod = true;
    Pt2di aSz = mMasqIm.sz();
    mMasqImErod = Im2D_Bits<1>(aSz.x,aSz.y,0);
    mTMasqImErod = TIm2DBits<1>(mMasqImErod);

    Pt2di aP0 = aBox._p0;
    Pt2di aP1 = aBox._p1;
    int aNb = (aP1.x-aP0.x + 1) *  (aP1.y-aP0.y + 1);
    ELISE_COPY
    (
         mMasqIm.all_pts(),
         rect_som(mMasqIm.in(0),aBox) == aNb,
         mMasqImErod.out()
    );

    if (0)
    {
        std::cout << "cLoadedImage::DoMasqErod " << aSz << " " << aChrono.uval() << "\n";
        std::string aName = "MASQ-ERRRR.tif";
        Tiff_Im::Create8BFromFonc(aName,aSz,mMasqIm.in()+2*mMasqImErod.in());
        getchar();
    }
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
