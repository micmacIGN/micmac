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

#include "general/sys_dep.h"

#if (ELISE_X11)

#include "Vino.h"


/****************************************/
/*                                      */
/*          Grab Geom                   */
/*                                      */
/****************************************/

double VerifInt(const int * anInput,int aNb)
{
   return 0;
}

double  VerifInt(const double * anInput,int aNb)
{
   double aSom = 0.0;

   for (int aK=0 ; aK<aNb ; aK++)
   {
        aSom += ElAbs(anInput[aK]-round_ni(anInput[aK]));
   }
   return (aNb==0) ? 0.0 : (aSom/aNb);
}

template <class Type,class TypeOut> void  cAppli_Vino_TplChgDyn<Type,TypeOut>::SetDyn(cAppli_Vino & anAppli,TypeOut * anOut,const Type * anInput,int aNb)
{
    // std::cout << " VerifInt== " << VerifInt(anInput,aNb) << "\n"; getchar();
    if (anAppli.mTabulDynIsInit)
    {
       int aMaxInd = anAppli.mTabulDyn.size() - 1;
       int * aTD = & (anAppli.mTabulDyn[0]);
       double aV0 = anAppli.mV0TabulDyn;
       double aStep = anAppli.mStepTabulDyn;
       for (int aK=0 ; aK<aNb ; aK++)
       {
           // int aInd = round_ni((anInput[aK]-mV0TabulDyn)/mStepTabulDyn);
           int aInd = round_ni((anInput[aK]-aV0)/aStep);
           aInd = ElMax(0,ElMin(aMaxInd,aInd));
           anOut[aK] = aTD[aInd];
       }
       return;
    }

   const cXml_StatVino & aStats = *(anAppli.mCurStats);

    switch (aStats.Type())
    {
          case eDynVinoModulo :
          {
              for (int aK=0 ; aK<aNb ; aK++)
                   anOut[aK] =  int(anInput[aK]) % 256;
              return;
          }

          case eDynVinoMaxMin :
          {
              
              Type aV0 = aStats.IntervDyn().x; 
              Type anEcart = aStats.IntervDyn().y -aV0; 
              for (int aK=0 ; aK<aNb ; aK++)
              {
                   anOut[aK] = ElMax(0,ElMin(255, round_ni(((anInput[aK] -aV0) * 255) / anEcart)));
              }
              return;
          }

          case eDynVinoStat2 :
          {
              
              double aMoy   = aStats.Soms()[0];
              double anECT  = aStats.ECT()[0] / aStats.MulDyn();
              for (int aK=0 ; aK<aNb ; aK++)
              {
                  float aVal = (anInput[aK]-aMoy)/ anECT;
                   // anOut[aK] = 128 * (1+ aVal / (ElAbs(aVal) +0.5));
                   anOut[aK] = ElMax(0,ElMin(255,round_ni(256 * erfcc (aVal))));
              }
              return;
          }

          default :
          {
              for (int aK=0 ; aK<aNb ; aK++)
                   anOut[aK] =  anInput[aK] ;
              return;
          }
    }
}


void cAppli_Vino::ChgDyn(int * anOut,const double * anInput,int aNb) 
{
    cAppli_Vino_TplChgDyn<double,int>::SetDyn(*this,anOut,anInput,aNb);
/*
    if (0) // Laisser : force instatiation 
    {
       cAppli_Vino_TplChgDyn<double,double>::SetDyn(*this,(double *)0,(const double *)0,0);
    }
*/
    // TplChgDyn(*mCurStats,anOut,anInput,aNb);
}

void cAppli_Vino::ChgDyn(int * anOut,const int * anInput,int aNb) 
{
    cAppli_Vino_TplChgDyn<int,int>::SetDyn(*this,anOut,anInput,aNb);
    // TplChgDyn(*mCurStats,anOut,anInput,aNb);
}


template <class Type> class cFoncNum_AppliVino_ChgDyn : public Simple_OP_UN<Type>
{
   public :
       void  calc_buf
             (
                           Type ** output,
                           Type ** input,
                           INT        nb,
                           const Arg_Comp_Simple_OP_UN  &
             ) 
             {
                for (int aD=0 ; aD< mDim ; aD++)
                {
                    cAppli_Vino_TplChgDyn<Type,Type>::SetDyn(*mAppli,output[aD],input[aD],nb);
                }
             }

             cFoncNum_AppliVino_ChgDyn(cAppli_Vino  &anAppli,int aDim) :
                mAppli (&anAppli),
                mDim (aDim)
             {
             }
        
   private :
             cAppli_Vino * mAppli;
             int           mDim;
};

Fonc_Num  ChgDynAppliVino(Fonc_Num aF,cAppli_Vino & anAppli)
{
    int aDim = aF.dimf_out();
    return create_users_oper
           (
              new cFoncNum_AppliVino_ChgDyn<INT>(anAppli,aDim),
              new cFoncNum_AppliVino_ChgDyn<double>(anAppli,aDim),
              aF,
              aDim
           );
}




void cAppli_Vino::InitTabulDyn()
{
   if (mCurStats==0) return;
   if (!mCurStats->IsInit()) return;


   mTabulDynIsInit = false;
   double aMoy   = mCurStats->Soms()[0];
   double anECT  = mCurStats->ECT()[0] ;

   double  anEcart = anECT * 10 ; // 10 Sigma

   mV0TabulDyn = aMoy - anEcart;
   mStepTabulDyn = anECT / (255.0 * 5);

   int aNbTabul = round_ni((anEcart/mStepTabulDyn) * 2);

   if (mCurStats->Type() == eDynVinoStat2)
   {
       mTabulDyn.clear();
       double aDiv  = anECT / mCurStats->MulDyn();
       mTabulDynIsInit = true;
       for (int aK=0 ; aK<= aNbTabul ; aK++)
       {
           double  aVal = mV0TabulDyn + aK * mStepTabulDyn;
           aVal = (aVal-aMoy)/aDiv;
           mTabulDyn.push_back(ElMax(0,ElMin(255,round_ni(256 * erfcc (aVal)))));
       }
   }

   if (mCurStats->Type()==eDynVinoEqual)
   {
       double * aDC = mHistoCum.data();
       int aNbH =  mHistoCum.tx();

       mTabulDyn.clear();
       mTabulDynIsInit = true;
       double aVEnd = aDC[aNbH-1];
       for (int aK=0 ; aK<= aNbTabul ; aK++)
       {
           double  aVal = mV0TabulDyn + aK * mStepTabulDyn;
           aVal = (aVal-mCurStats->VMinHisto())/mCurStats->StepHisto();

           aVal = ElMax(0.0,ElMin(aVal,aNbH-1.001));
           int iVal = round_down(aVal);
           double aP1 = aVal-iVal;
           double aP0 = 1 - aP1;
           aVal = aDC[iVal] * aP0 + aDC[iVal+1] * aP1;
           
           mTabulDyn.push_back(ElMax(0,ElMin(255,round_ni(256  * (aVal/aVEnd)))));
       }
   }
}

void cAppli_Vino::HistoSetDyn()
{
    std::string aMes = "Clik  for polygone ; Shift Clik  to finish ; Enter 2 point for rectangle";
    ElList<Pt2di> aL = GetPtsImage(false,false,false);
    if (aL.card() >= 2)
    {
        Flux_Pts aFlux = rectangle(aL.car(),aL.cdr().car());
        if (aL.card() >2)
           aFlux = polygone(aL);

         FillStat(*mCurStats,aFlux,mScr->CurScale()->in());
/*
        if (aL.card()== 2)
           FillStat(*mCurStats,rectangle(aL.car(),aL.cdr().car()),mScr->CurScale()->in());
        else
           FillStat(*mCurStats,polygone(aL),mScr->CurScale()->in());
*/

        if (mCaseCur==mCaseHStat)
           mCurStats->Type() =  eDynVinoStat2;

        if (mCaseCur==mCaseHMinMax)
           mCurStats->Type() =  eDynVinoMaxMin;

        if (mCaseCur==mCaseHEqual)
        {
             DoHistoEqual(aFlux);
             mCurStats->Type() =  eDynVinoEqual;
        }


        mCurStats->IsInit() = true;
        InitTabulDyn();
        SaveState();
    }
    Refresh();
}

template <class Type,class TyBase> Im2D<Type,TyBase> Im1D2Im2D(Im1D<Type,TyBase> aI1)
{
    Im2D<Type,TyBase>  aI2(aI1.tx(),1);
    ELISE_COPY(aI2.all_pts(),aI1.in()[FX],aI2.out());
    return aI2;
}

void PlotHisto(Video_Win aW,Im1D_REAL8 anIm,int aCoul,int aWidth)
{
  Im2D_REAL8 aI2DInit = Im1D2Im2D(anIm);
  int aSzWX = aW.sz().x-20;
  int aSzWY =  aW.sz().y;
  int aSHw = anIm.tx();
  double aCompr = double(aSzWX) / double(aSHw);

   Im2D_REAL8 aI2DRed(aSzWX,1);
   // ELISE_COPY(aI2D.all_pts(),,aI2D.out());

   if (1) // (aCompr < 1)
   {
      ELISE_COPY
      (
           aI2DRed.all_pts(),
           StdFoncChScale
           (
               aI2DInit.in_proj(),
               Pt2dr(0,0),
               Pt2dr(1.0/aCompr,1.0),
               Pt2dr(1,1)
           ),
           aI2DRed.out()
   
      );
   }
   else
   {
       ELISE_COPY(aI2DRed.all_pts(), anIm.in(0)[FX*aCompr], aI2DRed.out());
   }

   double aVMax;
   ELISE_COPY(aI2DRed.all_pts(),aI2DRed.in(),VMax(aVMax));
   double * aData = aI2DRed.data()[0];

   double aY0 = aSzWY *(1 - 1/20.0);
   double aMulY = (aSzWY * 0.8) / aVMax;
   
   std::vector<Pt2dr> aVPts;
   for (int aK= 0 ; aK<aSzWX ; aK++)
   {
       aVPts.push_back(Pt2dr(aK,aY0 - aMulY * aData[aK]));

   }
   for (int aK= 1 ; aK<aSzWX ; aK++)
   {
       aW.draw_seg(aVPts[aK-1],aVPts[aK],Line_St(aW.pdisc()(aCoul),2));
   }
}

void cAppli_Vino::DoHistoEqual(Flux_Pts aFlux)
{
   
    mCurStats->VMinHisto() = 1e30;
    mVMaxHisto = -1e30;
    double aNbSigma = 15;
    int aNbCh = mCurStats->VMax().size();
    double anEctGlob = 0;

    for (int aK=0 ; aK<aNbCh ; aK++)
    {
         double aMoy = mCurStats->Soms()[aK];
         double anEct = mCurStats->ECT()[aK];
         double aVMin = ElMax(mCurStats->VMin()[aK],aMoy-aNbSigma*anEct);
         double aVMax = ElMin(mCurStats->VMax()[aK],aMoy+aNbSigma*anEct);
         anEctGlob += anEct;
         mCurStats->VMinHisto() = ElMin(mCurStats->VMinHisto(),aVMin);
         mVMaxHisto = ElMax(mVMaxHisto,aVMax);
    }

    anEctGlob /= aNbCh;

    Fonc_Num aF = mScr->CurScale()->in().kth_proj(0);
    for (int aK =1 ; aK<aNbCh ; aK++)
    {
        aF = aF + mScr->CurScale()->in().kth_proj(aK);
    }
    aF = aF / double(aNbCh);

    mCurStats->StepHisto() = 1;
    if (  ((mVMaxHisto-mCurStats->VMinHisto()) < mNbHistoMax) && (aF.integral_fonc(true)))
    {
         mNbHisto  = mVMaxHisto-mCurStats->VMinHisto() + 1;
    }
    else
    {
         mNbHisto = mNbHistoMax;
         mCurStats->StepHisto() =  (mVMaxHisto - mCurStats->VMinHisto()) / (mNbHisto-1);
    }
    mHisto.Resize(mNbHisto);
    mHistoLisse.Resize(mNbHisto);
    mHistoCum.Resize(mNbHisto);

    mHisto.raz();
    Fonc_Num aFI = Max(0,Min(mNbHisto-1,round_ni((aF-mCurStats->VMinHisto()) / mCurStats->StepHisto())));


    ELISE_COPY
    (
         aFlux,
         1,
         mHisto.histo(true).chc(aFI)
    );

    Im2D_REAL8 aI2L = Im1D2Im2D(mHisto);
    int aNbIter = 4;
    double aSigmRest = ElSquare((0.05*anEctGlob) / mCurStats->StepHisto());
    double aFact  = FromSzW2FactExp(sqrt(aSigmRest),aNbIter);
    // std::cout << "Sigma000 " << sqrt(aSigmRest) << " Fact " << aFact  << " " << anEctGlob/mCurStats->StepHisto() << "\n";
    for (int aK=0 ; aK< aNbIter ; aK++)
    {
         ELISE_COPY
         (
              aI2L.all_pts(),
              canny_exp_filt(aI2L.in(0),aFact,aFact) / canny_exp_filt(aI2L.inside(),aFact,aFact),
              aI2L.out()
         );
    }
/*
    for (int aK=0 ; aK< aNbIter ; aK++)
    {
        if (aSigmRest>0)
        {
               double aSigma = aSigmRest / (aNbIter-aK);
               int aNb = round_ni(sqrt(aSigma));
               aSigmRest -= ElSquare(aNb);
               if (aNb >=0)
               {
                  ELISE_COPY
                  (
                       aI2L.all_pts(),
                       rect_som(aI2L.in(0),Pt2di(aNb,1)) / rect_som(aI2L.inside(),Pt2di(aNb,1)),
                       aI2L.out()
                  );
               }
               std::cout << "Nbbbb " << aNb << "\n";
        }
    }
*/
    ELISE_COPY(mHistoLisse.all_pts(),aI2L.in()[Virgule(FX,0)],mHistoLisse.out());

    double * aDH = mHistoLisse.data();
    double * aDC = mHistoCum.data();
    aDC[0] = aDH[0];

    for (int aK=1 ; aK<mNbHisto ; aK++)
         aDC[aK] =  aDC[aK-1] + aDH[aK];


       
    // std::cout  << "DoHistoEqual " << aVMinGlob << " " << aVMaxGlob << " "<< anEctGlob << "\n";


    PlotHisto(*mW,mHisto,P8COL::red,2);
    PlotHisto(*mW,mHistoLisse,P8COL::blue,2);
    PlotHisto(*mW,mHistoCum,P8COL::green,2);
    EffaceMessageRelief();
    PutMessageRelief(0,"Histo diplaid, Clik to continue");
    mW->clik_in();


    Im2D_REAL8 aI2C = Im1D2Im2D(mHistoCum);
    Tiff_Im::CreateFromIm(aI2C, mNameHisto);


/*
     plot.set(NewlArgPl1d(PlModePl(Plots::line)));
     plot.set(NewlArgPl1d(PlotLinSty(lst)));
*/


    //  FillStat(*mCurStats,aFlux,mScr->CurScale()->in());
    // int aVMin = ElMax(mCurStats->VMax
}


#endif



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
