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
#include "api/cox_roy.h"


#include "ext_stl/appli_tab.h"


// template class cMatrOfSMV<int>;
// template class cMatrOfSMV<Pt2di>;


namespace NS_ParamMICMAC
{


/*********************************************/
/*                                           */
/*       cSurfaceOptimiseur                  */
/*                                           */
/*********************************************/


cSurfaceOptimiseur::cSurfaceOptimiseur
(
    cAppliMICMAC &    anAppli,
    cLoadTer&               aLT,
    double                  aCsteCost,
    const cEquiv1D &        anEqX,
    const cEquiv1D &        anEqY,
    bool                    AcceptEBI,
    bool                    CanFillCorrel
)  :
   mCsteCost   (aCsteCost),
   mAppli      (anAppli),
   mDefCost     (anAppli.DefCost()),
   mOneDCAllDC (anAppli.CurDCAllDC()),
   mDimPx      (mAppli.DimPx()),
   mEtape      (*(anAppli.CurEtape())),
   mDeZoom     (mEtape.DeZoomTer()),
   mCarDZ      (mAppli.GetCaracOfDZ(mDeZoom)),
   mAlgoSR     (mEtape.SsResAlgo()),
   mEqX        (cEquiv1D::cCstrFusion(),anEqX,mAlgoSR),
   mEqY        (cEquiv1D::cCstrFusion(),anEqY,mAlgoSR),
   mEBI        (AcceptEBI ? anAppli.CurEtape()->EBI() : 0),
   mGlobSR     (mAlgoSR / mCarDZ->RatioPtsInt()),
   mWithEQ     (mCarDZ->HasMasqPtsInt() || (mAlgoSR !=1)),
   mLTInit     (aLT),
   mSzInit     (mLTInit.Sz()),
   //  mSRA        (mEtape.SsResAlgo()),
   mLTRed      (mWithEQ ? new cLoadTer(mLTInit,mEqX,mEqY) : 0),
   mLTCur      (mWithEQ ?  mLTRed : & mLTInit),
   mSzCur      (mLTCur->Sz()),
   mMemoCorrel (0),
   mCanFillCorrel  (CanFillCorrel),
   mDoFileCorrel   (mEtape.GenImageCorrel()),
   mReducCpt   (0),
   mReducCost  (0),
   mMaskCalcDone (false),
   mMaskCalc     (1,1)
{


   for (int aK=0 ; aK<theDimPxMax ; aK++)
   {
      mCostRegul[aK] = mCostRegul_Quad[aK] = 0.0;
   }

   for (int aK=0 ; aK<mAppli.DimPx() ; aK++)
   {
      const cFilePx & aFP = mEtape.KPx(aK);
      bool isRugInit = anAppli.RugositeMNT().IsInit();
      double aEc = isRugInit ? anAppli.EnergieExpCorrel().Val() : -2.0;
      double aEp = isRugInit ? anAppli.EnergieExpRegulPlani().Val() : -1.0 ;
      double aEa = isRugInit ? anAppli.EnergieExpRegulAlti().Val() :  -1.0;

      double aFacRugos =  pow(aFP.Pas(),-aEa)
                        * pow((double)mEtape.DeZoomTer(),aEc-aEp-aEa)
                        * pow(mGlobSR,aEc-aEp);


      mCostRegul[aK] =    aFP.Regul() * aFacRugos;
      mCostRegul_Quad[aK] = aFP.Pas() *   aFP.Regul_Quad() * aFacRugos;


      mSeuilAttenZReg[aK] = 1e5;
      mCostRegulAttenue[aK] = mCostRegul[aK];

      if (aK==0)
      {
          const cEtapeMEC &  anEM= mEtape.EtapeMEC();
          double aPasAbs = mAppli.GeomDFPx().PasPx0() ;
          double aPasRel = mAppli.GeomDFPx().PasPxRel0() ;

          if ( anEM.SeuilAttenZRegul().IsInit())
          {
             mSeuilAttenZReg[aK] = anEM.SeuilAttenZRegul().Val() ;
             if (mAppli.InversePx())
                mSeuilAttenZReg[aK] /= aPasRel;
             else
                mSeuilAttenZReg[aK] /= aPasAbs;

             mCostRegulAttenue[aK]  =  mCostRegul[aK] * anEM.AttenRelatifSeuilZ().ValWithDef(0.2);
          }
      }
      mCsteCostSeuilAtten[aK]  =  mSeuilAttenZReg[aK] * ( mCostRegul[aK]-mCostRegulAttenue[aK]);


      mImResInit.push_back(mLTInit.KthNap(aK).mPxRes);
      mImRes.push_back(mLTCur->KthNap(aK).mPxRes);
      mDataImRes.push_back(mImRes.back().data());
   }
   // Rajouter la mise a niveau de min et max
   INT2 ** mDXMin = mLTCur->KthNap(0).mImPxMin.data();
   INT2 ** mDXMax = mLTCur->KthNap(0).mImPxMax.data();
   INT2 ** mDYMin = (mDimPx>1) ? mLTCur->KthNap(1).mImPxMin.data() : 0;
   INT2 ** mDYMax = (mDimPx>1) ? mLTCur->KthNap(1).mImPxMax.data() : 0;
   Box2di aBox(Pt2di(0,0),mSzCur);


   if (mWithEQ)
   {
      mReducCpt    = new cMatrOfSMV<U_INT2>(aBox,mDXMin,mDYMin,mDXMax,mDYMax,0);
      mReducCost   = new cMatrOfSMV<REAL4> (aBox,mDXMin,mDYMin,mDXMax,mDYMax,0.0);
   }


   if (mEtape.GenImageCorrel())
   {
      if (! mCanFillCorrel)
      {
          mMemoCorrel = new cMatrOfSMV<U_INT1>(aBox,mDXMin,mDYMin,mDXMax,mDYMax,0);
      }
   }
}

double cSurfaceOptimiseur::CostTransEnt(int aTrans,int aKPx)
{
   double  aCL1 =                               (aTrans < mSeuilAttenZReg[aKPx])             ?
                  mCostRegul[aKPx] * aTrans                                    :
                  mCsteCostSeuilAtten[aKPx] + mCostRegulAttenue[aKPx] * aTrans ;


   return aCL1 + mCostRegul_Quad[aKPx] *  ElSquare(aTrans);
}

      // mCsteCostSeuilAtten[aK]  =  mSeuilAttenZReg[aK] * ( mCostRegul[aK]-mCostRegulAttenue[aK]);
      //  aTrans*(mCostRegul-mCostRegulAttenue) + mCostRegulAttenue*aTrans


cSurfaceOptimiseur::~cSurfaceOptimiseur()
{
     delete mLTRed;
     delete mMemoCorrel;
     delete mReducCpt;
     delete mReducCost;
}

double   cSurfaceOptimiseur::CostAmpl(const double & aCost) const
{
   return aCost*mCsteCost;
}

int   cSurfaceOptimiseur::CostR2I(const double & aCost) const
{
   return round_ni(CostAmpl(aCost));
}

double cSurfaceOptimiseur::CostI2R(const int & aCost) const
{
    return aCost / mCsteCost;
}


int   cSurfaceOptimiseur::CostI2CorExport(const int & aCostI) const
{
    double aCostR = CostI2R(aCostI);
    double aCor = mAppli.StatGlob()->Cout2Correl(aCostR);
    return ElMax(0,ElMin(255,round_ni(128*(1+aCor))));
}


int cSurfaceOptimiseur::CostToMemoCorrel(double aCost)  const
{
   return ElMax(0,ElMin(255,round_ni(128.0*(2-aCost)-0.5)));
}


cSurfaceOptimiseur * cSurfaceOptimiseur::Alloc
(
    cAppliMICMAC &    mAppli,
    cLoadTer&         aLT,
    const cEquiv1D &        anEqX,
    const cEquiv1D &        anEqY
)  
{
   cSurfaceOptimiseur *aRes=0;

   switch( mAppli.CurEtape()->AlgoRegul())
   {
       case eAlgoCoxRoy :
            aRes = cSurfaceOptimiseur::AllocCoxRoy(mAppli,aLT,anEqX,anEqY);
       break;

       case eAlgoMaxOfScore :
            aRes = cSurfaceOptimiseur::AllocCoxMaxOfCorrel(mAppli,aLT,anEqX,anEqY);
       break;

       case eAlgo2PrgDyn :
            aRes = cSurfaceOptimiseur::AllocPrgDyn(mAppli,aLT,anEqX,anEqY);
       break;


       default :
              ELISE_ASSERT(false,"Optimization non supportee");
       break;
   }

   return aRes;
}


void cSurfaceOptimiseur::SetCout(Pt2di aPTer,int * aPX,REAL aCost,int aLabel)
{

   if ((aLabel !=0) && (! mEBI))
   {
        ELISE_ASSERT(false,"cSurfaceOptimiseur::SetCout ilicit labeling");
   }


   bool aDC = mOneDCAllDC && (aCost==mDefCost);
   if (aDC)
   {
         mLTInit.SetDefCorrActif(aPTer,1);
   }

   if (! mWithEQ)
   {

       Local_SetCout(aPTer,aPX,aCost,aLabel);
       if (mMemoCorrel)
       {
          Pt2di aPPx = mAppli.Px2Point(aPX);
          (*mMemoCorrel)[aPTer][aPPx]= CostToMemoCorrel(aCost);
       }
   }
   else
   {
      if (mLTInit.IsInMasq(aPTer))
      {
          Pt2di aPPx = mAppli.Px2Point(aPX);
          Pt2di aPRed = ToSRAlg(aPTer);
          (*mReducCpt) [aPRed][aPPx] ++;
          (*mReducCost)[aPRed][aPPx] += (float) aCost;
	  if (aDC)
	  {
             mLTRed->SetDefCorrActif(aPRed,1);
          }
      }
   }
}

void cSurfaceOptimiseur::Local_SetCpleRadiom(Pt2di aPTer,int * aPX,U_INT2 aR1,U_INT2 aR2)
{
    ELISE_ASSERT(false,"Cur Opt do not handle cple of radiom");
}

void  cSurfaceOptimiseur::Local_VecInt1(Pt2di aPTer,int * aPX,const  std::vector<INT1> &)
{
    ELISE_ASSERT(false,"Cur Opt do not handle Local_VecInt1");
}


typedef std::list<cSpecFitrageImage> tLPFP;

bool Apply(ePxApply aPA,int aK)
{
   switch (aPA)
   {
      case eApplyPx1 : 
           return (aK==0);
      break;

      case eApplyPx2 : 
           return (aK==1);
      break;

      case eApplyPx12 : 
           return ((aK==0)||(aK==1));
      break;

      default:
      break;
   }
   ELISE_ASSERT(false,"Bad ePxApply value");
   return false;
}

Fonc_Num  OneItereFiltrageImMicMac
                 (
                      const cSpecFitrageImage & aParam,
                      Fonc_Num aFonc,
                      Fonc_Num aFoncMasq,
                      double      aSRA
                 )
{
   double aSz =  aParam.SzFiltrNonAd().Val() + aParam.SzFiltrage() / aSRA;
   int aSzI = round_ni(aSz);
   switch (aParam.TypeFiltrage())
   {
       case eFiltrageMedian :
       {
            int VMil = 1<<15;
            Symb_FNum aF =
                        rect_median
                        (
                          Virgule(aFonc + VMil,1+50*aFoncMasq),
                          round_up(aSz-1e-5),
                          (2*VMil),
                          false,
                          true,
                          Histo_Kieme::last_rank
                        )  ;
            return (aF>0) * (aF-VMil) ;
       }
       break;

       case eFiltrageMoyenne :
       {
            ELISE_ASSERT(aSzI==aSz,"Taille non entiere dans eFiltrageMoyenne");

            return rect_som(aFonc *aFoncMasq,aSzI)
	           /  Max(rect_som(aFoncMasq,aSzI),Fonc_Num(1.0));
       }
       break;

       case eFiltrageDeriche :
       {
            aFonc = deriche(aFonc.v0(),1/aSz);
            aFonc = polar(aFonc,0).v0();
            return aFonc;
       }
       break;

       case eFiltrageGamma :
       {
           double anAmpl = aParam.AmplitudeSignal().Val();
	   // std::cout << "GAMA " << aSz << "\n"; getchar();
           return  anAmpl*pow(aFonc/anAmpl,1/aSz);
       }
       break ;

       case eFiltrageEqLoc :
       {
         Symb_FNum  sF (Rconv(aFonc));
	 Symb_FNum  sM (Rconv(aFoncMasq));

	 Fonc_Num fSom = Virgule(sM,sM*sF,sM*Square(sF));
         fSom = rect_som(fSom,aSzI)/ElSquare(1.0+2.0*aSzI);  // Pour Eviter les divergences
         Symb_FNum  S012 (fSom);

         Symb_FNum s0 (S012.v0());
         Symb_FNum s1 (S012.v1()/s0);
         Symb_FNum s2 (S012.v2()/s0-Square(s1));
         Symb_FNum ect  (sqrt(Max(1,s2)));
	 return 255*erfcc((aFonc-s1)/(ect*sqrt(2.0)));

       }
       break ;

       default:
       break;
   }

   ELISE_ASSERT(false,"Bad ePxApply value");
   return false;
}


cSpecFitrageImage  Str2Filtr(const std::string & aStr)
{
//std::cout << "STR=" << aStr << "\n";
   const char * aC = aStr.c_str();
   ELISE_ASSERT(strlen(aC) > 0,"Chaine vide dans Str2Filtr");
   char aC0 = *aC++;

   std::vector<double> aV;
   while (*aC)
   {
       char * EndC;
       double aD = strtod(aC,&EndC);
       ELISE_ASSERT(EndC != aC,"Non Double in Str2Filtr");
       aV.push_back(aD);
       aC = EndC;
       if (*aC=='|')
         aC++;
   }

   ELISE_ASSERT(aV.size() >= 1,"Pas d'arg num dans Str2Filtr");
   cSpecFitrageImage aRes;
   aRes.SzFiltrNonAd().SetVal(0.0);
   aRes.PxApply().SetVal(eApplyPx12);
   aRes.PatternSelFiltre().SetNoInit();
   aRes.NbIteration().SetVal(1);
   aRes.AmplitudeSignal().SetVal(255);
   

   aRes.SzFiltrage() = aV[0];
   int aNbArgMax = 1; GccUse(aNbArgMax);
   


   if (aC0 == 'M')
      aRes.TypeFiltrage() = eFiltrageMedian;
   else if (aC0 == 'D')
      aRes.TypeFiltrage() = eFiltrageDeriche;
   else if (aC0 == 'G')
   {
      aRes.TypeFiltrage() = eFiltrageGamma;
      if (aV.size() >=2)
         aRes.AmplitudeSignal().SetVal(aV[1]);
      aNbArgMax = 2;
   }
   else if (aC0 == 'E')
      aRes.TypeFiltrage() = eFiltrageEqLoc;
   else
   {
       ELISE_ASSERT(false,"Unnown filtre in Str2Filtr");
   }


    return aRes;
}

tLPFP  Str2Filtr(const std::vector<std::string> & aStr)
{
   tLPFP aRes;
   for (int aK=0 ; aK<int(aStr.size()) ; aK++)
       aRes.push_front(Str2Filtr(aStr[aK]));  // Ordre inver pouq que G2D0.7 soit le gamma du grad
   return aRes;
}

Fonc_Num  FiltrageImMicMac
                 (
                      const cSpecFitrageImage & aParam,
                      Fonc_Num aFonc,
                      Fonc_Num aFoncMasq,
                      double      aSRA
                 )
{
   for (int aK=0; aK<aParam.NbIteration().Val() ; aK++)
       aFonc = OneItereFiltrageImMicMac(aParam,aFonc,aFoncMasq,aSRA);

   return aFonc;
}

Fonc_Num  FiltrageImMicMac
                 (
                      const tLPFP & aLPFP,
                      Fonc_Num aFonc,
                      Fonc_Num aFoncMasq
                 )
{
    for (tLPFP::const_iterator itF=aLPFP.begin(); itF!=aLPFP.end(); itF++)
    {
         aFonc = FiltrageImMicMac(*itF,aFonc,aFoncMasq,1);
    }
    return aFonc;
}

Fonc_Num  FiltrageImMicMac
          (
		const std::vector<std::string> & aParam,
                Fonc_Num aFonc,
                Fonc_Num aFoncMasq
          )
{
    return FiltrageImMicMac(Str2Filtr(aParam),aFonc,aFoncMasq);
}


bool           cSurfaceOptimiseur::MaskCalcDone() {return mMaskCalcDone;}
Im2D_Bits<1>   cSurfaceOptimiseur::MaskCalc()     {return mMaskCalc;}



void cSurfaceOptimiseur::SolveOpt()
{
     mMaskCalcDone = false;
     // Si c'est le cas il faut utiliser le comptage pour normaliser
     // Corr en fonction de Cpt puis transferer par Local_SetCout
     {
         Pt2di aPTer;
         int aPxMin[theDimPxMax] = {0,0};
         int aPxMax[theDimPxMax] = {1,1};
         for (aPTer.y=0 ; aPTer.y<mSzCur.y ; aPTer.y++)
         {
             for (aPTer.x=0 ; aPTer.x<mSzCur.x ; aPTer.x++)
             {
                   mLTCur->GetBornesPax(aPTer,aPxMin,aPxMax);
                   int aVPx[theDimPxMax];
                   // isDefCor  : le coup du 1 def corr, all def corr (def = false)
		   bool isDefCor = mLTCur->IsDefCorrActif(aPTer);


                   for (aVPx[1] = aPxMin[1] ;aVPx[1]<aPxMax[1] ; aVPx[1]++)
                       for (aVPx[0] = aPxMin[0] ;aVPx[0]<aPxMax[0] ; aVPx[0]++)
                       {
                          Pt2di aPPx(aVPx[0],aVPx[1]);
                          if  (mWithEQ)
			  {
                              int aCpt  = (*mReducCpt)[aPTer][aPPx] ;
                              double aCost = (*mReducCost)[aPTer][aPPx];
			      if (isDefCor)
			      {
			          aCost = mDefCost;
			          aCpt  = 1;
			      }
                              // Ce cas est rare mais tout a fait possible
                              else if (aCpt ==0)
                              {
                                   aCpt = 1;
                                   aCost = 1.0;
                              }

                              ELISE_ASSERT(mEBI==0,"Expect no label");
                              Local_SetCout(aPTer,aVPx,aCost/aCpt,0);
                              if (mMemoCorrel)
                              {
                                   Pt2di aPPx = mAppli.Px2Point(aVPx);
                                   (*mMemoCorrel)[aPTer][aPPx] = CostToMemoCorrel(aCost/aCpt);
                              }
                          }
			  else
			  {
			      if (isDefCor)
			      {
                                  ELISE_ASSERT(mEBI==0,"Expect no label");
                                  Local_SetCout(aPTer,aVPx,mDefCost,0);
                                  if (mMemoCorrel)
                                  {
                                       (*mMemoCorrel)[aPTer][aPPx] = CostToMemoCorrel(mDefCost);
                                  }
                              }
			  }
                       }
             }
         }
     }

     Local_SolveOpt(mLTInit.ImCorrelSol());

     // Utilisation d'un eventuel post-filtrage des paralaxe
     const cTplValGesInit< cPostFiltragePx > & aPF = mEtape.EtapeMEC().PostFiltragePx();
     if (aPF.IsInit())
     {
         // ELISE_ASSERT (mMemoCorrel==0, "Filtrage-et-Resultat-Correl incompatibles");
         const tLPFP & aLPFP = aPF.Val().OneFitragePx();
         for (int aK=0 ; aK<int(mImRes.size()) ; aK++)
         {
            Fonc_Num fFiltree = mImRes[aK].in_proj();
            bool GotChgt = false;
            for (tLPFP::const_iterator itF=aLPFP.begin(); itF!=aLPFP.end(); itF++)
            {
                if (Apply(itF->PxApply().Val(),aK))
                {
                   fFiltree = FiltrageImMicMac(*itF,fFiltree,mLTCur->ImMasqTer().in(0),mGlobSR);
                   GotChgt = true;
                }
            }
            if (GotChgt)
            {
              ELISE_COPY
              (
                   mImRes[aK].all_pts(),
                   fFiltree,
                   mImRes[aK].out()
              );
            }
         }
     }

     // Sinon il faut remettre la solution reduite a la
     // resolution initiale

     if  (mWithEQ)
     {
         for (int aK=0 ; aK<mAppli.DimPx() ; aK++)
         {
             if (mEtape.KthNapIsEpaisse(aK))
             {
                TIm2D<INT2,INT> aIR1 (mImResInit[aK]);
                TIm2D<INT2,INT> aIRed (mImRes[aK]);
                Pt2di aSz = aIR1.sz();
                Pt2di aP;
                for (aP.y=0; aP.y<aSz.y ; aP.y++)
                {
                    for (aP.x=0; aP.x<aSz.x ; aP.x++)
                    {
		        Pt2di aPR = ToSRAlg(aP);
                        aIR1.oset(aP,aIRed.get(aPR));
	                
                        mLTInit.SetDefCorrActif(aP,mLTRed->IsDefCorrActif(aPR));
                    }
                }
             }
         }
     }

     if (mMemoCorrel)
     {
        Im2D_U_INT1 aICor = mLTInit.ImCorrelSol();
        TIm2D<U_INT1,INT> aTCor(aICor);
        TIm2DBits<1> aTMaskCal(mMaskCalc);
        Pt2di aSz = aICor.sz();
        Pt2di aP;
        int aVPx[theDimPxMax] ={0,0};
        std::vector<INT2 **> aVRes;
        for (int aK=0 ; aK<mAppli.DimPx() ; aK++)
           aVRes.push_back(mImResInit[aK].data());
        INT2 *** aDRes = &aVRes[0];
        for (aP.y=0; aP.y<aSz.y ; aP.y++)
        {
            for (aP.x=0; aP.x<aSz.x ; aP.x++)
            {
                if((!mMaskCalcDone) || (aTMaskCal.get(aP,0)))
                {
                    for (int aK=0 ; aK<mAppli.DimPx() ; aK++)
                        aVPx[aK] = aDRes[aK][aP.y][aP.x] ;
                    Pt2di aPPx = mAppli.Px2Point(aVPx);
                    const cSmallMatrixOrVar<U_INT1> & aM =(*mMemoCorrel)[ToSRAlg(aP)];
                    const Box2di & aB = aM.Box();
                    Pt2di aPClip = Pt2di
                               (
                                  ElMax(aB._p0.x,ElMin(aPPx.x,aB._p1.x-1)),
                                  ElMax(aB._p0.y,ElMin(aPPx.y,aB._p1.y-1))
                               );

                    aTCor.oset(aP,aM[aPClip]);
                 }
// PB
            }
        }
     }

     // Ensuite on rajoute un eventuel redressement de paralaxe
     for (int aK=0 ; aK<mAppli.DimPx() ; aK++)
     {

         const cOneNappePx & aNap = mLTInit.KthNap(aK);
         if (aNap.mRedrPx && (! mEtape.EtapeMEC().RedrSauvBrut().ValWithDef(false)))
         {
            TIm2D<INT2,INT> aIR1 (mImResInit[aK]);
            TIm2D<REAL4,REAL8> aTPxInit (aNap.mPxRedr);
            Pt2di aSz(mImResInit[aK].sz());

            
            ElImplemDequantifier aDeq(aSz);
            aDeq.SetTraitSpecialCuv(true);
            aDeq.DoDequantif(aSz,mImResInit[aK].in(),true);
            Im2D_REAL4 aIResDeq(aSz.x,aSz.y);
            ELISE_COPY
            (
               aIResDeq.all_pts(),
               aDeq.ImDeqReelle(),
               aIResDeq.out()
            );
            TIm2D<REAL4,REAL8> aTIResDeq (aIResDeq);



            // double aS2=0;
            // double aS3=0;
            Pt2di aP;
            for (aP.y=0; aP.y<aSz.y ; aP.y++)
            {
                for (aP.x=0; aP.x<aSz.x ; aP.x++)
                {
                    double aPxR = aTPxInit.get(aP)+aNap.FromDiscPx(aTIResDeq.get(aP));
                    double aPxI = aNap.ToDiscPx(aPxR);
                    aIR1.oset ( aP, round_ni (aPxI));
/*
                    double aV1 = aNap.ToDiscPx(aTPxInit.get(aP));
                    double aV2 = aTIResDeq.get(aP);
                     aS1 += aV1;
                     aS2 += aV2;
                     aS3 += aTPxInit.get(aP);
*/
                }
            }


         }
     }

    
}

/*********************************************/
/*                                           */
/*       cMaxOfScoreOptimiseur               */
/*                                           */
/*********************************************/

#include "im_tpl/image.h"

class cMaxOfScoreOptimiseur : public cSurfaceOptimiseur
{
    public :
       cMaxOfScoreOptimiseur
       (
           cAppliMICMAC &    anAppli,
           cLoadTer&         aLT,
           const cEquiv1D &        anEqX,
           const cEquiv1D &        anEqY
       );
       ~cMaxOfScoreOptimiseur();
    private :
      void Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int aLabel);
      void Local_SolveOpt(Im2D_U_INT1 );
      Im2D_REAL8            mScoreMax;
      TIm2D<double,double>  mTSM;

};

cMaxOfScoreOptimiseur::cMaxOfScoreOptimiseur
(
   cAppliMICMAC &    anAppli,
   cLoadTer&               aLT,
   const cEquiv1D &        anEqX,
   const cEquiv1D &        anEqY
) :
  cSurfaceOptimiseur  (anAppli,aLT,1.0,anEqX,anEqY,false,false),
  mScoreMax           (aLT.Sz().x,aLT.Sz().y,1e9),
  mTSM                (mScoreMax)
{
}

void cMaxOfScoreOptimiseur::Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int)
{
   if (aCost < mTSM.get(aPTer))
   {
      mTSM.oset(aPTer,aCost);
      for (int aK=0 ; aK<int(mDataImRes.size()) ; aK++)
         mDataImRes[aK][aPTer.y][aPTer.x] =  aPX[aK];
   }
}

void cMaxOfScoreOptimiseur::Local_SolveOpt(Im2D_U_INT1)
{
}

cMaxOfScoreOptimiseur::~cMaxOfScoreOptimiseur()
{
}

cSurfaceOptimiseur * cSurfaceOptimiseur::AllocCoxMaxOfCorrel
                     (
                         cAppliMICMAC &    anAppli,
                         cLoadTer&         aLT,
                         const cEquiv1D &        anEqX,
                         const cEquiv1D &        anEqY
                     )
{
   return new cMaxOfScoreOptimiseur(anAppli,aLT,anEqX,anEqY);
}

/*********************************************/
/*                                           */
/*       cCoxRoyOptimiseur                   */
/*                                           */
/*********************************************/


class cCoxRoyOptimiseur : public cSurfaceOptimiseur
{
    public :
       cCoxRoyOptimiseur
       (
           cAppliMICMAC &    anAppli,
           cLoadTer&         aLT,
           bool                    OnUChar,
           const cEquiv1D &        anEqX,
           const cEquiv1D &        anEqY
       );
       ~cCoxRoyOptimiseur();
    private :
      int                    mNumNap;
      const cOneNappePx &    mONP;
      cInterfaceCoxRoyAlgo * mICRA;
      void Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int);
      void Local_SolveOpt(Im2D_U_INT1);

};


cCoxRoyOptimiseur::cCoxRoyOptimiseur
(
   cAppliMICMAC &    anAppli,
   cLoadTer&               aLT,
   bool                    OnUChar,
   const cEquiv1D &        anEqX,
   const cEquiv1D &        anEqY
) :
   cSurfaceOptimiseur   (anAppli,aLT,OnUChar ? 1e2 : 1e4,anEqX,anEqY,false,false),
   mNumNap              (mEtape.NumSeuleNapEp()),
   mONP                 (mLTCur->KthNap(mNumNap)),
   mICRA                ( cInterfaceCoxRoyAlgo::NewOne
                          (
                              mSzCur.x,mSzCur.y,
                              mONP.mImPxMin.data(),
                              mONP.mImPxMax.data(),
                              mEtape.EtapeMEC().CoxRoy8Cnx().ValWithDef(false),
                              OnUChar
                          )
                        )
{
}

void cCoxRoyOptimiseur::Local_SetCout(Pt2di aPTer,int * aPX,REAL aCost,int )
{
   mICRA->SetCostVert(aPTer.x,aPTer.y,aPX[mNumNap],CostR2I(aCost));
}

cCoxRoyOptimiseur::~cCoxRoyOptimiseur()
{
   delete mICRA;
}

void cCoxRoyOptimiseur::Local_SolveOpt(Im2D_U_INT1)
{
   mICRA->SetStdCostRegul(0,CostAmpl(mCostRegul[mNumNap]),1);
   mICRA->TopMaxFlowStd(mImRes[mNumNap].data());
}


cSurfaceOptimiseur * cSurfaceOptimiseur::AllocCoxRoy
                     (
                         cAppliMICMAC &    anAppli,
                         cLoadTer&         aLT,
                         const cEquiv1D &        anEqX,
                         const cEquiv1D &        anEqY
                     )
{
   bool OnUC = anAppli.CurEtape()->EtapeMEC().CoxRoyUChar().ValWithDef(true);
   return new cCoxRoyOptimiseur(anAppli,aLT,OnUC,anEqX,anEqY);
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
