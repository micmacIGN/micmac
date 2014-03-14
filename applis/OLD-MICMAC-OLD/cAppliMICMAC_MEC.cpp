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

int cAppliMICMAC::MemSizePixelImage() const
{
    // Nombre d'octet par pixel et par image,
    // compte tenu d'un rab de recouvrement
    return   12;
}



double cAppliMICMAC::MemSizeProcess(int aTxy) const
{

   double aRes=     ElSquare(aTxy+2.0*mSzRec)
	   *  (
	         mMemPart.NbMaxImageOn1Point().Val() * MemSizePixelImage() 
               + mCurEtape->MemSizePixelAlgo()
	      );

  return aRes;
}

int cAppliMICMAC::GetTXY() const
{
    int aSz=mSzDMin;
    int aStep = 20;
    while
    ( 
          (aSz <mSzDMax)
       && (MemSizeProcess(aSz)<(AvalaibleMemory().Val()*1e6))
    )
	   aSz += aStep;
    return aSz-aStep;
}



void cAppliMICMAC::DoAllMEC()
{
     for 
     (
        tContEMC::const_iterator itE = mEtapesMecComp.begin();
	itE != mEtapesMecComp.end();
	itE++
     )
     { 
        OneEtapeSetCur(**itE);
        if (DoMEC().Val()  && (!DoNothingBut().IsInit()))
           DoOneEtapeMEC(**itE);
        if (
                 ( (*itE)->Num()>=FirstEtapeMEC().Val())
           &&    ( (*itE)->Num()<LastEtapeMEC().Val())
           )
        {
            if (! CalledByProcess().Val())
               MakeResultOfEtape(**itE);


            if (
                      (DoMEC().Val()  && (!DoNothingBut().IsInit()))
                 ||   (DoNothingBut().IsInit() && (ButDoOrtho().Val()||ButDoPartiesCachees().Val()))
                 ||   ( Paral_Pc_NbProcess().IsInit())
               )
            {
               MakePartiesCachees();
            }


            if (     (! CalledByProcess().Val())
                 &&  (
                            (DoMEC().Val()  && (!DoNothingBut().IsInit()))
                        ||  (DoNothingBut().IsInit() &&  ButDoRedrLocAnam().Val())
                     )
               )
            {
               MakeRedrLocAnam();
            }
        }
     }
}

/*
 Formule ancienne, egalise les ecart moyen avec une somme discrete et
 approximation
*/

double Old_FromSzW2FactExp(double aSzW,double mCurNbIterFenSpec)
{
    aSzW  /= sqrt(mCurNbIterFenSpec);
   return (2*aSzW+1)/(2*aSzW+5);
}

/* Formule nouvelle, egalise les ecart type avec une somme integrale,
 * sans approximation
 */

/*
double FromSzW2FactExp(double aSzW,double mCurNbIterFenSpec)
{
   double aRes = exp(- (sqrt(6*mCurNbIterFenSpec))/(1+aSzW));
   // std::cout << "ANCIEN " << Old_FromSzW2FactExp(aSzW,mCurNbIterFenSpec) << "\n";
   // std::cout << "FromSzW2FactExp : " << aRes << "\n"; getchar();
   return aRes;
}
*/


void cAppliMICMAC::OneEtapeSetCur(cEtapeMecComp & anEtape)
{
     mCurEtape = & anEtape;
     mEBI = mCurEtape->EBI();
     const cEtapeMEC & anEM = mCurEtape->EtapeMEC();

     mSzWR.x = anEM.SzW().Val();
     mSzWR.y = anEM.SzWy().ValWithDef(mSzWR.x);
     mCurTypeWC = anEM.TypeWCorr().ValWithDef(eWInCorrelFixe);
     mCurFenSpec = (mCurTypeWC != eWInCorrelFixe);

     if (mCurFenSpec)
     {
         ELISE_ASSERT(mMapEquiv==0,"Incompatibilite Fentre-Speciale/Equivalenc");
     }
     mCurNbIterFenSpec =  anEM.NbIterFenSpec().ValWithDef(1);

     mCurWSpecUseMasqGlob =   mCurFenSpec
                           && anEM.WSpecUseMasqGlob().ValWithDef(mNbPDV==2);

     mFactFenetreExp.x = FromSzW2FactExp(mSzWR.x,mCurNbIterFenSpec);
     mFactFenetreExp.y = FromSzW2FactExp(mSzWR.y,mCurNbIterFenSpec);

     mCurForceCorrelPontcuelle = false;
     mCurForceCorrelByRect = false;

     if (InterditCorrelRapide().Val())
     {
         mCurForceCorrelPontcuelle = true;
     }


     if (anEM.AggregCorr().Val()==eAggregInfoMut)
     {
         mCurForceCorrelPontcuelle =true;
     }
     if (ForceCorrelationByRect().Val())
     {
	 mCurForceCorrelByRect = true;
     }
     
     if (mCurFenSpec)
     {
         mSzWFixe = round_ni(mSzWR.x);
	 if (mCurTypeWC == eWInCorrelRectSpec)
	    mPtSzWMarge = round_up(mSzWR)+Pt2di(1,1); // Un peu lache
	 else
            mPtSzWMarge = round_ni(mSzWR*3.0);

 	 mCurForceCorrelByRect = true;
	 ELISE_ASSERT
	 (
	     !mCurEtape->EtapeMEC().SzWInt().IsInit(),
	     "En mode FenSpec pas de SzWInt"
	 );
	 ELISE_ASSERT
	 (
	     !mCurEtape->EtapeMEC().SurEchWCor().IsInit(),
	     "En mode FenSpec pas de SurEchWCor"
	 );
     }
     else
     {
         mSzWFixe = round_ni(mSzWR.x);
         mPtSzWMarge = round_ni(mSzWR);
         ELISE_ASSERT
	 (
	      mSzWFixe==mSzWR.x,
	      "Fenetre reelle en mode fenetre fixe"
	 );
         ELISE_ASSERT
	 (
	      mSzWR.x==mSzWR.y,
	      "tx!=ty en mode fenetre fixe"
	 );
     }
     mPtSzWFixe = Pt2di(mSzWFixe,mSzWFixe);


     mModeIm1Maitre = IsModeIm1Maitre(mCurEtape->EtapeMEC().AggregCorr().Val());
     mCurSurEchWCor =  mCurEtape->EtapeMEC().SurEchWCor().ValWithDef(1);
     mCurSzWInt =  mCurEtape->EtapeMEC().SzWInt().ValWithDef(mSzWFixe*mCurSurEchWCor);

     if (mCurSurEchWCor != 1)
     {
        mCurForceCorrelPontcuelle = true;
     }

     if (mModeIm1Maitre)
     {
         ELISE_ASSERT
         (
              mGIm1IsInPax,
              "Aggreg with Im1 Master requires Geom Im1 master"
         );
     }


     mCorrelAdHoc = anEM.CorrelAdHoc().PtrVal();
     if (mCorrelAdHoc)
     {
         mCurForceCorrelPontcuelle = true;
         ELISE_ASSERT(mDimPx==1,"Multiple Px in GPU");
         ELISE_ASSERT(mCurSurEchWCor==1,"Sur ech in GPU");
/*
         ELISE_ASSERT(mCurEtape->EtapeMEC().AggregCorr().Val()==eAggregSymetrique,"Aggreg non sym in GPU");
         ELISE_ASSERT(mCurEtape->EtapeMEC().ModeInterpolation().Val()==eInterpolMPD,"Interp non MPD in GPU");
*/


     }


     mIsOptDiffer = anEtape.IsOptDiffer();
     mIsOptDequant = anEtape.IsOptDequant();
     mIsOptimCont = anEtape.IsOptimCont();

     mCurCarDZ =  GetCaracOfDZ(mCurEtape->DeZoomTer());
     mSom0 = ElSquare(1+2*mSzWFixe * mCurSurEchWCor);
     if (
             (! CalledByProcess().Val())
          ||(anEtape.Num() <LastEtapeMEC().Val())
        )
     {
        mCurMAI = mCurEtape->ModeleAnImported();
        if (mCurMAI)
        {
           mCurMAI->LoadMA();
           for 
           (
               tCsteIterPDV itP = PdvBegin();
	       itP != PdvEnd();
	       itP++
           )
           {
                (*itP)->Geom().CorrectModeleAnalytique(mCurMAI);
           }
        }
     }
     ELISE_ASSERT
     (
        ! (mCurForceCorrelByRect && mCurForceCorrelPontcuelle),
	"Incohe mCurForceCorrelByRect/mCurForceCorrelPontcuelle"
     );
}


std::string cAppliMICMAC::PrefixGenerikRecalEtapeMicmMac(cEtapeMecComp & anEtape)
{
    int aNumEt = anEtape.Num();
    //mNameExe + std::string(" ")
    //  Modif MPD, reordonne pour mettre Arg d'etape a la fin
    //  Modif Greg: probleme de '"' pour condor
    //std::string aNameProcess = std::string("\"")+mNameXML+std::string("\"")
   std::string aNameProcess = mNameXML 
                               + std::string(" CalledByProcess=1 ")
                               + std::string(" ByProcess=0 ");

    // MODIF MPD mise entre " des parametre pour etre completement reentrant
    for (int aKArg=0; aKArg<mNbArgAux ; aKArg++)
        aNameProcess =   aNameProcess
                       + std::string(" \"")
                       + std::string(mArgAux[aKArg]) 
                       +  std::string("\"");

    aNameProcess = aNameProcess
                   + std::string(" FirstEtapeMEC=") + ToString(aNumEt)
                   + std::string(" LastEtapeMEC=") + ToString(aNumEt+1);

    return aNameProcess;
/*
                   + std::string(" FirstBoiteMEC=") + ToString(mKBox)
                   + std::string(" NbBoitesMEC=1") ;
*/

}



void cAppliMICMAC::DoOneEtapeMEC(cEtapeMecComp & anEtape)
{
     if (
            (anEtape.Num() <FirstEtapeMEC().Val())
          ||(anEtape.Num() >=LastEtapeMEC().Val())
          ||(mNbBoitesToDo <=0)
        )
        return;

    std::list<std::string> aLStrProcess;
    if (mShowMes)
    {
        mCout << "-------- BEGIN ETAPE,  "
           << ", Num = " << anEtape.Num()
           << ", DeZoomTer = " << anEtape.DeZoomTer()
           << ", DeZoomIm = " << anEtape.DeZoomIm()
           << "\n";
    }



     mNbPtsWFixe = (1+2*mPtSzWFixe.x*mCurSurEchWCor)*(1+2*mPtSzWFixe.y*mCurSurEchWCor);
     int aDZIm = anEtape.DeZoomIm();
     
     mVecPtsW = std::vector<Pt2dr>(mNbPtsWFixe,Pt2dr(0,0));
     mTabPtsW = &mVecPtsW[0];

     mGeomDFPx = mCurEtape->GeomTer();
     for 
     (
         tCsteIterPDV itP = PdvBegin();
	 itP != PdvEnd();
	 itP++
     )
     {
         (*itP)->Geom().SetDeZoomIm(aDZIm);
     }

     Box2dr aPropClip = ProportionClipMEC().ValWithDef
                     (Box2dr(Pt2dr(0,0),Pt2dr(1,1)));
     Pt2dr aSzDz = Pt2dr(mGeomDFPx.SzDz());
     Box2di aBoxClip 
            (
                   aPropClip._p0.mcbyc(aSzDz),
                   aPropClip._p1.mcbyc(aSzDz)
            );
      if ((!  ClipMecIsProp().Val()) && ProportionClipMEC().IsInit())
      {
         Box2dr  aB = ProportionClipMEC().Val();
         double aZ =ZoomClipMEC().Val();
         double aF = aZ / anEtape.DeZoomTer();

         aBoxClip = Box2di(aB._p0 * aF,aB._p1 * aF);
      }

/*
std::cout << "CCMMM = " << aBoxClip._p0 << " " << aBoxClip._p1 << "\n"; getchar();
*/

      const cEtapeMEC & anEM = anEtape.EtapeMEC();
      if (mPtrIV)
      {
          Pt2di aCTer = mPtrIV->CentreVisuTerrain();
          int aSzWV   = mPtrIV->SzWTerr().Val()/2;
          aBoxClip._p0.SetSup(aCTer - Pt2di(aSzWV,aSzWV));
          aBoxClip._p1.SetInf(aCTer + Pt2di(aSzWV,aSzWV));
      }
      mSzRec =  anEM.SzRecouvrtDalles().ValWithDef(SzRecouvrtDalles().Val());
      mSzDMin=  anEM.SzDalleMin().ValWithDef(SzDalleMin().Val());
      mSzDMax=  anEM.SzDalleMax().ValWithDef(SzDalleMax().Val());


      int aTXY = GetTXY();
      if (mPtrIV)
          aTXY = 100000;
      Pt2di aPRec(mSzRec,mSzRec);

      cDecoupageInterv2D aDecInterv
                         (
                              aBoxClip,
                              Pt2di(aTXY,aTXY),
                              Box2di(-aPRec,aPRec)
                         );
     
     for (mKBox=0 ; mKBox<aDecInterv.NbInterv() ; mKBox++)
     {
          if (
                (   (mKBox>=FirstBoiteMEC().Val())
                 || (anEtape.Num() >FirstEtapeMEC().Val())
                )
                && (mNbBoitesToDo >0)
             )
          {
               if (mShowMes)
               {
                  mCout << "   -- BEGIN BLOC  "
                        << "  Bloc= " << mKBox 
                        << ", Out of " << aDecInterv.NbInterv()  
                        << "\n";
               }
               if (ByProcess().Val()==0)
               {
                  DoOneBloc
                  (
                      aDecInterv.KthIntervOut(mKBox),
                      aDecInterv.KthIntervIn(mKBox),
                      0,
                      aBoxClip
                  );
               }
               else
               {
/*
                   int aNumEt = anEtape.Num();
                   //mNameExe + std::string(" ")
		   //  Modif MPD, reordonne pour mettre Arg d'etape a la fin
                   //  Modif Greg: probleme de '"' pour condor
                   //std::string aNameProcess = std::string("\"")+mNameXML+std::string("\"")
		   std::string aNameProcess = mNameXML 
                               + std::string(" CalledByProcess=1 ")
                               + std::string(" ByProcess=0 ");

                   for (int aKArg=0; aKArg<mNbArgAux ; aKArg++)
                       aNameProcess =   aNameProcess
                                      + std::string(" ")
                                      + std::string(mArgAux[aKArg]);

                   aNameProcess = aNameProcess
                               + std::string(" FirstEtapeMEC=") + ToString(aNumEt)
                               + std::string(" LastEtapeMEC=") + ToString(aNumEt+1)
*/
		   std::string aNameProcess = 
                                 PrefixGenerikRecalEtapeMicmMac(anEtape)
                               + std::string(" FirstBoiteMEC=") + ToString(mKBox)
                               + std::string(" NbBoitesMEC=1") ;

                   aLStrProcess.push_back(aNameProcess);
               }
          }
     }
     if (ByProcess().Val()!=0)
        ExeProcessParallelisable(true,aLStrProcess);

}


void cAppliMICMAC::SauvFileChantier(Fonc_Num aF,Tiff_Im aFile) const
{
      ELISE_COPY
      (
          rectangle(mBoxOut._p0,mBoxOut._p1),
          trans(aF,-mBoxIn._p0),
          aFile.out()
      );

}

void cAppliMICMAC::DoOneBloc
     (
          const Box2di & aBoxOut,
          const Box2di & aBoxIn,
          int aNiv,
          const Box2di & aBoxGlob
     )
{
   // std::cout << "DO ONE BLOC " << aBoxOut._p0 << " " << aBoxIn._p0 << "\n";
   //  mStatN =0;
   mStatGlob =0;
   mLTer = 0;
   mSurfOpt = 0;


   mBoxIn = aBoxIn;
   mBoxOut = aBoxOut;
   mLTer = new cLoadTer(mDimPx,aBoxIn.sz(),*mCurEtape);
   double aNbCel = mCurEtape->LoadNappesAndSetGeom(*mLTer,aBoxIn);

   int aLMin = ElMin(aBoxOut._p1.x-aBoxOut._p0.x,aBoxOut._p1.y-aBoxOut._p0.y);

/*
   for (int aK=0; aK<=aNiv; aK++)  std::cout <<"---|";
   std::cout << "OUT " << aBoxOut._p0 << aBoxOut._p1 << " IN " << aBoxIn._p0 << aBoxIn._p1
             << "  NbCel " << aNbCel << "\n"; 
*/

   if (
              aNbCel>NbCelluleMax().Val()
          // && (aBoxOut._p1.x-aBoxOut._p0.x) > (3*mSzRec+5)  // Evite recursion trop profonde, voir infinie
          // && (aBoxOut._p1.y-aBoxOut._p0.y) > (3*mSzRec+5)
      )
   {
       cLoadTer * aLTInit = mLTer;
       int aSzRecInit = mSzRec;

       Box2di aSplitOut[4];
       aBoxOut.QSplit(aSplitOut);

       mSzRec = ElMax(0,ElMin(mSzRec,aLMin/4));

       for (int aK=0 ; aK<4 ; aK++)
       {
/*
           std::cout  << "SPLIT " << aBoxOut._p0 << aBoxOut._p1  
                      << "  ; O = " << aBoxIn._p0 << aBoxIn._p1  
                      <<  "  => OUT " << aSplitOut[aK]._p0 << aSplitOut[aK]._p1 
                      <<  " REC " << mSzRec
                      <<  "  => IN " << aSplitIn[aK]._p0 << aSplitIn[aK]._p1 << "\n";
*/
           // getchar();
           DoOneBloc
           (
                aSplitOut[aK],
                Inf(aBoxGlob,aSplitOut[aK].dilate(mSzRec)),
                aNiv+1,
                aBoxGlob
          );
       }

       mSzRec = aSzRecInit;
       mLTer = aLTInit;
       return;
   }



   mGeomDFPx = mCurEtape->GeomTer();  // actualisation en fonction du LoadNappes

   cEquiv1D anEqX;
   anEqX.InitByClipAndTr(mCurCarDZ->EqX(),mBoxIn._p0.x,0,mBoxIn.sz().x);
   cEquiv1D anEqY;
   anEqY.InitByClipAndTr(mCurCarDZ->EqY(),mBoxIn._p0.y,0,mBoxIn.sz().y);

  // Les structures de ChoixCalcCorrByQT sont assez consommatrices de
  // memoires, le surcout a spliter en different rectangle etant assez
  // faible, on a tout interet a le faire
   
/*
  int aRatio = 20;
  // Qd on utilise mCurFenSpec, bcp + sensible a la decoupe
  if (mCurFenSpec) 
      aRatio = 60;
*/
  int aSzMaxChCorr= round_ni(100 * SzMinDecomposCalc().Val());
  if  (    GetCurCaracOfDZ()->HasMasqPtsInt() 
        || (mCurSurEchWCor != 1)
        || mIsOptimCont 
        )
      aSzMaxChCorr = 10000;
  cDecoupageInterv2D aDecInterv =
	                cDecoupageInterv2D::SimpleDec(aBoxIn.sz(),aSzMaxChCorr,0);
  Pt2di aSzMaxDec = aDecInterv.SzMaxOut();

   
   // Chargement des images

   mPDVBoxGlobAct.clear();
   bool isFirstImLoaded = true;
   for (tCsteIterPDV itFI=PdvBegin(); itFI!=PdvEnd(); itFI++)
   {
       // bool Loaded = false;
       if (
                (mCurEtape->SelectImage(*itFI))
            &&  (*itFI)->LoadImage(*mLTer,aSzMaxDec,isFirstImLoaded)
          )
       {
         // Loaded = true;
          isFirstImLoaded = false;
          if ((mKBox == 0) && ShowLoadedImage().Val())
          {
               mCout << "== " << (*itFI)->Name() << "\n";
          }
          mPDVBoxGlobAct.push_back(*itFI);

          if (mCurEtape->UsePC() || (*itFI)->Geom().UseMasqAnam())
          {
              (*itFI)->LoadedIm().MakePC
                       (
                           **itFI,
                           mCurEtape,
                           mCurEtape->PredPC(),
                           aBoxIn,
                           mCurEtape->UsePC(),
                           (*itFI)->Geom().UseMasqAnam()
                       );
          }
       }
   }
   mNbImChCalc = (int) mPDVBoxGlobAct.size();
   if (mShowMes)
      mCout << "       Images Loaded\n";

   // Initialisation de mTabV1 et aVPtInEc
   //  mTabV1 : zone memoire pour stocker les valeurs de Im1 (acceleration "speciale")
   // aVPtInEc : au cas ou fenetre de correl != fenetre de normalisation
   std::vector<int> aVPtInEc;
   std::vector<int> aVPtIndOK;
   std::vector<Pt2di> aVPtOK;


   mVecV1.clear();
   int aNbTot = mSzWFixe *mCurSurEchWCor;
   for (int anY=-aNbTot ; anY<=aNbTot ;anY++)
   {
       for (int anX=-aNbTot ; anX<=aNbTot ;anX++)
       {
            aVPtInEc.push_back(dist8(Pt2di(anX,anY))<=mCurSzWInt);
            if (aVPtInEc.back())
            {
               aVPtIndOK.push_back(aVPtInEc.size()-1);
               aVPtOK.push_back(Pt2di(anX,anY));
            }
            mVecV1.push_back(0);
       }
   }
   mTabV1 = &(mVecV1[0]);


   if (mPtrIV)
   {
      Visualisation(aBoxIn);
      exit(0);
   }
   // --------------------------------------
   double aTimeCorrel = 0;
   ElTimer aChrono;

   if (mIsOptimCont)
   {
      OptimisationContinue();
   }
   else
   {

       // mStatN = new cStatOneClassEquiv(*this,(int) mPDVBoxGlobAct.size(),aVPtInEc);

       mStatGlob = new cStatGlob(*this,aVPtInEc,aVPtIndOK,aVPtOK);

       for (int aKPdv=0;aKPdv<int( mPDVBoxGlobAct.size()); aKPdv++)
       {
           mStatGlob->AddVue(*(mPDVBoxGlobAct[aKPdv]));
       }

       mStatGlob->InitSOCE();
   
       mDefCost =  mStatGlob->CorrelToCout(mDefCorr);
       mSurfOpt = cSurfaceOptimiseur::Alloc(*this,*mLTer,anEqX,anEqY);


       InitCostCalcCorrel();



       if (mCorrelAdHoc)
       {
          GlobDoCorrelAdHoc(aBoxIn);
       }
       else
       {
          for (int aKBox=0 ; aKBox<aDecInterv.NbInterv() ; aKBox++)
          {
              ChoixCalcCorrByQT(aDecInterv.KthIntervOut(aKBox));
          }  
       }

   
        aTimeCorrel = aChrono.ValAndInit();
        if (mShowMes)
            mCout << "       Correl Calc, Begin Opt\n";
        mSurfOpt->SolveOpt();
    }



// for (int aK=0; aK<=aNiv; aK++)  std::cout <<"---|";
// std::cout << "AVANT SN " << "\n";

    mCurEtape->SauvNappes(*mLTer,aBoxOut,aBoxIn);

// for (int aK=0; aK<=aNiv; aK++)  std::cout <<"---|";
// std::cout << "APRES SN  " << "\n";


    double aTimeOptim = aChrono.ValAndInit();

    mTimeTotCorrel += aTimeCorrel;
    mTimeTotOptim += aTimeOptim;

    if (mShowMes)
    {
       double aNbPtsTot= mNbPointsByRect2 + mNbPointsByRectN + mNbPointsIsole + mNbPointByRectGen;
       mCout 
             << "       TCor " << mTimeTotCorrel 
             << " CTimeC " << (mTimeTotCorrel * 1e6)/ aNbPtsTot
             <<       " TOpt " << mTimeTotOptim
             << " Pts , R2 " <<  (mNbPointsByRect2/aNbPtsTot)*100.0
             << ", RN " <<  (mNbPointsByRectN/aNbPtsTot)*100.0
             << " Pts , R-GEN " <<  (mNbPointByRectGen/aNbPtsTot)*100.0
             << ", Isol " << (mNbPointsIsole/aNbPtsTot)*100.0
             << "  PT  "   <<  aNbPtsTot

             <<"\n";
    }
  
    //  delete mStatN;
    delete mStatGlob;
    delete mLTer;
    delete mSurfOpt;
    mSurfOpt = 0;
    mNbBoitesToDo--;
}


void cAppliMICMAC::InitBlocInterne( const Box2di & aBox)
{
   mBoxInterne = aBox;
   int aPxMin[theDimPxMax],aPxMax[theDimPxMax];   
   mLTer->CalculBornesPax(aBox,aPxMin,aPxMax);

   // On calcule les images qui "voient" au - une petite partie du terrain
   mPDVBoxInterneAct.clear();
   mNb_PDVBoxInterne=0;
   for (tCsteIterPDV itFI=mPDVBoxGlobAct.begin(); itFI!=mPDVBoxGlobAct.end(); itFI++)
   {
       if (  (*itFI)->Geom().BoxTerHasIntersection
             (
                 mGeomDFPx,
                 aPxMin,aPxMax,
                 Box2dr(aBox._p0,aBox._p1)
             )
          )
       {
          mPDVBoxInterneAct.push_back(*itFI);
          mNb_PDVBoxInterne++;
       }
   }

}

void cAppliMICMAC::DoOneBlocInterne
     (
            const cPxSelector & aSelector,
            const Box2di & aBox
     )
{

   InitBlocInterne(aBox);

/*
   mBoxInterne = aBox;
   int aPxMin[theDimPxMax],aPxMax[theDimPxMax];   
   mLTer->CalculBornesPax(aBox,aPxMin,aPxMax);

   // On calcule les images qui "voient" au - une petite partie du terrain
   mPDVBoxInterneAct.clear();
   mNb_PDVBoxInterne=0;
   for (tCsteIterPDV itFI=mPDVBoxGlobAct.begin(); itFI!=mPDVBoxGlobAct.end(); itFI++)
   {
       if (  (*itFI)->Geom().BoxTerHasIntersection
             (
                 mGeomDFPx,
                 aPxMin,aPxMax,
                 Box2dr(aBox._p0,aBox._p1)
             )
          )
       {
          mPDVBoxInterneAct.push_back(*itFI);
          mNb_PDVBoxInterne++;
       }
   }
*/


  // ii  const cEtapeMEC & anEM = mCurEtape->EtapeMEC();
   mCSAccelIm1 =    (mModeGeomMEC==eGeomMECIm1) 
                 &&  (
		          ( mCurEtape->EtapeMEC().AggregCorr().Val()==eAggregSymetrique)
		      ||  IsModeIm1Maitre(mCurEtape->EtapeMEC().AggregCorr().Val())
	             )
                 &&  (! mCurEtape->UsePC())
                 &&  (! mCurForceCorrelByRect)
                 && (mNb_PDVBoxInterne==2) 
                 && (mNbPDV==2) 
                 && (mPDV1 == *(mPDVBoxInterneAct.begin()))
                 && (mSzWFixe==mCurSzWInt)
                 && (mMapEquiv==0)
                 && (! InterditAccelerationCorrSpec().Val());


   if (mCorrelAdHoc)
   {
       DoCorrelAdHoc(aBox);
       return;
   }
   
   for (mCurPterI.x = aBox._p0.x ; mCurPterI.x <aBox._p1.x ; mCurPterI.x++)
   {
       for (mCurPterI.y = aBox._p0.y ; mCurPterI.y <aBox._p1.y ; mCurPterI.y++)
       {
           int aPxMax[theDimPxMax] ={1,1};
           int aPxMin[theDimPxMax] ={0,0};
           mLTer->GetBornesPax(mCurPterI,aPxMin,aPxMax);
           if (mLTer->IsInMasq(mCurPterI))
           {
               mCurBoxW._p0 = mCurPterI-mPtSzWFixe;
               mCurBoxW._p1 = mCurPterI+mPtSzWFixe;
               // Pour traiter de maniere generique 1 ou 2 dimension
               // de paralaxe, on va toujours faire comme si il y en 
               // avait deux, on s'arrange pour que la dimension inutile ne
               // rajoute qu'un imbrication dans une boucle vide


   
               // IsFirst evite de faire des PrepareAccelIm1 quand ce ne sera utile
               // pour aucune paralaxe, ce qui est le cas avec des tres grande fenetre
               // ou tout a deja ete fait par algo rapide
               bool isFirst = true;
               for (mCurPxI[1] = aPxMin[1] ; mCurPxI[1]<aPxMax[1] ; mCurPxI[1]++)
               {
                   for (mCurPxI[0]=aPxMin[0] ; mCurPxI[0]<aPxMax[0] ; mCurPxI[0]++)
                   {
                       if (aSelector.SelectPx(mCurPxI))
                       {
		          ELISE_ASSERT
			  (
			      ! mCurForceCorrelByRect,
			      "MICMAC:Incoherence interne / mCurForceCorrelByRect"
			  );
                          if (isFirst && mCSAccelIm1)
                          {
                              PrepareAccelIm1(aPxMin);
                          } 
                          REAL aCost =  mCSAccelIm1 ? 
                                        CalculAccelereScore() :
                                        CalculScore();
                          mSurfOpt->SetCout(mCurPterI,mCurPxI,aCost);
                          isFirst = false;
                       }
                   }
               }
           }
           else
           {
               double aDefCost =  mStatGlob->CorrelToCout(mDefCorr);
               for (mCurPxI[1] = aPxMin[1] ; mCurPxI[1]<aPxMax[1] ; mCurPxI[1]++)
               {
                   for (mCurPxI[0]=aPxMin[0] ; mCurPxI[0]<aPxMax[0] ; mCurPxI[0]++)
                   {
                       if (aSelector.SelectPx(mCurPxI))
                       {
                          mSurfOpt->SetCout(mCurPterI,mCurPxI,aDefCost);
                       }
                   }
               }
           }
       }
   }
// std::cout << "END " << aChrono.uval() << "\n";
}


void cAppliMICMAC::PrepareAccelIm1(int * aPx)
{
     mLineProj.InitLP
     (
        mCurBoxW,
        aPx,
        mPDV1->Geom(),
        mGeomDFPx,
        *mCurEtape,
        *mLTer,
        Pt2di(0,0),
        mCurSurEchWCor
     );

     mOkStatIm1 = mPDV1->LoadedIm().StatIm1(mLineProj,mSomI1,mSomI11,mTabV1);

     mSomI1 /= mSom0;
     mSomI11 /= mSom0;
     mSomI11 -= ElSquare(mSomI1);
}

REAL cAppliMICMAC::CalculAccelereScore()
{
    mNbPointsIsole++;
    if (! mOkStatIm1)
       return  mStatGlob->CorrelToCout(mDefCorr);
    mLineProj.InitLP
    (
        mCurBoxW,
        mCurPxI,
        mPDV2->Geom(),
        mGeomDFPx,
        *mCurEtape,
        *mLTer,
        Pt2di(0,0),
        mCurSurEchWCor
    );
    bool aOkStatIm2 = mPDV2->LoadedIm().StatIm2(mLineProj,mTabV1,mSomI2,mSomI22,mSomI12);

    if (! aOkStatIm2)
       return  mStatGlob->CorrelToCout(mDefCorr);

     mSomI2 /= mSom0;
     mSomI22 /= mSom0;
     mSomI22 -= ElSquare(mSomI2);

     mSomI12 /= mSom0;
     mSomI12 -= mSomI1 * mSomI2;
     double aCor = mSomI12 /sqrt(ElMax(mEpsCorr,mSomI11*mSomI22));


    return mStatGlob->CorrelToCout(aCor);
}


REAL cAppliMICMAC::CalculScore()
{
   mNbPointsIsole++;
   mStatGlob->Clear();
   mStatGlob->SetSomsMade(false);
   for 
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
       itFI!=mPDVBoxInterneAct.end(); 
       itFI++
   )
   {
       const cLoadedImage & aLIm = (*itFI)->LoadedIm();
       if (aLIm.IsVisible(mCurPterI))
       {
           mLineProj.InitLP
           (
                mCurBoxW,
                mCurPxI,
                (*itFI)->Geom(),
                mGeomDFPx,
                *mCurEtape,
                *mLTer,
                Pt2di(0,0),
                mCurSurEchWCor
           );
           bool IsOk = true;
           Pt2dr * mTopPts = mTabPtsW;
           while ( mLineProj.Continuer() && IsOk)
           {
                 Pt2dr aP = mLineProj.PCurIm();
                 if(aLIm.IsOk(aP))
                 {
                     *(mTopPts++) = aP;
                 }
                 else
                 {
                    IsOk =false;
                 }
                 mLineProj.NexStep();
           }
           if (IsOk)
           {
             cStat1Distrib *aDist = mStatGlob->NextDistrib(**itFI);
             aLIm.GetVals(mTabPtsW,aDist->Vals(),mNbPtsWFixe);
           }
       }
   }
   return mStatGlob->Cout();
}


void   cAppliMICMAC::CalcCorrelByRect(Box2di aBox,int * aPx)
{

   mLTer->MakeImTerOfPx(aBox,aPx);
   mStatGlob->SetSomsMade(false);
   mPDVBoxInterneAct.clear();
   for 
   (
       tCsteIterPDV itFI=mPDVBoxGlobAct.begin(); 
       itFI!=mPDVBoxGlobAct.end(); 
       itFI++
   )
   {
       if (  (*itFI)->Geom().BoxTerHasIntersection
             (
                 mGeomDFPx,
                 aPx,aPx,
                 Box2dr(aBox._p0,aBox._p1)
             )
          )
       {
             mPDVBoxInterneAct.push_back(*itFI);
       }
   }

   // On le fait en deux fois, car LoadImInGeomTerr depend
   // de la taille de mPDVBoxInterneAct
   bool IsFirstLIIGT = true;
   for 
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
       itFI!=mPDVBoxInterneAct.end(); 
       itFI++
   )
   {
        (*itFI)->LoadedIm().LoadImInGeomTerr(aBox,aPx,IsFirstLIIGT);
	IsFirstLIIGT = false;
   }
   for 
   (
       tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
       itFI!=mPDVBoxInterneAct.end(); 
       itFI++
   )
   {
        (*itFI)->LoadedIm().PostLoadImInGeomTerr(aBox);
   }


   if (UseAlgoSpecifCorrelRect())
   {
       INT aNbIm = (int) mPDVBoxInterneAct.size();
       if (aNbIm<2)
       {
           TIm2D<REAL8,REAL8> aTIC(mLTer->ImCorrel());
           Pt2di aP;

           double aDef = DefCorrelation().Val();
           for (aP.y =aBox._p0.y; aP.y<aBox._p1.y ; aP.y++)
           {
               for (aP.x =aBox._p0.x; aP.x<aBox._p1.x ; aP.x++)
               {
                  if (mLTer->IsInMasqOfPx(aP))
                      aTIC.oset(aP,aDef);
               }
          }
       }
       else if (mCurFenSpec)
       {
// std::cout << "Enter " << aBox._p0 << aBox._p1 << "\n";
             mPDVBoxInterneAct[0]->LoadedIm().CalcFenSpec
               (
                    aBox,
                    mPDVBoxInterneAct
               );
             mNbPointsByRect2 += aBox.surf();
// std::cout << "----END \n";
          // Correl Spec Exp
       }
       else if (aNbIm == 2)
       {
          mPDVBoxInterneAct[0]->LoadedIm().CalcCorrelRapide
               (
                    aBox,
                    mPDVBoxInterneAct[1]->LoadedIm()
               );
           mNbPointsByRect2 += aBox.surf();
       }
       else if (aNbIm <= theNbImageMaxAlgoRapide)
       {
           mPDVBoxInterneAct[0]->LoadedIm().NCalcCorrelRapide
                 (
                       aBox,
                       mPDVBoxInterneAct
                 );
           mNbPointsByRectN += aBox.surf();
       }
       else
       {
          ELISE_ASSERT(false,"CalcCorrelByRect");
       }
       TIm2D<REAL8,REAL8> aTIC(mLTer->ImCorrel());
       Pt2di aP;


       for (aP.y =aBox._p0.y; aP.y<aBox._p1.y ; aP.y++)
       {
           for (aP.x =aBox._p0.x; aP.x<aBox._p1.x ; aP.x++)
           {
               if (mLTer->IsInMasqOfPx(aP))
               {
                  double aCost = mStatGlob->CorrelToCout(aTIC.get(aP));
                  mSurfOpt->SetCout(aP,aPx,aCost);
               }
           }
       }
   }
   else
   {
      Pt2di aP;
      mStatGlob->SetSomsMade(true);
      for (aP.y =aBox._p0.y; aP.y<aBox._p1.y ; aP.y++)
      {
          for (aP.x =aBox._p0.x; aP.x<aBox._p1.x ; aP.x++)
          {
              // if (mLTer->OkPx(aP,aPx))
              if (mLTer->IsInMasqOfPx(aP))
              {
                 mNbPointByRectGen++;
                 mStatGlob->Clear();
                 for 
                 (
                     tCsteIterPDV itFI=mPDVBoxInterneAct.begin(); 
                     itFI!=mPDVBoxInterneAct.end(); 
                     itFI++
                 )
                 {
                        (*itFI)->LoadedIm().AddToStat(*mStatGlob,aP);
                 }
                 REAL aCost = mStatGlob->Cout();
                 mSurfOpt->SetCout(aP,aPx,aCost);
              }
          }
      }
      mStatGlob->SetSomsMade(false);
   }
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
